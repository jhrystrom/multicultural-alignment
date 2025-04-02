import gc
import os
import random
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import polars as pl
import torch
from huggingface_hub import login
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, RequestOutput, SamplingParams

import multicultural_alignment.fileio as fileio
from multicultural_alignment.constants import COUNTRY_LANG_MAP, COUNTRY_MAP
from multicultural_alignment.directories import OUTPUT_DIR

MODELS = [
    # "google/gemma-2-2b-it",
    # "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
    # "allenai/OLMo-2-1124-7B-Instruct",
    # "allenai/OLMo-2-1124-13B-Instruct",
    # "allenai/OLMo-2-0325-32B-Instruct",
]

SAMPLING_PARAMS = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=600, stop="11.")


class Message(TypedDict):
    role: str
    content: str


@contextmanager
def vllm_attention_backend(model_name: str):
    """
    Context manager to set VLLM_ATTENTION_BACKEND for Gemma models.
    """
    original_backend = os.environ.get("VLLM_ATTENTION_BACKEND")
    if model_name.lower().startswith("google/gemma"):
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    try:
        yield
    finally:
        if original_backend is None:
            os.environ.pop("VLLM_ATTENTION_BACKEND", None)
        else:
            os.environ["VLLM_ATTENTION_BACKEND"] = original_backend


def create_language_to_countries_map(country_lang_map, country_map):
    """
    Create a dictionary with language codes as keys and lists of full country names as values.

    Args:
        country_lang_map: Dictionary mapping country codes to language codes
        country_map: Dictionary mapping country codes to full country names

    Returns:
        Dictionary with language codes as keys and lists of country names as values
    """
    language_to_countries = {}

    # Iterate through the country-language mappings
    for country_code, lang_code in country_lang_map.items():
        # Get the full country name from country_map
        country_name = country_map.get(country_code)

        # If the country name exists
        if country_name:
            # If the language code is already a key, append the country name
            if lang_code in language_to_countries:
                language_to_countries[lang_code].append(country_name)
            # Otherwise, create a new list with the country name
            else:
                language_to_countries[lang_code] = [country_name]

    return language_to_countries


LANG_DICT = create_language_to_countries_map(country_lang_map=COUNTRY_LANG_MAP, country_map=COUNTRY_MAP)
all_countries = {country for countries in LANG_DICT.values() for country in countries}


DEMOGRAPHIC_DICT = {
    "en": "The respondents are from COUNTRY",
    "da": "Respondenterne er fra COUNTRY",
    "pt": "Os entrevistados são de COUNTRY",
    "nl": "De respondenten komen uit COUNTRY",
}


DATA_PATH = OUTPUT_DIR / "all-language-requests.jsonl"
PROMPT_PATH = OUTPUT_DIR / "translated_prompts_with_english_no_controversy.csv"


prompt_df = pl.read_csv(PROMPT_PATH)
prompts = fileio.read_jsonl(DATA_PATH)[: prompt_df.height]


new_prompts = []
new_df = prompt_df.filter(pl.col("template_type") == "survey_hypothetical")
for row, prompt in tqdm(zip(prompt_df.iter_rows(named=True), prompts, strict=True)):
    if row["template_type"] != "survey_hypothetical":
        continue
    prompt_text = prompt["body"]["messages"][0]["content"]
    new_text = (
        prompt_text[: prompt_text.find(".") + 1]
        + " "
        + DEMOGRAPHIC_DICT[row["language"]]
        + prompt_text[prompt_text.find(".") :]
    )
    new_prompts.append([{"role": "user", "content": new_text}])

prompt_variation_congruent = []
prompt_variation_incongruent = []
for row, prompt in zip(new_df.iter_rows(named=True), new_prompts):
    # Congruent country
    country = random.choice(LANG_DICT[row["language"]])
    content = prompt[0]["content"]
    content = content.replace("COUNTRY", country)
    prompt_variation_congruent.append(([{"role": "user", "content": content}], country))
    # incongruent country
    incongruent_country = random.choice(list(all_countries - set(LANG_DICT[row["language"]])))
    content = prompt[0]["content"]
    content = content.replace("COUNTRY", incongruent_country)
    prompt_variation_incongruent.append(([{"role": "user", "content": content}], incongruent_country))

variation_df = pl.concat([new_df, new_df]).with_columns(
    pl.Series(
        "country",
        [country for _, country in prompt_variation_congruent] + [country for _, country in prompt_variation_incongruent],
    ),
    pl.Series("is_congruent", [True] * len(prompt_variation_congruent) + [False] * len(prompt_variation_incongruent)),
)


def tokenize_messages(tokenizer: AutoTokenizer, messages: list[list[Message]]) -> list[list[int]]:
    return [
        tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True)
        for message in tqdm(messages, desc="Tokenizing messages")
    ]


def get_text_output(output: RequestOutput) -> str:
    return output.outputs[0].text


def parse_outputs(outputs: list[RequestOutput]) -> list[str]:
    return [get_text_output(output) for output in outputs]


def generate_responses(tokenized: list[list[int]], llm: LLM, sampling_params: SamplingParams):
    raw_outputs = llm.generate(prompt_token_ids=tokenized, sampling_params=sampling_params)
    return parse_outputs(raw_outputs)


def cleanup_llm(llm: LLM) -> None:
    del llm
    gc.collect()
    torch.cuda.empty_cache()


def model_generation_pipeline(model_name: str, all_prompts: list[dict], prompt_df: pl.DataFrame) -> Path:
    output_path = create_model_output_path(model_name)
    if output_path.exists():
        logger.info("Already ran model!")
        return output_path
    logger.info(f"Loading {model_name}...")

    with vllm_attention_backend(model_name):
        logger.info(f"Loading {model_name}...")
        logger.debug(f"Backend: {os.environ.get('VLLM_ATTENTION_BACKEND')}")
        llm = LLM(model=model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        all_tokenized_prompts = tokenize_messages(tokenizer=tokenizer, messages=all_prompts)
        assert len(all_tokenized_prompts) == prompt_df.shape[0], "Not matching!"
        outputs = generate_responses(all_tokenized_prompts, llm=llm, sampling_params=SAMPLING_PARAMS)

    logger.info("Done with the tough stuff! Saving responses...")
    # final_df = prompt_df.copy().assign(model=model_name, response=outputs)
    final_df = prompt_df.with_columns(
        pl.Series("response", outputs),
        pl.lit(model_name).alias("model"),
    )
    final_df.write_csv(output_path)
    logger.info("Clean up LLM")
    cleanup_llm(llm=llm)
    return output_path


def create_model_output_path(model_name: str) -> Path:
    output_path = OUTPUT_DIR / f"demographic-all_prompts_responses_{extract_model_name(model_name)}.csv"
    return output_path


def extract_model_name(model: str) -> str:
    return model[model.find("/") + 1 :]


def get_current_time() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def main():
    login()
    prompt_df = variation_df
    all_prompts: list[list[Message]] = [prompt for prompt, _ in prompt_variation_congruent + prompt_variation_incongruent]
    model_families: dict[str, list[pl.DataFrame]] = {family: [] for family in ["gemma", "olmo"]}
    for model in tqdm(MODELS, desc="Getting responses for models."):
        output_path = model_generation_pipeline(model_name=model, all_prompts=all_prompts, prompt_df=prompt_df)
        model_family = "gemma" if "gemma" in model else "olmo"
        model_families[model_family].append(pl.read_csv(output_path))

    for family, response_dfs in model_families.items():
        if not response_dfs:
            continue
        pl.concat(response_dfs).write_csv(OUTPUT_DIR / f"demographic-all_prompts_responses_{family}-combined.csv")


if __name__ == "__main__":
    main()
