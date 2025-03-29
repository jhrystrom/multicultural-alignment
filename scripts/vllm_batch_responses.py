import gc
import json
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import login
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, RequestOutput, SamplingParams

DATA_DIR = Path("output")
MODELS = [
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
    "allenai/OLMo-2-1124-7B-Instruct",
    "allenai/OLMo-2-1124-13B-Instruct",
    "allenai/OLMo-2-0325-32B-Instruct",
]
DATA_PATH = DATA_DIR / "all-language-requests.jsonl"
PROMPT_PATH = DATA_DIR / "translated_prompts_with_english_no_controversy.csv"
assert DATA_PATH.exists()
SAMPLING_PARAMS = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=600, stop="11.")


@contextmanager
def vllm_attention_backend(model_name: str):
    """
    Context manager to set VLLM_ATTENTION_BACKEND for Gemma models.
    """
    original_backend = os.environ.get("VLLM_ATTENTION_BACKEND")
    if model_name.lower().startswith("google/gemma"):
        os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
    try:
        yield
    finally:
        if original_backend is None:
            os.environ.pop("VLLM_ATTENTION_BACKEND", None)
        else:
            os.environ["VLLM_ATTENTION_BACKEND"] = original_backend


def read_jsonl(file_path: Path) -> list[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f.readlines()]


def get_messages(req: dict) -> list[dict]:
    return req["body"]["messages"]


def tokenize_all_messages(tokenizer: AutoTokenizer, all_requests: list[dict]) -> list[list[int]]:
    return [
        tokenizer.apply_chat_template(get_messages(req), tokenize=True, add_generation_prompt=True)
        for req in tqdm(all_requests, desc="Tokenizing messages")
        if req["custom_id"].startswith("gpt-3.5")
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


def model_generation_pipeline(model_name: str, all_prompts: list[dict], prompt_df: pd.DataFrame) -> Path:
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
        all_tokenized_prompts = tokenize_all_messages(tokenizer=tokenizer, all_requests=all_prompts)
        assert len(all_tokenized_prompts) == prompt_df.shape[0], "Not matching!"
        outputs = generate_responses(all_tokenized_prompts, llm=llm, sampling_params=SAMPLING_PARAMS)

    logger.info("Done with the tough stuff! Saving responses...")
    final_df = prompt_df.copy().assign(model=model_name, response=outputs)
    final_df.to_csv(output_path, index=False)
    logger.info("Clean up LLM")
    cleanup_llm(llm=llm)
    return output_path


def create_model_output_path(model_name: str) -> Path:
    output_path = DATA_DIR / f"all_prompts_responses_{extract_model_name(model_name)}.csv"
    return output_path


def extract_model_name(model: str) -> str:
    return model[model.find("/") + 1 :]


def get_current_time() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def main():
    login()
    prompt_df = pd.read_csv(PROMPT_PATH)
    all_prompts = read_jsonl(DATA_PATH)
    model_families: dict[str, list[pd.DataFrame]] = {family: [] for family in ["gemma", "olmo"]}
    for model in tqdm(MODELS, desc="Getting responses for models."):
        output_path = model_generation_pipeline(model_name=model, all_prompts=all_prompts, prompt_df=prompt_df)
        model_family = "gemma" if "gemma" in model else "olmo"
        model_families[model_family].append(pd.read_csv(output_path))

    for family, response_dfs in model_families.items():
        pd.concat(response_dfs).to_csv(DATA_DIR / f"all_prompts_responses_{family}-combined.csv")


if __name__ == "__main__":
    main()
