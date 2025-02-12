"""
Here, we use gpt-3.5-turbo with the batch upload api to translate the different questions to Dutch and Portuguese.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from openai.types import Batch

import multicultural_alignment.fileio as fileio
import multicultural_alignment.openai_batch as openai_batch
import multicultural_alignment.prompt_clean as pc
from multicultural_alignment.constants import OUTPUT_DIR

LANGUAGES = {
    "nl": "Dutch",
    "pt": "Portuguese",
}
TRANSLATION_PROMPT = "```translate\n{prompt}\n```"
SYSTEM_PROMPT = "You are a perfect translation machine. You take any user message and translate it perfectly, idiomatically and directly to {language}. You leave anything in curly brackets exactly as it is. Do not write anything but the translation."  # noqa: E501
assert OUTPUT_DIR.exists(), f"Output directory {OUTPUT_DIR} does not exist."


def create_requests_format(all_messages: list[list[dict]]) -> list[dict]:
    all_requests = []
    for i, messages in enumerate(all_messages):
        req = {
            "custom_id": f"translate-request-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-3.5-turbo-0125",
                "messages": messages,
            },
        }
        all_requests.append(req)
    return all_requests


def create_prompt_df(raw_prompts: pd.DataFrame) -> pd.DataFrame:
    english_prompts = raw_prompts[raw_prompts["language"] == "en"]
    unique_topics = (
        english_prompts[["topic", "question_key"]]
        .drop_duplicates()
        .assign(prompt_type="topic")
        .rename(columns={"topic": "prompt", "question_key": "key"})
    )
    unique_prompts = (
        english_prompts[["prompt", "template_type"]]
        .drop_duplicates()
        .assign(prompt_type="template")
        .rename(columns={"template_type": "key"})
    )
    all_prompts = pd.concat([unique_topics, unique_prompts], ignore_index=True).reset_index(drop=True)
    return all_prompts


def add_languages(all_prompts: pd.DataFrame, languages: dict[str, str] = LANGUAGES):
    new_all_prompts = pd.concat([all_prompts.assign(language=lang_code) for lang_code in languages], ignore_index=True)
    return new_all_prompts


def construct_messages(prompt_df: pd.DataFrame) -> list[list[dict]]:
    all_messages = [
        [
            {"role": "system", "content": SYSTEM_PROMPT.format(language=LANGUAGES[row.language])},
            {"role": "user", "content": TRANSLATION_PROMPT.format(prompt=row.prompt)},
        ]
        for row in prompt_df.itertuples()
    ]
    return all_messages


def constuct_requests(new_all_prompts: pd.DataFrame) -> list[dict]:
    all_messages = construct_messages(prompt_df=new_all_prompts)
    all_requests = create_requests_format(all_messages)
    return all_requests


def construction_pipeline() -> pd.DataFrame:
    raw_prompts = get_raw_prompts()
    all_prompts = create_prompt_df(raw_prompts)
    return add_languages(all_prompts)


def get_raw_prompts():
    raw_prompts = pd.read_csv(OUTPUT_DIR / "joined_df_newest.csv")
    return raw_prompts


def prompt_pipeline(output_path: Path) -> None:
    all_prompts = construction_pipeline()
    all_messages = constuct_requests(all_prompts)
    fileio.write_jsonl(all_messages, output_path)


def output_file_path(languages: dict[str, str] = LANGUAGES, output_dir: Path = OUTPUT_DIR) -> Path:
    lang_prefixes = "_".join(languages)
    return output_dir / f"translated_prompts_{lang_prefixes}.jsonl"


def main():
    output_path = output_file_path()
    prompt_pipeline(output_path=output_path)

    CLIENT = OpenAI()

    openai_batch.create_batch_job(
        client=CLIENT, output_path=output_path, description="Translate prompts to Dutch and Portuguese"
    )


def get_custom_id_number(dict_item: dict):
    assert "custom_id" in dict_item, f"custom_id not in dict_item: {dict_item}"
    return int(dict_item["custom_id"].split("-")[-1])


def clean_responses(responses: list[str]) -> pd.Series:
    return pc.replace_braces(pd.Series(responses)).apply(pc.extract_backtick_content)


def get_sorted_responses(client: OpenAI, batch: Batch) -> list[str]:
    output_dicts = openai_batch.download_batch(client, batch)
    sorted_output_dicts = sorted(output_dicts, key=get_custom_id_number)
    responses = openai_batch.extract_responses(sorted_output_dicts)
    return responses


if __name__ == "__main__":
    # main()
    all_prompts = construction_pipeline()
    CLIENT = OpenAI()
    batches = CLIENT.batches.list()

    batch = batches.data[0]

    responses = get_sorted_responses(CLIENT, batch)

    all_prompts["translated_prompt"] = clean_responses(responses)

    all_prompts.to_csv(OUTPUT_DIR / "translated_prompts.csv", index=False)

    raw_prompts = get_raw_prompts().filter()
    english_prompts = raw_prompts.loc[raw_prompts["language"] == "en"].drop(columns=["language", "topic", "prompt"])

    new_langs: list[dict] = []
    for lang_code, language_prompts in all_prompts.groupby("language"):
        for row in english_prompts.itertuples():
            prompt = (
                language_prompts.loc[
                    (language_prompts["prompt_type"] == "template") & (language_prompts["key"] == row.template_type),
                    "translated_prompt",
                ]
                .sample(n=1)
                .iloc[0]
            )
            topic = (
                language_prompts.loc[
                    (language_prompts["prompt_type"] == "topic") & (language_prompts["key"] == row.question_key),
                    "translated_prompt",
                ]
                .sample(n=1)
                .iloc[0]
            )
            new_langs.append(
                {
                    "question_key": row.question_key,
                    "question_name": row.question_name,
                    "language": lang_code,
                    "prompt": prompt,
                    "topic": topic,
                }
            )

    new_langs_df = pd.DataFrame(new_langs)
    with_translated_prompts = pd.concat([raw_prompts, new_langs_df], ignore_index=True).reset_index()
    with_translated_prompts.to_csv(OUTPUT_DIR / "translated_prompts_with_english.csv", index=False)

    assert np.all(sorted(new_langs_df.columns) == sorted(raw_prompts.columns)), (
        f"Columns do not match: {new_langs_df.columns} != {raw_prompts.columns}"
    )
