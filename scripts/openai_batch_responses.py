import time
from datetime import datetime

import pandas as pd
from openai import OpenAI
from openai.types import Batch

import multicultural_alignment.fileio as fileio
import multicultural_alignment.openai_batch as openai_batch
from multicultural_alignment.constants import OUTPUT_DIR
from multicultural_alignment.models import MODEL_FAMILIES

JSONL_PATTERN = "{model}-controversy-requests.jsonl"
CLIENT = OpenAI()
OPENAI_MODELS = MODEL_FAMILIES["openai"]


def validate(batches: list[Batch]) -> None:
    """Runs until validated."""
    both_validated = False
    i = 0
    while not both_validated:
        print("Checking if all batches are validated...")
        num_valid = 0
        for batch in batches:
            status = CLIENT.batches.retrieve(batches[0].id).status
            has_validated = status == "in_progress"
            print(f"Batch {batch.id} status: {status}")
            num_valid += int(has_validated)
        if num_valid == len(batches):
            both_validated = True
            break
        i += 1
        sleep_secs = next_fibonacci(i)
        print(f"Sleeping for {sleep_secs} seconds...")
        time.sleep(sleep_secs)


def create_requests_list(formatted_df: pd.DataFrame) -> list[list[dict]]:
    all_requests = []
    for model_name, model_df in formatted_df.groupby("model"):
        all_messages = [
            [
                {"role": "user", "content": row.prompt},
            ]
            for row in model_df.itertuples()
        ]
        all_requests.append(
            openai_batch.create_requests_format(all_messages, model=model_name, id_prefix=f"{model_name}-answers")
        )
    return all_requests


def add_models(df: pd.DataFrame, models: list[str] = OPENAI_MODELS) -> pd.DataFrame:
    new_df = pd.concat([df.assign(model=model) for model in models], ignore_index=True)
    return new_df


def create_formatted_df(all_prompts: pd.DataFrame) -> pd.DataFrame:
    formatted_prompts = []
    for row in all_prompts.itertuples():
        formatted_prompts.append(
            {
                "language": row.language,
                "question_key": row.question_key,
                "prompt": row.prompt.format(topic=row.topic),
            }
        )
    formatted_df = add_models(pd.DataFrame(formatted_prompts))
    return formatted_df


def next_fibonacci(n: int) -> int:
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    fib = [0, 1]
    while len(fib) <= n + 1:
        fib.append(fib[-1] + fib[-2])
    return fib[-1]


def create_batches(all_requests: list[list[dict]]) -> list[Batch]:
    batches = []
    for MODEL_NAME, reqs in zip(OPENAI_MODELS, all_requests, strict=True):
        batch = create_batch(reqs, model_name=MODEL_NAME)
        batches.append(batch)
    return batches


def create_batch(reqs, model_name: str):
    output_path = OUTPUT_DIR / JSONL_PATTERN.format(model=model_name)
    fileio.write_jsonl(reqs, output_path)
    assert output_path.exists(), f"Output path {output_path} does not exist"
    batch = openai_batch.create_batch_job(
        client=CLIENT, output_path=output_path, description="Get all responses from OpenAI models"
    )
    return batch


if __name__ == "__main__":
    all_prompts = pd.read_csv(OUTPUT_DIR / "translated_prompts_with_english.csv")
    formatted_df = create_formatted_df(all_prompts)
    all_requests = create_requests_list(formatted_df)
    assert len(all_requests) == len(OPENAI_MODELS), "Number of models does not match number of requests"

    done_batches = [batch for batch in CLIENT.batches.list().data if batch.status == "completed"]
    current_batches = [
        batch for batch in done_batches if batch.metadata["description"] == "Get all responses from OpenAI models"
    ]

    parsed_batches = [openai_batch.download_batch(CLIENT, batch) for batch in current_batches]

    new_dfs = []
    for model in OPENAI_MODELS:
        for batch in parsed_batches:
            if model not in batch[0]["custom_id"]:
                continue
            extracted_responses = openai_batch.extract_responses(batch)
            result_df = all_prompts.copy()
            result_df = result_df.assign(response=extracted_responses, model=model)
            new_dfs.append(result_df)

    new_results = pd.concat(new_dfs, ignore_index=True).reset_index(drop=True)
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    new_results.drop(columns=["index"]).to_csv(
        OUTPUT_DIR / f"all_prompts_responses_openai_models-{current_time}.csv", index=False
    )
