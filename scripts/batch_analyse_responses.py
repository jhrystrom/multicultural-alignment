import argparse
import json

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from openai.types import Batch

import multicultural_alignment.fileio as fileio
import multicultural_alignment.openai_batch as openai_batch
from multicultural_alignment.constants import OUTPUT_DIR
from multicultural_alignment.models import MODEL_FAMILIES
from multicultural_alignment.schemas import ProcessOpinions

assert OUTPUT_DIR.exists(), f"Data directory {OUTPUT_DIR} does not exist"

load_dotenv()

MODEL_FAMILY_NAMES = list(MODEL_FAMILIES)
MODEL_FAMILY_NAMES.remove("Baseline")


def create_messages(response_row: dict) -> list[dict]:
    return [
        {"role": "system", "content": "Analyze the opinions in the following response."},
        {"role": "user", "content": json.dumps(response_row, ensure_ascii=False)},
    ]


def create_batch(families: list[str]):
    combined_data = get_family_data(families=families)
    client = OpenAI()
    family_str = "_".join(families)
    run_batch(
        client,
        combined_data,
        output_name=f"{family_str}_responses_batch.jsonl",
        additional_metadata={"model_families": family_str},
    )


def get_family_data(families: list[str]) -> pd.DataFrame:
    all_data = []
    for family in families:
        models = MODEL_FAMILIES[family]
        data_paths = [OUTPUT_DIR / f"all_prompts_responses_{model}.csv" for model in models]
        family_data = pd.concat(pd.read_csv(data_path) for data_path in data_paths)
        all_data.append(family_data)
    combined_data = pd.concat(all_data, ignore_index=True).reset_index(drop=True)
    return combined_data


def run_batch(
    client,
    combined_data: pd.DataFrame,
    output_name: str = "all_responses_batch.jsonl",
    additional_metadata: dict | None = None,
):
    responses = combined_data[["response", "topic"]].to_dict(orient="records")
    all_messages = [create_messages(response) for response in responses]
    all_requests = openai_batch.create_requests_format(all_messages, tool=ProcessOpinions, id_prefix="analyze-opinions")

    metadata = {"description": "Analyze opinions in responses", "task": "analyze"}
    if additional_metadata is not None:
        metadata.update(additional_metadata)
    batch = openai_batch.batch_from_messages(
        client,
        all_requests,
        output_path=OUTPUT_DIR / output_name,
        metadata=metadata,
    )
    return batch


def download_batch(model_families: list[str]):
    model_families_str = "_".join(model_families)
    client = OpenAI()
    batches = client.batches.list()
    analysis_batch = next(
        batch
        for batch in batches.data
        if (batch.metadata.get("task") == "analyze") and (batch.metadata.get("model_families") == model_families_str)
    )
    if not analysis_batch.status == "completed":
        raise ValueError("Batch is not completed")
    downloaded = openai_batch.download_batch(client, batch=analysis_batch)
    fileio.write_jsonl(downloaded, OUTPUT_DIR / f"{model_families_str}-raw-responses.jsonl")


def run_gpt4o_batch():
    gpt4o_data = pd.read_csv(OUTPUT_DIR / "gpt-4o_responses.csv")
    client = OpenAI()
    batch = run_batch(
        client, combined_data=gpt4o_data, output_name="gpt-4o_responses_batch.jsonl", additional_metadata={"model": "gpt-4o"}
    )
    print(batch)


def retrieve_gpt4_batch() -> Batch:
    client = OpenAI()
    batches = client.batches.list()
    gpt4_batch = next(
        batch
        for batch in batches.data
        if batch.metadata.get("model") == "gpt-4o" and batch.metadata.get("task") == "analyze"
    )
    if not gpt4_batch.status == "completed":
        raise ValueError("Batch is not completed")
    return gpt4_batch


def download_gpt4o_analysis() -> None:
    client = OpenAI()
    batch = retrieve_gpt4_batch()
    responses = openai_batch.download_batch(client, batch=batch)
    fileio.write_jsonl(responses, OUTPUT_DIR / "gpt-4o_responses_analysis.jsonl")


def main(args: argparse.Namespace):
    if args.mode == "run":
        create_batch(families=args.model_families)
    elif args.mode == "download":
        download_batch(model_families=args.model_families)
    else:
        raise ValueError("Invalid mode")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process responses for analysis")
    parser.add_argument(
        "--model-families",
        "-m",
        type=str,
        nargs="+",
        help="Models families for which to run batch processing",
        choices=MODEL_FAMILY_NAMES,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="run",
        choices=["run", "download"],
        help="Whether to run the batch or download the results",
    )
    args = parser.parse_args()
    main(args=args)

    client = OpenAI()
    download_batch(model_families=["gemma"])
