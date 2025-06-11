import argparse
import json

import polars as pl
from dotenv import load_dotenv
from openai import OpenAI

import multicultural_alignment.data
import multicultural_alignment.fileio as fileio
import multicultural_alignment.openai_batch as openai_batch
from multicultural_alignment.constants import OUTPUT_DIR
from multicultural_alignment.models import MODEL_FAMILIES
from multicultural_alignment.schemas import ProcessOpinions
from multicultural_alignment.structured import OPINION_SYSTEM_MSG

assert OUTPUT_DIR.exists(), f"Data directory {OUTPUT_DIR} does not exist"

load_dotenv(override=True)

MODEL_FAMILY_NAMES = list(MODEL_FAMILIES)
MODEL_FAMILY_NAMES.remove("Baseline")


def create_messages(response_row: dict, system_msg: str = OPINION_SYSTEM_MSG) -> list[dict]:
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": json.dumps(response_row, ensure_ascii=False)},
    ]


def create_batch(families: list[str]):
    combined_data = multicultural_alignment.data.get_family_data(families=families)
    client = OpenAI()
    family_str = multicultural_alignment.data.get_family_string(families)
    run_batch(
        client,
        combined_data,
        output_name=f"{family_str}_responses_batch.jsonl",
        additional_metadata={"model_families": family_str},
    )


def run_batch(
    client,
    combined_data: pl.DataFrame,
    output_name: str = "all_responses_batch.jsonl",
    additional_metadata: dict | None = None,
):
    stances = multicultural_alignment.data.get_stance_labels()
    # responses = combined_data[["response", "topic"]].to_dict(orient="records")
    responses = (
        combined_data.select(["question_key", "response"]).join(stances, on="question_key").drop("question_key").to_dicts()
    )

    all_messages = [create_messages(response) for response in responses]
    all_requests = openai_batch.create_requests_format(
        all_messages, tool=ProcessOpinions, id_prefix="analyze-opinions", model="gpt-4.1"
    )

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
    model_families_str = multicultural_alignment.data.get_family_string(model_families)
    client = OpenAI()
    batches = client.batches.list()
    analysis_batch = next(
        batch
        for batch in batches.data
        if (batch.metadata.get("task") == "analyze") and (batch.metadata.get("model_families") == model_families_str)
    )
    if not analysis_batch.status == "completed":
        raise ValueError(
            f"Batch is not completed: Status is {analysis_batch.status}. Current progress is: {analysis_batch.request_counts}"  # noqa: E501
        )
    downloaded = openai_batch.download_batch(client, batch=analysis_batch)
    fileio.write_jsonl(downloaded, OUTPUT_DIR / f"{model_families_str}-raw-responses.jsonl")


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
