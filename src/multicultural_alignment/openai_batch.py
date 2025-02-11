import json
from pathlib import Path

from instructor import OpenAISchema
from openai import OpenAI
from openai.types import Batch
from tqdm import tqdm

import multicultural_alignment.fileio as fileio


def batch_from_messages(
    client: OpenAI, all_requests: list[list[dict]], output_path: Path, metadata: dict | None = None
) -> Batch:
    fileio.write_jsonl(all_requests, output_path)
    assert output_path.exists(), f"Output path {output_path} does not exist"
    batch = create_batch_job(client, output_path=output_path, metadata=metadata)
    return batch


def create_batch_job(client: OpenAI, output_path: Path, metadata: dict = {"description": "Batch Job"}) -> Batch:
    batch_input_file = client.files.create(file=output_path.open("rb"), purpose="batch")
    batch_input_file_id = batch_input_file.id
    return client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata=metadata,
    )


def download_batch(client: OpenAI, batch: Batch | None = None, batch_id: str | None = None) -> list[dict]:
    if batch is None and batch_id is None:
        raise ValueError("Either batch or batch_id must be provided")
    output_file_id = batch.output_file_id if batch is not None else batch_id
    output = client.files.content(file_id=output_file_id).read().decode()
    output_dicts = [json.loads(line) for line in output.split("\n") if line]
    return output_dicts


def extract_responses(batch_output: list[dict]) -> list[str]:
    responses = [item["response"]["body"]["choices"][0]["message"]["content"] for item in batch_output]
    return responses


def format_tool_choice(tool: type[OpenAISchema]) -> dict:
    return {"type": "function", "function": {"name": tool.openai_schema["name"]}}


def format_tool(tool: type[OpenAISchema]) -> dict:
    return {"type": "function", "function": tool.openai_schema}


def create_requests_format(
    all_messages: list[list[dict]],
    model: str = "gpt-3.5-turbo",
    id_prefix: str = "translate-request",
    tool: type[OpenAISchema] | None = None,
) -> list[dict]:
    all_requests = []
    for i, messages in tqdm(enumerate(all_messages)):
        req = {
            "custom_id": f"{id_prefix}-{i:06}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": messages,
            },
        }
        if tool is not None:
            req["body"]["tools"] = [format_tool(tool)]
            req["body"]["tool_choice"] = format_tool_choice(tool)
        all_requests.append(req)
    return all_requests
