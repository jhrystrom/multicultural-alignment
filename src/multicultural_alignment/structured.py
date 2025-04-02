import asyncio
import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TypedDict

import instructor
import instructor.exceptions
from loguru import logger
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel

import multicultural_alignment.fileio as fileio
from multicultural_alignment.directories import CACHE_DIR


@dataclass
class ModelConfig:
    model: str
    client: OpenAI | AsyncOpenAI
    temperature: float = 0.7


class Message(TypedDict):
    role: str
    content: str


def get_default_config(is_async: bool = True, model_name: str = "gpt-4o-mini") -> ModelConfig:
    from dotenv import load_dotenv

    load_dotenv(override=True)
    return ModelConfig(
        model=model_name,
        # client=AsyncOpenAI() if is_async else OpenAI(),
        client=instructor.from_openai(client=AsyncOpenAI() if is_async else OpenAI()),
        temperature=0.0,
    )


def _create_messages(prompt: str, system_message: str | None) -> list[Message]:
    messages_to_send = [Message(role="system", content=system_message)] if system_message is not None else []
    messages_to_send.append(Message(role="user", content=prompt))
    return messages_to_send


def generate_cache_name(messages: list[Message], model_name: str, schema: type[BaseModel]) -> Path:
    """Hash a name for the above"""
    messages_str = str(messages)
    schema_str = str(schema.model_json_schema())
    return (CACHE_DIR / sha256(f"{messages_str}{model_name}{schema_str}".encode()).hexdigest()).with_suffix(".json")


async def agenerate_structured(
    prompt: str,
    schema: type[BaseModel],
    model_config: ModelConfig,
    system_message: str | None = None,
    use_cache: bool = True,
    cache_name: Path | None = None,
) -> BaseModel:
    client = model_config.client
    messages_to_send = _create_messages(prompt, system_message)

    if use_cache:
        cache_name = (
            generate_cache_name(messages=messages_to_send, model_name=model_config.model, schema=schema)
            if cache_name is None
            else cache_name
        )
        # Use async file I/O operations
        if await asyncio.to_thread(cache_name.exists):
            data = await asyncio.to_thread(fileio.read_json, cache_name)
            return schema.model_validate(data)

    try:
        response = await client.chat.completions.create(
            messages=messages_to_send,
            model=model_config.model,
            temperature=model_config.temperature,
            response_model=schema,
        )
    except instructor.exceptions.InstructorRetryException:
        logger.warning(f"Error on {messages_to_send}")
        response = schema(language="NULL", opinions=["NULL"])

    # response = await client.beta.chat.completions.parse(
    #     messages=messages_to_send,
    #     response_format=schema,
    #     model=model_config.model,
    #     temperature=model_config.temperature,
    # )

    # if has attribute choices
    if hasattr(response, "choices"):
        # Use the first choice
        json_response = json.loads(response.choices[0].message.content)
    else:
        json_response = response.model_dump()

    if use_cache:
        # Use async file I/O for writing too
        await asyncio.to_thread(fileio.write_json, json_response, cache_name)

    return schema.model_validate(json_response)


async def agenerate_structured_multi(
    prompts: list[str],
    schema: type[BaseModel],
    model_config: ModelConfig,
    system_message: str | None = None,
    concurrency_limit: int = 50,
) -> list[BaseModel]:
    if model_config.client is None:
        raise ValueError("ModelConfig must have a client")
    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrency_limit)

    all_cache_names = [
        generate_cache_name(messages=_create_messages(prompt, system_message), model_name=model_config.model, schema=schema)
        for prompt in prompts
    ]

    async def limited_generate(prompt: str, cache_name: Path) -> BaseModel:
        # Use the semaphore to limit concurrent executions
        async with semaphore:
            return await agenerate_structured(
                prompt=prompt,
                schema=schema,
                system_message=system_message,
                model_config=model_config,
                cache_name=cache_name,
            )

    all_results = await asyncio.gather(
        *[limited_generate(prompt=prompt, cache_name=cache_name) for prompt, cache_name in zip(prompts, all_cache_names)]
    )
    return all_results
