import asyncio
import json

import polars as pl
from loguru import logger

import multicultural_alignment.fileio as fileio
import multicultural_alignment.structured as structured
from multicultural_alignment.directories import OUTPUT_DIR
from multicultural_alignment.schemas import AnalysisResponse, ProcessOpinions


def opinions_to_responses(responses: list[ProcessOpinions]) -> list[AnalysisResponse]:
    return [
        AnalysisResponse(custom_id=f"response-{idx}", language=response.language, opinions=response.opinions)
        for idx, response in enumerate(responses)
    ]


async def main():
    config = structured.get_default_config(model_name="gpt-4o-mini")
    logger.info(f"Using model: {config.model}")
    olmo_responses = (
        pl.read_csv(OUTPUT_DIR / "all_prompts_responses_olmo-combined.csv")
        .filter(pl.col("template_type") == "survey_hypothetical")
        .select("topic", "response")
        .head(975)
        .to_dicts()
    )

    logger.info(f"Loaded {len(olmo_responses)} responses from OLMO.")
    system_prompt = "Analyze the opinions in the following response."
    all_prompts = [json.dumps(row, ensure_ascii=False) for row in olmo_responses]

    logger.info("Generating responses...")
    responses = await structured.agenerate_structured_multi(
        prompts=all_prompts,
        schema=ProcessOpinions,
        model_config=config,
        system_message=system_prompt,
    )

    logger.info("Done! Writing to file...")
    cleaned_responses = opinions_to_responses(responses)
    fileio.write_jsonl(
        cleaned_responses,
        OUTPUT_DIR / "olmo-analysis-responses.jsonl",
    )
    logger.info("All done!")


if __name__ == "__main__":
    asyncio.run(main())

    config = structured.get_default_config(is_async=False)
    problem_message = [
        {
            "role": "system",
            "content": "Analyze the opinions in the following response. Return immediately if input doesn't follow format.",
        },
    ]
    response = config.client.beta.chat.completions.parse(
        messages=problem_message, model=config.model, temperature=config.temperature, response_format=ProcessOpinions
    )
