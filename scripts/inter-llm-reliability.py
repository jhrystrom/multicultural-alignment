import asyncio
import json

import polars as pl
from loguru import logger
from tqdm import tqdm

import multicultural_alignment.structured as structured
from multicultural_alignment.directories import OUTPUT_DIR
from multicultural_alignment.schemas import ProcessOpinions


async def main():
    models = ["gpt-4o-mini", "gpt-3.5-turbo", "gpt-4o"]

    llm_responses = (
        pl.read_csv(OUTPUT_DIR / "all_prompts_responses_gemma-2-27b-it.csv")
        .filter(pl.col("template_type") == "survey_hypothetical")
        .select("topic", "response")
        .sample(n=200, seed=110)
        .to_dicts()
    )

    logger.info(f"Loaded {len(llm_responses)} responses from LLM.")
    system_prompt = "Analyze the opinions in the following response."
    all_prompts = [json.dumps(row, ensure_ascii=False) for row in llm_responses]

    logger.info("Generating responses...")

    response_dict = dict.fromkeys(models, [])
    for model in tqdm(models):
        config = structured.get_default_config(model_name=model)
        responses = await structured.agenerate_structured_multi(
            prompts=all_prompts,
            schema=ProcessOpinions,
            model_config=config,
            system_message=system_prompt,
        )
        response_dict[model] = responses
    return response_dict, all_prompts


if __name__ == "__main__":
    response_dict, all_prompts = asyncio.run(main())

    json_prompts = [json.loads(prompt) for prompt in all_prompts]
    (OUTPUT_DIR / "example-json-prompts.jsonl").write_text(
        "\n".join(json.dumps(prompt, ensure_ascii=False) for prompt in json_prompts)
    )

    # create DF
    combined = (
        pl.concat(
            [
                pl.DataFrame(responses).with_columns(pl.lit(model).alias("model")).with_row_index()
                for model, responses in response_dict.items()
            ]
        )
        .explode("opinions")
        .with_columns((pl.col("index").cast(pl.String) + "-" + pl.col("model")).alias("custom_id"))
        .with_columns(pl.int_range(pl.len()).over("custom_id").alias("opinion_id"))
        .with_columns((pl.col("index").cast(str) + "-" + pl.col("opinion_id").cast(str)).alias("statement_id"))
        .drop("custom_id")
    )

    from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

    pivoted = (
        combined.filter(pl.col("opinions") != "NULL")
        .with_columns(pl.col("opinions").rank("dense").cast(pl.Int8))
        .pivot(index="statement_id", columns="model", values="opinions")
        .drop("statement_id")
        .drop_nulls()
    )

    from reliabilipy import reliability_analysis

    combined["opinions"].rank("dense").cast()

    results = reliability_analysis(raw_dataset=pivoted.to_pandas(), is_corr_matrix=False, impute="drop")
    results.fit()
    results.omega_total

    table, labels = aggregate_raters(pivoted)
    fleiss_kappa(table)
