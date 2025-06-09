import json
from pathlib import Path

import pandas as pd
import polars as pl

import multicultural_alignment.score as score
from multicultural_alignment import fileio
from multicultural_alignment.constants import OUTPUT_DIR
from multicultural_alignment.models import MODEL_FAMILIES
from multicultural_alignment.schemas import AnalysisResponse


def sort_by_custom_id(dict_list):
    return sorted(dict_list, key=lambda d: int(d["custom_id"].split("-")[-1]))


def parse_response(response: dict) -> AnalysisResponse:
    try:
        choice = response["response"]["body"]["choices"][0]
        arguments = choice["message"]["tool_calls"][0]["function"]["arguments"]
    except KeyError:
        return AnalysisResponse(**response)
    try:
        return AnalysisResponse(**json.loads(arguments), custom_id=response["custom_id"])
    except json.decoder.JSONDecodeError:
        return AnalysisResponse(language=None, opinions=None, custom_id=response["custom_id"])


def sort_and_parse(responses: list[dict]) -> list[AnalysisResponse]:
    return [parse_response(response) for response in sort_by_custom_id(responses)]


def clean_list(lst: list[str]) -> list[str]:
    if lst is None:
        return []
    return [item.lower() for item in lst]


def score_response(response: AnalysisResponse) -> float:
    pro_count = clean_list(response["opinions"]).count("pro")
    con_count = clean_list(response["opinions"]).count("con")
    return score.pro_score(num_pro=pro_count, num_con=con_count)


def create_score_df(raw_responses: list[dict]) -> pd.DataFrame:
    parsed_responses = sort_and_parse(raw_responses)
    scores = [score_response(response) for response in parsed_responses]
    return pd.DataFrame(scores, columns=["response_pro_score"]).assign(
        response_language=[response.get("language", "NULL") for response in parsed_responses],
        response_pro_con=[";".join(clean_list(response.get("opinions", []))) for response in parsed_responses],
        custom_id=[response["custom_id"] for response in parsed_responses],
    )


def create_full_truth_df(prompt_responses: pl.DataFrame, analysis_responses: list[dict]) -> pl.DataFrame:
    score_df = create_score_df(analysis_responses)
    assert prompt_responses.shape[0] == score_df.shape[0], "Mismatch between response and score count"
    return prompt_responses.with_columns(pl.DataFrame(score_df))


def _load_openai_mistral() -> pd.DataFrame:
    openai_responses = load_model_family("openai")
    mistral_responses = load_model_family("mistral")
    return pd.concat([openai_responses, mistral_responses], ignore_index=True).reset_index(drop=True)


def load_model_family(model_family: str, output_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    model_pattern = "all_prompts_responses_{model_name}.csv"
    family_pattern = "all_prompts_responses_{model_family}*.csv"
    if model_family not in MODEL_FAMILIES:
        raise ValueError(f"Invalid model family: {model_family}")
    # Special case: openai is stored in a single file...
    if model_family == "gemma":
        return pd.concat(
            [
                pd.read_csv(output_dir / model_pattern.format(model_name=model_name))
                for model_name in MODEL_FAMILIES[model_family]
            ],
            ignore_index=True,
        )
    return pd.read_csv(fileio.find_latest_path(family_pattern.format(model_family=model_family), directory=output_dir))


def get_data_pipeline():
    gemma_demographic = pl.read_csv(OUTPUT_DIR / "demographic-all_prompts_responses.csv").with_columns(
        pl.lit("gemma-2-27b-it").alias("model"),
        pl.lit("gemma").alias("model_family"),
    )
    gemma_responses = fileio.read_jsonl(OUTPUT_DIR / "demographic-analysis-responses.jsonl")

    gemma_truth = create_full_truth_df(prompt_responses=gemma_demographic, analysis_responses=gemma_responses)

    ground_truth = pl.read_csv(OUTPUT_DIR / "ground_truth.csv").rename({"pro_score": "ground_truth_pro_score"})

    all_results = (
        gemma_truth.drop("controversy")
        .join(ground_truth, on=["question_key", "language"], how="left")
        .drop_nulls(subset=["response_pro_score", "ground_truth_pro_score", "model", "language"])
        .drop("custom_id", "template_type", "cateogry", "prompt", "topic", "question_name")
        .filter(pl.col("language") == pl.col("response_language"))
    )
    all_results

    all_results.write_csv(OUTPUT_DIR / "demographic-gemma-results.csv")


if __name__ == "__main__":
    get_data_pipeline()
