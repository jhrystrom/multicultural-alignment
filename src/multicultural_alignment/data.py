from typing import Generator, Literal

import pandas as pd
import polars as pl

from multicultural_alignment.constants import OUTPUT_DIR
from multicultural_alignment.models import MODEL_FAMILIES

GTType = Literal["global", "language", "country"]


def _as_polars(df: pd.DataFrame) -> pl.DataFrame:
    return pl.DataFrame(df)


def load_results(as_polars: bool = False) -> pd.DataFrame | pl.DataFrame:
    results = pd.read_csv(
        OUTPUT_DIR / "all_lang_model_responses_with_scores.csv",
        usecols=[
            "language",
            "response_pro_score",
            "response_language",
            "model_name",
            "question_key",
            "model_family",
            "template_type",
        ],
    )
    return results if not as_polars else _as_polars(results)


def load_ground_truth(as_polars: bool = False) -> pd.DataFrame | pl.DataFrame:
    gt = pd.read_csv(OUTPUT_DIR / "ground_truth_all_countries.csv").rename(columns={"pro_score": "ground_truth_pro_score"})
    return gt if not as_polars else _as_polars(gt)


def load_raw_responses(as_polars: bool = False) -> pd.DataFrame | pl.DataFrame:
    raw_responses = pl.read_csv(OUTPUT_DIR / "all_lang_model_responses_with_scores.csv")
    return raw_responses if as_polars else raw_responses.to_pandas()


def yield_family_responses(family: str) -> Generator[tuple[str, pl.DataFrame], None, None]:
    models = [model for model in MODEL_FAMILIES[family]]
    for model in models:
        model_responses = pl.read_csv(OUTPUT_DIR / f"all_prompts_responses_{model}.csv")
        yield model, model_responses


def get_all_responses(family: str) -> pl.DataFrame:
    return pl.concat([responses for _, responses in yield_family_responses(family)])


def join_all_gt(results: pd.DataFrame | pl.DataFrame, gt_type: GTType = "language") -> pl.DataFrame:
    file_names = {
        "global": "ground_truth_global.csv",
        "language": "ground_truth_per_language.csv",
        "country": "ground_truth_every_country.csv",
    }
    ground_truth = pl.read_csv(OUTPUT_DIR / file_names[gt_type]).with_columns(pl.col("question_key").str.to_lowercase())
    if gt_type == "global":
        ground_truth = ground_truth.with_columns(pl.lit("global").alias("gt_group"))
    elif gt_type == "language":
        ground_truth = ground_truth.rename({"lnge_iso": "gt_group"})
    elif gt_type == "country":
        ground_truth = ground_truth.rename({"cntry_an": "gt_group"})
    else:
        raise ValueError(f"Unknown gt_type: {gt_type}")
    return pl.DataFrame(results).join(ground_truth, on="question_key")
