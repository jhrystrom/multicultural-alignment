"""
Reads in two dataframes; wvs (with stances from the world value survey)
and responses (with responses from an LLM)
Calculates the alignment between the two dataframes
"""

import polars as pl
from tqdm import tqdm

from multicultural_alignment.constants import OUTPUT_DIR

EXTRA_COLUMNS = ["cntry_AN", "gwght", "pwght", "lnge_iso"]
TESTED_LANGUAGES = {"da", "en", "pt", "nl"}


def find_evs5_wvs7_cols(usdk_df: pl.DataFrame) -> set:
    return {column for column in usdk_df.columns if column.endswith("EVS5") or column.endswith("WVS7")}


# TODO: figure out how to process the `e` columns
def extract_relevant_columns(usdk_df) -> list:
    evs5_wvs7_cols = find_evs5_wvs7_cols(usdk_df)
    # The e columns are too difficult to process
    DIFFICULT_QUESTIONS = {f"E{i}" for i in range(224, 235)} | {"F028B_WVS7", "F025", "F034", "E028", "B008"}
    IRRELEVANT_COLUMNS = (
        {
            "cntry",
            "cntrycow",
            "fw_start",
            "fw_end",
            "cntry_y",
        }
        | evs5_wvs7_cols
        | DIFFICULT_QUESTIONS
    )
    relevant_columns = {col for col in usdk_df.columns if col.startswith(tuple("bcdefh".upper()))}
    return list(relevant_columns - IRRELEVANT_COLUMNS) + EXTRA_COLUMNS


if __name__ == "__main__":
    fulldf = pl.scan_csv("data/EVS_WVS_Joint_Csv_v5_0.csv", infer_schema_length=100000)

    sample = fulldf.head(2).collect()

    relevant_columns = extract_relevant_columns(fulldf.collect())

    max_values = fulldf.select(relevant_columns).max()
    selected_relevant = fulldf.select(relevant_columns)

    max_long = max_values.unpivot(value_name="maximum", variable_name="question_key")
    selected_long = (
        selected_relevant.unpivot(index=EXTRA_COLUMNS, variable_name="question_key", value_name="score")
        .join(max_long, on="question_key")
        .cast({"maximum": pl.Int64})
        .filter(pl.col("maximum").is_between(2, 11))
    )

    mid_value = (pl.col("maximum") // 2).alias("middle")

    pro_score_expr = (pl.sum("weighted_pro") / pl.sum("row_weight")).alias("pro_score")
    is_pro_score = (
        pl.when(pl.col("maximum") == 3)
        .then(pl.col("score") == 1)
        .when(pl.col("maximum") == 10)
        .then(pl.col("score") > 5)
        .otherwise(pl.col("score") <= pl.col("middle"))
    )

    long_with_pro = (
        selected_long.with_columns(mid_value)
        .filter(pl.col("score") > 0)
        .with_columns(
            (is_pro_score).alias("is_pro"),
        )
        .with_columns(
            pl.when(pl.col("question_key").str.starts_with("E035"))
            .then(~pl.col("is_pro"))
            .otherwise(pl.col("is_pro"))
            .cast(pl.UInt8)
        )
        .with_columns(
            (pl.col("gwght") * pl.col("pwght") * pl.col("is_pro")).alias("weighted_pro"),
            (pl.col("gwght") * pl.col("pwght")).alias("row_weight"),
        )
        .collect()
    )

    long_with_pro.group_by("cntry_AN").agg(pl.col("row_weight").sum()).sort("row_weight", descending=True)

    languages = (
        selected_relevant.group_by("cntry_AN", "lnge_iso")
        .agg(pl.col("gwght").sum())
        .sort("gwght", descending=True)
        .group_by("cntry_AN", maintain_order=True)
        .first()
        .drop("gwght")
        .filter(pl.col("lnge_iso").is_in(TESTED_LANGUAGES))
        .collect()
    )
    languages.write_csv(OUTPUT_DIR / "country_languages.csv")

    cntry_weights = long_with_pro.group_by("cntry_AN", "question_key").agg(pro_score_expr).join(languages, on="cntry_AN")

    GROUP_SIZE = 10
    relevant_long = long_with_pro.join(languages, on="cntry_AN")
    full = pl.concat(
        [
            relevant_long.filter(pl.int_range(pl.len()).shuffle().over("cntry_AN", "question_key") < GROUP_SIZE)
            .group_by("cntry_AN", "question_key")
            .agg(pro_score_expr)
            for _ in tqdm(range(100))
        ]
    )

    full.join(full, on=["cntry_AN", "question_key"]).group_by("cntry_AN").agg(
        pl.corr("pro_score", "pro_score_right").alias("self_consistency")
    ).sort("self_consistency").write_csv(OUTPUT_DIR / "gt_self_consistency.csv")

    cntry_bootstrapped = pl.concat(
        [
            long_with_pro.sample(fraction=1, with_replacement=True).group_by("cntry_AN", "question_key").agg(pro_score_expr)
            for _ in tqdm(range(50))
        ]
    )
    cntry_bootstrapped.join(cntry_bootstrapped, on=["cntry_AN", "question_key"]).group_by("cntry_AN").agg(
        pl.corr("pro_score", "pro_score_right").alias("self_consistency")
    ).sort("self_consistency")
    long_with_pro

    lang_weights = (
        long_with_pro.group_by("lnge_iso", "question_key")
        .agg(pro_score_expr)
        .filter(pl.col("lnge_iso").is_in(TESTED_LANGUAGES))
    )
    global_weights = long_with_pro.group_by("question_key").agg(pro_score_expr)

    print(cntry_weights.filter(pl.col("cntry_AN") == "US").sort("pro_score"))

    lang_weights.write_csv(OUTPUT_DIR / "ground_truth_per_language.csv")
    cntry_weights.rename({"cntry_AN": "cntry_an"}).write_csv(OUTPUT_DIR / "ground_truth_every_country.csv")
    global_weights.write_csv(OUTPUT_DIR / "ground_truth_global.csv")
