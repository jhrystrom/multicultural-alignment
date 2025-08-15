import polars as pl

from multicultural_alignment.directories import DATA_DIR

if __name__ == "__main__":
    results = (
        pl.read_parquet(DATA_DIR / "lang_model_responses_with_scores.parquet")
        .filter(
            (pl.col("language") == pl.col("response_language"))
            & (pl.col("template_type") == "survey_hypothetical")
            & (pl.col("model_family").is_in({"openai", "gemma", "olmo"}))
        )
        .select(
            "model_name",
            "model_family",
            "language",
            "prompt",
            "topic",
            "question_key",
            "response",
            "response_pro_con",
            "response_pro_score",
            "ground_truth_pro_score",
        )
    )
    results["model_name"].value_counts()

    clean_results = results.with_columns(
        pl.Series("prompt", [row["prompt"].format(topic=row["topic"]) for row in results.iter_rows(named=True)])
    )
    clean_results.write_parquet(DATA_DIR / "cleaned_lang_model_responses_with_scores.parquet")
