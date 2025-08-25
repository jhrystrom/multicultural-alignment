import argparse

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from tqdm import tqdm

from multicultural_alignment.constants import CLEAN_MODEL_NAMES, LANGUAGE_MAP, PLOT_DIR
from multicultural_alignment.data import load_results
from multicultural_alignment.models import get_model_enum
from multicultural_alignment.plot import get_renamed_colours


def calculate_language_self_correlations(
    example_model: pl.DataFrame, pro_column: str = "response_pro_score", scoring_noise: float = 0.045
) -> pl.DataFrame:
    """
    Calculate self-consistency of language responses for a given model.
    Parameters:
    - example_model: Polars DataFrame containing the model's responses.
    - pro_column: The column name for the pro score in the DataFrame.
    - scoring_noise: The noise level to simulate in the scoring process. Based on human validated data.
    """
    language_responses = example_model.select("language", "question_key", pro_column).with_row_index("index")
    self_joined_responses = language_responses.join(language_responses, on=["question_key", "language"]).filter(
        pl.col("index") != pl.col("index_right")
    )
    observed_correlations = (
        self_joined_responses.group_by("language")
        .agg(pl.corr(pro_column, f"{pro_column}_right", method="spearman").alias("observed_self_consistency"))
        .with_columns(
            (pl.col("observed_self_consistency") / (1 - scoring_noise)).alias("self-consistency")  # Adjust for scoring noise
        )
    )
    return observed_correlations


def simplified_create_consistency(correlation_dicts: list[dict[str, pl.DataFrame]]) -> pl.DataFrame:
    return pl.concat(
        [
            consistency.with_columns(pl.lit(model).alias("model"))
            for correlation_dict in correlation_dicts
            for model, consistency in correlation_dict.items()
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot model self-consistency.")
    parser.add_argument("--scale", type=float, default=1.7)
    args = parser.parse_args()

    sns.set_theme(font_scale=args.scale, style="whitegrid")
    # 3x3 grid for the plot
    raw_responses = load_results(as_polars=True).filter(pl.col("template_type") == "survey_hypothetical")
    filtered_results = raw_responses.filter(pl.col("model_family") != "mistral").cast({"model_name": get_model_enum()})
    correlations_dicts = []
    NUM_ITERATIONS = 100
    for i in tqdm(range(NUM_ITERATIONS)):
        sampled_results = filtered_results.sample(fraction=1.0, with_replacement=True)
        correlation_dict = {}
        for (family, model_name), model_data in sampled_results.sort("model_name").group_by(
            "model_family", "model_name", maintain_order=True
        ):
            correlations = calculate_language_self_correlations(model_data)
            correlation_dict[model_name] = correlations
        correlations_dicts.append(correlation_dict)

    consistency_df = simplified_create_consistency(correlation_dicts=correlations_dicts).with_columns(
        pl.col("language").replace(LANGUAGE_MAP),
        pl.col("model").replace(CLEAN_MODEL_NAMES),
    )

    plt.figure(figsize=(10, 5))
    color_dict = get_renamed_colours()
    plot = sns.barplot(
        data=consistency_df.sort("language"),
        x="language",
        y="self-consistency",
        hue="model",
        palette=color_dict,
    )
    font_size = 30
    plot.set_ylabel("self-consistency", fontsize=font_size)
    plot.set_xlabel(None)
    plot.set_xticklabels(plot.get_xticklabels(), fontsize=font_size - 2)
    # move legend below and 1 row
    plt.axhline(y=0.66, color="black", linestyle=":", alpha=0.9, label="country minimum")
    plt.axhline(y=0.84, color="black", linestyle="--", alpha=0.9, label="country maximum")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    plt.ylim(0, 1)
    plt.xlabel(None)
    # Add horizontal reference lines
    plt.savefig(PLOT_DIR / "model-self_consistency.png", bbox_inches="tight")
