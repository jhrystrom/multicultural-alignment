import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from tqdm import tqdm

from multicultural_alignment.constants import LANGUAGE_MAP, PLOT_DIR
from multicultural_alignment.data import load_results
from multicultural_alignment.models import get_model_enum
from multicultural_alignment.plot import get_model_color_dict


def calculate_language_self_correlations(
    example_model: pl.DataFrame, pro_column: str = "response_pro_score"
) -> pl.DataFrame:
    language_responses = example_model.select("language", "question_key", pro_column).with_row_index("index")
    self_joined_responses = language_responses.join(language_responses, on=["question_key", "language"]).filter(
        pl.col("index") != pl.col("index_right")
    )
    return self_joined_responses.group_by("language").agg(
        pl.corr(pro_column, f"{pro_column}_right", method="spearman").alias("self-consistency")
    )


def simplified_create_consistency(correlation_dicts: list[dict[str, pl.DataFrame]]) -> pl.DataFrame:
    return pl.concat(
        [
            consistency.with_columns(pl.lit(model).alias("model"))
            for correlation_dict in correlation_dicts
            for model, consistency in correlation_dict.items()
        ]
    )


if __name__ == "__main__":
    sns.set_theme(font_scale=1.7, style="whitegrid")
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
        pl.col("language").replace(LANGUAGE_MAP)
    )

    sns.barplot(
        data=consistency_df.to_pandas(), x="language", y="self-consistency", hue="model", palette=get_model_color_dict()
    )
    # move legend below and 1 row
    plt.axhline(y=0.66, color="black", linestyle=":", alpha=0.9, label="country minimum")
    plt.axhline(y=0.84, color="black", linestyle="--", alpha=0.9, label="country maximum")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    plt.ylim(0, 1)
    plt.xlabel(None)
    # Add horizontal reference lines
    plt.savefig(PLOT_DIR / "model-self_consistency.png", bbox_inches="tight")
