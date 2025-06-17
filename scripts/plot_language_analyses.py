import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

import multicultural_alignment.models
from multicultural_alignment.constants import COUNTRY_MAP, LANGUAGE_MAP, OUTPUT_DIR, PLOT_DIR
from multicultural_alignment.plot import get_model_color_dict

# Constants
GT_ALWAYS = {"global", "US", "en"}
TARGET_FAMILIES = ["openai", "gemma", "olmo"]


def get_model_filter(family: str, language: str = "en") -> pl.Expr:
    """Get model filter for a specific family and language."""
    return pl.col("family") == family


def load_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load and prepare experiment and country language data."""
    country_languages = pl.read_csv(OUTPUT_DIR / "country_languages.csv")
    experiment_data = pl.read_csv(OUTPUT_DIR / "spearman_n100_extreme_total_gt_alignment.csv")
    experiment_data = multicultural_alignment.models.add_families_df(experiment_data)

    return experiment_data, country_languages


def filter_experiment_data(experiment_data: pl.DataFrame, country_languages: pl.DataFrame) -> pl.DataFrame:
    """Apply initial filtering to experiment data."""
    experiment_data = experiment_data.join(country_languages, left_on="gt_group", right_on="cntry_AN", how="left")

    language_filter = (
        (pl.col("response_language") == pl.col("lnge_iso"))
        | (pl.col("response_language") == pl.col("gt_group"))
        | (pl.col("gt_group").is_in(GT_ALWAYS))
    )

    return experiment_data.filter(language_filter)


def prepare_plot_data(experiment_data: pl.DataFrame, country_languages: pl.DataFrame) -> tuple[pl.DataFrame, str]:
    """Prepare data for plotting and extract metric value."""
    responses_languages = experiment_data["response_language"].unique()
    common_filter = (
        pl.col("gt_group").is_in(responses_languages)
        | pl.col("lnge_iso").is_in(responses_languages)
        | (pl.col("gt_type") == "global")
    )
    model_filter = pl.col("family").is_in(TARGET_FAMILIES) | pl.col("model_name").cast(str).str.starts_with(
        "Baseline_uniform"
    )

    filtered_plot_data = (
        experiment_data.filter(common_filter & model_filter)
        .rename({"response_language": "language", "metric_value": "alignment", "gt_type": "level"})
        .with_columns(
            pl.col("language").replace(LANGUAGE_MAP),
            pl.col("gt_group").replace(LANGUAGE_MAP).replace(COUNTRY_MAP),
            pl.col("level").str.to_titlecase(),
        )
    )

    # Extract metric from the data (assuming all rows have the same metric)
    metric = experiment_data["metric"][0]

    return filtered_plot_data, metric


def create_english_plot(filtered_plot_data: pl.DataFrame, metric: str) -> None:
    """Create and save English language plot."""
    sns.set_theme(style="whitegrid", font_scale=1.7)

    english_plot = filtered_plot_data.filter(pl.col("language") == "English")
    plot = sns.catplot(
        data=english_plot.sort("model_name", "gt_group").cast({"model_name": pl.String}),
        col="level",
        x="gt_group",
        hue="model_name",
        y="alignment",
        kind="bar",
        sharex=False,
        errorbar="ci",
        palette=get_model_color_dict(),
    ).set_titles("{col_name}")

    plot.set_xticklabels(rotation=57)
    plot.set(xlabel=None)
    sns.move_legend(plot, loc="upper center", bbox_to_anchor=(0.45, -0.15), ncol=4, title=None)
    plt.savefig(PLOT_DIR / f"english_{metric}_gt_alignment.png", bbox_inches="tight")


def create_monocultural_plot(filtered_plot_data: pl.DataFrame, metric: str) -> None:
    """Create and save monocultural (Danish/Dutch) plot."""
    sns.set_theme(style="whitegrid", font_scale=2.3)

    monocultural_plot = filtered_plot_data.filter(pl.col("language").is_in({"Danish", "Dutch"})).with_columns(
        pl.col("language").replace(LANGUAGE_MAP),
        pl.col("gt_group").replace(LANGUAGE_MAP).replace(COUNTRY_MAP),
    )

    plot = sns.catplot(
        data=monocultural_plot.sort("model_name", "gt_group").cast({"model_name": pl.String}),
        col="level",
        x="gt_group",
        row="language",
        hue="model_name",
        y="alignment",
        kind="bar",
        sharex=False,
        errorbar="ci",
        aspect=1.3,
        palette=get_model_color_dict(),
    ).set_titles("{row_name} | {col_name}")

    sns.move_legend(plot, loc="upper center", bbox_to_anchor=(0.42, 0.05), ncol=4, title=None)

    for ax in plot.axes.flat:
        ax.set_ylabel("")

    plot.figure.supylabel("cultural alignment", x=0.035)
    plot.set(xlabel=None)
    plt.savefig(PLOT_DIR / f"monolingual-{metric}-gt_alignment.png", bbox_inches="tight")


def create_portuguese_plot(filtered_plot_data: pl.DataFrame, metric: str) -> None:
    """Create and save Portuguese language plot."""
    portuguese_plot = filtered_plot_data.filter(pl.col("language") == "Portuguese")
    plot = sns.catplot(
        data=portuguese_plot.sort("model_name", "gt_group").cast({"model_name": pl.String}),
        col="level",
        x="gt_group",
        hue="model_name",
        y="alignment",
        kind="bar",
        sharex=False,
        errorbar="ci",
        palette=get_model_color_dict(),
        aspect=1.1,
    ).set_titles("{col_name}")

    sns.move_legend(plot, loc="upper center", bbox_to_anchor=(0.42, 0.1), ncol=4, title=None)
    plot.set(xlabel=None)
    plt.savefig(PLOT_DIR / f"pt-{metric}_gt_alignment.png", bbox_inches="tight")


def main() -> None:
    """Main function to orchestrate the cultural alignment analysis."""
    # Load data
    experiment_data, country_languages = load_data()

    # Filter experiment data
    experiment_data = filter_experiment_data(experiment_data, country_languages)

    # Prepare plot data
    filtered_plot_data, metric = prepare_plot_data(experiment_data, country_languages)

    # Create all plots
    create_english_plot(filtered_plot_data, metric)
    create_monocultural_plot(filtered_plot_data, metric)
    create_portuguese_plot(filtered_plot_data, metric)


if __name__ == "__main__":
    main()
