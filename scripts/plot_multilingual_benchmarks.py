import argparse
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import statsmodels.formula.api as smf
from loguru import logger
from statsmodels.regression.linear_model import RegressionResults
from tqdm import tqdm

from multicultural_alignment.constants import COUNTRY_LANG_MAP, LANGUAGE_MAP, OUTPUT_DIR, PLOT_DIR
from multicultural_alignment.models import add_families_df, get_model_enum
from multicultural_alignment.plot import get_family_color_dict, get_model_color_dict
from multicultural_alignment.regression import extract_results_df, extract_term, save_regression_results


class RSquared(TypedDict):
    marginal: float
    conditional: float


MODEL_NAME_MAPPING = {
    # OpenAI models
    "GPT-3.5 Turbo": "gpt-3.5-turbo-0125",
    "GPT-4 Turbo": "gpt-4-turbo-2024-04-09",
    "gpt-4-turbo": "gpt-4-turbo-2024-04-09",
    "GPT-4o": "gpt-4o",
    "gpt-4o-2024-05-13": "gpt-4o",
    # Gemma models
    "Gemma-2 2B": "gemma-2-2b-it",
    "Gemma-2 9B": "gemma-2-9b-it",
    "Gemma-2 27B": "gemma-2-27b-it",
    # OLMo models
    "OLMo-2 7B": "OLMo-2-1124-7B-Instruct",
    "OLMo-2 13B": "OLMo-2-1124-13B-Instruct",
    "OLMo-2 32B": "OLMo-2-0325-32B-Instruct",
}
ADJUSTMENT_FORMULAS = {
    "interaction": "alignment ~ 0 + consistency:language + language:model_name",
    "normal": "alignment ~ 0 + consistency + language:model_name",
}


def invert_dict(d: dict) -> dict:
    return {v: k for k, v in d.items()}


def calculate_rsquared(model) -> RSquared:
    fixed_effects_variance = np.var(model.fittedvalues)
    random_effects_variance = model.cov_re.iloc[0, 0]
    residual_variance = model.scale

    # Calculate Marginal and Conditional R^2 from theese extracted variances:
    R2_m = fixed_effects_variance / (fixed_effects_variance + random_effects_variance + residual_variance)
    R2_c = (fixed_effects_variance + random_effects_variance) / (
        fixed_effects_variance + random_effects_variance + residual_variance
    )
    return RSquared(marginal=R2_m, conditional=R2_c)


def get_benchmark_data():
    benches = (
        pl.read_csv(OUTPUT_DIR / "multilingual_benchmarks.csv")
        .with_columns(pl.col("model").replace(MODEL_NAME_MAPPING))
        .cast({"model": get_model_enum()})
        .filter(~pl.col("benchmark").str.contains("ENEM"))
        .drop_nulls()
    )
    return benches


def calculate_adjusted_alignments(regression_data: pl.DataFrame, regression_type: str = "normal") -> pl.DataFrame:
    all_alignments = []
    N_ITERATIONS = 100
    for _ in tqdm(range(N_ITERATIONS)):
        resampled = regression_data.sample(fraction=1.0, with_replacement=True)
        multilingual_model = smf.ols(
            formula=ADJUSTMENT_FORMULAS[regression_type],
            data=resampled.cast({"model_name": str}).to_pandas(),
        )
        multilingual_results = multilingual_model.fit()
        all_alignments.append(
            extract_results_df(multilingual_results)
            .with_columns(
                extract_term("language"),
                extract_term("model_name"),
            )
            .select(
                pl.col("model_name"),
                pl.col("language"),
                pl.col("coefficient").alias("adjusted_alignment"),
            )
        )
    return pl.concat(all_alignments).drop_nulls()


def get_language_regression_data(benchmarks: pl.DataFrame, experiment_data: pl.DataFrame) -> pl.DataFrame:
    aggregated_bench = benchmarks.group_by("model", "language").agg(pl.col("score").mean())
    regression_data = add_families_df(
        experiment_data.filter(pl.col("gt_type") == "language")
        .with_row_index()
        .with_columns((pl.col("gt_group") == pl.col("language")).alias("native"))
    )
    regression_data = (
        regression_data.filter(pl.col("family") != "mistral")
        .filter(pl.col("model_name") != "Baseline_fifty_percent")
        .sort("model_name")
        .filter("native")
        .with_columns(pl.col("language").replace(LANGUAGE_MAP))
    )
    return (
        regression_data.join(aggregated_bench, left_on=["model_name", "language"], right_on=["model", "language"])
        .rename({"score": "multilingual", "metric_value": "alignment"})
        .with_columns(
            pl.col("multilingual") / 100.0,
        )
    )


def get_country_regression_data(benchmarks: pl.DataFrame, experiment_data: pl.DataFrame) -> pl.DataFrame:
    aggregated_bench = benchmarks.group_by("model", "language").agg(pl.col("score").mean())
    regression_data = add_families_df(
        experiment_data.filter(pl.col("gt_type") == "country")
        .with_columns(pl.col("gt_group").replace(COUNTRY_LANG_MAP).alias("spoken_language"))
        .with_columns((pl.col("spoken_language") == pl.col("language")).alias("native"))
        .with_row_index()
    )
    regression_data = (
        regression_data.filter(pl.col("family") != "mistral")
        .filter(pl.col("model_name") != "Baseline_fifty_percent")
        .sort("model_name")
        .with_columns(pl.col("language").replace(LANGUAGE_MAP))
    )
    return (
        regression_data.join(aggregated_bench, left_on=["model_name", "language"], right_on=["model", "language"])
        .rename({"score": "multilingual", "metric_value": "alignment"})
        .with_columns(
            pl.col("multilingual") / 100.0,
        )
    )


def get_experiment_data() -> pl.DataFrame:
    return pl.read_csv(OUTPUT_DIR / "spearman_n100_extreme_total_gt_alignment.csv").rename({"response_language": "language"})


def create_plot_data(
    benches: pl.DataFrame,
    multilingual_coefficients: pl.DataFrame,
    regression_data: pl.DataFrame,
    regression_type: str = "normal",
) -> pl.DataFrame:
    alignment_clean = calculate_adjusted_alignments(regression_data=regression_data, regression_type=regression_type)
    avg_score = benches.group_by(["model", "language"]).agg(pl.col("score").mean()).rename({"model": "model_name"})
    return (
        add_families_df(alignment_clean)
        .join(avg_score, on=["model_name", "language"])
        .sort("model_name")
        .cast({"model_name": str})
        .rename({"score": "multilingual capability"})
        .with_columns(pl.col("language").replace(LANGUAGE_MAP))
        .rename({"adjusted_alignment": "cultural alignment"})
        .join(multilingual_coefficients.filter(pl.col("regression_type") == regression_type), on=["family", "language"])
    )


def run_multilingual_regression(regression_data: pl.DataFrame, regression_formula: str | None = None) -> RegressionResults:
    if regression_formula is None:
        regression_formula = "alignment ~ 0 + consistency + multilingual:family:language"
    multilingual_relations = smf.mixedlm(
        formula=regression_formula,
        groups="model_name",
        data=regression_data.to_pandas(),
    )
    family_results = multilingual_relations.fit()
    logger.info("Regression results:")
    logger.info(family_results.summary())
    return family_results


def run_multicountry_regression(country_regression_data: pl.DataFrame) -> pl.DataFrame:
    for (language,), language_data in country_regression_data.group_by("language"):
        if language != "English":
            continue
        logger.info(f"Running regression for language: {language}")
        regression_formula = "alignment ~ 0 + consistency + multilingual:family:gt_group"
        country_results = smf.mixedlm(
            formula=regression_formula,
            groups="model_name",
            data=language_data.to_pandas(),
        ).fit()
    return country_results


def plot_multilingual_coefficients(regression_data: pl.DataFrame, regression_type: str | None = None) -> pl.DataFrame:
    formulas = {
        "normal": "alignment ~ 0 + consistency + multilingual:family:language",
        "no_consistency": "alignment ~ 0 + multilingual:family:language",
        "interaction": "alignment ~ 0 + consistency:language + multilingual:family:language",
    }
    all_multilingual_results_df = []
    for name, formula in formulas.items():
        if regression_type is not None and name != regression_type:
            continue
        logger.info(f"Running regression with formula: {formula}")
        regression_results = run_multilingual_regression(regression_data, regression_formula=formula)
        save_regression_results(regression_results, regression_type=name)
        rsquared = calculate_rsquared(regression_results)
        logger.info(f"R-squared values: {rsquared}")
        multilingual_results_df = (
            extract_results_df(regression_results)
            .with_columns(
                extract_term("family"),
                extract_term("language"),
            )
            .drop_nulls()
        )
        assert "language" in multilingual_results_df.columns, multilingual_results_df.columns
        plot_coefficients_lower = multilingual_results_df.select(
            ["language", "family", pl.col("conf_int_lower").alias("coefficient")]
        )
        plot_coefficients_upper = multilingual_results_df.select(
            ["language", "family", pl.col("conf_int_upper").alias("coefficient")]
        )
        plot_coefficients_combined = pl.concat(
            [
                plot_coefficients_lower,
                plot_coefficients_upper,
                multilingual_results_df.select(["language", "family", "coefficient"]),
            ]
        ).with_columns(pl.col("language").replace(LANGUAGE_MAP))
        sns.barplot(
            data=plot_coefficients_combined,
            x="language",
            y="coefficient",
            hue="family",
            palette=get_family_color_dict(),
            errorbar=("pi", 100),
        )
        plt.ylabel("$\\beta_{multilingual}$")
        plt.xlabel(None)
        plt.legend(title=None)
        plt.savefig(PLOT_DIR / f"multilingual_coefficients_{name}.png", bbox_inches="tight")
        plt.close()
        all_multilingual_results_df.append(
            multilingual_results_df.with_columns(
                pl.lit(name).alias("regression_type"),
            )
        )
    return pl.concat(all_multilingual_results_df).select("family", "language", "coefficient", "p_value", "regression_type")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot multilingual benchmarks alignment")
    parser.add_argument(
        "--regression-type",
        type=str,
        choices=["normal", "no_consistency", "interaction"],
        default="normal",
        help="Type of regression to use for plotting coefficients",
    )
    args = parser.parse_args()
    regression_type = args.regression_type

    sns.set_theme(font_scale=1.6, style="whitegrid")
    benchmarks = get_benchmark_data()
    experiment_data = get_experiment_data()
    regression_data = get_language_regression_data(benchmarks=benchmarks, experiment_data=experiment_data)
    # country_regression_data = get_country_regression_data(benchmarks=benchmarks, experiment_data=experiment_data)
    multilingual_coefficients = plot_multilingual_coefficients(
        regression_data=regression_data, regression_type=regression_type
    )

    benchmark_alignment = create_plot_data(
        benches=benchmarks,
        multilingual_coefficients=multilingual_coefficients,
        regression_data=regression_data,
        regression_type=regression_type,
    )

    plot = sns.relplot(
        data=benchmark_alignment,
        x="multilingual capability",
        y="cultural alignment",
        hue="model_name",
        col="language",
        col_wrap=2,
        palette=get_model_color_dict(),
    )

    # Now add regression lines for each language and family
    # Loop through each subplot (FacetGrid axes)
    family_colours = get_family_color_dict()
    for ax in plot.axes.flat:
        # Get the language for this subplot from the title
        lang = ax.get_title().split(" = ")[-1]  # This gets the language name from the subplot title

        # For each family, plot the regression line
        for family in ["gemma", "openai", "olmo"]:
            plot_data = benchmark_alignment.filter((pl.col("family") == family) & (pl.col("language") == lang))

            # Get intercept and slope
            slope = plot_data["coefficient"].first() / 100
            intercept = (plot_data["cultural alignment"] - (slope * plot_data["multilingual capability"])).mean()

            # Create x values spanning the plot range
            x = np.array([plot_data["multilingual capability"].min(), plot_data["multilingual capability"].max()])

            # Calculate y values
            y = slope * x + intercept

            # Add the line to the plot
            ax.plot(x, y, "--", alpha=0.8, color=family_colours[family])
            # Add star if significant (p < 0.05)
            p_value = plot_data["p_value"].first()
            if p_value < 0.05 and slope > 0:
                # Calculate middle point of the line
                x_mid = x.mean()
                y_mid = slope * x_mid + intercept
                # Add star slightly above the middle point
                offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05
                ax.text(x_mid, y_mid - offset, "*", ha="center", va="bottom", color=family_colours[family], fontsize=40)

    sns.move_legend(plot, "upper center", bbox_to_anchor=(0.41, -0.0), ncol=3, title=None)
    plt.savefig(PLOT_DIR / "multilingual_benchmarks_alignment-model.png", bbox_inches="tight")
