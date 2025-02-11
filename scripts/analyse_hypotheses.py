import argparse
from dataclasses import dataclass
from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from line_profiler import profile
from loguru import logger
from scipy import stats
from tqdm import tqdm

from multicultural_alignment.constants import COUNTRY_LANG_MAP, OUTPUT_DIR, PLOT_DIR
from multicultural_alignment.data import GTType, join_all_gt, load_ground_truth, load_results
from multicultural_alignment.models import MODEL_FAMILIES, get_families_df, get_model_enum
from multicultural_alignment.plot import get_model_color_dict

# Function Interfaces
MetricFunction = Callable[[pd.Series, pd.Series], float]
AnalysisFunction = Callable[[pd.DataFrame, pd.DataFrame, MetricFunction, int], pd.DataFrame]


@dataclass
class Analysis:
    name: str
    analysis_func: AnalysisFunction
    plot_func: Callable
    plot_family_func: Callable | None
    filename: str


# Constants
N = 500
NUM_ROUNDS = 1000
RANDOM_SEED = 5125
ENGLISH_SPEAKING_COUNTRIES = [country for country, language in COUNTRY_LANG_MAP.items() if language == "en"]


# Metric Implementations
def spearman_correlation(x: pd.Series, y: pd.Series) -> float:
    return stats.spearmanr(x, y).statistic


def mean_alignment(x: pd.Series, y: pd.Series) -> float:
    return 1 - np.abs(x - y).mean()


def pearson_correlation(x: pd.Series, y: pd.Series) -> float:
    return stats.pearsonr(x, y).statistic


# Metric Factory
METRICS: dict[str, MetricFunction] = {
    "spearman": spearman_correlation,
    "mean_alignment": mean_alignment,
    "pearson": pearson_correlation,
}


def generate_fifty_percent_baseline(data: pd.DataFrame, sd: float = 0.01) -> pd.DataFrame:
    data["response_pro_score"] = np.random.normal(0.5, scale=sd, size=len(data))
    data["response_pro_score"] = data["response_pro_score"].clip(0, 1)
    return data


def generate_uniform_random_baseline(data: pd.DataFrame) -> pd.DataFrame:
    data["response_pro_score"] = np.random.uniform(0, 1, len(data))
    return data


# Dictionary of baseline generators
BASELINE_GENERATORS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "fifty_percent": generate_fifty_percent_baseline,
    "uniform_random": generate_uniform_random_baseline,
}


def get_metric(metric_name: str) -> MetricFunction:
    return METRICS.get(metric_name, spearman_correlation)  # Default to Spearman if not found


# Utility Functions
def calculate_metric(data: pd.DataFrame, x: str, y: str, metric_func: MetricFunction) -> float:
    return metric_func(data[x], data[y])


def sample_and_calculate(
    data: pd.DataFrame, group_cols: list[str], x: str, y: str, n: int, metric_func: MetricFunction
) -> pd.DataFrame:
    return (
        data.groupby(group_cols)
        .sample(n=n, replace=True)
        .groupby(group_cols)
        .apply(lambda df: calculate_metric(df, x, y, metric_func), include_groups=False)
        .reset_index()
        .rename(columns={0: "metric_value"})
    )


@profile
def calculate_consistency(
    example_model: pl.DataFrame,
    language_column: str = "language",
    pro_column: str = "response_pro_score",
    group_column: str = "model_name",
) -> pd.DataFrame:
    language_responses = example_model.select("language", "question_key", pro_column, group_column).with_row_index("index")
    self_joined_responses = language_responses.join(
        language_responses, on=["question_key", group_column, language_column]
    ).filter(pl.col("index") != pl.col("index_right"))
    return self_joined_responses.group_by("language", group_column).agg(
        pl.corr(pro_column, f"{pro_column}_right", method="pearson").alias("consistency")
    )


@profile
def polars_sample_and_calculate(
    data: pl.DataFrame, group_cols: list[str], x: str, y: str, metric_func: Literal["spearman", "mean_alignment", "pearson"]
) -> pl.DataFrame:
    metric_funcs = {
        "spearman": pl.corr(a=pl.col(x), b=pl.col(y), method="spearman"),
        "mean_alignment": 1 - (pl.col(x) - pl.col(y)).abs().mean(),
        "pearson": pl.corr(a=pl.col(x), b=pl.col(y), method="pearson"),
    }
    sampled_data = data.sample(fraction=1.0, with_replacement=True)
    new_results = sampled_data.group_by(group_cols).agg(metric_funcs[metric_func].alias("metric_value"))
    consistency = calculate_consistency(sampled_data)
    return new_results.join(consistency, left_on=["response_language", "model_name"], right_on=["language", "model_name"])


def polars_bootstrap(
    data: pl.DataFrame, group_cols: list[str], x: str, y: str, metric_func: str, n_bootstraps: int = 500
) -> pl.DataFrame:
    metric_funcs = {
        "spearman": pl.corr(a=pl.col(x), b=pl.col(y), method="spearman"),
        "mean_alignment": 1 - (pl.col(x) - pl.col(y)).abs().mean(),
        "pearson": pl.corr(a=pl.col(x), b=pl.col(y), method="pearson"),
    }
    bootstrap_list = pl.Series("bootstrap_index", np.arange(n_bootstraps).reshape(1, -1).repeat(data.height, axis=0))
    return (
        data.with_columns(bootstrap_list)
        .explode("bootstrap_index")
        .sample(fraction=1.0, with_replacement=True)
        .group_by(group_cols + ["bootstrap_index"])
        .agg(metric_funcs[metric_func].alias("metric_value"))
    )


# Updated Baseline Function
def generate_baseline_responses(results: pd.DataFrame, baseline_types: list[str]) -> pd.DataFrame:
    # Copy the first model's data
    first_model = results["model_name"].unique()[0]
    baseline_template = results[results["model_name"] == first_model].copy()

    all_baselines = []

    for baseline_type in baseline_types:
        # Get the appropriate baseline generator
        baseline_generator = BASELINE_GENERATORS.get(baseline_type)
        if baseline_generator is None:
            raise ValueError(f"Unknown baseline type: {baseline_type}")

        # Generate baseline responses
        baseline_responses = baseline_generator(baseline_template.copy())

        # Update model name and family
        baseline_responses["model_name"] = f"Baseline_{baseline_type}"
        baseline_responses["model_family"] = "Baseline"

        all_baselines.append(baseline_responses)

    return pd.concat(all_baselines, ignore_index=True)


# Analysis Functions
def analyze_american_superiority_bias(
    results: pd.DataFrame, ground_truth: pd.DataFrame, metric_func: MetricFunction, num_rounds: int = NUM_ROUNDS
) -> pd.DataFrame:
    english_results = results[results["language"] == "en"].drop(columns=["language"])
    english_ground_truth = ground_truth[ground_truth["language"] == "en"].drop(columns=["language"])
    results_with_gt = english_results.merge(english_ground_truth, on="question_key", suffixes=("_response", "_ground_truth"))

    return pd.concat(
        [
            sample_and_calculate(
                results_with_gt, ["model_name", "cntry_an"], "response_pro_score", "ground_truth_pro_score", N, metric_func
            )
            for _ in tqdm(range(num_rounds))
        ]
    )


def analyze_cross_linguistic_transfer(
    results: pd.DataFrame, ground_truth: pd.DataFrame, metric_func: MetricFunction, num_rounds: int = NUM_ROUNDS
) -> pd.DataFrame:
    non_american = results[(results["language"] == results["response_language"])]
    non_american_gt = non_american.merge(
        ground_truth, on=["question_key", "language"], suffixes=("_response", "_ground_truth")
    )
    american_gt = (
        ground_truth[ground_truth["cntry_an"] == "US"]
        .drop(columns=["language", "cntry_an"])
        .rename(columns={"ground_truth_pro_score": "american_pro_score"})
    )
    non_american_gt = non_american_gt.merge(american_gt, on="question_key", suffixes=("_native", "_american"))
    non_american_gt = non_american_gt[non_american_gt["cntry_an"] != "US"]

    def single_round_analysis():
        native_metric = sample_and_calculate(
            non_american_gt,
            ["model_name", "cntry_an", "language"],
            "response_pro_score",
            "ground_truth_pro_score",
            N,
            metric_func,
        ).rename(columns={"metric_value": "Native Alignment"})
        us_metric = sample_and_calculate(
            non_american_gt,
            ["model_name", "cntry_an", "language"],
            "response_pro_score",
            "american_pro_score",
            N,
            metric_func,
        ).rename(columns={"metric_value": "US Alignment"})
        return native_metric.merge(us_metric, on=["model_name", "cntry_an"])

    return pd.concat([single_round_analysis() for _ in tqdm(range(num_rounds))])


def add_perfect_model(results_with_gt: pd.DataFrame) -> pd.DataFrame:
    perfect_model = (
        results_with_gt.groupby(["question_key", "cntry_an", "response_language", "language", "template_type"])[
            "ground_truth_pro_score"
        ]
        .mean()
        .reset_index()
    )
    perfect_model["model_name"] = "Baseline_perfect"
    perfect_model["model_family"] = "Baseline"
    perfect_model["response_pro_score"] = perfect_model["ground_truth_pro_score"].round(decimals=1)
    return perfect_model


def analyze_total_gt_alignment(
    results: pd.DataFrame | pl.DataFrame,
    metric_func: str = "mean_alignment",
    num_rounds: int = NUM_ROUNDS,
    gt_type: GTType = "language",
) -> pl.DataFrame:
    results_with_gt = join_all_gt(results, gt_type=gt_type)
    analysis_results = pl.concat(
        [
            polars_sample_and_calculate(
                data=results_with_gt,
                group_cols=["model_name", "gt_group", "response_language"],
                x="response_pro_score",
                y="pro_score",
                metric_func=metric_func,
            ).with_columns(pl.lit(i).alias("round"))
            for i in tqdm(range(num_rounds))
        ]
    )
    return analysis_results


def analyze_absolute_all_combinations(
    results: pd.DataFrame,
    ground_truth: pd.DataFrame | None,
    metric_func: MetricFunction = "mean_alignment",
    num_rounds: int = NUM_ROUNDS,
) -> pd.DataFrame:
    return pl.concat(
        [
            analyze_total_gt_alignment(
                results=results, metric_func=metric_func, num_rounds=num_rounds, gt_type=gt_type
            ).with_columns(pl.lit(gt_type).alias("gt_type"))
            for gt_type in tqdm(["global", "language", "country"])
        ]
    ).with_columns(pl.lit(metric_func).alias("metric"))


def analyze_relative_performance_all_combinations(
    results: pd.DataFrame,
    ground_truth: pd.DataFrame,
    metric_func: MetricFunction = mean_alignment,
    num_rounds: int = NUM_ROUNDS,
) -> pd.DataFrame:
    results_with_gt = results.merge(
        ground_truth, on=["question_key"], suffixes=("_response", "_ground_truth"), how="outer"
    ).dropna()

    def single_round_analysis():
        baseline_mask = results_with_gt["model_name"] == "Baseline_uniform_random"
        baseline_results = results_with_gt[baseline_mask]
        baseline_round = sample_and_calculate(
            baseline_results,
            ["model_name", "response_language", "cntry_an"],
            "response_pro_score",
            "ground_truth_pro_score",
            N,
            metric_func,
        )
        normal_results = sample_and_calculate(
            results_with_gt[~baseline_mask],
            ["model_name", "response_language", "cntry_an"],
            "response_pro_score",
            "ground_truth_pro_score",
            N,
            metric_func,
        )
        joined_results = normal_results.merge(
            baseline_round.drop("model_name", axis=1), on=["cntry_an", "response_language"], how="left"
        )
        joined_results = joined_results.assign(
            metric_value=joined_results["metric_value_x"] - joined_results["metric_value_y"]
        ).drop(["metric_value_x", "metric_value_y"], axis=1)
        return joined_results

    return pd.concat([single_round_analysis() for _ in tqdm(range(num_rounds))])


def analyze_relative_performance(
    results: pd.DataFrame,
    ground_truth: pd.DataFrame,
    metric_func: MetricFunction = mean_alignment,
    num_rounds: int = NUM_ROUNDS,
) -> pd.DataFrame:
    results_with_gt = results.merge(ground_truth, on=["question_key", "language"], suffixes=("_response", "_ground_truth"))

    def single_round_analysis():
        baseline_mask = results_with_gt["model_name"] == "Baseline_uniform_random"
        baseline_results = results_with_gt[baseline_mask]
        baseline_round = sample_and_calculate(
            baseline_results,
            ["model_name", "response_language", "cntry_an"],
            "response_pro_score",
            "ground_truth_pro_score",
            N,
            metric_func,
        )
        normal_results = sample_and_calculate(
            results_with_gt[~baseline_mask],
            ["model_name", "response_language", "cntry_an"],
            "response_pro_score",
            "ground_truth_pro_score",
            N,
            metric_func,
        )
        joined_results = normal_results.merge(
            baseline_round.drop("model_name", axis=1), on=["cntry_an", "response_language"], how="left"
        )
        joined_results = joined_results.assign(
            metric_value=joined_results["metric_value_x"] - joined_results["metric_value_y"]
        ).drop(["metric_value_x", "metric_value_y"], axis=1)
        return joined_results

    return pd.concat([single_round_analysis() for _ in tqdm(range(num_rounds))])


def analyze_absolute_performance(
    results: pd.DataFrame,
    ground_truth: pd.DataFrame,
    metric_func: MetricFunction = mean_alignment,
    num_rounds: int = NUM_ROUNDS,
) -> pd.DataFrame:
    results_with_gt = results.merge(ground_truth, on=["question_key", "language"], suffixes=("_response", "_ground_truth"))
    analysis_results = pd.concat(
        [
            sample_and_calculate(
                data=results_with_gt,
                group_cols=["model_name", "response_language", "cntry_an"],
                x="response_pro_score",
                y="ground_truth_pro_score",
                n=N,
                metric_func=metric_func,
            )
            for _ in tqdm(range(num_rounds))
        ]
    )
    return analysis_results


def plot_relative_performance(data: pd.DataFrame, filename: str):
    color_dict = get_model_color_dict()
    g = sns.barplot(
        data=data.reset_index(drop=True),
        x="cntry_an",
        y="metric_value",
        hue="model_name",
        palette=color_dict,
    )
    g.set_title("Relative Performance")
    g.set_ylabel("Alignment Score (Model - Random)")
    g.set_xlabel("Country")
    # Move legend outside of the plot (to the left)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.xticks(rotation=45)
    plt.savefig(PLOT_DIR / filename, bbox_inches="tight")


def plot_relative_family_performance(results: pl.DataFrame | pd.DataFrame, filename: str) -> None:
    plot_data = results if isinstance(results, pl.DataFrame) else pl.DataFrame(results)
    plot_data = plot_data.cast({"model_name": get_model_enum()})
    color_dict = get_model_color_dict()
    for (family,), family_data in plot_data.group_by("model_family"):
        plt.clf()
        g = sns.catplot(
            data=family_data.sort("model_name").cast({"model_name": pl.String}),
            x="cntry_an",
            y="metric_value",
            hue="model_name",
            palette=color_dict,
            kind="bar",
            sharex=False,
            sharey=True,
            col="response_language",
        )
        g.set_axis_labels("Country", "Alignment Score (Model - Random)")
        g.set_titles("{col_name}")
        g.figure.suptitle(f"Relative Performance by Model Family ({family})", fontsize=16)
        g.figure.subplots_adjust(top=0.9, bottom=0.15)  # Adjusted bottom margin
        plt.savefig(PLOT_DIR / f"{family}_{filename}", bbox_inches="tight")


def plot_absolute_performance(data: pd.DataFrame, filename: str):
    color_dict = get_model_color_dict()
    g = sns.barplot(
        data=data.reset_index(drop=True),
        x="cntry_an",
        y="metric_value",
        hue="model_name",
        palette=color_dict,
    )
    g.set_title("Absolute Performance")
    g.set_ylabel("Alignment Score")
    g.set_xlabel("Country")
    # Move legend outside of the plot (to the left)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.xticks(rotation=45)
    plt.savefig(PLOT_DIR / filename, bbox_inches="tight")


def analyze_monocultural_alignment(
    results: pd.DataFrame, ground_truth: pd.DataFrame, metric_func: MetricFunction, num_rounds: int = NUM_ROUNDS
) -> pd.DataFrame:
    us_results_with_other_gt = (
        results[results["language"] == "en"]
        .merge(
            ground_truth[~ground_truth["cntry_an"].isin(ENGLISH_SPEAKING_COUNTRIES)],
            on="question_key",
            suffixes=("_response", "_ground_truth"),
        )
        .drop(columns=["language_ground_truth"])
        .rename(columns={"language_response": "language"})
        .assign(response_language="English")
    )
    non_english_gt = (
        results[(results["language"] != "en") & (results["language"] == results["response_language"])]
        .merge(ground_truth, on=["question_key", "language"], suffixes=("_response", "_ground_truth"))
        .assign(response_language="Native")
    )
    hypothesis_3_data = pd.concat([us_results_with_other_gt, non_english_gt])[
        ["model_name", "response_pro_score", "ground_truth_pro_score", "response_language", "cntry_an"]
    ]
    return pd.concat(
        [
            sample_and_calculate(
                hypothesis_3_data,
                ["model_name", "response_language", "cntry_an"],
                "response_pro_score",
                "ground_truth_pro_score",
                N,
                metric_func,
            )
            for _ in tqdm(range(num_rounds))
        ]
    )


# Plotting Functions
def plot_american_superiority_bias(data: pd.DataFrame, filename: str):
    sns.catplot(data=data, x="model_name", y="metric_value", hue="cntry_an", kind="bar")
    plt.title("English Language Alignment")
    plt.ylabel("Alignment Score")
    plt.xticks(rotation=45)
    plt.savefig(PLOT_DIR / filename, bbox_inches="tight")
    plt.close()


def plot_cross_linguistic_transfer(data: pd.DataFrame, filename: str):
    data_long = data.melt(id_vars=["model_name", "cntry_an"], var_name="Type", value_name="Alignment Score")
    sns.catplot(data=data_long, x="cntry_an", y="Alignment Score", hue="Type", col="model_name", kind="bar", col_wrap=3)
    plt.xticks(rotation=45)
    plt.savefig(PLOT_DIR / filename, bbox_inches="tight")
    plt.close()


def plot_monocultural_alignment(data: pd.DataFrame, filename: str):
    g = sns.catplot(
        data=data.rename(columns={"response_language": "Language"}),
        x="cntry_an",
        y="metric_value",
        hue="Language",
        col="model_name",
        kind="bar",
        col_wrap=3,
    )
    g.set_axis_labels("Country", "Alignment to Country's Values")
    plt.savefig(PLOT_DIR / filename, bbox_inches="tight")
    plt.close()


def _sample_df(data: pd.DataFrame, sample_size: int):
    return data.sample(n=min(sample_size, data.shape[0]), random_state=RANDOM_SEED)


def plot_monocultural_by_family(data: pd.DataFrame, filename: str, sample_size: int = 1000):
    # Data Transformation
    logger.debug(f"Data shape: {data.head()}")
    english_data = _sample_df(data[data["response_language"] == "English"], sample_size=sample_size)
    native_data = _sample_df(data[data["response_language"] == "Native"], sample_size=sample_size)
    transformed_data = english_data.merge(
        native_data, on=["cntry_an", "model_family", "model_name"], suffixes=("_English", "_Native")
    )
    transformed_data = transformed_data.assign(
        difference=transformed_data["metric_value_Native"] - transformed_data["metric_value_English"]
    )
    logger.debug(f"Transformed Shape: {transformed_data.shape}")

    # Create a categorical type for model_family and model_name to control their order
    transformed_data = transform_model_categories(transformed_data)
    logger.debug(f"New transformed data: {transformed_data.head()}")

    g = sns.catplot(
        data=transformed_data,
        x="cntry_an",
        y="difference",
        hue="model_name",
        col="model_family",
        kind="bar",
        height=6,
        aspect=1.5,
        col_wrap=2,
        palette=get_model_color_dict(),
        legend_out=True,
    )

    # Customize
    g.set_axis_labels("Country", "Difference (Native - English)")
    g.figure.suptitle("Monocultural Alignment by Model Family (Survey Hypothetical)", fontsize=16)
    g.figure.subplots_adjust(top=0.9, bottom=0.15)  # Adjusted bottom margin
    # Save the plot
    plt.savefig(PLOT_DIR / filename, bbox_inches="tight")
    plt.close()


def plot_cross_linguistic_transfer_by_family(data: pd.DataFrame, filename: str):
    logger.debug(f"Data outline: {data.head()}")
    logger.debug(f"Data shape: {data.shape}")
    transformed_data = data.assign(Difference=data["Native Alignment"] - data["US Alignment"]).rename(
        {"language_x": "language"}, axis=1
    )

    transformed_data = transform_model_categories(transformed_data)

    g = sns.catplot(
        data=transformed_data,
        x="cntry_an",
        y="Difference",
        hue="model_name",
        col="language",
        row="model_family",
        sharex=False,
        sharey=True,
        kind="bar",
        palette=get_model_color_dict(),
        legend_out=True,
    )
    # Customize
    g.set_axis_labels("Country", "Alignment Difference (Native - US)")
    g.figure.suptitle("Cross-Linguistic Cultural Transfer by Model Family", fontsize=16)
    g.figure.subplots_adjust(top=0.9, bottom=0.15)  # Adjusted bottom margin
    # Save the plot
    plt.savefig(PLOT_DIR / filename, bbox_inches="tight")
    plt.close()


def transform_model_categories(model_data: pd.DataFrame) -> pd.DataFrame:
    model_data_copy = model_data.copy()
    model_data_copy["model_family"] = pd.Categorical(
        model_data_copy["model_family"], categories=MODEL_FAMILIES.keys(), ordered=True
    )
    all_models = [model for models in MODEL_FAMILIES.values() for model in models]
    model_data_copy["model_name"] = pd.Categorical(model_data["model_name"], categories=all_models, ordered=True)
    return model_data_copy


# Main Analysis Pipeline
def run_analysis(
    analysis: Analysis,
    results: pd.DataFrame,
    ground_truth: pd.DataFrame,
    metric_name: str,
    num_rounds: int = NUM_ROUNDS,
):
    analysis_func = analysis.analysis_func
    plot_func = analysis.plot_func
    analysis_results = analysis_func(results, ground_truth, metric_name, num_rounds)
    pl.DataFrame(analysis_results).write_csv(
        OUTPUT_DIR / f"{metric_name}_n{num_rounds}_{analysis.filename.split('.')[0]}.csv"
    )
    if plot_func is not None:
        plot_func(analysis_results, f"{metric_name}_n{num_rounds}_{analysis.filename}")
    if analysis.plot_family_func is not None:
        plot_family_func = analysis.plot_family_func
        analysis_results = analysis_results.merge(results[["model_name", "model_family"]].drop_duplicates(), on="model_name")
        plot_family_func(analysis_results, f"{metric_name}_n{num_rounds}_by_family_{analysis.filename}")
    return analysis_results


def plot_absolute_family_performance(results: pl.DataFrame | pd.DataFrame, filename: str) -> None:
    plot_results = results if isinstance(results, pl.DataFrame) else pl.DataFrame(results)
    families = get_families_df()
    plot_results = plot_results.cast({"model_name": get_model_enum()}).join(families, left_on="model_name", right_on="model")
    color_dict = get_model_color_dict()
    BASELINE_FAMILY = "Baseline"
    plt.clf()
    for i, family in tqdm(enumerate(plot_results["family"].unique())):
        filtered_data = plot_results.filter(pl.col("family").is_in([BASELINE_FAMILY, family]))
        plt.clf()
        sns.catplot(
            data=filtered_data.sort("model_name").cast({"model_name": pl.String}),
            x="cntry_an",
            y="metric_value",
            hue="model_name",
            palette=color_dict,
            col="response_language",
            kind="bar",
            sharex=False,
            sharey=True,
            legend=True,
        )
        plt.savefig(PLOT_DIR / f"{family}_{filename}", bbox_inches="tight")


def main(
    threshold: float | None = None,
    baselines: list[str] | None = None,
    metrics: list[str] | None = None,
    choose_analyses: list[str] | None = None,
    num_rounds: int | None = None,
) -> None:
    results = load_results()
    # TESTING: Limit to only survey results
    results = results[results["template_type"] == "survey_hypothetical"]
    ground_truth = load_ground_truth()

    # TESTING: Only "extreme" ground truth values
    if threshold is not None:
        extreme_ground_truth = ground_truth[(ground_truth["ground_truth_pro_score"] - 0.5).abs() > threshold]
    else:
        extreme_ground_truth = ground_truth

    if baselines is None:
        baselines = ["fifty_percent", "uniform_random"]

    logger.debug(f"Extreme ground truth shape: {extreme_ground_truth.shape}")

    # Generate baseline responses
    baseline_responses = generate_baseline_responses(results, baseline_types=baselines)

    # Combine baseline responses with other results
    results_with_baseline = pd.concat([results, baseline_responses], ignore_index=True)

    # Save latest results with baseline as a test
    results_with_baseline.to_csv(OUTPUT_DIR / "latest_results_with_baseline.csv", index=False)

    analyses = [
        Analysis(
            name="American Superiority Bias",
            analysis_func=analyze_american_superiority_bias,
            plot_func=plot_american_superiority_bias,
            plot_family_func=None,
            filename="english_language_alignment.png",
        ),
        Analysis(
            name="Cultural Transfer",
            analysis_func=analyze_cross_linguistic_transfer,
            plot_func=plot_cross_linguistic_transfer,
            plot_family_func=plot_cross_linguistic_transfer_by_family,
            filename="cross_linguistic_cultural_transfer.png",
        ),
        Analysis(
            name="Monocultural Alignment",
            analysis_func=analyze_monocultural_alignment,
            plot_func=plot_monocultural_alignment,
            plot_family_func=plot_monocultural_by_family,
            filename="monocultural_alignment.png",
        ),
        Analysis(
            name="Absolute Performance",
            analysis_func=analyze_absolute_performance,
            plot_func=plot_absolute_performance,
            plot_family_func=plot_absolute_family_performance,
            filename="extreme_absolute_performance.png",
        ),
        Analysis(
            name="Relative Performance",
            analysis_func=analyze_relative_performance,
            plot_func=plot_relative_performance,
            plot_family_func=plot_relative_family_performance,
            filename="extreme_relative_performance.png",
        ),
        Analysis(
            name="All Combinations",
            analysis_func=analyze_relative_performance_all_combinations,
            plot_func=None,
            plot_family_func=None,
            filename="extreme_relative_all_combinations.png",
        ),
        Analysis(
            name="All Countries",
            analysis_func=analyze_absolute_all_combinations,
            plot_func=None,
            plot_family_func=None,
            filename="extreme_total_gt_alignment.png",
        ),
    ]

    for metric in metrics:
        for analysis in analyses:
            if analysis.name not in choose_analyses:
                logger.info(f"Skipping analysis for {analysis.name}")
                continue
            logger.info(f"Running analysis for {analysis.name} using {metric} metric")
            run_analysis(
                analysis,
                results_with_baseline,
                extreme_ground_truth,
                metric_name=metric,
                num_rounds=num_rounds,
            )

    logger.info("Analysis complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse hypotheses using different metrics", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["mean_alignment"],
        choices=list(METRICS),
        help="The metrics to use for analysis",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        nargs="+",
        default=["fifty_percent", "uniform_random"],
        choices=list(BASELINE_GENERATORS),
        help="The types of baselines to use (can specify multiple)",
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=NUM_ROUNDS,
        help="The number of rounds to sample and calculate the metric for each analysis",
    )
    parser.add_argument(
        "--analyses",
        type=str,
        nargs="+",
        default=[
            "American Superiority Bias",
            "Cultural Transfer",
            "Monocultural Alignment",
            "Absolute Performance",
            "All Combinations",
        ],
        choices=[
            "American Superiority Bias",
            "Cultural Transfer",
            "Monocultural Alignment",
            "Absolute Performance",
            "Relative Performance",
            "All Combinations",
            "All Countries",
        ],
    )
    parser.add_argument("--threshold", type=float, default=None, help="Threshold for extreme values")

    arguments = parser.parse_args()
    main(
        threshold=arguments.threshold,
        baselines=arguments.baselines,
        metrics=arguments.metrics,
        choose_analyses=arguments.analyses,
        num_rounds=arguments.num_rounds,
    )
