import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResults

from multicultural_alignment.constants import LANGUAGE_MAP, OUTPUT_DIR, PLOT_DIR
from multicultural_alignment.models import add_families_df
from multicultural_alignment.plot import get_model_color_dict

language_pattern = r"language\[([^\]]+)\]"
model_pattern = r"model_name\[T?\.?([^\]]+)\]"


def get_us_data() -> pl.DataFrame:
    experiment_data = pl.read_csv(OUTPUT_DIR / "spearman_n100_extreme_total_gt_alignment.csv").rename(
        {"response_language": "language"}
    )
    wvs_country = pl.read_csv(OUTPUT_DIR / "ground_truth_every_country.csv")
    country_langs = wvs_country.unique(["cntry_an", "lnge_iso"]).select(["cntry_an", "lnge_iso"])
    country_corrs_us = (
        wvs_country.join(wvs_country, on=["question_key"])
        .group_by("cntry_an", "cntry_an_right")
        .agg(pl.corr("pro_score", "pro_score_right").alias("us_corr"))
        .filter(pl.col("cntry_an_right") == "US")
        .drop("cntry_an_right")
    )
    return (
        add_families_df(experiment_data)
        .filter(pl.col("gt_type") == "country")
        .join(country_langs, left_on="gt_group", right_on="cntry_an")
        .with_columns((pl.col("language") == pl.col("lnge_iso")).alias("native"), (pl.col("gt_group") == "US").alias("us"))
        .filter(pl.col("native") | pl.col("us"))
        .filter(pl.col("model_name") != "Baseline_fifty_percent")
        .filter(pl.col("family") != "mistral")
        .sort("family")
        .cast({"model_name": pl.String})
        .join(country_corrs_us, left_on="gt_group", right_on="cntry_an")
        .unique()
    )


def extract_results_df(results: RegressionResults) -> pl.DataFrame:
    params = results.params
    conf_int = results.conf_int()
    return pl.DataFrame(
        {
            "term": params.index.tolist(),
            "coefficient": params.values,
            "std_err": results.bse.values,
            "p_value": results.pvalues.values,
            "conf_int_lower": conf_int[0].values,
            "conf_int_upper": conf_int[1].values,
        }
    )


def run_us_regression() -> RegressionResults:
    us_data = get_us_data()
    us_model = smf.ols(formula="metric_value ~  us*model_name:language", data=us_data.to_pandas())
    return us_model.fit()


def get_coefficients(us_results) -> pl.DataFrame:
    results_pl = extract_results_df(us_results)
    significant = (
        results_pl.filter(pl.col("term").str.contains(":model_name"))
        .with_columns(
            pl.col("term").str.extract(language_pattern).alias("language"),
            pl.col("term").str.extract(model_pattern).alias("model_name"),
        )
        .drop("term")
    )
    return add_families_df(significant)


def get_confidence_data(significant: pl.DataFrame) -> pl.DataFrame:
    significant_lower = significant.select(
        pl.col("conf_int_lower").alias("coefficient"),
        "model_name",
        "language",
    )
    significant_higher = significant.select(
        pl.col("conf_int_upper").alias("coefficient"),
        "model_name",
        "language",
    )
    significant_mid = significant.select(
        pl.col("coefficient"),
        "model_name",
        "language",
    )
    return pl.concat([significant_lower, significant_higher, significant_mid])


def plot_us_centric_bias(plot_data: pl.DataFrame) -> None:
    sns.barplot(
        data=plot_data.sort("model_name").cast({"model_name": str}).with_columns(pl.col("language").replace(LANGUAGE_MAP)),
        x="language",
        y="coefficient",
        hue="model_name",
        palette=get_model_color_dict(),
        errorbar=("pi", 100),
    )
    plt.ylabel("$\\beta_{BiasUS}$")
    plt.xlabel(None)
    # legend below plot in two columns
    plt.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.43))
    plt.savefig(PLOT_DIR / "us_bias_coefficients.png", bbox_inches="tight")


def main():
    sns.set_theme(style="whitegrid", font_scale=1.5)
    us_results = run_us_regression()
    plot_data = get_confidence_data(get_coefficients(us_results))
    plot_us_centric_bias(plot_data)


if __name__ == "__main__":
    main()
