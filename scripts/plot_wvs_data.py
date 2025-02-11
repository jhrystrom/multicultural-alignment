import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from multicultural_alignment.constants import COUNTRY_MAP, OUTPUT_DIR, PLOT_DIR


def main() -> None:
    sns.set_theme(font_scale=1.6, style="white")
    wvs_data_country = pl.read_csv(OUTPUT_DIR / "ground_truth_every_country.csv").rename({"lnge_iso": "language"})
    wvs_data_global = pl.read_csv(OUTPUT_DIR / "ground_truth_global.csv").with_columns(
        pl.lit("global").alias("cntry_an"), pl.lit("global").alias("language")
    )
    wvs_data = pl.concat([wvs_data_country, wvs_data_global.select(wvs_data_country.columns)]).with_columns(
        pl.col("cntry_an").replace(COUNTRY_MAP)
    )
    wvs_language_data = pl.read_csv(OUTPUT_DIR / "ground_truth_per_language.csv")

    plot_country_correlations(wvs_data)
    plot_country_distributions(wvs_data)
    plot_language_correlations(wvs_language_data)


def plot_language_correlations(wvs_language_data: pl.DataFrame):
    pivoted = wvs_language_data.sort(by="lnge_iso").pivot(index="question_key", on="lnge_iso", values="pro_score")
    corr_matrix = pivoted.to_pandas().corr(numeric_only=True, min_periods=1)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )
    plt.title("Correlation Heatmap of Pro Scores by Language")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "wvs_language_correlations.png", bbox_inches="tight")


def plot_country_correlations(wvs_data: pl.DataFrame):
    pivoted = wvs_data.pivot(index="question_key", on="cntry_an", values="pro_score")
    corr_matrix = pivoted.to_pandas().corr(numeric_only=True, min_periods=1)
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    # Create a heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "country_correlations.png", bbox_inches="tight")
    # clear the plot
    plt.clf()


def plot_country_distributions(wvs_data) -> None:
    sns.catplot(data=wvs_data, x="cntry_an", y="pro_score", kind="violin", col="language", sharex=False)
    plt.savefig(PLOT_DIR / "country_distributions.png", bbox_inches="tight")
    plt.clf()


if __name__ == "__main__":
    main()
