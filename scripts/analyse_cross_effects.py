import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from tqdm import tqdm

from multicultural_alignment.constants import PLOT_DIR
from multicultural_alignment.data import load_results
from multicultural_alignment.models import add_families_df, get_model_enum
from multicultural_alignment.plot import get_model_color_dict

if __name__ == "__main__":
    results = load_results(as_polars=True).filter(pl.col("template_type") == "survey_hypothetical")

    self_joined = results.join(results, on=["question_key", "model_name"]).with_columns(
        (pl.col("language") == pl.col("language_right")).alias("same_language"),
    )

    N_BOOTSTRAPS = 100
    all_dfs = []
    for _ in tqdm(range(N_BOOTSTRAPS)):
        corr_result = (
            self_joined.sample(fraction=1.0, with_replacement=True)
            .group_by("model_name", "same_language")
            .agg(pl.corr(pl.col("response_pro_score"), pl.col("response_pro_score_right")))
            .pivot(on=["same_language"], index="model_name")
        ).select("model_name", "true", "false")
        all_dfs.append(corr_result)

    combined_analysis = (
        add_families_df(pl.concat(all_dfs))
        .with_columns((pl.col("true") - pl.col("false")).alias("Consistency Gap"))
        .with_columns(pl.col("model_name").cast(get_model_enum()))
        .sort("model_name")
        .cast({"model_name": pl.String})
    )

    sns.set_theme(style="whitegrid", font_scale=1.7)
    sns.barplot(
        combined_analysis,
        y="model_name",
        hue="model_name",
        x="Consistency Gap",
        palette=get_model_color_dict(),
        legend=False,
    )
    plt.ylabel("")
    plt.xlabel("Consistency Gap (Within - Between)")

    plt.savefig(PLOT_DIR / "cross_effects.png", bbox_inches="tight")
    plt.clf()
