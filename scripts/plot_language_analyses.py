import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

import multicultural_alignment.models
from multicultural_alignment.constants import COUNTRY_MAP, LANGUAGE_MAP, OUTPUT_DIR, PLOT_DIR
from multicultural_alignment.plot import get_model_color_dict

sns.set_theme(style="whitegrid", font_scale=1.7)

GT_ALWAYS = {"global", "US", "en"}
TARGET_FAMILIES = ["openai", "gemma", "olmo"]


def get_model_filter(family: str, language: str = "en") -> pl.Expr:
    return pl.col("family") == family


country_languages = pl.read_csv(OUTPUT_DIR / "country_languages.csv")
experiment_data = pl.read_csv(OUTPUT_DIR / "spearman_n100_extreme_total_gt_alignment.csv")
experiment_data = multicultural_alignment.models.add_families_df(experiment_data)

baseline_data = experiment_data.filter(pl.col("model_name").cast(pl.String).str.starts_with("Baseline_uniform"))
experiment_data = experiment_data.join(country_languages, left_on="gt_group", right_on="cntry_AN", how="left")


language_filter = (
    (pl.col("response_language") == pl.col("lnge_iso"))
    | (pl.col("response_language") == pl.col("gt_group"))
    | (pl.col("gt_group").is_in(GT_ALWAYS))
)
experiment_data = experiment_data.filter(language_filter)

for (family, metric), group_data in experiment_data.group_by("family", "metric"):
    if family in {"mistral", "Baseline"}:
        continue
    if metric in {"pearson"}:
        continue
    baseline_data = experiment_data.filter(pl.col("model_name").cast(pl.String).str.starts_with("Baseline_uniform"))
    joined_gemma = pl.concat([group_data, baseline_data]).join(
        country_languages, left_on="gt_group", right_on="cntry_AN", how="left"
    )
    responses_languages = joined_gemma["response_language"].unique()
    common_filter = (
        pl.col("gt_group").is_in(responses_languages)
        | pl.col("lnge_iso").is_in(responses_languages)
        | (pl.col("gt_type") == "global")
    )
    filtered_gemma = joined_gemma.filter(common_filter)


common_filter = (
    pl.col("gt_group").is_in(responses_languages)
    | pl.col("lnge_iso").is_in(responses_languages)
    | (pl.col("gt_type") == "global")
)
model_filter = pl.col("family").is_in(TARGET_FAMILIES) | pl.col("model_name").cast(str).str.starts_with("Baseline_uniform")
filtered_plot_data = (
    experiment_data.filter(common_filter & model_filter)
    .rename({"response_language": "language", "metric_value": "alignment", "gt_type": "level"})
    .with_columns(
        pl.col("language").replace(LANGUAGE_MAP),
        pl.col("gt_group").replace(LANGUAGE_MAP).replace(COUNTRY_MAP),
        pl.col("level").str.to_titlecase(),
    )
)

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
).set_titles(
    "{row_name} | {col_name}",
)
sns.move_legend(plot, loc="upper center", bbox_to_anchor=(0.42, 0.05), ncol=4, title=None)
for ax in plot.axes.flat:
    ax.set_ylabel("")
plot.figure.supylabel("cultural alignment", x=0.035)
plot.set(xlabel=None)
plt.savefig(PLOT_DIR / f"monolingual-{metric}-gt_alignment.png", bbox_inches="tight")

# If that doesn't work, you can try falling back to multiple fonts:
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
