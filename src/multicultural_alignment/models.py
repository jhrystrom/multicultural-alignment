from collections import OrderedDict

import polars as pl

MODEL_FAMILIES = OrderedDict(
    [
        ("Baseline", ["Baseline_uniform_random", "Baseline_fifty_percent", "Baseline_perfect"]),
        ("openai", ["gpt-3.5-turbo-0125", "gpt-4-turbo-2024-04-09", "gpt-4o"]),
        ("gemma", ["gemma-2-2b-it", "gemma-2-9b-it", "gemma-2-27b-it"]),
        ("olmo", ["OLMo-2-1124-7B-Instruct", "OLMo-2-1124-13B-Instruct", "OLMo-2-0325-32B-Instruct"]),
    ]
)


def add_families_df(df: pl.DataFrame, model_col: str = "model_name") -> pl.DataFrame:
    families_df = get_families_df()
    return df.cast({model_col: get_model_enum()}).join(families_df, left_on=model_col, right_on="model", how="left")


def get_families_df() -> pl.DataFrame:
    model_list = [model for models in MODEL_FAMILIES.values() for model in models]
    return pl.DataFrame(
        {
            "model": model_list,
            "family": [family for family, models in MODEL_FAMILIES.items() for _ in models],
        }
    ).cast({"model": pl.Enum(categories=model_list)})


def get_model_enum() -> pl.Enum:
    model_list = [model for models in MODEL_FAMILIES.values() for model in models]
    return pl.Enum(categories=model_list)


def get_model_name(model_path: str) -> str:
    return model_path.split("/")[-1]


def get_model_family(model: str) -> str:
    for family, models in MODEL_FAMILIES.items():
        if get_model_name(model) in models:
            return family
    raise ValueError(f"Model {model} not found in any model family")
