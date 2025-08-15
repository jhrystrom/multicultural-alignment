from collections import OrderedDict

import seaborn as sns

from multicultural_alignment.models import MODEL_FAMILIES


def _generate_color_palettes(ordered_models: OrderedDict[str, list[str]]) -> dict[str, dict[str, tuple]]:
    palette_names = ["spring", "winter", "summer", "autumn"]
    palettes: dict[str, dict[str, tuple]] = {}
    for (family, models), palette_name in zip(ordered_models.items(), palette_names):
        n_models = len(models)
        family_palette = sns.color_palette(palette_name, n_colors=n_models)
        palettes[family] = dict(zip(models, family_palette))
    return palettes


def get_model_color_dict() -> dict[str, tuple]:
    color_palettes = _generate_color_palettes(MODEL_FAMILIES)
    return {model: color for family_palette in color_palettes.values() for model, color in family_palette.items()}


def rename_color_dict(new_names: dict[str, str], color_dict: dict[str, tuple]) -> dict[str, tuple]:
    return {new_names.get(model, model): color for model, color in color_dict.items()}


def _last_value(dictionary: dict) -> tuple:
    return list(dictionary.values())[-1]


def get_family_color_dict() -> dict[str, tuple]:
    color_palettes = _generate_color_palettes(MODEL_FAMILIES)
    return {family: _last_value(family_palette) for family, family_palette in color_palettes.items()}
