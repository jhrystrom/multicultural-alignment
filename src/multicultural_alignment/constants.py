from pathlib import Path


def _get_root_dir():
    return Path(__file__).parent.parent


# Directories
PLOT_DIR = _get_root_dir() / "plots"
OUTPUT_DIR = _get_root_dir() / "output"
DATA_DIR = _get_root_dir() / "data"


COUNTRY_LANG_MAP = {
    "US": "en",
    "DK": "da",
    "NL": "nl",
    "BR": "pt",
    "NZ": "en",
    "PT": "pt",
    "CA": "en",
    "AU": "en",
    "GB": "en",
    "KE": "en",
    "ZW": "en",
}

LANGUAGE_MAP = {
    "nl": "Dutch",
    "en": "English",
    "da": "Danish",
    "pt": "Portuguese",
}

COUNTRY_MAP = {
    "global": "Global",
    "CA": "Canada",
    "NIR": "Northern Ireland",
    "NZ": "New Zealand",
    "KE": "Kenya",
    "GB": "Great Britain",
    "PT": "Portugal",
    "AU": "Australia",
    "NL": "Netherlands",
    "US": "United States",
    "BR": "Brazil",
    "NG": "Nigeria",
    "DK": "Denmark",
    "SG": "Singapore",
}
