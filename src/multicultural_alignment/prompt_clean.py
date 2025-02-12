import re

import pandas as pd


def replace_braces(series: pd.Series) -> pd.Series:
    """Replace content inside curly braces with '{topic}'."""
    return series.str.replace(r"\{.*?\}", "{topic}", regex=True)


def extract_backtick_content(text: str) -> str:
    """Extract content between triple backticks, ignoring the first line within the backticks."""
    match = re.search(r"```.*?\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text  # Return the original text if no pattern is matched.
