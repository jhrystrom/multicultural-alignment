import itertools

import numpy as np
from loguru import logger


def binary_fraction_to_frequencies(frac: float) -> tuple[int, int]:
    """Convert a fraction to frequencies."""
    if not 0 <= frac <= 1:
        raise ValueError("Fraction must be between 0 and 1.")
    return int(frac * 100), int((1 - frac) * 100)


def controversy(*group_frequencies) -> float:
    k = len(group_frequencies)
    if k == 1:
        return 0.0
    if np.all(np.array(group_frequencies) == 0):
        return np.nan
    scaling_factor = k / (k - 1)
    total_product = sum(2 * a * b for a, b in itertools.combinations(group_frequencies, 2))
    population_squared = sum(group_frequencies) ** 2
    return (total_product / population_squared) * scaling_factor


def pro_score(num_pro: int, num_con: int) -> float:
    """Calculate the fraction of positive responses"""
    try:
        return num_pro / (num_pro + num_con)
    except ZeroDivisionError:
        logger.debug(f"ZeroDivision with: num_pro: {num_pro}, num_con: {num_con}")
        return np.nan


def likert_pro_score(likert_col: np.ndarray, max_value: int = 5) -> float:
    """Calculate the fraction of positive responses to negative responses
    Notes:
    - Negative numbers are missing values
    for five points:
    - 1 & 2 are strongly agree and agree
    - 4 & 5 are disagree and strongly disagree
    for four points:
    - no neutral point
    """
    likert_positives = likert_col[likert_col > 0]
    likert_mid = (max_value + 1) / 2
    num_con = (likert_positives > likert_mid).sum()
    num_pro = (likert_positives < likert_mid).sum()
    pro_score = num_pro / (num_pro + num_con)
    if max_value == 10:
        logger.warning("10 points means higher is positive!")
        pro_score = 1 - pro_score
    return pro_score
