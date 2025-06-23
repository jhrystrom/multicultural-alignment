import polars as pl
from statsmodels.regression.linear_model import RegressionResults

REGEX_PATTERNS = {
    "language": r"language\[([^\]]+)\]",
    "model_name": r"model_name\[T?\.?([^\]]+)\]",
    "family": r"family\[([^\]]+)\]",
    "gt_group": r"gt_group\[([^\]]+)\]",
}


def extract_term(pattern: str, term_name: str = "term") -> pl.Expr:
    return pl.col(term_name).str.extract(REGEX_PATTERNS[pattern]).alias(pattern)


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
