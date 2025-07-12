import polars as pl
from statsmodels.regression.linear_model import RegressionResults

from multicultural_alignment.directories import SUPPLEMENTARY_DIR

REGEX_PATTERNS = {
    "language": r"language\[([^\]]+)\]",
    "model_name": r"model_name\[T?\.?([^\]]+)\]",
    "family": r"family\[([^\]]+)\]",
    "gt_group": r"gt_group\[([^\]]+)\]",
}


def save_regression_results(results: RegressionResults, regression_type: str, rq_method: str = "multilingual") -> None:
    supplementary_path = SUPPLEMENTARY_DIR / "regressions"
    supplementary_path.mkdir(parents=True, exist_ok=True)
    output_path = supplementary_path / f"{rq_method}_regression_results_{regression_type}.txt"
    output_path.write_text(results.summary().as_text())


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
