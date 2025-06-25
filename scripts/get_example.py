"""
Gets an example of a full prompt and response (used in the paper).
"""

import polars as pl

from multicultural_alignment.directories import DATA_DIR


def get_all_responses() -> pl.DataFrame:
    """
    Get all responses from the output directory.
    """
    return pl.read_csv(DATA_DIR / "lang_model_responses_raw.csv")


if __name__ == "__main__":
    all_responses = get_all_responses()
    sample_response = (
        all_responses.filter(pl.col("response_language") == "en")
        .filter(pl.col("language") == "en")
        .filter(pl.col("response_pro_con").str.contains("pro") & pl.col("response_pro_con").str.contains("con"))
        .sample(n=1, seed=888)
        .to_dicts()[0]
    )
    print(sample_response["response"])
    response_list = sample_response["response"].split("\n\n")
    rating_list = sample_response["response_pro_con"].split(";")
    formatted_list = "\n\n".join(
        [f"{response} [{rating.upper()}]" for response, rating in zip(response_list, rating_list, strict=True)]
    )
    sample_response["response_pro_con"].count("pro")
    sample_response["response_pro_con"].count("con")
    print(formatted_list)
