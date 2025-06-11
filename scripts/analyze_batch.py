import json
from pathlib import Path

import pandas as pd

import multicultural_alignment.score as score
from multicultural_alignment import fileio
from multicultural_alignment.constants import OUTPUT_DIR
from multicultural_alignment.data import get_family_data, get_family_string
from multicultural_alignment.models import MODEL_FAMILIES, get_model_family, get_model_name
from multicultural_alignment.schemas import AnalysisResponse


def sort_by_custom_id(dict_list):
    return sorted(dict_list, key=lambda d: int(d["custom_id"].split("-")[-1]))


def parse_response(response: dict) -> AnalysisResponse:
    choice = response["response"]["body"]["choices"][0]
    arguments = choice["message"]["tool_calls"][0]["function"]["arguments"]
    try:
        return AnalysisResponse(**json.loads(arguments), custom_id=response["custom_id"])
    except json.decoder.JSONDecodeError:
        return AnalysisResponse(language=None, opinions=None, custom_id=response["custom_id"])


def sort_and_parse(responses: list[dict]) -> list[AnalysisResponse]:
    return [parse_response(response) for response in sort_by_custom_id(responses)]


def clean_list(lst: list[str]) -> list[str]:
    if lst is None:
        return []
    return [item.lower() for item in lst]


def score_response(response: AnalysisResponse) -> float:
    pro_count = clean_list(response["opinions"]).count("pro")
    con_count = clean_list(response["opinions"]).count("con")
    return score.pro_score(num_pro=pro_count, num_con=con_count)


def create_score_df(raw_responses: list[dict]) -> pd.DataFrame:
    parsed_responses = sort_and_parse(raw_responses)
    scores = [score_response(response) for response in parsed_responses]
    return pd.DataFrame(scores, columns=["response_pro_score"]).assign(
        response_language=[response.get("language", "NULL") for response in parsed_responses],
        response_pro_con=[";".join(clean_list(response.get("opinions", []))) for response in parsed_responses],
        custom_id=[response["custom_id"] for response in parsed_responses],
    )


def create_full_truth_df(prompt_responses: pd.DataFrame, analysis_responses: list[dict]) -> pd.DataFrame:
    score_df = create_score_df(analysis_responses)
    assert prompt_responses.shape[0] == score_df.shape[0], "Mismatch between response and score count"
    return pd.concat([prompt_responses, score_df], axis=1)


def _load_openai_mistral() -> pd.DataFrame:
    openai_responses = load_model_family("openai")
    mistral_responses = load_model_family("mistral")
    return pd.concat([openai_responses, mistral_responses], ignore_index=True).reset_index(drop=True)


def load_model_family(model_family: str, output_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    model_pattern = "all_prompts_responses_{model_name}.csv"
    family_pattern = "all_prompts_responses_{model_family}*.csv"
    if model_family not in MODEL_FAMILIES:
        raise ValueError(f"Invalid model family: {model_family}")
    # Special case: openai is stored in a single file...
    if model_family == "gemma":
        return pd.concat(
            [
                pd.read_csv(output_dir / model_pattern.format(model_name=model_name))
                for model_name in MODEL_FAMILIES[model_family]
            ],
            ignore_index=True,
        )
    return pd.read_csv(fileio.find_latest_path(family_pattern.format(model_family=model_family), directory=output_dir))


def get_data_pipeline():
    relevant_families = ["olmo", "gemma", "openai"]
    family_data = get_family_data(families=relevant_families)
    family_analysis_responses = fileio.read_jsonl(OUTPUT_DIR / f"{get_family_string(relevant_families)}-raw-responses.jsonl")
    full_df = create_full_truth_df(family_data.to_pandas(), family_analysis_responses)

    # openai_mistral = _load_openai_mistral()
    # openai_mistral_responses = fileio.read_jsonl(OUTPUT_DIR / "all-languages-raw-responses.jsonl")
    # full_openai_mistral = create_full_truth_df(openai_mistral, openai_mistral_responses)

    # # GPT-4o was done later, so it's not included in openai
    # gpt4o_prompt_responses = pd.read_csv(fileio.find_latest_path("gpt-4o_responses.csv", directory=OUTPUT_DIR))
    # gpt4o_analysis_responses = fileio.read_jsonl(OUTPUT_DIR / "gpt-4o_responses_analysis.jsonl")
    # full_gpt4o = create_full_truth_df(gpt4o_prompt_responses, gpt4o_analysis_responses)

    # # Gemma responses
    # gemma_prompt_responses = load_model_family("gemma")
    # gemma_analysis_responses = fileio.read_jsonl(OUTPUT_DIR / "gemma-raw-responses.jsonl")
    # full_gemma = create_full_truth_df(gemma_prompt_responses, gemma_analysis_responses)

    # full_df = pd.concat([full_openai_mistral, full_gpt4o, full_gemma], ignore_index=True).reset_index(drop=True)
    # assert full_df.loc[0, "model"].startswith("gpt-4-turbo"), "Model name does not start with gpt-4-turbo"
    full_df.to_csv("data/lang_model_responses_raw.csv", index=False)

    ground_truth = pd.read_csv(OUTPUT_DIR / "ground_truth.csv").rename(columns={"pro_score": "ground_truth_pro_score"})
    all_results = (
        full_df.drop(columns="controversy")
        .merge(ground_truth, on=["question_key", "language"], how="left")
        .dropna(subset=["response_pro_score", "ground_truth_pro_score", "model", "language"])
    )[(full_df["language"] == full_df["response_language"])]
    all_results = all_results.assign(
        model_family=all_results["model"].apply(get_model_family), model_name=all_results["model"].apply(get_model_name)
    )
    all_results["model_name"].unique()

    columns_to_save = [
        "model_name",
        "model_family",
        "template_type",
        "language",
        "response_language",
        "question_key",
        "response_pro_score",
        "ground_truth_pro_score",
    ]
    all_results[columns_to_save].to_csv(OUTPUT_DIR / "all_lang_model_responses_with_scores.csv", index=False)


if __name__ == "__main__":
    get_data_pipeline()
