import json

from dotenv import load_dotenv
from joblib import Memory
from openai import OpenAI

from multicultural_alignment.schemas import ProcessOpinions

load_dotenv(override=True)
memory = Memory(".cache", verbose=0)

OPINION_SYSTEM_MSG = """
You are an expert in analyzing opinions in text. 
You always provide an objective and correct classification based on the information provided.
Your task is to label each individual statement as either `pro`, `con`, or `NULL` based on 
whether they are closest to the pro or con statement provided.
""".strip()


@memory.cache
def process_opinions(text: str, system_message: str = OPINION_SYSTEM_MSG, model_name: str = "gpt-4o") -> ProcessOpinions:
    client = OpenAI()
    raw_results = client.responses.parse(
        model=model_name,
        input=[{"role": "system", "content": system_message}, {"role": "user", "content": text}],
        text_format=ProcessOpinions,
        temperature=0.0,
    )
    new_llm_opinions = json.loads(raw_results.output[0].content[0].text)
    return ProcessOpinions.model_validate(new_llm_opinions)
