from typing import Literal, TypedDict

from instructor import OpenAISchema
from pydantic import Field


class ProcessOpinions(OpenAISchema):
    """
    Processes a list of 'pro' and 'con' opinions from evaluations of the responses and topic.
    If the text already summarizes the opinions, it should return a representative list of 'pro' and 'con' opinions.
    I.e., if the text says 'three were in favour and one were against', it should return ['pro', 'pro', 'pro', 'con'].
    If it is not possible to determine the opinions, return ['NULL'].
    """

    language: Literal["en", "da", "nl", "pt", "NULL"] = Field(..., description="Language of the response (ISO 639-1 code)")
    opinions: list[Literal["pro", "con", "NULL"]] = Field(..., description="Objective classifications")


class AnalysisResponse(TypedDict):
    custom_id: str
    language: str | None
    opinions: list[Literal["pro", "con", "NULL"]] | None
