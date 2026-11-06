
from typing import Literal

from pydantic import BaseModel, Field, field_validator


Sentiment = Literal["negative", "neutral", "positive"]


class PredictBatchRequest(BaseModel):
    comments: list[str] = Field(
        min_length=1,
        max_length=100,
        description="Liste de 1 à 100 commentaires"
    )

    @field_validator("comments")
    @classmethod
    def validate_comments(cls, values):

        cleaned = [comment.strip() for comment in values]

        if any(not comment for comment in cleaned):
            raise ValueError("Les commentaires ne doivent pas être vides.")

        if any(len(comment) > 10000 for comment in cleaned):
            raise ValueError("Un commentaire dépasse 10000 caractères.")

        return cleaned


class PredictionItem(BaseModel):
    comment: str
    label: int
    sentiment: Sentiment
    confidence: float
    probabilities: dict[Sentiment, float]


class BatchStatistics(BaseModel):
    total: int
    counts: dict[Sentiment, int]
    percentages: dict[Sentiment, float]


class PredictBatchResponse(BaseModel):
    predictions: list[PredictionItem]
    statistics: BatchStatistics
    inference_time_ms: float
