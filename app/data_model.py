
from datetime import datetime

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    pickup_datetime: datetime
    PULocationID: int = Field(gt=0)
    DOLocationID: int = Field(gt=0)
    passenger_count: float | None = Field(default=None, ge=0)


class PredictionResponse(BaseModel):
    predicted_duration: float
