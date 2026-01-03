# app/schemas.py

from pydantic import BaseModel

class PredictionResponse(BaseModel):
    result: str
    confidence: float
    threshold: float
    raw_score: float
    model_version: str