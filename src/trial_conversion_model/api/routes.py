import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from fastapi import APIRouter

from trial_conversion_model.api.schemas import PredictionRequest, PredictionResponse
from trial_conversion_model.predict import load_model, predict_proba

logger = logging.getLogger(__name__)

router = APIRouter()

# Load the model once, when the service starts, not on every request.
model = load_model()

PREDICTION_LOG = Path("logs/predictions.jsonl")


def log_prediction(features: dict, probability: float, band: str) -> None:
    """Append one scored request to the prediction log, one JSON per line.

    The human-readable log line is for debugging; this file is for
    machines: monitoring reads it to see what the model has been asked
    and what it answered.
    """
    PREDICTION_LOG.parent.mkdir(exist_ok=True)
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        **features,
        "conversion_probability": probability,
        "conversion_band": band,
    }
    with PREDICTION_LOG.open("a") as f:
        f.write(json.dumps(record) + "\n")


def to_band(probability: float) -> str:
    """Turn a raw probability into a label a human can act on."""
    if probability < 0.33:
        return "low"
    if probability < 0.66:
        return "medium"
    return "high"


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "0.3.0"}


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict conversion probability for a single live trial."""
    row = pd.DataFrame([request.model_dump()])
    probability = round(float(predict_proba(model, row).iloc[0]), 4)
    band = to_band(probability)
    logger.info(
        "prediction | %s -> probability=%.4f band=%s",
        request.model_dump(),
        probability,
        band,
    )
    log_prediction(request.model_dump(), probability, band)
    return PredictionResponse(conversion_probability=probability, conversion_band=band)
