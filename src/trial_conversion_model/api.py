from fastapi import FastAPI
from pydantic import BaseModel

from trial_conversion_model.predict import load_model, predict_proba

app = FastAPI(title="Beam trial conversion model")
model = load_model()


class TrialAggregates(BaseModel):
    """One trial's first-3-day base aggregates, as the caller knows them."""

    sessions_day1: int
    sessions_day2: int
    sessions_day3: int
    listen_sessions_3d: int
    total_minutes_3d: float
    country: str
    device_type: str


class Prediction(BaseModel):
    conversion_probability: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(trial: TrialAggregates) -> Prediction:
    proba = predict_proba(model, trial.model_dump())
    return Prediction(conversion_probability=round(proba, 4))
