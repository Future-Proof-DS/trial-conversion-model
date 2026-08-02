from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from trial_conversion_model.features import CATEGORICAL, FEATURES, add_features

MODEL_PATH = Path("models/model.json")


def load_model(path: Path = MODEL_PATH) -> XGBClassifier:
    """Load the trained model from its native XGBoost artifact."""
    model = XGBClassifier()
    model.load_model(path)
    return model


def predict_proba(model: XGBClassifier, aggregates: dict) -> float:
    """Score one trial from its base aggregates.

    The row goes through the same add_features as training. A single row
    can only carry one country and one device, so its dummy columns are
    reindexed against the model's training columns; the categories the
    row does not have become explicit zeros.
    """
    df = add_features(pd.DataFrame([aggregates]))
    row = pd.get_dummies(df[FEATURES], columns=CATEGORICAL)
    row = row.reindex(columns=model.get_booster().feature_names, fill_value=0)
    return float(model.predict_proba(row)[0, 1])
