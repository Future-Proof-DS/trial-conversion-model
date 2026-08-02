import json
from pathlib import Path

import mlflow
import mlflow.xgboost
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from trial_conversion_model.data import load_processed
from trial_conversion_model.features import TARGET

MODEL_DIR = Path("models")
TEST_SIZE = 0.25
RANDOM_STATE = 42
EXPERIMENT = "trial-conversion"

# One dict, used for training AND logged to MLflow, so the recorded
# params are provably the ones the model was trained with.
PARAMS = {
    "n_estimators": 400,
    "max_depth": 3,
    "learning_rate": 0.05,
    "min_child_weight": 8,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "eval_metric": "auc",
}


def train(model_dir: Path = MODEL_DIR) -> dict:
    """Train the trial conversion model from the processed training table."""
    load_dotenv()
    mlflow.set_experiment(EXPERIMENT)

    table = load_processed()
    X = table.drop(columns=[TARGET])
    y = table[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    with mlflow.start_run():
        mlflow.log_params(PARAMS)

        model = XGBClassifier(**PARAMS)
        model.fit(X_train, y_train)

        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        mlflow.log_metric("test_auc", auc)
        mlflow.xgboost.log_model(model, name="model")

    model_dir.mkdir(exist_ok=True)
    model.save_model(model_dir / "model.json")
    metrics = {
        "test_auc": round(float(auc), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features": list(X.columns),
    }
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
