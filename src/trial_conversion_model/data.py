from pathlib import Path

import pandas as pd

RAW_DATA = Path("data/raw/trial_snapshot.csv")
CATEGORICAL = ["country", "device_type"]
TARGET = "converted"
ID_COLUMNS = ["trial_id", "user_id", "snapshot_date", "trial_started_at"]


def load_raw(path: Path = RAW_DATA) -> pd.DataFrame:
    """Load the trial snapshot extract pulled from ml.trial_snapshot_latest."""
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Turn the raw extract into a model-ready feature matrix and target."""
    X = df.drop(columns=ID_COLUMNS + [TARGET])
    X = pd.get_dummies(X, columns=CATEGORICAL)
    y = df[TARGET]
    return X, y
