from pathlib import Path

import pandas as pd

RAW_DATA = Path("data/01_raw/trial_snapshot.csv")
PROCESSED_DATA = Path("data/03_processed/training_data.csv")


def load_raw(path: Path = RAW_DATA) -> pd.DataFrame:
    """Load the trial snapshot extract pulled from ml.trial_snapshot_latest."""
    return pd.read_csv(path)


def load_processed(path: Path = PROCESSED_DATA) -> pd.DataFrame:
    """Load the model-ready training table produced by features.build_training_data."""
    return pd.read_csv(path)
