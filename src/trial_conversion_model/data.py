import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

RAW_DATA = Path("data/01_raw/trial_snapshot.csv")
PROCESSED_DATA = Path("data/03_processed/training_data.csv")

QUERY = "SELECT * FROM ml.trial_snapshot_latest"


def fetch(out_path: Path = RAW_DATA) -> None:
    """Materialize the training extract into data/01_raw.

    Training always runs from this file, never from the live table, so the
    training data cannot shift between runs. All connection details come
    from the environment; code never knows which database it is pointed at.
    """
    load_dotenv()
    engine = create_engine(
        "postgresql://{user}:{password}@{host}:{port}/{name}".format(
            user=os.environ["DB_USER"],
            password=os.environ["DB_PASSWORD"],
            host=os.environ["DB_HOST"],
            port=os.environ["DB_PORT"],
            name=os.environ["DB_NAME"],
        )
    )
    df = pd.read_sql(QUERY, engine)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"wrote {len(df)} rows to {out_path}")


def load_raw(path: Path = RAW_DATA) -> pd.DataFrame:
    """Load the trial snapshot extract pulled from ml.trial_snapshot_latest."""
    return pd.read_csv(path)


def load_processed(path: Path = PROCESSED_DATA) -> pd.DataFrame:
    """Load the model-ready training table produced by features.build_training_data."""
    return pd.read_csv(path)
