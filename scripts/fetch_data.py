import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

from trial_conversion_model.data import RAW_DATA

QUERY = "SELECT * FROM ml.trial_snapshot_latest"
DB_HOST = "dpg-d35pib0dl3ps7394mc4g-a.oregon-postgres.render.com"
DB_NAME = "beam_neb0"
DB_USER = "students"


def fetch(out_path: Path = RAW_DATA) -> None:
    """Materialize the training extract into data/01_raw.

    Training always runs from this file, never from the live table, so the
    training data cannot shift between runs.
    """
    load_dotenv()
    password = os.environ["BEAM_DB_PASSWORD"]
    engine = create_engine(
        f"postgresql://{DB_USER}:{password}@{DB_HOST}:5432/{DB_NAME}"
    )
    df = pd.read_sql(QUERY, engine)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    fetch()
