import os
from pathlib import Path

import boto3
import pandas as pd
from dotenv import load_dotenv

from trial_conversion_model.cohorts import pull_newest_cohort
from trial_conversion_model.predict import load_model, predict_proba

PREDICTIONS_DIR = Path("data/04_predictions")

if __name__ == "__main__":
    load_dotenv()
    cohort_path = pull_newest_cohort()
    cohort = pd.read_parquet(cohort_path)

    # The lifecycle team's daily ranked list: highest probability first,
    # scored by the same predict function the API serves.
    scored = cohort.assign(conversion_probability=predict_proba(load_model(), cohort))
    scored = scored.sort_values("conversion_probability", ascending=False)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (
        PREDICTIONS_DIR / f"scored_{cohort_path.stem.removeprefix('cohort_')}.csv"
    )
    scored.to_csv(out_path, index=False)
    s3 = boto3.client("s3")
    s3.upload_file(
        str(out_path), os.environ["S3_BUCKET"], f"predictions/{out_path.name}"
    )
    print(f"scored {len(scored)} trials from {cohort_path.name} -> {out_path} and S3")
