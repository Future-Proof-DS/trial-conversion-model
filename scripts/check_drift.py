import os
import sys
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from trial_conversion_model.cohorts import newest_local_cohort
from trial_conversion_model.monitoring import (
    load_reference,
    publish_report,
    run_drift_check,
)


def alert(message: str) -> None:
    """Tell a human. A cron job nobody watches needs a way to interrupt one."""
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook:
        print("SLACK_WEBHOOK_URL not set; alert printed only")
        return
    requests.post(webhook, json={"text": message}, timeout=10)


if __name__ == "__main__":
    load_dotenv()
    if len(sys.argv) > 2:
        sys.exit("usage: uv run scripts/check_drift.py [cohort file]")
    cohort_path = Path(sys.argv[1]) if len(sys.argv) == 2 else newest_local_cohort()
    current = pd.read_parquet(cohort_path)
    snapshot = run_drift_check(current, load_reference())
    result = publish_report(snapshot, f"drift_{cohort_path.stem}")
    verdict = "DRIFT" if result["drift"] else "OK"
    print(
        f"{verdict}: {result['drifted_share']:.0%} of features drifted ({cohort_path.name})"
    )
    if result["drift"]:
        alert(
            f"Drift alert: {result['drifted_share']:.0%} of features drifted "
            f"in {cohort_path.name}. Latest report is in S3 under monitoring/."
        )
