import sys
from pathlib import Path

import pandas as pd

from trial_conversion_model.monitoring import (
    load_reference,
    publish_report,
    run_drift_check,
)

if __name__ == "__main__":
    cohort_path = Path(sys.argv[1])
    current = pd.read_parquet(cohort_path)
    snapshot = run_drift_check(current, load_reference())
    result = publish_report(snapshot, f"drift_{cohort_path.stem}")
    verdict = "DRIFT" if result["drift"] else "OK"
    print(f"{verdict}: {result['drifted_share']:.0%} of features drifted ({cohort_path.name})")
