"""One screen showing the whole system's state, read from S3.

Run locally with: uv run streamlit run dashboard.py

The dashboard is a reader, not a component: it displays what the
scheduled jobs produced (scored cohorts under predictions/, drift
verdicts under monitoring/) and never computes anything itself.
"""

import io
import json
import os
from pathlib import Path

import boto3
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
BUCKET = os.environ["S3_BUCKET"]
METRICS_PATH = Path("models/metrics.json")

st.set_page_config(page_title="Trial conversion model", layout="wide")


@st.cache_data(ttl=300)
def list_keys(prefix: str) -> list[str]:
    s3 = boto3.client("s3")
    listing = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    return sorted(o["Key"] for o in listing.get("Contents", []))


@st.cache_data(ttl=300)
def read_object(key: str) -> bytes:
    s3 = boto3.client("s3")
    return s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()


st.title("Trial conversion model")

prediction_keys = [k for k in list_keys("predictions/") if k.endswith(".csv")]
verdict_keys = [k for k in list_keys("monitoring/") if k.endswith(".json")]

# Status row: is the system alive, and what does it think of the world?
left, middle, right = st.columns(3)

if prediction_keys:
    latest_key = prediction_keys[-1]
    scored = pd.read_csv(io.BytesIO(read_object(latest_key)))
    left.metric("Latest scored cohort", Path(latest_key).stem.removeprefix("scored_"))
    left.caption(f"{len(scored)} trials")
else:
    scored = None
    left.metric("Latest scored cohort", "none yet")

if verdict_keys:
    verdict = json.loads(read_object(verdict_keys[-1]))
    label = "DRIFT" if verdict["drift"] else "OK"
    middle.metric(
        "Latest drift verdict", label, f"{verdict['drifted_share']:.0%} of features"
    )
else:
    middle.metric("Latest drift verdict", "none yet")

if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text())
    right.metric("Model test AUC", f"{metrics['test_auc']:.4f}")
    right.caption(f"trained on {metrics['n_train']} trials")
else:
    right.metric("Model test AUC", "no local model")

st.divider()

predictions_col, drift_col = st.columns(2)

with predictions_col:
    st.subheader("Latest predictions")
    if scored is not None:
        st.bar_chart(
            scored["conversion_probability"].value_counts(bins=20, sort=False),
            x_label="conversion probability",
            y_label="trials",
        )
        st.caption("Top of the ranked list")
        st.dataframe(
            scored.head(10)[
                ["trial_id", "country", "device_type", "conversion_probability"]
            ],
            hide_index=True,
        )
    else:
        st.write("No scored cohorts in S3 yet. The daily score job writes them.")

with drift_col:
    st.subheader("Drift history")
    if verdict_keys:
        history = pd.DataFrame(
            [json.loads(read_object(k)) for k in verdict_keys]
        ).assign(report=[Path(k).stem for k in verdict_keys])
        history["verdict"] = history["drift"].map({True: "DRIFT", False: "OK"})
        st.dataframe(history[["report", "drifted_share", "verdict"]], hide_index=True)
    else:
        st.write("No drift reports in S3 yet. The daily drift check writes them.")
