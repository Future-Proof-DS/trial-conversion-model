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


# Reading from S3. Cached for five minutes so the page does not re-download
# every file each time it redraws.
@st.cache_data(ttl=300)
def list_keys(prefix: str) -> list[str]:
    # A paginator, because a single listing silently stops at 1000 keys.
    s3 = boto3.client("s3")
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=BUCKET, Prefix=prefix
    ):
        keys += [o["Key"] for o in page.get("Contents", [])]
    return sorted(keys)


@st.cache_data(ttl=300)
def read_object(key: str) -> bytes:
    s3 = boto3.client("s3")
    return s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()


# Load everything the page shows, once: the scored cohorts from predictions/
# and the drift verdicts from monitoring/.
# A month or two is what the charts need, and the bucket only grows.
scored_keys = [k for k in list_keys("predictions/") if k.endswith(".csv")][-60:]
verdict_keys = [k for k in list_keys("monitoring/") if k.endswith(".json")]

# One row per day. Days with no file are filled in as empty rows, so a missing
# day is a gap in the line, not a point on it.
probabilities = [
    pd.read_csv(io.BytesIO(read_object(k)), usecols=["conversion_probability"])
    for k in scored_keys
]
by_day = pd.DataFrame(
    {
        "day": pd.to_datetime(
            [Path(k).stem.removeprefix("scored_") for k in scored_keys]
        ),
        "mean_probability": [p["conversion_probability"].mean() for p in probabilities],
        "trials": [len(p) for p in probabilities],
    }
).set_index("day")
if not by_day.empty:
    by_day = by_day.reindex(
        pd.date_range(by_day.index.min(), by_day.index.max(), name="day")
    )

# One row per day for the drift verdicts too, with the same gaps.
verdicts = pd.DataFrame(
    [json.loads(read_object(k)) for k in verdict_keys],
    columns=["drifted_share", "drift"],
)
verdicts["day"] = pd.to_datetime(
    [Path(k).stem.removeprefix("drift_cohort_") for k in verdict_keys]
)
verdicts = verdicts.set_index("day")
if not verdicts.empty:
    verdicts = verdicts.reindex(
        pd.date_range(verdicts.index.min(), verdicts.index.max(), name="day")
    )

# The newest scored cohort in full, for the section at the bottom.
latest = pd.read_csv(io.BytesIO(read_object(scored_keys[-1]))) if scored_keys else None

st.title("Trial conversion model")

# Status row: is the system alive, and what does it think of the world?
# Fed by the newest scored cohort, the newest verdict, and models/metrics.json.
left, middle, right = st.columns(3)

if latest is not None:
    left.metric("Latest scored cohort", f"{by_day.index[-1]:%Y-%m-%d}")
    left.caption(f"{int(by_day['trials'].iloc[-1])} trials")
else:
    left.metric("Latest scored cohort", "none yet")

if not verdicts.empty:
    last = verdicts.iloc[-1]
    label = "DRIFT" if last["drift"] else "OK"
    middle.metric(
        "Latest drift verdict",
        label,
        f"{last['drifted_share']:.0%} of features",
        delta_color="off",
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

# Over time: one point per scored cohort, one bar per drift verdict.
predictions_col, drift_col = st.columns(2)

with predictions_col:
    st.subheader("Predictions over time")
    if not by_day.empty:
        st.line_chart(by_day["mean_probability"])
        st.caption(
            "Average conversion probability of each day's cohort. "
            "A move here means the model is answering differently."
        )
    else:
        st.write("No scored cohorts in S3 yet. The daily score job writes them.")

with drift_col:
    st.subheader("Drift over time")
    if not verdicts.empty:
        # Bars, because the share is a count out of eight features; a line
        # would imply something continuous.
        st.bar_chart(verdicts["drifted_share"])
        st.caption(
            "Share of features the drift check flagged each day. "
            "Half or more counts as drifted."
        )
    else:
        st.write("No drift reports in S3 yet. The daily drift check writes them.")

st.divider()

# Latest cohort: the newest scored file from predictions/, as the lifecycle
# team receives it, least likely to convert first.
st.subheader("Latest cohort")
if latest is not None:
    # Bins are intervals, which the chart prints as raw JSON; label each by
    # its left edge instead.
    counts = latest["conversion_probability"].value_counts(bins=20, sort=False)
    counts.index = [f"{b.left:.2f}" for b in counts.index]
    st.bar_chart(
        counts,
        x_label="conversion probability",
        y_label="trials",
    )
    st.caption("Top of the ranked list")
    st.dataframe(
        latest.head(10)[
            ["trial_id", "country", "device_type", "conversion_probability"]
        ],
        hide_index=True,
    )
else:
    st.write("No scored cohorts in S3 yet. The daily score job writes them.")
