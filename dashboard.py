"""One screen showing the whole system's state, read from S3.

Run locally with: uv run streamlit run dashboard.py

The dashboard is a reader, not a component: it displays what the
scheduled jobs produced (scored cohorts under predictions/, drift
verdicts under monitoring/, the model under models/) and never computes
anything itself.

It answers three questions, top to bottom: is the system alive and what
is it running, has anything moved over the past month, and what did the
lifecycle team receive today.
"""

import io
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import altair as alt
import boto3
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
BUCKET = os.environ["S3_BUCKET"]
MODEL_KEY = "models/model.json"
METRICS_PATH = Path("models/metrics.json")

# The same number the drift check uses (DRIFT_SHARE in monitoring.py):
# a day counts as drifted when at least half of the features drifted.
DRIFT_THRESHOLD = 0.5

# A cohort arrives every day, so a list older than this means the score
# job stopped running. Cron does not tell you that; this page does.
STALE_AFTER_DAYS = 2

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


@st.cache_data(ttl=300)
def last_modified(key: str) -> datetime | None:
    # When the object was last written, or None if it is not there.
    s3 = boto3.client("s3")
    found = s3.list_objects_v2(Bucket=BUCKET, Prefix=key).get("Contents", [])
    return found[0]["LastModified"] if found else None


# Load everything the page shows, once: the scored cohorts from predictions/
# and the drift verdicts from monitoring/.
# A month or two is what the charts need, and the bucket only grows.
scored_keys = [k for k in list_keys("predictions/") if k.endswith(".csv")][-60:]
verdict_keys = [k for k in list_keys("monitoring/") if k.endswith(".json")]

# One row per day: the average probability and the middle half of the
# cohort around it, because a model can answer differently without its
# average moving. Days with no file are filled in as empty rows, so a
# missing day is a gap in the chart, not a point on it.
probabilities = [
    pd.read_csv(io.BytesIO(read_object(k)), usecols=["conversion_probability"])[
        "conversion_probability"
    ]
    for k in scored_keys
]
by_day = pd.DataFrame(
    {
        "day": pd.to_datetime(
            [Path(k).stem.removeprefix("scored_") for k in scored_keys]
        ),
        "mean_probability": [p.mean() for p in probabilities],
        "p25": [p.quantile(0.25) for p in probabilities],
        "p75": [p.quantile(0.75) for p in probabilities],
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

# Status row: is the system alive, what does it think of the world, and
# which model is running? Fed by the newest scored cohort, the verdicts,
# models/metrics.json, and the model file's date in S3.
left, middle, right = st.columns(3)

# Every tile is a value and one pill. The pill is a metric "delta" with its
# colour and arrow switched off: these are facts, not trends.
if latest is not None:
    latest_day = by_day.index[-1]
    # The course dates its cohorts in UTC, so the age is measured in UTC too.
    age = (datetime.now(UTC).date() - latest_day.date()).days
    when = {0: "today", 1: "yesterday"}.get(age, f"{age} days ago")
    left.metric(
        "Latest scored cohort",
        f"{latest_day:%Y-%m-%d}",
        f"{int(by_day['trials'].iloc[-1])} trials, scored {when}",
        delta_color="off",
        delta_arrow="off",
    )
    if age >= STALE_AFTER_DAYS:
        left.warning(
            f"Nothing scored since {latest_day:%Y-%m-%d}. "
            "The score job has stopped running; check logs/cron.log on the server."
        )
else:
    left.metric("Latest scored cohort", "none yet")

if not verdicts.empty:
    last = verdicts.iloc[-1]
    middle.metric(
        "Latest drift verdict",
        "DRIFT" if last["drift"] else "OK",
        f"{last['drifted_share']:.0%} of features",
        delta_color="off",
        delta_arrow="off",
    )
else:
    middle.metric("Latest drift verdict", "none yet")

if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text())
    published = last_modified(MODEL_KEY)
    right.metric(
        "Model test AUC",
        f"{metrics['test_auc']:.4f}",
        f"published {published:%Y-%m-%d}" if published else "not in S3",
        delta_color="off",
        delta_arrow="off",
    )
else:
    right.metric("Model test AUC", "no local model")

st.divider()

# Over time: one point per scored cohort, one bar per drift verdict.
predictions_col, drift_col = st.columns(2)

with predictions_col:
    st.subheader("Predictions over time")
    if not by_day.empty:
        days = by_day.reset_index()
        # The band is where the middle half of each day's trials landed; the
        # line is the average. The whole band moving is what prediction drift
        # looks like.
        band = (
            alt.Chart(days)
            .mark_area(opacity=0.2)
            .encode(x=alt.X("day:T", title=None), y="p25:Q", y2="p75:Q")
        )
        mean = (
            alt.Chart(days)
            .mark_line()
            .encode(
                x="day:T",
                y=alt.Y(
                    "mean_probability:Q",
                    title="conversion probability",
                    scale=alt.Scale(domain=[0, 1]),
                ),
            )
        )
        st.altair_chart(band + mean, width="stretch")
        st.caption(
            "Average conversion probability of each day's cohort, with the band "
            "holding the middle half of its trials. A move here means the model "
            "is answering differently."
        )
    else:
        st.write("No scored cohorts in S3 yet. The daily score job writes them.")

with drift_col:
    st.subheader("Drift over time")
    if not verdicts.empty:
        days = verdicts.reset_index()
        # Bars, because the share is a count out of eight features; a line
        # would imply something continuous. A bar that crosses the dashed
        # threshold is a day the check said DRIFT, and it turns red.
        bars = (
            alt.Chart(days)
            .mark_bar()
            .encode(
                x=alt.X("day:T", title=None),
                y=alt.Y(
                    "drifted_share:Q",
                    title="features drifted",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format="%"),
                ),
                color=alt.condition(
                    "datum.drift", alt.value("#ff4b4b"), alt.value("#83c9ff")
                ),
            )
        )
        threshold = (
            alt.Chart(pd.DataFrame({"threshold": [DRIFT_THRESHOLD]}))
            .mark_rule(strokeDash=[6, 4], color="#ff4b4b")
            .encode(y="threshold:Q")
        )
        st.altair_chart(bars + threshold, width="stretch")
        st.caption(
            "Share of features the drift check flagged each day. "
            "Half or more counts as drifted, and that is the dashed line."
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
