# The course generator takes a naive datetime at midnight, so these do too.
# ruff: noqa: DTZ001
import importlib.util
from datetime import datetime
from pathlib import Path

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "backfill.py"
spec = importlib.util.spec_from_file_location("backfill", SCRIPT)
backfill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backfill)

COLUMNS = [
    "trial_id",
    "user_id",
    "trial_started_at",
    "country",
    "device_type",
    "sessions_day1",
    "sessions_day2",
    "sessions_day3",
    "listen_sessions_3d",
    "total_minutes_3d",
]


def test_cohort_has_the_course_columns():
    cohort = backfill.make_cohort(datetime(2026, 9, 20), 120, drift=False)
    assert len(cohort) == 120
    assert list(cohort.columns) == COLUMNS


def test_same_date_gives_the_same_cohort():
    first = backfill.make_cohort(datetime(2026, 9, 20), 120, drift=False)
    second = backfill.make_cohort(datetime(2026, 9, 20), 120, drift=False)
    assert first.equals(second)


def test_cohort_matches_the_course_published_file():
    # If this fails, the copy in backfill.py has drifted from the course generator.
    cohort = backfill.make_cohort(datetime(2026, 9, 20), 120, drift=False)
    assert cohort["trial_id"].iloc[0] == 1820000
    assert cohort["trial_id"].iloc[-1] == 1820119
    assert cohort["sessions_day1"].sum() == 583
    assert cohort["sessions_day3"].sum() == 235
    assert cohort["total_minutes_3d"].sum() == 11497.0
    assert cohort["trial_started_at"].iloc[0] == pd.Timestamp("2026-09-17 09:55:11")


def test_verdict_is_clean_and_marked_as_backfill():
    verdict = backfill.ok_verdict(datetime(2026, 9, 20))
    assert verdict["drift"] is False
    assert verdict["drifted_share"] in {0.0, 0.125, 0.25}
    assert verdict["source"] == "backfill"
    assert verdict["name"] == "drift_cohort_2026-09-20"
