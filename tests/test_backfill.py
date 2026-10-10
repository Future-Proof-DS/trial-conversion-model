# The course generator takes a naive datetime at midnight, so these do too.
# ruff: noqa: DTZ001
import importlib.util
from datetime import date, datetime, timedelta
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


def test_empty_bucket_fills_the_whole_month():
    days = backfill.days_to_fill(set(), date(2026, 10, 18))
    assert len(days) == 30
    assert days[0] == date(2026, 9, 18)
    assert days[-1] == date(2026, 10, 17)


def test_week5_verdicts_are_real():
    existing = {
        "predictions/scored_2026-10-17.csv",
        "monitoring/drift_cohort_2026-09-13.json",
        "monitoring/drift_cohort_2026-09-13.html",
        "monitoring/drift_cohort_2026-10-10.json",
        "monitoring/drift_cohort_2026-10-10.html",
    }
    days = backfill.days_to_fill(existing, date(2026, 10, 18))
    assert days[0] == date(2026, 9, 18)
    assert days[-1] == date(2026, 10, 9)
    assert len(days) == 22


def test_real_day_at_the_window_start_leaves_nothing_to_fill():
    existing = {"predictions/scored_2026-09-18.csv"}
    assert backfill.days_to_fill(existing, date(2026, 10, 18)) == []


def test_rerun_sees_its_own_days_as_fillable():
    filled = [date(2026, 9, 18) + timedelta(days=i) for i in range(22)]
    existing = {f"predictions/scored_{d}.csv" for d in filled}
    existing |= {f"monitoring/drift_cohort_{d}.json" for d in filled}
    existing |= {
        "predictions/scored_2026-10-17.csv",
        "monitoring/drift_cohort_2026-10-10.json",
        "monitoring/drift_cohort_2026-10-10.html",
    }
    assert backfill.days_to_fill(existing, date(2026, 10, 18)) == filled


def test_half_written_day_is_reached():
    existing = {"monitoring/drift_cohort_2026-09-20.json"}
    days = backfill.days_to_fill(existing, date(2026, 10, 18))
    assert len(days) == 30
    assert days[0] == date(2026, 9, 18)
    assert days[-1] == date(2026, 10, 17)
