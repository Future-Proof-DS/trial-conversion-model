# trial-conversion-model

Predicts, from a trial's first 3 days of behavior, whether the trial will convert to a paid plan at the end of day 14, so the growth team can reach trials that look unlikely to convert while they are still live. The business case and rollout plan are in the model plan document.

## Setup

Install the dependencies and the package:

```
uv sync
```

The training data is not committed to the repository. Copy `.env.example` to `.env` and replace the password placeholder with the one from the course's Tools & Setup lesson, then materialize the extract:

```
uv run scripts/fetch_data.py
```

This writes `data/01_raw/trial_snapshot.csv`. Training always runs from that file, never from the live table, so the training data cannot shift between runs.

## Train

```
uv run scripts/train.py
```

This builds the processed training table from the raw extract, trains the model, and writes `models/model.json` and `models/metrics.json`.

## Serve

The model is served over HTTP so the teams that need scores can get them on demand. Run the service locally:

```
uv run uvicorn trial_conversion_model.api.main:app --reload
```

Or build and run it as a container, which is how it ships:

```
docker build -t trial-conversion-model .
docker run -p 8000:8000 trial-conversion-model
```

`POST /predict` takes one trial's first-3-day base aggregates and returns its conversion probability plus a low/medium/high band; `GET /health` reports service status. Interactive docs live at `/docs` while the service runs.

## Test

```
uv run pytest
```

## Monitor

Labels for a live trial arrive 11+ days after the prediction, so input drift is the earliest signal that the world has shifted under the model. The service logs every scored request to `logs/predictions.jsonl`, and the drift check compares a current cohort file against the training extract:

```
uv run scripts/check_drift.py [cohort file]
```

This writes an HTML report (for eyes) and a JSON verdict (for machines) to `monitoring/` and to the S3 bucket, and prints OK or DRIFT. With no argument it checks the newest pulled cohort, which is the form the daily cron job uses; on a DRIFT verdict it posts an alert to the Slack webhook in `SLACK_WEBHOOK_URL`.

## Score

The lifecycle team's daily ranked list. A new day-3 cohort file lands in the course's public bucket every day; scoring pulls the newest one, runs it through the same `predict` the API serves, and writes the trials ranked by conversion probability to `data/04_predictions/` and to the S3 bucket:

```
uv run scripts/score.py
```

On the server, cron runs this and the drift check daily; nothing about either script knows whether a human or a schedule invoked it.

## Dashboard

One screen for the whole system, read from S3: the latest scored cohort, the latest drift verdict, and the model's metrics.

```
uv run streamlit run dashboard.py
```

## Layout

- `src/trial_conversion_model/`: the package. `data.py` acquires the extract from the database and loads the pipeline's inputs; `features.py` derives the model features from the snapshot's base aggregates and writes the processed training table; `train.py` trains, evaluates, and saves the model; `predict.py` scores trials from their base aggregates; `cohorts.py` pulls day-3 cohort files from the course's public bucket; `api/` is the FastAPI service (`main.py` builds the app, `routes.py` holds the endpoints, `schemas.py` defines the request and response shapes).
- `scripts/`: thin entry points that call into the package (`fetch_data.py` materializes the extract, `train.py` builds the training table and trains, `score.py` scores the newest cohort, `check_drift.py` runs the drift check). Production runs these; the logic stays importable and testable in `src/`.
- `tests/`: pytest checks for the feature logic and the API contract.
- `monitoring.py` (in the package): the drift check; `scripts/check_drift.py` runs it against a cohort file.
- `dashboard.py`: the Streamlit dashboard; a reader of what the scheduled jobs produced, never a computer of anything itself.
- `notebooks/`: exploration only. Notebooks import from the package; no pipeline logic lives here.
- `data/01_raw/`: the immutable extract as pulled from the database, and under `cohorts/` the day-3 cohort files pulled from the course bucket (never committed, never modified).
- `data/02_interim/`: reserved for intermediate outputs in multi-step pipelines; this project goes straight from raw to processed, so it stays empty.
- `data/03_processed/`: the model-ready training table written by the pipeline (never committed).
- `data/04_predictions/`: the scored, ranked cohort lists written by `scripts/score.py` (never committed).
- `models/`: trained model artifacts and metrics (not committed).
