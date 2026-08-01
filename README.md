# trial-conversion-model

Beam's trial conversion model as a structured, runnable project. It predicts, from a trial's first 3 days of behavior, whether the trial will convert to a paid plan at the end of day 14. The business context lives in the model plan document that accompanied the original notebook.

## Setup

Install the dependencies and the package:

```
uv sync
```

The training data is not committed to the repository. Pull the extract from the analytics database into `data/raw/` with your database credentials:

```
\copy (SELECT * FROM ml.trial_snapshot_latest) TO 'data/raw/trial_snapshot.csv' WITH CSV HEADER
```

## Train

```
uv run scripts/train.py
```

This loads the extract, trains the model, and writes `models/model.json` and `models/metrics.json`.

## Layout

- `src/trial_conversion_model/`: the package. `data.py` loads and prepares the extract; `train.py` trains, evaluates, and saves the model.
- `scripts/`: thin entry points that call into the package. Production runs these; the logic stays importable and testable in `src/`.
- `notebooks/`: exploration only. Notebooks import from the package; no pipeline logic lives here.
- `data/raw/`: the immutable training extract (never committed, never modified).
- `models/`: trained model artifacts and metrics (not committed).
