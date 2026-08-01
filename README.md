# trial-conversion-model

Beam's trial conversion model as a structured, runnable project. It predicts, from a trial's first 3 days of behavior, whether the trial will convert to a paid plan at the end of day 14. The business context lives in the model plan document that accompanied the original notebook.

## Setup

Install the dependencies and the package:

```
uv sync
```

The training data is not committed to the repository. Pull the extract from the analytics database into `data/01_raw/` with your database credentials:

```
\copy (SELECT * FROM ml.trial_snapshot_latest) TO 'data/01_raw/trial_snapshot.csv' WITH CSV HEADER
```

## Train

```
uv run scripts/train.py
```

This builds the processed training table from the raw extract, trains the model, and writes `models/model.json` and `models/metrics.json`.

## Layout

- `src/trial_conversion_model/`: the package. `data.py` loads the pipeline's inputs; `features.py` derives the model features from the snapshot's base aggregates and writes the processed training table; `train.py` trains, evaluates, and saves the model.
- `scripts/`: thin entry points that call into the package. Production runs these; the logic stays importable and testable in `src/`.
- `notebooks/`: exploration only. Notebooks import from the package; no pipeline logic lives here.
- `data/01_raw/`: the immutable extract as pulled from the database (never committed, never modified).
- `data/03_processed/`: the model-ready training table written by the pipeline (never committed; `02_interim` is reserved for multi-step pipelines this project does not need).
- `models/`: trained model artifacts and metrics (not committed).
