# FlowCast

Short-horizon traffic prediction for the Northline Corridor. Covers 25 road segments in 30-minute windows, January
to May 2025. Forecasts traffic volume and average speed 30 to 120 minutes ahead per segment. Derives travel time,
classifies congestion, scores accident risk, attaches a confidence estimate to each output. Results are displayed on
a Streamlit dashboard.

Built to `docs/FlowCast_PRD.docx` v1.0. That file is the PRD this project is measured against: requirements, the nine
dashboard views, and the success criteria in its section 3.3.

## New here? Read in this order

Each file opens with a plain-language guide: purpose, steps in execution order, concepts required.

| # | Open | You'll learn |
|---|---|---|
| 1 | `notebooks/01_eda.ipynb` | Data structure and every defect in it, examined one question at a time |
| 2 | `notebooks/02_cleaning.ipynb` | Fix applied to each defect, demonstrated slowly on a few rows |
| 3 | `notebooks/03_modelling.ipynb` | Regression vs classification, score interpretation, reason the winning model won |
| 4 | `run_pipeline.py` | End-to-end project execution, step by step |
| 5 | `src/ingest.py` → `clean.py` → `features.py` → `ml_models.py` → `dl_model.py` | Production code behind the notebooks, in execution order |
| 6 | `dashboard/app.py` | Presentation of results to end users |

Notebooks read saved results only: no retraining, open and run in seconds. If `data/processed/` is empty, run
`run_pipeline.py` once first.

## Setup (one command)

Every command on this page is written as it appears, from the folder that *contains* `flowcast/` — that is where you
land after `git clone https://github.com/<you>/flowcast.git`. Inside `flowcast/` itself, drop the `flowcast/` prefix:
`pip install -r requirements.txt`, `pytest tests`, `streamlit run dashboard/app.py`. Both work; `run_pipeline.py`
resolves its own paths, so it runs from any directory.

```
pip install -r flowcast/requirements.txt
```

(CPU-only PyTorch index is configured inside `requirements.txt`.)

Requires Python 3.13. Built, tested and pinned on 3.13. PRD names 3.11: upgraded to 3.13, listed as a deviation in
`reports/final_report.md`.

## Reproduce every result from the raw files (one command)

```
python flowcast/run_pipeline.py
```

| Option | What it does |
|---|---|
| *(none)* | Runs M1 → M2 → M3 → M5 → M6 from `data/raw/` |
| `--skip-dl` | Stops after the classical models |
| `--from ml` | Retrains only, reuses the saved feature table |

Runs from any working directory. All randomness is seeded, all settings live in `config.yaml`: a rerun reproduces
the reported metrics. Each stage logs to `reports/pipeline.log`.

Then run the tests and open the dashboard:

```
pytest flowcast/tests
streamlit run flowcast/dashboard/app.py
```

Style check: PEP 8 with the 99-column limit PEP 8 permits teams to adopt.
`pycodestyle --max-line-length=99 --ignore=E203,W503 flowcast/src flowcast/dashboard flowcast/tests`

## Layout (PRD section 16)

```
flowcast/
├── data/
│   ├── raw/            source tables as delivered (read-only)
│   ├── interim/        cleaned, pre-merge tables (traffic, weather, calendar) + rows quarantined at ingestion
│   └── processed/      analysis-ready, versioned dataset: clean + feature tables, manifest.json
├── notebooks/          01_eda, 02_cleaning, 03_modelling
├── src/
│   ├── ingest.py       M1  load, schema, types, ranges, quarantine
│   ├── clean.py        M2  dedup, repairs, missing windows, imputation, merge, quality report
│   ├── features.py     M3  time, calendar, weather, lag/rolling features; horizon shifting
│   ├── ml_models.py    M5  CV-tuned classical models, scratch NumPy linear regression
│   ├── dl_model.py     M6  LSTM / BiLSTM, multi-task, from scratch
│   ├── evaluate.py         metrics, thresholds, error analysis, model cards
│   └── utils.py            config, paths, logging, time split, hashing
├── models/             trained models + model_cards.md / .json
├── reports/            data_quality.md, figures/, final_report.md, scoreboard.json, dl_result.json, test predictions
├── dashboard/app.py    M7
├── tests/
├── docs/                FlowCast_PRD.docx — the PRD this build follows
├── requirements.txt
├── config.yaml
├── run_pipeline.py     the one-command entry point (PRD 3.3 / NFR reproducibility)
└── README.md
```

Matches the PRD section 16 tree. Two additions: `run_pipeline.py` (PRD requires a one-command rerun but its tree does
not name the script) and `docs/` (holds the PRD itself, so a reader of this repository can open the requirements the
build answers to). Dashboard upload sample is generated from `data/raw/` on demand, not stored as a separate file.

## Where to read the results

- `reports/final_report.md` — Methods, underlying mathematics, results, acceptance check, recommendations
- `reports/data_quality.md` — Every raw-data defect and its resolution (generated)
- `models/model_cards.md` — Per saved model: training window, features, settings, test scores (generated)
- `notebooks/` — EDA, cleaning walkthrough, model comparison, error analysis
