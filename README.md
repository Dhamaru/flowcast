# FlowCast

Short-horizon traffic prediction for the Northline Corridor: 25 road segments, 30-minute windows, January to May 2025.
For each segment it forecasts traffic volume and average speed 30 to 120 minutes ahead, derives travel time, classifies
congestion and scores accident risk, all with a confidence estimate, and shows them on a Streamlit dashboard.

Built to `Imported/FlowCast_PRD.docx` v1.0.

## Setup (one command)

```
pip install -r flowcast/requirements.txt
```

(The CPU-only PyTorch index is set inside `requirements.txt`.)

Built and tested on Python 3.13. The PRD names 3.11; nothing here relies on a 3.12+ feature, but 3.11 has not been tested.

## Reproduce every result from the raw files (one command)

```
python flowcast/run_pipeline.py
```

| Option | What it does |
|---|---|
| *(none)* | M1 → M2 → M3 → M5 → M6 from `data/raw/` |
| `--skip-dl` | stop after the classical models |
| `--from ml` | retrain only, reusing the saved feature table |

Works from any folder. All randomness is seeded and every setting is in `config.yaml`, so a rerun reproduces the
reported metrics. Every stage logs to `reports/pipeline.log`.

Then run the tests and open the dashboard:

```
pytest flowcast/tests
streamlit run flowcast/dashboard/app.py
```

Style check (PEP 8, with the 99-column limit PEP 8 allows teams to agree on):
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
├── requirements.txt
├── config.yaml
├── run_pipeline.py     the one-command entry point (PRD 3.3 / NFR reproducibility)
└── README.md
```

Everything matches the PRD's section 16 tree. `run_pipeline.py` is the only addition: the PRD requires a one-command
rerun but its tree doesn't name the script. The dashboard's upload sample is built from `data/raw/` on demand rather
than stored as a separate file.

## Where to read the results

- `reports/final_report.md` — methods, the mathematics behind them, results, acceptance check, recommendations
- `reports/data_quality.md` — every defect in the raw data and how it was resolved (generated)
- `models/model_cards.md` — each saved model: training window, features, settings, test scores (generated)
- `notebooks/` — EDA, the cleaning walkthrough, model comparison and error analysis
