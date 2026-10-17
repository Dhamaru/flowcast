# FlowCast v1.0 — Final Technical Report

Northline Corridor · 25 segments · 30-minute windows · 1 Jan – 31 May 2025
Model version `1.0-5bb519c5` · feature table sha256 `5bb519c5efc9` · raw traffic sha256 `8f793f3643c8`

Every number in this report is read from the pipeline's own outputs (`reports/scoreboard.json`, `reports/dl_result.json`,
`data/processed/manifest.json`) produced by `python flowcast/run_pipeline.py`.

## 1. Summary

FlowCast forecasts traffic volume and average speed per segment 30, 60, 90 and 120 minutes ahead, derives travel time,
classifies next-window congestion, and scores next-window accident risk, each with a confidence estimate, and serves
them on a 9-view Streamlit dashboard. 1 of the four model targets is met;
the reproducibility and dashboard criteria are met.

| PRD 3.3 criterion | Target | Result | Status |
|---|---|---|---|
| Volume forecast accuracy | MAPE ≤ 12% | 9.55% (XGBoost) | **met** |
| Congestion classification | Macro-F1 ≥ 0.80 | 0.769 (XGBoost) | not met |
| Accident-risk ranking | ROC-AUC ≥ 0.75 | 0.617 (Logistic Regression) | not met |
| DL vs ML benchmark | Sequence model beats best classical RMSE | BiLSTM 60.45 vs 58.29 | not met |
| Reproducibility | One command reruns from raw | `python flowcast/run_pipeline.py`, seeded | **met** |
| Dashboard | Nine views on real outputs | 9 views + 4 tools, all reading saved outputs | **met** |

The 3 misses are reported as they are, with what limits them (sections 6, 7 and 11), rather than tuned
until the test set agreed.

## 2. Data and cleaning (M1, M2)

The full defect table is in `reports/data_quality.md`. The decisions that mattered most:

- **Two "impossible" fields had an exact backup in the same row.** `vehicle_count` equals `traffic_volume` on every
  valid row, so 4,587 negative or missing volumes were recovered exactly. Travel
  time is `60 × length / speed` with a fixed length per segment, so 4,579 bad or
  missing speeds were recomputed exactly. Imputation was used only where no backup existed.
- **4,499 sensor windows were absent entirely.** Without rebuilding them, `shift(1)` lag
  features would silently reach back an hour or more across a gap. The table was rebuilt on the full segment × window
  grid (181,200 rows), gaps of up to 2 windows were interpolated, longer ones filled with the
  segment-and-window median, and every rebuilt row is flagged `is_imputed` and never used as a training target.
- **Extremes were flagged, not capped.** After the rule fixes, the remaining statistical outliers
  (26 volume, 463 speed) are mostly real event days
  and incidents; capping them would teach the models that events don't happen.
- Duplicates (1,767) removed, weather labels 13 → 5,
  both date formats parsed to one Asia/Kolkata timestamp, hourly weather broadcast to both 30-minute windows,
  31,123 blank congestion labels derived from V/C, and every stage asserted and logged.

## 3. Features (M3) and the leakage guard

41 features in five groups: time (cyclical hour and weekday, weekend, peak), calendar (holiday, event, roadwork,
holiday × peak, days to nearest event), segment (capacity, length), history (volume and speed lags at 1, 2 and 48
windows; rolling mean/std over 4 and 8 windows; last occupancy, V/C ratio and headroom), and last observed weather.

PRD 9.2 forbids any feature that uses a future window. Applied strictly, that rules out the target window's own V/C
ratio (it contains the volume being predicted) and its weather (not yet observed when the forecast is issued), so both
enter as last-observed values. An earlier build used them directly; congestion scored a macro-F1 of 0.991 and volume an
RMSE of 12.48 until the leak was found. For a forecast h windows ahead, the whole history block is shifted h − 1 more
windows, so the newest information is genuinely h windows old.

## 4. Mathematics, and where it lives

| Concept (PRD 10) | Where in FlowCast |
|---|---|
| Matrices, dot products | `ScratchLinearRegression`: prediction `X @ w + b`; LSTM gates are matrix–vector products |
| Gradient descent, loss | Scratch NumPy linear regression trained by `w -= lr · 2/n · Xᵀ(Xw − y)` (PRD 10.2 build-it-once rule): test RMSE 73.97 after 300 epochs; Adam on MSE + cross-entropy for the LSTM |
| Descriptive statistics | Segment-and-window medians for long gaps; robust z-score (median, MAD) for extremes |
| Standard deviation | Rolling std features; residual spread for confidence ranges |
| Correlation, covariance | EDA (`notebooks/01_eda.ipynb`) |
| Probability | Calibrated accident probability (Platt scaling); class probabilities for congestion confidence |
| Quantiles | 80% prediction ranges = 80th percentile of validation errors per hour of day |
| Feature scaling | z-score for unbounded inputs, Min–Max for bounded ones (LSTM); z-score inside the SVM / linear pipelines |
| Normal distribution | 1.4826 × MAD ≈ σ for normally distributed readings, used in the outlier flag |

## 5. Evaluation protocol

The timeline is split in time order: the first 70% trains, the next 10% validates, the last 20%
(from 2025-05-01 19:00) tests. Never shuffled. Within the training window each candidate was grid-searched
with 3-fold time-series cross-validation (each fold trains on earlier data and validates on later). **Models were
selected on their CV score, not on test.** The validation slice sized the confidence ranges, calibrated accident
probabilities, chose the alert threshold and early-stopped the LSTM. The test window was scored once, at the end.

## 6. Results — classical models (M5)

### Traffic volume, next window
| Model | CV RMSE (± fold std) | Test RMSE | Test MAPE | Test R² | Settings |
|---|---|---|---|---|---|
| XGBoost ✓ | 60.92 ± 3.04 | 58.29 | 9.55% | 0.958 | `{"learning_rate": 0.05, "max_depth": 4, "n_estimators": 300}` |
| Random Forest | 63.44 ± 3.77 | 58.52 | 9.39% | 0.958 | `{"max_depth": 12, "n_estimators": 100}` |
| Linear Regression | 71.96 ± 0.85 | 70.40 | 12.63% | 0.939 | `{}` |
| Decision Tree | 73.25 ± 1.93 | 70.19 | 11.46% | 0.940 | `{"max_depth": 8}` |

Selection: XGBoost had the best CV RMSE and a fold spread comparable to Random Forest, predicts
35,334 test rows in 0.075 s (PRD limit 30 s for the corridor), and
its importance ranking is inspectable. Top features: `traffic_volume_lag1`, `is_peak_period`, `traffic_volume_lag48`, `avg_speed_roll8_mean`, `event_flag`.

| Horizon | RMSE (vehicles) | MAPE | 80% range coverage |
|---|---|---|---|
| 30 min | 58.29 | 9.55% | 80.9% |
| 60 min | 58.43 | 9.64% | 80.3% |
| 90 min | 58.45 | 9.58% | 81.4% |
| 120 min | 58.80 | 9.82% | 81.3% |

Error barely grows from 30 to 120 minutes ahead. The model leans on daily rhythm (same time yesterday, time of day)
more than on the last half hour, and daily rhythm is just as known two hours out.

### Average speed and travel time
| Model | CV RMSE (± fold std) | Test RMSE | Test MAPE | Test R² | Settings |
|---|---|---|---|---|---|
| XGBoost ✓ | 3.84 ± 0.51 | 3.47 | 8.46% | 0.911 | `{"learning_rate": 0.1, "max_depth": 4, "n_estimators": 300}` |
| Random Forest | 3.98 ± 0.58 | 3.51 | 8.47% | 0.909 | `{"max_depth": 18, "n_estimators": 100}` |
| Decision Tree | 4.27 ± 0.64 | 3.70 | 8.82% | 0.899 | `{"max_depth": 8}` |
| Linear Regression | 4.48 ± 0.58 | 3.99 | 9.95% | 0.882 | `{}` |

Travel time is computed as `60 × segment length / predicted speed`, as PRD 11.1 prescribes:

| Horizon | RMSE (min) | MAPE | 80% range coverage |
|---|---|---|---|
| 30 min | 1.09 | 7.80% | 79.5% |
| 60 min | 1.07 | 7.80% | 79.7% |
| 90 min | 1.06 | 7.85% | 79.9% |
| 120 min | 1.07 | 7.84% | 80.3% |

An earlier direct travel-time model reached only about 34% MAPE; deriving it from speed gives 7.8%.

### Congestion level, next window
| Model | CV macro-F1 (± std) | Test macro-F1 | Test accuracy |
|---|---|---|---|
| XGBoost ✓ | 0.751 ± 0.021 | 0.769 | 0.877 |
| Random Forest | 0.746 ± 0.015 | 0.762 | 0.873 |
| Decision Tree | 0.732 ± 0.027 | 0.755 | 0.862 |
| SVM | 0.720 ± 0.033 | 0.709 | 0.845 |

Test recall by class: Free-flow 0.95, Moderate 0.84, Heavy 0.66, Severe 0.61. Confusion matrix (rows actual,
columns predicted):

| | Free-flow | Moderate | Heavy | Severe |
|---|---|---|---|---|
| **Free-flow** | 20,326 | 974 | 0 | 0 |
| **Moderate** | 739 | 7,134 | 625 | 18 |
| **Heavy** | 0 | 682 | 2,311 | 506 |
| **Severe** | 0 | 6 | 783 | 1,230 |

Most errors fall between neighbouring classes (Moderate/Heavy, Heavy/Severe), where the V/C boundary sits close to the
predicted volume's own error. The 0.80 target is not met.

### Accident risk, next window
| Model | CV ROC-AUC (± std) | Test ROC-AUC |
|---|---|---|
| Logistic Regression ✓ | 0.626 ± 0.014 | 0.617 |
| SVM | 0.619 ± 0.011 | 0.612 |
| XGBoost | 0.601 ± 0.017 | 0.581 |
| Random Forest | 0.568 ± 0.025 | 0.575 |

With class weighting, the raw score ranks windows but overstates probability. Platt scaling on the validation slice
turns it into a probability: the test window's mean predicted probability is 0.87%
against an actual rate of 0.97%. At the alert threshold chosen on validation
(0.013) the model raises 77 alerts per 1,000 windows with precision
0.027 and recall 0.212. That is a lift over the base rate, but far from operational
reliability, and every model family landed in the same 0.56–0.63 band. The limit is signal in the data, not tuning.

## 7. Results — deep learning (M6)

A two-layer LSTM with a volume head and a congestion head (multi-task loss: MSE + 0.5 × cross-entropy), trained from
random initialisation with dropout, weight decay, learning-rate reduction on plateau and early stopping. Inputs are the
last N windows of 11 scaled readings per step (traffic_volume, avg_speed, temperature, rainfall, occupancy, visibility, hour_sin, hour_cos, is_peak_period, is_rain, is_weekend). Tuned on validation loss:

| Window N | Hidden size | Validation loss |
|---|---|---|
| 8 | 32 | 0.2047 |
| 8 | 64 | 0.2039 |
| 16 | 32 | 0.2037 |
| 16 | 64 | 0.2021 |

| Variant | Test RMSE | MAPE | R² | Congestion macro-F1 | Epochs |
|---|---|---|---|---|---|
| LSTM | 61.86 | 11.12% | 0.953 | 0.750 | 8 |
| BiLSTM | 60.45 | 10.72% | 0.955 | 0.739 | 9 |

Same test rows as the classical models. Best sequence model: BiLSTM, 3.7%
behind XGBoost on RMSE. Verdict:
the deep model did not earn its added complexity; XGBoost is the production volume model.

## 8. Error analysis (PRD 13.3)

- By hour, volume RMSE is highest at 08:00 (99), 09:00 (94), 17:00 (93) and lowest at
  03:00 (19), 01:00 (19), 02:00 (20). Absolute error follows traffic level, which is why
  the confidence range is sized per hour.
- By weather: Clear 58, Cloudy 57, Fog 50, Overcast 63, Rain 67.
- By day: Fri 62, Mon 63, Sat 46, Sun 44, Thu 67, Tue 62, Wed 61.
- Figures: `reports/figures/error_analysis_traffic_volume.png`, `congestion_confusion.png`, `lstm_training_curve.png`.

## 9. Non-functional requirements

| Requirement | Evidence |
|---|---|
| Performance ≤ 30 s | 35,334 predictions in 0.075 s; dashboard shows per-request time |
| Scalability ≥ 150k rows in 16 GB | 181,200 rows, runs on a laptop CPU |
| Reproducibility | Seeded; one command; config in `config.yaml`; feature table hashed into every model version |
| Maintainability | Modules follow PRD 16; PEP 8 (99-column limit, check command in README); every function documented; `pytest flowcast/tests` |
| Usability ≤ 3 clicks | Role picker lands each role on its view; any view is 2 clicks away |
| Reliability, 0 silent drops | Rejects written with reasons to `data/interim/`; every stage logs to `reports/pipeline.log` |
| Portability | `requirements.txt` pinned; one install command |
| Transparency | Every prediction carries its model version; `models/model_cards.md` per model |

## 10. Deviations from the PRD

- **Python 3.13** instead of 3.11 (the version installed); nothing depends on 3.12+ features, but 3.11 is untested.
- **Weather enters as last observed**, not target-window weather, under the strict leakage reading of 9.2. A weather
  forecast feed would allow target-window weather.
- **Congestion beyond 30 minutes** is derived from predicted volume via the V/C bands; the classifier covers the next
  window, as framed in 11.1.
- **`Overcast`** (358 weather rows) is kept as its own label; it is not in the PRD vocabulary.
- **Model-training interface**: the dashboard's Retrain tool runs the pipeline from the saved feature table; the
  dashboard otherwise never trains, per section 14.

## 11. Recommendations

1. **Accident risk**: ask the client whether minor incidents are under-reported (0.91% of windows
   have one). More incidents, or incident-precursor signals such as speed variance within the window, are the likely
   route to 0.75; more model tuning is not.
2. **Congestion**: the class boundaries follow V/C exactly, so a better volume forecast is the main lever; a
   cost-sensitive loss favouring Heavy/Severe recall would suit operators better than macro-F1 alone.
3. **Deploy XGBoost for volume and speed**; keep the LSTM as a benchmark to re-run as more history accumulates.
4. **Add a weather-forecast feed** so target-window weather can be used without leaking.
5. Monitor coverage of the 80% ranges in production as a simple drift alarm.
