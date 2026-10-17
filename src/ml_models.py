"""
ml_models.py - M5: train the classical machine-learning models and pick the best.

WHAT GETS PREDICTED
    traffic_volume    a number   -> regression,      30 / 60 / 90 / 120 min ahead
    avg_speed         a number   -> regression,      30 / 60 / 90 / 120 min ahead
    travel_time       a number   -> not trained: 60 x road length / predicted speed
    congestion_level  4 labels   -> classification,  next 30 min
    accident risk     yes / no   -> classification,  next 30 min, as a probability

THE MODELS COMPARED (simplest first)
    Linear / Logistic Regression  one weight per feature, added up
    Decision Tree                 a chain of yes/no questions, e.g. "lag1 <= 490?"
    Random Forest                 100 trees on random slices of the data, averaged
    XGBoost                       trees built one after another, each fixing the last one's errors
    SVM                           draws the widest possible boundary between classes

HOW THE WINNER IS CHOSEN (the part beginners most often get wrong)
    1. Split by TIME: first 70% train, next 10% validation, last 20% test. Never shuffle.
    2. For each model, try a few settings (a "grid") with cross-validation INSIDE the
       training data: train on weeks 1-4 and check on week 5, then 1-5 and check on 6...
    3. Pick the model with the best cross-validation score.
    4. Only then look at the test window, once, to report how good it really is.
    If you pick the model by its test score, the test set stops being a fair exam.

Each model is saved as a Pipeline: any scaling step is saved together with the model,
so it can't be forgotten later (that mistake once made every accident risk show 0%).

Also here: ScratchLinearRegression, the same idea as LinearRegression but written by hand
with NumPy, to see how training works inside (the PRD asks for this once).
"""

import pickle
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

import evaluate as ev
from features import FEATURE_COLS, horizon_features
from utils import CONFIG, get_logger, path, split_cutoffs

log = get_logger("M5")
SEED = CONFIG["project"]["seed"]
LEVELS = CONFIG["cleaning"]["congestion_levels"]
NEEDS_SCALING = {"Linear Regression", "Logistic Regression", "SVM"}


class ScratchLinearRegression:
    """
    Linear regression trained by gradient descent in plain NumPy (PRD 10.2, build-it-once rule).

    prediction = X @ w + b,  loss = mean((prediction - y)^2)
    each epoch: w -= lr * dLoss/dw,  b -= lr * dLoss/db
    """

    def __init__(self, learning_rate, epochs):
        """Store the learning rate and the number of epochs."""
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_history = []

    def fit(self, X, y):
        """Run gradient descent on X, y and record the loss each epoch."""
        n, k = X.shape  # n rows, k features
        self.w, self.b = np.zeros(k), 0.0  # start with every weight at 0
        for epoch in range(self.epochs):
            # 1. predict with the current weights and see how wrong each prediction is
            error = X @ self.w + self.b - y
            loss = np.mean(error**2)  # mean squared error: one number for "how wrong overall"
            if not np.isfinite(loss):  # learning rate too high: stop instead of returning garbage
                log.info("scratch linreg diverged at epoch %d", epoch)
                break
            self.loss_history.append(float(loss))
            # 2. the gradient says which way each weight should move to lower the loss;
            #    step that way, by a small amount (the learning rate). Repeat every epoch.
            #    Too big a step overshoots: with lr = 0.1 the loss blew up to ~10^100.
            self.w -= self.learning_rate * (2 / n) * (X.T @ error)
            self.b -= self.learning_rate * (2 / n) * error.sum()
        return self

    def predict(self, X):
        """X @ w + b."""
        return X @ self.w + self.b


def candidates(task):
    """The model families PRD 11.2 asks to compare, per task."""
    if task == "regression":
        return {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=SEED),
            "Random Forest": RandomForestRegressor(random_state=SEED, n_jobs=-1),
            "XGBoost": XGBRegressor(random_state=SEED, n_jobs=-1),
        }
    if task == "congestion":
        return {
            "Decision Tree": DecisionTreeClassifier(random_state=SEED),
            "Random Forest": RandomForestClassifier(random_state=SEED, n_jobs=-1),
            "XGBoost": XGBClassifier(random_state=SEED, n_jobs=-1, eval_metric="mlogloss"),
            "SVM": SVC(random_state=SEED),
        }
    # accident: class weighting because positives are ~1% of windows (M4 finding)
    return {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=2000),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", random_state=SEED, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(random_state=SEED, n_jobs=-1, eval_metric="logloss"),
        "SVM": SVC(class_weight="balanced", random_state=SEED),
    }


def pipeline(name, estimator):
    """Wrap a model in a Pipeline, with a scaler first if it needs one."""
    # Linear models and SVM compare features by size, so a column in thousands (volume)
    # would drown one between 0 and 1 (rain flag). StandardScaler puts every column on the
    # same scale. Tree models ask "is x <= 490?" and don't care about scale.
    steps = [("scale", StandardScaler())] if name in NEEDS_SCALING else []
    return Pipeline(steps + [("model", estimator)])


def frames(df, target, h=1):
    """
    Feature matrix for horizon h plus train / validation / test row indexes.

    Target rows must be real readings (is_imputed == 0) with full history.
    Indexes are in time order, which TimeSeriesSplit relies on.
    """
    X = horizon_features(df, h)
    # Usable rows: every feature present, and a REAL reading as the answer. Rebuilt windows
    # (is_imputed = 1) help with history, but a model must never learn from a filled-in guess.
    ok = X.notna().all(axis=1) & df[target].notna() & (df["is_imputed"] == 0)
    order = df.loc[ok].sort_values(["timestamp", "road_id"]).index
    train_end, val_end = split_cutoffs(df["timestamp"])
    ts = df.loc[order, "timestamp"]
    return X, {
        "train": order[ts < train_end],
        "val": order[(ts >= train_end) & (ts < val_end)],
        "test": order[ts >= val_end],
    }


def scores(model, X):
    """Ranking score for a binary classifier: probability if it has one, else the SVM margin."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def svm_rows(train_idx, y, task):
    """SVM cost grows ~quadratically with rows, so it trains on a sample (kept in time order)."""
    rng = np.random.default_rng(SEED)
    n = CONFIG["svm_sample_rows"]
    if task == "accident":  # keep every rare positive, sample the negatives
        pos = train_idx[y.loc[train_idx].to_numpy() == 1]
        neg = train_idx[y.loc[train_idx].to_numpy() == 0]
        picked = np.concatenate([pos, rng.choice(neg, size=min(n, len(neg)), replace=False)])
    else:
        picked = rng.choice(train_idx, size=min(n, len(train_idx)), replace=False)
    return train_idx[np.isin(train_idx, picked)]


def tune_and_compare(task, X, y, idx, scoring):
    """Grid-search each candidate with time-series CV on the training window; pick by CV mean."""
    grids = CONFIG["grids"]["regression" if task == "regression" else task]
    results, fitted = [], {}
    for name, estimator in candidates(task).items():
        rows = svm_rows(idx["train"], y, task) if name == "SVM" else idx["train"]
        if task == "accident" and name == "XGBoost":
            positives = y.loc[rows].sum()
            estimator.set_params(scale_pos_weight=(len(rows) - positives) / positives)
        grid = {f"model__{k}": v for k, v in grids.get(name, {}).items()}
        search = GridSearchCV(
            pipeline(name, estimator),
            grid,
            scoring=scoring,
            cv=TimeSeriesSplit(n_splits=CONFIG["split"]["cv_folds"]),
        )
        start = time.perf_counter()
        search.fit(X.loc[rows], y.loc[rows])
        i = search.best_index_
        result = {
            "model": name,
            "cv_mean": float(search.best_score_),
            "cv_std": float(search.cv_results_["std_test_score"][i]),
            "params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
            "train_rows": int(len(rows)),
            "fit_seconds": round(time.perf_counter() - start, 1),
        }
        log.info(
            "  %-19s CV %s = %.4f (+/- %.4f)  params %s",
            name,
            scoring,
            abs(result["cv_mean"]),
            result["cv_std"],
            result["params"],
        )
        results.append(result)
        fitted[name] = search.best_estimator_
    best = ev.select_winner(results)
    log.info("  selected on CV: %s", best)
    return results, fitted, best


def card(
    name, file, model_name, result, target, h, idx, df, metrics, version, data_sha, selected_by
):
    """Model card fields for one saved model (PRD 13.4)."""
    train_end, _ = split_cutoffs(df["timestamp"])
    return {
        "name": name,
        "file": file,
        "algorithm": model_name,
        "target": target,
        "horizon": h,
        "version": version,
        "train_window": f"{df['timestamp'].min():%Y-%m-%d} to before {train_end:%Y-%m-%d %H:%M}",
        "train_rows": result["train_rows"],
        "params": result["params"],
        "selected_by": selected_by,
        "features": FEATURE_COLS,
        "data_sha": data_sha,
        "test_metrics": metrics,
    }


def save_bundle(bundle, file):
    """Pickle a model bundle into models/."""
    with open(path("models", file), "wb") as f:
        pickle.dump(bundle, f)


def regression_target(df, target, version, data_sha, cards, scratch=False):
    """Tune at horizon 1, then refit the winner for horizons 2-4 with its best settings."""
    log.info("=== %s ===", target)
    X, idx = frames(df, target, 1)
    y = df[target]
    results, fitted, best = tune_and_compare(
        "regression", X, y, idx, "neg_root_mean_squared_error"
    )
    for r in results:
        r["cv_rmse"] = abs(r.pop("cv_mean"))
        r["test"] = ev.regression_metrics(
            y.loc[idx["test"]], fitted[r["model"]].predict(X.loc[idx["test"]])
        )

    out = {"selected": best, "candidates": results, "horizons": {}}
    if scratch:
        scaler = StandardScaler().fit(X.loc[idx["train"]])
        cfg = CONFIG["scratch_linreg"]
        model = ScratchLinearRegression(cfg["learning_rate"], cfg["epochs"])
        model.fit(scaler.transform(X.loc[idx["train"]]), y.loc[idx["train"]].to_numpy())
        out["scratch_linreg"] = {
            **ev.regression_metrics(
                y.loc[idx["test"]], model.predict(scaler.transform(X.loc[idx["test"]]))
            ),
            "final_train_mse": model.loss_history[-1],
            "epochs": len(model.loss_history),
        }

    winner = next(r for r in results if r["model"] == best)
    preds = []
    for h in CONFIG["horizons"]:
        Xh, idx_h = (X, idx) if h == 1 else frames(df, target, h)
        model = (
            fitted[best]
            if h == 1
            else clone(fitted[best]).fit(Xh.loc[idx_h["train"]], y.loc[idx_h["train"]])
        )
        val = idx_h["val"]
        band = ev.interval_by_hour(
            y.loc[val] - model.predict(Xh.loc[val]), df.loc[val, "timestamp"].dt.hour
        )
        test = idx_h["test"]
        p = model.predict(Xh.loc[test])
        half = df.loc[test, "timestamp"].dt.hour.map(band).to_numpy()
        metrics = ev.regression_metrics(y.loc[test], p)
        metrics["coverage"] = ev.coverage(y.loc[test], p - half, p + half)
        metrics["avg_half_width"] = float(half.mean())
        out["horizons"][h] = metrics
        log.info(
            "  h=%d (%3d min): RMSE %.2f  MAPE %.2f%%  coverage of %d%% range %.1f%%",
            h,
            h * 30,
            metrics["rmse"],
            metrics["mape"],
            CONFIG["confidence"] * 100,
            metrics["coverage"] * 100,
        )
        file = f"{target}_h{h}.pkl"
        save_bundle(
            {
                "model": model,
                "features": FEATURE_COLS,
                "target": target,
                "horizon": h,
                "band_by_hour": band,
                "version": version,
                "algorithm": best,
            },
            file,
        )
        cards.append(
            card(
                f"{target}, {h * 30} min ahead",
                file,
                best,
                winner,
                target,
                h,
                idx_h,
                df,
                metrics,
                version,
                data_sha,
                "time-series CV RMSE at horizon 1; same settings refit per horizon",
            )
        )
        preds.append(
            pd.DataFrame(
                {
                    "road_id": df.loc[test, "road_id"],
                    "road_name": df.loc[test, "road_name"],
                    "timestamp": df.loc[test, "timestamp"],
                    "horizon": h,
                    "target": target,
                    "actual": y.loc[test],
                    "pred": p,
                    "lower": p - half,
                    "upper": p + half,
                    "weather_condition": df.loc[test, "weather_condition"],
                    "segment_length_km": df.loc[test, "segment_length_km"],
                }
            )
        )
    return out, fitted[best], pd.concat(preds)


def travel_time_from_speed(df, speed_preds, version):
    """Travel time = 60 x segment length / predicted speed (PRD 11.1: from predicted speed)."""
    log.info("=== travel_time (derived from predicted speed) ===")
    out, preds = {}, []
    for h in CONFIG["horizons"]:
        with open(path("models", f"avg_speed_h{h}.pkl"), "rb") as f:
            bundle = pickle.load(f)
        model = bundle["model"]
        Xh, idx = frames(df, "avg_speed", h)
        val = idx["val"]
        tt_val = 60 * df.loc[val, "segment_length_km"] / model.predict(Xh.loc[val])
        band = ev.interval_by_hour(
            df.loc[val, "travel_time"] - tt_val, df.loc[val, "timestamp"].dt.hour
        )
        sp = speed_preds[speed_preds["horizon"] == h]
        pred = 60 * sp["segment_length_km"] / sp["pred"]
        actual = df.loc[sp.index, "travel_time"]
        half = sp["timestamp"].dt.hour.map(band).to_numpy()
        metrics = ev.regression_metrics(actual, pred)
        metrics["coverage"] = ev.coverage(actual, pred - half, pred + half)
        out[h] = metrics
        log.info(
            "  h=%d: RMSE %.3f min  MAPE %.2f%%  coverage %.1f%%",
            h,
            metrics["rmse"],
            metrics["mape"],
            metrics["coverage"] * 100,
        )
        preds.append(
            sp.assign(
                target="travel_time",
                actual=actual,
                pred=pred,
                lower=pred - half,
                upper=pred + half,
            )
        )
        bundle["travel_time_band_by_hour"] = band
        save_bundle(bundle, f"avg_speed_h{h}.pkl")
    return out, pd.concat(preds)


def congestion_target(df, version, data_sha, cards):
    """Tune, compare and save the 4-class congestion classifier."""
    log.info("=== congestion_level ===")
    X, idx = frames(df, "congestion_code", 1)
    y = df["congestion_code"].astype(int)
    results, fitted, best = tune_and_compare("congestion", X, y, idx, "f1_macro")
    for r in results:
        r["test"] = ev.congestion_metrics(
            y.loc[idx["test"]], fitted[r["model"]].predict(X.loc[idx["test"]]), LEVELS
        )
        r["cv_macro_f1"] = r.pop("cv_mean")
    test = next(r for r in results if r["model"] == best)["test"]
    ev.plot_confusion(test["confusion_matrix"], LEVELS)
    log.info("  test macro-F1 %.3f  accuracy %.3f", test["macro_f1"], test["accuracy"])
    save_bundle(
        {
            "model": fitted[best],
            "features": FEATURE_COLS,
            "labels": LEVELS,
            "horizon": 1,
            "version": version,
            "algorithm": best,
        },
        "congestion_h1.pkl",
    )
    cards.append(
        card(
            "congestion_level, next window",
            "congestion_h1.pkl",
            best,
            next(r for r in results if r["model"] == best),
            "congestion_level",
            1,
            idx,
            df,
            {k: v for k, v in test.items() if k in ("macro_f1", "accuracy")},
            version,
            data_sha,
            "time-series CV macro-F1",
        )
    )
    return {"selected": best, "candidates": results, "test": test}


def accident_target(df, version, data_sha, cards):
    """Rank windows by accident risk; calibrate scores into probabilities on validation."""
    log.info("=== accident risk ===")
    df = df.assign(
        has_accident=(df["accident_count"] > 0).astype(float).where(df["accident_count"].notna())
    )
    X, idx = frames(df, "has_accident", 1)
    y = df["has_accident"].fillna(-1).astype(int)  # -1 = rebuilt window, never in idx
    results, fitted, best = tune_and_compare("accident", X, y, idx, "roc_auc")
    for r in results:
        r["cv_roc_auc"] = r.pop("cv_mean")
        r["test_roc_auc"] = ev.accident_metrics(
            y.loc[idx["test"]].to_numpy(), scores(fitted[r["model"]], X.loc[idx["test"]]), 0.5
        )["roc_auc"]
    model = fitted[best]
    # Only ~1 window in 100 has an accident. class_weight="balanced" (see candidates()) makes
    # the model take those rare cases seriously, but it also pushes every score up, so a raw
    # score of 0.37 does NOT mean a 37% chance. "Calibration" learns, on the validation data,
    # how to turn the raw score into a real probability (average 0.87% vs the true 0.97%).
    val_scores = scores(model, X.loc[idx["val"]]).reshape(-1, 1)
    calibrator = LogisticRegression().fit(val_scores, y.loc[idx["val"]].to_numpy())
    val_prob = calibrator.predict_proba(val_scores)[:, 1]
    threshold = ev.best_f1_threshold(y.loc[idx["val"]].to_numpy(), val_prob)
    test_prob = calibrator.predict_proba(scores(model, X.loc[idx["test"]]).reshape(-1, 1))[:, 1]
    test = ev.accident_metrics(y.loc[idx["test"]].to_numpy(), test_prob, threshold)
    test["base_rate"] = float(y.loc[idx["test"]].mean())
    test["mean_predicted_probability"] = float(test_prob.mean())
    log.info(
        "  test ROC-AUC %.3f; alert threshold %.3f -> precision %.3f recall %.3f",
        test["roc_auc"],
        threshold,
        test["precision"],
        test["recall"],
    )
    save_bundle(
        {
            "model": model,
            "calibrator": calibrator,
            "threshold": threshold,
            "features": FEATURE_COLS,
            "horizon": 1,
            "version": version,
            "algorithm": best,
        },
        "accident_h1.pkl",
    )
    cards.append(
        card(
            "accident risk, next window",
            "accident_h1.pkl",
            best,
            next(r for r in results if r["model"] == best),
            "accident_count > 0",
            1,
            idx,
            df,
            {k: v for k, v in test.items() if isinstance(v, float)},
            version,
            data_sha,
            "time-series CV ROC-AUC; Platt-calibrated and threshold chosen on validation",
        )
    )
    return {"selected": best, "candidates": results, "test": test}


def run(df, version, data_sha):
    """Train, compare and save every classical model; return the scoreboard."""
    cards = []
    train_end, val_end = split_cutoffs(df["timestamp"])
    board = {
        "version": version,
        "data_sha": data_sha,
        "split": {"train_before": str(train_end), "test_from": str(val_end)},
    }

    board["volume"], volume_model, volume_preds = regression_target(
        df, "traffic_volume", version, data_sha, cards, scratch=True
    )
    board["speed"], speed_model, speed_preds = regression_target(
        df, "avg_speed", version, data_sha, cards
    )
    board["travel_time"], tt_preds = travel_time_from_speed(df, speed_preds, version)
    board["congestion"] = congestion_target(df, version, data_sha, cards)
    board["accident"] = accident_target(df, version, data_sha, cards)

    h1 = volume_preds[volume_preds["horizon"] == 1]
    start = time.perf_counter()
    volume_model.predict(horizon_features(df, 1).loc[h1.index])
    board["volume"]["infer_seconds"] = time.perf_counter() - start
    board["volume"]["infer_rows"] = len(h1)
    # PRD 14.1: ranked drivers from the tree ensembles, for every tree-based winner
    board["volume"]["feature_importance"] = ev.feature_importance(volume_model, FEATURE_COLS)
    board["speed"]["feature_importance"] = ev.feature_importance(speed_model, FEATURE_COLS)
    with open(path("models", "congestion_h1.pkl"), "rb") as f:
        congestion_model = pickle.load(f)["model"]
    board["congestion"]["feature_importance"] = ev.feature_importance(
        congestion_model, FEATURE_COLS
    )
    breakdown = ev.error_breakdown(h1.rename(columns={"actual": "y", "pred": "p"}), "y", "p")
    board["volume"]["error_breakdown"] = breakdown
    ev.plot_error_analysis(h1, "actual", "pred", breakdown, "traffic volume")

    pd.concat([volume_preds, speed_preds, tt_preds]).to_parquet(
        path("reports", "test_predictions.parquet")  # model output, not part of the dataset
    )
    ev.write_model_cards(cards)
    return board
