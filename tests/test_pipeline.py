"""
test_pipeline.py - small automatic checks that each step does what it promises.

    pytest flowcast/tests          runs every test in about 15 seconds

HOW TO READ A TEST
    Each test builds a tiny made-up table (a few rows), runs ONE function on it, and uses
    `assert` to check the answer. If the answer is wrong, pytest prints which test failed.
    Tiny tables make the right answer obvious, e.g. "a volume of -5 with vehicle_count 250
    must become 250".

WHY BOTHER
    If you change clean.py later and accidentally break the repair of negative volumes,
    the test fails straight away, instead of you finding out from strange model scores.
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import clean  # noqa: E402
import evaluate  # noqa: E402
import features  # noqa: E402
import ingest  # noqa: E402
from dl_model import build_sequences  # noqa: E402
from ml_models import ScratchLinearRegression  # noqa: E402
from utils import split_masks  # noqa: E402


def raw_traffic(n=6, **overrides):
    """A small raw traffic table in the traffic_sensor_log format, one segment, n windows."""
    times = pd.date_range("2025-01-01 00:00", periods=n, freq="30min")
    df = pd.DataFrame(
        {
            "road_id": "NL-001",
            "road_name": "Test Rd",
            "latitude": 28.6,
            "longitude": 77.1,
            "weather_station_id": "WS-NORTH",
            "date": times.strftime("%Y-%m-%d"),
            "time": times.strftime("%H:%M"),
            "traffic_volume": 400.0,
            "vehicle_count": 400.0,
            "vehicle_type_dist": '{"2W": 0.4, "Car": 0.4, "LCV": 0.1, "HCV": 0.1}',
            "avg_speed": 40.0,
            "occupancy": 30.0,
            "congestion_level": "Moderate",
            "travel_time": 3.0,
            "accident_count": 0.0,
            "signal_timing": 45.0,
            "road_capacity": 1600.0,
        }
    )
    for col, values in overrides.items():
        df[col] = values
    return df


# ---------------------------------------------------------------- M1: checking raw rows
def test_validation_quarantines_impossible_values():
    df = raw_traffic(
        traffic_volume=[400, -5, 400, 400, 400, 400],
        avg_speed=[40, 40, 250, 40, 40, 40],
        occupancy=[30, 30, 30, 120, 30, 30],
    )
    _, rejects = ingest.validate_traffic(df)
    assert len(rejects) == 3
    assert set(rejects["reject_reason"].str.strip(";")) == {
        "negative_traffic_volume",
        "avg_speed_over_max",
        "occupancy_over_max",
    }


def test_validation_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        ingest.validate_traffic(raw_traffic().drop(columns=["avg_speed"]))


def test_mixed_date_formats_parse_to_same_timestamp():
    a = ingest.make_timestamp(pd.Series(["2025-03-04"]), pd.Series(["07:30"]), "%Y-%m-%d")
    b = ingest.make_timestamp(pd.Series(["04/03/2025"]), pd.Series(["07:30"]), "%d/%m/%Y")
    assert a.iloc[0] == b.iloc[0] and a.iloc[0].tzinfo is not None


# ---------------------------------------------------------------- M2: repairing and filling
def test_bad_readings_recovered_from_redundant_fields():
    df = raw_traffic(
        traffic_volume=[400, -5, np.nan, 400, 400, 400],
        vehicle_count=[400, 250, 260, 400, 400, 400],
        avg_speed=[40, 40, 40, 300, 40, 40],
    )
    df, _ = ingest.validate_traffic(df)
    out = clean.fix_outliers(df, {})
    assert out["traffic_volume"].tolist()[1:3] == [250, 260]  # from vehicle_count, not guessed
    assert out["avg_speed"].iloc[3] == pytest.approx(40.0)  # 60 x length / travel_time


def test_missing_windows_are_rebuilt_and_flagged():
    df, _ = ingest.validate_traffic(raw_traffic().drop(index=[2]))
    df = clean.fix_outliers(df, {})
    out = clean.fill_missing_traffic(df, {})
    assert len(out) == 6 and out["is_imputed"].sum() == 1
    assert out["traffic_volume"].notna().all()


def test_congestion_derived_from_vc_bands():
    df = pd.DataFrame(
        {
            "traffic_volume": [300, 500, 700, 900],
            "road_capacity": 1600,
            "congestion_level": [None] * 4,
        }
    )  # capacity/2 = 800 per window
    out = clean.derive_congestion(df, {})
    assert out["congestion_level"].tolist() == ["Free-flow", "Moderate", "Heavy", "Severe"]


# ---------------------------------------------------------------- M3: features, no peeking ahead
def test_cyclical_hour_puts_midnight_next_to_2330():
    df = pd.DataFrame(
        {"timestamp": pd.to_datetime(["2025-01-01 23:30", "2025-01-02 00:00", "2025-01-02 12:00"])}
    )
    f = features.add_time_features(df)
    p = f[["hour_sin", "hour_cos"]].to_numpy()
    assert np.linalg.norm(p[0] - p[1]) < np.linalg.norm(p[1] - p[2])


def test_lags_and_rolling_use_only_earlier_windows():
    times = pd.date_range("2025-01-01", periods=60, freq="30min", tz="Asia/Kolkata")
    df = pd.DataFrame(
        {
            "road_id": "NL-001",
            "timestamp": times,
            "traffic_volume": np.arange(60.0),
            "avg_speed": 40.0,
            "occupancy": 30.0,
            "road_capacity": 1600.0,
        }
    )
    for col in features.WEATHER_SOURCE:
        df[col] = 0.0
    f = features.add_history_features(df)
    assert f.loc[10, "traffic_volume_lag1"] == 9
    assert f.loc[10, "traffic_volume_roll4_mean"] == np.mean(
        [6, 7, 8, 9]
    )  # excludes window 10 itself
    assert f.loc[50, "traffic_volume_lag48"] == 2


def test_horizon_features_shift_history_back():
    times = pd.date_range("2025-01-01", periods=60, freq="30min", tz="Asia/Kolkata")
    df = pd.DataFrame({"road_id": "NL-001", "timestamp": times})
    for col in features.FEATURE_COLS:
        df[col] = 0.0
    df["traffic_volume_lag1"] = np.arange(60.0)
    X3 = features.horizon_features(df, 3)
    assert X3.loc[20, "traffic_volume_lag1"] == 18  # newest volume is 3 windows before the target


def test_no_feature_uses_the_target_window():
    banned = {
        "traffic_volume",
        "avg_speed",
        "occupancy",
        "vc_ratio",
        "capacity_headroom",
        "congestion_level",
        "travel_time",
        "temperature",
        "rainfall",
        "visibility",
        "is_rain",
    }
    assert banned.isdisjoint(features.FEATURE_COLS)


# ---------------------------------------------------------------- M5 / M6: training and scoring
def test_split_is_in_time_order():
    df = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=100, freq="30min")})
    train, val, test = split_masks(df)
    assert df.loc[train, "timestamp"].max() < df.loc[val, "timestamp"].min()
    assert df.loc[val, "timestamp"].max() < df.loc[test, "timestamp"].min()


def test_scratch_linear_regression_finds_the_line():
    X = np.linspace(-1, 1, 200).reshape(-1, 1)
    model = ScratchLinearRegression(learning_rate=0.1, epochs=2000).fit(X, 2 * X.ravel() + 1)
    assert model.w[0] == pytest.approx(2, abs=0.01) and model.b == pytest.approx(1, abs=0.01)


def test_regression_metrics():
    m = evaluate.regression_metrics([100, 200], [110, 190])
    assert m["rmse"] == pytest.approx(10) and m["mape"] == pytest.approx(7.5)


def test_sequences_take_the_n_rows_before_the_target():
    values = np.arange(20, dtype=float).reshape(-1, 1)
    seq = build_sequences(values, np.array([10]), 4)
    assert seq[0].ravel().tolist() == [6, 7, 8, 9]


def test_winner_is_best_cv_score_not_test_score():
    results = [{"model": "A", "cv_mean": -60.0}, {"model": "B", "cv_mean": -58.0}]
    assert evaluate.select_winner(results) == "B"  # sklearn scores: higher is better


def test_head_to_head_reports_gap_and_verdict():
    c = evaluate.head_to_head("XGBoost", 58.0, {"rmse": 60.9})
    assert c["beats_classical"] is False
    assert c["gap_pct"] == pytest.approx(5.0)
