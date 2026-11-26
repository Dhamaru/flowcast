"""
features.py - M3: turn the clean table into the columns a model learns from.

A model only knows what's in its columns. A raw reading says what's happening NOW;
to forecast, the model also needs the recent past and the time of day. This file adds
41 "feature" columns, grouped like this:

    TIME      hour_sin / hour_cos   hour of day as a point on a circle (see below)
              is_weekend, is_peak_period
    CALENDAR  public_holiday, event_flag, roadwork_flag, days_to_event
    ROAD      road_capacity, segment_length_km  (never change for a road)
    HISTORY   traffic_volume_lag1    volume 30 min ago     (lag = "look back")
              traffic_volume_lag48   volume same time yesterday (48 x 30 min = 24 h)
              traffic_volume_roll4_mean   average of the last 4 windows (2 hours)
              ...and the same for speed, plus last window's weather and V/C ratio

KEY IDEAS FOR BEGINNERS
    * Why sin/cos for the hour? As a plain number, 23 and 0 look far apart, but 11 pm and
      midnight are neighbours. sin and cos put the hours on a clock face, so they are close.
    * The no-cheating rule (called "leakage"): every feature must be something you'd KNOW
      at the moment you make the forecast. This window's own volume or weather isn't known
      yet, so only LAST window's values are used. Breaking this rule gives great test
      scores that collapse in real use.
    * Forecasting further ahead: horizon_features(df, h) moves the history back h-1 more
      windows. For a 90-minute forecast (h=3) the newest volume the model sees is 90 min old.
"""

import json

import numpy as np
import pandas as pd

from utils import CONFIG, get_logger, path

log = get_logger("M3")
FEAT = CONFIG["features"]

TIME_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "is_peak_period"]
CALENDAR_COLS = ["public_holiday", "event_flag", "roadwork_flag", "holiday_peak_interaction",
                 "days_to_event"]
STATIC_COLS = ["road_capacity", "segment_length_km"]
WEATHER_ONEHOT = [f"weather_{w}" for w in ["Clear", "Cloudy", "Fog", "Overcast", "Rain"]]
WEATHER_SOURCE = ["is_rain", "is_low_visibility", "temperature", "temp_band", "rainfall",
                  "visibility"] + WEATHER_ONEHOT
HISTORY_COLS = (
    [f"{c}_lag{k}" for c in ["traffic_volume", "avg_speed"] for k in FEAT["lags"]]
    + [f"{c}_roll{w}_{s}" for c in ["traffic_volume", "avg_speed"]
       for w in FEAT["rolling_windows"] for s in ["mean", "std"]]
    + ["occupancy_lag1", "vc_ratio_lag1", "capacity_headroom_lag1"]
    + [f"{c}_lag1" for c in WEATHER_SOURCE]
)
FEATURE_COLS = TIME_COLS + CALENDAR_COLS + STATIC_COLS + HISTORY_COLS  # one list for every model


def add_time_features(df):
    """Cyclical hour and day-of-week (sin/cos), weekend and peak-period flags."""
    hour = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60  # 07:30 -> 7.5
    dow = df["timestamp"].dt.dayofweek  # Monday 0 ... Sunday 6
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    df["dow_sin"], df["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)
    df["is_weekend"] = (dow >= 5).astype(int)
    in_peak = np.zeros(len(df), dtype=bool)
    for start, end in FEAT["peak_hours"]:
        in_peak |= hour.between(start, end).to_numpy()
    df["is_peak_period"] = (in_peak & (df["is_weekend"] == 0)).astype(int)
    return df


def add_calendar_features(df):
    """Holiday x peak interaction and days to the nearest local event (capped at 7)."""
    df["holiday_peak_interaction"] = df["public_holiday"] * df["is_peak_period"]
    day = pd.to_datetime(df["date"])
    event_days = np.sort(day[df["event_flag"] == 1].unique())
    if len(event_days):
        gaps = np.abs(day.to_numpy()[:, None] - event_days[None, :]).min(axis=1)
        df["days_to_event"] = np.minimum(gaps / np.timedelta64(1, "D"), 7)
    else:
        df["days_to_event"] = 7.0
    return df


def add_weather_features(df):
    """Rain and low-visibility flags, temperature band (cold / mild / hot as 0 / 1 / 2)."""
    df["is_rain"] = (df["weather_condition"] == "Rain").astype(int)
    df["is_low_visibility"] = (df["visibility"] < FEAT["low_visibility_m"]).astype(int)
    df["temp_band"] = np.digitize(df["temperature"], FEAT["temperature_bands"])
    for col in WEATHER_ONEHOT:
        if col not in df:
            df[col] = 0  # a small upload might not contain every condition
    return df


def add_history_features(df):
    """Lags, rolling mean/std, capacity ratios and weather - all from earlier windows only."""
    # Sort by road then time, and group by road, so shift() never mixes two roads:
    # NL-002's first row must not get NL-001's last reading as its "previous" value.
    df = df.sort_values(["road_id", "timestamp"]).reset_index(drop=True)
    g = df.groupby("road_id")
    # V/C ratio = how full the road is. Capacity is per hour, a window is half an hour.
    # This window's own value is NOT a feature (it contains the answer); only _lag1 is.
    df["vc_ratio"] = df["traffic_volume"] / (df["road_capacity"] / 2)
    df["capacity_headroom"] = df["road_capacity"] / 2 - df["traffic_volume"]
    for col in ["traffic_volume", "avg_speed"]:
        for k in FEAT["lags"]:
            # shift(k) moves every value down k rows: row t now holds the value from t-k.
            df[f"{col}_lag{k}"] = g[col].shift(k)
        # shift(1) BEFORE rolling, so the average covers the windows before this one.
        # Without it, each window would include its own value - a sneaky form of leakage.
        previous = g[col].shift(1)
        for w in FEAT["rolling_windows"]:
            rolled = previous.groupby(df["road_id"]).rolling(w, min_periods=w)
            df[f"{col}_roll{w}_mean"] = rolled.mean().reset_index(level=0, drop=True)
            df[f"{col}_roll{w}_std"] = rolled.std().reset_index(level=0, drop=True)
    for col in ["occupancy", "vc_ratio", "capacity_headroom"] + WEATHER_SOURCE:
        df[f"{col}_lag1"] = g[col].shift(1)
    return df


def parse_vehicle_mix(df):
    """vehicle_type_dist is a JSON string; split it into share columns."""
    shares = pd.DataFrame(df["vehicle_type_dist"].map(json.loads).tolist(), index=df.index)
    return df.join(shares.add_prefix("share_"))


def build_features(df):
    """All feature groups, in order. Shared by the pipeline and the dashboard's upload view."""
    df = add_time_features(df)
    df = add_calendar_features(df)
    df = add_weather_features(df)
    df = add_history_features(df)
    return parse_vehicle_mix(df)


def horizon_features(df, h):
    """
    Feature matrix for forecasting h windows ahead (PRD 3.1: 1-4 windows).

    History columns are shifted by h-1 more windows within each segment, so for
    h = 3 the newest volume the model sees is 90 minutes before the target.
    Time, calendar and static columns describe the target window and are known in advance.
    """
    X = df[FEATURE_COLS].copy()
    if h > 1:
        X[HISTORY_COLS] = df.groupby("road_id")[HISTORY_COLS].shift(h - 1)
    return X


def run(clean_df):
    """Build features on M2's table and save the versioned feature table."""
    df = build_features(clean_df)
    version = CONFIG["project"]["version"]
    out = path("processed", f"flowcast_features_v{version}.csv")
    df.to_csv(out, index=False)
    # compact copy for the dashboard: the CSV is over GitHub's 100 MB file limit
    table = pd.read_csv(out, parse_dates=["timestamp"])
    table.to_parquet(out.with_suffix(".parquet"), index=False)
    warmup = df[HISTORY_COLS].isna().any(axis=1).sum()
    log.info("%d rows x %d columns, %d model features; %d warm-up rows lack full history",
             len(df), df.shape[1], len(FEATURE_COLS), warmup)
    return df
