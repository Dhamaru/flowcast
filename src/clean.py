"""
clean.py - M2: fix the data, then join the three tables into one.

WHAT IT DOES, IN ORDER (the order matters - each step relies on the one before)
    Stage 2  remove_duplicates()      drop rows the sensor logged twice (1,767 of them)
    Stage 3  fix_outliers()           repair impossible values:
                                        negative volume  -> copied from vehicle_count
                                        speed > 200 km/h -> recomputed from travel_time
                                        occupancy > 100% -> capped at 100
    Stage 4  fill_missing_traffic()   add back the 4,499 windows the sensor never logged,
                                      then fill missing numbers
             derive_congestion()      fill blank congestion labels using the V/C formula
    Stage 5  harmonise_categories()   "rain ", "RAIN", "rainy" -> "Rain" (13 spellings -> 5)
    Stage 6  merge_sources()          join weather and calendar onto every traffic row
    Stage 8  final_checks()           stop with an error if anything is still wrong
             write_quality_report()   write reports/data_quality.md

KEY IDEAS FOR BEGINNERS
    * Recover before you guess. vehicle_count equals traffic_volume on every good row, so a
      broken volume can be copied back exactly. Guessing (interpolating) is the last resort.
    * Missing ROWS are more dangerous than missing VALUES. Later, "the previous row" is used
      to mean "30 minutes ago". If a row is missing, that's silently wrong, so the full grid
      of every road x every 30-minute window is rebuilt first. Rebuilt rows are marked
      is_imputed = 1 and never used as answers to learn from.
    * Every stage writes a count into `stats`, which becomes the data-quality report.
"""

import json

import numpy as np
import pandas as pd

from utils import CONFIG, file_hash, get_logger, path

log = get_logger("M2")
CLEAN = CONFIG["cleaning"]
LEVELS = CLEAN["congestion_levels"]
SENSOR_COLS = ["traffic_volume", "avg_speed", "occupancy"]


# ---------------------------------------------------------------- stage 2
def remove_duplicates(df, stats):
    """Drop exact duplicates, then road_id + timestamp duplicates, keeping the fullest row."""
    before = len(df)
    df = df.drop_duplicates()
    stats["duplicates_exact"] = before - len(df)

    # Two rows for the same road and time but with different values: keep the one with
    # the fewest blanks. Sort fullest-first, then drop_duplicates keeps the first it sees.
    completeness = df.notna().sum(axis=1)
    df = df.assign(_complete=completeness).sort_values("_complete", ascending=False)
    before = len(df)
    df = df.drop_duplicates(["road_id", "timestamp"]).drop(columns="_complete")
    stats["duplicates_key"] = before - len(df)
    log.info("stage 2: removed %d exact + %d key-level duplicates",
             stats["duplicates_exact"], stats["duplicates_key"])
    return df


# ---------------------------------------------------------------- stage 3
def segment_lengths(df):
    """
    Segment length in km, recovered from the data: travel_time (min) x speed (km/h) / 60.

    The data dictionary says travel_time is derived from speed and segment
    length, so this product is constant per segment (spread ~0.002 km). It lets
    a faulty speed reading be recovered exactly from the travel time.
    """
    ok = df["avg_speed"].between(1, CONFIG["validation"]["max_speed_kmh"])
    length = df.loc[ok, "travel_time"] * df.loc[ok, "avg_speed"] / 60
    return length.groupby(df["road_id"]).median()


def fix_outliers(df, stats):
    """
    Rule-based repairs, then a statistical check.

    Policy per field (PRD 9.1: "cap or remove per field policy"):
      traffic_volume < 0 or missing -> vehicle_count (equal on 100% of valid rows)
      avg_speed > 200 or missing    -> 60 x segment length / travel_time (exact)
      occupancy > 100               -> capped at 100 (no backup field to recover it from)
      statistical extremes          -> flagged and kept: robust z-score > 4 within the same
                                       segment and time-of-day slot. After the rule fixes,
                                       what's left is mostly real (event days, incidents),
                                       and capping it would teach the model events don't happen.
    """
    max_speed = CONFIG["validation"]["max_speed_kmh"]
    df["segment_length_km"] = df["road_id"].map(segment_lengths(df))

    # Volume: example row NL-009 at 21:30 has traffic_volume = -6 but vehicle_count = 247.
    # vehicle_count is the same number on every good row, so 247 is the true value.
    bad_volume = (df["traffic_volume"] < 0) | df["traffic_volume"].isna()
    recoverable = bad_volume & (df["vehicle_count"] >= 0)
    stats["volume_recovered_from_vehicle_count"] = int(recoverable.sum())
    df.loc[recoverable, "traffic_volume"] = df.loc[recoverable, "vehicle_count"]
    df.loc[df["traffic_volume"] < 0, "traffic_volume"] = np.nan

    # Speed: travel_time = 60 x length / speed, so speed = 60 x length / travel_time.
    # Example: NL-002 says 317 km/h, but its travel time (1.02 min) gives 50.6 km/h.
    bad_speed = (df["avg_speed"] > max_speed) | df["avg_speed"].isna()
    recoverable = bad_speed & (df["travel_time"] > 0)
    stats["speed_recovered_from_travel_time"] = int(recoverable.sum())
    fix = df.loc[recoverable]
    df.loc[recoverable, "avg_speed"] = 60 * fix["segment_length_km"] / fix["travel_time"]
    df.loc[df["avg_speed"] > max_speed, "avg_speed"] = np.nan

    stats["occupancy_capped"] = int((df["occupancy"] > 100).sum())
    df["occupancy"] = df["occupancy"].clip(upper=CONFIG["validation"]["max_occupancy_pct"])

    # Statistical check: is a reading far from normal FOR THAT ROAD AT THAT TIME OF DAY?
    # 900 cars at 8 am is normal; 900 at 3 am is not. So compare within (road, time slot).
    # We only count these, we don't change them: they are mostly real events.
    slot = [df["road_id"], df["timestamp"].dt.strftime("%H:%M")]
    for col in SENSOR_COLS:
        median = df.groupby(slot)[col].transform("median")
        mad = (df[col] - median).abs().groupby(slot).transform("median")
        z = (df[col] - median).abs() / (1.4826 * mad)  # 1.4826 x MAD ~ one standard deviation
        stats[f"{col}_extreme_flagged"] = int(((z > CLEAN["zscore_flag"]) & (mad > 0)).sum())
    log.info("stage 3: volume recovered %d, speed recovered %d, occupancy capped %d, "
             "extremes flagged (kept) volume/speed/occupancy %d/%d/%d",
             stats["volume_recovered_from_vehicle_count"],
             stats["speed_recovered_from_travel_time"], stats["occupancy_capped"],
             *(stats[f"{c}_extreme_flagged"] for c in SENSOR_COLS))
    return df


# ---------------------------------------------------------------- stage 4
def full_grid(df, freq=f"{CLEAN['window_minutes']}min"):
    """Every key x every time step, so steps the sensor never logged become rows."""
    windows = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq=freq)
    grid = pd.MultiIndex.from_product([df["road_id"].unique(), windows],
                                      names=["road_id", "timestamp"]).to_frame(index=False)
    return grid.merge(df, on=["road_id", "timestamp"], how="left")


def fill_gaps(df, cols, group, stats, label):
    """
    Short gaps (<= config short_gap_windows) by time interpolation,
    longer gaps by the median for that group and time-of-day slot (PRD 9.1 stage 4).
    """
    df = df.sort_values([group, "timestamp"])
    slot = df["timestamp"].dt.strftime("%H:%M")
    for col in cols:
        # Measure how long each gap is. cumsum() gives every gap the same number as the
        # last good value before it, so grouping by that number counts the gap's length.
        missing = df[col].isna()
        run = (~missing).cumsum()
        gap_len = missing.groupby([df[group], run]).transform("sum")
        # Short gap (up to 1 hour): draw a straight line between the readings either side.
        # Longer gap: use the usual value for that road at that time of day.
        short = missing & (gap_len <= CLEAN["short_gap_windows"])
        interpolated = df.groupby(group)[col].transform(lambda s: s.interpolate(limit_area="inside"))
        df.loc[short, col] = interpolated[short]
        still = df[col].isna()
        df.loc[still, col] = df.groupby([df[group], slot])[col].transform("median")[still]
        stats[f"{label}_{col}_interpolated"] = int((short & df[col].notna()).sum())
        stats[f"{label}_{col}_median_filled"] = int((still & df[col].notna()).sum())
    return df


def fill_missing_traffic(df, stats):
    """Rebuild absent windows, then fill sensor gaps and the fields that follow from them."""
    observed = len(df)
    df = full_grid(df)
    # A rebuilt row has no road_name yet (the sensor never sent it), which is how we spot it.
    df["is_imputed"] = df["road_name"].isna().astype(int)
    stats["missing_windows_rebuilt"] = len(df) - observed

    # fields that never change for a road: copy them into the rebuilt rows
    static = ["road_name", "latitude", "longitude", "weather_station_id", "road_capacity",
              "segment_length_km"]
    df[static] = df.groupby("road_id")[static].transform("first")
    df = fill_gaps(df, SENSOR_COLS + ["signal_timing"], "road_id", stats, "traffic")

    df["vehicle_count"] = df["vehicle_count"].fillna(df["traffic_volume"])
    df["vehicle_type_dist"] = df.groupby("road_id")["vehicle_type_dist"].transform(
        lambda s: s.ffill().bfill())
    df["travel_time"] = df["travel_time"].fillna(60 * df["segment_length_km"] / df["avg_speed"])
    # an accident in a window the sensor never logged is unknown, not zero: kept as NaN
    # so the window feeds lag history but is never used as an accident training target
    log.info("stage 4: rebuilt %d missing windows; filled sensor gaps (see report)",
             stats["missing_windows_rebuilt"])
    return df


def derive_congestion(df, stats):
    """Fill blank congestion_level from V/C = volume / (capacity / 2), dictionary banding."""
    vc = df["traffic_volume"] / (df["road_capacity"] / 2)
    banded = pd.cut(vc, [-np.inf, *CLEAN["congestion_bands"], np.inf], right=False, labels=LEVELS)
    blank = df["congestion_level"].isna()
    stats["congestion_derived"] = int(blank.sum())
    df.loc[blank, "congestion_level"] = banded[blank].astype(str)
    return df


# ---------------------------------------------------------------- stage 5
def harmonise_categories(df, weather, stats):
    """Weather labels to the controlled vocabulary; day labels; ordinal and one-hot encodings."""
    raw_labels = weather["weather_condition"].str.strip().str.lower()
    stats["weather_labels_before"] = int(weather["weather_condition"].nunique())
    # a spelling not in the config map keeps its own (trimmed) text
    weather["weather_condition"] = (raw_labels.map(CLEAN["weather_labels"])
                                    .fillna(weather["weather_condition"].str.strip()))
    stats["weather_labels_after"] = int(weather["weather_condition"].nunique())
    stats["overcast_rows"] = int((weather["weather_condition"] == "Overcast").sum())
    stats["accident_rate_pct"] = round(float((df["accident_count"] > 0).mean() * 100), 2)

    df["day_of_week"] = df["timestamp"].dt.strftime("%a")
    # Models need numbers, not words. The levels have a natural order, so they become
    # 0 Free-flow, 1 Moderate, 2 Heavy, 3 Severe. The text column is kept as well.
    df["congestion_code"] = df["congestion_level"].map({lvl: i for i, lvl in enumerate(LEVELS)})
    log.info("stage 5: weather labels %d -> %d; congestion encoded ordinal 0-3",
             stats["weather_labels_before"], stats["weather_labels_after"])
    return df, weather


# ---------------------------------------------------------------- stage 6
def merge_sources(df, weather, calendar, stats):
    """Broadcast hourly weather onto both 30-min windows; attach calendar flags."""
    # full_grid() expects a road_id column, so the station id borrows that name for a moment
    weather = (full_grid(weather.rename(columns={"station_id": "road_id"}), freq="h")
               .rename(columns={"road_id": "station_id"}))
    weather = fill_gaps(weather, ["temperature", "visibility", "rainfall"], "station_id", stats,
                        "weather")
    weather["weather_condition"] = weather.groupby("station_id")["weather_condition"].transform(
        lambda s: s.ffill().bfill())

    # Weather is hourly, traffic is every 30 minutes. floor("h") turns 07:30 into 07:00,
    # so both the 07:00 and 07:30 traffic windows pick up the 07:00 weather reading.
    df["weather_hour"] = df["timestamp"].dt.floor("h")
    w = weather[["station_id", "timestamp", "weather_condition", "temperature", "rainfall",
                 "visibility"]].rename(columns={"timestamp": "weather_hour"})
    df = df.merge(w, left_on=["weather_station_id", "weather_hour"],
                  right_on=["station_id", "weather_hour"], how="left")
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    df["time"] = df["timestamp"].dt.strftime("%H:%M")
    cal = calendar.assign(date=calendar["date"].dt.strftime("%Y-%m-%d"))
    df = df.merge(cal[["date", "public_holiday", "event_flag", "roadwork_flag"]], on="date",
                  how="left")

    # One-hot encoding: one 0/1 column per weather type (weather_Rain, weather_Fog, ...).
    # Weather types have no natural order, so they can't just be numbered like congestion.
    df = pd.concat([df, pd.get_dummies(df["weather_condition"], prefix="weather", dtype=int)],
                   axis=1)
    # weather_station_id is only a join key - PRD 8.2 drops it after the merge
    return df.drop(columns=["weather_hour", "station_id", "weather_station_id"])


# ---------------------------------------------------------------- stage 8
def final_checks(df, stats):
    """Assert the table is model-ready: no nulls where models read, sane ranges, full grid."""
    # `assert` stops the whole pipeline with a message if the condition is False.
    # Better to fail loudly here than to train models on broken data without noticing.
    must_be_complete = ["traffic_volume", "avg_speed", "occupancy", "travel_time",
                        "congestion_level", "temperature", "rainfall", "visibility",
                        "public_holiday", "event_flag"]
    nulls = df[must_be_complete].isna().sum()
    assert nulls.sum() == 0, f"nulls left in modelling columns: {nulls[nulls > 0].to_dict()}"
    assert df["traffic_volume"].ge(0).all() and df["avg_speed"].between(0, 200).all()
    assert df["occupancy"].between(0, 100).all()
    assert not df.duplicated(["road_id", "timestamp"]).any()
    days = df["timestamp"].dt.date.nunique()
    expected = df["road_id"].nunique() * days * (24 * 60 // CLEAN["window_minutes"])
    assert len(df) == expected, f"expected {expected} rows, got {len(df)}"
    stats["final_rows"] = len(df)
    log.info("stage 8: checks passed - %d rows, no nulls in modelling columns", len(df))


def write_quality_report(stats, rejects, out_path):
    """reports/data_quality.md: every defect found and what was done about it (PRD deliverable)."""
    reasons = rejects["reject_reason"].str.split(";").explode()
    reasons = reasons[reasons != ""].value_counts()
    s = stats
    filled = {c: s[f'traffic_{c}_interpolated'] + s[f'traffic_{c}_median_filled']
              for c in SENSOR_COLS}
    lines = [
        "# FlowCast - Data Quality Report",
        "",
        f"Generated by the pipeline from `{CONFIG['raw_files']['traffic']}` "
        f"(sha256 {s['raw_hash']}). Every number below is counted by the code, not typed in.",
        "",
        "## Defects found and how each was resolved",
        "",
        "| Defect (PRD 8.3) | Found | Resolution |",
        "|---|---|---|",
        f"| Duplicate log rows | {s['duplicates_exact']} exact, {s['duplicates_key']} key-level "
        "| Dropped; for key duplicates the most complete row was kept |",
        f"| Negative or missing traffic_volume | {s['volume_recovered_from_vehicle_count']} "
        "| Recovered exactly from vehicle_count (equal on every valid row) |",
        f"| avg_speed above 200 km/h or missing | {s['speed_recovered_from_travel_time']} "
        "| Recovered exactly as 60 x segment length / travel_time |",
        f"| Occupancy above 100% | {s['occupancy_capped']} | Capped at 100 |",
        f"| Statistically extreme readings (robust z > {CLEAN['zscore_flag']} vs the same segment "
        f"and time-of-day slot) | volume {s['traffic_volume_extreme_flagged']}, "
        f"speed {s['avg_speed_extreme_flagged']}, occupancy {s['occupancy_extreme_flagged']} "
        "| Flagged and kept: after the rule fixes these are mostly real events and incidents |",
        f"| Missing sensor windows (absent rows) | {s['missing_windows_rebuilt']} "
        "| Rebuilt on the full segment x 30-min grid, flagged is_imputed = 1 |",
        "| Values still missing after recovery, mostly in rebuilt windows "
        "(volume / speed / occupancy) | "
        f"{filled['traffic_volume']} / {filled['avg_speed']} / {filled['occupancy']} "
        f"| Gaps of up to {CLEAN['short_gap_windows']} windows interpolated in time; longer gaps "
        "filled with the segment-and-window median |",
        f"| Blank congestion_level | {s['congestion_derived']} "
        "| Derived from V/C = volume / (capacity / 2) using the dictionary bands |",
        f"| Inconsistent weather labels | {s['weather_labels_before']} spellings "
        f"| Mapped to {s['weather_labels_after']} labels (Clear, Cloudy, Overcast, Rain, Fog) |",
        "| Mixed date formats | weather DD/MM/YYYY vs traffic YYYY-MM-DD "
        f"| Both parsed to one timestamp, timezone {CONFIG['project']['timezone']} |",
        "| Hourly weather vs 30-min traffic | 2 windows per weather hour "
        "| Each hour's reading broadcast to both windows |",
        "",
        "## Rows quarantined at ingestion (M1)",
        "",
        "Kept in `data/interim/traffic_rejects.csv` with their reason, then repaired above.",
        "",
        "| Reason | Rows |",
        "|---|---|",
        *[f"| {reason} | {count} |" for reason, count in reasons.items()],
        "",
        "## Open questions for the client",
        "",
        f"- `Overcast` appears in {s['overcast_rows']} weather rows but is not in the PRD "
        "vocabulary. "
        "Kept as its own label rather than guessing it means Cloudy.",
        f"- accident_count is positive in {s['accident_rate_pct']}% of windows. Is that the true "
        "incident rate, or are minor incidents under-reported?",
        "",
        f"Final table: {s['final_rows']:,} rows (25 segments x {s['final_rows'] // 25 // 48} days "
        "x 48 windows), no nulls in any modelling column.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def clean_traffic(traffic, weather, stats):
    """Stages 2-6 in order. Shared with the dashboard's upload tool, so both clean the same way."""
    traffic = remove_duplicates(traffic, stats)
    traffic = fix_outliers(traffic, stats)
    traffic = fill_missing_traffic(traffic, stats)
    traffic = derive_congestion(traffic, stats)
    return harmonise_categories(traffic, weather, stats)


def run(traffic, weather, calendar, traffic_rejects):
    """Run stages 2-6 and 8 on M1's output; save the versioned table and the quality report."""
    stats = {"raw_hash": file_hash(path("raw", CONFIG["raw_files"]["traffic"]))}
    traffic, weather = clean_traffic(traffic, weather, stats)
    # PRD 16: data/interim holds the cleaned, pre-merge tables
    for name, table in [("traffic", traffic), ("weather", weather), ("calendar", calendar)]:
        table.to_csv(path("interim", f"{name}_clean.csv"), index=False)
    merged = merge_sources(traffic, weather, calendar, stats)
    final_checks(merged, stats)

    version = CONFIG["project"]["version"]
    out = path("processed", f"flowcast_clean_v{version}.csv")
    merged.to_csv(out, index=False)
    manifest = {"version": version, "file": out.name, "rows": len(merged),
                "sha256": file_hash(out), "raw_sha256": stats["raw_hash"], "stats": stats}
    path("processed", "manifest.json").write_text(json.dumps(manifest, indent=2))
    write_quality_report(stats, traffic_rejects, path("reports", "data_quality.md"))
    log.info("saved %s and reports/data_quality.md", out.name)
    return merged
