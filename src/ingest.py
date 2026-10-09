"""
M1 - Ingestion & Validation (PRD FR-01, FR-02, module M1, cleaning stage 1).

Loads the three raw tables, enforces the schema from the data dictionary,
parses both date formats into one timezone-aware timestamp, and range-checks
numeric fields. Rows that break a rule are quarantined with a reason in
data/interim/ - never silently dropped (PRD NFR: 0 silent drops). Quarantined
traffic rows still go to M2, which repairs the bad field instead of losing
the whole row.
"""

import pandas as pd

from utils import CONFIG, get_logger, path

log = get_logger("M1")
VAL = CONFIG["validation"]
TZ = CONFIG["project"]["timezone"]
UNRECOVERABLE = "bad_road_id|bad_timestamp|unknown_station|bad_capacity"

SCHEMA = {
    "traffic": {
        "road_id": "string",
        "road_name": "string",
        "latitude": "float",
        "longitude": "float",
        "weather_station_id": "string",
        "date": "string",
        "time": "string",
        "traffic_volume": "float",
        "vehicle_count": "float",
        "vehicle_type_dist": "string",
        "avg_speed": "float",
        "occupancy": "float",
        "congestion_level": "string",
        "travel_time": "float",
        "accident_count": "float",
        "signal_timing": "float",
        "road_capacity": "float",
    },
    "weather": {
        "station_id": "string",
        "date": "string",
        "time": "string",
        "weather_condition": "string",
        "temperature": "float",
        "rainfall": "float",
        "visibility": "float",
    },
    "calendar": {
        "date": "string",
        "public_holiday": "float",
        "holiday_name": "string",
        "event_flag": "float",
        "event_name": "string",
        "roadwork_flag": "float",
    },
}


def enforce_schema(df, table):
    """Check every documented column is present, then cast to the documented types."""
    missing = [c for c in SCHEMA[table] if c not in df.columns]
    if missing:
        raise ValueError(f"{table}: missing columns {missing}")
    extra = [c for c in df.columns if c not in SCHEMA[table]]
    if extra:
        log.info("%s: %d undocumented column(s) ignored: %s", table, len(extra), extra)
    df = df[list(SCHEMA[table])].copy()
    for col, kind in SCHEMA[table].items():
        if kind == "float":
            df[col] = pd.to_numeric(
                df[col], errors="coerce"
            )  # unparseable text becomes NaN, caught below
        else:
            df[col] = df[col].astype("string")
    return df


def make_timestamp(date, time, date_format):
    """Join a date and an HH:MM time into one tz-aware timestamp (PRD 9.1 stage 1)."""
    ts = pd.to_datetime(date + " " + time, format=f"{date_format} %H:%M", errors="coerce")
    return ts.dt.tz_localize(TZ)


def split_rejects(df, reasons, table):
    """Separate rows with any reason and log why. run() writes them to data/interim/."""
    bad = reasons != ""
    rejected = df[bad].assign(reject_reason=reasons[bad])
    if len(rejected):
        for reason, count in (
            reasons[bad].str.split(";").explode().loc[lambda s: s != ""].value_counts().items()
        ):
            log.info("  %s rejected for %s: %d", table, reason, count)
    log.info("%s: %d rows, %d valid, %d quarantined", table, len(df), (~bad).sum(), bad.sum())
    return df[~bad], rejected


def validate_traffic(df):
    """Schema, timestamp and range checks for the traffic log."""
    df = enforce_schema(df, "traffic")
    df["timestamp"] = make_timestamp(df["date"], df["time"], "%Y-%m-%d")
    reasons = pd.Series("", index=df.index)
    rules = {
        "bad_road_id": ~df["road_id"].str.match(VAL["road_id_pattern"], na=False),
        "bad_timestamp": df["timestamp"].isna(),
        "unknown_station": ~df["weather_station_id"].isin(VAL["stations"]),
        "negative_traffic_volume": df["traffic_volume"] < 0,
        "avg_speed_over_max": df["avg_speed"] > VAL["max_speed_kmh"],
        "occupancy_over_max": df["occupancy"] > VAL["max_occupancy_pct"],
        "bad_capacity": ~(df["road_capacity"] > 0),
        "negative_accident_count": df["accident_count"] < 0,
    }
    for name, mask in rules.items():
        reasons[mask.fillna(False)] += (
            name + ";"
        )  # NaN comparisons count as "no match", not as a failure
    return df, split_rejects(df, reasons, "traffic")[1]


def validate_weather(df):
    """Schema, DD/MM/YYYY timestamp and range checks for weather."""
    df = enforce_schema(df, "weather")
    df["timestamp"] = make_timestamp(df["date"], df["time"], "%d/%m/%Y")  # dictionary: DD/MM/YYYY
    reasons = pd.Series("", index=df.index)
    rules = {
        "unknown_station": ~df["station_id"].isin(VAL["stations"]),
        "bad_timestamp": df["timestamp"].isna(),
        "negative_rainfall": df["rainfall"] < 0,
        "negative_visibility": df["visibility"] < 0,
    }
    for name, mask in rules.items():
        reasons[mask.fillna(False)] += name + ";"
    valid, rejected = split_rejects(df, reasons, "weather")
    return valid, rejected


def validate_calendar(df):
    """Schema, date and 0/1 flag checks for the calendar."""
    df = enforce_schema(df, "calendar")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    reasons = pd.Series("", index=df.index)
    reasons[df["date"].isna()] += "bad_date;"
    for flag in ["public_holiday", "event_flag", "roadwork_flag"]:
        reasons[~df[flag].isin([0, 1])] += f"bad_{flag};"
    valid, rejected = split_rejects(df, reasons, "calendar")
    return valid, rejected


def run():
    """Validate all three raw tables and write the quarantine files."""
    files = CONFIG["raw_files"]
    traffic, traffic_rejects = validate_traffic(pd.read_csv(path("raw", files["traffic"])))
    weather, weather_rejects = validate_weather(pd.read_csv(path("raw", files["weather"])))
    calendar, calendar_rejects = validate_calendar(pd.read_csv(path("raw", files["calendar"])))
    for table, rejects in [
        ("traffic", traffic_rejects),
        ("weather", weather_rejects),
        ("calendar", calendar_rejects),
    ]:
        out = path("interim", f"{table}_rejects.csv")
        if len(rejects):
            rejects.to_csv(out, index=False)
        elif out.exists():
            out.unlink()  # no stale reject file from an older run
    # Flagged traffic rows stay in: M2 repairs the bad field. Only rows that can't be placed
    # on the map or the clock are dropped - the reject file still records them.
    fatal = traffic_rejects["reject_reason"].str.contains(UNRECOVERABLE)
    traffic = traffic.drop(traffic_rejects.index[fatal])
    log.info(
        "traffic: %d unrecoverable rows dropped, %d flagged rows passed on for repair",
        fatal.sum(),
        (~fatal).sum(),
    )
    return traffic, weather, calendar, traffic_rejects


if __name__ == "__main__":
    run()
