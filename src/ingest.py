"""
ingest.py - M1: load the raw data and check it before anything else touches it.

WHAT IT DOES, IN ORDER
    1. Read the three CSV files from data/raw/ (traffic, weather, calendar).
    2. Check every expected column is there, and give each one the right type
       (numbers become numbers; text that can't be read as a number becomes NaN).
    3. Turn the separate "date" and "time" text columns into one real timestamp.
       Careful: traffic writes dates as 2025-03-04, weather as 04/03/2025.
    4. Test every row against simple rules, e.g. "volume can't be negative",
       "speed can't be over 200 km/h". A row that breaks a rule gets a reason.
    5. Save the rule-breaking rows to data/interim/*_rejects.csv, with their reason.

KEY IDEA FOR BEGINNERS: quarantine, don't delete
    A bad row is set aside with a note, never silently thrown away. If something looks
    wrong weeks later, you can still see exactly which rows were flagged and why.
    Most flagged traffic rows are passed on to clean.py, which repairs the one bad value
    and keeps the rest of the row.

TRY IT
    python flowcast/src/ingest.py     prints how many rows passed and failed each rule
"""

import pandas as pd

from utils import CONFIG, get_logger, path

log = get_logger("M1")
VAL = CONFIG["validation"]
TZ = CONFIG["project"]["timezone"]
# Problems we can't repair: if we don't know which road or which time a reading belongs to,
# the row is useless. Everything else (a bad speed, a negative count) can be fixed in M2.
UNRECOVERABLE = "bad_road_id|bad_timestamp|unknown_station|bad_capacity"

# The expected columns, in order, copied from the data dictionary.
# NUMBER columns become floats; every other column is text.
SCHEMA = {
    "traffic": ["road_id", "road_name", "latitude", "longitude", "weather_station_id", "date",
                "time", "traffic_volume", "vehicle_count", "vehicle_type_dist", "avg_speed",
                "occupancy", "congestion_level", "travel_time", "accident_count", "signal_timing",
                "road_capacity"],
    "weather": ["station_id", "date", "time", "weather_condition", "temperature", "rainfall",
                "visibility"],
    "calendar": ["date", "public_holiday", "holiday_name", "event_flag", "event_name",
                 "roadwork_flag"],
}
NUMBER = {"latitude", "longitude", "traffic_volume", "vehicle_count", "avg_speed", "occupancy",
          "travel_time", "accident_count", "signal_timing", "road_capacity", "temperature",
          "rainfall", "visibility", "public_holiday", "event_flag", "roadwork_flag"}


def enforce_schema(df, table):
    """Check every documented column is present, then cast to the documented types."""
    missing = [c for c in SCHEMA[table] if c not in df.columns]
    if missing:
        raise ValueError(f"{table}: missing columns {missing}")
    extra = [c for c in df.columns if c not in SCHEMA[table]]
    if extra:
        log.info("%s: %d undocumented column(s) ignored: %s", table, len(extra), extra)
    df = df[SCHEMA[table]].copy()
    for col in SCHEMA[table]:
        if col in NUMBER:  # unparseable text becomes NaN, caught by the rules below
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = df[col].astype("string")
    return df


def rule_reasons(df, rules):
    """
    One text box per row, listing every rule the row breaks, e.g. "negative_traffic_volume;".
    An empty box means the row passed every rule.
    """
    reasons = pd.Series("", index=df.index)
    for name, mask in rules.items():
        # A missing value can't be "negative" or "too fast", so fillna(False) treats it as
        # passing here. Missing values are a different problem, handled in clean.py.
        reasons[mask.fillna(False)] += name + ";"
    return reasons


def make_timestamp(date, time, date_format):
    """Join a date and an HH:MM time into one tz-aware timestamp (PRD 9.1 stage 1)."""
    ts = pd.to_datetime(date + " " + time, format=f"{date_format} %H:%M", errors="coerce")
    return ts.dt.tz_localize(TZ)


def split_rejects(df, reasons, table):
    """Separate rows with any reason and log why. run() writes them to data/interim/."""
    bad = reasons != ""
    counts = reasons[bad].str.split(";").explode().loc[lambda s: s != ""].value_counts()
    for reason, count in counts.items():
        log.info("  %s rejected for %s: %d", table, reason, count)
    log.info("%s: %d rows, %d valid, %d quarantined", table, len(df), (~bad).sum(), bad.sum())
    return df[~bad], df[bad].assign(reject_reason=reasons[bad])


def validate_traffic(df):
    """Schema, timestamp and range checks for the traffic log."""
    df = enforce_schema(df, "traffic")
    df["timestamp"] = make_timestamp(df["date"], df["time"], "%Y-%m-%d")
    # Each rule is a "mask": a True/False column, True where the row breaks the rule.
    # Writing df["traffic_volume"] < 0 checks all 178,468 rows at once - no loop needed.
    reasons = rule_reasons(df, {
        "bad_road_id": ~df["road_id"].str.match(VAL["road_id_pattern"], na=False),
        "bad_timestamp": df["timestamp"].isna(),
        "unknown_station": ~df["weather_station_id"].isin(VAL["stations"]),
        "negative_traffic_volume": df["traffic_volume"] < 0,
        "avg_speed_over_max": df["avg_speed"] > VAL["max_speed_kmh"],
        "occupancy_over_max": df["occupancy"] > VAL["max_occupancy_pct"],
        "bad_capacity": ~(df["road_capacity"] > 0),
        "negative_accident_count": df["accident_count"] < 0,
    })
    # every row goes on (M2 repairs them); only the rejects list is split off
    return df, split_rejects(df, reasons, "traffic")[1]


def validate_weather(df):
    """Schema, DD/MM/YYYY timestamp and range checks for weather."""
    df = enforce_schema(df, "weather")
    # Weather writes dates day-first (04/03/2025 = 4 March). Reading it as month-first would
    # silently turn it into 3 April, and the weather would join to the wrong traffic.
    df["timestamp"] = make_timestamp(df["date"], df["time"], "%d/%m/%Y")
    reasons = rule_reasons(df, {
        "unknown_station": ~df["station_id"].isin(VAL["stations"]),
        "bad_timestamp": df["timestamp"].isna(),
        "negative_rainfall": df["rainfall"] < 0,
        "negative_visibility": df["visibility"] < 0,
    })
    return split_rejects(df, reasons, "weather")


def validate_calendar(df):
    """Schema, date and 0/1 flag checks for the calendar."""
    df = enforce_schema(df, "calendar")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    flags = ["public_holiday", "event_flag", "roadwork_flag"]
    reasons = rule_reasons(df, {"bad_date": df["date"].isna(),
                                **{f"bad_{f}": ~df[f].isin([0, 1]) for f in flags}})
    return split_rejects(df, reasons, "calendar")


def run():
    """Validate all three raw tables and write the quarantine files."""
    files = CONFIG["raw_files"]
    traffic, traffic_rejects = validate_traffic(pd.read_csv(path("raw", files["traffic"])))
    weather, weather_rejects = validate_weather(pd.read_csv(path("raw", files["weather"])))
    calendar, calendar_rejects = validate_calendar(pd.read_csv(path("raw", files["calendar"])))
    for table, rejects in [("traffic", traffic_rejects), ("weather", weather_rejects),
                           ("calendar", calendar_rejects)]:
        out = path("interim", f"{table}_rejects.csv")
        if len(rejects):
            rejects.to_csv(out, index=False)
        elif out.exists():
            out.unlink()  # no stale reject file from an older run
    # Flagged traffic rows stay in: M2 repairs the bad field. Only rows that can't be placed
    # on the map or the clock are dropped - the reject file still records them.
    fatal = traffic_rejects["reject_reason"].str.contains(UNRECOVERABLE)
    traffic = traffic.drop(traffic_rejects.index[fatal])
    log.info("traffic: %d unrecoverable rows dropped, %d flagged rows passed on for repair",
             fatal.sum(), (~fatal).sum())
    return traffic, weather, calendar, traffic_rejects


if __name__ == "__main__":
    run()
