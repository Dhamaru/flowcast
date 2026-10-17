"""
M7 - FlowCast Analytics Dashboard (PRD FR-13 to FR-18, section 14).

Nine views (PRD 14.1) and four tools (PRD 14.2). Reads only what the pipeline
saved; it never trains, except when a reviewer presses Retrain.

    streamlit run flowcast/dashboard/app.py
"""

import json
import pathlib
import pickle
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import clean  # noqa: E402
import features  # noqa: E402
import ingest  # noqa: E402
from utils import CONFIG, path  # noqa: E402

LEVELS = CONFIG["cleaning"]["congestion_levels"]
# PRD 14.3: colour encodes severity the same way in every view. These four colours mean
# congestion severity and nothing else; other highlights use the non-severity colours below.
COLORS = {"Free-flow": "#3fb67a", "Moderate": "#e8c13a", "Heavy": "#e8873a", "Severe": "#e0543f"}
BASE, HIGHLIGHT, TARGET, ACTUAL = "#9cbfe0", "#2c5f8f", "#6b7785", "#8e6cc1"


def severity_cell(value):
    """CSS for one table cell: tinted with its congestion level's colour."""
    for level, hex_color in COLORS.items():
        if isinstance(value, str) and value.startswith(level):
            r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
            return f"background-color: rgba({r},{g},{b},0.35); font-weight: 600"
    return ""


def with_severity(frame, columns):
    """Styled table with congestion columns coloured by severity."""
    return frame.style.map(severity_cell, subset=[c for c in columns if c in frame])


HORIZONS = CONFIG["horizons"]
VERSION_FILE = f"flowcast_features_v{CONFIG['project']['version']}.parquet"

ROLES = {  # PRD 6 roles. 14.3: analysts land on the heatmap, planners on road comparison
    "Traffic Operations Analyst": "Congestion heatmap",
    "Incident Response Coordinator": "Live prediction",
    "Transport Planner": "Road comparison",
    "System Owner / Reviewer": "Model performance",
}
VIEWS = [
    "Live prediction",
    "Historical trends",
    "Congestion heatmap",
    "Road comparison",
    "Model performance",
    "Feature importance",
    "Forecast visualisation",
    "Prediction confidence",
    "Weather vs traffic",
]
TOOLS = ["Forecast a segment", "Upload data", "Retrain models", "Reports & insights"]


# ------------------------------------------------------------------ loading
@st.cache_data
def load_table():
    """The versioned feature table written by M3."""
    return pd.read_parquet(path("processed", VERSION_FILE))


@st.cache_data
def load_test_predictions():
    """Test-window predictions with their ranges, written by M5."""
    return pd.read_parquet(path("reports", "test_predictions.parquet"))


@st.cache_data
def load_json(name):
    """A JSON report from reports/, or None if the pipeline hasn't written it yet."""
    p = path("reports", name)
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_resource
def load_bundles():
    """Every saved model bundle, keyed by (target, horizon) or by name."""
    bundles = {}
    for h in HORIZONS:
        for target in ["traffic_volume", "avg_speed"]:
            with open(path("models", f"{target}_h{h}.pkl"), "rb") as f:
                bundles[(target, h)] = pickle.load(f)
    for name in ["congestion", "accident"]:
        with open(path("models", f"{name}_h1.pkl"), "rb") as f:
            bundles[name] = pickle.load(f)
    return bundles


@st.cache_data
def horizon_matrix(h):
    """Feature matrix for horizon h over the whole table (cached)."""
    return features.horizon_features(load_table(), h)


def risk_probability(bundle, X):
    """Calibrated accident probability for each row."""
    model = bundle["model"]
    raw = (
        model.predict_proba(X)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X)
    )
    return bundle["calibrator"].predict_proba(raw.reshape(-1, 1))[:, 1]


def predict_rows(df, X, h, bundles):
    """All targets for these rows at horizon h, each stamped with its model version (lineage)."""
    vol, spd = bundles[("traffic_volume", h)], bundles[("avg_speed", h)]
    hours = df["timestamp"].dt.hour
    out = pd.DataFrame(index=df.index)
    out["volume"] = vol["model"].predict(X)
    out["volume_band"] = hours.map(vol["band_by_hour"]).to_numpy()
    out["speed"] = spd["model"].predict(X)
    out["travel_time"] = 60 * df["segment_length_km"] / out["speed"]
    out["travel_time_band"] = hours.map(spd["travel_time_band_by_hour"]).to_numpy()
    if h == 1:
        probs = bundles["congestion"]["model"].predict_proba(X)
        out["congestion"] = [LEVELS[i] for i in probs.argmax(1)]
        out["congestion_confidence"] = probs.max(1)
        out["accident_probability"] = risk_probability(bundles["accident"], X)
        out["alert"] = out["accident_probability"] >= bundles["accident"]["threshold"]
    else:  # classifiers cover the next window only; later ones use V/C of predicted volume
        vc = out["volume"] / (df["road_capacity"] / 2)
        out["congestion"] = pd.cut(
            vc,
            [-np.inf, *CONFIG["cleaning"]["congestion_bands"], np.inf],
            right=False,
            labels=LEVELS,
        ).astype(str)
    out["model_version"] = vol["version"]
    return out


def bars(frame, x, y, title, target=None, highlight=None, lower_is_better=True):
    """Bar chart of one metric per model, winner highlighted, optional PRD target line."""
    colors = [HIGHLIGHT if m == highlight else BASE for m in frame[x]]
    fig = go.Figure(
        go.Bar(
            x=frame[x],
            y=frame[y],
            marker_color=colors,
            text=frame[y].round(3),
            textposition="outside",
        )
    )
    if target is not None:
        fig.add_hline(
            y=target, line_dash="dash", line_color=TARGET, annotation_text="PRD target"
        )
    fig.update_layout(title=title, height=340, margin=dict(t=50))
    return fig


# ------------------------------------------------------------------ shell
st.set_page_config(page_title="FlowCast", layout="wide")
needed = [
    path("processed", VERSION_FILE),
    path("reports", "test_predictions.parquet"),
    path("reports", "scoreboard.json"),
    path("models", "accident_h1.pkl"),
]
if not all(p.exists() for p in needed):
    st.error("Pipeline outputs are missing. Run `python flowcast/run_pipeline.py` first.")
    st.stop()

df = load_table()
board = load_json("scoreboard.json")
dl = load_json("dl_result.json")
bundles = load_bundles()
tz = df["timestamp"].dt.tz
test_from = pd.Timestamp(board["split"]["test_from"])
road_ids = sorted(df["road_id"].unique())
names = df.drop_duplicates("road_id").set_index("road_id")["road_name"].to_dict()


def label(r):
    """Segment id with its name, for dropdowns and tables."""
    return f"{r} · {names[r]}"


st.sidebar.title("FlowCast")
role = st.sidebar.selectbox("Your role", list(ROLES))
section = st.sidebar.radio("Section", ["Views", "Tools"], horizontal=True)
options = VIEWS if section == "Views" else TOOLS
default = options.index(ROLES[role]) if ROLES[role] in options else 0
view = st.sidebar.radio("Go to", options, index=default, label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption(
    f"Model version `{board['version']}`  \nTrained on data before "
    f"{pd.Timestamp(board['split']['train_before']):%d %b %Y}. Predictions shown are for the "
    f"test window from {test_from:%d %b}, which no model saw."
)


# ------------------------------------------------------------------ views
if view == "Live prediction":
    st.header("Live prediction")
    st.caption(
        "Next 30-minute window for every segment, with confidence. Sorted by accident risk."
    )
    test = df[(df["timestamp"] >= test_from) & (df["is_imputed"] == 0)]
    days = sorted(test["timestamp"].dt.date.unique())
    c1, c2 = st.columns(2)
    day = c1.selectbox(
        "Date", days, index=len(days) // 2, format_func=lambda d: f"{d:%a %d %b %Y}"
    )
    times = sorted(
        test.loc[test["timestamp"].dt.date == day, "timestamp"].dt.strftime("%H:%M").unique()
    )
    hhmm = c2.selectbox("Window", times, index=times.index("08:00") if "08:00" in times else 0)
    rows = df[df["timestamp"] == pd.Timestamp(f"{day} {hhmm}").tz_localize(tz)]
    X = horizon_matrix(1).loc[rows.index]
    rows, X = rows[X.notna().all(axis=1)], X.dropna()
    start = time.perf_counter()
    p = predict_rows(rows, X, 1, bundles)
    ms = (time.perf_counter() - start) * 1000
    table = pd.DataFrame(
        {
            "segment": rows["road_id"].map(label),
            "volume": p["volume"].round(0),
            "± 80%": p["volume_band"].round(0),
            "actual volume": rows["traffic_volume"],
            "speed km/h": p["speed"].round(1),
            "travel time min": p["travel_time"].round(2),
            "congestion": p["congestion"],
            "confidence": (p["congestion_confidence"] * 100).round(0).astype(int).astype(str)
            + "%",
            "actual congestion": rows["congestion_level"],
            "accident probability": (p["accident_probability"] * 100).round(2),
            "alert": np.where(p["alert"], "ALERT", ""),
            "model version": p["model_version"],
        }
    ).sort_values("accident probability", ascending=False)
    m = st.columns(4)
    m[0].metric(
        "Corridor volume",
        f"{p['volume'].sum():,.0f}",
        f"actual {rows['traffic_volume'].sum():,.0f}",
        delta_color="off",
    )
    m[1].metric("Heavy or Severe", int(p["congestion"].isin(["Heavy", "Severe"]).sum()))
    m[2].metric(
        "Accident alerts",
        int(p["alert"].sum()),
        help=f"Alert at probability ≥ {bundles['accident']['threshold']:.3f} (best F1 on val).",
    )
    m[3].metric("Inference", f"{ms:.0f} ms", help="All 25 segments, all models. PRD limit: 30 s.")
    st.dataframe(
        with_severity(table, ["congestion", "actual congestion"]),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Accident probability is calibrated: across the test window it averages "
        f"{board['accident']['test']['mean_predicted_probability'] * 100:.2f}% vs a real rate of "
        f"{board['accident']['test']['base_rate'] * 100:.2f}%."
    )

elif view == "Historical trends":
    st.header("Historical trends")
    c1, c2 = st.columns([1, 2])
    road = c1.selectbox("Segment", road_ids, format_func=label)
    lo, hi = df["timestamp"].dt.date.min(), df["timestamp"].dt.date.max()
    span = c2.date_input("Dates", (hi - pd.Timedelta(days=13), hi), min_value=lo, max_value=hi)
    seg = df[df["road_id"] == road]
    if len(span) == 2:
        part = seg[seg["timestamp"].dt.date.between(*span)]
        fig = px.line(
            part,
            x="timestamp",
            y=["traffic_volume", "avg_speed"],
            labels={"value": "", "variable": ""},
        )
        fig.update_layout(height=360, legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
    prof = seg.assign(
        hour=seg["timestamp"].dt.hour, day=np.where(seg["is_weekend"] == 1, "weekend", "weekday")
    )
    prof = prof.groupby(["day", "hour"])["traffic_volume"].mean().reset_index()
    st.plotly_chart(
        px.line(
            prof,
            x="hour",
            y="traffic_volume",
            color="day",
            markers=True,
            title="Peak-hour pattern, Jan–May",
            labels={"traffic_volume": "vehicles / 30 min"},
        ),
        use_container_width=True,
    )

elif view == "Congestion heatmap":
    st.header("Congestion heatmap")
    st.caption("Segment × 30-minute window for one day. Rush hours show up as vertical bands.")
    days = sorted(df["timestamp"].dt.date.unique())
    day = st.selectbox("Date", days, index=len(days) - 1, format_func=lambda d: f"{d:%a %d %b %Y}")
    d = df[df["timestamp"].dt.date == day].assign(
        window=lambda x: x["timestamp"].dt.strftime("%H:%M")
    )
    grid = d.pivot_table(
        index="road_id", columns="window", values="congestion_code", aggfunc="first"
    )
    c = [COLORS[lvl] for lvl in LEVELS]
    scale = [
        [0, c[0]],
        [1 / 6, c[0]],
        [1 / 6, c[1]],
        [1 / 2, c[1]],
        [1 / 2, c[2]],
        [5 / 6, c[2]],
        [5 / 6, c[3]],
        [1, c[3]],
    ]
    fig = go.Figure(
        go.Heatmap(
            z=grid.values,
            x=list(grid.columns),
            y=[label(r) for r in grid.index],
            zmin=0,
            zmax=3,
            colorscale=scale,
            colorbar=dict(tickvals=[0, 1, 2, 3], ticktext=LEVELS),
        )
    )
    fig.update_layout(height=640, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"{int(d['is_imputed'].sum())} of {len(d)} cells this day are rebuilt windows.")

elif view == "Road comparison":
    st.header("Road comparison")
    picked = st.multiselect("Segments", road_ids, default=road_ids[:6], format_func=label)
    if picked:
        real = df[df["road_id"].isin(picked) & (df["is_imputed"] == 0)]
        stats = (
            real.groupby("road_id")
            .agg(
                volume=("traffic_volume", "mean"),
                speed=("avg_speed", "mean"),
                heavy_or_severe_pct=("congestion_code", lambda s: (s >= 2).mean() * 100),
                travel_time_spread=("travel_time", "std"),
            )
            .reset_index()
        )
        mix = (
            real.groupby("road_id")["congestion_level"]
            .value_counts(normalize=True)
            .mul(100)
            .rename("share")
            .reset_index()
        )
        fig = px.bar(
            mix,
            x="road_id",
            y="share",
            color="congestion_level",
            color_discrete_map=COLORS,
            category_orders={"congestion_level": LEVELS},
            title="Congestion mix (% of windows)",
            labels={"share": "", "road_id": "", "congestion_level": ""},
        )
        st.plotly_chart(fig, use_container_width=True)
        cols = st.columns(3)
        for col, (field, title) in zip(
            cols,
            [
                ("volume", "Mean volume"),
                ("speed", "Mean speed km/h"),
                ("travel_time_spread", "Reliability: travel-time spread (min)"),
            ],
        ):
            col.plotly_chart(
                px.bar(
                    stats, x="road_id", y=field, title=title, labels={field: "", "road_id": ""}
                ),
                use_container_width=True,
            )
        st.dataframe(stats.round(2), hide_index=True, use_container_width=True)

elif view == "Model performance":
    st.header("Model performance")
    st.caption(
        "Each model was tuned with time-series cross-validation on the training window and "
        "chosen on that CV score. The test scores below come from the untouched test window."
    )
    vol = board["volume"]
    cand = pd.DataFrame(
        [
            {
                "model": c["model"],
                "rmse": c["test"]["rmse"],
                "cv": c["cv_rmse"],
                "std": c["cv_std"],
            }
            for c in vol["candidates"]
        ]
    )
    extra = [{"model": "Scratch NumPy LinReg", "rmse": vol["scratch_linreg"]["rmse"]}]
    if dl:
        extra += [{"model": k, "rmse": v["test"]["rmse"]} for k, v in dl["variants"].items()]
    allv = pd.concat([cand, pd.DataFrame(extra)], ignore_index=True).sort_values("rmse")
    a, b = st.columns(2)
    a.plotly_chart(
        bars(allv, "model", "rmse", "Traffic volume · test RMSE", highlight=vol["selected"]),
        use_container_width=True,
    )
    h = pd.DataFrame([{"horizon": f"{int(k) * 30} min", **v} for k, v in vol["horizons"].items()])
    b.plotly_chart(
        px.bar(
            h,
            x="horizon",
            y="rmse",
            text_auto=".1f",
            title=f"{vol['selected']} RMSE by forecast horizon",
        ),
        use_container_width=True,
    )
    c, d_ = st.columns(2)
    cong = board["congestion"]
    cf = pd.DataFrame(
        [{"model": x["model"], "macro_f1": x["test"]["macro_f1"]} for x in cong["candidates"]]
    )
    c.plotly_chart(
        bars(cf, "model", "macro_f1", "Congestion · test macro-F1", 0.80, cong["selected"]),
        use_container_width=True,
    )
    acc = board["accident"]
    af = pd.DataFrame(
        [{"model": x["model"], "roc_auc": x["test_roc_auc"]} for x in acc["candidates"]]
    )
    d_.plotly_chart(
        bars(af, "model", "roc_auc", "Accident risk · test ROC-AUC", 0.75, acc["selected"]),
        use_container_width=True,
    )

    st.subheader("Acceptance check · PRD 3.3")
    best_vol = next(x for x in vol["candidates"] if x["model"] == vol["selected"])["test"]
    rows = [
        (
            "Volume forecast accuracy",
            "MAPE ≤ 12%",
            f"{best_vol['mape']:.2f}%",
            best_vol["mape"] <= 12,
        ),
        (
            "Congestion classification",
            "Macro-F1 ≥ 0.80",
            f"{cong['test']['macro_f1']:.3f}",
            cong["test"]["macro_f1"] >= 0.80,
        ),
        (
            "Accident-risk ranking",
            "ROC-AUC ≥ 0.75",
            f"{acc['test']['roc_auc']:.3f}",
            acc["test"]["roc_auc"] >= 0.75,
        ),
    ]
    if dl:
        best_dl = min(dl["variants"].items(), key=lambda kv: kv[1]["test"]["rmse"])
        rows.append(
            (
                "DL vs ML benchmark",
                "Sequence model beats best classical RMSE",
                f"{best_dl[0]} {best_dl[1]['test']['rmse']:.2f} vs {best_vol['rmse']:.2f}",
                best_dl[1]["test"]["rmse"] < best_vol["rmse"],
            )
        )
    st.dataframe(
        pd.DataFrame(rows, columns=["criterion", "target", "result", "met"]),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Error analysis · PRD 13.3")
    eb = vol["error_breakdown"]
    e1, e2, e3 = st.columns(3)
    for col, key, title in [
        (e1, "hour", "RMSE by hour"),
        (e2, "day_of_week", "RMSE by day"),
        (e3, "weather", "RMSE by weather"),
    ]:
        s = pd.Series(eb[key])
        col.plotly_chart(
            px.bar(x=s.index, y=s.values, title=title, labels={"x": "", "y": "RMSE"}),
            use_container_width=True,
        )
    # cell shading = row count (blues); the class labels carry the severity colours
    tinted = [f"<span style='color:{COLORS[lvl]}'><b>{lvl}</b></span>" for lvl in LEVELS]
    cm = pd.DataFrame(cong["test"]["confusion_matrix"], index=tinted, columns=tinted)
    st.plotly_chart(
        px.imshow(
            cm,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="predicted", y="actual", color="windows"),
            title="Congestion confusion matrix",
        ),
        use_container_width=True,
    )
    t = acc["test"]
    st.caption(
        f"Accident alerts at threshold {t['threshold']:.3f}: precision {t['precision']:.2f}, "
        f"recall {t['recall']:.2f}, {t['alerts_per_1000_windows']:.0f} alerts per 1,000 windows."
    )

elif view == "Feature importance":
    st.header("Feature importance")
    st.caption(
        f"What the volume model ({board['volume']['selected']}) relies on. Importance shows "
        "model uses to predict, not what causes traffic."
    )
    imp = pd.DataFrame(board["volume"]["feature_importance"]).head(15).sort_values("importance")
    st.plotly_chart(
        px.bar(imp, x="importance", y="feature", orientation="h", height=520),
        use_container_width=True,
    )

elif view == "Forecast visualisation":
    st.header("Forecast visualisation")
    tp = load_test_predictions()
    c1, c2, c3, c4 = st.columns(4)
    road = c1.selectbox("Segment", road_ids, format_func=label)
    target = c2.selectbox("Target", ["traffic_volume", "avg_speed", "travel_time"])
    h = c3.selectbox("Horizon", HORIZONS, format_func=lambda x: f"{x * 30} min ahead")
    lo = tp["timestamp"].dt.date.min()
    span = c4.date_input(
        "Dates",
        (lo, lo + pd.Timedelta(days=2)),
        min_value=lo,
        max_value=tp["timestamp"].dt.date.max(),
    )
    if len(span) == 2:
        t = tp[
            (tp["road_id"] == road)
            & (tp["target"] == target)
            & (tp["horizon"] == h)
            & tp["timestamp"].dt.date.between(*span)
        ].sort_values("timestamp")
        fig = go.Figure(
            [
                go.Scatter(
                    x=t["timestamp"],
                    y=t["upper"],
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                go.Scatter(
                    x=t["timestamp"],
                    y=t["lower"],
                    fill="tonexty",
                    fillcolor="rgba(90,167,224,0.2)",
                    line=dict(width=0),
                    name="80% range",
                ),
                go.Scatter(
                    x=t["timestamp"], y=t["pred"], name="predicted", line=dict(color="#5aa7e0")
                ),
                go.Scatter(
                    x=t["timestamp"], y=t["actual"], name="actual", line=dict(color=ACTUAL)
                ),
            ]
        )
        fig.update_layout(height=420, legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)
        if len(t):
            st.metric(
                "RMSE, this segment and range",
                f"{np.sqrt(((t['actual'] - t['pred']) ** 2).mean()):.2f}",
            )

elif view == "Prediction confidence":
    st.header("Prediction confidence")
    st.caption(
        "Ranges are sized on the validation slice, then checked on the test window. "
        "A range that claims "
        "80% should contain the real value about 80% of the time."
    )
    rows = []
    for target, key in [("traffic_volume", "volume"), ("avg_speed", "speed")]:
        for hh, v in board[key]["horizons"].items():
            rows.append(
                {
                    "target": target,
                    "horizon": f"{int(hh) * 30} min",
                    "coverage %": v["coverage"] * 100,
                    "avg ±": v["avg_half_width"],
                }
            )
    for hh, v in board["travel_time"].items():
        rows.append(
            {
                "target": "travel_time",
                "horizon": f"{int(hh) * 30} min",
                "coverage %": v["coverage"] * 100,
                "avg ±": None,
            }
        )
    cov = pd.DataFrame(rows)
    fig = px.bar(
        cov,
        x="horizon",
        y="coverage %",
        color="target",
        barmode="group",
        title="Coverage on the test window",
    )
    fig.add_hline(y=80, line_dash="dash", line_color=TARGET)
    st.plotly_chart(fig, use_container_width=True)
    vb = cov[cov["target"] == "traffic_volume"]
    st.plotly_chart(
        px.bar(vb, x="horizon", y="avg ±", title="Volume range width grows with horizon"),
        use_container_width=True,
    )
    band = pd.Series(bundles[("traffic_volume", 1)]["band_by_hour"])
    st.plotly_chart(
        px.bar(
            x=band.index,
            y=band.values,
            labels={"x": "hour", "y": "± vehicles"},
            title="Next-window volume range by hour: wider at rush hour",
        ),
        use_container_width=True,
    )

elif view == "Weather vs traffic":
    st.header("Weather vs traffic")
    peak = st.toggle(
        "Peak hours only",
        value=True,
        help="Comparing only peak hours stops time of day from muddying the weather comparison.",
    )
    w = df[(df["is_imputed"] == 0) & ((df["is_peak_period"] == 1) if peak else True)]
    by = (
        w.groupby("weather_condition")
        .agg(
            volume=("traffic_volume", "mean"),
            speed=("avg_speed", "mean"),
            windows=("traffic_volume", "size"),
        )
        .reset_index()
    )
    a, b = st.columns(2)
    a.plotly_chart(
        px.bar(by, x="weather_condition", y="volume", title="Mean volume"),
        use_container_width=True,
    )
    b.plotly_chart(
        px.bar(by, x="weather_condition", y="speed", title="Mean speed km/h"),
        use_container_width=True,
    )
    rain = (
        w.groupby(
            pd.cut(
                w["rainfall"],
                [-0.01, 0, 2, 5, np.inf],
                labels=["dry", "0–2 mm", "2–5 mm", "5+ mm"],
            ),
            observed=True,
        )["avg_speed"]
        .mean()
        .reset_index()
    )
    st.plotly_chart(
        px.bar(rain, x="rainfall", y="avg_speed", title="Mean speed by rainfall"),
        use_container_width=True,
    )
    st.dataframe(by.round(1), hide_index=True)

# ------------------------------------------------------------------ tools
elif view == "Forecast a segment":
    st.header("Forecast a segment")
    st.caption(
        "Pick a segment and the moment the forecast is issued. Each horizon has its own model."
    )
    test = df[df["timestamp"] >= test_from]
    c1, c2, c3 = st.columns(3)
    road = c1.selectbox("Segment", road_ids, format_func=label)
    days = sorted(test["timestamp"].dt.date.unique())
    day = c2.selectbox(
        "Issued on", days[:-1], index=len(days) // 2, format_func=lambda d: f"{d:%a %d %b}"
    )
    hhmm = c3.selectbox(
        "Issued at (last observed window)",
        [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 30)],
        index=14,
    )
    issued = pd.Timestamp(f"{day} {hhmm}").tz_localize(tz)
    out = []
    for h in HORIZONS:
        target_time = issued + pd.Timedelta(minutes=30 * h)
        row = df[(df["road_id"] == road) & (df["timestamp"] == target_time)]
        X = horizon_matrix(h).loc[row.index]
        if row.empty or X.isna().any(axis=1).iloc[0]:
            continue
        p = predict_rows(row, X, h, bundles).iloc[0]
        out.append(
            {
                "horizon": f"+{h * 30} min",
                "window": f"{target_time:%H:%M}",
                "volume": round(p["volume"]),
                "range": f"{p['volume'] - p['volume_band']:.0f}–"
                f"{p['volume'] + p['volume_band']:.0f}",
                "actual volume": row["traffic_volume"].iloc[0],
                "speed km/h": round(p["speed"], 1),
                "travel time min": f"{p['travel_time']:.2f} ± {p['travel_time_band']:.2f}",
                "congestion": p["congestion"] + (" (classifier)" if h == 1 else " (from volume)"),
                "model": f"traffic_volume_h{h}.pkl · {p['model_version']}",
            }
        )
    st.dataframe(
        with_severity(pd.DataFrame(out), ["congestion"]), hide_index=True, use_container_width=True
    )

elif view == "Upload data":
    st.header("Upload data")
    st.caption("Same steps as training: M1 validation, M2 cleaning, M3 features, then the models.")
    st.info(
        "Each segment needs 24 h of history before a row can be predicted "
        "(the same-time-yesterday lag). "
        "Weather is joined from the stored Jan–May 2025 observations."
    )
    file = st.file_uploader("Traffic CSV (traffic_sensor_log format)", type="csv")
    sample = st.button("Use the sample file (30–31 May 2025)")
    raw = None
    if file is not None:
        raw = pd.read_csv(file)
    elif sample:  # built from the raw log on demand, so no extra file sits outside the PRD layout
        log_ = pd.read_csv(path("raw", CONFIG["raw_files"]["traffic"]))
        raw = log_[log_["date"].isin(["2025-05-30", "2025-05-31"])]
    if raw is not None:
        try:
            traffic, rejects = ingest.validate_traffic(raw)
        except ValueError as e:
            st.error(f"This file can't be used: {e}")
            st.stop()
        weather, _ = ingest.validate_weather(
            pd.read_csv(path("raw", CONFIG["raw_files"]["weather"]))
        )
        calendar, _ = ingest.validate_calendar(
            pd.read_csv(path("raw", CONFIG["raw_files"]["calendar"]))
        )
        stats = {}
        fatal = rejects["reject_reason"].str.contains(ingest.UNRECOVERABLE)
        traffic = traffic.drop(rejects.index[fatal])
        traffic = clean.remove_duplicates(traffic, stats)
        traffic = clean.fix_outliers(traffic, stats)
        traffic = clean.fill_missing_traffic(traffic, stats)
        traffic = clean.derive_congestion(traffic, stats)
        traffic, weather = clean.harmonise_categories(traffic, weather, stats)
        merged = clean.merge_sources(traffic, weather, calendar, stats)
        built = features.build_features(merged)
        X = features.horizon_features(built, 1)
        ready = built[X.notna().all(axis=1) & (built["is_imputed"] == 0)]
        m = st.columns(4)
        m[0].metric("Rows uploaded", f"{len(raw):,}")
        m[1].metric("Quarantined & repaired", f"{int((~fatal).sum()):,}")
        m[2].metric("Missing windows rebuilt", f"{stats['missing_windows_rebuilt']:,}")
        m[3].metric("Rows predicted", f"{len(ready):,}")
        if ready.empty:
            st.warning("No row has a full day of history. Upload at least 25 hours per segment.")
        else:
            p = predict_rows(ready, X.loc[ready.index], 1, bundles)
            result = pd.concat(
                [ready[["road_id", "timestamp", "traffic_volume", "congestion_level"]], p], axis=1
            )
            st.dataframe(
                with_severity(result.round(3), ["congestion_level", "congestion"]),
                hide_index=True,
                use_container_width=True,
            )
            st.download_button(
                "Download predictions (CSV)",
                result.to_csv(index=False),
                "flowcast_predictions.csv",
            )

elif view == "Retrain models":
    st.header("Retrain models")
    st.caption(
        "Reruns M5 (and optionally M6) on the saved feature table, then reloads every view."
    )
    cards = json.loads(path("models", "model_cards.json").read_text())
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "model": c["name"],
                    "algorithm": c["algorithm"],
                    "version": c["version"],
                    "file": c["file"],
                }
                for c in cards
            ]
        ),
        hide_index=True,
        use_container_width=True,
    )
    full_table = path("processed", VERSION_FILE.replace(".parquet", ".csv"))
    if not full_table.exists():
        st.info(
            "Retraining needs the full feature table, which only exists where the pipeline has "
            "been run. On the hosted dashboard, retrain locally with "
            "`python flowcast/run_pipeline.py` and push the new models."
        )
        st.stop()
    with_dl = st.checkbox("Include the LSTM (adds ~10 min)")
    st.warning("Retraining takes several minutes and replaces the current models.")
    if st.button("Retrain now"):
        cmd = [sys.executable, str(ROOT / "run_pipeline.py"), "--from", "ml"] + (
            [] if with_dl else ["--skip-dl"]
        )
        with st.spinner("Training..."):
            done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode == 0:
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Retrained. Open Model performance to see the new scores.")
        else:
            st.error(
                "Retraining failed. Some model files may already be replaced, so fix the error "
                "in the log below and retrain before trusting the dashboard."
            )
        st.code((done.stdout + done.stderr)[-3000:])

elif view == "Reports & insights":
    st.header("Reports & insights")
    lo, hi = df["timestamp"].dt.date.min(), df["timestamp"].dt.date.max()
    span = st.date_input("Period", (hi - pd.Timedelta(days=6), hi), min_value=lo, max_value=hi)
    if len(span) == 2:
        r = df[df["timestamp"].dt.date.between(*span) & (df["is_imputed"] == 0)]
        by_road = r.groupby("road_id")["traffic_volume"].sum().sort_values(ascending=False)
        congested = (r["congestion_code"] >= 2).groupby(r["road_id"]).mean()
        hour = r.groupby(r["timestamp"].dt.hour)["traffic_volume"].mean().idxmax()
        weather = (
            r.drop_duplicates("timestamp")["weather_condition"]
            .value_counts(normalize=True)
            .mul(100)
            .round(1)
        )
        vol = next(
            x for x in board["volume"]["candidates"] if x["model"] == board["volume"]["selected"]
        )["test"]
        report = "\n".join(
            [
                f"# FlowCast corridor report: {span[0]:%d %b %Y} to {span[1]:%d %b %Y}",
                "",
                "## Traffic",
                f"- Vehicles counted: {r['traffic_volume'].sum():,.0f} "
                f"on {r['road_id'].nunique()} segments",
                f"- Busiest segment: {label(by_road.index[0])}, {by_road.iloc[0]:,.0f} vehicles",
                f"- Busiest hour on average: {hour:02d}:00",
                f"- Windows at Heavy or Severe: {(r['congestion_code'] >= 2).mean() * 100:.1f}%",
                f"- Most congested segment: {label(congested.idxmax())}, "
                f"Heavy/Severe in {congested.max() * 100:.1f}% of windows",
                "",
                "## Incidents",
                f"- Windows with a reported accident: {int((r['accident_count'] > 0).sum()):,}",
                f"- Reported accidents: {int(r['accident_count'].sum()):,}",
                "",
                "## Weather (share of windows)",
                *[f"- {k}: {v}%" for k, v in weather.items()],
                "",
                f"## Models in use (version {board['version']})",
                f"- Volume: {board['volume']['selected']}, MAPE {vol['mape']:.2f}% (target ≤ 12%)",
                f"- Congestion: {board['congestion']['selected']}, "
                f"macro-F1 {board['congestion']['test']['macro_f1']:.3f} (target ≥ 0.80)",
                f"- Accident risk: {board['accident']['selected']}, "
                f"ROC-AUC {board['accident']['test']['roc_auc']:.3f} (target ≥ 0.75)",
            ]
        )
        st.markdown(report)
        st.download_button(
            "Download report (.md)", report, f"flowcast_report_{span[0]}_{span[1]}.md"
        )
