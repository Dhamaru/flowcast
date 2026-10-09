"""
Metrics, comparison, error analysis and model cards (PRD sections 13.2-13.4).
Shared by the classical models (M5) and the LSTM (M6) so both are scored the same way.
"""

import json

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from utils import CONFIG, path

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def regression_metrics(y_true, y_pred):
    """RMSE is the headline; MAE, MAPE and R2 support it (PRD 13.2)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    nonzero = y_true != 0
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(
            np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
        ),
        "r2": float(r2_score(y_true, y_pred)),
    }


def congestion_metrics(y_true, y_pred, labels):
    """Macro-F1 headline, plus per-class precision/recall, accuracy and confusion matrix."""
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": dict(
            zip(
                labels,
                precision_score(
                    y_true, y_pred, average=None, labels=range(len(labels)), zero_division=0
                )
                .round(3)
                .tolist(),
            )
        ),
        "recall": dict(
            zip(
                labels,
                recall_score(
                    y_true, y_pred, average=None, labels=range(len(labels)), zero_division=0
                )
                .round(3)
                .tolist(),
            )
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(labels))).tolist(),
    }


def best_f1_threshold(y_true, scores):
    """Alert threshold that maximises F1 - chosen on the validation slice, never on test."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    return float(thresholds[np.argmax(f1[:-1])])


def accident_metrics(y_true, scores, threshold):
    """ROC-AUC is the headline; precision/recall/F1 at the operating threshold support it."""
    alert = scores >= threshold
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, alert, zero_division=0)),
        "recall": float(recall_score(y_true, alert, zero_division=0)),
        "f1": float(f1_score(y_true, alert, zero_division=0)),
        "alerts_per_1000_windows": float(alert.mean() * 1000),
    }


def interval_by_hour(residuals, hours):
    """Half-width of the confidence range per hour of day, from validation errors."""
    q = CONFIG["confidence"]
    return pd.Series(np.abs(residuals)).groupby(np.asarray(hours)).quantile(q).to_dict()


def coverage(actual, lower, upper):
    """Share of actual values that landed inside the stated range."""
    actual = np.asarray(actual)
    return float(((actual >= lower) & (actual <= upper)).mean())


def error_breakdown(frame, actual_col, pred_col):
    """RMSE by hour, day of week and weather (PRD 13.3): where does the model struggle?"""
    err = (frame[actual_col] - frame[pred_col]) ** 2
    out = {}
    for name, key in [
        ("hour", frame["timestamp"].dt.hour),
        ("day_of_week", frame["timestamp"].dt.strftime("%a")),
        ("weather", frame["weather_condition"]),
    ]:
        out[name] = np.sqrt(err.groupby(key).mean()).round(2).to_dict()
    return out


def plot_error_analysis(frame, actual_col, pred_col, breakdown, title):
    """Residual histogram, predicted-vs-actual for two segments, RMSE by hour."""
    resid = frame[actual_col] - frame[pred_col]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].hist(resid, bins=80, color="#5aa7e0")
    axes[0].set_title(f"{title}: residuals (actual - predicted)")
    for road in frame["road_id"].drop_duplicates().iloc[:2]:
        seg = frame[frame["road_id"] == road].sort_values("timestamp").iloc[: 48 * 3]
        axes[1].plot(seg["timestamp"], seg[actual_col], label=f"{road} actual")
        axes[1].plot(seg["timestamp"], seg[pred_col], "--", label=f"{road} predicted")
    axes[1].set_title("Predicted vs actual, first 3 test days")
    axes[1].legend(fontsize=7)
    axes[1].tick_params(axis="x", rotation=30)
    hours = pd.Series(breakdown["hour"])
    axes[2].bar(hours.index, hours.values, color="#e8a93a")
    axes[2].set_title("RMSE by hour of day")
    fig.tight_layout()
    fig.savefig(path("figures", f"error_analysis_{title.lower().replace(' ', '_')}.png"), dpi=110)
    plt.close(fig)


def plot_confusion(matrix, labels):
    """Save the congestion confusion matrix as a figure."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap="Blues")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, matrix[i][j], ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(labels)), labels, rotation=30)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("Congestion confusion matrix (test)")
    fig.tight_layout()
    fig.savefig(path("figures", "congestion_confusion.png"), dpi=110)
    plt.close(fig)


def write_model_cards(cards):
    """models/model_cards.json + models/model_cards.md - one card per saved model (PRD 13.4)."""
    path("models", "model_cards.json").write_text(json.dumps(cards, indent=2, default=str))
    lines = [
        "# FlowCast - Model Cards",
        "",
        "Generated by the pipeline. One card per saved model: what it was trained on, "
        "with which settings, and how it scored on the untouched test window.",
        "",
    ]
    for c in cards:
        lines += [
            f"## {c['name']}",
            "",
            f"- **File:** `models/{c['file']}`  ·  **Model version:** `{c['version']}`",
            f"- **Algorithm:** {c['algorithm']}  ·  **Target:** `{c['target']}`  ·  "
            f"**Horizon:** {c['horizon']} window(s) ahead",
            f"- **Training window:** {c['train_window']}  ·  **Rows:** {c['train_rows']:,}",
            f"- **Selected by:** {c['selected_by']}",
            f"- **Hyperparameters:** `{json.dumps(c['params'])}`",
            f"- **Features ({len(c['features'])}):** {', '.join(c['features'])}",
            f"- **Data:** feature table sha256 `{c['data_sha']}`",
            "- **Test metrics:**",
            "",
        ]
        lines += [
            f"  - {k}: {v:.4f}" if isinstance(v, float) else f"  - {k}: {v}"
            for k, v in c["test_metrics"].items()
            if not isinstance(v, (dict, list))
        ]
        lines.append("")
    path("models", "model_cards.md").write_text("\n".join(lines), encoding="utf-8")
