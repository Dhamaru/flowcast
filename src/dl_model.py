"""
M6 - Deep-Learning Engine (PRD FR-09, section 12).

A recurrent sequence model trained from scratch (random init, no pretrained
weights), benchmarked head-to-head against the classical winner from M5.

  input   the last N windows of one segment, each a vector of scaled readings
  core    2 LSTM layers, dropout between them (PRD 12.1)
  heads   next-window volume (regression) + congestion class (multi-task, PRD 12.1)
  loss    MSE(volume) + weight x cross-entropy(congestion)       (PRD 12.2)
  control dropout, early stopping, LR reduction on plateau, weight decay (PRD 12.3)

A bidirectional variant is trained for comparison. Reading the input window in
both directions is safe here: the whole window is in the past.
"""

import itertools
import json
import pickle

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import evaluate as ev
from ml_models import frames
from utils import CONFIG, get_logger, path, set_seeds

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = get_logger("M6")
DL = CONFIG["lstm"]
LEVELS = CONFIG["cleaning"]["congestion_levels"]
# PRD 10.1: Min-Max for bounded features, z-score for unbounded ones
BOUNDED = [
    "occupancy",
    "visibility",
    "hour_sin",
    "hour_cos",
    "is_peak_period",
    "is_rain",
    "is_weekend",
]
UNBOUNDED = ["traffic_volume", "avg_speed", "temperature", "rainfall"]
STEP_COLS = UNBOUNDED + BOUNDED


class SequenceModel(nn.Module):
    """LSTM (or BiLSTM) with a volume head and a congestion head."""

    def __init__(self, n_features, hidden, layers, dropout, bidirectional):
        """Recurrent core plus two output heads."""
        super().__init__()
        self.rnn = nn.LSTM(
            n_features,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        width = hidden * (2 if bidirectional else 1)
        self.drop = nn.Dropout(dropout)  # dropout before the dense heads (PRD 12.3)
        self.volume_head = nn.Linear(width, 1)
        self.congestion_head = nn.Linear(width, len(LEVELS))

    def forward(self, x):
        """Return (scaled volume, congestion logits) for a batch of windows."""
        out, _ = self.rnn(x)
        last = self.drop(out[:, -1, :])  # the network's summary of the whole window
        return self.volume_head(last).squeeze(-1), self.congestion_head(last)


def build_sequences(values, rows, n):
    """
    Window of the n rows before each target row: shape (targets, n, features).

    The feature table is sorted by segment then time on a complete grid, so
    "the n rows before" are exactly the n windows before, within the same segment.
    """
    return values[rows[:, None] + np.arange(-n, 0)[None, :]]


def prepare(df, idx, n):
    """Scaled sequences and targets for train / val / test, using M5's exact row split."""
    positions = {k: df.index.get_indexer(v) for k, v in idx.items()}
    for k, pos in positions.items():
        assert (df["road_id"].to_numpy()[pos - n] == df["road_id"].to_numpy()[pos]).all(), (
            "window crosses segments"
        )
    train_rows = df.iloc[positions["train"]]
    zscore = StandardScaler().fit(
        train_rows[UNBOUNDED]
    )  # fit on training rows only: no test statistics leak in
    minmax = MinMaxScaler().fit(train_rows[BOUNDED])
    values = np.hstack([zscore.transform(df[UNBOUNDED]), minmax.transform(df[BOUNDED])]).astype(
        np.float32
    )
    y_scaler = StandardScaler().fit(train_rows[["traffic_volume"]])
    data = {}
    for k, pos in positions.items():
        part = df.iloc[pos]
        data[k] = (
            torch.tensor(build_sequences(values, pos, n)),
            torch.tensor(
                y_scaler.transform(part[["traffic_volume"]]).ravel(), dtype=torch.float32
            ),
            torch.tensor(part["congestion_code"].to_numpy(), dtype=torch.long),
        )
    return data, {"zscore": zscore, "minmax": minmax, "y": y_scaler}


def train(data, hidden, bidirectional, epochs):
    """Mini-batch training with early stopping on validation loss; returns the best weights."""
    set_seeds()
    model = SequenceModel(len(STEP_COLS), hidden, DL["layers"], DL["dropout"], bidirectional)
    opt = torch.optim.Adam(
        model.parameters(), lr=DL["learning_rate"], weight_decay=DL["weight_decay"]
    )
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    mse, ce, w = nn.MSELoss(), nn.CrossEntropyLoss(), DL["congestion_loss_weight"]
    X, y, c = data["train"]
    Xv, yv, cv = data["val"]
    best, best_state, wait, history = np.inf, None, 0, {"train": [], "val": []}
    for epoch in range(epochs):
        model.train()
        order, total = torch.randperm(len(X)), 0.0
        for start in range(0, len(X), DL["batch_size"]):
            b = order[start : start + DL["batch_size"]]
            vol, logits = model(X[b])
            loss = mse(vol, y[b]) + w * ce(logits, c[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            vol, logits = model(Xv)
            val_loss = (mse(vol, yv) + w * ce(logits, cv)).item()
        plateau.step(val_loss)
        history["train"].append(total / len(X))
        history["val"].append(val_loss)
        if val_loss < best - 1e-5:
            best, best_state, wait = (
                val_loss,
                {k: v.clone() for k, v in model.state_dict().items()},
                0,
            )
        else:
            wait += 1
            if wait >= DL["patience"]:
                break
    model.load_state_dict(best_state)
    return model, history, best


def evaluate_model(model, data, scalers, df, idx):
    """Test-window volume metrics and congestion macro-F1 for a trained model."""
    model.eval()
    X, _, c = data["test"]
    with torch.no_grad():
        vol, logits = model(X)
    pred = scalers["y"].inverse_transform(vol.numpy().reshape(-1, 1)).ravel()
    metrics = ev.regression_metrics(df.loc[idx["test"], "traffic_volume"], pred)
    metrics["congestion_macro_f1"] = ev.congestion_metrics(
        c.numpy(), logits.argmax(1).numpy(), LEVELS
    )["macro_f1"]
    return metrics, pred


def run(df, board, version, data_sha):
    """Tune the LSTM, train LSTM + BiLSTM, benchmark against the classical winner."""
    _, idx = frames(df, "traffic_volume", 1)  # identical rows to the classical models (PRD 12.4)
    tuning = []
    for n, hidden in itertools.product(DL["grid"]["seq_length"], DL["grid"]["hidden_size"]):
        data, _ = prepare(df, idx, n)
        _, _, val_loss = train(data, hidden, False, DL["tune_epochs"])
        tuning.append({"seq_length": n, "hidden_size": hidden, "val_loss": val_loss})
        log.info("tuning: seq_length %d hidden %d -> val loss %.4f", n, hidden, val_loss)
    best = min(tuning, key=lambda t: t["val_loss"])
    log.info("selected: seq_length %d hidden %d", best["seq_length"], best["hidden_size"])

    data, scalers = prepare(df, idx, best["seq_length"])
    classical = board["volume"]
    classical_rmse = next(
        c for c in classical["candidates"] if c["model"] == classical["selected"]
    )["test"]["rmse"]
    result = {
        "tuning": tuning,
        "selected": best,
        "classical": {"model": classical["selected"], "rmse": classical_rmse},
        "variants": {},
    }
    cards = json.loads(path("models", "model_cards.json").read_text())
    cards = [c for c in cards if not c["file"].endswith(".pt")]
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, bidirectional in [("LSTM", False), ("BiLSTM", True)]:
        model, history, _ = train(data, best["hidden_size"], bidirectional, DL["epochs"])
        metrics, pred = evaluate_model(model, data, scalers, df, idx)
        metrics["epochs_run"] = len(history["val"])
        metrics["beats_classical"] = metrics["rmse"] < classical_rmse
        result["variants"][label] = {"test": metrics, "history": history}
        log.info(
            "%s: RMSE %.2f MAPE %.2f%% congestion macro-F1 %.3f (%d epochs) vs %s RMSE %.2f",
            label,
            metrics["rmse"],
            metrics["mape"],
            metrics["congestion_macro_f1"],
            metrics["epochs_run"],
            classical["selected"],
            classical_rmse,
        )
        file = f"{label.lower()}_volume.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "hidden": best["hidden_size"],
                "seq_length": best["seq_length"],
                "bidirectional": bidirectional,
                "step_cols": STEP_COLS,
                "version": version,
            },
            path("models", file),
        )
        with open(path("models", f"{label.lower()}_scalers.pkl"), "wb") as f:
            pickle.dump(scalers, f)
        ax.plot(history["train"], label=f"{label} train")
        ax.plot(history["val"], "--", label=f"{label} validation")
        cards.append(
            {
                "name": f"{label}, traffic_volume + congestion, next window",
                "file": file,
                "algorithm": f"{label} ({DL['layers']} layers, hidden {best['hidden_size']}, "
                f"window {best['seq_length']}), multi-task",
                "target": "traffic_volume (+ congestion_level head)",
                "horizon": 1,
                "version": version,
                "train_window": "same as classical models",
                "train_rows": len(idx["train"]),
                "params": {
                    **best,
                    "dropout": DL["dropout"],
                    "lr": DL["learning_rate"],
                    "weight_decay": DL["weight_decay"],
                },
                "selected_by": "validation loss over the seq_length x hidden_size grid",
                "features": STEP_COLS,
                "data_sha": data_sha,
                "test_metrics": {k: v for k, v in metrics.items() if not isinstance(v, bool)},
            }
        )
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (MSE + weighted cross-entropy)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path("figures", "lstm_training_curve.png"), dpi=110)
    plt.close(fig)
    ev.write_model_cards(cards)
    path("reports", "dl_result.json").write_text(json.dumps(result, indent=2, default=str))
    return result
