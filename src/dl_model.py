"""
dl_model.py - M6: a neural network (LSTM) that reads the recent past as a sequence.

THE IDEA
    The classical models get the past handed to them as ready-made columns (lag1, lag48,
    rolling averages). An LSTM instead reads the raw readings of the last N windows one
    by one, in order, and learns for itself which parts of that history matter.
    "LSTM" = Long Short-Term Memory: a network with a memory it updates at each step.

WHAT IT LOOKS LIKE
    input      the last 16 windows (8 hours) of one road, 11 readings per window
               (volume, speed, occupancy, rain, hour of day, ...), all scaled to similar sizes
    LSTM       2 layers that read the 16 steps in order and summarise them
    two heads  one predicts next window's volume (regression),
               one predicts next window's congestion level (classification)

HOW TRAINING WORKS
    The data is fed in small batches of 256. After each pass through all the training data
    (one "epoch"), the score on the validation data is checked. If it hasn't improved for
    5 epochs, training stops and the best version is kept ("early stopping"). That stops
    the network memorising the training data instead of learning general patterns.

THE RESULT
    It is tested on exactly the same rows as XGBoost. It came in about 3.7% worse, so the
    simpler XGBoost model is the one used. "The fancier model didn't help" is a valid result.
    A "BiLSTM" (reads the window forwards and backwards) is also tried; that's fine here
    because the whole window is already in the past.
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
BOUNDED = ["occupancy", "visibility", "hour_sin", "hour_cos", "is_peak_period", "is_rain",
           "is_weekend"]
UNBOUNDED = ["traffic_volume", "avg_speed", "temperature", "rainfall"]
STEP_COLS = UNBOUNDED + BOUNDED


class SequenceModel(nn.Module):
    """LSTM (or BiLSTM) with a volume head and a congestion head."""

    def __init__(self, n_features, hidden, layers, dropout, bidirectional):
        """Recurrent core plus two output heads."""
        super().__init__()
        self.rnn = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout, bidirectional=bidirectional)
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
    # Example with n = 4 and target row 10: rows[:, None] + [-4, -3, -2, -1] -> rows 6, 7, 8, 9.
    # Doing this for every target at once gives a 3-D block: (targets, 4 steps, features).
    return values[rows[:, None] + np.arange(-n, 0)[None, :]]


def prepare(df, idx, n):
    """Scaled sequences and targets for train / val / test, using M5's exact row split."""
    positions = {k: df.index.get_indexer(v) for k, v in idx.items()}
    roads = df["road_id"].to_numpy()
    for pos in positions.values():
        # the oldest step of every window must be on the same road as its target
        assert (roads[pos - n] == roads[pos]).all(), "window crosses segments"
    train_rows = df.iloc[positions["train"]]
    # Neural networks learn badly when one input is in thousands and another is 0 or 1, so
    # every input is rescaled. The scalers learn their numbers from TRAINING rows only:
    # using test rows here would quietly leak information about the test window.
    zscore = StandardScaler().fit(train_rows[UNBOUNDED])  # -> average 0, spread 1
    minmax = MinMaxScaler().fit(train_rows[BOUNDED])  # -> squeezed into 0..1
    values = np.hstack([zscore.transform(df[UNBOUNDED]),
                        minmax.transform(df[BOUNDED])]).astype(np.float32)
    y_scaler = StandardScaler().fit(train_rows[["traffic_volume"]])
    data = {}
    for k, pos in positions.items():
        part = df.iloc[pos]
        y = y_scaler.transform(part[["traffic_volume"]]).ravel()
        data[k] = (torch.tensor(build_sequences(values, pos, n)),  # inputs
                   torch.tensor(y, dtype=torch.float32),  # scaled volume target
                   torch.tensor(part["congestion_code"].to_numpy(), dtype=torch.long))
    return data, {"zscore": zscore, "minmax": minmax, "y": y_scaler}


def train(data, hidden, bidirectional, epochs):
    """Mini-batch training with early stopping on validation loss; returns the best weights."""
    set_seeds()
    model = SequenceModel(len(STEP_COLS), hidden, DL["layers"], DL["dropout"], bidirectional)
    opt = torch.optim.Adam(model.parameters(), lr=DL["learning_rate"],
                           weight_decay=DL["weight_decay"])
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    mse, ce, w = nn.MSELoss(), nn.CrossEntropyLoss(), DL["congestion_loss_weight"]
    X, y, c = data["train"]
    Xv, yv, cv = data["val"]
    best, best_state, wait, history = np.inf, None, 0, {"train": [], "val": []}
    for epoch in range(epochs):
        model.train()  # training mode: dropout is switched on
        order, total = torch.randperm(len(X)), 0.0
        for start in range(0, len(X), DL["batch_size"]):
            b = order[start : start + DL["batch_size"]]  # the next 256 sequences
            vol, logits = model(X[b])  # 1. predict
            # 2. measure how wrong: squared error for volume + cross-entropy for congestion
            loss = mse(vol, y[b]) + w * ce(logits, c[b])
            opt.zero_grad()  # 3. clear the previous step's gradients
            loss.backward()  # 4. work out how each weight affected the loss
            opt.step()  # 5. nudge every weight to lower the loss
            total += loss.item() * len(b)
        # After each epoch, score the validation data (never trained on) to decide when to stop.
        model.eval()
        with torch.no_grad():
            vol, logits = model(Xv)
            val_loss = (mse(vol, yv) + w * ce(logits, cv)).item()
        plateau.step(val_loss)
        history["train"].append(total / len(X))
        history["val"].append(val_loss)
        if val_loss < best - 1e-5:  # improved: remember these weights, reset the patience count
            best, wait = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
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
    congestion = ev.congestion_metrics(c.numpy(), logits.argmax(1).numpy(), LEVELS)
    metrics["congestion_macro_f1"] = congestion["macro_f1"]
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
    classical_rmse = next(c for c in classical["candidates"]
                          if c["model"] == classical["selected"])["test"]["rmse"]
    result = {"tuning": tuning, "selected": best,
              "classical": {"model": classical["selected"], "rmse": classical_rmse},
              "variants": {}}
    cards = json.loads(path("models", "model_cards.json").read_text())
    cards = [c for c in cards if not c["file"].endswith(".pt")]
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, bidirectional in [("LSTM", False), ("BiLSTM", True)]:
        model, history, _ = train(data, best["hidden_size"], bidirectional, DL["epochs"])
        metrics, pred = evaluate_model(model, data, scalers, df, idx)
        metrics["epochs_run"] = len(history["val"])
        comparison = ev.head_to_head(classical["selected"], classical_rmse, metrics)
        metrics["beats_classical"] = comparison["beats_classical"]
        result["variants"][label] = {"test": metrics, "history": history,
                                     "vs_classical": comparison}
        log.info("%s: RMSE %.2f MAPE %.2f%% congestion macro-F1 %.3f (%d epochs) vs %s RMSE %.2f",
                 label, metrics["rmse"], metrics["mape"], metrics["congestion_macro_f1"],
                 metrics["epochs_run"], classical["selected"], classical_rmse)
        file = f"{label.lower()}_volume.pt"
        torch.save({"state_dict": model.state_dict(), "hidden": best["hidden_size"],
                    "seq_length": best["seq_length"], "bidirectional": bidirectional,
                    "step_cols": STEP_COLS, "version": version}, path("models", file))
        with open(path("models", f"{label.lower()}_scalers.pkl"), "wb") as f:
            pickle.dump(scalers, f)
        ax.plot(history["train"], label=f"{label} train")
        ax.plot(history["val"], "--", label=f"{label} validation")
        cards.append({
            "name": f"{label}, traffic_volume + congestion, next window",
            "file": file,
            "algorithm": f"{label} ({DL['layers']} layers, hidden {best['hidden_size']}, "
                         f"window {best['seq_length']}), multi-task",
            "target": "traffic_volume (+ congestion_level head)",
            "horizon": 1,
            "version": version,
            "train_window": "same as classical models",
            "train_rows": len(idx["train"]),
            "params": {**best, "dropout": DL["dropout"], "lr": DL["learning_rate"],
                       "weight_decay": DL["weight_decay"]},
            "selected_by": "validation loss over the seq_length x hidden_size grid",
            "features": STEP_COLS,
            "data_sha": data_sha,
            "test_metrics": {k: v for k, v in metrics.items() if not isinstance(v, bool)},
        })
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (MSE + weighted cross-entropy)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path("figures", "lstm_training_curve.png"), dpi=110)
    plt.close(fig)
    ev.write_model_cards(cards)
    path("reports", "dl_result.json").write_text(json.dumps(result, indent=2, default=str))
    return result
