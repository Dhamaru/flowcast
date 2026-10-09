"""Shared helpers: config, paths, logging, the time split and lineage."""

import hashlib
import logging
import pathlib
import random
import sys

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent  # the flowcast/ folder

with open(ROOT / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)


def path(name, filename=None):
    """Absolute path to a configured folder, so scripts work from any cwd."""
    folder = ROOT / CONFIG["paths"][name]
    folder.mkdir(parents=True, exist_ok=True)
    return folder / filename if filename else folder


def get_logger(stage):
    """Logger that writes to the console and to reports/pipeline.log (PRD 9: every stage logs)."""
    logger = logging.getLogger(stage)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s  %(name)-8s  %(message)s", "%H:%M:%S")
        handlers = [logging.StreamHandler()]
        if "pytest" not in sys.modules:  # tests must never write into the real pipeline log
            handlers.append(logging.FileHandler(path("reports", "pipeline.log"), encoding="utf-8"))
        for handler in handlers:
            handler.setFormatter(fmt)
            logger.addHandler(handler)
    return logger


def set_seeds():
    """Seed every random number generator the pipeline touches (PRD NFR: deterministic)."""
    seed = CONFIG["project"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
    return seed


def split_cutoffs(timestamps):
    """
    Train / validation / test boundaries by time.

    Returned as two timestamps, so every model (classical and LSTM) uses the
    exact same windows - PRD 12.4 wants the head-to-head on the same test window.
    """
    unique = np.sort(timestamps.unique())
    n = len(unique)
    train_end = unique[int(n * CONFIG["split"]["train"])]
    val_end = unique[int(n * (CONFIG["split"]["train"] + CONFIG["split"]["validation"]))]
    return train_end, val_end


def split_masks(df):
    """Boolean masks (train, validation, test) for a frame with a timestamp column."""
    train_end, val_end = split_cutoffs(df["timestamp"])
    train = df["timestamp"] < train_end
    test = df["timestamp"] >= val_end
    return train, ~train & ~test, test


def file_hash(file_path):
    """Short SHA-256 of a file, recorded in model cards so a model can be traced to its data."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]
