"""
run_pipeline.py - START HERE. Runs the whole project, from raw CSVs to trained models.

    python flowcast/run_pipeline.py              everything (about 15-20 minutes)
    python flowcast/run_pipeline.py --skip-dl    skip the neural network (the slowest part)
    python flowcast/run_pipeline.py --from ml    only retrain models, reuse the saved features

THE STEPS, AND THE FILE THAT DOES EACH ONE
    M1  src/ingest.py     load the 3 raw CSVs and check every row
    M2  src/clean.py      fix bad values, fill gaps, join the tables     -> data/processed/
    M3  src/features.py   add the 41 columns the models learn from
    M5  src/ml_models.py  train and compare the classical models         -> models/
    M6  src/dl_model.py   train the LSTM and compare it with the best classical model
    (M4, exploring the data, is in notebooks/01_eda.ipynb.)

Every step prints what it did and saves it to reports/pipeline.log.
Run it twice and you get the same numbers: every random choice is seeded.
"""

import argparse
import json
import pathlib
import sys
import time
import warnings

# Let Python find the files in src/ (ingest.py, clean.py, ...) so they can be imported below.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
warnings.filterwarnings("ignore", category=FutureWarning)  # hide "will change in future" notices

import pandas as pd  # noqa: E402

import clean  # noqa: E402
import features  # noqa: E402
import ingest  # noqa: E402
import ml_models  # noqa: E402
from utils import CONFIG, file_hash, get_logger, path, set_seeds  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--skip-dl", action="store_true")
parser.add_argument("--from", dest="start", choices=["raw", "ml"], default="raw")
args = parser.parse_args()

log = get_logger("pipeline")
set_seeds()
began = time.perf_counter()
features_file = path("processed", f"flowcast_features_v{CONFIG['project']['version']}.csv")

if args.start == "raw":
    # M1 -> M2 -> M3: each step's output is the next step's input.
    features.run(clean.run(*ingest.run()))

# Read the feature table back from disk, so the models train on exactly the file we fingerprint.
df = pd.read_csv(features_file, parse_dates=["timestamp"])
data_sha = file_hash(features_file)
# The model version includes the data's fingerprint, e.g. "1.0-5bb519c5". New data -> new version,
# so every prediction can be traced to the exact data its model was trained on.
version = f"{CONFIG['project']['version']}-{data_sha[:8]}"
log.info("model version %s (features sha256 %s)", version, data_sha)

board = ml_models.run(df, version, data_sha)
path("reports", "scoreboard.json").write_text(json.dumps(board, indent=2, default=str))
if not args.skip_dl:
    import dl_model  # imported only here, so PyTorch isn't needed for --skip-dl runs

    dl_model.run(df, board, version, data_sha)

log.info("pipeline finished in %.1f min", (time.perf_counter() - began) / 60)
