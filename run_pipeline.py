"""
FlowCast pipeline, raw CSVs to trained models, in PRD order:
    M1 ingest -> M2 clean -> M3 features -> M5 classical ML -> M6 deep learning
(M4, EDA, lives in notebooks/01_eda.ipynb and reads the M2/M3 outputs.)

    python flowcast/run_pipeline.py              everything
    python flowcast/run_pipeline.py --skip-dl    stop after M5 (M6 is the slowest stage)
    python flowcast/run_pipeline.py --from ml    retrain only, reusing the saved feature table
"""

import argparse
import json
import pathlib
import sys
import time
import warnings

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))
warnings.filterwarnings("ignore", category=FutureWarning)

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
    features.run(clean.run(*ingest.run()))
df = pd.read_csv(
    features_file, parse_dates=["timestamp"]
)  # read back: the exact file the hash describes
data_sha = file_hash(features_file)
version = f"{CONFIG['project']['version']}-{data_sha[:8]}"
log.info("model version %s (features sha256 %s)", version, data_sha)

board = ml_models.run(df, version, data_sha)
path("reports", "scoreboard.json").write_text(json.dumps(board, indent=2, default=str))
if not args.skip_dl:
    import dl_model  # imported only here, so PyTorch isn't needed for --skip-dl runs

    dl_model.run(df, board, version, data_sha)

log.info("pipeline finished in %.1f min", (time.perf_counter() - began) / 60)
