from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def ensure_package(import_name: str, pip_name: str | None = None) -> None:
    try:
        __import__(import_name)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name or import_name])


ensure_package("nibabel")
ensure_package("pandas")

import torch


print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("hip:", getattr(torch.version, "hip", None))
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))


DATA_ROOT = Path(os.environ.get("MRI_DATA_ROOT", "/kaggle/input/competitions/ehl-paris-medical-image-retrieval"))
OUT = Path(os.environ.get("MRI_SUBMISSION_OUT", "/tmp/iou_submission.csv"))

print("DATA_ROOT:", DATA_ROOT)
print("exists:", DATA_ROOT.exists())
if not DATA_ROOT.exists():
    print("Dataset root is missing on this server. Mount/copy the competition dataset, then rerun.")
    raise SystemExit(0)


try:
    from src.iou_submission import MRIIoUSubmission
except Exception:
    MRIIoUSubmission = globals()["MRIIoUSubmission"]


query_csvs = [
    DATA_ROOT / "dataset1" / "test_queries.csv",
    DATA_ROOT / "dataset2" / "test_queries.csv",
    DATA_ROOT / "dataset3" / "test_queries.csv",
]
gallery_csvs = [
    DATA_ROOT / "dataset1" / "test_gallery.csv",
    DATA_ROOT / "dataset2" / "test_gallery.csv",
    DATA_ROOT / "dataset3" / "test_gallery.csv",
]

missing = [path for path in [*query_csvs, *gallery_csvs] if not path.exists()]
if missing:
    print("Missing CSVs:")
    for path in missing:
        print(" -", path)
    raise SystemExit(1)

runner = MRIIoUSubmission(
    data_root=DATA_ROOT,
    output_csv=OUT,
    projection_size=int(os.environ.get("MRI_IOU_PROJECTION_SIZE", "128")),
    threshold_quantile=float(os.environ.get("MRI_IOU_THRESHOLD_QUANTILE", "0.60")),
    batch_size=int(os.environ.get("MRI_IOU_BATCH_SIZE", "64")),
)
submission = runner.build_submission(query_csvs=query_csvs, gallery_csvs=gallery_csvs)
print("submission:", OUT)
print("rows:", len(submission))
print(submission.head())
