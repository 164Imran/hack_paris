from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REMOTE_ROOT = Path(os.environ.get("REMOTE_HACK_PARIS_ROOT", "/tmp/hack_paris"))
SCRIPT_PATH = REMOTE_ROOT / "src" / "volumetric_jepa_vae_submission.py"
LOG_PATH = Path(os.environ.get("MRI_REMOTE_LOG", str(REMOTE_ROOT / "vae_jepa_full.log")))
OUT_PATH = Path(os.environ.get("MRI_SUBMISSION_OUT", str(REMOTE_ROOT / "vae_jepa_submission.csv")))
SAVE_DIR = Path(os.environ.get("MRI_SAVE_DIR", str(REMOTE_ROOT / "artifacts" / "vae_jepa")))


def ensure_package(import_name: str, pip_name: str | None = None) -> None:
    try:
        __import__(import_name)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name or import_name])


def candidate_data_roots() -> list[Path]:
    values = [
        os.environ.get("MRI_DATA_ROOT"),
        "/kaggle/input/competitions/ehl-paris-medical-image-retrieval",
        "/kaggle/input/ehl-paris-medical-image-retrieval",
        "/kaggle/input/ehl-paris-2026-medical-retrieval/ehl-paris-medical-image-retrieval",
        "/workspace/data/ehl-paris-medical-image-retrieval",
        "/root/data/ehl-paris-medical-image-retrieval",
        str(REMOTE_ROOT / "data" / "ehl-paris-medical-image-retrieval"),
    ]
    roots = [Path(value) for value in values if value]
    for base in [Path("/kaggle/input"), Path("/workspace"), Path("/root"), Path("/tmp")]:
        if base.exists():
            roots.extend(base.glob("**/dataset1/train_pairs.csv"))
    normalized = []
    for path in roots:
        root = path.parent if path.name == "train_pairs.csv" else path
        if root.name == "dataset1":
            root = root.parent
        if root not in normalized:
            normalized.append(root)
    return normalized


def find_data_root() -> Path | None:
    for root in candidate_data_roots():
        if (root / "dataset1" / "train_pairs.csv").exists():
            return root
    return None


def existing_sample_submission(data_root: Path) -> Path | None:
    candidates = [
        data_root / "sample_submission.csv",
        data_root.parent / "sample_submission.csv",
        Path("/kaggle/input/competitions/ehl-paris-medical-image-retrieval/sample_submission.csv"),
        Path("/kaggle/input/ehl-paris-medical-image-retrieval/sample_submission.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


ensure_package("nibabel")
ensure_package("pandas")
ensure_package("numpy")

import torch


print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("hip:", getattr(torch.version, "hip", None))
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))

print("remote root:", REMOTE_ROOT)
print("script path:", SCRIPT_PATH, "exists:", SCRIPT_PATH.exists())
if not SCRIPT_PATH.exists():
    raise SystemExit(f"Missing uploaded script: {SCRIPT_PATH}")

data_root = find_data_root()
print("data root:", data_root)
if data_root is None:
    print("Dataset root not found. Checked candidates:")
    for root in candidate_data_roots()[:50]:
        print(" -", root)
    raise SystemExit(2)

query_csvs = [
    data_root / "dataset1" / "val_queries.csv",
    data_root / "dataset1" / "test_queries.csv",
    data_root / "dataset2" / "val_queries.csv",
    data_root / "dataset2" / "test_queries.csv",
    data_root / "dataset3" / "val_queries.csv",
    data_root / "dataset3" / "test_queries.csv",
]
gallery_csvs = [
    data_root / "dataset1" / "val_gallery.csv",
    data_root / "dataset1" / "test_gallery.csv",
    data_root / "dataset2" / "val_gallery.csv",
    data_root / "dataset2" / "test_gallery.csv",
    data_root / "dataset3" / "val_gallery.csv",
    data_root / "dataset3" / "test_gallery.csv",
]
missing = [path for path in [*query_csvs, *gallery_csvs, data_root / "dataset1" / "train_pairs.csv"] if not path.exists()]
if missing:
    print("Missing required files:")
    for path in missing:
        print(" -", path)
    raise SystemExit(3)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
SAVE_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable,
    str(SCRIPT_PATH),
    "--data-root",
    str(data_root),
    "--train-pairs-csv",
    str(data_root / "dataset1" / "train_pairs.csv"),
    "--out",
    str(OUT_PATH),
    "--include-dataset3-test-for-vae",
    "--image-size",
    os.environ.get("MRI_IMAGE_SIZE", "96"),
    "--slice-step",
    os.environ.get("MRI_SLICE_STEP", "5"),
    "--max-slices",
    os.environ.get("MRI_MAX_SLICES", "32"),
    "--patch-size",
    os.environ.get("MRI_PATCH_SIZE", "16"),
    "--token-dim",
    os.environ.get("MRI_TOKEN_DIM", "128"),
    "--vae-latent-dim",
    os.environ.get("MRI_VAE_LATENT_DIM", "128"),
    "--vit-depth",
    os.environ.get("MRI_VIT_DEPTH", "2"),
    "--axial-depth",
    os.environ.get("MRI_AXIAL_DEPTH", "2"),
    "--heads",
    os.environ.get("MRI_HEADS", "4"),
    "--vae-epochs",
    os.environ.get("MRI_VAE_EPOCHS", "5"),
    "--jepa-epochs",
    os.environ.get("MRI_JEPA_EPOCHS", "15"),
    "--batch-size",
    os.environ.get("MRI_BATCH_SIZE", "4"),
    "--lr",
    os.environ.get("MRI_LR", "0.001"),
    "--score-window",
    os.environ.get("MRI_SCORE_WINDOW", "1"),
    "--trim-fraction",
    os.environ.get("MRI_TRIM_FRACTION", "0.85"),
    "--save-dir",
    str(SAVE_DIR),
]
sample = existing_sample_submission(data_root)
if sample is not None:
    cmd.extend(["--sample-submission", str(sample)])
for query_csv, gallery_csv in zip(query_csvs, gallery_csvs):
    cmd.extend(["--query-csv", str(query_csv), "--gallery-csv", str(gallery_csv)])
if torch.cuda.is_available():
    cmd.extend(["--device", "cuda"])

print("output:", OUT_PATH)
print("log:", LOG_PATH)
print("save dir:", SAVE_DIR)
print("command:", " ".join(cmd))

if os.environ.get("MRI_FOREGROUND") == "1":
    with LOG_PATH.open("w") as log:
        completed = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    print("return code:", completed.returncode)
    raise SystemExit(completed.returncode)

with LOG_PATH.open("w") as log:
    process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True)

(REMOTE_ROOT / "vae_jepa_full.pid").write_text(str(process.pid))
print("started pid:", process.pid)
