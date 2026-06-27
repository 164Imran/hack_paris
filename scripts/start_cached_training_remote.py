from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import torch


ROOT = Path("/tmp/hack_paris")
CACHE_PT = ROOT / "data" / "mri_volumetric_cache_step5_96_uint8.pt"
SCRIPT = ROOT / "src" / "volumetric_jepa_vae_submission.py"
OUT = ROOT / "data" / "vae_jepa_submission.csv"
SAVE_DIR = ROOT / "artifacts" / "vae_jepa"
LOG_DIR = ROOT / "logs"
LOG = LOG_DIR / "vae_jepa_train.log"
PID = LOG_DIR / "vae_jepa_train.pid"


print("python:", sys.executable)
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("hip:", getattr(torch.version, "hip", None))
print("cache:", CACHE_PT, CACHE_PT.exists(), CACHE_PT.stat().st_size if CACHE_PT.exists() else None)
print("script:", SCRIPT, SCRIPT.exists())

if not CACHE_PT.exists():
    raise FileNotFoundError(CACHE_PT)
if not SCRIPT.exists():
    raise FileNotFoundError(SCRIPT)

LOG_DIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

cmd = [
    sys.executable,
    str(SCRIPT),
    "--precomputed-pt",
    str(CACHE_PT),
    "--out",
    str(OUT),
    "--device",
    "cuda",
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

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
with LOG.open("w") as log:
    process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
PID.write_text(str(process.pid), encoding="utf-8")

print("started pid:", process.pid)
print("log:", LOG)
print("out:", OUT)
print("save_dir:", SAVE_DIR)
print("command:", " ".join(cmd))
