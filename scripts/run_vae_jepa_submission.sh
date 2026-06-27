#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data/ehl-paris-medical-image-retrieval}"
OUT="${OUT:-data/vae_jepa_submission.csv}"
DEVICE="${DEVICE:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PRECOMPUTED_PT="${PRECOMPUTED_PT:-}"

ARGS=(
  src/volumetric_jepa_vae_submission.py
  --data-root "$DATA_ROOT" \
  --train-pairs-csv "$DATA_ROOT/dataset1/train_pairs.csv" \
  --sample-submission data/sample_submission.csv \
  --query-csv "$DATA_ROOT/dataset1/val_queries.csv" \
  --gallery-csv "$DATA_ROOT/dataset1/val_gallery.csv" \
  --query-csv "$DATA_ROOT/dataset1/test_queries.csv" \
  --gallery-csv "$DATA_ROOT/dataset1/test_gallery.csv" \
  --query-csv "$DATA_ROOT/dataset2/val_queries.csv" \
  --gallery-csv "$DATA_ROOT/dataset2/val_gallery.csv" \
  --query-csv "$DATA_ROOT/dataset2/test_queries.csv" \
  --gallery-csv "$DATA_ROOT/dataset2/test_gallery.csv" \
  --query-csv "$DATA_ROOT/dataset3/val_queries.csv" \
  --gallery-csv "$DATA_ROOT/dataset3/val_gallery.csv" \
  --query-csv "$DATA_ROOT/dataset3/test_queries.csv" \
  --gallery-csv "$DATA_ROOT/dataset3/test_gallery.csv" \
  --out "$OUT" \
  --image-size "${IMAGE_SIZE:-96}" \
  --slice-step "${SLICE_STEP:-5}" \
  --max-slices "${MAX_SLICES:-32}" \
  --patch-size "${PATCH_SIZE:-16}" \
  --token-dim "${TOKEN_DIM:-128}" \
  --vae-latent-dim "${VAE_LATENT_DIM:-128}" \
  --vit-depth "${VIT_DEPTH:-2}" \
  --axial-depth "${AXIAL_DEPTH:-2}" \
  --heads "${HEADS:-4}" \
  --vae-epochs "${VAE_EPOCHS:-5}" \
  --jepa-epochs "${JEPA_EPOCHS:-15}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --lr "${LR:-0.001}" \
  --score-window "${SCORE_WINDOW:-1}" \
  --trim-fraction "${TRIM_FRACTION:-0.85}" \
  --save-dir "${SAVE_DIR:-artifacts/vae_jepa}"
)

if [[ -n "${INCLUDE_DATASET3_TEST_FOR_VAE:-}" ]]; then
  ARGS+=(--include-dataset3-test-for-vae)
fi
if [[ -n "$PRECOMPUTED_PT" ]]; then
  ARGS+=(--precomputed-pt "$PRECOMPUTED_PT")
fi
if [[ -n "$DEVICE" ]]; then
  ARGS+=(--device "$DEVICE")
fi
if [[ -n "${MAX_VAE_VOLUMES:-}" ]]; then
  ARGS+=(--max-vae-volumes "$MAX_VAE_VOLUMES")
fi
if [[ -n "${MAX_TRAIN_PAIRS:-}" ]]; then
  ARGS+=(--max-train-pairs "$MAX_TRAIN_PAIRS")
fi

"$PYTHON_BIN" "${ARGS[@]}"
