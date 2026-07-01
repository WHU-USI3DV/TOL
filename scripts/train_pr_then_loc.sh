#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

: "${BACKBONE:=clip-b16}"
: "${DEVICE:=cuda}"
: "${BATCH_SIZE:=32}"
: "${PR_EPOCHS:=30}"
: "${LOC_EPOCHS:=30}"
: "${LR:=1e-5}"
: "${WEIGHT_DECAY:=0.01}"
: "${LOC_WEIGHT:=0.1}"
: "${TEXT_NUM:=5}"
: "${POS_SCALE:=25.0}"
: "${ORDER:=TNSWE}"
: "${OUTPUT_ROOT:=outputs/train_pr_then_loc}"

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: $name" >&2
    exit 1
  fi
}

require_var IMAGE_DIR
require_var TEXT_DIR
require_var POSE_OSM_DIR
require_var POSE_TEXT_DIR

CITIES_ARGS=()
if [[ -n "${CITIES:-}" ]]; then
  read -r -a CITY_LIST <<< "$CITIES"
  CITIES_ARGS=(--cities "${CITY_LIST[@]}")
fi

PR_OUTPUT="$OUTPUT_ROOT/pr"
LOC_OUTPUT="$OUTPUT_ROOT/full"
mkdir -p "$PR_OUTPUT" "$LOC_OUTPUT"

python tools/train.py \
  --stage pr \
  --backbone "$BACKBONE" \
  --image-dir "$IMAGE_DIR" \
  --text-dir "$TEXT_DIR" \
  --pose-osm-dir "$POSE_OSM_DIR" \
  --pose-text-dir "$POSE_TEXT_DIR" \
  --output-dir "$PR_OUTPUT" \
  --device "$DEVICE" \
  --epochs "$PR_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --text-num "$TEXT_NUM" \
  --pos-scale "$POS_SCALE" \
  --order "$ORDER" \
  "${CITIES_ARGS[@]}"

PR_CKPT="$(printf "%s/tol_epoch_%03d.pth" "$PR_OUTPUT" "$PR_EPOCHS")"

python tools/train.py \
  --stage full \
  --checkpoint "$PR_CKPT" \
  --backbone "$BACKBONE" \
  --image-dir "$IMAGE_DIR" \
  --text-dir "$TEXT_DIR" \
  --pose-osm-dir "$POSE_OSM_DIR" \
  --pose-text-dir "$POSE_TEXT_DIR" \
  --output-dir "$LOC_OUTPUT" \
  --device "$DEVICE" \
  --epochs "$LOC_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --loc-weight "$LOC_WEIGHT" \
  --text-num "$TEXT_NUM" \
  --pos-scale "$POS_SCALE" \
  --order "$ORDER" \
  "${CITIES_ARGS[@]}"

echo "PR checkpoint: $PR_CKPT"
echo "Full model output: $LOC_OUTPUT"
