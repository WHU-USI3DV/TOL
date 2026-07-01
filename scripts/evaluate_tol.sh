#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

: "${BACKBONE:=clip-b16}"
: "${DEVICE:=cuda}"
: "${BATCH_SIZE:=1}"
: "${TEXT_NUM:=5}"
: "${TOPK:=10}"
: "${POS_SCALE:=25.0}"
: "${ORDER:=TNSWE}"
: "${SAVE_DIR:=outputs/eval}"
: "${RECALL_THRESHOLDS:=10 25}"
: "${SUCCESS_THRESHOLDS:=5 10 25}"
: "${ERROR_PERCENTILES:=5 10 25}"

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: $name" >&2
    exit 1
  fi
}

require_file() {
  local name="$1"
  require_var "$name"
  if [[ ! -f "${!name}" ]]; then
    echo "File does not exist: $name=${!name}" >&2
    exit 1
  fi
}

require_dir() {
  local name="$1"
  require_var "$name"
  if [[ ! -d "${!name}" ]]; then
    echo "Directory does not exist: $name=${!name}" >&2
    exit 1
  fi
}

require_file CHECKPOINT
require_dir IMAGE_DIR
require_dir TEXT_DIR
require_dir POSE_OSM_DIR
require_dir POSE_TEXT_DIR

CITIES_ARGS=()
if [[ -n "${CITIES:-}" ]]; then
  read -r -a CITY_LIST <<< "$CITIES"
  CITIES_ARGS=(--cities "${CITY_LIST[@]}")
fi

read -r -a RECALL_THRESHOLD_ARGS <<< "$RECALL_THRESHOLDS"
read -r -a SUCCESS_THRESHOLD_ARGS <<< "$SUCCESS_THRESHOLDS"
read -r -a ERROR_PERCENTILE_ARGS <<< "$ERROR_PERCENTILES"

python tools/evaluate.py \
  --backbone "$BACKBONE" \
  --checkpoint "$CHECKPOINT" \
  --image-dir "$IMAGE_DIR" \
  --text-dir "$TEXT_DIR" \
  --pose-osm-dir "$POSE_OSM_DIR" \
  --pose-text-dir "$POSE_TEXT_DIR" \
  --save-dir "$SAVE_DIR" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --text-num "$TEXT_NUM" \
  --topk "$TOPK" \
  --pos-scale "$POS_SCALE" \
  --order "$ORDER" \
  --recall-thresholds "${RECALL_THRESHOLD_ARGS[@]}" \
  --success-thresholds "${SUCCESS_THRESHOLD_ARGS[@]}" \
  --error-percentiles "${ERROR_PERCENTILE_ARGS[@]}" \
  "${CITIES_ARGS[@]}"
