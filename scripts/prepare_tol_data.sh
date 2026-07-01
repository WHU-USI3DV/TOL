#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

: "${MODE:=all}"
: "${TILE_SIZE:=50}"
: "${PPM:=4}"
: "${TEXT_RADIUS:=100}"
: "${ANGLE_SAMPLES:=360}"
: "${RUN_TEXT:=1}"
: "${NUSCENES_OUTPUT_ROOT:=data/TOL-N}"
: "${K360_OUTPUT_ROOT:=data/TOL-K360}"
: "${NUSCENES_OSM_DIR:=maploc/data/nuscenes}"
: "${K360_OSM_PATH:=maploc/data/kitti/karlsruhe.osm}"
: "${NUSCENES_CITIES:=singapore-onenorth singapore-hollandvillage singapore-queenstown boston-seaport}"

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: $name" >&2
    exit 1
  fi
}

prepare_nuscenes() {
  require_var NUSCENES_ROOT
  python data_prepare/prepare_nuscenes_tiles.py \
    --nuscenes-root "$NUSCENES_ROOT" \
    --osm-dir "$NUSCENES_OSM_DIR" \
    --output-root "$NUSCENES_OUTPUT_ROOT" \
    --tile-size "$TILE_SIZE" \
    --ppm "$PPM"

  if [[ "$RUN_TEXT" == "1" ]]; then
    read -r -a NUSCENES_CITY_ARGS <<< "$NUSCENES_CITIES"
    python data_prepare/generate_text_from_raster.py \
      --raster-dir "$NUSCENES_OUTPUT_ROOT/raster/raster_osm_${TILE_SIZE}_${PPM}" \
      --output-dir "$NUSCENES_OUTPUT_ROOT/texts/texts_osm_${TILE_SIZE}_${PPM}" \
      --radius "$TEXT_RADIUS" \
      --angle-samples "$ANGLE_SAMPLES" \
      --cities "${NUSCENES_CITY_ARGS[@]}"
  fi
}

prepare_kitti360() {
  require_var K360_POSE_ROOT
  python data_prepare/prepare_kitti360_tiles.py \
    --pose-root "$K360_POSE_ROOT" \
    --osm-path "$K360_OSM_PATH" \
    --output-root "$K360_OUTPUT_ROOT" \
    --tile-size "$TILE_SIZE" \
    --ppm "$PPM"

  if [[ "$RUN_TEXT" == "1" ]]; then
    if [[ -n "${K360_CITIES:-}" ]]; then
      read -r -a K360_CITY_ARGS <<< "$K360_CITIES"
    else
      mapfile -t K360_CITY_ARGS < <(find "$K360_OUTPUT_ROOT/raster/raster_osm_${TILE_SIZE}_${PPM}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
    fi
    python data_prepare/generate_text_from_raster.py \
      --raster-dir "$K360_OUTPUT_ROOT/raster/raster_osm_${TILE_SIZE}_${PPM}" \
      --output-dir "$K360_OUTPUT_ROOT/texts/texts_osm_${TILE_SIZE}_${PPM}" \
      --radius "$TEXT_RADIUS" \
      --angle-samples "$ANGLE_SAMPLES" \
      --cities "${K360_CITY_ARGS[@]}"
  fi
}

case "$MODE" in
  all)
    prepare_nuscenes
    prepare_kitti360
    ;;
  nuscenes|tol-n|TOL-N)
    prepare_nuscenes
    ;;
  kitti360|tol-k360|TOL-K360)
    prepare_kitti360
    ;;
  *)
    echo "Unknown MODE: $MODE" >&2
    echo "Use MODE=all, MODE=nuscenes, or MODE=kitti360." >&2
    exit 1
    ;;
esac
