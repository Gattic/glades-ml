#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT_DIR/unit-tests/build/glades-unit-tests"
NSYS_BIN="$(command -v nsys)"
NSYS_HOST_DIR="$(cd "$(dirname "$NSYS_BIN")/../lib/nsight-systems/host-linux-x64" 2>/dev/null && pwd || true)"
QDSTRM_IMPORTER="${QDSTRM_IMPORTER:-${NSYS_HOST_DIR}/QdstrmImporter}"

GPU_DEVICE=0
BENCHMARKS=("token-lm-document" "token-lm-corpus-large")
EPOCHS=1
SKIP_BUILD=0
MUON_GEOM=1.0
MUON_PRED=0.05
MUON_MAX_ASPECT=1.50
MUON_MIN_DIM=8
MUON_DAMPING=0.01
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR_DEFAULT="$ROOT_DIR/artifacts/muon_nsys_${TIMESTAMP}"
OUT_DIR="${OUT_DIR:-$OUT_DIR_DEFAULT}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Profiles AdamW vs MUON-lite with Nsight Systems on the hard transformer benchmarks.
This is a profiling script, not an acceptance gate.

Options:
  --gpu-device N                   CUDA device id to request (default: 0)
  --benchmark NAME                 Benchmark to profile; may be passed multiple times
                                   (default: token-lm-document and token-lm-corpus-large)
  --epochs N                       Token epochs per run (default: 1)
  --out-dir PATH                   Output directory
  --skip-build                     Skip build + controller verification
  --atlas-muon-geometry-scale X
  --atlas-muon-predictive-scale X
  --atlas-muon-max-aspect X
  --atlas-muon-min-dim N
  --atlas-muon-damping X
  --qdstrm-importer PATH             Explicit QdstrmImporter path
  --help                           Show this message
EOF
}

BENCHMARKS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-device)
      GPU_DEVICE="$2"
      shift 2
      ;;
    --benchmark)
      BENCHMARKS+=("$2")
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --atlas-muon-geometry-scale)
      MUON_GEOM="$2"
      shift 2
      ;;
    --atlas-muon-predictive-scale)
      MUON_PRED="$2"
      shift 2
      ;;
    --atlas-muon-max-aspect)
      MUON_MAX_ASPECT="$2"
      shift 2
      ;;
    --atlas-muon-min-dim)
      MUON_MIN_DIM="$2"
      shift 2
      ;;
    --atlas-muon-damping)
      MUON_DAMPING="$2"
      shift 2
      ;;
    --qdstrm-importer)
      QDSTRM_IMPORTER="$2"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ${#BENCHMARKS[@]} -eq 0 ]]; then
  BENCHMARKS=("token-lm-document" "token-lm-corpus-large")
fi

mkdir -p "$OUT_DIR/raw" "$OUT_DIR/stats"

RUN_LOG="$OUT_DIR/run.log"

log_cmd() {
  printf '$' >> "$RUN_LOG"
  printf ' %q' "$@" >> "$RUN_LOG"
  printf '\n' >> "$RUN_LOG"
}

run_capture() {
  local name="$1"
  shift
  local logfile="$OUT_DIR/raw/${name}.log"
  echo "[run] $name" | tee -a "$RUN_LOG" >&2
  log_cmd "$@"
  {
    printf '$'
    printf ' %q' "$@"
    printf '\n'
    "$@"
  } 2>&1 | tee "$logfile"
}

run_nsys_profile() {
  local tag="$1"
  shift
  local rep_base="$OUT_DIR/${tag}"
  local rep_file="${rep_base}.nsys-rep"
  local qdstrm_file="${rep_base}.qdstrm"
  local stats_base="$OUT_DIR/stats/${tag}"

  echo "[profile] $tag" | tee -a "$RUN_LOG" >&2
  log_cmd nsys profile --force-overwrite=true -o "$rep_base" --sample=none --cpuctxsw=none --trace=cuda,nvtx,osrt "$@"
  nsys profile \
    --force-overwrite=true \
    -o "$rep_base" \
    --sample=none \
    --cpuctxsw=none \
    --trace=cuda,nvtx,osrt \
    "$@" \
    >"$OUT_DIR/raw/${tag}.stdout.log" 2>"$OUT_DIR/raw/${tag}.stderr.log"

  if [[ ! -f "$rep_file" && -f "$qdstrm_file" ]]; then
    if [[ -x "$QDSTRM_IMPORTER" ]]; then
      log_cmd "$QDSTRM_IMPORTER" -f -i "$qdstrm_file" -o "$rep_file"
      "$QDSTRM_IMPORTER" -f -i "$qdstrm_file" -o "$rep_file" \
        >"$OUT_DIR/raw/${tag}.import.stdout.log" 2>"$OUT_DIR/raw/${tag}.import.stderr.log"
    else
      echo "missing QdstrmImporter: $QDSTRM_IMPORTER" | tee -a "$RUN_LOG" >&2
      return 1
    fi
  fi

  if [[ ! -f "$rep_file" ]]; then
    echo "nsys profile did not produce a report for $tag" | tee -a "$RUN_LOG" >&2
    return 1
  fi

  log_cmd nsys stats --force-overwrite true --report cudaapisum,gpukernsum,gpumemtimesum,osrtsum --format csv --output "$stats_base" "$rep_file"
  nsys stats \
    --force-overwrite true \
    --report cudaapisum,gpukernsum,gpumemtimesum,osrtsum \
    --format csv \
    --output "$stats_base" \
    "$rep_file" \
    >"$OUT_DIR/raw/${tag}.stats.stdout.log" 2>"$OUT_DIR/raw/${tag}.stats.stderr.log"
}

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  run_capture 00_nvidia_smi nvidia-smi
  run_capture 01_build_main cmake --build "$ROOT_DIR/build" -j4
  run_capture 02_build_tests cmake --build "$ROOT_DIR/unit-tests/build" -j4 --target glades-unit-tests
  run_capture 03_atlas_controller "$BIN" atlas-controller
fi

for benchmark in "${BENCHMARKS[@]}"; do
  run_nsys_profile "${benchmark}_adamw_e${EPOCHS}" \
    "$BIN" atlas-alt-bench \
    --mode "$benchmark" \
    --token-epochs "$EPOCHS" \
    --repeats 1 \
    --variant adamw \
    --gpu-enable 1 \
    --gpu-device "$GPU_DEVICE"

  run_nsys_profile "${benchmark}_muon_lite_e${EPOCHS}" \
    "$BIN" atlas-alt-bench \
    --mode "$benchmark" \
    --token-epochs "$EPOCHS" \
    --repeats 1 \
    --variant muon \
    --atlas-muon-geometry-scale "$MUON_GEOM" \
    --atlas-muon-predictive-scale "$MUON_PRED" \
    --atlas-muon-max-aspect "$MUON_MAX_ASPECT" \
    --atlas-muon-min-dim "$MUON_MIN_DIM" \
    --atlas-muon-damping "$MUON_DAMPING" \
    --gpu-enable 1 \
    --gpu-device "$GPU_DEVICE"
done

echo "Saved Nsight profiles to: $OUT_DIR"
