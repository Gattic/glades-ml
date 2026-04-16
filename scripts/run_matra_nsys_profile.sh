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
MATRA_GEOM=1.0
MATRA_ORTH=0.5
MATRA_PRED=0.05
MATRA_TRUST=0.5
MATRA_CADENCE=1
MATRA_ORTH_CADENCE=1
MATRA_MAX_ASPECT=1.50
MATRA_MIN_DIM=8
MATRA_DAMPING=0.01
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR_DEFAULT="$ROOT_DIR/artifacts/matra_nsys_${TIMESTAMP}"
OUT_DIR="${OUT_DIR:-$OUT_DIR_DEFAULT}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Profiles AdamW vs MATRA with Nsight Systems on the hard transformer benchmarks.
This is a profiling script, not an acceptance gate.

Options:
  --gpu-device N                     CUDA device id to request (default: 0)
  --benchmark NAME                   Benchmark to profile; may be passed multiple times
                                     (default: token-lm-document and token-lm-corpus-large)
  --epochs N                         Token epochs per run (default: 1)
  --out-dir PATH                     Output directory
  --skip-build                       Skip build + controller verification
  --atlas-matra-geometry-scale X
  --atlas-matra-orthogonal-scale X
  --atlas-matra-predictive-scale X
  --atlas-matra-trust-radius X
  --atlas-matra-cadence N
  --atlas-matra-orth-cadence N
  --atlas-matra-max-aspect X
  --atlas-matra-min-dim N
  --atlas-matra-damping X
  --qdstrm-importer PATH             Explicit QdstrmImporter path
  --help                             Show this message
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
    --atlas-matra-geometry-scale)
      MATRA_GEOM="$2"
      shift 2
      ;;
    --atlas-matra-orthogonal-scale)
      MATRA_ORTH="$2"
      shift 2
      ;;
    --atlas-matra-predictive-scale)
      MATRA_PRED="$2"
      shift 2
      ;;
    --atlas-matra-trust-radius)
      MATRA_TRUST="$2"
      shift 2
      ;;
    --atlas-matra-cadence)
      MATRA_CADENCE="$2"
      shift 2
      ;;
    --atlas-matra-orth-cadence)
      MATRA_ORTH_CADENCE="$2"
      shift 2
      ;;
    --atlas-matra-max-aspect)
      MATRA_MAX_ASPECT="$2"
      shift 2
      ;;
    --atlas-matra-min-dim)
      MATRA_MIN_DIM="$2"
      shift 2
      ;;
    --atlas-matra-damping)
      MATRA_DAMPING="$2"
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
  run_capture 04_matra_core "$BIN" atlas-matra-core
  run_capture 05_matra_parity "$BIN" atlas-matra-parity
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

  run_nsys_profile "${benchmark}_matra_e${EPOCHS}" \
    "$BIN" atlas-alt-bench \
    --mode "$benchmark" \
    --token-epochs "$EPOCHS" \
    --repeats 1 \
    --variant matra \
    --atlas-matra-geometry-scale "$MATRA_GEOM" \
    --atlas-matra-orthogonal-scale "$MATRA_ORTH" \
    --atlas-matra-predictive-scale "$MATRA_PRED" \
    --atlas-matra-trust-radius "$MATRA_TRUST" \
    --atlas-matra-cadence "$MATRA_CADENCE" \
    --atlas-matra-orth-cadence "$MATRA_ORTH_CADENCE" \
    --atlas-matra-max-aspect "$MATRA_MAX_ASPECT" \
    --atlas-matra-min-dim "$MATRA_MIN_DIM" \
    --atlas-matra-damping "$MATRA_DAMPING" \
    --gpu-enable 1 \
    --gpu-device "$GPU_DEVICE"
done

echo "Saved Nsight profiles to: $OUT_DIR"
