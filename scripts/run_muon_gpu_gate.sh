#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT_DIR/unit-tests/build/glades-unit-tests"
BUILD_DIR="$ROOT_DIR/build"
TEST_BUILD_DIR="$ROOT_DIR/unit-tests/build"

GPU_DEVICE=0
REPEATS=5
RUN_ACCEPTANCE=0
SKIP_BUILD=0
MUON_GEOM=1.0
MUON_PRED=0.05
MUON_MAX_ASPECT=1.50
MUON_MIN_DIM=8
MUON_DAMPING=0.01
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR_DEFAULT="$ROOT_DIR/artifacts/muon_gpu_gate_${TIMESTAMP}"
OUT_DIR="${OUT_DIR:-$OUT_DIR_DEFAULT}"
EPOCHS=(1 2 3 4)

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Runs the MUON-lite-vs-AdamW GPU benchmark gate and saves raw logs plus TSV summaries.

Variants:
  - adamw
  - muon_lite

Options:
  --gpu-device N              CUDA device id to request (default: 0)
  --repeats N                 Repeats per epoch point (default: 5)
  --out-dir PATH              Output directory (default: $OUT_DIR_DEFAULT)
  --skip-build                Skip build + atlas-controller steps
  --acceptance                Also run the 10-repeat acceptance pass
  --atlas-muon-geometry-scale X
  --atlas-muon-predictive-scale X
  --atlas-muon-max-aspect X
  --atlas-muon-min-dim N
  --atlas-muon-damping X
  --help                      Show this message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-device)
      GPU_DEVICE="$2"
      shift 2
      ;;
    --repeats)
      REPEATS="$2"
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
    --acceptance)
      RUN_ACCEPTANCE=1
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

mkdir -p "$OUT_DIR/raw"

SUMMARY_TSV="$OUT_DIR/epoch_sweep_summary.tsv"
ACCEPT_TSV="$OUT_DIR/acceptance_summary.tsv"
RUN_LOG="$OUT_DIR/run.log"

cat > "$SUMMARY_TSV" <<'EOF'
benchmark	optimizer	epochs	train_s_mean	train_s_pm	tok_s_mean	tok_s_pm	train_nll_mean	train_nll_pm	test_nll_mean	test_nll_pm	status	logfile
EOF

cat > "$ACCEPT_TSV" <<'EOF'
benchmark	optimizer	train_s_mean	train_s_pm	tok_s_mean	tok_s_pm	train_nll_mean	train_nll_pm	test_nll_mean	test_nll_pm	status	logfile
EOF

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

append_epoch_summary() {
  local benchmark="$1"
  local optimizer="$2"
  local epochs="$3"
  local logfile="$4"
  awk -v benchmark="$benchmark" -v optimizer="$optimizer" -v epochs="$epochs" -v logfile="$logfile" '
    /^(AdamW|ATLAS-MUON)[[:space:]]/ && NF >= 20 {
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
             benchmark, optimizer, epochs,
             $2, $4, $5, $7, $8, $10, $14, $16, $20, logfile;
      exit 0;
    }
  ' "$logfile" >> "$SUMMARY_TSV"
}

append_acceptance_summary() {
  local benchmark="$1"
  local optimizer="$2"
  local logfile="$3"
  awk -v benchmark="$benchmark" -v optimizer="$optimizer" -v logfile="$logfile" '
    /^(AdamW|ATLAS-MUON)[[:space:]]/ && NF >= 20 {
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
             benchmark, optimizer,
             $2, $4, $5, $7, $8, $10, $14, $16, $20, logfile;
      exit 0;
    }
  ' "$logfile" >> "$ACCEPT_TSV"
}

run_variant_epoch() {
  local name="$1"
  local benchmark="$2"
  local epochs="$3"
  local optimizer="$4"
  shift 4
  run_capture "$name" \
    "$BIN" atlas-alt-bench \
    --mode "$benchmark" \
    --token-epochs "$epochs" \
    --repeats "$REPEATS" \
    "$@" \
    --gpu-enable 1 \
    --gpu-device "$GPU_DEVICE"
  append_epoch_summary "$benchmark" "$optimizer" "$epochs" "$OUT_DIR/raw/${name}.log"
}

run_variant_accept() {
  local name="$1"
  local benchmark="$2"
  local optimizer="$3"
  shift 3
  run_capture "$name" \
    "$BIN" atlas-alt-bench \
    --mode "$benchmark" \
    --repeats 10 \
    "$@" \
    --gpu-enable 1 \
    --gpu-device "$GPU_DEVICE"
  append_acceptance_summary "$benchmark" "$optimizer" "$OUT_DIR/raw/${name}.log"
}

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  run_capture 00_nvidia_smi nvidia-smi
  run_capture 01_build_main cmake --build "$BUILD_DIR" -j4
  run_capture 02_build_tests cmake --build "$TEST_BUILD_DIR" -j4 --target glades-unit-tests
  run_capture 03_atlas_controller "$BIN" atlas-controller
fi

run_capture 04_smoke_muon_gpu \
  "$BIN" atlas-alt-bench \
  --mode token-lm \
  --token-epochs 1 \
  --repeats 1 \
  --variant muon \
  --atlas-muon-geometry-scale "$MUON_GEOM" \
  --atlas-muon-predictive-scale "$MUON_PRED" \
  --atlas-muon-max-aspect "$MUON_MAX_ASPECT" \
  --atlas-muon-min-dim "$MUON_MIN_DIM" \
  --atlas-muon-damping "$MUON_DAMPING" \
  --gpu-enable 1 \
  --gpu-device "$GPU_DEVICE"

for benchmark in token-lm-document token-lm-corpus-large; do
  for epochs in "${EPOCHS[@]}"; do
    run_variant_epoch "sweep_${benchmark}_adamw_e${epochs}" \
      "$benchmark" "$epochs" "adamw" \
      --variant adamw

    run_variant_epoch "sweep_${benchmark}_muon_lite_e${epochs}" \
      "$benchmark" "$epochs" "muon_lite" \
      --variant muon \
      --atlas-muon-geometry-scale "$MUON_GEOM" \
      --atlas-muon-predictive-scale "$MUON_PRED" \
      --atlas-muon-max-aspect "$MUON_MAX_ASPECT" \
      --atlas-muon-min-dim "$MUON_MIN_DIM" \
      --atlas-muon-damping "$MUON_DAMPING"
  done
done

if [[ "$RUN_ACCEPTANCE" -eq 1 ]]; then
  for benchmark in token-lm-document token-lm-corpus-large; do
    run_variant_accept "accept_${benchmark}_adamw" \
      "$benchmark" "adamw" \
      --variant adamw

    run_variant_accept "accept_${benchmark}_muon_lite" \
      "$benchmark" "muon_lite" \
      --variant muon \
      --atlas-muon-geometry-scale "$MUON_GEOM" \
      --atlas-muon-predictive-scale "$MUON_PRED" \
      --atlas-muon-max-aspect "$MUON_MAX_ASPECT" \
      --atlas-muon-min-dim "$MUON_MIN_DIM" \
      --atlas-muon-damping "$MUON_DAMPING"
  done
fi

cat > "$OUT_DIR/README.txt" <<EOF
Output directory: $OUT_DIR

Files:
  raw/*.log                 Raw benchmark output
  epoch_sweep_summary.tsv   5-repeat epoch sweep summary
  acceptance_summary.tsv    10-repeat acceptance summary (when requested)

Primary decision rule:
  MUON-lite only survives if it beats AdamW on wall-clock time-to-target
  at matched or better TestNLL on token-lm-document or token-lm-corpus-large.
EOF

echo "Saved results to: $OUT_DIR"
echo "Epoch sweep summary: $SUMMARY_TSV"
if [[ "$RUN_ACCEPTANCE" -eq 1 ]]; then
  echo "Acceptance summary: $ACCEPT_TSV"
fi
