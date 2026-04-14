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
ARGOS_GEOM=1.0
ARGOS_ORTH=0.5
ARGOS_PRED=0.05
ARGOS_TRUST=0.20
ARGOS_WARMUP=32
ARGOS_SCOPE=head
ARGOS_CADENCE=1
ARGOS_ORTH_CADENCE=1
ARGOS_MAX_ASPECT=1.50
ARGOS_MIN_DIM=8
ARGOS_DAMPING=0.01
ARGOS_OBS=0.75
ARGOS_HEAD=0.35
ARGOS_LATE=0.00
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR_DEFAULT="$ROOT_DIR/artifacts/argos_gpu_gate_${TIMESTAMP}"
OUT_DIR="${OUT_DIR:-$OUT_DIR_DEFAULT}"
EPOCHS=(1 2 3 4)

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Runs the ARGOS-vs-AdamW GPU benchmark gate and saves raw logs plus TSV summaries.

Variants:
  - adamw
  - argos

Options:
  --gpu-device N                  CUDA device id to request (default: 0)
  --repeats N                     Repeats per epoch point (default: 5)
  --out-dir PATH                  Output directory (default: $OUT_DIR_DEFAULT)
  --skip-build                    Skip build + controller/unit-test steps
  --acceptance                    Also run the 10-repeat acceptance pass
  --atlas-argos-geometry-scale X
  --atlas-argos-orthogonal-scale X
  --atlas-argos-predictive-scale X
  --atlas-argos-trust-radius X
  --atlas-argos-warmup-steps N
  --atlas-argos-scope S
  --atlas-argos-cadence N
  --atlas-argos-orth-cadence N
  --atlas-argos-max-aspect X
  --atlas-argos-min-dim N
  --atlas-argos-damping X
  --atlas-argos-observability-scale X
  --atlas-argos-head-bonus X
  --atlas-argos-late-bonus X
  --help                          Show this message
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
    --atlas-argos-geometry-scale)
      ARGOS_GEOM="$2"
      shift 2
      ;;
    --atlas-argos-orthogonal-scale)
      ARGOS_ORTH="$2"
      shift 2
      ;;
    --atlas-argos-predictive-scale)
      ARGOS_PRED="$2"
      shift 2
      ;;
    --atlas-argos-trust-radius)
      ARGOS_TRUST="$2"
      shift 2
      ;;
    --atlas-argos-warmup-steps)
      ARGOS_WARMUP="$2"
      shift 2
      ;;
    --atlas-argos-scope)
      ARGOS_SCOPE="$2"
      shift 2
      ;;
    --atlas-argos-cadence)
      ARGOS_CADENCE="$2"
      shift 2
      ;;
    --atlas-argos-orth-cadence)
      ARGOS_ORTH_CADENCE="$2"
      shift 2
      ;;
    --atlas-argos-max-aspect)
      ARGOS_MAX_ASPECT="$2"
      shift 2
      ;;
    --atlas-argos-min-dim)
      ARGOS_MIN_DIM="$2"
      shift 2
      ;;
    --atlas-argos-damping)
      ARGOS_DAMPING="$2"
      shift 2
      ;;
    --atlas-argos-observability-scale)
      ARGOS_OBS="$2"
      shift 2
      ;;
    --atlas-argos-head-bonus)
      ARGOS_HEAD="$2"
      shift 2
      ;;
    --atlas-argos-late-bonus)
      ARGOS_LATE="$2"
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
    /^(AdamW|ATLAS-ARGOS)[[:space:]]/ && NF >= 20 {
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
    /^(AdamW|ATLAS-ARGOS)[[:space:]]/ && NF >= 20 {
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
  run_capture 04_argos_core "$BIN" atlas-argos-core
  run_capture 05_argos_parity "$BIN" atlas-argos-parity
fi

run_capture 06_smoke_argos_gpu \
  "$BIN" atlas-alt-bench \
  --mode token-lm \
  --token-epochs 1 \
  --repeats 1 \
  --variant argos \
  --atlas-argos-geometry-scale "$ARGOS_GEOM" \
  --atlas-argos-orthogonal-scale "$ARGOS_ORTH" \
  --atlas-argos-predictive-scale "$ARGOS_PRED" \
  --atlas-argos-trust-radius "$ARGOS_TRUST" \
  --atlas-argos-warmup-steps "$ARGOS_WARMUP" \
  --atlas-argos-scope "$ARGOS_SCOPE" \
  --atlas-argos-cadence "$ARGOS_CADENCE" \
  --atlas-argos-orth-cadence "$ARGOS_ORTH_CADENCE" \
  --atlas-argos-max-aspect "$ARGOS_MAX_ASPECT" \
  --atlas-argos-min-dim "$ARGOS_MIN_DIM" \
  --atlas-argos-damping "$ARGOS_DAMPING" \
  --atlas-argos-observability-scale "$ARGOS_OBS" \
  --atlas-argos-head-bonus "$ARGOS_HEAD" \
  --atlas-argos-late-bonus "$ARGOS_LATE" \
  --gpu-enable 1 \
  --gpu-device "$GPU_DEVICE"

for benchmark in token-lm-document token-lm-corpus-large; do
  for epochs in "${EPOCHS[@]}"; do
    run_variant_epoch "sweep_${benchmark}_adamw_e${epochs}" \
      "$benchmark" "$epochs" "adamw" \
      --variant adamw

    run_variant_epoch "sweep_${benchmark}_argos_e${epochs}" \
      "$benchmark" "$epochs" "argos" \
      --variant argos \
      --atlas-argos-geometry-scale "$ARGOS_GEOM" \
      --atlas-argos-orthogonal-scale "$ARGOS_ORTH" \
      --atlas-argos-predictive-scale "$ARGOS_PRED" \
      --atlas-argos-trust-radius "$ARGOS_TRUST" \
      --atlas-argos-warmup-steps "$ARGOS_WARMUP" \
      --atlas-argos-scope "$ARGOS_SCOPE" \
      --atlas-argos-cadence "$ARGOS_CADENCE" \
      --atlas-argos-orth-cadence "$ARGOS_ORTH_CADENCE" \
      --atlas-argos-max-aspect "$ARGOS_MAX_ASPECT" \
      --atlas-argos-min-dim "$ARGOS_MIN_DIM" \
      --atlas-argos-damping "$ARGOS_DAMPING" \
      --atlas-argos-observability-scale "$ARGOS_OBS" \
      --atlas-argos-head-bonus "$ARGOS_HEAD" \
      --atlas-argos-late-bonus "$ARGOS_LATE"
  done
done

if [[ "$RUN_ACCEPTANCE" -eq 1 ]]; then
  for benchmark in token-lm-document token-lm-corpus-large; do
    run_variant_accept "accept_${benchmark}_adamw" \
      "$benchmark" "adamw" \
      --variant adamw

    run_variant_accept "accept_${benchmark}_argos" \
      "$benchmark" "argos" \
      --variant argos \
      --atlas-argos-geometry-scale "$ARGOS_GEOM" \
      --atlas-argos-orthogonal-scale "$ARGOS_ORTH" \
      --atlas-argos-predictive-scale "$ARGOS_PRED" \
      --atlas-argos-trust-radius "$ARGOS_TRUST" \
      --atlas-argos-warmup-steps "$ARGOS_WARMUP" \
      --atlas-argos-scope "$ARGOS_SCOPE" \
      --atlas-argos-cadence "$ARGOS_CADENCE" \
      --atlas-argos-orth-cadence "$ARGOS_ORTH_CADENCE" \
      --atlas-argos-max-aspect "$ARGOS_MAX_ASPECT" \
      --atlas-argos-min-dim "$ARGOS_MIN_DIM" \
      --atlas-argos-damping "$ARGOS_DAMPING" \
      --atlas-argos-observability-scale "$ARGOS_OBS" \
      --atlas-argos-head-bonus "$ARGOS_HEAD" \
      --atlas-argos-late-bonus "$ARGOS_LATE"
  done
fi

cat > "$OUT_DIR/README.txt" <<EOF
Output directory: $OUT_DIR

Files:
  raw/*.log                 Raw benchmark and unit-test output
  epoch_sweep_summary.tsv   ${REPEATS}-repeat epoch sweep summary
  acceptance_summary.tsv    10-repeat acceptance summary (when requested)
  run.log                   Full command log

Files to send back for review:
  - epoch_sweep_summary.tsv
  - acceptance_summary.tsv (if present)
  - raw/05_argos_parity.log
  - raw/06_smoke_argos_gpu.log
  - any raw/sweep_*_argos_*.log that look suspicious

Primary decision rule:
  ARGOS only survives as a practical branch if this scheduled head-only gate improves held-out TestNLL
  enough to justify wall-clock cost against AdamW on token-lm-document or
  token-lm-corpus-large while keeping the dedicated ARGOS unit tests green.
EOF

echo "Saved results to: $OUT_DIR"
echo "Epoch sweep summary: $SUMMARY_TSV"
if [[ "$RUN_ACCEPTANCE" -eq 1 ]]; then
  echo "Acceptance summary: $ACCEPT_TSV"
fi
