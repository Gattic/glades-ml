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
PACT_GEOM=1.0
PACT_CADENCE=8
PACT_COST=0.0010
PACT_PROMOTE=0.0
PACT_DEMOTE=-0.0005
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR_DEFAULT="$ROOT_DIR/artifacts/pact_gpu_gate_${TIMESTAMP}"
OUT_DIR="${OUT_DIR:-$OUT_DIR_DEFAULT}"
EPOCHS=(1 2 3 4)

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Runs the PACT-lite-vs-AdamW GPU benchmark gate and saves raw logs plus TSV summaries.

Variants:
  - adamw
  - pact_lite  (PACT with low-rank off, predictive transport off)

Options:
  --gpu-device N            CUDA device id to request (default: 0)
  --repeats N               Repeats per epoch point (default: 5)
  --out-dir PATH            Output directory (default: $OUT_DIR_DEFAULT)
  --skip-build              Skip build + atlas-controller steps
  --acceptance              Also run the 10-repeat acceptance pass
  --pact-geometry-scale X   PACT geometry scale (default: 1.0)
  --pact-factor-cadence N   PACT factor cadence (default: 8)
  --pact-cost-scale X       PACT analytical cost penalty scale (default: 0.0010)
  --pact-promote X          PACT promotion threshold (default: 0.0)
  --pact-demote X           PACT demotion threshold (default: -0.0005)
  --help                    Show this message
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
    --pact-geometry-scale)
      PACT_GEOM="$2"
      shift 2
      ;;
    --pact-factor-cadence)
      PACT_CADENCE="$2"
      shift 2
      ;;
    --pact-cost-scale)
      PACT_COST="$2"
      shift 2
      ;;
    --pact-promote)
      PACT_PROMOTE="$2"
      shift 2
      ;;
    --pact-demote)
      PACT_DEMOTE="$2"
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
    /^(AdamW|ATLAS-PACT)[[:space:]]/ && NF >= 20 {
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
    /^(AdamW|ATLAS-PACT)[[:space:]]/ && NF >= 20 {
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

run_capture 04_smoke_pact_gpu \
  "$BIN" atlas-alt-bench \
  --mode token-lm \
  --token-epochs 1 \
  --repeats 1 \
  --variant pact \
  --atlas-pact-low-rank 0 \
  --atlas-pact-predictive-scale 0 \
  --atlas-pact-geometry-scale "$PACT_GEOM" \
  --atlas-pact-factor-cadence "$PACT_CADENCE" \
  --atlas-pact-cost-scale "$PACT_COST" \
  --atlas-pact-promote-threshold "$PACT_PROMOTE" \
  --atlas-pact-demote-threshold "$PACT_DEMOTE" \
  --gpu-enable 1 \
  --gpu-device "$GPU_DEVICE"

for benchmark in token-lm-document token-lm-corpus-large; do
  for epochs in "${EPOCHS[@]}"; do
    run_variant_epoch "sweep_${benchmark}_adamw_e${epochs}" \
      "$benchmark" "$epochs" "adamw" \
      --variant adamw

    run_variant_epoch "sweep_${benchmark}_pact_lite_e${epochs}" \
      "$benchmark" "$epochs" "pact_lite" \
      --variant pact \
      --atlas-pact-low-rank 0 \
      --atlas-pact-predictive-scale 0 \
      --atlas-pact-geometry-scale "$PACT_GEOM" \
      --atlas-pact-factor-cadence "$PACT_CADENCE" \
      --atlas-pact-cost-scale "$PACT_COST" \
      --atlas-pact-promote-threshold "$PACT_PROMOTE" \
      --atlas-pact-demote-threshold "$PACT_DEMOTE"
  done
done

if [[ "$RUN_ACCEPTANCE" -eq 1 ]]; then
  for benchmark in token-lm-document token-lm-corpus-large; do
    run_variant_accept "accept_${benchmark}_adamw" \
      "$benchmark" "adamw" \
      --variant adamw

    run_variant_accept "accept_${benchmark}_pact_lite" \
      "$benchmark" "pact_lite" \
      --variant pact \
      --atlas-pact-low-rank 0 \
      --atlas-pact-predictive-scale 0 \
      --atlas-pact-geometry-scale "$PACT_GEOM" \
      --atlas-pact-factor-cadence "$PACT_CADENCE" \
      --atlas-pact-cost-scale "$PACT_COST" \
      --atlas-pact-promote-threshold "$PACT_PROMOTE" \
      --atlas-pact-demote-threshold "$PACT_DEMOTE"
  done
fi

cat > "$OUT_DIR/README.txt" <<EOF
Output directory: $OUT_DIR

Files:
- run.log: command trace
- epoch_sweep_summary.tsv: one row per benchmark/optimizer/epoch point
- acceptance_summary.tsv: optional 10-repeat summary rows when --acceptance is used
- raw/*.log: full raw benchmark outputs

Quick comparison examples:

  column -t -s \$'\\t' "$SUMMARY_TSV"

  rg '^token-lm-document' "$SUMMARY_TSV" | column -t -s \$'\\t'

  rg '^token-lm-corpus-large' "$SUMMARY_TSV" | column -t -s \$'\\t'
EOF

echo
echo "Saved results to: $OUT_DIR"
echo "Epoch sweep summary: $SUMMARY_TSV"
if [[ "$RUN_ACCEPTANCE" -eq 1 ]]; then
  echo "Acceptance summary: $ACCEPT_TSV"
fi
