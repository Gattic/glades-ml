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
SCALES_CSV="0.25,0.5,0.75,1.0"
BENCHMARKS_CSV="token-lm-document,token-lm-corpus-large"
EPOCHS_CSV="1,2,3,4"
ECHO_SCOPE=all
ECHO_CADENCE=1
ECHO_TRUST_SCALE=0.0
ECHO_PREDICTIVE_SCALE=0.0
ECHO_STRUCTURAL_SCALE=0.0
ECHO_STRUCTURAL_GROUPS=1
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR_DEFAULT="$ROOT_DIR/artifacts/echo_geometry_sweep_${TIMESTAMP}"
OUT_DIR="${OUT_DIR:-$OUT_DIR_DEFAULT}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Runs a narrow ECHO geometry-scale GPU sweep against AdamW and saves raw logs plus TSV summaries.

Defaults:
  benchmarks: token-lm-document,token-lm-corpus-large
  scales:     0.25,0.5,0.75,1.0
  epochs:     1,2,3,4

Options:
  --gpu-device N        CUDA device id to request (default: 0)
  --repeats N           Repeats per epoch point (default: 5)
  --scales CSV          Comma-separated ECHO geometry scales
  --benchmarks CSV      Comma-separated benchmark list
  --epochs CSV          Comma-separated epoch list
  --echo-cadence N      Optimizer steps between ECHO metric refreshes (default: 1)
  --echo-scope NAME     ECHO scope: all|large-only|late-head|late-head-large (default: all)
  --echo-trust-scale X  ECHO trust-gate strength (default: 0.0)
  --echo-predictive-scale X
                        ECHO bounded predictive blend strength (default: 0.0)
  --echo-structural-scale X
                        ECHO grouped structural factor strength (default: 0.0)
  --echo-structural-groups N
                        ECHO contiguous row/col group count (default: 1)
  --out-dir PATH        Output directory (default: $OUT_DIR_DEFAULT)
  --skip-build          Skip build + unit test verification steps
  --acceptance          Also run the 10-repeat acceptance pass
  --help                Show this message
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
    --scales)
      SCALES_CSV="$2"
      shift 2
      ;;
    --benchmarks)
      BENCHMARKS_CSV="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS_CSV="$2"
      shift 2
      ;;
    --echo-cadence|--atlas-echo-cadence)
      ECHO_CADENCE="$2"
      shift 2
      ;;
    --echo-scope|--atlas-echo-scope)
      ECHO_SCOPE="$2"
      shift 2
      ;;
    --echo-trust-scale|--atlas-echo-trust-scale)
      ECHO_TRUST_SCALE="$2"
      shift 2
      ;;
    --echo-predictive-scale|--atlas-echo-predictive-scale)
      ECHO_PREDICTIVE_SCALE="$2"
      shift 2
      ;;
    --echo-structural-scale|--atlas-echo-structural-scale)
      ECHO_STRUCTURAL_SCALE="$2"
      shift 2
      ;;
    --echo-structural-groups|--atlas-echo-structural-groups)
      ECHO_STRUCTURAL_GROUPS="$2"
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

IFS=',' read -r -a SCALES <<< "$SCALES_CSV"
IFS=',' read -r -a BENCHMARKS <<< "$BENCHMARKS_CSV"
IFS=',' read -r -a EPOCHS <<< "$EPOCHS_CSV"

mkdir -p "$OUT_DIR/raw"

ECHO_COMMON_ARGS=(
  --atlas-echo-cadence "$ECHO_CADENCE"
  --atlas-echo-scope "$ECHO_SCOPE"
  --atlas-echo-trust-scale "$ECHO_TRUST_SCALE"
  --atlas-echo-predictive-scale "$ECHO_PREDICTIVE_SCALE"
  --atlas-echo-structural-scale "$ECHO_STRUCTURAL_SCALE"
  --atlas-echo-structural-groups "$ECHO_STRUCTURAL_GROUPS"
)

SUMMARY_TSV="$OUT_DIR/epoch_sweep_summary.tsv"
ACCEPT_TSV="$OUT_DIR/acceptance_summary.tsv"
RUN_LOG="$OUT_DIR/run.log"

cat > "$SUMMARY_TSV" <<'EOF'
benchmark	optimizer	echo_scale	epochs	train_s_mean	train_s_pm	tok_s_mean	tok_s_pm	train_nll_mean	train_nll_pm	test_nll_mean	test_nll_pm	status	logfile
EOF

cat > "$ACCEPT_TSV" <<'EOF'
benchmark	optimizer	echo_scale	train_s_mean	train_s_pm	tok_s_mean	tok_s_pm	train_nll_mean	train_nll_pm	test_nll_mean	test_nll_pm	status	logfile
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
  local scale="$3"
  local epochs="$4"
  local logfile="$5"
  awk -v benchmark="$benchmark" -v optimizer="$optimizer" -v scale="$scale" -v epochs="$epochs" -v logfile="$logfile" '
    /^(AdamW|ATLAS-ECHO)[[:space:]]/ && NF >= 20 {
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
             benchmark, optimizer, scale, epochs,
             $2, $4, $5, $7, $8, $10, $14, $16, $20, logfile;
      exit 0;
    }
  ' "$logfile" >> "$SUMMARY_TSV"
}

append_acceptance_summary() {
  local benchmark="$1"
  local optimizer="$2"
  local scale="$3"
  local logfile="$4"
  awk -v benchmark="$benchmark" -v optimizer="$optimizer" -v scale="$scale" -v logfile="$logfile" '
    /^(AdamW|ATLAS-ECHO)[[:space:]]/ && NF >= 20 {
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
             benchmark, optimizer, scale,
             $2, $4, $5, $7, $8, $10, $14, $16, $20, logfile;
      exit 0;
    }
  ' "$logfile" >> "$ACCEPT_TSV"
}

scale_tag() {
  local scale="$1"
  scale="${scale//- /}"
  scale="${scale//./p}"
  scale="${scale//- /}"
  printf '%s' "$scale"
}

run_variant_epoch() {
  local name="$1"
  local benchmark="$2"
  local epochs="$3"
  local optimizer="$4"
  local scale="$5"
  shift 5
  run_capture "$name" \
    "$BIN" atlas-alt-bench \
    --mode "$benchmark" \
    --token-epochs "$epochs" \
    --repeats "$REPEATS" \
    "$@" \
    --gpu-enable 1 \
    --gpu-device "$GPU_DEVICE"
  append_epoch_summary "$benchmark" "$optimizer" "$scale" "$epochs" "$OUT_DIR/raw/${name}.log"
}

run_variant_accept() {
  local name="$1"
  local benchmark="$2"
  local optimizer="$3"
  local scale="$4"
  shift 4
  run_capture "$name" \
    "$BIN" atlas-alt-bench \
    --mode "$benchmark" \
    --repeats 10 \
    "$@" \
    --gpu-enable 1 \
    --gpu-device "$GPU_DEVICE"
  append_acceptance_summary "$benchmark" "$optimizer" "$scale" "$OUT_DIR/raw/${name}.log"
}

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  run_capture 00_nvidia_smi nvidia-smi
  run_capture 01_build_main cmake --build "$BUILD_DIR" -j4
  run_capture 02_build_tests cmake --build "$TEST_BUILD_DIR" -j4 --target glades-unit-tests
  run_capture 03_atlas_controller "$BIN" atlas-controller
  run_capture 04_echo_core "$BIN" atlas-echo-core
  run_capture 05_echo_micro "$BIN" atlas-echo-micro
fi

run_capture 06_smoke_echo_gpu \
  "$BIN" atlas-alt-bench \
  --mode token-lm \
  --token-epochs 1 \
  --repeats 1 \
  --variant echo \
  --atlas-echo-geometry-scale "${SCALES[0]}" \
  "${ECHO_COMMON_ARGS[@]}" \
  --gpu-enable 1 \
  --gpu-device "$GPU_DEVICE"

for benchmark in "${BENCHMARKS[@]}"; do
  for epochs in "${EPOCHS[@]}"; do
    run_variant_epoch "sweep_${benchmark}_adamw_e${epochs}" \
      "$benchmark" "$epochs" "adamw" "-" \
      --variant adamw

    for scale in "${SCALES[@]}"; do
      tag="$(scale_tag "$scale")"
      run_variant_epoch "sweep_${benchmark}_echo_s${tag}_e${epochs}" \
        "$benchmark" "$epochs" "echo" "$scale" \
        --variant echo \
        --atlas-echo-geometry-scale "$scale" \
        "${ECHO_COMMON_ARGS[@]}"
    done
  done
done

if [[ "$RUN_ACCEPTANCE" -eq 1 ]]; then
  for benchmark in "${BENCHMARKS[@]}"; do
    run_variant_accept "accept_${benchmark}_adamw" \
      "$benchmark" "adamw" "-" \
      --variant adamw

    for scale in "${SCALES[@]}"; do
      tag="$(scale_tag "$scale")"
      run_variant_accept "accept_${benchmark}_echo_s${tag}" \
        "$benchmark" "echo" "$scale" \
        --variant echo \
        --atlas-echo-geometry-scale "$scale" \
        "${ECHO_COMMON_ARGS[@]}"
    done
  done
fi

cat > "$OUT_DIR/README.txt" <<EOF
Output directory: $OUT_DIR

Files:
- run.log: command trace
- epoch_sweep_summary.tsv: one row per benchmark/optimizer/scale/epoch point
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
