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
BENCHMARKS_CSV="token-lm-document,token-lm-corpus-large"
EPOCHS_CSV="1,2,3,4"
SCHEDULES_CSV="0.25:0.25:0,0.75:0.25:48,1.0:0.25:48,0.75:0.25:96,1.0:0.25:96"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR_DEFAULT="$ROOT_DIR/artifacts/echo_schedule_sweep_${TIMESTAMP}"
OUT_DIR="${OUT_DIR:-$OUT_DIR_DEFAULT}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Runs a narrow ECHO schedule GPU sweep against AdamW and saves raw logs plus TSV summaries.

Defaults:
  benchmarks: token-lm-document,token-lm-corpus-large
  epochs:     1,2,3,4
  schedules:  0.25:0.25:0,0.75:0.25:48,1.0:0.25:48,0.75:0.25:96,1.0:0.25:96

Schedule format:
  start_scale:final_scale:decay_steps

Options:
  --gpu-device N        CUDA device id to request (default: 0)
  --repeats N           Repeats per epoch point (default: 5)
  --benchmarks CSV      Comma-separated benchmark list
  --epochs CSV          Comma-separated epoch list
  --schedules CSV       Comma-separated schedule specs (start:final:steps)
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
    --benchmarks)
      BENCHMARKS_CSV="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS_CSV="$2"
      shift 2
      ;;
    --schedules)
      SCHEDULES_CSV="$2"
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

IFS=',' read -r -a BENCHMARKS <<< "$BENCHMARKS_CSV"
IFS=',' read -r -a EPOCHS <<< "$EPOCHS_CSV"
IFS=',' read -r -a SCHEDULES <<< "$SCHEDULES_CSV"

mkdir -p "$OUT_DIR/raw"

SUMMARY_TSV="$OUT_DIR/epoch_sweep_summary.tsv"
ACCEPT_TSV="$OUT_DIR/acceptance_summary.tsv"
RUN_LOG="$OUT_DIR/run.log"

cat > "$SUMMARY_TSV" <<'EOF'
benchmark	optimizer	echo_scale_start	echo_scale_final	echo_decay_steps	epochs	train_s_mean	train_s_pm	tok_s_mean	tok_s_pm	train_nll_mean	train_nll_pm	test_nll_mean	test_nll_pm	status	logfile
EOF

cat > "$ACCEPT_TSV" <<'EOF'
benchmark	optimizer	echo_scale_start	echo_scale_final	echo_decay_steps	train_s_mean	train_s_pm	tok_s_mean	tok_s_pm	train_nll_mean	train_nll_pm	test_nll_mean	test_nll_pm	status	logfile
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
  local start_scale="$3"
  local final_scale="$4"
  local decay_steps="$5"
  local epochs="$6"
  local logfile="$7"
  awk -v benchmark="$benchmark" -v optimizer="$optimizer" -v start_scale="$start_scale" -v final_scale="$final_scale" \
      -v decay_steps="$decay_steps" -v epochs="$epochs" -v logfile="$logfile" '
    /^(AdamW|ATLAS-ECHO)[[:space:]]/ && NF >= 20 {
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
             benchmark, optimizer, start_scale, final_scale, decay_steps, epochs,
             $2, $4, $5, $7, $8, $10, $14, $16, $20, logfile;
      exit 0;
    }
  ' "$logfile" >> "$SUMMARY_TSV"
}

append_acceptance_summary() {
  local benchmark="$1"
  local optimizer="$2"
  local start_scale="$3"
  local final_scale="$4"
  local decay_steps="$5"
  local logfile="$6"
  awk -v benchmark="$benchmark" -v optimizer="$optimizer" -v start_scale="$start_scale" -v final_scale="$final_scale" \
      -v decay_steps="$decay_steps" -v logfile="$logfile" '
    /^(AdamW|ATLAS-ECHO)[[:space:]]/ && NF >= 20 {
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
             benchmark, optimizer, start_scale, final_scale, decay_steps,
             $2, $4, $5, $7, $8, $10, $14, $16, $20, logfile;
      exit 0;
    }
  ' "$logfile" >> "$ACCEPT_TSV"
}

parse_schedule_spec() {
  local spec="$1"
  local start final decay
  IFS=':' read -r start final decay <<< "$spec"
  if [[ -z "${start:-}" || -z "${final:-}" || -z "${decay:-}" ]]; then
    echo "invalid schedule spec: $spec" >&2
    exit 2
  fi
  printf '%s %s %s\n' "$start" "$final" "$decay"
}

tag_component() {
  local value="$1"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "$value"
}

schedule_tag() {
  local start="$1"
  local final="$2"
  local decay="$3"
  printf 's%s_f%s_d%s' "$(tag_component "$start")" "$(tag_component "$final")" "$decay"
}

run_variant_epoch() {
  local name="$1"
  local benchmark="$2"
  local epochs="$3"
  local optimizer="$4"
  local start_scale="$5"
  local final_scale="$6"
  local decay_steps="$7"
  shift 7
  run_capture "$name" \
    "$BIN" atlas-alt-bench \
    --mode "$benchmark" \
    --token-epochs "$epochs" \
    --repeats "$REPEATS" \
    "$@" \
    --gpu-enable 1 \
    --gpu-device "$GPU_DEVICE"
  append_epoch_summary "$benchmark" "$optimizer" "$start_scale" "$final_scale" "$decay_steps" "$epochs" "$OUT_DIR/raw/${name}.log"
}

run_variant_accept() {
  local name="$1"
  local benchmark="$2"
  local optimizer="$3"
  local start_scale="$4"
  local final_scale="$5"
  local decay_steps="$6"
  shift 6
  run_capture "$name" \
    "$BIN" atlas-alt-bench \
    --mode "$benchmark" \
    --repeats 10 \
    "$@" \
    --gpu-enable 1 \
    --gpu-device "$GPU_DEVICE"
  append_acceptance_summary "$benchmark" "$optimizer" "$start_scale" "$final_scale" "$decay_steps" "$OUT_DIR/raw/${name}.log"
}

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  run_capture 00_nvidia_smi nvidia-smi
  run_capture 01_build_main cmake --build "$BUILD_DIR" -j4
  run_capture 02_build_tests cmake --build "$TEST_BUILD_DIR" -j4 --target glades-unit-tests
  run_capture 03_atlas_controller "$BIN" atlas-controller
  run_capture 04_echo_core "$BIN" atlas-echo-core
  run_capture 05_echo_micro "$BIN" atlas-echo-micro
fi

read -r SMOKE_START SMOKE_FINAL SMOKE_DECAY <<< "$(parse_schedule_spec "${SCHEDULES[0]}")"
run_capture 06_smoke_echo_gpu \
  "$BIN" atlas-alt-bench \
  --mode token-lm \
  --token-epochs 1 \
  --repeats 1 \
  --variant echo \
  --atlas-echo-geometry-scale "$SMOKE_START" \
  --atlas-echo-final-geometry-scale "$SMOKE_FINAL" \
  --atlas-echo-decay-steps "$SMOKE_DECAY" \
  --gpu-enable 1 \
  --gpu-device "$GPU_DEVICE"

for benchmark in "${BENCHMARKS[@]}"; do
  for epochs in "${EPOCHS[@]}"; do
    run_variant_epoch "sweep_${benchmark}_adamw_e${epochs}" \
      "$benchmark" "$epochs" "adamw" "-" "-" "-" \
      --variant adamw

    for spec in "${SCHEDULES[@]}"; do
      read -r start_scale final_scale decay_steps <<< "$(parse_schedule_spec "$spec")"
      tag="$(schedule_tag "$start_scale" "$final_scale" "$decay_steps")"
      run_variant_epoch "sweep_${benchmark}_echo_${tag}_e${epochs}" \
        "$benchmark" "$epochs" "echo" "$start_scale" "$final_scale" "$decay_steps" \
        --variant echo \
        --atlas-echo-geometry-scale "$start_scale" \
        --atlas-echo-final-geometry-scale "$final_scale" \
        --atlas-echo-decay-steps "$decay_steps"
    done
  done
done

if [[ "$RUN_ACCEPTANCE" -eq 1 ]]; then
  for benchmark in "${BENCHMARKS[@]}"; do
    run_variant_accept "accept_${benchmark}_adamw" \
      "$benchmark" "adamw" "-" "-" "-" \
      --variant adamw

    for spec in "${SCHEDULES[@]}"; do
      read -r start_scale final_scale decay_steps <<< "$(parse_schedule_spec "$spec")"
      tag="$(schedule_tag "$start_scale" "$final_scale" "$decay_steps")"
      run_variant_accept "accept_${benchmark}_echo_${tag}" \
        "$benchmark" "echo" "$start_scale" "$final_scale" "$decay_steps" \
        --variant echo \
        --atlas-echo-geometry-scale "$start_scale" \
        --atlas-echo-final-geometry-scale "$final_scale" \
        --atlas-echo-decay-steps "$decay_steps"
    done
  done
fi

cat > "$OUT_DIR/README.txt" <<EOF
Output directory: $OUT_DIR

Files:
- run.log: command trace
- epoch_sweep_summary.tsv: one row per benchmark/optimizer/schedule/epoch point
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
