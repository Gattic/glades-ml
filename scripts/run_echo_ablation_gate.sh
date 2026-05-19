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
ECHO_SCOPE=late-head
ECHO_GEOMETRY_SCALE=1.0
ECHO_FINAL_GEOMETRY_SCALE=1.0
ECHO_DECAY_STEPS=0
ECHO_CADENCE=1
TRUST_SCALE=1.0
PREDICTIVE_SCALE=0.35
STRUCTURAL_SCALE=0.5
STRUCTURAL_GROUPS=2
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR_DEFAULT="$ROOT_DIR/artifacts/echo_ablation_gate_${TIMESTAMP}"
OUT_DIR="${OUT_DIR:-$OUT_DIR_DEFAULT}"
EPOCHS=(1 2 3 4)

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Runs the ECHO hybrid ablation gate on the hard transformer benchmarks.

Variants:
  - adamw
  - echo_base           (plain ECHO late-head baseline)
  - echo_trust          (trust only)
  - echo_trust_pred     (trust + predictive)
  - echo_trust_struct   (trust + structural)

Options:
  --gpu-device N              CUDA device id to request (default: 0)
  --repeats N                 Repeats per epoch point (default: 5)
  --echo-scope NAME           ECHO scope: all|large-only|late-head|late-head-large (default: late-head)
  --echo-geometry-scale X     ECHO operand-geometry strength (default: 1.0)
  --echo-final-geometry-scale X
                              Final ECHO geometry strength after linear decay (default: 1.0)
  --echo-decay-steps N        Optimizer steps for linear ECHO geometry decay (default: 0)
  --echo-cadence N            Optimizer steps between ECHO metric refreshes (default: 1)
  --trust-scale X             Trust-only and hybrid trust scale (default: 1.0)
  --predictive-scale X        Predictive blend strength for trust+predictive (default: 0.35)
  --structural-scale X        Structural factor strength for trust+structural (default: 0.5)
  --structural-groups N       Structural group count for trust+structural (default: 2)
  --out-dir PATH              Output directory (default: $OUT_DIR_DEFAULT)
  --skip-build                Skip build + unit test verification steps
  --acceptance                Also run the 10-repeat acceptance pass
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
    --echo-scope|--atlas-echo-scope)
      ECHO_SCOPE="$2"
      shift 2
      ;;
    --echo-geometry-scale|--atlas-echo-geometry-scale)
      ECHO_GEOMETRY_SCALE="$2"
      shift 2
      ;;
    --echo-final-geometry-scale|--atlas-echo-final-geometry-scale)
      ECHO_FINAL_GEOMETRY_SCALE="$2"
      shift 2
      ;;
    --echo-decay-steps|--atlas-echo-decay-steps)
      ECHO_DECAY_STEPS="$2"
      shift 2
      ;;
    --echo-cadence|--atlas-echo-cadence)
      ECHO_CADENCE="$2"
      shift 2
      ;;
    --trust-scale|--echo-trust-scale|--atlas-echo-trust-scale)
      TRUST_SCALE="$2"
      shift 2
      ;;
    --predictive-scale|--echo-predictive-scale|--atlas-echo-predictive-scale)
      PREDICTIVE_SCALE="$2"
      shift 2
      ;;
    --structural-scale|--echo-structural-scale|--atlas-echo-structural-scale)
      STRUCTURAL_SCALE="$2"
      shift 2
      ;;
    --structural-groups|--echo-structural-groups|--atlas-echo-structural-groups)
      STRUCTURAL_GROUPS="$2"
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

mkdir -p "$OUT_DIR/raw"

SUMMARY_TSV="$OUT_DIR/epoch_sweep_summary.tsv"
ACCEPT_TSV="$OUT_DIR/acceptance_summary.tsv"
RUN_LOG="$OUT_DIR/run.log"

cat > "$SUMMARY_TSV" <<'EOF'
benchmark	optimizer	label	trust_scale	predictive_scale	structural_scale	structural_groups	epochs	train_s_mean	train_s_pm	tok_s_mean	tok_s_pm	train_nll_mean	train_nll_pm	test_nll_mean	test_nll_pm	status	logfile
EOF

cat > "$ACCEPT_TSV" <<'EOF'
benchmark	optimizer	label	trust_scale	predictive_scale	structural_scale	structural_groups	train_s_mean	train_s_pm	tok_s_mean	tok_s_pm	train_nll_mean	train_nll_pm	test_nll_mean	test_nll_pm	status	logfile
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
  local label="$3"
  local trust_scale="$4"
  local predictive_scale="$5"
  local structural_scale="$6"
  local structural_groups="$7"
  local epochs="$8"
  local logfile="$9"
  awk -v benchmark="$benchmark" -v optimizer="$optimizer" -v label="$label" \
      -v trust_scale="$trust_scale" -v predictive_scale="$predictive_scale" \
      -v structural_scale="$structural_scale" -v structural_groups="$structural_groups" \
      -v epochs="$epochs" -v logfile="$logfile" '
    /^(AdamW|ATLAS-ECHO)[[:space:]]/ && NF >= 20 {
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
             benchmark, optimizer, label,
             trust_scale, predictive_scale, structural_scale, structural_groups, epochs,
             $2, $4, $5, $7, $8, $10, $14, $16, $20, logfile;
      exit 0;
    }
  ' "$logfile" >> "$SUMMARY_TSV"
}

append_acceptance_summary() {
  local benchmark="$1"
  local optimizer="$2"
  local label="$3"
  local trust_scale="$4"
  local predictive_scale="$5"
  local structural_scale="$6"
  local structural_groups="$7"
  local logfile="$8"
  awk -v benchmark="$benchmark" -v optimizer="$optimizer" -v label="$label" \
      -v trust_scale="$trust_scale" -v predictive_scale="$predictive_scale" \
      -v structural_scale="$structural_scale" -v structural_groups="$structural_groups" \
      -v logfile="$logfile" '
    /^(AdamW|ATLAS-ECHO)[[:space:]]/ && NF >= 20 {
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
             benchmark, optimizer, label,
             trust_scale, predictive_scale, structural_scale, structural_groups,
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
  local label="$5"
  local trust_scale="$6"
  local predictive_scale="$7"
  local structural_scale="$8"
  local structural_groups="$9"
  shift 9
  run_capture "$name" \
    "$BIN" atlas-alt-bench \
    --mode "$benchmark" \
    --token-epochs "$epochs" \
    --repeats "$REPEATS" \
    "$@" \
    --gpu-enable 1 \
    --gpu-device "$GPU_DEVICE"
  append_epoch_summary "$benchmark" "$optimizer" "$label" \
    "$trust_scale" "$predictive_scale" "$structural_scale" "$structural_groups" "$epochs" \
    "$OUT_DIR/raw/${name}.log"
}

run_variant_accept() {
  local name="$1"
  local benchmark="$2"
  local optimizer="$3"
  local label="$4"
  local trust_scale="$5"
  local predictive_scale="$6"
  local structural_scale="$7"
  local structural_groups="$8"
  shift 8
  run_capture "$name" \
    "$BIN" atlas-alt-bench \
    --mode "$benchmark" \
    --repeats 10 \
    "$@" \
    --gpu-enable 1 \
    --gpu-device "$GPU_DEVICE"
  append_acceptance_summary "$benchmark" "$optimizer" "$label" \
    "$trust_scale" "$predictive_scale" "$structural_scale" "$structural_groups" \
    "$OUT_DIR/raw/${name}.log"
}

COMMON_ECHO_ARGS=(
  --variant echo
  --atlas-echo-geometry-scale "$ECHO_GEOMETRY_SCALE"
  --atlas-echo-final-geometry-scale "$ECHO_FINAL_GEOMETRY_SCALE"
  --atlas-echo-decay-steps "$ECHO_DECAY_STEPS"
  --atlas-echo-cadence "$ECHO_CADENCE"
  --atlas-echo-scope "$ECHO_SCOPE"
)

BASE_ARGS=(
  "${COMMON_ECHO_ARGS[@]}"
  --atlas-echo-trust-scale 0.0
  --atlas-echo-predictive-scale 0.0
  --atlas-echo-structural-scale 0.0
  --atlas-echo-structural-groups 1
)

TRUST_ARGS=(
  "${COMMON_ECHO_ARGS[@]}"
  --atlas-echo-trust-scale "$TRUST_SCALE"
  --atlas-echo-predictive-scale 0.0
  --atlas-echo-structural-scale 0.0
  --atlas-echo-structural-groups 1
)

TRUST_PRED_ARGS=(
  "${COMMON_ECHO_ARGS[@]}"
  --atlas-echo-trust-scale "$TRUST_SCALE"
  --atlas-echo-predictive-scale "$PREDICTIVE_SCALE"
  --atlas-echo-structural-scale 0.0
  --atlas-echo-structural-groups 1
)

TRUST_STRUCT_ARGS=(
  "${COMMON_ECHO_ARGS[@]}"
  --atlas-echo-trust-scale "$TRUST_SCALE"
  --atlas-echo-predictive-scale 0.0
  --atlas-echo-structural-scale "$STRUCTURAL_SCALE"
  --atlas-echo-structural-groups "$STRUCTURAL_GROUPS"
)

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  run_capture 00_nvidia_smi nvidia-smi
  run_capture 01_build_main cmake --build "$BUILD_DIR" -j4
  run_capture 02_build_tests cmake --build "$TEST_BUILD_DIR" -j4 --target glades-unit-tests
  run_capture 03_echo_core "$BIN" atlas-echo-core
  run_capture 04_echo_micro "$BIN" atlas-echo-micro
  run_capture 05_echo_parity "$BIN" atlas-echo-parity
fi

run_capture 06_smoke_echo_base \
  "$BIN" atlas-alt-bench \
  --mode token-lm \
  --token-epochs 1 \
  --repeats 1 \
  "${BASE_ARGS[@]}" \
  --gpu-enable 1 \
  --gpu-device "$GPU_DEVICE"

for benchmark in token-lm-document token-lm-corpus-large; do
  for epochs in "${EPOCHS[@]}"; do
    run_variant_epoch "sweep_${benchmark}_adamw_e${epochs}" \
      "$benchmark" "$epochs" "adamw" "adamw" "-" "-" "-" "-" \
      --variant adamw

    run_variant_epoch "sweep_${benchmark}_echo_base_e${epochs}" \
      "$benchmark" "$epochs" "echo" "echo_base" "0.0" "0.0" "0.0" "1" \
      "${BASE_ARGS[@]}"

    run_variant_epoch "sweep_${benchmark}_echo_trust_e${epochs}" \
      "$benchmark" "$epochs" "echo" "echo_trust" "$TRUST_SCALE" "0.0" "0.0" "1" \
      "${TRUST_ARGS[@]}"

    run_variant_epoch "sweep_${benchmark}_echo_trust_pred_e${epochs}" \
      "$benchmark" "$epochs" "echo" "echo_trust_pred" "$TRUST_SCALE" "$PREDICTIVE_SCALE" "0.0" "1" \
      "${TRUST_PRED_ARGS[@]}"

    run_variant_epoch "sweep_${benchmark}_echo_trust_struct_e${epochs}" \
      "$benchmark" "$epochs" "echo" "echo_trust_struct" "$TRUST_SCALE" "0.0" "$STRUCTURAL_SCALE" "$STRUCTURAL_GROUPS" \
      "${TRUST_STRUCT_ARGS[@]}"
  done
done

if [[ "$RUN_ACCEPTANCE" -eq 1 ]]; then
  for benchmark in token-lm-document token-lm-corpus-large; do
    run_variant_accept "accept_${benchmark}_adamw" \
      "$benchmark" "adamw" "adamw" "-" "-" "-" "-" \
      --variant adamw

    run_variant_accept "accept_${benchmark}_echo_base" \
      "$benchmark" "echo" "echo_base" "0.0" "0.0" "0.0" "1" \
      "${BASE_ARGS[@]}"

    run_variant_accept "accept_${benchmark}_echo_trust" \
      "$benchmark" "echo" "echo_trust" "$TRUST_SCALE" "0.0" "0.0" "1" \
      "${TRUST_ARGS[@]}"

    run_variant_accept "accept_${benchmark}_echo_trust_pred" \
      "$benchmark" "echo" "echo_trust_pred" "$TRUST_SCALE" "$PREDICTIVE_SCALE" "0.0" "1" \
      "${TRUST_PRED_ARGS[@]}"

    run_variant_accept "accept_${benchmark}_echo_trust_struct" \
      "$benchmark" "echo" "echo_trust_struct" "$TRUST_SCALE" "0.0" "$STRUCTURAL_SCALE" "$STRUCTURAL_GROUPS" \
      "${TRUST_STRUCT_ARGS[@]}"
  done
fi

cat > "$OUT_DIR/README.txt" <<EOF
Output directory: $OUT_DIR

Purpose:
- compare the live plain ECHO late-head baseline against the three next ablations
- isolate which borrowed term helps document quality without damaging corpus-large

Variants:
- adamw
- echo_base
- echo_trust
- echo_trust_pred
- echo_trust_struct

Files:
- run.log: command trace
- epoch_sweep_summary.tsv: one row per benchmark/optimizer/ablation/epoch point
- acceptance_summary.tsv: optional 10-repeat summary rows when --acceptance is used
- raw/*.log: full raw benchmark outputs
EOF

echo
echo "Saved results to: $OUT_DIR"
echo "Epoch sweep summary: $SUMMARY_TSV"
if [[ "$RUN_ACCEPTANCE" -eq 1 ]]; then
  echo "Acceptance summary: $ACCEPT_TSV"
fi
