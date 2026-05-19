#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT_DIR/unit-tests/build/glades-unit-tests"
BUILD_DIR="$ROOT_DIR/build"
TEST_BUILD_DIR="$ROOT_DIR/unit-tests/build"

GPU_DEVICE=0
REPEATS=5
RUN_ACCEPTANCE=1
SKIP_BUILD=0
BENCHMARK="token-lm-corpus-large"
EPOCHS=(1 2 3 4)

ECHO_GEOMETRY_SCALE=1.0
ECHO_FINAL_GEOMETRY_SCALE=1.0
ECHO_DECAY_STEPS=0
ECHO_CADENCE=1
ECHO_SCOPE="late-head"
ECHO_TRUST_SCALE=0.0
ECHO_PREDICTIVE_SCALE=0.0
ECHO_STRUCTURAL_SCALE=0.0
ECHO_STRUCTURAL_GROUPS=1

MATRA_GEOM=1.0
MATRA_ORTH=0.5
MATRA_PRED=0.05
MATRA_TRUST=0.50
MATRA_CADENCE=1
MATRA_ORTH_CADENCE=2
MATRA_MAX_ASPECT=1.50
MATRA_MIN_DIM=8
MATRA_DAMPING=0.01

ARGOS_GEOM=1.0
ARGOS_ORTH=0.5
ARGOS_PRED=0.05
ARGOS_TRUST=0.20
ARGOS_SCOPE="head"
ARGOS_WARMUPS_CSV="0,16,32"
ARGOS_START_SCALES_CSV="0.00,0.25,0.50"
ARGOS_CADENCE=1
ARGOS_ORTH_CADENCE=1
ARGOS_MAX_ASPECT=1.50
ARGOS_MIN_DIM=8
ARGOS_DAMPING=0.01
ARGOS_OBS=0.75
ARGOS_HEAD=0.35
ARGOS_LATE=0.00

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR_DEFAULT="$ROOT_DIR/artifacts/argos_corpus_frontier_${TIMESTAMP}"
OUT_DIR="${OUT_DIR:-$OUT_DIR_DEFAULT}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Runs a same-run token-lm-corpus-large GPU comparison for:
  - AdamW
  - ATLAS-ECHO
  - ATLAS-MATRA
  - ATLAS-ARGOS warmup/start-scale sweep

Default ARGOS warmup sweep:
  ${ARGOS_WARMUPS_CSV}

Default ARGOS warmup start-scale sweep:
  ${ARGOS_START_SCALES_CSV}

Outputs:
  - epoch_sweep_summary.tsv
  - acceptance_summary.tsv
  - acceptance_rank_by_nll.tsv
  - epoch4_rank_by_nll.tsv
  - argos_acceptance_rank_by_nll.tsv
  - argos_epoch4_rank_by_nll.tsv
  - raw/*.log

Options:
  --gpu-device N                        CUDA device id to request (default: 0)
  --repeats N                           Repeats per epoch point (default: 5)
  --out-dir PATH                        Output directory (default: $OUT_DIR_DEFAULT)
  --skip-build                          Skip build + ARGOS verification (only safe after a clean rebuild)
  --skip-acceptance                     Skip the 10-repeat acceptance pass
  --atlas-echo-geometry-scale X
  --atlas-echo-final-geometry-scale X
  --atlas-echo-decay-steps N
  --atlas-echo-cadence N
  --atlas-echo-scope NAME
  --atlas-echo-trust-scale X
  --atlas-echo-predictive-scale X
  --atlas-echo-structural-scale X
  --atlas-echo-structural-groups N
  --atlas-matra-geometry-scale X
  --atlas-matra-orthogonal-scale X
  --atlas-matra-predictive-scale X
  --atlas-matra-trust-radius X
  --atlas-matra-cadence N
  --atlas-matra-orth-cadence N
  --atlas-matra-max-aspect X
  --atlas-matra-min-dim N
  --atlas-matra-damping X
  --atlas-argos-geometry-scale X
  --atlas-argos-orthogonal-scale X
  --atlas-argos-predictive-scale X
  --atlas-argos-trust-radius X
  --atlas-argos-warmups CSV             Comma-separated warmup step list (default: ${ARGOS_WARMUPS_CSV})
  --atlas-argos-warmup-start-scales CSV Comma-separated warmup start scales in [0,1] (default: ${ARGOS_START_SCALES_CSV})
  --atlas-argos-scope S
  --atlas-argos-cadence N
  --atlas-argos-orth-cadence N
  --atlas-argos-max-aspect X
  --atlas-argos-min-dim N
  --atlas-argos-damping X
  --atlas-argos-observability-scale X
  --atlas-argos-head-bonus X
  --atlas-argos-late-bonus X
  --help                                Show this message
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
    --skip-acceptance)
      RUN_ACCEPTANCE=0
      shift
      ;;
    --atlas-echo-geometry-scale)
      ECHO_GEOMETRY_SCALE="$2"
      shift 2
      ;;
    --atlas-echo-final-geometry-scale)
      ECHO_FINAL_GEOMETRY_SCALE="$2"
      shift 2
      ;;
    --atlas-echo-decay-steps)
      ECHO_DECAY_STEPS="$2"
      shift 2
      ;;
    --atlas-echo-cadence)
      ECHO_CADENCE="$2"
      shift 2
      ;;
    --atlas-echo-scope)
      ECHO_SCOPE="$2"
      shift 2
      ;;
    --atlas-echo-trust-scale)
      ECHO_TRUST_SCALE="$2"
      shift 2
      ;;
    --atlas-echo-predictive-scale)
      ECHO_PREDICTIVE_SCALE="$2"
      shift 2
      ;;
    --atlas-echo-structural-scale)
      ECHO_STRUCTURAL_SCALE="$2"
      shift 2
      ;;
    --atlas-echo-structural-groups)
      ECHO_STRUCTURAL_GROUPS="$2"
      shift 2
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
    --atlas-argos-warmups)
      ARGOS_WARMUPS_CSV="$2"
      shift 2
      ;;
    --atlas-argos-warmup-start-scales)
      ARGOS_START_SCALES_CSV="$2"
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

IFS=',' read -r -a ARGOS_WARMUP_LIST <<< "$ARGOS_WARMUPS_CSV"
if [[ "${#ARGOS_WARMUP_LIST[@]}" -eq 0 ]]; then
  echo "empty --atlas-argos-warmups list" >&2
  exit 2
fi
for warmup in "${ARGOS_WARMUP_LIST[@]}"; do
  if [[ ! "$warmup" =~ ^[0-9]+$ ]]; then
    echo "invalid ARGOS warmup step: $warmup" >&2
    exit 2
  fi
done

IFS=',' read -r -a ARGOS_START_SCALE_LIST <<< "$ARGOS_START_SCALES_CSV"
if [[ "${#ARGOS_START_SCALE_LIST[@]}" -eq 0 ]]; then
  echo "empty --atlas-argos-warmup-start-scales list" >&2
  exit 2
fi
for start_scale in "${ARGOS_START_SCALE_LIST[@]}"; do
  if [[ ! "$start_scale" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "invalid ARGOS warmup start scale: $start_scale" >&2
    exit 2
  fi
  if ! awk -v x="$start_scale" 'BEGIN { exit !(x >= 0.0 && x <= 1.0) }'; then
    echo "ARGOS warmup start scale out of range [0,1]: $start_scale" >&2
    exit 2
  fi
done

argos_scale_label() {
  local scale="$1"
  local label="${scale//./}"
  label="${label//[^0-9]/}"
  if [[ -z "$label" ]]; then
    label="0"
  fi
  printf '%s' "$label"
}

ARGOS_VARIANT_LABELS=()
ARGOS_VARIANT_WARMUPS=()
ARGOS_VARIANT_START_SCALES=()
for warmup in "${ARGOS_WARMUP_LIST[@]}"; do
  if [[ "$warmup" == "0" ]]; then
    ARGOS_VARIANT_LABELS+=("argos_w0")
    ARGOS_VARIANT_WARMUPS+=("$warmup")
    ARGOS_VARIANT_START_SCALES+=("0.00")
    continue
  fi
  for start_scale in "${ARGOS_START_SCALE_LIST[@]}"; do
    local_label="argos_w${warmup}_s$(argos_scale_label "$start_scale")"
    ARGOS_VARIANT_LABELS+=("$local_label")
    ARGOS_VARIANT_WARMUPS+=("$warmup")
    ARGOS_VARIANT_START_SCALES+=("$start_scale")
  done
done

mkdir -p "$OUT_DIR/raw"

SUMMARY_TSV="$OUT_DIR/epoch_sweep_summary.tsv"
ACCEPT_TSV="$OUT_DIR/acceptance_summary.tsv"
ACCEPT_RANK_TSV="$OUT_DIR/acceptance_rank_by_nll.tsv"
E4_RANK_TSV="$OUT_DIR/epoch4_rank_by_nll.tsv"
ARGOS_ACCEPT_RANK_TSV="$OUT_DIR/argos_acceptance_rank_by_nll.tsv"
ARGOS_E4_RANK_TSV="$OUT_DIR/argos_epoch4_rank_by_nll.tsv"
RUN_LOG="$OUT_DIR/run.log"

if [[ "$SKIP_BUILD" -eq 1 ]]; then
  echo "[warn] --skip-build reuses existing binaries; this is unsafe after C++ header/layout changes. Use a clean rebuild first." | tee -a "$RUN_LOG" >&2
fi

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
    /^(AdamW|ATLAS-ECHO|ATLAS-MATRA|ATLAS-ARGOS)[[:space:]]/ && NF >= 20 {
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
    /^(AdamW|ATLAS-ECHO|ATLAS-MATRA|ATLAS-ARGOS)[[:space:]]/ && NF >= 20 {
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
             benchmark, optimizer,
             $2, $4, $5, $7, $8, $10, $14, $16, $20, logfile;
      exit 0;
    }
  ' "$logfile" >> "$ACCEPT_TSV"
}

generate_rank_table() {
  local source_tsv="$1"
  local out_tsv="$2"
  local nll_col="$3"
  local extra_prefix="$4"
  {
    if [[ -n "$extra_prefix" ]]; then
      printf "%s\n" "$extra_prefix"
    fi
    tail -n +2 "$source_tsv" \
      | sort -t $'\t' -k1,1 -k${nll_col},${nll_col}g -k3,3g \
      | awk -F '\t' -v nll_col="$nll_col" '
          BEGIN { OFS = "\t"; prev = ""; rank = 0; }
          {
            if ($1 != prev) {
              prev = $1;
              rank = 1;
            } else {
              rank += 1;
            }
            print $1, rank, $2, $3, $(nll_col), $NF;
          }
        '
  } > "$out_tsv"
}

generate_epoch4_rank() {
  {
    printf "benchmark\trank\toptimizer\tepochs\ttrain_s_mean\ttest_nll_mean\tlogfile\n"
    awk -F '\t' '$3 == "4"' "$SUMMARY_TSV" \
      | sort -t $'\t' -k1,1 -k10,10g -k4,4g \
      | awk -F '\t' '
          BEGIN { OFS = "\t"; prev = ""; rank = 0; }
          {
            if ($1 != prev) {
              prev = $1;
              rank = 1;
            } else {
              rank += 1;
            }
            print $1, rank, $2, $3, $4, $10, $13;
          }
        '
  } > "$E4_RANK_TSV"

  {
    printf "benchmark\trank\toptimizer\tepochs\ttrain_s_mean\ttest_nll_mean\tlogfile\n"
    awk -F '\t' '$1 == "'"$BENCHMARK"'" && $2 ~ /^argos_w/ && $3 == "4"' "$SUMMARY_TSV" \
      | sort -t $'\t' -k10,10g -k4,4g \
      | awk -F '\t' '
          BEGIN { OFS = "\t"; rank = 0; }
          {
            rank += 1;
            print $1, rank, $2, $3, $4, $10, $13;
          }
        '
  } > "$ARGOS_E4_RANK_TSV"
}

generate_acceptance_rank() {
  {
    printf "benchmark\trank\toptimizer\ttrain_s_mean\ttest_nll_mean\tlogfile\n"
    tail -n +2 "$ACCEPT_TSV" \
      | sort -t $'\t' -k1,1 -k9,9g -k3,3g \
      | awk -F '\t' '
          BEGIN { OFS = "\t"; prev = ""; rank = 0; }
          {
            if ($1 != prev) {
              prev = $1;
              rank = 1;
            } else {
              rank += 1;
            }
            print $1, rank, $2, $3, $9, $12;
          }
        '
  } > "$ACCEPT_RANK_TSV"

  {
    printf "benchmark\trank\toptimizer\ttrain_s_mean\ttest_nll_mean\tlogfile\n"
    awk -F '\t' '$1 == "'"$BENCHMARK"'" && $2 ~ /^argos_w/' "$ACCEPT_TSV" \
      | sort -t $'\t' -k9,9g -k3,3g \
      | awk -F '\t' '
          BEGIN { OFS = "\t"; rank = 0; }
          {
            rank += 1;
            print $1, rank, $2, $3, $9, $12;
          }
        '
  } > "$ARGOS_ACCEPT_RANK_TSV"
}

run_variant_epoch() {
  local name="$1"
  local optimizer_label="$2"
  local epochs="$3"
  shift 3
  run_capture "$name" \
    "$BIN" atlas-alt-bench \
    --mode "$BENCHMARK" \
    --token-epochs "$epochs" \
    --repeats "$REPEATS" \
    "$@" \
    --gpu-enable 1 \
    --gpu-device "$GPU_DEVICE"
  append_epoch_summary "$BENCHMARK" "$optimizer_label" "$epochs" "$OUT_DIR/raw/${name}.log"
}

run_variant_accept() {
  local name="$1"
  local optimizer_label="$2"
  shift 2
  run_capture "$name" \
    "$BIN" atlas-alt-bench \
    --mode "$BENCHMARK" \
    --repeats 10 \
    "$@" \
    --gpu-enable 1 \
    --gpu-device "$GPU_DEVICE"
  append_acceptance_summary "$BENCHMARK" "$optimizer_label" "$OUT_DIR/raw/${name}.log"
}

ECHO_ARGS=(
  --variant echo
  --atlas-echo-geometry-scale "$ECHO_GEOMETRY_SCALE"
  --atlas-echo-final-geometry-scale "$ECHO_FINAL_GEOMETRY_SCALE"
  --atlas-echo-decay-steps "$ECHO_DECAY_STEPS"
  --atlas-echo-cadence "$ECHO_CADENCE"
  --atlas-echo-scope "$ECHO_SCOPE"
  --atlas-echo-trust-scale "$ECHO_TRUST_SCALE"
  --atlas-echo-predictive-scale "$ECHO_PREDICTIVE_SCALE"
  --atlas-echo-structural-scale "$ECHO_STRUCTURAL_SCALE"
  --atlas-echo-structural-groups "$ECHO_STRUCTURAL_GROUPS"
)

MATRA_ARGS=(
  --variant matra
  --atlas-matra-geometry-scale "$MATRA_GEOM"
  --atlas-matra-orthogonal-scale "$MATRA_ORTH"
  --atlas-matra-predictive-scale "$MATRA_PRED"
  --atlas-matra-trust-radius "$MATRA_TRUST"
  --atlas-matra-cadence "$MATRA_CADENCE"
  --atlas-matra-orth-cadence "$MATRA_ORTH_CADENCE"
  --atlas-matra-max-aspect "$MATRA_MAX_ASPECT"
  --atlas-matra-min-dim "$MATRA_MIN_DIM"
  --atlas-matra-damping "$MATRA_DAMPING"
)

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  run_capture 00_nvidia_smi nvidia-smi
  run_capture 00a_clean_main cmake --build "$BUILD_DIR" --target clean
  run_capture 01_build_main cmake --build "$BUILD_DIR" -j4
  run_capture 01a_clean_tests cmake --build "$TEST_BUILD_DIR" --target clean
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
  --atlas-argos-warmup-steps "${ARGOS_VARIANT_WARMUPS[0]}" \
  --atlas-argos-warmup-start-scale "${ARGOS_VARIANT_START_SCALES[0]}" \
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

for epochs in "${EPOCHS[@]}"; do
  run_variant_epoch "sweep_${BENCHMARK}_adamw_e${epochs}" \
    "adamw" "$epochs" \
    --variant adamw

  run_variant_epoch "sweep_${BENCHMARK}_echo_e${epochs}" \
    "echo" "$epochs" \
    "${ECHO_ARGS[@]}"

  run_variant_epoch "sweep_${BENCHMARK}_matra_e${epochs}" \
    "matra" "$epochs" \
    "${MATRA_ARGS[@]}"

  for ((argos_idx = 0; argos_idx < ${#ARGOS_VARIANT_LABELS[@]}; ++argos_idx)); do
    label="${ARGOS_VARIANT_LABELS[argos_idx]}"
    warmup="${ARGOS_VARIANT_WARMUPS[argos_idx]}"
    start_scale="${ARGOS_VARIANT_START_SCALES[argos_idx]}"
    run_variant_epoch "sweep_${BENCHMARK}_${label}_e${epochs}" \
      "$label" "$epochs" \
      --variant argos \
      --atlas-argos-geometry-scale "$ARGOS_GEOM" \
      --atlas-argos-orthogonal-scale "$ARGOS_ORTH" \
      --atlas-argos-predictive-scale "$ARGOS_PRED" \
      --atlas-argos-trust-radius "$ARGOS_TRUST" \
      --atlas-argos-warmup-steps "$warmup" \
      --atlas-argos-warmup-start-scale "$start_scale" \
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
  run_variant_accept "accept_${BENCHMARK}_adamw" \
    "adamw" \
    --variant adamw

  run_variant_accept "accept_${BENCHMARK}_echo" \
    "echo" \
    "${ECHO_ARGS[@]}"

  run_variant_accept "accept_${BENCHMARK}_matra" \
    "matra" \
    "${MATRA_ARGS[@]}"

  for ((argos_idx = 0; argos_idx < ${#ARGOS_VARIANT_LABELS[@]}; ++argos_idx)); do
    label="${ARGOS_VARIANT_LABELS[argos_idx]}"
    warmup="${ARGOS_VARIANT_WARMUPS[argos_idx]}"
    start_scale="${ARGOS_VARIANT_START_SCALES[argos_idx]}"
    run_variant_accept "accept_${BENCHMARK}_${label}" \
      "$label" \
      --variant argos \
      --atlas-argos-geometry-scale "$ARGOS_GEOM" \
      --atlas-argos-orthogonal-scale "$ARGOS_ORTH" \
      --atlas-argos-predictive-scale "$ARGOS_PRED" \
      --atlas-argos-trust-radius "$ARGOS_TRUST" \
      --atlas-argos-warmup-steps "$warmup" \
      --atlas-argos-warmup-start-scale "$start_scale" \
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

if [[ "$RUN_ACCEPTANCE" -eq 1 ]]; then
  generate_acceptance_rank
fi
generate_epoch4_rank

cat > "$OUT_DIR/README.txt" <<EOF
Output directory: $OUT_DIR

Files:
  raw/*.log                      Raw benchmark and unit-test output
  epoch_sweep_summary.tsv        ${REPEATS}-repeat epoch sweep summary
  acceptance_summary.tsv         10-repeat acceptance summary
  acceptance_rank_by_nll.tsv     Overall acceptance ranking
  epoch4_rank_by_nll.tsv         Overall epoch-4 ranking
  argos_acceptance_rank_by_nll.tsv
                                 ARGOS warmup ranking by acceptance NLL
  argos_epoch4_rank_by_nll.tsv   ARGOS warmup ranking by epoch-4 NLL
  run.log                        Full command log

Files to send back for review:
  - epoch_sweep_summary.tsv
  - acceptance_summary.tsv
  - acceptance_rank_by_nll.tsv
  - epoch4_rank_by_nll.tsv
  - argos_acceptance_rank_by_nll.tsv
  - argos_epoch4_rank_by_nll.tsv
  - raw/05_argos_parity.log

Benchmark:
  $BENCHMARK

ARGOS warmups:
  ${ARGOS_WARMUPS_CSV}

ARGOS warmup start scales:
  ${ARGOS_START_SCALES_CSV}

Decision rule:
  Continue tuning ARGOS only if at least one warmup setting closes the
  acceptance gap materially while preserving or improving the late epoch-4
  advantage on token-lm-corpus-large versus AdamW, ECHO, and MATRA.
EOF

echo "Saved results to: $OUT_DIR"
echo "Epoch sweep summary: $SUMMARY_TSV"
if [[ "$RUN_ACCEPTANCE" -eq 1 ]]; then
  echo "Acceptance summary: $ACCEPT_TSV"
  echo "Acceptance ranking: $ACCEPT_RANK_TSV"
  echo "ARGOS acceptance ranking: $ARGOS_ACCEPT_RANK_TSV"
fi
echo "Epoch-4 ranking: $E4_RANK_TSV"
echo "ARGOS epoch-4 ranking: $ARGOS_E4_RANK_TSV"
