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
BENCHMARK="token-lm-corpus-xlarge"
EPOCHS=(1 2 4 6 8)

ECHO_GEOMETRY_SCALE=1.0
ECHO_FINAL_GEOMETRY_SCALE=1.0
ECHO_DECAY_STEPS=0
ECHO_CADENCE=1
ECHO_SCOPE="late-head"
ECHO_TRUST_SCALE=0.0
ECHO_PREDICTIVE_SCALE=0.0
ECHO_STRUCTURAL_SCALE=0.0
ECHO_STRUCTURAL_GROUPS=1

BIMAP_SCOPE="late-head"
BIMAP_LITE_CADENCE=2

ARGOS_GEOM=1.0
ARGOS_ORTH=0.5
ARGOS_PRED=0.05
ARGOS_TRUST=0.20
ARGOS_WARMUP=32
ARGOS_WARMUP_START=0.25
ARGOS_ACT=1.0
ARGOS_SCOPE="head"
ARGOS_CADENCE=1
ARGOS_ORTH_CADENCE=1
ARGOS_MAX_ASPECT=1.50
ARGOS_MIN_DIM=8
ARGOS_DAMPING=0.01
ARGOS_OBS=0.75
ARGOS_HEAD=0.20
ARGOS_LATE=0.00

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR_DEFAULT="$ROOT_DIR/artifacts/corpus_xlarge_wide_train_${TIMESTAMP}"
OUT_DIR="${OUT_DIR:-$OUT_DIR_DEFAULT}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Runs a wider-train token-lm-corpus-xlarge GPU comparison for the only live
practical branches on this scale:
  - AdamW
  - ATLAS-BiMAP-lite
  - ATLAS-ECHO late-head
  - ATLAS-ARGOS

Defaults:
  - benchmark: token-lm-corpus-xlarge
  - epoch sweep: 1,2,4,6,8
  - acceptance: enabled

Outputs:
  - epoch_sweep_summary.tsv
  - acceptance_summary.tsv
  - acceptance_rank_by_nll.tsv
  - epoch4_rank_by_nll.tsv
  - final_epoch_rank_by_nll.tsv
  - raw/*.log

Options:
  --gpu-device N      CUDA device id to request (default: 0)
  --repeats N         Repeats per epoch point (default: 5)
  --epochs CSV        Comma-separated epoch list (default: 1,2,4,6,8)
  --out-dir PATH      Output directory (default: $OUT_DIR_DEFAULT)
  --skip-build        Skip build + verification (only safe after a clean rebuild)
  --skip-acceptance   Skip the 10-repeat acceptance pass
  --help              Show this message
EOF
}

parse_csv_words() {
  local input="$1"
  local -n out_ref="$2"
  local old_ifs="$IFS"
  IFS=','
  read -r -a out_ref <<< "$input"
  IFS="$old_ifs"
}

max_epoch() {
  local max="$1"
  shift
  local value
  for value in "$@"; do
    if (( value > max )); then
      max="$value"
    fi
  done
  echo "$max"
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
    --epochs)
      parse_csv_words "$2" EPOCHS
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
ACCEPT_RANK_TSV="$OUT_DIR/acceptance_rank_by_nll.tsv"
E4_RANK_TSV="$OUT_DIR/epoch4_rank_by_nll.tsv"
FINAL_RANK_TSV="$OUT_DIR/final_epoch_rank_by_nll.tsv"
RUN_LOG="$OUT_DIR/run.log"
FINAL_EPOCH="$(max_epoch "${EPOCHS[@]}")"

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
    /^(AdamW|ATLAS-ECHO|ATLAS-BIMAP|ATLAS-ARGOS)[[:space:]]/ && NF >= 20 {
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
    /^(AdamW|ATLAS-ECHO|ATLAS-BIMAP|ATLAS-ARGOS)[[:space:]]/ && NF >= 20 {
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
             benchmark, optimizer,
             $2, $4, $5, $7, $8, $10, $14, $16, $20, logfile;
      exit 0;
    }
  ' "$logfile" >> "$ACCEPT_TSV"
}

generate_acceptance_rank() {
  {
    printf "benchmark\trank\toptimizer\ttrain_s_mean\ttest_nll_mean\tstatus\tlogfile\n"
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
            printf "%s\t%d\t%s\t%s\t%s\t%s\t%s\n",
                   $1, rank, $2, $3, $9, $11, $12;
          }
        '
  } > "$ACCEPT_RANK_TSV"
}

generate_epoch_rank() {
  local epoch="$1"
  local out_tsv="$2"
  {
    printf "benchmark\trank\toptimizer\tepochs\ttrain_s_mean\ttest_nll_mean\tstatus\tlogfile\n"
    awk -F '\t' -v epoch="$epoch" 'NR == 1 || $3 == epoch' "$SUMMARY_TSV" \
      | tail -n +2 \
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
            printf "%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\n",
                   $1, rank, $2, $3, $4, $10, $12, $13;
          }
        '
  } > "$out_tsv"
}

run_variant_epoch() {
  local name="$1"
  local epochs="$2"
  local optimizer="$3"
  shift 3
  run_capture "$name" \
    "$BIN" atlas-alt-bench \
    --mode "$BENCHMARK" \
    --token-epochs "$epochs" \
    --repeats "$REPEATS" \
    "$@" \
    --gpu-enable 1 \
    --gpu-device "$GPU_DEVICE"
  append_epoch_summary "$BENCHMARK" "$optimizer" "$epochs" "$OUT_DIR/raw/${name}.log"
}

run_variant_accept() {
  local name="$1"
  local optimizer="$2"
  shift 2
  run_capture "$name" \
    "$BIN" atlas-alt-bench \
    --mode "$BENCHMARK" \
    --repeats 10 \
    "$@" \
    --gpu-enable 1 \
    --gpu-device "$GPU_DEVICE"
  append_acceptance_summary "$BENCHMARK" "$optimizer" "$OUT_DIR/raw/${name}.log"
}

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  run_capture 00_nvidia_smi nvidia-smi
  run_capture 00a_clean_main cmake --build "$BUILD_DIR" --target clean
  run_capture 01_build_main cmake --build "$BUILD_DIR" -j4
  run_capture 01a_clean_tests cmake --build "$TEST_BUILD_DIR" --target clean
  run_capture 02_build_tests cmake --build "$TEST_BUILD_DIR" -j4 --target glades-unit-tests
  run_capture 03_atlas_controller "$BIN" atlas-controller
  run_capture 04_echo_core "$BIN" atlas-echo-core
  run_capture 05_echo_micro "$BIN" atlas-echo-micro
  run_capture 06_bimap_micro "$BIN" atlas-bimap-micro
  run_capture 07_bimap_parity "$BIN" atlas-bimap-parity
  run_capture 08_argos_core "$BIN" atlas-argos-core
  run_capture 09_argos_parity "$BIN" atlas-argos-parity
fi

run_capture 10_smoke_corpus_xlarge_adamw \
  "$BIN" atlas-alt-bench \
  --mode "$BENCHMARK" \
  --token-epochs 1 \
  --repeats 1 \
  --variant adamw \
  --gpu-enable 1 \
  --gpu-device "$GPU_DEVICE"

for epochs in "${EPOCHS[@]}"; do
  run_variant_epoch "sweep_${BENCHMARK}_adamw_e${epochs}" \
    "$epochs" "adamw" \
    --variant adamw

  run_variant_epoch "sweep_${BENCHMARK}_echo_e${epochs}" \
    "$epochs" "echo" \
    --variant echo \
    --atlas-echo-geometry-scale "$ECHO_GEOMETRY_SCALE" \
    --atlas-echo-final-geometry-scale "$ECHO_FINAL_GEOMETRY_SCALE" \
    --atlas-echo-decay-steps "$ECHO_DECAY_STEPS" \
    --atlas-echo-cadence "$ECHO_CADENCE" \
    --atlas-echo-scope "$ECHO_SCOPE" \
    --atlas-echo-trust-scale "$ECHO_TRUST_SCALE" \
    --atlas-echo-predictive-scale "$ECHO_PREDICTIVE_SCALE" \
    --atlas-echo-structural-scale "$ECHO_STRUCTURAL_SCALE" \
    --atlas-echo-structural-groups "$ECHO_STRUCTURAL_GROUPS"

  run_variant_epoch "sweep_${BENCHMARK}_bimap_lite_e${epochs}" \
    "$epochs" "bimap_lite" \
    --variant bimap \
    --atlas-bimap-scope "$BIMAP_SCOPE" \
    --atlas-bimap-low-rank 0 \
    --atlas-bimap-factor-cadence "$BIMAP_LITE_CADENCE"

  run_variant_epoch "sweep_${BENCHMARK}_argos_e${epochs}" \
    "$epochs" "argos" \
    --variant argos \
    --atlas-argos-geometry-scale "$ARGOS_GEOM" \
    --atlas-argos-orthogonal-scale "$ARGOS_ORTH" \
    --atlas-argos-predictive-scale "$ARGOS_PRED" \
    --atlas-argos-trust-radius "$ARGOS_TRUST" \
    --atlas-argos-warmup-steps "$ARGOS_WARMUP" \
    --atlas-argos-warmup-start-scale "$ARGOS_WARMUP_START" \
    --atlas-argos-actuation-scale "$ARGOS_ACT" \
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

if [[ "$RUN_ACCEPTANCE" -eq 1 ]]; then
  run_variant_accept "accept_${BENCHMARK}_adamw" \
    "adamw" \
    --variant adamw

  run_variant_accept "accept_${BENCHMARK}_echo" \
    "echo" \
    --variant echo \
    --atlas-echo-geometry-scale "$ECHO_GEOMETRY_SCALE" \
    --atlas-echo-final-geometry-scale "$ECHO_FINAL_GEOMETRY_SCALE" \
    --atlas-echo-decay-steps "$ECHO_DECAY_STEPS" \
    --atlas-echo-cadence "$ECHO_CADENCE" \
    --atlas-echo-scope "$ECHO_SCOPE" \
    --atlas-echo-trust-scale "$ECHO_TRUST_SCALE" \
    --atlas-echo-predictive-scale "$ECHO_PREDICTIVE_SCALE" \
    --atlas-echo-structural-scale "$ECHO_STRUCTURAL_SCALE" \
    --atlas-echo-structural-groups "$ECHO_STRUCTURAL_GROUPS"

  run_variant_accept "accept_${BENCHMARK}_bimap_lite" \
    "bimap_lite" \
    --variant bimap \
    --atlas-bimap-scope "$BIMAP_SCOPE" \
    --atlas-bimap-low-rank 0 \
    --atlas-bimap-factor-cadence "$BIMAP_LITE_CADENCE"

  run_variant_accept "accept_${BENCHMARK}_argos" \
    "argos" \
    --variant argos \
    --atlas-argos-geometry-scale "$ARGOS_GEOM" \
    --atlas-argos-orthogonal-scale "$ARGOS_ORTH" \
    --atlas-argos-predictive-scale "$ARGOS_PRED" \
    --atlas-argos-trust-radius "$ARGOS_TRUST" \
    --atlas-argos-warmup-steps "$ARGOS_WARMUP" \
    --atlas-argos-warmup-start-scale "$ARGOS_WARMUP_START" \
    --atlas-argos-actuation-scale "$ARGOS_ACT" \
    --atlas-argos-scope "$ARGOS_SCOPE" \
    --atlas-argos-cadence "$ARGOS_CADENCE" \
    --atlas-argos-orth-cadence "$ARGOS_ORTH_CADENCE" \
    --atlas-argos-max-aspect "$ARGOS_MAX_ASPECT" \
    --atlas-argos-min-dim "$ARGOS_MIN_DIM" \
    --atlas-argos-damping "$ARGOS_DAMPING" \
    --atlas-argos-observability-scale "$ARGOS_OBS" \
    --atlas-argos-head-bonus "$ARGOS_HEAD" \
    --atlas-argos-late-bonus "$ARGOS_LATE"

  generate_acceptance_rank
fi

if printf '%s\n' "${EPOCHS[@]}" | rg -qx '4'; then
  generate_epoch_rank 4 "$E4_RANK_TSV"
fi
generate_epoch_rank "$FINAL_EPOCH" "$FINAL_RANK_TSV"

cat > "$OUT_DIR/README.txt" <<EOF
Output directory: $OUT_DIR

Purpose:
- wider-train token-lm-corpus-xlarge comparison for the only live practical
  branches on this scale
- compares AdamW, ECHO late-head, BiMAP-lite, and ARGOS
- extends the xlarge sweep beyond the short 1-4 epoch window

Files:
- run.log: command trace
- epoch_sweep_summary.tsv: one row per optimizer/epoch point
- acceptance_summary.tsv: 10-repeat acceptance summary (unless --skip-acceptance)
- acceptance_rank_by_nll.tsv: acceptance rows ranked by TestNLL, then time
- epoch4_rank_by_nll.tsv: e4 rows ranked by TestNLL, then time when epoch 4 is included
- final_epoch_rank_by_nll.tsv: highest configured epoch ranked by TestNLL, then time
- raw/*.log: full raw benchmark outputs

Benchmark settings:
- benchmark: $BENCHMARK
- epochs: ${EPOCHS[*]}
- final epoch rank target: $FINAL_EPOCH
- repeats per sweep point: $REPEATS
- acceptance repeats: $(if [[ "$RUN_ACCEPTANCE" -eq 1 ]]; then echo 10; else echo skipped; fi)

Optimizer settings:
- ECHO scope: $ECHO_SCOPE
- BiMAP-lite scope/cadence: $BIMAP_SCOPE / $BIMAP_LITE_CADENCE
- ARGOS trust/warmup/start/obs/head/scope:
  $ARGOS_TRUST / $ARGOS_WARMUP / $ARGOS_WARMUP_START / $ARGOS_OBS / $ARGOS_HEAD / $ARGOS_SCOPE
EOF

echo "Saved results to: $OUT_DIR"
echo "Epoch sweep summary: $SUMMARY_TSV"
if [[ "$RUN_ACCEPTANCE" -eq 1 ]]; then
  echo "Acceptance summary: $ACCEPT_TSV"
  echo "Acceptance ranking: $ACCEPT_RANK_TSV"
fi
if [[ -f "$E4_RANK_TSV" ]]; then
  echo "Epoch-4 ranking: $E4_RANK_TSV"
fi
echo "Final epoch ranking: $FINAL_RANK_TSV"
