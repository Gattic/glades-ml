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
INCLUDE_BIMAP_V2=0

ECHO_GEOMETRY_SCALE=1.0
ECHO_FINAL_GEOMETRY_SCALE=1.0
ECHO_DECAY_STEPS=0
ECHO_CADENCE=1
ECHO_SCOPE="late-head"
ECHO_TRUST_SCALE=0.0
ECHO_PREDICTIVE_SCALE=0.0
ECHO_STRUCTURAL_SCALE=0.0
ECHO_STRUCTURAL_GROUPS=1

MUON_GEOM=1.0
MUON_PRED=0.05
MUON_MAX_ASPECT=1.50
MUON_MIN_DIM=8
MUON_DAMPING=0.01

MATRA_GEOM=1.0
MATRA_ORTH=0.5
MATRA_PRED=0.05
MATRA_TRUST=0.50
MATRA_CADENCE=1
MATRA_ORTH_CADENCE=1
MATRA_MAX_ASPECT=1.50
MATRA_MIN_DIM=8
MATRA_DAMPING=0.01

BIMAP_SCOPE="late-head"
BIMAP_RANK=8
BIMAP_LITE_CADENCE=4
BIMAP_V2_CADENCE=8
BIMAP_V2_PRED=0.15

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR_DEFAULT="$ROOT_DIR/artifacts/optimizer_clean_ranking_${TIMESTAMP}"
OUT_DIR="${OUT_DIR:-$OUT_DIR_DEFAULT}"
BENCHMARKS=(token-lm-document token-lm-corpus-large)
EPOCHS=(1 2 3 4)

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Runs a clean same-codebase GPU ranking gate for:
  - AdamW
  - ATLAS-ECHO late-head
  - ATLAS-MUON-lite
  - ATLAS-MATRA
  - ATLAS-BiMAP-lite

Optional:
  - ATLAS-BiMAP-v2 (--include-bimap-v2)

Benchmarks:
  - token-lm-document
  - token-lm-corpus-large

Outputs:
  - epoch_sweep_summary.tsv
  - acceptance_summary.tsv
  - acceptance_rank_by_nll.tsv
  - epoch4_rank_by_nll.tsv
  - raw/*.log

Options:
  --gpu-device N              CUDA device id to request (default: 0)
  --repeats N                 Repeats per epoch point (default: 5)
  --out-dir PATH              Output directory (default: $OUT_DIR_DEFAULT)
  --skip-build                Skip build + unit-test verification
  --skip-acceptance           Skip the 10-repeat acceptance pass
  --include-bimap-v2          Also rank BiMAP-v2 with current late-head settings
  --atlas-matra-orth-cadence N
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
    --skip-acceptance)
      RUN_ACCEPTANCE=0
      shift
      ;;
    --include-bimap-v2)
      INCLUDE_BIMAP_V2=1
      shift
      ;;
    --atlas-matra-orth-cadence)
      MATRA_ORTH_CADENCE="$2"
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
ACCEPT_RANK_TSV="$OUT_DIR/acceptance_rank_by_nll.tsv"
E4_RANK_TSV="$OUT_DIR/epoch4_rank_by_nll.tsv"
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
    /^(AdamW|ATLAS-ECHO|ATLAS-BIMAP|ATLAS-MUON|ATLAS-MATRA)[[:space:]]/ && NF >= 20 {
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
    /^(AdamW|ATLAS-ECHO|ATLAS-BIMAP|ATLAS-MUON|ATLAS-MATRA)[[:space:]]/ && NF >= 20 {
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
             benchmark, optimizer,
             $2, $4, $5, $7, $8, $10, $14, $16, $20, logfile;
      exit 0;
    }
  ' "$logfile" >> "$ACCEPT_TSV"
}

bimap_lite_cadence_for_benchmark() {
  case "$1" in
    token-lm-corpus-large|token-lm-corpus-xlarge)
      echo 2
      ;;
    *)
      echo 4
      ;;
  esac
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

generate_epoch4_rank() {
  {
    printf "benchmark\trank\toptimizer\tepochs\ttrain_s_mean\ttest_nll_mean\tstatus\tlogfile\n"
    awk -F '\t' 'NR == 1 || $3 == "4"' "$SUMMARY_TSV" \
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
  } > "$E4_RANK_TSV"
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
  run_capture 04_echo_core "$BIN" atlas-echo-core
  run_capture 05_echo_micro "$BIN" atlas-echo-micro
  run_capture 06_bimap_micro "$BIN" atlas-bimap-micro
  run_capture 07_muon_core "$BIN" atlas-muon-core
  run_capture 08_muon_micro "$BIN" atlas-muon-micro
  run_capture 09_matra_core "$BIN" atlas-matra-core
  run_capture 10_matra_parity "$BIN" atlas-matra-parity
fi

run_capture 11_smoke_echo_gpu \
  "$BIN" atlas-alt-bench \
  --mode token-lm \
  --token-epochs 1 \
  --repeats 1 \
  --variant echo \
  --atlas-echo-geometry-scale "$ECHO_GEOMETRY_SCALE" \
  --atlas-echo-final-geometry-scale "$ECHO_FINAL_GEOMETRY_SCALE" \
  --atlas-echo-decay-steps "$ECHO_DECAY_STEPS" \
  --atlas-echo-cadence "$ECHO_CADENCE" \
  --atlas-echo-scope "$ECHO_SCOPE" \
  --atlas-echo-trust-scale "$ECHO_TRUST_SCALE" \
  --atlas-echo-predictive-scale "$ECHO_PREDICTIVE_SCALE" \
  --atlas-echo-structural-scale "$ECHO_STRUCTURAL_SCALE" \
  --atlas-echo-structural-groups "$ECHO_STRUCTURAL_GROUPS" \
  --gpu-enable 1 \
  --gpu-device "$GPU_DEVICE"

run_capture 12_smoke_bimap_gpu \
  "$BIN" atlas-alt-bench \
  --mode token-lm \
  --token-epochs 1 \
  --repeats 1 \
  --variant bimap \
  --atlas-bimap-scope "$BIMAP_SCOPE" \
  --atlas-bimap-low-rank 0 \
  --atlas-bimap-factor-cadence "$BIMAP_LITE_CADENCE" \
  --gpu-enable 1 \
  --gpu-device "$GPU_DEVICE"

run_capture 13_smoke_muon_gpu \
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

run_capture 14_smoke_matra_gpu \
  "$BIN" atlas-alt-bench \
  --mode token-lm \
  --token-epochs 1 \
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

for benchmark in "${BENCHMARKS[@]}"; do
  bimap_lite_cadence="$(bimap_lite_cadence_for_benchmark "$benchmark")"
  for epochs in "${EPOCHS[@]}"; do
    run_variant_epoch "sweep_${benchmark}_adamw_e${epochs}" \
      "$benchmark" "$epochs" "adamw" \
      --variant adamw

    run_variant_epoch "sweep_${benchmark}_echo_e${epochs}" \
      "$benchmark" "$epochs" "echo" \
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

    run_variant_epoch "sweep_${benchmark}_bimap_lite_e${epochs}" \
      "$benchmark" "$epochs" "bimap_lite" \
      --variant bimap \
      --atlas-bimap-scope "$BIMAP_SCOPE" \
      --atlas-bimap-low-rank 0 \
      --atlas-bimap-factor-cadence "$bimap_lite_cadence"

    run_variant_epoch "sweep_${benchmark}_muon_lite_e${epochs}" \
      "$benchmark" "$epochs" "muon_lite" \
      --variant muon \
      --atlas-muon-geometry-scale "$MUON_GEOM" \
      --atlas-muon-predictive-scale "$MUON_PRED" \
      --atlas-muon-max-aspect "$MUON_MAX_ASPECT" \
      --atlas-muon-min-dim "$MUON_MIN_DIM" \
      --atlas-muon-damping "$MUON_DAMPING"

    run_variant_epoch "sweep_${benchmark}_matra_e${epochs}" \
      "$benchmark" "$epochs" "matra" \
      --variant matra \
      --atlas-matra-geometry-scale "$MATRA_GEOM" \
      --atlas-matra-orthogonal-scale "$MATRA_ORTH" \
      --atlas-matra-predictive-scale "$MATRA_PRED" \
      --atlas-matra-trust-radius "$MATRA_TRUST" \
      --atlas-matra-cadence "$MATRA_CADENCE" \
      --atlas-matra-orth-cadence "$MATRA_ORTH_CADENCE" \
      --atlas-matra-max-aspect "$MATRA_MAX_ASPECT" \
      --atlas-matra-min-dim "$MATRA_MIN_DIM" \
      --atlas-matra-damping "$MATRA_DAMPING"

    if [[ "$INCLUDE_BIMAP_V2" -eq 1 ]]; then
      run_variant_epoch "sweep_${benchmark}_bimap_v2_e${epochs}" \
        "$benchmark" "$epochs" "bimap_v2" \
        --variant bimap \
        --atlas-bimap-scope "$BIMAP_SCOPE" \
        --atlas-bimap-low-rank 1 \
        --rank "$BIMAP_RANK" \
        --atlas-bimap-predictive-scale "$BIMAP_V2_PRED" \
        --atlas-bimap-factor-cadence "$BIMAP_V2_CADENCE"
    fi
  done
done

if [[ "$RUN_ACCEPTANCE" -eq 1 ]]; then
  for benchmark in "${BENCHMARKS[@]}"; do
    bimap_lite_cadence="$(bimap_lite_cadence_for_benchmark "$benchmark")"
    run_variant_accept "accept_${benchmark}_adamw" \
      "$benchmark" "adamw" \
      --variant adamw

    run_variant_accept "accept_${benchmark}_echo" \
      "$benchmark" "echo" \
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

    run_variant_accept "accept_${benchmark}_bimap_lite" \
      "$benchmark" "bimap_lite" \
      --variant bimap \
      --atlas-bimap-scope "$BIMAP_SCOPE" \
      --atlas-bimap-low-rank 0 \
      --atlas-bimap-factor-cadence "$bimap_lite_cadence"

    run_variant_accept "accept_${benchmark}_muon_lite" \
      "$benchmark" "muon_lite" \
      --variant muon \
      --atlas-muon-geometry-scale "$MUON_GEOM" \
      --atlas-muon-predictive-scale "$MUON_PRED" \
      --atlas-muon-max-aspect "$MUON_MAX_ASPECT" \
      --atlas-muon-min-dim "$MUON_MIN_DIM" \
      --atlas-muon-damping "$MUON_DAMPING"

    run_variant_accept "accept_${benchmark}_matra" \
      "$benchmark" "matra" \
      --variant matra \
      --atlas-matra-geometry-scale "$MATRA_GEOM" \
      --atlas-matra-orthogonal-scale "$MATRA_ORTH" \
      --atlas-matra-predictive-scale "$MATRA_PRED" \
      --atlas-matra-trust-radius "$MATRA_TRUST" \
      --atlas-matra-cadence "$MATRA_CADENCE" \
      --atlas-matra-orth-cadence "$MATRA_ORTH_CADENCE" \
      --atlas-matra-max-aspect "$MATRA_MAX_ASPECT" \
      --atlas-matra-min-dim "$MATRA_MIN_DIM" \
      --atlas-matra-damping "$MATRA_DAMPING"

    if [[ "$INCLUDE_BIMAP_V2" -eq 1 ]]; then
      run_variant_accept "accept_${benchmark}_bimap_v2" \
        "$benchmark" "bimap_v2" \
        --variant bimap \
        --atlas-bimap-scope "$BIMAP_SCOPE" \
        --atlas-bimap-low-rank 1 \
        --rank "$BIMAP_RANK" \
        --atlas-bimap-predictive-scale "$BIMAP_V2_PRED" \
        --atlas-bimap-factor-cadence "$BIMAP_V2_CADENCE"
    fi
  done
  generate_acceptance_rank
fi

generate_epoch4_rank

cat > "$OUT_DIR/README.txt" <<EOF
Output directory: $OUT_DIR

Purpose:
- apples-to-apples ranking gate on the current codebase
- compares AdamW, live ECHO late-head, current exact-path MUON-lite, MATRA, and BiMAP-lite
- optional BiMAP-v2 inclusion via --include-bimap-v2

Files:
- run.log: command trace
- epoch_sweep_summary.tsv: one row per benchmark/optimizer/epoch point
- acceptance_summary.tsv: 10-repeat acceptance summary (unless --skip-acceptance)
- acceptance_rank_by_nll.tsv: acceptance rows ranked by benchmark, then TestNLL, then time
- epoch4_rank_by_nll.tsv: e4 rows ranked by benchmark, then TestNLL, then time
- raw/*.log: full raw benchmark outputs

Benchmark settings:
- benchmarks: ${BENCHMARKS[*]}
- epochs: ${EPOCHS[*]}
- repeats per sweep point: $REPEATS
- acceptance repeats: $(if [[ "$RUN_ACCEPTANCE" -eq 1 ]]; then echo 10; else echo skipped; fi)

Optimizer settings:
- ECHO scope: $ECHO_SCOPE
- ECHO geometry/final/cadence/trust/pred/struct/groups:
  $ECHO_GEOMETRY_SCALE / $ECHO_FINAL_GEOMETRY_SCALE / $ECHO_CADENCE / $ECHO_TRUST_SCALE / $ECHO_PREDICTIVE_SCALE / $ECHO_STRUCTURAL_SCALE / $ECHO_STRUCTURAL_GROUPS
- MUON geom/pred/maxAspect/minDim/damping:
  $MUON_GEOM / $MUON_PRED / $MUON_MAX_ASPECT / $MUON_MIN_DIM / $MUON_DAMPING
- BiMAP scope/rank/liteCadence/v2Cadence/v2Pred:
  $BIMAP_SCOPE / $BIMAP_RANK / family-specific(4 non-corpus-large, 2 corpus-large/xlarge) / $BIMAP_V2_CADENCE / $BIMAP_V2_PRED
EOF

echo "Saved results to: $OUT_DIR"
echo "Epoch sweep summary: $SUMMARY_TSV"
if [[ "$RUN_ACCEPTANCE" -eq 1 ]]; then
  echo "Acceptance summary: $ACCEPT_TSV"
  echo "Acceptance ranking: $ACCEPT_RANK_TSV"
fi
echo "Epoch-4 ranking: $E4_RANK_TSV"
