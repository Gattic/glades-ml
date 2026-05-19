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
EPOCHS=(1 2 3 4)

ECHO_SCOPE="late-head"
ECHO_GEOM=1.0
ECHO_FINAL_GEOM=1.0
ECHO_CADENCE=1
ECHO_TRUST=0.0
ECHO_PRED=0.0
ECHO_STRUCT=0.0
ECHO_GROUPS=1

BIMAP_SCOPE="late-head"
BIMAP_GEOM=1.0
BIMAP_RANK=8
BIMAP_LITE_PRED=0.15
BIMAP_V2_PRED=0.15
BIMAP_V2_CADENCE=8
LITE_CADENCE_LIST=(1 2 4)

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR_DEFAULT="$ROOT_DIR/artifacts/bimap_gpu_focus_${TIMESTAMP}"
OUT_DIR="${OUT_DIR:-$OUT_DIR_DEFAULT}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Runs a focused GPU sweep for the BiMAP family against practical baselines.

Default variants:
  - adamw
  - echo_late_head
  - bimap_lite_c1
  - bimap_lite_pred0
  - bimap_lite_c2
  - bimap_lite_c4
  - bimap_v2_c8

Outputs:
  - epoch_sweep_summary.tsv
  - acceptance_summary.tsv
  - acceptance_rank_by_nll.tsv
  - epoch4_rank_by_nll.tsv
  - bimap_acceptance_rank_by_nll.tsv
  - bimap_epoch4_rank_by_nll.tsv
  - raw/*.log

Options:
  --benchmark MODE            token-lm-corpus-large or token-lm-corpus-xlarge
  --gpu-device N             CUDA device id to request (default: 0)
  --repeats N                Repeats per epoch point (default: 5)
  --out-dir PATH             Output directory (default: $OUT_DIR_DEFAULT)
  --skip-build               Skip clean build + verification
  --skip-acceptance          Skip 10-repeat acceptance pass
  --atlas-bimap-scope S      all|head|late|late-head (default: $BIMAP_SCOPE)
  --atlas-bimap-geometry-scale X
  --atlas-bimap-rank N
  --atlas-bimap-lite-predictive-scale X
  --atlas-bimap-v2-predictive-scale X
  --atlas-bimap-v2-cadence N
  --help                     Show this message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark)
      BENCHMARK="$2"
      shift 2
      ;;
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
    --atlas-bimap-scope)
      BIMAP_SCOPE="$2"
      shift 2
      ;;
    --atlas-bimap-geometry-scale)
      BIMAP_GEOM="$2"
      shift 2
      ;;
    --atlas-bimap-rank)
      BIMAP_RANK="$2"
      shift 2
      ;;
    --atlas-bimap-lite-predictive-scale)
      BIMAP_LITE_PRED="$2"
      shift 2
      ;;
    --atlas-bimap-v2-predictive-scale)
      BIMAP_V2_PRED="$2"
      shift 2
      ;;
    --atlas-bimap-v2-cadence)
      BIMAP_V2_CADENCE="$2"
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

case "$BENCHMARK" in
  token-lm-corpus-large|token-lm-corpus-xlarge)
    ;;
  *)
    echo "unsupported benchmark: $BENCHMARK" >&2
    exit 2
    ;;
esac

mkdir -p "$OUT_DIR/raw"

SUMMARY_TSV="$OUT_DIR/epoch_sweep_summary.tsv"
ACCEPT_TSV="$OUT_DIR/acceptance_summary.tsv"
ACCEPT_RANK_TSV="$OUT_DIR/acceptance_rank_by_nll.tsv"
E4_RANK_TSV="$OUT_DIR/epoch4_rank_by_nll.tsv"
BIMAP_ACCEPT_RANK_TSV="$OUT_DIR/bimap_acceptance_rank_by_nll.tsv"
BIMAP_E4_RANK_TSV="$OUT_DIR/bimap_epoch4_rank_by_nll.tsv"
RUN_LOG="$OUT_DIR/run.log"

if [[ "$SKIP_BUILD" -eq 1 ]]; then
  echo "[warn] --skip-build reuses existing binaries; unsafe after header/layout changes. Use one clean run first." | tee -a "$RUN_LOG" >&2
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
    /^(AdamW|ATLAS-ECHO|ATLAS-BIMAP)[[:space:]]/ && NF >= 20 {
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
    /^(AdamW|ATLAS-ECHO|ATLAS-BIMAP)[[:space:]]/ && NF >= 20 {
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
             benchmark, optimizer,
             $2, $4, $5, $7, $8, $10, $14, $16, $20, logfile;
      exit 0;
    }
  ' "$logfile" >> "$ACCEPT_TSV"
}

generate_acceptance_rank() {
  {
    printf "benchmark\trank\toptimizer\ttrain_s_mean\ttok_s_mean\ttest_nll_mean\tstatus\tlogfile\n"
    tail -n +2 "$ACCEPT_TSV" \
      | sort -t $'\t' -k1,1 -k9,9g -k5,5gr \
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
                   $1, rank, $2, $3, $5, $9, $11, $12;
          }
        '
  } > "$ACCEPT_RANK_TSV"
}

generate_epoch4_rank() {
  {
    printf "benchmark\trank\toptimizer\tepochs\ttrain_s_mean\ttok_s_mean\ttest_nll_mean\tstatus\tlogfile\n"
    awk -F '\t' 'NR == 1 || $3 == "4"' "$SUMMARY_TSV" \
      | tail -n +2 \
      | sort -t $'\t' -k1,1 -k10,10g -k6,6gr \
      | awk -F '\t' '
          BEGIN { OFS = "\t"; prev = ""; rank = 0; }
          {
            if ($1 != prev) {
              prev = $1;
              rank = 1;
            } else {
              rank += 1;
            }
            printf "%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
                   $1, rank, $2, $3, $4, $6, $10, $12, $13;
          }
        '
  } > "$E4_RANK_TSV"
}

generate_bimap_filtered_rank() {
  local src="$1"
  local out="$2"
  local is_epoch="$3"
  if [[ "$is_epoch" -eq 1 ]]; then
    {
      printf "benchmark\trank\toptimizer\tepochs\ttrain_s_mean\ttok_s_mean\ttest_nll_mean\tstatus\tlogfile\n"
      awk -F '\t' 'NR == 1 || $3 ~ /^bimap_/' "$src" \
        | tail -n +2 \
        | sort -t $'\t' -k1,1 -k7,7g -k6,6gr \
        | awk -F '\t' '
            BEGIN { OFS = "\t"; prev = ""; rank = 0; }
            {
              if ($1 != prev) {
                prev = $1;
                rank = 1;
              } else {
                rank += 1;
              }
              print $1, rank, $3, $4, $5, $6, $7, $8, $9;
            }
          '
    } > "$out"
  else
    {
      printf "benchmark\trank\toptimizer\ttrain_s_mean\ttok_s_mean\ttest_nll_mean\tstatus\tlogfile\n"
      awk -F '\t' 'NR == 1 || $3 ~ /^bimap_/' "$src" \
        | tail -n +2 \
        | sort -t $'\t' -k1,1 -k6,6g -k5,5gr \
        | awk -F '\t' '
            BEGIN { OFS = "\t"; prev = ""; rank = 0; }
            {
              if ($1 != prev) {
                prev = $1;
                rank = 1;
              } else {
                rank += 1;
              }
              print $1, rank, $3, $4, $5, $6, $7, $8;
            }
          '
    } > "$out"
  fi
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
  run_capture 03_bimap_micro "$BIN" atlas-bimap-micro
  run_capture 04_bimap_parity "$BIN" atlas-bimap-parity
fi

run_capture 05_smoke_${BENCHMARK}_adamw \
  "$BIN" atlas-alt-bench \
  --mode "$BENCHMARK" \
  --token-epochs 1 \
  --repeats 1 \
  --variant adamw \
  --gpu-enable 1 \
  --gpu-device "$GPU_DEVICE"

run_capture 06_smoke_${BENCHMARK}_bimap_lite \
  "$BIN" atlas-alt-bench \
  --mode "$BENCHMARK" \
  --token-epochs 1 \
  --repeats 1 \
  --variant bimap \
  --atlas-bimap-scope "$BIMAP_SCOPE" \
  --atlas-bimap-low-rank 0 \
  --atlas-bimap-geometry-scale "$BIMAP_GEOM" \
  --atlas-bimap-predictive-scale "$BIMAP_LITE_PRED" \
  --atlas-bimap-factor-cadence 1 \
  --gpu-enable 1 \
  --gpu-device "$GPU_DEVICE"

for epochs in "${EPOCHS[@]}"; do
  run_variant_epoch "sweep_${BENCHMARK}_adamw_e${epochs}" \
    "$epochs" "adamw" \
    --variant adamw

  run_variant_epoch "sweep_${BENCHMARK}_echo_e${epochs}" \
    "$epochs" "echo_late_head" \
    --variant echo \
    --atlas-echo-scope "$ECHO_SCOPE" \
    --atlas-echo-geometry-scale "$ECHO_GEOM" \
    --atlas-echo-final-geometry-scale "$ECHO_FINAL_GEOM" \
    --atlas-echo-cadence "$ECHO_CADENCE" \
    --atlas-echo-trust-scale "$ECHO_TRUST" \
    --atlas-echo-predictive-scale "$ECHO_PRED" \
    --atlas-echo-structural-scale "$ECHO_STRUCT" \
    --atlas-echo-structural-groups "$ECHO_GROUPS"

  run_variant_epoch "sweep_${BENCHMARK}_bimap_lite_c1_e${epochs}" \
    "$epochs" "bimap_lite_c1" \
    --variant bimap \
    --atlas-bimap-scope "$BIMAP_SCOPE" \
    --atlas-bimap-low-rank 0 \
    --atlas-bimap-geometry-scale "$BIMAP_GEOM" \
    --atlas-bimap-predictive-scale "$BIMAP_LITE_PRED" \
    --atlas-bimap-factor-cadence 1

  run_variant_epoch "sweep_${BENCHMARK}_bimap_lite_pred0_e${epochs}" \
    "$epochs" "bimap_lite_pred0" \
    --variant bimap \
    --atlas-bimap-scope "$BIMAP_SCOPE" \
    --atlas-bimap-low-rank 0 \
    --atlas-bimap-geometry-scale "$BIMAP_GEOM" \
    --atlas-bimap-predictive-scale 0.0 \
    --atlas-bimap-factor-cadence 1

  for cadence in "${LITE_CADENCE_LIST[@]}"; do
    if [[ "$cadence" == "1" ]]; then
      continue
    fi
    run_variant_epoch "sweep_${BENCHMARK}_bimap_lite_c${cadence}_e${epochs}" \
      "$epochs" "bimap_lite_c${cadence}" \
      --variant bimap \
      --atlas-bimap-scope "$BIMAP_SCOPE" \
      --atlas-bimap-low-rank 0 \
      --atlas-bimap-geometry-scale "$BIMAP_GEOM" \
      --atlas-bimap-predictive-scale "$BIMAP_LITE_PRED" \
      --atlas-bimap-factor-cadence "$cadence"
  done

  run_variant_epoch "sweep_${BENCHMARK}_bimap_v2_c${BIMAP_V2_CADENCE}_e${epochs}" \
    "$epochs" "bimap_v2_c${BIMAP_V2_CADENCE}" \
    --variant bimap \
    --atlas-bimap-scope "$BIMAP_SCOPE" \
    --atlas-bimap-low-rank 1 \
    --atlas-bimap-geometry-scale "$BIMAP_GEOM" \
    --atlas-bimap-predictive-scale "$BIMAP_V2_PRED" \
    --atlas-bimap-factor-cadence "$BIMAP_V2_CADENCE" \
    --rank "$BIMAP_RANK"
done

if [[ "$RUN_ACCEPTANCE" -eq 1 ]]; then
  run_variant_accept "accept_${BENCHMARK}_adamw" \
    "adamw" \
    --variant adamw

  run_variant_accept "accept_${BENCHMARK}_echo" \
    "echo_late_head" \
    --variant echo \
    --atlas-echo-scope "$ECHO_SCOPE" \
    --atlas-echo-geometry-scale "$ECHO_GEOM" \
    --atlas-echo-final-geometry-scale "$ECHO_FINAL_GEOM" \
    --atlas-echo-cadence "$ECHO_CADENCE" \
    --atlas-echo-trust-scale "$ECHO_TRUST" \
    --atlas-echo-predictive-scale "$ECHO_PRED" \
    --atlas-echo-structural-scale "$ECHO_STRUCT" \
    --atlas-echo-structural-groups "$ECHO_GROUPS"

  run_variant_accept "accept_${BENCHMARK}_bimap_lite_c1" \
    "bimap_lite_c1" \
    --variant bimap \
    --atlas-bimap-scope "$BIMAP_SCOPE" \
    --atlas-bimap-low-rank 0 \
    --atlas-bimap-geometry-scale "$BIMAP_GEOM" \
    --atlas-bimap-predictive-scale "$BIMAP_LITE_PRED" \
    --atlas-bimap-factor-cadence 1

  run_variant_accept "accept_${BENCHMARK}_bimap_lite_pred0" \
    "bimap_lite_pred0" \
    --variant bimap \
    --atlas-bimap-scope "$BIMAP_SCOPE" \
    --atlas-bimap-low-rank 0 \
    --atlas-bimap-geometry-scale "$BIMAP_GEOM" \
    --atlas-bimap-predictive-scale 0.0 \
    --atlas-bimap-factor-cadence 1

  for cadence in "${LITE_CADENCE_LIST[@]}"; do
    if [[ "$cadence" == "1" ]]; then
      continue
    fi
    run_variant_accept "accept_${BENCHMARK}_bimap_lite_c${cadence}" \
      "bimap_lite_c${cadence}" \
      --variant bimap \
      --atlas-bimap-scope "$BIMAP_SCOPE" \
      --atlas-bimap-low-rank 0 \
      --atlas-bimap-geometry-scale "$BIMAP_GEOM" \
      --atlas-bimap-predictive-scale "$BIMAP_LITE_PRED" \
      --atlas-bimap-factor-cadence "$cadence"
  done

  run_variant_accept "accept_${BENCHMARK}_bimap_v2_c${BIMAP_V2_CADENCE}" \
    "bimap_v2_c${BIMAP_V2_CADENCE}" \
    --variant bimap \
    --atlas-bimap-scope "$BIMAP_SCOPE" \
    --atlas-bimap-low-rank 1 \
    --atlas-bimap-geometry-scale "$BIMAP_GEOM" \
    --atlas-bimap-predictive-scale "$BIMAP_V2_PRED" \
    --atlas-bimap-factor-cadence "$BIMAP_V2_CADENCE" \
    --rank "$BIMAP_RANK"

  generate_acceptance_rank
  generate_bimap_filtered_rank "$ACCEPT_RANK_TSV" "$BIMAP_ACCEPT_RANK_TSV" 0
fi

generate_epoch4_rank
generate_bimap_filtered_rank "$E4_RANK_TSV" "$BIMAP_E4_RANK_TSV" 1

cat > "$OUT_DIR/README.txt" <<EOF
Output directory: $OUT_DIR

Purpose:
- focused GPU ranking for the BiMAP family after the lite-path scale-vector optimization
- compares AdamW and ECHO against practical BiMAP-lite/BiMAP-v2 variants
- intended to answer both quality and throughput questions on $BENCHMARK

Files:
- run.log: command trace
- epoch_sweep_summary.tsv: one row per optimizer/epoch point
- acceptance_summary.tsv: 10-repeat acceptance summary (unless --skip-acceptance)
- acceptance_rank_by_nll.tsv: all variants ranked by acceptance TestNLL, then tok/s
- epoch4_rank_by_nll.tsv: all variants ranked by epoch-4 TestNLL, then tok/s
- bimap_acceptance_rank_by_nll.tsv: BiMAP-only acceptance ranking
- bimap_epoch4_rank_by_nll.tsv: BiMAP-only epoch-4 ranking
- raw/*.log: full raw benchmark output

Settings:
- benchmark: $BENCHMARK
- epochs: ${EPOCHS[*]}
- repeats per sweep point: $REPEATS
- acceptance repeats: $(if [[ "$RUN_ACCEPTANCE" -eq 1 ]]; then echo 10; else echo skipped; fi)
- BiMAP scope/geom/rank: $BIMAP_SCOPE / $BIMAP_GEOM / $BIMAP_RANK
- BiMAP lite predictive: $BIMAP_LITE_PRED
- BiMAP v2 predictive/cadence: $BIMAP_V2_PRED / $BIMAP_V2_CADENCE
- BiMAP lite cadence sweep: ${LITE_CADENCE_LIST[*]}
EOF

echo "Saved results to: $OUT_DIR"
echo "Epoch sweep summary: $SUMMARY_TSV"
if [[ "$RUN_ACCEPTANCE" -eq 1 ]]; then
  echo "Acceptance summary: $ACCEPT_TSV"
  echo "Acceptance ranking: $ACCEPT_RANK_TSV"
  echo "BiMAP acceptance ranking: $BIMAP_ACCEPT_RANK_TSV"
fi
echo "Epoch-4 ranking: $E4_RANK_TSV"
echo "BiMAP epoch-4 ranking: $BIMAP_E4_RANK_TSV"
