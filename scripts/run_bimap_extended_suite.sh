#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT_DIR/unit-tests/build/glades-unit-tests"
BUILD_DIR="$ROOT_DIR/build"
TEST_BUILD_DIR="$ROOT_DIR/unit-tests/build"
CTX_DELTA_PARSER="$ROOT_DIR/scripts/parse_ctx_delta.py"

GPU_DEVICE=0
RANK=8
REPEATS=5
RUN_ACCEPTANCE=0
SKIP_BUILD=0
BIMAP_SCOPE="late-head"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR_DEFAULT="$ROOT_DIR/artifacts/bimap_extended_suite_${TIMESTAMP}"
OUT_DIR="${OUT_DIR:-$OUT_DIR_DEFAULT}"
BENCHMARKS=(
  "token-lm-large"
  "token-lm-context"
  "token-lm-context-large"
  "token-lm-document"
  "token-lm-corpus"
  "token-lm-corpus-large"
)
EPOCHS=(1 2 3 4)

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Runs an extended BiMAP-vs-AdamW GPU benchmark suite and saves raw logs plus TSV summaries.

Default benchmarks:
  - token-lm-large
  - token-lm-context
  - token-lm-context-large
  - token-lm-document
  - token-lm-corpus
  - token-lm-corpus-large

Variants:
  - adamw
  - bimap_lite   (BiMAP low-rank off)
  - bimap_v2_0   (BiMAP low-rank on, predictive scale 0)
  - bimap_v2     (BiMAP low-rank on, default predictive scale)

Options:
  --gpu-device N         CUDA device id to request (default: 0)
  --rank N               BiMAP rank to use for v2 variants (default: 8)
  --bimap-scope S        BiMAP scope: all|head|late|late-head (default: late-head)
  --repeats N            Repeats per epoch point (default: 5)
  --benchmarks CSV       Comma-separated benchmark list overriding defaults
  --epochs CSV           Comma-separated epoch list overriding defaults
  --out-dir PATH         Output directory (default: $OUT_DIR_DEFAULT)
  --skip-build           Skip build + atlas-controller steps
  --acceptance           Also run the 10-repeat acceptance pass
  --help                 Show this message
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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-device)
      GPU_DEVICE="$2"
      shift 2
      ;;
    --rank)
      RANK="$2"
      shift 2
      ;;
    --bimap-scope)
      BIMAP_SCOPE="$2"
      shift 2
      ;;
    --repeats)
      REPEATS="$2"
      shift 2
      ;;
    --benchmarks)
      parse_csv_words "$2" BENCHMARKS
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
    /^(AdamW|ATLAS-BIMAP)[[:space:]]/ && NF >= 20 {
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
    /^(AdamW|ATLAS-BIMAP)[[:space:]]/ && NF >= 20 {
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

run_ctx_delta_parser() {
  if [[ ! -f "$CTX_DELTA_PARSER" ]]; then
    echo "[warn] missing context delta parser: $CTX_DELTA_PARSER" | tee -a "$RUN_LOG" >&2
    return 0
  fi

  echo "[post] ctx delta parse" | tee -a "$RUN_LOG" >&2
  log_cmd python3 "$CTX_DELTA_PARSER" "$OUT_DIR"
  if python3 "$CTX_DELTA_PARSER" "$OUT_DIR" 2>&1 | tee "$OUT_DIR/raw/ctx_delta_parse.log"; then
    return 0
  fi

  echo "[warn] context delta parser did not produce output for this run" | tee -a "$RUN_LOG" >&2
  return 0
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

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  run_capture 00_nvidia_smi nvidia-smi
  run_capture 01_build_main cmake --build "$BUILD_DIR" -j4
  run_capture 02_build_tests cmake --build "$TEST_BUILD_DIR" -j4 --target glades-unit-tests
  run_capture 03_atlas_controller "$BIN" atlas-controller
  run_capture 04_bimap_micro "$BIN" atlas-bimap-micro
fi

run_capture 05_smoke_bimap_gpu \
  "$BIN" atlas-alt-bench \
  --mode token-lm \
  --token-epochs 1 \
  --repeats 1 \
  --variant bimap \
  --atlas-bimap-scope "$BIMAP_SCOPE" \
  --atlas-bimap-low-rank 1 \
  --rank "$RANK" \
  --atlas-bimap-factor-cadence 8 \
  --gpu-enable 1 \
  --gpu-device "$GPU_DEVICE"

for benchmark in "${BENCHMARKS[@]}"; do
  bimap_lite_cadence="$(bimap_lite_cadence_for_benchmark "$benchmark")"
  for epochs in "${EPOCHS[@]}"; do
    run_variant_epoch "sweep_${benchmark}_adamw_e${epochs}" \
      "$benchmark" "$epochs" "adamw" \
      --variant adamw

    run_variant_epoch "sweep_${benchmark}_bimap_lite_e${epochs}" \
      "$benchmark" "$epochs" "bimap_lite" \
      --variant bimap \
      --atlas-bimap-scope "$BIMAP_SCOPE" \
      --atlas-bimap-low-rank 0 \
      --atlas-bimap-factor-cadence "$bimap_lite_cadence"

    run_variant_epoch "sweep_${benchmark}_bimap_v2_0_e${epochs}" \
      "$benchmark" "$epochs" "bimap_v2_0" \
      --variant bimap \
      --atlas-bimap-scope "$BIMAP_SCOPE" \
      --atlas-bimap-low-rank 1 \
      --rank "$RANK" \
      --atlas-bimap-predictive-scale 0 \
      --atlas-bimap-factor-cadence 8

    run_variant_epoch "sweep_${benchmark}_bimap_v2_e${epochs}" \
      "$benchmark" "$epochs" "bimap_v2" \
      --variant bimap \
      --atlas-bimap-scope "$BIMAP_SCOPE" \
      --atlas-bimap-low-rank 1 \
      --rank "$RANK" \
      --atlas-bimap-factor-cadence 8
  done
done

if [[ "$RUN_ACCEPTANCE" -eq 1 ]]; then
  for benchmark in "${BENCHMARKS[@]}"; do
    bimap_lite_cadence="$(bimap_lite_cadence_for_benchmark "$benchmark")"
    run_variant_accept "accept_${benchmark}_adamw" \
      "$benchmark" "adamw" \
      --variant adamw

    run_variant_accept "accept_${benchmark}_bimap_lite" \
      "$benchmark" "bimap_lite" \
      --variant bimap \
      --atlas-bimap-scope "$BIMAP_SCOPE" \
      --atlas-bimap-low-rank 0 \
      --atlas-bimap-factor-cadence "$bimap_lite_cadence"

    run_variant_accept "accept_${benchmark}_bimap_v2_0" \
      "$benchmark" "bimap_v2_0" \
      --variant bimap \
      --atlas-bimap-scope "$BIMAP_SCOPE" \
      --atlas-bimap-low-rank 1 \
      --rank "$RANK" \
      --atlas-bimap-predictive-scale 0 \
      --atlas-bimap-factor-cadence 8

    run_variant_accept "accept_${benchmark}_bimap_v2" \
      "$benchmark" "bimap_v2" \
      --variant bimap \
      --atlas-bimap-scope "$BIMAP_SCOPE" \
      --atlas-bimap-low-rank 1 \
      --rank "$RANK" \
      --atlas-bimap-factor-cadence 8
  done
fi

run_ctx_delta_parser

cat > "$OUT_DIR/README.txt" <<EOF
Output directory: $OUT_DIR

Files:
- run.log: command trace
- epoch_sweep_summary.tsv: one row per benchmark/optimizer/epoch point
- acceptance_summary.tsv: optional 10-repeat summary rows when --acceptance is used
- ctx_delta_bimap_lite_vs_adamw.tsv: automatic CTX bucket deltas when CTX rows are present
- ctx_delta_bimap_lite_vs_adamw.md: markdown view of the same CTX bucket deltas
- raw/*.log: full raw benchmark outputs

BiMAP scope:
  - $BIMAP_SCOPE
- BiMAP-lite cadence policy:
  - 4 for non-corpus-large families
  - 2 for corpus-large/xlarge families

Configured benchmarks:
$(printf '  - %s\n' "${BENCHMARKS[@]}")

Quick comparison examples:

  column -t -s \$'\\t' "$SUMMARY_TSV"

  rg '^token-lm-context' "$SUMMARY_TSV" | column -t -s \$'\\t'

  rg '^token-lm-document' "$SUMMARY_TSV" | column -t -s \$'\\t'
EOF

echo
echo "Saved results to: $OUT_DIR"
echo "Epoch sweep summary: $SUMMARY_TSV"
if [[ "$RUN_ACCEPTANCE" -eq 1 ]]; then
  echo "Acceptance summary: $ACCEPT_TSV"
fi
if [[ -f "$OUT_DIR/ctx_delta_bimap_lite_vs_adamw.tsv" ]]; then
  echo "CTX delta TSV: $OUT_DIR/ctx_delta_bimap_lite_vs_adamw.tsv"
  echo "CTX delta Markdown: $OUT_DIR/ctx_delta_bimap_lite_vs_adamw.md"
fi
