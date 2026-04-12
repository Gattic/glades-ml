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
EPOCHS_CSV="1,2,3,4"
BENCHMARKS_CSV="token-lm-document,token-lm-corpus-large"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR_DEFAULT="$ROOT_DIR/artifacts/echo_scaleup_gate_${TIMESTAMP}"
OUT_DIR="${OUT_DIR:-$OUT_DIR_DEFAULT}"

# Moderate scale-up defaults chosen to increase matrix work enough to test
# ECHO overhead amortization without making the gate impractically heavy.
DOCUMENT_DMODEL=96
DOCUMENT_DFF=384
DOCUMENT_LAYERS=6
DOCUMENT_HEADS=8
DOCUMENT_SEQ_LEN=128
DOCUMENT_TRAIN_SEQS=64
DOCUMENT_TEST_SEQS=16

CORPUS_DMODEL=112
CORPUS_DFF=448
CORPUS_LAYERS=6
CORPUS_HEADS=8
CORPUS_SEQ_LEN=128
CORPUS_TRAIN_SEQS=80
CORPUS_TEST_SEQS=20

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Runs the frozen ECHO scale-up study on larger transformer shapes.

Candidates:
  - AdamW
  - ECHO 1.0 late-head
  - ECHO 1.0 large-only

Defaults:
  benchmarks: token-lm-document,token-lm-corpus-large
  epochs:     1,2,3,4
  document:   dModel=${DOCUMENT_DMODEL} dFF=${DOCUMENT_DFF} layers=${DOCUMENT_LAYERS} heads=${DOCUMENT_HEADS} seqLen=${DOCUMENT_SEQ_LEN} trainSeqs=${DOCUMENT_TRAIN_SEQS} testSeqs=${DOCUMENT_TEST_SEQS}
  corpus:     dModel=${CORPUS_DMODEL} dFF=${CORPUS_DFF} layers=${CORPUS_LAYERS} heads=${CORPUS_HEADS} seqLen=${CORPUS_SEQ_LEN} trainSeqs=${CORPUS_TRAIN_SEQS} testSeqs=${CORPUS_TEST_SEQS}

Options:
  --gpu-device N              CUDA device id to request (default: 0)
  --repeats N                 Repeats per epoch point (default: 5)
  --benchmarks CSV            Comma-separated benchmark list
  --epochs CSV                Comma-separated epoch list
  --out-dir PATH              Output directory
  --skip-build                Skip build + ECHO verification steps
  --acceptance                Also run the 10-repeat acceptance pass

  --document-dmodel N         token-lm-document dModel override
  --document-dff N            token-lm-document dFF override
  --document-layers N         token-lm-document layer count override
  --document-heads N          token-lm-document head count override
  --document-seq-len N        token-lm-document sequence length override
  --document-train-seqs N     token-lm-document train sequence override
  --document-test-seqs N      token-lm-document test sequence override

  --corpus-dmodel N           token-lm-corpus-large dModel override
  --corpus-dff N              token-lm-corpus-large dFF override
  --corpus-layers N           token-lm-corpus-large layer count override
  --corpus-heads N            token-lm-corpus-large head count override
  --corpus-seq-len N          token-lm-corpus-large sequence length override
  --corpus-train-seqs N       token-lm-corpus-large train sequence override
  --corpus-test-seqs N        token-lm-corpus-large test sequence override

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
    --benchmarks)
      BENCHMARKS_CSV="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS_CSV="$2"
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
    --document-dmodel)
      DOCUMENT_DMODEL="$2"
      shift 2
      ;;
    --document-dff)
      DOCUMENT_DFF="$2"
      shift 2
      ;;
    --document-layers)
      DOCUMENT_LAYERS="$2"
      shift 2
      ;;
    --document-heads)
      DOCUMENT_HEADS="$2"
      shift 2
      ;;
    --document-seq-len)
      DOCUMENT_SEQ_LEN="$2"
      shift 2
      ;;
    --document-train-seqs)
      DOCUMENT_TRAIN_SEQS="$2"
      shift 2
      ;;
    --document-test-seqs)
      DOCUMENT_TEST_SEQS="$2"
      shift 2
      ;;
    --corpus-dmodel)
      CORPUS_DMODEL="$2"
      shift 2
      ;;
    --corpus-dff)
      CORPUS_DFF="$2"
      shift 2
      ;;
    --corpus-layers)
      CORPUS_LAYERS="$2"
      shift 2
      ;;
    --corpus-heads)
      CORPUS_HEADS="$2"
      shift 2
      ;;
    --corpus-seq-len)
      CORPUS_SEQ_LEN="$2"
      shift 2
      ;;
    --corpus-train-seqs)
      CORPUS_TRAIN_SEQS="$2"
      shift 2
      ;;
    --corpus-test-seqs)
      CORPUS_TEST_SEQS="$2"
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

IFS=',' read -r -a BENCHMARKS <<< "$BENCHMARKS_CSV"
IFS=',' read -r -a EPOCHS <<< "$EPOCHS_CSV"

mkdir -p "$OUT_DIR/raw"

SUMMARY_TSV="$OUT_DIR/epoch_sweep_summary.tsv"
ACCEPT_TSV="$OUT_DIR/acceptance_summary.tsv"
RUN_LOG="$OUT_DIR/run.log"

cat > "$SUMMARY_TSV" <<'EOF'
benchmark	optimizer	echo_scope	epochs	dmodel	dff	layers	heads	seq_len	train_seqs	test_seqs	train_s_mean	train_s_pm	tok_s_mean	tok_s_pm	train_nll_mean	train_nll_pm	test_nll_mean	test_nll_pm	status	logfile
EOF

cat > "$ACCEPT_TSV" <<'EOF'
benchmark	optimizer	echo_scope	dmodel	dff	layers	heads	seq_len	train_seqs	test_seqs	train_s_mean	train_s_pm	tok_s_mean	tok_s_pm	train_nll_mean	train_nll_pm	test_nll_mean	test_nll_pm	status	logfile
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

set_benchmark_shape() {
  local benchmark="$1"
  case "$benchmark" in
    token-lm-document)
      SHAPE_DMODEL="$DOCUMENT_DMODEL"
      SHAPE_DFF="$DOCUMENT_DFF"
      SHAPE_LAYERS="$DOCUMENT_LAYERS"
      SHAPE_HEADS="$DOCUMENT_HEADS"
      SHAPE_SEQ_LEN="$DOCUMENT_SEQ_LEN"
      SHAPE_TRAIN_SEQS="$DOCUMENT_TRAIN_SEQS"
      SHAPE_TEST_SEQS="$DOCUMENT_TEST_SEQS"
      ;;
    token-lm-corpus-large)
      SHAPE_DMODEL="$CORPUS_DMODEL"
      SHAPE_DFF="$CORPUS_DFF"
      SHAPE_LAYERS="$CORPUS_LAYERS"
      SHAPE_HEADS="$CORPUS_HEADS"
      SHAPE_SEQ_LEN="$CORPUS_SEQ_LEN"
      SHAPE_TRAIN_SEQS="$CORPUS_TRAIN_SEQS"
      SHAPE_TEST_SEQS="$CORPUS_TEST_SEQS"
      ;;
    *)
      echo "unsupported benchmark for scale-up gate: $benchmark" >&2
      exit 2
      ;;
  esac

  SHAPE_ARGS=(
    --token-dmodel "$SHAPE_DMODEL"
    --token-dff "$SHAPE_DFF"
    --token-layers "$SHAPE_LAYERS"
    --token-heads "$SHAPE_HEADS"
    --token-seq-len "$SHAPE_SEQ_LEN"
    --token-train-seqs "$SHAPE_TRAIN_SEQS"
    --token-test-seqs "$SHAPE_TEST_SEQS"
  )
}

append_epoch_summary() {
  local benchmark="$1"
  local optimizer="$2"
  local scope="$3"
  local epochs="$4"
  local logfile="$5"
  awk -v benchmark="$benchmark" -v optimizer="$optimizer" -v scope="$scope" -v epochs="$epochs" \
      -v dmodel="$SHAPE_DMODEL" -v dff="$SHAPE_DFF" -v layers="$SHAPE_LAYERS" -v heads="$SHAPE_HEADS" \
      -v seq_len="$SHAPE_SEQ_LEN" -v train_seqs="$SHAPE_TRAIN_SEQS" -v test_seqs="$SHAPE_TEST_SEQS" \
      -v logfile="$logfile" '
    /^(AdamW|ATLAS-ECHO)[[:space:]]/ && NF >= 20 {
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
             benchmark, optimizer, scope, epochs,
             dmodel, dff, layers, heads, seq_len, train_seqs, test_seqs,
             $2, $4, $5, $7, $8, $10, $14, $16, $20, logfile;
      exit 0;
    }
  ' "$logfile" >> "$SUMMARY_TSV"
}

append_acceptance_summary() {
  local benchmark="$1"
  local optimizer="$2"
  local scope="$3"
  local logfile="$4"
  awk -v benchmark="$benchmark" -v optimizer="$optimizer" -v scope="$scope" \
      -v dmodel="$SHAPE_DMODEL" -v dff="$SHAPE_DFF" -v layers="$SHAPE_LAYERS" -v heads="$SHAPE_HEADS" \
      -v seq_len="$SHAPE_SEQ_LEN" -v train_seqs="$SHAPE_TRAIN_SEQS" -v test_seqs="$SHAPE_TEST_SEQS" \
      -v logfile="$logfile" '
    /^(AdamW|ATLAS-ECHO)[[:space:]]/ && NF >= 20 {
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
             benchmark, optimizer, scope,
             dmodel, dff, layers, heads, seq_len, train_seqs, test_seqs,
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
  local scope="$5"
  shift 5
  set_benchmark_shape "$benchmark"
  run_capture "$name" \
    "$BIN" atlas-alt-bench \
    --mode "$benchmark" \
    --token-epochs "$epochs" \
    --repeats "$REPEATS" \
    "${SHAPE_ARGS[@]}" \
    "$@" \
    --gpu-enable 1 \
    --gpu-device "$GPU_DEVICE"
  append_epoch_summary "$benchmark" "$optimizer" "$scope" "$epochs" "$OUT_DIR/raw/${name}.log"
}

run_variant_accept() {
  local name="$1"
  local benchmark="$2"
  local optimizer="$3"
  local scope="$4"
  shift 4
  set_benchmark_shape "$benchmark"
  run_capture "$name" \
    "$BIN" atlas-alt-bench \
    --mode "$benchmark" \
    --repeats 10 \
    "${SHAPE_ARGS[@]}" \
    "$@" \
    --gpu-enable 1 \
    --gpu-device "$GPU_DEVICE"
  append_acceptance_summary "$benchmark" "$optimizer" "$scope" "$OUT_DIR/raw/${name}.log"
}

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  run_capture 00_nvidia_smi nvidia-smi
  run_capture 01_build_main cmake --build "$BUILD_DIR" -j4
  run_capture 02_build_tests cmake --build "$TEST_BUILD_DIR" -j4 --target glades-unit-tests
  run_capture 03_echo_core "$BIN" atlas-echo-core
  run_capture 04_echo_micro "$BIN" atlas-echo-micro
fi

run_variant_epoch "05_smoke_document_echo_late_head_e1" \
  "token-lm-document" "1" "echo" "late-head" \
  --variant echo \
  --atlas-echo-geometry-scale 1.0 \
  --atlas-echo-scope late-head

for benchmark in "${BENCHMARKS[@]}"; do
  for epochs in "${EPOCHS[@]}"; do
    run_variant_epoch "sweep_${benchmark}_adamw_e${epochs}" \
      "$benchmark" "$epochs" "adamw" "-" \
      --variant adamw

    run_variant_epoch "sweep_${benchmark}_echo_late_head_e${epochs}" \
      "$benchmark" "$epochs" "echo" "late-head" \
      --variant echo \
      --atlas-echo-geometry-scale 1.0 \
      --atlas-echo-scope late-head

    run_variant_epoch "sweep_${benchmark}_echo_large_only_e${epochs}" \
      "$benchmark" "$epochs" "echo" "large-only" \
      --variant echo \
      --atlas-echo-geometry-scale 1.0 \
      --atlas-echo-scope large-only
  done
done

if [[ "$RUN_ACCEPTANCE" -eq 1 ]]; then
  for benchmark in "${BENCHMARKS[@]}"; do
    run_variant_accept "accept_${benchmark}_adamw" \
      "$benchmark" "adamw" "-" \
      --variant adamw

    run_variant_accept "accept_${benchmark}_echo_late_head" \
      "$benchmark" "echo" "late-head" \
      --variant echo \
      --atlas-echo-geometry-scale 1.0 \
      --atlas-echo-scope late-head

    run_variant_accept "accept_${benchmark}_echo_large_only" \
      "$benchmark" "echo" "large-only" \
      --variant echo \
      --atlas-echo-geometry-scale 1.0 \
      --atlas-echo-scope large-only
  done
fi

cat > "$OUT_DIR/README.txt" <<EOF
Output directory: $OUT_DIR

Purpose:
- freeze the small-model ECHO search
- compare only the live candidate (late-head) and the document quality control (large-only)
- check whether ECHO overhead amortizes on a meaningfully larger transformer shape

Configs:
- token-lm-document: dModel=$DOCUMENT_DMODEL dFF=$DOCUMENT_DFF layers=$DOCUMENT_LAYERS heads=$DOCUMENT_HEADS seqLen=$DOCUMENT_SEQ_LEN trainSeqs=$DOCUMENT_TRAIN_SEQS testSeqs=$DOCUMENT_TEST_SEQS
- token-lm-corpus-large: dModel=$CORPUS_DMODEL dFF=$CORPUS_DFF layers=$CORPUS_LAYERS heads=$CORPUS_HEADS seqLen=$CORPUS_SEQ_LEN trainSeqs=$CORPUS_TRAIN_SEQS testSeqs=$CORPUS_TEST_SEQS

Files:
- run.log: command trace
- epoch_sweep_summary.tsv: one row per benchmark/optimizer/scope/epoch point
- acceptance_summary.tsv: optional 10-repeat summary rows when --acceptance is used
- raw/*.log: full raw benchmark outputs
EOF

echo
echo "Saved results to: $OUT_DIR"
echo "Epoch sweep summary: $SUMMARY_TSV"
if [[ "$RUN_ACCEPTANCE" -eq 1 ]]; then
  echo "Acceptance summary: $ACCEPT_TSV"
fi
