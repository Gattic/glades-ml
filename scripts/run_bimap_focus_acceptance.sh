#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT_DIR/unit-tests/build/glades-unit-tests"
BUILD_DIR="$ROOT_DIR/build"
TEST_BUILD_DIR="$ROOT_DIR/unit-tests/build"
CTX_DELTA_PARSER="$ROOT_DIR/scripts/parse_ctx_delta.py"

GPU_DEVICE=0
SKIP_BUILD=0
BIMAP_SCOPE="late-head"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR_DEFAULT="$ROOT_DIR/artifacts/bimap_focus_acceptance_${TIMESTAMP}"
OUT_DIR="${OUT_DIR:-$OUT_DIR_DEFAULT}"
BENCHMARKS=(
  "token-lm-large"
  "token-lm-context"
  "token-lm-context-large"
)

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Runs the focused BiMAP-lite acceptance pass against AdamW on the remaining
interesting transformer modes:
  - token-lm-large
  - token-lm-context
  - token-lm-context-large

Each benchmark is run with:
  - AdamW
  - BiMAP-lite (--variant bimap --atlas-bimap-low-rank 0)

Options:
  --gpu-device N      CUDA device id to request (default: 0)
  --bimap-scope S     BiMAP scope: all|head|late|late-head (default: late-head)
  --out-dir PATH      Output directory (default: $OUT_DIR_DEFAULT)
  --skip-build        Skip build + atlas-controller steps
  --help              Show this message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-device)
      GPU_DEVICE="$2"
      shift 2
      ;;
    --bimap-scope)
      BIMAP_SCOPE="$2"
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

ACCEPT_TSV="$OUT_DIR/acceptance_summary.tsv"
RUN_LOG="$OUT_DIR/run.log"

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

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  run_capture 00_nvidia_smi nvidia-smi
  run_capture 01_build_main cmake --build "$BUILD_DIR" -j4
  run_capture 02_build_tests cmake --build "$TEST_BUILD_DIR" -j4 --target glades-unit-tests
  run_capture 03_atlas_controller "$BIN" atlas-controller
  run_capture 04_bimap_micro "$BIN" atlas-bimap-micro
fi

for benchmark in "${BENCHMARKS[@]}"; do
  run_variant_accept "accept_${benchmark}_adamw" \
    "$benchmark" "adamw" \
    --variant adamw

  run_variant_accept "accept_${benchmark}_bimap_lite" \
    "$benchmark" "bimap_lite" \
    --variant bimap \
    --atlas-bimap-scope "$BIMAP_SCOPE" \
    --atlas-bimap-low-rank 0 \
    --atlas-bimap-factor-cadence 1
done

run_ctx_delta_parser

cat > "$OUT_DIR/README.txt" <<EOF
Output directory: $OUT_DIR

Files:
- run.log: command trace
- acceptance_summary.tsv: focused 10-repeat summary rows
- ctx_delta_bimap_lite_vs_adamw.tsv: automatic CTX bucket deltas when CTX rows are present
- ctx_delta_bimap_lite_vs_adamw.md: markdown view of the same CTX bucket deltas
- raw/*.log: full raw benchmark outputs

BiMAP scope:
  - $BIMAP_SCOPE

Benchmarks:
$(printf '  - %s\n' "${BENCHMARKS[@]}")

Quick comparison:

  column -t -s \$'\\t' "$ACCEPT_TSV"
EOF

echo
echo "Saved results to: $OUT_DIR"
echo "Acceptance summary: $ACCEPT_TSV"
if [[ -f "$OUT_DIR/ctx_delta_bimap_lite_vs_adamw.tsv" ]]; then
  echo "CTX delta TSV: $OUT_DIR/ctx_delta_bimap_lite_vs_adamw.tsv"
  echo "CTX delta Markdown: $OUT_DIR/ctx_delta_bimap_lite_vs_adamw.md"
fi
