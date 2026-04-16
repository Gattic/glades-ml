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
BIMAP_GEOM=1.0
BIMAP_PRED=0.15
CADENCE_LIST=(1 2 4)
BENCHMARKS=(
  "token-lm-document"
  "token-lm-context"
  "token-lm-context-large"
)

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR_DEFAULT="$ROOT_DIR/artifacts/bimap_mode_cadence_acceptance_${TIMESTAMP}"
OUT_DIR="${OUT_DIR:-$OUT_DIR_DEFAULT}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Runs a focused BiMAP-lite cadence acceptance sweep on the modes most relevant to
deciding whether cadence=2 should replace the broader default:
  - token-lm-document
  - token-lm-context
  - token-lm-context-large

Variants:
  - adamw
  - bimap_lite_c1
  - bimap_lite_c2
  - bimap_lite_c4

Outputs:
  - acceptance_summary.tsv
  - acceptance_rank_by_nll.tsv
  - bimap_acceptance_rank_by_nll.tsv
  - ctx_delta_bimap_lite_c1_vs_adamw.tsv/.md
  - ctx_delta_bimap_lite_c2_vs_adamw.tsv/.md
  - ctx_delta_bimap_lite_c4_vs_adamw.tsv/.md
  - raw/*.log

Options:
  --gpu-device N              CUDA device id to request (default: 0)
  --bimap-scope S             all|head|late|late-head (default: late-head)
  --atlas-bimap-geometry-scale X
  --atlas-bimap-predictive-scale X
  --benchmarks CSV            Override benchmarks (default: token-lm-document,token-lm-context,token-lm-context-large)
  --out-dir PATH              Output directory (default: $OUT_DIR_DEFAULT)
  --skip-build                Skip clean build + verification
  --help                      Show this message
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
    --bimap-scope|--atlas-bimap-scope)
      BIMAP_SCOPE="$2"
      shift 2
      ;;
    --atlas-bimap-geometry-scale)
      BIMAP_GEOM="$2"
      shift 2
      ;;
    --atlas-bimap-predictive-scale)
      BIMAP_PRED="$2"
      shift 2
      ;;
    --benchmarks)
      parse_csv_words "$2" BENCHMARKS
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
ACCEPT_RANK_TSV="$OUT_DIR/acceptance_rank_by_nll.tsv"
BIMAP_ACCEPT_RANK_TSV="$OUT_DIR/bimap_acceptance_rank_by_nll.tsv"
RUN_LOG="$OUT_DIR/run.log"

if [[ "$SKIP_BUILD" -eq 1 ]]; then
  echo "[warn] --skip-build reuses existing binaries; unsafe after header/layout changes. Use one clean run first." | tee -a "$RUN_LOG" >&2
fi

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
            print $1, rank, $2, $3, $5, $9, $11, $12;
          }
        '
  } > "$ACCEPT_RANK_TSV"
}

generate_bimap_acceptance_rank() {
  {
    printf "benchmark\trank\toptimizer\ttrain_s_mean\ttok_s_mean\ttest_nll_mean\tstatus\tlogfile\n"
    tail -n +2 "$ACCEPT_TSV" \
      | awk -F '\t' '$2 ~ /^bimap_lite_c/' \
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
            print $1, rank, $2, $3, $5, $9, $11, $12;
          }
        '
  } > "$BIMAP_ACCEPT_RANK_TSV"
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
  local cand="$1"
  if [[ ! -f "$CTX_DELTA_PARSER" ]]; then
    echo "[warn] missing context delta parser: $CTX_DELTA_PARSER" | tee -a "$RUN_LOG" >&2
    return 0
  fi

  local prefix="ctx_delta_${cand}_vs_adamw"
  echo "[post] ctx delta parse for $cand" | tee -a "$RUN_LOG" >&2
  log_cmd python3 "$CTX_DELTA_PARSER" "$OUT_DIR" --cand "$cand" --out-prefix "$prefix"
  if python3 "$CTX_DELTA_PARSER" "$OUT_DIR" --cand "$cand" --out-prefix "$prefix" 2>&1 | tee "$OUT_DIR/raw/${prefix}.log"; then
    return 0
  fi

  echo "[warn] context delta parser produced no output for cand=$cand" | tee -a "$RUN_LOG" >&2
  return 0
}

has_ctx_rows() {
  local raw_dir="$1"
  if [[ ! -d "$raw_dir" ]]; then
    return 1
  fi

  rg -q '^CTX[[:space:]]' "$raw_dir"/accept_*.log 2>/dev/null
}

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  run_capture 00_nvidia_smi nvidia-smi
  run_capture 00a_clean_main cmake --build "$BUILD_DIR" --target clean
  run_capture 01_build_main cmake --build "$BUILD_DIR" -j4
  run_capture 01a_clean_tests cmake --build "$TEST_BUILD_DIR" --target clean
  run_capture 02_build_tests cmake --build "$TEST_BUILD_DIR" -j4 --target glades-unit-tests
  run_capture 03_atlas_controller "$BIN" atlas-controller
  run_capture 04_bimap_micro "$BIN" atlas-bimap-micro
  run_capture 05_bimap_parity "$BIN" atlas-bimap-parity
fi

run_capture 06_smoke_document_bimap_lite_c2 \
  "$BIN" atlas-alt-bench \
  --mode token-lm-document \
  --repeats 1 \
  --variant bimap \
  --atlas-bimap-scope "$BIMAP_SCOPE" \
  --atlas-bimap-low-rank 0 \
  --atlas-bimap-geometry-scale "$BIMAP_GEOM" \
  --atlas-bimap-predictive-scale "$BIMAP_PRED" \
  --atlas-bimap-factor-cadence 2 \
  --gpu-enable 1 \
  --gpu-device "$GPU_DEVICE"

for benchmark in "${BENCHMARKS[@]}"; do
  run_variant_accept "accept_${benchmark}_adamw" \
    "$benchmark" "adamw" \
    --variant adamw

  for cadence in "${CADENCE_LIST[@]}"; do
    run_variant_accept "accept_${benchmark}_bimap_lite_c${cadence}" \
      "$benchmark" "bimap_lite_c${cadence}" \
      --variant bimap \
      --atlas-bimap-scope "$BIMAP_SCOPE" \
      --atlas-bimap-low-rank 0 \
      --atlas-bimap-geometry-scale "$BIMAP_GEOM" \
      --atlas-bimap-predictive-scale "$BIMAP_PRED" \
      --atlas-bimap-factor-cadence "$cadence"
  done
done

generate_acceptance_rank
generate_bimap_acceptance_rank

if has_ctx_rows "$OUT_DIR/raw"; then
  for cadence in "${CADENCE_LIST[@]}"; do
    run_ctx_delta_parser "bimap_lite_c${cadence}"
  done
else
  echo "[info] skipping ctx delta parse; selected benchmarks emitted no CTX rows" | tee -a "$RUN_LOG" >&2
fi

cat > "$OUT_DIR/README.txt" <<EOF
Output directory: $OUT_DIR

Purpose:
- focused BiMAP-lite cadence acceptance sweep on document/context modes
- meant to decide whether cadence=2 should replace cadence=1 as the broader
  BiMAP-lite default, or stay corpus-scale specific

Files:
- run.log: command trace
- acceptance_summary.tsv: 10-repeat acceptance summary rows
- acceptance_rank_by_nll.tsv: all variants ranked by TestNLL then tok/s
- bimap_acceptance_rank_by_nll.tsv: BiMAP-lite-only acceptance ranking
- ctx_delta_bimap_lite_c*_vs_adamw.tsv/.md: context bucket deltas when the
  selected benchmarks emit CTX rows
- raw/*.log: full raw benchmark outputs

BiMAP settings:
- scope: $BIMAP_SCOPE
- geometry scale: $BIMAP_GEOM
- predictive scale: $BIMAP_PRED
- cadence sweep: ${CADENCE_LIST[*]}

Benchmarks:
$(printf '  - %s\n' "${BENCHMARKS[@]}")
EOF

echo "Saved results to: $OUT_DIR"
echo "Acceptance summary: $ACCEPT_TSV"
echo "Acceptance ranking: $ACCEPT_RANK_TSV"
echo "BiMAP acceptance ranking: $BIMAP_ACCEPT_RANK_TSV"
for cadence in "${CADENCE_LIST[@]}"; do
  if [[ -f "$OUT_DIR/ctx_delta_bimap_lite_c${cadence}_vs_adamw.tsv" ]]; then
    echo "CTX delta TSV (c${cadence}): $OUT_DIR/ctx_delta_bimap_lite_c${cadence}_vs_adamw.tsv"
    echo "CTX delta Markdown (c${cadence}): $OUT_DIR/ctx_delta_bimap_lite_c${cadence}_vs_adamw.md"
  fi
done
