#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT_DIR/unit-tests/build/glades-unit-tests"
CALLGRIND_ANNOTATE_BIN="$(command -v callgrind_annotate)"

BENCHMARKS=("token-lm-document" "token-lm-corpus-large")
EPOCHS=1
SKIP_BUILD=0
ECHO_FIXED_SCALE=0.25
ECHO_SCHED_START=1.0
ECHO_SCHED_FINAL=0.25
ECHO_SCHED_DECAY=96
ECHO_SCOPE=all
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR_DEFAULT="$ROOT_DIR/artifacts/echo_callgrind_${TIMESTAMP}"
OUT_DIR="${OUT_DIR:-$OUT_DIR_DEFAULT}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Profiles CPU-side AdamW vs the two live ECHO candidates with valgrind/callgrind.
This is useful for host-side overhead analysis, not CUDA kernel timing.

Default ECHO candidates:
  - fixed:     ${ECHO_FIXED_SCALE}
  - scheduled: ${ECHO_SCHED_START} -> ${ECHO_SCHED_FINAL} over ${ECHO_SCHED_DECAY} steps

Options:
  --benchmark NAME         Benchmark to profile; may be passed multiple times
                           (default: token-lm-document and token-lm-corpus-large)
  --epochs N               Token epochs per run (default: 1)
  --out-dir PATH           Output directory
  --skip-build             Skip build + controller verification
  --echo-fixed-scale X     Fixed-scale ECHO candidate (default: 0.25)
  --echo-sched-start X     Scheduled ECHO start scale (default: 1.0)
  --echo-sched-final X     Scheduled ECHO final scale (default: 0.25)
  --echo-sched-decay N     Scheduled ECHO decay steps (default: 96)
  --echo-scope NAME        ECHO scope: all|large-only|late-head|late-head-large (default: all)
  --help                   Show this message
EOF
}

BENCHMARKS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark)
      BENCHMARKS+=("$2")
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
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
    --echo-fixed-scale)
      ECHO_FIXED_SCALE="$2"
      shift 2
      ;;
    --echo-sched-start)
      ECHO_SCHED_START="$2"
      shift 2
      ;;
    --echo-sched-final)
      ECHO_SCHED_FINAL="$2"
      shift 2
      ;;
    --echo-sched-decay)
      ECHO_SCHED_DECAY="$2"
      shift 2
      ;;
    --echo-scope)
      ECHO_SCOPE="$2"
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

if [[ ${#BENCHMARKS[@]} -eq 0 ]]; then
  BENCHMARKS=("token-lm-document" "token-lm-corpus-large")
fi

mkdir -p "$OUT_DIR/raw" "$OUT_DIR/annotated"

RUN_LOG="$OUT_DIR/run.log"

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

run_callgrind_profile() {
  local tag="$1"
  shift
  local callgrind_out="$OUT_DIR/${tag}.callgrind"
  local annotate_out="$OUT_DIR/annotated/${tag}.txt"

  echo "[profile] $tag" | tee -a "$RUN_LOG" >&2
  log_cmd valgrind --tool=callgrind --callgrind-out-file="$callgrind_out" "$@"
  valgrind \
    --tool=callgrind \
    --callgrind-out-file="$callgrind_out" \
    "$@" \
    >"$OUT_DIR/raw/${tag}.stdout.log" 2>"$OUT_DIR/raw/${tag}.stderr.log"

  log_cmd "$CALLGRIND_ANNOTATE_BIN" --inclusive=yes --threshold=0.5 "$callgrind_out"
  "$CALLGRIND_ANNOTATE_BIN" \
    --inclusive=yes \
    --threshold=0.5 \
    "$callgrind_out" \
    >"$annotate_out"
}

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  run_capture 00_build_main cmake --build "$ROOT_DIR/build" -j4
  run_capture 01_build_tests cmake --build "$ROOT_DIR/unit-tests/build" -j4 --target glades-unit-tests
  run_capture 02_atlas_controller "$BIN" atlas-controller
  run_capture 03_echo_core "$BIN" atlas-echo-core
  run_capture 04_echo_micro "$BIN" atlas-echo-micro
fi

for benchmark in "${BENCHMARKS[@]}"; do
  run_callgrind_profile "${benchmark}_adamw_e${EPOCHS}" \
    "$BIN" atlas-alt-bench \
    --mode "$benchmark" \
    --token-epochs "$EPOCHS" \
    --repeats 1 \
    --variant adamw

  run_callgrind_profile "${benchmark}_echo_fixed_e${EPOCHS}" \
    "$BIN" atlas-alt-bench \
    --mode "$benchmark" \
    --token-epochs "$EPOCHS" \
    --repeats 1 \
    --variant echo \
    --atlas-echo-geometry-scale "$ECHO_FIXED_SCALE" \
    --atlas-echo-final-geometry-scale "$ECHO_FIXED_SCALE" \
    --atlas-echo-decay-steps 0 \
    --atlas-echo-scope "$ECHO_SCOPE"

  run_callgrind_profile "${benchmark}_echo_sched_e${EPOCHS}" \
    "$BIN" atlas-alt-bench \
    --mode "$benchmark" \
    --token-epochs "$EPOCHS" \
    --repeats 1 \
    --variant echo \
    --atlas-echo-geometry-scale "$ECHO_SCHED_START" \
    --atlas-echo-final-geometry-scale "$ECHO_SCHED_FINAL" \
    --atlas-echo-decay-steps "$ECHO_SCHED_DECAY" \
    --atlas-echo-scope "$ECHO_SCOPE"
done

echo "Saved callgrind profiles to: $OUT_DIR"
