#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT_DIR/unit-tests/build/glades-unit-tests"
NSYS_BIN="$(command -v nsys)"
NSYS_REAL_BIN="$(readlink -f "$NSYS_BIN" 2>/dev/null || printf '%s' "$NSYS_BIN")"
NSYS_BIN_DIR="$(dirname "$NSYS_REAL_BIN")"
NSYS_HOST_DIR_CANDIDATES=(
  "/usr/lib/nsight-systems/host-linux-x64"
  "$NSYS_BIN_DIR/../host-linux-x64"
  "$NSYS_BIN_DIR/../../host-linux-x64"
  "$(cd "$NSYS_BIN_DIR/../host-linux-x64" 2>/dev/null && pwd || true)"
  "$(cd "$NSYS_BIN_DIR/../../host-linux-x64" 2>/dev/null && pwd || true)"
)
QDSTRM_IMPORTER="${QDSTRM_IMPORTER:-}"

GPU_DEVICE=0
BENCHMARKS=("token-lm-document" "token-lm-corpus-large")
EPOCHS=1
SKIP_BUILD=0
ECHO_FIXED_SCALE=1.0
ECHO_SCHED_START=1.0
ECHO_SCHED_FINAL=0.25
ECHO_SCHED_DECAY=96
ECHO_CADENCE=1
ECHO_SCOPE=late-head
ECHO_TRUST_SCALE=0.0
ECHO_PREDICTIVE_SCALE=0.0
ECHO_STRUCTURAL_SCALE=0.0
ECHO_STRUCTURAL_GROUPS=1
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR_DEFAULT="$ROOT_DIR/artifacts/echo_nsys_${TIMESTAMP}"
OUT_DIR="${OUT_DIR:-$OUT_DIR_DEFAULT}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Profiles AdamW vs the two live ECHO candidates with Nsight Systems on the hard transformer benchmarks.
This is a profiling script, not an acceptance gate.

Default ECHO candidates:
  - fixed:     ${ECHO_FIXED_SCALE}
  - scheduled: ${ECHO_SCHED_START} -> ${ECHO_SCHED_FINAL} over ${ECHO_SCHED_DECAY} steps

Options:
  --gpu-device N                   CUDA device id to request (default: 0)
  --benchmark NAME                 Benchmark to profile; may be passed multiple times
                                   (default: token-lm-document and token-lm-corpus-large)
  --epochs N                       Token epochs per run (default: 1)
  --out-dir PATH                   Output directory
  --skip-build                     Skip build + controller verification
  --echo-fixed-scale X             Fixed-scale ECHO candidate (default: ${ECHO_FIXED_SCALE})
  --echo-sched-start X             Scheduled ECHO start scale (default: 1.0)
  --echo-sched-final X             Scheduled ECHO final scale (default: 0.25)
  --echo-sched-decay N             Scheduled ECHO decay steps (default: 96)
  --echo-cadence N                 Optimizer steps between ECHO metric refreshes (default: 1)
  --echo-scope NAME                ECHO scope: all|large-only|late-head|late-head-large (default: late-head)
  --echo-trust-scale X             ECHO trust-gate strength (default: 0.0)
  --echo-predictive-scale X        ECHO bounded predictive blend strength (default: 0.0)
  --echo-structural-scale X        ECHO grouped structural factor strength (default: 0.0)
  --echo-structural-groups N       ECHO contiguous row/col group count (default: 1)
  --qdstrm-importer PATH           Explicit QdstrmImporter path
  --help                           Show this message
EOF
}

resolve_qdstrm_importer() {
  if [[ -n "$QDSTRM_IMPORTER" && -x "$QDSTRM_IMPORTER" ]]; then
    return 0
  fi
  if command -v QdstrmImporter >/dev/null 2>&1; then
    QDSTRM_IMPORTER="$(command -v QdstrmImporter)"
    return 0
  fi
  local candidate
  for candidate in "${NSYS_HOST_DIR_CANDIDATES[@]}"; do
    if [[ -n "$candidate" && -x "$candidate/QdstrmImporter" ]]; then
      QDSTRM_IMPORTER="$candidate/QdstrmImporter"
      return 0
    fi
  done
  return 1
}

BENCHMARKS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-device)
      GPU_DEVICE="$2"
      shift 2
      ;;
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
    --echo-cadence|--atlas-echo-cadence)
      ECHO_CADENCE="$2"
      shift 2
      ;;
    --echo-scope|--atlas-echo-scope)
      ECHO_SCOPE="$2"
      shift 2
      ;;
    --echo-trust-scale|--atlas-echo-trust-scale)
      ECHO_TRUST_SCALE="$2"
      shift 2
      ;;
    --echo-predictive-scale|--atlas-echo-predictive-scale)
      ECHO_PREDICTIVE_SCALE="$2"
      shift 2
      ;;
    --echo-structural-scale|--atlas-echo-structural-scale)
      ECHO_STRUCTURAL_SCALE="$2"
      shift 2
      ;;
    --echo-structural-groups|--atlas-echo-structural-groups)
      ECHO_STRUCTURAL_GROUPS="$2"
      shift 2
      ;;
    --qdstrm-importer)
      QDSTRM_IMPORTER="$2"
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

mkdir -p "$OUT_DIR/raw" "$OUT_DIR/stats"
echo "Writing Nsight artifacts to: $OUT_DIR"

if ! resolve_qdstrm_importer; then
  echo "Unable to locate QdstrmImporter. Set --qdstrm-importer PATH or QDSTRM_IMPORTER." >&2
  exit 1
fi

ECHO_COMMON_ARGS=(
  --atlas-echo-cadence "$ECHO_CADENCE"
  --atlas-echo-scope "$ECHO_SCOPE"
  --atlas-echo-trust-scale "$ECHO_TRUST_SCALE"
  --atlas-echo-predictive-scale "$ECHO_PREDICTIVE_SCALE"
  --atlas-echo-structural-scale "$ECHO_STRUCTURAL_SCALE"
  --atlas-echo-structural-groups "$ECHO_STRUCTURAL_GROUPS"
)

RUN_LOG="$OUT_DIR/run.log"
printf 'Using QdstrmImporter: %s\n' "$QDSTRM_IMPORTER" >> "$RUN_LOG"

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

run_nsys_profile() {
  local tag="$1"
  shift
  local rep_base="$OUT_DIR/${tag}"
  local rep_file="${rep_base}.nsys-rep"
  local sqlite_file="${rep_base}.sqlite"
  local qdstrm_file="${rep_base}.qdstrm"
  local stats_base="$OUT_DIR/stats/${tag}"
  local stats_input=""
  local profile_rc=0

  echo "[profile] $tag" | tee -a "$RUN_LOG" >&2
  log_cmd nsys profile --force-overwrite=true --export=sqlite -o "$rep_base" --sample=none --cpuctxsw=none --trace=cuda,nvtx,osrt "$@"
  set +e
  nsys profile \
    --force-overwrite=true \
    --export=sqlite \
    -o "$rep_base" \
    --sample=none \
    --cpuctxsw=none \
    --trace=cuda,nvtx,osrt \
    "$@" \
    >"$OUT_DIR/raw/${tag}.stdout.log" 2>"$OUT_DIR/raw/${tag}.stderr.log"
  profile_rc=$?
  set -e

  if [[ "$profile_rc" -ne 0 && ! -f "$rep_file" && ! -f "$qdstrm_file" ]]; then
    echo "nsys profile failed for $tag (rc=$profile_rc)" | tee -a "$RUN_LOG" >&2
    return "$profile_rc"
  fi
  if [[ "$profile_rc" -ne 0 && -f "$qdstrm_file" ]]; then
    echo "nsys profile returned rc=$profile_rc for $tag; recovering via manual QDSTRM import" | tee -a "$RUN_LOG" >&2
  fi

  if [[ -f "$sqlite_file" ]]; then
    stats_input="$sqlite_file"
  fi

  if [[ ! -f "$rep_file" && -f "$qdstrm_file" ]]; then
    if [[ -x "$QDSTRM_IMPORTER" ]]; then
      log_cmd "$QDSTRM_IMPORTER" -f -i "$qdstrm_file" -o "$rep_file"
      "$QDSTRM_IMPORTER" -f -i "$qdstrm_file" -o "$rep_file" \
        >"$OUT_DIR/raw/${tag}.import.stdout.log" 2>"$OUT_DIR/raw/${tag}.import.stderr.log"
    else
      echo "missing QdstrmImporter: $QDSTRM_IMPORTER" | tee -a "$RUN_LOG" >&2
      return 1
    fi
  fi

  if [[ -z "$stats_input" && -f "$sqlite_file" ]]; then
    stats_input="$sqlite_file"
  fi

  if [[ -z "$stats_input" && -f "$rep_file" ]]; then
    stats_input="$rep_file"
  fi

  if [[ -z "$stats_input" ]]; then
    echo "nsys profile did not produce a usable sqlite/report for $tag" | tee -a "$RUN_LOG" >&2
    return 1
  fi

  log_cmd nsys stats --force-overwrite true --report cudaapisum,gpukernsum,gpumemtimesum,osrtsum --format csv --output "$stats_base" "$stats_input"
  nsys stats \
    --force-overwrite true \
    --report cudaapisum,gpukernsum,gpumemtimesum,osrtsum \
    --format csv \
    --output "$stats_base" \
    "$stats_input" \
    >"$OUT_DIR/raw/${tag}.stats.stdout.log" 2>"$OUT_DIR/raw/${tag}.stats.stderr.log"
}

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  run_capture 00_nvidia_smi nvidia-smi
  run_capture 01_build_main cmake --build "$ROOT_DIR/build" -j4
  run_capture 02_build_tests cmake --build "$ROOT_DIR/unit-tests/build" -j4 --target glades-unit-tests
  run_capture 03_atlas_controller "$BIN" atlas-controller
  run_capture 04_echo_core "$BIN" atlas-echo-core
  run_capture 05_echo_micro "$BIN" atlas-echo-micro
fi

for benchmark in "${BENCHMARKS[@]}"; do
  run_nsys_profile "${benchmark}_adamw_e${EPOCHS}" \
    "$BIN" atlas-alt-bench \
    --mode "$benchmark" \
    --token-epochs "$EPOCHS" \
    --repeats 1 \
    --variant adamw \
    --gpu-enable 1 \
    --gpu-device "$GPU_DEVICE"

  run_nsys_profile "${benchmark}_echo_fixed_e${EPOCHS}" \
    "$BIN" atlas-alt-bench \
    --mode "$benchmark" \
    --token-epochs "$EPOCHS" \
    --repeats 1 \
    --variant echo \
    --atlas-echo-geometry-scale "$ECHO_FIXED_SCALE" \
    --atlas-echo-final-geometry-scale "$ECHO_FIXED_SCALE" \
    --atlas-echo-decay-steps 0 \
    "${ECHO_COMMON_ARGS[@]}" \
    --gpu-enable 1 \
    --gpu-device "$GPU_DEVICE"

  run_nsys_profile "${benchmark}_echo_sched_e${EPOCHS}" \
    "$BIN" atlas-alt-bench \
    --mode "$benchmark" \
    --token-epochs "$EPOCHS" \
    --repeats 1 \
    --variant echo \
    --atlas-echo-geometry-scale "$ECHO_SCHED_START" \
    --atlas-echo-final-geometry-scale "$ECHO_SCHED_FINAL" \
    --atlas-echo-decay-steps "$ECHO_SCHED_DECAY" \
    "${ECHO_COMMON_ARGS[@]}" \
    --gpu-enable 1 \
    --gpu-device "$GPU_DEVICE"
done

echo "Saved Nsight profiles to: $OUT_DIR"
