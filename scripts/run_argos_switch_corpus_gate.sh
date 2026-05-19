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

ARGOS_GEOM=1.0
ARGOS_ORTH=0.5
ARGOS_PRED=0.05
ARGOS_TRUST=0.20
ARGOS_WARMUP=32
ARGOS_WARMUP_START=0.25
ARGOS_ACT=1.0
ARGOS_SCOPE=head
ARGOS_CADENCE=1
ARGOS_ORTH_CADENCE=1
ARGOS_MAX_ASPECT=1.50
ARGOS_MIN_DIM=8
ARGOS_DAMPING=0.01
ARGOS_OBS=0.75
ARGOS_HEAD=0.35
ARGOS_LATE=0.00
ARGOS_SWITCH_FRACTIONS_CSV="0.50,0.67,0.75"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR_DEFAULT="$ROOT_DIR/artifacts/argos_switch_corpus_gate_${TIMESTAMP}"
OUT_DIR="${OUT_DIR:-$OUT_DIR_DEFAULT}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Runs a corpus-large GPU gate comparing:
  - AdamW
  - ATLAS-ARGOS
  - ATLAS-ARGOS-SWITCH across switch fractions

Default switch fractions:
  ${ARGOS_SWITCH_FRACTIONS_CSV}

Outputs:
  - epoch_sweep_summary.tsv
  - acceptance_summary.tsv
  - acceptance_rank_by_nll.tsv
  - epoch4_rank_by_nll.tsv
  - switch_acceptance_rank_by_nll.tsv
  - switch_epoch4_rank_by_nll.tsv
  - raw/*.log

Options:
  --gpu-device N                       CUDA device id to request (default: 0)
  --repeats N                          Repeats per epoch point (default: 5)
  --out-dir PATH                       Output directory (default: $OUT_DIR_DEFAULT)
  --skip-build                         Skip build + ARGOS verification (only safe after a clean rebuild)
  --skip-acceptance                    Skip the 10-repeat acceptance pass
  --atlas-argos-switch-fractions CSV   Comma-separated switch fractions in [0,1] (default: ${ARGOS_SWITCH_FRACTIONS_CSV})
  --atlas-argos-geometry-scale X
  --atlas-argos-orthogonal-scale X
  --atlas-argos-predictive-scale X
  --atlas-argos-trust-radius X
  --atlas-argos-warmup-steps N
  --atlas-argos-warmup-start-scale X
  --atlas-argos-actuation-scale X
  --atlas-argos-scope S
  --atlas-argos-cadence N
  --atlas-argos-orth-cadence N
  --atlas-argos-max-aspect X
  --atlas-argos-min-dim N
  --atlas-argos-damping X
  --atlas-argos-observability-scale X
  --atlas-argos-head-bonus X
  --atlas-argos-late-bonus X
  --help                               Show this message
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
    --atlas-argos-switch-fractions)
      ARGOS_SWITCH_FRACTIONS_CSV="$2"
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
    --atlas-argos-warmup-steps)
      ARGOS_WARMUP="$2"
      shift 2
      ;;
    --atlas-argos-warmup-start-scale)
      ARGOS_WARMUP_START="$2"
      shift 2
      ;;
    --atlas-argos-actuation-scale)
      ARGOS_ACT="$2"
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

IFS=',' read -r -a ARGOS_SWITCH_FRACTIONS <<< "$ARGOS_SWITCH_FRACTIONS_CSV"
if [[ "${#ARGOS_SWITCH_FRACTIONS[@]}" -eq 0 ]]; then
  echo "empty --atlas-argos-switch-fractions list" >&2
  exit 2
fi
for fraction in "${ARGOS_SWITCH_FRACTIONS[@]}"; do
  if [[ ! "$fraction" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "invalid ARGOS switch fraction: $fraction" >&2
    exit 2
  fi
  if ! awk -v x="$fraction" 'BEGIN { exit !(x >= 0.0 && x <= 1.0) }'; then
    echo "ARGOS switch fraction out of range [0,1]: $fraction" >&2
    exit 2
  fi
done

fraction_label() {
  local fraction="$1"
  local label="${fraction//./}"
  label="${label//[^0-9]/}"
  if [[ -z "$label" ]]; then
    label="0"
  fi
  printf '%s' "$label"
}

mkdir -p "$OUT_DIR/raw"

SUMMARY_TSV="$OUT_DIR/epoch_sweep_summary.tsv"
ACCEPT_TSV="$OUT_DIR/acceptance_summary.tsv"
ACCEPT_RANK_TSV="$OUT_DIR/acceptance_rank_by_nll.tsv"
E4_RANK_TSV="$OUT_DIR/epoch4_rank_by_nll.tsv"
SWITCH_ACCEPT_RANK_TSV="$OUT_DIR/switch_acceptance_rank_by_nll.tsv"
SWITCH_E4_RANK_TSV="$OUT_DIR/switch_epoch4_rank_by_nll.tsv"
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
    /^(AdamW|ATLAS-ARGOS|ATLAS-ARGOS-SWITCH)[[:space:]]/ && NF >= 20 {
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
    /^(AdamW|ATLAS-ARGOS|ATLAS-ARGOS-SWITCH)[[:space:]]/ && NF >= 20 {
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
             benchmark, optimizer,
             $2, $4, $5, $7, $8, $10, $14, $16, $20, logfile;
      exit 0;
    }
  ' "$logfile" >> "$ACCEPT_TSV"
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
    awk -F '\t' '$1 == "'"$BENCHMARK"'" && $2 ~ /^argos_switch_f/ && $3 == "4"' "$SUMMARY_TSV" \
      | sort -t $'\t' -k10,10g -k4,4g \
      | awk -F '\t' '
          BEGIN { OFS = "\t"; rank = 0; }
          {
            rank += 1;
            print $1, rank, $2, $3, $4, $10, $13;
          }
        '
  } > "$SWITCH_E4_RANK_TSV"
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
    awk -F '\t' '$1 == "'"$BENCHMARK"'" && $2 ~ /^argos_switch_f/' "$ACCEPT_TSV" \
      | sort -t $'\t' -k9,9g -k3,3g \
      | awk -F '\t' '
          BEGIN { OFS = "\t"; rank = 0; }
          {
            rank += 1;
            print $1, rank, $2, $3, $9, $12;
          }
        '
  } > "$SWITCH_ACCEPT_RANK_TSV"
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

ARGOS_ARGS=(
  --variant argos
  --atlas-argos-geometry-scale "$ARGOS_GEOM"
  --atlas-argos-orthogonal-scale "$ARGOS_ORTH"
  --atlas-argos-predictive-scale "$ARGOS_PRED"
  --atlas-argos-trust-radius "$ARGOS_TRUST"
  --atlas-argos-warmup-steps "$ARGOS_WARMUP"
  --atlas-argos-warmup-start-scale "$ARGOS_WARMUP_START"
  --atlas-argos-actuation-scale "$ARGOS_ACT"
  --atlas-argos-scope "$ARGOS_SCOPE"
  --atlas-argos-cadence "$ARGOS_CADENCE"
  --atlas-argos-orth-cadence "$ARGOS_ORTH_CADENCE"
  --atlas-argos-max-aspect "$ARGOS_MAX_ASPECT"
  --atlas-argos-min-dim "$ARGOS_MIN_DIM"
  --atlas-argos-damping "$ARGOS_DAMPING"
  --atlas-argos-observability-scale "$ARGOS_OBS"
  --atlas-argos-head-bonus "$ARGOS_HEAD"
  --atlas-argos-late-bonus "$ARGOS_LATE"
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

run_capture 06_smoke_argos_switch_gpu \
  "$BIN" atlas-alt-bench \
  --mode token-lm \
  --token-epochs 4 \
  --repeats 1 \
  --variant argos-switch \
  --atlas-argos-switch-fraction "${ARGOS_SWITCH_FRACTIONS[0]}" \
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
  --atlas-argos-late-bonus "$ARGOS_LATE" \
  --gpu-enable 1 \
  --gpu-device "$GPU_DEVICE"

for epochs in "${EPOCHS[@]}"; do
  run_variant_epoch "sweep_${BENCHMARK}_adamw_e${epochs}" \
    "adamw" "$epochs" \
    --variant adamw

  run_variant_epoch "sweep_${BENCHMARK}_argos_e${epochs}" \
    "argos" "$epochs" \
    "${ARGOS_ARGS[@]}"

  for fraction in "${ARGOS_SWITCH_FRACTIONS[@]}"; do
    label="argos_switch_f$(fraction_label "$fraction")"
    run_variant_epoch "sweep_${BENCHMARK}_${label}_e${epochs}" \
      "$label" "$epochs" \
      --variant argos-switch \
      --atlas-argos-switch-fraction "$fraction" \
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
done

if [[ "$RUN_ACCEPTANCE" -eq 1 ]]; then
  run_variant_accept "accept_${BENCHMARK}_adamw" \
    "adamw" \
    --variant adamw

  run_variant_accept "accept_${BENCHMARK}_argos" \
    "argos" \
    "${ARGOS_ARGS[@]}"

  for fraction in "${ARGOS_SWITCH_FRACTIONS[@]}"; do
    label="argos_switch_f$(fraction_label "$fraction")"
    run_variant_accept "accept_${BENCHMARK}_${label}" \
      "$label" \
      --variant argos-switch \
      --atlas-argos-switch-fraction "$fraction" \
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
fi

if [[ "$RUN_ACCEPTANCE" -eq 1 ]]; then
  generate_acceptance_rank
fi
generate_epoch4_rank

cat > "$OUT_DIR/README.txt" <<EOF
Output directory: $OUT_DIR

Files:
  raw/*.log                       Raw benchmark and unit-test output
  epoch_sweep_summary.tsv         ${REPEATS}-repeat epoch sweep summary
  acceptance_summary.tsv          10-repeat acceptance summary
  acceptance_rank_by_nll.tsv      Overall acceptance ranking
  epoch4_rank_by_nll.tsv          Overall epoch-4 ranking
  switch_acceptance_rank_by_nll.tsv
                                  ARGOS-switch acceptance ranking
  switch_epoch4_rank_by_nll.tsv   ARGOS-switch epoch-4 ranking
  run.log                         Full command log

Files to send back for review:
  - epoch_sweep_summary.tsv
  - acceptance_summary.tsv
  - acceptance_rank_by_nll.tsv
  - epoch4_rank_by_nll.tsv
  - switch_acceptance_rank_by_nll.tsv
  - switch_epoch4_rank_by_nll.tsv
  - raw/05_argos_parity.log

Benchmark:
  $BENCHMARK

ARGOS switch fractions:
  ${ARGOS_SWITCH_FRACTIONS_CSV}

Decision rule:
  Continue staged ARGOS only if at least one switch fraction improves
  late epoch-4 corpus-large NLL over pure AdamW and pure ARGOS without
  giving back too much acceptance quality.
EOF

echo "Saved results to: $OUT_DIR"
echo "Epoch sweep summary: $SUMMARY_TSV"
if [[ "$RUN_ACCEPTANCE" -eq 1 ]]; then
  echo "Acceptance summary: $ACCEPT_TSV"
  echo "Acceptance ranking: $ACCEPT_RANK_TSV"
  echo "Switch acceptance ranking: $SWITCH_ACCEPT_RANK_TSV"
fi
echo "Epoch-4 ranking: $E4_RANK_TSV"
echo "Switch epoch-4 ranking: $SWITCH_E4_RANK_TSV"
