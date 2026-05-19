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
EPOCHS_CSV="1,2,3,4"
SPECS_CSV="late-head:1.0:1.0:0:0.00,late-head:1.0:0.50:96:0.00,late-head:1.0:0.25:96:0.00,late-head:1.0:1.0:0:0.02,late-head:1.0:0.50:96:0.02,late-head-large:1.0:1.0:0:0.00,late-head-large:1.0:0.50:96:0.00,late-head-large:1.0:1.0:0:0.02"
ECHO_CADENCE=1
ECHO_TRUST_SCALE=0.0
ECHO_STRUCTURAL_SCALE=0.0
ECHO_STRUCTURAL_GROUPS=1

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
OUT_DIR_DEFAULT="$ROOT_DIR/artifacts/echo_corpus_large_sweep_${TIMESTAMP}"
OUT_DIR="${OUT_DIR:-$OUT_DIR_DEFAULT}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Runs a dedicated token-lm-corpus-large ECHO sweep against AdamW.

Purpose:
  - test only cheap ECHO knobs that still look plausible for beating AdamW
  - sweep scope, final geometry scale, and small predictive scale
  - rank on acceptance and epoch 4

Default benchmark:
  - token-lm-corpus-large

Default epochs:
  - 1,2,3,4

Default ECHO specs:
  - late-head:1.0:1.0:0:0.00
  - late-head:1.0:0.50:96:0.00
  - late-head:1.0:0.25:96:0.00
  - late-head:1.0:1.0:0:0.02
  - late-head:1.0:0.50:96:0.02
  - late-head-large:1.0:1.0:0:0.00
  - late-head-large:1.0:0.50:96:0.00
  - late-head-large:1.0:1.0:0:0.02

Spec format:
  scope:start_scale:final_scale:decay_steps:predictive_scale

Supported scopes:
  - all
  - large-only
  - late-head
  - late-head-large

Outputs:
  - epoch_sweep_summary.tsv
  - acceptance_summary.tsv
  - acceptance_rank_by_nll.tsv
  - echo_acceptance_rank_by_nll.tsv
  - epoch4_rank_by_nll.tsv
  - echo_epoch4_rank_by_nll.tsv
  - final_epoch_rank_by_nll.tsv
  - echo_final_epoch_rank_by_nll.tsv
  - raw/*.log

Options:
  --gpu-device N                 CUDA device id to request (default: 0)
  --repeats N                    Repeats per epoch point (default: 5)
  --epochs CSV                   Comma-separated epoch list (default: 1,2,3,4)
  --specs CSV                    Comma-separated ECHO specs
  --echo-cadence N               ECHO metric refresh cadence (default: 1)
  --echo-trust-scale X           ECHO trust-gate strength (default: 0.0)
  --echo-structural-scale X      ECHO structural factor strength (default: 0.0)
  --echo-structural-groups N     ECHO structural group count (default: 1)
  --out-dir PATH                 Output directory (default: $OUT_DIR_DEFAULT)
  --skip-build                   Skip clean build + verification
  --skip-acceptance              Skip the 10-repeat acceptance pass
  --help                         Show this message
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

parse_spec() {
  local spec="$1"
  local scope start_scale final_scale decay_steps predictive_scale
  IFS=':' read -r scope start_scale final_scale decay_steps predictive_scale <<< "$spec"
  if [[ -z "${scope:-}" || -z "${start_scale:-}" || -z "${final_scale:-}" || -z "${decay_steps:-}" || -z "${predictive_scale:-}" ]]; then
    echo "invalid spec: $spec" >&2
    exit 2
  fi
  case "$scope" in
    all|large-only|late-head|late-head-large)
      ;;
    *)
      echo "invalid ECHO scope in spec: $scope" >&2
      exit 2
      ;;
  esac
  printf '%s %s %s %s %s\n' "$scope" "$start_scale" "$final_scale" "$decay_steps" "$predictive_scale"
}

tag_component() {
  local value="$1"
  value="${value//late-head/lh}"
  value="${value//large-only/lo}"
  value="${value//./p}"
  value="${value//-/m}"
  printf '%s' "$value"
}

spec_tag() {
  local scope="$1"
  local start_scale="$2"
  local final_scale="$3"
  local decay_steps="$4"
  local predictive_scale="$5"
  printf '%s_g%s_f%s_d%s_p%s' \
    "$(tag_component "$scope")" \
    "$(tag_component "$start_scale")" \
    "$(tag_component "$final_scale")" \
    "$decay_steps" \
    "$(tag_component "$predictive_scale")"
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
      EPOCHS_CSV="$2"
      shift 2
      ;;
    --specs)
      SPECS_CSV="$2"
      shift 2
      ;;
    --echo-cadence|--atlas-echo-cadence)
      ECHO_CADENCE="$2"
      shift 2
      ;;
    --echo-trust-scale|--atlas-echo-trust-scale)
      ECHO_TRUST_SCALE="$2"
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

parse_csv_words "$EPOCHS_CSV" EPOCHS
parse_csv_words "$SPECS_CSV" SPECS

mkdir -p "$OUT_DIR/raw"

SUMMARY_TSV="$OUT_DIR/epoch_sweep_summary.tsv"
ACCEPT_TSV="$OUT_DIR/acceptance_summary.tsv"
ACCEPT_RANK_TSV="$OUT_DIR/acceptance_rank_by_nll.tsv"
ECHO_ACCEPT_RANK_TSV="$OUT_DIR/echo_acceptance_rank_by_nll.tsv"
E4_RANK_TSV="$OUT_DIR/epoch4_rank_by_nll.tsv"
ECHO_E4_RANK_TSV="$OUT_DIR/echo_epoch4_rank_by_nll.tsv"
FINAL_RANK_TSV="$OUT_DIR/final_epoch_rank_by_nll.tsv"
ECHO_FINAL_RANK_TSV="$OUT_DIR/echo_final_epoch_rank_by_nll.tsv"
RUN_LOG="$OUT_DIR/run.log"
FINAL_EPOCH="$(max_epoch "${EPOCHS[@]}")"

if [[ "$SKIP_BUILD" -eq 1 ]]; then
  echo "[warn] --skip-build reuses existing binaries; this is unsafe after C++ header/layout changes. Use a clean rebuild first." | tee -a "$RUN_LOG" >&2
fi

cat > "$SUMMARY_TSV" <<'EOF'
benchmark	optimizer	variant	echo_scope	echo_scale_start	echo_scale_final	echo_decay_steps	echo_predictive_scale	epochs	train_s_mean	train_s_pm	tok_s_mean	tok_s_pm	train_nll_mean	train_nll_pm	test_nll_mean	test_nll_pm	status	logfile
EOF

cat > "$ACCEPT_TSV" <<'EOF'
benchmark	optimizer	variant	echo_scope	echo_scale_start	echo_scale_final	echo_decay_steps	echo_predictive_scale	train_s_mean	train_s_pm	tok_s_mean	tok_s_pm	train_nll_mean	train_nll_pm	test_nll_mean	test_nll_pm	status	logfile
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
  local variant="$3"
  local scope="$4"
  local start_scale="$5"
  local final_scale="$6"
  local decay_steps="$7"
  local predictive_scale="$8"
  local epochs="$9"
  local logfile="${10}"
  awk -v benchmark="$benchmark" -v optimizer="$optimizer" -v variant="$variant" -v scope="$scope" \
      -v start_scale="$start_scale" -v final_scale="$final_scale" -v decay_steps="$decay_steps" \
      -v predictive_scale="$predictive_scale" -v epochs="$epochs" -v logfile="$logfile" '
    /^(AdamW|ATLAS-ECHO)[[:space:]]/ && NF >= 20 {
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
             benchmark, optimizer, variant, scope,
             start_scale, final_scale, decay_steps, predictive_scale, epochs,
             $2, $4, $5, $7, $8, $10, $14, $16, $20, logfile;
      exit 0;
    }
  ' "$logfile" >> "$SUMMARY_TSV"
}

append_acceptance_summary() {
  local benchmark="$1"
  local optimizer="$2"
  local variant="$3"
  local scope="$4"
  local start_scale="$5"
  local final_scale="$6"
  local decay_steps="$7"
  local predictive_scale="$8"
  local logfile="$9"
  awk -v benchmark="$benchmark" -v optimizer="$optimizer" -v variant="$variant" -v scope="$scope" \
      -v start_scale="$start_scale" -v final_scale="$final_scale" -v decay_steps="$decay_steps" \
      -v predictive_scale="$predictive_scale" -v logfile="$logfile" '
    /^(AdamW|ATLAS-ECHO)[[:space:]]/ && NF >= 20 {
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
             benchmark, optimizer, variant, scope,
             start_scale, final_scale, decay_steps, predictive_scale,
             $2, $4, $5, $7, $8, $10, $14, $16, $20, logfile;
      exit 0;
    }
  ' "$logfile" >> "$ACCEPT_TSV"
}

generate_acceptance_rank() {
  {
    printf "benchmark\trank\tvariant\toptimizer\techo_scope\techo_scale_start\techo_scale_final\techo_decay_steps\techo_predictive_scale\ttrain_s_mean\ttok_s_mean\ttest_nll_mean\tstatus\tlogfile\n"
    tail -n +2 "$ACCEPT_TSV" \
      | sort -t $'\t' -k1,1 -k15,15g -k9,9g \
      | awk -F '\t' '
          BEGIN { OFS = "\t"; prev = ""; rank = 0; }
          {
            if ($1 != prev) {
              prev = $1;
              rank = 1;
            } else {
              rank += 1;
            }
            print $1, rank, $3, $2, $4, $5, $6, $7, $8, $9, $11, $15, $17, $18;
          }
        '
  } > "$ACCEPT_RANK_TSV"
}

generate_echo_acceptance_rank() {
  {
    printf "benchmark\trank\tvariant\techo_scope\techo_scale_start\techo_scale_final\techo_decay_steps\techo_predictive_scale\ttrain_s_mean\ttok_s_mean\ttest_nll_mean\tstatus\tlogfile\n"
    tail -n +2 "$ACCEPT_TSV" \
      | awk -F '\t' '$2 == "echo"' \
      | sort -t $'\t' -k1,1 -k15,15g -k9,9g \
      | awk -F '\t' '
          BEGIN { OFS = "\t"; prev = ""; rank = 0; }
          {
            if ($1 != prev) {
              prev = $1;
              rank = 1;
            } else {
              rank += 1;
            }
            print $1, rank, $3, $4, $5, $6, $7, $8, $9, $11, $15, $17, $18;
          }
        '
  } > "$ECHO_ACCEPT_RANK_TSV"
}

generate_epoch_rank() {
  local epoch="$1"
  local out_tsv="$2"
  {
    printf "benchmark\trank\tvariant\toptimizer\techo_scope\techo_scale_start\techo_scale_final\techo_decay_steps\techo_predictive_scale\tepochs\ttrain_s_mean\ttok_s_mean\ttest_nll_mean\tstatus\tlogfile\n"
    awk -F '\t' -v epoch="$epoch" 'NR == 1 || $9 == epoch' "$SUMMARY_TSV" \
      | tail -n +2 \
      | sort -t $'\t' -k1,1 -k16,16g -k10,10g \
      | awk -F '\t' '
          BEGIN { OFS = "\t"; prev = ""; rank = 0; }
          {
            if ($1 != prev) {
              prev = $1;
              rank = 1;
            } else {
              rank += 1;
            }
            print $1, rank, $3, $2, $4, $5, $6, $7, $8, $9, $10, $12, $16, $18, $19;
          }
        '
  } > "$out_tsv"
}

generate_echo_epoch_rank() {
  local epoch="$1"
  local out_tsv="$2"
  {
    printf "benchmark\trank\tvariant\techo_scope\techo_scale_start\techo_scale_final\techo_decay_steps\techo_predictive_scale\tepochs\ttrain_s_mean\ttok_s_mean\ttest_nll_mean\tstatus\tlogfile\n"
    awk -F '\t' -v epoch="$epoch" '$2 == "echo" && $9 == epoch' "$SUMMARY_TSV" \
      | sort -t $'\t' -k1,1 -k16,16g -k10,10g \
      | awk -F '\t' '
          BEGIN { OFS = "\t"; prev = ""; rank = 0; }
          {
            if ($1 != prev) {
              prev = $1;
              rank = 1;
            } else {
              rank += 1;
            }
            print $1, rank, $3, $4, $5, $6, $7, $8, $9, $10, $12, $16, $18, $19;
          }
        '
  } > "$out_tsv"
}

run_variant_epoch() {
  local name="$1"
  local benchmark="$2"
  local epochs="$3"
  local optimizer="$4"
  local variant="$5"
  local scope="$6"
  local start_scale="$7"
  local final_scale="$8"
  local decay_steps="$9"
  local predictive_scale="${10}"
  shift 10
  run_capture "$name" \
    "$BIN" atlas-alt-bench \
    --mode "$benchmark" \
    --token-epochs "$epochs" \
    --repeats "$REPEATS" \
    "$@" \
    --gpu-enable 1 \
    --gpu-device "$GPU_DEVICE"
  append_epoch_summary "$benchmark" "$optimizer" "$variant" "$scope" "$start_scale" "$final_scale" "$decay_steps" "$predictive_scale" "$epochs" "$OUT_DIR/raw/${name}.log"
}

run_variant_accept() {
  local name="$1"
  local benchmark="$2"
  local optimizer="$3"
  local variant="$4"
  local scope="$5"
  local start_scale="$6"
  local final_scale="$7"
  local decay_steps="$8"
  local predictive_scale="$9"
  shift 9
  run_capture "$name" \
    "$BIN" atlas-alt-bench \
    --mode "$benchmark" \
    --repeats 10 \
    "$@" \
    --gpu-enable 1 \
    --gpu-device "$GPU_DEVICE"
  append_acceptance_summary "$benchmark" "$optimizer" "$variant" "$scope" "$start_scale" "$final_scale" "$decay_steps" "$predictive_scale" "$OUT_DIR/raw/${name}.log"
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
fi

read -r SMOKE_SCOPE SMOKE_START SMOKE_FINAL SMOKE_DECAY SMOKE_PRED <<< "$(parse_spec "${SPECS[0]}")"
SMOKE_TAG="$(spec_tag "$SMOKE_SCOPE" "$SMOKE_START" "$SMOKE_FINAL" "$SMOKE_DECAY" "$SMOKE_PRED")"
run_capture 06_smoke_corpus_large_echo \
  "$BIN" atlas-alt-bench \
  --mode "$BENCHMARK" \
  --token-epochs 1 \
  --repeats 1 \
  --variant echo \
  --atlas-echo-scope "$SMOKE_SCOPE" \
  --atlas-echo-geometry-scale "$SMOKE_START" \
  --atlas-echo-final-geometry-scale "$SMOKE_FINAL" \
  --atlas-echo-decay-steps "$SMOKE_DECAY" \
  --atlas-echo-cadence "$ECHO_CADENCE" \
  --atlas-echo-trust-scale "$ECHO_TRUST_SCALE" \
  --atlas-echo-predictive-scale "$SMOKE_PRED" \
  --atlas-echo-structural-scale "$ECHO_STRUCTURAL_SCALE" \
  --atlas-echo-structural-groups "$ECHO_STRUCTURAL_GROUPS" \
  --gpu-enable 1 \
  --gpu-device "$GPU_DEVICE"

for epochs in "${EPOCHS[@]}"; do
  run_variant_epoch "sweep_${BENCHMARK}_adamw_e${epochs}" \
    "$BENCHMARK" "$epochs" "adamw" "adamw" "-" "-" "-" "-" "-" \
    --variant adamw

  for spec in "${SPECS[@]}"; do
    read -r scope start_scale final_scale decay_steps predictive_scale <<< "$(parse_spec "$spec")"
    tag="$(spec_tag "$scope" "$start_scale" "$final_scale" "$decay_steps" "$predictive_scale")"
    run_variant_epoch "sweep_${BENCHMARK}_echo_${tag}_e${epochs}" \
      "$BENCHMARK" "$epochs" "echo" "$tag" "$scope" "$start_scale" "$final_scale" "$decay_steps" "$predictive_scale" \
      --variant echo \
      --atlas-echo-scope "$scope" \
      --atlas-echo-geometry-scale "$start_scale" \
      --atlas-echo-final-geometry-scale "$final_scale" \
      --atlas-echo-decay-steps "$decay_steps" \
      --atlas-echo-cadence "$ECHO_CADENCE" \
      --atlas-echo-trust-scale "$ECHO_TRUST_SCALE" \
      --atlas-echo-predictive-scale "$predictive_scale" \
      --atlas-echo-structural-scale "$ECHO_STRUCTURAL_SCALE" \
      --atlas-echo-structural-groups "$ECHO_STRUCTURAL_GROUPS"
  done
done

if [[ "$RUN_ACCEPTANCE" -eq 1 ]]; then
  run_variant_accept "accept_${BENCHMARK}_adamw" \
    "$BENCHMARK" "adamw" "adamw" "-" "-" "-" "-" "-" \
    --variant adamw

  for spec in "${SPECS[@]}"; do
    read -r scope start_scale final_scale decay_steps predictive_scale <<< "$(parse_spec "$spec")"
    tag="$(spec_tag "$scope" "$start_scale" "$final_scale" "$decay_steps" "$predictive_scale")"
    run_variant_accept "accept_${BENCHMARK}_echo_${tag}" \
      "$BENCHMARK" "echo" "$tag" "$scope" "$start_scale" "$final_scale" "$decay_steps" "$predictive_scale" \
      --variant echo \
      --atlas-echo-scope "$scope" \
      --atlas-echo-geometry-scale "$start_scale" \
      --atlas-echo-final-geometry-scale "$final_scale" \
      --atlas-echo-decay-steps "$decay_steps" \
      --atlas-echo-cadence "$ECHO_CADENCE" \
      --atlas-echo-trust-scale "$ECHO_TRUST_SCALE" \
      --atlas-echo-predictive-scale "$predictive_scale" \
      --atlas-echo-structural-scale "$ECHO_STRUCTURAL_SCALE" \
      --atlas-echo-structural-groups "$ECHO_STRUCTURAL_GROUPS"
  done

  generate_acceptance_rank
  generate_echo_acceptance_rank
fi

if printf '%s\n' "${EPOCHS[@]}" | rg -qx '4'; then
  generate_epoch_rank 4 "$E4_RANK_TSV"
  generate_echo_epoch_rank 4 "$ECHO_E4_RANK_TSV"
fi
generate_epoch_rank "$FINAL_EPOCH" "$FINAL_RANK_TSV"
generate_echo_epoch_rank "$FINAL_EPOCH" "$ECHO_FINAL_RANK_TSV"

cat > "$OUT_DIR/README.txt" <<EOF
Output directory: $OUT_DIR

Purpose:
- dedicated token-lm-corpus-large ECHO sweep against AdamW
- only tests cheap ECHO knobs still plausible for beating AdamW
- focuses on scope, final geometry scale, and small predictive scale

Files:
- run.log: command trace
- epoch_sweep_summary.tsv: one row per optimizer/spec/epoch point
- acceptance_summary.tsv: 10-repeat acceptance summary (unless --skip-acceptance)
- acceptance_rank_by_nll.tsv: all candidates ranked by TestNLL, then time
- echo_acceptance_rank_by_nll.tsv: ECHO-only acceptance ranking
- epoch4_rank_by_nll.tsv: all candidates ranked at epoch 4, then time
- echo_epoch4_rank_by_nll.tsv: ECHO-only epoch 4 ranking
- final_epoch_rank_by_nll.tsv: all candidates ranked at the highest configured epoch
- echo_final_epoch_rank_by_nll.tsv: ECHO-only ranking at the highest configured epoch
- raw/*.log: full raw benchmark outputs

Benchmark settings:
- benchmark: $BENCHMARK
- epochs: ${EPOCHS[*]}
- final epoch rank target: $FINAL_EPOCH
- repeats per sweep point: $REPEATS
- acceptance repeats: $(if [[ "$RUN_ACCEPTANCE" -eq 1 ]]; then echo 10; else echo skipped; fi)

Shared ECHO settings:
- cadence: $ECHO_CADENCE
- trust scale: $ECHO_TRUST_SCALE
- structural scale/groups: $ECHO_STRUCTURAL_SCALE / $ECHO_STRUCTURAL_GROUPS

ECHO specs:
$(printf '  - %s\n' "${SPECS[@]}")
EOF

echo "Saved results to: $OUT_DIR"
echo "Epoch sweep summary: $SUMMARY_TSV"
if [[ "$RUN_ACCEPTANCE" -eq 1 ]]; then
  echo "Acceptance summary: $ACCEPT_TSV"
  echo "Acceptance ranking: $ACCEPT_RANK_TSV"
  echo "ECHO acceptance ranking: $ECHO_ACCEPT_RANK_TSV"
fi
if [[ -f "$E4_RANK_TSV" ]]; then
  echo "Epoch-4 ranking: $E4_RANK_TSV"
  echo "ECHO epoch-4 ranking: $ECHO_E4_RANK_TSV"
fi
echo "Final epoch ranking: $FINAL_RANK_TSV"
echo "ECHO final epoch ranking: $ECHO_FINAL_RANK_TSV"
