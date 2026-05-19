#!/bin/bash
# EALRMN Phase-1 GPU production sweep.
#
# Usage:
#   ./run_sweep.sh smoke      # ~10 min smoke run
#   ./run_sweep.sh focused    # ~1 hour focused comparison
#   ./run_sweep.sh full       # ~12 hour full production sweep
#
# Output: appends JSONL to ./results/sweep_<tag>.jsonl

set -e
cd "$(dirname "$0")"

mkdir -p results
TAG=${1:-smoke}
OUT=results/sweep_${TAG}.jsonl
LOG=results/log_${TAG}.txt
: > "$OUT"   # truncate fresh

echo "Sweep '$TAG' starting at $(date), output: $OUT" | tee -a "$LOG"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)" | tee -a "$LOG"

run_one () {
  local model=$1 task=$2 m=$3 T=$4 seed=$5 batch=$6 steps=$7 H=$8
  # LR scales down at long T because BPTT gradient compounding is harder to optimize.
  # Empirically determined: lr=1e-4 OK at T=2048; lr=5e-5 needed at T≥4096 for EALRMN stability.
  local lr=${LR:-}
  if [ -z "$lr" ]; then
    if [ "$T" -le 2048 ]; then lr=1e-4
    else lr=5e-5
    fi
  fi
  local gc=${GC:-}
  if [ -z "$gc" ]; then
    if [ "$T" -le 2048 ]; then gc=1.0
    else gc=0.5
    fi
  fi
  local extra=${EXTRA:-}
  local tag_suffix=${TAG_SUFFIX:-}
  local effective_tag="$TAG"
  if [ -n "$tag_suffix" ]; then effective_tag="${TAG}_${tag_suffix}"; fi
  local desc="${model}/${task} m=${m} T=${T} seed=${seed} steps=${steps} lr=${lr} gc=${gc} ${extra:+extra=$extra}"
  echo "[$(date +%H:%M:%S)] BEGIN $desc" | tee -a "$LOG"
  local start=$(date +%s)
  ./ealrmn_gpu --mode=train \
      --model="$model" --task="$task" \
      --m="$m" --T="$T" --seed="$seed" --lr="$lr" \
      --batch="$batch" --steps="$steps" --H="$H" \
      --warmup=$((steps / 8)) --grad-clip="$gc" \
      --eval-every=$((steps / 5)) --print-every=$((steps / 10)) \
      --jsonl="$OUT" --tag="$effective_tag" $extra 2>&1 | tail -8 | tee -a "$LOG"
  local end=$(date +%s)
  echo "[$(date +%H:%M:%S)] END   $desc  wall=$((end-start))s" | tee -a "$LOG"
}

case "$TAG" in
prod_v1)
  # ~60 min: full production-scale comparison at m=1024 across T ∈ {2048, 4096, 16384}.
  # Phase A: m=1024, T=2048, EALRMN+RNN+Transformer, 3 seeds, 800 steps  (~25 min)
  for seed in 0 1 2; do
    run_one ealrmn_attmem  needle 1024 2048 $seed 4 800 8
    run_one rnn            needle 1024 2048 $seed 4 800 8
    run_one transformer_1l needle 1024 2048 $seed 4 800 8
  done
  # Phase B: m=1024, T=4096, EALRMN+RNN (Transformer OOMs above T=4096 at m=1024), 3 seeds, 500 steps  (~20 min)
  for seed in 0 1 2; do
    run_one ealrmn_attmem  needle 1024 4096 $seed 2 500 8
    run_one rnn            needle 1024 4096 $seed 2 500 8
  done
  # Phase C: m=1024, T=16384, EALRMN+RNN, 2 seeds, 200 steps  (~20 min)
  for seed in 0 1; do
    run_one ealrmn_attmem  needle 1024 16384 $seed 1 200 8
    run_one rnn            needle 1024 16384 $seed 1 200 8
  done
  ;;
prod_bc)
  # Phase B + Phase C only — when Phase A is already in the JSONL.
  # Phase B: m=1024, T=4096, EALRMN+RNN, 3 seeds, 500 steps
  for seed in 0 1 2; do
    run_one ealrmn_attmem  needle 1024 4096 $seed 2 500 8
    run_one rnn            needle 1024 4096 $seed 2 500 8
  done
  # Phase C: m=1024, T=16384, EALRMN+RNN, 2 seeds, 300 steps
  for seed in 0 1; do
    run_one ealrmn_attmem  needle 1024 16384 $seed 1 300 8
    run_one rnn            needle 1024 16384 $seed 1 300 8
  done
  ;;
scale_m)
  # ~40 min: m-scaling test at T=2048. Tests whether the EALRMN-vs-RNN gap is m-dependent.
  for m in 256 512 1024; do
    for seed in 0 1 2; do
      run_one ealrmn_attmem  needle $m 2048 $seed 8 800 8
      run_one rnn            needle $m 2048 $seed 8 800 8
    done
  done
  ;;
long_t)
  # ~60 min: deeper test at T=16384 with more seeds and longer training.
  for seed in 0 1 2; do
    run_one ealrmn_attmem  needle 1024 16384 $seed 1 500 8
    run_one rnn            needle 1024 16384 $seed 1 500 8
  done
  ;;
tasks_v1)
  # ~30 min: secondary tasks (HMM, syntheticlm) at m=1024 T=2048 to check task generality
  for seed in 0 1 2; do
    run_one ealrmn_attmem  hmm 1024 2048 $seed 4 800 8
    run_one rnn            hmm 1024 2048 $seed 4 800 8
  done
  ;;
iso_params)
  # ~30 min: iso-param-budget comparison.
  # EALRMN at m=1024 has ~3.26M params; RNN at m=1024 has ~2.20M.
  # To equalize: RNN at m=1448 has ~4.2M params (closer match).
  # Test: does the EALRMN advantage survive vs same-param RNN?
  for seed in 0 1 2; do
    run_one ealrmn_attmem  needle 1024 2048 $seed 4 800 8
    run_one rnn            needle 1448 2048 $seed 4 800 8
  done
  ;;
iso_params_more)
  # ~35 min: 7 more seeds (3-9) for iso-param comparison, on top of existing iso_params seeds (0-2).
  # Total target: 10 seeds per cell.
  for seed in 3 4 5 6 7 8 9; do
    run_one ealrmn_attmem  needle 1024 2048 $seed 4 800 8
    run_one rnn            needle 1448 2048 $seed 4 800 8
  done
  ;;
ablations)
  # ~55 min: 4 ablations × 5 seeds at T=2048 to isolate the EALRMN training-robustness cause.
  # Each ablation variant uses a sub-tag baked into the --tag for grouping.
  for seed in 0 1 2 3 4; do
    EXTRA="--init-K=xavier"     TAG_SUFFIX="ealrmn_xavierK" run_one ealrmn_attmem needle 1024 2048 $seed 4 800 8
    EXTRA="--readout=s_only"    TAG_SUFFIX="ealrmn_no_readout" run_one ealrmn_attmem needle 1024 2048 $seed 4 800 8
    EXTRA="--init-Wh=xavier"    TAG_SUFFIX="rnn_xavierWh" run_one rnn needle 1448 2048 $seed 4 800 8
    EXTRA="--use-tanh=0"        TAG_SUFFIX="rnn_linear" run_one rnn needle 1448 2048 $seed 4 800 8
  done
  ;;
smoke)
  # 5 min: validate sweep pipeline at small scale
  for seed in 0 1; do
    run_one ealrmn_attmem needle 128 256 $seed 8 200 4
    run_one rnn needle 128 256 $seed 8 200 4
  done
  ;;
focused)
  # ~1 hour: EALRMN vs RNN vs Transformer at T=2048, m=512, 3 seeds, needle
  for seed in 0 1 2; do
    run_one ealrmn_attmem  needle 512 2048 $seed 8 1500 4
    run_one rnn            needle 512 2048 $seed 8 1500 4
    run_one transformer_1l needle 512 2048 $seed 4 1500 8
  done
  # plus EALRMN vs RNN at T=4096, m=512
  for seed in 0 1; do
    run_one ealrmn_attmem  needle 512 4096 $seed 4 1000 4
    run_one rnn            needle 512 4096 $seed 4 1000 4
  done
  ;;
scale)
  # ~3 hours: scale-up test at m=1024, T={2048, 4096}, EALRMN vs RNN
  for seed in 0 1 2; do
    run_one ealrmn_attmem  needle 1024 2048 $seed 4 1500 8
    run_one rnn            needle 1024 2048 $seed 4 1500 8
  done
  for seed in 0 1 2; do
    run_one ealrmn_attmem  needle 1024 4096 $seed 2 1200 8
    run_one rnn            needle 1024 4096 $seed 2 1200 8
  done
  ;;
long)
  # ~4 hours: T=16384 production-scale at m=1024, 2 seeds
  for seed in 0 1; do
    run_one ealrmn_attmem  needle 1024 16384 $seed 1 600 8
    run_one rnn            needle 1024 16384 $seed 1 600 8
  done
  ;;
tasks)
  # ~1 hour: test on HMM and syntheticlm at m=512 T=2048
  for seed in 0 1 2; do
    run_one ealrmn_attmem  hmm 512 2048 $seed 8 1500 4
    run_one rnn            hmm 512 2048 $seed 8 1500 4
  done
  for seed in 0 1; do
    run_one ealrmn_attmem  syntheticlm 512 2048 $seed 8 1500 4
    run_one rnn            syntheticlm 512 2048 $seed 8 1500 4
  done
  ;;
full)
  echo "Running ALL sweep stages: focused + scale + long + tasks"
  $0 focused && $0 scale && $0 long && $0 tasks
  ;;
*)
  echo "Unknown sweep tag: $TAG"
  echo "Options: smoke focused scale long tasks full"
  exit 1
  ;;
esac

echo "Sweep '$TAG' finished at $(date)" | tee -a "$LOG"
echo "Output: $OUT" | tee -a "$LOG"
ls -l "$OUT" | tee -a "$LOG"
