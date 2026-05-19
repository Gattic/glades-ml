#!/bin/bash
# VESTA Claim N1 — A_5 word-problem recognition.
#
# Tests whether the K>1 Givens-product lifting gives an empirical advantage
# over the LRU-equivalent (disjoint planes + fixed angles, diagonal SSM).
#
# A_5 is the simplest non-solvable group; by Merrill, Petty, Sabharwal 2024,
# diagonal SSMs provably cannot recognize A_5 word-problems in O(1) depth.
# GRP-RNN with K interlocking Givens has H_K = SO(m) and should be able to.
#
# Models (all in glades-ml/research/ealrmn_gpu with our extensions):
#   - grp_rnn_full:       interlocking K=m, stride=3, input-dep angles. NOVEL CELL.
#   - grp_rnn_lru:        --grp-disjoint-planes --grp-fixed-angles. LRU baseline.
#   - grp_rnn_tanh:       --grp-tanh-state=1. Strawman (recapitulates EALRMN tanh-RNN).
#
# Config: m=256, T in {64, 256}, 5000 steps, 3 seeds per cell.
# T=1024 deferred unless T=256 results are clear.

set -e
cd "$(dirname "$0")"

BIN=../ealrmn_gpu/ealrmn_gpu
OUT=results/n1_a5.jsonl
LOG=logs/n1_a5.txt
: > "$OUT"

echo "VESTA N1 (A_5 word recognition) starting at $(date)" | tee -a "$LOG"

run_one () {
    local variant=$1 T=$2 seed=$3
    shift 3
    local extra="$@"
    local desc="${variant}/a5_word m=256 T=${T} seed=${seed} ${extra}"
    echo "[$(date +%H:%M:%S)] BEGIN $desc" | tee -a "$LOG"
    local start=$(date +%s)
    "$BIN" --mode=train \
        --model=grp_rnn --task=a5_word \
        --m=256 --T="$T" --seed="$seed" --lr=3e-4 \
        --batch=32 --steps=5000 --H=8 \
        --warmup=400 --grad-clip=1.0 \
        --eval-batch=64 --eval-every=500 --print-every=500 \
        $extra \
        --jsonl="$OUT" --tag="n1_${variant}_T${T}" 2>&1 | tail -3 | tee -a "$LOG"
    local end=$(date +%s)
    echo "[$(date +%H:%M:%S)] END   $desc  wall=$((end-start))s" | tee -a "$LOG"
}

for seed in 0 1 2; do
    # T=64 only — should be decisive: GRP-RNN solves, LRU+tanh fail.
    # Add T=256 later if T=64 doesn't discriminate.
    T=64
    # Novel cell: interlocking K=m, stride 3, input-dependent angles, no tanh.
    run_one "grp_full" $T $seed
    # LRU equivalent: disjoint planes (2k, 2k+1), no input dep angles, K=m/2.
    run_one "grp_lru"  $T $seed --grp-disjoint-planes=1 --grp-fixed-angles=1 --grp-K=128
    # Strawman: tanh-state. Expected to fail.
    run_one "grp_tanh" $T $seed --grp-tanh-state=1
done

echo "VESTA N1 finished at $(date)" | tee -a "$LOG"

python3 - <<'PY' >> "$LOG" 2>&1 || echo "(python aggregation skipped)" >> "$LOG"
import json
rows = [json.loads(l) for l in open('results/n1_a5.jsonl')]
final = {}
for r in rows:
    k = (r['tag'], r['seed'])
    if k not in final or r['step'] > final[k]['step']:
        final[k] = r
by_variant = {}
for (tag, seed), r in final.items():
    by_variant.setdefault(tag, []).append((seed, r['val_acc'], r['val_loss']))
print("\n=== Per-cell ===")
for tag, runs in sorted(by_variant.items()):
    accs = [v for _,v,_ in runs]
    losses = [l for _,_,l in runs]
    print(f"{tag:25s}  mean_val_acc={sum(accs)/len(accs):.4f}  per_seed_acc={[(s,f'{v:.4f}') for s,v,_ in runs]}  mean_val_loss={sum(losses)/len(losses):.4f}")
print("\n=== N1 verdict (mean acc across seeds) ===")
for T in [64, 256]:
    have_any = False
    for variant in ['grp_full', 'grp_lru', 'grp_tanh']:
        tag = f"n1_{variant}_T{T}"
        if tag in by_variant:
            have_any = True
            break
    if not have_any:
        continue
    print(f"\n--- T={T} ---")
    for variant in ['grp_full', 'grp_lru', 'grp_tanh']:
        tag = f"n1_{variant}_T{T}"
        if tag in by_variant:
            accs = [v for _,v,_ in by_variant[tag]]
            print(f"  {variant:12s}  acc = {sum(accs)/len(accs):.4f}  per_seed={[f'{v:.4f}' for v in accs]}")
PY
