#!/bin/bash
# VESTA Claim N1 v2 — A_5 word recognition at short T where optimization succeeds.
#
# Pilot at T=8 and T=16 (where GRP-RNN learns) and T=32 (where it doesn't).
# 5 seeds per cell. Higher LR (1e-3), larger batch (64), 3000 steps.

set -e
cd "$(dirname "$0")"

BIN=../ealrmn_gpu/ealrmn_gpu
OUT=results/n1_a5_v2.jsonl
LOG=logs/n1_a5_v2.txt
: > "$OUT"

echo "VESTA N1 v2 starting at $(date)" | tee -a "$LOG"

run_one () {
    local variant=$1 T=$2 seed=$3
    shift 3
    local extra="$@"
    local desc="${variant}/a5_word m=256 T=${T} seed=${seed} ${extra}"
    echo "[$(date +%H:%M:%S)] BEGIN $desc" | tee -a "$LOG"
    local start=$(date +%s)
    "$BIN" --mode=train \
        --model=grp_rnn --task=a5_word \
        --m=256 --T="$T" --seed="$seed" --lr=1e-3 \
        --batch=64 --steps=3000 --H=8 \
        --warmup=200 --grad-clip=1.0 \
        --eval-batch=128 --eval-every=500 --print-every=1500 \
        $extra \
        --jsonl="$OUT" --tag="n1v2_${variant}_T${T}" 2>&1 | tail -3 | tee -a "$LOG"
    local end=$(date +%s)
    echo "[$(date +%H:%M:%S)] END   $desc  wall=$((end-start))s" | tee -a "$LOG"
}

for seed in 0 1 2 3 4; do
    for T in 8 16 32; do
        run_one "grp_full" $T $seed
        run_one "grp_lru"  $T $seed --grp-disjoint-planes=1 --grp-fixed-angles=1 --grp-K=128
    done
done

echo "VESTA N1 v2 finished at $(date)" | tee -a "$LOG"

python3 - <<'PY' >> "$LOG" 2>&1 || echo "(python aggregation skipped)" >> "$LOG"
import json
rows = [json.loads(l) for l in open('results/n1_a5_v2.jsonl')]
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
    print(f"{tag:25s}  mean_val_acc={sum(accs)/len(accs):.4f}  per_seed_acc={[f'{v:.4f}' for s,v,_ in runs]}  mean_val_loss={sum(losses)/len(losses):.4f}")
print("\n=== N1 v2 verdict ===")
for T in [8, 16, 32]:
    print(f"\n--- T={T} ---")
    for variant in ['grp_full', 'grp_lru']:
        tag = f"n1v2_{variant}_T{T}"
        if tag in by_variant:
            accs = [v for _,v,_ in by_variant[tag]]
            print(f"  {variant:12s}  mean_acc = {sum(accs)/len(accs):.4f}  per_seed = {[f'{v:.4f}' for v in accs]}")
    if f'n1v2_grp_full_T{T}' in by_variant and f'n1v2_grp_lru_T{T}' in by_variant:
        full = sum(v for _,v,_ in by_variant[f'n1v2_grp_full_T{T}']) / len(by_variant[f'n1v2_grp_full_T{T}'])
        lru = sum(v for _,v,_ in by_variant[f'n1v2_grp_lru_T{T}']) / len(by_variant[f'n1v2_grp_lru_T{T}'])
        gap = full - lru
        threshold_met = "PASS" if gap >= 0.05 else "FAIL"
        print(f"  gap (grp_full - grp_lru) = {gap:+.4f}  vs 0.05 threshold -> {threshold_met}")
PY
