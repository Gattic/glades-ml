#!/bin/bash
# Step 3: LM transfer test.
# Task: syntheticlm (existing) — Markov chain, V=64, T=256, predict one token after sequence.
# Compare GRP-RNN + LN vs LRU equivalents on a task other than A_5.
# Random baseline = log(64) ≈ 4.16 nats.

set -e
cd "$(dirname "$0")"

BIN=../ealrmn_gpu/ealrmn_gpu
OUT=results/n1v7_lm_transfer.jsonl
LOG=logs/n1v7_lm_transfer.txt
: > "$OUT"

echo "VESTA N1v7 LM transfer starting at $(date)" | tee -a "$LOG"

run_one () {
    local variant=$1 seed=$2
    shift 2
    local extra="$@"
    local desc="${variant}/syntheticlm V=64 T=256 seed=$seed ${extra}"
    echo "[$(date +%H:%M:%S)] BEGIN $desc" | tee -a "$LOG"
    local s=$(date +%s)
    # syntheticlm task w/ V=64 needs --lm-vocab=64. Default order=2.
    # Use curriculum: warmup at T=32 for 4000 steps to bootstrap.
    "$BIN" --mode=train \
        --model=grp_rnn --task=syntheticlm \
        --m=256 --T=256 --lm-vocab=64 --seed="$seed" --lr=5e-4 \
        --batch=32 --steps=15000 \
        --warmup=500 --grad-clip=1.0 \
        --eval-batch=128 --eval-every=2500 --print-every=15000 \
        --grp-layernorm=1 \
        $extra \
        --jsonl="$OUT" --tag="lm_${variant}" 2>&1 | tail -3 | tee -a "$LOG"
    local e=$(date +%s); echo "[$(date +%H:%M:%S)] END   $desc  wall=$((e-s))s" | tee -a "$LOG"
}

for seed in 0 1 2; do
    # Novel: interlocking K=m, input-dep angles, LN.
    run_one "grp_full" $seed
    # Strong baseline: input-dep-phase LRU + LN.
    run_one "grp_idp"  $seed --grp-disjoint-planes=1 --grp-K=128
    # Minimum-floor: fixed-phase LRU + LN.
    run_one "grp_lru"  $seed --grp-disjoint-planes=1 --grp-fixed-angles=1 --grp-K=128
done

echo "VESTA N1v7 LM transfer finished at $(date)" | tee -a "$LOG"

python3 - <<'PY' >> "$LOG" 2>&1 || true
import json, math
rows = [json.loads(l) for l in open('results/n1v7_lm_transfer.jsonl')]
final = {}
for r in rows:
    k = (r['tag'], r['seed'])
    if k not in final or r['step'] > final[k]['step']:
        final[k] = r
by_variant = {}
for (tag, seed), r in final.items():
    by_variant.setdefault(tag, []).append((seed, r['val_acc'], r['val_loss']))
print("\n=== LM transfer final ===")
print(f"random baseline (V=64): acc ≈ {1/64:.4f}, loss = log(64) = {math.log(64):.4f}")
for tag, runs in sorted(by_variant.items()):
    accs = [v for _,v,_ in runs]
    losses = [l for _,_,l in runs]
    print(f"{tag:15s}  mean_acc={sum(accs)/len(accs):.4f}  per_seed={[f'{v:.4f}' for s,v,_ in runs]}  mean_loss={sum(losses)/len(losses):.4f}")
PY
