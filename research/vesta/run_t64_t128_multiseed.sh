#!/bin/bash
# Multi-seed confirmation of LN + curriculum at T=64 and T=128.

set -e
cd "$(dirname "$0")"

BIN=../ealrmn_gpu/ealrmn_gpu
OUT=results/n1v5_t64_t128.jsonl
LOG=logs/n1v5_t64_t128.txt
: > "$OUT"

echo "VESTA N1v5 starting at $(date)" | tee -a "$LOG"

# T=64 with LN + 3-phase curriculum, 3 seeds
for seed in 0 1 2; do
    desc="T=64 LN+3phase seed=$seed"
    echo "[$(date +%H:%M:%S)] BEGIN $desc" | tee -a "$LOG"
    s=$(date +%s)
    "$BIN" --mode=train \
        --model=grp_rnn --task=a5_word \
        --m=256 --T=64 --seed="$seed" --lr=5e-4 \
        --batch=64 --steps=30000 \
        --warmup=200 --grad-clip=1.0 \
        --eval-batch=128 --eval-every=5000 --print-every=30000 \
        --grp-layernorm=1 --curriculum-schedule=8:4000,32:12000 \
        --jsonl="$OUT" --tag="n1v5_t64_lncurr" 2>&1 | tail -3 | tee -a "$LOG"
    e=$(date +%s); echo "[$(date +%H:%M:%S)] END   $desc  wall=$((e-s))s" | tee -a "$LOG"
done

# T=128 with LN + 4-phase curriculum, 2 seeds
for seed in 1 2; do
    desc="T=128 LN+4phase seed=$seed"
    echo "[$(date +%H:%M:%S)] BEGIN $desc" | tee -a "$LOG"
    s=$(date +%s)
    "$BIN" --mode=train \
        --model=grp_rnn --task=a5_word \
        --m=256 --T=128 --seed="$seed" --lr=5e-4 \
        --batch=64 --steps=40000 \
        --warmup=200 --grad-clip=1.0 \
        --eval-batch=128 --eval-every=5000 --print-every=40000 \
        --grp-layernorm=1 --curriculum-schedule=8:4000,32:10000,64:20000 \
        --jsonl="$OUT" --tag="n1v5_t128_lncurr" 2>&1 | tail -3 | tee -a "$LOG"
    e=$(date +%s); echo "[$(date +%H:%M:%S)] END   $desc  wall=$((e-s))s" | tee -a "$LOG"
done

echo "VESTA N1v5 finished at $(date)" | tee -a "$LOG"

python3 - <<'PY' >> "$LOG" 2>&1 || true
import json
rows = [json.loads(l) for l in open('results/n1v5_t64_t128.jsonl')]
final = {}
for r in rows:
    k = (r['tag'], r['seed'])
    if k not in final or r['step'] > final[k]['step']:
        final[k] = r
by_variant = {}
for (tag, seed), r in final.items():
    by_variant.setdefault(tag, []).append((seed, r['val_acc'], r['val_loss']))
print("\n=== N1v5 final results ===")
for tag, runs in sorted(by_variant.items()):
    accs = [v for _,v,_ in runs]
    losses = [l for _,_,l in runs]
    print(f"{tag:30s}  mean_acc={sum(accs)/len(accs):.4f}  per_seed={[f'{v:.4f}' for s,v,_ in runs]}  mean_loss={sum(losses)/len(losses):.4f}")
PY
