#!/bin/bash
# T=256 with LN + 5-phase curriculum: T=8 -> T=32 -> T=64 -> T=128 -> T=256
# 2 seeds, 60K steps total. ~30 min each, ~60 min total.

set -e
cd "$(dirname "$0")"

BIN=../ealrmn_gpu/ealrmn_gpu
OUT=results/n1v6_t256.jsonl
LOG=logs/n1v6_t256.txt
: > "$OUT"

echo "VESTA N1v6 T=256 starting at $(date)" | tee -a "$LOG"

for seed in 0 1; do
    desc="T=256 LN+5phase seed=$seed"
    echo "[$(date +%H:%M:%S)] BEGIN $desc" | tee -a "$LOG"
    s=$(date +%s)
    "$BIN" --mode=train \
        --model=grp_rnn --task=a5_word \
        --m=256 --T=256 --seed="$seed" --lr=5e-4 \
        --batch=64 --steps=60000 \
        --warmup=200 --grad-clip=1.0 \
        --eval-batch=128 --eval-every=5000 --print-every=60000 \
        --grp-layernorm=1 \
        --curriculum-schedule=8:4000,32:10000,64:20000,128:35000 \
        --jsonl="$OUT" --tag="n1v6_t256_lncurr" 2>&1 | tail -4 | tee -a "$LOG"
    e=$(date +%s); echo "[$(date +%H:%M:%S)] END   $desc  wall=$((e-s))s" | tee -a "$LOG"
done

echo "VESTA N1v6 T=256 finished at $(date)" | tee -a "$LOG"

python3 - <<'PY' >> "$LOG" 2>&1 || true
import json
rows = [json.loads(l) for l in open('results/n1v6_t256.jsonl')]
final = {}
for r in rows:
    k = (r['tag'], r['seed'])
    if k not in final or r['step'] > final[k]['step']:
        final[k] = r
print("\n=== T=256 final ===")
for k, r in sorted(final.items()):
    print(f"{k}  val_acc={r['val_acc']:.4f}  val_loss={r['val_loss']:.4f}")
PY
