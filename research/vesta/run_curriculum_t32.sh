#!/bin/bash
# Test the curriculum F2 mitigation at T=32.
# Phase 1: train at T=8 for 4000 steps (solves it).
# Phase 2: continue training at T=32 for 26000 more steps.
# Compare to no-curriculum baseline (already known to fail at random).

set -e
cd "$(dirname "$0")"

BIN=../ealrmn_gpu/ealrmn_gpu
OUT=results/n1v3_curriculum.jsonl
LOG=logs/n1v3_curriculum.txt
: > "$OUT"

echo "VESTA N1v3 curriculum at T=32 starting at $(date)" | tee -a "$LOG"

run_one () {
    local variant=$1 seed=$2
    shift 2
    local extra="$@"
    local desc="${variant}/a5_word T=32 seed=$seed ${extra}"
    echo "[$(date +%H:%M:%S)] BEGIN $desc" | tee -a "$LOG"
    local start=$(date +%s)
    "$BIN" --mode=train \
        --model=grp_rnn --task=a5_word \
        --m=256 --T=32 --seed="$seed" --lr=5e-4 \
        --batch=64 --steps=30000 \
        --warmup=200 --grad-clip=1.0 \
        --eval-batch=128 --eval-every=5000 --print-every=15000 \
        $extra \
        --jsonl="$OUT" --tag="n1v3_${variant}" 2>&1 | tail -3 | tee -a "$LOG"
    local end=$(date +%s)
    echo "[$(date +%H:%M:%S)] END   $desc  wall=$((end-start))s" | tee -a "$LOG"
}

for seed in 0 1 2; do
    run_one "no_curriculum_T32" $seed
    run_one "curriculum_T8_T32" $seed --curriculum-T=8 --curriculum-steps=4000
done

echo "VESTA N1v3 curriculum finished at $(date)" | tee -a "$LOG"

python3 - <<'PY' >> "$LOG" 2>&1 || echo "(python aggregation skipped)" >> "$LOG"
import json
rows = [json.loads(l) for l in open('results/n1v3_curriculum.jsonl')]
final = {}
for r in rows:
    k = (r['tag'], r['seed'])
    if k not in final or r['step'] > final[k]['step']:
        final[k] = r
by_variant = {}
for (tag, seed), r in final.items():
    by_variant.setdefault(tag, []).append((seed, r['val_acc'], r['val_loss']))
print("\n=== N1v3 final result @ step 30000 ===")
for tag, runs in sorted(by_variant.items()):
    accs = [v for _,v,_ in runs]
    print(f"{tag:30s}  mean_acc={sum(accs)/len(accs):.4f}  per_seed={[f'{v:.4f}' for s,v,_ in runs]}")
PY
