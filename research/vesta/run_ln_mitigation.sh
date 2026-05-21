#!/bin/bash
# VESTA N1 v4 — test LayerNorm + curriculum mitigations at T=32.
#
# Cells (5 seeds each):
#  - LN+curriculum at T=32, 15K steps   (the breakthrough config)
#  - LN only at T=32, 20K steps         (does LN alone fix it?)
#  - curriculum only at T=32, 15K steps (confirm baseline)

set -e
cd "$(dirname "$0")"

BIN=../ealrmn_gpu/ealrmn_gpu
OUT=results/n1v4_mitigations.jsonl
LOG=logs/n1v4_mitigations.txt
: > "$OUT"

echo "VESTA N1v4 mitigations at T=32 starting at $(date)" | tee -a "$LOG"

run_one () {
    local variant=$1 steps=$2 seed=$3
    shift 3
    local extra="$@"
    local desc="${variant}/a5_word T=32 seed=$seed steps=$steps ${extra}"
    echo "[$(date +%H:%M:%S)] BEGIN $desc" | tee -a "$LOG"
    local start=$(date +%s)
    "$BIN" --mode=train \
        --model=grp_rnn --task=a5_word \
        --m=256 --T=32 --seed="$seed" --lr=5e-4 \
        --batch=64 --steps="$steps" \
        --warmup=200 --grad-clip=1.0 \
        --eval-batch=128 --eval-every=2500 --print-every="$steps" \
        $extra \
        --jsonl="$OUT" --tag="n1v4_${variant}" 2>&1 | tail -3 | tee -a "$LOG"
    local end=$(date +%s)
    echo "[$(date +%H:%M:%S)] END   $desc  wall=$((end-start))s" | tee -a "$LOG"
}

for seed in 0 1 2 3 4; do
    # The breakthrough: LN + curriculum.
    run_one "ln_curr"   15000 $seed --grp-layernorm=1 --curriculum-T=8 --curriculum-steps=4000
    # Curriculum only (baseline mitigation).
    run_one "curr_only" 15000 $seed --curriculum-T=8 --curriculum-steps=4000
    # LN only (does LN alone help with no curriculum?).
    run_one "ln_only"   20000 $seed --grp-layernorm=1
done

echo "VESTA N1v4 finished at $(date)" | tee -a "$LOG"

python3 - <<'PY' >> "$LOG" 2>&1 || echo "(python aggregation skipped)" >> "$LOG"
import json
rows = [json.loads(l) for l in open('results/n1v4_mitigations.jsonl')]
final = {}
for r in rows:
    k = (r['tag'], r['seed'])
    if k not in final or r['step'] > final[k]['step']:
        final[k] = r
by_variant = {}
for (tag, seed), r in final.items():
    by_variant.setdefault(tag, []).append((seed, r['val_acc'], r['val_loss']))
print("\n=== N1v4 mitigation final results ===")
for tag, runs in sorted(by_variant.items()):
    accs = [v for _,v,_ in runs]
    losses = [l for _,_,l in runs]
    print(f"{tag:20s}  mean_acc={sum(accs)/len(accs):.4f}  per_seed={[f'{v:.4f}' for s,v,_ in runs]}  mean_loss={sum(losses)/len(losses):.4f}")
PY
