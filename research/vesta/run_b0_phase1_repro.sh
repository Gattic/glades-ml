#!/bin/bash
# VESTA B0 Phase-1-config rerun — replicates the headline 4400x ratio from
# research/EALRMN_PHASE1_GPU_RESULTS.md ablation cell `ablations_rnn_linear`.
#
# linear RNN: m=1448 (param-matched to EALRMN ~4.3M), 5 seeds
# tanh RNN:   m=1024 (the standard cell, ~2.2M), 5 seeds
# Both at T=2048, lr=1e-4, 800 steps.
#
# Reuses `research/ealrmn_gpu/ealrmn_gpu` binary unchanged.

set -e
cd "$(dirname "$0")"

BIN=../ealrmn_gpu/ealrmn_gpu
OUT=results/b0_phase1_repro.jsonl
LOG=logs/b0_phase1_repro.txt
: > "$OUT"

echo "VESTA B0 Phase-1-repro starting at $(date)" | tee -a "$LOG"

run_one () {
    local m=$1 use_tanh=$2 seed=$3
    local variant=$([ "$use_tanh" = "1" ] && echo "tanh_rnn_m${m}" || echo "linear_rnn_m${m}")
    local desc="rnn(m=$m use_tanh=$use_tanh)/needle T=2048 seed=$seed"
    echo "[$(date +%H:%M:%S)] BEGIN $desc" | tee -a "$LOG"
    local start=$(date +%s)
    "$BIN" --mode=train \
        --model=rnn --task=needle \
        --m="$m" --T=2048 --seed="$seed" --lr=1e-4 \
        --batch=4 --steps=800 --H=8 \
        --warmup=100 --grad-clip=1.0 \
        --eval-every=160 --print-every=160 \
        --use-tanh="$use_tanh" \
        --jsonl="$OUT" --tag="b0_p1_${variant}" 2>&1 | tail -4 | tee -a "$LOG"
    local end=$(date +%s)
    echo "[$(date +%H:%M:%S)] END   $desc  wall=$((end-start))s" | tee -a "$LOG"
}

# 5 seeds each cell, linear at m=1448, tanh at m=1024
for seed in 0 1 2 3 4; do
    run_one 1024 1 $seed  # tanh m=1024
done
for seed in 0 1 2 3 4; do
    run_one 1448 0 $seed  # linear m=1448
done

echo "B0 Phase-1-repro finished at $(date)" | tee -a "$LOG"
echo "Aggregating..." | tee -a "$LOG"

python3 - <<'PY' >> "$LOG" 2>&1 || echo "(python aggregation skipped)" >> "$LOG"
import json, math
rows = [json.loads(l) for l in open('results/b0_phase1_repro.jsonl')]
final = {}
for r in rows:
    k = (r['tag'], r['seed'])
    if k not in final or r['step'] > final[k]['step']:
        final[k] = r
by_variant = {}
for (tag, seed), r in final.items():
    by_variant.setdefault(tag, []).append((seed, r['val_loss'], r['step']))
print("\n=== Per-cell ===")
for tag, runs in sorted(by_variant.items()):
    losses = [v for _,v,_ in runs]
    geom = math.exp(sum(math.log(max(v,1e-30)) for v in losses)/len(losses))
    print(f"{tag:30s}  geo_mean={geom:.4e}  per_seed={[(s,f'{v:.4e}') for s,v,_ in runs]}")
linear_key = next((k for k in by_variant if 'linear' in k), None)
tanh_key = next((k for k in by_variant if 'tanh' in k), None)
if linear_key and tanh_key:
    l_losses = [v for _,v,_ in by_variant[linear_key]]
    t_losses = [v for _,v,_ in by_variant[tanh_key]]
    l_gm = math.exp(sum(math.log(max(v,1e-30)) for v in l_losses)/len(l_losses))
    t_gm = math.exp(sum(math.log(max(v,1e-30)) for v in t_losses)/len(t_losses))
    print(f"\n=== Phase-1-repro RESULT ===")
    print(f"tanh m=1024   geo_mean = {t_gm:.4e}")
    print(f"linear m=1448 geo_mean = {l_gm:.4e}")
    print(f"ratio (tanh/linear) = {t_gm/l_gm:.1f}x")
    print(f"Phase-1 reference: 4425x. Within order of magnitude: {abs(math.log10(t_gm/l_gm) - math.log10(4425)) < 1}")
PY
