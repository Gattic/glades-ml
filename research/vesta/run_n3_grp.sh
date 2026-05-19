#!/bin/bash
# VESTA Claim N3 — GRP-RNN internal B0 ablation.
#
# Compare GRP-RNN with --grp-tanh-state=0 (linear, by construction orthogonal
# at init) vs --grp-tanh-state=1 (tanh after rotation; strawman that recreates
# the EALRMN Phase-1 tanh-RNN failure mode).
#
# Config: m=1024, T=2048, K=128 (small for throughput), 3 seeds, 800 steps.
# K=128 is enough to show the linear-vs-tanh-state ratio; the K=m expressivity
# claim is tested separately under N1.

set -e
cd "$(dirname "$0")"

BIN=../ealrmn_gpu/ealrmn_gpu
OUT=results/n3_grp.jsonl
LOG=logs/n3_grp.txt
: > "$OUT"

echo "VESTA N3 starting at $(date)" | tee -a "$LOG"

run_one () {
    local tanh_state=$1 seed=$2
    local variant=$([ "$tanh_state" = "1" ] && echo "tanh_state" || echo "linear")
    local desc="grp_rnn(tanh_state=$tanh_state)/needle m=1024 T=2048 K=128 seed=$seed"
    echo "[$(date +%H:%M:%S)] BEGIN $desc" | tee -a "$LOG"
    local start=$(date +%s)
    "$BIN" --mode=train \
        --model=grp_rnn --task=needle \
        --m=1024 --T=2048 --seed="$seed" --lr=1e-4 \
        --batch=4 --steps=800 --H=8 \
        --warmup=100 --grad-clip=1.0 \
        --eval-every=160 --print-every=160 \
        --grp-K=128 --grp-stride=3 \
        --grp-tanh-state="$tanh_state" \
        --jsonl="$OUT" --tag="n3_${variant}" 2>&1 | tail -4 | tee -a "$LOG"
    local end=$(date +%s)
    echo "[$(date +%H:%M:%S)] END   $desc  wall=$((end-start))s" | tee -a "$LOG"
}

for seed in 0 1 2; do
    run_one 0 $seed   # linear (default)
    run_one 1 $seed   # tanh-state (strawman)
done

echo "VESTA N3 finished at $(date)" | tee -a "$LOG"

python3 - <<'PY' >> "$LOG" 2>&1 || echo "(python aggregation skipped)" >> "$LOG"
import json, math
rows = [json.loads(l) for l in open('results/n3_grp.jsonl')]
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
    print(f"{tag:25s}  geo_mean={geom:.4e}  per_seed={[(s,f'{v:.4e}') for s,v,_ in runs]}")
if 'n3_linear' in by_variant and 'n3_tanh_state' in by_variant:
    l_losses = [v for _,v,_ in by_variant['n3_linear']]
    t_losses = [v for _,v,_ in by_variant['n3_tanh_state']]
    l_gm = math.exp(sum(math.log(max(v,1e-30)) for v in l_losses)/len(l_losses))
    t_gm = math.exp(sum(math.log(max(v,1e-30)) for v in t_losses)/len(t_losses))
    print(f"\n=== N3 RESULT ===")
    print(f"tanh_state   geo_mean = {t_gm:.4e}")
    print(f"linear       geo_mean = {l_gm:.4e}")
    print(f"ratio (tanh/linear) = {t_gm/l_gm:.1f}x")
PY
