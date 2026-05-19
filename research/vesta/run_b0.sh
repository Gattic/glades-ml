#!/bin/bash
# VESTA Claim B0 baseline replication
#
# Pre-registered claim from newmodel.txt:
#   "Linear recurrence + orthogonal init beats tanh RNN by >= 100x val_loss
#    on the needle task at T=2048 m=1024."
#
# Iso-param-iso-m design: both variants at m=1024, 3 seeds, 800 steps.
# Uses the existing research/ealrmn_gpu/ rnn model with --use-tanh={0,1}.
# Note: model_rnn.cuh initializes W_h via orthogonal Gram-Schmidt regardless
# of --use-tanh, so the linear variant gets orthogonal init "for free".

set -e
cd "$(dirname "$0")"

BIN=../ealrmn_gpu/ealrmn_gpu
OUT=results/b0.jsonl
LOG=logs/b0.txt
: > "$OUT"

if [ ! -x "$BIN" ]; then
    echo "ERROR: binary $BIN missing or not executable. Run ../ealrmn_gpu/build.sh first." | tee -a "$LOG"
    exit 1
fi

echo "VESTA B0 starting at $(date)" | tee -a "$LOG"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)" | tee -a "$LOG"

run_one () {
    local use_tanh=$1 seed=$2
    local variant=$([ "$use_tanh" = "1" ] && echo "tanh_rnn" || echo "linear_rnn")
    local desc="rnn(use_tanh=$use_tanh)/needle m=1024 T=2048 seed=$seed"
    echo "[$(date +%H:%M:%S)] BEGIN $desc" | tee -a "$LOG"
    local start=$(date +%s)
    "$BIN" --mode=train \
        --model=rnn --task=needle \
        --m=1024 --T=2048 --seed="$seed" --lr=1e-4 \
        --batch=4 --steps=800 --H=8 \
        --warmup=100 --grad-clip=1.0 \
        --eval-every=160 --print-every=80 \
        --use-tanh="$use_tanh" \
        --jsonl="$OUT" --tag="b0_${variant}" 2>&1 | tail -8 | tee -a "$LOG"
    local end=$(date +%s)
    echo "[$(date +%H:%M:%S)] END   $desc  wall=$((end-start))s" | tee -a "$LOG"
}

for seed in 0 1 2; do
    run_one 1 $seed   # tanh RNN
    run_one 0 $seed   # linear RNN (orthogonal init already default)
done

echo "VESTA B0 finished at $(date)" | tee -a "$LOG"
echo "Aggregating..." | tee -a "$LOG"

# Pull final-step val_loss per seed/variant
python3 - <<'PY' >> "$LOG" 2>&1 || echo "(python aggregation skipped)" >> "$LOG"
import json, math, statistics
rows = [json.loads(l) for l in open('results/b0.jsonl')]
final = {}
for r in rows:
    k = (r['tag'], r['seed'])
    if k not in final or r['step'] > final[k]['step']:
        final[k] = r
by_variant = {}
for (tag, seed), r in final.items():
    by_variant.setdefault(tag, []).append((seed, r['val_loss'], r['step']))
for tag, runs in sorted(by_variant.items()):
    losses = [v for _,v,_ in runs]
    geom = math.exp(sum(math.log(max(v,1e-30)) for v in losses)/len(losses))
    print(f"{tag:20s}  geo_mean_val_loss={geom:.4e}  per_seed={[(s,f'{v:.4e}') for s,v,_ in runs]}")
if 'b0_tanh_rnn' in by_variant and 'b0_linear_rnn' in by_variant:
    t_losses = [v for _,v,_ in by_variant['b0_tanh_rnn']]
    l_losses = [v for _,v,_ in by_variant['b0_linear_rnn']]
    t_gm = math.exp(sum(math.log(max(v,1e-30)) for v in t_losses)/len(t_losses))
    l_gm = math.exp(sum(math.log(max(v,1e-30)) for v in l_losses)/len(l_losses))
    ratio = t_gm/l_gm
    print(f"\n=== B0 RESULT ===")
    print(f"tanh   geo_mean = {t_gm:.4e}")
    print(f"linear geo_mean = {l_gm:.4e}")
    print(f"ratio  (tanh/linear) = {ratio:.1f}x")
    print(f"B0 PASS (>=100x): {ratio >= 100.0}")
PY
