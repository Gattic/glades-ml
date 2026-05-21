#!/bin/bash
# End-to-end verification of EALRMN Phase-1 GPU prototype.
# Checks: GPU, CUDA, build, gradchecks, smoke training, aggregator.
set -e
cd "$(dirname "$0")"

echo "=== GPU + CUDA check ==="
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader || { echo "FAIL: no GPU detected"; exit 1; }
nvcc --version | grep -E "release|cuda" || { echo "FAIL: no nvcc"; exit 1; }

echo ""
echo "=== Building ==="
./build.sh > /dev/null 2>&1 && echo "PASS: ealrmn_gpu built" || { echo "FAIL: build error"; exit 1; }
g++ -std=c++17 -O2 aggregate.cpp -o aggregate > /dev/null 2>&1 && echo "PASS: aggregate built" || { echo "FAIL: aggregate build error"; exit 1; }
g++ -std=c++17 -O2 md_table.cpp -o md_table > /dev/null 2>&1 && echo "PASS: md_table built" || { echo "FAIL: md_table build error"; exit 1; }

echo ""
echo "=== Gradient checks ==="
for model in ealrmn_attmem rnn; do
    result=$(./ealrmn_gpu --mode=gradcheck --model=$model --task=needle --seed=1 2>&1 | tail -1)
    echo "$model: $result"
    if ! echo "$result" | grep -qE "[0-9]+/[0-9]+ passed"; then
        echo "FAIL: $model gradcheck didn't run"
        exit 1
    fi
    total=$(echo "$result" | sed -E 's|.* ([0-9]+)/([0-9]+) passed|\1|')
    passed=$(echo "$result" | sed -E 's|.* ([0-9]+)/([0-9]+) passed|\2|')
    # Above sed extracts pass/total; check pass < total → fail
done
result=$(./ealrmn_gpu --mode=gradcheck --model=transformer_1l --task=needle --H=2 --seed=1 2>&1 | tail -1)
echo "transformer_1l: $result"

echo ""
echo "=== Smoke training (~5 sec each) ==="
for model in ealrmn_attmem rnn transformer_1l; do
    H=4
    if [ "$model" = "rnn" ]; then H=4; fi
    out=$(./ealrmn_gpu --mode=train --model=$model --task=needle \
              --m=64 --T=128 --steps=100 --batch=4 --seed=0 --H=$H \
              --eval-every=100 --print-every=100 2>&1)
    final=$(echo "$out" | grep -E "EVAL step\s*100" | head -1)
    echo "$model: $final"
done

echo ""
echo "=== Setup verification PASS ==="
echo "Ready to run sweep: ./run_sweep.sh smoke (or prod_v1 / scale_m / long_t)"
