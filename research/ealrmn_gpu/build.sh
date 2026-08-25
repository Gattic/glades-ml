#!/bin/bash
# Build script for EALRMN Phase-1 GPU prototype.
set -e
cd "$(dirname "$0")"

NVCC=${NVCC:-nvcc}

$NVCC -std=c++17 -O3 -arch=sm_89 -lineinfo \
    -Xcompiler "-Wno-unused-result -Wno-unused-variable -Wno-unused-but-set-variable" \
    -diag-suppress 20012,20014,20015,177 \
    main.cu \
    -lcublas -lcurand \
    -o ealrmn_gpu

echo "Built: $(pwd)/ealrmn_gpu"
ls -lh ealrmn_gpu
