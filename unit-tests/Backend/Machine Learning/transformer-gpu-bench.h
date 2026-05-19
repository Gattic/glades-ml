#ifndef _UT_TRANSFORMER_GPU_BENCH
#define _UT_TRANSFORMER_GPU_BENCH

// Dedicated microbenchmark runner for GPU-backed transformer token-LM train/infer paths.
// Example:
//   ./build/glades-unit-tests transformer-gpu-bench --repeats 3 --epochs 1 --seq-len 64 --layers 4
void TransformerGpuBenchmark(int argc, char* argv[]);

#endif
