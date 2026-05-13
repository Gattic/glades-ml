// FP8 (E4M3 / E5M2) matmul wrappers via cuBLASLt (paradigm #50 HELIUM).
//
// FP8 tensor cores on Ada (sm_8.9) and Hopper (sm_9.0) double BF16 throughput
// at the cost of dynamic range — E4M3 max-finite is 448, so per-tensor scales
// are mandatory.  The wrappers below take FP32-valued inputs, cast to FP8 with
// caller-supplied scales, run cuBLASLt FP8 matmul, and write FP32 output
// (cuBLASLt applies the inverse scale internally so the caller sees an FP32
// result with the same numerical meaning as a BF16 matmul).
//
// Calibration model: the caller maintains a small (1,) device tensor per
// scale slot, initialised via `fp8_calibrate_amax_e4m3` over a warmup batch
// (or just `cudaMemset` to the typical activation scale 1/127 for safety).
// `sgemm_rowmajor_fp8_e4m3` rejects inputs with NaN/Inf scales.
//
// Status: scaffolding (commit 2026-05-13).  Wraps cuBLASLt's
// CUBLAS_COMPUTE_32F + CUDA_R_8F_E4M3 path.  Calibration support is
// minimal (amax + 448 / amax).  Production wiring into the CHIRON shear
// will follow once a per-tensor scale tracker is added to ChironParams.
#pragma once

#ifdef GLADES_HAVE_CUDA

#include <cstddef>

namespace glades {
namespace gpu {

// Lazily create the cublasLt handle bound to `computeStream()`.  Returns
// false if cuBLASLt is unavailable or the device lacks FP8 tensor cores
// (sm_8.9+ on Ada, sm_9.0+ on Hopper).  Cached on first success.
bool fp8_init();

// Returns true if the current device supports FP8 tensor cores.  Safe to
// call before `fp8_init()`.
bool fp8_supported();

// Compute max|x| over a device array (FP32) and write `448.0f / amax` into
// `*d_scale` (clamped to a safe range).  Used to set the per-tensor scale
// for an E4M3 GEMM input.  Returns false on CUDA error.
bool fp8_calibrate_amax_e4m3(const float* d_x, size_t n, float* d_scale);

// FP8 row-major GEMM: C = alpha · op(A) · op(B) + beta · C.
//
//   A, B are FP32 in HBM but the kernel internally casts to E4M3 using the
//   supplied per-tensor scales (d_scaleA, d_scaleB).  C is written as FP32.
//   The cast kernel rounds A/scaleA to E4M3 (max 448) and similarly for B.
//
//   M, N, K: matmul dims (row-major: A is [M, K], B is [K, N], C is [M, N])
//   lda, ldb, ldc: row strides (typically K, N, N)
//   d_scaleA, d_scaleB: device pointers to one FP32 value each
//
// Reuses cuBLASLt's CUBLAS_COMPUTE_32F + CUDA_R_8F_E4M3 path.  Output is
// FP32; the GEMM internally applies (1/scaleA)·(1/scaleB) so callers see
// the correct unscaled result.
bool sgemm_rowmajor_fp8_e4m3(int M, int N, int K,
                              float alpha,
                              const float* A, int lda,
                              const float* B, int ldb,
                              float beta,
                              float* C, int ldc,
                              const float* d_scaleA,
                              const float* d_scaleB);

} // namespace gpu
} // namespace glades

#else // !GLADES_HAVE_CUDA

namespace glades {
namespace gpu {

inline bool fp8_init() { return false; }
inline bool fp8_supported() { return false; }
inline bool fp8_calibrate_amax_e4m3(const float*, size_t, float*) { return false; }
inline bool sgemm_rowmajor_fp8_e4m3(int, int, int, float, const float*, int,
                                     const float*, int, float, float*, int,
                                     const float*, const float*) { return false; }

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
