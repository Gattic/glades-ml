// cuBLAS wrappers for Glades ML.
//
// Provides row-major SGEMM/SGEMV wrappers using cuBLAS (which is column-major).
// The wrappers handle the transpose trick: C_row = (C_col)^T = (B^T A^T)_col.
#pragma once

#include <cstddef>

#include "gpu_device.h"

#ifdef GLADES_HAVE_CUDA

namespace glades {
namespace gpu {

// Initialize the cuBLAS handle. Called automatically by initDevice if needed.
// Returns true on success.
bool blasInit();

// Destroy the cuBLAS handle.
void blasDestroy();

// ralph-loop iter 6 (2026-05-14): accessor for the lazy-initialized side
// stream that the *_fast16bf_side wrappers dispatch on.  Returns 0 if no
// _side call has yet been made.  Callers use this with recordEvent /
// streamWaitEvent to synchronize the side stream's results back into the
// main computeStream() before any consumer reads them.
cudaStream_t sideComputeStream();

// Row-major SGEMM: C[M,N] = alpha * A[M,K] * B[K,N] + beta * C[M,N]
// All pointers are device pointers.
bool sgemm_rowmajor(int M, int N, int K,
                     float alpha,
                     const float* A, int lda,
                     const float* B, int ldb,
                     float beta,
                     float* C, int ldc);

// iter 180: global runtime toggle for TF32 tensor-core SGEMM.  When disabled
// (via set_tf32_enabled(false)), all SGEMM wrappers fall through to
// CUBLAS_DEFAULT_MATH (true FP32 accumulation, 23-bit mantissa).  Used by
// the trainer's --fp32-attn-strict flag to remove all bf16/TF32 precision
// concerns when L-transition stability is the bottleneck.  Default: true.
void set_tf32_enabled(bool enabled);
bool get_tf32_enabled();

// Exact row-major SGEMM using default cuBLAS math mode.
// This avoids TF32 tensor-core contraction on Ampere+ for parity-sensitive paths.
bool sgemm_rowmajor_exact(int M, int N, int K,
                          float alpha,
                          const float* A, int lda,
                          const float* B, int ldb,
                          float beta,
                          float* C, int ldc);

// Row-major SGEMV: y[M] = alpha * A[M,N] * x[N] + beta * y[M]
// All pointers are device pointers.
bool sgemv_rowmajor(int M, int N,
                     float alpha,
                     const float* A, int lda,
                     const float* x,
                     float beta,
                     float* y);

// Row-major SGEMM with A transposed:
// C[M,N] = alpha * A^T[M,K] * B[K,N] + beta * C[M,N]
// where A is stored as [K,M] row-major, B as [K,N], C as [M,N].
// Used for weight gradient accumulation: gW += dY^T * X.
bool sgemm_rowmajor_atb(int M, int N, int K,
                          float alpha,
                          const float* A, int lda,
                          const float* B, int ldb,
                          float beta,
                          float* C, int ldc);

bool sgemm_rowmajor_atb_exact(int M, int N, int K,
                              float alpha,
                              const float* A, int lda,
                              const float* B, int ldb,
                              float beta,
                              float* C, int ldc);

// Row-major SGEMM with B transposed:
// C[M,N] = alpha * A[M,K] * B^T[K,N] + beta * C[M,N]
// where A is [M,K], B is stored as [N,K] row-major, C is [M,N].
bool sgemm_rowmajor_abt(int M, int N, int K,
                          float alpha,
                          const float* A, int lda,
                          const float* B, int ldb,
                          float beta,
                          float* C, int ldc);

bool sgemm_rowmajor_abt_exact(int M, int N, int K,
                              float alpha,
                              const float* A, int lda,
                              const float* B, int ldb,
                              float beta,
                              float* C, int ldc);

// === BF16 GEMM wrappers ===
//
// Inputs A and B are BF16 (uint16_t storage). Compute accumulates in FP32 via
// tensor cores (CUBLAS_COMPUTE_32F). Output C is FP32. The row-major layout
// matches the float variants: C[M,N] = alpha * A[M,K] * B[K,N] + beta * C[M,N]
// (plus the usual transposed variants for weight-grad and attention paths).
//
// Tensor cores on SM 8.0+ (Ampere/Ada/Hopper) accelerate these. On pre-Ampere
// hardware cuBLAS will fall back to a software path.
bool sgemm_rowmajor_bf16(int M, int N, int K,
                         float alpha,
                         const unsigned short* A, int lda,
                         const unsigned short* B, int ldb,
                         float beta,
                         float* C, int ldc);

bool sgemm_rowmajor_atb_bf16(int M, int N, int K,
                             float alpha,
                             const unsigned short* A, int lda,
                             const unsigned short* B, int ldb,
                             float beta,
                             float* C, int ldc);

bool sgemm_rowmajor_abt_bf16(int M, int N, int K,
                             float alpha,
                             const unsigned short* A, int lda,
                             const unsigned short* B, int ldb,
                             float beta,
                             float* C, int ldc);

// iter 61 (2026-05-16): BF16-out variant of sgemm_rowmajor_atb_bf16.  Same
// semantics (C = alpha * A^T · B + beta * C) but C is BF16 storage instead
// of FP32.  cuBLAS gemmEx handles the FP32-internal-accumulator → BF16
// rounding on the write.  Used by the weight-grad backward to write
// directly into persistent BF16 dW buffers, eliminating the separate
// bf16_accum_axpy commit kernel (3.1% of GPU time at iter60 baseline).
bool sgemm_rowmajor_atb_bf16_dst_bf16(int M, int N, int K,
                                       float alpha,
                                       const unsigned short* A, int lda,
                                       const unsigned short* B, int ldb,
                                       float beta,
                                       unsigned short* C, int ldc);

// === FAST_16BF GEMM wrappers (FP32 in/out, BF16 tensor-core compute) ===
//
// Same signature as sgemm_rowmajor* (FP32 A, FP32 B, FP32 C) but routes
// through cublasGemmEx with CUBLAS_COMPUTE_32F_FAST_16BF. cuBLAS converts
// the FP32 inputs to BF16 on-chip and accumulates products in FP32 on BF16
// tensor cores (~2x TF32-tensor-core throughput on Ampere/Ada/Hopper).
//
// Useful when callers can tolerate BF16 precision (~7-bit mantissa) on the
// inputs but already pass FP32 buffers (e.g. shared bases that aren't
// stored in BF16 elsewhere).  No operand casting needed; no extra VRAM.
bool sgemm_rowmajor_fast16bf(int M, int N, int K,
                              float alpha,
                              const float* A, int lda,
                              const float* B, int ldb,
                              float beta,
                              float* C, int ldc);

bool sgemm_rowmajor_atb_fast16bf(int M, int N, int K,
                                  float alpha,
                                  const float* A, int lda,
                                  const float* B, int ldb,
                                  float beta,
                                  float* C, int ldc);

bool sgemm_rowmajor_abt_fast16bf(int M, int N, int K,
                                  float alpha,
                                  const float* A, int lda,
                                  const float* B, int ldb,
                                  float beta,
                                  float* C, int ldc);

// CUDA 13.2 optimization (2026-05-21): register a (FP32, BF16) pointer pair
// as a "constant" — when the FAST_16BF wrappers see fp32_ptr as A or B with
// matching element count, they skip the FP32→BF16 cast and use bf16_ptr
// directly.  Caller must guarantee bf16_ptr is a valid BF16 representation
// of *fp32_ptr for at least `n` elements AND that fp32_ptr's values never
// change.  Cleared by unregister_fast16bf_constant().  Up to 16 constants
// can be registered.  Returns false if the table is full or already has an
// entry with the same fp32_ptr.
bool register_fast16bf_constant(const float* fp32_ptr,
                                 const unsigned short* bf16_ptr,
                                 size_t n);
bool unregister_fast16bf_constant(const float* fp32_ptr);

// ralph-loop iter 6 (2026-05-14): FAST_16BF ATB variant dispatched on a
// dedicated side cuBLAS handle bound to its own CUDA stream
// (sideComputeStream()).  Used for GEMMs that can run concurrently with
// main-stream work (e.g. the two readout backward GEMMs are data-
// independent).  Caller must:
//   1. Issue this call.
//   2. recordEvent(sideEvent, sideComputeStream()) afterward.
//   3. streamWaitEvent(computeStream(), sideEvent) before any main-stream
//      op that reads the output buffer C.
bool sgemm_rowmajor_atb_fast16bf_side(int M, int N, int K,
                                       float alpha,
                                       const float* A, int lda,
                                       const float* B, int ldb,
                                       float beta,
                                       float* C, int ldc);

// ralph-loop iter 8 (2026-05-14): non-ATB FAST_16BF variant on side handle.
// Used by SCFA branch-parallel forward/backward for B · q_compr / B · y_compr
// GEMMs that should run concurrently with the inner shear on the main stream.
bool sgemm_rowmajor_fast16bf_side(int M, int N, int K,
                                   float alpha,
                                   const float* A, int lda,
                                   const float* B, int ldb,
                                   float beta,
                                   float* C, int ldc);

// ralph-loop iter 10 (2026-05-14): BF16-in, BF16-out ABT GEMM.
// cuBLAS does not support BF16 output with FP32 inputs through
// CUBLAS_COMPUTE_32F_FAST_16BF (returns CUBLAS_STATUS_NOT_SUPPORTED).
// Workaround: cast both inputs to BF16 upfront (reusing the E_bf_cache and
// q_L_bf scratches that backward already needs), then call this wrapper
// which uses CUDA_R_16BF in/out with CUBLAS_COMPUTE_32F_FAST_16BF compute.
// FP32 accumulate, BF16 store with RNE rounding.  Numerical envelope
// identical to the FP32-in / FP32-out FAST_16BF path plus the C-side BF16
// cast.  Used by --bf16-logits-storage forward readout to write the
// (T × V) logits tensor in BF16 directly.
bool sgemm_rowmajor_abt_bf16_bf16out(int M, int N, int K,
                                      float alpha,
                                      const unsigned short* A, int lda,
                                      const unsigned short* B, int ldb,
                                      float beta,
                                      unsigned short* C, int ldc);

// NOTE: Mixed-precision FP32×BF16→FP32 wrappers were explored for Phase 2f
// but cuBLAS (through at least CUDA 12.x) doesn't support mixed input types
// to cublasGemmEx — both A and B must match.  Stiefel Phase 2f instead
// uses an FP32 cache on GpuStiefelWeight that is refreshed once per
// Adam step at retraction time (see gpu_stiefel.cu).

// Batched-strided BF16 GEMMs (attention path). Inputs are BF16 bit
// patterns (unsigned short); output is FP32. Uses BF16 tensor cores
// via cublasGemmStridedBatchedEx + CUBLAS_COMPUTE_32F_FAST_16BF, giving
// ~2x throughput over TF32-tensor-core SGEMM on Ampere/Ada/Hopper.
bool sgemm_batched_strided_bf16(int M, int N, int K,
                                float alpha,
                                const unsigned short* A, int lda, long long strideA,
                                const unsigned short* B, int ldb, long long strideB,
                                float beta,
                                float* C, int ldc, long long strideC,
                                int batchCount);
bool sgemm_batched_strided_abt_bf16(int M, int N, int K,
                                    float alpha,
                                    const unsigned short* A, int lda, long long strideA,
                                    const unsigned short* B, int ldb, long long strideB,
                                    float beta,
                                    float* C, int ldc, long long strideC,
                                    int batchCount);
bool sgemm_batched_strided_atb_bf16(int M, int N, int K,
                                    float alpha,
                                    const unsigned short* A, int lda, long long strideA,
                                    const unsigned short* B, int ldb, long long strideB,
                                    float beta,
                                    float* C, int ldc, long long strideC,
                                    int batchCount);

// Row-major right-side upper-triangular solve (in-place):
// Solves X[M,N] * R[N,N] = alpha * B[M,N]  where R is upper triangular.
// B is overwritten with the solution X.
bool strsm_rowmajor_right_upper(int M, int N,
                                 float alpha,
                                 const float* R, int ldr,
                                 float* B, int ldb);

// Row-major left-side transpose solve (in-place):
// Solves R^T[M,M] * X[M,N] = alpha * B[M,N] where R is upper triangular.
// B is overwritten with the solution X.
bool strsm_rowmajor_left_upper_transpose(int M, int N,
                                         float alpha,
                                         const float* R, int ldr,
                                         float* B, int ldb);

// Row-major batched left-side transpose solve (in-place):
// R_i^T[M,M] * X_i[M,N] = alpha * B_i[M,N] for i in [0, batchCount).
// Arrays of device pointers are themselves device-resident.
bool strsm_rowmajor_left_upper_transpose_batched(int M, int N,
                                                 float alpha,
                                                 float** Rarray, int ldr,
                                                 float** Barray, int ldb,
                                                 int batchCount);

// Row-major batched right-side upper-triangular solve (in-place):
// X_i[M,N] * R_i[N,N] = alpha * B_i[M,N] for i in [0, batchCount).
// Arrays of device pointers are themselves device-resident.
bool strsm_rowmajor_right_upper_batched(int M, int N,
                                        float alpha,
                                        float** Rarray, int ldr,
                                        float** Barray, int ldb,
                                        int batchCount);

// Row-major batched strided SGEMM:
// C_i[M,N] = alpha * A_i[M,K] * B_i[K,N] + beta * C_i[M,N]
// for i in [0, batchCount).
// A_i = A + i*strideA, B_i = B + i*strideB, C_i = C + i*strideC.
bool sgemm_batched_strided(int M, int N, int K,
                            float alpha,
                            const float* A, int lda, long long int strideA,
                            const float* B, int ldb, long long int strideB,
                            float beta,
                            float* C, int ldc, long long int strideC,
                            int batchCount);

// Exact row-major batched strided SGEMM using default cuBLAS math mode.
// This avoids TF32 tensor-core contraction on Ampere+ for parity-sensitive paths.
bool sgemm_batched_strided_exact(int M, int N, int K,
                                  float alpha,
                                  const float* A, int lda, long long int strideA,
                                  const float* B, int ldb, long long int strideB,
                                  float beta,
                                  float* C, int ldc, long long int strideC,
                                  int batchCount);

// Row-major batched strided SGEMM with B transposed:
// C_i[M,N] = alpha * A_i[M,K] * B_i^T[K,N] + beta * C_i[M,N]
// where each B_i is stored as [N,K] row-major.
bool sgemm_batched_strided_abt(int M, int N, int K,
                                float alpha,
                                const float* A, int lda, long long int strideA,
                                const float* B, int ldb, long long int strideB,
                                float beta,
                                float* C, int ldc, long long int strideC,
                                int batchCount);

// Row-major batched strided SGEMM with A transposed:
// C_i[M,N] = alpha * A_i^T[M,K] * B_i[K,N] + beta * C_i[M,N]
// where each A_i is stored as [K,M] row-major.
bool sgemm_batched_strided_atb(int M, int N, int K,
                                float alpha,
                                const float* A, int lda, long long int strideA,
                                const float* B, int ldb, long long int strideB,
                                float beta,
                                float* C, int ldc, long long int strideC,
                                int batchCount);

// Exact row-major batched strided SGEMM with A transposed using default cuBLAS
// math mode. Avoids TF32 tensor-core contraction on Ampere+.
bool sgemm_batched_strided_atb_exact(int M, int N, int K,
                                      float alpha,
                                      const float* A, int lda, long long int strideA,
                                      const float* B, int ldb, long long int strideB,
                                      float beta,
                                      float* C, int ldc, long long int strideC,
                                      int batchCount);

// Row-major batched pointer-array SGEMM with A transposed:
// C_i[M,N] = alpha * A_i^T[M,K] * B_i[K,N] + beta * C_i[M,N]
// where each A_i is stored as [K,M] row-major, B_i as [K,N], and C_i as [M,N].
// Arrays of pointers are themselves device-resident.
bool sgemm_batched_pointer_atb(int M, int N, int K,
                               float alpha,
                               float** Aarray, int lda,
                               float** Barray, int ldb,
                               float beta,
                               float** Carray, int ldc,
                               int batchCount);

// Row-major batched pointer-array SGEMM with B transposed:
// C_i[M,N] = alpha * A_i[M,K] * B_i^T[K,N] + beta * C_i[M,N]
// where each B_i is stored as [N,K] row-major.
bool sgemm_batched_pointer_abt(int M, int N, int K,
                               float alpha,
                               float** Aarray, int lda,
                               float** Barray, int ldb,
                               float beta,
                               float** Carray, int ldc,
                               int batchCount);

// Row-major batched pointer-array SGEMM:
// C_i[M,N] = alpha * A_i[M,K] * B_i[K,N] + beta * C_i[M,N]
// Arrays of pointers are themselves device-resident.
bool sgemm_batched_pointer(int M, int N, int K,
                           float alpha,
                           float** Aarray, int lda,
                           float** Barray, int ldb,
                           float beta,
                           float** Carray, int ldc,
                           int batchCount);

// Same as above, but alpha/beta are device-resident scalars.
bool sgemm_batched_pointer_device_scalars(int M, int N, int K,
                                          const float* d_alpha,
                                          float** Aarray, int lda,
                                          float** Barray, int ldb,
                                          const float* d_beta,
                                          float** Carray, int ldc,
                                          int batchCount);

// Same as above, but alpha/beta are device-resident scalars.
bool sgemm_batched_pointer_atb_device_scalars(int M, int N, int K,
                                              const float* d_alpha,
                                              float** Aarray, int lda,
                                              float** Barray, int ldb,
                                              const float* d_beta,
                                              float** Carray, int ldc,
                                              int batchCount);

// Same as above, but for the ABT row-major variant with device-resident scalars.
bool sgemm_batched_pointer_abt_device_scalars(int M, int N, int K,
                                              const float* d_alpha,
                                              float** Aarray, int lda,
                                              float** Barray, int ldb,
                                              const float* d_beta,
                                              float** Carray, int ldc,
                                              int batchCount);

} // namespace gpu
} // namespace glades

#else // !GLADES_HAVE_CUDA

namespace glades {
namespace gpu {

inline bool blasInit() { return false; }
inline void blasDestroy() {}
inline cudaStream_t sideComputeStream() { return 0; }

inline bool strsm_rowmajor_right_upper(int, int, float, const float*, int, float*, int) { return false; }
inline bool strsm_rowmajor_left_upper_transpose(int, int, float, const float*, int, float*, int) { return false; }
inline bool strsm_rowmajor_left_upper_transpose_batched(int, int, float, float**, int, float**, int, int) { return false; }
inline bool strsm_rowmajor_right_upper_batched(int, int, float, float**, int, float**, int, int) { return false; }
inline bool sgemm_rowmajor(int, int, int, float, const float*, int, const float*, int, float, float*, int) { return false; }
inline bool sgemm_rowmajor_exact(int, int, int, float, const float*, int, const float*, int, float, float*, int) { return false; }
inline bool sgemm_rowmajor_atb(int, int, int, float, const float*, int, const float*, int, float, float*, int) { return false; }
inline bool sgemm_rowmajor_atb_exact(int, int, int, float, const float*, int, const float*, int, float, float*, int) { return false; }
inline bool sgemm_rowmajor_abt(int, int, int, float, const float*, int, const float*, int, float, float*, int) { return false; }
inline bool sgemm_rowmajor_abt_exact(int, int, int, float, const float*, int, const float*, int, float, float*, int) { return false; }
inline bool sgemm_rowmajor_bf16(int, int, int, float, const unsigned short*, int, const unsigned short*, int, float, float*, int) { return false; }
inline bool sgemm_rowmajor_atb_bf16(int, int, int, float, const unsigned short*, int, const unsigned short*, int, float, float*, int) { return false; }
inline bool sgemm_rowmajor_abt_bf16(int, int, int, float, const unsigned short*, int, const unsigned short*, int, float, float*, int) { return false; }
inline bool sgemm_rowmajor_fast16bf(int, int, int, float, const float*, int, const float*, int, float, float*, int) { return false; }
inline bool sgemm_rowmajor_atb_fast16bf(int, int, int, float, const float*, int, const float*, int, float, float*, int) { return false; }
inline bool sgemm_rowmajor_abt_fast16bf(int, int, int, float, const float*, int, const float*, int, float, float*, int) { return false; }
inline bool register_fast16bf_constant(const float*, const unsigned short*, size_t) { return false; }
inline bool unregister_fast16bf_constant(const float*) { return false; }
inline bool sgemm_rowmajor_atb_fast16bf_side(int, int, int, float, const float*, int, const float*, int, float, float*, int) { return false; }
inline bool sgemm_rowmajor_fast16bf_side(int, int, int, float, const float*, int, const float*, int, float, float*, int) { return false; }
inline bool sgemm_rowmajor_abt_bf16_bf16out(int, int, int, float, const unsigned short*, int, const unsigned short*, int, float, unsigned short*, int) { return false; }
inline void set_tf32_enabled(bool) {}
inline bool get_tf32_enabled() { return false; }
inline bool sgemv_rowmajor(int, int, float, const float*, int, const float*, float, float*) { return false; }
inline bool sgemm_batched_strided(int, int, int, float, const float*, int, long long int, const float*, int, long long int, float, float*, int, long long int, int) { return false; }
inline bool sgemm_batched_strided_exact(int, int, int, float, const float*, int, long long int, const float*, int, long long int, float, float*, int, long long int, int) { return false; }
inline bool sgemm_batched_strided_abt(int, int, int, float, const float*, int, long long int, const float*, int, long long int, float, float*, int, long long int, int) { return false; }
inline bool sgemm_batched_strided_atb(int, int, int, float, const float*, int, long long int, const float*, int, long long int, float, float*, int, long long int, int) { return false; }
inline bool sgemm_batched_strided_atb_exact(int, int, int, float, const float*, int, long long int, const float*, int, long long int, float, float*, int, long long int, int) { return false; }
inline bool sgemm_batched_pointer(int, int, int, float, float**, int, float**, int, float, float**, int, int) { return false; }
inline bool sgemm_batched_pointer_device_scalars(int, int, int, const float*, float**, int, float**, int, const float*, float**, int, int) { return false; }
inline bool sgemm_batched_pointer_atb(int, int, int, float, float**, int, float**, int, float, float**, int, int) { return false; }
inline bool sgemm_batched_pointer_abt(int, int, int, float, float**, int, float**, int, float, float**, int, int) { return false; }
inline bool sgemm_batched_pointer_atb_device_scalars(int, int, int, const float*, float**, int, float**, int, const float*, float**, int, int) { return false; }
inline bool sgemm_batched_pointer_abt_device_scalars(int, int, int, const float*, float**, int, float**, int, const float*, float**, int, int) { return false; }

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
