// cuBLAS wrappers for Glades ML.
//
// Provides row-major SGEMM/SGEMV wrappers using cuBLAS (which is column-major).
// The wrappers handle the transpose trick: C_row = (C_col)^T = (B^T A^T)_col.
#pragma once

#include <cstddef>

#ifdef GLADES_HAVE_CUDA

namespace glades {
namespace gpu {

// Initialize the cuBLAS handle. Called automatically by initDevice if needed.
// Returns true on success.
bool blasInit();

// Destroy the cuBLAS handle.
void blasDestroy();

// Row-major SGEMM: C[M,N] = alpha * A[M,K] * B[K,N] + beta * C[M,N]
// All pointers are device pointers.
bool sgemm_rowmajor(int M, int N, int K,
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

// Row-major SGEMM with B transposed:
// C[M,N] = alpha * A[M,K] * B^T[K,N] + beta * C[M,N]
// where A is [M,K], B is stored as [N,K] row-major, C is [M,N].
bool sgemm_rowmajor_abt(int M, int N, int K,
                          float alpha,
                          const float* A, int lda,
                          const float* B, int ldb,
                          float beta,
                          float* C, int ldc);

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

} // namespace gpu
} // namespace glades

#else // !GLADES_HAVE_CUDA

namespace glades {
namespace gpu {

inline bool blasInit() { return false; }
inline void blasDestroy() {}

inline bool sgemm_rowmajor(int, int, int, float, const float*, int, const float*, int, float, float*, int) { return false; }
inline bool sgemm_rowmajor_atb(int, int, int, float, const float*, int, const float*, int, float, float*, int) { return false; }
inline bool sgemm_rowmajor_abt(int, int, int, float, const float*, int, const float*, int, float, float*, int) { return false; }
inline bool sgemv_rowmajor(int, int, float, const float*, int, const float*, float, float*) { return false; }
inline bool sgemm_batched_strided(int, int, int, float, const float*, int, long long int, const float*, int, long long int, float, float*, int, long long int, int) { return false; }
inline bool sgemm_batched_strided_abt(int, int, int, float, const float*, int, long long int, const float*, int, long long int, float, float*, int, long long int, int) { return false; }
inline bool sgemm_batched_strided_atb(int, int, int, float, const float*, int, long long int, const float*, int, long long int, float, float*, int, long long int, int) { return false; }

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
