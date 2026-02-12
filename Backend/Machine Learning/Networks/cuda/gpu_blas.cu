// cuBLAS wrappers implementation.
#include "gpu_blas.h"

#ifdef GLADES_HAVE_CUDA

#include <cublas_v2.h>
#include <cstdio>
#include "gpu_device.h"

namespace glades {
namespace gpu {

namespace {
static cublasHandle_t g_handle = 0;
static bool g_initialized = false;
} // namespace

bool blasInit()
{
	if (g_initialized)
		return true;

	cublasStatus_t st = cublasCreate(&g_handle);
	if (st != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] cublasCreate failed: %d\n", static_cast<int>(st));
		return false;
	}
	// Enable TF32 tensor core math on Ampere+ (SM 8.0+) for ~2x SGEMM speedup.
	if (computeCapabilityMajor() >= 8)
	{
		cublasSetMathMode(g_handle, CUBLAS_TF32_TENSOR_OP_MATH);
	}

	g_initialized = true;
	return true;
}

void blasDestroy()
{
	if (g_initialized && g_handle)
	{
		cublasDestroy(g_handle);
		g_handle = 0;
		g_initialized = false;
	}
}

// Row-major SGEMM via cuBLAS (column-major).
//
// We want: C_row[M,N] = alpha * A_row[M,K] * B_row[K,N] + beta * C_row[M,N]
//
// cuBLAS sees column-major, so a row-major [M,N] matrix with leading dim N
// looks like a column-major [N,M] matrix with leading dim N.
//
// Trick: C_col^T = alpha * B_col^T * A_col^T + beta * C_col^T
// => cublasSgemm(N, N, N, M, K, alpha, B, ldb, A, lda, beta, C, ldc)
//
// Where A_col^T has shape [K,M] with ld=lda, B_col^T has shape [N,K] with ld=ldb.
bool sgemm_rowmajor(int M, int N, int K,
                     float alpha,
                     const float* A, int lda,
                     const float* B, int ldb,
                     float beta,
                     float* C, int ldc)
{
	if (!g_initialized && !blasInit())
		return false;

	// cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
	// For row-major: swap A<->B, use CUBLAS_OP_N, dimensions are (N, M, K).
	cublasStatus_t st = cublasSgemm(g_handle,
	                                 CUBLAS_OP_N, CUBLAS_OP_N,
	                                 N, M, K,
	                                 &alpha,
	                                 B, ldb,
	                                 A, lda,
	                                 &beta,
	                                 C, ldc);
	if (st != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] cublasSgemm failed: %d (M=%d N=%d K=%d)\n",
		        static_cast<int>(st), M, N, K);
		return false;
	}
	return true;
}

// Row-major SGEMM with A transposed:
// C[M,N] = alpha * A^T[M,K] * B[K,N] + beta * C[M,N]
// where A stored [K,M] row-major, B [K,N], C [M,N].
//
// Derivation (column-major view):
//   row-major A[K,M] = col-major A'[M,K]
//   row-major B[K,N] = col-major B'[N,K]
//   row-major C[M,N] = col-major C'[N,M]
//   C = A^T*B  =>  C'[N,M] = B'[N,K] * A'[M,K]^T
//   cuBLAS: C' = B' * A'^T => (CUBLAS_OP_N, CUBLAS_OP_T, N, M, K, B, A, C)
bool sgemm_rowmajor_atb(int M, int N, int K,
                          float alpha,
                          const float* A, int lda,
                          const float* B, int ldb,
                          float beta,
                          float* C, int ldc)
{
	if (!g_initialized && !blasInit())
		return false;

	cublasStatus_t st = cublasSgemm(g_handle,
	                                 CUBLAS_OP_N, CUBLAS_OP_T,
	                                 N, M, K,
	                                 &alpha,
	                                 B, ldb,
	                                 A, lda,
	                                 &beta,
	                                 C, ldc);
	if (st != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] cublasSgemm(ATB) failed: %d (M=%d N=%d K=%d)\n",
		        static_cast<int>(st), M, N, K);
		return false;
	}
	return true;
}

// Row-major SGEMV: y[M] = alpha * A[M,N] * x[N] + beta * y[M]
//
// cuBLAS sees A as column-major [N,M], so we use CUBLAS_OP_T:
// y = alpha * A_col^T * x + beta * y
// cublasSgemv(handle, CUBLAS_OP_T, N, M, alpha, A, lda, x, 1, beta, y, 1)
//
// But lda for column-major is the number of rows = N (leading dimension of the
// column-major view). For a row-major [M,N] array with row stride = lda, the
// column-major view has N rows and M columns, with ld = lda.
bool sgemv_rowmajor(int M, int N,
                     float alpha,
                     const float* A, int lda,
                     const float* x,
                     float beta,
                     float* y)
{
	if (!g_initialized && !blasInit())
		return false;

	cublasStatus_t st = cublasSgemv(g_handle,
	                                 CUBLAS_OP_T,
	                                 N, M,
	                                 &alpha,
	                                 A, lda,
	                                 x, 1,
	                                 &beta,
	                                 y, 1);
	if (st != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] cublasSgemv failed: %d (M=%d N=%d)\n",
		        static_cast<int>(st), M, N);
		return false;
	}
	return true;
}

// Row-major SGEMM with B transposed:
// C[M,N] = alpha * A[M,K] * B^T[K,N] + beta * C[M,N]
// where A is [M,K], B stored [N,K] row-major, C [M,N].
//
// Derivation (column-major view):
//   row-major A[M,K] = col-major A'[K,M]
//   row-major B[N,K] = col-major B'[K,N]
//   row-major C[M,N] = col-major C'[N,M]
//   C = A*B^T => C'[N,M] = B'[K,N]^T * A'[K,M]
//   cuBLAS: C' = B'^T * A' => (CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, B, A, C)
bool sgemm_rowmajor_abt(int M, int N, int K,
                          float alpha,
                          const float* A, int lda,
                          const float* B, int ldb,
                          float beta,
                          float* C, int ldc)
{
	if (!g_initialized && !blasInit())
		return false;

	cublasStatus_t st = cublasSgemm(g_handle,
	                                 CUBLAS_OP_T, CUBLAS_OP_N,
	                                 N, M, K,
	                                 &alpha,
	                                 B, ldb,
	                                 A, lda,
	                                 &beta,
	                                 C, ldc);
	if (st != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] cublasSgemm(ABT) failed: %d (M=%d N=%d K=%d)\n",
		        static_cast<int>(st), M, N, K);
		return false;
	}
	return true;
}

bool sgemm_batched_strided(int M, int N, int K,
                            float alpha,
                            const float* A, int lda, long long int strideA,
                            const float* B, int ldb, long long int strideB,
                            float beta,
                            float* C, int ldc, long long int strideC,
                            int batchCount)
{
	if (!g_initialized && !blasInit())
		return false;

	// Same row-major trick as sgemm_rowmajor, applied per batch.
	cublasStatus_t st = cublasSgemmStridedBatched(g_handle,
	                                               CUBLAS_OP_N, CUBLAS_OP_N,
	                                               N, M, K,
	                                               &alpha,
	                                               B, ldb, strideB,
	                                               A, lda, strideA,
	                                               &beta,
	                                               C, ldc, strideC,
	                                               batchCount);
	if (st != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] cublasSgemmStridedBatched failed: %d\n",
		        static_cast<int>(st));
		return false;
	}
	return true;
}

bool sgemm_batched_strided_abt(int M, int N, int K,
                                float alpha,
                                const float* A, int lda, long long int strideA,
                                const float* B, int ldb, long long int strideB,
                                float beta,
                                float* C, int ldc, long long int strideC,
                                int batchCount)
{
	if (!g_initialized && !blasInit())
		return false;

	// Same ABT trick as sgemm_rowmajor_abt, applied per batch.
	cublasStatus_t st = cublasSgemmStridedBatched(g_handle,
	                                               CUBLAS_OP_T, CUBLAS_OP_N,
	                                               N, M, K,
	                                               &alpha,
	                                               B, ldb, strideB,
	                                               A, lda, strideA,
	                                               &beta,
	                                               C, ldc, strideC,
	                                               batchCount);
	if (st != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] cublasSgemmStridedBatched(ABT) failed: %d\n",
		        static_cast<int>(st));
		return false;
	}
	return true;
}

bool sgemm_batched_strided_atb(int M, int N, int K,
                                float alpha,
                                const float* A, int lda, long long int strideA,
                                const float* B, int ldb, long long int strideB,
                                float beta,
                                float* C, int ldc, long long int strideC,
                                int batchCount)
{
	if (!g_initialized && !blasInit())
		return false;

	// Same ATB trick as sgemm_rowmajor_atb, applied per batch.
	cublasStatus_t st = cublasSgemmStridedBatched(g_handle,
	                                               CUBLAS_OP_N, CUBLAS_OP_T,
	                                               N, M, K,
	                                               &alpha,
	                                               B, ldb, strideB,
	                                               A, lda, strideA,
	                                               &beta,
	                                               C, ldc, strideC,
	                                               batchCount);
	if (st != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] cublasSgemmStridedBatched(ATB) failed: %d\n",
		        static_cast<int>(st));
		return false;
	}
	return true;
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
