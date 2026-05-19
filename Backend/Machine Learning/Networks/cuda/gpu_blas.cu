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
static float* g_deviceOne = 0;

static bool sgemm_rowmajor_impl(cublasMath_t mathMode,
                                cublasOperation_t transa,
                                cublasOperation_t transb,
                                int M, int N, int K,
                                float alpha,
                                const float* A, int lda,
                                const float* B, int ldb,
                                float beta,
                                float* C, int ldc,
                                const char* label)
{
	if (!g_initialized && !blasInit())
		return false;

	cublasMath_t oldMathMode = CUBLAS_DEFAULT_MATH;
	cublasStatus_t st = cublasGetMathMode(g_handle, &oldMathMode);
	if (st != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] cublasGetMathMode failed: %d\n", static_cast<int>(st));
		return false;
	}
	if (oldMathMode != mathMode)
	{
		st = cublasSetMathMode(g_handle, mathMode);
		if (st != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "[glades-cuda] cublasSetMathMode failed: %d\n", static_cast<int>(st));
			return false;
		}
	}

	st = cublasSgemm(g_handle,
	                 transa, transb,
	                 N, M, K,
	                 &alpha,
	                 B, ldb,
	                 A, lda,
	                 &beta,
	                 C, ldc);

	if (oldMathMode != mathMode)
	{
		cublasStatus_t rst = cublasSetMathMode(g_handle, oldMathMode);
		if (rst != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "[glades-cuda] cublasSetMathMode restore failed: %d\n",
			        static_cast<int>(rst));
			return false;
		}
	}

	if (st != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] %s failed: %d (M=%d N=%d K=%d)\n",
		        label, static_cast<int>(st), M, N, K);
		return false;
	}
	return true;
}

static bool sgemm_batched_pointer_impl(cublasMath_t mathMode,
                                       cublasOperation_t transa,
                                       cublasOperation_t transb,
                                       int M, int N, int K,
                                       float alpha,
                                       float** Aarray, int lda,
                                       float** Barray, int ldb,
                                       float beta,
                                       float** Carray, int ldc,
                                       int batchCount,
                                       const char* label)
{
	if (!g_initialized && !blasInit())
		return false;
	if (!Aarray || !Barray || !Carray || batchCount <= 0)
		return true;

	cublasMath_t oldMathMode = CUBLAS_DEFAULT_MATH;
	cublasStatus_t st = cublasGetMathMode(g_handle, &oldMathMode);
	if (st != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] cublasGetMathMode failed: %d\n", static_cast<int>(st));
		return false;
	}
	if (oldMathMode != mathMode)
	{
		st = cublasSetMathMode(g_handle, mathMode);
		if (st != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "[glades-cuda] cublasSetMathMode failed: %d\n", static_cast<int>(st));
			return false;
		}
	}

	st = cublasSgemmBatched(g_handle,
	                        transa, transb,
	                        N, M, K,
	                        &alpha,
	                        reinterpret_cast<const float* const*>(Barray), ldb,
	                        reinterpret_cast<const float* const*>(Aarray), lda,
	                        &beta,
	                        Carray, ldc,
	                        batchCount);

	if (oldMathMode != mathMode)
	{
		cublasStatus_t rst = cublasSetMathMode(g_handle, oldMathMode);
		if (rst != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "[glades-cuda] cublasSetMathMode restore failed: %d\n",
			        static_cast<int>(rst));
			return false;
		}
	}

	if (st != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] %s failed: %d (M=%d N=%d K=%d batch=%d)\n",
		        label, static_cast<int>(st), M, N, K, batchCount);
		return false;
	}
	return true;
}
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
	cublasSetStream(g_handle, computeStream());
	// Enable TF32 tensor core math on Ampere+ (SM 8.0+) for ~2x SGEMM speedup.
	if (computeCapabilityMajor() >= 8)
	{
		cublasSetMathMode(g_handle, CUBLAS_TF32_TENSOR_OP_MATH);
	}
	float hostOne = 1.0f;
	cudaError_t e = cudaMalloc(&g_deviceOne, sizeof(float));
	if (e != cudaSuccess)
	{
		fprintf(stderr, "[glades-cuda] cudaMalloc for BLAS scalar failed: %d\n", static_cast<int>(e));
		cublasDestroy(g_handle);
		g_handle = 0;
		return false;
	}
	e = cudaMemcpy(g_deviceOne, &hostOne, sizeof(float), cudaMemcpyHostToDevice);
	if (e != cudaSuccess)
	{
		fprintf(stderr, "[glades-cuda] cudaMemcpy for BLAS scalar failed: %d\n", static_cast<int>(e));
		cudaFree(g_deviceOne);
		g_deviceOne = 0;
		cublasDestroy(g_handle);
		g_handle = 0;
		return false;
	}

	g_initialized = true;
	return true;
}

void blasDestroy()
{
	if (g_initialized && g_handle)
	{
		if (g_deviceOne)
		{
			cudaFree(g_deviceOne);
			g_deviceOne = 0;
		}
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
	return sgemm_rowmajor_impl(computeCapabilityMajor() >= 8 ? CUBLAS_TF32_TENSOR_OP_MATH
	                                                        : CUBLAS_DEFAULT_MATH,
	                           CUBLAS_OP_N, CUBLAS_OP_N,
	                           M, N, K,
	                           alpha, A, lda, B, ldb, beta, C, ldc,
	                           "cublasSgemm");
}

bool sgemm_rowmajor_exact(int M, int N, int K,
                          float alpha,
                          const float* A, int lda,
                          const float* B, int ldb,
                          float beta,
                          float* C, int ldc)
{
	return sgemm_rowmajor_impl(CUBLAS_DEFAULT_MATH,
	                           CUBLAS_OP_N, CUBLAS_OP_N,
	                           M, N, K,
	                           alpha, A, lda, B, ldb, beta, C, ldc,
	                           "cublasSgemm(exact)");
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
	return sgemm_rowmajor_impl(computeCapabilityMajor() >= 8 ? CUBLAS_TF32_TENSOR_OP_MATH
	                                                        : CUBLAS_DEFAULT_MATH,
	                           CUBLAS_OP_N, CUBLAS_OP_T,
	                           M, N, K,
	                           alpha, A, lda, B, ldb, beta, C, ldc,
	                           "cublasSgemm(ATB)");
}

bool sgemm_rowmajor_atb_exact(int M, int N, int K,
                              float alpha,
                              const float* A, int lda,
                              const float* B, int ldb,
                              float beta,
                              float* C, int ldc)
{
	return sgemm_rowmajor_impl(CUBLAS_DEFAULT_MATH,
	                           CUBLAS_OP_N, CUBLAS_OP_T,
	                           M, N, K,
	                           alpha, A, lda, B, ldb, beta, C, ldc,
	                           "cublasSgemm(ATB exact)");
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
	return sgemm_rowmajor_impl(computeCapabilityMajor() >= 8 ? CUBLAS_TF32_TENSOR_OP_MATH
	                                                        : CUBLAS_DEFAULT_MATH,
	                           CUBLAS_OP_T, CUBLAS_OP_N,
	                           M, N, K,
	                           alpha, A, lda, B, ldb, beta, C, ldc,
	                           "cublasSgemm(ABT)");
}

bool sgemm_rowmajor_abt_exact(int M, int N, int K,
                              float alpha,
                              const float* A, int lda,
                              const float* B, int ldb,
                              float beta,
                              float* C, int ldc)
{
	return sgemm_rowmajor_impl(CUBLAS_DEFAULT_MATH,
	                           CUBLAS_OP_T, CUBLAS_OP_N,
	                           M, N, K,
	                           alpha, A, lda, B, ldb, beta, C, ldc,
	                           "cublasSgemm(ABT exact)");
}

// Row-major right-side upper-triangular solve:
// X[M,N] * R[N,N] = alpha * B[M,N], R upper triangular, B overwritten with X.
//
// cuBLAS col-major view: row-major R[N,N] becomes col-major R^T[N,N] (lower tri).
// Row-major B[M,N] becomes col-major B^T[N,M].
// X*R = B  =>  R^T * X^T = B^T  =>  SIDE_LEFT, LOWER, OP_N on the transposed view.
bool strsm_rowmajor_right_upper(int M, int N,
                                 float alpha,
                                 const float* R, int ldr,
                                 float* B, int ldb)
{
	if (!g_initialized && !blasInit())
		return false;

	cublasStatus_t st = cublasStrsm(g_handle,
	                                 CUBLAS_SIDE_LEFT,
	                                 CUBLAS_FILL_MODE_LOWER,
	                                 CUBLAS_OP_N,
	                                 CUBLAS_DIAG_NON_UNIT,
	                                 N, M,
	                                 &alpha,
	                                 R, ldr,
	                                 B, ldb);
	if (st != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] cublasStrsm failed: %d (M=%d N=%d)\n",
		        static_cast<int>(st), M, N);
		return false;
	}
	return true;
}

bool strsm_rowmajor_left_upper_transpose(int M, int N,
                                         float alpha,
                                         const float* R, int ldr,
                                         float* B, int ldb)
{
	if (!g_initialized && !blasInit())
		return false;

	cublasStatus_t st = cublasStrsm(g_handle,
	                                 CUBLAS_SIDE_RIGHT,
	                                 CUBLAS_FILL_MODE_UPPER,
	                                 CUBLAS_OP_N,
	                                 CUBLAS_DIAG_NON_UNIT,
	                                 N, M,
	                                 &alpha,
	                                 R, ldr,
	                                 B, ldb);
	if (st != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr,
		        "[glades-cuda] cublasStrsm left-transpose failed: %d (M=%d N=%d)\n",
		        static_cast<int>(st), M, N);
		return false;
	}
	return true;
}

bool strsm_rowmajor_left_upper_transpose_batched(int M, int N,
                                                 float alpha,
                                                 float** Rarray, int ldr,
                                                 float** Barray, int ldb,
                                                 int batchCount)
{
	if (!g_initialized && !blasInit())
		return false;
	if (!Rarray || !Barray || batchCount <= 0)
		return true;

	const bool useDeviceAlpha = (alpha == 1.0f && g_deviceOne != 0);
	cublasPointerMode_t oldPointerMode = CUBLAS_POINTER_MODE_HOST;
	if (useDeviceAlpha)
	{
		cublasStatus_t pst = cublasGetPointerMode(g_handle, &oldPointerMode);
		if (pst != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "[glades-cuda] cublasGetPointerMode failed: %d\n",
			        static_cast<int>(pst));
			return false;
		}
		pst = cublasSetPointerMode(g_handle, CUBLAS_POINTER_MODE_DEVICE);
		if (pst != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "[glades-cuda] cublasSetPointerMode(device) failed: %d\n",
			        static_cast<int>(pst));
			return false;
		}
	}

	const float* alphaPtr = useDeviceAlpha ? g_deviceOne : &alpha;
	cublasStatus_t st = cublasStrsmBatched(g_handle,
	                                       CUBLAS_SIDE_RIGHT,
	                                       CUBLAS_FILL_MODE_UPPER,
	                                       CUBLAS_OP_N,
	                                       CUBLAS_DIAG_NON_UNIT,
	                                       N, M,
	                                       alphaPtr,
	                                       reinterpret_cast<const float* const*>(Rarray), ldr,
	                                       reinterpret_cast<float* const*>(Barray), ldb,
	                                       batchCount);
	if (useDeviceAlpha)
	{
		cublasStatus_t rst = cublasSetPointerMode(g_handle, oldPointerMode);
		if (rst != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "[glades-cuda] cublasSetPointerMode(host) failed: %d\n",
			        static_cast<int>(rst));
			return false;
		}
	}
	if (st != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr,
		        "[glades-cuda] cublasStrsmBatched left-transpose failed: %d (M=%d N=%d batch=%d)\n",
		        static_cast<int>(st), M, N, batchCount);
		return false;
	}
	return true;
}

bool strsm_rowmajor_right_upper_batched(int M, int N,
                                        float alpha,
                                        float** Rarray, int ldr,
                                        float** Barray, int ldb,
                                        int batchCount)
{
	if (!g_initialized && !blasInit())
		return false;
	if (!Rarray || !Barray || batchCount <= 0)
		return true;

	const bool useDeviceAlpha = (alpha == 1.0f && g_deviceOne != 0);
	cublasPointerMode_t oldPointerMode = CUBLAS_POINTER_MODE_HOST;
	if (useDeviceAlpha)
	{
		cublasStatus_t pst = cublasGetPointerMode(g_handle, &oldPointerMode);
		if (pst != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "[glades-cuda] cublasGetPointerMode failed: %d\n",
			        static_cast<int>(pst));
			return false;
		}
		pst = cublasSetPointerMode(g_handle, CUBLAS_POINTER_MODE_DEVICE);
		if (pst != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "[glades-cuda] cublasSetPointerMode(device) failed: %d\n",
			        static_cast<int>(pst));
			return false;
		}
	}

	const float* alphaPtr = useDeviceAlpha ? g_deviceOne : &alpha;
	cublasStatus_t st = cublasStrsmBatched(g_handle,
	                                       CUBLAS_SIDE_LEFT,
	                                       CUBLAS_FILL_MODE_LOWER,
	                                       CUBLAS_OP_N,
	                                       CUBLAS_DIAG_NON_UNIT,
	                                       N, M,
	                                       alphaPtr,
	                                       reinterpret_cast<const float* const*>(Rarray), ldr,
	                                       reinterpret_cast<float* const*>(Barray), ldb,
	                                       batchCount);
	if (useDeviceAlpha)
	{
		cublasStatus_t rst = cublasSetPointerMode(g_handle, oldPointerMode);
		if (rst != CUBLAS_STATUS_SUCCESS)
		{
			fprintf(stderr, "[glades-cuda] cublasSetPointerMode(host) failed: %d\n",
			        static_cast<int>(rst));
			return false;
		}
	}
	if (st != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] cublasStrsmBatched failed: %d (M=%d N=%d batch=%d)\n",
		        static_cast<int>(st), M, N, batchCount);
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

bool sgemm_batched_pointer_atb(int M, int N, int K,
                               float alpha,
                               float** Aarray, int lda,
                               float** Barray, int ldb,
                               float beta,
                               float** Carray, int ldc,
                               int batchCount)
{
	return sgemm_batched_pointer_impl(computeCapabilityMajor() >= 8 ? CUBLAS_TF32_TENSOR_OP_MATH
	                                                                : CUBLAS_DEFAULT_MATH,
	                                  CUBLAS_OP_N, CUBLAS_OP_T,
	                                  M, N, K,
	                                  alpha,
	                                  Aarray, lda,
	                                  Barray, ldb,
	                                  beta,
	                                  Carray, ldc,
	                                  batchCount,
	                                  "cublasSgemmBatched(ATB)");
}

bool sgemm_batched_pointer_abt(int M, int N, int K,
                               float alpha,
                               float** Aarray, int lda,
                               float** Barray, int ldb,
                               float beta,
                               float** Carray, int ldc,
                               int batchCount)
{
	return sgemm_batched_pointer_impl(computeCapabilityMajor() >= 8 ? CUBLAS_TF32_TENSOR_OP_MATH
	                                                                : CUBLAS_DEFAULT_MATH,
	                                  CUBLAS_OP_T, CUBLAS_OP_N,
	                                  M, N, K,
	                                  alpha,
	                                  Aarray, lda,
	                                  Barray, ldb,
	                                  beta,
	                                  Carray, ldc,
	                                  batchCount,
	                                  "cublasSgemmBatched(ABT)");
}

bool sgemm_batched_pointer(int M, int N, int K,
                           float alpha,
                           float** Aarray, int lda,
                           float** Barray, int ldb,
                           float beta,
                           float** Carray, int ldc,
                           int batchCount)
{
	if (!g_initialized && !blasInit())
		return false;
	if (!Aarray || !Barray || !Carray || batchCount <= 0)
		return true;

	cublasStatus_t st = cublasSgemmBatched(g_handle,
	                                       CUBLAS_OP_N, CUBLAS_OP_N,
	                                       N, M, K,
	                                       &alpha,
	                                       reinterpret_cast<const float* const*>(Barray), ldb,
	                                       reinterpret_cast<const float* const*>(Aarray), lda,
	                                       &beta,
	                                       Carray, ldc,
	                                       batchCount);
	if (st != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] cublasSgemmBatched failed: %d (M=%d N=%d K=%d batch=%d)\n",
		        static_cast<int>(st), M, N, K, batchCount);
		return false;
	}
	return true;
}

bool sgemm_batched_pointer_device_scalars(int M, int N, int K,
                                          const float* d_alpha,
                                          float** Aarray, int lda,
                                          float** Barray, int ldb,
                                          const float* d_beta,
                                          float** Carray, int ldc,
                                          int batchCount)
{
	if (!g_initialized && !blasInit())
		return false;
	if (!d_alpha || !d_beta || !Aarray || !Barray || !Carray || batchCount <= 0)
		return true;

	cublasPointerMode_t oldPointerMode = CUBLAS_POINTER_MODE_HOST;
	cublasStatus_t pst = cublasGetPointerMode(g_handle, &oldPointerMode);
	if (pst != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] cublasGetPointerMode failed: %d\n",
		        static_cast<int>(pst));
		return false;
	}
	pst = cublasSetPointerMode(g_handle, CUBLAS_POINTER_MODE_DEVICE);
	if (pst != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] cublasSetPointerMode(device) failed: %d\n",
		        static_cast<int>(pst));
		return false;
	}

	cublasStatus_t st = cublasSgemmBatched(g_handle,
	                                       CUBLAS_OP_N, CUBLAS_OP_N,
	                                       N, M, K,
	                                       d_alpha,
	                                       reinterpret_cast<const float* const*>(Barray), ldb,
	                                       reinterpret_cast<const float* const*>(Aarray), lda,
	                                       d_beta,
	                                       Carray, ldc,
	                                       batchCount);
	cublasStatus_t rst = cublasSetPointerMode(g_handle, oldPointerMode);
	if (rst != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] cublasSetPointerMode(host) failed: %d\n",
		        static_cast<int>(rst));
		return false;
	}
	if (st != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] cublasSgemmBatched failed: %d (M=%d N=%d K=%d batch=%d)\n",
		        static_cast<int>(st), M, N, K, batchCount);
		return false;
	}
	return true;
}

bool sgemm_batched_pointer_atb_device_scalars(int M, int N, int K,
                                              const float* d_alpha,
                                              float** Aarray, int lda,
                                              float** Barray, int ldb,
                                              const float* d_beta,
                                              float** Carray, int ldc,
                                              int batchCount)
{
	if (!g_initialized && !blasInit())
		return false;
	if (!d_alpha || !d_beta || !Aarray || !Barray || !Carray || batchCount <= 0)
		return true;

	cublasPointerMode_t oldPointerMode = CUBLAS_POINTER_MODE_HOST;
	cublasStatus_t pst = cublasGetPointerMode(g_handle, &oldPointerMode);
	if (pst != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] cublasGetPointerMode failed: %d\n",
		        static_cast<int>(pst));
		return false;
	}
	pst = cublasSetPointerMode(g_handle, CUBLAS_POINTER_MODE_DEVICE);
	if (pst != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] cublasSetPointerMode(device) failed: %d\n",
		        static_cast<int>(pst));
		return false;
	}

	cublasStatus_t st = cublasSgemmBatched(g_handle,
	                                       CUBLAS_OP_N, CUBLAS_OP_T,
	                                       N, M, K,
	                                       d_alpha,
	                                       reinterpret_cast<const float* const*>(Barray), ldb,
	                                       reinterpret_cast<const float* const*>(Aarray), lda,
	                                       d_beta,
	                                       Carray, ldc,
	                                       batchCount);

	cublasStatus_t rst = cublasSetPointerMode(g_handle, oldPointerMode);
	if (rst != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] cublasSetPointerMode(host) failed: %d\n",
		        static_cast<int>(rst));
		return false;
	}
	if (st != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] cublasSgemmBatched(ATB,device) failed: %d (M=%d N=%d K=%d batch=%d)\n",
		        static_cast<int>(st), M, N, K, batchCount);
		return false;
	}
	return true;
}

bool sgemm_batched_pointer_abt_device_scalars(int M, int N, int K,
                                              const float* d_alpha,
                                              float** Aarray, int lda,
                                              float** Barray, int ldb,
                                              const float* d_beta,
                                              float** Carray, int ldc,
                                              int batchCount)
{
	if (!g_initialized && !blasInit())
		return false;
	if (!d_alpha || !d_beta || !Aarray || !Barray || !Carray || batchCount <= 0)
		return true;

	cublasPointerMode_t oldPointerMode = CUBLAS_POINTER_MODE_HOST;
	cublasStatus_t pst = cublasGetPointerMode(g_handle, &oldPointerMode);
	if (pst != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] cublasGetPointerMode failed: %d\n",
		        static_cast<int>(pst));
		return false;
	}
	pst = cublasSetPointerMode(g_handle, CUBLAS_POINTER_MODE_DEVICE);
	if (pst != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] cublasSetPointerMode(device) failed: %d\n",
		        static_cast<int>(pst));
		return false;
	}

	cublasStatus_t st = cublasSgemmBatched(g_handle,
	                                       CUBLAS_OP_T, CUBLAS_OP_N,
	                                       N, M, K,
	                                       d_alpha,
	                                       reinterpret_cast<const float* const*>(Barray), ldb,
	                                       reinterpret_cast<const float* const*>(Aarray), lda,
	                                       d_beta,
	                                       Carray, ldc,
	                                       batchCount);
	cublasStatus_t rst = cublasSetPointerMode(g_handle, oldPointerMode);
	if (rst != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] cublasSetPointerMode(host) failed: %d\n",
		        static_cast<int>(rst));
		return false;
	}
	if (st != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] cublasSgemmBatched(ABT) failed: %d (M=%d N=%d K=%d batch=%d)\n",
		        static_cast<int>(st), M, N, K, batchCount);
		return false;
	}
	return true;
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
