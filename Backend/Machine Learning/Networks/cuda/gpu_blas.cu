// cuBLAS wrappers implementation.
#include "gpu_blas.h"

#ifdef GLADES_HAVE_CUDA

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>
#include "gpu_device.h"
#include "gpu_kernels.h"

namespace glades {
namespace gpu {

namespace {
static cublasHandle_t g_handle = 0;
// Two-handle dispatch design for CUDA Graphs compatibility:
// g_handleStrict is always set to CUBLAS_DEFAULT_MATH (strict FP32).
// g_handleTf32 is always set to CUBLAS_TF32_TENSOR_OP_MATH (when CC >= 8.0).
// Per-call mathMode picks the appropriate handle WITHOUT toggling state.
// This eliminates cublasSet/GetMathMode calls inside the hot path, which
// are not capture-compatible (paradigm #51 ATLAS-COMPILE).
static cublasHandle_t g_handleStrict = NULL;
static cublasHandle_t g_handleTf32   = NULL;
static bool g_initialized = false;
// iter 180: when false, all wrappers below select CUBLAS_DEFAULT_MATH (no TF32).
static bool g_tf32_enabled = true;
static float* g_deviceOne = 0;

// CUDA 13.2 mitigation (2026-05-21): cuBLAS 13.x heuristic for cublasGemmEx
// with CUDA_R_32F inputs + CUBLAS_COMPUTE_32F_FAST_16BF dispatches to a
// non-bf16-specialized s1688gemm kernel on Ada (sm_8.9), ~1.7-1.8x slower
// than the s16816_bf16 fast path that CUDA 12.0 picks for the same call.
// The FAST_16BF wrappers below now pre-cast FP32 inputs to BF16 in scratch
// and call cublasGemmEx with CUDA_R_16BF inputs to force the fast path.
// Math envelope unchanged (one BF16 round-trip before TC, same as before).
static uint16_t* g_fast16bf_scratch_a = 0;
static uint16_t* g_fast16bf_scratch_b = 0;
static size_t    g_fast16bf_scratch_a_n = 0;
static size_t    g_fast16bf_scratch_b_n = 0;

static bool ensureFast16bfScratch(uint16_t** scratch, size_t* size_n, size_t needed_n)
{
	if (*size_n >= needed_n) return true;
	if (*scratch) cudaFree(*scratch);
	cudaError_t err = cudaMalloc(scratch, needed_n * sizeof(uint16_t));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "[glades-cuda] fast16bf-bf16cast scratch alloc failed: %zu elements (%s)\n",
		        needed_n, cudaGetErrorString(err));
		*scratch = 0;
		*size_n = 0;
		return false;
	}
	*size_n = needed_n;
	return true;
}

// Constant-BF16 cache: skip the FP32->BF16 cast for pre-registered constant
// inputs (e.g. SCFA basis scfa_B that never changes).  Caller supplies the
// BF16 mirror buffer; we just remember the pointer pair.
struct Fast16bfConstantEntry
{
	const float*    fp32_ptr;
	const uint16_t* bf16_ptr;
	size_t          n;
};
static const int kMaxFast16bfConstants = 16;
static Fast16bfConstantEntry g_fast16bf_constants[kMaxFast16bfConstants];
static int g_fast16bf_constants_count = 0;

static const uint16_t* lookupFast16bfConstant(const float* fp32_ptr, size_t needed_n)
{
	for (int i = 0; i < g_fast16bf_constants_count; ++i)
	{
		if (g_fast16bf_constants[i].fp32_ptr == fp32_ptr &&
		    g_fast16bf_constants[i].n >= needed_n)
		{
			return g_fast16bf_constants[i].bf16_ptr;
		}
	}
	return 0;
}

// ralph-loop iter 6 (2026-05-14): side cuBLAS handle bound to a dedicated
// side stream, used to dispatch GEMMs that can run concurrently with the
// main computeStream() GEMMs.  Lazily initialized on first request via
// ensureSideHandle().  The two readout backward GEMMs (dq_L = dlogits·E,
// dE += dlogits^T·q_L) are independent and benefit from cross-stream
// concurrency at T=8192 V=32000 m=2048 — each is 1.07 TFLOP at BF16-TC,
// ~33 ms / GEMM, parallelization saves ~21 ms / step ≈ +5% e2e.
static cublasHandle_t g_handleSide = 0;
static cudaStream_t   g_sideStream = 0;
static bool g_sideInitialized = false;

static bool ensureSideHandle()
{
	if (g_sideInitialized) return true;
	if (!g_initialized && !blasInit()) return false;
	cudaError_t e = cudaStreamCreateWithFlags(&g_sideStream, cudaStreamNonBlocking);
	if (e != cudaSuccess)
	{
		fprintf(stderr, "[glades-cuda] cudaStreamCreate (side) failed: %d\n",
		        static_cast<int>(e));
		g_sideStream = 0;
		return false;
	}
	cublasStatus_t st = cublasCreate(&g_handleSide);
	if (st != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] cublasCreate (side) failed: %d\n",
		        static_cast<int>(st));
		cudaStreamDestroy(g_sideStream);
		g_sideStream = 0;
		g_handleSide = 0;
		return false;
	}
	cublasSetStream(g_handleSide, g_sideStream);
	// Side handle follows the same two-handle invariant: set to native mode
	// at init, never toggled afterwards. Callers dispatch via mathMode arg.
	if (computeCapabilityMajor() >= 8)
	{
		cublasSetMathMode(g_handleSide, CUBLAS_TF32_TENSOR_OP_MATH);
	}
	g_sideInitialized = true;
	return true;
}

// Pick the appropriate handle for the requested mathMode.
// At init, g_handleTf32 is always TF32 (CC>=8) or DEFAULT (CC<8);
// g_handleStrict is always DEFAULT.  This dispatcher eliminates
// per-call cublasSet/GetMathMode, making the call sequence
// graph-captureable.
static inline cublasHandle_t pick_handle(cublasMath_t mathMode)
{
	if (mathMode == CUBLAS_TF32_TENSOR_OP_MATH)
		return g_handleTf32;
	return g_handleStrict;
}

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

	// Two-handle dispatch: pick the handle whose math mode matches the
	// caller's request.  No state toggling, no cublasSet/GetMathMode in
	// the hot path — capture-compatible (paradigm #51 ATLAS-COMPILE).
	cublasHandle_t h = pick_handle(mathMode);

	cublasStatus_t st = cublasSgemm(h,
	                                transa, transb,
	                                N, M, K,
	                                &alpha,
	                                B, ldb,
	                                A, lda,
	                                &beta,
	                                C, ldc);

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

	cublasHandle_t h = pick_handle(mathMode);

	cublasStatus_t st = cublasSgemmBatched(h,
	                                       transa, transb,
	                                       N, M, K,
	                                       &alpha,
	                                       reinterpret_cast<const float* const*>(Barray), ldb,
	                                       reinterpret_cast<const float* const*>(Aarray), lda,
	                                       &beta,
	                                       Carray, ldc,
	                                       batchCount);

	if (st != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] %s failed: %d (M=%d N=%d K=%d batch=%d)\n",
		        label, static_cast<int>(st), M, N, K, batchCount);
		return false;
	}
	return true;
}
} // namespace

// iter 180: public TF32 toggle.  __attribute__((used,visibility("default")))
// forces the linker to keep these in libglades.so even though no other
// glades-ml TU references them — the trainer is the only consumer.
__attribute__((used, visibility("default")))
void set_tf32_enabled(bool enabled)
{
	// Flag-only API.  The two-handle design (Task 1.1) means callers
	// explicitly request TF32 or strict via the mathMode arg per call;
	// this flag is read by callers that want to HONOR the user's global
	// toggle.
	//
	// IMPORTANT for cuda-graphs compatibility: NO cublasSetMathMode
	// happens here.  The handles' math modes are set ONCE at blasInit()
	// and never toggled afterwards.  This preserves the two-handle
	// invariant (g_handleStrict always DEFAULT, g_handleTf32 always
	// TF32 on CC>=8) which is required for capture-compatible dispatch.
	g_tf32_enabled = enabled;
}

__attribute__((used, visibility("default")))
bool get_tf32_enabled() { return g_tf32_enabled; }

bool blasInit()
{
	if (g_initialized)
		return true;

	// Create the strict-FP32 handle.
	cublasStatus_t st = cublasCreate(&g_handleStrict);
	if (st != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] cublasCreate (strict) failed: %d\n", static_cast<int>(st));
		return false;
	}
	cublasSetStream(g_handleStrict, computeStream());
	cublasSetMathMode(g_handleStrict, CUBLAS_DEFAULT_MATH);

	// Create the TF32 handle.
	st = cublasCreate(&g_handleTf32);
	if (st != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] cublasCreate (tf32) failed: %d\n", static_cast<int>(st));
		cublasDestroy(g_handleStrict);
		g_handleStrict = NULL;
		return false;
	}
	cublasSetStream(g_handleTf32, computeStream());
	// Enable TF32 tensor core math on Ampere+ (SM 8.0+) for ~2x SGEMM speedup.
	if (computeCapabilityMajor() >= 8)
	{
		cublasSetMathMode(g_handleTf32, CUBLAS_TF32_TENSOR_OP_MATH);
	}
	else
	{
		// Pre-Ampere: no TF32; both handles use CUBLAS_DEFAULT_MATH.
		cublasSetMathMode(g_handleTf32, CUBLAS_DEFAULT_MATH);
	}

	// Maintain g_handle as an alias to g_handleTf32 for backward compatibility
	// with code paths that haven't been updated yet.  Will be removed in
	// Task 1.4 once all callers route through the two-handle dispatch.
	g_handle = g_handleTf32;

	float hostOne = 1.0f;
	cudaError_t e = cudaMalloc(&g_deviceOne, sizeof(float));
	if (e != cudaSuccess)
	{
		fprintf(stderr, "[glades-cuda] cudaMalloc for BLAS scalar failed: %d\n", static_cast<int>(e));
		cublasDestroy(g_handleTf32);
		g_handleTf32 = NULL;
		cublasDestroy(g_handleStrict);
		g_handleStrict = NULL;
		g_handle = 0;
		return false;
	}
	e = cudaMemcpy(g_deviceOne, &hostOne, sizeof(float), cudaMemcpyHostToDevice);
	if (e != cudaSuccess)
	{
		fprintf(stderr, "[glades-cuda] cudaMemcpy for BLAS scalar failed: %d\n", static_cast<int>(e));
		cudaFree(g_deviceOne);
		g_deviceOne = 0;
		cublasDestroy(g_handleTf32);
		g_handleTf32 = NULL;
		cublasDestroy(g_handleStrict);
		g_handleStrict = NULL;
		g_handle = 0;
		return false;
	}

	g_initialized = true;
	return true;
}

void blasDestroy()
{
	// ralph-loop iter 6: tear down side handle/stream first if initialized.
	if (g_sideInitialized)
	{
		if (g_handleSide) { cublasDestroy(g_handleSide); g_handleSide = 0; }
		if (g_sideStream) { cudaStreamDestroy(g_sideStream); g_sideStream = 0; }
		g_sideInitialized = false;
	}
	if (g_initialized)
	{
		if (g_deviceOne)
		{
			cudaFree(g_deviceOne);
			g_deviceOne = 0;
		}
		// Tear down both handles; g_handle is an alias to g_handleTf32 so
		// destroy through the canonical pointers to avoid double-free.
		if (g_handleStrict) { cublasDestroy(g_handleStrict); g_handleStrict = NULL; }
		if (g_handleTf32)   { cublasDestroy(g_handleTf32);   g_handleTf32   = NULL; }
		g_handle = 0;
		g_initialized = false;
	}
}

// ralph-loop iter 6 (2026-05-14): public accessor for the side stream so
// callers can record events on it for cross-stream synchronization.
// Returns 0 if the side handle has never been initialized (the cuBLAS
// handle is lazy-init'd by the first sgemm_*_side call below).
cudaStream_t sideComputeStream()
{
	return g_sideStream;
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
	return sgemm_rowmajor_impl((g_tf32_enabled && computeCapabilityMajor() >= 8) ? CUBLAS_TF32_TENSOR_OP_MATH
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
	return sgemm_rowmajor_impl((g_tf32_enabled && computeCapabilityMajor() >= 8) ? CUBLAS_TF32_TENSOR_OP_MATH
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
	return sgemm_rowmajor_impl((g_tf32_enabled && computeCapabilityMajor() >= 8) ? CUBLAS_TF32_TENSOR_OP_MATH
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
	return sgemm_batched_pointer_impl((g_tf32_enabled && computeCapabilityMajor() >= 8) ? CUBLAS_TF32_TENSOR_OP_MATH
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
	return sgemm_batched_pointer_impl((g_tf32_enabled && computeCapabilityMajor() >= 8) ? CUBLAS_TF32_TENSOR_OP_MATH
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

// === BF16 GEMM wrappers ===
//
// All wrappers use cublasGemmEx with CUDA_R_16BF inputs and CUDA_R_32F output.
// Compute type is CUBLAS_COMPUTE_32F so products accumulate in FP32 on the
// tensor cores, matching the numerical profile of TF32 SGEMM with half the
// input memory bandwidth and 2x the arithmetic throughput on Ampere/Ada/Hopper.
//
// Row-major -> column-major transpose trick is identical to the float path:
//   C_row[M,N] = alpha * A_row[M,K] * B_row[K,N] + beta * C_row[M,N]
//   cuBLAS sees it as (in column-major):
//     C_col[N,M] = alpha * B_col[N,K] * A_col[K,M] + beta * C_col[N,M]
// so we pass (B, A) in swapped order with shape (N, M, K).

static bool gemmex_bf16_impl(cublasOperation_t transa, cublasOperation_t transb,
                             int M, int N, int K,
                             float alpha,
                             const unsigned short* A, int lda,
                             const unsigned short* B, int ldb,
                             float beta,
                             float* C, int ldc,
                             const char* label)
{
	if (!g_initialized && !blasInit())
		return false;

	// CUBLAS_COMPUTE_32F_FAST_16BF: compute via BF16 tensor cores with FP32
	// accumulate. This is the explicit BF16 tensor-core path and can be faster
	// on Ampere/Ada than plain CUBLAS_COMPUTE_32F with BF16 inputs, which may
	// defensively promote to TF32.
	cublasStatus_t st = cublasGemmEx(g_handle,
	                                 transa, transb,
	                                 N, M, K,
	                                 &alpha,
	                                 B, CUDA_R_16BF, ldb,
	                                 A, CUDA_R_16BF, lda,
	                                 &beta,
	                                 C, CUDA_R_32F, ldc,
	                                 CUBLAS_COMPUTE_32F_FAST_16BF,
	                                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	if (st != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] %s failed: %d (M=%d N=%d K=%d)\n",
		        label, static_cast<int>(st), M, N, K);
		return false;
	}
	return true;
}

// C[M,N] = alpha * A[M,K] * B[K,N] + beta * C (row-major, BF16 inputs, FP32 out).
bool sgemm_rowmajor_bf16(int M, int N, int K,
                         float alpha,
                         const unsigned short* A, int lda,
                         const unsigned short* B, int ldb,
                         float beta,
                         float* C, int ldc)
{
	return gemmex_bf16_impl(CUBLAS_OP_N, CUBLAS_OP_N,
	                        M, N, K,
	                        alpha, A, lda, B, ldb, beta, C, ldc,
	                        "cublasGemmEx(BF16)");
}

// C[M,N] = alpha * A^T[M,K] * B[K,N] + beta * C (A stored as [K,M] row-major).
// Same col-major transpose logic as the float variant: (transa, transb) = (N, T).
bool sgemm_rowmajor_atb_bf16(int M, int N, int K,
                             float alpha,
                             const unsigned short* A, int lda,
                             const unsigned short* B, int ldb,
                             float beta,
                             float* C, int ldc)
{
	return gemmex_bf16_impl(CUBLAS_OP_N, CUBLAS_OP_T,
	                        M, N, K,
	                        alpha, A, lda, B, ldb, beta, C, ldc,
	                        "cublasGemmEx(BF16,ATB)");
}

// C[M,N] = alpha * A[M,K] * B^T[K,N] + beta * C (B stored as [N,K] row-major).
// Col-major transpose: (transa, transb) = (T, N).
bool sgemm_rowmajor_abt_bf16(int M, int N, int K,
                             float alpha,
                             const unsigned short* A, int lda,
                             const unsigned short* B, int ldb,
                             float beta,
                             float* C, int ldc)
{
	return gemmex_bf16_impl(CUBLAS_OP_T, CUBLAS_OP_N,
	                        M, N, K,
	                        alpha, A, lda, B, ldb, beta, C, ldc,
	                        "cublasGemmEx(BF16,ABT)");
}

// iter 61 (2026-05-16): BF16-out variant.  Same row-major→col-major transpose
// trick as gemmex_bf16_impl but D type is CUDA_R_16BF.  cuBLAS reads C as
// BF16 (sign-extended to FP32 internally for the beta*C term), accumulates
// alpha*A*B in FP32 inside the tensor cores, then casts the FP32 result back
// to BF16 (RN-even) on store.  Used by the weight-grad backward path to
// commit dW directly into the BF16 persistent grad buffer, eliminating the
// downstream bf16_accum_axpy commit kernel.
static bool gemmex_bf16_impl_dst_bf16(cublasOperation_t transa, cublasOperation_t transb,
                                       int M, int N, int K,
                                       float alpha,
                                       const unsigned short* A, int lda,
                                       const unsigned short* B, int ldb,
                                       float beta,
                                       unsigned short* C, int ldc,
                                       const char* label)
{
	if (!g_initialized && !blasInit())
		return false;

	cublasStatus_t st = cublasGemmEx(g_handle,
	                                 transa, transb,
	                                 N, M, K,
	                                 &alpha,
	                                 B, CUDA_R_16BF, ldb,
	                                 A, CUDA_R_16BF, lda,
	                                 &beta,
	                                 C, CUDA_R_16BF, ldc,
	                                 CUBLAS_COMPUTE_32F_FAST_16BF,
	                                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	if (st != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] %s failed: %d (M=%d N=%d K=%d)\n",
		        label, static_cast<int>(st), M, N, K);
		return false;
	}
	return true;
}

bool sgemm_rowmajor_atb_bf16_dst_bf16(int M, int N, int K,
                                       float alpha,
                                       const unsigned short* A, int lda,
                                       const unsigned short* B, int ldb,
                                       float beta,
                                       unsigned short* C, int ldc)
{
	return gemmex_bf16_impl_dst_bf16(CUBLAS_OP_N, CUBLAS_OP_T,
	                                  M, N, K,
	                                  alpha, A, lda, B, ldb, beta, C, ldc,
	                                  "cublasGemmEx(BF16,ATB,dstBF16)");
}

// === FAST_16BF GEMM (FP32 in/out, BF16 tensor-core compute) ===
//
// Same row-major→col-major transpose trick as sgemm_rowmajor.  Routes
// through cublasGemmEx with CUBLAS_COMPUTE_32F_FAST_16BF, which converts
// FP32 inputs to BF16 on-chip (RNE rounding) and accumulates products in
// FP32.  Throughput matches sgemm_rowmajor_bf16 (~2x TF32-TC) but avoids
// the operand-cast plumbing required when the FP32 source isn't already
// mirrored to a BF16 buffer (e.g. the SCFA shared DCT basis scfa_B and
// the per-call FP32 intermediates q_compr/q_par/y_compr/y_par/dy).
//
// Numerical envelope is identical to sgemm_rowmajor_bf16 — the operands
// pass through one BF16 round-trip before TC compute.  Match-mode is
// independent of cublasSetMathMode (cublasGemmEx takes computeType
// explicitly), so no math-mode toggle / restore.
static bool sgemm_rowmajor_fast16bf_impl(cublasOperation_t transa,
                                          cublasOperation_t transb,
                                          int M, int N, int K,
                                          float alpha,
                                          const float* A, int lda,
                                          const float* B, int ldb,
                                          float beta,
                                          float* C, int ldc,
                                          const char* label)
{
	if (!g_initialized && !blasInit()) return false;

	// CUDA 13.2 mitigation: pre-cast FP32 inputs to BF16 scratch, then call
	// cublasGemmEx with CUDA_R_16BF inputs.  Routes dispatch to the
	// s16816_bf16 fast path on Ada (vs the s1688gemm slow path that
	// cuBLAS 13.x picks for FP32-input FAST_16BF).  Numerics unchanged
	// (FAST_16BF compute path also casts to BF16 internally).
	//
	// Size derivation in our row-major calling convention:
	//   (transa,transb)=(N,N) no-trans wrapper:  A=[M,K] lda=K, B=[K,N] ldb=N
	//   (transa,transb)=(N,T) atb wrapper (A^T): A=[K,M] lda=M, B=[K,N] ldb=N
	//   (transa,transb)=(T,N) abt wrapper (B^T): A=[M,K] lda=K, B=[N,K] ldb=K
	// transa applies to A_cublas (= our B); transb applies to B_cublas (= our A).
	// So A's storage flips (M→K rows) when transb=T; B's storage flips when transa=T.
	size_t A_n = (size_t)lda * (size_t)(transb == CUBLAS_OP_N ? M : K);
	size_t B_n = (size_t)ldb * (size_t)(transa == CUBLAS_OP_N ? K : N);

	// Look up registered constants first (skip cast if hit).
	const uint16_t* A_bf16 = lookupFast16bfConstant(A, A_n);
	const uint16_t* B_bf16 = lookupFast16bfConstant(B, B_n);

	if (!A_bf16)
	{
		if (!ensureFast16bfScratch(&g_fast16bf_scratch_a, &g_fast16bf_scratch_a_n, A_n))
			return false;
		if (!cast_f32_to_bf16(A, g_fast16bf_scratch_a, A_n)) return false;
		A_bf16 = g_fast16bf_scratch_a;
	}
	if (!B_bf16)
	{
		if (!ensureFast16bfScratch(&g_fast16bf_scratch_b, &g_fast16bf_scratch_b_n, B_n))
			return false;
		if (!cast_f32_to_bf16(B, g_fast16bf_scratch_b, B_n)) return false;
		B_bf16 = g_fast16bf_scratch_b;
	}

	cublasStatus_t st = cublasGemmEx(g_handle,
	                                 transa, transb,
	                                 N, M, K,
	                                 &alpha,
	                                 B_bf16, CUDA_R_16BF, ldb,
	                                 A_bf16, CUDA_R_16BF, lda,
	                                 &beta,
	                                 C, CUDA_R_32F, ldc,
	                                 CUBLAS_COMPUTE_32F_FAST_16BF,
	                                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	if (st != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] %s failed: %d (M=%d N=%d K=%d)\n",
		        label, static_cast<int>(st), M, N, K);
		return false;
	}
	return true;
}

bool sgemm_rowmajor_fast16bf(int M, int N, int K,
                              float alpha,
                              const float* A, int lda,
                              const float* B, int ldb,
                              float beta,
                              float* C, int ldc)
{
	return sgemm_rowmajor_fast16bf_impl(CUBLAS_OP_N, CUBLAS_OP_N,
	                                     M, N, K,
	                                     alpha, A, lda, B, ldb, beta, C, ldc,
	                                     "cublasGemmEx(FAST_16BF)");
}

bool sgemm_rowmajor_atb_fast16bf(int M, int N, int K,
                                  float alpha,
                                  const float* A, int lda,
                                  const float* B, int ldb,
                                  float beta,
                                  float* C, int ldc)
{
	return sgemm_rowmajor_fast16bf_impl(CUBLAS_OP_N, CUBLAS_OP_T,
	                                     M, N, K,
	                                     alpha, A, lda, B, ldb, beta, C, ldc,
	                                     "cublasGemmEx(FAST_16BF,ATB)");
}

bool sgemm_rowmajor_abt_fast16bf(int M, int N, int K,
                                  float alpha,
                                  const float* A, int lda,
                                  const float* B, int ldb,
                                  float beta,
                                  float* C, int ldc)
{
	return sgemm_rowmajor_fast16bf_impl(CUBLAS_OP_T, CUBLAS_OP_N,
	                                     M, N, K,
	                                     alpha, A, lda, B, ldb, beta, C, ldc,
	                                     "cublasGemmEx(FAST_16BF,ABT)");
}

bool register_fast16bf_constant(const float* fp32_ptr,
                                 const unsigned short* bf16_ptr,
                                 size_t n)
{
	if (!fp32_ptr || !bf16_ptr || n == 0) return false;
	// Already registered? Update in place.
	for (int i = 0; i < g_fast16bf_constants_count; ++i)
	{
		if (g_fast16bf_constants[i].fp32_ptr == fp32_ptr)
		{
			g_fast16bf_constants[i].bf16_ptr = bf16_ptr;
			g_fast16bf_constants[i].n        = n;
			return true;
		}
	}
	if (g_fast16bf_constants_count >= kMaxFast16bfConstants)
	{
		fprintf(stderr, "[glades-cuda] register_fast16bf_constant: table full (max %d)\n",
		        kMaxFast16bfConstants);
		return false;
	}
	g_fast16bf_constants[g_fast16bf_constants_count].fp32_ptr = fp32_ptr;
	g_fast16bf_constants[g_fast16bf_constants_count].bf16_ptr = bf16_ptr;
	g_fast16bf_constants[g_fast16bf_constants_count].n        = n;
	++g_fast16bf_constants_count;
	return true;
}

bool unregister_fast16bf_constant(const float* fp32_ptr)
{
	for (int i = 0; i < g_fast16bf_constants_count; ++i)
	{
		if (g_fast16bf_constants[i].fp32_ptr == fp32_ptr)
		{
			// Compact (swap with last, pop).
			g_fast16bf_constants[i] = g_fast16bf_constants[g_fast16bf_constants_count - 1];
			--g_fast16bf_constants_count;
			return true;
		}
	}
	return false;
}

// ralph-loop iter 6 (2026-05-14): FAST_16BF GEMMs dispatched on the side
// cuBLAS handle / side stream.  Use for ops that are data-independent of
// concurrent main-stream work — readout backward dE = dlogits^T · q_L is
// the headline case.  Caller is responsible for cross-stream event
// synchronization (record event on sideComputeStream() after the call,
// streamWaitEvent on computeStream before any consumer of the output).
static bool sgemm_rowmajor_fast16bf_side_impl(cublasOperation_t transa,
                                               cublasOperation_t transb,
                                               int M, int N, int K,
                                               float alpha,
                                               const float* A, int lda,
                                               const float* B, int ldb,
                                               float beta,
                                               float* C, int ldc,
                                               const char* label)
{
	if (!ensureSideHandle()) return false;
	cublasStatus_t st = cublasGemmEx(g_handleSide,
	                                 transa, transb,
	                                 N, M, K,
	                                 &alpha,
	                                 B, CUDA_R_32F, ldb,
	                                 A, CUDA_R_32F, lda,
	                                 &beta,
	                                 C, CUDA_R_32F, ldc,
	                                 CUBLAS_COMPUTE_32F_FAST_16BF,
	                                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	if (st != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] %s failed: %d (M=%d N=%d K=%d)\n",
		        label, static_cast<int>(st), M, N, K);
		return false;
	}
	return true;
}

bool sgemm_rowmajor_atb_fast16bf_side(int M, int N, int K,
                                       float alpha,
                                       const float* A, int lda,
                                       const float* B, int ldb,
                                       float beta,
                                       float* C, int ldc)
{
	return sgemm_rowmajor_fast16bf_side_impl(CUBLAS_OP_N, CUBLAS_OP_T,
	                                          M, N, K,
	                                          alpha, A, lda, B, ldb, beta, C, ldc,
	                                          "cublasGemmEx(FAST_16BF,ATB,side)");
}

// ralph-loop iter 8 (2026-05-14): non-ATB FAST_16BF variant on side handle.
// Used by SCFA branch-parallel forward step 2 (y_par/q_par = B · *).
bool sgemm_rowmajor_fast16bf_side(int M, int N, int K,
                                   float alpha,
                                   const float* A, int lda,
                                   const float* B, int ldb,
                                   float beta,
                                   float* C, int ldc)
{
	return sgemm_rowmajor_fast16bf_side_impl(CUBLAS_OP_N, CUBLAS_OP_N,
	                                          M, N, K,
	                                          alpha, A, lda, B, ldb, beta, C, ldc,
	                                          "cublasGemmEx(FAST_16BF,side)");
}

// Phase 2f exploration: mixed-precision FP32×BF16 GEMM wrappers were
// prototyped here but cublasGemmEx (through CUDA 12.x) does not accept
// mismatched input dtypes for matrix multiply.  Silent correctness failures
// result — removed.  Stiefel Phase 2f uses a persistent FP32 cache on the
// GpuStiefelWeight struct instead, refreshed once per Adam step.

// ralph-loop iter 10 (2026-05-14): BF16-in / BF16-out ABT GEMM.  cuBLAS
// rejects FP32-in / BF16-out with CUBLAS_COMPUTE_32F_FAST_16BF (returns
// CUBLAS_STATUS_NOT_SUPPORTED = 15).  Workaround: pre-cast both inputs to
// BF16 (the E_bf_cache and q_L_bf scratches that backward already needs are
// reused so no new allocations).  FP32 accumulate via BF16-TC, BF16 store
// with RNE rounding.  This is the linchpin of --bf16-logits-storage:
// materializes the (T × V) readout logits tensor in BF16 from the start.
bool sgemm_rowmajor_abt_bf16_bf16out(int M, int N, int K,
                                      float alpha,
                                      const unsigned short* A, int lda,
                                      const unsigned short* B, int ldb,
                                      float beta,
                                      unsigned short* C, int ldc)
{
	if (!g_initialized && !blasInit()) return false;
	cublasStatus_t st = cublasGemmEx(g_handle,
	                                 CUBLAS_OP_T, CUBLAS_OP_N,
	                                 N, M, K,
	                                 &alpha,
	                                 B, CUDA_R_16BF, ldb,
	                                 A, CUDA_R_16BF, lda,
	                                 &beta,
	                                 C, CUDA_R_16BF, ldc,
	                                 CUBLAS_COMPUTE_32F_FAST_16BF,
	                                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	if (st != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] cublasGemmEx(BF16,ABT,BF16out) "
		                "failed: %d (M=%d N=%d K=%d)\n",
		        static_cast<int>(st), M, N, K);
		return false;
	}
	return true;
}

// === Batched-strided BF16 GEMMs (attention path) ===
//
// Mirrors the FP32 batched-strided variants but uses
// cublasGemmStridedBatchedEx with CUDA_R_16BF inputs and FP32 accumulate
// on BF16 tensor cores (CUBLAS_COMPUTE_32F_FAST_16BF). Throughput on
// Ampere/Ada is ~2x the TF32-tensor-core SGEMM path.
//
// Inputs are unsigned short (BF16 bit patterns); outputs remain FP32.
// Caller must pre-cast FP32 tensors to BF16 via cast_f32_to_bf16 once
// per forward (cheap relative to the attention FLOP cost).

static bool gemmex_sb_bf16_impl(cublasOperation_t transa, cublasOperation_t transb,
                                int M, int N, int K,
                                float alpha,
                                const unsigned short* A, int lda, long long strideA,
                                const unsigned short* B, int ldb, long long strideB,
                                float beta,
                                float* C, int ldc, long long strideC,
                                int batchCount,
                                const char* label)
{
	if (!g_initialized && !blasInit())
		return false;
	cublasStatus_t st = cublasGemmStridedBatchedEx(g_handle,
	    transa, transb,
	    N, M, K,
	    &alpha,
	    B, CUDA_R_16BF, ldb, strideB,
	    A, CUDA_R_16BF, lda, strideA,
	    &beta,
	    C, CUDA_R_32F, ldc, strideC,
	    batchCount,
	    CUBLAS_COMPUTE_32F_FAST_16BF,
	    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	if (st != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] %s failed: %d (M=%d N=%d K=%d bc=%d)\n",
		        label, static_cast<int>(st), M, N, K, batchCount);
		return false;
	}
	return true;
}

bool sgemm_batched_strided_bf16(int M, int N, int K,
                                float alpha,
                                const unsigned short* A, int lda, long long strideA,
                                const unsigned short* B, int ldb, long long strideB,
                                float beta,
                                float* C, int ldc, long long strideC,
                                int batchCount)
{
	return gemmex_sb_bf16_impl(CUBLAS_OP_N, CUBLAS_OP_N,
	                            M, N, K,
	                            alpha, A, lda, strideA, B, ldb, strideB,
	                            beta, C, ldc, strideC,
	                            batchCount,
	                            "cublasGemmStridedBatchedEx(BF16)");
}

bool sgemm_batched_strided_abt_bf16(int M, int N, int K,
                                    float alpha,
                                    const unsigned short* A, int lda, long long strideA,
                                    const unsigned short* B, int ldb, long long strideB,
                                    float beta,
                                    float* C, int ldc, long long strideC,
                                    int batchCount)
{
	return gemmex_sb_bf16_impl(CUBLAS_OP_T, CUBLAS_OP_N,
	                            M, N, K,
	                            alpha, A, lda, strideA, B, ldb, strideB,
	                            beta, C, ldc, strideC,
	                            batchCount,
	                            "cublasGemmStridedBatchedEx(BF16,ABT)");
}

bool sgemm_batched_strided_atb_bf16(int M, int N, int K,
                                    float alpha,
                                    const unsigned short* A, int lda, long long strideA,
                                    const unsigned short* B, int ldb, long long strideB,
                                    float beta,
                                    float* C, int ldc, long long strideC,
                                    int batchCount)
{
	return gemmex_sb_bf16_impl(CUBLAS_OP_N, CUBLAS_OP_T,
	                            M, N, K,
	                            alpha, A, lda, strideA, B, ldb, strideB,
	                            beta, C, ldc, strideC,
	                            batchCount,
	                            "cublasGemmStridedBatchedEx(BF16,ATB)");
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
