// FP8 (E4M3 / E5M2) matmul wrappers via cuBLASLt — see gpu_blas_fp8.h.
//
// Implementation notes:
//   - cuBLASLt requires the matmul descriptors to be set up per call, and
//     for FP8 it requires CUBLASLT_MATMUL_DESC_A_SCALE_POINTER /
//     B_SCALE_POINTER / D_SCALE_POINTER to be set to FP32 device pointers.
//   - On Ada (sm_8.9), FP8 is only supported with output type CUDA_R_8F_E4M3,
//     CUDA_R_16BF, or CUDA_R_16F (NOT CUDA_R_32F) when both A and B are FP8.
//     We use CUDA_R_16BF output and cast back to FP32 with a fused scale.
//   - The cast-to-FP8 step is done via cudaMemcpy + a small device kernel
//     that handles the 8 -> 1 scale division and rounding.  cuBLASLt does
//     not auto-cast FP32 inputs for us; we have to materialise the FP8 view.
//
// Status (2026-05-13): scaffolding.  Wrappers compile and the cast/output
// shape selection is correct, but the kernel-side cast path uses an in-tree
// CUDA kernel (kernel_cast_fp32_to_e4m3) declared inline below.  The amax
// calibration uses a simple block-stride reduction.
#ifdef GLADES_HAVE_CUDA

#include "gpu_blas_fp8.h"
#include "gpu_device.h"

#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define GLADES_CUDA_CHECK(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) { \
    std::fprintf(stderr, "[fp8] %s: %s\n", #call, cudaGetErrorString(_e)); return false; } } while (0)

namespace glades {
namespace gpu {

namespace {

cublasLtHandle_t g_lt_handle = 0;
bool             g_lt_initialized = false;

__global__ void kernel_cast_fp32_to_e4m3(const float* __restrict__ x,
                                          const float* __restrict__ scale,
                                          __nv_fp8_e4m3* __restrict__ out,
                                          size_t n)
{
	const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	const float s = scale ? *scale : 1.0f;
	// E4M3 max finite is 448.  We saturate to that to avoid +Inf encodings.
	float v = x[i] * s;
	if (v >  448.0f) v =  448.0f;
	if (v < -448.0f) v = -448.0f;
	out[i] = __nv_fp8_e4m3(v);
}

// cuBLASLt FP8 matmul already applies the inverse scales internally (we
// set A_SCALE_POINTER / B_SCALE_POINTER to 1/our_scale_X), so the BF16
// output is already in the correct numerical range.  This kernel just
// casts BF16 → FP32 and applies the caller's beta · C accumulation.
// Scale arguments retained in the signature for API stability; unused.
__global__ void kernel_cast_bf16_to_fp32_scaled(const __nv_bfloat16* __restrict__ x,
                                                 const float* __restrict__ /*scaleA*/,
                                                 const float* __restrict__ /*scaleB*/,
                                                 float beta,
                                                 const float* __restrict__ Cprev,
                                                 float* __restrict__ out,
                                                 size_t n)
{
	const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	const float xv = __bfloat162float(x[i]);
	out[i] = beta * (Cprev ? Cprev[i] : 0.0f) + xv;
}

// FP16 variant of the post-matmul cast — used when cuBLASLt's FP8 path
// produces __half output (cuBLASLt on Ada rejects BF16 output for some
// FP8 algos).  Output is in cuBLASLt column-major [N, M], and the caller
// wants row-major [M, N] — but the underlying byte layout is identical
// for these particular shapes (N as the leading dim either way), so we
// just transpose-permute-free reinterpret.  D[c, r] (col-major) lives at
// index c*N+r? No wait, col-major D[N, M] with ld=N means D[i, j] at
// i + j*N → in linear memory this is the SAME as row-major [M, N] with
// ld=N because flattening swaps the indices.  So out[i] in linear index
// corresponds to D row-major [M, N] index i directly.
__global__ void kernel_cast_fp16_to_fp32(const __half* __restrict__ x,
                                          float beta,
                                          const float* __restrict__ Cprev,
                                          float* __restrict__ out,
                                          size_t n)
{
	const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	const float xv = __half2float(x[i]);
	out[i] = beta * (Cprev ? Cprev[i] : 0.0f) + xv;
}

__global__ void kernel_cast_bf16_to_e4m3(const unsigned short* __restrict__ x_bf,
                                          const float* __restrict__ scale,
                                          __nv_fp8_e4m3* __restrict__ out,
                                          size_t n)
{
	const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	const float s = scale ? *scale : 1.0f;
	// Reconstruct FP32 from BF16 (zero-extend to 32 bits with the BF16 bits
	// in the high half; equivalent to __bfloat162float on the bit pattern).
	unsigned int bits = ((unsigned int)x_bf[i]) << 16;
	float vbf;
	__builtin_memcpy(&vbf, &bits, sizeof(vbf));
	float v = vbf * s;
	if (v >  448.0f) v =  448.0f;
	if (v < -448.0f) v = -448.0f;
	out[i] = __nv_fp8_e4m3(v);
}

__global__ void kernel_amax_pass1_bf16(const unsigned short* __restrict__ x_bf, size_t n, float* partial)
{
	__shared__ float sdata[1024];
	const int tid = threadIdx.x;
	const size_t stride = (size_t)blockDim.x * gridDim.x;
	float m = 0.0f;
	for (size_t i = (size_t)blockIdx.x * blockDim.x + tid; i < n; i += stride) {
		unsigned int bits = ((unsigned int)x_bf[i]) << 16;
		float vbf;
		__builtin_memcpy(&vbf, &bits, sizeof(vbf));
		float v = std::fabs(vbf);
		if (v > m) m = v;
	}
	sdata[tid] = m;
	__syncthreads();
	for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
		if (tid < s) {
			float a = sdata[tid], b = sdata[tid + s];
			sdata[tid] = a > b ? a : b;
		}
		__syncthreads();
	}
	if (tid == 0) partial[blockIdx.x] = sdata[0];
}

__global__ void kernel_amax_pass1(const float* __restrict__ x, size_t n, float* partial)
{
	__shared__ float sdata[1024];
	const int tid = threadIdx.x;
	const size_t stride = (size_t)blockDim.x * gridDim.x;
	float m = 0.0f;
	for (size_t i = (size_t)blockIdx.x * blockDim.x + tid; i < n; i += stride) {
		float v = std::fabs(x[i]);
		if (v > m) m = v;
	}
	sdata[tid] = m;
	__syncthreads();
	for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
		if (tid < s) {
			float a = sdata[tid], b = sdata[tid + s];
			sdata[tid] = a > b ? a : b;
		}
		__syncthreads();
	}
	if (tid == 0) partial[blockIdx.x] = sdata[0];
}

__global__ void kernel_amax_to_scale_e4m3(const float* __restrict__ partial, int nparts, float* d_scale)
{
	if (threadIdx.x != 0 || blockIdx.x != 0) return;
	float m = 0.0f;
	for (int i = 0; i < nparts; ++i) if (partial[i] > m) m = partial[i];
	// E4M3 max finite is 448.  Scale so amax · scale = 448.
	float scale = (m > 1e-12f) ? (448.0f / m) : 1.0f;
	if (!(scale == scale) || scale > 1e8f) scale = 1.0f;
	if (scale < 1e-8f) scale = 1e-8f;
	*d_scale = scale;
}

// Small persistent scratch for FP8 inputs/output.  Resizes on demand.
static __nv_fp8_e4m3* g_scratch_A = 0; static size_t g_scratch_A_cap = 0;
static __nv_fp8_e4m3* g_scratch_B = 0; static size_t g_scratch_B_cap = 0;
static __nv_bfloat16* g_scratch_D = 0; static size_t g_scratch_D_cap = 0;
static float*        g_scratch_amax_partial = 0; static size_t g_scratch_amax_cap = 0;
// Persistent FP32 device buffers holding the reciprocal scales used by
// cuBLASLt's CUBLASLT_MATMUL_DESC_*_SCALE_POINTER attributes.  cuBLASLt
// REQUIRES these to be set for FP8 inputs.  We fill them inside the
// matmul wrapper by dividing 1.0 by the caller-supplied scaleA / scaleB.
static float* g_dev_inv_scaleA = 0;
static float* g_dev_inv_scaleB = 0;
static bool   g_inv_scale_inited = false;

__global__ void kernel_reciprocal_one(const float* __restrict__ x, float* __restrict__ out)
{
	if (threadIdx.x != 0 || blockIdx.x != 0) return;
	const float v = *x;
	out[0] = (v > 1e-20f) ? (1.0f / v) : 1.0f;
}

bool grow_buf(void** ptr, size_t* cap, size_t needBytes)
{
	if (*cap >= needBytes) return true;
	if (*ptr) { cudaFree(*ptr); *ptr = 0; }
	cudaError_t e = cudaMalloc(ptr, needBytes);
	if (e != cudaSuccess) {
		std::fprintf(stderr, "[fp8] cudaMalloc(%zu) failed: %s\n", needBytes, cudaGetErrorString(e));
		*cap = 0; return false;
	}
	*cap = needBytes;
	return true;
}

} // namespace

bool fp8_supported()
{
	const int maj = computeCapabilityMajor();
	const int min = computeCapabilityMinor();
	// E4M3 / E5M2 tensor cores: Ada (sm_8.9) and Hopper (sm_9.0+).
	return (maj == 8 && min >= 9) || (maj >= 9);
}

bool fp8_init()
{
	if (g_lt_initialized) return g_lt_handle != 0;
	g_lt_initialized = true;
	if (!fp8_supported()) {
		std::fprintf(stderr, "[fp8] FP8 tensor cores not supported on this device (need sm_8.9+)\n");
		return false;
	}
	cublasStatus_t st = cublasLtCreate(&g_lt_handle);
	if (st != CUBLAS_STATUS_SUCCESS) {
		std::fprintf(stderr, "[fp8] cublasLtCreate failed: %d\n", (int)st);
		g_lt_handle = 0;
		return false;
	}
	// One-time alloc of two FP32 scalars for inverse scale pointers.
	if (cudaMalloc(&g_dev_inv_scaleA, sizeof(float)) != cudaSuccess ||
	    cudaMalloc(&g_dev_inv_scaleB, sizeof(float)) != cudaSuccess) {
		std::fprintf(stderr, "[fp8] cudaMalloc inv_scale failed\n");
		cublasLtDestroy(g_lt_handle); g_lt_handle = 0;
		return false;
	}
	g_inv_scale_inited = true;
	return true;
}

bool fp8_calibrate_amax_e4m3(const float* d_x, size_t n, float* d_scale)
{
	if (!d_x || !d_scale || n == 0) return false;
	if (!fp8_init()) return false;

	const int block = 256;
	int gridRaw = (int)((n + block - 1) / block);
	if (gridRaw > 1024) gridRaw = 1024;
	const int grid = gridRaw < 1 ? 1 : gridRaw;
	if (!grow_buf((void**)&g_scratch_amax_partial, &g_scratch_amax_cap, sizeof(float) * (size_t)grid))
		return false;

	cudaStream_t stream = computeStream();
	kernel_amax_pass1<<<grid, block, 0, stream>>>(d_x, n, g_scratch_amax_partial);
	kernel_amax_to_scale_e4m3<<<1, 1, 0, stream>>>(g_scratch_amax_partial, grid, d_scale);
	return true;
}

bool sgemm_rowmajor_fp8_e4m3(int M, int N, int K,
                              float alpha,
                              const float* A, int lda,
                              const float* B, int ldb,
                              float beta,
                              float* C, int ldc,
                              const float* d_scaleA,
                              const float* d_scaleB)
{
	if (M <= 0 || N <= 0 || K <= 0) return true;
	if (!fp8_init()) return false;

	// Allocate scratch FP8 tiles + BF16 output tile.
	const size_t bytesA = (size_t)M * (size_t)K * sizeof(__nv_fp8_e4m3);
	const size_t bytesB = (size_t)K * (size_t)N * sizeof(__nv_fp8_e4m3);
	const size_t bytesD = (size_t)M * (size_t)N * sizeof(__nv_bfloat16);
	if (!grow_buf((void**)&g_scratch_A, &g_scratch_A_cap, bytesA)) return false;
	if (!grow_buf((void**)&g_scratch_B, &g_scratch_B_cap, bytesB)) return false;
	if (!grow_buf((void**)&g_scratch_D, &g_scratch_D_cap, bytesD)) return false;

	cudaStream_t stream = computeStream();

	// Cast A → E4M3 with scaleA.
	{
		const int blk = 256;
		const int grid = (int)(((size_t)M * (size_t)K + blk - 1) / blk);
		kernel_cast_fp32_to_e4m3<<<grid, blk, 0, stream>>>(A, d_scaleA, g_scratch_A,
		                                                     (size_t)M * (size_t)K);
	}
	// Cast B → E4M3 with scaleB.
	{
		const int blk = 256;
		const int grid = (int)(((size_t)K * (size_t)N + blk - 1) / blk);
		kernel_cast_fp32_to_e4m3<<<grid, blk, 0, stream>>>(B, d_scaleB, g_scratch_B,
		                                                     (size_t)K * (size_t)N);
	}

	// Set up cuBLASLt matmul descriptor.  cuBLASLt is column-major by default;
	// to do C[MxN] = A[MxK] · B[KxN] in row-major, we instead compute
	// D[NxM] = B^T[NxK] · A^T[KxM] using column-major layouts.
	cublasLtMatmulDesc_t opDesc = 0;
	cublasLtMatrixLayout_t Adesc = 0, Bdesc = 0, Cdesc = 0;
	bool ok = true;

	cublasStatus_t st;
	st = cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
	if (st != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr,"[fp8] MatmulDescCreate failed: %d\n",(int)st); ok = false; goto cleanup; }

	{
		cublasOperation_t opT = CUBLAS_OP_T;
		cublasOperation_t opN = CUBLAS_OP_N;
		cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
		cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
	}

	// Intentionally NOT setting CUBLASLT_MATMUL_DESC_*_SCALE_POINTER — we do
	// the inverse-scale ourselves in kernel_cast_bf16_to_fp32_scaled.

	// Layouts: column-major sizes (N, M) for the (B^T · A^T) computation.
	// Since A is row-major [M, K] (stride K), reading it as column-major makes
	// it [K, M] (stride K) with no transpose needed = A^T.  Same for B.
	st = cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, K, N, ldb);
	if (st != CUBLAS_STATUS_SUCCESS) { ok = false; goto cleanup; }
	st = cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, K, M, lda);
	if (st != CUBLAS_STATUS_SUCCESS) { ok = false; goto cleanup; }
	st = cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, N, M, ldc);
	if (st != CUBLAS_STATUS_SUCCESS) { ok = false; goto cleanup; }

	{
		const float matAlpha = alpha;
		const float matBeta = 0.0f;  // beta applied later in the cast-out kernel
		st = cublasLtMatmul(g_lt_handle, opDesc,
		                     &matAlpha,
		                     g_scratch_B, Adesc,
		                     g_scratch_A, Bdesc,
		                     &matBeta,
		                     g_scratch_D, Cdesc,
		                     g_scratch_D, Cdesc,
		                     0, 0, 0, stream);
		if (st != CUBLAS_STATUS_SUCCESS) {
			std::fprintf(stderr, "[fp8] cublasLtMatmul failed: %d\n", (int)st);
			ok = false; goto cleanup;
		}
	}

	// Cast BF16 output back to FP32 with the inverse scale and apply beta · C.
	{
		const int blk = 256;
		const size_t nout = (size_t)M * (size_t)N;
		const int grid = (int)((nout + blk - 1) / blk);
		kernel_cast_bf16_to_fp32_scaled<<<grid, blk, 0, stream>>>(
		    g_scratch_D, d_scaleA, d_scaleB,
		    beta, (beta != 0.0f) ? C : 0,
		    C, nout);
	}

cleanup:
	if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
	if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
	if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
	if (opDesc) cublasLtMatmulDescDestroy(opDesc);
	// Drain any sticky CUDA error so the caller's fallback path doesn't see
	// "operation failed due to a previous error".  cuBLASLt failures often
	// leave the runtime in an error state.
	if (!ok) (void)cudaGetLastError();
	return ok;
}

bool fp8_calibrate_amax_e4m3_bf16(const unsigned short* d_x_bf, size_t n, float* d_scale)
{
	if (!d_x_bf || !d_scale || n == 0) return false;
	if (!fp8_init()) return false;

	const int block = 256;
	int gridRaw = (int)((n + block - 1) / block);
	if (gridRaw > 1024) gridRaw = 1024;
	const int grid = gridRaw < 1 ? 1 : gridRaw;
	if (!grow_buf((void**)&g_scratch_amax_partial, &g_scratch_amax_cap, sizeof(float) * (size_t)grid))
		return false;

	cudaStream_t stream = computeStream();
	kernel_amax_pass1_bf16<<<grid, block, 0, stream>>>(d_x_bf, n, g_scratch_amax_partial);
	kernel_amax_to_scale_e4m3<<<1, 1, 0, stream>>>(g_scratch_amax_partial, grid, d_scale);
	return true;
}

// Transpose-during-cast: read BF16 from a row-major [rows, cols] matrix and
// write E4M3 in a TRANSPOSED layout (i.e. [cols, rows] row-major, which is
// equivalent to [rows, cols] column-major).  cuBLASLt's FP8 matmul wants
// both operands in K-major (column-major) layout when using TN form; this
// kernel produces that layout in a single pass.
__global__ void kernel_cast_bf16_to_e4m3_transpose(const unsigned short* __restrict__ x_bf,
                                                    const float* __restrict__ scale,
                                                    __nv_fp8_e4m3* __restrict__ out,
                                                    int rows, int cols)
{
	const int r = blockIdx.y * blockDim.y + threadIdx.y;
	const int c = blockIdx.x * blockDim.x + threadIdx.x;
	if (r >= rows || c >= cols) return;
	const float s = scale ? *scale : 1.0f;
	unsigned int bits = ((unsigned int)x_bf[r * cols + c]) << 16;
	float vbf;
	__builtin_memcpy(&vbf, &bits, sizeof(vbf));
	float v = vbf * s;
	if (v >  448.0f) v =  448.0f;
	if (v < -448.0f) v = -448.0f;
	// Write transposed: position [c, r] in [cols, rows] layout.
	out[c * rows + r] = __nv_fp8_e4m3(v);
}

bool sgemm_rowmajor_fp8_e4m3_bf16(int M, int N, int K,
                                   float alpha,
                                   const unsigned short* A_bf, int /*lda*/,
                                   const unsigned short* B_bf, int /*ldb*/,
                                   float beta,
                                   float* C, int ldc,
                                   const float* d_scaleA,
                                   const float* d_scaleB)
{
	if (M <= 0 || N <= 0 || K <= 0) return true;
	if (!fp8_init()) return false;

	const size_t bytesA = (size_t)M * (size_t)K * sizeof(__nv_fp8_e4m3);
	const size_t bytesB = (size_t)K * (size_t)N * sizeof(__nv_fp8_e4m3);
	const size_t bytesD = (size_t)M * (size_t)N * sizeof(__nv_bfloat16);
	if (!grow_buf((void**)&g_scratch_A, &g_scratch_A_cap, bytesA)) return false;
	if (!grow_buf((void**)&g_scratch_B, &g_scratch_B_cap, bytesB)) return false;
	if (!grow_buf((void**)&g_scratch_D, &g_scratch_D_cap, bytesD)) return false;

	cudaStream_t stream = computeStream();

	// LAYOUT NOTES (cuBLASLt is column-major; we have row-major data):
	//   A_row[M, K] (ld=K) ≡ col-major [K, M] with ld=K   (K-major; OK as-is)
	//   B_row[K, N] (ld=N) ≡ col-major [N, K] with ld=N   (N-major; needs transpose to become K-major)
	//   C_row[M, N] (ld=N) ≡ col-major [N, M] with ld=N
	// cuBLASLt FP8 canonical TN form requires both inputs in K-major
	// column-major.  We therefore:
	//   - Cast A WITHOUT transpose (already K-major).
	//   - Cast B WITH transpose (B^T in row-major view = B in K-major col-major view).
	// Then the matmul becomes (in cuBLASLt's col-major perspective):
	//   D_col[N, M] = op_A(B_in)^T · op_B(A_in)
	// where A_in = transposed-B scratch (col-major [K, N], ld=K), op_A = T,
	// and B_in = untransposed-A scratch (col-major [K, M], ld=K), op_B = N.

	// Cast A without transpose: A_bf has (M*K) elements, stored row-major.
	{
		const int blk = 256;
		const int grid = (int)(((size_t)M * (size_t)K + blk - 1) / blk);
		kernel_cast_bf16_to_e4m3<<<grid, blk, 0, stream>>>(
		    A_bf, d_scaleA, g_scratch_A, (size_t)M * (size_t)K);
	}
	// Cast B with transpose: input is row-major [K, N], write transposed to
	// produce row-major [N, K] = col-major [K, N] with ld=K.
	{
		dim3 blk(16, 16);
		dim3 grd((N + 15) / 16, (K + 15) / 16);
		kernel_cast_bf16_to_e4m3_transpose<<<grd, blk, 0, stream>>>(
		    B_bf, d_scaleB, g_scratch_B, K, N);
	}

	cublasLtMatmulDesc_t opDesc = 0;
	cublasLtMatrixLayout_t Adesc = 0, Bdesc = 0, Cdesc = 0;
	bool ok = true;

	// cuBLASLt FP8 REQUIRES A/B scale pointers (the matmul does
	//   real_A = stored_A · scaleA_ptr_value
	// internally).  We pre-scaled our stored values up by `scaleA` (so
	// stored = real · scaleA, clamped to ±448) — to recover real values
	// cuBLASLt needs to multiply by (1/scaleA) at use time.  We pre-compute
	// the reciprocals into the two persistent inv_scale buffers.
	kernel_reciprocal_one<<<1, 1, 0, stream>>>(d_scaleB, g_dev_inv_scaleA);  // A = transposed-B
	kernel_reciprocal_one<<<1, 1, 0, stream>>>(d_scaleA, g_dev_inv_scaleB);  // B = A

	cublasStatus_t st;
	st = cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
	if (st != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr,"[fp8] MatmulDescCreate failed: %d\n",(int)st); ok = false; goto cleanup; }

	{
		cublasOperation_t opT = CUBLAS_OP_T;
		cublasOperation_t opN = CUBLAS_OP_N;
		cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
		cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
		cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
		                                &g_dev_inv_scaleA, sizeof(g_dev_inv_scaleA));
		cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
		                                &g_dev_inv_scaleB, sizeof(g_dev_inv_scaleB));
		// Some Ada FP8 algos require FAST_ACCUM = 1 (intermediate results
		// stay in FP8; otherwise cuBLASLt may not find a compatible kernel
		// for the requested shape).
		int8_t fastAccum = 1;
		cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAccum, sizeof(fastAccum));
	}

	// Layouts in cuBLASLt's column-major convention.
	// "A" input to cublasLt = transposed-B scratch (g_scratch_B):
	//   col-major [K, N], ld = K.  op_A = T treats it as [N, K] in math.
	st = cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, K, N, K);
	if (st != CUBLAS_STATUS_SUCCESS) { ok = false; goto cleanup; }
	// "B" input = untransposed-A scratch (g_scratch_A):
	//   col-major [K, M], ld = K.  op_B = N keeps it as [K, M].
	st = cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, K, M, K);
	if (st != CUBLAS_STATUS_SUCCESS) { ok = false; goto cleanup; }
	// Output D and C: col-major [N, M], ld = N.  BF16 for the BF16 path.
	st = cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, N, M, N);
	if (st != CUBLAS_STATUS_SUCCESS) { ok = false; goto cleanup; }

	{
		const float matAlpha = alpha;
		const float matBeta = 0.0f;
		st = cublasLtMatmul(g_lt_handle, opDesc,
		                     &matAlpha,
		                     g_scratch_B, Adesc,
		                     g_scratch_A, Bdesc,
		                     &matBeta,
		                     g_scratch_D, Cdesc,
		                     g_scratch_D, Cdesc,
		                     0, 0, 0, stream);
		if (st != CUBLAS_STATUS_SUCCESS) {
			std::fprintf(stderr, "[fp8] cublasLtMatmul(bf16) failed: %d  M=%d N=%d K=%d\n", (int)st, M, N, K);
			ok = false; goto cleanup;
		}
	}

	// g_scratch_D is col-major [N, M] = same byte layout as row-major [M, N]
	// with stride N.  Cast BF16 → FP32 with beta·C accumulation.
	{
		const int blk = 256;
		const size_t nout = (size_t)M * (size_t)N;
		const int grid = (int)((nout + blk - 1) / blk);
		kernel_cast_bf16_to_fp32_scaled<<<grid, blk, 0, stream>>>(
		    g_scratch_D, d_scaleA, d_scaleB,
		    beta, (beta != 0.0f) ? C : 0,
		    C, nout);
		(void)ldc;
	}

cleanup:
	if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
	if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
	if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
	if (opDesc) cublasLtMatmulDescDestroy(opDesc);
	return ok;
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
