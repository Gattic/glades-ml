// ATC-Δ (Activation Temporal Cache with Delta updates) GPU primitives.
// See gpu_atcd.h and research/PARADIGM_SHIFT_26_DESIGN.md.

#include "gpu_atcd.h"
#include "gpu_device.h"
#include "gpu_blas.h"

#ifdef GLADES_HAVE_CUDA

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

namespace glades {
namespace gpu {

namespace {

// ========================================================================
// Fused drift-norm reduction: single-block kernel that computes
//     ratio = sqrt(Σ Δh²) / sqrt(max(Σ h² , eps_denom²))
// in one pass.  Two parallel accumulators (Δh² and h²) reduced
// cooperatively within a warp and across warps via shared memory.
// ========================================================================
__global__ void k_atcd_drift_norm(const float* __restrict__ delta_h,
                                  const float* __restrict__ h_cache,
                                  unsigned int n_elem,
                                  float eps_denom,
                                  float* __restrict__ ratio_out)
{
	extern __shared__ float smem[];
	const int tid = threadIdx.x;
	const int bs  = blockDim.x;

	float acc_delta = 0.0f;
	float acc_h     = 0.0f;
	for (unsigned int i = tid; i < n_elem; i += bs)
	{
		const float dv = delta_h[i];
		const float hv = h_cache[i];
		acc_delta += dv * dv;
		acc_h     += hv * hv;
	}
	// Warp reduce both accumulators.
	for (int off = 16; off > 0; off >>= 1) {
		acc_delta += __shfl_xor_sync(0xffffffffu, acc_delta, off);
		acc_h     += __shfl_xor_sync(0xffffffffu, acc_h,     off);
	}
	const int lane  = tid & 31;
	const int warp  = tid >> 5;
	const int nw    = (bs + 31) >> 5;
	// Pack both sums into shared memory as [warp0_delta, warp0_h, warp1_delta, ...]
	if (lane == 0) {
		smem[warp * 2 + 0] = acc_delta;
		smem[warp * 2 + 1] = acc_h;
	}
	__syncthreads();

	if (warp == 0) {
		float v_delta = (tid < nw) ? smem[tid * 2 + 0] : 0.0f;
		float v_h     = (tid < nw) ? smem[tid * 2 + 1] : 0.0f;
		for (int off = 16; off > 0; off >>= 1) {
			v_delta += __shfl_xor_sync(0xffffffffu, v_delta, off);
			v_h     += __shfl_xor_sync(0xffffffffu, v_h,     off);
		}
		if (tid == 0) {
			const float num = sqrtf(v_delta);
			const float den = sqrtf(v_h) + eps_denom;
			*ratio_out = num / den;
		}
	}
}

// ========================================================================
// Refresh-path kernel: copy h_full → h_cache, z_full → z_cache, and
// elementwise evaluate σ'(z) → sigma_prime.  Three operations fused
// into one kernel to amortize launch overhead.  One element per thread
// along row-major [T × d].
//
// Activation derivatives:
//   0 GELU:  σ'(z) = Φ(z) + z·φ(z)  where Φ is standard-normal CDF,
//                                     φ is standard-normal PDF
//            Approximation used: tanh-GELU, σ'(z) = 0.5·(1 + tanh(u)) +
//                                                 0.5·z·(1 − tanh²(u))·(√(2/π)·(1 + 0.134145·z²))
//            where u = √(2/π) · (z + 0.044715·z³)
//   1 SiLU:  σ'(z) = s + z·s·(1−s)  where s = sigmoid(z)
//   2 ReLU:  UNSUPPORTED — σ'' is distributional at the kink.  Caller
//            must check activation_kind != 2.
//   3 IDENT: σ'(z) = 1
// ========================================================================
__global__ void k_atcd_cache_refresh(const float* __restrict__ h_full,
                                     const float* __restrict__ z_full,
                                     float* __restrict__ h_cache,
                                     float* __restrict__ z_cache,
                                     float* __restrict__ sigma_prime,
                                     unsigned int n_elem,
                                     int activation_kind)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elem) return;

	const float h = h_full[i];
	const float z = z_full[i];
	h_cache[i] = h;
	z_cache[i] = z;

	float sp = 1.0f;
	if (activation_kind == 0) {
		// GELU tanh approximation derivative.
		const float k0 = 0.7978845608028654f;   // sqrt(2/π)
		const float k1 = 0.044715f;
		const float u = k0 * (z + k1 * z * z * z);
		const float tanh_u = tanhf(u);
		const float du_dz = k0 * (1.0f + 3.0f * k1 * z * z);
		const float dtanh_du = 1.0f - tanh_u * tanh_u;
		// f(z) = 0.5 · z · (1 + tanh_u);  f'(z) = 0.5 · (1 + tanh_u) + 0.5 · z · dtanh_du · du_dz
		sp = 0.5f * (1.0f + tanh_u) + 0.5f * z * dtanh_du * du_dz;
	} else if (activation_kind == 1) {
		// SiLU: x · sigmoid(x); σ'(x) = sigmoid(x) · (1 + x · (1 − sigmoid(x)))
		const float s = 1.0f / (1.0f + expf(-z));
		sp = s + z * s * (1.0f - s);
	} else if (activation_kind == 3) {
		sp = 1.0f;
	}
	// activation_kind == 2 (ReLU) intentionally leaves sp = 1 but is
	// contract-unsupported; host-side check should reject it.
	sigma_prime[i] = sp;
}

// ========================================================================
// Elementwise multiplication: out = a ⊙ b.  Used to apply σ' mask to
// the Taylor Δz output.
// ========================================================================
__global__ void k_atcd_elemwise_mul(const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    float* __restrict__ out,
                                    unsigned int n_elem)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elem) return;
	out[i] = a[i] * b[i];
}

} // anonymous namespace

// ========================================================================
// Public API implementations
// ========================================================================

bool atcd_drift_norm(const float* delta_h, const float* h_cache,
                     unsigned int T, unsigned int d,
                     float eps_denom,
                     float* ratio_out)
{
	if (delta_h == nullptr || h_cache == nullptr || ratio_out == nullptr)
		return false;
	if (T == 0u || d == 0u) return false;

	const unsigned int n_elem = T * d;
	const int block = 256;
	const int nwarps = (block + 31) >> 5;
	const size_t smemBytes = 2u * nwarps * sizeof(float);
	k_atcd_drift_norm<<<1, block, smemBytes, computeStream()>>>(
	    delta_h, h_cache, n_elem, eps_denom, ratio_out);
	return cudaGetLastError() == cudaSuccess;
}

bool atcd_cache_refresh(const float* h_full, const float* z_full,
                        float* h_cache, float* z_cache, float* sigma_prime,
                        unsigned int T, unsigned int d,
                        int activation_kind)
{
	if (h_full == nullptr || z_full == nullptr) return false;
	if (h_cache == nullptr || z_cache == nullptr || sigma_prime == nullptr)
		return false;
	if (T == 0u || d == 0u) return false;
	if (activation_kind == 2) {
		// ReLU unsupported; ATC-Δ requires C¹ activations.
		return false;
	}

	const unsigned int n_elem = T * d;
	const int block = 256;
	const int grid = (int)((n_elem + (unsigned)block - 1u) / (unsigned)block);
	k_atcd_cache_refresh<<<grid, block, 0, computeStream()>>>(
	    h_full, z_full, h_cache, z_cache, sigma_prime, n_elem, activation_kind);
	return cudaGetLastError() == cudaSuccess;
}

// ========================================================================
// Small utility kernels for power iteration: normalize a vector in place
// (returns its norm) and a matrix-vector multiply (row-major) used by
// both the forward (ΔW · v) and backward (ΔW^T · u) directions.
// ========================================================================
namespace {

__global__ void k_vec_sum_of_squares(const float* __restrict__ x,
                                     unsigned int n,
                                     float* __restrict__ sum2_out)
{
	extern __shared__ float smem[];
	const int tid = threadIdx.x;
	const int bs  = blockDim.x;
	float acc = 0.0f;
	for (unsigned int i = tid; i < n; i += bs) {
		const float v = x[i];
		acc += v * v;
	}
	for (int off = 16; off > 0; off >>= 1)
		acc += __shfl_xor_sync(0xffffffffu, acc, off);
	const int lane = tid & 31;
	const int warp = tid >> 5;
	const int nw   = (bs + 31) >> 5;
	if (lane == 0) smem[warp] = acc;
	__syncthreads();
	if (warp == 0) {
		float v = (tid < nw) ? smem[tid] : 0.0f;
		for (int off = 16; off > 0; off >>= 1)
			v += __shfl_xor_sync(0xffffffffu, v, off);
		if (tid == 0) *sum2_out = v;
	}
}

__global__ void k_vec_scale_inv_sqrt(float* __restrict__ x,
                                     unsigned int n,
                                     const float* __restrict__ sum2)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	const float s = *sum2;
	const float inv_norm = 1.0f / (sqrtf(s) + 1e-20f);
	x[i] = x[i] * inv_norm;
}

__global__ void k_vec_fill_const(float* __restrict__ x,
                                 unsigned int n, float val)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	x[i] = val;
}

__global__ void k_write_sqrt(const float* __restrict__ in, float* __restrict__ out)
{
	if (threadIdx.x == 0 && blockIdx.x == 0) *out = sqrtf(*in);
}

} // anonymous namespace

bool atcd_extract_rank1_power(const float* dW,
                              unsigned int d_in, unsigned int d_out,
                              int n_iters,
                              const float* v_init,
                              float* u_out, float* v_out, float* sigma_out)
{
	if (dW == nullptr || u_out == nullptr || v_out == nullptr || sigma_out == nullptr)
		return false;
	if (d_in == 0u || d_out == 0u || n_iters <= 0) return false;

	const int block = 256;

	// Step 1: initialize v_out (d_in) — either from v_init or uniform 1/√d_in.
	if (v_init != nullptr) {
		cudaMemcpyAsync(v_out, v_init, d_in * sizeof(float),
		                cudaMemcpyDeviceToDevice, computeStream());
	} else {
		const int grid = (int)((d_in + (unsigned)block - 1u) / (unsigned)block);
		k_vec_fill_const<<<grid, block, 0, computeStream()>>>(
		    v_out, d_in, 1.0f / sqrtf((float)d_in));
	}

	// Scratch for norm reductions (one float).
	float* d_scratch_norm = nullptr;
	cudaMalloc(&d_scratch_norm, sizeof(float));
	if (d_scratch_norm == nullptr) return false;

	// Power iteration.  dW is row-major [d_in × d_out], so:
	//   u = dW^T · v  is a GEMV with A^T v: treat dW as [d_in × d_out]
	//       u[j] = Σ_i dW[i,j] · v[i]  for j ∈ [0, d_out)
	//   v = dW · u  is GEMV with A u:
	//       v[i] = Σ_j dW[i,j] · u[j]  for i ∈ [0, d_in)
	//
	// Use sgemm_rowmajor for these as T=1 matmuls.
	//   u[d_out × 1] = dW^T[d_out × d_in] · v[d_in × 1]
	//     sgemm_rowmajor_atb(M=d_out, N=1, K=d_in, 1, dW, d_out, v, 1, 0, u, 1)
	//   v[d_in × 1]  = dW[d_in × d_out] · u[d_out × 1]
	//     sgemm_rowmajor(M=d_in, N=1, K=d_out, 1, dW, d_out, u, 1, 0, v, 1)
	const int grid_u = (int)((d_out + (unsigned)block - 1u) / (unsigned)block);
	const int grid_v = (int)((d_in  + (unsigned)block - 1u) / (unsigned)block);
	const int nwarps = (block + 31) >> 5;
	const size_t smemBytes = nwarps * sizeof(float);

	for (int it = 0; it < n_iters; ++it)
	{
		// u = dW^T · v
		if (!sgemm_rowmajor_atb((int)d_out, 1, (int)d_in,
		                        1.0f, dW, (int)d_out, v_out, 1,
		                        0.0f, u_out, 1)) {
			cudaFree(d_scratch_norm); return false;
		}
		// normalize u
		k_vec_sum_of_squares<<<1, block, smemBytes, computeStream()>>>(
		    u_out, d_out, d_scratch_norm);
		k_vec_scale_inv_sqrt<<<grid_u, block, 0, computeStream()>>>(
		    u_out, d_out, d_scratch_norm);

		// v = dW · u
		if (!sgemm_rowmajor((int)d_in, 1, (int)d_out,
		                    1.0f, dW, (int)d_out, u_out, 1,
		                    0.0f, v_out, 1)) {
			cudaFree(d_scratch_norm); return false;
		}

		// On the last iteration, compute σ = ‖v‖ before normalizing
		// (so we can write σ = sqrt(Σ v²) to sigma_out, then re-normalize).
		k_vec_sum_of_squares<<<1, block, smemBytes, computeStream()>>>(
		    v_out, d_in, d_scratch_norm);
		if (it == n_iters - 1) {
			k_write_sqrt<<<1, 1, 0, computeStream()>>>(
			    d_scratch_norm, sigma_out);
		}
		k_vec_scale_inv_sqrt<<<grid_v, block, 0, computeStream()>>>(
		    v_out, d_in, d_scratch_norm);
	}

	cudaFree(d_scratch_norm);
	return cudaGetLastError() == cudaSuccess;
}

bool atcd_taylor_weight_delta(const float* U, const float* V,
                              const float* h_cache,
                              const float* sigma_prime,
                              unsigned int T, unsigned int d_in,
                              unsigned int d_out, unsigned int r,
                              float* delta_z_scratch,
                              float* delta_h_out)
{
	if (U == nullptr || V == nullptr || h_cache == nullptr ||
	    sigma_prime == nullptr || delta_z_scratch == nullptr ||
	    delta_h_out == nullptr)
		return false;
	if (T == 0u || d_in == 0u || d_out == 0u || r == 0u) return false;

	// Step 1: tmp_r [T × r] = h_cache [T × d_in] · V [d_in × r]
	//   (row-major sgemm with standard layout)
	// Use delta_z_scratch as the tmp_r buffer's first r·T floats.
	// This works since delta_z_scratch is T × d_out ≥ T × r for r ≤ d_out.
	// If r > d_out the caller must pre-size scratch as T·max(d_out, r).
	// We reuse delta_z_scratch: first we write tmp_r at the start of scratch,
	// then overwrite it with the final Δz after sgemm 2.
	//
	// ACTUALLY: the two GEMMs need a separate tmp buffer if scratch isn't
	// large enough. Caller is expected to allocate scratch of size
	// T · max(d_out, r). To simplify, allocate a tiny scratch for tmp_r.

	// Step 1: tmp_r [T × r] = h_cache · V   (row-major)
	//   M = T, N = r, K = d_in
	//   lda = d_in, ldb = r, ldc = r
	// For simplicity: we use delta_z_scratch's first T·r entries as tmp_r.
	// This is safe if r ≤ d_out (typical: r ≤ 32, d_out ≥ 128).
	if (!sgemm_rowmajor((int)T, (int)r, (int)d_in,
	                    1.0f,
	                    h_cache, (int)d_in,
	                    V, (int)r,
	                    0.0f,
	                    delta_z_scratch, (int)r)) return false;

	// Step 2: Δz [T × d_out] = tmp_r · Uᵀ   (since U is d_out × r)
	//   Use sgemm_rowmajor_abt: C = A · Bᵀ with A=tmp_r, B=U.
	//   M = T, N = d_out, K = r
	//   lda = r, ldb = r (U has r columns), ldc = d_out
	// After this, delta_z_scratch has been CONSUMED (read from [0, T·r))
	// and the result is written to the SAME buffer.  We need separate.
	//
	// Safer: write the step-2 result to delta_h_out (FP32 output buffer)
	// temporarily, then apply σ' mask into delta_h_out in-place via the
	// elementwise kernel.

	if (!sgemm_rowmajor_abt((int)T, (int)d_out, (int)r,
	                        1.0f,
	                        delta_z_scratch, (int)r,  // tmp_r [T × r]
	                        U, (int)r,                // U [d_out × r]
	                        0.0f,
	                        delta_h_out, (int)d_out)) return false;

	// Step 3: elementwise Δh = σ' ⊙ Δz.  delta_h_out currently has Δz; mask it.
	const unsigned int n_elem = T * d_out;
	const int block = 256;
	const int grid = (int)((n_elem + (unsigned)block - 1u) / (unsigned)block);
	k_atcd_elemwise_mul<<<grid, block, 0, computeStream()>>>(
	    sigma_prime, delta_h_out, delta_h_out, n_elem);
	return cudaGetLastError() == cudaSuccess;
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
