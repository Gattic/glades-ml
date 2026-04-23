// SPAREC (Sparse Post-Activation-derivative REweighted Coordinate gradient)
// GPU primitives.  See gpu_sparec.h and research/PARADIGM_SHIFT_35_DESIGN.md.
//
// Phase 1a: threshold controller (this file) + mask kernel (this file).
// Phase 1b (next iter): backward_gathered with real speedup.

#include "gpu_sparec.h"
#include "gpu_device.h"

#ifdef GLADES_HAVE_CUDA

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

namespace glades {
namespace gpu {

namespace {

// ========================================================================
// k_sparec_mask_token: one block per token.  Threads scan across the d_ff
// dimension, each thread computes |σ'(x[t,i])| > τ.  Block-scan prefix sum
// to build active_idx[t] and k_per_tok[t].  Packed bit mask mask_packed.
// ========================================================================
__global__ void k_sparec_mask_token(const float* __restrict__ sigma_prime_cache,
                                    unsigned int T, unsigned int d_ff,
                                    float tau,
                                    unsigned int* __restrict__ mask_packed,
                                    unsigned int* __restrict__ active_idx,
                                    unsigned int* __restrict__ k_per_tok,
                                    unsigned int* __restrict__ global_active_sum)
{
	extern __shared__ unsigned int smem[];
	const unsigned int t = blockIdx.x;
	const unsigned int tid = threadIdx.x;
	const unsigned int bs = blockDim.x;
	if (t >= T) return;

	const float* sp_t = sigma_prime_cache + (size_t)t * d_ff;
	const unsigned int words_per_tok = (d_ff + 31u) / 32u;
	unsigned int* mask_t = mask_packed + (size_t)t * words_per_tok;
	unsigned int* idx_t  = active_idx   + (size_t)t * d_ff;

	// Pass 1: compute per-thread active count on strided indices.
	unsigned int my_count = 0;
	for (unsigned int i = tid; i < d_ff; i += bs) {
		const float v = sp_t[i];
		const unsigned int bit = (fabsf(v) > tau) ? 1u : 0u;
		if (bit) ++my_count;
	}

	// Block reduce to total.
	smem[tid] = my_count;
	__syncthreads();
	for (unsigned int off = bs >> 1; off > 0; off >>= 1) {
		if (tid < off) smem[tid] += smem[tid + off];
		__syncthreads();
	}
	const unsigned int k_t = smem[0];
	__syncthreads();

	// Pass 2: write mask bits and active indices with scan-based output.
	// Simple approach: thread 0 accumulates write positions serially — good
	// enough at d_ff ≤ 8192 (the scan overhead is dominated by GEMMs).
	if (tid == 0) {
		unsigned int write_pos = 0;
		for (unsigned int w = 0; w < words_per_tok; ++w)
			mask_t[w] = 0u;
		for (unsigned int i = 0; i < d_ff; ++i) {
			const float v = sp_t[i];
			const bool active = (fabsf(v) > tau);
			if (active) {
				mask_t[i >> 5] |= (1u << (i & 31));
				idx_t[write_pos++] = i;
			}
		}
		k_per_tok[t] = write_pos;
		atomicAdd(global_active_sum, write_pos);
	}
}

// ========================================================================
// k_sparec_finalize_rho: reduce global_active_sum → ρ_observed scalar.
// Single block.  ρ_observed = 1 − global_active_sum / (T · d_ff).
// ========================================================================
__global__ void k_sparec_finalize_rho(const unsigned int* __restrict__ global_active_sum,
                                      unsigned int T, unsigned int d_ff,
                                      float* __restrict__ rho_observed_out)
{
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		const float active = (float)(*global_active_sum);
		const float total = (float)T * (float)d_ff;
		*rho_observed_out = 1.0f - active / total;
	}
}

// ========================================================================
// k_sparec_threshold_step: integral controller update on τ.  One block,
// one thread.
//   ρ_observed_ema ← β · ρ_observed_ema + (1 − β) · ρ_new
//   τ ← clip(τ · (1 + η · (ρ_target − ρ_observed_ema)), τ_min, τ_max)
// ========================================================================
__global__ void k_sparec_threshold_step(float* __restrict__ tau,
                                        float* __restrict__ rho_observed_ema,
                                        const float* __restrict__ rho_new,
                                        float rho_target,
                                        float eta_tau,
                                        float tau_min, float tau_max,
                                        float beta_rho)
{
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		const float rho_ema_new = beta_rho * (*rho_observed_ema)
		                        + (1.0f - beta_rho) * (*rho_new);
		*rho_observed_ema = rho_ema_new;
		const float gap = rho_target - rho_ema_new;
		float tau_new = (*tau) * (1.0f + eta_tau * gap);
		if (tau_new < tau_min) tau_new = tau_min;
		if (tau_new > tau_max) tau_new = tau_max;
		*tau = tau_new;
	}
}

} // anonymous namespace

// ===================================================================

bool sparec_compute_active_mask(const float* sigma_prime_cache,
                                unsigned int T, unsigned int d_ff,
                                float tau,
                                unsigned int* mask_packed,
                                unsigned int* active_idx,
                                unsigned int* k_per_tok,
                                float* rho_observed_out)
{
	if (!sigma_prime_cache || !mask_packed || !active_idx || !k_per_tok
	    || !rho_observed_out) return false;
	if (T == 0 || d_ff == 0) return false;

	// Allocate per-call scratch for the global active-count accumulator.
	unsigned int* d_global_active = nullptr;
	if (cudaMalloc((void**)&d_global_active, sizeof(unsigned int)) != cudaSuccess)
		return false;
	cudaMemset(d_global_active, 0, sizeof(unsigned int));

	// Launch mask kernel — one block per token, block size 128 threads.
	const unsigned int block = 128;
	const unsigned int shmem = block * sizeof(unsigned int);
	k_sparec_mask_token<<<T, block, shmem, computeStream()>>>(
	    sigma_prime_cache, T, d_ff, tau,
	    mask_packed, active_idx, k_per_tok, d_global_active);

	k_sparec_finalize_rho<<<1, 1, 0, computeStream()>>>(
	    d_global_active, T, d_ff, rho_observed_out);

	cudaStreamSynchronize(computeStream());
	cudaFree(d_global_active);

	const cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::printf("[sparec] compute_active_mask CUDA error: %s\n",
		            cudaGetErrorString(err));
		return false;
	}
	return true;
}

bool sparec_update_threshold(float* tau,
                             float* rho_observed_ema,
                             const float* rho_new,
                             float rho_target,
                             float eta_tau,
                             float tau_min, float tau_max,
                             float beta_rho)
{
	if (!tau || !rho_observed_ema || !rho_new) return false;
	k_sparec_threshold_step<<<1, 1, 0, computeStream()>>>(
	    tau, rho_observed_ema, rho_new,
	    rho_target, eta_tau, tau_min, tau_max, beta_rho);
	cudaStreamSynchronize(computeStream());
	const cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::printf("[sparec] update_threshold CUDA error: %s\n",
		            cudaGetErrorString(err));
		return false;
	}
	return true;
}

// ========================================================================
// Phase 1b stub: sparec_backward_gathered returns false for now.  Phase 2
// will implement the gather + cuBLAS SGEMM path.  For Phase 1a the trainer
// should route FFN backward through the existing dense path when SPAREC
// is active — this lets the mask kernel run, τ controller tune, and
// sparsity stats accumulate, WITHOUT the gathered-sparse GEMM speedup.
// ========================================================================
bool sparec_backward_gathered(const float* grad_sigma,
                              const float* sigma_prime_cache,
                              const float* h_in,
                              const float* W_up,
                              const unsigned int* active_idx,
                              const unsigned int* k_per_tok,
                              unsigned int T, unsigned int d_ff,
                              unsigned int d_model,
                              float* grad_W_up,
                              float* grad_h_in)
{
	(void)grad_sigma; (void)sigma_prime_cache; (void)h_in; (void)W_up;
	(void)active_idx; (void)k_per_tok;
	(void)T; (void)d_ff; (void)d_model;
	(void)grad_W_up; (void)grad_h_in;
	// Phase 1a: not implemented.  Returns false so trainer falls back to
	// dense backward — correctness preserved, speedup deferred to Phase 2.
	return false;
}

// ========================================================================
// Host reference: dense FFN backward.  Matches what sparec_backward_gathered
// will eventually compute at τ=0 (all active).  For parity tests.
// ========================================================================
void sparec_backward_dense_reference_cpu(const float* grad_sigma,
                                         const float* sigma_prime_cache,
                                         const float* h_in,
                                         const float* W_up,
                                         unsigned int T, unsigned int d_ff,
                                         unsigned int d_model,
                                         float* grad_W_up,
                                         float* grad_h_in)
{
	// Zero outputs.
	for (size_t k = 0; k < (size_t)d_ff * d_model; ++k) grad_W_up[k] = 0.0f;
	for (size_t k = 0; k < (size_t)T * d_model; ++k)    grad_h_in[k] = 0.0f;

	// Dense backward: for each (t, i) compute scalar then accumulate.
	for (unsigned int t = 0; t < T; ++t) {
		const float* gs_t = grad_sigma + (size_t)t * d_ff;
		const float* sp_t = sigma_prime_cache + (size_t)t * d_ff;
		const float* h_t  = h_in + (size_t)t * d_model;
		float* gh_t       = grad_h_in + (size_t)t * d_model;
		for (unsigned int i = 0; i < d_ff; ++i) {
			const float scalar = sp_t[i] * gs_t[i];
			float* gW_i = grad_W_up + (size_t)i * d_model;
			const float* W_i = W_up + (size_t)i * d_model;
			for (unsigned int c = 0; c < d_model; ++c) {
				gW_i[c] += scalar * h_t[c];
				gh_t[c] += scalar * W_i[c];
			}
		}
	}
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
