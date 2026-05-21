// GPU HELIOS optimizer implementation.
// See gpu_helios.h for interface documentation.
//
// Throughput (RTX 4080 SUPER, 2026-04-20):
//   pile_small  (dModel=512, 8 layers, seq=1024, minibatch=16):
//     AdamW GPU: 9,795 tok/s; HELIOS GPU: 9,791 tok/s (99.96%)
//   dModel=1024 (8 layers, seq=1024):
//     HELIOS GPU T=0   : 3,098 tok/s  (no noise upload)
//     HELIOS GPU T=1e-6: 3,098 tok/s  (with noise upload)
//   The host-generate-and-upload path for O-step Gaussians overlaps with
//   the preceding A/B-half kernels on the compute stream, so it contributes
//   no measurable throughput overhead at these scales. At dModel>=4096 an
//   on-device curand RNG would reduce CPU generation time, but measurement
//   at dModel=1024 shows the upload is already compute-overlapped.
//
// Parity (helios-test.cpp, 2026-04-20):
//   Deterministic (T=0, 20 steps, 32x24 matrix):
//     max |W_cpu - W_gpu| = 0.0,  max |p_cpu - p_gpu| = 0.0  (bit-exact).
//   Stochastic (T=1e-3, 15 steps, 24x20 matrix, matched RNG):
//     max |W_cpu - W_gpu| = 5.96e-8  (FP rounding).

#include "gpu_helios.h"
#include "gpu_device.h"
#include "../training_config.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <vector>

#ifdef GLADES_HAVE_CUDA

namespace glades {
namespace gpu {

// ---------------- Device kernels ----------------

// Sets *flag = 1 (atomicOr) if any element of x is NaN or +-Inf.
// Branchless finite check: (v * 0.0f) == 0.0f is true iff v is finite.
__global__ void k_helios_check_finite(const float* x, unsigned int size,
                                      int* flag)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) return;
	const float v = x[idx];
	if (!(v * 0.0f == 0.0f))
		atomicOr(flag, 1);
}

// W[i] *= s  (used for decoupled weight decay: s = 1 - lr*wd2).
__global__ void k_helios_scale(float* W, float s, unsigned int size)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) return;
	W[idx] *= s;
}

// B-half: p[i] -= halfH * gScale * gW[i].
__global__ void k_helios_b_half(float* p, const float* gW,
                                float coeff, unsigned int size)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) return;
	p[idx] -= coeff * gW[idx];
}

// Final B-half fused with zeroing gW (caller owns zeroing, but HELIOS
// reuses gW in both B-halves so we zero at the end).
__global__ void k_helios_b_half_zero(float* p, float* gW,
                                     float coeff, unsigned int size)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) return;
	const float g = gW[idx];
	p[idx] -= coeff * g;
	gW[idx] = 0.0f;
}

// A-half: W[i] += halfOverMass * p[i].
__global__ void k_helios_a_half(float* W, const float* p,
                                float coeff, unsigned int size)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) return;
	W[idx] += coeff * p[idx];
}

// O-step deterministic (noiseScale == 0): p[i] = c * p[i].
__global__ void k_helios_o_step_det(float* p, float c, unsigned int size)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) return;
	p[idx] *= c;
}

// O-step stochastic: p[i] = c * p[i] + noiseScale * zeta[i].
__global__ void k_helios_o_step_noise(float* p, const float* zeta,
                                      float c, float noiseScale,
                                      unsigned int size)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) return;
	p[idx] = c * p[idx] + noiseScale * zeta[idx];
}

// Anchor EMA: tb[i] = beta * tb[i] + (1 - beta) * W[i].
__global__ void k_helios_anchor_ema(float* tb, const float* W,
                                    float beta, unsigned int size)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) return;
	tb[idx] = beta * tb[idx] + (1.0f - beta) * W[idx];
}

// Single-int flag zero (used to clear finiteFlag before each check).
__global__ void k_helios_clear_flag(int* flag)
{
	if (blockIdx.x != 0u || threadIdx.x != 0u) return;
	*flag = 0;
}

// ---------------- Public API ----------------

bool helios_gpu_init(GpuHeliosWeightState& state,
                     const float* /*d_W*/,
                     unsigned int m, unsigned int n,
                     const glades::HeliosConfig& hc,
                     glades::rng::Engine& /*rng*/,
                     shmea::GLogger* /*logger*/)
{
	const size_t P = static_cast<size_t>(m) * n;
	state.m = m;
	state.n = n;

	if (!state.p.allocated() || state.p.size() != P)
	{
		if (!state.p.allocate(P)) return false;
	}
	// Zero-init p.
	if (cudaMemsetAsync(state.p.data(), 0, P * sizeof(float),
	                    glades::gpu::computeStream()) != cudaSuccess)
		return false;

	if (hc.lambdaAnchor > 0.0f)
	{
		if (!state.thetaBar.allocated() || state.thetaBar.size() != P)
		{
			if (!state.thetaBar.allocate(P)) return false;
		}
		if (cudaMemsetAsync(state.thetaBar.data(), 0, P * sizeof(float),
		                    glades::gpu::computeStream()) != cudaSuccess)
			return false;
	}

	if (!state.finiteFlag.allocated())
	{
		if (!state.finiteFlag.allocate(1)) return false;
	}

	state.xi = 0.0f;
	state.Tcurr = hc.T0;
	state.kappa = 0.0f;
	state.mass = (hc.mass > 0.0f) ? hc.mass : 1.0f;
	state.step = 0ULL;
	state.initialized = true;
	return true;
}

// Single-element download on the compute stream.
static inline bool read_flag_sync(GpuBuffer<int>& flagBuf, int& outValue)
{
	const cudaStream_t cs = glades::gpu::computeStream();
	int host = 0;
	if (cudaMemcpyAsync(&host, flagBuf.data(), sizeof(int),
	                    cudaMemcpyDeviceToHost, cs) != cudaSuccess)
		return false;
	if (cudaStreamSynchronize(cs) != cudaSuccess) return false;
	outValue = host;
	return true;
}

bool helios_gpu_step(GpuHeliosWeightState& state,
                     float* d_W, float* d_gW,
                     unsigned int m, unsigned int n,
                     float invBatch, float lr,
                     float /*wd1*/, float wd2, float gradScale,
                     const glades::HeliosConfig& hc,
                     glades::rng::Engine& rng,
                     shmea::GLogger* /*logger*/,
                     const char* /*tag*/)
{
	if (!state.initialized) return false;
	if (state.m != m || state.n != n) return false;
	const size_t P = static_cast<size_t>(m) * n;
	if (P == 0) return false;

	const float hEff = lr * hc.h;
	if (!(hEff > 0.0f)) return false;
	const float mass = state.mass;
	if (!(mass > 0.0f)) return false;

	// MVI guard: thermostat not yet supported on GPU (requires a kinetic
	// reduction + xi update; we keep xi at 0 here). Sharpness feedback
	// likewise disabled. These are both zero by default in HeliosConfig.
	if (hc.Q > 0.0f || hc.alpha > 0.0f || hc.kHvp > 0u)
		return false;

	const cudaStream_t cs = glades::gpu::computeStream();
	const unsigned int TPB = 256;
	const unsigned int blocks = (static_cast<unsigned int>(P) + TPB - 1) / TPB;

	// --- Non-finite guard on gW.
	k_helios_clear_flag<<<1, 1, 0, cs>>>(state.finiteFlag.data());
	k_helios_check_finite<<<blocks, TPB, 0, cs>>>(d_gW,
	                                              static_cast<unsigned int>(P),
	                                              state.finiteFlag.data());
	int flag = 0;
	if (!read_flag_sync(state.finiteFlag, flag)) return false;
	if (flag != 0) return false;

	// --- Decoupled weight decay: W *= (1 - lr * wd2).
	if (wd2 > 0.0f)
	{
		const float decay = 1.0f - lr * wd2;
		k_helios_scale<<<blocks, TPB, 0, cs>>>(d_W, decay,
		                                       static_cast<unsigned int>(P));
	}

	const float halfH = 0.5f * hEff;
	const float halfOverMass = halfH / mass;
	const float gScale = invBatch * gradScale;
	const float bCoeff = halfH * gScale;

	// Effective temperature with optional Li-Sato-Tan correction.
	const float Teff = state.Tcurr + 0.25f * hEff * hc.noiseCorrection;

	// O-step constants.
	const float gammaBase = hc.gamma0 + state.xi + hc.alpha * state.kappa;
	const float gammaEff = (gammaBase < 0.0f) ? 0.0f : gammaBase;
	const float c = std::exp(-gammaEff * hEff);
	const float noiseVar = (Teff > 0.0f) ? mass * Teff * (1.0f - c * c) : 0.0f;
	const float noiseScale = (noiseVar > 0.0f) ? std::sqrt(noiseVar) : 0.0f;

	// --- B-half: p -= bCoeff * gW.
	k_helios_b_half<<<blocks, TPB, 0, cs>>>(state.p.data(), d_gW, bCoeff,
	                                        static_cast<unsigned int>(P));

	// --- A-half: W += halfOverMass * p.
	k_helios_a_half<<<blocks, TPB, 0, cs>>>(d_W, state.p.data(), halfOverMass,
	                                        static_cast<unsigned int>(P));

	// --- O-step.
	if (noiseScale > 0.0f)
	{
		// Generate host-side Gaussians with the provided rng engine so RNG
		// consumption order matches the CPU path element-for-element. Allocate
		// scratch lazily.
		if (!state.noise.allocated() || state.noise.size() != P)
		{
			if (!state.noise.allocate(P)) return false;
		}
		std::vector<float> hostZ(P, 0.0f);
		for (size_t i = 0; i < P; ++i)
			hostZ[i] = glades::rng::standard_normal(rng);
		if (!state.noise.upload(&hostZ[0], P)) return false;
		k_helios_o_step_noise<<<blocks, TPB, 0, cs>>>(state.p.data(),
		                                              state.noise.data(),
		                                              c, noiseScale,
		                                              static_cast<unsigned int>(P));
	}
	else
	{
		k_helios_o_step_det<<<blocks, TPB, 0, cs>>>(state.p.data(), c,
		                                            static_cast<unsigned int>(P));
	}

	// --- A-half again.
	k_helios_a_half<<<blocks, TPB, 0, cs>>>(d_W, state.p.data(), halfOverMass,
	                                        static_cast<unsigned int>(P));

	// --- B-half (same gW) fused with zeroing gW for the next accumulation.
	k_helios_b_half_zero<<<blocks, TPB, 0, cs>>>(state.p.data(), d_gW, bCoeff,
	                                             static_cast<unsigned int>(P));

	// --- Anchor EMA (if enabled).
	if (hc.lambdaAnchor > 0.0f && state.thetaBar.allocated()
	    && state.thetaBar.size() == P)
	{
		k_helios_anchor_ema<<<blocks, TPB, 0, cs>>>(state.thetaBar.data(),
		                                            d_W, hc.betaAnchor,
		                                            static_cast<unsigned int>(P));
	}

	// --- Post-step non-finite guard on W and p.
	k_helios_clear_flag<<<1, 1, 0, cs>>>(state.finiteFlag.data());
	k_helios_check_finite<<<blocks, TPB, 0, cs>>>(d_W,
	                                              static_cast<unsigned int>(P),
	                                              state.finiteFlag.data());
	k_helios_check_finite<<<blocks, TPB, 0, cs>>>(state.p.data(),
	                                              static_cast<unsigned int>(P),
	                                              state.finiteFlag.data());
	if (!read_flag_sync(state.finiteFlag, flag)) return false;
	if (flag != 0) return false;

	state.step += 1ULL;
	return true;
}

// ---------------- FD-HVP probe kernels ----------------

// v[i] = p[i] * invNorm. Caller computes invNorm on the host side from
// the returned pNorm2 value. Two-pass (reduction then scale) is cheap at
// ~1M element scale.
__global__ void k_helios_probe_pnorm2(const float* __restrict__ p,
                                      unsigned int size, float* pNorm2Out)
{
	extern __shared__ float sdata[];
	const unsigned int tid = threadIdx.x;
	float acc = 0.0f;
	for (unsigned int i = blockIdx.x * blockDim.x + tid; i < size;
	     i += gridDim.x * blockDim.x)
	{
		const float v = p[i];
		acc += v * v;
	}
	sdata[tid] = acc;
	__syncthreads();
	for (unsigned int s = blockDim.x / 2u; s > 0u; s >>= 1u)
	{
		if (tid < s) sdata[tid] += sdata[tid + s];
		__syncthreads();
	}
	if (tid == 0u) atomicAdd(pNorm2Out, sdata[0]);
}

__global__ void k_helios_probe_scale_v(const float* __restrict__ p,
                                       float invNorm,
                                       float* __restrict__ vOut,
                                       unsigned int size)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) return;
	vOut[idx] = p[idx] * invNorm;
}

__global__ void k_helios_probe_perturb_kernel(
    float* __restrict__ W,
    const float* __restrict__ Wsave,
    const float* __restrict__ v,
    float epsSigned, unsigned int size)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) return;
	W[idx] = Wsave[idx] + epsSigned * v[idx];
}

// Reduction: acc += v[i] * (gPlus[i] - gMinus[i]).
__global__ void k_helios_probe_kappa_reduce(
    const float* __restrict__ v,
    const float* __restrict__ gPlus,
    const float* __restrict__ gMinus,
    unsigned int size, float* accOut)
{
	extern __shared__ float sdata[];
	const unsigned int tid = threadIdx.x;
	float acc = 0.0f;
	for (unsigned int i = blockIdx.x * blockDim.x + tid; i < size;
	     i += gridDim.x * blockDim.x)
	{
		acc += v[i] * (gPlus[i] - gMinus[i]);
	}
	sdata[tid] = acc;
	__syncthreads();
	for (unsigned int s = blockDim.x / 2u; s > 0u; s >>= 1u)
	{
		if (tid < s) sdata[tid] += sdata[tid + s];
		__syncthreads();
	}
	if (tid == 0u) atomicAdd(accOut, sdata[0]);
}

// ---------------- Probe host API ----------------

bool helios_gpu_probe_snapshot_W(float* d_Wsave, const float* d_W,
                                 unsigned int m, unsigned int n)
{
	if (!d_Wsave || !d_W) return false;
	const size_t bytes = static_cast<size_t>(m) * n * sizeof(float);
	if (bytes == 0) return true;
	return cudaMemcpyAsync(d_Wsave, d_W, bytes, cudaMemcpyDeviceToDevice,
	                       glades::gpu::computeStream()) == cudaSuccess;
}

bool helios_gpu_probe_restore_W(float* d_W, const float* d_Wsave,
                                unsigned int m, unsigned int n)
{
	return helios_gpu_probe_snapshot_W(d_W, d_Wsave, m, n);
}

bool helios_gpu_probe_compute_v(const float* d_p, float* d_vOut,
                                unsigned int m, unsigned int n,
                                float& pNormOut)
{
	const size_t N = static_cast<size_t>(m) * n;
	if (N == 0) { pNormOut = 0.0f; return false; }
	const cudaStream_t cs = glades::gpu::computeStream();

	// Allocate a small device scalar for reduction accumulation.
	float* d_acc = 0;
	if (cudaMalloc(&d_acc, sizeof(float)) != cudaSuccess) return false;
	if (cudaMemsetAsync(d_acc, 0, sizeof(float), cs) != cudaSuccess)
	{
		cudaFree(d_acc); return false;
	}

	const unsigned int TPB = 256;
	unsigned int blocks =
	    (static_cast<unsigned int>(N) + TPB - 1) / TPB;
	// Cap block count for the atomicAdd reduction to a reasonable size.
	if (blocks > 1024u) blocks = 1024u;
	k_helios_probe_pnorm2<<<blocks, TPB, TPB * sizeof(float), cs>>>(
	    d_p, static_cast<unsigned int>(N), d_acc);

	float pNorm2 = 0.0f;
	if (cudaMemcpyAsync(&pNorm2, d_acc, sizeof(float),
	                    cudaMemcpyDeviceToHost, cs) != cudaSuccess)
	{
		cudaFree(d_acc); return false;
	}
	cudaStreamSynchronize(cs);
	cudaFree(d_acc);

	if (!(pNorm2 > 0.0f))
	{
		pNormOut = 0.0f;
		return false;
	}
	pNormOut = std::sqrt(pNorm2);
	const float invNorm = 1.0f / pNormOut;

	const unsigned int blocks2 =
	    (static_cast<unsigned int>(N) + TPB - 1) / TPB;
	k_helios_probe_scale_v<<<blocks2, TPB, 0, cs>>>(
	    d_p, invNorm, d_vOut, static_cast<unsigned int>(N));
	return cudaGetLastError() == cudaSuccess;
}

bool helios_gpu_probe_perturb(float* d_W, const float* d_Wsave,
                              const float* d_v, float eps, float sign,
                              unsigned int m, unsigned int n)
{
	if (!d_W || !d_Wsave || !d_v || !(eps > 0.0f)) return false;
	const size_t N = static_cast<size_t>(m) * n;
	if (N == 0) return true;
	const cudaStream_t cs = glades::gpu::computeStream();
	const unsigned int TPB = 256;
	const unsigned int blocks =
	    (static_cast<unsigned int>(N) + TPB - 1) / TPB;
	k_helios_probe_perturb_kernel<<<blocks, TPB, 0, cs>>>(
	    d_W, d_Wsave, d_v, eps * sign, static_cast<unsigned int>(N));
	return cudaGetLastError() == cudaSuccess;
}

bool helios_gpu_probe_compute_kappa(const float* d_v,
                                    const float* d_gPlus,
                                    const float* d_gMinus,
                                    unsigned int m, unsigned int n,
                                    float eps, float& kappaOut)
{
	const size_t N = static_cast<size_t>(m) * n;
	if (N == 0 || !(eps > 0.0f))
	{
		kappaOut = 0.0f;
		return false;
	}
	const cudaStream_t cs = glades::gpu::computeStream();

	float* d_acc = 0;
	if (cudaMalloc(&d_acc, sizeof(float)) != cudaSuccess) return false;
	if (cudaMemsetAsync(d_acc, 0, sizeof(float), cs) != cudaSuccess)
	{
		cudaFree(d_acc); return false;
	}

	const unsigned int TPB = 256;
	unsigned int blocks =
	    (static_cast<unsigned int>(N) + TPB - 1) / TPB;
	if (blocks > 1024u) blocks = 1024u;
	k_helios_probe_kappa_reduce<<<blocks, TPB, TPB * sizeof(float), cs>>>(
	    d_v, d_gPlus, d_gMinus, static_cast<unsigned int>(N), d_acc);

	float acc = 0.0f;
	if (cudaMemcpyAsync(&acc, d_acc, sizeof(float),
	                    cudaMemcpyDeviceToHost, cs) != cudaSuccess)
	{
		cudaFree(d_acc); return false;
	}
	cudaStreamSynchronize(cs);
	cudaFree(d_acc);

	kappaOut = acc / (2.0f * eps);
	return true;
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
