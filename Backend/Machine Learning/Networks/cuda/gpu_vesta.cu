// GPU VESTA optimizer implementation.
// See gpu_vesta.h for interface documentation.

#include "gpu_vesta.h"
#include "gpu_blas.h"
#include "gpu_device.h"
#include "../training_config.h"
#include "../vesta_optimizer.h"
#include "Backend/Database/GLogger.h"

#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <cstdio>

#ifdef GLADES_HAVE_CUDA

namespace glades {
namespace gpu {

// ---------------- Device kernels ----------------

__global__ void k_exp_ell(const float* ell, unsigned int r,
                          float* expEll, float* invExpEll)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= r) return;
	const float l = ell[i];
	expEll[i] = __expf(l);
	invExpEll[i] = __expf(-l);
}

// Log-scale mirror step with spectral entropy curvature and homeostasis.
// Uses OLD ell (before update) via the passed-in value.
// Writes new ell in-place along with updated beta.
__global__ void k_log_scale_update(float* ell, float* beta,
                                   const float* ellStar,
                                   const float* Adiag,
                                   unsigned int r,
                                   float lr, float mu, float tau,
                                   float gamma, float kappa,
                                   float ellMin, float ellMax, float phiDdFloor)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= r) return;
	const float l = ell[i];
	float phi_dd = -2.0f * l - 3.0f + mu;
	if (phi_dd < phiDdFloor) phi_dd = phiDdFloor;
	const float sigma = __expf(l);
	const float denom = phi_dd * sigma * sigma;
	float lNext = l - lr * Adiag[i] / denom - lr * tau * (l - ellStar[i]);
	if (lNext < ellMin) lNext = ellMin;
	if (lNext > ellMax) lNext = ellMax;

	const float betaPrev = beta[i];
	const float betaNew = (1.0f - gamma) * betaPrev + gamma * lNext;
	float ellBlended = (1.0f - kappa) * lNext + kappa * betaNew;
	if (ellBlended < ellMin) ellBlended = ellMin;
	if (ellBlended > ellMax) ellBlended = ellMax;
	ell[i] = ellBlended;
	beta[i] = betaNew;
}

__global__ void k_extract_diag(const float* A, unsigned int r, float* Adiag)
{
	const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= r) return;
	Adiag[i] = A[i * r + i];
}

// Scale each column c of M[rows, r] by invExpEll[c].
__global__ void k_scale_cols_inv_exp_ell(float* M, unsigned int rows, unsigned int r,
                                          const float* invExpEll)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int total = rows * r;
	if (idx >= total) return;
	const unsigned int c = idx % r;
	M[idx] *= invExpEll[c];
}

// M[rows, r] -= U[rows, r] * UtM[r, r]   (projector (I - UU^T) applied)
__global__ void k_project_out_span(float* M, const float* U, const float* UtM,
                                   unsigned int rows, unsigned int r)
{
	const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= rows || c >= r) return;
	float acc = 0.0f;
	for (unsigned int a = 0; a < r; ++a)
		acc += U[i * r + a] * UtM[a * r + c];
	M[i * r + c] -= acc;
}

// URaw[i,c] = U[i,c] - lr * Omega[i,c]
__global__ void k_form_raw(const float* U, const float* Omega, float lr,
                           float* URaw, unsigned int size)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) return;
	URaw[idx] = U[idx] - lr * Omega[idx];
}

// W += (WrNew - WrOld) - lrCperp * sign(gPerp)
// Stateless complement step; matches CPU path when complementMomentumEnabled is false.
__global__ void k_apply_W_delta(float* W,
                                const float* WrNew, const float* WrOld,
                                const float* gPerp,
                                float lrCperp,
                                unsigned int size)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) return;
	const float gp = gPerp[idx];
	const float sgn = (gp > 0.0f) ? 1.0f : ((gp < 0.0f) ? -1.0f : 0.0f);
	W[idx] += (WrNew[idx] - WrOld[idx]) - lrCperp * sgn;
}

// Momentum-sign variant:
//   m = beta * m + (1-beta) * gPerp
//   W += (WrNew - WrOld) - lrCperp * sign(m)
__global__ void k_apply_W_delta_mom_sign(float* W,
                                         const float* WrNew, const float* WrOld,
                                         const float* gPerp,
                                         float* m,
                                         float beta,
                                         float lrCperp,
                                         unsigned int size)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) return;
	const float gp = gPerp[idx];
	float mi = m[idx];
	mi = beta * mi + (1.0f - beta) * gp;
	m[idx] = mi;
	const float sgn = (mi > 0.0f) ? 1.0f : ((mi < 0.0f) ? -1.0f : 0.0f);
	W[idx] += (WrNew[idx] - WrOld[idx]) - lrCperp * sgn;
}

// Raw-momentum (heavy-ball) variant:
//   m = beta * m + (1-beta) * gPerp
//   W += (WrNew - WrOld) - lrLambdaPerp * m
__global__ void k_apply_W_delta_mom_raw(float* W,
                                        const float* WrNew, const float* WrOld,
                                        const float* gPerp,
                                        float* m,
                                        float beta,
                                        float lrLambdaPerp,
                                        unsigned int size)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) return;
	const float gp = gPerp[idx];
	float mi = m[idx];
	mi = beta * mi + (1.0f - beta) * gp;
	m[idx] = mi;
	W[idx] += (WrNew[idx] - WrOld[idx]) - lrLambdaPerp * mi;
}

// Zero-init a buffer. Used when lazy-allocating complementMomentum.
__global__ void k_zero_f(float* x, unsigned int size)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) return;
	x[idx] = 0.0f;
}

// Wr[i,j] = sum_k U[i,k] * expEll[k] * V[j,k]
__global__ void k_reconstruct_rank_block(float* Wr,
                                         const float* U, const float* V,
                                         const float* expEll,
                                         unsigned int m, unsigned int n, unsigned int r)
{
	const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= m || j >= n) return;
	float acc = 0.0f;
	for (unsigned int k = 0; k < r; ++k)
		acc += U[i * r + k] * expEll[k] * V[j * r + k];
	Wr[i * n + j] = acc;
}

// gPerp[i,j] = gW[i,j] - (UA)[i,:] . V[j,:]   where UA = U * A has been precomputed.
__global__ void k_form_gperp(const float* gW, const float* UA, const float* V,
                             float* gPerp,
                             unsigned int m, unsigned int n, unsigned int r)
{
	const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= m || j >= n) return;
	float acc = 0.0f;
	for (unsigned int k = 0; k < r; ++k)
		acc += UA[i * r + k] * V[j * r + k];
	gPerp[i * n + j] = gW[i * n + j] - acc;
}

__global__ void k_zero(float* x, unsigned int size)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) return;
	x[idx] = 0.0f;
}

__global__ void k_scale(float* x, float s, unsigned int size)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) return;
	x[idx] *= s;
}

// ---------------- Host helpers ----------------

static inline bool alloc_or_check(GpuBuffer<float>& buf, size_t count)
{
	if (buf.allocated() && buf.size() == count)
		return true;
	return buf.allocate(count);
}

static bool allocate_buffers(GpuVestaWeightState& s,
                             unsigned int m, unsigned int n, unsigned int r)
{
	const unsigned int over = 8u;
	unsigned int rp = r + over;
	if (rp > m) rp = m;
	if (rp > n) rp = n;
	if (!alloc_or_check(s.U, static_cast<size_t>(m) * r)) return false;
	if (!alloc_or_check(s.V, static_cast<size_t>(n) * r)) return false;
	if (!alloc_or_check(s.ell, r)) return false;
	if (!alloc_or_check(s.beta, r)) return false;
	if (!alloc_or_check(s.ellStar, r)) return false;
	if (!alloc_or_check(s.A, static_cast<size_t>(r) * r)) return false;
	if (!alloc_or_check(s.UA, static_cast<size_t>(m) * r)) return false;
	if (!alloc_or_check(s.WrOld, static_cast<size_t>(m) * n)) return false;
	if (!alloc_or_check(s.WrNew, static_cast<size_t>(m) * n)) return false;
	if (!alloc_or_check(s.gPerp, static_cast<size_t>(m) * n)) return false;
	if (!alloc_or_check(s.Omega_U, static_cast<size_t>(m) * r)) return false;
	if (!alloc_or_check(s.Omega_V, static_cast<size_t>(n) * r)) return false;
	if (!alloc_or_check(s.URaw, static_cast<size_t>(m) * r)) return false;
	if (!alloc_or_check(s.VRaw, static_cast<size_t>(n) * r)) return false;
	if (!alloc_or_check(s.expEll, r)) return false;
	if (!alloc_or_check(s.invExpEll, r)) return false;
	if (!alloc_or_check(s.Adiag, r)) return false;
	if (!alloc_or_check(s.UtOmU, static_cast<size_t>(r) * r)) return false;
	if (!alloc_or_check(s.VtOmV, static_cast<size_t>(r) * r)) return false;
	if (!alloc_or_check(s.sketchOmega, static_cast<size_t>(n) * rp)) return false;
	if (!alloc_or_check(s.sketchY, static_cast<size_t>(m) * rp)) return false;
	if (!alloc_or_check(s.sketchB, static_cast<size_t>(rp) * n)) return false;
	return true;
}

// Thin QR on device via CPU fallback: download, orthogonalize with gramSchmidt, upload.
static bool gpu_gramSchmidt_hostfallback(float* d_Q, unsigned int m, unsigned int r)
{
	std::vector<float> host(static_cast<size_t>(m) * r, 0.0f);
	if (cudaMemcpy(&host[0], d_Q, static_cast<size_t>(m) * r * sizeof(float),
	               cudaMemcpyDeviceToHost) != cudaSuccess)
		return false;
	vesta::gramSchmidt(&host[0], m, r);
	if (cudaMemcpy(d_Q, &host[0], static_cast<size_t>(m) * r * sizeof(float),
	               cudaMemcpyHostToDevice) != cudaSuccess)
		return false;
	return true;
}

// ---------------- Public API ----------------

bool vesta_gpu_init(GpuVestaWeightState& state,
                    const float* d_W,
                    unsigned int m, unsigned int n,
                    const glades::VestaConfig& vc,
                    glades::rng::Engine& rng,
                    shmea::GLogger* /*logger*/)
{
	unsigned int r = vc.rank;
	const unsigned int dMin = (m < n) ? m : n;
	if (r > dMin) r = dMin;
	if (r == 0u) r = 1u;

	state.m = m;
	state.n = n;
	state.r = r;
	state.step = 0ULL;
	if (!allocate_buffers(state, m, n, r)) return false;

	// Reference init via CPU: download W, run CPU init, upload state.
	std::vector<float> hostW(static_cast<size_t>(m) * n, 0.0f);
	if (cudaMemcpy(&hostW[0], d_W, static_cast<size_t>(m) * n * sizeof(float),
	               cudaMemcpyDeviceToHost) != cudaSuccess)
		return false;
	vesta::WeightState cpuState;
	vesta::initWeightState(cpuState, &hostW[0], m, n, vc, rng, 0);

	if (!state.U.upload(&cpuState.U[0], static_cast<size_t>(m) * r)) return false;
	if (!state.V.upload(&cpuState.V[0], static_cast<size_t>(n) * r)) return false;
	if (!state.ell.upload(&cpuState.ell[0], r)) return false;
	if (!state.beta.upload(&cpuState.beta[0], r)) return false;
	if (!state.ellStar.upload(&cpuState.ellStar[0], r)) return false;
	state.maxExpEllPrev = cpuState.maxExpEllPrev;
	state.initialized = true;
	return true;
}

bool vesta_gpu_refresh(GpuVestaWeightState& state,
                       const float* d_W,
                       unsigned int m, unsigned int n,
                       const glades::VestaConfig& vc,
                       glades::rng::Engine& rng,
                       shmea::GLogger* /*logger*/)
{
	std::vector<float> hostW(static_cast<size_t>(m) * n, 0.0f);
	if (cudaMemcpy(&hostW[0], d_W, static_cast<size_t>(m) * n * sizeof(float),
	               cudaMemcpyDeviceToHost) != cudaSuccess)
		return false;

	vesta::WeightState cpuState;
	cpuState.m = m; cpuState.n = n; cpuState.r = state.r;
	cpuState.U.assign(static_cast<size_t>(m) * state.r, 0.0f);
	cpuState.V.assign(static_cast<size_t>(n) * state.r, 0.0f);
	cpuState.ell.assign(state.r, 0.0f);
	cpuState.beta.assign(state.r, 0.0f);
	cpuState.ellStar.assign(state.r, 0.0f);
	const unsigned int over = 8u;
	unsigned int rp = state.r + over;
	if (rp > m) rp = m;
	if (rp > n) rp = n;
	cpuState.scratch_sketchOmega.assign(static_cast<size_t>(n) * rp, 0.0f);
	cpuState.scratch_sketchY.assign(static_cast<size_t>(m) * rp, 0.0f);
	cpuState.scratch_sketchB.assign(static_cast<size_t>(rp) * n, 0.0f);
	cpuState.scratch_sketchVr.assign(static_cast<size_t>(n) * state.r, 0.0f);
	cpuState.scratch_sketchS.assign(rp, 0.0f);
	cpuState.initialized = true;

	if (!vesta::refreshSubspace(cpuState, &hostW[0], m, n, vc, rng, 0))
		return false;

	if (!state.U.upload(&cpuState.U[0], static_cast<size_t>(m) * state.r)) return false;
	if (!state.V.upload(&cpuState.V[0], static_cast<size_t>(n) * state.r)) return false;
	if (!state.ell.upload(&cpuState.ell[0], state.r)) return false;
	return true;
}

bool vesta_gpu_step(GpuVestaWeightState& state,
                    float* d_W, float* d_gW,
                    unsigned int m, unsigned int n,
                    float invBatch, float lr,
                    float /*wd1*/, float /*wd2*/, float gradScale,
                    const glades::VestaConfig& vc,
                    glades::rng::Engine& rng,
                    shmea::GLogger* /*logger*/,
                    const char* /*tag*/)
{
	if (!state.initialized || state.m != m || state.n != n) return false;
	const unsigned int r = state.r;
	const size_t mn = static_cast<size_t>(m) * n;
	const unsigned int TPB = 256;

	// Step 0: scale gradient.
	{
		const unsigned int blocks = (static_cast<unsigned int>(mn) + TPB - 1) / TPB;
		k_scale<<<blocks, TPB>>>(d_gW, gradScale * invBatch, static_cast<unsigned int>(mn));
	}

	// Step 1: maybe refresh.
	if (state.step != 0ULL && vc.tSk > 0u && (state.step % vc.tSk) == 0ULL)
	{
		if (!vesta_gpu_refresh(state, d_W, m, n, vc, rng, 0))
			return false;
	}

	// Step 2a: compute OLD expEll, invExpEll from current state.ell (pre-update).
	{
		const unsigned int blocks = (r + 63) / 64;
		k_exp_ell<<<blocks, 64>>>(state.ell.data(), r,
		                          state.expEll.data(), state.invExpEll.data());
	}

	// Step 2b: reconstruct WrOld = U diag(exp(ell_old)) V^T.
	{
		const dim3 TPB2(16, 16);
		const dim3 blocks((n + 15) / 16, (m + 15) / 16);
		k_reconstruct_rank_block<<<blocks, TPB2>>>(state.WrOld.data(),
		                                           state.U.data(), state.V.data(),
		                                           state.expEll.data(), m, n, r);
	}

	// Step 2c: UA = gW * V  [m, r]
	if (!sgemm_rowmajor(m, r, n, 1.0f, d_gW, n, state.V.data(), r,
	                    0.0f, state.UA.data(), r))
		return false;
	// Step 2d: A = U^T * UA  [r, r]
	if (!sgemm_rowmajor_atb(r, r, m, 1.0f, state.U.data(), r, state.UA.data(), r,
	                        0.0f, state.A.data(), r))
		return false;

	// Step 2e: g_perp = gW - (U * A) V^T.
	// First compute UA2 = U * A into state.UA (reusing buffer).
	if (!sgemm_rowmajor(m, r, r, 1.0f, state.U.data(), r, state.A.data(), r,
	                    0.0f, state.UA.data(), r))
		return false;
	{
		const dim3 TPB2(16, 16);
		const dim3 blocks((n + 15) / 16, (m + 15) / 16);
		k_form_gperp<<<blocks, TPB2>>>(d_gW, state.UA.data(), state.V.data(),
		                               state.gPerp.data(), m, n, r);
	}

	// Step 3: Extract diag(A) → Adiag, then log-scale update + momentum.
	{
		const unsigned int blocks = (r + 63) / 64;
		k_extract_diag<<<blocks, 64>>>(state.A.data(), r, state.Adiag.data());
		k_log_scale_update<<<blocks, 64>>>(state.ell.data(), state.beta.data(),
		                                   state.ellStar.data(), state.Adiag.data(),
		                                   r, lr, vc.mu, vc.tau,
		                                   vc.gamma, vc.kappa,
		                                   vc.ellMin, vc.ellMax, vc.phiDdFloor);
	}

	// Step 4: Stiefel QR retraction on U using OLD invExpEll (already in state.invExpEll
	//   because we haven't recomputed it after ell update).
	// Omega_U = (I - U U^T) gW V diag(invExpEll_old)  [m, r]
	if (!sgemm_rowmajor(m, r, n, 1.0f, d_gW, n, state.V.data(), r,
	                    0.0f, state.Omega_U.data(), r))
		return false;
	if (!sgemm_rowmajor_atb(r, r, m, 1.0f, state.U.data(), r, state.Omega_U.data(), r,
	                        0.0f, state.UtOmU.data(), r))
		return false;
	{
		const dim3 TPB2(16, 16);
		const dim3 blocks((r + 15) / 16, (m + 15) / 16);
		k_project_out_span<<<blocks, TPB2>>>(state.Omega_U.data(), state.U.data(),
		                                     state.UtOmU.data(), m, r);
	}
	{
		const unsigned int total = m * r;
		const unsigned int blocks = (total + TPB - 1) / TPB;
		k_scale_cols_inv_exp_ell<<<blocks, TPB>>>(state.Omega_U.data(), m, r,
		                                          state.invExpEll.data());
		k_form_raw<<<blocks, TPB>>>(state.U.data(), state.Omega_U.data(), lr,
		                            state.URaw.data(), total);
	}
	if (!gpu_gramSchmidt_hostfallback(state.URaw.data(), m, r))
		return false;

	// Stiefel retraction on V: Omega_V = (I - V V^T) gW^T U diag(invExpEll_old)
	if (!sgemm_rowmajor_atb(n, r, m, 1.0f, d_gW, n, state.U.data(), r,
	                        0.0f, state.Omega_V.data(), r))
		return false;
	if (!sgemm_rowmajor_atb(r, r, n, 1.0f, state.V.data(), r, state.Omega_V.data(), r,
	                        0.0f, state.VtOmV.data(), r))
		return false;
	{
		const dim3 TPB2(16, 16);
		const dim3 blocks((r + 15) / 16, (n + 15) / 16);
		k_project_out_span<<<blocks, TPB2>>>(state.Omega_V.data(), state.V.data(),
		                                     state.VtOmV.data(), n, r);
	}
	{
		const unsigned int total = n * r;
		const unsigned int blocks = (total + TPB - 1) / TPB;
		k_scale_cols_inv_exp_ell<<<blocks, TPB>>>(state.Omega_V.data(), n, r,
		                                          state.invExpEll.data());
		k_form_raw<<<blocks, TPB>>>(state.V.data(), state.Omega_V.data(), lr,
		                            state.VRaw.data(), total);
	}
	if (!gpu_gramSchmidt_hostfallback(state.VRaw.data(), n, r))
		return false;

	// Commit U = URaw, V = VRaw.
	if (cudaMemcpy(state.U.data(), state.URaw.data(),
	               static_cast<size_t>(m) * r * sizeof(float),
	               cudaMemcpyDeviceToDevice) != cudaSuccess) return false;
	if (cudaMemcpy(state.V.data(), state.VRaw.data(),
	               static_cast<size_t>(n) * r * sizeof(float),
	               cudaMemcpyDeviceToDevice) != cudaSuccess) return false;

	// Step 5: recompute expEll with NEW ell.
	{
		const unsigned int blocks = (r + 63) / 64;
		k_exp_ell<<<blocks, 64>>>(state.ell.data(), r,
		                          state.expEll.data(), state.invExpEll.data());
	}
	// Reconstruct WrNew.
	{
		const dim3 TPB2(16, 16);
		const dim3 blocks((n + 15) / 16, (m + 15) / 16);
		k_reconstruct_rank_block<<<blocks, TPB2>>>(state.WrNew.data(),
		                                           state.U.data(), state.V.data(),
		                                           state.expEll.data(), m, n, r);
	}

	// Step 6: compute c_perp on host via tiny download.
	std::vector<float> hostEll(r, 0.0f);
	cudaMemcpy(&hostEll[0], state.ell.data(), r * sizeof(float), cudaMemcpyDeviceToHost);
	float meanInvSigma = 0.0f;
	for (unsigned int i = 0; i < r; ++i)
		meanInvSigma += expf(-hostEll[i]);
	meanInvSigma /= static_cast<float>(r);
	const float cPerp = vc.lambdaPerp / (meanInvSigma > 1e-12f ? meanInvSigma : 1e-12f);
	const float lrCperp = lr * cPerp;

	// Step 7: apply combined tracked-block delta + complement step.
	{
		const unsigned int total = static_cast<unsigned int>(mn);
		const unsigned int blocks = (total + TPB - 1) / TPB;
		if (vc.complementMomentumEnabled)
		{
			// Lazy-allocate and zero-init the momentum buffer on first use.
			if (state.complementMomentum.size() != mn)
			{
				if (!state.complementMomentum.allocate(mn))
					return false;
				k_zero_f<<<blocks, TPB>>>(state.complementMomentum.data(), total);
			}
			if (vc.complementUseSign)
			{
				k_apply_W_delta_mom_sign<<<blocks, TPB>>>(
				    d_W, state.WrNew.data(), state.WrOld.data(),
				    state.gPerp.data(), state.complementMomentum.data(),
				    vc.complementBeta, lrCperp, total);
			}
			else
			{
				// Raw heavy-ball: scale by lr * lambdaPerp (c_perp not used
				// here — m already carries magnitude from the gradient EMA).
				k_apply_W_delta_mom_raw<<<blocks, TPB>>>(
				    d_W, state.WrNew.data(), state.WrOld.data(),
				    state.gPerp.data(), state.complementMomentum.data(),
				    vc.complementBeta, lr * vc.lambdaPerp, total);
			}
		}
		else
		{
			k_apply_W_delta<<<blocks, TPB>>>(d_W, state.WrNew.data(), state.WrOld.data(),
			                                 state.gPerp.data(), lrCperp, total);
		}
	}

	// Step 8: trust-region clamp on host.
	{
		cudaMemcpy(&hostEll[0], state.ell.data(), r * sizeof(float), cudaMemcpyDeviceToHost);
		float curMaxExpEll = expf(hostEll[0]);
		for (unsigned int i = 1; i < r; ++i)
		{
			const float c = expf(hostEll[i]);
			if (c > curMaxExpEll) curMaxExpEll = c;
		}
		if (curMaxExpEll > (1.0f + vc.rho) * state.maxExpEllPrev)
		{
			const float allowed = (1.0f + vc.rho) * state.maxExpEllPrev;
			unsigned int iMax = 0;
			for (unsigned int i = 1; i < r; ++i)
				if (hostEll[i] > hostEll[iMax]) iMax = i;
			hostEll[iMax] = logf(allowed);
			curMaxExpEll = allowed;
			cudaMemcpy(state.ell.data(), &hostEll[0], r * sizeof(float),
			           cudaMemcpyHostToDevice);
		}
		state.maxExpEllPrev = curMaxExpEll;
	}

	// Step 9: homeostasis.
	if (vc.tHom > 0u && ((state.step + 1ULL) % vc.tHom) == 0ULL)
	{
		std::vector<float> ellStarH(r, 0.0f);
		cudaMemcpy(&ellStarH[0], state.ellStar.data(), r * sizeof(float),
		           cudaMemcpyDeviceToHost);
		cudaMemcpy(&hostEll[0], state.ell.data(), r * sizeof(float),
		           cudaMemcpyDeviceToHost);
		for (unsigned int i = 0; i < r; ++i)
			ellStarH[i] = (1.0f - vc.nu) * ellStarH[i] + vc.nu * hostEll[i];
		cudaMemcpy(state.ellStar.data(), &ellStarH[0], r * sizeof(float),
		           cudaMemcpyHostToDevice);
	}

	// Step 10: zero gW.
	{
		const unsigned int total = static_cast<unsigned int>(mn);
		const unsigned int blocks = (total + TPB - 1) / TPB;
		k_zero<<<blocks, TPB>>>(d_gW, total);
	}

	cudaDeviceSynchronize();
	state.step += 1ULL;
	return true;
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
