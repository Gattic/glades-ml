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

// Transpose-aware variant: M[rows, r] -= U[rows, r] * A^T[r, r]
// where A is stored as [r, r] row-major (so A^T[a, c] = A[c, a]).
// Used in the V-direction Stiefel retraction where VtOmV = A^T and we avoid
// computing it explicitly.
__global__ void k_project_out_span_AT(float* M, const float* U, const float* A,
                                      unsigned int rows, unsigned int r)
{
	const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= rows || c >= r) return;
	float acc = 0.0f;
	for (unsigned int a = 0; a < r; ++a)
		acc += U[i * r + a] * A[c * r + a];  // A^T[a, c]
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

// Stateless raw path (no momentum, no sign): W += tracked delta - lr*lp * gPerp.
__global__ void k_apply_W_delta_raw(float* W,
                                    const float* WrNew, const float* WrOld,
                                    const float* gPerp,
                                    float lrLambdaPerp,
                                    unsigned int size)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) return;
	W[idx] += (WrNew[idx] - WrOld[idx]) - lrLambdaPerp * gPerp[idx];
}

// Fused reconstruct-and-update kernel.
// For each (i, j):
//   WrNew[i,j] = sum_k U_new[i,k] * expEll_new[k] * V_new[j,k]
//   WrOld[i,j] = sum_k U_old[i,k] * expEll_old[k] * V_old[j,k]
//   gPerp[i,j] = gW[i,j] - sum_k UA[i,k] * V_old[j,k]
//   (if momentum) m[i,j] = beta*m + (1-beta)*gPerp
//   cStep = sign(m) or m or sign(gPerp) or gPerp  based on variant
//   W[i,j] += (WrNew - WrOld) - cScale * cStep
//
// Variants (compile-time via template parameters):
//   UseMomentum (bool): whether to update/use the m[m*n] momentum buffer.
//   UseSign (bool):     whether to sign() the complement step (Lion-style).
//
// Eliminates the three O(m*n) scratch buffers WrOld, WrNew, gPerp previously
// materialized by three separate kernels; inner loop over k runs three fused
// accumulators of length r.
template <bool UseMomentum, bool UseSign>
__global__ void k_vesta_fused_update(
    float* __restrict__ W,
    unsigned int m, unsigned int n, unsigned int r,
    const float* __restrict__ U_old,
    const float* __restrict__ V_old,
    const float* __restrict__ expEll_old,
    const float* __restrict__ U_new,
    const float* __restrict__ V_new,
    const float* __restrict__ expEll_new,
    const float* __restrict__ UA,
    float* __restrict__ gW,
    float* __restrict__ momentum,
    float beta,
    float cScale)
{
	const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= m || j >= n) return;
	const size_t idx = static_cast<size_t>(i) * n + j;

	float accNew = 0.0f, accOld = 0.0f, accUAV = 0.0f;
	#pragma unroll 8
	for (unsigned int k = 0; k < r; ++k)
	{
		const float ek_new = expEll_new[k];
		const float ek_old = expEll_old[k];
		const float vj_new = V_new[j * r + k];
		const float vj_old = V_old[j * r + k];
		accNew += U_new[i * r + k] * ek_new * vj_new;
		accOld += U_old[i * r + k] * ek_old * vj_old;
		accUAV += UA[i * r + k] * vj_old;
	}
	const float delta = accNew - accOld;
	const float gPerp = gW[idx] - accUAV;

	float cStep;
	if (UseMomentum)
	{
		float mi = momentum[idx];
		mi = beta * mi + (1.0f - beta) * gPerp;
		momentum[idx] = mi;
		if (UseSign)
			cStep = (mi > 0.0f) ? 1.0f : ((mi < 0.0f) ? -1.0f : 0.0f);
		else
			cStep = mi;
	}
	else
	{
		if (UseSign)
			cStep = (gPerp > 0.0f) ? 1.0f : ((gPerp < 0.0f) ? -1.0f : 0.0f);
		else
			cStep = gPerp;
	}

	W[idx] += delta - cScale * cStep;
	// Zero gW for next step's gradient accumulation. Fused here so we avoid
	// a separate k_zero kernel over the full m*n buffer.
	gW[idx] = 0.0f;
}

// In-place upper Cholesky on M[r x r] row-major: writes R such that R^T R = M
// in the upper triangle; zeros the lower triangle. Single block, single thread.
// r is small (typically 4-16), so serial is optimal.
//
// Regularized: if the running diagonal would be non-positive due to round-off
// (common when U has nearly-parallel columns, e.g., at large invExpEll scaling
// in VESTA's Stiefel retraction), we clamp it to a tiny positive value so
// Cholesky always succeeds. This degrades to identity-like factorization for
// deficient columns — numerically equivalent to MGS's rank-deficiency zeroing
// for the downstream Q = U * R^(-1). Eliminates the ~45% fallback-to-MGS rate
// observed in the realistic dModel=4096 profile.
//
// Writes status[0] = 1 always (reserved for future hard failures like NaN).
__global__ void k_cholesky_small_upper(float* __restrict__ M, unsigned int r,
                                       int* __restrict__ status)
{
	if (blockIdx.x != 0u || threadIdx.x != 0u) return;
	*status = 1;
	// Diagonal regularization scale: ε ≈ 1e-6 * trace(M) / r. Small relative
	// to the typical positive diagonal; acts as a Levenberg-Marquardt nudge
	// for poorly-conditioned cases. Computed up-front so all pivots see it.
	float trace = 0.0f;
	for (unsigned int k = 0; k < r; ++k) trace += M[k * r + k];
	const float regFloor = fmaxf(1e-12f, 1e-6f * trace / static_cast<float>(r));
	for (unsigned int k = 0; k < r; ++k)
	{
		float diag = M[k * r + k];
		for (unsigned int i = 0; i < k; ++i)
		{
			const float v = M[i * r + k];
			diag -= v * v;
		}
		if (!isfinite(diag))
		{
			*status = 0;
			return;
		}
		// Clamp to regFloor to keep Cholesky well-defined.
		if (diag < regFloor) diag = regFloor;
		const float Rkk = sqrtf(diag);
		M[k * r + k] = Rkk;
		const float invRkk = 1.0f / Rkk;
		for (unsigned int j = k + 1; j < r; ++j)
		{
			float s = M[k * r + j];
			for (unsigned int i = 0; i < k; ++i)
				s -= M[i * r + k] * M[i * r + j];
			M[k * r + j] = s * invRkk;
		}
	}
	// Zero lower triangle (optional but cleaner).
	for (unsigned int i = 1; i < r; ++i)
		for (unsigned int j = 0; j < i; ++j)
			M[i * r + j] = 0.0f;
}

// Modified Gram-Schmidt for Q[m x r] stored row-major (Q[i*r + j] = row i, col j).
// Runs as a single cooperative block. One thread per row (blockDim.x = min(m, 1024));
// block-wide reduction for dot products and squared norms. This avoids the
// host roundtrip that synchronizes the GPU pipeline every step.
//
// Shared memory layout: sdata[0..blockDim.x) for reductions.
__global__ void k_gram_schmidt(float* Q, unsigned int m, unsigned int r)
{
	extern __shared__ float sdata[];
	const unsigned int tid = threadIdx.x;
	const float kTiny = 1e-12f;
	const float kRelThresh = 1e-5f;
	for (unsigned int j = 0; j < r; ++j)
	{
		// Original column norm (before orthogonalization), for rank-deficiency.
		float local = 0.0f;
		for (unsigned int i = tid; i < m; i += blockDim.x)
		{
			const float v = Q[i * r + j];
			local += v * v;
		}
		sdata[tid] = local;
		__syncthreads();
		for (unsigned int s = blockDim.x / 2u; s > 0u; s >>= 1u)
		{
			if (tid < s) sdata[tid] += sdata[tid + s];
			__syncthreads();
		}
		const float origNorm = sqrtf(sdata[0]);

		// Subtract projections onto previous normalized columns 0..j-1.
		for (unsigned int k = 0; k < j; ++k)
		{
			float d = 0.0f;
			for (unsigned int i = tid; i < m; i += blockDim.x)
				d += Q[i * r + k] * Q[i * r + j];
			sdata[tid] = d;
			__syncthreads();
			for (unsigned int s = blockDim.x / 2u; s > 0u; s >>= 1u)
			{
				if (tid < s) sdata[tid] += sdata[tid + s];
				__syncthreads();
			}
			const float dot = sdata[0];
			for (unsigned int i = tid; i < m; i += blockDim.x)
				Q[i * r + j] -= dot * Q[i * r + k];
			__syncthreads();
		}

		// Compute residual norm.
		float local2 = 0.0f;
		for (unsigned int i = tid; i < m; i += blockDim.x)
		{
			const float v = Q[i * r + j];
			local2 += v * v;
		}
		sdata[tid] = local2;
		__syncthreads();
		for (unsigned int s = blockDim.x / 2u; s > 0u; s >>= 1u)
		{
			if (tid < s) sdata[tid] += sdata[tid + s];
			__syncthreads();
		}
		const float norm = sqrtf(sdata[0]);

		// Zero out if rank-deficient (residual norm below threshold); otherwise normalize.
		if (norm <= kTiny || (origNorm > 0.0f && norm <= kRelThresh * origNorm))
		{
			for (unsigned int i = tid; i < m; i += blockDim.x)
				Q[i * r + j] = 0.0f;
		}
		else
		{
			const float inv = 1.0f / norm;
			for (unsigned int i = tid; i < m; i += blockDim.x)
				Q[i * r + j] *= inv;
		}
		__syncthreads();
	}
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
	if (!alloc_or_check(s.Omega_U, static_cast<size_t>(m) * r)) return false;
	if (!alloc_or_check(s.Omega_V, static_cast<size_t>(n) * r)) return false;
	if (!alloc_or_check(s.URaw, static_cast<size_t>(m) * r)) return false;
	if (!alloc_or_check(s.VRaw, static_cast<size_t>(n) * r)) return false;
	if (!alloc_or_check(s.expEll, r)) return false;
	if (!alloc_or_check(s.invExpEll, r)) return false;
	if (!alloc_or_check(s.expEllPrev, r)) return false;
	if (!alloc_or_check(s.Adiag, r)) return false;
	if (!alloc_or_check(s.UtOmU, static_cast<size_t>(r) * r)) return false;
	if (!alloc_or_check(s.VtOmV, static_cast<size_t>(r) * r)) return false;
	if (!s.cholStatus.allocated() && !s.cholStatus.allocate(1)) return false;
	if (!alloc_or_check(s.sketchOmega, static_cast<size_t>(n) * rp)) return false;
	if (!alloc_or_check(s.sketchY, static_cast<size_t>(m) * rp)) return false;
	if (!alloc_or_check(s.sketchB, static_cast<size_t>(rp) * n)) return false;
	return true;
}

// On-device modified Gram-Schmidt for Q[m x r]. Avoids host roundtrip.
static bool gpu_gramSchmidt_device(float* d_Q, unsigned int m, unsigned int r,
                                   cudaStream_t stream = 0)
{
	if (m == 0u || r == 0u) return true;
	unsigned int threads = 1024u;
	if (threads > m) threads = m;
	// Round down to power-of-2 for the reduction pattern.
	unsigned int pow2 = 1u;
	while ((pow2 * 2u) <= threads) pow2 *= 2u;
	threads = pow2;
	const size_t shmem = threads * sizeof(float);
	k_gram_schmidt<<<1, threads, shmem, stream>>>(d_Q, m, r);
	return cudaGetLastError() == cudaSuccess;
}

// CholQR-based orthonormalization: Q = U * R^(-1) where R^T R = U^T U.
// Produces the same Q as modified Gram-Schmidt (both are the unique thin-QR
// Q with R having positive diagonal) for full-rank U, at a fraction of the
// wall-clock cost for small r:
//   1. sgemm:   M = U^T U              [r, r]   (cuBLAS, multi-SM)
//   2. chol:    M = R^T R              [r, r]   (custom single-block, regularized)
//   3. strsm:   U := U * R^(-1)        [m, r]   (cuBLAS, multi-SM)
// Scratch: r*r float matrix + 1 int status (status unused since regularized
// Cholesky always produces usable output).
//
// Always returns true. The regularized Cholesky (adds a tiny relative epsilon
// to the running diagonal) guarantees R is invertible; for rank-deficient U
// the result degrades gracefully to a well-conditioned approximation, same
// semantics as MGS's column-zeroing for deficient cases.
static bool gpu_cholqr_device(float* d_U, unsigned int m, unsigned int r,
                              float* d_scratch_M,   // [r * r]
                              int* d_scratch_status,
                              cudaStream_t /*stream (ignored, cuBLAS uses computeStream)*/)
{
	if (m == 0u || r == 0u) return true;

	// Step 1: M = U^T U  [r, r].
	if (!sgemm_rowmajor_atb(static_cast<int>(r), static_cast<int>(r), static_cast<int>(m),
	                         1.0f, d_U, static_cast<int>(r), d_U, static_cast<int>(r),
	                         0.0f, d_scratch_M, static_cast<int>(r)))
		return false;

	// Step 2: regularized Cholesky on the r×r matrix (custom single-block).
	// Launch on computeStream so it is serialized with the cuBLAS call above.
	k_cholesky_small_upper<<<1, 1, 0, glades::gpu::computeStream()>>>(
	    d_scratch_M, r, d_scratch_status);

	// Step 3: Q = U * R^(-1). strsm: X * R = alpha * B where X overwrites B.
	if (!strsm_rowmajor_right_upper(static_cast<int>(m), static_cast<int>(r),
	                                 1.0f, d_scratch_M, static_cast<int>(r),
	                                 d_U, static_cast<int>(r)))
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

// Host-roundtrip refresh: download W, run CPU sketched SVD, upload U/V/ell.
// Used for strict CPU/GPU parity paths (parity tests). Slow at large dModel.
static bool vesta_gpu_refresh_host(GpuVestaWeightState& state,
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

// On-device refresh: keeps W resident, runs GEMMs + Gram-Schmidt on GPU,
// only downloads the small B[rp x n] matrix for the Jacobi-based SVD, uploads
// the right singular vectors back. RNG consumption matches the CPU path
// exactly (n*rp Omega draws, then per-deficient-column m+n draws in order).
// Numerical result differs from CPU-refresh by cuBLAS vs. host-SGEMM rounding,
// typically <= 1e-4 per U/V element.
static bool vesta_gpu_refresh_device(GpuVestaWeightState& state,
                                     const float* d_W,
                                     unsigned int m, unsigned int n,
                                     const glades::VestaConfig& vc,
                                     glades::rng::Engine& rng,
                                     shmea::GLogger* /*logger*/)
{
	const unsigned int r = state.r;
	const unsigned int over = 8u;
	unsigned int rp = r + over;
	if (rp > m) rp = m;
	if (rp > n) rp = n;
	if (rp < r) return false;

	// 1. Sample Omega ~ N(0,1) [n, rp] on host; upload.
	std::vector<float> hostOmega(static_cast<size_t>(n) * rp, 0.0f);
	for (size_t i = 0; i < hostOmega.size(); ++i)
		hostOmega[i] = glades::rng::standard_normal(rng);
	if (!state.sketchOmega.upload(&hostOmega[0], static_cast<size_t>(n) * rp))
		return false;

	// 2. Y = W Omega  [m, rp].  Row-major SGEMM: M=m, N=rp, K=n.
	if (!sgemm_rowmajor(static_cast<int>(m), static_cast<int>(rp), static_cast<int>(n), 1.0f,
	                    d_W, static_cast<int>(n),
	                    state.sketchOmega.data(), static_cast<int>(rp),
	                    0.0f,
	                    state.sketchY.data(), static_cast<int>(rp)))
		return false;

	// 3. Power iterations. Alternates GS → WtY → Y = W WtY.
	// GS runs on compute stream for ordering with the cuBLAS GEMMs that
	// immediately read/write Y; without this the cuBLAS call on compute stream
	// is not ordered with the default-stream GS and reads stale Y.
	const cudaStream_t cstream = glades::gpu::computeStream();
	for (unsigned int p = 0; p < vc.powerIters; ++p)
	{
		if (!gpu_gramSchmidt_device(state.sketchY.data(), m, rp, cstream)) return false;
		// WtY [n, rp] = W^T Y. ATB: M=n, N=rp, K=m. A=W[m,n] lda=n; B=Y[m,rp] ldb=rp.
		if (!sgemm_rowmajor_atb(static_cast<int>(n), static_cast<int>(rp), static_cast<int>(m), 1.0f,
		                        d_W, static_cast<int>(n),
		                        state.sketchY.data(), static_cast<int>(rp),
		                        0.0f,
		                        state.sketchOmega.data(), static_cast<int>(rp)))
			return false;
		// Y = W WtY  [m, rp]. SGEMM: M=m, N=rp, K=n.
		if (!sgemm_rowmajor(static_cast<int>(m), static_cast<int>(rp), static_cast<int>(n), 1.0f,
		                    d_W, static_cast<int>(n),
		                    state.sketchOmega.data(), static_cast<int>(rp),
		                    0.0f,
		                    state.sketchY.data(), static_cast<int>(rp)))
			return false;
	}

	// 4. Final GS on Y → orthonormal range basis.
	if (!gpu_gramSchmidt_device(state.sketchY.data(), m, rp, cstream)) return false;

	// 5. B = Y^T W  [rp, n]. ATB: M=rp, N=n, K=m. A=Y[m,rp] lda=rp; B=W[m,n] ldb=n.
	if (!sgemm_rowmajor_atb(static_cast<int>(rp), static_cast<int>(n), static_cast<int>(m), 1.0f,
	                        state.sketchY.data(), static_cast<int>(rp),
	                        d_W, static_cast<int>(n),
	                        0.0f,
	                        state.sketchB.data(), static_cast<int>(n)))
		return false;

	// 6. Download B and run the tiny SVD on CPU (Jacobi on B B^T, rp x rp).
	// Explicit sync: cuBLAS on compute stream just wrote sketchB; the
	// subsequent pageable cudaMemcpy is synchronous w.r.t. host, but we make
	// stream ordering explicit here.
	cudaStreamSynchronize(cstream);
	std::vector<float> hostB(static_cast<size_t>(rp) * n, 0.0f);
	if (cudaMemcpy(&hostB[0], state.sketchB.data(),
	               static_cast<size_t>(rp) * n * sizeof(float),
	               cudaMemcpyDeviceToHost) != cudaSuccess)
		return false;

	std::vector<float> Vrp(static_cast<size_t>(n) * rp, 0.0f);
	std::vector<float> srp(rp, 0.0f);
	if (!vesta::denseSVD_rightV(&hostB[0], rp, n, &Vrp[0], &srp[0], rp))
		return false;

	// Rank-deficiency detection: generous relative threshold (matches CPU).
	float threshBase = srp[0] * 1e-3f;
	if (threshBase < 1e-6f) threshBase = 1e-6f;
	std::vector<unsigned char> validCol(r, 0);
	for (unsigned int i = 0; i < r; ++i)
		validCol[i] = (srp[i] > threshBase) ? 1u : 0u;

	// 7. Build V_final [n, r]: first r cols of Vrp for valid, random for deficient.
	std::vector<float> Vfinal(static_cast<size_t>(n) * r, 0.0f);
	for (unsigned int j = 0; j < n; ++j)
		for (unsigned int c = 0; c < r; ++c)
			if (validCol[c])
				Vfinal[j * r + c] = Vrp[j * rp + c];

	// 8. Build V_scaled [n, r] with Vrp[:,c]/srp[c] for valid cols (zero for deficient).
	// Used to compute U_B = B * V_scaled  [rp, r]  on device.
	std::vector<float> VrpScaled(static_cast<size_t>(n) * r, 0.0f);
	for (unsigned int c = 0; c < r; ++c)
	{
		if (!validCol[c]) continue;
		const float inv = 1.0f / srp[c];
		for (unsigned int j = 0; j < n; ++j)
			VrpScaled[j * r + c] = Vrp[j * rp + c] * inv;
	}

	// Stuff VrpScaled into state.sketchOmega (size n*rp >= n*r; first n*r floats).
	if (!state.sketchOmega.upload(&VrpScaled[0], static_cast<size_t>(n) * r))
		return false;

	// 9. U_B [rp, r] = B [rp, n] * V_scaled [n, r]. SGEMM: M=rp, N=r, K=n.
	GpuBuffer<float> dUB;
	if (!dUB.allocate(static_cast<size_t>(rp) * r)) return false;
	if (!sgemm_rowmajor(static_cast<int>(rp), static_cast<int>(r), static_cast<int>(n), 1.0f,
	                    state.sketchB.data(), static_cast<int>(n),
	                    state.sketchOmega.data(), static_cast<int>(r),
	                    0.0f,
	                    dUB.data(), static_cast<int>(r)))
		return false;

	// 10. U_final [m, r] = Y [m, rp] * U_B [rp, r]. SGEMM: M=m, N=r, K=rp.
	if (!sgemm_rowmajor(static_cast<int>(m), static_cast<int>(r), static_cast<int>(rp), 1.0f,
	                    state.sketchY.data(), static_cast<int>(rp),
	                    dUB.data(), static_cast<int>(r),
	                    0.0f,
	                    state.U.data(), static_cast<int>(r)))
		return false;

	// 11. Write Vfinal to device.
	if (!state.V.upload(&Vfinal[0], static_cast<size_t>(n) * r)) return false;

	// 12. Rank-deficient columns: CPU fills U, V with fresh Gaussians in
	// per-column (m for U, then n for V) order to match CPU RNG consumption.
	bool anyDeficient = false;
	for (unsigned int c = 0; c < r; ++c) if (!validCol[c]) { anyDeficient = true; break; }
	if (anyDeficient)
	{
		std::vector<float> hostU(static_cast<size_t>(m) * r, 0.0f);
		if (!state.U.download(&hostU[0], static_cast<size_t>(m) * r)) return false;
		// Vfinal is already on host (from step 7 construction — we'll re-upload).
		for (unsigned int c = 0; c < r; ++c)
		{
			if (validCol[c]) continue;
			for (unsigned int i = 0; i < m; ++i)
				hostU[i * r + c] = glades::rng::standard_normal(rng);
			for (unsigned int j = 0; j < n; ++j)
				Vfinal[j * r + c] = glades::rng::standard_normal(rng);
		}
		if (!state.U.upload(&hostU[0], static_cast<size_t>(m) * r)) return false;
		if (!state.V.upload(&Vfinal[0], static_cast<size_t>(n) * r)) return false;
	}

	// 13. Orthonormalize U and V on device (matches CPU final gramSchmidt passes).
	if (!gpu_gramSchmidt_device(state.U.data(), m, r, cstream)) return false;
	if (!gpu_gramSchmidt_device(state.V.data(), n, r, cstream)) return false;

	// 14. Set ell = clamp(log(sigma)) on host → upload.
	std::vector<float> hostEll(r, 0.0f);
	for (unsigned int i = 0; i < r; ++i)
	{
		float l = vc.ellMin;
		if (validCol[i])
		{
			float si = srp[i];
			if (si < 1e-20f) si = 1e-20f;
			l = logf(si);
		}
		if (l < vc.ellMin) l = vc.ellMin;
		if (l > vc.ellMax) l = vc.ellMax;
		hostEll[i] = l;
	}
	if (!state.ell.upload(&hostEll[0], r)) return false;

	return true;
}

bool vesta_gpu_refresh(GpuVestaWeightState& state,
                       const float* d_W,
                       unsigned int m, unsigned int n,
                       const glades::VestaConfig& vc,
                       glades::rng::Engine& rng,
                       shmea::GLogger* logger)
{
	if (vc.gpuRefreshOnDevice)
		return vesta_gpu_refresh_device(state, d_W, m, n, vc, rng, logger);
	return vesta_gpu_refresh_host(state, d_W, m, n, vc, rng, logger);
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
	// Run all custom kernels on the compute stream (same as cuBLAS) so the
	// whole step is naturally serialized with the transformer's forward /
	// backward and with the next weight matrix's step. Eliminates the need
	// for a device-wide cudaDeviceSynchronize at the end of every step call.
	const cudaStream_t cs = glades::gpu::computeStream();

	// Step 0: scale gradient.
	{
		const unsigned int blocks = (static_cast<unsigned int>(mn) + TPB - 1) / TPB;
		k_scale<<<blocks, TPB, 0, cs>>>(d_gW, gradScale * invBatch, static_cast<unsigned int>(mn));
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
		k_exp_ell<<<blocks, 64, 0, cs>>>(state.ell.data(), r,
		                                 state.expEll.data(), state.invExpEll.data());
	}

	// Step 2b: save expEll_old to expEllPrev (for fused reconstruct in step 7).
	// state.expEll will be overwritten at step 5 with expEll_new.
	if (cudaMemcpyAsync(state.expEllPrev.data(), state.expEll.data(),
	                    static_cast<size_t>(r) * sizeof(float),
	                    cudaMemcpyDeviceToDevice, cs) != cudaSuccess)
		return false;

	// Step 2c: Omega_U = gW * V  [m, r].
	// This is both the tracked-space coefficient input AND (after scaling
	// and projection in step 4) the Stiefel update; computing it once into
	// state.Omega_U avoids a duplicate gW*V later.
	if (!sgemm_rowmajor(m, r, n, 1.0f, d_gW, n, state.V.data(), r,
	                    0.0f, state.Omega_U.data(), r))
		return false;
	// Step 2d: A = U^T * Omega_U = U^T * gW * V  [r, r].
	if (!sgemm_rowmajor_atb(r, r, m, 1.0f, state.U.data(), r, state.Omega_U.data(), r,
	                        0.0f, state.A.data(), r))
		return false;

	// Step 2e: Compute UA = U * A (used in step 7's fused reconstruct-and-update).
	if (!sgemm_rowmajor(m, r, r, 1.0f, state.U.data(), r, state.A.data(), r,
	                    0.0f, state.UA.data(), r))
		return false;

	// Step 3: Extract diag(A) → Adiag, then log-scale update + momentum.
	{
		const unsigned int blocks = (r + 63) / 64;
		k_extract_diag<<<blocks, 64, 0, cs>>>(state.A.data(), r, state.Adiag.data());
		k_log_scale_update<<<blocks, 64, 0, cs>>>(state.ell.data(), state.beta.data(),
		                                   state.ellStar.data(), state.Adiag.data(),
		                                   r, lr, vc.mu, vc.tau,
		                                   vc.gamma, vc.kappa,
		                                   vc.ellMin, vc.ellMax, vc.phiDdFloor);
	}

	// Step 4: Stiefel QR retraction on U (URaw = U_new; do NOT commit yet).
	// Omega_U already holds gW * V from step 2c. UtOmU = U^T * Omega_U = A
	// (same computation as step 2d), so we reuse state.A for the projection.
	{
		const dim3 TPB2(16, 16);
		const dim3 blocks((r + 15) / 16, (m + 15) / 16);
		k_project_out_span<<<blocks, TPB2, 0, cs>>>(state.Omega_U.data(), state.U.data(),
		                                     state.A.data(), m, r);
	}
	{
		const unsigned int total = m * r;
		const unsigned int blocks = (total + TPB - 1) / TPB;
		k_scale_cols_inv_exp_ell<<<blocks, TPB, 0, cs>>>(state.Omega_U.data(), m, r,
		                                          state.invExpEll.data());
		k_form_raw<<<blocks, TPB, 0, cs>>>(state.U.data(), state.Omega_U.data(), lr,
		                            state.URaw.data(), total);
	}
	// CholQR for URaw (regularized — always succeeds).
	if (!gpu_cholqr_device(state.URaw.data(), m, r,
	                        state.UtOmU.data(), state.cholStatus.data(), cs))
		return false;


	// Stiefel retraction on V: Omega_V = (I - V V^T) gW^T U diag(invExpEll_old).
	// VtOmV = V^T * Omega_V = V^T * gW^T * U = (U^T * gW * V)^T = A^T.
	// Skip the VtOmV SGEMM and apply the A^T projection directly. VRaw = V_new
	// (not committed yet — fused kernel below reads both U_old/V_old from
	// state.U/state.V and U_new/V_new from state.URaw/state.VRaw).
	if (!sgemm_rowmajor_atb(n, r, m, 1.0f, d_gW, n, state.U.data(), r,
	                        0.0f, state.Omega_V.data(), r))
		return false;
	{
		const dim3 TPB2(16, 16);
		const dim3 blocks((r + 15) / 16, (n + 15) / 16);
		k_project_out_span_AT<<<blocks, TPB2, 0, cs>>>(state.Omega_V.data(), state.V.data(),
		                                        state.A.data(), n, r);
	}
	{
		const unsigned int total = n * r;
		const unsigned int blocks = (total + TPB - 1) / TPB;
		k_scale_cols_inv_exp_ell<<<blocks, TPB, 0, cs>>>(state.Omega_V.data(), n, r,
		                                          state.invExpEll.data());
		k_form_raw<<<blocks, TPB, 0, cs>>>(state.V.data(), state.Omega_V.data(), lr,
		                            state.VRaw.data(), total);
	}
	if (!gpu_cholqr_device(state.VRaw.data(), n, r,
	                        state.VtOmV.data(), state.cholStatus.data(), cs))
		return false;

	// Step 5: recompute expEll with NEW ell (overwrites the expEll_old in
	// state.expEll; saved copy in state.expEllPrev from step 2b).
	{
		const unsigned int blocks = (r + 63) / 64;
		k_exp_ell<<<blocks, 64, 0, cs>>>(state.ell.data(), r,
		                          state.expEll.data(), state.invExpEll.data());
	}

	// Step 6: compute c_perp on host via tiny download. hostEll is reused
	// in step 8 for the trust-region clamp, so we always download.
	std::vector<float> hostEll(r, 0.0f);
	cudaMemcpy(&hostEll[0], state.ell.data(), r * sizeof(float), cudaMemcpyDeviceToHost);
	float meanInvSigma = 0.0f;
	for (unsigned int i = 0; i < r; ++i)
		meanInvSigma += expf(-hostEll[i]);
	meanInvSigma /= static_cast<float>(r);
	const float cPerp = vc.lambdaPerp / (meanInvSigma > 1e-12f ? meanInvSigma : 1e-12f);
	const float lrCperp = lr * cPerp;

	// Step 7: FUSED reconstruct-and-update.
	// One kernel computes (WrNew - WrOld) - cScale * cStep inline, without
	// materializing the O(m*n) scratch buffers WrOld/WrNew/gPerp.
	//
	//   U_old = state.U, V_old = state.V, expEll_old = state.expEllPrev
	//   U_new = state.URaw, V_new = state.VRaw, expEll_new = state.expEll
	//   UA    = state.UA (= U_old * A, from step 2e)
	{
		const dim3 TPB2(16, 16);
		const dim3 blocks((n + 15) / 16, (m + 15) / 16);

		if (vc.complementMomentumEnabled)
		{
			// Lazy-allocate and zero-init the momentum buffer on first use.
			if (state.complementMomentum.size() != mn)
			{
				if (!state.complementMomentum.allocate(mn))
					return false;
				const unsigned int zBlocks =
				    (static_cast<unsigned int>(mn) + TPB - 1) / TPB;
				k_zero_f<<<zBlocks, TPB, 0, cs>>>(state.complementMomentum.data(),
				                           static_cast<unsigned int>(mn));
			}

			if (vc.complementUseSign)
			{
				k_vesta_fused_update<true, true><<<blocks, TPB2, 0, cs>>>(
				    d_W, m, n, r,
				    state.U.data(), state.V.data(), state.expEllPrev.data(),
				    state.URaw.data(), state.VRaw.data(), state.expEll.data(),
				    state.UA.data(), d_gW,
				    state.complementMomentum.data(),
				    vc.complementBeta, lrCperp);
			}
			else
			{
				k_vesta_fused_update<true, false><<<blocks, TPB2, 0, cs>>>(
				    d_W, m, n, r,
				    state.U.data(), state.V.data(), state.expEllPrev.data(),
				    state.URaw.data(), state.VRaw.data(), state.expEll.data(),
				    state.UA.data(), d_gW,
				    state.complementMomentum.data(),
				    vc.complementBeta, lr * vc.lambdaPerp);
			}
		}
		else
		{
			if (vc.complementUseSign)
			{
				k_vesta_fused_update<false, true><<<blocks, TPB2, 0, cs>>>(
				    d_W, m, n, r,
				    state.U.data(), state.V.data(), state.expEllPrev.data(),
				    state.URaw.data(), state.VRaw.data(), state.expEll.data(),
				    state.UA.data(), d_gW,
				    /*momentum=*/static_cast<float*>(0),
				    /*beta=*/0.0f, lrCperp);
			}
			else
			{
				k_vesta_fused_update<false, false><<<blocks, TPB2, 0, cs>>>(
				    d_W, m, n, r,
				    state.U.data(), state.V.data(), state.expEllPrev.data(),
				    state.URaw.data(), state.VRaw.data(), state.expEll.data(),
				    state.UA.data(), d_gW,
				    /*momentum=*/static_cast<float*>(0),
				    /*beta=*/0.0f, lr * vc.lambdaPerp);
			}
		}
	}

	// Commit U = URaw, V = VRaw AFTER the fused update (which read old values).
	if (cudaMemcpyAsync(state.U.data(), state.URaw.data(),
	                    static_cast<size_t>(m) * r * sizeof(float),
	                    cudaMemcpyDeviceToDevice, cs) != cudaSuccess)
		return false;
	if (cudaMemcpyAsync(state.V.data(), state.VRaw.data(),
	                    static_cast<size_t>(n) * r * sizeof(float),
	                    cudaMemcpyDeviceToDevice, cs) != cudaSuccess)
		return false;

	// Step 8: trust-region clamp on host. Reuses hostEll from step 6 (ell is
	// not modified between step 6 and here, since the fused kernel reads
	// expEll_new but doesn't write state.ell).
	{
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

	// Step 10: gW zeroing is fused into k_vesta_fused_update (step 7).

	// No device-wide sync here. All VESTA kernels and cuBLAS calls in this
	// function run on computeStream(), and the transformer's forward/backward
	// also uses computeStream. Naturally serialized -- next weight matrix's
	// step and the next iteration's forward pass both run on the same stream
	// as this step's writes to d_W / state.U / state.V.
	state.step += 1ULL;
	return true;
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
