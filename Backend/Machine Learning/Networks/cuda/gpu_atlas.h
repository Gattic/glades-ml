// GPU-accelerated ATLAS optimizer (BRSP variant).
//
// Mirrors the CPU atlas_optimizer.h interface but operates on device memory,
// using cuBLAS for GEMMs and custom kernels for elementwise/reduction ops.
#pragma once

#include "gpu_buffer.h"
#include <cstddef>

#ifdef GLADES_HAVE_CUDA

namespace glades {
namespace gpu {

// Per-weight-matrix ATLAS optimizer state on GPU.
// Mirrors atlas::WeightState but uses GpuBuffer for device allocations.
struct GpuAtlasWeightState
{
	unsigned int m;     // rows of weight matrix
	unsigned int n;     // cols of weight matrix
	unsigned int r;     // subspace rank

	GpuBuffer<float> U;          // [m * r] orthonormal subspace basis (row-major)
	GpuBuffer<float> fisherDiag; // [r] EMA of Fisher eigenvalues
	GpuBuffer<float> prevGz;     // [r * n] previous compressed gradient

	// Scratch buffers (persistent to avoid per-step allocation)
	GpuBuffer<float> gz;         // [r * n] current projected gradient
	GpuBuffer<float> gPred;      // [r * n] scaled prediction for correction SGEMM
	GpuBuffer<float> d_reduce;   // [2] reduction output (errNormSq, gzNormSq)

	// Subspace refresh scratch (lazily allocated on first refresh step)
	GpuBuffer<float> U_old;      // [m * r]
	GpuBuffer<float> f_old;      // [r]
	GpuBuffer<float> B;          // [n * r] power iteration intermediate
	GpuBuffer<float> overlap;    // [r * r]
	GpuBuffer<float> prevGzOld;  // [r * n]
	bool refreshAllocated;       // true after refresh scratch buffers allocated

	// Host-side scalars (passed to kernels as parameters, updated on host)
	float sigma2;
	float mu;
	unsigned long long step;
	bool initialized;

	GpuAtlasWeightState()
	    : m(0u), n(0u), r(0u),
	      refreshAllocated(false),
	      sigma2(1.0f), mu(0.01f), step(0ULL), initialized(false)
	{
	}
};

// Initialize ATLAS state for a weight matrix [m x n] with subspace rank r.
// Allocates all device buffers and initializes U with random orthonormal basis.
bool atlas_gpu_init(GpuAtlasWeightState& state,
                    unsigned int m, unsigned int n,
                    unsigned int rank, float muInit);

// Apply one ATLAS optimizer step on GPU (BRSP variant).
// d_W: device pointer to weight matrix [m*n]
// d_gW: device pointer to gradient matrix [m*n] (cleared to zero after use)
// Returns true on success.
bool atlas_gpu_step(GpuAtlasWeightState& state,
                    float* d_W, float* d_gW,
                    unsigned int m, unsigned int n,
                    float invBatch, float lr,
                    float wd1, float wd2, float gradScale,
                    float beta, float muMin, float muMax,
                    float eps, unsigned int tSub,
                    unsigned int powerIters, float betaRefresh,
                    float kappaMax);

// Retrieve diagnostic info from the last step (for logging).
// Only meaningful when step % tSub == 0.
struct AtlasGpuDiag
{
	float sigma2;
	float mu;
	float baselineRate;
	float gzNorm;
	float updateNorm;
	float fisherMin, fisherMax, fisherMean;
	unsigned long long step;
	bool valid;
	AtlasGpuDiag() : sigma2(0), mu(0), baselineRate(0), gzNorm(0), updateNorm(0),
	                  fisherMin(0), fisherMax(0), fisherMean(0), step(0), valid(false) {}
};

// --- Individual CUDA kernels (host wrappers) ---

// Gram-Schmidt orthonormalization of Q[m, r] on GPU (row-major).
bool atlas_gpu_gram_schmidt(float* d_Q, int m, int r);

// Elementwise weight decay: W[i] -= lr * (wd1*sign(W[i]) + wd2*W[i])
bool atlas_gpu_weight_decay(float* d_W, int mn, float lr, float wd1, float wd2);

// Baseline update: W[i] -= baseScaled * gW[i]
bool atlas_gpu_baseline_update(float* d_W, const float* d_gW, int mn, float baseScaled);

// Fisher diagonal update: EMA of row-wise mean-squared of gz[r, n].
bool atlas_gpu_fisher_update(const float* d_gz, float* d_fisherDiag, int r, int n, float beta);

// Prepare scaled prediction for subspace correction:
// out[c*n+j] = corrScale[c] * ((1+mu)*gz[c*n+j] - mu*prevGz[c*n+j])
// where corrScale[c] = baselineRate - min(lr/(fisherDiag[c]+eps), kappaLr)
bool atlas_gpu_prepare_correction(const float* d_gz, const float* d_prevGz,
                                   const float* d_fisherDiag,
                                   float* d_out, int r, int n,
                                   float onePlusMu, float negMu,
                                   float baselineRate, float lr, float eps,
                                   float kappaLr);

// Compute norms for mu adaptation:
// d_out[0] = sum((gz[i]-prevGz[i])^2), d_out[1] = sum(gz[i]^2)
bool atlas_gpu_mu_norms(const float* d_gz, const float* d_prevGz,
                         int rn, float* d_out);

// EMA blend: dst[i] = (1-beta)*a[i] + beta*b[i]
bool atlas_gpu_ema_blend(float* d_dst, const float* d_a, const float* d_b,
                          int count, float beta);

// Transform Fisher diagonal into new basis:
// f_new[c] = sum_j overlap[c*r+j]^2 * f_old[j]
bool atlas_gpu_transform_fisher(const float* d_overlap, const float* d_f_old,
                                  float* d_f_new, int r);

// Scale gradient for refresh: out[i] = gW[i] * gScale
bool atlas_gpu_scale_grad(const float* d_gW, float* d_out, int mn, float gScale);

// Transform prevGz into new basis: prevGz_new = overlap * prevGz_old
// overlap[r,r] * prevGzOld[r,n] -> prevGz[r,n]
// Uses cuBLAS sgemm_rowmajor.

} // namespace gpu
} // namespace glades

#else // !GLADES_HAVE_CUDA

namespace glades {
namespace gpu {

struct GpuAtlasWeightState
{
	bool initialized;
	GpuAtlasWeightState() : initialized(false) {}
};

struct AtlasGpuDiag
{
	bool valid;
	AtlasGpuDiag() : valid(false) {}
};

inline bool atlas_gpu_init(GpuAtlasWeightState&, unsigned int, unsigned int,
                           unsigned int, float) { return false; }
inline bool atlas_gpu_step(GpuAtlasWeightState&, float*, float*,
                           unsigned int, unsigned int,
                           float, float, float, float, float,
                           float, float, float, float, unsigned int,
                           unsigned int, float) { return false; }

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
