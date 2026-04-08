// GPU-accelerated ATLAS optimizer (BRSP variant).
//
// Mirrors the CPU atlas_optimizer.h interface but operates on device memory,
// using cuBLAS for GEMMs and custom kernels for elementwise/reduction ops.
//
// Determinism: GPU ATLAS uses deterministic two-pass reductions (no atomicAdd)
// and synchronous D2H transfers, so sigma2 and mu are updated in the same step
// as on CPU. Given the same RNG seed and inputs, GPU and CPU produce the same
// training trajectory (within float32 rounding from cuBLAS vs CPU GEMM).
#pragma once

#include "gpu_buffer.h"
#include "../training_config.h"
#include "../../rng.h"
#include <cstddef>

namespace shmea { class GLogger; }

#ifdef GLADES_HAVE_CUDA

namespace glades {
namespace gpu {

// Per-weight-matrix ATLAS optimizer state on GPU.
// Mirrors atlas::WeightState but uses GpuBuffer for device allocations.
//
// Dual-space subspace selection:
//   When m <= n: left subspace U ∈ R^{m×r}, gz ∈ R^{r×n} (row-major, direction = row)
//   When m >  n: right subspace V ∈ R^{n×r}, gz ∈ R^{m×r} (row-major, direction = col)
// This ensures the subspace always operates in the smaller dimension for
// better coverage with fixed rank r.
struct GpuAtlasWeightState
{
	unsigned int m;     // rows of weight matrix
	unsigned int n;     // cols of weight matrix
	unsigned int r;     // subspace rank
	bool rightSubspace; // true when m > n (use right singular vectors)

	// subDim = min(m,n): dimension the subspace basis lives in
	// outerDim = max(m,n): the other dimension
	// U/V: [subDim * r] orthonormal subspace basis (row-major)
	// gz/gPred/prevGz: [outerDim * r] projected gradient
	//   Left subspace (rightSubspace=false): gz[r, n] — direction c is row c
	//   Right subspace (rightSubspace=true):  gz[m, r] — direction c is col c

	GpuBuffer<float> U;          // [subDim * r] orthonormal subspace basis
	GpuBuffer<float> fisherDiag; // [r] EMA of Fisher eigenvalues
	GpuBuffer<float> prevGz;     // [outerDim * r] previous compressed gradient

	// Scratch buffers (persistent to avoid per-step allocation)
	GpuBuffer<float> gz;         // [outerDim * r] current projected gradient
	GpuBuffer<float> gPred;      // [outerDim * r] scaled prediction for correction SGEMM
	GpuBuffer<float> d_reduce;   // [2] reduction output (errNormSq, gzNormSq)
	GpuBuffer<float> d_partials; // [512] per-block partial sums for deterministic reductions

	// Subspace refresh scratch
	GpuBuffer<float> U_old;      // [subDim * r]
	GpuBuffer<float> f_old;      // [r]
	GpuBuffer<float> B;          // [outerDim * r] power iteration intermediate
	GpuBuffer<float> overlap;    // [r * r * 2]
	GpuBuffer<float> prevGzOld;  // [outerDim * r]
	GpuBuffer<float> qrTemp;     // [subDim * r] scratch for Cholesky QR SGEMM output
	bool refreshAllocated;       // true after refresh scratch buffers allocated

	// Host-side scalars (passed to kernels as parameters, updated on host)
	float totalTrace;
	float sigma2;
	float mu;
	unsigned long long step;
	bool initialized;
	float lastBaselineRate;

	GpuAtlasWeightState()
	    : m(0u), n(0u), r(0u), rightSubspace(false),
	      refreshAllocated(false),
	      totalTrace(0.0f), sigma2(0.0f), mu(0.01f), step(0ULL), initialized(false),
	      lastBaselineRate(0.0f)
	{
	}

private:
	// Non-copyable (GpuBuffer resources are owned exclusively).
	GpuAtlasWeightState(const GpuAtlasWeightState&);
	GpuAtlasWeightState& operator=(const GpuAtlasWeightState&);
};

// Initialize ATLAS state for a weight matrix [m x n] with subspace rank r.
// Allocates all device buffers and initializes U with random orthonormal basis.
// rng: network RNG engine (generates U on CPU for determinism, then uploads to GPU).
bool atlas_gpu_init(GpuAtlasWeightState& state,
                    unsigned int m, unsigned int n,
                    unsigned int rank, float muInit,
                    glades::rng::Engine& rng);

// Apply one ATLAS optimizer step on GPU (BRSP variant).
// d_W: device pointer to weight matrix [m*n]
// d_gW: device pointer to gradient matrix [m*n] (cleared to zero after use)
// Returns true on success.
bool atlas_gpu_step(GpuAtlasWeightState& state,
                    float* d_W, float* d_gW,
                    unsigned int m, unsigned int n,
                    float invBatch, float lr,
                    float wd1, float wd2, float gradScale,
                    const glades::ATLASConfig& ac,
                    shmea::GLogger* logger = 0,
                    const char* tag = 0);

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
	float fisherMedian;
	float corrScaleMin, corrScaleMax;

	// Spectral efficiency metrics (adaptive rank diagnostics)
	float effectiveRank;      // exp(entropy of Fisher distribution), in [1, r]
	float spectralEfficiency; // effectiveRank / r, in [0, 1]
	float top1Concentration;  // fraction of total Fisher in strongest direction
	float top10Concentration; // fraction of total Fisher in top-10 directions

	unsigned long long step;
	bool rightSubspace;
	bool valid;
	AtlasGpuDiag() : sigma2(0), mu(0), baselineRate(0), gzNorm(0), updateNorm(0),
	                  fisherMin(0), fisherMax(0), fisherMean(0), fisherMedian(0),
	                  corrScaleMin(0), corrScaleMax(0),
	                  effectiveRank(0), spectralEfficiency(0),
	                  top1Concentration(0), top10Concentration(0),
	                  step(0), rightSubspace(false), valid(false) {}
};

// Convenience wrapper: initializes state if needed, then calls atlas_gpu_step.
// Mirrors the CPU atlas::update() function. Returns true on success.
bool atlas_gpu_update(GpuAtlasWeightState& state,
                      float* d_W, float* d_gW,
                      unsigned int m, unsigned int n,
                      float invBatch, float lr,
                      float wd1, float wd2, float gradScale,
                      const glades::ATLASConfig& ac,
                      glades::rng::Engine& rng,
                      shmea::GLogger* logger = 0,
                      const char* tag = 0);

// Retrieve diagnostic info from the current state.
// Downloads Fisher diagonal from GPU — call sparingly (e.g. every tSub steps).
AtlasGpuDiag atlas_gpu_get_diag(const GpuAtlasWeightState& state);

// --- Individual CUDA kernels (host wrappers) ---

// Gram-Schmidt orthonormalization of Q[m, r] on GPU (row-major).
bool atlas_gpu_gram_schmidt(float* d_Q, int m, int r);

// Elementwise weight decay: W[i] -= lr * (wd1*sign(W[i]) + wd2*W[i])
bool atlas_gpu_weight_decay(float* d_W, size_t mn, float lr, float wd1, float wd2);

// Baseline update: W[i] -= baseScaled * gW[i]
bool atlas_gpu_baseline_update(float* d_W, const float* d_gW, size_t mn, float baseScaled);

// Fisher diagonal update: EMA of mean-squared projected gradient per subspace direction.
// Left subspace:  gz[r, n], direction c = row c (contiguous, stride=n)
// Right subspace: gz[m, r], direction c = col c (strided, stride=r)
// outerDim: n (left) or m (right) — number of elements per direction
bool atlas_gpu_fisher_update(const float* d_gz, float* d_fisherDiag,
                              int r, int outerDim, float beta, float sampleScale,
                              bool rightSubspace, bool bootstrap);

// Prepare scaled prediction for subspace correction.
// Left subspace:  gz/out are [r, outerDim], direction c = row c
// Right subspace: gz/out are [outerDim, r], direction c = col c
// corrScale[c] = baselineRate - min(lr/(fisherDiag[c]*bcFactor+eps), kappaLr)
bool atlas_gpu_prepare_correction(const float* d_gz, const float* d_prevGz,
                                   const float* d_fisherDiag,
                                   float* d_out, int r, int outerDim,
                                   float onePlusMu, float negMu,
                                   float baselineRate, float lr, float eps,
                                   float kappaLr, float bcFactor,
                                   bool rightSubspace);

// Compute norms for mu adaptation (deterministic two-pass reduction):
// d_out[0] = sum((gz[i]-prevGz[i])^2), d_out[1] = sum(gz[i]^2)
// d_partials: device scratch buffer of at least 2*min(256,ceil(rn/256)) floats.
bool atlas_gpu_mu_norms(const float* d_gz, const float* d_prevGz,
                         size_t rn, float* d_out, float* d_partials);

// EMA blend: dst[i] = (1-beta)*a[i] + beta*b[i]
bool atlas_gpu_ema_blend(float* d_dst, const float* d_a, const float* d_b,
                          size_t count, float beta);

// Transform Fisher diagonal into new basis:
// f_new[c] = sum_j overlap[c*r+j]^2 * f_old[j]
bool atlas_gpu_transform_fisher(const float* d_overlap, const float* d_f_old,
                                  float* d_f_new, int r);

// Scale gradient for refresh: out[i] = gW[i] * gScale
bool atlas_gpu_scale_grad(const float* d_gW, float* d_out, size_t mn, float gScale);

// Guard: replace NaN/Inf values with 0.0f. Returns true on success.
bool atlas_gpu_guard(float* d_W, size_t mn);

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
	float totalTrace;
	float sigma2;
	float mu;
	unsigned long long step;
	bool initialized;
	GpuAtlasWeightState() : totalTrace(0.0f), sigma2(0.0f), mu(0.0f), step(0ULL), initialized(false) {}
};

struct AtlasGpuDiag
{
	bool valid;
	AtlasGpuDiag() : valid(false) {}
};

inline bool atlas_gpu_init(GpuAtlasWeightState&, unsigned int, unsigned int,
                           unsigned int, float, glades::rng::Engine&) { return false; }
inline bool atlas_gpu_step(GpuAtlasWeightState&, float*, float*,
                           unsigned int, unsigned int,
                           float, float, float, float, float,
                           const glades::ATLASConfig&,
                           shmea::GLogger* = 0,
                           const char* = 0) { return false; }
inline AtlasGpuDiag atlas_gpu_get_diag(const GpuAtlasWeightState&)
{ return AtlasGpuDiag(); }
inline bool atlas_gpu_update(GpuAtlasWeightState&, float*, float*,
                              unsigned int, unsigned int,
                              float, float, float, float, float,
                              const glades::ATLASConfig&,
                              glades::rng::Engine&,
                              shmea::GLogger* = 0,
                              const char* = 0) { return false; }
inline bool atlas_gpu_guard(float*, size_t) { return false; }

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
