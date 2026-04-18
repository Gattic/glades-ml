// GPU-accelerated VESTA optimizer.
//
// Mirrors the CPU vesta_optimizer.h interface but operates on device memory,
// using cuBLAS for GEMMs and custom kernels for elementwise/reduction ops.
//
// Determinism: given the same RNG seed and inputs, GPU and CPU produce
// matching trajectories to within cuBLAS vs. CPU GEMM rounding (typically
// ~1e-4 abs / ~1e-3 rel after a few steps).
#pragma once

#include "gpu_buffer.h"
#include "../../rng.h"
#include <cstddef>
#include <vector>

namespace shmea { class GLogger; }

namespace glades {

struct VestaConfig; // forward; defined in training_config.h.

#ifdef GLADES_HAVE_CUDA

namespace gpu {

// Per-weight-matrix VESTA state on GPU.
struct GpuVestaWeightState
{
	unsigned int m;
	unsigned int n;
	unsigned int r;

	GpuBuffer<float> U;          // [m * r]
	GpuBuffer<float> V;          // [n * r]
	GpuBuffer<float> ell;        // [r]
	GpuBuffer<float> beta;       // [r]
	GpuBuffer<float> ellStar;    // [r]

	// Scratch buffers.
	GpuBuffer<float> A;          // [r * r]
	GpuBuffer<float> UA;         // [m * r]
	GpuBuffer<float> Omega_U;    // [m * r]
	GpuBuffer<float> Omega_V;    // [n * r]
	GpuBuffer<float> URaw;       // [m * r]
	GpuBuffer<float> VRaw;       // [n * r]
	GpuBuffer<float> expEll;     // [r]
	GpuBuffer<float> invExpEll;  // [r]
	GpuBuffer<float> expEllPrev; // [r] — saved before log_scale_update for fused reconstruct
	GpuBuffer<float> Adiag;      // [r]
	GpuBuffer<float> UtOmU;      // [r * r]
	GpuBuffer<float> VtOmV;      // [r * r]

	// Sketch scratch (oversampled by 8).
	GpuBuffer<float> sketchOmega; // [n * (r+8)]
	GpuBuffer<float> sketchY;     // [m * (r+8)]
	GpuBuffer<float> sketchB;     // [(r+8) * n]

	// Optional Lion-style / heavy-ball complement momentum buffer [m*n],
	// allocated lazily on first enabled step. Mirrors CPU WeightState::
	// complementMomentum.
	GpuBuffer<float> complementMomentum;

	unsigned long long step;
	float maxExpEllPrev;
	bool initialized;

	GpuVestaWeightState()
	    : m(0u), n(0u), r(0u),
	      step(0ULL),
	      maxExpEllPrev(1.0f),
	      initialized(false)
	{
	}

private:
	GpuVestaWeightState(const GpuVestaWeightState&);
	GpuVestaWeightState& operator=(const GpuVestaWeightState&);
};

// Initialize GPU VESTA state for a weight matrix [m x n].
// Performs sketched SVD on CPU (reusing vesta::sketched_svd determinism),
// then uploads U, V, ell to device. All scratch buffers are allocated here.
bool vesta_gpu_init(GpuVestaWeightState& state,
                    const float* d_W,
                    unsigned int m, unsigned int n,
                    const glades::VestaConfig& vc,
                    glades::rng::Engine& rng,
                    shmea::GLogger* logger = 0);

// Apply one VESTA optimizer step on GPU.
bool vesta_gpu_step(GpuVestaWeightState& state,
                    float* d_W, float* d_gW,
                    unsigned int m, unsigned int n,
                    float invBatch, float lr,
                    float wd1, float wd2, float gradScale,
                    const glades::VestaConfig& vc,
                    glades::rng::Engine& rng,
                    shmea::GLogger* logger = 0,
                    const char* tag = 0);

// Refresh (U, V, ell) by sketched SVD of current W.
bool vesta_gpu_refresh(GpuVestaWeightState& state,
                       const float* d_W,
                       unsigned int m, unsigned int n,
                       const glades::VestaConfig& vc,
                       glades::rng::Engine& rng,
                       shmea::GLogger* logger = 0);

} // namespace gpu

#else // !GLADES_HAVE_CUDA

namespace gpu {
struct GpuVestaWeightState
{
	unsigned int m, n, r;
	unsigned long long step;
	float maxExpEllPrev;
	bool initialized;
	GpuVestaWeightState() : m(0), n(0), r(0), step(0), maxExpEllPrev(1.0f), initialized(false) {}
};
inline bool vesta_gpu_init(GpuVestaWeightState&, const float*,
                           unsigned int, unsigned int,
                           const glades::VestaConfig&, glades::rng::Engine&,
                           shmea::GLogger* = 0) { return false; }
inline bool vesta_gpu_step(GpuVestaWeightState&, float*, float*,
                           unsigned int, unsigned int,
                           float, float, float, float, float,
                           const glades::VestaConfig&, glades::rng::Engine&,
                           shmea::GLogger* = 0, const char* = 0) { return false; }
inline bool vesta_gpu_refresh(GpuVestaWeightState&, const float*,
                              unsigned int, unsigned int,
                              const glades::VestaConfig&, glades::rng::Engine&,
                              shmea::GLogger* = 0) { return false; }
} // namespace gpu

#endif // GLADES_HAVE_CUDA

} // namespace glades
