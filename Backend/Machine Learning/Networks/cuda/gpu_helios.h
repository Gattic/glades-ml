// GPU-accelerated HELIOS optimizer.
//
// Mirrors the CPU helios_optimizer.h interface but operates on device memory.
// The integrator is BAOAB (Leimkuhler-Matthews). The minimum viable
// instantiation (Q=0, alpha=0, lambdaAnchor=0) runs as pure underdamped
// Langevin; the thermostat / sharpness / anchor paths are guarded by the
// corresponding HeliosConfig fields and are no-ops when those fields are 0.
//
// Determinism: given the same RNG seed + inputs, the GPU trajectory matches
// the CPU trajectory to within elementwise-fp rounding. The O-step noise is
// generated on host using the same glades::rng engine the CPU path uses, then
// uploaded to device; this keeps RNG consumption order matched.
#pragma once

#include "gpu_buffer.h"
#include "../../rng.h"
#include <cstddef>

namespace shmea { class GLogger; }

namespace glades {

struct HeliosConfig; // forward; defined in training_config.h.

#ifdef GLADES_HAVE_CUDA

namespace gpu {

// Per-weight-matrix HELIOS state on GPU.
struct GpuHeliosWeightState
{
	unsigned int m;
	unsigned int n;

	GpuBuffer<float> p;        // [m * n] momentum
	GpuBuffer<float> thetaBar; // [m * n] anchor EMA (empty if disabled)

	// Scratch for the O-step: host-generated N(0,1) samples are uploaded here
	// then consumed by k_helios_o_step. Sized to m*n on first use.
	GpuBuffer<float> noise;

	// Single-int device flag used by the non-finite guard. Set to 1 by
	// k_helios_check_finite if any input element is NaN or +-Inf; read back
	// once per step to abort on corrupt gradient or state.
	GpuBuffer<int> finiteFlag;

	// Host-resident scalar state (matches CPU fields).
	float xi;
	float Tcurr;
	float kappa;
	float mass;
	unsigned long long step;
	bool initialized;

	GpuHeliosWeightState()
	    : m(0u), n(0u),
	      xi(0.0f),
	      Tcurr(0.0f),
	      kappa(0.0f),
	      mass(1.0f),
	      step(0ULL),
	      initialized(false)
	{
	}

private:
	GpuHeliosWeightState(const GpuHeliosWeightState&);
	GpuHeliosWeightState& operator=(const GpuHeliosWeightState&);
};

// Initialize GPU HELIOS state for a weight matrix [m x n].
// Allocates p (zero-initialized) and, if hc.lambdaAnchor > 0, thetaBar
// (zero-initialized). Scalars are copied from config. d_W is accepted for
// signature parity with vesta_gpu_init but is not read.
bool helios_gpu_init(GpuHeliosWeightState& state,
                     const float* d_W,
                     unsigned int m, unsigned int n,
                     const glades::HeliosConfig& hc,
                     glades::rng::Engine& rng,
                     shmea::GLogger* logger = 0);

// Apply one HELIOS BAOAB step on GPU.
//
// Signature mirrors vesta_gpu_step. Returns false if the step could not
// complete (state uninitialized, dimensions mismatch) or if a non-finite
// value was observed in gW, W, or p. The caller should reset state on
// failure. gW is zeroed at the end of the step (BAOAB reuses gW in both
// B-halves, so we zero only once at the end).
bool helios_gpu_step(GpuHeliosWeightState& state,
                     float* d_W, float* d_gW,
                     unsigned int m, unsigned int n,
                     float invBatch, float lr,
                     float wd1, float wd2, float gradScale,
                     const glades::HeliosConfig& hc,
                     glades::rng::Engine& rng,
                     shmea::GLogger* logger = 0,
                     const char* tag = 0);

// ---------------- GPU FD-HVP probe infrastructure ----------------
//
// These helpers implement the framework Section 7 step 10 sharpness probe
// on device. The training-loop integrator (transformerGpuTrainEpoch) calls
// these at minibatch boundaries when HeliosConfig::alpha > 0 and kHvp > 0.
//
// Probe protocol (gradient-FD form, matching the CPU path):
//   1. helios_gpu_probe_snapshot_W: save W_target to a backup buffer.
//   2. helios_gpu_probe_compute_v:  v = p / ||p||  (returns ||p|| for the caller's sanity check).
//   3. helios_gpu_probe_perturb:    W_target := W_save + eps * v.
//   4. [caller runs full forward+backward; extracts gW_target into gPlus buffer]
//   5. helios_gpu_probe_perturb_negative: W_target := W_save - eps * v.
//   6. [caller runs full forward+backward again; gW_target is gMinus]
//   7. helios_gpu_probe_compute_kappa: compute v . (gPlus - gMinus) / (2 eps)
//      as a host-returned scalar.
//   8. helios_gpu_probe_restore_W: W_target := W_save (via cudaMemcpy D2D).
//
// All kernels run on computeStream() for natural serialization with the
// transformer's forward/backward.

// (1) Snapshot target matrix (cudaMemcpy device-to-device).
// Caller-provided scratch buffer must be allocated with N >= m*n floats.
bool helios_gpu_probe_snapshot_W(float* d_Wsave, const float* d_W,
                                 unsigned int m, unsigned int n);

// (2) Compute v = p / ||p|| into d_vOut. Returns ||p|| in host-side
// pNormOut; caller can decide to skip the probe when the norm is too
// small (below ~1e-6 relative to weight scale) to avoid an ill-defined
// direction.
bool helios_gpu_probe_compute_v(const float* d_p, float* d_vOut,
                                unsigned int m, unsigned int n,
                                float& pNormOut);

// (3 / 5) W_target := W_save + sign * eps * v. sign = +1 for the + probe,
// -1 for the - probe. Does NOT modify d_Wsave or d_v.
bool helios_gpu_probe_perturb(float* d_W, const float* d_Wsave,
                              const float* d_v, float eps, float sign,
                              unsigned int m, unsigned int n);

// (7) kappa = v . (gPlus - gMinus) / (2 eps). Runs an in-place reduction
// on device, downloads a single scalar to the host.
bool helios_gpu_probe_compute_kappa(const float* d_v,
                                    const float* d_gPlus,
                                    const float* d_gMinus,
                                    unsigned int m, unsigned int n,
                                    float eps,
                                    float& kappaOut);

// (8) Restore W from snapshot (cudaMemcpy device-to-device).
bool helios_gpu_probe_restore_W(float* d_W, const float* d_Wsave,
                                unsigned int m, unsigned int n);

} // namespace gpu

#else // !GLADES_HAVE_CUDA

namespace gpu {
struct GpuHeliosWeightState
{
	unsigned int m, n;
	float xi, Tcurr, kappa, mass;
	unsigned long long step;
	bool initialized;
	GpuHeliosWeightState()
	    : m(0u), n(0u), xi(0.0f), Tcurr(0.0f), kappa(0.0f),
	      mass(1.0f), step(0ULL), initialized(false) {}
};
inline bool helios_gpu_init(GpuHeliosWeightState&, const float*,
                            unsigned int, unsigned int,
                            const glades::HeliosConfig&, glades::rng::Engine&,
                            shmea::GLogger* = 0) { return false; }
inline bool helios_gpu_step(GpuHeliosWeightState&, float*, float*,
                            unsigned int, unsigned int,
                            float, float, float, float, float,
                            const glades::HeliosConfig&, glades::rng::Engine&,
                            shmea::GLogger* = 0, const char* = 0) { return false; }
inline bool helios_gpu_probe_snapshot_W(float*, const float*, unsigned int, unsigned int) { return false; }
inline bool helios_gpu_probe_compute_v(const float*, float*, unsigned int, unsigned int, float&) { return false; }
inline bool helios_gpu_probe_perturb(float*, const float*, const float*, float, float, unsigned int, unsigned int) { return false; }
inline bool helios_gpu_probe_compute_kappa(const float*, const float*, const float*, unsigned int, unsigned int, float, float&) { return false; }
inline bool helios_gpu_probe_restore_W(float*, const float*, unsigned int, unsigned int) { return false; }
} // namespace gpu

#endif // GLADES_HAVE_CUDA

} // namespace glades
