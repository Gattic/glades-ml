// HELIOS optimizer: Hamiltonian Ensemble Langevin Integrator with
// Sharpness-adaptive Thermostat.
//
// HELIOS treats training as a dissipative stochastic Hamiltonian flow on the
// augmented phase space (theta, p, xi) and discretizes it with the BAOAB
// stochastic-symplectic splitting (Leimkuhler & Matthews 2013). The stationary
// distribution is the Gibbs measure pi(theta) prop. exp(-U(theta)/T), and with
// sharpness-adaptive temperature T(kappa) it concentrates on flat minima -- a
// property AdamW/ATLAS/VESTA do not have.
//
// This header declares the MINIMUM VIABLE INSTANTIATION: plain BAOAB on
// underdamped Langevin dynamics (Q=0 disables the Nose-Hoover N-step, alpha=0
// disables the sharpness feedback, lambdaAnchor=0 disables the anchor). The
// full machinery (per-group thermostat, sharpness probe, anchor EMA) can be
// enabled by setting the corresponding HeliosConfig fields to non-zero values;
// the extra code paths are guarded by those fields.
//
// Reference: research/HELIOS_framework.md.

#pragma once

#include <vector>
#include <cmath>
#include <cstddef>
#include "../rng.h"

namespace shmea { class GLogger; }

namespace glades {

struct HeliosConfig; // forward; defined in training_config.h.

namespace helios {

// Per-weight-matrix HELIOS optimizer state.
//
// For W in R^{m x n}:
//   p        [m * n] conjugate momentum (row-major).
//   thetaBar [m * n] anchor EMA (allocated only when lambdaAnchor > 0).
//
// Per-matrix scalar state:
//   xi       Nose-Hoover thermostat variable (used only when Q > 0).
//   Tcurr    effective temperature (starts at T0; adapts with kappa when
//            alpha > 0 and kHvp > 0).
//   kappa    EMA of local sharpness along momentum direction (0 until the
//            first HVP probe).
//   mass     mass m (copied from config; may be overridden per-layer role
//            by higher-level logic for muP compatibility).
//   step     iteration counter.
struct WeightState
{
	unsigned int m;
	unsigned int n;

	std::vector<float> p;         // [m * n] momentum
	std::vector<float> thetaBar;  // [m * n] anchor EMA (empty if disabled)

	float xi;
	float Tcurr;
	float kappa;
	float mass;

	unsigned long long step;
	bool initialized;

	WeightState()
	    : m(0u), n(0u),
	      xi(0.0f),
	      Tcurr(0.0f),
	      kappa(0.0f),
	      mass(1.0f),
	      step(0ULL),
	      initialized(false)
	{
	}

	void reset()
	{
		m = n = 0u;
		p.clear();
		thetaBar.clear();
		xi = 0.0f;
		Tcurr = 0.0f;
		kappa = 0.0f;
		mass = 1.0f;
		step = 0ULL;
		initialized = false;
	}
};

// Initialize HELIOS state for a weight matrix of dimensions [m x n].
// Allocates p (zero-initialized); allocates thetaBar only if lambdaAnchor > 0.
// Copies T0 / mass / scalars from config. Safe to call repeatedly; a second
// call with different (m, n) resets and re-allocates.
void initWeightState(WeightState& state,
                     const float* W,
                     unsigned int m, unsigned int n,
                     const HeliosConfig& hc,
                     glades::rng::Engine& rng,
                     shmea::GLogger* logger = 0);

// Apply one HELIOS BAOAB step to W using gradient gW.
//
// Signature mirrors ATLAS/VESTA. The effective step is h_eff = lr * hc.h;
// hc.h defaults to 1 so lr is the base step.
//
// Integrator layout (minimum viable, Q=0, alpha=0, lambdaAnchor=0):
//   B-half : p = p - (h/2) * (gradScale / invBatch) * gW
//   A-half : theta = theta + (h / (2 * mass)) * p
//   O-step : c = exp(-gamma_0 * h);
//            p = c * p + sqrt(mass * T * (1 - c^2)) * zeta, zeta ~ N(0, I)
//   A-half : theta = theta + (h / (2 * mass)) * p
//   B-half : p = p - (h/2) * (gradScale / invBatch) * gW
//
// When Q > 0 the N-half steps are inserted around O-step (Leimkuhler-Shang-
// Matthews BNAOANB). When lambdaAnchor > 0 an anchor penalty lambda * (theta -
// thetaBar) is added to the force in both B-halves. When kHvp > 0 and alpha >
// 0, the Tcurr adaptation feeds kappa into the O-step friction.
//
// gW is not cleared after the step (BAOAB reuses the same g for both B-halves,
// so the caller must clear gW after the outer loop).
//
// invBatch: reciprocal of the effective minibatch size used during
// accumulation of gW. gradScale: an additional scaling factor (e.g., for
// gradient compression undo). wd1: unused in HELIOS (no L1 penalty; future
// extension). wd2: decoupled weight decay coefficient; theta is multiplied by
// (1 - lr * wd2) before the integrator step.
//
// Returns false if a non-finite value is detected during the step (state is
// left in an unspecified but not crashing condition; the caller should reset).
bool applyStep(WeightState& state,
               float* W, float* gW,
               unsigned int m, unsigned int n,
               float invBatch, float lr,
               float wd1, float wd2, float gradScale,
               const HeliosConfig& hc,
               glades::rng::Engine& rng,
               shmea::GLogger* logger = 0,
               const char* tag = 0);

// Convenience: initializes state if needed, then calls applyStep.
bool update(WeightState& state,
            float* W, float* gW,
            unsigned int m, unsigned int n,
            float invBatch, float lr,
            float wd1, float wd2, float gradScale,
            const HeliosConfig& hc,
            glades::rng::Engine& rng,
            shmea::GLogger* logger = 0,
            const char* tag = 0);

} // namespace helios
} // namespace glades
