// HELIOS optimizer: BAOAB stochastic-symplectic integrator for the
// underdamped Langevin-Nose-Hoover SDE.
//
// See helios_optimizer.h and research/HELIOS_framework.md for the mathematics.
// This file implements the MINIMUM VIABLE INSTANTIATION; optional paths
// (thermostat, sharpness feedback, anchor) are guarded by HeliosConfig flags.

#include "helios_optimizer.h"
#include "training_config.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace glades {
namespace helios {

namespace {

// Detect NaN or +-Inf in one branchless test. For finite v, v*0 == 0; for
// NaN and Inf, v*0 == NaN which compares unequal to anything.
inline bool notFinite(float v)
{
	return !(v * 0.0f == 0.0f);
}

inline bool arrayHasNonFinite(const float* a, size_t n)
{
	for (size_t i = 0; i < n; ++i)
	{
		if (notFinite(a[i])) return true;
	}
	return false;
}

} // namespace

void initWeightState(WeightState& state,
                     const float* /*W*/,
                     unsigned int m, unsigned int n,
                     const HeliosConfig& hc,
                     glades::rng::Engine& /*rng*/,
                     shmea::GLogger* /*logger*/)
{
	const size_t P = static_cast<size_t>(m) * n;

	state.m = m;
	state.n = n;
	state.p.assign(P, 0.0f);
	if (hc.lambdaAnchor > 0.0f)
		state.thetaBar.assign(P, 0.0f);
	else
		state.thetaBar.clear();

	state.xi = 0.0f;
	state.Tcurr = hc.T0;
	state.kappa = 0.0f;
	state.mass = (hc.mass > 0.0f) ? hc.mass : 1.0f;
	state.step = 0ULL;
	state.initialized = true;
}

bool applyStep(WeightState& state,
               float* W, float* gW,
               unsigned int m, unsigned int n,
               float invBatch, float lr,
               float /*wd1*/, float wd2, float gradScale,
               const HeliosConfig& hc,
               glades::rng::Engine& rng,
               shmea::GLogger* /*logger*/,
               const char* /*tag*/)
{
	if (!state.initialized) return false;
	if (state.m != m || state.n != n) return false;

	const size_t P = static_cast<size_t>(m) * n;
	if (state.p.size() != P) return false;

	if (arrayHasNonFinite(gW, P)) return false;

	const float hEff = lr * hc.h;
	if (!(hEff > 0.0f)) return false;

	const float mass = state.mass;
	if (!(mass > 0.0f)) return false;

	const float invMass = 1.0f / mass;
	const float halfH = 0.5f * hEff;
	const float halfOverMass = halfH * invMass;
	const float gScale = invBatch * gradScale;

	// Decoupled weight decay: theta *= (1 - lr * wd2) before the integrator.
	if (wd2 > 0.0f)
	{
		const float decay = 1.0f - lr * wd2;
		for (size_t i = 0; i < P; ++i) W[i] *= decay;
	}

	// Effective temperature (Li-Sato-Tan correction, 0 in MVI).
	const float Teff = state.Tcurr + 0.25f * hEff * hc.noiseCorrection;

	float* p = &state.p[0];

	// The BAOAB layout uses the SAME gradient in both B-halves (one force
	// evaluation per step), which is what preserves the stochastic-symplectic
	// structure. gW is therefore not cleared here; the caller owns it.
	// In the MVI there is no anchor term; when lambdaAnchor > 0 is added in a
	// future cycle, the anchor force must be cached at theta_0 to keep both
	// B-halves consistent.

	// --- B-half: p -= (hEff / 2) * gScale * gW.
	for (size_t i = 0; i < P; ++i)
		p[i] -= halfH * gScale * gW[i];

	// --- A-half: theta += (hEff / (2 * mass)) * p.
	for (size_t i = 0; i < P; ++i)
		W[i] += halfOverMass * p[i];

	// --- N-half (Nose-Hoover thermostat; skipped when Q == 0).
	float xi = state.xi;
	const bool useThermostat = (hc.Q > 0.0f);
	if (useThermostat)
	{
		double kin = 0.0;
		for (size_t i = 0; i < P; ++i)
			kin += static_cast<double>(p[i]) * static_cast<double>(p[i]);
		kin *= invMass;
		const float Nf = static_cast<float>(P);
		xi += (halfH / hc.Q) * (static_cast<float>(kin) - Nf * state.Tcurr);
	}

	// --- O-step: Ornstein-Uhlenbeck.
	const float gammaBase = hc.gamma0 + xi + hc.alpha * state.kappa;
	const float gammaEff = (gammaBase < 0.0f) ? 0.0f : gammaBase;
	const float c = std::exp(-gammaEff * hEff);
	const float noiseVar = (Teff > 0.0f) ? mass * Teff * (1.0f - c * c) : 0.0f;
	const float noiseScale = (noiseVar > 0.0f) ? std::sqrt(noiseVar) : 0.0f;
	if (noiseScale > 0.0f)
	{
		for (size_t i = 0; i < P; ++i)
			p[i] = c * p[i] + noiseScale * glades::rng::standard_normal(rng);
	}
	else
	{
		for (size_t i = 0; i < P; ++i) p[i] = c * p[i];
	}

	// --- N-half (mirror).
	if (useThermostat)
	{
		double kin = 0.0;
		for (size_t i = 0; i < P; ++i)
			kin += static_cast<double>(p[i]) * static_cast<double>(p[i]);
		kin *= invMass;
		const float Nf = static_cast<float>(P);
		xi += (halfH / hc.Q) * (static_cast<float>(kin) - Nf * state.Tcurr);
	}
	state.xi = xi;

	// --- A-half.
	for (size_t i = 0; i < P; ++i)
		W[i] += halfOverMass * p[i];

	// --- B-half (same gW as the first B-half).
	for (size_t i = 0; i < P; ++i)
		p[i] -= halfH * gScale * gW[i];

	// Anchor EMA update (applied after the step; anchor force itself is not
	// yet active -- see note above).
	if (hc.lambdaAnchor > 0.0f && !state.thetaBar.empty())
	{
		float* tb = &state.thetaBar[0];
		const float betaA = hc.betaAnchor;
		const float oneMinus = 1.0f - betaA;
		for (size_t i = 0; i < P; ++i)
			tb[i] = betaA * tb[i] + oneMinus * W[i];
	}

	state.step++;

	// Post-step non-finite guard on W and p.
	if (arrayHasNonFinite(W, P)) return false;
	if (arrayHasNonFinite(p, P)) return false;
	return true;
}

bool update(WeightState& state,
            float* W, float* gW,
            unsigned int m, unsigned int n,
            float invBatch, float lr,
            float wd1, float wd2, float gradScale,
            const HeliosConfig& hc,
            glades::rng::Engine& rng,
            shmea::GLogger* logger,
            const char* tag)
{
	if (!state.initialized || state.m != m || state.n != n)
		initWeightState(state, W, m, n, hc, rng, logger);
	return applyStep(state, W, gW, m, n, invBatch, lr, wd1, wd2, gradScale,
	                 hc, rng, logger, tag);
}

} // namespace helios
} // namespace glades
