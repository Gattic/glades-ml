// Copyright 2026 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
// associated documentation files (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge, publish, distribute,
// sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "chiron-test.h"
#include "../../unit-test.h"

#include "../../../Backend/Machine Learning/Networks/transformer_chiron_ops.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

// Deterministic LCG for test data (no dependency on <random>, keeps C++98 clean).
struct LCG
{
	unsigned int state;
	explicit LCG(unsigned int seed) : state(seed ? seed : 2471u) {}
	float next_unit()
	{
		// Numerical Recipes LCG; maps to (-1, 1).
		state = state * 1664525u + 1013904223u;
		const unsigned int mant = (state >> 9) & 0x007fffffu;
		const float u01 = static_cast<float>(mant) / static_cast<float>(1u << 23);
		return 2.0f * u01 - 1.0f;
	}
};

static float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b)
{
	if (a.size() != b.size())
		return 1e30f;
	float worst = 0.0f;
	for (size_t i = 0; i < a.size(); ++i)
	{
		const float d = fabsf(a[i] - b[i]);
		if (d > worst) worst = d;
	}
	return worst;
}

// Simple deterministic nonlinear "potential" map f: R^m -> R^m used as a
// stand-in for MLP(q) during reversibility tests.
//
// f(q)[i] = tanh( weight[i] * q[i] + bias[i] )
// Nonlinear, bounded, elementwise — easy to reason about but still exercises
// the block-level bijectivity since the composition is not a trivial identity.
static void potential_f(const float* q, const float* weight, const float* bias,
                        float* out, unsigned int m)
{
	for (unsigned int i = 0; i < m; ++i)
		out[i] = tanhf(weight[i] * q[i] + bias[i]);
}

static void potential_f_apply_all(const float* qMat, const float* weight, const float* bias,
                                  float* outMat, unsigned int T, unsigned int m)
{
	for (unsigned int t = 0; t < T; ++t)
		potential_f(qMat + t * m, weight, bias, outMat + t * m, m);
}

} // namespace

// ---------------------------------------------------------------------------
// Case 1: bare symplectic shear round-trip (no ReLN).
// Verifies that Shear^p followed by Shear^p^-1 is bit-exact.
// ---------------------------------------------------------------------------
void CHIRONShearReversibilityTest()
{
	const unsigned int T = 4;
	const unsigned int m = 8;

	LCG rng(1729u);
	std::vector<float> q0(T * m), p0(T * m);
	std::vector<float> weight(m), bias(m);

	for (unsigned int i = 0; i < q0.size(); ++i) q0[i] = 0.5f * rng.next_unit();
	for (unsigned int i = 0; i < p0.size(); ++i) p0[i] = 0.5f * rng.next_unit();
	for (unsigned int i = 0; i < m; ++i) { weight[i] = 0.3f * rng.next_unit(); bias[i] = 0.1f * rng.next_unit(); }

	std::vector<float> q(q0), p(p0);
	std::vector<float> u(T * m);

	// Forward shear: p += f(q).
	potential_f_apply_all(&q[0], &weight[0], &bias[0], &u[0], T, m);
	for (unsigned int t = 0; t < T; ++t)
		glades::chiron::shear_add_to_p(&p[t * m], &u[t * m], m);

	// Inverse shear: p -= f(q_new).  q unchanged by Shear^p, so f(q_new) = f(q).
	potential_f_apply_all(&q[0], &weight[0], &bias[0], &u[0], T, m);
	for (unsigned int t = 0; t < T; ++t)
		glades::chiron::shear_sub_from_p(&p[t * m], &u[t * m], m);

	const float q_err = max_abs_diff(q, q0);
	const float p_err = max_abs_diff(p, p0);
	char msg[256];
	std::snprintf(msg, sizeof(msg),
	              "CHIRON shear reversibility: q_err=%.3e p_err=%.3e (want < 1e-6)",
	              q_err, p_err);
	ASSERT(msg, q_err < 1e-6f && p_err < 1e-6f);
}

// ---------------------------------------------------------------------------
// Case 2: reversible-LayerNorm round-trip.
// Forward ReLN followed by inverse ReLN should recover q exactly in FP32.
// ---------------------------------------------------------------------------
void CHIRONReLNRoundtripTest()
{
	const unsigned int T = 4;
	const unsigned int m = 8;
	const float eps = 1e-4f;

	LCG rng(31337u);
	std::vector<float> q0(T * m), gamma(m), beta(m);
	std::vector<float> stats(T * 2u);

	for (unsigned int i = 0; i < q0.size(); ++i) q0[i] = 0.8f * rng.next_unit();
	// Keep gamma well away from zero to avoid ill-conditioned inverse.
	for (unsigned int i = 0; i < m; ++i)
	{
		gamma[i] = 1.0f + 0.2f * rng.next_unit();
		beta[i]  = 0.1f * rng.next_unit();
	}

	std::vector<float> q_fwd(T * m);
	glades::chiron::reln_forward(&q0[0], &q_fwd[0], &stats[0],
	                              &gamma[0], &beta[0], T, m, eps);

	// Stats must be non-trivially populated.
	float stats_mag = 0.0f;
	for (unsigned int i = 0; i < stats.size(); ++i) stats_mag += fabsf(stats[i]);
	ASSERT("CHIRON ReLN forward must populate external stats buffer", stats_mag > 1e-6f);

	std::vector<float> q_back(T * m);
	glades::chiron::reln_inverse(&q_fwd[0], &q_back[0], &stats[0],
	                              &gamma[0], &beta[0], T, m);

	const float q_err = max_abs_diff(q_back, q0);
	char msg[256];
	std::snprintf(msg, sizeof(msg),
	              "CHIRON ReLN roundtrip: q_err=%.3e (want < 1e-4)", q_err);
	ASSERT(msg, q_err < 1e-4f);
}

// ---------------------------------------------------------------------------
// Case 3: full (reduced) CHIRON block round-trip — composition of
//   Shear^p  ∘  Shear^q  ∘  ReLN
// No attention yet (attention shear is a follow-up iteration).
// ---------------------------------------------------------------------------
void CHIRONBlockRoundtripTest()
{
	const unsigned int T = 4;
	const unsigned int m = 8;
	const float eps = 1e-4f;

	LCG rng(2718u);

	std::vector<float> q0(T * m), p0(T * m);
	std::vector<float> mlp_p_weight(m), mlp_p_bias(m);
	std::vector<float> mlp_q_weight(m), mlp_q_bias(m);
	std::vector<float> gamma(m), beta(m);
	std::vector<float> stats(T * 2u);

	for (unsigned int i = 0; i < q0.size(); ++i) q0[i] = 0.5f * rng.next_unit();
	for (unsigned int i = 0; i < p0.size(); ++i) p0[i] = 0.5f * rng.next_unit();
	for (unsigned int i = 0; i < m; ++i)
	{
		mlp_p_weight[i] = 0.3f * rng.next_unit();
		mlp_p_bias[i]   = 0.1f * rng.next_unit();
		mlp_q_weight[i] = 0.3f * rng.next_unit();
		mlp_q_bias[i]   = 0.1f * rng.next_unit();
		gamma[i] = 1.0f + 0.2f * rng.next_unit();
		beta[i]  = 0.1f * rng.next_unit();
	}

	std::vector<float> q(q0), p(p0);
	std::vector<float> u(T * m);

	// FORWARD block
	// 1. Shear^p:  p += f_p(q)
	potential_f_apply_all(&q[0], &mlp_p_weight[0], &mlp_p_bias[0], &u[0], T, m);
	for (unsigned int t = 0; t < T; ++t)
		glades::chiron::shear_add_to_p(&p[t * m], &u[t * m], m);

	// 2. Shear^q:  q += f_q(p)
	potential_f_apply_all(&p[0], &mlp_q_weight[0], &mlp_q_bias[0], &u[0], T, m);
	for (unsigned int t = 0; t < T; ++t)
		glades::chiron::shear_add_to_q(&q[t * m], &u[t * m], m);

	// 3. ReLN: q' = normalize(q); stats saved to external buffer.
	std::vector<float> q_norm(T * m);
	glades::chiron::reln_forward(&q[0], &q_norm[0], &stats[0],
	                              &gamma[0], &beta[0], T, m, eps);
	q.swap(q_norm);

	// INVERSE block
	// 3^-1. ReLN inverse.
	std::vector<float> q_back(T * m);
	glades::chiron::reln_inverse(&q[0], &q_back[0], &stats[0],
	                              &gamma[0], &beta[0], T, m);
	q.swap(q_back);

	// 2^-1. Shear^q inverse: q -= f_q(p). p unchanged by Shear^q and by ReLN
	// (external stats), so f_q(p) equals its forward value.
	potential_f_apply_all(&p[0], &mlp_q_weight[0], &mlp_q_bias[0], &u[0], T, m);
	for (unsigned int t = 0; t < T; ++t)
		glades::chiron::shear_sub_from_q(&q[t * m], &u[t * m], m);

	// 1^-1. Shear^p inverse: p -= f_p(q). q unchanged by Shear^p.
	potential_f_apply_all(&q[0], &mlp_p_weight[0], &mlp_p_bias[0], &u[0], T, m);
	for (unsigned int t = 0; t < T; ++t)
		glades::chiron::shear_sub_from_p(&p[t * m], &u[t * m], m);

	const float q_err = max_abs_diff(q, q0);
	const float p_err = max_abs_diff(p, p0);

	char msg[256];
	std::snprintf(msg, sizeof(msg),
	              "CHIRON block roundtrip: q_err=%.3e p_err=%.3e (want < 1e-4)",
	              q_err, p_err);
	ASSERT(msg, q_err < 1e-4f && p_err < 1e-5f);
}

// ---------------------------------------------------------------------------
// Case 4: multi-block round-trip with L = 8 blocks using the external-stats
// ReLN API.  Verifies composition-of-inverses is correct across L layers.
// ---------------------------------------------------------------------------
void CHIRONMultiBlockRoundtripTest()
{
	const unsigned int T = 8;
	const unsigned int m = 16;
	const unsigned int L = 8;
	const float eps = 1e-4f;

	LCG rng(9001u);

	std::vector<float> q0(T * m), p0(T * m);
	for (unsigned int i = 0; i < q0.size(); ++i) q0[i] = 0.5f * rng.next_unit();
	for (unsigned int i = 0; i < p0.size(); ++i) p0[i] = 0.5f * rng.next_unit();

	// Per-block parameters.
	std::vector<std::vector<float> > w_p(L), b_p(L), w_q(L), b_q(L), gamma(L), beta(L);
	for (unsigned int l = 0; l < L; ++l)
	{
		w_p[l].resize(m); b_p[l].resize(m);
		w_q[l].resize(m); b_q[l].resize(m);
		gamma[l].resize(m); beta[l].resize(m);
		for (unsigned int i = 0; i < m; ++i)
		{
			w_p[l][i] = 0.2f * rng.next_unit();
			b_p[l][i] = 0.05f * rng.next_unit();
			w_q[l][i] = 0.2f * rng.next_unit();
			b_q[l][i] = 0.05f * rng.next_unit();
			gamma[l][i] = 1.0f + 0.15f * rng.next_unit();
			beta[l][i]  = 0.05f * rng.next_unit();
		}
	}

	// Stats buffer [L, T, 2]: the only cross-block per-token memory CHIRON holds.
	std::vector<float> stats(L * T * 2u, 0.0f);

	std::vector<float> q(q0), p(p0);
	std::vector<float> u(T * m);
	std::vector<float> q_tmp(T * m);

	// Forward L blocks.
	for (unsigned int l = 0; l < L; ++l)
	{
		// 1. Shear^p
		potential_f_apply_all(&q[0], &w_p[l][0], &b_p[l][0], &u[0], T, m);
		for (unsigned int t = 0; t < T; ++t)
			glades::chiron::shear_add_to_p(&p[t * m], &u[t * m], m);
		// 2. Shear^q
		potential_f_apply_all(&p[0], &w_q[l][0], &b_q[l][0], &u[0], T, m);
		for (unsigned int t = 0; t < T; ++t)
			glades::chiron::shear_add_to_q(&q[t * m], &u[t * m], m);
		// 3. ReLN: stats -> stats[l, :, :]
		float* stats_l = &stats[l * T * 2u];
		glades::chiron::reln_forward(&q[0], &q_tmp[0], stats_l,
		                              &gamma[l][0], &beta[l][0],
		                              T, m, eps);
		q.swap(q_tmp);
	}

	// Inverse L blocks, in reverse order.
	for (unsigned int ll = 0; ll < L; ++ll)
	{
		const unsigned int l = L - 1 - ll;
		// 3^-1 ReLN inverse
		const float* stats_l = &stats[l * T * 2u];
		glades::chiron::reln_inverse(&q[0], &q_tmp[0], stats_l,
		                              &gamma[l][0], &beta[l][0],
		                              T, m);
		q.swap(q_tmp);
		// 2^-1 Shear^q inverse
		potential_f_apply_all(&p[0], &w_q[l][0], &b_q[l][0], &u[0], T, m);
		for (unsigned int t = 0; t < T; ++t)
			glades::chiron::shear_sub_from_q(&q[t * m], &u[t * m], m);
		// 1^-1 Shear^p inverse
		potential_f_apply_all(&q[0], &w_p[l][0], &b_p[l][0], &u[0], T, m);
		for (unsigned int t = 0; t < T; ++t)
			glades::chiron::shear_sub_from_p(&p[t * m], &u[t * m], m);
	}

	const float q_err = max_abs_diff(q, q0);
	const float p_err = max_abs_diff(p, p0);

	char msg[256];
	std::snprintf(msg, sizeof(msg),
	              "CHIRON %u-block roundtrip: q_err=%.3e p_err=%.3e (want < 5e-4)",
	              L, q_err, p_err);
	ASSERT(msg, q_err < 5e-4f && p_err < 5e-5f);

	std::printf("  CHIRON %u-block FP32 reversibility: q_err=%.3e p_err=%.3e\n",
	            L, q_err, p_err);
}

void CHIRONUnitTest()
{
	std::printf("\n=== CHIRON (reversible-flow transformer) unit tests ===\n");
	CHIRONShearReversibilityTest();
	CHIRONReLNRoundtripTest();
	CHIRONBlockRoundtripTest();
	CHIRONMultiBlockRoundtripTest();
	std::printf("=== CHIRON tests done ===\n\n");
}
