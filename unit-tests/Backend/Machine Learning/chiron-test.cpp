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

#ifdef GLADES_HAVE_CUDA
#include "../../../Backend/Machine Learning/Networks/cuda/gpu_chiron.h"
#include "../../../Backend/Machine Learning/Networks/cuda/gpu_device.h"
#include "../../../Backend/Machine Learning/Networks/cuda/gpu_buffer.h"
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
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

// BF16 round-trip (float -> bf16 -> float) to emulate BF16 arithmetic at
// operation boundaries. Mirrors transformer_kernels.h's float_to_bf16_rn
// but inlined here to avoid the glades::rng include chain.
static float bf16_round(float f)
{
	union { float fv; uint32_t u; } v;
	v.fv = f;
	const uint32_t lsb = (v.u >> 16) & 1u;
	const uint32_t roundingBias = 0x7FFFu + lsb;
	const uint16_t bits = static_cast<uint16_t>((v.u + roundingBias) >> 16);
	union { uint32_t u; float fv; } w;
	w.u = static_cast<uint32_t>(bits) << 16;
	return w.fv;
}

// BF16-round a whole buffer in place. Emulates storing activations in BF16.
static void bf16_round_all(float* buf, size_t n)
{
	for (size_t i = 0; i < n; ++i)
		buf[i] = bf16_round(buf[i]);
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

// Minimal dense attention for CHIRON testing (single head, row-major inputs).
// attn_shear_output(q) computes Y = Wo^T · (softmax(Q·K^T/√dH) · V) where
// Q = q·Wq, K = q·Wk, V = q·Wv. All projections from q alone — this is the
// defining symplectic-attention property: the shear direction (q) is the
// sole source of Q, K, V, so the map on (q, p) -> (q, p+Y(q)) is unit
// lower-triangular.
//
// Arguments:
//   q        [T, m]  input
//   Wq       [m, dH] query projection
//   Wk       [m, dH] key projection
//   Wv       [m, dH] value projection
//   Wo       [dH, m] output projection
//   causal   apply causal mask (future tokens masked out)
//   out      [T, m]  output to be added to p
static void chiron_attn_shear(const float* q, const float* Wq, const float* Wk,
                              const float* Wv, const float* Wo,
                              unsigned int T, unsigned int m, unsigned int dH,
                              bool causal, float* out)
{
	// Compute Q, K, V as [T, dH] row-major via q · W{q,k,v}.
	std::vector<float> Q(T * dH, 0.0f), K(T * dH, 0.0f), V(T * dH, 0.0f);
	for (unsigned int t = 0; t < T; ++t)
	{
		for (unsigned int h = 0; h < dH; ++h)
		{
			float qsum = 0.0f, ksum = 0.0f, vsum = 0.0f;
			for (unsigned int i = 0; i < m; ++i)
			{
				qsum += q[t * m + i] * Wq[i * dH + h];
				ksum += q[t * m + i] * Wk[i * dH + h];
				vsum += q[t * m + i] * Wv[i * dH + h];
			}
			Q[t * dH + h] = qsum;
			K[t * dH + h] = ksum;
			V[t * dH + h] = vsum;
		}
	}

	const float inv_sqrt_dH = 1.0f / sqrtf(static_cast<float>(dH));

	// Scaled dot-product attention, per query row.
	std::vector<float> attn_out(T * dH, 0.0f);
	std::vector<float> scores(T);
	for (unsigned int t = 0; t < T; ++t)
	{
		const unsigned int maxU = causal ? (t + 1u) : T;
		float max_score = -1e30f;
		for (unsigned int u = 0; u < maxU; ++u)
		{
			float s = 0.0f;
			for (unsigned int h = 0; h < dH; ++h)
				s += Q[t * dH + h] * K[u * dH + h];
			scores[u] = s * inv_sqrt_dH;
			if (scores[u] > max_score) max_score = scores[u];
		}
		float denom = 0.0f;
		for (unsigned int u = 0; u < maxU; ++u)
		{
			scores[u] = expf(scores[u] - max_score);
			denom += scores[u];
		}
		const float inv_denom = (denom > 0.0f) ? 1.0f / denom : 0.0f;
		for (unsigned int u = 0; u < maxU; ++u)
			scores[u] *= inv_denom;

		for (unsigned int h = 0; h < dH; ++h)
		{
			float acc = 0.0f;
			for (unsigned int u = 0; u < maxU; ++u)
				acc += scores[u] * V[u * dH + h];
			attn_out[t * dH + h] = acc;
		}
	}

	// Output projection: out = attn_out · Wo, where Wo is [dH, m].
	for (unsigned int t = 0; t < T; ++t)
	{
		for (unsigned int i = 0; i < m; ++i)
		{
			float acc = 0.0f;
			for (unsigned int h = 0; h < dH; ++h)
				acc += attn_out[t * dH + h] * Wo[h * m + i];
			out[t * m + i] = acc;
		}
	}
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

// ---------------------------------------------------------------------------
// Case 5: symplectic attention shear reversibility.
// Verifies that the CHIRON attention shear (Q, K, V all from q; output added
// to p) is a unit-lower-triangular bijection whose inverse is obtained by
// re-running the same attention forward on q (which is unchanged by the
// shear) and subtracting from p.
// ---------------------------------------------------------------------------
void CHIRONAttentionShearReversibilityTest()
{
	const unsigned int T = 6;
	const unsigned int m = 16;
	const unsigned int dH = 4;
	const bool causal = true;

	LCG rng(4242u);
	std::vector<float> q0(T * m), p0(T * m);
	std::vector<float> Wq(m * dH), Wk(m * dH), Wv(m * dH), Wo(dH * m);

	for (unsigned int i = 0; i < q0.size(); ++i) q0[i] = 0.5f * rng.next_unit();
	for (unsigned int i = 0; i < p0.size(); ++i) p0[i] = 0.5f * rng.next_unit();

	// Scale projections small so the attention output doesn't dominate p.
	const float init_scale = 0.2f;
	for (unsigned int i = 0; i < Wq.size(); ++i) Wq[i] = init_scale * rng.next_unit();
	for (unsigned int i = 0; i < Wk.size(); ++i) Wk[i] = init_scale * rng.next_unit();
	for (unsigned int i = 0; i < Wv.size(); ++i) Wv[i] = init_scale * rng.next_unit();
	for (unsigned int i = 0; i < Wo.size(); ++i) Wo[i] = init_scale * rng.next_unit();

	std::vector<float> q(q0), p(p0);
	std::vector<float> y(T * m);

	// Forward attention shear: p += Y(q).
	chiron_attn_shear(&q[0], &Wq[0], &Wk[0], &Wv[0], &Wo[0], T, m, dH, causal, &y[0]);
	for (unsigned int t = 0; t < T; ++t)
		glades::chiron::shear_add_to_p(&p[t * m], &y[t * m], m);

	// Sanity: p must have changed.
	float delta_p = 0.0f;
	for (unsigned int i = 0; i < p.size(); ++i) delta_p += fabsf(p[i] - p0[i]);
	ASSERT("CHIRON attention shear must non-trivially update p", delta_p > 1e-4f);

	// Inverse: p -= Y(q_new). q is unchanged by Shear^p, so Y(q_new) = Y(q).
	chiron_attn_shear(&q[0], &Wq[0], &Wk[0], &Wv[0], &Wo[0], T, m, dH, causal, &y[0]);
	for (unsigned int t = 0; t < T; ++t)
		glades::chiron::shear_sub_from_p(&p[t * m], &y[t * m], m);

	const float q_err = max_abs_diff(q, q0);
	const float p_err = max_abs_diff(p, p0);

	char msg[256];
	std::snprintf(msg, sizeof(msg),
	              "CHIRON attention shear reversibility: q_err=%.3e p_err=%.3e (want < 1e-5)",
	              q_err, p_err);
	ASSERT(msg, q_err < 1e-6f && p_err < 1e-5f);
}

// ---------------------------------------------------------------------------
// Case 6: full CHIRON block round-trip — attention shear + two MLP shears +
// ReLN. This is the full forward block described in §3.5 of the framework
// (modulo that MLPs here are the elementwise tanh potential from earlier
// tests rather than a learned two-layer net; the reversibility argument is
// the same).
// ---------------------------------------------------------------------------
void CHIRONFullBlockRoundtripTest()
{
	const unsigned int T = 6;
	const unsigned int m = 16;
	const unsigned int dH = 4;
	const bool causal = true;
	const float eps = 1e-4f;

	LCG rng(7777u);

	std::vector<float> q0(T * m), p0(T * m);
	std::vector<float> Wq(m * dH), Wk(m * dH), Wv(m * dH), Wo(dH * m);
	std::vector<float> mlp_p_weight(m), mlp_p_bias(m);
	std::vector<float> mlp_q_weight(m), mlp_q_bias(m);
	std::vector<float> gamma(m), beta(m);
	std::vector<float> stats(T * 2u);

	for (unsigned int i = 0; i < q0.size(); ++i) q0[i] = 0.5f * rng.next_unit();
	for (unsigned int i = 0; i < p0.size(); ++i) p0[i] = 0.5f * rng.next_unit();
	const float init_scale = 0.2f;
	for (unsigned int i = 0; i < Wq.size(); ++i) Wq[i] = init_scale * rng.next_unit();
	for (unsigned int i = 0; i < Wk.size(); ++i) Wk[i] = init_scale * rng.next_unit();
	for (unsigned int i = 0; i < Wv.size(); ++i) Wv[i] = init_scale * rng.next_unit();
	for (unsigned int i = 0; i < Wo.size(); ++i) Wo[i] = init_scale * rng.next_unit();
	for (unsigned int i = 0; i < m; ++i)
	{
		mlp_p_weight[i] = 0.25f * rng.next_unit();
		mlp_p_bias[i]   = 0.05f * rng.next_unit();
		mlp_q_weight[i] = 0.25f * rng.next_unit();
		mlp_q_bias[i]   = 0.05f * rng.next_unit();
		gamma[i] = 1.0f + 0.15f * rng.next_unit();
		beta[i]  = 0.05f * rng.next_unit();
	}

	std::vector<float> q(q0), p(p0);
	std::vector<float> y(T * m);
	std::vector<float> u(T * m);

	// FORWARD block: Attn -> Shear^p_MLP -> Shear^q_MLP -> ReLN
	// 1. Attention shear: p += Y(q)
	chiron_attn_shear(&q[0], &Wq[0], &Wk[0], &Wv[0], &Wo[0], T, m, dH, causal, &y[0]);
	for (unsigned int t = 0; t < T; ++t)
		glades::chiron::shear_add_to_p(&p[t * m], &y[t * m], m);

	// 2. Symplectic MLP on q (updates p): p += f_p(q)
	potential_f_apply_all(&q[0], &mlp_p_weight[0], &mlp_p_bias[0], &u[0], T, m);
	for (unsigned int t = 0; t < T; ++t)
		glades::chiron::shear_add_to_p(&p[t * m], &u[t * m], m);

	// 3. Symplectic MLP on p (updates q): q += f_q(p)
	potential_f_apply_all(&p[0], &mlp_q_weight[0], &mlp_q_bias[0], &u[0], T, m);
	for (unsigned int t = 0; t < T; ++t)
		glades::chiron::shear_add_to_q(&q[t * m], &u[t * m], m);

	// 4. ReLN
	std::vector<float> q_norm(T * m);
	glades::chiron::reln_forward(&q[0], &q_norm[0], &stats[0],
	                              &gamma[0], &beta[0], T, m, eps);
	q.swap(q_norm);

	// INVERSE block in reverse order.
	// 4^-1 ReLN inverse
	std::vector<float> q_back(T * m);
	glades::chiron::reln_inverse(&q[0], &q_back[0], &stats[0],
	                              &gamma[0], &beta[0], T, m);
	q.swap(q_back);

	// 3^-1 Symplectic MLP on p inverse: q -= f_q(p)
	potential_f_apply_all(&p[0], &mlp_q_weight[0], &mlp_q_bias[0], &u[0], T, m);
	for (unsigned int t = 0; t < T; ++t)
		glades::chiron::shear_sub_from_q(&q[t * m], &u[t * m], m);

	// 2^-1 Symplectic MLP on q inverse: p -= f_p(q)
	potential_f_apply_all(&q[0], &mlp_p_weight[0], &mlp_p_bias[0], &u[0], T, m);
	for (unsigned int t = 0; t < T; ++t)
		glades::chiron::shear_sub_from_p(&p[t * m], &u[t * m], m);

	// 1^-1 Attention shear inverse: p -= Y(q)
	chiron_attn_shear(&q[0], &Wq[0], &Wk[0], &Wv[0], &Wo[0], T, m, dH, causal, &y[0]);
	for (unsigned int t = 0; t < T; ++t)
		glades::chiron::shear_sub_from_p(&p[t * m], &y[t * m], m);

	const float q_err = max_abs_diff(q, q0);
	const float p_err = max_abs_diff(p, p0);

	char msg[256];
	std::snprintf(msg, sizeof(msg),
	              "CHIRON full-block roundtrip: q_err=%.3e p_err=%.3e (want < 5e-4)",
	              q_err, p_err);
	ASSERT(msg, q_err < 5e-4f && p_err < 5e-5f);

	std::printf("  CHIRON full-block FP32 reversibility: q_err=%.3e p_err=%.3e\n",
	            q_err, p_err);
}

// ---------------------------------------------------------------------------
// Case 7: full-block multi-layer roundtrip. Compose L=4 full CHIRON blocks
// (attention + MLP shears + ReLN) and invert them. This is the most
// integrated reversibility check — any bug in any of the four block
// components at any layer will surface here.
// ---------------------------------------------------------------------------
void CHIRONMultiFullBlockRoundtripTest()
{
	const unsigned int T = 6;
	const unsigned int m = 16;
	const unsigned int dH = 4;
	const unsigned int L = 4;
	const bool causal = true;
	const float eps = 1e-4f;

	LCG rng(12345u);

	std::vector<float> q0(T * m), p0(T * m);
	for (unsigned int i = 0; i < q0.size(); ++i) q0[i] = 0.5f * rng.next_unit();
	for (unsigned int i = 0; i < p0.size(); ++i) p0[i] = 0.5f * rng.next_unit();

	// Per-block parameters.
	std::vector<std::vector<float> > Wq(L), Wk(L), Wv(L), Wo(L);
	std::vector<std::vector<float> > w_p(L), b_p(L), w_q(L), b_q(L);
	std::vector<std::vector<float> > gamma(L), beta(L);
	for (unsigned int l = 0; l < L; ++l)
	{
		Wq[l].resize(m * dH); Wk[l].resize(m * dH);
		Wv[l].resize(m * dH); Wo[l].resize(dH * m);
		w_p[l].resize(m); b_p[l].resize(m);
		w_q[l].resize(m); b_q[l].resize(m);
		gamma[l].resize(m); beta[l].resize(m);
		for (unsigned int i = 0; i < m * dH; ++i)
		{
			Wq[l][i] = 0.15f * rng.next_unit();
			Wk[l][i] = 0.15f * rng.next_unit();
			Wv[l][i] = 0.15f * rng.next_unit();
		}
		for (unsigned int i = 0; i < dH * m; ++i)
			Wo[l][i] = 0.15f * rng.next_unit();
		for (unsigned int i = 0; i < m; ++i)
		{
			w_p[l][i] = 0.2f * rng.next_unit();
			b_p[l][i] = 0.03f * rng.next_unit();
			w_q[l][i] = 0.2f * rng.next_unit();
			b_q[l][i] = 0.03f * rng.next_unit();
			gamma[l][i] = 1.0f + 0.1f * rng.next_unit();
			beta[l][i]  = 0.03f * rng.next_unit();
		}
	}

	std::vector<float> stats(L * T * 2u, 0.0f);

	std::vector<float> q(q0), p(p0);
	std::vector<float> y(T * m), u(T * m), q_tmp(T * m);

	// Forward L full blocks.
	for (unsigned int l = 0; l < L; ++l)
	{
		// 1. Attention shear
		chiron_attn_shear(&q[0], &Wq[l][0], &Wk[l][0], &Wv[l][0], &Wo[l][0],
		                   T, m, dH, causal, &y[0]);
		for (unsigned int t = 0; t < T; ++t)
			glades::chiron::shear_add_to_p(&p[t * m], &y[t * m], m);
		// 2. Symplectic MLP on q: p += f_p(q)
		potential_f_apply_all(&q[0], &w_p[l][0], &b_p[l][0], &u[0], T, m);
		for (unsigned int t = 0; t < T; ++t)
			glades::chiron::shear_add_to_p(&p[t * m], &u[t * m], m);
		// 3. Symplectic MLP on p: q += f_q(p)
		potential_f_apply_all(&p[0], &w_q[l][0], &b_q[l][0], &u[0], T, m);
		for (unsigned int t = 0; t < T; ++t)
			glades::chiron::shear_add_to_q(&q[t * m], &u[t * m], m);
		// 4. ReLN
		float* stats_l = &stats[l * T * 2u];
		glades::chiron::reln_forward(&q[0], &q_tmp[0], stats_l,
		                              &gamma[l][0], &beta[l][0],
		                              T, m, eps);
		q.swap(q_tmp);
	}

	// Inverse L full blocks, reverse order.
	for (unsigned int ll = 0; ll < L; ++ll)
	{
		const unsigned int l = L - 1 - ll;
		// 4^-1 ReLN inverse
		const float* stats_l = &stats[l * T * 2u];
		glades::chiron::reln_inverse(&q[0], &q_tmp[0], stats_l,
		                              &gamma[l][0], &beta[l][0],
		                              T, m);
		q.swap(q_tmp);
		// 3^-1 Shear^q inverse
		potential_f_apply_all(&p[0], &w_q[l][0], &b_q[l][0], &u[0], T, m);
		for (unsigned int t = 0; t < T; ++t)
			glades::chiron::shear_sub_from_q(&q[t * m], &u[t * m], m);
		// 2^-1 Shear^p inverse
		potential_f_apply_all(&q[0], &w_p[l][0], &b_p[l][0], &u[0], T, m);
		for (unsigned int t = 0; t < T; ++t)
			glades::chiron::shear_sub_from_p(&p[t * m], &u[t * m], m);
		// 1^-1 Attention shear inverse
		chiron_attn_shear(&q[0], &Wq[l][0], &Wk[l][0], &Wv[l][0], &Wo[l][0],
		                   T, m, dH, causal, &y[0]);
		for (unsigned int t = 0; t < T; ++t)
			glades::chiron::shear_sub_from_p(&p[t * m], &y[t * m], m);
	}

	const float q_err = max_abs_diff(q, q0);
	const float p_err = max_abs_diff(p, p0);

	char msg[256];
	std::snprintf(msg, sizeof(msg),
	              "CHIRON %u-full-block roundtrip: q_err=%.3e p_err=%.3e (want < 5e-4)",
	              L, q_err, p_err);
	ASSERT(msg, q_err < 5e-4f && p_err < 5e-5f);

	std::printf("  CHIRON %u-full-block FP32 reversibility: q_err=%.3e p_err=%.3e\n",
	            L, q_err, p_err);
}

// ---------------------------------------------------------------------------
// Case 8: BF16 reconstruction drift (negative control).
// Runs the forward block chain in BF16 precision (each op's output is
// rounded to BF16), then runs the inverse chain in BF16. Measures the
// reconstruction error `‖x̂_0 − x_0‖_∞` as a function of L.
// Expected (per framework §6.4): without sketch correction the error grows
// approximately linearly with L times ε_BF16 ≈ 2^−8 ≈ 4e-3, with a
// multiplicative factor from the Lipschitz constant of the block.
// This test does NOT assert a tight bound on drift — its purpose is to
// PROVE that BF16 reconstruction is materially worse than FP32 (thus
// motivating the sketch correction in the next phase). It asserts:
//   (1) BF16 error > FP32 error by an order of magnitude at L=4.
//   (2) BF16 error grows with L (at L=12 strictly larger than L=4).
// ---------------------------------------------------------------------------
static float run_multifullblock_roundtrip(unsigned int L, unsigned int T,
                                          unsigned int m, unsigned int dH,
                                          bool causal, float eps,
                                          unsigned int seed, bool bf16_emul)
{
	LCG rng(seed);

	std::vector<float> q0(T * m), p0(T * m);
	for (unsigned int i = 0; i < q0.size(); ++i) q0[i] = 0.5f * rng.next_unit();
	for (unsigned int i = 0; i < p0.size(); ++i) p0[i] = 0.5f * rng.next_unit();

	std::vector<std::vector<float> > Wq(L), Wk(L), Wv(L), Wo(L);
	std::vector<std::vector<float> > w_p(L), b_p(L), w_q(L), b_q(L);
	std::vector<std::vector<float> > gamma(L), beta(L);
	for (unsigned int l = 0; l < L; ++l)
	{
		Wq[l].resize(m * dH); Wk[l].resize(m * dH);
		Wv[l].resize(m * dH); Wo[l].resize(dH * m);
		w_p[l].resize(m); b_p[l].resize(m);
		w_q[l].resize(m); b_q[l].resize(m);
		gamma[l].resize(m); beta[l].resize(m);
		for (unsigned int i = 0; i < m * dH; ++i)
		{
			Wq[l][i] = 0.15f * rng.next_unit();
			Wk[l][i] = 0.15f * rng.next_unit();
			Wv[l][i] = 0.15f * rng.next_unit();
		}
		for (unsigned int i = 0; i < dH * m; ++i)
			Wo[l][i] = 0.15f * rng.next_unit();
		for (unsigned int i = 0; i < m; ++i)
		{
			w_p[l][i] = 0.2f * rng.next_unit();
			b_p[l][i] = 0.03f * rng.next_unit();
			w_q[l][i] = 0.2f * rng.next_unit();
			b_q[l][i] = 0.03f * rng.next_unit();
			gamma[l][i] = 1.0f + 0.1f * rng.next_unit();
			beta[l][i]  = 0.03f * rng.next_unit();
		}
		if (bf16_emul)
		{
			bf16_round_all(&Wq[l][0], Wq[l].size());
			bf16_round_all(&Wk[l][0], Wk[l].size());
			bf16_round_all(&Wv[l][0], Wv[l].size());
			bf16_round_all(&Wo[l][0], Wo[l].size());
			bf16_round_all(&w_p[l][0], w_p[l].size());
			bf16_round_all(&b_p[l][0], b_p[l].size());
			bf16_round_all(&w_q[l][0], w_q[l].size());
			bf16_round_all(&b_q[l][0], b_q[l].size());
			bf16_round_all(&gamma[l][0], gamma[l].size());
			bf16_round_all(&beta[l][0], beta[l].size());
		}
	}

	std::vector<float> stats(L * T * 2u, 0.0f);
	std::vector<float> q(q0), p(p0);
	std::vector<float> y(T * m), u(T * m), q_tmp(T * m);

	if (bf16_emul)
	{
		bf16_round_all(&q[0], q.size());
		bf16_round_all(&p[0], p.size());
	}

	// Forward.
	for (unsigned int l = 0; l < L; ++l)
	{
		chiron_attn_shear(&q[0], &Wq[l][0], &Wk[l][0], &Wv[l][0], &Wo[l][0],
		                   T, m, dH, causal, &y[0]);
		if (bf16_emul) bf16_round_all(&y[0], y.size());
		for (unsigned int t = 0; t < T; ++t)
			glades::chiron::shear_add_to_p(&p[t * m], &y[t * m], m);
		if (bf16_emul) bf16_round_all(&p[0], p.size());

		potential_f_apply_all(&q[0], &w_p[l][0], &b_p[l][0], &u[0], T, m);
		if (bf16_emul) bf16_round_all(&u[0], u.size());
		for (unsigned int t = 0; t < T; ++t)
			glades::chiron::shear_add_to_p(&p[t * m], &u[t * m], m);
		if (bf16_emul) bf16_round_all(&p[0], p.size());

		potential_f_apply_all(&p[0], &w_q[l][0], &b_q[l][0], &u[0], T, m);
		if (bf16_emul) bf16_round_all(&u[0], u.size());
		for (unsigned int t = 0; t < T; ++t)
			glades::chiron::shear_add_to_q(&q[t * m], &u[t * m], m);
		if (bf16_emul) bf16_round_all(&q[0], q.size());

		float* stats_l = &stats[l * T * 2u];
		glades::chiron::reln_forward(&q[0], &q_tmp[0], stats_l,
		                              &gamma[l][0], &beta[l][0], T, m, eps);
		if (bf16_emul)
		{
			bf16_round_all(&q_tmp[0], q_tmp.size());
			// Stats kept in FP32 per the framework (§4.4: z_ℓ in FP32).
		}
		q.swap(q_tmp);
	}

	// Inverse.
	for (unsigned int ll = 0; ll < L; ++ll)
	{
		const unsigned int l = L - 1 - ll;
		const float* stats_l = &stats[l * T * 2u];
		glades::chiron::reln_inverse(&q[0], &q_tmp[0], stats_l,
		                              &gamma[l][0], &beta[l][0], T, m);
		if (bf16_emul) bf16_round_all(&q_tmp[0], q_tmp.size());
		q.swap(q_tmp);

		potential_f_apply_all(&p[0], &w_q[l][0], &b_q[l][0], &u[0], T, m);
		if (bf16_emul) bf16_round_all(&u[0], u.size());
		for (unsigned int t = 0; t < T; ++t)
			glades::chiron::shear_sub_from_q(&q[t * m], &u[t * m], m);
		if (bf16_emul) bf16_round_all(&q[0], q.size());

		potential_f_apply_all(&q[0], &w_p[l][0], &b_p[l][0], &u[0], T, m);
		if (bf16_emul) bf16_round_all(&u[0], u.size());
		for (unsigned int t = 0; t < T; ++t)
			glades::chiron::shear_sub_from_p(&p[t * m], &u[t * m], m);
		if (bf16_emul) bf16_round_all(&p[0], p.size());

		chiron_attn_shear(&q[0], &Wq[l][0], &Wk[l][0], &Wv[l][0], &Wo[l][0],
		                   T, m, dH, causal, &y[0]);
		if (bf16_emul) bf16_round_all(&y[0], y.size());
		for (unsigned int t = 0; t < T; ++t)
			glades::chiron::shear_sub_from_p(&p[t * m], &y[t * m], m);
		if (bf16_emul) bf16_round_all(&p[0], p.size());
	}

	// If we emulate BF16 we must also bf16-round the reference start to
	// measure drift correctly (else we compare FP32-precision original
	// against BF16-precision reconstruction, which conflates the two
	// error sources).
	std::vector<float> q0_bf16(q0), p0_bf16(p0);
	if (bf16_emul)
	{
		bf16_round_all(&q0_bf16[0], q0_bf16.size());
		bf16_round_all(&p0_bf16[0], p0_bf16.size());
	}

	const float qe = max_abs_diff(q, q0_bf16);
	const float pe = max_abs_diff(p, p0_bf16);
	return (qe > pe) ? qe : pe;
}

void CHIRONBf16DriftTest()
{
	const unsigned int T = 6;
	const unsigned int m = 16;
	const unsigned int dH = 4;
	const bool causal = true;
	const float eps = 1e-4f;

	const float fp32_L4  = run_multifullblock_roundtrip(4u,  T, m, dH, causal, eps, 24601u, false);
	const float bf16_L4  = run_multifullblock_roundtrip(4u,  T, m, dH, causal, eps, 24601u, true);
	const float bf16_L12 = run_multifullblock_roundtrip(12u, T, m, dH, causal, eps, 24601u, true);

	std::printf("  CHIRON drift: FP32 L=4 err=%.3e | BF16 L=4 err=%.3e | BF16 L=12 err=%.3e\n",
	            fp32_L4, bf16_L4, bf16_L12);

	char msg[256];
	// Assertion 1: BF16 is materially worse than FP32.
	std::snprintf(msg, sizeof(msg),
	              "BF16 at L=4 should be materially worse than FP32 at L=4: "
	              "fp32_L4=%.3e bf16_L4=%.3e (want bf16 > 10x fp32)",
	              fp32_L4, bf16_L4);
	ASSERT(msg, bf16_L4 > 10.0f * fp32_L4);

	// Assertion 2: drift grows with depth under BF16.
	std::snprintf(msg, sizeof(msg),
	              "BF16 drift must grow with L: bf16_L4=%.3e bf16_L12=%.3e",
	              bf16_L4, bf16_L12);
	ASSERT(msg, bf16_L12 > bf16_L4);

	// Assertion 3: BF16 drift at L=12 should be in the 10^-3 to 1 regime.
	// This is NOT a tight bound; it's a loose sanity check that our
	// emulation produces reasonable-magnitude drift, not astronomical
	// (which would indicate a bug in the emulation).
	ASSERT("BF16 L=12 drift must be in loose expected regime (< 10)",
	       bf16_L12 < 10.0f);
}

// Generate a Gaussian sketch matrix S [r, N] with deterministic seed.
// Uses Box-Muller on top of the LCG so the result is N(0, 1) per entry.
// Returns the dense matrix in row-major.
static void generate_gaussian_sketch(std::vector<float>& S, unsigned int r,
                                     unsigned int N, unsigned int seed)
{
	S.resize(static_cast<size_t>(r) * static_cast<size_t>(N));
	LCG rng(seed);
	size_t i = 0;
	const size_t total = static_cast<size_t>(r) * static_cast<size_t>(N);
	while (i < total)
	{
		// Box-Muller: draw u1, u2 in (0, 1); produce two N(0,1) samples.
		float u1 = 0.5f * (rng.next_unit() + 1.0f); // in (0, 1)
		float u2 = 0.5f * (rng.next_unit() + 1.0f);
		if (u1 < 1e-7f) u1 = 1e-7f;
		const float radius = sqrtf(-2.0f * logf(u1));
		const float theta = 6.28318530717958647692f * u2;
		const float z0 = radius * cosf(theta);
		const float z1 = radius * sinf(theta);
		S[i] = z0;
		if (i + 1 < total)
		{
			S[i + 1] = z1;
			i += 2;
		}
		else
		{
			i += 1;
		}
	}
}

// ---------------------------------------------------------------------------
// Case 9: sketch-correction primitive — CHIRON use case.
// Tests: given x and a perturbed x̃ = x + δ with small ||δ||, the sketch
// correction x̂ = x̃ + (S^T / r)(S x − S x̃) = x̃ + (S^T S / r) · (−δ + ...)
// produces x̂ ≈ x with per-coord error O(||δ|| · √(N/r)).
//
// This is the actual JL identity used by CHIRON's backward pass: we don't
// reconstruct x from z alone (which would take r ≫ N); we *correct* a
// nearly-correct x̃ toward x using the stored z.
// ---------------------------------------------------------------------------
void CHIRONSketchProjectLiftTest()
{
	const unsigned int N = 192;
	const unsigned int r = 1024; // oversample 5x over N

	LCG rng(5555u);
	std::vector<float> x(N);
	for (unsigned int i = 0; i < N; ++i) x[i] = rng.next_unit();

	// Perturb x by a small BF16-ish delta.
	std::vector<float> delta(N);
	const float delta_scale = 1e-2f; // ~BF16 drift at L=4
	for (unsigned int i = 0; i < N; ++i) delta[i] = delta_scale * rng.next_unit();

	std::vector<float> x_tilde(N);
	for (unsigned int i = 0; i < N; ++i) x_tilde[i] = x[i] + delta[i];

	std::vector<float> S;
	generate_gaussian_sketch(S, r, N, 8675309u);

	// Store sketch of true x.
	std::vector<float> z_stored(r, 0.0f);
	glades::chiron::sketch_project(&S[0], &x[0], r, N, &z_stored[0]);

	// Sketch x̃ and compute residual in sketch space.
	std::vector<float> z_fresh(r, 0.0f);
	glades::chiron::sketch_project(&S[0], &x_tilde[0], r, N, &z_fresh[0]);
	std::vector<float> residual(r);
	for (unsigned int k = 0; k < r; ++k) residual[k] = z_stored[k] - z_fresh[k];

	// Apply correction: x̂ = x̃ + (S^T / r) · residual.
	std::vector<float> x_hat(x_tilde);
	glades::chiron::sketch_lift_add(&S[0], &residual[0], r, N, &x_hat[0]);

	// Measure max per-coord |x̂ - x| vs. uncorrected |x̃ - x| = |δ|.
	float max_raw = 0.0f, max_corrected = 0.0f;
	for (unsigned int i = 0; i < N; ++i)
	{
		const float raw_err = fabsf(x_tilde[i] - x[i]);
		const float cor_err = fabsf(x_hat[i] - x[i]);
		if (raw_err > max_raw) max_raw = raw_err;
		if (cor_err > max_corrected) max_corrected = cor_err;
	}

	std::printf("  CHIRON sketch correction primitive: r=%u N=%u delta=%g max_raw=%.4e "
	            "max_corrected=%.4e reduction=%.2fx\n",
	            r, N, delta_scale, max_raw, max_corrected,
	            max_corrected > 0.0f ? (max_raw / max_corrected) : 0.0f);

	char msg[256];
	// The sketch correction must reduce the max error.
	std::snprintf(msg, sizeof(msg),
	              "sketch correction should reduce max error: "
	              "raw=%.3e corrected=%.3e", max_raw, max_corrected);
	ASSERT(msg, max_corrected < max_raw);

	// Expected per-coord noise: ~||δ|| · √(N/r). For r=1024, N=192, ||δ||~1e-2,
	// expect ~4.3e-3. Use a 2x safety margin.
	const float per_coord_bound =
	    2.0f * delta_scale * sqrtf(static_cast<float>(N) / static_cast<float>(r));
	std::snprintf(msg, sizeof(msg),
	              "sketch correction per-coord error should be within "
	              "2·||δ||·√(N/r): max_corrected=%.3e bound=%.3e",
	              max_corrected, per_coord_bound);
	ASSERT(msg, max_corrected < per_coord_bound * 3.0f);  // 3x safety for L_inf vs. RMS
}

// ---------------------------------------------------------------------------
// Case 10: sketch-corrected BF16 reconstruction.
// Runs the full L-block forward+inverse in BF16, with an FP32 sketch stored
// per block at forward time, and applied as a lift-correction at backward
// time. Asserts that the corrected error is materially smaller than the
// uncorrected BF16 error (Case 8).
// ---------------------------------------------------------------------------
static float run_multifullblock_roundtrip_sketch(unsigned int L, unsigned int T,
                                                 unsigned int m, unsigned int dH,
                                                 bool causal, float eps,
                                                 unsigned int seed,
                                                 unsigned int r_sketch,
                                                 unsigned int sketch_seed_base)
{
	// Same setup as run_multifullblock_roundtrip(bf16_emul=true), plus sketch.
	// State x_ℓ at each block is (q, p) concatenated, size N = 2*T*m.
	const unsigned int Nstate = 2u * T * m;

	LCG rng(seed);
	std::vector<float> q0(T * m), p0(T * m);
	for (unsigned int i = 0; i < q0.size(); ++i) q0[i] = 0.5f * rng.next_unit();
	for (unsigned int i = 0; i < p0.size(); ++i) p0[i] = 0.5f * rng.next_unit();

	std::vector<std::vector<float> > Wq(L), Wk(L), Wv(L), Wo(L);
	std::vector<std::vector<float> > w_p(L), b_p(L), w_q(L), b_q(L);
	std::vector<std::vector<float> > gamma(L), beta(L);
	for (unsigned int l = 0; l < L; ++l)
	{
		Wq[l].resize(m * dH); Wk[l].resize(m * dH);
		Wv[l].resize(m * dH); Wo[l].resize(dH * m);
		w_p[l].resize(m); b_p[l].resize(m);
		w_q[l].resize(m); b_q[l].resize(m);
		gamma[l].resize(m); beta[l].resize(m);
		for (unsigned int i = 0; i < m * dH; ++i)
		{
			Wq[l][i] = 0.15f * rng.next_unit();
			Wk[l][i] = 0.15f * rng.next_unit();
			Wv[l][i] = 0.15f * rng.next_unit();
		}
		for (unsigned int i = 0; i < dH * m; ++i)
			Wo[l][i] = 0.15f * rng.next_unit();
		for (unsigned int i = 0; i < m; ++i)
		{
			w_p[l][i] = 0.2f * rng.next_unit();
			b_p[l][i] = 0.03f * rng.next_unit();
			w_q[l][i] = 0.2f * rng.next_unit();
			b_q[l][i] = 0.03f * rng.next_unit();
			gamma[l][i] = 1.0f + 0.1f * rng.next_unit();
			beta[l][i]  = 0.03f * rng.next_unit();
		}
		// BF16-round all weights to match the no-sketch BF16 test setup.
		bf16_round_all(&Wq[l][0], Wq[l].size());
		bf16_round_all(&Wk[l][0], Wk[l].size());
		bf16_round_all(&Wv[l][0], Wv[l].size());
		bf16_round_all(&Wo[l][0], Wo[l].size());
		bf16_round_all(&w_p[l][0], w_p[l].size());
		bf16_round_all(&b_p[l][0], b_p[l].size());
		bf16_round_all(&w_q[l][0], w_q[l].size());
		bf16_round_all(&b_q[l][0], b_q[l].size());
		bf16_round_all(&gamma[l][0], gamma[l].size());
		bf16_round_all(&beta[l][0], beta[l].size());
	}

	// Generate per-layer Gaussian sketches.
	std::vector<std::vector<float> > S(L);
	for (unsigned int l = 0; l < L; ++l)
		generate_gaussian_sketch(S[l], r_sketch, Nstate, sketch_seed_base + l);

	// Per-layer stored sketch values (FP32). Shape [L, r].
	std::vector<std::vector<float> > z_stored(L);
	for (unsigned int l = 0; l < L; ++l) z_stored[l].resize(r_sketch, 0.0f);

	std::vector<float> stats(L * T * 2u, 0.0f);
	std::vector<float> q(q0), p(p0);
	bf16_round_all(&q[0], q.size());
	bf16_round_all(&p[0], p.size());

	std::vector<float> state_flat(Nstate);
	std::vector<float> y(T * m), u(T * m), q_tmp(T * m);

	// Forward: at start of each block, compute and store sketch of the
	// block-input state. Then run the BF16 block.
	for (unsigned int l = 0; l < L; ++l)
	{
		// Concatenate (q, p) into state_flat = [q..., p...].
		std::memcpy(&state_flat[0], &q[0], sizeof(float) * q.size());
		std::memcpy(&state_flat[q.size()], &p[0], sizeof(float) * p.size());
		// Sketch is stored in FP32 (no bf16_round).
		glades::chiron::sketch_project(&S[l][0], &state_flat[0],
		                                r_sketch, Nstate, &z_stored[l][0]);

		// Run block in BF16 (same as no-sketch case).
		chiron_attn_shear(&q[0], &Wq[l][0], &Wk[l][0], &Wv[l][0], &Wo[l][0],
		                   T, m, dH, causal, &y[0]);
		bf16_round_all(&y[0], y.size());
		for (unsigned int t = 0; t < T; ++t)
			glades::chiron::shear_add_to_p(&p[t * m], &y[t * m], m);
		bf16_round_all(&p[0], p.size());

		potential_f_apply_all(&q[0], &w_p[l][0], &b_p[l][0], &u[0], T, m);
		bf16_round_all(&u[0], u.size());
		for (unsigned int t = 0; t < T; ++t)
			glades::chiron::shear_add_to_p(&p[t * m], &u[t * m], m);
		bf16_round_all(&p[0], p.size());

		potential_f_apply_all(&p[0], &w_q[l][0], &b_q[l][0], &u[0], T, m);
		bf16_round_all(&u[0], u.size());
		for (unsigned int t = 0; t < T; ++t)
			glades::chiron::shear_add_to_q(&q[t * m], &u[t * m], m);
		bf16_round_all(&q[0], q.size());

		float* stats_l = &stats[l * T * 2u];
		glades::chiron::reln_forward(&q[0], &q_tmp[0], stats_l,
		                              &gamma[l][0], &beta[l][0], T, m, eps);
		bf16_round_all(&q_tmp[0], q_tmp.size());
		q.swap(q_tmp);
	}

	// Inverse: at each block, run BF16 inverse to get x̃, then correct
	// with the stored sketch.
	std::vector<float> state_tilde(Nstate);
	std::vector<float> z_fresh(r_sketch, 0.0f);
	std::vector<float> residual(r_sketch, 0.0f);
	for (unsigned int ll = 0; ll < L; ++ll)
	{
		const unsigned int l = L - 1 - ll;
		const float* stats_l = &stats[l * T * 2u];
		glades::chiron::reln_inverse(&q[0], &q_tmp[0], stats_l,
		                              &gamma[l][0], &beta[l][0], T, m);
		bf16_round_all(&q_tmp[0], q_tmp.size());
		q.swap(q_tmp);

		potential_f_apply_all(&p[0], &w_q[l][0], &b_q[l][0], &u[0], T, m);
		bf16_round_all(&u[0], u.size());
		for (unsigned int t = 0; t < T; ++t)
			glades::chiron::shear_sub_from_q(&q[t * m], &u[t * m], m);
		bf16_round_all(&q[0], q.size());

		potential_f_apply_all(&q[0], &w_p[l][0], &b_p[l][0], &u[0], T, m);
		bf16_round_all(&u[0], u.size());
		for (unsigned int t = 0; t < T; ++t)
			glades::chiron::shear_sub_from_p(&p[t * m], &u[t * m], m);
		bf16_round_all(&p[0], p.size());

		chiron_attn_shear(&q[0], &Wq[l][0], &Wk[l][0], &Wv[l][0], &Wo[l][0],
		                   T, m, dH, causal, &y[0]);
		bf16_round_all(&y[0], y.size());
		for (unsigned int t = 0; t < T; ++t)
			glades::chiron::shear_sub_from_p(&p[t * m], &y[t * m], m);
		bf16_round_all(&p[0], p.size());

		// At this point, (q, p) is the BF16 approximation x̃_ℓ to the
		// input of block l.  Apply the sketch correction.
		std::memcpy(&state_tilde[0], &q[0], sizeof(float) * q.size());
		std::memcpy(&state_tilde[q.size()], &p[0], sizeof(float) * p.size());
		glades::chiron::sketch_project(&S[l][0], &state_tilde[0],
		                                r_sketch, Nstate, &z_fresh[0]);
		for (unsigned int k = 0; k < r_sketch; ++k)
			residual[k] = z_stored[l][k] - z_fresh[k];
		// Apply correction in place on state_tilde.
		glades::chiron::sketch_lift_add(&S[l][0], &residual[0],
		                                 r_sketch, Nstate, &state_tilde[0]);
		// Split state_tilde back into q, p (in BF16, mimicking lowp storage).
		std::memcpy(&q[0], &state_tilde[0], sizeof(float) * q.size());
		std::memcpy(&p[0], &state_tilde[q.size()], sizeof(float) * p.size());
		bf16_round_all(&q[0], q.size());
		bf16_round_all(&p[0], p.size());
	}

	std::vector<float> q0_bf16(q0), p0_bf16(p0);
	bf16_round_all(&q0_bf16[0], q0_bf16.size());
	bf16_round_all(&p0_bf16[0], p0_bf16.size());
	const float qe = max_abs_diff(q, q0_bf16);
	const float pe = max_abs_diff(p, p0_bf16);
	return (qe > pe) ? qe : pe;
}

void CHIRONSketchCorrectedBf16Test()
{
	const unsigned int T = 6;
	const unsigned int m = 16;
	const unsigned int dH = 4;
	const bool causal = true;
	const float eps = 1e-4f;

	const unsigned int L = 12;
	const unsigned int sketch_seed_base = 111111u;

	// Baseline: same setup WITHOUT sketch (same seed for weights).
	const float bf16_L12_uncorrected =
	    run_multifullblock_roundtrip(L, T, m, dH, causal, eps, 24601u, true);

	// With sketch correction at rank r.
	const float bf16_L12_corrected_r64 =
	    run_multifullblock_roundtrip_sketch(L, T, m, dH, causal, eps,
	                                         24601u, 64u, sketch_seed_base);
	const float bf16_L12_corrected_r256 =
	    run_multifullblock_roundtrip_sketch(L, T, m, dH, causal, eps,
	                                         24601u, 256u, sketch_seed_base);

	std::printf("  CHIRON sketch correction @ L=12: "
	            "uncorrected=%.3e  r=64 corrected=%.3e  r=256 corrected=%.3e\n",
	            bf16_L12_uncorrected, bf16_L12_corrected_r64,
	            bf16_L12_corrected_r256);

	char msg[256];
	// Primary assertion: sketch correction reduces the drift.
	std::snprintf(msg, sizeof(msg),
	              "sketch correction @ r=256 must reduce L=12 BF16 drift: "
	              "uncorrected=%.3e corrected=%.3e",
	              bf16_L12_uncorrected, bf16_L12_corrected_r256);
	ASSERT(msg, bf16_L12_corrected_r256 < bf16_L12_uncorrected);

	// Monotone in r: larger r => lower error (statistical, but should hold
	// for our single seed).
	std::snprintf(msg, sizeof(msg),
	              "sketch correction should improve with r: "
	              "r=64 err=%.3e r=256 err=%.3e",
	              bf16_L12_corrected_r64, bf16_L12_corrected_r256);
	ASSERT(msg, bf16_L12_corrected_r256 <= bf16_L12_corrected_r64 * 1.2f);
}

// ---------------------------------------------------------------------------
// Case 11: per-token local sketch in BF16 pipeline.
//
// Instead of sketching the full flattened block state (size N = 2·T·m),
// apply an independent rank-r sketch to each of the T tokens (size 2m
// each).  The sketch matrix is SHARED across tokens within a single
// layer (for memory efficiency), but different per layer.
//
// Per-coord noise factor per the corrected scaling law:
//   √(N_eff / r)  with  N_eff = 2m  (per token) rather than 2·T·m (global)
//
// For T=6, m=16: global N = 192, per-token N = 32. Same r=256 gives
// √(N_eff/r) = √(32/256) ≈ 0.35 vs. √(192/256) ≈ 0.87 for the global
// sketch — ~2.5× tighter per-coord correction.
// ---------------------------------------------------------------------------
static float run_multifullblock_roundtrip_per_token_sketch(
    unsigned int L, unsigned int T, unsigned int m, unsigned int dH,
    bool causal, float eps, unsigned int seed,
    unsigned int r_sketch, unsigned int sketch_seed_base)
{
	const unsigned int Ntok = 2u * m;  // per-token flattened state size
	const unsigned int qSize = T * m;

	LCG rng(seed);
	std::vector<float> q0(T * m), p0(T * m);
	for (unsigned int i = 0; i < q0.size(); ++i) q0[i] = 0.5f * rng.next_unit();
	for (unsigned int i = 0; i < p0.size(); ++i) p0[i] = 0.5f * rng.next_unit();

	std::vector<std::vector<float> > Wq(L), Wk(L), Wv(L), Wo(L);
	std::vector<std::vector<float> > w_p(L), b_p(L), w_q(L), b_q(L);
	std::vector<std::vector<float> > gamma(L), beta(L);
	for (unsigned int l = 0; l < L; ++l)
	{
		Wq[l].resize(m * dH); Wk[l].resize(m * dH);
		Wv[l].resize(m * dH); Wo[l].resize(dH * m);
		w_p[l].resize(m); b_p[l].resize(m);
		w_q[l].resize(m); b_q[l].resize(m);
		gamma[l].resize(m); beta[l].resize(m);
		for (unsigned int i = 0; i < m * dH; ++i)
		{
			Wq[l][i] = 0.15f * rng.next_unit();
			Wk[l][i] = 0.15f * rng.next_unit();
			Wv[l][i] = 0.15f * rng.next_unit();
		}
		for (unsigned int i = 0; i < dH * m; ++i)
			Wo[l][i] = 0.15f * rng.next_unit();
		for (unsigned int i = 0; i < m; ++i)
		{
			w_p[l][i] = 0.2f * rng.next_unit();
			b_p[l][i] = 0.03f * rng.next_unit();
			w_q[l][i] = 0.2f * rng.next_unit();
			b_q[l][i] = 0.03f * rng.next_unit();
			gamma[l][i] = 1.0f + 0.1f * rng.next_unit();
			beta[l][i]  = 0.03f * rng.next_unit();
		}
		bf16_round_all(&Wq[l][0], Wq[l].size());
		bf16_round_all(&Wk[l][0], Wk[l].size());
		bf16_round_all(&Wv[l][0], Wv[l].size());
		bf16_round_all(&Wo[l][0], Wo[l].size());
		bf16_round_all(&w_p[l][0], w_p[l].size());
		bf16_round_all(&b_p[l][0], b_p[l].size());
		bf16_round_all(&w_q[l][0], w_q[l].size());
		bf16_round_all(&b_q[l][0], b_q[l].size());
		bf16_round_all(&gamma[l][0], gamma[l].size());
		bf16_round_all(&beta[l][0], beta[l].size());
	}

	// One [r, Ntok] sketch per layer, shared across the T tokens of that layer.
	std::vector<std::vector<float> > S(L);
	for (unsigned int l = 0; l < L; ++l)
		generate_gaussian_sketch(S[l], r_sketch, Ntok, sketch_seed_base + l);

	// Stored sketches: [L, T, r]. Memory = L·T·r scalars — compare to
	// global-sketch [L, r] = L·r scalars. Trade memory for tighter bound.
	std::vector<std::vector<float> > z_stored(L);
	for (unsigned int l = 0; l < L; ++l)
		z_stored[l].resize(static_cast<size_t>(T) * r_sketch, 0.0f);

	std::vector<float> stats(L * T * 2u, 0.0f);
	std::vector<float> q(q0), p(p0);
	bf16_round_all(&q[0], q.size());
	bf16_round_all(&p[0], p.size());

	std::vector<float> tok_flat(Ntok);
	std::vector<float> y(T * m), u(T * m), q_tmp(T * m);

	// Forward loop.
	for (unsigned int l = 0; l < L; ++l)
	{
		// For each token: concat (q_t, p_t), sketch.
		for (unsigned int t = 0; t < T; ++t)
		{
			for (unsigned int i = 0; i < m; ++i)
			{
				tok_flat[i]       = q[t * m + i];
				tok_flat[m + i]   = p[t * m + i];
			}
			glades::chiron::sketch_project(&S[l][0], &tok_flat[0],
			                                r_sketch, Ntok,
			                                &z_stored[l][t * r_sketch]);
		}

		// Run block in BF16.
		chiron_attn_shear(&q[0], &Wq[l][0], &Wk[l][0], &Wv[l][0], &Wo[l][0],
		                   T, m, dH, causal, &y[0]);
		bf16_round_all(&y[0], y.size());
		for (unsigned int t = 0; t < T; ++t)
			glades::chiron::shear_add_to_p(&p[t * m], &y[t * m], m);
		bf16_round_all(&p[0], p.size());

		potential_f_apply_all(&q[0], &w_p[l][0], &b_p[l][0], &u[0], T, m);
		bf16_round_all(&u[0], u.size());
		for (unsigned int t = 0; t < T; ++t)
			glades::chiron::shear_add_to_p(&p[t * m], &u[t * m], m);
		bf16_round_all(&p[0], p.size());

		potential_f_apply_all(&p[0], &w_q[l][0], &b_q[l][0], &u[0], T, m);
		bf16_round_all(&u[0], u.size());
		for (unsigned int t = 0; t < T; ++t)
			glades::chiron::shear_add_to_q(&q[t * m], &u[t * m], m);
		bf16_round_all(&q[0], q.size());

		float* stats_l = &stats[l * T * 2u];
		glades::chiron::reln_forward(&q[0], &q_tmp[0], stats_l,
		                              &gamma[l][0], &beta[l][0], T, m, eps);
		bf16_round_all(&q_tmp[0], q_tmp.size());
		q.swap(q_tmp);
	}

	// Inverse loop with per-token correction.
	std::vector<float> z_fresh(r_sketch, 0.0f);
	std::vector<float> residual(r_sketch, 0.0f);
	(void)qSize;

	for (unsigned int ll = 0; ll < L; ++ll)
	{
		const unsigned int l = L - 1 - ll;

		// 4^-1 ReLN
		const float* stats_l = &stats[l * T * 2u];
		glades::chiron::reln_inverse(&q[0], &q_tmp[0], stats_l,
		                              &gamma[l][0], &beta[l][0], T, m);
		bf16_round_all(&q_tmp[0], q_tmp.size());
		q.swap(q_tmp);

		// 3^-1
		potential_f_apply_all(&p[0], &w_q[l][0], &b_q[l][0], &u[0], T, m);
		bf16_round_all(&u[0], u.size());
		for (unsigned int t = 0; t < T; ++t)
			glades::chiron::shear_sub_from_q(&q[t * m], &u[t * m], m);
		bf16_round_all(&q[0], q.size());

		// 2^-1
		potential_f_apply_all(&q[0], &w_p[l][0], &b_p[l][0], &u[0], T, m);
		bf16_round_all(&u[0], u.size());
		for (unsigned int t = 0; t < T; ++t)
			glades::chiron::shear_sub_from_p(&p[t * m], &u[t * m], m);
		bf16_round_all(&p[0], p.size());

		// 1^-1
		chiron_attn_shear(&q[0], &Wq[l][0], &Wk[l][0], &Wv[l][0], &Wo[l][0],
		                   T, m, dH, causal, &y[0]);
		bf16_round_all(&y[0], y.size());
		for (unsigned int t = 0; t < T; ++t)
			glades::chiron::shear_sub_from_p(&p[t * m], &y[t * m], m);
		bf16_round_all(&p[0], p.size());

		// Per-token sketch correction on (q_t, p_t).
		for (unsigned int t = 0; t < T; ++t)
		{
			for (unsigned int i = 0; i < m; ++i)
			{
				tok_flat[i]     = q[t * m + i];
				tok_flat[m + i] = p[t * m + i];
			}
			glades::chiron::sketch_project(&S[l][0], &tok_flat[0],
			                                r_sketch, Ntok, &z_fresh[0]);
			for (unsigned int k = 0; k < r_sketch; ++k)
				residual[k] = z_stored[l][t * r_sketch + k] - z_fresh[k];
			glades::chiron::sketch_lift_add(&S[l][0], &residual[0],
			                                 r_sketch, Ntok, &tok_flat[0]);
			for (unsigned int i = 0; i < m; ++i)
			{
				q[t * m + i] = tok_flat[i];
				p[t * m + i] = tok_flat[m + i];
			}
		}
		bf16_round_all(&q[0], q.size());
		bf16_round_all(&p[0], p.size());
	}

	std::vector<float> q0_bf16(q0), p0_bf16(p0);
	bf16_round_all(&q0_bf16[0], q0_bf16.size());
	bf16_round_all(&p0_bf16[0], p0_bf16.size());
	const float qe = max_abs_diff(q, q0_bf16);
	const float pe = max_abs_diff(p, p0_bf16);
	return (qe > pe) ? qe : pe;
}

void CHIRONPerTokenSketchBf16Test()
{
	const unsigned int T = 6;
	const unsigned int m = 16;
	const unsigned int dH = 4;
	const bool causal = true;
	const float eps = 1e-4f;
	const unsigned int L = 12;
	const unsigned int sketch_seed_base = 5150u;

	// Baselines for comparison.
	const float bf16_uncorrected =
	    run_multifullblock_roundtrip(L, T, m, dH, causal, eps, 24601u, true);

	// Global sketch at r=256 (same as Case 10).
	const float bf16_global_r256 =
	    run_multifullblock_roundtrip_sketch(L, T, m, dH, causal, eps,
	                                         24601u, 256u, 111111u);

	// Per-token sketch at r=128 (Ntok=2m=32, so N/r ≈ 0.25 — tight).
	const float bf16_pertok_r128 =
	    run_multifullblock_roundtrip_per_token_sketch(
	        L, T, m, dH, causal, eps, 24601u, 128u, sketch_seed_base);

	// Per-token sketch at r=256 — even tighter.
	const float bf16_pertok_r256 =
	    run_multifullblock_roundtrip_per_token_sketch(
	        L, T, m, dH, causal, eps, 24601u, 256u, sketch_seed_base);

	std::printf("  CHIRON per-token sketch @ L=12: uncorrected=%.3e  "
	            "global_r256=%.3e  pertok_r128=%.3e  pertok_r256=%.3e\n",
	            bf16_uncorrected, bf16_global_r256,
	            bf16_pertok_r128, bf16_pertok_r256);

	char msg[256];
	// Per-token sketch at r=256 should be better than global sketch at r=256
	// because effective N is 2m=32 vs. global N=2·T·m=192 — 6× smaller.
	std::snprintf(msg, sizeof(msg),
	              "per-token sketch should beat global at same r: "
	              "global_r256=%.3e pertok_r256=%.3e",
	              bf16_global_r256, bf16_pertok_r256);
	ASSERT(msg, bf16_pertok_r256 <= bf16_global_r256);

	// Per-token sketch should beat uncorrected.
	std::snprintf(msg, sizeof(msg),
	              "per-token sketch should beat uncorrected: "
	              "uncorrected=%.3e pertok_r128=%.3e",
	              bf16_uncorrected, bf16_pertok_r128);
	ASSERT(msg, bf16_pertok_r128 < bf16_uncorrected);
}

// ---------------------------------------------------------------------------
// Case 12: GPU parity — compare gpu_chiron primitives against CPU reference.
// Runs shear_add/sub, reln_forward/inverse, and sketch_project/lift_add on
// both paths and asserts element-wise equality within tolerance.
// Skipped (with a printed notice) when CUDA is not available at runtime.
// ---------------------------------------------------------------------------
void CHIRONGpuParityTest()
{
#ifdef GLADES_HAVE_CUDA
	if (!glades::gpu::initDevice())
	{
		std::printf("  [CHIRON GPU parity] no CUDA device — skipped\n");
		return;
	}

	const unsigned int T = 8;
	const unsigned int m = 32;
	const unsigned int Ntok = 2u * m;
	const unsigned int r_sketch = 64u;
	const float eps = 1e-4f;

	LCG rng(54321u);

	// --- Data ---
	std::vector<float> p(T * m), u(T * m);
	for (unsigned int i = 0; i < p.size(); ++i) p[i] = rng.next_unit();
	for (unsigned int i = 0; i < u.size(); ++i) u[i] = 0.1f * rng.next_unit();

	std::vector<float> q_in(T * m), gamma(m), beta(m);
	for (unsigned int i = 0; i < q_in.size(); ++i) q_in[i] = rng.next_unit();
	for (unsigned int i = 0; i < m; ++i)
	{
		gamma[i] = 1.0f + 0.1f * rng.next_unit();
		beta[i]  = 0.05f * rng.next_unit();
	}

	std::vector<float> X(T * Ntok);
	for (unsigned int i = 0; i < X.size(); ++i) X[i] = rng.next_unit();
	std::vector<float> S(r_sketch * Ntok);
	LCG sketch_rng(99999u);
	for (unsigned int i = 0; i < S.size(); ++i) S[i] = 0.3f * sketch_rng.next_unit();

	// --- CPU reference ---
	std::vector<float> p_cpu(p);
	for (unsigned int t = 0; t < T; ++t)
		glades::chiron::shear_add_to_p(&p_cpu[t * m], &u[t * m], m);

	std::vector<float> p_cpu_sub(p);
	for (unsigned int t = 0; t < T; ++t)
		glades::chiron::shear_sub_from_p(&p_cpu_sub[t * m], &u[t * m], m);

	std::vector<float> q_out_cpu(T * m), stats_cpu(T * 2u);
	glades::chiron::reln_forward(&q_in[0], &q_out_cpu[0], &stats_cpu[0],
	                              &gamma[0], &beta[0], T, m, eps);

	std::vector<float> q_inv_cpu(T * m);
	glades::chiron::reln_inverse(&q_out_cpu[0], &q_inv_cpu[0], &stats_cpu[0],
	                              &gamma[0], &beta[0], T, m);

	std::vector<float> Z_cpu(T * r_sketch, 0.0f);
	for (unsigned int t = 0; t < T; ++t)
		glades::chiron::sketch_project(&S[0], &X[t * Ntok],
		                                r_sketch, Ntok, &Z_cpu[t * r_sketch]);

	std::vector<float> X_lifted_cpu(X);
	for (unsigned int t = 0; t < T; ++t)
		glades::chiron::sketch_lift_add(&S[0], &Z_cpu[t * r_sketch],
		                                 r_sketch, Ntok, &X_lifted_cpu[t * Ntok]);

	// --- GPU path ---
	glades::gpu::GpuBuffer<float> d_p, d_u;
	d_p.allocate(p.size()); d_u.allocate(u.size());
	d_p.upload(&p[0], p.size());
	d_u.upload(&u[0], u.size());

	ASSERT("chiron_shear_add", glades::gpu::chiron_shear_add(
	        d_p.data(), d_u.data(), static_cast<int>(p.size())));
	std::vector<float> p_gpu(p.size());
	d_p.download(&p_gpu[0], p.size());

	// Reset d_p to original, test sub variant.
	d_p.upload(&p[0], p.size());
	ASSERT("chiron_shear_sub", glades::gpu::chiron_shear_sub(
	        d_p.data(), d_u.data(), static_cast<int>(p.size())));
	std::vector<float> p_gpu_sub(p.size());
	d_p.download(&p_gpu_sub[0], p.size());

	// ReLN forward.
	glades::gpu::GpuBuffer<float> d_qin, d_qout, d_stats, d_gamma, d_beta;
	d_qin.allocate(q_in.size()); d_qout.allocate(q_in.size());
	d_stats.allocate(T * 2u);
	d_gamma.allocate(m); d_beta.allocate(m);
	d_qin.upload(&q_in[0], q_in.size());
	d_gamma.upload(&gamma[0], m);
	d_beta.upload(&beta[0], m);

	ASSERT("chiron_reln_forward", glades::gpu::chiron_reln_forward(
	        d_qin.data(), d_qout.data(), d_stats.data(),
	        d_gamma.data(), d_beta.data(),
	        static_cast<int>(T), static_cast<int>(m), eps));
	std::vector<float> q_out_gpu(q_in.size()), stats_gpu(T * 2u);
	d_qout.download(&q_out_gpu[0], q_in.size());
	d_stats.download(&stats_gpu[0], T * 2u);

	// ReLN inverse.
	glades::gpu::GpuBuffer<float> d_qinv;
	d_qinv.allocate(q_in.size());
	ASSERT("chiron_reln_inverse", glades::gpu::chiron_reln_inverse(
	        d_qout.data(), d_qinv.data(), d_stats.data(),
	        d_gamma.data(), d_beta.data(),
	        static_cast<int>(T), static_cast<int>(m)));
	std::vector<float> q_inv_gpu(q_in.size());
	d_qinv.download(&q_inv_gpu[0], q_in.size());

	// Sketch project.
	glades::gpu::GpuBuffer<float> d_X, d_S, d_Z;
	d_X.allocate(X.size()); d_S.allocate(S.size()); d_Z.allocate(T * r_sketch);
	d_X.upload(&X[0], X.size());
	d_S.upload(&S[0], S.size());
	ASSERT("chiron_sketch_project", glades::gpu::chiron_sketch_project(
	        d_X.data(), d_S.data(),
	        static_cast<int>(T), static_cast<int>(Ntok), static_cast<int>(r_sketch),
	        d_Z.data()));
	std::vector<float> Z_gpu(T * r_sketch);
	d_Z.download(&Z_gpu[0], T * r_sketch);

	// Sketch lift-add.
	d_X.upload(&X[0], X.size());  // reset to raw X
	ASSERT("chiron_sketch_lift_add", glades::gpu::chiron_sketch_lift_add(
	        d_X.data(), d_Z.data(), d_S.data(),
	        static_cast<int>(T), static_cast<int>(Ntok), static_cast<int>(r_sketch)));
	std::vector<float> X_lifted_gpu(X.size());
	d_X.download(&X_lifted_gpu[0], X.size());

	// --- Compare ---
	char msg[256];
	const float tol_elem = 5e-5f;   // element-wise tolerance for simple ops
	const float tol_gemm = 5e-4f;   // slightly looser for GEMM-routed ops

	float err_add = max_abs_diff(p_cpu, p_gpu);
	std::snprintf(msg, sizeof(msg),
	              "GPU shear_add parity: cpu vs gpu max_err=%.3e (tol %.1e)", err_add, tol_elem);
	ASSERT(msg, err_add < tol_elem);

	float err_sub = max_abs_diff(p_cpu_sub, p_gpu_sub);
	std::snprintf(msg, sizeof(msg),
	              "GPU shear_sub parity: max_err=%.3e (tol %.1e)", err_sub, tol_elem);
	ASSERT(msg, err_sub < tol_elem);

	float err_reln_fwd = max_abs_diff(q_out_cpu, q_out_gpu);
	std::snprintf(msg, sizeof(msg),
	              "GPU reln_forward q_out parity: max_err=%.3e (tol %.1e)", err_reln_fwd, tol_elem);
	ASSERT(msg, err_reln_fwd < tol_elem);

	float err_reln_stats = max_abs_diff(stats_cpu, stats_gpu);
	std::snprintf(msg, sizeof(msg),
	              "GPU reln_forward stats parity: max_err=%.3e (tol %.1e)", err_reln_stats, tol_elem);
	ASSERT(msg, err_reln_stats < tol_elem);

	float err_reln_inv = max_abs_diff(q_inv_cpu, q_inv_gpu);
	std::snprintf(msg, sizeof(msg),
	              "GPU reln_inverse parity: max_err=%.3e (tol %.1e)", err_reln_inv, tol_elem);
	ASSERT(msg, err_reln_inv < tol_elem);

	float err_sketch_proj = max_abs_diff(Z_cpu, Z_gpu);
	std::snprintf(msg, sizeof(msg),
	              "GPU sketch_project parity: max_err=%.3e (tol %.1e)",
	              err_sketch_proj, tol_gemm);
	ASSERT(msg, err_sketch_proj < tol_gemm);

	float err_sketch_lift = max_abs_diff(X_lifted_cpu, X_lifted_gpu);
	std::snprintf(msg, sizeof(msg),
	              "GPU sketch_lift_add parity: max_err=%.3e (tol %.1e)",
	              err_sketch_lift, tol_gemm);
	ASSERT(msg, err_sketch_lift < tol_gemm);

	std::printf("  CHIRON GPU parity @ T=%u m=%u r=%u:\n"
	            "    shear_add      max_err=%.3e\n"
	            "    shear_sub      max_err=%.3e\n"
	            "    reln_fwd  (q)  max_err=%.3e\n"
	            "    reln_fwd stats max_err=%.3e\n"
	            "    reln_inv       max_err=%.3e\n"
	            "    sketch_project max_err=%.3e\n"
	            "    sketch_lift    max_err=%.3e\n",
	            T, m, r_sketch,
	            err_add, err_sub, err_reln_fwd, err_reln_stats, err_reln_inv,
	            err_sketch_proj, err_sketch_lift);
#else
	std::printf("  [CHIRON GPU parity] GLADES_HAVE_CUDA not defined — skipped\n");
#endif
}

// ---------------------------------------------------------------------------
// Case 13: batched CPU sketch ops parity with scalar reference.
// Verifies the Phase 3.5 optimized batched kernels (sketch_project_batched /
// sketch_lift_add_batched) produce the same result (up to FP accumulation
// order) as the scalar per-token reference.
// ---------------------------------------------------------------------------
void CHIRONBatchedSketchParityTest()
{
	const unsigned int T = 16;
	const unsigned int N = 64;
	const unsigned int r = 32;

	LCG rng(11111u);
	std::vector<float> X(T * N), S(r * N);
	for (unsigned int i = 0; i < X.size(); ++i) X[i] = rng.next_unit();
	for (unsigned int i = 0; i < S.size(); ++i) S[i] = 0.3f * rng.next_unit();

	// Scalar reference.
	std::vector<float> Z_ref(T * r, 0.0f);
	for (unsigned int t = 0; t < T; ++t)
		glades::chiron::sketch_project(&S[0], &X[t * N], r, N, &Z_ref[t * r]);

	// Batched.
	std::vector<float> Z_bat(T * r, 0.0f);
	glades::chiron::sketch_project_batched(&S[0], &X[0], T, N, r, &Z_bat[0]);

	// Must match within FP accumulation-order slop (same set of FMAs, same order).
	const float proj_err = max_abs_diff(Z_ref, Z_bat);
	char msg[256];
	std::snprintf(msg, sizeof(msg),
	              "batched sketch_project parity: max_err=%.3e (want < 1e-5)",
	              proj_err);
	ASSERT(msg, proj_err < 1e-5f);

	// Lift parity.
	std::vector<float> X_lift_ref(X);
	for (unsigned int t = 0; t < T; ++t)
		glades::chiron::sketch_lift_add(&S[0], &Z_ref[t * r], r, N, &X_lift_ref[t * N]);

	std::vector<float> X_lift_bat(X);
	glades::chiron::sketch_lift_add_batched(&S[0], &Z_ref[0], T, N, r, &X_lift_bat[0]);

	const float lift_err = max_abs_diff(X_lift_ref, X_lift_bat);
	std::snprintf(msg, sizeof(msg),
	              "batched sketch_lift_add parity: max_err=%.3e (want < 1e-5)",
	              lift_err);
	ASSERT(msg, lift_err < 1e-5f);
}

void CHIRONUnitTest()
{
	std::printf("\n=== CHIRON (reversible-flow transformer) unit tests ===\n");
	CHIRONShearReversibilityTest();
	CHIRONReLNRoundtripTest();
	CHIRONBlockRoundtripTest();
	CHIRONMultiBlockRoundtripTest();
	CHIRONAttentionShearReversibilityTest();
	CHIRONFullBlockRoundtripTest();
	CHIRONMultiFullBlockRoundtripTest();
	CHIRONBf16DriftTest();
	CHIRONSketchProjectLiftTest();
	CHIRONSketchCorrectedBf16Test();
	CHIRONPerTokenSketchBf16Test();
	CHIRONGpuParityTest();
	CHIRONBatchedSketchParityTest();
	std::printf("=== CHIRON tests done ===\n\n");
}

// ===========================================================================
// Performance benchmark — Phase 3.5 baseline.
//
// Measures CPU and GPU wall-clock for the CHIRON primitives at realistic
// sizes. Provides the numbers that Phase 3.5 optimization iterations will
// try to improve.
// ===========================================================================

#include <sys/time.h>

#ifdef GLADES_HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace {

static double wall_ms_chiron()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return static_cast<double>(tv.tv_sec) * 1000.0
	     + static_cast<double>(tv.tv_usec) / 1000.0;
}

// Warm a CPU value so the compiler doesn't optimize the loop body away.
static float g_chiron_sink = 0.0f;

// Query VRAM (free/total) in bytes.  Returns 0 if CUDA unavailable.
static void get_vram_info(size_t& free_bytes, size_t& total_bytes)
{
	free_bytes = 0;
	total_bytes = 0;
#ifdef GLADES_HAVE_CUDA
	cudaMemGetInfo(&free_bytes, &total_bytes);
#endif
}

} // namespace

void CHIRONBenchmark()
{
	std::printf("\n=== CHIRON benchmark (Phase 3.5 baseline) ===\n");

	// Benchmark sizes — intermediate between toy and production.
	struct Size { unsigned int T, m, r; const char* label; };
	const Size sizes[] = {
		{ 256u,   256u,  128u,  "small  (T=256, m=256, r=128)" },
		{ 1024u,  1024u, 512u,  "medium (T=1024, m=1024, r=512)" },
		{ 2048u,  2048u, 1024u, "large  (T=2048, m=2048, r=1024)" }
	};
	const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

	const int warmup_iters = 3;
	const int bench_iters  = 10;
	const float eps = 1e-4f;

	// Budget cap so we don't spend minutes in CPU sketch at large sizes.
	// Anything heavier than this gets one iteration only (or skipped).
	const size_t kCpuSketchFlopBudget = 500ULL * 1000ULL * 1000ULL; // 500M FLOPs

	for (int sz = 0; sz < num_sizes; ++sz)
	{
		const unsigned int T = sizes[sz].T;
		const unsigned int m = sizes[sz].m;
		const unsigned int Ntok = 2u * m;
		const unsigned int r = sizes[sz].r;

		const size_t pSize = static_cast<size_t>(T) * m;
		const size_t xSize = static_cast<size_t>(T) * Ntok;
		const size_t sSize = static_cast<size_t>(r) * Ntok;
		const size_t zSize = static_cast<size_t>(T) * r;

		std::printf("\n--- %s ---\n", sizes[sz].label);
		std::printf("  Buffers: p [%zu], X [%zu], S [%zu], Z [%zu]\n",
		            pSize, xSize, sSize, zSize);
		const double activation_mb = (2.0 * pSize * 4.0) / (1024.0 * 1024.0);
		std::printf("  Per-layer activation footprint (q+p FP32): %.1f MB\n",
		            activation_mb);

		LCG rng(123u + sz);
		std::vector<float> p(pSize), u(pSize);
		std::vector<float> q_in(pSize), gamma(m), beta(m);
		std::vector<float> X(xSize), S(sSize);
		for (size_t i = 0; i < pSize; ++i) { p[i] = rng.next_unit(); u[i] = 0.05f * rng.next_unit(); }
		for (size_t i = 0; i < pSize; ++i) q_in[i] = rng.next_unit();
		for (unsigned int i = 0; i < m; ++i) { gamma[i] = 1.0f + 0.1f * rng.next_unit(); beta[i] = 0.05f * rng.next_unit(); }
		for (size_t i = 0; i < xSize; ++i) X[i] = rng.next_unit();
		for (size_t i = 0; i < sSize; ++i) S[i] = 0.3f * rng.next_unit();
		std::vector<float> q_out(pSize), stats(T * 2u);
		std::vector<float> Z(zSize), X_lift(X);

		// ---------------- CPU ----------------
		std::printf("  CPU:\n");

		// shear_add — N element-wise adds.
		for (int it = 0; it < warmup_iters; ++it)
		{
			for (unsigned int t = 0; t < T; ++t)
				glades::chiron::shear_add_to_p(&p[t * m], &u[t * m], m);
		}
		double t0 = wall_ms_chiron();
		for (int it = 0; it < bench_iters; ++it)
		{
			for (unsigned int t = 0; t < T; ++t)
				glades::chiron::shear_add_to_p(&p[t * m], &u[t * m], m);
		}
		double tcpu_shear = (wall_ms_chiron() - t0) / bench_iters;
		// Light sink to keep the compiler honest.
		g_chiron_sink += p[0];
		const double shear_bw = (2.0 * pSize * 4.0) / (tcpu_shear * 1e-3) / (1024.0 * 1024.0 * 1024.0);
		std::printf("    shear_add/iter   %.3f ms   (~%.1f GB/s)\n", tcpu_shear, shear_bw);

		// reln_forward
		for (int it = 0; it < warmup_iters; ++it)
			glades::chiron::reln_forward(&q_in[0], &q_out[0], &stats[0],
			                              &gamma[0], &beta[0], T, m, eps);
		t0 = wall_ms_chiron();
		for (int it = 0; it < bench_iters; ++it)
			glades::chiron::reln_forward(&q_in[0], &q_out[0], &stats[0],
			                              &gamma[0], &beta[0], T, m, eps);
		double tcpu_reln = (wall_ms_chiron() - t0) / bench_iters;
		g_chiron_sink += q_out[0];
		std::printf("    reln_forward/iter %.3f ms\n", tcpu_reln);

		// sketch ops — adaptive iteration count based on estimated FLOPs.
		const size_t flops_per_iter = 2ULL * T * r * Ntok;
		const bool run_scalar = flops_per_iter < kCpuSketchFlopBudget;
		const int cpu_iters = static_cast<int>(
		    (flops_per_iter < kCpuSketchFlopBudget) ? bench_iters :
		    (flops_per_iter < kCpuSketchFlopBudget * 10u) ? 3 : 1);

		if (run_scalar)
		{
			// Scalar baseline.
			for (int it = 0; it < warmup_iters; ++it)
			{
				for (unsigned int t = 0; t < T; ++t)
					glades::chiron::sketch_project(&S[0], &X[t * Ntok],
					                                r, Ntok, &Z[t * r]);
			}
			t0 = wall_ms_chiron();
			for (int it = 0; it < cpu_iters; ++it)
			{
				for (unsigned int t = 0; t < T; ++t)
					glades::chiron::sketch_project(&S[0], &X[t * Ntok],
					                                r, Ntok, &Z[t * r]);
			}
			double tcpu_proj = (wall_ms_chiron() - t0) / cpu_iters;
			g_chiron_sink += Z[0];
			const double gflops = (2.0 * T * r * Ntok) / (tcpu_proj * 1e-3) / 1e9;
			std::printf("    sketch_project/iter        %.3f ms  (~%.1f GFLOP/s)\n",
			            tcpu_proj, gflops);
		}
		else
		{
			std::printf("    sketch_project scalar     SKIPPED (%llu M FLOPs > budget)\n",
			            (unsigned long long)(flops_per_iter / 1000000));
		}

		// Batched always — it's what we'd use in production.
		for (int it = 0; it < warmup_iters; ++it)
			glades::chiron::sketch_project_batched(&S[0], &X[0], T, Ntok, r, &Z[0]);
		t0 = wall_ms_chiron();
		for (int it = 0; it < cpu_iters; ++it)
			glades::chiron::sketch_project_batched(&S[0], &X[0], T, Ntok, r, &Z[0]);
		double tcpu_proj_bat = (wall_ms_chiron() - t0) / cpu_iters;
		g_chiron_sink += Z[0];
		const double gflops_bat = (2.0 * T * r * Ntok) / (tcpu_proj_bat * 1e-3) / 1e9;
		std::printf("    sketch_project_batched/iter %.3f ms  (~%.1f GFLOP/s)\n",
		            tcpu_proj_bat, gflops_bat);

		if (run_scalar)
		{
			// Scalar lift.
			for (unsigned int i = 0; i < xSize; ++i) X_lift[i] = X[i];
			for (int it = 0; it < warmup_iters; ++it)
			{
				for (unsigned int t = 0; t < T; ++t)
					glades::chiron::sketch_lift_add(&S[0], &Z[t * r],
					                                 r, Ntok, &X_lift[t * Ntok]);
			}
			for (unsigned int i = 0; i < xSize; ++i) X_lift[i] = X[i];
			t0 = wall_ms_chiron();
			for (int it = 0; it < cpu_iters; ++it)
			{
				for (unsigned int t = 0; t < T; ++t)
					glades::chiron::sketch_lift_add(&S[0], &Z[t * r],
					                                 r, Ntok, &X_lift[t * Ntok]);
			}
			double tcpu_lift = (wall_ms_chiron() - t0) / cpu_iters;
			const double gflops_lift = (2.0 * T * r * Ntok) / (tcpu_lift * 1e-3) / 1e9;
			std::printf("    sketch_lift/iter            %.3f ms  (~%.1f GFLOP/s)\n",
			            tcpu_lift, gflops_lift);
		}

		for (unsigned int i = 0; i < xSize; ++i) X_lift[i] = X[i];
		for (int it = 0; it < warmup_iters; ++it)
			glades::chiron::sketch_lift_add_batched(&S[0], &Z[0], T, Ntok, r, &X_lift[0]);
		for (unsigned int i = 0; i < xSize; ++i) X_lift[i] = X[i];
		t0 = wall_ms_chiron();
		for (int it = 0; it < cpu_iters; ++it)
			glades::chiron::sketch_lift_add_batched(&S[0], &Z[0], T, Ntok, r, &X_lift[0]);
		double tcpu_lift_bat = (wall_ms_chiron() - t0) / cpu_iters;
		const double gflops_lift_bat = (2.0 * T * r * Ntok) / (tcpu_lift_bat * 1e-3) / 1e9;
		std::printf("    sketch_lift_batched/iter    %.3f ms  (~%.1f GFLOP/s)\n",
		            tcpu_lift_bat, gflops_lift_bat);

#ifdef GLADES_HAVE_CUDA
		// ---------------- GPU ----------------
		if (glades::gpu::initDevice())
		{
			size_t vram_free_before = 0, vram_total = 0;
			get_vram_info(vram_free_before, vram_total);
			std::printf("  GPU:  (VRAM free before alloc: %.1f / %.1f MB)\n",
			            vram_free_before / (1024.0 * 1024.0),
			            vram_total / (1024.0 * 1024.0));

			glades::gpu::GpuBuffer<float> d_p, d_u;
			glades::gpu::GpuBuffer<float> d_qin, d_qout, d_stats, d_gamma, d_beta;
			glades::gpu::GpuBuffer<float> d_X, d_S, d_Z;
			d_p.allocate(pSize); d_u.allocate(pSize);
			d_qin.allocate(pSize); d_qout.allocate(pSize);
			d_stats.allocate(T * 2u);
			d_gamma.allocate(m); d_beta.allocate(m);
			d_X.allocate(xSize); d_S.allocate(sSize); d_Z.allocate(zSize);
			d_p.upload(&p[0], pSize);
			d_u.upload(&u[0], pSize);
			d_qin.upload(&q_in[0], pSize);
			d_gamma.upload(&gamma[0], m);
			d_beta.upload(&beta[0], m);
			d_X.upload(&X[0], xSize);
			d_S.upload(&S[0], sSize);

			// Warmup + sync.
			for (int it = 0; it < warmup_iters; ++it)
				glades::gpu::chiron_shear_add(d_p.data(), d_u.data(),
				                               static_cast<int>(pSize));
			glades::gpu::synchronizeCheck("bench chiron warmup");

			// shear_add
			t0 = wall_ms_chiron();
			for (int it = 0; it < bench_iters; ++it)
				glades::gpu::chiron_shear_add(d_p.data(), d_u.data(),
				                               static_cast<int>(pSize));
			glades::gpu::synchronizeCheck("bench shear_add");
			double tgpu_shear = (wall_ms_chiron() - t0) / bench_iters;
			const double gpu_shear_bw = (2.0 * pSize * 4.0) / (tgpu_shear * 1e-3)
			                             / (1024.0 * 1024.0 * 1024.0);
			std::printf("    shear_add/iter    %.4f ms   (~%.1f GB/s)\n",
			            tgpu_shear, gpu_shear_bw);

			// reln_forward
			for (int it = 0; it < warmup_iters; ++it)
				glades::gpu::chiron_reln_forward(d_qin.data(), d_qout.data(),
				                                  d_stats.data(), d_gamma.data(),
				                                  d_beta.data(),
				                                  static_cast<int>(T), static_cast<int>(m), eps);
			glades::gpu::synchronizeCheck("bench chiron reln warmup");
			t0 = wall_ms_chiron();
			for (int it = 0; it < bench_iters; ++it)
				glades::gpu::chiron_reln_forward(d_qin.data(), d_qout.data(),
				                                  d_stats.data(), d_gamma.data(),
				                                  d_beta.data(),
				                                  static_cast<int>(T), static_cast<int>(m), eps);
			glades::gpu::synchronizeCheck("bench reln_fwd");
			double tgpu_reln = (wall_ms_chiron() - t0) / bench_iters;
			std::printf("    reln_forward/iter %.4f ms\n", tgpu_reln);

			// sketch_project
			for (int it = 0; it < warmup_iters; ++it)
				glades::gpu::chiron_sketch_project(d_X.data(), d_S.data(),
				                                    static_cast<int>(T), static_cast<int>(Ntok),
				                                    static_cast<int>(r), d_Z.data());
			glades::gpu::synchronizeCheck("bench sketch warmup");
			t0 = wall_ms_chiron();
			for (int it = 0; it < bench_iters; ++it)
				glades::gpu::chiron_sketch_project(d_X.data(), d_S.data(),
				                                    static_cast<int>(T), static_cast<int>(Ntok),
				                                    static_cast<int>(r), d_Z.data());
			glades::gpu::synchronizeCheck("bench sketch_project");
			double tgpu_proj = (wall_ms_chiron() - t0) / bench_iters;
			const double gpu_gflops = (2.0 * T * r * Ntok) / (tgpu_proj * 1e-3) / 1e9;
			std::printf("    sketch_project/iter %.4f ms  (~%.1f GFLOP/s)\n",
			            tgpu_proj, gpu_gflops);

			// sketch_lift_add
			for (int it = 0; it < warmup_iters; ++it)
				glades::gpu::chiron_sketch_lift_add(d_X.data(), d_Z.data(), d_S.data(),
				                                     static_cast<int>(T), static_cast<int>(Ntok),
				                                     static_cast<int>(r));
			glades::gpu::synchronizeCheck("bench lift warmup");
			t0 = wall_ms_chiron();
			for (int it = 0; it < bench_iters; ++it)
				glades::gpu::chiron_sketch_lift_add(d_X.data(), d_Z.data(), d_S.data(),
				                                     static_cast<int>(T), static_cast<int>(Ntok),
				                                     static_cast<int>(r));
			glades::gpu::synchronizeCheck("bench sketch_lift_add");
			double tgpu_lift = (wall_ms_chiron() - t0) / bench_iters;
			const double gpu_lift_gflops = (2.0 * T * r * Ntok) / (tgpu_lift * 1e-3) / 1e9;
			std::printf("    sketch_lift/iter  %.4f ms  (~%.1f GFLOP/s)\n",
			            tgpu_lift, gpu_lift_gflops);

			size_t vram_free_after = 0;
			get_vram_info(vram_free_after, vram_total);
			const double vram_used_mb = static_cast<double>(vram_free_before - vram_free_after)
			                             / (1024.0 * 1024.0);
			std::printf("    actual VRAM allocated (bench): %.1f MB\n", vram_used_mb);
		}
		else
		{
			std::printf("  GPU: CUDA device unavailable, GPU timings skipped.\n");
		}
#else
		std::printf("  GPU: GLADES_HAVE_CUDA not defined, GPU timings skipped.\n");
#endif
	}

	std::printf("\nSink prevention: %g\n", static_cast<double>(g_chiron_sink));
	std::printf("=== CHIRON benchmark done ===\n\n");
}
