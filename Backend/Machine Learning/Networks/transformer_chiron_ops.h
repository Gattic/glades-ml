// CHIRON reversible-flow building blocks.
//
// CHIRON ("Canonical Hamiltonian Involutive Reversible Operator Network")
// expresses a transformer block as a composition of symplectic diffeomorphisms
// on a paired hidden state (q, p) ∈ R^{T × m} × R^{T × m}, giving an explicit
// block inverse that reconstructs activations from outputs without storing
// them. See research/CHIRON_framework.md for the full derivation.
//
// This header is the CPU reference implementation. It is header-only and
// C++98-compatible, matching the style of transformer_ops.h.
//
// Building blocks:
//   - Shear^p (momentum kick):   (q, p) → (q, p + u)   for u a pure function of q
//   - Shear^q (position drift):  (q, p) → (q + u, p)   for u a pure function of p
//   - ReLN forward/inverse:      reversible LayerNorm on q with (μ, σ)
//                                written to a caller-provided external stats
//                                buffer (primary API) OR into reserved coords
//                                of p (future optimization).
//
// Invariants (block inverses rely on these):
//   - All shear updates must be a pure function of the OPPOSITE branch only.
//   - For the external-stats ReLN, the caller sizes an [L, T, 2] FP32 buffer
//     and passes a distinct slice per block. Memory cost is 2 FP32 per token
//     per block — negligible compared to the O(L·T·d) activations CHIRON
//     eliminates.
//
// Convention:
//   - Tensors are row-major, flattened as [T, m].
//   - All ops work per-token; the token-axis loop is explicit in the caller.

#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace glades {
namespace chiron {

// Shear^p: p += u. u must be a pure function of q (not of p).
// Forward: p[i] += u[i] for i in [0, m).
inline void shear_add_to_p(float* p, const float* u, unsigned int m)
{
	for (unsigned int i = 0; i < m; ++i)
		p[i] += u[i];
}

// Shear^p inverse: p -= u. Caller must pass the SAME u as forward (re-computed
// from q_new which equals q_old in the shear, so recomputation is exact).
inline void shear_sub_from_p(float* p, const float* u, unsigned int m)
{
	for (unsigned int i = 0; i < m; ++i)
		p[i] -= u[i];
}

// Shear^q: q += u. u must be a pure function of p (not of q).
inline void shear_add_to_q(float* q, const float* u, unsigned int m)
{
	for (unsigned int i = 0; i < m; ++i)
		q[i] += u[i];
}

inline void shear_sub_from_q(float* q, const float* u, unsigned int m)
{
	for (unsigned int i = 0; i < m; ++i)
		q[i] -= u[i];
}

// Zero reserved coordinates on a token-row vector. Used to enforce the
// invariant that nonlinear maps do not write to reserved coords.
inline void zero_reserved_coords_row(float* v, unsigned int r0, unsigned int r1)
{
	v[r0] = 0.0f;
	v[r1] = 0.0f;
}

// Zero reserved coordinates across all T tokens of a row-major [T, m] tensor.
inline void zero_reserved_coords_all_tokens(float* v, unsigned int T, unsigned int m,
                                            unsigned int r0, unsigned int r1)
{
	for (unsigned int t = 0; t < T; ++t)
	{
		v[t * m + r0] = 0.0f;
		v[t * m + r1] = 0.0f;
	}
}

// Reversible LayerNorm (ReLN), per-token forward — "external stats" variant.
//
// This is the primary ReLN API used by the prototype: stats (μ, σ) are
// written to a caller-provided `stats_out[2]` buffer rather than packed into
// reserved coordinates of p. This decouples the ReLN round-trip from the
// per-block channel-permutation bookkeeping and makes composition trivially
// block-local: the caller sizes a `[L, T, 2]` stats buffer, block ℓ writes
// to its own slice, and the inverse reads from that slice.
//
// Given:
//   q_in        row-major [m], input q coordinates of one token
//   gamma, beta row-major [m], LN affine parameters
//   m           hidden width per branch (= dModel / 2 in CHIRON)
//   eps         numerical epsilon added to variance
// Writes:
//   q_out       row-major [m], normalized + affine-transformed q
//   stats_out   [2], { μ, σ } for this token
//
// Math:
//   μ   = (1/m) Σ q_in[i]
//   σ²  = (1/m) Σ (q_in[i] − μ)² + eps
//   σ   = √σ²
//   q_out[i] = gamma[i] * (q_in[i] − μ) / σ + beta[i]
//
// Memory cost per block: 2 FP32 per token. For L=96, T=4096 that is
// ~3 MB total, which is negligible compared to the O(L·T·d) activations
// that CHIRON eliminates.
inline void reln_forward_row(const float* q_in, float* q_out, float* stats_out,
                             const float* gamma, const float* beta,
                             unsigned int m, float eps)
{
	// Mean.
	double sum = 0.0;
	for (unsigned int i = 0; i < m; ++i)
		sum += q_in[i];
	const float mu = static_cast<float>(sum / static_cast<double>(m));

	// Variance.
	double var_sum = 0.0;
	for (unsigned int i = 0; i < m; ++i)
	{
		const float d = q_in[i] - mu;
		var_sum += static_cast<double>(d) * static_cast<double>(d);
	}
	const float var = static_cast<float>(var_sum / static_cast<double>(m)) + eps;
	const float sigma = sqrtf(var);
	const float inv_sigma = 1.0f / sigma;

	// Normalize + affine.
	for (unsigned int i = 0; i < m; ++i)
		q_out[i] = gamma[i] * (q_in[i] - mu) * inv_sigma + beta[i];

	// Record (μ, σ) into the caller-owned stats buffer for this token.
	stats_out[0] = mu;
	stats_out[1] = sigma;
}

// Reversible LayerNorm inverse, per-token — "external stats" variant.
//
// Given the outputs of reln_forward_row:
//   q_out       [m], post-ReLN q
//   stats_in    [2], { μ, σ } recorded at forward time
//   gamma, beta [m], same as forward
// Writes:
//   q_in        [m], recovered original q (bit-exact in FP32)
//
// Math:
//   μ          = stats_in[0]
//   σ          = stats_in[1]
//   q_in[i]    = σ · (q_out[i] − beta[i]) / gamma[i]  +  μ
inline void reln_inverse_row(const float* q_out, float* q_in, const float* stats_in,
                             const float* gamma, const float* beta, unsigned int m)
{
	const float mu = stats_in[0];
	const float sigma = stats_in[1];

	for (unsigned int i = 0; i < m; ++i)
	{
		// gamma[i] may be tuned small; caller should avoid gamma[i] ≈ 0.
		q_in[i] = sigma * (q_out[i] - beta[i]) / gamma[i] + mu;
	}
}

// Whole-tensor wrappers for a [T, m] q tensor and a [T, 2] stats tensor.
// Convenience for the block-level forward/inverse.
inline void reln_forward(const float* q_in, float* q_out, float* stats_out,
                         const float* gamma, const float* beta,
                         unsigned int T, unsigned int m, float eps)
{
	for (unsigned int t = 0; t < T; ++t)
	{
		reln_forward_row(q_in + t * m, q_out + t * m, stats_out + t * 2u,
		                 gamma, beta, m, eps);
	}
}

inline void reln_inverse(const float* q_out, float* q_in, const float* stats_in,
                         const float* gamma, const float* beta,
                         unsigned int T, unsigned int m)
{
	for (unsigned int t = 0; t < T; ++t)
	{
		reln_inverse_row(q_out + t * m, q_in + t * m, stats_in + t * 2u,
		                 gamma, beta, m);
	}
}

// ------------------------------------------------------------------
// OBSD per-layer drift (CPU reference).  Forward (sign=+1): q += scale·a ⊙ tanh(gamma·x̂ + beta),
// x̂ = (p−μ)/σ, μ,σ the per-row mean/std of p.  Inverse (sign=−1): subtract the same term
// (recompute x̂ from p, which is untouched by the drift).  Parameter-free normalize: μ,σ carry
// no learnable affine; gamma=M⁻¹ and beta act AFTER the normalize, INSIDE the tanh.
inline void drift_into_q_row(const float* p, float* q, const float* a,
                             const float* gamma, const float* beta,
                             float sign, float scale, unsigned int m, float eps)
{
	double sum=0.0; for (unsigned i=0;i<m;++i) sum += p[i];
	const float mu = static_cast<float>(sum/static_cast<double>(m));
	double vs=0.0; for (unsigned i=0;i<m;++i){ float d=p[i]-mu; vs += static_cast<double>(d)*d; }
	const float sigma = sqrtf(static_cast<float>(vs/static_cast<double>(m)) + eps);
	const float inv = 1.0f/sigma;
	for (unsigned i=0;i<m;++i){
		float xhat = (p[i]-mu)*inv;
		float u = gamma[i]*xhat + beta[i];
		q[i] += sign*scale*a[i]*tanhf(u);
	}
}

inline void drift_into_q(const float* p, float* q, const float* a,
                         const float* gamma, const float* beta,
                         float sign, float scale, unsigned int T, unsigned int m, float eps)
{
	for (unsigned t=0;t<T;++t)
		drift_into_q_row(p + t*m, q + t*m, a, gamma, beta, sign, scale, m, eps);
}

// OBSD drift backward (CPU reference).  Given dq_out and p (μ,σ re-derived from p — reanchor),
// accumulate da, dgamma(=dM⁻¹), dbeta, and dp.  Chain:
//   u = gamma·x̂ + beta ; s = tanh(u) ; q_out = q_in + scale·a·s
//   da    += Σ_t scale·s·dq_out
//   du     = scale·a·(1−s²)·dq_out ;  dgamma += Σ_t du·x̂ ;  dbeta += Σ_t du
//   dp     = normalize_backward(du as dout, p, gamma=1)   [parameter-free; reanchored stats]
// dp/da/dgamma/dbeta are ACCUMULATED (pre-zero by caller).
inline void drift_backward(const float* dq_out, const float* p, const float* a,
                           const float* gamma, const float* beta, float scale,
                           unsigned int T, unsigned int m, float eps,
                           float* dp, float* da, float* dgamma, float* dbeta)
{
	// Per-row scratch hoisted out of the t-loop and reused across rows.
	std::vector<float> xhat(m), g(m);
	for (unsigned t=0;t<T;++t)
	{
		const float* pr = p + t*m; const float* dr = dq_out + t*m; float* dpr = dp + t*m;
		double sum=0.0; for (unsigned i=0;i<m;++i) sum+=pr[i];
		const float mu=static_cast<float>(sum/static_cast<double>(m));
		double vs=0.0; for (unsigned i=0;i<m;++i){ float d=pr[i]-mu; vs+=static_cast<double>(d)*d; }
		const float sigma=sqrtf(static_cast<float>(vs/static_cast<double>(m))+eps); const float inv=1.0f/sigma;
		// Per-row du (=dL/du), and the grad flowing into the parameter-free
		// normalize: g_i = dL/dx̂_i = du_i·gamma_i (the affine gamma·x̂ sits
		// BETWEEN x̂ and the loss, so it carries into the LN-backward upstream).
		double sum_g=0.0, sum_g_xh=0.0;
		for (unsigned i=0;i<m;++i){
			float xh=(pr[i]-mu)*inv; xhat[i]=xh;
			float u=gamma[i]*xh+beta[i]; float s=tanhf(u); float sp=1.0f-s*s;
			da[i]     += scale*s*dr[i];
			float dui = scale*a[i]*sp*dr[i];
			dgamma[i] += dui*xh;            // dM⁻¹ = Σ_t dL/du·x̂
			dbeta[i]  += dui;
			float gi  = dui*gamma[i]; g[i]=gi;
			sum_g     += gi; sum_g_xh += static_cast<double>(gi)*xh;
		}
		const float mean_g=static_cast<float>(sum_g/static_cast<double>(m)), mean_g_xh=static_cast<float>(sum_g_xh/static_cast<double>(m));
		// Standard LN-backward for a parameter-free normalize y=(p−μ)/σ, with
		// upstream grad g (=dL/dx̂):  dp_i = (1/σ)·( g_i − mean(g) − x̂_i·mean(g·x̂) )
		for (unsigned i=0;i<m;++i)
			dpr[i] += inv*( g[i] - mean_g - xhat[i]*mean_g_xh );
	}
}

// ------------------------------------------------------------------
// SORC: per-channel symplectic rotation coupling (CPU reference).
//
// Rotates the (q,p) state via a per-channel angle theta(phi) = s_warm * theta_max * tanh(phi).
// The rotation is realized as 3 shears: q += a*p, p += c*q, q += a*p,
// where a = -tan(theta/2) and c = sin(theta).
// This composes to the 2D rotation matrix R(theta) = [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]].

inline void rot_coeffs(float phi, float theta_max, float s_warm, float& a, float& c, float& theta_eff) {
	theta_eff = s_warm * theta_max * tanhf(phi);
	a = -tanf(0.5f * theta_eff);
	c = sinf(theta_eff);
}

// Forward: (q,p) <- R(theta)(q,p) per element, via 3 shears. a,c are per-channel [m], broadcast over t.
inline void rot_forward(float* q, float* p, const float* a, const float* c, unsigned int T, unsigned int m) {
	for (unsigned t=0;t<T;++t) for (unsigned i=0;i<m;++i) {
		unsigned long k=(unsigned long)t*m+i; float qv=q[k], pv=p[k], ai=a[i], ci=c[i];
		qv = qv + ai*pv;   // shear1: q += a*p
		pv = pv + ci*qv;   // shear2: p += c*q
		qv = qv + ai*pv;   // shear3: q += a*p
		q[k]=qv; p[k]=pv;
	}
}

// Inverse from (q2,p1): recover (q0,p0).
inline void rot_inverse(float* q, float* p, const float* a, const float* c, unsigned int T, unsigned int m) {
	for (unsigned t=0;t<T;++t) for (unsigned i=0;i<m;++i) {
		unsigned long k=(unsigned long)t*m+i; float qv=q[k], pv=p[k], ai=a[i], ci=c[i];
		qv = qv - ai*pv;   // q1 = q2 - a*p1
		pv = pv - ci*qv;   // p0 = p1 - c*q1
		qv = qv - ai*pv;   // q0 = q1 - a*p0
		q[k]=qv; p[k]=pv;
	}
}

// Backward: given output adjoints (dq_out=dL/dq2, dp_out=dL/dp1) and the rotation INPUT (q_in=q0,p_in=p0),
// produce input adjoints (dq_in,dp_in) and accumulate da,dc (per channel). Recomputes q1,p1 from inputs.
inline void rot_backward(const float* dq_out, const float* dp_out, const float* q_in, const float* p_in,
                         const float* a, const float* c, unsigned int T, unsigned int m,
                         float* dq_in, float* dp_in, float* da, float* dc) {
	for (unsigned t=0;t<T;++t) for (unsigned i=0;i<m;++i) {
		unsigned long k=(unsigned long)t*m+i; float ai=a[i], ci=c[i];
		float q0=q_in[k], p0=p_in[k];
		float q1=q0+ai*p0;          // shear1
		float p1=p0+ci*q1;          // shear2  (q2 not needed; dq_out is adjoint of q2)
		float dq2=dq_out[k], dp1=dp_out[k];
		// shear3 bwd: q2=q1+a*p1
		float dq1=dq2; float dp1_acc=dp1 + ai*dq2; float da_el=p1*dq2;
		// shear2 bwd: p1=p0+c*q1
		float dp0=dp1_acc; dq1 += ci*dp1_acc; float dc_el=q1*dp1_acc;
		// shear1 bwd: q1=q0+a*p0
		float dq0=dq1; dp0 += ai*dq1; da_el += p0*dq1;
		dq_in[k]=dq0; dp_in[k]=dp0;
		da[i]+=da_el; dc[i]+=dc_el;
	}
}

// WhiSC per-channel symplectic whitening scale. sign=+1 whitens (q/=a, p*=a);
// sign=-1 unwhitens (q*=a, p/=a). a is per-channel [m], broadcast over T. det=1.
inline void whisc_scale(float* q, float* p, const float* a, float sign, unsigned int T, unsigned int m) {
	for (unsigned t=0;t<T;++t) for (unsigned i=0;i<m;++i) {
		unsigned long k=(unsigned long)t*m+i; float ai=a[i];
		if (sign > 0.0f) { q[k] = q[k]/ai; p[k] = p[k]*ai; }
		else             { q[k] = q[k]*ai; p[k] = p[k]/ai; }
	}
}

// WhiSC per-channel EMA second-moment stats + whitening scale derivation.
// Qbar=E[q^2], Pbar=E[p^2] over T tokens, EMA-blended; a=clamp((Qbar/(Pbar+eps))^0.25, 1/clamp, clamp).
inline void whisc_update_stats(const float* q, const float* p, unsigned int T, unsigned int m,
                               float ema, float eps, float clamp, float* Pbar, float* Qbar, float* a) {
	for (unsigned i=0;i<m;++i) {
		double sq=0.0, sp=0.0;
		for (unsigned t=0;t<T;++t) { float qv=q[(unsigned long)t*m+i], pv=p[(unsigned long)t*m+i]; sq+=(double)qv*qv; sp+=(double)pv*pv; }
		float mq=(float)(sq/(double)T), mp=(float)(sp/(double)T);
		Qbar[i]=(1.0f-ema)*Qbar[i]+ema*mq;
		Pbar[i]=(1.0f-ema)*Pbar[i]+ema*mp;
		float ai=powf(Qbar[i]/(Pbar[i]+eps), 0.25f);
		float lo=1.0f/clamp, hi=clamp;
		a[i] = (ai<lo)?lo:((ai>hi)?hi:ai);
	}
}

// WhiSC coefficient folding: absorbs the whitening scale into the rotation
// coefficients in place so that chiron_rot_forward(q,p,a,c,...) IS Phi=W^-1 R W.
// a[i] *= wa[i]^2  (sorc_a folds in the a^2 scale for the q-shear)
// c[i] /= wa[i]^2  (sorc_c folds in the 1/a^2 scale for the p-shear)
// wa is the per-channel WhiSC whitening scale (output of whisc_update_stats).
inline void chiron_whisc_fold_coeffs(float* a, float* c, const float* wa, unsigned int m) {
	for (unsigned int i = 0; i < m; ++i) {
		float wa2 = wa[i] * wa[i];
		a[i] *= wa2;
		c[i] /= wa2;
	}
}

// ------------------------------------------------------------------
// PIED — Phase-Increment Ensemble Dropout (CPU reference, 2026-07-01).
// docs/superpowers/specs/2026-07-01-chiron-pied-increment-dropout-design.md
//
// Mean-one two-point mask on the SCFA attention increment at the shear
// commit (p += eta ⊙ Y_l(q)), regenerated from a stateless counter hash —
// no RNG state, no stored masks.  Forward, inverse walk, and backward
// evaluate the identical pure function of (key, i):
//   eta_i = (mix32(key ^ i*0x9E3779B9) >= thr) ? hi : lo
//   Bernoulli arm:  thr = floor(pi * 2^32), lo = 0,      hi = 1/(1-pi)
//   Symmetric arm:  thr = 0x80000000,       lo = 1-amp,  hi = 1+amp
// Must stay bit-identical to the GPU kernels (gpu_chiron.cu
// chiron_pied_eta_dev) — guarded by the chiron-pied parity unit test.

inline unsigned int chiron_pied_mix32(unsigned int x)
{
	x ^= x >> 16; x *= 0x7FEB352Du;
	x ^= x >> 15; x *= 0x846CA68Bu;
	x ^= x >> 16;
	return x;
}

inline float chiron_pied_eta(unsigned int key, unsigned int i,
                             unsigned int thr, float lo, float hi)
{
	const unsigned int h = chiron_pied_mix32(key ^ (i * 0x9E3779B9u));
	return (h >= thr) ? hi : lo;
}

// Masked shear-commit: p[i] += alpha * eta_i * (a[i] + b[i]).  The inverse
// walk passes -alpha (exact IEEE sign flip of the identical product), so the
// subtracted increment is bit-identical to the added one.
inline void chiron_scfa_axpy2_masked_cpu(float* p, float alpha,
                                         const float* a, const float* b,
                                         unsigned int n,
                                         unsigned int key, unsigned int thr,
                                         float lo, float hi)
{
	for (unsigned int i = 0; i < n; ++i)
	{
		const float eta = chiron_pied_eta(key, i, thr, lo, hi);
		p[i] += alpha * (eta * (a[i] + b[i]));
	}
}

// Masked dy hand-off: dst[i] = alpha * eta_i * src[i].  Same key as the
// commit so the increment-branch adjoint sees the identical eta field.
inline void chiron_incdrop_scale_copy_cpu(float* dst, float alpha,
                                          const float* src, unsigned int n,
                                          unsigned int key, unsigned int thr,
                                          float lo, float hi)
{
	for (unsigned int i = 0; i < n; ++i)
	{
		const float eta = chiron_pied_eta(key, i, thr, lo, hi);
		dst[i] = alpha * (eta * src[i]);
	}
}

// ------------------------------------------------------------------
// PACT — Profile Anti-Cancellation Tax (2026-07-04): CPU references.
//
// Design: docs/superpowers/specs/2026-07-04-chiron-pact-anti-cancellation-design.md §5.
// A deterministic, gated penalty on the null component of each (token,channel)
// depth profile Y = u ∈ R^L (u_l = clean increment at layer l), taxing only
// actual sign-cancellation, function-preserving in the damped sum A.
//   A     = sum_l D_l·u_l          (damped signal;  D_l = D[l,i] > 0)
//   M     = sum_l D_l·|u_l|        (damped mass)
//   Dsq   = sum_l D_l^2 ,  D1 = sum_l D_l
//   res_l = u_l − A·D_l/Dsq        (component orthogonal to the damping dir)
//   chi   = clamp01((M^2−A^2)/(M^2+epsM·Dsq·sigma^2))   (detached; 1 if gate off)
//   r_l   = res_l/sigma ,  H(r) = r^2 if |r|<=kappa else kappa(2|r|−kappa)
//   penalty(t,i) = chi·(1/Dsq)·sum_l H(r_l) ;  L += (lambda/(T·m))·sum_{t,i} penalty
//   field  g_l   = coef·chi·clamp(r_l,±kappa)/(Dsq·sigma) ,  coef = 2·lambda/(T·m)
// Exact identity: sum_l D_l·g_l = 0 (function-preserving).  Must stay
// bit-parity with the GPU kernels — guarded by the chiron-pact unit test.

static const float PACT_EPS0 = 1e-6f;   // sigma floor
static const float PACT_EPSM = 1.0f;    // gate floor multiplier

// Damping table from per-layer angles phi[l] (each length m):
//   D[l*m+i]  = prod_{l'>=l} cos(thetaMax·tanh(phi[l'][i]))
//   Dsq[i]    = sum_l D[l*m+i]^2 ,  D1[i] = sum_l D[l*m+i]
inline void chiron_pact_damp_ref(const float* const* phi, int L, int m,
                                 float thetaMax, float* D, float* Dsq, float* D1)
{
	for (int i = 0; i < m; ++i) { Dsq[i] = 0.0f; D1[i] = 0.0f; }
	for (int i = 0; i < m; ++i)
	{
		double suffix = 1.0; // prod_{l'>l} cos(theta_{l'})
		for (int l = L - 1; l >= 0; --l)
		{
			const double th = (double)thetaMax * (double)tanhf(phi[l][i]);
			const double c = cos(th);
			const float d = (float)(suffix * c);
			D[(size_t)l * (size_t)m + (size_t)i] = d;
			Dsq[i] += d * d;
			D1[i] += d;
			suffix *= c;
		}
	}
}

// Detached sign-coherence gate for one (t,i).  gateOn==0 => returns 1.
inline float chiron_pact_chi(float A, float M, float Dsq_i, float sigma_i, int gateOn)
{
	if (!gateOn) return 1.0f;
	const float sg = (sigma_i > PACT_EPS0) ? sigma_i : PACT_EPS0;
	const float num = M * M - A * A;
	const float den = M * M + PACT_EPSM * Dsq_i * (sg * sg);
	float chi = (den > 0.0f) ? (num / den) : 0.0f;
	if (chi < 0.0f) chi = 0.0f;
	if (chi > 1.0f) chi = 1.0f;
	return chi;
}

// Huber value H(r) and derivative H'(r) = 2·clamp(r,±kappa).
inline float chiron_pact_huber(float r, float kappa)
{
	const float ar = (r < 0.0f) ? -r : r;
	return (ar <= kappa) ? (r * r) : (kappa * (2.0f * ar - kappa));
}
inline float chiron_pact_huber_d(float r, float kappa)
{
	float rc = r;
	if (rc > kappa) rc = kappa;
	if (rc < -kappa) rc = -kappa;
	return 2.0f * rc;
}

// Field for one element (layer l, channel i) given the profile summaries A,M
// (over the full depth) and the detached scale sigma_i (the Huber KNEE only —
// NOT a magnitude normalizer):
//   g = coef·chi·sigma·clamp(res/sigma, ±kappa) / Dsq ,  res = u − A·Dl/Dsq
// The field is increment-ENERGY scaled: g ~ O(res) ~ O(‖increment‖), bounded
// and vanishing as increments → 0 (safe at init), rho-free (sigma is the
// increment scale, not p/q).  This is the gradient of the increment-energy
// penalty phi = chi/Dsq · sum_l Huber_{kappa·sigma}(res_l) — see 2026-07-04
// E2 refinement: the original scale-free phi = chi·‖P_⊥Y‖²/(Dsq·sigma²) had a
// gradient ~1/sigma that diverged as increments → 0 (field/dy blew to 5.4 at
// fresh init).  Energy scaling keeps the field O(increment), tunable by lambda.
inline float chiron_pact_field(float u, float A, float M,
                               float Dl, float Dsq_i, float sigma_i,
                               float coef, float kappa, int gateOn)
{
	const float sg = (sigma_i > PACT_EPS0) ? sigma_i : PACT_EPS0;
	const float chi = chiron_pact_chi(A, M, Dsq_i, sg, gateOn);
	const float res = u - A * Dl / Dsq_i;
	float rc = res / sg;
	if (rc > kappa) rc = kappa;
	if (rc < -kappa) rc = -kappa;
	return coef * chi * sg * rc / Dsq_i;
}

// Length-carrying profile helpers (used by the unit tests and as the spec of
// record for the fused GPU kernels).  penalty value for one (t,i) is
// chi·(1/Dsq)·sum_l H(res_l/sigma); caller scales the sum over (t,i) by
// lambda/(T·m).
inline void chiron_pact_profile_reduce(const float* u, const float* Dcol, int L,
                                       float* A, float* M, float* Dsq)
{
	float a = 0.0f, mm = 0.0f, dsq = 0.0f;
	for (int l = 0; l < L; ++l)
	{
		a += Dcol[l] * u[l];
		mm += Dcol[l] * ((u[l] < 0.0f) ? -u[l] : u[l]);
		dsq += Dcol[l] * Dcol[l];
	}
	*A = a; *M = mm; *Dsq = dsq;
}

inline double chiron_pact_value_L(const float* u, const float* Dcol, int L,
                                  float sigma_i, float kappa, int gateOn)
{
	float A, M, Dsq;
	chiron_pact_profile_reduce(u, Dcol, L, &A, &M, &Dsq);
	const float sg = (sigma_i > PACT_EPS0) ? sigma_i : PACT_EPS0;
	const float chi = chiron_pact_chi(A, M, Dsq, sg, gateOn);
	double acc = 0.0;
	for (int l = 0; l < L; ++l)
	{
		const float res = u[l] - A * Dcol[l] / Dsq;
		acc += (double)chiron_pact_huber(res / sg, kappa); // = (res/sg)^2 in the quad region
	}
	// Energy penalty phi = chi/Dsq · sum_l sigma^2·H(res/sigma) (Huber knee kappa·sigma).
	return (double)chi * (double)(sg * sg) * acc / (double)Dsq;
}

inline void chiron_pact_field_L(const float* u, const float* Dcol, int L,
                                float sigma_i, float coef, float kappa, int gateOn,
                                float* g)
{
	float A, M, Dsq;
	chiron_pact_profile_reduce(u, Dcol, L, &A, &M, &Dsq);
	for (int l = 0; l < L; ++l)
		g[l] = chiron_pact_field(u[l], A, M, Dcol[l], Dsq, sigma_i, coef, kappa, gateOn);
}

// ------------------------------------------------------------------
// Sketch residual correction (framework §4.4).
//
// To prevent BF16 round-off from compounding across the block-inverse
// chain, each forward pass stores a tiny FP32 "sketch" z_ℓ = S_ℓ · x_ℓ,
// where S_ℓ ∈ R^{r × N} is a Gaussian random matrix and x_ℓ is the flat
// block-input state of size N (typically N = 2·T·m for the paired (q, p)
// hidden state). During backward, after running the BF16 inverse to get
// an approximation x̃_ℓ, the stored sketch z_ℓ is compared against
// S_ℓ · x̃_ℓ and the residual is lifted back into x-space to correct
// x̃_ℓ:
//
//   z_ℓ_fresh   = S_ℓ · vec(x̃_ℓ)
//   residual    = z_ℓ − z_ℓ_fresh
//   correction  = (S_ℓ^T / r) · residual
//   x̂_ℓ        = x̃_ℓ + correction
//
// Theorem (unbiasedness). If S_ℓ has i.i.d. N(0, 1) entries and S_ℓ is
// independent of (x_ℓ, x̃_ℓ), then E[x̂_ℓ] = x_ℓ. Variance of each
// coordinate of x̂_ℓ is O(||x − x̃||² / r).
//
// Proof sketch. E[S_ℓ^T S_ℓ] = r · I_N (each entry is a sum of r
// independent mean-zero products of variance 1). Thus
// E[(S_ℓ^T S_ℓ / r) · (x − x̃)] = x − x̃ identically.
//
// The sketch matrix can be seeded (e.g., hash of layer index) so neither
// S_ℓ nor the random seed needs to be stored beyond a short seed word
// per layer. This prototype stores S_ℓ explicitly for clarity; GPU
// production kernels should regenerate S_ℓ on-the-fly per layer from
// a deterministic seed.

// sketch_project: compute z = S · x, where S is [r, N] row-major,
// x is [N] row-major.  Output z is [r] row-major.
inline void sketch_project(const float* S, const float* x,
                           unsigned int r, unsigned int N, float* z_out)
{
	for (unsigned int k = 0; k < r; ++k)
	{
		float acc = 0.0f;
		const float* row = S + k * N;
		for (unsigned int i = 0; i < N; ++i)
			acc += row[i] * x[i];
		z_out[k] = acc;
	}
}

// sketch_lift: compute correction = (S^T / r) · residual, adding in-place
// to x_out.  S is [r, N], residual is [r], x_out is [N].
//
// This is the minimum-norm lift of residual back to the full state space;
// E[correction] = (1/r) E[S^T S] (x - x̃) = (x - x̃) by the identity
// above.
inline void sketch_lift_add(const float* S, const float* residual,
                            unsigned int r, unsigned int N, float* x_out)
{
	const float inv_r = 1.0f / static_cast<float>(r);
	for (unsigned int i = 0; i < N; ++i)
	{
		float acc = 0.0f;
		for (unsigned int k = 0; k < r; ++k)
			acc += S[k * N + i] * residual[k];
		x_out[i] += acc * inv_r;
	}
}

// -------------- Batched + tiled CPU variants (Phase 3.5 optimized) --------
//
// The per-token sketch ops above are correct but CPU-inefficient: they load
// each row of S once per token, yielding ~3 GFLOP/s on modern x86 because
// the inner loop is memory-bound and not vectorizable across tokens.
//
// The batched variants process all T tokens in one blocked SIMD-friendly
// matrix multiply:
//   - sketch_project_batched: Z [T, r] = X [T, N] · S^T [N, r]
//   - sketch_lift_add_batched: X [T, N] += (1/r) · Z [T, r] · S [r, N]
//
// Tiling: T/M_TILE outer, r or N inner. The `K_TILE` inner loop accumulates
// FMAs with S streamed into L1. On a 4-wide SSE or 8-wide AVX host, GCC can
// auto-vectorize the inner K loop; we keep the loop structure compiler-
// friendly (contiguous strided loads, known trip counts via tile bounds).

#ifndef GLADES_CHIRON_TILE_M
#define GLADES_CHIRON_TILE_M 8
#endif
#ifndef GLADES_CHIRON_TILE_N
#define GLADES_CHIRON_TILE_N 32
#endif
#ifndef GLADES_CHIRON_TILE_K
#define GLADES_CHIRON_TILE_K 64
#endif

// Z [T, r] = X [T, N] · S^T [N, r]   (i.e. Z[t, k] = Σ_i X[t, i] * S[k, i])
//
// Performance: blocked tiling on (M, N, K) plus an unrolled inner FMA loop.
// Outer M dim is the token index (trivially independent) — can be parallelized
// by the caller via OpenMP around the mb loop. We keep the function itself
// serial-safe to retain header-only reusability.
inline void sketch_project_batched(const float* S, const float* X,
                                    unsigned int T, unsigned int N, unsigned int r,
                                    float* Z)
{
	// Zero output.
	const size_t total = static_cast<size_t>(T) * static_cast<size_t>(r);
	for (size_t i = 0; i < total; ++i) Z[i] = 0.0f;

	const unsigned int TM = GLADES_CHIRON_TILE_M;
	const unsigned int TN = GLADES_CHIRON_TILE_N; // tile over r (output cols)
	const unsigned int TK = GLADES_CHIRON_TILE_K;

	for (unsigned int mb = 0; mb < T; mb += TM)
	{
		const unsigned int me = (mb + TM < T) ? (mb + TM) : T;
		for (unsigned int nb = 0; nb < r; nb += TN)
		{
			const unsigned int ne = (nb + TN < r) ? (nb + TN) : r;
			for (unsigned int kb = 0; kb < N; kb += TK)
			{
				const unsigned int ke = (kb + TK < N) ? (kb + TK) : N;

				// Inner: Z[mb:me, nb:ne] += X[mb:me, kb:ke] · S[nb:ne, kb:ke]^T.
				// Token-major outer: better Z locality across the k loop.
				for (unsigned int t = mb; t < me; ++t)
				{
					const float* xRow = X + static_cast<size_t>(t) * N;
					float*       zRow = Z + static_cast<size_t>(t) * r;
					for (unsigned int k = nb; k < ne; ++k)
					{
						const float* sRow = S + static_cast<size_t>(k) * N;
						// Accumulate into four parallel lanes so the compiler
						// can emit 4-wide FMAs even without SIMD intrinsics.
						float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
						unsigned int i = kb;
						const unsigned int ke4 = kb + ((ke - kb) & ~3u);
						for (; i < ke4; i += 4)
						{
							acc0 += xRow[i + 0] * sRow[i + 0];
							acc1 += xRow[i + 1] * sRow[i + 1];
							acc2 += xRow[i + 2] * sRow[i + 2];
							acc3 += xRow[i + 3] * sRow[i + 3];
						}
						float acc = acc0 + acc1 + acc2 + acc3;
						for (; i < ke; ++i) acc += xRow[i] * sRow[i];
						zRow[k] += acc;
					}
				}
			}
		}
	}
}

// X [T, N] += (1/r) · Z [T, r] · S [r, N]  (i.e. X[t, i] += (1/r) · Σ_k Z[t, k] · S[k, i])
//
// Loop order: outer-product with k outermost inside a token.  Each row of
// S[k, :] is loaded once and multiplied by Z[t, k] into a contiguous stripe
// of X[t, :].  This gives SEQUENTIAL access to both X[t, :] and S[k, :],
// which is dramatically more cache-friendly than the sum-k-over-stride-N
// access pattern of a straightforward dot-product formulation.
inline void sketch_lift_add_batched(const float* S, const float* Z,
                                     unsigned int T, unsigned int N, unsigned int r,
                                     float* X)
{
	const float inv_r = 1.0f / static_cast<float>(r);
	const unsigned int TM = GLADES_CHIRON_TILE_M;
	const unsigned int TN = GLADES_CHIRON_TILE_N;
	const unsigned int TK = GLADES_CHIRON_TILE_K;

	for (unsigned int mb = 0; mb < T; mb += TM)
	{
		const unsigned int me = (mb + TM < T) ? (mb + TM) : T;
		for (unsigned int nb = 0; nb < N; nb += TN)
		{
			const unsigned int ne = (nb + TN < N) ? (nb + TN) : N;
			for (unsigned int kb = 0; kb < r; kb += TK)
			{
				const unsigned int ke = (kb + TK < r) ? (kb + TK) : r;

				for (unsigned int t = mb; t < me; ++t)
				{
					float*       xRow = X + static_cast<size_t>(t) * N;
					const float* zRow = Z + static_cast<size_t>(t) * r;
					// Outer over k: each k-row of S is contiguous in i.
					for (unsigned int k = kb; k < ke; ++k)
					{
						const float* sRow = S + static_cast<size_t>(k) * N;
						const float  scale = inv_r * zRow[k];
						unsigned int i = nb;
						const unsigned int ne4 = nb + ((ne - nb) & ~3u);
						for (; i < ne4; i += 4)
						{
							xRow[i + 0] += scale * sRow[i + 0];
							xRow[i + 1] += scale * sRow[i + 1];
							xRow[i + 2] += scale * sRow[i + 2];
							xRow[i + 3] += scale * sRow[i + 3];
						}
						for (; i < ne; ++i)
							xRow[i] += scale * sRow[i];
					}
				}
			}
		}
	}
}

// ------------------------------------------------------------------
// SIRA: Symplectic-Invariant Regularized Action (2026-05-24)
//
// Train-only CHIRON-native phase-space regularizer.  These CPU reference
// helpers intentionally live beside the CHIRON phase-state primitives rather
// than in the generic transformer kernels: SIRA is defined on paired
// trajectories (p_l, q_l), q-driven shears S_l(q_l), and q-state evolution.
//
// Production safety invariant: coef <= 0 is a strict disabled path.  The
// public helpers return exactly 0.0f before reading any phase pointers or
// touching any output buffers, so callers can wire the flag through hot paths
// without perturbing default production behavior.
// ------------------------------------------------------------------

inline bool sira_should_apply(float coef, long long step, int warmupSteps)
{
	return (coef > 0.0f) && (step >= static_cast<long long>(warmupSteps));
}

inline float sira_pseudo_huber(float z, float tau)
{
	const float t = (tau > 0.0f) ? tau : 0.2f;
	const float r = z / t;
	return t * t * (sqrtf(1.0f + r * r) - 1.0f);
}

inline float sira_safe_positive(float x, float eps)
{
	const float e = (eps > 0.0f) ? eps : 1e-12f;
	return (x > e) ? x : e;
}

inline float sira_pseudo_huber_grad(float z, float tau)
{
	const float t = (tau > 0.0f) ? tau : 0.2f;
	const float r = z / t;
	return z / sqrtf(1.0f + r * r);
}

// Terminal phase-state SIRA loss used by the first active training-path port.
//
// This low-overhead CHIRON-native objective can be evaluated from the final
// paired state (p_L, q_L) already resident in the O(1)-activation trainer.  It
// is intentionally default-off and returns exactly 0 at coef <= 0 before
// reading p/q.  The richer trajectory SIRA objective below remains the
// reference target for later ports that stage per-layer/bucket terms.
//
// p2Mean/q2Mean/pqMean are means over all T*m coordinates.  The terms are:
//   energy:  pseudo-Huber(log(0.5*(E[p^2]+E[q^2])))
//   balance: pseudo-Huber(0.5*log(E[p^2]/E[q^2]))
//   action:  pseudo-Huber(E[p*q]/sqrt(E[p^2]*E[q^2]))
inline float sira_terminal_phase_loss_from_stats(float p2Mean,
                                                  float q2Mean,
                                                  float pqMean,
                                                  float coef,
                                                  float energyWeight,
                                                  float balanceWeight,
                                                  float actionWeight,
                                                  float huberTau,
                                                  float eps)
{
	if (coef <= 0.0f)
		return 0.0f;
	if (energyWeight <= 0.0f && balanceWeight <= 0.0f && actionWeight <= 0.0f)
		return 0.0f;

	const float safeEps = (eps > 0.0f) ? eps : 1e-12f;
	const float p2 = sira_safe_positive(p2Mean, safeEps);
	const float q2 = sira_safe_positive(q2Mean, safeEps);
	const float energyZ = logf(sira_safe_positive(0.5f * (p2Mean + q2Mean), safeEps));
	const float balanceZ = 0.5f * (logf(p2) - logf(q2));
	const float actionDenom = sqrtf(p2 * q2);
	const float actionZ = (actionDenom > safeEps) ? (pqMean / actionDenom) : 0.0f;

	double total = 0.0;
	if (energyWeight > 0.0f)
		total += static_cast<double>(energyWeight) *
		         static_cast<double>(sira_pseudo_huber(energyZ, huberTau));
	if (balanceWeight > 0.0f)
		total += static_cast<double>(balanceWeight) *
		         static_cast<double>(sira_pseudo_huber(balanceZ, huberTau));
	if (actionWeight > 0.0f)
		total += static_cast<double>(actionWeight) *
		         static_cast<double>(sira_pseudo_huber(actionZ, huberTau));
	return static_cast<float>(static_cast<double>(coef) * total);
}

inline float sira_terminal_phase_loss(const float* p,
                                       const float* q,
                                       unsigned int n,
                                       float coef,
                                       float energyWeight,
                                       float balanceWeight,
                                       float actionWeight,
                                       float huberTau,
                                       float eps)
{
	if (coef <= 0.0f)
		return 0.0f;
	if (n == 0u)
		return 0.0f;
	if (energyWeight <= 0.0f && balanceWeight <= 0.0f && actionWeight <= 0.0f)
		return 0.0f;
	if (!p || !q)
		return 0.0f;

	double p2 = 0.0;
	double q2 = 0.0;
	double pq = 0.0;
	for (unsigned int i = 0u; i < n; ++i)
	{
		p2 += static_cast<double>(p[i]) * static_cast<double>(p[i]);
		q2 += static_cast<double>(q[i]) * static_cast<double>(q[i]);
		pq += static_cast<double>(p[i]) * static_cast<double>(q[i]);
	}
	const double denom = static_cast<double>(n);
	return sira_terminal_phase_loss_from_stats(static_cast<float>(p2 / denom),
	                                           static_cast<float>(q2 / denom),
	                                           static_cast<float>(pq / denom),
	                                           coef, energyWeight, balanceWeight,
	                                           actionWeight, huberTau, eps);
}

// Phase-0 SIRA diagnostics from a CHIRON phase trajectory.
//
// pStates/qStates: [nTransitions+1, T, m] state boundaries.
// shearStates:     [nTransitions, T, m] q-driven shears added to p.
//
// Optional state outputs are [nTransitions+1, nBuckets]:
//   rmsP, rmsQ:  raw bucket RMS for p/q.
//   energy:      normalized 0.5 * (||p||^2/sigma_p^2 + ||q||^2/sigma_q^2).
//   balance:     normalized 0.5 * log((||p||^2/sigma_p^2)/(||q||^2/sigma_q^2)).
// Optional transition outputs are [nTransitions, nBuckets]:
//   rmsShear:    raw bucket RMS of q-driven shear S_l(q_l).
//   action:      normalized midpoint-action proxy used by SIRA.
//
// This helper performs detached reductions only; it does not add a loss or
// affect production training unless a caller explicitly invokes it.
inline bool sira_phase_diagnostics_from_trajectory(const float* pStates,
                                                    const float* qStates,
                                                    const float* shearStates,
                                                    unsigned int nTransitions,
                                                    unsigned int T,
                                                    unsigned int m,
                                                    unsigned int nBuckets,
                                                    float* rmsP,
                                                    float* rmsQ,
                                                    float* rmsShear,
                                                    float* energy,
                                                    float* balance,
                                                    float* action,
                                                    float eps)
{
	if (T == 0u || m == 0u || nBuckets == 0u)
		return false;

	const unsigned int nStates = nTransitions + 1u;
	const bool needEnergyBalance = (energy != NULL) || (balance != NULL);
	const bool needP = (rmsP != NULL) || needEnergyBalance || (action != NULL);
	const bool needQ = (rmsQ != NULL) || needEnergyBalance || (action != NULL);
	const bool needShear = (rmsShear != NULL) || (action != NULL);
	if (!needP && !needQ && !needShear)
		return true;
	if (needP && pStates == NULL)
		return false;
	if (needQ && qStates == NULL)
		return false;
	if (needShear && shearStates == NULL)
		return false;

	const float safeEps = (eps > 0.0f) ? eps : 1e-12f;
	const size_t stateStride = static_cast<size_t>(T) * static_cast<size_t>(m);
	const size_t stateDiagCount = static_cast<size_t>(nStates) * nBuckets;
	const size_t transDiagCount = static_cast<size_t>(nTransitions) * nBuckets;
	for (size_t i = 0u; i < stateDiagCount; ++i)
	{
		if (rmsP) rmsP[i] = 0.0f;
		if (rmsQ) rmsQ[i] = 0.0f;
		if (energy) energy[i] = 0.0f;
		if (balance) balance[i] = 0.0f;
	}
	for (size_t i = 0u; i < transDiagCount; ++i)
	{
		if (rmsShear) rmsShear[i] = 0.0f;
		if (action) action[i] = 0.0f;
	}

	std::vector<float> sigmaP2;
	std::vector<float> sigmaQ2;
	if (needEnergyBalance)
	{
		sigmaP2.assign(nStates, 0.0f);
		sigmaQ2.assign(nStates, 0.0f);
		for (unsigned int l = 0u; l < nStates; ++l)
		{
			double p2 = 0.0;
			double q2 = 0.0;
			const float* p = pStates + static_cast<size_t>(l) * stateStride;
			const float* q = qStates + static_cast<size_t>(l) * stateStride;
			for (size_t i = 0u; i < stateStride; ++i)
			{
				p2 += static_cast<double>(p[i]) * static_cast<double>(p[i]);
				q2 += static_cast<double>(q[i]) * static_cast<double>(q[i]);
			}
			const double denom = (stateStride > 0u) ? static_cast<double>(stateStride) : 1.0;
			sigmaP2[l] = sira_safe_positive(static_cast<float>(p2 / denom), safeEps);
			sigmaQ2[l] = sira_safe_positive(static_cast<float>(q2 / denom), safeEps);
		}
	}

	if ((rmsP != NULL) || (rmsQ != NULL) || needEnergyBalance)
	{
		for (unsigned int l = 0u; l < nStates; ++l)
		{
			const float* p = pStates ? (pStates + static_cast<size_t>(l) * stateStride) : NULL;
			const float* q = qStates ? (qStates + static_cast<size_t>(l) * stateStride) : NULL;
			for (unsigned int b = 0u; b < nBuckets; ++b)
			{
				const unsigned int tb = static_cast<unsigned int>((static_cast<size_t>(b) * T) / nBuckets);
				const unsigned int te = static_cast<unsigned int>((static_cast<size_t>(b + 1u) * T) / nBuckets);
				if (te <= tb)
					continue;
				double p2 = 0.0;
				double q2 = 0.0;
				for (unsigned int t = tb; t < te; ++t)
				{
					const size_t row = static_cast<size_t>(t) * m;
					for (unsigned int j = 0u; j < m; ++j)
					{
						if (p)
						{
							const float pv = p[row + j];
							p2 += static_cast<double>(pv) * static_cast<double>(pv);
						}
						if (q)
						{
							const float qv = q[row + j];
							q2 += static_cast<double>(qv) * static_cast<double>(qv);
						}
					}
				}
				const double count = static_cast<double>(te - tb) * static_cast<double>(m);
				const size_t idx = static_cast<size_t>(l) * nBuckets + b;
				if (rmsP)
					rmsP[idx] = sqrtf(static_cast<float>(p2 / count));
				if (rmsQ)
					rmsQ[idx] = sqrtf(static_cast<float>(q2 / count));
				if (needEnergyBalance)
				{
					const float pNorm = sira_safe_positive(static_cast<float>(p2 / (count * sigmaP2[l])), safeEps);
					const float qNorm = sira_safe_positive(static_cast<float>(q2 / (count * sigmaQ2[l])), safeEps);
					if (energy)
						energy[idx] = 0.5f * (pNorm + qNorm);
					if (balance)
						balance[idx] = 0.5f * (logf(pNorm) - logf(qNorm));
				}
			}
		}
	}

	if ((rmsShear != NULL) || (action != NULL))
	{
		for (unsigned int l = 0u; l < nTransitions; ++l)
		{
			const float* p0 = pStates ? (pStates + static_cast<size_t>(l) * stateStride) : NULL;
			const float* q0 = qStates ? (qStates + static_cast<size_t>(l) * stateStride) : NULL;
			const float* q1 = qStates ? (qStates + static_cast<size_t>(l + 1u) * stateStride) : NULL;
			const float* sh = shearStates + static_cast<size_t>(l) * stateStride;
			for (unsigned int b = 0u; b < nBuckets; ++b)
			{
				const unsigned int tb = static_cast<unsigned int>((static_cast<size_t>(b) * T) / nBuckets);
				const unsigned int te = static_cast<unsigned int>((static_cast<size_t>(b + 1u) * T) / nBuckets);
				if (te <= tb)
					continue;
				double sh2 = 0.0;
				double dot = 0.0;
				double pbar2 = 0.0;
				double dq2 = 0.0;
				for (unsigned int t = tb; t < te; ++t)
				{
					const size_t row = static_cast<size_t>(t) * m;
					for (unsigned int j = 0u; j < m; ++j)
					{
						const float sv = sh[row + j];
						sh2 += static_cast<double>(sv) * static_cast<double>(sv);
						if (action)
						{
							const float pbar = p0[row + j] + 0.5f * sv;
							const float dq = q1[row + j] - q0[row + j];
							dot += static_cast<double>(pbar) * static_cast<double>(dq);
							pbar2 += static_cast<double>(pbar) * static_cast<double>(pbar);
							dq2 += static_cast<double>(dq) * static_cast<double>(dq);
						}
					}
				}
				const double count = static_cast<double>(te - tb) * static_cast<double>(m);
				const size_t idx = static_cast<size_t>(l) * nBuckets + b;
				if (rmsShear)
					rmsShear[idx] = sqrtf(static_cast<float>(sh2 / count));
				if (action)
				{
					const float denom = sqrtf(static_cast<float>(pbar2 * dq2)) + safeEps;
					action[idx] = (denom > safeEps) ? static_cast<float>(dot) / denom : 0.0f;
				}
			}
		}
	}

	return true;
}

// ------------------------------------------------------------------
// PHS: Phase-Homeostatic Servo shadow diagnostics (2026-05-27)
//
// The first PHS implementation is diagnostics/logging-only.  It reduces
// detached terminal or probe phase states by data group `g` and position
// bucket `b`, producing the quantities a future controller may consume:
//   r = log(rms(p) / rms(q))              p/q imbalance
//   s = rms(shear) / rms(p)               shear magnitude proxy
//   a = cos(p, shear)                     shear alignment
//   c = max(|q|) / rms(q)                 q-state outlier proxy
//   T = mean(tempProxy)                   optional logit/QK-temp proxy
//   ell = mean(tokenNll)                  optional unweighted CE/NLL
// No loss is returned and no gradient-bearing state is written.  The public
// helper takes an explicit `enabled` flag; when false, it returns true before
// validating dimensions, reading input pointers, or touching output buffers.
// ------------------------------------------------------------------

inline bool phs_should_log(bool enabled, long long step, int logEverySteps)
{
	if (!enabled || logEverySteps <= 0 || step < 0)
		return false;
	return (step % static_cast<long long>(logEverySteps)) == 0LL;
}

inline bool phs_detached_diagnostics_from_phase(bool enabled,
                                                const float* p,
                                                const float* q,
                                                const float* shear,
                                                const float* tokenNll,
                                                const float* tempProxy,
                                                const unsigned int* groupIds,
                                                unsigned int T,
                                                unsigned int m,
                                                unsigned int nGroups,
                                                unsigned int nBuckets,
                                                float* tokenCount,
                                                float* rmsP,
                                                float* rmsQ,
                                                float* logPqRatio,
                                                float* rmsShear,
                                                float* shearOverP,
                                                float* shearAlignment,
                                                float* qOutlier,
                                                float* meanNll,
                                                float* meanTempProxy,
                                                float eps)
{
	if (!enabled)
		return true;
	if (T == 0u || m == 0u || nGroups == 0u || nBuckets == 0u)
		return false;

	const bool needP = (rmsP != NULL) || (logPqRatio != NULL) ||
	                   (shearOverP != NULL) || (shearAlignment != NULL);
	const bool needQ = (rmsQ != NULL) || (logPqRatio != NULL) ||
	                   (qOutlier != NULL);
	const bool needShear = (rmsShear != NULL) || (shearOverP != NULL) ||
	                       (shearAlignment != NULL);
	const bool needNll = (meanNll != NULL);
	const bool needTemp = (meanTempProxy != NULL);
	const bool needAny = (tokenCount != NULL) || needP || needQ || needShear ||
	                     needNll || needTemp;
	if (!needAny)
		return true;
	if (needP && p == NULL)
		return false;
	if (needQ && q == NULL)
		return false;
	if (needShear && shear == NULL)
		return false;
	if (needNll && tokenNll == NULL)
		return false;
	if (needTemp && tempProxy == NULL)
		return false;
	if (nGroups > 1u && groupIds == NULL)
		return false;

	const float safeEps = (eps > 0.0f) ? eps : 1e-12f;
	const size_t cellCount = static_cast<size_t>(nGroups) * static_cast<size_t>(nBuckets);
	for (size_t i = 0u; i < cellCount; ++i)
	{
		if (tokenCount) tokenCount[i] = 0.0f;
		if (rmsP) rmsP[i] = 0.0f;
		if (rmsQ) rmsQ[i] = 0.0f;
		if (logPqRatio) logPqRatio[i] = 0.0f;
		if (rmsShear) rmsShear[i] = 0.0f;
		if (shearOverP) shearOverP[i] = 0.0f;
		if (shearAlignment) shearAlignment[i] = 0.0f;
		if (qOutlier) qOutlier[i] = 0.0f;
		if (meanNll) meanNll[i] = 0.0f;
		if (meanTempProxy) meanTempProxy[i] = 0.0f;
	}

	std::vector<double> counts(cellCount, 0.0);
	std::vector<double> p2(needP ? cellCount : 0u, 0.0);
	std::vector<double> q2(needQ ? cellCount : 0u, 0.0);
	std::vector<double> sh2(needShear ? cellCount : 0u, 0.0);
	std::vector<double> dot((shearAlignment != NULL) ? cellCount : 0u, 0.0);
	std::vector<double> qAbsMax((qOutlier != NULL) ? cellCount : 0u, 0.0);
	std::vector<double> nllSum(needNll ? cellCount : 0u, 0.0);
	std::vector<double> tempSum(needTemp ? cellCount : 0u, 0.0);

	for (unsigned int t = 0u; t < T; ++t)
	{
		const unsigned int g = groupIds ? groupIds[t] : 0u;
		if (g >= nGroups)
			return false;
		unsigned int b = static_cast<unsigned int>((static_cast<size_t>(t) * nBuckets) / T);
		if (b >= nBuckets)
			b = nBuckets - 1u;
		const size_t cell = static_cast<size_t>(g) * nBuckets + b;
		counts[cell] += 1.0;
		if (needNll)
			nllSum[cell] += static_cast<double>(tokenNll[t]);
		if (needTemp)
			tempSum[cell] += static_cast<double>(tempProxy[t]);

		const size_t row = static_cast<size_t>(t) * m;
		for (unsigned int j = 0u; j < m; ++j)
		{
			float pv = 0.0f;
			float qv = 0.0f;
			float sv = 0.0f;
			if (needP)
			{
				pv = p[row + j];
				p2[cell] += static_cast<double>(pv) * static_cast<double>(pv);
			}
			if (needQ)
			{
				qv = q[row + j];
				q2[cell] += static_cast<double>(qv) * static_cast<double>(qv);
				if (qOutlier != NULL)
				{
					const double aq = static_cast<double>(fabsf(qv));
					if (aq > qAbsMax[cell]) qAbsMax[cell] = aq;
				}
			}
			if (needShear)
			{
				sv = shear[row + j];
				sh2[cell] += static_cast<double>(sv) * static_cast<double>(sv);
			}
			if (shearAlignment != NULL)
				dot[cell] += static_cast<double>(pv) * static_cast<double>(sv);
		}
	}

	for (size_t cell = 0u; cell < cellCount; ++cell)
	{
		const double count = counts[cell];
		if (tokenCount)
			tokenCount[cell] = static_cast<float>(count);
		if (count <= 0.0)
			continue;
		const double coordCount = count * static_cast<double>(m);
		float pR = 0.0f;
		float qR = 0.0f;
		float shR = 0.0f;
		if (needP)
			pR = sqrtf(static_cast<float>(p2[cell] / coordCount));
		if (needQ)
			qR = sqrtf(static_cast<float>(q2[cell] / coordCount));
		if (needShear)
			shR = sqrtf(static_cast<float>(sh2[cell] / coordCount));
		if (rmsP) rmsP[cell] = pR;
		if (rmsQ) rmsQ[cell] = qR;
		if (rmsShear) rmsShear[cell] = shR;
		if (logPqRatio)
			logPqRatio[cell] = logf(sira_safe_positive(pR, safeEps)) -
			                    logf(sira_safe_positive(qR, safeEps));
		if (shearOverP)
			shearOverP[cell] = shR / sira_safe_positive(pR, safeEps);
		if (shearAlignment)
		{
			const double denom = sqrt(p2[cell] * sh2[cell]);
			shearAlignment[cell] = (denom > static_cast<double>(safeEps))
			    ? static_cast<float>(dot[cell] / (denom + static_cast<double>(safeEps)))
			    : 0.0f;
		}
		if (qOutlier)
			qOutlier[cell] = static_cast<float>(qAbsMax[cell]) /
			                 sira_safe_positive(qR, safeEps);
		if (meanNll)
			meanNll[cell] = static_cast<float>(nllSum[cell] / count);
		if (meanTempProxy)
			meanTempProxy[cell] = static_cast<float>(tempSum[cell] / count);
	}

	return true;
}

inline bool phs_update_ema(bool enabled,
                           const float* current,
                           unsigned int n,
                           float decay,
                           bool initialized,
                           float* ema)
{
	if (!enabled)
		return true;
	if (n == 0u)
		return true;
	if (current == NULL || ema == NULL)
		return false;
	if (decay < 0.0f || decay >= 1.0f)
		return false;
	if (!initialized)
	{
		for (unsigned int i = 0u; i < n; ++i)
			ema[i] = current[i];
		return true;
	}
	const float keep = decay;
	const float add = 1.0f - decay;
	for (unsigned int i = 0u; i < n; ++i)
		ema[i] = keep * ema[i] + add * current[i];
	return true;
}

// ------------------------------------------------------------------
// PTOC: Phase-space Tangent Operator Consistency shadow diagnostics.
//
// The helper is deliberately detached and generic: the caller supplies sampled
// finite-difference triplets y- = F(x-eps*u), y0 = F(x), y+ = F(x+eps*u)
// plus the normalized perturbation direction u.  PTOC summaries are local
// tangent gain and curvature diagnostics only; no loss is returned and no
// gradient-bearing state is written.  As with SIRA/PHS helpers, disabled mode
// returns before validating pointers or writing outputs.
// ------------------------------------------------------------------

inline bool ptoc_should_log(bool enabled, long long step, int logEverySteps)
{
	if (!enabled || logEverySteps <= 0 || step < 0)
		return false;
	return (step % static_cast<long long>(logEverySteps)) == 0LL;
}

inline bool ptoc_detached_diagnostics_from_triplets(bool enabled,
                                                    const float* yMinus,
                                                    const float* y0,
                                                    const float* yPlus,
                                                    const float* direction,
                                                    unsigned int nSamples,
                                                    unsigned int sampleDim,
                                                    float eps,
                                                    float eta,
                                                    float* gainMean,
                                                    float* gainMax,
                                                    float* curvatureMean,
                                                    float* curvatureMax,
                                                    float* cycleMean,
                                                    float* cycleMax)
{
	if (!enabled)
		return true;
	if (nSamples == 0u || sampleDim == 0u)
		return false;
	if (eps <= 0.0f || eta <= 0.0f)
		return false;
	if (yMinus == NULL || y0 == NULL || yPlus == NULL || direction == NULL)
		return false;

	const bool needGain = (gainMean != NULL) || (gainMax != NULL);
	const bool needCurv = (curvatureMean != NULL) || (curvatureMax != NULL);
	const bool needCycle = (cycleMean != NULL) || (cycleMax != NULL);
	if (!needGain && !needCurv && !needCycle)
		return true;

	double gainSum = 0.0;
	double curvSum = 0.0;
	double cycleSum = 0.0;
	double gainM = 0.0;
	double curvM = 0.0;
	double cycleM = 0.0;
	for (unsigned int s = 0u; s < nSamples; ++s)
	{
		const size_t row = static_cast<size_t>(s) * static_cast<size_t>(sampleDim);
		double diff2 = 0.0;
		double second2 = 0.0;
		double dir2 = 0.0;
		for (unsigned int j = 0u; j < sampleDim; ++j)
		{
			const size_t idx = row + j;
			const double ym = static_cast<double>(yMinus[idx]);
			const double y = static_cast<double>(y0[idx]);
			const double yp = static_cast<double>(yPlus[idx]);
			const double u = static_cast<double>(direction[idx]);
			const double d = yp - ym;
			const double c = yp - 2.0 * y + ym;
			diff2 += d * d;
			second2 += c * c;
			dir2 += u * u;
		}
		const double diffNorm = sqrt(diff2);
		const double secondNorm = sqrt(second2);
		const double dirNorm = sqrt(dir2);
		const double safeDir = dirNorm + static_cast<double>(eta);
		const double safeDiff = diffNorm + static_cast<double>(eta);
		const double gain = diffNorm / (2.0 * static_cast<double>(eps) * safeDir);
		const double curv = secondNorm / (static_cast<double>(eps) * static_cast<double>(eps) * safeDir);
		const double cycle = secondNorm / safeDiff;
		gainSum += gain;
		curvSum += curv;
		cycleSum += cycle;
		if (s == 0u || gain > gainM) gainM = gain;
		if (s == 0u || curv > curvM) curvM = curv;
		if (s == 0u || cycle > cycleM) cycleM = cycle;
	}
	const double invN = 1.0 / static_cast<double>(nSamples);
	if (gainMean) *gainMean = static_cast<float>(gainSum * invN);
	if (gainMax) *gainMax = static_cast<float>(gainM);
	if (curvatureMean) *curvatureMean = static_cast<float>(curvSum * invN);
	if (curvatureMax) *curvatureMax = static_cast<float>(curvM);
	if (cycleMean) *cycleMean = static_cast<float>(cycleSum * invN);
	if (cycleMax) *cycleMax = static_cast<float>(cycleM);
	return true;
}

// SIRA loss from pre-reduced layer/bucket terms.
//
// energy, balance: [nStates, nBuckets] for state boundaries l=0..nStates-1.
// action:          [nStates-1, nBuckets] for transitions l -> l+1.
//
// Returns coef * (wE*energy_drift + wB*balance_drift + wA*action_curvature).
// At coef <= 0, returns exactly 0 without reading any pointer.
inline float sira_loss_from_terms(const float* energy,
                                  const float* balance,
                                  const float* action,
                                  unsigned int nStates,
                                  unsigned int nBuckets,
                                  float coef,
                                  float energyWeight,
                                  float balanceWeight,
                                  float actionWeight,
                                  float huberTau,
                                  float eps)
{
	if (coef <= 0.0f)
		return 0.0f;
	if (nStates < 2u || nBuckets == 0u)
		return 0.0f;
	if (energyWeight > 0.0f && !energy)
		return 0.0f;
	if (balanceWeight > 0.0f && !balance)
		return 0.0f;
	if (actionWeight > 0.0f && nStates >= 4u && !action)
		return 0.0f;
	if (energyWeight <= 0.0f && balanceWeight <= 0.0f && actionWeight <= 0.0f)
		return 0.0f;

	double eLoss = 0.0;
	double bLoss = 0.0;
	double aLoss = 0.0;
	unsigned int eCount = 0u;
	unsigned int bCount = 0u;
	unsigned int aCount = 0u;

	if (energyWeight > 0.0f)
	{
		for (unsigned int l = 0u; l + 1u < nStates; ++l)
		{
			double mean = 0.0;
			for (unsigned int b = 0u; b < nBuckets; ++b)
			{
				const size_t i0 = static_cast<size_t>(l) * nBuckets + b;
				const size_t i1 = static_cast<size_t>(l + 1u) * nBuckets + b;
				const float d = logf(sira_safe_positive(energy[i1], eps))
				              - logf(sira_safe_positive(energy[i0], eps));
				mean += static_cast<double>(d);
			}
			mean /= static_cast<double>(nBuckets);
			for (unsigned int b = 0u; b < nBuckets; ++b)
			{
				const size_t i0 = static_cast<size_t>(l) * nBuckets + b;
				const size_t i1 = static_cast<size_t>(l + 1u) * nBuckets + b;
				const float d = logf(sira_safe_positive(energy[i1], eps))
				              - logf(sira_safe_positive(energy[i0], eps));
				eLoss += static_cast<double>(sira_pseudo_huber(d - static_cast<float>(mean), huberTau));
				++eCount;
			}
		}
	}

	if (balanceWeight > 0.0f)
	{
		for (unsigned int l = 0u; l + 1u < nStates; ++l)
		{
			double mean = 0.0;
			for (unsigned int b = 0u; b < nBuckets; ++b)
			{
				const size_t i0 = static_cast<size_t>(l) * nBuckets + b;
				const size_t i1 = static_cast<size_t>(l + 1u) * nBuckets + b;
				mean += static_cast<double>(balance[i1] - balance[i0]);
			}
			mean /= static_cast<double>(nBuckets);
			for (unsigned int b = 0u; b < nBuckets; ++b)
			{
				const size_t i0 = static_cast<size_t>(l) * nBuckets + b;
				const size_t i1 = static_cast<size_t>(l + 1u) * nBuckets + b;
				const float d = balance[i1] - balance[i0];
				bLoss += static_cast<double>(sira_pseudo_huber(d - static_cast<float>(mean), huberTau));
				++bCount;
			}
		}
	}

	if (actionWeight > 0.0f && nStates >= 4u)
	{
		const unsigned int nTransitions = nStates - 1u;
		for (unsigned int l = 1u; l + 1u < nTransitions; ++l)
		{
			for (unsigned int b = 0u; b < nBuckets; ++b)
			{
				const size_t im = static_cast<size_t>(l - 1u) * nBuckets + b;
				const size_t i0 = static_cast<size_t>(l) * nBuckets + b;
				const size_t ip = static_cast<size_t>(l + 1u) * nBuckets + b;
				const float d2 = action[ip] - 2.0f * action[i0] + action[im];
				aLoss += static_cast<double>(sira_pseudo_huber(d2, huberTau));
				++aCount;
			}
		}
	}

	const double eMean = (eCount > 0u) ? (eLoss / static_cast<double>(eCount)) : 0.0;
	const double bMean = (bCount > 0u) ? (bLoss / static_cast<double>(bCount)) : 0.0;
	const double aMean = (aCount > 0u) ? (aLoss / static_cast<double>(aCount)) : 0.0;
	const double total = static_cast<double>(coef) *
	    (static_cast<double>(energyWeight) * eMean +
	     static_cast<double>(balanceWeight) * bMean +
	     static_cast<double>(actionWeight) * aMean);
	return static_cast<float>(total);
}

// End-to-end SIRA loss from a CHIRON phase trajectory.
//
// pStates/qStates: [nTransitions+1, T, m] state boundaries.
// shearStates:     [nTransitions, T, m] q-driven shears added to p;
//                  required only when actionWeight contributes.
// nBuckets:        position buckets over T; bucket b covers
//                  [floor(b*T/B), floor((b+1)*T/B)).
//
// The helper performs detached reductions only.  It is intended as the
// reference objective for trainer/GPU ports; disabled mode is exact no-op.
inline float sira_loss_from_phase_trajectory(const float* pStates,
                                             const float* qStates,
                                             const float* shearStates,
                                             unsigned int nTransitions,
                                             unsigned int T,
                                             unsigned int m,
                                             unsigned int nBuckets,
                                             float coef,
                                             float energyWeight,
                                             float balanceWeight,
                                             float actionWeight,
                                             float huberTau,
                                             float eps)
{
	if (coef <= 0.0f)
		return 0.0f;
	if (nTransitions == 0u || T == 0u || m == 0u || nBuckets == 0u)
		return 0.0f;

	const unsigned int nStates = nTransitions + 1u;
	const bool useEnergy = (energyWeight > 0.0f);
	const bool useBalance = (balanceWeight > 0.0f);
	// Action curvature is a second difference across transition actions, so it
	// has no defined samples until there are at least three transitions.
	const bool useAction = (actionWeight > 0.0f) && (nTransitions >= 3u);
	if (!useEnergy && !useBalance && !useAction)
		return 0.0f;
	if ((useEnergy || useBalance || useAction) && (!pStates || !qStates))
		return 0.0f;
	if (useAction && !shearStates)
		return 0.0f;

	std::vector<float> energy;
	std::vector<float> balance;
	std::vector<float> action;
	if (useEnergy)
		energy.assign(static_cast<size_t>(nStates) * nBuckets, 0.0f);
	if (useBalance)
		balance.assign(static_cast<size_t>(nStates) * nBuckets, 0.0f);
	if (useAction)
		action.assign(static_cast<size_t>(nTransitions) * nBuckets, 0.0f);

	const float safeEps = (eps > 0.0f) ? eps : 1e-12f;
	const size_t stateStride = static_cast<size_t>(T) * static_cast<size_t>(m);

	if (useEnergy || useBalance)
	{
		std::vector<float> sigmaP2(nStates, 0.0f);
		std::vector<float> sigmaQ2(nStates, 0.0f);
		for (unsigned int l = 0u; l < nStates; ++l)
		{
			double p2 = 0.0;
			double q2 = 0.0;
			const float* p = pStates + static_cast<size_t>(l) * stateStride;
			const float* q = qStates + static_cast<size_t>(l) * stateStride;
			for (size_t i = 0u; i < stateStride; ++i)
			{
				p2 += static_cast<double>(p[i]) * static_cast<double>(p[i]);
				q2 += static_cast<double>(q[i]) * static_cast<double>(q[i]);
			}
			const double denom = (stateStride > 0u) ? static_cast<double>(stateStride) : 1.0;
			sigmaP2[l] = sira_safe_positive(static_cast<float>(p2 / denom), safeEps);
			sigmaQ2[l] = sira_safe_positive(static_cast<float>(q2 / denom), safeEps);
		}

		for (unsigned int l = 0u; l < nStates; ++l)
		{
			const float* p = pStates + static_cast<size_t>(l) * stateStride;
			const float* q = qStates + static_cast<size_t>(l) * stateStride;
			for (unsigned int b = 0u; b < nBuckets; ++b)
			{
				const unsigned int tb = static_cast<unsigned int>((static_cast<size_t>(b) * T) / nBuckets);
				const unsigned int te = static_cast<unsigned int>((static_cast<size_t>(b + 1u) * T) / nBuckets);
				if (te <= tb)
					continue;
				double p2 = 0.0;
				double q2 = 0.0;
				for (unsigned int t = tb; t < te; ++t)
				{
					const size_t row = static_cast<size_t>(t) * m;
					for (unsigned int j = 0u; j < m; ++j)
					{
						const float pv = p[row + j];
						const float qv = q[row + j];
						p2 += static_cast<double>(pv) * static_cast<double>(pv);
						q2 += static_cast<double>(qv) * static_cast<double>(qv);
					}
				}
				const double count = static_cast<double>(te - tb) * static_cast<double>(m);
				const float pNorm = sira_safe_positive(static_cast<float>(p2 / (count * sigmaP2[l])), safeEps);
				const float qNorm = sira_safe_positive(static_cast<float>(q2 / (count * sigmaQ2[l])), safeEps);
				const size_t idx = static_cast<size_t>(l) * nBuckets + b;
				if (useEnergy)
					energy[idx] = 0.5f * (pNorm + qNorm);
				if (useBalance)
					balance[idx] = 0.5f * (logf(pNorm) - logf(qNorm));
			}
		}
	}

	if (useAction)
	{
		for (unsigned int l = 0u; l < nTransitions; ++l)
		{
			const float* p0 = pStates + static_cast<size_t>(l) * stateStride;
			const float* q0 = qStates + static_cast<size_t>(l) * stateStride;
			const float* q1 = qStates + static_cast<size_t>(l + 1u) * stateStride;
			const float* sh = shearStates + static_cast<size_t>(l) * stateStride;
			for (unsigned int b = 0u; b < nBuckets; ++b)
			{
				const unsigned int tb = static_cast<unsigned int>((static_cast<size_t>(b) * T) / nBuckets);
				const unsigned int te = static_cast<unsigned int>((static_cast<size_t>(b + 1u) * T) / nBuckets);
				if (te <= tb)
					continue;
				double dot = 0.0;
				double pbar2 = 0.0;
				double dq2 = 0.0;
				for (unsigned int t = tb; t < te; ++t)
				{
					const size_t row = static_cast<size_t>(t) * m;
					for (unsigned int j = 0u; j < m; ++j)
					{
						const float pbar = p0[row + j] + 0.5f * sh[row + j];
						const float dq = q1[row + j] - q0[row + j];
						dot += static_cast<double>(pbar) * static_cast<double>(dq);
						pbar2 += static_cast<double>(pbar) * static_cast<double>(pbar);
						dq2 += static_cast<double>(dq) * static_cast<double>(dq);
					}
				}
				const float denom = sqrtf(static_cast<float>(pbar2 * dq2)) + safeEps;
				action[static_cast<size_t>(l) * nBuckets + b] = (denom > safeEps)
				    ? static_cast<float>(dot) / denom : 0.0f;
			}
		}
	}

	return sira_loss_from_terms(useEnergy ? &energy[0] : NULL,
	                            useBalance ? &balance[0] : NULL,
	                            useAction ? &action[0] : NULL,
	                            nStates, nBuckets, coef,
	                            useEnergy ? energyWeight : 0.0f,
	                            useBalance ? balanceWeight : 0.0f,
	                            useAction ? actionWeight : 0.0f,
	                            huberTau, safeEps);
}

} // namespace chiron
} // namespace glades
