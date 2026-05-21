// SFA (Sheaf-Focal Attention) CPU prototype primitives — Gate-0 validation only.
//
// Paradigm #250 from research/PARADIGM_SHIFT_250_DESIGN.md.
// Mathematical specification in research/PARADIGM_SHIFT_250_PROOFS.md.
// Implementation plan in research/CSA_GATE0_IMPLEMENTATION_PLAN.md.
//
// This is a CPU-only reference implementation sufficient for Gate-0 probes A-J on
// the existing flagship checkpoint chiron_1B_T16384.step30000. Production CUDA
// kernels are deferred to Phase 2 of the paradigm #250 roadmap.
//
// Conventions (match transformer_ops.h):
// - Matrices flattened row-major.
// - Stalk frames U_i flattened as i*d_s*r + alpha*r + beta.
// - Section s flattened as i*d_s + alpha.
// - Edges stored as parallel arrays (edge_src[e], edge_tgt[e]).
//
// Determinism: all operations are deterministic given inputs. No atomics.
// BF16 is NOT used here; CPU prototype operates in FP32 for clarity. GPU port
// will introduce BF16 with FP32 accumulation per the iter-2 §7.5 analysis.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

// Note: the CPU prototype is scalar — no SIMD helpers needed.
// GPU port (Phase 2) will pull in transformer_kernels.h for fused axpy / dot.

namespace glades
{
namespace transformer_sfa_ops
{

// Per-layer SFA parameters. CPU prototype storage; GPU port will replace
// std::vector with device-side buffers managed by gpu_chiron.
//
// Memory layout:
//   U[T][d_s][r]: stalk frames, U[i*d_s*r + alpha*r + beta]
//   Sigma[|E|][r]: diagonal edge modulators
//   edge_src[|E|], edge_tgt[|E|]: parallel arrays of (i, j) edges with i <= j
//   P_q[d_s][d_h], P_v[d_s][d_h], P_o[d_h][d_s]: stalk injection / readout
//   lambda: Tikhonov regulariser (positive). Default 1e-2.
//   gamma: value injection scale. Default 0 (no value bypass at init).
struct SFAParams
{
	int T;            // sequence length
	int d_s;          // stalk dimension
	int d_h;          // per-head dimension (target representation)
	int r;            // stalk-frame rank
	int W;            // sliding-window half-width
	int n_sinks;      // number of sink vertices (first n_sinks positions)

	std::vector<float> U;          // [T * d_s * r]
	std::vector<float> Sigma;      // [|E| * r]
	std::vector<int> edge_src;     // [|E|]
	std::vector<int> edge_tgt;     // [|E|]

	std::vector<float> P_q;        // [d_s * d_h]
	std::vector<float> P_v;        // [d_s * d_h]
	std::vector<float> P_o;        // [d_h * d_s]

	float lambda;                  // Tikhonov regulariser
	float gamma;                   // value injection scale
};

// Build causal-sliding-window-plus-sinks edge set:
//   E = { (i, j) : i <= j AND (|i - j| <= W OR i in {0, ..., n_sinks - 1}) }
// Edges are appended to edge_src / edge_tgt in deterministic order (sink edges
// first, then window edges per source vertex).
inline void buildEdgeSet(int T, int W, int n_sinks,
                          std::vector<int>& edge_src,
                          std::vector<int>& edge_tgt)
{
	edge_src.clear();
	edge_tgt.clear();
	// Sink edges first: each sink i connects to all j >= i.
	for (int i = 0; i < n_sinks && i < T; ++i)
	{
		for (int j = i; j < T; ++j)
		{
			edge_src.push_back(i);
			edge_tgt.push_back(j);
		}
	}
	// Window edges: for each non-sink i, add edges to j in [i, min(T-1, i+W)].
	for (int i = n_sinks; i < T; ++i)
	{
		const int j_max = std::min(T - 1, i + W);
		for (int j = i; j <= j_max; ++j)
		{
			edge_src.push_back(i);
			edge_tgt.push_back(j);
		}
	}
}

// Compute (R_{j <- i} * s_i) for one edge in a temporary buffer.
//   R_{j <- i} = U_j * diag(Sigma(i, j)) * U_i^T
//   Action: out = U_j (Sigma_e .* (U_i^T s_i))
// All vectors are length d_s; intermediate length r.
inline void applyRestrictionEdge(const float* U_i, const float* U_j,
                                  const float* Sigma_e,
                                  const float* s_i,
                                  int d_s, int r,
                                  float* tmp_r,
                                  float* out_j)
{
	// tmp_r = U_i^T s_i   (length r)
	for (int beta = 0; beta < r; ++beta)
	{
		float acc = 0.0f;
		for (int alpha = 0; alpha < d_s; ++alpha)
		{
			acc += U_i[alpha * r + beta] * s_i[alpha];
		}
		tmp_r[beta] = acc * Sigma_e[beta];
	}
	// out_j = U_j * tmp_r   (length d_s)
	for (int alpha = 0; alpha < d_s; ++alpha)
	{
		float acc = 0.0f;
		for (int beta = 0; beta < r; ++beta)
		{
			acc += U_j[alpha * r + beta] * tmp_r[beta];
		}
		out_j[alpha] = acc;
	}
}

// Apply L_F to a section s. Result placed in `out`.
//
// L_F = delta^T delta, with delta_e(s) = s_j - R_{j <- i} s_i for edge e = (i, j).
// Block structure:
//   [L_F]_{ii} = (out-deg_i + in-deg_i) * I  +  sum_{j: (j, i) in E} R_{i <- j}^T R_{i <- j}
//   [L_F]_{ij} = -R_{j <- i}^T   for (i, j) in E
//
// CPU prototype: O(|E| * d_s * r) FLOPs. Deterministic single-thread reduction.
inline void laplacianMatvec(const SFAParams& p,
                             const float* s,
                             float* out)
{
	const int T = p.T;
	const int d_s = p.d_s;
	const int r = p.r;
	const int E = static_cast<int>(p.edge_src.size());

	// Zero accumulator.
	std::memset(out, 0, sizeof(float) * static_cast<size_t>(T) * d_s);

	// Per-edge contribution. We compute (delta_e s) = s_j - R_{j <- i} s_i, then
	// accumulate (delta_e s) into out_j and -(R_{j <- i}^T (delta_e s)) into out_i.
	//
	// This is the standard delta^T delta computation realised edge-by-edge without
	// materialising the |E|*d_s C^1 cochain space.
	std::vector<float> tmp_r(r);
	std::vector<float> Rs(d_s);          // R_{j <- i} s_i
	std::vector<float> delta(d_s);       // s_j - Rs
	std::vector<float> Rt_delta(d_s);    // R_{j <- i}^T delta

	for (int e = 0; e < E; ++e)
	{
		const int i = p.edge_src[e];
		const int j = p.edge_tgt[e];
		const float* U_i = &p.U[static_cast<size_t>(i) * d_s * r];
		const float* U_j = &p.U[static_cast<size_t>(j) * d_s * r];
		const float* Sigma_e = &p.Sigma[static_cast<size_t>(e) * r];
		const float* s_i = &s[static_cast<size_t>(i) * d_s];
		const float* s_j = &s[static_cast<size_t>(j) * d_s];

		// Rs = R_{j <- i} s_i
		applyRestrictionEdge(U_i, U_j, Sigma_e, s_i, d_s, r, &tmp_r[0], &Rs[0]);

		// delta = s_j - Rs
		for (int a = 0; a < d_s; ++a)
		{
			delta[a] = s_j[a] - Rs[a];
		}

		// Contribution to out_j: += delta
		float* out_j = &out[static_cast<size_t>(j) * d_s];
		for (int a = 0; a < d_s; ++a)
		{
			out_j[a] += delta[a];
		}

		// Contribution to out_i: -= R_{j <- i}^T delta
		// R^T action: tmp_r[beta] = sum_a U_j[a, beta] * delta[a]; mult by Sigma; lift by U_i.
		for (int beta = 0; beta < r; ++beta)
		{
			float acc = 0.0f;
			for (int a = 0; a < d_s; ++a)
			{
				acc += U_j[a * r + beta] * delta[a];
			}
			tmp_r[beta] = acc * Sigma_e[beta];
		}
		for (int a = 0; a < d_s; ++a)
		{
			float acc = 0.0f;
			for (int beta = 0; beta < r; ++beta)
			{
				acc += U_i[a * r + beta] * tmp_r[beta];
			}
			Rt_delta[a] = acc;
		}
		float* out_i = &out[static_cast<size_t>(i) * d_s];
		for (int a = 0; a < d_s; ++a)
		{
			out_i[a] -= Rt_delta[a];
		}
	}
}

// Estimate mu_max = ||L_F||_op via power iteration. Deterministic given seed.
//
// Default: n_iters = 2 plus 10% safety margin (see PARADIGM_SHIFT_250_DESIGN.md §5.3).
inline float estimateMuMax(const SFAParams& p,
                            int n_iters,
                            unsigned int seed)
{
	const int Tds = p.T * p.d_s;
	std::vector<float> v(Tds);
	std::vector<float> Lv(Tds);

	// Deterministic init: pseudo-random based on seed.
	unsigned int state = seed ? seed : 0xC0FFEEu;
	for (int k = 0; k < Tds; ++k)
	{
		state = state * 1664525u + 1013904223u;
		v[k] = ((state >> 16) & 0xFFFFu) / 65535.0f - 0.5f;
	}

	// Normalise.
	float norm = 0.0f;
	for (int k = 0; k < Tds; ++k)
		norm += v[k] * v[k];
	norm = std::sqrt(norm);
	const float inv = 1.0f / std::max(norm, 1e-12f);
	for (int k = 0; k < Tds; ++k)
		v[k] *= inv;

	// Power iteration.
	float lambda_est = 0.0f;
	for (int it = 0; it < n_iters; ++it)
	{
		laplacianMatvec(p, &v[0], &Lv[0]);
		float dot = 0.0f;
		float Lnorm = 0.0f;
		for (int k = 0; k < Tds; ++k)
		{
			dot += v[k] * Lv[k];
			Lnorm += Lv[k] * Lv[k];
		}
		Lnorm = std::sqrt(Lnorm);
		lambda_est = dot;
		const float inv2 = 1.0f / std::max(Lnorm, 1e-12f);
		for (int k = 0; k < Tds; ++k)
			v[k] = Lv[k] * inv2;
	}

	return lambda_est * 1.10f;  // 10% safety margin
}

// Source assembly: b_i = U_i U_i^T P_q W_Q x_i + gamma * P_v W_V x_i
//
// Inputs:
//   x[T][m]: residual stream
//   W_Q[m][d_h], W_V[m][d_h]: standard projections (NOT scratch-allocated here;
//   we pass the already-computed q[T][d_h] = x*W_Q and v[T][d_h] = x*W_V).
//   p.P_q[d_s][d_h], p.P_v[d_s][d_h]: stalk injection maps.
//
// Output: b[T][d_s].
inline void sourceAssembly(const SFAParams& p,
                            const float* q,        // [T * d_h]
                            const float* v,        // [T * d_h]
                            float* b)              // [T * d_s]
{
	const int T = p.T;
	const int d_s = p.d_s;
	const int d_h = p.d_h;
	const int r = p.r;

	for (int i = 0; i < T; ++i)
	{
		const float* U_i = &p.U[static_cast<size_t>(i) * d_s * r];
		const float* q_i = &q[static_cast<size_t>(i) * d_h];
		const float* v_i = &v[static_cast<size_t>(i) * d_h];
		float* b_i = &b[static_cast<size_t>(i) * d_s];

		// Step 1: lifted_q = P_q * q_i  (length d_s)
		std::vector<float> lifted_q(d_s, 0.0f);
		for (int a = 0; a < d_s; ++a)
		{
			float acc = 0.0f;
			for (int h = 0; h < d_h; ++h)
			{
				acc += p.P_q[a * d_h + h] * q_i[h];
			}
			lifted_q[a] = acc;
		}

		// Step 2: project onto U_i U_i^T column space  (length d_s)
		// projected = U_i (U_i^T lifted_q)
		std::vector<float> tmp_r(r, 0.0f);
		for (int beta = 0; beta < r; ++beta)
		{
			float acc = 0.0f;
			for (int a = 0; a < d_s; ++a)
			{
				acc += U_i[a * r + beta] * lifted_q[a];
			}
			tmp_r[beta] = acc;
		}
		for (int a = 0; a < d_s; ++a)
		{
			float acc = 0.0f;
			for (int beta = 0; beta < r; ++beta)
			{
				acc += U_i[a * r + beta] * tmp_r[beta];
			}
			b_i[a] = acc;
		}

		// Step 3: add gamma * P_v * v_i
		if (p.gamma != 0.0f)
		{
			for (int a = 0; a < d_s; ++a)
			{
				float acc = 0.0f;
				for (int h = 0; h < d_h; ++h)
				{
					acc += p.P_v[a * d_h + h] * v_i[h];
				}
				b_i[a] += p.gamma * acc;
			}
		}
	}
}

// Readout: y_i = P_o^T s_i  (length d_h)
// The "identity-attention" residual W_Q x_i is added by the caller in the
// CHIRON shear context.
inline void readout(const SFAParams& p,
                     const float* s,     // [T * d_s]
                     float* y)           // [T * d_h]
{
	const int T = p.T;
	const int d_s = p.d_s;
	const int d_h = p.d_h;

	for (int i = 0; i < T; ++i)
	{
		const float* s_i = &s[static_cast<size_t>(i) * d_s];
		float* y_i = &y[static_cast<size_t>(i) * d_h];
		for (int h = 0; h < d_h; ++h)
		{
			float acc = 0.0f;
			for (int a = 0; a < d_s; ++a)
			{
				// P_o is stored [d_h][d_s], i.e., P_o[h*d_s + a].
				acc += p.P_o[h * d_s + a] * s_i[a];
			}
			y_i[h] = acc;
		}
	}
}

}  // namespace transformer_sfa_ops
}  // namespace glades
