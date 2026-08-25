// Custom CUDA kernel declarations for Glades ML.
//
// Provides host wrapper functions for normalization, activation, attention,
// embedding, optimizer, and utility kernels.  When GLADES_HAVE_CUDA is not
// defined the wrappers degrade to inline no-ops / stubs that return false.
#pragma once

#include <cstddef>
#include <stdint.h>
#include "gpu_device.h"  // cudaStream_t (iter 8)

#ifdef GLADES_HAVE_CUDA

namespace glades {
namespace gpu {

// ---------------------------------------------------------------------------
// Layer normalization
// ---------------------------------------------------------------------------

// Forward: out[rows,cols] = gamma * (x - mean) * invStd + beta
// mean[rows] and invStd[rows] are written as side-outputs.
bool layernorm_forward(const float* x, const float* gamma, const float* beta,
                       float eps, int rows, int cols,
                       float* out, float* mean, float* invStd);

// Backward: computes dx, and *accumulates* into dgamma / dbeta.
bool layernorm_backward(const float* dout, const float* x,
                        const float* gamma, const float* mean,
                        const float* invStd, int rows, int cols,
                        float* dx, float* dgamma, float* dbeta);

// Phase 3 (q-side source cure): layernorm backward with xhat=(x-mean)*invStd
// clamped to [-xhatMax, xhatMax] in the dgamma/dbeta reduction — bounds the
// drift-driven dgamma overflow at its source.  xhatMax<=0 = plain backward.
// See docs/superpowers/plans/2026-06-16-chiron-stability-techniques.md.
bool layernorm_backward_bounded(const float* dout, const float* x,
                                const float* gamma, const float* mean,
                                const float* invStd, int rows, int cols,
                                float* dx, float* dgamma, float* dbeta,
                                float xhatMax);

// ---------------------------------------------------------------------------
// RMSNorm (LLaMA-style)
// ---------------------------------------------------------------------------

bool rmsnorm_forward(const float* x, const float* gamma, float eps,
                     int rows, int cols, float* out, float* invRms);

bool rmsnorm_backward(const float* dout, const float* x,
                      const float* gamma, const float* invRms,
                      int rows, int cols,
                      float* dx, float* dgamma);

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------

// Numerically-stable row-wise softmax.
bool softmax_forward(const float* x, int rows, int cols, float* out);

// Same as softmax_forward but also writes logZ[rows] = log(sum_i exp(x[row,i])).
// Required by the Z-loss backward path — logZ must be a device buffer of
// length rows.
bool softmax_forward_with_lse(const float* x, int rows, int cols,
                               float* out, float* logZ);

// Standard softmax-CE backward. The trainer dispatches this when
// zlossCoef == 0.0f; otherwise it dispatches softmax_cross_entropy_bwd_zloss.
// Both produce bit-identical output at zlossCoef == 0.0f.
bool softmax_cross_entropy_bwd(const float* probs, const int* targets,
                               int rows, int cols, float* dlogits);

// Z-loss-aware variant: adds (2 * zlossCoef * logZ[row]) * probs[row, i]
// on top of the standard CE gradient. At zlossCoef == 0.0f the output is
// bit-identical to softmax_cross_entropy_bwd.
bool softmax_cross_entropy_bwd_zloss(const float* probs, const int* targets,
                                     const float* logZ, float zlossCoef,
                                     int rows, int cols, float* dlogits);

// ralph-loop iter 10 (2026-05-14) BF16-storage variants — backing the
// --bf16-logits-storage flag.  Same math as the FP32 paths, BF16 on
// load/store (uint16_t bit-pattern), FP32 in registers.  Used to
// materialize (T × V) logits/probs/dlogits in BF16 so T=16384 fits on
// 16 GB hardware (saves 3 × 2 GB = 3 GB net).
bool softmax_forward_bf16(const unsigned short* x, int rows, int cols,
                           unsigned short* out);
// BF16-storage variant of softmax_forward_with_lse: also writes per-row
// logsumexp into a FP32 logZ buffer.  Required by the Z-loss path on the
// --bf16-logits-storage trainer recipe.
bool softmax_forward_bf16_with_lse(const unsigned short* x, int rows, int cols,
                                    unsigned short* out, float* logZ);
bool softmax_cross_entropy_bwd_bf16(const unsigned short* probs,
                                     const int* targets,
                                     int rows, int cols,
                                     unsigned short* dlogits);
// BF16-storage Z-loss CE backward: dlogits[t, v] = (probs[t, v] - 1_{v==target})
// + 2·zlossCoef·logZ[t]·probs[t, v].  Required on --bf16-logits-storage.
bool softmax_cross_entropy_bwd_bf16_zloss(const unsigned short* probs,
                                           const int* targets,
                                           const float* logZ,
                                           float zlossCoef,
                                           int rows, int cols,
                                           unsigned short* dlogits);
// ECHO — Excess-Copy Hinged Objective (2026-07-09,
// docs/superpowers/specs/2026-07-09-chiron-loss-regularizers-design.md §5).
// Semantics contract + bit-exact CPU references: chiron_echo_*_cpu in
// transformer_chiron_ops.h.  Training-only readout regularizer; the trainer
// dispatches none of these at --echo-coef 0 (E0 discipline).
// Per-row excess-copy stats over the trailing w-token window: PA[T] (dense
// gradient mass), Rrow[T] (hinge loss terms), activeBits[T*ceil(w/32)] (owner
// window-slot bitmap), activeCount[T].  Keeping owner slots rather than copied
// vocabulary ids cuts hard-mode scratch from O(T*w) ints to O(T*w/32) words;
// scatter recovers each id from tokens.  blockDim == w; w in [1, 1024].  The
// legacy entry point is the exact hard hinge.  The Huber entry point implements
// h_delta(x)=x^2/(2delta) for 0<x<delta and x-delta/2 for x>=delta;
// activeWeights[T*w] stores h'_delta(x) at the original owner slot (only bits
// marked active are valid), PA=sum(p*h'), and maxActiveProb is an optional
// per-row diagnostic.
bool echo_repeat_stats(const unsigned short* probs, const int* tokens,
                       const int* targets, int T, int V, int w,
                       float kappa, float tau0,
                       float* PA, float* Rrow, uint32_t* activeBits,
                       int* activeCount);
bool echo_repeat_stats_huber(const unsigned short* probs, const int* tokens,
                             const int* targets, int T, int V, int w,
                             float kappa, float tau0, float huberDelta,
                             float* PA, float* Rrow, uint32_t* activeBits,
                             float* activeWeights, int* activeCount,
                             float* maxActiveProb);
// Compact GPU telemetry summary, avoiding four O(T) device-to-host copies per
// training step.  All fields are floats; count fields are exact for supported
// T/w.  maxActiveProb uses max reduction while all other fields use sum.
enum EchoSummaryIndex
{
	ECHO_SUM_R = 0,
	ECHO_SUM_PA,
	ECHO_SUM_PMAX,
	ECHO_MAX_P,
	ECHO_ACTIVE_ROWS,
	ECHO_ACTIVE_IDS,
	ECHO_HIST_0,
	ECHO_HIST_1,
	ECHO_HIST_2,
	ECHO_HIST_3,
	ECHO_HIST_4,
	ECHO_SUMMARY_SIZE
};
bool echo_summarize_stats(const float* Rrow, const float* PA,
                          const float* maxActiveProb, const int* activeCount,
                          int T, float* summary);
// Shipped zloss CE backward + the dense ECHO term (-echoCoef*PA[t])*probs;
// bit-identical to softmax_cross_entropy_bwd_bf16_zloss at echoCoef == 0.
bool softmax_cross_entropy_bwd_bf16_zloss_echo(const unsigned short* probs,
                                                const int* targets,
                                                const float* logZ,
                                                float zlossCoef,
                                                float echoCoef,
                                                const float* PA,
                                                int rows, int cols,
                                                unsigned short* dlogits);
// Dense term of the standalone ECHO-only backward used by detached gradient
// attribution probes: dlogits[t,v] = -echoCoef*PA[t]*probs[t,v].  Follow with
// echo_scatter_bf16[_weighted] to complete the exact ECHO field.
bool echo_dense_bwd_bf16(const unsigned short* probs, float echoCoef,
                          const float* PA, int rows, int cols,
                          unsigned short* dlogits);
// Sparse ECHO scatter.  Hard hinge adds echoCoef*probs[t,id]; the weighted
// form additionally multiplies by activeWeights[t,ownerSlot]=h'_delta(p-margin).
// tokens + the owner-slot bitmap recover ids without a dense active-id buffer.
bool echo_scatter_bf16(const unsigned short* probs, const int* tokens,
                       const uint32_t* activeBits, float echoCoef,
                       int rows, int cols, int w, unsigned short* dlogits);
bool echo_scatter_bf16_weighted(const unsigned short* probs,
                                const int* tokens,
                                const uint32_t* activeBits,
                                const float* activeWeights, float echoCoef,
                                int rows, int cols, int w,
                                unsigned short* dlogits);
bool scale_array_bf16(unsigned short* x, float scale, int n);

// Contextual Rank Margin (CRM), training-only and additive. Both entry points
// add the weighted, unnormalized CRM field to caller-owned dlogits and emit
// row-major stats matching glades::chiron::ChironCrmRowStatIndex. A zero
// coefficient is a host-side no-op: no kernel launch and no output mutation.
bool chiron_crm_forward_backward(const float* logits, const int* targets,
                                 float coefficient, float margin,
                                 float temperature, int rows, int cols,
                                 float* dlogits, float* rowStats);
bool chiron_crm_forward_backward_bf16(const unsigned short* logits,
                                      const int* targets,
                                      float coefficient, float margin,
                                      float temperature, int rows, int cols,
                                      unsigned short* dlogits, float* rowStats);
// Frozen-step diagnostic variant. probs supplies the exact stored CE field;
// hardNegativeMass, when non-null and pre-zeroed, receives unit mass per row
// shared uniformly over the exact maximum set. No second optimizer pass runs.
bool chiron_crm_forward_backward_bf16_observed(const unsigned short* logits,
                                               const unsigned short* probs,
                                               const int* targets,
                                               float coefficient, float margin,
                                               float temperature, int rows, int cols,
                                               unsigned short* dlogits,
                                               float* rowStats,
                                               float* hardNegativeMass);

bool cross_entropy_nll_loss_bf16(const unsigned short* probs,
                                  const int* targets,
                                  int T, int vocabSize, int padToken,
                                  float* loss_sum, int* valid_count);
bool argmax_count_matches_bf16(const unsigned short* probs,
                                const int* targets,
                                int T, int vocabSize, int padToken,
                                int* correct_count, int* valid_count);
// V10/V16 read-only output vectors, fused into the existing CE/argmax scans.
bool chiron_vitals_output_vectors_bf16(const unsigned short* probs,
                                        const int* targets,
                                        int T, int vocabSize, int padToken,
                                        float* loss_sum, int* loss_count,
                                        int* correct_count, int* valid_count,
                                        float* perTokenNll,
                                        unsigned char* perTokenTop1);

// 2026-05-14 live-eval suite — position-bucketed NLL + top-k accuracy.
// Both operate on BF16 probs (the storage type used at runtime when
// --bf16-logits-storage is active).
//
// Position-bucketed NLL: bucket b ∈ [0, numBuckets) covers positions
// [b*T/numBuckets, (b+1)*T/numBuckets).  Reports loss_sum[b] and
// valid_count[b] per bucket.  Used by --val-every to track NLL as a
// function of position-in-context (directly tests SCFA long-context).
bool cross_entropy_nll_bucketed_bf16(const unsigned short* probs,
                                      const int* targets,
                                      int T, int vocabSize, int padToken,
                                      int numBuckets,
                                      float* loss_sum, int* valid_count);

// Top-k accuracy: for each row, target is "correct" at k iff it's among
// the top-k highest-probability tokens.  k_values is a device buffer of
// numK ints (sorted ascending for clarity).  correct_counts is parallel
// device array; valid_count is shared across all k (target validity is
// k-independent).
bool topk_accuracy_bf16(const unsigned short* probs, const int* targets,
                         int T, int vocabSize, int padToken,
                         int numK, const int* k_values,
                         int* correct_counts, int* valid_count);

// Paradigm shift #56 DISTILL-FORWARD — combined KL + CE backward:
//   dlogits = probs_student - alpha · probs_teacher - (1 - alpha) · one_hot(targets)
// Teacher distribution is frozen (no grad).
bool distill_combined_bwd(const float* probs_student, const float* probs_teacher,
                          const int* targets, int rows, int cols, float alpha,
                          float* dlogits);

// Paradigm shift #56 DISTILL-FORWARD — combined KL + CE scalar loss for logging.
// L = α · KL(p_T || p_S) + (1 - α) · CE(p_S, target), averaged over valid rows.
// padToken < 0 disables padding skip; otherwise rows with target == padToken
// (or out-of-vocab) are skipped.
bool distill_combined_loss(const float* probs_student, const float* probs_teacher,
                           const int* targets,
                           int T, int vocabSize, int padToken, float alpha,
                           float* loss_sum, int* valid_count);

// ---------------------------------------------------------------------------
// Paradigm shift #43 ORION — Galerkin model-order reduction primitives.
// V is stored column-major as a flat [n × r] buffer: V[i, k] = V_flat[k*n + i].
// V uses BF16 storage exposed as uint16_t in the public API (the .cu side
// reinterpret-casts to __nv_bfloat16).  All kernels operate on a single
// parameter tensor of size n with rank-r basis.
// ---------------------------------------------------------------------------

// α_out[k] := Σ_i V[i, k] · g[i]   (zero-initialized internally).
bool orion_proj_left(const uint16_t* V, const float* g,
                     int n, int r, float* alpha_out);

// θ[i] += Σ_k V[i, k] · α[k]
bool orion_lift_add(float* theta, const uint16_t* V,
                    const float* alpha, int n, int r);

// BF16-weights variant: θ_bf16[i] = bf16(θ_anchor_fp32[i] + Σ_k V[i, k] · α[k])
bool orion_lift_add_bf16w(uint16_t* theta_bf16, const float* theta_anchor,
                          const uint16_t* V,
                          const float* alpha, int n, int r);

// θ_pert[i] = θ[i] + eps · V[i, col]   (for FD-HVP).
bool orion_perturb_col(float* theta_pert, const float* theta,
                       const uint16_t* V, int n, int col, float eps);

// BF16-weights variant: θ_bf16[i] = bf16(θ_anchor_fp32[i] + eps · V[i, col]).
bool orion_perturb_col_bf16w(uint16_t* theta_bf16, const float* theta_anchor,
                             const uint16_t* V, int n, int col, float eps);

// Oja's tilt: V += η · g_⊥ · (V^⊤g)^⊤  (subspace tilt toward gradient
// direction not yet in span(V)).  g_proj = V^⊤ g must be pre-computed.
bool orion_oja_tilt(uint16_t* V, const float* g, const float* g_proj,
                    int n, int r, float eta);

// Modified Gram-Schmidt orthonormalization of V columns IN PLACE.
// Caller supplies two single-float device scratches.  At r ≤ 8 the host-side
// outer loop is negligible; only r(r+1)/2 single-row dot/subtract/normalize
// kernels touch n.
bool orion_gram_schmidt(uint16_t* V, int n, int r,
                        float* scratch_dot, float* scratch_normsq);

// Phase-4 BF16-anchor variants: anchor is BF16-stored, master is BF16.
// θ_bf16[i] = bf16(__bfloat162float(anchor_bf16[i]) + eps · V[i, col])
bool orion_perturb_col_bf16w_bf16anchor(uint16_t* theta_bf16,
                                        const uint16_t* theta_anchor_bf16,
                                        const uint16_t* V,
                                        int n, int col, float eps);
// θ_bf16[i] = bf16(__bfloat162float(anchor_bf16[i]) + Σ_k V[i,k] · α[k])
bool orion_lift_add_bf16w_bf16anchor(uint16_t* theta_bf16,
                                     const uint16_t* theta_anchor_bf16,
                                     const uint16_t* V,
                                     const float* alpha, int n, int r);
// α[k] += Σ_i V[i, k] · __bfloat162float(g_bf16[i])
bool orion_proj_left_bf16_src(const uint16_t* V, const uint16_t* g_bf16,
                              int n, int r, float* alpha_out);

// Phase-3 INT8 V kernels.  V is stored column-major as int8_t with per-block
// FP32 scales (block size 256).  Total VRAM ≈ 0.508 × BF16 V at all r ≤ 8.
int  orion_v_scale_count(int n);
bool orion_v_quantize_int8_column(const float* src, int8_t* dst_q,
                                   float* scales, int n);
bool orion_v_dequantize_int8_column(const int8_t* src_q, const float* scales,
                                     float* dst, int n);
bool orion_proj_left_int8(const int8_t* V_q, const float* V_scales,
                           const float* g, int n, int r, float* alpha_out);
bool orion_proj_left_int8_bf16src(const int8_t* V_q, const float* V_scales,
                                   const uint16_t* g_bf16,
                                   int n, int r, float* alpha_out);
bool orion_lift_add_int8(float* theta, const int8_t* V_q, const float* V_scales,
                          const float* alpha, int n, int r);
bool orion_lift_add_int8_bf16w(uint16_t* theta_bf16, const float* theta_anchor,
                                const int8_t* V_q, const float* V_scales,
                                const float* alpha, int n, int r);
bool orion_lift_add_int8_bf16w_bf16anchor(uint16_t* theta_bf16,
                                           const uint16_t* theta_anchor_bf16,
                                           const int8_t* V_q,
                                           const float* V_scales,
                                           const float* alpha, int n, int r);
bool orion_perturb_col_int8(float* theta_pert, const float* theta,
                             const int8_t* V_q, const float* V_scales,
                             int n, int col, float eps);
bool orion_perturb_col_int8_bf16w(uint16_t* theta_bf16, const float* theta_anchor,
                                   const int8_t* V_q, const float* V_scales,
                                   int n, int col, float eps);
bool orion_perturb_col_int8_bf16w_bf16anchor(uint16_t* theta_bf16,
                                              const uint16_t* theta_anchor_bf16,
                                              const int8_t* V_q,
                                              const float* V_scales,
                                              int n, int col, float eps);

// ---------------------------------------------------------------------------
// Paradigm shift #42 SCFA — Spectral Compressed Flow Attention primitives.
//
// The original global DCT projection B B^T was not token-causal: an output at
// position t changed when tokens after t changed.  The causal path partitions
// T positions into k contiguous blocks.  `block_compress` summarizes each
// block; `causal_lag_lift` exposes summary b only to block b+1.  The matching
// transpose operators keep trainer backward exact while preserving O(T*m)
// outer work and exact prefix invariance.
// ---------------------------------------------------------------------------

// out[b,c] = alpha * sum_{t in block(b)} x[t,c] / sqrt(|block(b)|)
//          + beta * out[b,c].
bool scfa_block_compress(const float* x, int T, int m, int k,
                         float alpha, float beta, float* out,
                         cudaStream_t stream = 0);

// out[t,c] = alpha * x[block(t)-1,c] / sqrt(|block(t)-1|)
//          + beta * out[t,c], or beta*out for the first block.
bool scfa_causal_lag_lift(const float* x, int T, int m, int k,
                          float alpha, float beta, float* out,
                          cudaStream_t stream = 0);

// Transpose of scfa_causal_lag_lift:
// out[b,c] = alpha * sum_{t in block(b+1)} x[t,c] / sqrt(|block(b)|)
//          + beta * out[b,c].
bool scfa_causal_lag_reduce(const float* x, int T, int m, int k,
                            float alpha, float beta, float* out,
                            cudaStream_t stream = 0);

// Transpose of scfa_block_compress:
// out[t,c] = alpha * x[block(t),c] / sqrt(|block(t)|)
//          + beta * out[t,c].
bool scfa_block_expand(const float* x, int T, int m, int k,
                       float alpha, float beta, float* out,
                       cudaStream_t stream = 0);

// Decode helper for the lag lift of one completed block. Uses device rsqrtf,
// matching scfa_causal_lag_lift rather than a host-rounded scale constant:
// out[c] = summary[c] / sqrt(blockWidth).
bool scfa_lag_row(const float* summary, int m, int blockWidth, float* out,
                  cudaStream_t stream = 0);

// y[t, c] = Σ_{i=0..w} K[c, i] · x[t-i, c]    (causal depthwise 1-D conv).
// One filter per channel; m channels, T positions, half-width w (kernel size
// w+1 since we only use the causal half + center tap).  Out-of-bounds is 0.
// iter 8 (2026-05-14): optional `stream` parameter for multi-stream branch
// parallelism (--scfa-parallel-branches).  Default = 0 means use
// computeStream() (backwards compatible).
bool scfa_depthwise_causal_conv_fwd(const float* x, const float* K,
                                     int T, int m, int w, float* y,
                                     cudaStream_t stream = 0);

// iter 73 (2026-05-19): shared-memory tiled variant of conv fwd.  Each block
// processes N_OUT=32 output rows × COLS_PER_BLOCK=256 columns; loads x rows
// + filter slice into smem once and reuses across N_OUT outputs.  Eliminates
// L2 thrashing on x reads.  Bit-identical math to the row-major kernel
// (same K*x accumulation order, same break-on-negative-src termination).
// Falls back to the row-major kernel when w != 8 (templated W_FILTER=9).
// iter 96 (2026-05-21): added W_FILTER=5 specialization for w=4 (production
// triple-stack flagship); bit-identical sub-ULP NULL on wall (per iter 96
// bench) but kept for clean code path.
bool scfa_depthwise_causal_conv_fwd_tiled(const float* x, const float* K,
                                            int T, int m, int w, float* y,
                                            cudaStream_t stream = 0);

// iter 97 (2026-05-21): fused-sub variant of conv fwd tile.  Reads (q, q_par)
// inputs and computes q_perp = q - q_par AT SMEM-LOAD TIME (single FP32 sub
// per element, in registers, before smem store).  Eliminates the explicit
// chiron_scfa_sub kernel launch + q_perp materialization round-trip through
// global memory (5.1% of step wall per iter 96 nsys; fwd half = ~2.5%).
// Math: bit-identical to (chiron_scfa_sub THEN scfa_depthwise_causal_conv_fwd_tiled)
// chain — same FP32 q - q_par sub, same K * q_perp FMA accumulation order.
// Specializes W_FILTER=5 (w=4 prod) and W_FILTER=9 (w=8 prior); returns false
// for other w (caller must check before dispatching).
bool scfa_depthwise_causal_conv_fwd_sub_fused_tiled(
    const float* q, const float* q_par, const float* K,
    int T, int m, int w, float* y, cudaStream_t stream = 0);

// iter 101 (2026-05-21): dual-output variant of iter 97 fused-sub fwd tile.
// Same fused-sub conv but ALSO writes q_perp = q - q_par to a side output
// buffer.  Enables fusion at the bwd recompute path (line ~7066 in
// scfa_attention_backward) where q_perp materialization is needed for the
// downstream bwd_dwconv (line ~7440) dK computation.  Specializes W_FILTER=5
// (w=4) and W_FILTER=9 (w=8); returns false for other w.
bool scfa_depthwise_causal_conv_fwd_sub_fused_dual_out_tiled(
    const float* q, const float* q_par, const float* K,
    int T, int m, int w,
    float* y, float* q_perp_out,
    cudaStream_t stream = 0);

// Backward through depthwise causal conv.
//   dx[t, c] += Σ_{i=0..w, t+i<T} K[c, i] · dy[t+i, c]
//   dK[c, i] += Σ_{t=i..T-1}     x[t-i, c]    · dy[t, c]
// Caller must zero dx and dK before this call (the kernels accumulate +=).
bool scfa_depthwise_causal_conv_bwd(const float* x, const float* K,
                                     const float* dy,
                                     int T, int m, int w,
                                     float* dx, float* dK,
                                     cudaStream_t stream = 0);

// iter 95 (2026-05-21): tiled-dx variant of conv bwd.  Mirrors iter 73's
// forward-conv shared-mem tiling to the acausal dx path (dy[t+i] forward).
// Each block processes N_OUT=16 t-rows × COLS=256 columns of dx, loading
// (N_OUT + w) dy rows + filter slice into smem once.  Bit-identical math to
// scfa_depthwise_causal_conv_bwd (same FMA order, same break-on-OOB).
// dx kernel: tiled for w == 4 (W_FILTER=5) or w == 8 (W_FILTER=9); row-major
// fallback for other w.  dK kernel: unchanged (already _par optimized).
bool scfa_depthwise_causal_conv_bwd_tiled(const float* x, const float* K,
                                            const float* dy,
                                            int T, int m, int w,
                                            float* dx, float* dK,
                                            cudaStream_t stream = 0);

// iter 99 (2026-05-21): dual-output bwd dispatch.  dx kernel writes the
// per-element dx contribution to BOTH dx_primary (+= accumulator) AND
// dx_secondary (= single-assign).  Eliminates the explicit axpy at the end
// of scfa_attention_backward (line ~7495: `s.dq_buf += scfa_yperp`) by
// having this kernel accumulate directly into s.dq_buf inline.  Math:
// bit-identical accumulator value; FP32 add ordering differs vs current
// (cuBLAS then axpy) chain — sub-ULP drift class.  Tiled for w == 4 / w == 8;
// row-major fallback.  dK kernel unchanged.
bool scfa_depthwise_causal_conv_bwd_dual_out(const float* x, const float* K,
                                              const float* dy,
                                              int T, int m, int w,
                                              float* dx_primary,
                                              float* dx_secondary,
                                              float* dK,
                                              cudaStream_t stream = 0);

// Cast-elim Port B (2026-06-12): dual_out variant that also side-writes the
// BF16 RNE mirror of dx_secondary (bit-identical to a subsequent
// cast_f32_to_bf16), so the downstream B^T·dq_perp FAST_16BF GEMM can
// consume the mirror via the fast16bf constant table.  NULL mirror falls
// back to scfa_depthwise_causal_conv_bwd_dual_out; supports w ∈ {4, 8}
// only (the production iter-99 dispatch gate).
bool scfa_depthwise_causal_conv_bwd_dual_out_bf16mirror(
    const float* x, const float* K,
    const float* dy,
    int T, int m, int w,
    float* dx_primary,
    float* dx_secondary,
    unsigned short* dx_secondary_bf16,
    float* dK,
    cudaStream_t stream = 0);

// Legacy research helper: fill B[T × k] with a truncated orthonormal DCT-II
// basis. Causal CHIRON SCFA does not use this globally noncausal basis.
bool scfa_dct_basis_init(float* B_flat, int T, int k);

// ---------------------------------------------------------------------------
// Activation functions (element-wise, n elements)
// ---------------------------------------------------------------------------

bool gelu_forward(const float* x, int n, float* out);
bool gelu_backward(const float* dout, const float* x, int n, float* dx);

bool silu_forward(const float* x, int n, float* out);
bool silu_backward(const float* dout, const float* x, int n, float* dx);

bool relu_forward(const float* x, int n, float* out);
bool relu_backward(const float* dout, const float* x, int n, float* dx);

// ---------------------------------------------------------------------------
// SwiGLU
// ---------------------------------------------------------------------------

// Forward: gate_up[n, 2*dFF] -> out[n, dFF].
// out = silu(gate_up[:, :dFF]) * gate_up[:, dFF:]
bool swiglu_forward(const float* gate_up, int n, int dFF, float* out);

// Backward: d_gate_up[n, 2*dFF] from dout[n, dFF].
bool swiglu_backward(const float* dout, const float* gate_up,
                     int n, int dFF, float* d_gate_up);

// ---------------------------------------------------------------------------
// Rotary positional encoding (RoPE)
// ---------------------------------------------------------------------------

// Apply RoPE in-place. x[T, nHeads, dHead], invFreq[halfDim].
// halfDim: number of rotation pairs (halfDim <= dHead/2; 0 => dHead/2).
// inverse: if true, apply inverse rotation (negate sin terms) for backward pass.
bool rope_apply(float* x, const float* invFreq,
                int T, int nHeads, int dHead,
                int halfDim = 0, bool inverse = false);

// Fused Q+K RoPE: apply RoPE to both Q and K arrays in a single kernel launch.
bool rope_apply_qk(float* Q, float* K, const float* invFreq,
                    int T, int nQHeads, int nKVHeads, int dHead,
                    int halfDim = 0, bool inverse = false);

// ---------------------------------------------------------------------------
// Simple vector ops
// ---------------------------------------------------------------------------

// out[rows, cols] += bias[cols]   (broadcast add bias to each row)
bool add_bias(float* out, const float* bias, int rows, int cols);

// out[n] += residual[n]
bool add_residual(float* out, const float* residual, int n);

// out[n] = a[n] + b[n]
bool add_two(float* out, const float* a, const float* b, int n);
// out[n] = a[n] + beta * b[n]
bool add_two_scaled(float* out, const float* a, const float* b, float beta, int n);

// y[n] += alpha * x[n]
bool axpy(float alpha, const float* x, float* y, int n);

// x[n] *= scale
bool scale_array(float* x, float scale, int n);

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

// Forward: out[T, dModel] = E[tokenIds[T], :].
bool embedding_gather(const float* E, const int* tokenIds,
                      int T, int vocabSize, int dModel, float* out);
// BF16-master variant: gather from a bf16 embedding table.  Used when
// MixedPrecisionConfig::weightStorageBf16 is true and the FP32 master is
// retired (only the bf16 mirror remains).  Output stays FP32 because the
// downstream activation path is FP32.
bool embedding_gather_bf16(const uint16_t* E_bf16, const int* tokenIds,
                            int T, int vocabSize, int dModel, float* out);

// Backward: dE[tokenIds[T], :] += dout[T, dModel].
bool embedding_scatter_add(float* dE, const int* tokenIds,
                           const float* dout,
                           int T, int vocabSize, int dModel);

// BF16-output variant: dE_bf16[tokenIds[T], :] +=bf16 dout[T, dModel] via
// atomicCAS-on-uint32 for atomic bf16 add.  Used by Phase-3 BF16-grad path
// to commit token-embedding grads directly to the bf16 mirror.
bool embedding_scatter_add_bf16(uint16_t* dE_bf16, const int* tokenIds,
                                const float* dout,
                                int T, int vocabSize, int dModel);
// V15 read-only row-coverage tap: rowMass[token[t]] += sum_d |dout[t,d]|.
bool embedding_coverage_accumulate(const int* tokenIds, const float* dout,
                                   int T, int vocabSize, int dModel,
                                   float* rowMass);

// Per-row RMS clamp with non-finite sanitization, in place on x [rows×cols].
//  - any non-finite element in a row → entire row zeroed, ++*d_nonfiniteCount
//  - else row RMS > tauRms           → row scaled by tauRms/rms, ++*d_clampedCount
//  - else                            → row untouched (no write; bit-identical)
// Row sum-of-squares accumulates in double so huge-but-finite rows rescale
// correctly instead of overflowing FP32 to inf.  Deterministic.  Count
// pointers are device ints and may be NULL.  Returns false on invalid args
// or tauRms <= 0 (the disabled path must not call).  Used by the CHIRON
// trainer to bound dq_0 rows before embedding_scatter_add (SIRA stability
// mitigation, 2026-06-11 — see SIRA_TERMINAL_30K_RESULT_2026_05_27.md).
bool row_rms_clamp(float* x, int rows, int cols, float tauRms,
                   int* d_clampedCount, int* d_nonfiniteCount);
// V1 fused form. d_preClampSumSq is ADDITIVE and records the raw sum of
// squares before any sanitization/rescale; caller controls reset/cadence.
bool row_rms_clamp_vitals(float* x, int rows, int cols, float tauRms,
                          int* d_clampedCount, int* d_nonfiniteCount,
                          float* d_preClampSumSq,
                          int* d_boundaryClamped = NULL,
                          int* d_boundaryNonfinite = NULL);

// Per-vector L2-norm clamp, in place on x[n].  Non-finite vector → zeroed;
// else ‖x‖₂ > maxNorm → scaled by maxNorm/‖x‖; else untouched (bit-identical).
// Sum-of-squares accumulates in double so huge-but-finite gradients rescale
// instead of overflowing.  d_clampedCount may be NULL.  Returns false on
// invalid args or maxNorm <= 0.  Used by the per-group gradient clamp
// (q-side instability mitigation 1): bound each layer's dgamma/dbeta L2 norm
// before the global-norm sum.  See research/QSIDE_INSTABILITY_INVESTIGATION_2026_06_14.md.
bool clamp_vector_l2norm(float* x, int n, float maxNorm, int* d_clampedCount);

// Elementwise hard cap: clamp each element of x to [-cap, +cap].  No-op (returns
// true) on n<=0 or cap<=0.  Used by the OBSD a_drift gate cap (--drift-gate-cap,
// bounded-gate salvage for the un-capped gate that grew to maxA ~1.8).
bool clamp_abs(float* x, int n, float cap);

// Adaptive Gradient Clipping (AGC, Phase 1): clip g to lambda*max(‖w‖, eps),
// auto-scaled to the parameter norm.  d_count may be NULL.  Returns false on
// invalid args / lambda<=0.  See docs/superpowers/plans/2026-06-16-chiron-stability-techniques.md.
bool agc_clamp_vector(float* g, const float* w, int n, float lambda, float eps, int* d_count);

// Gradient Centralization (Phase 2): subtract each row's mean from a
// [rows, cols] gradient in place. See docs/superpowers/plans/2026-06-16-chiron-stability-techniques.md.
bool gradient_centralize(float* g, int rows, int cols);
// BF16-in/BF16-out variant for the bf16Grads weight-grad path (dWq/k/v/o_bf).
bool gradient_centralize_bf16(uint16_t* g, int rows, int cols);

// Spectral norm via power iteration (Phase 4): estimate σ_max(W) for the
// [rows, cols] row-major FP32 matrix W. u (rows) is the persistent left
// singular vector (caller inits nonzero; reuse warm across steps); v (cols) is
// scratch. Returns σ_max in *sigmaOut.
bool spectral_norm_estimate(const float* W, int rows, int cols,
                            float* u, float* v, int iters, float* sigmaOut);
// Per-step spectral normalization (on-device conditional down-scale to maxSigma).
// FP32 master variant scales W in place; BF16 variant scales the BF16 master Wbf
// using a caller-materialized FP32 view Wf32. u/v are cold-start scratch (rows/
// cols). Pass sigmaOut=NULL on the hot path to skip the host σ readback.
bool spectral_normalize(float* W, int rows, int cols,
                        float* u, float* v, int iters, float maxSigma, float* sigmaOut);
bool spectral_normalize_bf16(uint16_t* Wbf, const float* Wf32, int rows, int cols,
                             float* u, float* v, int iters, float maxSigma, float* sigmaOut);

// SAM (Phase 5) perturb/restore: W += scale*g (scale = ±rho/‖g‖). FP32 + BF16.
bool sam_perturb(float* W, const float* g, int n, float scale);
bool sam_perturb_bf16(uint16_t* W, const uint16_t* g, int n, float scale);

// ---------------------------------------------------------------------------
// Adam optimizer
// ---------------------------------------------------------------------------

// In-place Adam update for n parameters.
// gradScale is applied to grad before the update (use 1.0f for no scaling).
bool adam_update(float* param, const float* grad, float* m, float* v,
                 float lr, float beta1, float beta2, float eps,
                 float weightDecay, float gradScale, int step, int n);

// SOPHIA-G — paradigm shift #55 (Liu et al. 2023, Sophia-G variant).
// Drop-in replacement for adam_update with the Sophia clipped second-order
// update rule.  Uses g² as a Hessian proxy (no HVP); m/h state same shape as
// Adam's m/v.  Defaults: beta1=0.965, beta2=0.99, gamma=0.05, rho=0.04
// (Liu 2023 paper for LLM pre-training).
// Empirical 1.5-2× steps reduction to fixed final NLL vs Adam.
bool sophia_g_update(float* param, const float* grad, float* m, float* h,
                      float lr, float beta1, float beta2,
                      float gamma, float rho, float eps,
                      float weightDecay, float gradScale, int step, int n);

// SOPHIA-G with bf16 m/h state — half the optimizer-state VRAM.
bool sophia_g_update_bf16_state(float* param, const float* grad,
                                 uint16_t* m_bf16, uint16_t* h_bf16,
                                 float lr, float beta1, float beta2,
                                 float gamma, float rho, float eps,
                                 float weightDecay, float gradScale,
                                 int step, int n);

// iter 181 — ASTRA paradigm #41 Gate-0 (m=1 stateless v).
// Replaces Adam's persistent v EMA with the within-step instantaneous
// magnitude v_t = g_t².  Persistent state collapses to momentum m only —
// no v, no Kahan c.  At m=1 this is the limiting case; the full framework
// uses microbatch variance v_t = (1/m) Σ_i (g_t^{(i)})² with m≥4.
// Memory at 1.84B: ~7.4 GB freed vs Adam-bf16, ~14.7 GB freed vs Adam-fp32.
bool astra_update(float* param, const float* grad, float* m,
                  float lr, float beta1, float eps,
                  float weightDecay, float gradScale, int step, int n);

// Adam with BF16-packed optimizer state (m, v as uint16_t BF16 views).
// Loads are lossless-upcast to FP32, compute is FP32, stores are
// round-to-nearest-even FP32 -> BF16. Weights + grads stay FP32.
// Halves optimizer-state VRAM (4 bytes -> 2 bytes per parameter per moment).
// Same mathematical update as adam_update up to BF16 precision of the EMAs.
bool adam_update_bf16_state(float* param, const float* grad,
                            uint16_t* m_bf16, uint16_t* v_bf16,
                            float lr, float beta1, float beta2, float eps,
                            float weightDecay, float gradScale,
                            int step, int n);

// iter 171: Kahan-compensated BF16 Adam.  Adds one extra BF16 buffer per
// param (c_bf16) that carries the truncation residual of v's BF16 store
// into the next step's update.  Eliminates the slow "bf16 v underestimate"
// drift that destabilized 1.84B × 650k run-3 mid-Phase-C (surprise #17).
// Memory cost: +1 BF16 / parameter (= 50% more Adam VRAM than plain bf16).
// Math is unchanged from adam_update_bf16_state up to the recovered low bits.
bool adam_update_bf16_kahan_state(float* param, const float* grad,
                                   uint16_t* m_bf16, uint16_t* v_bf16,
                                   uint16_t* c_bf16,
                                   float lr, float beta1, float beta2, float eps,
                                   float weightDecay, float gradScale,
                                   int step, int n);

// Adam with int8-packed optimizer state (block-wise absmax scale).
// Asymmetric: m as signed int8 [-127, 127] scaled by absmax/127;
//             v as UNSIGNED uint8 [0, 255] scaled by absmax/255
//             (v is non-negative so uint8 doubles the precision that
//              matters for the 1/√v Adam denominator).
//
// Each block of 256 parameters stores one FP32 absmax scale per moment.
// Memory: ~1.016 bytes/param/moment, half the BF16 cost and one-quarter
// of FP32.  param and grad stay FP32.
//
// The caller must allocate m_scale, v_scale with at least
// adam_int8_scale_count(n) FP32 entries; they are persistent state updated
// each step.  Initial values: zero (first-step read dequantizes to 0).
bool adam_update_int8_state(float* param, const float* grad,
                             int8_t* m_int8, uint8_t* v_uint8,
                             float* m_scale, float* v_scale,
                             float lr, float beta1, float beta2, float eps,
                             float weightDecay, float gradScale,
                             int step, int n, float* vitalsStats = NULL);

// BF16-grad variants of the above two: read gradient from a BF16 buffer
// instead of FP32.  Used when MixedPrecisionConfig::gradStorageBf16 is
// true (the persistent grad accumulator is BF16, halving its VRAM cost).
// adam_update_int8_state_bf16grad takes a scratch FP32 buffer (size >= n)
// for the bf16->fp32 cast pass; the BF16-Adam variant decodes inline.
bool adam_update_bf16_state_bf16grad(float* param, const uint16_t* grad_bf16,
                                      uint16_t* m_bf16, uint16_t* v_bf16,
                                      float lr, float beta1, float beta2, float eps,
                                      float weightDecay, float gradScale,
                                      int step, int n);
bool adam_update_int8_state_bf16grad(float* param, const uint16_t* grad_bf16,
                                      int8_t* m_int8, uint8_t* v_uint8,
                                      float* m_scale, float* v_scale,
                                      float* scratch_fp32,
                                      float lr, float beta1, float beta2, float eps,
                                      float weightDecay, float gradScale,
                                      int step, int n, float* vitalsStats = NULL);

// BF16-WEIGHT variants: param is a bf16 buffer (Lowp mirror as canonical
// weight store, no FP32 master).  Each step: cast bf16 weight → weight_scratch
// (FP32) → existing FP32 Adam path applies update in place → cast back to
// bf16 with stochastic rounding (baseSeed XOR step → kernel RNG).  Caller
// supplies stochastic-round seed + step counter so the rounding is
// deterministic per (model, step, tensor).
bool adam_update_bf16_state_bf16grad_bf16w(uint16_t* param_bf16,
                                            float* weight_scratch_fp32,
                                            const uint16_t* grad_bf16,
                                            uint16_t* m_bf16, uint16_t* v_bf16,
                                            float lr, float beta1, float beta2, float eps,
                                            float weightDecay, float gradScale,
                                            int step, int n,
                                            uint32_t srBaseSeed, uint32_t srStepIdx);
bool adam_update_int8_state_bf16grad_bf16w(uint16_t* param_bf16,
                                            float* weight_scratch_fp32,
                                            const uint16_t* grad_bf16,
                                            int8_t* m_int8, uint8_t* v_uint8,
                                            float* m_scale, float* v_scale,
                                            float* grad_scratch_fp32,
                                            float lr, float beta1, float beta2, float eps,
                                            float weightDecay, float gradScale,
                                            int step, int n,
                                            uint32_t srBaseSeed, uint32_t srStepIdx);

// Iter 49: fused int8 Adam with BF16-weight + BF16-grad inline I/O.  No FP32
// scratch needed.  Replaces the cast+adam+cast 4-kernel chain with one kernel.
bool adam_update_int8_state_bf16w_bf16g_fused(uint16_t* param_bf16,
                                               const uint16_t* grad_bf16,
                                               int8_t* m_int8, uint8_t* v_uint8,
                                               float* m_scale, float* v_scale,
                                               float lr, float beta1, float beta2, float eps,
                                               float weightDecay, float gradScale,
                                               int step, int n,
                                               uint32_t srBaseSeed, uint32_t srStepIdx,
                                               float* vitalsStats = NULL);

// Returns the number of FP32 scale entries required for int8 Adam state
// given a parameter count n.
int adam_int8_scale_count(int n);

bool adam_group_scale_batch(float** d_params, float** d_grads,
                            float** d_ms, float** d_vs,
                            float* d_groupScales, float* d_groupPrevStepRms,
                            const int* d_sizes,
                            float beta1, float beta2, float eps,
                            float gradScale, int step, int groupCount,
                            unsigned int minGroupSize,
                            float stabilityScale, float snrScale, float ratioScale,
                            float minScale, float maxScale);

// Batched Adam: process all parameter groups in a single kernel launch.
// d_params/d_grads/d_ms/d_vs are device arrays of groupCount pointers.
// d_baseLrs/d_wds are device arrays of groupCount floats (static per-group
// base lr and weight decay). lrScale is multiplied into each base lr in-kernel.
// d_sizes is a device array of groupCount ints (element counts).
// maxSize is the largest element count across all groups.
bool adam_update_batch(float** d_params, float** d_grads,
                       float** d_ms, float** d_vs,
                       const float* d_baseLrs, const float* d_wds,
                       float lrScale,
                       const float* d_stepScales,
                       const int* d_sizes, int maxSize,
                       float** d_rowMetrics, float** d_colMetrics,
                       float** d_rowStructMetrics, float** d_colStructMetrics,
                       float** d_prevMhats, float** d_metricScratch,
                       const int* d_metricRows, const int* d_metricCols,
                       float beta1, float beta2, float eps,
                       float gradScale, int step, int groupCount);

// SOPHIA-G batched variant — drop-in replacement for adam_update_batch
// when OptimizerConfig::type == SOPHIA_G.  Reuses the same buffer-of-pointers
// layout (d_params/d_grads/d_ms/d_hs, where d_hs replaces d_vs).  No ECHO
// metric scaling.  Uses the Sophia clipped second-order rule:
//   ratio = clip(m_hat / max(γ · h_hat, ε), -ρ, ρ)
//   θ ← θ - lr · ratio  (decoupled WD applied separately)
bool sophia_g_update_batch(float** d_params, float** d_grads,
                            float** d_ms, float** d_hs,
                            const float* d_baseLrs, const float* d_wds,
                            float lrScale, const int* d_sizes, int maxSize,
                            float beta1, float beta2, float gamma, float rho,
                            float eps, float gradScale, int step, int groupCount);

// ---------------------------------------------------------------------------
// Flash attention (simplified single-head)
// ---------------------------------------------------------------------------

// Forward: O[T, dV] = softmax(Q K^T / sqrt(dK)) V, with optional causal mask.
bool flash_attention_forward(const float* Q, const float* K, const float* V,
                             int T, int dK, int dV, bool causal,
                             float* O);

// Backward: dQ, dK, dV from dO.
bool flash_attention_backward(const float* Q, const float* K, const float* V,
                              const float* O, const float* dO,
                              int T, int dK, int dV, bool causal,
                              float* dQ, float* dK_out, float* dV_out);

// Packed multi-head/GQA flash-style attention for training.
// Q[T, dModel], K/V[T, dModelKV], O[T, dModel] are row-major with heads packed
// contiguously inside the model dimension.
bool flash_attention_multihead_forward(const float* Q, const float* K, const float* V,
                                       int T, int nHeads, int nKVHeads,
                                       int dHead, int dModel, int dModelKV,
                                       bool causal, float* O);

// BF16-input variant: Q/K/V are uint16_t BF16 values; softmax/accumulation
// stays FP32 inside the kernel. Reduces attention memory traffic by 2x
// (the dominant cost at long seq length).
bool flash_attention_multihead_forward_bf16(const uint16_t* Q, const uint16_t* K,
                                            const uint16_t* V,
                                            int T, int nHeads, int nKVHeads,
                                            int dHead, int dModel, int dModelKV,
                                            bool causal, float* O);

// Local-window BF16 flash attention.  Each query attends only to keys
// within ±windowSize tokens (causal: [q-W, q]; non-causal: [q-W, q+W]).
// Tiles falling entirely outside the window are skipped — compute goes
// from O(T·T·dH) to O(T·W·dH), a 10-100× reduction at long context.
// windowSize <= 0 or >= T falls back to full attention.
// See research/SUBQUADRATIC_ATTENTION_DESIGN.md.
// Paradigm #78 ATTENTION-SINK: passing sinkCount > 0 makes the first
// `sinkCount` keys always allowed regardless of window. Default 0 preserves
// existing call sites.
bool flash_attention_multihead_forward_bf16_local(const uint16_t* Q, const uint16_t* K,
                                                   const uint16_t* V,
                                                   int T, int nHeads, int nKVHeads,
                                                   int dHead, int dModel, int dModelKV,
                                                   bool causal, int windowSize,
                                                   float* O, int sinkCount = 0);

// Local-window BF16 flash attention backward.  Same windowing rules as the
// forward kernel; dK, dV accumulated via atomicAdd only on in-window keys.
// windowSize <= 0 or >= T falls back to the full backward (unless sinkCount>0).
bool flash_attention_multihead_backward_bf16_local(
    const uint16_t* Q, const uint16_t* K, const uint16_t* V,
    const float* O, const float* dO,
    int T, int nHeads, int nKVHeads,
    int dHead, int dModel, int dModelKV,
    bool causal, int windowSize,
    float* dQ, float* dK_out, float* dV_out, int sinkCount = 0);

// BF16-input backward variant. Q/K/V BF16; O/dO/dQ/dK/dV FP32 (each loaded
// or written once so the traffic savings are negligible there). Returns
// false if the kernel cannot be launched for the requested shape (caller
// should route through the FP32 variant in that case).
bool flash_attention_multihead_backward_bf16(
    const uint16_t* Q, const uint16_t* K, const uint16_t* V,
    const float* O, const float* dO,
    int T, int nHeads, int nKVHeads,
    int dHead, int dModel, int dModelKV,
    bool causal,
    float* dQ, float* dK_out, float* dV_out);

// ---------------------------------------------------------------------------
// Paradigm-shift GPU kernels (impl(paradigm-74/76/78))
// ---------------------------------------------------------------------------

// #74 PHOENIX-1BIT binary GEMM. Y[M,N] = X[M,K] @ unpack(W_bits) where
// W_bits is column-major bit packed (per phoenix_pack_signs_colmajor).
// All pointers are device pointers. Reference (correctness) kernel; not
// production-tuned.
bool phoenix_binary_gemm_gpu(const float* X,
                              const unsigned char* W_bits,
                              int M, int N, int K,
                              float* Y);

// #74 PHOENIX-1BIT FFN-side helper: Y[M,N] = X[M,K] @ sign(W[N,K]).T
// Operates on float weights in-place via on-the-fly sign extraction.
// Tiled (BM=BN=BK=32) shared-memory kernel for M,N >= 32; naive scalar
// fallback otherwise.
bool binary_gemm_abt_from_float(const float* X, const float* W,
                                int M, int N, int K, float* Y);

// #74 BF16 fast path: cast sign(W) to BF16 ±1.0 in W_bf16 (n elements).
// Pair with cuBLAS bf16 sgemm via gpu_gemm_abt_mp for tensor-core throughput.
bool binarize_to_bf16_signs(const float* W, uint16_t* W_bf16, size_t n);

// (b) WMMA B1 tensor-core binary GEMM (SM 7.5+, including Ada SM 8.9).
// C[M,N] (int32) = popcount( A_bits[M,K] AND B_bits[N,K] ) over K bits.
// K_bits must be a multiple of 128. To recover ±1 GEMM:
//   signed_dot = K_bits - 2 * popcount(A ^ B)
// Production deployment requires binarizing both operands (BitNet b1.0 style).
bool wmma_b1_gemm(const unsigned int* A_bits, const unsigned int* B_bits,
                   int M, int N, int K_bits, int* C);

// (b) Full BitNet b1.0 binary inference helpers (paradigm #74 inference path).
// Quantize X[M,K] (float) → X_bits + alpha[M] (per-row mean(|x|) scale).
// K must be a multiple of 32.
bool quantize_x_to_b1_with_scale(const float* X, int M, int K,
                                 unsigned int* X_bits, float* alpha);

// Full BitNet forward: Y[M,N] = alpha_x[m] * alpha_w[n] * (K_bits - 2*popcount(X_bits ⊕ W_bits))
// C_pop_scratch must be M*N int32 device memory; reused across calls.
bool bitnet_b1_forward(const unsigned int* X_bits,
                       const unsigned int* W_bits,
                       const float* alpha_x,
                       const float* alpha_w,
                       int* C_pop_scratch,
                       int M, int N, int K_bits,
                       float* Y);

// (b) Full BitNet QAT FFN forward primitive: takes float X, float W;
// quantizes both with per-row scales; runs WMMA B1 XOR-popcount; recovers
// the scaled result into Y. Backward via STE goes through standard cuBLAS
// sgemm on float X and float W (caller's responsibility).
// Scratch buffers must be allocated by caller; K must be multiple of 128.
bool bitnet_ffn_forward_gpu(const float* X, const float* W,
                            int M, int N, int K,
                            unsigned int* X_bits_scratch,
                            unsigned int* W_bits_scratch,
                            float* alpha_x_scratch,
                            float* alpha_w_scratch,
                            int* C_pop_scratch,
                            float* Y);

// #76 MLA latent compression: c = h @ W_DKV via cuBLAS sgemm.
bool mla_compute_latent_gpu(const float* h, const float* W_DKV,
                             int T, int d_h, int d_c, float* c_out);

// #76 MLA KV decompression: K = c @ W_UK, V = c @ W_UV via cuBLAS.
bool mla_decompress_kv_gpu(const float* c,
                            const float* W_UK, const float* W_UV,
                            int T, int d_c, int dKVtotal,
                            float* K_out, float* V_out);

// (a) Full MLA forward — single trainer-ready entry point.
// h[T, dHidden] -> c_scratch[T, dC] -> K_out, V_out [T, dKVtotal] each.
bool mla_attention_forward_gpu(const float* h,
                                const float* W_DKV,
                                const float* W_UK,
                                const float* W_UV,
                                int T, int dHidden, int dC, int dKVtotal,
                                float* c_scratch,
                                float* K_out,
                                float* V_out);

// (a) Full MLA backward — produces dh (accumulated), dW_DKV, dW_UK, dW_UV.
// c_cached should be the c from forward; dc_scratch is FP32 workspace [T*dC].
// bf16_scratch_A and bf16_scratch_B must each be sized for the largest
// operand: max(T*dHidden, T*dKVtotal, T*dC, dHidden*dC, dC*dKVtotal).
// Routes through cuBLAS BF16 atb/abt path to avoid the FP32 atb cuBLAS bug.
bool mla_attention_backward_gpu(const float* h,
                                 const float* c_cached,
                                 const float* dK,
                                 const float* dV,
                                 const float* W_DKV,
                                 const float* W_UK,
                                 const float* W_UV,
                                 int T, int dHidden, int dC, int dKVtotal,
                                 float* dh_accum,
                                 float* dW_DKV,
                                 float* dW_UK,
                                 float* dW_UV,
                                 float* dc_scratch,
                                 unsigned short* bf16_scratch_A,
                                 unsigned short* bf16_scratch_B);

// #78 ATTENTION-SINK forward (single-head FP32 reference). One block per
// query position; inner thread reduces over keys with sink+window mask.
bool sw_attention_forward_gpu(const float* Q, int qStride,
                               const float* K, int kStride,
                               const float* V, int vStride,
                               int T, int dHead, bool causal,
                               int sinkCount, int windowSize,
                               float* O, int oStride);

// #78 ATTENTION-SINK backward (recompute, FP32, single-head). Caller
// must zero dQ/dK_out/dV_out — kernel uses atomicAdd accumulation.
bool sw_attention_backward_gpu(const float* Q, int qStride,
                                const float* K, int kStride,
                                const float* V, int vStride,
                                const float* dO, int dOStride,
                                int T, int dHead, bool causal,
                                int sinkCount, int windowSize,
                                float* dQ, int dQStride,
                                float* dK_out, int dKStride,
                                float* dV_out, int dVStride);

// Backward for packed multi-head/GQA flash-style attention.
// dQ is written per query head; dK/dV are accumulated per KV head.
bool flash_attention_multihead_backward(const float* Q, const float* K, const float* V,
                                        const float* O, const float* dO,
                                        int T, int nHeads, int nKVHeads,
                                        int dHead, int dModel, int dModelKV,
                                        bool causal,
                                        float* dQ, float* dK_out, float* dV_out);

// ---------------------------------------------------------------------------
// Incremental KV-cache attention (single-query, multi-head)
// ---------------------------------------------------------------------------

// Single-query incremental attention against KV cache for inference.
// Q[nHeads * dHead]: current token's query vectors (all heads concatenated).
// K_cache[maxLen, dModelKV]: key cache for one layer (dModelKV = nKVHeads*dHead).
// V_cache[maxLen, dModelKV]: value cache for one layer.
// scores_scratch[nHeads * maxLen]: global memory scratch for attention scores.
// keyValid[maxLen]: 1=valid, 0=masked (NULL => all valid). Host or device ptr.
// pos: current position index (attend to positions 0..pos inclusive).
// invSqrt: 1.0/sqrt(dHead).
// out[nHeads * dHead]: output attention vectors (all heads concatenated).
bool kv_attention_incremental(const float* Q,
                              const float* K_cache, const float* V_cache,
                              float* scores_scratch,
                              const unsigned char* keyValid,
                              int nHeads, int nKVHeads, int dHead,
                              int dModelKV, int maxLen, int pos,
                              float invSqrt, float* out);

// ---------------------------------------------------------------------------
// Reduction
// ---------------------------------------------------------------------------

// Sum across rows: out[col] = beta * out[col] + sum_{row} input[row, col].
// input is [rows, cols] row-major.  out is [cols].
// beta=0.0 for overwrite, beta=1.0 for accumulation.
bool reduce_rows_sum(const float* input, int rows, int cols,
                     float beta, float* out);

// ---------------------------------------------------------------------------
// Attention softmax helpers
// ---------------------------------------------------------------------------

// In-place causal-masked softmax on S[batchSize, T, T] row-major.
// Each (batch, row) block: set S[i,j]==-FLT_MAX for j>i, then stable softmax.
bool causal_mask_softmax_inplace(float* S, int batchSize, int T);

// iter 102 (2026-05-21): fused causal-masked softmax + FP32→BF16 cast.
// Same FP32 mask/max/exp/sum/normalize as causal_mask_softmax_inplace, but
// pass 3 writes the BF16 result directly to P_bf instead of normalizing
// in-place on S.  Eliminates a separate cast_f32_to_bf16 launch + the
// FP32 S → BF16 P memory round-trip.  Math bit-identical to
// (softmax_inplace THEN cast_f32_to_bf16): same FP32 math, same RN-even
// BF16 rounding (matching k_cast_f32_to_bf16).  S is modified during
// passes 1-2 (mask + exp/sum) but its post-call content is no longer
// needed by downstream callers in the bf16-inner attention path.
bool causal_mask_softmax_bf16_out(float* S, uint16_t* P_bf,
                                   int batchSize, int T);

// Softmax backward for attention: dS = outputScale * P * (dP - row_sum(dP * P)),
// zero above-diagonal for causal mask.
// P, dP, dS are [batchSize, T, T] row-major.
bool softmax_backward_attn(const float* P, const float* dP,
                           int batchSize, int T, float outputScale, float* dS);

// iter 115 (2026-05-21): fused causal-masked softmax + softmax_backward_attn.
// Math bit-identical to (causal_mask_softmax_inplace THEN softmax_backward_attn).
// S is FP32 input/output (in-place: S → P after pass 3, then dS computation
// reads P from same buffer in passes A-B).  dP and dS may alias (in-place
// dP → dS overwrite).
// Caller MUST issue cuBLAS dP = dO · V^T BEFORE this call (reorder vs legacy).
bool causal_softmax_with_bwd_attn(float* S, const float* dP,
                                   int batchSize, int T,
                                   float outputScale, float* dS);

// ---------------------------------------------------------------------------
// Loss computation
// ---------------------------------------------------------------------------

// Cross-entropy NLL loss on GPU.
// probs[T, vocabSize] row-major.  targets[T].
// Skips rows where targets[t] == padToken (if padToken >= 0).
// Writes total NLL to *loss_sum and number of valid tokens to *valid_count.
// Both must be device pointers (will be zeroed internally).
bool cross_entropy_nll_loss(const float* probs, const int* targets,
                            int T, int vocabSize, int padToken,
                            float* loss_sum, int* valid_count);

// Argmax accuracy on GPU.
// probs[T, vocabSize] row-major.  targets[T].
// Writes number of correct predictions to *correct_count and valid tokens
// to *valid_count. Both must be device pointers (zeroed internally).
bool argmax_count_matches(const float* probs, const int* targets,
                          int T, int vocabSize, int padToken,
                          int* correct_count, int* valid_count);
bool chiron_vitals_output_vectors(const float* probs, const int* targets,
                                  int T, int vocabSize, int padToken,
                                  float* loss_sum, int* loss_count,
                                  int* correct_count, int* valid_count,
                                  float* perTokenNll,
                                  unsigned char* perTokenTop1);

// ---------------------------------------------------------------------------
// Chunked cross-entropy loss — never materializes the dense T × V logits
// or probs tensor.  Unlocks large-vocab training (V >= 65k).
//
// Inputs:
//   X          [T × d]       row-major activations (last layer output).
//   W_lm       [V × d]       row-major LM-head weight (W_lm[v] is row v).
//   targets    [T]           target token IDs.
//   T, V, d                  dimensions.
//   padToken                 skip rows where targets[t] == padToken (< 0 → no skip).
//   V_chunk_size             columns of W_lm processed per chunk.  At typical
//                            configs V_chunk = 4096 gives T × 4096 logits
//                            scratch (e.g., 32 MB at T=2048).  V_chunk = V
//                            degenerates to the dense path.
//
// Algorithm (streaming log-sum-exp):
//   running_max[t]  = -inf;  running_sum[t] = 0;  target_logit[t] = NaN
//   for chunk_start = 0; chunk_start < V; chunk_start += V_chunk_size:
//     compute logits_chunk [T × V_ch] = X · W_lm[chunk_start:chunk_end, :]^T
//     update running_max, running_sum in a single kernel pass
//     if target[t] falls in this chunk: capture target_logit[t]
//   lse[t]  = log(running_sum[t]) + running_max[t]
//   loss[t] = lse[t] - target_logit[t]
//   sum valid losses → loss_sum, count valid → valid_count
//
// Scratch buffer layout (must be caller-allocated; contents clobbered):
//   logits_chunk [T × V_chunk_size]      floats
//   running_max  [T]                      floats
//   running_sum  [T]                      floats
//   target_logit [T]                      floats
// Total floats: T · (V_chunk_size + 3).
//
// Outputs loss_sum, valid_count are device scalars, zeroed internally.
bool chunked_cross_entropy_loss(const float* X, const float* W_lm,
                                const int* targets,
                                int T, int V, int d, int padToken,
                                int V_chunk_size,
                                float* loss_sum, int* valid_count,
                                float* scratch);

// Chunked cross-entropy BACKWARD — produces dL/dX [T × d] and dL/dW_lm
// [V × d] given the forward's running_max, running_sum (carried over in
// the forward scratch — DO NOT clobber those between forward and
// backward).
//
// For each (valid) token t with target[t] = τ, the softmax-CE gradient
// at column v is
//     dL/dlogits[t, v] = softmax(logits[t])[v] - (v == τ ? 1 : 0)
// scaled by (1 / valid_count) for token-averaged loss.  Invalid rows
// (pad, out-of-range target, or target not captured by forward) emit
// zero gradient.
//
// Chunked backward path (mirrors the forward structure):
//   for cs, ce = 0, V_chunk; cs < V; cs += V_chunk:
//     recompute logits_chunk  = X · W_lm[cs:ce, :]^T       (same GEMM)
//     kernel: softmax_chunk[t, v] = exp(logits[t, v] - running_max[t])
//                                   / running_sum[t]
//                                 - (cs + v == target[t] ? 1 : 0)
//             (scaled by 1/valid_count, masked to 0 on invalid rows)
//     dX    += softmax_chunk · W_lm[cs:ce, :]              (T × d)
//     dW_lm_chunk += softmax_chunk^T · X                   (V_ch × d)
//
// Memory: same scratch as forward (T × (V_chunk + 3) floats, of which
// only T × V_chunk is used here since running_max/running_sum/
// target_logit are supplied by the forward).  dX and dW_lm MUST be
// pre-zeroed (or caller must set accumulate=false).
//
// accumulate=false:  dX and dW_lm are overwritten (zeroed first).
// accumulate=true:   dX and dW_lm are added into (caller pre-seeds).
bool chunked_cross_entropy_backward(const float* X, const float* W_lm,
                                    const int* targets,
                                    const float* running_max,
                                    const float* running_sum,
                                    int T, int V, int d, int padToken,
                                    int V_chunk_size,
                                    int valid_count,
                                    bool accumulate,
                                    float* dX, float* dW_lm,
                                    float* scratch);

// Deterministic chunked squared-hinge max-margin objective used by bounded
// frozen-feature diagnostics. W_aug is [V,dAug], where the final column is bias
// and X_aug's final feature is 1. Scratch layout (floats): logits[T*chunk],
// best_logit[T], target_logit[T], hinge[T]; competitor[T] is a separate int
// buffer. Loss is the SUM of squared hinges; backward scales by 1/valid_count.
bool chunked_squared_hinge_loss(const float* X_aug, const float* W_aug,
                                const int* targets,
                                int T, int V, int dAug, int V_chunk_size,
                                float margin,
                                float* loss_sum, int* valid_count,
                                int* competitor, float* scratch);
bool chunked_squared_hinge_backward(const float* X_aug,
                                    const int* targets,
                                    const int* competitor,
                                    const float* hinge,
                                    int T, int V, int dAug, int V_chunk_size,
                                    int valid_count,
                                    bool accumulate,
                                    float* dW_aug, float* scratch);

// Fixed-order vector primitives for deterministic L-BFGS. Dot products emit
// one partial per 256-element block; callers download and sum partials in order
// using FP64 host accumulation.
int deterministic_dot_partial_count(int n);
bool deterministic_dot_partials(const float* a, const float* b, int n,
                                float* partials, int partial_count);
bool add_anchor_regularizer(const float* value, const float* anchor,
                            int rows, int weight_cols, float lambda,
                            float* gradient, float* partials,
                            int partial_count);
bool project_augmented_bias_gauge(float* value, int rows, int stride,
                                  float* partials, int partial_count);

// ---------------------------------------------------------------------------
// Batch zero: zero multiple GPU buffers with a single kernel launch
// ---------------------------------------------------------------------------

// d_ptrs[count] and d_sizes[count] must be device pointers.
// Each buffer d_ptrs[i] of d_sizes[i] floats is zeroed.
bool zero_buffers_batch(float** d_ptrs, const int* d_sizes, int count);

// ---------------------------------------------------------------------------
// Pack loss scalars: copy 4 device scalars into a contiguous 16-byte buffer
// ---------------------------------------------------------------------------

// Packs lossSum (float), lossCount (int), correctCount (int), validCount (int)
// into out[4] as raw int bits (lossSum reinterpreted). Single D2H download.
bool pack_loss_scalars(const float* lossSum, const int* lossCount,
                       const int* correctCount, const int* validCount,
                       int* out);

// Fused token-LM metrics reducer.
// Computes cross-entropy NLL sum, valid-token count, argmax-correct count,
// and writes the packed 4-scalar payload directly to out[4].
bool collect_token_lm_metrics(const float* probs, const int* targets,
                              int T, int vocabSize, int padToken,
                              int* out);

// ---------------------------------------------------------------------------
// Gradient norm computation
// ---------------------------------------------------------------------------

// Atomically accumulate sum(data[i]^2) into *d_accumulator.
// Caller must zero d_accumulator before the first call.
// Multiple calls accumulate across different buffers.
bool sum_squared_accumulate(const float* data, int n, float* d_accumulator);
// Same read pass, atomically accumulates into global and a VITALS group scalar.
bool sum_squared_accumulate_dual(const float* data, int n,
                                 float* d_accumulator, float* d_secondary);

// BF16-input variant of sum_squared_accumulate.  Decodes each element as bf16->f32
// (zero-extend low 16 bits) and accumulates v*v into d_accumulator.  Used for the
// global grad-norm pass under BF16-grad Phase-2 where the FP32 grad buffers have
// been retired and only the BF16 mirrors are live.
bool sum_squared_accumulate_bf16(const uint16_t* data, int n, float* d_accumulator);
bool sum_squared_accumulate_bf16_dual(const uint16_t* data, int n,
                                      float* d_accumulator, float* d_secondary);

// BF16 ↔ FP32 element-wise casts. Operates element-wise on GPU buffers.
// `n` is the number of elements (not bytes). Designed as primitives for
// mixed-precision training: store a weight matrix as BF16 on the GPU (half
// the memory) and cast to FP32 just before a kernel consumes it. Pair with
// FP32 master weights in the optimizer to avoid compounding rounding errors
// across training steps.
bool cast_f32_to_bf16(const float* src, uint16_t* dst, size_t n);
bool cast_bf16_to_f32(const uint16_t* src, float* dst, size_t n);

// Stochastic FP32 -> BF16 cast.  Rounds up with probability equal to the
// low-16-bit fractional part of the source (so the expected value matches
// the true FP32 value exactly, preserving sub-ULP updates that deterministic
// round-to-nearest-even would quantize to zero).  Needed for BF16-master
// weight training where small Adam updates would otherwise vanish.
//
// RNG is a per-element splittable hash of (idx, stepIdx, baseSeed); no
// global state.  baseSeed and stepIdx should vary per Adam step to avoid
// biased rounding across steps on the same weight entry.
bool cast_f32_to_bf16_stochastic(const float* src, uint16_t* dst, size_t n,
                                  uint32_t baseSeed, uint32_t stepIdx);

// iter 72 (2026-05-19): batched multi-buffer FP32 -> BF16 RN-even cast.
// Accepts up to 8 (src, dst, count) tuples per launch.  Saves kernel
// launch overhead when many short-pipeline cast calls must run in
// sequence on the same stream (e.g., the iter 69 BF16-checkpoint-inner
// cache writes 7 buffers per layer per direction at L=24).  Math is
// bit-identical to N sequential cast_f32_to_bf16 calls.  Constraint:
// num_jobs <= 8.
bool cast_f32_to_bf16_batched(int num_jobs,
                               const float* const* srcs,
                               uint16_t* const* dsts,
                               const size_t* counts);

// BF16 gradient accumulation helper.  Computes in FP32:
//   dst_bf16[i] = bf16( alpha * src_f32[i] + beta * fp32(dst_bf16[i]) )
// with round-to-nearest-even on the output cast.  Used to accumulate FP32
// gradient chunks into a BF16 persistent accumulator across gradient-
// accumulation micro-steps:
//   - first micro-step of window:  beta=0, alpha=1  (overwrite)
//   - subsequent micro-steps:      beta=1, alpha=1  (add into accum)
// Single-pass compute — no intermediate FP32 materialization of dst.
bool bf16_accum_axpy(uint16_t* dst_bf16, const float* src_f32,
                      float alpha, float beta, size_t n);

// ---------------------------------------------------------------------------
// Device memory operations (callable from .cpp files without cuda_runtime.h)
// ---------------------------------------------------------------------------

void device_memcpy_d2d(void* dst, const void* src, size_t bytes);
void device_memcpy_h2d(void* dst, const void* src, size_t bytes);
void device_memcpy_d2h(void* dst, const void* src, size_t bytes);
void device_memcpy_2d_d2d(void* dst, size_t dpitch, const void* src, size_t spitch,
                           size_t width, size_t height);
void device_memset_bytes(void* ptr, int value, size_t bytes);

// ---------------------------------------------------------------------------
// QK-Norm GPU kernels (Task 2.3). Forward: per-token, per-head L2 normalize
// in place; writes invNorm[t,h] = 1/||x_orig|| for backward. Backward:
// jacobian of the normalize step.
// ---------------------------------------------------------------------------
bool qknorm_forward_gpu(float* x, float* invNorm,
                        int T, int nHeads, int dHead, float eps);

bool qknorm_backward_gpu(const float* xNorm, const float* invNorm,
                         const float* dxNorm, int T, int nHeads, int dHead,
                         float* dxOrig);

// Multiply each [t,h] row of Q by gammaScale[h].  Q: [T, nHeads, dHead].
bool scale_q_per_head(float* Q, const float* gammaScale,
                      int T, int nHeads, int dHead);

// Accumulate γ_h gradient: dGamma[h] = sqrt(dHead) * sum_{t,i} dQPost[t,h,i]*qNorm[t,h,i].
// dQPost, qNorm: [T, nHeads, dHead].  dGamma: [nHeads] (overwritten, not accumulated).
bool qknorm_gamma_grad(const float* dQPost, const float* qNorm,
                       float sqrtDh, int T, int nHeads, int dHead,
                       float* dGamma);

// Compute gamma_scale[h] = gamma[h] * sqrtDh element-wise on the GPU.
// Replaces the prior D2H-CPU-H2D round-trip; cuda-graphs capture-safe.
// gamma_d, gamma_scale_d: device pointers of length nH.
bool qknorm_gamma_scale_gpu(const float* gamma_d,
                            float sqrtDh,
                            float* gamma_scale_d,
                            int nH);

} // namespace gpu
} // namespace glades

#else // !GLADES_HAVE_CUDA --------------------------------------------------

namespace glades {
namespace gpu {

inline bool layernorm_forward(const float*, const float*, const float*, float, int, int, float*, float*, float*) { return false; }
inline bool layernorm_backward(const float*, const float*, const float*, const float*, const float*, int, int, float*, float*, float*) { return false; }
inline bool layernorm_backward_bounded(const float*, const float*, const float*, const float*, const float*, int, int, float*, float*, float*, float) { return false; }

inline bool rmsnorm_forward(const float*, const float*, float, int, int, float*, float*) { return false; }
inline bool rmsnorm_backward(const float*, const float*, const float*, const float*, int, int, float*, float*) { return false; }

inline bool softmax_forward(const float*, int, int, float*) { return false; }
inline bool softmax_forward_with_lse(const float*, int, int, float*, float*) { return false; }
inline bool softmax_cross_entropy_bwd(const float*, const int*, int, int, float*) { return false; }
inline bool softmax_cross_entropy_bwd_zloss(const float*, const int*, const float*, float, int, int, float*) { return false; }
inline bool softmax_forward_bf16(const unsigned short*, int, int, unsigned short*) { return false; }
inline bool softmax_forward_bf16_with_lse(const unsigned short*, int, int, unsigned short*, float*) { return false; }
inline bool softmax_cross_entropy_bwd_bf16(const unsigned short*, const int*, int, int, unsigned short*) { return false; }
inline bool softmax_cross_entropy_bwd_bf16_zloss(const unsigned short*, const int*, const float*, float, int, int, unsigned short*) { return false; }
inline bool echo_repeat_stats(const unsigned short*, const int*, const int*, int, int, int, float, float, float*, float*, uint32_t*, int*) { return false; }
inline bool echo_repeat_stats_huber(const unsigned short*, const int*, const int*, int, int, int, float, float, float, float*, float*, uint32_t*, float*, int*, float*) { return false; }
enum EchoSummaryIndex { ECHO_SUM_R = 0, ECHO_SUM_PA, ECHO_SUM_PMAX, ECHO_MAX_P, ECHO_ACTIVE_ROWS, ECHO_ACTIVE_IDS, ECHO_HIST_0, ECHO_HIST_1, ECHO_HIST_2, ECHO_HIST_3, ECHO_HIST_4, ECHO_SUMMARY_SIZE };
inline bool echo_summarize_stats(const float*, const float*, const float*, const int*, int, float*) { return false; }
inline bool softmax_cross_entropy_bwd_bf16_zloss_echo(const unsigned short*, const int*, const float*, float, float, const float*, int, int, unsigned short*) { return false; }
inline bool chiron_crm_forward_backward(const float*, const int*, float coefficient, float, float, int, int, float*, float*) { return coefficient == 0.0f; }
inline bool chiron_crm_forward_backward_bf16(const unsigned short*, const int*, float coefficient, float, float, int, int, unsigned short*, float*) { return coefficient == 0.0f; }
inline bool chiron_crm_forward_backward_bf16_observed(const unsigned short*, const unsigned short*, const int*, float coefficient, float, float, int, int, unsigned short*, float*, float*) { return coefficient == 0.0f; }
inline bool echo_dense_bwd_bf16(const unsigned short*, float, const float*, int, int, unsigned short*) { return false; }
inline bool echo_scatter_bf16(const unsigned short*, const int*, const uint32_t*, float, int, int, int, unsigned short*) { return false; }
inline bool echo_scatter_bf16_weighted(const unsigned short*, const int*, const uint32_t*, const float*, float, int, int, int, unsigned short*) { return false; }
inline bool scale_array_bf16(unsigned short*, float, int) { return false; }
inline bool cross_entropy_nll_loss_bf16(const unsigned short*, const int*, int, int, int, float*, int*) { return false; }
inline bool argmax_count_matches_bf16(const unsigned short*, const int*, int, int, int, int*, int*) { return false; }
inline bool chiron_vitals_output_vectors_bf16(const unsigned short*, const int*, int, int, int, float*, int*, int*, int*, float*, unsigned char*) { return false; }
inline bool cross_entropy_nll_bucketed_bf16(const unsigned short*, const int*, int, int, int, int, float*, int*) { return false; }
inline bool topk_accuracy_bf16(const unsigned short*, const int*, int, int, int, int, const int*, int*, int*) { return false; }
inline bool distill_combined_bwd(const float*, const float*, const int*, int, int, float, float*) { return false; }
inline bool distill_combined_loss(const float*, const float*, const int*, int, int, int, float, float*, int*) { return false; }
inline bool orion_proj_left(const void*, const float*, int, int, float*) { return false; }
inline bool orion_lift_add(float*, const void*, const float*, int, int) { return false; }
inline bool orion_lift_add_bf16w(void*, const float*, const void*, const float*, int, int) { return false; }
inline bool orion_perturb_col(float*, const float*, const void*, int, int, float) { return false; }
inline bool orion_perturb_col_bf16w(void*, const float*, const void*, int, int, float) { return false; }
inline bool orion_oja_tilt(void*, const float*, const float*, int, int, float) { return false; }
inline bool orion_gram_schmidt(void*, int, int, float*, float*) { return false; }
inline bool orion_perturb_col_bf16w_bf16anchor(void*, const void*, const void*, int, int, float) { return false; }
inline bool orion_lift_add_bf16w_bf16anchor(void*, const void*, const void*, const float*, int, int) { return false; }
inline bool orion_proj_left_bf16_src(const void*, const void*, int, int, float*) { return false; }
inline int  orion_v_scale_count(int) { return 0; }
inline bool orion_v_quantize_int8_column(const float*, void*, float*, int) { return false; }
inline bool orion_v_dequantize_int8_column(const void*, const float*, float*, int) { return false; }
inline bool orion_proj_left_int8(const void*, const float*, const float*, int, int, float*) { return false; }
inline bool orion_proj_left_int8_bf16src(const void*, const float*, const void*, int, int, float*) { return false; }
inline bool orion_lift_add_int8(float*, const void*, const float*, const float*, int, int) { return false; }
inline bool orion_lift_add_int8_bf16w(void*, const float*, const void*, const float*, const float*, int, int) { return false; }
inline bool orion_lift_add_int8_bf16w_bf16anchor(void*, const void*, const void*, const float*, const float*, int, int) { return false; }
inline bool orion_perturb_col_int8(float*, const float*, const void*, const float*, int, int, float) { return false; }
inline bool orion_perturb_col_int8_bf16w(void*, const float*, const void*, const float*, int, int, float) { return false; }
inline bool orion_perturb_col_int8_bf16w_bf16anchor(void*, const void*, const void*, const float*, int, int, float) { return false; }
inline bool scfa_block_compress(const float*, int, int, int, float, float, float*, cudaStream_t = 0) { return false; }
inline bool scfa_causal_lag_lift(const float*, int, int, int, float, float, float*, cudaStream_t = 0) { return false; }
inline bool scfa_causal_lag_reduce(const float*, int, int, int, float, float, float*, cudaStream_t = 0) { return false; }
inline bool scfa_block_expand(const float*, int, int, int, float, float, float*, cudaStream_t = 0) { return false; }
inline bool scfa_lag_row(const float*, int, int, float*, cudaStream_t = 0) { return false; }
inline bool scfa_depthwise_causal_conv_fwd(const float*, const float*, int, int, int, float*, cudaStream_t = 0) { return false; }
inline bool scfa_depthwise_causal_conv_fwd_tiled(const float*, const float*, int, int, int, float*, cudaStream_t = 0) { return false; }
inline bool scfa_depthwise_causal_conv_fwd_sub_fused_tiled(const float*, const float*, const float*, int, int, int, float*, cudaStream_t = 0) { return false; }
inline bool scfa_depthwise_causal_conv_fwd_sub_fused_dual_out_tiled(const float*, const float*, const float*, int, int, int, float*, float*, cudaStream_t = 0) { return false; }
inline bool scfa_depthwise_causal_conv_bwd(const float*, const float*, const float*, int, int, int, float*, float*, cudaStream_t = 0) { return false; }
inline bool scfa_depthwise_causal_conv_bwd_tiled(const float*, const float*, const float*, int, int, int, float*, float*, cudaStream_t = 0) { return false; }
inline bool scfa_depthwise_causal_conv_bwd_dual_out(const float*, const float*, const float*, int, int, int, float*, float*, float*, cudaStream_t = 0) { return false; }
inline bool scfa_depthwise_causal_conv_bwd_dual_out_bf16mirror(const float*, const float*, const float*, int, int, int, float*, float*, unsigned short*, float*, cudaStream_t = 0) { return false; }
inline bool scfa_dct_basis_init(float*, int, int) { return false; }

inline bool gelu_forward(const float*, int, float*) { return false; }
inline bool gelu_backward(const float*, const float*, int, float*) { return false; }

inline bool silu_forward(const float*, int, float*) { return false; }
inline bool silu_backward(const float*, const float*, int, float*) { return false; }

inline bool relu_forward(const float*, int, float*) { return false; }
inline bool relu_backward(const float*, const float*, int, float*) { return false; }

inline bool swiglu_forward(const float*, int, int, float*) { return false; }
inline bool swiglu_backward(const float*, const float*, int, int, float*) { return false; }

inline bool rope_apply(float*, const float*, int, int, int, int = 0, bool = false) { return false; }
inline bool rope_apply_qk(float*, float*, const float*, int, int, int, int, int = 0, bool = false) { return false; }

inline bool add_bias(float*, const float*, int, int) { return false; }
inline bool add_residual(float*, const float*, int) { return false; }
inline bool add_two(float*, const float*, const float*, int) { return false; }
inline bool add_two_scaled(float*, const float*, const float*, float, int) { return false; }
inline bool axpy(float, const float*, float*, int) { return false; }
inline bool scale_array(float*, float, int) { return false; }

inline bool embedding_gather(const float*, const int*, int, int, int, float*) { return false; }
inline bool embedding_gather_bf16(const uint16_t*, const int*, int, int, int, float*) { return false; }
inline bool embedding_scatter_add(float*, const int*, const float*, int, int, int) { return false; }
inline bool embedding_scatter_add_bf16(uint16_t*, const int*, const float*, int, int, int) { return false; }
inline bool embedding_coverage_accumulate(const int*, const float*, int, int, int, float*) { return false; }
inline bool row_rms_clamp(float*, int, int, float, int*, int*) { return false; }
inline bool row_rms_clamp_vitals(float*, int, int, float, int*, int*, float*,
                                  int* = NULL, int* = NULL) { return false; }
inline bool clamp_vector_l2norm(float*, int, float, int*) { return false; }
inline bool clamp_abs(float*, int, float) { return false; }
inline bool agc_clamp_vector(float*, const float*, int, float, float, int*) { return false; }
inline bool gradient_centralize(float*, int, int) { return false; }
inline bool gradient_centralize_bf16(uint16_t*, int, int) { return false; }
inline bool spectral_norm_estimate(const float*, int, int, float*, float*, int, float*) { return false; }
inline bool spectral_normalize(float*, int, int, float*, float*, int, float, float*) { return false; }
inline bool spectral_normalize_bf16(uint16_t*, const float*, int, int, float*, float*, int, float, float*) { return false; }
inline bool sam_perturb(float*, const float*, int, float) { return false; }
inline bool sam_perturb_bf16(uint16_t*, const uint16_t*, int, float) { return false; }

inline bool adam_update(float*, const float*, float*, float*, float, float, float, float, float, float, int, int) { return false; }
inline bool sophia_g_update(float*, const float*, float*, float*, float, float, float, float, float, float, float, float, int, int) { return false; }
inline bool sophia_g_update_bf16_state(float*, const float*, uint16_t*, uint16_t*, float, float, float, float, float, float, float, float, int, int) { return false; }
inline bool adam_update_batch(float**, float**, float**, float**,
                              const float*, const float*, float, const float*,
                              const int*, int,
                              float**, float**, float**, float**, float**, float**, const int*, const int*,
                              float, float, float, float, int, int) { return false; }
inline bool sophia_g_update_batch(float**, float**, float**, float**,
                                   const float*, const float*, float,
                                   const int*, int,
                                   float, float, float, float, float, float, int, int) { return false; }

inline bool flash_attention_forward(const float*, const float*, const float*, int, int, int, bool, float*) { return false; }
inline bool flash_attention_backward(const float*, const float*, const float*, const float*, const float*, int, int, int, bool, float*, float*, float*) { return false; }
inline bool flash_attention_multihead_forward(const float*, const float*, const float*, int, int, int, int, int, int, bool, float*) { return false; }
inline bool flash_attention_multihead_forward_bf16(const unsigned short*, const unsigned short*, const unsigned short*, int, int, int, int, int, int, bool, float*) { return false; }
inline bool flash_attention_multihead_forward_bf16_local(const unsigned short*, const unsigned short*, const unsigned short*, int, int, int, int, int, int, bool, int, float*, int = 0) { return false; }
inline bool flash_attention_multihead_backward_bf16_local(const unsigned short*, const unsigned short*, const unsigned short*, const float*, const float*, int, int, int, int, int, int, bool, int, float*, float*, float*, int = 0) { return false; }
inline bool flash_attention_multihead_backward_bf16(const unsigned short*, const unsigned short*, const unsigned short*, const float*, const float*, int, int, int, int, int, int, bool, float*, float*, float*) { return false; }
inline bool flash_attention_multihead_backward(const float*, const float*, const float*, const float*, const float*, int, int, int, int, int, int, bool, float*, float*, float*) { return false; }

inline bool reduce_rows_sum(const float*, int, int, float, float*) { return false; }
inline bool causal_mask_softmax_inplace(float*, int, int) { return false; }
inline bool causal_mask_softmax_bf16_out(float*, uint16_t*, int, int) { return false; }
inline bool softmax_backward_attn(const float*, const float*, int, int, float, float*) { return false; }
inline bool causal_softmax_with_bwd_attn(float*, const float*, int, int, float, float*) { return false; }
inline bool cross_entropy_nll_loss(const float*, const int*, int, int, int, float*, int*) { return false; }
inline bool argmax_count_matches(const float*, const int*, int, int, int, int*, int*) { return false; }
inline bool chiron_vitals_output_vectors(const float*, const int*, int, int, int, float*, int*, int*, int*, float*, unsigned char*) { return false; }
inline bool chunked_cross_entropy_loss(const float*, const float*, const int*,
                                       int, int, int, int, int,
                                       float*, int*, float*) { return false; }
inline bool chunked_cross_entropy_backward(const float*, const float*, const int*,
                                           const float*, const float*,
                                           int, int, int, int, int, int, bool,
                                           float*, float*, float*) { return false; }
inline bool chunked_squared_hinge_loss(const float*, const float*, const int*, int, int,
                                       int, int, float, float*, int*, int*, float*) { return false; }
inline bool chunked_squared_hinge_backward(const float*, const int*, const int*,
                                           const float*, int, int, int, int, int,
                                           bool, float*, float*) { return false; }
inline int deterministic_dot_partial_count(int) { return 0; }
inline bool deterministic_dot_partials(const float*, const float*, int, float*, int) { return false; }
inline bool add_anchor_regularizer(const float*, const float*, int, int, float,
                                   float*, float*, int) { return false; }
inline bool project_augmented_bias_gauge(float*, int, int, float*, int) { return false; }

inline bool kv_attention_incremental(const float*, const float*, const float*, float*, const unsigned char*, int, int, int, int, int, int, float, float*) { return false; }

inline bool zero_buffers_batch(float**, const int*, int) { return false; }
inline bool pack_loss_scalars(const float*, const int*, const int*, const int*, int*) { return false; }

inline bool sum_squared_accumulate(const float*, int, float*) { return false; }
inline bool sum_squared_accumulate_dual(const float*, int, float*, float*) { return false; }
inline bool sum_squared_accumulate_bf16(const uint16_t*, int, float*) { return false; }
inline bool sum_squared_accumulate_bf16_dual(const uint16_t*, int, float*, float*) { return false; }
inline bool cast_f32_to_bf16(const float*, uint16_t*, size_t) { return false; }
inline bool cast_bf16_to_f32(const uint16_t*, float*, size_t) { return false; }
inline bool cast_f32_to_bf16_stochastic(const float*, uint16_t*, size_t, uint32_t, uint32_t) { return false; }
inline bool cast_f32_to_bf16_batched(int, const float* const*, uint16_t* const*, const size_t*) { return false; }
inline bool bf16_accum_axpy(uint16_t*, const float*, float, float, size_t) { return false; }
inline bool adam_update_int8_state(float*, const float*, int8_t*, uint8_t*,
                                    float*, float*, float, float, float, float,
                                    float, float, int, int, float* = NULL) { return false; }
inline int adam_int8_scale_count(int n) { return (n + 255) / 256; }
inline bool adam_update_bf16_state(float*, const float*, uint16_t*, uint16_t*,
                                   float, float, float, float, float, float,
                                   int, int) { return false; }
inline bool adam_update_bf16_kahan_state(float*, const float*, uint16_t*, uint16_t*,
                                          uint16_t*, float, float, float, float, float, float,
                                          int, int) { return false; }
inline bool adam_update_bf16_state_bf16grad(float*, const uint16_t*, uint16_t*, uint16_t*,
                                             float, float, float, float, float, float,
                                             int, int) { return false; }
inline bool adam_update_int8_state_bf16grad(float*, const uint16_t*, int8_t*, uint8_t*,
                                             float*, float*, float*,
                                             float, float, float, float, float, float,
                                             int, int, float* = NULL) { return false; }
inline bool adam_update_bf16_state_bf16grad_bf16w(uint16_t*, float*, const uint16_t*,
                                                   uint16_t*, uint16_t*,
                                                   float, float, float, float, float, float,
                                                   int, int, uint32_t, uint32_t) { return false; }
inline bool adam_update_int8_state_bf16grad_bf16w(uint16_t*, float*, const uint16_t*,
                                                   int8_t*, uint8_t*, float*, float*, float*,
                                                   float, float, float, float, float, float,
                                                   int, int, uint32_t, uint32_t) { return false; }
inline bool adam_update_int8_state_bf16w_bf16g_fused(uint16_t*, const uint16_t*,
                                                      int8_t*, uint8_t*, float*, float*,
                                                      float, float, float, float, float, float,
                                                      int, int, uint32_t, uint32_t,
                                                      float* = NULL) { return false; }
inline bool astra_update(float*, const float*, float*,
                          float, float, float, float, float,
                          int, int) { return false; }

inline bool phoenix_binary_gemm_gpu(const float*, const unsigned char*,
                                     int, int, int, float*) { return false; }
inline bool binary_gemm_abt_from_float(const float*, const float*,
                                        int, int, int, float*) { return false; }
inline bool binarize_to_bf16_signs(const float*, unsigned short*, size_t) { return false; }
inline bool wmma_b1_gemm(const unsigned int*, const unsigned int*,
                          int, int, int, int*) { return false; }
inline bool quantize_x_to_b1_with_scale(const float*, int, int,
                                         unsigned int*, float*) { return false; }
inline bool bitnet_b1_forward(const unsigned int*, const unsigned int*,
                               const float*, const float*, int*,
                               int, int, int, float*) { return false; }
inline bool bitnet_ffn_forward_gpu(const float*, const float*, int, int, int,
                                    unsigned int*, unsigned int*, float*, float*, int*,
                                    float*) { return false; }
inline bool mla_compute_latent_gpu(const float*, const float*,
                                    int, int, int, float*) { return false; }
inline bool mla_decompress_kv_gpu(const float*, const float*, const float*,
                                   int, int, int, float*, float*) { return false; }
inline bool mla_attention_forward_gpu(const float*, const float*, const float*, const float*,
                                       int, int, int, int, float*, float*, float*) { return false; }
inline bool mla_attention_backward_gpu(const float*, const float*, const float*, const float*,
                                        const float*, const float*, const float*,
                                        int, int, int, int,
                                        float*, float*, float*, float*, float*,
                                        unsigned short*, unsigned short*) { return false; }
inline bool sw_attention_forward_gpu(const float*, int, const float*, int,
                                      const float*, int, int, int, bool,
                                      int, int, float*, int) { return false; }
inline bool sw_attention_backward_gpu(const float*, int, const float*, int,
                                       const float*, int, const float*, int,
                                       int, int, bool, int, int,
                                       float*, int, float*, int, float*, int) { return false; }

inline void device_memcpy_d2d(void*, const void*, size_t) {}
inline void device_memcpy_h2d(void*, const void*, size_t) {}
inline void device_memcpy_d2h(void*, const void*, size_t) {}
inline void device_memcpy_2d_d2d(void*, size_t, const void*, size_t, size_t, size_t) {}
inline void device_memset_bytes(void*, int, size_t) {}

inline bool qknorm_forward_gpu(float*, float*, int, int, int, float) { return false; }
inline bool qknorm_backward_gpu(const float*, const float*, const float*, int, int, int, float*) { return false; }
inline bool scale_q_per_head(float*, const float*, int, int, int) { return false; }
inline bool qknorm_gamma_grad(const float*, const float*, float, int, int, int, float*) { return false; }
inline bool qknorm_gamma_scale_gpu(const float*, float, float*, int) { return false; }

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
