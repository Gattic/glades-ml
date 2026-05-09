// Custom CUDA kernel declarations for Glades ML.
//
// Provides host wrapper functions for normalization, activation, attention,
// embedding, optimizer, and utility kernels.  When GLADES_HAVE_CUDA is not
// defined the wrappers degrade to inline no-ops / stubs that return false.
#pragma once

#include <cstddef>
#include <stdint.h>

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

// Fused softmax-cross-entropy backward: dlogits = probs - one_hot(targets).
bool softmax_cross_entropy_bwd(const float* probs, const int* targets,
                               int rows, int cols, float* dlogits);

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

// Backward: dE[tokenIds[T], :] += dout[T, dModel].
bool embedding_scatter_add(float* dE, const int* tokenIds,
                           const float* dout,
                           int T, int vocabSize, int dModel);

// ---------------------------------------------------------------------------
// Adam optimizer
// ---------------------------------------------------------------------------

// In-place Adam update for n parameters.
// gradScale is applied to grad before the update (use 1.0f for no scaling).
bool adam_update(float* param, const float* grad, float* m, float* v,
                 float lr, float beta1, float beta2, float eps,
                 float weightDecay, float gradScale, int step, int n);

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
                             int step, int n);

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
bool flash_attention_multihead_forward_bf16_local(const uint16_t* Q, const uint16_t* K,
                                                   const uint16_t* V,
                                                   int T, int nHeads, int nKVHeads,
                                                   int dHead, int dModel, int dModelKV,
                                                   bool causal, int windowSize,
                                                   float* O);

// Local-window BF16 flash attention backward.  Same windowing rules as the
// forward kernel; dK, dV accumulated via atomicAdd only on in-window keys.
// windowSize <= 0 or >= T falls back to the full backward.
bool flash_attention_multihead_backward_bf16_local(
    const uint16_t* Q, const uint16_t* K, const uint16_t* V,
    const float* O, const float* dO,
    int T, int nHeads, int nKVHeads,
    int dHead, int dModel, int dModelKV,
    bool causal, int windowSize,
    float* dQ, float* dK_out, float* dV_out);

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

// #76 MLA latent compression: c = h @ W_DKV via cuBLAS sgemm.
bool mla_compute_latent_gpu(const float* h, const float* W_DKV,
                             int T, int d_h, int d_c, float* c_out);

// #76 MLA KV decompression: K = c @ W_UK, V = c @ W_UV via cuBLAS.
bool mla_decompress_kv_gpu(const float* c,
                            const float* W_UK, const float* W_UV,
                            int T, int d_c, int dKVtotal,
                            float* K_out, float* V_out);

// #78 ATTENTION-SINK forward (single-head FP32 reference). One block per
// query position; inner thread reduces over keys with sink+window mask.
bool sw_attention_forward_gpu(const float* Q, int qStride,
                               const float* K, int kStride,
                               const float* V, int vStride,
                               int T, int dHead, bool causal,
                               int sinkCount, int windowSize,
                               float* O, int oStride);

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

// Softmax backward for attention: dS = outputScale * P * (dP - row_sum(dP * P)),
// zero above-diagonal for causal mask.
// P, dP, dS are [batchSize, T, T] row-major.
bool softmax_backward_attn(const float* P, const float* dP,
                           int batchSize, int T, float outputScale, float* dS);

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

} // namespace gpu
} // namespace glades

#else // !GLADES_HAVE_CUDA --------------------------------------------------

namespace glades {
namespace gpu {

inline bool layernorm_forward(const float*, const float*, const float*, float, int, int, float*, float*, float*) { return false; }
inline bool layernorm_backward(const float*, const float*, const float*, const float*, const float*, int, int, float*, float*, float*) { return false; }

inline bool rmsnorm_forward(const float*, const float*, float, int, int, float*, float*) { return false; }
inline bool rmsnorm_backward(const float*, const float*, const float*, const float*, int, int, float*, float*) { return false; }

inline bool softmax_forward(const float*, int, int, float*) { return false; }
inline bool softmax_cross_entropy_bwd(const float*, const int*, int, int, float*) { return false; }

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
inline bool axpy(float, const float*, float*, int) { return false; }
inline bool scale_array(float*, float, int) { return false; }

inline bool embedding_gather(const float*, const int*, int, int, int, float*) { return false; }
inline bool embedding_scatter_add(float*, const int*, const float*, int, int, int) { return false; }

inline bool adam_update(float*, const float*, float*, float*, float, float, float, float, float, float, int, int) { return false; }
inline bool adam_update_batch(float**, float**, float**, float**,
                              const float*, const float*, float, const float*,
                              const int*, int,
                              float**, float**, float**, float**, float**, float**, const int*, const int*,
                              float, float, float, float, int, int) { return false; }

inline bool flash_attention_forward(const float*, const float*, const float*, int, int, int, bool, float*) { return false; }
inline bool flash_attention_backward(const float*, const float*, const float*, const float*, const float*, int, int, int, bool, float*, float*, float*) { return false; }
inline bool flash_attention_multihead_forward(const float*, const float*, const float*, int, int, int, int, int, int, bool, float*) { return false; }
inline bool flash_attention_multihead_forward_bf16(const unsigned short*, const unsigned short*, const unsigned short*, int, int, int, int, int, int, bool, float*) { return false; }
inline bool flash_attention_multihead_forward_bf16_local(const unsigned short*, const unsigned short*, const unsigned short*, int, int, int, int, int, int, bool, int, float*) { return false; }
inline bool flash_attention_multihead_backward_bf16_local(const unsigned short*, const unsigned short*, const unsigned short*, const float*, const float*, int, int, int, int, int, int, bool, int, float*, float*, float*) { return false; }
inline bool flash_attention_multihead_backward_bf16(const unsigned short*, const unsigned short*, const unsigned short*, const float*, const float*, int, int, int, int, int, int, bool, float*, float*, float*) { return false; }
inline bool flash_attention_multihead_backward(const float*, const float*, const float*, const float*, const float*, int, int, int, int, int, int, bool, float*, float*, float*) { return false; }

inline bool reduce_rows_sum(const float*, int, int, float, float*) { return false; }
inline bool causal_mask_softmax_inplace(float*, int, int) { return false; }
inline bool softmax_backward_attn(const float*, const float*, int, int, float, float*) { return false; }
inline bool cross_entropy_nll_loss(const float*, const int*, int, int, int, float*, int*) { return false; }
inline bool argmax_count_matches(const float*, const int*, int, int, int, int*, int*) { return false; }
inline bool chunked_cross_entropy_loss(const float*, const float*, const int*,
                                       int, int, int, int, int,
                                       float*, int*, float*) { return false; }
inline bool chunked_cross_entropy_backward(const float*, const float*, const int*,
                                           const float*, const float*,
                                           int, int, int, int, int, int, bool,
                                           float*, float*, float*) { return false; }

inline bool kv_attention_incremental(const float*, const float*, const float*, float*, const unsigned char*, int, int, int, int, int, int, float, float*) { return false; }

inline bool zero_buffers_batch(float**, const int*, int) { return false; }
inline bool pack_loss_scalars(const float*, const int*, const int*, const int*, int*) { return false; }

inline bool sum_squared_accumulate(const float*, int, float*) { return false; }
inline bool cast_f32_to_bf16(const float*, uint16_t*, size_t) { return false; }
inline bool cast_bf16_to_f32(const uint16_t*, float*, size_t) { return false; }
inline bool cast_f32_to_bf16_stochastic(const float*, uint16_t*, size_t, uint32_t, uint32_t) { return false; }
inline bool bf16_accum_axpy(uint16_t*, const float*, float, float, size_t) { return false; }
inline bool adam_update_int8_state(float*, const float*, int8_t*, uint8_t*,
                                    float*, float*, float, float, float, float,
                                    float, float, int, int) { return false; }
inline int adam_int8_scale_count(int n) { return (n + 255) / 256; }
inline bool adam_update_bf16_state(float*, const float*, uint16_t*, uint16_t*,
                                   float, float, float, float, float, float,
                                   int, int) { return false; }
inline bool adam_update_bf16_kahan_state(float*, const float*, uint16_t*, uint16_t*,
                                          uint16_t*, float, float, float, float, float, float,
                                          int, int) { return false; }
inline bool astra_update(float*, const float*, float*,
                          float, float, float, float, float,
                          int, int) { return false; }

inline bool phoenix_binary_gemm_gpu(const float*, const unsigned char*,
                                     int, int, int, float*) { return false; }
inline bool mla_compute_latent_gpu(const float*, const float*,
                                    int, int, int, float*) { return false; }
inline bool mla_decompress_kv_gpu(const float*, const float*, const float*,
                                   int, int, int, float*, float*) { return false; }
inline bool sw_attention_forward_gpu(const float*, int, const float*, int,
                                      const float*, int, int, int, bool,
                                      int, int, float*, int) { return false; }

inline void device_memcpy_d2d(void*, const void*, size_t) {}
inline void device_memcpy_h2d(void*, const void*, size_t) {}
inline void device_memcpy_d2h(void*, const void*, size_t) {}
inline void device_memcpy_2d_d2d(void*, size_t, const void*, size_t, size_t, size_t) {}
inline void device_memset_bytes(void*, int, size_t) {}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
