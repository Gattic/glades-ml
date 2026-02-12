// Custom CUDA kernel declarations for Glades ML.
//
// Provides host wrapper functions for normalization, activation, attention,
// embedding, optimizer, and utility kernels.  When GLADES_HAVE_CUDA is not
// defined the wrappers degrade to inline no-ops / stubs that return false.
#pragma once

#include <cstddef>

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

// ---------------------------------------------------------------------------
// Simple vector ops
// ---------------------------------------------------------------------------

// out[rows, cols] += bias[cols]   (broadcast add bias to each row)
bool add_bias(float* out, const float* bias, int rows, int cols);

// out[n] += residual[n]
bool add_residual(float* out, const float* residual, int n);

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
bool adam_update(float* param, const float* grad, float* m, float* v,
                 float lr, float beta1, float beta2, float eps,
                 float weightDecay, int step, int n);

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

// Softmax backward for attention: dS = P * (dP - row_sum(dP * P)),
// zero above-diagonal for causal mask.
// P, dP, dS are [batchSize, T, T] row-major.
bool softmax_backward_attn(const float* P, const float* dP,
                           int batchSize, int T, float* dS);

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
// Device memory operations (callable from .cpp files without cuda_runtime.h)
// ---------------------------------------------------------------------------

void device_memcpy_d2d(void* dst, const void* src, size_t bytes);
void device_memcpy_h2d(void* dst, const void* src, size_t bytes);
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

inline bool add_bias(float*, const float*, int, int) { return false; }
inline bool add_residual(float*, const float*, int) { return false; }
inline bool axpy(float, const float*, float*, int) { return false; }
inline bool scale_array(float*, float, int) { return false; }

inline bool embedding_gather(const float*, const int*, int, int, int, float*) { return false; }
inline bool embedding_scatter_add(float*, const int*, const float*, int, int, int) { return false; }

inline bool adam_update(float*, const float*, float*, float*, float, float, float, float, float, int, int) { return false; }

inline bool flash_attention_forward(const float*, const float*, const float*, int, int, int, bool, float*) { return false; }
inline bool flash_attention_backward(const float*, const float*, const float*, const float*, const float*, int, int, int, bool, float*, float*, float*) { return false; }

inline bool reduce_rows_sum(const float*, int, int, float, float*) { return false; }
inline bool causal_mask_softmax_inplace(float*, int, int) { return false; }
inline bool softmax_backward_attn(const float*, const float*, int, int, float*) { return false; }
inline bool cross_entropy_nll_loss(const float*, const int*, int, int, int, float*, int*) { return false; }
inline bool argmax_count_matches(const float*, const int*, int, int, int, int*, int*) { return false; }

inline bool kv_attention_incremental(const float*, const float*, const float*, float*, const unsigned char*, int, int, int, int, int, int, float, float*) { return false; }

inline void device_memcpy_d2d(void*, const void*, size_t) {}
inline void device_memcpy_h2d(void*, const void*, size_t) {}
inline void device_memcpy_2d_d2d(void*, size_t, const void*, size_t, size_t, size_t) {}
inline void device_memset_bytes(void*, int, size_t) {}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
