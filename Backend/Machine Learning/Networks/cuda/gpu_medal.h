// MEDAL (Measure-Evolution Denoising Autoregression-free Language model)
// GPU primitives — paradigm #262.  Iter 43 (2026-05-16): initial GPU port.
//
// Design: see research/PARADIGM_SHIFT_262_MEDAL_DESIGN.md.
//
// Forward (training):
//   1. Sample t ~ Uniform[eps, 1-eps] per microbatch (host-side scalar).
//      alpha(t) := t  (linear schedule).
//   2. medal_corrupt_tokens(d_tokens, T, V_mask, alpha,
//                           d_tokens_corr, d_mask, seed)
//        For each position i:
//          if rand() < alpha:  d_tokens_corr[i] = V_mask; d_mask[i] = 1
//          else:               d_tokens_corr[i] = d_tokens[i]; d_mask[i] = 0
//   3. Standard embedding_gather(E[V+1, m], d_tokens_corr, T, V+1, m, q).
//      The MASK token V indexes the (V+1)-th row of the augmented embedding.
//   4. Standard transformer forward with causal=false (bidirectional denoiser).
//   5. Tied readout: logits[T, V] = q · E[0:V, :]^T  (predict over real vocab).
//   6. softmax → probs → softmax_cross_entropy_bwd with d_targets = d_tokens
//      (the ORIGINAL uncorrupted tokens — the denoiser learns to reconstruct).
//   7. medal_mask_dlogits(dlogits, d_mask, T, V):  zeros out dlogits[i, :]
//      where d_mask[i] == 0 (only masked positions contribute to loss).
//   8. Scale by 1 / (sum(d_mask) * alpha) for ELBO weighting (host scalar).
//   9. Continue standard backward.
//
// The mechanism is no-op at init only in the sense that the (V+1)-th
// embedding row can be zero-init and the model recovers AR if MEDAL is
// disabled (no corruption fired).  Train-from-scratch is the supported
// path; cf. paradigm #261 HMTA's no-op-at-init for trained-baseline
// augmentation.

#pragma once

#include <cstdint>

#ifdef GLADES_HAVE_CUDA
namespace glades {
namespace gpu {

// Per-position absorbing-mask CTMC corruption sampler.
//
// Inputs:
//   d_tokens    : [T] int32 — original token IDs (0..V-1).
//   T           : sequence length.
//   V_mask      : int — index of the MASK token in the extended vocab
//                       (typically V; the embedding table is [V+1, m]).
//   alpha       : float — Pr[mask] per position, in (0, 1).
//   seed        : uint64_t — Philox4 counter base for cuRAND.
//
// Outputs:
//   d_tokens_corr : [T] int32 — corrupted token IDs.  Each position is
//                                independently set to V_mask with prob alpha
//                                or left unchanged with prob 1-alpha.
//   d_mask        : [T] uint8 — 1 if position was corrupted (masked), 0 else.
//
// Implementation: one block per chunk of T positions; one cuRAND state per
// thread; deterministic given (seed, T).
//
// Returns true on success, false on CUDA launch failure.
bool medal_corrupt_tokens(const int* d_tokens, int T, int V_mask,
                          float alpha, uint64_t seed,
                          int* d_tokens_corr, unsigned char* d_mask);

// Zero out dlogits[i, :] for positions where d_mask[i] == 0.  In-place.
//
// Inputs:
//   d_mask   : [T] uint8 — corruption mask from medal_corrupt_tokens.
//   T, V     : logit-tensor shape (logits over real vocab only).
// Inout:
//   d_logits : [T, V] float — dlogits buffer; rows where mask==0 are zeroed.
//
// This is the masked-CE backward modification: only positions that were
// MASKED contribute to the gradient.  Unmasked positions have already been
// "seen" by the denoiser as their true value, so they trivially get 0 loss
// at the optimum and we don't gradient-train against them.
//
// Returns true on success.
bool medal_mask_dlogits(float* d_logits, const unsigned char* d_mask,
                        int T, int V);

// Same but for BF16-storage backward path.
bool medal_mask_dlogits_bf16(uint16_t* d_logits_bf, const unsigned char* d_mask,
                             int T, int V);

// Forward NLL accumulation with mask.  Computes sum_{i: mask[i]==1} -log
// p[i, targets[i]].  Used for forward-only ELBO reporting during val.
//
// Inputs:
//   d_probs    : [T, V] float — softmax probabilities.
//   d_targets  : [T] int32 — true token IDs.
//   d_mask     : [T] uint8 — corruption mask.
//   T, V       : shape.
// Output:
//   d_nll_sum  : [1] float — accumulated NLL.
//   d_n_masked : [1] int  — count of masked positions.
//
// Returns true on success.
bool medal_masked_nll(const float* d_probs, const int* d_targets,
                      const unsigned char* d_mask, int T, int V,
                      float* d_nll_sum, int* d_n_masked);

} // namespace gpu
} // namespace glades
#else
// CPU-only build: stubs that always return false.
namespace glades { namespace gpu {
inline bool medal_corrupt_tokens(const int*, int, int, float, uint64_t,
                                 int*, unsigned char*) { return false; }
inline bool medal_mask_dlogits(float*, const unsigned char*, int, int) { return false; }
inline bool medal_mask_dlogits_bf16(uint16_t*, const unsigned char*, int, int) { return false; }
inline bool medal_masked_nll(const float*, const int*, const unsigned char*,
                             int, int, float*, int*) { return false; }
}}
#endif
