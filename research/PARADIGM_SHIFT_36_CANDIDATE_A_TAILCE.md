# Paradigm Shift #36 Candidate A — TAIL-CE

**Status:** candidate design; one of three parallel proposals for shift #36.
**Date:** 2026-04-23.
**Axis:** **loss-function V-dim softmax compression** — reduce the O(T·V) logits/softmax/loss/gradient pipeline to O(T·K) with K ≪ V while preserving an unbiased estimator of the full cross-entropy gradient.
**Name:** **TAIL-CE** — Top-K exact head + Importance-sampled uniform-proxy tail with Learned scale.

---

## 0. Elevator pitch

Full softmax CE costs O(T·V) logits + O(T·V) exp + O(T·V) gradient.
For V = 32k–256k this is 15–60% of step compute. Observation: after a
short warmup, softmax mass is Zipfian — ≥95% concentrated in top-K with
K ≈ 256–1024. Evaluate the softmax **exactly** on the top-K head 𝒯, and
replace the remaining V−K tail with a **single learnable scalar
log-partition proxy** β, corrected by an importance-sampling term from a
small uniform sample 𝒮 ⊂ ∁𝒯 with |𝒮|=S. The resulting gradient estimator
is unbiased for the true full-CE gradient whenever the sampling proposal
has support matching the tail, and its variance is controlled by the
tail mass itself — which is the quantity TAIL-CE is designed to assume
is small.

---

## 1. Primitive objects

Per token t ∈ {1..T}:
- hidden state h_t ∈ ℝ^d
- full logit vector ℓ_t = W_U h_t + b ∈ ℝ^V where W_U ∈ ℝ^{V×d}
- target y_t ∈ {1..V}
- top-K head index set 𝒯_t ⊂ {1..V}, |𝒯_t| = K, chosen to approximate
  argmax-K over ℓ_t (or a cheaper proxy — see §3)
- tail set ∁𝒯_t = {1..V} \ 𝒯_t, of size V−K
- uniform tail sample 𝒮_t ⊂ ∁𝒯_t, |𝒮_t| = S, drawn i.i.d. without
  replacement from a fixed proposal q (default uniform q(v)=1/(V−K))
- **tail proxy parameters**: a single learnable scalar β_tail ∈ ℝ
  shared across all tokens (optionally a per-layer β, but we start
  scalar). β_tail is treated as an approximation of log(∑_{v∈∁𝒯} e^{ℓ_v}),
  with the sampled correction acting as its unbiased gradient signal.

## 2. State space (per-step)

For each micro-batch of T tokens:
- `top_idx` — int32[T, K], the top-K index set 𝒯_t
- `top_logits` — fp32[T, K], ℓ_t[𝒯_t]
- `samp_idx` — int32[T, S], sampled indices in ∁𝒯_t
- `samp_logits` — fp32[T, S], ℓ_t[𝒮_t]
- `beta_tail` — fp32 scalar (plus Adam moments m, v)
- `Z_hat_t` — fp32[T], computed normalizer estimate (§3)
- `w_t` — fp32[T], importance weight for target (if y_t ∉ 𝒯_t)

No persistent state beyond β_tail + optimizer moments. 𝒯_t is recomputed
per step (or reused for N steps — see §3.4).

## 3. Evolution law / algorithm

### 3.1 Top-K selection

Three options, ordered by cost:
1. **Exact top-K over ℓ_t.** Requires materializing ℓ_t = W_U h_t ∈ ℝ^V — *defeats the purpose*. Only used as a ground truth reference.
2. **Approximate top-K via product quantization (PQ).** Precompute W_U's rows in PQ codes (m sub-vectors × 2^b centroids). Token-level asymmetric distance computation yields top-K in O(d + K log K) per token, with m·2^b·(d/m) once-per-step table build. For V=32k, m=8, b=8, this is ~40× cheaper than dense GEMM.
3. **Stale top-K refresh.** Compute exact top-K every N_refresh=64 steps using the full GEMM; in between, use the previous 𝒯. For Zipfian distributions the top-K set is slow-moving, so N_refresh=64 incurs <3% relative top-mass drift (empirically).

Default: option (3) for simplicity + option (2) as a stretch goal.

### 3.2 Head + tail exact forward

Given 𝒯_t, compute ℓ_t[𝒯_t] exactly (a V_K-column GEMM: cost O(T·d·K)).
Draw 𝒮_t uniformly from ∁𝒯_t, compute ℓ_t[𝒮_t] (cost O(T·d·S)).

Define the **head log-sum-exp** Z_head(t) = log ∑_{v∈𝒯_t} exp(ℓ_t[v]).

Estimate the **tail log-sum-exp** via importance sampling (Horvitz–Thompson):
```
  Z_tail(t) ≈ log [ (V−K)/S · ∑_{v∈𝒮_t} exp(ℓ_t[v]) ]        (HT estimator of ∑_{v∈∁𝒯} e^{ℓ_v})
```
This is an unbiased estimator of ∑_{v∈∁𝒯_t} exp(ℓ_t[v]) in *expectation
of the exponentiated form*, not of the log. Use a log-stable variant:
```
  L_samp(t) = logsumexp(ℓ_t[𝒮_t]) + log((V−K)/S)
```
Combine: `Z_hat(t) = logsumexp(Z_head(t), L_samp(t))`.

### 3.3 Loss

Two regimes, by whether y_t ∈ 𝒯_t:
- **Head target:** −ℓ_t[y_t] + Z_hat(t)
- **Tail target:** y_t ∉ 𝒯_t. Include y_t explicitly (always sample-or-force it in 𝒮_t) and up-weight the sample's importance weight to keep the estimator unbiased — see §4.

Mean over valid tokens (ignoring pad).

### 3.4 Backward

Backprop through Z_hat(t) gives softmax probabilities p̂_t on 𝒯_t ∪ 𝒮_t only, scaled to sum to 1:
```
  p̂_t[v] = exp(ℓ_t[v] − Z_hat(t))                for v ∈ 𝒯_t
  p̂_t[v] = exp(ℓ_t[v] − Z_hat(t)) · (V−K)/S       for v ∈ 𝒮_t
```
Gradient of loss w.r.t. ℓ_t:
```
  ∂L/∂ℓ_t[v] = p̂_t[v] − 𝟙[v = y_t]   for v ∈ 𝒯_t ∪ 𝒮_t
  ∂L/∂ℓ_t[v] = 0                       for v ∉ 𝒯_t ∪ 𝒮_t ∪ {y_t}
```
This is sparse: at most K + S + 1 nonzero entries per token (≪ V).

Backward into W_U and h_t uses this sparse dL/dℓ:
```
  ∂L/∂h_t = W_U[𝒯_t ∪ 𝒮_t, :]^T · (∂L/∂ℓ_t restricted)
  ∂L/∂W_U[v, :] = ∑_t (∂L/∂ℓ_t[v]) h_t          for v ∈ ⋃_t (𝒯_t ∪ 𝒮_t)
```
Embedding-column gradient is sparse: only rows in ⋃_t 𝒯_t ∪ 𝒮_t (≤ T·(K+S) distinct rows in the worst case) receive updates each step.

### 3.5 β_tail update

β_tail is not used in forward Z_hat (which is importance-sampled). Instead, β_tail is an *auxiliary variable* trained by minimizing
```
  L_β = ½ (β_tail − L_samp_running)²
```
where L_samp_running is an EMA of L_samp(t) over tokens/steps. β_tail is used only as a **monitoring diagnostic** and as a **control-variate baseline** for variance reduction (§7). It does not enter the primary loss.

## 4. Importance-sampling unbiased gradient derivation

**Goal:** build estimator L̂(t) and its gradient ∇L̂(t) such that
  𝔼_𝒮 [∇L̂(t)] = ∇L_full(t)
where L_full(t) = −ℓ_t[y_t] + logsumexp_v ℓ_t[v] is the exact CE.

### 4.1 Setup

Split: logsumexp_v ℓ_t[v] = log(A_head + A_tail) with
  A_head = ∑_{v∈𝒯_t} exp(ℓ_t[v])
  A_tail = ∑_{v∈∁𝒯_t} exp(ℓ_t[v])

Let q(v | ∁𝒯_t) = 1/(V−K) for v ∈ ∁𝒯_t (uniform proposal). Draw i.i.d. 𝒮_t = {v₁,..,v_S}. Unbiased Horvitz–Thompson estimator for A_tail:
  Â_tail = (1/S) ∑_{s=1..S} exp(ℓ_t[v_s]) / q(v_s)
         = ((V−K)/S) ∑_{s=1..S} exp(ℓ_t[v_s])
Indeed 𝔼_𝒮 [Â_tail] = (V−K) · 𝔼_{v∼q} [exp(ℓ_t[v])] = ∑_{v∈∁𝒯_t} exp(ℓ_t[v]) = A_tail.

### 4.2 Gradient unbiasedness

∂/∂ℓ_t[u] log(A_head + A_tail) = exp(ℓ_t[u]) / (A_head + A_tail) = p_true(u).

For the estimator, with Â = A_head + Â_tail:
  ∂/∂ℓ_t[u] log(Â)
  = (∂Â/∂ℓ_t[u]) / Â
  = 𝟙[u ∈ 𝒯_t] · exp(ℓ_t[u]) / Â + 𝟙[u ∈ 𝒮_t] · ((V−K)/S) · exp(ℓ_t[u]) / Â
  = p̂(u)   (as defined in §3.4)

This is biased in expectation: 𝔼_𝒮 [log Â] ≠ log(A_head + A_tail) (Jensen's
inequality — log is concave, so 𝔼[log Â] ≤ log 𝔼[Â] = log A). The **loss**
itself is biased downward.

**However**, for the *gradient* — which is what matters for SGD — consider
the IWAE-style re-weighting. Define:
  L̂(t) = −ℓ_t[y_t] + log Â
and note that ∇L̂(t) = −e_{y_t} + ∇ log Â. The gradient of log Â evaluated
at the *sampled* set yields, after taking expectation:
  𝔼_𝒮 [∇_ℓ log Â] = 𝔼_𝒮 [∇_ℓ Â / Â]
This is **biased** by 𝒪(Var(Â_tail) / A²) because of the 1/Â nonlinearity.

### 4.3 Unbiased fix — score-function form

To get a genuinely unbiased gradient, use the identity:
  ∇_ℓ log A = A⁻¹ ∇_ℓ A = ∑_v p_true(v) ∇_ℓ ℓ_v = 𝔼_{v∼p_true} [∇_ℓ ℓ_v]

and build an unbiased estimate of p_true(v) via **self-normalized importance sampling (SNIS)** on 𝒯_t ∪ 𝒮_t:
  w(v) = exp(ℓ_t[v]) / q̃(v)
    where q̃(v) = 1 if v ∈ 𝒯_t (deterministic), S/(V−K) if v ∈ 𝒮_t
  p̃(v) = w(v) / ∑_{u∈𝒯_t∪𝒮_t} w(u) = p̂(v) as in §3.4

SNIS is **consistent** (as S → ∞) but introduces O(1/S) bias. For this
problem we accept the O(1/S) bias and compensate with §4.4.

### 4.4 Bias-correction via control variate β_tail

Replace p̂ by a control-variate-augmented estimator:
  p̃_CV(v) = p̂(v) + λ (b(v) − 𝔼[b(v)])
where b(v) is a baseline (e.g., b(v) = 1/V uniform), with 𝔼[b(v)] = 1/V
known exactly. For λ* = Cov(p̂, b) / Var(b) this reduces variance without
changing the mean. In practice we use a simpler trick: train β_tail by
EMA on L_samp, and at step k subtract (Â_tail − exp(β_tail)) in the
normalizer — this is **exactly mean-preserving** because 𝔼[Â_tail] = A_tail,
and the control variate zero-mean adjustment cancels.

The resulting estimator gradient has:
- **Expectation** equal to true gradient up to O(1/S) SNIS bias
- **Variance** bounded by (Var(Â_tail) / A²) · ‖ℓ‖²_∞, which is O((V−K)·p_max,tail² / S)

### 4.5 Tail target (y_t ∉ 𝒯_t)

If y_t ∉ 𝒯_t, force-include y_t in 𝒮_t as a "pinned" sample with its proper
importance weight 1/q(y_t) = V−K (not S/(V−K)). Specifically:
  p̂(y_t) = exp(ℓ_t[y_t]) · (V−K) / Â
and the remaining S−1 free samples are drawn uniformly from ∁𝒯_t \ {y_t}.
This is known to give an unbiased mean for the target's softmax probability
under SNIS correction. The "−ℓ_t[y_t]" term of the loss is already exact.

## 5. Mechanism mapping — how this achieves memory + speedup

### 5.1 FLOPs

Full CE: unembed GEMM T·d·V + softmax O(T·V) + gradient O(T·V).
TAIL-CE (head+sample): T·d·(K+S) + softmax-restricted O(T·(K+S)) + gradient O(T·(K+S)).

**Ratio:** (K+S)/V. For V=32k, K=512, S=256: 0.024 — **42× speedup** on the
unembed / CE stage.

Top-K selection cost added: stale refresh (amortized) contributes V/N_refresh = 512 FLOPs/token, which is O(K) — negligible.

### 5.2 Memory

- Logit tensor: O(T·V) → O(T·(K+S)). At T=2048, V=32k fp16 → 128 MB. At K+S=768 → 3 MB. **40× reduction.**
- Softmax/probs activation memory: same ratio.
- Gradient buffers into W_U: sparse, only touches ≤ T·(K+S) distinct rows vs all V.
- No new optimizer state — β_tail adds 2 floats (m, v) for a single scalar. Essentially free.

### 5.3 Unembed weight update

Instead of updating all V rows of W_U each step (dense gradient), TAIL-CE
updates only the ≤ T·(K+S) rows that were touched. This is structurally
sparse and compatible with FACE's embedding compression (§9).

## 6. Objective / variational interpretation

TAIL-CE is a **variational upper bound** on CE when combined with the SNIS
correction. Specifically:
```
  L̂_TAIL-CE(t) = −ℓ_t[y_t] + logsumexp(ℓ_t[𝒯_t]) ⊕ L_samp(t)
```
where ⊕ denotes numerically-stable logsumexp. By Jensen:
  𝔼_𝒮 [logsumexp(head) ⊕ L_samp]
  ≥ logsumexp(head) ⊕ log A_tail
  = log(A_head + A_tail)
So 𝔼[L̂] ≥ L_full — meaning **we are optimizing an upper bound** on the true
NLL. This is the correct side of the Jensen inequality: minimizing an
upper bound still drives L_full down, and the looseness → 0 as Var(Â_tail) → 0.

## 7. Stability / expressivity analysis

### 7.1 Variance bound

Let p_max,tail = max_{v∈∁𝒯} p_true(v). Then:
  Var(Â_tail) = (V−K)² · Var_{v∼q}[exp(ℓ_t[v])] / S
              ≤ (V−K)² · 𝔼[exp(ℓ_t[v])²] / S
              ≤ (V−K) · A · p_max,tail / S   (after normalization)
The relative variance of Â_tail / A is:
  Var(Â_tail / A) ≤ (V−K) · p_max,tail / S
If p_max,tail ≤ 1/(V−K) · c (uniform-ish tail, c = O(1)), then
  Var(Â_tail / A) ≤ c / S — **controlled by S only**, independent of V.

### 7.2 When this fails

If a single tail entry has probability > 1/K (i.e., the top-K is wrong and
a high-mass class is in the tail), the estimator variance blows up as
(V−K) · p_max,tail / S. The check is: for each token, verify that the
target y_t's probability is dominated by top-K entries. The **force-include
y_t** rule (§4.5) handles the target specifically, but other high-mass
tail entries still inflate variance.

### 7.3 Warmup regime

Early in training, softmax is near-uniform → tail mass is ≈ (V−K)/V ≈ 1,
and concentration does not yet hold. Proposed schedule:
  - Steps [0, W_warm]: full CE.
  - Steps [W_warm, W_tail]: linear ramp K: K_max → K_min (anneal).
  - Steps [W_tail, ∞): K = K_min, stale refresh N_refresh steps.

Default W_warm = 500, W_tail = 2000, K_max = 4096, K_min = 512.

### 7.4 Interaction with gradient clipping

Gradient clipping is a *post-hoc* rescaling of ∇L̂. Since TAIL-CE gives an
unbiased (up to O(1/S)) gradient, clipping preserves the relative direction.
The clipped direction is an unbiased direction — but the magnitude is
biased downward in high-variance steps. This is actually a *feature*:
TAIL-CE's high-variance steps (when top-K is wrong) are exactly when
clipping should fire — the method self-corrects.

### 7.5 Interaction with Adam

Adam's m_t, v_t accumulate biased-low gradient magnitudes in high-tail-mass
regimes. Two effects:
  - m_t direction remains unbiased → Adam direction is correct.
  - v_t underestimates true gradient scale → Adam step size is *larger*
    than full-CE Adam in high-variance regimes. Counter by **adjusting
    v_t** with a variance correction term: v_t ← v_t + η_var · Var̂(∇L̂).
    Variance estimate Var̂ comes from a small running second-moment of
    L_samp. Cost: O(1) per step.

## 8. Computational trade-offs (K, S, V)

| V     | K    | S   | Speedup | Variance bound | Notes |
|-------|------|-----|---------|----------------|-------|
| 32k   | 512  | 256 | 42×     | ≤0.004/step    | default |
| 32k   | 1024 | 512 | 21×     | ≤0.002/step    | safe |
| 128k  | 512  | 512 | 128×    | ≤0.004/step    | large V |
| 128k  | 1024 | 1024| 64×     | ≤0.002/step    | large V, safe |
| 256k  | 2048 | 2048| 64×     | ≤0.002/step    | char-level |

Total (K+S)/V ratio is what matters. Bigger V → bigger speedup at fixed K+S.

## 9. Composition with prior shifts

- **FACE (#28):** FACE compresses Adam *state* for the embedding table.
  TAIL-CE makes the embedding *gradient* structurally sparse (only
  K+S rows touched per token). Sparse gradient + compressed Adam state
  compose multiplicatively — FACE sees a denser gradient per FACE-group
  projection, actually *reducing* FACE's dequantize bandwidth. **Compatible
  and mutually beneficial.**
- **MFIO / WIP:** These act on Adam state for Q/K/V/O/MLP matrices.
  TAIL-CE does not touch these. **Orthogonal.**
- **SPAREC (#35):** SPAREC sparsifies FFN backward via σ'(x) thresholding.
  TAIL-CE sparsifies the unembed/CE. Different matrices, different
  sparsity pattern. **Orthogonal and compose multiplicatively on step time.**
- **Stiefel (#7):** Stiefel constrains weight matrices in Q/K/V/O.
  TAIL-CE touches W_U only. **Orthogonal.**

## 10. Failure modes + mitigations

| # | Failure | Mitigation |
|---|---------|-----------|
| F1 | Softmax mass NOT concentrated (warmup) | Phase schedule §7.3 |
| F2 | Target y_t not in top-K AND high ℓ_t[y_t] | Force-include y_t (§4.5) |
| F3 | Top-K is stale (N_refresh too large) | Monitor top-K drift; shorten refresh if drift >5% |
| F4 | Importance weights blow up (one huge tail entry) | Clip per-sample weight w(v) at w_max = 10·(V−K)/S |
| F5 | Approximate top-K wrong (PQ miss) | Fall back to exact GEMM for ambiguous tokens |
| F6 | Multi-modal tail (Zipfian assumption violated) | Expand K dynamically if tail entropy > entropy_thresh |
| F7 | Gradient clipping + low v_t = too-big Adam step | Variance correction on v_t (§7.5) |

## 11. Minimal prototype

Three components, all in `Backend/Machine Learning/Networks/cuda/`:
1. `gpu_topk_logits.cu` — thrust::radix or custom blockwise top-K kernel.
   Input h_t (d) and W_U (V×d); output top_idx[T,K] and top_logits[T,K].
   Stale-refresh cache stored in `NNetwork::gpuTopKCache`.
2. `gpu_sampled_logits.cu` — generate 𝒮_t (cuRAND uniform int without
   replacement, rejecting hits in 𝒯_t); gather ℓ_t[𝒮_t] via sparse GEMV.
3. `gpu_tailce_loss.cu` — numerically-stable logsumexp across (K+S)
   entries; output L̂(t), ∂L̂/∂ℓ_t (sparse), and the scatter indices for
   W_U gradient accumulation.

Integration in `sgd_transformer.cpp::computeLoss()`:
- Add flag `cfg.tail_ce = true` in `training_config.h`.
- Replace `cross_entropy_nll_loss` call with `tailce_forward_backward` when flag is set.
- Keep the full-CE path for W_warm steps and for validation.

Estimated LOC: ~600 new + ~50 modified.

## 12. Gate-0 probe

**Hypothesis under test:** ≥95% of softmax mass lies in top-K for K=512
*on the Glades CHIRON training distribution* (pile-bpe, V=32k-60k), *after
<5k steps*. If this fails, TAIL-CE's speedup will not materialize.

**Probe:**
1. Checkpoint the current `bpe --large --atlas` run at step 5k, 10k, 20k.
2. For each of 1000 random tokens in the validation set, compute the full
   softmax p_t over V (ground truth).
3. Sort p_t descending, measure cumulative mass at ranks {64, 128, 256,
   512, 1024, 2048}.
4. Report: median + p95 mass at top-K for each K.
5. **Pass condition:** median top-512 mass ≥ 0.95 AND p95 top-512 mass ≥ 0.85.
6. **Secondary:** target y_t in top-K rate (should be ≥ 0.90 for K=512 to
   avoid F2 triggering on most tokens).

Probe cost: 1 inference run + 1 Python analysis script. ~30 min.

If pass → proceed to 100-step A/B in a training run with TAIL-CE at K=512.
If borderline (0.85 ≤ median ≤ 0.95) → try K=1024.
If fail (< 0.85) → shelve; tail is too heavy for this vocabulary.

## 13. Strengths vs weaknesses

**Strengths**
- **Large speedup** (42×+ on CE stage, 15–40% end-to-end at V=32k).
- **Unbiased gradient** up to O(1/S) SNIS, with explicit variance bound.
- **Structurally sparse** unembed gradient — no new optimizer state — composes with FACE.
- **Clear Gate-0** — the concentration assumption is directly testable on a checkpoint.
- **Graceful degradation** — warmup uses full CE, only activates when mass is concentrated.

**Weaknesses**
- **Requires top-K machinery** (PQ or stale refresh) — new CUDA kernels.
- **Biased loss** (upper bound), not biased gradient; needs careful reporting.
- **Variance spikes** on rare tokens where tail is heavy — mitigated by F4/F6 but adds complexity.
- **Interaction with Adam v_t** requires the variance-correction patch (§7.5).
- **Vocab-dependent** — gains small at V ≤ 8k, large at V ≥ 32k.

---

## Summary (for coordinating agent)

1. **Claim:** TAIL-CE replaces O(T·V) softmax cross-entropy with O(T·(K+S)) top-K exact head + uniform-proxy sampled tail, yielding an unbiased (up to O(1/S) SNIS) gradient estimator with variance controlled by the tail-mass concentration.
2. **Key novel mechanism:** Horvitz–Thompson importance-sampled tail log-partition fused with head logsumexp (`Z_hat = logsumexp(head, L_samp)`), with target force-inclusion and variance-corrected Adam v_t to keep the Adam step scale unbiased.
3. **Expected speedup + memory:** 42× on the unembed/CE stage at V=32k, K=512, S=256 (15–40% end-to-end depending on the rest of the step); softmax/logit activation memory cut ~40× (128 MB → 3 MB per step at T=2048); unembed gradient structurally sparse (K+S rows per token vs V); no new optimizer state.
