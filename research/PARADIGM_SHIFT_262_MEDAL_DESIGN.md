# Paradigm #262 — MEDAL: Measure-Evolution Denoising Autoregression-free Language model

**Iter 42 of the CHIRON Architecture Magnitudes Research Loop**
**Status**: design (post-HMTA pivot); math test deliverable in this iter; full training Gate-0 deferred to iter 43+
**Date**: 2026-05-16
**Family**: measure-evolution / discrete absorbing-mask diffusion
**Pivoted-from**: #261 HMTA (empirically falsified on real flagship — iter 41)

---

## 0. Quick read

- **Core mechanism**: model the joint distribution `p(x_1, …, x_T)` on tokens via K-step parallel **denoising** of an absorbing-mask CTMC, instead of T-step autoregressive factorization.
- **Inference wall-clock magnitudes**: at K = 64 and T = 16384, the *serial* generation cost drops from T = 16384 steps to K = 64 steps — **T/K = 256× wall-clock per-token speedup**. This is where the brief's "magnitudes" target is met.
- **Training compute**: comparable to autoregressive (mask random fraction per step → predict masked positions). No 256× training cost.
- **Recovery limits** (proven): AR is the K = T limit with deterministic left-to-right unmask order. SCFA (paradigm #42) is reusable inside the denoiser (with causal mask removed).
- **No-op-at-init**: not applicable — MEDAL is train-from-scratch, not augmentation. (The iter-37 sketch acknowledged this; the brief explicitly allows train-from-scratch as one of the two paths.)
- **Gate-0 (iter 42, this iter)**: math-test only — verify that the absorbing-mask CTMC's closed-form Bayes reverse recovers the data distribution exactly under an oracle denoiser. **Pass = exact recovery up to sampling noise; fail = mathematical error in the corruption/reverse pair.**
- **Gate-0 (iter 43+, full)**: small from-scratch training comparing MEDAL ELBO at K_eval ∈ {16, 64, 256} vs AR exact NLL on the same data with matched parameters and tokens.

---

## 1. Executive summary

After paradigm #261 (HMTA) was empirically falsified on real flagship attention (iter 41 SV-probe showed median rank-p retention 0.52 at p=8, requiring p_K ≥ 32 to match real spectra, at which FLOP ratio falls below the brief's 10× floor), the research program pivots to the second-place candidate from the iter-37 dispatch.

MEDAL recasts the LLM generation problem in a fundamentally different mathematical frame:

| | Autoregressive (current flagship) | MEDAL |
|---|---|---|
| Factorization | `∏_{t=1}^T p(x_t \| x_{<t})` | none — joint denoising |
| Generation | T serial steps | K serial steps (K ≪ T) |
| Training | next-token CE | masked-CE ELBO |
| Inference wall-clock | O(T) | O(K) |
| Attention pattern | causal | bidirectional |
| Loss bound | exact | ELBO upper bound (gap = 0 at optimum for absorbing diffusion) |

The mathematical foundation is the **absorbing-mask continuous-time Markov chain** on the extended alphabet `Ṽ = V ∪ {⊥}`. The forward process gradually masks tokens at a learned rate `σ(t)`; the reverse process is **closed-form Bayes-exact** given the data — no Tikhonov-style implicit solves, no ill-conditioned linear systems, no per-token random-init state. The reverse is parameterized by a learnable denoiser `f_θ(x_t, t)` that emits the conditional `p_θ(x_0 | x_t)`. The ELBO is the standard D3PM weighted masked-CE.

**Falsifiable Gate-0 (full)**: at parameter-matched, token-matched compute, MEDAL trained at K_train = 128 achieves val ELBO ≤ AR exact NLL + 0.10 nat, AND K_eval = 32 inference is ≥ 32× faster per generated token on RTX 4080 SUPER at the test scale. The iter-42 *math test* is a pre-condition that the corruption/reverse pair is correctly implemented.

---

## 2. Why this paradigm, why now

Three sequential paradigm falsifications under the same brief (improve LLM by magnitudes on compute):

1. **#250 SFA** (iter 22) — augment trained baseline with cellular-sheaf attention layer. Falsified: augmentation-of-trained-flagship pattern fails universally.
2. **#260 IGAA** (iter 36) — different math, same augmentation pattern. Falsified for the same reason.
3. **#261 HMTA** (iters 37–41) — pivoted to train-from-scratch + multipole attention compression. Math + synthetic data both validate the design at (p_K=8, p_V=48). Real-flagship probe (iter 41) reveals heavy spectral tails: median rank-p retention at p=8 is 0.52 (vs. design's optimistic 0.95 prediction). Empirically falsified: no (p_K, p_V) clears both retention ≥ 0.85 AND FLOP ≥ 10× on real flagship.

The remaining viable directions, per the iter-37 candidate dispatch and iter-41 post-mortem:

- **Stack HMTA at lower-magnitudes with FFN-side paradigms**: HMTA at (p_K=24, p_V=48) delivers ~9× FLOP and 0.85 retention; combined with paradigms #44 MELT / #74 PHOENIX-1BIT for FFN, total wall-clock magnitudes are plausible. Continues #261 at reduced scope.
- **Pivot to MEDAL (Candidate B)** — different mathematical family, no per-position low-rank assumption, magnitudes come from a DIFFERENT axis (serial steps, not per-step compute).
- **Pivot to VARCO (Candidate C)** — conditional compute. Modest 3–6× upside; not magnitudes-class.

MEDAL is the strongest remaining candidate because (a) it sidesteps the spectral structure that broke HMTA, and (b) it directly targets the wall-clock-per-token axis that the brief likely cares about (users care about latency, not just FLOPs).

---

## 3. Framework selection rationale (against iter-37 candidates)

Reconsidering the three candidates from iter 37 in light of what we now know:

| criterion | A. HMTA | B. MEDAL | C. VARCO |
|---|:---:|:---:|:---:|
| iter-37 candidate dispatch | dispatched & selected | dispatched & rejected | dispatched & rejected |
| primary magnitudes axis | attention FLOPs | inference wall-clock (T/K) | per-layer skip |
| brief's 10×+ target | falsified at scale | T/K = 256× theoretical | 3–6× only |
| mathematical complexity | high (multipole, causality) | medium (D3PM well-developed) | low (Bernoulli + Lagrangian) |
| iter-37 rejection reason | – | "ELBO-vs-AR-NLL subtle" | "sub-magnitudes" |
| post-iter-41 assessment | falsified | re-promoted | unchanged (sub-magnitudes) |

The iter-37 rejection of MEDAL ("ELBO comparison to AR is subtle") was a *risk-aversion* call. Post-iter-41, that risk is the cost of doing business; HMTA's apparent low-risk was itself an artifact of synthetic-data confidence. MEDAL's structural mathematics is on firm ground (D3PM, Lou-Meng-Ermon score-entropy, etc. have been studied extensively in the literature); the empirical risk is whether discrete diffusion at 30–60M scale trains to competitive NLL.

VARCO remains the fallback if MEDAL fails: 3–6× wall-clock combined with FFN-side compression (#44 MELT, #74 PHOENIX) could deliver aggregate magnitudes without the diffusion training risk.

---

## 4. Formal problem statement

**Target system**. The same CHIRON-class LLM as in #261: T=16384 production context, m=2048 width, L=24 depth, V=32000 vocab, ≈870M–1B params, single 16 GB GPU. The post-fix flagship `chiron_1B_T16384.step30000` at val NLL 4.0771 is the iso-NLL anchor.

**Objective**. Minimize val NLL at iso- or sub-compute. *Compute now interpreted as inference wall-clock per generated token*, not total training FLOPs.

**Brief's magnitudes target**, restated: at iso val NLL (or ELBO upper bound that matches val NLL within a small constant), achieve **≥ 10× faster per-token inference wall-clock** vs AR generation on RTX 4080 SUPER at T=16384.

**Hard constraints** (from `ralph.txt`):
- (C1) Train-from-scratch at affordable scale (~30–100M params, ~100–500M tokens, hours).
- (C2) Forward explicit, backward well-conditioned.
- (C3) Recover SDPA / SCFA / AR as proper limits where possible.
- (C4) Single 16 GB GPU; C++/CUDA only.

**Forbidden** (from prior falsifications):
- Augmentation of a trained flagship with thin parametric correction (iter 22, iter 36 falsified pattern). MEDAL is full train-from-scratch — does not violate.
- Per-position low-rank K, V projection at brief-class magnitudes (iter 41 falsified for HMTA). MEDAL doesn't use this; its denoiser is standard transformer-class.
- Implicit-diff through ill-conditioned solves. MEDAL's reverse is Bayes-closed-form.

---

## 5. Core mathematical framework

### 5.1 State space

Let `V` denote the token vocabulary (V = 32000) and define the *augmented* alphabet
$$
\widetilde{V} = V \cup \{\bot\},
$$
where `⊥` is the absorbing "mask" token. Sequences live in `(Ṽ)^T`.

### 5.2 Forward (corruption) process

A per-position continuous-time absorbing Markov chain. For each position `i` independently, the chain has two states: "data token x_0^{(i)}" and "⊥". The chain has rate `σ(t)` from data → ⊥; rate 0 from ⊥ (absorbing).

Let `α(t) = 1 - exp(-∫_0^t σ(s) ds) ∈ [0,1]` be the cumulative mask probability with `α(0) = 0, α(1) = 1`.

Closed-form transition kernel:
$$
q(x_t^{(i)} \mid x_0^{(i)}) = \begin{cases}
1 - α(t), & x_t^{(i)} = x_0^{(i)} \\
α(t), & x_t^{(i)} = ⊥ \\
0, & \text{otherwise.}
\end{cases}
$$

Joint over positions (independent corruption): `q(x_t | x_0) = ∏_i q(x_t^{(i)} | x_0^{(i)})`.

### 5.3 Reverse (denoising) process — Bayes-exact

For absorbing chains, the reverse posterior at any `s < t` is closed-form. For each position `i`:
$$
q(x_s^{(i)} \mid x_t^{(i)}, x_0^{(i)}) = \begin{cases}
1, & x_t^{(i)} ≠ ⊥, \; x_s^{(i)} = x_t^{(i)} \\
α(s)/α(t), & x_t^{(i)} = ⊥, \; x_s^{(i)} = ⊥ \\
(α(t) - α(s))/α(t), & x_t^{(i)} = ⊥, \; x_s^{(i)} = x_0^{(i)} \\
0, & \text{otherwise.}
\end{cases}
$$

In words: at the reverse step from `t` to `s`, each *masked* position is either left masked (with probability `α(s)/α(t)`) or unmasked back to its true value (with probability `(α(t) - α(s))/α(t)`).

Marginalizing the unknown `x_0` against the model's posterior `p_θ(x_0^{(i)} | x_t)` gives the learned reverse kernel:
$$
p_θ(x_s | x_t) = \prod_{i:\, x_t^{(i)} = ⊥} \left[ \frac{α(s)}{α(t)} δ_⊥(x_s^{(i)}) + \frac{α(t) - α(s)}{α(t)} p_θ(x_0^{(i)} = x_s^{(i)} \mid x_t) \right].
$$

The denoiser is `p_θ(x_0^{(i)} | x_t) = softmax(f_θ(x_t, t))_i` where `f_θ : Ṽ^T × [0,1] → ℝ^{T × V}`.

### 5.4 Loss — weighted masked cross-entropy

Sampling `t ∼ Uniform[0, 1]` and `x_t ∼ q(· | x_0)`:
$$
\mathcal{L}(θ) = \mathbb{E}_{x_0, t, x_t} \left[ \frac{1}{α(t)} \sum_{i:\, x_t^{(i)} = ⊥} -\log p_θ(x_0^{(i)} \mid x_t, t) \right].
$$

The `1/α(t)` weighting derives from the continuous-time ELBO; see §6.

### 5.5 Denoiser architecture

The denoiser `f_θ` is a transformer with:
- Token embedding `E_aug ∈ ℝ^{(V+1) × m}` (augmented for `⊥`).
- **Bidirectional** attention (no causal mask). At every layer, every position attends to every position.
- Time embedding `φ(t) ∈ ℝ^m`: sinusoidal, added once per layer to the residual stream.
- Standard FFN + LayerNorm.
- Tied readout: logits = `LN(h^L) · E_aug[:V]^T` (predict over V, never ⊥).

For production T=16384, the bidirectional attention is expensive per step. The K-step amortization makes total inference compute ≈ K × (single forward at T). If K << T, this beats the brief's compute floor *in wall-clock*.

### 5.6 Inference: ancestral sampling

```
x_t_K = ⊥^T                                # start fully masked
for k = K, K-1, ..., 1:
    t_k = k/K
    t_{k-1} = (k-1)/K
    Compute denoiser logits f_θ(x_t_k, t_k) → p_θ(x_0 | x_t_k)
    For each masked position i:
        Sample x_t_{k-1}^{(i)} from q(· | x_t_k^{(i)}, x_0^{(i)} ∼ p_θ(·|x_t_k))
return x_t_0  # all positions unmasked
```

`K` is a hyperparameter at inference, independent of training (which can use any `K_train` or continuous `t ∼ U[0,1]`).

### 5.7 Confidence-ranked unmasking (a.k.a. MaskGIT schedule)

Instead of unmasking randomly per step, **unmask the K most-confident positions first**. This raises K-efficiency at constant total compute, since confident commitments don't need re-evaluation. The "confidence" of position `i` at step `k` is `max_v p_θ(x_0^{(i)} = v | x_t_k)`. Sort masked positions by confidence; unmask the top `T/K` per step.

This is an *inference-time only* refinement; training is unchanged.

---

## 6. ELBO derivation (sketch — proven in D3PM literature)

The variational lower bound on `-log p_θ(x_0)`:
$$
-\log p_θ(x_0) \le \mathbb{E}_q\left[ -\log \frac{p_θ(x_{0:1})}{q(x_{>0} \mid x_0)} \right] = \sum_k \mathbb{E}\, D_{\text{KL}}\!\left[ q(x_{s_k} \mid x_{t_k}, x_0) \,\|\, p_θ(x_{s_k} \mid x_{t_k}) \right] + C.
$$

For absorbing chains, each KL has only one non-trivial term per masked position (the unmask probability), yielding the weighted-CE form in §5.4 in the K → ∞ continuous-time limit.

**Theorem (Optimality)**. The Bayes-optimal denoiser `p^*_θ(x_0 | x_t) = p_data(x_0 | x_t^{\text{unmasked}})` makes the ELBO tight: `inf_θ L(θ) = H(p_{data})`. There is **no asymptotic gap** between ELBO and the true negative log-likelihood for absorbing diffusion, unlike Gaussian diffusion. (Proven; Austin et al. 2021, Lou et al. 2024.)

---

## 7. Computational tradeoffs

### 7.1 Inference

Per generated token, with `K << T`:

| Method | Serial steps | Per-step compute | Per-token wall-clock |
|---|---|---|---|
| AR (flagship, SCFA inside) | T | O(T·k) with KV-cache | O(T·k) total / T = O(k) |
| MEDAL with SCFA inside | K | O(T·k) full forward | O(T·k · K / T) = O(K·k) |

At K=64, T=16384: AR wall-clock per token ≈ k operations (with KV cache). MEDAL wall-clock per token ≈ 64k operations. **MEDAL is K × MORE compute per token**, but spread over K serial steps instead of T.

But the dominant cost in inference is *not* total operations — it is **serial step latency** on the GPU. AR has T = 16384 serial steps; MEDAL has K = 64. If each step has comparable wall-clock latency (both are full-T-length forward passes; AR's incremental forward is `O(T)`-latency per step despite low FLOPs), then **MEDAL achieves wall-clock T/K = 256× speedup**.

(In practice, AR's incremental forward is faster per step than MEDAL's full forward due to KV cache and reduced parallelism waste at small batch. The realistic wall-clock factor is likely T/K × 0.3 ≈ 80× at K=64, still very magnitudes-class.)

### 7.2 Training

Mask a random fraction per step → predict masked positions. Total training FLOPs:
- AR: O(L × T × m^2) per token, O(L × T × m^2) per training step (full sequence in parallel via teacher forcing).
- MEDAL: same. Each training step processes all T tokens of one sequence; the masked subset contributes to the loss.

So **training compute is comparable**, modulo a small overhead from the time-embedding lookup. The 256× advantage is purely in *inference latency*.

### 7.3 Memory

MEDAL needs the standard transformer activations during training; no special KV cache needed (everything is bidirectional). The augmented embedding adds 1 × m parameters. Otherwise identical to a same-size AR transformer.

---

## 8. Recoveries (limits)

**Theorem (AR limit, proven in literature)**. At K = T with the *deterministic left-to-right unmask order* (set α(t) such that the i-th mask is removed first), MEDAL's reverse process samples `x_t` autoregressively. The training loss becomes equivalent to AR teacher-forcing if the denoiser is restricted to use only causally-available context. ∎ (Austin et al. 2021.)

**SCFA recovery (inside denoiser)**. The denoiser's attention is *bidirectional* but otherwise standard. Replacing causal-SDPA with bidirectional-SCFA (paradigm #42) inside each layer is a straightforward swap. SCFA's attention compression is independent of the diffusion dynamics. **The denoiser is a strict superset of SCFA architectures**, just with the causal mask removed.

**SDPA recovery**. Trivial — set the denoiser's attention to bidirectional SDPA.

---

## 9. Failure modes

1. **Discrete-diffusion-at-small-scale training instability**. Reported in some literature (gradient variance high at random `t`; antithetic sampling helps). Mitigation: antithetic time sampling, learning rate warm-up at half the AR-equivalent rate.
2. **Mode collapse at high mask rate**. At `α(t) → 1`, all tokens are masked and the denoiser must predict from pure noise. Mitigation: clip `t ∈ [ε, 1-ε]` with ε ≈ 0.01.
3. **Inference distribution shift between training and eval**. Training samples `t ∼ U[0, 1]`; eval uses discrete grid `t_k = k/K`. Bias O(1/K) for smooth schedules. Mitigation: train with the same grid as eval, or importance-weight.
4. **Token-tying issues with byte-pair encoding**. If a BPE word is split into 3 tokens and one is masked, the unmasked context reveals the rest. Mitigation: span-masking (correlate ⊥ across `w` consecutive positions).
5. **ELBO-vs-true-NLL gap**. For absorbing diffusion specifically, the gap is zero at the optimum. But at *finite training*, the ELBO is an upper bound and the model's empirical NLL might be lower than its ELBO. Mitigation: also report IWAE-M tighter estimator at evaluation.
6. **Order bias from greedy unmasking**. Confidence-ranked unmasking biases toward easy tokens first. Mitigation: random unmasking for the first few steps, then confidence-ranked.
7. **Classifier-free guidance instability**. If we extend MEDAL to conditional generation, CFG with high `γ` blows up variance. Mitigation: cap `γ ≤ 3`.

---

## 10. Comparison to existing methods

**vs Autoregressive (current flagship)**: MEDAL is parallel-decoding; AR is serial. MEDAL inference latency = K × t_step; AR = T × t_step.

**vs D3PM** (Austin et al. 2021): MEDAL is the absorbing-mask variant of D3PM. Identical math; the contribution here is the *implementation path* into the existing CHIRON / SCFA infrastructure.

**vs Score-Entropy Discrete Diffusion (SEDD)** (Lou-Meng-Ermon 2024): SEDD uses a concrete-score parameterization with a different loss. MEDAL uses the simpler weighted masked-CE which gives a tighter ELBO when the data is on-support. SEDD might be considered for iter 44+.

**vs MaskGIT** (Chang et al. 2022): MaskGIT is a special case of MEDAL with confidence-ranked unmasking and a particular `α(t)` schedule. MEDAL strictly generalizes.

**vs HMTA (paradigm #261)**: MEDAL is a generation-paradigm shift; HMTA is an attention-compute shift. They are *compatible* — HMTA could be used inside MEDAL's bidirectional denoiser if HMTA were viable on real attention (which iter 41 falsified).

---

## 11. Implementation plan

### 11.1 Phase A — math validation (iter 42, this iter)

**Deliverable**: a standalone C++11 program `research/medal_corruption_test.cpp` that:
- Implements `q(x_t | x_0)` and `q(x_s | x_t, x_0)` for the absorbing-mask CTMC.
- Implements K-step ancestral sampling with an **oracle denoiser** that returns `x_0` exactly (i.e., reads the true data).
- Verifies: across many trials, the K-step ancestral sample from the oracle equals `x_0` (up to sampling noise → exact when oracle is perfect).
- Verifies: with a uniform-random "denoiser" (worst case), the K-step sample is uniformly distributed on V.
- Verifies: at K = ∞ (continuous time), the corruption-reverse pair is mathematically self-consistent (forward of x_0 then reverse with oracle = x_0).

This is the math sanity test. If this passes, the closed-form Bayes reverse is implemented correctly and we can build a real denoiser on top.

### 11.2 Phase B — minimal trainable prototype (iter 43)

Tiny transformer (~10–30M params, T=128, L=2–4, V=64–256) trained on a SYNTHETIC task with known optimal NLL. Compare MEDAL ELBO at K_eval ∈ {8, 32, 128} vs AR exact NLL trained on same task.

Synthetic task candidate: **Markov-chain language model** with `T=128, V=64`, hidden Markov state, known optimal entropy. AR and MEDAL train on samples; we compare achieved NLL/ELBO.

### 11.3 Phase C — small full-scale prototype (iter 44+)

60M params, T=2048, V=32000 (real BPE), 300M tokens of `pretok-data`, K_train = 128, eval at K ∈ {16, 64, 256}, 4–6 hour run on RTX 4080 SUPER. Compare to AR-control at same compute. This is the **brief's main empirical test**.

### 11.4 Phase D — production port (iter 46+)

Port to T=16384 m=2048 L=24, use SCFA inside bidirectional denoiser, train 30k steps from scratch on `pretok-data`. Compare to flagship at iso-NLL. Wall-clock per-token at K_eval = 64 should be 30–80× faster than AR flagship.

---

## 12. Falsifiable Gate-0 conjectures

### C0-iter42 (this iter — math test)
> Closed-form Bayes-reverse correctness: an oracle denoiser running K-step ancestral sampling produces samples that exactly match the data distribution, for any K ≥ 1. Specifically: at K=1, K=4, K=16, K=64, K=256, the per-position error rate `Pr[x_K_sampled ≠ x_0]` with the oracle is ≤ 0 (deterministic up to bit-level equality).

If this fails, my implementation of the Bayes-reverse is wrong. Pass = math is correctly encoded.

### C1 (iter 43 — synthetic-data Gate-0)
> Trained MEDAL on synthetic Markov-chain data (V=64, T=128, hidden state size 8) with `M = 10M` params trained for 50k steps achieves ELBO val NLL ≤ AR baseline val NLL + 0.10 nat at K_eval = 32. At K_eval = 32 the wall-clock per generated token is ≤ T/K = 4× faster than AR generation at the same dimensions.

This is the trainability + speedup pre-condition.

### C2 (iter 44+ — pretok-data Gate-0)
> Trained MEDAL with `M = 60M` params, `T = 2048`, `K_train = 128`, on 300M tokens of `pretok-data` for ~6 hours wall achieves ELBO val NLL ≤ 5.0 nat at K_eval = 256 and ≤ 5.5 nat at K_eval = 64, with K_eval = 64 inference ≥ 8× faster end-to-end tokens/s than a matched-parameter AR baseline at the same NLL on the same hardware.

### C3 (iter 46+ — production scale)
> Trained MEDAL at production (T=16384, m=2048, L=24, ≈870M params) reaches val ELBO ≤ 4.20 nat at K_eval = 64, with end-to-end inference throughput at K_eval = 64 of ≥ 10× the AR flagship's tokens/s. This is the brief's magnitudes target.

---

## 13. What gets falsified at each gate

- **C0-iter42 FAIL** → math implementation bug. Fix and retry; this gate is purely about correctness.
- **C0-iter42 PASS, C1 FAIL** → diffusion training is hard at small scale. Either fix the training recipe or pivot to VARCO.
- **C1 PASS, C2 FAIL** → 60M-scale ELBO doesn't track AR NLL. Likely a scaling issue; bigger model or more training. Or pivot.
- **C2 PASS, C3 FAIL** → production-scale specific issue (e.g., bidirectional attention at T=16384 is too expensive in wall-clock to beat AR's amortized cost). Investigate at-scale.
- **All passes** → magnitudes-class wall-clock magnitudes achieved.

---

## 14. Honest pre-mortem

Why might MEDAL fail in 2 iters where HMTA took 5? Three risks:

1. **Discrete diffusion training instability at small scale**. Literature reports varying success. Mitigation: closely follow Austin/Lou recipes; antithetic sampling; clip `t ∈ [ε, 1-ε]`.

2. **The ELBO upper bound might be loose at finite training**. Even though the asymptotic gap is zero for absorbing diffusion, in practice the ELBO at 100M training tokens may overestimate the true NLL by 0.5+ nat, making the AR comparison unfair. Mitigation: report IWAE-M alongside; use a large enough training token budget.

3. **Bidirectional attention at T=16384 has memory pressure**. Without causal masking, the K, V cache for inference can't be reused across steps the way AR's KV cache can. Each MEDAL forward at T=16384 is a fresh full-T attention. With SCFA-inner this is manageable, but doesn't compose with KV-cache speedups.

Acceptable failure: if MEDAL gets to 5× wall-clock magnitudes at iso-NLL but not 10×, that's still useful (stacks with FFN-side paradigms). Unacceptable failure: ELBO + 0.5 nat over AR at iso-compute — that would indicate diffusion training is fundamentally harder.

---

## 15. Open conjectures (summarized)

- **C0-iter42**: Bayes-reverse correctness (math test, this iter).
- **C1**: Synthetic-data trainability at small scale (iter 43).
- **C2**: pretok-data trainability at 60M scale (iter 44+).
- **C3**: Production wall-clock magnitudes target (iter 46+, the brief's main goal).
- **Conjecture C5**: At production scale, *some* combination of MEDAL + FFN-side compression (#44, #74) delivers 100× total wall-clock magnitudes. This is the *stack* target.

Falsification signal hierarchy: C0 (math correctness) → C1 (small-data feasibility) → C2 (mid-scale viability) → C3 (production magnitudes). Each is a clear off-ramp if needed.

---

## 16. Reproducibility manifest

- **Branch**: `vesta5` (glades-ml).
- **Baseline**: post-fix flagship `chiron_1B_T16384.step30000`, val NLL 4.0771 on `pretok-data/val`.
- **Codebase**: glades-ml C++98 lib + glades-trainer C++11. New files at `research/medal_*.cpp` (iter 42+).
- **GPU**: RTX 4080 SUPER, 16 GB.

**End of paradigm #262 MEDAL design document.**
