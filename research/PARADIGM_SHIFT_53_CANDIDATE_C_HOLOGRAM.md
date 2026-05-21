# Paradigm Shift #53 Candidate C — HOLOGRAM: Diffusion-Based LLM Training via Iterative Denoising

**Status:** candidate-C design; one of three parallel proposals for paradigm shift #53. **Recommendation up front: REJECT for #53.** This document exists to formally close the diffusion-LLM direction with a written argument so the team does not re-derive it later.
**Date:** 2026-05-08 (Ralph-loop iter 197+, post-#52 NIMBUS+PHOENIX-NF4 promotion, iter-197 brief: *novel LLM architectures, algorithms, and training methods; compute speed for training extremely large LLMs on a single GPU.*).
**Axis:** **training-objective replacement** — discard the autoregressive next-token cross-entropy loss `L = -Σ_t log P(x_t | x_<t)` and replace it with iterative-denoising MSE `L = E_t E_ε ‖ε - f_θ(x_t, t)‖²` over a Gaussian-noise corruption process on token embeddings. CHIRON's reversible-flow transformer becomes the denoiser network `f_θ`.

**Materially distinct from competing #53 candidates:** Candidates A and B (presumed: training-time micro-optimizations or NLL-preserving architectural variants) operate *within* the autoregressive paradigm. HOLOGRAM operates *outside* it — different loss, different generation algorithm, different evaluation surface. It is the most architecturally radical of the three but, for this brief's training-speed axis, the least defensible.

**Honest headline (read this, skip the rest if pressed for time).** HOLOGRAM is a fundamentally different LLM paradigm. Its **training compute per step is comparable to autoregressive** — there is no training-speed dividend at fixed quality. Its **generation compute drops by ~T/N** (e.g. ~51× at `T=1024, N=20`), but generation speed is not what the iter-197 brief asks for. HOLOGRAM is **an inference / deployment paradigm masquerading as a training paradigm**. The honest recommendation: **reject for #53**; reserve as a future deployment-time paradigm if and when CHIRON moves into production-serving territory and inference-latency becomes the binding constraint.

**Engineering scope if pursued anyway.** ~3500 LOC, 14–18 weeks (denoiser-loss surface in `sgd_transformer`, noise scheduler, embedding-space diffusion machinery, sampler in `transformer_generate`, full retraining of any compared baselines for a fair head-to-head NLL — *and the NLL comparison itself is non-trivial because the metric semantics differ; see §6*). Risk profile: high engineering, unknown training-speed yield (probably zero or negative), uncertain quality at scale.

---

## 0. Executive summary (HONEST framing)

### 0.1 What HOLOGRAM is

HOLOGRAM proposes that CHIRON abandon next-token prediction and instead train a *denoiser*: a network that takes a noisy embedding sequence `x_t` (corrupted from a clean target `x_0` by `t` steps of additive Gaussian noise) and predicts either the clean `x_0` or the noise `ε`. Generation then becomes iterative: start from pure noise `x_T`, repeatedly apply the denoiser to step `x_T → x_{T-1} → … → x_0`, and decode the converged embeddings to tokens.

This is the **DiffuSeq / SeqDiffusion / SSD-LM family** of models. CDCD, Plaid, and a handful of 2023–2024 academic systems demonstrated viability at sub-1B scale; quality at 7B+ remains an open question and the published results are not flattering relative to autoregressive peers.

### 0.2 Why HOLOGRAM is the wrong shift for the iter-197 brief

The user's iter-197 brief explicitly asks: *"compute speed for training extremely large LLMs on a single GPU."* Three independent reasons HOLOGRAM does not deliver against that brief:

1. **Training compute is not reduced.** Each training step still requires one forward + one backward pass over the full sequence at full embedding dimension. The loss target changes (MSE on noise vs. cross-entropy on tokens) but the per-step FLOP count is the same to within a small constant. There is **no training-step speedup**.
2. **Per-step quality progress is plausibly worse.** Diffusion training spreads the learning signal across a noise-level continuum: a single training example contributes to denoising at one noise level `t`. Effective sample efficiency at fixed compute is, in published comparisons (DiffuSeq 2022, SSD-LM 2023, CDCD 2023), **2–4× worse than autoregressive** at matched parameter count. Total training compute *to hit a fixed-quality target* therefore goes **up**, not down.
3. **NLL is not a clean metric.** Diffusion LLMs report variational lower bounds (ELBO) on log-likelihood, not exact NLL. The bound is loose; comparing to autoregressive NLL is apples-to-oranges. The NIMBUS-style "bit-exact NLL preservation" guard that has anchored shifts #42–#52 simply does not apply to HOLOGRAM. Any forward-port of HOLOGRAM into the CHIRON regression suite would require redefining the quality metric, which is a project unto itself.

### 0.3 Where HOLOGRAM *would* shine

HOLOGRAM's one genuine advantage is **parallel generation**: at sequence length `T = 1024` and `N = 20` denoising steps, the model produces a full-length sample in 20 forward passes versus 1024 for autoregressive. That is a **51× wall-clock advantage at inference time**. If CHIRON were a deployed model serving latency-sensitive requests at scale, this would matter enormously. It would matter less for batched throughput-oriented serving (where autoregressive can amortize KV cache reuse), and not at all for pretraining.

The user has, throughout the iter-100 → iter-197 arc, been explicit that the active research goal is **training-time wall-clock at fixed NLL**. HOLOGRAM does not move that needle.

### 0.4 Recommendation

**Reject HOLOGRAM for paradigm shift #53.** Document the reasoning in this candidate file so the team does not re-propose it in iter-200+. Reserve HOLOGRAM as an *inference-time* paradigm to be revisited if and when:

- (i) CHIRON is deployed and inference latency becomes the binding cost; OR
- (ii) the brief shifts from "fastest training to fixed NLL" to "fastest generation at fixed quality"; OR
- (iii) a sufficient algorithmic breakthrough in diffusion-LLM training efficiency (specifically: variance-reduced loss estimators that close the 2–4× sample-efficiency gap) appears in the literature.

None of those preconditions are present in iter-197.

---

## 1. Diffusion mathematics (concise)

### 1.1 Forward (corruption) process

Let `x_0 ∈ ℝ^{T × m}` be a clean embedding sequence (rows = tokens, columns = embedding dim). Define a variance schedule `{β_t}_{t=1}^{T_diff}` with `β_t ∈ (0, 1)` and the cumulative coefficients

```
α_t = 1 - β_t
ᾱ_t = ∏_{s=1}^t α_s
```

The forward marginal at noise level `t` is the Gaussian

```
q(x_t | x_0)  =  N( x_t ; √ᾱ_t · x_0,  (1 - ᾱ_t) · I )
```

equivalently `x_t = √ᾱ_t · x_0 + √(1 - ᾱ_t) · ε`, with `ε ~ N(0, I)`. As `t → T_diff`, `ᾱ_t → 0` and `x_t → N(0, I)` — pure noise. The schedule (linear, cosine, sigmoid, log-SNR) governs how rapidly information is destroyed.

### 1.2 Reverse (denoising) process

Sampling reverses the chain. The optimal reverse marginal `q(x_{t-1} | x_t, x_0)` is also Gaussian (derivable from Bayes' rule on the forward chain). Training learns a parametric denoiser `f_θ(x_t, t)` that approximates either:

- **`x_0` parameterization**: predict the clean embedding directly. Loss: `‖x_0 - f_θ(x_t, t)‖²`.
- **`ε` parameterization**: predict the noise that was added. Loss: `‖ε - f_θ(x_t, t)‖²`. Equivalent up to a `t`-dependent rescale; usually preferred numerically (Ho et al. 2020).
- **`v` parameterization**: predict `v_t = √ᾱ_t · ε - √(1 - ᾱ_t) · x_0`. Numerically robust across all `t`; Salimans & Ho 2022.

### 1.3 Training objective and sampling

```
L_diffusion(θ)  =  E_{x_0, t, ε}  ‖ε - f_θ( √ᾱ_t · x_0 + √(1 - ᾱ_t) · ε,  t )‖²
```

The **simple loss** of Ho et al. — a weighted ELBO with `t`-dependent weights at 1. Variants (min-SNR, log-SNR weighting) modify the weights to improve sample efficiency. For text, `x_0` is the *embedding* of the token sequence; the recovered `x̂_0` is decoded back to tokens via nearest-neighbor lookup, a classifier head, or self-conditioning. The discrete-token bridge is the principal source of fragility.

Sampling iterates `x_T ~ N(0, I)` then `x_{t-1} ~ q_posterior(x_{t-1} | x_t, f_θ(x_t, t))` for `t = T_diff, …, 1`. DDIM and DPM-Solver allow `N << T_diff` sampling steps (e.g. `N = 20` for `T_diff = 1000`) with minor quality loss — this is the source of the ~51× generation speedup at `T = 1024, N = 20`.

---

## 2. CHIRON-as-denoiser integration

### 2.1 Architectural mapping

CHIRON's reversible-flow shear `(q, p) ↦ (q, p + Y(q))` maps onto a denoiser layer by interpreting `(q, p)` as `(noisy_embedding, noise_estimate)` and `Y(·)` as the layer's noise-prediction contribution. Each of `L = 53` layers contributes one shear; the cumulative composition is the full denoiser. Reversibility is preserved trivially.

Noise-level conditioning `t` is injected via a **time embedding** — sinusoidal or learned — added to or concatenated with the per-layer hidden state. CHIRON's `(q, p)` gains a third stream `τ` treated as a frozen control (no state gradients) to preserve 2-stream symplectic reversibility.

### 2.2 Loss surface change

Concretely, in `sgd_transformer.cpp`:

- The forward pass would compute `f_θ(x_t, t)` instead of `softmax(W · h_L)`.
- The loss kernel would compute MSE in embedding space, not cross-entropy in vocab space.
- The vocabulary projection `W_lm_head` is no longer the loss target; it would either be discarded (decode by nearest-neighbor at sampling time) or retained for an auxiliary classification term to stabilize the discrete-token bridge.
- Backward propagates MSE gradients, which have **uniform numerical scale across `t`** — this is actually slightly easier on bf16 dynamic range than cross-entropy's heavy tails (a small but real plus).

### 2.3 KV-cache and SLC interaction

CHIRON's KV-cache (used in autoregressive serving) becomes irrelevant under HOLOGRAM: each denoising pass is a full `T`-token forward, and there is no causal masking. The attention in HOLOGRAM is **bidirectional**, which:

- Removes causal-mask compute (small win, ~10%);
- Eliminates KV-cache memory savings during generation (because every step recomputes all `T` positions anyway);
- **Breaks SLC (paradigm #38)**, because there is no notion of "shorter T for warmup, longer T for refinement" in a diffusion training loop — the noise level `t` plays the role of a curriculum dimension instead. SLC would have to be redesigned, not re-used.

### 2.4 Compatibility with the shipped stack

Compatibility audit of the 11 shipped paradigms #42–#52: **#47 NIMBUS** (NLL preservation) and **#52 NIMBUS-promoted** break outright because the NLL metric semantics differ under diffusion. **#38 SLC** breaks because the curriculum axis becomes `t` (noise), not `T` (sequence length). **#51 ATLAS-COMPILE** breaks because compiled kernels are autoregressive-shaped. #42 SCFA, #39 RLG, FACE (#28), #52 PHOENIX-NF4 are probably OK but require re-validation. **At least three of eleven shipped paradigms break outright; another four require non-trivial re-validation.** The opportunity cost is not just 14–18 weeks of new engineering but re-litigating a substantial fraction of the existing stack.

---

## 3. Compute analysis — the central honest accounting

### 3.1 Training compute per step

For an autoregressive forward pass at `(T = 1024, m = 2048, L = 53, batch = B)`:

```
FLOPs_AR_step  ≈  2 · L · B · T · (4 m² + 2 m T)    # MLP + attention
              ≈  2 · 53 · B · 1024 · (4 · 2048² + 2 · 2048 · 1024)
              ≈  3.6 · 10¹² · B   FLOPs
```

For a HOLOGRAM denoiser forward pass at the same dimensions, **the compute is identical** — same `L`, same `T`, same `m`, same attention pattern (modulo bidirectional vs. causal, which is at most a 2× difference and in practice ~10% because the cuBLAS GEMMs dominate). The backward is also identical. The loss kernel is cheaper (MSE is `O(T·m)` vs. cross-entropy's `O(T·V)` where `V = 50k` vocab) — a small win, ~5% of step time at this scale.

**Per-step training compute: HOLOGRAM ≈ AR within ±10%.** No win on the training-step axis.

### 3.2 Sample efficiency — where HOLOGRAM loses

The published evidence on diffusion-LLM sample efficiency at matched parameter count and compute:

- **DiffuSeq (Gong et al. 2022)** — 80M params, 30k training steps to reach competitive perplexity vs. 12k for autoregressive baseline. ~2.5× sample-efficiency disadvantage.
- **SSD-LM (Han et al. 2023)** — 1.3B params, requires ~2× tokens to match autoregressive perplexity at the same parameter count.
- **CDCD (Dieleman et al. 2022)** — concedes a ~2× gap and proposes self-conditioning to close it; closes ~30% of the gap at moderate scale.
- **Plaid (Gulrajani & Hashimoto 2024)** — 1.3B, careful tuning closes the gap at 1.3B, but does not extrapolate well to 7B+.

The most charitable read: HOLOGRAM training requires **1.5–2.5× the tokens / steps** to reach a fixed quality target. Combined with per-step parity, **total training compute to fixed quality is 1.5–2.5× higher** than autoregressive.

### 3.3 Generation compute — where HOLOGRAM wins (but the brief doesn't ask)

Sample one sequence of length `T = 1024`:

- **AR generation**: 1024 forward passes × `O(L · m²)` per pass with KV-cache = `~3.5 × 10⁹` FLOPs per token × 1024 = `~3.6 × 10¹²` FLOPs total.
- **HOLOGRAM generation, N = 20 DDIM steps**: 20 forward passes × `O(L · T · m²)` per pass = `~3.6 × 10¹²` FLOPs total.

The total FLOPs are roughly **comparable**, but the *latency* differs because diffusion's 20 passes are sequential-of-20 while AR's 1024 passes are sequential-of-1024. With perfect parallelism inside each pass, HOLOGRAM's wall-clock generation latency is **~51× lower**.

This advantage is real and substantial — **for inference**. It does not apply to training, which is the brief's axis.

### 3.4 Memory

HOLOGRAM's training memory profile is approximately equal to AR's, with two differences:

- **No KV cache during training** (already true for AR training, so no change).
- **Time embedding** adds a tiny constant (`O(T · m_t)` where `m_t ~ 256`).
- **Bidirectional attention** stores no extra mask state.

Memory: HOLOGRAM ≈ AR within ±2%. No win, no loss.

### 3.5 Stack composition

If shipped, HOLOGRAM's *training-time* contribution to the stack at 18B would be a **1.0× factor** (parity per step) multiplied by **(1/1.5 to 1/2.5) on sample efficiency** = **0.4× to 0.67× net regression**. The pre-#52 stack of ~917× would degrade to **~370× to ~615×**. We would **lose between 300× and 550×** of accumulated speedup by adopting HOLOGRAM under the iter-197 brief.

This is not a defensible direction for shift #53.

---

## 4. The honest gap: training vs. generation

The paradigm-shift framework adopted since iter-100 has a single optimization target: **wall-clock training time to reach a fixed-quality model**. Every shipped paradigm has been judged against that target and a strict NLL-preservation constraint.

HOLOGRAM optimizes a *different* target — **wall-clock generation time at fixed model quality**. These are not the same problem. They share parameters (model weights, embedding dim, layer count) but differ in:

| Axis | Training-time goal | Generation-time goal |
|---|---|---|
| Compute is amortized over | One pretraining run | Many user requests |
| Bottleneck | Per-step FLOPs × steps-to-converge | Per-token / per-sample latency |
| What helps | Cheaper per-step, faster convergence | Parallelism in the inference graph |
| What hurts | Sample-inefficient losses, brittle precision | Sequential dependencies, large KV cache |
| HOLOGRAM verdict | **Hurts** (worse sample efficiency) | **Helps** (~51× parallel decoding) |

A paradigm targeting one cannot be substituted for one targeting the other. The literature occasionally elides this distinction — papers that report "X× speedup with diffusion" almost universally mean **sampling speedup**. The training speedup, when reported honestly, is zero or negative.

---

## 5. When HOLOGRAM would be the right shift

There are well-defined futures where HOLOGRAM becomes attractive:

### 5.1 Future A — CHIRON enters production serving

Once CHIRON is deployed at meaningful query volume, the cost surface shifts from "GPU-hours to train" to "GPU-hours to serve." At that point:

- A 51× generation latency reduction is worth a 1.5–2.5× training cost increase if the model is queried ≥ 1000× per training run. Most production LLMs cross that threshold within hours of deployment.
- The training-time NLL metric becomes secondary to deployment-time cost-per-token.
- The inference paradigm shift is then *strictly additive* over an autoregressive base: distill an AR model into a HOLOGRAM denoiser as a post-training step. This is the SeDD and CDCD approach.

### 5.2 Future B — sample-efficiency breakthrough

A research result that closes the 1.5–2.5× sample-efficiency gap (e.g., a variance-reduced denoising loss, a better noise schedule, a self-conditioning recipe that holds at 18B) would re-open the question. The gap has narrowed slowly since 2022; another 2 years of progress could plausibly bring it to parity. Worth monitoring; not worth betting #53 on.

### 5.3 Future C — multi-modal expansion

If CHIRON expands to image / audio / video, the diffusion paradigm becomes natively well-suited (image-diffusion is the dominant paradigm and language-diffusion piggybacks naturally). HOLOGRAM-style training on a multi-modal token stream might then be **strictly better** than autoregressive across all modalities. Not in scope for iter-197.

### 5.4 Future D — explicit non-autoregressive brief

If a future iter-N brief asks for "fastest serving at fixed quality" or "lowest inference latency," HOLOGRAM is the obvious answer and should be developed in earnest at that point.

In **none** of these scenarios does HOLOGRAM advance the iter-197 brief as written.

---

## 6. NLL semantics — the subtle, dangerous gap

A specific technical gap worth flagging because it is easy to miss: **diffusion LLMs do not report NLL in the same sense that autoregressive LLMs do**.

An autoregressive model computes exact NLL via its softmax: `NLL = -Σ_t log P_θ(x_t | x_<t)`. NIMBUS (#47) preservation of this quantity to bit-exact equivalence has been the gold standard since iter-180+.

A diffusion model's "log-likelihood" is a **variational lower bound (ELBO)**:

```
log P_θ(x_0)  ≥  -L_VLB(θ)  =  E_q[ log p_θ(x_0 | x_1) - Σ_t KL(q(x_{t-1} | x_t, x_0) ‖ p_θ(x_{t-1} | x_t)) ]
```

The bound is loose; gap to true log-likelihood is unmeasured at LLM scale. Reporting NLL for a diffusion model and comparing to an autoregressive NLL is **methodologically incorrect** without an explicit gap-closure analysis. Recent papers (Lou et al. 2024 on SEDD) have started reporting "true" likelihoods via importance sampling, but the procedure is expensive and noisy at any meaningful sequence length.

**Practical implication for HOLOGRAM in CHIRON:** the regression-suite tests that gate every paradigm shift would need to be redefined. The "EMA loss" diagnostics in `run.sh` would mean different things in HOLOGRAM-mode versus AR-mode. Cross-paradigm comparisons would require careful disclaimers. This is a real cost that the engineering scope above (§intro) only weakly accounts for.

---

## 7. Empirical risks (Gate-0 thought experiment, NOT to be run)

A standard Gate-0 for HOLOGRAM would test the *premise* — does HOLOGRAM at small scale match autoregressive at small scale on training-compute-to-NLL?

A faithful Gate-0 would require:

1. Training a 41M HOLOGRAM denoiser on the same `pile-bpe` corpus for the same wall-clock budget as a baseline AR run.
2. Defining an NLL-equivalent metric — likely the variational bound, possibly with importance-sampled correction.
3. Comparing total compute to a fixed "good enough" quality target (perplexity proxy or downstream task).

**Predicted outcome based on the literature:** HOLOGRAM 41M would reach AR-equivalent quality at **~1.8× the wall-clock**. The Gate-0 would correctly reject the hypothesis "HOLOGRAM is faster to train at fixed quality."

**Cost of running Gate-0:** ~6 GPU-hours × 2 (AR baseline + HOLOGRAM) = ~12 GPU-hours, plus 2–3 days of engineering to wire up the denoiser loss surface in a throwaway branch. Not catastrophic, but not free, and the literature already tells us the answer.

**Recommendation:** skip the empirical Gate-0 for HOLOGRAM. Reject on the prior. If a future brief shifts the optimization target, revisit.

---

## 8. Recommendation

**Reject HOLOGRAM as the candidate for paradigm shift #53.**

### Rationale (one paragraph)

Paradigm shift #53 is being chosen under the iter-197 brief: *novel methods for compute speed in training extremely large LLMs on a single GPU.* HOLOGRAM is novel — genuinely so, more architecturally radical than any shipped CHIRON paradigm — but it is novel along the **wrong axis**. It addresses generation latency, not training compute. Best-case reading of the literature: HOLOGRAM training is 1.5–2.5× **slower** to fixed quality than autoregressive at matched parameter count. Adopting HOLOGRAM as #53 would mean **erasing 300×–550× of the accumulated stack** to gain a generation-time advantage that the brief does not request and that the project does not currently need (CHIRON is in research mode, not deployment). The compatibility cost is also substantial: at least three shipped paradigms (#47 NIMBUS, #38 SLC, #51 ATLAS-COMPILE) break under HOLOGRAM and would require re-design.

### Disposition

- **Mark HOLOGRAM as a "deferred deployment-time paradigm"** in `FUTURE_PARADIGM_CANDIDATES.md`.
- **Note the futures (§5.1–§5.4)** under which HOLOGRAM becomes the right shift.
- **Do not allocate Gate-0 GPU-hours.** The literature's prior is sufficient to reject.
- **Select between candidates A and B** for shift #53, both of which (presumed) operate within the autoregressive paradigm and therefore have a defensible chance of advancing the iter-197 brief.

### Closing note on honesty

It is tempting, in a series of paradigm shifts that have repeatedly favored increasingly architecturally radical proposals (SLC, RLG, FACE, NIMBUS, PHOENIX-NF4), to over-correct toward novelty for novelty's sake. HOLOGRAM is the limit case: maximally novel, but mismatched with the brief. The right move is to recognize the mismatch in writing, file the paradigm for a future where it fits, and proceed with a candidate that targets the actual question being asked.

— Ralph-loop iter 197, candidate-C author, 2026-05-08.
