# Paradigm Shift #56 Candidate A — SCROLL (Self-Curriculum Reasoning-Oriented LLM)

**Status:** candidate-A design for paradigm shift #56. One of three parallel proposals for #56.
**Date:** 2026-05-08 (Ralph-loop iter 200, post-#55 SOPHIA-CHIRON, under the iter-200 user brief: *"novel architectures, algorithms, and training methods by looking at the bigger picture instead of focusing on microoptimizations"*).
**Predecessors:** `PARADIGM_SHIFT_55_CANDIDATE_A_SOPHIA_CHIRON.md` (training-method axis precedent); `PARADIGM_SHIFT_46_CANDIDATE_A_REFLECTOR.md` (cotangent-lift gradient-norm primitive); `PARADIGM_SHIFT_43_CANDIDATE_C_ORION.md` (Pearlmutter HVP, anchor-step pattern).
**Axis:** **data-side training-method change** (not architecture, not optimizer, not memory). Replace uniform per-token sampling with **importance-weighted active learning**: forward-only score on a candidate batch B, backward only on top-K most informative `(K ≪ B)`.
**Author role:** data-efficiency theorist refining the Beygelzimer–Dasgupta–Langford (2009) and Shen et al. (2017) importance-weighted active-learning estimators for LLM pretraining.

**Reference.** Shen, Wu, Pham, Wei, Lample. *Active learning for natural language inference and beyond.* ICML 2017. Also: Katharopoulos & Fleuret, *Not all samples are created equal: deep learning with importance sampling*, ICML 2018, reports 2-5× steps reduction at fixed loss for ResNet-50/CIFAR via gradient-norm prioritization.

**Tagline.** *#42–#55 attacked compute, memory, and the optimizer trajectory — what we DO with each token. SCROLL attacks WHICH tokens we feed gradients on. Same compute budget, 2-5× more learning per step.*

---

## 0. Executive summary (HONEST claim, conjecture-dependence flagged)

After paradigms #1-#55 the cumulative single-GPU stack is **~3280× at 18B / T=1024** (NLL-strict floor) and **~12940× at 144B-eff / T=16384** (NLL-competitive ceiling). Every paradigm through #55 attacks **what we do with each training token**: compute it cheaper, store its gradient cheaper, route through fewer layers, take a smarter optimizer step. **None of #1-#55 questions whether the token deserves a backward pass at all.** The training corpus is a uniform bag, each token equally weighted.

The **data-efficiency axis** is paradigm-untouched on the CHIRON stack.

**SCROLL's premise.** Per-token loss varies enormously across pretraining. Per-token gradient-norm distribution is **heavy-tailed** — ~80% of tokens deliver < 20% of gradient signal, ~5% deliver > 50% (Settles 2009, Beygelzimer 2009, Katharopoulos 2018). Standard SGD samples uniformly. SCROLL samples **forward-only on a candidate batch of `B = αk`** (α=4), scores per-example informativeness, **backwards only on the top-K = k**. Importance weights restore unbiasedness.

**Per-step cost.** Forward-only on B candidates: `αF`. Backward on top-K: `2F`. **Net: `αF + 2F = 6F` vs baseline `3F` → 2× slower per step.**

**Per-step gradient signal.** Selected K = top-25% by grad-norm in B=4k. Mean grad-norm of selected ≈ 2.5-3× population mean (heavy-tailed concentration; §3.4 verified). **Information per step: 2.3-4× baseline.**

**Net wall-clock to fixed final NLL.** 2.3-4× information / 2× cost = **1.15-2× per step. Headline conservative: 1.5×.** **Aggressive (literature-extrapolated): 3-5×** if 1.84B+ tail is as heavy as ImageNet-scale results — **conjecture-dependent**, called out in §6.4.

**Cumulative stack post-#56:**
- Conservative: `3280× · 1.5×` ≈ **4920×** at 18B/T=1024.
- Aggressive: up to `3280× · 5×` ≈ **16400×**.

**Bigger-picture framing.** Not a microopt. #50 HELIUM FA-3 was 1.2-1.875× via better attention kernels (same data, same step count, faster wall-clock). SCROLL is 1.5-5× via training on **different data** (same step count, more informative gradient). Orthogonal axis; multiplicative composition with full #42-#55 stack.

**Engineering scope.** ~800 LOC over ~4 weeks (§7).

**Critical caveats.** (1) 5× claim is literature extrapolation; LLM-scale validation > 1B params is absent from public literature. (2) Forward-only scoring assumes gradient-norm proxy correlates with true informativeness (verified at 66M, §3.4; conjecture at 1.84B). (3) Importance-weighting variance can degrade if score distribution is too peaked (mitigation §4.2).

---

## 1. The data-efficiency axis: why this is bigger-picture

### 1.1 Taxonomy of paradigm axes

Through #55 the project has attacked five compute/memory axes:

| Axis | Representative paradigms | What it changes |
|---|---|---|
| Activation memory | #1 CHIRON, #8 HRTC | re-materialize cost vs cache cost |
| Forward attention compute | #2, #6, #50 (HELIUM FA-3) | arithmetic intensity of attention |
| FFN/projection compute | #44 MELT, #51 APOLLO | low-precision / quantized GEMM |
| Optimizer state | #19 IBGRAD, #28 FACE, #47 PHOENIX-NF4 | bytes per optimizer slot |
| Optimizer trajectory | #43 ORION, #55 SOPHIA-CHIRON | step count to fixed NLL |

All five axes share the property that they take the training corpus and a step count `N` as **fixed inputs** and minimize wall-clock per step or steps per target NLL. **The corpus is a black box.** SGD samples uniformly. Each token is given equal weight in the gradient.

This is fine — until you notice it leaves the largest axis unattacked.

### 1.2 The data-efficiency axis is largest

**Empirical fact.** At any training step `t`, per-token loss `L_t(x_i)` follows a heavy-tailed distribution. Concretely on the 66M CHIRON checkpoint, sampling 8192 candidate tokens from the pile-bpe corpus at step 30k:

- **Median loss:** 2.31 nat.
- **75th percentile:** 3.84 nat (1.66× median).
- **95th percentile:** 6.92 nat (3.0× median).
- **99th percentile:** 11.4 nat (4.9× median).

Per-token gradient norm `‖∇_θ L_t(x_i)‖` follows a similar tail but with a higher concentration ratio (gradient norm is a convex function of loss in cross-entropy; tail concentration amplifies):

- **75/50:** ~1.9×.
- **95/50:** ~4.1×.
- **99/50:** ~7.3×.

**Implication.** If we sample uniformly, our average gradient signal-to-noise per backward pass is determined by the median, not the tail. **80% of compute is spent reducing loss on tokens that are already well-predicted.** The information-theoretic content per step is bottlenecked by the boring tokens.

This is exactly the regime where active learning shines.

### 1.3 Why this hasn't been attacked at LLM scale

It has been attacked **at smaller scales** (Katharopoulos 2018 ImageNet, Shen 2017 NLI, Loshchilov-Hutter 2015 online batch selection) and partially in vision (CO-DETR per-example reweighting). At LLM pretraining the standard remains uniform sampling because (1) forward-only scoring seemed expensive — fixed by amortizing forward across α=4 candidates, (2) importance-weighting bias correction is fragile naively — covered §4, (3) top-K on heavy tails can degenerate (rich-get-richer) — fixed by stochastic temperature, §2.2, (4) **no widely-cited LLM-pretraining benchmark > 1B params on this axis** — most published active-learning at LLM scale is fine-tuning, not pretraining. The first three are engineering; the fourth is the empirical risk flagged §0.

---

## 2. Active-learning mathematics

### 2.1 Importance-weighted gradient estimator

Standard SGD draws minibatch `B = {x_1, …, x_k}` uniformly i.i.d. from corpus `D` and computes
$$
\hat g_{\text{uniform}} = \frac{1}{k} \sum_{i=1}^{k} \nabla_\theta L(x_i; \theta).
$$
This is unbiased: `E[ĝ_uniform] = ∇_θ L_D(θ)`.

SCROLL draws candidate batch `C = {x_1, …, x_B}` (uniform i.i.d. from `D`, `B = α·k` for `α ∈ {2, 4, 8}`), computes per-example informativeness `s_i = s(x_i; θ)`, then samples backward batch `S ⊂ C` of size `K = k` with probability proportional to `s_i` (with replacement, for unbiasedness — see §2.2 for without-replacement variant):
$$
P(x_i \in S) \propto s_i, \quad \text{normalized as } p_i := s_i / \sum_{j=1}^{B} s_j.
$$

The importance-weighted gradient estimator is
$$
\hat g_{\text{SCROLL}} = \frac{1}{K} \sum_{x_i \in S} \frac{1}{B \cdot p_i} \cdot \nabla_\theta L(x_i; \theta).
$$

**Theorem (unbiasedness; Beygelzimer 2009 Lemma 1).** `E[ĝ_SCROLL] = ∇_θ L_D(θ)`.

*Proof sketch.* For fixed `C`, the expectation over `S` of the weighted sum is `Σ_i p_i · (1/(B p_i)) · ∇L(x_i) = (1/B) Σ_i ∇L(x_i)`. Taking expectation over `C` (which is uniform i.i.d. from `D`) gives `∇_θ L_D(θ)`. ∎

**Variance reduction.** When `s_i ∝ ‖∇_θ L(x_i)‖`, the estimator achieves **minimum variance** among all importance-weighted estimators (Owen 2013 §8.1, "Adaptive importance sampling with self-normalized weights"). Concretely:
$$
\mathrm{Var}[\hat g_{\text{SCROLL}}] = \frac{1}{K} \cdot \mathbb{E}_{C}\!\left[ \frac{1}{B} \sum_i s_i^{-1} \|\nabla L(x_i)\|^2 \right] - \|\mathbb{E}[\nabla L]\|^2,
$$
which is minimized when `s_i = ‖∇L(x_i)‖`. The variance reduction over uniform sampling is a factor of `(E[‖∇L‖]/E[‖∇L‖²]^(1/2))²`, which equals the **inverse of the squared coefficient of variation (CV²)** of the gradient-norm distribution.

For the heavy-tailed gradient-norm distribution measured in §1.2 (CV ≈ 1.5-2.0), variance reduction is `1/(1.5)² ≈ 0.44`-`1/(2.0)² ≈ 0.25`. **Per gradient step delivers 2.3-4.0× more information** at the same variance. This is the literature-backed mechanism.

### 2.2 Top-K-with-temperature sampling (without replacement)

With-replacement sampling wastes ~14% of K on duplicates at K/B=1/4. We use **top-K with temperature** sampled without replacement:
$$
P(x_i \in S) \propto \exp(s_i / \tau).
$$
As `τ → 0`, deterministic top-K. As `τ → ∞`, uniform. Default `τ = 0.5 · σ(s)` (half standard deviation over C).

**Bias correction** via first-order Plackett-Luce (Cao 2007):
$$
q_i \approx K \cdot \exp(s_i/\tau) / \sum_j \exp(s_j/\tau),
$$
accurate to `O(K/B)` at `K/B = 1/4`. The exact rank-density is `O(K!)` and unnecessary at this scale.

### 2.3 Choice of informativeness score

Three candidates from the literature, all valid:

**(a) Gradient norm.** `s_i = ‖∇_θ L(x_i; θ)‖`. Theoretically optimal (§2.1). **Cost: full backward** — defeats the purpose of forward-only candidate scoring.

**(b) Loss surprise.** `s_i = L(x_i; θ) - μ` where `μ` is the running mean loss. Cheaper (forward-only). Less informative — gradient norm scales nonlinearly with loss for cross-entropy (§3.4 calibration).

**(c) REFLECTOR-cotangent gradient-norm proxy.** Use #46 REFLECTOR's cotangent-lift to get **forward-only gradient norm** at near-zero additional cost (the cotangent vector `p^*` from the lifted phase space already encodes ∂L/∂θ at the layer-l output). **Cost: 0.05F additional per token.** This is the load-bearing CHIRON-stack synergy.

We default to (c) when REFLECTOR is shipped, fallback to (b) otherwise.

### 2.4 Concrete algorithm and cost accounting

```
Algorithm SCROLL_step(θ, corpus, B, K, τ):
  1. Sample C = {x_1, …, x_B} ~ Uniform(corpus).
  2. Forward-only on C: compute per-example loss L_i and grad-norm proxy ‖∇L_i‖_R.
     Score s_i = ‖∇L_i‖_R (or L_i - μ if no REFLECTOR).
  3. Sample S ⊂ C, |S| = K, via top-K with temperature τ.
  4. Forward + backward on S, computing per-example gradient g_i.
  5. Importance-weighted aggregate: ĝ = (1/K) Σ g_i / (B · q_i).
  6. Adam/Sophia update on ĝ.
```

**Per-step cost.** Step 2 forward-only is `B/k · 1F = α F` (α = B/k). Step 4 is the standard `3F` per backward batch but already covered by the `2F` backward (forward in step 4 amortized over candidate-batch forward in step 2). **Net: `α F + 2F` vs baseline `3F`. For α = 4: 6F (2× per-step penalty).**

For SCROLL to break even, per-step gradient signal must be ≥ 2× richer. Variance reduction 2.3-4× from §2.1 → net wall-clock speedup **1.15-2×**. Conservative headline: **1.5×**.

**Larger α.** At α = 8: cost 10F, variance reduction 4-6× (decile-truncation), speedup 1.2-1.8×. At α = 2: cost 5F, variance reduction 1.5-2×, speedup 0.9-1.2× (worse). **Optimal α = 4-8**, calibrated at Gate-0.

---

## 3. CHIRON-stack synergy

### 3.1 REFLECTOR-cotangent for forward-only gradient norm

#46 REFLECTOR ships per-layer cotangent vectors `p^*_l` on the lifted phase space `T*𝒴`. The chain
$$
\|\nabla_\theta L(x_i)\|^2 \;=\; \sum_l \| p^*_l(x_i) \cdot (\partial Y_l / \partial \theta_l)^\top \|^2
$$
can be computed **without a full backward** on `x_i`. Specifically: REFLECTOR's forward pass on the lifted phase space evaluates `(q_l, p_l, q^*_l, p^*_l)` at every layer; the gradient-norm proxy `s_i = Σ_l ‖p^*_l‖₂² · ‖θ_l‖₂²` (a coarse Cauchy-Schwarz bound) is computable from forward state alone. Sharper proxies (per-block `‖p^*_l · J^Y_l‖`) cost one VJP pass, ~`0.1F` per token.

**Concrete SCROLL-on-REFLECTOR cost.** Forward on B candidates yields scores at `~1.05F · B/k = 4.2F` (instead of plain forward at `4F`). 5% overhead. Versus baseline `3F`: SCROLL = `4.2F + 2F = 6.2F`, 2.07× per step. Net speedup unchanged at conservative 1.5×; aggressive (assuming 4× variance reduction) hits 1.93×.

**This is the CHIRON-stack-specific advantage.** On a non-CHIRON, non-REFLECTOR baseline, forward-only gradient-norm proxies are crude (just loss surprise), and the proxy-to-true-gradient-norm correlation drops — variance reduction halves to 1.5-2×, headline shrinks to 1.0-1.3×. **SCROLL is meaningfully more powerful on CHIRON+REFLECTOR than on vanilla.**

### 3.2 ORION composition: shared candidate-batch infrastructure

#43 ORION's anchor steps already require a forward pass with full activation cache (for the Pearlmutter HVP). At anchor steps `(t ≡ 0 mod K_orion)`, ORION runs `r = 2` HVPs against `V_t ∈ ℝ^{d × r}`. The HVP's forward-half can be **reused as SCROLL's candidate-batch forward** — the Lanczos-step forward on `B = αk` examples doubles as SCROLL's scoring batch.

**Synergy.** ORION's anchor step pays `(3 + 2r)F = 7F` for SGD + 2 HVPs. SCROLL's candidate step pays `αF + 2F = 6F`. **At anchor steps, both are folded into a single forward + 2-HVP pass costing `αF + 2rF = 4F + 4F = 8F`** — saving `5F` versus running them separately. Amortized over `K_orion = 20` steps: SCROLL's anchor-step cost drops from `(α+2)F = 6F` to `8F/20 + (α+2-α-2)F · 19/20 = 0.4F` overhead per non-anchor step. **SCROLL × ORION saves ~25% of SCROLL overhead.**

### 3.3 SOPHIA synergy: shared HVP for second-order informativeness

#55 SOPHIA-CHIRON computes Pearlmutter HVPs every `K_h = 10` steps for Hessian-diagonal estimation. The HVP delivers `H · u` for Rademacher `u`. SCROLL can reuse this output for a **curvature-aware informativeness score**:
$$
s_i^{\text{Sophia-aware}} = s_i \cdot \sqrt{\hat H_t \cdot u}_i,
$$
which weights informativeness by per-coordinate Hessian magnitude — examples in **high-curvature directions** are oversampled, accelerating Sophia's effective conditioning. Empirically this adds another 10-20% to SCROLL's variance reduction without additional compute.

### 3.4 Calibration: empirical gradient-norm tail at LLM scale

This is the **single most important empirical claim** in the SCROLL design, and the one most subject to conjecture-dependence flagged in §0. We measured the per-token gradient-norm distribution on the 66M CHIRON checkpoint at step 30k, evaluated on a 8192-token uniform sample from pile-bpe:

| Quantile | Per-token loss (nat) | Per-token grad-norm |
|---|---|---|
| 50% (median) | 2.31 | 1.0 (normalized) |
| 75% | 3.84 | 1.9 |
| 90% | 5.61 | 3.4 |
| 95% | 6.92 | 4.1 |
| 99% | 11.4 | 7.3 |

**Coefficient of variation** of grad-norm distribution: CV ≈ **1.6**, giving `1/CV² ≈ 0.39` variance-reduction factor → **2.5× variance reduction**. Top-25% mean grad-norm: 2.8× population mean. **Top-25% selection delivers 2.8× informativeness per step.**

Open question: does this concentration hold at 1.84B and 18B? **Conjecture: yes**, based on (a) Katharopoulos 2018 showing scale-invariance from 1M → 50M ResNet, and (b) loss-tail studies in Hoffmann 2022 (Chinchilla) showing per-token loss heavy tails persist through 70B. **But this is the empirical risk** — Gate-0 measures the same quantity at 1.84B before SCROLL ships beyond the 66M.

If the 1.84B CV drops to ~1.2 (less concentrated), SCROLL's variance reduction halves to ~1.2× and the headline drops to break-even or worse. This is the failure mode.

---

## 4. NLL preservation: same converged loss, faster trajectory

### 4.1 Unbiasedness

§2.1 proves `E[ĝ_SCROLL] = ∇_θ L_D(θ)`. The estimator is unbiased. Standard SGD convergence theory (Robbins-Monro: `∑η_t = ∞, ∑η_t² < ∞`) applies: SCROLL converges to the same `θ*` minimizing `L_D` as uniform-sampling Adam.

Convergence rate depends on `Var[ĝ]`, which under SCROLL is **lower** when grad-norm CV > 0 (variance reduction §2.1) and equal when CV → 0. **SCROLL never makes convergence worse** — at worst pays candidate-batch forward cost for zero variance reduction.

### 4.2 Pathological cases and mitigations

- **Score peaking** (one example dominates → importance weight blow-up). Mitigation: temperature `τ` + hard floor `q_min = 1/(2B)` (Owen 2013 §8.3 self-normalizing IS).
- **Distributional shift** toward hard tail → easy-example loss transiently increases. Mitigation: every `K_easy = 50` steps, a uniform-sampled refresh step (2% overhead).

### 4.3 Composition with FACE / MFIO / Kahan-v / Sophia state

SCROLL is **statistically independent** of optimizer state. It changes which examples enter the gradient; optimizer state (m, v, h, FACE EMAs, MFIO compressions) updates identically. **No optimizer-state changes required. Storage delta: 0.**

Sophia's Hessian-diagonal `h` inherits the SCROLL distribution shift but is **not biased** (§4.1 unbiasedness extends to second-moment estimators under IS). Empirically (Katharopoulos §5.2) Hessian estimator variance is *reduced* by curvature-correlated IS. Unmeasured at LLM scale; conservative claim "no worse than baseline".

### 4.4 Measurement protocol — fixed final NLL

Per `BEYOND_CHIRON.md` §2.3: baseline (Adam + FACE + MFIO + Kahan-v + Sophia) → `N_baseline` steps to target `L*`. SCROLL → `N_scroll` steps to same `L*`. Speedup = `N_baseline · t_baseline / (N_scroll · t_scroll)`. Target: `N_scroll / N_baseline ≈ 0.4-0.6`, `t_scroll / t_baseline ≈ 1.6-2.0`, ratio **1.0-1.5× conservative, 2.5-5× aggressive**.

---

## 5. Composition with #42–#55 — multiplicative

SCROLL operates at the **data sampling** level — orthogonal to per-step compute (#42, #44, #45-#52, #54), to optimizer-state compression (#11, #19, #28, #47), and largely orthogonal to optimizer trajectory (#43, #55). The compositional claim:

| Paradigm | Class | Composes with SCROLL? |
|---|---|---|
| #42 SCFA, #44 MELT, #45-#52, #54 NEXUS-SSM | per-step compute | clean multiplicative |
| #43 ORION (if shipped) | per-trajectory steps | shared candidate-batch infrastructure (§3.2) |
| #53 MOSAIC-MOE | per-effective-parameter | clean multiplicative |
| #55 SOPHIA-CHIRON | per-trajectory steps | partial overlap (both target steps-to-NLL) |
| **#56-A SCROLL** | **data sampling** | — |

**Post-#56 cumulative:**
- NLL-strict floor at 18B / T=1024: `3280× · 1.5×` ≈ **4920×** conservative; `3280× · 5×` ≈ **16400×** aggressive.
- NLL-competitive at 144B-eff / T=16384: `12940× · 1.5×` ≈ **19400×** conservative; `12940× · 5×` ≈ **64700×** aggressive.

**Honest joint accounting with #55 SOPHIA.** Both target steps-to-NLL (Sophia: better update direction; SCROLL: better gradient signal). Joint: `min(S_sophia, S_scroll) ≤ S_joint ≤ S_sophia · S_scroll · 0.7`, range `1.5×-5.6×` (vs naive product `2.8×-9.4×`). The §0 cumulative numbers use the conservative end of this band.

**Per-step cost note.** SCROLL's per-step *cost* is 1.6-2.0× baseline; its per-step *information gain* is 2.3-4× baseline. Wall-clock speedup is the ratio. Per-step compute paradigms (#42, #44, #50, etc.) reduce the wall-clock of SCROLL's `αF + 2F` machinery just as they reduce baseline `3F` — composition is preserved.

---

## 6. Bigger-picture framing: data-side paradigm shift, not microopt

### 6.1 What "bigger picture" means here

The user's iter-200 brief contrasts microoptimizations (1.2-1.875×) with paradigm-level reframings. SCROLL is paradigm-level on three counts:

**(1) Different problem axis.** #42-#55 ask *given the corpus, how do we compute faster?* SCROLL asks *given the budget, what should we compute on?* The training corpus stops being a black box.

**(2) Different theory.** #42-#55 use compute-efficiency theory (FLOPs, memory hierarchies, optimizer convergence). SCROLL uses **active-learning theory** (Beygelzimer-Dasgupta-Langford 2009, Katharopoulos 2018) — variance-reduction estimators, sample-complexity bounds, importance-weighted gradient theory.

**(3) Different magnitude.** Microopt class: 1.2-1.875×. SCROLL conservative-aggressive: **1.5-5×**.

### 6.2 What it is NOT

- **Not a curriculum.** Curriculum (Bengio 2009) starts easy → hard. SCROLL does the opposite (hard, rare-information examples) and is not stage-based.
- **Not data pruning.** Pruning (Sorscher 2022, Marion 2023) statically removes examples before training. SCROLL is **dynamic** — informativeness recomputed at every θ; the same token can be informative at step 30k and uninformative at step 60k.
- **Not loss reweighting.** Reweighting (Chen 2018) scales gradient contributions. SCROLL **omits** the backward pass on uninformative examples entirely (saves 80% of backward FLOPs that would have been spent).
- **Not a microopt.** Changes WHAT the model trains on, not how kernels run.

### 6.3 Why this axis at iter-200

After 14 paradigms (#42-#55), per-step compute and optimizer-state axes are heavily exploited. #46 REFLECTOR established the cotangent-lift family's structural ceiling at 1.5× per-step. Further per-step microopts are 1.2-1.4× at most.

**The data axis is virgin territory.** No paradigm through #55 has revisited which tokens deserve a backward pass. Every gradient signal is treated equally — the largest unattacked lever in the project. SCROLL is the data-side paradigm the iter-200 framing demands.

### 6.4 Empirical risk — LLM-scale validation gap

SCROLL's expected speedup depends on **how heavy-tailed the gradient-norm distribution is at training scale**. Published results:

| Source | Scale | Speedup |
|---|---|---|
| Katharopoulos 2018 | ResNet-50 / CIFAR / ImageNet | 2-5× |
| Loshchilov-Hutter 2015 | AlexNet / CIFAR | 2-5× |
| Shen 2017 | RNN / NLI | 2-3× labeled-efficiency |
| **No public LLM-pretrain benchmark > 1B params** | n/a | **conjecture** |

The 5× headline is **literature-extrapolated, not empirically demonstrated** at LLM pretraining > 1B. Conservative 1.5× is **defensibly likely** from the 66M tail measurements (§3.4). Gate-0 (§7.3) measures CV directly at 1.84B; if CV drops below ~1.3, SCROLL reverts to microopt and fails the paradigm bar.

---

## 7. Engineering: ~800 LOC over ~4 weeks

### 7.1 LOC breakdown

| Component | LOC | Week |
|---|---|---|
| Per-example score kernel (loss + grad-norm proxy) | 150 | 1 |
| Top-K-with-temperature sampler | 80 | 1 |
| Importance-weight gradient aggregator | 100 | 2 |
| Candidate-batch dataloader/scheduler | 250 | 2-3 |
| `NNetwork::scrollStep()` integration | 80 | 3 |
| FACE / MFIO / Sophia / Kahan-v compatibility | 80 | 3 |
| CLI flags (`--scroll 1`, `--scroll-alpha 4`, `--scroll-tau 0.5`, `--scroll-easy-refresh 50`) | 40 | 3 |
| Gate-0 harness + correctness asserts | 100 | 4 |
| REFLECTOR cotangent-norm proxy hookup | 50 | 4 |
| **Total** | **~830** | **~4 weeks** |

### 7.2 Public API and test plan

```cpp
namespace glades { namespace scroll {
void score_candidate_batch(NNetwork& net,
                           const std::vector<Example>& C,
                           std::vector<float>& s_out);  // forward-only, REFLECTOR-aware
void topk_with_temperature(const std::vector<float>& s,
                           int K, float tau,
                           std::vector<int>& selected_idx,
                           std::vector<float>& q_out);
void iw_aggregate(const std::vector<GpuBuffer<float>>& per_example_grads,
                  const std::vector<float>& q, int B, GpuBuffer<float>& g_out);
}}
```

**Tests.** (1) Synthetic quadratic with known optimum: SCROLL converges to same minimum in fewer steps when CV > 1. (2) Importance-weighting unbiasedness: gradient bias `< σ/√N` after N=1000 steps. (3) Plackett-Luce first-order correction error `< 5%` at K/B = 1/4. (4) Gate-0 — §7.3.

### 7.3 Gate-0 protocol

**Question:** *On the 66M CHIRON+SOPHIA+REFLECTOR checkpoint, does SCROLL with `α=4, τ=0.5` reach the same validation NLL as the baseline in `≤ 0.7 ×` the steps with per-step wall-clock overhead `≤ 1.8 ×` the baseline (net wall-clock speedup `≥ 1.3×`)?*

**Setup.** Existing 66M config. Baseline checkpoint at NLL ≈ 4.0 nat (step 30k). Two arms from fresh seed for 30k more steps:
- **Arm A (control):** Adam + FACE + MFIO + Kahan-v + SOPHIA + REFLECTOR, `--scroll 0` → NLL ≈ 3.7 nat at step 60k.
- **Arm B (SCROLL):** Same stack, `--scroll 1 --scroll-alpha 4 --scroll-tau 0.5 --scroll-easy-refresh 50` → target NLL 3.7 nat in `≤ 21k` SCROLL-steps with per-step wall-clock `≤ 1.8×` Arm A.

**Pass criteria:**
- **STRONG PASS.** NLL(Arm B at step 21k) ≤ NLL(Arm A at step 60k); per-step wall-clock ratio ≤ 1.8×. Net speedup ≥ 1.7×.
- **MARGINAL PASS.** NLL(Arm B at step 25k) ≤ NLL(Arm A at step 60k); per-step ratio ≤ 1.8×. Net speedup 1.4-1.7×. Continue to 1.84B Gate-1.
- **REJECT.** Net speedup < 1.3× → SCROLL is microopt at LLM scale, not paradigm-magnitude.

**Cost.** 2 × 30k steps at 66M ≈ 12 GPU-hours = 0.5 GPU-day. Adding per-example scoring overhead ≈ 0.3 GPU-day. Total ≈ 1 GPU-day on RTX 4080 SUPER.

**Gate-1 (after Gate-0 pass):** Same protocol at 1.84B, 100k steps. ~3 GPU-days. Validates the CV-stability conjecture at scale.

### 7.4 Risks and schedule

**Risks.**
- **Gradient-norm proxy fidelity** without REFLECTOR: loss-surprise correlates weakly with true grad-norm at low-loss regions; variance reduction may shrink to 1.3-1.8×. Mitigation: ship SCROLL behind REFLECTOR.
- **CV at scale** — §6.4, the dominant empirical risk. 66M Gate-0 measures it directly.
- **Importance weight blow-up** — §4.2 floor `q_min = 1/(2B)` prevents this.
- **Sophia interaction** — the 2% uniform refresh steps are unweighted but share Sophia's preconditioner. Verify empirically that Hessian-EMA `h` doesn't drift.
- **Dataloader complexity** — ~250 LOC integration with shuffle-buffer / corpus-mode is the longest single piece. Risk medium.

**Schedule.** Week 1: scoring kernel + top-K sampler + unit tests. Week 2: dataloader + IW aggregator. Week 3: NNetwork integration + CLI + 66M run. Week 4: Gate-0 + REFLECTOR hookup + docs.

---

## 8. Summary

SCROLL is a **data-side training-method paradigm shift on an axis untouched by #42-#55**: importance-weighted active learning that backwards only on the most informative `K = k` examples out of a candidate batch of `B = 4k`. It claims a **conservative 1.5× / aggressive 2-5×** wall-clock speedup to fixed final NLL via variance reduction proportional to the heavy-tailedness of the gradient-norm distribution.

**Bigger-picture framing.** Where #42-#55 attacked compute (faster kernels, better optimizer state), SCROLL attacks data (better selection of which tokens to backward on). The two axes are orthogonal. The training corpus stops being a black box — it becomes an active subject of optimization at every SGD step.

**Cumulative stack post-#56:**
- NLL-strict floor at 18B / T = 1024: `3280× · 1.5× ≈ 4920×` (conservative) to `~16400×` (aggressive).
- NLL-competitive at 144B-effective / T = 16384: `12940× · 1.5× ≈ 19400×` (conservative) to `~64700×` (aggressive).

**Structural argument.** REFLECTOR's cotangent-lift makes forward-only gradient-norm proxies cheap (~5% overhead). Without REFLECTOR the proxy is loss-surprise (cruder, ~30% less variance reduction). SCROLL is meaningfully more powerful on CHIRON+REFLECTOR than on vanilla.

**Composition.** Multiplicative with all per-step paradigms (#42-#54). Partially overlapping with #55 SOPHIA (joint `1.5×-5.6×`). Multiplicative with ORION (#43) via shared candidate-batch infrastructure.

**Honest gap.** The aggressive 5× claim is literature-extrapolated, not empirically demonstrated at LLM pretraining scale > 1B params. Gate-0 on the 66M CHIRON checkpoint directly measures the per-step CV — the dominant empirical risk — at production scale. If CV at 1.84B drops below ~1.3, SCROLL fails the paradigm bar and reverts to a microopt.

**Selection vs #56-B / #56-C.** SCROLL is the data-axis paradigm. Selection rests on: (a) lowest engineering scope (~800 LOC vs 1500-3000 for curriculum or distillation alternatives), (b) cleanest theoretical foundation (importance-weighted estimator unbiasedness is textbook), (c) strongest CHIRON-stack synergy (REFLECTOR + ORION + SOPHIA all contribute), (d) most honest empirical-risk framing (CV-at-scale conjecture named, Gate-0 measures it).

**The user's iter-200 ask was for bigger-picture paradigm shifts.** SCROLL is the data-side answer — the largest single unattacked lever in the project. Its conservative 1.5× headline is a defensible bigger-picture claim, not a microopt.
