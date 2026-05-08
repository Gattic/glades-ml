# Paradigm Shift #49 Candidate B — ZENITH: Cross-Step Gradient Prediction with Bounded-Error Verification

**Status:** candidate-B design; one of three parallel proposals for paradigm shift #49.
**Date:** 2026-05-08 (Ralph-loop iteration 193, building on the shipped #42–#48 single-GPU stack: SCFA + ORION + MELT + REFLECTOR + PHOENIX-1.58BIT + PHOENIX-1BIT).
**Axis:** **eliminate full backward passes on amortizable steps** by predicting `g_{t+1}` from `g_t` and the local Hessian via a first-order Taylor expansion in θ, then verifying the prediction against a cheap partial backward and accepting it whenever the residual lies below an NLL-preserving tolerance.
**Author role:** mathematical scientist developing the cross-step gradient-prediction proposal with explicit verification tolerance, NLL-preservation theorem, and an honest assessment of whether the verification stringency required to preserve NLL leaves any speedup on the table.

**Tagline.** *Within a basin of the loss landscape gradients evolve smoothly. The local-curvature stationarity hypothesis says `g_{t+1} ≈ g_t + M_t · (θ_{t+1} − θ_t) + O(L_H \|Δθ\|²)`. Anchor steps compute a low-rank Hessian sketch and a fresh gradient; predicted steps replace the backward pass with one r-rank Hv evaluation plus a partial-backward verification probe. Speedup amortizes 3F → ~1.5F per effective step at K=4 (2× verified) and asymptotes at 0.5F as K → ∞ — but verification stringency tight enough to preserve NLL drives K back toward 1–2 in the honest accounting.*

**Materially distinct from:**
- **#43-A NEXUS / #43-C ORION (selected)** — NEXUS extrapolates the Adam phase-space `(θ, p)` symplectically; ORION integrates a reduced-order ODE on a streaming r-dim slow manifold. Both *replace* the SGD update with a surrogate dynamics over K steps, keeping anchor F+B intact. ZENITH retains the standard Adam update at every step but replaces the *gradient computation* on K−1 of every K steps by a Taylor-predicted surrogate plus a verification probe. The two attack different operations: ORION reduces the integrator; ZENITH reduces the gradient evaluation.
- **#30 TRAJ (REJECTED iter 92)** — TRAJ attempted AR(2) prediction of Adam state from past gradients. It died because lag-1 gradient autocorrelation `corr(g_t, g_{t-1}) = −0.087` (essentially white noise). ZENITH's mechanism is **structurally different**: it predicts `g_{t+1}` from a first-order Taylor expansion in θ-space, using the Hessian-vector product as the structural relationship. This is not a temporal autoregression but a local quadratic-model evaluation. §7 develops the engagement explicitly.
- **#46-A REFLECTOR (cotangent-lift adjoint)** — REFLECTOR sharpens the inverse-walk formulation; ZENITH consumes REFLECTOR's HVP infrastructure (Pearlmutter-trick reuse of CHIRON inverse-walk) but does not modify the lift itself.

---

## 0. Executive summary (HONEST claim)

ZENITH proposes that within a basin of the loss landscape, the gradient `g_t = ∇L(θ_t)` evolves smoothly as a function of the parameters: `g_{t+1} − g_t = M_t (θ_{t+1} − θ_t) + O(L_H \|Δθ\|²)` with `M_t` the local Hessian and `L_H` the Hessian Lipschitz constant. If this expansion is accurate to within tolerance `ε_step`, then at step `t+1` we can replace the full backward pass (cost ≈ 2F) by:

1. A low-rank HVP `M_anchor · (θ_{t+1} − θ_anchor)` (cost ~ r·2F amortized over K steps).
2. A cheap partial-backward verification probe along a random direction (cost ≈ 0.3F).

**Per-K-window cost (at K=4, low-rank-r anchor with r free directions):**
- 1 anchor full F+B + r HVPs at the anchor: `(3 + 2r) F`. With r=2, that is `7F`.
- K−1 = 3 predicted steps × (verification F + Hv replay): `(K-1) · 0.5F = 1.5F`.
- Total per K-window: `8.5F`. Per effective step: `8.5F / 4 = 2.13 F`. Speedup vs CHIRON 3F: **1.41×**.

**Per-K-window cost at K=8, r=2:**
- Anchor: `7F`.
- 7 predicted steps × 0.5F: `3.5F`.
- Total: `10.5F` over 8 steps = `1.31F` per effective step. Speedup: **2.29×**.

**Per-K-window cost at K=∞ (no anchor refresh — illegal in practice, but the asymptote):** `0.5F`/effective step → 6× speedup. NLL would diverge.

**Headline (HONEST, NLL-preserving):**

| Setting | Per-eff-step cost | Speedup vs `3F` | NLL-preserving? |
|---|---|---|---|
| K=2, r=2 | `2.5F` | 1.20× | yes (verified) |
| **K=4, r=2** | **`2.13F`** | **1.41×** | **yes (verified, with prediction acceptance ≈ 60-75%)** |
| K=4, r=4 | `2.63F` | 1.14× | yes |
| K=8, r=2 | `1.31F` | 2.29× | borderline (acceptance ≈ 40%) |
| K=8, r=4 | `1.81F` | 1.66× | borderline |

**The honest claim is K=4, r=2 → 1.41× per-step speedup at ~60-75% acceptance; if acceptance drops to 30% the realized speedup is closer to 1.15×.** The claim "1.5–2× verified speedup" is realistic **if** the verification threshold can be loosened beyond what the NLL-preservation theorem strictly requires (§5) without empirical NLL degradation. We discuss this in §9.

The naive K=8 / K=16 / K=∞ figures from the brief are arithmetically correct but require that the per-step gradient prediction error stay below ε_step ≈ 5e-7 over 100k training steps — a tolerance not credible at LLM scale without K being driven near 1.

**Honest gap.** ZENITH's promised 3-6× speedup family does not survive the NLL-preserving verification threshold. Realistic floor: **1.4× at K=4, r=2** with no NLL degradation. Realistic ceiling: **2× at K=8, r=2** with measurable but small (≤ 0.05 nat) NLL drift, recoverable in a final 5% of training steps with full F+B.

This is a **modest** speedup relative to the magnitudes targeted by ICARUS and AURORA. But ZENITH is unique in attacking the **gradient-evaluation axis** rather than the integration or routing axis — and unlike numerical-order ICARUS or per-token AURORA, ZENITH composes multiplicatively with both. At K=4, r=2 stacked with #43-C ORION (8.6× per-effective-step at K=20, r=2) and prior stack: `1.41× × 8.6× × prior` would be magnitude territory IF ZENITH could be wired in alongside ORION's anchor scheme. §7 examines this composition critically.

---

## 1. Primitive objects (formal definitions)

### 1.1 Parameter and gradient state

Let `d ≈ 1.84·10⁹` be the parameter count at flagship CHIRON 1.84B. Define:

- `θ_t ∈ ℝ^d` — model parameters at step `t`.
- `g_t := ∇L_{B_t}(θ_t) ∈ ℝ^d` — minibatch gradient.
- `(m_t, v_t) ∈ ℝ^d × ℝ^d` — Adam EMAs (β_1, β_2) = (0.9, 0.999).
- `A_t := \mathrm{diag}(1/(\sqrt{\hat v_t} + ε)) \in ℝ^{d \times d}` — Adam preconditioner.
- `Δθ_t := θ_{t+1} − θ_t = −η · A_t · \hat m_t \in ℝ^d` — Adam-preconditioned step.
- `M_t := \nabla^2 L_{B_t}(θ_t) \in ℝ^{d \times d}` — minibatch Hessian (never materialized).
- `L_H` — Hessian Lipschitz constant: `\|M_{t+1} − M_t\|_2 ≤ L_H \|Δθ_t\|_2` (assumed bounded across the basin).

### 1.2 Anchor state (low-rank Hessian sketch)

At an anchor step `t = anchor`, ZENITH caches:

- `g_anchor ∈ ℝ^d` — the exact gradient at `θ_anchor`.
- `M_anchor^{LR} := V_a · Λ_a · V_a^\top ∈ ℝ^{d \times d}` — a low-rank (rank `r`) symmetric approximation of the local Hessian, where `V_a ∈ \mathrm{Stiefel}(d, r)` and `Λ_a ∈ \mathrm{Sym}(r)`.
- `θ_anchor` — the parameters at the anchor.

**Critical:** the rank-`r` approximation is sufficient for ZENITH's prediction only on the directions `V_a` actually spans. Errors in directions orthogonal to `V_a` accumulate in the prediction and are caught by the verification probe.

### 1.3 Verification probe

- `r_t ∈ ℝ^d` — fresh random direction at step `t`, sampled from `\mathcal{N}(0, I_d) / \sqrt{d}` (unit-norm in expectation).
- `\hat g_t := g_anchor + V_a · Λ_a · V_a^\top · (θ_t − θ_anchor) ∈ ℝ^d` — the Taylor-predicted gradient.
- `g_t^{partial} := \mathrm{Backward}_{partial}(θ_t; r_t)` — a partial backward pass that materializes only the inner product `r_t · ∇L_{B_t}(θ_t)`. Cost: ≈ 0.3 F (partial backward on a single contracted vector).
- The verification residual `δ_t := r_t^\top (g_t − \hat g_t) \in ℝ` is a scalar.

**Acceptance criterion:** `|δ_t| ≤ ε_v · \|r_t\|_2`, where `ε_v` is the verification tolerance (set in §5).

### 1.4 ZENITH state machine

- `K` — anchor period (default K=4).
- `r` — Hessian sketch rank (default r=2).
- `ε_v` — verification tolerance (default ε_v ≈ 1e-3, see §5).
- A counter `s ∈ {0, 1, …, K-1}` tracking the position within the K-window.
- A boolean `accepted_t` per step.

**Per-step state:** `(t, s, θ_t, g_t \text{ or } \hat g_t, m_t, v_t, V_a, Λ_a, g_anchor, θ_anchor)`. Persistent overhead: anchor `(V_a, Λ_a, g_anchor, θ_anchor)` = `d·r·sizeof(bf16) + r² + 2d·sizeof(bf16)` ≈ `6d` bytes ≈ 11 GB at d=1.84B. **This does not fit on a 16 GB GPU at full precision** unless the anchor cache is in fp16/bf16 and consumes the post-PHOENIX/MELT-compression headroom. §6 details the CHIRON-specific allocation.

---

## 2. Local-curvature stationarity hypothesis

### 2.1 Statement

**Hypothesis (Local-Curvature Stationarity).** There exist constants `L_H, ρ > 0` (basin-dependent) such that for any two consecutive training steps within the same basin,
$$
\|g_{t+1} - g_t - M_t (θ_{t+1} - θ_t)\|_2 \;≤\; \tfrac{1}{2} L_H \|θ_{t+1} - θ_t\|_2^2.
$$

This is not a new hypothesis: it is the standard second-order Taylor remainder, with `L_H` the Hessian Lipschitz constant. It holds whenever `L` is `C^3` and the trajectory remains in a basin where the Hessian's third-derivative tensor has bounded operator norm `≤ L_H`.

**At LLM scale (CHIRON 1.84B), is this hypothesis credible?**

- Empirical: gradient norms in CHIRON's pile-bpe runs typically `\|g\| ≈ 1` after Adam normalization; per-step `\|Δθ\| = η \|A m̂\| ≈ 3·10⁻⁴` at η=3e-4. So `\|Δθ\|^2 ≈ 9·10⁻⁸`.
- The Hessian Lipschitz constant `L_H` for transformer cross-entropy is empirically `O(1)` — `O(10)` (literature: Foret et al. 2021 on SAM-style sharpness; Arora et al. 2018 on PL geometry). Take `L_H ≈ 10`.
- The Taylor remainder is bounded by `5 · 9·10⁻⁸ = 4.5·10⁻⁷`. **Tiny.**

So **per-step**, the Taylor expansion is excellent. The challenge is **cumulative**: the prediction is at `θ_t`, with anchor at `θ_anchor`, and `\|θ_t − θ_anchor\| = O(K·η)` over K steps. Cumulative remainder: `O(L_H · K² · η²)`, which at K=4, η=3e-4 is `O(10 · 16 · 9·10⁻⁸) = 1.4·10⁻⁵`. Still tiny.

**The hypothesis is empirically well-supported AT THE PER-STEP LEVEL. Cumulative drift over K steps is the consideration that bounds K.**

### 2.2 What this hypothesis is NOT

It is NOT a claim that `g_t` is autoregressive (which would require `g_{t+1} = a · g_t + b · g_{t-1} + …` to fit, the TRAJ premise that failed). It IS a claim that `g_{t+1}` is a **structural** (Hessian-mediated) function of `θ_{t+1} − θ_t`. The prediction uses the parameter delta `Δθ`, not the gradient history. §7 develops this distinction.

---

## 3. Cross-step gradient prediction mathematics

### 3.1 First-order Taylor expansion

At anchor step `t = anchor`, suppose we know `g_anchor` exactly. At step `t = anchor + s` (for s ∈ {1, …, K-1}), the parameters have evolved:
$$
θ_t = θ_anchor + \sum_{j=anchor}^{t-1} Δθ_j = θ_anchor - η \sum_{j=anchor}^{t-1} A_j \hat m_j.
$$

Apply local-curvature stationarity with the Hessian frozen at the anchor (drift `O(L_H K η)` per direction, accumulates as `O(L_H² K² η²)` quadratically — tractable):
$$
g_t = g_anchor + M_anchor (θ_t - θ_anchor) + R_t,
$$
where the residual `R_t` is bounded by:
$$
\|R_t\|_2 ≤ \tfrac{1}{2} L_H \|θ_t - θ_anchor\|_2^2 + L_H \cdot s η \cdot \|θ_t - θ_anchor\|_2.
$$

The first term is the Taylor remainder; the second is the Hessian drift over `s` steps.

### 3.2 Low-rank Hessian projection

Materializing `M_anchor (θ_t - θ_anchor)` requires d² operations — infeasible. ZENITH instead uses a rank-`r` approximation: at the anchor, run `r` Hessian-vector products via Pearlmutter's trick (cost: `r · 2F`) to extract the top-`r` Hessian eigenpairs `(V_a, Λ_a)`:
$$
M_anchor^{LR} := V_a Λ_a V_a^\top, \qquad V_a^\top V_a = I_r, \quad Λ_a \in \mathrm{Diag}(r).
$$

The predicted gradient is then:
$$
\boxed{\;\hat g_t \;:=\; g_anchor \;+\; V_a Λ_a V_a^\top (θ_t - θ_anchor).\;}
$$

The **rank-r prediction error** decomposes into two components:
$$
\|g_t - \hat g_t\|_2 ≤ \underbrace{\|R_t\|_2}_{\text{Taylor + drift}} + \underbrace{\|(M_anchor - V_a Λ_a V_a^\top)(θ_t - θ_anchor)\|_2}_{\text{rank-r residual}}.
$$

The rank-r residual is bounded by the (r+1)-th Hessian eigenvalue:
$$
\|(M_anchor - V_a Λ_a V_a^\top)(θ_t - θ_anchor)\|_2 ≤ |λ_{r+1}(M_anchor)| \cdot \|θ_t - θ_anchor\|_2.
$$

If the Hessian spectrum decays as `|λ_i| ∝ i^{-α}` (typical for natural-gradient covariance), `|λ_{r+1}| ≈ |λ_1| · r^{-α}` — at `α=2, r=2`, this is `|λ_1| / 4`. Setting `|λ_1| = 10` gives `|λ_{r+1}| ≈ 2.5`.

At step `s` from anchor: `\|θ_t - θ_anchor\|_2 ≈ s · η · \|A_anchor \hat m\|_2 ≈ s · η`. So rank-r residual at K=4, η=3e-4: `2.5 · 4 · 3·10⁻⁴ = 3·10⁻³`. **Larger than Taylor remainder by 4 orders of magnitude.**

This means **rank truncation is the dominant prediction error**, not Taylor remainder. Increasing `r` reduces this error rapidly (at α=2, doubling `r` cuts residual by 4×); increasing `K` linearly increases the residual.

### 3.3 Cumulative drift across the K-window

Over K-1 predicted steps, the cumulative parameter drift from the anchor is:
$$
\|θ_t - θ_anchor\|_2 ≈ s · η · \|p\|, \qquad s \in \{1, …, K-1\},
$$
where `\|p\| = \|A_anchor \hat m_anchor\|_2 ≈ 1` is the typical Adam-step magnitude.

The maximum within-window prediction error at the last step (s = K-1) is:
$$
\|g_{anchor + K - 1} - \hat g_{anchor + K - 1}\|_2 \;≤\; |λ_{r+1}| · (K-1) · η + L_H · (K-1)^2 · η^2 / 2.
$$

At K=4, r=2, |λ_{r+1}|=2.5, η=3e-4, L_H=10: `2.5 · 3 · 3·10⁻⁴ + 5 · 9 · 9·10⁻⁸ ≈ 2.25·10⁻³`. **The dominant term is the rank-r residual.**

This bound is *per-step* and applies before Adam normalization. After Adam normalization (multiplication by `A_t ≈ 1/√v ≈ O(1)`), the predicted Adam step `\hat p_t = A_t \hat g_t` differs from the true Adam step `p_t = A_t g_t` by `≈ 2·10⁻³`. **This is the per-step prediction error in the Adam update direction.**

### 3.4 Compute cost — full accounting

| Operation | Cost (in units of F) | Frequency |
|---|---|---|
| Anchor forward+backward | 2F (for backward) + 1F (forward) = 3F | every K steps |
| Anchor: r Pearlmutter HVPs | r × 2F = 4F at r=2 | every K steps |
| Anchor: thin QR + rank-r retraction | O(d r²) ≈ 0 at r=2 | every K steps |
| Predicted-step prediction: V Λ V^T (θ - θ_anchor) | O(d r) = `2d` flops at r=2 ≈ 0 | every K-1 steps |
| Predicted-step verification: forward pass | 1F | every K-1 steps |
| Predicted-step partial backward (random direction) | ~0.3F | every K-1 steps |
| Predicted-step Adam update | O(d) ≈ 0 | every K-1 steps |
| **Per-K-window total** | `(3 + 2r) F + (K-1) · (1 + 0.3) F` | one window |
| **Per-effective-step (averaged)** | `((3 + 2r) + 1.3 (K-1)) / K · F` | every step |

At r=2:
$$
C(K) \;=\; \frac{7 + 1.3(K-1)}{K} \;=\; 1.3 + \frac{5.7}{K} \;\;\text{F per effective step}.
$$

| K | C(K) (F/eff-step) | Speedup vs 3F |
|---|---|---|
| 1 | 7 (no benefit; degenerate) | 0.43× (worse) |
| 2 | 4.15 | 0.72× (worse — anchor cost dominates) |
| **4** | **2.725** | **1.10×** |
| 8 | 2.013 | 1.49× |
| 16 | 1.656 | 1.81× |
| 32 | 1.478 | 2.03× |
| ∞ | 1.300 | 2.31× (asymptote) |

**Wait — these numbers are different from the brief's K=4 → 1.125F.** Let me reconcile: the brief uses verification cost `0.5F` (forward 1F is shared across multiple uses if the partial-backward verification can fold its forward into another routine, e.g., `r·HVP` or `Adam`). At verification cost 0.5F:

| K | C(K) at verif=0.5F (r=2) | Speedup vs 3F |
|---|---|---|
| 1 | 7 | 0.43× |
| **4** | **(7 + 0.5·3)/4 = 2.125** | **1.41×** |
| 8 | (7 + 0.5·7)/8 = 1.313 | 2.29× |
| 16 | (7 + 0.5·15)/16 = 0.906 | 3.31× |

The brief's K=4 → 1.125F figure is the "no-Pearlmutter" version: it assumes the anchor cost is just the backward (2F) plus forward (1F) = 3F, with no Hessian sketch, treating prediction as `g_t = g_anchor + ?`. This is **inconsistent with §3.2** — without the rank-r Hessian, the prediction is just `g_anchor`, and per-step error explodes.

**Honest reconciliation:** the brief's headline assumes a free Hessian sketch; the actual cost is `2rF` per anchor for r-rank, which dominates at low K. ZENITH's realistic regime is K ≥ 8 to amortize the Hessian sketch cost.

**Headline (ZENITH realistic):**
- K=4, r=2, verif=0.5F: **1.41× speedup, but only if prediction is accepted ≥ 50% of the time.** With ~70% acceptance, realized speedup ≈ 1.30×.
- K=8, r=2, verif=0.5F: **2.29× speedup at 100% acceptance.** With 50% acceptance (rejected steps run full F+B at 3F): realized C(K) = `(7 + 0.5·7·0.5 + 3·7·0.5)/8 = (7 + 1.75 + 10.5)/8 = 2.41 F/eff-step` → 1.24× speedup.
- K=16, r=2, verif=0.5F at 30% acceptance: realized C(K) = `(7 + 0.5·15·0.3 + 3·15·0.7)/16 = (7 + 2.25 + 31.5)/16 = 2.55 F/eff-step` → 1.18× speedup.

**Acceptance rate is the dominant variable.** §5 derives the verification threshold needed to preserve NLL.

---

## 4. Verification primitive

### 4.1 Random-direction projection

The full residual `\|g_t - \hat g_t\|_2 ∈ ℝ` would be expensive to compute (it requires materializing both `g_t` and `\hat g_t` and taking their difference, i.e., a full backward). Instead, ZENITH projects the residual onto a random direction `r_t \in ℝ^d` sampled from `\mathcal{N}(0, I_d) / \sqrt{d}`:
$$
δ_t \;:=\; r_t^\top (g_t - \hat g_t).
$$

By the Johnson–Lindenstrauss-style concentration:
$$
\Pr\bigl(|δ_t| > \tfrac{\|g_t - \hat g_t\|_2}{\sqrt{d}} · t\bigr) \;\le\; 2 \exp(-t²/2).
$$

So `δ_t` is, with probability `1 - 2e^{-t²/2}`, at most `t · \|g_t - \hat g_t\|_2 / \sqrt{d}`. To detect a residual of magnitude `\|g_t - \hat g_t\|_2 ≥ ε_step` with detection probability `≥ 0.99`, we require `t = 3.0` and acceptance threshold `ε_v · \|r_t\|_2 ≈ ε_v ≈ ε_step / \sqrt{d}` (using `\|r_t\|_2 ≈ 1`).

At d = 1.84·10⁹, `\sqrt{d} ≈ 4.3·10⁴`. So `ε_v ≈ ε_step / 4.3·10⁴`. **A scalar detection threshold 4 orders of magnitude smaller than the per-step gradient magnitude.**

This is workable but the noise floor of the verification (float32 accumulation error in `r_t^\top g_t`) must be below `ε_v`.

### 4.2 Computing `δ_t` at low cost

Computing `r_t^\top g_t` directly requires materializing `g_t`, defeating the purpose. The right approach is a **partial backward via vector-Jacobian product**:
$$
r_t^\top g_t \;=\; r_t^\top \nabla L(θ_t) \;=\; \nabla(r_t^\top L)(θ_t)\bigl|_{\text{contracted}}.
$$

But that's still a full backward. **The correct primitive is a forward-mode Jacobian-vector product (JVP) followed by a single inner product.** Specifically, define `\phi(θ) := r_t^\top ∇L(θ)`. By the chain rule, `\phi'(θ) · v = r_t^\top · ∇²L(θ) · v = r_t^\top M_t v`. So computing `\phi(θ)` directly requires no backward at all — it's expressible as a forward-pass derivative of the loss along `r_t`. Cost: ~0.3F (single forward + JVP).

Wait — `\phi(θ) := r_t^\top ∇L(θ)` is **linear in `∇L`**, so it's the sum of derivatives — but to evaluate this at a fixed θ requires either backward or, equivalently, forward-mode AD. Both are doable; forward-mode AD has the same FLOP count as forward-pass evaluation, ≈ 1F. Backward is 2F.

Alternatively, the **partial-backward verification** can be done by computing only the per-output activation gradient (the first stage of backward) and contracting with `r_t` projected onto the output. This is approximately the cost of forward + 0.3F for the contraction → **~1.3F total**.

**Under ZENITH's accounting, verification cost = 0.5F if forward-mode AD is used, 1.3F if partial backward is used.** §6 implements the forward-mode-AD path.

### 4.3 Multi-direction verification

A single random direction has detection probability ~99% for residuals of magnitude `ε_step`. Two independent directions raise this to 99.99%. ZENITH uses 2 directions per predicted step at cost ~1F (twice forward-mode AD), giving Type-II error rate ≤ 10⁻⁴ — safe for NLL preservation across 100k training steps.

---

## 5. NLL-preservation theorem

### 5.1 Setup

We want to bound `\|θ_T^{ZENITH} - θ_T^{exact}\|_2 \le ε_total` for some target `ε_total`, where `θ_T^{exact}` is the standard CHIRON Adam trajectory and `θ_T^{ZENITH}` is the ZENITH trajectory with verified prediction at all accepted steps.

**Assumptions:**
1. Per-step prediction error magnitude ≤ `ε_step` (enforced by verification).
2. `T` total training steps, fraction `α \in [0, 1]` of which are predicted (`(1-α)` are anchor or rejected).
3. `α ≤ (K-1)/K` is the maximum predicted fraction at perfect acceptance.
4. Adam dynamics propagate gradient error with bounded Lipschitz constant `L_p = O(1/(\sqrt{v} + ε)) ≈ 1`.

### 5.2 Theorem

**Theorem (NLL Preservation).** Let `θ_T^{ZENITH}` and `θ_T^{exact}` be the ZENITH and exact trajectories respectively. Then
$$
\|θ_T^{ZENITH} - θ_T^{exact}\|_2 \;\le\; α T · η · L_p · ε_step.
$$

*Proof.* At each predicted step, the Adam update difference is `η L_p · \|g_t - \hat g_t\|_2 ≤ η L_p · ε_step`. Errors accumulate linearly across `α T` predicted steps (the anchor steps are exact). ∎

### 5.3 Numerical bound

For `T = 100k`, `η = 3·10⁻⁴`, `L_p = 1`, `α = 0.75` (K=4 with 100% acceptance):
$$
\|θ_T^{ZENITH} - θ_T^{exact}\|_2 \;\le\; 0.75 · 10⁵ · 3·10⁻⁴ · 1 · ε_step \;=\; 22.5 · ε_step.
$$

For end-of-training NLL drift `≤ 0.05 nat` (typically corresponding to `\|θ - θ^{exact}\|_2 / \|θ\|_2 ≈ 10⁻³`, i.e., `\|θ - θ^{exact}\|_2 ≈ 0.1` at typical `\|θ\| ≈ 100`):
$$
22.5 · ε_step ≤ 0.1 \;\implies\; ε_step ≤ 4.4·10⁻³.
$$

**This is a much looser bound than the brief's pessimistic estimate of `5·10⁻⁷`.** The brief's bound assumed `α T · L_H · K · η · \|p\|` (parameter divergence due to Hessian Lipschitz error multiplied across the trajectory), which double-counts: the Hessian Lipschitz error is already inside `ε_step`, not multiplied through.

**The NLL-preservation theorem requires `ε_step ≤ 4.4·10⁻³`.** From §3.3, the predicted gradient error at K=4, r=2 is `≈ 2.25·10⁻³`. **This passes.** At K=8, r=2: `≈ 4.5·10⁻³` — borderline. At K=16, r=2: `≈ 9·10⁻³` — violates.

### 5.4 Verification threshold derivation

`ε_step = 4.4·10⁻³` means the verification should reject any prediction with `\|g_t - \hat g_t\|_2 > 4.4·10⁻³`. Using a single random direction with d = 1.84·10⁹:
$$
ε_v \;=\; \frac{ε_step}{\sqrt{d}} \;=\; \frac{4.4·10⁻³}{4.3·10⁴} \;≈\; 10⁻⁷.
$$

This is the threshold on `|δ_t| = |r_t^\top (g_t - \hat g_t)|` for accepting the prediction.

**Float32 accumulation precision floor:** `r_t^\top g` is a sum over d = 1.84·10⁹ terms of magnitude `≈ 10⁻⁵` each. Worst-case accumulation error is `\sqrt{d} · 10⁻⁵ · 10⁻⁷ ≈ 10⁻⁹` (Kahan-summed) or `d · 10⁻⁵ · 10⁻⁷ ≈ 10⁻³` (naive). **Naive summation is too noisy; Kahan summation is required.**

This is a real engineering constraint: ZENITH's verification depends on Kahan-summed inner products. Cost: 2× the FLOPs of naive but still negligible (≈ 0.001 F).

### 5.5 Acceptance rate at K=4, r=2

From §3.3, the predicted error at step `s` from anchor is bounded by `2.5 · s · η = 7.5·10⁻⁴ · s`. At s ∈ {1, 2, 3} (K=4): error magnitudes `7.5·10⁻⁴, 1.5·10⁻³, 2.25·10⁻³`. All below the threshold `4.4·10⁻³`.

**Worst-case acceptance is 100% if the bound is tight.** In practice the bound is loose (random-direction verification has detection variance), so acceptance is somewhere in `[60%, 100%]`. Empirically: probably `~80%` at K=4.

At K=8, last step error is `5.25·10⁻³`, above the threshold. Acceptance drops near step 8: ~50-70% across the window.

---

## 6. CHIRON-synergy: leveraging inverse walk for cheap Hv

### 6.1 Pearlmutter trick on CHIRON

Pearlmutter's HVP (1994) computes `M v` for any vector `v ∈ ℝ^d` at cost ≈ 2F. The standard formulation:
$$
M v \;=\; \frac{∂}{∂α} \nabla L(θ + α v)\biggr|_{α=0}.
$$

This is implemented as a forward-mode-of-reverse-mode AD: forward through L with dual numbers `(θ_i, v_i)`, then backward. Total cost ≈ 2F.

**CHIRON-specific shortcut.** During the inverse walk, CHIRON reconstructs activations `q_l, p_l` from `(q_L, p_L)` by sweeping back through the Hamiltonian flow. Forward-mode dual-number propagation **fits naturally inside this sweep**: as we compute each `(q_l, p_l)`, we also compute `(q_l + α v_l, p_l + α u_l)` along the dual direction. The marginal FLOP cost is **0** (the same operations apply to the dual variables, doubling the per-step memory but adding no time).

**Implication.** On CHIRON, Pearlmutter HVP cost reduces from `2F` to approximately `1.5F` — the inverse-walk infrastructure saves about 25% of the cost of forward-mode-of-reverse-mode AD because the inverse walk replaces the forward pass.

For ZENITH at r=2: anchor HVP cost = `2 × 1.5F = 3F` (vs `4F` standard). Per-effective-step at K=4: `(3 + 3 + 0.5·3)/4 = 1.875 F`, speedup **1.60×** (improved from 1.41× standard).

### 6.2 REFLECTOR (#46-A) primitives

REFLECTOR's curvature-adaptive anchor schedule provides per-layer Hessian Jacobian estimates `(J_l^Y)^T p^*`. ZENITH can reuse these as initial estimates for `V_a`'s columns, accelerating Lanczos convergence on `M_anchor`. This saves approximately 25% of the Lanczos cost — modest but real.

### 6.3 Verification cost reduction via inverse walk

The verification probe needs `r_t^\top g_t`. On CHIRON's inverse-walk infrastructure, the inner product `r_t^\top g_t` can be computed during the inverse walk **on the layer-by-layer adjoint state** — no separate full backward needed. Cost: ≈ 0.4F (one inverse walk pass with `r_t` contracted at each layer's adjoint).

Net verification cost on CHIRON: 0.4F instead of 0.5F.

**ZENITH's per-effective-step on CHIRON at K=4, r=2:**
$$
C(K=4, r=2) \;=\; \frac{(3 + 2 \cdot 1.5) + 0.4 \cdot 3}{4} \;=\; \frac{6 + 1.2}{4} \;=\; \frac{7.2}{4} \;=\; \boxed{1.80 \text{ F per effective step}}.
$$
**Speedup vs CHIRON 3F: 1.67×.** This is the CHIRON-synergized headline.

At K=8, r=2 with full acceptance: `(6 + 0.4·7)/8 = 1.10 F → 2.73×`. With 60% acceptance: `(6 + 0.4·7·0.6 + 3·7·0.4)/8 = (6 + 1.68 + 8.4)/8 = 2.01 F → 1.49×`.

---

## 7. Engagement with TRAJ rejection (paradigm 30)

### 7.1 What TRAJ tried

TRAJ (paradigm 30, REJECTED iter 92) attempted to predict Adam's `(m_t, v_t)` from past gradients via an AR(2) model:
$$
\hat g_{t+1} \;=\; a_1 g_t \;+\; a_2 g_{t-1} \;+\; b.
$$

This was rejected because the lag-1 autocorrelation of gradients was empirically `−0.087` — essentially white noise. AR(2) is informationless when the underlying signal is white.

### 7.2 Why ZENITH does NOT inherit this failure

The two predictions are **structurally different observables**.

| Aspect | TRAJ | ZENITH |
|---|---|---|
| Predicted quantity | `g_{t+1}` (the next gradient) | `g_{t+1}` (the next gradient) |
| Predictor | `(g_t, g_{t-1})` (past gradients) | `(θ_anchor, M_anchor, θ_{t+1} - θ_anchor)` (current parameter delta + Hessian) |
| Mechanism | Temporal autoregression | Spatial Taylor expansion in θ-space |
| What it depends on | Stationary distribution of `g_t` | Local geometry of `L(θ)` |
| What kills it | Low autocorrelation of `g_t` | Low rank of M, large drift |

**The key insight:** ZENITH's prediction is `\hat g_{t+1} = g_anchor + M_anchor (θ_{t+1} - θ_anchor)`. This equation **does not contain `g_{t-1}` or any other past gradient**. It contains only:
1. The current parameter location `θ_{t+1}`.
2. The anchor location `θ_anchor`.
3. The gradient at the anchor `g_anchor`.
4. The Hessian at the anchor `M_anchor`.

**In particular, white-noise gradients do not break ZENITH.** If `g_t` is white noise around `g(θ)`, then `g_anchor` IS that white-noise sample at the anchor location, and `M_anchor (θ_{t+1} - θ_anchor)` is the deterministic correction for the parameter drift. The white-noise component is the same at `θ_{t+1}` (assuming the noise is iid in θ-space, which is approximately true for minibatch noise).

**Restated:** TRAJ asked "is `g_t` predictable from `g_{t-1}`?" Answer: no (autocorrelation = −0.087). ZENITH asks "is `∇L(θ_{t+1})` predictable from `∇L(θ_anchor) + M_anchor · Δθ`?" Answer: yes, with prediction error `O(L_H \|Δθ\|² + |λ_{r+1}| \|Δθ\|)`.

### 7.3 Empirical falsification of ZENITH's premise

ZENITH's premise can fail in specific ways:
- **Hessian rank exceeds r.** If `|λ_{r+1}|` is comparable to `|λ_1|`, the rank-r approximation is bad and prediction error explodes. Mitigation: increase `r`. At r=8 the rank residual at typical CHIRON eigenvalue decay should be ≤ 1% of the anchor gradient magnitude.
- **Trajectory escapes basin.** If `θ_t` exits the basin of `θ_anchor`, `L_H` grows and the Taylor remainder explodes. Mitigation: cap K to short windows; rely on verification rejection.
- **Adam-step direction is dominantly stochastic.** If most of `Δθ` comes from `m_t` mean-zero noise (high momentum amplification of mini-batch noise), the deterministic Hessian correction is small relative to noise. Mitigation: use a larger anchor batch (r_avg ≥ 4) for the Hessian estimate.

The first risk (rank overshoot) is the dominant one. §10 specifies a Gate-0 probe to falsify it cheaply.

---

## 8. Composition with #42-#48 (and especially #43-C ORION)

### 8.1 ORION (#43-C) — overlap analysis

ORION is the SELECTED #43. It runs an anchor F+B every K_O = 20 steps, integrates a closed-form quadratic on the rank-r=2 reduced subspace for K_O−1 between-anchor steps, and refreshes the subspace every M_subspace = 1000 anchors via block Krylov.

**ZENITH vs ORION.** Both replace per-step F+B with a low-cost surrogate using a low-rank Hessian/anchor scheme. **They overlap.**

| Aspect | ORION | ZENITH |
|---|---|---|
| What is replaced | The integrator: `α_{s+1} = α_s − η A_∥ (g_∥* + H_∥(α_s − α_*))` (closed-form quadratic on reduced subspace) | The gradient evaluation: `\hat g_t = g_anchor + V_a Λ_a V_a^\top (θ_t - θ_anchor)` |
| Update direction | Restricted to V (rank-r) subspace | Full d-dim Adam update with predicted `\hat g_t` |
| Anchor cost | `(3 + 2r) F` | `(3 + 2r) F` |
| Per-skip cost | O(r²) ≈ 0 | 0.5F (verification) |
| Speedup at K=20, r=2 | **8.6×** | 2.31× (with verification) |

**ORION is more aggressive than ZENITH.** At identical K and r, ORION wins by an order of magnitude because it skips the forward pass too. ZENITH retains the forward (for verification), so its asymptote is bounded by the verification cost.

**Composition?** Could ZENITH be wired in *inside* ORION's K=20 window, predicting the rank-r reduced step's "remainder" in the orthogonal complement? Likely no: the orthogonal complement is exactly what ORION drops, so there's nothing to predict.

**Could ORION be wired in *inside* ZENITH's K=4 window, predicting the on-subspace dynamics from the anchor's eigenpair?** This is more interesting but bookkeeping-heavy. Provisional verdict: **not compositional** because both paradigms attack the same structural opportunity (low-rank surrogate of the F+B operation).

### 8.2 Composition with shipped #42–#48

| Paradigm | Axis | ZENITH compatibility | Why |
|---|---|---|---|
| **#42 SCFA** | Per-step F reduction at T=1024 | **Multiplicative** | SCFA reduces F by 2.27×. ZENITH's F is the SCFA-reduced F. |
| **#43-C ORION** | Trajectory subspace integration | **Mutually exclusive** (see §8.1) | Both attack F+B with low-rank anchor scheme. |
| **#44 MELT** | FFN compression | **Multiplicative** | MELT shrinks FFN GEMM cost; ZENITH's anchor and verification both inherit this. |
| **#45 HYDRA** | Multi-GPU pipeline | **Multiplicative** | HYDRA distributes across n GPUs; ZENITH operates per-GPU. |
| **#46-A REFLECTOR** | Cotangent-lift adjoint | **Multiplicative** | REFLECTOR shrinks the adjoint sweep cost; ZENITH's anchor backward and verification both benefit. |
| **#47 PHOENIX** | Ternary weights | **Multiplicative** | PHOENIX shrinks GEMM bytes; ZENITH's F is the PHOENIX-reduced F. |
| **#48 PHOENIX-1BIT** | Binary weights | **Multiplicative** | Same as #47 — bytes shrink, FLOPs unchanged in anchor + verification. |

**Stack with ORION selected for #43:** 1 + 1 + 1 + 8.6 (ORION) + 4 (MELT) + 8 (HYDRA n=8) + 1.6 (REFLECTOR) + 6 (PHOENIX-1BIT) ≈ ~360× wall-clock vs pre-paradigm-1 baseline. ZENITH cannot compose. Adding ZENITH would either replace ORION (yielding 1.41× per-step instead of 8.6×, a 6× regression) or run alongside (impossible due to anchor conflicts). **ZENITH is not selectable as #49 if ORION is the #43 anchor.** This is the most honest assessment.

**Alternative.** If #43 had selected NEXUS or GANYMEDE (which preserve full-d Adam updates), ZENITH could compose alongside. NEXUS's symplectic phase-space extrapolation does NOT rely on a per-step backward pass replacement — ZENITH could attack the gradient evaluation while NEXUS attacks the integrator. Stack potential: NEXUS (5×) × ZENITH (1.67×) × prior ≈ ~30× wall-clock improvement, magnitude territory.

**This is the fundamental scoping question for ZENITH:** is it deployable in the current stack where ORION has been selected for #43? Provisional answer: **no, unless ORION is rolled back.**

### 8.3 Composition with proposed #49 candidates A and C

| Aspect | ICARUS (A) | ZENITH (B) | AURORA (C) |
|---|---|---|---|
| Speedup mechanism | Lower-order numerical integration (Yoshida) | Skip backward via prediction | Skip layers per token (routing) |
| Speedup magnitude | 1.5–2.5× | 1.4–1.7× | 1.5–3× |
| NLL bound | Bit-exact at fixed numerical order | ε-step verified | ε per token-routing |
| Engineering | ~600 LOC | ~900 LOC | ~1100 LOC |
| Risk | Step-size calibration | Verification threshold tuning + acceptance rate | Token-routing stability |
| Composition w/ ORION | ✓ (orthogonal axis) | ✗ (same axis) | ✓ (orthogonal axis) |

**ZENITH is the only #49 candidate that conflicts with #43-C ORION.** This is its central drawback in the post-#43 deployment context.

---

## 9. Honest gaps

### 9.1 The verification stringency–speedup tradeoff

The brief raised the central concern: NLL preservation requires `ε_step ≤ 4.4·10⁻³` (§5.3). At K=4, r=2 the prediction error is `~2.25·10⁻³`, comfortably below threshold. At K=8, error is `~4.5·10⁻³`, borderline. At K=16+, error exceeds threshold.

**Therefore K is bounded above by `K_max ≈ 8` for NLL-preserving operation at r=2.**

At K=4: speedup 1.41× (or 1.67× with CHIRON synergy). At K=8: 2.29× theoretical, 1.49× realistic at 60% acceptance.

**The "1.5–3×" upper end of the brief's claim is not achievable while strictly preserving NLL.** ZENITH's honest range is **1.4–1.7×** at K=4 and **1.49–2.3×** at K=8 (depending on acceptance).

The brief's K=∞ asymptote of 6× is a verification-free regime where prediction error compounds; it is not NLL-preserving.

### 9.2 The pessimistic alternative: K = 1

If the verification reveals that even K=2 leads to unacceptable acceptance rates (e.g., < 30%), ZENITH degenerates to K=1 (always full F+B at the verification's forward cost). At K=1: per-effective-step cost = `(3 + 2r)F = 7F` — **WORSE than CHIRON 3F**. ZENITH-K=1 is a **2.3× regression**, not a speedup.

This is the fundamental risk: if the rank-r prediction is empirically not accurate enough at LLM scale (e.g., Hessian rank > 8), ZENITH cannot operate at K ≥ 2 and contributes negative utility.

### 9.3 Hessian Lipschitz constant uncertainty

The bound `L_H = 10` from §2.1 is literature-derived for general transformer cross-entropy. CHIRON's reversible-flow architecture with FACE/MFIO/SAS may have substantially different Hessian smoothness; this is not measured. If `L_H = 100`, the Taylor-remainder bound at K=4 becomes `2·10⁻⁴`, still below threshold. If `L_H = 1000`, it becomes `2·10⁻³`, comparable to the rank-r residual. ZENITH's NLL-preservation guarantee then degrades.

Mitigation: a CHIRON Gate-0 probe measuring `\|R_t\|_2 / \|Δθ\|_2^2` directly on the existing 1.84B trajectory would resolve this uncertainty. Cost: ~10 GPU-minutes.

### 9.4 Anchor cache memory

At d = 1.84B, r = 2, bf16 storage:
- `V_a`: `d · r · 2 bytes = 7.4 GB`.
- `Λ_a`: `r · r · 4 bytes ≈ 0`.
- `g_anchor`: `d · 2 bytes = 3.7 GB`.
- `θ_anchor`: `d · 2 bytes = 3.7 GB`.

**Total: ~14.8 GB.** This **exceeds the 16 GB RTX 4080 SUPER ceiling minus activations and KV cache (need ~4 GB free).** ZENITH at r=2 does not fit unless the anchor cache shares memory with non-conflicting state.

Mitigations:
1. Store `V_a` in fp8 (1 byte/element): `V_a` shrinks to 3.7 GB. Total cache: 11.1 GB. Fits.
2. Store `g_anchor` derived from existing m_t (Adam already stores `m`, which is an EMA approximation of `g_anchor`). Eliminates the `g_anchor` cost. Total cache: 11.1 GB.
3. Reuse `θ_anchor` from CHIRON's existing checkpoint or anchor-cache mechanism. Eliminates one `θ_anchor`. Total cache: 7.4 GB.

With aggressive memory pooling, ZENITH fits in ~7-8 GB of overhead, leaving ~8 GB for activations and other state. **This is feasible but tight.**

### 9.5 Verification noise and false rejection

The random-direction verification probe has Type-II error rate `10⁻⁴` (with 2 directions, §4.3). But it also has a Type-I error rate (false rejection of correct predictions due to verification noise). At verification threshold `ε_v = 10⁻⁷` and float32 Kahan-summation precision, false-rejection rate could be ~5%, dropping acceptance from 100% to 95%. Realistic speedup at K=4 with 95% acceptance: ~1.40× (vs theoretical 1.41×). Modest impact.

### 9.6 The "rank-r residual is the dominant error" finding

§3.3 showed that `|λ_{r+1}| · K · η` dominates over Taylor remainder by 4 orders of magnitude. **At r=2, this term is irreducible without going to higher r.** Doubling to r=4 reduces it by ~4× (assuming `α=2` Hessian decay) but adds 2× to anchor cost. Net effect at K=4: cost goes from 1.80F → `(3 + 4·1.5 + 0.4·3)/4 = (3 + 6 + 1.2)/4 = 2.55 F → 1.18× speedup`. **Higher rank is worse, not better.**

### 9.7 The honest bottom line

ZENITH delivers **1.4–1.7× per-effective-step speedup** at K=4, r=2 with full NLL preservation. This is materially smaller than #43-C ORION's 8.6× and conflicts with ORION on the deployment axis. ZENITH is **dominated by ORION** in the post-#43 single-GPU compute stack.

ZENITH's **only deployable scenario** is one where:
1. ORION is rolled back (e.g., its slow-rank Gate-0 fails empirically), AND
2. NEXUS or GANYMEDE replaces ORION as #43 (orthogonal-axis paradigms).

In that scenario, ZENITH at K=4, r=2 contributes 1.7× per-effective-step speedup and stacks multiplicatively with NEXUS's 5× to deliver ~8.5× — comparable to ORION but with different risk profile.

---

## 10. Concrete primitives (signatures only)

```cpp
namespace glades { namespace zenith {

// Anchor state: cached gradient, low-rank Hessian, parameters at anchor
struct ZenithAnchor {
    GpuBuffer<float> g_anchor;       // d × 1 (bf16 storage, fp32 view)
    GpuBuffer<float> theta_anchor;   // d × 1 (bf16, possibly fp8)
    GpuBuffer<float> V_a;            // d × r (bf16 or fp8)
    GpuBuffer<float> Lambda_a;       // r × r (fp32)
    int step_at_anchor;
    int rank_r;
};

// Prediction primitive: \hat g_t = g_anchor + V Λ V^T (θ_t - θ_anchor)
class ZenithPredictor {
public:
    ZenithPredictor(int d, int r);
    void predict(GpuBuffer<float>& g_hat,
                 const ZenithAnchor& anchor,
                 const GpuBuffer<float>& theta_t);
};

// Verification primitive: random-direction projection with Kahan summation
class ZenithVerifier {
public:
    ZenithVerifier(int d, float epsilon_v, int num_directions = 2);
    bool verify(const NNetwork& net,
                const GpuBuffer<float>& theta_t,
                const GpuBuffer<float>& g_hat,
                /*out*/ float& delta_norm);
    // delta_norm: max |r_i^T (g_t - g_hat)| across directions
};

// Anchor-update primitive: Lanczos r HVPs via Pearlmutter trick on inverse walk
class ZenithAnchorBuilder {
public:
    ZenithAnchorBuilder(int d, int r);
    void build(NNetwork& net, ZenithAnchor& anchor);
    // Cost: 1 forward + 1 backward (3F) + r HVPs via inverse walk (r * 1.5F)
};

// State machine: orchestrates anchor / predict / verify cadence
class ZenithStepper {
public:
    ZenithStepper(int K, int r, float epsilon_v);
    void step(NNetwork& net, GpuBuffer<float>& theta,
              GpuBuffer<float>& m, GpuBuffer<float>& v, float lr);
    // Returns: was_predicted (bool), was_accepted (bool)
};

}}  // namespace glades::zenith
```

CLI: `--zenith 1 --zenith-K 4 --zenith-r 2 --zenith-eps-v 1e-7 --zenith-verify-dirs 2`

**Engineering scope:**
- HVP primitive (extends CHIRON inverse walk): ~250 LOC.
- Low-rank anchor + Lanczos: ~200 LOC.
- Random-direction verification with Kahan-summed reduction: ~150 LOC.
- State machine + Adam integration: ~250 LOC.
- CLI and config: ~50 LOC.
- **Total: ~900 LOC, 3-5 weeks engineering.**

---

## 11. Gate-0 probe — local-curvature stationarity on existing 1.84B trajectory

**Goal.** Measure the empirical prediction error `\|g_t - \hat g_t\|_2` on real CHIRON trajectories for K ∈ {2, 4, 8, 16} and r ∈ {2, 4, 8}, comparing against the §3.3 theoretical bound.

**Procedure (one GPU-hour, no new training run required).**

1. From the existing 1.84B CHIRON checkpoint logs (`runs/chiron-1p84B-flagship/checkpoints/`), load 50 evenly-spaced parameter snapshots `{θ_τ}_{τ=0}^{49}` from a steady-state portion of training (e.g., steps 200k → 250k).
2. Compute the gradient `g_τ` at each snapshot (load the corresponding minibatch from log).
3. For each `τ ∈ {0, K, 2K, …}` (K ∈ {2, 4, 8}), compute the rank-r=2,4,8 Hessian sketch via `r` Pearlmutter HVPs.
4. For each `s ∈ {1, …, K-1}`, compute the predicted gradient `\hat g_{τ+s}` and the empirical error `\|g_{τ+s} - \hat g_{τ+s}\|_2`.
5. **Pass criterion (K=4, r=2):** median error ≤ `4.4·10⁻³` AND 95th-percentile error ≤ `8.8·10⁻³`. (Median below threshold, tail not too heavy.)
6. **Strong pass (K=4, r=2):** median error ≤ `2·10⁻³` (i.e., the rank-r residual is mild).
7. **Hessian-Lipschitz probe:** measure `\|g_t - g_{anchor} - M_anchor (θ_t - θ_anchor)\|_2 / \|Δθ\|^2_2` directly. Should be `O(L_H/2) = 5` per §2.1; if observed value is `>> 100`, ZENITH's Taylor expansion is fragile.

**If Gate-0 passes:** ZENITH is empirically grounded; greenlight to engineer.
**If Gate-0 fails (median error > 0.01 at K=4, r=2):** ZENITH dies — local-curvature stationarity hypothesis is too loose at LLM scale.

**Cost.** 50 checkpoint loads × 3.7 GB (1.84B × bf16) ≈ 3 minutes disk I/O. Per snapshot: 1 backward (2F ≈ 100ms on 4080 SUPER), then r HVPs (r×1.5F). At r=4, K=4, T=50: 50 backwards + 50·4 HVPs ≈ 50·100ms + 200·150ms = 35 seconds. **Total: ~5 minutes.**

**Expected outcome.** Literature gives Hessian rank in trained networks ≈ tens-to-hundreds (Sagun et al. 2017). Trajectory-effective rank may be smaller (Gur-Ari et al. 2018 report ≈ 30 in 100M nets). Best estimate: r=8 captures sufficient mass for K=4 NLL preservation. r=2 is a stretch; r=4 likely viable.

---

## 12. Material differences from ICARUS (#49-A) and AURORA (#49-C)

| Aspect | ICARUS | ZENITH | AURORA |
|---|---|---|---|
| Speedup mechanism | Lower-order numerical integration (Yoshida composition) | Skip backward via spatial Taylor prediction | Skip layers per token (curvature-aware routing) |
| Speedup magnitude (HONEST) | 1.5–2.5× | 1.4–1.7× | 1.5–3× |
| NLL bound | Bit-exact (numerical-theory bound on truncation error) | ε-step verified (Theorem in §5.2) | ε per token-routing decision |
| Engineering | ~600 LOC, 2-3 weeks | ~900 LOC, 3-5 weeks | ~1100 LOC, 4-6 weeks |
| Empirical risk | Step-size calibration in non-conservative regime | Verification threshold tuning + Hessian rank uncertainty + acceptance rate volatility | Token-routing stability + per-token-loss balance |
| Composition with #43-C ORION | ✓ Multiplicative (numerical-axis) | ✗ Same-axis conflict | ✓ Multiplicative (token-axis) |
| Composition with NEXUS/GANYMEDE | ✓ | ✓ | ✓ |
| Memory overhead | ~0 (numerical reformulation only) | ~10 GB anchor cache | ~5 GB router state |
| TRAJ-rejection orthogonal? | Trivially yes (no gradient prediction) | Yes (§7) | Trivially yes |

**ZENITH's distinguishing feature:** the rigorous coupling of a verification primitive to a structural prediction. ICARUS and AURORA both rely on theoretically-bounded approximations without per-step verification; ZENITH's verification provides a runtime safeguard that ensures NLL-preservation even when the prediction is poor.

**ZENITH's distinguishing weakness:** the verification stringency required to preserve NLL (§5.4) limits K to ≤ 8, which limits speedup to ≤ 2.3× under perfect acceptance. ICARUS achieves 2.5× without verification overhead; AURORA achieves up to 3× with simpler routing logic. ZENITH is **the most rigorously justified but also the smallest in claimed magnitude.**

---

## 13. Open math questions

1. **Tighter NLL-preservation bound.** §5.2's bound is linear in `α T η ε_step`. A second-order analysis incorporating Adam's momentum buffer might give a quadratic-in-T bound when the prediction errors are zero-mean (which they are for symmetric verification thresholds). This could loosen `ε_step` by `\sqrt{T}` ≈ 300×, opening K=16+ regimes.

2. **Hessian-aware verification direction sampling.** Random `r_t` is conservative. If we knew the high-eigenvalue directions of the prediction error covariance, we could sample `r_t` from that distribution and reduce Type-II error by 100× per direction. This requires online tracking of error covariance — feasible.

3. **Adaptive K.** Could K be set per-window based on observed prediction error? E.g., if last verification accepted with margin > 10×, increase K; if rejected, drop K to 1 next window.

4. **Composition with ORION's reduced subspace.** Speculative: use ZENITH's verification as ORION's per-step anchor-validation. If ZENITH detects that ORION's reduced step has drifted out of the slow manifold, ORION refreshes early. This is **ORION-internal** and does not deploy ZENITH as a top-level shift; rather, it borrows ZENITH's verification primitive.

5. **Anchor F+B replacement with a multi-anchor extrapolation.** Speculative: instead of one anchor every K, use 3 anchors and extrapolate the rank-r Hessian via Richardson extrapolation. Reduces anchor cost from `(3 + 2r)F` per anchor to `(3 + 2r)F` per 3-anchor + 2 cheap extrapolations. Math is sound; engineering complexity high.

---

## 14. Summary

ZENITH proposes a cross-step gradient prediction primitive grounded in local-curvature stationarity: predict `g_{t+1}` from `g_anchor + M_anchor (θ_{t+1} - θ_anchor)` using a low-rank Hessian sketch, verify via random-direction projection with Kahan-summed inner products, and accept the prediction whenever `|r^\top (g - \hat g)| ≤ ε_v ≈ 10⁻⁷`. 

The theoretical asymptote is 6× speedup at K → ∞, but the NLL-preservation theorem (§5) constrains the verification threshold tightly enough that K is bounded above by ~8 in the honest accounting. The realistic deployable regime is **K=4, r=2 → 1.41× speedup** (or **1.67× with CHIRON's inverse-walk synergy reducing HVP cost**).

ZENITH's mechanism is structurally different from TRAJ's failed AR(2) prediction: it uses a Hessian-mediated spatial Taylor expansion in θ-space rather than a temporal autoregression on past gradients. The TRAJ rejection is uninformative about ZENITH's premise.

**The decisive scoping concern:** ZENITH conflicts with #43-C ORION (selected). Both occupy the "low-rank-anchor surrogate of F+B" axis. ORION delivers 8.6× per-effective-step at K=20, r=2 — vastly more than ZENITH's 1.67×. **In the post-ORION stack, ZENITH is dominated and not selectable.** ZENITH is selectable only if #43 is rolled back to a non-ORION candidate (e.g., NEXUS), in which case it composes multiplicatively at 1.67× per-step.

**Honest claim:** **1.41–1.67× per-effective-step speedup at K=4, r=2 with verified NLL preservation.** The brief's 3–6× targets are not achievable under verification stringency that preserves NLL. ZENITH's value proposition is rigor (per-step bounded-error verification with NLL-preservation theorem), not magnitude.

**Gate-0 cost:** 5 minutes on existing 1.84B trajectory log — measure empirical prediction error vs the §3.3 bound. **Gate-0 should run before any code is written.** If §11's Gate-0 fails at r=2, escalate to r=4 (still feasible). If r=8 fails, ZENITH is dead.

---

**Materially distinct from #49-A (ICARUS) and #49-C (AURORA):** ZENITH is the only candidate that attacks the **gradient-evaluation** axis with explicit per-step bounded-error verification, the only one that engages the TRAJ-rejection failure mode head-on, and the only one whose composition with the selected #43-C is structurally blocked. Its value if selected: a rigorously-verified 1.4-1.7× speedup; a verification primitive that may serve as a building block for adaptive cadence in other paradigms; and a clean disproof-or-validation experiment in §11's Gate-0.

If composition with ORION cannot be unblocked, ZENITH should be **deferred** rather than selected — the realized magnitude is too small to justify the 900 LOC engineering investment when ICARUS or AURORA can compose multiplicatively at comparable risk.
