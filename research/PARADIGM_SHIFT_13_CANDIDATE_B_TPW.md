# Paradigm shift #13 — Candidate B: Trajectory-Predictive Weights (TPW)

**Formulation class:** operator-theoretic / continuous-time dynamics over
the *temporal* axis of training.
**Author pass:** subagent-dispatched design (2026-04-22).
**Status:** candidate — design phase.

---

## 1. Short name

**TPW — Trajectory-Predictive Weights.**  Working codename:
*Padé-Flow Optimizer*.  Model the weight trajectory θ(t) as the solution
of a learned operator-valued ODE θ̇ = −P_θ(t)·g(θ_k), fit the Padé
operator P_θ from a handful of anchor-point gradient evaluations, and
evaluate θ_{k+K} from the closed-form flow — jumping K optimizer steps
with ~3 full forward+backward passes instead of K.

TPW is the first paradigm shift in the #1–#12 stack that attacks the
*cross-step* axis of redundancy: every preceding shift still pays the
price of one full forward+backward per optimizer step.  TPW **predicts**
optimizer updates rather than compressing them.

---

## 2. Primitive objects

Let N = Σ_ℓ m_ℓ n_ℓ ≈ 2.23·10⁹ be total parameter count and let
ℒ : ℝ^N → ℝ be the cross-entropy loss.

| symbol                   | shape                  | meaning                                                      |
|--------------------------|------------------------|--------------------------------------------------------------|
| θ(t) ∈ ℝ^N               | curve in ℝ^N           | weight trajectory parameterized by continuous training time |
| g(θ, t)                  | vector field           | instantaneous gradient g = ∇_θ ℒ_t(θ)                        |
| θ_k = θ(t_k)             | anchor                 | weights at step k                                            |
| g_k                      | vector                 | anchor gradient at step k (BF16 from backward)               |
| P_θ(t)                   | rank-r linear operator | "trajectory operator": maps anchor gradient to flow direction |
| (L_ℓ, R_ℓ, Σ_ℓ)          | m_ℓ×r, n_ℓ×r, r×r      | factored per-layer parameterization of P_ℓ                   |
| r                        | scalar                 | TPW bond rank; production default r = 32                     |
| K                        | integer                | step horizon jumped per fit; target K = 4–8                  |
| Δ ∈ (0, 1)               | fractional step        | probe offset for third anchor; Δ = 0.5                       |
| ρ_fit, ρ_Lyap            | scalars                | Padé fit residual and Lyapunov dissipation check             |

**Per-layer factorization.**  A global P_θ on ℝ^N is infeasible
(2·N·r ≈ 286 GB at r = 32).  We factor P_θ **per trainable matrix**
W_ℓ ∈ ℝ^{m_ℓ×n_ℓ}:
    P_ℓ(t) · g_ℓ = g_ℓ + L_ℓ · Σ_ℓ(t) · R_ℓ^⊤ g_ℓ,
with L_ℓ ∈ ℝ^{m_ℓ × r}, R_ℓ ∈ ℝ^{n_ℓ × r}, Σ_ℓ(t) ∈ ℝ^{r×r}.  This
is exactly the rank-r geometry OVFG (#9) already carries — TPW reuses
OVFG's (L_ℓ, R_ℓ) pair verbatim.  Storage adds only r² FP32 per layer
(r = 32: 4 KB/layer; across L = 48: ~200 KB total).

---

## 3. Mathematical state space

TPW lives on two coupled objects:

1. **Weight space** M = ∏_ℓ (Stiefel × Σ × V^⊤) (inherited from shift #7) × {LN, bias}.  The trajectory θ(·) is a smooth curve γ : [t_k, t_k + K·η] → M.
2. **Operator space.**  Per layer, P_ℓ ∈ GL(ℝ^{m_ℓ n_ℓ}) is a rank-r perturbation of identity.  Over the window we fit a Padé rational Σ_ℓ(t) of degree (p, q).

Tangent decomposition: T_{θ_k} M = T^{fit} ⊕ T^{skip}, where T^{fit} is
the span of L_k across the window.  TPW trusts Padé on T^{fit}; on
T^{skip} it relies on the verification step (§5.3).

---

## 4. Evolution law (derivation from first principles)

### 4.1  Gradient flow in continuous time

Adam's small-η continuous limit is

    θ̇(t) = −H_t(θ) g(θ, t)                                    (1)

with H_t the (bias-corrected) Adam preconditioner.  Integrating from
t_k to t_k + K·η gives the exact K-step update

    θ_{k+K} = θ_k − ∫_{t_k}^{t_k+Kη} H_s(θ(s)) g(θ(s), s) ds.   (2)

Eq. 2 is exact but useless — it needs θ(s) on the interval.

### 4.2  Linearization of g along the trajectory

Taylor-expand g along the flow.  Using θ(s) − θ_k = −(s−t_k)·H_k g_k + O(η²) and ∂_θ g = ∇²ℒ:

    g(θ(s), s) ≈ exp(J_k · (s − t_k)) · g_k                     (3)

where J_k := −∇²ℒ(θ_k) H_k + ∂_s H^{-1}ġ|_k is the **effective
Jacobian of the gradient field along the flow**.  This is the
structural insight — g(θ(s), s) satisfies (to linear order) a *linear
ODE* driven by g_k alone.  Substituting into Eq. 2:

    θ_{k+K} − θ_k ≈ −H_k · φ(Kη J_k) · g_k                      (4)

with φ(z) = (e^z − 1)/z the standard ETD function.  Eq. 4 is the
**exponential integrator step** — the K-step update depends on g_k and
on a matrix function of J_k, not on any intermediate gradient.

### 4.3  Padé rational restriction to an r-dim subspace

Direct evaluation of exp(KηJ_k) requires an N×N operator.  Padé diagonal
(p, p) approximates exp(z) with local error O(z^{2p+1}); for p = 1 the
approximant is Cayley (1+z/2)/(1−z/2).  But the Padé denominator is
still N×N — infeasible.

Restrict J_k to its dominant r-dim invariant subspace:
    J_k ≈ L_k J̃_k R_k^⊤,   L_k, R_k ∈ ℝ^{N×r},  J̃_k ∈ ℝ^{r×r}.

The Padé approximant of exp(Kη L J̃ R^⊤) then factors as
I + L·f(J̃)·R^⊤ with f a matrix-rational on r×r space.  The **TPW
operator** is this rank-r perturbation:
    P_θ(t) = I + L_k · Σ(t) · R_k^⊤,
    Σ(t) = ∫_0^{t/Kη} (1−s) · f(s·Kη·J̃_k) ds.

Σ(t) has a closed r×r rational form.  Padé (1,1) yields
    Ψ_{1,1}(z) = (1/2 + z/12) / (1 − z/2 + z²/12);
Padé (2,2) (production default) yields
    Ψ_{2,2}(z) = (1/2 + z/10 + z²/120) / (1 − z/2 + 3z²/28 − z³/840).

### 4.4  Closed-form K-step update

    h_j = R_k^⊤ g(θ_anchor_j, t_anchor_j),   j ∈ {0, Δ, 1},        (anchors)
    J̃_k = argmin_{J∈ℝ^{r×r}} Σ_{j∈{Δ,1}} ‖h_j − e^{t_j J} h_0‖²,   (fit)
    Σ_k = (Kη) · Ψ_{p,q}(Kη · J̃_k),                                (r×r rational)
    Δθ_k = −L_k · Σ_k · h_0,                                        (closed form)
    θ_{k+K} = θ_k + Δθ_k.                                           (5)

Per layer, Eq. 5 costs two rank-r GEMMs plus a handful of r×r matrix
inversions.  For r = 32, r³ = 32 k — microseconds.

### 4.5  Fitting J̃_k from three anchors

Three anchors, at (t_k, t_k + Δη, t_k + η), provide three h_j vectors
in ℝ^r.  Only J̃ ∈ ℝ^{r×r} is fit (r² = 1024 unknowns, (m+n)·r scalar
equations after projection — hugely overdetermined).  (L_k, R_k) are
**inherited from OVFG's factored-gradient basis** — they are not
degrees of freedom of the Padé fit, only of the gradient accumulator.

Closed form for Padé (1,1): J̃ = (2/η)·(h_1 − h_0)(h_1 + h_0)^{-1}.
For (2,2), solve a linear r×r matrix equation via cuSOLVER — still
O(r³).

**Anchor budget.**  Anchors (1) and (3) are the gradients Adam would
compute anyway on the first and normal-step of the window — **free**.
Only anchor (2) is TPW-specific: one probe forward+backward at
θ_tiny := θ_k − Δ·η·H_k g_k.  Net TPW overhead per window: **1 extra
forward+backward for the probe**.

---

## 5. Mechanism mapping

### 5.1  Memory — O(K) amortization with no persistent per-step state

Per-layer overhead: r² FP32 (≈4 KB at r=32).  Three anchor gradients
are **streamed** layer-by-layer into R^⊤ g, yielding an r-vector h_j
— never persisted in full.  No intermediate θ_{k+1}, …, θ_{k+K-1} is
ever materialized.  Total TPW memory across L = 48: ~200 KB.

### 5.2  Speed — (K−3) full forward+backward skipped per K steps

Per K-step window:

| step                             | f+b cost    |
|----------------------------------|:-----------:|
| Anchor (1) g_0                   | 1.0 (free)  |
| Probe θ_tiny + anchor (2) g_Δ    | 1.0 (extra) |
| Anchor (3) g_1 at normal step    | 1.0 (free)  |
| 3-point fit J̃_k + eval Ψ        | <0.1        |
| **Total per window**             | **~3.1**    |
| **Baseline (K full steps)**      | **K**       |

Savings = K − 3.1.  K = 4: 1.23× speedup.  K = 6: 1.94×.  K = 8: 2.58×.

Production: start at K = 4 (safe, modest win), anneal up as ρ_fit
residuals shrink with training stability.

### 5.3  Stability — Padé residual + Lyapunov check

Two safeguards:

1. **ρ_fit.**  After fitting J̃_k:
       ρ_fit = ‖h_Δ − e^{Δη J̃_k} h_0‖ / ‖h_Δ‖.
   If ρ_fit > τ_fit (default 0.1), abort TPW for this window — take K
   plain Adam steps instead.

2. **ρ_Lyap** (gradient-flow Lyapunov check).  Compute g_end =
   g(θ_{k+K}, t_k + Kη) (one verification f+b), require
       ρ_Lyap = ⟨g_end, Δθ_k/(Kη)⟩ ≤ 0.
   ρ_Lyap > 0 means Padé flow is **non-dissipative** over the window —
   reject the TPW jump, restore θ_k, do K Adam steps.

Verification cost eats into speedup.  Effective cost = 3.1 + 1.0 = 4.1
f+b per window; break-even at K = 5.  Mitigate by **amortizing
verification**: run it every 2nd window once three consecutive ρ_fit
values stay below τ_fit.

---

## 6. Objective

Training objective unchanged — cross-entropy over next-token prediction:
    ℒ(θ) = 𝔼_{(x,y) ∼ D_train} CE(f_θ(x), y).

TPW's **internal fitting objective** is the r-dim least squares
    J̃_k* = argmin_{J ∈ ℝ^{r×r}}  Σ_j ‖h_j − e^{t_j J} h_0‖²,
which is the L² projection of the linear-regime gradient-flow
approximation onto the OVFG subspace.  This is a *numerical* fit, not
a new model loss.

---

## 7. Expected stability (theoretical)

**Theorem (informal).**  Suppose ℒ is M-smooth with Hessian-Lipschitz
constant L_H.  In a basin where J_k stays within ε_J of J(t_k) for all
t ∈ [t_k, t_k + Kη], the Padé (p, q) approximant with p = q = r yields

    ‖θ_{k+K}^{TPW} − θ_{k+K}^{exact}‖ ≤
        C · (Kη)^{2r+1} · ‖J_k‖^{2r+1} + C' · L_H · (Kη)² · ε_J.

For r = 1 (Padé (1,1)): **O(K³) local truncation**.  For r = 2 (Padé
(2,2)): **O(K⁵)**.  Far below Adam's single-step O(Kη) error when the
basin hypothesis holds.

Proof sketch: standard Padé bounds (Higham, *Functions of Matrices*,
Ch. 10) specialized to the r-dim invariant subspace, plus first-order
perturbation of J_k.  Details deferred to implementation.

**Caveat.**  Bound holds only inside a single basin.  Phase transitions
(loss sudden-drops, grokking, attention-sink formation) violate ε_J
bounded; Lyapunov check is the practical safeguard.

---

## 8. Failure modes

1. **Strongly nonlinear regions.**  Linearization in Eq. 3 fails when
   the trajectory curves sharply (early training, phase transitions).
   *Mitigation:* disable TPW during warmup (< 500 steps) and after any
   Lyapunov violation; resume after 3 consecutive low-ρ_fit windows.

2. **Effective-LR collapse as K grows.**  Padé bounds the step but not
   its effectiveness — TPW may "land" in a worse loss region.
   *Mitigation:* trust-region clip ‖Δθ_k‖ ≤ Kη · ‖g_0‖.

3. **Fit cost eats savings.**  Anchor (2) is an irreducible 1.0 f+b;
   if K ≤ 4 is the stable ceiling, net speedup vanishes.  *Mitigation:*
   adaptive K — if K = 4 is the max, shut TPW off for that phase.

4. **Catastrophic drift under stale refit.**  Without refit each
   window, integration error compounds geometrically (K·ε)^n.
   *Mitigation:* mandatory J̃ refit every window; CI test enforces
   stale-refit divergence within 10 windows (sanity).

5. **Non-dissipative Padé.**  If fitted J̃ has positive real eigenvalues,
   the flow amplifies the gradient.  *Detection:* eigendecomp of J̃
   (r³, microseconds); reject if max Re(eig(J̃)) > 0.

6. **Anchor (2) probe destabilizes training.**  If Δ·η is too large,
   g_Δ is noisy and J̃ is mis-fit.  *Mitigation:* Δ = 0.5 default;
   adapt Δ based on observed ρ_fit.

7. **Stiefel retraction (shift #7).**  Padé flow does not preserve
   Stiefel.  *Mitigation:* one QR retraction after Eq. 5 (cost:
   O(m·r²) per layer — negligible).

8. **DFA (#12) incompatibility.**  DFA gradients are biased; fitting
   J̃ to biased anchors amplifies bias through Eq. 5.  *Mitigation:*
   use true backprop gradients for TPW anchors; DFA is disabled for
   the 3 anchor steps per window.

---

## 9. Composability with shifts #1–#12

- **#1 CHIRON:** orthogonal.
- **#2 TC attn, #6 local-window:** orthogonal.
- **#3 int8 Adam, #4 BF16 grads, #5 SR BF16 weights:** compose —
  anchors are the gradients Adam produces; Padé eval in FP32.
- **#7 Stiefel:** Riemannian variant required — project Δθ onto
  tangent, retract via QR.
- **#9 OVFG:** **deeply synergistic** — OVFG's (L_ℓ, R_ℓ) pair is
  *exactly* the rank-r basis TPW needs.  Marginal TPW cost on top of
  OVFG: r² FP32 per layer.
- **#10 MPOT:** orthogonal (TPW operates on gradients; MPOT on weights).
- **#11 MFIO:** compose — H_k replaced by MFIO's σ_ℓ in Eq. 4.
- **#12 DFA:** see failure mode 8.  A "DFA-compatible TPW" that fits
  J̃ to random-projected gradients is an open direction.

---

## 10. Comparison vs contemporaneous optimizers

- **Shampoo / K-FAC:** per-step natural-gradient, no cross-step
  prediction.  Orthogonal — could replace H_k in Eq. 3.
- **Sophia:** Hutchinson-Hessian per-step.  Same remark.
- **Anderson-Adam:** extrapolates the *update vector* (linear
  fixed-point acceleration).  TPW is strictly more general: Padé
  rational of any degree, data-fit from *gradients not updates*,
  closed-form *trajectory* rather than discrete extrapolation.
- **Full-rank exponential integrators (Li–Tai 2019):** require dense
  exp(J) — not feasible at N = 2 B+.  TPW is the rank-r restriction
  that makes exponential integrators tractable at LLM scale.

TPW is, to the author's knowledge at 2026-04-22, the **first optimizer
that predicts the optimizer update K steps ahead** in a closed-form,
operator-theoretic way, as opposed to per-step preconditioning or
update-level extrapolation.

---

## 11. Memory accounting at 2.23 B on 16 GB

| component                          | w/o TPW | w/ TPW       |
|------------------------------------|--------:|-------------:|
| Weights (Stiefel × MPOT)           | 1.0 GB  | 1.0 GB       |
| Gradients (OVFG factored)          | 0.5 GB  | 0.5 GB       |
| Optimizer state (MFIO)             | ~0      | ~0           |
| Activations (CHIRON)               | 0.05 GB | 0.05 GB      |
| Scratch + logits                   | 1.5 GB  | 1.5 GB       |
| TPW J̃_ℓ + fit residuals           | —       | 0.0002 GB    |
| **Total**                          | **3.1 GB** | **3.1 GB**|

**TPW is memory-free on top of #1–#12** — pure compute win.

Projected end-to-end throughput at production K = 6 with amortized
verification: **~1.7× at 2.23 B**; at 30 B (unlocked by other shifts)
projected **~2.0×** because the per-step forward+backward dominates
more strongly.

---

## 12. Minimal prototype (≤ 3 weeks)

**Phase 1 — TPW primitives (week 1)**

- New `Backend/Machine Learning/Networks/cuda/gpu_tpw.{h,cu}`:
  - `tpw_project_gradient(L, R, g_bf16, h_fp32)` — r-GEMM R^⊤ g.
  - `tpw_fit_jtilde(h_0, h_Δ, h_1, Δ, η, jtilde_out)` — three-point
    matrix-log fit via cuSOLVER.
  - `tpw_eval_psi(jtilde, Kη, psi_out)` — r×r matrix rational.
  - `tpw_apply_closed_form(L, psi, h_0, Kη, dtheta_out)` — two r-GEMMs.
  - `tpw_lyapunov_check(g_end, dtheta, rho_out)` — scalar reduction.
- Unit tests: K=1 TPW parity with Adam; synthetic-quadratic Padé
  truncation bound; stale-refit divergence (sanity).

**Phase 2 — OVFG integration (week 2)**

- Reuse OVFG's (L_ℓ, R_ℓ).  Padé (1,1) first, then (2,2).
- 3-anchor fit on 80M-param model; verify ρ_fit < 0.05 in stable basin.

**Phase 3 — trainer wire-in (week 3)**

- `chiron_main.cpp --tpw --tpw-K 4 --tpw-rank 32 --tpw-pade 1,1` flags.
- Lyapunov check + adaptive-K logic.
- 500-step pile_large-small smoke test (d = 1024, L = 24); compare
  wall-clock and loss trajectory vs Adam.

**Tracked as task #37** (to be created on implementation start).

---

## 13. Research risks and selection

TPW's central risk is that **cross-step predictability of LLM dynamics
at scale is empirically unknown**.  The O(K^{2r+1}) bound holds in
convex basins, but LLM training has documented non-convex phase
transitions where Padé extrapolation is expected to fail — the Lyapunov
check triggers, TPW falls back to Adam, and the speedup disappears.

The **open empirical question** is: what fraction of training time is
spent in "TPW-friendly" basins?  Plausible answers span 30% (only
late-training refinement benefits) to 90% (most of pre-training is
slow basin-descent).  Shift #13's selection between Candidates A, B, C
should weight each on **expected realized speedup after fallback**,
not best-case theoretical speedup.

If TPW is selected, the first validation milestone must be **measuring
ρ_fit and Lyapunov-trigger rate on a 500-step pile_large-small run** —
one empirical datum worth more than further derivation.

---

*End of Candidate B — Trajectory-Predictive Weights (TPW).*
