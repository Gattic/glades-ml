# CHIRON-Native Optimizer Design Study — ORBIT selected, RELIC/KRONOS portfolio

**Date:** 2026-07-09
**Status:** DESIGN ONLY — nothing built, nothing run (GPU occupied). No code in this arc.
**Brief:** design several CHIRON-native optimizers to add to or replace AdamW for the
CHIRON 1B flagship (PIED ship, `chiron_1B_pied_e4.final` lineage).
**Method:** 3-candidate research-framework-design study (same discipline as the
PIED / SIPHON / ECHO studies): three materially different formulations developed
independently, compared, one selected and fully developed; the others held with
explicit revive triggers.

---

## 0. Executive summary

- Baseline to beat: **int8-state AdamW** (β1=0.9, β2=0.95, eps outside sqrt,
  decoupled wd=0.01, bias correction, block-256 absmax int8 m / uint8 v,
  ≈1.77 GB state), lr 3e-4 constant + 750 warmup, global L2 grad-clip τ=0.5.
- Three candidates were developed:
  - **KRONOS** (information-geometric): whitened-frame Adam with forward-derived
    diagonal Kronecker factors; ~1.5 MB new state; full-Fisher exponent on top of
    Adam's sqrt-Fisher.
  - **RELIC** (dissipative conformal-symplectic flow): relativistic bounded-velocity
    momentum; subsumes the global clip into a hard speed ceiling; deletes per-element
    v (−50% state).
  - **ORBIT** (operator-theoretic / optimal control): function-space trust region in
    **increment space** — the metric is the pullback of the readout Fisher through
    CHIRON's linear p-accumulator; factored second moments replace per-element v
    (−50% state); the global clip is kept verbatim.
- **Selected: ORBIT.** It exploits the strongest CHIRON-specific structural fact
  (Proposition 1 below: all attention-parameter curvature factors through a single
  T×m interface, giving a depth-shared output Fisher factor for the entire Wo
  family — independently discovered by two of the three candidates); it keeps the
  empirically load-bearing τ=0.5 clip untouched; and its worst-case outcome
  (factored-Adam tie) still banks a pre-declared structural win: **optimizer state
  1.77 GB → ≈0.89 GB (−50%)**, freeing ~0.9 GB VRAM for the deferred MTP /
  reversible-FFN arcs.
- **RELIC is HELD** (revive trigger: owner wants clip-machinery simplification or a
  schedule-free finish-anneal analogue; or as the long-term composition
  "RELIC kinetics on the ORBIT metric"). **KRONOS-W is HELD** (revive trigger:
  E2 shows the factored-v replacement is the blocker but the Wo/E factors carry
  signal — then apply them as multiplicative whitening on top of stock per-element
  Adam). KRONOS-F (factored v) is subsumed by ORBIT.
- Ship bar (pre-registered): beat baseline by **≥0.05 nat wide-32 val at matched
  30k/1.97B**, at ≤1% wall — or tie within ±0.02 nat with the −50% state win banked.
- Everything default-off, E0 bit-parity, C++98/CUDA, no external libs.

---

## 1. Problem statement and constraints

### 1.1 Target system (ground truth, verified in-tree 2026-07-09)

- Trainer: `glades-trainer/trainer/chiron_main.cpp`; kernels in
  `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_kernels.cu` / `gpu_chiron.cu`.
  Trainer links glades CUDA kernels **statically** (rebuild trainer after lib changes).
- Model: m=2048 stream width, dModel=4096=2m, L=24 reversible symplectic blocks,
  nH=16, dH=256, V=32000 tied embedding/readout, T=16384. 870.94M params:
  - `E` (V×m, FP32, tied readout: logits = q̃_L·Eᵀ) — 65.54M.
  - Per layer: `Wq/Wk/Wv` (m×dModel each) and `Wo` (dModel×m) — BF16 persistent,
    **no FP32 master**, updated via FP32 scratch + stochastic-rounding (SR)
    writeback (`cast_f32_to_bf16_stochastic`, counter-hash `sr_hash32`).
    603.98M + 201.33M.
  - `gamma/beta` (ReLN affine, m each, FP32) — 98k. Ship recipe also trains QK-Norm
    per-head γ (384) and WhiSC `rot_phi` (24×2048), plain FP32 Adam.
  - `scfa_B` DCT-II basis is fixed (not trainable). (`scfa_D` has a grad buffer but
    no wired update — de-facto frozen; out of scope here, flagged separately.)
- Forward per layer: `(p,q) → (p + σ_l·D_l(q), reln(q))` where `D_l` is the SCFA
  attention increment (PIED-masked during training, mean-one mask on the branch
  only); the p-accumulator is **linear** and never re-normalized; p folds into the
  readout only at the final layer; reln re-normalizes q every layer (per-token
  unit-RMS + affine). Ship recipe adds the WhiSC per-channel coupling
  `Φ_l = W⁻¹R(θ)W`, |θ| ≤ θ_max = 0.07, which mixes the (q,p) trunk by a bounded
  per-channel rotation each layer. Trained-in asymmetry ρ = E[p²]/E[q²] reaches
  ~10³ within ~600 steps (measured, `chiron_init_pq_ratio_measured`).
- Optimizer baseline: `adam_update_int8_state_kernel` (AdamW math: decoupled wd
  applied before the moment step, bias correction, eps outside sqrt), dispatched
  per-tensor by `adam_one` inside the `adam_step` driver. Pipeline order per step:
  backward accumulates (attention grads in **BF16** via `bf16_accum_axpy`;
  dE/dgamma/dbeta FP32; accum=4 mean-normalized), per-layer dq row-RMS clamps
  inside backward, optional per-group clamps, **global grad-norm + clip τ=0.5**,
  NaN/Inf grad-skip guard, LR schedule, `adam_one` per tensor. The trainer already
  supports heterogeneous per-tensor-class branches (FACE, MFIO, ASTRA, Sophia-G —
  all default-off).

### 1.2 Empirical constraints (established, binding on the design)

1. **τ=0.5 global clip is load-bearing and near-optimal** (2026-07-06 sweep:
   loosening monotonically worse, tightening flat). Keep it, or subsume it with
   something provably no looser.
2. **Instability is a non-event** in the current recipe (reln-reanchor; 0 grad-skips
   over 30k). Do not design around instability/energy recapture (SIPHON/CHARGE
   arc CLOSED).
3. **Ported mechanisms fail here** (MTP/UL2/LayerDrop). Sophia-G and Adafactor-style
   row/col preconditioners already exist in-tree; rename-ports are worthless.
4. **Early gains wash out** (PACT: −0.3..−0.5 nat at 2500 → +0.026 nat WORSE at
   30k). Design for the asymptote; E3 is a kill gate only, never bankable evidence.
5. Era-drift between binaries ~0.10–0.15 nat ⇒ matched paired baselines mandatory.
6. Determinism: all stochasticity via stateless counter-hash; no RNG state.
7. Budgets: optimizer state ≤ int8-Adam parity (~1.8 GB), wall overhead ≤ ~1%
   (~25.1k tok/s baseline), no per-step dense GEMMs at V×m or m×dModel scale.

### 1.3 Objective

At the fixed 30k-step/1.97B-token budget, matched recipe and seed:
**wide-32 val NLL ≤ baseline − 0.05 nat** at ≤1% wall and ≤ parity memory — OR
NLL tie within ±0.02 nat **with a structural win** (≥25% optimizer-state reduction,
or subsuming the clip machinery into one principled rule) — via CHIRON-specific
structure unavailable to generic optimizers.

---

## 2. Candidate formulations (condensed; full candidate docs in the study record)

### 2.1 KRONOS — Kronecker-diagonal Reversible-Origin Natural-metric Optimizer Stats
*(information-geometric / manifold-constrained)*

Whitened-frame int8-AdamW: each tensor's gradient is divided by
`(r̂_i ĉ_j)^{1/2}` (diagonal Kronecker factors estimated from **activations**, FP32,
immune to the BF16 grad noise floor), Adam runs unchanged on the whitened gradient,
and the step is un-whitened on the way out. Because Adam's m̂/√v̂ is invariant to
static per-element rescaling, the net effect is Adam's step multiplied by
`(r̂_i ĉ_j)^{-1/2}` — i.e. Adam's implicit sqrt-Fisher scaling is **squared to a
full-Fisher scaling** where the factors are trustworthy. Factors: q-side input
factor justified diagonal by reln (the token-norm mode — the dominant spectral
spread — is removed exactly); Wo output factor depth-shared (see Proposition 1);
tied-E rows get multinomial-Fisher frequency scaling
`min(κ, (n̄_T/(n̄_v+δ))^{1/2})` subsuming FACE's hard gate. New state ≈1.5 MB;
wall +0.3–0.5%; clip untouched. Optional KRONOS-F deletes uint8 v for QKV/Wo
(−805 MB). Main risks: "Adam-already-does-this" null; over-whitening starves
ρ-hot columns (steps shrink ~ρ^{1/2}≈30×, possibly under the BF16-SR noise scale).

### 2.2 RELIC — RElativistic LImited-Celerity integrator
*(dynamical-systems / dissipative conformal-symplectic flow)*

Optimizer phase space (Θ,P) with Hamiltonian H = f(Θ) + K(P),
K(P) = Σᵢ c(√(Pᵢ²+sᵢ²) − sᵢ) — a relativistic kinetic energy whose velocity map
`φ(P) = c·P/√(P²+s²)` is hard-bounded by c = lr per coordinate; a Sundman
time-rescaling stage caps the global step norm at a calibrated D_max. Splitting
integrator friction ∘ kick ∘ rescaled-drift, each sub-step exactly
conformal-symplectic. Saturation scale s = block-256 RMS-gradient EMA (≈7 MB)
replaces per-element v; unsaturated coordinates recover Adam-like 1/RMS steps,
coherent ones move at the ceiling — "effective update norm ≈ const" becomes a
property of the dynamics, and the global clip is deleted on the RELIC path
(provably no looser: |Δθᵢ| ≤ lr always, spike energy enters P through a (1−α)=0.1
gate). State ≈0.89 GB (−50%). Friction α decoupled from the ceiling gives a
plateau-variance knob (α 0.9→0.98 late ≈ a finish anneal without an LR schedule).
Limiting cases: heavy-ball (s→∞), sign-momentum/Lion (s→0), clipped-Adam-like
(s=ν). Main risks: the per-coordinate (L∞-like) ceiling is a *different geometry*
from the validated global-L2 clip — the τ=0.5 sweep evidence does not transfer
cleanly; saturation staleness (sign-regime discards magnitude info → misses
late-training descent, the OBSD signature); row-pooled ν under-resolving
embedding curvature.

### 2.3 ORBIT — Optimal-control Readout-metric Backward-collected Increment Trust-region
*(operator-theoretic / optimal control — SELECTED)*

Mirror descent whose geometry is a **function-space trust region in increment
space**: layer index = time, (q,p) = state, weights = controls; the metric is the
pullback of the readout Gauss-Newton factor through the **linear** p-accumulator.
The constant-in-depth p-adjoint gives one shared readout row-metric for the whole
Wo family; factored second moments (adjoint-side × activation-side for Wo,
grad row/col moments for QKV, Fisher-occupancy + frequency rows for tied E)
replace per-element v on all matrix tensors (−50% state). Global clip kept
verbatim; an optional global function-space multiplier is a second, tighter cap.
Full development in §4–§12.

---

## 3. Selection rationale

| Criterion | KRONOS | RELIC | ORBIT |
|---|---|---|---|
| CHIRON-native core | Prop. 1 + reln isotropy | philosophy-level (symplectic integrator), mechanism generic | Prop. 1 + control view + increment lineage (PIED/PACT) |
| Keeps validated τ=0.5 geometry | yes | **no — replaces it** | yes (verbatim, + optional tighter cap) |
| Outcome if NLL ties | nothing banked (W-mode) | −50% state, but high regression risk | **−50% state banked** |
| Dominant risk | over-whitening starves steps; "Adam already does this" | geometry swap unanchored from sweep evidence; sign-regime asymptote | Adafactor-in-disguise null (still banks state win) |
| Wall | +0.3–0.5% | ≤+0.5% | ≤+0.4% |
| Implementation surface | small | medium (clip removal + new step kernels) | medium (5 kernels, all reductions/elementwise) |

**Why ORBIT wins.** (i) It is built on the single strongest architecture-specific
fact available (Proposition 1), which two candidates found independently — the
highest-confidence signal in the study. (ii) It is the only candidate whose
failure mode is *contained by construction*: if the CHIRON-specific factors carry
no signal, it degenerates to a scale-pinned factored-moment Adam at −50% state —
a pre-declared structural win, directly fundable into the MTP (needs ~2.6 GB
scratch) and reversible-FFN arcs. (iii) It respects constraint 1 exactly rather
than arguing around it. (iv) It converts the house's increment-geometry lineage
(PIED shipped; PACT failed as a *loss term*) into the metric, with a precise
reason the metric role differs: a loss-side penalty relocates stationary points
(PACT's failure), a metric only reparameterizes the descent path — the optimum
set is invariant.

**Why the others are held, not killed.** RELIC's clip-subsumption and
schedule-free anneal are genuinely attractive but gamble the one geometry the
sweep validated; it becomes compelling *after* ORBIT establishes whether metric
shape matters (and composes with it — §14). KRONOS-W is the cheap fallback if
E2 shows the factored-v replacement (not the factor content) is the blocker:
its 1.5 MB bolt-on form can ride on stock Adam.

---

## 4. ORBIT — formal setup

### 4.1 Notation

- Layers l = 1..L (L=24). Per-token states q_l, p_l ∈ ℝ^m; batch-time index t
  ranges over the T×(batch) token stream.
- Dynamics (training forward):
  - p_l = Φ^p_l( p_{l−1} + σ_l · D_l(q_{l−1}; θ^a_l) , q_{l−1} )
  - q_l = Φ^q_l( reln_{γ_l,β_l}(q_{l−1}) , p_{l−1} )
  where D_l ∈ ℝ^{T×m} is the SCFA increment (PIED mask η applied on the branch,
  E[η]=1), σ_l ∈ {±1}, θ^a_l = (Wq,Wk,Wv,Wo)[l], and Φ_l is the WhiSC per-channel
  rotation with angle |θ_ch| ≤ θ_max = 0.07 (Φ = identity if `--whisc-coupling`
  off).
- Readout: logits Z = q̃_L Eᵀ after the final fold of p into q
  (fuse-attn-reln); loss ℓ = mean CE + z-loss.
- Adjoints: λ_{p,l} = ∂ℓ/∂p_l, λ_{q,l} = ∂ℓ/∂q_l ∈ ℝ^{T×m} — these ARE the
  existing backward's dp/dq buffers at layer boundaries (the inverse walk
  recomputes activations there; no new passes, no new memory).
- G_τ = accumulated gradient of tensor τ (post-accum, post-clip — the exact tensor
  `adam_one` currently receives).

### 4.2 The control view

Treat l as time, x_l = (q_l, p_l) as state, θ_l as controls, ℓ as terminal cost.
The Pontryagin adjoint recursion for the p-component is

  λ_{p,l−1} = (∂p_l/∂p_{l−1})ᵀ λ_{p,l} + (∂q_l/∂p_{l−1})ᵀ λ_{q,l}.

**Proposition 1 (increment-interface factorization).** With WhiSC off (Φ = id):
∂p_l/∂p_{l−1} = I exactly and ∂q_l/∂p_{l−1} = 0, hence

  λ_{p,l} = λ_{p,L} =: g_p  for all l  (the p-adjoint is constant in depth),

and, because q_l depends only on (q_{l−1}; γ,β) and attention writes only into p,
**every attention parameter influences the loss exclusively through the single
T×m interface p_L = p_0 + Σ_l σ_l D_l.** Consequently the Gauss-Newton metric of
all attention parameters jointly is the pullback of one readout factor:

  GN(dθ^a) = ⟨ Σ_l σ_l dD_l , M ( Σ_l σ_l dD_l ) ⟩,
  M := J_pᵀ H_Z J_p,  J_p = ∂Z/∂p_L,  H_Z = diag(π) − ππᵀ (softmax GN).

*Proof sketch:* linearity of the accumulator plus the absence of any θ^a-dependent
path into q. ∎

**WhiSC correction.** With the ship recipe's coupling on, ∂p_l/∂p_{l−1} is a
per-channel rotation with |θ| ≤ 0.07 (observed max 0.067), so λ_{p,l} is constant
only up to a composition of bounded rotations, and attention leaks into the q-path
at O(θ) per layer. Design consequence (not a hand-wave): the row-metric estimator
below samples dp statistics at **several layer boundaries**, not only the readout,
so the estimated factor absorbs the actual trunk coloring whatever θ does.
PIED does *not* break Proposition 1 (the mask multiplies the branch D_l, never the
trunk; its E[η²] = 1/(1−π) inflation of second moments is uniform and absorbed by
scale-pinning, §6.3).

### 4.3 The metric: layer-diagonal pullback

For a weight perturbation dθ_l, dD_l = (∂D_l/∂θ_l) dθ_l. ORBIT's trust-region
norm drops the cross-layer terms of Proposition 1's exact quadratic form:

  ‖dθ‖²_F := Σ_l ⟨ dD_l , M dD_l ⟩,  with M ≈ diag(r),
  r_i := E_t[ (g_p)_{t,i}² ]  (empirical-Fisher diagonal of the pullback).

**Why layer-diagonal, precisely.** By Cauchy–Schwarz,
‖Σ_l dD_l‖²_M ≤ L·Σ_l ‖dD_l‖²_M, so F_diag ⪰ (1/L)·F_exact: the layer-diagonal
trust region is contained in a √L-inflated exact region. The two disagree most on
**cancellation directions** (Σ dD_l ≈ 0 with individual dD_l large), where F_exact
assigns ~zero cost — i.e. the *exact* function-space metric would license
unbounded parameter drift along the function-null cone (a singular, ill-posed
direction). F_diag charges those directions fully. The approximation error
therefore has the conservative sign, and it is the *same* sign as PIED's implicit
anti-cancellation regularizer — the one increment-geometry intervention that
shipped. This is also the categorical difference from PACT: PACT added a loss
term (moves stationary points; failed at 30k); ORBIT changes the metric
(stationary points invariant; only the path is re-timed).

---

## 5. Per-tensor metric factors (closed forms and estimators)

All factors are FP32 vectors; per-element v is **deleted** for matrix tensors.
Small tensors (γ/β, QK-γ, rot_phi: ~150k params) keep stock Adam untouched.

### 5.1 Wo[l] (writes increments directly)

D_l[t,·] = h_tᵀ Wo with h_t ∈ ℝ^{dModel} the attention-combined input. Then
⟨dD_l, diag(r) dD_l⟩ = tr(dWo diag(r) dWoᵀ H), H = Σ_t h_t h_tᵀ. Diagonal-factored:

  F_{Wo[l]} ≈ diag(ĉ^{(l)}) ⊗ diag(r),
  ĉ^{(l)}_j = EMA_t[ h_{t,j}² ]  (per-layer, dModel-side),
  r = **one shared m-vector for the whole Wo family** (Proposition 1).

Estimators: r from row-square reductions of the dp buffer at layer boundaries
{L, 2L/3, L/3, 0} (µstep-0 only, EMA-merged — the strided set is what makes the
estimator robust to WhiSC trunk mixing); ĉ^{(l)} from a stride-16 column-square
pass over h during the inverse walk (activations are recomputed there — zero
extra memory).

### 5.2 Wq/Wk/Wv[l] (enter through the attention nonlinearity)

Linearizing D_l in θ ∈ {Wq,Wk,Wv}, the GN output factor is colored per layer by
Attnᵀ Woᵀ diag(r) Wo Attn (with the fixed DCT basis contributing a fixed
near-orthogonal factor). No cheap shared structure survives the coloring, so here
ORBIT uses the factored **empirical Fisher of the gradient itself**:

  F_{QKV} ≈ diag(R) ⊗ diag(C),
  R_i = EMA[ mean_j G_ij² ],  C_j = EMA[ mean_i G_ij² ]

(one fused row+col squared-sum pass over each accumulated gradient). This is
where ORBIT deliberately degenerates to Adafactor-shape — the increment-space
derivation contributes the *placement* (output rows carry the attention coloring;
input cols see reln-normalized q̃, near-isotropic by the KRONOS argument) and the
scale-pinning below, not a new factor source.

### 5.3 E (tied embedding/readout — appears at t=0 and t=L)

Row v of E has two Fisher contributions:
- **Readout side:** logit v's multinomial Fisher diag is E_t[π_{t,v}(1−π_{t,v})]
  =: occupancy ô_v (EMA; fused into the readout backward, which already touches π).
- **Input side:** row v moves q_0 only where token v occurs — weight
  κ·f_v·s̄, f_v = batch token count (free from the ids), s̄ = EMA E[‖dq_0‖²]/m.

Combined row factor ρ_{E,v} = ô_v + κ·f_v·s̄ (default κ=1), floored at
quantile_{0.1}(ρ_E) so unseen/rare rows get boosted steps but never unbounded
(this subsumes FACE's hard active-row gating with the smooth exact-Fisher weight).
Column factor c_{E,j} = EMA column moments of dE (FP32, already resident).

### 5.4 Scale-pinning (lr transfer)

Factors define only the **shape** of the second moment; the overall scale is
pinned per tensor to the true gradient energy:

  v̂_ij := S_τ · r̃_i · c̃_j,  r̃ = r̂/mean(r̂), c̃ = ĉ/mean(ĉ),
  S_τ := EMA[ mean_ij G_ij² ]  (one scalar per tensor, from the existing
  `sum_squared_accumulate[_bf16]` reduction).

Properties: (i) if factors are flat, ORBIT ≡ per-tensor-RMS Adam — lr 3e-4
transfers; (ii) PIED's uniform E[η²] inflation and the BF16-grad additive noise
floor cancel in r̃, c̃ (mean-one normalized) and enter only S_τ, where they act as
mild extra damping (shrink, never flip, the step — §9.3).

---

## 6. The ORBIT update rule

Per optimizer step, per matrix tensor τ, after the **unchanged** pipeline
(accum → dq clamps → group clamps → global τ=0.5 clip → grad-skip guard → LR
schedule):

```
(1) m ← β1·m + (1−β1)·G                    # int8 block-256 momentum, machinery unchanged, β1=0.9
(2) m̂ = m / (1−β1^k)                       # bias correction as today
(3) v̂_ij = S_τ · r̃_i · c̃_j                 # factored second moment (FP32 vectors, §5)
(4) u_ij = m̂_ij / (sqrt(v̂_ij) + ε)         # ε=1e-8, outside sqrt (baseline convention)
(5) ℓ_F² = Σ_τ Σ_ij v̂_ij · (lr_k·u_ij)²    # step length measured IN the metric (optional stage)
(6) s = min(1, Δ_F / ℓ_F^{prev})           # one-step-delayed global multiplier; Δ_F=0 ⇒ s≡1 (default)
(7) θ ← θ − s·lr_k·u − lr_k·wd·θ           # decoupled wd unchanged; BF16 tensors via existing SR writeback
```

Adam ordering is kept deliberately (momentum on the raw gradient, precondition
m̂): it minimizes the delta from the baseline's m dynamics and keeps the int8-m
machinery bit-compatible. Factor EMAs use β_f = 0.98 (~50-step time constant:
≪ the ρ e-folding early, ≫ per-step noise), bias-corrected, frozen at 1 for the
first 200 steps (identity warmup inside the existing lr warmup), and are **not
updated on grad-skip steps** (the metric never sees NaN batches). Stats
collection is µstep-0-only. The one-step delay in (6) keeps the step single-pass
and deterministic. Default configuration disables stage (5)–(6) (Δ_F=0): the
primary mechanism is the metric; the function-space cap is a separately gated
second experiment, calibrated at E2 to bind only on outliers
(Δ_F = 1.5× median ℓ_F).

**Variational statement.** Steps (3)–(4) with s from (5)–(6) solve, to first
order, max_dθ ⟨−m̂, dθ⟩ s.t. ‖dθ‖²_F ≤ Δ², whose closed form is the
preconditioned step dθ ∝ −F^{-1/2}-scaled m̂ with the Lagrange multiplier
appearing as the single global scalar s — the function-space analogue of the
parameter-space clip. The sqrt-Fisher convention (F^{-1/2}, not F^{-1}) is chosen
to match Adam's noise robustness and lr transfer; the full-Fisher exponent is
exactly the held KRONOS-W variant.

**Clip interaction (invariant preserved).** The τ=0.5 statistic is computed on
the raw gradient, identical trigger and scaling to baseline — ORBIT is strictly
no looser. On the ρ≈10³ geometry the *difference* from the baseline is that the
parameter-space clip spends one global scalar on raw L2 mass (dominated by
p-side-energy channels), while ORBIT's per-channel sqrt(r̃c̃) discounts
readout-irrelevant p-energy coordinate-wise and lets the global budget act on
output-relevant length.

---

## 7. Temporal dynamics and theoretical analysis

### 7.1 Continuous-time view

ORBIT discretizes θ̇ = −F(θ,t)^{-1/2} ∇f(θ) + momentum smoothing, a
time-inhomogeneous mirror flow whose Bregman geometry is the (layer-diagonal,
empirical-Fisher-diagonal) pullback of the readout metric through the increment
map. Stationary points satisfy ∇f = 0 independent of F ≻ 0: **the metric cannot
relocate optima** (the PACT distinction, made formal).

### 7.2 Descent under inexact metric (staleness, factorization defect)

Let v̂ be the factored estimate and V* = E[G²] the per-element target. Define the
multiplicative defect χ := max_ij max( v̂_ij/V*_ij , V*_ij/v̂_ij ). Standard
preconditioned-descent bounds for Adam-family methods depend on the second-moment
estimate only through its upper/lower envelope; hence ORBIT inherits baseline-Adam
stationarity guarantees with constants degraded by at most χ. χ has three
sources: (a) rank-one factorization defect of V* (measured at E2 — kill
criterion); (b) EMA staleness — bounded by the relative drift of r̃c̃S over ~50
steps; the ρ trajectory e-folds over ≳100 steps early and plateaus after ~1k, so
staleness is worst in the warmup window, which the 200-step identity freeze
covers; (c) BF16-gradient noise — an additive, near-uniform inflation of squared
moments ⇒ upward bias in S_τ ⇒ extra damping (slows, never destabilizes).

### 7.3 Cancellation geometry

F_diag/F_exact = 1/L on depth-coherent perturbation directions and → ∞ on
cancelling ones: relative to the exact function-space metric, ORBIT under-steps
cancellation directions by construction. Relative to the baseline, the parameter
clip is *blind* to function relevance (one scalar for all directions); ORBIT is
the first optimizer in the lineage whose step size distinguishes
function-relevant from function-cancelling parameter directions — the same
distinction PIED's implicit regularizer rewards.

### 7.4 Invariances

- Exact under per-tensor gradient rescaling (scale-pinning + Adam ratio).
- Exact under PIED's mean-one mask statistics (uniform E[η²] absorbed by S_τ).
- The r estimator is invariant to the increment signs σ_l (squares) and robust to
  WhiSC trunk rotation by the multi-boundary sampling (§5.1).
- NOT invariant to wd equilibria: decoupled wd in parameter space + a changed
  preconditioner shifts steady-state per-tensor weight norms — monitored risk
  (§12.3), `--orbit-wd-metric` held in reserve.

### 7.5 Limiting cases

- r̃ = c̃ = 1: per-tensor-RMS Adam (lr transfers; E2 sanity anchor).
- Per-element v̂: exact baseline AdamW.
- Factors collapsed to per-tensor scalars: LARS-style trust ratios.
- QKV-only mechanism: Adafactor/MFIO-shaped. What ORBIT adds beyond each: the
  depth-shared readout row factor (cross-layer metric coupling *without* cross
  terms), the tied-E dual-role Fisher rows, function-space placement of every
  factor, scale-pinning, and the optional function-space global cap.

---

## 8. Memory and wall accounting (flagship shape)

| Item | baseline int8-AdamW | ORBIT |
|---|---|---|
| Momentum m (int8 block-256 + FP32 scales) | 871 MB + 13.6 MB | unchanged |
| Second moment, matrix tensors (870.8M params) | ~871 MB uint8 + 13.6 MB scales | **~2.5 MB FP32 factors** (r 8 KB; ĉ_wo 0.4 MB; QKV R/C 1.8 MB; ρ_E/c_E 0.15 MB; S_τ 97 scalars) |
| Second moment, small tensors (~150k) | ~0.15 MB | ~0.6 MB (stock FP32 Adam) |
| **Total optimizer state** | **≈1.77 GB** | **≈0.89 GB (−50%)** |

Wall (per step, ~2.6 s baseline): QKV/E fused row+col squared-sum passes over
accumulated grads ≈1.2 GB traffic ≈ 0.06–0.17%; dp row stats at 4 boundaries,
µstep-0 only ≈ 0.03%; strided h column stats ≈ 0.1%; step kernel traffic ≤
baseline's (no v read/write; two broadcast vectors L2-resident). **Total ≤0.4%**
(budget 1%). No GEMMs, no eigendecompositions, no extra forward/backward passes.
PACT perf lesson pre-applied: all reductions warp-shuffle, no shared-memory
double atomics.

---

## 9. Comparison to existing methods

| Method | Object it adapts | Structural assumption | ORBIT's relation |
|---|---|---|---|
| AdamW (baseline) | per-element E[G²] | none (coordinate-wise) | ORBIT replaces v with a structure-derived factored moment; recovers AdamW as the per-element limit |
| Adafactor / in-tree MFIO | grad row/col moments | rank-1 V* | same state class for QKV; differs on Wo (adjoint×activation factors from Prop. 1, FP32-clean vs BF16 grad moments), on E (Fisher rows), + scale-pinning + placement derivation |
| KFAC | dense Kronecker factors per layer | layer-independent GN factors | ORBIT = KFAC-diagonal with the output factor *provably shared across all 24 Wo* (KFAC would estimate 24 dense factors and invert them); no inversion anywhere |
| LARS/LAMB | per-tensor norms | uniform within tensor | ORBIT's scalar-collapse limit |
| Sophia-G (in-tree) | diagonal Hessian | curvature source orthogonal to shape | composable later (Sophia's clipped h could replace S_τ) |
| FACE (in-tree) | E rows (active gating) | embedding-only | subsumed: ρ_E is the smooth Fisher generalization of the gate |
| PACT (closed) | loss term on increment Gram | penalties move optima | ORBIT moves the same geometry into the metric; optima invariant |

---

## 10. Failure modes and mitigations

1. **Adafactor-in-disguise null.** The Wo/E-specific factors carry no signal;
   QKV factored moments ≈ MFIO; NLL ties. *Contained:* −50% state at parity is the
   pre-declared structural win. Kill at E4 only if wide-32 gap > +0.02 nat.
2. **Rank-one defect too large** (χ ≫ 1 on QKV): factored v̂ mis-scales exactly
   the attention-signal columns → slower descent. E2 measures χ directly
   (correlation of r̃c̃S vs per-element E[G²]); kill < 0.3 correlation on Wo.
3. **Early mis-scaling of shared r** (nonstationary g_p in warmup; dead
   p-channels): throttles/inflates all 24 Wo at once. Mitigations: 200-step
   identity freeze, damping floor quantile_{0.1}(r̂), multi-boundary sampling.
   Kill at E3 on any grad-skip or ‖g‖ > 2× paired-baseline max.
4. **wd-equilibrium shift → late reversal** (OBSD/PACT signature). Monitor
   stepRMS/weightRMS via the existing per-group stats kernel; `--orbit-wd-metric`
   reserve variant (wd measured in the metric). Kill at E4 if the val-gap trend
   changes sign between 9k and 21k (stop at 21k).
5. **BF16 attention-grad noise inflating factors non-uniformly** → under-trained
   attention. Bounded at E2 by CPU-FP32 reference comparison of factors.

---

## 11. Implementation sketch (for the future build arc — NOT this arc)

**Kernels (glades-ml):**
- `orbit_row_col_sqsum` — fused single-pass row+col squared sums over an
  m×dModel gradient (BF16/FP32 input), warp-shuffle reductions.
- `orbit_adjoint_rows` — [T×m] row-square reduction over dp at a given layer
  boundary; EMA-merge into shared r.
- `orbit_h_colsq_strided` — stride-16 column stats over h during the inverse walk.
- `orbit_occupancy_bwd` — flag-gated variant of the readout backward adding a
  π(1−π) column accumulate (separate kernel name — E0 discipline).
- `orbit_factored_step` — elementwise step (1)–(4),(7): grad + r/c/S + int8-m →
  FP32 scratch → existing SR writeback; emits ℓ_F² block partials for stage (5).
- CPU references for each; suite `test.sh chiron-orbit`.

**Trainer (`chiron_main.cpp`):** factor buffers allocated beside the FACE state
block; stats hooks (µstep-0-gated) at the dp boundaries / inverse-walk h /
readout backward; a new branch in the `adam_one`/`adam_step` per-tensor dispatch
keyed matrix-vs-small. **Naming caution:** an unrelated `--orion` branch exists
in the same dispatch — keep symbols distinct (`orbit_*`). Rebuild order gotcha
applies (`make install` then trainer `bash build.sh`; static kernel link).

**Flags (all default-off ⇒ E0 bit-parity):** `--orbit`, `--orbit-beta-f 0.98`,
`--orbit-freeze 200`, `--orbit-damp-q 0.1`, `--orbit-delta 0` (0 = stage (5)–(6)
disabled), `--orbit-e-fisher` (default-on under `--orbit`), `--orbit-kappa 1.0`,
`--orbit-act-stride 16`, `--orbit-wd-metric` (reserve).

**Monitor:** `[orbit]` line at log cadence — per-class stepRMS/weightRMS, factor
dynamic ranges, χ-proxy on one sampled tensor, clip-fire count (invariant: within
2× of paired baseline).

---

## 12. Gate plan (pre-registered; to be executed in a future arc when GPU frees)

- **E0 — flag-off bit-parity.** 200-step production-shape run, hash vs master
  binary. Kill: any bit difference.
- **E1 — units vs CPU refs** (`test.sh chiron-orbit`): all five kernels vs CPU;
  factored step vs naive dense-metric reference at toy shape; grad-skip leaves
  factors untouched; determinism. Kill: any assert.
- **E2 — small shape** (L=4, m=256, T=1024, 500 steps): (a) loss within noise of
  matched Adam; (b) factorization-defect measurement: corr(r̃c̃S, per-element
  E[G²]) > 0.7 on Wo, > 0.5 on QKV (kill < 0.3); (c) BF16-vs-FP32 factor bias
  < 2×; (d) calibrate Δ_F. Decision point: if (b) fails on QKV but passes on Wo,
  pivot to the KRONOS-W bolt-on form (keep v, apply factors as whitening).
- **E3 — matched-pair 2500 steps** (T=16384, seed 1337, fresh paired baseline,
  same binary, step-1-matched). Kills: val@2500 > baseline + 0.02 nat; any
  grad-skip; max ‖g‖ > 2× baseline max; wall > +1%; clip-fire rate > 2× baseline.
  **PACT rule: a favorable E3 gap is NOT bankable evidence — E3 kills only.**
- **E4 — matched 30k** (wide-32 val, mid-checkpoints 9k/15k/21k). Ship:
  ≤ baseline − 0.05 nat; or within ±0.02 nat with the −50% state win banked.
  Kill: > +0.02 nat, or gap-trend sign-flip after 10k (washout signature — stop
  at 21k). Single-seed per lineage precedent; multi-seed remains the standing
  caveat.

Estimated gate cost (for planning): E0–E2 < 1 GPU-hr; E3 ≈ 2×1.7 GPU-hr;
E4 ≈ 2×33 GPU-hr (treatment + fresh paired baseline — mandatory per era-drift).

---

## 13. Portfolio holds and revive triggers

- **RELIC (hold).** Revive if: (a) ORBIT E4 ties and the owner wants the
  clip-machinery consolidation + the schedule-free α-anneal; or (b) as the
  long-term composition — RELIC's bounded-velocity conformal-symplectic kinetics
  running on ORBIT's metric ("relativistic natural gradient": kinetic energy
  K(P) = Σ c(√(P²+s²)−s) with s from ORBIT's v̂ instead of block-RMS). Pre-req
  before any build: the E2.5 calibration probe (200 baseline steps recording the
  global stepRMS envelope) and the saturation-fraction telemetry, because the
  candidate's main risk is unanchoring from the validated τ=0.5 geometry.
- **KRONOS-W (hold).** Revive if ORBIT E2 shows the factored-v replacement is the
  blocker while Wo/E factors carry signal: apply the same factors as
  multiplicative whitening on top of stock per-element int8-Adam (~1.5 MB state,
  no v deletion). Also independently interesting: its int8-moment-SNR claim
  (whitening equalizes within-block scales → better quantization) is measurable
  offline at E1 from captured production gradients.
- **KRONOS-F** is subsumed by ORBIT (same state cut, better-derived factors).

---

## 14. Long-term research program

1. **ORBIT build + gates** (next GPU window): E0→E4 as §12.
2. **Function-space cap arm** (`--orbit-delta`): does a global cap in increment
   space beat the parameter-space clip when *added* to it? (Strictly-no-looser
   design keeps this safe.)
3. **RELIC×ORBIT composition**: conformal-symplectic flow on the increment
   metric — the full "CHIRON-native optimizer" (architecture-matched geometry AND
   dynamics). Only after both parents are individually understood.
4. **Cross-layer metric correction**: PACT's increment-Gram infrastructure can
   estimate the L×L cross-layer block as a low-rank correction to F_diag —
   revisit only if E4 telemetry shows cancellation directions dominating steps.
5. **Sophia-G composition**: replace S_τ with clipped Hessian-diagonal scale
   (curvature source orthogonal to factor shape).
6. **Spend the freed ~0.9 GB**: MTP production port (was blocked at ~2.6 GB
   scratch) and/or reversible-FFN+GQA arc headroom.

---

## 15. Claim ledger and open conjectures

**Theorem-level (proved here under stated assumptions):**
- Proposition 1 (increment-interface factorization) at θ_WhiSC = 0; the
  depth-constancy of λ_p; PIED-mask compatibility.
- F_diag ⪰ (1/L)F_exact and the conservative sign of the layer-diagonal
  approximation on cancellation directions.
- Metric changes cannot relocate stationary points (vs loss-side penalties).
- ORBIT ⊆ baseline trust structure (clip retained ⇒ no looser).

**Derivable under assumptions:** χ-degraded Adam-style stationarity bounds
(§7.2); staleness bounds from the measured ρ trajectory; BF16-noise ⇒ damping
(not instability).

**Heuristics:** sqrt-Fisher exponent choice; β_f = 0.98; the specific boundary
set {L, 2L/3, L/3, 0}; κ=1; Δ_F = 1.5× median.

**Conjectures (falsifiable at gates):**
- C1: rank-one defect χ is small on Wo (the Prop.-1 factors are near-exact) —
  E2(b).
- C2: readout-relevance row scaling of Wo steps improves 30k asymptote, not just
  early trajectory — E4 with mid-checkpoints.
- C3: Fisher-occupancy E rows beat both stock Adam and FACE gating on the Zipf
  tail at 1.97B tokens (late-training lever) — E4 ablation `--orbit-e-fisher`
  off/on if E4 headline passes.
- C4 (portfolio): the saturated-velocity regime (RELIC) reproduces the clip
  optimum with state −50% — held.

**Validation criteria:** the §12 gates verbatim; ship bar §1.3; all engineering
default-off regardless of outcome.
