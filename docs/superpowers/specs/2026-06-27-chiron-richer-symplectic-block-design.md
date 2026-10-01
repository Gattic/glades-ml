# CHIRON Richer Symplectic Block — Operator-Budgeted Symplectic Drift (OBSD)

**Date:** 2026-06-27
**Status:** Design (approved for planning)
**Author:** brainstorming + research-framework-design (3-candidate parallel synthesis)
**Baseline:** CHIRON 1B reanchor-cure flagship (`chiron_1B_T16384_reanchor5B_finish.final`, val NLL ~1.92)
**Target:** lower validation NLL at production scale by enabling stable cross-depth attention
composition through the `q` channel, preserving exact reversibility / O(1)-in-depth activation memory.

---

## 1. Executive summary

The production CHIRON block is a **kick-only symplectic integrator**: each layer applies the
attention shear `p += Y_l(q)` and a per-layer reln on `q`, but momentum `p` is folded into `q`
**exactly once**, at the final layer. Consequently every interior layer attends over a `q` that
has only been affine-renormalized — never attention-transformed. The model does not compose
attention across depth; it behaves like **L parallel attention groups summed and read out once**,
not a deep transformer. This is the hypothesized dominant perplexity ceiling.

OBSD restores the missing per-layer **drift** step

> **`q += a_l ⊙ φ( M_l⁻¹ ⊙ N(p) + b_l )`** , `N(p) = (p − μ_p)/σ_p` (parameter-free row-normalize)

at every layer, with `φ` a bounded saturating nonlinearity (`tanh` or relativistic
`u/√(1+u²)`), `M_l⁻¹ ∈ ℝᵐ` a learnable **diagonal inverse-mass** (inside `φ`), `b_l ∈ ℝᵐ` an
operating-point offset, and `a_l ∈ ℝᵐ` a **ReZero per-channel gate initialized to 0** so the block
is bit-identical to the flagship at init and ramps coupling in as it learns. The drift reads only
`p` (untouched during the step), so the block stays unit-triangular and **exactly invertible**;
O(1) activation memory is preserved (the drift stores nothing — its normalization stats are
recomputed from the reconstructed `p`, reanchor-style).

The disabled legacy coupling (`q += α·reln(p)`, `α=1/√L` or `k/(T√L)`) exploded gradients
(‖g‖ ≈ 1.4e6–4.6e18 at step 1, L=24/m=2048). OBSD's central theoretical result is that the
**binding stability quantity is the operator budget**

> **`B = Σ_l √( ‖G_l‖₂ · ‖A_l‖₂ )`** , `G_l = ∂(drift)/∂p` , `A_l = ∂Y_l/∂q`,

a geometric-mean Lyapunov exponent. The backward amplification obeys `‖∏_l J_lᵀ‖ ≤ C·exp(B)`.
This (i) quantitatively reproduces the observed explosion magnitude, and (ii) explains why **both**
legacy dampings failed — they shrink `Σ g_l`, not the geometric mean `Σ √(g_l a_l)`. OBSD keeps
`B = O(1)` by construction via three **source-level** controls — zero-init `a_l`, bounded `φ`, and
p-side reanchor — plus an optional parameter-space budget projection (acting on parameters, not
gradients; categorically not gg-clamp).

OBSD is the principled generalization of the dormant `chiron_reln_axpy_into_q` kernel: it adds no
new GEMM (one attention pass/layer suffices — see §8.6), costs ~+8–10% wall, ~150K new params
(`3·L·m`), and recovers the flagship exactly at `a_l ≡ 0`.

---

## 2. Candidate formulations (provenance)

Three materially different lenses were developed in parallel and **converged on the same
mechanism**, which is itself evidence the mechanism is correct. They differed in the analytical
certificate and the control strategy.

- **A — HKD (Hamiltonian Kick–Drift).** Symplectic-integrator lens: attention = force `−∇_q V`,
  drift = `∇_p T(p)`, diagonal mass = Hessian of a learnable kinetic energy `T`, relativistic `φ`
  = gradient of a convex non-quadratic kinetic term. Key results: **single kick + exact drift is
  optimal** (multi-substep drift is a no-op for a separable Hamiltonian, so no second attention
  pass), and shadow-Hamiltonian energy conservation bounds state norms. Weakness: energy
  conservation holds only for fixed/slowly-varying `H`, **explicitly not** for the
  non-conservative attention curl (the actual regime); its mass placement (outside `φ`) is
  degenerate with the gate.

- **B — OBSD (Operator-Budgeted Symplectic Drift).** Operator/spectral lens: gradient explosion
  as control of `‖∏_l J_lᵀ‖`. Key results: the operator budget `B = Σ_l √(‖G_l‖‖A_l‖)`
  (quantitatively matches the explosion and explains both damping failures); **mass
  non-absorbability** (Theorem, §8.3) requiring nonlinear `φ` + parameter-free normalize; reanchor
  as an operator-norm-conditioning device; the structural fact that **monotone gradient decay is
  impossible** for symplectic coupling, so bounded-growth-under-budget is the correct target (and
  per-step clamping is the wrong tool).

- **C — SYMFLOW-H (homotopy-controlled symplectic mirror flow).** Continuous-depth / control +
  information-geometry lens: depth-ODE with a homotopy parameter `ε:0→1` deforming flagship →
  coupled; drift = mirror-descent step, mass = diagonal Fisher; budget
  `‖a_l‖∞ ≲ C²σ_p/(L²‖A_l‖)` (an `O(1/L²)` ceiling). Key result: **reanchor is necessary but not
  sufficient** — the budgeted gate is independently required. Weakness: its `L`-independent
  no-explosion theorem holds only under convex attention (false in general); its bilevel
  controller adds operational complexity.

---

## 3. Framework selection rationale

**Selected: OBSD (B), grafting A's single-kick result and C's ramp schedule.**

1. **Its certificate covers the actual regime.** A's energy conservation and C's `L`-independence
   require conservative / convex attention, which CHIRON violates. OBSD's product-of-Jacobians
   budget assumes neither and bounds the gradient directly — the precise failure to prevent.
2. **Sharpest, falsifiable diagnosis.** `B = Σ√(‖G_l‖‖A_l‖)` is measurable per step, reproduces
   the 1e6–1e18 explosion, and uniquely explains why `α=1/√L` and `k/(T√L)` both failed.
3. **Non-degenerate by construction** (mass inside `φ`, gate outside — B and C agree; A's is
   degenerate).
4. **Doctrine fit.** "Bounded growth under a budget, not monotone decay" matches the project's
   cure-at-source principle and rejects the gg-clamp containment that taxed perplexity ~0.7–0.8
   nat in the prior lineage.
5. **Most direct code mapping** — the principled generalization of the dormant
   `chiron_reln_axpy_into_q` kernel; each of its four defects maps to one mandatory ingredient.

**Grafted from A:** the Hamiltonian reading (justifies mass-as-curvature and the relativistic
kinetic term) and the **single-kick optimality** result (cost stays +8–10%, not +60%).
**Grafted from C:** the `O(1/L²)` warmup-ramp ceiling and the **reanchor-necessary-but-not-
sufficient** finding (need reanchor *and* the budgeted gate). C's bilevel controller is demoted to
an optional enhancement (scope discipline).

---

## 4. Formal problem statement

- **System.** CHIRON, reversible symplectic-flow transformer. Flagship: L=24 layers, m=2048
  channels per branch, T=16384 tokens, nH=16, dH=256, V=32000 BPE, ~871M params. Per-layer state
  `(q, p)`, `q,p ∈ ℝ^{T×m}`. Block map `(q,p) ↦ (q',p')` is unit-triangular and **exactly
  invertible**; backward reconstructs activations via the inverse walk (no stored activations →
  O(1)-in-depth activation memory). Reversibility is the architecture's reason to exist.
- **Current block** (l < L−1): `p += Y_l(q)` (SCFA spectral attention, function of `q` only);
  `q = reln(q; γ_l, β_l)`. Fold `q += p` only at l=L−1; `logits = reln(q+p)·Eᵀ`.
- **Objective.** Minimize validation next-token NLL at production scale by enabling stable
  cross-depth attention composition through `q`, **without** regressing the flagship (val ≈1.92)
  and **without** sacrificing exact reversibility / O(1) memory.
- **Mandatory mechanisms.** (1) per-layer reversible drift `q += g_l(p)`; (2) learnable diagonal
  mass `M_l⁻¹` (per-channel, distinct from reln γ); (3) nonlinear gated drift `φ`; (4) ReZero gate
  `a_l ∈ ℝᵐ`, init 0; (5) source-level (reanchor) backward stability.
- **Constraints.** Exact reversibility (q-update reads only p; p-update only q; unit-triangular,
  invertible). BF16/FP8 backward stability at L=24/m=2048/T=16384. Prefer elementwise ops (no new
  GEMMs). C++98 + CUDA in glades-ml/glades-trainer. Default-off / gated; `a=0` ⇒ default-on
  identical to flagship. Deterministic.
- **Forbidden simplifications.** Mass ≠ scalar; nonlinear drift ≠ redundant rescale; no breaking
  reversibility for expressivity; no clamp/containment-only stability; no fix that enriches only
  the final fold.
- **Evaluation.** Val NLL < 1.92; 0 grad-skips through the previously-exploding regime; exact
  flagship reproduction at init; wall ≤ +10% / negligible VRAM; inverse-reconstruction error
  within BF16 ULP.

---

## 5. Core mathematical framework

### 5.1 Symbols

| Symbol | Space | Meaning | Init |
|---|---|---|---|
| `q, p` | ℝ^{T×m} | position / momentum half-state; row `t`, channel `j` | — |
| `Y_l(·)` | ℝ^{T×m}→ℝ^{T×m} | SCFA attention output (the kick / force); reads `q` only | (existing) |
| `N(p)` | ℝ^{T×m} | parameter-free row-normalize `(p−μ_p·1)/σ_p`; `μ_p,σ_p` per-row mean/std | — |
| `reln(·;γ,β)` | — | reversible LayerNorm of `q`; saves per-row `(μ,σ)` | — |
| `γ_l, β_l` | ℝᵐ | q-side reln affine (existing; **not** the mass) | flagship |
| `M_l⁻¹` | ℝᵐ_{>0} | **diagonal inverse mass** — per-channel gain *inside* `φ` | `1` |
| `b_l` | ℝᵐ | operating-point offset *inside* `φ` | `0` |
| `a_l` | ℝᵐ | **ReZero output gate** *outside* `φ` | `0` |
| `φ` | ℝ→ℝ | bounded, `L_φ`-Lipschitz nonlinearity (`tanh`, `L_φ=1`; or relativistic `u/√(1+u²)`) | — |
| `g_l(p)` | ℝ^{T×m} | the drift `a_l ⊙ φ(M_l⁻¹ ⊙ N(p) + b_l)` (broadcast over `t`) | — |
| `A_l := ∂Y_l/∂q` | n×n | attention Jacobian, `n=Tm` | — |
| `G_l := ∂g_l/∂p` | n×n | drift Jacobian (§8.4) | — |
| `R_l := ∂reln/∂q` | n×n | reln Jacobian | — |
| `E` | ℝ^{V×m} | tied output embedding | — |

### 5.2 Forward block (l < L−1), order kick → drift → reln

```
(K)  p̃ = p + Y_l(q)                                  # p-update reads q only        (unchanged)
(D)  q̃ = q + a_l ⊙ φ( M_l⁻¹ ⊙ N(p̃) + b_l )           # q-update reads p only         (NEW)
(G)  q' = reln(q̃; γ_l, β_l) ,   p' = p̃               # q-only reversible map         (unchanged)
```

Final layer l=L−1: `(K)`, `(D)`, then **fold** `q ← q + p̃`, then `(G)`; `logits = q'·Eᵀ`.
The flagship is exactly the `a_l ≡ 0` (l<L−1) slice with the final fold retained.

### 5.3 Inverse block (exact; stores nothing new)

Given `(q', p')` and the saved q-reln stats `(μ,σ)`:

```
q̃ = reln⁻¹(q'; γ_l, β_l, μ, σ)
p̃ = p'
q  = q̃ − a_l ⊙ φ( M_l⁻¹ ⊙ N(p̃) + b_l )              # recompute drift from p̃ = p'
p  = p̃ − Y_l(q)                                      # chiron_attention_shear(invert=true)
```

`N(p̃)` and the drift are recomputed identically from `p̃` (= `p'`, untouched by `(D)`), so the
inverse is exact in real arithmetic. The drift's `(μ_p, σ_p)` are **not stored** — they are
re-derived from the reconstructed `p̃` (reanchor; §8.5). O(1)-in-depth memory preserved.

### 5.4 Block Jacobian (operator form)

Elementary Jacobians: `S_K=[[I,0],[A_l,I]]`, `S_D=[[I,G_l],[0,I]]`, `S_G=[[R_l,0],[0,I]]`. Composite:

```
J_l = S_G S_D S_K = [[ R_l(I + G_l A_l),  R_l G_l ],
                     [ A_l,               I       ]] ,   det J_l = det R_l.
```

The off-diagonal block `R_l G_l` is *exactly* the operator that lets a `p`-perturbation reach `q`
at every layer — the cross-depth composition that is absent in the flagship (`G_l = 0`).

---

## 6. Objective and flagship recovery

**Training objective is unchanged** (next-token NLL + existing regularizers z-loss / terminal
SIRA):

```
L(θ) = E[ NLL(reln_L(q_L+p_L)·Eᵀ, target) ] + λ_z·zloss + λ_S·SIRA(q_L,p_L),
       θ ⊇ {a_l, M_l⁻¹, b_l} ∪ flagship params.
```

No new loss term is required (stability is structural). An **optional** parameter-space budget
penalty `λ_B · max(0, B − B_max)²` enforces `B ≤ B_max` smoothly — distinct from gradient
clamping (it shapes parameters, Adam-compatible).

**Flagship recovery [THEOREM].** `a_l ≡ 0` (l<L−1) ⇒ `g_l ≡ 0` exactly (before any rounding) ⇒
`(D)` is the identity ⇒ `J_l = [[R_l,0],[A_l,I]]` = the flagship block Jacobian. Forward, inverse,
and gradients are bit-identical to `chiron_1B_T16384_reanchor5B_finish.final`. Default-on OBSD
cannot regress val 1.92; all improvement comes from the optimizer choosing `a_l ≠ 0`.

---

## 7. Training schedule (the control law)

- **Zero-init + warmup ramp.** `a_l = 0` at step 0 (B=0). Ramp the gate magnitude from 0 over
  ~250 steps reusing the existing `--sira-warmup` mechanism, with ceiling informed by the C-graft
  budget `‖a_l‖∞ ≲ C²/(L²·‖A_l‖₂)` (an `O(1/L²)` cap; `C ∈ [1,3]`).
- **Source-level structural controls (primary).** (1) zero-init `a_l`; (2) bounded `φ`
  (`L_φ=1`); (3) p-side reanchor (re-derive `(μ_p,σ_p)` from reconstructed `p`). Per the C-graft,
  reanchor alone is **necessary but not sufficient** — the budgeted gate is independently required.
- **Optional safety net.** Every N steps, estimate `‖A_l‖₂` by 1–2 power-iterations on the
  attention block; if `B = Σ_l √(‖G_l‖₂‖A_l‖₂)` approaches `B_max`, project `a_l` onto the budget
  ball (closed-form `‖G_l‖₂ ≤ ‖a_l‖∞‖M_l⁻¹‖∞κ_N`). Acts on parameters, not gradients.
- Gated by `--per-layer-drift` (default off). `--per-layer-drift` with `a=0` is bit-identical to
  the flagship and serves as the in-run control.

---

## 8. Theoretical analysis

### 8.1 Exact invertibility / O(1) memory [THEOREM]
Each sub-step is an elementary unit-triangular shear `(K),(D)` or an invertible elementwise reln
`(G)`; the inverse is the listed reverse sequence (§5.3), exact by direct substitution.
`det J_l = det R_l` (shears unit-determinant). The drift stores no activations — its stats are
recomputed — so depth memory is O(1), unchanged from the flagship.

### 8.2 BF16 reconstruction [DERIVABLE]
The only new inverse term is `−g_l(p̃)`. If `g_l` is computed in FP32 with a single final rounding
to storage dtype on **both** forward and inverse (existing kernels use FP32 reduction
accumulators), forward and inverse evaluations are bit-identical ⇒ zero added reconstruction error
in the drift term ⇒ within BF16 ULP. Requires FP32 `tanhf` + single rounding (FM-3).

### 8.3 Mass non-absorbability [THEOREM]
With `φ` nonlinear and the p-normalize parameter-free (no free affine on `N(p)`), `M_l⁻¹` (inside
`φ`) is not absorbable into `a_l` (outside `φ`), `b_l` (additive inside), or the q-side `(γ_l,β_l)`
(applied to `q̃` after the drift, a different operand). Proof: `a⊙φ(M⁻¹x̂+b)` equals `ã⊙φ(x̂+b)` for
some `ã` iff `φ(cs)=k(c)φ(s)` jointly in `c,s`, i.e. iff `φ` is linear. **This is why the
nonlinearity is mandatory:** with a linear `φ` (the legacy path) `M⁻¹` collapses into the affine —
the "trivial scalar mass" failure the brief forbids.

### 8.4 Off-diagonal norm bound [DERIVABLE]
`G_l = diag(a_l)·diag(φ'(M_l⁻¹⊙x̂+b_l))·diag(M_l⁻¹)·J_N(p)`, where
`J_N(p) = (1/σ_p)(I − 11ᵀ/m − x̂x̂ᵀ/m)` (per row) is the normalize Jacobian, `x̂=N(p)`. The
projector has `‖·‖₂ ≤ 1`, so
`‖G_l‖₂ ≤ ‖a_l‖∞ · L_φ · ‖M_l⁻¹‖∞ · κ_N`, `κ_N := sup‖J_N‖₂ ≤ 1/σ_p`. Under reanchor `σ_p` is the
true row-std ⇒ `x̂` is unit-RMS ⇒ `κ_N = O(1)`. Thus `‖G_l‖₂` is a quantity the optimizer controls
through `a_l` and `M_l⁻¹`, never an unbounded data spike.

### 8.5 The operator budget and the explosion diagnosis [DERIVABLE; central result]
Backward, `δ_l = J_lᵀ δ_{l+1}`. With the nonnegative envelope `P_l` from the block-norms
`a=‖A_l‖₂, g=‖G_l‖₂, ρ=‖R_l‖₂≤1`, for `ρ=1` (`det P_l=1`) the per-layer Lyapunov exponent is
`log λ₊(P_l) = √(g_l a_l) + O(g_l a_l)`, giving the **sharp rate**

```
‖∏_l J_lᵀ‖  ~  C · exp( Σ_l √( ‖G_l‖₂ ‖A_l‖₂ ) )  =  C · exp(B).
```

`B = O(1)` ⇒ depth-uniform gradient bound ⇒ no explosion. The flagship is `B = 0` (block-
triangular, `‖∏J‖ = O(L)` — benign linear growth, matching the additive `p`-accumulation).

**Why the legacy dampings failed.** For `α=1/√L`: `g_l ~ Θ(1/√L)`, `a=‖A_l‖₂ = Θ(1)–Θ(10)`, so
`√(g_l a_l) ~ L^{-1/4}√a` and `B ~ L^{3/4}√a`. For L=24, `a∈[1.7,18]` ⇒ `B∈[14,43]` ⇒
`exp(B) ≈ [1.4e6, 4.6e18]` — the observed step-1 range. Both `α=1/√L` and `k/(T√L)` shrink `Σ g_l`
but leave the geometric mean `Σ√(g_l a_l)` large (the former `Θ(L^{3/4})`, the latter still
`Θ(L^{3/4}√a)` once `a` dominates at large `m`). The binding quantity is the geometric mean, which
neither targets.

**Monotone decay is impossible [REMARK].** `det P_l = ρ_l` and the shears are area-preserving, so
`λ₊(P_l) > 1` whenever coupling is on. No fixed norm makes the recurrence non-increasing for
nonzero symplectic coupling — the correct target is bounded growth under a budget, which is why
clamp-every-step containment is the wrong tool and a structural budget is right.

### 8.6 Single-kick optimality [THEOREM, grafted from A]
For a separable Hamiltonian `H=V(q)+T(p)`, the drift sub-flow `q̇=∇_p T(p)` with `p` frozen has
exact one-shot solution `q += ∇_p T(p)`; sub-stepping the drift is a provable no-op. A full
per-layer leapfrog needs two attention passes (the second half-kick uses the drift-moved `q`),
~2× the 99%-hot-path GEMMs. A stack of symplectic-Euler layers already equals a leapfrog
integrator up to boundary half-kicks. ⇒ **one kick + exact drift per layer is correct**; cost
stays +8–10%, no new GEMM. Optional Strang boundary half-kicks (free 2nd order) are a long-term
enhancement.

### 8.7 Reanchor as operator conditioning [DERIVABLE]
Saved-but-drifted stats `(μ_s,σ_s)`, `ρ_drift=σ(p)/σ_s`, give
`‖J_N^bwd(s)‖₂ ≤ (1/σ_s)(1+ρ_drift²)`. At the observed `ρ_drift≈13` this is a ~10³× operator-norm
spike (the dgamma/dp overflow). Re-deriving `(μ_s,σ_s)` from the reconstructed `p` forces
`ρ_drift=1` ⇒ `‖J_N^bwd‖₂ ≤ 2/σ(p) = O(1)` by construction. This caps `κ_N` in §8.4 — the source
cure, applied p-side exactly as the shipped q-side cure.

### 8.8 Expressivity [DERIVABLE]
For any `B>0` the off-diagonal `R_l G_l` is generically full-rank (`J_N` rank `m−1`/row;
`diag(φ')diag(M⁻¹)` full-rank for `M⁻¹>0` off-saturation), so `q` at layer `l+1` depends on all
channels of `p̃ = Σ_{k≤l} Y_k(q_k)` ⇒ attention composes across depth. Bounded `φ` caps *gain*, not
*rank* — expressivity is not traded for stability.

### 8.9 Claim-level ledger
- **[THEOREM]** invertibility & O(1) memory (8.1); flagship recovery at init (§6); mass
  non-absorbability (8.3); single-kick optimality (8.6).
- **[DERIVABLE]** BF16 ULP reconstruction (8.2); off-diagonal norm bound (8.4); operator-budget
  rate + explosion diagnosis (8.5, modulo the bounded eigenvector-condition prefactor `C`);
  reanchor operator-norm cap (8.7); expressivity (8.8).
- **[HEURISTIC]** Hamiltonian/relativistic interpretation; `M_l⁻¹ ≈ 1/diag(A_l)` /
  running-second-moment as the conditioning target; `a_l M_l⁻¹ ≲ 1` joint budget.
- **[CONJECTURE / EMPIRICAL]** ∃ usable `B* = O(1)` that is sub-overflow **and** lowers val NLL
  below 1.92 (the central bet, §11 FM-1, §12).

---

## 9. Computational tradeoffs

| Resource | Current block | OBSD delta |
|---|---|---|
| GEMMs / layer | 4 proj + flash core (~99% hot path) | **0 new** (single-kick, §8.6) |
| Elementwise / layer | reln + shears | +~3 streaming passes over T×m fwd, +~5 bwd |
| Wall | baseline | **+8–10%** (bandwidth-bound; quality budget, not the 5% perf bar) |
| Params | flagship | `+3·L·m = 3·24·2048 ≈ 147K` (`a_l, M_l⁻¹, b_l`) ≈ 0.017% of 871M |
| Optimizer state | Adam | +2·147K floats |
| VRAM | flagship | ≈ +1.8 MB; **no new activation storage** |
| Activation memory order | O(1) in depth | unchanged O(1) |
| Inverse-reconstruction error | BF16 ULP | within BF16 ULP (8.2) |

**Code mapping (no new GEMM).** Generalize the dormant `chiron_reln_axpy_into_q`
(`gpu_chiron.cu`): reuse `gamma_p[l]` → `M_l⁻¹` (init ones, already), `beta_p[l]` → `b_l` (init
zeros, already); add one per-layer `a_l ∈ ℝᵐ` buffer (init 0) + Adam state; insert FP32 `tanhf` +
`a_l` multiply into pass 3 (new kernel `chiron_drift_into_q_rows`); clone
`chiron_reln_backward_reanchor` / `chiron_reln_reanchor_stats_kernel` for the **p side**. Call the
drift between `chiron_attention_shear` and `chiron_reln_forward` at every l<L−1 in the trainer
forward (`chiron_main.cpp`), and its backward in the inverse walk. Flag `--per-layer-drift`,
default off.

---

## 10. Comparison to existing approaches

| Approach | Object / assumption | OBSD relation |
|---|---|---|
| **Flagship** (no coupling) | `K≡0`; q only renormalizes; `J_l` block-triangular | exact `a=0` limit; OBSD adds the off-diagonal `R_l G_l` |
| **Legacy `q+=α·reln(p)`** | linear `φ`, scalar `α`, free `γ^p`, no reanchor | special case; explodes (`B=Θ(L^{3/4})`); OBSD fixes each factor |
| **gg-clamp / dq-clamp** | clamp gradients every step | OBSD bounds the operator norm by construction; optional budget acts on *parameters* — no chronic clamping / perplexity tax |
| **Reanchor (shipped, q-side)** | re-derive reln stats from activation | reused p-side as the drift's conditioning device |
| **ReZero / SkipInit** | scalar gate, residual `x+α·f(x)` | per-*channel* gate inside a *reversible symplectic* shear with mass-preconditioned bounded drift + explicit budget |
| **Hamiltonian / leapfrog nets, RevNets** | exact reversibility | OBSD adds the operator budget `Σ√(g_l a_l) ≤ B` as the design principle + relativistic bounded kinetic term |

OBSD reduces to: flagship (`a=0`), legacy coupling (`φ=id, M⁻¹↔γ^p, a=α1`, no reanchor), and a
leapfrog Hamiltonian step (idealized kinetic term); it strictly generalizes all three.

---

## 11. Failure modes and mitigations

- **FM-1 (dominant): budget too small to help.** If val-NLL-improving coupling needs `B > B_max(BF16)`,
  OBSD stays stable but at flagship perplexity. The `√(g a)` exponent means useful and safe
  coupling may be hard to separate. *Mitigation:* per-channel `M_l⁻¹` + per-layer `a_l` concentrate
  coupling on low-`‖A_l‖` layers / responsive channels; measure `B` vs val-NLL on the small-shape
  sweep **before** scale-up. This is the central empirical bet.
- **FM-2: `‖A_l‖₂` larger than assumed at scale.** *Mitigation:* budget uses *measured* `‖A_l‖`
  (power iteration), allocate `a_l ∝ 1/√‖A_l‖`; QK-Norm already bounds attention.
- **FM-3: BF16/FP8 drift recompute mismatch.** *Mitigation:* FP32 `tanhf` + FP32 reductions +
  single final rounding; verify on the roundtrip test.
- **FM-4: `ρ_l = ‖R_l‖ > 1` (expansive reln).** Adds `L·ln ρ_l` to `B`. *Mitigation:* shipped
  q-side reanchor keeps `ρ_l = O(1)`; monitor `‖R_l‖`.
- **FM-5: saturation freezes learning** (`M⁻¹` grows ⇒ `φ'→0` ⇒ `da, dM⁻¹` vanish).
  *Mitigation:* `M⁻¹` init 1 keeps `φ` near-linear while `a_l` ramps; soft cap `M⁻¹`.
- **FM-6: budget projection ↔ Adam interaction.** *Mitigation:* smooth ball reparametrization /
  penalty (§6), not a hard per-step clamp.
- **FM-7: Hamiltonian interpretation is heuristic.** The exact claims rest on invertibility (8.1)
  and the envelope analysis (8.5), which do not need energy conservation.

---

## 12. Minimal prototype and experiment ladder

**Minimum viable instantiation.** As in §9 code mapping; optional budget projection *disabled*
initially (rely on zero-init + bounded `φ` + reanchor). `φ = tanh`, `M⁻¹` init 1, `b` init 0,
`a` init 0.

**Experiment ladder.**
- **E0 — bit-identity gate.** `--per-layer-drift` with `a=0` ⇒ loss and gradients bit-identical to
  the flagship at step 0 (verifies §6).
- **E1 — stability A/B (small shape, e.g. m=256, L=24, T=512, then m=512).** Run (i) legacy
  `q+=α·reln(p)`, `α=1/√L` and (ii) OBSD. PASS: (i) explodes `‖g‖~1e6+` at step 1 while (ii) holds
  `‖g‖~1` with **0 grad-skips**, and measured `Σ√(‖G_l‖‖A_l‖)` tracks `log‖g‖` (validates §8.5
  quantitatively). Ablations: reanchor-off (expect divergence per §8.7 + C-graft), fixed-`α`
  (expect divergence), `M⁻¹≡scalar` (expect worse).
- **E2 — reconstruction.** Inverse-walk error ≤ BF16 ULP (FM-3).
- **E3 — `B`-sweep for `B*`.** Vary `M⁻¹`/`a` init scale; locate the budget that lowers val NLL
  while staying sub-overflow (settles FM-1).
- **E4 — production quality gate (T=16384, L=24, m=2048).**
  - *Primary (single-seed 1337):* the ship val-NLL run. Ship iff val NLL < 1.92, 0 grad-skips
    through the ~1.6B-token regime that previously needed gg-clamp, wall ≤ +10%, reconstruction
    within BF16 ULP.
  - *Multi-seed confirmation:* seeds `{1337, 2024, 4242}`, each capped at **5k or 15k steps at
    most** (NOT a full 30k per seed — owner cost-control decision 2026-06-27). This confirms the
    *sign* of the improvement and **0 grad-skips cross-seed** at reduced cost; it is a
    stability/sign gate, not a full-trajectory multi-seed ship. Consistent with the flagship's
    single-seed ship precedent (the ship metric is the single-seed primary run; the capped
    multi-seed run hardens confidence without 3× the production spend).

---

## 13. Full research program (long-term)

- **Structured mass.** Replace diagonal `M_l⁻¹` with a block-diagonal-per-head (`dH×dH`) or
  DCT-conjugated (reusing the cached SCFA spectral basis) mass operator — richer kinetic metric,
  still no full GEMM, retains non-absorbability (8.3).
- **Strang boundary half-kicks** for free 2nd-order accuracy (8.6).
- **Learned budget allocation** `B_l` under a global `Σ B_l ≤ B` (controllability / optimal
  spectral-budget control across depth) — the principled version of C's bilevel controller.
- **Convex-attention surrogate** to obtain an `L`-independent no-explosion certificate (close the
  gap A and C both hit).
- **Rigorous finite-L bound** by controlling the eigenvector-condition prefactor `C` in §8.5
  (`[DERIVABLE]→[THEOREM]`).
- **SIRA coupling:** regularize the shadow-Hamiltonian energy drift via the existing SIRA action
  term, if the curl-driven energy injection (FM-7) needs explicit control.

---

## 14. Open conjectures and validation criteria

- **C1 (central).** ∃ `B* = O(1)` with `exp(B*)` sub-overflow and val NLL < 1.92. *Falsified if*
  the E3 sweep shows quality only above the BF16-safe budget.
- **C2.** `Σ√(g_l a_l)` is the *sharp* predictor of `log‖g‖` (not `Σg` or `Σ(g+a)`). *Test:*
  regress measured `log‖g‖` on the three candidates across the E1 sweep.
- **C3.** Per-channel `M⁻¹` + per-layer `a_l` beats a single scalar budget. *Test:* ablate
  `M⁻¹≡scalar` (E1).
- **C4.** Reanchor on the drift path is *necessary* (not just helpful) for 0 grad-skips at scale.
  *Test:* OBSD with saved-stat backward should re-exhibit the κ_N-driven spike.

**Acceptance (ship gate).** val NLL < 1.92 at production scale (single-seed 1337 primary run); 0
grad-skips through the previously-exploding regime; bit-identical flagship at init; wall ≤ +10% /
negligible VRAM; inverse-reconstruction error within BF16 ULP. **Multi-seed confirmation is capped
at 5k–15k steps per seed** (sign + cross-seed stability only, not a full 30k×3 trajectory — owner
cost-control decision).

---

## 15. One-line thesis

Make the per-layer `p→q` coupling a **bounded, mass-preconditioned, zero-initialized relativistic
drift** whose block-Jacobian off-diagonal norm is controlled by construction, so the product of
reversible block Jacobians stays under a spectral budget `B = Σ_l √(‖G_l‖‖A_l‖) = O(1)` — turning
the previously exponential (`√(g·a)`-driven) gradient blow-up back into the flagship's benign
linear growth, while letting attention finally compose across depth through `q`.
