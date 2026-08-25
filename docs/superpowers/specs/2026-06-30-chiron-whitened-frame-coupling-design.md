# CHIRON WhiSC — Whitened Symplectic Coupling

**Date:** 2026-06-30
**Status:** Design (approved for planning; pre-registration of the E3 falsifier below)
**Author:** research-framework-design (3-candidate parallel synthesis)
**Predecessors:**
- **SORC** (Symplectic Orthogonal Rotation Coupling) — **CLOSED NO-GO** (`research/CHIRON_SORC_RESULT_2026_06_30.md`): a hard-bounded per-channel rotation `(q,p)→R(θ)(q,p)`, `‖R‖₂=1`. Implemented correctly, angle bound provably held, E0/E1/E2 passed — but the E3 production gate **DIVERGED** at step ~831 (val 14.45 vs 3.58, ‖g‖→4.7e10); the backward `drot_phi` exploded ~2.5×/layer.
- **OBSD** (per-layer additive drift) — **CLOSED NO-GO** (`research/CHIRON_OBSD_RESULT_2026_06_27.md`): stable but −0.66 nat (unbounded ReZero gate → ‖g‖ inflation → grad-clip throttled LR).
**Measured root cause (this session, `chiron_init_pq_ratio_measured`):** the symplectic block's phase ratio `p²/q²` is **trained-in within ~100 steps** — init 0.096 → step100 171 → step600 ~338 → step1000 ~2000 and still rising (‖p‖/‖q‖: 0.31→13→~45). ReLN pins `mean(q²)≈1`; `p` accumulates attention un-normalized across depth. Any joint-norm-conserving coupling perturbs `q` by a **p-scaled** amount; conserving the (p-dominated) joint norm does **not** protect the q-subspace. **There is no early window where p≈q — a whitening lever must engage from step 1.**
**Target:** lower val NLL at production scale via cross-depth attention composition, with a coupling whose **backward q-subspace gain is bounded independent of ‖p‖/‖q‖** (the requirement SORC failed), preserving exact reversibility / O(1) activation memory.

---

## 1. Executive summary

The SORC post-mortem isolated the exact lesson: **forward norm-preservation ≠ bounded backward gradient under scale asymmetry; joint-norm conservation is the wrong invariant; the right one is per-subspace scale control.** WhiSC operationalizes that lesson with one structural move:

> **Conjugate a hard-bounded symplectic coupling by a *detached, per-channel whitening* of the `(q,p)` pair, kept entirely out of the autodiff graph, computed every step from step 1.**

Symbolically the per-layer coupling (inserted at the SORC slot — after the attention shear `p += Y_l(q)`, before ReLN) is

```
  Φ_l  =  W_l⁻¹ ∘ G(φ_l) ∘ W_l ,   G ∈ SO(2) ⊂ Sp(2,ℝ),   W_l = per-channel whitening (DETACHED).
```

- `W_l` moves the entire `√(p²/q²)` scale gap into a **detached change of frame**. Because it is detached, the autodiff graph never differentiates it, so no `∂W/∂p` term can re-inject a p-scaled adjoint (the SORC failure).
- In the whitened frame the `q` and `p` subspaces share scale, so a bounded `G` perturbs `q` by `O(‖q‖)` — in **both** forward and backward.
- `det Φ_l = det G = 1` ⇒ exactly symplectic (in 2D `Sp(2)=SL(2)`), so reversibility is structural, not bolted on.
- `φ_l = 0 ⇒ G = I ⇒ Φ_l = I` **bit-exactly**, independent of `W_l` ⇒ E0 parity is free; the mechanism is default-off and gated.

**The mechanism is SORC's kernel with the scale gap factored out of the coupling and into a detached frame.** SORC already passed E0/E1/E2 and has shipped (default-off) kernels (`chiron_rot_*`), a checkpoint bit (1024), and a plateau monitor — so WhiSC reuses that infrastructure with a small, well-scoped delta.

**Three variants, one family** (selection in §3):

| Variant | Whitening `W_l` | DOF/channel | Role |
|---|---|---|---|
| **WhiSC-D** (diagonal) | symplectic scaling `diag(1/a,a)`, `a²=σ^q/σ^p` | 1 | **minimal prototype** — folds into the SORC kernel for free |
| **WhiSC-M** (Mahalanobis) | `Σ̂^{-1/2}` (full 2×2, captures the q–p correlation) | 3 | expressive upgrade if D passes the gate but plateaus |
| **WhiSC-T** (thermostat) | feedback gain / reversible squeeze under PI control | + controller | active R-regulation; the control-theoretic theory + ambitious version |

**Falsifiable claim:** WhiSC-D run on the matched single-seed 2500-step T=16384 gate that SORC failed will **not diverge** (‖g‖ ~ O(1), 0–few grad-skips, val@2500 ≈ baseline 3.58). The cheap E3 gate (~3.4 GPU-hr) is the decisive test.

---

## 2. Candidate formulations (3-candidate parallel synthesis)

Three independent theory agents developed materially different formulations from the identical problem statement.

### 2A — CPSS / Canonical Per-channel Symplectic Scaling (symplectic/Hamiltonian view)
Per-channel scalar `a_i>0`, map `S_a = diag(1/a_i, a_i)` (det 1 ⇒ symplectic for **any** `a_i` ⇒ reversibility automatic). Balance `E[q̃²]=E[p̃²]` forces `a_i² = σ^q_i/σ^p_i` (geometric-mean balance — the *optimum within the symplectic class*; full unit-variance whitening has det≠1 and is not symplectic). Conjugate a hard-bounded `SO(2)` rotation. Closed form `M_i = (cos θ, −ρ⁻¹ sin θ; ρ sin θ, cos θ)`, `det=1`, which factors **exactly** into the SORC 3-shear kernel with two folded coefficients `A_i = −ρ⁻¹ tan(θ/2)`, `C_i = ρ sin θ` (`ρ = σ^p/σ^q`). R1: forward q-kick `≤ θ_max·σ^q`; the angle gradient is the **whitened symplectic-area form** `q̃'p̄̃' − p̃'q̄̃' = O(1)`, independent of ρ. (Caught a sign/inversion trap: `a²=σ^q/σ^p`, *not* `(σ^p/σ^q)^½`, which would anti-balance.)

### 2B — MWC / Mahalanobis Whitening Coupling (information-geometric/operator view)
Per-channel `Σ̂_i = E[z_i z_iᵀ] + εI`, `z_i=(q_i,p_i)ᵀ` — captures the q–p **correlation** `B_i=E[q_i p_i]` that scalar scaling misses. Whiten `z̃=Σ̂^{-1/2}z` (isotropic frame), couple, unwhiten. Closed-form 2×2 root via Cayley–Hamilton: with `τ=A+C`, `d=√(AC−B²)`, `s=√(τ+2d)`, `Σ̂^{1/2}=(A+d,B;B,C+d)/s`, `Σ̂^{-1/2}=(C+d,−B;−B,A+d)/(d·s)`. `det M = det R = 1` ⇒ symplectic (2D). R1: metric-norm exactly 1; raw Euclidean adjoint `≤ √κ`, capped at `√(2/ρ_floor+2)` by a **trace-relative floor** `ε=½ρ_floor·tr(Σ)`; **non-compounding across depth** (the conjugation telescopes: `∏ Σ̂^{1/2}R Σ̂^{-1/2} ≈ Σ̂^{1/2}(∏R)Σ̂^{-1/2}`, a single √κ, not `(√κ)^L`).

### 2C — PRT / Phase-Ratio Thermostat (control-theoretic/dynamical-systems view)
Reframe depth `l` as time and `R_l = mean(p²)/mean(q²)` as a regulated temperature. Two regimes: (κ=0) a **feedforward** gain `g=√(R*/r̄)` whitens only the coupling's *read* of `p` (state `p`-flow untouched) — the coupling is a `shear^q` `q += a·φ(g·p̂)`, so the **q→q adjoint is exactly identity**; (κ>0) a reversible **symplectic squeeze** `(q,p)→(e^ξ q, e^{−ξ}p)` under PI feedback actively regulates `R→R*`. Central insight: SORC is an **open-loop** coupling on an **unstable plant** — the open-loop depth-pole is `λ_p = 1+φ'(y*) ≈ 2.5`, and `λ_p^{24} ≈ 1.4e9` **is** the observed `‖g‖→4.7e10` / 2.5×-per-layer `drot_phi` cascade. Feedback moves the pole inside the unit disk (Jury/Routh–Hurwitz condition `κk_P > φ'`). The controller is **detached** (out of the loss) so the unstable plant is never in the autodiff graph. Reframes ReLN as the **q-subspace controller**; WhiSC adds the **p-subspace controller**.

---

## 3. Framework selection rationale

**A and B are the same mechanism** (conjugate a bounded `SO(2)` by a detached per-channel whitening) at different whitening expressivity: B's 3-DOF `Σ̂^{-1/2}` reduces to A's 1-DOF symplectic scaling when the correlation `B_i=0` *and* one restricts to the det-1 family. **C is genuinely orthogonal**: it supplies (i) the *control-theoretic explanation* that unifies the family and pinpoints SORC's failure as an open-loop unstable pole, and (ii) an *active* extension that regulates `R` rather than only whitening the coupling's read.

**Selection:**
- **Primary / minimal prototype = WhiSC-D (CPSS).** Decisive advantages: (1) it reuses the **exact** SORC kernel (`chiron_rot_forward/inverse/backward`) — the whitening is two pre-multiplied per-channel coefficients, **zero new GEMMs / passes**; SORC's E0/E1/E2 already pass, so only the whitening stats + the whitened-frame dθ accumulation are new. (2) The gate is **hard-bounded** (`θ=θ_max·tanh φ`, `|θ|≤θ_max`) — even under whitening lag or `φ→∞` the q-perturbation is `≤θ_max·σ^q`; this is strictly safer than an uncapped additive gate (OBSD's failure mode). (3) Fully symplectic rotation ⇒ structural reversibility (R5/R3). (4) Cleanest R1 (whitened isometry, `cond_W=1`).
- **Expressive upgrade = WhiSC-M (MWC).** If D passes the gate but does not improve perplexity, the missing degree of freedom is the q–p correlation `B_i`; M adds it at the cost of a 2×2 root + a 3-entry stat. M is the principled "diagonal wasn't enough" step, not a different mechanism.
- **Theory + ambitious extension = WhiSC-T (PRT).** Adopt C's control-theoretic analysis as the framework's *stability theory* (it explains why D/M are gate-safe and predicts the pole). D/M are **passive** (they whiten the coupling's read but do not stop the real network's `p` from growing — `R` still rises); T is the **active** version that additionally damps `p` to hold `R≈R*`. T is the long-horizon research vehicle, deferred behind D's gate result.

**Why the rejected-as-primary options are weaker for the *first* experiment.** B's 2×2 root and correlation estimate add implementation surface and a conditioning knob (`ρ_floor`) before we know whether *any* whitened coupling helps — premature. C's κ=0 feedforward shear has an even cleaner q-adjoint (exact identity) but an **uncapped** gate `a` (OBSD-reminiscent yellow flag) and, in its κ>0 form, modifies the `p`-flow and needs `ξ` store/restore for reversibility — more moving parts. D is the smallest, hardest-bounded, most-reused first step; it cleanly tests the core hypothesis (does *whitening the frame* fix the backward?) and the cheap E3 gate adjudicates it.

The remainder fully develops **WhiSC-D** as the shipped prototype, with M and T specified as the upgrade path, and the unified theory (including C's pole analysis) as the stability argument.

---

## 4. Formal problem statement

Per token `t∈{1..T}` the phase state is `(q_t, p_t) ∈ ℝ^m × ℝ^m` (`m=2048`, `L=24`, `T=16384`), with per-channel symplectic form `ω = Σ_i dq_i ∧ dp_i`. Grounded layer-`l` forward (`transformer_chiron_ops.h`, `chiron_main.cpp`):

```
 (shear)    p ← p + Y_l(q),   Y_l(q) = W_o·Attn(W_q q, W_k q, W_v q)   [q unchanged]   — symplectic shear^p
 (slot)     (q,p) ← C_l(q,p)                                            [OBSD/SORC/WhiSC here]
 (ReLN)     q_i ← γ_i·(q_i−μ)/σ + β_i,  μ,σ per token,  (μ,σ) stored    [reversible; q-only]
```
`p_0=0`, `q_0=embed`; the final layer folds `q ← q + p` before `logits = qEᵀ`. The block is reversible (backward inverse-walk: `ReLN⁻¹`, then `C_l⁻¹`, then `p −= Y_l(q)`; no per-layer activations stored).

Define per-channel second moments (detached) `σ^q_i² = E_t[q_{t,i}²]`, `σ^p_i² = E_t[p_{t,i}²]` (post-shear), ratio `ρ_i = σ^p_i/σ^q_i ≥ 1`. Measured: `ρ` rises 0.31→13→~45 over the first ~1000 steps; ReLN pins `σ^q_i ≈ √(γ_i²+β_i²) = O(1)`.

**Requirements.**
- **R1 (the target).** The backward adjoint of `C_l` has q-subspace operator gain bounded **independent of `ρ`** (and of depth/step). SORC's forward `‖R‖₂=1` did not imply this; its backward grew `2.5×/layer`.
- **R2.** Whiten from step 1 (no warmup — no early window with `ρ≈1`); exact identity when off (`φ=0 ⇒ C_l=I` bit-exactly; E0 parity).
- **R3.** Reversible with a cheap closed-form inverse (frozen-per-step stats ⇒ bit-exact inverse-walk).
- **R4.** Per-channel `O(T·m)` statistics only (no `m×m` covariance for the prototype).
- **R5.** Symplectic-friendly (preserve `ω` so reversibility is structural).
- **R6.** Statistics detached, online (EMA across steps + per-batch over `T` samples), valid at step 1.

**Eval bar.** (a) *Hard*: matched single-seed 2500-step T=16384 gate does **not** diverge (val@2500 ≈ baseline 3.58, ‖g‖ ~ O(1), 0–few grad-skips) — the bar SORC failed. (b) *Soft*: improves val NLL by letting attention compose across depth. (c) Default-off, gated, checkpoint-bit, E0 bit-parity when off.

---

## 5. Core framework — the whitened-frame symplectic coupling

### 5.1 General form
Per channel `i`, with a **detached** whitening `W_i ∈ GL(2)` and a hard-bounded `G(φ_i)=R(θ_i)∈SO(2)`, `θ_i = s_warm·θ_max·tanh φ_i` (`|θ_i|≤θ_max`):

```
  M_i = W_i⁻¹ R(θ_i) W_i ,    (q_i', p_i')ᵀ = M_i (q_i, p_i)ᵀ ,   broadcast over tokens t.
```
`det M_i = det R = 1` for any `W_i` ⇒ each 2×2 block preserves `dq_i∧dp_i` ⇒ the global block-diagonal map preserves `ω` **exactly** (R5). `φ_i=0 ⇒ R=I ⇒ M_i=I` bit-exactly, regardless of `W_i` (R2).

### 5.2 WhiSC-D instantiation (the prototype)
`W_i = S_{a_i} = diag(1/a_i, a_i)`, `a_i² = σ^q_i/σ^p_i = ρ_i⁻¹` (geometric-mean balance; `det S=1`). Closed form:

```
  M_i = ( cos θ_i        −ρ_i⁻¹ sin θ_i )      forward state update:
        ( ρ_i sin θ_i     cos θ_i       )       q_i' = cos θ_i·q_i − ρ_i⁻¹ sin θ_i·p_i
                                                 p_i' = ρ_i sin θ_i·q_i + cos θ_i·p_i
```
**Kernel reuse — the whitening is free.** `M_i` factors exactly into the SORC 3-shear pattern `q += A_i p; p += C_i q; q += A_i p` (`chiron_rot_forward`) with **whitening-folded coefficients**:

```
  A_i = −a_i²·tan(θ_i/2) = −ρ_i⁻¹·tan(θ_i/2),     C_i = a_i⁻²·sin θ_i = ρ_i·sin θ_i.
```
(SORC is the special case `a≡1`: `A=−tan(θ/2)`, `C=sin θ`.) Inverse = same kernel with `θ→−θ` (`q −= A_i p; p −= C_i q; q −= A_i p`), bit-exact when `a_i` is frozen across the step (R3).

### 5.3 WhiSC-M instantiation (upgrade)
`W_i = Σ̂_i^{-1/2}` (closed-form 2×2 root, §2B), `M_i = Σ̂_i^{1/2} R(θ_i) Σ̂_i^{-1/2}`, still `det=1`. Same 3-shear realization with the now-general `M_i` (the SORC factorization `M=U(x)Lo(y)U(x')`, `y=M₂₁, x=(M₁₁−1)/M₂₁, x'=(M₂₂−1)/M₂₁` handles any `det=1` `M`). Captures the q–p correlation; needs the 3-entry stat `(E[q²],E[qp],E[p²])` + a trace-relative floor `ε=½ρ_floor·tr(Σ̂)`.

### 5.4 Control-theoretic reading (the unifying theory, from C)
ReLN is the **q-subspace scale controller** (pins `σ^q≈1` every layer). WhiSC adds **p-subspace scale control**. The whitening is exactly the act of **keeping the unstable `R`-plant out of the autodiff graph**: the differentiable coupling reads only whitened, `O(‖q‖)` quantities; the `ρ`-scale lives in the detached `W`. Passive (D/M) whitens the coupling's *read*; active (T) additionally regulates the *plant* `R_l→R*`.

---

## 6. The R1 derivation (backward-bounded, independent of ρ)

Stats detached ⇒ `M_i` is a constant linear map in the state, smooth in the scalar `φ_i`. Backprop is exact:
```
  state adjoint:   (q̄_i, p̄_i)ᵀ = M_iᵀ (q̄_i', p̄_i')ᵀ
  param  grad:     ∂L/∂θ_i = Σ_t [ (∂M_i/∂θ) (q_i,p_i)ᵀ ]·(q̄_i',p̄_i')ᵀ ,   ∂L/∂φ_i = ∂L/∂θ_i·s_warm θ_max sech²φ_i
```

**(a) Whitened-metric isometry — exact, ρ-independent.** Define `‖x‖_W = ‖W_i x‖₂` (measuring `q` in units of `σ^q`, `p` in units of `σ^p`). Then `W_i M_i W_i⁻¹ = R(θ_i)` and `W_i⁻ᵀ M_iᵀ W_iᵀ = R(θ_i)ᵀ`, both orthogonal ⇒
```
  ‖M_i‖_W = ‖M_iᵀ‖_{W*} = ‖R‖₂ = 1   for all θ_i, ρ_i, depth, step.
```
The forward and backward are **isometries in the per-subspace scale metric**, exactly. The q-perturbation in q's own scale is `‖δq‖_W ≤ 2|sin(θ/2)| ≤ θ_max = O(‖q‖)` — never `O(‖p‖)`. (SORC had `M=R` in the **raw** frame: `‖R‖₂=1` but the q-perturbation in q-units was `|a|·ρ ≈ 17|a|`, and nothing re-equalized it per layer — the cascade.)

**(b) Bounded raw-Euclidean adjoint (the buffers autodiff uses).** `‖M_iᵀ‖₂ ≤ ‖W⁻¹‖₂‖W‖₂ = √κ(W_iᵀW_i)`. For WhiSC-D, `κ = ρ_i²` (so the raw `p̄→q̄` entry `ρ_i sin θ_i` is large) — **but it does not compound across depth**: the chain telescopes, `∏_l M_l = ∏_l W_l⁻¹R_l W_l ≈ W⁻¹(∏_l R_l)W` for slowly-varying `W`, a **single** `√κ` factor, not `(√κ)^L`. For WhiSC-M the trace-relative floor caps `κ ≤ 2/ρ_floor+2` (a constant, e.g. `√10` at `ρ_floor=0.25`). The decisive quantity is not the raw entry but the parameter gradient:

**(c) Parameter gradient = whitened symplectic area = O(1) (the quantity that exploded in SORC).** Using `∂_θ R = RJ`, in whitened coordinates `z̃=W_i z`, `z̄̃=W_i⁻ᵀ z̄`:
```
  ∂L/∂θ_i = Σ_t ( q̃_i'·p̄̃_i' − p̃_i'·q̄̃_i' )          (whitened symplectic area form)
          ≤ Σ_t ‖z̃_i'‖·‖z̄̃_i'‖ = O(1)              [ E_t‖z̃‖² = tr(W ΣW)= O(1) by construction (R6) ]
```
independent of `ρ_i`, by Cauchy–Schwarz, with **no balance assumption on the adjoint**. SORC's identical bound read `Σ_t ‖z‖·‖z̄‖` with `‖z‖≈‖p‖` growing 0.31→13→45 and compounding ≈2.5×/layer — exactly `drot_phi`. WhiSC replaces the unbounded `‖p‖` by the unit-scale `‖z̃‖`.

> **Load-bearing implementation note (R1-critical).** Accumulate `∂L/∂θ_i` **in the whitened frame** via the symplectic-area form `q̃'p̄̃' − p̃'q̄̃'` (equivalently keep the intermediates whitened), **not** through the raw per-coefficient grads `∂L/∂A_i, ∂L/∂C_i`, which individually carry `a^{∓2}=ρ^{±1}` factors that cancel only at the very end — a naive port of `chiron_rot_backward` would reintroduce a `ρ`-amplified partial sum and lose precision. The math is identical; the conditioning of the intermediate is not.

**(d) Why this is the genuine fix.** SORC's joint isometry conserved the p-dominated joint norm and left the q-subspace an expander. WhiSC moves the whole `ρ` into a **detached** frame, so the metric-norm of the coupling (forward and adjoint) is exactly 1, the raw factor is a single non-compounding `√κ` (D) or a floored constant (M), and the parameter gradient is `O(1)`. Forward norm-preservation was never the problem; **per-subspace metric control of the perturbation, with the statistic out of the graph, is.**

**(e) Control-theoretic corollary (C).** The open-loop depth-pole `λ_p=1+φ'(y*)≈2.5` (`λ_p^{24}≈1.4e9 ≈ ‖g‖`) is the same number as `drot_phi`'s per-layer growth. WhiSC-D/M keep this plant **out of the autodiff graph** (detached `W`), so the differentiated path no longer rides `λ_p`. WhiSC-T additionally moves the *forward* pole inside the unit disk by feedback (`κk_P>φ'`).

---

## 7. Optimization algorithm — WhiSC-D (minimal prototype)

**Per-step statistics (detached, R6).** Once per step, after the shear at each layer (reuse the existing per-token `(q,p)` sweep that already feeds SIRA/reanchor):
```
  P_i = (1/T)Σ_t (p_{t,i}^+)²,  Q_i = (1/T)Σ_t q_{t,i}²            [two O(T·m) reductions/layer]
  EMA:  P̄_i ← (1−η)P̄_i + η P_i,   Q̄_i ← (1−η)Q̄_i + η Q_i          [η≈0.05–0.1; init from step-1 batch]
  a_i² = clamp( Q̄_i / (P̄_i+ε_p) , [a_min², a_max²] )               [= σ^q/σ^p; clamp handles init p→0]
```
All under `stop_gradient`. Compute `a_i` **once per step**, cache, reuse in forward / backward / inverse-walk (R3 exactness).

**Forward (per layer, after shear, before ReLN):**
```
  θ_i = s_warm·θ_max·tanh φ_i;   A_i = −a_i²·tan(θ_i/2);   C_i = a_i⁻²·sin θ_i
  q += A_i⊙p;   p += C_i⊙q;   q += A_i⊙p                          [= chiron_rot_forward, folded coeffs]
```

**Backward (inverse-walk + grads):**
```
  inverse-walk:  q −= A_i⊙p;  p −= C_i⊙q;  q −= A_i⊙p             [recover pre-coupling (q,p)]
  state adjoint via M_iᵀ (the 3-shear adjoint);  accumulate ∂L/∂φ_i from the WHITENED area form (§6 note)
```

**Parameters / gating.** New trainable `φ ∈ ℝ^{L×m}` (init 0; Adam, like `rot_phi`); detached EMA buffers `P̄,Q̄ ∈ ℝ^{L×m}` (FP32, ~0.4 MB). Checkpoint bit (reuse SORC's slot family). Flags: `--whisc-coupling` (default off), `--whisc-theta-max` (default 0.07), `--whisc-ema 0.05`, `--whisc-floor` (`a²` clamp range), `--whisc-warmup` (optional `s_warm` ramp; free insurance, not required). `φ=0 ⇒ A=C=0 ⇒` exact identity (E0).

**Complexity.** 2 reductions + a folded-coefficient compute per layer per step, reusing the SORC token-loop kernel. Wall ≈ SORC (which was ~parity). No new GEMMs, no m×m anything.

---

## 8. Temporal / depth dynamics

Let `y_l = log R_l`. Open loop (no coupling, just shear+ReLN): `y_{l+1} = y_l + φ(y_l)`, pole `λ_p = 1+φ'(y*) ≈ 2.5` (empirically), unstable. **Passive WhiSC (D/M)** does **not** change this plant — `R` still grows across depth and over training; D/M only guarantee the *coupling and its gradient* are whitened (R1), so the network is free to keep its large `p` while the cross-depth lever stays bounded. **Active WhiSC-T** inserts a reversible symplectic squeeze `(q,p)→(e^ξ q, e^{−ξ}p)` (Δy = −4ξ) under detached PI control `ξ = ¼κ(k_P(y−y*)+k_I ζ)`, giving the closed-loop reduced map
```
  [e_{l+1}; z_{l+1}] = J [e_l; z_l],  J = [α −β; 1 1],  α=1+φ'−κk_P,  β=κk_I,
```
attracting (Jury) iff `κk_P > φ'` and `k_I>0` small — moving the pole inside the unit disk. The continuous-depth form is a Nosé–Hoover thermostat whose temperature is the log phase-ratio. T is deferred behind D's gate result (§14).

---

## 9. Theoretical analysis

- **Well-posedness / reversibility (R3, R5).** Each `M_i` is `det=1` (symplectic); `Φ_l = ∏_t M_i` preserves `ω`. Inverse is `θ→−θ` (D) or `M⁻¹=W⁻¹R⁻¹W` (M), bit-exact under frozen-per-step stats. The 3-shear realization is exactly invertible by negating shear params (machine-exact `det=1` even in FP32 if the shear factorization is used).
- **Conditioning.** `cond_W(M_i)=1` (perfect, all ρ,θ). Raw `cond₂(M_i)=ρ²` (D) is the expected coordinate distortion the whitening factors out of the *gradient*; it does **not** compound across depth (telescoping, §6b). M's floor caps it at `2/ρ_floor+2`.
- **Stability.** R1 (§6) bounds the parameter gradient at `O(1)` and the adjoint at metric-norm 1; inserted into the reanchor baseline (which already trains to 4.33B at ‖g‖~1), an isometry cannot create a multiplicative cascade in the metric the rest of the block is conditioned in. C's pole analysis (§8) predicts gate-passing.
- **Init `p→0` singularity (R2, 2nd order).** At init `σ^p→0 ⇒ a²→∞`; the *products* stay bounded (`δq ~ θσ^q→` finite, `δp ~ θσ^p→0`), but the float `a²` is huge×tiny → catastrophic cancellation; the clamp `a²∈[a_min²,a_max²]` removes it. The regime is also benign because `φ` ramps from 0 (block ≈ identity exactly when `σ^p` is smallest). With the gate off, `M=I` bit-exactly independent of `a`.
- **Expressivity.** Per layer, `m` learnable bounded canonical rotations (D) / rotations-in-the-isotropic-frame (M) of each `(q,p)` pair; across `L` they compose to a learnable per-channel symplectic transport that lets attention compose across depth (the flagship folds `p→q` only at the last layer). New params `m·L = 49,152` (D), negligible. Deliberate R4 boundary: per-channel only (no cross-channel mixing).

---

## 10. Computational tradeoffs

| | WhiSC-D | WhiSC-M | WhiSC-T |
|---|---|---|---|
| Stats/layer | 2 reductions (P,Q) | 3 (P,Q,E[qp]) + 2×2 root | P,Q + controller `ζ` |
| Coupling kernel | **SORC, folded coeffs (free)** | SORC 3-shear, general M | + reversible squeeze (store `ξ`) |
| New state | `φ` (L×m), `P̄,Q̄` | + `E[qp]` EMA, floor | + `ζ`, `ξ` store/restore |
| Reversibility | frozen-`a`, bit-exact | frozen-Σ̂ | needs `ξ` store |
| Wall | ≈ SORC (≈ parity) | + small (2×2 root) | + squeeze passes |
| Risk | lowest | conditioning knob `ρ_floor` | grad-clip×`Σξ` interaction |

---

## 11. Comparison to existing CHIRON mechanisms

| | OBSD | SORC | **WhiSC (this)** | SIRA |
|---|---|---|---|---|
| Coupling | additive ReZero drift | raw-frame rotation | **whitened-frame rotation** | none (terminal regularizer) |
| Gate | unbounded | hard angle bound | hard angle bound | — |
| Frame | raw | raw | **detached per-channel whitening** | raw |
| Invariant | none | joint norm (p-dominated) | **per-subspace scale** | soft terminal balance |
| Backward q-bound | gate-inflated | `O(ρ)`, 2.5×/layer cascade | **`O(1)`, ρ-independent** | n/a |
| Acts | per layer | per layer | per layer | terminal `(q_L,p_L)` only |
| When | per step | per step | **per step, from step 1** | per step (loss term) |
| Outcome | stable, −0.66 nat | diverged step ~831 | **predicted gate-pass** | shipped (default-off) |

WhiSC = SORC's kernel + bound, with the scale gap factored into a detached frame. Relation to SIRA: SIRA softly penalizes terminal `½log(E[p²]/E[q²])` by backprop, globally; WhiSC mechanically whitens per-layer in the forward, from step 1, with no gradient path — complementary (SIRA shapes the learned terminal balance; WhiSC makes the in-block coupling safe). Relation to reanchor: orthogonal — reanchor fixes the ReLN-backward stat drift; WhiSC acts before ReLN and leaves `γ` untouched; reanchor's `q_in`-derived stats are downstream of WhiSC and unaffected.

---

## 12. Failure modes and mitigations

1. **EMA estimation lag (primary).** `a_i²` uses stale moments while `ρ` rises fast (13→45 over ~900 steps); under-whitening leaves residual `O(√(ρ_true/ρ_est)·‖q‖)` q-perturbation. *Mitigation:* fast EMA (`η≈0.05–0.1` — a detached scale, lag costs conditioning not correctness); floor the EMA with the current batch moment so a `σ^p` jump is partly tracked within-step; the hard `θ_max` caps the residual regardless. Monitor `max_l ρ_i` (reuse the `[sorc]` plateau monitor → `[whisc]`).
2. **Init `p→0` singularity.** Handled by the `a²` clamp + `φ` init 0 (block ≈ identity when `σ^p` smallest); bit-parity off is independent of `a` (§9).
3. **Reconstruction drift if `a` not frozen.** Forward/inverse must use identical `a` snapshot. *Mitigation:* compute `a_l` once per step, cache, reuse fwd/bwd/inverse.
4. **Naive backward reintroduces ρ.** The §6 note: accumulate `dθ` in the whitened frame, not via raw `dA,dC`. Unit test: dθ parity between whitened-area and a finite-difference reference at `ρ=45` (must match; a raw-coefficient port will not).
5. **ReLN-γ interaction.** `σ^q_i ≈ √(γ_i²+β_i²)` is measured empirically, so whitening tracks the learned affine. A suppressed channel (`γ_i→0 ⇒ σ^q→0 ⇒ a→0, a⁻²→∞` in `C_i`) is bounded by the same `a²` clamp + small `θ_max`. WhiSC unwhitens before ReLN, so `γ` never fights the whitening (and at `φ=0`, `q` is untouched).
6. **Diagonal misses correlation (D-specific).** If `p`'s energy lives off the per-channel axes / correlates with `q`, D under-whitens those modes → a *passing-but-not-improving* gate. *This is the trigger to upgrade to WhiSC-M* (§14).
7. **(M) Σ̂ conditioning / over-whitening.** Trace-relative floor caps `κ≤2/ρ_floor+2`; tune `ρ_floor` (start 0.25). **(T) grad-clip × `Σξ`.** Keep `ξ_max` so `e^{Lξ_max}=O(10)`; the κ=0 feedforward regime has q-adjoint factor exactly 1 (no global rescale, no grad-clip interaction) — the recommended first form of T.

---

## 13. Minimal prototype (implementable now) + the pre-registered gate

**Build = WhiSC-D, feedforward, hard-bounded, default-off.** Reuse `chiron_rot_forward/inverse/backward`; add (i) the 2 detached EMA reductions + `a_i²` per layer, (ii) the folded coefficients `A_i,C_i`, (iii) the whitened-frame `dθ` accumulation, (iv) `φ` param + checkpoint bit + flags (§7). E0 bit-parity at `φ=0`.

**Validation ladder (mirrors SORC's E0–E4):**
- **E0** — `φ=0` step-1 bit-identical to baseline (incl. SCFA path). *Must pass.*
- **E1** — unit tests: `M_iᵀM_i` symplectic (`det=1`); inverse-walk reconstruction at `ρ=45`; **dθ whitened-frame vs finite-difference parity at `ρ=45`** (the failure-mode-4 guard); `φ=0⇒` identity.
- **E2** — small-shape reconstruction + a sanity run showing `‖g‖` bounded with `φ≠0` at large synthetic `ρ`.
- **E3 (the decisive falsifier, ~3.4 GPU-hr)** — matched single-seed 2500-step T=16384, production reanchor recipe, `--whisc-coupling --whisc-theta-max 0.07`. **PASS = does not diverge**: val@2500 ≈ baseline 3.58 (not 14.45), ‖g‖ ~ O(1) (not 4.7e10), 0–few grad-skips, and the `[whisc]` monitor shows `dθ`-grad bounded across depth (no 2.5×/layer cascade). This is the bar SORC failed; R1 predicts a pass.
- **E4** — only if E3 passes: 30k decisive perplexity gate vs matched baseline (soft bar (b)).

**Pre-registered prediction.** WhiSC-D passes E3 (no divergence). If E3 passes but E4 shows no perplexity gain, the diagnosis is the missing correlation DOF → upgrade to WhiSC-M (do not abandon the family). If E3 *fails*, the whitening hypothesis itself is wrong (the asymmetry is not the whole story) — a genuine, cheap kill.

---

## 14. Full research program

1. **WhiSC-D E3/E4** (above) — the gate that adjudicates the whole family cheaply.
2. **WhiSC-M** — add the 2×2 Mahalanobis whitening (correlation DOF) if D passes-but-plateaus; closed-form root (§2B) + trace-relative floor; same kernel via the general `M=U Lo U` factorization.
3. **WhiSC-T** — the active thermostat: start with the κ=0 feedforward shear (exact-identity q-adjoint, no grad-clip interaction), then the κ>0 reversible-squeeze PI controller (pole placement §8); the vehicle for testing whether *actively regulating* `R≈R*` across all depth (not just whitening the read) further helps. Requires `ξ` store/restore.
4. **Cross-channel whitening** — relax R4: block / low-rank (not full `m×m`) whitening if per-channel proves insufficient (failure mode 6).
5. **Learnable per-channel setpoint `R*_i`** (T) on a slow timescale — let the model choose the phase balance it wants rather than forcing `R=1`.

---

## 15. Open conjectures and validation criteria

- **C1 (hard, falsifiable).** WhiSC-D does not diverge on the SORC E3 gate (‖g‖~O(1), val@2500≈3.58). *Falsifier:* E3 diverges ⇒ whitening the frame is insufficient ⇒ the asymmetry is not the operative cause and the cross-depth-coupling direction is closed for CHIRON.
- **C2 (soft).** A whitened cross-depth coupling lowers val NLL at 30k (E4) by letting attention compose across depth. *Falsifier:* E3-pass + E4 flat across D **and** M ⇒ cross-depth composition is not a useful lever (independent of the stability fix).
- **C3 (mechanism).** The `dθ` gradient norm is bounded and **non-compounding** across depth (no 2.5×/layer cascade) — directly observable in the `[whisc]` monitor at E3; this is the operational signature distinguishing WhiSC from SORC.
- **C4 (theory).** The active thermostat (T) moves the open-loop pole `λ_p≈2.5` inside the unit disk; testable by logging `R_l` vs depth with κ=0 (rising) vs κ>0 (flat at `R*`).

**Engineering discipline (per project convention):** all variants default-off, gated, checkpoint-bit, E0 bit-parity when off; reuse SORC's kernels/monitor/flags; commit reusable even on a NO-GO. The E3 gate is the cheap (~3.4 GPU-hr) decision point before any 30k/E4 spend.
