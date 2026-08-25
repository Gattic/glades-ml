# CHIRON SORC — Symplectic Orthogonal Rotation Coupling

**Date:** 2026-06-30
**Status:** Design (approved for planning + implementation)
**Author:** research-framework-design (3-candidate parallel synthesis)
**Predecessor:** OBSD per-layer drift — **CLOSED NO-GO** (`research/CHIRON_OBSD_RESULT_2026_06_27.md`):
helped early (+0.05 nat) then regressed −0.66 nat at 30k because a free ReZero gate grew unbounded
(maxA 0.46→1.8) → inflated ‖g‖ → grad-clip throttled the effective LR → missed the baseline's
late-training acceleration. Stable throughout (0 grad-skips); the failure was **optimization
dynamics**, not instability.
**Target:** lower val NLL at production scale via cross-depth attention composition whose coupling
strength is **bounded by construction** (so it cannot run away the way OBSD's gate did), preserving
exact reversibility / O(1) activation memory.

---

## 1. Executive summary

OBSD established that **cross-depth coupling has real value** (it helped early) and that the killer is
**not** the coupling itself but the *unboundedness of an additive gate*: OBSD does `q += a_l⊙φ(N(p))`
— it **injects magnitude** into `q`, and an additive shear can grow without bound, so the optimizer
grew it monotonically until it dominated the gradient budget.

SORC fixes this **categorically** by replacing **additive injection** with a **norm-preserving
exchange**. The coupling is a per-channel **symplectic rotation** of the `(q,p)` pair:

> per channel `i`, rotate `(q_{t,i}, p_{t,i})` by angle `θ_i`, realized as **three reversible shears**.

Because a rotation `R(θ) ∈ SO(2)` has `‖R‖₂ = 1` and conserves `q_i²+p_i²`, the coupling **cannot
inject magnitude into `q` for any angle** — the runaway is not *regulated*, it is **mathematically
impossible**. The angle is bounded by construction (`θ_i = θ_max·tanh(φ_i)`, `θ_max < π/2`), so the
*parameter* also cannot run away and its gradient self-attenuates at saturation. `φ=0 ⇒ θ=0 ⇒ R=I`
recovers the flagship bit-identically. The rotation is realized as three unit-triangular shears, each
reading only the opposite branch, so exact reversibility and O(1)-in-depth memory are preserved; it
uses **no GEMM, no reductions, no nonlinearity in the hot path** (3 per-channel axpy ops), making it
**cheaper than the OBSD drift** (+4–6% wall).

SORC is also the **clean instrument** to settle OBSD's one unresolved question (the failed late cap
couldn't): since SORC's coupling magnitude **provably plateaus**, if its val-NLL gap *still* reverses
late, that **falsifies "gate magnitude was the problem"** and implicates cross-depth composition
itself — a decisive negative the cap could never deliver.

---

## 2. Candidate formulations (provenance)

Three materially-different formulations of "intrinsically bounded depth coupling" were developed in
parallel:

- **A — GBC (manifold-constrained).** Inject accumulated momentum into `q` as a bounded-angle
  **secant on the reln shell**, magnitude a bounded fraction of `q`'s shell radius, angle on a
  compact `S¹`. Self-regulation: compact parameter manifold (radial gradient **exactly 0**) +
  tangential gradient **self-attenuates at the cap**. Key contribution adopted here: the
  **bounded-angle parameterization** that makes the *parameter side* runaway-proof. Weakness: the
  effect is an *approximation* of a rotation (chord-vs-arc, +√2 scale leak re-projected each layer).

- **B — HBSD (homeostatic control).** Keep OBSD's drift verbatim; regulate the coupling-energy
  fraction `c_l` to a setpoint `ρ*` via an augmented-Lagrangian primal–dual loop in the optimizer
  step. Sharp result: bare dual ascent is only *marginally* stable in OBSD's non-diminishing-returns
  regime (`f_L'≈0` → undamped orbits); the **augmented penalty `μ > f_L'`** supplies the missing
  damping. Adopted here as *methodology* (the "decisive early-intervention test" framing, the
  calibration-from-E4-trajectory). Rejected as backbone: introduces tuned setpoint `ρ*` + control
  hyperparameters (`η_λ, μ`) the brief disfavors, keeps OBSD's +12–19% wall, and its boundedness is
  *dynamic* (relies on the controller converging) rather than structural.

- **C — SORC (symplectic rotation).** Replace additive injection with a norm-preserving rotation.
  Boundedness by **conservation law** (`‖R‖₂=1`). Selected as backbone (§3).

---

## 3. Framework selection rationale

**Selected: SORC (C) as backbone, grafting A's bounded-angle parameterization and B's methodology.**

1. **Deepest diagnosis of OBSD's root cause.** OBSD failed because it *adds* `g(p)` to `q` —
   injecting magnitude that can grow unbounded. SORC's rotation **structurally cannot inject
   magnitude** (conservation), so the runaway is *impossible*, not *regulated*. This attacks the
   exact mechanism that killed OBSD.
2. **Structural over dynamic boundedness, no tuned setpoint.** The brief explicitly disfavors a tuned
   operating point. B regulates to a setpoint `ρ*` (tuned, dynamic); A and C bound by geometry
   (structural, no setpoint). C's conservation bound is the cleanest ("‖R‖=1 for any angle").
3. **A's parameter-side analysis is the missing half of C.** C's angle, left free in ℝ, can still
   wander (benignly — periodic effect — but unclean, and the 3-shear `tan(θ/2)` needs `|θ|<π/2`
   anyway). Grafting A's `θ = θ_max·tanh(φ)` bounds the angle *and* gives the self-attenuating
   gradient — so SORC is bounded on **both** the effect (conservation) and the parameter
   (saturation).
4. **Cheapest + exact.** C's effect is an *exact* rotation (vs A's secant approximation with √2 scale
   leak), and it's the cheapest (3 plain axpy-shears, no reductions/tanh in the hot path).
5. **Most architecture-native.** A norm-preserving exchange between the `q` and `p` channels is the
   genuine symplectic completion of the block — OBSD did the `q`-shear but omitted the
   energy-conserving back-reaction on `p`; SORC adds exactly that.

B's virtues (verbatim-OBSD forward, lowest kernel risk, the decisive-test framing) are real but
outweighed; its decisive-test framing is preserved as the experimental methodology (§13).

---

## 4. Formal problem statement

- **System.** CHIRON reversible symplectic-flow transformer (L=24, m=2048, T=16384, nH=16, dH=256,
  V=32000). Per-layer state `(q,p)`, each ∈ ℝ^{T×m}. Block map unit-triangular & exactly invertible
  (q-update reads only p; p-update only q) → O(1)-in-depth activation memory. Reversibility
  non-negotiable.
- **Current block.** `p += Y_l(q)` (SCFA attention, function of q only); `q = reln(q; γ_l, β_l)`.
  Momentum folds into q **once** (final layer) → interior layers attend over attention-blind q → no
  cross-depth composition (the perplexity ceiling).
- **Objective.** Lower val next-token NLL at T=16384 via cross-depth attention composition that
  **holds its advantage through full training** (the gap must NOT reverse in the late regime —
  OBSD's specific failure), without regressing the flagship or sacrificing exact reversibility / O(1)
  memory.
- **Mandatory mechanisms.** (1) per-layer reversible p→q coupling; (2) **intrinsic
  boundedness/self-regulation by construction** (here: conservation-law on the effect + bounded-angle
  saturation on the parameter), NOT a tuned cap/setpoint, NOT a free monotonic gate; (3) exact
  reversibility + O(1) memory; (4) flagship recovery at init; (5) cheap (no new GEMMs); (6)
  symplectic-native.
- **Forbidden.** Re-capping the OBSD gate; any unbounded monotonic coupling; breaking reversibility;
  new attention passes/GEMMs; tuned setpoint/clamp; relying on integrator order.
- **Evaluation.** val NLL < matched baseline @30k with a **non-shrinking** gap through 24–30k; 0
  grad-skips; coupling magnitude **plateaus** (not monotone); reversibility within BF16 ULP; flagship
  recovery at init; wall ≤ +10%.

---

## 5. Core mathematical framework

### 5.1 Symbols
| Symbol | Space | Meaning | Init |
|---|---|---|---|
| `q, p` | ℝ^{T×m} | position / momentum half-state | — |
| `Y_l(·)` | — | SCFA attention (kick); reads q only | existing |
| `reln(·;γ_l,β_l)` | — | per-layer reversible LayerNorm of q | existing |
| `φ_l` | ℝ^m | **per-channel angle parameter** (the only new learned coupling param) | **0** |
| `θ_l` | ℝ^m | rotation angle `θ_{l,i} = θ_max·tanh(φ_{l,i})`, `|θ|<θ_max<π/2` | 0 |
| `θ_max` | scalar | angle cap (e.g. π/3 = 60°), fixed geometric constant | const |
| `a_{l,i}` | ℝ | shear-1/3 coefficient `= −tan(θ_{l,i}/2)` | 0 |
| `c_{l,i}` | ℝ | shear-2 coefficient `= sin(θ_{l,i})` | 0 |
| `s_warm` | [0,1] | optional fixed warmup ramp (reuses `--drift-warmup`); not a regulator | ramp |

### 5.2 The per-channel symplectic rotation as 3 shears [THEOREM — exact]
For a 2-vector `(q_i, p_i)`, the rotation `R(θ) = [[cos θ, −sin θ],[sin θ, cos θ]]` factorizes exactly:
```
R(θ) = S_u(a) · S_l(c) · S_u(a),   a = −tan(θ/2),  c = sin θ,
   S_u(a) = [[1, a],[0, 1]]  (Shear^q: q += a·p, reads p),
   S_l(c) = [[1, 0],[c, 1]]  (Shear^p: p += c·q, reads q).
```
*Verification:* `1+ac = cos θ`; `a(2+ac) = −sin θ`; `(2,1)=c=sin θ`; `(2,2)=1+ca=cos θ`. ∎
(Numerically confirmed orthogonal/`det=1` to ~1e-16; `dR/dθ = R(θ+π/2)`.)

### 5.3 Forward block (l < L−1), order kick → rotate → reln
```
(K)   p̃ = p + Y_l(q)                                   # p reads q only           (unchanged)
(Rot) per channel i (broadcast over t), with the EFFECTIVE angle
        θ_eff_{l,i} = s_warm · θ_max · tanh(φ_{l,i}),   a_i = −tan(θ_eff_{l,i}/2),  c_i = sin θ_eff_{l,i}:
        q ← q + a ⊙ p̃             # Shear^q  (reads p̃)
        p̃ ← p̃ + c ⊙ q             # Shear^p  (reads q)
        q ← q + a ⊙ p̃             # Shear^q  (reads p̃)
(N)   q' = reln(q; γ_l, β_l),   p' = p̃                 # q-only reln              (unchanged)
```
Rotate **before** reln so reln re-supplies scale (resolves "norm-preservation too restrictive"; keeps
reln-reanchor intact — `q_in` at the reln is the rotated q, and reanchor is provenance-agnostic). At
`θ_eff=0`: `a=c=0`, Rot is the identity → block = flagship. **The warmup ramps the ANGLE, not the
coefficients** — `s_warm` enters inside `θ_eff = s_warm·θ_max·tanh(φ)`, and `a,c` are recomputed from
`θ_eff`, so the step is an **exact rotation by `θ_eff`** at every point of the ramp (norm-preserving
throughout). Scaling `a,c` directly would NOT be a rotation. (Warmup is optional — `φ=0` is already
identity with full sensitivity, so the angle ramps in naturally as `φ` learns; `s_warm≡1` is the
default.) Final layer keeps the existing fold `q += p̃`.

### 5.4 Inverse block (exact, stores nothing) [THEOREM]
Reverse the 3 shears with negated coefficients, then reln-inverse, then kick-inverse:
```
q  ← q' (reln⁻¹ first):  q = reln⁻¹(q'; γ_l, β_l)
p̃ = p'
(Rot⁻¹), same a,c from θ_eff as the forward of this step:  q ← q − a⊙p̃ ;  p̃ ← p̃ − c⊙q ;  q ← q − a⊙p̃
p  = p̃ − Y_l(q)
```
Each shear reads only the opposite branch and is a pure additive shear → BF16-ULP exact (like the
OBSD drift's `sign=±1`); the inverse uses the identical `a,c` (recomputed from the same `θ_eff`), and
the inverse walk reconstructs activations with nothing stored → O(1) memory.

### 5.5 Backward (adjoint)
Each shear's adjoint is the same per-channel axpy with the transposed (sign-appropriate) coefficient;
the angle gradient `dφ_{l,i}` is a column reduction `Σ_t (·)` over T (reuse the deterministic
`chiron_col_accumulate`), chain-ruled through `a=−tan(θ/2)`, `c=sin θ`, `θ=θ_max·tanh(φ)`:
`dθ/dφ = θ_max·sech²(φ)`, `da/dθ = −½sec²(θ/2)`, `dc/dθ = cos θ`. All bounded.

---

## 6. Objective and flagship recovery

**Objective** unchanged (next-token NLL + existing aux stack: Z-loss, terminal SIRA, dq-clamps,
QK-Norm). SORC is added *capacity*, not a new loss term — deliberately no regularizer/penalty (so no
gradient pathway can grow the strength).

**[THEOREM] Flagship recovery.** `φ_l ≡ 0 ⇒ θ_l ≡ 0 ⇒ a=c=0 ⇒` Rot = identity ⇒ block = flagship
bit-identically (forward, inverse, gradients). Default init `φ=0` ⇒ SORC *is* the flagship at step 0
(no regression risk), with full first-order sensitivity `dθ/dφ|_0 = θ_max` (live at init; no warmup
needed to unfreeze).

---

## 7. Theoretical analysis

### 7.1 [THEOREM] Conservation ⇒ runaway impossible (effect side)
`R(θ) ∈ SO(2)`: `‖R(θ)‖₂ = 1`, `det R = 1`, and per channel `q_i² + p_i²` is **exactly conserved by
Rot** (for any θ). Therefore the coupling **cannot inject magnitude into q** — it can only rotate the
`(q_i,p_i)` pair. The Rot factor of the block Jacobian is orthogonal (all singular values = 1), so it
is **gradient-norm-preserving across all L layers** (no amplification). OBSD's failure mode —
‖g‖-inflation via unbounded magnitude injection — is **structurally absent**: the coupling's
contribution to ‖g‖ is bounded by 1 in operator norm regardless of training step. So it cannot
progressively steal the grad-clip budget / throttle the effective LR.

### 7.2 [THEOREM] Bounded parameter + self-attenuating gradient (parameter side)
`θ = θ_max·tanh(φ)` ⇒ `|θ| < θ_max < π/2` for all φ (so `tan(θ/2) < tan(π/4)=1`, finite, and the
3-shear realization never blows up), and `dθ/dφ = θ_max·sech²(φ) → 0` as `|φ|→∞`. So the angle
**saturates** at θ_max and the optimizer's incentive to push φ **vanishes** at saturation — Adam
cannot march φ the way it marched OBSD's linear gate (whose `dφ/da` was constant and sign-definite).
Bounded on **both** manifolds: the effect (conservation, §7.1) and the parameter (saturation).

### 7.3 [DERIVABLE] Cross-depth composition
Rot folds a bounded component of the accumulated momentum `p̃ = p_0 + Σ_{l'≤l} Y_{l'}(q_{l'})` into q
(via the Shear^q steps), so `Y_{l+1}(q)` attends over a q carrying prior attention — the targeted
composition — while the Shear^p back-reaction conserves total per-channel energy (information
**redistributed**, not injected). Over L=24 layers with reln between rotations, the achievable
cumulative directional travel is large (`L·θ_max`); the per-layer bound is on the *fraction*, not the
cumulative reach — expressivity is not the binding constraint of the bound itself.

### 7.4 [THEOREM] Exact reversibility / O(1) memory
Each shear is unit-triangular (reads only the opposite branch); the composite Rot has det = 1; the
block determinant = flagship's `det(∂reln)`. Inverse = 3 negated shears reversed (§5.4); pure
additive shears → BF16-ULP exact. No activation stored beyond the `m·L` angle params (γ recomputes).

### 7.5 Honest limits
- **reln breaks *global* energy conservation.** Only Rot conserves `q²+p²`; the subsequent reln
  rescales q (the deliberate scale re-supply). So SORC is "conserve-then-renormalize," not globally
  Hamiltonian. This is intended (a pure rotation with no scale re-supply would be too restrictive),
  and reversibility (§7.4) is independent of it. [DERIVABLE]
- **Per-channel scalar angle** — same granularity as OBSD's gate; the bet (§12) is that a per-channel
  bounded rotation carries the cross-depth signal that lowers NLL. [CONJECTURE]

---

## 8. Computational tradeoffs

| Resource | Flagship | OBSD drift | **SORC** |
|---|---|---|---|
| GEMMs/attention passes / layer | — | 0 new | **0 new** |
| Forward hot-path / layer | — | 2 reductions + tanh + axpy | **3 per-channel axpy (no reductions, no tanh)** |
| Inverse / layer | — | recompute drift | 3 negated axpy |
| Backward / layer | — | pre-bwd + col-reduce + reanchor | 3 adjoint axpy + 1 col-reduce (`dφ`) |
| Params | flagship | +3·L·m (a,M⁻¹,b) | **+L·m ≈ 49K (φ only)** |
| Wall | baseline | +12–19% (measured) | **+4–6%** (cheaper — no reductions/tanh) |
| Activation memory | O(1) | O(1) | **O(1)** |
| Reversibility | — | BF16 ULP | BF16 ULP |

---

## 9. Comparison to existing approaches

| Approach | Coupling form | Boundedness | vs SORC |
|---|---|---|---|
| **Flagship** | none (fold once at L−1) | n/a | SORC adds bounded per-layer exchange; recovered at θ=0 |
| **OBSD** | additive drift `q += a⊙φ(N(p))` | none (free gate → maxA 1.8) | SORC replaces injection with conservation; runaway impossible |
| **GBC (A)** | secant on reln shell | compact S¹ angle | SORC exact rotation (no √2 leak); adopts A's bounded angle |
| **HBSD (B)** | OBSD drift + dual regulator | dynamic setpoint ρ* | SORC structural (no setpoint, no controller, cheaper) |
| **Leapfrog/Strang** | 2nd-order integrator | — | order is meaningless for learned dynamics (established); SORC uses symplecticity for *conservation*, not accuracy |

SORC reduces to: the flagship (θ=0), and (formally) the OBSD-helpful regime geometrically pinned (the
bounded-angle exchange occupies the `|coupling|≲1` regime where OBSD was winning, structurally
excluding the `>1` regime that produced the reversal).

---

## 10. Failure modes and mitigations

- **FM-1 (dominant, the open bet).** Bounded cross-depth composition still doesn't lower NLL, or the
  coupling *itself* (not its magnitude) blocks the baseline's late acceleration. *This is the decisive
  test:* SORC's magnitude **provably plateaus** (rotation + bounded angle), so a late reversal here
  *falsifies* "gate magnitude was the problem" and implicates composition-itself. Either outcome is
  informative. Mitigation for the design: calibrate θ_max from OBSD's winning regime (the coupling
  magnitude at maxA≈1.0 / step≈16.5k, where OBSD was +0.05 ahead).
- **FM-2 norm-preservation too restrictive.** A pure rotation can't grow q. *Mitigated* by K→Rot→N
  (reln re-supplies scale). If still too tight, the long-term lift is a **symplectic squeeze**
  (general `Sp(2,ℝ)`, allowing *controlled* growth) instead of a pure orthogonal rotation — but that
  reintroduces a (bounded) growth channel; start with the pure rotation.
- **FM-3 `tan(θ/2)` blow-up.** Kept finite by `θ_max < π/2` (e.g. 60° → tan(30°)=0.577). Hard
  invariant of the parameterization.
- **FM-4 BF16 reconstruction.** The 3 shears must be FP32-computed with single rounding (like the
  OBSD drift) for BF16-ULP-exact inverse; verify on the roundtrip test.
- **FM-5 per-channel angle expressivity.** Same granularity as OBSD; long-term lift = per-head or
  per-channel-pair rotations / butterfly pairing across depth, preserving the conservation proof.
- **FM-6 wall above budget at scale.** SORC's own cost is +4–6%; if the shared (OBSD-era)
  `chiron_col_accumulate` reduction dominates, the 2-phase-tiled reduction optimization applies.

---

## 11. Minimal prototype (reuse the OBSD/shear scaffolding)

**glades-ml (kernels + CPU ref + unit tests):**
- CPU ref: `shear_add_to_q`/`_p`, `shear_sub_from_q`/`_p` already exist
  (`transformer_chiron_ops.h:42–67`); SORC's Rot composes exactly these. Add
  `rot_forward_row`/`_inverse`/`_backward` building on them + the `a=−tan(θ/2)`, `c=sin θ`,
  `θ=θ_max·tanh(φ)` coefficient maps.
- GPU: one new per-channel-broadcast axpy kernel `chiron_qp_chan_axpy(dst, coef[m], src, sign, T, m)`
  (`dst += sign·coef⊙src`, broadcast coef over t) modeled on the elementwise body of
  `chiron_drift_into_q_rows` (`gpu_chiron.cu:908`) with the reductions/tanh removed; a coefficient
  kernel `rot_coeffs(φ[m], θ_max → a[m], c[m])`; the angle-gradient column reduce reuses
  `chiron_col_accumulate` (`gpu_chiron.cu:1207`); reln backward unchanged
  (`chiron_reln_backward_reanchor`).
- Tests (model on the OBSD drift tests in `chiron-test.cpp`): (i) `R(θ)` 3-shear composition vs a
  reference `[[cos,−sin],[sin,cos]]` (max err < 1e-5); (ii) forward∘inverse reconstructs (q,p) within
  BF16 ULP; (iii) CPU/GPU parity fwd+bwd; (iv) FD grad-check of `dφ`; (v) **norm-conservation**: the
  Rot step conserves `q²+p²` per channel to FP32 precision.

**glades-trainer (param/flag/wiring/checkpoint):**
- New per-layer param `rot_phi[l] ∈ ℝ^m` (init 0), Adam state, checkpoint bit (next free, e.g. 1024),
  mirroring the `a_drift` lifecycle.
- Flags `--rot-coupling`, `--rot-theta-max` (default e.g. π/3), `--rot-warmup` (reuse `--drift-warmup`
  plumbing). Mutually exclusive with `--per-layer-drift` and `--fuse-attn-per-layer` (shared insertion
  point); the `--scfa requires --bf16-weights` guard already in place.
- Insert Rot at the OBSD drift insertion point (after the kick/fold, before the q-side reln) in all
  attention branches; wire the inverse (negated shears) and backward; include `dφ` in grad-norm/clip;
  the angle's effect at φ=0 is exactly identity (E0 bit-identity gate, as for OBSD).
- Build gotcha (lineage): trainer links the installed glades (`~/.local`) → `make install` after
  kernel changes.

---

## 12. Full research program / the decisive experiment

**Experiment ladder (mirrors OBSD's, leveraging its harness):**
- **E0** — φ=0 bit-identity gate (loss + grads bit-identical to flagship).
- **E1** — reversibility + norm-conservation + grad-check at small shape (unit tests).
- **E2** — small-shape training: Rot active, 0 grad-skips, reconstruction within ULP, `|θ|` plateaus
  ≤ θ_max, φ learns from 0.
- **E3** — production T=16384, ~2.5–7.5k step sweep over θ_max ∈ {π/6, π/3, π/2−δ}; pick the θ_max
  that improves val vs the matched baseline trajectory; confirm `|θ|` plateaus and the coupling
  ‖g‖-share is flat.
- **E4 (decisive)** — production 30k, single-seed 1337, matched baseline. **Ship iff** val NLL <
  baseline with a **non-shrinking gap through 24–30k**, `|θ|` plateaued, 0 grad-skips, wall ≤ +10%,
  reconstruction within BF16 ULP. Multi-seed `{2024,4242}` capped 5–15k for sign + cross-seed
  plateau (owner cost-control precedent).

**The open bet (FM-1).** SORC is the principled instrument that the failed late-cap could not be: its
magnitude plateaus by construction, so E4 cleanly separates "OBSD failed because the gate grew too
big" (→ SORC should hold the early gain and ship) from "cross-depth composition itself blocks the
late acceleration" (→ SORC reverses despite a plateaued magnitude — a decisive negative for the whole
composition thesis, not just for unbounded gates).

**Long-term lifts** (if E4 is positive but marginal): per-head/per-channel-pair rotations; symplectic
*squeeze* (controlled growth) vs pure rotation; data-dependent angle (θ a cheap function of p̂, still
bounded).

---

## 13. Open conjectures and validation criteria

- **[CONJECTURE — central]** A bounded per-channel rotation retains OBSD's early +0.05 nat (it
  occupies the winning `|coupling|≲1` regime) while structurally excluding the reversal, so the gap
  holds through 30k.
- **[CONJECTURE]** Bounding the coupling's ‖g‖-share to a step-independent constant (§7.1) is
  *sufficient* to preserve the late-training acceleration OBSD lost.
- **[EMPIRICAL]** `max_i |θ_{l,i}|` plateaus ≤ θ_max for all steps (vs OBSD's maxA→1.8); coupling
  ‖g‖-share flat; 0 grad-skips; reversibility within BF16 ULP.
- **Acceptance (ship gate):** val NLL < matched baseline @30k, non-shrinking gap through the late
  regime; coupling magnitude plateaus; 0 grad-skips; bit-identical flagship at init; wall ≤ +10%;
  reversibility within BF16 ULP.

---

## 14. One-line thesis

Replace OBSD's **additive** drift (which injects unbounded magnitude into q) with a per-channel
**norm-preserving symplectic rotation** of `(q,p)`, realized as three reversible shears with a
bounded angle `θ = θ_max·tanh(φ)`: the coupling **exchanges** information between q and p instead of
injecting it, so by the conservation law `‖R‖₂=1` it **cannot run away** for any parameter value —
removing OBSD's failure (gate↑ → ‖g‖↑ → clip↑ → eff-LR↓) **by a conservation law, not a cap** — while
preserving exact reversibility, O(1) memory, flagship recovery, and a +4–6% wall.
