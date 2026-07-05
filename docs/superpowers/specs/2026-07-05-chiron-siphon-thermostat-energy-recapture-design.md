# CHIRON SIPHON — Symplectic Increment-Power Harvested into an Off-graph Nosé reservoir

**Date:** 2026-07-05
**Status:** DESIGN. Not yet implemented. Ladder pre-registered in §10 (Phase-0 = CHARGE diagnostic,
own plan `docs/superpowers/plans/2026-07-05-chiron-charge-phase0-diagnostic.md`).
**Author:** research-framework-design (3-candidate parallel synthesis: CHARGE / SIPHON / SPARK).
**Target:** turn the recurring training-instability gradient spikes of the CHIRON 1B PIED flagship
(`chiron_1B_pied_e4.final`) from *discarded* energy into *recaptured* energy that lowers val NLL.
Ship metric = matched-30k wide-32 val NLL, on the PIED recipe (WhiSC-D + `--inc-dropout 0.1`).

**Predecessor lessons (binding):**
- **WhiSC-D** — SHIPPED: the p/q asymmetry `ρ=σ^p/σ^q ~10³` is *trained-in within ~100 steps*
  (no early p≈q window); any phase-space lever must act in the **detached whitened frame** from
  step 1. Reuse its whitening `a=(σ^q/σ^p)^{1/4}` and its `[whisc]`/`[sorc]` monitors.
- **SORC** — NO-GO (diverged ‖g‖→4.7e10): joint-(q,p)-norm conservation is the WRONG invariant
  (p-dominated); a differentiable coupling `q+=a·p` gave a backward pole `∂L/∂φ ∝ ρ` cascading
  `λ_p^L≈1.4e9`. **Derive backward bounds; never assume them from forward structure. The
  differentiable content must be a contraction, never ρ-scaled.**
- **OBSD** — NO-GO (regressed −0.66 @30k): an *unbounded ReZero gate* on a per-layer drift
  throttled effective LR. **Bound every rate; damp toward a self-calibrating target, never fight
  the baseline signal.**
- **PIED** — SHIPPED: a *detached per-(l,µstep,t,i) scalar mask* on the p-bus adds no gradient
  path and is exactly invertible. SIPHON's friction is the same object (a detached exp-scalar
  gain) — that is the safe way to touch the forward.
- **PACT** — E4 NEGATIVE (elegant, principled, E0–E2 PASS, no 30k win): **elegant ≠ effective.**
  A cheap falsifiable gate MUST precede any 30k commitment, and the gate must separate
  "stabilized" from "did useful work."

---

## 1. Executive summary

The training exhibits recurring gradient-norm spikes (isolated hard-batch ‖g‖ 10.8 @step9001;
PACT-arm ‖g‖ 12.5 @step1661; chronic late-training ‖g‖≈2–4; catastrophic SORC ‖g‖→4.7e10). Two
mechanisms *discard* the associated energy: the **global grad-clip** `g↦0.5·g/‖g‖`, which — since
‖g‖ is chronically 2–4 — **fires almost every step**, throwing away a 4–8× factor of the gradient
length along the (preserved) clip direction; and **loss-scale/grad-skip**, which discards the whole
step on overflow. This design recaptures that energy instead of discarding it.

A decisive observation constrains the whole design space: **the global clip is isotropic**, so the
discarded gradient residual `r_t = g_t − clip(g_t) = (1−τ/‖g_t‖)_+ g_t` is *collinear* with `g_t`.
A single step's escaping *gradient* energy is therefore pure **magnitude**, carrying no new
direction. The recapturable value in the gradient domain is bounded to *de-biasing the clip's
magnitude suppression* (this is the CHARGE diagnostic, §10 Phase-0) or *aggregating the overflow
direction across spikes* (SPARK, App. B). Only a mechanism that captures a **different** energy —
the forward-pass phase-space energy of the p-accumulator — reaches the *root* of which the gradient
spike is a *symptom*. SIPHON is that mechanism.

SIPHON attaches a **per-channel Nosé–Hoover thermostat to the whitened phase-space energy of the
p-accumulator**, integrated down depth alongside the reversible block. Friction removes exactly
`ξ·𝔼[p̃²]` from the phase-space energy `H̃` and **books it into a reservoir** (`½Qξ² + P*s`), with
the **extended energy `H_ext = H̃ + ½Qξ² + P*s` exactly conserved** — energy is *relocated, not
destroyed*. A self-calibrating target `P*` makes the thermostat store-and-release (bank on spikes,
release on calm); an optional cross-step reservoir cashes the banked energy out as plasticity
pressure on a learnable increment-gain field. The friction is a **detached exp-scalar gain** on
`p` (identical autodiff status to a PIED mask ⇒ no gradient path, exactly reversible), so SIPHON
cannot host the SORC backward pole and provably **caps** the accumulator temperature.

## 2. The three candidates

| | **A — CHARGE** | **B — SIPHON (selected)** | **C — SPARK** |
|---|---|---|---|
| Class | discrete-optimization / conservation law | extended-Hamiltonian / dynamical-systems | information-geometric / bilevel |
| "Energy" | clipped first-moment **gradient mass** along Adam consensus `m̂` | **whitened phase-space energy** `½Σ(q̃²+p̃²)` of the p-accumulator | **tail-Fisher** `tr(Σ βφ·ggᵀ)` of the overflow subspace |
| Recapture | store clipped mass; discharge into unused clip headroom on calm steps | thermostat banks spike energy in `ξ,s`; cash out via reinjection / gain-field | low-rank sketch preconditions sharp directions + energy bank |
| Locus | optimizer state only | **inside the forward** (down depth) | optimizer state only |
| Stability proof | `‖update‖ ≤ τ` always (≤ baseline) | thermostat **caps** p-temperature (better than baseline) | `‖update‖ ≤ (1+ε)c` |
| VRAM | ~0 (scalars) | ~80 KB | 6–35 MB |
| Central risk | "persistent = useful" (repetition attractor) | cash-out inert → collapses to a whitened clip | curvature staleness; K-FAC rarely beats tuned Adam |

Full candidate documents: CHARGE = App. A, SPARK = App. B.

## 3. Selection rationale

**SIPHON is selected as the framework** — with CHARGE demoted to its **Phase-0 de-risking probe**,
not discarded.

Why SIPHON:
1. **It is the only literal realization of the "escaping energy" metaphor.** Recapture is a
   *conservation law* (`dH_ext/dℓ = 0`), not a slogan. CHARGE/SPARK reinterpret "energy" as
   gradient mass / curvature; only SIPHON conserves an actual energy.
2. **It is the only CHIRON-native candidate.** It exploits the symplectic (q,p) structure,
   integrates *down depth* like the block, and reuses WhiSC's whitening. CHARGE/SPARK are generic
   optimizer overlays that would work on any model — the brief deprecates ported/generic levers.
3. **It attacks the root.** The gradient spike is a forward-pass energy imbalance (mean-p² ×213 at
   the SORC explosion; ‖g‖ 10.8→4.9 co-moved with damped increment energy under PIED). The clip is
   downstream.
4. **Its stability is a net positive, not neutral:** the thermostat provably caps the p-temperature
   (§6), so it would have damped the step-9001 spike and bounded the SORC mode at the source.

The honest counter-case for CHARGE (and its resolution): CHARGE is lower-variance (provably
`‖update‖≤τ`, can never be less stable than today), near-zero cost, and attacks a *plausibly-real*
tax — the reanchor arc found chronic clamping "silently taxes perplexity ~0.7–0.8 nat," and the
flagship still runs a global clip that fires almost every step. The PACT result warns that
*elegant ≠ effective*. **Resolution:** the core hypothesis under all three candidates is *"the
energy discarded by clip/skip contains recoverable perplexity value."* CHARGE is the cheapest,
safest, provably-can't-hurt test of that hypothesis, so it is SIPHON's **Phase-0 gate** (§10). And
because CHARGE acts on the *gradient* energy while SIPHON acts on the *phase-space* energy, the two
are **composable**, not exclusive. SPARK is held as a fallback if a preconditioner angle is later
wanted.

## 4. Formal problem statement

Let `θ∈ℝ^P` (P≈8.7·10⁸), `L(θ)`=val NLL. Forward per layer `ℓ=0..L−1` on `(q_ℓ,p_ℓ)∈ℝ^{T×m}`
(`T=16384`, `m=2048`, `L=24`): `y_ℓ=SCFA(q_ℓ)`, `Δ_ℓ=shear(y_ℓ)`, `p_{ℓ+1}=p_ℓ+Δ_ℓ`,
`q_{ℓ+1}=ReLN(q_ℓ)`; p folds into q at `ℓ=L`. Per-channel second moments
`σ^q_i=𝔼_t[q²_{·,i}]`, `σ^p_i=𝔼_t[p²_{·,i}]`, ratio `ρ_i=σ^p_i/σ^q_i∈[300,2000]`. Training: Adam,
global clip `g↦τ·g/‖g‖` (`τ=0.5`), dynamic loss-scale. **Objective:** design an operator `𝒮` on
the training dynamics that (i) defines a precise energy functional whose spikes coincide with the
‖g‖ bursts, (ii) conserves that energy into a reservoir instead of discarding it, (iii) re-injects
it to lower `L`, subject to: exact reversibility, determinism, p/q-scale-safety, `<0.9 GB`,
cheap-when-calm, and `𝒮_off ≡` baseline bit-identically.

## 5. SIPHON — core framework

### 5.1 Whitening (detached, reused from WhiSC)
`a_i=(σ^q_i/σ^p_i)^{1/4}=ρ_i^{-1/4}`. Whitened coordinates `q̃_i=q_i/a_i`, `p̃_i=a_i p_i` satisfy
`𝔼[q̃²_i]=𝔼[p̃²_i]=√(σ^q_iσ^p_i)` — **balanced, ρ-free** (the invariant SORC violated). `a` is a
per-channel EMA kept OUT of the autodiff graph (η≈0.05, current-batch floor). Whitened phase-space
energy and spike observable:
```
H̃ = ½ Σ_{t,i} ( q̃²_{t,i} + p̃²_{t,i} ),        T̃_{ℓ,i} = a²_i · 𝔼_t[ p²_{ℓ,·,i} ]  (whitened temperature).
```
`T̃` is exactly the quantity that ran away ×213 at SORC.

### 5.2 Thermostat (continuous extended flow)
Per channel `i`, detached DOFs `ξ_i` (Nosé–Hoover momentum) and `s_i` (reservoir), carried down
depth (`ℓ`=time, `f̃`=whitened SCFA force), thermostat mass `Q>0`, self-calibrating target `P*_i`:
```
dq̃_i/dℓ = p̃_i
dp̃_i/dℓ = f̃_i(q̃) − ξ_i p̃_i           # force + thermostat friction
dξ_i/dℓ = (1/Q)( T̃_{ℓ,i} − P*_i )      # coupling reads the WHITENED temperature
ds_i/dℓ = ξ_i                          # reservoir integrates the thermostat coordinate
```
**Conserved extended energy [proven, standard Nosé–Hoover]:**
```
H_ext = H̃ + Σ_i ( ½ Q ξ²_i + P*_i s_i ),      dH_ext/dℓ = 0.
```
The friction removes `ξ𝔼[p̃²]` from `H̃`; it reappears in `½Qξ²+P*s`. **That is the recapture, as a
conservation law.** `P*_i` is a slow EMA of `T̃_i` (η≈0.05): the thermostat is a *nudge toward the
running mean*, biting only on excursions — it never fights the baseline p-signal (the OBSD-trap
mitigation).

### 5.3 Discrete reversible integrator (BAB / Strang splitting)
Carry `ξ_i,s_i` down depth; `a_i,P*_i` frozen within a step; depth step `h=1`:
```
for ℓ = 0..L−1:
  # B (half thermostat, EXACT-exponential — unconditionally stable):
  ξ_i     += (h/2Q)( a²_i·𝔼_t[p²_{ℓ,·,i}] − P*_i );   p_{ℓ,i}   *= exp(−ξ_i h/2);   s_i += (h/2)ξ_i
  # A (CHIRON block, UNCHANGED):
  Δ_ℓ = shear(SCFA(q_ℓ)); p_{ℓ+1}=p_ℓ+Δ_ℓ; q_{ℓ+1}=ReLN(q_ℓ)
  # B (half thermostat, symmetric on p_{ℓ+1}):
  ξ_i     += (h/2Q)( a²_i·𝔼_t[p²_{ℓ+1,·,i}] − P*_i ); p_{ℓ+1,i} *= exp(−ξ_i h/2); s_i += (h/2)ξ_i
```
`exp(−ξh/2)` is a **detached per-(ℓ,i) scalar gain** — same autodiff status as a PIED mask ⇒ it
adds **no gradient path** and inverts exactly (`p ← e^{+ξh/2}p`; `ξ` inverts from the inverse-walked
`⟨p²⟩`). Store terminal `(ξ_L,s_L)` = `2m` FP32 seeds for the reverse walk.

### 5.4 Recapture / cash-out
- **SIPHON-0 (zero new params):** the finite `P*` makes store-and-release automatic. `T̃>P* ⇒ ξ>0
  ⇒` damp (bank into `ξ,s`); `T̃<P* ⇒ ξ<0 ⇒ e^{−ξh}>1 ⇒` **amplify** (release). Energy grad-clip
  would zero is time-redistributed to layers/steps with headroom; "useful work" = the increment
  information is *retained and re-expressed* rather than clipped to zero.
- **SIPHON-1 (upgrade):** persisted leaky reservoir `S_i ← ρ_S S_i + E^res_i` (`ρ_S≈0.99`,
  `E^res_i` = net energy banked this step) drives an energy-proportional plasticity pressure on a
  learnable per-channel increment gain `g_i=e^{β_i}` (init 0):
  `∇_{β_i} ← ∇_{β_i}^{loss} − κ·clip(S_i)`. Channels that chronically bank energy get extra
  optimization on their capacity; as `β` moves, `T̃→P*` and `S` drains — a closed
  **reservoir → work → drain** loop, vetoed by the loss.

## 6. Theoretical analysis

- **Energy bound / SORC-mode cap [derivable].** For `ξ>0`, `p̃↦e^{−ξh}p̃` is a strict contraction,
  so `T̃_{ℓ+1}≤T̃_ℓ` whenever `T̃_ℓ>P*`. Lyapunov `V=½(T̃−P*)²+½Qξ²` decreases while `T̃>P*`, giving
  `T̃_ℓ ≤ max(T̃_0, P*+O(1/Q))` — a hard cap on accumulator temperature; the ×213 blow-up is
  bounded at the source.
- **Non-amplification of SORC [derivable].** SORC's pole was *backward*: `∂L/∂φ ∝ ρ` cascading
  `λ_p^L`. SIPHON's differentiable content is a detached gain `e^{−ξ}≤1` (for `ξ≥0`): it
  contributes a *contraction* to the state-adjoint, never a ρ-scaled factor, and there is **no
  `∂L/∂ξ` term**. Structurally it cannot host the SORC pole.
- **Reversibility + determinism [derivable].** BAB is time-reversible; friction inverts as
  `p←e^{+ξh/2}p`; no RNG anywhere ⇒ same-seed bit-reproducible (modulo atomic noise).
  `Q→∞ ⇒ ξ≡0 ⇒` bit-identical to CHIRON (clean E0).
- **Thermodynamic reading.** SIPHON minimizes `L` subject to a soft **equipartition** constraint
  `T̃_ℓ≈P*` across depth; the multiplier energy `S` is *spent on `L`* rather than dissipated —
  "clipping → thermalization into a recoverable reservoir."

## 7. Comparison to existing methods (limiting cases + strict departures)
- **Berendsen thermostat** = drop the `ξ` integral, replace by a proportional per-step rescale
  `λ=√(1+(h/τ)(P*/T̃−1))`. No conserved `H_ext`, no reservoir, no cash-out.
- **Temperature-controlled SGLD** = add `√(2ξ/β)dW` to `dp̃`. SIPHON keeps only the deterministic
  dissipation half (determinism constraint).
- **Ordinary grad-clip** = `ξ→∞`-on-excursion, no memory, no reservoir → the whitened per-channel
  discard corner.
- **PIED** = the same detached-scalar-on-p object, but multiplicative *mask* (variance injection)
  vs SIPHON's *temperature control* (energy relocation). Composable (shared whitening buffer).
- **Strict departures:** the whitened metric (not raw p-kinetic), the recoverable reservoir `s,S`,
  and the cash-out into the model — none of the above recover any energy.

## 8. Computational cost
`ξ,a,P*`: `m` FP32 each; `S,β`: `2m` (upgrade); stored seeds `ξ_L,s_L`: `2m`. Total `<80 KB`
persistent + `O(m)` transient — `≪0.9 GB`. Wall: the temperature reduction coalesces into the
existing WhiSC `σ^p` reduction (≈free); a masked `exp`+scale on `p` (≈PIED masked-commit cost)
gated by `|ξ|<ε`. Estimate **+1–2% engaged, ≈0 when calm**. Composable with PIED/WhiSC.

## 9. Failure modes & mitigations
1. **Over-damping suppresses the p-signal (the OBSD trap).** Mitigated by design: `P*` is a
   self-calibrating EMA, so the thermostat bites only on excursions above the running mean, never
   fighting the baseline signal.
2. **Cash-out inert → collapses to a whitened clip** (releases where re-clipped, no net val gain).
   The **primary kill-target**: SIPHON then reduces to stability-only (passes 0-grad-skips /
   bounded-T̃, but the *recapture* claim fails). The gate must separate "stabilized" from "did
   useful work."
3. **EMA lag on `a,P*`** during the fast ρ-ramp → mis-targeted damping; costs efficacy not
   stability (contraction is stabilizing regardless). Fast EMA + current-batch floor (WhiSC).
4. **`S`-windup (upgrade)** → leaky `ρ_S<1` + `clip(S)` + `tanh` cash-out; `β` still loss-trained.

## 10. Research program (phased, falsifiable)

**Phase 0 — CHARGE diagnostic (cheap, safe, tests the premise).** Optimizer-only per-scalar
reservoir on the clipped gradient magnitude (App. A; provably `‖update‖≤τ`). Matched **2500-step
pair, seed 1337**. Tests the *shared* hypothesis — does recaptured gradient energy help at all, and
is the chronic-clip tax real — with zero forward-map risk. **Signature:** val gap concentrated
post-spike, `Σ‖a_t‖>Σ‖u_t‖`, 0 grad-skips. **If null → the whole direction is likely dead; do not
build SIPHON.** Plan: `docs/superpowers/plans/2026-07-05-chiron-charge-phase0-diagnostic.md`.

**Phase 1 — SIPHON-0 gate.** Matched 2500-step pair ± `--siphon` (default-off, E0 bit-parity).
Kill-target signature: (a) `max_ℓ T̃` bounded vs baseline excursions (the SORC-mode kill, on a new
`[siphon]` monitor); (b) 0 grad-skips + reduced chronic-clip firing; (c) `val@2500 ≤ baseline` with
the gain **concentrated in the ~50–200 steps after the largest excursions** (the reinjection-lag
fingerprint distinguishing cash-out from mere damping). **Falsifier:** T̃ bounded but val ≥ baseline
and no post-spike gain ⇒ cash-out inert ⇒ SIPHON is stability-only.

**Phase 2 — SIPHON-1 cash-out** (add `S,β`), only if Phase 1 shows a real post-spike gain. Then the
standard E3→E4 ladder (matched 30k + wide-32), **PACT lesson enforced**: no 30k commitment until the
cheap gate shows a widening post-spike advantage that is not era-drift.

## 11. Open conjectures / validation criteria
- **[Conjecture]** The recaptured phase-space energy does *useful* work (lowers NLL), not merely
  stabilizes. *Falsifier:* Phase-1(c).
- **[Conjecture]** Damping `T̃` toward a self-calibrating `P*` does not suppress the useful
  increment signal. *Falsifier:* val regression with T̃ over-bounded (sweep `Q`).
- **[Empirical]** SIPHON damps the step-9001-class spike *in the forward* (lower `max T̃`) more
  cleanly than PIED's incidental damping.
- **[Empirical]** CHARGE (Phase-0) recovers a measurable slice of the chronic-clip tax —
  independently interesting even if SIPHON fails.

---

## Appendix A — Candidate CHARGE (Phase-0 probe; full text)

**CHARGE** — a bounded gradient reservoir treating the clipped residual `r_t = g_t − clip_τ(g_t)`
as a conserved first-moment charge, stored per parameter-group and discharged into the update only
on demonstrably-calm steps, within the *existing* clip norm-budget (never above it).

- **State:** per-group scalar charge `ρ^{(G)}∈[0,ρ_max]` (Adam-adjacent; the projected clipped mass
  onto the group's Adam-consensus direction `m̂^{(G)}=m^{(G)}/‖m^{(G)}‖`). Global-scalar reduction
  for the minimal prototype.
- **Charge (always):** `ρ^{(G)} ← Π_{[0,ρmax]}( ρ^{(G)} + ⟨r_t^{(G)}, m̂^{(G)}⟩ )`.
- **Discharge (if calm ∧ high-SNR):** `δ^{(G)} = γ·min( ρ^{(G)}, (τ_G − ‖u^{(G)}‖)_+ )`,
  `ρ^{(G)} ← ρ^{(G)} − δ^{(G)}`; effective grad `a_t^{(G)} = u_t^{(G)} + s_t·δ^{(G)}·m̂^{(G)}`,
  whitened budget `τ_G = (‖m^{(G)}‖/‖m‖)·τ`.
- **Conservation law:** `∫a_t = ∫g_t − overflow − undischarged_tail`; when the reservoir drains,
  the time-integral of the delivered update equals the unclipped first-moment mass — the clipped
  direction is time-delayed, not destroyed.
- **Objective:** global-clip Adam descends a Huberized loss (per-step trust region `‖update‖≤τ`);
  chronic clipping runs the *saturated* branch (biased). CHARGE de-biases the **time-averaged**
  update to equal `∇L` while keeping each step inside the trust region — the minimal de-bias that
  does NOT enlarge the trust region.
- **Stability [proven]:** by Minkowski with `Σw_G²=1`, `‖a_t‖ ≤ (1−γ)‖u_t‖ + γτ ≤ τ`. The gradient
  Adam sees never exceeds the baseline clip radius → CHARGE **cannot** produce the SORC failure
  mode; it is at most as unstable as baseline.
- **Limiting cases:** `ρ_max→∞, γ=1, gate≡1, no projection ⇒` error-feedback / EF-SGD; `‖r‖`-EMA
  ⇒ momentum; within-step ⇒ grad-accumulation. Strict departures: hard output budget `τ`,
  headroom-gated emission, cap `ρ_max`, `m̂`-projection (recapture only persistent mass), whitened
  per-group budget.
- **VRAM ≈ 0** (scalars; `m̂` reuses resident Adam `m`). **Risk:** "persistent = useful" — could
  recapture a harmful direction (repetition attractor). Full-vector conservation exceeds the 0.9 GB
  budget (871 MB int8 all-param); restrict to top-mass tensors (~260 MB) or use the scalar-`m̂`
  reduction (budget-safe, principled).

## Appendix B — Candidate SPARK (fallback; full text summary)

**SPARK** — a Fisher-beacon energy-recapture optimizer overlay. The escaping energy primes a
whitened low-rank curvature memory (rank-k Kronecker eigensketches of the clipped residual — the
tail-Fisher `F̂^clip=Σ_t β_t φ_t g_t g_tᵀ`) and a scalar energy bank; both re-inject the discarded
work as **bounded natural-gradient descent** along the sharp directions, `δ=−η(F̂^clip+λI)^{-1}g`
restricted to the rank-k subspace. Sign-aligned (`⟨p̂,g⟩≥0`), `‖update‖≤(1+ε)c` (no SORC),
optimizer-only (determinism/reversibility safe), whitened before sketching (p/q-safe), ~6–35 MB.
Beacon = `1[‖g_t‖>c]` (free). Key point: since `r_t` is collinear with `g_t`, the *direction* is
recovered only by cross-spike aggregation into the tail-Fisher — the novel object. Departs from
Adam/K-FAC/natural-gradient by using **only** the escaping (clipped) tail as its information source.
**Risks:** curvature staleness, locus misplacement, K-FAC-family rarely net-positive over tuned
Adam at scale; the replay-buffer variant is rejected on determinism grounds (data reorder perturbs
the counter-hash RNG stream). Held as a fallback if a preconditioner angle is wanted after
Phase-0/1.
