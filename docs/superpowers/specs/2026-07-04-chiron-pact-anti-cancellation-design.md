# CHIRON PACT — Profile Anti-Cancellation Tax

**Date:** 2026-07-04
**Status:** M0 GO, **E0/E1/E2 PASS** (2026-07-04; E2 applied a field-scaling refinement of
record — see §13). Next gate: E3 (matched 2500-step fresh pair). Ladder pre-registered in §13.
**Author:** research-framework-design (3-candidate parallel synthesis)
**Target:** the next CHIRON-native regularizer for the production flagship (CHIRON 1B PIED E4,
`chiron_1B_pied_e4.final`), ship metric = matched-30k val NLL (E4), on the PIED recipe
(WhiSC-D + `--inc-dropout 0.1`).
**Predecessor lessons (binding):**
- **PIED** — **SHIPPED −0.35 matched**: increment gating on the p-bus; implicit regularizer =
  Fisher-weighted increment energy; its distinctive content (anti-cancellation across depth) is
  the demonstrated-productive direction. Extend it, don't re-run it.
- **SIRA** — SHIPPED (−0.03): terminal (q_L,p_L) magnitude penalty. Small; regularizes the sum,
  not the assembly.
- **OBSD / SORC** — NO-GO: hard-bound every rate; derive backward bounds, never assume them from
  forward structure; q-perturbations O(‖q‖); watch parameter-gradient channels.
- **WhiSC-D** — SHIPPED: scale statistics detached, engaged from step 1; ρ = σ^p/σ^q ~10³ is
  trained-in within ~100 steps (no early p≈q window).
- **LayerDrop** — FAIL: never bimodalize Adam moments; never gate states.

---

## 1. Executive summary

PIED's win came from an *implicit* penalty: mean-one increment dropout prices, at second order,
the Fisher-weighted energy of every attention increment — and its distinctive content is
**anti-cancellation across depth** (a mutually-cancelling increment pair contributes nothing to
the function but is taxed in full). This design starts from a small theorem about where that
mechanism family ends:

> **T1 (diagonal blindness).** For *any* mask family that is mean-one and independent across
> layers — PIED at any π, the symmetric arm, per-head pre-Wo masks, spectral (l,j) masks, any
> adversarial reweighting of their rates — the second-order implicit regularizer is
> `Σ_l Var(η_l)·H_ll`: **diagonal in the layer index**. The cross-layer Gram
> `Y_{l,t,i}·Y_{l',t,i}` — the entirety of the sign structure that *is* cancellation — sits
> off-diagonal and is invisible to the whole family, forever, at any rate.

PIED taxes cancellation only through its diagonal shadow (total energy at fixed sum), with
strength welded to the mask variance (raising it raises the Jensen tax and gradient noise in
lockstep). **PACT taxes the off-diagonal object directly, deterministically, and only where it
is actual cancellation:**

> **PACT** adds a training-loss penalty on the **null component of each (token, channel) depth
> profile** — the part of `Y_{·,t,i} ∈ ℝ^L` orthogonal to the damping direction `D_{·,i}`,
> i.e. the energy the assembly spends that provably never reaches `p_L` — **gated by a detached
> sign-coherence statistic `χ ∈ [0,1]` that is exactly zero on every sign-consistent profile**
> (concentration, banding, specialization all exempt) and approaches 1 under pure cancellation.

Exact properties (proven in §6–§7):

- **Function-preserving field.** The injected gradient satisfies `Σ_l D_{l,i}·g_{l,t,i} = 0`
  exactly: the field redistributes the assembly at *fixed damped sum*, i.e. fixed contribution
  to `p_L` — the same zero-function-space-cost property that made PIED's anti-cancellation
  content safe. Because the p-bus adjoint damps upstream `dp` by exactly the same `D_l` per
  channel, the field's first-order effect on the task loss is zero in mask-expectation *through
  the WhiSC damping*, not just at θ=0.
- **Minimal intervention (T3').** The penalty's descent flow moves a cancelling profile along a
  straight line toward its minimal-energy realization and **stops at the sign-coherence
  boundary** (the gate releases): PACT removes opposition; it never equalizes the magnitudes of
  an already-coherent profile. Plain depth-variance (ungated) would fight specialization; the
  gate is what makes this a cancellation tax rather than a homogenization prior.
- **Noise-decoupled strength.** Deterministic penalty ⇒ no Jensen tax, no gradient noise; λ is a
  free knob where PIED's implicit strength is `π/(1−π)`·Fisher, welded to the noise budget.
- **Variance-reduction coupling with PIED (T2).** At fixed delivered signal, PACT-preferred
  profiles minimize (to within 0.3%) the noise PIED itself injects — so PACT should measurably
  *reduce PIED's Jensen tax* in vivo: a distinctive, falsifiable cross-mechanism observable.
- **Safety by inspection**, in the lineage's own terms: parameter-free (no checkpoint delta, no
  serving change), no scale inputs (a scale-free ratio with detached, floored per-channel
  normalization — ρ appears nowhere), no new backward operator (the field rides the existing
  SCFA adjoint seam), hard elementwise clamp (Huberized), no trunk writes, no randomness.

Cost: two running [T×m] BF16 accumulators (128 MiB, within headroom), FMAs fused into the two
seams PIED already owns, ~1.0–1.5% wall. **Measurement-first:** M0 probes the shipped flagship
checkpoint for cancellation occupancy (minutes of GPU) before any training spend.

**Honesty (pre-registered):** some sign-opposition is *functional* (WhiSC reads intermediate
`p_l` at O(θ_max)); the under-trained one-pass regime historically pays conditioning fixes more
than penalties; PIED may already have shrunk the taxable mass. Modal-outcome priors: ~30% win /
~45% neutral / ~25% regress conditional on M0+E3 pass (~22/50/28 unconditional). The M0 gate and
a cheap E3 price the thesis before any 30k spend.

---

## 2. Candidate formulations (3-candidate parallel synthesis)

Three independent theory agents developed materially different formulations from the identical
problem statement (architecture, F1–F3, lineage lessons, R1–R9, forbidden list).

### 2A — PACT / Profile Anti-Cancellation Tax (variational / explicit-penalty view)
The thesis above. Considered three functional forms and rejected two with derived reasons:
**Form A** (hinged cancelled-energy `[‖Y‖²−A²]_+`) *pays the model to inflate the aligned sum*
(`dA/dτ = 2(L−1)A` — first-order function distortion, collides with SIRA, re-opens the OBSD
‖g‖ channel) and hinge-gates whole-(t,i) gradients (Adam bimodality). **Form B**
(opposing-mass product `S⁺S⁻`) has a gradient that deletes minority-sign mass — exactly the
late-layer *corrections* that are the legitimate form of sign opposition — with indicator
bimodality near zero. **Form C (chosen)** uses B's value as a detached gate and A's geometry
projected onto the function-preserving cone. Contributes T1 (the family-blindness theorem), the
exact D-aware orthogonality, and the M0 measurement gate.

### 2B — AVEC / Adversarially-Varying Erasure Channel (game-theoretic / DRO view)
Replace PIED's fixed i.i.d. channel with a hard-capped, budget-conserving adversary reallocating
erasure variance across (layer, channel) classes: `sup_{P∈𝒫} E_η[L]` over a KL-anchored,
variance-budgeted uncertainty set; closed-form Gibbs inner maximizer; effective regularizer =
`(1−λ)·R_PIED + λ·soft-CVaR` of the Fisher-weighted class energies; online detached Hedge
adversary at O(L×m); total noise power held exactly at the PIED budget. Correct and complete
(exact monotonicity lemma; Danskin envelope; a derived two-timescale stability condition
`g = μ/τ̂ < 1` with damped-ringing signature). Own honest analysis: if the class energies are
already near-equalized, **AVEC collapses to PIED exactly** — an informative null, and its modal
outcome. Priors: ~25% win / ~45–50% neutral.

### 2C — TROPE / Temporal Rank-One Participation Excess (operator-theoretic / token-time view)
The one axis no shipped mechanism touches: penalize the **off-diagonal purity** of the whitened
cross-band Gram of `y_compr` (the k=1024 DCT-II token-frequency coefficients SCFA already
materializes in both walks) — taxing cross-band channel-pattern *coherence* (one channel pattern
replayed across positions) while structurally exempting corpus band structure (detached EMA band
whitening) and per-sequence spectra (Gram-diagonal deletion). Exact radial orthogonality
`⟨∇,Y⟩ = 0` (pure mode redistribution, zero energy pressure); forward-state invariance (the
strongest possible R3); 0.24% FLOPs. Own honest analysis: the NLL-win mechanism (coherent
increment fields under-differentiate logits across positions → late-bucket losses) is the
longest causal chain of the three; covers only the y_par band; priors ~25% win / **55%
neutral**. Also costed and rejected the true transfer-operator (Jacobian) variant honestly:
≥ +25% wall, 12–25× over budget.

**Convergent findings (all three, independently):** deterministic-or-budget-fixed mechanisms
with zero checkpoint delta; R1 derived as "no scale inputs, by inspection"; a measurement-first
kill gate before any training spend; honest modal outcome neutral. The convergence is evidence
the *family discipline* is right; the candidates differ in which mathematical object they claim
is live.

---

## 3. Framework selection rationale

**Selected: PACT (2A), with AVEC's protocol machinery and TROPE's measurement-first discipline
absorbed.**

- **Only PACT attacks a provably untouched object.** T1 shows the entire mask family — including
  every known PIED follow-up and AVEC's adversarial reweighting of it — is blind to the
  off-diagonal Gram. AVEC is, by its own construction, a reallocation of the *diagonal* pricing
  PIED already applies; TROPE moves to a genuinely virgin axis but with the weakest ship-metric
  thesis (its own modal outcome is neutral). PACT is the unique non-redundant continuation of
  the direction the lineage has already demonstrated productive (PIED −0.35 from this object's
  *diagonal shadow*; in-vivo hard-batch spike damping 10.8→4.9 from taxing the same object).
- **Only PACT inherits PIED's exact-safety signature.** The function-preservation identity
  `Σ_l D_l g_l = 0` is the explicit-penalty analogue of PIED's mean-one unbiasedness: zero
  first-order function cost, by construction, through the WhiSC damping. Neither AVEC (which
  deliberately *raises* the train tax on hot classes) nor TROPE (which redistributes spectral
  modes) has an equivalent.
- **Best risk profile of the three.** PACT is deterministic, parameter-free, clamped, and adds
  no new backward operator. AVEC introduces a genuinely new dynamical system (the
  model–adversary game) with a stability condition that must be measured (μ), plus
  nonstationary rates against Adam moments. TROPE is comparably safe but spends its safety on a
  weaker thesis.
- **Cheapest decisive falsification.** M0 measures the actual taxed object (cancellation
  occupancy) on the shipped checkpoint in minutes. AVEC's degeneracy check needs E2 dynamics;
  TROPE's probe is comparable (absorbed into M0's design).
- **Complementarity is exact, not hoped-for.** Zero first-order interaction with SIRA (the field
  preserves the damped sum SIRA's energy term reads); complementary to PIED by T1's
  diagonal/off-diagonal decomposition; T2 predicts PACT *reduces* PIED's measured Jensen tax —
  a cross-mechanism observable no other candidate offers.

**Absorbed from the rejected candidates:**
- From AVEC: the **concentration observables** (depth-Herfindahl of per-layer contributions;
  equalization trajectories) as E3/E4 secondaries; the **informative-null framing** (a
  degenerate outcome is a measurement, not a failure); the matched-budget comparison
  discipline; the detached readout-Fisher probe design (`φ̂_i` from subsampled
  `Var_{v∼s_t}[E_{v,i}]`, every K steps) — reused verbatim for PACT's Fisher-weighted ŵ
  upgrade (§14.2).
- From TROPE: the **measurement-first gate** (M0 mirrors TROPE's E1 probe: kill-before-spend);
  forward-state invariance as a design ideal (PACT's accumulators are write-only side products —
  the forward *function* is bit-identical with the flag on); the detached-EMA null-model
  pattern; the **position-stratified signature** discipline (pre-register where the win should
  appear, so the bucket protocol adjudicates direction, not just magnitude). TROPE itself
  remains a documented independent follow-up arc (§14.4) — the token-time axis stays open.

**Why the rejected are weaker for this experiment.** AVEC: second-order refinement of a shipped
mechanism at fixed noise power; modal outcome is collapse-to-PIED; two new risk channels
(oscillation, rate-nonstationarity vs Adam) that need their own gates. TROPE: self-predicted
neutral on the ship metric; partial coverage (y_par band only); the right second arc, not the
right first one.

---

## 4. Formal problem statement

Per token t ∈ {1..T}, phase state (q_t, p_t) ∈ ℝ^m × ℝ^m (m = 2048, L = 24, T = 16384,
nH = 16, V = 32000). Grounded layer-l forward (flagship recipe: WhiSC-D + PIED on):

```
 (shear)  p ← p + η_l ⊙ Y_l(q),  Y_l(q) = Wo·SCFA(Wq q, Wk q, Wv q) = y_par + y_perp  [q unchanged]
 (WhiSC)  (q,p) ← Φ_l(q,p),  per-channel M_i = W_i⁻¹R(θ_{l,i})W_i, |θ| ≤ θ_max = 0.07, detached W_i
 (ReLN)   q ← reln(q; γ,β)   per-token, reversible; backward re-derives stats from q_in (reanchor)
```

η_l = PIED mean-one mask (π = 0.1, counter-hash, training only). p_0 = 0, q_0 = embed; terminal
fold q ← q + p before logits z_t = (q_t+p_t)Eᵀ. Backward is the exact inverse walk (reln⁻¹ →
Φ⁻¹ → recompute Y_l(q), p −= η_l⊙Y_l); nothing stored per layer.

**Load-bearing structural facts.**
(F1) Attention reads only q; the shear writes only p; WhiSC (bounded by θ_max) is the sole
p→q route.
(F2) p_L is linear in the increments: `p_{L,t,i} = Σ_l D_{l,i}·η_{l,t,i}·Y_{l,t,i}` + q-leak
terms O(sin θ_max), with per-channel damping `D_{l,i} = Π_{l'>l} cos θ_{l',i} ∈ [0.945, 1]`.
The p-bus adjoint damps upstream `dp` by the same `D_{l,i}`.
(F3) ρ_i = σ^p_i/σ^q_i ≈ 4·10³ at convergence, trained-in within ~100 steps. Scale statistics
must be detached or the mechanism scale-free.

**Objective.** A default-off training mechanism such that
`L_train = NLL + Z-loss + SIRA + λ·R_PACT` improves matched-30k held-out NLL vs the PIED
flagship (ship bar Δ ≤ −0.02 nat, 4-batch, confirmed on wide-32), with 0 grad-skips and
wall ≤ +2%.

**Requirements.** R1 backward q-gain bounded independent of ρ (derived, §6); R2 exact identity
at λ=0 (E0 bit-parity); R3 inverse-walk compatibility, no per-layer [T×m] storage; R4 exact
train/inference consistency statement; R5 safe from step 1 and at ρ≈4100, no warmup-dependent
safety; R6 deterministic given seed; R7 wall ≤ ~1–2%, no new GEMMs, no unjustified [T×m]
round-trips; R8 C++98 + CUDA kernels with CPU refs, parameter-free (no checkpoint delta, no
serving change); R9 pre-registered ladder with kill criteria (§13).

**Forbidden:** state-zeroing; unbounded rates; raw-frame p→q mixing; stored per-layer
activations; warmup-dependent safety; learned rates in the prototype; changing flag-off math.

---

## 5. Core framework — the gated null-component penalty

### 5.1 Primitive objects (per µstep; accum = 4, each µstep completes forward+backward)

- `Y_{l,t,i}` — the **clean** (pre-PIED-mask) increment at layer l's commit seam
  (`y_par + y_perp`, the argument of `chiron_scfa_axpy2*`). Recomputed bit-exactly in the
  inverse walk (the baseline's own guarantee).
- `D_{l,i} = Π_{l'>l} cos θ_{l',i}` — the damping table, `[L×m]`, computed once per µstep by a
  suffix scan over the current θ table (all layers' θ are parameters, available at µstep
  start). `‖D‖²_i = Σ_l D²_{l,i} ≈ 0.96·L`, an `[m]` table.
- `A_{t,i} = Σ_l D_{l,i} Y_{l,t,i}` — the **signal accumulator**: the clean damped sum = the
  increments' entire contribution to `p_{L,t,i}` (exact at θ_max = 0; to O(sin θ_max) at
  0.07). ONE running [T×m] buffer, filled by a fused FMA at each commit.
- `M_{t,i} = Σ_l D_{l,i} |Y_{l,t,i}|` — the **mass accumulator** (detached; gate input only).
  ONE running [T×m] buffer. With `S^± = ` positive/negative damped mass, `A = S⁺−S⁻`,
  `M = S⁺+S⁻`, so `S⁺S⁻ = (M²−A²)/4` — opposing-mass statistics need no third buffer.
- `σ̂²_i` — detached per-channel EMA of `Y²` over (l,t) (WhiSC-stats kernel pattern; floored at
  ε₀ = 10⁻¹²; **not checkpointed** — re-warms in ~10 steps on resume; a gradient-scale input,
  not a correctness input).
- Knobs (compile-time defaults, none learned): coefficient λ (`--pact-coef`, default 0 ⇒
  mechanism entirely off), Huber clamp κ = 4, gate floor ε_M = 1, EMA rate = WhiSC's.

### 5.2 The penalty

Write `Y ≡ Y_{·,t,i} ∈ ℝ^L` for one (token, channel) depth profile, `D ≡ D_{·,i}`. The exact
identity

```
Σ_l (Y_l − A·D_l/‖D‖²)²  =  Σ_l Y_l²  −  A²/‖D‖²        (cross term telescopes; A = DᵀY)
```

says the **excess energy** of a profile over the minimum needed to deliver its own signal
(`min{ΣY² : DᵀY = A} = A²/‖D‖²`, attained at `Y* = (A/‖D‖²)·D`, Cauchy–Schwarz) **is** the
squared norm of the profile's component orthogonal to the damping direction — the null
component `P_⊥Y`, `P_⊥ = I − DDᵀ/‖D‖²`: the part of the assembly that provably never reaches
`p_L` at θ_max = 0.

```
ϕ_{t,i} = χ_{t,i} · ŵ_i · ‖P_⊥ Y‖²  =  χ_{t,i} · ŵ_i · (Σ_l Y²_{l,t,i} − A²_{t,i}/‖D‖²_i)

χ_{t,i} = sg[ (M² − A²) / (M² + ε_M·‖D‖²_i·(σ̂²_i+ε₀)) ]  ∈ [0,1)   (detached sign-coherence gate)
ŵ_i     = 1 / sg[ ‖D‖²_i · (σ̂²_i + ε₀) ]                            (detached per-channel normalizer)

R_PACT  = mean_{t,i} [ ϕ_{t,i} ] ;      L_train += λ·R_PACT          (training only; logged separately)
```

`sg[·]` = stop-gradient. Gate semantics (D_l > 0 always):

- `χ = 0 ⟺ M = |A| ⟺` all nonzero `Y_l` share one sign ⟺ **no cancellation**. One-hot
  concentration, disjoint band specialization, aligned profiles — all exempt, *exactly*.
  (Without the gate, a one-hot profile pays `c²(1−D_1²/‖D‖²) ≈ c²(1−1/L)` — depth-variance
  taxes concentration and fights specialization. The gate is what makes PACT a cancellation
  tax rather than a homogenization prior.)
- `χ → 1` under pure cancellation (`A = 0, M² ≫ ε_M‖D‖²σ̂²`).
- With minority-mass fraction `f = S⁻/M`: `χ ≈ 4f(1−f)` — a smooth ramp in the opposing-mass
  fraction; no thresholds, no bimodality.
- **Automatic dead-zone:** profiles with negligible damped mass (`M² ≲ ε_M‖D‖²σ̂²`, i.e. total
  mass below ~√L·σ̂) release the gate — PACT never chases noise-level profiles (TROPE's
  floor-chasing failure class, excluded by construction).

Scale-freeness (F3): under a per-channel rescale `Y → cY`: `A → cA`, `M → cM`, `σ̂² → c²σ̂²`
(EMA-converged) ⇒ ϕ and χ invariant; the field (below) scales as 1/c, as the gradient of a
scale-invariant functional must. ρ, σ^p, and every p-statistic appear **nowhere**.

### 5.3 The gradient field — exact, seam-local, function-preserving

χ, ŵ, D detached; differentiate ϕ through both `Y_l` and `A`:

```
∂ϕ/∂Y_l = 2χŵ·(Y_l − A·D_l/‖D‖²)  −  (2χŵ/‖D‖²)·D_l·(DᵀY − A)   =   2χŵ·(Y_l − A·D_l/‖D‖²)
```

(the second term is identically zero). The injected field, with the normalization of §5.2:

```
g_{l,t,i} = (2λ/(T·m)) · χ_{t,i} · ŵ_i · (Y_{l,t,i} − A_{t,i}·D_{l,i}/‖D‖²_i)          (G1)
```

**Exact orthogonality:** `Σ_l D_{l,i} g_{l,t,i} = (2λχŵ/(Tm))·(A − A) = 0`. The field is the
projection flow direction `−P_⊥Y`: it perturbs the damped sum — the increments' contribution to
`p_L` — by exactly zero, per (t,i), per step. A naive damped-variance penalty
(`ΣD²Y² − (ΣDY)²/L`) would miss this by O(1−D²) ≈ 5%; (P1) is constructed so it is exact.

**Hard bound (shipped variant — OBSD lesson).** Huberize the normalized residual
`r = (Y_l − A·D_l/‖D‖²)/σ̂_i`:

```
g = (2λ/(Tm)) · χ · (σ̂_i·ŵ_i) · clamp(r, ±κ)     ⇒     |g_{l,t,i}| ≤ 2λκ / (Tm·‖D‖²_i·max(σ̂_i,√ε₀))
```

This is exactly the gradient of the Huberized penalty (quadratic core, linear tails beyond
κσ̂). Where the clamp binds, the orthogonality identity degrades by at most the clamp-tail mass
(logged; kill-monitored at ≤1% sustained).

### 5.4 Walk integration and memory

- **Forward, per layer (training only):** inside the existing fused masked commit
  (`chiron_scfa_axpy2_masked_dual_p`, which already streams y_par, y_perp, p): two extra FMAs
  per element — `A += D_l⊙(y_par+y_perp)`, `M += D_l⊙|y_par+y_perp|`. The accumulators are
  **write-only side products**: forward outputs (states, logits) are bit-identical with PACT on
  or off. Zeroed at µstep start.
- **Backward inverse walk, per layer (L→1, after a completed forward — A, M final):** state
  reconstruction (`p −= η⊙Y_l`) untouched. At the existing dy hand-off
  (`chiron_incdrop_scale_copy_dual`, which already writes `dy = η⊙dp`): `dy = η⊙dp + g`, with
  g from (G1) using the recomputed clean `Y_l`, the carried A, M, and the D table. **g carries
  no η factor** — the penalty reads clean increments, so its chain rule bypasses the mask (PACT
  works identically with PIED off). Downstream, g flows through the *unchanged* SCFA backward
  into dWo/dWq,k,v/dq exactly as the task adjoint does.
- **σ̂ update:** once per µstep from the recomputed increments ([m] deterministic reduction).
  Penalty value for logging reduced to a scalar during the walk.
- **Memory:** A, M in BF16: 2 × 64 MiB = 128 MiB (+ [L×m]/[m] tables ≤ 1 MiB) against ~0.9 GB
  headroom. BF16 RMW accumulation error over 24 adds ≈ 1–2% relative — acceptable for a
  regularizer *field* (never part of the reversible state); E1 bounds it vs FP64 CPU refs, with
  an FP32-accumulator fallback flag (256 MiB, still within headroom) if the bar fails. **No
  per-layer [T×m] storage anywhere.**

### 5.5 Interaction with the live PIED mask (derived)

PACT is computed on clean Y, so `R_PACT` is deterministic: `E_η[L + λR_PACT] = E_η[L] +
λR_PACT` — no Jensen tax on the penalty, no mask cross-terms, λ fully decoupled from π. The
rejected alternative (penalty on masked Ŷ = ηY) gives `E_η[ϕ_masked] ≈ ϕ_clean +
v·(1−1/L)·ΣY²` — it silently re-adds an isotropic energy tax (double-counting PIED) and
injects mask noise into the field. Residual coupling of the *field* to the *task* gradient: at
the seam, task-dy_l = η_l ⊙ dp_l with `dp_l ≈ D_l ⊙ dp_L` (the p-bus adjoint damps by the same
cos products), so

```
E_η⟨task-dy, g⟩ = Σ_{t,i} dp_{L,t,i} · Σ_l D_{l,i} g_{l,t,i} = 0     (exact through the damping;
                                                                      O(sin θ_max) q-leak residual)
```

PACT's descent is task-neutral in mask-expectation at first order — through WhiSC, not just at
θ = 0. Per-realization fluctuation O(√Var(η)).

---

## 6. The R1 / R4 / R5 derivations

**R1 (backward q-gain bounded independent of ρ).** The penalty's *only* backward entry is the
additive seam field g, hard-bounded elementwise (§5.3). It reaches dq exclusively through
`(∂Y_l/∂q)ᵀ` — the identical operator the task gradient rides; PACT creates **no new backward
path**: no reln-mediated route, no parameter-gradient channel (parameter-free — the SORC killer
`drot_phi` has no analogue). Raw p never enters: the p-relevant quantity A is the increments'
own sum, normalized by the increments' own detached scale σ̂_i and clamped, so the normalized
residual r is invariant under per-channel increment rescaling. Hence the q-subspace backward
gain vs baseline is

```
gain ≤ 1 + max‖g‖ / ‖η⊙dp‖  =  1 + O(λκ)-controlled constant,
```

with **no factor of ρ, σ^p, depth, or step** — calibrated at E2 so the field is 3–5% of task-dy
RMS, i.e. gain ≤ ~1.05. Contrast SORC: its exploding channel was a trunk-coupled, ρ-amplified
*parameter* gradient; PACT's field rides the increment branch whose trunk block is exactly I
(the same structural safety as PIED §6). **[derived; the only assumption is the bounded
`∂Y/∂q` of the flagship's certified regime]**

**R4 (train/inference consistency, exact).** Training loss = NLL + Z-loss + SIRA + λR_PACT.
Validation/inference: R_PACT is not evaluated, and because the forward is unchanged even when
the flag is on, **train and inference compute the identical forward function** — PACT changes
only which weights are learned. No checkpoint delta, no serving change. **[proven]**

**R5 (safe from step 1 and at ρ ≈ 4100).** At init Y → 0: r → 0 (σ̂ floored), the gate's
dead-zone holds (M² ≪ ε_M‖D‖²σ̂²) ⇒ g → 0 — smooth engagement, no warmup. At ρ ≈ 4100: all
PACT statistics live in the increment frame; ρ appears nowhere (F3 met by construction:
detached σ̂ + scale-free ratio). **[proven by inspection of §5.2]**

**R2/R3/R6.** λ = 0: accumulator FMAs and the field term are not dispatched (flag-gated) —
step-1 bit-parity (E0). State reconstruction untouched (the field enters adjoints only, after
p-recovery); no randomness introduced (PIED's hash untouched); A, M are forward-produced
running accumulators — no per-layer storage. Deterministic: elementwise accumulator updates
(one thread per (t,i), no atomics), deterministic [m] reductions, fixed evaluation order.
**[proven]**

---

## 7. Objective — the variational principle

**T1 (diagonal blindness of independent masks) [proven, second order].** For masks η_l
mean-one, independent across layers, per (t,i): logits are affine in η (F2) with
`∂z_t/∂η_{l,t,i} = D_{l,i}Y_{l,t,i}E_{:,i}`, so the second-order expansion of `E_η[L]` contains
`½Σ_{l,l'} E[δη_l δη_{l'}]·H_{ll'} = ½·Var(η)·Σ_l H_ll`, where
`H_{ll'} = D_lD_{l'}·Y_lY_{l'}·Var_{v∼s_t}[E_{v,i}]` is the increment Gram in the softmax-Fisher
metric. Independence kills every off-diagonal term. All cancellation structure (negative
`Y_lY_{l'}`) is off-diagonal ⇒ invisible to the entire independent-mask family at any rate,
granularity, or adversarial reweighting. (Correlated masks could reach the off-diagonal only as
a two-sided term with equicorrelation bounded by −1/(L−1) ≈ −0.04, welded to added noise; the
explicit penalty realizes the gated, one-sided, normalized version with zero variance.)

**T2 (max-SNR assembly / variance-reduction coupling) [derived, tight].** At fixed signal
A_{t,i}, PIED injects terminal noise `Var[δp_L] = v·Σ_l D²_lY²_l` and pays implicit tax
∝ `Σ_l D²Y²·Fisher`. The constrained minimizer of `Σ D²Y²` at `DᵀY = A` is `D_lY_l = A/L`;
PACT's preferred profile `Y* = (A/‖D‖²)D` gives `Σ D²Y*² = A²·ΣD⁴/(‖D‖²)²`, and

```
[ A²·ΣD⁴/(‖D‖²)² ] / [ A²/L ]  =  1 + Var_l(D²)/mean_l(D²)²  ≤  1.0032      (D² ∈ [0.893,1])
```

— PACT-preferred profiles inject at most **0.32% more** PIED-variance than the true minimum
(exactly the minimum at θ_max = 0). So minimizing ϕ minimizes, to within 0.3%, both PIED's
injected noise and PIED's implicit tax at fixed function. **Falsifiable consequence:** under
PACT, the measured PIED Jensen tax Δ_tax (noise-on vs noise-off eval at the same checkpoint)
should decline relative to the PIED-only trajectory — a cross-mechanism observable (E4
secondary).

**T3 (frozen-gate flow) [proven for the quadratic core, per (t,i)].** With χ, ŵ frozen,
descent on ϕ is the projection flow `dY/dτ = −2χŵ·P_⊥Y`:
`Y(τ) = (A/‖D‖²)D + e^{−2χŵτ}·P_⊥Y(0)` — exponential decay of the null component, A invariant,
`‖p_L‖`-contribution invariant, penalty strictly decreasing.

**T3' (minimal intervention — stopping at the coherence boundary) [derived].** Across steps the
gate re-evaluates: the effective flow is the straight-line homotopy `P_⊥Y → 0` at
state-dependent rate ∝ χ(Y). Since sign-coherence (`M = |A|`) is reached strictly before
`Y ∝ D` on that line (whenever A ≠ 0), **the flow terminates at the sign-coherence boundary**:
PACT removes opposition and then releases; it never equalizes magnitudes of an already-coherent
profile. Pure dead-weight profiles (A = 0) decay to the gate's dead-zone mass and release.
Stationary set: {sign-coherent profiles} ∪ {negligible-mass profiles}.

**Interpretations.**
- *Constrained-optimization:* PACT is a Lagrangian relaxation of "assemble each (t,i) of p_L
  with a sign-consistent depth profile" — the constraint set is the union of the coherent
  cones; λ prices distance from it in the σ̂-normalized excess-energy metric.
- *Information-theoretic / MDL:* among assemblies delivering the same message A to the p-bus,
  prefer the minimal-power code; the gate exempts codes that are power-minimal *within their
  sign pattern* (specialization is a legitimate code).
- *Bias–variance:* PIED converts co-adaptation into train-time variance; PACT removes the
  variance-generating configurations themselves, lowering the noise floor PIED's ensemble
  training pays (T2). The two mechanisms are the implicit and explicit faces of one
  regularization program, coupled in the direction that reduces total tax.
- *Limiting cases:* λ→0 ⇒ baseline. χ≡1, ŵ unnormalized, D≡1 ⇒ plain depth-variance
  `L·Var_l(Y_l)` — the deterministic shadow of the mask family's diagonal at uniform Fisher
  (and a known-bad homogenization prior; the gate is the innovation). θ_max→0 ⇒ all statements
  exact.

**Honest scope [heuristic beyond this point].** The analysis lives in increment-function space;
realized updates project the field through shared weights (Wq,k,v,o serve all tokens), so
simultaneous uniformization across (t,i) is generally infeasible — PACT is a soft prior, its
realizable content an empirical question (E3/E4). "Only A matters" is exact only at θ_max = 0:
intermediate prefixes p_l feed q through WhiSC at O(θ_max) per crossing, so some sign-opposition
is *functional* (corrections whose intermediate values are read). PACT prices this honestly:
the gate does not exempt it; λ is calibrated so the field is a ≤5% perturbation of task-dy; the
E3/E4 val gap is the arbiter, and M0 measures the opposing-mass occupancy before any spend.
Channel-diagonal scope: like PIED, PACT sees cancellation per channel i; cross-channel
cancellation through the readout Gram (E_{:,i} ≈ E_{:,i'}) is out of scope for the prototype
(the Fisher-weighted ŵ upgrade, §14.2, is the theory-matched extension).

---

## 8. Dynamics and stack interactions

- **PIED:** complementary by T1 (diagonal/off-diagonal decomposition); coupled in the
  tax-reducing direction by T2; the field carries no η and the penalty no Jensen term (§5.5).
- **SIRA:** SIRA's energy term reads ‖p_L‖², a function of A (plus q-terms); PACT's field
  preserves every A_{t,i} exactly ⇒ **zero first-order interaction** — SIRA governs the sum's
  magnitude, PACT the profile's shape at fixed sum: orthogonal coordinates. (This is precisely
  the degeneracy on which candidate Form A failed — it inflated the aligned sum, colliding with
  SIRA head-on.)
- **WhiSC:** PACT reads θ through the detached D table; it shifts intermediate p_l
  distributions only at second order; the detached whitening EMA adapts as under normal drift
  (`[whisc]` monitor at E3). The D-aware projector makes task-neutrality exact *through* the
  cos-damping (§5.5).
- **Reanchor:** untouched — no reln-path term; q states never touched.
- **Adam:** the field is smooth, dense, deterministic; the gate is continuous in (A, M) (the
  |·| kinks in M affect only the detached gate *value*, and χ is continuous); no bimodality
  anywhere (the LayerDrop channel is empty — nothing is masked, hinged, or thresholded in the
  differentiated path). Second-moment shift O((λκ)²) — negligible at the E2 calibration.
- **Grad-clip 0.5:** field contribution to ‖g‖ bounded by the clamp; expected within noise of
  baseline. If anything, removing large opposing increment pairs should *damp* hard-batch
  spikes — the lineage precedent is PIED's own step-9001 damping (10.8 → 4.9), obtained by
  taxing this same object's diagonal. E3 ‖g‖ census regardless; kill at sustained >1.5×.
- **Depth composition:** the field at layer l depends on the whole profile only through the two
  scalars (A, M) per (t,i) — no geometric depth channel exists (the per-layer field magnitude
  is depth-uniform up to D_l ∈ [0.945,1]).

---

## 9. Theoretical analysis (claim ledger)

- **Proven:** T1; the excess-energy = null-component identity; (G1) and the exact D-weighted
  orthogonality `Σ_l D_l g_l = 0`; T3 (frozen-gate flow); gate exactness (χ = 0 ⟺
  sign-coherent; dead-zone); hard elementwise field bound; forward-function invariance (R4);
  bit-parity/invertibility (R2/R3); determinism (R6); scale-freeness of ϕ, χ, r under
  per-channel rescaling.
- **Derived under assumptions:** T2 and its 1.0032 tightness bound (second order, θ = 0
  channel; GN form — same standing as shipped R_PIED); mask-expectation task-neutrality through
  the WhiSC damping (O(sin θ_max) q-leak residual); R1 gain constant (bounded ∂Y/∂q — the
  flagship's certified regime); T3' (gate re-evaluation across steps); BF16 accumulator error
  bar.
- **Heuristic:** taxed cancellation is mostly co-adaptation rather than computation (the
  WhiSC-read fraction is the exposure — M0/E3 adjudicate); explicit noise-decoupled strength
  beats the v-welded implicit tax; spike-damping improvement.
- **Conjecture:** PACT reduces PIED's measured Jensen tax in vivo (T2's signature, E4
  secondary); profile coherence improves late-training descent in the one-pass regime; the win,
  if any, is late-position-bucket-loaded (absorbed TROPE discipline: pre-registered signature,
  §13 E4).

---

## 10. Computational tradeoffs

| | PACT (gated, prototype) | ungated fallback (`--pact-gate 0`) |
|---|---|---|
| Randomness | none (deterministic) | none |
| Accumulators | A, M BF16 [T×m] ×2 = 128 MiB + tables ≤ 1 MiB | A only, 64 MiB |
| Extra DRAM/µstep | ~9 GB (fwd RMW + bwd reads), fused into seam kernels ≈ 1.4% of step traffic | ~4.5 GB |
| New GEMMs | 0 | 0 |
| Kernels | 4 (2 fused variants of existing seam kernels + stats + D-table) + CPU refs | 3 |
| Wall estimate | **~1.0–1.5%** | ~0.7% |
| Checkpoint / serving delta | none / none | none / none |

Fallbacks if E2 perf-bench exceeds 1.5%: ungated variant; every-other-layer M updates (the
detached gate tolerates approximation); FP16 accumulators.

---

## 11. Comparison to existing CHIRON mechanisms

| | SIRA | PIED | mask-family follow-ups (π/symmetric/per-head/spectral/DRO) | **PACT (this)** |
|---|---|---|---|---|
| Object | terminal ‖(q_L,p_L)‖ | increment energies (Gram **diagonal**) | Gram diagonal, reweighted | **Gram off-diagonal (sign structure)** |
| Action | soft penalty on the sum | mean-one noise | mean-one noise, reshaped rates | **deterministic gated penalty on the assembly** |
| Sees cancellation? | no (magnitude only) | only via diagonal shadow | no (T1) | **directly, one-sided, gated** |
| Strength coupling | free λ | welded to π (Jensen tax) | welded to rates/budget | **free λ, zero tax** |
| Function distortion | first-order (by design) | zero-mean noise | zero-mean noise | **exactly zero at first order (Σ D_l g_l = 0)** |
| Params/ckpt | none | none | none | **none** |
| Outcome | shipped −0.03 | shipped −0.35 | untested | **predicted M0/E3-pass; E4 = the experiment** |

PACT is to PIED what PIED was to LayerDrop: the same intent (price co-adaptation on the p-bus),
rebuilt on the architecture's invariants — here, the linearity of the p-bus made *exact* by the
damped-sum accumulator, and the WhiSC damping made exact by the D-aware projector. T1 is the
formal statement that this step cannot be taken from inside the mask family.

---

## 12. Failure modes and mitigations (ranked)

1. **Functional cancellation is load-bearing** (WhiSC-mediated corrections larger than
   estimated): val regresses despite falling penalty. *Caught:* E3 val gap + trajectory rule;
   position-stratified buckets (a late-bucket regression is the mirror of the predicted win —
   direction, not just magnitude, adjudicates).
2. **Lever not live** (cancellation occupancy already small — PIED mopped it up): penalty mass
   tiny; everything neutral. *Caught:* **M0 before any training spend** (kill < 3% of increment
   energy); E3 requires the penalty value to fall ≥30% from its step-100 level (engagement
   proof).
3. **Capacity drain / over-taxing inside the gated cone** (λ too big): train NLL flat, val
   worse, depth-Herfindahl collapses. *Caught:* E2 λ calibration (field ≤ 5% of task-dy); E3
   Herfindahl + val.
4. **Clamp-tail bias** breaks A-preservation where the Huber binds. *Caught:* clamp-rate log;
   kill > 1% sustained.
5. **σ̂ / gate pathologies at init or under EMA lag** (rapid scale change transiently
   mis-scales r). *Caught:* E1 synthetic tests at ρ = 45; E3 first-100-step field/‖g‖ trace;
   the clamp bounds the worst case.
6. **BF16 accumulator error** biases the field. *Caught:* E1 bound vs FP64 CPU ref (bar ≤ 1e-2
   relative field error); FP32 fallback flag.
7. **Displacement** (cancellation re-expressed via the O(θ) q-route): coherence observables
   improve, val flat, [whisc]/q-energy stats shift. *Caught:* E4 secondaries; closes honestly
   as "shape moved, not removed".
8. **Wall > 2%.** *Caught:* E2 bench; fallbacks in §10.

---

## 13. Minimal prototype + the pre-registered ladder

**Build = PACT, per-(t,i) profiles, gated, Huberized, default-off.**
Library (glades-ml): fused accumulate variant of the masked commit (`chiron_pact_accum` folded
into `chiron_scfa_axpy2_masked_dual_p`'s stream), fused field variant of the dy hand-off
(`chiron_pact_grad` folded into `chiron_incdrop_scale_copy_dual`), `chiron_pact_stats` (σ̂ EMA),
`chiron_pact_damp_table` (suffix scan over θ); CPU references for all; unit selector
`test.sh chiron-pact`. Trainer (glades-trainer): `--pact-coef <λ>` (default 0.0),
`--pact-gate {0,1}` (default 1), `--pact-clamp κ` (default 4). No new params, no checkpoint
bit, no serving change; σ̂ is uncheckpointed transient state (documented ~10-step resume
re-warm). **Build discipline:** the trainer links the glades CUDA kernels statically — rebuild
the trainer (`bash build.sh`) after any kernel change; `make install` alone is not enough.

**Validation ladder (pre-registered):**

- **M0 — measurement gate** (~minutes GPU, before full implementation; env-gated probe in the
  trainer forward, pq-ratio-probe precedent): one eval pass over `chiron_1B_pied_e4.final`;
  record per-(t,i) A/M, the χ distribution, gated excess share of total increment energy,
  per-layer sign-opposition occupancy. **Kill: gated share < 3%** ⇒ lever not live; close the
  arc for the cost of a probe. *Prior: ~70% pass* (PIED's in-vivo spike damping implies live
  opposing mass).
  **RESULT (2026-07-04): GO — aggregate gated share 72.01% (8 val windows, 71.3–73.5%), 24×
  over the bar; occupancy rises with depth (16%→47%); A-vs-p residual ~0.50 = the in-vivo
  WhiSC q→p leak (sharpens failure mode #1; the field-RMS λ rule is the binding control).
  Record: `research/CHIRON_PACT_M0_2026_07_04.md`.**
- **E0** — `--pact-coef 0`: step-1 bit-identical to the PIED flagship path. *Must pass.*
  **RESULT (2026-07-04): PASS** — flag-off eval on `chiron_1B_pied_e4.final` reproduces the M0
  baseline val NLL exactly (1.0891 / 1.1174 / 1.1909, 8-batch windows). All PACT training-path
  code gated on `cfg.pactCoef != 0.0f`.
- **E1** — units: CPU ≡ GPU field parity; FD gradient checks including the A-coupling at
  synthetic ρ = 45; `Σ_l D_l g_l = 0` to FP tolerance (unclamped); Huber-consistency (clamped
  field = gradient of the Huberized penalty); BF16-accumulator error bar (≤ 1e-2 relative vs
  FP64); gate values on constructed profiles (one-hot ⇒ χ = 0; pure cancelling pair ⇒ χ ≈ 1;
  dead-zone release); inverse-walk reconstruction bit-parity with PACT ON; determinism replay.
  **RESULT (2026-07-04): PASS** — `test.sh chiron-pact` (4 tests, 0 asserts failed): orthogonality
  residual 5.7e-7, FD-vs-analytic 4.2e-5, GPU field bit-exact vs CPU, Huber clamp count exact,
  gate goldens, coef=0 numeric identity, degree(+1) scale covariance. chiron-pied / chiron-whisc
  regressions green.
- **E2** — small shape + calibration: set λ so field RMS = 3–5% of task-dy RMS at step ~1k
  (pre-registered candidates λ ∈ {3e-3, 1e-2}); numerically verify mask-expectation
  task-neutrality `E⟨task-dy, g⟩ ≈ 0`; penalty trajectory decreasing on a 500-step probe; perf
  bench ≤ 1.5%. **Kill:** field/dy ratio uncontrollable, or wall > 2%.
  **RESULT (2026-07-04): PASS, with a field-scaling refinement of record.** The §5.2–5.3
  **scale-free** penalty `ϕ = χ‖P_⊥Y‖²/(Dsq·σ²)` has a scale-**divergent** gradient (∝ 1/σ →
  field/dy = 5.4 at fresh init). **Fix (superseding §5.2–5.3 on this point only):**
  increment-**energy** scaling `ŵ = 1/Dsq` (drop the 1/σ²), giving
  `g = coef·χ·σ·clamp(res/σ,±κ)/Dsq ~ O(‖increment‖)` — bounded, vanishing at init, still ρ-free
  (σ = Huber knee only); degree(+1) homogeneous. Orthogonality, the gate, and function-preservation
  are unchanged. Also: trainer normalization `2λ/(Tm) → 2λ/T` (token-mean). Calibrated **λ* = 3e-3**
  (fresh-init field/dy ramps 1.1%→2.4% early → 3–5% band mid-training; 0 grad-skips; clampRate
  <3%); λ=1e-2 ruled out (would exceed target). Full record:
  `research/CHIRON_PACT_E0_E2_2026_07_04.md`.
- **E3** — matched 2500-step pair at T = 16384, **fresh same-binary baseline arm mandatory**
  (era-drift 0.10–0.15 nat dwarfs 2500-step effects), flagship recipe ± `--pact-coef λ*`
  (~7 GPU-hr both arms). Predictions: 0 skips; ‖g‖ max ≤ 1.1× baseline arm; penalty falls ≥30%
  from step-100 level; val gap ≤ +0.02 and shrinking across {1k, 1.5k, 2k, 2.5k}; clamp rate
  < 1%; `[whisc]` nominal. **Kill:** any skip wave; ‖g‖ > 1.5×; val gap > +0.04 or
  non-shrinking; penalty not decreasing. One retry at the other λ permitted.
- **E4 — matched 30k, decisive, PAIRED same-binary arms** (PIED recipe ± `--pact-coef λ*`,
  seed 1337, ~44 GPU-hr total). Unlike PIED's E4 (margin 3–5× era drift), the expected effect
  here is near the ±0.02 bar, so reusing the archived flagship run as baseline is **not
  admissible** — the 4-batch-val-jitter lesson makes the fresh paired arm mandatory. **Ship:**
  Δ ≤ −0.02 nat (4-batch final-val) confirmed in direction on wide-32 matched windows, 0
  skips, wall ≤ +2%, TF-parity via chiron_infer pre-ship. **Secondaries (banked regardless):**
  PIED Jensen-tax trajectory (T2's signature); sign-coherence occupancy; depth-Herfindahl
  (absorbed AVEC observable); position-stratified buckets (pre-registered signature: win
  late-bucket-loaded). **Neutral** (−0.02, +0.02): keep default-off; if coherence observables
  moved but NLL didn't, the one follow-up worth an E3 is the Fisher-weighted ŵ (§14.2). **Kill:**
  Δ ≥ +0.02.

**Pre-registered priors:** conditional on M0+E3 pass — win ~30%, neutral ~45%, regress ~25%;
unconditional ~22/50/28. Bull case: T1 says the axis is untouched by the entire mask family,
and the lineage's largest regularization win (−0.35) came from this object's diagonal shadow.
Bear case: PIED may already have shrunk the taxable mass (M0 prices this first, for pennies),
and the one-pass regime historically pays penalties less than conditioning fixes. The asymmetry
— measurement-first, deterministic, parameter-free, ~1% wall, cheap E3 — prices the probe well
below its information value either way.

---

## 14. Full research program

1. **PACT M0/E0–E4** (above) — adjudicates the off-diagonal question.
2. **Fisher-weighted ŵ** — replace the isotropic per-channel normalizer with a detached EMA of
   the readout Fisher diagonal `Var_{v∼s_t}[E_{v,i}]` (AVEC's φ̂ probe design, O(T_s·V·m)/K
   amortized): matches the penalty metric to R_PIED's exactly and partially extends scope
   toward cross-channel (logit-space) cancellation. Only after a fixed-ŵ result exists.
3. **AVEC** — the DRO reallocation of PIED's rates; run only if PACT's M0/observables show the
   *diagonal* is also mis-allocated (its degeneracy analysis makes it cheap to pre-screen: the
   class-energy Herfindahl `C_F ≈ 1` ⇒ don't run).
4. **TROPE** — the token-time off-diagonal-purity penalty on y_compr; an independent axis,
   unblocked by PACT's outcome; its E1 measurement probe (trained coherence vs analytic floor)
   is near-free and can piggyback on M0 instrumentation.
5. **Correlated-mask PIED** — the stochastic realization of a (weak, two-sided) off-diagonal
   tax via negatively equicorrelated masks (bounded by −1/(L−1)); only interesting as a
   mechanism-separation control if PACT wins (does the *implicit* version of the same pressure
   reproduce it?).
6. **Generation hook** — if E4 secondaries move (coherence up, repetition metrics down), PACT
   becomes a component of the deferred generation-training plan; never a perplexity-ship claim.

---

## 15. Open conjectures and validation criteria

- **C1 (hard, falsifiable):** PACT does not destabilize the flagship recipe (E3: 0 skips, ‖g‖
  within baseline envelope). *Falsifier:* any skip wave — which would indict the seam-injection
  safety argument (§6) and close the explicit-penalty family at this seam.
- **C2 (the experiment):** the gated off-diagonal tax buys val NLL at 30k (E4 ship bar
  Δ ≤ −0.02, paired arms). *Falsifier:* E4 kill/neutral with observables also flat.
- **C3 (mechanism):** sign-opposition occupancy falls ≥ 30% and depth-Herfindahl of per-layer
  contributions is stable-or-lower at matched NLL — the signature that the penalty reshapes the
  assembly without homogenizing it. *Falsifier:* occupancy flat under a falling penalty value
  (the model is paying the tax rather than restructuring — λ mis-calibrated or opposition is
  functional).
- **C4 (cross-mechanism, distinctive):** PIED's measured Jensen tax Δ_tax declines under PACT
  relative to the PIED-only trajectory (T2's in-vivo signature). *Falsifier:* Δ_tax unchanged
  while occupancy falls — would mean the taxed configurations were not the noise-dominant ones,
  weakening T2's practical relevance while leaving T1 intact.
- **C5 (signature):** any E4 win is late-position-bucket-loaded. *Falsifier:* uniform or
  early-bucket gains — would indicate the mechanism works through a different channel than the
  temporal-differentiation story, prompting a mechanism re-read before any follow-up.

**Engineering discipline (project convention):** default-off, gated, no checkpoint delta, E0
bit-parity when off; kernels + CPU refs + unit tests land regardless of outcome; M0 and E3 are
the cheap decision points before any 30k spend.
