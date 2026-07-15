# CHIRON Loss-Regularizer Design Study — ECHO / BERM / LOFT (2026-07-09)

**Status (2026-07-15): ECHO IMPLEMENTED through engineering gates E0–E2; E3/E3.5/E4 training and generation gates remain unrun, so this is not a ship claim.**
The original 2026-07-09 design study ran no GPU experiments. Implementation now spans
`glades-ml` CPU/CUDA kernels and unit tests plus `glades-trainer` CLI/training wiring,
telemetry, wrapper routing, and a one-step end-to-end smoke (implementation record §6).
Owner brief: "design several regularization terms we can add to our loss."
Method: research-framework-design 3-candidate protocol (the PIED/SIPHON precedent) — three
independent theory agents developed materially different formulations from the identical
parsed problem statement; the orchestrator verified the central derivations by hand
(one correction found and recorded, §4.3), compared, selected, and fully developed the
strongest. Baseline anchor: the PIED flagship (`chiron_1B_pied_e4.final`, wide-32 val
1.1788/1.3019), recipe as in CLAUDE.md.

---

## 1. Executive summary

Three new regularization terms are fully designed, on three disjoint sites with three
distinct mechanism classes. All are default-off, parameter-free, training-only (no
checkpoint/serving delta), one-sided (exactly zero in their healthy regions — the
structural answer to the PACT washout), and ≤ ~0.5% wall on already-materialized
quantities.

| | Site | Object | One-line thesis | Honest modal outcome |
|---|---|---|---|---|
| **ECHO** (selected) | readout softmax/LSE | conditional next-token measure π_t vs the trailing window's empirical measure | hinged, margin-calibrated, truth-gated tax on *excess copy mass* — the TF-trainable boundary of the diagnosed repetition attractor | NLL-neutral; large TF copy-mass reduction; moderate nucleus-generation gain |
| **BERM** | shear commit (p-bus) | participation ratio of whitened per-channel increment energies per layer | one-sided barrier against channel-energy collapse; identically zero when healthy; collapse insurance + free PR diagnostics | park-at-shadow ~0.55; EV concentrates in long runs (FineWeb 305k) |
| **LOFT** | y_compr [k×m] spectral buffer | occupancy fraction of the long-period (≥512-token) band of the depth-aggregated increment spectrum | deterministic occupancy *floor* — the one instrument the mean-one mask family (PIED/SFD) provably cannot express; long-context-utilization lever | null-by-inactivity ~0.5; win channel = late-position val |

**Selection: ECHO.** It is the only candidate whose target pathology is *observed and
diagnosed* in this model (the 2026-06-27 repetition-attractor record) rather than
conjectured; it is the cheapest (≤0.3% wall); its NLL risk is bounded and λ-controlled;
and it is the first rung of the deferred "generation-aware training" plan — the
flagship's single standing caveat — at near-zero cost to the ship metric. BERM and LOFT
remain fully specified portfolio items: BERM re-scoped as cheap insurance/diagnostics for
the pending FineWeb 305k run; LOFT parked behind its M0 occupancy probe.

**Review correction (orchestrator, pre-registration honesty):** ECHO's original "sign
theorem" (its own gradient step strictly decreases TF NLL on every active position) is
FALSE as stated — it drops the log-partition coupling. The corrected Directional-NLL
Lemma (§5.3) is weaker: strictly helpful in the modal-copy pathological regime, bounded
`|δ log π(y)| ≤ ηλ·P_A` of either sign in mixed regimes. NLL safety rests on the bound ×
gate sparsity, not on a uniform sign.

---

## 2. Formal problem statement (shared by all candidates)

Per token t ∈ {1..T}, phase state (q_t, p_t) ∈ ℝ^m × ℝ^m (m=2048, L=24, T=16384,
nH=16, dH=256, V=32000). Layer-l update: `p += Y_l(q)` at the shear commit
(`chiron_scfa_axpy2_masked_dual_p`; PIED mask η, training only), `q → ReLN(q)`;
WhiSC-D whitened rotation with detached per-channel EMA stats a_{l,i}; SCFA compresses
the token dimension by a fixed DCT-II basis to k=1024 rows (y_compr [k×m] materialized
per layer; kept band = token-periods ≥ 32); attention folds p→q only at layer L.
Trained-in p²/q² ~ 2000+. Existing loss: CE + Z-loss(1e-4·mean logZ²) + terminal SIRA
(1e-2; energy 1.0 / balance 0.25 / action 0). Recipe: 30k steps, constant lr 3e-4,
accum 4, clip 0.5, seed 1337.

**Design target:** an additive term λ·R (default-off flag) that plausibly improves
wide-32 val NLL at 30k and/or the documented free-generation repetition attractor
(loop periods 2–6; π(copy) → 0.95+ with logit-climb 8.6→38; nucleus top-p 0.95 cannot
escape; NOT inference-fixable) — without regressing NLL.

**Hard constraints:** training-only; wall ≤ ~2%; no extra full passes; FP32 warp-shuffle
reductions, no per-step global double atomics (the PACT −40% bug class); gradient
bounded O(‖activation‖), non-divergent at init (scale-free penalties have ∝1/σ gradients
— PACT E2); anything cross-depth acts in the whitened frame from step 1.

**Binding graveyard:** PACT (always-on gated structural penalty, 72% gate occupancy at
the healthy optimum → early −0.3..−0.5 gain washed out to +0.026 WORSE at 30k); SIRA
(−0.031, small); PHS (servo, rejected); PTOC (shadow); PIED (implicit Fisher
increment-energy, layer-diagonal); WhiSK/SFD (rejected design-study candidates; SFD's
honest finding: loop periods 2–6 lie outside the DCT grid, so no band-content
anti-repetition story is admissible); SIPHON/CHARGE (closed); LayerDrop/UL2/MTP (port
failures); OBSD/SORC (unwhitened couplings). Track record: big wins were conditioning
(QK-Norm −0.97) and stochastic ensembles (PIED); explicit penalties small or zero — every
candidate must confront this.

---

## 3. Candidate formulations

### 3A. ECHO — Excess-Copy Hinged Objective (measure-evolution / information-geometric, readout)

**Primitive objects.** x_1..x_T ground-truth ids; y_t = x_{t+1} the TF target; z_t ∈ ℝ^V
logits (materialized bf16 + FP32 LSE_t from `softmax_forward_bf16_with_lse`);
π_t = softmax(z_t); trailing window W_t = {x_s : max(1,t−w+1) ≤ s ≤ t}, w = 128
(t < w uses the available prefix); counts n_t(v), window-empirical measure
ν_t(v) = n_t(v)/w.

**Margin (the legitimate-reuse allowance):** `m_t(v) = κ·ν_t(v) + τ₀` (defaults κ=2,
τ₀=0.1). No z-dependence.

**Gate + active set:** `A_t = { v : n_t(v)>0, v ≠ y_t, π_t(v) > m_t(v) }` — window
tokens, excluding the truth, on which the model over-bets beyond margin.
`P_{A,t} = Σ_{v∈A_t} π_t(v)`.

**Functional:**

```
L = CE + ζ·mean(logZ²) + (λ/T)·Σ_t R_t,     R_t = Σ_{v∈A_t} (π_t(v) − m_t(v))_+
```

R_t is piecewise-linear and convex in π_t. Exact logit gradient (m and the gate carry no
z-dependence; ∂π_v/∂z_u = π_v(δ_uv − π_u)):

```
∂R_t/∂z_t(u) = π_t(u)·(a_t(u) − P_{A,t}),        a_t(u) = 1[u ∈ A_t]
```

i.e. a per-row scalar × probs (the exact algebraic shape of the shipped z-loss
correction) plus a sparse scatter on ≤ w window ids. Identities: Σ_u ∂R/∂z_u = 0 (shift
invariance); row-ℓ1 = 2P_A(1−P_A) ≤ 1/2 ⇒ ECHO's worst-case row norm ≤ λ/4 of CE's.
Hinge subgradient: closed-inactive at the knee; optional Huber knee δ=0.01.

**Mechanism.** One-sided barrier: zero value AND zero gradient unless the model holds
over-margin mass on a *non-true* window token. Legitimate repetition untaxed two ways:
(i) the truth is gate-excluded — a correct repeat prediction is never penalized;
(ii) high-frequency window tokens get allowance ∝ their local base rate ("the": margin
≈ 0.26; a content token seen once: ≈ 0.12). The pathology signature (π(copy) → 0.95+ on
a lag-2..6 token the data contradicts) sits far above every margin.

**At init:** π ≈ uniform ⇒ π(v) ≈ 3×10⁻⁵ ≪ τ₀ ⇒ A_t = ∅ ⇒ R ≡ 0 exactly. The hinge is
its own warmup.

**Regularized optimum (single active token v, population limit, gate-aware):** the gate
excludes v when y = v (probability p_v), so stationarity gives

```
π_v* − p_v + λ(1−p_v)·π_v*(1−π_v*) = 0   while π_v* > m_v
```

— downward shrinkage of over-margin repeat probabilities with relative bias
≤ λ(1−p_v)/4, **vanishing as p_v → 1** (deterministic legitimate repeats asymptotically
untouched), switching off below margin. Elsewhere π* = CE optimum exactly.

**Limiting cases (exact reductions; the family ∇_z = π ⊙ (φ − ⟨φ⟩_π)):** unlikelihood
training = (τ₀,κ)→0 + hinge→log-barrier (divergent slope as π→1, always-on — the port's
NLL tax is exactly what the margin+hinge remove); confidence penalty = support→V, gate
off, weight −log π (the isotropic version; ECHO is its anisotropic, data-gated
restriction); label smoothing = uniform-pull, two-sided, always-on (not recovered except
degenerately); Z-loss = the shift mode, to which ECHO is exactly orthogonal
(Σ_u g_u = 0).

**Cost/site:** Phase A `echo_repeat_stats` (per-token shared-mem window histogram +
≤w-logit gather + FP32 hinge eval; warp-shuffle partials, no global atomics; 8.4 MB int
scratch); Phase B extend the per-row scalar already applied by
`softmax_cross_entropy_bwd_bf16_zloss` with −(λ/T)·P_{A,t}; Phase C row-disjoint sparse
scatter `dz[u] += (λ/T)π(u)` on active ids. O(T·w) work, V-free. **≤ 0.3% wall.**

**Flags:** `--echo-coef` (0=off), `--echo-window` (128), `--echo-margin` (0.1),
`--echo-kappa` (2), `--echo-huber` (0), `--echo-warmup` (0).
**Variant (reserved):** ECHO-lag — hinge weight ω(v)=1/ρ_min(v) (inverse minimal lag),
concentrating pressure on the diagnosed period-2..6 band; identical gradient structure.

**Failure modes (ranked):** TF-effective/generation-inert ~35% (detect: E[P_A] drops
≥50% but --gen-metrics loop rate unchanged); late calibration tax ~20% (activation rate
>10% at 25k+, repetitive-region val regress); sub-margin ranking evasion ~15% (π(copy)
piles at margin; greedy/nucleus divergence — partially acceptable, nucleus is the
realistic decoding); gate starvation where truth=repeat ~10%; optimization interference
~5%.

### 3B. BERM — Barrier on whitened Energy-Rank Minima (symplectic/dynamical, p-bus)

**Objects.** Clean (pre-mask) increments Y_{l,t,i} at the shear commit; detached WhiSC
whitening a_{l,i}; live whitened channel energies
`e_{l,i} = ε_e + (1/T)Σ_t (a_{l,i}Y_{l,t,i})²`; S1=Σe, S2=Σe², ê=e/S1, ē=S1/m;
**participation ratio** `PR_l = S1²/S2 ∈ (0,m]`, r_l = PR_l/m. Clean-not-masked is
decided: PR is degree-0 so PIED's uniform 1/(1−π) energy inflation cancels exactly in
the statistic; clean gives a deterministic field with zero Jensen coupling.

**Functional (hinge², energy-scaled per the PACT E2 fix):**

```
h_l = [ρ_l^min − r_l]_+ ;   R = (1/L)·Σ_l sg[ē_l]·h_l² ;   L += λR
```

**Exact gradient** (quotient rule ∂PR/∂e_i = (2S1/S2)(1 − PR·ê_i); identity
ē·S1/S2 = r):

```
g_{l,t,i} = −(8λ/(LmT)) · clamp(h_l·r_l·(1 − PR_l·ê_{l,i}), ±κ_c) · a²_{l,i} · Y_{l,t,i}
```

— an elementwise, sign-preserving multiplicative rescale of the increment itself,
strictly O(‖Y‖), vanishing identically when feasible. Below-mean channels are amplified,
heavy channels shrunk; flow stops exactly at the floor (C¹ at the boundary). At init
r ≈ 1 − 2/d − 2/T ≈ 0.999 (random-increment derivation) ⇒ h = 0 ⇒ g ≡ 0 exactly; armed
from step 1, force-free from step 1.

**Constrained view:** quadratic exterior penalty for `min CE s.t. r_l ≥ ρ_l^min`. If the
data optimum is interior, KKT multipliers are zero and the penalized stationary set
coincides *exactly* with the unpenalized one — 0% occupancy at a healthy optimum, vs
PACT's 72%. Causal channel = trajectory shaping (catching collapse transients) +
deterrence.

**Floor calibration (shadow-first):** theory anchor r_init ≈ 0.999; S0 shadow probe of
the flagship checkpoint + E3-baseline trajectory (`--berm-shadow`, zero gradient); floor
rule `ρ_l^min = min(0.5·PR_l^med/m, cap)` set once, frozen. **Not PHS:** PHS was a
closed-loop servo (EMA → live coefficients each step); BERM's threshold is fixed once
offline — in-run feedback gain exactly zero (same epistemic class as PACT's λ*
calibration or LR sweeps). Fallback if the owner rules otherwise: measurement-free
absolute floor ρ_abs = 0.05.

**Invariances:** channel-permutation invariant (permutation gaming impossible); degree-0
under uniform bus rescale (the p²/q² asymmetry direction is structurally invisible);
per-channel individual rescaling deliberately NOT invariant — that is the signal,
measured in the whitened (geometric-mean) frame.

**Known gap (flagged):** energy-PR is a diagonal proxy — correlated channels can hold
functional rank low while energy PR is high. Audit statistic (log cadence only):
eigen-PR tr(G)²/‖G‖_F² of the increment Gram. Depth-coherence gap: per-layer PR bounds
the terminal profile only if collapse is depth-coherent; the shadow logs terminal-p_L
channel-PR too.

**Cost/site:** statistic fused into the commit kernel (one FMA + the shipped
WhiSC-stats reduction shape: warp-shuffle → one FP32 atomic per (block,channel));
[L×m] FP32 persistence = 196 KB; backward one FMA at the existing
`chiron_incdrop_scale_copy_dual` hand-off. **≤ 0.5% wall.**
**Flags:** `--berm-coef`, `--berm-floor[-file]`, `--berm-clamp` (4), `--berm-shadow`.

**Failure modes:** never binds → park (~0.5–0.6, the modal outcome; adjudicated at S0
for minutes of GPU); floor-high → chronic activity → PACT-class interference (~0.15;
kill at sustained >5% active fraction after 2k); diagonal-PR gaming (~0.10 | eigen-PR
audit); statistic noise (~0.05–0.10); perf (low). Priors: null ≈ 0.55 / small win
≈ 0.15–0.20 / regress ≈ 0.15. Candid value: near-zero interference + collapse insurance
whose EV concentrates in *long* runs (FineWeb 305k has 10× the exposure) + permanent PR
diagnostics.

### 3C. LOFT — Long-period Occupancy Floor Term (operator-theoretic, SCFA spectrum)

**DCT convention (verified in source, `gpu_kernels.cu:9921`):** B[t,j] =
α_j·cos((2t+1)jπ/(2T)) — row j has token-period P_j = 2T/j; the kept k=1024 rows are the
LOWEST frequencies, spanning P ∈ [32, ∞); j=0 is DC (excluded from the statistic —
function-critical and the cheapest gaming mode).

**Objects.** Row energies s_{l,j} = Σ_i y_compr[j,i]²; **depth-aggregated** spectrum
s_j = Σ_l s_{l,j} (aggregate by design: per-layer floors would be a uniformity prior
across depth — the PACT trap; the aggregate floor is satisfiable by any covering subset
of layers, so specialization is free); long-range band 𝔅 = {1 ≤ j ≤ 64} (P ≥ 512);
occupancy F = S_𝔅/(S+ε); detached scale σ̄² = detach(S)/(L(k−1)m).

**Functional and field:**

```
h = [τ − F]_+ ;   R = λ·h²·σ̄²
g_l[j,i] = −c·(𝟙_𝔅(j) − F)·y_l[j,i],   c = 4λh/(L(k−1)m) ≥ 0
```

— a uniform multiplicative subsidy +c(1−F) on in-band coefficients, weak shrink −cF
out-of-band; |g| = O(|y|) with absolute multiplier (the detached σ̄² cancels the 1/S of
the scale-free form exactly — the PACT init-divergence mode is closed by construction).
Backward: one elementwise FMA at the existing y_compr adjoint seam; zero new [T×m]
traffic. Regularized object: pre-mask y_compr.

**The starvation hypothesis (assumption of record):** mutual information in text decays
as a power law (Lin & Tegmark); gradient signal for period-P structure arrives rarely
(∝ I(P)) and incoherently (T/P independent within-window contributions), so long-period
occupancy is optimization-starved in a 30k single-epoch run while the near-cutoff end
gets dense signal. LOFT holds the band open (a floor, not a target) so sparse long-range
gradient has standing structure to accumulate into; if structure forms, F rises and the
hinge self-releases. Explicitly NOT an anti-repetition claim (SFD's finding inherited:
loop periods 2–6 are outside the grid).

**The core differentiation (why deterministic-floor ≠ stochastic-mask):** a mean-one
mask family's expected objective adds `½·Var(ξ)·Σ_j y_jᵀH_j y_j` — a non-negative,
occupancy-proportional TAX with zero first-order drift; **no mean-one mask can implement
a floor** (a first-order subsidy toward under-occupancy). If the band is healthy, LOFT
is exactly zero where SFD still pays its Jensen tax — deterministic-one-sided dominates
the stochastic mask on both branches of the hypothesis. This is also why SFD
self-predicted neutrality: taxing an already-thin band is a no-op.

**Init (white-noise model):** orthonormal B ⇒ flat spectrum ⇒ F_init ≈ 64/1023 ≈ 0.063;
real init (near-uniform attention ⇒ near-DC increments) plausibly higher. Hinge
inactive at init with any calibrated τ, and the field additionally vanishes with σ̄².

**Gaming mode (analyzed honestly):** the functional cannot distinguish informative from
noise occupancy. Pre-registered secondary statistic: band informativeness
`I_𝔅 = 1 − ‖E_seq[y_𝔅]‖²_F / E_seq[‖y_𝔅‖²_F]` (position-locked noise ⇒ I_𝔅 → 0) +
inference-time band-ablation ΔNLL (informative occupancy ⇒ ablation hurts late
positions). Gaming = F↑ with I_𝔅↓ and ablation flat. Variant: three octave sub-floors
(𝔅₁=[1,8], 𝔅₂=(8,32], 𝔅₃=(32,64]) close the edge-gaming route.

**Cost/site:** `chiron_loft_band_stats` FP32 warp-shuffle row reduction over [k×m] per
layer (~2.3 GB/step traffic ≈ ~0.4–0.6% wall); `chiron_loft_finalize` tiny; no D2H in
the step path. **Flags:** `--loft-coef`, `--loft-tau`, `--loft-band-hi` (64),
`--loft-octave`, `--loft-shadow`.

**M0/kill:** probe the shipped checkpoint: predict F ∈ [0.02, 0.12], flat-or-declining
trajectory; kill before any training if F > 0.15 (not starved) or F rising (hypothesis
false). Calibrate τ = min(1.25×median shadow F, 0.10). Priors: ~45–50% null,
~20% small win (late buckets −0.02..−0.05), ~15% neutral-with-signal, ~15–20% regress.

---

## 4. Selection

### 4.1 Comparison

| Criterion | ECHO | BERM | LOFT |
|---|---|---|---|
| Target pathology exists? | **OBSERVED** (2026-06-27 diagnosis: loops, logit-climb, nucleus-proof collapse) | conjectured (channel collapse never observed here) | conjectured (band starvation unmeasured) |
| P(mechanism live at safe calibration) | high (activation 2–5% predicted; the excess-copy mass is documented) | ~0.4 (park-at-S0 modal) | ~0.5 (null-by-inactivity modal) |
| Ship-metric risk | bounded: row-ℓ1 ≤ λ/4 of CE worst case; λ-controlled; §5.3 lemma | ≈0 when feasible (exact zero) | ≈0 when feasible; subsidy field most preference-like when active |
| Ship-metric upside | ~0 (honest) | 0..−0.05 conditional on binding | ~20% small win via long-context |
| Program value beyond NLL | **attacks the flagship's standing caveat** (perplexity-not-generator) — rung 0 of the deferred generation-aware training plan | collapse insurance + PR diagnostics; EV peaks on FineWeb 305k | long-context utilization axis; occupancy diagnostics |
| Wall | ≤0.3% | ≤0.5% | ~0.5% |
| Port-risk class | none (readout-side; architecture-agnostic site) | none (increment branch, trunk=I) | none (spectral buffer) |
| Scientific weak link | TF↔free-running transfer (~35% inert) | collapse may never occur | starvation hypothesis unverified |

### 4.2 Dominant tradeoffs and decision

BERM and LOFT are *insurance instruments*: superbly safe (exact-zero feasible regions),
but their honest modal outcome is that the guarded pathology is absent and the arc parks
at a shadow probe. That is cheap and worth doing opportunistically — but it is not the
strongest first experiment. ECHO is the only candidate aimed at a pathology this program
has already *measured*, with an instrument whose activation is near-certain, whose cost
is the lowest, and whose failure would itself be informative (it would localize the
attractor strictly off-distribution, sharpening the deferred generation-aware-training
scope). The lineage's explicit-penalty track record (SIRA −0.031, Z-loss ~0) is
confronted rather than contradicted: ECHO does not claim the penalty family will buy
nats; it claims the family can buy a *different, documented observable* at ~zero nat
cost — something no shipped mechanism has attempted.

**Selected: ECHO.** BERM: hold for the FineWeb long-run era (its own analysis says
that's where its EV lives); its S0 shadow probe (minutes of GPU) can piggyback on any
future eval. LOFT: hold behind its M0 occupancy probe (also minutes); kill/green-light
on the measured F before any implementation spend.

### 4.3 Verification note (orchestrator review of the candidate derivations)

- ECHO gradient `∂R/∂z_u = π_u(a_u − P_A)`, shift-invariance, and row-ℓ1 ≤ 2P_A(1−P_A):
  **verified**. Gate-aware fixed point with the (1−p_v) factor: **verified**.
- ECHO "sign theorem": **REFUTED as stated** (it dropped δLSE = Σ_u π_u δz_u).
  Corrected statement and proof in §5.3. Counterexample on record: π_dom=0.7 (∉W),
  π_copy=0.15 (active), π_y=0.01 ⇒ δ log π(y) ∝ −0.053 < 0.
- BERM `∂PR/∂e_i = (2S1/S2)(1−PR·ê_i)` and the identity ē·S1/S2 = r: **verified**.
- LOFT field `g = −c(𝟙_𝔅 − F)y` with c = 4λh/(L(k−1)m) via σ̄²/S cancellation:
  **verified**.

---

## 5. Selected framework: ECHO — full development

### 5.1 Core mathematical framework

**Spaces.** Vocabulary [V]; contexts c ∈ 𝒳 = [V]^≤T; the readout map
Φ_θ : 𝒳 → Δ^{V−1}, c ↦ π(·|c) (the full CHIRON forward + softmax). Training measure 𝒟
over (c_t, y_t) pairs induced by the corpus stream under teacher forcing. For each
position, the *trailing empirical measure* ν_t ∈ Δ^{V−1} supported on ≤ w atoms.

**The regularized object** is the pair (π_t, ν_t) — the model's conditional measure
against its own context's empirical measure. ECHO constrains the **Radon–Nikodym-style
ratio on window support**: the feasible set is

```
C_t = { π ∈ Δ^{V−1} : π(v) ≤ κ·ν_t(v) + τ₀   ∀ v ∈ supp(ν_t) \ {y_t} }
```

i.e. a one-sided, data-dependent upper confidence bound on copy mass: the model may
assign any window token up to κ× its local base rate plus an absolute allowance τ₀, and
may assign the *true* next token anything. R_t is the ℓ1 exterior penalty for C_t:

```
R_t(π) = dist₁⁺(π, C_t) = Σ_{v ∈ A_t} (π(v) − m_t(v))₊ ,   m_t(v) = κν_t(v) + τ₀
```

**Constrained-optimization interpretation:** min E[CE] s.t. π_t ∈ C_t 𝒟-a.s., penalized.
The constraint set moves with the data (window and truth) — ECHO is *gated by the data*,
structurally unable to fight the data gradient the way a fixed internal preference
(PACT) must.

**Information-theoretic interpretation:** on window support, ECHO caps the pointwise
likelihood ratio π/ν at κ + τ₀/ν — an upper bound on the per-token evidence the model
may claim for "the context predicts its own recent past," calibrated by how often the
past actually recurs locally.

### 5.2 Objective and exact gradient (final form)

```
L = CE + ζ·mean(logZ²) + (λ/T)·Σ_{t=1}^{T} Σ_{v∈A_t} (π_t(v) − m_t(v))₊

dz_t(u) += (λ/T)·π_t(u)·(a_t(u) − P_{A,t})
        =  (λ/T)·[ −P_{A,t}·π_t(u)  (dense, z-loss-shaped per-row scalar × probs)
                    + π_t(u)·a_t(u) (sparse, ≤ w ids) ]
```

Subgradient at the hard-hinge knee: closed-inactive (a=0).  The optional Huberized
positive hinge (flag `--echo-huber δ`) is now specified exactly as
`Hδ(x)=0` for `x≤0`, `x²/(2δ)` for `0<x<δ`, and `x−δ/2` for `x≥δ`.  Writing
`b_v=Hδ'(π_v−m_v)` and `P_b=Σ_v π_v b_v`, its exact logit field is
`π_u(b_u−P_b)`; `δ=0` recovers the hard field above bit-for-bit.  The implementation
emits the sparse `b_v` values only in Huber mode, so the hard path pays no extra scratch.
Properties (all verified): Σ_u dz-correction = 0 per row (shift-invariant);
|dz(u)| ≤ λ/T; row-ℓ1 ≤ λ/(2T) vs CE's ≤ 2/T (≤ λ/4 relative); vocabulary-permutation
equivariant; no dependence on any internal statistic (first purely-readout regularizer
since Z-loss, and orthogonal to it: z-loss acts on the shift mode, ECHO sums to zero
along it).

### 5.3 Directional-NLL Lemma (corrected; replaces the candidate's "sign theorem")

Let δz = −ηλ∇_z R_t with fixed active set A_t ∌ y_t, and write P_A = Σ_{A}π_u,
Q_A = Σ_{A}π_u², Q = Σ_{V}π_u², Q_{A^c} = Q − Q_A. Then

```
δ log π_t(y_t) = δz_y − Σ_u π_u δz_u = ηλ·[ π_y·P_A + Q_A − P_A·Q ]
              = ηλ·[ π_y·P_A + Q_A·(1−P_A) − P_A·Q_{A^c} ]
```

**(i) Pathological regime (the target): strictly helpful.** If A_t = {c} and the copy c
is the modal token, then δ log π(y) = ηλ·π_c·(π_y + π_c − Q) ≥ ηλ·π_c·π_y > 0, since
Q ≤ max_u π_u = π_c. More generally the response is positive whenever
Q_A(1−P_A) + π_y P_A > P_A Q_{A^c} — i.e. whenever the active copy set holds the
dominant share of the squared mass.

**(ii) Mixed regime: bounded of either sign.** Since y ∉ A ⇒ π_y + P_A ≤ 1 and Q ≤ 1:
|δ log π(y)| ≤ ηλ·P_A. The negative case requires the coincidence of (a) over-margin
copy mass, (b) a *different*, dominant non-window mode, (c) tiny truth mass — a regime
whose corpus measure is small and which the stratified-val observable monitors
(failure mode 2).

**Consequence for the NLL-risk budget.** The per-position first-order NLL response is
O(ηλ·P_A) of either sign, positive precisely in the diagnosed pathology. With activation
rate q̄ (predicted 2–5%) and E[P_A | active] ≤ ~0.5, the aggregate direct effect is
≲ λ·q̄/2 per unit CE gradient — at λ=0.1, ≲ 0.003 nat scale, an order below the 0.05
4-batch jitter. The population-level shrinkage bias on legitimate repeats (§3A fixed
point) adds ≲ λ·q̄·(1−p̄)/4, same order. **NLL safety = bound × sparsity, enforced by the
margin calibration — not a free-lunch sign guarantee.** This is the honest, weaker claim
the E3/E4 gates test.

### 5.4 Temporal dynamics: the free-running attractor and what TF training reaches

**Free-running model.** Under decoding kernel D (greedy/top-p), generation is a Markov
chain on contexts; for a loop σ of period ρ ∈ {2..6}, let 𝒞_σ(j) = contexts ending in j
consecutive loop copies, and the **copy-gain curve** G(j) = E[π(next loop token) | c ∈
𝒞_σ(j)]. The diagnosis measured the interior: G(j≫1) > 0.95 with logits climbing 8.6→38
— absorbing under top-p 0.95.

**Escape-rate bound (the quantitative bridge).** Under top-p with threshold P, if
π(copy) ≤ G < P at some phase, the nucleus includes non-copy mass ≥ P − G before
renormalization, so per-step escape probability ε ≥ (P − G)/P and loop residence is
geometric with mean ≤ P/(P − G). **The operative training target is therefore weaker
than calibration: cap the reachable copy-gain below the nucleus threshold.** G ≤ 0.9
under P = 0.95 gives ε ≥ 5%, mean loop length ≤ 20 — degeneration becomes a transient,
not an attractor.

**What ECHO provably reaches:** the TF-measurable boundary G(0..j_small) — natural text
contains lag-2..6 repeats (enumerations, "that that", the numeric/markdown val regions),
so the copy-gain at small j is on-distribution and directly trained down wherever it
exceeds margin against the data. **The two bridge hypotheses (explicit):**
- **H1 (entry suppression, solid):** a free run from a clean seed is near-distribution
  until the *first* pathological repeat; its probability is exactly what ECHO trains
  down; deep-basin entry rates multiply down through the trained boundary.
- **H2 (interior flattening, conjecture):** the interior climb G(j)↑ extrapolates the
  boundary trend because the same circuit (the copy/induction pathway) produces G at all
  j — suppressing its gain on-distribution suppresses it off-distribution. Mechanistic,
  plausible, unprovable in-distribution; priced at ~35% in failure mode 1.

**Corollary (testable):** ECHO should help nucleus generation more than greedy (mass
suppression vs rank change), and π(copy)-at-emission should drop measurably in both.

**Continuous-time view.** Gradient flow ż = −∇_z(CE + λR) per position: inside C_t pure
CE flow; on ∂C_t the flow acquires an inward ℓ1-projection component of magnitude ≤ λ;
the ω-limit set is contained in the KKT points of the constrained problem, which
coincide with CE stationary points wherever the constraint is slack.

### 5.5 Theoretical analysis (consolidated)

- **Well-posedness:** R finite, convex piecewise-linear in π; C^∞ off the knee set
  (measure-zero; data expectation smooths it); globally bounded gradient — the
  divergence-at-init class is structurally empty (no activation-scale normalization
  anywhere).
- **Invariances:** logit shift; vocab permutation; independent of (q,p) internals — no
  interaction with WhiSC/ρ/reanchor machinery.
- **Bias characterization:** the regularized optimum deviates from the CE optimum only
  on over-margin copy coordinates, by ≤ λ(1−p_v)/4 relative, argmax-preserving wherever
  p_v − λ(1−p_v)/4 > max competing mass. TF top-1 (0.72) essentially untouched.
- **Init/self-gating:** R ≡ 0 at uniform π; activation grows only as the model learns to
  over-concentrate on copies — no warmup needed.
- **Adam:** correction is sparse-plus-rank-one per token, aggregated over 16384 tokens in
  the readout GEMM backward — weight-gradient perturbation is a smooth average; no
  bimodality (LayerDrop channel empty). Hinge flicker perturbs dz below its own BF16
  storage quantization.
- **Scaling:** cost O(T·w), V-free; margins V-free; λ per-token-normalized (T-invariant);
  w enters only through ν.
- **BF16:** hinge is 1-Lipschitz in π; π computed FP32 from bf16 logits + FP32 LSE;
  scalar R accumulated via FP32 warp-shuffle partials.

### 5.6 Algorithm (per µstep, training only)

```
forward:  logits z (bf16) + LSE (fp32)  [existing]
ECHO-A:   for each t (1 block/token):
            shared-mem histogram of W_t (w ids)  →  n_t(v) for distinct v
            gather π_t(v) from materialized bf16 probs (fp32 registers)
            gate: v ≠ y_t, π > κ·n/w + τ₀  →  active list, P_b, partial ΣHδ
            emit P_b (fp32[T]), active ids; emit b_v only when δ>0
loss:     L += (λ/T)·ΣHδ(π_t(v)−m_t(v))
backward: combined CE/Z/ECHO row scalar += −(λ/T)·P_b
ECHO-C:   dz_t[u] += (λ/T)·b_u·π_t(u) for u in active list (row-disjoint scatter)
```

Val/inference: term absent (same contract as PIED). E0 bit-parity at
`--echo-coef 0` (kernels not dispatched).

### 5.7 Computational tradeoffs

~2M gathers per µstep (T·w) + ~10 MB traffic beside the ~1-TFLOP readout GEMMs and
the [T×V] CE sweep: **≤ 0.3% wall predicted (production wall gate remains E3)**;
~8.6 MB hard-hinge scratch at T=16384,w=128, plus 8.0 MB only when Huber weights are
enabled; zero new GEMMs and no global atomics.  The current trainer downloads O(T)
ECHO vectors when it reads/logs step loss and telemetry; no [T×V] D2H occurs.

### 5.8 Comparison to existing methods (supporting the formulation)

| Method | Object/action | ECHO's departure |
|---|---|---|
| Unlikelihood (Welleck) | always-on log-barrier on context-token mass; slope →∞ as π→1; known NLL tax; port class | hinged, margin-calibrated, bounded-slope, truth-gated; recovered only as (margin→0, log-barrier) limit |
| Confidence penalty (Pereyra) | isotropic entropy bonus over V | anisotropic: only copy directions, only beyond base-rate, only against the data |
| Label smoothing | two-sided uniform pull, always-on | none of its bias off the window; one-sided |
| Z-loss (shipped) | logZ magnitude (shift mode) | exactly orthogonal (Σg=0); shares the kernel path |
| Decode-time repetition penalties | inference-side | refuted class per the 2026-06-27 diagnosis; ECHO is training-side |
| Scheduled sampling / DPO (deferred plan) | on-policy generation-aware training | ECHO is the zero-cost TF-domain rung 0; results scope the on-policy rungs |

Distinctness from the internal graveyard (PACT/SIRA/PHS/PIED/SFD): different space
entirely (probability simplex at the readout vs phase space/spectrum), data-gated,
one-sided, sparse — see §3A table logic; notably ECHO addresses exactly the period-2..6
band that SFD proved is unreachable from the DCT grid, in token space where period is
directly observable as lag.

### 5.9 Failure modes and mitigations

As §3A, with the §5.3 correction feeding mode 2: the mixed-regime negative response is
an additional (small, bounded) contributor to the calibration-tax channel — monitored by
the same observable (position-stratified val on repetitive regions + activation-rate
trajectory). Mitigations: raise τ₀/κ (shrinks A), ECHO-lag reweighting (concentrates
pressure where the diagnosis lives), λ down.

### 5.10 Minimal prototype and gate plan

Implemented instantiation: flat ECHO (no lag weights), w=128, κ=2, τ₀=0.1, λ opt-in,
hard hinge by default. Kernels: `echo_repeat_stats[_huber]`, the combined
`softmax_cross_entropy_bwd_bf16_zloss_echo`, `echo_scatter_bf16[_weighted]`, CPU refs,
`test.sh chiron-echo`, and trainer smoke `scripts/echo_training_loss_smoke.sh`.

- **E0 engineering PASS (2026-07-15):** coefficient-zero trainer loss parity and shipped
  z-loss-kernel bit parity; no ECHO allocation/dispatch at the default coefficient.
- **E1 engineering PASS (2026-07-15):** FD hard-hinge logit gradient, shift invariance,
  truth exclusion, prefix-window edges, dedup/margins, and CPU≡GPU hard stats/scatter.
- **E2 engineering PASS (2026-07-15):** Huber FD and δ→0 hard consistency, per-row
  zero-sum field, init inactivity, λ-linear RMS calibration, and CPU≡GPU Huber
  stats/weighted-scatter/composed backward.  The end-to-end tiny trainer smoke verifies
  warmup, all wrapper flags, Z-loss independence, loss ordering, and telemetry.
- **E3** matched 2500-step pair (fresh baseline, same binary — era-drift discipline):
  kill if ΔNLL > +0.05, activation ≈ 0 by 2500 (inert), or any grad-skip delta; record
  E[P_A] + π(copy) histogram (margin-evasion tripwire).
- **E3.5** free-gen probe on both 2500-step checkpoints (--gen-metrics, prose seeds per
  the diagnosis protocol; repeated-token-fraction + type-token ratio, NOT distinct-4).
- **E4** 30k matched pair. **Ship bar:** wide-32 within +0.02 of baseline AND ≥2×
  nucleus loop-rate reduction from prose seeds; either miss = no-ship, publish negative.

**Predictions (falsifiable, λ=0.1, w=128):** wide-32 Δ ∈ [−0.02,+0.02] (neutral modal);
TF excess-copy mass E[P_A] −50%+ (high confidence — failure here = mechanism inert);
activation 2–5% early, decaying, >10% at 25k = tax signal; nucleus loop rate ~40% chance
of ≥2× reduction, greedy ~20%; position-stratified val flat except ±0.02 in repetitive
regions.

### 5.11 Full research program

1. **ECHO rung 0** (this design): TF-domain boundary suppression. Decision point at
   E3.5/E4 generation metrics.
2. **ECHO-lag** (variant, reserved): inverse-lag hinge weights if activation mass is
   dominated by long-lag function words.
3. **Curriculum coupling**: λ or κ schedule tied to measured G(j-small) trajectory —
   only if rung 0 shows a real but insufficient effect.
4. **On-policy rungs** (the deferred generation-aware plan, scoped by rung-0 results):
   short self-sample rollouts + unlikelihood/DPO on loop entries — needed iff H2 fails
   (TF-effective/generation-inert outcome), which rung 0 will have demonstrated cheaply.
5. **Portfolio**: BERM S0 shadow probe piggybacked on the next eval pass; activate for
   the FineWeb 305k run (collapse insurance where exposure is 10×). LOFT M0 occupancy
   probe (minutes) → kill or green-light on measured F.

### 5.12 Open conjectures and validation criteria

- **C1 (escape-rate bridge):** capping TF-reachable copy-gain below the nucleus
  threshold converts loops from absorbing to transient (mean length ≤ P/(P−G)).
  *Validate:* E3.5/E4 loop-length distribution vs π(copy)-at-emission.
- **C2 (shared-circuit extrapolation, H2):** boundary suppression transfers to basin
  interiors because one copy-circuit produces G(j) at all j. *Validate:* G(j) curve on
  synthetic j-repeat probes (TF eval with constructed contexts — cheap, off-line,
  no training).
- **C3 (nucleus > greedy asymmetry):** mass-level term helps sampled decoding more than
  argmax decoding. *Validate:* paired gen-metrics.
- **C4 (margin sufficiency):** κν + τ₀ separates legitimate from pathological reuse at
  the corpus level (activation concentrated on data-contradicted copies). *Validate:*
  activation-rate stratification by (ground-truth-agrees vs disagrees).

---

## 6. Record

- Candidates developed 2026-07-09 by three parallel theory agents from the shared parsed
  problem (§2); orchestrator verification + correction §4.3; selection §4.2.
- ECHO hard kernels/CPU refs and initial E0/E1 suite landed in `glades-ml` commit
  `cfd1a31ea`; initial trainer seams landed in sibling `glades-trainer` commit `e97a292`.
- 2026-07-15 completion work adds the optional Huber field, warmup, independent-Z-loss
  dispatch, all wrapper flags, active-rate/mean-mass/copy-probability telemetry, expanded
  E2 tests, and an end-to-end trainer smoke.  Evidence is recorded in
  `research/CHIRON_ECHO_IMPLEMENTATION_2026_07_15.md`.
- E3/E3.5/E4 remain explicitly unrun; no NLL, generation, or ship claim is made.
- Companion memory: `regularizer_design_echo_berm_loft.md` (auto-memory).
