# VITALS — Zero-Cost Training Telemetry for CHIRON (design study, 2026-07-09)

**VITALS** = **V**erified **I**nstrumentation of **T**raining via **A**djoint, **L**edger &
**S**tream statistics. A suite of (near-)zero-cost, read-only training statistics for the
CHIRON 1B trainer covering four axes: **(1) gradient health, (2) generalization,
(3) overfitting, (4) token/capacity saturation** — every statistic computable from
quantities already resident during training, total budget **≤0.5% wall**, ≤50 MB VRAM,
C++98/CUDA-native, default-off behind `--vitals`.

Status: **IMPLEMENTED 2026-07-15** in glades-ml plus the sibling CHIRON trainer.
The deterministic estimator/kernel/unit/smoke gates are implemented and passing; the
long-horizon empirical gates G1–G10 remain run-level validation work (G9 necessarily
rides the 20B run) and are not represented as passed by implementation tests.
Produced by the 3-candidate research-framework-design protocol (candidates: LEDGER
estimation-theoretic / STROBE dynamical-observability / TIDE measure-transport; STROBE
selected as spine, LEDGER + TIDE absorbed as certificate terms — §2–3).

Certainty tags used throughout: **[P]** proven (exact algebra / by construction),
**[D]** derivable under stated assumptions, **[H]** heuristic (mechanistically motivated,
uncalibrated), **[C]** conjecture (falsifiable, gated).

---

## 1. Executive summary

Training health for CHIRON is formalized as a **certificate over three coupled flows**:

1. the **fast depth flow** — the symplectic per-token dynamics `(q,p)` over layers
   `l = 0..24` and its adjoint (backward) co-flow;
2. the **slow parameter flow** — the clipped, Adam-preconditioned stochastic estimation
   process over steps `t`;
3. the **output measure flow** — the empirical per-token loss measure transported toward
   the entropy floor.

The trainer is *healthy* iff four certificate terms are bounded/stationary:

```
I.   Amplification × forcing   (adjoint gain spectrum bounded; energy budget stationary)
II.  Observer residual         (backward's recomputed orbit shadows the forward's)
III. Estimation efficiency     (signal power ≫ preconditioned noise-fit rate)
IV.  Transport coherence       (loss-measure drift ≫ churn; strata still descending)
```

An **alarm is a certificate-term violation, typed by term**, split into FAST alarms
(explosion precursors, ~10-step horizon) and SLOW alarms (OBSD-class secular drift,
~1000-step horizon). The four user-facing axes map onto terms: gradient health = I+II;
generalization = III + IV-shape (validated against occasional wide-val); overfitting =
III's noise-fit rate + probe gaps + PIED susceptibility; saturation = IV's stratified
slopes, Péclet/anneal trigger, and coverage.

Headline properties:

- **~18 statistics, ≈0.45% wall total**, dominated by cadence-gated probe forwards; the
  every-step statistics are fused in-kernel reductions with zero extra DRAM traffic.
- **Self-testing**: four of the statistics are exact identities the taps must satisfy
  ([P] ledger closure, PIED mask calibration = 1/(1−π), dp depth-flatness, stratum-slope
  reconstruction) — the suite validates its own plumbing continuously.
- **Every statistic has a pre-registered falsifiable gate**; the two strongest gates run
  on in-tree deterministic testbeds (the `--rot-coupling` SORC divergence at step ~831,
  the step-9001 hard batch) and require no purged checkpoints.
- Directly instrument the four historical failure classes: SORC geometric cascade (fast),
  OBSD secular gate growth (slow), q-side ReLN stat drift (observer), datascale-era
  divergence — and the two open risks: repetition-attractor deepening and
  memorization-at-scale on the planned FineWeb 20B run.

---

## 2. Candidate formulations (summary)

Three materially different formulations were developed independently:

**LEDGER (estimation-theoretic).** Training = sequential stochastic estimation; monitors
= finite-sample moment functionals of the per-sequence gradient distribution and the
per-token loss distribution. Governing identity [D]:
`E[ΔL_batch − ΔL_pop] = −λ·tr(A·Σ)/B` (A = clip∘Adam preconditioner, Σ = per-sequence
gradient covariance, B = batch size in sequences). Defines overfitting under single-pass
data as the **noise-fit rate** ω_t = λ·tr(AΣ)/B_eff — the portion of each step's apparent
loss decrease a fresh batch would not confirm. Key estimators: gradient noise scale from
the (µstep-1, full-accum) norm pair; per-element signal/noise decomposition from Adam's
(m̂, v̂) registers; clip ledger with a "signal-clipped fraction"; canary/twin probes.

**STROBE (dynamical-systems / observability).** Depth = fast time; forward pass = flow on
phase space, backward = adjoint co-flow; training step = slow time; monitors = stability
and observability functionals of the fast system sampled along slow time. Proved the
**two-track adjoint theorem** [P]: in the pure shear+reln system the p co-state is
constant across depth (`dp_{l−1} = dp_l` exactly), so the entire adjoint dynamics is a
forced linear recursion in the q channel — deriving why dq clamps were historically the
right lever and making dp-profile flatness a free structural self-test. Key statistics:
adjoint gain spectrum γ_l / depth log-gain Λ, reanchor observer residual, depth-resolved
energy ledger with proven closure identity, PIED fluctuation–dissipation susceptibility,
depth participation ratio. FAST/SLOW alarm typology.

**TIDE (measure-evolution / transport).** Primitive = the empirical per-token loss
measure, stratified by token frequency band × position band × **copy-availability**
(target continues an n-gram already in the window) × probe type, evolving in Wasserstein
space (quantile trajectories = exact coordinates of the measure path [P]). Key
statistics: exact copy-gain/generalization-gain decomposition of NLL improvement (a
train-side repetition-attractor monitor); drift-vs-diffusion decomposition from canary
replays with a **Péclet number** whose collapse at constant LR is a principled
*anneal-now* trigger (formalizing the lineage's late+sharp finish lesson); stream-local
memorization gap with a deliberate-replay positive control; vocabulary coverage operator;
a pre-registered stopping rule for the 20B run.

## 3. Selection rationale

No single candidate covers all four axes best: STROBE dominates gradient health
(architecture-native, proven identities, strongest gates), LEDGER dominates
generalization theory (the ledger identity gives the only principled train-side
generalization *rate*), TIDE dominates saturation/overfitting measurement (actionable
anneal/stopping triggers, the copy/generalize split the lineage lacked when the
reanchor ship still degenerated in free generation).

**STROBE is selected as the spine** because its certificate formalism structurally
subsumes the other two: LEDGER's estimation terms are precisely the slow-time certificate
term (III), and TIDE's measure transport is the output-observable certificate term (IV).
The rejected-as-spine candidates are weaker as *organizing principles* — LEDGER has no
natural home for depth-resolved instability precursors (its per-group split localizes in
parameter space, not depth), and TIDE cannot see the backward pass at all — but both are
retained nearly whole as terms. This mirrors physical monitoring practice: a plant is
instrumented by (i) internal state stability, (ii) state-estimator consistency,
(iii) actuation efficiency, (iv) output quality — VITALS terms I–IV.

---

## 4. Formal problem statement

Let `w_t ∈ R^d` (d = 870.94M) evolve under `w_{t+1} = w_t − λ_t c_t P_t m̂_t` where
`c_t = min(1, τ/‖ĝ_t‖)` (τ = 0.5, binds nearly every step — healthy per the CHARGE
Phase-0 sweep), `P_t = diag(1/(√v̂+ε))`, `m̂, v̂` int8-AdamW state, and
`ĝ_t = (1/4)Σ_{k=1..4} γ_k` accumulated over 4 µsteps of one 16384-token sequence each.
Per token τ and layer l the forward computes `x_l = (q_l, p_l) ∈ R^{2×2048}`:

```
Φ_l = Φ_reln ∘ Φ_shear ∘ Φ_rot :
  rot   (WhiSC): per-channel (q,p) ← W⁻¹R(θ_l)W (q,p),  W = diag(1/a, a) detached
  shear (SCFA):  p ← p + s_l·η ⊙ y_l(q),  η = PIED mask ∈ {0, 1/(1−π)}, E[η]=1
  reln:          q ← N(q)   (unit-RMS per token)
```

Design a statistic vector `S(t) ∈ R^k` and alarm map such that: (i) every component is
estimable from resident quantities with total added wall ≤0.5% and VRAM ≤50 MB;
(ii) taps are read-only (bit-identical training math); (iii) each component has a
formal definition, healthy dynamics, pathological signature, and a falsifiable
pre-registered validation gate; (iv) jointly, S(t) detects the four historical failure
classes and answers the four monitoring axes.

**Resident-quantity inventory (grounded in the current tree):**

| Quantity | Where it lives today |
|---|---|
| per-µstep loss/acc scalars (`lossSum/lossCount/correctCount/validCount`) | already downloaded every µstep (`read_loss_scalars`, chiron_main.cpp) |
| per-token `logZ[T]` | z-loss path (`softmax_forward_bf16_with_lse`), downloaded on log steps already |
| per-token NLL vector | computed in CE kernel, currently reduced only — a 64 KB/µstep store exposes it |
| global grad norm | `compute_grad_norm_sq` every step (clip); per-tensor `log_grad_norm_breakdown` exists (trigger-only today) |
| µstep-1 gradient | backward is called with `accumulate=false` on the first µstep — the buffer *is* γ₁ exactly |
| dq row RMS + clamp counts, per layer | `row_rms_clamp(x, rows, cols, tau, d_clampedCount, d_nonfiniteCount)` at every boundary |
| reanchor recomputed ReLN stats | `chiron_reln_backward_reanchor(..., scratch_stats_split)` writes 2T floats/layer |
| WhiSC per-channel `E[q²], E[p²]` EMAs | `chiron_whisc_update_stats`; `[whisc]` monitor precedent |
| SCFA increment `y`, mask `η`, `p` pre-commit | resident in the PIED commit kernel (`chiron_scfa_axpy2_masked_dual_p`) |
| Adam per-element `m̂, v̂`, update | in registers in the fused int8-AdamW kernel |
| embedding-row updates | `embedding_scatter_add` touches exactly the batch-present rows |
| live-eval loop | `--val-every/--val-batches/--val-position-buckets` already in-tree |
| shadow-diagnostics pattern | SIRA shadow / PHS / PTOC: default-off, accum-boundary, logEvery-gated |
| copy-mass per-token stat | ECHO `echo_Rrow[T]` (when `--echo-*` active) |
| SAM replay machinery | `samRho` path: ascend to ρ-ball, replay window, restore — usable as sparse sharpness probe |

**Cost model.** Extra DRAM traffic is the currency (~1 TB/s, ~640 ms/step). A full
gradient-buffer norm pass ≈ 1.5–3.5 GB ≈ 0.25–0.55% wall if run every step → cadence-gate
it. Fused in-kernel warp-shuffle reductions to per-layer scalars ≈ free. **Mandate from
the PACT −40% incident: warp-shuffle reductions only, never shared-memory double atomics;
anything contended is cadence-gated.**

---

## 5. Core framework: the three-flow certificate

**Fast flow / adjoint.** Jacobian transposes of the constituent maps [P]:

- shear: `Jᵀ = [[I, F_l'ᵀ],[0, I]]` — injects dp into dq, leaves dp untouched;
- reln: `Jᵀ = [[N'ᵀ, 0],[0, I]]` — propagates dq, leaves dp untouched;
- WhiSC rot (detached whitening): off-diagonals into dq are O(θ_max)·scale-matched [D]
  (the R1 bound, verified in vivo); without whitening they carry the raw
  ρ^{1/2} ≈ 17–30× factor (SORC) [D].

**Two-track adjoint theorem [P].** Without rotation, `dp_{l−1} = dp_l` exactly: the p
co-state is constant across depth and the adjoint is the forced linear recursion
`dq_{l−1} = N_l'ᵀ dq_l + F_l'ᵀ dp_L`. Consequences: (1) all backward pathology at fixed
dp_L lives in the q-channel gain and forcing — the resident dq row-RMS taps observe
exactly the right object, and the historical dq clamps were provably the correct lever;
(2) dp-profile flatness is a free self-test, deviating only by O(L·θ_max) under WhiSC [D]
and unboundedly under SORC-class couplings.

**Slow flow / ledger identity [D]** (first-order Taylor, small λ, A ⊥ ĝ):

```
E[ΔL_batch − ΔL_pop] = −λ · tr(A·Σ) / B_seq
```

The right side is the **expected one-step overfit increment** ω_t: with never-revisited
data the classical train/val gap is ≈0 by construction, so ω_t *is* the correct
redefinition of overfitting — the rate at which the optimizer fits transient batch noise.
Its running integral M_t is the cumulative memorization ledger.

**Output flow.** The empirical loss measure µ_t decomposes [P] as
`E_µt[ℓ] = H(X|CTX) + E_ctx KL(p* ‖ p_θt)`; only differences and *rates* are
interpretable (per-stratum floors are unknown constants) — hence the suite is
quantile-first and rate-based, with fitted asymptotes advisory only.

**Healthy manifold and alarms.** `S(t)` is compared against: adaptive robust bands
(trailing median ± c·MAD, window 500) for FAST components; **frozen absolute thresholds
from one designated calibration run** for SLOW components (an adaptive band tracks
OBSD-speed drift and never alarms — the frozen threshold trades false positives under
legitimate regime change for sensitivity to secular drift). All alarms are warmup-gated
(steps < 1500 log-only; ρ moves 0.096→~2000 in the first ~1k steps by design).

---

## 6. THE STATISTIC CATALOG

Format: **definition · estimator/tap · cost · healthy · pathological · gate.**
V-numbers are the shipped identifiers; tier T0 = fused/every-logEvery, T1 = cadence-25
norm pass, T2 = cadence-100..1000 probe forwards.

### Term I — depth-flow stability (gradient health, FAST+SLOW)

**V1. Adjoint gain spectrum** `γ_l, Λ, b_l, π_l` (T0).
`g_l` = RMS over (τ,i) of **pre-clamp** dq at boundary l; `γ_l = g_{l−1}/g_l`;
`Λ = Σ_l log γ_l = log(g_0/g_L)` [P, telescoping]; `b_l` = clamp-bind fraction (already
output as `d_clampedCount`); `π_l = RMS(dp_l)/RMS(dp_L)` (dp-flatness self-test).
*Tap:* extend `row_rms_clamp` with a fused per-layer mean-of-squares of the pre-clamp
value; one fused dp RMS in the commit backward. Zero extra DRAM. *Cost:* <0.02%.
*Healthy:* γ_l ≈ 1, Λ = O(1) stationary; b_l low, non-contiguous; π_l = 1 ± O(Lθ_max).
Global clip cancels in every ratio [P] — V1 stays quiet on healthy clip-binding runs.
*Pathological:* γ_l > 1 on a contiguous depth band with Λ trending up = geometric cascade
(SORC: 2.5×/layer ⇒ Λ ≈ 22 vs healthy O(1)); π_l non-flat = rotation-channel runaway
(invisible to dq clamps). *Gate G1 (strongest in suite):* rerun `--rot-coupling`
(deterministic divergence ~step 831, in-tree): pre-register (i) Λ crosses median+6·MAD
for 3 consecutive steps ≥50 steps before the first grad-skip; (ii) blow-up profile slope
log γ ≈ 0.9±0.3/layer; (iii) zero FAST Λ-alarms on a matched healthy 2500-step arm.

**V2. Reanchor observer residual** `r_{l,τ}` (T0).
`r = |log(invStd_saved/invStd_recomputed)| + |µ_s − µ_r|·i_r` — when means agree,
`i_s/i_r` is exactly the xhat inflation factor that killed the datascale run (the 13×
event is r ≈ 2.56) [D]. Aggregate as per-layer (q999, max, exceedance counts N_{0.1},
N_{0.7}). *Tap:* the reanchor backward already computes the recomputed stats
(`scratch_stats_split`); persist the forward stats (2T floats/layer ≈ 3 MB — negligible)
and fuse the residual in-register. *Cost:* <0.01%. *Healthy:* r at the recompute
round-off + atomics floor (~1e−3–1e−2, calibrated in-run); N_{0.1} ≈ 0. *Pathological:*
heavy-tail growth of q999/N in specific layers over hundreds of steps = the q-side stat
drift re-emerging — the direct precursor the reanchor cure masks but no longer surfaces.
Two-tier: warn at sustained 0.1, critical at 0.7 (= 2× inflation). *Gate G2:* replay
through step 9001: R^{999} and Λ must show a transient ≥3× the neighboring band at that
batch; a PIED-on arm must damp the Λ transient vs PIED-off (anchor: 10.8→4.9).

**V3. Depth energy ledger** `E_l, Ẽ_l, c_l, P_l, κ, m_l` (T0).
Unmasked increment energy `E_l = mean(y_l²)`; committed energy `Ẽ_l = mean((η⊙y_l)²)`;
state–increment alignment `c_l = mean(p_{l−1}⊙s_lη⊙y_l)`; state energies `P_l, Q_l`.
**Ledger identity [P]:** `P_L = P_0 + Σ_l (Ẽ_l + 2c_l)` — an in-run exactness self-test
of the taps. **Cancellation fraction** `κ = −2Σc_l/ΣẼ_l` — the cross-term share; κ > 0 =
net destructive interference across depth (exactly the co-adaptation PIED's implicit
penalty taxes). **Mask calibration [P]:** `m_l = Ẽ_l/E_l = 1/(1−π) = 1.111…` under mask
independence — a free tripwire on the PIED counter-hash. *Tap:* three fused reductions in
the commit kernel (p, y, η all resident). *Cost:* <0.05%. *Healthy:* Q_l = 1 exactly
[P, reln]; P profile → ~10³ (trained-in); E_l stationary-to-slow; m_l = 1.111±noise.
*Pathological:* SLOW `d log ΣE_l/dt > 5e−5/step` sustained 2000 steps = OBSD-class
secular inflation (OBSD's gate ×15 over 13.5k steps ⇒ log-slope ≈ 2e−4/step [D]); FAST
P_L step-jump (SORC: ×213); κ secular growth at flat NLL = deepening cancellation —
**monitor-only** (PACT E4 teaches penalizing κ buys nothing; watching it is free).
*Gate G4:* (i) ledger closure |P_L − P_0 − Σ(Ẽ+2c)|/P_L < 1e−3 every logged step;
(ii) m_l ∈ 1.111 ± 3σ; (iii) [C] PIED-on/off 2500-step pair shows κ_on < κ_off.

**V4. Clip & loss-scale ledger** `c_t, bind fraction, ς_t` (T0, trainer-side only).
`ς_t = (1−c_t)·Ĝ²/(Ĝ²+Ŝ/4)` = fraction of clipped mass that was *signal* [D, uses V6's
split]. *Tap:* c_t already computed; ς from V6. *Cost:* 0. *Healthy:* clip binds
(known-healthy) with ς modest and stable — consistent with Phase-0's "clip discards
harmful excess". *Pathological:* ς rising secularly = clip throttling learning (the OBSD
failure channel: gate growth → ‖g‖ inflation → effective-LR starvation). *Gate:*
reconstruct ς on the stored Phase-0 τ-sweep logs; must rank consistently with the
measured val ordering (pre-registered as a *conditioning covariate*, not an objective —
tightening τ was flat, loosening was worse).

**V5. Optimizer-state health** (T0).
Fused in the int8-AdamW kernel: int8 blockwise saturation counts, v̂ dead-fraction
(|update| < ε), per-group |Δw|/|w| tails, sign-agreement `P(sign(m̂)=sign(ĝ))` (a bounded
monotone transform of per-element SNR, heavy-tail-robust [D]). *Cost:* 0 (registers).
*Healthy:* saturation ≈ 0; stable tails. *Pathological:* rising int8 saturation or a
widening V6-vs-V7 noise-estimate discrepancy = quantized-state degradation (feeds the
ORBIT state-compression decision); a group with collapsing sign-agreement but high update
norm = churn (candidate: rot_phi at tanh saturation; embeddings on saturated strata; the
de-facto-frozen scfa_D precedent shows silent no-update paths exist). *Gate:* the V6/V7
discrepancy must shrink in near-stationary late-training windows, else the int8 noise
model is wrong.

### Term II — gradient estimation regime (generalization, SLOW)

**V6. Gradient noise scale** `Ŝ, Ĝ², B_noise` global + per-group (T1, cadence 25).
From the (b=1, B=4)-sequence norm pair: `E‖γ₁‖² = G²+S`, `E‖ĝ‖² = G²+S/4` ⇒
`Ŝ = (4/3)(‖γ₁‖²−‖ĝ‖²)`, `Ĝ² = (4‖ĝ‖²−‖γ₁‖²)/3` — unbiased despite ĝ containing γ₁
(only marginal expectations needed) [P, i.i.d. sequences]. Report ratio-of-window-means
with median-of-means over 50–100 samples (single-step Ŝ is noisy/negative under
anisotropic noise: rel-std ≈ √(2/r_eff)(1+G²/S) [D]). Per-group versions (embeddings /
q-path / p-path Wo / gamma-beta / rot_phi) from per-tensor partials in the same passes.
*Tap:* trainer calls `compute_grad_norm_sq` (extended to a small vector of group
partials) right after µstep-1's backward on instrumented steps — the buffer is exactly γ₁
because the first µstep runs `accumulate=false`; the full-accum norm is free (clip pass).
*Cost:* one extra grad-buffer read ≈ 0.25% at cadence 1 → **0.01% at cadence 25**.
*Healthy:* B_noise drifts up as loss falls [H]; group values stable. *Pathological:*
Ĝ² collapse at steady Ŝ = signal starvation (OBSD signature); a single group's Ĝ²
spiking = coherent instability (SORC's drot_phi explosion is a *signal* explosion — V6
distinguishes it from noise blow-up). Units: per-sequence, not per-token (within-sequence
independence is false; the actionable accum-question lives at sequence granularity).
*Gate G3:* on a 200-step instrumented run take norms after every µstep:
`E[M_k] = k²G² + kS` [P] — the overdetermined (G²,S) fit's residual ζ is a per-step
i.i.d.-violation test; pairwise-implied values must agree within CI. Falsified ⇒
shard-local correlation dominates; re-scope B_noise as stream-local.

**V7. GSNR / efficiency / noise-fit rate** `r_j, eff_g, ω_t, M_t` (T0).
Under EMA stationarity over the β2 horizon: `E[v̂_j] = G_j²+n_j`,
`E[m̂_j²] = G_j²+n_j/ESS` (ESS = (1+β1)/(1−β1) = 19) ⇒ per-element
`n̂_j = ESS(v̂_j−m̂_j²)/(ESS−1)`, `Ĝ_j² = (ESS·m̂_j²−v̂_j)/(ESS−1)` [D]. Group
**efficiency** = Σ Ĝ_j²/(Ĝ_j²+n_j) mass-weighted — the fraction of the group's update
energy that generalizes forward; **noise-fit rate** `ω̂_t = λc_tΣ_j n̂_j/(√v̂_j+ε)/ESS`
accumulated per tensor in the Adam kernel; `M_t = Σω̂` = memorization ledger. *Tap:*
fused, zero-traffic (registers). *Cost:* 0. *Healthy:* efficiency declining slowly and
uniformly; ω_t a modest, falling fraction of the batch-loss decrease; M_t sublinear.
*Pathological:* ω_t/|ΔL_batch| → 1 = the run is predominantly noise-fitting (capacity or
LR misallocation even while loss falls); one group's efficiency collapsing at high update
norm = churn. Bias chain [D]: needs G-stationarity over ~20 steps (suppress during
regime shifts via a loss-slope flag); clip is a global scalar so GSNR ratios are exactly
c-invariant, ω̂ biased only by Var(c_t) (itself monitored, V4); int8 quantization
inflates v̂−m̂² — hence **V6 is the load-bearing cross-check** (FP32/BF16 buffers, no
quantization); where they disagree beyond CI, V6 wins. *Gate G5:* regimes ranked by ω̂
(early vs late thirds of a 2500-step run) must rank the canary memorization jump (V13)
in the same order (sign agreement pre-registered; magnitude within ×2 is [C]).

### Term III — function-space response (overfitting/flatness)

**V8. PIED susceptibility** `χ̃, F_l` (T0 + T2 validator).
Second-order expansion in the mask forcing δη = η−1 (Var = π/(1−π) = 1/9 [P]):
`Var_η(L) ≈ σ_η² Σ_j G_j²` with `G_j = s_l y_j · dp_j` [D] — the across-mask loss
variance at fixed data is a **Fisher-weighted increment-energy quadratic form**, i.e. the
diagonal of exactly what PIED's implicit regularizer targets; and
`E_η[L]−L(1) ≈ ½σ_η²Σ H_jj ≥ 0` (the PIED tax; PSD readout Gauss–Newton; NOT estimated —
no HVPs). Per-layer `F_l = Σ_{τ,i}(y·dp)²` fused in the commit backward (y recomputed by
the reversible walk, dp resident). *Validator (T2):* K = 4 forwards of the canary µbatch
with distinct mask counters (stateless hash ⇒ reproducible draws) → direct sample
variance, exact at any θ [P]. *Cost:* in-kernel free; validator ≈0.13% at cadence 500.
*Healthy:* χ̃ flat or declining relative to per-µstep data variance — the implicit
ensemble members agree (flatness certificate [H]). *Pathological:* χ̃ rising at flat
train NLL = increments becoming individually load-bearing = co-adaptation /
function-space overfitting (the single-pass-relevant overfitting notion, jointly with
V13); F_l concentrating in few layers cross-reads with V10. *Gate G6:* in-kernel σ_η²χ₁
vs K-draw variance within ×[0.5,2] at ≥20/25 cadence points on a 2500-step run; outside
⇒ Taylor regime invalid ⇒ demote to trend-only. Also: χ̃(step-9001 batch) must exceed the
running 99th percentile [C].

**V9. SAM directional sharpness** (T2, cadence ~1000; optional).
`sharp(t) = L(w+ρĝ/‖ĝ‖; probe) − L(w; probe)` using the **existing samRho
ascend/replay/restore machinery** as a read-only probe (restore is exact). One extra
fwd+bwd per 1000 steps ≈ 0.1% wall. *Healthy:* stable/declining. *Pathological:* rising
sharpness with rising χ̃ corroborates re-sharpening; disagreement between them bounds the
mask-diagonal approximation. *Gate:* rank-correlation with χ̃ over a 2500-step run
(pre-register ρ_s ≥ 0.6, else the two measure different curvature and both are reported).

### Term IV — output measure transport (saturation, repetition, generalization proxies)

**V10. Stratified quantile trajectories** `q_α(t|s)` (T0).
Strata from token ids alone: 8 log-frequency bands (frozen reference unigram table —
stratum identity must never drift); 4 log-spaced position bands; copy-availability
`c ∈ {0,1}` (the preceding (n−1)-gram recurs in-window followed by the target, n = 8, plus
a bigram channel; rolling-hash, host or device); probe type. Quantile grid
α ∈ {.1,.25,.5,.75,.9,.99}; per-µstep D2H of per-token (ℓ, ζ, top-1) vectors ≈ 3 MB/step
overlapped ⇒ host streaming quantiles. In 1-D, quantile trajectories are exact
W₁-coordinates of the measure path [P]. *Cost:* <0.05% + 64 KB/µstep store. *Healthy:*
all quantiles descend; head strata floor first; tails still descending late (the lineage
is data-hungry — this must show). *Pathological:* median frozen while q_{.1} falls =
floor-polishing; q_{.99} rising = tail inflation (see V16); a stratum's quantiles
reversing = stratum regression. *Gate:* stratum-mixture must reproduce the scalar loss
curve to <0.005 nat [P — plumbing check]; copy-stratum median must undercut
matched-frequency no-copy median after induction onset (>0.3 nat by step 2500, else the
copy label is broken).

**V11. Saturation rates & stopping rule** `u(t), σ_b, R_s` (T0/host).
Token-utility rate `u = −dL̄/dD` with the **exact mixture decomposition** [P]
`ΔL̄ = Σ_s w̄_s ΔL̄_s + Σ_s Δw_s L̄̄_s` separating within-stratum learning from
composition drift (FineWeb shard drift lands in the second term; on fixed probes it
vanishes by construction ⇒ clean `u_fresh`). Per-band slopes σ_b with OLS CI; a stratum
is **saturated** iff CI(σ_b) ∋ 0 AND its embedding-row uptake (V15) is below floor [D —
operational definition]. Remaining-extractable `R_s` from dual-family fits
(power/exponential, selected by 70/30 out-of-sample extrapolation) — **advisory only**
[H]. **20B stopping statistic (pre-registered):** STOP-consider when for 3 consecutive
5k-step windows (i) u_fresh < 0.005 nat/1B tok, (ii) Pe_fresh < 0.5 (V14), (iii) no
stratum has R_s > 0.02 nat with rising coverage share. *Gate G7:* token-weighted Σσ_b
must reconstruct the realized NLL slope to <10% at every cadence point [P-identity];
backtest the fit family on the stored PIED-30k host logs (fit 3k–15k, predict 21k–30k
within 0.05 nat) — failure ⇒ rates-only fallback (designed-in).

**V12. Inter-stratum gaps** `Γ, G_HT, G_P⊥` (T0).
Copy gap `Γ = q_{.5}(t|c=0) − q_{.5}(t|c=1)`; head–tail gap; position gap computed
within c=0 only (removes the position×copy confound [D]). *Healthy:* Γ jumps at
induction-head formation (good phase transition) then stabilizes; G_P⊥ grows then
plateaus (long-context exploitation). *Pathological:* G_P⊥ flat from early = context
length unused (capacity signal); G_HT frozen with tail rows starving (V15) = tail
unreachable at this recipe. *Gate:* G_P⊥ > 0 and increasing on any healthy 2500-step
run; the copy adjustment must reduce the position–copy partial correlation to |ρ|<0.05.

**V13. Canary/fresh memorization probes** `Δ_mem, G_recent, G_old` (T2, cadence 100).
Canary = fixed µbatch trained on exactly once at known t₀, replayed val-mode; fresh =
matched never-trained µbatch; offset-corrected gap
`Δ_mem(t) = [L̄_fresh−L̄_canary](t) − [same](t₀⁺)`. Two canary ages (≈1k, ≈10k steps) give
a two-point retention curve. Twin-matching makes Δ_mem era-drift-immune *within run* [D].
*Cost:* ≈0.16–0.32%. *Healthy:* small transient after t₀ decaying to ≈0 within SE
(~0.03 nat after DEFF deflation, §7) — single exposures are overwritten [C, gated].
*Pathological:* Δ_mem drifting positive across successive canaries = capacity shifting
from generalization to storage; predicts wide-val stall at constant train descent.
*Gate G8 (positive control):* a 2500-step run with one shard region deliberately replayed
twice must produce a Δ_mem jump >5×SE on the replayed-region canary while the single-pass
control stays within 2×SE — the monitor must detect *engineered* memorization or it
cannot claim to detect organic memorization.

**V14. Drift/diffusion & Péclet — the anneal trigger** `v̂, D̂, Pe` (T2, reuses V13).
Same-token increments δ_i between probe replays at spacings Δ and 2Δ:
`Var(δ(Δ)) = aΔ² + bΔ` separates deterministic drift from diffusive churn [D];
`Pe = |mean δ|/SD(δ)` after subtracting the atomics reproducibility floor (double
forward at identical w, measured once). *Healthy:* Pe ≫ 1 mid-training (coherent
transport). *Pathological / decision rule:* v̂ → 0 with SD ≫ floor = **churn at zero
drift**: at constant LR this is the SGD stationary state ⇒ **anneal now** (the principled
version of the lineage's late+sharp-finish lesson); at already-annealed LR the same
signature = capacity exhaustion [H]. v̂_canary < 0 with v̂_fresh ≈ 0 = transport aimed at
seen data (transport-level overfitting). *Gate G9 (on the 20B run):* when Pe < 1
sustained 5k steps at constant LR, branch a 2500-step flat-low-LR probe; it must yield
≥0.02 nat over continuing constant-LR — else the drift–diffusion reading is falsified.

**V15. Vocabulary coverage** `PR(t), ψ_k` (T0).
Per-row cumulative update mass M_r (fused L1 accumulation in `embedding_scatter_add`;
one V-length FP32 accumulator = 128 KB); participation ratio
`PR = (ΣM_r)²/(V·ΣM_r²)`; per-band mass shares ψ_k vs the band's token-mass share.
Necessary-condition logic [D]: a tail stratum cannot improve through rows receiving no
update mass. *Healthy:* PR stabilizes near the token distribution's own PR; ψ_tail
tracks tail token mass. *Pathological:* ψ_tail → 0 while tail R_s large = tail is
**data-starved, not capacity-starved** — an argument for continuing the 20B run even if
head strata saturate. *Gate:* windows with flat tail-band mass must show tail-stratum
improvement ≈ 0; a stratum improving with zero row updates falsifies the
necessary-condition claim (would indicate context-routed improvement — itself
informative).

**V16. Loss-tail & normalizer health** `α_Hill, z_batch, |ζ| drift` (T0, host).
Hill tail index on top-k = 512 order statistics of per-token NLL; batch z-score vs
trailing median/MAD (the step-9001 detector); logZ mean and q_{.99} (already downloaded).
*Healthy:* α stable; |z| < 4; |ζ| pinned by z-loss. *Pathological:* α dropping below its
running 5th percentile = heavy-tail emergence, hypothesized hard-batch precursor [H —
the loss tail is a proxy, not a bound: per-token CE logit-grads are bounded, amplification
is through the backward]; |ζ| drift = normalizer inflation, early instability sign.
*Gate:* forward-only on the 20B run — pre-register ≥50% of |z|>6 events preceded within
50 steps by an α-alarm at <1/5k-step false-alarm rate (no historical per-token data
exists; scalars only were logged).

**V17. Copy-gain share & repeat preference** `σ_copy, ρ_rep` (T2, cadence 200).
On the fresh probe (composition term ≡ 0):
`σ_copy = w_cΔL̄_c / (w_cΔL̄_c + w_ncΔL̄_nc)` [P-decomposition] — the fraction of NLL
improvement earned on copy-available tokens. On a **template probe** (synthetic batch
with induced n-gram repeats whose ground-truth continuation *differs* from the copy
continuation at pre-registered positions):
`ρ_rep = mean[log p_θ(x_copy) − log p_θ(x_true)]` — a direct, forward-only probe of the
diagnosed repetition attractor. Complements ECHO: ECHO's `Rrow` penalizes excess copy
mass in training; V17 *measures* attractor depth without requiring ECHO enabled (and can
reuse ECHO's copy-labeling machinery). *Cost:* ≈0.08%. *Healthy:* σ_copy elevated during
induction formation then declining toward the copy token-mass share; ρ_rep < 0 stable.
*Pathological:* late σ_copy re-inflation + ρ_rep rising toward 0 = perplexity gains are
copy gains; the attractor deepens even as val NLL improves — **the monitor the lineage
lacked when the −0.62-nat reanchor ship still degenerated in free generation**.
*Gate G10:* ρ_rep must rank-correlate (Spearman ≥ 0.7) with measured free-generation
repetition from the existing `--gen-metrics` path across a 2500-step run's checkpoints;
failure falsifies the template probe as an attractor proxy [C until gated].

**V18. Contextual gain** `G_ctx` (T0).
`G_ctx = E[s_uni − ℓ]`, s_uni = running add-1 unigram surprisal (host table) — information
per token extracted from context beyond unigram statistics. *Healthy:* monotone,
decelerating. *Pathological:* G_ctx plateau while stratum slopes still negative = gains
have shifted to unigram/composition level, not learning. *Gate:* must reproduce the loss
trajectory minus computable stream unigram entropy within estimator CI [D].

### Axis → statistic map

| Axis | Primary | Corroborating |
|---|---|---|
| (1) Gradient health | V1 Λ/γ, V2 observer residual, V3 ledger, V16 tail/z/logZ | V4 clip ledger, V5 optimizer state, V6 group-GNS |
| (2) Generalization | V7 efficiency, V8 χ̃ flatness, V13 fresh-probe NLL | V10 shape, V12 gaps, V18 G_ctx; validated vs wide-val |
| (3) Overfitting | V7 ω_t/M_t, V13 Δ_mem, V8 χ̃ trend | V14 canary-vs-fresh drift asymmetry, V17 copy monoculture |
| (4) Token/capacity saturation | V11 u/σ_b/stopping rule, V14 Pe anneal trigger | V15 coverage, V10 quantile flattening, V3-PR depth utilization |

### Self-testing identities (run continuously, alarm = plumbing or math bug)

1. Ledger closure `|P_L − P_0 − Σ(Ẽ+2c)|/P_L < 1e−3` [P] (V3)
2. PIED mask calibration `m_l = 1/(1−π)` [P] (V3)
3. dp depth-flatness `π_l = 1 ± O(Lθ_max)` [P/D] (V1)
4. Stratum-mixture loss reconstruction `<0.005 nat`; token-weighted Σσ_b slope
   reconstruction `<10%` [P] (V10/V11)

---

## 7. Estimator conditioning (the honest fine print)

**Design effect.** Tokens within a document are correlated: with mean in-window document
length B̄ and post-stratification ICC ρ, `n_eff = n/(1+(B̄−1)ρ)`; plausible DEFF 10–70.
Measure ρ̂ once at setup and deflate every reported SE; block-bootstrap over µsteps for
gap CIs. Consequences: probe means (32k tokens) resolve ~0.02–0.04 nat ⇒ all probe gates
stated in ≥5×SE units; per-stratum medians resolve trends at cadence 25, never per-step.

**Nonstationarity.** All Term-II/III estimators assume drift ≫ window separation; a
regime flag (20-step loss-slope threshold) suppresses V7/V8 reporting during warmup and
recipe changes — without it the suite produces confident nonsense during exactly the
interesting transients.

**Atomics floor.** Production shape is not bit-reproducible; V2/V14 have stochastic
floors calibrated in-run (V2) or at setup via double-forward (V14). If the late-run
dispersion is comparable to the atomics floor, Pe loses meaning — this fails loudly
(measured), not silently.

**Clip interaction.** All V1/V3 ratio statistics are exactly invariant to the global clip
scalar [P]; V7's ω̂ carries a Var(c_t)-bounded bias, monitored by V4.

---

## 8. Cost roll-up

| Tier | Items | Wall |
|---|---|---|
| T0 fused/host | V1–V5, V8-inkernel, V10–V12, V15, V16, V18 | <0.15% |
| T1 cadence-25 norm pass | V6 (γ₁ + per-group partials) | ~0.01% |
| T2 probes | V13+V14 (cadence 100), V17 (200), V8-MC (500), V9 (1000) | ~0.28% |
| **Total** | | **≈0.45%** |

VRAM: per-token stat store 64 KB/µstep + coverage accumulator 128 KB + probe batches +
forward-stat persistence ~3 MB ≪ 50 MB. All default-off behind `--vitals` with
`--vitals-cadence/--vitals-probe-every` knobs; graph-incompatible host downloads follow
the SIRA-shadow precedent (accum-boundary, default-off).

**Perf mandates (PACT −40% lesson):** warp-shuffle reductions only; per-layer scalars
only; no shared-memory double atomics; contended accumulations cadence-gated. Ship
precondition: bit-identical training math with `--vitals` on vs off (monitor-only taps),
and ≤0.5% wall at production shape, n=2.

---

## 9. Comparison to existing practice

*In-repo:* the `[whisc]`/`[sorc]` monitors, grad-norm breakdown (trigger-only), SIRA
shadow/PHS/PTOC (detached diagnostics), probe-attn-gini, `--val-every` live-eval — VITALS
subsumes these patterns into one typed certificate with pre-registered alarms, and adds
the backward-pass, optimizer-state, and measure-transport layers none of them touch.
*External reference points:* gradient noise scale (McCandlish et al.) — V6 recomputes it
from the accumulation structure at zero marginal cost and localizes it per parameter
group; GSNR-generalization links (Liu et al.) — V7 derives the one-step version from the
ledger identity rather than assuming it; loss scaling-law fits — V11 uses them only as
advisory extrapolation with out-of-sample selection and a rates-only fallback; classical
train/val gap — replaced by ω_t, Δ_mem, and χ̃, which remain defined under single-pass
data. The depth-resolved certificate (V1–V3) has no external analogue because it is
CHIRON-specific: it exploits the linear p-accumulator, the reversible walk, the reanchor
double-computation, and the PIED counter-hash — structure standard transformers do not
expose.

## 10. Failure modes and mitigations (ranked)

1. **Forced-recursion confound (V1):** γ_l conflates Jacobian gain with dp-forcing shape
   → false FAST positives. Alarm on Λ conditioned on stationary V3 forcing; G1(iii)
   measures the false-positive rate directly.
2. **Copy-label ≠ copy-mechanism (V17/V12):** c_i marks opportunity, not use; healthy
   induction learning also cashes copy tokens. Pathology requires the conjunction (late
   re-inflation + ρ_rep rising + u_nc → 0); G10 is the arbiter.
3. **Adaptive-baseline masking of slow drift:** frozen absolute SLOW thresholds from a
   calibration run; recalibration required on data swaps (FineWeb) — accepted cost.
4. **Taylor breakdown of χ̃ at π=0.1:** G6's demotion path (trend-only).
5. **Composition drift masquerading as dynamics:** the exact mixture decomposition (V11)
   + fixed probes anchor all overfitting/saturation verdicts; rotate auxiliary fresh
   probes, keep one permanent for continuity.
6. **Asymptote non-identifiability (V11):** one decade of tokens cannot pin L_∞;
   designed-in rates-only fallback; R_s advisory.
7. **Hard-batch aliasing (V2/V16):** persistence requirements + co-occurrence rules;
   residual alert-fatigue risk if data hardness is heavy-tailed.
8. **κ over-interpretation (V3):** cancellation is partly functional computation (PACT E4
   washout is direct evidence); κ is descriptive; any future controller keyed on it needs
   its own gate.
9. **Hot-kernel cost creep:** the commit/reanchor/CE/Adam/clip kernels are the hottest
   paths; the perf gate is a ship precondition, not an afterthought.
10. **Single-seed rare-event statistics (V16):** one-realization spike detectors may
    simply lack events in a gate window — a no-event outcome is published as "untested,"
    not "passing."

## 11. Minimal prototype (M0 — trainer-side only, zero kernel edits)

Implementable immediately with no glades-ml changes:

1. **Clip ledger (V4-partial):** log c_t, bind fraction, ‖g‖ excess distribution
   (all host scalars already present).
2. **GNS pair (V6-global):** call `compute_grad_norm_sq` after µstep-1's backward on
   cadence-25 steps (the buffer is γ₁ exactly); run `log_grad_norm_breakdown` on the same
   cadence for per-group partials.
3. **Batch z-score + logZ stats (V16-partial):** host math on the existing per-µstep loss
   scalars and the already-downloaded logZ vector.
4. **Probe forwards (V13-partial):** reuse the `--val-every` machinery with two fixed
   µbatches (canary/fresh) at cadence 100.
5. **Host loss-history rates (V11-partial):** u(t) windows over the existing log stream.

M0 exit gate: numbers logged for 2500 steps of the flagship recipe; V6 windowed Ŝ, Ĝ² > 0
and stable; probe SEs measured (DEFF calibration); zero wall regression beyond 0.1%.

## 12. Full program

- **M1 (kernel-fused taps):** per-token NLL store (64 KB/µstep); V1 pre-clamp RMS +
  fused dp RMS; V2 residual (persist forward ReLN stats, ~3 MB); V3 ledger reductions in
  the commit kernel; V8 F_l in the commit backward; V15 coverage in scatter-add; V5 Adam
  stats; V10 stratum labeling. Each lands with: bit-parity proof (training math
  unchanged), warp-shuffle-only reductions, ≤0.5% cumulative wall (n=2), unit tests per
  the chiron-* suite pattern. Self-testing identities (§6) active from day one.
- **M2 (probes + alarm engine):** template probe (V17), mask-redraw validator (V8-MC),
  Pe two-spacing design (V14), SAM sharpness (V9); FAST/SLOW alarm bands; calibration-run
  freeze of SLOW thresholds.
- **M3 (validation gates):** G1 SORC replay; G2 step-9001 replay; G3 µstep
  self-consistency; G4 ledger/mask identities; G5 ω-vs-canary ranking; G6 χ̃ MC
  agreement; G7 slope reconstruction + PIED-30k backtest; G8 replay-injection
  memorization control; G10 ρ_rep vs `--gen-metrics`. (G9 anneal-probe rides the 20B
  run.)
- **M4 (deployment):** `--vitals` default-off on the flagship recipe; the 20B FineWeb run
  is the first full-scale consumer (stopping rule V11, anneal trigger V14, memorization
  watch V13, attractor watch V17).

### Implementation record (2026-07-15)

- Host estimator/alarm engine: `Backend/Machine Learning/Networks/chiron_vitals.{h,cpp}`
  implements V1–V18, bounded history, robust FAST alarms, absolute/self-test alarms,
  quantiles/strata, dual-family saturation fits, GNS/GSNR, drift–diffusion, coverage,
  Hill-tail, copy and contextual-gain estimators.
- CUDA taps: `gpu_chiron.cu/.h` (V2/V3/V8) and `gpu_kernels.cu/.h`
  (V1/V5/V7/V10/V15/V16), with warp-shuffle reductions and unchanged default-off
  dispatch. The int8-Adam taps retain six parameter groups and use a deterministic
  1/16 block sample at production scale with one atomic aggregate per sampled block.
  V2 stores a deterministic stride-16 token sample and V15 accumulates only on the
  cadence-gated capture step, reducing the VITALS device allocation by 82.9%.
- Trainer integration: sibling `glades-trainer/trainer/chiron_main.cpp` provides
  `--vitals`, `--vitals-cadence`, `--vitals-probe-every`, and
  `--vitals-unigram`; fixed canary/fresh probes, two-spacing replay deltas, atomics-floor
  calibration, K=4 PIED redraws, and the repetition template are cadence-gated.
- Tests: `chiron-vitals-test.cpp` covers all catalog outputs and self-tests plus CUDA
  parity for commit/BF16-SR, observer residual, Fisher, row RMS, coverage, output
  vectors, and int8-Adam read-only behavior. A one-step small-shape trainer A/B produced
  byte-identical CHRN checkpoints with VITALS off/on. The optimized M1 production-shape
  hot-path n=2 measured median wall overhead +0.213% (elapsed +0.235%) and 0.29 MB
  VITALS buffers, passing the ≤0.5% wall and ≤1% VRAM gates.
- Validation boundary: deterministic implementation tests do not substitute for the
  pre-registered production replays/correlations in G1–G10. Those gates remain open
  until their specified training runs are executed.

## 13. Open conjectures (each gated)

1. Λ crosses alarm ≥50 steps before loss divergence on SORC-class cascades (G1).
2. PIED damps the step-9001 Λ/χ̃ transient (G2; anchor 10.8→4.9).
3. κ_on < κ_off under PIED at matched steps (G4iii).
4. ω̂ ranks canary memorization across training thirds (G5).
5. χ̃ (mask-ensemble disagreement) is a usable flatness/generalization proxy — rising χ̃
   predicts wide-val stall (G6 + wide-val correlation).
6. Pe collapse at constant LR predicts anneal-extractable ≥0.02 nat (G9).
7. ρ_rep tracks free-generation repetition (G10) — if true, VITALS gives the first
   train-time leading indicator for the lineage's repetition attractor.
8. Hill-index drop precedes hard-batch spikes (V16 forward gate).

---

*Design produced 2026-07-09 via the 3-candidate research-framework-design protocol.
Candidates: LEDGER (estimation-theoretic), STROBE (dynamical-systems/observability;
selected as spine), TIDE (measure-transport). Implementation landed 2026-07-15;
long-horizon empirical gates remain governed by the pre-registrations above.*
