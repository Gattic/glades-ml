# Paradigm shift #19 — Candidate A: Per-parameter Precision Heterogeneity (PRX)

**Formulation class:** KKT-derived per-parameter precision schedule
adapted online during training.
**Author pass:** subagent-dispatched design (2026-04-22).
**Status:** candidate — awaiting selection at the shift-19 gate.

---

## 1. Short name and core thesis

**PRX — Per-parameter Precision Heterogeneity.**  Working codename:
*Fisher-priced bit width*.

Every paradigm shift from #1 through #16 treats precision as a global
property.  Shift #5 stores all weights at BF16.  Shift #3 stores all
Adam state at int8.  Shift #4 emits all gradients at BF16.  Shift #11
(MFIO) removes per-parameter moments entirely, making the remaining
moment state layer-uniform.  No shift has personalised **storage
precision** at the granularity of the individual parameter.

Yet the distribution of per-parameter Fisher information F_i ≈ E[(∂L/∂θ_i)²]
is massively heavy-tailed (the very property GFIB uses to sparsify
*updates*).  A parameter whose Fisher is 10⁴× smaller than the median
can tolerate 10² times more quantization noise variance without
commensurate loss degradation.  PRX spends bits where they matter:
high-F̂ parameters at BF16 or FP16, mid-F̂ at int8, low-F̂ at int4, and
near-zero-F̂ at signed 1-bit.  The precision assignment is recomputed
periodically from a running Fisher EWMA and adapted to a global memory
budget by a KKT-tied Lagrangian, giving an **optimal** per-parameter
schedule rather than a hand-tuned block-wise quantization.

The novelty is not any one of the tier representations (int4 per-row
absmax, 1-bit sign × scale — both from prior art).  It is the
**online, KKT-derived, Fisher-monotone assignment policy during
training** that places parameters into tiers and moves them between
tiers as the training trajectory evolves, with explicit hysteresis,
per-step re-quantization budgets, and composition with GFIB (shift
#17, deferred) which supplies F̂ for free.

---

## 2. Primitive objects and state space

Let the network have N scalar parameters indexed i = 1..N.  Let
𝔅 = {1, 4, 8, 16} be the precision tier set.

| symbol          | shape / dtype  | meaning                                      |
|-----------------|----------------|----------------------------------------------|
| θ_i             | storage varies | logical parameter value                      |
| θ_i^q           | b_i bits       | stored quantized parameter                   |
| g_i             | BF16           | instantaneous gradient (from shift #4)       |
| F̂_i            | BF16           | EWMA of g_i² — per-parameter Fisher proxy    |
| b_i ∈ 𝔅        | 2 bits         | current precision tier                       |
| μ               | FP32           | KKT multiplier / budget-shadow price         |
| q(b)            | scalar         | quantization noise variance at b bits        |
| K ∈ (0, 1)      | FP32           | re-quantization cap (max fraction / window)  |
| 𝓑_target       | FP32           | target total bits per parameter (e.g. 5.6)   |
| s_i             | FP16           | per-row scale (for int4/int1 tiers)          |
| (k_P, k_I)      | FP32 scalars   | PI-controller gains                          |

**State space.**  Σ = (θ^q, F̂, b, μ, s).  The persistent storage cost
is dominated by θ^q (variable per i) plus F̂ (a fixed BF16 overhead of
2 B/p × N) plus b (2 bits/p) plus s (per-row, negligible: ~0.01% of N).
F̂ is the same 4.46 GB line item GFIB pays at 2.23 B; if shift #17
(GFIB) lands before PRX, PRX reuses F̂ at zero marginal memory cost.
This is the composition hinge.

**Quantization tier semantics.**

- **b_i = 16 — BF16.**  Same representation as shift #5.  q(16) ≈ 2⁻¹⁶·‖θ_row‖² per row (per-element variance), but the relevant figure for KKT bookkeeping is the per-parameter squared-error variance σ_q²(16) ≈ 2⁻²³·Var(θ_row).
- **b_i = 8 — int8 per-block absmax.**  Store θ_i as a signed int8 times a per-block (e.g. 32-element block) FP16 absmax scale.  σ_q²(8) = s_block² / 12·(1/127)² by uniform-rounding theory.
- **b_i = 4 — int4 per-row absmax.**  4-bit signed integer times a per-row FP16 absmax scale.  Packs two params per byte.  σ_q²(4) = s_row² / 12·(1/7)² ≈ 1.68e-2 × s_row².
- **b_i = 1 — sign × per-row magnitude.**  Store the sign bit, reconstruct θ_i ≈ sign_i · m_row where m_row is the per-row mean absolute value (FP16).  σ_q²(1) = Var(|θ_row|) which for approximately-Gaussian rows ≈ (1 − 2/π)·s_row² ≈ 0.36·s_row².

In the ratio q(b_j)/q(b_k) (which is what matters for KKT), the per-row
scale s cancels.  The canonical form used throughout §4 is

    q(1) / q(4) ≈ 22,    q(4) / q(8) ≈ 64,   q(8) / q(16) ≈ 256.

So halving the bit width costs roughly 10-50× in noise variance.
Keep this ratio as the central KKT input.

---

## 3. Evolution — one optimizer step

A step is nearly unchanged from the current stack; PRX intervenes in a
dequant-before-GEMM and a periodic re-quantization:

1. **Dequant cache.**  At step start, for each weight matrix W, ensure
   a BF16 activation-cast cache `W_bf16[·]` exists — row i at tier b_i
   is dequantized into that row of the cache.  Only rows whose b_i
   changed last re-quant are re-dequantized; the rest are clean.
   Cache is the input to BF16 cuBLAS GEMM (same path as shift #5).
2. **Forward, backward, loss** (unchanged).  Produces g_i at BF16.
3. **Fisher EWMA.**  F̂_i ← β_F · F̂_i + (1 − β_F)·g_i² with β_F = 0.99.
   (Identical to GFIB's update.  Under composition with GFIB this line
   is shared, not duplicated.)
4. **Adam/MFIO step** on the BF16-cached dequantized W (not on θ^q
   directly).  Write result into the BF16 cache tile for this row.
5. **Re-quantization** (every K_Q = 100 steps; otherwise skip):
   - Compute proposed b_i^* = clamp(round(log₂(F̂_i / μ)),  {1, 4, 8, 16}).
   - Apply **hysteresis** (§9-F2): accept b_i^* only if |log₂(F̂_i) − log₂(μ) − b_cur| > δ_hyst, δ_hyst = 0.15.
   - Apply **churn cap** (§9-F2): if count of accepted tier changes exceeds 1% × N for this K_Q window, retain the K most Fisher-mass-weighted-important transitions and defer the rest.
   - For each accepted row, quantize the (now-Adam-updated) BF16 cache back into its tier storage θ^q_i.
6. **μ feedback.**  Measure realized bits-per-param 𝓑̂_obs = (1/N)·Σ_i b_i; error e_n = 𝓑̂_obs − 𝓑_target.
   μ_{n+1} = clip(μ_n + k_P·e_n + k_I·Σ_j e_j, μ_low, μ_high).
   Gains (k_P, k_I) chosen so that μ settles within O(K_Q·100) steps.

Per-step extra compute (between re-quants): zero beyond a fresh EWMA
update and μ feedback, both O(N) fused into existing kernels.  At
re-quant boundaries, cost is one O(N) scan + per-accepted-row re-pack
(≤1% churn cap ⇒ 0.01N re-packs per window), amortised over K_Q steps
at <0.1% of step time.

---

## 4. KKT derivation of the optimal assignment

**Outer problem.**  Minimize expected loss degradation induced by
quantization subject to a memory budget.  Model the per-step loss
increase from quantizing parameter i at b bits, to leading order in
the quantization noise, as

    ΔL_i(b) ≈ ½ · σ_q²(b) · F_i               (1)

(second-order Taylor expansion of L in ‖Δθ_i‖ around the master
weight; the cross-terms vanish in expectation under unbiased
quantizers like stochastic rounding).  The total budget constraint
is Σ_i b_i ≤ 𝓑_target · N.

**Continuous relaxation.**  Treat b ∈ ℝ⁺ with σ_q²(b) = σ₀²·2^{-2b} (the exact log-linear law for uniform rounders; stochastic rounding matches this bound to within a small constant).

**Lagrangian.**

    𝓛(b, μ) = Σ_i ½·σ₀²·2^{-2b_i}·F_i  +  μ·(Σ_i b_i − 𝓑_target·N)      (2)

**Stationarity.**  ∂𝓛/∂b_i = −σ₀²·ln2·2^{-2b_i}·F_i + μ = 0, giving

    b_i^* = ½·log₂(σ₀²·ln2·F_i / μ)                                    (3)

i.e. **b_i^\* is monotone in F̂_i and linear in log(F̂_i / μ)**.  This is
the PRX policy.  Parameters with high Fisher get proportionally more
bits, with slope ½ bit per doubling of F_i.  The Lagrangian μ is the
marginal bit-cost of global memory — the shadow price.

**Projection onto tier set.**  We need b_i ∈ {1, 4, 8, 16}.  Rounding
(3) to the nearest element of 𝔅 induces a piecewise-constant policy
with three tier boundaries.  Up to a constant absorbed into μ the
boundaries are at F̂_i = μ, 2⁶·μ, 2¹²·μ.  The PI controller in §3
drives μ to the value at which 𝓑_target is realized.

**Uniqueness.**  Convex objective, linear constraint, single
multiplier — the KKT point is unique up to tier-rounding ties.

**Optimality bound.**  Relative to any feasible policy π with the same
total budget, the PRX loss-degradation penalty satisfies

    E_π[ΔL] − E_PRX[ΔL]  ≥  0,

with equality iff π coincides with the rounded KKT schedule.  This is
the information-theoretic claim.

---

## 5. Mechanism — memory, speed, stability

**Memory (the headline).**  At N = 2.23 B, with a tier distribution
aimed at 𝓑_target ≈ 5.6 bits/param (Pareto-extrapolation of the
Fisher distribution suggests ~10% of params at BF16, ~20% at int8,
~65% at int4, ~5% at 1-bit):

    weights PRX   ≈ 0.10·16  + 0.20·8  + 0.65·4  + 0.05·1  ≈ 5.85 bits/p.
    weights total ≈ 5.85 bits × 2.23 B = 1.63 GB  (vs 4.46 GB BF16).

A 2.7× memory reduction on weights, unlocking ~2.8 GB of VRAM for
larger model, longer context, or larger batch.  Composed with OVFG
(#9) for gradients, MFIO (#11) for moments, and Stiefel/MPOT (#7/#10)
for weight structure, PRX lets the 2.23 B weight line shrink to ~1 GB
total persistent on-GPU storage.

**Speed.**  The BF16 cuBLAS path is unchanged: PRX dequantizes rows
into a BF16 tile cache at step start and hands the cache to the same
`sgemm_rowmajor_bf16` kernels used today.  The net wall-clock effect
is (dequant overhead) − (cache-fetch bandwidth savings).  At the
5.85-bit average the raw DRAM traffic for weight reads during dequant
is 37% of the BF16 traffic, but dequant is compute-bound only for the
1-bit tier (sign expand + scale multiply).  For 4-bit and 8-bit rows
the dequant is bandwidth-bound and pays for itself against the smaller
DRAM footprint.  Net: approximately parity on step time, small
speed-up (1-3%) from better cache behaviour on warm mini-epochs, but
we do not claim a wall-clock win.  The value is **memory**, which
unlocks larger N which unlocks better loss — the usual scale-law
arbitrage.

**Stability.**  The KKT schedule (3) places low bits only where F_i is
small, which is precisely where the loss is insensitive to
quantization noise per equation (1).  The stochastic-rounding
formulation of each quantizer guarantees unbiasedness (E[θ_i^q] = θ_i),
which composes naturally with BF16-master stochastic rounding (the
existing shift-#5 mechanism).  No double-biased noise: SR at the
BF16→tier step, not at the FP32→BF16 step.  Formal stability bound:

**Theorem (informal).**  If the per-parameter loss-Hessian eigenvalues
are bounded and σ_q²(b_i)·F_i ≤ ε for all i (the KKT-optimal schedule
at 𝓑_target satisfies this with ε set by μ), then the trajectory of
quantized training stays within O(√(T·ε/N)) of the full-precision
trajectory over T steps in L²-norm, and the final loss satisfies
|L_PRX − L_FP32| ≤ O(√ε).  Sketch: standard SGD-with-noise analysis
(Ghadimi-Lan) applied to unbiased multiplicative noise with variance
σ_q²(b)·‖θ‖²; the KKT budget bounds the noise variance uniformly.

---

## 6. Quantization tier specification

### Tier b = 16: BF16
Representation: 16-bit brain-float.  Already in use.  q(16) taken as
the baseline unit.  SR at the FP32→BF16 step already landed in shift #5.

### Tier b = 8: int8 per-block absmax
Block size B_q = 32.  Per block, compute absmax α_B = max_j |θ_j|;
store θ_j as int8 round(θ_j · 127 / α_B) with SR.  Dequant:
θ_j ≈ q_j · α_B / 127.  σ_q² per parameter = (α_B / 127)²/12.  This
is exactly shift #3's representation promoted from Adam moments to
weights.

### Tier b = 4: int4 per-row absmax
Per-row absmax α_r = max_j |θ_j| within row r.  Store θ_j as signed
4-bit round(θ_j · 7 / α_r) with SR, two params per byte.  Dequant:
θ_j ≈ q_j · α_r / 7.  σ_q² per parameter = (α_r / 7)²/12.  Row-level
scale is O(rows) FP16 scalars — 0.01% overhead.

### Tier b = 1: sign × per-row magnitude
Per-row mean-absolute m_r = (1/R) Σ_j |θ_{r,j}| (FP16).  Store the
sign bit of θ_j only.  Dequant: θ_j ≈ sign(θ_j) · m_r.  σ_q² per
parameter = Var(|θ_{r}|) ≈ (1 − 2/π)·s_r² ≈ 0.36·s_r² for
Gaussian-like rows.  This is the BitNet-1-bit representation (not
1.58-bit — we use exact ±m_r, not the {−m,0,+m} ternary).  The 0
tier is omitted because it requires extra bits to encode the sparse
set.

**Rationale for 4-tier rather than continuous.**  Hardware-dispatch
kernels are easier with a small fixed set.  Four tiers span the
interesting precision range (1 / 4 / 8 / 16 = 0.25 / 1 / 2 / 4
bytes-equivalent).  Continuous precision (Chinchilla-style fractional
bits) requires codebook-style representations whose hardware cost
exceeds the bit savings at N < 10¹⁰.

---

## 7. Re-quantization cadence and the churn law

**Why not per-step.**  Per-step re-quantization costs O(N) bit-packing
operations per step.  Per-parameter Fisher F̂_i changes slowly (EWMA
with β_F = 0.99 has 100-step half-life), so daily re-assignment
already captures the relevant dynamics.  We set K_Q = 100.

**Expected churn rate.**  Under the heavy-tailed Fisher hypothesis
(Pareto α ≈ 1.5–2.0; shared with GFIB's central conjecture), the
fraction of parameters crossing a tier boundary per K_Q window is
bounded by the rate of change of rank-quantiles of F̂.  Empirically
on pile_small smoke tests we expect ~0.3-0.5% / K_Q-window; the 1%
churn cap in §3 is a safety clamp, not a typical-case constraint.

**Amortisation.**  Re-quant at 1% × N × (bit-pack cost of ~2 ns/param)
per 100 steps = 0.02 ms/step at N = 2.23 B.  Step time is ~35 ms.
Overhead < 0.1%.

---

## 8. Composability with shipped and deferred shifts

| shift        | interaction                                                    |
|--------------|----------------------------------------------------------------|
| #1 CHIRON    | orthogonal — activation tape unchanged                         |
| #2 TC attn   | orthogonal                                                     |
| #3 int8 Adam | **composable**: Adam state uses the same tier-quantization infrastructure; m, v tiers track weight tiers — low-F̂ params cascade to low-precision moments |
| #4 BF16 grads| orthogonal (gradients always BF16 at creation)                |
| #5 SR BF16 weights | **subsumed**: PRX's BF16 tier IS shift #5; other tiers extend it |
| #6 local-window attn | orthogonal                                               |
| #7 Stiefel × Σ | **direct gain**: Stiefel factor rows can be assigned tiers independently of Σ diagonal (Σ always BF16 — it's rank-1) |
| #9 OVFG      | orthogonal — grads factored in BF16, then used to update PRX-tiered weights |
| #10 MPOT     | **composable**: MPOT cores are per-tier-assignable (big fat cores BF16, small edge cores int4) |
| #11 MFIO     | orthogonal — MFIO removes moments entirely, PRX moves to weights instead |
| #12 DFA      | orthogonal — DFA gradients still produce g_i at BF16          |
| #13 TRCD     | orthogonal                                                     |
| #16 LCP      | orthogonal — LCP's detail network is itself a small weight matrix, gets its own tier |
| #17 GFIB     | **tight composition**: GFIB's F̂ is exactly PRX's input; shared memory line, shared EWMA kernel; PRX uses F̂ to assign tiers, GFIB uses F̂ to gate updates |
| #18 SGS      | orthogonal — SGS surrogates get their own PRX tiers           |

The composition with GFIB (#17) is the architectural hinge: F̂
appears in both schedulers and is persisted once.  The 4.46 GB F̂
line for 2.23 B pays for both shifts; taken alone PRX cannot
justify the F̂ cost, but jointly (GFIB + PRX) each sees the 2.23 GB
memory savings it would see alone plus the F̂ cost is amortised.

---

## 9. Failure modes and mitigations

**(F1) Non-stationary F̂.**  If the Fisher distribution shifts
during training (curriculum transitions, modality changes,
fine-tuning onset), tiers reshuffle.  Mitigations:
1. **Hysteresis band δ_hyst = 0.15** on tier boundaries: a row
   already at tier b must have log₂(F̂_i) exceed the neighbouring
   boundary by 15% before switching.
2. **Faster EWMA on detection**: if a global drift indicator
   (moving-window KL-divergence of the F̂ histogram over a
   500-step window) exceeds a threshold, temporarily drop β_F to
   0.95 until drift subsides.
3. **Dense re-quant on emergency**: operator tool to force a
   full re-quant without churn cap.

**(F2) Tier-boundary oscillation.**  Parameters flipping between
adjacent tiers step-over-step inflate re-quant bandwidth and
destabilise training.  Mitigations:
1. **Churn cap** (1% of N per K_Q window) — a hard ceiling.
2. **Hysteresis** (above) — soft prevention.
3. **Per-row rather than per-parameter tiers** for int4 and
   1-bit: whole rows move together, amortising per-row scale
   updates across many params.

**(F3) Quantization of Adam state compounds with weight
quantization.**  If both θ_i and (m_i, v_i) are at int4, the noise
in the update compounds multiplicatively.  Mitigations:
1. **Asymmetric tiering**: Adam-state tier = max(weight tier, 4).
   Never go below int4 for moments, even when the weight is at
   1-bit.  The moment memory still benefits from PRX (65% × 8-bit
   vs BF16 is 2× savings) without compounded instability.
2. **Master-BF16 option**: for the 10% BF16-weight tier, keep
   FP32 master (as today).  For lower tiers, drop the FP32
   master — the SR-BF16 master is the canonical representation.

**(F4) Unbounded-rank row dispersion under 1-bit tier.**  A row
whose elements span many orders of magnitude is poorly
represented by sign × mean-abs.  Mitigations:
1. **Row-dispersion guard**: if kurtosis(θ_row) > κ_max = 15, row
   is ineligible for 1-bit and pinned at ≥ 4-bit.
2. **Group-wise instead of row-wise** for extremely wide matrices
   (ff_width = 4096 typical): split the row into groups of 256
   with per-group absmax / mean-abs, at the cost of negligible
   extra scale storage.

**(F5) Reproducibility.**  All re-quant randomness (SR noise)
draws from a dedicated Philox stream per our
DETERMINISM_AND_CONCURRENCY policy.  Unit test
`PRXDeterminismTest` asserts identical tier assignments and
identical quantized values under a fixed seed.

---

## 10. Memory and wall-clock ledger at 2.23 B params, 16 GB

**Baseline (shifts 1-16 landed):**
- weights (MPOT × Stiefel × BF16 master):   ~2.2 GB
- gradients (OVFG factored):                ~600 MB
- Adam int8 or MFIO-layer-σ:               ~1.5 GB
- activations (CHIRON × LCP):               ~8 MB
- F̂ (if GFIB also lands):                  +4.46 GB
- scratch/logits:                          ~1.5 GB

**With PRX (5.85 bits/param target):**
- weights:                                 ~1.6 GB (from 2.2 GB: −600 MB)
- Adam state (asymmetric tier ≥ 4):         ~800 MB (from 1.5 GB: −700 MB)
- F̂ (reused from GFIB):                    +0 new (amortised)
- tier map b:                               ~550 MB (2 bits/p × 2.23 B)
- per-row scales:                          <10 MB

**Net savings at 2.23 B:**  ~750 MB of weights + ~700 MB of Adam −
550 MB of tier map = ~900 MB of VRAM freed without counting GFIB
composition.  With GFIB's 4.46 GB F̂ line shared, the joint GFIB +
PRX system has ~5 GB net savings vs either alone.

**At 3-4 B (the next scale target):**  The savings scale linearly in
N, so at 4 B total weight memory goes from 8 GB (BF16) to ~3 GB —
more than halving the single largest memory line item and making
4-5 B feasible on 16 GB.

**Wall-clock.**  ~parity on step time (§5).  The win is the memory
dimension, which opens a new scaling axis.

---

## 11. Relation to prior work

- **BitNet (Wang et al. 2023, 2024).**  Uniform 1.58-bit ternary
  weights throughout training.  PRX differs by being
  **heterogeneous** (not every param at 1-bit) and **Fisher-derived**
  (not uniform).  The 1-bit tier in PRX borrows BitNet's
  sign × per-row magnitude representation.

- **Mixed-precision training (Micikevicius et al. 2018).**  Uniform
  FP16 weights + FP32 master.  Not per-parameter.  PRX generalises
  this to four tiers with per-parameter assignment.

- **GPTQ / AWQ (Frantar 2022, Lin 2023).**  **Post-hoc** quantization:
  after training, find the optimal quantization per weight block.
  PRX runs quantization optimisation **during** training, which (a)
  lets the loss trajectory adapt to the chosen precision, and (b)
  uses the trajectory's Fisher rather than a fixed activation
  calibration set.

- **k-means / codebook quantization (Han et al. 2015, LLM.int8).**
  Discrete codebook lookup; different noise structure.  Incompatible
  with SR and with our per-row absmax convention.

- **LLM.int8 (Dettmers 2022).**  Identifies "emergent outlier
  features" and keeps them at higher precision.  This is a
  **feature-based**, not a **parameter-based**, policy and is
  post-hoc.  PRX generalises to a per-parameter, KKT-derived,
  online policy.

- **Adaptive rounding (Nagel et al. 2020).**  Learns per-weight
  rounding decisions via a small auxiliary loss.  Orthogonal to
  PRX — PRX decides bit count; adaptive rounding decides
  within-tier rounding policy.  Composable.

**Novelty claim.**  The combination of (a) KKT-derived log-Fisher
threshold policy for per-parameter precision, (b) PI-adaptive
Lagrangian μ tracking a memory budget, (c) hysteresis + churn-cap
stability envelope, (d) online re-quantization during training
rather than post-hoc, (e) tight composition with GFIB via shared
F̂ infrastructure, appears unattempted in the literature.

---

## 12. Implementation sketch and evaluation plan

**Deliverables (≈ 3 engineering weeks):**

1. `gpu_prx.{h,cu}` (~800 LOC):
   - `prx_assign_tiers(F̂, μ, b_prev, b_out)` — compute b_i^*
     with hysteresis, clamp to tier set
   - `prx_quantize_row_int4(θ_row_bf16, out_int4_row, out_scale)`
   - `prx_quantize_row_int8_block(θ_bf16, out_int8, out_scales)`
   - `prx_quantize_row_sign_mag(θ_row_bf16, out_sign, out_mag)`
   - `prx_dequantize_to_bf16(θ_tier, tier_id, scale, out_bf16)`
     — row-granular dispatch on tier_id
   - `prx_pack_mixed_dequant(θ^q, b, scales, out_bf16_cache)` —
     one kernel that walks rows, dispatches to the correct
     dequantizer per row, fills the BF16 GEMM input tile
2. `mu_pi_controller.{h,cpp}` (~80 LOC) — reused from GFIB /
   TRCD's controller pattern.
3. Unit tests:
   - `PRXTierQuantParityTest` — each tier's quant/dequant is unbiased
     on Gaussian data: E[θ_q] = θ within SR noise tolerance
   - `PRXBudgetControllerTest` — μ converges to 𝓑_target within
     10·K_Q steps
   - `PRXHysteresisTest` — no tier flips on F̂ trajectories within
     ±δ_hyst of boundary
   - `PRXDeterminismTest` — identical tier map + quantized values
     under same seed
   - `PRXEndToEndConvergenceTest` — 1000-step pile_small run,
     loss within 5% of BF16 baseline at 𝓑_target = 5.6
4. Trainer flags: `--prx-target-bits 𝓑`, `--prx-requant-cadence K_Q`,
   `--prx-hysteresis δ_hyst`, `--prx-churn-cap 0.01`.

**Validation.**

1. **pile_small (80 M, L=12, 2000 steps)**: sweep 𝓑_target
   ∈ {16, 10, 6, 4, 2} against BF16 baseline.  Measure final NLL
   and memory footprint.  Target: 𝓑 = 6 within 3% of BF16 NLL.
2. **pile_large (2.23 B, full stack)**: memory smoke test plus
   1000-step convergence.  Target: memory reduced by ≥ 700 MB, NLL
   degradation < 5%.
3. **Ablations**:
   - Uniform int4 (no F̂ scheduling) baseline
   - Uniform int8
   - Magnitude-instead-of-Fisher policy (control for KKT derivation)
   - No hysteresis (expect oscillation)
   - No churn cap (expect re-quant cost blow-up)
4. **Composition with GFIB (once #17 lands)**: joint-F̂ memory
   accounting, update gating × tier assignment interaction.

---

## 13. Open conjectures

1. **Fisher-quantization alignment hypothesis**: the top 10% of
   per-parameter Fisher mass concentrates in ≤ 10% of parameters, and
   those parameters suffer disproportionately under low precision.
   Falsifiable by binning params by F̂ quantile and running uniform
   quantization per bin.
2. **KKT-optimality over heuristic policies**: at matched 𝓑_target
   (bit-budget), PRX's KKT-derived policy dominates magnitude-based
   and uniform policies on final NLL.  Falsifiable by the ablation
   grid in §12.
3. **1-bit tier viability at scale**: for large matrices (ff_width ≥
   2048), the 1-bit sign × mean-abs representation captures enough of
   the low-F̂ tail that < 5% of params at 1-bit loses < 1% NLL.
   Falsifiable by sweep on the 1-bit fraction at fixed 𝓑_target.
4. **Composition multiplier**: joint PRX + GFIB delivers strictly
   greater-than-sum memory savings via shared F̂.  Falsifiable by
   direct measurement once both shifts ship.

---

*End of candidate A.*
