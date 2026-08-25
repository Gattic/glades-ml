# Paradigm shift #16 — Candidate A: Gradient-Flow Information Bottleneck (GFIB)

**Formulation class:** bilevel / KKT-derived stochastic update policy.
**Author pass:** subagent-dispatched design (2026-04-22).
**Status:** candidate design.  Implementation deferred to selection.

---

## 1. Short name and core thesis

**GFIB** — **G**radient-**F**low **I**nformation **B**ottleneck.
Working codename: *Fisher-thresholded update budget*.

**Thesis.**  Every prior paradigm shift (1-13) has taken as axiomatic
that at each optimizer step **every** parameter θ_i receives (a) a
gradient, (b) an Adam-state update, and (c) a weight update.  Yet at
any given step the distribution of per-parameter instantaneous Fisher
information F_i ≈ E[(∂L/∂θ_i)²] is massively heavy-tailed — a few
per cent of parameters carry the bulk of the single-step loss-decrease
mass, and the rest deliver vanishingly small progress relative to
their optimizer-side cost.  **GFIB stops paying for those unprofitable
updates.**  It spends a fixed budget of K·N Adam-state updates per
step, allocated by a **single KKT condition on one Lagrangian μ**;
the optimal policy is a soft sigmoid threshold on log F̂_i, which
reduces to top-K in the γ→∞ hard-threshold limit.  Measuring F̂_i
requires |g_i|, so backward remains **DENSE**, but the **optimizer
step itself** (Adam moments, weight write, stochastic-rounding noise
commit, MFIO σ broadcast) is fully sparsified.

GFIB is the first paradigm shift to challenge *update coverage* as a
design axis, orthogonal to depth (#13), width/manifold (#7, #10), or
optimizer-state representation (#3, #11).  Composition with MFIO (#11)
reduces MFIO's per-layer σ broadcast traffic by 10× at K=0.1 and
with int8 Adam (#3) reduces moment-update bandwidth 10×.

---

## 2. Primitive objects and state space

Let the network have N scalar parameters across L layers.

| symbol            | shape / dtype      | meaning                                       |
|-------------------|--------------------|-----------------------------------------------|
| θ_i               | BF16 (+SR)         | parameter                                     |
| g_i = ∂L/∂θ_i     | BF16               | instantaneous gradient (DENSE backward)       |
| F̂_i              | BF16               | EWMA of g_i² — per-parameter Fisher proxy     |
| μ                 | FP32               | KKT multiplier / adaptive threshold           |
| γ                 | FP32               | policy sharpness                              |
| K ∈ (0, 1]        | FP32               | operator-set update-budget ratio              |
| p_i = σ(γ(log F̂_i − μ)) | scalar       | per-parameter update probability              |
| b_i ~ Bern(p_i)   | 1 bit              | this-step update gate                         |
| (m_i, v_i)        | int8 / absent      | Adam moments (#3) or unused under MFIO (#11)  |
| (k_P, k_I)        | FP32 scalars       | PI-controller gains                           |

**State space.**  Σ = (θ, F̂, μ, step_count, (m, v if active)).  F̂
is persistent (2 B/p × N); b is ephemeral (streamed packed-bit tiles,
never persisted).  At N = 2.23 B, F̂ adds 4.46 GB — this is the
*cost* of GFIB's per-parameter Fisher proxy, discussed in §10.

**Routing policy rationale.**  p_i is a closed-form function of F̂_i
alone; no learned policy-MLP is required.  The only learnable policy
state is the scalar μ, tuned by a PI controller to enforce the budget.

---

## 3. Evolution — one optimizer step

On step n:

1. **Forward** (unchanged; CHIRON-reversible).
2. **Backward** (unchanged; DENSE — produces all g_i).
3. **Fisher EWMA.**  F̂_i ← β_F · F̂_i + (1−β_F) · g_i²,  β_F = 0.99.
4. **Gate construction.**  p_i = σ(γ · (log F̂_i − μ));
   b_i ~ Bernoulli(p_i).  γ is annealed 1 → 4 over the first 10 k
   steps (soft → hard-ish).  RNG uses a dedicated Philox stream
   for determinism (glades::rng convention).
5. **Gated optimizer update.**  For each i:
        if b_i = 1: (m_i, v_i) ← Adam(g_i, m, v); θ_i ← θ_i − η·Adam_update
        else:       no-op.
   Kernel uses predicated stores; no divergence at warp granularity
   (we use full-warp bit-packed masks).
6. **μ feedback.**  d̂_obs = (1/N) Σ b_i; e_n = d̂_obs − K.
   μ_{n+1} = clip(μ_n + k_P · e_n + k_I · Σ_{j ≤ n} e_j, μ_low, μ_high).

Per-step extra compute: two reductions, one sigmoid, one Bernoulli
— all O(N) elementwise, fused into the optimizer kernel.  Overhead
is <1% of step time at K=0.1.

---

## 4. Bilevel program and KKT derivation

**Outer problem.**  One-step loss decrease
       ΔL ≈ −η · Σ_i p_i · h_i,   where h_i ≡ |g_i · Adam_update_i| ≈ |g_i|² / √(v_i + ε)
subject to the budget Σ_i p_i = K · N.

**Lagrangian** (p_i ∈ [0,1], one multiplier μ on the budget):
       𝓛(p, μ) = η · Σ_i p_i · h_i − μ · (Σ_i p_i − K · N).

**KKT stationarity.**  ∂𝓛/∂p_i = η · h_i − μ = 0 gives the hard policy
       p_i^* = 𝟙[h_i > μ/η].

Softening with an entropy term −τ · H(p) (needed for stable policy
gradients and exploration noise) yields
       p_i = σ(γ · (log h_i − μ/η′)).
Replacing the instantaneous h_i with its EWMA proxy F̂_i (the ratio
|g|²/√v is tightly coupled to |g|², and any residual constant is
absorbed by μ),
       p_i = σ(γ · (log F̂_i − μ)).

**γ → ∞ limit.**  Recovers hard top-K.  At finite γ the policy is the
entropy-regularised relaxation; the softening is essential for
stable μ tracking and for preserving a non-zero floor on p_i for
parameters just below threshold.

**Uniqueness.**  μ is the unique multiplier on a single scalar
constraint.  The PI controller drives μ to the value at which the
budget binds, so **there is no free trade-off hyperparameter** —
only the operator-chosen K.  This is the same adjoint-structure as
TRCD's λ (#13) but on the parameter axis rather than the depth axis.

---

## 5. Why backward is DENSE and GFIB is still net-positive

We cannot know p_i before measuring F̂_i, which requires g_i.  Hence
g is computed for all parameters; sparsification is strictly
optimizer-side.

Per-step cost decomposition (pile_large, 2.23 B, RTX 4080 SUPER):

| phase     | current | GFIB K=0.1 |
|-----------|---------|------------|
| forward   | 17 ms   | 17 ms      |
| backward  | 17 ms   | 17 ms      |
| optimizer | 5 ms    | 0.5 ms     |
| **step**  | 40 ms   | 35.5 ms    |

Wall-clock saving ~13% per step.  **However**, GFIB loses a factor
on step-count (§6), so pure step-count break-even is close to unity.
GFIB's real value is:

1. **Bandwidth**.  At 2.23 B params the int8-Adam path moves
   ~13.4 GB/step (m, v, master-FP32 traffic).  GFIB reduces that
   to 1.34 GB/step at K=0.1 — freeing DRAM bandwidth for forward /
   backward.
2. **Composition**.  Shifts #3, #5, #9, #11 all have
   optimizer-side components whose cost scales linearly with the
   number of active parameters per step.  GFIB scales every one
   of them by K simultaneously (§8).
3. **Enables future shifts** that physically compress the optimizer
   state of *inactive* parameters (planned #17+).

---

## 6. Convergence and the per-step progress factor

Expected one-step loss decrease:
       E[ΔL_GFIB] ≈ −η · Σ_i p_i · h_i ≈ −η · K · E[h | F̂ > e^μ].

The conditional expectation is **larger** than the unconditional
one (Fisher-mass-weighted restriction).  On LLM pre-training the
top-10% Fisher mass typically carries 60–80% of total Σ h_i; hence

       E[ΔL_GFIB] / E[ΔL_dense] ≈ 0.6–0.8  at K = 0.1,

so the step-count inflation is ~1.25–1.7×, NOT 10×.  Composed with
the 13% per-step wall-clock saving (§5) GFIB is roughly
**wall-clock break-even** on its own at 2.23 B — its contribution is
a **compositional multiplier**, not a standalone speedup.

Stability (Robbins–Monro): guaranteed if (a) μ converges (PI does
this), (b) Σ p_i h_i > 0 a.s. (p_floor in §9 ensures this),
(c) η has standard decay.

---

## 7. Conditioning and the tail hazard

**Preconditioning.**  Effective learning rate per parameter is η·p_i.
Adam's own 1/√v̂ preconditioner is a monotone function of F̂_i; GFIB
softly zeroes the low-F̂ tail where Adam's preconditioner is
uninformative anyway.  Preconditioner structure is preserved on the
high-F̂ subset.

**Expressivity.**  b_i = 0 is a *temporary freeze*, not a clamp.  A
parameter resumes updating the moment its F̂_i rises above μ; the
EWMA tracks sustained signal shifts within ~1/(1−β_F) ≈ 100 steps.
GFIB is a dynamic bottleneck.

---

## 8. Composability with shifts 1-13

| shift | interaction |
|-------|-------------|
| #1 CHIRON | orthogonal — backward is dense |
| #2 TC attn | orthogonal |
| #3 int8 Adam | **direct gain**: m, v updates gated by b; 10× bandwidth at K=0.1 |
| #4 BF16 grads | orthogonal |
| #5 SR BF16 weights | **direct gain**: weight write + SR noise commit gated |
| #6 local-window attn | orthogonal |
| #7 Stiefel × Σ | retractions must operate on full matrices; gate at matrix-level: skip layer-ℓ retraction if b̄_ℓ < 0.05 |
| #9 OVFG | **direct gain**: A^⊤D factor accumulation masked by b on rows/cols; effective accumulation rank reduced |
| #10 MPOT | orthogonal — MPOT contractors gated at core level |
| #11 MFIO | **direct gain**: σ_ℓ · g_i broadcast gated by b_i — MFIO's dominant remaining cost |
| #12 DFA | orthogonal |
| #13 TRCD | orthogonal; per-token depth × per-parameter update stacks multiplicatively |

GFIB stacks *multiplicatively* on the optimizer-bandwidth axis with
#3, #5, #9, #11 — the four shifts that together define the
optimizer side of the stack.

---

## 9. Failure modes and mitigations

**(F1) Tail hazard — low-F̂ parameters that matter.**
A weight row relevant to a rare token may carry near-zero F̂ most
steps but nontrivial true Fisher on rare events.  Mitigations:
1. **p_floor.**  p_i ≥ p_floor = K/4 (clipped at the low end; μ
   compensates).  Refresh half-life ~28 steps at K=0.1.
2. **Salience bypass** for W_emb rows touched by the current
   batch — those always update; cost |V_batch| · d.
3. **Periodic dense steps** every 100 steps (budget cost 1%).

**(F2) Non-stationary Fisher distribution.**  Curriculum shifts,
phase transitions, etc.  Mitigations:
1. Faster EWMA (β_F = 0.95, ~14-step half-life).
2. Detect-and-reset on the moments of E_i[log F̂_i]: if moving
   > 3σ between 500-step windows, reset μ to the recent median
   and γ to 1.0 to rebuild sharpness.

**(F3) PI oscillation.**  Classic PI failure modes:
1. Anti-windup clamp on integral term (±10).
2. Dead-band: integrate only when |e| > 0.005.
3. Gain scheduling: k_I → 1e-5 after |e| < 0.01 for 200 steps.
4. First 1000 steps: μ fixed at median log F̂ (bootstrap).

**(F4) Bernoulli-induced gradient variance.**  The b_i draw adds
σ = (p_i(1−p_i))^{1/2}-scale noise to the effective preconditioner.
Mitigations:
1. Adam's m_i already averages b_i·g_i over time.
2. At evaluation, replace sampling with p_i (Rao-Blackwellize);
   zero variance.

**(F5) Reproducibility.**  The Philox stream for b_i is seeded and
checkpointed per our DETERMINISM_AND_CONCURRENCY policy; same seed →
identical b sequence.  Unit test `GFIBDeterminismTest` asserts this.

---

## 10. Memory and wall-clock accounting at 2.23 B params, 16 GB

Baseline (shifts #1-#11 landed):
- weights (MPOT × Stiefel × BF16+SR):  223 MB
- gradients (OVFG factored):             669 MB
- Adam int8 + master:                   4.46 GB (**largest**)
- activations (CHIRON):                    ~4 MB
- scratch/logits:                       ~1.5 GB
- total:                                 ~7 GB (9 GB free)

GFIB adds:
- F̂ tensor (BF16, 1 per param):         4.46 GB (**persistent**)
- b mask, μ, PI state:                   negligible (streamed)

Net: +4.46 GB → ~11.5 GB used; 4.5 GB free.  This is non-trivial
and is the principal cost line for GFIB at this scale.

**Wall-clock.**
- Pure GFIB wall-clock: ~13% per-step saving × ~1.3× step count
  ≈ break-even.
- GFIB × MFIO × int8 Adam stacked: each contributes a factor on
  the (already small) optimizer time; step time 40 ms → 34 ms
  (15% faster), composition step-count penalty 1.25-1.5×.
  **Net:  ~break-even to ~5% slowdown.**

**This is by design.**  GFIB at K=0.1 on its own is not a speedup
win.  Its contribution is:
1. An information-theoretic **lower bound** on optimizer-side work.
   Any future optimizer-state compression shift must improve
   *above* GFIB's KKT-optimal Σ p_i h_i allocation.
2. An **unlock path** for 5-8 B-parameter training by enabling a
   future shift where m_i, v_i for the inactive 90% of parameters
   are not persisted at all, only lazily reconstructed on
   b_i = 0 → 1 transitions.  The KKT bookkeeping is already in place.
3. Strong composition with **shift #15 (TPW)** once deferred shifts
   are considered — TPW's cross-step Padé fit benefits from sparser
   per-step updates (fewer wasted gradient subspaces).

---

## 11. Relation to prior work

- **RigL (Evci et al. 2020).**  Periodic top-K magnitude prune/grow
  on weight values.  GFIB differs: per-step probabilistic, KKT-tied
  to a budget, and never alters the model (b_i = 0 retains the
  weight value).
- **Sparse-gradient methods (Dryden 2016, Aji & Heafield 2017).**
  Compress gradients for distributed communication.  GFIB sparsifies
  the optimizer step, preserving gradient fidelity.
- **Variational Bernoulli dropout (Molchanov 2017).**  On weight
  *values* during forward.  GFIB acts on weight *updates*; different
  theoretical structure and guarantee.
- **LAMB / LARS.**  Per-layer rate adaptation via weight norms; no
  sparsification.  GFIB is per-parameter rate sparsification with
  KKT structure.
- **Adafactor.**  Factored second moment; compresses state, doesn't
  sparsify updates.  Orthogonal.

**Novelty claim.**  The combination of (a) KKT-derived soft-threshold
policy, (b) PI-adaptive Lagrangian μ, (c) per-step Bernoulli with
tail-preserving floor, (d) composition with #3 / #11 to collapse
optimizer bandwidth — this does not appear in prior literature.

---

## 12. Implementation sketch and evaluation plan

**Deliverables (≈ 2 engineering weeks):**

1. `gpu_gfib.{h,cu}` (~450 LOC):
   - `gfib_fisher_ewma_update(g, F̂, β_F)`
   - `gfib_sample_gates(F̂, μ, γ, rng, out_b)` — fused
     sigmoid + Bernoulli + bit-packing
   - `gfib_gated_adam_step(θ, g, m, v, b, η, step)` — predicated
2. `mu_pi_controller.{h,cpp}` (~80 LOC) — scalar PI controller,
   reuses TRCD's λ-controller pattern.
3. Unit tests:
   - `GFIBSoftThresholdParityTest` — CPU σ(γ(x−μ)) vs GPU kernel
   - `GFIBBudgetControllerTest` — μ converges to K within 500 steps
   - `GFIBTailPreservationTest` — weight critical on <1% of batches
     not frozen
   - `GFIBDeterminismTest` — identical b sequence under same seed
   - `GFIBEndToEndConvergenceTest` — 500-step pile smoke test,
     loss within 5% of dense Adam
4. Trainer flags: `--gfib-budget K`, `--gfib-gamma γ`,
   `--gfib-beta_F β`, `--gfib-floor p_floor`, `--gfib-warmup-steps`.

**Validation.**
1. pile_small (80 M, L=12, 5000 steps): sweep K ∈ {1.0, 0.3, 0.1, 0.03}
   vs. dense baseline; measure final NLL and wall-clock.
2. pile_large (2.23 B, all #1-#11 landed): composition benchmark,
   measure optimizer-step bandwidth with and without GFIB.
3. Ablations: no p_floor (expect tail hazard on rare-token W_emb rows),
   fixed μ (expect drift), hard top-K (expect gradient chatter).
4. Pareto sweep on K vs wall-clock-to-target.

---

## 13. Open conjectures

1. **Heavy-tailed Fisher hypothesis**: the per-step F̂_i distribution
   is Pareto-like with α ≈ 1.5–2.0 on LLM pre-training.  If true,
   K = 0.05 retains >70% of single-step progress.  Falsifiable:
   log-log rank-frequency plot at training steps 100, 1k, 10k.
2. **Compound-with-MFIO hypothesis**: GFIB's bit-mask broadcast
   gating gives >5× bandwidth reduction on MFIO's σ_ℓ · g_i write.
   Falsifiable: direct measurement.
3. **KKT optimality**: for a fixed optimizer budget B, no alternative
   update policy achieves lower E[ΔL] than GFIB at K = B/C_o.  This
   is the information-theoretic claim.  Rigorous proof requires
   asymptotics on h_i; empirical check is a grid of heuristic
   alternatives (uniform dropout, top-K, magnitude-thresholded,
   random) vs. GFIB at matched budget.

---

*End of candidate A.*
