# Paradigm shift #9 — candidate comparison and selection

**Date**: 2026-04-22.
**Context**: after shifts #1–#8, the 2.23 B-parameter ceiling on a
16 GB RTX 4080 SUPER breaks down into three roughly-equal memory
categories: weights (BF16, ~4.5 GB), gradients (BF16, ~4.5 GB), and
optimizer moments (int8 m + u8 v, ~4.5 GB).  Shift #7 (Stiefel × Σ)
attacks *weights*.  Shift #9 targets the remaining ~9 GB of
{gradients, optimizer state}.

Per the research-framework-design skill protocol, three materially
different candidate formulations were dispatched as parallel subagents.
Their full candidate documents are saved as:

- **Candidate A**: `PARADIGM_SHIFT_9_CANDIDATE_A_MCTB.md`
  *Manifold-Coherent Tangent Bundle* — Riemannian/geometric,
  shared low-dim tangent bundle over Stiefel-coherent M_coh.
- **Candidate B**: `PARADIGM_SHIFT_9_CANDIDATE_B_SIU_FPF.md`
  *Stochastic-Implicit Update via Fokker-Planck Flow* — JKO-proximal
  Langevin dynamics with activation-derived preconditioner; eliminates
  m, v entirely.
- **Candidate C**: `PARADIGM_SHIFT_9_CANDIDATE_C_OVFG.md`
  *Operator-Valued Factored Gradient* — exploits
  G = A^⊤D ⇒ rank(G) ≤ T; stores gradient + moments in factored form.

## Comparison matrix

| axis                          | A / MCTB                     | B / SIU-FPF                  | C / OVFG                        |
|-------------------------------|------------------------------|------------------------------|---------------------------------|
| **formulation class**         | Riemannian manifold          | stochastic dynamics (SDE)    | operator-theoretic repr.        |
| **originality**               | medium (GaLore-family)       | **high** (zero m, v)         | medium (factor push-through)    |
| **memory compression target** | 5.3× (→ 9× w/ int8 frame)    | **≫10×** (m, v → 0)          | 6× alone, **17× w/ Stiefel**    |
| **composability with #7**     | strict super-structure       | manifold Langevin (natural)  | **multiplicative** (dU,dV,dΣ)   |
| **composability with CHIRON** | good (per-block contract)    | **synergistic** (CV + cache) | good (A, D already live)        |
| **composability with HRTC**   | orthogonal                   | orthogonal (Haar preserves ξ)| **direct 4× on accumulator**    |
| **implementation risk**       | medium (frame mgmt)          | **high** (Langevin at LLM scale unproven) | **low** (representational) |
| **engineering scope ≤2w**     | ~1500 LOC + tests            | ~200 LOC + careful tuning    | **~900 LOC + parity tests**     |
| **convergence risk**          | medium (rank collapse)       | **high** (no explicit m)     | **low** (projected Adam)        |
| **empirical validation path** | parity vs Stiefel+Adam       | full convergence run needed  | **rank-parity + Adam parity**   |

## Dominant tradeoffs

1. **Upside ceiling**.  B is the only candidate that could plausibly
   eliminate both m and v entirely (>>10× compression).  If it works
   at 2.23 B it leapfrogs both A and C.  But it depends on an unproven
   claim — that the implicit-midpoint + activation-norm preconditioner
   recovers Adam's behavior on LLM loss landscapes at >1 B scale.

2. **Implementation risk vs iteration cadence**.  The Ralph-loop
   cadence rewards shifts that can be validated within one or two
   iterations.  C (OVFG) has parity tests as intermediate milestones,
   so correctness can be verified step-by-step.  A and B require
   end-to-end convergence runs (expensive, noisy).

3. **Composition with the existing stack**.  All three compose with
   shifts #1–#8 in some way, but C is the only one that composes
   **multiplicatively** with shift #7 Stiefel: OVFG's r× compression on
   grads+moments stacks on top of Stiefel's ρ× compression on weights,
   projected to reach 17× combined on {weights, grads, moments}.  A
   *replaces* part of shift #7 (overlaps with the manifold structure).
   B *subsumes* shifts #3, #4 (makes them moot), which is elegant but
   riskier — it removes working infrastructure.

4. **Disruption vs continuation**.  The prior Ralph-loop arc has been
   representation changes: BF16 weights, int8 Adam, Stiefel
   factorization, HRTC compression.  C continues this arc; A pivots to
   geometric framing; B pivots to dynamics.  The "disruption" in C is
   in the compound effect, not the representation itself.

## Selection: **Candidate C (OVFG)**

### Why C is the strongest choice for shift #9

1. **Cleanest composition with in-flight shift #7.**  OVFG's
   gradient factor pair (A, D) feeds directly into Stiefel's tangent
   gradient formulas dU, dV, dΣ via two r×ρ GEMMs — no dense m×n
   ever materializes.  Memory compression multiplies.

2. **Lowest convergence risk.**  OVFG is a rank-truncated Adam with
   Adafactor row/col 2nd moment.  Both GaLore (rank-r grad projection)
   and Adafactor (rank-1 v) have published evidence at 1 B+ scale.
   The novel piece is stacking them with factored storage + Stiefel —
   not a new training dynamic.

3. **Incremental validation.**  Parity tests are cheap: verify
   ovfg_adam at r=min(m,n) matches dense Adam (should be bit-exact),
   then at r=256 measure the energy captured.  Each test is seconds,
   not hours.

4. **Highest ROI per engineering hour.**  ~900 LOC of kernel + glue,
   clear milestones.  A requires frame management, parallel transport,
   and coherence constraints — much more moving parts.  B requires
   tuning an SDE schedule on a 2 B model, which is weeks of
   experimentation.

5. **Failure containment.**  If OVFG shows truncation error, we fall
   back to dense Adam for affected layers — graceful degradation.  If
   SIU-FPF fails to converge, there's no fallback without undoing the
   entire optimizer rewrite.

### Why the other candidates were rejected

- **A (MCTB) rejected**: overlaps with shift #7 (Stiefel already places
  weights on a manifold).  The "moving frame over M_coh" adds
  machinery without clear compounding advantage over OVFG's simpler
  factor-pair view.  Strong candidate for a **future shift #10** once
  OVFG establishes the infrastructure; the global tangent bundle is a
  natural generalization.

- **B (SIU-FPF) rejected for now**: scientifically the most ambitious,
  but the activation-norm preconditioner surrogate (ŝ_ℓ = E[‖z‖²·‖δ‖²])
  is a heuristic, and Langevin training at 2 B scale is empirically
  unvalidated.  Adopting it would require a multi-week empirical
  program before trusting it as the production path.  Keep as a
  **reserve candidate for shift #11** if OVFG composes cleanly with
  #7 and #8 (so we have VRAM headroom to experiment).

## Implementation plan — shift #9 = OVFG

Following candidate C's minimal-prototype protocol:

**Phase 1 — primitives (week 1)**
- `gpu_ovfg.{h,cu}` — new CUDA module with:
  - `ovfg_store_factors`  (replaces `sgemm_rowmajor_atb` in backward)
  - `ovfg_append_factored` (scaled-append to L, R)
  - `ovfg_rsvd_truncate` (randomized QR via cuSOLVER)
  - `ovfg_adafactor_moments` (row/col sums via L(R^⊤R)L^⊤)
  - `ovfg_apply_update_dense` (fallback dense-W path)
- Unit tests in `unit-tests/Backend/Machine Learning/chiron-test.cpp`:
  - rank parity at r=min(m,n)
  - Adam parity (vs dense) at r=min(m,n)
  - Adafactor 2nd-moment parity
  - RSVD stability at BF16 factors + FP32 sketch

**Phase 2 — Stiefel composition (week 2)**
- `ovfg_stiefel_tangent_grad` — compute dU, dΣ, dV directly from
  (L, R, U, Σ, V) factors; never materialize dG
- Unit test: parity with dense dU/dΣ/dV on small-rank case
- Trainer wire-in: `chiron_main.cpp` `--ovfg-rank R` flag

**Phase 3 — HRTC composition (optional, week 3)**
- Apply HRTC Haar transform to A and D factors
- Memory benchmarking composed vs standalone

**Phase 4 — end-to-end validation**
- pile_large-small config (d=1024, L=24) 1000-step convergence run
- Target: loss trajectory within 5% of BF16-Adam baseline
- VRAM budget measurement at r=128, 256, 512

## Task registration

Create **task #33: "Paradigm shift #9: OVFG — Operator-Valued Factored Gradient (Phase 1)"**
to track the above plan; blocked by task #30 completing its Stiefel
compute wire-in (because the multiplicative win is realized only in
composition).  Once #30 lands the Stiefel forward/backward on pile_large,
#33 can proceed in parallel.
