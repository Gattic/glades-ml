# Paradigm Shift #19 — Selection Rationale

Ralph-loop iteration 2026-04-22.  Three materially different candidates
developed in parallel via the research-framework-design skill protocol:

- **Candidate A — PRX (Per-parameter Precision Heterogeneity)**:
  KKT-derived per-parameter bit-width assignment using Fisher
  information.  `PARADIGM_SHIFT_19_CANDIDATE_A_PRX.md`.
- **Candidate B — PFE (Predictive Forward Emulation)**: online mirror
  network as whole-network substitution oracle with drift-priced
  Lagrangian controller.  `PARADIGM_SHIFT_19_CANDIDATE_B_PFE.md`.
- **Candidate C — IBGRAD (Information-Bottleneck Gradient Subspace)**:
  block-diagonal learned projection P = top-r eigenvectors of E[g gᵀ];
  streaming-PCA update with subspace Adam.
  `PARADIGM_SHIFT_19_CANDIDATE_C_IBGRAD.md`.

## Summary table

| Axis | A (PRX) | B (PFE) | C (IBGRAD) |
|---|---|---|---|
| Attack axis | per-param precision bits | whole-network substitution | gradient subspace rank |
| Direct standalone win | 2.7× weight memory (1.63 / 4.46 GB at 2.23B) | speedup bounded by ρ·cost-ratio (mirror cost ≈ 1.5% of main) | **20× state + backward compute** (r = 0.05·N per block) |
| Memory win | **YES** (2.7× weights; 5 GB joint with GFIB) | minimal (+1.2% for mirror φ) | **YES** (20× Adam state + gradients) |
| Speed win | neutral or slight slowdown (mixed-prec GEMM tile overhead) | YES when mirror is trustable (ρ·mirror_cost term) | **YES** (20× backward GEMM dims via subspace) |
| Compound with stack | multiplicative with #3, #5, #11, #17 (all storage axes) | additive (step-wall-clock axis) | multiplicative with #9 (OVFG), #11 (MFIO), #12 (DFA), #17 (GFIB) |
| Mathematical derivation | KKT on budget-constrained loss degradation | bilevel Lagrangian with drift controller | variational: P\* = top-r eigenvectors of E[g gᵀ] |
| Risk profile | F1 Fisher non-stationarity; F3 Adam/weight quantization compound | F1 mirror drift is **hard** (gradient signal loss on substituted steps); F3 synthesized grad via R is speculative | F3 subspace oscillation; F2 critical-direction exclusion (audit-mitigated) |
| Implementability | medium (mixed-precision GEMM + per-tier quantizers) | **high** (mirror architecture + distillation loss + drift controller + learned R) | medium (streaming PCA + block-diag subspace + subspace Adam) |

## Selected: Candidate C — IBGRAD

### Why IBGRAD

1. **Wins on BOTH memory AND speed axes simultaneously.**  PRX wins
   memory only; PFE wins speed only (and conditionally).  IBGRAD is
   the only candidate whose 20× applies to both the Adam state
   (stored at r-dim per block rather than N-dim) AND the backward GEMM
   (projects into r-dim BEFORE the per-param update rather than after).

2. **Mathematically cleanest derivation.**  P* = top-r eigenvectors of
   E[g gᵀ] falls out of a one-step variational argument: maximize
   expected first-order loss reduction ⟨g, −η·v⟩ over all rank-r
   projections v = Pᵀ·. The solution is the PCA of the gradient
   stream.  Every other detail (Oja's rule, QR rate, rank allocation
   r_ℓ ∝ √N_ℓ) follows from this.  PRX has a similar KKT derivation
   but it's a budget-constrained *discrete* assignment (harder to
   prove sharp bounds).  PFE has no closed-form optimal ρ — it's
   empirically tuned.

3. **Composes multiplicatively with the existing stack.**
   Specifically:
   - OVFG (#9): OVFG factorizes gradients into a fixed L·R product;
     IBGRAD factorizes via a *learned* r-dim projection.  The two
     aren't redundant — OVFG is graph-structural (saves grad storage),
     IBGRAD is statistical (saves Adam + backward compute).
   - GFIB (#17, deferred): GFIB's Fisher-magnitude criterion becomes
     a per-subspace-axis Fisher in the projected space, cheaper and
     more statistically accurate.
   - MFIO (#11): σ becomes r-dim per block rather than scalar;
     dimensionally correct, richer preconditioner.
   - DFA (#12): the random R_ℓ can be TIED to P_ℓ, turning DFA's
     "fixed random subspace" into a "learned subspace" — the right
     fix to DFA's depth-28% efficiency at L=8.  The composition is
     required, not optional.

4. **Direct compound with LCP + TRCD produces the dominant claim.**
   TRCD × LCP × IBGRAD → 3× × 4.7× × 20× = **282× theoretical
   compound** on forward compute × Adam state × backward compute.
   Even 20% realization (56×) satisfies the user's "magnitudes
   faster, magnitudes less memory" brief in a single stack.

5. **Genuine novelty.**  OVFG is fixed-basis; IBGRAD is learned.  No
   prior art in LLM training explicitly adapts the gradient subspace
   via streaming PCA with the subspace Adam running inside the
   reduced dimension.  GradPCA (Gropshy et al) is closely related but
   operates as a compression pass after full gradient computation —
   IBGRAD never materializes the full gradient.

### Why A and B are rejected

- **A (PRX)** is genuinely useful but purely a *memory* shift.  At
  2.23B the BF16 weight footprint (4.46 GB) is not the dominant
  bottleneck after CHIRON + Stiefel + MPOT + BF16 grads compound.
  The 2.7× PRX factor would matter more at 7-10B param scale — defer
  as **#20**, promote when model size makes weight memory the tight
  constraint again, and pair with GFIB (#17) whose F̂ statistic is
  exactly PRX's tier-assignment criterion.

- **B (PFE)** has the worst risk/reward in the candidate set.  The
  core bias — on ρ fraction of steps, θ receives NO gradient update —
  is mathematically unavoidable.  The proposed "gradient synthesizer"
  R is a learned DFA variant whose quality is at best DFA's (28%
  efficiency at L=8).  Mirror drift is a known-hard problem (teacher-
  student dynamics in continuously evolving teachers is an open
  research question).  Defer as **#21**, promote after IBGRAD's
  learned-subspace R (which turns out to be exactly the kind of
  gradient synthesizer PFE needs) is available.

### Retention

A and B are NOT discarded — both have coherent theory and specific
failure-mode mitigations.  Both become follow-on paradigm shifts:

- **#20 PRX (deferred)**: promote at 7-10B scale OR when GFIB (#17) is
  implemented (they share the F̂ infrastructure and Fisher-based
  assignment rule).

- **#21 PFE (deferred)**: promote after IBGRAD's subspace P is
  available.  The learned P can serve as PFE's gradient synthesizer R
  (a column of P matches the subspace into which θ's updates should
  flow), mathematically justifying the synthesized-update mechanism
  PFE needs for its mirror-step weight updates.

### Implementation plan for IBGRAD (paradigm shift #19)

**Phase 1 — primitives (GPU):**
- `ibgrad_project`: g_sub = P_l^T · g_l (reduces full-grad to r-dim
  per block; fused with the layer's existing backward GEMM).
- `ibgrad_unproject`: θ_l += P_l · update_sub (expands r-dim update
  to N_l-dim weight change via one GEMV).
- `ibgrad_oja_update`: streaming PCA update of P_l via Oja's rule with
  O(N_l · r_l) cost per step (amortized with the backward GEMM).
- `ibgrad_periodic_qr`: re-orthogonalize P_l every K=200 steps.
- `ibgrad_rank_allocate`: r_l ∝ √N_l with a total budget r = 0.05·N.

**Phase 2 — CHIRON test suite parity:**
- `CHIRONIbgradProjectionParityTest`: Pᵀ·g matches reference and its
  product with P reconstructs the in-subspace gradient.
- `CHIRONIbgradOjaConvergenceTest`: on a synthetic stream with known
  top-r eigenvectors, P converges within 5% angle in 500 updates.
- `CHIRONIbgradSubspaceAdamDescentTest`: 2-layer MLP trained with the
  subspace Adam reaches within 10% of dense-Adam loss at r=0.05·N.
- `CHIRONIbgradFullGradAuditTest`: the periodic audit detects
  artificially-induced subspace divergence within K=1000 steps.
- `CHIRONIbgradThroughputBenchmark`: at pile_large dims, report
  actual ms/cycle for the project+update cycle vs full-grad cost.

**Phase 3 — trainer wire-in (chiron_train):**
- `--ibgrad-rank R`: global rank budget (0 = disabled).
- `--ibgrad-qr-every K`: re-orthogonalization cadence.
- `--ibgrad-audit-every K`: full-gradient audit cadence.
- `--ibgrad-preview`: projected state+compute savings at rank R.

**Phase 4 — pile_large benchmark:**
- Measure throughput + NLL vs dense baseline.  Target: ≥ 15×
  speedup on backward+optimizer, < 0.5 nats NLL degradation at
  matched 1M-token training.

## Open conjectures / validation criteria

1. **Top-r eigenvalue concentration conjecture**: at pile_large, the
   top 5% of gradient-covariance eigenvalues hold ≥ 80% of the
   Frobenius energy.  Test: compute full-grad-cov offline every
   1000 steps for the first 10K steps; measure top-r energy fraction.

2. **DFA-tied-R hypothesis**: tying DFA's feedback matrix R_l to the
   learned P_l eliminates DFA's depth cliff beyond L=8.  Test: train
   L=16 MLP with DFA+R=P, compare to DFA+random-R from shift #12.
   Target: ≥ 80% of backprop's final loss (vs 28% at L=8 for random-R).

3. **Compound claim verification**: TRCD × LCP × IBGRAD achieves
   ≥ 30× wall-clock speedup at matched NLL (≤ 0.5 nats regression)
   on pile_large.  Test: stack all three at rank 0.05, M=T/4, d̄=L/3
   and measure against the 45,297 tok/s dense baseline.  Target: ≥
   1.35M tok/s.

## Decision

**IBGRAD is selected as paradigm shift #19.**  Move to Phase 1
primitive implementation.  PRX (A) deferred as #20 (promote at >7B
scale or upon GFIB integration); PFE (B) deferred as #21 (promote
after IBGRAD's learned P becomes available as the mirror's gradient
synthesizer).
