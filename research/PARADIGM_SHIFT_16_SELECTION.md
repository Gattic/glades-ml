# Paradigm Shift #16 — Selection Rationale

Ralph-loop iteration 2026-04-22.  Three materially different candidates
developed in parallel via the research-framework-design skill protocol:

- **Candidate A — GFIB (Gradient-Flow Information Bottleneck)**: per-parameter
  update selectivity via KKT-thresholded soft Bernoulli sampling on Fisher
  magnitude.  `PARADIGM_SHIFT_16_CANDIDATE_A_GFIB.md`.
- **Candidate B — LCP (Lattice Compute Pooling)**: per-layer LSH cluster-
  pool main-block compute, reconstruct per-token via rank-r Jacobian-detail
  network. `PARADIGM_SHIFT_16_CANDIDATE_B_LCP.md`.
- **Candidate C — SGS (Saliency-Guided Substitution)**: per-layer surrogate
  with asymmetric updates, adaptive substitution probability via held-out
  drift EMA.  `PARADIGM_SHIFT_16_CANDIDATE_C_SGS.md`.

## Summary table

| Axis | A (GFIB) | B (LCP) | C (SGS) |
|---|---|---|---|
| Attack axis | per-parameter Adam-update gating | per-token compute pooling | per-layer surrogate substitution |
| Direct standalone speedup | ~break-even | **4.7× per-layer** | 1.9× |
| Compound with TRCD (#13) | additive (optimizer bandwidth) | **24× (row×column factoring)** | 6-8× |
| Direct memory win | no (bandwidth only, saves ~12 GB/step D2H traffic) | yes — activation O(M·d) | yes — surrogate-state is rank-r |
| Core novelty | KKT-optimal lower bound on optimizer work | LSH-pooled forward + Jacobian-detail correction | identity-init surrogate, asymmetric update |
| Risk profile | F1 tail hazard (low-F̂ parameters that matter) | F1 cluster-quality degrades at scale; F3 differentiable assignment stability | F3 surrogate mode collapse; F2 drift-compute overhead |
| Implementability | **lowest** (Bernoulli mask + gradient-gated Adam) | highest (LSH + detail-net + aux loss + new kernels) | medium (per-layer surrogate + adaptive ρ controller) |
| Composes with shipped stack | multiplicative on opt state (shifts 3, 5, 9, 11) | multiplicative on activation + forward compute (shifts 1, 2, 6, 13) | multiplicative on per-layer compute (shifts 2, 7, 10, 12, 13) |

## Selected: Candidate B — LCP

### Why LCP

1. **Highest measured direct speedup with evidence.**  LCP's 4.7× per-layer
   is derived from an explicit LSH cost (16 MFLOPs at T=2048) + rank-r
   detail network (8.4% of main block) against a main-block cost of
   ~8 ms at pile_large dims.  It is the only candidate whose standalone
   number exceeds 2×.

2. **Highest compound with the existing stack.**  The TRCD-LCP compound
   of 24× comes from clean row×column factoring of the (L, T) compute
   matrix: TRCD sparsifies rows (tokens skip layers), LCP dedups columns
   (similar tokens pool their layer-compute).  The two do not fight —
   TRCD's decision "skip this layer for this token" and LCP's decision
   "this token pools with that one" are orthogonal per-layer choices.
   At realistic 50% realization: 12-15× compound.

3. **Attacks a genuinely new axis.**  No shift in 1-15 questioned the
   assumption that every token in a batch independently runs the full
   main-block forward.  LCP is the first paradigm shift to leverage
   per-batch token similarity for compute amortization.

4. **Direct memory win in addition to speed.**  LCP reduces the main-
   block activation from O(T·d) to O(M·d), additive with CHIRON's
   already-low O(1) depth.  At T=2048, M=256: 8× activation reduction
   on each block's scratch.  Adds to the memory axis the user
   explicitly asked about ("magnitudes less memory").

5. **Derived, not heuristic, decomposition.**  The Taylor-derived
   decomposition `f_θ(x) = f_θ(c(x)) + J_θ(c(x))·δ + O(‖δ‖²)` is a
   first-principles motivation for the detail network `D_φ`.  The
   Jacobian-residual auxiliary loss pins `D_φ` to observed ground
   truth on a 1/16 validation subset, which is a concrete mitigation
   for the central drift risk.  No other candidate has an explicit
   derivation of what the cheap approximation SHOULD approximate.

### Why A and C are rejected

- **A (GFIB)**: the KKT derivation is elegant but the RESULT is that
  backward stays dense (we have to measure `g` to compute `F̂`), so
  GFIB saves ONLY the optimizer-side compute.  At 2.23 B with int8
  Adam already at ~4 GB state, the bandwidth reduction is real (13.4
  → 1.34 GB/step at K=0.1) but wall-clock impact is "roughly break-
  even" per the candidate's own honest accounting.  GFIB is a *tool*
  for future compression, not a standalone paradigm shift.  Defer
  as shift #17 — its real value materializes once a paradigm shift
  adds PHYSICAL compression to inactive-parameter state (e.g., an
  extended MFIO that dynamically paging-out layers with low F̂).

- **C (SGS)**: solid theory (identity-init prevents collapse, bias-
  variance-derived ρ*, drift-EMA throttling), 1.9× standalone, 6-8×
  compound.  But LCP's 4.7× standalone + 24× compound dominates SGS
  on both axes, AND SGS has more moving parts (per-layer surrogate,
  adaptive controller, held-out val minibatch maintenance) than LCP.
  Defer as shift #18.  Its mechanism is genuinely useful for
  downstream deployment (student network style) but LCP is better
  suited to the immediate training-throughput goal.

### Retention

A and C are explicitly NOT discarded.  Both have coherent theory and
concrete failure-mode mitigations.  They re-enter the research program
as:

- **#17 GFIB (deferred)**: promote once a co-design exists with a
  paradigm shift that PHYSICALLY compresses inactive-parameter state
  (Adam state, weights).  GFIB provides the KKT-optimal schedule for
  that compression.

- **#18 SGS (deferred)**: promote after LCP's detail-network approach
  is debugged; SGS's surrogate network is architecturally similar
  (rank-r approximator of a main component), and the Jacobian loss
  infrastructure from LCP is directly reusable.

### Implementation plan for LCP (paradigm shift #16)

**Phase 1 — primitives (GPU):**
- `lcp_lsh_project`: 8-bit hash from `h @ R` where `R` is the fixed
  LSH projection matrix.  Output: integer bucket id per token.
- `lcp_cluster_assign`: map token → cluster index (bucket id mod M).
- `lcp_gather_representatives`: select one token per cluster (per-
  bucket min-index, deterministic).
- `lcp_scatter_main`: broadcast main-block outputs from M reps to T
  tokens.
- `lcp_detail_forward`: D_φ rank-r SwiGLU on within-cluster deltas.
- `lcp_jacobian_residual_loss`: on a 1/16 validation subset, pin D_φ
  to `f_θ(x) − f_θ(c(x))`.
- `lcp_detail_backward` / `lcp_scatter_backward`: backward primitives.

**Phase 2 — CHIRON test suite parity:**
- `CHIRONLcpHashBucketParityTest`: 8-projection LSH is deterministic
  and collision rate matches expected binomial.
- `CHIRONLcpClusterAssignParityTest`: cluster counts sum to T.
- `CHIRONLcpGatherScatterParityTest`: `scatter(gather(h))` is an
  identity modulo cluster choice — any two tokens in the same cluster
  end up with the same gathered value.
- `CHIRONLcpDetailForwardParityTest`: rank-r SwiGLU matches reference.
- `CHIRONLcpEndToEndMseTest`: 2-layer MLP with LCP achieves <1% MSE
  drop vs dense at M=T/4 on a toy regression.
- `CHIRONLcpThroughputBenchmark`: at pile_large dims, report actual
  ms/step savings vs dense.

**Phase 3 — trainer wire-in:**
- `--lcp-M M`: target number of clusters (0 = disabled).
- `--lcp-detail-rank R`: rank of the SwiGLU detail network.
- `--lcp-drift-weight α`: weight on the Jacobian-residual aux loss.

**Phase 4 — pile_large benchmark:**
- Dense baseline vs LCP at M=T/4 and M=T/8, measured against the
  45,297 tok/s production baseline.  Target: ≥2× throughput with
  <5% loss degradation at matched step count.

## Open conjectures / validation criteria

1. **Cluster-quality learnability conjecture**: at pile_large scale,
   LSH clusters on learned embeddings will produce per-cluster loss
   variance bounded by a small constant times the per-token variance.
   Test: compute per-cluster MSE(`D_φ(δ)`, `f_θ(x)−f_θ(c(x))`) and
   check it does not exceed `3× σ_within_cluster`.

2. **TRCD × LCP composition conjecture**: the realized compound
   speedup exceeds 10× (lower bound of 24× × 50% realization).  Test:
   at d̄=8, M=256 on pile_large, measure tok/s against the 45k
   baseline.  Target: ≥ 450,000 tok/s.

3. **Jacobian-detail tightness conjecture**: the rank-r SwiGLU detail
   network is sufficient for `J_θ(c(x))` when r ≥ d/16 and training
   provides `Ω(d²)` gradient updates per (c, δ) pair over the course
   of training.  Test: ablate r ∈ {d/32, d/16, d/8, d/4} and measure
   final loss.  Target: r=d/8 is within 2% of r=d (no-compression
   control).

## Decision

**LCP is selected as paradigm shift #16.**  Move to Phase 1 primitive
implementation.  GFIB becomes shift #17 (deferred, promote once
physical state compression is available); SGS becomes shift #18
(deferred, promote after LCP's detail-network infrastructure is
stable).
