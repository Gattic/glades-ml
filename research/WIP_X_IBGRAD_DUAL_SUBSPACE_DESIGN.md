# WIP × IBGRAD — Dual-Subspace Optimization Design

**Status:** design complete; implementation handoff ready.
**Date:** 2026-04-23 (Ralph-loop iteration 36).
**Priority:** #1 from `DEFERRED_SHIFTS_RESCORE_2026-04-23.md`.

---

## 1. Summary

Combine **WIP** (Weight Interpolation Pretraining, paradigm #22) with
**IBGRAD** (Information-Bottleneck Gradient Subspace, paradigm #19) into
a single dual-subspace optimization:

- **WIP constrains parameter space**: θ = Σ softmax(α)_k · W_k, where
  α ∈ ℝ^K is the K-dim optimization variable and {W_k} is a pool of
  frozen weight snapshots.
- **IBGRAD constrains gradient space**: gradients are computed/stored
  in the r-dim subspace P ⊂ ℝ^N via Oja streaming PCA.

Effective optimization DOF = min(K, r).  With K=16, r=32: **DOF = 16**
(WIP dominates).

## 2. Why this compound is attractive

Every deferred shift was re-scored in `DEFERRED_SHIFTS_RESCORE_2026-04-23.md`.
WIP × IBGRAD came out top because:

1. **Zero additional primitives needed** — the compound is a trivial
   composition of shipped IBGRAD primitives plus a small Adam loop on α.
2. **Dual compression**: 16-dim α optimization × 32-dim IBGRAD subspace
   = reduces the effective Adam state to 16 floats per weight matrix
   (vs 2·N floats for dense Adam).
3. **Memory: K·N snapshot storage is shared across training** — each
   W_k is frozen once recorded.  Compose with MPOT (#10) to compress
   each snapshot: K=16 × MPOT-25× = 0.64× of dense weight memory.

## 3. Mathematical core

**Parameter space:** θ ∈ ℝ^N lives in the convex hull of K snapshots.

    θ(α) = Σ_k σ(α)_k W_k     where σ = softmax  (α ∈ ℝ^K unconstrained)

**Gradient w.r.t. α** (chain rule through softmax):

    ∂L/∂α_k = σ(α)_k · ⟨W_k - θ(α), ∇_θ L⟩
           = σ(α)_k · (⟨W_k, ∇_θ L⟩ - ⟨θ(α), ∇_θ L⟩)

Second term is a scalar shared across all k.  First term is K inner
products of N-dim vectors.

**Key efficiency trick**: with IBGRAD active, we have the projected
gradient `y = Pᵀ g ∈ ℝ^r` cheaply.  The inner products can be
computed via the PROJECTED subspace:

    ⟨W_k, g⟩ ≈ ⟨Pᵀ W_k, Pᵀ g⟩ = ⟨z_k, y⟩          where z_k = Pᵀ W_k

- `z_k = Pᵀ · W_k` is an r-dim vector per snapshot.
- Computed ONCE when W_k is recorded (not per-step).
- Stored as K·r floats total (at K=16, r=32: 512 floats = 2 KB).
- Updated when P rotates significantly (synchronize with audit cadence).

**Per-step cost** (after initial `z_k` cache):
1. Compute y = Pᵀ g via ibgrad_project.
2. g_α_k = σ(α)_k · (⟨z_k, y⟩ - θα_dot)  — K·r flops ≈ 512 ops per step.
3. Adam on α (K=16 state).
4. Recompute θ(α) = Σ σ(α)_k W_k — this is the expensive step.

## 4. The θ materialization cost

Cost of θ(α) = Σ σ(α)_k W_k is K matrix sums (K=16, N=1M → 16M ops).
At ~500 GB/s bandwidth, this is ~32 ms per step for dense weights.
That's substantial — potentially dominating training time.

**Mitigation options:**

- **Lazy materialization**: θ is never materialized; instead forward
  pass uses `h · θ = Σ σ(α)_k · (h · W_k)` — K matmuls of size M×N,
  each of which is *a single forward of the current W_k*.  Cost: K·
  forward cost.  At K=16 this is 16× forward compute (INFEASIBLE).

- **On-demand materialization**: θ(α) is materialized at the START of
  each step and reused.  Cost: 1·K·N = 16 memcopies per step = 32 ms.
  With ~200 ms typical step time, this is 16% overhead.

- **Rare refresh**: only materialize θ when α changes significantly
  (delta threshold).  Most steps reuse cached θ.  Cost amortized.

We recommend **Option 2 (on-demand, once-per-step materialization)**.
16% overhead is acceptable for the 20× savings on optimizer state.

## 5. Snapshot management

- **Initialization**: bootstrap with K different random init of θ.
  Alternative: first K training steps each produce one snapshot,
  then training switches to WIP mode.

- **Snapshot refresh**: every `N_refresh` steps (typical 100), promote
  the current θ(α) as the newest W_k and evict the oldest.

- **Eviction policy**: LRU (oldest), OR argmin_k σ(α)_k (least-used).

## 6. Required GPU primitives

All from shipped IBGRAD + new thin compositions:

### New (can all be host-side composition of existing kernels):

- `wip_materialize_theta(W_pool, alpha_sm, K, N, theta_out)`:
  weighted sum Σ α_k · W_k.  K sgemm_axpy calls with the current
  α values.

- `wip_project_snapshots(P, W_pool, K, N, r, z_pool_out)`:
  compute `z_k = Pᵀ · W_k` for all K snapshots.  K ibgrad_project
  calls.  Refreshed when P rotates.

- `wip_grad_alpha(z_pool, y, alpha_sm, K, r, g_alpha_out)`:
  compute `g_α_k = σ(α)_k · (⟨z_k, y⟩ - Σ_j σ(α)_j·⟨z_j, y⟩)`.
  K·r flops.  Trivial to implement host-side.

### Reused:

- ibgrad_project, ibgrad_oja_rank1_update, ibgrad_qr_reorthogonalize,
  ibgrad_refresh_first_column (shipped).
- adam_update (shipped).

## 7. Phased rollout

### Phase 1 — primitives (expected: ~300 LOC total):

- `wip_materialize_theta` — simple sgemm-axpy loop.
- `wip_project_snapshots` — K ibgrad_project calls.
- `wip_grad_alpha` — small host-side matmul.
- `wip_snapshot_refresh` — in-place swap of oldest W_k.

### Phase 2 — parity tests:

- `CHIRONWipMaterializeParityTest` — θ = Σ α_k W_k matches CPU reference.
- `CHIRONWipGradAlphaParityTest` — g_α matches chain-rule reference.
- `CHIRONWipIbgradCompoundE2ETest` — 2-layer MLP with both shifts active,
  target: matches IBGRAD-alone loss within 20% at K=16.

### Phase 3 — trainer wire-in (chiron_train):

- `--wip-K N`: enable WIP with K snapshots (default 0 = disabled).
- `--wip-refresh-every R`: snapshot refresh cadence.
- Combines with `--ibgrad-rank R` for the dual-subspace compound.

### Phase 4 — scale benchmark:

- Measure wall-clock vs dense Adam at L=8, m=256.
- Target: 1.5× overall throughput improvement (from α-state Adam + small
  backward).
- Target: 20× Adam-state memory reduction (α is K=16 per matrix vs
  N=500K dense).

## 8. Expected compound numbers

At pile_large (L=24, Wo=1024×512=500K params per matrix):

| Shift | Adam state | Throughput overhead |
|-------|-----------|---------------------|
| Dense | 500K × 8B × 24 layers × 4 matrices = **384 MB** FP32 | baseline |
| int8 Adam (shipped) | 96 MB | -5% |
| IBGRAD (Phase 5) | 1.5 GB (P dominates) | -6% |
| **WIP × IBGRAD** | **K = 16 floats × 24 × 4 = 1.5 KB** (α-only) | -16% (θ materialization) |

**Adam-state compression vs dense: 384 MB → 1.5 KB = 250,000×.**
This is the "magnitudes less memory" demonstration in its purest form.
Caveat: K·N snapshot memory still at 16 × 500K × 24 × 4 × 4B = 6 GB,
which MUST be MPOT-compressed (Option: MPOT 25× → 240 MB) to be
feasible alongside.

## 9. Open questions for implementation

1. **Does K=16 snapshots suffice for expressivity?**  Prior art on
   SWA with K=4-8 snapshots shows small loss gain; with K=16 and
   learned α this should strictly dominate.
2. **How often does α need to be refreshed?**  Hypothesis: every
   N_refresh ≈ 50-100 steps is sufficient.
3. **Gradient projection approximation error**: `⟨W_k, g⟩ ≈ ⟨z_k, y⟩`
   is exact only when W_k lies in P's column span.  Does the Oja-
   learned P eventually span the snapshot space?

## 10. Summary

WIP × IBGRAD is the **highest-priority compound candidate** after the
2026-04-23 re-scoring.  It delivers:

- 250,000× Adam-state compression (384 MB → 1.5 KB at pile_large).
- Composes with MPOT (snapshot pool compression).
- Zero new CUDA kernels — all primitives are compositions of shipped
  IBGRAD + small host-side sum operations.

Implementation effort estimate: **2–3 Ralph-loop iterations** (Phase
1 + Phase 2 parity tests; Phases 3-4 depend on further rank-study).

This doc is the handoff point.  Next iteration: begin Phase 1
(wip_materialize_theta + small parity test).
