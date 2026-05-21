# Paradigm shift #16 — Candidate B: Lattice Compute Pooling (LCP)

**Formulation class:** data-side compute compression via cluster-and-correct
decomposition with residual detail network.
**Author pass:** subagent-dispatched design (2026-04-22).
**Status:** candidate — awaiting selection at the shift-16 gate.

---

## 1. Short name and core thesis

**LCP — Lattice Compute Pooling.**  Working codename: *cluster-and-correct
batch*.

Paradigm shifts #1–#13 all compress some axis of the *computation graph*
— activation memory (CHIRON), weight storage (MPOT, Stiefel × Σ),
gradient rank (OVFG), backward dependence (DFA), per-token depth (TRCD).
Every one of them still runs a full forward+backward over all T tokens
in the batch.  LCP attacks the remaining axis: **redundant per-token
compute**.  In a T=2048 pile_large minibatch many tokens are
near-duplicates in representation — the word "the" in N contexts, the
third token of an indented block, the closing paren of a call.  Their
residual-stream trajectories are close.  LCP clusters the batch into
M ≪ T cluster representatives at layer entry, runs the expensive
transformer block on M activations only, and restores per-token
fidelity through a small *detail network* that corrects the
within-cluster deviation.  At M = T/8 this is an 8× reduction in
main-network FLOPs, 7× net after the detail network's fixed cost.

TRCD (#13) proved *per-token compute is a negotiable resource* — it
negotiated along depth.  LCP negotiates along an orthogonal axis: "these
N tokens are close enough that one layer evaluation suffices for all
of them."  A token can be both cheap to depth (few layers) *and* cheap
to batch (shared with 7 peers).  The savings compound.

---

## 2. Formal decomposition

For a smooth function f_θ : ℝ^d → ℝ^d Lipschitz in a neighborhood of
c(x) ∈ ℝ^d, Taylor's theorem gives

    f_θ(x)  =  f_θ(c(x))  +  J_{f_θ}(c(x))·(x − c(x))  +  R_2(x, c(x)),

with second-order remainder ‖R_2‖ ≤ ½·L_θ^{(2)}·‖x − c(x)‖².  If we let
c be the nearest-cluster centroid and d(x − c) = J_{f_θ}(c(x))·(x − c(x))
be a learned Jacobian approximator,

    ‖f_θ(x) − (f_θ(c(x)) + d(x − c(x)))‖  ≤  ½·L_θ^{(2)}·‖x−c‖²
                                              + ‖d − J_{f_θ}(c)·‖·‖x−c‖.

Two error terms: the **curvature residual** shrinks quadratically with
cluster tightness (smaller cluster → smaller ‖x−c‖); the **detail
approximation error** shrinks as d trains to match the local Jacobian.
Both are controllable.  The decomposition is **tight** under the joint
conditions (i) f_θ is twice-continuously-differentiable with bounded
L_θ^{(2)}, (ii) clusters have bounded diameter, (iii) d has rank at
least equal to the local Jacobian spectral width of f_θ at c.

This is the entire mathematical justification for LCP.

---

## 3. Primitive objects

| symbol          | shape                     | meaning                                                 |
|-----------------|---------------------------|---------------------------------------------------------|
| h_t             | ℝ^d                       | per-token residual-stream state at layer entry          |
| c_m             | ℝ^d                       | m-th cluster centroid                                   |
| π : T → M       | assignment                | cluster assignment (soft during training)              |
| A ∈ ℝ^{T×M}     | soft weights              | Σ_m A_{t,m} = 1                                         |
| B_θ             | ℝ^{M×d}→ℝ^{M×d}           | main block (attention + MLP); run on centroids only    |
| D_φ             | ℝ^d→ℝ^d                   | detail correction network, rank-r bottleneck            |
| δ_t             | ℝ^d                       | within-cluster deviation δ_t = h_t − c_{π(t)}           |
| α ∈ [0,1]       | scalar                    | detail-network gain, annealed 0 → 1                    |
| M(l)            | integer                   | per-layer cluster count, annealed T → T/8               |

**Shared detail network.**  D_φ is a single 2-layer SwiGLU MLP of rank
r = d/8 shared across all L layers (plus a per-layer scalar gain
β_l).  Parameter count: 3·r·d ≈ 393 K (for d=1024, r=128), negligible
against the main network.  A single D_φ sees δ inputs from every cluster
at every layer — it learns a *universal local-Jacobian approximator*
for the transformer-block family.  Per-layer D_φ would cost
L·3·r·d = 9.4 M, still tolerable; we default to shared-with-β_l as a
stronger inductive bias and revert to per-layer if warranted (F2
mitigation, §9).

**No persistent codebook.**  π and {c_m} are ephemeral per batch.
Cluster directions are re-computed every forward pass.

---

## 4. Evolution law

### 4.1 Forward pass at layer l

Input: h ∈ ℝ^{T×d}.

1. **Cluster.**  (π, {c_m}, A) ← Cluster(h, M).  (§4.3)
2. **Main block.**  C' = B_θ(stack(c_1,…,c_M))  — the only expensive compute, M ≪ T.
3. **Gather.**  h_t^{main} = Σ_m A_{t,m} · C'_m  (hard limit: h_t^{main} = C'_{π(t)}).
4. **Deviation.**  δ_t = h_t − c_{π(t)}.
5. **Detail.**  y_t = α·β_l·D_φ(δ_t).
6. **Output.**  h_t^{out} = h_t^{main} + y_t.

Total FLOPs per layer: main = (M/T)·F_block(T); detail = T·O(rd); cluster
(LSH) = O(T·d).  At M = T/8: 12.5% + 8% + 1.5% = 22%.  **4.7× reduction
per layer.**  Backward mirrors it: gradient to C' on M rows, backprop
through B_θ at M-token cost, scatter to per-token h via A.

### 4.2 Backward pass

Given ∇_{h^{out}} L:

1. Split: same gradient flows to both branches.
2. **Detail branch.**  ∇_φ L += Σ_t (∂D_φ/∂φ)^⊤·g_{t}; g_h^{detail} = α·β_l·(∂D_φ/∂δ)^⊤·g_t.
3. **Pool to centroids.**  ∇_{C'_m} L = Σ_t A_{t,m}·g_t.  This is *per-cluster gradient pooling* — each cluster's centroid gradient is the A-weighted sum of its members' output gradients.
4. **Main backward on M rows.**  (∇_θ L, ∇_C L) ← back_B_θ(C, C', ∇_{C'} L).
5. **Scatter to per-token h.**  ∇_{h_t}^{cluster} = (A_{t,π(t)}/Σ_s A_{s,π(t)})·∇_{c_{π(t)}} L.
6. Sum: ∇_{h_t} L = ∇_{h_t}^{cluster} + ∇_{h_t}^{detail}.

**Unbiasedness.**  In hard-π, α=1 limit, if D_φ = J_{B_θ}(c) then
h_t^{out} = B_θ(c) + J·δ exactly equals the first-order Taylor
expansion of B_θ(h_t); the gradient w.r.t. θ is the true per-token
gradient up to O(‖δ‖²) bias bounded by §2.

### 4.3 Clustering mechanism

**LSH (production default).**  SimHash with 8 fixed random projections
r_k ∈ ℝ^d (Gaussian, unit-norm):

    bucket(h_t)  =  (sign(h_t^⊤ r_1), …, sign(h_t^⊤ r_8)) ∈ {±1}^8.

256 buckets = M, constant-time assignment, cost O(8·T·d) = 16 MFLOPs
at T=2048, d=1024 — orders of magnitude cheaper than a block.  Fully
parallel on GPU; cluster-stable under small h perturbations; no learned
parameters.  Empty buckets merged with their nearest Hamming neighbor.
Projections fixed at init; changing them across training would
destabilize cluster identities.

**Gumbel-softmax k-means (differentiable; retained for ablation).**
One k-means iteration per layer with τ annealed 1.0 → 0.1:
a_t = softmax_M(−‖h_t − c_m‖²/τ + Gumbel).  Advantage: clusters track
geometry; disadvantage: one step ≈ 50% of a block, destroying the
saving.  Used only to probe curvature-residual contribution.

**Prefix-sort by hash** — treated as equivalent to LSH for rank-8
binary hashes; no extra machinery needed.

### 4.4 Joint loss

    L  =  L_CE  +  λ_d · L_detail_aux  +  λ_r · L_residual

- **L_CE** — standard cross-entropy on final logits (propagates
  through both branches).
- **L_detail_aux** — pins D_φ to the *observed* Jacobian residual:

      L_detail_aux  =  Σ_{l,t_sub} ‖ D_φ(δ_{l,t}) − (B_θ(h_{l,t})^{sg} − c'_{π(l,t)}^{sg}) ‖²,

  evaluated on a 1/16 random subset of tokens per step (t_sub).  The
  full-T block output is computed for just 128 tokens per batch — cost
  ≈ 6% of one extra main block.  This *directly supervises* D_φ with
  the residual it must predict, blocking collapse to zero.  λ_d = 0.05.
- **L_residual** — contraction penalty keeping ‖δ‖ in range:
  L_residual = λ_r · mean_t max(0, ‖δ_t‖² − R²).  λ_r = 0.01.

### 4.5 Annealing schedule

Three primitives annealed jointly over a 500-step warm-up:
1. M : T → T/8 linearly.  Early steps see the full signal; compression
   engages once the network has coherent representations.
2. α : 0 → 1 linearly.  Early, cluster-only main output — fast even if
   clustering is noisy; detail engages later on a stationary target.
3. τ : 1.0 → 0.1 (if using differentiable clustering).

**Lyapunov argument.**  At (M=T, α=0) LCP is exact vanilla training.
Smoothly decreasing M with Lipschitz B_θ grows the Taylor bound
continuously at rate O(1/M); α ramp engages D_φ only after L_CE
stabilizes at the cluster-coarse level.  No schedule admits runaway.

---

## 5. Mechanism ledger

**Speed.**  Per-layer FLOPs: baseline 6.2·10⁹; LCP 1.32·10⁹ → **4.7×
reduction** at M=T/8.  Symmetric for backward.

**Memory.**  Main-branch activations: M·d = 256·1024 = 0.5 MB/layer
BF16 vs baseline 4 MB — **8× activation reduction additive with
CHIRON's O(1)-in-depth**.  δ derivable from h − c without extra
storage; D_φ hidden = T·r = 0.5 MB.

**Stability.**  Detail network + auxiliary loss preserves per-token
signal.  Taylor bound quantifies fidelity.  α ramp and M ramp ensure
the network is never compressed past what it can bear.

---

## 6. Composition with TRCD (#13) — the multiplicative story

Both are data-side compute reductions, so verifying composition is
first-order.  TRCD skips *layers* per token; LCP compresses *tokens*
per layer.  These are a row × column factoring of the (L, T)
forward-compute matrix — TRCD is row-reduction, LCP column-reduction.
Composed:

    FLOPs_compound  =  Σ_l  M_l · d̄_l · F_block_per_token
                    ≤  (T/8) · (L/3) · F_block
                    =  L·T·F_block / 24.

**24× of dense baseline** at pile_large L=24, T=2048.  No interaction
term — LCP clusters *over the alive subset at each layer* after TRCD
has decided which tokens continue, setting M_l = max(8, N_l/8).  No
extra math; the algorithm is literally "TRCD gate → LSH cluster → main
block on M_l centroids."  Practical wall-clock factor ~12–15× after
overheads; still a >1 dex improvement.

This composability is LCP's structural fit against the existing stack
and the main reason to prefer it as shift #16 over a redesign of a
more compressed axis.

---

## 7. Composition with shifts #1–#12

- **#1 CHIRON** — main block reversibility preserved on centroids; detail block optional (irreversible D_φ requires storing 0.25 MB/layer).
- **#2/#6 attention kernels** — local-window and gather/scatter kernels already accept variable counts.  LCP adds a **cluster-Q / full-KV** attention variant (§F6): Q from M centroids, K/V from T tokens, scores M×T not T×T — a further T/M FLOP reduction specific to attention.
- **#7 Stiefel × Σ / #10 MPOT** — θ on product manifold; ∇_θ is pooled average over centroid gradients; retraction unchanged.
- **#9 OVFG** — rank-r factorization of ∇_θ invariant; LCP only changes the batch seen by ∇_θ from T to M.
- **#11 MFIO** — operates on the M×d sub-batch.
- **#12 DFA** — feedback B_l applied to the centroid gradient then scattered via A^⊤.

Every shift in 1-13 operates on an axis orthogonal to the batch axis
LCP compresses, so LCP adds to the stack with only minor kernel-level
adjustments.

---

## 8. Relation to prior work

- **Mixture-of-Experts** routes tokens to different *weight subsets*;
  LCP routes tokens to representatives of the *same* shared network.
  Orthogonal objectives.
- **Perceiver** uses cross-attention to a fixed K-latent bottleneck,
  destroying per-token output; LCP *preserves* per-token fidelity
  through the detail network.
- **Reformer / Routing Transformer** cluster tokens for attention only;
  LCP clusters for the whole block including MLP (the FLOP-dominant
  half).
- **DEQ / fixed-point networks** pool compute over depth; LCP over
  batch.  Complementary.
- **Matryoshka / Poincaré / hyperbolic clustering** — share a
  representation across widths or geometries; none reduces main-network
  forward FLOPs per token.

**Novelty.**  No prior method (a) clusters tokens at each layer into a
sub-batch, (b) runs the main block on cluster representatives,
(c) reconstructs per-token outputs through a Jacobian-approximating
detail network, and (d) trains all of it under cross-entropy plus a
local-Jacobian auxiliary.  The triple [cluster + main-pool +
Jacobian-detail-network] is the novel combination — the
representation stays dense while compute is sparsified.

---

## 9. Failure modes and mitigations

**(F1) Cluster-assignment cost eats savings.**  Naive k-means is ~40%
of a block; would destroy the win.
*Mitigation.*  LSH at 16 MFLOPs/layer — 500× cheaper than k-means,
sufficient per §4.3.  Probe: compare Taylor residual ‖h − h^{LCP}‖
between LSH and k-means on a 1000-step probe; accept LSH if residual
≤ 1.5× k-means residual.  Fallback: warm-started single-pass k-means
at ~2% of a block.

**(F2) Detail network too small to preserve per-token signal.**
*Mitigation.*  (a) L_detail_aux supervises D_φ directly on the
residual it must predict — measurable and bounded.  (b) Rank r scaled
to empirical cluster variance σ²_δ: if 99th-%ile ‖δ‖² exceeds
threshold, bump r to d/4 (detail cost 16% of block, net still 5×).
(c) Per-layer β_l absorbs depth-dependent Jacobian scale.
(d) Escape hatch: promote to per-layer D_φ (L·393 K = 9.4 M params) —
still trivial cost.

**(F3) Discrete clustering breaks gradient flow.**
*Mitigation.*  (a) Soft A via Gumbel-softmax in the differentiable
path; hard decision only affects which index gathers C'_m.
(b) Straight-through estimator for A_hard / A_soft.  (c) LSH itself
is treated as a fixed non-differentiable operator — admissible because
the projections are fixed and cluster membership is piecewise-constant
in h.  LCP tolerates the zero ∂π/∂h because *D_φ provides continuous
per-token gradient signal that bypasses π entirely* — the chain-rule
path through the cluster branch is the smooth gather+scatter operator,
not the discrete argmin.

**(F4) M too small → mode collapse.**  Extreme M = 1 yields one
cluster, detail network alone cannot do the full task.
*Mitigation.*  (a) M annealed T → T/8 across 500 steps; if loss
deviates >2% from baseline curve, M ramp halts at the largest M where
tracking holds.  (b) Hard floor M ≥ max(8, T/32).  (c) Auto-scaled r
via F2(b).

**(F5) Cross-layer cluster inconsistency.**  π_l and π_{l+1} differ,
introducing gradient noise.
*Mitigation.*  (a) Cluster persistence: reuse π_l for l+1 when LSH
hash of h_{l+1} is Hamming-close to h_l (early experiments: >80%
persistence after warm-up).  (b) Re-cluster otherwise.  (c) β_l gain
absorbs cross-layer misalignment.

**(F6) Attention sublayer needs full K/V.**  Q, K, V come from the
same residual stream; clustering before attention would drop context.
*Mitigation.*  **Cluster-Q / full-KV split.**  Q materialized from M
centroids; K, V from all T tokens.  Attention scores M×T, not T×T —
a further T/M FLOP reduction specific to attention, on top of the
block win.  Output at M centroids then gathered to T tokens via A.

---

## 10. Implementation sketch

**Phase 1 — GPU primitives.**
- `lcp_lsh_assign` — batched SimHash (T,)→(M,), ~80 LOC CUDA.
- `lcp_cluster_mean` — `indexed_scatter_add_rowmajor` + normalization wrapper, ~40 LOC.
- `lcp_detail_mlp_fwd/bwd` — SwiGLU at rank r, ~120 LOC.
- `lcp_detail_aux_loss` — Jacobian-residual loss on 1/16 subset, ~60 LOC.
- `lcp_cluster_attn` — cluster-Q/full-KV attention variant of local-window kernel, ~200 LOC.

**Phase 2 — CHIRON-test suite parity.**
- `CHIRONLcpLshAssignmentTest` — >95% cluster persistence under perturbation.
- `CHIRONLcpGradientParityTest` — at (M=T, α=1) must match dense exactly.
- `CHIRONLcpTaylorResidualTest` — measure ‖h^{full} − h^{LCP}‖/‖h‖ at M=T/8 across 100 random inputs; accept if mean < 0.05.
- `CHIRONLcpClusterAttnParityTest` — cluster-Q/full-KV vs full attention within 1e-3 at r = d/8.
- `CHIRONLcpEndToEndConvergenceTest` — 300-step toy pile within 5% of dense.
- `CHIRONLcpTrcdComposeTest` — LCP + TRCD target ≥12× vs dense; verify ≥8×.

**Phase 3 — trainer flags (glades_pile_train).**
- `--lcp-cluster-ratio R` (M = T/R; 0 disabled).
- `--lcp-detail-rank r` (default d/8).
- `--lcp-warmup-steps K` (default 500).
- `--lcp-cluster-method {lsh, kmeans}` (default lsh).
- `--lcp-detail-aux-weight λ_d` (default 0.05).

**Phase 4 — scale benchmark.**
- pile_large L=24, T=2048, M=256: target ≥3× end-to-end throughput.
- 2.23 B B-run, L=48, M=256: target 3.5–4.0 B memory ceiling.
- Compound run (LCP + TRCD + CHIRON + #7 stack): throughput ≥180 k tok/s at NLL parity within 0.2 nats.

**Estimated effort.** ~2 engineering weeks.  Core risk: detail-network
convergence, monitored by L_detail_aux and Taylor-residual probe.  All
three risks have pre-identified fallbacks — raise r, raise M, upgrade
LSH to k-means — none forcing redesign.

---

## 11. Open conjectures

1. **Cluster-Taylor hypothesis.**  σ²_δ/‖h‖² ≤ 0.1 at M = T/8.  Test:
   log σ²_δ across 500 warm-up steps.  If violated, enlarge r.
2. **Jacobian-detail hypothesis.**  A single shared MLP at rank d/8
   suffices to approximate the transformer-block local Jacobian
   uniformly over the training trajectory.  Test: L_detail_aux → <0.05
   relative within warm-up.  If not, fall back to per-layer D_φ.
3. **LCP × TRCD compound hypothesis.**  The two shifts compose
   multiplicatively with NLL parity within 0.2 nats of dense.  Test:
   3000-step compound run vs dense, LCP-only, TRCD-only baselines.
4. **Attention-cluster-split sanity.**  Cluster-Q / full-KV attention
   is a valid drop-in at M = T/8 with NLL parity within 0.05 nats.  If
   not, decouple attention-M from MLP-M.

---

*End of candidate B.*
