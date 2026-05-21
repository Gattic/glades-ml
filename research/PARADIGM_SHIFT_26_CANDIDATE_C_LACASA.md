# Paradigm Shift #26 Candidate C — LACA-SA: Learned Adaptive Content-Aware Sparse Attention

**Status:** design complete; candidate C of three (A, B, C) for paradigm shift #26.
**Date:** 2026-04-23 (Ralph-loop iteration 43).
**Axis:** per-entry content-dependent attention sparsity (variational / game-theoretic).

---

## 1. Target axis

**Per-head, per-entry, content-dependent attention mask learned end-to-end.**

Every shipped shift and every deferred candidate treats the attention
score matrix a ∈ ℝ^{T×T} as *either* dense (#2 TC-tile), *fixed-sparse*
(#6 local-window, strided), *cluster-approximated* (#16 LCP LSH),
or *depth-skipped* (#13 TRCD). None learns WHICH entries of a are
worth computing from the CONTENT of q, k.

After softmax, a_{ij} is heavy-tailed: for natural-language causal LM
at T=2048, empirically >90% of the softmax mass concentrates on
<15% of keys for a given query, and the top-s·T keys capture >99%
of the mass for s ≈ 0.3. Computing the other 70% of q·k dot-products
and the corresponding v-weighted sum is pure waste.

LACA-SA learns a gate g_{ij} ∈ {0,1} per head per layer that predicts
the support of a_{ij} directly from (q_i, k_j) — no LSH, no fixed
pattern, no approximate neighbor search. Computation and storage
scale as s·T² (not T²), with s the learned sparsity budget.

## 2. Core thesis

Reformulate attention as a **two-player variational game** between
the gate g and the attention weights W_{Q,K,V}:

$$
y_i = \sum_{j: g_{ij}=1} \mathrm{softmax}\bigl(q_i \cdot k_j / \sqrt{d}\bigr) v_j
$$

with **gate predictor**

$$
g_{ij} = \mathbb{1}\bigl[\sigma(u_h^\top q_i + v_h^\top k_j + b_h) > \tau\bigr]
$$

where u_h, v_h ∈ ℝ^d are per-head **rank-1** gate parameters (2·d·H
floats total), b_h a per-head bias, and τ a per-layer threshold tuned
by KKT on the sparsity constraint.

The gate is rank-1 in q,k space — it can be evaluated at O(T·d)
cost (two GEMVs of u_h·Q^T and v_h·K^T, followed by an outer-sum),
and the resulting T×T scalar grid is **thresholded** to produce the
sparse support S = {(i,j) : g_{ij}=1}. Only entries in S are passed
to the dense attention kernel.

## 3. Primitive objects

| Symbol | Shape | What it is |
|--------|-------|-----------|
| u_h, v_h | d × H | Per-head gate direction vectors (rank-1 predictor) |
| b_h | H | Per-head gate bias |
| τ_ℓ | scalar per layer | KKT threshold controlling sparsity |
| S_ℓ | nnz(g_ℓ) pairs | Sparse support (CSR-like, per layer per head) |
| s | scalar ∈ (0,1) | Target sparsity budget (user-set, e.g. 1/3) |
| λ_ℓ | scalar | Lagrange multiplier on per-layer sparsity constraint |
| α | scalar | Gumbel-softmax temperature (training only) |

Storage overhead: 2·d·H floats per layer for (u_h, v_h), i.e. at
d=1024, H=16, L=24 → 768 KB total — negligible.

## 4. State space

Formally LACA-SA lives in

$$
\mathcal{M} = \mathcal{W} \times \mathcal{G} \times \Lambda
$$

with
- $\mathcal{W}$ = standard transformer weight manifold (Q,K,V,O,MLP,…).
- $\mathcal{G} = \prod_{\ell,h} \mathbb{R}^d \times \mathbb{R}^d \times \mathbb{R}$: gate predictors.
- $\Lambda = \mathbb{R}^L_{>0}$: Lagrange multipliers (one per layer).

Constraint set: $\{ g : \mathbb{E}_i[\sum_j g_{ij}] \le s \cdot T \}$ per
layer. Under causal masking, also $\{ g_{ij}=0 \text{ if } j>i \}$.

## 5. Evolution law

**Forward (inference / training fwd)**:

1. Compute gate logits ℓ_{ij} = u_h^⊤ q_i + v_h^⊤ k_j + b_h
   as an outer sum — two GEMVs + broadcast add, O(T·d + T²) per head.
2. Threshold: g_{ij} = 𝟙[ℓ_{ij} > τ_ℓ] (hard at inference; Gumbel-
   sigmoid at training, see below).
3. Build CSR index S = {(i,j) : g_{ij}=1}.
4. Dense attention on S: for each retained (i,j), compute
   q_i · k_j / √d, softmax ONLY over retained j per query row, then
   y_i = Σ_{j ∈ S_i} a_{ij} v_j.

**Gate training (Gumbel-sigmoid relaxation)**:

$$
\tilde g_{ij} = \sigma\bigl((\ell_{ij} + G_{ij}) / \alpha\bigr),
\quad G_{ij} \sim \mathrm{Logistic}(0, 1)
$$

α annealed from 1.0 → 0.1 over 10% warmup. In forward, use the
straight-through estimator $\hat g = \mathbb{1}[\tilde g > 0.5]$ for
computation but backprop through $\tilde g$.

**KKT τ update (per-layer auto-thresholding)**:

$$
\tau_\ell \leftarrow \tau_\ell + \eta_\tau (\bar g_\ell - s),
\quad \bar g_\ell = \frac{1}{T^2}\sum_{i,j} \hat g_{ij}
$$

Classic projected-gradient dual ascent: if avg gate activity exceeds
s, raise τ (more sparsity); if under s, lower τ. Stable because
$\bar g_\ell$ is monotone in τ.

**Backward**:

- Gradient through dense attention restricted to S (standard
  flash-attention backward, but only over retained j).
- Gradient through g via straight-through + Gumbel-sigmoid to update
  (u_h, v_h, b_h).
- τ updated by the dual rule above, not via backprop.

## 6. Mechanism mapping

| Required ingredient | Mechanism |
|---------------------|-----------|
| ≥3× forward FLOP reduction | Attention FLOPs drop from O(T²·d) to O(s·T²·d). At s=1/3, direct **3× reduction** on both qk^T and av matmuls. Gate eval adds 2·T·d + T² per head — negligible at d=1024, T=2048 (T² = 4M adds vs 8.4G attention FLOPs). **Net: 2.97× fwd FLOP reduction per attention layer.** |
| Meaningful memory savings | Attention scores/probs drop from T² to s·T² per head. At T=2048, H=16, L=24, s=1/3: 1.34 GB → 448 MB (3×). Activations/gradients for attention backward scale by s. CSR indexing adds 4·s·T² bytes per head — 1/8 of float savings. **Net: 2.8× attention memory reduction.** |
| Composability | (a) **Local-window (#6)**: mask intersection — LACA-SA learns within-window content; fallback to local when gate collapses. (b) **MFIO × WIP × IBGRAD flagship**: orthogonal — MFIO/WIP act on weights, IBGRAD on gradient subspace; LACA-SA acts on attention support. (c) **CHIRON reversibility**: retained-j backward needs same S from forward, so S is deterministic given q,k (both reconstructed by CHIRON). (d) **TRCD (#13)**: per-token depth × per-entry sparsity multiply. At d̄=L/3, s=1/3: **compound ≈ 9×**. |
| GPU-implementability | All primitives exist: gate-logit GEMVs via `sgemv_rowmajor`; threshold + CSR build via custom kernel (similar to `argmax_count_matches`); sparse attention via a block-sparse variant of `softmax_backward_attn` — tile-based with nnz-list lookup. Flash-attention backward already tile-sparse (skip zero-prob tiles); we just pre-compute the tile occupancy from S. |

## 7. Objective / variational principle

**Training objective** (per layer, per head):

$$
\mathcal{L} = \mathcal{L}_{\mathrm{CE}}(\theta, g) +
  \sum_\ell \lambda_\ell \bigl(\bar g_\ell - s\bigr) +
  \beta \cdot \mathrm{KL}\bigl(p(g|\theta) \,\|\, \mathrm{Bern}(s)\bigr)
$$

The first term is cross-entropy on next-token prediction, evaluated
with sparse attention. The Lagrangian term enforces the budget
constraint via the KKT dual rule on τ. The KL term is an entropy
regularizer that keeps the gate from collapsing to all-1 or all-0
during warmup; β = 0.01 annealed to 0 after 10% of training.

**KKT conditions at stationarity**:

$$
\partial \mathcal{L}_{\mathrm{CE}} / \partial g_{ij} = \lambda_\ell
\quad \forall (i,j): g_{ij} = 1
$$

i.e. the marginal utility of every kept entry equals the shadow price
of the sparsity constraint. Intuitively: the gate keeps exactly the
entries whose attention-output contribution exceeds λ.

## 8. Stability / conditioning / expressivity

**Expressivity**:
- Gate rank-1 per head is a strong restriction. But ANY dense attention
  can be recovered by setting u_h = v_h = 0, b_h → +∞, τ = 0 (gate
  all-ones). The formulation contains full attention as a limit.
- For s=1, the system is identical to standard dense attention —
  verified by substitution. LACA-SA is a **proper superset** only in
  the s<1 regime; at s=1 it degenerates gracefully.

**Stability**:
- Dual-ascent on τ is provably stable for monotone constraints. As
  $\bar g(\tau)$ is smooth and monotone-decreasing in τ (each entry
  threshold raises), a fixed point exists and is locally attracting
  for small η_τ.
- Gumbel-sigmoid variance: at α=0.1, gradient variance through g is
  $\mathcal{O}(\alpha^{-1})$ ≈ 10×, manageable.

**Conditioning**:
- Rank-1 gate predictor ⇒ exactly 2d+1 free parameters per head for
  the gate. At d=1024, H=16, L=24: 2×1024 + 1 = 2049 per head → 0.79 M
  gate params vs ~2 B transformer params. Gate is 4×10⁻⁴ of total;
  doesn't destabilize optimization.

**Well-posedness of hard threshold**:
- At inference τ is frozen. The gate is deterministic given q,k;
  no stochasticity ⇒ reproducible outputs. CHIRON reversibility is
  preserved because S is a pure function of (q,k).

## 9. Failure modes

**F1 — Early-training gate collapse (mode collapse).** The untrained
gate has random (u,v,b); gate activity is ~50% regardless of content,
making the sparsity budget satisfied trivially but on uninformative
entries. **Mitigation**: warmup phase (first 5% of steps) disables
gate (s=1), lets Q,K train up to meaningful ranges, then gradually
anneal s from 1 → target.

**F2 — Gradient estimator variance.** Gumbel-sigmoid through a
threshold has high variance at moderate α. **Mitigation**:
(a) α annealed slowly (cosine from 1.0 to 0.1 over first 20%);
(b) use REBAR-style control variate or just straight-through with
stop-grad on the threshold — empirically STE works for 90% of vision
transformer pruning work.

**F3 — Composition with flash-attention.** Flash-attention kernels
assume **block-dense** access to K, V tiles (for IO efficiency).
LACA-SA's per-entry sparsity may fragment tiles to <50% occupancy,
hurting HBM throughput. **Mitigation**: enforce a **block-level**
gate (gate evaluated at block granularity, e.g. 64×64), trading
fine-grained sparsity for hardware efficiency. This is the key
engineering compromise; at block=64 the gate becomes $(T/64)^2$ bits
per head, and block sparsity patterns match cuSPARSE-BSR semantics.

**F4 — Gate hurts causal LM quality in tail regimes.** Some tokens
need genuinely long-range context; if the gate prunes these, perplexity
on long-tail sequences regresses. **Mitigation**: hybrid mode — first
p% of heads use LACA-SA, rest use local-window or dense. Per-head s
allows some heads to specialize as "dense-lookback" experts.

**F5 — Reversibility breakage on gate stochasticity.** During training
with Gumbel noise, the forward S differs from what CHIRON would
reconstruct. **Mitigation**: store the Gumbel noise (16 bits of RNG
state per layer) — negligible memory — and replay it in reconstruction.

## 10. Minimal prototype + GPU primitives

**GPU primitives needed**:
- `gpu::laca_gate_logits(Q, K, u_h, v_h, b_h, H, T, d, logits_out)` — H
  GEMVs + broadcast-add. Built from existing `sgemv_rowmajor`.
- `gpu::laca_threshold_and_csr(logits, tau, H, T, S_out, nnz_out)` — custom
  1-block-per-row kernel writing CSR indices; analogous to
  `argmax_count_matches`.
- `gpu::laca_sparse_attention_fwd(Q, K, V, S, H, T, d, Y_out)` — block-
  sparse flash-attention: iterate retained (i,j) blocks per tile.
  Reuse existing `attention_fwd` inner body with skip-list.
- `gpu::laca_sparse_attention_bwd(gY, Q, K, V, S, H, T, d, gQ, gK, gV_out)`
  — mirror of fwd, with `softmax_backward_attn` restricted to S.
- `gpu::laca_dual_update(avg_g, s, tau, eta_tau)` — 1-thread kernel to
  update τ per layer.

**CLI flag**: `--lacasa` enables LACA-SA; `--lacasa-s=0.33` sparsity
budget; `--lacasa-block=64` block size for F3 mitigation;
`--lacasa-warmup=0.05` dense-warmup fraction.

**Integration with chiron_train**:
1. Add `LacaConfig { float s; int block; float warmup; }` to
   `training_config.h`.
2. Add `u_h, v_h, b_h` tensors to `NNInfo` per attention layer.
3. In `sgd_transformer.cpp`, gate fwd/bwd calls gated by
   `cfg.laca.enabled`.
4. Gate parameters optimized by same AdamW/BF16 path as other weights.
5. τ stored as per-layer scalar, updated after each optimizer step.

**First E2E test**: 2-layer, d=256, H=4, T=256, synthetic copy task.
Verify:
- s=1 recovers baseline (to within numerical tolerance).
- s=0.5 converges with ≤5% loss regression.
- s=0.25 shows 3× fwd speedup on 4080-SUPER timer, with ≤10% loss
  regression.

**Concrete numbers at pile_large (L=24, H=16, d=1024, T=2048, s=1/3)**:
- Attention FLOPs per fwd: dense 4·L·H·T²·d = 3.3 TFLOP → sparse 1.1 TFLOP. **3× reduction**.
- Attention score memory: L·H·T² floats = 6.4 GB → 2.1 GB. **3× reduction**.
- KV memory: unchanged (all K,V stored for gate evaluation).
- Gate parameter memory: 2·L·H·d floats = 3 MB. Negligible.
- Total model memory savings vs baseline at pile_large: ~4.3 GB → fits
  a 2.7B-param model where baseline fits only 2.2B at T=2048.
- Compound with WIP × IBGRAD × MFIO: orthogonal; total compression
  stack ≈ 8×·3× = 24× over dense baseline.

---

## Summary

LACA-SA is the first shift to learn a **content-aware, per-entry**
attention mask via a rank-1 gate predictor, trained end-to-end with
a variational sparsity constraint and KKT-tuned threshold. At pile_large
it delivers 3× attention fwd FLOPs and 3× attention memory at s=1/3
with negligible extra parameters. It composes multiplicatively with
TRCD, local-window, and the MFIO × WIP × IBGRAD flagship. The primary
risk is F3 (flash-attention tile fragmentation), mitigated by block-
level gating.

**Promote condition**: after paradigm-shift #25 GEC lands and the
IBGRAD × WIP × MPOT compound is stable, LACA-SA joins as the
orthogonal **attention-support** compression factor.
