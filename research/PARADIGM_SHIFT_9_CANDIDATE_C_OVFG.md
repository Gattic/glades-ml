# Paradigm shift #9 — Candidate C: Operator-Valued Factored Gradient (OVFG)

**Formulation class:** operator-theoretic / representational.
**Author pass:** subagent-dispatched design (2026-04-22).
**Status:** **SELECTED** for shift #9 implementation (see `PARADIGM_SHIFT_9_SELECTION.md`).

---

## Short name

**OVFG** — Operator-Valued Factored Gradient.  Working codename:
*Kernel-Only Optimizer*.

## Premise

For every trainable matrix W ∈ R^{m×n} in a transformer, the gradient
produced by one microbatch is exactly
    G = A^⊤ · D,     A ∈ R^{T×m} (input activations),
                     D ∈ R^{T×n} (upstream gradient).

So **rank(G) ≤ T**.  At pile_large (T=1024, m=n=2368) the operator G
lives in a 1024-dim subspace of a 5.6 M-dim ambient space.  OVFG never
materializes G.  We keep it as the pair (A, D) — two slim matrices
totaling T·(m+n) entries — and push that pair all the way through Adam
and into the weight update.

## Primitive objects

| symbol        | shape              | meaning                                              |
|---------------|--------------------|------------------------------------------------------|
| A_ℓ           | T × m_ℓ            | activation at layer ℓ (already in fwd; BF16)         |
| D_ℓ           | T × n_ℓ            | upstream gradient at layer ℓ (BF16, from backward)   |
| M_ℓ = L_ℓ R_ℓ^⊤ | m_ℓ×r, n_ℓ×r    | first-moment factors, rank r                          |
| c_ℓ ∈ R^{m_ℓ} | —                  | Adafactor row sum (FP32)                             |
| d_ℓ ∈ R^{n_ℓ} | —                  | Adafactor col sum (FP32)                             |
| s_ℓ ∈ R       | scalar             | global Adam step scale                               |
| Q_ℓ ∈ R^{T×r} | sketch             | streaming accumulator basis for microbatch buffer    |
| K_ℓ           | Kronecker tags     | per-layer block structure annotation                  |

**Layer-specific block structure K_ℓ:**

- **Attention QKV** W_{QKV} ∈ R^{m×3n}: three independent factored
  blocks (A, D_Q), (A, D_K), (A, D_V) — same A, three D's.  Fused
  kernel benefits.
- **Per-head attention output** W_O: block-diagonal across heads with
  shared output mix; Kronecker tag K = I_H ⊗ kernel_{d_h}.
- **MLP up/down**: plain (A, D) pairs; dominate memory.
- **Embedding/unembedding** E ∈ R^{V×m}: **SparseFactor** — store
  (tok_ids[T], D), storage T·(1+m).
- **LayerNorm/RMSNorm γ, β**: diagonal, Θ(m) storage; trivial.

## State space

Per non-trivial layer ℓ:
    Σ_ℓ = (L_ℓ, R_ℓ, c_ℓ, d_ℓ, buf_ℓ = {(A_k, D_k)}_{k<k_acc})

Rank r ≤ r_max = T = 1024; production default r_max = 256.
buf_ℓ is per-microbatch, **deleted** at end of every optimizer step.
L, R stored BF16; reductions stay FP32.

Total bytes per non-embedding layer:
    2(m + n) r + 4(m + n) + 2 k_acc T (m + n).

## Evolution law

### (a) Microbatch accumulation

Let k_acc microbatches produce (A_k, D_k).  Accumulated gradient:
    G_acc = Σ_k A_k^⊤ D_k,     rank ≤ k_acc · T.

**Lazy append** (default): stack A_stack = [A_1 | … | A_{k_acc}],
similarly D_stack.  Memory k_acc T (m+n).

**Randomized re-sketch** (when k_acc T > r_max):
    (L_acc, Σ_acc, R_acc) ← RSVD_r(G_acc + A_k^⊤ D_k).

### (b) Adam first moment (factored)

    L' = [√β₁ · L  |  √(1-β₁) · L_acc]
    R' = [√β₁ · R  |  √(1-β₁) · R_acc]

so M' = L' R'^⊤ = β₁ L R^⊤ + (1-β₁) L_acc R_acc^⊤ exactly.  Rank
doubles, then randomized-SVD truncate to r_max.

### (c) Adam second moment (Adafactor)

    c ← β₂ c + (1-β₂) · rowsum(G⊙G)
    d ← β₂ d + (1-β₂) · colsum(G⊙G) / sum(d)

Key reduction:
    rowsum(LR^⊤ ⊙ LR^⊤) = diag(L (R^⊤ R) L^⊤)
→ one r×r Gram + one m×r GEMV = O(mr² + nr²), no mn materialization.

Chosen over Shampoo (m×m, n×n preconditioners blow budget at m=2368).

### (d) Update

    ΔW = − η · diag(1/√c) · L R^⊤ · diag(1/√d) / (normalizer + ε)

**Dense-W path**: one SGEMM, O(mnr) — same as GaLore.

**Stiefel-W path** (composing with shift #7): see below.

## Composability with Stiefel (shift #7)

W = U Σ V^⊤ with U ∈ St(m, ρ), V ∈ St(n, ρ), Σ ∈ R_+^ρ.  Receive
factored gradient G = L R^⊤.  Stiefel tangent-space gradients:

    dU = (I − U U^⊤) · L (R^⊤ V) Σ⁻¹
    dV = (I − V V^⊤) · R (L^⊤ U) Σ⁻¹
    dΣ = diag(U^⊤ L · R^⊤ V)

All three via r×ρ and m×r, n×r, m×ρ GEMMs — **never an m×n tensor**.
Cost O((m+n)(r+ρ)ρ + rρ²).  **This is the payoff clause.**

**Compound compression:**
    Stiefel: 2ρ/d                (ρ=0.25 ⇒ 1/8)
    OVFG:    2r/d                (r=256, d=2368 ⇒ 0.216)
    Joint:   moment & weight compression MULTIPLY

## Composability with HRTC (shift #8)

HRTC compresses T → T/k.  A, D live in the OVFG pipeline and are
T-indexed → HRTC applies directly.  Post-HRTC factor sizes
(T/k)·(m+n).  At k=4: another 4× on the microbatch buffer.

Adam state (L, R) is T-independent (rank r is chosen, not T-coupled);
HRTC compresses the **grad accumulator**, not the **moment**.

## Fused backward-Adam kernel

`ovfg_fused_backward_adam`:

1. Inputs: A [T×m] BF16, D [T×n] BF16, L [m×r] BF16, R [n×r] BF16,
   c [m] FP32, d [n] FP32, Σ [ρ] FP32, U [m×ρ], V [n×ρ].
2. Compute R^⊤V [r×ρ], L^⊤U [r×ρ], R^⊤R [r×r], L^⊤L [r×r] — four
   small GEMMs in shared mem.
3. Append: form [√β₁·L | ΔL] and [√β₁·R | ΔR] logically.
4. Re-rank truncate via randomized SVD with (r+T) × r Gaussian sketch.
5. Row/col-square sums via L(R^⊤R)L^⊤ trick.
6. Emit L, R, c, d; compute dU, dΣ, dV; apply Cayley/QR retraction.

Register pressure on SM 8.9: tiled per r-block (block=32 along r),
<64 registers/thread.  Target occupancy: 2 blocks/SM.

## Memory accounting (honest)

pile_large = 2.23 B (L=48, d=2048, T=1024), ρ=1 (pre-Stiefel), r=256.

| layer type               | count | dense G   | OVFG (L,R) at r=256 | compression |
|--------------------------|------:|----------:|--------------------:|------------:|
| attn QKV (m=2048,n=6144) | 48    | 25.2 MB   | 4.2 MB              | 6.0×        |
| attn out (m=2048,n=2048) | 48    |  8.4 MB   | 2.1 MB              | 4.0×        |
| MLP up (2048→8192)       | 48    | 33.6 MB   | 5.2 MB              | 6.5×        |
| MLP down (8192→2048)     | 48    | 33.6 MB   | 5.2 MB              | 6.5×        |
| unembedding (2048×50257) |  1    |  206 MB   | sparse T·(1+m) = 2.1 MB | 98×     |
| **total optimizer+grad** |       | **~9.6 GB** | **~1.6 GB**       | **6.0×**    |

**Composed with Stiefel ρ=0.25**: dU, dV, dΣ via OVFG factors; total
≈ **0.55 GB** for grad+moments — **~17× vs baseline**.  Headline.

**Composed with HRTC k=4**: microbatch buffer at k_acc=4, T=1024:
without HRTC 60 MB/layer; with HRTC 15 MB/layer.  Aggregate ≈ 2.2 GB.

## Objective / variational principle

Plain NLL objective preserved.  Implicit variational content:

> Among all rank-≤r updates ΔW, OVFG picks the one minimizing
> ‖ΔW − η β₁⁻ᵗ G_true‖_F subject to the Adafactor-preconditioned metric.

A Mahalanobis low-rank projection of the true Adam update.  Novel is
that **we never form G_true**.

## Stability and convergence

**Claim (rank-truncation error):** For transformer gradients with
σ_i ∼ i⁻^α, α ∈ [0.8, 1.5], setting r=256 captures >98% Frobenius
energy at T=1024.

**Claim (BF16 outer-product preservation):** BF16 L, R with FP32 Gram
reductions gives relative error ≤ 2⁻⁷ √r ≈ 0.125 on reconstructed
LR^⊤ — matches BF16-Adam noise floor.

**Claim (convergence):** OVFG is a low-rank-projected Adam.  Under
standard L-smooth + bounded-variance assumptions, rate is O(1/√K) +
rank-truncation bias.  Combined with stochastic-rounded weights
(shift #5), expected update remains unbiased.

**Stiefel composition:** dU, dV are projections of G onto
(m+n)ρ-dim tangent space.  OVFG low-rank approximation of G implies
same for dU, dV with rank cap min(r, ρ).  When r ≥ ρ (our regime),
**no additional error from composition**.

## Minimal prototype (≤ 2 weeks)

1. **Week 1 d1–3**: kernel `ovfg_store_factors` — at backward, write
   A, D to ring buffer instead of dense G.  Measure memory.
2. **Week 1 d4–6**: kernel `ovfg_adam_append_factored` — scaled-append
   to L, R.  Then `ovfg_rsvd_truncate` via cuSOLVER randomized QR.
   Validate parity vs dense Adam at r=min(m,n).
3. **Week 2 d1–3**: Adafactor row/col 2nd-moment via L(R^⊤R)L^⊤ trick.
4. **Week 2 d4–5**: Stiefel coupling — dU, dΣ, dV from (L, R) only.
5. **Week 2 d6–7**: end-to-end on pile_large-small (d=1024, L=24);
   verify loss trajectory within 5% of baseline for 1000 steps at r=256.

Deliverables: `gpu_ovfg.cu` ~900 LOC.  Tests: rank parity, Adam parity,
Stiefel coupling parity, memory bench.

## Relationship to prior work

- **GaLore**: projects gradient onto rank-r subspace; OVFG never forms
  the full gradient and keeps Adam state rank-r.
- **Adafactor**: rank-1 row/col for 2nd moment; OVFG borrows this.
- **Shampoo / SOAP**: Kronecker m×m / n×n preconditioners — OVFG
  restricts to Adafactor-diagonal.
- **KFAC**: statistical Kronecker; OVFG uses exact outer-product
  factorization.
- **Muon**: orthogonalized momentum — OVFG's moment is already rank-r;
  Muon-style Newton–Schulz orthogonalization of LR^⊤ is O(r³), composes
  cleanly.
- **LoRA / Stiefel (shift #7)**: weight-side low-rank — OVFG is the
  **gradient-side dual**.

## Failure modes

1. **Heavy singular-value tail** — σ_i decay slower than i⁻^0.5 (early
   training, MoE routers) → r=256 captures <80% energy → slowdown.
   *Mitigation*: adaptive r_ℓ per layer via cumulative-energy threshold.
2. **Kronecker collapse on attention heads** — head-wise K_ℓ assumes
   heads are gradient-orthogonal.  MoE/tied heads break this.
   *Mitigation*: periodic dense-grad rehearsal every N updates.
3. **Register pressure on SM 8.9** — fused kernel at r=256, T=1024
   uses ~120 regs/thread → occupancy 1 block/SM → 25% theoretical peak.
   *Mitigation*: two-pass variant for production; fused only for r≤128.
4. **Embedding outlier** — V=50257 dwarfs everything; sparse-row works
   only if batch-token coverage sparse.  At batch·T=16384 unique tokens
   ≈ 13000 → 26% coverage, sparse wins.  *Fallback*: Adafactor-only.
5. **RSVD instability at BF16** — BF16 Gaussian sketch rank-deficient
   ≈ 2⁻⁴ fraction → NaN in (QR)⁻¹.  *Mitigation*: FP32 sketch, BF16 matvec.

## Summary

OVFG reorganizes the optimizer around the algebraic identity
∇W = A^⊤ D, storing gradients and moments as factor pairs rather than
dense tensors.  Compounded with Stiefel weights (shift #7) it reaches
≈ 17× total grad+optimizer compression on the pile_large 2.23 B config,
meeting the 4–10× target with headroom.  The approach is orthogonal to
manifold-constrained dynamics (candidate A) and stochastic Langevin
(candidate B); it is a **representational** change, not a dynamical one.
