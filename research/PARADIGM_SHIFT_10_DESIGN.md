# Paradigm shift #10 — design brief

**Date**: 2026-04-22.
**Status**: design phase.  Implementation deferred to subsequent
Ralph-loop iterations.

---

## Context — where the 16 GB ceiling is now

After shifts #1–#9 (pending OVFG wire-in) + chunked CE, the projected
memory breakdown at 2.23 B parameters on a 16 GB consumer GPU is:

| category                       | size (projected) | attacked by |
|--------------------------------|-----------------:|-------------|
| Weights (Stiefel, ρ=0.25)      |    ~1 GB         | shift #7    |
| Gradients + Adam moments       |    ~0.5 GB       | shift #9    |
| Activations (CHIRON)           |    ~0.05 GB      | shift #1    |
| Logits + scratch (chunked CE)  |    ~0.1 GB       | chunked-CE  |
| Trunk tiles + residuals        |    ~1.5 GB       | —           |
| **Total**                      |    **~3.2 GB**   |             |
| **Headroom**                   |    **~12.8 GB**  |             |

Headroom now supports ~6–8 B parameters.  The next binding constraint
is **weights** — at ρ=0.25 Stiefel, a 10 B model has ~4 GB weight
storage, and a 20 B has ~8 GB.  To reach the 10–30 B range on the
same hardware, weight compression needs another 4–10× independent
axis beyond Stiefel.

## Target

**Paradigm shift #10 compresses the weight axis by an additional 4–10×
on top of Stiefel-ρ=0.25**, without loss of convergence at ≥ 2.23 B
scale, while composing with shifts #1–#9 and chunked CE.  Combined
compression target: 16–40× on weight storage, unlocking 10–30 B
parameter training on a single 16 GB GPU.

## Three candidate formulations (narrative, not dispatched)

### Candidate A — Matrix Product Operator decomposition (MPOT)

*Mathematical object*: replace each weight matrix `W ∈ R^{m × n}` with
an MPO of bond dimension `D`:

    W[i_1 … i_k, j_1 … j_k]
        = Σ_{α_1 … α_{k-1}}  A^(1)[i_1, j_1, α_1]
                            · A^(2)[α_1, i_2, j_2, α_2]
                            · … · A^(k)[α_{k-1}, i_k, j_k]

where `m = m_1 · m_2 · … · m_k`, `n = n_1 · n_2 · … · n_k`.  Cost
ratio: `m·n  →  k · D² · max(m_l · n_l)` ≈ **(D² · k) / (m · n)^{(k-1)/k}**.
At m = n = 2048, k = 3, D = 8: compression ≈ 100× (heuristic; depends
on singular-value decay in the factorization).

*Composability*: MPO W enters every GEMM as a chained contraction.
Forward pass cost is `O(k · D² · T · max(m_l, n_l))` vs `O(T · m · n)`
— potentially faster, not just smaller.  Composes with Stiefel by
applying MPO *inside* the σ factor (treat Σ as a length-r vector that
modulates a shared MPO operator).

*Prior art*: Novikov et al. 2015 "Tensorizing Neural Networks"
(TTs for FC layers), Garipov et al. 2016 "Ultimate tensorization"
showed 200k× compression for specific MNIST layers.  Transformer-scale
application is largely unexplored — a real paradigm-shift opportunity
if we can preserve training dynamics.

*Risks*: MPO-parameterized weights may have worse conditioning (the
training landscape loses isotropy).  Convergence at LLM scale is
unproven.  Kernel engineering (chained contractions) is substantial.

### Candidate B — Implicit Neural Weights (INW)

*Mathematical object*: each weight tensor `W[i, j] = f_θ(enc(i, j))`
where `f_θ` is a tiny MLP (≤ 10k params) and `enc` is a coordinate
encoding (e.g., Fourier feature positional encoding).

*Cost ratio*: m · n → |θ|.  For m = n = 2048, |θ| = 10k: 419×
compression.  Weights are materialized lazily on demand in tiles
during each forward pass.

*Composability*: f_θ tiles into arbitrary shapes, so composes with
Stiefel, HRTC, and local attention naturally.  Per-layer θ_l allows
expressivity while sharing the tiny MLP structure.

*Risks*: on-demand weight generation is *slow* (per-tile MLP eval in
forward + backward).  Total extra FLOPs per training step: |θ| · (m + n)
per tile, times number of tiles.  At small tile sizes this is a
substantial constant factor.  Training dynamics for implicit-weight
networks are largely unstudied at scale.  Gradient flow through
`f_θ` requires careful design to avoid collapse.

*Prior art*: INRs (Implicit Neural Representations) are well-developed
for 3D / images (Occupancy Networks, NeRF).  Weight INRs are a fringe
research direction; no published LLM-scale result.

### Candidate C — Weight sharing via cyclic modulation (CMOD)

*Mathematical object*: a 48-layer transformer uses 48 "logical" weight
tensors but only stores 8 "physical" ones.  Each logical weight
`W_l = W_{g(l)} ⊙ (1 + A_l)` where `g(l)` is a cyclic group index
(L layers mod 8 physical groups), `A_l` is a rank-1 per-layer
modulation `u_l · v_l^T`, and ⊙ is element-wise multiply.

*Cost ratio*: L · m · n → (L / G) · m · n + L · (m + n)
where G is the group count.  At L = 48, G = 8, m = n = 2048: compression
≈ 7×.  Rank-1 modulation allows each layer to differentiate without
storing a full weight tensor.

*Composability*: compose with Stiefel by sharing `U_{g(l)}`, `V_{g(l)}`
across groups, storing per-layer `Σ_l` + `u_l, v_l` rank-1
modulations.  Combined compression: G · ρ · d.

*Risks*: forced parameter sharing across depth constrains expressivity.
Reversible-flow (CHIRON) pairs well with this since the inverse walk
already assumes depth-cyclic structure.  Recent work (CrammBERT,
cramming-LLMs) shows moderate weight sharing works; CMOD extends to
aggressive G = L/8 or even L/16.

## Selection — MPOT

**Primary**: Candidate A (MPOT).  Rationale:

1. **Biggest upside**: 100× weight compression in favorable regimes,
   compared to 7× for CMOD and 419× for INW-but-slow.
2. **Forward-pass speedup is a BONUS**: MPO contractions run faster
   than a dense GEMM when D is small, potentially delivering memory
   AND speed together.
3. **Mathematically clean**: bond-dimension D is the single hyperparam,
   with known approximation theory (SVD-truncation of bond).
4. **Composes with Stiefel**: the Σ factor in Stiefel × Σ can live
   *inside* the MPO as a length-r modulation on the chain.
5. **Implementable incrementally**: start with 2-site MPO (k=2), which
   is just a 3-tensor contraction — simpler than a full MPS solver.

**Reserve candidates**:
- CMOD for a near-term win if MPO convergence proves difficult.
- INW as a long-term research target with a dedicated experimental
  branch.

## MPOT formulation — mathematical detail

### Primitive objects

Let `W ∈ R^{m × n}` be a weight matrix.  Factor dims as `m = m_1 · m_2`,
`n = n_1 · n_2` (choose m_l ≈ √m, n_l ≈ √n for balanced bond traffic).
The MPO representation:

    W[i, j]  where  i = i_1 · m_2 + i_2,  j = j_1 · n_2 + j_2
    W[i, j] = Σ_α  A[i_1, j_1, α]  ·  B[α, i_2, j_2]

with bond dim `α ∈ [0, D)`.  Storage:
`m_1 · n_1 · D  +  D · m_2 · n_2`
≈ `D · (m_1 n_1 + m_2 n_2)` ≈ `2 D · √(m · n)` for balanced m_l ≈ n_l.

At m = n = 2048, D = 16: storage = 2 · 16 · 2048 = 65 K.  Compared to
m · n = 4.2 M: **65× compression**.

### Forward GEMM

Given input `X ∈ R^{T × m}` and weight as (A, B), compute `Y = X · W^T ∈ R^{T × n}`:

    Y[t, j_1, j_2]
        = Σ_{i_1, i_2, α}  X[t, i_1 · m_2 + i_2]  ·  A[i_1, j_1, α]  ·  B[α, i_2, j_2]

Contract order (greedy):

    (1) T1[t, i_1, α, j_2] = Σ_{i_2} X[t, i_1 · m_2 + i_2] · B[α, i_2, j_2]
        cost: T · m_1 · n_2 · m_2 · D
    (2) Y[t, j_1, j_2] = Σ_{i_1, α} T1[t, i_1, α, j_2] · A[i_1, j_1, α]
        cost: T · m_1 · n_1 · n_2 · D

Total: `T · D · (m_1 · m_2 · n_2 + m_1 · n_1 · n_2)`
     = `T · D · n_2 · (m + m_1 · n_1)`
     ≈ `T · D · (m + n)^{3/2}` for balanced factoring.

Compare dense `T · m · n`.  Speed ratio: `(m · n) / (D · (m + n)^{3/2})`.
At m = n = 2048, D = 16: `4.2M / (16 · 2048^{1.5})` = `4.2M / 1.48M` = **2.8×** faster.

### Backward

Chain rule through the two contractions yields three small GEMMs per
weight (one per factor tensor + two for activations).  Each remains
O(T · D · √(m·n)), so the backward has the same asymptotic cost as
the forward.

### Retraction / optimizer

Because A and B are unconstrained, no Riemannian retraction is needed.
Standard Adam on (A, B) works; moment storage is proportional to
|A| + |B| = 2 D √(m n) — far below dense m·n.

Composing with OVFG: accumulate factored gradients directly into A, B
via their respective contractions with X.

### Composition with Stiefel × Σ (shift #7)

Option 1 (independent): Stiefel on one set of matrices, MPO on another.
Disjoint, easy to wire in per-layer.

Option 2 (nested): Stiefel weights already factor as `W = U Σ V^T`
where U ∈ St(m, r), V ∈ St(n, r).  Replace U with an MPO of shape
[m, r], V with an MPO of shape [n, r].  Because r is small (e.g., 512),
the MPO's bond dim D can be ≤ r without approximation loss.  Storage:
each of U, V costs O(D · √(m · r)) vs O(m · r).  At m = 2048, r = 512,
D = 16: 16 · √(2048 · 512) ≈ 16 · 1024 = 16 K per factor, vs m · r =
1 M.  **64× extra compression on top of Stiefel**.

### Composition with CHIRON (shift #1)

CHIRON's reversible flow is weight-agnostic — it operates on (q, p)
phase-space states, invariant to how the weights are stored.  MPOT
weights participate in the flow equations unchanged; the only change
is the GEMM primitive inside each block.

## Theoretical analysis — open questions

1. **Approximation error vs D**.  For which classes of weight matrices
   is bond D = 16 sufficient?  Empirically, trained transformer
   weights have rapidly-decaying singular spectra (the basis of Stiefel);
   the MPO-bond spectrum is harder to characterize analytically.

2. **Training convergence**.  Standard Adam should converge on (A, B),
   but the effective loss landscape is unknown.  Preconditioned
   Adam (Shampoo-style) may be needed.

3. **Conditioning**.  The nonlinear composition W = A · B introduces
   saddle points in (A, B) space.  Initialization from SVD of a
   pre-trained W_dense bootstraps near a good basin.

4. **Scaling to k > 2**.  At k = 3 or k = 4 factors, storage goes
   as k · D² · (m · n)^{1/k}, asymptotically better.  But contraction
   cost grows and numerical stability requires gauge-fixing.

## Minimal prototype (≤ 3 weeks)

1. **Week 1**: new CUDA module `gpu_mpot.{h,cu}` with two-site (k=2)
   MPO primitives:
   - `mpot_forward(X, A, B, m_1, m_2, n_1, n_2, D, Y)` via two
     chained sgemm_rowmajor.
   - `mpot_backward(X, dY, A, B, …, dX, dA, dB)` via three GEMMs.
   - `mpot_init_from_dense(W_dense, A, B)` via iterative SVD
     (truncate to bond D).

2. **Week 1**: unit tests
   - Round-trip: init_from_dense → forward → recompute W → parity
     with W_dense (max_err ≤ 2 · σ_{D+1}(W)).
   - Adam descent: 50-step toy regression, compare loss trajectory
     vs dense Adam at same lr.

3. **Week 2**: composition
   - Stiefel × MPOT: replace U, V in Stiefel by MPOT tensors;
     verify forward parity with dense Stiefel; train toy example.
   - Comparison benchmark: VRAM + ms/step at d = 2048, r = 512,
     D ∈ {8, 16, 32}.

4. **Week 3**: trainer wire-in
   - Gate behind `--mpot` / `--mpot-bond D` flags in chiron_main.cpp.
   - Integrate into the existing Stiefel Wo path.
   - Run pile_large smoke test at 300 steps; measure tok/s + loss
     trajectory.

## Composability table (all shifts compound multiplicatively)

Weight storage for a m = n = 2048 MLP at d_model = 2048, with and
without each shift:

| shifts                                | size  | mul  |
|---------------------------------------|-----:|-----:|
| Dense FP32                            | 16 MB | 1.0× |
| + BF16 weights (#5)                   | 8 MB  | 2×   |
| + Stiefel ρ=0.25 (#7)                 | 2 MB  | 4×   |
| + MPOT D=16 (#10, nested in Stiefel)  | 0.03 MB | **500×** |

**Projected ceiling at 500× weight compression + unchanged other axes**:
a 10 B-parameter MLP trunk fits in ~500 MB weight storage.  The rest
of the stack stays at its current ~2 GB, giving headroom for 30+ B
parameters on 16 GB — matching the paradigm-shift brief.

## Failure modes and mitigations

1. **MPO bond saturation**: transformer weights may require D ≫ 16
   to preserve quality.  *Mitigation*: per-layer adaptive D, start
   high (D = 128) and anneal down as training progresses.

2. **Initialization collapse**: random init of (A, B) produces W with
   unbalanced spectrum.  *Mitigation*: SVD-init from a small
   pre-trained model (distilled bootstrap) before paradigm-shift
   training.

3. **Backward instability**: chain-rule through two factors gives
   O(‖A‖ · ‖B‖) gradient amplification.  *Mitigation*: gauge-fix
   after each Adam step to enforce ‖A‖_F = ‖B‖_F = √‖W‖_F per
   factor pair.

4. **Kernel engineering cost**: chained tensor contractions in CUDA
   are complex.  *Mitigation*: start with two cuBLAS SGEMMs (the
   two-factor k=2 case); defer multi-factor k ≥ 3 to a later phase
   once k=2 is validated.

5. **Quality regression at scale**: toy problems may succeed while
   2 B training diverges.  *Mitigation*: validate on a 100M-parameter
   proxy before pile_large.

## Tracked as task #XX (to be created when implementation begins).
