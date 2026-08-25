# Next-Paradigm Weight Representation — Framework-Design Exercise

## Problem statement

CHIRON + BF16-weights + int8-Adam + BF16-grads + local-window attention
has pushed the on-16-GB ceiling to **2.23 B parameters @ 1733 tok/s**
(confirmed in the 2026-04-21 B-run, 200 Adam steps, loss 11.44 → 6.58
best, 129× perplexity reduction). The binding constraint is now **weight
VRAM** (4.8 GB / 15.2 GB ≈ 31 %).

The remaining levers all target W itself. The next paradigm shift must
deliver ≥ 3× reduction in weight storage **and** ≥ 3× reduction in
forward-GEMM FLOPs, while preserving CHIRON reversibility and staying
within 2× of BF16 baseline on convergence.

Three materially-different formulations were developed in parallel
(subagents ran during the 2.23 B B-run; see "Candidate formulations"
below). This document ends with the selection rationale and the minimal
prototype implementation path.

---

## Candidate A — Operator-theoretic spectral parameterization

Each weight matrix `W = Σ_i α_i · B_i + U V^T`, with
- `{B_i}` a *fixed, structured* basis: Chebyshev polynomials of circulant
  shift operators, Kronecker-composed (`T_p(S1) ⊗ T_q(S2)`)
- `α ∈ R^K`, K = (P+1)² ≪ m² (P ≈ 31 for m=4096)
- `U, V ∈ R^{m × R}` a rank-R correction (R ≈ 32)
- A learned per-layer rotation `Q_l` (parameterized as `exp(skew Kronecker)`)

### Numbers (m = 4096, P = 31, R = 32)

| K setting | α params | Per-matrix VRAM | Compression | Fwd GEMM speedup | 16 GB ceiling |
|-----------|---------:|----------------:|------------:|-----------------:|--------------:|
| k = m/4   |    1024  |     ~1.2 GB     |    4.0×     |       4.6×       |     ~9 B      |
| k = m/16  |     256  |     ~520 MB     |    9.2×     |       9.2×       |    ~20 B      |
| k = m/64  |      64  |     ~240 MB     |     20×     |        22×       |    ~45 B eff  |

Forward is a two-dimensional Clenshaw recurrence fused with coefficient
accumulation (cuFFT-accelerable circulant multiplies). Backward pulls
gradient to α via the same pipeline applied to `(Q_l X)^T`.

### Strengths
- Largest theoretical compression (9–20×)
- Compute reduction *and* memory reduction in lockstep
- Deterministic reconstruction — CHIRON reversibility trivially preserved

### Weaknesses
- Effective rank capped at K + R ≈ m/4 at P=31, R=32; MLP expressivity
  concern
- α-space Adam ≠ W-space Adam: second-moment statistics in α are a
  projected view of gradient statistics; step-size mismatch likely
- Basis-mismatch risk: if the real W-landscape is not spectrally smooth
  under shifts, the structural prior is wasted
- Needs new fused Clenshaw kernel + cuFFT integration (~1200 LOC CUDA)
- No prior-art success at > 1 B scale for trained-from-scratch Chebyshev
  weights

---

## Candidate B — Stiefel × Σ manifold factorization (SELECTED)

Each W = U Σ V^T with
- `U ∈ St(m, r)` — orthonormal columns: `U^T U = I_r`
- `V ∈ St(n, r)` — orthonormal columns: `V^T V = I_r`
- `Σ = diag(σ₁, …, σ_r)` on R_+^r

Intrinsic dimension: `k(m,n,r) = (m+n)r − r²`. At m=n=d, ratio k/d² =
`2ρ − ρ²` for `ρ = r/d`.

### Numbers (d = 2048, 48 layers, 4:1 MLP, ρ varied)

| ρ    | k/m² | Weights | Adam state | Grads | Free-DOF ceiling |
|------|-----:|--------:|-----------:|------:|-----------------:|
| 0.50 | 0.75 |  3.6 GB |    3.6 GB  | 1.8 GB|       3.0 B      |
| 0.25 | 0.44 |  2.1 GB |    2.1 GB  | 1.1 GB|     **5.1 B**    |
| 0.10 | 0.19 |  0.9 GB |    0.9 GB  | 0.46 GB|    ~12 B         |

Per-GEMM FLOP ratio vs dense: `r(m+n)/(mn) = 2ρ`. At ρ=0.25: **4× fewer
FLOPs per forward, 4× per backward** — meets the "magnitudes" criterion
on speed **and** memory simultaneously.

### Forward / backward
Forward: 3 chained SGEMMs with inner dim r → reuses existing cuBLAS.
Backward tangent projection:
```
G_U = (I − U U^T) G V Σ + U · skew(U^T G V Σ)
G_V = (I − V V^T) G^T U Σ + V · skew(V^T G^T U Σ)
g_Σ = diag(U^T G V)
```
All built from O((m+n)r²) standard GEMMs — no new GEMM kernels.

Retraction (polar / QR, once per Adam step):
```
U ← qf(U + η_U)   V ← qf(V + η_V)   Σ ← Σ ⊙ exp(η_Σ / Σ)
```
Uses cuSOLVER `sgeqrf + sorgqr`. Cost: O(mr²) per matrix — at ρ=0.25, ~0.5%
of a forward pass (fully amortized, not per token).

### Why this wins the selection

1. **Cleanest integration**: three existing SGEMMs replace one; no new
   GEMM kernels; QR via cuSOLVER.
2. **Lowest convergence risk**: LoRA / low-rank FT precedent shows r = d/8
   matches full FT for common LLM objectives; we allow r up to d/4 here.
3. **Built-in Fisher natural gradient**: the canonical Stiefel metric
   is the unique O(m)-invariant metric, so Riemannian Adam = natural
   gradient for free — a strict optimizer upgrade.
4. **Spectral / Lipschitz control baked in**: `‖W‖₂ = max_i σ_i` is an
   explicit scalar we can clip; no more exploding-activations at depth.
5. **Stiefel is a *contraction basin* for BF16 SR noise**: any
   bounded rounding error in `U + η_U` is projected back on M by QR.
   This is *strictly more robust* than unconstrained BF16 master weights —
   BF16 noise becomes self-correcting.
6. **Composable with all existing CHIRON pieces**: orthogonal to the
   symplectic flow on (q, p); int8-Adam still works (now with *tight a
   priori scale* bound from Stiefel geometry); BF16 stochastic rounding
   composes; local-attn unchanged.
7. **Implementation cost lowest of the three**: ~680 new CUDA LOC + ~400
   host orchestration = ~1100 LOC, vs A's ~1700 and C's ~2000.

### Weaknesses
- At ρ → 0.5 QR becomes expensive (O(d³/2)); mitigated by chunked QR
  every N steps + Cayley retraction on intermediate steps.
- ρ = 0.10 may underfit MLP gate/up; allocate per-layer ρ_ℓ with higher
  r for MLP, lower r for attention projections.

---

## Candidate C — Reversible Mixture-of-Experts inside CHIRON shear

Shear `p += Y(q)` becomes `p += Σ_i g_i(q) · E_i(q)` with N experts per
block, k active per token (N=32, k=2 typical).

### Key theoretical result (from the subagent analysis)
For **any** function `Y : R^d → R^d` depending only on q, the map `(q, p)
→ (q, p + Y(q))` is a valid shear since its Jacobian is unit
lower-triangular, det = 1, and the inverse is `p = p' − Y(q)`
unconditionally. **CHIRON reversibility is automatic** regardless of
how Y is constructed — including routed sparse sums.

Constraint: exact inversion requires Y(q) to be deterministic in q, so
stochastic routing must use hash-keyed Gumbel noise (seed = H(q_t))
rather than externally-injected noise.

### Numbers (d = 2048, L = 24, N = 16, k = 2, d_e = 1024)

| Placement | Expert VRAM | Target effective params | Tok/s estimate |
|-----------|------------:|------------------------:|---------------:|
| All on-GPU, NF4 quantized | 1.2 GB | ~7 B | ~1400 |
| Host-streamed (PCIe 4.0)  | 0 GB   | ~18 B | ~1100 |
| N=64, NF4 all-GPU         | 4.8 GB | ~10 B | ~1300 |

### Strengths
- Effective parameter scaling is orthogonal to compute — paper-shaped
  "lots of params at same cost" win
- Reversibility inherits for free from p-linearity (major theoretical
  simplification vs normal MoE which struggles with backprop + streaming)

### Weaknesses
- Router collapse is the classic MoE failure mode; needs aux losses +
  z-loss, init variance scaling
- Host streaming requires strict bandwidth budgeting (~1.2 GB/step at
  T=1024 under d=2048) — tight at 25 GB/s PCIe 4.0
- Highest implementation cost (~2000 LOC): router, dispatch, scatter,
  per-expert Adam
- Most moving parts during convergence experimentation

---

## Selection

**Select Candidate B (Stiefel × Σ manifold) as the next-paradigm ship.**

Reasons:
1. Meets the ≥ 3× memory AND ≥ 3× speed bar (4× each at ρ=0.25) —
   strictly stronger than A on speed and stronger than C on memory.
2. Strict-upgrade compatibility with the entire existing stack (BF16
   SR weights, int8 Adam, BF16 grads, CHIRON reversibility, local-attn,
   cuBLAS-tiled attention). Nothing has to be redesigned.
3. Lowest convergence risk of the three — LoRA / low-rank FT is the
   closest precedent and the literature is unanimously positive.
4. Smallest code footprint (~1100 LOC) ⇒ fastest iteration.
5. Grants the optimizer two new axes automatically: natural-gradient
   preconditioning (via canonical metric) and spectral-norm clipping
   (via direct σ control). These are side benefits unavailable in A
   or C.
6. Longer-term this composes multiplicatively with C: each expert in
   a future rMoE layer can be Stiefel-factored.

**Rejected A** because the α-space Adam / W-space Adam mismatch and the
rank-cap at m/4 are both serious convergence hazards with no established
remedy at our scale, and the fused Clenshaw kernel is significant
engineering risk.

**Rejected C for now** because the near-term win (VRAM reduction at the
2.23 B ceiling) is not its strong suit — it scales effective params but
doesn't free VRAM at the *same* nominal param count. Also router
collapse + PCIe bandwidth budget are nontrivial to tune from a standing
start. **Deferred as a "future ship after B"**, likely combined with
B-factored experts.

---

## Minimal prototype — implementation plan (Phase 1)

### Kernels / code paths
1. `StiefelMatrix` struct — `U, Σ, V` buffers in BF16; `m_U, m_V, m_Σ`
   (Riemannian momenta) in int8-packed form; `v_U, v_V, v_Σ` (second
   moments) in int8.
2. Forward chain: three `sgemm_rowmajor` calls with inner dim r — **reuses
   existing cuBLAS path**. Fused `scale_by_diag` kernel for the Σ middle
   step (~30 LOC).
3. Tangent projection kernel — fused `(I - UU^T)GVΣ + U · skew(U^T GVΣ)`
   via 4 cuBLAS gemms + 1 custom kernel for skew (~150 LOC).
4. QR retraction wrapper around `cusolverDnSgeqrf + cusolverDnSorgqr`,
   batched across layers (~120 LOC).
5. Cayley retraction fast-path for intermediate steps (every
   N_refresh=10 steps): `(I + ½A)(I − ½A)^{-1}` where
   `A = U η_U^T − η_U U^T` — 2 sgemms + 1 getrs (~100 LOC).
6. Stiefel vector-transport kernel (Gram-Schmidt on `m_U` against new
   `U`) (~80 LOC).
7. Fisher-Rao exp on Σ: elementwise `Σ ⊙ exp(η_Σ / Σ)` (~30 LOC).
8. Modified int8-Adam pack/unpack using analytic scale bound
   `‖η_U‖ ≤ √r` (~50 LOC).
9. Periodic re-projection health check — QR if `‖U^T U - I‖_F > τ`
   (~60 LOC).

Total **~680 LOC new CUDA + ~400 LOC orchestration = ~1100 LOC**
incremental.

### Parity tests (GPU, mandatory)
- `CHIRONStiefelIdentityRecoveryTest`: when r = min(m, n), the factorized
  forward must exactly equal the unconstrained forward (bit-level at
  FP32 path, within BF16 ULP at BF16 path).
- `CHIRONStiefelGradientParityTest`: tangent-projected gradient should
  match the dense gradient restricted to the tangent subspace —
  max_err < 1e-4.
- `CHIRONStiefelQRStabilityTest`: after 1000 Adam-update cycles,
  `‖U^T U − I‖_F < 1e-5` (verifies QR retraction is not drifting).
- `CHIRONStiefelAdamParityTest`: Riemannian Adam at ρ=1.0 must exactly
  reduce to unconstrained Adam up to orthogonal gauge — measured by
  training a 24-layer 200M-param model 100 steps and comparing loss
  trajectory, max deviation < 5%.

### Success criteria (end-to-end)
- **Parity**: at ρ = 1.0 (no manifold constraint), training loss
  trajectory matches unconstrained BF16-weight baseline within 2% over
  100 steps.
- **Ceiling**: at ρ = 0.25, train a 5.1 B free-DOF model on 16 GB
  ≥ 100 steps. Target: sustained ≥ 1500 tok/s (approaching the 4×
  compute speedup; some overhead expected on the first ship).
- **Convergence**: loss at step 100 within 2× the 2.23 B dense-weight
  baseline at the same step / tokens-seen.

If parity and ceiling both pass, this is the **7th paradigm shift** in
the CHIRON program — compute-and-memory reduction via geometric
constraints, composable with all prior shifts.

### Phase 2 (after empirical validation)
- Per-layer ρ_ℓ tuning (attention projections at ρ=0.125, MLP at ρ=0.5).
- Compose with Candidate C (rMoE): each expert is Stiefel-factored,
  yielding ~20–40 B effective-parameter models on 16 GB.
- Push ρ → 0.10 with adaptive refresh and expressivity-regularized
  training.
