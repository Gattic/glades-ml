# Paradigm Shift #44 Candidate A — MELT (MEmory-Lattice Tensor-train factorization)

**Status:** candidate-A design, single formulation. Companion to two parallel #44 candidates (B/C, separate axes).
**Date:** 2026-05-08.
**Axis:** sub-quadratic CHIRON FFN via *weight-tensor* decomposition.
**Tagline:** *Replace each FFN weight matrix `W ∈ ℝ^{m × dFFN}` with a tensor-train (TT) of rank ρ. Compute drops 3.2× per matvec, weight memory drops 205×, the symplectic shear is preserved unchanged.*
**Materially distinct from:** CSP (#27, sketches the FFN hidden activation, not the weights), SPAREC (#35, sparsifies σ'(x) on backward), Stiefel × Σ (#7, low-rank attention QKV decomposition — different tensor, different geometry), SAS (#40, attacks per-step probability not per-step F). MELT attacks the **forward FFN weight tensor** directly via a multilinear factorization. Compressed object is `W_in, W_out`; saved object is dense activation `x` and gradient `g`.

---

## 0. Executive summary

After 43 paradigm shifts CHIRON's per-step compute has been compressed ≈65× at T=1024. #42 (SCFA) closed attention; #43 (ORION) collapses step count. The remaining uncompressed bucket is **FFN forward compute** — now ~25% of per-step budget. At flagship `(m=2048, dFFN=8m=8192, L=53)` the FFN GEMM is `2·T·m·dFFN ≈ 34 GFLOP / layer / forward`.

MELT factors each `W ∈ ℝ^{m × dFFN}` as a tensor train (TT). For `d=2` (MPO factorization) with `m = m_1 m_2 = 64 · 32`, `dFFN = n_1 n_2 = 128 · 64`, two cores
$$
G_1 \in \mathbb{R}^{1 \times m_1 \times n_1 \times \rho}, \qquad G_2 \in \mathbb{R}^{\rho \times m_2 \times n_2 \times 1}
$$
encode `W[i_1 i_2, j_1 j_2] = Σ_α G_1[1, i_1, j_1, α] · G_2[α, i_2, j_2, 1]`. Total params: `(m_1 n_1 + m_2 n_2)·ρ = 10240·ρ`.

| ρ | Params W | vs dense (16.78M) | FLOPs / matvec | Speedup |
|---|---|---|---|---|
| 4 | 41 k | 410× | 2.62 M | 6.4× |
| 8 | 82 k | 205× | 5.24 M | 3.2× |
| 16 | 164 k | 102× | 10.49 M | 1.6× |
| 32 | 328 k | 51× | 20.97 M | regression |

**Headline (d=2, ρ=8): 3.2× FFN compute, 205× FFN weight-memory compression.** Per-step wall-clock impact via `1/(0.75 + 0.25/3.2) = 1.22×`; cumulative with #42+#43 reaches ≈ 80×.

The dramatic win is **memory**. Flagship FFN weights are `53·2·16.78M·2 B = 3.55 GB` BF16; MELT at ρ=8 cuts these to **17.4 MB**, freeing 3.55 GB on the 16 GB ceiling. This unlocks scaling **from 1.84B to ≈ 18B parameters** on the same single GPU.

The MLP shear `p ↦ p + W_out · σ(W_in · q + b_in) + b_out` is a function `Y(q)` inside `(q,p) ↦ (q, p + Y(q))`. Theorem 3 of #42 says **reversibility is structural for any continuous Y**. So TT replacement of `W_in, W_out` is **algebraically invisible** at the symplectic level. ✓

**Critical empirical risk.** MELT's premise is `r_eff(W_FFN) ≤ 16`. Literature is mixed: LoRA fine-tuning succeeds at ρ ≤ 8 only as adaptive *delta*; from-scratch transformer FFN may have `r_eff ∈ [50, 500]`. §10 specifies a 10-GPU-minute Gate-0 SVD probe. If `r_eff > 32`, headline collapses; framework still valid as a memory-only win at ρ=32.

---

## 1. Primitive objects

Fix one MLP shear, layer index suppressed. Let

- `m` — token-state dim (CHIRON's `q,p ∈ ℝ^{T×m}`), e.g. `m=2048`
- `dFFN` — FFN hidden dim, e.g. `dFFN = 4m = 8192`
- `T` — sequence length, e.g. `1024`
- `d` — TT order (number of cores), `d ∈ {2, 3, 4}`. Default `d=2` (matrix-product / MPO).
- `ρ` — TT-rank cap, `ρ ∈ {4, 8, 16}`. Boundary ranks `ρ_0 = ρ_d = 1`, internal `ρ_k = ρ`.
- `m = m_1 m_2 ... m_d`, `dFFN = n_1 n_2 ... n_d` — chosen tensor-product factorization.

Default factorization for `m=2048, dFFN=8192, d=2`: `m_1 = 64, m_2 = 32, n_1 = 128, n_2 = 64`. (These satisfy `m_1 n_1 + m_2 n_2 = 8192 + 2048 = 10240`, the per-ρ parameter cost.)

| symbol | shape | meaning |
|---|---|---|
| `G_k` | `ℝ^{ρ_{k-1} × m_k × n_k × ρ_k}` | k-th TT core, k=1..d |
| `W_TT` | `ℝ^{m × dFFN}` | dense matrix encoded by the TT, *never materialized* |
| `b_in, b_out` | `ℝ^{dFFN}, ℝ^m` | biases, kept dense |
| `x` | `ℝ^{dFFN}` (or `ℝ^{T×dFFN}`) | post-activation FFN hidden state |
| `r_k` | sequence of intermediate tensors | TT-matvec contraction state, see §3 |

**Two TT objects per shear**: `W_in` (`m × dFFN`) and `W_out` (`dFFN × m`). For symmetry we factor both with the same `(d, ρ, m_k, n_k)` shape. Total per-MLP-shear params: `2 · 10240·ρ = 20480·ρ`.

**Invariant (no new optimizer state).** TT cores are stored in BF16. Adam state for each core uses standard `(m, v)` BF16/INT8 buffers (per existing CHIRON budget). Composition with FACE/MFIO is via *core-by-core* application: each `G_k` is a 4-tensor, FACE compresses its Adam state on the `(m_k, n_k)` flattened axes if `min(m_k, n_k) ≥ 32`. Crucially the gradient flow into `G_k` (§4.2) is itself a low-rank object — Adam state at the core level is already `ρ × m_k n_k`-shaped which is small enough that compression is optional.

---

## 2. State space — TT manifold `\mathcal{T}_ρ`

The TT-rank-bounded matrices form a smooth (but not affine) manifold:
$$
\mathcal{T}_ρ \;:=\; \bigl\{W \in \mathbb{R}^{m \times dFFN} \;:\; \mathrm{TT-rank}(W) \le ρ \bigr\}.
$$
For `d=2` this is exactly the rank-`≤ρ` manifold of `m_1 m_2 × n_1 n_2` matrices when reshaped as `(m_1 n_1) × (m_2 n_2)` — a non-trivial reshape that mixes "row" and "column" indices.

**Dimension.** Counting parameters minus the gauge (each internal rank `ρ_k` carries `ρ_k²` redundant DOF from the gauge `G_k → G_k R, G_{k+1} → R^{-1} G_{k+1}`):
$$
\dim \mathcal{T}_ρ \;=\; \sum_{k=1}^{d} ρ_{k-1} \, m_k \, n_k \, ρ_k \;-\; \sum_{k=1}^{d-1} ρ_k^2.
$$
For `d=2, ρ_1=ρ`: `dim = (m_1 n_1 + m_2 n_2)·ρ - ρ² = 10240·ρ - ρ²`. At `ρ=8`: `dim = 81920 - 64 = 81856` independent DOFs.

**Tangent space at `W = G_1 ◦ G_2`.** A perturbation `δW = δG_1 ◦ G_2 + G_1 ◦ δG_2` lives in `T_W \mathcal{T}_ρ`. The tangent space has dimension `dim \mathcal{T}_ρ`; the gauge constraint is `V_1^⊤ δG_1^{(L)} = 0` where `G_1^{(L)} := \mathrm{reshape}(G_1, ρ_0 m_1 n_1, ρ_1)` is the left-unfolding (this fixes the gauge by left-orthogonalizing).

**Stiefel structure on cores.** Pin gauge by left-orthogonalizing each non-final core: for `k = 1, ..., d-1`,
$$
G_k^{(L)} := \mathrm{reshape}(G_k, [\rho_{k-1} m_k n_k, \rho_k]) \in \mathrm{Stiefel}(\rho_{k-1} m_k n_k, \rho_k).
$$
This kills the `O(ρ_k²)` gauge ambiguity. The final core `G_d` carries the magnitude. A periodic sweep (every `M_gauge = 1000` steps, see §4.5) enforces `G_k^{(L)\top} G_k^{(L)} = I_{ρ_k}` via thin QR.

---

## 3. Forward law — TT-matvec contraction

Given input `x ∈ ℝ^{dFFN}` (or batched `X ∈ ℝ^{T × dFFN}`), compute `y = W_TT · x` in `d` reduction steps.

### 3.1 Index reshape

View `x` as a `d`-mode tensor:
$$
x \;\equiv\; x[j_1, j_2, \ldots, j_d] \in \mathbb{R}^{n_1 \times n_2 \times \cdots \times n_d}, \qquad j = j_1 + n_1 j_2 + n_1 n_2 j_3 + \cdots
$$
(little-endian for the outermost axis to match standard reshape semantics). Similarly `y` will be reshaped as `y[i_1, ..., i_d]`.

### 3.2 Sequential contraction (d=2 spelled out)

```
(1)  z[i_1, j_2, α] = Σ_{j_1} G_1[1, i_1, j_1, α] · x[j_1, j_2]
                       — shape [m_1, n_2, ρ]   — cost m_1 · n_1 · n_2 · ρ
                                                  = 64 · 128 · 64 · ρ
                                                  = 524288·ρ FLOPs

(2)  y[i_1, i_2]    = Σ_{j_2, α} G_2[α, i_2, j_2, 1] · z[i_1, j_2, α]
                       — shape [m_1, m_2]      — cost m_1 · m_2 · n_2 · ρ
                                                  = 64 · 32 · 64 · ρ
                                                  = 131072·ρ FLOPs
```

**Total: `655360·ρ` FLOPs per matvec.** At `ρ=8`: `5.24M` FLOPs vs dense `16.78M` → **3.2×**. At `T=1024`: `5.4 GFLOPs` vs dense `17.2 GFLOPs`.

### 3.3 Batched form

For a batch `X ∈ ℝ^{T × dFFN}`, treat the batch axis as a co-axis:
- `Z[t, i_1, j_2, α] = Σ_{j_1} G_1[1, i_1, j_1, α] · X[t, j_1, j_2]` — one batched GEMM `(T n_2) × n_1 → (T n_2) × m_1 ρ`.
- `Y[t, i_1, i_2] = Σ_{j_2, α} G_2[α, i_2, j_2, 1] · Z[t, i_1, j_2, α]` — one batched GEMM `(T m_1) × (n_2 ρ) → (T m_1) × m_2`.

Both reductions map cleanly to cuBLAS sgemm/bgemm. **No new kernels needed** — only correct stride bookkeeping. CSP (#27) already provides the batched-reshape infrastructure.

### 3.4 General `d`

For `d ≥ 3`, contract left-to-right; cost of step `k` is `(m_1 ... m_k) · n_k · (n_{k+1} ... n_d) · ρ_{k-1} · ρ_k`. Total: `C_{TT}(d,ρ) = Σ_k (∏_{i≤k} m_i)(∏_{i≥k} n_i) ρ_{k-1} ρ_k`. At `d=3, m=8·16·16, dFFN=8·16·64, ρ=8`: ≈ `2.2M` FLOPs/matvec ≈ 7.5× over dense, but contraction loses batched-GEMM unit-stride structure. **Default for #44: `d=2`, trading modest extra compute for clean implementation.**

---

## 4. Backward and optimizer

### 4.1 TT-tangent gradient

Forward: `y = G_1 ◦ G_2 · x`. Given `dy`, chain rule on the contraction graph yields
$$
dG_1[1, i_1, j_1, α] = \sum_{i_2, j_2} dy[i_1, i_2] · G_2[α, i_2, j_2, 1] · x[j_1, j_2],
$$
$$
dG_2[α, i_2, j_2, 1] = \sum_{i_1, j_1} dy[i_1, i_2] · G_1[1, i_1, j_1, α] · x[j_1, j_2],
$$
$$
dx[j_1, j_2] = \sum_{i_1, i_2, α} dy[i_1, i_2] · G_1[1, i_1, j_1, α] · G_2[α, i_2, j_2, 1].
$$

Three TT-contractions, each ≈ same cost as forward. **Backward ≈ 3· forward = `1.97M·ρ`** FLOPs/matvec. At ρ=8: 15.7M backward vs dense 33.6M → **2.1× backward**. Combined fwd+bwd MELT speedup: `(16.78 + 33.6) / (5.24 + 15.7) = 2.4×`.

### 4.2 Adam state on TT cores

Each `G_k` carries Adam `(m_k, v_k)` of matching shape. Per MLP shear (BF16): `2·2·10240·ρ·2 B = 81920·ρ` bytes; at ρ=8: `0.66 MB/shear` vs dense `134 MB/shear` (200× savings, matching param compression). Across L=53 × 2 shears: dense `14.2 GB` Adam → MELT `70 MB`. Combined with weight savings, MELT frees ≈ **17.7 GB** of FFN-related storage on the optimizer side (modulo INT8/CPU-offload Adam tactics that already trim a fraction of this).

### 4.3 Gauge maintenance (left-orthogonalization sweep)

Every `M_gauge = 1000` steps, sweep left-to-right:

```
for k = 1, ..., d-1:
    G_k_L = reshape(G_k, [ρ_{k-1} m_k n_k, ρ_k])
    Q, R = qr(G_k_L)
    G_k = reshape(Q, [ρ_{k-1}, m_k, n_k, ρ_k])
    G_{k+1} = einsum('αi..., αβ -> βi...', G_{k+1}, R)
    rotate Adam (m_k, v_k) into new gauge via ρ×ρ block solve
```

Cost ≈ `O(ρ³ · max_k m_k n_k) ≈ 8³·8192 = 4.2M` FLOPs/sweep. At every 1000 steps: ≈ 4200 FLOPs/step — negligible.

### 4.4 Initialization

Standard FFN init draws `W_in ~ N(0, 2/m)`, `W_out ~ N(0, 2/dFFN)`. For TT, draw each core IID Gaussian then run one left-orthogonalization sweep:

```
G_k ~ N(0, σ_k²)  with  σ_k² = (2/m)^{1/d} · (1/ρ_k)^{(some balance)}
```

A clean choice: draw each core as a uniform random `Stiefel(ρ_{k-1} m_k n_k, ρ_k)` then scale the final core `G_d` by `sqrt(2/m_d) / sqrt(ρ_{d-1})` to set the operator norm. This preserves the variance preservation of Glorot/Kaiming inits at the TT level. Tested at d=2 in TT-init literature (Novikov 2015, Tjandra 2018).

### 4.5 Progressive rank growth

Train at low ρ initially, grow over first 50% of training, freeze. Rank-ρ cores extend to rank-(ρ+1) by appending a Stiefel-side column `~ N(0, σ_grow²)`. This is smooth growth on the manifold (`\mathcal{T}_ρ ⊂ \mathcal{T}_{ρ+1}`). Schedule: ρ=4 at step 0, ρ=8 at 0.25N, ρ=16 at 0.5N, frozen thereafter. Composes naturally with #38 SLC (T-ramp) and #39 RLG (L-insertion) — different axes (sequence, depth, FFN-rank).

---

## 5. Theorems

### Theorem 1 (TT compression).
Let `W ∈ ℝ^{m × dFFN}` admit a `d`-core TT decomposition with internal ranks `ρ_1, ..., ρ_{d-1}`. Then the parameter count is
$$
P_{TT} \;=\; \sum_{k=1}^{d} \rho_{k-1} \cdot m_k \cdot n_k \cdot \rho_k.
$$
For `d=2` with `ρ_1 = ρ`: `P_{TT} = (m_1 n_1 + m_2 n_2) · ρ`. The compression ratio versus dense storage `m · dFFN` is
$$
\frac{m \cdot dFFN}{P_{TT}} \;=\; \frac{m_1 m_2 \cdot n_1 n_2}{(m_1 n_1 + m_2 n_2) \rho}.
$$

**Optimal balanced factorization.** For fixed `m, dFFN, d=2`, minimize `m_1 n_1 + m_2 n_2` subject to `m_1 m_2 = m, n_1 n_2 = dFFN`. By AM-GM:
$$
m_1 n_1 + m_2 n_2 \;\ge\; 2 \sqrt{m_1 m_2 \cdot n_1 n_2} \;=\; 2 \sqrt{m \cdot dFFN}
$$
with equality iff `m_1 n_1 = m_2 n_2`. At equality, `P_{TT}^{*} = 2 \sqrt{m·dFFN} · ρ` and compression is `\sqrt{m·dFFN} / (2ρ)`.

For `m=2048, dFFN=8192`: optimal `m_1 n_1 = m_2 n_2 = \sqrt{2048·8192} = 4096`, giving `P_{TT}^* = 8192·ρ` and **compression `= 4096 / (2ρ) = 2048/ρ`**. At ρ=8: compression `= 256×` (vs 205× achieved at our `(64,32,128,64)` factorization, which is slightly suboptimal because `(m_1, m_2) = (64, 32)` is constrained to be a power-of-2 divisor of `m`).

**Proof.** Direct count of free parameters across the `d` cores. Equality at AM-GM bound is standard. □

### Theorem 2 (TT compute).
Forward `y = W · x` via TT contraction has cost
$$
C_{TT} \;=\; \sum_{k=1}^{d} \Bigl(\prod_{i \le k} m_i\Bigr) n_k \Bigl(\prod_{i > k} n_i\Bigr) \rho_{k-1} \rho_k.
$$
For `d=2, m_1=m_2=√m, n_1=n_2=√dFFN, ρ_1=ρ`:
$$
C_{TT}^{(d=2)} \;=\; \sqrt{m} \cdot \sqrt{dFFN} \cdot \sqrt{dFFN} \cdot ρ \;+\; \sqrt{m} \cdot \sqrt{m} \cdot \sqrt{dFFN} \cdot ρ \;=\; \sqrt{m·dFFN}·(\sqrt{dFFN} + \sqrt{m})·ρ.
$$
For `m=2048, dFFN=8192`: `C_{TT}^{(d=2)} = 4096 · (90.5 + 45.3) · ρ ≈ 556·ρ·1000` FLOPs ≈ `0.56M·ρ`. At ρ=8: `4.48M` per matvec.

The dense baseline is `m · dFFN = 2048 · 8192 ≈ 16.78M`. Speedup at ρ=8: `16.78 / 4.48 ≈ 3.7×` (slightly higher than our default `(64,32,128,64)` factorization yields, because optimal `(m_1, m_2)` and `(n_1, n_2)` are √-balanced).

**Practical factorization.** We use `(m_1, m_2) = (64, 32)` (closest-power-of-2 to `√2048 ≈ 45`) and `(n_1, n_2) = (128, 64)` (closest-power-of-2 to `√8192 ≈ 90.5`). This gives the **3.2×** compute speedup quoted in the executive summary.

**Proof.** Direct accounting of the contraction costs. Each step's cost is the product of all open indices times the contracted index. □

### Theorem 3 (CHIRON reversibility preservation under TT replacement).

Let `Y_TT(q) := W_out_TT · σ(W_in_TT · q + b_in) + b_out` where `W_in_TT, W_out_TT` are TT-rank-bounded matrices and `σ` is a continuous pointwise nonlinearity. Then the symplectic shear
$$
\Phi(q, p) \;:=\; (q,\; p + Y_{TT}(q))
$$
is a unit lower-triangular bijection on `ℝ^{T m} × ℝ^{T m}`, with inverse
$$
\Phi^{-1}(q', p') \;=\; (q',\; p' - Y_{TT}(q')).
$$

**Proof.** A TT-rank-bounded matrix is, in particular, a continuous linear map, so `W_TT · q` is continuous in q. The composition with continuous `σ` and addition of constants `b_in, b_out` keeps `Y_TT` continuous. The map `(q,p) ↦ (q, p + Y_{TT}(q))` therefore satisfies the hypothesis of CHIRON Theorem (paradigm #1, restated as Theorem 3 of #42): any continuous `Y` produces an involutive shear. The inverse is given by the explicit formula above; one verifies `Φ^{-1} ∘ Φ = id` by direct computation. □

**Corollary (Jacobian determinant).** `det dΦ/d(q,p) = 1`. The flow is volume-preserving and incompressible. ✓ This is unchanged from baseline CHIRON; TT replacement does not perturb the symplectic structure.

---

## 6. Conjectures (with falsifiability)

### Conjecture C1 — TT-rank sufficiency at LLM scale.

**Statement.** For trained CHIRON 1.84B (paradigm #38+#39 flagship), the FFN weight matrices have effective rank `r_eff ≤ 16` per layer, where
$$
r_{eff}(W) \;:=\; \min \bigl\{ k : \sigma_k(W) / \sigma_1(W) < 0.01 \bigr\}.
$$

**Stronger form (C1+).** Cumulative singular-value energy at rank `ρ=8` exceeds 95%: `\frac{\sum_{k=1}^{8} \sigma_k^2}{\sum_{k} \sigma_k^2} \ge 0.95`.

**Falsifiability.** SVD of FFN weights at any 1.84B checkpoint. **Cost: ~10 GPU-min (§10).** If `r_eff ∈ [16, 32]`, MELT viable at `ρ=16` with reduced 1.6× compute headline. If `r_eff > 32`, MELT compute-axis dies; memory axis still viable.

**Risk assessment.** Literature on attention rank suggests `r_attn ≈ 50–256`. FFN is harder to characterize; some studies suggest higher rank, others lower. LoRA (ρ ≤ 8) succeeds for *fine-tuning* but as additive delta to a fully-trained dense base — this is *not* evidence that the trained dense base itself is low-rank. The honest prior: P(C1+ at ρ=8) ≈ 0.3, P(C1+ at ρ=16) ≈ 0.6.

### Conjecture C2 — Training convergence parity (with rank growth).

**Statement.** With the progressive-ρ schedule of §4.5 (ρ: 4→8→16 over the first 50% of training), MELT achieves within 0.1 nat of dense-baseline final loss at 5000 steps on 66M pile-bpe.

**Falsifiability.** Run flagship `--mfio 2 --wip-K 4 --face 1 --t-schedule auto --rlg auto` with and without `--melt 1 --melt-rho-schedule "4@0,8@1250,16@2500"` for 5000 steps on 66M, compare final EMA-loss at step 5000.

**Risk.** Progressive growth from `ρ=4` to `ρ=16` is a dynamical-systems claim about loss-landscape navigation: starting in a low-dim manifold (`\mathcal{T}_4`) and growing into `\mathcal{T}_{16}` may trap the optimizer in a basin that's not present in the full-rank space. Mitigating evidence: this is the standard story for matrix completion / nuclear-norm relaxation, where rank-growth optimization is well-behaved. Counter-evidence: from-scratch transformer training is not a convex problem, and basin selection is sensitive to early dynamics.

---

## 7. Composition with prior paradigms

| Paradigm | Object compressed | Compose? | Notes |
|---|---|---|---|
| #7 Stiefel × Σ | QKV weight | ✓ | Different blocks; multiplicative |
| #13 TRCD | per-token depth | ✓ | TRCD picks layers; MELT cheapens each |
| #27 CSP | FFN hidden `x` | ⚠ partial | CSP sketches σ(W_in q); TT op runs over k_csp not dFFN. Re-derive `n_k` factorization once CSP's `k_csp` is fixed. Default: keep CSP off until shipped |
| #28 FACE | embed-Adam state | ✓ | Different tensors; multiplicative on memory |
| #35 SPAREC | σ' on backward | ✓ | SPAREC operates inside dG_k; trivially compositional |
| #38 SLC | sequence T | ✓ | Pure-orthogonal axes |
| #39 RLG | layer count L | ✓ | New RLG blocks init at current ρ |
| #40 SAS | layer-skip prob | ✓ | Multiplicative |
| #42 SCFA | attention seq | ✓ | Different layer block; multiplicative |
| #43 ORION | step count | ✓ | Different axis; multiplicative |

**Verdict.** MELT is orthogonal to all 10 shipped/proposed shifts except CSP (#27, partial overlap). Multiplicative compute with #42, #43; multiplicative memory with #28. Headline composition: `3.2× · 8.6× (ORION) · 2.27× (SCFA) ≈ 62×` atop the shipped `7.6×`, giving paradigm-#44 cumulative `≈ 470× / baseline`. Even conservatively (ρ=16 → 1.6× MELT, half-ORION) the cumulative still ≥ 100×.

---

## 8. CUDA / kernel primitives

MELT requires three new primitives, all expressible as cuBLAS calls + reshapes:

### 8.1 `tt_matvec_d2` (forward)

```
tt_matvec_d2(G1: [m1, n1, ρ], G2: [ρ, m2, n2], X: [T, n1, n2])
  → Y: [T, m1, m2]

  // step 1: Z = einsum('a c d, t c b -> t d a b', G1, X)  reshape-friendly
  Z_flat = sgemm(reshape(G1, m1·ρ, n1), reshape(X.T, n1, T·n2))
  Z = reshape(Z_flat.T, [T, n2, m1, ρ])

  // step 2: Y = einsum('e a d, t b a d -> t b e', G2, Z)
  Y_flat = sgemm(reshape(G2, m2, n2·ρ), reshape(Z.permute(0,2,1,3), T·m1, n2·ρ).T)
  Y = reshape(Y_flat.T, [T, m1, m2])
```

Two batched-strided sgemms. Stride bookkeeping is the main complexity; **no new CUDA kernel**.

### 8.2 `tt_matvec_d2_grad` (backward)

```
tt_matvec_d2_grad(G1, G2, X, dY)
  → dG1, dG2, dX

  // dG2: contract dY against G1 ◦ X
  Z = same as forward step 1
  dG2 = einsum('t b e, t b a d -> e a d', dY, Z) — one sgemm

  // dG1: contract dY against G2 against X
  // dG1[a, c, d] = Σ_{t, b, e} dY[t, b, e] · G2[e, a, d] · X[t, c, b]
  T1 = einsum('e a d, t b e -> t b a d', G2, dY)  — one sgemm
  dG1 = einsum('t b a d, t c b -> a c d', T1, X)  — one sgemm

  // dX: contract dY against G1 against G2
  T2 = einsum('e a d, t b e -> t b a d', G2, dY)  — same as T1, reuse
  dX = einsum('a c d, t b a d -> t c b', G1, T2)  — one sgemm
```

Four sgemms total for backward (one shared). Costs `≈ 3.0× forward`, matching theoretical estimate.

### 8.3 `tt_left_orthogonalize_sweep`

```
for k in 1..d-1:
    G_k_L = G_k.reshape(ρ_{k-1}·m_k·n_k, ρ_k)
    Q, R = qr(G_k_L)  // tall-thin QR on small matrix
    G_k = Q.reshape(ρ_{k-1}, m_k, n_k, ρ_k)
    G_{k+1} = einsum('αi..., αβ -> βi...', G_{k+1}, R)  // sgemm
    rotate_adam_state_in_gauge(m_k, v_k, R)
```

Cost is dominated by QR of a `(ρ_{k-1} m_k n_k) × ρ_k` matrix; for `d=2, ρ=8`: QR of `64·128 × 8 = 8192 × 8` matrix, easily ≤ 1 ms. Run every `M_gauge = 1000` steps.

### 8.4 Drop-in API

Within `Backend/Machine Learning/Networks/cuda/gpu_kernels.h`, add a `MeltFFN` struct holding `(G1, G2, b_in, b_out)` per-shear. Replace existing FFN sgemm dispatch with a `melt_enabled` flag check. **Lines of code estimate: ≈ 600 LOC** (forward, backward, init, save/load, gauge sweep, progressive growth hook).

---

## 9. Honest gaps

1. **Rank sufficiency is the single binary risk.** If C1+ fails (FFN at LLM scale needs ρ > 32), MELT compute headline drops from 3.2× to 1.6× or worse. Addressed by Gate-0 (§10).

2. **Factorization `(m_k, n_k)` is heuristic.** `(64, 32, 128, 64)` for power-of-2 alignment; AM-GM-optimum is `(45.25, 45.25, 90.5, 90.5)` (non-integer). The ~20% sub-optimality (256× → 205× compression) is acceptable.

3. **Rank-growth schedule is empirical.** ρ: 4→8→16 over first 50% of training is a tunable hyperparameter; interacts with FACE/SLC/RLG schedules. C2 falsifiability covers this empirically.

4. **No principled MoE comparison.** MELT is single-GPU-friendly (no expert sharding); MoE is a different mechanism. We do not claim superiority — just orthogonality to CHIRON's symplectic structure.

5. **Activation memory not addressed.** MELT compresses *weights*, not `x ∈ ℝ^{T × dFFN}`. CSP (#27) is the activation-axis tool; §7 notes partial overlap.

6. **Inference (T=1) matvec.** Two cuBLAS calls × ~1.2 µs ≈ 2.4 µs per FFN per token at ρ=8, kernel-launch dominated. KV-cache unaffected (FFN doesn't see KV).

7. **Save/load format.** Extend checkpoint format: MLP shear stores either `{W_in, W_out}` (dense legacy) or `{G_1_in, G_2_in, G_1_out, G_2_out, ρ, factorization}` (TT) with versioned header. Dense↔TT round-trip via `tt_compress` (one TT-SVD) and `tt_decompress`.

8. **BF16 numerical conditioning.** Operator norm bound `‖W_TT‖_op ≤ ‖G_d‖_op` after left-orthogonalizing earlier cores. Risk is BF16 underflow in `dG_k` at small ρ when gradients concentrate; Kahan-v (surprise #17) is the standard mitigation.

---

## 10. Gate-0 probe — FFN-weight SVD (≤ 10 GPU-min)

**Goal.** Falsify or anchor C1+ before implementation.

**Procedure.**
1. Load existing CHIRON 1.84B flagship checkpoint (or 66M for cheaper probe).
2. For each MLP shear (L=53 layers × 2 shears = 106 matrices), compute SVD of `W_in` and `W_out`. (At m=2048, dFFN=8192, each SVD is < 1 sec on GPU.)
3. Report cumulative-energy ratio at ρ ∈ {4, 8, 16, 32, 64} per layer:
   $$
   E_ρ(W) := \frac{\sum_{k=1}^{ρ} \sigma_k^2}{\sum_{k} \sigma_k^2}.
   $$
4. Pass criteria:
   - **C1+ at ρ=8 holds**: median `E_8 ≥ 0.95` across all 106 matrices ⇒ MELT viable at headline ρ=8.
   - **C1 at ρ=16 holds**: median `E_{16} ≥ 0.95` ⇒ MELT viable at ρ=16 with reduced 1.6× compute speedup; memory axis still strong.
   - **C1 fails at ρ=32**: median `E_{32} < 0.95` ⇒ MELT compute axis dies; reconsider.

**Cost.** SVD of 106 matrices, each at most `2048 × 8192`, on RTX 3090: ≈ 8 GPU-min total. Add 2 min for checkpoint load + reporting. **Total: ~10 GPU-min.**

**Decision.** If pass at ρ=8: schedule iter 187 implementation (next sprint). If pass at ρ=16: implement at ρ=16 (still 1.6× compute, 100× memory); revisit ρ=8 with progressive-growth schedule. If fail: archive MELT, deliver paradigm-44 candidate B/C.

**Auxiliary Gate-0.5 probe.** TT-SVD reconstruction error: do an actual TT-SVD on a 1.84B FFN matrix and compute `‖W − W_TT‖_F / ‖W‖_F` at ρ=4, 8, 16. This validates not just the SVD spectrum (which is a 2D rank statement) but the actual TT-rank-2D structure (which is a 4D-tensor rank statement). Slightly more expensive (≈ 30 min including TT-SVD library setup) but more direct. Run if Gate-0 (SVD) is borderline (E_8 ∈ [0.85, 0.95]).

---

## 11. Implementation roadmap

Iter 187: Gate-0 SVD probe (§10), ~1 day. Iter 188–190: implement `tt_matvec_d2`, backward, gauge sweep, init, save/load with unit tests vs dense reference, ~3 days. Iter 191: 66M flagship + `--melt 1 --melt-rho 8` for 5000 steps, ~6 GPU-hr. Iter 192: 1.84B flagship + MELT, ≥100k steps. Iter 193: cross-composition with #43 (if shipped). If Gate-0 fails: archive, redirect to candidates B/C of #44.

---

## 12. Summary card

| Property | Value | Notes |
|---|---|---|
| Compute speedup (FFN forward) | **3.2×** at d=2, ρ=8 | Theorem 2; 6.4× at ρ=4, 1.6× at ρ=16 |
| Compute speedup (FFN backward) | **2.1×** at d=2, ρ=8 | §4.1 |
| Memory compression (weights) | **205×** at d=2, ρ=8 | Theorem 1 |
| Memory savings (1.84B FFN total) | **3.55 GB → 17.4 MB** | 53 layers × 2 shears |
| Adam state savings (FFN) | **14.2 GB → 70 MB** | At BF16; see §4.2 |
| Reversibility preservation | **structural ✓** | Theorem 3 |
| New optimizer state | **none** (just on TT cores) | §4.2 |
| Composition with #42 SCFA | **multiplicative ✓** | §7 |
| Composition with #43 ORION | **multiplicative ✓** | §7 |
| Composition with #28 FACE | **multiplicative on memory ✓** | §7 |
| Single binary risk | C1+ rank sufficiency | §6, §10 |
| Gate-0 probe | FFN-weight SVD | ~10 GPU-min |
| Implementation LOC estimate | ~600 LOC | §8.4 |
| Falsifiable empirical claim | C2: within 0.1 nat at 5000 steps 66M | §6 |

---

## 13. Closing remark

MELT applies a known mathematical object (tensor train) to a known target (transformer FFN weights). The non-trivial new content is the **symplectic-shear preservation argument (Theorem 3)** and the **composition-orthogonality matrix vs CHIRON's existing 9 shipped paradigms (§7)**. Theorem 3 is what makes TT a CHIRON-specific paradigm-shift rather than a generic transformer optimization: in vanilla transformers, FFN replacement perturbs residual-stream behavior, but in CHIRON's reversible-flow architecture the symplectic shear is structurally invariant under continuous changes to `Y(q)`.

The single empirical risk is **rank sufficiency** (C1+). Gate-0 settles this in 10 GPU-min before any implementation lift. If `r_eff ≤ 16`, MELT delivers a `1.6×–3.2×` per-step FFN speedup multiplicative atop the shipped 7.6× and the proposed 8.6× (#43), with a `205×` weight-memory free-floor that unlocks 18B-parameter models on the 16 GB single-GPU ceiling. If `r_eff > 32`, the compute headline collapses but `~50×` weight memory at ρ=32 remains, and the parameter-ceiling-extension argument (1.84B → ≥ 9B at unchanged compute) survives.

**Recommendation: green-light Gate-0 probe (iter 187). Decision on full implementation at probe completion.**
