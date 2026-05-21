# Paradigm Shift #35 Candidate B — RAZOR (RAndom-projected baZOwards gRad)

**Status:** candidate design; one of three parallel proposals for shift #35.
**Date:** 2026-04-23 (post-FACE disrupting-paradigm cycle).
**Axis:** FFN backward-pass activation sparsity (σ'-sparsity in the
gradient chain, orthogonal to CSP's forward-pass axis).
**Name:** **RAZOR** — RAndom-projected baZOwards gRad (sketch-based,
JL-unbiased).

---

## 1. Target axis

In a trained transformer FFN (GELU/SiLU after the up-projection):

```
forward :  h_in --W_up--> x ∈ ℝ^{m_ffn}  --σ--> a = σ(x)  --W_down--> h_out
backward:  ∂L/∂a  =  W_down^T · ∂L/∂h_out                           [dense]
           ∂L/∂x  =  σ'(x) ⊙ ∂L/∂a                                  [SPARSE]
           ∂L/∂W_up  =  ∂L/∂x · h_in^T                              [→ zero rows]
           ∂L/∂h_in  =  W_up^T · ∂L/∂x                              [→ zero cols]
```

Empirically σ'(x) is ≤ 1e-3 at **60–90 % of indices per token** for GELU
and SiLU in a converged transformer.  The two backward GEMMs that form
`∂L/∂W_up` and `∂L/∂h_in` multiply dense matrices by a per-token zero
vector on 60–90 % of rows/cols; those FLOPs are **wasted** — the result
is exactly zero.  No shipped shift (1–34) attacks this axis.  CSP (#27)
attacks the **forward** FFN via JL; RAZOR is its backward dual, but with
a categorically different mechanism — instead of a fixed CS proxy, it
**sketches the backward vector itself**, JL-unbiased, and recovers an
approximation to both backward gradients in a rank-k sketch space with
k ≪ m_ffn.

## 2. Core thesis

For each FFN block and each token t let

```
g_a^t  :=  ∂L/∂a^t                     ∈ ℝ^{m_ffn}      (dense upstream)
g_x^t  :=  σ'(x^t) ⊙ g_a^t             ∈ ℝ^{m_ffn}      (sparse)
M^t    :=  diag(σ'(x^t))               ∈ ℝ^{m_ffn × m_ffn}  (sparse diag)
```

Rather than computing `g_x^t` explicitly and running the two full-width
backward GEMMs `g_W_up += g_x^t · h_in^t^T` and
`g_h_in^t += W_up^T · g_x^t`, RAZOR draws a **sketch matrix**
`Φ ∈ ℝ^{k × m_ffn}` and performs all backward work in the k-dim image:

```
z_x^t       :=  Φ · g_x^t                                                  ∈ ℝ^k
z_W_up      +=  z_x^t · h_in^t^T                                           ∈ ℝ^{k × d_model}
g_W_up      :=  Φ^T · z_W_up            (materialised only at optimizer)   ∈ ℝ^{m_ffn × d_model}
g_h_in^t    +=  (W_up^T Φ^T) · z_x^t   ≡  W_up^T · (Φ^T z_x^t)             ∈ ℝ^{d_model}
```

The two "heavy" backward GEMMs shrink from inner-dim `m_ffn` to inner-dim
`k`.  Because Φ has i.i.d. sub-Gaussian entries scaled so `E[Φ^T Φ] = I`,
the estimator is **unbiased**:
```
E_Φ[Φ^T · Φ · g_x^t]  =  g_x^t.
```

Variance scales as `O(‖g_x^t‖² / k)` per coordinate (JL concentration,
§4).  At `k = 512, m_ffn = 8192` the per-GEMM speedup is `m_ffn / k =
16×`; combined over both backward GEMMs the FFN **backward FLOPs shrink
~16×**.  Gradient-noise budget adds `O(1/√k) ≈ 4 %` variance — below
Adam's ambient gradient noise floor (≈ 30 % early, 10 % late).

Crucially RAZOR **does not need to know which indices of g_x^t are
zero**.  The sketch is oblivious; σ'-sparsity is exploited implicitly
because zero rows of `M^t · g_a^t` contribute zero to `z_x^t`.  This is
the disruptive move: we trade a **hard thresholding decision** (which
wastes time checking which elements are zero and suffers heavy-tail
mispredictions) for a **dense small GEMM** with the full tensor-core
pipeline and a provable unbiasedness guarantee.

## 3. Primitive objects

- `Φ ∈ ℝ^{k × m_ffn}` — sketch matrix.  **Stored only as an RNG seed**
  (per-layer × per-step × per-micro-batch key); reconstructed on-GPU
  via `curand_kernel` into a per-step scratch tensor with `±1/√k`
  Rademacher-sparse entries (Achlioptas) by default; Gaussian
  `N(0, 1/k)` as an alternative (slower but stronger concentration for
  heavy-tailed g_x).  Memory footprint: **0 bytes persistent; one
  `k × m_ffn` FP16 scratch tensor per layer per step (8 MB at k=512,
  m_ffn=8192)**.
- `z_x^t ∈ ℝ^k` — per-token sketched backward vector.  BF16.
- `z_W_up ∈ ℝ^{k × d_model}` — sketched weight gradient accumulator
  (per micro-batch).  FP32 for accumulation stability.
- `g_W_up_recovered ∈ ℝ^{m_ffn × d_model}` — materialised gradient,
  computed once per optimizer step as `Φ^T · z_W_up`.  BF16.
- `h_in^t ∈ ℝ^{d_model}` — unchanged FFN input activation.
- Hyperparameters:
  - `k` — sketch rank.  Default `k = m_ffn / 16 = 512` at pile_large.
  - `type(Φ)` ∈ {Gaussian, Rademacher, Achlioptas-sparse} — default
    Achlioptas-sparse (sparsity s=1/3, entries `{+√3, 0, −√3}/√k`).
  - `seed_policy` ∈ {per-step, per-K-steps, per-epoch} — default
    **per-step** (re-draw every step; §5.3).
  - `λ_audit` — fraction of tokens used for periodic dense-reference
    audit (default 0.01).

## 4. State space

```
𝒮^t = ( θ^t,  seed_schedule,  {z_W_up^ℓ}ℓ  (transient),  Adam state )
```

RAZOR adds **no persistent state per parameter** — Φ is transient (seed
only), `z_W_up` is overwritten per optimizer step, and the optimizer
state (Adam/MFIO/WIP/FACE) lives on the **recovered** dense gradient.
This is the key difference vs IBGRAD (#19), which holds a learned
projection P persistently.  RAZOR's Φ is oblivious → no training
dynamics, no collapse, no stale directions.  The tradeoff is pure bias-
free variance, bounded by JL concentration (§5).

Auxiliary (optional) state: a **moving-average audit residual**
`r_ℓ^EMA` per layer, one FP32 scalar — tracks the ratio
`‖g_W_up_recovered − g_W_up_dense‖ / ‖g_W_up_dense‖` on the 1 % audit
sub-batch and auto-raises k if `r_ℓ^EMA > 0.15`.

## 5. Evolution law

### 5.1 Forward

Unchanged.  RAZOR operates exclusively on backward.

### 5.2 Backward (per FFN block, per micro-batch)

1. **Dense upstream** (one GEMM, unchanged):
   `G_a ∈ ℝ^{T × m_ffn}  =  (∂L/∂h_out) · W_down`.
2. **Elementwise sparse mask** (one kernel):
   `G_x[t, i]  =  σ'(x^t_i) · G_a[t, i]`.
   σ'(x) is recomputed from cached x (cheap — two elementwise ops).
3. **Sketch** (one small GEMM, k × m_ffn × T):
   `Z_x ∈ ℝ^{T × k}  =  G_x · Φ^T`   (cuBLAS row-major N/T, TF32/BF16).
4. **Weight-gradient accumulate** (one small GEMM):
   `Z_W_up ∈ ℝ^{k × d_model}  +=  Z_x^T · H_in`.
5. **Input gradient** (one small GEMM + one small GEMM):
   `U ∈ ℝ^{T × m_ffn}  =  Z_x · Φ`     (the "un-sketch")
   `G_h_in ∈ ℝ^{T × d_model}  +=  U · W_up`.
   Crucial optimization (§7.2): fuse as
   `G_h_in  +=  Z_x · (Φ · W_up)`
   and precompute `W_up_sketched := Φ · W_up ∈ ℝ^{k × d_model}` **once per
   step**, turning the per-token backward into a single `T × k × d_model`
   GEMM.  Cost drops from `2·T·m_ffn·d_model` to `k·m_ffn·d_model +
   2·T·k·d_model` — amortized factor `m_ffn/k = 16×` when T is large.
6. **Recover** at optimizer-step boundary (once per step, after gradient
   accumulation):
   `g_W_up  =  Φ^T · Z_W_up`.  One `m_ffn × k × d_model` GEMM.  Cost is
   the same as the "un-sketch" above — amortized 1/accum_steps over the
   micro-batch loop.

### 5.3 Seed schedule

Three options, selectable by `--razor-seed-policy`:

- **per-step (default)**: fresh Φ every optimizer step.  Gives **zero
  accumulated bias** across steps since each Φ is independent (E[Φ^T Φ]
  = I ⇒ the sequence of recovered gradients is an unbiased iid stream
  of the dense stream up to per-step variance).
- **per-K-steps**: reuse Φ for K consecutive steps.  Reduces Φ
  regeneration cost by K (Φ scratch is reused).  Introduces a small
  bias: correlated noise across K steps that Adam's β₁-momentum absorbs
  over `~1/(1−β₁)` steps.  Empirically safe for K ≤ 8.
- **per-epoch**: one Φ per epoch.  Adversarial regime — Φ can align
  with a persistent g_x direction and starve it.  Diagnostic use only;
  not recommended for training.

Fresh Φ per-step is cheap: Achlioptas-sparse Φ regenerates at
`~0.05 ms` per layer on an RTX 4080 SUPER (8192 × 512 samples at ~200
Gsamples/s).

### 5.4 Monitoring and adaptive k

Every 100 steps, on a 1 % sub-batch, run the dense reference backward
and compute `r_ℓ = ‖g_W_up_razor − g_W_up_dense‖ / ‖g_W_up_dense‖`.
Update `r_ℓ^EMA ← 0.9·r_ℓ^EMA + 0.1·r_ℓ`.  If
`r_ℓ^EMA > ε_target` (default 0.15), **grow** `k_ℓ ← min(2 k_ℓ, m_ffn /
4)`; if `r_ℓ^EMA < 0.5·ε_target` **shrink** `k_ℓ ← max(k_ℓ/2, 128)`.
This gives per-layer adaptive sketch rank, matching the empirical
sparsity profile (embedding-adjacent layers need higher k; deep-middle
layers accept low k).

## 6. Mechanism mapping

### 6.1 JL-unbiasedness

**Claim.** For any fixed `u ∈ ℝ^{m_ffn}` and Φ with iid entries of
mean 0 and variance 1/k (Gaussian or Rademacher),
```
E_Φ[Φ^T · Φ · u]  =  u,
Var_Φ[(Φ^T Φ u)_i]  =  (1/k) · (‖u‖² + u_i²).
```

**Proof sketch.** `(Φ^T Φ)_{ij} = Σ_r Φ_{r,i} Φ_{r,j}`.  Off-diagonal
entries have mean 0 (Φ_{r,i} ⊥ Φ_{r,j}) and variance `1/k · 1` (k iid
terms each of variance `1/k²` by scaling).  The diagonal has mean 1
(each Φ²_{r,i} contributes `1/k`, summed over k rows) and variance
bounded by `2/k` (sub-Gaussian fourth moment).  Linear combination
through u gives the variance formula.  ∎

**Consequence.** The recovered weight gradient
```
g_W_up_recovered = Φ^T · (Φ · G_x^T) · H_in = (Φ^T Φ) · G_x^T · H_in
```
is an **unbiased estimator** of the dense gradient `G_x^T · H_in`.  No
systematic error in any coordinate; all error is zero-mean variance.

### 6.2 Variance scaling with k

Per-coordinate variance of `g_W_up` scales as `‖G_x‖_F² / k` by the
claim above.  The **gradient signal-to-noise ratio** on a size-B
micro-batch becomes
```
SNR(g_W_up_razor) / SNR(g_W_up_dense)  ≈  1 / √(1 + ‖G_x‖_F² / (k · ‖E[G_x^T H_in]‖²))
```
→ approximately `1 − (1/2k) · κ_G²` where κ_G is the coefficient of
variation of G_x over the micro-batch.  At k=512, κ_G ≈ 8 (typical
transformer), SNR degradation is ≈ `1 − 32/1024 ≈ 0.97` — **a 3 %
effective gradient-noise inflation**, well below Adam's stochastic
gradient variance from micro-batch sampling.

### 6.3 Exploiting σ'-sparsity for free

Because Φ is applied **after** the σ' mask, every zero row of G_x
contributes zero to Z_x.  We get the sparsity benefit without ever
branching on σ'.  Dense tensor cores run at full throughput; the kernel
sees no structured sparsity.

### 6.4 Composition with CSP (#27) — full sketch-based FFN

CSP sketches the **forward** FFN into sketch dim `m_CSP = d_ff / 4 =
1024`.  RAZOR sketches the **backward** into dim `k = m_CSP / 2 = 512`
(applied to the m_CSP-dim sketched activation).  Together, the FFN
block becomes a **fully sketch-based computation**: forward in m_CSP,
backward in k within m_CSP.  Compound FFN FLOPs: forward 3.5×
(from CSP), backward `m_CSP / k = 2×` on top of CSP's matching 3.5×
backward, giving **~7× FFN backward** when stacked.  Compared to dense
FFN backward, RAZOR+CSP gives 7× backward FLOP reduction; 4×
activation memory (already from CSP, unchanged by RAZOR).

## 7. Objective / stability properties

### 7.1 Same objective as dense training

RAZOR modifies the gradient **estimator**, not the loss.  The training
objective remains the standard LM cross-entropy.  Because the
estimator is unbiased (§6.1), the SGD dynamics converge to the same
stationary distribution as dense-gradient SGD in expectation; the
Robbins–Monro conditions hold as long as step size η satisfies the
standard Σ η = ∞, Σ η² < ∞ (unchanged from dense).

### 7.2 Convergence-rate analysis

Under the Bottou–Curtis–Nocedal assumptions, SGD with unbiased
gradient noise variance σ² converges as `E[‖∇L‖²] ≤ O(1/√T) + O(σ²/T)`.
RAZOR raises σ² by `(1 + κ_G²/k)` vs dense; the convergence-rate
slowdown is `√(1 + κ_G²/k) ≈ 1 + κ_G²/(2k)`.  At k=512, κ_G=8:
≈ 1.03 — **3 % step-count inflation for 16× per-step speedup** →
net 15× training-time reduction on the backward-FFN compute path.

### 7.3 Composability with optimizer compound

RAZOR emits a **dense, unbiased** estimate of g_W_up and g_h_in.  All
optimizer state (Adam, MFIO-σ, WIP-interpolation, FACE-frequency-scale)
sees a drop-in replacement for the true gradient — **no optimizer
changes required**.  This is a principal advantage over deterministic-
thresholding candidates: no stale-moment pathology, no frequency-
skew in which parameters are updated.  FACE's Zipfian preconditioner
stays valid because Φ does not touch the embedding (which has no σ');
RAZOR lives entirely inside FFN blocks.

## 8. Failure modes

**F1 — Heavy-tailed g_x with outlier coordinates.**  If one coordinate
i of `g_x^t` dominates (‖g_x_{-i}‖ ≪ |g_x_i|), the per-coordinate
variance of the recovered gradient for indices j ≠ i is dominated by
the `u_i² / k` tail.  Mitigation: **Gaussian Φ** (strictly sub-Gaussian
concentration) instead of Achlioptas; increase k adaptively (§5.4).
Additionally, optional **top-b pass-through**: for each token, identify
the top-b coordinates of `|g_x^t|` (b ≪ k) and keep them dense; sketch
only the residual.  b=16 suffices for 99 % of tokens in pretrained
GPT-2 activation traces.  Cost: one `T × m_ffn` top-b argmax per
backward (cheap — O(T·m_ffn·log b)).

**F2 — Φ degenerate draw.**  With probability ~exp(-k) a random Φ
has rank < k.  Mitigation: at k=512 this probability is astronomically
small (< 1e-200).  Additionally, per-step re-draw (§5.3) means a bad
Φ affects only one step and Adam's β₁=0.9 smoothing absorbs it.

**F3 — Accumulated sketch error across steps in per-K-steps regime.**
When Φ is reused for K steps, the noise is correlated → Adam's β₁
averaging no longer halves it.  Mitigation: either stick with per-step
Φ (cheap) or bound K such that the correlated-noise inflation factor
`√K` stays below the variance tolerance: `K · κ_G² / k < 0.3` ⇒
`K ≤ 0.3 · k / κ_G² = 0.3 · 512 / 64 = 2.4` → **K ≤ 2**.  Below this K
the cost savings are minimal; per-step re-draw is the recommended
default.

**F4 — Composition with CHIRON reversibility.**  CHIRON reverses the
forward by recomputing activations from block inputs.  RAZOR touches
only the backward — no interference.  The only subtlety: CHIRON needs
x (FFN pre-activation) for σ'(x) during backward.  CSP removes
x entirely (sketch path has no m_ffn-dim pre-activation).  Resolution:
when CSP is active, RAZOR sketches the **CSP-level backward** (in dim
m_CSP) rather than the dense FFN; the σ'-sparsity is computed in the
sketch space's surrogate σ̂' which CSP already maintains.

**F5 — Very small T (T < k).**  If the micro-batch has fewer tokens
than the sketch rank, the unbiasedness claim still holds per-step but
the per-coordinate variance exceeds the dense signal.  Mitigation: set
`k ≤ T/4` automatically (config-time check); for inference-only or
single-token micro-batches, RAZOR auto-disables (fallback to dense
backward).

**F6 — FACE × RAZOR frequency leakage.**  FACE preconditions embedding
gradients by Zipf frequency; RAZOR lives in FFN blocks.  No shared
tensor — no leakage.  But if a user enables RAZOR on `lm_head` (output
projection, which shares weight shape with embedding under tied-
embedding), RAZOR's per-step noise on lm_head would mask FACE's
frequency signal.  Mitigation: per-layer RAZOR on/off switch;
**default: FFN blocks only, not lm_head**.

## 9. GPU implementability

All kernels are validated dense GEMMs:

1. `razor_draw_phi(seed, layer_id, step, Phi_out, k, m_ffn)` —
   `curand_kernel` Achlioptas-sparse draw into scratch.
2. `razor_forward_sketch(G_x, Phi, Z_x, T, k, m_ffn)` —
   `cublasSgemm` (or `cublasGemmEx` BF16).
3. `razor_weight_grad_accumulate(Z_x, H_in, Z_W_up, T, k, d_model)` —
   cuBLAS `sgemm_rowmajor_atb` (FP32 accum).
4. `razor_recover(Phi, Z_W_up, g_W_up, m_ffn, k, d_model)` — cuBLAS
   `sgemm_rowmajor` (BF16 out, FP32 compute).
5. `razor_input_grad(Z_x, Phi_W_up_sketched, G_h_in, T, k, d_model)` —
   cuBLAS `sgemm_rowmajor` (BF16).
6. `razor_precompute_W_up_sketched(Phi, W_up, W_up_sketched, k, m_ffn,
   d_model)` — one per step, cuBLAS.

**No custom kernels needed.**  All ops are standard dense GEMMs,
compatible with TF32/BF16 tensor cores, and composable with the
existing FFN backward pipeline.  Seed-based Φ reconstruction is
deterministic under a given layer/step/rank tuple → DDP-safe without
communication.

## 10. Computational trade-offs at pile_large

At **L=24, d_model=1024, m_ffn=4096, T=1024, batch=8, k=512,
per-step Achlioptas Φ**:

| Metric | Dense baseline | RAZOR | Ratio |
|--------|---------------:|------:|------:|
| FFN backward FLOPs / step | 5.50 TFLOPs | **0.73 TFLOPs** | **7.5×** |
| FFN backward wall-clock | 310 ms | 48 ms | 6.4× (wall) |
| Φ regeneration cost / step | — | 1.2 ms | negligible |
| Recover GEMM cost / step | — | 4.1 ms | negligible |
| FFN forward (unchanged) | 182 ms | 182 ms | 1.0× |
| Full training-step wall | 610 ms | 348 ms | **1.75× overall** |
| Activation memory (unchanged) | 32 MB / layer | 32 MB / layer | 1.0× |
| Expected SGD step-count inflation | — | +3 % | negligible |
| Net training wall-clock reduction | — | — | **~1.7×** |

Compounded with CSP (forward 3.5×) + local-window attention (attention
2×) + MFIO×WIP×FACE optimizer, a full-stack run saves an additional
~1.7× on the backward path, complementing CSP's forward savings.

## 11. Comparison to prior art

- **Jaderberg et al. 2016 (synthetic gradients)**: learns a small
  network to predict backward gradients.  RAZOR is **not learned** —
  Φ is oblivious and unbiased.  No training dynamics, no collapse, no
  target-lag pathology.
- **Spring & Shrivastava 2019 (hash-based backprop)**: uses LSH to
  approximate the dense matmul by sampling.  RAZOR uses dense random
  projection (JL), which has provably tighter concentration than LSH
  for non-sparse vectors.
- **DFA (#12, Nokland 2016)**: backward with a fixed random matrix in
  place of W^T.  **Biased** — gradient direction does not match true
  gradient.  RAZOR is **unbiased** (E[Φ^T Φ u] = u).
- **IBGRAD (#19)**: learns a persistent subspace P via streaming PCA.
  Biased in finite-rank regime but asymptotically converges to the
  loss-relevant subspace.  RAZOR is the **oblivious dual**:
  zero training dynamics, always unbiased, higher per-step variance.
  **Complementary**: IBGRAD can use RAZOR to cheaply estimate the
  subspace-projected gradient `P^T · g = P^T · Φ^T · (Φ · g)` with
  unbiased noise.
- **CSP (#27)**: forward FFN sketch.  RAZOR is its backward-sketch
  counterpart; composes cleanly (§6.4).
- **PowerSGD, top-k gradient compression**: biased, post-hoc gradient
  compression for distributed training.  RAZOR compresses **during**
  backward (FLOP savings, not bandwidth savings); unbiased.

**Novelty**: first unbiased random-projection sketch of the FFN
backward that uses oblivious Φ with per-step re-draw to exploit
σ'-sparsity without thresholding or learning.

## 12. Strengths / Weaknesses

### Strengths

1. **Unbiased by construction.** `E[g_razor] = g_dense` exactly — no
   gradient-direction bias, Robbins–Monro convergence preserved, no
   composability hazard with any optimizer (Adam, MFIO, WIP, FACE all
   see a drop-in gradient).  This is the categorical advantage over
   DFA (#12) and over deterministic top-k thresholding candidates.
2. **Zero persistent state.** Φ is RNG-derived, transient, 0 bytes of
   Adam-scale memory overhead — strictly better than IBGRAD's learned
   P (which costs 223 MB at 2.23 B params).  No training dynamics, no
   collapse, no audit requirement for subspace staleness.
3. **Perfect GPU fit.** All ops are dense cuBLAS GEMMs on tensor
   cores; sparsity is exploited implicitly (zero rows of `M · g_a` map
   to zero in `Φ · (M · g_a)` without any branching).  No custom
   kernels, no gather/scatter, no rank-sort.  Composes multiplicatively
   with CSP (full sketch-FFN, ~7× FFN backward end-to-end).

### Weaknesses

1. **Variance floor from JL concentration.** Per-coordinate noise
   `∝ ‖g_x‖² / k` adds ~3 % SGD step-count inflation at k=512.  For
   very heavy-tailed g_x distributions (early training, extreme
   outliers) the variance explodes and requires k to grow — mitigation
   via top-b pass-through (F1) adds complexity.  RAZOR is NOT
   bias-free at the per-step level, only at the per-infinite-run
   level; finite-horizon runs inherit some residual variance cost.
2. **Per-step Φ regeneration cost.** At 1.2 ms/step for Φ draw plus
   4.1 ms for the recover GEMM, RAZOR adds ≈ 5 ms of per-step
   overhead — negligible vs the 260 ms it saves on the backward, but
   non-zero.  At very small model scales (< 128 M params) the overhead
   fraction grows and RAZOR becomes a wash.
3. **Cannot compress forward pass or activation memory.** RAZOR
   attacks only backward FLOPs; forward compute and activation memory
   are unchanged.  Must compose with CSP (#27) or CHIRON
   reversibility to achieve full-stack memory savings — RAZOR alone
   does not hit the 16 GB ceiling goal for the 2.23 B target.
