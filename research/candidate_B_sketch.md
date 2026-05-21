# Candidate B — SPECTRA: Sketched-Propagation Estimator for Compressed Transformer Activations

*(Sketch-native / randomized-numerical-linear-algebra training)*

---

## 1. Name and core thesis

**SPECTRA — Sketched Propagation via Embedded Compressed Tensor Representation of Activations.**

Activation memory `O(L·B·T·d)` is the largest residual cost in large-transformer training, and no optimizer-side trick (ATLAS, VESTA, HELIOS) touches it. SPECTRA changes *what is propagated*: each layer carries a **structured random sketch** `Ŝ = ΩX` of its hidden state, never the full activation; the backward pass reconstructs an **unbiased estimator** of `dW = Xᵀ dY` using the same sharing matrix Ω drawn in the forward pass from a seeded deterministic RNG. Concentration is controlled by Matrix-Bernstein and Johnson–Lindenstrauss inequalities with explicit constants. Softmax and pointwise nonlinearities are handled by a bounded-cost **patch-unsketch** that re-materialises only the rows carrying spectral mass, with a formally characterised *unsketching gap* `η(k)` that vanishes as `k → rank(X)`. The result is magnitudes-level reductions in activation memory (`d → k`, `T → m`) and proportional speedups in per-layer backward compute, at the cost of a variance inflation `O(d/k + T/m)` in the gradient estimator, which is mitigated by a linear control variate.

---

## 2. Primitive objects and state space

Fix a transformer of depth `L`, batch `B`, sequence `T`, model dim `d`, head dim `dₕ`, heads `H`, KV heads `H_kv` (GQA, with group size `g = H/H_kv`), MLP hidden `d_ff`. Let `V` denote vocabulary size.

**Full state (never materialised).** `X_ℓ ∈ ℝ^{(BT)×d}` is the input activation of layer ℓ in row-major "token as row" layout, reshaped from `ℝ^{B×T×d}`.

**Sketch dimensions.** Two independent reduced dimensions:
- `k` — feature sketch dim, typically `k ∈ {d/8, d/16, d/32}`.
- `m` — temporal sketch dim, typically `m ∈ {T/4, T/8}`.

**Sketch operator family.** Product structure `Ω_ℓ = Ω^(T)_ℓ ⊗ Ω^(d)_ℓ`:
- `Ω^(d)_ℓ ∈ ℝ^{k×d}` — **Subsampled Randomised Hadamard Transform (SRHT)**: `Ω^(d) = √(d/k) · P · H · D`, with `D` a diagonal of iid Rademacher signs, `H` the normalised Hadamard matrix, `P` uniform row-subsampling. Chosen because (i) `Ω x` is `O(d log d)` via FWHT, (ii) it is a `(1±ε)`-subspace embedding with sample complexity `k = O(d log(d/δ) / ε²)`, (iii) the signed-add-subtract structure is BF16-benign.
- `Ω^(T)_ℓ ∈ ℝ^{m×T}` — **CountSketch** (sparse `±1`, one nonzero per column). Chosen over SRHT along `T` because heavy-hitter structure (attention sinks) is captured by Charikar–Chen–Farach-Colton feature hashing, and applying it is `O(BT d)` — strictly linear.

Each `Ω_ℓ` is **regenerated from a global seed `σ` plus `(ℓ, step, layer_role)`**; never stored. This is the sole randomness source and satisfies the determinism constraint.

**Sketched activation state.** At layer ℓ we carry simultaneously
- `S_ℓ^{(d)} := X_ℓ (Ω^(d)_ℓ)ᵀ ∈ ℝ^{(BT)×k}` (feature-compressed), stored in BF16.
- `S_ℓ^{(T)} := Ω^(T)_ℓ X_ℓ ∈ ℝ^{(Bm)×d}` (token-compressed), stored in BF16, only when a softmax block demands it.
- A **tracer set** `R_ℓ ⊂ {1,…,BT}` of size `r ≪ BT`, containing the token indices that were "unsketched" (patch-reconstructed). `r = O(H · m)` typically.

**Sketched parameter view.** Weights `W` are *not* sketched — they sit in standard BF16 parameter memory. SPECTRA compresses activations, not parameters. This is the critical distinction from ATLAS/VESTA which sketch the optimiser state.

---

## 3. Forward sketched dynamics

Let `X = X_ℓ` for brevity. Let `A_ℓ`, `W_q`, `W_k`, `W_v`, `W_o`, `W_1`, `W_2` be the attention and MLP projection weights.

**Embedding.** Token IDs `t ∈ {1,…,V}^{B×T}` are embedded directly in sketched form. We keep a *pre-sketched embedding table* `E·(Ω^(d))ᵀ ∈ ℝ^{V×k}` whose rows are looked up; this costs `V·k` (≪ `V·d`) on the critical path. The full `E` is kept only for the language-model head — see §6.

**Sketched linear layer.** For linear `Y = X W`, `W ∈ ℝ^{d×d'}`:
- *Sketch-through*: `S_Y^{(d')} = X · W̃` where `W̃ := W (Ω^(d'))ᵀ ∈ ℝ^{d×k}` is precomputed after each optimiser step. This is a BF16 SGEMM `(BT)×d · d×k`, already `d/k`× cheaper than the baseline `(BT)×d · d×d'`.
- *Sketch-input*: when only `S_X^{(d)}` is available and we need `Y W`, we use `X ≈ S_X^{(d)} (Ω^(d))⁺` only when necessary (rare — softmax or patch points); normally we stay in sketch space.

**Sketched attention (GQA).** Within each head:
- Project `Q, K, V` normally per KV head (with `g` query heads sharing each KV head). Store `Q` in full, but store `K, V` in token-compressed form: `K̃ = Ω^(T) K ∈ ℝ^{m×d_h}`, `Ṽ = Ω^(T) V ∈ ℝ^{m×d_h}`.
- Scores: `Z = Q K̃ᵀ = Q Kᵀ (Ω^(T))ᵀ = A (Ω^(T))ᵀ ∈ ℝ^{T×m}`, true scores *right-sketched*.
- *Exact score reconstruction.* Since each CountSketch column for token `j` is a signed unit vector `s(j) e_{h(j)}`, we have `A_{i,j} = s(j) · Z_{i, h(j)}` **exactly**, computable in `O(T)` per query with `(s,h)` tables in constant memory. Softmax then runs on the exact `T`-dim score row — no precision loss on attention scores; the sketch only changes how `K` is stored.
- Output: `O_h = softmax(A_h) V`. We compute `O_i = Σ_j p_{i,j} · s(j) · Ṽ_{h(j)} + ε_{top}`, with collision error `𝔼‖ε‖² ≤ ‖V‖_F²/m` (§5).
- Causal mask: `K̃, Ṽ` are built incrementally — `Ṽ_{h(j)} ← Ṽ_{h(j)} + s(j) V_j` as tokens stream.

**Sketched MLP with SwiGLU.** `Y = (σ(X W_1^a) ⊙ X W_1^b) W_2`. Precompute `W̃_1^a = W_1^a (Ω^(d_ff))ᵀ` (similarly for `b`), reducing hidden width from `d_ff` to `k_ff` on the forward path. `σ` (SiLU) does not commute with the sketch, so we partially unsketch on a proxy `g_ff := S_X^{(d)} · W̃_1^a ∈ ℝ^{BT×k_ff}` — controlled bias per §6.

**Residual stream.** A *single* feature sketch `S_ℓ^{(d)}` threads the whole residual stream; residual addition commutes with `Ω^(d)`. LayerNorm is handled in §6.

---

## 4. Backward sketched gradient estimator

The central mathematical claim. For a linear layer `Y = X W`, the true weight gradient is
```
g := ∇_W L = Xᵀ · dY   ∈ ℝ^{d×d'}.
```

**Primary estimator (Ω-shared JL estimator).**
```
ĝ_Ω := Xᵀ (Ω^(T))ᵀ (Ω^(T)) dY = (Ω^(T) X)ᵀ (Ω^(T) dY).
```
Because `Ω^(T)` has independent columns in expectation satisfying `𝔼[(Ω^(T))ᵀ Ω^(T)] = I_T` (CountSketch and SRHT both satisfy this), we have:

**Theorem 1 (Unbiasedness).** `𝔼[ĝ_Ω] = Xᵀ 𝔼[(Ω^(T))ᵀ Ω^(T)] dY = Xᵀ dY = g`.

**Theorem 2 (Variance, CountSketch case).** For `Ω^(T)` a CountSketch with `m` buckets,
```
Var(ĝ_Ω)_{a,b} = (1/m) · [ Σ_i X_{i,a}² · Σ_i (dY)_{i,b}² - (Σ_i X_{i,a}(dY)_{i,b})² ]
              ≤ (1/m) · ‖X_{:,a}‖² · ‖dY_{:,b}‖².
```
So entrywise relative variance is `O(1/m)` — it shrinks linearly in the temporal sketch. Summing entrywise variances:
```
𝔼 ‖ĝ_Ω − g‖_F² ≤ (1/m) ‖X‖_F² ‖dY‖_F².
```

**Secondary estimator (subspace-embedding JL on features).** For a block `Y = X W`, we also sketch in the `d` direction, setting
```
ĝ_{Ω,Ψ} := (Ω X)ᵀ (Ω dY) / c(k) · (Ψ)ᵀ Ψ
```
with `Ψ ∈ ℝ^{k×d}` an SRHT and `c(k) = 1` by construction. For two independent sketches:
```
𝔼‖ĝ_{Ω,Ψ} − g‖₂ ≤ (√(d/k) + √(d/m)) ‖g‖₂    with probability ≥ 1 − δ.
```
See §5 for constants.

**Block-recursive sketching for the backward pass.** During the backward pass, the upstream gradient arriving at layer ℓ is itself in sketched form: we carry `Ŝ_{dY} := Ω^(T) dY ∈ ℝ^{m×d'}` rather than `dY`. Then:
- Weight gradient `ĝ = (Ω^(T) X_ℓ)ᵀ Ŝ_{dY}` reuses the same stored `Ω^(T) X_ℓ` computed on the forward pass. **No activation re-materialisation.** This is the dominant memory saving.
- Input gradient `dX_ℓ = dY · Wᵀ` is propagated sketched: `Ω^(T) dX_ℓ = Ω^(T) dY · Wᵀ = Ŝ_{dY} · Wᵀ`. So we can propagate `Ŝ_{dY}` through any linear layer with a `(m×d) · (d×d')` GEMM.

**Nonlinear block (controlled-bias estimator).** For `Y = φ(X W)` with `φ` a pointwise SiLU / GELU / softmax-normalised:
- The true upstream is `dW = Xᵀ · diag(φ'(X W)) · dY`.
- SPECTRA computes `dW̃ = (Ω^(T) X)ᵀ · (Ω^(T) · diag(φ'(X W)) · dY)`. This is **still unbiased conditional on the sketched stored row activations** iff we unsketch `φ'(X W)` exactly on the rows that survived (tracer set `R_ℓ`) and approximate elsewhere. The resulting bias is bounded by the unsketching gap — see §6.

**Control variate.** The dominant variance source is the cross term between `X` and `dY` rows sharing a bucket but carrying uncorrelated signal. Anchor `g̃_lin = (W_ℓ^{prev})ᵀ ⟨dY⟩ ⟨X⟩` from running means (EMA over ≈100 steps). Then
```
ĝ_CV := ĝ_Ω − α · (ĝ_Ω^{(lin)} − g̃_lin),   α* = Cov(ĝ_Ω, ĝ_Ω^{(lin)}) / Var(ĝ_Ω^{(lin)}).
```
Estimate `α*` with EMA; clip `α ∈ [0, 1]`. Conjectured variance reduction ≥ 3×.

---

## 5. Concentration / error bounds

**Theorem 3 (JL subspace embedding).** Let `Ω ∈ ℝ^{k×d}` be SRHT with `k ≥ C · ε⁻² · (log(r/δ))² · r`, `r = rank(X)`. Then with probability ≥ 1 − δ,
```
(1 − ε) ‖X v‖² ≤ ‖Ω X v‖² ≤ (1 + ε) ‖X v‖²    ∀ v ∈ ℝ^d.
```
Tropp 2011: `C ≤ 6`. For `d = 8192`, `k = 512`, `ε = 0.1`, `δ = 10⁻⁶`, bound requires `r ≲ k/200`. This is **satisfied** when activations are low effective-rank (empirically ≈200 for trained `d=8192` transformers). SPECTRA relies on this empirical-rank assumption, made explicit.

**Theorem 4 (Matrix Bernstein for sketched outer product).** For `M = (1/m) Σ X_i Y_iᵀ`, `‖X_i‖, ‖Y_i‖ ≤ R`, `𝔼 M = g`:
```
Pr[‖M − g‖_op > t] ≤ 2d · exp(− m t² / (2σ² + (2/3) R² t)),
```
with `σ² = ‖X‖_F² ‖dY‖_F² / T`. **Corollary:** with prob ≥ 1 − δ,
```
‖ĝ_Ω − g‖_op ≤ √(2σ² log(2d/δ) / m) + (2/3) R² log(2d/δ) / m.
```
With `m = T/8, δ = 10⁻⁶, d = 8192` this is ≤ `0.03 · ‖g‖_op` for training-regime activation statistics.

**Theorem 5 (CountSketch collision bound).** For CountSketch `Ω^(T)` with `m` buckets and attention-output reconstruction `Ô = Σ_j p_j · s(j) Ṽ_{h(j)}`, where `p_j` is the softmax probability:
```
𝔼 ‖Ô − O‖² ≤ (‖p‖_∞ · ‖V‖_F²) / m.
```
Attention softmaxes are heavy-tailed, so `‖p‖_∞` is typically `≈ 0.3`; with `m = T/8`, `T = 4096`, `‖V‖_F²/d_h = 1`, the per-output squared error is `≈ 10⁻⁴`.

**Probabilistic guarantee (combined).** Call an optimiser step **ε-good** if every per-layer gradient estimator satisfies `‖ĝ_ℓ − g_ℓ‖_F ≤ ε ‖g_ℓ‖_F`. Union-bounding over `L` layers and per-layer `O(1)` estimators:
```
Pr[step is ε-good] ≥ 1 − L · (δ_JL + δ_MB + δ_CS).
```
For `L = 96`, `δ_* = 10⁻⁸`, this is `≥ 1 − 10⁻⁶`. Non-good steps are detected (§9) and retried with `k → 2k`.

---

## 6. Compatibility with attention and nonlinearities

Three classes of operation:
1. **Linear** — `XW`, residual add, embedding, output projection: commute with sketch exactly.
2. **Separable nonlinear** — LayerNorm, RMSNorm, SiLU/GELU, softmax: do **not** commute.
3. **Attention score mix** — has an exact exploit via CountSketch signed hashing (§3).

**LayerNorm / RMSNorm.** `LN(x) = (x − μ)/σ`. Maintain per-token `(μ,σ)` as a two-scalar side channel (`2·BT` floats, uncompressed). Then `LN(X)·Ωᵀ = diag(1/σ)·(X − μ·1ᵀ)·Ωᵀ`, expressible entirely in sketch space because the mean is a rank-1 correction: `μ·1ᵀ·Ωᵀ` precomputed once per token. **Zero-bias LayerNorm in sketch space** — *hard claim*.

**SiLU / GELU.** Do not commute. **Patch-unsketch protocol:**
- Score `ρ_i = ‖(S_X^{(d)})_i‖_∞` per token; select top-`r` rows into tracer set `R_ℓ` (`r = m`).
- Fully materialise `X_{R} W_1 → φ → W_2` at full `d_ff` width for tracer rows.
- For remaining `BT − r` rows: first-order expansion `φ(z) ≈ φ(z₀) + φ'(z₀)(z − z₀)` around empirical mean `z₀`, propagated through the sketch.
- **Unsketching gap:** `η(k, r) = 𝔼 ‖φ − φ̂‖_F / ‖φ‖_F ≤ 0.5 · (‖φ''‖_∞/‖φ‖_2) · √(Var(XW)·(1 − r/BT)/k)`. For SiLU with `k = d/8, r = T/8` this is `≤ 2%` — within BF16 quantisation noise.

**Softmax.** Attention softmax: handled exactly via signed-expansion (§3). LM-head softmax over `V`: **not sketched**, because loss gradient needs exact logits. Output projection `W_vocab ∈ ℝ^{d×V}` is applied to the reconstructed residual `X_L = S_L^{(d)}·(Ω^(d))⁺`; cost `O(BT·V)` matches baseline. LM-head gradient is back-projected into sketch space: `Ω^(T)·(∇X_L)`.

---

## 7. Objective and training dynamics

The optimisation target is unchanged:
```
L(W) = 𝔼_{(x,y)∼D} [ CE(f_W(x), y) ].
```

**Sketched-SGD.** We run any base optimiser (AdamW / ATLAS / VESTA) on the SPECTRA gradient `ĝ`:
```
W_{t+1} = Opt(W_t, ĝ_t).
```

**Theorem 6 (Convergence under unbiased estimator).** If `ĝ_t` is unbiased with `𝔼‖ĝ_t − g_t‖² ≤ σ²_SGD + τ²‖g_t‖²` and `L` is `β`-smooth, then AdamW on `ĝ` gives
```
min_{t≤T} 𝔼‖g_t‖² ≤ O((L⋆ + σ²_SGD)/√T + τ²β),
```
matching classical noisy-SGD. SPECTRA inflates `σ²_SGD` by `(1 + d/k + T/m)` per layer and introduces `τ² = O((d/k)(T/m))` from the nonlinear patch. With `k=d/8, m=T/8`, `τ² = 64`; the control variate reduces it to `≈ 8`, within 3× of vanilla AdamW.

**Learning-rate.** Optimal LR scales as `1/(σ² + ‖g‖²)`. SPECTRA inflation factor `r_σ = 1 + d/k + T/m ≈ 17`. Therefore `LR /= √17 ≈ 4×` or `B *= 17`.

**Conjecture (SPECTRA–warmup equivalence).** SPECTRA with warmup phase `[0, T_w]` at `k=d, m=T` (baseline), annealed linearly to target `k` by step `10T_w`, reaches final loss within `10⁻²` of unsketched. Warmup is required because early-training gradients are high-rank.

---

## 8. Memory and compute complexity

Per transformer block, per training step, compared to AdamW + flash-attention BF16 baseline.

**Activation memory (dominant axis).**
| | Baseline | SPECTRA |
|---|---|---|
| Residual stream | `2 · BTd` (fwd + saved for bwd) | `2 · BTk` |
| Attention K,V saved | `2 · BTd_h · H_kv` | `2 · m · d_h · H_kv` |
| MLP hidden saved | `BT · d_ff` | `BT · k_ff` on non-tracer + `r · d_ff` on tracers |
| **Per-block activation** | `BT·(2d + d_ff + 2d_h H_kv) ≈ BT · 6d` | `BT·(2k + k_ff) + m · d_h · H_kv + r · d_ff ≈ BT · k` |
| **Total L-block** | `L · BT · 6d` | `L · BT · k` |

With `k = d/8`: **6×** × **8×** = **48× less activation memory per layer**. For a 100B model with `L=96, d=12288, B·T = 2M`, baseline activation is `≈ 13 TB`; SPECTRA is `≈ 270 GB` — fits on 8×H100.

**Compute per block.**
| Operation | Baseline FLOPs | SPECTRA FLOPs |
|---|---|---|
| QKV projection | `3 · BT · d · d` | `3 · BT · d · k` (precompute `W̃`) |
| Attention scores | `BT·T·d_h` | `BT·m·d_h` |
| Attention softmax | `BT·T` | `BT·T` (unchanged, exact) |
| Attention output | `BT·T·d_h` | `BT·m·d_h` |
| MLP up + gate | `BT · d · 2d_ff` | `BT · d · 2k_ff + r · d · 2d_ff` |
| MLP down | `BT · d_ff · d` | `BT · k_ff · d + r · d_ff · d` |
| Backward (symmetric) | `2×` forward | `2×` forward |
| **Per-block fwd** | `≈ BT · (6d² + 2Td_h + 4d·d_ff)` | `≈ BT · (6dk + 2md_h + 4d·k_ff)` |

With `k = d/8`, `m = T/8`, `k_ff = d_ff/8`: **8× FLOP reduction** per block, matched by BF16 tensor cores.

**Regime of strict dominance.** SPECTRA strictly dominates whenever
```
k / d + m / T + r / (BT) < 1 − (overhead of Ω·X and FWHT) / (baseline cost).
```
FWHT overhead is `d log d / d² = log d / d ≈ 13 / 8192 ≈ 10⁻³`. CountSketch overhead is `O(BT)` which is dwarfed by the GEMM. So SPECTRA dominates for any `k ≤ d/2, m ≤ T/2`.

**DDP.** Gradients `ĝ` are already low-dimensional (either `k×d` or `m×d` per block), so all-reduce is `O(k·d · L)` = `8×` smaller per step.

---

## 9. Failure modes

1. **Heavy-hitter miss (CountSketch attention).** Hash collisions between attended key tokens inject bias `s(j)s(j') p_{j'} V_{j'}`. Mitigation: rehash `(s,h)` every 100 steps via seeded RNG — CountSketch is rotationally unbiased, so persistent bias is broken. Detection: spot-check full-attention output on 1% of steps; if bias > `10⁻²‖O‖`, bump `m`.
2. **Unsketchable LM-head softmax.** Gradient requires exact logits. Keep un-sketched; cost `O(BT·V)` matches baseline. No speedup at the output layer; no regression either.
3. **Rank explosion during warmup.** Early activations are near-isotropic and high-rank; JL `ε` grows as `√(d/k)`. Mitigation: warmup with `k=d`, linearly anneal to target over `10·T_w` steps (`T_w ≈ 2000` conjectured sufficient).
4. **Variance-induced optimizer divergence.** Inflated `σ²` biases Adam's second moment. Mitigation: scale `β₂ → 1 − (1 − β₂)/r_σ`; clip by SPECTRA-aware norm `‖ĝ‖/√r_σ`.
5. **BF16 cancellation in SRHT.** Length-8192 Hadamard accumulated in 7-bit BF16 mantissa loses precision. Mitigation: BF16 inputs, FP32 accumulators, BF16 output — matches the existing `CUBLAS_COMPUTE_32F_FAST_16BF` path.
6. **Nondeterministic GPU reductions.** Mitigation: CPU-seeded Xorshift128+ for Hadamard signs; deterministic BF16 GEMM path; MurmurHash3 with seeded salt for CountSketch.
7. **Mid-training low-rank breakage.** Phase transitions may push activation rank past `k`. Detection: monitor `‖ΩX‖²/‖X‖²`; if it falls below `k/d · (1 − ε)`, auto-double `k` for that block.

---

## 10. Minimal prototype implementation path

**Stage 0 — sketch primitives (1–2 weeks).** New `gpu_sketch.h / .cu` in `Networks/cuda/`. Kernels: `srht_apply_bf16` (diag-sign · FWHT · subsample), `countsketch_apply_bf16`, `srht_transpose_apply_bf16`. Unit test `sketch_kernels_test.cpp` — CPU/GPU parity under BF16, verify `𝔼‖Ωx‖²/‖x‖² = 1±ε` over 1000 samples.

**Stage 1 — sketched linear layer (1 week).** New `SketchedLinear` layer; stores `W̃ := W Ωᵀ`, refreshed post optimiser step. Forward: one `(BT×d)·(d×k)` SGEMM. Backward: `ĝ = (Ω^(T)X)ᵀ · (Ω^(T)dY)` reusing stored `Ω^(T)X`. Flag `GLADES_SKETCH_ACTIVATIONS` in `training_config.h`.

**Stage 2 — sketched attention (2 weeks).** Extend `transformer_ops.h` with `attention_sketched_fwd/bwd`. Online K/V CountSketch kernel `countsketch_accumulate_kv_bf16`. Score-expansion kernel `countsketch_expand_scores` with `(s,h)` in constant memory. Softmax kernel unchanged.

**Stage 3 — patch-unsketch MLP (1 week).** `select_top_r_by_norm` → tracer set `R`. Full-precision MLP for tracers; linearised for rest. Test: `‖φ̂ − φ‖/‖φ‖ < 5%`.

**Stage 4 — end-to-end parity (1 week).** Train 125M-param GPT on TinyStories; target ≥4× memory, ≥2× speed, ≤1% loss degradation at `k=d/8, m=T/8`. CPU-GPU bit-parity test via seeded RNG; determinism test (two runs bit-identical BF16-wise).

**Stage 5 — scale-out.** Compose with ATLAS/VESTA/HELIOS (they act on parameters, SPECTRA acts on activations — orthogonal). DDP all-reduce of already-compressed `ĝ`.

**Deliverable footprint.** ~2500 LOC CUDA, ~800 LOC C++ glue, ~400 LOC tests. ~6 person-weeks for stages 0–4.

---

*End of Candidate B.*
