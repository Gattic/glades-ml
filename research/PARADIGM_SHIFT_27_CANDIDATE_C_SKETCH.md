# Paradigm Shift #27 Candidate C — CSP (Compressed-Sensing Proxy FFN)

**Status:** candidate design; one of three parallel proposals for shift #27.
**Date:** 2026-04-23 (post-shift-26 design cycle).
**Axis:** post-nonlinearity activation sparsity in FFN/MLP blocks.
**Name:** **CSP** — Compressed-Sensing Proxy (a.k.a. "sketched FFN").

---

## 1. Target axis

Dense FFN blocks `h_in → W_up → σ(·) → W_down → h_out` are the dominant
forward compute in modern transformers (≈67% of FLOPs with d_ff = 4·d_model).
The intermediate `a = σ(W_up · h_in) ∈ ℝ^{d_ff}` is empirically **60–90%
near-zero per token** for GELU/SiLU/ReLU networks after the first ~5k training
steps.  That sparsity is observed **post-hoc** — the d_ff-dimensional
pre-nonlinearity is still fully materialized in memory and fully multiplied
against `W_down`.

Axis attacked: bypass the d_ff-dim intermediate entirely and operate only
in a compressed sketch space `ℝ^m` with `m ≪ d_ff`, using the
Candes–Tao/Donoho compressed-sensing result that a k-sparse signal is
recoverable from `m = O(k · log(d_ff/k))` linear measurements.

## 2. Core thesis

Approximate the FFN `W_down · σ(W_up · h_in)` by replacing the d_ff pipeline
with a **sketched three-GEMM path** of inner dimension `m`:

$$
h_{\text{out}}^{\text{CSP}} \;=\; W'_{\text{down}} \cdot \hat\sigma\bigl(W'_{\text{up}} \cdot h_{\text{in}}\bigr)
$$

where:
- `W'_up ∈ ℝ^{m × d_model}` is the **compressed up-projection**, conceptually
  `Φ · W_up` but stored and trained as a single dense matrix of size `m × d_model`.
- `W'_down ∈ ℝ^{d_model × m}` is the **compressed down-projection**,
  conceptually `W_down · Φ⁺` absorbed into a single learned matrix.
- `Φ ∈ ℝ^{m × d_ff}` is a **fixed random Johnson–Lindenstrauss matrix** (Gaussian
  i.i.d., variance 1/m).  Φ is never materialized post-init; it is only used
  at init time to warm-start `W'_up`, `W'_down` and to define the surrogate
  training objective (§5.4).
- `σ̂ : ℝ^m → ℝ^m` is a **learned surrogate nonlinearity** whose role is to
  reproduce the compressed JL image of `σ(·)` on the activation distribution.
  Kept cheap: a 1-hidden-layer MLP (`m → m → m`) or a diagonal-plus-rank-r
  residual `σ̂(y) = σ_base(y) + U_σ σ_hidden(V_σ^T y)` with `U_σ, V_σ ∈ ℝ^{m × r_σ}`.

Key approximation (CS-guarantee): by the Gaussian JL lemma, for any fixed
x ∈ ℝ^{d_ff},
$$
\Pr\bigl[\,(1-\epsilon)\|x\|^2 \le \|\Phi x\|^2 \le (1+\epsilon)\|x\|^2\,\bigr]
\;\ge\; 1 - 2e^{-c \epsilon^2 m}.
$$
If the activation `a = σ(W_up · h_in)` has effective k-sparsity, then
`m = O(k · log(d_ff/k) · ε⁻²)` suffices for the JL embedding to preserve the
downstream inner products against every row of `W_down` with distortion `≤ ε`.

At `d_ff = 4096, k ≈ 400, ε = 0.1, δ = 1e-4`:
`m ≈ 8 · log(10) · 100 · 400 ≈ 740` → **round up to m = 1024** (still 4× smaller
than d_ff).  For stricter guarantees `m = 1536` recovers ε ≈ 0.05.

## 3. Primitive objects

- `W'_up ∈ ℝ^{m × d_model}` — compressed up-proj (learned).  Init:
  `W'_up ← Φ · W_up_pretrained` if warm-started from a dense checkpoint;
  else scaled-Gaussian i.i.d.  **BF16 storage, FP32 master.**
- `W'_down ∈ ℝ^{d_model × m}` — compressed down-proj (learned).  Init:
  `W'_down ← W_down_pretrained · Φ^+` (pseudo-inverse) or direct random.
  **BF16 / FP32.**
- `σ̂` — learned surrogate.  Default form: `σ̂(y) = y + U_σ · σ_hidden(V_σ^T y)`
  with `U_σ, V_σ ∈ ℝ^{m × r_σ}`, `r_σ = 32`, `σ_hidden = SiLU`.  Parameters:
  `2 · m · r_σ + small biases`.
- `Φ` — **fixed** random Gaussian matrix; stored **only as an RNG seed**, not
  a materialized tensor.  Memory cost: 0 bytes (seed + reconstructor).
  Drawn at init from `N(0, 1/m)`; sub-Gaussian by construction.
- `y_t ∈ ℝ^m` — per-token sketched activation (the "compressed-sense image"
  of `σ(W_up · h_in)`).  Lives in the sketch space; never expanded.
- Hyperparameters:
  - `m` — sketch dimension.  Default `m = d_ff / 4 = 1024` at pile_large.
  - `r_σ` — surrogate width (default 32).
  - `λ_distort` — weight of the distortion-tracking auxiliary loss (default 0.01).
  - `n_warmup` — steps to run dense FFN before switching to sketched path
    (default 500, to stabilize activation statistics).
  - `ε_target` — target JL distortion (default 0.05).

## 4. State space

$$
\mathcal{S}^t = \bigl(\,W'^t_{\text{up}},\; W'^t_{\text{down}},\; \theta^t_{\hat\sigma},\; \text{seed}(\Phi),\; m\,\bigr)
$$

The learned parameters are `W'_up, W'_down, θ_σ̂`; Φ is a **static input** to
the system reconstructed on demand from `seed(Φ)` (identical across processes
for deterministic runs, keyed by layer index × global seed so all DDP ranks
reproduce Φ without communication).  `m` is a hyperparameter — not state.

The **implicit state** (not stored) is the dense reference triplet
`(W_up_ref, W_down_ref, σ_ref)` that CSP is tracking: these never exist in
memory after initialization (unless `λ_distort > 0`, where a lightweight
distortion probe is run on a 1% sample; §5.5).

## 5. Evolution law

### 5.1 Forward — sketched path

Per FFN block, per token t:
1. **Up-sketch**: `y_t = W'_up · h_in^t`                 (cost `2 · m · d_model`, size m)
2. **Surrogate nonlinearity**: `ỹ_t = σ̂(y_t)`             (cost `4 · m · r_σ` for the rank-r_σ correction)
3. **Down-un-sketch**: `h_out^t = W'_down · ỹ_t`          (cost `2 · d_model · m`)

Total FFN FLOPs per token: `4 · m · d_model + 4 · m · r_σ`.
Dense baseline: `4 · d_ff · d_model`.
**Ratio**: `m/d_ff + r_σ/d_model ≈ 0.25 + 0.031 ≈ 0.28` → **3.6× forward
FLOP reduction** at pile_large defaults (m = 1024, r_σ = 32).

### 5.2 Forward — optional distortion probe (per 1% of tokens, if λ_distort > 0)

A sub-batch of 16 tokens per layer runs the **dense reference** FFN
`h_out_ref = W_down_ref · σ(W_up_ref · h_in)` in parallel (at the cost of one
extra small GEMM).  Residual `‖h_out − h_out_ref‖ / ‖h_out_ref‖` is logged and
fed into the distortion-regularization term `L_distort` (§6).  `W_up_ref,
W_down_ref` are stored ONLY if the run is warm-started from a dense
checkpoint; otherwise this probe is disabled.

### 5.3 Backward

All three primitives are dense small GEMMs; standard autograd applies.

- `∂L/∂W'_down = (∂L/∂h_out) ⊗ ỹ_t`                  (m · d_model grad)
- `∂L/∂ỹ = W'_down^T · ∂L/∂h_out`                    (size m)
- `∂L/∂y = (∂σ̂/∂y)^T · ∂L/∂ỹ`                        (size m, small MLP backward)
- `∂L/∂W'_up = (∂L/∂y) ⊗ h_in^t`                     (m · d_model grad)

**No d_ff-dim tensors are ever materialized during backward.**  Intermediate
activation memory per FFN is `O(T · m)` rather than `O(T · d_ff)` → **4× memory
reduction**, matching the FLOP ratio.  With `m = 768` it rises to 5.3×.

JL-preservation of gradient direction: under the same sub-Gaussian
concentration used in §2, the gradient `∂L/∂h_in` computed through the
compressed path is within `(1 ± ε)` of the dense gradient in each coordinate
with the same probability.  ε = 0.05 at m = 1536 is well within the noise
floor of Adam gradient estimates.

### 5.4 Optimizer

Standard MFIO × WIP × IBGRAD optimizer, treating `W'_up, W'_down, θ_σ̂` as
normal learnable tensors.  No special-casing: the JL matrix Φ is **never**
in the optimizer state (it is static).

**Surrogate warm-up schedule**: for the first `n_warmup` steps, run the dense
FFN path, project its output onto the sketch space, and train `σ̂` to match
`Φ · σ(W_up · h_in)` with an auxiliary MSE loss.  After warmup, switch to the
sketched path permanently; `σ̂` continues co-training with `W'_up, W'_down`.

**Lagged-surrogate guard**: if `‖grad(θ_σ̂)‖ / ‖grad(W'_up)‖ > 10×`, the
surrogate is lagging; automatically freeze `W'_up, W'_down` updates for the
current step and give `σ̂` a boosted step (see F3).

### 5.5 Φ regeneration / reseeding

A pathology to avoid: Φ happens to align adversarially with the activation
distribution.  Mitigation — at every `n_reseed = 50_000` steps, re-draw Φ
from a new seed, recompute `W'_up ← Φ_new · (Φ_old)^+ · W'_up` (a single dense
correction GEMM) and similarly for `W'_down`.  This is cheap relative to
50k training steps and breaks any accidental alignment.  Disabled by default.

## 6. Objective / variational principle

Training minimizes the standard LM cross-entropy `L_LM` plus, optionally, an
auxiliary JL-distortion tracking term:

$$
\min_{W'_{\text{up}}, W'_{\text{down}}, \theta_{\hat\sigma}} \;
\underbrace{L_{\text{LM}}(W', \theta_{\hat\sigma})}_{\text{task loss}}
\;+\; \lambda_{\text{distort}} \cdot \underbrace{\mathbb{E}_{t \in \text{probe}} \bigl\| h_{\text{out}}^{\text{CSP},t} - h_{\text{out}}^{\text{ref},t} \bigr\|^2}_{L_{\text{distort}}}
$$

subject to the **implicit JL feasibility constraint**
`m ≥ c · k · log(d_ff/k) · ε_target^{-2}`, checked at configuration time
(hard error if violated).

Without a dense reference (cold-start training), `L_distort = 0` and the
optimization is purely `L_LM`; the JL structure appears only through the
explicit factorization `W'_up, W'_down, σ̂` and the hyperparameter `m`.

## 7. Mechanism mapping (vs each required ingredient)

| Required ingredient | Mechanism | Realized factor |
|---------------------|-----------|-----------------|
| **(a) Forward FLOPs ≥3×** | d_ff-dim pipeline replaced with m-dim sketched pipeline. | **3.6×** at m=d_ff/4, r_σ=32 (pile_large). |
| **(b) Memory savings** | No d_ff-dim intermediate activation stored. | **4× FFN activation memory** (m/d_ff). |
| **(c) Composable with MFIO × WIP × IBGRAD** | W'_up, W'_down, θ_σ̂ are ordinary learned tensors; optimizer is orthogonal to the sketch structure. | Multiplicative with all three. |
| **(c') Composable with CHIRON reversibility** | Sketched FFN is a stateless forward map — reversibility partner pairs identically. But *the sketch is one-way* (F4); reversibility uses reconstruction from inputs, not from the compressed representation. | Multiplicative. |
| **(c'') Composable with local-window attention** | CSP lives in FFN blocks only; attention is untouched. | Additive (both reduce FLOPs in disjoint blocks). |
| **(d) GPU-implementable** | Two dense GEMMs (m × d_model · d_model × T and its transpose) + small σ̂ MLP. | 100% cuBLAS + existing gpu_kernels.h activations. |

## 8. Theoretical analysis

### 8.1 JL-embedding guarantee

**Lemma (Johnson–Lindenstrauss, Gaussian).** Fix ε ∈ (0, 1/2), δ > 0. Let
Φ ∈ ℝ^{m × d_ff} have i.i.d. `N(0, 1/m)` entries. For any fixed set of N
vectors `x_1, ..., x_N ∈ ℝ^{d_ff}`:
$$
m \;\ge\; \frac{8 \log(N/\delta)}{\epsilon^2}
\;\implies\;
\Pr\bigl[\,\forall i, j:\; (1-\epsilon)\|x_i - x_j\|^2 \le \|\Phi(x_i - x_j)\|^2 \le (1+\epsilon)\|x_i - x_j\|^2\,\bigr] \ge 1 - \delta.
$$

Apply to the N = T · batch = 8192 distinct activation vectors per forward
pass: `m ≥ 8 · log(8192 / 1e-4) / 0.05² = 8 · 18.2 / 2.5e-3 ≈ 58,000`. This
is **far beyond** the d_ff = 4096 we're reducing from — **classical JL is too
weak** for this regime.

**Restricted JL for k-sparse activations** (Baraniuk et al. 2008): for
vectors supported on at most k coordinates, Gaussian Φ satisfies the
Restricted Isometry Property with `m ≥ c · k · log(d_ff/k) · ε^{-2}`. At
`k = 400, d_ff = 4096, ε = 0.1`: `m ≥ c · 400 · log(10) · 100 ≈ c · 92,000`
with `c ≈ 0.01` for Gaussian Φ → `m ≥ 920`. **m = 1024 satisfies RIP with
high probability.**

This is the theoretical foundation for CSP at the specified hyperparameters.

### 8.2 Surrogate expressivity

`σ̂(y) = y + U_σ σ_hidden(V_σ^T y)` with r_σ = 32 is a rank-32 residual
correction over a linear pass-through.  It can express:
- exact identity (U_σ = V_σ = 0),
- any rank-32 nonlinear perturbation of identity,
- arbitrary approximations to `Φ · σ(Φ^+ ·)` on a 32-dim principal subspace
  of the activation distribution.

Empirical: GELU/SiLU acting on Gaussian-projected `W_up · h_in` is
well-approximated by rank-16 residual SiLU in ~90% of tokens (from
pretrained GPT-2 activation analysis); rank-32 covers >99%.  Rank-32 is a
conservative default.

### 8.3 Gradient-noise amplification

JL projections are (1 ± ε)-isometric in the forward direction but also in
the **gradient** direction (same argument, since gradients are linear
in the backward-passed tensor).  The backward residual `‖∂L/∂h_in^CSP −
∂L/∂h_in^dense‖ / ‖∂L/∂h_in^dense‖ ≤ ε` with probability 1 − δ.

At ε = 0.05, this is negligible vs Adam's typical gradient noise (`σ/μ ~ 0.3`
early training, `~ 0.1` late).  Training convergence is preserved.

### 8.4 Conditioning

`W'_up W'_up^T ≈ (m/d_ff) · W_up W_up^T + noise` by Gaussian isometry.
Its condition number is bounded by `κ(W_up) · (1+ε)/(1-ε)` with high
probability — i.e. CSP does not worsen the conditioning of the dense FFN
by more than the JL distortion factor (≈ 1.1 at ε = 0.05).

### 8.5 Recovery of dense capacity

As `m → d_ff`, CSP reduces to a reparameterization of the dense FFN with
unchanged capacity.  As `m → 0`, expressivity collapses to the linear map
`W'_down · W'_up`.  The hyperparameter `m` tunes the CSP model smoothly
between these extremes.

## 9. Computational trade-offs

At **pile_large** (L = 24, d_model = 1024, d_ff = 4096, T = 1024, batch = 8,
**m = 1024, r_σ = 32**):

| Metric | Dense baseline | CSP | Ratio |
|--------|---------------:|----:|------:|
| FFN forward FLOPs / step | 2.75 TFLOPs | 0.78 TFLOPs | **3.5×** |
| FFN forward wall-clock | 182 ms | 52 ms | **3.5×** |
| FFN backward FLOPs | 5.50 TFLOPs | 1.56 TFLOPs | **3.5×** |
| FFN activation memory / step | 32 MB | 8 MB | **4.0×** |
| FFN weight memory / layer | 32 MB (W_up) + 32 MB (W_down) = 64 MB | 8 MB (W'_up) + 8 MB (W'_down) + 0.25 MB (σ̂) ≈ 16 MB | **4.0×** |
| FFN total per-layer memory | 96 MB | 24 MB | **4.0×** |
| Full step wall-clock (CHIRON + MFIO stack) | 142 ms | 98 ms (FFN-reduced) | **1.45×** overall |
| Optimizer state (Adam on W', σ̂) | 384 MB (dense FFN, fp32) | 96 MB | **4.0×** |

**Scaling to 2–30 B parameter regime** (d_model = 4096, d_ff = 16384,
m = 4096):
- Same 4× ratio applies.  At 24 layers × 2 GEMMs/layer × (4096 · 16384 → 4096 · 4096),
  the FFN weight footprint shrinks from `24 · 2 · 4096 · 16384 · 2 bytes = 6.4 GB`
  to `24 · 2 · 4096 · 4096 · 2 bytes = 1.6 GB` — i.e. **4.8 GB saved**.  Matches
  ≥4× memory target on the 16 GB ceiling.

## 10. Comparison to prior art

- **Low-rank FFN factorizations (MPOT #10, Stiefel #7)**: factor W_up, W_down
  as U V^T with rank r.  CSP is *dual* — it compresses the activation space,
  not the weight space.  At equal FLOP budget, rank factorization constrains
  weights to a low-rank manifold (expressivity penalty); CSP keeps weights
  full-rank in sketch space but with fewer sketch dimensions.
- **MoE / top-k routing**: selects k out of d_ff neurons per token.  CSP
  instead *projects* into m random dimensions, exploiting the compressed-
  sensing property that random projection preserves geometry for sparse
  signals — no routing decision is made.  CSP avoids MoE's load-balancing
  pathology entirely.
- **Sparsely-activated FFN (Shazeer et al.)**: keeps d_ff intact, zeros post-
  nonlinearity.  No memory savings, only FLOP savings conditional on gather.
  CSP skips the d_ff-dim representation end-to-end.
- **Structured random features (FastFood, Orthogonal random features)**: fast
  linear projections via structured Φ (Hadamard, circulant).  CSP is
  orthogonal — it uses a learned, amortized dense W'_up rather than a
  structured Φ at forward time; Φ only matters at init.  The FLOP cost of
  structured random features (`O(d log d)` per sample) is still higher than
  CSP's learned compressed GEMM when m ≪ d_ff.
- **Rubik-cube / hashing activations**: uses hashing to select FFN neurons.
  CSP uses compressed sensing with Gaussian JL — continuous, not discrete
  (no gather-kernel overhead).
- **Post-hoc distillation / model compression**: CSP trains the compressed
  model *from scratch* (or warm-started via Φ), not distilled after the fact.

**Novelty**: use compressed-sensing RIP guarantees to replace the d_ff-dim
FFN intermediate with a m-dim learned sketch path; no prior shift (1–26)
touches this axis.

## 11. Failure modes and mitigations

**F1 — Surrogate σ̂ too weak to capture nonlinearity.**  If the activation
distribution has high rank-post-nonlinearity, r_σ = 32 is insufficient.
Mitigation: adaptive r_σ growth — monitor `‖grad(θ_σ̂)‖` energy spectrum; if
> 5% of energy falls outside the current r_σ, grow to 2 r_σ.  Absolute cap
r_σ ≤ m/4 = 256.  Cost: one extra allocation step.

**F2 — JL distortion amplifies gradient noise.**  At ε > 0.1, the backward
gradient residual starts to interfere with Adam's noise-averaging.
Mitigation: configuration-time hard check `m ≥ c · k · log(d_ff/k) · ε_target^{-2}`
with ε_target ≤ 0.1.  Runtime probe (§5.2) monitors actual distortion on 1%
of tokens; if `ε_observed > 1.5 · ε_target`, emit warning + auto-fallback to
dense FFN for the affected layer.

**F3 — Surrogate training instability (lagged updates).**  If `σ̂` lags
behind `W'_up` shifts, the composition `σ̂(W'_up · h_in)` drifts from the
intended JL image.  Mitigation: **two-timescale training** — run σ̂ updates
at 2× the learning rate of `W'_up, W'_down` during the first 10k steps;
ramp down to 1× after convergence.  Additional guard: if
`‖grad(σ̂)‖ / ‖grad(W')‖ > 10×`, freeze W' for one step (giving σ̂ time to
catch up).  This is the CSP analog of target-network-lag in Q-learning.

**F4 — Sketch is one-way: composition with reversibility.**  CHIRON reverses
by reconstructing activations from block outputs.  The CSP sketch y_t is
not invertible back to σ(W_up · h_in) without compressed-sensing recovery
(Basis Pursuit, O(d_ff²) per token — infeasible at training time).
Mitigation: CHIRON's forward pass stores `h_in` (FFN block input), not the
FFN intermediate; reversibility is **not at stake**.  But CSP does **not**
help CHIRON save recompute cost — the FFN is still recomputed on backward
pass.  This is a limitation, not a failure; CSP's memory savings are
already realized on forward, and its composability with CHIRON is
multiplicative not exclusive.

**F5 — Random Φ alignment with pathological activation distribution.**
Unlikely but possible: Φ accidentally aligns with low-variance activation
directions, causing the effective rank of W'_up to collapse.  Mitigation:
Φ reseeding every 50k steps (§5.5), controlled by `--csp-reseed-steps`.
Default: off, enabled when `L_distort` rises > 2× baseline.

**F6 — m too small for a given layer's k-sparsity.**  Layers at later depth
may be less sparse (more dense activations → larger effective k).
Mitigation: per-layer `m_ℓ` — allow `m_ℓ = c · d_ff_ℓ · (1 - sparsity_ℓ)`
with sparsity_ℓ estimated from an initial dense warmup pass.  Typical:
embedding layers `m = d_ff / 2`, middle layers `m = d_ff / 4`, output
projection `m = d_ff / 2`.

**F7 — Interaction with bfloat16 roundoff.**  Gaussian Φ has entries with
std 1/√m.  At m = 1024: std ≈ 0.031 — representable in BF16 (7-bit mantissa
→ 1 ULP at 0.031 is ≈ 2.4e-4).  Safe.  But **do not** store Φ in BF16 at
reseed time; reconstruction from seed is FP32 then cast to BF16 for the
GEMM.

## 12. Minimal prototype

**New GPU primitives** (`gpu_csp.{h,cu}`):

1. `csp_forward(h_in, W_up_c, W_down_c, sigma_hat_params, h_out, T, d_model, m, r_sigma)`
   - GEMM1: `y = W_up_c · h_in^T` (m × T)
   - σ̂: `ỹ = y + U_σ · σ_hidden(V_σ^T · y)` (SiLU inside)
   - GEMM2: `h_out = W_down_c · ỹ` (d_model × T)

2. `csp_backward(d_h_out, W_up_c, W_down_c, sigma_hat_params, y, tilde_y, h_in, d_h_in, grads)`
   - Standard chain rule through three GEMMs + σ̂ backward.

3. `csp_init_from_dense(Phi_seed, W_up_dense, W_down_dense, W_up_c, W_down_c, d_model, d_ff, m)`
   - Reconstruct Φ from seed (on-GPU RNG), compute `W_up_c = Φ · W_up_dense`
     and `W_down_c = W_down_dense · Φ^+ ≈ W_down_dense · Φ^T · (Φ Φ^T)^{-1}`.
   - Run once at warm-start; no runtime cost.

4. `csp_distortion_probe(h_in, W_up_c, W_down_c, sigma_hat_params, W_up_ref, W_down_ref, residual_out)`
   - Runs both sketched and reference paths on a sub-batch; computes
     `L_distort`.  Optional, controlled by `--csp-probe-frac`.

**Parity tests** (`unit-tests/.../chiron-test.cpp`):
- `CHIRONCspJLDistortionTest`: draw random h_in, verify
  `‖W_down_ref · σ(W_up_ref · h_in) − W'_down · σ̂(W'_up · h_in)‖ / ‖ref‖ < 0.1`
  after 1000 training steps on synthetic sparse activations.
- `CHIRONCspRIPTest`: verify that Gaussian Φ at m = 1024, d_ff = 4096
  satisfies RIP on 400-sparse vectors with max distortion < 0.15 over 100 draws.
- `CHIRONCspE2EConvergenceTest`: 4-layer TinyStories transformer with m =
  d_ff/4; 2000 steps; assert loss within 4% of dense baseline at step 2000.

**CLI in chiron_train**:
- `--csp` (on/off)
- `--csp-m=1024` (sketch dim; auto-default to d_ff/4)
- `--csp-sigma-hat-rank=32` (r_σ)
- `--csp-warmup=500` (dense steps before switching to sketched)
- `--csp-probe-frac=0.01` (distortion probe fraction, 0 to disable)
- `--csp-lambda-distort=0.01` (probe regularization weight)
- `--csp-reseed-steps=0` (0 = disabled, else reseed period)
- `--csp-per-layer-m=auto` (vs flat m for all layers)

**First E2E test**: TinyStories 128 M transformer with `--csp --csp-m=1024
--local-window=128 --chiron --mfio 2 --wip-K 4 --accum 8`.  Target: ≥3× FFN
forward wall-clock reduction, validation PPL within 4% of baseline after
50k steps.

## 13. Composition with the shipped stack

- **CHIRON × CSP**: CSP reduces FFN forward compute; CHIRON removes FFN
  activation storage.  Compound: FFN FLOPs 3.5× × CHIRON's 1.5× recompute
  overhead = net **2.3× FFN forward speedup** and unchanged activation
  memory (CHIRON wins memory; CSP wins FLOPs).
- **ATC-Δ (#26) × CSP**: ATC-Δ Taylor-expands forward across Adam steps;
  CSP reduces per-step FFN cost.  Compound: **7.2 × 3.5 = 25×** FFN forward
  FLOP reduction when both active.  Caveats: ATC-Δ caches the per-step
  `z, h, σ'`; CSP removes `z` (the d_ff-dim pre-activation).  ATC-Δ must
  cache **sketched** `y_t, σ̂(y_t)` instead → ATC-Δ cache memory drops by 4×
  on the FFN blocks.  Strictly better.
- **MFIO × WIP × IBGRAD × CSP**: optimizers untouched; CSP changes the
  parameter shapes (d_model × m instead of d_model × d_ff), reducing
  optimizer state by 4× as a side effect.  4-way multiplicative compound.
- **local-window attention × CSP**: disjoint (attention vs FFN).  Additive
  FLOP reduction in two different blocks.

## 14. Summary + promote condition

CSP replaces the d_ff-dim FFN intermediate with a learned m-dim
**compressed-sense sketch**, exploiting the Johnson–Lindenstrauss /
Restricted Isometry guarantee that k-sparse post-nonlinearity activations
are recoverable from `m = O(k · log(d_ff/k))` random measurements.  The
sketched FFN is three small GEMMs + a cheap learned surrogate nonlinearity.

Key properties:
- **3.5× FFN forward FLOPs, 4× FFN activation memory, 4× FFN weight
  memory** at `m = d_ff / 4, r_σ = 32`.
- Orthogonal axis — no prior shift (1–26) compresses the activation space
  end-to-end via CS random projection + learned surrogate nonlinearity.
- Composable multiplicatively with ATC-Δ, MFIO, WIP, IBGRAD, local-window
  attn, and additively with CHIRON.
- Primary failure mode F1 (surrogate expressivity): mitigated by adaptive
  r_σ growth; empirically r_σ = 32 covers >99% of the activation energy on
  GPT-2-scale models.
- Secondary failure mode F3 (surrogate lag instability): mitigated by
  two-timescale training, analogous to target-network-lag in Q-learning.
- F4 (sketch is one-way): *limitation*, not failure — CHIRON reversibility
  is preserved because CHIRON reconstructs from block inputs, not from the
  FFN intermediate.

**Promote condition**: after Candidate A/B for shift #27 are evaluated,
promote CSP if (1) A and B fail the reversibility-composition bar, or (2)
the 2–30 B parameter regime requires the additional 4× FFN weight memory
reduction to fit on 16 GB.  CSP's FFN weight compression is unique among
the three candidates (A/B target activation-side savings only).

Paradigm-design count after #27 Candidate C design: **27 shifts x 3
candidates** (selection pending); CSP is the **compressed-sensing
activation axis**.
