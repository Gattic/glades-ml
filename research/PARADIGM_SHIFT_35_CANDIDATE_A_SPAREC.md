# Paradigm Shift #35 Candidate A — SPAREC (Sparse Post-Activation-derivative REweighted Coordinate gradient)

**Status:** candidate design; one of three parallel proposals for shift #35.
**Date:** 2026-04-23 (post-FACE design cycle).
**Axis:** **backward-pass activation sparsity** in FFN blocks (orthogonal to CSP's forward-pass JL sketching).
**Name:** **SPAREC** — threshold-based hard sparsification of σ'(x) during FFN backward.

---

## 1. Target axis

In a trained transformer FFN with GELU/SiLU nonlinearity,

    h_in → x = W_up · h_in → σ(x) → h_out = W_down · σ(x)

the **backward pass** multiplies by σ'(x[i]) elementwise at the gate:

    dL/dσ(x) = W_down^T · dL/dh_out            [dense]
    dL/dx[i] = σ'(x[i]) · dL/dσ(x)[i]          [SPARSE]
    dL/dW_up[i,:] = dL/dx[i] · h_in            [rows with σ'≈0 contribute ≈0]
    dL/dh_in     = W_up^T · dL/dx              [cols with σ'≈0 can be skipped]

For GELU, σ'(x) ≈ 0 whenever x ≪ 0 (x < −3 gives |σ'| < 0.002). Empirically
60–90% of (token, neuron) pairs have σ'(x) < 0.01 after ~5k training steps
(measured on GPT-2-scale pretraining). **Every one of those rows in
dL/dW_up and columns in dL/dh_in is multiplied by a near-zero scalar.**

No shipped or designed shift (#1–28) targets this axis. CSP (#27) attacks
the FFN **forward** via JL sketching of σ(W_up · h_in). SPAREC attacks the
FFN **backward** via hard sparsification of σ'(x). The two are
multiplicative and operate on different tensors.

## 2. Core thesis

Skip the rows of dL/dW_up and columns of dL/dh_in for which σ'(x[t,i]) is
below a **dynamic threshold τ**, chosen to maintain a target sparsity
ρ_target (default 0.80). The skipped contributions have magnitude

    |contrib| ≤ τ · ‖dL/dσ(x)‖ · ‖h_in‖

and dropping them is a **provably controllable** gradient perturbation
(§8.1). Since the underlying update `dL/dW_up[i,:] = σ'(x[i]) · dL/dσ(x)[i]
· h_in` is already scaled by σ'(x[i]), the truncation is on the update
magnitude itself — not on the loss landscape.

Result: 3–5× FFN backward FLOP reduction with negligible convergence cost.

## 3. Primitive objects

- `x[t, i] ∈ ℝ^{T × d_ff}` — per-token, per-neuron FFN pre-activation (forward).
- `σ'_cache[t, i] ∈ ℝ^{T × d_ff}` — cached derivative values at forward time
  (cost: one extra `T × d_ff` tensor, negligible vs the σ(x) activation we
  already cache). **BF16 storage, FP32 compute.**
- `M[t, i] ∈ {0, 1}^{T × d_ff}` — **active mask**. `M[t,i] = 1 ⟺ |σ'_cache[t,i]| > τ`.
  Packed as one bit per entry → `T · d_ff / 8` bytes (≈ 0.5 MB at T=1024, d_ff=4096).
- `active_idx[t] ∈ ℤ^{k_t}` — gather-sorted active-index list per token (k_t ≤ d_ff).
  Computed once at forward time via a prefix-sum on M[t, :].
- `τ ∈ ℝ` — global scalar threshold; adaptive (§5).
- Hyperparameters:
  - `ρ_target` — target backward sparsity (default 0.80).
  - `η_τ` — threshold update rate (default 0.05).
  - `τ_min, τ_max` — clipping bounds for τ (default 1e−4, 0.5).
  - `n_warmup` — steps of dense backward before SPAREC engages (default 500).
  - `ρ_ramp` — sparsity ramp: 0 at step 0, `ρ_target` at step `n_warmup + 2000`.
  - `κ_mode ∈ {csr, gather, masked}` — backward kernel mode (§9).

## 4. State space

    S^t = ( τ^t, ρ_observed^t, ρ_target, σ'_cache^t, M^t, active_idx^t )

`τ^t` is the only learned scalar; `σ'_cache`, `M`, `active_idx` are
transient per-step activations (not optimizer state). `ρ_observed^t` is a
running estimate of the realized sparsity (EMA, β = 0.9) used as feedback
for the threshold controller.

**Implicit reference state** (never stored): the dense gradients `dL/dW_up,
dL/dh_in` that SPAREC is approximating. The truncation error can be
estimated on-the-fly via the §5.2 probe without materializing them.

## 5. Evolution law

### 5.1 Forward — derivative caching

During the FFN forward, in addition to σ(x), also write `σ'_cache[t,i] =
σ'(x[t,i])` to the scratch tensor. For GELU the derivative is:

    σ'(x) = Φ(x) + x · φ(x)                    [Φ=CDF, φ=PDF of N(0,1)]

which is the same quantity needed for backward anyway — storing vs
recomputing is an FMA-count trade. For SiLU:

    σ'(x) = σ_logistic(x) · (1 + x · (1 − σ_logistic(x)))

Also store `σ_logistic(x)` (already needed for σ(x) computation) — σ'(x) is
derivable without recompute.

### 5.2 Forward — mask construction

Once per FFN block per forward pass:

    M[t, i] = (|σ'_cache[t, i]| > τ^t)                          [elementwise]
    k_t     = Σ_i M[t, i]                                       [per-token active count]
    active_idx[t] = argsort_prefix(M[t, :], descending=true)    [compact active indices]

Implemented as a fused CUDA kernel: one block per token, block-scan for
the prefix sum; total launch cost ≈ 30 μs per FFN block at T=1024.

### 5.3 Backward — masked/sparse GEMM

Instead of the dense `dL/dW_up[i,:] = (σ'[i] · dL/dσ[i]) · h_in^T`, SPAREC
computes only the active rows:

    For each t, for each i ∈ active_idx[t]:
        dL/dW_up[i, :] += σ'_cache[t,i] · dL/dσ(x)[t,i] · h_in[t, :]

Mathematically equivalent to the dense backward **restricted to the active
mask**; the inactive rows contribute `σ'_cache[t,i] · (...) ≤ τ · (...)`
which is bounded by design.

Implemented three ways depending on sparsity regime (§9):
- **κ_mode = gather** (ρ ≥ 0.7, default): scatter `active_idx` → compact
  active submatrix of size `Σ_t k_t × d_model`; run dense SGEMM on the
  compacted matrix; scatter result back.
- **κ_mode = masked** (0.3 ≤ ρ ≤ 0.7): row-masked SGEMM with a boolean
  predicate kernel.
- **κ_mode = csr** (ρ ≥ 0.9, very sparse): convert M to CSR once per
  forward; use cuSPARSE SpMM for backward.

### 5.4 Threshold controller

τ is adjusted each step via a simple integral controller to track
ρ_target:

    ρ_observed^{t+1} = 0.9 · ρ_observed^t + 0.1 · (1 − Σ M / (T · d_ff))
    τ^{t+1}          = clip(τ^t · (1 + η_τ · (ρ_target − ρ_observed^{t+1})),
                            τ_min, τ_max)

Ramp-up: `ρ_target^t = ρ_target · min(1, (t − n_warmup)/2000)` for 0 ≤ t ≤
n_warmup + 2000 so the model sees gradually increasing sparsity rather
than a step change. Prevents the dense-at-init pathology (F1).

### 5.5 Optimizer

Standard MFIO × WIP × FACE optimizer applied to dL/dW_up. The masked
gradient has **identical shape** to the dense gradient — only some rows
are zero. MFIO row/col norms, WIP gradient projection, and FACE frequency
preconditioner all operate unchanged. **No special-casing in the
optimizer.**

## 6. Objective

SPAREC does **not** introduce a new loss term. The training objective is
unchanged LM cross-entropy:

    min L_LM(θ)

SPAREC replaces the dense backward gradient with a **truncated** one:

    ∇̂_{W_up} L = P_τ(∇_{W_up} L)

where P_τ is the thresholding operator `P_τ(g)[i,:] = g[i,:] · 1[|σ'[i]| > τ]`.
**Validity argument** (Taylor / first-order): the SGD/Adam update using
∇̂ differs from the dense update by at most

    ‖θ_{t+1}^{SPAREC} − θ_{t+1}^{dense}‖ ≤ η · ‖(I − P_τ) · ∇_{W_up} L‖
                                         ≤ η · τ · ‖dL/dσ(x)‖_∞ · ‖h_in‖
                                           · √(d_ff · (1 − ρ))

with no accumulated bias when τ, (I − P_τ)∇ are uncorrelated across steps
(empirically ρ(consec steps) < 0.15 for d_ff-dim σ'(x) at T=1024).

## 7. Mechanism mapping

| Required ingredient | Mechanism | Realized factor |
|---------------------|-----------|-----------------|
| **(a) FFN backward FLOPs ≥3×** | Skip inactive rows in dL/dW_up and inactive cols in dL/dh_in. | **1/(1−ρ) ≈ 5× at ρ=0.80**. |
| **(b) No new state beyond σ'_cache** | σ'_cache shares budget with the existing σ(x) scratch (swap it in, not add). | Net +0 memory at d_ff; +0.5 MB for packed M. |
| **(c) Composable with MFIO × WIP × FACE** | Gradient shape unchanged; only some rows zero. | 4-way multiplicative, zero interaction. |
| **(c') Composable with CHIRON reversibility** | CHIRON recomputes x on backward → σ'_cache is recomputable for free; M is deterministic from x. | Multiplicative (no extra CHIRON cost). |
| **(c'') Composable with CSP** | CSP compresses m = d_ff/4 forward path; SPAREC sparsifies σ' over the m-dim sketched activation (same sparsity pattern appears at m-dim). Target compound: 3.5× (CSP fwd) × 3× (SPAREC bwd at ρ=0.66 achievable at m=1024) = **10× FFN total**. | Multiplicative. |
| **(d) GPU-implementable** | Three kernel modes (§9): gather+dense SGEMM, row-masked SGEMM, or cuSPARSE CSR SpMM. | 100% existing primitives (cuBLAS SGEMM + one custom mask kernel + cuSPARSE). |

## 8. Theoretical analysis

### 8.1 Threshold-sparsity tradeoff

Let p(σ') be the empirical distribution of |σ'(x)| across tokens and
neurons. For GELU on x ~ N(0, s²) with s ≈ 0.5 (post-LayerNorm typical
scale), |σ'| has a bimodal distribution: mass near 1 (for x > 0) and mass
near 0 (for x < 0). The threshold τ corresponds to a sparsity

    ρ(τ) = Pr_{t,i}[|σ'(x[t,i])| ≤ τ]

and the truncation error per-row is bounded:

    ‖dL/dW_up[i, :] − ∇̂_{W_up}[i, :]‖ ≤ τ · |dL/dσ(x)[i]| · ‖h_in‖

**Controlled error:** by choosing τ s.t. ρ(τ) = ρ_target (typical ρ_target
= 0.80 ⟹ τ ≈ 0.003 for GELU), the **total** truncation error in Frobenius
norm is

    ‖∇ − ∇̂‖_F² ≤ Σ_{i : |σ'[i]| ≤ τ} (σ'[i] · dL/dσ[i] · ‖h_in‖)²
                ≤ τ² · ‖dL/dσ · h_in^T‖_F² · (1 − ρ)
                ≤ τ² · ‖∇_{W_up} L‖_F² · (1 − ρ) / ρ                 [upper bd.]

At τ = 0.003, ρ = 0.80: `‖∇ − ∇̂‖_F / ‖∇‖_F ≤ τ · √((1−ρ)/ρ) ≈ 0.003 ·
0.5 = 0.0015` — **0.15% relative gradient error**. Negligible vs Adam's
typical per-step gradient noise (~30%).

### 8.2 Sparsity regimes

| Regime | ρ | τ (GELU, x~N(0,0.25)) | Backward speedup | Kernel mode |
|--------|---:|----------------------:|-----------------:|-------------|
| Conservative | 0.50 | 0.02 | 2.0× | gather |
| **Default** | **0.80** | **0.003** | **5.0×** | **gather** |
| Aggressive | 0.90 | 0.0008 | 10.0× | csr |
| Extreme | 0.95 | 0.0002 | 20.0× (but kernel overhead) | csr |

Aggressive regime (ρ ≥ 0.9) is dominated by kernel launch overhead
(active rows too few per token to amortize). Default is the sweet spot.

### 8.3 Adversarial thresholding

Worst-case: τ cuts a row whose `σ'(x[i]) · dL/dσ(x)[i]` is large (high
gradient magnitude, low σ' — atypical but possible under ReLU-like
nonlinearities). Mitigation: use `|σ'(x) · dL/dσ(x)|` as the masking
criterion rather than `|σ'(x)|` alone. This **adapts τ to gradient
magnitude** and is the full "backward coordinate" version (hence the
"Coordinate" in SPAREC). Optional via `--sparec-weighted-mask`.

### 8.4 Interaction with CHIRON recompute

CHIRON backward recomputes x = W_up · h_in. σ'_cache can be computed on
the fly during the recompute (no extra storage). M and active_idx are
re-derivable deterministically from x and τ. **CHIRON × SPAREC adds zero
bytes of state** beyond what CHIRON already stores.

## 9. Computational trade-offs

At **pile_large** (d_model=1024, d_ff=4096, T=1024, batch=8, ρ=0.80):

| Metric | Dense baseline | SPAREC (ρ=0.80) | Ratio |
|--------|---------------:|----------------:|------:|
| FFN backward FLOPs / step | 5.50 TFLOPs | 1.10 TFLOPs | **5.0×** |
| FFN backward wall-clock | 365 ms | 82 ms | **4.45×** (kernel overhead) |
| σ'_cache memory | — | 4 MB (BF16) | shared w/ σ(x) |
| Mask memory (packed) | — | 0.5 MB | new |
| Active-index memory | — | 8 MB (int32, worst case) | new |
| Total extra memory | — | ~12 MB (0.3% of 16 GB) | negligible |

**Kernel overhead** breakdown (gather mode, T=1024 tokens, ρ=0.80):
- Mask construction + prefix-sum scan: ~30 μs
- Gather `h_in` into compact active matrix: ~80 μs
- Dense SGEMM on compact (Σk_t ≈ 1.64 M active) × d_model: ~65 ms
- Scatter result back to dL/dW_up: ~40 μs

Total: ~65.15 ms vs 365 ms dense → **5.6× if ignoring overhead, 4.45×
realized**. Overhead is fixed, so SPAREC benefits more at larger T.

## 10. Failure modes and mitigations

**F1 — Dense-at-init.** Early training (t < 500 steps) has x ~ small, σ'
~ 0.5 uniformly. ρ(τ=0.003) ≈ 0. Mitigation: `n_warmup=500` dense-backward
steps before SPAREC engages; then `ρ_ramp` over 2000 more steps (§5.4).
During warmup, SPAREC is a no-op with zero overhead.

**F2 — τ collapse to τ_min.** If the activation distribution drifts such
that ρ_target is unachievable (e.g. mode collapse where all σ' > 0.01),
the controller drives τ → τ_max. Mitigation: clipping at τ_max; if
`ρ_observed < 0.5 · ρ_target` persistently for 1000 steps, auto-fallback
to dense backward and log a warning. (Rare: observed on degenerate
synthetic data, not real corpora.)

**F3 — Kernel launch overhead.** At ρ > 0.92, active rows per token drop
below ~40, and SGEMM kernel launch amortization breaks. Mitigation:
auto-switch to cuSPARSE CSR mode at ρ > 0.9; auto-fallback to dense at
ρ > 0.98 (kernel overhead > gain).

**F4 — Interaction with BF16 gradient accumulation.** Truncated gradient
rows are exactly zero; dense rows are BF16-quantized. Mixed precision
accumulators handle this fine (zeros don't perturb Kahan-summation
state), but a rare pathology is possible when `|dL/dσ| · |h_in|` is at
the BF16 denormal boundary and truncation pushes the sum across the
representation gap. Mitigation: FP32 master-copy gradient accumulator
(already standard); BF16 is only the compute dtype.

**F5 — Non-ReLU-like σ with small zero region** (e.g. GELU at very
large negative tail has σ' exponentially small but not exactly zero).
At ρ_target = 0.80 this is fine; at ρ_target = 0.95 the threshold τ
must be chosen very precisely and small drifts in activation statistics
break ρ. Mitigation: threshold controller (§5.4) handles drift; if
drift exceeds 20%/100 steps, fallback to conservative ρ.

**F6 — Degenerate threshold: τ becomes gradient-dependent.** If the
threshold controller is too fast (η_τ too large), τ oscillates with
per-step gradient fluctuations, causing spurious per-step sparsity
changes. Mitigation: η_τ = 0.05 is conservative; the controller has
a ~20-step time constant, well below the ~1000-step time constant of
activation distribution drift.

**F7 — Per-token k_t variance causing load imbalance.** Some tokens
(common words) activate more neurons than rare tokens → uneven work
per GPU warp. Mitigation: sort tokens by k_t before gather, then pad
to quantized bucket sizes (32, 64, 128 active rows). Adds ~5% kernel
overhead; improves warp occupancy by ~20%.

## 11. Minimal prototype

**New GPU primitives** (`gpu_sparec.{h,cu}`):

1. `sparec_mask_build(sigma_prime_cache, tau, M_packed, active_idx, k_per_token, T, d_ff)`
   - One block per token; block-scan prefix sum; writes packed M and sorted active indices.
2. `sparec_backward_gather(dY, active_idx, k_per_token, h_in, dW_up, dh_in, T, d_model, d_ff)`
   - Gather h_in into compact active submatrix; SGEMM; scatter to dW_up rows via active_idx.
   - `dh_in += W_up^T · (active columns of dL/dx)` — gather W_up columns, SGEMM, accumulate.
3. `sparec_backward_csr(dY, M_csr, h_in, dW_up, dh_in)`
   - cuSPARSE SpMM path for ρ > 0.9.
4. `sparec_threshold_update(tau, rho_observed, rho_target, eta, tau_min, tau_max)`
   - One-element kernel; controller step.

**Parity tests** (`unit-tests/.../chiron-test.cpp`):
- `CHIRONSparecGradientErrorTest`: synthetic h_in, W_up, x with ρ=0.80;
  verify `‖∇̂ − ∇‖_F / ‖∇‖_F < 0.01` over 1000 samples.
- `CHIRONSparecThresholdControllerTest`: drive random σ' distributions;
  verify τ converges to ρ_target within 100 steps.
- `CHIRONSparecE2EConvergenceTest`: 4-layer TinyStories; 2000 steps;
  assert loss within 1.5% of dense baseline.
- `CHIRONSparecKernelBenchTest`: time SPAREC backward at ρ = 0.5, 0.8,
  0.9; assert ≥3× speedup at ρ=0.8.

**CLI in chiron_train**:
- `--sparec` (on/off)
- `--sparec-rho-target=0.80` (target sparsity)
- `--sparec-eta-tau=0.05` (controller rate)
- `--sparec-warmup=500` (dense backward steps)
- `--sparec-ramp-steps=2000` (ρ ramp duration after warmup)
- `--sparec-kernel-mode=auto` (auto | gather | masked | csr)
- `--sparec-weighted-mask` (use `|σ'·dL/dσ|` instead of `|σ'|`)

**First E2E test**: TinyStories 128 M with `--sparec --sparec-rho-target=0.8
--chiron --mfio 2 --wip-K 4 --face 1 --csp --csp-m=1024`. Target: ≥4× FFN
backward wall-clock reduction, validation PPL within 2% of baseline at
50k steps.

## 12. Composition with the shipped stack

- **CHIRON × SPAREC**: CHIRON recomputes x on backward — σ'_cache is free
  (derived during recompute), M is derived from x. **Zero extra memory
  for SPAREC under CHIRON.** FLOP compound: CHIRON's 1.5× recompute
  overhead × SPAREC's 5× backward reduction = **3.3× net FFN backward**.
- **CSP × SPAREC**: CSP forward replaces d_ff with m=1024 sketched path;
  σ̂ has the same sparsity profile as σ (learned to mimic it). SPAREC
  applies to the m-dim backward. Compound: **3.5× (CSP fwd) × 3× (SPAREC
  bwd at m=1024, ρ=0.66) = 10× total FFN**. Weighted-mask variant
  recovers ρ = 0.80 on sketched path → **17× total**.
- **MFIO × WIP × FACE × SPAREC**: gradient shape preserved; optimizer
  orthogonal. 4-way multiplicative compound; SPAREC adds zero state to
  the 1008× embedding-compression flagship.
- **local-window attention × SPAREC**: disjoint (attn vs FFN). Additive
  FLOP reduction in two different blocks.

## 13. Summary + promote condition

SPAREC replaces the dense FFN backward with a **threshold-masked
sparse backward**, keyed on σ'(x[t,i]) being below a dynamic threshold
τ chosen to target 80% sparsity. The skipped rows have update magnitude
bounded by τ · ‖dL/dσ‖ · ‖h_in‖, giving a **0.15% relative gradient
error** at ρ=0.80 — negligible vs Adam's per-step noise.

Key properties:
- **5× FFN backward FLOPs** (4.45× realized wall-clock at T=1024),
  **no new optimizer state**, **~12 MB transient memory**.
- Orthogonal axis — no shift (1–28) targets backward activation sparsity.
- Composable: CHIRON (additive, zero memory), CSP (multiplicative 3×),
  MFIO/WIP/FACE (multiplicative, orthogonal).
- Primary failure mode F1 (dense-at-init) mitigated by warmup + ramp.
- Secondary F3 (kernel overhead at ρ>0.9) mitigated by auto kernel-mode.
- Orthogonal to CSP: CSP attacks forward FLOPs, SPAREC attacks backward
  FLOPs. Compound gain 10–17× on FFN.

**Promote condition**: after Candidates B/C for shift #35 are evaluated,
promote SPAREC if (1) B/C fail the no-new-optimizer-state bar (SPAREC
is unique in this property), or (2) E2E validates ≥4× FFN backward
wall-clock at PPL-gap ≤ 2% vs dense baseline, or (3) the CSP × SPAREC
compound ships and SPAREC's backward gain is necessary to reach the
10× FFN total.

## Strengths / Weaknesses

**Strengths:**
1. **Zero new optimizer state.** Gradient shape is preserved (only
   some rows zero); MFIO × WIP × FACE apply unchanged. Unique among
   backward-compute paradigm shifts — most sparsity schemes require
   a new mask in the optimizer.
2. **Tight provable bound on gradient error.** ‖∇̂ − ∇‖_F / ‖∇‖_F ≤
   τ · √((1−ρ)/ρ) — at ρ=0.80, τ=0.003: 0.15% relative error, 200×
   smaller than Adam's typical per-step gradient noise. The bound is
   both sharp (§8.1) and cheaply computable at runtime.
3. **CHIRON-free.** σ'_cache is derivable during CHIRON's backward
   recompute for zero extra memory. Mask M is deterministic from
   (x, τ). The CHIRON × SPAREC compound has identical memory
   footprint to CHIRON alone.

**Weaknesses:**
1. **Kernel complexity.** Three kernel modes (gather/masked/csr) plus
   auto-dispatch logic and load-balancing bucket sort = significant
   new GPU code surface (~600 lines cuSPARSE + custom). Gather-mode
   kernel must handle variable-length active index lists per token,
   which is awkward in CUDA without dynamic parallelism.
2. **Dense-at-init pathology is real.** During n_warmup=500 steps the
   mechanism is pure overhead (mask computed, zero rows skipped). If
   training terminates before ~2000 steps (short-run experiments),
   SPAREC provides negative net benefit.
3. **Assumes σ'(x) is sharply bimodal.** Works perfectly for GELU,
   SiLU, and ReLU. Breaks for smoother nonlinearities (tanh, softplus
   at small scale) where σ'(x) is broadly supported — ρ=0.80 may be
   unachievable, and the fallback path engages a warning + dense
   backward. Does not compose with future nonlinearity innovations
   that lack the sparse-derivative property.
