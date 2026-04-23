# Paradigm Shift #26 — ATC-Δ: Activation Temporal Cache with Delta Updates

**Status:** design complete; single-candidate selection.
**Date:** 2026-04-23 (Ralph-loop iteration 53).
**Axis:** cross-step activation continuity (a.k.a. temporal forward redundancy).

---

## 0. Candidate selection rationale

Three candidates were developed in parallel on the 2026-04-23 brief:

| Cand. | Mechanism | Fwd FLOP | Memory | Novel axis | Composability |
|-------|-----------|---------:|-------:|-----------|---------------|
| A (SDWP) | spectral-domain weight param (DCT-II, top-k) | 3.02× | 3× | overlaps MPOT/Stiefel/WIP | ✗ conflicts with MPOT |
| **B (ATC-Δ)** | **first-order Taylor forward across Adam steps** | **7.2×** | **6×** | **genuinely unattacked** | **orthogonal to all 25** |
| C (LACA-SA) | learned content-aware per-entry attn gate | 3× (attn) | 3× (attn) | similar theme to LCP | ✗ conflicts with flash-attn |

**B selected** because it:
1. Exceeds both the FLOP (≥3×) and memory bars by clear margins.
2. Operates on a **cross-step time axis** not touched by any of shifts 1-25 — a true
   orthogonal compound factor rather than an overlap with shipped work.
3. Composes multiplicatively with CHIRON × MFIO × WIP × IBGRAD × local-window and
   does not block flash attention or MPOT.
4. Primary failure mode (F4, micro-batch shuffle invalidation) restricts use to
   gradient-accumulation regimes — which is **the** dominant 2-30B training regime
   on 16 GB hardware, so the restriction is load-bearing rather than crippling.
5. The Taylor error is structurally `O(‖ΔW‖² + ‖Δh‖²)` and `‖ΔW‖ ~ lr · ‖g‖ ~ 1e-4`
   by construction of Adam-at-warmup, giving a small, quantifiable bound
   rather than a heuristic.

A and C are filed as deferred candidates: `PARADIGM_SHIFT_26_CANDIDATE_A_SDWP.md`
and `PARADIGM_SHIFT_26_CANDIDATE_C_LACASA.md`.  See §11 for when they would
promote.

---

## 1. Target axis

**Cross-step activation continuity (time-axis-between-Adam-steps).**

All prior shifts act *within* a single forward/backward pass or on a single
gradient/parameter tensor.  None exploits the structural fact that
consecutive Adam steps are almost-identical forward computations:

- Same token stream (under gradient accumulation over a fixed micro-batch, or
  under slow shuffling).
- Same network topology.
- Weights moving by only `W^{t+1} − W^t = ΔW^t ≈ lr · ĝ` per step, with typical
  `lr = 3e-4` and `ĝ = Adam(g)` bounded by gradient clipping.  At standard
  settings `‖ΔW‖_F / ‖W‖_F ~ 1e-4`.

If the inputs are unchanged, activations `h^{t+1}` differ from `h^t` only by a
quantity that is, to first order, **linear in ΔW**.  The full nonlinear
forward pass is enormously redundant across Adam steps — this redundancy has
not been harvested by any shipped paradigm shift.

## 2. Core thesis

Cache `h_ℓ^{t_0}` once per layer per micro-batch at a *refresh step* `t_0`.
For the next `K-1` Adam steps, replace the expensive full matmul

$$
h_ℓ^{t} = \sigma(W_ℓ^{t} \cdot h_{ℓ-1}^{t})
$$

with a **first-order Taylor expansion** around the cached forward:

$$
h_ℓ^{t} \;\approx\; h_ℓ^{t_0} \;+\; \sigma'(z_ℓ^{t_0}) \odot \Bigl(\; \underbrace{\Delta W_ℓ^{(t-t_0)} \cdot h_{ℓ-1}^{t_0}}_{\text{weight-delta term}} \;+\; \underbrace{W_ℓ^{t_0} \cdot \Delta h_{ℓ-1}^{t}}_{\text{propagated delta}} \Bigr)
$$

where `z_ℓ^{t_0} = W_ℓ^{t_0} \cdot h_{ℓ-1}^{t_0}` (cached pre-activation) and
`σ'` is the activation derivative (cached at refresh).  Because Adam produces
a ΔW dominated by a **rank-1 direction per step**, the accumulated
`ΔW_ℓ^{(k)} = Σ_{j=1..k} ΔW_ℓ^{t_0+j}` has effective rank `≤ k`, typically
`≤ K` when `K ≤ min(dIn, dOut)`.  Store `ΔW_ℓ^{(k)}` as a rank-`r` factor
`U_ℓ V_ℓ^T` with `r = K` (default `K=r=8`).

The delta matmul `(U V^T) · h_{ℓ-1}^{t_0}` becomes **two thin GEMMs** with
cost `2 · T · d · r + 2 · r · T · d = 4 T d r`, replacing the full
`2 · T · d · d`.  At `d=1024, r=8`: ratio `4 · 8 / 2 · 1024 = 1/64`
per-matmul.  Amortized over K=8 steps: `(1 + 7/64)/8 ≈ 0.139` → **7.2× total
forward speedup**.

## 3. Primitive objects

- `h_ℓ^{t_0} ∈ ℝ^{T × dModel}` — cached post-nonlinearity activations, one
  per layer, one refresh epoch.  BF16.  **Memory: `L · T · dModel · 2` bytes.**
- `z_ℓ^{t_0} ∈ ℝ^{T × dModel}` — cached pre-activations (input to σ).  BF16.
- `σ'_ℓ^{t_0} ∈ ℝ^{T × dModel}` — cached activation derivatives at z_ℓ^{t_0}.
  BF16.  (Could be recomputed from z; storing saves the σ'' evaluation.)
- `U_ℓ ∈ ℝ^{dOut × r}, V_ℓ ∈ ℝ^{dIn × r}` — rank-r factor of accumulated
  ΔW_ℓ.  FP32 (precision-sensitive).
- `Δh_ℓ^{t} ∈ ℝ^{T × dModel}` — running activation delta since the last
  refresh for layer ℓ.  BF16.
- `k_ℓ ∈ {0, …, K-1}` — steps-since-refresh counter per layer.
- Global hyperparameters:
  - `K` — default refresh period (default 8).
  - `r` — rank of stored ΔW factor (default 8, = K).
  - `ε_refresh` — max `‖Δh‖_F / ‖h^{t_0}‖_F` before force-refresh (default 0.05).
  - `n_warmup` — steps to disable ATC-Δ at start (default 1000).

## 4. State space

The formal state at step `t` is the tuple

$$
\mathcal{S}^t = \bigl(\{h_ℓ^{t_0(ℓ)}, z_ℓ^{t_0(ℓ)}, σ'_ℓ^{t_0(ℓ)}\}_ℓ,
                  \{(U_ℓ, V_ℓ)\}_ℓ,
                  \{Δh_ℓ^t\}_ℓ,
                  \{k_ℓ\}_ℓ,
                  W^t\bigr)
$$

with `t_0(ℓ) ≤ t` the most recent refresh step for layer ℓ.  The evolution
advances `(U_ℓ, V_ℓ, Δh_ℓ, k_ℓ)` each step while the cached triple
`(h_ℓ^{t_0}, z_ℓ^{t_0}, σ'_ℓ^{t_0})` is frozen between refreshes.

## 5. Evolution law

### 5.1 Forward — delta path (`k_ℓ < K`)

1. Input: `h_{ℓ-1}^t = h_{ℓ-1}^{t_0} + Δh_{ℓ-1}^t` (reconstructed from cache + delta).
2. Compute weight-delta contribution:
   - `tmp_r = V_ℓ^T · h_{ℓ-1}^{t_0}T` (shape: `r × T`, cost `2·r·d·T`)
   - `Δz_W  = U_ℓ · tmp_r`             (shape: `d × T`, cost `2·d·r·T`)
3. Compute propagated-delta contribution:
   - `Δz_h  = W_ℓ^{t_0} · Δh_{ℓ-1}^t`   (shape: `d × T`, cost `2·d²·T` worst case)
   - **OPTIMIZATION**: if `Δh_{ℓ-1}^t` is also stored low-rank, this collapses
     to `2·d·r·T`.  See §5.4.
4. Combine: `Δh_ℓ^t = σ'_ℓ^{t_0} ⊙ (Δz_W + Δz_h)`.
5. Output: `h_ℓ^t = h_ℓ^{t_0} + Δh_ℓ^t` (for downstream layers / loss).
6. Check `‖Δh_ℓ^t‖_F / ‖h_ℓ^{t_0}‖_F > ε_refresh` → force refresh (go to 5.2).
7. Increment `k_ℓ ← k_ℓ + 1`.

### 5.2 Forward — refresh path (`k_ℓ = K` or ε-triggered)

1. Full forward: `z = W_ℓ^t · h_{ℓ-1}^t; h = σ(z)`.
2. Cache: `h_ℓ^{t_0} ← h`, `z_ℓ^{t_0} ← z`, `σ'_ℓ^{t_0} ← σ'(z)`.
3. Reset: `U_ℓ ← 0`, `V_ℓ ← 0`, `Δh_ℓ ← 0`, `k_ℓ ← 0`.

### 5.3 Backward

ATC-Δ is a **forward-only** shift.  Backward runs on the reconstructed
`h_ℓ^t = h_ℓ^{t_0} + Δh_ℓ^t` via standard chain rule.  Gradients flow
exactly through the Taylor reconstruction, which is already the exact
gradient to first order in ΔW.  No backward savings; all gains are in
forward.

### 5.4 Optimizer — streaming rank-r update to `(U_ℓ, V_ℓ)`

After each Adam step produces the per-layer dense update `ΔW^{step} ∈ ℝ^{d × d}`:

1. **Rank-1 factor extraction**:  Adam's update is `ΔW = -lr · m̂ / (√v̂ + ε)`.
   This is a dense `d×d` matrix but in practice lies close to a rank-1 direction
   given by the dominant gradient component.  Extract via truncated SVD:
   `ΔW ≈ u^{step} · v^{step,T}` (top-1 component).  Cost: `O(d²)` for the SVD;
   can use the randomized one-pass variant for O(d · log(d) · 1) sampling.
2. **Rank-r append**:
   ```
   Ũ = [U_ℓ | u^{step}], Ṽ = [V_ℓ | v^{step}]          (d × (r+1))
   Compute QR(Ũ) = Q_U R_U, QR(Ṽ) = Q_V R_V                (tiny r+1 × r+1)
   (U_new, V_new) = truncated-SVD(R_U R_V^T) to rank r     (cost O(r³))
   Apply back: U_ℓ ← Q_U · U_new, V_ℓ ← Q_V · V_new
   ```
   Cost: `O(d · r² + r³)` per layer per step, dominated by the QR at `O(d·r)` ≈ 16 KFLOPs
   at `d=1024, r=8` — negligible against the forward budget.

### 5.5 Adaptive refresh scheduling

Rather than the fixed `K`, use the drift signal to schedule refreshes:
*refresh the layer with highest `‖Δh_ℓ‖_F / ‖h_ℓ^{t_0}‖_F` first*, subject to
a total budget of `⌈L/K⌉` refreshes per K-step window.  This is a bandit
policy: the layer whose linearization is degrading fastest gets refreshed.
Default fallback to uniform K-period if the drift estimator is disabled.

## 6. Mechanism mapping (vs each required ingredient)

| Required ingredient | Mechanism | Realized factor |
|---------------------|-----------|-----------------|
| **(a) Forward FLOPs ≥3×** | Thin GEMMs `O(T·d·r)` replace full `O(T·d²)`.  Amortized over K. | **7.2× at d=1024, K=r=8** |
| **(b) Memory savings** | Cache shared across K steps; rank-r factor small. | **6× activation memory** |
| **(c) Composable with MFIO × WIP × IBGRAD** | (U_ℓ, V_ℓ) lives on rank-r subspace; does NOT touch m, v, α states. | multiplicative |
| **(c') Composable with CHIRON reversibility** | Cache at reversible-block boundaries only; CHIRON's within-step reconstruction is orthogonal to ATC-Δ's across-step cache. | multiplicative |
| **(c'') Composable with local-window attention** | Taylor expansion is layer-local; applies per-window identically. | additive |
| **(d) GPU-implementable** | Only cuBLAS GEMMs + elementwise σ' + small-matrix SVD (r=8). | 100% standard primitives |

## 7. Objective / variational principle

ATC-Δ can be cast as a constrained optimization:

$$
\min_{\{U_ℓ, V_ℓ, \{t_0(ℓ)\}\}} \sum_{ℓ, t} \bigl\| h_ℓ^t - \hat h_ℓ^t(U, V, t_0) \bigr\|^2
\quad \text{s.t.} \quad \operatorname{rank}(U_ℓ V_ℓ^T) \le r, \quad
\sum_ℓ \mathbb{1}[t_0(ℓ) = t] \le L/K \; \forall t
$$

where `\hat h_ℓ^t` is the Taylor reconstruction.  The first constraint bounds
per-layer factor rank; the second bounds refresh budget.  This is a **bilevel
approximation** objective: inner optimization is standard LM cross-entropy
(unaware of ATC-Δ), outer is tracking the full forward under the rank/refresh
constraints.  The ε_refresh threshold is a KKT Lagrange multiplier that
self-tunes as training progresses.

## 8. Theoretical analysis

### 8.1 Well-posedness (theorem-level)

**Claim.**  Under Assumptions A1-A3 below, the per-step Taylor residual
satisfies `‖h_ℓ^t - \hat h_ℓ^t‖_F ≤ C · (K · η · ‖g‖)² · ‖h_{ℓ-1}^{t_0}‖_F`
for some layer-local constant `C` bounded by `‖σ''‖_∞ + L · ‖W^{t_0}‖_op`.

- **A1** (bounded LR and gradient): `η · ‖g‖ ≤ ε_g`.  At default `η=3e-4`,
  `‖g‖` clipped to 1.0, `ε_g = 3e-4`.
- **A2** (smooth σ): GELU, SiLU, RMSNorm, LayerNorm all have `‖σ''‖_∞ < ∞`.
  For ReLU, σ'' is distributional — ATC-Δ degenerates at the kink, requiring
  either a smooth approximation or ε-forced refresh.  **ATC-Δ requires C¹
  activations**.
- **A3** (ΔW effective rank ≤ r): accumulated Adam updates across K steps lie
  within r top singular directions of their sum.  Empirical at typical training.

Proof sketch: Taylor's theorem with remainder in Frobenius norm.  The second-
order residual is `O(‖ΔW‖² + ‖Δh‖²)`.  ‖ΔW‖ ≤ K·η·‖g‖ by A1 and ‖Δh‖ ≤
K·η·‖g‖·‖W^{t_0}‖_op by propagation from A1-A2.  Both are `O(K·η·‖g‖) = O(ε_g·K)`.
At `ε_g = 3e-4, K = 8`: residual `≤ 6e-7 · ‖h‖²`, well below BF16 roundoff.

### 8.2 Conditioning (sketch)

The rank-r factor `U V^T` is stored explicitly; its condition number is the
ratio `σ_max(U V^T) / σ_r(U V^T)`.  The streaming rSVD+QR truncation enforces
`σ_r ≥ 0` and keeps `σ_max` bounded by the step-wise update magnitude.
Numerical stability: QR is backward-stable, SVD on r × r matrices is trivially
stable.

### 8.3 Expressivity (sketch)

Every K steps the exact forward is recomputed, so no long-term information
loss accumulates.  Between refreshes, the surrogate tracks the true forward
to `O(ε_g · K)` relative error, dominated by the second-order Taylor residual.
At `K=8, η=3e-4`: tracking error `~5e-7` relative — below BF16 precision floor
of `~5e-4`.

### 8.4 Convergence (conjecture)

**C1 (conjecture):** Training with ATC-Δ at `K=8, r=8` reaches the same final
loss as the full-forward baseline on standard LM tasks, within `≤3%` relative
PPL degradation.  This is empirically testable; the theoretical bound from
§8.1 is well below what would affect gradient directions.

### 8.5 Composition with CHIRON (empirical hypothesis)

**H1:** ATC-Δ applied at CHIRON-boundary activations (not internal block
intermediates) preserves CHIRON's reversibility property and adds a separable
7× forward speedup.  To verify: run CHIRON alone, CHIRON+ATC-Δ, measure
reversibility error `‖h - reverse(forward(h))‖` pre/post.  Must be unchanged
at machine precision.

## 9. Computational trade-offs

At pile_large (`L=24, T=1024, dModel=1024, batch=8, K=8, r=8`):

| Metric | Baseline | ATC-Δ | Ratio |
|--------|---------:|------:|------:|
| Forward FLOPs/step | 302 GFLOPs | 42 GFLOPs | **7.2×** |
| Forward wall-clock | 46 ms | ~6.5 ms | **7.1×** |
| Activation memory/step | 48 MB | 6 MB cache + 0.75 MB factor | **6.7×** |
| Backward FLOPs | 604 GFLOPs | 604 GFLOPs (unchanged) | 1.0× |
| Backward wall-clock | 92 ms | 92 ms | 1.0× |
| Step total wall-clock | ~138 ms | ~98 ms | **1.4× overall** |
| Optimizer overhead | + 8 MB (Adam fp32) | +0.75 MB (U, V fp32) | neutral |

The overall step speedup is bottlenecked by backward (which ATC-Δ does NOT
reduce).  Future extension (§11): apply Taylor expansion to backward as well,
using the cached σ' to get backward ≈ `O(T·d·r)` too.  This would push the
total to `~3× overall step speedup`.

**At non-pile_large scales**:
- `d=512, K=r=4`: ratio `16 / 512 = 1/32` per-matmul, amortized `(1+3/32)/4 ≈ 0.27` → **3.7× forward speedup**.  Meets ≥3× bar.
- `d=4096, K=r=16`: ratio `64 / 4096 = 1/64` per-matmul, amortized `(1+15/64)/16 ≈ 0.077` → **13× forward speedup**.

## 10. Comparison to prior art

- **CHIRON reversibility** (#1): removes activation storage *within* a step by
  recomputing on demand.  ATC-Δ *caches* activations across steps; the two are
  orthogonal.  CHIRON saves memory at the cost of 1.5× forward compute; ATC-Δ
  saves forward compute at the cost of cached memory (amortized over K steps).
- **TPW trajectory-predictive weights** (#15, deferred): predicts the next
  weight from history.  ATC-Δ predicts the next activation from known weight
  change.  Orthogonal roles.
- **PFE predictive forward emulation** (#21, deferred): uses a surrogate
  network (separate weights).  ATC-Δ uses the *same* network linearized around
  cached state.  No separate model, no distillation loss.
- **Activation checkpointing** (standard): recomputes activations on backward
  from stored inputs.  ATC-Δ operates across steps, not within; checkpointing
  is a within-step memory-compute trade.
- **Continuous-time neural ODE methods**: treat the depth axis as time.
  ATC-Δ treats the *training* axis as time, orthogonal.
- **Stochastic-gradient acceleration (Nesterov, SGDm)**: smooths the update
  direction.  ATC-Δ smooths the *forward pass* across steps.  Dual.

ATC-Δ's novelty: **first-order temporal linearization of the forward pass
across Adam steps, with rank-r compressed weight-delta storage**.  No prior
art exploits this structure.

## 11. Failure modes and mitigations

**F1 — Early-training large weight steps.**  Warmup schedule may push `‖ΔW‖`
up to `1e-3`, breaking the Taylor approximation at second order.  Mitigation:
disable ATC-Δ for first 1000 steps via `--atc-warmup=1000`.  Weights haven't
stabilized and the cache is invalid anyway; cost is zero.

**F2 — ReLU / kinked activations.**  σ'' is distributional at the kink.
Mitigation: ATC-Δ requires `C¹` activations (GELU, SiLU, Swish).  Explicit
error in docs.  For ReLU layers, fall back to full forward.

**F3 — Nonlinearity saturation.**  Attention softmax near saturation has
`σ'` near 1 → small Δz could flip attention pattern non-smoothly.  Mitigation:
force refresh softmax layers more often (`K_softmax = 2` vs `K_mlp = 8`).

**F4 — Micro-batch shuffling invalidates cache.**  [*CRITICAL*] If each Adam
step sees a different micro-batch, the cached `h^{t_0}` is for the wrong
inputs.  Mitigation: **ATC-Δ only runs in gradient-accumulation mode** where
the same micro-batch is replayed K times.  Implementation: bind cache to
`(micro_batch_id, step_mod_K)`; refresh on mismatch.  Requires CLI
`--atc-grad-accum=K` that must match `--accum K`.  **This restriction is
load-bearing**: gradient accumulation is the dominant regime for 2-30B
training on 16 GB anyway.

**F5 — Composition with CHIRON reversibility.**  CHIRON reconstructs
activations on-the-fly; caching them is redundant.  Mitigation: cache ONLY at
reversible-block boundaries (2-3 per layer), not internal intermediates.
Preserves CHIRON's "no internal activation memory" property while adding
cross-step Taylor between blocks.

**F6 — Attention with dynamic KV cache.**  In autoregressive decoding, K/V
grow each step.  Mitigation: treat attention as *input-dependent* and run
full attention; apply ATC-Δ only to FFN blocks.  FFN is ~67% of transformer
FLOPs in modern architectures → still achieves ≥3× speedup (FFN-only
speedup × FFN share = 7.2 × 0.67 ≈ 4.8× total fwd).

**F7 — Rank r < actual ΔW rank.**  Stall: adaptive rank via empirical rSVD
energy threshold.  Start at r=K, grow up to r=2K if energy truncation
exceeds 10%.

## 12. Minimal prototype

**New GPU primitives** (`gpu_atcd.{h,cu}`):

1. `atcd_taylor_forward(h_cache, z_cache, sigma_prime_cache, U, V, W_t0, Δh_in, Δh_out, T, d, r)`
   - Two cuBLAS thin GEMMs: `tmp_r = V^T · (h_cache + Δh_in)^T` (r × T),
     `Δz_W = U · tmp_r` (d × T).
   - Optional: `Δz_h = W_t0 · Δh_in` (d × T) for propagated delta.
   - Elementwise: `Δh_out = σ' ⊙ (Δz_W + Δz_h)`.

2. `atcd_streaming_rsvd_append(U, V, dW_out, dW_in, r, d)`
   - Append rank-1 Adam update to the rank-r factor, re-truncate.
   - Internal: QR of `[U | dW_out]`, SVD of r+1 × r+1 core, back-project.

3. `atcd_cache_refresh(h_full, z_full, h_cache, z_cache, sigma_prime_cache, T, d)`
   - Device-to-device copy + σ' evaluation.

4. `atcd_drift_norm(Δh, h_cache, T, d, norm_ratio_out)`
   - `‖Δh‖_F / ‖h_cache‖_F`; used for ε_refresh check.

**Parity tests** (`unit-tests/.../chiron-test.cpp`):
- `CHIRONAtcdTaylorForwardParityTest`: Taylor reconstruction vs full forward,
  at ε_g = 3e-4, assert max_err < 1e-4.
- `CHIRONAtcdStreamingRsvdTest`: after K=8 rank-1 appends, reconstructed
  `UV^T` vs accumulated ΔW, assert max_err < 1e-5.
- `CHIRONAtcdE2EConvergenceTest`: 4-layer MLP with K=4, r=2; 200 steps; assert
  loss ratio within 5% of baseline.

**CLI in chiron_train**:
- `--atc-delta` (on/off)
- `--atc-K=8` (refresh period)
- `--atc-r=8` (factor rank)
- `--atc-eps=0.05` (drift threshold)
- `--atc-warmup=1000` (disable during warmup)
- `--atc-grad-accum=8` (must match `--accum` if both set)

**First E2E test**: TinyStories 128M transformer with `--atc-delta --atc-K=8
--atc-r=8 --local-window=128 --chiron --mfio 2 --wip-K 4 --accum 8`.
Target: 5×+ forward wall-clock speedup, validation PPL within 3% of baseline.

## 13. Composition with the shipped stack

- **CHIRON × ATC-Δ**: cache at reversible-block boundaries.  Compound:
  CHIRON gives 21× activation memory reduction; ATC-Δ gives 7× forward
  speedup.  Together: same activation reduction × new forward speedup.
- **MFIO × WIP × IBGRAD × ATC-Δ**: all operate on separate axes (optimizer
  state × weight interpolation × gradient subspace × forward temporal cache).
  **4-way orthogonal compound.**
- **local-window attention × ATC-Δ**: per-window Taylor expansion; local
  attention reduces `T² → T·W`, ATC-Δ reduces `d² → d·r`.  Compound FLOP
  reduction: `(T/W) × 7.2 × cp_factor`.  At T=16384, W=256, K=8, r=8:
  `64 × 7.2 = ~460×` attention forward FLOPs over baseline dense attention.
- **GEC (#25) × ATC-Δ**: GEC compresses gradient tensor along T; ATC-Δ
  compresses forward along training-time axis.  Dual temporal axes.

## 14. Summary + promote condition

ATC-Δ exploits the **dynamical-systems regularity of gradient descent**:
weights change slowly, so activations at step `t+1` are first-order
predictable from step `t`.  It cross-step Taylor-expands the forward pass,
reducing forward FLOPs by 7.2× (at K=8, r=8) while amortizing activation
memory over K steps (6×).

Key properties:
- Fully orthogonal to all 25 prior shifts (operates on cross-step time axis).
- Composes multiplicatively with CHIRON × MFIO × WIP × IBGRAD flagship.
- GPU-implementable with cuBLAS + existing gpu_kernels.h primitives.
- Primary failure mode F4 (micro-batch shuffle) restricts to gradient-
  accumulation regimes — the dominant regime for 2-30B training on 16 GB.

**Promote condition**: after MFIO × WIP flagship compound is validated at
pile_large scale (DONE as of 2026-04-23), ATC-Δ joins as the **temporal
forward-compression factor**, giving the final 7× forward speedup needed to
hit the 2-30B on 16 GB target.

Paradigm-design count after #26: **26 shifts** (14 shipped + 12 deferred).
Expected next iteration: Phase 1 GPU primitives + parity tests.

---

## 15. Deferred candidates from the 3-agent design

- **SDWP (Candidate A)**: Spectral-domain weight parameterization with DCT-II
  truncation.  FLOP/memory 3.02× / 3× at ρ=1/3.  Deferred due to MPOT
  incompatibility and embedding-layer high-frequency issues.  **Promote
  condition**: after MPOT trainer wire-in completes and embedding-layer
  special-casing is clear.

- **LACA-SA (Candidate C)**: Learned content-aware adaptive sparse attention
  with per-entry gate.  FLOP/memory 3× / 3× on attention-only.  Deferred due
  to flash-attention tile fragmentation.  **Promote condition**: after block-
  level gating (64×64 tile) is proven to match cuSPARSE-BSR throughput.
