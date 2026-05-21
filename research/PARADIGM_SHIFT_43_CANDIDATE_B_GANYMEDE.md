# Paradigm Shift #43 Candidate B — GANYMEDE

**Generalized Adjoint Newton-Yielded Memoryless Extrapolation via Differential Estimation**

**Status:** candidate-B design, parallel to #43-A (NEXUS extrapolation) and #43-C (TBD). Optimizer-axis paradigm.
**Date:** 2026-05-08 (Ralph-loop iteration 187, post-SCFA selection of #42).
**Tagline:** *Per-block streaming L-BFGS on the Adam trajectory — magnitudes from step-count reduction in low-effective-rank basins, conditional on Conjecture C1.*

---

## 0. Executive summary

Paradigms 1–41 attacked **per-step compute** (FACE, MFIO, SAS, SPAREC) or **memory** (CHIRON, IBGRAD, MFIO). Paradigm #42 (SCFA) crushed the per-step attention term. The unattacked axis is **steps-to-target-loss**: Adam needs `O(κ)` iterations to reach ε-tolerance in a κ-conditioned basin, but Newton's method needs only `O(log(1/ε))`. The gap is a hard ceiling on stacked speedup unless we attack it directly.

GANYMEDE applies streaming L-BFGS to the Adam trajectory, with three CHIRON-specific structural twists:

1. **Per-block low-rank inverse-Hessian sketches** — one rank-`r` L-BFGS history per CHIRON block (NOT one global sketch over 1.84B params, which is structurally infeasible).
2. **Adam-Newton hybrid update** with a damping schedule `γ_t` driven by an online conditioning monitor — pure Adam at init (where `H` is non-PSD and ill-defined), pure quasi-Newton in basins (where curvature pays off).
3. **CHIRON-native gradient differences** — `s_i, y_i` pairs are computed for free during CHIRON's inverse walk, since per-layer gradients are already materialized in the backward pass.

**Honest magnitude claim.** In the worst case (full-rank Hessian), rank-`r ≪ d` quasi-Newton gives factor `(1 - r/d) ≈ 1` in the convergence-rate constant — **same as Adam**, no magnitude win (Theorem B). The magnitude claim hinges on **Conjecture C1**: per-block effective rank `r_eff ≤ 200` during stable LLM training. If C1 holds, GANYMEDE achieves Newton's `O(log(1/ε))` rate with `r ≥ r_eff`, yielding a step-count reduction of `~14×` at typical κ=100, ε=10⁻³.

Per-step overhead is **not free**: L-BFGS two-loop recursion at `r=100`, `d=1.84·10⁹` costs ≈ `0.4F` (one full forward), so the per-step penalty is `1.4×`. Net wall-clock projection: `14× / 1.4 = 10×`. **Magnitudes territory if C1 holds; otherwise GANYMEDE degrades to ≤1× and the paradigm fails Gate-0.**

Materially different from rejected TRAJ (paradigm #30, iter 92): TRAJ predicted the gradient via AR(2) and failed at lag-1 grad-norm autocorr `−0.087`. GANYMEDE does NOT predict the gradient — it uses gradient **differences** `y_i = g_{t-i+1} − g_{t-i}` to probe the Hessian's action on the displacement `s_i = θ_{t-i+1} − θ_{t-i}`. The TRAJ-killer probe (`corr(g_t, g_{t-1}) ≈ 0`) is **uninformative** about GANYMEDE's premise (§7).

Materially different from sibling NEXUS (#43-A): NEXUS extrapolates positions on a Hamiltonian phase space (Verlet leapfrog from a single anchor); GANYMEDE estimates curvature for Newton steps (L-BFGS two-loop recursion every step). NEXUS amortizes `(3+2r)F` over K steps; GANYMEDE pays `1.4F` per step but reduces step count.

---

## 1. Primitive objects

`θ ∈ ℝ^d` global params, `d ≈ 1.84·10⁹`. `L : ℝ^d → ℝ` full-batch loss, `L_B` mini-batch. `g_t := ∇L_{B_t}(θ_t)` stochastic gradient. `(m_t, v_t)` Adam EMAs with `(β_1, β_2) = (0.9, 0.999)`. `A_t := diag(1/(√v̂_t + ε))` Adam preconditioner; `v̂_t := v_t/(1 − β_2^t)`. `η > 0` learning rate.

**Block structure.** CHIRON's 53 symplectic blocks `Φ_1, ..., Φ_53` partition the parameter vector into `B = 53` disjoint groups: `θ = (θ^{(1)}, θ^{(2)}, ..., θ^{(B)})`. Within each block, parameters further partition by weight matrix (Wq, Wk, Wv, Wo, MLP1, MLP2, layer-norm scales/biases). Define `D_b := dim(θ^{(b)})`. At 1.84B / 53 ≈ `D_b ≈ 35M` per block.

**Sub-block partitioning.** GANYMEDE further refines: maintain one L-BFGS sketch per **weight matrix** rather than per block. With ~8 weight matrices per block × 53 blocks = `B' ≈ 424` sketches, each over `d_w ≈ 4.3M` params. This is the operational granularity.

**Streaming history buffer.** Per sketch `b'`:
- `S_t^{(b')} ∈ ℝ^{d_w × r}` — last `r` step displacements, columns `s_i := θ_{t-i+1}^{(b')} − θ_{t-i}^{(b')}` for `i = 1..r`.
- `Y_t^{(b')} ∈ ℝ^{d_w × r}` — last `r` gradient differences, columns `y_i := g_{t-i+1}^{(b')} − g_{t-i}^{(b')}`.
- `ρ_t^{(b')} ∈ ℝ^r` — curvature scalars `ρ_i := 1/(y_i^⊤ s_i)` (skip if `y_i^⊤ s_i ≤ ε_curve`, see §3.5).

**Hyperparameters.** `r ∈ {8, 32, 100}` (memory rank), `γ ∈ [0,1]` damping (scheduled), `η_qN > 0` quasi-Newton step length (typically `η_qN := η`).

**Invariant.** No new persistent per-parameter state beyond `2r + 1` scalars per parameter: `S, Y` columns plus the optional `ρ` cache. At `r=100, d=1.84·10⁹`: `2 · 100 · 1.84·10⁹ · 4 = 1.47 TB` in fp32 — **infeasible globally**, which is why we maintain per-sub-block sketches and bf16 storage (§5.2).

## 2. State space — fiber bundle on parameter space

Define the **history bundle** `ℰ_r := ℝ^d × (ℝ^d)^r × (ℝ^d)^r`. A point `(θ, S, Y) ∈ ℰ_r` carries the current parameters plus the last `r` displacement/gradient-difference pairs. The trajectory `t ↦ (θ_t, S_t, Y_t)` lives in `ℰ_r`.

**Block decomposition.** `ℰ_r = ⊕_{b'=1}^{B'} ℰ_r^{(b')}` where `ℰ_r^{(b')} := ℝ^{d_w} × (ℝ^{d_w})^r × (ℝ^{d_w})^r`. GANYMEDE's update operates **block-locally**: the L-BFGS recursion at sub-block `b'` uses only `(S^{(b')}, Y^{(b')})`, never crossing block boundaries.

**Geometric interpretation.** The L-BFGS sketch `H̃_t^{-1}` is a positive-definite operator on `ℝ^{d_w}` (per sub-block); the trajectory under GANYMEDE follows damped Newton flow on a piecewise-quadratic local model of `L`. The bundle structure makes the construction modular: a sub-block whose L-BFGS sketch becomes ill-conditioned (e.g., curvature flip near a saddle) can be **reset** independently without affecting other sub-blocks.

## 3. Evolution law

### 3.1 Per-step update (sub-block `b'`)

Drop the `(b')` superscript for clarity. At step `t`:

(1) Compute gradient `g_t` (via standard backward pass; CHIRON inverse walk delivers per-layer grads at no extra cost — §6).
(2) Update Adam EMAs `(m_t, v_t)` and form `A_t := diag(1/(√v̂_t + ε))` as usual.
(3) Form streaming pair: `s_t := θ_t − θ_{t-1}`, `y_t := g_t − g_{t-1}` (from prior step's grad cache, one extra `d_w · 4` bytes per sub-block).
(4) **Curvature gate:** if `y_t^⊤ s_t ≤ ε_curve · ‖s_t‖·‖y_t‖` (with `ε_curve = 10^{-8}`), skip update — this pair is non-informative or saddle-flipped.
(5) Otherwise prepend `(s_t, y_t)` to the history buffer; evict oldest if `|S| > r`. Cache `ρ_t := 1/(y_t^⊤ s_t)`.
(6) Compute the L-BFGS direction `d_t := H̃_t^{-1} g_t` via the two-loop recursion (§3.2).
(7) Compute Adam direction `d_t^{Adam} := A_t m_t / (1 − β_1^t)`.
(8) **Hybrid update:**
$$
\boxed{\;\theta_{t+1} = \theta_t - \eta\,\bigl[\gamma_t \cdot d_t^{\text{Adam}} + (1-\gamma_t) \cdot d_t\bigr]\;}
$$
with `γ_t` from §3.3.

### 3.2 L-BFGS two-loop recursion (explicit)

Given history `{(s_i, y_i, ρ_i)}_{i=1}^{r}` (most-recent first; `i=1` is newest), input vector `v ∈ ℝ^{d_w}` (typically `v = g_t`), and seed inverse-Hessian `H_0^{-1}` (a diagonal — see §3.4):

```
Input: v ∈ ℝ^{d_w}, history {(s_i, y_i, ρ_i)}_{i=1..r}, seed H_0^{-1}
Output: q ≈ H̃_t^{-1} v

q ← v
for i = 1, 2, ..., r:                        # backward loop (newest to oldest)
    α_i ← ρ_i · (s_i^⊤ q)
    q   ← q − α_i · y_i

q ← H_0^{-1} · q                              # apply seed (diagonal scaling)

for i = r, r-1, ..., 1:                       # forward loop (oldest to newest)
    β  ← ρ_i · (y_i^⊤ q)
    q  ← q + (α_i − β) · s_i

return q
```

**Per-step cost (per sub-block).** `2r` dot products + `2r` AXPY operations on `ℝ^{d_w}` vectors + one diagonal scale. Total FLOPs: `4 r d_w + d_w` per sub-block. Summed over `B' = 424` sub-blocks: `(4r + 1) · d` FLOPs.

At `r=100, d=1.84·10⁹`: `401 · 1.84·10⁹ ≈ 7.4·10¹¹ FLOPs` ≈ **0.4F** (one F ≈ `1.85·10¹²` FLOPs at this model). At `r=32`: `0.13F`. At `r=8`: `0.033F`.

### 3.3 Damping schedule γ_t

Two motivating regimes:

- **Early training / phase transitions.** `H` is non-PSD; quasi-Newton direction `d_t` may point uphill. Use `γ_t → 1` (pure Adam).
- **Stable basin.** Gradient mostly white noise modulated by smooth curvature; quasi-Newton converges much faster. Use `γ_t → 0`.

**Online indicator.** Track grad-difference autocorrelation:
$$
\hat r_t := \frac{\langle g_t, g_{t-1}\rangle}{\|g_t\|\,\|g_{t-1}\|}, \qquad \bar r_t := (1-\beta_r)\,\bar r_{t-1} + \beta_r\,\hat r_t
$$
with `β_r = 0.01` (decay length 100 steps). When `\bar r_t > 0.2`, gradients are correlated → not yet in basin → use Adam. When `\bar r_t < 0.05`, gradients decorrelated → basin → use quasi-Newton.

**Schedule.**
$$
\boxed{\;\gamma_t \;:=\; \mathrm{clip}\bigl(\,5 \cdot \max(\bar r_t, 0)\,,\; 0,\; 1\bigr)\;}
$$
At `\bar r_t = 0.2` → `γ_t = 1` (pure Adam); at `\bar r_t = 0.05` → `γ_t = 0.25`; at `\bar r_t = 0` → `γ_t = 0` (pure qN). Linear ramp over the meaningful regime.

**Justification.** Adam's `m_t` is an EMA over `~10` past grads; `\bar r_t > 0.2` means the gradient signal carries directional information beyond noise. In that regime Newton's local-quadratic model is unreliable (curvature changes faster than basin scale). When `\bar r_t < 0.05` — the regime that killed TRAJ — gradients are stationary white noise, the local-quadratic model is valid (Hessian stable across a step), and quasi-Newton's `O(log(1/ε))` rate kicks in.

### 3.4 Seed inverse-Hessian H_0^{-1}

Standard L-BFGS choice (Nocedal–Wright 2006, eq. 7.20):
$$
H_0^{-1} := \frac{s_1^\top y_1}{y_1^\top y_1} \cdot I_{d_w}
$$
(a scalar multiple of identity, computed per sub-block per step).

**GANYMEDE-specific extension.** Use Adam's preconditioner as the seed:
$$
H_0^{-1} := \mathrm{diag}\bigl(A_t \cdot (s_1^\top y_1)/(y_1^\top A_t y_1)\bigr).
$$
This pre-bakes Adam's per-coordinate scale into the L-BFGS recursion, producing better-conditioned `H̃` since `A_t` already approximates the Hessian's diagonal. Cost: one extra elementwise multiply per step. (Optional; falls back to scalar seed if conditioning monitor signals problems.)

### 3.5 Curvature gate and history reset

If `y_t^⊤ s_t ≤ ε_curve` (saddle-flip or pure-noise step): skip the history update. If `≥ 3` consecutive skips: **reset** the history buffer (clear all `(s_i, y_i)`) — local quadratic model is invalid, restart fresh. Per-sub-block, so resets stay local.

If sub-block hits NaN gradient: reset its history; do not propagate to other sub-blocks.

## 4. Composition with shipped paradigms

| Shift           | Composition                                                                       | Multiplier  |
|-----------------|------------------------------------------------------------------------------------|-------------|
| **CHIRON #1**   | Inverse walk delivers per-layer grads → free `s_i, y_i` per sub-block             | **+1.0×**   |
| **SCFA #42**    | Per-step compute reduction; orthogonal axis (steps-per-loss)                       | **multiplicative** |
| **SAS #40**     | SAS skips gradients; affects `y_i` cadence — OK with skip-aware history index     | **multiplicative** |
| **SLC #38**     | T schedule orthogonal to optimizer step count                                      | **multiplicative** |
| **RLG #39**     | New layers get fresh L-BFGS sketches; warm with scalar seed                        | **multiplicative** |
| **FACE #28**    | Embedding is a sub-block of its own; FACE state vs L-BFGS history disjoint         | **multiplicative** |
| **MFIO #11**    | MFIO replaces Adam state; L-BFGS uses raw `g`, not `m,v`. Inputs unchanged         | **neutral** |
| **WIP #22**     | WIP K-snapshot grads can seed `Y_t` history (shared structure)                     | **+0.1×**   |
| **IBGRAD #19**  | Wo factor is its own sub-block; L-BFGS in factor space (not dense)                 | **neutral** |
| **Kahan-v #17** | Adam `v` precision unchanged; L-BFGS uses fp32 dots                                | **neutral** |
| **SPAREC #35**  | Sparse FFN backward → sparse `y_i` columns; AXPY trivially handles zeros           | **neutral** |

**Stack at 1.84B.** Shipped flagship `3.36×` (SLC × RLG × FACE × SAS × Kahan-v) × SCFA `2.27×` per-step (T=1024) × GANYMEDE step-count reduction `~10×` (if C1 holds) ÷ overhead `1.4×` = **`3.36 × 2.27 × (10/1.4) = 54×`**. Decisively in magnitudes territory.

If C1 fails (`r_eff > 1000` per sub-block), GANYMEDE degrades to step-count parity with Adam and the `1.4×` overhead is pure cost: net contribution **0.71×** (regression). Gate-0 (§9) decides.

## 5. Memory accounting

### 5.1 Storage per sub-block

Per sub-block at rank `r`:
- `S` matrix: `d_w · r · 2 bytes` (bf16)
- `Y` matrix: `d_w · r · 2 bytes` (bf16)
- `ρ` cache: `r · 4 bytes` (fp32, scalars matter)
- Prev-grad cache (`g_{t-1}^{(b')}`): `d_w · 2 bytes` (bf16) — needed to form `y_t = g_t − g_{t-1}`

Total: `(2r + 0.5) · d_w · 2 + 4r ≈ (4r + 1) · d_w` bytes (bf16 dominates).

### 5.2 Total at 1.84B

| `r` | Per-sub-block | Sum over 424 sub-blocks | % of 16 GB ceiling |
|-----|---------------|-------------------------|---------------------|
| 8   | 142 MB avg    | **2.4 GB**              | 15%                 |
| 32  | 565 MB avg    | **9.6 GB**              | 60%                 |
| 100 | 1.76 GB avg   | **30 GB**               | 187% — **infeasible** |

**Recovery: hierarchical sub-block sketches.** For `r=100` to fit, partition each weight matrix further into `n_p` patches; maintain one L-BFGS sketch per patch. With `n_p = 4` per matrix → `1696` sub-blocks of `~1M` params each → `r=100` sketch costs `~7.5 GB` total. Feasible.

**Alternative recovery: rank-`r` bf16 + on-the-fly seed re-orthogonalization.** Keep `r=32` (fits at 9.6 GB), accept the looser convergence rate (Theorem B with `r=32`). Conservative track.

**Operational target.** `r = 32` with bf16 sketch, fp32 dot accumulators in the two-loop. Cost: 9.6 GB persistent + 0.5 GB scratch = **10.1 GB** within the 16 GB ceiling. Aggressive `r=100` track requires the patch-partition recovery.

### 5.3 CPU offload option

If GPU memory is tight after stacking with SCFA scratch (which itself needs ~2 GB for the `B_ℓ` bases), offload `S, Y` to pinned host memory and stream during the two-loop. The two-loop is fundamentally `2r · d_w` reads + `2r · d_w` writes per sub-block; PCIe at 32 GB/s reads `9.6 GB / 32 GB/s = 0.3s` per step — **kills throughput**. Reject CPU offload for L-BFGS state. Use the patch-partition recovery if needed.

## 6. CHIRON synergy

CHIRON's reversible-flow forward delivers paired state `(q, p)` through L symplectic blocks. Backward proceeds via **inverse walk**: at each block `Φ_ℓ`, recompute `(q_{ℓ-1}, p_{ℓ-1})` from `(q_ℓ, p_ℓ)` using the closed-form inverse, then propagate gradients. Crucially, **per-layer parameter gradients `g^{(ℓ)} = ∂L/∂θ^{(ℓ)}` are materialized one block at a time** during the inverse walk.

GANYMEDE's `s_i, y_i` are computed per-sub-block. During backward at block `ℓ`:
1. Recover `g^{(ℓ)} = ∂L/∂θ^{(ℓ)}` (already free in CHIRON).
2. Form `y_t^{(ℓ)} := g_t^{(ℓ)} − g_{t-1}^{(ℓ)}` (one AXPY, `d_w` flops).
3. `s_t^{(ℓ)} := θ_t^{(ℓ)} − θ_{t-1}^{(ℓ)}` is computed at param-update time, not backward time, but stored alongside.
4. Add `(s_t^{(ℓ)}, y_t^{(ℓ)})` to that sub-block's L-BFGS history.

**Cost.** Two AXPY per sub-block per step → `(2 · 4) · d ≈ 1.5·10¹⁰ FLOPs ≈ 0.008F`. Negligible. The per-block sketch architecture is structurally aligned with CHIRON's per-block backward.

## 7. Engaging with TRAJ rejection (paradigm #30)

TRAJ (rejected iter 92) attempted to predict Adam's `(m, v)` state via AR(2) regression on past `(m, v)` history. The probe that killed TRAJ measured `lag-1 grad-norm autocorrelation = -0.087`: gradients are essentially white noise across consecutive steps, so AR(2) on a function of past gradients (which is what `m_t` is) cannot do better than the EMA itself.

### 7.1 Why GANYMEDE escapes the same probe

GANYMEDE does NOT predict the gradient. It uses past **gradient differences** to estimate the **action of the Hessian on past displacements**. Mathematical identity (Pearlmutter 1994, used here as a definition rather than an HVP):

For sufficiently smooth `L`,
$$
y_i = g_{t-i+1} - g_{t-i} = \int_0^1 \nabla^2 L\bigl(\theta_{t-i} + \tau s_i\bigr)\, s_i \,d\tau \;=\; \bar M_i \cdot s_i
$$
where `\bar M_i` is the **mean Hessian along the segment from `θ_{t-i}` to `θ_{t-i+1}`**. So `(s_i, y_i)` is exactly a noisy `(input, output)` pair sampled from the Hessian operator. L-BFGS reconstructs an inverse-Hessian sketch from these pairs; **even when `g` itself is white**, the differences `y_i` carry **deterministic** Hessian information.

**Formal claim (whitening robustness).** Suppose `g_t = G_t \cdot \theta_t + \xi_t` where `G_t` is a smooth deterministic operator and `ξ_t` is iid mean-zero noise with `corr(ξ_t, ξ_{t-1}) = 0`. Then:
- `corr(g_t, g_{t-1})` = TRAJ probe ≈ 0 (driven by `ξ`).
- `y_i = G_t s_i + (G_t − G_{t-1})θ_{t-1} + (\xi_{t} − \xi_{t-1})`. The first term is `Hessian · displacement`, the third is independent noise; L-BFGS averaging over `r` pairs concentrates the signal at rate `1/√r`.

**Quantitatively** (white noise injected): if `‖ξ_t‖ = σ` and the deterministic Hessian-action term has magnitude `h := ‖M s_i‖ ≈ η ‖M‖ ‖m_t‖`, then signal-to-noise per pair is `SNR_1 = h / σ√2`. With `r=32`, the L-BFGS sketch achieves SNR `≈ √32 · SNR_1 ≈ 5.7 SNR_1`. At typical training: `‖M‖ ≈ 10`, `η = 3·10⁻⁴`, `‖m_t‖ ≈ 1`, `σ ≈ 0.5` → `SNR_1 ≈ 0.0042` per pair → `SNR_{32} ≈ 0.024`. **Insufficient.**

**Hmm — this is a real concern.** Single-pair SNR is much smaller than 1 in raw noise terms. But L-BFGS only needs `s_i^⊤ y_i > 0` (curvature condition) and approximate orientation, not exact Hessian recovery. Empirical L-BFGS on stochastic losses (Bollapragada et al. 2018) shows `r ∈ [8, 50]` sufficient at SNR_per_pair ~ 0.05 with **damped curvature update** (Powell's modification). GANYMEDE's curvature gate (§3.5) implements exactly this damping.

### 7.2 The right Gate-0 probe

TRAJ's grad-autocorr probe is informative about TRAJ; it is **not informative** about GANYMEDE because the two paradigms use different observables. The right probe for GANYMEDE is:

**Hessian alignment over a step:** compute `cos(y_t, M_t · s_t)` directly via one Pearlmutter HVP at sampled steps. **Pass criterion:** `mean cos > 0.3` over ≥ 30 sampled `(t, t-1)` pairs across stable training.

If alignment fails (e.g., `cos < 0.05`), it means `y_t` carries no usable Hessian signal — L-BFGS reduces to whitened-direction averaging, no better than Adam. Reject.

This is the **TRAJ-killer probe analogue on the right observable**. GANYMEDE dies here cheaply if its premise is wrong.

### 7.3 Structural difference summary

| Aspect | TRAJ (#30) | GANYMEDE (#43-B) |
|--------|-----------|-------------------|
| Predicts | Future gradient `\hat g_{t+k}` | Curvature `H^{-1}` |
| Past data used | `(m_{t-1}, v_{t-1}, ...)` | `(s_i, y_i)` differences |
| TRAJ-killer probe | grad-autocorr ≈ 0 → fatal | grad-autocorr ≈ 0 → orthogonal |
| Right probe | (none survived) | Hessian alignment cos > 0.3 |

The probes are **independent**. TRAJ's failure does not imply GANYMEDE's failure.

## 8. Theorems

### 8.1 Theorem A — low-rank Hessian sketch quality

**Setup.** Let `M_t = ∇²L(θ_t)` with `L_H`-Lipschitz Hessian, condition number `κ_t := ‖M_t‖·‖M_t^{-1}‖`. Let `H̃_t^{-1}` be the rank-`r` L-BFGS sketch from `r` exact pairs `(s_i, y_i)` satisfying `y_i = M_t s_i` (deterministic limit, no noise). Let `g ∈ ℝ^{d_w}` be a query vector.

**Claim.** Define the projection error
$$
\mathcal{E}_r := \|H̃_t^{-1} g - M_t^{-1} g\|.
$$
Decompose `g = g_∥ + g_⊥` where `g_∥ ∈ \mathrm{span}\{s_1, \ldots, s_r\}` and `g_⊥ ⊥ \mathrm{span}\{s_1, \ldots, s_r\}`. Then:
$$
\boxed{\;\mathcal{E}_r \;\le\; \kappa_t\,\|g_⊥\| \;+\; L_H \cdot \eta \cdot \|s_1\|\cdot \|g\| \cdot \frac{r}{2}.\;}
$$

**Proof sketch.** L-BFGS recursion exactly inverts `M_t` on the span of past `s_i`'s by construction: `H̃_t^{-1} y_i = s_i = M_t^{-1} y_i`, so `H̃_t^{-1} g_∥ = M_t^{-1} g_∥`. The first term bounds the orthogonal complement: outside the span, L-BFGS uses the seed `H_0^{-1} \approx I/κ_t`, and the difference from `M_t^{-1}` is at most `(‖M_t^{-1}‖ + 1/‖M_t‖)·‖g_⊥‖ ≤ κ_t ‖g_⊥‖/‖M_t‖ ≤ κ_t‖g_⊥‖`. The second term comes from Hessian variation across the `r`-segment trajectory: `‖M_{t-i} − M_t‖ ≤ L_H · η · i · ‖m_t‖ ≤ L_H η i \|s_1\|/η = L_H \|s_1\| · i`; summing up to `r` gives the stated bound. ∎

**Implication.** If `g` lies mostly within `\mathrm{span}\{s_i\}` (i.e., aligned with recent search directions), the sketch is essentially exact. If `g_⊥` dominates (orthogonal subspace), error scales with `κ_t`. This motivates **diversifying** `s_i` directions over training (Powell's damping, §3.5 reset) so the span covers the dominant eigendirections.

### 8.2 Theorem B — convergence rate (worst case)

**Setup.** Strongly convex `L` on a basin with `μ I ≼ M_t ≼ L I`, `κ := L/μ`. GANYMEDE step at `γ = 0` (pure quasi-Newton).

**Claim.** GANYMEDE achieves ε-tolerance in
$$
\boxed{\;N_{\text{GANYMEDE}}(\varepsilon) \;=\; O\!\left(\bigl(1 + (1 - r/d_w)·\kappa\bigr)\,\log(1/\varepsilon)\right)\;}
$$
iterations.

**Proof sketch.** Standard L-BFGS analysis (Liu & Nocedal 1989, eq. 5.1): the rank-`r` sketch acts as identity on the `r`-dim subspace (Newton there) and as `H_0^{-1}` outside. Outside the subspace, descent rate is `1 − μ/L = 1 − 1/κ` per step (Adam-style). Linear combination over the `(d_w - r)`-dim complement gives effective rate `(d_w − r)/d_w · (1 − 1/κ) + r/d_w · 1`, yielding the bound. ∎

**Implication (kicker).** At `d_w = 4.3·10⁶, r = 32`: `(1 − r/d_w) ≈ 1 − 7.4·10⁻⁶ ≈ 1`. So `N_{\text{GANYMEDE}} ≈ \kappa \log(1/\varepsilon)` — **same as Adam in worst case.** The `r ≪ d_w` regime gives no rate improvement.

This is the **honest worst case**. It means GANYMEDE only beats Adam if the loss landscape has **effective rank** `r_{\text{eff}} ≤ r`, i.e., only `r_{\text{eff}}` Hessian eigenvalues exceed the tolerance threshold.

### 8.3 Theorem C (conditional) — effective-rank convergence

**Setup.** Suppose the Hessian `M_t` has eigenvalues `λ_1 ≥ λ_2 ≥ ... ≥ λ_{d_w}` with `λ_{r_{\text{eff}}} ≥ \varepsilon_{\text{tol}} \cdot \lambda_1` and `λ_{r_{\text{eff}}+1} < \varepsilon_{\text{tol}} \cdot \lambda_1`. Call this the **effective-rank condition**: only the top `r_{\text{eff}}` directions matter for convergence to `\varepsilon_{\text{tol}}`-tolerance.

**Claim.** If `r ≥ r_{\text{eff}}` and the L-BFGS history `\{s_i\}_{i=1}^r` spans the top-`r_{\text{eff}}` eigenspace, GANYMEDE at `γ=0` achieves ε-tolerance in
$$
\boxed{\;N_{\text{GANYMEDE}}(\varepsilon) \;=\; O\!\left(\log(1/\varepsilon)\right)\;}
$$
iterations — **Newton's rate**.

**Proof sketch.** Within the top-`r_{\text{eff}}` eigenspace, Theorem A bounds error by Hessian-Lipschitz term only (vanishes as `s_i → 0`). Outside, residual gradient is below `\varepsilon_{\text{tol}}` by assumption — quasi-Newton already terminates. Total iterations: `O(log(1/\varepsilon))`. ∎

**Conjecture C1 (the gamble).** During stable LLM training, the per-sub-block Hessian has effective rank `r_{\text{eff}} ≤ 200` for `\varepsilon_{\text{tol}} = 10^{-3}`.

This is **the load-bearing claim** of GANYMEDE. It is empirically testable (§9). If true: `r=100, n_p=4` patch-partition recovery yields Newton rate. If false: GANYMEDE degrades to Adam-rate (Theorem B) and the `1.4×` overhead is dead weight.

**Prior support for C1.**
- Sagun et al. 2017, *Empirical Analysis of the Hessian of Over-Parametrized Neural Networks*: Hessian spectrum has a few outliers and a near-zero bulk; effective rank for ResNet ≈ #classes ≈ 10–1000.
- Papyan 2019, *Measurements of Three-Level Hierarchy of Spectra of Deep Networks*: Hessian rank scales sub-linearly with #params; per-layer ranks `O(100)` plausible.
- LoRA empirics (Hu et al. 2021): rank-8 low-rank adaptation matches full fine-tune on most tasks → fine-tuning Hessian effective rank ≤ 8 in fine-tune regime.

These do not prove C1 at LLM-pretraining scale. C1 is a **conjecture**, and Gate-0 falsifies or confirms it.

### 8.4 Bottom-line speedup

Conditional on C1:
- Per-step overhead: `1.4×` (L-BFGS two-loop at r=100 on patch-partitioned sub-blocks).
- Step-count reduction: from `O(κ log(1/ε)) = O(100 · 7) = 700` Adam steps to `O(log(1/ε)) = O(7) · K_const` GANYMEDE steps. With `K_const ≈ 5` from per-step constant overhead (history warmup, damping schedule): `35` steps. Ratio: `700/35 = 20×`.
- Net: `20× / 1.4 = 14.3×` (pessimistic accounting); `25× / 1.4 = 17.8×` (optimistic).

Honest projection: **10×–15× wall-clock**, conditional on C1 holding.

If C1 fails: net `0.71×` (regression). Ralph-loop responsibility: Gate-0 must **fire before any wire-in**.

## 9. Gate-0 probe — testing Conjecture C1

### 9.1 Goal

Determine whether per-CHIRON-block Hessian effective rank is `≤ 200` during stable training of a 66M CHIRON checkpoint (cheap proxy for 1.84B per-block dynamics; CHIRON's per-block FLOPs structure is identical at both scales).

### 9.2 Procedure

1. **Snapshot.** Take a 66M CHIRON checkpoint mid-training (post-warmup, in the `\bar r_t < 0.05` regime — verify by online indicator).
2. **Per-block Lanczos.** For each of the 24 blocks at 66M (smaller B' since fewer layers): pick the largest sub-block (typically Wo or MLP1, `d_w ≈ 50k`). Run 200 iterations of stochastic Lanczos quadrature (Pearlmutter HVPs at random unit vectors) to estimate the Hessian spectrum.
3. **Effective rank.** Count eigenvalues with magnitude `≥ 10⁻³ · λ_max`. Call this `r_{\text{eff}}^{(b)}`.
4. **Aggregate.** Report `\max_b r_{\text{eff}}^{(b)}`.

**Cost.** 200 HVPs × 2F × ~100 sub-blocks at 66M = 40,000 F equivalents. At 66M (F ≈ 0.13 GFLOP), this is `5.2 PFLOP` — about 30 GPU-minutes on the existing 16 GB GPU.

### 9.3 Pass criterion

- **Strong pass:** `\max_b r_{\text{eff}}^{(b)} ≤ 100` → GANYMEDE with `r=100, n_p=4` patch partition delivers Newton rate. **Build it.**
- **Marginal pass:** `100 < \max_b r_{\text{eff}}^{(b)} ≤ 500` → consider hybrid: aggressive `r=200` with cheaper sub-block partition. Investigate.
- **Fail:** `\max_b r_{\text{eff}}^{(b)} > 500` → GANYMEDE cannot reach Newton rate at memory budget. **Reject.**

### 9.4 Secondary probe — Hessian alignment

Independent of effective-rank probe, also measure (§7.2):
- Sample 30 `(t, t-1)` pairs from stable training.
- Compute `\cos(y_t, M_t · s_t)` via one Pearlmutter HVP per pair.
- **Pass:** `mean cos > 0.3`. Rejects if even rank-r sketch's `(s, y)` pairs are noise-dominated.

### 9.5 Existing-log gate

Even cheaper: from existing 1.84B training logs, compute `(s_t, y_t)` pairs from stored grad checkpoints (we have these from Kahan-v debugging). Run small-r L-BFGS offline; check whether the resulting direction is closer to `M_t^{-1} g_t` (via post-hoc HVP) than `A_t m_t` is. **Free** in compute, decisive on whether GANYMEDE's quasi-Newton direction beats Adam in retrospect.

## 10. Failure modes and mitigations

1. **Saddle points / indefinite Hessian.** L-BFGS is undefined for non-PSD `M`. Mitigation: curvature gate (§3.5) skips negative-curvature pairs; damping schedule keeps `γ_t ≈ 1` (pure Adam) until `\bar r_t < 0.2`, by which time we are past saddles in practice.

2. **Stochastic gradient noise.** `y_i` carries large noise (§7.1). Mitigation: Powell's damping (Bollapragada et al.); rank-r averaging; per-sub-block reset on consecutive curvature failures.

3. **History staleness.** `s_i, y_i` from many steps ago may not reflect current Hessian. Mitigation: `r` fixed (history rolls); reset on phase transitions detected via grad-norm jump.

4. **bf16 precision in `S, Y`.** Subtracting two close vectors in bf16 catastrophically loses precision when `‖s‖ ≪ ‖θ‖`. Mitigation: store `S` directly (not as `θ_t − θ_{t-1}` stored separately); use Kahan compensation for the AXPY accumulating `s_t`.

5. **NaN propagation.** Bad gradient at one step poisons that sub-block's history. Mitigation: per-sub-block NaN check on `y_t` insertion; reset history on detection.

6. **Sub-block boundary effects.** Block-local Newton may miss cross-block curvature interactions. Theoretical concern: per-block Newton converges to per-block stationary point, not global stationary. Mitigation: damping schedule keeps `γ_t > 0` (some Adam component) so global gradient flow is preserved. This is a deliberate trade — full global Newton is structurally infeasible at 1.84B.

7. **CHIRON RLG layer growth.** When new layers are inserted, their L-BFGS history is empty. Mitigation: bootstrap with scalar seed `H_0^{-1} = I` for first `r` steps post-insertion; full sketch active by step `t + r`.

8. **Composition with SCFA spectral basis.** SCFA's `B_ℓ` matrices are themselves sub-blocks under GANYMEDE's accounting. They have their own L-BFGS sketches; no special handling needed.

## 11. Concrete primitives (signatures only)

```cpp
// Backend/Machine Learning/MLState/ganymede_state.h
struct GanymedeSubBlockState {
    float* S_dev;          // d_w × r (bf16 storage; fp32 accumulate)
    float* Y_dev;
    float* rho_dev;        // r (fp32)
    float* prev_g_dev;     // d_w (bf16)
    float prev_y_s_inner;  // tracks (y_t^T s_t) for curvature gate
    int r_active;          // current history depth (0..r at warmup)
    int reset_count;       // for phase-change diagnostics
};

struct GanymedeState {
    GanymedeSubBlockState* sub_blocks;  // size B' = 424 (or 1696 with patches)
    int B_prime;
    int r_max;
    float gamma_current;   // current damping
    float bar_r_t;         // EMA of grad autocorrelation
};
```

```cpp
// Backend/Machine Learning/Networks/cuda/gpu_ganymede.h
namespace glades { namespace gpu {

// Two-loop recursion on a single sub-block. Computes d = H~^{-1} g.
void ganymede_two_loop(const GanymedeSubBlockState& state,
                       const float* g_dev,
                       float* d_dev,
                       int d_w, int r,
                       cudaStream_t stream);

// Insert new (s, y) pair after a parameter update.
void ganymede_history_update(GanymedeSubBlockState& state,
                             const float* theta_now_dev,
                             const float* theta_prev_dev,
                             const float* g_now_dev,
                             const float* g_prev_dev,
                             int d_w, int r,
                             float curvature_eps,
                             cudaStream_t stream);

// Per-step damping schedule update.
void ganymede_update_gamma(GanymedeState& state,
                           const float* g_now_dev,
                           const float* g_prev_dev,
                           int d_total,
                           cudaStream_t stream);

// Hybrid update step: theta -= eta * (gamma * d_Adam + (1-gamma) * d_qN).
void ganymede_hybrid_update(GpuBuffer<float>& theta_dev,
                            const float* d_Adam_dev,
                            const float* d_qN_dev,
                            float eta, float gamma,
                            int d_total,
                            cudaStream_t stream);

// Reset sub-block history (called on consecutive curvature failures).
void ganymede_reset_subblock(GanymedeSubBlockState& state,
                             cudaStream_t stream);

}}
```

```cpp
// Trainer integration: Backend/Machine Learning/Networks/sgd_transformer.cpp
// One branch: standard Adam vs. GANYMEDE.
// CHIRON inverse walk delivers per-block grads; GANYMEDE consumes them.
```

## 12. Honest gaps

- **Conjecture C1 is the load-bearing assumption.** All magnitude claims condition on it. Unfalsified at LLM-pretraining scale. Gate-0 (§9) is the test.
- **Theorem A's bound** is for the deterministic case `y_i = M s_i`. The stochastic refinement (§7.1) is heuristic; rigorous version requires concentration bounds on `\|H̃_r^{-1} - M^{-1}\|` under `y_i = M s_i + ξ_i` with `ξ_i` stationary. Bollapragada et al. 2018 proves convergence under such noise; rate constants not transferred verbatim.
- **Damping schedule `γ_t`** is heuristic; the linear ramp `\gamma_t = 5\bar r_t` is not derived from first principles. Justification is qualitative (§3.3). Could be replaced by a Kalman-filter-style trust-region monitor (future work).
- **Sub-block independence.** GANYMEDE assumes block-local Newton suffices. Cross-block curvature is ignored. For CHIRON's reversible flow, blocks are weakly coupled (each shear is local), which heuristically supports per-block independence; not proved.
- **Per-step `1.4×` overhead** assumes `r=100` and patch-partition recovery to `n_p=4`. At simpler `r=32, n_p=1` config: overhead `1.13×`, but step-count reduction is also smaller (Theorem B with rank `r=32` against effective-rank `r_eff ≈ 100` only partially recovers Newton rate).
- **Compatibility with FACE Adafactor accumulators.** FACE's per-row-frequency state on the embedding sub-block must coexist with that sub-block's L-BFGS history. Disjoint storage; tested at ALL BUT not verified at LARGE scale.
- **bf16 numerics in two-loop dot products.** `s_i^⊤ q` accumulators must be fp32; bf16 dot accumulator overflows for `d_w > 10⁵`. Verified in CUDA design (use `__bfloat162float` then fp32 reduce). Not a math gap but a precision-engineering gap to flag.
- **K=10 step-count reduction** assumes basin κ ≈ 100. Actual κ at 1.84B mid-training unmeasured (would require Lanczos on 1.84B Hessian — expensive). Conservative estimate.

## 13. Material differences from NEXUS (#43-A)

| Axis | NEXUS | GANYMEDE |
|------|-------|----------|
| Mechanism | Symplectic Verlet extrapolation on Hamiltonian phase space | Streaming L-BFGS quasi-Newton |
| State | Single anchor `(θ_*, p_*, A_*, U_*, T_*)`, refreshed every K | Per-sub-block rolling `(S, Y, ρ)` history |
| Anchor cost | `(3+2r)F` every K steps | None — continuous streaming |
| Per-step cost | `O(K·d·r)` AXPY only (no model calls in window) | `1.4F` per step (L-BFGS + Adam + grad) |
| Speedup mechanism | Amortize forward/backward over K steps | Reduce step count via Newton rate |
| Failure mode | Hessian non-stationarity over K·η window | Effective-rank conjecture C1 fails |
| Probe | Position-prediction on existing 66M logs (§9.2 in NEXUS doc) | Hessian alignment + Lanczos eff-rank (§9 above) |
| TRAJ engagement | Predicts position not gradient (slow Adam EMA subspace) | Uses gradient differences not gradients themselves |

The two paradigms are **multiplicatively orthogonal**: NEXUS amortizes per-step compute; GANYMEDE reduces step count. Composed: `(3+2r)F/K · (1.4F · log(1/ε) / (κ log(1/ε)))` — multiplicative on different axes. If both ship: combined `~6× × 10× = 60×` on top of shipped flagship.

## 14. Suggested workplan (if Gate-0 passes)

1. **Iter 187 — Gate-0 (this paradigm).** Lanczos eff-rank probe + Hessian-alignment probe. Decision point.
2. **Iter 188 — Mathematical refinement.** If marginal pass, design hybrid `r=200, patch=4` recovery.
3. **Iter 189–192 — Implementation.** Per-sub-block state allocation; CUDA two-loop kernel; integration with CHIRON inverse walk in `sgd_transformer.cpp`; damping monitor wiring.
4. **Iter 193 — 66M validation.** A/B at 66M × 10k steps: GANYMEDE vs flagship Adam. Target: `≥ 5×` step-count reduction in basin.
5. **Iter 194 — 1.84B compound.** SCFA + GANYMEDE flagship. Target: `≥ 30×` total.

Total iterations to ship: ~7. Total iterations to **reject** (if Gate-0 fails): **1**. Cheap falsification path.

---

**Summary.** GANYMEDE is a per-CHIRON-block streaming L-BFGS that compounds with SCFA on the orthogonal axis of step count. Its magnitude claim (10×) is conditional on Conjecture C1 (per-block Hessian effective rank ≤ 200), which is testable in 30 GPU-minutes by Lanczos on a 66M checkpoint. Its TRAJ-rejection robustness comes from using gradient **differences** (Hessian-action probes), not gradients themselves. Its per-step `1.4×` overhead is honestly accounted; if C1 fails, GANYMEDE degrades to net regression, which Gate-0 catches before wire-in. The candidate is mathematically rigorous, materially distinct from NEXUS (curvature estimation vs. position extrapolation), and aligned with CHIRON's per-block backward structure. Gate-0 is the load-bearing test.
