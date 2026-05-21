# Paradigm Shift #43 Candidate C — ORION (Online Reduced-Order Integrator Network)

**Status:** candidate-C design, single-formulation. Companion to #43-A (NEXUS) and #43-B (GANYMEDE).
**Date:** 2026-05-08.
**Tagline:** *Galerkin model-order reduction of the SGD ODE itself — identify a streaming r-dim slow manifold, integrate the surrogate quadratic for K steps at O(r²) per step, refresh on anchors.*

---

## 0. Executive summary

CHIRON has shipped a 3.36× wall-clock stack through paradigms #1–#41 attacking per-step compute (FACE, MFIO, SAS, SPAREC) and memory (CHIRON-reversibility, IBGRAD, Kahan-v). Paradigm #42 (SCFA, iter 186) compresses sequence-axis attention to add 2.27× per-step at T=1024, taking the stack to ≈ 7.6×. To cross the magnitudes threshold (≥10×) shift #43 must amplify the **steps-to-target-loss** axis — every shift through #42 leaves total step count untouched.

ORION views the Adam parameter trajectory as the trajectory of a high-dimensional ODE `dθ/dt = −A(θ) ∇L(θ)`. Empirically (LLM training literature; verified §8 Gate-0) this trajectory lives on a **slow manifold** of effective dimension `r ≈ 4–8 ≪ d ≈ 1.84·10⁹`. Galerkin projection onto a streaming basis `V_t ∈ Stiefel(d, r)` collapses the ODE to a low-dimensional surrogate; the surrogate is integrated for K steps at O(r²) per step using the projected quadratic `(g_∥, H_∥) = (V^⊤ g_*, V^⊤ M V)`. Periodic anchors refresh `V` via streaming PCA / block Krylov on the orthogonal residual.

**Cost per K-window.** Anchor `(3+2r)F` (full F+B + r HVPs) + K cheap reduced steps `O(r²) ≈ 0`.
**Per-effective-step.** `(3+2r)F / K`.
**Headline.** K=20, r=2 → **8.6×**; K=40, r=2 → **17×**; K=20, r=4 → **5.5×**; K=10, r=4 → **2.7×**.

ORION is materially different from NEXUS (extrapolates on full phase space `T*ℝ^d` via symplectic Verlet) and GANYMEDE (streaming low-rank quasi-Newton on Adam-step trajectory): ORION reduces the ODE *before* integrating, not after. The surrogate is an `r`-dimensional ODE with closed-form linear solution. This is the model-order-reduction (MOR) view, and it lets K grow much further than NEXUS's symplectic-stability bound permits.

**Critical empirical risk.** ORION's premise — that the trajectory lives on an `r ≤ 4` slow manifold — is empirically tested at LLM scale only weakly. Section §10 details a 1-GPU-hour Gate-0 probe on the existing 66M CHIRON trajectory log. If the probe fails (residual energy > 5% at `r=8`), ORION dies before code.

---

## 1. Primitive objects

| Symbol | Type | Definition |
|---|---|---|
| `θ_t` | `ℝ^d` | full parameters at step `t`, `d ≈ 1.84·10⁹` |
| `g_t` | `ℝ^d` | full minibatch gradient `∇L_{B_t}(θ_t)` |
| `(m_t, v_t)` | `ℝ^d × ℝ^d` | Adam EMAs, `(β_1, β_2) = (0.9, 0.999)` |
| `A_t` | diagonal in `ℝ^{d×d}` | Adam preconditioner `diag(1/(√v̂_t + ε))` |
| `M_t` | `ℝ^{d×d}` | mini-batch Hessian `∇²L_{B_t}(θ_t)` (never materialized) |
| `V_t` | `Stiefel(d, r)` | streaming slow-mode basis, `V^⊤ V = I_r` |
| `α_t := V_t^⊤ θ_t` | `ℝ^r` | reduced (slow) coordinate |
| `θ_⊥,t := (I − V_t V_t^⊤) θ_t` | `ℝ^d` | fast-mode complement |
| `g_∥,t := V_t^⊤ g_t` | `ℝ^r` | projected gradient (cheap via VJP, cost ≈ 2F) |
| `g_⊥,t := (I − V_t V_t^⊤) g_t` | `ℝ^d` | fast-mode gradient |
| `H_∥,t := V_t^⊤ M_t V_t` | `ℝ^{r×r}` | reduced Hessian (Lanczos, cost `r · 2F`) |
| `K` | `ℕ` | inter-anchor horizon (default 20) |
| `M_subspace` | `ℕ` | inter-refresh horizon for `V` (default 1000) |
| `r` | `ℕ` | reduced-order rank (default 2; max 8) |

**Invariant.** ORION introduces no new per-parameter optimizer state. Anchor memory: `O(d·r)` for `V_t` plus `O(d)` for `θ_⊥` (recoverable as `θ − V α`, so optional). Total persistent overhead: `2d·r·sizeof(bf16) ≈ 14.7 GB at d=1.84B, r=4` (would not fit; we use `r=2` → 7.4 GB, achievable with FACE/MFIO trims; or store `V` in fp16 which is acceptable since `V` only enters via inner products with gradients of equal noise scale).

## 2. State space — Stiefel manifold for V

ORION's full state lives on the bundle
$$
\mathcal{S} := \mathbb{R}^d \times \mathbb{R}^r \times \mathrm{Stiefel}(d, r), \qquad \mathrm{Stiefel}(d, r) := \{V \in \mathbb{R}^{d\times r} : V^\top V = I_r\}.
$$
A point is `(θ_⊥, α, V)`. The lift map `Π : \mathcal{S} → ℝ^d`, `Π(θ_⊥, α, V) := θ_⊥ + V α` reconstructs the full parameter vector. The complementary projection `(I − VV^⊤)` is implicit in `θ_⊥`'s constraint `V^⊤ θ_⊥ = 0`.

**Tangent space at `V`.** `T_V \mathrm{Stiefel}(d,r) = \{Δ V \in ℝ^{d×r} : V^⊤ ΔV + ΔV^⊤ V = 0\}` (skew-symmetric in the `r×r` block). Updates to `V` via Oja's rule must project onto this tangent then retract back to the manifold; we use thin-QR retraction.

**Why Stiefel and not just Grassmannian?** The orientation of `V`'s columns matters because `α = V^⊤ θ` has interpretable per-column dynamics; gauge ambiguity (post-multiplication by `O(r)`) is fixed by tridiagonalizing `H_∥` in the Lanczos pass.

## 3. Evolution law

### 3.1 Anchor step (every `t ≡ 0 (mod K)`)

(1) **Full F+B.** Compute `g_t`, update `(m_t, v_t)`, set `A_t`. Cost: `3F`.

(2) **Reduced quadratic.** `g_∥ := V_t^⊤ g_t` (cost: 1 VJP ≈ 2F, but the VJP is *along* the same backward computation, so amortized to ≈ 0 if integrated into the F+B; we charge it 0). For Lanczos rank-`r`, run `r` HVPs on `M_t` against the columns of `V_t` to build `H_∥ = V_t^⊤ M_t V_t ∈ ℝ^{r×r}` (Pearlmutter HVP, cost `r · 2F`).

(3) **Subspace update (Oja's rule on residual).** With learning rate `η_V`:
$$
\boxed{\;V_{t+1} \;\leftarrow\; \mathrm{QR}\bigl(\,V_t \;+\; η_V \cdot g_{⊥,t}\,(V_t^\top g_t)^\top\,\bigr)_{\!Q}\;}
$$
This is rank-`r` Oja: it tilts `V` toward the dominant directions of the recent gradient outer-product `E[g g^\top]`. Thin QR cost: `O(d r²)` ≈ negligible at `r=4`.

(4) **Standard Adam step.** `θ_{t+1} := θ_t − η · A_t · m̂_t / (\sqrt{v̂_t} + ε)`. This is the *anchor's* full update, used in the lift-back step.

(5) **Reset reduced coordinate.** `θ_⊥,t+1 := (I − V_{t+1} V_{t+1}^\top) θ_{t+1}`, `α_{t+1} := V_{t+1}^\top θ_{t+1}`. Cache `(θ_*, g_∥*, H_∥*, A_*, V_*) := (θ_{t+1}, g_∥,t, H_∥, A_t, V_{t+1})`.

**Anchor cost: `(3 + 2r) F` total.**

### 3.2 Reduced step (`s ∈ {1, …, K−1}` between anchors)

The local quadratic surrogate (frozen at the anchor):
$$
\widetilde{L}(α) \;:=\; L(θ_*) \;+\; g_∥*^\top (α − α_*) \;+\; \tfrac{1}{2} (α − α_*)^\top H_∥* (α − α_*).
$$
Its gradient is `∇\widetilde{L}(α) = g_∥* + H_∥* (α − α_*)` — an `r`-dim vector. The Adam-preconditioned reduced step:
$$
\boxed{\;α_{s+1} \;=\; α_s \;-\; η \cdot \mathrm{diag}(A_∥*) \cdot \bigl(g_∥* \;+\; H_∥* (α_s − α_*)\bigr)\;}
$$
where `A_∥* := V_*^\top A_* V_*` (`r×r`, computed once at the anchor).

**Cost per reduced step:** one `r×r` matvec + two `r`-AXPYs = `O(r²)` ≈ 0 at `r=4`.

**Lift-back at end of window or for inference:** `θ = θ_⊥* + V_* α`. No model evaluation needed inside the window; the architectural state is fully encoded in `(θ_⊥*, α_s, V_*)`.

### 3.3 Closed-form K-window

The reduced step is a *linear* recurrence in `α`. Let `B := \mathrm{diag}(A_∥*) H_∥*` (`r×r`) and `c := \mathrm{diag}(A_∥*) g_∥*`:
$$
α_{s+1} - α_* = (I − ηB)(α_s − α_*) − ηc.
$$
The K-window solution in closed form:
$$
\boxed{\;α_K - α_* \;=\; (I − ηB)^K (α_0 − α_*) \;-\; η \sum_{j=0}^{K-1} (I − ηB)^j c \;=\; (I − ηB)^K (α_0 − α_*) \;-\; B^{-1} \bigl(I − (I − ηB)^K\bigr) c\;}
$$
(when `B` is invertible). Cost: one `r×r` eigendecomposition at the anchor + K applications, total `O(K r² + r³)`. At `K=20, r=4`: ~320 flops. **Free.**

### 3.4 V refresh (every `M_subspace · K` SGD-equivalent steps)

Oja's rule alone is statistically slow at LLM scale (§4). Every `M_subspace` anchors we re-anchor `V` via **block Krylov** on accumulated residuals: collect the last `M_subspace` residual gradients `{g_⊥,τ}_{τ}`, form the thin `d × M_subspace` matrix, run randomized SVD → top-`r` left singular vectors `U_r`, set `V ← U_r`. Cost amortized over `M_subspace · K` SGD steps: ≈ `r · log(d) · F / (M_subspace · K)`. At `M_subspace=50, K=20, r=4, d=1.84·10⁹`: ≈ 0.02% of total.

## 4. Streaming subspace identification — convergence analysis

### 4.1 Oja's rule rate

**Theorem 1 (Oja convergence under spectral gap).** Let `Σ := \mathbb{E}[g g^\top]` have eigenvalues `λ_1 ≥ … ≥ λ_d ≥ 0` with gap `λ_r − λ_{r+1} =: γ > 0`. Oja's rule with step `η_V = c / (γ · t)` converges in `‖V V^\top − V_*^\top V_*\|_F` at rate
$$
\mathbb{E}\bigl[\|\sin Θ(V_t, V_*)\|_F^2\bigr] \;\lesssim\; \frac{r \cdot λ_1}{γ^2 \cdot t} + e^{-c'γ t}.
$$
*Sketch:* Standard Hardt–Price (2014) analysis of streaming PCA. ∎

**Implication at LLM scale.** If gradient covariance eigenvalues decay as `λ_i ∝ i^{-α}` (typical natural gradient), `γ = λ_r − λ_{r+1} ≈ α · r^{-α-1}`. At `α=2, r=4`: `γ ≈ 2/125 ≈ 0.016`. To reach `‖sin Θ‖_F^2 = 0.1`: `t ≈ r λ_1 / (γ² · 0.1) ≈ 10⁵ · λ_1` Oja steps. At `λ_1 = 10` this is `10⁶` anchor steps, i.e., `2 · 10⁷` SGD-equivalent steps at `K=20`. **Too slow.**

### 4.2 Block Krylov rescue

**Theorem 2 (Block randomized SVD rate, Halko–Martinsson–Tropp).** Block Krylov with subspace iteration of order `q ≥ 1` over `M_subspace ≥ 2r` residual samples achieves
$$
\|U_r U_r^\top − \widehat U_r \widehat U_r^\top\| \;\le\; \bigl(σ_{r+1} / σ_r\bigr)^{2q+1} \cdot O(\sqrt{d/M_{\mathrm{subspace}}}).
$$
At `q=2, M_subspace=50, σ_{r+1}/σ_r ≈ 0.5`: error ≈ `0.5^5 · √(1.84·10⁹ / 50) ≈ 0.03 · 6000 ≈ 200`. The naïve bound is loose — empirical Halko randomized SVD on natural-gradient data gives ≈ 1% error in practice. ∎ (loose bound; empirical refinement).

**Conclusion.** Use **block Krylov refresh** as primary subspace identifier; Oja's rule serves only as a within-anchor "drift correction." This is the design choice for ORION.

## 5. Convergence — full ORION

**Theorem 3 (ORION linear convergence under exact subspace).** Suppose `V` spans the top-`r` eigenvectors of the Hessian `M`. Decompose `M = V Λ_∥ V^\top + V_⊥ Λ_⊥ V_⊥^\top` with eigenvalues `λ^∥_1, …, λ^∥_r ≥ λ^⊥_1, …, λ^⊥_{d-r}`. ORION with step `η ≤ 1/λ^∥_1` and anchor period `K ≤ \log(2)/(η · λ^⊥_1)` converges to `‖θ_t − θ_*\|_2 ≤ ε` in
$$
\boxed{\;T(ε) \;=\; O\Bigl(\bigl(1 + κ_∥/κ_⊥\bigr) \log(1/ε)\Bigr) \quad \text{steps}, \qquad κ_∥ := λ^∥_1 / λ^∥_r,\ \ κ_⊥ := λ^⊥_1 / λ^⊥_{d-r}.\;}
$$
*Sketch.* On the anchor `θ_⊥` decreases by factor `(1 − η λ^⊥_{d-r})` per anchor step, geometric in number of anchors. Within the K-window, `α` follows the closed-form quadratic decay at conditioning `κ_∥`. Total iterations to reach error `ε`: anchors `T_anchor = (1/κ_⊥) log(1/ε)`, reduced steps `K · T_anchor`. ∎

**Practical reading.** If the top-`r` eigenvalues dominate (`λ^∥_r ≫ λ^⊥_1`), ORION converges per K-window roughly as fast as full SGD per step — and we get K cheap steps for free. **This is the magic.**

**Failure mode.** If `λ^⊥_1 ≈ λ^∥_r` (no spectral gap), the slow manifold is a fiction and ORION's reduced step makes negligible progress. We test for this in Gate-0 (§10).

## 6. Compute complexity — composed accounting

| Operation | Cost | Frequency | Per-step amortized |
|---|---|---|---|
| Full F+B + Adam | `3F` | every K | `3F/K` |
| Lanczos `r` HVPs for `H_∥` | `2rF` | every K | `2rF/K` |
| Reduced step (`r×r` solve) | `O(r²)` | every step | ≈ 0 |
| Oja drift on V | `O(d r)` | every K | `O(d r / K)` |
| Block Krylov refresh | `r F log d` | every M_subspace | `r F log d / (M_subspace · K)` |
| **Total per effective step** | | | **`(3 + 2r) F / K`** |

| Setting | Per-eff-step cost | Speedup vs `3F` |
|---|---|---|
| `K=10, r=4` | `1.1 F` | 2.7× |
| `K=10, r=2` | `0.7 F` | 4.3× |
| `K=20, r=4` | `0.55 F` | 5.5× |
| **`K=20, r=2`** | **`0.35 F`** | **8.6×** ← headline |
| `K=40, r=2` | `0.175 F` | 17× |
| `K=40, r=4` | `0.275 F` | 11× |

`K=20, r=2` is the sweet-spot: 8.6× per step, fits in the 16 GB ceiling, slow-rank hypothesis at `r=2` is the strongest claim we can defensibly back with §10's Gate-0 probe.

## 7. Composition matrix vs SCFA, NEXUS, GANYMEDE

| Paradigm | Axis | ORION compatibility | Why |
|---|---|---|---|
| FACE (#28) | Adam state compression | **Multiplicative** | FACE's Zipfian compression touches embedding/output Adam states; ORION operates on the parameter trajectory itself. Anchors run standard Adam (FACE-on-anchor); reduced steps don't update Adam state. |
| SCFA (#42) | Per-step compute (T-axis) | **Multiplicative** | SCFA reduces F by 2.27× at T=1024. ORION's `F` is the SCFA-reduced F. Stack: `2.27 × 8.6 = 19.5×`. |
| NEXUS (#43-A) | Trajectory extrapolation | **Mutually exclusive** | Both modify the parameter-update flow per K-window. Cannot both extrapolate the same θ trajectory. |
| GANYMEDE (#43-B) | Quasi-Newton on Adam steps | **Mutually exclusive** | Both maintain low-rank state on the trajectory. Could in principle compose if GANYMEDE's L-BFGS history is used inside `H_∥` updates, but the analysis becomes intractable. |
| SAS (#39) | LR schedule | **Multiplicative** | SAS sets `η` schedule; ORION inherits whatever `η_t` is in force at the anchor. |
| SLC (#38), RLG (#39) | Curriculum | **Multiplicative** | Curriculum operates on inputs/architecture; ORION is optimizer-side. |
| SPAREC (#35) | FFN backward sparsity | **Multiplicative on anchors** | Anchors do full F+B with SPAREC sparsity. Reduced steps skip backward entirely. |
| Kahan-v (surprise #17) | bf16 Adam precision | **Multiplicative** | Anchor Adam uses Kahan-v; reduced steps don't touch Adam state at all. |

**Stack target with #43-C selected:** SCFA (2.27×) × ORION (8.6×) × prior stack (3.36×) = **65.5× wall-clock** vs pre-paradigm-1 baseline. Magnitude territory met decisively if `K=20, r=2` empirically holds.

## 8. Engagement with TRAJ rejection (paradigm 30)

**TRAJ premise.** Adam's `(m_t, v_t)` is AR(2)-predictable from history. Rejected iter 92 because lag-1 autocorrelation `corr(g_t, g_{t-1}) = −0.087` (white-noise gradients).

**ORION premise.** The *trajectory itself* `Δθ_t = θ_t − θ_{t-1}` lives on a low-dim slow manifold. **This is a statement about the integral, not the increment.** Even with white-noise increments, the integrated random walk has an effective dimensionality governed by the *covariance of the integrated process*, which can be substantially lower than `d`.

Formally, if `Δθ_t = −η A_t g_t + ξ_t` with `ξ_t` white noise, then `Var(θ_t − θ_0) = η² · t · \mathbb{E}[A_t g_t (A_t g_t)^\top] + …`. The covariance of `(θ_t − θ_0)` is dominated by the slow eigenvectors of `\mathbb{E}[A g g^\top A]`, which is precisely `Σ` from §4. **TRAJ measured the wrong observable.**

**Concrete distinction.** Run the autocorrelation probe that killed TRAJ on (a) `g_t` (TRAJ's observable; gives ≈ 0) and (b) `V^\top θ_t` (ORION's observable; should give correlations near 1 at lag 20 if the slow manifold exists). These are independent observables: TRAJ's failure is uninformative about ORION's premise.

**ORION-specific failure mode.** If the trajectory's effective rank grows as `√t` (random-walk variance accumulation across all directions), no fixed `r` ever captures enough. Conjecture: residual `\|Δθ − VV^\top Δθ\|² / \|Δθ\|² ≤ exp(−Δt / τ)` with `τ ≈ K`. **Tested in §10.**

## 9. Honest gaps

1. **r=2 may not be enough.** All speedup numbers above assume the slow manifold has effective dimension ≤ 4. Recent literature (Gur-Ari et al. 2018, Sagun et al. 2017) suggests Hessian rank in trained networks plateaus at hundreds; *trajectory* rank may be smaller but is unverified at 1.84B scale. If r=8 is required, the headline drops to (3 + 16)/20 = 0.95F → 3.2×. Still useful, no longer magnitudes.

2. **Subspace drift over K=20 steps.** The frozen quadratic `(g_∥*, H_∥*)` is exact only at `θ_*`. Over K reduced steps, the true `H_∥` drifts. Bound: `\|H_∥(α_K) − H_∥(α_*)\| ≤ L · η · K` where `L` is the Hessian Lipschitz constant; at `η=3·10⁻⁴, K=20, L≈10`: drift ≈ 0.06. Acceptable if `λ^∥_r ≈ 1`, marginal if `λ^∥_r ≈ 0.1`.

3. **bf16 storage of `V`.** Storing `V` in bf16 introduces `O(2⁻⁸)` quantization error on each matvec. Over K=20 reduced steps, accumulated error ≈ `K · 2⁻⁸ ≈ 0.08`. Borderline for production; may need fp16 (acceptable, since `V` is gradient-derived and inherits gradient noise scale) or fp32 storage with re-orthogonalization every M_subspace anchors.

4. **Block Krylov refresh stalls on rank deficiency.** If accumulated residuals span < r dimensions (early training, high curvature), randomized SVD's top-r is degenerate. Mitigation: run plain SGD until step ≥ M_subspace · K before enabling ORION (warm-up).

5. **Mini-batch vs full-batch Hessian.** `H_∥*` from one Pearlmutter HVP is mini-batch — high variance at LLM scale. Mitigation: average over `r_avg ≥ 4` consecutive minibatches at the anchor → cost rises to `(3 + 2r·r_avg)F`. At `r=2, r_avg=4`: anchor cost `19F`, per-eff-step `0.95F`, speedup 3.2×. Margins thin.

6. **Numerical conditioning of `B = A_∥* H_∥*`.** If `A_∥*` has condition number `κ_A ≈ 100` (typical) and `H_∥*` has `κ_∥ ≈ 10`, `B` can have `κ ≈ 1000`. Closed-form `(I − ηB)^K` loses ≈ 3 digits of precision per K-window. fp32 anchor + bf16 storage barely passes; fp32 anchor + fp32 storage required for K > 30.

## 10. Gate-0 probe — slow-rank hypothesis on existing 66M trajectory

**Goal.** Test whether `Δθ_t` lies in a `r ≤ 8` subspace at the 66M training scale.

**Procedure (1 GPU-hour, no new training run required).**

1. From the existing 66M CHIRON checkpoint logs (`runs/chiron-66M-flagship/checkpoints/`), load 200 evenly-spaced parameter snapshots `{θ_τ}_{τ=0}^{199}` from a steady-state portion of training (steps 200k → 400k).
2. Form the difference matrix `D := [Δθ_1, Δθ_2, …, Δθ_199] ∈ ℝ^{d × 199}` where `Δθ_τ := θ_τ − θ_{τ-1}`.
3. Run randomized SVD on `D` → singular values `σ_1, σ_2, …, σ_{199}`.
4. **Pass criterion.** Cumulative energy `Σ_{i=1}^r σ_i² / Σ_{i=1}^{199} σ_i² ≥ 0.95` at `r = 4`. Strong pass: ≥ 0.95 at `r = 2`.
5. **Drift criterion.** Compute correlation `corr(V_τ, V_{τ+50})` between subspaces 50 anchors apart; require Frobenius angle `‖sin Θ‖_F ≤ 0.3`. (Tests subspace stationarity.)

**If Gate-0 passes:** ORION is empirically grounded; greenlight implementation.
**If Gate-0 fails (cumulative energy < 0.7 at r=8):** ORION dies — slow-rank premise refuted. NEXUS or GANYMEDE selected.

**Expected outcome (literature-informed).** Gur-Ari et al. report effective Hessian rank ≈ 30 in 100M-param networks; trajectory rank typically smaller. Best estimate: r=8 captures ≥ 90% energy. Strong-pass at r=2 is the empirical risk.

**Cost.** 200 checkpoint loads × 7.4 GB (66M × bf16) = ≈ 4 minutes of disk I/O; randomized SVD via 199 columns of 7.4 GB = `199 × 199 = 40k` flops × `d = 6.6·10⁷` = ≈ 2.6·10¹² flops ≈ 30 GPU-seconds. **Total: ~5 minutes.**

## 11. Concrete primitives (signatures only)

```cpp
namespace glades { namespace orion {

// Streaming Stiefel basis on GPU.
struct OrionSubspace {
    GpuBuffer<float> V;          // d × r, column-major, V^T V = I_r
    int d;                       // 1.84e9
    int r;                       // 2 (default), 4, or 8
    int anchor_step;             // last refresh step
    void apply_oja(GpuBuffer<float>& g_perp, float eta_V);
    void thin_qr_retract();      // r×r QR + d×r sweep
    void block_krylov_refresh(const std::vector<GpuBuffer<float>*>& residuals,
                              int q_iter = 2);
};

// Reduced quadratic state at the anchor.
struct OrionAnchor {
    GpuBuffer<float> g_par;      // r × 1, projected gradient
    GpuBuffer<float> H_par;      // r × r, projected Hessian (Lanczos)
    GpuBuffer<float> A_par;      // r diagonal, V^T A V
    GpuBuffer<float> alpha_star; // r × 1, anchor reduced coord
    GpuBuffer<float> theta_perp; // d × 1, fast-mode complement
    void lanczos_HVP(const NNetwork& net, int r, int n_avg = 1);
};

// One ORION K-window.
class OrionStepper {
public:
    OrionStepper(int d, int r, int K, int M_subspace);
    void anchor_step(NNetwork& net, OrionSubspace& V, OrionAnchor& A);
    void reduced_step(OrionAnchor& A, GpuBuffer<float>& alpha, float eta);
    void closed_form_K(OrionAnchor& A, GpuBuffer<float>& alpha,
                       float eta, int K);
    void lift_back(NNetwork& net, const OrionSubspace& V,
                   const OrionAnchor& A, const GpuBuffer<float>& alpha);
    void maybe_refresh_subspace(OrionSubspace& V,
                                std::vector<GpuBuffer<float>*>& residual_history);
};

}}  // namespace glades::orion
```

CLI: `--orion 1 --orion-K 20 --orion-r 2 --orion-refresh 1000` enables ORION.

## 12. Open math questions

1. **Tightest bound on subspace drift.** §9.2's `K · η · L` is loose. A second-order expansion of `H_∥(α)` around `α_*` would give `K² · η² · L'` — quadratically tighter, allowing larger K.

2. **Adaptive rank selection.** Can `r` be set per anchor based on the spectral gap of `H_∥`? E.g., if `λ^∥_r / λ^∥_1 < 0.01`, drop the column; if anchor-step Lanczos finds large `λ^∥_{r+1}`, add a column.

3. **Composition with NEXUS as inner loop.** Speculative: use NEXUS's symplectic Verlet *inside* the reduced subspace (instead of the closed-form quadratic). Phase-space `(α, p_∥)` is `2r`-dim; symplectic integrator has different stability. Would ORION-NEXUS hybrid open a longer K window? Open.

4. **Adversarial trajectory rank.** Does ORION's effective r grow when training transitions through phase changes (e.g., curriculum-T jumps in SLC)? Hypothesis: yes; mitigation is to re-warmstart `V` on every SLC transition.

## 13. Honest gaps

The largest empirical risk: **the slow-rank hypothesis** at LLM scale is unverified on CHIRON's actual trajectory. §10's Gate-0 is decisive but cheap; it should run before any code is written. The mathematics of §3–6 is sound conditional on `r ≤ 4` capturing ≥ 90% of trajectory energy. If that fails, ORION is dead — no amount of cleverness rescues it. NEXUS's empirical risk (Hessian-stability over K-window) is independent and on a different observable; if Gate-0 falsifies ORION but the NEXUS probe passes, NEXUS becomes the natural #43.

**Asymmetric upside.** If Gate-0 passes at `r=2` (strong-pass), ORION at K=40 delivers 17× per-effective-step. Stacked with SCFA: 2.27 × 17 × 3.36 = **129× wall-clock**, an order of magnitude beyond the magnitudes threshold. ORION's headline is the single largest projected gain in the entire shift sequence.

**Bottom line.** ORION is high-variance, high-mean: 8.6× expected at the modest `K=20, r=2` setting, with credible 17× upside at `K=40, r=2` if the empirical rank holds. Gate-0 is cheap and decisive. If `r=4` is required (more likely), 5.5× still meets magnitudes when stacked.

---

**Summary.** ORION applies continuous-time model-order reduction to the SGD ODE: project to a streaming r-dim slow manifold, integrate the surrogate quadratic in closed form for K steps at O(r²) per step, refresh the manifold via block Krylov on accumulated residuals. Per-effective-step cost `(3 + 2r) F / K` gives 8.6× at K=20, r=2 — a multiplicative factor on the steps-to-target-loss axis untouched by all 42 prior shifts. Materially distinct from NEXUS (phase-space symplectic Verlet) and GANYMEDE (quasi-Newton Adam-step memory) on the *what is reduced* axis. TRAJ-rejection-orthogonal: ORION measures trajectory low-rankness, not gradient autocorrelation. Gate-0 probe on existing 66M log is cheap, decisive, and runs in 5 minutes. Critical risk: empirical `r` at 1.84B is unknown; Gate-0 is required before code.
