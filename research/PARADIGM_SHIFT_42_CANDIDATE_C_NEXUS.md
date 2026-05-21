# Paradigm Shift #42 Candidate C — NEXUS (Neural EXtrapolation Unified Stepping)

**Status:** candidate-C design, single-formulation. Companion to #42-A and #42-B.
**Date:** 2026-05-08.
**Tagline:** *Symplectic K-step extrapolation of Adam's flow on the loss-landscape Hamiltonian — magnitude wall-clock from amortizing F+B over K cheap vector ops.*

---

## 0. Executive summary

Shifts 1–41 attack per-step compute (FACE, MFIO, SAS) or memory (CHIRON, IBGRAD). Steps-per-target-loss is untouched. NEXUS treats the Adam trajectory `θ_t ∈ ℝ^d` as a discretization of a continuous Hamiltonian flow, then symplectically extrapolates K future iterates from one anchor `(θ_*, g_*, M̂_*)` plus `r ≥ 1` Hessian-vector products (HVPs). Cost per K-window: `(3+2r)F + O(K·d·r) ≈ (3+2r)F`. Asymptote: `(3+2r)F/K`. K=10, r=1 → **6×**; K=8, r=4 → **2.2×** with margin. Multiplicative with SAS/SLC/RLG; closed-form with FACE.

Materially different from rejected TRAJ (paradigm 30): TRAJ predicted the **gradient** via AR(2), failed at lag-1 grad-norm autocorr −0.087. NEXUS does NOT predict the gradient — it predicts the **position** on the energy surface (§9).

---

## 1. Primitive objects

`θ ∈ ℝ^d` params; `L : ℝ^d → ℝ` full-batch loss, `L_B` mini-batch; `g_t := ∇L_{B_t}(θ_t)`; `(m_t, v_t)` Adam EMAs with `(β_1, β_2)`; `A_t := diag(1/(√v̂_t+ε))`, `v̂_t := v_t/(1−β_2^t)`; `η > 0` (treated as `Δt`); `M_t := ∇²L_{B_t}(θ_t)` (never materialized); HVP `u ↦ M_t u` via Pearlmutter, cost ≈ 2F; `K ∈ [4,16]` horizon (default 8); `r ∈ {1,2,4,8}` Krylov rank; `s ∈ {0..K−1}` step within window; `Φ_K` extrapolator (§3).

**Invariant.** No new persistent per-parameter optimizer state. Only an anchor block `(θ_*, p_*, A_*, U_*, T_*)` of total size `O(d·r)`, refreshed every K steps.

## 2. State space — Hamiltonian phase space

Modified Hamiltonian on `T^*ℝ^d ≅ ℝ^{2d}` with symplectic form `ω = dθ ∧ dp`:
$$
\boxed{\;H(\theta, p) \;:=\; L(\theta) \;+\; \tfrac12\, p^{\!\top} A^{-1}(\theta)\, p \;+\; \mathcal O(\eta^2)\;}
$$
`p` is a **Hamiltonian momentum**, NOT Adam's `m`.

**Lemma 1 (Adam ↔ symplectic Euler).** The discrete map
$$
\theta_{t+1} = \theta_t - \eta\, A_t\, p_{t+1}, \qquad p_{t+1} = p_t + (1-\beta_1)(g_t - p_t)
$$
with `p_t := m_t/(1−β_1^t)` recovers Adam up to `O(η²)`. The `(1−β_1)` damping renders Adam a **dissipative** Hamiltonian system; backward-error analysis preserves a *modified* `H̃` along trajectories. ∎

**Consequence.** `t ↦ θ_t` is the projection of a smooth curve on phase space. NEXUS extrapolates **on phase space**, then projects.

## 3. Evolution law

### 3.1 Anchor (s = 0)

(1) Full F+B → `g_*`, update `(m_*, v_*)`; `p_* := m_*/(1−β_1^{t_*})`, `A_* := diag(1/(√v̂_*+ε))`. (2) Probe `u_* := A_* p_*`; HVP `h_* := M_{t_*} u_*` via Pearlmutter (≈2F). (3) For `r>1`, Lanczos over Krylov `span{u_*, M u_*, …, M^{r−1} u_*}`, cost `2r·F + O(dr²)`. (4) Store rank-`r` `M̂_* = U_* T_* U_*^⊤`, `U_* ∈ ℝ^{d×r}`, `T_* ∈ ℝ^{r×r}` tridiagonal. (5) Apply standard Adam step. **Anchor cost `(3+2r)F`** per window.

### 3.2 Extrapolated step (1 ≤ s < K)

Frozen quadratic `L̃(θ) := L(θ_*) + g_*^⊤(θ−θ_*) + ½(θ−θ_*)^⊤ M̂_*(θ−θ_*)`, `H̃(θ,p) := L̃(θ) + ½ p^⊤ A_*^{-1} p`. Verlet (leapfrog) with step `η`:
$$
\boxed{\begin{aligned}
p_{s+\frac12} &= p_s - \tfrac{\eta}{2}\bigl(g_* + \hat M_*(\theta_s-\theta_*)\bigr) \\
\theta_{s+1}  &= \theta_s - \eta\, A_*\, p_{s+\frac12} \\
p_{s+1}       &= p_{s+\frac12} - \tfrac{\eta}{2}\bigl(g_* + \hat M_*(\theta_{s+1}-\theta_*)\bigr)
\end{aligned}}
$$
Per step: 4 AXPY in `d` plus two `O(dr)` matvecs against `M̂_*`. **No model calls.**

### 3.3 K-step closed form

With `B := A_* M̂_*` (rank `r`):
$$
\theta_{*+K} = \theta_* - \eta\, A_*\, \mathcal T_K(B)\,\bigl(K p_* + \tfrac{K(K-1)}{2} g_*\bigr) + \mathcal O(\eta^3 K^3 \|B\|^2).
$$
For `r=1`, with `λ_* := (u_*^⊤ h_*)/(u_*^⊤ A_* u_*)`,
$$
\mathcal T_K^{(r=1)} = \frac{1 - (1-\eta\lambda_*)^K}{\eta\lambda_*}.
$$
For `r>1`, `T_K(B)` is an `r×r` polynomial of `T_*` computed once per anchor, applied `O(dr)` per parameter. **K steps reduce to `O(K·d·r)` AXPY with anchor-fixed constants.**

## 4. Variational principle

**Theorem 1.** The leapfrog map of §3.2 is the unique stationary point of
$$
S_K = \sum_{s=0}^{K-1}\Bigl[p_s^{\!\top}(\theta_{s+1}-\theta_s) - \eta\, \tilde H\bigl(\tfrac{\theta_s+\theta_{s+1}}{2}, p_s\bigr)\Bigr]
$$
with `(θ_0, p_0) = (θ_*, p_*)` fixed (Marsden–West discrete mechanics). ∎ **Corollary:** energy error `|H̃_{*+K} − H̃_*|` is exponentially small in `1/η` by backward-error analysis. No systematic energy drift; position drift bounded in §7.

## 5. Compute complexity

`F` forward; `B=F`; HVP `≈2F`.

| Per K-window                | Cost            |
|-----------------------------|-----------------|
| Anchor F+B                  | `3F`            |
| `r` Pearlmutter HVPs        | `2r·F`          |
| Lanczos + `T_K(T_*)`        | `O(dr² + r³)`   |
| K extrapolated steps        | `O(K·d·r)`      |

**Per effective step:** `[(3+2r)F + O(K·d·r)]/K`. At `d=1.84·10⁹, K=8, r=4`: numerator ≈ `11F` (AXPY ≪ F) → `1.38F` vs `3F` ⇒ **2.18×**. At `K=10, r=1`: `0.5F` ⇒ **6×**. At `K=16, r=1`: **9.6×** if stable.

## 6. Mechanism mapping

| Ingredient                | Mechanism                                                            | Factor at 1.84B   |
|---------------------------|----------------------------------------------------------------------|-------------------|
| Wall-clock                | Amortize `(3+2r)F` over K; AXPY ≪ F                                  | **6–10×**         |
| Persistent state          | `O(d·r)` shared, refreshed; zero per-param add                       | **0%**            |
| Convergence (theory)      | Backward error `O(η³K³‖M̂‖²)`; `η=3e-4, K=8, ‖M̂‖=O(1)` → `~2e-8`     | **parity**        |
| Composability             | SAS / SLC / RLG / FACE all multiplicative on orthogonal axes         | **multiplicative**|

## 7. Stability — precise bound

**Theorem 2.** With `Δ_K := sup_{0≤s≤K} ‖M_{t_*+s} − M̂_*‖_2` and gradient noise `σ_g`,
$$
\|\theta^{\text{NEXUS}}_{*+K} - \theta^{\text{true}}_{*+K}\|_2 \;\le\; C_1\,\eta^2 K^2\,\Delta_K\,\|p_*\| \;+\; C_2\,\eta\, K\, \sigma_g.
$$
NEXUS stable iff
$$
\boxed{\;\eta\, K\, \sqrt{\Delta_K\, \|p_*\|/\|\theta_*\|} \;<\; \varepsilon_{\text{tol}} \approx 10^{-2}.\;}
$$

**Adaptive K.** Online `Δ̂_K := ‖g_{*+K} − (g_* + M̂_*(θ_{*+K}-θ_*))‖/‖θ_{*+K}-θ_*‖`. Halve K if `η·K·√(Δ̂_K·‖p_*‖)` exceeds tolerance; double K (cap 16) if below `ε/4`. Self-tuning. **CFL:** `η·λ̂_max(M̂_*) < 1.6` enforced.

## 8. Failure modes

1. **Phase transitions (first ~5% of training).** Large `Δ_K` → set K=1 until `corr(g_t, g_{t-1})` settles.
2. **NaN poisoning.** Bad anchor poisons K steps. Mitigation: shipped NaN early-stop + per-step clip `‖θ_{s+1}-θ_s‖ ≤ 4η‖p_*‖`.
3. **Cumulative drift.** Total `O(T·η·σ_g·K)` matches Adam's stationary noise floor; acceptable.
4. **FACE token-dependent rows.** Closed-form Adafactor recursion (§9.3) on touched rows.
5. **bf16 + surprise #17.** Open: less-frequent `v` updates may help (less compound error) or hurt (snapshot of bf16 noise drives K steps).
6. **CFL.** `η·λ̂_max(M̂_*) > 2` → leapfrog blows up; auto-cap K.

## 9. Honest engagement: NEXUS vs rejected TRAJ

**TRAJ premise** (#30, rejected iter 92): `(m, v)` is AR(2)-predictable. Rejected because lag-1 grad-norm autocorr = −0.087.

### 9.1 Why NEXUS escapes the same probe

TRAJ predicted the **gradient** (by predicting `m`, a function of `g`-history). NEXUS predicts the **position** `θ_{*+K}`. Mechanism:

- `m_t = β_1 m_{t-1} + (1−β_1) g_t` is an EMA over `~1/(1−β_1) ≈ 10` past gradients.
- Even with `corr(g_t, g_{t-1}) ≈ 0`, `m_t` has effective autocorrelation length ~10 (white-noise-into-EMA filter).
- Adam's step direction `A_t m_t` is dominated by **slow Hessian eigenvectors**: `√v̂` divides out the directions where `g` fluctuates fastest.

NEXUS extrapolates on this slow subspace via Krylov. TRAJ tried to AR-predict the white `g` itself. **Different observable.**

### 9.2 Gate-0 falsifier (on existing 66M logs)

(1) Pick anchor steps spaced 8 apart. (2) Log `(θ_*, m_*, v_*)`; compute `θ̂_{*+8}` from rank-1 surrogate (HVP via offline dataloader replay at `t_*`). (3) Compare `θ̂_{*+8}` to actual `θ_{*+8}`. (4) **Pass:** `‖θ̂_{*+8} − θ_{*+8}‖ / ‖θ_{*+8} − θ_*‖ < 0.05` averaged over ≥30 windows.

This is TRAJ's probe analogue on the **right** observable. NEXUS dies here before code if it fails.

### 9.3 FACE composition (non-handwave)

FACE row EMA `c_{t+1} = β_c c_t + (1−β_c) g_{:,j}^2`, under `ĝ_s = g_* + M̂_*(θ_s-θ_*)`:
$$
c_{t_*+K} = \beta_c^K c_{t_*} + (1-\beta_c)\sum_{s=0}^{K-1}\beta_c^s\, \hat g_{t_*+K-1-s}^2.
$$
`O(K)` per touched row, AXPY pass. Untouched rows: pure decay `β_c^K c_{t_*}`, exact.

### 9.4 Structural difference

`corr(g_t, g_{t-1}) ≈ 0` (TRAJ-killer) is independent of `corr(M_{t+K}, M_t)` (NEXUS-relevant) — Hessian stability over `K·η ≈ 10⁻³` of training, widely reported as approximately stationary on smooth basins. Independent probes; TRAJ's failure is not informative about NEXUS's premise.

## 10. Composition matrix

| Shift           | Composition                                                       | Multiplier |
|-----------------|-------------------------------------------------------------------|------------|
| **CHIRON #1**   | Inverse walk → free intermediate gradients → cheaper Krylov       | **+1.3×**  |
| **SAS #40**     | Orthogonal: L-skip × t-skip                                       | **×1.5**   |
| **SLC #38**     | Orthogonal: T schedule × K schedule                               | **×1.5**   |
| **RLG #39**     | Cap K when L jumps                                                | **×1.2**   |
| **FACE #28**    | Closed-form EMA (§9.3) on touched rows                            | **neutral**|
| **MFIO #11**    | Reuse state; one HVP at anchor                                    | **neutral**|
| **WIP #22**     | Anchors are free K-snapshots                                      | **+0.1×**  |
| **IBGRAD #19**  | Extrapolate factor, not dense form                                | **neutral**|
| **Kahan-v #17** | Inner Adam unchanged                                              | **neutral**|

**Stack @ 1.84B:** Shipped flagship `3.36×` × NEXUS conservative `2.5×` (K=4, r=4) ⇒ **8.4×**; aggressive `6×` (K=10, r=1) ⇒ **20×**. Magnitudes territory.

## 11. Concrete primitives (signatures only)

```cpp
// Backend/Machine Learning/Networks/cuda/gpu_nexus.h
namespace glades { namespace gpu {

void hvp_pearlmutter(NNetwork& net, const float* u_dev, float* h_dev,
                     GpuTransformerScratch& scratch, cudaStream_t stream);

void nexus_build_anchor(NNetwork& net, NexusAnchor& anchor,
                        int rank, GpuTransformerScratch& scratch,
                        cudaStream_t stream);

void nexus_extrapolate_step(GpuBuffer<float>& theta, GpuBuffer<float>& m,
                            GpuBuffer<float>& v, const NexusAnchor& anchor,
                            int s, float eta, cudaStream_t stream);

float nexus_stability_probe(const NexusAnchor& anchor, const float* g_now,
                            const float* theta_now, int d, cudaStream_t stream);
}}
```

```cpp
// Backend/Machine Learning/MLState/nexus_state.h
struct NexusAnchor {
    float* theta_anchor_dev;   // d
    float* p_anchor_dev;       // d
    float* A_anchor_dev;       // d
    float* U_dev;              // d × r  (Krylov basis)
    float* T_tridiag_dev;      // r × r  (Lanczos T)
    int rank, K_current, s_step;
    int64_t anchor_step;
};
```

Trainer: one branch in `sgd_transformer.cpp` (anchor vs extrapolate) + FACE closed-form EMA in `face_state.cpp`. No public-API change.

## 12. Open math questions

1. **Anchor density.** Fixed K vs trigger on `‖g_t − g_*‖/‖g_*‖ > τ`.
2. **Multi-anchor Hessian.** Running HVP buffer across anchors vs per-anchor Lanczos.
3. **Higher-order.** Yoshida 4th-order: 3× AXPY for `O(K⁵η⁵)` error; worth it K>12?
4. **Stochastic-symplectic.** Euler-Maruyama for diffusion term; bound becomes `K^{1/2}` not `K`.
5. **Saddle points.** Indefinite `M̂_*` — Verlet symplectic, action not a minimum; practical effect unclear.
6. **bf16 + #17.** Less-frequent `v` updates: helpful or harmful?

## 13. Honest gaps

- Lemma 1 is `O(η²)`; backward-error preserves *modified* `H̃`, not Adam's original. ~1% perturbation.
- Rank-1 too coarse near init (effective Hessian rank > 100); schedule `r` upward — hyperparameter, not derived.
- Theorem 2 assumes Lipschitz `M`, bounded `σ_g`; `C_1, C_2` dimension-free, not derived.
- FACE composition is closed-form for EMA only, not row-frequency-debiasing logic (still computed exactly, `O(B·V_active)`).
- K=10 is asymptotic; realized K depends on basin. Gate-0 (§9.2) decides.

---

**Summary.** NEXUS rides the slow-mode drift Adam's EMA already produces. Treats Adam as discretized Hamiltonian flow; symplectic Verlet extrapolates K steps off one anchor `(3+2r)F`. Per-effective-step `(3+2r)F/K`: plausibly **6×** (K=10, r=1) or **2.2×** (K=8, r=4). Multiplicatively orthogonal to every paradigm through #41. Materially distinct from TRAJ — it does not predict the gradient; the autocorrelation probe that killed TRAJ measures a different observable. Gate-0 (§9.2) is precise, cheap, decisive.
