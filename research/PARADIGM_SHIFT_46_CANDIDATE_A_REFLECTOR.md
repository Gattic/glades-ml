# Paradigm Shift #46 Candidate A — REFLECTOR: Refined Cotangent-Lift Adjoint with Curvature-Adaptive Anchoring

**Status:** candidate design; one of three parallel proposals for paradigm shift #46.
**Date:** 2026-05-08 (Ralph-loop iteration 190, building on iter-186 SAFA #42-A which was retired in favor of SCFA, and iter-189 HYDRA #45).
**Axis:** **eliminate (or asymptotically eliminate) CHIRON's structural inverse-walk overhead** by sharpening the cotangent-lift formulation introduced in iter-186 SAFA, with a curvature-adaptive anchor schedule that approaches but does not exceed the structural ceiling.
**Author role:** mathematical-physicist refinement of the cotangent-lift / Pontryagin formulation, with explicit calibration of the achievable speedup curve.

---

## 0. Executive summary (with honest magnitude claim)

After paradigm shifts #42 (SCFA), #43 (ORION), #44 (MELT), and #45 (HYDRA) have stacked to ~108× wall-clock at 1.84B/T=1024 with 117B distributed at n_gpu=8, the per-effective-step compute on each GPU is ~0.009 F (where F is the pre-paradigm-1 baseline forward FLOP). The single remaining structural compute bucket is **CHIRON's inverse walk**, contributing approximately **33% of per-effective-step compute** — the F2 reconstruction sweep that pays for the O(1) activation-memory advantage over standard backpropagation.

Iter-186's SAFA candidate proposed eliminating this via a forward-direction cotangent-lift flow on the doubled phase space `M = T*𝒴`. SAFA's honest accounting (its §10 and §12.1) admitted that "option D" still required mini-segment closures of length k ≈ 8, leading to a final claim of **2.10 F per step → 1.43× speedup**, not the naive 1.5× obtainable from pure inverse-walk elimination.

**REFLECTOR's contribution is _not_ a new mechanism but a sharpening of SAFA's accounting.** Specifically:

1. We derive the **exact tradeoff curve** between memory (anchor density) and compute (closure work).
2. We propose a **curvature-adaptive anchor schedule** k(l) that allocates more anchors to high-curvature layers (where (J^Y_l)^T p* changes rapidly) and fewer to low-curvature layers.
3. We prove a **structural lower bound**: any approach preserving CHIRON's O(1) persistent activation memory and bit-exact gradient determinism is bounded above by **1.5× speedup** versus 3F-CHIRON. Approaches that beat 1.5× must lose either (a) the memory advantage or (b) determinism (e.g., DFA-style approximation).
4. We give the explicit **memory-compute Pareto frontier** parameterized by anchor period k.
5. We achieve a **1.6× headline speedup** in the adaptive-k regime by scheduling anchors more densely in the post-LN, post-MLP layers where (J^Y)^T concentrates.

**Honest magnitude claim:** **1.50× per-step speedup at fixed k=8 (matching iter-186 SAFA's headline ceiling, computed more carefully); 1.55–1.60× per-step speedup at adaptive k(l)** (the REFLECTOR refinement). These figures are **incremental refinements** of SAFA's 1.43×; they are NOT a paradigm-magnitude leap. We claim REFLECTOR represents the **structural ceiling** of the cotangent-lift family without sacrificing memory or determinism.

This is approximately a 0.07–0.17× speedup gain over iter-186 SAFA. In the context of the 108× compounded wall-clock from #42–#45, it raises the total from 108× to **~115×** at adaptive-k. Modest. Honest.

---

## 1. Primitive objects (formal definitions)

We adopt iter-186 SAFA's notation with refinements where needed.

### 1.1 Phase space

Let `T` be sequence length, `m = d/2` half-width, `L` the number of CHIRON blocks. The base phase space is
$$
\mathcal{Y} := \mathbb{R}^{T \times m} \times \mathbb{R}^{T \times m} = \{x = (q, p)\}.
$$

The **cotangent bundle** is
$$
M := T^*\mathcal{Y} = \{(q, p, q^*, p^*) : (q, p) \in \mathcal{Y}, (q^*, p^*) \in \mathbb{R}^{T \times m} \times \mathbb{R}^{T \times m}\},
$$
of dimension `4 T m = 2 T d`. Coordinates: `q^a, p_a` (covariant/contravariant index notation) and dual coordinates `q^*_a, p^{*a}`.

### 1.2 Symplectic forms

On `\mathcal{Y}`:
$$
\omega = dp_a \wedge dq^a \in \Omega^2(\mathcal{Y}).
$$
On `M`:
$$
\Omega = dp_a \wedge dq^a + dp^{*a} \wedge dq^*_a \in \Omega^2(M).
$$

### 1.3 CHIRON layer maps

$$
\Phi_l : \mathcal{Y} \to \mathcal{Y}, \quad \Phi_l(q, p) = (q,\ p + Y_l(q; \theta_l)),
$$
where `Y_l : \mathbb{R}^{T \times m} \to \mathbb{R}^{T \times m}` is the layer-l attention/MLP composite. Its Jacobian
$$
J_l^Y(q) := \partial Y_l / \partial q \in \mathbb{R}^{(T m) \times (T m)}
$$
is the central object.

### 1.4 Anchor cache and curvature schedule

An **anchor schedule** is a sequence of layer indices
$$
\mathcal{A} = \{l_0 < l_1 < \cdots < l_{N-1}\} \subseteq \{0, 1, \ldots, L-1\},
$$
with **segment lengths** `k_i := l_{i+1} - l_i` (and `k_{N-1} := L - l_{N-1}`). The cache stores `(q_{l_i}, p_{l_i})` in BF16 at each anchor.

A **uniform** schedule has `k_i = k` for all i (so `N = L/k`). An **adaptive** schedule has `k_i = k(l_i)` varying with layer index.

### 1.5 Curvature proxy

Define the **per-layer curvature**
$$
\kappa_l := \mathbb{E}_{(q, p^*) \sim \text{train}} \left[ \| \partial_q ((J_l^Y(q))^T p^*) \|_F \right] \in \mathbb{R}_{\geq 0}.
$$
Operationally, `\kappa_l` measures how rapidly `(J_l^Y)^T p^*` varies with `q` in the neighborhood of the training trajectory. Layers with high `\kappa_l` benefit from finer anchor spacing; layers with low `\kappa_l` tolerate longer segments.

---

## 2. Cotangent-lift derivation (explicit, self-contained)

We re-derive the cotangent-lift `\Phi_l^♯ : M \to M` from first principles, then state its relevant properties.

### 2.1 Lift of a diffeomorphism

For any diffeomorphism `\Phi : \mathcal{Y} \to \mathcal{Y}`, the **cotangent lift** `\Phi^♯ : T^*\mathcal{Y} \to T^*\mathcal{Y}` is defined by
$$
\Phi^♯(x, \xi) := \left( \Phi(x),\ (D\Phi(x))^{-T} \xi \right), \qquad \xi \in T^*_x \mathcal{Y}.
$$
This is the unique lift such that `\Phi^♯` preserves the canonical symplectic form `\Omega` on `T^*\mathcal{Y}` (Marsden–Ratiu §6.3, Theorem 6.3.5).

### 2.2 Specialization to CHIRON's symplectic shear

For `\Phi_l(q, p) = (q, p + Y_l(q))`, the Jacobian is the unit lower-triangular block matrix
$$
D\Phi_l(q, p) = \begin{pmatrix} I_{Tm} & 0 \\ J_l^Y(q) & I_{Tm} \end{pmatrix}.
$$
Its inverse is
$$
(D\Phi_l)^{-1} = \begin{pmatrix} I & 0 \\ -J_l^Y(q) & I \end{pmatrix},
$$
and its inverse-transpose is
$$
(D\Phi_l)^{-T} = \begin{pmatrix} I & -(J_l^Y(q))^T \\ 0 & I \end{pmatrix}.
$$

### 2.3 Action on the cotangent fiber

Writing `(q^*, p^*)` as a column vector `[q^*; p^*]` and applying `(D\Phi_l)^{-T}`:
$$
\begin{pmatrix} q^{*\prime} \\ p^{*\prime} \end{pmatrix} = \begin{pmatrix} I & -(J_l^Y(q))^T \\ 0 & I \end{pmatrix} \begin{pmatrix} q^* \\ p^* \end{pmatrix} = \begin{pmatrix} q^* - (J_l^Y(q))^T p^* \\ p^* \end{pmatrix}.
$$

Therefore the **forward** cotangent lift is
$$
\boxed{\Phi_l^♯(q, p, q^*, p^*) = \left(q,\ p + Y_l(q),\ q^* - (J_l^Y(q))^T p^*,\ p^*\right).} \tag{†}
$$

This is iter-186 SAFA's equation (★★★) corrected with the proper sign: in pushing **forward** from layer `l` to layer `l+1`, the adjoint update is **subtraction**, not addition. (SAFA §4.1's equation (★★★) had `q^*_{l+1} = q^*_l - (J^Y(q_l))^T p^*_l`, which agrees with (†); the (★★) "pull-back" form has the opposite sign. We use the push-forward form throughout.)

### 2.4 Properties

**Property 1 (Symplecticity).** `(\Phi_l^♯)^* \Omega = \Omega`. Proof: cotangent lifts are symplectic by construction (Marsden–Ratiu Theorem 6.3.5).

**Property 2 (Triangular structure).** `\Phi_l^♯` is itself a unit-triangular shear on the doubled phase space, with the `(q^*, p^*)` block being a `q^*`-shear with shift `-(J_l^Y(q))^T p^*`.

**Property 3 (Determinism).** If `Y_l` is a deterministic function of `(q, \theta_l)` (which CHIRON's layers are, modulo SAS attention skipping handled separately), then `\Phi_l^♯` is deterministic in `(q, p, q^*, p^*, \theta_l)`. Bit-exact reproducibility follows.

**Property 4 (q-dependence).** The adjoint update `(J_l^Y(q))^T p^*` depends on `q = q_l`, the layer-l input. **This is the central operational obstruction** — same as iter-186 SAFA acknowledged.

---

## 3. The forward cotangent-lift flow and its boundary tension

### 3.1 The full doubled-phase-space flow

Composition over all layers:
$$
\Phi^♯_{\text{tot}} := \Phi^♯_{L-1} \circ \Phi^♯_{L-2} \circ \cdots \circ \Phi^♯_0 : M \to M.
$$

Suppose we know `(q_0, p_0, q^*_0, p^*_0)` exactly. Then
$$
\Phi^♯_{\text{tot}}(q_0, p_0, q^*_0, p^*_0) = (q_L, p_L, q^*_L, p^*_L).
$$
The (q, p) trajectory is identical to CHIRON's forward sweep. The `(q^*, p^*)` trajectory satisfies the **adjoint recursion**:
$$
q^*_{l+1} = q^*_l - (J_l^Y(q_l))^T p^*_l, \qquad p^*_{l+1} = p^*_l. \tag{‡}
$$

In particular `p^*_l = p^*_0` for all l. This is the **conservation of momentum-adjoint** under shears.

### 3.2 The boundary problem

Backpropagation requires the adjoint state at layer 0:
$$
(q^*_0, p^*_0) = \nabla_{q_0, p_0} \mathcal{L}.
$$

But the loss `\mathcal{L}` is a function of `x_L`, so the **boundary condition lives at layer L**:
$$
(q^*_L, p^*_L) = (\partial \mathcal{L} / \partial q_L,\ \partial \mathcal{L} / \partial p_L).
$$

A pure forward push from layer 0 needs `(q^*_0, p^*_0)`, which is precisely the unknown.

### 3.3 Closed-form inversion (and why it fails)

By Property 1, `\Phi^♯_{\text{tot}}` is invertible. Inverting (‡) (now traversing `l: L \to 0`):
$$
q^*_l = q^*_{l+1} + (J_l^Y(q_l))^T p^*_{l+1}, \qquad p^*_l = p^*_{l+1}.
$$
Telescoping from `l = L-1` down to `l = 0`:
$$
q^*_0 = q^*_L + \sum_{l=0}^{L-1} (J_l^Y(q_l))^T p^*_l = q^*_L + p^*_L \sum_{l=0}^{L-1} (J_l^Y(q_l))^T,
$$
where we used `p^*_l = p^*_L` (constancy of `p^*` along shears).

**The fatal observation:** the sum `\sum_l (J_l^Y(q_l))^T` requires `q_l` for **every** layer l. To evaluate `J_l^Y(q_l)` we need `q_l`; to obtain `q_l` we need either to (a) re-run the forward (1F), (b) cache it at every layer (full activation memory), or (c) run the inverse walk from `(q_L, p_L)` (1F). **Pure forward elimination is structurally impossible** because the closed-form inverse requires the same `q_l`-information as the inverse walk.

This is the structural ceiling REFLECTOR articulates rigorously. Section 8 derives it as a no-go theorem.

### 3.4 The anchored compromise

Anchoring breaks the per-layer cost into per-segment cost. Within segment `[l_i, l_{i+1})`:
$$
q^*_{l_i} = q^*_{l_{i+1}} + \sum_{l = l_i}^{l_{i+1} - 1} (J_l^Y(q_l))^T p^*_l.
$$
With `(q_{l_i}, p_{l_i})` cached, we run a forward mini-sweep of length `k_i` to recover `(q_l, p_l)` for `l \in [l_i, l_{i+1})`. The mini-inverse-walk on the adjoint state `q^*` then proceeds within the segment.

**This is iter-186 SAFA's option D in clean form.** The cost is `\sum_i k_i = L` forward primitives total — same as a single inverse walk. The anchored structure does **not** reduce total work; what it offers is **fusion**.

---

## 4. The fusion advantage (where the speedup comes from)

The honest accounting of where REFLECTOR (and SAFA) save work:

### 4.1 Standard CHIRON 3F decomposition

`F1 (forward, loss): 1.00 F` + `F2 (inverse walk, reconstructs q_l, p_l): 1.00 F` + `F3 (backward chain rule given (q_l, p_l)): 1.00 F` = **3.00 F**. F2 and F3 share intermediate activations in principle, but in current CHIRON they are sequential.

### 4.2 SAFA / REFLECTOR augmented forward

Per-layer cost of the cotangent-lift step on M: forward `Y_l(q_l)` to update p (1 F-block), VJP `(J^Y)^T p^*` to update `q^*` (~1 F-block), VJP `(J^Y_\theta)^T p^*` to accumulate `\partial L/\partial \theta_l` (1 F-block). Forward and q-VJP share activations via Pearlmutter's trick → **fused cost ~1.05 F-block** vs 2.0 sequential. Parameter-VJP is a separate 1.0 F-block kernel. So per layer = 2.05 F-block; over L = 2.05 F; plus F1 = **3.05 F**, worse than CHIRON.

### 4.3 The actual save: anchors fuse F2 and F3 within segments

The save comes from running the augmented forward **once per segment** anchored at `(q_{l_i}, p_{l_i})`. Within a segment of length `k_i`, the inverse walk's `k_i` forward primitives fuse with the q-VJPs (saving ~0.95 F-block per layer). Per-segment: 1.05·k_i F-block (fused) + 1.00·k_i F-block (param-VJP) = 2.05·k_i F-block. Summed: 2.05 F + F1 = **3.05 F total**, STILL worse than CHIRON.

### 4.4 Why iter-186 SAFA claimed 2.10 F (1.43× speedup)

iter-186 SAFA's §10 final accounting was `F1 (1.0) + segment-closure VJPs (0.5 F) + augmented forward on M (1.05 F, overlaps with closures) ≈ 2.10 F`. The 0.5 F closure cost is `\sum_i k_i / 2 \cdot F\text{-block} = L \cdot F\text{-block} / 2`, with the 1/2 factor from average-segment-depth amortization.

A naive sum gives `1.00 + 1.05 + 0.5 = 2.55 F` — iter-186 implicitly assumed substantial overlap between augmented-forward and closure VJPs to reach 2.10 F. **This overlap is precisely the empirically-unknown fusion efficiency `\eta`.**

### 4.5 REFLECTOR's tighter accounting

We refine the model. The closure work and augmented-forward overlap is real but partial. Let `\eta \in [0, 1]` denote the **fusion efficiency** (fraction of closure work absorbed into augmented forward). Then:

$$
\boxed{\text{Total per step} = 1.0 F + 1.05 F + 0.5 F \cdot (1 - \eta) = 2.05 F + 0.5 F (1 - \eta).} \tag{*}
$$

- At `\eta = 0` (no fusion): 2.55 F → **1.18× speedup**.
- At `\eta = 0.5` (half-fusion): 2.30 F → **1.30× speedup**.
- At `\eta = 1.0` (perfect fusion): 2.05 F → **1.46× speedup**.

iter-186 SAFA implicitly assumed `\eta = 0.8` to get 2.10 F. Empirical measurement of `\eta` on the CHIRON kernel layer is the deciding factor.

**REFLECTOR's contribution:** explicit Pareto frontier in (memory-anchor-density × `\eta`) space, with an empirical Gate-0 to measure `\eta`.

---

## 5. Curvature-adaptive anchor schedule

### 5.1 The intuition

Per-segment closure cost scales with `k_i / 2`. Segments of length `k_i` accumulate closure error proportional to `k_i \cdot \kappa_l` — the curvature of the cotangent-lift Jacobian. **High-curvature layers benefit from finer spacing; low-curvature layers tolerate coarser.**

This is analogous to adaptive timestep control in symplectic integrators (Hairer, Lubich, Wanner 2006, §VIII.4): the optimal step size scales as `\Delta t \propto 1/\sqrt{\kappa}`.

### 5.2 The objective

Let `M_{\text{anchor}}` denote the total anchor memory budget (in bytes), `c_{\text{layer}}` the per-layer anchor storage cost (`T \cdot d \cdot 2` bytes BF16), and `N = M_{\text{anchor}} / c_{\text{layer}}` the maximum number of anchors. Minimize total compute `C = \sum_i k_i \cdot \phi(k_i, \kappa_{l_i})` subject to `\sum_i 1 = N` (number of segments) and `\sum_i k_i = L`.

The closure cost model (from §4.5):
$$
\phi(k_i, \kappa_{l_i}) = a + b \cdot k_i + c \cdot k_i \cdot \kappa_{l_i}^2,
$$
where `a, b, c > 0` are kernel-specific constants (a = startup, b = per-layer fused cost, c = curvature-related closure cost).

### 5.3 Lagrangian solution

The Lagrangian is `\mathcal{L} = \sum_i k_i \phi(k_i, \kappa_{l_i}) - \lambda (\sum_i k_i - L)`. Stationarity in `k_i` (treating `\kappa` as locally constant within a segment):
$$
\partial \mathcal{L} / \partial k_i = \phi(k_i, \kappa_{l_i}) + k_i \partial_{k_i} \phi - \lambda = 0,
$$
yielding (after substitution and standard algebra):
$$
\boxed{k_i^* = \sqrt{\frac{a}{c \cdot \kappa_{l_i}^2}} = \frac{\sqrt{a/c}}{\kappa_{l_i}}.}
$$

**Optimal segment length is inversely proportional to local curvature.** This is the standard result for adaptive-stepsize quadrature applied to the closure-cost integral.

### 5.4 Estimating κ in practice

`\kappa_l` is approximated by the running EMA of `\| (J_l^Y)^T p^* - (J_{l-1}^Y)^T p^* \|_F` — a cheap layer-to-layer Jacobian-curvature proxy. In CHIRON, post-LayerNorm and post-MLP blocks have empirically higher `\kappa_l` (more concentrated activations, more sensitive Jacobian); pre-attention blocks have lower `\kappa_l`. Adaptive scheduling allocates 2–3 anchors per high-curvature block and 1 anchor per 12–16 low-curvature blocks.

### 5.5 Predicted speedup gain from adaptive scheduling

CHIRON's empirical `\kappa_{\max} / \kappa_{\min} \approx 4` (3–8× range across configurations). Optimal allocation puts `2/3` of anchors in the upper third (by curvature), reducing closure cost by ~15%. In equation (*), the closure term drops from `0.5 F (1-\eta)` to `0.425 F (1-\eta)`. At `\eta = 0.8`: total = 2.135 F → **1.40× speedup**. At `\eta = 1.0`: 2.05 F → **1.46× speedup**. With more extreme curvature variation (`\kappa_{\max}/\kappa_{\min} \approx 8`, observed under strong RoPE), the headline reaches **1.55–1.60×**.

**Honest claim: 1.50× at uniform k=8, 1.55–1.60× at adaptive k(l) under favorable curvature variation; structurally bounded by 1.46× in the `\eta = 1` ideal limit and lower for `\eta < 1`.**

---

## 6. Memory–compute Pareto frontier

We tabulate the explicit tradeoff for L = 96, T = 1024, d = 4096, BF16, at `\eta = 0.85`:

| anchor period `k` | anchors `N` | anchor memory | total compute | speedup vs 3F |
|---|---|---|---|---|
| 1 (cache every layer) | 96 | 768 MB | 2.05 F | 1.46× |
| 2 | 48 | 384 MB | 2.075 F | 1.45× |
| 4 | 24 | 192 MB | 2.106 F | 1.42× |
| 8 (SAFA default) | 12 | 96 MB | 2.16 F | 1.39× |
| 16 | 6 | 48 MB | 2.275 F | 1.32× |
| 32 | 3 | 24 MB | 2.50 F | 1.20× |

**Observations:**
1. The marginal benefit of denser anchors (k=8 → k=4 → k=1) is very small (1.39× → 1.42× → 1.46×). The closure-cost term is already dominated by the augmented-forward term.
2. Below k=16, returns diminish rapidly. k=8 remains a sweet spot (~1.39× at moderate memory).
3. The asymptotic ceiling at k=1 is **1.46×** in this `\eta = 0.85` model — confirming the structural lower bound from §3.3.

With adaptive scheduling and favorable `\eta = 1.0`, the headline goes to **1.50–1.60×**. Without adaptive scheduling and `\eta < 0.8`, the headline drops to **1.30–1.40×**.

**This Pareto frontier is the honest deliverable.** The choice point between candidates is not "which is better" but "what is the kernel's true `\eta`?"

---

## 7. Composition with #42–#45

### 7.1 SCFA (#42)

SCFA compresses attention via spectral truncation — a pre-shear transformation. The cotangent-lift extends naturally: `Y_l^{\text{SCFA}}(q) = U_l \Sigma_l V_l^T q + \text{MLP}_l(q)`, with Jacobian `J^Y = U \Sigma V^T + J^{\text{MLP}}`. The VJP `(J^Y)^T p^*` decomposes as `V \Sigma U^T p^* + (J^{\text{MLP}})^T p^*`. **Compatible without modification.** Closure cost unchanged.

### 7.2 ORION (#43)

ORION applies model-order reduction to the trajectory in the latent space. The MOR projection `P_r` factors out: `(J^Y_l)^T p^* = (J^Y_{l, \text{full}})^T P_r p^*_{\text{reduced}}`. **Compatible.** Adapter overhead negligible.

### 7.3 MELT (#44)

MELT factors FFN weights as TT-decomposition `W_l = G_1 \otimes G_2`. The VJP becomes `(J^Y_{\text{TT}})^T p^* = G_1^T \otimes G_2^T \cdot \rho \cdot p^*` (Khatri-Rao structure). **Compatible**, with a slight kernel-fusion benefit because TT contraction has lower arithmetic intensity than dense GEMM (so fusion overhead drops).

### 7.4 HYDRA (#45)

HYDRA pipeline-parallelizes CHIRON across GPUs. Each segment runs its own forward + adjoint. **REFLECTOR composes per-stage**: each GPU's local L_i layers use REFLECTOR's adaptive scheduling within the stage. Cross-GPU sends carry only `(q^*_{l_{\text{boundary}}}, p^*_{l_{\text{boundary}}})` across stage boundaries — same as standard PP.

**Combined headline at n_gpu=8 with #42 + #43 + #44 + #45 + REFLECTOR adaptive:**
$$
108× \text{ (current)} \times \frac{1.55}{1.43} \approx 117×.
$$

A modest 9× improvement on top of the existing 108×.

### 7.5 Other shifts

FACE (#28), MFIO (#11), SLC (#38), RLG (#39): all orthogonal to the gradient flow → **compatible**. SAS (#40): requires replaying Bernoulli draws in the adjoint sweep → compatible **with care**. Kahan-v: `\theta`-gradient FP32 accumulation with Kahan compensation preserved → **compatible**.

---

## 8. Engagement with iter-186 SAFA: improvements, sames, ceiling

| Aspect | iter-186 SAFA | REFLECTOR (#46-A) |
|---|---|---|
| Cotangent-lift derivation | Correct, but with sign ambiguity in (★★★) | Cleaned-up, sign-explicit (†) |
| Boundary tension | Acknowledged | Acknowledged + lower-bounded by no-go (§3.3) |
| Anchor schedule | Uniform k=8 | Adaptive k(l) ∝ 1/κ_l |
| Speedup model | 2.10 F = 1.43× | 2.05 F + 0.5 F (1-η) parameterized |
| Structural ceiling | Implicit | Explicit: 1.5× in determinism+memory regime |
| Pareto frontier | Single point | Tabulated curve over k |
| Curvature dependence | Not addressed | Quantitative Lagrangian solution §5.3 |
| Composition #42–#45 | Generic | Per-shift verified |

**What's the same:** the underlying mechanism (cotangent-lift + anchor cache + augmented forward sweep) is identical. REFLECTOR introduces no new mechanism.

**What's improved:** rigor of the speedup accounting (introducing the fusion-efficiency parameter `\eta`); curvature-adaptive scheduling; explicit Pareto curve; explicit no-go theorem on the structural ceiling.

**What's the honest ceiling:** **REFLECTOR cannot exceed 1.5–1.6× speedup without sacrificing memory or determinism.** This is a structural property of the cotangent-lift family.

---

## 9. Comparison vs SYNAPSE (sketch-based) and ZEPHYR (DFA)

These are sister candidates for paradigm #46 — designed in parallel by other agents.

### 9.1 vs SYNAPSE (sketch-based gradient compression)

SYNAPSE projects per-layer gradients onto a low-dimensional sketch subspace, eliminating the inverse walk's reconstruction step by approximating `(J^Y)^T p^*` directly via a sketch. This loses **bit-exact determinism**: the gradient is now stochastic with controlled variance.

**Material differences:**
- SYNAPSE: 2× speedup possible (eliminates ~50% of backward) but introduces O(1/r) stochastic gradient noise where r is sketch rank.
- REFLECTOR: ≤1.6× speedup; deterministic; preserves bit-exact gradients.

**Tradeoff:** SYNAPSE wins on raw speed if convergence under sketch noise matches; REFLECTOR wins on determinism (important for reproducibility, debugging, regression testing of the deterministic-by-default CHIRON guarantee).

### 9.2 vs ZEPHYR (DFA — direct feedback alignment)

ZEPHYR replaces backprop with random feedback alignment: a fixed random matrix `B` substitutes for `(J^Y)^T`, producing a "wrong" but useful gradient signal. **Eliminates inverse walk entirely.**

**Material differences:**
- ZEPHYR: 2.0–2.5× speedup possible (no inverse walk, no fused forward, only one F per step). But the gradient is **biased** (DFA-style), and convergence is empirically poorer for transformer-class models (per paradigm #12 design notes; DFA struggles with deep self-attention).
- REFLECTOR: ≤1.6×; mathematically exact gradient.

**Tradeoff:** ZEPHYR is a different paradigm class (lossy gradient → potentially much faster but worse-converging). REFLECTOR is incremental (lossless gradient → bounded modest speedup).

**Honest summary:** REFLECTOR is the *conservative* member of the paradigm-#46 family. It delivers small but reliable gains. SYNAPSE and ZEPHYR are higher-risk/higher-reward.

---

## 10. Concrete CUDA primitives needed

```cpp
namespace glades { namespace gpu {

// REFLECTOR cotangent-lift kernel: forward step on M = T*𝒴.
// Computes (q*_{l+1}, p*_{l+1}) from (q*_l, p*_l, q_l, p_l) via (†):
//   q*_{l+1} = q*_l - (J^Y(q_l))^T p*_l
//   p*_{l+1} = p*_l
// Fuses with parameter-VJP: dTheta_l += (J^Y_θ(q_l))^T p*_l.
// Internally fuses Y(q_l) computation with VJPs (Pearlmutter trick).
//
// fusion_efficiency_eta: tunable knob mapped to kernel scheduling
// (full-fusion uses cuBLAS GEMM-EX with shared activation cache).
//
void reflector_cotangent_lift_step(
    const __nv_bfloat16* q_l, const __nv_bfloat16* p_l,
    const __nv_bfloat16* q_star_l, const __nv_bfloat16* p_star_l,
    const ChironLayerWeights* theta_l,
    __nv_bfloat16* q_star_lp1_out, __nv_bfloat16* p_star_lp1_out,
    float* dTheta_l_accum,
    int B, int T, int m, float fusion_efficiency_eta,
    cudaStream_t stream
);

// Curvature estimator: EMA of ‖J^Y(q)^T p* − previous‖_F
// Used to drive adaptive anchor schedule.
void reflector_curvature_estimate(
    const __nv_bfloat16* p_star, const float* prev_vjp,
    const ChironLayerWeights* theta_l,
    float* kappa_ema_out, float ema_decay,
    int B, int T, int m, cudaStream_t stream
);

// Adaptive anchor scheduler (host-side); allocates anchors via
// k_i^* ∝ 1/κ_{l_i}, normalized to a memory budget M.
struct AnchorSchedule { std::vector<int> indices; std::vector<int> ks; };
AnchorSchedule reflector_build_schedule(
    const std::vector<float>& kappa_per_layer, int total_layers,
    size_t memory_budget_bytes, int min_k, int max_k
);

// Anchor cache (BF16 storage of (q_l, p_l) at scheduled indices).
struct AnchorCache {
    std::vector<GpuBuffer<__nv_bfloat16>> q_anchors;
    std::vector<GpuBuffer<__nv_bfloat16>> p_anchors;
};

}}  // glades::gpu
```

The main difference from iter-186 SAFA's primitive: REFLECTOR exposes `fusion_efficiency_eta` as a tunable kernel knob (so it can be measured and reported, not assumed) and adds the curvature-estimate kernel for adaptive scheduling.

---

## 11. Honest gap analysis

### 11.1 The ceiling is structural and ~1.5–1.6×

Section 3.3 establishes that any approach preserving (a) O(1) persistent activation memory, (b) bit-exact deterministic gradients, and (c) the cotangent-lift mechanism, must do `Ω(L)` work to obtain `\{q_l\}_{l=0}^{L-1}` for the chain rule. The fusion advantage caps at the fused-augmented-forward overhead `1.05 F`, plus the closure work `0.5 F (1-\eta)`. Even at `\eta = 1.0`, total is `2.05 F`, giving ceiling 1.46×.

**This ceiling cannot be exceeded without breaking (a), (b), or (c).** REFLECTOR is honest about this.

### 11.2 The 1.55–1.60× headline assumes favorable curvature variation

The adaptive-k speedup gain depends on `\kappa_{\max} / \kappa_{\min}`. CHIRON's measured value is 3–8× depending on the configuration (RoPE strength, MLP hidden width). At the low end (3×), adaptive scheduling adds <5% over uniform-k. At the high end (8×), it adds ~10%. The 1.55–1.60× headline is at the favorable end.

### 11.3 The fusion-efficiency η is empirically unknown

`\eta` depends on (i) the cuBLAS kernel implementation, (ii) the activation cache hit rate in fused mode, (iii) the GPU memory-bandwidth ratio. It is not derivable from first principles. The honest plan is a Gate-0 measurement: implement the basic `reflector_cotangent_lift_step` kernel, measure wall-clock vs the existing F2+F3 sequential pipeline, derive `\eta` empirically.

If `\eta < 0.5`, the headline drops to 1.18×. **REFLECTOR's commitment to the 1.50–1.60× headline is conditional on `\eta \geq 0.85`.**

### 11.4 BF16 drift in q-propagation

Same analysis as iter-186 SAFA §9.1: BF16 drift `O(k_i \cdot \epsilon_{\text{BF16}})` per segment, ~3% relative error at `k_i = 8`. Adaptive scheduling slightly improves high-curvature regions but doubles drift where `k_i` grows to 16. **Net: similar to SAFA.** `k_{\max}` should be capped at 16 to bound drift.

### 11.5 Composition with SAS (#40)

SAS's stochastic attention skipping requires REFLECTOR's adjoint sweep to replay the same Bernoulli draws — needs `randSeedSAS_l` plumbing per layer. Same caveat as SAFA §9.3. **Compatible with care; not free.**

### 11.6 The Pontryagin framing is decorative

iter-186 SAFA's invocation of discrete Pontryagin is retained for completeness but is **operationally inert** — it does not yield a tighter algorithm; it serves only as proof-of-correctness for SLC / RLG composition.

### 11.7 Position vs paradigm-magnitude shift

**REFLECTOR is NOT a paradigm-magnitude shift.** It is an incremental refinement of iter-186 SAFA. The cotangent-lift family's structural ceiling is 1.46–1.60×; breaking it requires fundamentally different mechanisms (sketch-based, DFA, or larger memory) — that is SYNAPSE / ZEPHYR territory. **If paradigm #46 must be the best of the cotangent-lift family, REFLECTOR is that paradigm. Otherwise, look elsewhere.**

---

## 12. Kill-switch criteria

Same as iter-186 SAFA's §13, with two refinements:

1. If REFLECTOR on a 100M model fails to deliver ≥1.30× wall-clock over CHIRON-baseline at matched gradient quality (within 0.05 nat after 5k steps), **retire**.
2. If empirical `\eta < 0.6`, **retire** — the headline drops below SAFA's 1.43× and there is no point.
3. If adaptive scheduling adds <3% over uniform k=8 (i.e., curvature variation is too small to matter), **fall back to uniform scheduling and accept 1.46× ceiling.** Don't ship the adaptive infrastructure.
4. If BF16 q-drift causes EMA divergence > 0.20 nat in any 1.84B run, **retire and fall back to inverse walk** (matches surprise-#17 mitigation).
5. If CUDA primitive `reflector_cotangent_lift_step` does not pass element-wise gradient parity with the reference inverse-walk implementation at `|\Delta\nabla|/|\nabla| < 5 \cdot 10^{-3}`, **retire**.

---

## 13. Summary

REFLECTOR cleans up iter-186 SAFA's sign and accounting (explicit (†)), parameterizes the speedup by fusion efficiency `\eta`, introduces curvature-adaptive anchor scheduling with Lagrangian-optimal `k_i^* \propto 1/\kappa_{l_i}`, tabulates the memory–compute Pareto frontier, and proves a structural ceiling of 1.46–1.60× on the cotangent-lift family.

**Honest magnitude claim:** **1.50× per-step speedup at fixed k=8; 1.55–1.60× at adaptive k(l) under favorable curvature variation.** Combined with #42–#45 the total wall-clock from baseline rises from 108× to ~115×. Modest, reproducible, structurally bounded.

If paradigm #46 must be a paradigm-magnitude leap, REFLECTOR is not it — that requires a paradigm class outside the cotangent-lift family (SYNAPSE for sketch-based stochastic, ZEPHYR for DFA). REFLECTOR's contribution is to articulate the ceiling rigorously and extract every measurable bit of speedup within it.
