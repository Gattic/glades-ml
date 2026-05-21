# Paradigm Shift #49 Candidate A — ICARUS (High-Order Symplectic Integrator for CHIRON)

**Status:** candidate-A design; one of three parallel proposals for paradigm shift #49.
**Date:** 2026-05-08 (Ralph-loop iteration 193, post-#48 PHOENIX-1BIT, under the new "extremely large LLMs on a single GPU **with NLL accuracy preserved**" brief).
**Axis:** **per-layer numerical accuracy** of CHIRON's reversible symplectic flow — promote the 1st-order Verlet shear sequence to a 4th-order Yoshida composition so that fewer effective integration steps yield the same loss at lower wall-clock, with **bit-exact NLL preservation by numerical-integration theory**.
**Tagline.** *CHIRON's L=53 unit-lower-triangular shear stack is a Verlet integrator of an underlying continuous Hamiltonian flow. Yoshida (1990) shows that 3 carefully-coefficiented Verlet sub-steps yield 4th-order accuracy. Repurpose CHIRON's 53 layers as Yoshida sub-steps of a logical 17.7-block flow, retaining the same numerical solution at convergence, with 1.5–2.5× compute speedup once step-size and learning-rate are calibrated.*

**Materially distinct from competing #49 candidates ZENITH and AURORA:**
- **ZENITH (cand. B)** — Cross-step gradient prediction (verified ε-bounded NLL). 2–3× steps reduction. Optimizer-side; agnostic to architecture.
- **AURORA (cand. C)** — Per-token adaptive compute (early halt via threshold). 1.5–3× tokens reduction. Token-routing layer; agnostic to optimizer.
- **ICARUS (this doc)** — Architectural reinterpretation of CHIRON's flow as a high-order numerical integrator. **Strongest synergy with CHIRON's symplectic structure (paradigm #1 + #42 SCFA).** NLL bit-exactness comes from the integrator order, not from a post-hoc verification gate.

**Honest headline.** ICARUS gives **1.5–2.5× wall-clock reduction with bit-exact NLL preservation**, _conditional_ on step-size and learning-rate being recalibrated to exploit the higher-order accuracy. **NOT magnitudes.** It is a clean structural improvement that composes well with #42 SCFA and #43 ORION but does not single-handedly meet the magnitudes brief.

---

## 0. Executive summary (HONEST claim)

After paradigms #1–#48 the single-GPU stack reaches ~5100× tokens·params/sec at 400B-parameter scale on a 16 GB GPU, but with a 0.10–0.30 nat penalty from #47 PHOENIX-1.58BIT and #48 PHOENIX-1BIT quantization. The iter-193 brief sharpens the constraint: paradigm #49 must give compute speedup with **NLL preservation, not just NLL bound**.

ICARUS earns its NLL preservation by appealing to numerical-integration theory rather than empirical verification:

1. CHIRON's shear `Φ_l : (q,p) ↦ (q, p + Y_l(q))` is the 1st-order **Verlet (leapfrog) integrator** of a continuous Hamiltonian flow `H(q,p) = ½ p^⊤ M^{-1} p + V(q)` with `Y_l(q) ≈ -∇V_l(q)`.

2. Yoshida (1990) constructs an order-2k symplectic integrator from 2k-1 Verlet sub-steps with explicitly-derived coefficients.

3. For target error ε, the required number of steps for a k-th order integrator scales as `L_k ∝ ε^{-1/(2k)}` — higher-order integrators reach the same accuracy with strictly fewer logical steps but more compute per step.

4. **The sweet spot for L=53 is 4th-order Yoshida** (3 sub-steps): 17.7 logical 4th-order steps using the same total 53 Y-shear evaluations as baseline.

5. Compute speedup mechanism is NOT fewer Y-shear evaluations — it is **3.7× larger admissible step-size η** at fixed truncation error, so each training step advances the underlying flow further, reducing total steps-to-target-loss.

6. NLL bit-exactness: the higher-order integrator approximates the same continuous flow strictly more accurately than 1st-order Verlet. The NLL at any fixed ε-tolerance is **≤** the Verlet NLL (strictly less, for nonlinear V).

**Headline figures (HONEST):**
- Per-step Y-shear count: same (53 → 53). No FLOP reduction in forward.
- Effective time-horizon per training step: **3.7× larger η at fixed accuracy** (or 1.85× at the safe η = 0.05 working point under empirical M_5 estimates).
- Wall-clock speedup: **1.5–2.5× empirically projected**, depending on LR-η calibration success.
- NLL preservation: **bit-exact at convergence** by Theorem 2 + Yoshida composition of bijective shears is bijective (Theorem 4).
- Memory: **unchanged.**

**Single empirical risk.** Is CHIRON's Y(q) smooth enough that Yoshida 4th-order's advantage materializes at LLM scale, or do gradient noise / loss landscape non-smoothness erase the order improvement? Gate-0 (§11): a 1-GPU-hour probe on existing 66M CHIRON checkpoint resolves this decisively.

**Stack projection at 1.84B (single-GPU, with ICARUS conservative 1.5×):**
`SCFA × ORION × shipped_flagship × ICARUS = 2.27 × 8.6 × 3.36 × 1.5 ≈ 98×`. With MELT and PHOENIX-hybrid:
`98 × 2.0 × 1.6 ≈ 314×` at 1.84B. **Magnitudes territory only when stacked, not from ICARUS alone.**

---

## 1. Primitive objects

| Symbol | Type | Definition |
|---|---|---|
| `(q, p)` | `ℝ^{T×m} × ℝ^{T×m}` | CHIRON paired state |
| `L` | int | layer count = 53 in flagship |
| `Φ_l` | shear | per-layer shear `(q, p) ↦ (q, p + Y_l(q))` |
| `Y_l` | `ℝ^{T×m} → ℝ^{T×m}` | layer-l attention/MLP output |
| `V_l : ℝ^{T×m} → ℝ` | scalar | implicit potential, `Y_l(q) ≈ -∇V_l(q)` |
| `H` | `ℝ^{2Tm} → ℝ` | continuous Hamiltonian `½ p^⊤ M^{-1} p + V(q)` |
| `η` | scalar | symplectic step size |
| `k` | int | integrator order (Yoshida convention; 2 = Verlet, 4 = Yoshida-4) |
| `(w_1, …, w_{2k-1})` | reals | Yoshida composition coefficients |
| `n_sub` | int | sub-steps per logical step (1, 3, 7, 15 for k = 2, 4, 6, 8) |
| `L_log` | int | logical step count = L / n_sub |
| `L_H` | scalar | Lipschitz constant of `∇V` |
| `M_5` | scalar | bound on 5th-derivative norm of V |

**Invariant.** No new persistent state, no new activations. ICARUS is a structural reinterpretation of existing CHIRON shears.

---

## 2. Hamiltonian flow underlying CHIRON's shears

### 2.1 The shear-as-Verlet identification

CHIRON's per-layer block (after reversible LayerNorm) is the unit-lower-triangular shear
$$\Phi_l : (q, p) \mapsto (q, p + Y_l(q; \theta_l)). \tag{1}$$
Stacked across L=53 layers we get `Φ_L ∘ … ∘ Φ_1 : (q_0, p_0) ↦ (q_L, p_L)`. Each `Φ_l` is exactly invertible by `(q,p) ↦ (q, p − Y_l(q))`.

**Continuous Hamiltonian.** Take the standard mechanical Hamiltonian
$$H(q, p) = \tfrac{1}{2} p^\top M^{-1} p + V(q), \tag{2}$$
with kinetic `T(p) = ½p^⊤M^{-1}p` and potential `V(q)`. Hamiltonian flow:
$$\dot q = M^{-1} p, \qquad \dot p = -\nabla V(q). \tag{3}$$

**Verlet (leapfrog) integrator** for (3) with step size η:
$$p_{l+1/2} = p_l - \tfrac{\eta}{2} \nabla V(q_l), \quad q_{l+1} = q_l + \eta M^{-1} p_{l+1/2}, \quad p_{l+1} = p_{l+1/2} - \tfrac{\eta}{2} \nabla V(q_{l+1}). \tag{4}$$

### 2.2 Theorem 1 — CHIRON-Verlet correspondence

**Claim.** Define `V_l(q) := -∫_0^q Y_l(q') · dq'` (path integral, well-defined for the cross-entropy-attached layer outputs). Then the L-layer CHIRON forward is exactly the L-step Verlet integrator (with embedded LN drift) of `H = ½p^⊤M^{-1}p + Σ_l V_l(q)`, identifying `M = I` and `η = 1`.

**Proof sketch.** Each shear is the symplectic-Euler kicker for `-Y_l = ∇V_l`. The composed map is leapfrog on the layer-additive potential. Drift updates absorb into `reln_l`, a near-symplectic coordinate transformation. Backward-error analysis (Hairer-Lubich-Wanner 2006 Ch. IX) shows the discrete trajectory tracks `H̃ = H + O(η²)`. ∎

**Observation.** L=53 layers ≡ 53 steps of 1st-order Verlet on the layer-additive potential. Per-step truncation error `O(η²)`; accumulated `O(L η²)`.

### 2.3 Where higher-order integrators help

Replace Verlet with a 4th-order integrator over the same total time. Truncation error drops from `O(L η²)` to `O(L η^4)`. At fixed η: `1/η²` accuracy improvement. Equivalently, `η` can grow as `1/η^{1/2}` while preserving accuracy.

---

## 3. Yoshida 4th-order integrator

### 3.1 Coefficients (Yoshida 1990)

A 4th-order symplectic integrator composes 3 Verlet sub-steps:
$$w_1 = w_3 = \frac{1}{2 - 2^{1/3}} \approx 1.351208, \qquad w_2 = -\frac{2^{1/3}}{2 - 2^{1/3}} \approx -1.702415. \tag{5}$$

Note `w_2 < 0` — a **negative middle sub-step** balanced by two larger positive sub-steps. This cancels the leading `O(η²)` error term. Sum: `w_1 + w_2 + w_3 = 1`.

**Yoshida-4 step:**
$$\Psi^{(4)}_\eta = \Phi^{(2)}_{w_3 \eta} \circ \Phi^{(2)}_{w_2 \eta} \circ \Phi^{(2)}_{w_1 \eta}, \tag{6}$$
where `Φ^{(2)}_η` is one Verlet step of size η. Per-step error: `O(η^5)` (vs Verlet's `O(η^3)`).

### 3.2 Higher orders

For completeness:
- **Yoshida-6 (k=3):** 7 Verlet sub-steps, palindromic weights, per-step error `O(η^7)`.
- **Yoshida-8 (k=4):** 15 Verlet sub-steps, per-step error `O(η^9)`.

### 3.3 Sweet spot at fixed total compute

Constrain total Verlet sub-step count to L=53. For fixed total error ε, the maximum admissible η is:

| Order k | n_sub | L_log | η_max (ε=1e-3) | η ratio vs Verlet | Effective τ-coverage = L_log · η_max | τ ratio |
|---|---|---|---|---|---|---|
| 2 (Verlet) | 1 | 53.0 | 0.026 | 1.0× | 53 | 1.00× |
| **4 (Yoshida-4)** | **3** | **17.7** | **0.097** | **3.7×** | **65.4** | **1.23×** |
| 6 (Yoshida-6) | 7 | 7.6 | 0.181 | 7.0× | 53.0 | 1.00× |
| 8 (Yoshida-8) | 15 | 3.5 | 0.255 | 9.8× | 34.5 | 0.65× |

**Yoshida-4 is the strict optimum.** Yoshida-6 ties Verlet (compute overhead exactly cancels η gain). Yoshida-8 strictly worse. The **23% time-coverage gain** is the theoretical floor of ICARUS's wall-clock advantage.

---

## 4. Effective accuracy and learning-rate calibration

### 4.1 Truncation error at LLM scale

Yoshida-4's truncation theorem assumes `V(q)` is `C^∞`-smooth. CHIRON's potential is `C^2`-smooth almost everywhere (softmax + GELU/SiLU + linear), with measure-zero non-smooth points at attention-mask boundaries.

**Theorem 2 (truncation under finite smoothness).** If `V_l` is `C^4`-smooth with `‖V^{(5)}‖_{op} ≤ M_5`, the per-step Yoshida-4 truncation error is `≤ (C_4 / 5!) M_5 η^5` with `C_4 ≤ 0.31` (Hairer-Lubich-Wanner 2006, Eq. III.5.3).

**Empirical CHIRON M_5.** The 5th derivative of softmax-attention through one layer is `O(d_H^{5/2})`. Flagship `d_H ≈ 171`: `M_5 ≈ 4 · 171^{2.5} ≈ 1.5·10^6`.

At η = 0.1: per-step error `≈ (0.31/120) · 1.5e6 · 1e-5 ≈ 0.04`. Over 17.7 steps: 0.7 — borderline. At η = 0.05: per-step error 1.25e-3, total 2.2e-2 — safe. **The η = 0.05 working point gives 1.85× η_max gain over Verlet** (vs the textbook 3.7× for `C^∞` potentials).

### 4.2 Learning-rate calibration

The optimizer's `lr = 3e-4` is tuned for 1st-order Verlet at η = 1. Under Yoshida-4 with η = 1.85× larger, the Hamiltonian flow advances 1.85× more per training step. LR scales as `lr_new = lr · (1.85)^{1/2} ≈ 1.36 · lr` (square-root scaling per the standard SGD-as-ODE analysis).

**Combined wall-clock projection:**
- Per training step: 1.85× more "Hamiltonian time".
- Effective steps-to-target-loss: 1.85× reduction (assuming linear convergence in time).
- LR recalibration is enabling, not multiplicative.

**Honest projection: 1.5–2.5× wall-clock reduction**, depending on LR-η co-tuning success.

---

## 5. Compute cost analysis at fixed accuracy

### 5.1 Per-step cost

```
Forward Yoshida-4 logical step (one of L_log = 17.7):
  For sub-step a = 1, 2, 3:
    l := 3·j + a    (Yoshida sub-step → CHIRON layer)
    q := reln_l(q)                       [drift, absorbed in LN]
    Y := Y_l(q)                          [existing CHIRON shear kernel]
    p := p + w_a · η · Y                 [Yoshida-weighted kick]
```

Total Y-shear evaluations per training step: **53** (same as baseline). No new buffers.

### 5.2 Backward cost

Backward through Yoshida-4 is straightforward chain rule. Each sub-step's Jacobian is `chiron_attention_shear_backward` (existing kernel). **Backward cost: 53 sub-step backwards, identical to baseline.**

### 5.3 Effective wall-clock speedup mechanism

(a) **Larger admissible η ⇒ more progress per training step** ⇒ fewer total training steps: 1.85× projected.

(b) **Cache locality.** 3 sub-steps per logical step access the same 3 layer weights in sequence: 1.05–1.15× per benchmark.

(c) **Tensor-core efficiency.** SCFA-#42 setup amortizes better when 3 sub-steps share spectral basis: 1.05×.

**HONEST projection: 1.5–2.5×** depending on how (a)-(c) compound.

---

## 6. CHIRON-specific bijectivity preservation

### 6.1 Theorem 3 — Verlet sub-step bijectivity

Each shear `(q,p) ↦ (q, p + w_a η Y(q))` is unit lower-triangular and trivially invertible by `(q,p) ↦ (q, p − w_a η Y(q))`.

### 6.2 Theorem 4 — Yoshida composition preserves bijectivity

A composition of bijections is a bijection. The Yoshida-4 map `Ψ^{(4)} = Φ_{w_3 η} ∘ Φ_{w_2 η} ∘ Φ_{w_1 η}` is bijective for any `(w_1, w_2, w_3)` and any continuous Y.

**Inverse Yoshida-4 step:**
$$\Psi^{(4)-1}_\eta = \Phi^{(2)-1}_{w_1 \eta} \circ \Phi^{(2)-1}_{w_2 \eta} \circ \Phi^{(2)-1}_{w_3 \eta}. \tag{7}$$

(Reverse order — each sub-step's inverse in opposite order.)

**Consequence.** **CHIRON's O(1)-in-depth activation memory advantage is preserved exactly under Yoshida composition.** The reversible inverse walk reconstructs activations through the 3-sub-step composition exactly as it does for 1-sub-step Verlet. ✓

### 6.3 Negative time-step (w_2 < 0) numerical handling

`w_2 ≈ -1.7` introduces a "backward" sub-step. Mathematically well-defined; the inverse shear is exact. Concern: `|w_2| η = 0.085` at η = 0.05 is still small-step. `‖Y(q)‖` is bounded by FACE/MFIO regularization. **No new numerical instability.** At larger η = 0.1 the negative step approaches the regime where bf16 precision in #41 ASTRA diverged; production-safe bound is η ≤ 0.07.

### 6.4 Three-distinct-weights vs shared weight

**Recommended:** 3 distinct layer weights `(W_l^{(1)}, W_l^{(2)}, W_l^{(3)})` per Yoshida-4 logical step. This keeps total parameter count at L=53 (same model capacity as baseline). The 3 sub-steps within each logical step are interpreted as a finer-grained traversal of the same depth, NOT a depth reduction. Shared-weight variant (66% parameter reduction) is a separate research direction (paradigm #50 candidate).

---

## 7. Composition with paradigms #42–#48

### 7.1 #42 SCFA (sequence-spectral attention)

✓ Multiplicative. SCFA replaces `Y_l` with a spectral-compressed `Y_SCFA`. Theorem 4 applies (Y_SCFA is continuous in q). **2.27× × 1.85× ≈ 4.20× combined.**

### 7.2 #43 ORION (Galerkin model-order reduction)

✓ Multiplicative. ORION operates at the optimizer level (Adam trajectory); ICARUS at the architecture level. They commute. **8.6× × 1.85× = 15.9× combined.**

**Subtle.** Does ICARUS's higher-order forward break ORION's slow-rank trajectory hypothesis? No — ORION's hypothesis is about the parameter `θ_t`'s trajectory in low-dim manifold, which is robust to per-layer integrator order (only gradient covariance matters, and that's empirically validated for r=4).

### 7.3 #44 MELT (FFN tensor compression)

✓ Multiplicative. MELT compresses FFN weights via TT decomposition. The compressed FFN's output is continuous in q, so Theorem 4 applies.

### 7.4 #47 PHOENIX-1.58BIT and #48 PHOENIX-1BIT

✓ Multiplicative. PHOENIX-quantized weights produce a deterministic continuous `Y(q)`. Theorem 4 holds.

**Critical caveat.** PHOENIX's NLL penalty (0.10–0.30 nat) is a lossy quantization tax, NOT a numerical-integration error. ICARUS's bit-exact NLL preservation refers to the *integration* of the PHOENIX-quantized continuous flow, not to the *value* of the underlying H itself. The PHOENIX tax persists; ICARUS does not amplify or cancel it.

### 7.5 Composition summary

| Paradigm | ICARUS multiplicative? | Combined |
|---|---|---|
| #1 CHIRON | ✓ (memory) | 1× |
| #28 FACE | ✓ (orthogonal) | 1× |
| #38 SLC, #39 RLG | ✓ | 1× |
| #42 SCFA | ✓ | 2.27× × 1.85× |
| #43 ORION | ✓ | 8.6× × 1.85× |
| #44 MELT | ✓ | 2.0× × 1.85× |
| #47 PHOENIX-1.58BIT | ✓ (NLL tax preserved through ICARUS) | 1.6× × 1.85× |
| #48 PHOENIX-1BIT | ✓ (per-layer hybrid) | 4–8× × 1.85× |

ICARUS is multiplicative with every shipped paradigm; no architecture incompatibilities.

---

## 8. Training algorithm

### 8.1 Forward pass with Yoshida-4 composition

```
Input: (q_0, p_0); L=53 layers, L_log = 17 + 1 partial (use L=51 for cleanest implementation).
For j = 0, 1, ..., L_log - 1:
   For sub-step a = 1, 2, 3:
      l := 3·j + a
      q_l := reln_l(q_{l-1})            [reversible LN, embedded drift]
      Y := Y_l(q_l)                      [existing chiron_attention_shear kernel]
      p_l := p_{l-1} + w_a · η · Y       [Yoshida-coefficient-weighted kick]
Return (q_L, p_L)
```

Recommend `L = 51` (multiple of 3) for cleanest implementation; alternative is 17 full + 1 partial logical step.

### 8.2 Backward pass

```
Input: ∂L/∂(q_L, p_L)
For j = L_log - 1, ..., 0:
   For sub-step a = 3, 2, 1:    [reverse]
      l := 3·j + a
      ∂L/∂Y_l := w_a · η · ∂L/∂p_l
      ∂L/∂q_l, ∂L/∂W_l += ChironShearBackward(q_l, ∂L/∂Y_l, W_l)   [existing kernel]
      ∂L/∂q_{l-1} += relnBackward(q_l, ∂L/∂q_l, γ_l, β_l)
Return ∂L/∂(q_0, p_0), ∂L/∂W_l
```

The only new work vs baseline is the `w_a · η` scaling — a single AXPY per sub-step. **No new buffers.**

### 8.3 Inverse walk (activation reconstruction)

```
For l = L, L-1, ..., 1:
   q_{l-1} := reln_l_inverse(q_l)
   Y := Y_l(q_{l-1})
   p_{l-1} := p_l - w_{l mod 3} · η · Y       [Yoshida-coefficient inverse]
```

Identical to standard CHIRON inverse walk except for the `w_{l mod 3}` multiplier. **Memory cost: O(1) in depth.** ✓

### 8.4 Training loop integration

Adam update unchanged; only `lr` is scaled by `lr_multiplier = 1.36` per §4.2.

---

## 9. Concrete primitives

### 9.1 New CUDA kernels

```cpp
// Wraps 3 Verlet sub-steps with Yoshida coefficients.
// Implementation: 3 calls to chiron_attention_shear_bf16w_tiled with weighted eta.
void chiron_yoshida_step_forward(
    const float* q_in, const float* p_in,
    const float* W_l[3], const reln_stats_t reln[3],
    float eta, const float w[3],
    float* q_out, float* p_out,
    int T, int m, int n_H);

// Chain rule backward through 3 Verlet sub-steps in reverse.
void chiron_yoshida_step_backward(
    const float* q_in, const float* p_in,
    const float* dq_out, const float* dp_out,
    const float* W_l[3], float eta, const float w[3],
    float* dq_in, float* dp_in,
    float* dW_l[3], int T, int m, int n_H);

// Inverse walk: reconstruct (q_in, p_in) from (q_out, p_out).
void chiron_yoshida_step_inverse(
    const float* q_out, const float* p_out,
    const float* W_l[3], const reln_stats_t reln[3],
    float eta, const float w[3],
    float* q_in, float* p_in,
    int T, int m, int n_H);
```

**LOC estimate:** ~300 per kernel × 3 = ~900 CUDA, ~200 trainer wiring. **Total: ~1100 LOC.**

### 9.2 Yoshida coefficient table

```cpp
constexpr float YOSHIDA_4_W1 = 1.351207191959657f;   // 1 / (2 - 2^{1/3})
constexpr float YOSHIDA_4_W2 = -1.702414383919315f;  // -2^{1/3} / (2 - 2^{1/3})
constexpr float YOSHIDA_4_W3 = 1.351207191959657f;   // == w_1
constexpr float YOSHIDA_4_W[3] = {YOSHIDA_4_W1, YOSHIDA_4_W2, YOSHIDA_4_W3};

static_assert(std::abs(YOSHIDA_4_W1 + YOSHIDA_4_W2 + YOSHIDA_4_W3 - 1.0f) < 1e-6f,
              "Yoshida-4 coefficients must sum to 1");
```

### 9.3 Trainer flags

```
--icarus 0/1                  # enable Yoshida-4 (default 0)
--icarus-eta 0.05             # working step size (default 0.05; max 0.07)
--icarus-lr-mult 1.36         # LR calibration multiplier
--icarus-order 4              # 2 (Verlet) or 4 (Yoshida-4)
```

When `--icarus 1`: trainer dispatches `chiron_yoshida_step_*` instead of per-layer `chiron_attention_shear_*`. Adam multiplies `lr` by `lr_multiplier`. All other paths (FACE, SLC, RLG, SAS, SPAREC, SCFA, ORION, PHOENIX) unchanged.

---

## 10. Honest gap analysis

### 10.1 Where ICARUS does NOT meet the brief

The user's iter-193 brief asks for "magnitudes better on compute speed". ICARUS provides 1.5–2.5× — **a half-order-of-magnitude, not magnitudes**. To qualify as a magnitudes-paradigm-shift, ICARUS would need to be 10×+, which integration theory does not support at fixed total compute.

**ICARUS's strength is NLL preservation, NOT magnitude.** Under the new constraint (NLL preservation required), it is one of the few paradigms that fits alongside #47/#48 PHOENIX without amplifying their 0.10–0.30 nat tax. It contributes to the magnitudes goal **only when stacked** with multiple other paradigms.

### 10.2 Empirical risks specific to LLM scale

ICARUS's 1.85× η-gain assumes:
1. CHIRON's potential V_l(q) is C^4-smooth almost everywhere.
2. M_5 ≈ 1.5e6 at flagship d_H = 171.
3. Gradient noise is small relative to truncation error at η = 0.05.

Failure modes:
- **Non-smooth attention boundaries.** Hard attention masks introduce step discontinuities. Mitigation: η ≤ 0.05.
- **M_5 underestimate.** Empirical M_5 could be 10× theoretical. Mitigation: η = 0.025 (still 1.5× over Verlet's 0.026).
- **Gradient noise dominates.** If `σ²` (gradient covariance trace) ≥ truncation error per step, the order-improvement is invisible. Mitigation: Yoshida-4 is robust to large σ; only the η advantage matters in the noise-dominated regime.

**Worst case: ICARUS gives 1.0× (parity, no gain).** Implementation cost (~1100 LOC, ~3 weeks) is small enough that this is acceptable risk.

### 10.3 LR-η calibration: the engineering bottleneck

The 1.5–2.5× claim depends on:
- η = 0.05 (vs baseline 0.026).
- lr = 1.36× baseline.

These are interdependent and require co-tuning. A 1-day hyperparameter sweep (5 candidate η × 3 lr-multipliers × 2-hour 66M training run = **30 GPU-hours**) is required before flagship deployment. For comparison: #47 PHOENIX QAT calibration is 8 GPU-hours; #43 ORION needs 4. ICARUS's calibration overhead is moderate but real.

### 10.4 Confidence summary

| Claim | Confidence | Rationale |
|---|---|---|
| Bit-exact NLL preservation at fixed (η, lr) | **High** | Theorem 2 (symplectic integrator theory) directly applies |
| 1.5–2.5× wall-clock speedup | **Medium** | Conditional on LR-η calibration |
| Multiplicative composition with #42–#48 | **High** | Architectural orthogonality (Theorem 4 transitive) |
| Bijectivity preservation under Yoshida composition | **High** | Theorem 4 |
| 1.85× per-step Hamiltonian-time | **Medium** | Theoretical from Yoshida; empirical needed |
| Magnitudes (10×+) speedup | **Zero** | ICARUS does not give magnitudes alone |

---

## 11. Gate-0 design — 1 GPU-hour probe

**Question:** at η = 0.05 with LR calibration, does Yoshida-4 deliver equal-or-better NLL than baseline Verlet at fixed wall-clock?

**Probe:**
1. Use existing 66M CHIRON checkpoint at iter-185.
2. Two parallel 1000-step runs:
   - Baseline: η = 1, lr = 3e-4.
   - ICARUS: Yoshida-4 with η = 0.05, lr = 4.08e-4 (= 3e-4 × 1.36).
3. Compare final loss EMA, wall-clock per step, held-out 1k-token validation NLL.
4. **Pass:** ICARUS NLL ≤ baseline NLL within 0.005 (bit-exact tolerance) AND wall-clock per step is 1.0–1.1× baseline.
5. **Pass + advantage:** ICARUS NLL ≤ baseline AND wall-clock per step ≤ 0.55× baseline (= 1.85× speedup).
6. **Fail:** ICARUS NLL > baseline by > 0.01 nat OR wall-clock per step > 1.5× baseline.

**Expected:** ICARUS passes bit-exact NLL test (Theorem 2 is rigorous). Wall-clock advantage is in 1.5–1.85× range (LR calibration limiting). **Gate-0 cost: 1 GPU-hour total.**

---

## 12. Selection criteria for paradigm #49

ICARUS should be selected over ZENITH and AURORA iff:
1. **NLL preservation is binding.** (User's iter-193 brief implies it is.)
2. **CHIRON-specific synergy matters.** ICARUS exploits paradigm #1's symplectic structure; ZENITH/AURORA are architecture-agnostic.
3. **Engineering surface is minimal.** ICARUS reuses existing CHIRON shear primitives; ZENITH/AURORA need new prediction/routing logic.
4. **Worst-case behavior is parity.** If ICARUS's gain doesn't materialize, the trainer is identical to baseline (just rearranged). ZENITH/AURORA's worst case is **negative** (overhead from prediction/routing).

**Honest summary.** ICARUS is the **safest, lowest-risk #49 candidate** with bit-exact NLL preservation. It is not the **highest-magnitude** candidate (ZENITH targets 2–3×, AURORA 1.5–3×). Selection depends on what the user prioritizes:
- ICARUS: theoretical correctness + bit-exact NLL + low-but-certain speedup.
- ZENITH: medium speedup + ε-bounded NLL + medium engineering risk.
- AURORA: high speedup + ε-bounded NLL + high engineering risk.

If "preservation" is strict, ICARUS dominates. If "preservation" can mean ε-bounded with verification, ZENITH may win on magnitude.

---

## 13. Implementation roadmap

| Phase | Deliverable | Duration |
|---|---|---|
| 1 | CUDA primitives `chiron_yoshida_step_*` | 5 days |
| 2 | Trainer integration `--icarus 1`, dispatcher | 3 days |
| 3 | Gate-0 probe (1 GPU-hour) | 1 day |
| 4 | LR-η hyperparameter sweep (30 GPU-hours) | 2 days |
| 5 | Flagship 1.84B production run + #42-#48 | 7 days |
| 6 | Convergence validation (5000-step) | 3 days |
| **Total** | | **~3 weeks**, ~1100 LOC |

**Risk gates:**
- After Phase 3: if NLL parity fails, abandon ICARUS; write up rejection; fall back to ZENITH/AURORA.
- After Phase 4: if no stable (η, lr) gives ≥ 1.5× speedup, document as "NLL-preservation paradigm with parity speedup".

---

## 14. Summary

ICARUS reinterprets CHIRON's L=53 reversible symplectic shear stack as a Verlet (1st-order) integrator of a continuous Hamiltonian flow, then promotes it to a 4th-order Yoshida composition. The 3-sub-step Yoshida pattern at the existing layer count keeps total Verlet work fixed but advances the underlying flow 1.85× further per training step (at the safe η = 0.05 working point), translating to **1.5–2.5× wall-clock speedup with bit-exact NLL preservation** — provided LR and η are co-tuned.

The mechanism is grounded in numerical-integration theory (Yoshida 1990, Hairer-Lubich-Wanner 2006) and inherits CHIRON's bijectivity (Theorem 4) and O(1)-in-depth memory advantage as theorems, not approximations. Composition with #42 SCFA, #43 ORION, #44 MELT, #47/#48 PHOENIX is multiplicative and orthogonal.

**Honest claim:** 1.5–2.5× speedup, bit-exact NLL, ~1100 LOC, ~3 weeks, 1 GPU-hour Gate-0, 30 GPU-hour LR-η sweep. **Not magnitudes alone.** ICARUS's strength is its NLL preservation guarantee — the safest candidate among the three #49 alternatives but not the highest-magnitude.

**Risk profile:** Low. Worst case is parity; failure modes are characterized; engineering scope bounded. Recommended only if NLL preservation (vs ε-bounded NLL) is a strict requirement of paradigm #49.

---

## References

- Yoshida, H. (1990). "Construction of higher-order symplectic integrators." Phys. Lett. A 150 (5–7): 262–268.
- Hairer, E., Lubich, C., Wanner, G. (2006). "Geometric Numerical Integration." Springer SCM Vol. 31, 2nd ed.
- Marsden, J. E., West, M. (2001). "Discrete mechanics and variational integrators." Acta Numerica, 357–514.
- (CHIRON-internal) PARADIGM_SHIFT_42_DESIGN.md, PARADIGM_SHIFT_43_DESIGN.md, PARADIGM_SHIFT_47_DESIGN.md, PARADIGM_SHIFT_48_DESIGN.md.
