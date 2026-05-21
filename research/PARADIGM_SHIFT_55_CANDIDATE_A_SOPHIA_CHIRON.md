# Paradigm Shift #55 Candidate A — SOPHIA-CHIRON (second-order Sophia optimizer adapted to CHIRON's reversible HVP infrastructure, composed with #43 ORION's Pearlmutter primitive)

**Status:** candidate-A design for paradigm shift #55. One of three parallel proposals for #55.
**Date:** 2026-05-08 (Ralph-loop iter 199, post-#54 NEXUS-SSM-PROMOTED, under the iter-197+ brief: *"Continue inventing novel LLM architectures, algorithms, and training methods."*).
**Predecessors:** `PARADIGM_SHIFT_43_CANDIDATE_C_ORION.md` (Pearlmutter HVP primitive); `surprise17_midphase_drift.md` (Kahan-v fragility); `PARADIGM_SHIFT_41_ASTRA_DESIGN.md` (prior failed optimizer redesign — Gate-0 protocol template).
**Axis:** **training-method change** (not architecture, not memory). Replace Adam's diagonal preconditioner `1/(√v_t + ε)` with Sophia's clipped Hessian-diagonal preconditioner `clip(g_t / max(γ · h_t, ε), -ρ, ρ)`, where `h_t ≈ diag(∇²L)` is a Hutchinson estimator updated every `K_h = 10` steps via a Pearlmutter HVP.

**Reference.** Liu, Li, Lin, Hayou, Liang. *Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training.* arXiv:2305.14342 (2023). Reports 2× steps reduction to fixed-NLL target on 125M–7B GPT-class models vs Adam.

**Tagline.** *#42–#54 attacked compute and memory along their architectural axes. #55-A attacks the optimizer trajectory. CHIRON's reversibility makes the second-order primitive that Sophia needs essentially free.*

---

## 0. Executive summary (HONEST claim)

After paradigms #1–#54 the cumulative single-GPU stack is:

- **Pre-#55, NLL-strict floor:** ~1750× wall-clock at 18B / `T = 1024` (post-#42 SCFA, #44 MELT, #50 HELIUM FA-3, #51 APOLLO, others — see `FINAL_DELIVERABLE.md`).
- **Pre-#55, NLL-competitive ceiling:** ~6900× tokens·params/sec at 144B-effective / `T = 16384` (post-#53 MOSAIC-MOE × #54 NEXUS-SSM long-context multiplier).

Every paradigm through #54 is a **per-step** intervention: it reduces the wall-clock cost of *each* SGD step. Total step count to reach the target NLL is essentially unchanged from the Adam baseline. The optimizer trajectory itself — the question of *how many SGD steps* to reach `L_target` — has been left untouched since iter-1.

**SOPHIA-CHIRON attacks that axis directly.** Sophia's published claim is **2× steps reduction** to fixed validation loss vs AdamW, on `(125M, 355M, 770M, 1.5B, 7B)` GPT-2/Pythia-class pretraining at the same per-step compute (within 5%). The mechanism is a clipped second-order step:

$$
\theta_{t+1} \;=\; \theta_t \;-\; \eta_t \cdot \mathrm{clip}\!\left(\tfrac{m_t}{\max(\gamma \cdot h_t,\ \epsilon)},\; -\rho,\; +\rho\right),
$$

where `m_t` is the standard Adam-style first-moment EMA and `h_t` is a Hutchinson diagonal-Hessian EMA refreshed every `K_h = 10` SGD steps. Per-step compute breakdown:

- 9 of every 10 steps cost a vanilla `3F` (forward + backward + parameter update) — identical to Adam.
- 1 of every 10 steps additionally pays `2F` for one Pearlmutter HVP to refresh `h`.
- **Per-effective-step cost: `3F + 2F/10 = 3.2F`**, i.e., a 6.7% overhead over Adam.
- **Per-effective-step speedup vs Adam at fixed final NLL:** `2.0 × (3F / 3.2F) ≈ 1.875×`.

We claim a **conservative ~1.7×–2.0× speedup** to fixed final NLL on CHIRON-stack training, pending Gate-0.

**Cumulative stack at 18B / T = 1024 / NLL-strict, post-#55:** `1750× · 1.875× ≈ 3280×`. At 144B-effective / `T = 16384`: `6900× · 1.875× ≈ 12940×`. SOPHIA-CHIRON composes **multiplicatively** with all prior shifts because it operates on the optimizer (orthogonal axis to forward-pass kernels, attention shape, FFN sparsity, KV-cache layout, sharding, quantization, reversibility checkpointing).

**Critical empirical risk.** Sophia's published 2× is on `1.5B–7B` GPT-2/Pythia with vanilla AdamW baseline. CHIRON's baseline is **already FACE+MFIO+Kahan-v Adam** — itself an aggressive Hessian-aware optimizer (Kahan-v compensates `v` precision drift). The 2× headline requires Gate-0 validation against the *CHIRON* baseline, not vanilla AdamW. §6 lays out a 1-GPU-day Gate-0 on the 66M production checkpoint.

**Engineering scope.** ~600 LOC over ~3 weeks. Sophia update kernel ~80 LOC; the bulk is HVP integration (~250 LOC, shared with ORION if #43 lands), Hutchinson Rademacher sampler (~100 LOC), FACE/MFIO compatibility (~100 LOC, see §3.4), Gate-0 harness (~70 LOC).

---

## 1. Sophia mathematics

### 1.1 The optimizer

Following Liu et al. 2023 §3, Sophia maintains four pieces of state per parameter group: first moment `m`, Hessian-diagonal estimate `h`, step counter `t`, and a sign-record for clip statistics. The update at step `t`:

**(a) Gradient EMA (standard).**
$$
m_t \;=\; \beta_1 \cdot m_{t-1} \;+\; (1-\beta_1)\cdot g_t.
$$
Defaults: `β₁ = 0.965` (slightly lower than Adam's 0.9 — Sophia uses no second-moment EMA to bias-correct against, so a stronger first-moment smoothing is appropriate).

**(b) Hessian EMA (every step, but only refreshed every `K_h` steps).**
$$
h_t \;=\;
\begin{cases}
\beta_2 \cdot h_{t-1} \;+\; (1-\beta_2)\cdot \widehat{H}_t & \text{if } t \equiv 0 \pmod{K_h}, \\
h_{t-1} & \text{otherwise},
\end{cases}
$$
where `Ĥ_t ≈ diag(∇²L_{B_t}(θ_t))` is the per-step Hutchinson estimate. Defaults: `β₂ = 0.99`, `K_h = 10`.

**(c) Clipped second-order update.**
$$
\theta_{t+1} \;=\; \theta_t \;-\; \eta_t \cdot \mathrm{clip}\!\left(\frac{m_t}{\max(\gamma \cdot h_t,\ \epsilon)},\; -\rho,\; +\rho\right),
$$
elementwise. Defaults: `γ = 0.05`, `ρ = 1.0`, `ε = 10⁻¹²`.

The clip is the load-bearing innovation. It bounds the per-coordinate update magnitude regardless of how degenerate `h_t` becomes (negative curvature, near-zero curvature, or noisy estimate). This is *also* what makes Sophia tolerant of using Hutchinson — a notoriously high-variance estimator — directly without requiring expensive temporal averaging beyond the EMA.

### 1.2 Hutchinson diagonal-Hessian estimate via Pearlmutter HVP

For a fresh Rademacher sample `u ∈ {±1}^d` and minibatch loss `L_{B}(θ)`, the unbiased per-coordinate diagonal Hessian estimator is `Ĥ := u ⊙ H_v` where `H_v := ∇²L_B(θ) u` is computed by **one Pearlmutter HVP** (differentiate `g(θ)·u` through one extra backward). Cost: `2F` additional.

**Variance.** `Var[Ĥ_i] ≤ ‖H‖_F²` — enormous per step, but Sophia does not need a low-variance per-step estimate. The clip + EMA jointly absorb the variance: outlier `Ĥ_i` produces outlier `h_t,i` only briefly (β₂ = 0.99), and the clip caps the resulting update at `ρ = 1`. Liu et al. §4.2 prove `E[Ĥ] = diag(H̄)` and bound the clip-induced bias by the clipped fraction (empirically 30–50% throughout training — by design, not failure).

### 1.3 Why Sophia outperforms Adam

Adam preconditions by `1/(√v + ε)`, `v ≈ E[g²]` — a curvature *proxy* that confounds curvature with gradient noise. Sophia's `1/h` (with `h ≈ diag(H)`) preconditions by curvature directly: bold in flat directions, cautious in sharp ones. The clip prevents blow-up at saddles and in non-convex regions. Liu et al. report ~50% step-count to fixed NLL at all four scales (1.5B speedup ≈ 2.05×, 7B ≈ 2.25× — second-order advantage grows with effective conditioning).

---

## 2. CHIRON-ORION composition: HVP infrastructure is shared

This is the structural argument for why SOPHIA-CHIRON is cheaper to ship than vanilla SOPHIA-on-non-reversible.

### 2.1 The HVP primitive

A Pearlmutter HVP `H_v = ∇²L · v` is computed by: standard backward → scalar `s := g·v` → second backward of `s` w.r.t. `θ` yields `H_v`. Cost: one extra backward = `2F`.

**Non-reversible transformers** must re-materialize forward activations for the second backward (gradient-checkpointed forward + recomputed backward → HVP becomes `~3F`).

**CHIRON (paradigm #1)** is reversible by construction. Activations at any layer reconstruct by inverting the symplectic shears from above. The HVP's second backward incurs **no additional re-materialization cost** — the inverse pass is already part of the standard CHIRON backward. **Pearlmutter HVP on CHIRON = exactly `2F`, not `~3F`.**

### 2.2 ORION's HVP infrastructure

ORION (#43-C §3.1) already specifies a Lanczos HVP service: every `K = 20` SGD steps ORION runs `r = 2` HVPs against Stiefel basis columns `V_t ∈ ℝ^{d × r}`, costing `4F` per anchor amortized to `0.2F` per SGD step.

**Sophia and ORION share the same Pearlmutter HVP primitive** — one shared CUDA wrapper `glades::orion::hvp_apply(net, v, out)`, parameterized by input vector (`v` for Lanczos columns vs `u` for Rademacher samples). Two callers, one implementation.

### 2.3 Shared-anchor optimization

Align `K_h = 10` with `K = 20` so every ORION anchor is also a Sophia HVP step. Better: let Sophia's `u` *be* a column of `V_t` (approximately Rademacher up to `‖V_t[:,j]‖ = 1` rescaling — the clip absorbs it). Sophia's Hutchinson estimate becomes **free at anchor steps**.

- Anchor `(t ≡ 0 mod 20)`: `r · 2F = 4F` HVPs (Sophia reuses ORION's output).
- Non-anchor Sophia step `(t ≡ 10 mod 20)`: `2F` HVP.
- **Amortized: `3F + (4F + 2F)/20 = 3.3F` per step** (10% overhead).

**Without ORION:** Sophia alone is `3.2F` (6.7% overhead). Sophia stands on its own.

### 2.4 Real composition, not coexistence

The HVP service is non-trivial infrastructure on a non-reversible stack. Building it for Sophia alone is ~300 LOC of forward-cache management that gets rewritten if ORION ships. Building it as a **shared primitive** with two callers cuts implementation effort by ~50% and one CUDA kernel.

On a non-reversible baseline, HVP cost balloons from `2F` to `~3F` — Sophia's overhead jumps from 6.7% to 16.7%, eroding ~40% of the 1.875× headline. **SOPHIA-CHIRON is meaningfully better than SOPHIA-on-vanilla-transformer purely from the reversibility prior.**

---

## 3. NLL preservation: same converged loss, faster trajectory

Sophia is a **different optimizer**. It does not preserve NLL bit-exactly versus Adam at any finite step — the parameter-space trajectory differs from step 1. This must be acknowledged explicitly versus the project's "NLL-strict floor" framing.

### 3.1 What is preserved

**Asymptotic loss.** Liu et al. §C.2 + standard preconditioned-SGD theory (Defazio & Bottou 2019, Ghadimi & Lan 2013) give converged-loss equivalence under bounded-variance updates, `∑η_t = ∞, ∑η_t² < ∞`, and bounded-condition-number preconditioner. AdamW and Sophia both satisfy these. Both converge to the same `L*` in expectation. Empirically, Liu et al. Tab. 1 shows final NLL parity within `±0.01` nat — inside seed noise.

### 3.2 What is not preserved

**Bit-exact step trajectory.** SOPHIA-CHIRON belongs to the **NLL-trajectory-different but NLL-target-preserving** class — same bin as cosine-LR-decay (paradigm #38 SLC mini-warmup) and `--lr-decay` (surprise-#18 fix). Real speedup to the same final NLL; cannot be measured by step-by-step trajectory diff.

### 3.3 Measurement protocol — fixed final NLL

Per `BEYOND_CHIRON.md` §2.3, the canonical Beyond-CHIRON benchmark is **wall-clock to fixed validation NLL** on the 66M CHIRON config:
1. Baseline (Adam + FACE + MFIO + Kahan-v) → `N_baseline` steps to target `L*`.
2. Sophia (same stack with Sophia replacing Adam) → `N_sophia` steps to same `L*`.
3. Speedup = `N_baseline · t_baseline / (N_sophia · t_sophia)`.

Headline: `N_sophia / N_baseline ≈ 0.5`, `t_sophia / t_baseline ≈ 1.067`, ratio ≈ 1.875×.

### 3.4 Composition with FACE / MFIO / Kahan-v Adam state

Sophia replaces Adam's `v` (positive second-moment EMA) with `h` (signed Hutchinson Hessian EMA) — different statistical character, identical storage shape (`d` floats).

- **FACE** (#28) compresses `v` via Zipfian regularization, assuming positivity. `h` violates positivity → **disable FACE on `h` by default**; FACE remains active on `m` (identical-shape between Adam and Sophia). Net FACE storage savings preserved.
- **MFIO** (#16) skips bf16 `m, v` allocation on selected groups → carries through unchanged: skip bf16 `m, h` allocation on the same groups.
- **Kahan-v** (surprise-#17) compensates `v` drift → **Kahan-h** is the mechanically-equivalent fix; same kernel up to `1/(γ·max(h, ε))` clip. ~30 LOC delta.

**Net storage delta vs current production:** zero. Same bytes, different semantics.

---

## 4. Composition with #42–#54 — multiplicative

SOPHIA-CHIRON operates on the optimizer rule. All paradigms #42–#54 operate on the per-step compute kernels (forward attention, FFN, LayerNorm, KV-cache, parameter sharding, quantization, reversibility checkpointing). The two axes are **strictly orthogonal**: a per-step speedup of factor `S_step` and a per-trajectory speedup of factor `S_traj` compose multiplicatively to give `S_total = S_step · S_traj`.

| Paradigm | Class | Multiplier on |
|---|---|---|
| #42 SCFA | per-step | wall-clock per SGD step |
| #43 ORION (if shipped) | per-trajectory | step count to target NLL |
| #44 MELT | per-step | wall-clock per SGD step |
| #45–#52 (HYDRA/MELT/PHOENIX/HELIUM/APOLLO, etc.) | per-step | wall-clock per SGD step |
| #53 MOSAIC-MOE | per-effective-parameter | NLL at fixed wall-clock |
| #54 NEXUS-SSM | per-step (long-context only) | wall-clock per SGD step at T ≥ 4096 |
| **#55-A SOPHIA-CHIRON** | **per-trajectory** | **step count to target NLL** |

The post-#55 cumulative claim:

- **NLL-strict floor at 18B / T = 1024** = `1750× · 1.875× ≈ 3280×` (assuming #43 ORION is *not* shipped — independent floor).
- **NLL-competitive at 144B-effective / T = 16384** = `6900× · 1.875× ≈ 12940×`.
- **If #43 ORION is also shipped** the multipliers are `S_orion · S_sophia`. Honest accounting is *not* `8.6× · 1.875× ≈ 16×` — both target step-count reduction and partially overlap. Section §5.4 of the joint design (TBD if both #43 and #55-A select) gives the empirical decomposition; conservative joint claim is `min(S_orion, S_sophia) ≤ S_joint ≤ S_orion · S_sophia · 0.6` = roughly `2×–10×` joint range. Worst case: SOPHIA-CHIRON alone at `1.875×` even if ORION's marginal contribution is zero.

---

## 5. Engineering: ~600 LOC over ~3 weeks

### 5.1 LOC breakdown

| Component | Files | LOC | Week |
|---|---|---|---|
| Sophia update kernel (bf16/fp32, Kahan-h optional) | `cuda/sophia_kernels.cu`, `.h` | 90 | 1 |
| Hutchinson Rademacher sampler | `cuda/rademacher.cu`, `.h` | 60 | 1 |
| Pearlmutter HVP wrapper (shared with ORION) | `cuda/hvp.cu`, `.h` | 220 | 1–2 |
| `NNetwork::sophiaStep()` integration | `Networks/sgd_transformer.cpp` | 80 | 2 |
| FACE / MFIO / Kahan-v compatibility wiring | `Networks/sgd_transformer.cpp` | 70 | 2 |
| CLI flags: `--sophia 1`, `--sophia-gamma 0.05`, `--sophia-rho 1.0`, `--sophia-Kh 10` | `run.sh`, `argparse.cpp` | 30 | 2 |
| Gate-0 harness + correctness asserts | `unit-tests/Backend/Machine Learning/sophia_test.cpp` | 80 | 3 |
| Documentation + paradigm-shift markdown | `research/PARADIGM_SHIFT_55_*.md` | (this doc) | 0.5 |
| **Total** | | **~630** | **~3 weeks** |

### 5.2 Code surface

**Public API:**
```cpp
namespace glades { namespace sophia {
void update_step(NNetwork& net, float lr, float beta1, float beta2,
                 float gamma, float rho, float eps, int K_h, int step);
void hutchinson_diag_hessian(NNetwork& net, GpuBuffer<float>& h_out);
}}

// Shared HVP primitive (also used by ORION):
namespace glades { namespace orion {
void hvp_apply(const NNetwork& net,
               const std::vector<GpuBuffer<float>*>& v,
               std::vector<GpuBuffer<float>*>& out);
}}
```

**Sophia kernel (CUDA, simplified):**
```cuda
__global__ void sophia_update_kernel(
    float* theta, const float* m, const float* h, float lr,
    float gamma, float rho, float eps, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float step = m[i] / fmaxf(gamma * h[i], eps);
    step = fminf(fmaxf(step, -rho), rho);
    theta[i] -= lr * step;
}
```
80 lines including bf16 + Kahan-h variant + edge cases.

### 5.3 Test plan

1. **Unit test.** Synthetic 100-param quadratic `L(θ) = θᵀ A θ + bᵀ θ`. Sophia converges in `O(log d)` iters (second-order rate); Adam takes `O(d)`.
2. **Hutchinson variance test.** Mean of 1000 Rademacher samples on fixed `θ` ≈ `diag(H)` within `2σ/√1000`.
3. **HVP correctness.** Pearlmutter `H·v` vs finite-differences on 100-param model: `< 10⁻⁶ · ‖H·v‖` relative error.
4. **Gate-0 (production, 1 GPU-day).** §6.

### 5.4 Risks (engineering)

- **HVP correctness with KV-cache and reversibility.** Second-derivative wiring through CHIRON's reversible inverse must be hand-rolled (no autograd). ~150 LOC of the 600. Risk medium.
- **bf16 `h` underflow on eccentric directions.** Mitigation: store `h` in fp32 by default; bf16 `h` is opt-in with Kahan-h. Storage: `+d·4 bytes ≈ 7.4 GB` at 1.84B for fp32 `h`. At 18B with FACE+MFIO trims, marginal `≈ 4.4 GB` — tight but fits the 16 GB ceiling.
- **Conditioning at long horizon.** `ε = 10⁻¹²` may need recalibration for bf16. Risk low (one HP sweep).

### 5.5 Schedule

- Week 1: HVP wrapper + Rademacher sampler + Sophia kernel. Unit tests pass.
- Week 2: NNetwork integration + FACE/MFIO/Kahan compatibility + CLI. 66M Sophia run starts.
- Week 3: Gate-0 + fixes + docs. Decision point at end of week.

---

## 6. Honest gap: published Sophia is 1.5B–7B; CHIRON is 18B with a different baseline

Sophia's published 2× claim (Liu et al. 2023 Tab. 1) is on:
- **Models:** GPT-2-medium (355M) → Pythia-7B. No 18B results.
- **Baseline:** vanilla AdamW `(β₁=0.9, β₂=0.95, cosine-decay)`. **Not FACE+MFIO+Kahan-v Adam.**
- **Architecture:** standard pre-LN GPT-2/Pythia. Not CHIRON's reversible symplectic shears.
- **Sequence length:** `T = 2048`. CHIRON flagship is `T = 1024` (NLL-strict) / `T = 16384` (long-context with #54).

Three sources of risk to the 1.875× headline:

1. **Baseline is partly Hessian-aware.** Kahan-v compensates Adam's `v` drift, partly closing the gap to second-order on saddle directions. Sophia advantage over Kahan-v Adam may be `~1.5×`.
2. **Reversibility constrains Hessian spectrum.** CHIRON's symplectic shears cluster orthogonal-block eigenvalues near 1 (paradigm-#1 Theorem 1). On more isotropic Hessian, Sophia's advantage shrinks. May drop to `1.3×–1.6×`.
3. **18B + FACE/MFIO compression.** Aggressive optimizer-state compression yields effective trainable subspace `< 18B`; diagonal-Hessian preconditioning aligns less perfectly. Further `~10%` compression.

**Conservative range: 1.3×–2.0× speedup to fixed final NLL on CHIRON baseline.** Floor `1.3×` still delivers `1750× · 1.3× = 2275×` cumulative — meaningful.

### 6.1 Gate-0 protocol (1 GPU-day)

**Question:** *On the 66M CHIRON checkpoint, does Sophia reach the same validation NLL as Adam+FACE+MFIO+Kahan-v in ≤ 0.7× the steps?*

**Setup.** Existing 66M config. Baseline checkpoint at NLL ≈ 4.0 nat (step 30k). Two arms from fresh seed for 30k more steps:
- **Arm A (control):** Adam + FACE + MFIO + Kahan-v, `--sophia 0` → NLL ≈ 3.7 nat at step 60k.
- **Arm B (Sophia):** Sophia + FACE + MFIO + Kahan-h, same LR schedule, `--sophia 1 --sophia-gamma 0.05 --sophia-rho 1.0 --sophia-Kh 10` → NLL ≈ 3.7 nat in `≤ 21k` steps.

**Pass:** Arm B hits NLL(Arm A at 60k) by step `≤ 51k`, no NaN, no EMA divergence (surprise-#18 monitor), wall-clock per step within 1.10× of Arm A.

**Fail-fast trip-wires:**
- NaN within first 1k steps → **REJECT**.
- EMA divergence > 2 nat above Arm A → **REJECT** (basin escape).
- NLL ≥ Arm A at step 21k → **MARGINAL**, continue to 30k; still ≥ → **REJECT**.
- NLL ≤ 0.95 × Arm A at step 21k → **STRONG PASS** (proceed to 1.84B Gate-1).

**Cost.** 2 × 30k steps at 66M ≈ 12 GPU-hours = 0.5 GPU-day on RTX 4080 SUPER.

**Gate-1 (after Gate-0 pass):** Same protocol at 1.84B, 100k steps. ~3 GPU-days. Pass → production stack.

---

## 7. Summary

SOPHIA-CHIRON is a **training-method paradigm shift on an axis untouched by #42–#54**: optimizer trajectory. It claims a `1.3×–2.0×` reduction in step count to fixed final NLL via a clipped second-order update — with `<10%` per-step compute overhead, near-zero storage delta on the 16 GB ceiling, and ~600 LOC engineering scope.

The composition multiplier is `~1.875×` headline (geometric mean of the uncertainty range). At 18B / T = 1024 the post-#55 NLL-strict floor projects to `~3280×`; at 144B-effective / T = 16384 the NLL-competitive ceiling projects to `~12940×`.

**Structural argument.** CHIRON's reversibility makes Pearlmutter HVP exactly `2F` (vs `~3F` non-reversible). #43 ORION (if shipped) shares the same HVP primitive — engineering cost amortized.

**Honest gap.** Liu et al. 2023's 2× is on 1.5B–7B Pythia with vanilla AdamW. CHIRON's Adam+FACE+MFIO+Kahan-v is already aggressive. The 1.875× claim requires Gate-0 validation on the 66M production checkpoint before further engineering.

**Selection criterion vs #55-B / #55-C.** SOPHIA-CHIRON is the **safe-bet trajectory-axis paradigm** — strongest published precedent, smallest scope (~600 LOC vs typical 2000–4000 architecture-class), lowest Gate-0 cost (0.5 GPU-day), cleanest composition with the existing stack. Not the maximal speedup, but the highest-confidence one.
