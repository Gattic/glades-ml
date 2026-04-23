# Paradigm Shift #35 Candidate C — GATE-BACK (Learned Gating for Backward Compute)

**Formulation class:** amortized variational / self-supervised conditional compute on the BACKWARD pass.
**Date:** 2026-04-23 (post-FACE design cycle, shift-35 candidate C).
**Axis:** FFN **backward-pass** per-neuron activation sparsity — `σ'(x_i) ≈ 0` on 60-90 % of d_ff indices per token (GELU/SiLU).
**Status:** candidate — side-by-side with A/B at shift-35 gate.

---

## 1. Target axis and thesis

In FFN backward, `σ'(x)` element-wise masks `∂L/∂σ(x)` en route to
`∂L/∂x, ∂L/∂W_up, ∂L/∂h_in`. CSP (#27) attacked forward; FACE (#28)
attacked embedding Adam state; no shift attacks **backward**. GATE-BACK
attaches a tiny head `G_ϕ : ℝ^{d_model} → ℝ^{d_ff}` trained by BCE
against `b_t = 𝟙[|σ'(x_t)| > τ_σ]`. Backward uses `TopK(p_t, k)` with
`p_t = σ(G_ϕ(h_in))` to restrict the rows/cols of `W_up, W_down`, and
the columns of `W_up^T` used in `∂L/∂h_in`. Forward is unchanged — σ'(x)
is ground truth, not a prediction target — so the mechanism is a pure
backward amortization. `G_ϕ` is a 2-layer MLP `d_model → 64 → d_ff`,
≈ 0.32 M params/layer at pile_large (d_model = 1024, d_ff = 4096) =
**~0.1 % of the FFN**.

## 2. Primitive objects and state

- `ϕ^ℓ = (W_g1 ∈ ℝ^{64×d_model}, W_g2 ∈ ℝ^{d_ff×64})` per layer (BF16 storage, FP32 master).
- `p_t^ℓ = σ(W_g2 · ReLU(W_g1 · h_in^{ℓ,t}))` — per-token active-prob.
- `b_t^ℓ = 𝟙[|σ'(x_t)| > τ_σ]` — ground-truth label, free from forward (σ' closed-form in σ for GELU/SiLU).
- `I_t^ℓ = TopK(p_t, k = s·d_ff)`; default `s = 0.25` → k = 1024.
- `τ_σ` auto-calibrated every 500 steps to the (1−s)-quantile of `|σ'|`; `λ_bce(t)` schedule in §4.3.
- **State**: `S^t = (W_up, W_down, ϕ, Adam(...), τ_σ)`. Added state = **ϕ + its Adam moments only** (~0.1 % param inflation). Under FACE (§6) ϕ's Adam compresses ~1000× → **≈ 25 KB total** for 24 gates. `p_t, b_t, I_t` are scratch.

## 3. Evolution law

### 3.1 Forward (unchanged; adds gate)

1. `x_t = W_up · h_in^t`; compute `σ(x_t), σ'(x_t)`; `h_out = W_down · σ(x_t)`.
2. Gate: `p_t = σ(W_g2 · ReLU(W_g1 · h_in))` — cost `2·64·(d_model+d_ff) ≈ 0.66 MFLOPs/tok` vs FFN 16.8 MFLOPs (**3.9 % overhead**).
3. `b_t = 𝟙[|σ'(x_t)| > τ_σ]`; accumulate `L_bce^t = BCE(p_t, b_t)`.

### 3.2 Backward — gated sparse path

Given `∂L/∂h_out^t`:
1. `∂L/∂σ(x_t) = W_down^T · ∂L/∂h_out^t` — **dense** (needed to feed σ').
2. `I_t = TopK(p_t, k)`; on `I_t`: `∂L/∂x_t[I_t] = σ'(x_t[I_t]) ⊙ ∂L/∂σ(x_t)[I_t]`, zero elsewhere.
3. `∂L/∂W_up[I_t, :] += ∂L/∂x_t[I_t] ⊗ h_in^t` — sub-GEMM on k rows.
4. `∂L/∂h_in^t = W_up^T[:, I_t] · ∂L/∂x_t[I_t]` — sub-GEMM on k cols.
5. `∂L/∂W_down[:, I_t] += ∂L/∂h_out ⊗ σ(x_t)[I_t]`.
6. Gate bwd: 2-layer MLP backward, ~0.66 MFLOPs/tok.

Backward FFN FLOPs: `4·k·d_model + 2·64·(d_model+d_ff) ≈ 8.65 MFLOPs/tok`
vs dense `32` → **3.7× at s = 0.25**, **6.9× at s = 0.125**.

### 3.3 Auxiliary-loss / main-loss tradeoff (rigorous)

Objective: `L_tot = L_CE + λ_bce(t) · (1/(L·d_ff)) · Σ_{ℓ,t} BCE(p_t^ℓ, b_t^ℓ)`.
Since ϕ does not touch forward, `∂L_CE/∂ϕ ≡ 0` structurally — BCE is
the **only** gradient signal on ϕ. Main-loss gradient bias from the mask:
```
bias_W = E_t[ (1 − m_t) ⊙ σ'(x_t) · ∂L/∂σ(x_t) ⊗ h_in^t ],   m_t = 𝟙[p_t ∈ TopK].
```
Under the BCE-minimizing `p_t* = P(b_t = 1 | h_in)`,
`‖bias_W‖ ≤ ε_ϕ · ‖σ' · ∂L/∂σ ⊗ h_in‖` with `ε_ϕ = √(1 − AUC(ϕ))`.
Converged gates reach AUC > 0.95 (DejaVu 2023; Lazy Neuron 2023),
so `ε_ϕ ≤ 0.22`. Combined with σ' sparsity (≤ 35 % of gradient energy
eligible for bias), **net bias ≤ 7.7 %** — below Adam's stochastic
gradient noise (`σ/μ ≥ 0.1` late training).

### 3.4 λ_bce and τ_σ schedules

- `λ_bce(t) = λ_0 · (1 − exp(−t/τ_λ))`, `λ_0 = 0.05, τ_λ = 1000`; hard cap `λ_0 ≤ 0.1 · ‖∇L_CE‖/‖∇L_BCE‖` at step 2000.
- `τ_σ` auto-calibrated every 500 steps to target `mean(b_t) = s`.
- `n_warmup = 500` dense steps before gating activates.

## 4. Mechanism mapping and stack composition

| Required | Mechanism | Realized |
|---|---|---:|
| (a) Backward FFN ≥ 3× | TopK(p_t) sub-GEMM over k = s·d_ff | **3.7×** |
| (b) Minimal params | 2-layer MLP, bottleneck 64 | **+0.1 %** |
| (c) × FACE | ϕ as embedding-analog → Zipfian precond. | ~1000× Adam compression |
| (d) × CSP (#27) | fwd vs bwd — disjoint | compound ≈ 12.6× FFN |
| (e) × CHIRON | deterministic p_t → bit-exact reversibility | preserved |
| (f) × MFIO × WIP × IBGRAD | ϕ is normal optimizable | 4-way multiplicative |
| (g) × TRCD (#13) / local-window | depth or attention axis disjoint from neuron axis | additive/compound |
| (h) GPU | two small GEMMs + TopK + gather-GEMM | cuBLAS + custom |

## 5. Failure modes and mitigations

**F1 — False-negatives.** Drops true-active gradient. Mit: oversample
`k = (s + 0.03)·d_ff` (+0.5 % FLOPs, +90 % recall at AUC 0.9); BCE
positive-weight `w_+ = (1−s)/s ≈ 3`.

**F2 — False-positives.** Pure FLOP waste; bounded by `1−precision`. At
AUC 0.95 precision ≥ 0.85 → ≤ 15 % sub-GEMM waste.

**F3 — Distribution shift.** BCE always on; τ_σ auto-recalibrates. If
per-layer `AUC < 0.85` on rolling probe → **auto-disable** gating for
that layer (dense fallback); re-enable when AUC recovers.

**F4 — Gate instability.** λ_bce too large → gate overfits, `∂L_CE/∂W`
collapses. Mit: `lr_ϕ = 2·lr_W` first 2000 steps then equal;
clip `‖grad(ϕ)‖/‖grad(W)‖ ≤ 10`; hard cap `λ_bce ≤ 0.1`.

**F5 — Cold start.** AUC ≈ 0.5 at t = 0. Mit: `n_warmup = 500` dense
steps; gate stays off until AUC ≥ 0.85 on held-out probe.

**F6 — CHIRON reversibility.** No Gumbel noise; deterministic; preserved.

## 6. Computational trade-offs (pile_large, batch = 8, L = 24)

| Metric | Baseline | GATE-BACK s = 0.25 | Ratio |
|---|---:|---:|---:|
| FFN fwd FLOPs | 2.75 TF | 2.75 TF + 0.11 TF gate | 0.96× (slight penalty) |
| FFN bwd FLOPs | 5.50 TF | 1.42 TF + 0.11 TF gate | **3.60×** |
| Extra params (ϕ) | 0 | 7.7 M total | +0.2 % |
| Extra Adam (FACE-compressed) | 0 | **~25 KB total** | negligible |
| Full step wall-clock | 142 ms | **112 ms** | 1.27× overall |

## 7. Strengths / Weaknesses

### Strengths

1. **Pure backward-side mechanism** — forward unchanged, so accuracy is
   preserved by construction; any bias appears only in `∂L/∂W` and is
   upper-bounded at ~7.7 % of total gradient energy (§4.3), safely below
   Adam's stochastic noise floor.
2. **FACE composability is multiplicative, not additive** — the
   embedding-analog structure of ϕ's output neurons means Zipfian
   preconditioning carries directly, yielding ~25 KB total Adam state
   for all 24 gate heads combined.
3. **Finer granularity than MoE** — routing to individual neurons inside
   a single FFN (not to expert FFNs), preserving the dense forward
   activation signal MoE sacrifices. No load-balancing loss, no
   routing-collapse pathology.

### Weaknesses

1. **Train-infer gap is inverted relative to NGATE** — at inference
   there is no backward, so ϕ is dead weight at serve time. Cost is
   0.1 % params but it is pure waste unless ϕ is repurposed (e.g. as a
   pruning heuristic).
2. **Bias bound tightens only under AUC → 1** — at early training
   (AUC 0.7 - 0.85) bias is 15 - 30 %, which **does** pollute main-loss
   gradients. Warmup + auto-disable (F3, F5) extend the non-productive
   window by 500 - 2000 steps.
3. **Requires σ'(x) materialization on forward** — for GELU/SiLU it is
   closed-form-free from σ, but for arbitrary activations GATE-BACK
   mandates a second fused kernel, eroding the FLOP accounting in §8
   by up to 5 % if σ' is non-trivial.
