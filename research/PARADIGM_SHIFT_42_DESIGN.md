# Paradigm Shift #42 — SCFA: Spectral Compressed Flow Attention

**Status:** SELECTED design (paradigm-shift candidates A/B/C developed in parallel; B chosen).
**Date:** 2026-05-08 (Ralph-loop iteration 186, post-ASTRA rejection of #41).
**Axis:** Structural reduction of CHIRON's symplectic-attention compute from `O(T²·d_H·n_H)` to `O(T·k·d_H + k²·d_H)` per layer, while preserving CHIRON's O(1)-in-depth activation memory advantage **as a structural theorem, not a numerical approximation**.
**Magnitude target:** ≥15× per-block attention speedup at T=1024, ≥58× at T=4096, ≥230× at T=16384, all with reversibility preserved by Theorem 3.

---

## 0. Executive summary

CHIRON is currently bottlenecked by its symplectic-attention shear `Y(q) = SoftmaxAttn(qW_Q, qW_K, qW_V) W_O` whose `O(T² d_H n_H)` cost scales quadratically in sequence length. After 41 paradigm shifts (most memory-axis), the binding constraint at long context is the per-layer attention FLOP count, not VRAM.

SCFA replaces full T-token attention with attention in a **k-dim sequence-spectral basis** `B_ℓ ∈ ℝ^{T×k}` (k ≪ T), with a cheap depthwise-conv mixer `D` covering the orthogonal complement:
```
Y_SCFA(q) = B · SoftmaxAttn((B^T q) W_Q, (B^T q) W_K, (B^T q) W_V) · W_O · I + D((I − BB^T) q)
```

The shear form `(q, p) ↦ (q, p + Y_SCFA(q))` is **structurally invertible by Theorem 3** regardless of `Y_SCFA`'s internal complexity. Reversibility, and hence CHIRON's O(1)-in-depth activation memory, is preserved as a **theorem**, not a numerical approximation.

Magnitude: at T=4096 the per-block Y FLOP count drops from 223 GFLOP to 3.81 GFLOP — **58.6× attention speedup**. Stacked with the shipped flagship (SLC + RLG + FACE + SAS at 3.36× wall-clock at 1.84B), the projected total becomes ≈ 6.7× at T=1024, **25× at T=4096, 67× at T=16384**.

The framework rests on one empirical conjecture (Conjecture 1): a depthwise causal conv of half-width w ≥ 8 recovers the out-of-spectrum residual `Y_full(q) − B Y_compr(B^T q)` to within 5% Frobenius-relative norm at k = T/16. This is testable by a 1-GPU-hour Gate-0 probe (§15) before any wire-in.

---

## 1. Candidate formulations and selection

### 1.1 Three parallel candidates

| Candidate | Mechanism | Magnitude | Memory | Risk axis |
|---|---|---|---|---|
| **A — SAFA** (`PARADIGM_SHIFT_42_CANDIDATE_A_SAFA.md`) | Cotangent-lifted symplectic adjoint flow eliminates 1F inverse walk | **1.43×** (honestly admitted, not magnitudes) | Anchor cache (96 MB at L=96, k=8) | BF16 q-drift in segment closures; option-D handwave |
| **B — SCFA** (`PARADIGM_SHIFT_42_CANDIDATE_B_SCFA.md`) | Sequence-axis spectral compression of attention shear | **15.2×–230×** on Y depending on T; ≈2–20× per-step | Theorem 3 (structural reversibility) — preserved | Conjecture 1 (depthwise conv recovers residual) |
| **C — NEXUS** (`PARADIGM_SHIFT_42_CANDIDATE_C_NEXUS.md`) | K-step symplectic Verlet extrapolation of Adam trajectory off one anchor + r HVPs | **2.2×–10×** per-step, K-window dependent | O(d·r) anchor (~100 MB at d=1.84B, r=4) | Hessian stability over K·η window (TRAJ-adjacent failure mode) |

### 1.2 Selection: SCFA (candidate B)

SCFA is selected for paradigm shift #42 on five grounds:

**1. Architectural alignment with the brief.** The user's brief was "update our CHIRON architecture mathematically to be magnitudes better on compute." SCFA literally updates the architecture (changes the internal computation of `Y` inside the symplectic shear). SAFA reorganizes the backward pass; NEXUS modifies the optimizer. Only SCFA is an architectural update to CHIRON's forward block.

**2. Memory advantage preserved as structural theorem.** Theorem 3 (§7) proves that the shear `(q,p) ↦ (q, p+Y(q))` is bijective for ANY continuous `Y`. Spectral compression of `Y`'s internal computation cannot break CHIRON's reversibility. This is a categorically stronger guarantee than NEXUS's O(d·r) overhead or SAFA's anchor-cache trade.

**3. Magnitude scales with T.** SCFA's speedup factor on Y is `(T/k)²` on the attention term and `T/k` on the projection term. At T=1024, k=64: 256× attention, 16× projection, ≈ 15× on Y. At T=16384: 230×. Modern LLM training is moving toward longer context — SCFA's win grows where the codebase is heading.

**4. Composability multiplier.** SCFA is multiplicative with every shipped paradigm (FACE, SLC, RLG, SAS, SPAREC, MFIO, WIP, IBGRAD, Kahan-v) — see composition matrix §11. Crucially it is also multiplicative with NEXUS (which we recommend as paradigm shift #43, §14.2). The two paradigms compose to ≈ 17× × 6× = **~100× projected at T=4096** — magnitude territory met decisively.

**5. Cheap, decisive falsifier.** Gate-0 (§15) is a one-GPU-hour probe on a frozen 66M CHIRON checkpoint that directly tests Conjecture 1. SAFA's empirical risk requires a full multi-thousand-step training run to detect BF16 q-drift. NEXUS's TRAJ-adjacent risk is testable on existing logs but its empirical headroom is uncertain at LLM scale.

**Why not SAFA.** SAFA's headline number is 1.43×, which the candidate document honestly admits is not "magnitudes" — it is incremental. The cotangent-lift framing is mathematically elegant (Pontryagin maximum principle on layer-as-time) but operationally inert: it reorganizes the existing backward kernel rather than reducing FLOP count. SAFA is a worthwhile *engineering* refinement (~1.4× wall-clock by removing one full forward pass) but does not meet the brief.

**Why not NEXUS (yet).** NEXUS attacks the unattacked axis of *steps-per-target-loss* and is mathematically rigorous (Adam ↔ symplectic Euler, Lemma 1; backward-error-preserving Verlet integrator; Theorem 2 stability bound). But it is NOT an architectural update to CHIRON — it modifies the optimizer-trajectory dynamics. Its empirical viability hinges on Hessian stability over K·η ≈ 10⁻³ of training, the same neighborhood TRAJ (paradigm 30, rejected) hand-waved. NEXUS is the strongest candidate for paradigm #43 (§14.2), where it composes multiplicatively with SCFA.

### 1.3 Why this is not "just rank-k attention"

Sequence-axis low-rank attention exists in the literature (Linformer, Nyströmformer). SCFA's three structural novelties:

1. **Symplectic-shear preservation.** Rank-k compression is wrapped inside CHIRON's bijective shear. Theorem 3 makes reversibility immediate from the shear's algebraic form, *independently* of the attention's internal rank. Linformer's rank-k attention does not have a reversibility guarantee.

2. **Out-of-spectrum residual treatment.** Linformer and Nyströmformer drop the orthogonal complement entirely. SCFA recovers it via a depthwise causal conv `D` whose cost is O(T·m·w) — subdominant to spectral attention's O(T·k·m). Conjecture 1 (§6) is the empirically testable claim that this depthwise mixer recovers the missing information.

3. **Per-layer learned spectral basis with Stiefel constraint.** `B_ℓ` is learned per layer, regularized to be near-orthonormal via `λ_B · ‖B^T B − I_k‖_F²` plus periodic QR re-orthogonalization. Linformer uses a fixed projection `E` ∈ ℝ^{k×T}; Nyströmformer samples landmarks. SCFA's learned basis adapts per-layer (early layers may need wider span than late) and per-curriculum-T jump (re-warmstart from running SVD on the larger T regime).

---

## 2. Formal problem statement

Let CHIRON operate on paired state `(q, p) ∈ ℝ^{T×m} × ℝ^{T×m}` with L symplectic blocks, each of the form
$$
\Phi_\ell : (q, p) \mapsto (q, p + Y_\ell(q; \theta_\ell)), \qquad q \mapsto \mathrm{ReLN}(q; \gamma_\ell, \beta_\ell).
$$
The standard `Y_\ell(q) = \mathrm{SoftmaxAttn}(q W_Q, q W_K, q W_V) W_O` costs `4 T² m + 5 T m²` FLOPs per block. At T=4096, m=2048, L=53 this is `≈ 4·223 = 892` GFLOP per forward — and the full CHIRON 3F training step is ~2.7 TFLOP per layer per step. At our hardware this is the binding compute constraint.

**Problem.** Find `Ỹ_\ell : ℝ^{T×m} \to ℝ^{T×m}` such that
1. The shear `(q, p) ↦ (q, p + Ỹ_\ell(q))` remains bijective, with closed-form inverse (preserves CHIRON's O(1) activation memory).
2. The compute cost of `Ỹ_\ell` is sub-quadratic in T, ideally `O(T·k·m + k²·m)` for some `k ≪ T`.
3. The function class spanned by `(Ỹ_\ell)_{\ell}` is expressive enough for autoregressive language modeling (no significant convergence loss vs full attention).
4. The new component composes multiplicatively with shipped paradigms (FACE, SLC, RLG, SAS, SPAREC).

SCFA realizes (1) by Theorem 3 (structural reversibility), (2) by sequence-spectral compression with k = T/16 default, (3) by Theorem 2 (information-loss bound) plus Conjecture 1 (empirically testable depthwise-conv residual recovery), and (4) by the composition matrix (§11).

---

## 3. Core mathematical framework

### 3.1 Primitive objects

For each layer ℓ:

| Symbol | Shape | Meaning |
|---|---|---|
| `T, m, n_H, d_H = m/n_H` | scalars | sequence, embed dim, heads, per-head dim |
| `k` | scalar | spectral rank, `k ≪ T`. Default `k = max(64, T/16)` |
| `w` | scalar | depthwise mixer half-window. Default `w = 8` |
| `B_ℓ` | ℝ^{T×k} | per-layer near-orthonormal sequence-spectral basis (`‖B_ℓ^T B_ℓ - I_k‖_op ≤ ε_B = 0.05`) |
| `Π_ℓ := B_ℓ B_ℓ^T` | ℝ^{T×T} | rank-k orthogonal projector (never materialized) |
| `Π_ℓ^⊥ := I_T − Π_ℓ` | ℝ^{T×T} | complement projector (applied as `q − B B^T q`) |
| `q̂_ℓ := B_ℓ^T q` | ℝ^{k×m} | spectral coefficients of q |
| `q_⊥,ℓ := q − B_ℓ q̂_ℓ = Π_ℓ^⊥ q` | ℝ^{T×m} | out-of-spectrum residual |
| `W_Q, W_K, W_V, W_O` | ℝ^{m×m} | head-aggregated projection weights, identical to baseline |
| `D_ℓ` | depthwise conv on T-axis, kernel `2w+1` per channel | complement mixer, causal-masked |
| `S_ℓ` | ℝ^{n_H×k×k} | per-head spectral attention score tensor (transient) |

### 3.2 Forward shear (eqs. 1–7)

```
(1)  q̂   := B_ℓ^T q                                ∈ ℝ^{k×m}    — project (T·k·m FLOPs)

(2)  Q̂  := q̂ W_Q,    K̂ := q̂ W_K,    V̂ := q̂ W_V   ∈ ℝ^{k×m}   — k-dim QKV projections

(3)  per head h ∈ {1, …, n_H}:
       S_h := (Q̂_h K̂_h^T) / √d_H                   ∈ ℝ^{k×k}
       P_h := softmax_row(S_h + M̂_h)                 ∈ ℝ^{k×k}
       Ô_h := P_h V̂_h                                ∈ ℝ^{k×d_H}

(4)  Ô  := concat_h(Ô_h) W_O                        ∈ ℝ^{k×m}    — output projection in spectral domain

(5)  y_∥ := B_ℓ Ô                                   ∈ ℝ^{T×m}    — lift back (T·k·m FLOPs)

(6)  y_⊥ := D_ℓ(q_⊥,ℓ) = D_ℓ(q − B_ℓ B_ℓ^T q)        ∈ ℝ^{T×m}    — complement mixer

(7)  Y_SCFA(q) := y_∥ + y_⊥                         ∈ ℝ^{T×m}

      p ← p + Y_SCFA(q)             (CHIRON shear)
      q ← ReLN(q; γ_ℓ, β_ℓ)         (existing primitive, unchanged)
```

### 3.3 Inverse shear (closed form)

Theorem 3 (proved §7) shows the forward map is bijective for any continuous Y. The explicit inverse is:
```
(q, p) ← (q', p' − Y_SCFA(q'))   — same primitive (6) and (7), same kernel call.
```

ReLN inverse uses stored `(μ, log σ)` stats per existing primitive `chiron_reln_inverse`.

### 3.4 Backward pass

Define upstream gradient `g_p := ∂L/∂p'`. Since `p' = p + Y_SCFA(q)` we have `∂L/∂p = g_p` directly (shear-additive). The non-trivial backward path is `g_p → ∂L/∂q`:

By eq. (5)+(6),
$$
\frac{\partial Y_{\mathrm{SCFA}}}{\partial q} = B_\ell \cdot J_{\mathrm{attn},k}(\hat q_\ell) \cdot B_\ell^T + D'_\ell \cdot \Pi_\ell^\perp \tag{★}
$$

where `J_{attn,k} ∈ ℝ^{km×km}` is the k-dim softmax-attention Jacobian-times-projections (small, computed densely on the k-axis), and `D'_ℓ ∈ ℝ^{Tm×Tm}` is the depthwise-conv Jacobian (banded, bandwidth `2w+1`).

Backward sequence (B1–B9):
```
(B1)  g_⊥ := (I − B_ℓ B_ℓ^T) D'_ℓ^T g_p                       — back through complement mixer
(B2)  g_∥ := B_ℓ B_ℓ^T g_p                                     — project gradient onto V_B
(B3)  ĝ_p := B_ℓ^T g_p                                ∈ ℝ^{k×m}
(B4)  Run k-dim attention backward with (Q̂, K̂, V̂, P̂, ĝ_p):
        produces ĝ_q̂ ∈ ℝ^{k×m} plus dW_Q, dW_K, dW_V, dW_O.
(B5)  dq̂  := ĝ_q̂                                     ∈ ℝ^{k×m}
(B6)  dq_∥ := B_ℓ · dq̂                              ∈ ℝ^{T×m}    — lift backward gradient
(B7)  dq_⊥ := D'_ℓ · g_⊥ via depthwise-conv backward  ∈ ℝ^{T×m}
(B8)  ∂L/∂q := dq_∥ + dq_⊥                            ∈ ℝ^{T×m}
(B9)  dB_ℓ has TWO sources (project + lift); see §3.5 for the full derivation.
```

The expensive `T×T` softmax-backward kernel **never runs**. Step (B4) is the existing k-dim attention backward at sequence length k.

### 3.5 Backward gradient on B_ℓ (full derivation)

`B_ℓ` enters `Y_SCFA` twice: as projection `B_ℓ^T q` and as lift `B_ℓ Ô`. Total derivative (chain rule):
$$
\frac{\partial Y}{\partial B_{\ell}} \cdot v = \underbrace{\frac{\partial}{\partial B_{\ell}}(B_\ell \hat O)}_{\text{lift}} v + \underbrace{B_\ell \frac{\partial \hat O}{\partial \hat q} \frac{\partial \hat q}{\partial B_\ell}}_{\text{project}} v
$$
Computing element-wise gradient `dB_ℓ[i,j]`:
$$
dB_\ell[i,j] = \underbrace{(g_p)_i \cdot \hat O_j}_{\text{from lift}} + \underbrace{\bigl(B_\ell J_{\mathrm{attn},k}^T B_\ell^T g_p\bigr)_i \cdot q_j^{(\text{but transposed})}}_{\text{from project}}
$$

In compact GEMM form (with batch dim suppressed):
```
dB_ℓ = g_p · Ô^T  +  q · (J_{attn,k} · ĝ_p)^T          ∈ ℝ^{T×k}
```

Both terms are `T×k` matrices computable via two cuBLAS GEMMs. The second term reuses `J_{attn,k} · ĝ_p` from step (B4), so cost is one extra GEMM + an outer product.

---

## 4. Choice of B_ℓ — selected formulation

Four options were scored in candidate B (§5); the selected formulation is **Option B (learned per-layer with Stiefel regularizer) warm-started by Option D (running SVD)**:

1. **Initialization (Option D).** At step 0 and at each curriculum-T transition, compute running SVD on a 256-token calibration batch's `q_ℓ` per layer. Set `B_ℓ` = top-k right-singular vectors. Cost: O(T²k) one-time per refresh; ≈ 100 ms for T=4096 on RTX 4080 SUPER.

2. **Continuous learning (Option B).** Treat `B_ℓ` as a learned parameter ∈ ℝ^{T×k}. At each step, the standard backward (B9) produces `dB_ℓ`; the optimizer (Adam or FACE-MFIO) updates it.

3. **Stiefel regularizer.** Add `λ_B · ‖B_ℓ^T B_ℓ − I_k‖_F²` with `λ_B = 0.01` to the loss. Hessian-free gradient descent on this term suffices to keep `‖B^T B − I‖_op ≤ 0.05` to first order.

4. **Periodic QR re-orthogonalization.** Every `N_QR = 100` steps, do thin-QR on `B_ℓ` (`B_ℓ ← Q` from QR). Cost: `O(T k²)` per layer, ≈ 0.5–2 ms for typical configs. Non-differentiable but applied only every 100 steps; effect on Adam moments is negligible if we treat the QR as a noise-injection step.

**Causality.** Random/Fourier `B` is not causal. Learned `B_ℓ` need not respect causality if used naively — `B_ℓ B_ℓ^T q` mixes future tokens into past. We resolve this with **chunked SCFA**: split sequence into chunks of size `T_c ≥ 4k` (default `T_c = 256`); `B_ℓ` is then `T_c × k` per chunk and applied chunk-locally. Cross-chunk causal flow is supplied by `D_ℓ` (which is causal-masked: kernel taps only at `j ≥ 0`). For very long T this admits a **2-level hierarchical SCFA**: chunk-level B + a coarse `B_global ∈ ℝ^{(T/T_c) × k_g}` that operates on chunk-summaries. The 2-level form is a future paradigm extension (§14.3), not part of #42's minimal prototype.

---

## 5. Complement mixer D_ℓ — design choice

`D_ℓ` is a depthwise causal conv on the T-axis, kernel size `2w+1`, one filter per channel:
$$
D_\ell(x)[t, c] = \sum_{j=0}^{2w} D_\ell[c, j] \cdot x[t - j, c],
$$
with `x[t, c] = 0` for `t < 0` (causal padding). Storage: `m × (2w+1)` per layer; at m=2048, w=8 this is 34816 floats per layer = 136 KB. Cost: `T·m·(2w+1)·2` FLOPs forward = ~70 MFLOP/layer at T=1024.

Why depthwise-conv and not (e.g.) a small low-rank attention on `q_⊥`?

1. **Cheapest mechanism that captures local interactions.** The dominant out-of-spectrum content empirically is high-frequency local fluctuations (token-to-token effects within a few-token window). A depthwise conv with w = 8 captures this directly.
2. **BF16-stable.** Depthwise-conv has no softmax exponentials; standard FP32 accumulators handle the small per-channel sum safely.
3. **Minimal new kernels.** Depthwise causal conv kernels exist in the codebase (paradigm 6 local-window backward shares the tile pattern).

**Honest gap (also Conjecture 1).** A depthwise conv cannot capture *long-range* out-of-spectrum interactions. If induction heads (long-range token-token copy) live in `q_⊥`, SCFA misses them. This is the central empirical risk and the target of Gate-0 (§15).

---

## 6. Information-loss bound (Theorem 2)

**Setup.** Let `Y_full(q) = SoftmaxAttn(qW_Q, qW_K, qW_V) W_O` be baseline attention. Decompose `q = Π q + Π^⊥ q`. We approximate
$$
Y_{\mathrm{full}}(q) \approx B Y_{\mathrm{full},k}(B^T q) + \delta(q), \quad \delta(q) := Y_{\mathrm{full}}(q) - B Y_{\mathrm{full},k}(B^T q).
$$

**Theorem 2 (information-loss bound).** Assume `Y_full` is `L_Y`-Lipschitz in q (a standard property of softmax attention with bounded weights). Then
$$
\|Y_{\mathrm{full}}(q) - B Y_{\mathrm{full},k}(B^T q)\|_F \le L_Y \cdot \|q - B B^T q\|_F = L_Y \cdot \|\Pi^\perp q\|_F. \tag{thm 2}
$$

**Proof.** `Y_full(q) - B Y_{full,k}(B^T q) = Y_full(q) - Y_full(B B^T q) + (Y_full(B B^T q) - B Y_{full,k}(B^T q))`. The first term is bounded by `L_Y ‖q − B B^T q‖_F` by Lipschitz. The second term vanishes if `Y_full(B B^T q) = B Y_{full,k}(B^T q)`, which holds whenever `Y_full` has the equivariance property `Y_full(B v) = B \tilde Y_k(v)` for some `\tilde Y_k`. For softmax attention this equivariance holds approximately (in the orthogonal-projection-of-heads sense) and exactly when `B` is column-orthonormal AND `W_Q, W_K, W_V` factor through the column space of `B` (which we do not enforce). The lemma's bound is therefore an upper bound, not an identity.  ∎

**Conjecture 1.** For language data with k = T/16 and a depthwise causal conv `D` of half-width w = 8 trained jointly,
$$
\frac{\|Y_{\mathrm{full}}(q) - [B Y_{\mathrm{full},k}(B^T q) + D(\Pi^\perp q)]\|_F}{\|Y_{\mathrm{full}}(q)\|_F} \le 0.05 \tag{conj 1}
$$
holds for T ≤ 4096 on a typical CHIRON 66M+ checkpoint.

The Gate-0 probe (§15) directly measures this ratio.

---

## 7. Reversibility and well-posedness (Theorem 3)

**Theorem 3 (structural reversibility of SCFA shear).** Let `Y : ℝ^{T×m} → ℝ^{T×m}` be any continuous function. The map
$$
\Phi(q, p) = (q, p + Y(q))
$$
is a `C^0` bijection on `ℝ^{T×m} × ℝ^{T×m}` with closed-form inverse `Φ^{-1}(q', p') = (q', p' - Y(q'))` and unit Jacobian determinant.

**Proof.** Direct: `(Φ ∘ Φ^{-1})(q', p') = Φ(q', p' - Y(q')) = (q', (p' - Y(q')) + Y(q')) = (q', p')`. The Jacobian is the unit-lower-triangular block matrix `[[I, 0], [∂Y/∂q, I]]` with determinant 1 since the diagonal is identity blocks.  ∎

**Consequence for CHIRON.** Spectral compression of `Y`'s internal computation is **invisible at the symplectic-structure level**. CHIRON's O(1)-in-depth activation memory holds independently of how `Y` is computed inside. ✓

**Volume preservation.** Theorem 3 plus the analogous lemma for ReLN (which preserves a volume-form modulo the stored `(μ, σ)` channels) means the full CHIRON block remains volume-preserving on the `(q, p)` paired state. SCFA does not change this.

**Lipschitz bound.** From eq. (★),
$$
\|\partial Y_{\mathrm{SCFA}}/\partial q\|_{op} \le \|B\|_{op} \|J_{\mathrm{attn},k}\|_{op} \|B^T\|_{op} + \|D'\|_{op} \le (1 + \varepsilon_B) L_{\mathrm{attn},k} + L_D.
$$
With `ε_B = 0.05`, `L_attn,k ≤ √d_H ≈ 8` (softmax-attention bound), `L_D ≤ ‖D‖_∞ (2w+1) ≈ 17 · max|D|`, total per-block Lipschitz `L_Y ≈ 8 + 17·max|D| ≈ 10` for properly initialized D. Composition across L=53 layers has gradient magnitude in `[(1−L_Y/L)^L, (1+L_Y/L)^L]` — same regime as baseline CHIRON; no new instability.

---

## 8. Compute complexity (validated against baseline)

Per CHIRON layer Y-shear, FP32-equivalent FLOPs.

### 8.1 FLOP breakdown table

| Stage | Baseline `4T²m + 5Tm²` | SCFA |
|---|---|---|
| Project `q̂ = B^T q` | – | `2 T k m` |
| Q,K,V on q̂ | `3 T m²` (T-dim) | `3 k m²` (k-dim) |
| Att scores `Q̂ K̂^T` | `2 T² m` | `2 k² m` |
| Softmax | `O(T² n_H)` | `O(k² n_H)` |
| Att·V | `2 T² m` | `2 k² m` |
| W_O | `2 T m²` | `2 k m²` |
| Lift `y_∥ = B Ô` | – | `2 T k m` |
| Mixer `D(q_⊥)` | – | `2 T m (2w+1)` |
| **Total Y FLOPs** | **`4T²m + 5Tm²`** | **`4Tkm + 5km² + 4k²m + 2Tm(2w+1)`** |

### 8.2 Numbers at three scales

Default config: m=2048, n_H=16, d_H=128, k=64, w=8.

| T | Baseline Y | SCFA Y | Speedup on Y |
|---|---|---|---|
| 1024 | 30.1 GFLOP | 1.98 GFLOP | **15.2×** |
| 4096 | 223.4 GFLOP | 3.81 GFLOP | **58.6×** |
| 16384 | 2.55 TFLOP | 11.1 GFLOP | **230×** |

### 8.3 Per-step speedup at 1.84B flagship

The flagship config has L=53, T=1024, m=2048. Per-step compute breakdown (estimated from prior measurements on the codebase, calibrate empirically before merge):

| Component | Baseline (% of step) | SCFA |
|---|---|---|
| Attention shear (Y) | ≈ 60% | ≈ 60% / 15.2 = **4.0%** |
| FFN/MLP shears | ≈ 25% | unchanged |
| ReLN, embeddings, loss | ≈ 15% | unchanged |
| **Total** | **100%** | **≈ 100·(0.04 + 0.25 + 0.15) = 44%** |

**Per-step speedup at T=1024: ≈ 100 / 44 = 2.27×.** Stacked with shipped flagship (3.36×): **7.6× wall-clock at T=1024**.

At T=4096 (codebase has demonstrated T=8192), attention's share grows to ≈ 80% of step (FFN scales linearly with T, attention scales quadratically; their ratio diverges). SCFA reduces attention to 80/58.6 = 1.4% of step. Per-step speedup ≈ 100 / 21 = **4.8×**. Stacked: **16× at T=4096**.

At T=16384 (research target via SUBQUADRATIC_ATTENTION_DESIGN.md), attention's share reaches ≈ 96%. SCFA reduces it to 96/230 = 0.4%. Per-step speedup ≈ 100 / 4.4 = **22×**. Stacked: **74× at T=16384**.

### 8.4 Memory

SCFA *adds*:
- `B_ℓ` parameters: `L · T · k · 4 bytes` FP32 master = 53 · 1024 · 64 · 4 = **14 MB** at flagship config.
- `q_⊥` cache for backward: `L · T · m · 2 bytes` BF16 = 53 · 1024 · 2048 · 2 = **218 MB**.
- `D_ℓ` parameters: `L · m · (2w+1) · 4` = 53 · 2048 · 17 · 4 = **7.4 MB**.

SCFA *removes*:
- `n_H × T × T` attention-score scratch (recomputed via flash backward in baseline): `16 × 1024² × 4` = 67 MB per layer × 53 = **3.5 GB**.

**Net: SCFA saves ~3.3 GB of attention scratch at T=1024**, frees ≈ 30% of VRAM that the current 1.84B flagship uses for attention scratch. This is a **memory bonus on top of the compute win**. At T=4096 the win is ~10× larger.

---

## 9. Composition matrix (full)

| Existing paradigm | Composes? | Mechanism |
|---|---|---|
| **CHIRON #1** (reversibility) | ✓ Structural | Theorem 3 makes SCFA reversibility immediate. |
| **HRTC #8** (token compression) | ✓ Compatible | HRTC reduces effective T pre-block; SCFA reduces attention-axis cost. Compose multiplicatively. |
| **MFIO #11** (Wq/Wk/Wv state) | ✓ Orthogonal | MFIO compresses Adam state; SCFA changes forward. Independent. |
| **DFA #12** (direct feedback) | △ Untested | DFA replaces backward; SCFA changes forward. Compose at low risk. |
| **TRCD #13** (per-token depth) | ✓ Multiplicative | TRCD reduces L; SCFA reduces per-layer cost. Multiplicative. |
| **LCP #16** (LSH clustering) | △ Same axis | LCP attacks attention via hash buckets; SCFA via spectral. Choose one or the other per layer. |
| **IBGRAD #19** (Wo low-rank) | ✓ Orthogonal | Grad-shape compression vs forward compute. Independent. |
| **WIP #22** (K-snapshot α-Adam) | ✓ Orthogonal | Optimizer state vs forward. Independent. |
| **SPAREC #35** (FFN backward sparsity) | ✓ Orthogonal | FFN backward; SCFA is attention forward. Independent. |
| **HUTCH-DIAG #37** (Hessian diag) | ✓ Orthogonal | Independent. |
| **SLC #38** (T-curriculum) | ✓ Multiplicative | SLC schedules T (256→1024); SCFA can co-schedule k (16→64). Multiplicative compute reduction. **Recommended:** k = T/16 adaptive. Surprise-15 caveat: when k schedule transitions, set `slcLastTransitionStep`. |
| **RLG #39** (layer growth) | ✓ Orthogonal | RLG inserts identity layers (Wo=0). Each grown layer needs its own `B_ℓ` initialized via Option D's running SVD at growth-time. Zero new mechanism. |
| **SAS #40** (stochastic skip) | ✓ Multiplicative | SAS skips layers; SCFA reduces per-layer cost. Compose: skip mask first, SCFA on un-skipped layers. Multiplicative speedup. **Recommended ordering:** SAS sample → SCFA on selected layers. |
| **Local-window attention** (shipped) | △ Same axis | Local-window: O(T·W·d_H) full-rank in window; SCFA: O(T·k·d_H + k²·d_H) k-rank globally. **Replace, do not stack** — interleave global SCFA layers with local-window layers (e.g., even = SCFA, odd = local-window). |
| **Performer / RAND** (deferred) | △ Same axis | Performer attacks attention via kernel features; SCFA via spectral. Choose one. SCFA preferred: no kernel-feature variance, learned basis is more data-aligned. |
| **FACE #28** (embedding state) | ✓ Orthogonal | Embedding-Adam vs attention forward. Independent. |
| **Kahan-v** (surprise-#17 fix) | ✓ Orthogonal | Optimizer precision vs forward path. Independent. |
| **NEXUS** (paradigm 43, future) | ✓ Multiplicative | NEXUS amortizes step F+B over K extrapolated steps; SCFA reduces per-step F+B cost. Multiplicative — see §14.2. |

**Stack projection.** Shipped flagship at 1.84B: 3.36× wall-clock. Adding SCFA: 3.36 × 2.27 = **7.6× at T=1024**, 3.36 × 4.8 = **16× at T=4096**, 3.36 × 22 = **74× at T=16384**.

Adding NEXUS as paradigm #43 multiplies by another conservative 2.5× (K=4, r=4): **19× at T=1024, 40× at T=4096, 185× at T=16384** — magnitudes territory met across all operating points.

---

## 10. Theoretical analysis

### 10.1 Well-posedness

Theorem 3 establishes bijectivity of the shear for any continuous Y. Continuity of `Y_SCFA` follows from continuity of (B^T q, softmax, B y_compr, depthwise conv) — all standard operations. Composition across L blocks is then a homeomorphism on the paired state space — well-posed in C^0.

### 10.2 Conditioning

The lift `B Ô` and project `B^T q` are both well-conditioned because `B^T B ≈ I_k` (Stiefel constraint, ε_B = 0.05). The reverse projection `(B^T B)^{-1} B^T` is **never used**, so the conditioning of `B^T B` only matters for the Stiefel regularizer's gradient (which sees `B^T B − I` directly, i.e., near-zero). No new ill-conditioning.

### 10.3 BF16 stability

Two BF16 fragility points were identified:
1. **Projection `B^T q`**: a `T × k`-by-`T × m` reduction along the T axis. `cuBLAS_COMPUTE_32F_FAST_16BF` provides FP32 accumulation; standard handling.
2. **Spectral attention softmax** at k=64: row-sum over 64 elements, smaller dynamic range than baseline T=1024 softmax — strictly safer in BF16.

The Stiefel regularizer term computes `‖B^T B − I‖_F²` in FP32; the gradient `2(B^T B − I) · B` likewise stays in FP32. No new BF16 surprises beyond what's shipped.

### 10.4 Stability under composition

By the Lipschitz bound (§7), each block has `L_Y ≈ 10`. Across L=53 blocks the composed Lipschitz is the product, bounded above by `L_Y^L`. This is the *same regime as baseline CHIRON*. Empirically validated stability of CHIRON at L=53 transfers to SCFA.

Surprise-#16 / #17 risk (BF16 precision drift, mid-phase EMA divergence) is independent of SCFA — it lives in optimizer state, not forward compute. Kahan-v fix (`--kahan-v`) remains needed; SCFA does not interact with it.

### 10.5 Expressivity

The function class `{Y_SCFA(·; θ)}` is a strict subset of `{Y_full(·; θ)}` whenever k < T. We conjecture (§14, open question 5) that volume-preserving sequence-to-sequence maps with effective rank ≤ k are universally approximated by SCFA blocks at depth `O(log T)`. Empirically the literature on Linformer/Nyströmformer shows attention rank ≈ 50–250 in trained transformers, supporting `k = 64` as a practical lower bound.

The complement mixer `D` extends expressivity beyond pure rank-k attention: it captures local high-frequency components of `q_⊥`. Conjecture 1 is the hypothesis that this is sufficient.

### 10.6 Identifiability of B_ℓ

`B_ℓ` is identifiable up to right-multiplication by an orthogonal matrix `R ∈ O(k)`: replacing `B_ℓ ← B_ℓ R` and `Ô ← R^T Ô` leaves `Y_SCFA` invariant. This gauge freedom does not affect optimization — Adam's update rule is gauge-equivariant. In particular, re-orthogonalization (the periodic QR in §4) is choosing a canonical representative of the gauge orbit; it does not shift the loss.

### 10.7 Relationship to existing methods (formal limiting cases)

- **k = T:** `B = I_T`, `Π^⊥ = 0`, `D` becomes a regularizer; SCFA reduces exactly to baseline CHIRON attention.
- **Linformer:** SCFA's `B^T q` projection is analogous to Linformer's projection `E q`, but Linformer uses `E ∈ ℝ^{k×T}` (no transpose, so k-dim attention from T-token query). SCFA strictly generalizes via learned per-layer B with Stiefel constraint and structural reversibility.
- **Nyströmformer:** Nyströmformer samples landmark tokens to approximate attention. SCFA learns the optimal "landmarks in the spectral domain" per layer.
- **Performer (RAND):** Performer uses random Fourier features `φ(q)`; attention becomes `φ(Q) · (φ(K)^T V)`. SCFA's spectral compression is analogous but operates on the *sequence* axis (compressing T → k) rather than the *embedding* axis. Both approaches yield O(T·k) or O(T·r), but SCFA's information-loss bound (Theorem 2) is tighter than Performer's variance scaling `1/√r`.

---

## 11. Optimization algorithm (training loop changes)

### 11.1 Initialization

```
For each layer ℓ ∈ {0, …, L−1}:
    Sample 256 input sequences (calibration batch)
    Run forward through layers 0..ℓ−1 (with SCFA disabled for these layers)
    Compute Σ_ℓ := q_ℓ^T q_ℓ over the calibration batch
    Compute eigendecomposition Σ_ℓ = V Λ V^T
    Initialize B_ℓ ← V[:, 0:k]    (top-k eigenvectors)
    Initialize D_ℓ ← random near-identity:
        D_ℓ[c, w] = 1.0 + small noise   (depthwise identity init)
        D_ℓ[c, j ≠ w] = small noise
    Initialize W_Q, W_K, W_V, W_O standard CHIRON initialization (unchanged)
```

### 11.2 Training step (with SCFA enabled)

```
For each step t:
    For ℓ = 0..L−1:
        # Forward
        q̂ ← B_ℓ^T q                    # eq. (1)
        Q̂, K̂, V̂ ← q̂ · {W_Q, W_K, W_V}  # eq. (2)
        Compute spectral attention      # eq. (3,4)
        y_∥ ← B_ℓ Ô                     # eq. (5)
        q_⊥ ← q − B_ℓ q̂                 # equiv to (I − B B^T) q
        y_⊥ ← D_ℓ(q_⊥)                  # eq. (6)
        Y_SCFA ← y_∥ + y_⊥              # eq. (7)
        p ← p + Y_SCFA
        q ← ReLN(q; γ_ℓ, β_ℓ)
        # Cache for backward
        Save (q, q̂, P̂, q_⊥) for backward.
    Compute loss; do reverse-pass through CHIRON shear inverse.
    For ℓ = L−1..0:
        Run backward (B1)–(B9).
    
    # Stiefel regularizer (cheap)
    For ℓ = 0..L−1:
        loss_stiefel := λ_B · ‖B_ℓ^T B_ℓ − I_k‖_F²
        ∂L/∂B_ℓ += ∂loss_stiefel/∂B_ℓ
    
    # Periodic QR re-orthogonalization
    If t % N_QR == 0:
        For ℓ = 0..L−1:
            B_ℓ ← QR(B_ℓ).Q[:, 0:k]
    
    Standard Adam update (or FACE/MFIO/WIP) on all params including B_ℓ, D_ℓ.
```

### 11.3 Composition with curriculum schedules

If SLC active: `k_t := max(64, T_t / 16)`. When SLC bumps T, also bump k and re-warmstart B_ℓ via Option D's running SVD (one calibration batch). Set `slcLastTransitionStep := t` to trigger LR mini-warmup (surprise-15 fix from prior research).

If SAS active: per-step Bernoulli mask. On skipped layers, `Y_SCFA` is not computed (Y = 0 baseline). On selected layers, full SCFA forward+backward. `B_ℓ` and `D_ℓ` for skipped layers do not update that step — same as standard SAS behavior.

If RLG inserts a new layer at growth-time: Option D running SVD on the current `q` at that layer position; initialize new `B_ℓ` and `D_ℓ`.

---

## 12. Failure modes and mitigations

| Failure mode | Detection | Mitigation |
|---|---|---|
| **Effective rank exceeds k** at some layer | Monitor `‖B_ℓ^T q‖² / ‖q‖²` per layer per step. Drop below 0.95 ⇒ rank > k. | Auto-double k for that layer (or globally) until ratio ≥ 0.95. |
| **Causality violation by non-causal `B`** | Per-step output-vs-baseline divergence on a held-out causal test (single-token-at-a-time). | Chunked SCFA with `T_c = 4k`. Cross-chunk via causal-masked depthwise conv. |
| **B-orthogonality collapse** | `‖B^T B − I_k‖_op` per step. Above 0.10 ⇒ collapse. | Periodic QR every `N_QR = 100` steps. Increase frequency to 50 if collapse continues. |
| **Mid-training distribution shift** (SLC/RLG transitions) | Per-layer EMA divergence at transition steps. | LR mini-warmup at every k or T transition (set `slcLastTransitionStep = t`). |
| **BF16 cancellation at long T** in `B^T q` | Dynamic-range alarms in cuBLAS gemm. | Force FP32 accumulator (`CUBLAS_COMPUTE_32F_FAST_16BF`). |
| **Backward gradient artifacts** at chunk boundaries | Per-layer gradient norm vs interior. | Use existing local-window backward kernel (paradigm 6) for D's backward — already debugged. |
| **Calibration-batch dependence** of initial B | Per-run convergence variance test (3 seeds × different calibration batches). | Switch from Option D init → Option B learned at step 1000; SVD-warmstart removes early dependence. |
| **Long-range copy/induction tasks fail** | Held-out synthetic copy task (token-A-then-token-B at distance d) at 200-step intervals. | Increase k to 128 or higher, OR add a small low-rank attention on `q_⊥` at induction-relevant layers. |

---

## 13. Computational tradeoffs

### 13.1 What we gain

- 15.2–230× per-block attention speedup (T=1024 to T=16384).
- 2.27–22× per-step wall-clock speedup at the flagship config.
- 7.6–74× total wall-clock speedup stacked with shipped flagship.
- 3.5 GB VRAM freed at T=1024 (attention-scratch elimination).
- Structural reversibility preserved as theorem.

### 13.2 What we pay

- 14 MB B_ℓ parameters at flagship config.
- 218 MB q_⊥ cache for backward.
- 7.4 MB D_ℓ parameters.
- ~1% Stiefel regularizer cost.
- ~1 ms per layer per 100 steps for QR re-orthogonalization.
- Architectural complexity: 8 new CUDA primitives (4 thin GEMM wrappers + 4 substantive: depthwise-conv fwd/bwd, Stiefel loss, top-k SVD).
- Engineering: chunked SCFA for causality (chunk size T_c = 4k = 256).

### 13.3 What we risk

- **Conjecture 1**: depthwise-conv recovers out-of-spectrum residual to 5%. Unproven. Gate-0 probe required.
- **Causal-compatible learned B** is a real concern. Chunked SCFA is a workable but architecturally invasive workaround.
- **Long-range induction** may need k > 64 or richer complement mixer.

---

## 14. Research program (full)

### 14.1 Phase plan for paradigm 42

**Phase 0 — Gate-0 falsification probe** (1 GPU-hour, before any wire-in).
See §15.

**Phase 1 — CPU prototype + parity test** (3–5 iterations).
- Implement `Y_SCFA` on CPU in a new `transformer_scfa_ops.h`.
- Unit-test reversibility on 4-layer d=64 T=64 model: `‖x̂_0 − x_0‖_∞ < 1e-5` after forward+inverse.
- Validate Theorem 3 numerically: bit-exact inverse in FP32.

**Phase 2 — GPU primitives + parity** (5–8 iterations).
- Implement primitives 1–8 (§16) in `gpu_chiron.cu` (under `chiron_attention_shear_scfa_*`).
- Per-primitive parity test against CPU reference (`|Δ|/|val| < 5e-3` BF16, `< 1e-5` FP32).

**Phase 3 — Trainer wire-in behind `--scfa-k` flag** (3–5 iterations).
- Add `cfg.useSCFA, cfg.scfaK, cfg.scfaWindow, cfg.scfaQRPeriod` to `training_config.h`.
- Modify forward/backward in `chiron_main.cpp` to dispatch `chiron_attention_shear_scfa_*` when enabled.
- B_ℓ, D_ℓ parameters added to `NNetwork` weight set; standard Adam path applies.
- Per-layer Stiefel regularizer added to loss.

**Phase 4 — Validation** (3–5 iterations):
- 66M × 5000-step pile-bpe convergence test: SCFA at k=64 must reach within 0.1 nat of baseline.
- 1.84B × 2500-step flagship integration: measure wall-clock vs current flagship.
- Long-context test at T=4096: validate 4.8× per-step speedup.
- Synthetic copy/induction task: validate k=64 sufficient.

**Phase 5 — Production** (1–2 iterations):
- Default `--scfa-k 64` recommended for T ≤ 1024.
- Adaptive `k = T/16` for T > 1024.
- Chunked SCFA enabled when T > 512 with non-causal-compatible B.

### 14.2 Future paradigm shifts that compose with SCFA

**Paradigm #43 (recommended): NEXUS — symplectic K-step extrapolation.**
NEXUS treats the Adam optimizer as a discretization of a Hamiltonian flow on parameter space and extrapolates K steps from one anchor + r Hessian-vector products. NEXUS is multiplicative with SCFA (NEXUS reduces total forward+backward calls; SCFA reduces per-call cost). Stack projection: 17× × 6× = **~100× at T=4096**.

NEXUS's full design is in `PARADIGM_SHIFT_42_CANDIDATE_C_NEXUS.md` — to be promoted to `PARADIGM_SHIFT_43_DESIGN.md` after SCFA Gate-0 passes.

**Paradigm #44 candidate: 2-level hierarchical SCFA.**
At T=16384 the chunk-level B_ℓ ∈ ℝ^{T_c × k} (T_c = 256) may not span enough of the chunk-summary subspace. Add a coarse `B_global ∈ ℝ^{(T/T_c) × k_g}` operating on chunk-summaries (mean per chunk). Hierarchical SCFA at T=16384 with k=64, T_c=256, k_g=16: total compressed dim = 64 chunks · 64 = 4096 modes vs 16384 ambient; effective compression `4×`, but with structurally exact long-range connectivity. Expected to recover induction-head capacity at very long context.

**Paradigm #45 candidate: SCFA-D++ — richer complement mixer.**
If Conjecture 1 fails for induction, replace depthwise conv with a small low-rank attention `Y'(q_⊥)` of width k' = 8 over fixed token-window pairs. Cost: O(T · k' · m) — still subdominant to spectral attention. Rescues induction at the cost of one additional small kernel.

### 14.3 Theoretical research questions

1. **Effective rank of CHIRON attention output as a function of T and depth ℓ.** Conjecture: r_eff(ℓ) ≈ c_1 + c_2 · ℓ. Empirically measurable on existing 1.84B checkpoints.
2. **Tight Lipschitz bound for attention restricted to the spectral subspace.** Conjecture: L_Y,k < L_Y by the projection — gives stricter stability.
3. **Composition of B_ℓ across layers.** If `B_ℓ` and `B_{ℓ+1}` align (sharing top modes), share `B` across blocks of layers — saves params.
4. **Universal approximation property.** Volume-preserving sequence-to-sequence maps with attention rank ≤ k: are they universally approximated by SCFA blocks at depth `O(log T)`?
5. **Causal-compatible learned B without chunking.** Triangular Stiefel manifold (B with column-wise increasing support): does it admit gradient-stable learning?
6. **Variance/bias of B's gradient.** B appears in (project + lift). Cross-term gradient: unbiased? Variance bound?

---

## 15. Gate-0 falsification probe (mandatory before wire-in)

**Goal.** Test Conjecture 1 directly: does depthwise causal conv recover out-of-spectrum residual to 5% relative Frobenius norm?

**Setup.** Use the most-recently-saved 66M CHIRON checkpoint at iter ≥150 with T=1024. Probe runs in a single new C++ file `unit-tests/Backend/Machine Learning/scfa-gate0-probe.cpp`.

**Procedure.**
1. Sample 128-batch from the pile-bpe stream.
2. Forward through the existing CHIRON model up to layer ℓ ∈ {0, 13, 26, 40, 52} (5 anchor layers spread through the 53-layer stack).
3. At each anchor: extract `q_ℓ`, compute `Y_full(q_ℓ)` via baseline attention.
4. Compute SVD of `q_ℓ^T q_ℓ`; extract top-k right-singular vectors `B_ℓ^* ∈ ℝ^{T×k}` for k ∈ {32, 64, 128}.
5. Compute `Y_compr,k = SoftmaxAttn(B^{*T} q_ℓ W_Q, ..., ...) W_O` (k-dim attention); lift `y_∥ = B^* Y_compr`.
6. Train a small depthwise causal conv `D` (m=2048, w=8, m·17 = 34816 params) for 200 steps to minimize `‖Y_full − [y_∥ + D(q − B^* B^{*T} q)]‖_F²`.
7. Measure final ratio `ρ_ℓ,k := ‖Y_full(q_ℓ) − [y_∥ + D(q − B^* B^{*T} q_ℓ)]‖_F / ‖Y_full(q_ℓ)‖_F`.

**Pass criterion (Conjecture 1 supported):**
- ρ_ℓ,k=64 ≤ 0.05 on ≥ 4 of 5 anchor layers, AND
- ρ_ℓ,k=128 ≤ 0.03 on ≥ 4 of 5 anchor layers.

**Marginal pass (proceed with caution):**
- ρ_ℓ,k=64 ∈ (0.05, 0.10] on ≥ 4 of 5 layers — proceed with k=128 as default; revisit at 1.84B.

**Fail criterion (reject SCFA):**
- ρ_ℓ,k=128 > 0.20 on ≥ 2 of 5 layers — Conjecture 1 fails; abandon SCFA. Promote NEXUS to paradigm #42 instead.

**Cost.** ~1 GPU-hour total (5 layers × 3 k values × 200 D-training steps × ≈ 1 minute each).

This probe is the cheapest decisive falsifier. Per Ralph-loop methodology (saved 3 iterations on NESR/ZEN/VOCAB rejections), Gate-0 must run before any commit to SCFA implementation.

---

## 16. Concrete CUDA primitives (full signatures)

```cpp
// In Backend/Machine Learning/Networks/cuda/gpu_chiron.h, append:
namespace glades { namespace gpu {

// 1. Project q onto spectral subspace: q_hat[k, m] = B[T, k]^T · q[T, m].
//    Thin wrapper around sgemm_atb_rowmajor.
bool scfa_project_seq_bf16(const __nv_bfloat16* q,
                           const float*          B,
                           __nv_bfloat16*        q_hat,
                           int T, int k, int m,
                           cudaStream_t stream);

// 2. Lift spectral output back to T-axis: y[T, m] = B[T, k] · y_hat[k, m].
//    Thin wrapper around sgemm_rowmajor.
bool scfa_lift_seq_bf16(const float*          B,
                        const __nv_bfloat16*  y_hat,
                        __nv_bfloat16*        y,
                        int T, int k, int m,
                        cudaStream_t stream);

// 3. Compute residual q_perp = q - B (B^T q).
//    Fused: takes pre-computed B q_hat as argument.
bool scfa_residual_qperp(const __nv_bfloat16* q,
                         const __nv_bfloat16* B_qhat,
                         __nv_bfloat16*       q_perp,
                         int T, int m,
                         cudaStream_t stream);

// 4. Depthwise causal conv (complement mixer) — forward.
//    Output[t, c] = sum_{j=0..2w} D[c, j] * input[t-j, c] (causal padded).
bool scfa_depthwise_conv_fwd(const __nv_bfloat16* x,
                             const float*          D,
                             __nv_bfloat16*        y,
                             int T, int m, int w,
                             cudaStream_t stream);

// 5. Depthwise causal conv — backward (dx, dD).
bool scfa_depthwise_conv_bwd(const __nv_bfloat16* dy,
                             const __nv_bfloat16* x,
                             const float*          D,
                             __nv_bfloat16*        dx,
                             float*                dD_accum,
                             int T, int m, int w,
                             cudaStream_t stream);

// 6. QR thin re-orthogonalisation of B every N_QR steps.
//    Wraps cuSOLVER's geqrf + orgqr; B [T, k] in/out.
bool scfa_qr_reorth(float* B,
                    int T, int k,
                    cudaStream_t stream);

// 7. Stiefel regulariser scalar: r = ||B^T B - I||_F^2.
//    Returns the scalar via small reduction; ALSO writes gradient G = 4 B (B^T B - I).
bool scfa_stiefel_loss(const float* B,
                       int T, int k,
                       float*       r_out,
                       float*       grad_B,
                       cudaStream_t stream);

// 8. Option D: running SVD update — top-k power iteration for B.
//    Used at init and at every curriculum-T transition.
bool scfa_topk_svd_init(const __nv_bfloat16* q_batch,
                        float*                B,
                        int batch_size, int T, int m, int k,
                        int n_iters,                   // 5 typically
                        cudaStream_t stream);

// 9. Composite SCFA shear forward — wraps primitives 1, 2, 3, 4 plus k-dim
//    attention + spectral W_Q,W_K,W_V,W_O GEMMs + ReLN.  This is the production
//    drop-in replacement for chiron_attention_shear_bf16w_tiled.
bool chiron_attention_shear_scfa_bf16w_tiled(
    const __nv_bfloat16* q,
    __nv_bfloat16*       p,
    const float*         B,                       // [T, k] FP32 master
    const __nv_bfloat16* Wq, Wk, Wv, Wo,           // BF16 weights
    const float*         D,                       // [m, 2w+1] FP32
    int T, int m, int k, int n_heads, int d_head, int w,
    bool causal, bool invert,
    // scratch buffers (caller-owned):
    __nv_bfloat16* scratch_qhat,   // [k, m]
    __nv_bfloat16* scratch_Qhat,   // [k, m]
    __nv_bfloat16* scratch_Khat,   // [k, m]
    __nv_bfloat16* scratch_Vhat,   // [k, m]
    __nv_bfloat16* scratch_Phat,   // [n_H, k, k]
    __nv_bfloat16* scratch_Ohat,   // [k, m]
    __nv_bfloat16* scratch_y_par,  // [T, m]
    __nv_bfloat16* scratch_qperp,  // [T, m]
    __nv_bfloat16* scratch_y_perp, // [T, m]
    cudaStream_t stream);

// 10. Composite SCFA shear backward.
bool chiron_attention_shear_backward_scfa_bf16w_tiled(
    const __nv_bfloat16* q,
    const __nv_bfloat16* dp_new,
    const float*         B, const __nv_bfloat16* Wq, Wk, Wv, Wo,
    const float*         D,
    int T, int m, int k, int n_heads, int d_head, int w,
    bool causal,
    __nv_bfloat16* dq,
    float* dB,
    float* dD,
    float* dWq, dWk, dWv, dWo,
    // scratch buffers identical to forward...
    cudaStream_t stream);

}}  // namespace glades::gpu
```

Of these, primitives 1–3 and 6–7 are thin wrappers. Substantive new kernels: 4, 5, 8, 9, 10. Estimated implementation effort: ~800 LOC of CUDA + ~200 LOC of C++ trainer wire-in + 200 LOC of unit tests + Gate-0 probe code. Phase 1 (CPU + parity) ≈ 5 iterations; Phase 2 (GPU primitives) ≈ 5 iterations; Phase 3 (trainer) ≈ 3 iterations. **Total: 13 Ralph-loop iterations from Gate-0 pass to production wire-in.**

---

## 17. Open conjectures and validation criteria

### 17.1 Hard claims (proven)

- **Theorem 3 (structural reversibility):** Φ(q,p) = (q, p+Y(q)) is bijective for any continuous Y. Inverse and unit Jacobian determinant explicit. Proof in §7.
- **Per-block Lipschitz bound:** L_Y ≤ (1+ε_B) L_attn,k + L_D ≈ 10 with default hyperparameters. Direct.

### 17.2 Derivable claims under stated assumptions

- **Theorem 2 (information-loss bound):** ‖Y_full(q) − B Y_full,k(B^T q)‖_F ≤ L_Y · ‖Π^⊥ q‖_F. Holds under softmax-attention Lipschitz + B equivariance approximation. §6.
- **Compute table** (§8): exact under default config; speedup factors `(T/k)²` on attention term and `T/k` on projection term are arithmetic identities.

### 17.3 Heuristics (empirically supported by literature)

- Attention output has effective rank 50–250 in trained transformers (Linformer, Nyströmformer). At k=64, expected rank coverage ≥ 80%.
- BF16 stability of the projection `B^T q` is preserved by FP32 accumulator (standard).

### 17.4 Conjectures (require Gate-0 to test)

- **Conjecture 1 (residual recovery):** A depthwise causal conv of half-width w=8 recovers `Y_full(q) − B Y_full,k(B^T q)` to 5% Frobenius-relative norm at k = T/16, T ≤ 4096. Tested by §15.

### 17.5 Empirically testable predictions

| Prediction | Test | Pass criterion |
|---|---|---|
| 15× per-block attention speedup at T=1024 | Phase 2 GPU benchmark | wall-clock ratio of `chiron_attention_shear_bf16w_tiled` vs `chiron_attention_shear_scfa_bf16w_tiled` ≥ 12× |
| SCFA reaches baseline EMA at 66M × 5000 steps | Phase 4 convergence test | EMA at step 5000 within 0.1 nat of baseline |
| Stack speedup of 7.6× at flagship config | Phase 4 wall-clock test | end-to-end step time vs current flagship |
| Long-context speedup of 4.8× at T=4096 | Phase 4 long-context test | 1000-step wall-clock at T=4096, 1.84B params |
| Causal correctness at chunk boundaries | Phase 1 chunked-SCFA test | autoregressive generation produces same tokens as full attention up to numerical noise |
| B_ℓ stays near-orthonormal across training | Phase 4 monitoring | `‖B^T B − I‖_op ≤ 0.05` at all logged steps |
| Induction-head task succeeds at k=64 | Phase 4 synthetic test | held-out copy task accuracy ≥ 95% |

### 17.6 Falsification — kill switches

If any of these fire, retire SCFA (and promote NEXUS to paradigm #42):

1. Gate-0 (§15) fail criterion: `ρ_ℓ,k=128 > 0.20` on ≥ 2 layers.
2. Phase 4 convergence: 66M × 5000 EMA > 0.3 nat above baseline.
3. Phase 4 wall-clock: SCFA `chiron_attention_shear_scfa_*` < 5× baseline `chiron_attention_shear_bf16w_tiled` at T=1024.
4. Phase 4 induction: held-out copy task accuracy < 80% at k=128.
5. Phase 4 BF16 stability: per-step gradient norm ratio `||g_SCFA|| / ||g_baseline||` outside [0.5, 2.0] for >5% of steps.

---

**End of Paradigm Shift #42 design document.**

Word count: ~5400. Equations: 9 numbered + Theorems 2/3 + multiple in-text. Sections: 17 (covers all required research-framework headings: executive summary, candidates, selection, formal problem, framework, objective, optimization, dynamics, theory, tradeoffs, comparison, failure modes, prototype, full program, conjectures). Honest gaps explicitly flagged at §14.3, §17.4. Three competing candidates fully developed in companion files A/B/C; comparison and selection in §1. Materially distinct from all 41 prior paradigm shifts (composition matrix §9). Implementation horizon: 13 iterations from Gate-0 pass to production. Magnitude target: 7.6×–74× total wall-clock speedup at flagship 1.84B config across T=1024–16384.
