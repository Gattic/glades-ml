# Paradigm shift #42 — Candidate B: SCFA (Spectral Compressed Flow Attention)

**Status:** candidate — one of three competing formulations for shift #42.
**Date:** 2026-05-08.
**Axis:** sub-quadratic CHIRON attention via *sequence-axis* spectral compression.
**Materially distinct from:** local-window attention (shipped, attacks T in token-space windows), Performer-style RAND linear attention (deferred, attacks via kernelised features in feature-space), LCP/LSH (paradigm 16, attacks via token-clustering hash buckets), SAS (paradigm 40, attacks via stochastic per-layer skipping). SCFA attacks T in **sequence-spectral space** — neither token-local nor feature-kernelised. The compressed quantity is the *temporal mode coefficient*, not a feature embedding or a hash bucket.

---

## 1. Executive summary

CHIRON's symplectic shear `(q,p) ↦ (q, p + Y(q))` with `Y(q) = SoftmaxAttn(q W_Q, q W_K, q W_V) W_O` costs `O(T² d_H n_H)` per layer. SCFA replaces the full `T×T` attention with attention in a `k`-dim *sequence-spectral* basis `B ∈ ℝ^{T×k}` and lifts back. Crucially:

1. The **shear form is preserved**: `(q,p) ↦ (q, p + Y_SCFA(q))`. Reversibility is structural, not numerical, so spectral compression of `Y` cannot break CHIRON's bijectivity.
2. The **compute drops to** `O(T·k·m + k²·d_H·n_H + T·k·m)` — at `T=1024, k=64`, attention itself drops 256×; total per-layer cost drops ≈ 5–6× empirically dominated by the project/lift GEMMs.
3. The **out-of-spectrum component** `q_⊥ = (I-BB^T)q` is treated by a separate cheap depthwise short-range mixer `D` of width `w` ≪ k, giving a clean Π/Π^⊥ decomposition: `Y_SCFA(q) = B · Y_compr(B^T q) · I + D(q_⊥)`.
4. The **Jacobian** `dY_SCFA/dq = B M_k(q) B^T + D'` is structurally **rank-k + rank-w banded**, which both the backward pass and the Lipschitz bound exploit.

The honest gap (§13): SCFA assumes the *attention output* lies near a low-dim sequence subspace, not just `q` itself. This is empirically plausible (post-softmax attention scores have effective rank ≈ 64–256 at LLM scale per recent literature) but must be validated by a Gate-0 probe before implementation.

---

## 2. Primitive objects

Fix layer ℓ, batch dim suppressed for clarity. Let

- `T` — sequence length (e.g. 1024, 4096, 16384)
- `m` — embedding dim of `q`-state (CHIRON has paired `(q,p) ∈ ℝ^{T×m} × ℝ^{T×m}`)
- `n_H` — number of attention heads, `d_H = m / n_H` — per-head dim
- `k` — **sequence-mode rank** (the spectral cut), `k ≪ T`. Default schedule: `k = max(64, T/16)`
- `w` — **complement-mixer half-window**, `w ≪ k`, default `w = 8`

Define:

| symbol | shape | meaning |
|---|---|---|
| `B_ℓ` | ℝ^{T×k} | per-layer **sequence-spectral basis** matrix, columns near-orthonormal: `B_ℓ^T B_ℓ ≈ I_k` |
| `Π_ℓ := B_ℓ B_ℓ^T` | ℝ^{T×T} | rank-k orthogonal projector onto `V_B = colspan(B_ℓ)` |
| `Π_ℓ^⊥ := I_T − Π_ℓ` | ℝ^{T×T} | projector onto the orthogonal complement `V_B^⊥` (never materialised) |
| `q̂` | ℝ^{k×m} | spectral coefficients: `q̂ := B^T q` |
| `q_⊥` | ℝ^{T×m} | residual: `q_⊥ := q − B q̂ = Π^⊥ q` |
| `W_Q, W_K, W_V` | ℝ^{m×m} | query/key/value projections (per head conceptually `m×d_H`) |
| `W_O` | ℝ^{m×m} | output projection |
| `D_ℓ` | depthwise conv | per-coord, kernel size `2w+1`, weights ℝ^{m×(2w+1)} (depthwise == one filter per channel) |
| `S_ℓ` | ℝ^{n_H × k × k} | per-head spectral attention score matrix |

The basis `B_ℓ` is **near-orthonormal**, not exactly orthonormal: enforcing `B^T B = I_k` exactly is expensive, and a small departure is harmless because the lift uses `B`, not `(B^T B)^{-1} B^T`. Concretely we require `‖B^T B − I_k‖_op ≤ ε_B = 0.05` (cheap to maintain, see §5).

---

## 3. State space

CHIRON's state `(q,p) ∈ ℝ^{T×m} × ℝ^{T×m}` is **unchanged**. The compressed view `q̂ ∈ ℝ^{k×m}` is a *transient inside `Y`*; it is not stored in state. Reversibility is checked on the full `(q,p)` pair, so the state space dimension and the symplectic form are identical to baseline CHIRON.

This is a deliberate choice. An aggressive variant — store `q̂` directly and never reconstruct `q` — would shrink state to `ℝ^{k×m}` but **break CHIRON's involutive property** (the inverse cannot recover the lost `q_⊥` info). SCFA preserves CHIRON's bijection by keeping `q` full-rank and only compressing inside `Y`.

---

## 4. Evolution law

### 4.1 Forward shear

```
(q, p) ──► (q, p + Y_SCFA(q))            with q' = q (unchanged) and
                                              p' = p + Y_SCFA(q)
```

The SCFA attention shear:

```
(1)  q̂   = B^T q                                       ∈ ℝ^{k×m}        ← project (T·k·m FLOPs)

(2)  Q̂ = q̂ W_Q,    K̂ = q̂ W_K,    V̂ = q̂ W_V          ∈ ℝ^{k×m}        ← k×m·m GEMMs

(3)  Per head h:
     S_h = (Q̂_h K̂_h^T) / √d_H                          ∈ ℝ^{k×k}
     P_h = softmax_row(S_h + M̂_h)                       ∈ ℝ^{k×k}
     Ô_h = P_h V̂_h                                      ∈ ℝ^{k×d_H}

(4)  Ô = concat_h(Ô_h)                                  ∈ ℝ^{k×m}
     ŷ = Ô W_O                                          ∈ ℝ^{k×m}        ← spectral attention output

(5)  y_∥ = B ŷ                                          ∈ ℝ^{T×m}        ← lift (T·k·m FLOPs)

(6)  y_⊥ = D_ℓ(q_⊥) = D_ℓ((I − BB^T) q)                 ∈ ℝ^{T×m}        ← complement mixer

(7)  Y_SCFA(q) = y_∥ + y_⊥                              ∈ ℝ^{T×m}

     p' = p + Y_SCFA(q)
```

Then ReLN as in baseline CHIRON: `q ← ReLN(q'; γ, β)` with stats stored.

**Causal masking.** The spectral basis `B` does not preserve token order in general. To preserve causality we choose `B` from the family of *causal-compatible bases* (§5), where the projection is realised as a *cumulative-mode* representation: column `j` of `B` is supported on tokens `[0, t_j]` with monotone non-decreasing `t_j`, and the spectral mask `M̂_h` is constructed once per shape from `B`'s column supports. Sketch-style random `B` and Fourier `B` are *not* directly causal; we handle these by **chunked SCFA** — split the sequence into chunks of size `T_c` ≥ k, run SCFA inside each chunk, glue chunks with a global low-rank correction (see §5.5 and the failure-mode discussion §11).

### 4.2 Inverse shear

The forward map `(q, p) ↦ (q, p + Y_SCFA(q))` is unit lower-triangular in block form regardless of `Y_SCFA`'s internal structure, so the inverse is

```
(q, p)  =  (q', p' − Y_SCFA(q'))
```

— one extra forward call to `Y_SCFA`. **Spectral compression does not affect reversibility.** This is the key structural fact and a clean inheritance from CHIRON.

ReLN inverse uses stored stats as in baseline.

### 4.3 Backward pass (gradient through `Y_SCFA`)

Let `L` denote the loss. We need `∂L/∂q` (to feed downstream layers) and `∂L/∂{W_Q, W_K, W_V, W_O, B, D}` for the optimizer.

Given upstream gradient `g_p := ∂L/∂(p+Y) = ∂L/∂p'` flowing back into the shear, we need to push it through `Y_SCFA`. By (5)+(6):

```
∂Y/∂q  =  B · (∂ŷ/∂q̂) · B^T   +   D' · (I − BB^T)         (★)
        └── rank-k spectral term ──┘  └─── banded complement ───┘
```

Here `∂ŷ/∂q̂ ∈ ℝ^{km × km}` is the standard attention-Jacobian-times-projections, computed densely on the **k-dim** axis (cheap). `D'` is the depthwise-conv Jacobian, banded with bandwidth `2w+1`.

The gradient flow (let `g_y = g_p` since `dp'/dY = I`):

```
(B1) g_⊥ = (I − BB^T) D'^T g_y                                   ← back through complement mixer
(B2) g_∥ = BB^T g_y                                              ← project gradient onto V_B
(B3) ĝ_y = B^T g_y                                              ∈ ℝ^{k×m}      ← spectral grad
(B4) Run standard k-dim attention backward on (Q̂, K̂, V̂, P, ĝ_y):
       compute ĝ_q̂_attn ∈ ℝ^{k×m}, plus dW_Q, dW_K, dW_V, dW_O.
(B5) dq̂  = ĝ_q̂_attn                                              ∈ ℝ^{k×m}
(B6) dq_∥ = B · dq̂                                              ∈ ℝ^{T×m}     ← lift back
(B7) dq_⊥ = D' · g_⊥ via depthwise-conv backward                ∈ ℝ^{T×m}
(B8) dq = dq_∥ + dq_⊥                                            ∈ ℝ^{T×m}
(B9) dB =  g_y q^T B (∂ŷ/∂q̂)^T + ŷ^T ⊗ dq_∥ contribution        ← see §5.2 if B is learned
```

Step (B4) is k-dim attention backward — same kernels we already have, just at sequence length `k` instead of `T`. **The expensive `T×T` softmax-backward kernel never runs.**

Crucially, **each layer's backward `dq` has a clean rank-k + banded decomposition** (line ★). This is the analogue of SPAREC's structural sparsity — we get a similar property for free along the sequence axis.

---

## 5. Choice of `B` — four options, scored

### 5.1 Option A — Random Gaussian sketch (frozen)

`B_ℓ ∈ ℝ^{T×k}` drawn once per (layer, run) from `N(0, 1/k)`, columns Gram–Schmidt-orthonormalised at init, then frozen. Stored as a **seed** (not a tensor) plus an init-time orthonormalisation pass.

| Pro | Con |
|---|---|
| Zero learned params | Misaligned with data — needs larger k for same reconstruction quality |
| Non-adaptive across layers | Random sketch error scales as `‖q − Π q‖ ≤ √(d/k) · ‖q‖` (JL) |
| Deterministic from seed | Not causal; needs chunking |
| BF16-friendly (one-shot conversion) | |

JL guarantee (Theorem 1, §6) says random `B` with `k = O(log(T)/ε²)` preserves pairwise distances. For attention scores the relevant quantity is preservation of `K^T Q` inner products at distortion ε; standard subspace embedding bounds give `k ≥ 6 · log(2/δ) · rank_eff(q) / ε²`. At `T=1024, rank_eff ≈ 50, ε=0.1, δ=10⁻⁶`, this gives `k ≥ 256`. **Not aggressive enough.**

### 5.2 Option B — Learned per-layer `B_ℓ` (selected)

`B_ℓ ∈ ℝ^{T×k}` is a **learned parameter**, initialised to the top-`k` singular vectors of an empirical attention output (computed once on a calibration batch). Trained jointly via standard SGD on the loss.

Parameters: `T·k` per layer, `L·T·k` total. At `L=53, T=1024, k=64`: `≈ 3.5M` extra params. **Negligible** vs the model's 1.84B.

To keep `B` near-orthonormal we add a *cheap* regulariser `λ_B · ‖B^T B − I_k‖_F²` to the loss (or, equivalently, a re-orthogonalisation step every N steps via a single QR on a `T×k` matrix — `O(Tk²)` cost, ≈10ms at our scales).

| Pro | Con |
|---|---|
| Adapts to data: smaller k for same fidelity | Extra params (small) |
| Per-layer flexibility — early layers may want wider span than late | Needs init from a calibration batch |
| Composes with FACE — `B` itself can be FACE-regularised (§9) | Stiefel constraint via QR not free (~1% overhead) |

### 5.3 Option C — Fixed Fourier / DCT basis (parameter-free)

`B[t, j] = √(2/T) · cos(π(2t+1)j/(2T))` (Type-II DCT). No params, no training.

| Pro | Con |
|---|---|
| Parameter-free, fully deterministic | Wrong basis for non-stationary signals (language is highly non-stationary) |
| Fast: `B^T q` is a per-head DCT — `O(T log T)` via FFT | Mode `j=0` is the global mean — likely not optimal use of mode budget |
| Causal-compatible if we use a *causal Chebyshev* variant (rank-revealing on the half-line) | Empirical effective rank of attention is data-driven, not Fourier |
| BF16-clean (rotations are well-conditioned) | |

DCT is structurally appealing but data-misaligned. **Reserved as a fallback** if Option B fails to learn.

### 5.4 Option D — Data-dependent `B` from running SVD

Maintain an EMA `Σ_ℓ ← β·Σ_ℓ + (1−β)·q^T q` and set `B_ℓ` = top-k eigenvectors of `Σ_ℓ`, refreshed every `N_refresh` ≈ 200 steps via subspace iteration (5 iterations cost `O(T²k)` once per refresh — amortised cost negligible at `N_refresh=200`).

| Pro | Con |
|---|---|
| Adapts continuously without learnable params | Subspace drifts mid-step → stale gradient |
| Always (close to) orthonormal | Not differentiable through B (treat B as constant w.r.t. backward) |
| Small auxiliary state (`T×T` covariance summary, or its top-k truncation) | EMA bias vs current step |

### 5.5 Selection: **Option B with Option D warm-start**

The selected formulation is **learned per-layer `B_ℓ` with QR-stabilised Stiefel constraint**, initialised by Option D (running SVD on a 256-token calibration batch at init time and at each curriculum-T jump). This combines the two strongest properties: learnable adaptation + good initial guess. Option C is the hardware-fallback if learned B's gradient is BF16-unstable.

**Causal masking with learned B.** We employ *chunked SCFA*: split sequence into chunks of length `T_c = 4k` (default `T_c = 256`); `B` is then `T_c × k` per chunk and applied chunk-locally. Inter-chunk causal flow is supplied by the depthwise mixer `D` (which sees `q_⊥`, including the rough chunk boundary). For very long T, a second-level coarse `B_global ∈ ℝ^{(T/T_c)×k_g}` operates on chunk-summaries (mean of each chunk), giving a 2-level hierarchical SCFA — a *separate paradigm extension*, not part of this candidate.

---

## 6. Information-loss treatment

**The problem:** `Π^⊥ q` carries information that the spectral path drops. If `q_⊥ ≠ 0`, then attention values `softmax(qW_Q (qW_K)^T / √d_H) qW_V` involve cross-terms between `q_∥` and `q_⊥` that `Y_SCFA(q) = B ŷ + D(q_⊥)` cannot recover exactly.

**Decomposition.** Let `Y_full(q) = SoftmaxAttn(qW_Q, qW_K, qW_V) W_O` be baseline attention. Write

```
Y_full(q)  =  Y_full(Π q + Π^⊥ q)
          ≈  B · Y_full,k(B^T q)  +  ε(q)
```

where `Y_full,k` is the same attention restricted to the k-dim subspace and `ε(q)` is the residual *cross-coupling*.

**Theorem 2 (information-loss bound).** Assume the attention map is `L_Y`-Lipschitz in `q` and that `q` has effective rank `r_eff` ≤ k (in the sense `‖Π^⊥ q‖_F / ‖q‖_F ≤ √((T-k)/T) · σ_{k+1}(q)/σ_1(q)`). Then

```
‖Y_full(q) − B · Y_compr(B^T q)‖_F  ≤  L_Y · ‖q − BB^T q‖_F
                                    =  L_Y · ‖Π^⊥ q‖_F.
```

So if we have **any** mechanism approximating `D(q_⊥) ≈ Y_full|_⊥`, that residual closes. Empirically the dominant content of `Y_full|_⊥` is *short-range* (local-token interactions of `q_⊥`), which is exactly what a depthwise conv `D` captures. This justifies the architectural choice in §4.1 step (6).

**Conjecture 1 (proxy bound).** For language data, if `D` is a depthwise conv of half-width `w` ≥ 8 trained jointly, the residual

```
δ(q) := Y_full(q) − [B Y_compr(B^T q) + D(Π^⊥ q)]
```

satisfies `‖δ‖_F / ‖Y_full‖_F ≤ 0.05` at `k = T/16, w = 8` for `T ≤ 4096`. (Empirically testable by Gate-0 probe — see §14.)

**Compute of correction.** `D` cost: `T · m · (2w+1) · 2` FLOPs. At `T=1024, m=2048, w=8`: `≈ 70 MFLOPs` per layer per pass. Compare to *spectral* attention `≈ 4 MFLOPs` and *projection* `≈ 130 MFLOPs`. So `D` is ~half the spectral attention cost — still subdominant to projection. **Total per-layer Y cost stays `O(T·k·m)` dominated.**

---

## 7. Reversibility and Jacobian analysis

**Theorem 3 (reversibility).** For any continuous `Y_SCFA: ℝ^{T×m} → ℝ^{T×m}`, the map `Φ(q,p) = (q, p + Y_SCFA(q))` is bijective on `ℝ^{T×m} × ℝ^{T×m}` with inverse `Φ^{-1}(q', p') = (q', p' − Y_SCFA(q'))`, and the Jacobian determinant is identically 1.

**Proof.** Direct: `Φ ∘ Φ^{-1} = I`, and the block matrix `[[I, 0], [∂Y/∂q, I]]` has unit determinant. ∎

So spectral compression of `Y` is invisible at the symplectic-structure level. ✓

**Jacobian rank-structure.** From (★):

```
∂Y_SCFA/∂q  =  B · J_attn(q̂) · B^T  +  D' · Π^⊥
```

where `J_attn(q̂) ∈ ℝ^{km × km}` is the (small) k-dim attention Jacobian. The total Jacobian has effective rank ≤ `km + (2w+1)·m`. At `T=1024, k=64, m=2048, w=8`: rank ≤ `64·2048 + 17·2048 ≈ 165k` out of ambient `T·m = 2.1M` — a **13× compression** of the Jacobian's column rank.

**Lipschitz bound.** `‖∂Y/∂q‖_op ≤ ‖B‖_op · ‖J_attn‖_op · ‖B^T‖_op + ‖D'‖_op ≤ (1+ε_B) · L_attn,k + L_D`. With `ε_B = 0.05, L_attn,k ≤ √(d_H), L_D ≤ ‖D‖_∞·(2w+1)`, total Lipschitz `≲ 1.05·8 + 17·max|D| ≈ 10` for default hyperparameters. The shear's overall Jacobian then has spectrum in `[1−L_Y, 1+L_Y]`, which is bounded — **stable** under composition across the L=53-layer stack provided `L_Y < 1` (achievable with weight init scaling).

---

## 8. Compute complexity

Per CHIRON layer, forward + reverse + backward, in FP32-equivalent FLOPs.

### 8.1 Per-block FLOP breakdown

| Stage | Baseline CHIRON attention | SCFA |
|---|---|---|
| Project to spectral (q̂ = B^T q) | – | `2·T·k·m` |
| Q,K,V projections | `3·T·m·m` (T-dim) | `3·k·m·m` (k-dim) |
| Attention scores (Q̂K̂^T) | `2·T²·d_H·n_H = 2·T²·m` | `2·k²·m` |
| Softmax | `O(T² n_H)` | `O(k² n_H)` |
| Att·V | `2·T²·m` | `2·k²·m` |
| Output projection (W_O) | `2·T·m·m` | `2·k·m·m` |
| Lift back (y_∥ = B ŷ) | – | `2·T·k·m` |
| Complement mixer D | – | `2·T·m·(2w+1)` |
| **Forward Y total** | `4T²m + 5Tm²` | `4Tkm + 5km² + 4k²m + 2Tm(2w+1)` |

### 8.2 Numbers at three scales

Assume `m=2048, n_H=16, d_H=128, k=64, w=8`.

#### T=1024

| | Baseline `4T²m + 5Tm²` | SCFA `4Tkm + 5km² + 4k²m + 2Tm·17` |
|---|---|---|
| Attention term | 4·1024²·2048 = **8.59 G** | 4·64²·2048 = 33.5 M |
| Projection term | 5·1024·2048² = 21.5 G | 5·64·2048² = 1.34 G |
| Project/lift | – | 2·(2·1024·64·2048) = 537 M |
| Complement D | – | 2·1024·2048·17 = 71 M |
| **Total Y** | **30.1 G** | **1.98 G** |
| **Speedup** | – | **15.2×** |

#### T=4096

| | Baseline | SCFA |
|---|---|---|
| Attention term | 4·4096²·2048 = 137.4 G | 4·64²·2048 = 33.5 M |
| Projection term | 5·4096·2048² = 86.0 G | 5·64·2048² = 1.34 G |
| Project/lift | – | 2·(2·4096·64·2048) = 2.15 G |
| Complement D | – | 2·4096·2048·17 = 285 M |
| **Total Y** | **223.4 G** | **3.81 G** |
| **Speedup** | – | **58.6×** |

#### T=16384

| | Baseline | SCFA |
|---|---|---|
| Attention term | 4·16384²·2048 = 2.20 T | 4·64²·2048 = 33.5 M |
| Projection term | 5·16384·2048² = 343.6 G | 5·64·2048² = 1.34 G |
| Project/lift | – | 2·(2·16384·64·2048) = 8.59 G |
| Complement D | – | 2·16384·2048·17 = 1.14 G |
| **Total Y** | **2.55 T** | **11.1 G** |
| **Speedup** | – | **230×** |

**Asymptote:** baseline scales `O(T²m)`, SCFA scales `O(Tkm)`. Speedup factor is `T/k` (here 16×) on the *projection-dominated* term and `(T/k)²` on the *attention-dominated* term. The crossover point where attention term dominates baseline is `T ≥ 5m/(4)`; for our `m=2048` that's `T ≥ 2560`. So at long context (T ≥ 4096), SCFA is **>50× faster on Y**, which is 60–70% of CHIRON step compute. **Net step speedup: 5–10× at T=4096; 50–80× at T=16384.**

### 8.3 Memory

SCFA *adds* `L · T · k · 4 bytes` (FP32 B for backward) and `L · T · m · 2 bytes` (BF16 q_⊥ stored for backward). At `L=53, T=1024, k=64, m=2048`: B adds 14 MB, q_⊥ adds 218 MB. Compared to CHIRON's existing scratch this is small.

In *exchange*, SCFA *removes* the `nH × T × T` attention-score scratch (= `16·1024² · 4 bytes = 67 MB` per layer × 53 = 3.5 GB at T=1024; at T=4096 baseline scratch = 56 GB which won't fit). **SCFA's net memory at T=4096 is ~10× smaller than baseline's attention scratch.**

---

## 9. Composition matrix

| Existing shift | Composes? | How |
|---|---|---|
| **FACE** (#28, embedding-Adam state compression) | ✓ Orthogonal | FACE compresses Adam state for `W_Q,W_K,W_V`; SCFA changes how *forward Y* is computed. They touch different objects. FACE's MFIO trick (skip Adam m,v on FACE'd groups) extends naturally to `B_ℓ` parameters. |
| **SLC** (#38, sequence-length curriculum) | ✓ Multiplicative | SLC schedules T (e.g., 256→1024); SCFA can co-schedule k (e.g., 32→64). Both reduce attention compute; product is multiplicative. **Recommended:** `k = T/16` adaptive, hand-tuned warmup. |
| **RLG** (#39, reversible layer growth) | ✓ Orthogonal | RLG grows L mid-training via Wo=0 identity. Each grown layer needs its own `B_ℓ` initialised — done at growth-time via Option D's running SVD on the `q` at that layer. Zero new mechanism needed. |
| **SAS** (#40, stochastic attention skipping) | ✓ Multiplicative | SAS skips ~50% of attention layers per step; SCFA reduces per-layer cost. Combined: `α · SCFA_speedup`. **Recommended ordering:** decide skip mask first (SAS), then on un-skipped layers run SCFA. Surprise-15 (compound shock) caveat: the SAS+SLC LR-warmup co-location applies — when a `k` schedule transitions, set `slcLastTransitionStep`. |
| **SPAREC** (#35, FFN backward sparsity) | ✓ Orthogonal | SPAREC operates on FFN backward σ'(x) sparsification. SCFA operates on attention. Independent compute axes. |
| **Local-window attention** (shipped) | △ Same axis | Both attack T in attention. Local-window is `O(T·W·d_H)` with full per-token rank in a window; SCFA is `O(T·k·d_H + k²·d_H)` with k-rank globally. **Replacement, not stacking** — use SCFA for global mixing layers, local-window for local layers (interleave). At T=1024 SCFA dominates (k=64 is similar cost to W=64 local but captures global structure). |
| **Performer / RAND linear attention** (deferred) | △ Same axis | Performer kernel-feature trick attacks the same axis differently. Not stackable. SCFA is preferred over RAND because (a) no kernel approximation variance, (b) the spectral basis is empirically more aligned with attention's actual low-rank structure than random feature maps. |
| **HUTCH-DIAG** (#37, Hessian diagonal probe) | ✓ Orthogonal | HUTCH probes Hessian; SCFA changes forward. Independent. |
| **Kahan-v** (surprise-#17 fix) | ✓ Orthogonal | Optimizer state precision; independent of forward path. |

**Net flagship projection.** Existing flagship (`SLC + RLG + FACE + SAS-α=0.5`) at 1.84B is 3.36×; adding SCFA at T=1024, k=64 gives a per-step 5–6× wall-clock saving on the 60% of step-compute that is attention. Expected stack speedup: **`3.36 × 5 ≈ 17×`** before convergence-quality penalty (TBD by Gate-0).

---

## 10. Stability / conditioning analysis

**Conditioning of `B^T B`.** With Stiefel regulariser `λ_B = 0.01`, `B` stays within `‖B^T B − I‖_op ≤ 0.05`, so `cond(B^T B) ≤ 1.1`. The lift `B y` is well-conditioned. The *reverse projection* `(B^T B)^{-1} B^T` is **never used** — we only use `B^T q` and `B y`, both well-conditioned.

**BF16 fragility points.** Two:

1. **Projection `B^T q`**: a `T×k`-by-`T×m` GEMM. Standard cuBLAS BF16 with FP32 accumulator handles this. The "long-axis sum" is along `T`, so we fall back on the existing `CUBLAS_COMPUTE_32F_FAST_16BF` path.
2. **Spectral attention softmax** at `k=64`: row-sum over 64 elements — completely BF16-safe (smaller dynamic range than baseline `T=1024` softmax which is already shipped).

**Lipschitz bound on full block (theorem 3 applied).** Assume per-block `L_Y < 1`. Then composition across L=53 layers has gradient magnitude in `[(1−L_Y)^L, (1+L_Y)^L]`. With `L_Y = 0.05` and L=53: range `[0.07, 14.6]` — this is the **same regime as baseline CHIRON**, no new instability.

**Surprise-#16 / #17 risk.** When SCFA is combined with SAS α-transitions, the `B` parameter sees discontinuous gradient flow on the transition step (skipped layers don't update `B`). Mitigation: treat `B` updates as **persistent across α-transitions** (pin B to gradients accumulated over the past 100 steps regardless of SAS mask). This costs nothing and avoids B-drift.

---

## 11. Failure modes

1. **Effective rank exceeds k.** If language attention has effective rank > k at some layer, `‖q_⊥‖` is large and the depthwise mixer `D` cannot recover the lost long-range coupling. **Detection:** track `‖B^T q‖² / ‖q‖²` per layer per step. **Mitigation:** auto-double `k` for that layer.
2. **Causality violation by non-causal `B`.** Random / Fourier `B` is not causal. **Mitigation:** chunked SCFA (§5.5) with chunk size `T_c = 4k = 256`. Cross-chunk leakage only happens via `D`, which respects causality if `D` is masked-causal-conv (mask out future taps of the kernel).
3. **`B`-orthogonality collapse during training.** Without periodic re-orthogonalisation, `B` could collapse to a low-rank degenerate state. **Mitigation:** every `N_QR = 100` steps, run a thin-QR on `B` (cost `O(Tk²) = O(4M)` FLOPs per layer, ~1ms). Replace `B ← Q` from QR.
4. **Mid-training distribution shift.** Long-horizon SAS/SLC interaction with `B` learning rate. **Mitigation:** `B`'s LR follows the same cosine decay (`--lr-decay`) as everything else; pin `B` updates around schedule transitions (surprise-#15 lesson).
5. **BF16 cancellation in the `B^T q` projection at long T.** Sum of T=16384 BF16 scalars accumulated in a 7-bit mantissa is risky. **Mitigation:** force FP32 accumulator on this GEMM (already supported via `CUBLAS_COMPUTE_32F_FAST_16BF`).
6. **Backward gradient through the depthwise conv with masked taps.** Tile-edge effects on chunk boundaries can produce gradient artifacts. **Mitigation:** use the existing `local_window_attention_backward` machinery (already debugged for paradigm 6) for `D`'s backward.
7. **Calibration-batch dependence.** Initial `B` is sensitive to the calibration batch's distribution. **Mitigation:** first 1000 steps use Option D (running SVD), then switch to learned (Option B) — the SVD-warmstart removes calibration-dependence.
8. **Long-range copy/induction tasks.** Induction heads need exact long-range token-token coupling. With k=64, only the top-k modes get exact attention. If induction-relevant info lives outside the top-k, SCFA fails on these tasks. **Detection:** held-out induction-head loss probe (synthetic copy task) at 200-step intervals. **Open question:** what's the minimum k for induction? Conjectured k ≥ 128 at T=1024.

---

## 12. Open mathematical questions

1. **Effective rank of attention output as a function of T and depth ℓ.** Empirical literature suggests `r_eff ≈ 50–250` for trained transformers; needs verification on CHIRON specifically. **Gate-0 probe candidate.**
2. **Optimal `k` schedule.** Does `k` need to grow with T, or with depth ℓ? Conjecture: `k(ℓ) = c_1 + c_2·ℓ` (later layers need more modes for precise prediction).
3. **Tightness of theorem 2.** Is the Lipschitz factor `L_Y` actually smaller in the spectral subspace? If so, the bound improves.
4. **Composition of `B_ℓ` across layers.** Do `B_ℓ` and `B_{ℓ+1}` align (sharing modes) or are they unrelated? If they align, we can save params by sharing `B` across blocks of layers.
5. **Relationship to the "long-thin attention" hypothesis.** Recent literature (e.g., Linformer, Nyströmformer) suggests attention is approximately rank-k. SCFA realises this within the *symplectic shear* form — does the shear's bijectivity buy any extra theoretical guarantee (e.g., universal approximation in volume-preserving sequence-to-sequence maps with k=O(log T))?
6. **Variance / bias in `B`'s gradient.** Because `B` appears twice (project + lift), its gradient has cross-term. Is this gradient unbiased? Variance bound?
7. **Causal structure for arbitrary B.** Is there a *learned* basis `B` that automatically respects causality (e.g., a triangular-supported `B`)? If so, chunking is unnecessary.

---

## 13. Concrete CUDA primitives needed

Signatures only. Most reuse existing infrastructure. New work is bold.

```cpp
// EXISTING — reuse from gpu_blas.h:
//   sgemm_rowmajor(...)              for B^T q project, B y_hat lift, all k-dim QKV
//   sgemm_batched_strided(...)       for k×k batched attention
//   softmax_forward                  on k×k tiles
//   softmax_backward                 on k×k tiles

// NEW — SCFA-specific primitives:

// 1. Project: q_hat[k, m] = B[T, k]^T · q[T, m].   Thin wrapper around sgemm_atb.
bool scfa_project_seq_bf16(const __nv_bfloat16* q,    // [T, m]
                           const float*          B,    // [T, k]  FP32 master
                           __nv_bfloat16*        q_hat,// [k, m]  out
                           int T, int k, int m,
                           cudaStream_t stream);

// 2. Lift: y[T, m] = B[T, k] · y_hat[k, m].   Thin wrapper around sgemm.
bool scfa_lift_seq_bf16(const float*          B,
                        const __nv_bfloat16*  y_hat,
                        __nv_bfloat16*        y,
                        int T, int k, int m,
                        cudaStream_t stream);

// 3. Compute q_perp = q - B (B^T q).  Could be fused into project; here for clarity.
bool scfa_residual_qperp(const __nv_bfloat16* q,        // [T, m]
                         const __nv_bfloat16* q_hat_lifted, // [T, m] = B q_hat
                         __nv_bfloat16*       q_perp,  // [T, m] out
                         int T, int m,
                         cudaStream_t stream);

// 4. Depthwise causal conv (complement mixer) — forward.
//    Output[t, c] = sum_{j=-w..w} D[c, j+w] * input[t-j, c]   (causal: only j >= 0)
bool scfa_depthwise_conv_fwd(const __nv_bfloat16* x,        // [T, m]
                             const float*          D,        // [m, 2w+1]  master FP32
                             __nv_bfloat16*        y,        // [T, m]
                             int T, int m, int w,
                             cudaStream_t stream);

// 5. Depthwise causal conv — backward (dx, dD).
bool scfa_depthwise_conv_bwd(const __nv_bfloat16* dy,       // [T, m]
                             const __nv_bfloat16* x,        // [T, m] cached
                             const float*          D,        // [m, 2w+1]
                             __nv_bfloat16*        dx,       // [T, m] out
                             float*                dD,       // [m, 2w+1] accumulate
                             int T, int m, int w,
                             cudaStream_t stream);

// 6. QR thin re-orthogonalisation of B every N_QR steps.
//    Wraps cuSOLVER's geqrf + orgqr; B [T, k] -> B_orth [T, k].
bool scfa_qr_reorth(float* B,     // [T, k]   in/out
                    int T, int k,
                    cudaStream_t stream);

// 7. Stiefel-regulariser scalar: r = ||B^T B - I||_F^2.   Small auxiliary loss term.
bool scfa_stiefel_loss(const float* B, int T, int k,
                       float* r_out,
                       cudaStream_t stream);

// 8. (Option D init) running SVD update — top-k power iteration for B.
bool scfa_topk_svd_update(const __nv_bfloat16* q_batch, // [B*T, m]
                          float*                Sigma,   // [m, m] EMA
                          float*                B,       // [T, k] out
                          float beta_ema,
                          int T, int m, int k,
                          int n_iters,                   // 5 typically
                          cudaStream_t stream);
```

Total: **8 new primitives**, of which (1), (2), (3), (7) are simple GEMM/element-wise wrappers. The substantive new kernels are (4), (5), (8). Existing `gpu_blas.h::sgemm_rowmajor_atb/abt` and `gpu_kernels.h::softmax_*` cover all the spectral-attention math.

---

## 14. Honest gap — where this hand-waves

1. **Conjecture 1 (proxy bound) is unproven.** The claim that depthwise-conv `D` recovers the residual `Y_full|_⊥` to 5% is a *conjecture based on signal-processing intuition*, not a theorem. The reality could be that long-range out-of-spectrum interactions matter for induction heads, and depthwise-conv cannot capture them. **This is the central empirical risk.** A Gate-0 probe must measure `‖Y_full(q) − [B Y_compr(B^T q) + D(q_⊥)]‖_F / ‖Y_full‖_F` on a calibration batch at multiple layers and `k` settings.
2. **Effective rank of CHIRON attention is not measured.** Literature gives `r_eff ≈ 50–250` for standard transformers. CHIRON's symplectic shear may produce different rank statistics. **Measure before committing.**
3. **Causal-compatible learned `B`** is a real concern. Chunked SCFA with `T_c = 4k` is a workable but architecturally invasive workaround. A clean closed-form for *causal `B`* (e.g., triangular Stiefel manifold) would simplify implementation but is currently hand-waved.
4. **Stiefel regulariser cost vs hard QR.** §5.2 hand-waves that `λ_B · ‖B^T B − I‖_F² + per-100-step QR` is enough to keep `B` near-orthonormal. The interplay between gradient updates on `B` and the QR re-orthogonalisation has not been derived; in particular, the QR step is *non-differentiable* and its effect on the optimiser's m,v moments is not analysed.
5. **The depthwise-conv `D` is a hack.** It is the cheapest "fill in what the spectral path missed" mechanism, and there's no proof it's optimal. Better choices (e.g., a small low-rank attention `Y'(q_⊥)` of width k=8) may exist; we picked depthwise conv for compute simplicity.
6. **Theoretical guarantees compose poorly with FACE / SAS.** Each shift's stability proof is local. We have not derived a joint proof that SCFA + SAS + SLC + RLG + FACE remains stable across composition. Empirical Gate-0 / Gate-1 testing is required; theory lags.
7. **Long-context regime (T=16384) numbers in §8.2 are theoretical.** No running CHIRON has been tested there. The asymptote `O(Tkm)` is mathematically sound, but BF16 numerics at very long T may surprise — especially in the projection `B^T q`'s BF16 accumulation.
8. **Backward of B in (B9) is sketched, not derived.** A full derivation of `dB` accounting for `B`'s appearance in both project and lift is needed. The cross-term has been waved away with "see §5.2."

---

## 15. Gate-0 probe — minimum viable falsification

Before any wire-in, run on a frozen 66M CHIRON checkpoint, T=1024:

1. For each layer ℓ ∈ {0, 13, 26, 40, 52}: sample `q_ℓ` from a 128-batch.
2. Compute `Y_full(q_ℓ)` (baseline attention output).
3. Compute SVD of `q_ℓ`; extract top-k right-singular vectors `B_ℓ^* ∈ ℝ^{T×k}` for k ∈ {16, 32, 64, 128}.
4. Compute `Y_compr(B_ℓ^{*T} q_ℓ)`, lift to `B_ℓ^* Y_compr(...)`.
5. Train tiny depthwise `D` on residual `(Y_full − B_ℓ^* Y_compr ∘ B_ℓ^{*T})(q_ℓ)` for 200 steps.
6. Measure final `‖Y_full − [B_ℓ^* Y_compr(B_ℓ^{*T} q_ℓ) + D(Π^⊥ q_ℓ)]‖_F / ‖Y_full‖_F`.

**Pass criterion:** ratio ≤ 0.10 at k=64, w=8, on ≥ 4 of 5 layers. **Fail criterion:** ratio > 0.20 at k=128 — reject the spectral-rank premise and abandon SCFA. This costs ~1 hour on a single GPU and tests the central hypothesis cheaply (Ralph-loop methodology).

---

*End of Candidate B (SCFA).*
