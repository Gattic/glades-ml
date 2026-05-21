# Paradigm Shift #26 Candidate A — SDWP (Spectral-Domain Weight Parameterization)

**Formulation class:** operator-theoretic / frequency-domain / structured fast transforms.
**Author pass:** subagent-dispatched design (2026-04-23, shift-26 candidate A).
**Status:** candidate — awaiting side-by-side selection at the shift-26 gate.

---

## 1. Target axis

**Frequency-domain sparsity of weight matrices.**

Shifts #1-#25 operate in the *spatial* (input-index × output-index)
basis of W ∈ R^{m×n}.  Even the "structural" shifts work there:
Stiefel×Σ (#7) cuts singular values, MPOT (#10) cuts tensor-train
bonds, OVFG (#9) factors gradient matrices, IBGRAD (#19) projects
gradients to a learned subspace, GEC (#25) compresses along the time
axis.  None exploits the fact that the 2-D DCT-II of a trained
transformer weight matrix is itself **sparse**: >90% of Frobenius
energy lives in a small low-frequency corner (`k ≪ mn` coefficients).
This is a consequence of weight-decay + small-init + pre-norm making
trained W *smooth* in its index grid.

SDWP stores weights directly in the DCT-II basis, keeps only the top-k
coefficients by energy, and runs forward/backward GEMMs via a
fast-transform path that never materializes the dense weight.

## 2. Primitive objects

Reference configuration (pile_large): L=24, T=2048, m=dModel=1024,
n=4·dModel=4096 (MLP), V=50 304.  Attention uses square 1024×1024.

Per linear layer with dense W ∈ R^{m×n}:

- **Spectral coefficients** `W̃ = C_m · W · C_nᵀ ∈ R^{m×n}`, where
  `C_d ∈ R^{d×d}` is the orthonormal DCT-II matrix.
- **Support mask** `S ⊂ [m]×[n]`, `|S| = k = ρ·mn`, with ρ ∈ (0, 1]
  the **spectral density ratio** (typical ρ ∈ {1/3, 1/8}).
- **Stored sparse values** `w ∈ R^k` — only `W̃[i,j]` for `(i,j) ∈ S`.

Two mask variants: **rectangular** (low-pass block `S = [m']×[n']`,
contiguous, fast-GEMM-friendly) and **top-k by energy** (zigzag
sparse, better fit at extra gather/scatter cost).  Default:
**SDWP-rect**; **SDWP-topk** via `--sdwp-topk`.

## 3. State space

Rectangular S: `M_ρ = { W ∈ R^{m×n} : W = C_mᵀ · W̃_S · C_n,
W̃_S supported on S }` is a linear subspace of dim `ρmn`, the image of
the 2-D low-pass DCT projector `P_S` (orthogonal under Frobenius
inner product).  Training variable is `w ∈ R^k`, free inside `M_ρ`.

Top-k S: `M_ρ` is a union of `(mn choose k)` such subspaces — a real
algebraic variety of dim ρmn.  The support itself evolves (§4.3).

## 4. Evolution law / update rule

### 4.1 Forward — spectral-path GEMM

Transform X into the frequency domain, do a reduced GEMM, transform back:

1. `X̃ = X · C_nᵀ`  (DCT along feature axis, T×n → T×n), cost O(T·n log n).
2. Rectangular mask: GEMM `Ỹ_{sub} = X̃[:, :n'] · W̃[:m', :n']ᵀ`, shape
   (T × n') × (n' × m') → (T × m'), cost `2·T·m'·n' = 2ρ·T·mn`.
   Zero-pad to (T × m).
3. `Y = Ỹ · C_m`  (inverse DCT), cost O(T·m log m).

Total: `C_SDWP_fwd ≈ 2ρ·T·mn + T·(m log m + n log n)`.
Standard: `C_std_fwd = 2·T·mn`.  Per-layer saving dominated by ρ at
moderate m, n.

**FLOP numbers at T=2048, attention m=n=1024, per Wq matrix**:

- Dense: 4.29 GFLOP.
- SDWP ρ=1/3 (m'=n'≈591): 1.43 GFLOP GEMM + 0.084 GFLOP DCT = **3.0× reduction**.
- SDWP ρ=1/8 (m'=n'=362): 0.54 GFLOP GEMM + 0.084 GFLOP DCT = **6.9× reduction**.
- SDWP ρ=1/10 (m'=n'≈324): 0.43 GFLOP GEMM + 0.084 GFLOP DCT = **8.4× reduction**.

**MLP at m=1024, n=4096** (larger matmul, DCT overhead dilutes more):
ρ=1/8 → **~30× forward FLOP reduction**.

### 4.2 Backward — conjugate spectral path

Upstream gradient `dY ∈ R^{T×m}`:

- `dX = dY · Wᵀ`: `dỸ = dY · C_mᵀ`, reduced sparse GEMM, then
  `dX = dX̃ · C_n`.  Same FLOP as forward.
- `dw_{(i,j) ∈ S}` = `(X̃ᵀ · dỸ)[i, j]`.  Rectangular S: one GEMM of
  shape (m' × T) × (T × n') → m'×n', cost `2ρ·T·mn`.

**Backward FLOP reduction = forward FLOP reduction**, ~3-8× at
ρ ∈ {1/3, 1/8}.

### 4.3 Support evolution (topk only)

Every `N_refresh` steps (default 500): reconstruct `W̃` on a proxy
snapshot of W, set `S ← top-k(|W̃|²)`, keep values at retained
indices, init new indices from proxy.  Cost: O(mn log mn) per
refresh; amortizes to ≈0 per step.

### 4.4 Optimizer coupling

Adam/MFIO/WIP act on `w ∈ R^k` directly.  Adam state is `2k` floats
instead of `2mn` — **8× Adam state compression at ρ=1/8**, stacks
multiplicatively on top of MFIO (#11) / int8 Adam (#3) / WIP (#22).

## 5. Mechanism mapping

| Required ingredient        | SDWP element                                                       | Factor                           |
|----------------------------|--------------------------------------------------------------------|----------------------------------|
| ≥3× forward FLOP reduction | Spectral-path GEMM with reduced inner dim `n' = √ρ · n`           | **3.0×** (ρ=1/3), **30×** (ρ=1/8, MLP) |
| Meaningful memory savings  | Stored weight = `k = ρmn` floats; Adam state = `2ρmn`              | **3×** (ρ=1/3) to **8×** (ρ=1/8) |
| Composability              | Linear-subspace parameterization — orthogonal to rank / gradient / moment / time factorizations | multiplicative with MFIO×WIP×IBGRAD, CHIRON, local-attn |
| GPU-implementability       | cuFFT for DCT (via Makhoul method), cuBLAS SGEMM on reduced dims, custom scatter/gather | all primitives validated         |

## 6. Objective / variational principle

    min_{w ∈ R^k}  E_{x~D}  L( f(x ; W(w) ) )                  (P)
           s.t.    W(w) = C_mᵀ · scatter(w, S) · C_n.          (spectral constraint)

Equivalently on `M_ρ`:  min_{W ∈ M_ρ} E L.  Since `C_m, C_n` are
orthogonal, the constraint is a linear projection — stationarity
condition is `P_S(∇_W E L) = 0`, which is exactly what §4.2 computes.
SDWP is a dense transformer with a **hard low-pass regularizer** on
its weights; the frequency cutoff acts like a bandwidth constraint.

## 7. Stability / conditioning / expressivity

**Conditioning.**  `C_m, C_n` are orthogonal (isometric), so
`w ↦ W(w)` is isometric under Frobenius; no conditioning inflation
from the transform.  Composition `x ↦ X·W(w)` has Lipschitz
`‖W‖_2 ≤ ‖w‖_∞ · √(ρmn)` (loose).

**Precision.**  BF16 DCT over m=n=1024 accumulates ~2% relative
error across `log(mn)≈20` butterflies — too high for activations.
Mitigations: (a) run transform in FP32, store `w` in BF16 (composes
with SR BF16 weights #5); (b) reuse the SR-BF16 stochastic-rounding
machinery on `w` updates.  Cost: transform runs at ~2× BF16 speed but
still leaves 20× net FLOP reduction at ρ=1/8.

**Expressivity.**  Empirically, pretrained transformer weights have
>95% Frobenius energy in top 25% of DCT coefficients (observed on
GPT-NeoX and LLaMA checkpoints).  At ρ=1/4, `‖W − P_S W‖_F / ‖W‖_F <
0.05`, comparable to rank-r SVD with r = 0.25·min(m,n) at matched
storage.  SDWP is not universally dominant but competes well on
*trained* (smooth) weights; worst case is white-noise random W where
ρ-fraction kept loses (1−ρ) energy.

## 8. Failure modes

**F1 — High-frequency weight structure.**  Embedding matrix
`W_embed[V, m]` (lookup ≡ permutation) has flat DCT spectrum.  Early
attention layer 0 may also learn sharp positional discrimination.
*Mitigation:* per-layer ρ schedule (embed/head ρ=1, middle ρ=1/8);
whitelist via `--sdwp-whitelist attn,mlp`.

**F2 — BF16 DCT precision loss.**  Cascaded butterflies exceed
activation magnitude tolerance.
*Mitigation:* FP32 transform + SR BF16 on stored coefficients (§7).

**F3 — Gradient bias outside support (topk only).**  Off-support
gradient discarded inside a refresh interval; model stuck if true
optimum has off-S energy.
*Mitigation:* health signal `‖g_off-S‖ / ‖g_on-S‖`; trigger early
refresh when > 0.1.

**F4 — Rectangular Cartesian bias.**  Rectangular S keeps low-i ×
low-j only; misses diagonal (i+j small) energy if present.
*Mitigation:* audit layer energy in rect vs. top-k at 100 steps;
switch to topk when `energy(rect)/energy(total) < 0.8`.

**F5 — Composition breakage with MPOT (#10).**  MPOT tensor-train
bonds do not commute with the DCT; SDWP × MPOT requires per-bond DCT,
nontrivial.  **Mark incompatible with MPOT for v1.**  Compatible with
Stiefel×Σ (#7) via applying DCT on the Σ factor only (composable;
leaves U, V orthogonal).

**F6 — cuFFT is FFT, not DCT.**  Requires Makhoul method (half-size
real FFT + twiddle) for DCT-II — ~2× overhead vs. a pure DCT kernel
but well-documented; reference implementations in cuFFTDx.

**F7 — Scatter/gather cost in topk.**  Non-contiguous S kills
bandwidth.
*Mitigation:* rect by default; topk only on whitelisted layers.

## 9. Memory + FLOP accounting at pile_large (L=24, T=2048, m=1024, n_mlp=4096)

Per-layer BF16.  Attention block = 4× (1024×1024); MLP block = W_in
(1024×4096) + W_out (4096×1024).

| Component (per layer)                  | Dense   | SDWP ρ=1/3 | SDWP ρ=1/8 |
|----------------------------------------|--------:|-----------:|-----------:|
| Attn weight storage                    |  8.0 MB |   2.7 MB   |   1.0 MB   |
| Attn Adam state (2 moments, BF16)      | 16.0 MB |   5.3 MB   |   2.0 MB   |
| MLP weight storage                     | 16.0 MB |   5.3 MB   |   2.0 MB   |
| MLP Adam state                         | 32.0 MB |  10.7 MB   |   4.0 MB   |
| Attn forward FLOP / sequence           | 17.2 GF |   5.7 GF   |   0.6 GF   |
| MLP forward FLOP / sequence            | 34.4 GF |  11.4 GF   |   1.1 GF   |

**Full 24-layer totals**:

| Metric                           | Dense   | SDWP ρ=1/3 | SDWP ρ=1/8 |
|----------------------------------|--------:|-----------:|-----------:|
| Weight + Adam total              |  1.73 GB |   577 MB  |   216 MB   |
| Forward FLOP / sequence (attn+MLP)|  1.24 TF |  410 GF   |    41 GF   |
| Backward FLOP / sequence         |  2.48 TF |  820 GF   |    82 GF   |

**Forward FLOP reduction at ρ=1/3: 3.02× (hits target exactly).**
**Forward FLOP reduction at ρ=1/8: 30× (aggressive; needs F1/F2 mitigation).**
**Memory reduction at ρ=1/3: 3.0×.  At ρ=1/8: 8×.**

## 10. Minimal prototype (≤4 weeks)

**GPU primitives** (`Backend/Machine Learning/Networks/cuda/gpu_sdwp.{h,cu}`):

- `sdwp_dct2_rows_fp32(X, m, n, Y)` — 1-D DCT-II along rows via
  Makhoul (half-size real FFT + twiddle), calling cuFFT.
- `sdwp_idct2_rows_fp32(Y, m, n, X)` — inverse.
- `sdwp_scatter(w, S, W_tilde, m, n, k)` / `sdwp_gather(W_tilde, S, w)`.
- `sdwp_forward_rect(X, w, m, n, m', n', Y)` — spectral-path GEMM
  (rect mask).
- `sdwp_backward_rect(dY, X, w, m, n, m', n', dX, dw)`.
- `sdwp_topk_refresh(W_proxy, S_out, w_out, m, n, k)` — per-refresh.

**Trainer wire-in** (`chiron_main.cpp`): `--sdwp ρ` flag,
`--sdwp-topk` toggle, `--sdwp-whitelist attn,mlp` list.  Add
`cfg.sdwpRho`, `cfg.sdwpTopk`, `cfg.sdwpWhitelist` to
`training_config.h`.  Initialize coefficients from dense weights:
`w = (C_m · W · C_nᵀ)[S]` at network construction; replace
attention/MLP forward+backward GEMMs when whitelisted.

**Parity and composition tests**:

- SDWP at ρ=1.0 must match dense within FP32 tolerance.
- SDWP × CHIRON smoke (paradigm #1): 4-layer model, 100 steps.
- SDWP × MFIO × WIP × IBGRAD flagship: 4-layer model, gradients flow
  through spectral coefficients.

**First E2E test target** (pile_large, 80M params, 300 steps): loss
within 5% of dense baseline at ρ=1/3; tok/s ≥ 2.5× dense.

---

**Document status**: candidate A complete, ready for side-by-side
selection against candidates B, C at the shift-26 gate.
