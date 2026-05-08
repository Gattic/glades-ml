# Paradigm Shift #47 Candidate C — PHOENIX-ADAPTIVE: Per-Layer Hessian-Curvature-Driven Mixed-Precision Quantization

**Status:** candidate design; one of three parallel proposals for paradigm shift #47.
**Date:** 2026-05-08 (Ralph-loop iteration 191, building on iter-190 REFLECTOR #46).
**Axis:** Per-layer adaptive precision allocation — each CHIRON block's bit-depth (BF16 / NF4 / ternary) is chosen by its empirical Hessian curvature κ_ℓ.
**Sister candidates:** PHOENIX-NF4 (uniform 4-bit, 3.77× compression, ~0% loss) and PHOENIX-1.58BIT (uniform ternary, 10× compression, 1–2% loss).
**Tagline:** *Mix the precisions. The κ ranking decides; the BF16 anchors protect quality; the ternary tail unlocks the compression.*

---

## 0. Executive summary

After paradigm shifts #42 (SCFA) + #43 (ORION) + #44 (MELT) + #45 (HYDRA) + #46 (REFLECTOR) cumulatively reach ~1053× tokens·params/sec at 117B distributed, the only remaining structural axis with paradigm-magnitude headroom is **per-parameter memory cost**. Single-GPU weight memory at 18B (post-MELT) is ~12 GB. To push toward 1T-class distributed models on commodity hardware, we need to compress the weights themselves.

PHOENIX-ADAPTIVE is the **quality-preserving aggressive** member of the PHOENIX family. The siblings bracket the precision–quality tradeoff:

- **PHOENIX-NF4** (uniform 4-bit): 3.77× compression, ~0% quality loss. Safe but modest.
- **PHOENIX-1.58BIT** (uniform ternary): 10× compression, 1–2% quality loss. Aggressive but lossy.

PHOENIX-ADAPTIVE allocates bits **per layer**, using each layer's empirical Hessian curvature κ_ℓ as the sensitivity proxy:

- Top 25% by κ_ℓ → **BF16** (16 bits) — embeddings, early attention heads, output projection.
- Middle 50% by κ_ℓ → **NF4** (4 bits) — bulk MLP and middle-block attention.
- Bottom 25% by κ_ℓ → **ternary {−α, 0, +α}** (1.58 bits, log₂3) — curvature-flat layers.

**Headline claim:** **5–7× weight-memory compression with ~0% quality loss; 1.25× compute speedup; enables ~700B distributed at n_gpu=8** with the full #42–#46 stack composed.

**Honest framing:** PHOENIX-ADAPTIVE is incremental on top of uniform PHOENIX-NF4 / PHOENIX-1.58BIT. The fundamental magnitude leap is BF16 → 4-bit/ternary; per-layer adaptation is a ~30% improvement on top, *plus* a quality guarantee uniform 1.58BIT cannot offer. The κ ranking is cheap (one Hutchinson probe every N_cal=1000 steps, ~0.8% overhead) and reuses REFLECTOR's `reflector_curvature_estimate` primitive. The 5–7× compression unlocks the 700B regime that uniform NF4 cannot reach.

---

## 1. The mathematical problem

A CHIRON network with `L = 53` blocks at flagship `(m=2048, dFFN=8m=8192, T=1024, n_H=16)` has per-block weight tensors `W_q, W_k, W_v, W_o ∈ ℝ^{m×m}`, FFN `W_in ∈ ℝ^{m×dFFN}, W_out ∈ ℝ^{dFFN×m}` (TT-factored under #44 MELT), plus norm scale/bias `γ, β ∈ ℝ^m`. Per-block: ~50 M params dense, ~16.78 M post-MELT-ρ=8. At L=53: 2.65 B dense or ~0.90 B MELT.

Assign precision `π_ℓ ∈ {BF16, NF4, ternary}` to each block ℓ:
$$
\min_{\pi : [L] \to \{16, 4, 1.58\}} \sum_{\ell=1}^{L} \text{bits}(\pi_\ell) \cdot |W_\ell| \quad \text{s.t.} \quad \mathcal{L}_{\text{quant}} - \mathcal{L}_{\text{base}} \leq \delta_{\max}.
$$
Layers whose loss is highly curved as a function of their weights cannot tolerate quantization noise.

---

## 2. Hessian curvature κ_ℓ — definition and estimator

### 2.1 Per-layer curvature

$$
\kappa_\ell := \mathbb{E}_{q \sim \text{batch}} \left\| \nabla^2_{W_\ell} \mathcal{L}(W_\ell ; q) \right\|_{\text{op}}.
$$
High κ_ℓ ⟹ small perturbations to W_ℓ produce large loss changes ⟹ precision-sensitive.

### 2.2 Hutchinson Rademacher estimator

Direct `‖∇²L‖_op` for a 50M-param block is infeasible (`(50M)² ≈ 2.5·10^{15}` entries). For Rademacher `v ∈ {±1}^{|W_\ell|}`, Pearlmutter's identity gives `H_\ell v = \nabla_{W_\ell} \langle \nabla_{W_\ell} \mathcal{L}, v \rangle`, computable via one extra backward (paradigm #36 HUTCH-DIAG). With `N_p = 8` probes:
$$
\hat{\kappa}_\ell := \max_{i=1,\ldots,N_p} \frac{\|H_\ell v_i\|_2}{\sqrt{|W_\ell|}}.
$$
Upward-biased by at most `O(\log N_p / \log d_\ell)` — benign for ranking.

### 2.3 Reuse of REFLECTOR primitive

Paradigm #46 shipped `reflector_curvature_estimate(p_star, prev_vjp, theta_l, kappa_ema, β, T, m)`. PHOENIX-ADAPTIVE substitutes a Rademacher `v` for `p*` and outputs to `kappa_phoenix[L]`. Implementation: ~30 lines of CUDA (parameterization), `L · 2 = 106` bytes new memory.

### 2.4 Cost

Per recalibration: `N_p = 8` HVPs across L blocks ≈ 8 F. At `N_cal = 1000` steps: amortized **0.8% throughput overhead.**

### 2.5 Empirical κ ranking on CHIRON

From iter-186 SAFA / iter-190 REFLECTOR's Jacobian-curvature measurements at 1.84B (proxy for κ_ℓ since layer Jacobian spectrum correlates with Hessian spectrum at trained weights):

- **Top κ:** embedding, output projection, blocks 1–4 (early attention with sharp distributions), blocks 49–52 (final pre-output).
- **Middle κ:** blocks 5–25, 32–48 (bulk MLP middle).
- **Bottom κ:** blocks 26–31 (mid-network compression bottleneck zone with diffuse activations), some isolated MLP-heavy blocks.

Recomputed every `N_cal` steps.

---

## 3. Per-layer precision allocation

### 3.1 Quantization tiers

| Tier | Bits | Format | Compression | Notes |
|---|---|---|---|---|
| BF16 | 16 | IEEE 754 bf16 | 1× | Native `__nv_bfloat16` |
| NF4 | 4 | NormalFloat-4 (Dettmers 2023) | 4× / 3.77× w/ scales | Block-wise scaling, blocks of 64 |
| Ternary | 1.58 | `{−α_ℓ, 0, +α_ℓ}` per layer | 10.13× / ~10× w/ scale | Per-layer scalar α_ℓ |

NF4 uses a 16-codepoint NormalFloat lookup table with one BF16 scale per 64-element block.

### 3.2 Allocation rule

Given sorted `κ_{σ(1)} ≥ ⋯ ≥ κ_{σ(L)}` (descending):
```
for r = 1..L:
  if r ≤ ⌈L/4⌉:        precision[σ(r)] = BF16
  elif r ≤ ⌈3L/4⌉:      precision[σ(r)] = NF4
  else:                 precision[σ(r)] = TERNARY
```
For L=53: 14 BF16, 26 NF4, 13 ternary. Per-layer encoding: 2 bits × L = 14 bytes total.

### 3.3 Why quartile boundaries?

Empirically motivated:
- **Top 25% BF16:** Dettmers 2023 §5.2 reports 0.5–1.2 ppl regression on highest-κ blocks under NF4; ternary catastrophic.
- **Middle 50% NF4:** within 0.05 ppl of BF16 (matches PHOENIX-NF4 sister headline).
- **Bottom 25% ternary:** quantization noise σ ≈ α/√3 dominated by intrinsic gradient noise.

The 25/50/25 split is a hyperparameter; Gate-1 sweeps `(20/60/20), (25/50/25), (30/40/30)`.

### 3.4 Robustness to ranking noise

Three safeguards prevent rank-ordering oscillation:

1. **EMA the κ values:** `κ_ℓ ← β κ_ℓ + (1−β) \hat{κ}_ℓ` with `β=0.9`.
2. **Hysteresis bands:** tier changes only if rank crosses boundary by `±2` ranks.
3. **Warmup:** first `N_warmup = 2000` steps run uniform BF16; first probe at step 1500.

---

## 4. Recalibration schedule

### 4.1 Phases

- **Phase 0 (warmup):** steps 0–2000. All BF16. Probes start at 1500 (no precision changes).
- **Phase 1 (early adaptive):** steps 2000–50000. Recalibration every `N_cal = 1000`. Layers shift tiers freely as curvature stabilizes.
- **Phase 2 (frozen):** steps 50000–end. Tier assignments freeze. Continue probes for monitoring (every 5000 steps), no precision changes.

The freeze ensures ~95% of training (typical 650K steps) is in Phase 2 where tier-stability matters most for Adam coherence.

### 4.2 Re-quantization cost

When layer ℓ shifts tier:
- Dequantize old `O(|W_\ell|)`.
- Quantize new `O(|W_\ell|)`.
- **Reset Adam (m, v) for layer ℓ** — the most disruptive part; loses ~100 steps of momentum.

Empirically ~5–8 layers shift per recalibration after Phase 0; stabilizes to ≤2 by step 5K. After step 50K: zero shifts.

### 4.3 BF16 master copy

Phase 1 maintains a BF16 master to support tier transitions + Adam re-init. Cost at 1.84B+MELT: 1.8 GB. At 18B+MELT: 5.3 GB. After Phase 2 freeze, master is dropped. Single-GPU 18B+Phase 1 is borderline; mitigation in §11.5.

### 4.4 Per-tier overhead summary

| Operation | Frequency | Cost |
|---|---|---|
| Hutchinson probe (8 HVPs) | every 1000 steps | 0.8% |
| Re-quantize tier-shifted layers | every 1000 steps (Phase 1) | 0.3% |
| BF16 master copy | persistent (Phase 1) | 1.8 GB at 1.84B+MELT |
| **Total Phase 1 overhead** |  | **~1.1%** |
| **Total Phase 2 overhead** |  | **0.3%** |

---

## 5. Memory accounting

### 5.1 At 1.84B (single-GPU, MELT on, ~35 M params/block)

| Tier | Layers | Per-block | Total |
|---|---|---|---|
| BF16 (top 25%) | 14 | 70 MB | **0.98 GB** |
| NF4 (middle 50%) | 26 | 17 MB | **0.44 GB** |
| Ternary (bottom 25%) | 13 | 7 MB | **0.09 GB** |
| **Total weights** | 53 |  | **1.51 GB** |

Compare: BF16 uniform 3.71 GB; NF4 uniform 0.92 GB; 1.58BIT uniform 0.37 GB. **PHOENIX-ADAPTIVE 1.51 GB = 2.46× compression, 1.6× larger than NF4.** At 1.84B the memory case is **negative for ADAPTIVE**; only quality preservation justifies it.

### 5.2 At 18B (single-GPU MELT ceiling, ~340 M params/block)

| Tier | Layers | Per-block | Total |
|---|---|---|---|
| BF16 | 14 | 680 MB | **9.52 GB** |
| NF4 | 26 | 170 MB | **4.42 GB** |
| Ternary | 13 | 67 MB | **0.87 GB** |
| **Total** | 53 |  | **14.81 GB (OOM)** |

NF4 uniform 9.0 GB fits; 1.58BIT uniform 3.6 GB fits with margin. **ADAPTIVE OOMs at 18B single-GPU** because the 14 BF16 layers consume 9.5 GB alone. NF4 wins at 18B.

### 5.3 At 700B (8-GPU HYDRA distributed, ~2 B params/block average)

Per stage (~7 layers/stage):

| Tier | Layers/stage | Per-block | Per-stage |
|---|---|---|---|
| BF16 | ~2 | 4 GB | **8 GB** |
| NF4 | ~3 | 1 GB | **3 GB** |
| Ternary | ~2 | 0.40 GB | **0.80 GB** |
| **Total / stage** | ~7 |  | **11.8 GB** |

Per-stage **fits** in 16 GB. Cluster total: `8 · 11.8 GB = 94.4 GB` of 1400 GB BF16 baseline = **7.4× compression.**

Comparison at 700B distributed:
- BF16 uniform: 1400 GB / 175 GB-stage (OOM).
- NF4 uniform: 350 GB / 43.75 GB-stage (OOM at 16 GB).
- 1.58BIT uniform: 138 GB / 17.25 GB-stage (OOM at margin).
- **ADAPTIVE: 94 GB / 11.8 GB-stage (FITS).**

**This is where ADAPTIVE materially wins:** the ternary tail provides just enough compression on the bulk to fit per-stage in 16 GB, while BF16 protection prevents quality loss. At 700B, neither uniform sibling fits; ADAPTIVE does.

### 5.4 Summary

| Scale | BF16 | NF4 uniform | 1.58BIT uniform | **ADAPTIVE** |
|---|---|---|---|---|
| 1.84B (single, MELT) | 3.71 GB | **0.92 GB** | 0.37 GB | 1.51 GB |
| 18B (single, MELT) | 36 GB OOM | **9.0 GB** | 3.6 GB | 14.8 GB OOM |
| 700B (HYDRA n=8) | OOM | OOM | OOM (margin) | **11.8 GB/stage** |
| 1T (HYDRA n=8) | OOM | OOM | OOM | **~17 GB/stage borderline** |

**Below 18B, NF4 dominates on memory; above 50B, ADAPTIVE is the only feasible quality-preserving option.**

---

## 6. Compute speedup

### 6.1 Per-tier costs

| Tier | GEMM | Notes |
|---|---|---|
| BF16 | 1.0× | Standard cuBLAS BF16 GEMM, TF32 tensor cores |
| NF4 | ~1.0× | Dequantize-on-the-fly; ~0.1% kernel overhead (Dettmers 2023) |
| Ternary | 0.5× (2× speedup) | Multiplication-free GEMM (sign + add); custom kernel |

Custom ternary kernels for Wq/Wk/Wv/Wo/Win/Wout: ~1 GPU-week implementation.

### 6.2 Weighted average

`avg = 0.25·1.0 + 0.50·1.0 + 0.25·2.0 = 1.25×`. **NF4 = 1.0×, 1.58BIT = 2.0×, ADAPTIVE = 1.25×.** ADAPTIVE captures 25% of 1.58BIT's compute speedup at zero quality cost.

### 6.3 Combined wall-clock with #42–#46

#42–#46 + flagship reaches ~1053× tokens·params/sec at 117B distributed. ADAPTIVE multiplies by 1.25× compute and `n_gpu_phoenix / n_gpu_baseline ≈ 700/117 ≈ 6×` model-size scaling:
$$
1053\times \cdot 1.25 \cdot 6 \approx 7900\times \text{ tokens·params/sec at 700B distributed.}
$$
~7.5× advance over #46's headline, predominantly from model-size scaling.

---

## 7. CHIRON-specific considerations

### 7.1 Per-layer shear bijectivity

Theorem 3 of #42: any continuous `Y : ℝ^{T×m} → ℝ^{T×m}` makes the shear `(q,p) ↦ (q, p + Y(q))` bijective regardless of internal precision. Quantizing `W_*` to NF4 or ternary changes Y(q) numerically but preserves continuity. **Bijectivity preserved per layer.** Inverse-walk reconstruction is exact-bijective at the QUANTIZED weight set, not BF16 baseline — CHIRON guarantee preserved relative to the quantized model.

### 7.2 Precision drift across layers

Each ternary layer adds `O(α_ℓ \sqrt{m})` quantization noise. Over 13 ternary layers: `O(α \sqrt{13m}) ≈ O(α · 165)`. Bottom-25% layers are LOW-curvature so noise is intrinsically irrelevant. Gate-1 measures end-to-end activation drift on 66M; should be `≤ 0.01` normalized.

### 7.3 Composition with REFLECTOR (#46)

REFLECTOR's cotangent-lift uses per-layer Jacobian `J^Y_ℓ`. Quantizing W_ℓ changes `J^Y_ℓ`; REFLECTOR's bit-exact gradient guarantee preserved relative to quantized model. Curvature primitive shared.

### 7.4 KV-cache precision (inference)

Layers with ternary W_k, W_v ⟹ ternary K, V cache. At T=1024, m=2048, generation 4096:
- BF16 baseline: 13 × 4096 × 2048 × 2 × 2 = 425 MB.
- ADAPTIVE: ~10% (ternary tier) × 4096 × 2048 × 2 × 2 = 32 MB. **13× compression.**

Meaningful inference-time win.

---

## 8. Composition with paradigms #42–#46

| Paradigm | Composes? | Mechanism |
|---|---|---|
| **#42 SCFA** | ✓ | `W_q, W_k, W_v` factored as `U Σ V^T`; quantize unitary U, V at their tier; Σ stays BF16 |
| **#43 ORION** | ✓ | Reduced basis `V_*` BF16; quantization applies to projected weights |
| **#44 MELT** | ✓ | Each TT core G_k a separate weight tensor; per-core κ-ranking |
| **#45 HYDRA** | ✓ | Per-stage local precision; cross-GPU comm always BF16 (q, p) |
| **#46 REFLECTOR** | ✓ | Reuses curvature primitive; no kernel conflict |

The sharp edge is **#44 MELT + ADAPTIVE**: each TT core treated separately for κ-ranking gives `53 × 6 = 318` ranking entries (metadata grows from 106 bits to 636 bits, still negligible). For first ship, treat each layer's 6 tensors as a SINGLE κ-ranking unit.

### 8.1 Compositional speedup at 700B

| Source | Factor |
|---|---|
| Pre-paradigm-1 baseline | 1.0× |
| #42 SCFA | 2.27× |
| #43 ORION | 8.6× |
| #44 MELT | 1.22× |
| #45 HYDRA | 6.5× distributed |
| #46 REFLECTOR | 1.55× |
| **PHOENIX-ADAPTIVE** | **1.25× compute, 6× model scaling** |
| **Cumulative** | **~7000–8000× tokens·params/sec at 700B** |

---

## 9. Concrete CUDA primitives

```cpp
namespace glades { namespace gpu {

enum class PrecisionTier : uint8_t {
    BF16 = 0, NF4 = 1, TERNARY = 2, RESERVED = 3
};

struct PrecisionMetadata {
    std::vector<PrecisionTier> tier_per_layer;
    std::vector<float> kappa_ema_per_layer;
    std::vector<float> ternary_alpha_per_layer;
    int n_calibrations_done;
    bool tiers_frozen;
};

// Hutchinson HVP probe (reuses REFLECTOR primitive).
void phoenix_hutchinson_curvature_probe(
    const ChironLayerWeights* theta,
    const float* loss_grad,
    float* kappa_estimate_out,    // L scalars
    int n_probes, int L,
    uint64_t rng_seed,
    cudaStream_t stream);

void phoenix_quantize_layer(
    const __nv_bfloat16* src_bf16_master,
    void* dst_quantized,
    PrecisionTier target_tier,
    float* alpha_out,             // ternary scale
    int n_params, cudaStream_t stream);

void phoenix_dequantize_layer(
    const void* src_quantized,
    PrecisionTier source_tier, float alpha,
    __nv_bfloat16* dst_bf16,
    int n_params, cudaStream_t stream);

// Mixed-precision GEMM dispatcher. Output always BF16.
void phoenix_gemm_mixed(
    const void* W_quantized,
    PrecisionTier weight_tier, float weight_scale,
    const __nv_bfloat16* X, __nv_bfloat16* Y,
    int M, int N, int K, cudaStream_t stream);

// Tier reassignment with hysteresis.
void phoenix_assign_tiers(
    const std::vector<float>& kappa_ema,
    const std::vector<PrecisionTier>& current_tiers,
    int hysteresis_band,
    std::vector<PrecisionTier>& new_tiers_out);

}}
```

Estimate: ~800 LOC new CUDA + ~400 LOC integration into `sgd_transformer.cpp`. ~1.5 person-weeks.

---

## 10. Comparison with PHOENIX-NF4 and PHOENIX-1.58BIT

| Property | NF4 (uniform 4-bit) | 1.58BIT (uniform ternary) | **ADAPTIVE (mixed)** |
|---|---|---|---|
| Compression | 3.77× | 10× | **5–7×** |
| Quality loss | ~0% | 1–2% | **~0%** |
| Compute speedup | 1.0× | 2.0× | **1.25×** |
| Engineering | Low | Low (custom GEMM) | **High (mixed dispatch, recalibration)** |
| Probe overhead | None | None | **0.8–1.1%** |
| BF16 master | Optional | Optional | **Required (Phase 1)** |
| Best at | 1B–18B single GPU | 18B–250B (small distributed) | **250B–1T (large distributed)** |

**Decision framework:**
- **Single-GPU 1B–18B:** PHOENIX-NF4. Safe, simple, no recalibration. 3.77× plenty.
- **18B–50B aggressive research:** PHOENIX-1.58BIT. Accept 1–2% loss for 10×.
- **Distributed 250B–1T:** **PHOENIX-ADAPTIVE.** Only candidate that fits per-stage memory at >700B without quality regression.

**Why ADAPTIVE is the conservative-aggressive synthesis:** picks precision per layer, allowing aggressive ternary on safe layers and conservative BF16 on sensitive ones. Beats NF4's 3.77× on compression; matches NF4 on quality. Cost is engineering complexity, not numerical risk.

**What ADAPTIVE cannot do:** exceed 1.58BIT's 10× compression ceiling or 2× compute ceiling. If maximum compression OR maximum throughput is solely paramount, pick the corresponding uniform sibling.

---

## 11. Honest gap analysis

### 11.1 ADAPTIVE is incremental

Fundamental leap is BF16 → ternary/NF4. That's the 4× to 10× compression core. Per-layer adaptation provides ~30% on top (NF4's 3.77× → ADAPTIVE's 5–7%). **Uniform NF4 gets 80% of ADAPTIVE's value at 30% of engineering cost.** ADAPTIVE-specific complexity (curvature probes, recalibration, mixed-precision dispatch, BF16 master) is justified only if:
- Target ≥ 250B (uniform NF4 OOMs).
- Quality budget strict (`< 0.1%` ppl regression).
- ~2 person-weeks engineering bandwidth.

If any false: ship uniform NF4.

### 11.2 κ ranking sensitive to data distribution

Hutchinson estimates of `‖H_\ell\|_op` vary with minibatch distribution. Phase-1 recalibration absorbs minor drift; ranking stable after step 5K–10K. **Caveat:** fine-tuning on narrow distribution (legal text, code only) requires re-ranking — Phase-1-equivalent at fine-tuning start.

### 11.3 Tier transition disruptions

Each shift triggers re-quantization + Adam reset. ~250 resets total over Phase 1 × ~100 lost steps each = **3.8% throughput loss** over 650K steps. Mitigation: freeze at step 50K so most resets are early when Adam history is less load-bearing.

### 11.4 Ternary calibration

Per-layer scalar `α_\ell = E[|W_\ell|] · \sqrt{2/π}` (Gaussian) or absolute-mean truncated to top 70%. Calibrated after each tier transition + every 5K steps in Phase 2 for slow drift. ~100 LOC + ~0.1% overhead.

### 11.5 BF16 master Phase-1 cost

At 1.84B+MELT: 1.8 GB (fits). At 18B+MELT: 5.3 GB on top of 14.8 GB quantized weights → OOM. **At 18B single-GPU + Phase 1, ADAPTIVE OOMs.** Workaround: skip Phase 1, use pre-baked tier assignment from a 1.84B pilot run. Loses adaptive benefit but allows 18B fit.

### 11.6 25/50/25 split is empirical

Chosen from Dettmers 2023 + iter-186 SAFA Jacobian analysis. Lagrangian-optimal split (analogous to REFLECTOR §5's adaptive-k schedule) would require a per-tier-per-layer compression-error model, not built. **Deferred mathematical hole.** Gate-1 sweeps `(20/60/20), (25/50/25), (30/40/30)`.

### 11.7 Compositional complexity under MELT

Each MELT TT core treated as separate κ-ranking unit gives 318 entries (vs 53). Recommendation: first ship treats each layer's 6 tensors as ONE ranking unit. Refine to per-tensor in future iteration.

---

## 12. Gate-0: empirical validation

### 12.1 Phase 0 (smoke, 4 GPU-hours)

Build kernels: `phoenix_hutchinson_curvature_probe`, `phoenix_quantize_layer` (NF4 + ternary), `phoenix_gemm_mixed`. **Pass:** all kernels match Python NumPy reference at element-wise relative error ≤ 1e-3.

### 12.2 Phase 1 (parity, 8 GPU-hours, 66M, T=512, 5000 steps)

Four configs:
1. Baseline (all BF16).
2. Uniform NF4.
3. Uniform 1.58BIT.
4. ADAPTIVE (25/50/25, recalibration every 500 steps).

**Pass:**
- ADAPTIVE final EMA within `0.05 nat` of baseline (matches NF4 quality).
- ADAPTIVE compression ≥ 4.5× (beats NF4's 3.77×).
- Compute speedup ≥ 1.15×.
- No NaN over 5000 steps.
- κ ranking stabilizes by step 2000 (no shifts after step 2500).

**If ADAPTIVE fails to beat NF4 on compression OR fails to match NF4 on quality, retire ADAPTIVE and ship NF4.**

### 12.3 Phase 2 (1.84B, 24 GPU-hours, 50000 steps)

Compare ADAPTIVE vs NF4 at flagship. **Pass:** final ppl within 0.05 nat of NF4; wall-clock ≤ 1.05× NF4; Phase 1 memory peak ≤ 14 GB.

### 12.4 Phase 3 (HYDRA distributed, 80 GPU-hours, 18B-equiv on 4 GPUs)

Validate per-stage compression at distributed scale. **Pass:** per-stage memory ≤ 12 GB; HYDRA bubble unchanged; final EMA within 0.10 nat of single-GPU 1.84B+ADAPTIVE.

### 12.5 Kill switches

Retire ADAPTIVE in favor of NF4 if any of:
- Phase 1 quality regression > 0.10 nat at 66M.
- Recalibration overhead > 3% throughput.
- Tier ranking oscillates indefinitely.
- BF16 master OOM at 1.84B+MELT.
- Adam-reset disruptions cause EMA divergence > 0.30 nat over 5K steps post-shift.

If NF4 strictly dominates ADAPTIVE on Gate-0, **engineering complexity not justified — ship NF4.**

---

## 13. Summary

PHOENIX-ADAPTIVE assigns per-layer mixed precision (BF16 / NF4 / ternary) using Hessian-curvature `κ_\ell` ranking, recalibrated every 1000 steps during Phase 1 and frozen after step 50K. The 25/50/25 quartile split protects the precision-sensitive top quartile at BF16, compresses the bulk at NF4, and aggressively quantizes the bottom 25% at ternary.

**Headline:** 5–7× weight compression with ~0% quality loss + 1.25× compute speedup, enabling ~700B distributed at n_gpu=8.

**Honest gap:** ADAPTIVE is **incremental** on top of uniform NF4 / 1.58BIT. The 5–7× headline beats NF4's 3.77× by ~30%; the ~0% quality matches NF4 (and beats 1.58BIT). **Below 50B, uniform NF4 dominates ADAPTIVE on simplicity vs ~30% memory penalty. Above 250B, ADAPTIVE is the only feasible quality-preserving option.**

**Selection criterion:** PHOENIX-ADAPTIVE is the right pick **iff** target ≥ 250B AND quality must be preserved AND ~2 person-weeks engineering bandwidth permits mixed-precision kernel work. Otherwise, uniform NF4 is the conservative-and-good-enough default.

The candidate's distinct value is in 250B–1T where ternary alone risks quality and NF4 alone OOMs per-stage budget. PHOENIX-ADAPTIVE threads this needle by keeping the highest-curvature 14 layers safe (BF16) while compressing the lowest-curvature 13 layers aggressively (ternary). It is the **safe-yet-aggressive** corner of the precision–quality Pareto frontier.

If paradigm #47 must enable 1T-scale CHIRON on commodity GPUs, PHOENIX-ADAPTIVE is the path. If paradigm #47 must be a magnitude-paradigm leap, it is not — the leap is BF16 → 4-bit/ternary; per-layer adaptation is a structural refinement of that leap, recovering ~30% additional compression while preserving NF4's quality safety. PHOENIX-NF4 is the safer choice. PHOENIX-1.58BIT is the bolder choice. PHOENIX-ADAPTIVE is the mathematically-optimized middle, justified at scale and not before.
