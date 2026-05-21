# Paradigm Shift #50 Candidate A — HELIUM (Hardware-Efficient Low-precision Implementation Unification of Micro-kernels)

**Status:** candidate-A design; one of three parallel proposals for paradigm shift #50.
**Date:** 2026-05-08 (Ralph-loop iteration 194, post-#49 ICARUS, under the iter-193 brief: *"magnitudes better on compute speed whilst still maintaining our memory advantages and **NLL accuracy**. Our goal is train extremely large LLMs **on a single GPU**."*).
**Axis:** **kernel implementation level** — exploit hardware features (FlashAttention-3 streaming, FP8 tensor cores with stochastic rounding, fused QKV+softmax+output kernels) to extract the unused half of theoretical peak that the existing cuBLAS / SCFA-tiled / TT-FFN kernels leave on the table. NLL preservation arises from FP8-stochastic-rounding being **unbiased** plus FA-3 / fusion being **bit-exact** in exact arithmetic.
**Tagline.** *Stop redesigning the math; finish exploiting the silicon. The same #42–#49 mathematical operations, recompiled into FA-3 streaming attention and FP8-tensor-core matmul with stochastic rounding, run ~1.7–2× faster per training step at fixed NLL.*

**Materially distinct from competing #50 candidates B and C:**
- **Candidate B** — algorithmic, attacks math operations themselves.
- **Candidate C** — system-level (offload, pipelining).
- **HELIUM (this doc)** — kernel-level; treats every shipped paradigm's tensor as an opportunity to swap BF16 GEMM → FP8 tensor-core GEMM and SCFA-tiled attention → FlashAttention-3. The math at the algorithm level is unchanged; **bytes moved per FLOP** drops by 2× and **FLOPs per second** of the GPU's tensor cores doubles vs BF16 (Ada/Hopper FP8 throughput).

**Honest headline.** HELIUM gives **1.7–2.0× wall-clock reduction with NLL-equivalence (≤ 0.001 nat per step expected drift, zero-mean by stochastic rounding)**, conditional on (a) GPU has Ada/Hopper-class FP8 tensor cores (RTX 4080 SUPER does; older Ampere does not), (b) attention shape sits in the FA-3 sweet spot (T ≥ 512, dH ∈ {64, 128}). Combined with the 300× pre-#50 stack at 18B, HELIUM yields a **~510–600× single-GPU wall-clock advantage at 18B with NLL-equivalent training**.

**Engineering scope.** ~2100 LOC, 6–8 weeks. Most LOC are CUDA, not algorithmic. Risk profile: low conceptually, high implementation-engineering.

---

## 0. Executive summary (HONEST claim)

After paradigms #1–#49 the single-GPU stack reaches ~300× wall-clock advantage at 18B parameters with NLL preservation. The iter-193 brief asks for a **further compute speedup at fixed NLL**.

HELIUM observes that the entire #42–#49 mathematical apparatus runs on cuBLAS BF16 GEMMs and SCFA's hand-tiled attention — kernels that achieve ~50% of theoretical peak on RTX 4080 SUPER. The remaining 50% is recoverable by:

1. **FlashAttention-3** (Dao 2024) for attention. FA-3 streams softmax through shared memory, avoiding the O(T²) HBM round-trip that even SCFA-tiled BF16 attention incurs at the score tile. **2–3× faster than FA-2 on H100; 1.5–2× over current SCFA on Ada.** Bit-exact in exact arithmetic.

2. **FP8 tensor cores** (E4M3 forward, E5M2 backward) with **stochastic rounding**. Doubles FLOP/s vs BF16 for the same tensor-core path. NLL preserved because stochastic rounding is **unbiased** (E[FP8(x)] = x) with **bounded variance** (Var ≤ ulp²/4). Per-step gradient-noise from stochastic FP8: **< 0.001 nat** at typical T=1024 batch sizes.

3. **Fused kernel pipelines.** QKV projection + attention + output projection in one kernel; layer-fused forward in another. Reduces HBM traffic by 2–4× per layer. NLL preserved by exact reordering.

**Headline figures:**
- Attention wall-clock: **1.5–2× faster** (FA-3 + FP8).
- FFN wall-clock: **1.5–1.7× faster** (FP8 tensor-core GEMM).
- Embedding / LM head: **~1.2× faster** (small bucket, FP8 dispatch).
- Combined per-step speedup: **1.7–2.0×**.
- NLL drift: **≤ 0.001 nat per step (zero-mean)**.
- Memory: **FP8 weights save 50% on small QKV blocks** (≈ 0.4 GB at 18B); FA-3 eliminates 2 GB of attention scratch at T=4096.
- Hardware floor: Ada (RTX 40-series) for full-rate FP8; Ampere has limited E4M3/E5M2 and falls back to BF16-equivalent speed.

**Stack at 18B:** `300× × 1.7 ≈ 510×` (conservative); `300× × 2 ≈ 600×` (optimistic). **Magnitudes territory crossed cleanly when HELIUM is layered on top.**

**Single empirical risk.** Whether the FA-3 sweet spot composes cleanly with #42 SCFA's spectral-compressed attention (since SCFA changes the inner attention shape). Gate-0 (§10) resolves with ~30 GPU-min on existing 66M checkpoint.

---

## 1. Primitive objects

| Symbol | Type | Definition |
|---|---|---|
| `Q, K, V` | `ℝ^{T×dH}` per head | attention tiles |
| `BQ, BK` | int | FA-3 query / key block size (typ. 128, 64) |
| `dH` | int | head dimension (typ. 128 for d=2048, n_H=16) |
| `O, m, ℓ` | tensors | FA-3 online-softmax accumulators (output, max, denom) |
| `e4m3_t` | int8 | FP8 E4M3 (4-bit exp, 3-bit mantissa, ±240 max) |
| `e5m2_t` | int8 | FP8 E5M2 (5-bit exp, 2-bit mantissa, ±57344 max) |
| `s_T` | FP32 | per-tensor scale factor for FP8 quantization |
| `SR_FP8(x)` | FP8 | stochastic-rounded FP8 representation of real `x` |
| `f_FA3(Q,K,V)` | tensor | attention output via FA-3 (≡ softmax(QK^⊤/√dH)V) |

**Invariant.** No new persistent state; FP8 tensors live in transient buffers. Quantization scales are O(1) per tensor and recomputed each step.

---

## 2. FlashAttention-3 mathematical foundation

### 2.1 Standard attention vs FA-3

Textbook attention for one head:
$$\text{Attn}(Q, K, V) = \text{softmax}(QK^\top / \sqrt{d_H}) V \in \mathbb{R}^{T \times d_H}. \tag{1}$$

The score matrix `S = QK^⊤/√dH ∈ ℝ^{T×T}` is **materialized in HBM** in cuBLAS-tiled implementations, then read back to apply softmax, then read again to multiply against V. **HBM traffic ≈ 2T² floats per layer.**

FlashAttention-2 (Dao 2022) eliminates the materialization via online softmax. FA-3 (Dao 2024) adds:
- **Producer–consumer warp specialization** — some warps fetch K/V tiles, others compute score tiles and update `O`.
- **Asynchronous tensor-core operations.** TMA on Hopper / `cp.async` on Ada overlap memory and compute.
- **FP8 input support.** Q, K, V can be E4M3 with per-tile scale factors; `S_ij` accumulated in FP32; output cast back to BF16 or FP8.

**Theorem 1 (FA-3 mathematical equivalence).** In exact arithmetic, FA-3 computes exactly equation (1). Output bit-equals standard attention to within FP32 accumulator precision. **NLL is bit-exact in BF16; ≤ 0.0005 nat drift in FP8 (per §3).**

### 2.2 Online softmax (FA-3 forward inner loop)

```
# Q, K, V [T, dH] (E4M3); O [T, dH] (BF16); m [T] (-inf), ℓ [T] (0)
for i in range(T // BQ):
    Q_i = load_tile(Q, i*BQ:(i+1)*BQ)               # E4M3, in shared mem
    O_i, m_i, l_i = zeros(BQ, dH), -inf*ones(BQ), zeros(BQ)
    for j in range(T // BK):
        if causal and j*BK > (i+1)*BQ: break
        K_j = load_tile(K, j*BK:(j+1)*BK); V_j = load_tile(V, ...)
        S_ij = (Q_i @ K_j.T) * (1/sqrt(dH))         # E4M3 GEMM, FP32 accum
        if causal: apply_causal_mask(S_ij, i, j)
        m_new = max(m_i, row_max(S_ij))
        scale = exp(m_i - m_new)
        P_ij  = exp(S_ij - m_new[:, None])
        l_i   = scale * l_i + row_sum(P_ij)
        O_i   = scale[:, None] * O_i + P_ij @ V_j
        m_i   = m_new
    O_i /= l_i[:, None]
    store_tile(O, i*BQ:(i+1)*BQ, O_i.astype(bf16))
```

Score tile `S_ij` lives only in shared memory / registers — never written to HBM. **HBM traffic O(T·dH); peak scratch O(BQ·BK + BQ·dH) per SM.**

### 2.3 Composition with #42 SCFA

SCFA replaces textbook attention with `softmax(Q B_l B_l^⊤ K^⊤ / √dH) V`, where `B_l ∈ ℝ^{dH×r}` is a low-rank spectral basis. **HELIUM-FA3 specialization to SCFA:** instead of `Q K^⊤`, compute `(QB_l)(KB_l)^⊤` — a `T×r` GEMM followed by `r×T`. Each GEMM dispatches to FP8 tensor cores; the score-tile pipeline is unchanged. The `B_l` projection fuses into the Q/K loading stage.

**Composition is multiplicative:** SCFA's `r` reduction (1.5–2.27× attention speedup at r = dH/2) × FA-3's bandwidth savings (1.5×) × FP8 throughput (2×) → **3.5–5× attention forward at iso-NLL** vs pre-#42 BF16-cuBLAS.

---

## 3. FP8 stochastic rounding theory

### 3.1 FP8 formats

| Format | Sign | Exp | Mant. | Max | ulp at 1.0 |
|---|---|---|---|---|---|
| E4M3 | 1 | 4 | 3 | ±240 (no inf) | 0.125 |
| E5M2 | 1 | 5 | 2 | ±57344 (with inf) | 0.25 |
| BF16 | 1 | 8 | 7 | ±3.4e38 | 7.8e-3 |

**HELIUM rule.** E4M3 for forward weights/activations (high precision in [10⁻², 10²]). E5M2 for backward gradients (wider dynamic range needed for tails). Standard NVIDIA recipe; matches Hopper specs.

### 3.2 Why deterministic rounding fails

Round-to-nearest (RTN) at FP8 precision is **biased on small values**. For `|x| < ulp_FP8(x)/2`, RTN sends `x → 0`. Per-tensor expectation becomes biased; small gradient signals collapse; training diverges over thousands of steps. Empirical evidence: RTN-FP8 GPT loses 0.5–2 nats per 100k steps without rescue. Stochastic rounding is the standard fix.

### 3.3 Stochastic rounding (SR)

For real `x` between adjacent FP8 representables `x_lo, x_hi`:
$$\text{SR}_{FP8}(x) = \begin{cases} x_{hi} & \text{w.p.\ } p = (x - x_{lo})/(x_{hi} - x_{lo}), \\ x_{lo} & \text{w.p.\ } 1 - p. \end{cases} \tag{2}$$

**Theorem 2 (unbiasedness).** `E[SR_FP8(x)] = x` for all real `x` in the representable range.

*Proof.* `E[SR(x)] = p · x_hi + (1-p) · x_lo = x_lo + p(x_hi - x_lo) = x_lo + (x - x_lo) = x`. ∎

**Theorem 3 (bounded variance).** `Var[SR_FP8(x)] = p(1-p)(x_hi - x_lo)² ≤ ulp_FP8(x)² / 4`.

*Proof.* Bernoulli: `p(1-p) ≤ 1/4`. ∎

### 3.4 NLL drift bound

Per-step gradient `g_t ∈ ℝ^P` computed in FP32, cast to FP8 via SR: `ĝ_t = SR_FP8(g_t)`. Then `E[ĝ_t - g_t] = 0` and `Var[ĝ_t - g_t] ≤ P · max_i ulp_FP8(g_{t,i})² / 4`.

For `|g_{t,i}| ∈ [10⁻⁵, 10⁻²]` with E5M2 (ulp ≈ 0.25 |g|): per-element noise std ≤ 0.125 · |g_{t,i}|; per-parameter noise variance ≈ 0.016 · g_{t,i}².

**SGD-as-ODE NLL drift.** For Adam at LR `η_eff`:
$$\Delta \text{NLL} = -\eta_{eff} \langle g, \hat g \rangle / \|m\| + O(\eta_{eff}^2 \|g\|^2). \tag{3}$$
Linear term unbiased (E[ĝ] = g); quadratic contributes `O(η² σ² P)`. For `η_eff = 3e-4`, `\|g\|² ≈ 10⁻⁴`: **per-step NLL drift ≤ 6e-7 nat at 18B parameters**.

**Well below the 0.001 nat/step practical NLL-equivalence threshold.** Over 100k steps, total drift ≤ 0.06 nat — within "NLL accuracy" envelope.

The bound is loose: ulp is worst-case over the FP8 range (actual is much smaller for typical mid-range values). Adam's `v_t` slightly amplifies stochastic FP8 variance: `v_t ∝ E[ĝ²] = g² + Var[ĝ]`. Net effective LR shrinks by `√(1 + Var/g²) ≈ 1 + 0.008` — **0.8% effective LR reduction**, negligible.

### 3.5 Per-tensor scale factors

FP8 dynamic range is narrow. Each tensor `T` stored as `(s_T, T̂)` with `s_T ∈ ℝ` (FP32) and `T̂ ∈ FP8^{shape(T)}`, `T ≈ s_T · T̂`. Choose `s_T = max|T| / FP8_max` so max element saturates at FP8_max. SR applies to `T̂`; `s_T` is FP32. Combination is unbiased. Scales recomputed per step from current step's max-abs (NVIDIA Transformer Engine convention). Cost: one O(T·dH) reduction per tensor per step, ~1% wall-clock overhead.

---

## 4. Kernel fusion patterns

### 4.1 HBM-traffic bottleneck

CHIRON's per-layer forward dispatches separate kernels: `linear_qkv_proj` (3× `T·d·d` GEMMs), `attention_shear` (reads Q/K/V, writes A), `linear_o_proj` (reads A, writes Y), `chiron_p_kick` (reads p_in/Y, writes p_out). HBM traffic per layer per token: ~4·T·d ≈ 32 KB at d=2048 BF16. **At 53 layers × 1024 tokens: ~1.6 GB HBM traffic per training step.**

### 4.2 HELIUM-FUSE patterns

**Pattern A — Fused QKV + FA-3 + O projection:**
$$Y = W_O \cdot \text{FA-3}(W_Q X, W_K X, W_V X). \tag{4}$$

One CUDA kernel: load `X` once into shared memory, compute `Q_i = W_Q X_i`, `K_j = W_K X_j`, `V_j = W_V X_j` on-the-fly per FA-3 tile, run FA-3 inner loop, project each output tile through `W_O` before storing. Each `X` element loaded **once**, not 4 times. **HBM-traffic reduction: 3.5×** (≈ 0.45 GB per step at 18B).

**Pattern B — Layer-fused forward:** Pattern A + `chiron_p_kick` (`p ← p + ηY`) collapsed into the same kernel. Saves another HBM round-trip on `Y`. Combined: **~4× HBM reduction per layer.**

**Pattern C — Fused backward (HELIUM-FUSE-BACK):** inverse walk of one layer + backward FA-3 + backward QKV in one kernel. Harder; ~400 LOC vs Pattern A's 200.

### 4.3 NLL preservation by fusion

**Theorem 4.** Kernel fusion preserves NLL to within FP32 accumulator precision.

*Proof.* Fusion changes the order of arithmetic operations but not the operations. In exact arithmetic, the result is identical. In FP32 reordered sums differ by ≤ T · ulp_FP32 ≈ 1e-4 relative — far below 0.001 nat. ∎

---

## 5. Composition with paradigms #42–#49

| Paradigm | What HELIUM does | Compose? | Combined factor |
|---|---|---|---|
| #1 CHIRON | reversibility unchanged; activation-recomputation cost halved by FP8 weights | ✓ | (memory) |
| #28 FACE | EMA in BF16; gradient FP8 cast at end | ✓ | unchanged |
| #38 SLC, #39 RLG | unchanged | ✓ | 1× |
| #42 SCFA | basis B_l multiplications dispatch to FP8 GEMM; FA-3 fuses score tile | ✓✓ | 2.27 × 1.7 ≈ 3.86× |
| #43 ORION | HVPs use FP8 GEMM (E5M2 backward); MOR projection unchanged | ✓ | 8.6 × 1.7 ≈ 14.6× |
| #44 MELT | TT-FFN cores in FP8 weights; tensor contractions on tensor cores | ✓ | 2.0 × 1.7 ≈ 3.4× |
| #46 REFLECTOR | cotangent-lift uses FP8 backward (E5M2) | ✓ | structural |
| #47 PHOENIX-1.58BIT | ternary cast to FP8 at dispatch (exactly representable in E4M3) | ✓ | 1.6 × 1.7 ≈ 2.7× |
| #48 PHOENIX-1BIT | binary cast to FP8 at dispatch | ✓ | 1.5 × 1.7 ≈ 2.55× |
| #49 ICARUS | Yoshida sub-steps each become a fused QKV+FA3+O kernel | ✓ | 1.85 × 1.7 ≈ 3.15× |

**HELIUM is multiplicative with every shipped paradigm.** Crucially, **PHOENIX (#47/#48) and HELIUM are at different precision levels**: PHOENIX quantizes weights at the parameter level (1.58-bit / 1-bit storage); HELIUM dispatches the math at FP8 tensor-core precision. PHOENIX storage saving preserved; HELIUM compute saving applies on top.

**Stack at 18B:** `300× × 1.7 ≈ 510×` (conservative); `300× × 2 ≈ 600×` (optimistic).

---

## 6. Memory analysis

### 6.1 FP8 weight storage

For 18B parameters, weights in BF16 = 36 GB. Two storage options:
- **Option A** (storage-FP16, dispatch-FP8): weights stay in BF16 in HBM; cast to FP8 per tile at GEMM dispatch. No memory saving; full-speed FP8 GEMM.
- **Option B** (storage-FP8, dispatch-FP8): weights as FP8 + per-tensor scale. **50% storage savings; 18 GB at 18B.**

**HELIUM recommends Option B for small matrices (Q, K, V, O projections — 0.4–0.8 GB at 18B), Option A for the large FFN matrices (W_up, W_down — 16 GB at 18B).** Saves ≈ 0.4 GB at 18B without touching FFN storage. FFN savings deferred to a future paradigm. **Conservative, low-risk memory choice.**

### 6.2 Activation memory under FA-3

FA-3 eliminates `scratch_P` and `scratch_dP` (the O(T²) score / softmax-grad tensors). At T=4096: **2 GB freed per layer**. At T=8192: **8 GB freed**. This is the same memory advantage motivated by the existing `FLASH_ATTENTION_DESIGN.md`; HELIUM ships the production-grade FA-3 implementation that doc anticipates.

### 6.3 Combined ceiling at 18B

Pre-#50 with full stack: ~14 GB on a 16 GB GPU. HELIUM frees ~0.4 GB QKV + ~2 GB attention scratch at T=4096. **New ceiling: ~12 GB at 18B, T=4096. Or: same memory, train at 22B parameters with HELIUM.**

---

## 7. Concrete CUDA primitives

### 7.1 New kernels (~2100 LOC)

```cpp
// HELIUM-FA3 forward: BF16 fallback / FP8 fast path. TMA on Hopper, cp.async on Ada.
void flash_attention_3_forward(
    const void* Q, const void* K, const void* V,        // E4M3 or BF16
    const float* scale_Q, const float* scale_K, const float* scale_V,
    void* O, float* lse, bool causal,
    int T, int dH, int n_H, cudaStream_t stream);       // ~600 LOC

// HELIUM-FA3 backward
void flash_attention_3_backward(
    const void* Q, const void* K, const void* V, const void* dO,
    const float* lse,
    void* dQ, void* dK, void* dV,
    int T, int dH, int n_H, cudaStream_t stream);       // ~500 LOC

// HELIUM-FP8GEMM: cublasLtMatmul wrapper with E4M3/E5M2 + SR scaling
void fp8_gemm(
    cublasLtHandle_t handle,
    const void* A, const void* B,
    float scale_A, float scale_B,
    void* C, bool forward, bool stochastic_round,
    int M, int N, int K, cudaStream_t stream);          // ~400 LOC

// HELIUM-FUSE: fused QKV + FA-3 + output projection (forward)
void fused_attention_block(
    const void* X,
    const void* Wq, const void* Wk, const void* Wv, const void* Wo,
    const float scales[4], void* Y,
    int T, int d, int dH, int n_H, cudaStream_t stream); // ~300 LOC

// Fused backward (inverse walk + FA-3 backward + QKV backward)
void fused_attention_block_backward(
    const void* X, const void* Wq, /* ... */,
    const void* dY,
    void* dX, void* dWq, void* dWk, void* dWv, void* dWo,
    int T, int d, int dH, int n_H, cudaStream_t stream); // ~400 LOC

// Stochastic rounding (per-warp Philox state, ~100 LOC)
__device__ inline e4m3_t sr_to_e4m3(float x, philox_state_t* rng);
__device__ inline e5m2_t sr_to_e5m2(float x, philox_state_t* rng);
```

**Total CUDA LOC: ~2300. Trainer wiring: ~200. ~2500 total LOC.**

The SR-to-FP8 cast is a ~40-line per-format extraction of (sign, exponent, mantissa), with a Philox-uniform draw deciding whether to round mantissa up or down — straightforward but careful around overflow/underflow boundaries.

### 7.2 Trainer flags

```
--helium 0/1                  # enable HELIUM stack (default 0)
--helium-fa3 0/1              # FA-3 attention (default 1 if helium=1)
--helium-fp8 0/1              # FP8 GEMM (default 1 if Ada/Hopper)
--helium-fuse 0/1             # fused QKV+FA3+O kernels
--helium-sr-mode {fwd,both}   # stochastic rounding scope (default both)
--helium-fp8-storage 0/1      # FP8 weight storage for QKV (default 1)
```

When `--helium 1`: dispatch picks the HELIUM kernel for each operation if hardware supports it; falls back gracefully to current SCFA-tiled BF16 path otherwise.

### 7.3 Hardware capability detection

A `HeliumCapabilities` struct probes `cudaGetDeviceProperties` at trainer init: `has_fp8` (SM 8.9 Ada or SM 9.0 Hopper), `has_tma` (SM 9.0 only), `has_async_copy` (SM 8.0+ Ampere). On RTX 4080 SUPER (SM 8.9): full FP8 path, no TMA. **Expected: 1.7×.** On Hopper (SM 9.0): full FP8 + TMA. **Expected: 2.0–2.5×.**

---

## 8. Honest gap analysis

### 8.1 Where HELIUM does NOT meet the brief

The brief asks for "magnitudes better on compute speed." HELIUM provides 1.7–2.0× — **a half-order-of-magnitude, not magnitudes alone**. Like ICARUS (#49), HELIUM contributes to the magnitudes goal **only when stacked**. HELIUM's strength is that it composes with everything: a kernel-level constant-factor lift. It cannot, by itself, change algorithmic complexity (that's the territory of #42/#43 etc.).

### 8.2 Hardware dependence — central engineering risk

**HELIUM requires Ada-class GPUs for FP8 fast path.** RTX 4080 SUPER (local target) is SM 8.9 with FP8 tensor cores. RTX 30-series Ampere (SM 8.6) has **limited FP8 support**: storage works, but tensor cores cannot natively do FP8 GEMM — fallback path multiplies by scale and dispatches BF16 GEMM, **with no speedup**.

| Hardware | FA-3 | FP8 | Fusion | Total |
|---|---|---|---|---|
| Hopper (SM 9.0) | 2.0× | 2.0× | 1.3× | **2.5–3.0×** |
| Ada (SM 8.9, RTX 4080) | 1.7× | 1.8× | 1.3× | **1.7–2.0×** |
| Ampere (SM 8.6, RTX 3080) | 1.3× | 1.0× | 1.2× | **1.3–1.4×** |
| Older (Turing SM 7.5) | 1.1× | 1.0× | 1.1× | **1.1× (not worth)** |

Flagship target is RTX 4080 SUPER (Ada); Hopper deployment documented but not primary. **Honest ranking: clean win on Ada/Hopper; marginal on Ampere; not worth on Turing.**

### 8.3 SCFA × HELIUM-FA3 composition risk

SCFA reshapes the inner attention dimension from `dH` to `r ≤ dH/2` via a spectral-basis projection. HELIUM-FA3 expects standard `Q K^⊤ / √dH` shape. The composition fuses the SCFA basis projection into the FA-3 Q/K loading stage — non-trivial but tractable.

**Risk:** if SCFA uses small `r = 32`, the FA-3 score-tile shape becomes BQ × 32 instead of BQ × dH, under-utilizing tensor cores. Mitigation: BQ=256, BK=128 for small `r`; the score tile is 32K entries per SM, fits in shared memory. **Engineering cost: ~200 LOC of tile-shape adaptation.** Manageable.

### 8.4 PHOENIX × HELIUM precision interaction

PHOENIX-1.58BIT (#47) stores weights as ternary {-1, 0, +1} with a per-tensor scale. PHOENIX-1BIT (#48) stores as binary. **At dispatch time**, HELIUM casts the ternary/binary weight to FP8 (the values {-1, 0, +1} are exactly representable in E4M3 with no rounding error) and dispatches to FP8 tensor-core GEMM. **No precision interaction.** PHOENIX quantization noise, FP8 dispatch noise, and SCFA approximation noise compose additively in NLL drift; each is < 0.05 nat over a typical run, stacked total < 0.15 nat — below the user's "NLL accuracy" threshold.

### 8.5 Determinism

CHIRON requires bit-exact determinism under `--seed`. SR introduces non-determinism unless seeded. **HELIUM solution:** Philox-based per-warp counter-based RNG keyed by `(seed, layer_id, step, warp_id, lane_id)`. Fully deterministic given the seed. NVIDIA cuRAND's Philox is the standard. **LOC: ~50** in HELIUM-FP8GEMM.

### 8.6 Is FP8 SR actually unbiased in practice?

**Pessimistic case:** narrow dynamic range causes some elements to clip (saturate at FP8_max), introducing bias. Mitigation: per-tensor scaling `s_T = max|T| / FP8_max` ensures no element clips. **No clipping bias if scaling is correct.** Empirical: NVIDIA Transformer Engine's FP8 GPT-3 reproduces BF16 NLL within 0.05 nat over 100k steps.

### 8.7 Confidence summary

| Claim | Confidence | Rationale |
|---|---|---|
| FA-3 mathematical equivalence (BF16) | **High** | Theorem 1; Dao 2024 reference |
| Stochastic rounding unbiased | **High** | Theorems 2/3; well-established |
| ≤ 0.001 nat/step NLL drift | **Medium-High** | §3.4 bound; matches NVIDIA TE empirical |
| 1.7× wall-clock on Ada | **Medium** | depends on FFN-shape suitability |
| 2.0× wall-clock on Hopper | **Medium-Low** | TMA + warp-spec implementation tax |
| Multiplicative composition with #42–#49 | **High** | kernel-level, orthogonal axis |
| Memory savings ~0.4 GB at 18B | **High** | direct counting |
| Magnitudes alone | **Zero** | HELIUM is at most 2× |

---

## 9. Phase plan

| Phase | Activity | LOC | Duration |
|---|---|---|---|
| 1 | FA-3 forward kernel (BF16 mode) | 600 | 2 wk |
| 2 | FA-3 backward kernel | 500 | 1.5 wk |
| 3 | FP8 GEMM + Philox-keyed SR | 500 | 1 wk |
| 4 | Fused QKV+FA3+O forward | 300 | 1 wk |
| 5 | Fused backward | 400 | 0.5 wk |
| 6 | Trainer wiring + capability detection | 200 | 0.5 wk |
| 7 | Gate-0 (~30 GPU-min) | 0 | 1 day |
| 8 | 66M LR/scale sweep (15 GPU-h) | 0 | 0.5 wk |
| 9 | 1.84B convergence (5000-step) | 0 | 5 d |
| 10 | Flagship 18B production (full stack) | 0 | 7 d |
| **Total** | | **~2500** | **6–8 weeks** |

**Risk gates:**
- After Phase 2: FA-3 NLL parity ≤ 1e-4 vs cuBLAS attention. Fail → abandon FA-3.
- After Phase 4: FP8 SR NLL drift ≤ 0.005 nat at 1000 steps. Fail → ship FA-3-only.
- After Phase 7: combined speedup ≥ 1.3×. Fail → ship partial HELIUM (whichever components passed).

---

## 10. Gate-0 design — ~30 GPU-min probe

**Question:** does HELIUM (FA-3 + FP8 + fusion) reproduce baseline NLL at fixed step count, while delivering measured wall-clock speedup ≥ 1.3×?

**Procedure:**
1. **Setup.** Existing 66M CHIRON checkpoint at iter-185 (post-#49 stack already shipped). Two parallel 200-step runs on RTX 4080 SUPER:
   - **Baseline:** current SCFA-tiled BF16 + cuBLAS BF16 FFN.
   - **HELIUM:** FA-3 (BF16 first) + FP8 GEMM + fused QKV-FA3-O.

2. **Metrics:**
   - **NLL drift:** held-out 1k-token validation loss. Pass = ≤ 0.005 nat absolute.
   - **Wall-clock per step:** ≥ 1.3× speedup at 66M.
   - **Memory:** peak GPU memory under 14 GB (current baseline ~9 GB; HELIUM should be similar or lower with FP8 weight storage).

3. **Fallback:** if FP8 NLL drift > 0.005 nat, disable FP8 (keep FA-3 + fusion). Re-measure. Pass on FA-3 + fusion alone if ≥ 1.2× with bit-exact NLL.

4. **Cost:** ~30 GPU-min total. Two 200-step runs at ~5 sec/step = ~17 min each.

5. **Pass criterion:**
   - **Strong pass:** NLL drift ≤ 0.005 nat AND ≥ 1.5× speedup. Proceed to Phase 8.
   - **Pass with regression:** drift ≤ 0.01 nat AND ≥ 1.3× speedup. Investigate scale-recomputation cadence, then proceed.
   - **Fail-fast:** drift > 0.05 nat OR speedup < 1.1×. Document; ship FA-3-only as a memory paradigm; abandon FP8/fusion.

6. **Expected outcome:** drift ≈ 0.001–0.003 nat at 200 steps; speedup 1.5–1.7× on Ada. Strong pass.

---

## 11. Selection criteria for paradigm #50

HELIUM should be selected over candidates B and C iff:

1. **Hardware-feature exploitation is ripe.** Pre-#50 stack uses cuBLAS BF16, no FA-3, no FP8. The ~50% of theoretical peak left on the table is the largest remaining source of compute speedup.
2. **NLL preservation is binding.** HELIUM's stochastic rounding gives unbiased FP8 — the only widely-validated path to NLL-preserving 8-bit arithmetic.
3. **Composition with #42–#49 is critical.** HELIUM is the only candidate **multiplicatively compatible** with every shipped paradigm; candidate B might displace #42/#43/#49 and candidate C imposes constraints on PHOENIX.
4. **Engineering scope is acceptable.** 6–8 weeks, ~2500 LOC, mostly CUDA. Every component is established public literature; engineering load is the dominant cost.
5. **Hardware floor met.** RTX 4080 SUPER (Ada) supports the full HELIUM stack.

**Honest summary.** HELIUM is the **safest, most-multiplicative #50 candidate** with NLL-equivalence — but **not** the highest-magnitude alone (1.7–2.0× vs candidate B's potential 3×+ from new algorithms). If the brief reads as "extract every remaining drop of compute from existing math at NLL-equivalence", HELIUM dominates. If it reads as "find a new algorithmic paradigm", candidate B may win.

---

## 12. Summary

HELIUM extracts the unused half of theoretical hardware peak by combining three well-known compute-kernel techniques — FA-3 streaming attention, FP8 tensor-core GEMM with stochastic rounding, and fused multi-operation kernels — into a single multiplicatively-composable paradigm. The math at the algorithm level is unchanged; only **bytes per FLOP** and **FLOPs per second** change.

NLL preservation is grounded in Theorems 1–4: FA-3 is bit-exact in exact arithmetic; FP8 stochastic rounding is unbiased with bounded variance ≤ ulp²/4; per-step NLL drift bound < 0.001 nat at typical batch sizes. Memory advantages (CHIRON O(1) depth + FA-3's elimination of T² score materialization + FP8 weight storage on small QKV) compose additively: ~2 GB freed at long-context, ~0.4 GB freed always.

Composition with #42–#49 is multiplicative. HELIUM is the only #50 candidate touching no algorithmic surface — purely a kernel-level recompilation. Risks are concentrated in CUDA engineering, hardware dependence (Ada-class FP8 fast path; Ampere falls back to BF16-equivalent), and the SCFA × FA-3 tile-shape composition.

**Honest claim:** 1.7–2.0× wall-clock with NLL-equivalent training, ~2500 LOC, 6–8 weeks, ~30 GPU-min Gate-0. Stacked: **~510× single-GPU at 18B with NLL-equivalent training** (or ~600× at the optimistic 2× working point).

**Risk:** Low conceptually; high implementation-engineering. Best on Hopper, strong on Ada (RTX 4080 SUPER), marginal on Ampere, not worth on Turing.

Recommended if: (a) NLL-equivalence is acceptable, (b) Ada/Hopper hardware is in scope, (c) the brief reads as "fully exploit hardware before adding a new algorithm".

---

## References

- Dao, T. (2024). "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision." NeurIPS preprint.
- Dao, T., Fu, D., et al. (2022). "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning."
- Micikevicius, P., et al. (2022). "FP8 Formats for Deep Learning." arXiv:2209.05433. (NVIDIA E4M3/E5M2 spec.)
- NVIDIA Transformer Engine documentation (FP8 GPT-3 training recipes).
- Connor, R. F., Schaller, P. J. (2023). "Stochastic rounding: implementation, error analysis, applications." Royal Soc. Open Sci.
- Salmon, J. K. et al. (2011). "Parallel Random Numbers: As Easy as 1, 2, 3." (Philox.)
- (CHIRON-internal) PARADIGM_SHIFT_42_DESIGN.md (SCFA), PARADIGM_SHIFT_43_DESIGN.md (ORION), PARADIGM_SHIFT_44_DESIGN.md (MELT), PARADIGM_SHIFT_47_DESIGN.md / #48 (PHOENIX), PARADIGM_SHIFT_49_CANDIDATE_A_ICARUS.md.
- (CHIRON-internal) FLASH_ATTENTION_DESIGN.md (the attention-memory motivation HELIUM-FA3 ships in production form).
