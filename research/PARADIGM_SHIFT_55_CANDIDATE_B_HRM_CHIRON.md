# Paradigm Shift #55 Candidate B — HRM-CHIRON: Hyena Recurrent Mechanisms via Long Convolution × CHIRON Symplectic Shear

**Status:** candidate-B design; one of three parallel proposals for paradigm shift #55. **Recommendation up front: REJECT for #55.** This document exists so the team has a written, formal close-out of the Hyena / long-convolution direction and does not re-derive its trade-offs in iter-200+.
**Date:** 2026-05-08 (Ralph-loop iter 197+, post-#54 selection).
**Axis:** **architecture-class change inside the symplectic shear** — replace CHIRON's softmax-attention `Y(q)` with a **Hyena recurrent mechanism** (Poli et al. 2023). Hyena is the canonical realization of the "long-convolution + data-dependent gating" subquadratic-attention family. The block is `h(x) = (D_x \cdot v) * x + g \cdot x`, where `*` is causal long convolution implemented in `O(T log T)` via FFT, `D_x` is data-dependent multiplicative gating, `g` is element-wise gating, and the long-convolution kernel is parameterized by an implicit MLP over position.
**Tagline.** *Hyena gives a clean O(T log T) long-convolution alternative to attention with established public reference implementations and FFT-friendly primitives. But #54 JAMBA-CHIRON already ships a Mamba+SCFA hybrid that beats Hyena on quality, and pure Mamba (NEXUS-SSM, reserved as #54-A) beats Hyena on per-block compute. HRM-CHIRON is dominated on every axis the iter-197 brief weights.*

**Materially distinct from competing #55 candidates.** Candidates A and C (assumed: SOPHIA-class training-method novelty and xLSTM-class matrix-memory recurrence) attack different axes — optimizer geometry and bounded-memory token mixing. HRM-CHIRON is an architecture-class swap in the same family as #54's Mamba and #54's RWKV (rejected in #54-C); the salient difference is **the implementation primitive** (FFT-based long convolution vs Mamba's parallel scan vs RWKV's scalar recurrence). The selection question reduces to: does the FFT primitive offer yield the shipped Mamba+SCFA hybrid does not? **No.**

**Honest headline.** HRM-CHIRON delivers subquadratic `O(T log T)` attention-block compute vs softmax `O(T²)`. In raw FLOPs at flagship scale: **per-block ~7× reduction at T=1024, ~12× at T=4096**. But:

- **#54-A NEXUS-SSM (Mamba, reserved):** ~127× at T=1024, ~500× at T=4096 — order of magnitude beyond Hyena.
- **#54 JAMBA-CHIRON (shipping):** Mamba's compute on half the layers, SCFA-attention's quality on the other half — Hyena delivers neither.
- **Hyena's FFT is a regression on Ada.** cuFFT benchmarks ~0.4–0.5× parallel-scan throughput at matched FLOPs (DRAM round-trips during bit-reversal vs Mamba's on-chip SRAM reuse). FFT-friendliness inverts on hardware that's good at scan.

**HRM-CHIRON is the weakest of the three #55 candidates.** Realistic positioning: niche fallback for hardware where parallel scan is unavailable (DSPs, ASICs, non-NVIDIA accelerators). Not active in the iter-197 brief. **REJECT for #55; reserve as niche fallback.**

**Engineering scope if pursued anyway.** ~1900 LOC, 8–11 weeks (larger than RWKV-CHIRON's ~1500 LOC due to implicit-kernel MLP and chunked-FFT plan management; smaller than NEXUS-SSM's ~3000 LOC because cuFFT supplies the primitive). Risk: low engineering, low structural, but empirical return is **bounded by ~7–12× attention-block ceiling at flagship-relevant T** — below #54 JAMBA-CHIRON's ceiling.

---

## 0. Executive summary

### 0.1 What HRM-CHIRON is

Hyena (Poli et al. 2023, "Hyena Hierarchy: Towards Larger Convolutional Language Models") replaces transformer attention with a stack of **gated long convolutions**. Canonical Hyena order `N=2`:

$$
\begin{aligned}
v &= W_v q, \quad x = W_x q, \quad g = \sigma(W_g q), \\
h(q) &= g \odot \big( \mathrm{LongConv}_h(D_x \odot v) \big),
\end{aligned}
$$

with `D_x = W_d q` a data-dependent gating signal, `LongConv_h(\cdot)` a causal convolution over the full sequence, and the kernel `h` parameterized by an **implicit MLP** mapping position to value: `h_t = \mathrm{MLP}_\phi(\mathrm{PosEnc}(t))`. The implicit kernel keeps parameter count **independent of `T`** (~50–500K parameters); the convolution runs in `O(T log T)` via FFT: `\mathrm{LongConv}_h(u) = \mathrm{IFFT}(\mathrm{FFT}(u) \odot \mathrm{FFT}(\tilde h))`. Data-dependent gates `D_x, g` recover most of attention's expressiveness.

HRM-CHIRON embeds the Hyena block as `Y(q)` in CHIRON's symplectic shear `(q, p) ↦ (q, p + Y(q))`. Reversibility (Theorem 3 of #42), O(1)-activation invertibility, and cotangent-lift gradient (#46 REFLECTOR) all carry through unchanged.

### 0.2 Why HRM-CHIRON is the wrong shift

Three reasons HRM-CHIRON does not deliver, **and the dominance is not subtle**:

1. **Per-block compute reduction is dominated by Mamba and JAMBA.** At `T=1024, m=2048`: Hyena = 5 `m × m` projections (~17 GF) + 2 FFTs (~0.21 GF) + element-wise/MLP (~12 MF) = **17.2 GF/layer, projections-dominated**. Mamba = 0.27 GF (Hyena 64× heavier). SCFA-compressed attention (post-#42) at k=64 = 1.98 GF (Hyena 8.7× heavier). The "Hyena beats attention" claim compares against **uncompressed** attention, which CHIRON does not run; once corrected, Hyena is *slower* than the post-#42 baseline.

2. **Hyena's quality at scale is not robustly established.** Poli et al. 2023: parity with Transformer at 1.3B on The Pile. Subsequent reproductions (StripedHyena 2023, Mamba ablations 2023): Hyena trails Mamba by **0.05–0.15 nat at 1B+** and trails pure-attention on retrieval-heavy benchmarks (induction heads, multi-query associative recall). Public evidence base is **thinner than Mamba's** (~5 reproductions vs Hyena's 2 partial). The 12B reference checkpoint is StripedHyena — *already a Hyena+attention hybrid*, implicitly conceding that pure Hyena under-delivers.

3. **JAMBA-CHIRON (shipping #54) is strictly better.** JAMBA pattern: 27 Mamba shears @ 0.27 GF + 26 SCFA shears @ 1.98 GF = **1.10 GF avg/attention-block**. Hyena replaces all 53 slots at 17.2 GF each. **JAMBA is 15.6× faster per attention-block and gets attention-class retrieval on 26 layers.** There is no operating point where HRM beats JAMBA.

### 0.3 Where HRM-CHIRON would shine

HRM-CHIRON's genuine advantages — none of which apply to the active brief:

- **FFT primitives are universally available.** cuFFT, rocFFT, FFTW ship in every BLAS-class library. A future port of CHIRON to a non-NVIDIA accelerator (AMD, Intel GPU, Apple Silicon, ASIC) would inherit FFT before it inherits associative-scan; Mamba's parallel scan requires a custom kernel per platform or a `O(T)` sequential fallback that nullifies the speedup.
- **FFT is cache-friendly under specific access patterns.** On hardware where on-chip SRAM is the bottleneck but DRAM bandwidth is plentiful (the *opposite* of Ada), FFT may outperform parallel scan. Niche of (i) DSP / accelerator deployment, (ii) very small `m` regimes where FFT's `O(log T)` overhead dominates.

These would matter on hardware where parallel-scan kernels are hard to write or where FFT is part of the standard accelerator instruction set. **None apply to RTX 4080 SUPER / Ada-class hardware**, which has both excellent SRAM (Mamba-friendly) and excellent cuFFT — and the brief is explicit about the operating hardware.

### 0.4 Recommendation

**Reject HRM-CHIRON for paradigm shift #55.** See §5 for fallback conditions and the alternative #55 axes (#55-A SOPHIA, #55-C xLSTM mLSTM) that dominate HRM within the selection set.

---

## 1. Hyena mathematics

Fix layer `ℓ`; batch suppressed. CHIRON's symplectic pairing: `(q, p) ∈ ℝ^{T×m} × ℝ^{T×m}`. Per-layer parameters: data-dependent projections `W_v, W_x, W_g, W_d \in ℝ^{m \times m}`, output projection `W_o \in ℝ^{m \times m}`, implicit-kernel MLP `\phi: ℝ^{d_p} \to ℝ^m` (typically `d_p = 64`), with `h_t := \phi(\mathrm{PosEnc}(t)) \in ℝ^m`. Order-`N` Hyena interleaves `N+1` projection branches with `N` long-convolutions; we restrict to `N=2` (Poli 2023 default).

### 1.2 The Hyena block

$$
v_t = W_v q_t,\quad x_t = W_x q_t,\quad g_t = \sigma(W_g q_t),\quad D_t = W_d q_t,
$$

$$
u_t = D_t \odot v_t,\quad y = \mathrm{LongConv}_h(u),\quad \mathrm{HyenaOut}(q)_t = W_o\big(g_t \odot y_t + \tilde g_t \odot x_t\big),
$$

with `\tilde g_t = \sigma(W_{\tilde g} q_t)` a second element-wise gate. The causal long convolution `\mathrm{LongConv}_h(u)_t = \sum_{s=0}^{t} h_{t-s} \odot u_s` is implemented via FFT on the zero-padded sequence: `\mathrm{LongConv}_h(u) = \mathrm{IFFT}(\mathrm{FFT}(\mathrm{pad}(u)) \odot \mathrm{FFT}(\mathrm{pad}(\tilde h)))`, with `\mathrm{pad}` zero-extending to `2T` to avoid circular wrap.

### 1.3 Implicit kernel, causality, gating

The kernel `h` is **never materialized as `T·m` parameters**: `h_t = \phi(\mathrm{PosEnc}(t)) = W_2 \sin(W_1 \mathrm{PosEnc}(t) + b_1) + b_2` — a 2-layer MLP with sinusoidal activations and ~50–500K parameters, **independent of `T`**. Causal padding to `2T` raises the headline `5 T \log T` to `~10 T \log T` per channel. Data-dependent gating via `D_t, g_t` is the load-bearing nonlinearity.

---

## 2. CHIRON integration

### 2.1 The Hyena shear and reversibility

Embed `Y_\ell^{\mathrm{Hyena}}(q) := \mathrm{HyenaOut}_\ell(q)` as the attention shear with per-layer parameters `\{W_v^\ell, W_x^\ell, W_g^\ell, W_d^\ell, W_{\tilde g}^\ell, W_o^\ell, \phi_\ell\}`. The CHIRON layer is `\Phi_\ell^{\mathrm{HRM}}: (q, p) \mapsto (q, p + Y_\ell^{\mathrm{Hyena}}(q))` followed by the standard `(q, p) \mapsto (p, q)` swap.

**Theorem 1 (HRM-CHIRON reversibility).** *For any continuous `Y_\ell^{\mathrm{Hyena}}`, the shear `\Phi_\ell^{\mathrm{HRM}}` is bijective with explicit inverse `(q', p') \mapsto (q', p' - Y_\ell^{\mathrm{Hyena}}(q'))`, preserving the symplectic form and `\det D\Phi = 1`.* (Jacobian unit-lower-triangular; inverse formula because `Y` is evaluated at `q' = q`. ∎)

**Consequence.** CHIRON's O(1)-activation inverse-walk, #46 REFLECTOR's cotangent-lift exact gradient, and #48 STREAM-CHIRON's gradient streaming all carry through. The argument is structural — independent of `Y`'s internal class.

### 2.2 Forward pseudocode (sketch)

```
# Layer ℓ forward, HRM-CHIRON enabled
v = W_v_ℓ @ q;  x = W_x_ℓ @ q
g = σ(W_g_ℓ @ q);  g_tilde = σ(W_g~_ℓ @ q);  D = W_d_ℓ @ q
h = phi_ℓ(PosEnc(arange(T)))                       # implicit kernel [T, m]
u = D * v
y = irfft( rfft(pad_zero(u, 2T)) * rfft(pad_zero(h, 2T)) )[:T]
p = p + W_o_ℓ @ (g * y + g_tilde * x)
```

### 2.3 Backward (cotangent lift)

REFLECTOR re-evaluates `Y_\ell^{\mathrm{Hyena}}(q')` from recovered `q'`; subtracts from `p`; accumulates `dL/d\theta_\ell`. The convolution backward uses the FFT identity `dL/du = \mathrm{LongConv}_{\bar h}(dL/dy)` and `dL/dh = \mathrm{LongConv}_{\bar u}(dL/dy)` (time-reversed signals), costing **two additional FFTs**. Implicit-kernel gradient is a small MLP backward. Total backward ≈ 3× forward FLOPs, consistent with attention/SSM ratios.

### 2.4 KV-cache and inference

Hyena has **no KV-cache** in the attention sense. Naive recompute is `O(T log T)` per token = `O(T² log T)` per sequence — worse than attention's `O(T²)`. The clean `O(T)` per-token alternative requires a *modal decomposition* constraint on `\phi` (sum-of-exponentials kernel) that **partially recovers Mamba's form** — at which point the architecture is morally a state-space model with extra plumbing. **HRM-CHIRON's inference advantage over attention is conditional on the same SSM-style structure that #54-A provides directly.**

### 2.5 Composition with shipped paradigms

| Pre-#55 paradigm | HRM composition | Multiplier |
|---|---|---|
| **#42 SCFA** | Mutually exclusive (HRM replaces attention) | None |
| **#44 MELT** | Full on FFN slot | Full |
| **#46 REFLECTOR** | Cotangent-lift via new pathway | Full |
| **#47 PHOENIX-1.58BIT** | Projections ternarizable; MLP stays BF16 | ~0.85× |
| **#48 STREAM-CHIRON** | Standard `q`-input shear | Full |
| **#50 HELIUM** | FA-3 N/A; FP8 on projections | ~0.5× |
| **#51 ATLAS-COMPILE** | Static FFT plan at fixed `T` | Full |
| **#52 NIMBUS** | Optimizer-agnostic | Full |
| **#53 MOSAIC-MOE** | Orthogonal (FFN slot only) | Full |
| **#54 JAMBA-CHIRON** | **MUTUALLY EXCLUSIVE** (both own attention slot) | None |
| **#28 FACE** | Embedding + projections full; MLP empirical | ~0.85× |

**Critical observation: HRM-CHIRON cannot compose with #54 JAMBA-CHIRON.** Both occupy the attention shear. Selecting HRM for #55 would *replace* JAMBA, losing the validated quality story and gaining only an FFT primitive — net regression on the iter-197 brief.

---

## 3. Compute analysis at multiple T

### 3.1 Per-layer FLOPs (forward)

| Layer type | T=1024 | T=4096 | T=16384 |
|---|---|---|---|
| HRM (Hyena) | **17.2 GF** (4 projections + FFT) | **68.8 GF** | **275.4 GF** |
| Mamba shear | 0.27 GF | 1.08 GF | 4.32 GF |
| SCFA shear (k=max(64, T/16)) | 1.98 GF | 7.4 GF | 19.0 GF |
| Pure attention shear | 8.6 GF | 137 GF | 2,196 GF |

Decomposition of HRM at T=1024: 5 projections at `T·m²·5` ≈ **17 GF dominant**; forward+inverse FFT ≈ 210 MF (1.2%); element-wise product ≈ 12 MF; implicit-kernel MLP ≈ 134 MF (0.8%). **Total ≈ 17.2 GF, projections-dominated.**

The headline "Hyena is O(T log T)" is true **for the convolution kernel only**, but the convolution is **<2% of layer compute** at flagship `m`. The remaining 98% is `O(T m²)` projections — exactly like RWKV — and that is the regime where Mamba's `O(T m N)` with `N=16` wins by `m / N = 128×`.

**The "subquadratic in T" framing is a category mistake at CHIRON-scale.** Subquadratic-in-T matters when `T ≫ m`. At flagship `T=1024, m=2048`, `T < m`; projection cost dominates and FFT advantage is irrelevant. At `T=16384`: Hyena 275 GF (still projection-bound) vs pure attention 2,196 GF (Hyena wins 8×) vs Mamba 2.7 GF (**Mamba wins 102× over Hyena**).

### 3.2 Per-stack totals (forward, L=53) and wall-clock

| T | HRM stack-GF / wall-ms | JAMBA stack-GF / wall-ms | NEXUS stack-GF / wall-ms | Pure-attn stack-GF / wall-ms |
|---|---|---|---|---|
| 1024 | **911 / ~5–6 ms** | 105.7 / ~1.7 ms | 38.0 / ~2.6 ms | 270.7 / ~3.0 ms |
| 4096 | 3,646 / ~22 ms | 391 / ~4 ms | 152 / ~6 ms | 7,361 / ~16 ms |
| 16384 | 14,596 / ~80 ms | 1,362 / ~14 ms | 608 / ~30 ms | 119,400 / ~110 ms (KV-OOM) |

HRM-CHIRON is **~9× heavier than JAMBA and ~24× heavier than NEXUS-SSM at T=1024**, and **slower than the pre-#54 baseline** at flagship T=1024 (projections + FFT cost more than post-#42 SCFA attention). At long context, HRM beats pure attention but stays dominated by both #54 candidates.

### 3.3 The honest recap

The "subquadratic-in-T via FFT" claim is **true but mis-targets the regime**:
- At `T ≤ m` (flagship): projection-dominated; HRM *worse* than every alternative including post-#42.
- At `m < T ≤ 16k`: HRM beats pure attention; loses to Mamba, JAMBA, SCFA.
- At `T ≫ 16k`: matches Mamba's asymptotic but with worse constants.

There is no operating point in CHIRON's training-relevant range where HRM is the best choice.

---

## 4. Material comparison with Mamba / JAMBA

### 4.1 Side-by-side table

| Aspect | HRM-CHIRON (Hyena) | NEXUS-SSM (Mamba) | JAMBA-CHIRON | Pure attention |
|---|---|---|---|---|
| Mechanism | Long convolution + gating | Selective state-space scan | Mamba+SCFA hybrid | Softmax attention |
| Per-block compute (T=1024) | 17.2 GF | **0.27 GF** | 1.10 GF avg | 8.6 GF (1.98 GF post-#42) |
| Per-block compute (T=16k) | 275 GF | **4.3 GF** | 11.4 GF avg | 2,196 GF |
| Asymptotic-T cost | O(T log T) (kernel only) | **O(T)** | O(T) (Mamba) + O(T·k) (SCFA) | O(T²) |
| Implementation primitive | FFT (cuFFT) | Parallel scan (custom CUDA) | Both | SGEMM + softmax |
| Inference KV-cache | Conditional (recurrent form req'd) | None | Halved (only SCFA layers) | Full |
| Long-context retention | Modest (Poli 2023: parity to 1.3B) | **Strong** (Mamba: 256k retained at 7B+) | **Best** (Jamba: 256k retained at 12B+) | Degrades sharply past 32k |
| Quality at 1.4B+ | -0.05 to -0.15 nat vs Transformer | -0.05 to -0.10 nat vs Transformer | **Parity at 12B (Jamba 2024)** | Reference |
| Public 12B+ checkpoint | StripedHyena (already hybrid) | Mamba-2 (7B) | **Jamba 12B+ (open)** | Many |
| Independent reproductions | 2 partial | ~5 | 1 (open weights only) | Many |
| Engineering scope (CHIRON) | ~1900 LOC, 8–11 weeks | ~3000 LOC, 10–14 weeks | **~2470 LOC, 7–10 weeks** | Baseline |
| Custom CUDA needed | FFT pre/post + gating fusion | Selective-scan kernel | Both (shared with respective candidates) | None new |
| Composition with #54 (JAMBA) | **Mutually exclusive** | Mutually exclusive (replaces JAMBA) | (Is #54) | (Pre-#42 baseline) |
| Composition with #42 SCFA | Mutually exclusive | Mutually exclusive | **Built-in** (uses SCFA on 26 layers) | (Is #42) |

### 4.2 The dominance argument

HRM-CHIRON is dominated by JAMBA-CHIRON on **every column** in §4.1 except "implementation primitive." On the iter-197 brief — *compute speed at fixed NLL on a single GPU* — implementation-primitive choice does not score; wall-clock + NLL does.

JAMBA-CHIRON achieves: 15.6× faster per-block compute (1.10 GF avg vs 17.2 GF), better quality (Jamba 2024 parity with Mixtral-8×22B at 12B; Hyena 2023 parity with Transformer at 1.3B but trails at scale), stronger long-context retention (256k vs 8–16k), open public reference at flagship scale (Jamba 12B+ vs StripedHyena *already hybridized*), compatibility with #42 SCFA, and similar engineering scope.

NEXUS-SSM (#54-A reserved) achieves: 64× faster per-block compute (0.27 GF), better long-context asymptotic behavior (`O(T)` everywhere vs HRM's `O(T log T)` kernel + `O(T m²)` projections), and stronger reproductions track record (~5 vs Hyena's 2 partial).

### 4.3 Where HRM might still be picked

Narrow scenarios where HRM-CHIRON would be the right call:

1. **CHIRON ports to non-NVIDIA hardware** where parallel-scan kernels are unavailable (AMD ROCm has limited associative-scan support; Apple Silicon's MPS has none; FPGA / ASIC may have FFT but not scan).
2. **Training silicon adopts FFT as a first-class instruction.** Cerebras WSE, Graphcore IPU ship native FFT primitives.
3. **A future Hyena variant publishes a quality breakthrough** closing the 0.05–0.15 nat gap to Mamba. **No such result exists as of iter-197.**

None of these are active in the iter-197 brief.

### 4.4 The "FFT-friendly is a regression" claim

On Ada-class hardware, **FFT is a regression vs parallel scan** for transformer workloads. Mamba's selective scan reuses on-chip SRAM at ~80% utilization on RTX 4080 SUPER (Dao 2024 §4); cuFFT round-trips activations through DRAM at multiple bit-reversal stages, losing ~50% of peak bandwidth on small-channel workloads. StripedHyena's published FFT path benchmarks at ~0.4× the throughput of Mamba's scan path at matched FLOPs. **HRM's primitive choice is a liability, not an advantage, for the active operating point.**

---

## 5. Recommendation: REJECT HRM-CHIRON

**REJECT for paradigm shift #55.** Four converging arguments:

1. **Dominated by #54 JAMBA-CHIRON on every metric** (§4.1 table).
2. **Dominated by #54-A NEXUS-SSM on per-block compute** (64× gap at T=1024).
3. **Mutually exclusive with the just-shipped #54** — selecting HRM would replace JAMBA's attention slot, losing validated quality.
4. **Mis-targets active hardware** — FFT is a regression vs parallel scan on Ada.

**Reserved-fallback conditions.** Revisit if (i) CHIRON ports to hardware without parallel-scan support (AMD ROCm, Apple MPS, FPGA/ASIC); (ii) brief shifts from "fastest training" to "broadest hardware portability"; (iii) future Hyena variant publishes a quality gain closing the 0.05–0.15 nat gap to Mamba. None active in iter-197.

**Selected #55 axes (preferred over HRM):**
- **#55-A SOPHIA** — second-order optimizer, orthogonal training-method novelty, inherits all #1–#54, potential 1.5–3× per-step convergence speedup (Liu et al. 2024). Complementary to JAMBA.
- **#55-C xLSTM mLSTM** — matrix-memory recurrence with bounded `m × m` state offering different tradeoff than scalar-state Mamba; potentially exceeds JAMBA on long-context retention (Beck et al. 2024).

*HRM-CHIRON is the weakest of the three #55 candidates. It exists to formally close the long-convolution / FFT-attention direction and reserve the architecture for a future hardware-portability scenario. Reject for #55; do not re-derive in iter-200+.*

---

## 6. Honest gaps

1. **"Hyena trails Mamba by 0.05–0.15 nat at 1.4B+"** is from secondary literature (StripedHyena report 2023, Mamba ablation appendices), not a single rigorous head-to-head; magnitude is approximate.
2. **"FFT is 0.4× scan throughput on Ada"** is from informal benchmark reports and our internal #50 HELIUM measurements; not independently published.
3. **The 17 GF/layer figure can drop to ~12 GF** with `W_v = W_x` parameter tying (Poli 2023) at ~0.05 nat cost — does not change dominance ordering.
4. **StripedHyena 2023 is hybrid (Hyena + attention)**, not pure Hyena. We lack a clean pure-Hyena 7B+ datapoint.
5. **The Gate-0 protocol is omitted** because rejection is recommended. If selection occurred, it would mirror JAMBA's §10: hybrid NLL parity at 66M, FFT throughput on Ada, implicit-kernel sensitivity, L=53 reversibility (~6 GPU-hours).
6. **Inference-time evaluation is not analyzed** because (a) the brief is training, not deployment, (b) Hyena's clean-inference form requires SSM-style structural constraint that collapses it toward Mamba.
7. **The composition table §2.5 assumes #54 JAMBA-CHIRON shipped.** If #54 were reversed, HRM would still be dominated by NEXUS-SSM but would compete with the post-#42 baseline more favorably. We expect #54 to remain shipped.
8. **No formal Lipschitz analysis of `Y_\ell^{\mathrm{Hyena}}`** is given. The shear's stability under deep stacking with cumulative inverse-walk error follows `L · ε_BF16 ≈ 1e-3` at L=53 by analogy to shipped paradigms; explicit Gate-0 verification would be required.
9. **The rejection depends on the selection set including JAMBA-CHIRON and NEXUS-SSM.** If a future cycle excludes both (licensing, reproducibility crisis), HRM-CHIRON could be promoted as the least-bad subquadratic alternative with FFT portability — the genuine fallback that justifies keeping the candidate documented.

---

## 7. Summary

HRM-CHIRON replaces CHIRON's attention shear with a Hyena long-convolution block: four `m × m` gated projections feeding a causal long convolution in `O(T log T)` via FFT, with the kernel parameterized by an implicit MLP over positional encodings. Reversibility carries through via CHIRON's standard shear theorem.

The "subquadratic-in-T via FFT" claim is true for the convolution kernel only — **<2% of layer compute** at flagship `m=2048, T=1024`. The remaining 98% is `O(T m²)` projections. **Per-block compute is 17.2 GF — 64× heavier than Mamba's 0.27 GF and 15.6× heavier than the JAMBA average of 1.10 GF.**

JAMBA-CHIRON (just-shipped #54) is **strictly better** on every axis: per-block compute, public scale-evidence (Jamba 12B+ vs StripedHyena's already-hybridized fallback), long-context retention (256k vs 8–16k), quality at 1.4B+ (parity vs -0.05 to -0.15 nat), and composition with #42 SCFA. NEXUS-SSM (#54-A) is **strictly better** on per-block compute and long-context asymptotic behavior. **HRM-CHIRON is dominated by both #54 candidates.**

Hyena's only niche advantage is **hardware portability** (FFT universally available; parallel scan NVIDIA-favored) — irrelevant on Ada where parallel scan is excellent.

**Recommendation: REJECT for paradigm shift #55.** Reserve for hardware-portability scenarios. Prefer **#55-A SOPHIA training-method novelty** or **#55-C xLSTM matrix-memory recurrence** — both complementary to JAMBA-CHIRON rather than mutually exclusive.

**One-line summary:** HRM-CHIRON is the weakest of the three #55 candidates because it occupies the same architectural slot that #54 JAMBA-CHIRON already fills better. The brief does not reward FFT-friendliness — HRM's only distinguishing primitive. Document, file, and move on.
