# Paradigm Shift #54 Candidate B — JAMBA-CHIRON (Hybrid Mamba × SCFA-Attention × MOSAIC-MOE inside the Symplectic Shear)

**Status:** candidate-B design; one of three parallel proposals for paradigm shift #54.
**Date:** 2026-05-08 (Ralph-loop iter 198, post-#53 MOSAIC-MOE selection, under the iter-197/198 brief: *"Ideally we invent novel LLM architectures, algorithms, and training methods."*).
**Axis:** **hybrid architecture-class shift** — alternate the inside of CHIRON's symplectic shear between **selective state-space (Mamba)** layers and **spectrally-compressed softmax-attention (SCFA, paradigm #42)** layers, with the FFN slot using either the dense MELT-TT backbone or the post-#53 MOSAIC-MOE shear on alternating blocks. Modeled on AI21 Labs' **Jamba** (Lieber et al. 2024).
**Tagline.** *Replace half the softmax attention with Mamba state-space scans, and let the surviving attention pay the retrieval / induction-head bill that pure SSM under-services. Compose with #42 SCFA on the attention slots, with #53 MOSAIC-MOE on the FFN slots, and inherit CHIRON reversibility through both because every shear's `Y(q)` remains a continuous function of `q` alone.*

**Materially distinct from competing #54 candidates A and C:**
- **NEXUS-SSM (#54-A)** — replaces **all** attention with selective SSM. Architecturally pure; loses sharp retrieval / induction-head behaviour empirically.
- **JAMBA-CHIRON (this doc, B)** — keeps half of CHIRON's attention layers (now SCFA-compressed) and replaces the other half with Mamba blocks. Adds MOE FFN on every-other block. **Hybrid by construction.** Jamba 2024's 12B hybrid Mamba+attention model matched Mixtral-8×22B on NLL at lower compute and **beat** both pure-Mamba and pure-Transformer on long-context retention.
- **Candidate C (separate doc)** — independent third proposal; not analyzed here.

**Materially distinct from prior shipped paradigms:** uses #42 SCFA (compressed attention on surviving attention layers), #44 MELT (TT-FFN backbone on dense slots and per-expert MOE backbone), #46 REFLECTOR (cotangent-lift exact gradient through both shear types), and #53 MOSAIC-MOE (sparse expert FFN on every-other FFN slot). Does **not** mutually exclude #42 like NEXUS-SSM does — JAMBA-CHIRON *consumes* SCFA on its 26 surviving attention layers.

---

## 0. Executive summary (HONEST claim)

The architecture is

$$
\text{Block}_\ell = \begin{cases}
\text{Mamba shear} \cdot \text{dense-MELT FFN shear}, & \ell \bmod 4 = 0, \\
\text{SCFA shear}  \cdot \text{MOE FFN shear},        & \ell \bmod 4 = 1, \\
\text{Mamba shear} \cdot \text{MOE FFN shear},        & \ell \bmod 4 = 2, \\
\text{SCFA shear}  \cdot \text{dense-MELT FFN shear}, & \ell \bmod 4 = 3.
\end{cases}
$$

At `L=53`: 27 Mamba shears + 26 SCFA shears, with MOE on 26 FFN slots and dense-MELT on 27 FFN slots. The dispatch is `constexpr` of `ℓ`, so #51 ATLAS-COMPILE CUDA-Graph capture remains static.

**Headline claim.** **Jamba-class quality** (Lieber et al. 2024 reports parity with Mixtral-8×22B at lower compute, with better long-context retention than either pure baseline) at **1.6–2.2× per-step wall-clock** vs pre-#54 at flagship `T=1024`, scaling to **5–7×** at `T=8192` and **qualitatively unblocking** `T=32768+` training. **Less aggressive than NEXUS-SSM at long context** (NEXUS-SSM gets ~10× at T=8192) but **more aggressive at short context** (NEXUS-SSM is only ~1.15× at T=1024 post-Amdahl), and **lower NLL risk** because half of the attention machinery survives.

**Headline figures at flagship 1.84B-active:**
- Per-step wall-clock at T=1024: **1.6–2.2×** (half-attention savings + MOE-FFN savings + KV-cache halving).
- Per-step wall-clock at T=8192: **5–7×**.
- Per-step wall-clock at T=32768: **15–25×** (pure attention would be infeasible).
- KV-cache memory: **halved** (only 26 SCFA layers carry KV-cache; Mamba layers carry tiny per-block SSM state).
- Effective parameter count via #53: **~4–5× active params** (vs full-MOE's 8×; tradeoff for routing stability).
- NLL: **competitive but not bit-exact**; ≤ 0.10 nat behind pure-Transformer-CHIRON; **better** than NEXUS-SSM on short-context retrieval; **better** than pure-Transformer on long-context retention.

**Stack at 18B-active, T=1024:** `917× × 1.8 ≈ 1650×` wall-clock at ~4× effective params (≈ 72B effective).
**Stack at 18B-active, T=8192:** `917× × 6 ≈ 5500×` wall-clock; long-context training tractable on a single GPU.

**Engineering scope.** ~2470 LOC, 7–10 weeks. ~35% CUDA (Mamba selective-scan), 30% C++ glue (per-layer dispatch), 20% MOE wiring, 15% testing.

**Risk profile.** **Medium empirical** — Jamba pattern is publicly validated at 12B+; reversibility re-derivation needed for both shear types but is structural per Theorem 3 of #42. **Medium structural** — two independent kernel paths to maintain permanently increases the testing surface.

---

## 1. Primitive objects

Standard CHIRON dims: `m=2048`, `T ∈ {1024, 8192, 32768}`, `L=53`, `n_H=16` (SCFA blocks), `d_H=128`. New JAMBA-CHIRON dims: per-Mamba-block state dim `N=16` (Mamba default per Gu & Dao 2023); per-SCFA-block sequence-spectral rank `k = max(64, T/16)` (per #42); layer-pattern period `P=4`.

**Layer-pattern indicator.**

$$\tau(\ell) := \begin{cases}
(\text{Mamba}, \text{dense}), & \ell \bmod 4 = 0, \\
(\text{SCFA},  \text{MOE}),   & \ell \bmod 4 = 1, \\
(\text{Mamba}, \text{MOE}),   & \ell \bmod 4 = 2, \\
(\text{SCFA},  \text{dense}), & \ell \bmod 4 = 3.
\end{cases}$$

At `L=53`: 27 Mamba shears, 26 SCFA shears, 26 MOE FFN shears, 27 dense-MELT FFN shears.

**Per-Mamba-block parameters** (per #54-A §1): `A_log_ℓ` (ℝ^{m×N}, BF16, **not** ternarized per #47), `W_B^ℓ, W_C^ℓ` (ℝ^{m×N}, ternary candidates), `W_Δ^ℓ` (ℝ^{m×m_Δ} + ℝ^{m_Δ×m} low-rank, `m_Δ=128`, ternary), `b_Δ^ℓ` (ℝ^m, BF16), `D_ℓ` (ℝ^m, BF16). Per-layer: ~590 K params (vs SCFA shear's ~16.8 M — Mamba is **30× smaller**).

**Per-SCFA-block parameters** (per #42 §2): `B_ℓ` (ℝ^{T×k} spectral basis), `W_Q/K/V/O^ℓ` (ℝ^{m×m} each), `D_ℓ^conv` (depthwise conv, kernel `2w+1`, `w=8`).

**Per-MOE-FFN-block parameters** (per #53 §1): one shared MELT-TT backbone, router `W_r^ℓ ∈ ℝ^{m×E}` with `E=8, k=2`, per-expert LoRA `(A_e, B_e) ∈ ℝ^{m×r} × ℝ^{r×m}` with `r=4`.

**Per-dense-MELT-FFN-block parameters:** standard MELT TT cores per #44.

---

## 2. The hybrid shear: mathematics and reversibility

### 2.1 Two shear types in one stack

CHIRON's per-block symplectic shear is `Φ_\ell : (q, p) \mapsto (q, p + Y_\ell(q))` with `Y_\ell` chosen per-layer from `\{Y_{\text{Mamba}}, Y_{\text{SCFA}}\}` according to `τ(ℓ)`.

**Theorem 1 (Hybrid reversibility).** *For any continuous `Y_\ell : ℝ^{T×m} \to ℝ^{T×m}`, the shear `Φ_\ell` is bijective with explicit inverse `(q', p') \mapsto (q', p' - Y_\ell(q'))`, preserving the symplectic form `ω = dq ∧ dp` and `det DΦ_\ell = 1`. The choice of `Y_\ell ∈ \{Y_{\text{Mamba}}, Y_{\text{SCFA}}\}` is irrelevant to the reversibility argument.* □

This is Theorem 3 of #42 SCFA — `Y`'s internal class never enters the proof. **JAMBA-CHIRON's hybrid pattern is immediately reversible.**

### 2.2 Mamba and SCFA shears (recap)

For `τ(ℓ).attn = \text{Mamba}` (per #54-A §3):

$$h_{t,n}^{(\ell)} = e^{Δ_t^{(\ell)} λ_n^{(\ell)}} h_{t-1,n}^{(\ell)} + \frac{e^{Δ_t^{(\ell)} λ_n^{(\ell)}} - 1}{λ_n^{(\ell)}} B_{t,n}^{(\ell)} q_t,$$

$$Y_\ell^{\text{Mamba}}(q)[t] = \sum_n C_{t,n}^{(\ell)} h_{t,n}^{(\ell)} + D^{(\ell)} \odot q_t.$$

Forward FLOPs at T=1024, m=2048, N=16: ~270 MFLOPs/layer (per #54-A §2).

For `τ(ℓ).attn = \text{SCFA}` (per #42 §4):

$$Y_\ell^{\text{SCFA}}(q) = B_\ell \cdot \text{Attn}_{\text{compr}}(B_\ell^{\top} q;\, W_Q, W_K, W_V) \cdot W_O + D_\ell^{\text{conv}}((I-B_\ell B_\ell^{\top}) q).$$

Forward FLOPs at T=1024, m=2048, k=64, n_H=16: ~1.98 GFLOPs/layer (per #42 §7).

### 2.3 The composed JAMBA-CHIRON block

Each block is a two-shear sequence: attention slot then FFN slot. Both shears keep `q` fixed, so any number compose into `(q, p) \mapsto (q, p + \sum_i Y^{(i)}(q))`. The standard CHIRON block-end coordinate swap `(q, p) \mapsto (p, q)` is unchanged.

### 2.4 Why MOE on every-other and not every

Per Lieber et al. 2024 §3.4: MOE on every block degrades stability at modest scale (≤ 12B) because routing variance compounds across layers. Jamba uses 1:1 MOE-to-non-MOE at 12B+; we follow the same prescription at CHIRON-1.84B. Mechanism: dense-FFN gradient flow stabilizes routing-loss variance; the non-MOE layers act as gradient anchors. **Honest caveat.** This is a **prior, not a measurement**, for CHIRON. Validation belongs to Gate-0 §10.

### 2.5 Routing reproducibility under inverse walk

**Theorem 2.** *In the JAMBA-CHIRON stack, the inverse-walked `q'` equals the forward `q` exactly (FP32; BF16-rev-tolerance). The MOE router `softmax(W_r q_t)` and #53 index-order tie-breaking produce identical expert assignment in forward and inverse passes.* □

Proof sketch: every shear keeps `q` fixed; the block-end swap is involutive. The router and tie-breaking are deterministic functions of `q` alone (per #53 §2). □

This is the load-bearing claim of JAMBA-CHIRON × #53 compatibility, inherited unchanged from #53 because JAMBA-CHIRON does not perturb the `q`-flow.

---

## 3. Compute analysis at multiple T

### 3.1 Per-layer FLOPs (forward)

| Layer type | T=1024 | T=8192 | T=16384 | T=32768 |
|---|---|---|---|---|
| Mamba shear | 0.27 GF | 2.16 GF | 4.32 GF | 8.64 GF |
| SCFA shear (k=max(64, T/16)) | 1.98 GF | 7.4 GF | 19.0 GF | 49.5 GF |
| Pure attention shear | 8.6 GF | 549 GF | 2,196 GF | 8,786 GF |
| Dense-MELT FFN | 1.4 GF | 11.2 GF | 22.4 GF | 44.8 GF |
| MOE FFN (k_MOE/E=1/4) | 0.35 GF | 2.8 GF | 5.6 GF | 11.2 GF |

### 3.2 Per-stack totals (forward, L=53)

| T | JAMBA-CHIRON total | Pure-attention CHIRON | NEXUS-SSM total |
|---|---|---|---|
| 1024 | **105.7 GF** | 270.7 GF (2.6×) | 38.0 GF |
| 8192 | **626 GF** | 30,036 GF (48×) | 304 GF |
| 16384 | **1,362 GF** | 119,400 GF (88×) | 608 GF |
| 32768 | **3,021 GF** | 477,000 GF (158×) | 1,217 GF |

### 3.3 Per-step wall-clock (memory-bandwidth-realistic on Ada, post-#50 + #52)

| T | JAMBA-CHIRON | NEXUS-SSM | Pure-attention CHIRON |
|---|---|---|---|
| 1024 | **~1.7 ms** | ~2.6 ms | ~3.0 ms (pre-#54 baseline) |
| 8192 | **~7 ms** | ~15 ms | ~37 ms |
| 16384 | **~14 ms** | ~30 ms | ~110 ms (KV-cache OOM at 1.84B) |
| 32768 | **~32 ms** | ~65 ms | infeasible (T² scratch ≈ 16 GB) |

JAMBA-CHIRON is **faster than NEXUS-SSM at short context** because surviving SCFA layers reuse the post-#50 FA-3 kernel. At long context the picture inverts — NEXUS-SSM is faster because every layer is O(T·m·N), while JAMBA's surviving SCFA grows as `O(T·k·m) ≈ O(T²·m/16)` (still asymptotically quadratic, even if 16× better than pure attention).

**Honest framing.** JAMBA-CHIRON is the **better** candidate for `T = 1024–8192` (current flagship + reasonable expansion) at moderate quality buy-back. NEXUS-SSM is the **better** candidate for `T ≥ 32768` where qualitative capability dominates incremental quality.

### 3.4 Composition with pre-#54 stack

| Pre-#54 paradigm | Composition | Multiplier retained |
|---|---|---|
| **#42 SCFA** | Built in (used on 26 surviving attention layers) | ≈ 0.5× the SCFA savings |
| **#44 MELT** | Dense-FFN backbone + per-expert MOE backbone | Full |
| **#46 REFLECTOR** | Two cotangent-lift pathways (Mamba, SCFA) | Full |
| **#47 PHOENIX-1.58BIT** | Full on SCFA W_Q/K/V/O; partial on Mamba (skip A, ternarize W_B/C/Δ) | ≈ 0.8× |
| **#48 STREAM-CHIRON** | Both shear types similar exposed-activation | Full |
| **#50 HELIUM** | FA-3 on 26 SCFA layers; FP8 on both | ≈ 0.5× of FA-3 component, full FP8 |
| **#51 ATLAS-COMPILE** | Static layer pattern; CUDA Graph clean | Full |
| **#52 NIMBUS** | Optimizer parameter-agnostic | Full |
| **#53 MOSAIC-MOE** | Built in on every-other FFN slot | ≈ 0.5× FLOP reduction; full capacity multiplier on MOE layers |
| **#28 FACE** | Embedding full; SCFA W_Q/K/V/O full; Mamba W_B/C empirical | ≈ 0.85× |

Stack composition (multiplicative): `917× × 1.8× ≈ 1650×` at T=1024; `917× × 6× ≈ 5500×` at T=8192; `917× × 20× ≈ 18,000×` at T=32768 (qualitative unblock).

---

## 4. Composition with #53 MOSAIC-MOE: detailed

### 4.1 Every-other-block MOE pattern

26 MOE FFN layers `\{1, 2, 5, 6, 9, 10, ..., 49, 50\}`; 27 dense-MELT layers (the complement). Pattern repeats every 4 layers via `τ(ℓ)`.

### 4.2 MOE compute on alternating layers

- 27 dense-MELT FFN layers @ 1.4 GF each = 37.8 GF.
- 26 MOE FFN layers @ 0.35 GF each = 9.1 GF.
- Combined: 46.9 GF (vs all-dense: 74.2 GF, vs all-MOE: 18.2 GF).
- Net FFN compute reduction: ~1.6× vs all-dense; **0.4× retained** of full-MOE FLOP reduction.

### 4.3 Capacity multiplier (halved vs pure-MOE)

Per-MOE-layer effective parameters: ~8× (per #53 §3). Stack-level: `27/53 + (26/53) × 8 ≈ 4.4×`. **1.84B-active becomes ~8.1B-effective** (vs 14.7B for all-MOE per #53; ~half).

**Honest tradeoff.** JAMBA-CHIRON gives **half the MOE capacity benefit** of pure MOSAIC-MOE in exchange for the alternating-stability buffer (§2.4).

### 4.4 Engineering: alternating dispatch

```cpp
for (int l = 0; l < cfg.numLayers; ++l) {
    LayerType lt = jamba_layer_type(l);  // constexpr lookup
    if (lt.attn == ATTN_MAMBA) {
        glades::gpu::selective_scan_forward(...);
    } else {
        glades::gpu::scfa_forward(...);
    }
    if (lt.ffn == FFN_MOE) {
        glades::gpu::moe_ffn_forward(...);
    } else {
        glades::gpu::dense_melt_ffn_forward(...);
    }
    swap_qp(...);
}
```

`jamba_layer_type` is `constexpr` of `l`; ATLAS-COMPILE captures as static graph. **No runtime dispatch overhead.**

---

## 5. Material distinction from NEXUS-SSM (#54-A)

| Aspect | NEXUS-SSM | **JAMBA-CHIRON** |
|---|---|---|
| Attention mechanism | All Mamba | 27 Mamba + 26 SCFA alternating |
| Long-context (T ≥ 32k) | Excellent (uniform O(T·m·N)) | Mixed; surviving SCFA still O(T²·m/16) |
| Short-context (T ≤ 1024) | Modest (~1.15× per-step) | **Better: ~1.8× per-step** (FA-3 reuse) |
| NLL quality | Mamba-class (≤ 0.10 nat at 1.4B; parity at 7B+) | **Jamba-class** (parity with Mixtral-8×22B at 12B; better long-context retention) |
| Engineering | One new architecture | **Two new dispatch paths** (more complexity) |
| Composition with #42 SCFA | Mutually exclusive | **Built in** (uses SCFA on 26 layers) |
| Composition with #50 FA-3 | N/A | Full FA-3 on 26 SCFA layers |
| Composition with #53 | Full (every layer can carry MOE) | Half (every-other-block) |
| KV-cache at flagship | **Eliminated** | Halved |
| Engineering scope | ~3000 LOC, 10–14 weeks | **~2470 LOC, 7–10 weeks** |
| Empirical risk | Medium (Mamba-inside-CHIRON novel) | **Lower** (Jamba pattern publicly validated at 12B+; attention-fallback for retrieval-heavy survives) |
| Failure mode | Sharp-retrieval task regression | Routing instability + load-imbalance compounding (mitigated per #53 §4.5) |
| Fallback | Hybrid mode (= this candidate) | Pure-attention fallback |

**The decisive distinction.** NEXUS-SSM is a **pure-architecture bet**; if Mamba-inside-CHIRON holds, it wins on long-context unblock. JAMBA-CHIRON is a **diversified bet**: half is publicly-validated Jamba 2024 (which retains attention's induction-head behavior on SCFA layers and Mamba's long-context retention on Mamba layers); the other half is organic composition with #53. Expected-value of JAMBA-CHIRON is **higher** because the loss case is bounded; upside is **lower** because long-context unblock is partial (limited by surviving SCFA's `O(T²/16)` growth).

**Mathematical novelty.** JAMBA-CHIRON is the **first** architecture to compose three different `Y(q)` mechanisms (Mamba, SCFA, MOE) inside one CHIRON stack. The reversibility argument applies to the *combination* — the inverse walk must dispatch correctly across all three mechanism types per layer. NEXUS-SSM has one mechanism inside the shear; JAMBA-CHIRON has three, all interleaved in a static pattern.

---

## 6. Empirical evidence and external grounding

### 6.1 Jamba 2024 (Lieber et al.)

AI21 Labs released **Jamba** (Lieber et al. 2024, "Jamba: A Hybrid Transformer-Mamba Language Model"), a 12B-active / 52B-total MoE model with **1:7 attention:Mamba ratio** and MOE on every other Mamba layer. Headline:
- Parity with Mixtral-8×22B on standard NLL benchmarks (HellaSwag, ARC, BoolQ).
- **Better** than Mixtral on long-context retention (256k context evaluated; Mixtral degrades sharply past 32k).
- Open-weight on HuggingFace; Jamba-1.5 follow-up (late 2024) confirms the pattern at larger scales.

JAMBA-CHIRON adopts a **similar but not identical** pattern: 1:1 attention:Mamba (vs Jamba's 1:7) — the conservative-hybrid sweet spot for CHIRON's smaller flagship. Jamba's 1:7 targets 12B+ where attention's quadratic cost dominates; CHIRON-1.84B is small enough that attention is still affordable, so we keep more of it.

### 6.2 Mamba (Gu & Dao 2023, Dao & Gu 2024)

Already covered in NEXUS-SSM #54-A §6.2: Mamba-1.4B is +0.05 nat behind Transformer; Mamba-2.8B parity; Mamba-7B+ competitive. JAMBA-CHIRON inherits this for its 27 Mamba shears. The other 26 SCFA shears address the empirically-known Mamba weakness (sharp retrieval / induction heads).

### 6.3 MoE (Switch, Mixtral, DeepSeek-V3)

Already covered in #53. JAMBA-CHIRON inherits this for its 26 MOE FFN layers.

### 6.4 The composition argument

The novel claim is that **all three patterns compose without interaction**. Plausible because:
- Mamba and SCFA never share a layer.
- MOE and dense-MELT never share a layer.
- Jamba 2024 validates *some* of this (Mamba + attention + MOE) at 12B; the *exact* layout `(Mamba+dense, SCFA+MOE, Mamba+MOE, SCFA+dense)` is JAMBA-CHIRON's contribution.

**Honest gap.** No public evidence that this *exact* layer pattern is optimal for CHIRON-1.84B-reversible. We claim only that Jamba 2024's general pattern is empirically validated at scale, and we adopt it conservatively.

---

## 7. Stability and CHIRON invariants

- **Per-shear stability** inherited from #42 (SCFA Lipschitz bound via `B B^T` projection norm) and #54-A (Mamba `A = -exp(A_log)` parameterization → contractive recurrence).
- **Cross-shear interaction:** Lipschitz constants **add** across shears (since they compose linearly via `p ← p + Σ_i Y^{(i)}(q)`). No new explosion mode.
- **Numerical reversibility under deep stacks:** cumulative inverse-walk error ~`L · ε_BF16 ≈ 1e-3` at L=53 — same as pre-#54.
- **Routing stability:** load-balance loss applies only to MOE layers; non-MOE layers act as gradient anchor (§2.4). **Plausible, not measured** — Gate-0 Probe C addresses.
- **BF16 dynamic range:** two stability tripwires per layer:
  - Mamba: `exp(Δ λ)` near 0 → state never decays (mitigation: keep `(A_log, Δ)` in FP32 per #54-A §8.3).
  - SCFA: softmax overflow at long T (mitigation: standard FA-3 stabilization per #50 HELIUM).
  Both inherited independently. **No new BF16 hazard from hybridization itself**, but two tripwire sets to maintain.

---

## 8. Concrete primitives summary

```
Backend/Machine Learning/Networks/cuda/
  gpu_ssm_scan.h               (new, ~400 LOC; SHARED with NEXUS-SSM if both ship — but #54 selects one)
  gpu_ssm_scan.cu              (new, ~1500 LOC: forward/backward/inverse-walk)
  gpu_jamba_dispatch.h         (new, ~80 LOC: layer-pattern enum + per-layer config)
  gpu_jamba_dispatch.cu        (new, ~120 LOC: constexpr per-layer dispatch)

Backend/Machine Learning/Networks/
  sgd_transformer.cpp          (~150 LOC delta)
  transformer_infer.cpp        (~100 LOC delta)
  transformer_generate.cpp     (~50 LOC delta)
  training_config.h            (~50 LOC: JambaConfig)

Backend/Machine Learning/MLState/
  jamba_state.h                (new, ~120 LOC)

unit-tests/Backend/Machine Learning/
  jamba_chiron_test.cpp        (new, ~250 LOC)
  jamba_chiron_smoke_test.cpp  (new, ~200 LOC: 66M Gate-0)

run.sh                         (~40 LOC: --jamba flag + arguments)
```

Total new code: **~2470 LOC**. Modifications: **~350 LOC**. Public API: unchanged.

---

## 9. Hyperparameters

| HP | Default | Range |
|---|---|---|
| Layer-pattern period `P` | 4 | {2, 4, 8} |
| Mamba state dim `N` | 16 | {8, 16, 32, 64} |
| SCFA spectral rank `k` | max(64, T/16) | {32, 64, 128, 256} |
| MOE fraction `f_MOE` | 0.5 | {0.25, 0.5, 0.75, 1.0} |
| MOE expert count `E` | 8 | {4, 8, 16, 32} |
| MOE top-k `k_MOE` | 2 | {1, 2, 4} |
| Mamba lr divisor | 10 | {1, 5, 10, 20} |

---

## 10. Gate-0 falsifier protocol (~6 GPU-hours on 66M)

### 10.1 Probe A: Hybrid NLL parity (3 GPU-hours)

Replace half of attention shears (alternating) with Mamba blocks at matched param count; continue 5K steps. **Pass:** NLL within 0.10 nat of unmodified at step 55K. Fail → revisit pattern (drop to 1:3).

### 10.2 Probe B: Layer-pattern sensitivity (1 GPU-hour)

Compare {all-attention, 1:1, 1:3} at 2K steps. **Pass:** 1:1 within 0.05 nat; 1:3 within 0.02 nat.

### 10.3 Probe C: MOE-on-every-other vs MOE-on-every (1 GPU-hour)

Compare {all-MOE, 1:1 MOE:dense}. **Pass:** 1:1 within 0.03 nat with **lower** routing-loss variance.

### 10.4 Probe D: Cumulative reversibility at L=53 (1 GPU-hour)

Forward pass → inverse walk → measure `‖Φ^{-1}(Φ(q,p)) - (q,p)‖_∞`. **Pass:** ≤ 1e-3 in BF16 across L=53. Fail → escalate select Mamba parameters to FP32.

### 10.5 Decision rule

- All four pass: full JAMBA-CHIRON, ~7 weeks.
- A passes, B/C marginal: adjusted pattern, ~9 weeks.
- A fails but conservative 1:3 passes: scope to fallback, ~9 weeks.
- A and conservative both fail: reject for #54.

Total: ~6 GPU-hours, ~1.5 days engineering. Cheap relative to 7–10 week implementation.

---

## 11. Open math questions

1. **Optimal Mamba:attention ratio for CHIRON-flagship.** Jamba uses 1:7 at 12B; we propose 1:1 at 1.84B-active. Likely scales with model size; Probe B addresses.
2. **Layer-position dependence.** Early layers as Mamba (long-range mixing), late as attention (sharp retrieval)? Or position-independent? Open.
3. **Optimal MOE fraction.** JAMBA-CHIRON default 1:1; 2:3 or 1:2 may be better at smaller scales.
4. **`A_log` lr coupling with Adam + #28 FACE.** Mamba's published lr_A = lr/10 may need adjustment under FACE-modulated optimizer. Empirical.
5. **FACE on Mamba layers.** Per #54-A §4.6, Zipfian mechanism may not fire on Mamba weights.
6. **Long-context emergence at T = 32k+.** Does JAMBA-CHIRON exhibit Jamba's reported 256k retention behavior? Out of scope for Gate-0; relevant for downstream eval.
7. **Mamba-2 (Dao & Gu 2024) substitution.** SSD-class state-space duality is mathematically cleaner; v2 of JAMBA-CHIRON could swap with ~500 LOC.

---

## 12. Honest gaps

1. **"Jamba-class quality" is by analogy.** Jamba is at 12B; JAMBA-CHIRON is at 1.84B-active. The claim transfers by Mamba's known scale-quality monotonicity (smaller Mamba slightly worse vs Transformer; gap closes at 7B+); CHIRON-1.84B-Jamba may be 0.05–0.10 nat behind 1.84B-attention-CHIRON. Gate-0 measures.

2. **Two architectures interleaved is more engineering complexity than one.** NEXUS-SSM has one new kernel pathway; JAMBA-CHIRON has two (Mamba + SCFA, both new + post-#42 respectively) plus alternating MOE dispatch. **Bug surface, testing surface, and maintenance burden are permanently larger.** Justification: diversification (§5 closing) makes empirical risk lower, which is the load-bearing tradeoff.

3. **"1:1 MOE-to-dense as gradient anchor" is a prior, not a measurement.** Plausible per Jamba 2024 §3.4 but not directly measured for CHIRON. Probe C addresses.

4. **Long-context performance is bounded by surviving SCFA's residual quadratic.** 26 SCFA layers grow as `O(T·k(T)·m) = O(T²·m/16)` — better than `O(T²·m)` but still asymptotically quadratic. NEXUS-SSM's pure-Mamba is `O(T·m·N)` everywhere. **At T ≥ 65k, NEXUS-SSM strictly dominates JAMBA-CHIRON on compute.**

5. **MOE capacity multiplier is halved.** Pure MOSAIC-MOE → ~8× effective; JAMBA-CHIRON → ~4×. Deliberate tradeoff for routing stability; real cost vs the #53-pure path.

6. **"Best of both" framing is a marketing claim, not a theorem.** The diversification argument is plausible and supported by Jamba 2024 empirically, but is not a *theorem*. A pure architecture trained for the same compute may converge to better NLL than JAMBA-CHIRON; we have no proof otherwise.

7. **Reversibility check at L=53 with two shear types is a new validation surface.** Pre-#54 CHIRON validated reversibility with one shear type per layer. JAMBA-CHIRON has two shear types per layer (attention + FFN, alternating between Mamba/SCFA and dense/MOE). Per-shear tolerance budget is now `ε_{rev}/(L·2)` — tighter than pre-#54 but still within BF16 range; expected to pass, must be explicitly verified (Probe D).

8. **Gate-0 only validates at 66M, not flagship.** Public Mamba is at 1.4B+; public Jamba at 12B+. 66M is below the parity-emergence regime. Gate-0 establishes **direction of gradient**, not **magnitude at flagship**. Full validation requires multi-day 1.84B run.

9. **Jamba 2024 has not been independently reproduced** at the time of writing (early 2026) in any publicly-released codebase. Open-weight checkpoint exists; clean training-run reproduction does not. JAMBA-CHIRON's evidence base is **slightly thinner than NEXUS-SSM's** (Mamba has multiple independent reproductions; Jamba has one).

10. **The user has previously preferred bit-exact paradigms.** Iter-197 brief invited the break, but if the user reverts to bit-exact preference, both #54 candidates A and B are rejected and a third axis must be selected.

---

## 13. Summary

JAMBA-CHIRON is the **conservative-hybrid architectural shift** for CHIRON #54: alternate the inside of the symplectic shear between **selective state-space (Mamba)** layers and **spectrally-compressed softmax-attention (SCFA, post-#42)** layers, with the FFN slot using **MOSAIC-MOE (post-#53)** on alternating blocks. Modeled on AI21 Labs' Jamba 2024 and adapted to CHIRON's reversibility constraint via Theorem 1 of §2.1: `Y_\ell` need only be continuous, so any combination of Mamba/SCFA/MOE shears is automatically reversible.

The deliverable is **Jamba-class quality** (per public Jamba 2024: parity with Mixtral-8×22B at 12B, better long-context retention) at **1.6–2.2× per-step wall-clock** at flagship `T=1024`, scaling to **5–7×** at `T=8192`, with a **qualitative unblock** at `T=32768` where pure attention is infeasible. NLL is **competitive but not bit-exact** (≤ 0.10 nat behind pure-Transformer at 1.84B-active per Jamba 2024 transfer; better than NEXUS-SSM on short-context retrieval).

**Honest tradeoff vs NEXUS-SSM (#54-A):**
- **JAMBA-CHIRON wins** at short-context (T ≤ 8192), in retrieval-heavy workloads, and on engineering-risk grounds (half the architecture is publicly validated; loss case bounded by attention fallback).
- **NEXUS-SSM wins** at extreme-context (T ≥ 32768), in long-context unblock magnitudes, and on architectural-purity grounds.

Engineering scope is **~2470 LOC over 7–10 weeks**, smaller than NEXUS-SSM because half the new kernel work is shared and the SCFA + MOE infrastructure already exists from #42 and #53. Gate-0 (§10, ~6 GPU-hours) validates four premises — hybrid NLL parity, layer-pattern sensitivity, MOE-fraction stability, cumulative L=53 reversibility — before commit.

The decisive case **for** JAMBA-CHIRON over NEXUS-SSM is **diversification**: when underlying mechanism premises are empirically open (Mamba-inside-CHIRON-reversibility is novel), splitting the bet across two mechanism families bounds the loss case at the cost of a smaller upside in the win case. The decisive case **against** is **engineering complexity**: two interleaved kernel pathways permanently increase the testing surface and bug-fix overhead vs NEXUS-SSM's single-pathway simplification. The selection is a question of **risk tolerance and complexity budget**, not of "which is better at NLL" — both are competitive at LLM scale per public literature.
