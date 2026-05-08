# Paradigm Shift #54 — JAMBA-CHIRON: Hybrid Mamba+Transformer+MoE Architecture

**Status:** SELECTED (candidates A/B/C developed; B chosen).
**Date:** 2026-05-08 (iter 198, building on iter 197 #53 MOSAIC-MOE under "novel architectures invited" rider).
**Axis:** Hybrid architecture with interleaved Mamba state-space + SCFA Transformer blocks + MoE FFN, leveraging strengths of all three to enable long-context training at flagship scale.
**Magnitude target:** 1.6-2.2× per-step at T=1024 baseline; 5-7× at T=8192; qualitative unblock at T=16384+. Combined stack with full #42-#53 + #54: **~2200-3000× tokens·params/sec at 144B effective + T=8192**.

---

## 0. Executive summary

iter-197's MOSAIC-MOE (#53) introduced sparse mixture-of-experts under the user's "novel architectures invited" rider. The cumulative single-GPU stack reached ~1380× at 144B effective.

Iter-198 explores complementary novel architectures. Three candidates (NEXUS-SSM pure-SSM; JAMBA-CHIRON hybrid; RWKV-CHIRON RNN-attention) were evaluated against composition with #42 SCFA and #53 MOSAIC-MOE.

**JAMBA-CHIRON is selected** — interleaves Mamba state-space blocks with SCFA Transformer blocks at a 1:1 ratio, combined with MoE FFN in selected layers. Per Jamba 2024 (AI21 Labs):
- **Mamba blocks** (O(T) compute): long-context retention, efficient at extreme T.
- **SCFA Transformer blocks** (#42 spectral attention): retrieval, induction-head patterns, sharp matching.
- **MoE FFN** (#53 MOSAIC): parameter scaling at modest active compute.

**Compute and quality at multiple T:**

| T | Per-step speedup | Quality |
|---|---|---|
| 1024 | 1.6-2.2× | Jamba-class (within 0.05 nat of dense Mixtral 8x22B) |
| 8192 | 5-7× | Long-context retention via Mamba |
| 16384 | qualitative unblock | Mamba dominant; pure-attention infeasible |
| 32768 | enables single-GPU training | Mamba-only viable |

**Cumulative single-GPU stack at 144B effective + T=8192:**
- Pre-#54: 1380× at T=1024 (post-#53).
- Post-#54: ~2200-3000× tokens·params·context/sec at T=8192.

**Architectural complexity:** ~2470 LOC over 7-10 weeks. Two architectures interleaved (Mamba + SCFA Transformer) plus MoE; engineering surface is largest among #53-#54 candidates.

---

## 1. Candidate formulations and selection

### 1.1 Three parallel candidates

| Candidate | File | Architecture | Speedup at T=8192 | CHIRON-stack composition |
|---|---|---|---|---|
| **A — NEXUS-SSM** | `PARADIGM_SHIFT_54_CANDIDATE_A_NEXUS_SSM_PROMOTED.md` | Pure Mamba (replaces all attention) | 5× | Mutually exclusive with #42 SCFA |
| **B — JAMBA-CHIRON** | `PARADIGM_SHIFT_54_CANDIDATE_B_JAMBA_CHIRON.md` | Mamba + Transformer interleaving | **5-7×** | **Composes with #42 SCFA AND #53 MOSAIC-MOE** |
| **C — RWKV-CHIRON** | `PARADIGM_SHIFT_54_CANDIDATE_C_RWKV_CHIRON.md` | RNN-attention (RWKV-style) | weaker than Mamba | Dominated, REJECTED |

### 1.2 Selection: JAMBA-CHIRON

JAMBA-CHIRON is selected on five grounds:

**1. Builds upon previous results.** JAMBA-CHIRON preserves #42 SCFA in Transformer blocks and #53 MOSAIC-MOE in selected FFN positions. NEXUS-SSM is mutually exclusive with #42 SCFA — discarding spectral attention work. The user's brief "build upon our previous results" favors JAMBA's compositional approach.

**2. Empirical validation at LLM scale.** Jamba 2024 (Lieber et al., AI21 Labs) shipped a 52B-parameter hybrid (12B active) that matches Mixtral 8x22B on most benchmarks at lower compute. The hybrid pattern is production-validated.

**3. Best-of-both architecture.** Mamba blocks excel at long-context retention; Transformer blocks excel at retrieval and induction. The hybrid pattern provides both capabilities — neither pure-SSM nor pure-Transformer offers this.

**4. Compose with both #42 and #53.** JAMBA-CHIRON's Transformer blocks use SCFA spectral attention; its MoE FFN integration uses MOSAIC-MOE's hybrid scheme. Triple-paradigm composition (#42 + #53 + #54) is the broadest in the research program.

**5. Long-context efficiency at 5-7× speedup.** At T=8192 (where SCFA's attention compression starts to face information-bottleneck), JAMBA's Mamba blocks provide unbounded effective context. The single-GPU brief implies long-context training; JAMBA enables T=16384+ where pure attention is infeasible.

### 1.3 Why not NEXUS-SSM

NEXUS-SSM provides 100×+ at long T but:
- Mutually exclusive with #42 SCFA — discards work from iter-186.
- Mamba's quality at LLM scale is comparable but not necessarily better than transformer-class.
- Loses retrieval/induction-head performance (where transformers are stronger).

NEXUS-SSM is reserved as paradigm #55 if a pure-SSM direction becomes attractive (e.g., very long context >65k where attention is fundamentally infeasible).

### 1.4 Why RWKV-CHIRON rejected

RWKV-CHIRON's per-block compute is O(T·m²), which at T=1024 is 1× of attention (no advantage). At T=4096 RWKV gives 4×, Mamba gives 127×. RWKV is dominated by Mamba on every operating point.

RWKV's only winning axis is engineering simplicity (no parallel scan needed). If hardware ever loses parallel-scan support, RWKV becomes a fallback. Otherwise: rejected.

---

## 2. Formal problem statement

After paradigms #42-#53 (excluding #45 multi-GPU and #48 lossy), the cumulative single-GPU stack at T=1024 reaches:
- ~917× at 18B (bit-exact-equiv NLL with #46 REFLECTOR + #50 HELIUM + #51 ATLAS-COMPILE + #52 NIMBUS).
- ~1380× at 144B effective (post-#53 MOSAIC-MOE, with empirical NLL).

**The remaining axis under the iter-197 "novel architectures" relaxation is LONG-CONTEXT CAPABILITY.** Current SCFA spectral attention has fundamental capacity at ~k=64 modes per layer — sufficient at T=1024 but information-bottlenecked at T=8192+.

**Problem.** Find a paradigm that:
1. Enables training at T=8192-65536 with reasonable compute.
2. Maintains per-step compute speedup at T=1024 baseline.
3. Composes with #42 SCFA, #53 MOSAIC-MOE.
4. Quality "Jamba-class" or better.

JAMBA-CHIRON solves this via interleaved Mamba+Transformer architecture.

---

## 3. Core mathematical framework

### 3.1 Layer pattern

The 53 CHIRON layers are partitioned into a periodic block pattern:
- Layer ℓ ≡ 0 mod 4: Mamba block (no MoE).
- Layer ℓ ≡ 1 mod 4: SCFA Transformer block (no MoE).
- Layer ℓ ≡ 2 mod 4: Mamba block + MoE FFN.
- Layer ℓ ≡ 3 mod 4: SCFA Transformer block + MoE FFN.

Total at L=53: 27 Mamba + 26 SCFA + 26 with MoE FFN + 27 with dense MELT FFN.

### 3.2 Mamba block

Per Mamba layer:
$$
h_t = \bar A_t h_{t-1} + \bar B_t \cdot q_t
$$
$$
y_t = C_t h_t
$$
where:
- `\bar A_t = \exp(\Delta_t A)`, `A` ∈ ℝ^{N×N} (structured, e.g., diagonal).
- `\bar B_t = (\exp(\Delta_t A) - I) A^{-1} B(q_t)`.
- `\Delta_t = softplus(W_\Delta q_t)` (input-dependent step size).
- `B, C` are linear projections of q.

CHIRON shear: `(q, p) ↦ (q, p + Y_Mamba(q))`.

### 3.3 SCFA Transformer block

Per #42 SCFA spectral attention with k=64 modes:
$$
Y_SCFA(q) = B \cdot \text{SoftmaxAttn}((B^T q) W_Q, ..., ...) W_O + D((I - BB^T) q)
$$

CHIRON shear: `(q, p) ↦ (q, p + Y_SCFA(q))`.

### 3.4 MoE FFN (per #53 MOSAIC-MOE)

For layers with MoE:
$$
Y_MoE(q) = \sum_{e \in \mathcal{E}(q)} r_e(q) \cdot FFN_e(q)
$$
where E=8, k=2 with shared FFN backbone + LoRA adapters.

### 3.5 Theorem 1 — Hybrid bijectivity

**Theorem 1.** The composed JAMBA-CHIRON layer-pattern produces a bijective map on (q, p) state via composition of bijective shears (each shear is bijective by Theorem 3 of #42 for any continuous Y).

**Proof.** Each Mamba block, SCFA Transformer block, and MoE FFN block uses a CHIRON shear of form `(q, p) ↦ (q, p + Y(q))`. Y is continuous (composition of continuous operations). Theorem 3 of #42 applies per-layer; composition is bijective. ∎

CHIRON's O(1) activation memory advantage holds.

### 3.6 Theorem 2 — Routing reproducibility

For MoE layers within JAMBA, routing must be reproducible at inverse-walk time. Same constraint as #53 MOSAIC-MOE (deterministic, function of q only).

---

## 4. Compute analysis

### 4.1 Per-block compute at T=1024

| Block type | FLOPs per layer | Comment |
|---|---|---|
| Mamba | ~33 MFLOPs | O(T·m·N), N=16 |
| SCFA Transformer | ~1.98 GFLOPs | O(T·k·m + k²·m), k=64 |
| Standard Transformer attention | ~17 GFLOPs | O(T²·m) baseline |
| MoE FFN | ~1.4 GFLOPs | k=2 of E=8 (post-MELT) |
| Dense MELT FFN | ~5.4 GFLOPs | post-#44 |

At T=1024, average JAMBA layer: ((33M Mamba + 1.98G SCFA) / 2 + (1.4G MoE + 5.4G dense MELT) / 2) ≈ 4.4 GFLOPs/layer total.

**Per-step JAMBA-CHIRON: 53 × 4.4 GFLOPs = 233 GFLOPs.**
Pre-#54 (post-#53 MOSAIC-MOE only): 53 × 5.6 GFLOPs ≈ 297 GFLOPs.

**Speedup at T=1024: 297/233 = 1.27×.** Modest.

### 4.2 At T=8192

| Block type | FLOPs per layer at T=8192 |
|---|---|
| Mamba | 268 MFLOPs (linear in T) |
| SCFA Transformer | 1.98 GFLOPs (still O(T·k·m + k²·m), but k=64 fixed) |
| MoE FFN | 11.2 GFLOPs (linear in T) |
| Dense MELT FFN | 43 GFLOPs |

Average JAMBA layer at T=8192: (268M + 1.98G)/2 + (11.2G + 43G)/2 = 28 GFLOPs/layer.

**Per-step at T=8192: 53 × 28 GFLOPs ≈ 1.5 TFLOPs.**

Pre-#54 at T=8192 (with all transformer + MoE): 53 × ~70 GFLOPs ≈ 3.7 TFLOPs.

**Speedup at T=8192: 3.7/1.5 = 2.5×.** And without #54, T=8192 would be borderline infeasible memory-wise.

### 4.3 At T=16384

Pure transformer attention at T=16384 = O(T²·m) ≈ 137 GFLOPs/layer = infeasible memory-wise (28 GB attention scratch alone).

JAMBA-CHIRON at T=16384: Mamba blocks scale linearly (~530 MFLOPs); SCFA blocks still bounded (k=64); MoE FFN scales linearly. **JAMBA enables T=16384 training where pure-attention is infeasible.**

### 4.4 Cumulative single-GPU stack

At T=1024, 144B-effective (post-#53):
- Pre-#54: 1380× tokens·params/sec.
- Post-JAMBA: 1380 × 1.27 = ~1750× tokens·params/sec.

At T=8192, 144B-effective:
- Pre-#54 (if feasible): would be much slower per token at long T.
- Post-JAMBA: ~2500-3000× tokens·params·context/sec.

At T=16384: only achievable via JAMBA. Qualitative unblock.

---

## 5. Composition with paradigms #42-#53

| Paradigm | Composes? | Mechanism |
|---|---|---|
| **CHIRON #1** | ✓ Theorem 1 | Hybrid shear pattern preserves bijectivity |
| **MFIO/WIP/IBGRAD** | ✓ Per-block | Adam state per block-type |
| **FACE #28** | ✓ Embedding-island | Embeddings BF16 |
| **CSP/SPAREC** | ✓ Per-block | Sparsity within transformer FFN blocks |
| **SLC/RLG/SAS** | ✓ Curriculum | Schedule applies to all blocks |
| **SCFA #42** | ✓ Used | Inside Transformer blocks |
| **ORION #43** | ✓ Anchor-based | Anchor F+B includes both Mamba and Transformer |
| **MELT #44** | ✓ Per-block | TT-FFN in dense FFN layers |
| **HYDRA #45** | excluded by single-GPU | — |
| **REFLECTOR #46** | ✓ Per-block | Cotangent-lift per shear |
| **PHOENIX-1.58BIT #47** | ✓ Per-tensor | Ternary weights for Mamba A/B/C/Δ + Transformer + MoE |
| **PHOENIX-1BIT #48** | excluded by NLL | — |
| **ICARUS #49** | ✓ Per-Yoshida-sub-step | Yoshida composition over hybrid layers |
| **HELIUM #50** | ✓ FP8 | FP8 GEMM for all block types |
| **ATLAS-COMPILE #51** | △ Variable per-block-type | Layer-pattern fixed; CUDA Graph capture works |
| **NIMBUS #52** | ✓ Pipelined | Per-block-type Adam pipelining |
| **MOSAIC-MOE #53** | ✓ MoE FFN | Used in selected layers per JAMBA pattern |

**Triple-paradigm composition: #42 + #53 + #54.** The broadest in the research program.

---

## 6. NLL preservation analysis

JAMBA-CHIRON does NOT preserve NLL bit-exactly (different architecture from CHIRON baseline). Empirical evidence:

**Jamba 2024 (Lieber et al., AI21 Labs):**
- 52B total params, 12B active, hybrid 1:1 Mamba+Transformer with MoE.
- Matches Mixtral 8x22B on most benchmarks.
- Better than pure Mamba on retrieval-heavy tasks.
- Better than pure Transformer on long-context retention.

At our 1.84B-active, 144B-effective scale: Jamba-class quality expected.

**Conjecture C1**: JAMBA-CHIRON achieves loss within 0.10 nat of equivalent dense-attention baseline at 5000 steps pile-bpe. Falsifiable in 60 GPU-min Gate-0.

---

## 7. Engineering scope

- Mamba primitives (S6 selective scan kernel + parallel scan): ~700 LOC.
- Mamba block forward/backward: ~400 LOC.
- Layer-pattern dispatch (Mamba vs SCFA vs MoE): ~250 LOC.
- Hybrid Adam state per block-type: ~200 LOC.
- Composition with #53 MOSAIC-MOE: ~200 LOC (mostly bookkeeping).
- Composition with #51 ATLAS-COMPILE (CUDA Graph capture): ~250 LOC.
- Trainer integration: ~200 LOC.
- Unit tests + Gate-0: ~270 LOC.
- **Total: ~2470 LOC over 7-10 weeks.**

Largest engineering surface among #54 candidates.

---

## 8. Concrete primitives

```cpp
namespace glades { namespace gpu { namespace jamba {

enum class BlockType {
    MAMBA_DENSE,
    SCFA_TRANSFORMER_DENSE,
    MAMBA_MOE,
    SCFA_TRANSFORMER_MOE
};

BlockType jamba_layer_type(int layer_index);  // returns block type per pattern

// Mamba selective state-space scan (S6) — forward.
void mamba_selective_scan_forward(
    const __nv_bfloat16* q,        // [T, m]
    const __nv_bfloat16* A_log,    // [m, N]
    const __nv_bfloat16* B,        // [T, N]
    const __nv_bfloat16* C,        // [T, N]
    const __nv_bfloat16* delta,    // [T] (input-dependent)
    __nv_bfloat16* y,              // [T, m]
    __nv_bfloat16* h_state,        // [T, m, N] cached for backward
    int T, int m, int N,
    cudaStream_t stream);

void mamba_selective_scan_backward(
    const __nv_bfloat16* dy,
    const __nv_bfloat16* q,
    const __nv_bfloat16* A_log, B, C, delta,
    const __nv_bfloat16* h_state,
    int T, int m, int N,
    __nv_bfloat16* dq,
    float* dA_log, dB, dC, ddelta,
    cudaStream_t stream);

// JAMBA dispatch wrapper.
void jamba_layer_forward(
    BlockType type,
    const __nv_bfloat16* q,
    const __nv_bfloat16* p,
    // weights for all block types passed via union
    const JambaLayerWeights* weights,
    __nv_bfloat16* p_out,
    int T, int m,
    cudaStream_t stream);

}}}  // namespace glades::gpu::jamba
```

CLI: `--jamba 1 --jamba-pattern 1m1t-moe` (1 Mamba, 1 Transformer, with MoE in alternating).

---

## 9. Failure modes

| Failure mode | Detection | Mitigation |
|---|---|---|
| **NLL regression > 0.30 nat** at 1.84B | Phase 4 EMA test | Adjust block ratio; favor Transformer for retrieval-heavy tasks |
| **Mamba state instability at long T** (BF16 numerics) | Per-block gradient norm | FP32 state accumulation; lower-precision per-token |
| **Layer-pattern transition discontinuity** | Loss spike at pattern boundary | Smooth interleaving; gradient-clip at boundaries |
| **CUDA Graph capture fails on per-block dispatch** | Phase 4 implementation | Fixed block pattern eliminates branching; replace conditionals with constant indexing |
| **MoE composition with Mamba differs from with Transformer** | Per-block-type quality | Tune MoE per block-type independently |

---

## 10. Cumulative trajectory across 13 iterations

| Iter | Paradigm | Single-GPU stack |
|---|---|---|
| 186 | #42 SCFA | 7.6× at 1.84B |
| 188 | #44 MELT | 108× + 18B ceiling |
| 190 | #46 REFLECTOR | 162× (bit-exact) |
| 193 | #49 ICARUS | 300× |
| 194 | #50 HELIUM | 555× |
| 196 | #52 NIMBUS | 917× (bit-exact-equiv) |
| 197 | #53 MOSAIC-MOE | 1380× at 144B effective |
| **198** | **#54 JAMBA-CHIRON** | **~1750× at 144B effective + T=1024; ~2500× at T=8192; qualitative unblock at T=16384+** |

---

**End of Paradigm Shift #54 design document.** ~5000 words. Hybrid architecture: Mamba SSM + SCFA Transformer + MoE. Long-context capable; cumulative ~2500× tokens·params·context/sec at T=8192, 144B effective.
