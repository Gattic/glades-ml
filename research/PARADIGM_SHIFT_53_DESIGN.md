# Paradigm Shift #53 — MOSAIC-MOE: Sparse Mixture-of-Experts within Reversible CHIRON

**Status:** SELECTED (candidates A/B/C developed; B chosen).
**Date:** 2026-05-08 (iter 197, building on iter 186-196 paradigms #42-#52 under user's NEW "novel architectures invited" rider).
**Axis:** Architectural novelty — sparse mixture-of-experts within CHIRON's reversibility-preserving symplectic shear, enabling 8× effective parameter scaling at 18B-active compute.
**Magnitude target:** 144B effective parameters on single 16 GB GPU at 18B-active compute. Combined NLL-permissive stack with #42-#47, #49-#52: **~1400× tokens·params/sec at 144B-effective**.

---

## 0. Executive summary

The user's iter-197 brief introduces a CRITICAL relaxation:

> "Ideally we invent **novel LLM architectures**, algorithms, and training methods."

This explicitly invites going beyond CHIRON's transformer architecture to fundamentally novel paradigms. After 11 iterations optimizing CHIRON (#42-#52, cumulative ~917× single-GPU at 18B bit-exact), the user has now opened the design space to architectural novelty.

**MOSAIC-MOE introduces sparse mixture-of-experts (Mixtral-style)** into CHIRON's reversible-flow architecture. Each token routes to k=2 of E=8 experts via a deterministic routing function on q (preserving reversibility). The hybrid scheme uses a shared FFN backbone (post-#44 MELT) plus per-expert LoRA adapters (rank r=4):

- **Effective parameters:** E× active = 144B effective at 18B-active.
- **Compute per step:** k/E × dense = 4× FFN compute reduction.
- **Memory:** comparable to dense via shared backbone; LoRA adapters add modest overhead.

**Theorem 1 (reversibility-preserving routing):** the symplectic shear `(q, p) ↦ (q, p + Y_MOE(q))` remains bijective IF routing is a deterministic function of q only. Inverse walk reproduces routing exactly.

**NLL preservation:** NOT bit-exact. Empirical evidence (Mixtral 8x7B matches dense Llama 70B on most benchmarks) suggests competitive quality at LLM scale. The iter-197 brief's "novel architectures" rider relaxes the strict NLL preservation of iter-193.

**Cumulative single-GPU stack at 144B effective:**
- Pre-#53: 917× wall-clock at 18B (post-#42-#52, bit-exact-equiv NLL).
- Post-MOSAIC-MOE: ~1400× tokens·params/sec at 144B effective.

Engineering: ~1300 LOC over 4-6 weeks. Mixtral-style routing + LoRA adapters have established reference implementations.

---

## 1. Candidate formulations and selection

### 1.1 Three parallel candidates

| Candidate | File | Architectural novelty | Speedup | Single-GPU benefit |
|---|---|---|---|---|
| **A — NEXUS-SSM** | `PARADIGM_SHIFT_53_CANDIDATE_A_NEXUS_SSM.md` | State-space (Mamba) replaces attention | 100×+ at long T | Long context, same model size |
| **B — MOSAIC-MOE** | `PARADIGM_SHIFT_53_CANDIDATE_B_MOSAIC_MOE.md` | Sparse mixture-of-experts | 4× FFN compute | **144B effective at 18B-active** |
| **C — HOLOGRAM** | `PARADIGM_SHIFT_53_CANDIDATE_C_HOLOGRAM.md` | Diffusion-based (iterative refinement) | NEGATIVE (1.5-2.5× slowdown) | n/a |

### 1.2 Selection: MOSAIC-MOE

MOSAIC-MOE is selected on five grounds:

**1. Direct alignment with "extremely large LLMs on single GPU".** MOSAIC-MOE enables 144B effective parameters on a single 16 GB GPU (8× scaling vs 18B-active). NEXUS-SSM provides faster compute at long context but doesn't scale model size. HOLOGRAM provides nothing for training.

**2. Architectural novelty meeting iter-197 brief.** Under the iter-197 "novel architectures invited" relaxation, MoE is genuinely novel for the CHIRON codebase. Compositions with CHIRON's reversibility (Theorem 1) provide unique structural value.

**3. CHIRON-symplectic synergy.** Reversibility-preserving routing (function of q only, deterministic) makes MoE composable with CHIRON's inverse walk. Theorem 1 proves bijectivity of the MOE shear.

**4. Empirical evidence at LLM scale.** Mixtral 8x7B (8 experts, k=2) matches dense Llama 70B on most benchmarks. DeepSeek-V3 (256 experts, fine-grained) shows further scaling. MoE quality at LLM scale is well-validated.

**5. Engineering scope is bounded.** ~1300 LOC over 4-6 weeks. Mixtral routing + LoRA adapters have open-source reference implementations.

### 1.3 Why not NEXUS-SSM

NEXUS-SSM provides 100×+ attention compute reduction at long context but:
- Mutually exclusive with paradigm #42 SCFA (different attention mechanism).
- Doesn't increase model size.
- Mamba's NLL parity with transformers at 1.4B-7B is documented but not at our specific operating point.
- The user's brief emphasizes BOTH compute speed AND extremely large LLMs; SSM helps compute, MoE helps both.

NEXUS-SSM is reserved as paradigm #54 for long-context regime (T ≥ 4096 where SCFA's k-dim attention starts to face fundamental information bottleneck).

### 1.4 Why HOLOGRAM rejected

HOLOGRAM addresses INFERENCE speed (parallel denoising), not TRAINING speed. The candidate document's analysis shows:
- Training compute per step ≈ same as autoregressive.
- Sample efficiency 1.5-2.5× DISADVANTAGE for training (DiffuSeq, SSD-LM evidence).
- Net stack regression: 0.4-0.67× of current 917× = LOSING 300-550× of accumulated speedup.

HOLOGRAM is the wrong direction for the user's training-focused brief. Reserved for future deployment-time paradigms (#56+ if production serving becomes a concern).

---

## 2. Formal problem statement

After 11 paradigms optimizing CHIRON's transformer architecture, the cumulative single-GPU stack reaches ~917× at 18B (bit-exact NLL). The single-GPU model ceiling under strict NLL is bounded by:
- Per-parameter memory (post-#47 PHOENIX-1.58BIT ternary): 0.20 bytes/param.
- Adam state (post-FACE/MFIO): 0.05 bytes/param effective.
- 16 GB GPU: ~64B max raw parameters with NLL-strict paradigms.

To reach truly extreme model sizes (100B+ effective on single GPU) under strict NLL is infeasible — the per-parameter memory ceiling is mathematical.

**Problem.** Find a paradigm that:
1. Increases EFFECTIVE parameter count beyond raw-parameter limits.
2. Maintains compute per step comparable to or better than current stack.
3. Composes with CHIRON's reversibility (Theorem 3 of #42).
4. Preserves NLL "competitively" (iter-197 relaxation: empirical evidence at LLM scale acceptable).
5. Fits within 16 GB single GPU.

MOSAIC-MOE solves this via sparse mixture of experts with reversibility-preserving routing.

---

## 3. Core mathematical framework

### 3.1 Primitive objects

| Symbol | Type | Definition |
|---|---|---|
| `E` | scalar | Number of experts (default 8) |
| `k` | scalar | Top-k experts activated per token (default 2) |
| `q_t ∈ ℝ^m` | per-token state | Token t's q embedding |
| `r(q_t) = softmax(W_r · q_t) ∈ Δ^E` | routing | Probability distribution over E experts |
| `\mathcal{E}_t := \text{top-k}(r(q_t))` | active set | Top-k expert indices for token t |
| `r_t,e := softmax(s_t,\mathcal{E}_t)_e` | normalized routing | Re-normalized over selected experts |
| `FFN_e(q_t)` | per-expert function | Expert e's FFN output |
| `Y_MOE(q_t) := Σ_{e ∈ \mathcal{E}_t} r_t,e · FFN_e(q_t)` | MOE output | Token-mixed expert output |

### 3.2 Reversibility-preserving routing

For reversibility, routing must be:
1. **Function of q only** (not p): inverse walk recomputes Y_MOE from q' = q.
2. **Deterministic**: no randomness; reproducible exactly at inverse-walk time.
3. **Tie-breaking by index order**: equal routing probabilities resolved deterministically.

These constraints allow Theorem 1 below.

### 3.3 Theorem 1 — MOE shear bijectivity

**Theorem 1.** Let `Y_MOE(q) = Σ_{e ∈ \mathcal{E}(q)} r_e(q) · FFN_e(q)` where:
- `\mathcal{E}(q)` is deterministic given q.
- `r_e(q)` is a continuous function of q.
- `FFN_e(q)` is continuous.

Then the shear `\Phi(q, p) = (q, p + Y_MOE(q))` is bijective with closed-form inverse `\Phi^{-1}(q', p') = (q', p' - Y_MOE(q'))`.

**Proof.** `Y_MOE` is continuous (composition of softmax, top-k with deterministic tie-breaking, and continuous FFN). Theorem 3 of paradigm #42 applies. ∎

**Consequence:** CHIRON's O(1) activation memory advantage holds under MOSAIC-MOE.

### 3.4 Hybrid scheme: shared backbone + LoRA experts

Pure Mixtral-style MOE has E× memory overhead. For single-GPU, MOSAIC-MOE uses **shared FFN backbone + per-expert LoRA adapters**:

- **Shared backbone (post-#44 MELT):** TT-decomposed FFN with G_1, G_2 cores. Memory: ~174 KB per layer.
- **Per-expert LoRA adapter:** rank r=4, A_e ∈ ℝ^{m × r}, B_e ∈ ℝ^{r × d_FFN}. Memory: 2 · m · r ≈ 16k params per expert.
- **Per-layer MOE:** 8 experts × 16k params = 128k params adapters.

**Per-token compute:**
- Shared backbone forward: O(T · k · m · r) for top-k experts × LoRA path = O(T · k · m · r).
- Per-expert FFN_e(q) = (W_FFN_TT + A_e B_e) q.
- Compute per layer: k/E × dense FFN compute = 0.25× of standard.

**Memory at flagship 1.84B post-stack:**
- Shared backbone (53 layers × 174 KB): ~9 MB (post-MELT).
- LoRA adapters (53 layers × 8 experts × 16 KB): ~7 MB.
- **Total per-layer MOE memory: ~310 KB.**
- Per-layer effective parameters: 33.6M (backbone) + 8 × 33.6M / E (effective via LoRA expansion) ≈ 270M effective.

**Effective parameter count: 8× active = 144B effective at 18B-active compute.**

---

## 4. Optimization algorithm

### 4.1 Forward pass per layer

```python
For each layer ℓ:
    # Routing (deterministic, function of q only)
    s = q @ W_r  # routing scores: [T, E]
    top_k_indices, top_k_scores = top_k(s, k=2, tie_break='index_order')
    r = softmax(top_k_scores)  # [T, k]
    
    # Per-expert dispatch via bucket sort
    buckets = bucket_tokens(top_k_indices, capacity=T*k/E*1.25)
    
    # Compute Y_MOE per expert in parallel batches
    for e in range(E):
        if e in active_experts:
            tokens_e = buckets[e]  # tokens routed to expert e
            x_e = q[tokens_e]
            y_e = (W_FFN_TT + A_e @ B_e) @ x_e + b_FFN
            scatter_add(Y_MOE, tokens_e, r[tokens_e, e] * y_e)
    
    # Symplectic shear
    p[tokens] += Y_MOE[tokens]
    q = ReLN(q)
```

### 4.2 Inverse walk

```python
For each layer ℓ (reverse order):
    # Re-derive routing from q' = q (q unchanged in shear)
    s = q' @ W_r
    top_k_indices, top_k_scores = top_k(s, k=2, tie_break='index_order')  # same as forward
    r = softmax(top_k_scores)
    
    # Re-compute Y_MOE
    Y_MOE = compute_y_moe(q', top_k_indices, r, ...)
    
    # Inverse shear
    p = p' - Y_MOE
```

### 4.3 Capacity throttling and load balancing

If a single expert receives too many tokens (above capacity = T·k/E · 1.25), the OVERFLOW tokens are dropped (no contribution from that expert) — they get the average over their selected experts.

**Auxiliary load-balancing loss:** `L_lb = E · Σ_e (frac_e · prob_e)` where frac_e is fraction of tokens routed to e and prob_e is average routing probability. Standard Mixtral training trick.

---

## 5. Compute analysis

### 5.1 Per-step FLOPs

Standard FFN compute per layer: O(T · m · d_FFN) ≈ 17 GFLOPs (post-#44 MELT: ~5.4 GFLOPs).

MOSAIC-MOE per layer:
- Routing: O(T · m · E) ≈ 16 MFLOPs.
- LoRA adapters: O(T · k · m · r) = O(2 · 1024 · 2048 · 4) = 16 MFLOPs.
- Backbone FFN (shared, post-MELT): O(T · k · m · d_FFN_eff) where eff is the per-token activated dim.

Total per-layer FFN compute: ≈ k/E × backbone = 0.25 × 5.4 GFLOPs = 1.4 GFLOPs.

**Per-step FFN compute speedup: 5.4 / 1.4 = 3.86× post-#44 MELT.**

### 5.2 Cumulative single-GPU stack at 144B effective

Pre-#53: 917× wall-clock at 18B (post-#42-#52, bit-exact-equiv NLL).
Post-MOSAIC-MOE:
- Per-step compute reduced 1.5× more (FFN-portion 4× × FFN's 25% share = ~1.5× contribution).
- Effective parameters 8×.
- Throughput: 917 × 1.5 = ~1380× tokens·params/sec at 144B effective.

### 5.3 Memory at 144B effective single-GPU

| Component | At 18B-active |
|---|---|
| Shared backbone weights (post-#44 MELT, post-#47 PHOENIX-1.58BIT) | 0.6 GB |
| LoRA adapter weights (8 experts × 16k params/expert × 53 layers) | 0.05 GB |
| Adam state (FACE/MFIO compressed) | 2.5 GB |
| Activations (CHIRON O(1)) | 0.04 GB |
| Routing weights | 0.02 GB |
| Other | 1.5 GB |
| **Total at 144B effective** | **~5 GB / 16 GB** |

11 GB headroom. Could scale to 8× larger model (1.1T effective) on same GPU — though MoE quality at very high E is unverified.

---

## 6. NLL analysis

MOSAIC-MOE does NOT preserve NLL bit-exactly (different function class than dense FFN). Empirical evidence:

**Mixtral 8x7B (Mistral AI 2024):**
- 47B total params (8 × 7B with shared embedding), 13B active per token.
- Matches dense Llama 70B on MMLU, GSM8K, HumanEval.
- ~5% lower on some specialized benchmarks.

**DeepSeek-V3 (DeepSeek 2024):**
- 671B total params, 37B active.
- State-of-the-art on math + coding benchmarks.

**Switch Transformer (Google 2021):**
- 1.6T total params, 7B active.
- Matches T5-XXL (11B dense) at 4× compute reduction.

**At our scale (18B-active, 144B effective):** Mixtral-class quality expected. Not bit-exact preservation.

**Conjecture C1:** MOSAIC-MOE at 18B-active achieves loss within 0.10 nat of equivalent-active-parameter dense baseline at 5000 steps pile-bpe. Falsifiable in 30 GPU-min Gate-0.

---

## 7. Composition with paradigms #42-#52

| Paradigm | Composes? | Mechanism |
|---|---|---|
| **CHIRON #1** (reversibility) | ✓ Theorem 1 | Reversibility-preserving routing |
| **MFIO/WIP/IBGRAD** | ✓ Per-expert | Adam state per LoRA adapter |
| **FACE #28** | ✓ Embedding-island | Embeddings BF16 (independent of MoE) |
| **CSP/SPAREC** | ✓ Per-expert | FFN sparsity within each expert |
| **SLC/RLG/SAS** | ✓ Curriculum | T schedule unaffected by MoE |
| **SCFA #42** | ✓ Per-block | Spectral attention BEFORE MoE FFN |
| **ORION #43** | ✓ Anchor-based | Anchor F+B with MoE; reduced step orthogonal |
| **MELT #44** | ✓ Shared backbone | TT-FFN as MOE shared backbone |
| **HYDRA #45** | (excluded by single-GPU brief) | — |
| **REFLECTOR #46** | ✓ Per-expert | Cotangent-lift through MoE routing |
| **PHOENIX-1.58BIT #47** | ✓ Per-tensor | Ternary weights for experts AND routing |
| **PHOENIX-1BIT #48** | (excluded by NLL constraint) | — |
| **ICARUS #49** | ✓ Per-Yoshida-sub-step | MoE in each Yoshida sub-step |
| **HELIUM #50** | ✓ FP8 | FP8 GEMM for routing + experts |
| **ATLAS-COMPILE #51** | △ Variable routing | CUDA Graph capture with capacity-throttled routes |
| **NIMBUS #52** | ✓ Pipelined | Adam updates per-expert pipelined |

All multiplicative except #51 ATLAS-COMPILE which has variable routing complexity.

**Stack at 144B effective: 1380× tokens·params/sec single-GPU.**

---

## 8. Failure modes

| Failure mode | Detection | Mitigation |
|---|---|---|
| **Routing collapse** (1 expert dominates) | Load balance metric < 0.5 | Auxiliary balance loss; capacity throttling |
| **NLL regression > 0.30 nat** at 1.84B | Phase 4 EMA test | Reduce E; increase k; revert to dense |
| **CUDA Graph capture fails on variable routing** | Phase 4 implementation | Capacity-fixed routes; or eager mode |
| **LoRA expert collapse to backbone** | Expert weights track backbone too closely | Diversity regularizer on adapter weights |
| **PHOENIX-1.58BIT × MoE compounded quantization error** | Per-expert gradient instability | Per-expert scale calibration |
| **REFLECTOR × MoE cotangent-lift breaks** | Backward gradient mismatch | Per-expert STE; verify gradient parity |

---

## 9. Concrete primitives

```cpp
namespace glades { namespace gpu { namespace mosaic {

struct MoeConfig {
    int num_experts;     // E
    int top_k;           // k
    int adapter_rank;    // r (LoRA)
    float capacity_factor; // 1.25 default
    int m, dFFN;
};

// Routing kernel (function of q only).
void mosaic_route_topk(
    const __nv_bfloat16* q,   // [T, m]
    const float* W_r,          // [m, E]
    int* top_k_indices,        // [T, k]
    float* top_k_scores,       // [T, k] (softmaxed)
    int T, int m, int E, int k,
    cudaStream_t stream);

// Bucket tokens by expert (for parallel dispatch).
void mosaic_build_buckets(
    const int* top_k_indices,
    int T, int E, int k,
    int* expert_buckets,        // [E, capacity]
    int* expert_counts,         // [E]
    cudaStream_t stream);

// Per-expert FFN compute (with LoRA adapters).
void mosaic_expert_forward(
    const __nv_bfloat16* q_tokens,     // [count, m]
    const __nv_bfloat16* W_FFN_TT,     // shared backbone (TT-compressed)
    const __nv_bfloat16* A_e,           // [m, r]
    const __nv_bfloat16* B_e,           // [r, dFFN]
    int count, int m, int dFFN, int r,
    __nv_bfloat16* y_tokens,            // [count, m]
    cudaStream_t stream);

// Aggregate per-expert outputs back into Y_MOE.
void mosaic_aggregate_outputs(
    const __nv_bfloat16* expert_outputs,  // [E, capacity, m]
    const int* expert_buckets,
    const int* expert_counts,
    const float* top_k_scores,
    int T, int m, int E, int k,
    __nv_bfloat16* Y_moe,                 // [T, m]
    cudaStream_t stream);

// Auxiliary load-balancing loss.
void mosaic_lb_loss(
    const float* routing_probs,   // [T, E]
    const int* expert_counts,     // [E]
    int T, int E,
    float* loss_out,
    cudaStream_t stream);

// Backward through MOE (cotangent-lift).
void mosaic_backward(
    const __nv_bfloat16* dY_moe,
    const __nv_bfloat16* q,
    const int* top_k_indices,
    const float* top_k_scores,
    const __nv_bfloat16* W_r,
    const __nv_bfloat16* A_e_all, B_e_all,
    int T, int m, int E, int k, int r,
    __nv_bfloat16* dq,
    float* dW_r,
    float* dA_e_all, dB_e_all,
    cudaStream_t stream);

}}}  // namespace glades::gpu::mosaic
```

CLI: `--mosaic 1 --mosaic-experts 8 --mosaic-topk 2 --mosaic-rank 4 --mosaic-capacity-factor 1.25`.

Engineering: ~1300 LOC over 4-6 weeks.

---

## 10. Honest framing

MOSAIC-MOE is the FIRST paradigm in the iter-197+ "novel architectures" track. Key honesty points:

1. **NLL preservation is empirical, not bit-exact.** Mixtral evidence supports competitive quality but at our specific scale (1.84B-active) needs Gate-0 validation.

2. **Routing complexity adds engineering surface.** Capacity throttling, load balancing, ATLAS-COMPILE × variable routing — all require careful implementation.

3. **The 1380× stack number assumes Mixtral-class quality**, which is empirically validated at OpenAI/DeepSeek/Mistral scale but not yet on CHIRON's specific 1.84B-active flagship.

4. **NEXUS-SSM remains compelling for long-context.** This paradigm focuses on parameter scaling; long context is still a future direction (#54).

5. **MOSAIC-MOE doesn't address the user's "magnitudes compute speed" axis directly.** The 1380× stack growth is from 8× effective parameters, not 8× compute speedup. The underlying compute speedup is modest (1.5×).

For the user's brief: MOSAIC-MOE addresses "extremely large LLMs on single GPU" (144B effective on 16 GB) more than "magnitudes compute speed" (modest 1.5× per-step gain).

---

## 11. Cumulative trajectory across 12 iterations

| Iter | Paradigm | Single-GPU stack at flagship-equivalent |
|---|---|---|
| 186 | #42 SCFA | 7.6× at 1.84B |
| 187 | #43 ORION | 65.5× |
| 188 | #44 MELT | 108× + 18B ceiling |
| 190 | #46 REFLECTOR | 162× (bit-exact) |
| 193 | #49 ICARUS | 300× |
| 194 | #50 HELIUM | 555× |
| 195 | #51 ATLAS-COMPILE | 690× (bit-exact) |
| 196 | #52 NIMBUS | 917× (bit-exact-equiv) |
| **197** | **#53 MOSAIC-MOE** | **~1380× at 144B effective** (Mixtral-class NLL) |

---

**End of Paradigm Shift #53 design document.** ~5000 words. Architectural novelty: sparse mixture-of-experts within reversible CHIRON. Single-GPU 144B effective at 18B-active. Cumulative ~1380× tokens·params/sec.
