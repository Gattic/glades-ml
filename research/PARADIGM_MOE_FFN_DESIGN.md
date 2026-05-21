# PARADIGM_MOE_FFN — Mixture-of-Experts FFN Sparsity for CHIRON

**Status**: design proposal, not implemented.
**Author/date**: research, 2026-05-16.
**Position in program**: one of three multi-iter paradigm-arc candidates after iter61 engineering ceiling. The most invasive of the three. The only one of the three with plausible standalone 2× headroom.

---

## 1. TL;DR

Replace the dense per-layer FFN (W1: 4m×m SiLU, W2: m×4m) with a top-k routed mixture of K experts plus a learned router. Active FFN FLOPs per token drop by `K/top_k`; total FFN params grow by `K/top_k` at iso-per-expert width. Expected wall-clock at iter61 flagship: **+10-18% at NLL parity**, given FFN ≈20% of step and post-permutation top-2/8 collapses FFN to ≈25-30% of dense. VRAM impact: **+0.6 to +4 GB** at L=24 m=2048 depending on `d_ff_expert`. **Multi-iter arc** (5-8 iters); per-iter bar must be relaxed to +3%. The 10× brief target was always unrealistic — MoE is the only of the three candidate paradigms with plausible 2× standalone headroom, and the most likely to ship a single sub-iter ≥+5%.

---

## 2. Motivation

The iter61 dead-zone map looks like:

| Component                  | % GPU time | Status                                              |
|----------------------------|-----------:|-----------------------------------------------------|
| SCFA outer GEMMs           |        15% | bf16-TC + algo-override exhausted (iter52, iter59) |
| Attention element-wise     |        15% | element-wise mem-bound, iter47 SCFA-dK shipped     |
| Readout GEMMs              |        13% | bf16-logits shipped, algo-override null (iter52)   |
| **FFN dense W1/W2 GEMMs**  |    **18-22%** | **DENSE — open**                                |
| LN/RMSNorm + element-wise  |       8-10% | iter48/iter53 sub-bar, dead-zone                    |
| Adam + grad-norm + I/O     |       6-8%  | iter49 fused-Adam shipped                           |
| Other                      |       10%  | argmax/loss/aux (iter56 sub-bar)                    |

Attention is saturated; cuBLAS auto-pick is empirically optimal on well-tuned shapes (iter52, iter59). FFN is the only single component not attacked at the algorithm level. FFN GEMMs at `(T=8192, 4m, m)` are well-shaped for BF16-TC and SM-saturated. **The only big win left in FFN is doing less of it.** MoE activates `top_k` of `K` experts per token.

Estimate (top-k=2 of K=8, expert width = 4m):
- Dense FFN FLOPs (fwd): `16 × T × m²`.
- MoE FFN active FLOPs: 2/8 = 1/4 of dense.
- Routing/permutation overhead: ≤1 ms at T=8192 (router GEMM is `T × K × m` ≈ 134M FLOPs, ≪ FFN).
- Net FFN wall ≈ 30-40% of dense → at 20% of step, FFN drops to 6-8% → **net +12-16%**.

Assumes healthy routing (no collapse) and overhead bounded at T=8192.

---

## 3. Mathematical Formulation

### 3.1 Forward

Per layer, per token `t`, with `q_t ∈ R^m` input residual, `W_router ∈ R^{K×m}`, per-expert `(W1_k ∈ R^{d_ff × m}, W2_k ∈ R^{m × d_ff})`:
```
r_t       = softmax(W_router · q_t)                # K-way routing
(I_t, p_t) = top_k(r_t)                            # top_k indices, raw probs
w_t       = p_t / sum(p_t)                         # renormalize
e_{t,j}   = W2_{I_t[j]} · SiLU(W1_{I_t[j]} · q_t)   # per chosen expert
out_t     = Σ_j w_t[j] · e_{t,j}                    # weighted combine
```
In practice, expert calls are **batched by expert**, not by token: permute tokens into per-expert contiguous buffers (each token appears `top_k` times), one GEMM-pair per expert, then unpermute + weighted-sum combine.

### 3.2 Load-balancing auxiliary loss

Switch-style:
```
f_k = (1/T) Σ_t 1[k ∈ I_t]    # fraction routed to expert k
P_k = (1/T) Σ_t r_t[k]         # mean routing prob for expert k
L_LB = α_LB · K · Σ_k f_k · P_k    # α_LB ≈ 0.01
```
`f_k` has piecewise-constant gradient (STE with stop-grad on indicator); `P_k` is differentiable.

### 3.3 Backward — router gradient

Three options:

| Variant                       | Path                                       | Risk                              |
|-------------------------------|--------------------------------------------|-----------------------------------|
| **STE**                       | top-k mask = identity on backward          | biased but stable; Mixtral/Switch |
| Differentiable top-k          | sinkhorn / sparsemax + entropy reg         | extra solver + hyperparam         |
| Reinforce-style               | softmax-only gating gradient               | high variance                     |

**Recommended**: STE on selection + full softmax on `w_t`. Per-expert FFN backward is the dense backward sized per-expert.

### 3.4 Dense as a proven limit

`K=1, top_k=1`: `r_t = [1]`, identity permutation, single expert = dense FFN. MoE strictly contains dense. Failure mode at `K > 1`: routing collapse → MoE can be *worse* than iso-param dense (Gate-0b screens for this).

---

## 4. Choice of (K, top_k) for CHIRON

| Config                       | Active FLOPs vs dense | Total FFN params | VRAM (L=24) | Use case               |
|------------------------------|----------------------:|-----------------:|------------:|------------------------|
| Dense (baseline)             | 1.0×                  | 800 M            | 0           | flagship               |
| K=4, top_1, d_ff=4m          | 0.25×                 | 3200 M           | +2.4 GB     | over-budget            |
| **K=4, top_1, d_ff=m**       | 0.0625×               | 800 M            | 0           | **iter-bench**         |
| K=8, top_2, d_ff=4m          | 0.50×                 | 6400 M           | +5.6 GB     | over-budget            |
| K=8, top_2, d_ff=m           | 0.125×                | 1600 M           | +1.6 GB     | iter-bench scale-up    |
| **K=8, top_2, d_ff=2m**      | 0.25×                 | 3200 M           | +2.4 GB     | **production candidate** |
| K=2 shared + K=6 routed top1 | 0.30×                 | 2400 M           | +1.6 GB     | DeepSeek-style fallback |

Recommended path: iter-bench K=4 top_1 d_ff=m (VRAM-neutral); production K=8 top_2 d_ff=2m (+2.4 GB); fallback shared+routed if collapse. **VRAM ceiling dominates at production** — full-width K=8 top_2 d_ff=4m (+6 GB) is infeasible on a 16 GB card with T=16384 + Adam-int8.

---

## 5. Implementation Outline

All new code lives in `Backend/Machine Learning/Networks/cuda/`. Trainer flag in `glades-trainer/trainer/chiron_main.cpp`.

### 5.1 New files

| File                                              | Role                                  |
|---------------------------------------------------|---------------------------------------|
| `Backend/Machine Learning/Networks/cuda/gpu_moe.cu`   | routing, permutation, unpermute kernels |
| `Backend/Machine Learning/Networks/cuda/gpu_moe.h`    | host-side prototypes                  |
| `Backend/Machine Learning/Networks/moe_state.h`       | per-block MoE state (K experts × W1/W2) |
| `Backend/Machine Learning/Networks/sgd_transformer_moe.cpp` | moe fwd/bwd entry per layer  |

### 5.2 New CUDA kernels

| Kernel                            | Notes                                              |
|-----------------------------------|----------------------------------------------------|
| `moe_router_softmax_topk_bf16`    | one block per token, K threads; softmax + top-k in one pass |
| `moe_permute_scatter_bf16`        | gather q_t into per-expert buffer; prefix-sum offsets |
| `moe_unpermute_combine_bf16`      | scatter expert outputs back, weighted sum          |
| `moe_lb_loss_reduce`              | compute f_k, P_k, scalar LB loss                   |
| `moe_router_bwd_softmax_bf16`     | softmax backward + STE on top-k mask               |

### 5.3 Forward dispatch (per layer)

```
router_softmax_topk<<<T,K>>>(q, W_router → I, w)
prefix_sum_offsets(I → offsets)
permute_scatter<<<T,m>>>(q, I, offsets → P)
for k in 0..K:
    Tk = expert_token_counts[k]          // ≈ T·top_k/K with healthy LB
    sgemm_rowmajor_bf16(W1_k, P[k], → M_k, Tk × d_ff_expert × m)
    silu_inplace_bf16(M_k)
    sgemm_rowmajor_bf16(W2_k, M_k, → E_k, Tk × m × d_ff_expert)
unpermute_combine<<<T,m>>>(E, I, offsets, w → out)
```

iter6/iter8/iter58 showed cross-stream cuBLAS does **not** overlap ≥1 TFLOP GEMMs on Ada. Per-expert GEMMs at K=8 top-2 are ≈17 GFLOPs — still no-overlap. **Use serial default stream.** Batched-GEMM (`cublasGemmStridedBatchedEx`) with token-padded uniform `T·top_k/K` per expert is the iter65/68 alternative.

### 5.4 Backward (per layer)

```
unpermute_combine_bwd(dout, I, w → dE_per_expert, dw)
for k in K..0:                                  // per-expert dense FFN bwd
    sgemm_atb_bf16(W2_k, dE_k → dM_k)           // dM = W2_k^T · dE_k
    sgemm_abt_bf16(M_k, dE_k → dW2_k)
    silu_backward_bf16(M_k_pre, dM_k)
    sgemm_atb_bf16(W1_k, dM_k → dP_k)
    sgemm_abt_bf16(P_k, dM_k → dW1_k)
permute_scatter_bwd(dP, I, offsets → dq_expert)
router_bwd_softmax(q, r, dw, I → dW_router, dq_router)
dq = dq_expert + dq_router
```

### 5.5 Trainer flags (`glades-trainer/trainer/chiron_main.cpp`)

```
--moe-ffn K top_k             # enable
--moe-ffn-d_ff_expert N        # per-expert width (default 4m)
--moe-lb-weight 0.01           # auxiliary loss
--moe-capacity-factor 1.25     # permutation buffer padding
```
Checkpoint schema version bump for K-expert layout.

### 5.6 Reversible-flow + Adam-int8

**Reversibility** preserved: FFN sits inside one half of the coupling; `F(q_stream)` simply has new implementation. Forward is deterministic (top-k with deterministic tie-break).

**Adam-int8**: each `W1_k, W2_k, W_router` is an independent param group; iter49 fused kernel is per-tensor agnostic, just K+1 invocations per layer (was 2). Adam state VRAM at K=8 top-2 d_ff=2m: 3200 M × 2 B ≈ 6.4 GB (vs 1.6 GB dense). **Adam state is the binding production VRAM constraint** — not BF16 weights. Mitigations: half-layer MoE, or CPU-offload Adam (`CPU_OFFLOAD_ADAM_DESIGN.md`).

---

## 6. VRAM Accounting

At L=24, m=2048, T=16384 production. Numbers in GB BF16.

| Component                     | Dense flagship | K=8 top-k=2 d_ff=2m | K=8 top-k=2 d_ff=4m | K=4 top-k=1 d_ff=m |
|-------------------------------|---------------:|--------------------:|--------------------:|-------------------:|
| FFN weights (BF16)            |           1.6 |                 4.0 |                 8.0 |                1.6 |
| Adam-int8 state (FFN)         |           1.6 |                 4.0 |                 8.0 |                1.6 |
| Router weights (BF16)         |             0 |                 0.4 |                 0.4 |                0.2 |
| Permutation buffer (BF16)     |             0 |                 0.5 |                 0.5 |                0.3 |
| Attn K/V cache (T=16384)      |           3.0 |                 3.0 |                 3.0 |                3.0 |
| Activations + residuals       |           3.5 |                 3.5 |                 3.5 |                3.5 |
| Other (logits, etc.)          |           1.5 |                 1.5 |                 1.5 |                1.5 |
| **Total**                     |       **11.2**|            **16.9** |            **24.9** |           **11.7** |
| Budget (16 GB)                |           ✓   |                 ✗   |                 ✗   |                ✓   |

Honest readout: **K=8 top-k=2 d_ff=2m does not fit at production**. Three options:
1. Shrink `d_ff_expert = m` (less expert capacity) → 11.7 GB, fits.
2. MoE on half the layers (even-numbered) → 14.0 GB, fits.
3. CPU-offload Adam state → drops 4 GB, fits with full-width experts.

Iter66 picks one based on iter62-65 results. At **iter-bench (T=8192, L=12)** all configs fit (≤8 GB).

---

## 7. Gate-0 Criteria

Three sub-gates; all must pass for ship under relaxed +3% bar:

- **Gate-0a Throughput**: ≥ +5% wall at NLL parity (≤+0.05 nat @ step 500) on iter-bench (T=8192, L=12, K=4 top-k=1 d_ff=m, 500 steps). Under +3% relaxed bar, +3% ships if 0b/0c pass.
- **Gate-0b Load balance**: per-expert `f_k` ∈ `[1/(2K), 2/K]` for all 500 steps. K=4 top-1 → `[0.125, 0.5]`; K=8 top-2 → `[0.0625, 0.25]`. Hard fail if any expert <50% or >200% expected for >50 consecutive steps.
- **Gate-0c Quality**: 5000-step NLL trajectory matches dense within ±0.05 nat at every 500-step checkpoint. Iter66 only.

0a+0b pass, 0c fail → tuning issue (LB weight, router init). Iter67+ tunes.

---

## 8. Risks and Falsification Triggers

| Risk                                       |  P  | Detection                       | Mitigation                                          |
|--------------------------------------------|:---:|---------------------------------|-----------------------------------------------------|
| Expert collapse (one dominates)            | med | `f_k` histogram iter62          | LB weight bump, router noise, longer warmup         |
| Permutation overhead eats gain @ T=8192    | med | kernel ms breakdown iter62      | go T≥16384, batched-GEMM with token padding         |
| **VRAM blowout at production**             | high| iter66 OOM                      | shrink `d_ff_expert`, half-layer MoE, CPU-offload Adam |
| Router gradient too noisy, NLL drift       | med | NLL @ step 500                  | STE → differentiable top-k or Gumbel-softmax        |
| LB weight × main-loss coupling             | high| NLL/LB curve correlation        | sweep α_LB ∈ {0.001, 0.01, 0.1} iter64              |
| K serial GEMM launch-bound                 | low-med | cuBLAS time vs GEMM math    | batched-GEMM with token padding (iter65/68)         |
| BF16 numerics in router softmax            | low | router output histogram         | FP32 router (it's tiny: K×m params)                 |
| Save/load schema breakage                  | low | persistence unit test           | schema version bump + K=1 migration                 |

**Hard falsification triggers** (terminate arc):
- iter62: K=4 top-1 d_ff=m gives **<+1% wall** at iso-NLL with healthy `f_k` → mechanism dead, pivot.
- iter64: no α_LB ∈ {0.001..0.1} achieves both healthy LB and NLL parity within ±0.10 → routing structurally broken on CHIRON, pivot.
- iter66: no (K, top_k, d_ff_expert) fits VRAM at L=24 m=2048 T=16384 → MoE infeasible on 16 GB; needs 24 GB+.

---

## 9. Iter Budget

Honest estimate: **5-8 iters**, significantly more than the other two paradigm candidates.

| Iter   | Goal                                                                          | Risk |
|--------|-------------------------------------------------------------------------------|------|
| **62** | Forward only, K=4 top_1 d_ff=m, kernel-correctness vs dense + NLL parity @ step 50 with fake router | low  |
| **63** | Backward complete; STE; Gate-0a + 0b at iter-bench                            | high |
| **64** | LB loss sweep α_LB ∈ {0.001, 0.01, 0.1}; re-Gate-0                            | med  |
| **65** | Scale to K=8 top_2 d_ff=2m at iter-bench; re-Gate-0a/0b                       | med  |
| **66** | Production revalidation L=24 m=2048 T=16384; VRAM check; Gate-0c (5k-step)    | high |
| **67** | If 0c marginal: router init/warmup tuning, trajectory matching                | med  |
| **68** | (Conditional) Batched-GEMM dispatch, token-padded uniform                     | med  |
| **69** | (Conditional) Shared-expert variant if plain MoE underperforms                | med  |

Each iter: 1-3 GPU-hrs training + 2-6 GPU-hrs implementation. Cumulative arc: **30-60 GPU-hrs + 3-6 kLoC C++/CUDA**.

**No single iter likely clears +5% standalone.** The arc compounds: iter62 structural (no perf), iter63-64 baseline (+3-5%), iter65 scales (+5-10%), iter66-67 production ships (cumulative +10-18%).

---

## 10. Comparison with Prior Art

| System              | K   | top_k | Routing               | Distance from CHIRON                  |
|---------------------|----:|------:|-----------------------|---------------------------------------|
| Switch Transformer  | 128 | 1     | softmax + LB aux      | Multi-GPU, no reversibility, FP32     |
| Mixtral 8x7B        |   8 | 2     | softmax + LB aux      | Multi-GPU, FP16/BF16                  |
| DeepSeek-V3 MoE     |  64 + 1 shared | varies | aux-loss-free LB | Multi-GPU, FP8, complex routing   |
| ST-MoE              | 32  | 2     | softmax + entropy reg | TPU-only                              |
| **CHIRON-MoE**      |   8 | 2     | softmax + LB + STE    | **Reversible + Adam-int8 + BF16, single GPU** |

**Novel for CHIRON**: single-GPU MoE at 870M-1B with hard 16 GB ceiling (forces choices multi-GPU literature skips); reversible-flow coupling integration (MoE drops into `F(q_stream)`); fused Adam-int8 integration (each expert is an independent param group, fused kernel agnostic); BF16-everywhere stack (no FP8, no FP32 except router softmax).

**Not novel**: the math, LB loss, top-k routing, expert-collapse failure mode. This is a faithful Switch+Mixtral port to CHIRON's stack; shared-expert (DeepSeek) deferred to iter69 conditional.

---

## 11. Recovery of Existing Dense as Limit

`--moe-ffn 1 1` with `d_ff = 4m, α_LB = 0`: one expert, router outputs `[1.0]`, identity permutation. Up to ~0.5% router-overhead, bit-equivalent to dense. Useful as a Gate-0a control to isolate **routing overhead** from **sparsity gains**. iter62 runs both `--moe-ffn 1 1` and dense baseline.

---

## 12. Honest Risk Assessment

MoE FFN is the **highest-effort, highest-variance** of the three post-iter61 candidates.

Attempt because: only candidate with credible **2× standalone headroom**; mechanism is mature (Switch/Mixtral/DeepSeek-V3); existing infra (BF16, fused Adam-int8, reversible-flow) compatible without redesign.

Caution because: 5-8 iters, +10-18% expected (cumulative) with **std-err ≈±10%**; production VRAM is binding (iter-bench ≠ production); routing failure modes need hyperparameter sweeping; even +18% on cumulative ~1.23× from iters 1-61 stays far from the 10× brief target.

**Recommendation**: pursue MoE only if the other two paradigms look weaker on first principles; otherwise revisit on their underperformance. First commitment: **iter62 (forward only, ~6-8 GPU-hrs)** — cleanest go/no-go for the full arc.

---

## 13. Appendix — Concrete Code Paths

| Existing path                                                                                | What MoE adds/replaces                                  |
|----------------------------------------------------------------------------------------------|---------------------------------------------------------|
| `Backend/Machine Learning/Networks/cuda/gpu_blas.cu:1100` (`sgemm_rowmajor_fast16bf_impl`)   | Reused unchanged for per-expert GEMMs                   |
| `Backend/Machine Learning/Networks/cuda/gpu_transformer_state.cu:423-434` (`b.W1/W2/...`)    | Replace with K-indexed expert arrays                    |
| `Backend/Machine Learning/Networks/cuda/gpu_transformer_state.cu:165-169` (VRAM accounting)  | Multiply FFN slots by K                                 |
| `Backend/Machine Learning/Networks/sgd_transformer.cpp` (dense FFN fwd/bwd)                  | Dispatch to `sgd_transformer_moe.cpp` when `--moe-ffn`  |
| `Backend/Machine Learning/Networks/checkpoint_persistence.cpp`                               | Schema version bump for K-expert layout                 |
| `glades-trainer/trainer/chiron_main.cpp`                                                     | New flags `--moe-ffn K top_k`, `--moe-lb-weight`        |

No changes needed: `transformer_ops.h` (attention untouched), `transformer_kernels.h` (CPU-eval SIMD only), `network.h`/`NNInfo` (per-layer flag), `gpu_kernels.cu` fused Adam-int8 (per-tensor agnostic).

---

*End of design.*
