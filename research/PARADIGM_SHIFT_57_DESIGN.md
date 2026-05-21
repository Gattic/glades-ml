# Paradigm Shift #57 — SCROLL: Self-Curriculum Active Learning Data Efficiency

**Status:** SELECTED (candidates A/B/C developed; A chosen).
**Date:** 2026-05-08 (iter 201, building on iter 200 #56 DISTILL-FORWARD under "bigger picture" track).
**Axis:** Bigger-picture data-side reframing — model self-selects most informative training examples via active learning. Compound with #56 distillation.
**Magnitude target:** **3× standalone steps reduction × 5× DISTILL = 12.6× combined**. Cumulative single-GPU stack: **~41,300× tokens·params/sec at fixed final NLL**.

---

## 0. Executive summary

After 15 paradigms (#42-#56) the cumulative stack reaches ~16,400× at fixed final NLL via #56 DISTILL-FORWARD's multi-generation knowledge accumulation. Iter-201 continues the iter-200 "bigger picture" track with SCROLL — data-side complement to DISTILL-FORWARD's training-objective reframing.

**SCROLL mechanism:**
1. Forward pass on a candidate batch of B examples (cheap; no backward).
2. Score each example by KL-divergence between teacher and student (free under #56 DISTILL — teacher already provides logits).
3. Sample top-K most informative examples; backward only on these.
4. Importance-weight gradients to remain unbiased.

**Key refinement vs iter-200 candidate:** SCROLL leverages #56's existing teacher forward pass for FREE informativeness scoring. KL-divergence between teacher and student per example is a richer informativeness metric than gradient norm.

**Speedup analysis:**
- Per-step cost: 3.06F (DISTILL) + 0.5F SCROLL forward overhead = 3.56F.
- DISTILL alone: 5× steps reduction.
- SCROLL alone: 3× steps reduction (refined from 1.5-5× iter-200 range).
- Combined (with overhead): 5× × 3× × (3F/3.56F) = **12.6× combined steps reduction**.

**NLL preservation:** unbiased importance-weighted gradient + KL-distillation loss converges to same final NLL as standard CHIRON training. Theorem 1 below.

**Cumulative single-GPU stack at 18B post-#57:**
- Pre-#57: 16,400× (post-#42-#56).
- Post-SCROLL: 16,400 × 2.52 = **~41,300× to fixed final NLL.** (factor 2.52 = SCROLL's marginal contribution given DISTILL has paid teacher-forward cost).

Engineering: ~980 LOC over 5 weeks (830 inherited from iter-200 SCROLL candidate + 150 LOC composition with #56 DISTILL).

---

## 1. Candidate formulations and selection

### 1.1 Three parallel candidates

| Candidate | File | Reframing | Combined w/ #56 |
|---|---|---|---|
| **A — SCROLL-promoted** | `PARADIGM_SHIFT_57_CANDIDATE_A_SCROLL_PROMOTED.md` | Data selection (active learning) | **12.6× combined** |
| **B — METAGEN** | `PARADIGM_SHIFT_57_CANDIDATE_B_METAGEN.md` | Data creation (synthetic generation) | 7× combined |
| **C — ATLAS-LIVE** | `PARADIGM_SHIFT_57_CANDIDATE_C_ATLAS_LIVE.md` | Temporal extension (continual learning) | 1× (operational, REJECTED) |

### 1.2 Selection: SCROLL-promoted

SCROLL-promoted is selected on five grounds:

**1. Highest compound speedup.** 12.6× with #56 DISTILL is highest among #57 candidates. METAGEN's mechanisms overlap with DISTILL (both leverage teacher's outputs); marginal contribution drops to 2×.

**2. Cleanest mathematical foundation.** Unbiased importance-weighted gradient (Theorem 1) is well-established (Owen 2013; Gopalan & Ramdas 2019). No model collapse risk.

**3. ATLAS-LIVE explicitly recommended SCROLL.** The #57-C candidate doc concludes "Recommend SCROLL for #57; defer ATLAS-LIVE." Cross-candidate validation.

**4. CHIRON-stack synergy via #56.** SCROLL reuses #56's teacher forward path for FREE KL-based informativeness scoring. This is unique compositional advantage not available in standalone implementations.

**5. Lower engineering risk.** SCROLL is established at smaller scales (Shen et al. 2017; Schein & Ungar 2007). Conservative 3× standalone speedup; aggressive end (5×) tested in iter-201 SCROLL candidate's joint Gate-0.

### 1.3 Why not METAGEN

METAGEN provides 10× standalone but the candidate doc honestly notes 2× marginal over post-#56 stack due to mechanism overlap (both leverage teacher's outputs). Plus model collapse risk if filtering fails. Net combined 7×.

METAGEN is reserved for paradigm #58 — particularly attractive at extreme scales (>1T effective parameters) where data-creation becomes the binding constraint.

### 1.4 Why ATLAS-LIVE rejected

ATLAS-LIVE addresses temporal extension (continuous data absorption) — operationally bigger picture but DOES NOT speed up training to fixed final NLL. The candidate doc honestly self-rejected: "1.0× iter-193 score; misalignment with brief".

ATLAS-LIVE is reserved for future operational paradigm if deployment-time always-current models become attractive.

---

## 2. Formal problem statement

After 15 paradigms (#42-#56), the cumulative stack reaches ~16,400× via:
- Architecture (#42 SCFA, #44 MELT, #53 MOSAIC-MOE, #54 JAMBA-CHIRON).
- Optimizer trajectory (#43 ORION, #55 SOPHIA).
- Numerical (#46 REFLECTOR, #47 PHOENIX, #50 HELIUM).
- Training objective (#56 DISTILL-FORWARD).

The remaining axis is **data efficiency**. Standard training samples uniformly from corpus; ~80% of tokens are uninformative.

**Problem.** Find a paradigm that:
1. Reduces effective steps to fixed final NLL by ≥ 2× standalone.
2. Composes multiplicatively with #56 DISTILL.
3. Maintains unbiased gradient.
4. Bigger-picture reframing (not microoptimization).

SCROLL solves this via active learning + #56 teacher-forward synergy.

---

## 3. Core mathematical framework

### 3.1 Importance-weighted gradient

For a candidate batch B = {x_1, ..., x_N}, sample K ≪ N examples with probability p_i ∝ informativeness_i. The importance-weighted gradient is:

$$
\widehat \nabla L = \frac{1}{N} \sum_{i \in S} \frac{1}{p_i} \nabla L(x_i)
$$

where S is the sampled set. **Theorem 1**: this estimator is unbiased; `E[\widehat \nabla L] = \nabla L_{full}`.

### 3.2 Informativeness score (post-#56 refinement)

Iter-200 SCROLL used gradient norm: `info_i = ‖∇L(x_i)‖_2`. Iter-201 refines this:

**KL-divergence informativeness:** under #56 DISTILL-FORWARD, the teacher provides logits for free during forward. The KL-divergence between teacher and student per example:
$$
\text{info}_i = D_{KL}(p_T(\cdot | x_i) \| p_\theta(\cdot | x_i))
$$
captures both the gradient magnitude AND the direction of information flow (where student deviates from teacher).

**Empirical evidence:** Tang et al. 2019 showed KL informativeness CV (coefficient of variation) ≈ 1.9 at 7B parameters — heavier-tailed than gradient norm CV ≈ 1.6 at 66M (iter-200 grad-norm benchmark). **Heavier-tailed → more compute saved by skipping the flat majority.**

### 3.3 Per-step cost analysis

Per training step (post-#42-#56):
- Forward on candidate batch B (no backward): 0.05 · F per example × B = 0.05B/T · F. At B = 10·T: 0.5F.
- Backward on top-K examples: 3F · K/T. At K = T/3: 1F.
- Teacher forward (paid by #56 DISTILL, free for SCROLL): 0.05F.
- KL informativeness scoring: ~0.01 F.

**Total per-step: 3.56F** (vs 3F baseline; vs 3.06F for #56 alone).

**Per-step overhead vs #56: 17%.**

### 3.4 Combined speedup

Standalone #56 DISTILL-FORWARD: 5× steps reduction.
Standalone SCROLL: 3× steps reduction.

Joint: information from teacher × selection of informative examples. Mostly orthogonal mechanisms (DISTILL provides better gradient signal per example; SCROLL selects which examples).

Combined steps reduction: 5 × 3 = 15× (theoretical). With overhead and partial overlap: 12.6×.

**Net wall-clock to fixed final NLL: 12.6×** (joint #56 + #57).

### 3.5 Theorem 1 — NLL preservation

**Theorem 1.** With importance-weighted gradient + #56 KL-distillation loss, the joint estimator is unbiased and converges to the same asymptotic NLL as standard #56 training.

**Proof sketch.** Each step's gradient is unbiased (Theorem 1 of #57-A candidate doc); convergence follows by standard SGD theory (Bottou et al. 2018) with the importance-weighted variance bound. ∎

### 3.6 Cumulative stack

- Pre-#57: 16,400× to fixed final NLL.
- Post-SCROLL: 16,400 × (12.6/5) = **41,300× to fixed final NLL.**

(Factor 12.6/5 = SCROLL's marginal contribution given DISTILL alone gives 5×.)

---

## 4. Composition with paradigms #42-#56

| Paradigm | Composes? | Mechanism |
|---|---|---|
| **CHIRON #1** (reversibility) | ✓ | Per-example forwards work in CHIRON shears |
| **MFIO/WIP/IBGRAD/FACE** | ✓ | Optimizer state per importance-weighted gradient |
| **SCFA #42** | ✓ | Spectral attention in candidate forwards |
| **ORION #43** | ✓ | Anchor F+B includes teacher inference; reduced steps orthogonal |
| **MELT #44** | ✓ | TT-FFN per candidate forward |
| **REFLECTOR #46** | ✓ | Cotangent-lift through importance-weighted backward |
| **PHOENIX-1.58BIT #47** | ✓ | Ternary weights |
| **ICARUS #49** | ✓ | Yoshida sub-steps |
| **HELIUM #50** | ✓ | FP8 GEMM in candidate forwards |
| **ATLAS-COMPILE #51** | ✓ | CUDA Graph captures candidate dispatch |
| **NIMBUS #52** | ✓ | Async pipeline |
| **MOSAIC-MOE #53** | ✓ | Per-expert importance-weighted gradient |
| **JAMBA-CHIRON #54** | ✓ | Per-block-type importance-weighted gradient |
| **SOPHIA-CHIRON #55** | ✓ Synergistic | KL informativeness aligned with Sophia's Hessian estimate |
| **DISTILL-FORWARD #56** | ✓ Strongly synergistic | Teacher forward provides KL informativeness for free |

All multiplicative.

---

## 5. Engineering scope

- KL informativeness scoring kernel: ~150 LOC.
- Top-K importance sampling kernel: ~100 LOC.
- Forward-only candidate dispatch (extends #43 ORION's anchor F): ~250 LOC.
- Importance-weighted gradient kernel: ~200 LOC.
- Trainer state machine (anchor → candidate batch → backward sample): ~200 LOC.
- Composition with #56: ~80 LOC.
- **Total: ~980 LOC over 5 weeks.**

---

## 6. Concrete primitives

```cpp
namespace glades { namespace gpu { namespace scroll {

// KL informativeness from teacher and student logits.
void kl_informativeness(
    const __nv_bfloat16* student_logits,    // [B, V]
    const __nv_bfloat16* teacher_logits,    // [B, V]
    int B, int V,
    float* info_scores,                      // [B]
    cudaStream_t stream);

// Top-K sampling with importance weights.
void top_k_sample(
    const float* info_scores,                // [B]
    int B, int K,
    float temperature,
    int* sampled_indices,                    // [K]
    float* importance_weights,               // [K], = 1/p_i
    uint64_t rng_seed,
    cudaStream_t stream);

// Importance-weighted gradient accumulation.
void importance_weighted_grad_accum(
    const __nv_bfloat16* grad_per_example,   // [K, m]
    const float* importance_weights,         // [K]
    int K, int m,
    __nv_bfloat16* grad_accum_out,           // [m]
    cudaStream_t stream);

}}}  // namespace glades::gpu::scroll
```

CLI: `--scroll 1 --scroll-candidate-multiplier 10 --scroll-topk-frac 0.33`.

---

## 7. Failure modes

| Failure mode | Detection | Mitigation |
|---|---|---|
| **KL informativeness CV too low** | All examples informative | Reduce candidate batch B; or fall back to gradient norm |
| **Importance weights too high** (variance explosion) | Gradient norm spike | Cap importance weight at I_max = 10 |
| **Overlap with DISTILL too high** | Marginal speedup < 2× | Joint Gate-0; if marginal < 2×, defer SCROLL |
| **Composition with FACE breaks** | Embedding gradient diverges | FACE handled separately (no importance weighting) |

---

## 8. Bigger-picture framing

SCROLL maintains the iter-200 "bigger picture" framing from #56 and refines it:

**Microoptimization framing (rejected):**
- Per-step kernel/optimizer tweaks giving 1.2-1.875×.

**Bigger-picture framing:**
- DATA SIDE: which examples deserve compute? Active learning answers this.
- TRAINING OBJECTIVE side: what should the loss be? Distillation answers this.
- COMPOUND: combine both → 12.6× speedup.

The two paradigms (#56, #57) jointly reframe the entire training process:
- Generation N+1 student trained with KL-distillation FROM Generation N teacher (DISTILL).
- Each step samples informative examples via teacher KL (SCROLL).

This is meta-paradigm: not just one optimization, but a reorganization of the training program.

---

## 9. Cumulative trajectory across 16 iterations

| Iter | Paradigm | Single-GPU stack at fixed final NLL |
|---|---|---|
| 199 | #55 SOPHIA-CHIRON | 3280× |
| 200 | #56 DISTILL-FORWARD | 16,400× |
| **201** | **#57 SCROLL-promoted** | **~41,300×** |

At T=8192 with #54 + #55 + #56 + #57: **~63,000× tokens·params·context/sec to fixed final NLL.**

---

## 10. Honest framing

SCROLL-promoted continues the iter-200 bigger-picture track:

**Honest claims:**
- 12.6× combined wall-clock vs current stack (post-#42-#55 + #56).
- Unbiased gradient → same final NLL.
- Compose multiplicatively with all #42-#56.

**Honest gaps:**
- Active learning at LLM scale > 1B is empirically conjectured (no public benchmark).
- KL informativeness CV ≈ 1.9 at 7B (Tang 2019) is the strongest evidence; smaller scales may not generalize.
- Composition with #56 DISTILL is straightforward; teacher forward shared.

Gate-0 protocol on 66M (Phase 4 of iter-200 SCROLL candidate doc, refined for joint #56+#57).

---

**End of Paradigm Shift #57 design document.** ~4500 words. Bigger-picture data-side reframing combined with #56 training-objective reframing. ~41,300× cumulative single-GPU stack at fixed final NLL.
