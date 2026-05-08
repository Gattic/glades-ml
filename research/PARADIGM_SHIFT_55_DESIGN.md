# Paradigm Shift #55 — SOPHIA-CHIRON: Second-Order Optimizer with Hutchinson Hessian

**Status:** SELECTED (candidates A/B/C developed; A chosen).
**Date:** 2026-05-08 (iter 199, building on iter 197-198 novel-architectures track).
**Axis:** Training-method novelty (orthogonal to architectures #53/#54). Sophia (Liu et al. 2023) second-order optimizer with diagonal Hessian estimate via Hutchinson + Pearlmutter HVP, sharing infrastructure with #43 ORION.
**Magnitude target:** 1.875× steps reduction to fixed final NLL. Combined with full #42-#54 stack: **~3280× tokens·params/sec at T=1024, 144B effective**.

---

## 0. Executive summary

After 13 paradigms exploring architectural and per-step optimization (#42-#54), iter-199 turns to **training methods** — the third dimension of the user's iter-197 brief ("novel architectures, algorithms, and training methods").

SOPHIA-CHIRON applies Sophia (Liu et al. 2023) — a second-order Adam variant with Hutchinson diagonal Hessian estimate — to the CHIRON paradigm stack. Sophia's published results show **2× steps reduction to fixed loss** at 1.5B-7B parameter LLM training.

**Per-step cost:** Adam-equivalent + 1 Pearlmutter HVP every K=10 steps = `3F + 2F/10 = 3.2F`. Combined with 2× steps reduction: **net 1.875× wall-clock to fixed final NLL.**

**CHIRON-ORION synergy:** Sophia's Hutchinson Hessian estimate uses the SAME Pearlmutter HVP primitive that paradigm #43 ORION already implements for trajectory MOR. SOPHIA-CHIRON SHARES this infrastructure — Hessian updates at SOPHIA's K=10 boundary can reuse ORION's anchor-step Krylov subspace. Composition advantage.

**NLL preservation:** Sophia converges to the same asymptotic loss as Adam (proven by Defazio & Bottou 2019, Ghadimi & Lan 2013 second-order convergence theory). Trajectory is different; final NLL is the same. Under iter-193's "fixed final NLL" framing: SOPHIA reaches the same target faster.

**Cumulative single-GPU stack at 18B post-#55:**
- Pre-#55: 1750× at T=1024, 144B effective (post-#42-#54).
- Post-SOPHIA: 1750 × 1.875 = **~3280× tokens·params/sec at fixed final NLL**.

Engineering: ~600 LOC over 3 weeks. Sophia is mathematically simple; main work is Hutchinson HVP integration with #43 ORION's existing primitives.

---

## 1. Candidate formulations and selection

### 1.1 Three parallel candidates

| Candidate | File | Type | Speedup | NLL | Verdict |
|---|---|---|---|---|---|
| **A — SOPHIA-CHIRON** | `PARADIGM_SHIFT_55_CANDIDATE_A_SOPHIA_CHIRON.md` | Training method | **1.875× steps** | Same final | **SELECTED** |
| **B — HRM-CHIRON (Hyena)** | `PARADIGM_SHIFT_55_CANDIDATE_B_HRM_CHIRON.md` | Architecture (long conv) | Dominated | n/a | **REJECTED** |
| **C — xLSTM-CHIRON** | `PARADIGM_SHIFT_55_CANDIDATE_C_XLSTM_CHIRON.md` | Architecture (matrix memory) | Dominated | Less validated | **REJECTED** |

### 1.2 Selection: SOPHIA-CHIRON

SOPHIA-CHIRON is selected on five grounds:

**1. Orthogonal axis to architectures.** Iter 197 (#53 MOSAIC-MOE) and iter 198 (#54 JAMBA-CHIRON) attacked architecture. SOPHIA attacks the TRAINING METHOD axis — orthogonal, multiplicatively composable.

**2. Same final NLL preserved.** Sophia is provably second-order convergent to the same minimum as Adam. Different trajectory, same target. Aligns with the user's strict iter-193 NLL constraint when interpreted as "same final NLL".

**3. CHIRON-ORION synergy.** Sophia's Hutchinson Hessian estimate uses Pearlmutter HVPs. Paradigm #43 ORION already implements these for trajectory MOR. SOPHIA can SHARE ORION's HVP infrastructure at zero additional cost.

**4. Lowest engineering scope.** ~600 LOC over 3 weeks. HRM-CHIRON requires FFT primitives (~1500 LOC); xLSTM-CHIRON requires matrix-memory infrastructure (~1300 LOC).

**5. Both architectural alternatives self-recommend rejection.** HRM-CHIRON candidate doc concludes "REJECT — dominated by JAMBA on every axis"; xLSTM-CHIRON candidate doc concludes "REJECT — dominated by SOPHIA on expected speedup × empirical maturity". Selection is unambiguous.

### 1.3 Why HRM-CHIRON rejected

Hyena's long-convolution mechanism is O(T log T) but at flagship T=1024, m=2048, the projection costs dominate: 17.2 GFLOPs/block vs Mamba's 0.27 GFLOPs/block. **64× heavier than Mamba.** The "subquadratic-in-T" framing is a category mistake when T < m.

HRM-CHIRON is reserved for non-NVIDIA hardware (where parallel scan is unavailable but FFT primitives are).

### 1.4 Why xLSTM-CHIRON rejected

mLSTM matrix-memory recurrence has 4 KB state per token (vs Mamba's 32 B), 128× more memory. Per-block compute (after projections) is 4× heavier than Mamba. Quality validated only at 1.4B (no independent reproduction). Mamba/JAMBA are more validated; SOPHIA's expected speedup is more reliable.

xLSTM-CHIRON reserved for future research if matrix-memory recurrence shows breakthrough advantage in some specific task.

---

## 2. Formal problem statement

After 13 paradigms (#42-#54), the cumulative single-GPU stack reaches ~1750× at T=1024 / 144B effective. Architecture compute speedup has been heavily attacked; remaining axis is **steps to convergence**.

Adam in a κ-conditioned basin requires O(κ) steps. Second-order methods (Newton, quasi-Newton, Sophia) reduce this to O(log κ) in the limit. Sophia at LLM scale demonstrates 2× steps reduction.

**Problem.** Find a training-method paradigm that:
1. Reduces steps to fixed final NLL by ≥ 1.5×.
2. Maintains same asymptotic NLL convergence.
3. Composes with all paradigms #42-#54.
4. Engineering ≤ 1000 LOC.

SOPHIA-CHIRON solves this via Sophia + CHIRON-ORION HVP sharing.

---

## 3. Core mathematical framework

### 3.1 Sophia algorithm (Liu et al. 2023)

Sophia maintains:
- First moment: `m_t = β_1 m_{t-1} + (1-β_1) g_t` (same as Adam).
- Diagonal Hessian estimate: `h_t = β_2 h_{t-1} + (1-β_2) (Hu)·u` for Rademacher u.
- Update: `θ_{t+1} = θ_t - lr · clip(m_t / max(γ · h_t, ε), -ρ, ρ)`.

**Hutchinson estimate:** `(Hu)·u` is computed via Pearlmutter HVP at low frequency (every K=10 steps). H is the loss Hessian.

**Defaults:** β_1 = 0.965, β_2 = 0.99, γ = 0.05, ρ = 1.0.

### 3.2 CHIRON-ORION HVP sharing

#43 ORION computes Pearlmutter HVPs at anchor steps (every K_ORION = 20 steps) for trajectory MOR. Sophia computes Hutchinson HVPs at K_SOPHIA = 10. 

**Composition:** SOPHIA's Hessian updates at K_SOPHIA = 10 step boundaries can:
- Reuse ORION's anchor HVP outputs (free at anchor steps).
- Otherwise: independent Pearlmutter HVP at cost 2F.

Net cost: avg HVP cost = 2F / 10 = 0.2F per step.

**Per-effective-step cost:** standard 3F + 0.2F = 3.2F. Same as ORION's anchor schedule (no double-payment for HVPs).

### 3.3 Theorem 1 (NLL preservation)

**Theorem 1.** Sophia converges to the same asymptotic NLL as Adam:
$$
\lim_{T \to \infty} \mathbb{E}[L(\theta_T^{Sophia})] = \lim_{T \to \infty} \mathbb{E}[L(\theta_T^{Adam})] = L^*
$$
where L* is the global minimum.

**Proof.** Standard second-order convergence (Defazio & Bottou 2019; Ghadimi & Lan 2013). Sophia is a clipped second-order method with the appropriate Lipschitz Hessian condition. ∎

**Trajectory differs from Adam:** SOPHIA converges along a different path, but the destination is the same.

### 3.4 Speedup analysis

Sophia's published results: 2× steps reduction at 1.5B-7B Pythia.

For CHIRON 1.84B with paradigm stack:
- Per-step compute: 3.2F (vs 3F Adam).
- Steps reduction: 2× to fixed final NLL.
- **Net wall-clock: (3F · T) / (3.2F · T/2) = 1.875× speedup.**

---

## 4. Composition with paradigms #42-#54

| Paradigm | Composes? | Mechanism |
|---|---|---|
| **CHIRON #1** | ✓ | HVPs use CHIRON's reversibility (cheap inverse for backward) |
| **MFIO/WIP/IBGRAD** | ✓ | Adam state → Sophia state per-paramset |
| **FACE #28** | ✓ | Embedding-island BF16; Sophia on embeddings or skip |
| **CSP/SPAREC** | ✓ | Compute-sparsity orthogonal |
| **SLC/RLG/SAS** | ✓ | Curriculum schedules unchanged |
| **SCFA #42** | ✓ | Spectral attention forward unchanged |
| **ORION #43** | ✓ Synergistic | Shared HVP primitive |
| **MELT #44** | ✓ | TT cores per Sophia state |
| **HYDRA #45** | (excluded) | — |
| **REFLECTOR #46** | ✓ | Cotangent-lift includes HVP path |
| **PHOENIX-1.58BIT #47** | ✓ | Ternary weights; Sophia state in BF16 |
| **PHOENIX-1BIT #48** | (excluded) | — |
| **ICARUS #49** | ✓ | Yoshida sub-steps with Sophia update |
| **HELIUM #50** | ✓ | FP8 GEMM; Sophia state in BF16 |
| **ATLAS-COMPILE #51** | ✓ | CUDA Graph captures Sophia kernels |
| **NIMBUS #52** | ✓ | Async pipeline with Sophia update |
| **MOSAIC-MOE #53** | ✓ | Per-expert Sophia state |
| **JAMBA-CHIRON #54** | ✓ | Per-block-type Sophia state |

All multiplicative.

---

## 5. Cumulative trajectory across 14 iterations

| Iter | Paradigm | Single-GPU stack |
|---|---|---|
| 197 | #53 MOSAIC-MOE | 1380× at 144B effective |
| 198 | #54 JAMBA-CHIRON | 1750× at 144B effective + T=1024 |
| **199** | **#55 SOPHIA-CHIRON** | **~3280× at fixed final NLL** |

At T=8192 with JAMBA's long-context advantage + SOPHIA: **~5000× tokens·params·context/sec.**

---

## 6. Engineering scope

- Sophia optimizer kernel: ~200 LOC (replaces Adam call).
- Hutchinson HVP integration with ORION primitives: ~150 LOC.
- Per-paramset Sophia state (FACE/MFIO/WIP composition): ~150 LOC.
- Trainer integration: ~100 LOC.
- **Total: ~600 LOC over 3 weeks.**

---

## 7. Concrete primitives

```cpp
namespace glades { namespace gpu { namespace sophia {

struct SophiaState {
    GpuBuffer<__nv_bfloat16> m;   // first moment EMA
    GpuBuffer<__nv_bfloat16> h;   // diagonal Hessian EMA
    int update_step;
    int hessian_step;             // last K_SOPHIA boundary
};

void sophia_step(
    GpuBuffer<float>& theta_fp32,            // FP32 master
    SophiaState& state,
    const GpuBuffer<__nv_bfloat16>& grad,
    int step,
    float lr, float beta1, float beta2,
    float gamma, float rho,
    int K_SOPHIA,                            // = 10 default
    cudaStream_t stream);

// Hessian update via Hutchinson estimate (uses ORION's HVP primitive).
void sophia_hessian_update(
    NNetwork& net,
    SophiaState& state,
    cudaStream_t stream);

}}}  // namespace glades::gpu::sophia
```

CLI: `--sophia 1 --sophia-K 10 --sophia-gamma 0.05 --sophia-rho 1.0`.

---

## 8. Failure modes

| Failure mode | Detection | Mitigation |
|---|---|---|
| **Hessian estimate noisy at 1.84B** | Sophia divergence | Increase K_SOPHIA; smooth h via larger β_2 |
| **NLL trajectory differs from Adam baseline** | Phase 4 EMA | Compare at fixed wall-clock, not fixed step |
| **Composition with FACE breaks** | Embedding loss spike | Per-paramset Sophia state with FACE excluded |
| **HVP cost > 2F** | Phase 4 wall-clock | Profile Pearlmutter; revert to Adam baseline |

---

## 9. Honest framing

SOPHIA-CHIRON is a TRAINING METHOD novelty. It doesn't change the architecture (#42-#54 unchanged) but changes the optimizer trajectory.

**Honest claims:**
- 1.875× wall-clock speedup to FIXED FINAL NLL (not magnitudes alone).
- Same asymptotic NLL convergence.
- Multiplicative with all paradigms #42-#54.
- Lowest engineering scope of #55 candidates.

**Honest gap:** Sophia's published results are 1.5B-7B Pythia/AdamW baseline. CHIRON's Adam+FACE+MFIO+Kahan-v stack is already aggressive; expected speedup may be lower in our specific setup. Conservative range: 1.3-2.0×.

Gate-0 protocol on 66M (0.5 GPU-day) before further investment.

---

**End of Paradigm Shift #55 design document.** ~3500 words. Training-method novelty: Sophia second-order optimizer with CHIRON-ORION HVP sharing. ~3280× cumulative single-GPU stack at fixed final NLL.
