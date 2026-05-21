## Iter 120 — Measurement-grounded ceiling META (post iter 118/119)

**Date**: 2026-05-21
**Iter**: 120 (META, no new code)
**Branch**: vesta5
**Verdict**: **META** — nsys measurement reveals launch overhead is only ~1.4% of step wall (not the 2-3% predicted in iter 117). CUDA Graph capture's realistic upper bound is +0.7-1% wall. Combined with iter 118/119 confirming FP32/BF16-without-TC FA is 2.75× slower than cuBLAS-TC, the remaining single-arc options have sub-3% expected value at high engineering cost. **iter 116 ship (+12.56% over iter 94, 1.86× cumulative) is the practical ceiling** for this CHIRON 1B configuration on RTX 4080 SUPER. Further improvements require architectural or hardware changes.

---

## What iter 120 measured

nsys profile of 25-step bench at production config (iter 116 ship stack):

```
Time (%)  Total (ns)       Calls    Avg     Name
  96.2  12,874,858,816    13,485  954,754  cudaStreamSynchronize  ← blocking waits (not pure overhead)
   1.3     172,847,334    46,540    3,714  cudaLaunchKernel
   1.6     207,964,104       335  620,788  cudaMemcpy             ← host roundtrips
   0.4      49,659,839     1,158   42,884  cudaMalloc             ← startup only
   0.2      31,067,753     6,506    4,775  cudaMemset             ← startup-heavy
   0.1      11,530,553     3,797    3,037  cudaMemsetAsync
   0.1       7,440,916     1,105    6,734  cudaMemcpyAsync
   ...
```

**Pure launch overhead per step** (steady-state, excluding startup):
- `cudaLaunchKernel`: 46,540 calls / 25 steps × 3,714 ns avg = **6.9 ms/step**
- `cudaMemsetAsync`: 3,797 calls / 25 steps × 3,037 ns avg = **0.5 ms/step**
- `cudaMemcpyAsync`: 1,105 calls / 25 steps × 6,734 ns avg = **0.3 ms/step**
- Other small APIs: **~0.4 ms/step**
- **Total**: ~8.1 ms/step

Step wall: 596 ms/step (14.9 s / 25 steps).

**Launch overhead percentage: ~1.4% of step wall.**

## Why this revises iter 117 META

iter 117 META estimated CUDA Graph capture at +2-3% wall, based on assumption that launch overhead was ~14-19 ms/step (~2-3%). Actual measurement is ~8 ms/step (~1.4%).

**Realistic CUDA Graph upper bound**: 0.7-1.0% wall savings.
- Graph launch itself has cost (~5 µs per graph launch)
- Not all kernels can be captured cleanly (host-mediated grad clip checkpoints)
- Memcpy/memset patterns have some unavoidable overhead

This is below the iter 60 +3% bar and barely above measurement noise.

## Combined picture: remaining options at iter 116 ship

| option | predicted wall | effort | risk | EV/effort |
|---|---:|---|---|---|
| **CUDA Graph capture** | +0.7-1.0% | 150-200 LOC | medium (capture mode complications) | **LOW** |
| **WMMA-based FA kernel** | +0% to +8% | 300-500 LOC, 1-2 days | HIGH (cuBLAS BF16-TC is well-tuned at SCFA dims) | **uncertain** |
| **FP8 precision** | +5-10% | multi-iter + library port | BLOCKED on CUDA 12.0 toolkit (need 12.3+) | **blocked** |
| **MoE / conditional compute** | architectural | multi-week | architectural change | **out of scope** |
| **GQA / sliding window** | architectural | multi-week | architectural change | **out of scope** |
| **Cross-stream parallelism** | +0% to +3% | ~200 LOC | iter 71 NULL on Ada (TC saturation) | **already NULL** |

The "remaining" single-arc options all have sub-3% expected value (CUDA Graph) OR uncertain positive value (WMMA) OR are blocked (FP8) OR require architectural change.

## WMMA-based FA: detailed risk analysis

iter 119 confirmed that FP32 SIMT compute is the bottleneck (2.75× slower than cuBLAS BF16-TC). A WMMA implementation would use BF16 tensor cores in the matmul, FP32 accumulators in smem.

**Best case**: matches cuBLAS BF16-TC throughput AND saves the materialized P matrix (~64 MB read/write per call eliminated) AND saves softmax kernel launch. Wall savings: maybe +5-8% on the ~10-15% of step wall that's inner attention.

**Realistic case**: matches cuBLAS for QK^T but loses some efficiency on PV (which cuBLAS already does at 128×128 tiles for [16, 1024, 256]); FA fusion savings = ~3-5% on inner attention = ~0.3-0.8% wall total.

**Worst case**: cuBLAS-BF16-TC at these specific dims is hand-tuned by NVIDIA cuBLAS team; a hand-rolled WMMA kernel runs 10-20% slower; FA fusion savings can't compensate; net negative.

The probability distribution skews toward "modest positive at best" given:
- cuBLAS already uses TC efficiently for SCFA inner dims
- T_inner=1024 is small by FA standards (typical FA usage is T≥4096)
- dH=256 is unusual; standard FA-2 templates target dH≤128

## The honest conclusion

After iter 95-117 ralph-loop session (+12.56% wall, 7 strict-bar PASSes, 1.86× cumulative) and iter 118-120 FA arc (math validated, wall not improvable without WMMA), the iter 116 ship represents the practical engineering ceiling for this configuration:

- **CHIRON 1B**: m=2048, L=24, nH=16, dH=256, T=16384, BF16 weights, int8-Adam, SCFA k=1024
- **Hardware**: RTX 4080 SUPER (Ada, SM 8.9, 16 GB VRAM)
- **Software**: CUDA 12.0 toolkit (FP8 cuBLAS LT path requires 12.3+)

**Production state at ceiling**:
- iter 116 ship: **28,257 tok/s**, NLL 4.2039 @ 30k
- Cumulative since pre-ralph-loop: **1.86×** (15,200 → 28,257)
- 2.0× target (~30,400): **~93% reached**
- 3× target (~45,600): **requires architectural or hardware change**

## What WOULD break past iter 116 ship

**Hardware change**:
- Hopper SM 9.0 (H100/H200) — FP8 tensor cores at full toolchain support, larger smem, async TMA
- Blackwell SM 10.0 (B100/B200) — FP4/FP6 tensor cores
- Estimated +30-100% over Ada on attention-heavy workloads

**Toolkit upgrade**:
- CUDA 12.3+ unlocks cuBLAS LT FP8 readout path
- Estimated +5-15% on cuBLAS-bound paths (readout, projections)

**Architectural change**:
- **GQA (Grouped-Query Attention)**: reduces K/V projection cost. SCFA already saves outer attention, but inner attention K/V projections could use GQA.
- **MoE (Mixture-of-Experts)**: routing reduces per-token compute. Multi-week implementation.
- **Sliding-window attention**: cap attention to local window. Conflicts with SCFA design.

**Multi-iter scoped work** (still single-config):
- **WMMA-based FA** (uncertain +0-5%): 300-500 LOC, 1-2 days
- **Cross-stream parallelism** (iter 71 was NULL; worth retesting with newer driver): ~200 LOC

## Recommendation

iter 120 = STOP at iter 116 ship. The 1.86× cumulative gain is a strong production deliverable. Further single-config improvements have low expected value at high engineering cost.

If user wants to continue, the highest-EV next step is **WMMA-based FA kernel** despite the uncertainty — it has the largest possible upside (+5-8%) even though the floor is closer to zero. Effort: 1-2 days for forward kernel.

Alternative: stop and revisit when CUDA 12.3 toolchain is available (unlocks FP8 path).

## Session arc closing remarks

The iter 95-120 ralph-loop session achieved its goals:
- 11 PASS (including 7 strict +5% multi-seed bar PASSes)
- 2 FAIL (informative mechanism-class boundaries)
- 5 NULL (calibrated noise floor)
- 7 META (closure + boundary documentation)
- +12.56% wall over iter 94 ship → 1.86× cumulative

The mechanism class "eliminate redundant memory ops" yielded 10 distinct PASS realizations. The session has thoroughly explored single-iter scope at this stack.

## Files

- This document.
- nsys profile data: `/tmp/iter120_profile_overhead.nsys-rep` (deleted after analysis).
- No new code in iter 120 (META only).
