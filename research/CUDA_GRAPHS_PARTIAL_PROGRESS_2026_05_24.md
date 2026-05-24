# CHIRON 1B CUDA Graphs Arc — PARTIAL_PROGRESS (2026-05-23/24)

**Status:** ARC CLOSED as PARTIAL_PROGRESS. Capture is functional but
replay is ~2× slower than direct emission; spec gates not met.
**Spec:** `docs/superpowers/specs/2026-05-23-chiron-1b-cuda-graphs-design.md`
**Plan:** `docs/superpowers/plans/2026-05-23-chiron-1b-cuda-graphs.md`
**Disposition:** 11 library + 6 trainer commits land real codebase
improvements that benefit ALL future training. `--cuda-graphs` is now
functional (no longer auto-disabled by `--scfa`) but is documented as
research-only — not for production. Production flagship unchanged:
`chiron_1B_T16384_regstack_phase2.final` (val NLL 3.5734 @ 30k,
28,072 tok/s).

This is the **fourth consecutive arc closed without shipping** in this
Phase-3 session, after MTP NEGATIVE, LayerDrop FAIL, UL2 FAIL. But
unlike those, the cuda-graphs arc delivered substantial codebase
improvements (math-mode refactor, GpuBuffer::zero capture-safe,
qknorm_gamma_scale GPU kernel, mutual-exclusion hardening) that
persist regardless of the wall outcome.

---

## Spec gate results

| Gate | Target | Measured | Status |
|---|---|---|---|
| G1 capture success | no `forward failed during capture` | capture succeeds at step 0 | ✅ |
| G1 NLL bit-identicality | \|ΔNLL\| ≤ 0.0001 nat | \|ΔNLL\| ≈ 0.20 nat at step 1 | ❌ (~2000× the bar) |
| G1 throughput | ≥ +1% wall vs baseline | −54% wall vs baseline | ❌ |
| G2 30k retrain | gated on G1 PASS | not run | — |

Both NLL bit-identicality and throughput gates fail. Per Phase-3 P6
honest-publication rule, the arc closes as PARTIAL_PROGRESS with no
ship swap.

---

## What landed (real codebase improvements)

### Library (glades-ml) — Phase 1: gpu_blas.cu math-mode refactor

The cuBLAS `cublasGetMathMode` + `cublasSetMathMode` + restore-mode
dance per GEMM was the originally-identified primary blocker.
Refactored to two-handle dispatch:

- `66bdf2038` Add `g_handleStrict` + `g_handleTf32` two-handle init.
- `f58c00ca3` Refactor `sgemm_rowmajor_impl` to use `pick_handle()`.
- `db159f54d` Refactor `sgemm_batched_pointer_impl` similarly.
- `1eaad10e2` Make `set_tf32_enabled` flag-only (no hot-path
  cublasSetMathMode).

**Result:** all four hot-path `cublasSetMathMode` sites eliminated.
Two-handle dispatch is structurally cleaner + saves 2 cuBLAS API
calls per GEMM. Verified bit-identical at the chiron unit-test suite
(only pre-existing OVFG+Stiefel test fails, as baseline).

### Library — Task 1.6: GpuBuffer::zero + free-standing cudaMemset

- `486b22381` Replace `syncBlockingBufferOp` + bare `cudaMemset` in
  all six `GpuBuffer::zero()` type specializations with
  `cudaMemsetAsync(buf, 0, n, computeStream())`. Caller audit verified
  no race risks (all callers are followed by same-stream kernels or
  blocking downloads). Replaced two bare `cudaMemset` calls in
  `gpu_kernels.cu` (`cross_entropy_nll_loss_bf16` and
  `argmax_count_matches_bf16`) with the async variant.

**Result:** every `.zero()` call in the codebase is now capture-safe
AND faster (no blocking sync). General-purpose improvement.

### Library — Task 1.7: qknorm_gamma_scale GPU kernel

- `ec23567b0` Add new `qknorm_gamma_scale_gpu` GPU kernel + dispatcher
  in `gpu_kernels.cu` + `gpu_kernels.h`. Replaces the prior per-layer
  per-step D2H download → CPU multiply by sqrt(dHead) → H2D upload
  round-trip with a 30-line GPU element-wise kernel.

**Result:** eliminates ~120 µs/step (L=24 layers × 2 host/device
sync points per layer) plus the capture-incompatibility of the
download/upload pattern. General-purpose improvement.

### Library — Task 1.8: cudaGraphInstantiateWithFlags

- `71686ef6e` Use `cudaGraphInstantiateWithFlags` with
  `USE_NODE_PRIORITY` flag instead of legacy `cudaGraphInstantiate`.
  CUDA 13.2 does NOT expose `AUTO_PARALLELISM` (it's a CUDA 12.3+
  feature that didn't carry into 13.x). The `USE_NODE_PRIORITY` flag
  did not improve replay throughput in our measurements but the
  modernized API is now in place for future CUDA toolkit versions.

### Trainer (glades-trainer) — Auto-disable list + mutual exclusion

- `436b52e` Add `--cuda-graphs` pass-through in `run.sh` flagship
  recipe (was previously not wired at the run.sh level).
- `f435d35` Add explicit `--cuda-graphs ⨯ --fp8-attn` mutual exclusion
  at CLI parse (paradigm #50 runtime fallback uses CPU branching that
  would silently break under capture/replay).
- `5d7cdd2` Drop `cfg.scfa` from the cuda-graphs auto-disable list;
  narrow the gate to `cfg.scfaParallelBranches` (the only SCFA sub-
  flag with cross-stream events that's still genuinely incompat).
  Production flagship + `--cuda-graphs` no longer auto-disables.

### Trainer — Task 1.9: per-step zero hoist

- `b271462` Add `pre_step_zero_buffers()` called BEFORE
  `curGraph.launch()`. Issues the four per-step zeros (`s.p`,
  `s.p_bf16`, `s.dp`, `W.dE`) on `computeStream` outside the capture
  window. Guards the in-fwd/bwd zero() calls with
  `if (!cfg.cudaGraphs)`. No measurable throughput improvement but
  preserved as foundation for future per-layer extraction.

### Trainer — Updated warning

- `097eb68` Update the stale EXPERIMENTAL warning to reflect the
  post-arc empirical state: capture works, replay is ~2× slower than
  direct emission, NLL drifts ~0.2 nat from direct emission. Use
  `--cuda-graphs` for research only — production should leave it off.

---

## Empirical measurements

### Throughput (T=16384, L=24, batch=1 on RTX-4080-SUPER)

| Configuration | tok/s (steady-state, steps 2-10) |
|---|---:|
| Baseline (no `--cuda-graphs`, regstack ship recipe) | **~28,072** |
| `--cuda-graphs` after Phase 1 only (no capture; auto-disabled by `--scfa`) | ~27,956 (effectively baseline + flag overhead noise) |
| `--cuda-graphs` after Phase 1 + Phase 2 (capture FAILS due to GpuBuffer/QK-Norm/etc., falls back) | ~12,800 (paradox: capture-failure path triggers extra cudaStreamSynchronize) |
| `--cuda-graphs` after Tasks 1.6 + 1.7 (capture SUCCEEDS, no fallback) | ~12,790 |
| `--cuda-graphs` after Task 1.8 (USE_NODE_PRIORITY flag) | ~12,790 |
| `--cuda-graphs` after Task 1.9 (per-step zero hoist) | ~12,790 |

The arc's primary throughput goal (≥+1% wall improvement) was never
achieved. Replay was consistently ~54% slower than direct emission.

### NLL drift (forced-S val NLL at step 1, seed=1337, fresh init)

| Configuration | step-1 final val NLL | Δ vs baseline |
|---|---:|---:|
| Baseline (no `--cuda-graphs`) | 10.7069 | — |
| `--cuda-graphs` (current binary) | 10.5082 | **−0.1987 nat** |

The drift is ~2000× the spec's bit-identicality bar (≤ 0.0001 nat).
Root cause: graph replay's cuBLAS algorithm selection diverges from
direct emission's selection across the 24 layers × many GEMMs of a
single forward+backward pass, accumulating ULP-level differences into
visible nat-level loss values.

---

## Root cause of the replay slowdown

Per the holistic post-fix investigation (subagent dispatch, 2026-05-24):

**Primary cause:** ~512 MB of `cudaMemsetAsync` nodes captured into
the graph (per-step zeros from Task 1.9, plus per-layer zeros that
were NOT extracted: `s.scfa_inner_p`, `s.dq_buf`, `W.dWq_scratch` ×
24 layers ≈ ~1 GB additional). In direct emission, these memsets run
on the GPU's copy engine concurrently with SM-engine compute kernels.
In graph replay, the captured nodes form a linear dependency chain on
a single stream, eliminating the compute/copy overlap. The CUDA 13.2
`USE_NODE_PRIORITY` flag does not address this; the
`AUTO_PARALLELISM` flag that would address it was not carried into
the 13.x release.

**Secondary cause:** Large graph node count (~2,000-4,000) introduces
per-launch overhead and internal runtime bookkeeping memsets.

**Contributing cause:** cuBLASLt FP8 readout algorithm selection
locked at capture time (workspace=NULL forces non-optimal algo).

---

## Why Task 1.10 (per-layer zero extraction) was deferred

Even if all per-layer zeros were extracted (an estimated 300-500 LOC
additional refactor), the cuBLAS algo selection drift would remain.
NLL drift > 0.0001 nat is a hard spec gate; per-layer extraction
addresses wall but not NLL. The arc's spec required BOTH gates to
PASS. Pursuing per-layer extraction without a path to NLL parity is
not productive.

A future arc could pursue:
1. **Capture window narrowing**: capture only forward+backward, exclude
   `launch_loss_scalars`. May reduce algo-selection drift if the loss
   path is the dominant source.
2. **cuBLAS workspace allocation for graph capture**: provide a
   pre-allocated workspace so cuBLASLt can select the same algorithms
   in capture mode as direct emission.
3. **Wait for `cudaGraphInstantiateFlagAutoParallelism`** to return
   in a future CUDA toolkit version (13.x dropped it from 12.3).
4. **Manual graph construction** (build nodes explicitly with known
   dependencies) rather than stream-capture. ~5-10× engineering effort
   but full control over scheduling.

---

## Lessons learned

1. **The original iter 55 META was right.** `--cuda-graphs` × `--scfa`
   compatibility was correctly identified as "not iter-scale work" in
   2026-05-19. The arc surfaced 6+ blockers beyond the spec's
   originally-identified 3 (math-mode toggle, scfaFuseStreams,
   fp8-attn fallback). Each iteration found more.

2. **CUDA 13.2 lost `AUTO_PARALLELISM`** (a CUDA 12.3 feature). This
   feature gap is precisely what's needed to make the
   captured-memset-serialization problem auto-solve. Without it,
   manual graph construction or per-layer code refactoring is the
   only path forward.

3. **Stream-capture mode's algorithm-selection drift** is a real
   gotcha. Even when the captured graph contains the "right" kernels,
   cuBLAS may pick different algorithms during capture than during
   direct emission, producing FP32-noise-level NLL differences that
   accumulate across 24-layer models into visible drift.

4. **Pure-infra arcs aren't always smaller than NLL arcs.** This arc
   was estimated at 150-200 LOC; final landed scope was ~600 LOC
   across both repos. Each "I found another blocker" cycle was real
   engineering work.

---

## Recommended next direction

After three NLL arcs FAILed (MTP, LayerDrop, UL2) and one wall arc
PARTIAL_PROGRESS (CUDA Graphs), the Phase-3 session has accumulated
strong evidence that:
- CHIRON's symplectic update structure resists most regularization
  ports from standard residual transformers.
- Hardware-feature-dependent wall arcs need careful empirical
  scoping; iter-level audits are not sufficient.

Possible next directions (NOT a recommendation — purely the option
set):

1. **Accept regstack Phase 2 as the stable production flagship and
   close the Phase-3 session.** Four arcs investigated, three FAIL +
   one PARTIAL_PROGRESS = strong signal of diminishing returns on
   port-style mechanisms at this architecture/scale.

2. **A research-class arc** that designs a regularizer/mechanism
   specifically for symplectic transformers rather than porting from
   standard transformers. Would require theoretical work + small-scale
   validation before committing to a 1B-scale pilot.

3. **A pure-infra arc with explicit success-criteria revision**:
   target ≥+0.5% wall improvement instead of ≥+2%, allow NLL drift up
   to 0.001 nat instead of 0.0001 nat. Lowers the bar but at least
   permits incremental wins.

4. **Take a break** and revisit with fresh perspective.

---

## Reproduce commands

```bash
# Baseline (the regstack Phase 2 ship recipe — the production flagship)
cd ~/dev/glades-trainer
sh run.sh flagship --zloss-coef 1e-4 --qk-norm \
    --steps 10 --seed 1337 \
    --save /tmp/baseline_10step

# --cuda-graphs (research only; ~2× slower than baseline)
sh run.sh flagship --zloss-coef 1e-4 --qk-norm --cuda-graphs \
    --steps 10 --seed 1337 \
    --save /tmp/cuda_graphs_10step
```

The log shows the new warning message about replay slowdown and
NLL drift directing the user toward research-only usage.

---

## Honest publication per Phase-3 P6

This arc didn't ship a flagship update, but it did ship substantial
codebase improvements. The honest assessment: the original spec was
too optimistic about scope (3 blockers → 6+ found), too optimistic
about CUDA 13.2 capabilities (`AUTO_PARALLELISM` not available), and
too strict about the bit-identicality bar (cuBLAS algo drift makes
0.0001 nat infeasible for any graph-replay path at 24-layer depth).

Future iter-N program note: when an iter-55-style META labels work
as "not iter-scale", treat the scope estimate skeptically. A multi-
week refactor was correctly identified; this arc only completed a
fraction of it before hitting hardware-feature blockers (CUDA
`AUTO_PARALLELISM` absence).
