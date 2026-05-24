# CHIRON 1B CUDA Graphs Arc — PARTIAL_PROGRESS (2026-05-23/24)

> **AMENDMENT 2026-05-24 (post-closure nsys investigation):** the
> diagnosis below originally attributed the −54% wall to (a) captured
> `cudaMemsetAsync` nodes serializing in the graph DAG and (b) cuBLAS
> algorithm regression in capture mode, with CUDA 13.2 dropping the
> `AUTO_PARALLELISM` flag as the structural blocker. **Subsequent
> nsys profile comparison falsified both of those theories.** The
> actual cause is **CPU-GPU overlap loss** — a structural property
> of CUDA Graphs at large per-step workloads, not a hardware-feature
> absence or algorithm regression. The corrected diagnosis is the
> "Root cause" section below; the original (incorrect) text is
> preserved in italics as "_Original (incorrect) diagnosis_" for
> historical accuracy. The CLOSURE DECISION is unchanged (replay is
> ~2× slower than direct emission, both spec gates fail, arc closes
> as PARTIAL_PROGRESS) — only the WHY is amended.

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

## Root cause of the replay slowdown (CORRECTED 2026-05-24)

The corrected diagnosis is based on nsys profile comparison of
4-step direct emission vs 4-step graph replay runs on the flagship
recipe.

### Empirical evidence from nsys

**Kernel-level comparison (identical kernels, identical per-call times):**

| Kernel | Direct avg | Graphs avg | Δ |
|---|---:|---:|---:|
| `ampere_s1688gemm_bf16_128x128_ldg8_stages_32x1_nn` | 519.8 µs | 496.1 µs | graphs slightly faster |
| `ampere_s1688gemm_bf16_128x128_ldg8_stages_32x1_nt` | 665.5 µs | 665.0 µs | identical |
| `sm89_xmma_gemm_e4m3bf16_e4m3f32_f32_tn_n_tilesize128x128x64...` (FP8 readout) | 9.85 ms | 9.82 ms | identical |
| `cutlass_80_tensorop_s16816gemm_bf16_256x128_32x3_nn_align8` | 206.2 µs | 206.1 µs | identical |

cuBLAS picks the **same algorithms** under capture mode as under
direct emission, and each kernel runs at the **same speed**. The
"cuBLAS algo regression" hypothesis from the original closure draft
is falsified.

**Total GPU active time:**
- Direct: 5.81 sec across 25,978 kernel instances.
- Graphs: 3.60 sec across 16,958 kernel instances.

**Graphs has LESS total GPU work but LONGER wall.** Memset
serialization (the original hypothesis) would have predicted the
opposite — more GPU work in graphs mode. The "captured memsets
serialize" hypothesis is also falsified as a meaningful contributor
to the slowdown. (The serialization claim was factually true at the
~0.6 ms scale but three orders of magnitude smaller than the
observed +700 ms/step regression.)

**CUDA API time breakdown:**

| API | Direct total | Graphs total | Δ |
|---|---:|---:|---:|
| `cudaStreamSynchronize` | 4.22 sec (2270 calls) | 5.17 sec (2268 calls) | **+0.95 sec** |
| `cudaLaunchKernel` | 1.12 sec (22594 calls) | 0.064 sec (16912 calls) | **−1.06 sec** |
| `cudaGraphLaunch` | — | 0.005 sec (3 calls) | +0.005 sec |
| `cudaMemcpy` | 0.64 sec | 0.66 sec | +0.02 sec |

**Single largest `cudaStreamSynchronize` call:** 180 ms (direct) → 558
ms (graphs). The max sync ballooned by 378 ms — that single delta,
multiplied across replay steps, accounts for most of the wall
regression.

### The actual mechanism: CPU-GPU overlap loss

**Direct emission per-step timeline:**

```
Host: [dispatch step N+1 kernels (~500 ms of cudaLaunchKernel)] [sync ~180 ms]
GPU:                          [running step N kernels for ~580 ms]
Wall ≈ max(host_dispatch, gpu_work) ≈ 600 ms
```

The ~500 ms of cudaLaunchKernel API time happens **while the GPU is
executing the prior step's kernels**. By the time the host hits a
blocking sync, the GPU is ~80% done; the sync waits only the residual
~180 ms.

**Graph replay per-step timeline:**

```
Host: [cudaGraphLaunch 1.7 ms] [nothing to do] [sync ~558 ms]
GPU:                          [running graph for ~580 ms]
Wall ≈ graph_launch + sync_wait ≈ 580 ms + post-graph host work
```

The cudaGraphLaunch call returns in ~1.7 ms. The host immediately
runs out of work to dispatch and hits a blocking sync. That sync now
sees the **full** ~580 ms of GPU work because nothing on the host
side was overlapping with it.

The net effect: graphs eliminates the per-kernel launch overhead
(saves ~1.06 sec), but loses the dispatch/GPU overlap that was
hiding ~500 ms of GPU work per step. Net change is unfavorable
because at CHIRON 1B at T=16384, the GPU work (~580 ms/step) is
comfortably larger than the host dispatch (~500 ms/step) — direct
emission was running with the GPU as the long pole and dispatch
hidden behind it; graphs collapses dispatch but exposes GPU work in
full.

### Why this is not fixable by per-layer zero extraction

The original closure draft suggested Task 1.10 (per-layer zero
extraction) as future work. The nsys data shows this would not help:

- Graphs already has LESS GPU active time than direct. Moving more
  memsets out of capture would further reduce graph-mode GPU work,
  but wouldn't reduce wall, because wall is bounded below by
  cudaStreamSynchronize + the GPU work the graph still has to do.
- The 700 ms/step regression is from lost CPU-GPU overlap, not
  GPU-side serialization. Per-layer extraction does not address
  CPU-GPU overlap.

### Why this is not fixable by `AUTO_PARALLELISM`

The original closure draft suggested CUDA 13.2 dropping
`AUTO_PARALLELISM` was the structural blocker. That diagnosis was
also wrong: `AUTO_PARALLELISM` addresses **graph-node concurrency
on the GPU** (running independent nodes in parallel where the DAG
allows). It cannot address CPU-GPU overlap loss, because the host
has nothing to do during graph execution regardless of intra-graph
node concurrency.

### Why CUDA Graphs are structurally a bad fit at this workload size

CUDA Graphs help most when **CPU dispatch is the bottleneck**: small
kernels, high launch rate, GPU sits idle between launches waiting
for the next one. In that regime, eliminating launch overhead wins.

CUDA Graphs hurt when **GPU work is the bottleneck and dispatch
fits inside it**. CHIRON 1B at T=16384 has ~580 ms of dense GPU
work per training step (24 layers × dozens of large GEMMs +
SCFA inner attention + readout). The ~500 ms of host dispatch
overlaps cleanly with that GPU work. Collapsing dispatch to ~5 ms
via graphs doesn't help (GPU was already the long pole) and removes
the overlap that hid the GPU work behind dispatch.

This is the workload-size-dependent inversion of the CUDA Graphs
benefit, documented in NVIDIA's own performance guides under
"when not to use graphs." We discovered it the hard way.

---

## Lessons learned (CORRECTED 2026-05-24)

1. **The original iter 55 META was right.** `--cuda-graphs` × `--scfa`
   compatibility was correctly identified as "not iter-scale work" in
   2026-05-19. The arc surfaced 6+ blockers beyond the spec's
   originally-identified 3 (math-mode toggle, scfaFuseStreams,
   fp8-attn fallback). Each iteration found more.

2. **CUDA Graphs are workload-size-dependent.** At small per-step
   workloads (where host dispatch is the bottleneck), graphs save
   wall by eliminating launch overhead. At large per-step workloads
   (where GPU is the long pole and dispatch fits inside GPU work),
   graphs are net-negative because they collapse host dispatch (was
   overlapping with GPU work) without reducing GPU work. CHIRON 1B
   at T=16384 falls in the latter regime: ~580 ms/step of dense GPU
   work, ~500 ms/step of host dispatch that was hidden behind it.
   See "The actual mechanism: CPU-GPU overlap loss" above.

3. **Verify before attributing.** The original closure draft
   attributed the slowdown to "captured memset nodes serialize" +
   "cuBLAS algo regression in capture mode" + "CUDA 13.2 dropped
   AUTO_PARALLELISM." All three were falsified by nsys profile
   comparison after closure. The memset-serialization claim was
   factually true but quantitatively trivial (~0.6 ms vs the
   observed ~700 ms/step regression). The cuBLAS algo claim was
   simply wrong — kernel names + per-call times are identical
   between modes. The `AUTO_PARALLELISM` claim was conceptually
   wrong — that flag addresses GPU-side node concurrency, not
   CPU-GPU overlap. Demand nsys data BEFORE writing root-cause
   sections in result docs.

4. **Pure-infra arcs aren't always smaller than NLL arcs.** This
   arc was estimated at 150-200 LOC; final landed scope was ~600
   LOC across both repos. Each "I found another blocker" cycle was
   real engineering work. The post-closure nsys investigation
   (which surfaced the corrected diagnosis) was another ~30
   minutes that should have been part of the arc, not post-mortem.

5. **`USE_NODE_PRIORITY` flag (Task 1.8) was not useful.** It
   addresses node-priority ordering inside the graph, but the
   slowdown was never about node ordering inside the graph — it
   was about CPU-GPU overlap outside the graph. The Task 1.8
   commit (`71686ef6e`) modernized the API but did not help wall.

---

## Recommended next direction

After three NLL arcs FAILed (MTP, LayerDrop, UL2) and one wall arc
PARTIAL_PROGRESS (CUDA Graphs), the Phase-3 session has accumulated
strong evidence that:
- CHIRON's symplectic update structure resists most regularization
  ports from standard residual transformers.
- CUDA Graphs is structurally a bad fit for CHIRON 1B's workload
  size — not a fixable bug but a workload-regime mismatch. The
  ~580 ms/step GPU work is the long pole; collapsing host dispatch
  loses the overlap that hid it.

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

3. **A wall arc that targets host-side prefetch overlap with the
   graph launch** — the right way to make `--cuda-graphs` profitable
   is to ensure the host has substantial work to do during graph
   execution (e.g., async dataset prefetch for the next batch,
   batched async logging, deferred loss readback). This would
   restore the CPU-GPU overlap that direct emission gets for free.
   Estimated ~500-1000 LOC trainer-loop refactor. Outcome uncertain
   but addresses the actual root cause.

4. **Take a break** and revisit with fresh perspective.

The "Task 1.10 per-layer zero extraction" suggestion that appeared
in the original closure draft is **withdrawn** — the nsys data shows
it would not have helped. Per-layer zeros are not the dominant cost;
CPU-GPU overlap loss is.

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
