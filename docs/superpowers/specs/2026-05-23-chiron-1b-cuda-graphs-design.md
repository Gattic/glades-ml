# CHIRON 1B CUDA Graphs Production-Readiness — Design

**Date:** 2026-05-23
**Status:** Pre-registered design (no code yet)
**Program scope:** Phase-3 of CHIRON 1B production line. Pure wall-time
arc following two NLL-arc FAILs (LayerDrop, UL2). Targets the
optimizer/wall-time/attention class where CHIRON has previously won.
**Baseline:** `chiron_1B_T16384_regstack_phase2.final` (regstack
Phase 2 ship 2026-05-22). Final val NLL 3.5734 @ step 30000,
throughput 28,072 tok/s, VRAM 14.97 / 15.56 GB on RTX-4080-SUPER.
**Output target:** A 30k Phase-2 production retrain with
`--cuda-graphs` enabled, bit-identical NLL ≤ 3.5734, throughput
≥ +2% over the ship.

---

## 1. Motivation

After three regularization arcs closed FAIL/NEGATIVE in a row (MTP
NEGATIVE earlier, LayerDrop FAIL on 2026-05-23, UL2 FAIL on
2026-05-23), the lesson is that CHIRON's symplectic
`(p, q) → (p + shear(q), reln(q))` update structure resists
objective-side and architecture-side regularization mechanisms ported
from standard residual transformers. The mechanism classes where
CHIRON has historically WON are:

- **Optimizer-side:** FACE Adafactor (paradigm #28), int8-Adam, Sophia
  (#55), MUON-lite (in ATLAS).
- **Wall-time:** iter 60-117 ralph-loop accumulated 1.85× cumulative
  throughput; ten iter-N mechanisms shipped at the iter 116 ship.
- **Attention-side:** SCFA (paradigm #42), QK-Norm (regstack Phase 2),
  SFA (paradigm #250 research line).

This arc continues that pattern by pursuing the
**iter 55 META "not iter-scale work" candidate**: making
`--cuda-graphs` (paradigm #51 ATLAS-COMPILE) production-ready by
removing the per-step CPU branching that auto-disables the flag for
the flagship recipe.

Per `research/ITER55_PARADIGM_FLAGS_NULL.md`:

> `--cuda-graphs` × `--scfa` could in principle be made compatible by
> removing the per-step CPU branching in SCFA. That's a larger
> refactor — multiple flags involved (scfa-fuse-streams's event
> records, fp8-attn's runtime fallback, etc). **Not iter-scale work.**

This arc IS that work.

**Why this fits CHIRON 1B specifically:**

1. **Math is unchanged.** CUDA graphs only optimize kernel scheduling —
   the actual fwd/bwd computation is byte-for-byte identical. This
   bypasses the symplectic-incompatibility class of failures that
   killed the prior three arcs.
2. **Wall-time win estimate is grounded.** Per-step kernel-launch
   overhead at T=16384 on Ada is ~30-50 ms (~5000-7000 kernels × 5-10
   µs/launch) on a ~585 ms step → ~5-10% wall reduction headroom.
3. **Existing infrastructure is mostly there.** Graph capture,
   instantiation, replay, invalidation, and fallback machinery is
   already in `chiron_main.cpp` (commit `d2351cc` wired the flag;
   lines 12894-13007 implement the capture/replay loop). The blocker
   is the auto-disable check.
4. **Clear pre-identified blockers.** `research/CUDA_GRAPHS_FIX4_NOTES.md`
   (referenced in `chiron_main.cpp:12662`) documents the
   cublasGet/SetMathMode issue. `research/ITER55_PARADIGM_FLAGS_NULL.md`
   documents the SCFA per-step branching issue. No discovery work
   needed; just refactor.

---

## 2. Architecture

### 2.1 Three concrete blockers + fixes

**Blocker A: `cublasSetMathMode` / `cublasGetMathMode` inside the hot
path** (in `gpu_blas.cu`'s `sgemm_rowmajor` / FAST_16BF wrappers).

The functions toggle TF32 vs strict-FP32 mode per GEMM call, which is
forbidden during graph capture (cuBLAS state changes are
non-captureable). Comment at `chiron_main.cpp:12656-12662`:

> EXPERIMENTAL: forward+backward still call cublasGetMathMode /
> cublasSetMathMode (sgemm_rowmajor's TF32-vs-strict guard) which is
> not capture-compatible. Capture will fail at step 0 and gracefully
> fall back to direct kernel emission for the rest of the run. Full
> graph capture needs gpu_blas.cu refactored so the math mode is set
> once at init and never changed inside the hot path.

**Fix A:** maintain TWO cuBLAS handles (one TF32, one strict-FP32) at
init time. Each GEMM dispatches to the appropriate handle based on
its precision requirement. The math mode is never changed inside the
hot path. ~50-100 LOC change in `gpu_blas.cu`.

**Blocker B: `scfaFuseStreams` per-step CPU branching.**

SCFA's stream-fusion uses CPU-side event records to coordinate
parallel streams (chiron_main.cpp lines ~3351, ~6942, ~7039 visible
under `if (cfg.scfaFuseStreams)`). Event records on streams outside
the main capture stream are not capture-compatible.

**Fix B (two options, decide at implementation):**

- **B1 — Single-stream collapse:** Remove the `scfaFuseStreams`
  parallel-stream design; run all SCFA ops on the main compute stream.
  Loses the original `scfaFuseStreams` wall benefit (~1-2% from
  parallel execution on small ops) but unlocks graphs for the entire
  fwd/bwd. Net: ~+3-5% wall if graphs save ~5-7%.
- **B2 — Graph-captured cross-stream dependencies:** CUDA Graphs
  natively support cross-stream sync via graph-internal dependencies
  (`cudaGraphAddEventRecordNode` + `cudaGraphAddEventWaitNode`).
  Preserves the parallel-stream design while making it
  capture-compatible. More engineering but no wall loss. Net: ~+5-7%
  wall.

**Recommendation:** start with B1 (cleaner + smaller refactor). If G1
pilot shows wall improvement is below the +1% gate due to lost
scfaFuseStreams parallelism, escalate to B2.

Estimate: ~200-300 LOC (B1); ~400-600 LOC (B2).

**Blocker C: `fp8Attn` runtime fallback.**

`chiron_main.cpp:9085-9110` has a `static bool s_fp8_runtime_disabled`
that's a per-step CPU branch. Not used by the flagship (`--fp8-attn`
is off in production per `research/FP8_ATTN_HELIUM_CUDA13_NULL.md`).

**Fix C:** add explicit mutual exclusion at CLI parse time:
```cpp
if (cfg.cudaGraphs && cfg.fp8Attn)
{
    log_error("chiron", "--cuda-graphs is mutually exclusive with "
              "--fp8-attn (paradigm #50 runtime fallback uses CPU "
              "branching; not capture-compatible).  Disable one.\n");
    return 1;
}
```

~10 LOC. No refactor needed.

### 2.2 Audit checklist (no code change expected)

- **`--qk-norm`** (in flagship): per-layer per-step kernel sequence is
  deterministic (γ_h scale_q + qknorm_forward + flash_attention).
  Expected graph-compatible.
- **`--zloss-coef 1e-4`** (in flagship): readout backward path adds
  one extra kernel (zloss term); deterministic per step. Expected
  graph-compatible.
- **`--scfa-bf16-inner` / `--scfa-bf16-outer`** (in flagship):
  BF16-TC path is unconditional per-step. Expected graph-compatible.
- **`--int8-adam`** (in flagship): optimizer state is int8; the Adam
  step is a single kernel per param. Deterministic per step. Expected
  graph-compatible.
- **`--bf16-logits-storage`** (in flagship): readout uses BF16 storage
  + FP32 compute. Deterministic per step. Expected graph-compatible.
- **`--fuse-attn-reln`** (in flagship): last-layer fused attn-reln
  kernel. Deterministic. Expected graph-compatible.

Verification: G1 500-step run; if capture succeeds at step 1 without
"beginCapture failed" or "launch failed" log messages, all flagship
mechanisms are graph-compatible. If capture fails, the error message
identifies the offending kernel for iterative debugging.

### 2.3 Drop SCFA + scfaFuseStreams from auto-disable list

`chiron_main.cpp:12644` currently:
```cpp
if (cfg.scfa)             { incompat = true; incompatReason = "--scfa"; }
```

After Blocker A + B fixes, this becomes a stale guard. Remove it.

Add explicit (terminal) mutual exclusion for genuinely-incompatible
flags that have been ruled out:
```cpp
if (cfg.cudaGraphs && cfg.fp8Attn)
{
    log_error("chiron", "--cuda-graphs is mutually exclusive with --fp8-attn\n");
    return 1;
}
if (cfg.cudaGraphs && cfg.medalTrain)
{
    log_error("chiron", "--cuda-graphs is mutually exclusive with --medal-train (varying alpha per step)\n");
    return 1;
}
if (cfg.cudaGraphs && cfg.ul2Enabled)
{
    log_error("chiron", "--cuda-graphs is mutually exclusive with --ul2-enabled (per-step denoiser branching)\n");
    return 1;
}
```

The existing auto-disable warnings (line 12649-12650 for sasAlpha,
orion, tSchedule) stay; those are dynamic-config items that the user
sets without realizing the conflict — auto-disable is the right UX.

### 2.4 Implementation sites

| File | Change |
|---|---|
| `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_blas.cu` | Blocker A: two-handle dispatch in `sgemm_rowmajor`, FAST_16BF wrappers. Move all `cublasSetMathMode` calls out of hot path to init. |
| `glades-trainer/trainer/chiron_main.cpp` SCFA dispatch | Blocker B: collapse `scfaFuseStreams` to single-stream OR convert event records to graph-captured cross-stream dependencies. Multiple sites (lines ~3351, ~6942, ~7039 and likely more — full audit required). |
| `glades-trainer/trainer/chiron_main.cpp` `--cuda-graphs` mutual exclusion | Blocker C: add explicit checks at CLI parse / config validation time. |
| `glades-trainer/trainer/chiron_main.cpp` line 12644 | Drop `cfg.scfa` from auto-disable list after Blockers A+B land. |
| `glades-ml/unit-tests/...` | Optional: add `CHIRONCudaGraphsCaptureSmokeTest` if a library-level smoke is feasible. System-level smoke runs at G1. |

### 2.5 Why graph capture is safe under the regstack Phase 2 stack

The full flagship kernel sequence per training step is **deterministic
modulo input data**. Specifically:
- The kernel launch sequence depends only on cfg.*, NOT on the
  data values streaming through.
- Per-layer kernel sequences are uniform across layers (or vary
  predictably with `li`).
- No per-step CPU conditionals after Blockers A-C are fixed.

The capture-once-replay-many pattern is therefore mathematically
safe. Replay produces bit-identical loss/gradient values at every
step (modulo FP32 FMA reordering that COULD differ between captured
and direct-emit paths — but typically these are bit-identical).

### 2.6 Bit-identicality at NLL level

The math doesn't change. Bit-identicality at FP32 is the expected
property. Acceptable measurement-floor drift: ≤ 0.0001 nat from
FMA-ordering differences across replay vs direct-emit.

If observed drift > 0.0001 nat, that's a real bug — investigate.

---

## 3. Validation arc

Pure-infra arc; much shorter than the NLL arcs (LayerDrop, UL2).

### 3.1 Run plan

| ID | Steps | Seed | Config | Compute |
|---|---:|---:|---|---:|
| G0 | 500 | 1337 | regstack Phase 2 ship (no graphs) | ~5 min |
| G1 | 500 | 1337 | regstack ship + `--cuda-graphs` | ~5 min (if graphs PASS) |
| **G2** | **30000** | **1337** | **regstack ship + `--cuda-graphs`** | **~4.5-5 h** |

500 steps is sufficient to: capture a graph at step 1, exercise the
capture/replay loop, run 4-5 val checkpoints, measure throughput
stable-state.

### 3.2 Gate criteria

**G1 pilot gate (vs G0):**

- **Capture success:** at step 1, log shows "graph captured
  successfully" (or equivalent — see existing
  `chiron_main.cpp:12900-13007` for log strings). No "beginCapture
  failed" / "launch failed" lines. If capture fails, parse the error
  message and iteratively fix until success.
- **NLL bit-identicality:** at every val checkpoint, `|G1_NLL -
  G0_NLL| ≤ 0.0001 nat`. The strict bit-identicality (or sub-ULP) is
  required. Drift > 0.0001 nat is a real bug.
- **Throughput:** `G1_TOKS ≥ G0_TOKS · 1.01` (≥+1% wall improvement).
  Goal: ~+5-7% (the kernel-launch overhead headroom). If <+1%, the
  arc's main thesis is wrong and the fix landing without measurable
  wall benefit is itself useful information — but the gate is +1%.
- **VRAM:** ≤ G0 VRAM (no new buffers introduced).

**G2 30k Phase-2 gate (only if G1 PASSes):**

- **NLL @ step 30000:** `|G2_NLL - 3.5734| ≤ 0.001 nat` (bit-identical
  to ship at FP32; sub-ULP tolerance for cumulative FMA-ordering
  effects over 30k steps).
- **Throughput:** `G2_TOKS ≥ 28,072 · 1.02 = 28,633 tok/s`. Realistic
  target ~29,500-30,000 tok/s.
- **VRAM:** ≤ 15.72 GB peak (1% above the 15.56 GB ceiling).
- **Graph capture stable:** no late-step capture invalidation events
  except expected RLG L-changes.

### 3.3 PASS handling

If G2 PASSes:
- Archive: `database/checkpoints/chiron_1B_T16384_regstack_phase2/`
  → keep loadable (the prior ship — both ships are bit-identical at
  the same checkpoint values; the new ship just has CUDA graphs in
  the recipe).
- New ship checkpoint:
  `chiron_1B_T16384_regstack_graphs_phase2.final`.
- Update `CLAUDE.md` "Current Production Flagship" block with new
  throughput number (~29,500-30,000 tok/s) and reproduce command:
  `sh run.sh flagship --zloss-coef 1e-4 --qk-norm --cuda-graphs`.
- Write `research/CUDA_GRAPHS_PASS_2026_MM_DD.md` with wall
  measurement + capture log summary + per-Blocker fix attribution.
- Memory entry: cuda-graphs ship.

### 3.4 FAIL handling

- **G1 capture fails after all known blockers fixed:** iterative
  triage. Each new blocker is fixed and G1 re-run. Document each
  blocker in the result doc.
- **G1 NLL drift > 0.0001 nat:** real bug. Likely a math-mode-handle
  mismatch (Fix A regression) or a missed kernel that depends on
  cuBLAS state. Debug + re-run.
- **G1 wall improvement < +1%:** the kernel-launch overhead at this
  scale is smaller than expected. Pure-infra arc still landed the
  fixes (math-mode refactor, SCFA single-stream, mutual exclusion);
  document as "infrastructure fixed but wall ceiling found". Library
  + trainer code stays as opt-in (`--cuda-graphs` available for
  future scales / hardware where launch overhead is larger).
- **G2 NLL drift > 0.001 nat at step 30000:** cumulative FP32 noise
  too large. Either replay introduces non-bit-identical math (real
  bug — debug) OR the noise envelope is wider than expected at FP32
  over 30k steps (acceptable, document and adjust ship-gate
  threshold).

### 3.5 Production retrain handoff (PASS)

The CHIRON 1B production flagship checkpoint values at the same step
are bit-identical between `--cuda-graphs` and `--cuda-graphs=off`
runs (modulo sub-ULP FMA-ordering). So:

- The prior ship's checkpoint (`chiron_1B_T16384_regstack_phase2.final`)
  is ALREADY the same model as G2's checkpoint at any step. Strictly,
  the G2 checkpoint at step 30000 is a fresh retrain that should
  bit-match the prior ship's checkpoint at the FP32 level (or sub-
  ULP). They can be used interchangeably for inference.
- The "ship swap" is therefore a recipe-only change: update
  `runner.sh --flagship` and CLAUDE.md to point to the G2 checkpoint
  and the new `--cuda-graphs`-included reproduce command.
- The prior ship is "demoted" to "kept for context" subsection just
  for historical record — both checkpoints are equivalent at the FP32
  level.

---

## 4. Risks (pre-registered)

| Risk | Probability | Mitigation |
|---|---|---|
| **R-Graphs-1:** Blocker A two-handle refactor introduces a math-mode mismatch at some GEMM site, producing NLL drift > 0.0001 nat at G1. | medium | Audit all `cublasSetMathMode` call sites; verify each GEMM dispatches to the correct handle. Run unit test suite + chiron suite + a non-flagship trainer smoke test before claiming PASS. |
| **R-Graphs-2:** Blocker B Fix path B1 (single-stream) loses 1-2% wall from the lost `scfaFuseStreams` parallelism, but graphs save only 3-4% net. Result wall improvement is marginal (+1-2%). | medium | If G1 wall improvement is below +3%, escalate to Fix path B2 (graph-captured cross-stream deps). +400-600 LOC additional engineering but preserves the parallel-stream wall benefit. |
| **R-Graphs-3:** Audit reveals more per-step CPU branches in the chiron_main forward/backward that aren't documented in the iter 55 NULL doc. | medium | G1 capture-failure logs will identify the offending kernel. Iteratively fix and re-run. May require several G1-attempt cycles. Each iteration is ~5 min wall. |
| **R-Graphs-4:** Graph capture at step 1 succeeds but is invalidated by RLG L-change mid-run, causing recapture overhead that eats wall savings. | low | Existing invalidation+recapture machinery handles this (line 12894+). RLG transitions are infrequent in flagship recipe (no `--rlg-initial-layers` in default). Verify in G1 + G2. |
| **R-Graphs-5:** Wall savings smaller than expected (<+1%) on Ada-class. Step time at T=16384 is dominated by GEMM compute (~80% of step), not launch overhead. | low-medium | If G1 shows <+1% wall, the arc result is "infrastructure fixed, no measurable wall benefit at this scale". Still useful: --cuda-graphs becomes a future-proofing feature for larger T / different hardware. Document as a known-ceiling finding. |
| **R-Graphs-6:** Cumulative FP32 FMA-reordering over 30k steps drifts G2's val NLL by > 0.001 nat from the ship's 3.5734. | low | The math IS deterministic per-step (FMA ordering within a kernel is fixed by the kernel binary). Across-kernel ordering changes are within a tight envelope. If observed drift > 0.001, debug; if endemic to FP32 over 30k, adjust gate threshold and document. |
| **R-Graphs-7:** The Blocker A two-handle refactor breaks a non-trainer code path (unit tests using cublas, ATLAS optimizer, other paradigms). | medium | Run the full unit-test suite at every commit during the refactor. Test the byte-level and BPE pile-trainer paths too (they use the same gpu_blas.cu). |
| **R-Graphs-8:** `scfaFuseStreams` single-stream collapse (B1) regresses non-graphs performance on prior ship checkpoint reload runs. | low | The change to scfaFuseStreams is unconditional (always single-stream after B1). Prior-ship reproducibility at scfaFuseStreams=on flag is broken in the strict sense, but the flag effectively becomes a no-op. Document in CLAUDE.md. |
| **R-Graphs-9:** Concurrent runs of LayerDrop + cuda-graphs OR UL2 + cuda-graphs (which would be allowed by some users) silently produce wrong gradients via graph replay of stale mask buffers. | already mitigated | Both LayerDrop and UL2 already added themselves to the `--cuda-graphs` mutual-exclusion list (LayerDrop in commit ef9d540, UL2 in commit ac68907). No new work needed. |
| **R-Graphs-10:** The cuBLAS handle refactor introduces a NaN at some GEMM site due to a mode-mismatch edge case (e.g., TF32 handle accidentally used for an int8-Adam state update). | low | Unit tests + smoke runs will catch a NaN immediately. The chiron suite has stable training-loss values that any GEMM regression will perturb. |

---

## 5. Out of scope

- **CUDA Graphs at higher batch sizes:** CHIRON 1B is batch=1; not
  relevant.
- **MTP / IGAA / SFA / distillForward / sasAlpha graph compatibility:**
  those paradigms have legitimate per-step CPU branching by design.
  Auto-disable list stays for them.
- **Half-graph capture (forward-only):** all-or-nothing capture; the
  partial mode adds complexity for half the benefit.
- **Async / overlapping graph capture:** capture happens
  synchronously at step 1; not optimizing capture itself.
- **CUDA Graphs API beyond `cudaGraphInstantiate` + `cudaGraphLaunch`:**
  no `cudaGraphExecUpdate` or graph-level performance tuning.
- **CUDA toolkit upgrade beyond CUDA 13.2:** current toolkit version.
- **Cross-GPU (multi-GPU) graphs:** single-GPU production.
- **Multi-seed validation of G2:** single-seed Phase-1/Phase-2 per the
  regstack / LayerDrop / UL2 precedent. Multi-seed deferred to a
  separate gate after G2 PASSes.

---

## 6. Deliverables checklist

- [ ] Pre-register this spec in memory.
- [ ] Implement Blocker A: `gpu_blas.cu` two-handle refactor.
- [ ] Implement Blocker B (B1 first): `scfaFuseStreams` single-stream
      collapse.
- [ ] Implement Blocker C: `--cuda-graphs --fp8-attn` mutual exclusion
      at CLI parse.
- [ ] Drop `cfg.scfa` and `cfg.scfaFuseStreams` from
      `chiron_main.cpp:12644` auto-disable list.
- [ ] Add `--medal-train` and `--ul2-enabled` mutual exclusion with
      `--cuda-graphs` (the latter is already there from UL2 arc; verify).
- [ ] Build + run chiron unit-test suite; verify no regressions.
- [ ] Run G0 500-step baseline (regstack ship recipe, no graphs).
- [ ] Run G1 500-step with `--cuda-graphs`; verify capture success +
      NLL bit-identicality + throughput gain.
- [ ] If G1 wall < +3% with B1, escalate to B2 (cross-stream graph
      dependencies); re-run G1.
- [ ] If G1 PASSes, run G2 30k Phase-2 retrain.
- [ ] Verify G2 NLL bit-identical to ship's 3.5734 (sub-ULP drift OK).
- [ ] Verify G2 throughput ≥ +2% over ship's 28,072 tok/s.
- [ ] Write `research/CUDA_GRAPHS_<RESULT>_2026_MM_DD.md`.
- [ ] If PASS: update CLAUDE.md, runner.sh path, memory entry.
- [ ] If FAIL: honest negative-result publication per Phase-3 P6.

---

## 7. Repro commands

**500-step pilot (Phase 1):**

```bash
cd ~/dev/glades-trainer

# G0: baseline (current regstack Phase 2 ship, NO cuda-graphs)
sh run.sh flagship --zloss-coef 1e-4 --qk-norm \
    --steps 500 --seed 1337 \
    --save database/checkpoints/chiron_1B_T16384_g0_500step

# G1: + cuda-graphs
sh run.sh flagship --zloss-coef 1e-4 --qk-norm --cuda-graphs \
    --steps 500 --seed 1337 \
    --save database/checkpoints/chiron_1B_T16384_g1_graphs_500step
```

**30k Phase 2 (G2; gated on G1 PASS):**

```bash
sh run.sh flagship --zloss-coef 1e-4 --qk-norm --cuda-graphs \
    --steps 30000 --seed 1337 \
    --save database/checkpoints/chiron_1B_T16384_regstack_graphs_phase2
```

If the existing `--cuda-graphs` machinery is sufficient (no need for
new flags), this is the full reproduce surface.

---

This pre-registration is committed before any implementation code
lands. Per Phase-3 P6 (no silent claim-dropping), if any gate above
fails, the result is published honestly and the program adapts — not
silently re-targeted. The LayerDrop and UL2 arcs' FAILs are
published examples of this discipline; CUDA Graphs follows the same
protocol.
