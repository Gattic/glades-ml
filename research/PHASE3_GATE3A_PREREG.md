# Phase-3 Gate-3A — Pre-registration

**Date:** 2026-05-19 (revised — no external baselines, no external libs).
**Status:** Pre-registered (no code yet).
**Gate:** Baseline characterization + bottleneck identification for the
CHIRON 1B flagship.
**Program scope:** strictly improve CHIRON 1B via novel research, all
internal (no external library installs, no external baseline
reproductions, no external eval datasets beyond `pretok-data/`). The
program's rules:
  - Audit-first: every new mechanism is preceded by an internal audit
    of existing related shifts (the 4 shipped + 15+ designed-but-unshipped
    paradigm shifts in `research/PARADIGM_SHIFT_*.md`).
  - Pre-commit baselines and thresholds before any code lands.
  - Every gate must include B0 = CHIRON 1B (this file), B1 = proposal,
    B2 = proposal-alone, B5 = simplest-in-class subsystem control.
  - Single-flag ablation, multi-seed (≥3), honest negative-result
    publication on failure.

---

## 1. What Gate-3A is

Three sub-deliverables that together characterize the CHIRON 1B
production flagship sufficiently for any Gate-3B/3C/3D candidate to be
proposed with a specific bottleneck target.

  **3A.1** Reproduce the CHIRON 1B reference trajectory from a clean
  fresh-init run. Confirms the codebase + recipe + hardware can
  recreate the published baseline.

  **3A.2** Per-kernel + per-buffer profile of a single training step
  at the production config. Identifies WHERE the time and memory are
  spent. Output: `research/PHASE3_GATE3A_PROFILE.md`.

  **3A.3** Per-component VRAM accounting summing to ≥95% of measured
  peak. Identifies which buffer classes are budget-binding. Output:
  `research/PHASE3_GATE3A_VRAM.md`.

These three are the empirical foundation on which any Gate-3B
mechanism shift will be proposed.

---

## 2. CHIRON 1B reference numbers (what 3A.1 must reproduce)

From `research/FLAGSHIP_T16384_2026_05_14.md`, the fresh-start T=16384
contiguous run:

| Step | Loss (inst) | EMA | Best | tok/s |
|---:|---:|---:|---:|---:|
| 1 | 10.58 | 10.58 | 10.58 | 17715 |
| 5000 | 4.69 | 4.93 | 4.60 | 20110 |
| 10000 | 4.39 | 4.61 | 4.08 | 20111 |
| 15000 | 4.29 | 4.53 | 3.96 | 20114 |
| 20000 | 4.78 | 4.42 | 3.79 | 20122 |
| 25000 | 4.07 | 4.16 | 3.768 | 20122 |
| 30000 | 4.27 | 4.29 | 3.7670 @ 29341 | 20108 |

Aggregate val metrics at step 32500 (post-live-eval suite):
val NLL = 4.71, BPB = 1.58, PPL = 111.6, top-1 = 11.6%, top-5 = 45.5%,
top-10 = 66.7%.

Position-stratified val NLL at step 32500 (8 buckets × 2048):
[4.99 / 4.72 / 4.60 / 4.61 / 4.60 / 4.70 / 4.80 / 4.70].

VRAM: 13.22 / 15.56 GB (15% headroom) on RTX-4080-SUPER.

### 3A.1 reproduction pass criterion

At step 1000 and step 2000 of the fresh-init repro, instantaneous loss
must be within ±0.05 nat of the linearly-interpolated published curve.
EMA loss within ±0.10 nat. Throughput within ±5% of 20,108 tok/s. Peak
VRAM within ±0.5 GB of 13.22 GB.

If the reproduction is OUTSIDE these bands, F-Phase3-1 fires:
investigate determinism / kernel-launch ordering / cuBLAS algo
selection before continuing.

---

## 3. Profiling methodology (3A.2)

Per the brief's P10 (no external library installs), profiling tools must
be either (a) already in the toolchain on the box (nvidia-nsight,
nvprof, cudaEvent timing in chiron_main itself) or (b) implemented as
new code in chiron_main as part of this gate.

The acceptable measurement methodology:
  - cudaEventRecord-bracketed kernels in chiron_main, summing wall
    time per kernel name across a 100-step warmed window.
  - NVML or `nvidia-smi --query-gpu=memory.used` polled at 10 Hz to
    track VRAM dynamics.
  - cudaMemGetInfo before / after each major allocation in chiron_main
    to attribute persistent allocations to buffer classes.

If `nvidia-nsys` / `nsys profile` is already installed system-wide
(it ships with CUDA toolkit), it MAY be used; that's not an external
lib install. Confirm at gate-start.

### 3A.2 pass criterion

Output `research/PHASE3_GATE3A_PROFILE.md` must contain:
  - Top-3 kernels by wall-clock %, each ≥5% of step time, named
    explicitly (e.g., "k_scfa_outer_qkv_proj", "softmax_ce_bwd",
    "k_int8_adamw_step"), with their tensor shapes.
  - Top-3 VRAM consumers by buffer class, each ≥5% of peak, named
    explicitly (e.g., "Adam m+v int8 state for all weights", "BF16
    weight master copies", "SCFA inner-attention activations").
  - Coverage check: top-3 + "other" sums to 100% of step time and
    ≥95% of peak VRAM.

If the top-3 kernels each are <5% (a flat profile), the bottleneck is
diffuse and the program reports that as the Gate-3A finding (which
shapes Gate-3B candidate selection — favoring mechanisms that broadly
reduce cost, not single-kernel ones).

---

## 4. VRAM accounting methodology (3A.3)

Per-component breakdown of the 13.22 GB peak. Counted classes:

  1. BF16 weight master copies (E + W_q/W_k/W_v/W_o per layer + MLP
     per layer + LN scales + W_out tied with E).
  2. int8 Adam moment state (m, v) + per-block FP32 scale.
  3. BF16 gradient buffer.
  4. SCFA inner-attention activations (k=1024 compressed-T buffers per
     layer).
  5. SCFA outer-attention activations / depthwise-conv state.
  6. Residual stream activations (BF16-residual-p storage).
  7. MLP activations (SwiGLU intermediate buffers).
  8. Logits-readout BF16 storage.
  9. Misc (cuBLAS workspace, NCCL buffers if any, kernel-launch
     overhead).

Methodology: cudaMemGetInfo bracketing each allocation site in
chiron_main; emit a single accounting JSON at gate-eval time.

### 3A.3 pass criterion

Output `research/PHASE3_GATE3A_VRAM.md` must contain:
  - A row per buffer class with: name, allocation site, size in MB,
    fraction of peak.
  - Sum of accounted classes ≥ 95% of measured peak (the remainder
    is allocator overhead and may be flagged separately).
  - The top-3 buffer classes by size, named explicitly.

---

## 5. Pre-committed pass/fail for Gate-3A overall

Gate-3A PASSES when ALL of:
  - 3A.1 fresh-init reproduction is within the bands in §2.
  - 3A.2 profile output identifies the top-3 wall-clock bottlenecks
    each ≥5% of step time.
  - 3A.3 VRAM accounting sums to ≥95% of measured peak, with named
    buffer classes.
  - Multi-seed not required for 3A.1 (single-seed reproduction is a
    sanity check, not a research claim).

Gate-3A FAILS if 3A.1 cannot reproduce (F-Phase3-1 fires) or if 3A.2/
3A.3 cannot be made to converge to the coverage thresholds. Either
failure mode must be addressed before Gate-3B candidates can be
proposed — without bottleneck identification, Gate-3B is shooting in
the dark.

---

## 6. Risks (pre-registered)

| Risk | Probability | Mitigation |
|---|---|---|
| **R-3A-1:** Fresh-init reproduction trajectory drifts from the published curve due to a determinism gap introduced in a recent commit | medium | bisect: run the exact git SHA that produced the original flagship; if that reproduces, identify which commit broke determinism |
| **R-3A-2:** cudaEvent-based per-kernel profiling has measurement overhead that distorts the profile at scale | low | warm 100 steps before measuring; use a separate measurement-only build with cudaEvents bracketed; verify total cudaEvent time ≤ 5% of step time |
| **R-3A-3:** cudaMemGetInfo doesn't expose per-allocation attribution | low | augment chiron_main with allocation-site logging (just before each cudaMalloc, print the buffer class + size); this is a small instrumentation change, ~1 day of work |
| **R-3A-4:** RTX-4080-SUPER thermals + boost-clock variance produce throughput noise > 5% across seeds | low | report median of 3 runs; lock GPU clocks via `nvidia-smi -lgc` if needed |
| **R-3A-5:** The "top-3 ≥5%" criterion is too loose because the profile is dominated by SCFA outer GEMMs (40%+) | none — that's the actual signal | report whatever the profile says; if SCFA outer is 40%, that IS the bottleneck and Gate-3B candidates should target it |

---

## 7. Out of scope for Gate-3A

- External baselines (Llama-3 / Mamba-2 / OLMo / SmolLM / etc.) — per
  P10, no external installs.
- Downstream eval suites (LAMBADA / HellaSwag / etc.) — these
  require eval-prompt datasets we don't have offline; Gate-3E will
  handle inference-time evaluation using our own methodology.
- Any new mechanism — Gate-3A is characterization-only; the first
  novel mechanism lands at Gate-3B.
- Multi-GPU / DDP profiling — single 16-GB-GPU only.
- Cross-codebase comparisons — only `glades-trainer` matters here.

---

## 8. Deliverables checklist

- [x] Pre-register this file in memory (`vesta-phase3-gate3a-prereg`).
- [x] Audit glades-trainer for the existing instrumentation. (Done
      2026-05-19 — see audit results below.)
- [ ] Run fresh-init flagship (`sh run.sh flagship --steps 2500
      --seed 1337`) for 2000 steps; compare to published curve.
- [ ] Add cudaEvent bracketing around the top-level training-step
      regions in chiron_main (forward, SCFA inner, SCFA outer, MLP,
      logits-readout, backward, Adam-step). Measurement-only build
      flag, e.g. `--profile-kernels`. ~1 day of instrumentation.
- [ ] Add cudaMemGetInfo bracketing around each major allocation in
      chiron_main; emit `vram_accounting.json` at gate-eval time.
      ~half day.
- [ ] Run the profile harness for 100 warm + 100 measured steps;
      produce `research/PHASE3_GATE3A_PROFILE.md`.
- [ ] Produce `research/PHASE3_GATE3A_VRAM.md` from
      `vram_accounting.json`.
- [ ] Pass/fail interpretation table.
- [ ] Memory entry recording Gate-3A status: PASSED / FAILED /
      PARTIAL.

### Audit results (2026-05-19)

Existing instrumentation in glades-trainer's chiron_main path:
- Per-step throughput (tok/s) is already logged.
- Loss is logged every `--log-every` steps with EMA.
- Validation eval at `--val-every` steps.
- Per-kernel timing: NOT INSTRUMENTED in chiron_main today;
  cudaEvent bracketing would be new code.
- VRAM tracking: only peak VRAM at end-of-step is logged; per-buffer
  attribution is NEW WORK.

Implication: Gate-3A.1 (reproduction) is immediately runnable.
Gate-3A.2 and Gate-3A.3 require ~1.5 days of new instrumentation
code before they can produce data.

---

## 9. After Gate-3A

The Gate-3A profile + VRAM report selects the bottleneck Gate-3B
will target. Likely candidates (informed by CHIRON's prior iter
journal, which showed SCFA outer GEMMs and Adam-state size as
recurring bottlenecks):

  - If top-3-wall-clock = SCFA outer projections + MLP forward +
    int8-Adam step → Gate-3B candidate could be a fused-Adam variant
    or an FFN-compression shift (PHOENIX-1BIT #74 production wire-in,
    or CSP #27 production wire-in).
  - If top-3-VRAM = BF16 weights + int8-Adam state + SCFA inner cache
    → Gate-3B candidate could be a weight-state compression shift
    (paradigm beyond int8 — int4 with per-block scale, or PHOENIX-1BIT
    on weights, or LoRA-style low-rank Adam state).
  - If top-3-VRAM = activations + residual stream + MLP intermediates
    → Gate-3B candidate could be activation checkpointing or a
    sub-quadratic activation-memory reduction (CASCADE's
    info-bottleneck, ATC-Δ's cross-step Taylor).

The exact Gate-3B candidate is NOT pre-committed here; it is selected
based on Gate-3A's profile and committed in
`research/PHASE3_GATE3B_PREREG.md` before any code lands.

---

This pre-registration is committed before any Gate-3A code lands. Per
the brief's P6 prohibition (no silent claim-dropping), if Gate-3A
falsifies any pre-committed threshold above, the result is published
honestly and the program adapts — not silently re-targeted.
