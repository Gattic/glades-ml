# CHIRON 1B Phase-3 Regularization Sub-Program — Closure (2026-05-24)

**Status:** CLOSED. Production flagship `chiron_1B_T16384_regstack_phase2.final`
accepted as the stable terminus.
**Decision date:** 2026-05-24
**Trigger:** four arcs investigated, zero ships, strong signal of
diminishing returns on port-style mechanisms at this
architecture/hardware combination.

---

## Production state at closure

**Flagship:** `chiron_1B_T16384_regstack_phase2.final`
- Stack: regstack Phase 2 = v5+FP8 ship + Z-loss (λ=1e-4) + QK-Norm
  (DeepSeek-V3 / Llama style, per-head L2 norm + learnable γ).
- Val NLL: **3.5734** @ step 30000 (vs v5+FP8 ship 4.1717, −0.5983 nat).
- Throughput: **28,072 tok/s** @ T=16384 on RTX-4080-SUPER.
- VRAM: 14.97 / 15.56 GB peak.
- Reproduce: `cd ~/dev/glades-trainer && sh run.sh flagship
  --zloss-coef 1e-4 --qk-norm`.
- Full spec: see `CLAUDE.md` "Current Production Flagship" section.

This is the production flagship that the Phase-3 sub-program could not
beat with any of its four investigated mechanism arcs.

---

## Phase-3 arc inventory (chronological)

### Arc 1: MTP (Multi-Token Prediction) — NEGATIVE

**Spec:** auxiliary-objective port from DeepSeek-V3 (predict +2 token
in addition to +1).
**Validation:** B3 (MTP only) and B4-full (Z-loss + QK-Norm + MTP)
plus sweeps at λ=0.01 + late-positions-only.
**Result:** B3 null at 5k; B4-full +0.04 nat WORSE than B4 (Z+QKN
only). Five MTP variants tested; none cleared a ship gate. Combined
recommendation variants showed training divergence.
**Disposition:** code stays in trainer (default off, bit-identical),
MTP investigation closed. See `research/REGSTACK_PHASE2_2026_05_22.md`
§§ "MTP follow-up arc" and "MTP sweeps".
**Lesson:** auxiliary-head mechanisms didn't translate; t+2 target
shift may be too short at T=16384, and the shared-weight assumption
collapses causal-LM ability.

### Arc 2: LayerDrop (Stochastic Depth) — FAIL

**Spec:** architecture-side depth-noise port from Fan 2019 / Huang
2016. Linear-rising per-layer Bernoulli skip at p_max=0.1.
**Validation:** L0 baseline + L1 5k pilot, seed=1337.
**Result:** ΔNLL = +0.073 nat (above the +0.05 outright-fail bar).
Wall +4.10% (real but doesn't offset NLL cost). Position-stratified
uniform regression (+0.04 to +0.12 nat across 8 buckets).
**Disposition:** code stays as opt-in flag (`--layer-drop-pmax`
default 0.0 = bit-identical). Arc closed per spec §3.2 OUTRIGHT FAIL
rule. Mid-arc val-mode bug discovered + fixed (forward() now takes
`isTraining` parameter; benefits future arcs).
**Lesson:** CHIRON's symplectic `(p, q) → (p + shear(q), reln(q))`
update structure required hard-drop semantics (no `1/(1-p_l)`
rescaling on the non-linear `reln` transformation). At p_max=0.1
density on a 1B model at 5k single-seed, the resulting train/inference
distribution mismatch is too costly.
See `research/LAYERDROP_5K_FAIL_2026_05_23.md`.

### Arc 3: UL2 (Mixture-of-Denoisers) — FAIL

**Spec:** objective-side mixture port from Tay et al. 2022. Three
denoisers (R short spans, S causal LM, X long spans) sampled uniformly
1/3 per training step; same-position prediction with span-corrupted
input.
**Validation:** U0 baseline + U1 5k pilot.
**Result:** **ΔNLL = +2.33 nat** at 5k (~46× the +0.05 outright-fail
bar; catastrophic regression). Wall stable (−0.16%, negligible). U1
trajectory roughly equivalent to U0's step-1500 throughout — causal-LM
convergence dramatically slowed by mixture training.
**Disposition:** code stays as opt-in flag (`--ul2-enabled` default
off). Arc closed per spec §3.2 R-UL2-9 trigger ("close arc if much
worse"). Mid-arc R-UL2-3 pre-registered risk fired (MASK row zero-init
caused NaN at p_max=0.1 X-denoiser density; reverted per spec
contingency to small-Gaussian init).
**Lesson:** R/X same-position prediction is structurally easier than
causal LM; 2/3 of training steps reinforce the easier task, only 1/3
reinforce causal LM, collapsing the shared-weight assumption. CHIRON's
symplectic update is tuned for next-token flow; same-position
prediction uses information flow differently.
See `research/UL2_5K_FAIL_2026_05_23.md`.

### Arc 4: CUDA Graphs Production-Readiness — PARTIAL_PROGRESS

**Spec:** pure wall-time arc. Make `--cuda-graphs` (paradigm #51
ATLAS-COMPILE) functional under the flagship recipe by removing
per-step CPU branching from SCFA + the cublasGet/SetMathMode hot-path
toggle.
**Validation:** smoke-tested at 10 steps after each fix-up.
**Result:** capture FUNCTIONAL but replay ~2× slower than direct
emission (~12,790 tok/s vs 28,072 baseline). Step-1 val NLL drifts
~0.2 nat from direct emission (~2000× the bit-identicality bar).
Both spec gates FAIL.
**Disposition:** 8 library + 6 trainer commits land real codebase
improvements that benefit ALL training paths (gpu_blas.cu two-handle
math-mode dispatch, GpuBuffer::zero now async/stream-tagged,
qknorm_gamma_scale GPU kernel, narrower auto-disable list).
`--cuda-graphs` flag is now functional but documented as research-only.
Production unchanged.
**Lesson (corrected 2026-05-24 post-closure nsys investigation):**
the real cause is **CPU-GPU overlap loss**, not captured-memset
serialization and not cuBLAS algorithm regression. Direct emission's
~500 ms/step of host `cudaLaunchKernel` dispatch was running in
parallel with the GPU's ~580 ms/step of work; graph replay collapses
host dispatch to ~5 ms (one `cudaGraphLaunch`), leaving the host
nothing to do during GPU execution. The host then hits a blocking
`cudaStreamSynchronize` that sees the full GPU work (~558 ms) instead
of the ~180 ms residual it saw in direct mode. This is a structural
property of CUDA Graphs at workload sizes where GPU is the long pole
and dispatch fits comfortably inside GPU work — exactly CHIRON 1B's
regime at T=16384. Not fixable by per-layer zero extraction (kernel
counts and per-call times were verified identical between modes via
nsys). Not fixable by AUTO_PARALLELISM (which would address GPU-side
node concurrency, not CPU-GPU overlap). The NLL drift mechanism is
still unexplained — possibly stochastic-rounding seed perturbation,
possibly an actual race exposed when capture-mode neutralizes
`cudaStreamSynchronize` calls inside the captured region. See the
"Root cause" + "Lessons learned" sections of
`research/CUDA_GRAPHS_PARTIAL_PROGRESS_2026_05_24.md` for the full
amended diagnosis.

---

## Session-aggregate evidence

| Class | Mechanism | Result | ΔNLL | ΔTok/s |
|---|---|---|---:|---:|
| Aux-objective | MTP | NEGATIVE | +0.04 (B4-full vs B4) | ~0 |
| Architecture-side | LayerDrop | FAIL | +0.073 | +4.10% |
| Objective-side | UL2 mixture | FAIL | +2.33 | −0.16% |
| Wall/infra | CUDA Graphs | PARTIAL | +0.20 (drift) | −54% (replay vs direct) |

Four mechanism classes investigated; none cleared its respective ship
gate. Two distinct failure modes:

1. **Symplectic architecture incompatibility** (LayerDrop, UL2, MTP):
   port-style mechanisms designed for standard residual transformers
   (`x_{l+1} = x_l + F_l(x_l)`) interact poorly with CHIRON's
   `(p, q) → (p + shear(q), reln(q))` symplectic update. The shared-
   weight assumption between training objectives + the directional
   information flow tuned for next-token prediction don't generalize
   to skip-deeper-layers or predict-same-position mechanisms.

2. **Workload-size mismatch with CUDA Graphs** (CUDA Graphs arc;
   diagnosis corrected 2026-05-24): graphs help when host dispatch
   is the bottleneck (small kernels, high launch rate, GPU idle
   between launches). At CHIRON 1B at T=16384, the per-step workload
   is dominated by ~580 ms of dense GPU work that comfortably
   overlaps with ~500 ms of host `cudaLaunchKernel` dispatch in
   direct mode. Collapsing dispatch to ~5 ms via a single
   `cudaGraphLaunch` doesn't help (GPU was the long pole) and breaks
   the overlap that was hiding the GPU work behind dispatch. This is
   a workload-regime mismatch, not a hardware-feature absence — the
   original closure-doc claim that "CUDA 13.2 dropped AUTO_PARALLELISM"
   was the structural blocker is **retracted**. `AUTO_PARALLELISM`
   addresses GPU-side node concurrency, which is not the cause here.

---

## What ships at closure

**Flagship:** unchanged. `chiron_1B_T16384_regstack_phase2.final` is
the stable terminus.

**Codebase improvements that DO land** (from arcs 2-4, even though
no shipped flagship update):

### From LayerDrop arc (architecture-side)
- `forward()` now takes `bool isTraining = true` parameter, propagated
  to all 4 forward call sites (`run_validation`, `run_nita_eval`,
  LAMBADA, ORION HVP). Future arcs benefit from this val-mode gating
  primitive.
- `layer_drop_p_l` + `layer_drop_keep` template helpers in
  `transformer_kernels.h` (paradigm-25-style stochastic-depth
  primitives, useful for any future depth-noise mechanism).

### From UL2 arc (objective-side)
- `ul2_sample_span_mask` template helper for Poisson-sampled
  span masks (general-purpose; usable for future span-corruption
  mechanisms).
- 3 unit tests (`CHIRONUL2SpanSamplerMeanSpanTest`,
  `CHIRONUL2SpanSamplerRateTest`, `CHIRONUL2DisabledParityTest`).

### From CUDA Graphs arc (wall-time / infra)
- **`gpu_blas.cu` two-handle dispatch**: removes hot-path
  `cublasGetMathMode`/`cublasSetMathMode` toggle, saves 2 cuBLAS API
  calls per GEMM. Structurally cleaner.
- **`GpuBuffer::zero()` now async/stream-tagged**: replaces blocking
  `cudaStreamSynchronize + cudaMemset` with `cudaMemsetAsync` on
  computeStream. Faster zero AND capture-safe. Applies to ALL training
  paths.
- **Two free-standing `cudaMemset` calls** in `gpu_kernels.cu`
  (`cross_entropy_nll_loss_bf16` + `argmax_count_matches_bf16`)
  similarly fixed.
- **`qknorm_gamma_scale_gpu` GPU kernel**: eliminates per-layer
  per-step D2H/CPU/H2D round-trip for QK-Norm's γ·sqrt(dHead) scale.
  Saves ~120 µs/step on ALL training paths (not just graphs).
- **`--cuda-graphs` flag is now functional** (no longer auto-disabled
  by `--scfa`). Available for research/profiling.

---

## Recommended future direction (NOT a commitment)

Per the closure decision, the Phase-3 regularization sub-program is
terminated. Future research arcs at CHIRON 1B should consider:

1. **Architecture-specific mechanisms.** Rather than porting from
   standard residual transformers, design mechanisms that interact
   *with* CHIRON's symplectic update. Examples: a symplectic-aware
   depth-noise mechanism, a momentum-preserving alternative to
   layer-drop, an objective designed for the `(p, q)` representation
   pair.

2. **Hardware-feature-dependent arcs need explicit pre-arc empirical
   validation.** The CUDA Graphs arc spec underestimated the work
   (3 blockers found, 6+ landed). Future infra arcs at this scale
   should include a 1-hour smoke pilot BEFORE committing to the spec,
   to surface unanticipated blockers.

3. **Iter-style wall mining is still tractable** but increasingly
   hits diminishing returns post-iter-117 META. Multi-iter scope
   items from that META (reln-fusion, FlashAttention-fused SCFA
   inner) remain open as research-grade arcs.

4. **The data side has been underexplored.** Curriculum, dataset
   composition, deduplication, quality-filter mechanisms — these are
   orthogonal to the architecture-vs-mechanism axes that dominated
   Phase-3 and may have higher EV given the architectural ceiling
   evidence.

5. **Take a break.** Four consecutive non-ship arcs is a strong
   psychological signal that fresh perspective may help more than
   the next mechanism investigation.

---

## Honest publication per Phase-3 P6

The Phase-3 regularization sub-program is closed without a flagship
update. This is the honest outcome of four well-scoped arcs that each
followed their pre-registered gates and closed at the spec's failure
threshold. No silent re-targeting, no rationalization of failed gates
into successes, no mid-arc scope drift that hid the real result.

Per the Phase-3 P6 rule, this closure document is the formal record
of four published negative/partial results. The codebase improvements
that DID land are real wins for future research; the architectural
ceiling evidence is real signal for direction-setting.

The production flagship `chiron_1B_T16384_regstack_phase2.final`
stands as the stable production state of the CHIRON 1B program.

---

## Cross-references

- Flagship spec + reproduce: `CLAUDE.md` "Current Production Flagship"
  section.
- Regstack Phase 2 ship doc: `research/REGSTACK_PHASE2_2026_05_22.md`.
- MTP investigation: `research/REGSTACK_PHASE2_2026_05_22.md` §
  "MTP follow-up arc" + "MTP sweeps".
- LayerDrop FAIL: `research/LAYERDROP_5K_FAIL_2026_05_23.md`.
- UL2 FAIL: `research/UL2_5K_FAIL_2026_05_23.md`.
- CUDA Graphs PARTIAL_PROGRESS:
  `research/CUDA_GRAPHS_PARTIAL_PROGRESS_2026_05_24.md`.
- Specs: `docs/superpowers/specs/2026-05-22-chiron-1b-regularization-stack-design.md`
  (regstack), `2026-05-23-chiron-1b-layerdrop-design.md`,
  `2026-05-23-chiron-1b-ul2-design.md`,
  `2026-05-23-chiron-1b-cuda-graphs-design.md`.
- Plans: parallel files under `docs/superpowers/plans/`.
- Memory entries: `regstack_phase2_ship`, `layerdrop-arc-fail`,
  `ul2-arc-fail`, `cuda-graphs-partial-progress`, and the new
  `phase3-closure`.
