# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current Production Flagship — CHIRON 1B @ T=16384 (reanchor cure ship 2026-06-27)

The current production LLM flagship is **CHIRON 1B reanchor-cure**
(checkpoint `chiron_1B_T16384_reanchor5B_finish.final`). It is a **perplexity
flagship**: verified **val NLL ~1.92, −0.62 nat over the prior data-scale ship
(2.54)** — the second-largest single jump in the lineage. The gain comes not from
more data but from **curing the q-side reverse-amplification instability at its
source**, which was silently taxing perplexity ~0.7–0.8 nat under the prior
gg-clamp containment.

- **Shape**: identical to the data-scale ship — m=2048, L=24, nH=16, dH=256,
  V=32000 BPE, T=16384; 870.94M params + 384 QK-Norm γ (persisted, CHRF bit 256).
- **The mechanism — `--reln-reanchor` (ReLN reverse-consistency cure)**: the
  q-side instability is a ReLN-backward overflow — the recomputed activation
  `q_in` is normalized with **saved forward stats that have drifted**, inflating
  `xhat` ~13× *before* the `dgamma = Σ_t dout·xhat` sum overflows it. The cure
  (`chiron_reln_backward_reanchor`, glades-ml `gpu_chiron.cu`) re-derives
  mean/invStd **from `q_in` itself** in the backward, so `xhat` is unit-RMS by
  construction — near-identity on healthy steps, removes the inflation at its
  source on drifted ones. This **replaces gg-clamp** (which only *contained* the
  instability by clamping ‖g‖→1700–5400 every step past 1.6B — a containment that
  silently crippled the optimization).
- **Stack/recipe**: the data-scale ship recipe (see "Prior data-scale Flagship"
  below) **MINUS `--grad-group-clamp`, PLUS `--reln-reanchor`** (default-off,
  gated). Same two-stage schedule: constant `lr 3e-4` base to step 60000 (3.93B)
  then flat `lr 3e-5` finish to 66000 (4.33B).
- **Stability**: trained to **4.33B with 0 grad-skips, 0 bad-gradient triggers** —
  the instability that killed the no-clamp control at 1.49B and that gg-clamp
  fights chronically past 1.6B is a **non-event** here (‖g‖ ~1.0 throughout). No
  prior run trained cleanly past ~2B.
- **Verified val** (two independent methods, both vs the prior flagship):
  trainer wide 32-batch val **1.92** (flagship 2.54); chiron_infer teacher-forcing
  **1.78** (flagship 2.44 ≈ its documented 2.348). acc1 0.51 vs 0.36. The
  finish checkpoint is ~0.06 better than the step-60000 base.
- **Reproduce training** (two deterministic stages, from `~/dev/glades-trainer`):
  1. base: `sh run.sh flagship --accum 4 --lr 3e-4 --warmup 750 --sira-warmup 250
     --zloss-coef 1e-4 --qk-norm --sira-coef 1e-2 --sira-energy-weight 1.0
     --sira-balance-weight 0.25 --sira-action-weight 0.0 --grad-clip 0.5
     --dq-layer-clamp 1.0 --dq-embed-clamp 1.0 --reln-reanchor --steps 60000
     --seed 1337 --save-every 6000` (NO `--grad-group-clamp`). Pre-create `--save`.
  2. finish: resume the step-60000 checkpoint with `--lr 3e-5 --warmup 0
     --no-resume-warmup --steps 66000 --reln-reanchor` + the same recipe flags.
- **Run inference**: `cd ~/dev/glades-trainer && sh runner.sh --flagship`
  (prefers `chiron_1B_T16384_reanchor5B_finish.final`). **Serving fix (2026-06-27)**:
  chiron_infer now **auto-enables `--fuse-attn-reln`** for SCFA checkpoints (no
  per-layer `gamma_p`) — without it, the inference forward omitted the fusion and
  gave teacher-forced nll ~63 (garbage) for ALL production checkpoints incl. every
  prior flagship, on every serving path incl. `runner.sh`. Combined with the
  earlier QK-Norm γ auto-enable (bit 256), inference now reproduces training val.
- **Ship record / evidence**: `research/RELN_REANCHOR_GATE2_2026_06_27.md`
  (Gate-2 result + verification + mechanism). Gate-1 (containment vs gg-clamp):
  `research/RELN_REANCHOR_GATE1_2026_06_24.md`. Design/plan:
  `docs/superpowers/specs/2026-06-23-reln-reverse-consistency-design.md`,
  `docs/superpowers/plans/2026-06-23-reln-reverse-consistency.md`.
- **Caveats**:
  - **Single-seed (1337) ship — multi-seed (≥3) Gate-0 confirm NOT run (owner
    decision 2026-06-27).** Unlike prior lineage ships (e.g. the SIRA+clamp
    3-seed gate), this flagship is promoted on one seed. It is verified two
    independent ways at that seed (trainer wide-val 1.92 + chiron_infer TF 1.78),
    and the −0.62 nat magnitude dwarfs seed variance (~0.02) — but cross-seed
    reproduction is NOT established. To add it later: re-run the base recipe under
    seeds 2024/4242, expect ~1.98 base val + 0 grad-skips.
  - **Perplexity flagship, not a generator** — same as all CHIRON flagships.
    Free generation degenerates into a repetition loop even from clean prose
    context, under greedy AND nucleus sampling (the next-token distribution
    collapses onto the repeat so hard top-p can't escape). Diagnosed 2026-06-27 as
    **exposure bias / an intrinsic repetition attractor — NOT inference-fixable**
    (window-slide, pad, decoding, and forward-bug hypotheses all refuted; the
    forward is correct, TF top1 0.53). The −0.62 nat perplexity win does NOT confer
    coherent generation. Fix is generation-aware *training*, deferred to a separate
    plan. Full diagnosis + the deferred-training scope:
    `research/CHIRON_GENERATION_LOOP_FIX_2026_06_27.md`. Diagnostic flags landed in
    chiron_infer (`--gen-metrics`, `--seed-tail`, `--tokens-file-offset`).
  - The flat-3e-5 finish added only ~0.06 here (vs ~0.26 for the gg-clamp
    flagship) because re-anchor's base was still descending — a longer base +
    later finish may extract more (open follow-up).

## Prior data-scale Flagship — CHIRON 1B @ T=16384 (data-scale ship 2026-06-22, kept for context)

The prior flagship **CHIRON 1B data-scale**
(checkpoint `chiron_1B_T16384_datascale5B_finish_clean.final`) remains the recipe
base for the reanchor-cure ship above (which swaps gg-clamp for `--reln-reanchor`).
It is a **perplexity flagship**: −0.9/−1.0 nat val NLL over the prior SIRA+clamp ship,
the largest single jump in the lineage, from training on **~10× more data**.

- **Shape**: m=2048, L=24, nH=16, dH=256, V=32000 BPE, T=16384 context;
  870.94M params + 384 per-head QK-Norm γ (now **persisted** in the checkpoint —
  CHRF flag bit 256, the serving fix below).
- **Stack**: the SIRA+clamp ship recipe (QK-Norm + Z-loss + terminal SIRA +
  dq-clamps on v5+FP8/CUDA 13.2 — see "Prior SIRA+clamp flagship" below) **PLUS
  the data-scale recipe landed 2026-06-18/22**:
  - **Data scale + batch recipe**: `--accum 4 --lr 3e-4` (effective batch 65536
    tok/step) trained to **~2B tokens** (vs the prior flagship's 0.49B / 30k
    steps). This is the dominant win — perplexity keeps dropping with tokens
    well past where the prior flagships stopped.
  - **`--grad-group-clamp 1.0`** (`clamp_vector_l2norm` on each layer's
    dgamma/dbeta L2 norm before the global grad-norm): the **containment** for
    the q-side reverse-amplification instability, which re-emerges at ~1.6B
    tokens and would otherwise diverge (accum=4 run #1 died @1.49B). gg-clamp
    fires ~every step past 1.6B but holds **0 grad-skips** to 2B+.
  - **Constant-LR + sharp finish schedule**: constant `lr 3e-4` for the data
    descent, then a **flat `lr 3e-5` finish** (`--warmup 0 --no-resume-warmup`,
    ~6k steps). A *gradual cosine* over the full run UNDERPERFORMED by ~0.3 nat
    (decays LR too early, starving the mid-phase descent) — the anneal must be
    late + sharp. See `research/DATASCALE_FLAGSHIP_AND_SERVING_2026_06_20.md`.
- **Perf / val**: same ~28k tok/s + ~14.7/15.56 GB VRAM as the SIRA+clamp ship
  (data-scale is a token-count change, not a per-step cost). **Held-out val
  ~2.5** (best 2.4346; 4-batch window variance 2.43–2.62; chiron_infer
  teacher-forcing nll **2.348** with the exact γ) vs the SIRA+clamp ship's
  **3.5062** → **Δ −0.9 to −1.0 nat**. 0 grad-skips throughout.
- **Reproduce training** (two deterministic stages):
  1. constant-LR base: `cd ~/dev/glades-trainer && sh run.sh flagship --steps
     60000 --accum 4 --lr 3e-4 --warmup 750 --sira-warmup 250 --zloss-coef 1e-4
     --qk-norm --sira-coef 1e-2 --sira-energy-weight 1.0 --sira-balance-weight
     0.25 --sira-action-weight 0.0 --grad-clip 0.5 --dq-layer-clamp 1.0
     --dq-embed-clamp 1.0 --grad-group-clamp 1.0 --save-every 10000 --seed 1337`
     (NO `--lr-decay` — constant LR). Pre-create the `--save` dir (silent
     save_full failure otherwise).
  2. flat-3e-5 finish: resume the step-60000 checkpoint with `--lr 3e-5
     --warmup 0 --no-resume-warmup --max-steps 66000` + the same recipe flags.
- **Run inference**: `cd ~/dev/glades-trainer && sh runner.sh --flagship`
  (prefers `chiron_1B_T16384_datascale5B_finish_clean.final` since 2026-06-22).
  **Serving fix**: chiron_infer historically lacked QK-Norm (silently broken for
  ALL SCFA checkpoints incl. prior flagships since the 2026-05-22 regstack
  landing); it now implements QK-Norm and **auto-enables it when the checkpoint
  carries per-head γ** (bit 256). Inference reproduces training val perplexity.
- **Ship record / evidence**: `research/DATASCALE_FLAGSHIP_AND_SERVING_2026_06_20.md`
  (full session: data-scale run, instability containment, the cosine-vs-finish
  lesson, the QK-Norm serving fix + γ-persistence, generation repetition).
- **Caveats**:
  - **Perplexity flagship, not a generator.** Free generation from short prompts
    is incoherent — short prompts pad to T with token-0 (OOD) AND the model has
    a strong repetition attractor. This is true of **all** CHIRON flagships
    (they're perplexity-optimized); the ship metric is val NLL, same standard as
    predecessors. chiron_infer has repetition control (`--freq-penalty` etc.) but
    coherent generation needs generation-aware fine-tuning, not decoding.
  - **Instability is contained, not cured.** gg-clamp holds it past 1.6B but it
    fires chronically; clean 5B isn't reachable with this recipe.
  - Val is window-noisy (4-batch); same-seed runs are not bit-reproducible at
    production shape (atomic-ordering noise). The constant+finish recipe IS
    reproducible step-for-step modulo that noise.

## Prior SIRA+clamp Flagship — CHIRON 1B @ T=16384 (SIRA+clamp ship 2026-06-12, kept for context)

The prior flagship **CHIRON 1B SIRA+clamp**
(checkpoint `chiron_1B_T16384_sira_clamp_phase2.final`) remains the recipe base
for the data-scale ship above (which adds only data scale + gg-clamp + the
finish schedule):

- **Shape**: m=2048, L=24, nH=16, dH=256, V=32000 BPE, T=16384 context;
  870.94M params + 384 QK-Norm γ (same as regstack Phase 2).
- **Stack**: regstack Phase 2 ship base (QK-Norm + Z-loss on v5+FP8/CUDA 13.2 —
  see "Prior regstack Phase 2 flagship" below) **PLUS the SIRA+clamp recipe
  landed 2026-06-11/12**:
  - **Terminal CHIRON-native SIRA** (`--sira-coef 1e-2 --sira-energy-weight 1.0
    --sira-balance-weight 0.25 --sira-action-weight 0.0 --sira-warmup 1000`):
    phase-space regularizer on the final `(q_L, p_L)` only.
  - **LR recipe change**: `--lr 7.5e-5 --grad-clip 0.5` (was 1e-4 / 0.5).
  - **q-side dq clamps** (`--dq-layer-clamp 1.0 --dq-embed-clamp 1.0`): per-row
    RMS clamp (`row_rms_clamp` kernel, `gpu_kernels.cu`) on the incoming dq at
    every backward layer boundary + on dq_0 before `embedding_scatter_add`.
    Bounds the early-layer q-side reverse amplification (dgamma/dbeta + dE
    overflow) that caused late guard-skip waves; identity on healthy steps.
- **Perf**: ~**28,100 tok/s** @ T=16384 with the cast-elim stack (shipped
  2026-06-12, +3.38% ± 0.17% n=3 multi-seed at NLL parity over the SIRA+clamp
  base's ~27,200; see `research/CAST_CENSUS_2026_06_12.md`), ~14.7/15.56 GB
  VRAM. **Final val NLL 3.5062 @ step 30000
  (seed 2024)**; 3-seed 30k gate (2024/4242/1337) mean **3.5223 ± 0.0146**
  (Δ −0.0672 best / −0.0511 mean vs regstack ship 3.5734). **Zero grad-skips**
  across all gate runs (seed 2024 previously had 5,809). Attribution via
  matched no-SIRA baseline: ≈ −0.036 nat from the LR recipe, ≈ −0.031 from SIRA.
- **Reproduce training**: `cd ~/dev/glades-trainer && sh run.sh flagship
  --zloss-coef 1e-4 --qk-norm --sira-coef 1e-2 --sira-energy-weight 1.0
  --sira-balance-weight 0.25 --sira-action-weight 0.0 --sira-warmup 1000
  --lr 7.5e-5 --grad-clip 0.5 --dq-layer-clamp 1.0 --dq-embed-clamp 1.0`.
  The cast-elim stack (wall-only, parity-clean, +3.38% n=3) is **default-ON
  since the 2026-06-12 Tier-2 promotion** (owner-blessed); strict
  reproduction of the pre-promotion training runs adds
  `--no-cast-elim-reln-q --no-cast-elim-dqperp --no-cast-elim-dy
  --no-cast-elim-inner-vo`.
- **Run inference**: `cd ~/dev/glades-trainer && sh runner.sh --flagship`
  (prefers `chiron_1B_T16384_sira_clamp_phase2.final` since 2026-06-12).
- **Ship record**: `research/SIRA_CLAMP_SHIP_2026_06_12.md`. Full evidence
  chain (candidate runs, overflow tracing, dE-clamp FAIL, per-layer PASS,
  attribution, multi-seed gate): `research/SIRA_TERMINAL_30K_RESULT_2026_05_27.md`.
  Clamp mechanism spec: `docs/superpowers/specs/2026-06-11-dq-embed-clamp-design.md`.
- **Caveats**: same-seed full-recipe runs are NOT bit-reproducible at
  production shape (atomic-ordering noise) — use rerun controls for parity
  claims. SIRA/clamp/regstack flags default off; omitting them reproduces
  the regstack Phase 2 ship (with `--no-cast-elim-*` for strict
  pre-2026-06-12 kernel-sequence reproduction — the cast-elim mechanisms are
  math-identical, so this matters only for exact replay/trace work).

## Prior regstack Phase 2 Flagship — CHIRON 1B @ T=16384 (ship 2026-05-22, kept for context)

The prior flagship **CHIRON 1B regstack Phase 2**
(checkpoint `chiron_1B_T16384_regstack_phase2.final`) remains loadable; the
SIRA+clamp ship builds on it with trainer-side, default-off flags only:

- **Shape**: m=2048, L=24, nH=16, dH=256, V=32000 BPE, T=16384 context.
- **Params**: 870.94M (~"1B") + 384 QK-Norm γ scalars (16 heads × 24 layers).
- **Stack**: v5+FP8 ship base (CUDA 13.2 + BF16 cast fix + FP8 readout —
  see "Prior v5+FP8 flagship" below) **PLUS the regularization-stack landed
  2026-05-22**:
  - **`--qk-norm`** (DeepSeek-V3 / Llama-3 style): per-head L2-normalize Q
    and K before attention, then replace `1/√d_h` with learnable per-head
    `γ_h` (init = log₂(T) = 14). Inserted between projection and attention
    core on both the FP32-inner and `--scfa-bf16-inner` BF16-TC paths via
    a new `qknorm_forward_gpu` + `scale_q_per_head` + flash-attention
    sequence. Backward recomputes qNorm/kNorm via BF16-TC projection
    rather than saving to L·k·dModel scratch (saves 768 MB at production
    shape).
  - **`--zloss-coef 1e-4`** (PaLM / T5 style): add zlossCoef·mean(logZ²)
    to the training loss plus 2·zlossCoef·logZ·probs in readout backward.
    Wired through the `--bf16-logits-storage` path via new
    `softmax_forward_bf16_with_lse` + `softmax_cross_entropy_bwd_bf16_zloss`
    kernels.
- **Perf**: **28,072 tok/s** @ T=16384 (was 28,887 at v5+FP8 ship,
  **−2.82%** wall — within the 5% spec budget; the BF16-TC backward
  recompute of qNorm/kNorm is the dominant cost). Same VRAM as v5+FP8
  ship (~14.97 GB peak). **Final val NLL 3.5734 @ step 30000** (vs
  v5+FP8 ship's 4.1717; Δ **−0.5983 nat BETTER**, ~30× the spec's 0.02
  nat improvement target). Cumulative NLL improvement since v5+FP8:
  **−0.5983 nat**.
- **Reproduce training**: `cd ~/dev/glades-trainer && sh run.sh flagship
  --zloss-coef 1e-4 --qk-norm`.
- **Run inference**: load `chiron_1B_T16384_regstack_phase2.final` explicitly
  (`runner.sh --flagship` prefers the SIRA+clamp ship since 2026-06-12).
- **Full spec**: `research/REGSTACK_PHASE2_2026_05_22.md` (Phase 2 ship
  doc with per-mechanism pilot deltas and B5 30k trajectory). Spec / plan
  that drove the arc: `docs/superpowers/specs/2026-05-22-chiron-1b-regularization-stack-design.md`
  and `docs/superpowers/plans/2026-05-22-chiron-1b-regularization-stack.md`.
- **Phase 2 evidence**: `research/REGSTACK_PHASE2_2026_05_22.md` (full 30k
  trajectory + position-stratified deltas + gate-by-gate evaluation).
- **Pilot evidence** (single-seed 5k @ T=16384): B0 baseline val NLL
  4.9140; B1 Z-loss 4.9207 (+0.007, noise); B2 QK-Norm 3.9412 (−0.97);
  B4 stacked 3.9396 (−0.97). QK-Norm dominates; Z-loss is a no-op at
  pilot scale but retained in B5 per §3.3 decision tree.
- **MTP deferred**: the original Phase 2 spec called for a third
  mechanism (MTP, B3). MTP scratch at production T·V (T=16384, V=32000)
  requires ~2.6 GB on top of the flagship's 14.97 GB working set —
  doesn't fit in the 15.6 GB ceiling without chunked-T or sparse-T
  implementation. Library port (commits d8098ee9d, d5ba43752) tested at
  small shape; production port deferred to follow-up arc.
- **Phase-3 program** (improving CHIRON 1B via novel research, no external
  libs / no external baselines): pre-registration in
  `research/PHASE3_GATE3A_PREREG.md`. Any new architecture work should
  anchor on the **SIRA+clamp flagship** (2026-06-12) as the baseline.

**Prior v5+FP8 flagship** (`chiron_1B_T16384_v5_fp8_phase2.final`,
28,887 tok/s, NLL 4.1717 @ 30k, CUDA 13.2 with FP8 readout) remains loadable
into the regstack Phase 2 stack with `--zloss-coef 0` (no `--qk-norm`) — both
flags default off; math bit-identical to v5+FP8 ship when off. For pure
"v5+FP8 ship reproduction without regstack" runs, omit the regstack flags
from `run.sh flagship`. The v5+FP8 ship was the CUDA 13.2 production
flagship that the regstack Phase 2 ship builds on; details:

### Phase-3 regularization sub-program — CLOSED 2026-05-24 (amended 2026-06-12)

**2026-06-12 amendment:** after this closure, the CHIRON-native
regularizer program (SIRA/PHS/PTOC, plan
`research/CHIRON_NATIVE_REGULARIZERS_GENERALIZERS_PLAN_2026_05_24.md`)
produced one ship: terminal SIRA + dq clamps became the production
flagship (see top of this doc). PHS and PTOC remain shadow-diagnostics
only (default-off; PHS default-enablement evaluated and rejected
2026-05-28, `research/PHS_DEFAULT_ENABLEMENT_RESULT_2026_05_28.md`).
The seed-2024 instability investigation that gated SIRA (late guard-skip
waves → q-side amplification → per-layer dq clamp) is fully recorded in
`research/SIRA_TERMINAL_30K_RESULT_2026_05_27.md`.

The original closure record below stands for the four ported-mechanism
arcs it covered. At closure time the regstack Phase 2 ship was the
**stable terminus**; four mechanism arcs investigated; zero
shipped:

| Arc | Class | Result | Δ |
|---|---|---|---|
| MTP (multi-token prediction) | aux-objective | NEGATIVE | +0.04 nat |
| LayerDrop (stochastic depth) | architecture-side | FAIL | +0.073 nat |
| UL2 (mixture-of-denoisers) | objective-side | FAIL | **+2.33 nat** |
| CUDA Graphs production-readiness | wall/infra | PARTIAL_PROGRESS | −54% wall, +0.2 nat drift |

Strong evidence accumulated:
1. **Symplectic architecture incompatibility:** port-style mechanisms
   from standard residual transformers don't translate to CHIRON's
   `(p, q) → (p + shear(q), reln(q))` update.
2. **Workload-size mismatch with CUDA Graphs** (diagnosis corrected
   2026-05-24 via nsys): graphs help when host dispatch is the
   bottleneck. At CHIRON 1B / T=16384, ~580 ms/step of GPU work
   already overlaps with ~500 ms/step of host `cudaLaunchKernel`
   dispatch in direct mode. Collapsing dispatch to ~5 ms via
   `cudaGraphLaunch` leaves the host nothing to do during GPU work,
   so the subsequent blocking `cudaStreamSynchronize` sees the full
   GPU latency instead of the residual it saw under overlap. Not
   fixable by per-layer zero extraction (kernel counts + per-call
   times verified identical between modes via nsys) and not fixable
   by `AUTO_PARALLELISM` (which addresses GPU-side node concurrency,
   not CPU-GPU overlap). The earlier "CUDA 13.2 dropped
   AUTO_PARALLELISM" framing is retracted. See
   `research/CUDA_GRAPHS_PARTIAL_PROGRESS_2026_05_24.md` for the
   amended diagnosis.

**Codebase improvements that DID land** (benefit ALL training paths,
not just the failed arcs):
- `forward()` now takes `bool isTraining = true` parameter (val-mode
  gating primitive; LayerDrop arc).
- `gpu_blas.cu` two-handle dispatch removes hot-path
  `cublasGet/SetMathMode` toggle (saves 2 cuBLAS API calls per GEMM;
  CUDA Graphs arc).
- `GpuBuffer::zero()` is now `cudaMemsetAsync` on computeStream
  (faster AND capture-safe; CUDA Graphs arc).
- `qknorm_gamma_scale_gpu` GPU kernel eliminates per-layer
  D2H/CPU/H2D round-trip for QK-Norm's γ·sqrt(dHead) scale (saves
  ~120 µs/step on ALL paths; CUDA Graphs arc).
- `--cuda-graphs` flag is now functional (research-only;
  not for production).
- `ul2_sample_span_mask` + `layer_drop_p_l` / `layer_drop_keep`
  template helpers in `transformer_kernels.h` (general-purpose
  primitives for any future mechanism that needs them).

**Full closure record:** `research/PHASE3_CLOSURE_2026_05_24.md`. See
also `research/LAYERDROP_5K_FAIL_2026_05_23.md`,
`research/UL2_5K_FAIL_2026_05_23.md`,
`research/CUDA_GRAPHS_PARTIAL_PROGRESS_2026_05_24.md`.

**Direction-setting:** future research arcs at CHIRON 1B should
consider architecture-specific mechanisms (designed for the symplectic
update, not ported from standard transformers) OR data-side work
(curriculum, composition, quality filters — orthogonal to the
architectural ceiling). Iter-style wall mining (per the iter 117 META)
remains tractable but increasingly diminishing returns. Of the multi-iter
scope items: **FlashAttention-fused SCFA inner is CLOSED NO-GO** (Gate-1 META
2026-06-12, `research/FA_INNER_GATE1_META_2026_06_12.md` — supersedes the
"still open" framing of this 2026-05-24 closure; iter 118 math-PASS but
+175% slow, iter 119 BF16-input NULL, then a fresh profile showed the inner
matmuls are <1% of the step so the realistic net was only +2–4%, below bar;
the thread was redirected to **cast-pipeline elimination, which shipped**
[+3.38% wall, the now-default-ON cast-elim stack]). Reconsider FA-inner only
if the EV changes (precision-tier change or much larger k). reln-fusion
remains open.

**Update (2026-06-30) — OBSD per-layer symplectic coupling: CLOSED NO-GO.**
The architecture-native arc this paragraph invites was tried: a per-layer
reversible, ReZero-gated, mass-preconditioned nonlinear drift
`q += a⊙tanh(M⁻¹⊙N(p)+b)` (let attention compose across depth, since the
flagship folds p→q only at the final layer). It is **stable** through the
1.6–2B-token instability regime (0 grad-skips; bounded-φ + p-side reanchor held
where the data-scale flagship died at 1.49B), but **regresses perplexity at
scale**: helps early (+0.05 nat to ~16.5k) then reverses to **−0.66 nat by 30k**
(matched single-seed E4, val 3.27 vs baseline 2.61) as the ReZero gate grows
unbounded (maxA 0.46→1.8) → ‖g‖ inflates → grad-clipping throttles effective LR
→ OBSD misses the baseline's late-training acceleration. A bounded-gate salvage
(`--drift-gate-cap`, late) had zero effect. So **per-layer symplectic coupling
is ruled out as a perplexity lever**; reln-fusion and data-side work remain the
open directions. Engineering (drift kernels reusing reanchor, `clamp_abs`, the
`a_drift` gate param + checkpoint bit 512, flags `--per-layer-drift` /
`--drift-warmup` / `--drift-gate-cap`) is committed **default-off** and reusable;
the arc also fixed a latent crash (**`--scfa` now requires `--bf16-weights`** —
else `scfa_attention_backward` SIGSEGVs). Full record:
`research/CHIRON_OBSD_RESULT_2026_06_27.md`; design/plan
`docs/superpowers/specs/2026-06-27-chiron-richer-symplectic-block-design.md` +
`docs/superpowers/plans/2026-06-27-chiron-richer-symplectic-block.md`.

## Prior v5+FP8 Flagship Details — CHIRON 1B @ T=16384 (kept for context)

The v5+FP8 flagship (predecessor):

- **Shape**: same as above.
- **Stack**: iter 116 ship 10-mechanism stack (see "Prior iter 116 flagship"
  below for the full mechanism list) **PLUS two layers landed 2026-05-21
  / 2026-05-22**:
  - **CUDA 13.2 toolchain** (was CUDA 12.0): cuBLAS 13.x BF16 GEMM dispatch.
  - **v5 BF16-cast fix** in `Backend/Machine Learning/Networks/cuda/gpu_blas.cu`:
    `sgemm_rowmajor_fast16bf_impl` pre-casts FP32 inputs to BF16 scratch +
    dispatches `cublasGemmEx` with `CUDA_R_16BF` inputs. Forces cuBLAS 13's
    `ampere_s1688gemm_bf16_*` fast path on Ada (without this, cuBLAS 13's
    heuristic for FP32-input FAST_16BF dispatches a non-BF16-specialized
    kernel ~1.82× slower per call on iter 116 ship shapes — a -15.18%
    regression vs CUDA 12.0 if unfixed).
  - **`scfa_B` constant cache**: trainer pre-casts the DCT-II basis to BF16
    once at SCFA init and registers it via the new `register_fast16bf_constant`
    API. The 216 SCFA outer GEMMs/step skip the per-call A-side cast.
  - **`--fp8-readout-fwd` enabled**: 3 readout GEMMs/step route through
    cuBLASLt FP8 (E4M3) at shape (T=16384, V=32000, m=2048) ≈ 1 TFLOP each.
    Adds +1.59% wall on top of v5 fix at strict NLL parity. Unblocked by
    CUDA 13.2 cuBLASLt FP8 algo coverage on Ada (was blocked on CUDA 12.0
    per `ITER62_FP8_READOUT_NULL.md`).
- **Perf**: **28,887 tok/s** @ T=16384 (was 28,257 at iter 116 ship,
  **+2.23%**; 14.97/15.56 GB VRAM, same as iter 116 ship), **final val NLL
  4.1717 @ step 30000** (iter 94 baseline at same recipe gives 4.1983; Δ
  **−0.0266 nat BETTER**, far within strict ±0.02). Mean trajectory drift
  vs iter 116 ship across 10 val checkpoints: +0.00007 nat (essentially
  zero). Cumulative since pre-ralph-loop 15,200 tok/s: **1.90×**.
- **Reproduce v5+FP8 (no regstack)**: `cd ~/dev/glades-trainer && sh run.sh flagship`
  (regstack flags `--zloss-coef` and `--qk-norm` default off → bit-identical
  to v5+FP8 ship). For the current regstack production flagship, see the
  reproduce command at the top of this doc.
- **Run inference (v5+FP8)**: load
  `chiron_1B_T16384_v5_fp8_phase2.final` explicitly.
- **Full spec**: `research/V5_FP8_30K_PHASE2_PASS_2026_05_22.md` (v5+FP8
  ship doc with full trajectory table) and `research/CUDA13_BF16_REGRESSION_FIX_2026_05_21.md`
  (diagnosis + fix details). Also see `research/FLAGSHIP_T16384_2026_05_14.md`
  for the iter 116 ship 10-mechanism details that v5+FP8 builds on.
- **Phase 2 evidence**: `research/V5_FP8_30K_PHASE2_PASS_2026_05_22.md`
  (apples-to-apples 30k retrain vs iter 116 ship & iter 94 triple-stack
  baselines; ship-clean PASS at +2.23% wall + −0.0322 nat NLL improvement
  vs iter 116 ship).
- **FP8 path evidence**: `research/FP8_READOUT_CUDA13_REVAL_2026_05_21.md`
  (n=3 multi-seed at 200 steps showing +1.38% standalone wall on CUDA 13.2,
  strict NLL parity).
- **Per-mechanism evidence**: `research/ITER<N>_*.md` for N ∈ {97, 99, 100,
  101, 103, 106, 107, 109, 113, 115, 116} (the iter 116 ship's 10 mechanisms).

**Prior iter 116 ship** (`chiron_1B_T16384_iter116_treatment_phase2.final`,
28,257 tok/s, NLL 4.2039 @ 30k, on CUDA 12.0) is archived at
`database/checkpoints/chiron_1B_T16384_iter116_treatment_phase2/` and remains
loadable into the v5+FP8 flagship (math is bit-identical at single-element
FP32 between iter 116 ship and v5; v5+FP8 adds a bounded FP8 readout drift
of ~0.005-0.007 nat per iter 62 design). The iter 116 ship was the CUDA 12.0
production flagship; v5+FP8 ships the CUDA 13.2 equivalent + FP8 readout.

**Prior iter 94 flagship** (`chiron_1B_T16384_triple_phase2.final`, 25,103
tok/s, NLL 4.1983) is archived at `database/checkpoints/chiron_1B_T16384_triple_phase2/`
and remains loadable. The iter 94 baseline
`chiron_1B_T16384_baseline_phase2.final` (24,246 tok/s, NLL 4.2228) is
the pre-w=4 archive at `database/checkpoints/chiron_1B_T16384_baseline_phase2/`.

The repository also contains a separate **CHIRON-stack research line** under
`run.sh chiron --scale {66M..1.84B}` using FACE + MFIO + WIP + SAS + RLG +
SLC. That is NOT the production flagship — it's the FACE-optimizer +
curriculum-scaling research program. When in doubt, default to the
`flagship` recipe above for new training runs.

## Build Commands

**Build the library** (from project root):
```bash
sh .configure.sh
```

**Build with CUDA**:
```bash
sh .configure.sh cuda
```

**Build and run unit tests**:
```bash
cd unit-tests/build && sh .configure.sh
cd unit-tests/build && sh .configure.sh cuda # compile with cuda
cd unit-tests && bash test.sh nnall    # run all tests
```

**Available single test names**: `nn`, `nn-recurrent`, `nn-transformer`, `transformer-serving` (or `serving`), `nn-bench`, `pca`, `kmeans`, `bayes`, `bayes-optimizer`, `bayes-optimizer-nd`, `ohe`, `mapped`, `cv`, `save-load`, `nn-mixed-precision` (or `nn-mp`), `prop-fuzz`, `parallel`, `ddp`, `transformer-improvements` (or `ti`), `gpu-training`, `cnn`, `cnn-mnist`, `garch`, `egarch`, `gan`, `search-space`, `hp-tuner`, `bayes-lr`, `hp-tuner-full`, `atlas`, `atlas-bench`

**Install**: `cd build && make install` (installs to `~/.local`)

## Project Overview

C++98 machine learning library built as a shared library (`libglades.so`). Depends on the `shmea` library (database/networking, installed separately). Uses CMake 3.10+, Release mode with `-O3 -march=native`.

The unit tests have their own CMake project under `unit-tests/` with a separate `build/` directory. Tests link against the in-tree build of glades (not the installed copy). The CMakeCache in each build dir is sticky — delete it when changing build type or flags.

## Architecture

### Library Targets (built by `Backend/Machine Learning/CMakeLists.txt`)
- **ML** — top-level ML module, links all sub-targets below
- **Networks** — all neural network implementations (forward/backward, training, inference, persistence)
- **MLStructure** — network architecture definitions (`NNInfo`, `LayerInfo` variants)
- **MLState** — training state management (`Terminator`)
- **DataObjects** — data input abstractions (`NumberInput`, `ImageInput`, `TokenInput`, `MappedMatrix`)
- **GMath** — math utilities (`PCA`, `KMeans`, `GARCH`, `OHE`, `CMatrix`)

### Network Types
Defined in `Backend/Machine Learning/Networks/network.h` as `TYPE_DFF`, `TYPE_RNN`, `TYPE_GRU`, `TYPE_LSTM`, `TYPE_TRANSFORMER_ENCODER`, `TYPE_TRANSFORMER_DECODER`, `TYPE_CNN`. GANs are a separate class wrapping these.

Each network type has a dedicated SGD implementation file: `sgd_dff.cpp`, `sgd_rnn.cpp`, `sgd_gru.cpp`, `sgd_lstm.cpp`, `sgd_cnn.cpp`, `sgd_transformer.cpp`.

### Transformer Stack
- `sgd_transformer.cpp` — training (backward passes, gradient accumulation)
- `transformer_infer.cpp` — forward-only inference
- `transformer_generate.cpp` — token generation and sampling
- `transformer_ops.h` — attention kernels, softmax, activation functions (GELU, SiLU, ReLU)
- `transformer_kernels.h` — SIMD-optimized math (`dot_f32`, `axpy_f32`) with scalar fallbacks
- `transformer_ops.h` includes `transformer_kernels.h` for SIMD helpers
- `training_config.h` — transformer configuration (RoPE/sinusoidal, LayerNorm/RMSNorm, MLP/SwiGLU, KV-cache dtypes)
- `transformer_public_api.h` — stable C++98 wrapper for generation APIs

### Public API Entry Point
`Backend/Machine Learning/main.h` defines the `glades` namespace with `train()`, `test()`, `trainOwned()`, `testOwned()` overloads. `glades::init()` must be called first.

### Determinism
The engine is deterministic by default. All randomness goes through `glades::rng::*` with per-network RNG engines controlled by `NNetwork::setSeed()`. See `Backend/Machine Learning/DETERMINISM_AND_CONCURRENCY.md` for the full policy.

### Test Framework
Custom assertion macro `ASSERT(failmsg, predicate)` defined in `unit-tests/unit-test.h`. Each test suite is a standalone function (e.g., `NNUnitTest()`, `PCAUnitTest()`) registered in `unit-tests/main.cpp`. Test source files live under `unit-tests/Backend/Machine Learning/`.
