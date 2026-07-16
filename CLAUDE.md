# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current Production Flagship — CHIRON 1B @ T=16384 (PIED ship 2026-07-03)

The current production LLM flagship is **CHIRON 1B PIED 30k**
(checkpoint `database/checkpoints/chiron_1B_pied_e4/chiron_1B_pied_e4.final`).
**Checkpoint purge 2026-07-04 (owner-directed): ALL other checkpoints were deleted
(~93 GB) — every "remains loadable"/"archived at" claim in the prior-flagship sections
below is now historical record only. The only checkpoints on disk are the PIED flagship
dir and the tiny CHRN interlock-test fixture `test_fixture_nowhisc` (regeneration
command in `scripts/chiron_serving_interlocks.sh`'s commit).**
It is a **perplexity flagship**: **wide 32-batch val NLL 1.1788 / 1.3019 (two 33.5M-token
windows) vs the prior WhiSC-D ship's 1.6390 / 1.8191 at matched windows — Δ −0.46/−0.52
nat, top-1 +10.6/+10.9 points**, at the identical 30k-step/1.97B-token budget and recipe.
Matched 4-batch final-val **1.0259 / acc@1 0.7224 vs whisc30k's 1.3753 / 0.633
(Δ −0.349 nat)**; the gap **widens with training** (−0.16@3k → −0.44@21k) and PIED crossed
the WhiSC-D ship's *final* quality at step ~15k (half the budget).

- **Shape**: identical to the WhiSC-D ship (m=2048, L=24, nH=16, dH=256, V=32000 BPE,
  T=16384; 870.94M params + 384 QK-Norm γ + 24×2048 rot_phi, CHRF bit 1024). **PIED adds
  NO parameters and NO checkpoint state** — the checkpoint is standard WhiSC-D format.
- **The mechanism — PIED (`--inc-dropout 0.1`)**: Phase-Increment Ensemble Dropout, the
  CHIRON-native dropout — a mean-one two-point Bernoulli mask `η ∈ {0, 1/(1−π)}` on the
  SCFA attention increment at the shear commit (`p += sign·η⊙(y_par+y_perp)`), i.i.d. per
  (layer, µstep, token, channel), regenerated from a stateless counter hash (no RNG state;
  exactly invertible in the inverse walk; states never masked; training passes only —
  val/inference never mask). Exact ensemble semantics on the linear p-accumulator; the
  implicit regularizer is a Fisher-weighted increment-energy penalty whose distinctive
  content is **anti-cancellation across depth** (taxes co-adapted cancelling increments;
  complementary to SIRA). Chosen by a 3-candidate design study over WhiSK (whitened
  symplectic kicks) and SFD (spectral gating); LayerDrop's three failure causes are each
  structurally addressed (increments not states; compensation exact by linearity, never
  through reln; per-token masks keep Adam moments at baseline statistics).
- **Stack/recipe**: the WhiSC-D ship recipe **PLUS `--inc-dropout 0.1`**, 30k steps at
  constant lr 3e-4 (accum 4, seed 1337). No finish anneal (still an untried upside).
- **Stability**: 0 grad-skips, 0 NaN over 30k steps / 1.97B tokens. The lineage's step-9001
  hard-batch spike (whisc30k's only anomaly, ‖g‖ 10.8) recurred at the identical batch
  **damped to 4.9** — in-vivo evidence of PIED's conditioning effect (also seen at E3:
  treatment max ‖g‖ 2.958 < baseline 3.211).
- **Perf**: **−1.87% wall vs no-PIED at the same binary** (E4 ran ~25,105 tok/s), after a
  perf pass: fused masked dual_p commit (`chiron_scfa_axpy2_masked_dual_p`, bit-identical
  to the unfused pair) + dual-output dy copy (`chiron_incdrop_scale_copy_dual`) with a
  `register_fast16bf_constant` BF16-RN mirror so the B^T·dy GEMM skips its per-layer
  re-cast. PIED is training-only: inference throughput is unchanged.
- **Verified three ways**: trainer 4-batch final-val 1.0259/0.7224; trainer wide-32
  1.1788/1.3019 (lr=0 + `--whisc-ema 1.0` resume trick); **chiron_infer teacher-forcing
  1.0897 / top1 0.7176** (serving parity, via `runner.sh --flagship`).
- **Serving**: **zero PIED-specific changes** — the checkpoint needs only the WhiSC flags
  (`--whisc-coupling --rot-theta-max 0.07`), which `runner.sh --flagship` injects
  automatically (prefers the PIED checkpoint since 2026-07-03; the `*pied*` path case was
  added to the flag-injection match).
- **Reproduce training**: the WhiSC-D ship command + `--inc-dropout 0.1`:
  `cd ~/dev/glades-trainer && sh run.sh flagship --steps 30000 --accum 4 --lr 3e-4
  --warmup 750 --sira-warmup 250 --zloss-coef 1e-4 --qk-norm --sira-coef 1e-2
  --sira-energy-weight 1.0 --sira-balance-weight 0.25 --sira-action-weight 0.0
  --grad-clip 0.5 --dq-layer-clamp 1.0 --dq-embed-clamp 1.0 --reln-reanchor
  --whisc-coupling --rot-theta-max 0.07 --inc-dropout 0.1 --save-every 6000 --seed 1337`.
- **Run inference**: `cd ~/dev/glades-trainer && sh runner.sh --flagship`.
- **Ship record / evidence**: `research/CHIRON_PIED_E4_GATE_2026_07_03.md` (E4 + wide-32 +
  TF parity); `research/CHIRON_PIED_E3_GATE_2026_07_02.md` (matched-pair stability gate —
  fresh baseline, gap negative from step 1k); `research/CHIRON_PIED_IMPLEMENTATION_2026_07_01.md`
  (E0/E1 + kernels). Design:
  `docs/superpowers/specs/2026-07-01-chiron-pied-increment-dropout-design.md`.
- **Caveats**:
  - **Single-seed (1337) ship — multi-seed (≥3) NOT run (same precedent as the WhiSC-D and
    reanchor ships).** The wide-32 matched-window numbers are the banked ship-metric claim.
  - **Era-drift attribution**: the E4 baseline is the whisc30k run (2026-07-01 binary), not
    a same-binary paired arm; measured drift scale ~0.10–0.15 nat — the margins are 3–5×
    beyond it, and the E3 paired arms bound PIED's own effect from below.
  - **Perplexity flagship, not a generator** — inherited from the lineage (see the WhiSC-D
    section's generation caveat; PIED's anti-repetition conjecture is untested).
  - Open follow-ups: π sweep, `--inc-dropout-symmetric` mechanism-separation arm, per-head
    (B2) / spectral (SFD-G1) variants, multi-seed, 60k+finish schedule.

## Prior WhiSC-D Flagship — CHIRON 1B @ T=16384 (WhiSC-D ship 2026-07-01, kept for context)

The prior flagship **CHIRON 1B WhiSC-D 30k**
(checkpoint `database/checkpoints/chiron_1B_T16384_whisc30k/chiron_1B_T16384_whisc30k.final`).
It is a **perplexity flagship**: **wide 32-batch val NLL ~1.64–1.82 (two 33.5M-token
windows), vs the prior reanchor ship's wide 1.92** — i.e. better than the prior
flagship on its own ship metric at **less than half the training** (30k steps / 1.97B
tok vs 66k+finish / 4.33B). At **matched** step/seed/recipe the gain is dramatic:
**val@30k 1.3753 / acc@1 0.633 vs the no-whisc base's 2.6488 / 0.318 (Δ ≈ −1.27 nat,
top-1 ~doubled; 4-batch windows)** — the largest matched jump in the lineage.

- **Shape**: identical to the reanchor ship — m=2048, L=24, nH=16, dH=256, V=32000
  BPE, T=16384; 870.94M params + 384 QK-Norm γ (bit 256) + **24×2048 `rot_phi`
  coupling angles (CHRF bit 1024, the last checkpoint section)**.
- **The mechanism — WhiSC-D (`--whisc-coupling`)**: a per-layer cross-depth coupling
  — the SORC per-channel rotation **conjugated by a detached per-channel whitening**
  `Φ = W⁻¹R(θ)W`, `W=diag(1/a,a)`, `a=(E[q²]/E[p²])^{1/4}` (EMA, out of the autodiff
  graph). The whitening factors the trained-in `p²/q²≈10³` phase asymmetry (which
  diverged SORC and regressed OBSD) into a detached frame, so the coupling perturbs
  q by O(‖q‖) and the backward q-gradient is bounded independent of ‖p‖/‖q‖.
  Implemented as **coefficient folding** into the SORC 3-shear kernel
  (`A=a²·sorc_a`, `C=sorc_c/a²`; FD-validated a²-aware dθ) — no extra passes.
- **Stack/recipe**: the reanchor-cure recipe (see prior flagship below) **PLUS
  `--whisc-coupling --rot-theta-max 0.07`**, trained 30k steps at constant lr 3e-4
  (accum 4, seed 1337). **No finish anneal** — a 60k base + flat-3e-5 finish (the
  full reanchor schedule) with WhiSC is an open upside, untried.
- **Stability**: 0 grad-skips, 0 NaN over 30k steps; one isolated recovered ‖g‖
  spike (10.8 @step 9001). The `[whisc]` monitor showed ρ_eff→~4100 absorbed by the
  whitening (a→0.125, clamp-binding); maxTheta 0.067 < the 0.07 cap.
- **Perf**: **~25,750 tok/s** (vs ~28,500 no-coupling: **−9.7%**, ×1.11/step) after
  four perf passes (cumulative +43.6% from the first WhiSC binary's 17,930/−37%):
  (1) coefficient folding +24% (eliminated the explicit whiten/unwhiten passes,
  19.1% GPU time); (2) coalesced stats reduction +3.7%; (3) fused da/dc column
  reduction into the rot pre-backward +10.9% (killed ~536 MB/layer-µstep of
  [T×m] scratch traffic, `chiron_rot_fused_backward`); (4) fused inverse-walk
  +1.1% (`chiron_rot_backward_invwalk` — the backward walk + adjoint in one
  pass; grad/dphi bit-exact vs the two-call path). Residual ≈ the inherent
  rotation forward + DRAM-bound stats. **Build gotcha: the trainer links the
  glades CUDA kernels STATICALLY — after any glades-ml kernel change,
  `make install` alone does NOT update the trainer; rebuild it (`bash build.sh`).**
- **Verified three ways**: trainer wide 32-batch val 1.639/1.819; trainer 4-batch
  final-val 1.3753/acc1 0.633; **chiron_infer teacher-forcing 1.4655/top1 0.626**
  (serving parity, see below).
- **Serving (chiron_infer, 2026-07-01)**: WhiSC checkpoints need
  `--whisc-coupling --rot-theta-max 0.07` — **`runner.sh --flagship` injects these
  automatically** (prefers the PIED ship since 2026-07-03; whisc30k is the first
  fallback). Two serving
  parameters are NOT in the checkpoint: the whitening scale (recomputed per-batch at
  inference, ema=1.0) and theta_max (CLI; 0.07 = the flagship recipe value).
  chiron_infer **hard-errors** (exit 7) on a bit-1024 checkpoint without the flags
  (and vice versa), guards dead-gamma_p fuse-mode misdetection, and rejects
  unknown-CHRF-bit checkpoints (the rot_phi EOF-tail read must stay last).
- **Reproduce training**: `cd ~/dev/glades-trainer && sh run.sh flagship --steps
  30000 --accum 4 --lr 3e-4 --warmup 750 --sira-warmup 250 --zloss-coef 1e-4
  --qk-norm --sira-coef 1e-2 --sira-energy-weight 1.0 --sira-balance-weight 0.25
  --sira-action-weight 0.0 --grad-clip 0.5 --dq-layer-clamp 1.0 --dq-embed-clamp 1.0
  --reln-reanchor --whisc-coupling --rot-theta-max 0.07 --save-every 6000 --seed 1337`.
- **Run inference**: load `chiron_1B_T16384_whisc30k.final` explicitly
  (`runner.sh --flagship` prefers the PIED ship since 2026-07-03).
- **Ship record / evidence**: `research/CHIRON_WHISC_D_GATE_2026_06_30.md` (E3
  divergence gate + E4 30k gate + wide-val + perf + serving). Design/plan:
  `docs/superpowers/specs/2026-06-30-chiron-whitened-frame-coupling-design.md`,
  `docs/superpowers/plans/2026-06-30-chiron-whisc-d.md`.
- **Caveats**:
  - **Single-seed (1337) ship — multi-seed (≥3) NOT run (owner decision
    2026-07-01, same precedent as the reanchor ship).** The matched Δ−1.27 (4-batch)
    is window-noisy; the wide 32-batch numbers (1.64–1.82 vs 1.92) are the banked
    ship-metric claim. Cross-seed reproduction is NOT established.
  - **Perplexity flagship, not a generator** — inherited from the whole lineage
    (see the reanchor section's generation caveat; unchanged by WhiSC).
  - The upstream WhiSC upgrades (WhiSC-M Mahalanobis, WhiSC-T thermostat) and the
    60k+finish schedule are untried follow-ups.

## Prior reanchor-cure Flagship — CHIRON 1B @ T=16384 (reanchor ship 2026-06-27, kept for context)

The prior flagship **CHIRON 1B reanchor-cure**
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
- **Run inference**: load `chiron_1B_T16384_reanchor5B_finish.final` explicitly
  (`runner.sh --flagship` prefers the WhiSC-D ship since 2026-07-01). **Serving fix (2026-06-27)**:
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

**Update (2026-06-30) — SORC orthogonal-rotation coupling: CLOSED NO-GO
(diverges at scale).** The natural successor to OBSD: replace the unbounded
ReZero *additive* drift with a **norm-preserving per-channel symplectic rotation**
`(q,p) → R(θ)(q,p)`, `θ = θ_max·tanh(φ)` (hard-bounded angle, realized as 3
reversible shears; `R∈SO(2)`, `‖R‖₂=1`). The hypothesis was that OBSD's failure
was the *unbounded gate*, so a conservation-bounded coupling would be safe. It is
**implemented correctly** (CPU-ref backward = exact orthogonal Rᵀ verified by hand;
trainer wiring reviewer-confirmed; E0 bit-identical at φ=0; E1 8/8 unit tests; E2
small-shape reconstruction clean; the angle bound provably holds), **but the E3
production gate DIVERGED**: matched single-seed 2500-step run at T=16384, the
treatment tracks baseline to step ~748 then **explodes at step ~831** (val@2500
**14.45** vs baseline **3.58**, 1619 grad-skips, ‖g‖→4.7e10, loss-scale→0). The
exploding gradient is **exclusively `drot_phi`** in a clean ~2.5×/layer geometric
cascade across depth. **Root cause is fundamental, not a bug**: the block runs with
**`‖p‖ ≈ 17×‖q‖`** (reln normalizes q each layer; p is un-normalized and accumulates
attention across depth, `mean(p²)/mean(q²)≈300`), so the rotation's `q += a·p` is a
*p-scaled* perturbation to the ~17× smaller q. The `‖R‖₂=1` conservation preserves
the **joint (q,p) norm — which is p-DOMINATED** — and therefore does **not** protect
the q subspace; positive feedback makes p run away (`mean(p²)` jumps ×213 at the
explosion) and the backward/`drot_phi` blow up. **Norm-preservation of the forward
map does not bound the backward gradient when the phase space is scale-asymmetric.**
So **both** additive (OBSD) and rotational (SORC) per-layer symplectic coupling are
now NO-GO, for the **same root-cause family** (the block's p/q asymmetry) — OBSD
stable-but-regresses, SORC unstable-diverges. A future cross-depth lever must act in
a **scale-normalized / whitened (q,p) frame** (bound the q-perturbation by `‖q‖`,
not `‖p‖`); joint-norm conservation is the wrong invariant. The cheap E3 gate caught
this in ~3.4 GPU-hr (E4 not run). Engineering committed **default-off** and reusable
(kernels `chiron_rot_coeffs` / `chiron_rot_forward` [fwd+inv] / `chiron_rot_backward`
+ dphi chain; CPU refs; 8 unit tests `test.sh chiron-rot`; `rot_phi[l]` param +
checkpoint bit 1024; `[sorc]` plateau monitor; flags `--rot-coupling` /
`--rot-theta-max` / `--rot-warmup`). Full record:
`research/CHIRON_SORC_RESULT_2026_06_30.md`; design/plan
`docs/superpowers/specs/2026-06-30-chiron-sorc-symplectic-rotation-design.md` +
`docs/superpowers/plans/2026-06-30-chiron-sorc-symplectic-rotation.md`.

**Update (2026-06-30) — WhiSC-D whitened coupling: E3 GATE PASS (fixes SORC; not yet
shipped).** The successor that *cures* both prior failures by attacking their shared root cause
(the p/q scale asymmetry, measured this session: `p²/q²` is trained-in within ~100 steps,
reaching ~2000+ — `research/CHIRON_INIT_PQ_RATIO` / memory `chiron_init_pq_ratio_measured`).
WhiSC-D conjugates the SORC rotation by a **detached per-channel whitening kept out of the
autodiff graph**: `Φ = W⁻¹R(θ)W`, `W=diag(1/a,a)`, `a=(E[q²]/E[p²])^{1/4}` (per-channel EMA),
so the differentiated signal is `O(‖q‖)` and the backward q-gradient is bounded independent of
`‖p‖/‖q‖` (det=1 symplectic; φ=0⇒identity/E0). On the **same matched 2500-step T=16384 gate that
SORC diverged**, WhiSC-D **completed cleanly**: ‖g‖ O(1) (max 3.54), 0 grad-skips, 0 NaN,
val@2500 **2.78** — vs SORC's 14.45 / ‖g‖ 4.7e10 / 1619 skips / explosion at step ~831. The
`[whisc]` monitor confirms `ρ_eff` grew to **~3000** (past the SORC-kill regime) and the
whitening absorbed it; the R1 bound holds in vivo. Perplexity is a promising bonus (matched
this-session base val@1000 4.21 vs 4.04 = −0.17 nat; vs prior SORC-base val@2500 3.58 vs 2.78 =
−0.80 nat, widening) but wants a matched-2500 / **E4 (30k, not yet run)** to bank. **Perf
addressed post-E3:** profiling corrected the bottleneck — the explicit `whisc_scale`
whiten/unwhiten passes (19.1% GPU time), NOT the stats kernel. **Coefficient folding** (`A=a²·sorc_a,
C=sorc_c/a²`, FD-validated `a²`-aware `dθ`) eliminated them → **+24% throughput, −37%→−22%**
(committed `c83303e9c`/`6e429d6`, E0 identity preserved); a stats-kernel coalesce (`18be3228d`) was
correct but overlap-hidden (~0 end-to-end), residual −22% = the inherent rotation kernels. **E4
(30k) — PASS (2026-07-01):** WhiSC-30k val@30000 **1.3753 / acc@1 0.633** vs the matched no-whisc
baseline (reanchor flagship *base*, this exact recipe minus whisc, seed 1337, step-1 bit-identical)
val@30000 **2.6488 / acc@1 0.318** → **Δ ≈ −1.27 nat, top-1 ~doubled** (best-train 0.995 vs 1.847);
robust lower bound −0.54 nat vs the *fully-trained* flagship (1.92@66k). 0 grad-skips, 1 isolated
recovered ‖g‖ spike. **Caveat: single-seed / 4-batch-window — the ~−1.0 to −1.27 nat magnitude is
the LARGEST in the lineage, so it needs multi-seed (≥3) + wide-val (32-batch) confirmation before a
ship.** (No baseline re-run — reused the reanchor base from the training logs.)
Engineering is **committed default-off**, fully reviewed (merge-ready), reusing the SORC
infrastructure (`rot_phi`/Adam/checkpoint-bit-1024/`[sorc]` monitor) + new kernels
`chiron_whisc_scale` / `chiron_whisc_update_stats` + flags `--whisc-coupling` / `--whisc-ema` /
`--whisc-clamp`. So per-layer cross-depth coupling for CHIRON is no longer a dead end: OBSD
stable-but-regressed, SORC unstable-diverged, **WhiSC-D stable AND (provisionally)
improves** — the lever just had to act in a scale-normalized (whitened) frame, from step 1. Full
record: `research/CHIRON_WHISC_D_GATE_2026_06_30.md`; design/plan
`docs/superpowers/specs/2026-06-30-chiron-whitened-frame-coupling-design.md` +
`docs/superpowers/plans/2026-06-30-chiron-whisc-d.md`.

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

**Available single test names**: `nn`, `nn-recurrent`, `nn-transformer`, `transformer-serving` (or `serving`), `nn-bench`, `pca`, `kmeans`, `bayes`, `bayes-optimizer`, `bayes-optimizer-nd`, `ohe`, `mapped`, `cv`, `save-load`, `nn-mixed-precision` (or `nn-mp`), `prop-fuzz`, `parallel`, `ddp`, `transformer-improvements` (or `ti`), `gpu-training`, `cnn`, `cnn-mnist`, `garch`, `egarch`, `gan`, `search-space`, `hp-tuner`, `bayes-lr`, `hp-tuner-full`, `atlas`, `atlas-bench`, `chiron-model`, `chiron`, `chiron-rot`, `chiron-whisc`, `chiron-pied`, `chiron-pact`, `chiron-echo` (or `chiron-echo-cpu` for the GPU-free subset), `chiron-orbit`, `chiron-orbit-bench`, `chiron-generate`

**Install**: `cd build && make install` (installs to `~/.local`; also installs the ML header tree to `~/.local/include/glades/Backend/Machine Learning/` — required by the trainer since the vendored `include/` was removed)

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

### CHIRON Serving Modules (2026-07-03)

Glades-ml owns CHIRON checkpoint I/O and serving since 2026-07-03:

- `chiron_checkpoint.{h,cpp}` — CHRN/CHRF format single source of truth: `ChironCkptBits`
  registry, block codecs, serving reader (rot_phi/a_drift EOF-tail reads), model-section
  writers. Section-order contract: a_drift immediately before rot_phi, rot_phi LAST.
- `chiron_serving.{h,cpp}` — `chiron_resolve_serving` decision table (WhiSC interlocks,
  fuse/QK-Norm/SCFA auto-enables, dead-gamma_p guard; CHRF bit 512 / a_drift refused only
  when trained/nonzero — zero-valued co-allocation is served with an info line) +
  `ChironEvalScratch` / `chiron_eval_forward` (kernel-sequence-identical to old forwardInfer).
- **Unit suite**: `bash test.sh chiron-model` (roundtrips, resolve table, eval-forward parity).
- `chiron_generate.{h,cpp}` — all CHIRON generation and evaluation logic (2026-07-03):
  `ChironMt19937` (C++98 MT19937 + libstdc++-13.3.0-compatible canonical double —
  **TOOLCHAIN-COUPLED**, pinned by golden-stream unit tests; do not change the RNG
  implementation without re-pinning goldens), `chiron_sample_token` (temperature/top-k/top-p
  + repetition penalty), `chiron_generate` (window-slide loop, sink streaming),
  `chiron_tf_eval` (**single source of truth for TF-NLL** — used by both chiron_parity and
  chiron_infer; do not duplicate this logic), `chiron_degeneration_metrics`.
  Per-call RNG seeding: the pre-2026-07-03 CLI used one process-level mt19937 shared
  across REPL prompts; one-shot/tokens-file paths are identical, multi-prompt REPL streams differ
  (acknowledged behavior change).
- **Unit suite**: `bash test.sh chiron-generate` (615+ asserts: RNG/sampler goldens, stochastic
  draw-parity, TF-eval correctness).

**Rebuild order** (static kernel link — `make install` alone does NOT update the trainer):
```
cd ~/dev/glades-ml/build && make install
cd ~/dev/glades-trainer && bash build.sh
```

**One-time shmea header setup** (shmea never installs headers):
```
cp -r ~/dev/ShmeaDB/Backend/{Database,Networking,Plotter} ~/.local/include/Backend/
```

### Public API Entry Point
`Backend/Machine Learning/main.h` defines the `glades` namespace with `train()`, `test()`, `trainOwned()`, `testOwned()` overloads. `glades::init()` must be called first.

### Determinism
The engine is deterministic by default. All randomness goes through `glades::rng::*` with per-network RNG engines controlled by `NNetwork::setSeed()`. See `Backend/Machine Learning/DETERMINISM_AND_CONCURRENCY.md` for the full policy.

### Test Framework
Custom assertion macro `ASSERT(failmsg, predicate)` defined in `unit-tests/unit-test.h`. Each test suite is a standalone function (e.g., `NNUnitTest()`, `PCAUnitTest()`) registered in `unit-tests/main.cpp`. Test source files live under `unit-tests/Backend/Machine Learning/`.
