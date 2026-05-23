# CHIRON 1B UL2 (Mixture-of-Denoisers) — Design

**Date:** 2026-05-23
**Status:** Pre-registered design (no code yet)
**Program scope:** Phase-3 of CHIRON 1B production line. Adds the
**objective-side** regularization arc — UL2 mixture-of-denoisers
(Tay et al. 2022) — after the depth-noise arc (LayerDrop)
closed FAIL on 2026-05-23.
**Baseline:** `chiron_1B_T16384_regstack_phase2.final` (regstack
Phase 2 ship 2026-05-22). Final val NLL 3.5734 @ step 30000,
throughput 28,072 tok/s, VRAM 14.97 / 15.56 GB on RTX-4080-SUPER.
**Output target:** A 30k Phase-2 production retrain at regstack
Phase 2 + UL2 mixture, gated on a 5k Phase-1 pilot arc.

---

## 1. Motivation

The LayerDrop arc (`docs/superpowers/specs/2026-05-23-chiron-1b-layerdrop-design.md`)
closed FAIL on 2026-05-23 — see
`research/LAYERDROP_5K_FAIL_2026_05_23.md`. Headline: hard-drop
LayerDrop (mandated by CHIRON's symplectic `(p, q) → (p + shear(q),
reln(q))` structure) at `p_max=0.1` 5k single-seed regressed val NLL
by +0.073 nat above the +0.05 fail bar despite +4.10% wall
improvement. Depth-noise architecture-side regularization is
empirically ruled out for CHIRON at this scale, at least in the
hard-drop convention.

The next mechanism class is **objective-side** regularization —
changing what the model is trained to predict, not how the layers
compute. UL2 (Unifying Language Learning Paradigms, Tay et al.
2022) is the strongest external prior for objective-side gains at
1B LLM scale:

- T5/PaLM-derived: validated at multiple model sizes, multiple
  pretraining datasets.
- Decoder-only adaptations (PaLM-2 reportedly uses; Olmo, MPT,
  etc.) report consistent NLL improvements at 1B-10B.
- Reported gains: −0.10 to −0.20 nat val NLL when mixed with
  standard causal LM.

CHIRON 1B has a natural foundation for UL2 already in place —
**paradigm #262 MEDAL** (`research/PARADIGM_SHIFT_262_MEDAL_DESIGN.md`,
production code at `glades-trainer/trainer/chiron_main.cpp:2480+,
5372+, 8651+`). MEDAL is a **single-denoiser** absorbing-mask
diffusion LM training mode. It has:

- `V+1` vocab extension (the MASK token at row V, zero-initialized).
- GPU embedding gather over the extended vocab.
- `medal_corrupt_tokens` GPU kernel that replaces input tokens with
  MASK at sampled positions (uniform α-fraction).
- `medal_mask_dlogits` kernel that zeros gradients at
  non-corrupted positions (so only masked positions contribute
  to loss).
- Sinusoidal α-time embedding (when sampled α varies per step).
- `tgtBuf = inBuf` mode in `fillWindow` (same-position prediction
  vs the default next-token shift).

UL2 extends this single-denoiser foundation to a **mixture-of-3**:

1. **R (Regular)** — short spans (Poisson μ=3), low corruption
   rate (~15%). Standard masked-LM regime.
2. **S (Sequential)** — pure causal LM (current ship recipe).
   No corruption; shifted-target loss as in the regstack ship.
3. **X (eXtreme)** — long spans (Poisson μ=32), high corruption
   rate (~50%). Aggressive denoising; forces longer-context
   reasoning.

Each training step rolls a denoiser type uniformly from {R, S, X}.
The mixture is sampled **per training step** (CHIRON's batch=1 with
T=16384 makes per-step and per-sample equivalent).

**Why this fits CHIRON 1B specifically:**

1. The MEDAL infrastructure is already production-validated —
   builds on a known-good code path rather than a new architecture.
2. The S-denoiser is the current ship's training recipe verbatim,
   so the marginal mechanism is just R/X denoisers added on top.
3. Same-position prediction layout (vs T5-authentic packed-sequence)
   keeps the data pipeline unchanged: T input tokens stream from
   pretok as-is; per-step the trainer corrupts some positions and
   re-routes the target setup. No sample-boundary tracking needed.
4. The decoder-only architecture sees all three denoisers via the
   SAME causal attention mask — no encoder-decoder split needed.
5. UL2 paper's defaults (μ_R=3, p_R=0.15, μ_X=32, p_X=0.50) are
   tuned for T5 base/large and translate naturally to 1B scale per
   the literature.

The mechanism is **architecturally different from LayerDrop's failure mode**:
LayerDrop dropped layer-level compute (a depth-noise mechanism that fought
CHIRON's symplectic momentum accumulator). UL2 keeps the full
forward+backward pass intact and varies only the training objective.
There is no analogous "symplectic incompatibility" risk; the symplectic
update is identical regardless of denoiser type — only the loss target
and corruption pattern of the *input* change.

---

## 2. Architecture

### 2.1 Three denoisers, same-position layout

Per training step, draw `denoiser ∈ {R, S, X}` independently and
uniformly (probability 1/3 each) using a dedicated RNG engine
`s.ul2_rng` (separate from `rngEngine` to avoid perturbing the
LayerDrop/MEDAL RNG paths).

| Denoiser | μ_span | p_corruption | Target | Loss positions |
|---|---:|---:|---|---|
| R (Regular) | 3 | 0.15 | same-position | corrupted positions only |
| S (Sequential) | — | 0.00 | shifted (`tgtBuf[i] = inBuf[i+1]`) | all non-pad positions |
| X (eXtreme) | 32 | 0.50 | same-position | corrupted positions only |

### 2.2 Span sampling

For denoisers R and X, sample a per-position bit-mask
`mask ∈ {0, 1}^T` indicating which positions are corrupted.
Algorithm:

```
ul2_sample_span_mask(eng, T, p_target, mu, mask_out):
    fill mask_out with 0
    target_corrupted = floor(p_target * T)
    corrupted_so_far = 0
    attempts = 0
    while corrupted_so_far < target_corrupted AND attempts < (T / 2):
        attempts += 1
        span_len = max(1, Poisson(eng, mu))     // min length 1
        start = uniform_int(eng, 0, T - 1)
        end = min(start + span_len, T)
        for i in [start, end):
            if mask_out[i] == 0:
                mask_out[i] = 1
                corrupted_so_far += 1
                if corrupted_so_far >= target_corrupted:
                    break
        if corrupted_so_far >= target_corrupted:
            break
    return corrupted_so_far
```

Notes:
- The bound `attempts < T / 2` prevents pathological infinite
  loops when overlapping spans saturate before reaching
  `target_corrupted` (rare for `p < 0.5`).
- Min span length 1 ensures Poisson zero-draws still place a
  one-token span.
- Sampling is CPU-side, deterministic from `eng`. Mask is
  uploaded to GPU per step (T bytes ≈ 16 KB at T=16384, dwarfed
  by per-step GEMM traffic).

### 2.3 Per-step denoiser switch (training fwd)

```cpp
// At top of training fwd loop body, before transformer call:
const int denoiser = pick_denoiser(s.ul2_rng);   // 0=R, 1=S, 2=X

if (denoiser == 1)
{
    // S-denoiser: pure causal LM, current ship recipe.
    // tgtBuf already shifted by fillWindow.  No corruption.
    // No mask buffer setup.  Proceed identically to regstack.
}
else
{
    // R or X: same-position prediction, span-corrupted input.
    const float p_target = (denoiser == 0) ? cfg.ul2_R_rate : cfg.ul2_X_rate;
    const int   mu       = (denoiser == 0) ? cfg.ul2_R_mu   : cfg.ul2_X_mu;

    // Override target: tgtBuf := inBuf (same-position).
    std::memcpy(tgtBufHost.data(), inBufHost.data(), sizeof(int) * T);

    // Sample span mask on host.
    glades::transformer_kernels::ul2_sample_span_mask(
        s.ul2_rng, T, p_target, mu, ul2_mask_host.data());

    // Upload mask to GPU buffer s.d_medal_mask (reuse MEDAL's buffer).
    s.d_medal_mask.upload(ul2_mask_host.data(), T);

    // Corrupt input tokens at mask=1 positions (reuse MEDAL kernel).
    glades::gpu::medal_corrupt_tokens(s.d_tokens.data(),
                                      s.d_medal_mask.data(),
                                      T, cfg.V,  // MASK token id = V
                                      s.d_tokens_corr.data());

    // forward() then sees corrupted tokens + the mask is consumed by
    // medal_mask_dlogits in backward to zero non-corrupted-position
    // gradients.
}
```

### 2.4 Eval mode (forced-S)

UL2 is gated on the `isTraining` parameter of `forward()` (added
in commit `5d1aa1c` for the LayerDrop val-mode fix). At
`isTraining=false`, the denoiser switch above is skipped; the val
pass runs the pure causal LM path (denoiser=S) regardless of mixture
config. This makes val NLL apples-to-apples with the regstack ship.

The trainer's existing `run_validation()` already passes
`isTraining=false` to `forward()` (LayerDrop fix landed it). No new
wiring needed.

### 2.5 Implementation sites

| File | Change |
|---|---|
| `glades-ml/Backend/Machine Learning/Networks/transformer_kernels.h` | Add `inline void ul2_sample_span_mask(eng, T, p, mu, mask_out)` helper (CPU). |
| `glades-trainer/trainer/chiron_main.cpp` Config struct | Add `bool ul2Enabled` (default false), `int ul2_R_mu=3`, `float ul2_R_rate=0.15f`, `int ul2_X_mu=32`, `float ul2_X_rate=0.50f`. |
| `glades-trainer/trainer/chiron_main.cpp` Scratch struct | Add `glades::rng::Engine ul2_rng` (seeded from `cfg.seed ^ <ul2-specific xor>` in `allocate()`), `std::vector<unsigned char> ul2_mask_host` (size T). |
| `glades-trainer/trainer/chiron_main.cpp` training loop | Per-step denoiser roll + branch (§2.3). |
| `glades-trainer/trainer/chiron_main.cpp` CLI parsing | Add `--ul2-enabled`, `--ul2-r-mu`, `--ul2-r-rate`, `--ul2-x-mu`, `--ul2-x-rate` flags. |
| `glades-trainer/run.sh` | Help text + flag pass-through alongside `--medal-train`. |
| `glades-ml/unit-tests/.../chiron-test.cpp` | New tests: span sampler mean/rate, disabled-parity, RNG determinism. |

**Reuse from MEDAL** (no new code): `V+1` vocab extension, MASK
token init at row V, GPU embedding gather, `medal_corrupt_tokens`
GPU kernel, `medal_mask_dlogits` GPU kernel, `d_medal_mask` and
`d_tokens_corr` GPU buffers, sinusoidal α-time embedding (when
applicable — UL2 doesn't strictly need it since denoiser type is
not α-parameterized, but we may surface it as an optional input).

**Code budget:** ~300-400 LOC total — span sampler (~50 LOC),
per-step denoiser switch + config wiring in chiron_main (~150 LOC),
CLI flags + run.sh (~30 LOC), unit tests (~80 LOC). Smaller than
the original 500-800 LOC estimate because MEDAL provides the
infrastructure.

### 2.6 Interactions with existing mechanisms

- **Z-loss** (`zlossCoef=1e-4` in ship recipe): unaffected — Z-loss
  is computed on the FP32 logits before FP8 quantization, regardless
  of which positions are non-corrupted. The Z-loss term is bounded
  by `mean_t(log²Z_t)` over all T positions, including masked ones.
  Verify Z-loss magnitude stays bounded at U1 (R-UL2-5).
- **QK-Norm** (`qkNormEnabled=true` in ship recipe): unaffected —
  QK-Norm operates on attention Q/K, independent of input
  corruption pattern.
- **FP8 readout** (`--fp8-readout-fwd`): unaffected — FP8 path
  operates on the logits matrix uniformly. The MASK token's
  initially-zero embedding row produces near-zero logits at masked
  input positions, which is well within FP8 dynamic range.
- **SCFA** (inner+outer attention): unaffected — input corruption
  affects only token embeddings; SCFA's compressed Q/K and
  depthwise conv operate on whatever embedding the model produces.
- **LayerDrop** (`--layer-drop-pmax`, default 0.0): unaffected at
  default; if explicitly enabled, the two mechanisms compose
  (LayerDrop drops some layers; UL2 mixes objectives) but this
  combination is **out of scope** for the UL2 arc (no claim to
  validate; the LayerDrop arc closed FAIL).

### 2.7 Strict bit-identicality at `ul2Enabled=false`

When `cfg.ul2Enabled == false`:
- No `s.ul2_rng` seeding (or seeded but unused; engine state has no
  observable effect).
- No `ul2_mask_host` allocation.
- Training loop skips the entire denoiser-switch block; runs the
  current regstack Phase 2 ship code path verbatim.
- Trainer behavior identical to running the regstack ship recipe.

Guard mechanism: a single `if (cfg.ul2Enabled)` block at the
appropriate point in the training-loop body wraps all UL2 code.

---

## 3. Validation arc

### 3.1 Run plan

| ID | Steps | Seed | Config | Compute |
|---|---:|---:|---|---:|
| U0 | 5000 | 1337 | regstack Phase 2 ship (current flagship) | ~47 min |
| U1 | 5000 | 1337 | U0 + `--ul2-enabled` (mixture R/S/X at paper defaults) | ~50-55 min |
| **U2** | **30000** | **1337** | **U0 + `--ul2-enabled`** (if U1 PASSes) | **~5.0-5.5 h** |

Total compute on the gated path: ~6 hours wall (pilot ~100 min +
30k Phase-2 ~5.2 h). Single-seed (1337) per the regstack /
LayerDrop / iter-93/94 Phase-1/Phase-2 precedent.

### 3.2 Gate criteria (pre-committed, no silent re-targeting)

**U0 sanity:** the re-run baseline U0 must reproduce the published
regstack Phase 2 5k trajectory to within ±0.05 nat at step 5000.
The LayerDrop arc's L0 demonstrated this baseline reproduces to
+0.0004 nat — same-day variance is well within bound. If U0
deviates >0.05 nat, F-UL2-1 fires; investigate non-determinism
before continuing.

**Pilot 5k gate (U1 vs U0):**

The gate uses **forced-S val NLL** (the trainer's `run_validation`
already passes `isTraining=false` per LayerDrop fix `5d1aa1c`, which
short-circuits the UL2 denoiser switch to S — so val is pure causal
LM, apples-to-apples with the regstack ship's val).

- Main NLL: `U1_S_NLL ≤ U0_NLL + 0.02 nat` → PASS.
- If `0.02 < U1_S_NLL - U0_NLL ≤ 0.05`: BORDERLINE → run seed=1338
  disambiguation pilot before deciding (cost ~50 min).
- If `U1_S_NLL - U0_NLL > 0.05`: OUTRIGHT FAIL → publish negative
  result; close arc; do not run U2. (Mirrors LayerDrop §3.2.)
- Throughput: `U1_TOKS ≥ U0_TOKS · 0.90` (≤10% wall regression
  budget — looser than LayerDrop's ≤5% because X-denoiser's higher
  corruption rate plus extra mask-upload + corruption kernel can
  cost wall).
- VRAM: ≤ 15.72 GB peak (1% above the ship ceiling).

Also report **mixture val NLL** as a secondary metric — val on
random R/S/X samples instead of forced-S. This shows whether the
model can ACTUALLY handle the mixture during eval (a non-trivial
claim — the model must not have catastrophically forgotten denoising
or causal-LM). Not a gate, just a diagnostic.

**30k Phase-2 gate (U2 vs regstack ship):**

- **Forced-S val NLL @ step 30000:** ≤ 3.5534 (+0.02 nat improvement
  vs ship 3.5734). Strictest spec target.
- **Throughput:** ≥ 25,265 tok/s (≤10% wall regression vs ship's
  28,072; expected ~24,000-26,000 tok/s given X-denoiser overhead).
- **VRAM:** ≤ 15.72 GB.
- **Position-stratified eval @ step 30000:** report main-head val
  NLL bucketed by position [0, 4k), [4k, 8k), [8k, 12k), [12k, 16k]
  on forced-S samples. Mirrors LayerDrop's check — should improve
  or hold steady across all buckets if UL2 is helping.
- **Mixture val NLL @ 30k:** report alongside forced-S as
  diagnostic.

**PASS handling:** archive current regstack Phase 2 ship at
`database/checkpoints/chiron_1B_T16384_regstack_phase2/`. New ship
`chiron_1B_T16384_regstack_ul2_phase2.final`. Update CLAUDE.md
"Current Production Flagship" block with NLL / throughput / VRAM
numbers and the reproduce command (`sh run.sh flagship
--zloss-coef 1e-4 --qk-norm --ul2-enabled`). Write
`research/UL2_PHASE2_PASS_2026_MM_DD.md` with trajectory + position-
stratified evidence + mixture-eval diagnostic.

**FAIL handling:** publish honest negative result at
`research/UL2_PHASE2_FAIL_2026_MM_DD.md`; UL2 code stays in the
trainer as opt-in (default off, bit-identical when off); close arc.

### 3.3 Production retrain handoff

If U2 PASSes:
- Archive: `database/checkpoints/chiron_1B_T16384_regstack_phase2/`
  → kept loadable (both the regstack stack at `--ul2-enabled=false`
  and the new UL2 stack are bit-identical-loadable into the trainer).
- New ship checkpoint:
  `chiron_1B_T16384_regstack_ul2_phase2.final`.
- Update `CLAUDE.md` "Current Production Flagship" block with the
  new reproduce command:
  `sh run.sh flagship --zloss-coef 1e-4 --qk-norm --ul2-enabled`.
- Update `glades-trainer/runner.sh` `--flagship` to point to the
  new checkpoint (at inference, UL2 is off via `isTraining=false`,
  so same runner works against either ship without flag changes).
- Write `research/UL2_PHASE2_PASS_2026_MM_DD.md` with trajectory
  table + per-denoiser loss decomposition (R/S/X contribution to
  total training loss over the run) + position-stratified val NLL.
- Memory entry: UL2 pilot + Phase-2 result.

---

## 4. Risks (pre-registered)

| Risk | Probability | Mitigation |
|---|---|---|
| **R-UL2-1:** Mixture-training has 3× larger per-step noise (each step is one of three different objectives), making 5k single-seed signal harder to detect than LayerDrop's. | medium | Same disambiguation pathway as LayerDrop: if borderline at seed=1337, run seed=1338 (~50 min). EMA over the 5k run smooths most variance. |
| **R-UL2-2:** Span sampler determinism break — the `ul2_rng` engine may be perturbed by val passes or other RNG-consuming code paths. | low | Use a dedicated `s.ul2_rng` engine seeded from `cfg.seed ^ 0xDEADBEEFCAFEBABEULL`; gate sampling on `isTraining=true` (the fix from commit `5d1aa1c` already in trainer). Covered by `CHIRONUL2DeterministicMasksTest`. |
| **R-UL2-3:** MASK token embedding init (row V, zero-initialized per MEDAL) may need re-tuning for the higher mask density of X-denoiser (50% of positions are MASK vs MEDAL's ~50% but uniform). | low | Use same zero-init as MEDAL (`chiron_main.cpp:2531-2533`). If U1 shows instability that traces to the MASK row, switch to small-Gaussian init (σ=0.02) matching the rest of the embedding table. |
| **R-UL2-4:** X-denoiser at p=0.50 corruption + loss-masked-to-corrupted gives a very-different-loss-landscape signal that may destabilize Adam/FACE optimizer state shared with S-denoiser steps. | medium | Monitor optimizer state norms (Adam v EMA, FACE row/col EMAs) across denoiser types at steps 1000/2500/5000 of U1. If observable per-denoiser dispersion exceeds 2× the baseline regstack EMA noise floor, consider lowering X mixture probability (e.g., R 0.40 / S 0.40 / X 0.20) or reducing p_X to 0.30. Decision lever: if R-UL2-4 fires, do not silently re-target — publish the per-denoiser optimizer-state evidence and adapt explicitly. |
| **R-UL2-5:** FP8 readout interaction — X-denoiser's high corruption rate means many positions get MASK tokens, which have untrained embeddings initially; pre-FP8-quantization logits at MASK positions may have unusual distribution and degrade FP8 calibration. | low | Z-loss (already in ship recipe at `--zloss-coef 1e-4`) bounds the logit magnitudes regardless of denoiser. Verify the Z-loss term magnitude stays bounded across denoiser types at U1 step 1000/2500/5000. If Z-loss term diverges (>2× ship baseline), flag — may need to gate FP8 readout on `denoiser != X` for the first few thousand steps. |
| **R-UL2-6:** Throughput regression — X-denoiser's denser MASK pattern may not save compute (still full attention); R/X both add an extra mask buffer upload + corruption kernel + the `medal_mask_dlogits` gradient zero-out in backward. | medium | Spec budget loosened to ≤10% wall regression. If empirical regression exceeds 10%, flag — may need to either reduce X probability or fuse the mask-upload + corruption kernel into one launch. |
| **R-UL2-7:** Train/eval val-mode bug — UL2 corruption active during val (would corrupt val NLL just like the LayerDrop val-mode bug did). | low | Already fixed in commit `5d1aa1c` for LayerDrop; UL2 reuses the `isTraining` param. Add explicit test: at val, no mask buffer is populated, denoiser forced to S. Covered by `CHIRONUL2DisabledParityTest` AND the val-NLL smoke check at U1 step 1 (should be bit-identical between `--ul2-enabled=true` and `false` for the step-1 pre-train val). |
| **R-UL2-8:** Pile-pretokenized data has document boundaries embedded in the token stream; UL2 X-denoiser's long spans (μ=32) may cross document boundaries, masking semantically unrelated tokens together. | low-medium | Same constraint as current ship — the pretok pipeline already determines document boundaries. The X-denoiser's effective "long context across documents" pattern may actually be a regularizer benefit (forces the model to denoise from arbitrary distant context, which is exactly the long-context objective). If X-denoiser specifically degrades while R-denoiser holds, flag for follow-up — possibly add a "no-cross-boundary" sampling mode. |
| **R-UL2-9:** Forced-S val NLL is the gate, but the model only sees S-denoiser samples ~1/3 of training time; val NLL may be unfair vs the regstack ship's 100%-S training. | known feature | This is intentional — the apples-to-apples comparison asks: does UL2's mixture training make the same val NLL achievable with 1/3 the causal-LM exposure? Spec target: U2 val NLL ≤ ship - 0.02 nat DESPITE the 3× reduced S exposure, demonstrating mixture training is data-efficient. If forced-S val NLL is barely worse than ship (e.g., +0.001), that's a positive result (UL2 didn't hurt causal LM at all). If much worse (>0.05), UL2 is hurting causal-LM performance — close arc. |
| **R-UL2-10:** Span-sampler edge case — Poisson(μ=32) at p_target=0.50 with T=16384 may saturate the `attempts < T/2` bound before reaching `target_corrupted = 8192` positions, leaving the X-denoiser running at lower-than-spec p. | low | Verify empirically in unit test (`CHIRONUL2SpanSamplerRateTest`): at μ=32, p=0.50, T=16384, the actual corruption rate should converge to within ±0.03 of p_target over n=100 trials. If saturation is a real concern, fall back to drawing spans of fixed length μ instead of Poisson. |

---

## 5. Out of scope for this spec

- **Packed-sequence prefix-LM variant** (T5-authentic UL2 with
  `<extra_id_X>` sentinels and explicit input/output split):
  deferred indefinitely; same-position chosen for engineering
  scope and MEDAL-reuse.
- **Paradigm signal** (sentinel tokens or embeddings signaling
  R/S/X): deferred; model infers from corruption pattern. If U1
  shows the model can't disambiguate denoisers, follow-up arc
  could add paradigm tokens.
- **Sweep over mixture probabilities** (R/S/X weighting): uniform
  1/3 chosen per UL2 paper default; sweep deferred to follow-up.
- **Sweep over span params** (μ_R, p_R, μ_X, p_X): paper defaults
  chosen; sweep deferred.
- **More than 3 denoisers** (e.g., bidirectional, causal-mask
  variants): out of scope.
- **Downstream eval** (LAMBADA, MQAR, etc.): per Phase-3 P10,
  position-stratified val NLL on `pretok-data/` validation split
  is the eval methodology.
- **MEDAL-only re-pilot or comparison run** — would have been
  useful but unrelated to UL2's mixture mechanism. Deferred.
- **Multi-seed validation of pilot or Phase-2** — single-seed per
  the regstack/LayerDrop precedent; multi-seed deferred to a
  separate N=3 confirmation arc after U2 PASS.
- **LayerDrop + UL2 composition** — out of scope; LayerDrop closed
  FAIL.

---

## 6. Deliverables checklist

- [ ] Pre-register this spec in memory (key: `chiron_1b_ul2_prereg`).
- [ ] Implement `ul2_sample_span_mask` helper in `transformer_kernels.h`.
- [ ] Add config fields + scratch members + CLI flags in chiron_main.cpp.
- [ ] Implement per-step denoiser switch in training fwd loop.
- [ ] Add 3 unit tests:
  - `CHIRONUL2SpanSamplerMeanSpanTest` (mean span length ≈ μ).
  - `CHIRONUL2SpanSamplerRateTest` (corruption rate ≈ p_target ± 0.03 over n=100).
  - `CHIRONUL2DisabledParityTest` (`--ul2-enabled=false` is bit-identical to baseline).
- [ ] Smoke-verify: at `--ul2-enabled` step 1 BEFORE training,
      val NLL is bit-identical to `--ul2-enabled=false` (proves val
      forces S correctly).
- [ ] Run U0 baseline 5k.
- [ ] Run U1 (UL2 enabled) 5k; compare U1_S_NLL to U0_NLL.
- [ ] Apply pilot gate. If borderline, run seed=1338 disambiguation.
- [ ] If pilot PASSes: run U2 30k Phase-2 retrain.
- [ ] Position-stratified eval @ U2 step 30000 (forced-S samples).
- [ ] Mixture val NLL at U2 step 30000 (secondary diagnostic).
- [ ] Write `research/UL2_PHASE2_<RESULT>_2026_MM_DD.md` with
      trajectory, per-denoiser loss decomposition, position-strat
      evidence.
- [ ] If PASS: archive regstack ship, publish new ship, update
      CLAUDE.md, runner.sh path verification, memory entry.
- [ ] If FAIL: honest negative-result publication per Phase-3 P6,
      no silent re-targeting.

---

## 7. Repro commands

**5k pilot arc (Phase 1):**

```bash
cd ~/dev/glades-trainer

# U0: baseline (current regstack Phase 2 ship recipe, 5k steps)
sh run.sh flagship --zloss-coef 1e-4 --qk-norm \
    --steps 5000 --seed 1337 \
    --save database/checkpoints/chiron_1B_T16384_u0_5k

# U1: + UL2 mixture-of-3 (paper defaults: μ_R=3, p_R=0.15, μ_X=32, p_X=0.50, uniform 1/3)
sh run.sh flagship --zloss-coef 1e-4 --qk-norm --ul2-enabled \
    --steps 5000 --seed 1337 \
    --save database/checkpoints/chiron_1B_T16384_u1_ul2_5k
```

**30k Phase 2 (U2; gated on U1 pilot PASS):**

```bash
cd ~/dev/glades-trainer
sh run.sh flagship --zloss-coef 1e-4 --qk-norm --ul2-enabled \
    --steps 30000 --seed 1337 \
    --save database/checkpoints/chiron_1B_T16384_regstack_ul2_phase2
```

Span hyperparams (μ_R=3, p_R=0.15, μ_X=32, p_X=0.50) are config-level
constants with `--ul2-r-mu`, `--ul2-r-rate`, `--ul2-x-mu`,
`--ul2-x-rate` overrides (defaults match UL2 paper). Mixture is
uniform 1/3 (not config-exposed in this arc — sweep deferred per §5).

If the flag names above don't exist in `run.sh` after implementation,
invoke `./build/glades_chiron_train` directly with the equivalent
flag list (the `run.sh` recipe is a convenience wrapper).

---

This pre-registration is committed before any implementation code
lands. Per the Phase-3 program P6 prohibition (no silent
claim-dropping), if any gate above fails, the result is published
honestly and the program adapts — not silently re-targeted. The
LayerDrop arc's FAIL is a published example of this discipline; UL2
follows the same protocol.
