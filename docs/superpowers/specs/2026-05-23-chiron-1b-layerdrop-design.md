# CHIRON 1B LayerDrop (Stochastic Depth) — Design

**Date:** 2026-05-23
**Status:** Pre-registered design + implementation-time addendum (2026-05-23)

> **Implementation addendum (2026-05-23, post-Task-4.3):** A material
> deviation from §2.1's inverted-dropout convention was made at the
> production-trainer wiring layer. The library (`glades-ml`
> `sgd_transformer.cpp`) implements the spec's inverted-dropout
> faithfully (mathematically clean, train≡inference). However, the
> production training binary `glades_chiron_train` (from
> `glades-trainer/chiron_main.cpp`, used by `sh run.sh flagship`) uses
> CHIRON's symplectic update structure `(p, q) → (p + shear(q),
> reln(q))`, not the standard residual transformer update
> `x_{l+1} = x_l + F(x_l)`. The `q = reln(q)` step is a
> non-linear *transformation*, not a residual add — it has no clean
> `1/(1-p_l)` scaling interpretation. To match the existing SAS
> paradigm #40 (which already skips layers in CHIRON without
> rescaling) and avoid contrived rescaling on `reln`, the production
> wiring uses **hard-drop semantics**: when `mask_l = 0`, both `shear`
> and `reln` are fully skipped (the layer is a no-op on both `p` and
> `q`); when `mask_l = 1`, `shear` and `reln` run UNSCALED. The
> resulting train/inference contribution gap (~5% of layer-stack
> output on average at `p_max = 0.1`, `L = 24`) is the intentional
> cost; the regularization effect is qualitatively similar to Fan
> 2019's original LayerDrop convention (which also did not rescale).
>
> **Two divergent LayerDrop implementations now coexist:**
> 1. **Library** (`sgd_transformer.cpp`): inverted-dropout per §2.1
>    of this spec. Used by unit tests and any future standard-residual
>    transformer training path. Math-verified bit-identical at
>    `p_max = 0`.
> 2. **Production trainer** (`chiron_main.cpp`): hard-drop semantics
>    adapted for CHIRON's symplectic structure. Used by
>    `sh run.sh flagship`. Math-verified at smoke shape: `p_max=0.0`
>    bit-identical to baseline; `p_max=0.33` produces a different
>    (lower) step-1 loss and gradient norm.
>
> §3.2 gate criteria (5k pilot main-head NLL parity ≤ +0.02 nat;
> 30k Phase-2 NLL improvement target ≥ +0.02 nat vs regstack ship)
> are unchanged — the regularization signal we're testing is
> CHIRON-LayerDrop, just with hard-drop semantics. Throughput budget
> (≤5% wall regression) is also unchanged.
**Program scope:** Phase-3 of CHIRON 1B production line. Adds the first
architecture-side stochastic-regularization mechanism to the stack —
layer-level Bernoulli skip (a.k.a. stochastic depth, Fan et al. 2019 /
Huang et al. 2016) on top of the regstack Phase 2 ship.
**Baseline:** `chiron_1B_T16384_regstack_phase2.final` (regstack Phase
2 ship 2026-05-22). Final val NLL 3.5734 @ step 30000, throughput
28,072 tok/s, VRAM 14.97 / 15.56 GB on RTX-4080-SUPER.
**Output target:** A 30k Phase-2 production retrain at regstack
Phase 2 + LayerDrop, gated on a 5k Phase-1 pilot arc.

---

## 1. Motivation

The current CHIRON 1B flagship (regstack Phase 2: Z-loss + QK-Norm)
landed −0.5983 nat val NLL improvement over the v5+FP8 base. QK-Norm
dominated the gain; Z-loss was noise at pilot scale but retained per
§3.3 of the regstack spec. The MTP follow-up arc closed NEGATIVE
across 5 variants — including a divergence event with the combined
late-positions/offset-3/λ=0.01 stack — so the auxiliary-loss /
multi-head class of mechanisms is provisionally exhausted on this
codebase.

The natural next regularization class is **architecture-side
stochastic regularization** — depth-noise mechanisms that perturb the
forward pass without adding new heads or new objectives. LayerDrop
(Fan, Grave, Joulin 2019) / stochastic depth (Huang et al. 2016) is
the canonical instance:

- Well-validated at LLM scale (DeepNet 1000-layer training; modern
  vision-transformer pipelines via `timm`; PaLM-2 reportedly uses
  per-sublayer drop).
- Implementable in <200 LOC inside the existing codebase — a per-step
  per-layer Bernoulli draw + branched forward/backward at the
  existing residual-add hooks (the residualDropoutRate sites in
  `sgd_transformer.cpp` are the same hook points).
- Composes cleanly with the regstack Phase 2 stack: Z-loss is a
  readout mechanism (unaffected by per-layer drops); QK-Norm γ_h
  gradients get reduced update frequency on deep layers (≤5% at
  p_max=0.1) which is well within FACE optimizer noise envelope.
- Carries a **wall improvement bonus** — skipping the block entirely
  on the backward pass saves ~5-6% compute at p_max=0.1 L=24 linear
  schedule, on top of any NLL gain. Even if NLL is null, the wall
  saving alone could be a ship-clean strict +5% PASS.

The arc is decomposed from the broader "regularization techniques"
brief into a single-mechanism, single-config 5k pilot followed by a
30k Phase-2 retrain — the same shape as the regstack B0/B2/B5 arc
that produced QK-Norm.

---

## 2. Architecture

### 2.1 LayerDrop (mechanism A — sole mechanism in this spec)

Per-layer Bernoulli mask drawn once per training step:

```
mask_l ∈ {0, 1}   ~   Bernoulli(1 - p_l),    l ∈ {0, 1, …, L-1}
```

with linear-rising schedule:

```
p_l = (l / (L - 1)) · p_max,    L = 24,    p_max = 0.1
```

so `p_0 = 0` (layer 0 never drops; protects the embedding-adjacent
representation), `p_{23} = 0.1` (deepest layer drops 10% of steps).
Mean `p̄ = p_max / 2 = 0.05` → expected **~1.2 skipped blocks per
fwd pass** of the L=24 stack.

**Block update (inverted-dropout convention, Approach A):**

Training:
```
x_{l+1} = x_l + (mask_l / (1 - p_l)) · F_l(x_l)
```

Inference:
```
x_{l+1} = x_l + F_l(x_l)
```

where `F_l(·)` is the entire transformer block — both the PreNorm →
attention (SCFA inner + outer) → residual sub-add AND the PreNorm →
MLP → residual sub-add, treated as one indivisible unit under the
same `mask_l`. Expected training output equals inference output
exactly (no train/inference scale gap).

**Bit-identicality at p_max = 0:** all `p_l = 0` → all `mask_l = 1`
deterministically (Bernoulli at p=0 has no draws) → `1/(1-p_l) = 1` →
fwd/bwd math is **bit-identical to the regstack Phase 2 ship**.
Guarded by early return on `layerDropPMax == 0.0f`.

**Why this fits CHIRON 1B specifically:**

- L=24 with p_max=0.1 linear gives ~1.2 expected skipped blocks per
  forward — strong enough to inject ensemble-like regularization
  without dropping more than ~5% of total layer-compute on average.
- The SCFA inner attention's compressed Q/K projection feeds only
  the residual stream — there is no other downstream consumer per
  layer, so skipping the block cleanly skips all SCFA buffers for
  that (layer, step) pair.
- The wall improvement (~5-6% expected at p_max=0.1) is a free
  by-product: backward pass on a dropped block is a single zero-grad
  store + skip, saving the layer's full bwd attention + MLP +
  QK-Norm + LayerNorm compute.
- QK-Norm γ_h gradients get reduced update rate (~5% at the deepest
  layer) but the per-layer init at log₂(T) = 14 plus FACE's
  Adafactor variance accumulation absorbs this without instability.
- No interaction with the FP8 readout path (LayerDrop is per-layer,
  the readout is post-stack).

**Forward (added at the existing block residual-add point):**

```c++
// At each layer l = 0..L-1, before the attention+MLP block:
const float p_l = (static_cast<float>(l) / static_cast<float>(L - 1))
                  * layerDropPMax;
const float keepProb = 1.0f - p_l;
const bool keep = glades::rng::bernoulli(networkRng, keepProb);
// Persist mask_l (1 byte) + scale (4 bytes) into the layer's fwd-saved
// state for bwd reuse.
savedState.layerDropKept[l] = keep ? 1u : 0u;
if (!keep) {
    // Skip the block; residual passes through unchanged.
    // x_{l+1} = x_l (memcpy-like; in-place residual stream means no-op).
    continue;
}
const float scale = 1.0f / keepProb;
// Run the block, scale its output, add to residual.
run_block_fwd(x_l, &block_out_l);
axpy(scale, block_out_l, x_l_residual);  // x_{l+1} = x_l + scale · F_l(x_l)
```

**Backward (mirrors fwd branching):**

```c++
for (int l = L - 1; l >= 0; --l) {
    if (savedState.layerDropKept[l] == 0u) {
        // Block was skipped on fwd; bwd grad of block is 0; residual
        // grad flows through unchanged. No work for this (layer, step).
        continue;
    }
    const float p_l = (static_cast<float>(l) / static_cast<float>(L - 1))
                      * layerDropPMax;
    const float scale = 1.0f / (1.0f - p_l);
    // The fwd applied 'scale' to F_l(x_l); the bwd grad through that
    // multiply is grad_out * scale, propagated into the block's bwd.
    axpy(scale, grad_residual, grad_block_out);  // pre-scale incoming
    run_block_bwd(grad_block_out, x_l, &grad_block_in);
    // grad_residual already accumulates the identity-path gradient.
    axpy(1.0f, grad_block_in, grad_residual);
}
```

**Implementation sites:**

- `Backend/Machine Learning/Networks/training_config.h` — add
  `float layerDropPMax` (default `0.0f`), `bool layerDropLinearSchedule`
  (default `true`, reserved for a possible future constant-schedule
  variant — not exercised in this arc).
- `Backend/Machine Learning/Networks/transformer_config.cpp` —
  validation: `0.0f ≤ layerDropPMax < 1.0f`, mirroring the existing
  `embeddingDropoutRate` / `residualDropoutRate` checks.
- `Backend/Machine Learning/Networks/sgd_transformer.cpp` — fwd
  branch at the per-layer block entry; bwd mirror branch. Hook in
  the same vicinity as the existing `resDropRate` sites (around
  lines 7780, 7983, 8072, 9005, 9095). Persist `layerDropKept` flags
  and scales into the per-step saved-state struct alongside the SCFA
  buffers and QK-Norm γ caches.
- `glades-trainer` `chiron_main` and `run.sh` — add
  `--layer-drop-pmax F` flag (default `0.0`). Wire into
  `TransformerRunConfig.layerDropPMax`. Add usage block parallel to
  `--zloss-coef` and `--qk-norm` in the `run.sh` flagship help.
- `unit-tests/Backend/Machine Learning/transformer_improvements.cpp`
  (or equivalent location next to existing CHIRON unit tests) — add:
  - `CHIRONLayerDropBitIdenticalAtZeroTest` (p_max=0 path is
    bit-identical to regstack ship at single-element FP32).
  - `CHIRONLayerDropMaskSchedulePmax01Test` (verifies `p_0 = 0`,
    `p_23 = 0.1`, intermediate p_l matches `(l/23)·0.1`).
  - `CHIRONLayerDropDeterministicMasksAtSeedTest` (same seed across
    2 process invocations yields identical mask sequence over a
    100-step trace).

**RNG / determinism:** mask draws use the existing
`glades::rng::bernoulli(rng, 1 - p_l)` against the per-network
deterministic engine (per
`Backend/Machine Learning/DETERMINISM_AND_CONCURRENCY.md`). At
seed=1337, the mask sequence is fully reproducible across reruns. No
new RNG state is introduced. Eval paths set `layerDropPMax = 0.0f`
internally (mirrors how `embeddingDropoutRate` / `residualDropoutRate`
are zeroed on eval — same call site).

**SCFA interaction:** when `mask_l = 0`, the entire block is skipped,
which means the SCFA inner attention's compressed Q/K projection,
depthwise-conv pass, and outer attention are all bypassed. SCFA
outputs feed only the per-layer residual stream — no other downstream
consumer per layer — so the skip is clean. Verified at implementation
time by auditing buffer lifetimes in `sgd_transformer.cpp`.

**QK-Norm interaction:** when `mask_l = 0`, that layer's `γ_h`
parameters receive zero gradient for that step (the block's bwd is
skipped entirely). At p_max=0.1 linear, the deepest layer's γ_h
update rate is reduced by 10%; intermediate layers see ≤5% reduction;
layer 0 is unaffected. FACE / Adam absorbs this without instability
(verified by monitoring γ_h evolution at steps 1000/2500/5000 — see
R-LD-2).

**Z-loss interaction:** Z-loss is computed at the readout
(post-stack), wholly independent of per-layer drops. Unaffected.

**FP8 readout interaction:** the FP8 path is post-stack readout; per
the regstack spec §2.1 it operates on the FP32 logits. LayerDrop is
per-layer block-level; no interaction.

**Strict bit-identicality test:** at `layerDropPMax == 0.0f`, the
fwd/bwd codepath must produce **bit-identical** loss and gradients vs
the regstack Phase 2 ship. Guarded by an early return on `keepProb ==
1.0f` (which is the path taken when p_l = 0; no Bernoulli call, no
scale multiply, no mask write).

---

## 3. Validation Arc

### 3.1 Run plan (1 baseline rerun + 1 pilot + 1 Phase-2 retrain)

| ID | Steps | Seed | Config | Compute |
|---|---:|---:|---|---:|
| L0 | 5000 | 1337 | regstack Phase 2 ship (current flagship) | ~47 min |
| L1 | 5000 | 1337 | L0 + `--layer-drop-pmax 0.1` | ~44 min (expected -6% wall) |
| **L2** | **30000** | **1337** | **L0 + `--layer-drop-pmax 0.1` (if L1 PASSes)** | **~4.5 h** |

Total compute: ~91 min for pilot arc + ~4.5 h for the 30k Phase-2
retrain = ~6 hours wall on the gated path. Single-seed (seed=1337) per
the regstack / iter-93 / iter-94 Phase-1 / Phase-2 precedent.
Multi-seed (n=3) deferred to a separate gate if L2 PASSes (per the
regstack precedent — multi-seed validates the ship; pilot+phase-2 is
single-seed).

### 3.2 Gate criteria (pre-committed, no silent re-targeting)

**L0 sanity:** the re-run baseline L0 must reproduce the published
regstack Phase 2 5k trajectory to within ±0.05 nat at step 5000
(same-day variance calibration). If not, F-LD-1 fires — investigate
non-determinism in the regstack Phase 2 stack (Z-loss BF16-storage path
or QK-Norm γ_h init) before continuing.

**Pilot 5k gate (L1 vs L0):**
- **Main-head val NLL** (CE only, no Z-loss aux term): `L1_NLL ≤
  L0_NLL + 0.02`. No >0.02 nat regression at 5k single-seed; an
  improvement is preferred but not required at the 5k noise floor
  (~±0.05 nat). Auxiliary terms (Z-loss log Z²) excluded from the
  gate per regstack §3.3 — apples-to-apples comparison requires
  main-head CE only.
- **Throughput:** `L1_tok_s ≥ L0_tok_s · 0.95` (≤5% wall regression).
  This is a soft floor; LayerDrop should *improve* wall by ~5-6%
  (expected `L1_tok_s ≈ 29,500 tok/s` vs L0 `28,072 tok/s`).
- **VRAM:** ≤ 15.72 GB peak (1% above the 15.56 GB ceiling — same as
  regstack budget).
- **Borderline-null case:** if `0.02 nat < L1 regression ≤ 0.05 nat`,
  run a seed=1338 pilot before deciding (mirrors iter-91 multi-seed
  noise-vs-signal disambiguation; cost ~+44 min).
- **Outright FAIL:** if `L1_NLL > L0_NLL + 0.05`, publish honest
  negative result; close arc; do not run L2.

**30k Phase-2 gate (L2 vs regstack Phase 2 ship):**
- **Val NLL @ step 30000:** ≤ 3.5534 (= 3.5734 − 0.02 nat improvement
  vs current ship). The +0.02 nat improvement bar is the strict ship
  bar inherited from regstack §3.2.
- **Throughput:** ≥ 26,668 tok/s (≤5% wall regression vs ship's 28,072
  tok/s). Expected: ~29,500 tok/s (positive +5-6%).
- **VRAM:** ≤ 15.72 GB peak.
- **Position-stratified eval at step 30000:** report main-head val NLL
  binned by position bucket [0, 4k), [4k, 8k), [8k, 12k), [12k, 16k]
  (the same buckets used in regstack §3.4) — if the deep-position
  bucket [12k, 16k] regresses >0.05 nat vs ship despite aggregate
  PASS, flag for R-LD-6 follow-up before committing the ship swap.
- **PASS handling:** archive current regstack Phase 2 ship at
  `database/checkpoints/chiron_1B_T16384_regstack_phase2/`. New ship
  `chiron_1B_T16384_regstack_layerdrop_phase2.final`. Update
  `CLAUDE.md` "Current Production Flagship" block with new NLL /
  throughput / VRAM numbers and the reproduce command
  (`sh run.sh flagship --zloss-coef 1e-4 --qk-norm --layer-drop-pmax 0.1`).
  Write `research/REGSTACK_LAYERDROP_PHASE2_PASS_2026_MM_DD.md` with
  trajectory table and position-stratified evidence.
- **FAIL handling (NLL):** publish honest negative result at
  `research/REGSTACK_LAYERDROP_PHASE2_FAIL_2026_MM_DD.md`; LayerDrop
  code stays in the trainer as opt-in (default `0.0`), bit-identical
  when off; close arc.
- **FAIL NLL but PASS wall improvement:** publish honest result; do
  NOT make `--layer-drop-pmax 0.1` the default; opt-in flag stays
  available for downstream experimentation. Don't ship a wall-only
  improvement at the cost of NLL.

### 3.3 Production retrain handoff

If L2 PASSes:
- Archive: `database/checkpoints/chiron_1B_T16384_regstack_phase2/`
  → keep loadable; both the current regstack stack (with
  `--layer-drop-pmax 0.0`) and the new layerdrop stack are
  bit-identical-loadable into the trainer (default off).
- New ship checkpoint: `chiron_1B_T16384_regstack_layerdrop_phase2.final`.
- Update `CLAUDE.md` "Current Production Flagship" block with the new
  reproduce command:
  `sh run.sh flagship --zloss-coef 1e-4 --qk-norm --layer-drop-pmax 0.1`.
- Update `glades-trainer/runner.sh` `--flagship` to point to the new
  checkpoint (or verify path consistency — at inference, LayerDrop is
  always off, so the same runner works against either ship without
  flag changes).
- Write `research/REGSTACK_LAYERDROP_PHASE2_PASS_2026_MM_DD.md` with
  the trajectory table, gate-by-gate evidence, position-stratified
  NLL, and mask-trace summary (mean keep-rate per layer over the 30k
  run).
- Memory entry: layerdrop pilot + Phase-2 result, with NLL delta and
  wall delta as the headline metrics.

---

## 4. Risks (pre-registered)

| Risk | Probability | Mitigation |
|---|---|---|
| **R-LD-1:** 5k single-seed noise (~±0.05 nat) hides NLL signal at p_max=0.1 (literature reports ~−0.05 to −0.15 nat at this scale) | medium | If L1 is borderline (Δ within ±0.02 nat), run a seed=1338 pilot before committing to L2. Mirrors iter-91 / regstack §3.2 multi-seed disambiguation. Cost: +44 min. |
| **R-LD-2:** Skipping deep layers degrades QK-Norm γ_h convergence (init=log₂(T)=14, requires gradient signal to settle) | low | γ_h update rate reduced ≤10% at deepest layer. Monitor γ_h trajectory at steps 1000/2500/5000 of L1 — compare against the published regstack Phase 2 γ_h evolution (logged in regstack ship doc). If divergence >2× the regstack std envelope, flag. Mitigation: linear schedule already protects layer 0 (`p_0 = 0`) so the embedding-adjacent γ_h is fully trained. |
| **R-LD-3:** Inverted dropout 1/(1-p_l) amplifies pre-softmax logit variance in training → interacts with Z-loss (penalizes log²Z) | low | Z-loss should naturally suppress any inflation. Verify L1's Z-loss term magnitude stays within 2× of L0's at step 5000; if >2× inflated, flag for investigation. |
| **R-LD-4:** Wall savings smaller than expected — SCFA inner attention buffers may not be cleanly skippable (audit needed at implementation) | low | Audit at implementation time: SCFA outputs (Q_par, K_par, V_par, depthwise-conv result, inner-attn output) feed only the per-layer residual stream, no cross-layer consumers. Should be fully skippable. If audit reveals any buffer dependency, fall back to Approach B (always-execute with mask multiply) for that buffer only, with documented wall cost. |
| **R-LD-5:** Determinism break — mask draws not properly seeded, runs at same seed give different mask traces | low | Use the per-network deterministic engine `glades::rng::*` per `DETERMINISM_AND_CONCURRENCY.md`. Covered by `CHIRONLayerDropDeterministicMasksAtSeedTest` (same seed, 2 process invocations, identical 100-step mask trace). |
| **R-LD-6:** L2 PASS at aggregate NLL gate but degrades position-stratified NLL at deep positions (LayerDrop's regularization may favor shallow contexts) | low | Report position-stratified val NLL at step 30000 (buckets [0,4k), [4k,8k), [8k,12k), [12k,16k]); if [12k,16k] regresses >0.05 nat despite aggregate PASS, flag for R-LD-6 follow-up — possibly retry with `p_max = 0.05` or non-linear schedule that protects deep layers more. Don't ship until resolved. |
| **R-LD-7:** Determinism break across eval / train transitions — eval path must force `layerDropPMax = 0` internally; if it leaks the training value, val NLL is artificially noisy | low | Mirror the existing residualDropoutRate eval-zeroing call site (already in sgd_transformer.cpp); covered by `CHIRONLayerDropBitIdenticalAtZeroTest` running on the eval-mode call signature. |

---

## 5. Out of scope for this spec

- **p_max sweep** (single config p_max=0.1 chosen for the pilot; if L1
  is borderline-null, a follow-on sweep at p_max ∈ {0.05, 0.15, 0.2}
  is a separate arc).
- **Per-sublayer drop** (drop attn and MLP sub-residuals independently
  with separate Bernoulli draws — PaLM-2 style). Full-block drop
  chosen for this arc; per-sublayer is a deferred follow-on.
- **No-rescale variant** (Fan 2019 original — train without
  1/(1-p_l) multiply, accepting the ~12% train/inference gap).
  Inverted-dropout chosen here for theoretical cleanliness.
- **Inference-time layer pruning** (Fan 2019's structured-pruning use
  case — drop layers at inference to make smaller model). Possible
  follow-on but not part of this arc; the regstack Phase 2 ship's
  inference path is unchanged.
- **Multi-seed validation of the new stack** at any step count —
  deferred to a separate N=3 confirmation gate after L2 PASSes (per
  the regstack / iter-93/94 / iter-100 precedent: single-seed
  Phase-1 + Phase-2, multi-seed in a subsequent confirmation arc).
- **UL2 mixture-of-denoisers** — separate spec/arc per scope
  decomposition in the brainstorm; brainstormed fresh after this arc
  resolves.
- **Constant (non-linear-rising) schedule** — `layerDropLinearSchedule
  = false` path is reserved in the config but not exercised here.
- **Downstream eval (LAMBADA, MQAR-Pile, etc.)** — per Phase-3 P10
  (no external datasets); position-stratified val NLL on
  `pretok-data/` validation split is the eval methodology.

---

## 6. Deliverables checklist

- [ ] Pre-register this spec in memory (key:
      `chiron_1b_layerdrop_prereg`).
- [ ] Implement Approach A — config field, fwd skip branch, bwd skip
      branch, RNG wiring, flag wiring at `run.sh` / chiron_main
      level, strict-bar test at `layerDropPMax == 0.0`.
- [ ] Add `CHIRONLayerDropBitIdenticalAtZeroTest`,
      `CHIRONLayerDropMaskSchedulePmax01Test`,
      `CHIRONLayerDropDeterministicMasksAtSeedTest`.
- [ ] Run L0 baseline 5k (verify same-day variance ≤±0.05 nat vs
      published regstack Phase 2 5k trajectory).
- [ ] Run L1 (LayerDrop p_max=0.1) 5k; compare to L0.
- [ ] Apply pilot gate (§3.2). If borderline, run seed=1338 pilot.
- [ ] If pilot PASSes: run L2 30k Phase-2 retrain.
- [ ] Position-stratified val NLL evaluation at L2 step 30000.
- [ ] Write `research/REGSTACK_LAYERDROP_PHASE2_<RESULT>_2026_MM_DD.md`
      with gate-by-gate evidence, mask-trace summary, and γ_h
      evolution comparison vs regstack ship.
- [ ] If PASS: archive current ship, publish new ship, update
      `CLAUDE.md`, runner.sh path verification, memory entries.
- [ ] If FAIL: honest negative-result publication per Phase-3 P6, no
      silent re-targeting.

---

## 7. Repro commands

**5k pilot arc (Phase 1):**
```bash
cd ~/dev/glades-trainer

# L0: baseline (current regstack Phase 2 ship recipe, 5k steps)
sh run.sh flagship --zloss-coef 1e-4 --qk-norm \
    --steps 5000 --seed 1337 --val-every 500 \
    --save database/checkpoints/chiron_1B_T16384_l0_5k

# L1: + LayerDrop p_max=0.1 linear-rising
sh run.sh flagship --zloss-coef 1e-4 --qk-norm --layer-drop-pmax 0.1 \
    --steps 5000 --seed 1337 --val-every 500 \
    --save database/checkpoints/chiron_1B_T16384_l1_layerdrop_5k
```

**30k Phase 2 (L2; gated on L1 pilot PASS):**
```bash
cd ~/dev/glades-trainer
sh run.sh flagship --zloss-coef 1e-4 --qk-norm --layer-drop-pmax 0.1 \
    --steps 30000 --seed 1337 --val-every 1500 \
    --save database/checkpoints/chiron_1B_T16384_regstack_layerdrop_phase2
```

If the `--layer-drop-pmax` flag name does not exist in `run.sh` after
implementation, invoke `./build/glades_chiron_train` directly with
the equivalent flag list (the `run.sh` recipe is a convenience
wrapper).

---

This pre-registration is committed before any implementation code
lands. Per the Phase-3 program P6 prohibition (no silent
claim-dropping), if any gate above fails, the result is published
honestly and the program adapts — not silently re-targeted.
