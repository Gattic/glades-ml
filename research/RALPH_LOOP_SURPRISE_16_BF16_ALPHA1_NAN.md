# Ralph-Loop Surprise #16 — α=1.0 + L=53 + bf16 NaN at scale

**Date:** 2026-04-25 (Ralph-loop iter 170, second 1.84B × 650k attempt)
**Type:** Scaling blind spot (taxonomy C from `RALPH_LOOP_METHODOLOGY_LESSONS.md`)
**Severity:** Run-terminating numerical divergence — silent NaN, ~6 h burned post-failure.

---

## What happened

The iter-169-fixed 1.84B × 650k run cleared all four early curriculum transitions cleanly (L 8→26 at 208k, T 256→512 at 260k, SAS α 0.3→0.5 at 299k, T 512→1024 at 390k). EMA recovered from the post-transition bump (10.17 → 9.25). Then the final two transitions fired:

- step 416,000 (08:34 wall): RLG L 26 → 53 (full depth)
- step 429,000 (09:58 wall): SAS α 0.5 → **1.0** (full attention)

At step 455,000 (15:31 wall, ~5.5 h later) the trainer logged:

```
[step 455000] loss=nan ema=nan best=2.2758@1541 acc=0.0000 ||g||=nan scale=0.000 ...
```

All gradient/loss values NaN. Gradient-clipping `scale=0.000` indicates the clip kernel saw inf/NaN and divided by it. The run continued under auto mode for several hours producing garbage updates before being noticed.

## Design-time claim that failed

The iter-168 SAS α-curriculum (0.3 → 0.5 → 1.0) was specified to "ramp to full attention by run end so the converged model is functionally identical to a no-SAS dense attention model." The terminal α=1.0 phase was assumed to be the safest part — it's just standard attention.

The hidden assumption: **bf16 attention at L=53 / T=1024 / α=1.0 / 1.84B is numerically stable.** No prior validation existed at that exact combination. iter 166's 6.33× speedup at 1.84B used α=0.1 (very heavy skipping) for the entire 2500-step run; α=1.0 was reached only briefly in development tests at smaller L.

## Actual mechanism (hypothesis)

α=1.0 doubles attention compute and gradient magnitude relative to the α=0.5 state the model spent ~13 k steps adapting to. At L=53/T=1024:

1. Per-head attention scores are O(T²) = ~10⁶ entries, softmax-normalized. Dynamic range demand: max(score) − min(score) can exceed bf16's ±240 representable range on rare batches.
2. With 53 layers stacked, gradient magnitudes through the deep stack accumulate; bf16 m, v in Adam start to overflow on outlier batches.
3. Once a single attention softmax overflows to inf, the inf propagates through value aggregation, layer norm, and the residual stream; the next backward produces NaN gradient; Adam steps a NaN into weights; from that point all forward passes are NaN.

The 500-step LR mini-warmup (iter 169) successfully bridged the α transition itself — the immediate post-transition steps were clean. The NaN appeared later, suggesting it's a long-tail outlier-batch event, not a transition-shock event. The iter-168 compound-shock failure mode is genuinely orthogonal to this one.

## Why this is taxonomy C (scaling blind spot)

- All α-curriculum validations at 5–10 k steps reached α=1.0 only briefly and with smaller L.
- All 1.84B / L=53 validations used fixed α ∈ {0.1, 0.3, 0.5}, never the curriculum's terminal α=1.0.
- The combination "L=53 AND T=1024 AND α=1.0 AND bf16 throughout" was unattacked before this run.
- bf16 dynamic range was the silent constraint; primitive parity tests passed at fp32 reference and didn't surface this.

## Fix applied (iter 170)

### Recipe patch (`glades-trainer/run.sh`)

1. **Cap auto-staggered α at 0.7** (was 1.0). Preserves 30 % attention compute savings, keeps gradient magnitudes bounded, avoids the bf16 ceiling.
2. **Anchor the second SAS transition to MAX(T2_step, L_max_step) + stagger** (was just T2_step). Prevents α jumping into the L=53 settling window. For a 650 k-step run this moves SAS 0.7 from step 429 k → 455 k.

```bash
SAS_T2_ANCHOR=$(max RLG_L2_STEP SLC_T2_STEP)
SAS_T2_STEP=$((SAS_T2_ANCHOR + STEPS*6/100))
CHIRON_SAS_SCHED="0.3@0,0.5@${SAS_T1_STEP},0.7@${SAS_T2_STEP}"
```

Users wanting α=1.0 at scale can still pass `--sas-schedule "0.3@0,0.5@N,1.0@M"` explicitly and accept the bf16 risk.

### Trainer patch (`glades-trainer/trainer/chiron_main.cpp`)

Added NaN/Inf early-stop in the per-step loss read:

```cpp
if (!(lossInst == lossInst) || lossInst > 1e6f) {
    log_error("...detected NaN/Inf — stopping run...");
    break;
}
```

Prevents the silent ~6 h compute waste that occurred between the actual NaN event (somewhere step 430 k–455 k) and the next 65 k log boundary (step 455 k). Previous behavior: if a user wasn't watching, the trainer would run to step 650 k producing a corrupted `.final` that auto-resume could later load.

## Methodology lessons

1. **bf16-throughout claims need the actual scale-AND-α-AND-T product validated.** Primitive parity at fp32 reference doesn't surface dynamic-range failures that only appear at the corner of the parameter space. Add a "scale-corner" gate: validate any new schedule axis at the full L_max × T_max × α_max combination of the flagship.
2. **Curriculum endpoints are not "safe by definition."** The iter-168 design treated α=1.0 as the trivial endpoint (no skipping ⇒ standard attention). At small L this is true; at L=53/bf16 it is not.
3. **Failure should be loud and fast.** A trainer that silently burns 6 h on garbage updates is worse than one that crashes at step 430 k. The iter-170 NaN early-stop is a one-line investment with day-of-compute payback.
4. **Stagger across all axes that change update magnitude.** iter 169 staggered SAS vs T; iter 170 extends to SAS vs L. Generalize: the new SAS endpoint should sit after `MAX` over all magnitude-changing axes' last transitions.

## Artifacts

- Code: `glades-trainer/trainer/chiron_main.cpp` lines ~3886-3905 (NaN early-stop)
- Recipe: `glades-trainer/run.sh` (α≤0.7 cap + L-anchored SAS endpoint)
- Checkpoints retained: `database/checkpoints/chiron_1.84B/chiron_1.84B.ckpt.step390000` (last good, EMA 9.25)
- Failed-run forensic data: terminal log lines step 1 → 455k (this surprise's primary observation set)
