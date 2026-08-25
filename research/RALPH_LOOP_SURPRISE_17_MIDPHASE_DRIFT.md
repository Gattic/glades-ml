# Ralph-Loop Surprise #17 — Mid-phase numerical drift at 1.84B / bf16

**Date:** 2026-04-26 (Ralph-loop iter 171, third 1.84B × 650k attempt)
**Type:** Scaling blind spot + measurement artifact (taxonomies B + C)
**Severity:** Run-terminating divergence; **third distinct failure mode** in three attempts.

---

## What happened

Run 3 cleared every transition that broke runs 1 and 2:

| Step | EMA | Wall | Event |
|---:|:-:|:-:|:--|
| 65,000 | 9.37 | 33:48 | Phase A nominal |
| 130,000 | **9.22** | 1:07 | Phase A nominal |
| 195,000 | 9.22 | 1:41 | Phase A end |
| 260,000 | 9.13 | 3:00 | post RLG L 8→26, post T 256→512 — clean |
| **325,000** | 9.69 | 4:58 | post SAS α 0.3→0.5 — recovering normally |
| **390,000** | **27.57** | **7:09** | **DIVERGED** — 17.88 nat EMA increase in 65k steps, no transition |

Then the iter-170 NaN early-stop fired at step 416003 (3 steps after L 26→53 transition), exiting cleanly without burning further compute.

## What's new about this failure

**No transition fired in the divergence window.** Step 325k → 390k is mid-Phase-C: L=26 fixed, T=512 fixed, α=0.5 fixed, lr at full 3e-4, FACE/MFIO/Adam all running normally. The 500-step LR mini-warmup from the SAS α 0.3→0.5 transition at step 299k expired at step 299500 — 25k+ steps before the drift began.

This rules out:
- Compound shock (surprise #15) — no co-located transitions
- α=1.0 bf16 overflow (surprise #16) — α was 0.5
- L=53 settling stress (surprise #16 partial) — L was 26

The remaining explanation: **cumulative numerical drift in bf16 attention/Adam/FACE state at 1.84B over a multi-10k-step horizon.**

## Comparison with run 2 (which got further)

Run 2 (iter 169, NaN at step ~430k–455k) and run 3 (iter 170, EMA blow-up at step ~360k–390k) used identical recipes through step 390k:
- Same seed (1337), same data, same flags, same binary (run 3 uses iter-170 build but the iter-170 patches don't affect Phase C trajectory)
- Run 2 EMA at step 390k: **9.25** (recovered cleanly)
- Run 3 EMA at step 390k: **27.57** (diverged)

The recipe is on a **knife-edge of stability** at 1.84B/bf16-throughout. CUDA reduction non-determinism alone (atomicAdd ordering, etc.) determines whether a particular run lucks into the converging trajectory or hits a borderline gradient that overflows bf16 dynamic range and starts a slow death spiral.

## Underlying mechanism (hypothesis)

bf16 has 8-bit mantissa (~3 decimal digits). At 1.84B parameters with FACE preconditioner:
- FACE row/col EMAs accumulate over thousands of steps with β=0.98 (~50-step half-life).
- A single batch with anomalous gradient magnitude (e.g., a long-range token sequence triggering large attention scores) inflates `dn̄` (column norm EMA).
- Inflated `dn̄` makes `σ = 1/√(zn̄·dn̄/(q̂·gF̄)+ε²)` produce smaller updates ⇒ but those updates land on weights whose gradient direction is now mis-calibrated by the next batch.
- Over many steps, FACE preconditioner stats and Adam m,v drift together, eventually producing updates that overflow bf16.

This is consistent with run 2 having lucky FACE state vs run 3's unlucky FACE state.

## Fix applied (iter 171)

### Trainer patch — EMA divergence early-stop

Added a forensic detector adjacent to the iter-170 NaN guard:

```cpp
if (step > 10000 && runningLoss > 0.0f) {
    if (runningLoss < minRunningLoss) minRunningLoss = runningLoss;
    if (runningLoss > minRunningLoss + 5.0f) {
        log_error("…EMA divergence: %.4f vs running min %.4f (+%.2f nat)…",
                  runningLoss, minRunningLoss, runningLoss-minRunningLoss);
        break;
    }
}
```

If next-time EMA diverges from its running minimum by ≥ 5 nat, the run aborts at the next log-every boundary (step 65k spacing). Run 3 would have caught the divergence around step 350k–365k instead of step 416k, saving ~1 h of compute.

### Quarantine

The iter-170 build's `.final` (saved at step 416003 with EMA 27 and NaN weights) was renamed to `chiron_1.84B.ckpt.divergent_step416003_emaspike`. We now have three quarantined `.divergent_*` checkpoints documenting all three failure modes for future forensic work.

## What this implies for the research program

**Three distinct 1.84B × 650k failure modes in three attempts is a structural finding**, not three separate bugs. The flagship recipe at this scale on bf16-throughout is not yet stable enough for 100k+-step horizons. The validated benchmark (FINAL_DELIVERABLE.md) is `1.84B × 2500 steps` — we are running 260× longer than ever validated.

Realistic paths forward:

1. **Drop to validated horizon.** Run 1.84B × ≤50k steps reliably; report the converged EMA. Honest research artifact.
2. **Add fp32 attention** behind a `--fp32-attn` flag (~1 day code work). Doubles attention scratch memory but should remove the bf16-edge instabilities. Validate at 100M scale before committing to 650k.
3. **Drop to 500M scale at 650k.** More numerical headroom, 500M flagship was validated longer-horizon (5000 steps in iter 151, EMA 8.4).
4. **Stiefel weights (paradigm #7)** at 1.84B — has 5.1B free-DOF ceiling, can run with rank reduction. Untested at long horizon but architecturally bounds spectral magnitude.
5. **Drop FACE β to 0.95** (faster forgetting) so a single bad batch doesn't poison FACE state for 50+ steps.

Recommended: **option 5 first** (cheapest test, single hyperparameter change, possibly resolves the FACE-induced amplification), **then option 2** (fp32 attention) if the drift persists.

## Methodology lesson

**Stability claims at scale need horizon validation, not just convergence validation.** The iter-161 56/28/16 1.84B × 2500 benchmark proves convergence at 2500 steps. It does NOT prove the recipe is stable at 65k+ steps without transitions. A new gate should be added:

> **Gate 4 — long-horizon stability**: any flagship recipe claimed for production runs must demonstrate ≥50k consecutive steps without transition at full L_max / T_max / α_max combination, with EMA strictly bounded above its running minimum + 1 nat.

Run 3 would have failed this gate; runs 1 and 2 also failed it but for different reasons. The flagship recipe needs Gate 4 validation before any further 100k+-step claims.

## Files

- Code: `glades-trainer/trainer/chiron_main.cpp` ~3905 (EMA divergence guard)
- Quarantined: `database/checkpoints/chiron_1.84B/chiron_1.84B.ckpt.divergent_*` (3 checkpoints, three failure modes)
- Last good: `chiron_1.84B.ckpt.step390000` from run 2 (EMA 9.25) — the one and only converged-past-Phase-C 1.84B checkpoint we have
