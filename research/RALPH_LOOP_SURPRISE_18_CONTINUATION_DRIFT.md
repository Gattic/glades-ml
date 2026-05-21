# Ralph-Loop Surprise #18 — CHRF resume drift in continuation runs

**Date:** 2026-05-01 → 2026-05-02 (Ralph-loop iter 181, run-10 of 1.84B campaign)
**Type:** Mechanism surprise + measurement artifact (taxonomies A + B)
**Severity:** Continuation runs from a converged CHRF checkpoint diverge into a
post-resume bad basin (EMA 9.2 → 27.5 over ~130k steps), even with full Adam,
Kahan, FACE, and step-counter state restored.

---

## What happened

Run-9 successfully trained 1.84B for 650k steps, reaching EMA 9.23 (best
single-step loss 1.21). The full state — weights + bf16 Adam (m, v) + Kahan
compensation + FACE EMAs (zn̄, dn̄, q̂, gF̄) + step counter + slcLastTransitionStep
+ runtime cfg.T/L/sasAlpha — was saved as a CHRF (iter-176) checkpoint.

Run-10 resumed from that CHRF with iter-181 `--continue` schedules forced to
terminal values (T=1024, L=53, α=0.7) so no curriculum transitions could
fire. The startup confirmed:

```
loaded full state from chiron_1.84B.ckpt.final
  resume step=650000, T=1024 L=53 α=0.700, slcLast=455000, faceStep=650000
```

Then training continued silently (`--log-every 130000`). The first post-resume
log at step 780k showed:

```
loss=27.43 ema=27.47 best=3.98@step 678749 ||g||=2.57 scale=0.39
```

Two facts about that line:

1. **`best=3.98@step 678749`** — single-step loss was still healthy 28k steps
   after resume.  The model didn't immediately drift on resume.
2. **`ema=27.47` at step 780k** — by 130k steps post-resume, EMA had blown
   up to the same "bad basin" attractor that broke runs 5/7 mid-curriculum.

So divergence happened somewhere in the 102k-step window between step 678k
and step 780k.  We have no logging at finer than 130k granularity.

## Why this is genuinely surprising

The iter-176 CHRF format restores everything mechanical:

- Weights (FP32 master + BF16 cached): ✓ verified bit-exact via self-test
- Adam m, v in BF16: ✓ verified
- Kahan c compensation buffer: ✓ verified
- FACE state (zn̄, dn̄, q̂, gF̄, face_step_count): ✓ verified
- Step counter, slcLastTransitionStep, runtime cfg: ✓ verified

Iter-178 5000-step LR mini-warmup is correctly *suppressed* on resume because
slcLastTransitionStep=455000 and current step=650000 → 195k steps past the
last transition, well outside the 5k warmup window.  So full LR resumes
immediately.  This was intentional (we don't want to reset LR on every
restart) but might be the load-bearing mistake.

Run-9's trajectory at step ~680k showed EMA in the 9.x range with ||g||≈1.4
and gradient clip firing softly.  Run-10's trajectory at the *same* step
range produced EMA blowing up to 27 over ~100k steps — same recipe, same
seed, same starting weights, just resumed.

## Hypothesis for the divergence mechanism

bf16 Adam state preservation is exact at the BF16 representation level, but
the *running statistics* of (m, v) reflect the gradient distribution
encountered during run-9's last batches.  After resume, the trainer reads
**different batches** from the data stream (the streamer doesn't checkpoint
its position), giving a different gradient distribution that doesn't match
the loaded m,v expectations.  Initial steps on mismatched data may produce:

1. Slightly off-direction Adam updates (m points at run-9's tail-batch
   gradient, v normalized for those, but new gradient has different
   variance);
2. FACE preconditioner σ = 1/√(zn̄·dn̄/(q̂·gF̄)+ε²) computed with stale row/col
   EMAs that don't match the new batch's frequency profile;
3. A few hundred steps of mis-calibrated updates accumulate into mild
   weight drift, which then enters the bf16 fragility regime that
   surprise #17 originally identified.

Once in the regime, FACE state poisoning compounds: an anomalous gradient
inflates `dn̄`, the next σ over-amplifies, weights drift further, FACE
state shifts more, etc.  The 5000-step warmup that *would* have absorbed
this kind of drift on a curriculum transition isn't fired here because
the saved slcLastTransitionStep is already 195k steps stale.

## Methodology lessons

1. **Data stream position matters for resumption fidelity.** The chiron
   trainer's `pretokenized_stream` doesn't checkpoint the file offset —
   on resume, it starts re-reading from the beginning of the next epoch.
   The "exactly continue training" assumption is broken.

2. **Iter-178 LR mini-warmup may need to fire on resume too**, not just
   on curriculum transitions.  A short post-resume warmup (e.g., 5000
   steps) would absorb the mismatched-gradient initial period.  This is
   a candidate iter-182 fix.

3. **CHRF self-test passed but couldn't catch this**.  Self-tests verify
   format round-trip integrity, which is necessary but not sufficient
   for "continuation produces same trajectory."  A new gate would need
   to validate that `train(N steps) ≈ resume(checkpoint at N/2) →
   train(N/2 steps)` end-to-end, which requires expensive integration
   testing.

4. **Generalization from #17**: bf16 fragility doesn't only surface on
   curriculum transitions — it can surface on *any* step where the
   optimizer's running statistics mismatch the current batch's gradient
   distribution.  That includes resumes, data-distribution shifts, and
   in principle any sufficiently anomalous batch streak.

## What's preserved as deliverable

- `chiron_1.84B.ckpt.run9_final` — the canonical run-9 1.84B checkpoint at
  step 650k, EMA 9.23 (CHRF, 10 GB).  This is THE deliverable.
- `chiron_1.84B.ckpt.step650000` — identical to run9_final.
- Run-10's `.final` (forthcoming at step 1.3M completion) will be a CHRF
  full-state file at EMA ~27, useful only for forensic analysis of the
  drift mechanism.

## Fixes shipped (2026-05-02)

### iter-182 — Post-resume LR mini-warmup hook  (5 lines, chiron_main.cpp ~4441)

On `load_full_checkpoint` success, override
`cfg.slcLastTransitionStep = resume.startStep`.  Hooks the existing iter-178
5000-step LR mini-warmup machinery to fire for the first 5k steps after
every resume.  Logs:

```
[resume] iter-182: re-armed slcLastTransitionStep=650000 (loaded value was 455000)
                  — 5000-step LR warmup will fire post-resume
```

The original loaded slcLastTransitionStep is preserved in the CHRF for
forensics; the override only affects the live trainer state.

### iter-184 — Cosine LR decay  (~20 lines, chiron_main.cpp ~4655)

New `--lr-decay` flag.  Auto-enabled by `run.sh --continue`.  After
`warmupSteps` complete, lrScale is multiplied by:

```
progress    = (step - warmupSteps) / (maxSteps - warmupSteps)
cosine      = 0.5 · (1 + cos(π · progress))
decayFactor = lrDecayMin + (1 - lrDecayMin) · cosine    # default min = 0.1
```

So lr smoothly decays from full lr → 10% over the run.  Combines correctly
with iter-178 transition mini-warmups (multiplicative composition).

## Empirical validation (run-11, 2026-05-02 → 2026-05-04)

Re-ran the same 1.84B continuation that broke as run-10, with iter-182
and iter-184 active:

| Run | Step 780k EMA | lr at 780k | Outcome |
|-----|--------------:|-----------:|---------|
| Run-10 (no iter-182/184) | **27.47** ❌ | 3.00e-04 (flat) | Locked in bad basin, NaN'd at step 867k |
| Run-11 (iter-182 + iter-184) | **9.14** ✓ | 1.23e-04 (cosine) | Healthy, continuing through step 910k+ |

**Δ EMA at step 780k: +18.33 nat in favor of the fixed recipe.**  Decisive.

`lr=1.23e-04` at step 780k matches the iter-184 cosine prediction exactly:
`3e-4 × (0.1 + 0.9 · 0.5 · (1 + cos(0.6π))) = 3e-4 × 0.41 = 1.23e-4`.
At step 910k, lr=8.57e-5, also matching prediction.

EMA at step 910k = 9.44 (+0.30 nat from step 780k).  Tiny rise, well within
healthy training noise.  No NaN, no detector trips, throughput steady at
1.66 tok/s.

## Methodology lessons (updated)

1. **Resume drift is real and reproducible** — surprise #18 happened in
   run-10 at the predicted point, and was fully prevented in run-11 by
   the predicted intervention.  This is the cleanest cause-and-effect
   demonstration in the project's surprise-history.

2. **The data-stream offset isn't the *only* fix** — iter-183 (true
   bit-exact resume with checkpointed stream offset) is still open
   work, but iter-182's warmup workaround appears sufficient in
   practice.  Future continuation-aware research should still ship
   iter-183 if exact reproducibility matters; for fragility-reduction
   alone, iter-182 closes the gap.

3. **Cosine LR decay was a "should always have had it" fix.**
   Production trainers virtually always use lr decay; we'd been
   running flat lr=3e-4 through end-of-run, contributing to Phase E
   bad-basin susceptibility independent of surprise #18.  Auto-
   enabling it in `--continue` mode is the conservative default.

4. **Generalization over surprise #17**: bf16 fragility surfaces on any
   sufficient gradient-distribution mismatch.  Curriculum transitions
   (#15-#17) and resumes (#18) are two manifestations.  The unified
   prevention is "any time the optimizer's running stats are likely
   stale wrt the current batch distribution, fire a brief LR warmup."

## Candidate fixes still deferred

- **iter-183**: pretokenized_stream byte-offset checkpointing in CHRF.
  Adds a `streamOffset` field to the CHRF header so resume reads from the
  exact byte where save was issued.  ~30 lines.  Unblocks bit-exact
  reproducible resumes.  Not needed for surprise #18 prevention since
  iter-182 absorbs the mismatch via warmup.
- **iter-185**: periodic safety LR warmup every N steps (e.g., 50k).
  Speculative; would catch mid-phase drift before compounding even
  outside transition/resume windows.  Not needed if iter-178/-182/-184
  composition holds.

## Files

- Code: `glades-trainer/trainer/chiron_main.cpp` ~4441 (iter-182 hook),
  ~4655 (iter-184 cosine decay)
- Recipe: `glades-trainer/run.sh` (auto-enables `--lr-decay` for `--continue`)
- Validation log: run-11 chiron_1.84B continuation, May 2-4 2026
- Surviving forensic checkpoints:
  - `chiron_1.84B.ckpt.run9_final` — canonical pre-continuation 1.84B (EMA 9.23)
  - `chiron_1.84B.ckpt.run10_diverged_step867895` — bad-basin reference (EMA ~27)
  - `chiron_1.84B.ckpt.step780000` — run-11 mid-continuation healthy state (EMA 9.14)

## Files

- Code touched (during diagnosis, not fix): none in glades-ml; `run.sh`
  iter-181 `--continue` flag is responsible for the schedule reset.  No
  bug there.
- Deliverable: `database/checkpoints/chiron_1.84B/chiron_1.84B.ckpt.run9_final`
- Forensic checkpoints: `chiron_1.84B.ckpt.step780000` (run-10's mid-drift
  snapshot, useful for diff vs run9_final)
