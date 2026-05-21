# Paradigm Shift #92 Candidate B — GATE-0-CAMPAIGN-CHIRON: Operational Implementation of the iter-236 Test-Before-Build Constraint

**Status:** CANDIDATE B — META-paradigm; operational-execution class.
**Date:** 2026-05-08 (Ralph-loop iter 236, post-#91 close).
**Axis:** **NONE (META-OPERATIONAL)** — does not add a new axis nor multiply existing axes; executes empirical validation of the top-5 unvalidated bold-claim paradigms.
**Magnitude target:** **N/A — 0× new magnitude.** Validates existing magnitudes at LLM-scale; does not generate them.

---

## 0. Executive summary

**Mechanism:** GATE-0-CAMPAIGN-CHIRON is **not a new training paradigm**. It is the operational implementation of the constraint the user added at iter-236: *"Any 'breakthroughs' or bold claims should be tested before we build off of them so we do not waste time in the wrong direction. Testing can take a long time so only test when relevant and make sure we only test for a maximum of 1 day per test."*

Across iter-231 to iter-235 the program produced four META-class artifacts (#87 META-VALIDATION, #88 META-VALIDATION-II, #89 META-VALIDATION-III, #91 META-VALIDATION-IV equivalents) that all reached the same conclusion as a recommendation: pivot from paradigm-design to empirical validation. Each of those artifacts was RESERVED-AS-RECOMMENDATION; the autonomous-loop rhythm continued producing axis-extension paradigms instead of running probes. **At iter-236 the user has now made the recommendation binding by sharpening the brief.** This document executes it.

**The campaign:**
- Identify the top-5 unvalidated paradigms with the largest claimed magnitudes.
- Allocate ≤ 1 GPU-day (~12 GPU-hours on the user's RTX 4080 SUPER) to a Gate-0 probe of each.
- Run the campaign **sequentially, highest-magnitude-first**, so that the first PASS unlocks immediate build-confidence and a FAIL aborts the wrong-direction work before further investment.
- Total campaign budget: **5 days = 5 paradigms × 1 day each** (with one 16-hour exception that may be moved to a smaller model to fit).

**Top-5 selected (by claimed-magnitude rank, all currently un-Gate-0'd):**

| Rank | Paradigm | Bold claim | Probe budget |
|---|---|---|---|
| 1 | **#73 PHOENIX-DISTILL-COMBO** | 100× wall-clock to fixed final NLL via Llama 3.1 405B teacher; 18B-effective at 16 GB | 8 GPU-h |
| 2 | **#74 PHOENIX-1BIT-DISTILL** | 32B-effective at 16 GB; binary middle layers; ≤ 0.15 nat additional loss | 12 GPU-h |
| 3 | **#69 REASONING-DISTILL** | Billion-multiplier on causal-reasoning subset via DeepSeek-R1 traces | 10 GPU-h |
| 4 | **#77 MOEFICATION-DISTILL** | 256B-effective via post-hoc 8-way MoE on PHOENIX base | 16 GPU-h (exception, see §2.4) |
| 5 | **#78 ATTENTION-SINK** | T → ∞ at fixed memory via sliding-window + sink tokens | 4 GPU-h |

**Honest framing:**
- This is **explicitly an operational paradigm, not a new mechanism**. Every probe tests claims that already exist in the catalog.
- It is **not a new axis**.
- It does **not add magnitude** to the cumulative single-GPU stack. The headline figures in the catalog (~6.6M× grounded-reasoning subset, etc.) remain unchanged in claim form. After the campaign they will be either **confirmed**, **refuted**, or **bounded** for the five probed paradigms.
- **Joint Gate-0 PASS / LLM-scale empirical confirmation probabilities are N/A** — this *is* the Gate-0 campaign. The output is per-paradigm PASS/FAIL/PARTIAL signals, not a magnitude claim.
- Engineering is real but bounded: ~700 LOC of probe infrastructure over 2 weeks of engineering, then 5 GPU-days of campaign execution.

**Likely verdict:** **SELECT.** The user's iter-236 brief change is binding; this is the directly-implementing paradigm. Selecting it converts the four prior META-recommendations into action.

---

## 1. Candidate framing and why this is the natural #92

### 1.1 Why this candidate, why now

The iter-236 slate lands one iteration after the user sharpened the brief. Candidates A and C in this slate continue the axis-extension pattern (e.g., another modality-distillation or another architectural-primitive). **This document (B) is the operational slot** — its sole content is a concrete execution plan for the validation pivot the user has now instructed.

The candidate is honestly framed: it cannot win selection on magnitude (it claims none). It wins selection because:
1. The iter-236 brief change is **explicitly binding** ("only test when relevant and make sure we only test for a maximum of 1 day per test"), and binding constraints take priority over magnitude in selection.
2. Sequential top-5 probing with hard 1-day budgets is the lowest-friction translation of the user's words into an execution plan.
3. Every prior META-recommendation (iter-231, iter-232, iter-233, iter-235) was RESERVED-AS-RECOMMENDATION. At iter-236 the user has removed the optionality. Reserving again would now contradict the brief.

### 1.2 Difference from prior META-class artifacts

The four prior META-recommendations (#87, #88, #89, #91) were **strategic-recommendation** class — they advised pivot but produced no execution plan, no probe specs, no schedule, and no LOC estimate. Each was RESERVED.

GATE-0-CAMPAIGN-CHIRON is **operational** class. It contains:
- A specific top-5 list with magnitude-rank justification (§2.1).
- Per-probe Gate-0 specs with PASS criteria, model size, dataset, GPU-hour budget, and explicit fail-fast triggers (§2.2-§2.6).
- Sequential pipeline ordering with stop-conditions (§2.7).
- Engineering scope broken down by infrastructure component (§5).
- A concrete iter-237+ output schema for downstream paradigm selection (§9).

### 1.3 Why "operational" deserves a paradigm slot

A reasonable objection: "this is just an engineering task, not a paradigm." The counter-argument: the program's catalog assigns paradigm slots to **structural commitments that change what iter-N+1 does**. Per the iter-236 brief, the program's next 5 days no longer add paradigms; they validate. That is a structural commitment. Recording it as paradigm #92 makes the commitment legible alongside the other 91 entries and prevents the autonomous-loop from drifting back to design-mode at iter-237 before the campaign finishes.

---

## 2. Mechanism: the campaign itself

### 2.1 Top-5 selection: highest-magnitude unvalidated bold-claim paradigms

The selection rule is mechanical: among paradigms whose headline claim has not been Gate-0'd or LLM-scale empirically confirmed, rank by claimed cumulative-stack contribution and pick the top 5. Result:

1. **#73 PHOENIX-DISTILL-COMBO.** Claim: composing #56 DISTILL-FORWARD with #74 PHOENIX-1BIT under a Llama 3.1 405B teacher gives ~100× wall-clock to fixed final NLL with 18B-effective single-GPU. The 100× is the largest single-paradigm claim post-#68 SUPER-DISTILL relaxation that has not been validated.
2. **#74 PHOENIX-1BIT-DISTILL.** Claim: binary {-1, +1} middle layers + cached-logit distillation give 32B-effective single-GPU at ≤ 0.15 nat loss vs the baseline. The model-size claim is the largest unvalidated memory claim.
3. **#69 REASONING-DISTILL.** Claim: DeepSeek-R1 reasoning-trace distillation gives a billion-multiplier on a narrow causal-reasoning subset benchmark (AIME-25, GSM-Symbolic). The "billion-multiplier" claim is the most extreme narrow-subset figure in the catalog.
4. **#77 MOEFICATION-DISTILL.** Claim: post-hoc 8-way MoE-fication of the PHOENIX-1BIT base gives 256B-effective. The 256B-effective is the largest claimed-effective-size figure in the catalog.
5. **#78 ATTENTION-SINK.** Claim: sliding-window attention with 4 sink tokens lets T grow to 64K-1M at fixed memory with NLL drift ≤ 0.05 nat. The T → ∞ claim is the most extreme context-axis claim.

**Why these five and not others.** The catalog contains other large-magnitude claims (e.g., #65 grounded-reasoning ~6.6M×, #66 cross-modal 5M×), but those compound from claims that themselves have prior Gate-0 PASS evidence (FACE, SLC, RLG, SUPER-DISTILL teacher chain). The five above are the unvalidated load-bearing claims — if any FAILS, the cumulative-stack arithmetic that builds on it must be revised.

### 2.2 Probe #1 — #73 PHOENIX-DISTILL-COMBO (8 GPU-hours)

**Setup.** 66M-parameter coordinator (CHIRON trunk, L=8, d=512, n_head=8); Llama 3.1 70B teacher as Tier-3 fallback (the 405B claim cannot be hosted on the user's hardware; the Tier-3 signal is whether 70B teacher already shows the kinematic shape the 405B claim depends on).

**Pipeline.**
- Cache teacher logits offline on a 50k-step pile-bpe stream (~200M tokens). Top-32 sparse logits per position; ~8 GB cached.
- Student trained 50k steps with the cached-logit distillation loss `L = 0.5·CE + 0.5·KL_distill` per #56.
- Wall budget: 6 GPU-hours train + 1 GPU-hour cache build + 1 GPU-hour evaluation = **8 GPU-hours**.

**PASS criterion.** Final NLL on a held-out 1M-token pile-bpe slice: **NLL_student ≤ NLL_baseline_no_distill − 0.5 nat** at 50k steps. Baseline-no-distill is a 66M trained on identical data with `L = CE` only.

**FAIL signals.**
- NLL_student ≥ NLL_baseline_no_distill: distillation provided no gain at this teacher size.
- NLL_student between baseline and baseline − 0.25 nat: marginal gain, claim of 100× wall-clock unsupported.
- Any divergence in first 5k steps: probe halts; record divergence step and EMA trajectory.

**Why this is fail-fast.** Most of the 100× claim arises from **cached logits eliminating the need to re-run the teacher**, not from any specific teacher size. If the 70B teacher fails to show ≥ 0.5-nat gap at 66M, the per-step compute argument is unchanged but the quality argument fails — the wall-clock-to-fixed-NLL multiplier collapses.

### 2.3 Probe #2 — #74 PHOENIX-1BIT-DISTILL (12 GPU-hours)

**Setup.** Same 66M coordinator, but with QAT (quantization-aware training) for the binary middle layers (L=4 binary, edges BF16, embedding-island BF16). Cached-logit pipeline reused from #73 probe to share infrastructure.

**Pipeline.**
- 1 GPU-hour: graft the BitNet-style XNOR-popcount kernels onto the 66M trunk. Reference: BitNet b1.58 official PyTorch.
- 9 GPU-hours: 50k QAT-distill steps.
- 2 GPU-hours: evaluation (NLL + memory verification).

**PASS criteria (both required).**
- NLL_student ≤ NLL_#73_baseline + 0.15 nat at the same 50k-step count. (Allows the BitNet-published 0.10-0.15 nat loss from binary weights.)
- Peak GPU memory verified: ≤ 60% of the BF16 baseline at the same forward batch.

**FAIL signals.**
- NLL gap > 0.30 nat: binary layers degrade quality more than the BitNet-published bound; the 32B-effective claim's NLL premise fails.
- Memory ≥ 80% of BF16 baseline: PHOENIX-1BIT memory math wrong; 32B-effective claim fails on the memory side.
- QAT divergence in first 10k steps: kernels broken or learning-rate envelope wrong; halt and record.

**Why this is the right probe size.** The smallest scale that exercises both binary and BF16 layers in the same architecture; full PHOENIX-1BIT at 18B is multi-week and multi-GPU. The 66M result transfers to 18B by the per-layer-equivalence theorem in #74's design doc; if it FAILs at 66M, it cannot succeed at 18B.

### 2.4 Probe #3 — #69 REASONING-DISTILL (10 GPU-hours)

**Setup.** 66M coordinator + ~10M tokens of DeepSeek-R1 reasoning traces (publicly distributed; chain-of-thought + final-answer pairs on math/physics/code).

**Pipeline.**
- 1 GPU-hour: download and tokenize the trace corpus.
- 8 GPU-hours: distill on the trace corpus (CE on visible tokens, no PRM in this probe to keep variables minimal).
- 1 GPU-hour: AIME-25 evaluation (run 30 problems with self-consistency-32 sampling).

**PASS criterion.** AIME-25 pass-rate ≥ 30% at 66M post-distill. (For reference: 66M base ~5% on AIME-25; published reasoning-distillation results hit 50-70% at 1.5B; 30% at 66M is the conservative interpolation that supports the directional claim.)

**FAIL signals.**
- AIME-25 < 15%: trace distillation insufficient to lift narrow-reasoning at 66M; the billion-multiplier claim's mechanism fails.
- Trace distillation degrades general NLL by > 1.0 nat: trade-off too steep; claim fails on the orthogonality side.

**Why this is fail-fast and well-bounded.** The "billion-multiplier" framing in #69 is a narrow-subset multiplier (capability that didn't exist at all → capability that exists at meaningful rate). The probe doesn't need to measure 10⁹×; it only needs to confirm the **presence** of the capability above floor at the smallest scale that can exhibit it.

### 2.5 Probe #4 — #77 MOEFICATION-DISTILL (16 GPU-hours, scale-down option)

**Setup.** This is the only probe whose default budget exceeds the 1-day envelope. Two execution options:

- **Default (16 GPU-h, 1.33 days).** 200M-parameter base + post-hoc 8-way MoE conversion + cached-logit distillation. Allows direct comparison to published MoEfication results (which start at ~1B).
- **Scale-down (12 GPU-h, fits 1 day).** 100M-parameter base + post-hoc 4-way MoE. Smaller signal but inside budget.

**Selection rule.** Default unless the prior probe (#69) overran. **The 1-day budget is binding for the default; if Probe #4 is the run that pushes the campaign over budget, scale-down is mandatory.**

**Pipeline (default).**
- 2 GPU-hours: train 200M dense base for 50k steps.
- 1 GPU-hour: post-hoc MoE-fication via FFN clustering.
- 11 GPU-hours: distill the 8-way MoE student on cached teacher logits.
- 2 GPU-hours: evaluation (per-token compute, load balance, NLL).

**PASS criteria (all three required).**
- Per-token active compute ratio: ~25% of dense (i.e., ≈4× saving), within ±10%.
- Load balance entropy: ≥ 0.85 of uniform-8 entropy after 25k distill steps.
- NLL_MoE_student ≤ NLL_dense_distill + 0.10 nat.

**FAIL signals.**
- Per-token compute ratio > 40%: MoE-fication not actually sparsifying compute.
- Load balance entropy < 0.6 of uniform: degenerate routing; the claim's compute argument fails.
- NLL gap > 0.30 nat: MoE-fication degrades quality past the published threshold.

### 2.6 Probe #5 — #78 ATTENTION-SINK (4 GPU-hours)

**Setup.** 200M coordinator with sliding-window attention (window W = 2048) + 4 sink tokens at sequence start. Reference: StreamingLLM (Xiao et al. 2024).

**Pipeline.**
- 0.5 GPU-hour: graft sliding-window + sink kernels into the 200M trunk.
- 2.5 GPU-hours: evaluation pass at T = 64K (synthetic long-doc reconstruction + perplexity drift).
- 1 GPU-hour: memory verification at T = 64K, T = 128K, T = 256K.

**PASS criteria (both required).**
- NLL drift at T = 64K vs T = 4K baseline ≤ 0.05 nat.
- Memory bounded: peak GPU memory at T = 256K ≤ 1.2× memory at T = 4K (sliding-window invariant).

**FAIL signals.**
- NLL drift > 0.15 nat: sink mechanism insufficient to anchor long-context attention.
- Memory growth > 2× from T = 4K to T = 256K: sliding-window plumbing leaking activations.

**Why this is the cheapest probe.** ATTENTION-SINK's mechanism is **inference-time only** (no training). The 4-hour budget is forward-pass and memory measurement, not optimization.

### 2.7 Sequential pipeline and stop-conditions

The campaign runs **sequentially** in the order #73 → #74 → #69 → #77 → #78 (highest-magnitude-first). Sequential is preferred over parallel for three reasons:

1. **Early-PASS unlocks build.** If #73 PASSES, the program can begin building the PHOENIX-DISTILL-COMBO production pipeline without waiting for the rest of the campaign.
2. **Early-FAIL aborts wrong-direction work.** If #73 FAILS, the build effort that would have followed is canceled before any LOC are written; the saving is the entire downstream engineering cost.
3. **Resource-budgeting.** Sequential lets each probe fully use the GPU; parallel would force splitting batches and hurt the per-probe signal-to-noise.

**Stop-conditions.**
- **Hard 1-day-per-probe budget.** If a probe's wall-clock exceeds 24 hours from start, halt; record the partial result; mark the probe INCOMPLETE; move to the next.
- **Hard 5-day campaign budget.** If the total campaign exceeds 5 days, halt before the next probe begins; mark remaining probes UNSCHEDULED.
- **Catastrophic-failure abort.** Divergence in first 10% of a probe's wall budget halts that probe and frees its remaining time for the next probe.

### 2.8 Composition with all 91 prior paradigms

| Paradigm | Composes? | Mechanism |
|---|---|---|
| **All 91 prior** | N/A | GATE-0-CAMPAIGN-CHIRON does not consume nor extend any paradigm; it gates downstream build commitments for the five probed paradigms. |

**No paradigm is invalidated** by the campaign existing. Five paradigms may be invalidated **by the campaign's results** (FAIL outcomes); that is the campaign's purpose.

---

## 3. Theoretical analysis

### 3.1 Theorem 1 — Magnitude preservation

GATE-0-CAMPAIGN-CHIRON adds 0× to the cumulative single-GPU stack. The five probes do not modify the trunk, optimizer, or dataloader of the production stack. They are isolated experiments on probe-only model copies. **Bit-exact preservation across all 26 axes by construction.**

### 3.2 Theorem 2 — Information-value of the campaign

**Claim.** Conditional on running the top-5 sequential campaign, the expected reduction in cumulative-stack-claim uncertainty is greater than the expected reduction from any single iter-236 paradigm-design slot.

**Sketch.** Let p_i be the prior probability of paradigm i's bold claim being true at LLM scale. The Bayesian update from a Gate-0 with sensitivity s_i = P(PASS | true) and specificity 1 − f_i = P(FAIL | false) is monotone in s_i and 1 − f_i. The five probes have:
- s_i ≥ 0.7 (the PASS thresholds are conservative and selected to be sensitive at the probe scale).
- 1 − f_i ≥ 0.7 (the FAIL signals are sharply separated from PASS thresholds; the divergence-abort path additionally guards against false-PASS by infrastructure bug).

By contrast, designing a single iter-236 paradigm and recording its design produces zero Bayesian update on any prior claim. Thus the campaign dominates the design slot in expected information gained per unit user-attention. **This is the load-bearing argument for SELECT.**

### 3.3 Theorem 3 — Sequential dominance over parallel

**Claim.** Under the 1-day-per-probe and 5-day-total constraints, sequential execution dominates parallel.

**Sketch.** Parallel execution on a single GPU forces batch-splitting; effective compute per probe is reduced by a factor of ~5× while wall-time savings approach ~1× (since GPU is the bottleneck). The signal-to-noise ratio of each probe's PASS/FAIL boundary degrades. Sequential execution preserves full per-probe SNR and additionally enables early-PASS-unlocks-build (Theorem 4 below).

### 3.4 Theorem 4 — Early-PASS unlocks downstream build

**Claim.** If probe i PASSES at day i, the production build of paradigm i can begin at day i (not day 5).

**Sketch.** Each probe's PASS criterion is sufficient to justify the production build of that paradigm without further validation. Rank-1 (#73) is the highest-magnitude target; its PASS at day 1 unlocks 4 days of build runway within the 5-day campaign window. Rank-5 (#78) PASSing at day 5 produces zero such unlock, but its 4-hour budget makes it a good campaign-tail entry regardless.

### 3.5 Joint Gate-0 PASS probability — N/A; per-probe priors instead

```
Campaign-level Joint Gate-0 PASS:                    N/A (this IS the Gate-0)
Per-probe prior (subjective, before running):
  #73 PHOENIX-DISTILL-COMBO PASS at 70B teacher:    ~55%
  #74 PHOENIX-1BIT-DISTILL PASS at 66M:             ~50%
  #69 REASONING-DISTILL PASS at AIME-25 30%:        ~40%
  #77 MOEFICATION-DISTILL PASS at 200M:             ~45%
  #78 ATTENTION-SINK PASS at T=64K NLL drift:       ~70%
Expected PASS count (sum):                          ~2.6 / 5
P(≥3 of 5 PASS):                                   ~30%
P(≥1 of 5 PASS):                                   ~95%
P(Rank-1 #73 PASS):                                ~55%
```

The campaign delivers high information regardless of PASS count: every FAIL is a corrective signal that prevents wrong-direction build, and every PASS unlocks immediate build.

---

## 4. Updated cumulative stack

```
Iter 235 close (post-#91):
  All 25 prior axes ≈preserved
  No new compute multiplier added in last 4 iterations (all META-class)

Iter 236 (GATE-0-CAMPAIGN-CHIRON if SELECT):
  All 25 prior axes ≈preserved (no executable trunk change)
  No new axis added
  Magnitude: 0× (META-paradigm)
  Strategic: top-5 unvalidated bold-claim paradigms become Gate-0'd within 5 days.

Iter 237+ (post-campaign):
  Cumulative stack figures involving #73, #74, #69, #77, #78
   are either CONFIRMED, REFUTED, or BOUNDED based on probe outcomes.
  All other axes' figures unchanged.
```

The cumulative-stack number itself does not change at iter-236. After the campaign closes, the stack number may be **refined downward** for paradigms that FAIL — that is the campaign's intended corrective action.

---

## 5. Engineering scope

| Component | LOC | Weeks |
|---|---|---|
| Cached-logit pipeline (probes #1, #2, #4) | ~250 | 0.7 |
| BitNet b1.58 kernel graft (#2) | ~150 | 0.5 |
| Reasoning-trace dataloader + AIME-25 eval harness (#3) | ~120 | 0.4 |
| Post-hoc MoE-fication script + load-balance metric (#4) | ~100 | 0.3 |
| Sliding-window + sink-token kernel (#5) | ~80 | 0.2 |
| Probe orchestration (sequential runner with budget enforcement) | ~50 | 0.1 |
| **Total infrastructure** | **~750 LOC** | **~2.2 weeks** |
| **Campaign execution (after infrastructure)** | **0 new LOC** | **5 GPU-days** |

**Total user-facing wall time:** ~3 weeks (2 weeks infrastructure + 1 week campaign + slack).

The infrastructure builds on existing references (Tri Dao FA-3, BitNet b1.58 PyTorch, StreamingLLM, MoEfication-Plus). No probe requires research-novel kernels.

---

## 6. Memory advantage preservation

| Component | GPU memory |
|---|---|
| GATE-0-CAMPAIGN-CHIRON probes (each isolated) | varies per probe; capped at 16 GB per design |
| Production trunk during campaign | unchanged |
| **Total additional GPU memory on production trunk** | **0 MB** |

Per-probe memory:
- #73: cached logits live on disk, not GPU; runtime peaks at ~10 GB for 66M + cached-logit decoder.
- #74: peak ~6 GB at 66M with 1-bit middle layers (PHOENIX memory verified by probe).
- #69: peak ~11 GB for 66M + reasoning-trace forward.
- #77: peak ~14 GB for 200M with cached-logit pipeline.
- #78: peak ~12 GB at 200M with T=64K sliding window.

**Single-GPU 16 GB ceiling preserved exactly** for every probe.

---

## 7. Gates

### Gate-0: N/A for the campaign itself

There is no probe-mechanism to Gate-0 the campaign — the campaign IS the Gate-0 layer. Replaced by:

### Gate-Schedule (executable)

**Probe.** Day 0 sanity check that all five probe specs run end-to-end in < 30 minutes on toy data.

**PASS.** All five probes start, run one mini-batch, write a metric, and exit cleanly. ≤ 30 minutes total wall.

**FAIL action.** Halt before campaign begins; debug infrastructure; do not consume the 5-day budget on infrastructure bugs.

### Gate-Per-Probe (campaign body)

Each probe has its own PASS/FAIL spec from §2.2-§2.6. Outputs are recorded into a campaign log for downstream paradigm selection.

### Gate-Campaign-Close (iter-237 input)

**Probe.** After day 5, summarize per-probe outcomes into a single iter-237-readable artifact:
- Confirmed claims (PASS).
- Refuted claims (FAIL with bound).
- Bounded claims (PARTIAL: between PASS and FAIL).
- Unscheduled probes (campaign-budget overrun).

**PASS criterion (campaign-level):** at least 3 of 5 probes complete (PASS or FAIL). At-least-3 ensures the campaign produced enough corrective signal to inform iter-237; below 3, the campaign is judged INCOMPLETE and a recovery plan is recorded.

---

## 8. Honest gaps

1. **This is an operational paradigm, not a new mechanism.** No new theorem, kernel, or training algorithm is contributed. The verdict will reflect this honestly.

2. **The "1-day-per-probe" envelope is binding but tight.** Probe #4 (MOEFICATION) at default 200M scale exceeds 1 day; the scale-down option to 100M+4-way is the in-budget alternative. The campaign cannot fully test MOEFICATION's 256B-effective claim within the constraint; the probe at 100M tests the **mechanism's presence**, not the **scaling claim**.

3. **The Llama 3.1 405B teacher cannot be hosted on user hardware.** Probe #1 uses 70B as a Tier-3 fallback. PASS at 70B is necessary but not sufficient for the 405B claim; the 405B-specific kinematic shape is untestable on this hardware. This is recorded explicitly in §2.2's "FAIL signals."

4. **Probe-scale to LLM-scale extrapolation is non-trivial.** Each probe is at 66M-200M, but the bold claims are at 1.84B-256B-effective. The probe-scale results inform the LLM-scale claim by mechanism-presence inference, not by direct verification.

5. **Sequential pipeline foregoes hedging.** If the autonomous-loop runtime is interrupted between probes, the campaign halts mid-flight. There is no checkpointing across probes. Worst case: only the first 2 probes complete in 5 elapsed days due to runtime interruptions outside the campaign's control.

6. **Per-probe priors in §3.5 are subjective.** The 30% P(≥3 of 5) figure depends on the prior estimates, which were chosen by reading the original paradigm design docs. A reviewer with a different prior model could compute different probabilities.

7. **Campaign does not validate the meta-claim that GATE-0 campaigns are worth running.** The campaign justifies its own information value by Theorem 2's expected-update argument, not by an empirical track record at this scale. Future iterations with multiple completed campaigns could measure this empirically.

8. **Five probes is a small-N sample.** The catalog has many more unvalidated paradigms; the top-5-by-magnitude rule selects the highest-stakes set, but stake-weighted selection over-emphasizes bold-claim correctness vs. modest-claim aggregate truth.

9. **No probe targets the META paradigms (#87-#91) themselves.** The four META-recommendations are not Gate-0-able by construction; this campaign cannot validate that the validation-pivot was correct, only that the underlying paradigms it validates were/were not correct.

10. **Iter-237's paradigm-design lane is implicitly suspended.** The campaign consumes 5 days of the autonomous-loop's iteration runway. This document does not explicitly say what happens at iter-237 if the campaign is mid-flight. Implicit answer: iter-237 produces a campaign-status update, not a new paradigm; the next paradigm slot is iter-242 at earliest. If the user wants design to resume sooner, that is an explicit signal.

---

## 9. Bottom line

**GATE-0-CAMPAIGN-CHIRON is the natural #92 candidate-B — operational implementation of the iter-236 brief change.** It:
- **Translates the binding constraint** ("test before we build off; max 1 day per test") into a concrete 5-day plan.
- **Selects the top-5 unvalidated bold-claim paradigms** (#73, #74, #69, #77, #78) by claimed-magnitude rank.
- **Specifies per-probe Gate-0 PASS/FAIL criteria** with hard 1-day budgets and fail-fast triggers.
- **Sequences the probes highest-magnitude-first** so early-PASS unlocks build and early-FAIL aborts wrong-direction work.
- **Adds 0× magnitude** to cumulative stack — explicitly not a magnitude claim. Validates existing magnitudes.
- **Engineering: ~750 LOC over ~2.2 weeks of infrastructure + 5 GPU-days of campaign execution.**

**Cumulative single-GPU stack at iter-236 close (regardless of verdict):**
- All 25 prior axes ≈preserved.
- No new axis added.
- No magnitude change at iter-236; magnitude **claims** for #73, #74, #69, #77, #78 will be confirmed/refuted/bounded after the campaign.

**Likely verdict:** **SELECT.** The iter-236 brief change is binding; this is the directly-implementing paradigm. Reserving it for the fifth time would contradict the user's explicit instruction. Selecting it converts the four prior META-recommendations (#87, #88, #89, #91) into action. Per-probe outcomes feed iter-237+ paradigm selection; the autonomous-loop's design lane is implicitly suspended for the campaign's 5-day duration and resumes at iter-242 at earliest, or earlier on explicit user signal.

**A and C dispositions (this slate):**
- **A** (continued axis-extension or architectural-primitive paradigm): if selected instead, contradicts the iter-236 brief change; this document then becomes a counter-record acknowledging the contradiction.
- **C** (new META-recommendation in the spirit of #87-#91): if selected instead, produces another reservation-class artifact — same content as the four prior META-recommendations but no execution; declines the brief's call for binding action.
- **B (this document):** SELECT. Operational implementation; binding-constraint discharge.

After 91 paradigms with 25 axes covered, the user has now drawn the line: validate before building. This document executes that line. **The recommendation, in one sentence:** Run the top-5 highest-magnitude unvalidated Gate-0 probes sequentially with hard 1-day-per-probe budgets, total 5 days, then let iter-237 paradigm selection use the actual outcomes.
