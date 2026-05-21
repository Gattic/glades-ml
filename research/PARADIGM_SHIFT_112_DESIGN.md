# Paradigm Shift #112 — GATE-1-SPECIFICATION-CHIRON: Full-Scale Validation Protocol

**Status:** SELECTED. First Gate-1-specification paradigm. Extends iter-246+ Gate-0 probe roadmap to full-scale 1.84B validation.
**Date:** 2026-05-08 (Ralph-loop iter 256, post-#111 ten-probe roadmap completion).
**Axis:** OPERATIONAL EXECUTION — Gate-1 (full-scale) specification.
**Magnitude target:** 0× new magnitude. Specifies how to validate Gate-0-PASSing paradigms at production scale.

---

## 0. Executive summary

Iter-256 pivots from Gate-0 specification (iter-246-255 ten-probe roadmap) to Gate-1 specification. **Gate-1 = full-scale 1.84B validation; ~150 GPU-hours per paradigm; ~6.25 wall-clock days per Gate-1.**

**Rationale:**
- Iter-236 brief specifies "max 1 day per test" — applies to Gate-0 probes (≤24h budget each).
- Gate-1 (full-scale) is post-Gate-0; effectively a "Phase 2" of the implementation roadmap from #100 synthesis.
- Gate-1 needs different framing: ~6.25 days per paradigm exceeds the 1-day-per-test rule, so Gate-1 should be selectively scheduled for Gate-0-PASSing paradigms only.

**Gate-1 protocol per paradigm (post-Gate-0 PASS):**
- 1.84B trunk; full training run to fixed FLOPs target.
- Composition with all stack-base paradigms (#42, #44, #66, #68, #74, #76, #78).
- Full benchmark suite (Pile NLL + downstream + memory verification).
- ~150 GPU-hours = ~6.25 wall-clock days on RTX 4080 SUPER.

**Ten-probe Gate-1 budget (if all 10 PASS Gate-0):**
- 10 × ~150 GPU-hours = 1500 GPU-hours = ~62.5 wall-clock days.
- **Selective Gate-1**: only top-3 Gate-0-PASS paradigms (~450 GPU-hours = ~19 days).

**Gate-1 ordering (per #100 synthesis priority):**
- Tier 1: highest-magnitude validated at Gate-0 (probably #69 REASONING, #73 PHOENIX-DISTILL-COMBO, #78 ATTENTION-SINK).
- Tier 2: composition-confirmed (e.g., #74 PHOENIX-1BIT post-#73 PASS).
- Tier 3: speculative (#77 MOEFICATION, #95 MULTI-TEACHER-ROUTING).

**Honest framing:**
- 0× new magnitude (operational paradigm).
- Gate-1 budget is the dominant program cost (~62.5 days vs Gate-0's 8 days).
- Selective Gate-1 (top-3) is realistic budget (19 days = under 1 month).
- iter-236 brief's "max 1 day per test" interpretation: each Gate-1 is multiple days, but the Gate-0 probe already validated the paradigm at 1-day budget; Gate-1 is "build" not "test".

---

## 1. Mechanism

### 1.1 Gate-1 protocol (per paradigm)

1. **Pre-condition:** Gate-0 PASS at probe scale (66M-200M).
2. **Setup:** integrate paradigm into 1.84B production CHIRON stack.
3. **Training:** fixed-FLOPs target equivalent to ~150 GPU-hours.
4. **Evaluation:**
   - Pile NLL on validation (vs from-scratch 1.84B baseline).
   - MMLU, HellaSwag, ARC-Challenge, GSM8K, AIME (per paradigm priority).
   - Memory verification (≤16 GB ceiling).
5. **PASS criteria:** per-paradigm Gate-1 spec (already documented in respective design docs).

### 1.2 Selective Gate-1 (recommended)

**Top-3 priority order:**
1. #69 REASONING-DISTILL (highest magnitude claim; 1B× threshold).
2. #73 PHOENIX-DISTILL-COMBO (highest model-size claim; 18B effective).
3. #78 ATTENTION-SINK (highest production precedent; T → ∞ capability).

**Cost: ~450 GPU-hours = ~19 wall-clock days on RTX 4080 SUPER.**

---

## 2. Updated cumulative stack

```
Iter 255 close (post-#111):
  All 27 axes ≈preserved
  Ten-probe Gate-0 roadmap complete (78h worst-case)

Iter 256 (GATE-1-SPECIFICATION-CHIRON):
  All 27 axes ≈preserved
  Gate-1 protocol specified for Gate-0-PASSing paradigms
  Recommended top-3 Gate-1: ~450h = ~19 days
```

---

## 3. Engineering scope

| Component | LOC | Wall-clock |
|---|---|---|
| Gate-1 protocol documentation | ~0 (design only) | ~1 day writing |
| Per-paradigm Gate-1 PASS criteria (already in design docs) | 0 (reuse) | 0 |
| Top-3 priority ordering rationale | ~0 | ~0.5 day |
| **Total** | **~0 LOC** | **~1.5 wall-clock days** |

---

## 4. Bottom line

**GATE-1-SPECIFICATION-CHIRON formalizes the post-Gate-0 validation protocol.** Selective Gate-1 (top-3) costs ~19 days; full Gate-1 (all 10) costs ~62.5 days.

**Cumulative single-GPU stack at iter-256 close:**
- All 27 prior axes ≈preserved
- Gate-0 roadmap: 10 probes; ~8 days execution
- Gate-1 selective: top-3 paradigms; ~19 days execution
- **Combined Gate-0 + selective Gate-1: ~27 days = ~1 month**

After 112 paradigms, **27 axes** unchanged.

**Iter-236+ pattern (20 paradigms / 20 iterations):**
- Operational/synthesis/sunset/execution (17): #92, #94, #96, #98, #100-#112.
- Novel-with-built-in-test (4): #93, #95, #97, #99.

**Two-phase validation roadmap now complete:**
- **Phase 1 (Gate-0):** ten probes at probe scale; ~8 days.
- **Phase 2 (Gate-1):** top-3 Gate-0-PASS at full scale; ~19 days.
- **Total: ~27 days = ~1 month execution.**

This is the program's first comprehensive end-to-end validation specification. **Iter-257+** can either continue specification work (e.g., Phase 3 production deployment) or pivot to actual execution.
