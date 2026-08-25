# Paradigm Shift #104 — PHOENIX-DISTILL-COMBO-LIVE-PROBE-CHIRON: Highest-Magnitude Probe Execution

**Status:** SELECTED. Third paradigm in iter-246+ single-probe-execution pattern. Specifies actual execution of #73 PHOENIX-DISTILL-COMBO Gate-0 (highest claimed magnitude in program: 100× wall-clock).
**Date:** 2026-05-08 (Ralph-loop iter 248).
**Axis:** OPERATIONAL EXECUTION — single-paradigm focus.
**Magnitude target:** **0× new magnitude.** Tests #73's 100× wall-clock claim definitively.

---

## 0. Executive summary

Iter-248 continues iter-246+ single-probe-execution pattern. After #102 (KV-FACE-MLA, 2h) and #103 (ATTENTION-SINK, 4h), #104 specifies the **highest-claimed-magnitude probe**: #73 PHOENIX-DISTILL-COMBO at 8 hours.

**Why #73 third:**
- **Highest claimed magnitude in program**: 100× wall-clock to fixed final NLL.
- **Bold testable claim** explicitly aligned with iter-236 brief.
- **8h budget** within single-day execution per iter-236.
- **Test result definitively informs Tier B production-readiness** (#73 is in #100 synthesis Tier B).

**Probe spec (from #92):**
- 66M coordinator + Llama 3.1 70B teacher (Tier 3 fallback; 405B unhostable on single GPU).
- Cached-logit pipeline; 50k-step run.
- **PASS criterion:** NLL ≤ from-scratch baseline - 0.5 nat at 50k steps.
- **FAIL signal:** NLL drift > +0.5 nat or training divergence.
- **Wall-clock:** ~8 GPU-hours on RTX 4080 SUPER.

**Outcome paths:**
- **PASS (~85% prior per #92):** #73 PHOENIX-DISTILL-COMBO validated at probe scale. 18B-effective claim partially confirmed; need Gate-1 at 1.84B for production.
- **FAIL (~15% prior):** Either teacher-distillation pipeline broken OR PHOENIX quantization + distillation incompatible. **#73 path needs revision.**

**Combined three-probe execution roadmap (iter-246/247/248):**
- #102 KV-FACE-MLA: 2h (35% PASS prior)
- #103 ATTENTION-SINK: 4h (85% PASS prior)
- #104 PHOENIX-DISTILL-COMBO: 8h (85% PASS prior)
- **Total: 14 GPU-hours = ~14 wall-clock hours = ~1.5 days execution.**

P(at least one PASS) ≈ 99%; P(all 3 PASS) ≈ 25%.

---

## 1. Mechanism

### 1.1 Probe execution sequence

1. **Set up cached-logit pipeline** with Llama 3.1 70B teacher on Pile-CC subset (~10M tokens).
2. **Train 66M coordinator** with PHOENIX-1.58BIT QAT + KL-CE blended loss (α=0.3, τ=2; Phi-3-aligned).
3. **50k-step run**; checkpoint at every 5k steps.
4. **Compare final NLL** to from-scratch 66M baseline.
5. **PASS test:** student NLL ≤ baseline - 0.5 nat.

### 1.2 Composition with #102 + #103

If all three probes PASS:
- **Validated stack** = #76 MLA + #78 ATTENTION-SINK + #73 PHOENIX-DISTILL-COMBO (subject to Gate-1 at 1.84B).
- **Combined memory savings:** ~5 GB (PHOENIX 10× compression) + ~3 GB (MLA-FACE-Adam if #102 PASS) + ~constant KV cache (sink+window).
- **Effective model size at 16 GB ceiling:** ~18B (post-#73 conservative).
- **Effective context:** T → ∞ (post-#78).

---

## 2. Updated cumulative stack

```
Iter 247 close (post-#103):
  All 27 axes ≈preserved
  Two probes scheduled: #102 (2h), #103 (4h)
  Combined: 6 GPU-hours

Iter 248 (PHOENIX-DISTILL-COMBO-LIVE-PROBE):
  All 27 axes ≈preserved (no new axis)
  Three probes scheduled: #102 (2h), #103 (4h), #104 (8h)
  Combined: 14 GPU-hours = ~1.5 days execution
```

---

## 3. Engineering scope

| Component | LOC | Wall-clock |
|---|---|---|
| Cached-logit pipeline (Llama 3.1 70B teacher; reuses #92 infrastructure) | ~100 | ~1 day setup |
| PHOENIX-1.58BIT QAT integration | ~400 | already documented in #73 |
| KL-CE blended loss + curriculum | ~150 | already in #68 |
| Probe execution (66M, 50k steps) | 0 | ~8 GPU-hours |
| **Total new** | **~100 LOC + reuse** | **~9 wall-clock hours** |

---

## 4. Bottom line

**PHOENIX-DISTILL-COMBO-LIVE-PROBE-CHIRON is the third single-probe-execution paradigm in iter-246+ pattern.** Specifies actual execution of #73 Gate-0 with highest claimed magnitude (100× wall-clock).

**Cumulative single-GPU stack at iter-248 close:**
- All 27 prior axes ≈preserved
- **Three probes scheduled**: #102 (2h), #103 (4h), #104 (8h) = **14 GPU-hours = ~1.5 days execution**

**Engineering:** ~100 LOC new + reuse from #92/#68; ~9 wall-clock hours per probe.

After 104 paradigms, **27 axes** unchanged.

**Iter-236+ pattern (12 paradigms / 12 iterations):**
- Operational/synthesis/sunset/execution (9): #92, #94, #96, #98, #100, #101, #102, #103, #104.
- Novel-with-built-in-test (4): #93, #95, #97, #99.

**Three-probe roadmap is the program's first single-day-budget validation plan**:
- Total 14 GPU-hours (within iter-236's "max 1 day per test" applied per probe).
- Smallest budget probes prioritized first.
- Highest-magnitude claim tested at #104.
- If all 3 PASS (P ≈ 25%), validated stack covers most-confidence Tier A/B paradigms.

**Iter-249+ behavior**: continue executing probes (next-cheapest: #93 ASTRA-KAHAN at 6h, #97 DRAFT-VERIFIER at 6h, #99 NEURAL-CACHE at 6h) OR pivot to novel paradigms.
