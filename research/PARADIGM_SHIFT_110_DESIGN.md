# Paradigm Shift #110 — MULTI-TEACHER-ROUTING-LIVE-PROBE-CHIRON: Ninth Probe Execution

**Status:** SELECTED. Ninth paradigm in iter-246+ single-probe-execution pattern.
**Date:** 2026-05-08 (Ralph-loop iter 254).
**Axis:** OPERATIONAL EXECUTION.
**Magnitude target:** 0× new magnitude. Tests #95 MULTI-TEACHER-ROUTING claim definitively.

---

## 0. Executive summary

Iter-254 continues iter-246+ probe-execution pattern with ninth probe: #95 MULTI-TEACHER-ROUTING at 8h.

**Probe spec (from #95):**
- 200M coordinator + 10M-param 5-class classifier (math/reasoning, code, general, vision, audio held-out).
- Cached-logit teachers: R1-Distill-Qwen-32B + DeepSeek-Coder-V2-Lite-16B + Llama-3.1-8B-Instruct + Llama-3.2-Vision-11B (Tier-3 proxies).
- ~10M training samples partitioned across 5 classes.
- 50k-step argmax routing.
- **PASS:** Per-class NLL ≥ single-best-teacher AND gradient std-dev within 1.2× single-teacher AND classifier accuracy ≥ 85%.
- **Three hard-FAIL signals:** classifier collapse (one class >80%; abort 2h), per-class NLL >0.5 nat worse (abort 4h), gradient std-dev >2× single-teacher (abort 3h).
- **Wall-clock:** ~8 GPU-hours.

**Outcome paths:**
- **PASS (~45-55%):** classifier-routing refines #70-B's variance issues; 2-3× per-class incremental over single-best-teacher.
- **FAIL (~45-55%):** routing variant closes (TEACHER PORTFOLIO meta-channel revisitation suspended).

**Combined nine-probe roadmap (iter-246-254):**
- #102 KV-FACE-MLA: 2h
- #103 ATTENTION-SINK: 4h
- #104 PHOENIX-DISTILL-COMBO: 8h
- #105 ASTRA-KAHAN: 6h / 1h FAIL
- #106 NEURAL-CACHE-COMPRESSION: 6h
- #107 DRAFT-VERIFIER-CO-LEARN: 6h
- #108 PHOENIX-1BIT-DISTILL: 12h
- #109 REASONING-DISTILL: 10h
- #110 MULTI-TEACHER-ROUTING: 8h
- **Total: 62 GPU-hours worst-case = ~6-7 days execution**

P(≥4 of 9 PASS) ≈ 60%; P(≥6 PASS) ≈ 30%.

---

## 1. Mechanism

Specifies actual execution of #95 Gate-0 via 4 cached-logit teacher pipelines + 5-class classifier training. Reuses cached-logit infrastructure from #102/#103/#104/#107/#109.

**Composition:** Independent of other probes; routing-axis orthogonal to model-size/KV/attention probes.

---

## 2. Updated cumulative stack

```
Iter 253 close (post-#109):
  All 27 axes ≈preserved
  Eight probes scheduled: 54h worst-case

Iter 254 (MULTI-TEACHER-ROUTING-LIVE-PROBE):
  All 27 axes ≈preserved (no new axis)
  Nine probes scheduled: 62h worst-case = ~6-7 days execution
```

---

## 3. Engineering scope

| Component | LOC | Wall-clock |
|---|---|---|
| 5-class classifier training (per #95 design) | ~250 | already documented |
| Per-class teacher integration (4 Tier-3 teachers) | ~400 | already documented |
| Probe execution (200M, 50k steps, 10M samples) | 0 | ~8 GPU-hours |
| **Total new** | **~50 LOC + reuse** | **~9 wall-clock hours** |

---

## 4. Bottom line

**MULTI-TEACHER-ROUTING-LIVE-PROBE-CHIRON is the ninth single-probe-execution paradigm.** Tests #95 with three hard-FAIL aborts.

**Cumulative single-GPU stack at iter-254 close:**
- All 27 prior axes ≈preserved
- **Nine probes scheduled: 62h worst-case = ~6-7 days execution**

After 110 paradigms, **27 axes** unchanged.

**Iter-236+ pattern (18 paradigms / 18 iterations):**
- Operational/synthesis/sunset/execution (15).
- Novel-with-built-in-test (4): #93, #95, #97, #99.

**Nine-probe roadmap covers all major claim categories at probe scale.** Total ~6-7 days execution; remaining unscheduled paradigm is #77 MOEFICATION-DISTILL at 16h (largest probe budget; speculative compounding-risk).
