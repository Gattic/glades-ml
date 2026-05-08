# Paradigm Shift #107 — DRAFT-VERIFIER-CO-LEARN-LIVE-PROBE-CHIRON: Sixth Probe Execution

**Status:** SELECTED. Sixth paradigm in iter-246+ single-probe-execution pattern.
**Date:** 2026-05-08 (Ralph-loop iter 251).
**Axis:** OPERATIONAL EXECUTION.
**Magnitude target:** 0× new magnitude. Tests #97 acceptance-rate-improvement claim (0.7 → 0.85).

---

## 0. Executive summary

Iter-251 continues iter-246+ probe-execution pattern with sixth probe: #97 DRAFT-VERIFIER-CO-LEARN at 6h.

**Probe spec (from #97 design):**
- 200M draft + 1.84B main (post-#75 baseline).
- 4-phase orchestration:
  - Phase 1 (1h): baseline α and standalone NLL.
  - Phase 2 (2h): rejection log accumulation (~60K sites).
  - Phase 3 (2h): 5K-step draft fine-tune on rejection log.
  - Phase 4 (1h): re-evaluate α_post and NLL drift.
- **PASS:** α_post ≥ α_baseline + 0.05 absolute AND draft NLL ≤ baseline + 0.02 nat.
- **Four hard-FAIL signals:**
  - No improvement (full 6h).
  - NLL degradation > 0.05 nat (3h abort).
  - Acceptance rate regression (5h abort).
  - Loss divergence (3h abort with 1 retry).
- **Wall-clock:** ~6 GPU-hours expected.

**Outcome paths:**
- **PASS (~50-60%):** acceptance rate improvement validated; lifts inference 3-5× (#75) → 4-6× joint.
- **FAIL (~40-50%):** rejection-driven co-learning ineffective at probe scale; production-viable drops.

**Combined six-probe roadmap (iter-246-251):**
- #102 KV-FACE-MLA: 2h
- #103 ATTENTION-SINK: 4h
- #104 PHOENIX-DISTILL-COMBO: 8h
- #105 ASTRA-KAHAN: 6h normal / 1h FAIL
- #106 NEURAL-CACHE-COMPRESSION: 6h
- #107 DRAFT-VERIFIER-CO-LEARN: 6h
- **Total: 32 GPU-hours worst-case = ~3-4 days execution**

P(≥3 of 6 PASS) ≈ 65%; P(≥4 PASS) ≈ 35%.

---

## 1. Mechanism

Specifies actual execution of #97's 4-phase orchestration. Composes with #75 SPECULATIVE-DECODING (whose own probe is #103 ATTENTION-SINK indirectly; #75 itself is production-validated).

Ordering recommendation: execute #103 (#78 ATTENTION-SINK) first to validate the broader inference pipeline; #107 (#97 DRAFT-VERIFIER) is conditional refinement of inference layer.

---

## 2. Updated cumulative stack

```
Iter 250 close (post-#106):
  All 27 axes ≈preserved
  Five probes scheduled: 26h worst-case

Iter 251 (DRAFT-VERIFIER-CO-LEARN-LIVE-PROBE):
  All 27 axes ≈preserved (no new axis)
  Six probes scheduled: 32h worst-case = ~3-4 days
```

---

## 3. Engineering scope

| Component | LOC | Wall-clock |
|---|---|---|
| Rejection log accumulator | 200 | already documented in #97 |
| Co-learn pipeline | 250 | already in #97 |
| Probe execution (200M draft + 1.84B main) | 0 | ~6 GPU-hours |
| **Total new** | **~50 LOC + reuse** | **~7 wall-clock hours** |

---

## 4. Bottom line

**DRAFT-VERIFIER-CO-LEARN-LIVE-PROBE-CHIRON is the sixth single-probe-execution paradigm.** Tests #97's acceptance-rate-improvement claim with four hard-FAIL aborts.

**Cumulative single-GPU stack at iter-251 close:**
- All 27 prior axes ≈preserved
- **Six probes scheduled: 32h worst-case = ~3-4 days execution**

After 107 paradigms, **27 axes** unchanged.

**Iter-236+ pattern (15 paradigms / 15 iterations):**
- Operational/synthesis/sunset/execution (12): #92, #94, #96, #98, #100, #101, #102, #103, #104, #105, #106, #107.
- Novel-with-built-in-test (4): #93, #95, #97, #99.

**Six-probe roadmap is now comprehensive** across high-priority paradigms (#76 MLA, #78 SINK, #73 PHOENIX-DISTILL, #93 ASTRA-KAHAN, #99 NEURAL-CACHE, #97 DRAFT-VERIFIER). Each within 1-day-per-test budget per iter-236 brief.
