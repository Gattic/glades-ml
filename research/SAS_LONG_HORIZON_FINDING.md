# SAS Long-Horizon Divergence Finding

**Date:** 2026-04-24 (Ralph-loop iter 167)
**Status:** SAS paradigm #40 has long-horizon stability issue. Documented
+ proposed fix.

---

## 1. Problem

At 66M × 10000 steps with SAS α=0.3 + FACE β=0.99 (horizon-safe recipe):

| Step | EMA | ||g|| | scale |
|------|:---:|:-----:|:-----:|
| 2000 | 8.43 | 1.80 | 0.56 |
| 4000 | 8.01 | 2.17 | 0.46 |
| 6000 | **11.96** | **9.11** | 0.11 |
| 8000 | 8.30 | 1.21 | 0.83 |
| 10000 | **27.61** | **2042** | 0.00 |

Training diverged around step 6000, partially recovered, then blew up
completely by step 10000.

## 2. Analysis

At α=0.3, each layer gets updated ~30% of steps. Over 10000 steps, each
layer sees ~3000 updates. That's sufficient in itself.

**Hypothesis — the problem is GRADIENT ACCUMULATION over inactive layers.**
When layer l is inactive for many consecutive steps, its Adam m, v state
becomes stale relative to the active parts of the model. When l is
finally activated, its stored m, v drive a big (stale-gradient-based)
update that doesn't match the current model state.

This is similar to the iter 138 SLC issue: FACE's long-memory EMA
became stale when training dynamics shifted.

## 3. Proposed fixes

### Fix A (simplest): α-schedule ramp
Apply α as a curriculum, similar to SLC's T schedule:
- Phase 1 (first 40% of steps): α=0.1-0.3 (aggressive skip for speed)
- Phase 2 (middle 20% of steps): α=0.5 (transition)
- Phase 3 (last 40% of steps): α=1.0 (full attention for convergence)

At 10000 steps: phases at 0-4000, 4000-6000, 6000-10000.

### Fix B: reset Adam state for newly-active layers
When a layer transitions from inactive → active, RESET its m, v to zero.
Forces fresh gradient-based updates, avoiding stale state.

### Fix C: tighter gradient clipping under SAS
Scale clip threshold down when many layers inactive. If α=0.3, the
active layers' gradients might be effectively "for the whole network"
— tighter clipping prevents accumulated blowup.

## 4. Short-horizon safety confirmed

At 2500 steps, all α values (1.0, 0.7, 0.5, 0.3, 0.1) were stable.
At 1.84B × 2500, SAS flagship delivered 6.33× speedup.

**SAS is safe for 2500-step training runs.** Long-horizon (≥5000 steps)
requires a fix.

## 5. Updated recipe guidance

**Safe regime (≤2500 steps):**
```
--sas-alpha 0.1     # 6× speedup
--sas-alpha 0.3     # 4× speedup, best EMA
--sas-alpha 0.5     # 3× speedup, conservative
```

**Unsafe regime (≥5000 steps):**
Use fix A (α schedule) or fix B (Adam reset). Not yet implemented.

## 6. Research-program implication

SAS at α=0.1 delivers 6.33× wall-clock speedup at 1.84B × 2500.
This is the MAGNITUDE-LEVEL result the Ralph-loop brief asked for.
Long-horizon extension requires an α-schedule mechanism (fix A) —
next design iteration.

Current status:
- ≤2500 step horizon: SAS production-ready
- ≥5000 step horizon: needs α-schedule fix (paradigm #40b)

Honest research accounting: iter 166's 6.33× is at 2500-step horizon.
Long-horizon safety is an unresolved boundary requiring one more
design iteration.
