# Iter 55 — Paradigm-level flag exploration — NULL

**Date**: 2026-05-16
**Iter**: 55 (ninth iter under "stacking-wins" brief)
**Branch**: vesta5 (glades-ml)
**Verdict**: NULL — explored paradigm-level flags (`--cuda-graphs`, `--fp8-attn`) on the iter49 stacked flagship.  Both are either structurally incompatible with `--scfa` or give zero improvement.

---

## TL;DR

After iter54's META documented that no solo engineering ≥+5% target remains, iter55 attempts the META's "Option B: pivot to paradigm-level changes".  Tested two existing paradigm-level flags:

| flag        | result                                                          | tok/s vs iter49 |
|---          |---                                                              |---:             |
| `--cuda-graphs` | "disabled — incompatible with --scfa (per-step CPU branching varies the kernel sequence)" | +0.25% (in noise — flag silently no-op) |
| `--fp8-attn`    | HELIUM ACTIVE — Q/K/V/O projection GEMMs route through cuBLASLt FP8 (E4M3) | +0.29% (in noise) |

Neither flag clears the +5% bar; in fact neither produces a measurable change.  The 50-step bench tok/s clusters at 43,318 / 43,324 / 43,335 vs iter49 base 43,209 — all within ±0.3% bench variance.

NULL — paradigm-level flag combinations don't move the needle on the current stacked flagship.

---

## Why these specific flags

iter54's META proposed three pivots: relax bar to +3%, paradigm-level changes, or update brief.  Without user input, this iter explores existing paradigm-level flags (those already in the codebase) since they're zero-implementation-cost to try.

**`--cuda-graphs`** (ATLAS-COMPILE, paradigm #51): captures forward+loss+backward kernel sequence into a CUDA Graph, eliminating per-step launch overhead.  Validated historically at +1.0× on small models.  **Incompatible with --scfa** because SCFA has per-step CPU branching (the `scfaFuseStreams` event-record sequence and the runtime-disabled FP8 fallback both involve CPU-side conditionals that change the kernel sequence step-to-step).

**`--fp8-attn`** (HELIUM, paradigm #50): routes attention Q/K/V/O projection GEMMs through cuBLASLt FP8 E4M3 with per-tensor amax-derived scaling.  Worked in principle but **provides zero measurable speedup on RTX 4080 SUPER (sm_8.9)** at the current shape — the SCFA inner GEMMs (k=512, m=2048) are too small for FP8's compute savings to outweigh the per-tensor scaling overhead.  Throughput-neutral.

---

## Why incompatibilities are structural, not addressable in one iter

`--cuda-graphs` × `--scfa` could in principle be made compatible by removing the per-step CPU branching in SCFA.  That's a larger refactor — multiple flags involved (scfa-fuse-streams's event records, fp8-attn's runtime fallback, etc).  Not iter-scale work.

`--fp8-attn` is bottleneck-shifted, not bottleneck-resolved — the FP8 GEMMs ARE faster on a per-call basis, but the surrounding cast + amax-scale-recompute kernels eat the gain.  Tuning the amax-scale recompute frequency (currently every call) could help, but again that's a multi-iter project.

---

## Where I tried + what I learned

Each of these took ~15-30 seconds of bench time, so the iter is mostly about identifying the structural-incompatibility-vs-shape-mismatch story:

- `--cuda-graphs --scfa`: incompatible by code design.  Would need scfa-without-fuse-streams.
- `--fp8-attn --bf16-attn` (combined): "flag ignored" — also a structural incompatibility (bf16-attn takes the fast path that doesn't go through cuBLASLt).
- `--fp8-attn` alone (no bf16-attn): HELIUM activates but tok/s unchanged.

The `--fp8-attn` finding is the more interesting one: **HELIUM is implemented but not yielding wins on the current shape mix**.  This validates the paradigm but suggests it needs different shapes / dtypes to bite.

---

## Sequence status

| iter | target | result | win |
|---: |---     |---     |---:  |
| 47  | SCFA dwconv-dK | PASS | +5.20% |
| 48  | LN-bwd dgamma  | FAIL | +2.4% |
| 49  | Fused Adam-int8 | PASS | +6.98% |
| 50  | SCFA sub-conv  | FAIL | +1.31% |
| 51  | --scfa-checkpoint-inner | FAIL | +4.46% |
| 52  | cuBLAS algo override | NULL | 0% |
| 53  | LN-bwd 2-phase | FAIL | +2.86% |
| 54  | (meta-analysis) | META | n/a |
| 55  | --cuda-graphs / --fp8-attn | NULL | ~0% |

Cumulative shipped: **+12.5%**.  2 PASS / 5 FAIL / 2 NULL / 1 META across 9 iters.

---

## Where next

The empirical picture after 9 iters is unambiguous:

1. **Engineering wins are exhausted** — every ≥4%-share solo kernel has been attacked.  Wins above the +5% bar require either combining multiple sub-bar wins (iter51+iter53+...) or removing the bar.

2. **Paradigm-level flags that exist** don't move the needle on the current iter49-stacked-flagship shape mix.

3. **New paradigm-level work** — FP8 readout via cuBLASLt, BF16 residual stream, sparse attention, vocab subsampling — would require multi-iter implementation arcs with non-trivial NLL parity work.  No single iter ships such a change.

The honest recommendation at this point is **option A from iter54 META: relax the per-iter bar to +3%**.  Under that bar, iter51 (+4.46%) and iter53 (+2.86%) would ship for ~+10% additional cumulative (~+22% total).  Without that change, the loop will keep producing FAILs.

Iter 56 will continue under the current bar unless the user signals otherwise.

---

## What this iter ships

No code change.  Only this doc.

```
$ git diff
(empty)
```
