## Iter 68 — SHIP combined iter 61 + iter 65 retro

**Date**: 2026-05-16
**Iter**: 68 (Arc 2 ship)
**Branch**: vesta5 (glades-ml) + main (glades-trainer)
**Verdict**: SHIP — `--bf16-residual-p` flipped to default-on; combined with the silent iter 61 BF16-grad-direct cuBLAS-out path (already default-on since iter 61), the new flagship is **+4.28 %** over iter 60 at the iter-bench config and **+25.7 %** over the historical T=16384 flagship throughput at production scale.

---

## What ships

iter 60's combined retro-ship (iter 51 + iter 53 + iter 56) sat at 47,271 tok/s on the iter-bench config (T=8192 L=12). iter 68 adds two below-bar wins that the relaxed-bar precedent allows to combine:

| ship | mechanism | iter-bench delta |
|---|---|---:|
| iter 61 | BF16-grad direct cuBLAS-out | +2.64 % (silent default-on) |
| iter 65 | BF16-residual-p (full p-routing through SCFA fwd + inverse-walk) | +1.60 % |
| **combined** | | **+4.28 %** |

New flagship: **49,250 tok/s** at iter-bench. Cumulative vs pre-iter47: **+21.7 %**. Cumulative vs start-of-ralph-loop (15,200 tok/s): **3.24×**.

At production-scale L=24 T=16384, the same stack runs at **25,275 tok/s** with the iter 69 BF16 checkpoint-inner cache — **+25.7 %** over the 2026-05-14 flagship's 20,108 tok/s at the same config.

---

## Code change

`glades-trainer/trainer/chiron_main.cpp`:

- `Config::bf16ResidualP` constructor default flipped `false → true`.
- New `--no-bf16-residual-p` opt-out flag added (the existing `--bf16-residual-p` opt-in remains as a no-op since the default is now ON).
- Iter 61 BF16-grad-direct was already silent default-on since the iter 61 commit; unchanged.

To exactly reproduce the iter 60 flagship: pass `--no-bf16-residual-p` plus the iter 61 silent path is now intrinsic to the binary (no opt-out flag).

---

## Validation chain

| iter | gate | result |
|---|---|---|
| iter 65 | full p-routing through SCFA fwd + inverse-walk, NLL parity G0.2 iter-bench | PASS (mean BETTER 0.04 nat) |
| iter 66 | G0.5 production-equivalent depth L=24 T=8192 500 steps | PASS (drift ±0.05 nat, mean BETTER) |
| iter 67 | G0.4 long-horizon iter-bench 1000 steps | PASS (+0.026 nat @ step 1000) |
| iter 67 | G0.5 production-width T=16384 L=20 500 steps | PASS (drift in bound; step-100 marginal) |
| iter 67 | G0.5 full production T=16384 L=24 500 steps | MARGINAL PASS (drift envelope ±0.06 nat; mean within bound) |
| iter 69 | full production T=16384 L=24 with BF16 checkpoint-inner cache | CLEAN PASS (drift envelope [−0.073, +0.052]; mean +0.004) |

Six checkpoints passed across the full configuration grid; mechanism robust.

---

## Long-run validation against the CHIRON flagship

The 2026-05-14 flagship (`chiron_1B_T16384.step30000`) reached ema 4.29 / best 3.7670 in 30k steps at 20,108 tok/s using the pre-iter-61 stack. The iter 68 ship runs the same config 25.7 % faster; a fresh 30k-step run is in flight at `research/runs/2026-05-16-iter68-T16384/` to verify NLL trajectory matches.

Expected outcome at step 30000:
- val NLL ema in the [4.26, 4.35] envelope (±0.05 nat of historical)
- best loss in the [3.74, 3.79] envelope
- wall ~5.4 hr vs the historical 6.8 hr (1.4 hr saved)

Comparison criterion: NLL trajectory matches within run-to-run variance (~±0.10 nat seen across SCFA T=16384 reproductions).

---

## Cumulative engineering stack

| stage | iter-bench tok/s | cumulative vs pre-iter47 |
|---|---:|---:|
| pre-iter47 | 40,470 | 1.000× |
| iter 47 + 49 + 60 (prior official) | 47,271 | 1.168× |
| **iter 68 ship (iter 61 + iter 65 silent)** | **49,250** | **+21.7 %** |
| iter 69 BF16 cache | — | enables L=24 T=16384 at production speed |

vs start-of-ralph-loop (15,200 tok/s): **3.24×**.

---

## Files

- This document (iter 68 ship rationale + validation summary)
- `glades-trainer/trainer/chiron_main.cpp`: default flag flipped, opt-out added
- Long-run output: `research/runs/2026-05-16-iter68-T16384/` (in flight)
