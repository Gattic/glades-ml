# VESTA Sweep — Lion-style complement momentum

**Run date:** 2026-04-17

**Harness:** `unit-tests/glades-unit-tests vesta-sweep-mom` → `VESTASweepMomentumCompare()`

**Raw log:** [`sweep.log`](sweep.log)

Addresses follow-up recommendations #1 (extended `lambdaPerp` sweep) and #2 (Lion-style momentum on the complement step) from the v2 sweep analysis.

## TL;DR

Lion-style EMA momentum on the signed complement step cuts the remaining VESTA–AdamW testNLL gap from **+0.184 → +0.106 nats (−42%)** at identical wall-clock.

| config | best `lambdaPerp` | testNLL ± stddev | vs AdamW | wall (s) |
|--------|-------------------|------------------|----------|----------|
| VESTA plain | 0.40 | 2.0580 ± 0.0264 | +0.1843 | 3.39 |
| **VESTA +mom** | **0.20** | **1.9794 ± 0.0286** | **+0.1057** | 3.42 |
| AdamW (reference) | — | 1.8737 ± 0.0279 | 0 | 0.79 |

## Part 1 — Extended `lambdaPerp` (no momentum)

Before adding momentum, we finished the monotone-trend check from v2 by extending `lambdaPerp` upward. The single-axis sweep at matched config:

| `lambdaPerp` | testNLL | Δ vs 0.4 | notes |
|--------------|---------|----------|-------|
| 0.20 | 2.1274 ± 0.0259 | +0.07 | v2 default |
| **0.40** | **2.0580 ± 0.0264** | — | prior best |
| 0.60 | 2.0618 ± 0.0254 | +0.00 | plateau |
| 0.80 | 2.0981 ± 0.0338 | +0.04 | degrading |
| 1.00 | 2.1291 ± 0.0639 | +0.07 | |
| 1.50 | 2.2735 ± 0.0819 | +0.22 | clearly too aggressive |
| 2.00 | 2.4386 ± 0.0798 | +0.38 | |
| 3.00 | 2.8294 ± 0.1013 | +0.77 | |

See `../vesta_sweep_lp_20260417-032400/sweep.log` for the raw 8-point table.

**Conclusion:** the monotone trend stopped at `lp≈0.4-0.6`. The first-order recommendation (extend `lambdaPerp` upward) was wrong — it was a plateau, not a rising curve. At `lp>0.8` the signed step overshoots and the complement kicks `W` around in the complementary subspace faster than the mirror step can settle the tracked directions.

## Part 2 — Lion-style complement momentum

### The mechanism

Replace the complement step from

```
W -= lr * c_perp * sign(g_perp)
```

with an EMA-smoothed version:

```
m ← β * m + (1 − β) * g_perp            (β = 0.9)
W -= lr * c_perp * sign(m)
```

where `m` is a new per-weight-matrix buffer of size `m × n`. This is the classical heavy-ball / Lion-style update: the momentum EMA averages out gradient noise before the sign operation, so the complement step stops jittering direction from step to step.

Implementation: `VestaConfig::complementMomentumEnabled` (default `false`), `VestaConfig::complementBeta` (default `0.9`). State added: `WeightState::complementMomentum[m*n]`, allocated lazily on first use when the flag is set.

### Cross-sweep (`lambdaPerp × momentum` at 5 seeds)

```
mom       lp=0.20    lp=0.40    lp=0.60    lp=0.80    lp=1.00
plain     2.1274     2.0580*    2.0618     2.0981     2.1291
+mom      1.9794*    1.9869     2.0297     2.1578     2.2518

(* = best on row)
```

Key observations:

1. **+mom wins at every `lambdaPerp` in the lower half** (lp ≤ 0.4). Best improvement is at lp=0.2: 2.1274 → 1.9794, a **−0.148 nats** reduction.
2. **+mom prefers a smaller `lambdaPerp`** (0.2 vs 0.4 plain). Intuition: the EMA already smooths noise, so we don't need a large complement-step magnitude to punch through stochasticity.
3. **+mom degrades faster at high `lambdaPerp`** (lp≥0.8 flips from win to loss). Large signed updates stack with the momentum's persistent direction and overshoot.
4. **Wall-clock impact is negligible.** 3.42s (+mom) vs 3.39s (plain). The extra element-wise EMA is free compared to the sketched SVD cost.

### Memory cost

The `complementMomentum` buffer adds `|θ|_matrix` fp32 floats per matrix when enabled — same as AdamW's `m` (first moment). Combined with VESTA's rank-`r` state, total VESTA+mom state is:

```
  (m+n)*r + m*n  per weight matrix
```

vs AdamW's `2 m n` per weight matrix. At `m=n=128, r=8`: VESTA+mom = 16384 + 2048 = 18432 floats vs AdamW = 32768. Still about **44% less state than AdamW**, while closing 42% of the remaining NLL gap.

## Part 3 — Head-to-head (best config per optimizer, 5 seeds)

| opt | config | trainNLL | testNLL | wall (s) |
|-----|--------|----------|---------|----------|
| **AdamW** | LR=1e-2 | 1.761 ± 0.036 | **1.874 ± 0.028** | 0.79 |
| VESTA plain | LR=1e-2, rank=8, lp=0.4 | 1.959 ± 0.022 | 2.058 ± 0.026 | 3.39 |
| **VESTA +mom** | LR=1e-2, rank=8, lp=0.2, β=0.9 | **1.875 ± 0.028** | **1.979 ± 0.029** | 3.42 |

### Progress across sweeps

| sweep | AdamW | VESTA best | gap |
|-------|-------|------------|-----|
| v1 | 2.770 | 3.062 | +0.292 |
| v2 (best LR + HP) | 1.874 | 2.058 | +0.184 |
| **v3 (+ Lion momentum)** | 1.874 | **1.979** | **+0.106** |

**63% total gap closure over three sweeps** — from +0.29 nats on default config to +0.11 nats at the new best. The three interventions that mattered, in order of marginal impact:

1. Matching LR to 1e-2 (v1 → v2): −0.05 nats
2. Tuning `lambdaPerp` up to 0.4 (v1 → v2): −0.05 nats
3. **Adding complement momentum** (v2 → v3): **−0.08 nats**

`rank`, `tau`, `tSk` together contributed essentially zero across all three sweeps. The dominant mechanism in VESTA at this scale remains the signed complement step; the Bregman-mirror spectral geometry is still not the load-bearing component.

## Honest read

- **VESTA +mom is now within 0.11 nats of AdamW** on a real 4-layer transformer token-LM at tuned LRs. That's the best we've gotten so far.
- **The improvement is from a fundamentally un-VESTA source.** Momentum on a sign update is the Lion algorithm; it's not specific to the spectral-entropy design. This tells us the current VESTA instantiation at dModel=128 is functionally **Lion + a low-rank sketched SGD component that contributes marginally**.
- **The gap that remains likely won't close** from further tuning at this scale. To exercise VESTA's core mechanisms (spectral steering, memory-wall advantage, heavy-tailed robustness) we need dModel ≥ 256 and 10× more training.
- **The next high-EV move is not another sweep.** It's either (a) the gradient-driven basis (sketch `U, V` from `g` instead of `W`, so the mirror step aligns with the learning direction) or (b) scaling up, where VESTA's memory advantage starts to matter.

## What the data says about VESTA's design

The sweep strongly suggests the Bregman-mirror step on the tracked subspace is **not contributing meaningfully** to learning at this scale. Evidence:

- `lambdaPerp=0` → testNLL 3.39 (random baseline, matches ATLAS rank-0)
- `lambdaPerp=0.4` → testNLL 2.06 (all 1.33 nats of learning from the signed complement)
- `rank` flat across 4,8,16; `tau` flat across 0.0–0.2; `tSk` flat across 4–64

When every useful learning signal lives in `g_perp` (the complement), the tracked subspace `U, V` is essentially a spectator. Making it a participant requires changing what `U, V` track: either the gradient subspace (Fisher-style) or an explicit second-order approximation, rather than the current weight SVD.

## Reproduce

```bash
cd unit-tests && sh .configure.sh cuda
bash test.sh vesta-sweep-mom 2>&1 | grep -v "^\[i\]2026" | tee sweep.log
```
