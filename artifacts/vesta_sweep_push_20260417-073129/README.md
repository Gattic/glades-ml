# VESTA Push-NLL Sweep at dModel=512 — LONG-HORIZON REGRESSION

**Run date:** 2026-04-17

**Harness:** `unit-tests/glades-unit-tests vesta-sweep-scale-push` → `VESTASweepScalePush()`

**Raw log:** [`sweep.log`](sweep.log)

## TL;DR — Honest reversal

The v6 scale ladder showed VESTA+mom beating AdamW by **−0.146 nats** at dModel=512 / 15 epochs. This sweep extends the same config to **50 epochs and 5 seeds** with a rank sweep. The result:

**At 50 epochs, AdamW beats VESTA+mom by +0.38 nats (r=32) to +0.88 nats (r=8).** VESTA's 15-epoch win was a short-horizon transient, not a sustained advantage.

| variant | trainNLL | testNLL | Δ vs AdamW | wall (s) |
|---------|----------|---------|------------|----------|
| **AdamW** | 1.115 ± 0.12 | **1.933 ± 0.08** | — | 12.3 |
| VESTA-plain r=8 | 3.267 ± 0.31 | 3.043 ± 0.18 | +1.110 | 54.5 |
| VESTA+mom r=8 | 2.608 ± 0.20 | 2.813 ± 0.27 | +0.880 | 55.0 |
| VESTA+mom r=16 | 2.518 ± 0.33 | 2.655 ± 0.17 | +0.722 | 92.7 |
| VESTA+mom r=32 | 1.981 ± 0.34 | 2.313 ± 0.22 | +0.380 | 174.5 |

## What actually happened at 15 vs 50 epochs

| horizon | AdamW testNLL | VESTA+mom r=8 testNLL | Δ |
|---------|---------------|-----------------------|---|
| 15 ep | 1.905 ± 0.24 | 1.758 ± 0.05 | **−0.146** (VESTA wins) |
| 50 ep | 1.933 ± 0.08 | 2.813 ± 0.27 | **+0.880** (AdamW wins) |

Between 15 and 50 epochs:
- **AdamW improved testNLL** from 1.905 → 1.933 (~flat, slight overfit trend — trainNLL dropped 1.76 → 1.11).
- **VESTA+mom r=8 testNLL got WORSE** from 1.758 → 2.813.

VESTA's training loss also behaved oddly: at 50 epochs VESTA+mom has trainNLL 2.61 (barely below test), while AdamW has trainNLL 1.11 (clear overfitting). AdamW fit the training set; VESTA couldn't fit the training set. This is the opposite of what "VESTA converges faster" would predict — it suggests VESTA **oscillated or plateaued** rather than continued descending.

## Rank trajectory is correct but insufficient

Rank scaling at 50 epochs does help:

| rank | testNLL | Δ vs AdamW |
|------|---------|------------|
| 8 | 2.813 | +0.880 |
| 16 | 2.655 | +0.722 |
| 32 | **2.313** | **+0.380** |

Each rank doubling reduces the gap by ~0.15–0.30 nats. Extrapolating linearly, rank=64 might bring VESTA within +0.15 of AdamW, rank=128 might tie. But at rank=64 the complement-momentum buffer alone is `4 * d^2 = 1 MiB` per weight matrix × ~25 matrices = 25 MiB of state, plus (m+n)·r·fp32 ≈ 1 MiB. That's about 40% of AdamW's state — a much weaker memory story, and wall-clock would be >5 minutes per run.

## Why VESTA stalls at long horizons (mechanism)

The likely culprit is the **signed complement step**. VESTA's complement update is:

```
W -= lr * lambdaPerp * c_perp * sign(g_perp)
```

The sign operation **discards gradient magnitude**. Late in training, when gradients shrink near a local minimum, `sign(g_perp)` still produces steps of fixed magnitude `lr * lambdaPerp * c_perp`. That's the same oscillation mechanism that causes constant-LR Lion to stall at long horizons without a schedule.

AdamW doesn't have this problem: its `g / √v` normalization scales with gradient-variance, so late-training steps naturally shrink as gradients shrink.

Evidence supporting this:
- VESTA+mom r=32 at 50 epochs has trainNLL 1.98, testNLL 2.31 — essentially saturated (stddev 0.34 across seeds, high variance suggesting oscillation around multiple basins).
- Rank=32 with more signal captured doesn't fix the stall; it only slows it.
- AdamW's trainNLL drops from ~1.76 (15 ep) to 1.11 (50 ep) — genuine fine-tuning.

## The 15-epoch win was real but misleading

The v6 result holds at what it claimed: at 15 epochs dModel=512, VESTA+mom is 0.15 nats ahead of AdamW. That reflects **faster initial convergence** — VESTA's constant-magnitude signed steps rapidly reduce loss from the initialization plateau.

But it doesn't reflect asymptotic quality. At any longer training budget (tested here at 50 epochs, expected to widen further at 100+ epochs), AdamW's adaptive magnitude dominates.

## What's actually needed to sustain the win

For VESTA to beat AdamW at long horizons, the complement step must be able to reduce its effective magnitude late in training. Three mechanisms that would achieve this:

1. **LR schedule (cosine decay).** Drop `lr` from 1e-2 → 1e-4 over training. Both optimizers benefit; VESTA benefits more because its constant-magnitude complement step becomes scale-aware via the schedule.

2. **Scale the complement step by EMA(|g_perp|).** Like AdamW's adaptation but applied only to the complement subspace:
   ```
   v_perp = beta2 * v_perp + (1-beta2) * g_perp^2
   W -= lr * lambdaPerp * (m_perp / sqrt(v_perp + eps))
   ```
   This makes the complement effectively AdamW-on-complement with VESTA's mirror on the tracked subspace. Adds one more `m*n` buffer per matrix; still below AdamW's total.

3. **Sign step gated by gradient magnitude.** Only apply `sign(g_perp)` above a threshold, skip below. Simpler but harder to tune.

Of these, **(2) is the most promising.** It's essentially: "do AdamW where it's doing useful work (the complement), and add VESTA's spectral structure on top (the tracked subspace)." That's the optimizer the data actually wants.

## Corrected seven-sweep progression

| sweep | config | AdamW | VESTA best | gap |
|-------|--------|-------|------------|-----|
| v1 | default LR=1e-3, 30ep | 2.770 | 3.062 | +0.292 |
| v2 | best LR + HP, 15ep | 1.874 | 2.058 | +0.184 |
| v3 | + Lion momentum, 15ep | 1.874 | 1.979 | +0.106 |
| v4 | + ema + gradbasis, 15ep | 1.874 | 1.978 | +0.104 |
| v5 | dModel=256, 15ep | 1.581 | 1.578 | −0.003 |
| v6 | dModel=512, 15ep | 1.905 | 1.758 | **−0.146** |
| **v7 (this)** | **dModel=512, 50ep** | **1.933** | **2.313** (r=32) | **+0.380** |

The v6 "breakthrough" was an artifact of the short training horizon. When both optimizers have time to converge, VESTA lags.

## What this DOESN'T falsify

- **The memory argument still stands.** VESTA-plain uses 1.4% of AdamW's optimizer state at dModel=512. If the choice is "train dModel=512 AdamW" or "train dModel=1024 VESTA in the same memory", the latter might still give lower NLL because of model-scale. We haven't tested this.
- **The early-training speed advantage is real.** VESTA converges faster to reasonable quality. For iterative development or early stopping, that matters.
- **The spectral-entropy curvature design is NOT the problem.** The tracked-subspace machinery works; it's the signed complement step that causes the long-horizon stall.

## What to do next

1. **Implement AdamW-scaled complement step.** Replace `sign(g_perp)` with the AdamW update rule restricted to the complement. Expected: closes the gap at long horizons while preserving VESTA's memory advantage on the tracked subspace.

2. **Re-run v7 with cosine LR schedule.** Cheap experiment (needs a ScheduleCfg change in the harness). Tests whether LR decay alone fixes the stall.

3. **Same-memory-budget comparison.** Train AdamW at dModel=512 vs VESTA-plain at dModel=2048 (same optimizer memory). If VESTA-plain at 2048 beats AdamW at 512 on NLL, the memory argument is empirically validated.

4. **Skip further 15-epoch-horizon sweeps.** The short-horizon regime has been thoroughly characterized and doesn't reflect production usage.

## Reproduce

```bash
cd unit-tests && sh .configure.sh cuda
bash test.sh vesta-sweep-scale-push 2>&1 | grep -v "^\[i\]2026" | tee sweep.log
```

Expected runtime: ~8 min. Reproduction with different seeds should fall within the stddevs reported (pooled ~0.2 nats).
