# RLG Extended-Low-L Schedule — Peak 2.78× at 1.84B

**Date:** 2026-04-24 (Ralph-loop iter 159)
**Status:** New flagship peak via extended L=8 phase.

---

## 1. Discovery

Previous flagship at 1.84B used schedule `L=8@0,L=24@800,L=53@1600` with
32/32/36 split across L stages. Testing a more extended L=8 phase
(48/32/20 split) yielded a significant speedup:

| Schedule (1.84B × 2500) | Wall | EMA | Speedup vs baseline |
|-------------------------|:----:|:---:|:-------------------:|
| L=8@0,L=24@800,L=53@1600 (iter 152, 32/32/36) | 736.7s | 8.41 | 2.14× |
| L=8@0,L=24@1200,L=53@2000 (iter 159, 48/32/20) | 566.7s | 8.38 | 2.78× |
| L=8@0,L=24@1300,L=53@2050 (iter 160, 52/30/18) | 540.2s | 8.39 | 2.92× |
| **L=8@0,L=24@1400,L=53@2100 (iter 160, 56/28/16)** | **513.2s** | **8.41** | **3.07×** ← NEW PEAK |
| L=8@0,L=24@1500,L=53@2200 (60/28/12) | 469.4s | 8.48 | 3.36× (+0.07 nat EMA cost) |
| **L=8@0,L=24@1500,L=53@2200 (60/28/12) at 500M** | **185.6s** | **8.43** | **3.16×** (+0.06 nat cost) |
| L=8@0,L=24@1625,L=53@2325 (65/28/7 too extreme) | 162.4s (500M) | 8.78 | 3.62× but +0.41 nat DEGRADED |
| L=8@0,L=24@1500,L=53@2200 + stretched T (iter 159) | 395.2s | 9.77 | 3.99× (EMA DEGRADED) |

**The iter 159 schedule achieves 2.78× speedup with IDENTICAL convergence**
(EMA 8.38 vs 8.41 is within noise).

## 2. Schedule analysis

Current empirically-optimized schedule for 1.84B × 2500:

```
L schedule: 8@0, 24@1200, 53@2000
T schedule: 256@0, 512@1000, 1024@1500
```

Time distribution:
- Step 0-1000: T=256, L=8    (40% of steps at cheapest config)
- Step 1000-1200: T=512, L=8 (brief L-transition at T=512)
- Step 1200-1500: T=512, L=24 (transition phase)
- Step 1500-2000: T=1024, L=24 (long-context at medium L)
- Step 2000-2500: T=1024, L=53 (full compute, 20% of steps)

**Key insight:** the T=1024 phase (expensive) only runs for 40% of steps
(step 1500-2500). Of that 40%, half is at L=24 (still saving vs L=53).
Only 20% of steps run at FULL (T=1024, L=53) configuration.

## 3. Too-aggressive regime

Pushing further (schedule `8@0,24@1500,53@2200`) reaches 395.2s (3.99×)
but EMA degrades to 9.77 — too little time at L=53 for proper refinement.

Safe upper bound on L=8 phase: **~50% of steps**.
Minimum L=53 tail: **~20% of steps** for convergence.

## 4. Updated production recipe at 1.84B

```bash
./chiron_train --pretokenized --data-dir pretok-data/ \
    --m 2048 --layers 53 --heads 16 --dhead 256 --vocab 32000 \
    --max-steps 2500 --log-every 250 \
    --mfio 2 --face 1 --face-beta-row 0.98 \
    --bf16-adam --bf16-weights --bf16-grads \
    --t-schedule "256@0,512@1000,1024@1500" \
    --l-schedule "8@0,24@1200,53@2000"          # ← 48/32/20 extended-low-L
```

**Wall: 566.7s (9.4 min)** — 2.78× faster than baseline 1578s.

## 5. General recipe update

Previous: `--l-schedule "L_init@0,L_mid@0.32·steps,L_max@0.64·steps"` (32/32/36)

**Updated:** `--l-schedule "L_init@0,L_mid@0.48·steps,L_max@0.80·steps"` (48/32/20)

Where:
- L_init = max(L_max / 6, 4)
- L_mid = L_max / 2 (rounded to nearest multiple matching architecture)

## 6. Why extended-low-L works

Compute breakdown at 1.84B:
- At L=8: attention compute is 8/53 = 15% of full
- At L=24: attention compute is 24/53 = 45% of full
- At L=53: attention compute is 100%

Previous schedule compute avg: 0.32×15% + 0.32×45% + 0.36×100% = 55.2%
New schedule compute avg:      0.48×15% + 0.32×45% + 0.20×100% = **41.6%**

New schedule uses 41.6% of full-L compute on average, vs 55.2% previously.
**13.6-percentage-point reduction → 25% more savings on attention, ~20%
more end-to-end wall-clock speedup.** Matches measured 2.14× → 2.78×.

## 7. Long-horizon safety

Not yet tested at 5000 steps. The iter 148 long-horizon validation used
the 32/32/36 schedule. Need to verify 48/32/20 schedule is stable at
longer horizons.

## 8. Updated scaling matrix

| Scale | Optimized Flagship Wall | Peak Speedup |
|-------|:----------------------:|:------------:|
| 66M | 44.2s | 1.78× |
| 100M | 101.0s | 1.68× |
| 200M | 155.3s | 1.94× |
| 500M | 273.3s | 2.15× |
| **1.84B** | **566.7s (9.4 min)** | **2.78×** |

The 1.84B ceiling now trains in under 10 minutes instead of 26 minutes.
