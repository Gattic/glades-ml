# VESTA Sweep v2 — Scale-up + LR sweep + VESTA HP sweep

**Run date:** 2026-04-16

**Harness:** `unit-tests/glades-unit-tests vesta-sweep-v2` → `VESTASweepV2Benchmark()` in `unit-tests/Backend/Machine Learning/vesta-test.cpp`

**Full raw log:** [`sweep.log`](sweep.log)

Addresses follow-up recommendations 2 (scale-up), 3 (VESTA HP), and 4 (per-optimizer LR sweep) from the v1 sweep.

## Setup

| Knob | v1 sweep | **v2 sweep** |
|---|---|---|
| vocab | 29 | 29 |
| dModel | 64 | **128** |
| dFF | 128 | **256** |
| Layers | 3 | **4** |
| Heads | 4 | 4 |
| Epochs | 30 | 15 |
| Corpus | 256 | **384** |
| Total FLOPs/run | 1× | ~4× |

Three phases:
- **Phase 1:** LR sweep `lr ∈ {3e-4, 1e-3, 3e-3, 1e-2}` per optimizer, 3 seeds each.
- **Phase 2:** VESTA hyperparameter axes (one at a time) at the VESTA-best LR:
  - `rank ∈ {4, 8, 16}`
  - `tau ∈ {0.0, 0.05, 0.1, 0.2}`
  - `tSk ∈ {4, 16, 64}`
  - `lambdaPerp ∈ {0.0, 0.1, 0.2, 0.4}`
- **Phase 3:** Head-to-head at each optimizer's best config, 5 seeds.

## Phase 1 — Per-optimizer LR sweep

All three optimizers pick **LR = 1e-2** as optimum in this range.

| opt | best LR | testNLL ± stddev |
|-----|---------|------------------|
| AdamW | 1e-2 | 1.8615 ± 0.0173 |
| ATLAS | 1e-2 | 3.3897 ± 0.0125 |
| VESTA | 1e-2 | 2.1091 ± 0.0093 |

The fact that all three want 1e-2 is unsurprising: the task is so small that the warmup-less, schedule-less training is in the "noise-limited" regime where any stable LR works if it's big enough. Adam's default 1e-3 is hurting it here.

## Phase 2 — VESTA HP single-axis sweeps

| axis | value | testNLL | Δ vs best |
|------|-------|---------|-----------|
| rank | 4 | 2.1129 ± 0.0110 | +0.0038 |
| rank | **8** | **2.1091 ± 0.0093** | — |
| rank | 16 | 2.1290 ± 0.0199 | +0.0199 |
| tau | 0.00 | 2.1093 ± 0.0093 | +0.0003 |
| tau | 0.05 | 2.1091 ± 0.0093 | +0.0001 |
| tau | 0.10 | 2.1091 ± 0.0093 | +0.0001 |
| tau | **0.20** | **2.1090 ± 0.0092** | — |
| tSk | 4 | 2.1096 ± 0.0309 | +0.0005 |
| tSk | **16** | **2.1091 ± 0.0093** | — |
| tSk | 64 | 2.1091 ± 0.0093 | +0.0000 |
| lambdaPerp | 0.00 | 3.3890 ± 0.0123 | **+1.3403** |
| lambdaPerp | 0.10 | 2.2439 ± 0.0223 | +0.1952 |
| lambdaPerp | 0.20 | 2.1091 ± 0.0093 | +0.0604 |
| lambdaPerp | **0.40** | **2.0487 ± 0.0196** | — |

### The dominant finding

**`lambdaPerp` is the only VESTA hyperparameter that meaningfully moves the needle in this regime.** Disabling the signed-complement step (`lambdaPerp = 0`) collapses VESTA's test NLL to **3.39** — the same entropy as ATLAS, same as a random token baseline `log(29)=3.37`. Raising it to `0.40` buys an extra **0.06 nats** over the default `0.20`.

Interpretation: at `dModel=128` with only 15 epochs and <2k optimizer steps total, the sketched-subspace mirror step barely has time to align the U, V bases with where the loss actually curves. The bulk of useful gradient energy lives in the complement, and the signed step on that complement is where learning actually happens. `rank`, `tau`, and `tSk` are all second-order knobs at this scale — all four values of each axis land within ~0.02 nats of each other.

Three consequences:

1. **VESTA's spectral-entropy geometry isn't providing useful signal here.** The Bregman-mirror step on the tracked subspace barely differs from a scaled SGD step on the same low-rank patch.
2. **A higher `lambdaPerp` would likely push farther**: we capped at 0.4 but the trend suggests `0.6–1.0` could be worth exploring. (Not done in this run; each extra value costs 3 additional 3-seed runs ≈ 30 s.)
3. **The regime where `tau` matters is different from the regime this sweep tests.** `tau` activates the spectral-homeostasis regularizer, which only starts mattering once `ell` has drifted significantly from its init value — which takes many more steps than 1.5k.

## Phase 3 — Head-to-head at best configs (5 seeds)

| opt | config | trainNLL | **testNLL** | wall (s) |
|-----|--------|----------|-------------|----------|
| **AdamW** | LR=1e-2 | 1.7609 ± 0.0363 | **1.8737 ± 0.0279** | 0.80 ± 0.03 |
| ATLAS | LR=1e-2, rank=8 | 3.3896 ± 0.0141 | 3.3926 ± 0.0101 | 0.98 ± 0.06 |
| **VESTA** | LR=1e-2, rank=8, tau=0.2, tSk=16, lp=0.4 | 1.9590 ± 0.0218 | **2.0577 ± 0.0262** | 3.39 ± 0.06 |

**Head-to-head Δ in test NLL vs AdamW:**

| opt | Δ testNLL | perplexity ratio |
|-----|-----------|------------------|
| ATLAS | **+1.519** | ≈4.57× worse |
| VESTA | **+0.184** | ≈1.20× worse |

## Progress vs v1 sweep

| | v1 (at default LR) | **v2 (at best LR + best HP)** |
|---|---|---|
| AdamW testNLL | 2.770 | **1.874** |
| ATLAS testNLL | 3.380 | 3.393 (still stuck) |
| VESTA testNLL | 3.062 | **2.058** |
| VESTA−AdamW gap | +0.292 | **+0.184** |

The gap to AdamW shrank from **0.29 nats to 0.18 nats** (a 37% reduction) after:
- matching LR (biggest effect — 1e-2 vs 1e-3 default),
- raising `lambdaPerp` from 0.2 → 0.4.

## Honest read

- **VESTA trains well and scales with tuning.** Every HP-tuning step we do narrows the gap to AdamW.
- **The signed complement step is doing most of the VESTA work** at this scale. The Bregman-mirror geometry component (tracked subspace) is not yet differentiating VESTA from a low-rank-sketched SGD.
- **AdamW remains the winner by 0.18 nats** on a 4-layer, 128-dim LM. This gap is real but not huge; a longer training run with an LR schedule and a hyperparameter search at larger scales could plausibly close it further.
- **ATLAS in bare-BSRP mode (no PRISM / SPARROW / aster / aegis) does not learn** on this task. Either the config is wrong for this scale, or its feature set is specifically designed for regimes much bigger than dModel=128. Not a comment on ATLAS's broader framework.
- **VESTA is ~4× slower than AdamW wall-clock** on CPU. This is the expected cost of the sketched-SVD refreshes + Stiefel retractions per matrix. The GPU path (`vesta_gpu_step`) is implemented and parity-tested but is not yet wired into the transformer training loop — when it is, wall-clock should approach AdamW.

## What we haven't tested (and what would matter)

1. **Larger scale (dModel ≥ 256, 8+ layers, 2k+ tokens).** VESTA's memory advantage (`O(r(m+n))` vs AdamW's `2mn`) is negligible at dModel=128 where AdamW fits in L2 cache. At dModel=1024+, it starts to matter.
2. **Heavy-tailed gradients.** VESTA's deterministic denominator should beat AdamW's noisy `√v_t` estimate when gradients are α-stable with α<2, which is where LLMs live at late training.
3. **Cross-axis HP interactions.** Single-axis sweep assumes independence; a joint `rank × tau × lambdaPerp` 3D grid might reveal non-separable optima.
4. **`lambdaPerp > 0.4`** — the monotone trend we saw may flatten or reverse; worth extending the sweep.
5. **Longer training with LR schedules.** All optimizers would benefit; the comparison at convergence may differ from the comparison at 15 epochs.

## Reproduce

```bash
cd unit-tests && sh .configure.sh cuda
bash test.sh vesta-sweep-v2 2>&1 | grep -v "^\[i\]2026" | tee sweep.log
```
