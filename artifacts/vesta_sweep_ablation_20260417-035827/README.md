# VESTA Sweep — Feature ablation (mom × ema × gradbasis)

**Run date:** 2026-04-17

**Harness:** `unit-tests/glades-unit-tests vesta-sweep-ablation` → `VESTASweepAblationCompare()`

**Raw log:** [`sweep.log`](sweep.log)

Addresses follow-up recommendations #1 (sign-stabilize tracked update via A-diagonal EMA) and #2 (gradient-driven basis) from the previous round. Implements both as optional VESTA features, then ablates them against the current best (momentum).

## Config

| Knob | Value |
|------|-------|
| dModel / dFF / layers / heads | 128 / 256 / 4 / 4 |
| Corpus / epochs | 384 / 15 |
| LR | 1e-2 |
| VESTA rank / lp / tSk | 8 / 0.20 / **4** |
| Seeds | 5 |

`tSk` was lowered from 16 to 4 because the training horizon (~15 optimizer steps per run) would otherwise never trigger a subspace refresh, making the `+gradbasis` arm a no-op. A side-effect is that every arm now pays ~4× more sketched-SVD cost (wall-clock 8.4s vs 3.4s).

## Ablation table (5 seeds)

| config | trainNLL | **testNLL** | Δ vs plain | Δ vs AdamW |
|--------|----------|-------------|------------|------------|
| plain | 2.0475 ± 0.0390 | 2.1308 ± 0.0366 | — | +0.2571 |
| **+mom** | 1.8722 ± 0.0269 | **1.9785 ± 0.0290** | −0.1523 | +0.1048 |
| +ema | 2.0539 ± 0.0254 | 2.1275 ± 0.0270 | −0.0033 (null) | +0.2538 |
| +gradbasis | 2.6712 ± 0.0887 | 2.7039 ± 0.1039 | **+0.5731** | +0.8302 |
| **+mom+ema** | 1.8733 ± 0.0274 | **1.9776 ± 0.0285** | **−0.1532** | +0.1039 |
| +mom+gradbasis | 2.6290 ± 0.2448 | 2.6805 ± 0.2096 | +0.5497 | +0.8068 |
| +ema+gradbasis | 2.7893 ± 0.1388 | 2.8015 ± 0.1010 | +0.6707 | +0.9278 |
| all3 | 2.6834 ± 0.2540 | 2.7292 ± 0.2292 | +0.5984 | +0.8555 |

**AdamW reference:** testNLL = 1.8737 ± 0.0279

## What each feature does

### Tracked EMA (`+ema`) — null result

Per-step EMA of the tracked-subspace diagonal `A[i,i] = (U^T g V)_ii`, with `β=0.9`. Before: each log-scale update used instantaneous `A_ii`. After: uses `EMA(A_ii)`.

**Effect on transformer training: essentially zero (−0.003 nats, within 1 stddev).**

Isolated unit test on a rank-4 noisy quadratic target did show a 3% loss reduction (3.319 → 3.218), confirming the EMA *does* reduce per-step variance when the tracked subspace is the active learning path. But on the transformer — where the active learning path is the signed *complement*, not the tracked subspace — averaging `A_ii` over time changes nothing measurable.

This is a clean falsification of the "noise in `ell` is what's holding VESTA back" hypothesis. The mirror step isn't noise-limited; it's mechanism-limited.

### Gradient-driven basis (`+gradbasis`) — **actively hurts** at this scale

Replaces `sketched_svd(W)` at refresh time with `sketched_svd(EMA(g))`. EMA β=0.7 (fast warmup chosen for the 15-step horizon).

**Effect on transformer training: +0.57 nats worse** (2.13 → 2.70). Combined with momentum: still +0.55 worse vs plain; completely erases momentum's gain.

Why the divergence from the unit test? Isolated test had a clean, *fixed* rank-4 target with no noise. Gradient-basis quickly locked `U, V` onto the target direction and learning accelerated dramatically (loss 4.22 → 0.37, a 91% reduction).

On the transformer, the loss landscape changes every step as the model learns. The gradient's principal directions shift continuously. `EMA(g)` with any `β < 1` lags this motion, and the sketched SVD of a lagged signal aligns `U, V` with directions that *were* useful but no longer are. Increasing `β` makes the EMA more stable but slower to warm; decreasing it makes it noisier. Neither regime beats simply sketching from `W`, whose principal directions are slowly varying by construction (weights change only as `lr × gradient`).

**Interpretation:** the premise of recommendation #2 was that W's SVD tracks "useless" noise directions. The data says otherwise: W's SVD is the most *stable* basis available, and stability beats "gradient-aligned but lagged" at this scale. The weight-driven basis was the right default all along — just not for the reason originally assumed.

### Momentum (`+mom`) — same 0.15-nat win as last sweep

No change in mechanism from the previous artifact. Reconfirmed at the adjusted `tSk=4` base config: plain → +mom shaves −0.15 nats.

## The best-of-VESTA remains `+mom+ema`

**testNLL 1.9776 ± 0.029, gap to AdamW +0.104 nats.** Essentially unchanged from the previous sweep's +mom (1.9794, +0.106). The tracked-EMA addition is a rounding-error win but free to keep on for stability.

## Four-sweep progression vs AdamW

| sweep | config | AdamW | VESTA best | gap |
|-------|--------|-------|------------|-----|
| v1 | default LR=1e-3 | 2.770 | 3.062 | +0.292 |
| v2 | best LR + HP | 1.874 | 2.058 | +0.184 |
| v3 | + Lion momentum | 1.874 | 1.979 | +0.106 |
| **v4 (this)** | + tracked EMA + gradbasis | **1.874** | **1.978** | **+0.104** |

The v4 work was a clean negative result: neither of the two highest-priority recommendations from the v3 report moved the needle further. The remaining 0.10-nat gap is not closable by tuning alone at this scale.

## What this tells us about VESTA's design

Three mechanisms were predicted to matter; four sweeps now say:

| mechanism | status at dModel=128 |
|-----------|----------------------|
| Bregman-mirror step on tracked subspace | **not load-bearing** — `rank`, `tau`, `tSk`, `ema` all flat |
| Spectral homeostasis (`tau`, `ellStar`) | **not load-bearing** — training horizon too short for ell drift |
| Signed complement step | **load-bearing** — carries 100% of learning at this scale |
| Lion-style complement momentum (our addition) | **load-bearing** — single biggest NLL gain available |

Current VESTA at dModel=128 is functionally **Lion plus a low-rank spectator**. The spectral-entropy geometry, the mirror step, and the tracked-subspace machinery are inert.

## What would actually differentiate VESTA now

The sweep data has ruled out every "more tuning" and "basis choice" hypothesis we had. Remaining paths that would genuinely distinguish VESTA from Lion:

1. **Scale up.** At dModel≥1024, VESTA's rank-`r` state is a real memory advantage. Lion's `m` buffer is fixed at `O(mn)`; VESTA's can be `O((m+n)r) ≪ O(mn)`. Memory-constrained regimes are the one place where the design has a free win.

2. **Heavy-tailed gradient test.** VESTA's deterministic `φ''(σ)σ²` denominator doesn't diverge under α-stable gradient noise, where AdamW's `√v_t` estimate does. This is a claim we haven't tested. If true, VESTA has a niche on late-LLM training regimes where gradient heavy-tailedness is documented.

3. **Second-order curvature per-direction.** The mirror step as designed does scale adaptation via `σ` and Φ. It could be extended to use actual curvature information (diagonal Hessian estimates or Gauss-Newton) per tracked direction. This is a design change, not a tuning knob.

4. **Abandon the mirror step** and position VESTA as "memory-efficient Lion with rank-`r` spectral side-info for diagnostics." Honest, deployable, ~50% less state than AdamW, known Lion-class NLL.

## Reproduce

```bash
cd unit-tests && sh .configure.sh cuda
bash test.sh vesta-sweep-ablation 2>&1 | grep -v "^\[i\]2026" | tee sweep.log
```
