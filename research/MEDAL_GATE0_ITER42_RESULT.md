# MEDAL Gate-0 Iter 42 — Math Test PASS

**Date**: 2026-05-16
**Iter**: 42 (post-HMTA pivot to paradigm #262 MEDAL)
**Branch**: vesta5
**Design**: `research/PARADIGM_SHIFT_262_MEDAL_DESIGN.md`
**Test**: `research/medal_corruption_test.cpp`
**Wall**: 0.12 s

---

## TL;DR

**C0-iter42 PASS — 13 of 13 sub-tests succeed.** The closed-form absorbing-mask corruption process `q(x_t | x_0)` and Bayes-exact reverse `q(x_s | x_t, x_0)` are correctly implemented. The math is sound; the path to iter 43 (small trainable prototype) is clear.

This is the cheapest viable Gate-0 for paradigm #262: it verifies the mathematical foundation before any training infrastructure is built. A FAIL here would have indicated an implementation bug in the corruption or reverse-step routines that would silently corrupt any downstream training. A PASS lets us proceed to the training prototype with high confidence in the math layer.

---

## Setup

`research/medal_corruption_test.cpp` is a standalone C++11 test (350 LOC) that implements the absorbing-mask CTMC and runs five mathematical sanity checks. Test parameters: `V_actual = 32`, MASK_TOKEN = 32 (one past last real), `T = 64`, linear schedule `α(t) = t`. Five seeds × multiple trials per test.

---

## Sub-test results

### T1 — Forward marginal correctness

The marginal probability that a position is masked at time `t` must equal `α(t)`.

| t | α(t) | observed Pr[mask] | error | verdict |
|---|---|---|---|:---:|
| 0.00 | 0.000 | 0.0000 | 0.0000 | PASS |
| 0.25 | 0.250 | 0.2489 | 0.0011 | PASS |
| 0.50 | 0.500 | 0.5011 | 0.0011 | PASS |
| 0.75 | 0.750 | 0.7503 | 0.0003 | PASS |
| 1.00 | 1.000 | 1.0000 | 0.0000 | PASS |

Error < 0.5% at all `t` (tolerance: 0.005). Forward marginal is exact in expectation.

### T2 — Forward independence (per-position mask events uncorrelated)

For position pairs `(i, j)` at `t = 0.5`:
- `Pr[mask_i ∧ mask_j]` should equal `α(t) = 0.5` when `i = j`, `α(t)² = 0.25` when `i ≠ j`.

| pair type | observed | expected | error |
|---|---|---|---|
| on-diagonal `i = j` | 0.4977 | 0.5000 | 0.0023 |
| off-diagonal `i ≠ j` | 0.2478 | 0.2500 | 0.0022 |

PASS. The factored kernel `q(x_t | x_0) = ∏_i q(x_t^{(i)} | x_0^{(i)})` is correctly implemented (off-diagonal = α² confirms independence).

### T3 — Oracle reverse recovery (EXACT, K-independent)

With an oracle denoiser `p_θ(x_0 | x_t) := δ_{x_0_true}`, K-step ancestral sampling must return `x_0` exactly (no stochasticity remains because all mass goes to the true value at every unmask).

| K | trials | total positions | mismatches | verdict |
|---|---|---|---|:---:|
| 1   | 100 | 6400 | **0** | PASS |
| 4   | 100 | 6400 | **0** | PASS |
| 16  | 100 | 6400 | **0** | PASS |
| 64  | 100 | 6400 | **0** | PASS |
| 256 | 100 | 6400 | **0** | PASS |

**Zero mismatches at every K**. The Bayes-exact reverse + oracle denoiser produces deterministic recovery, confirming the closed-form posterior derivation in §5.3 of the design doc:
```
q(x_s | x_t, x_0): with prob α(s)/α(t) stay masked, else unmask to x_0.
```

### T4 — Uniform denoiser produces uniform marginal

With a worst-case denoiser `p_θ(x_0 | x_t) := Uniform(V)`, K-step sampling should produce samples uniform on V at every unmasked position.

Chi-square test (V_actual = 32 categories, n = 320 000 samples):
```
χ² = 32.9   (threshold for failure: ≈ 80 at 31 dof, 99% confidence)
```

PASS. The reverse step doesn't introduce any spurious distribution bias.

### T5 — Continuous-time self-consistency

At K = 4096 (effectively continuous), corruption + reverse-with-oracle equals identity.

```
K=4096, 10 trials: mismatches = 0 / 640   PASS
```

### Aggregate

**13 / 13 sub-tests PASS.** Total wall time: 0.12 s.

---

## What this validates

The MEDAL design doc's §5.2 forward kernel and §5.3 reverse posterior are correctly encoded in the prototype. Specifically:

1. The mask probability schedule `α(t)` and the forward sampler `q(x_t | x_0)` are exactly as derived.
2. The closed-form reverse step
   ```
   Pr[x_s = ⊥ | x_t = ⊥]    = α(s) / α(t)
   Pr[x_s = x_0 | x_t = ⊥]  = (α(t) - α(s)) / α(t)
   ```
   is correctly implemented (verified by T3 zero-mismatch oracle test and T5 continuous-time identity).
3. The K-step ancestral sampling loop converges in expectation to the data distribution under the oracle (T3) and to the uniform distribution under the worst-case (T4).

This is the **necessary precondition** for any MEDAL training prototype. If the math layer were buggy, gradients computed against the wrong reverse would silently mis-train the denoiser.

---

## What this does NOT yet validate (deferred to iter 43+)

- **The denoiser is trainable.** We've used oracle and uniform denoisers; neither is learned. Iter 43 builds a small trainable transformer and shows that gradient descent reduces ELBO.
- **MEDAL competes with AR at iso-compute.** We need an actual training experiment on a synthetic next-token task to compare ELBO vs AR exact NLL.
- **Wall-clock magnitudes at production T.** Far down the road (iter 46+).

---

## Risks identified for iter 43

From design §14 (honest pre-mortem):

1. **Discrete diffusion training instability at small scale.** Literature reports varying success. Mitigation: antithetic time sampling, learning rate warm-up.
2. **ELBO upper bound looseness at finite training.** Even though asymptotic gap is zero for absorbing diffusion, at 10–60M tokens the gap could be 0.5+ nat. Mitigation: report IWAE-M tighter estimator.
3. **Bidirectional attention memory at production T.** Defer until iter 46+; iter 43 is at T = 128 where this is not an issue.

The iter-42 math validation removes one risk class (incorrect math); the remaining risks are empirical and addressed in subsequent iters.

---

## Reproducibility

```bash
g++ -std=c++11 -O2 -Wall -Wextra \
    research/medal_corruption_test.cpp \
    -o research/medal_corruption_test
./research/medal_corruption_test 42
```

Wall: 0.12 s. Deterministic per seed. Output above is from seed=42.

---

## Next-iter recommendation (iter 43)

Build a minimal trainable MEDAL prototype:
- ~10M params, T=128, V=64, L=2-4, m=32-64
- Synthetic Markov-chain task with known optimal NLL
- Forward + backward + Adam optimizer (no GPU; CPU is fine at this scale)
- Train for 50k steps; report training loss curve, val ELBO at K_eval ∈ {8, 32, 128}
- Compare to AR baseline trained on same data

The C1 conjecture in the design doc: trained MEDAL ELBO ≤ AR exact NLL + 0.10 nat at iso-compute. PASS → continue to mid-scale prototype (iter 44). FAIL → diffusion-LM training at small scale is fundamentally harder than AR; pivot to VARCO or revise.

Estimated iter-43 wall: 3-5 hours (transformer backward is the bulk; the diffusion machinery is largely the same as this iter's math test).
