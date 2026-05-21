# VESTA Sweep — vs AdamW and ATLAS on tiny token-LM

**Run date:** 2026-04-16

**Harness:** `unit-tests/glades-unit-tests vesta-sweep` → `VESTASweepBenchmark()` in `unit-tests/Backend/Machine Learning/vesta-test.cpp`

**Full raw log:** [`sweep.log`](sweep.log)

## Setup

| Knob | Value |
|---|---|
| Model | transformer decoder, token-LM, tied embeddings |
| Vocab | 29 |
| dModel | 64 |
| dFF | 128 |
| Layers | 3 |
| Heads | 4 |
| Positional encoding | none |
| Training corpus | 256 tokens (deterministic pattern + noise) |
| Test corpus | separate 256 tokens from same generator, different seed |
| Epochs | 30 |
| Seeds | {101, 202, 303, 404, 505} |
| Loss | full softmax cross-entropy, mean NLL per non-pad token |
| Optimizer state path | CPU reference |

**VESTA hyperparameters** (first-pass defaults, not tuned):
`rank=4, tau=0.1, rho=0.1, tSk=8, lambdaPerp=0.2, mu=4, gamma=0.01, kappa=0.1`.

**ATLAS hyperparameters:** `rank=4`, default `ATLASConfig`. No PRISM / SPARROW / RESOLVE features; bare BSRP.

**AdamW hyperparameters:** `β₁=0.9, β₂=0.999, ε=1e-8, biasCorrection=on`.

## Per-seed results

| opt | seed | trainNLL | trainPPL | testNLL | testPPL | wall (s) |
|-----|------|----------|----------|---------|---------|----------|
| AdamW | 101 | 2.5796 | 13.19 | 2.7806 | 16.13 | 0.39 |
| AdamW | 202 | 2.5999 | 13.46 | 2.8069 | 16.56 | 0.34 |
| AdamW | 303 | 2.5788 | 13.18 | 2.7930 | 16.33 | 0.33 |
| AdamW | 404 | 2.5707 | 13.07 | 2.7589 | 15.78 | 0.35 |
| AdamW | 505 | 2.5649 | 13.00 | 2.7092 | 15.02 | 0.35 |
| ATLAS | 101 | 3.3889 | 29.63 | 3.3904 | 29.68 | 0.40 |
| ATLAS | 202 | 3.3803 | 29.38 | 3.3879 | 29.60 | 0.40 |
| ATLAS | 303 | 3.3630 | 28.87 | 3.3637 | 28.90 | 0.33 |
| ATLAS | 404 | 3.3737 | 29.19 | 3.3777 | 29.30 | 0.32 |
| ATLAS | 505 | 3.3748 | 29.22 | 3.3816 | 29.42 | 0.34 |
| VESTA | 101 | 2.8912 | 18.01 | 3.1014 | 22.23 | 1.07 |
| VESTA | 202 | 2.9120 | 18.39 | 3.0590 | 21.31 | 1.03 |
| VESTA | 303 | 2.8972 | 18.12 | 3.0719 | 21.58 | 1.06 |
| VESTA | 404 | 2.8961 | 18.10 | 3.0468 | 21.05 | 1.13 |
| VESTA | 505 | 2.8819 | 17.85 | 3.0282 | 20.66 | 1.05 |

## Aggregate (mean ± stddev, n=5)

| opt | trainNLL | testNLL | testPPL | wall (s) | vs AdamW testNLL |
|-----|----------|---------|---------|----------|------------------|
| **AdamW** | 2.5788 ± 0.0133 | **2.7697 ± 0.0382** | **15.96 ± 0.60** | 0.35 ± 0.02 | — |
| ATLAS | 3.3761 ± 0.0095 | 3.3803 ± 0.0105 | 29.38 ± 0.31 | 0.36 ± 0.04 | +0.611 |
| VESTA | 2.8957 ± 0.0110 | 3.0615 ± 0.0276 | 21.37 ± 0.59 | 1.07 ± 0.04 | +0.292 |

Random baseline: `log(29) ≈ 3.367`.

## Honest read

- **VESTA trains and is clearly better than random** (testNLL 3.06 vs baseline 3.37 = 0.31 nats of learning). It also cleanly beats ATLAS on every seed (testNLL 3.06 vs 3.38 = 0.32 nats).
- **AdamW wins this run**. testNLL is 0.29 nats lower than VESTA (≈26% lower perplexity).
- **VESTA is ~3× slower** wall-clock because the CPU sketched-SVD refresh runs every `tSk=8` steps on every weight matrix. The GPU path (`vesta_gpu_step`) exists and passes parity tests but is not yet wired into `sgd_transformer.cpp`'s main training loop — only the CPU path runs here.
- **ATLAS at bare BSRP (no PRISM/SPARROW/etc.) with `rank=4` on a `dModel=64` 3-layer model is essentially stuck at random-token entropy.** This reflects ATLAS's documented need for either bigger models or one of its advanced sub-modes; it is not a comment on the full ATLAS framework.

## Why VESTA doesn't win here (and what would change that)

The VESTA design targets the **memory-wall** and **heavy-tailed-gradient** regime of 100B–10T parameter LLMs. On `dModel=64, 3 layers, ~60k parameters`:

1. **No memory advantage is realized.** AdamW's `2|θ| = 120k` fp32 state fits trivially. VESTA's rank-4 state is smaller but the savings are irrelevant.
2. **No heavy-tail gradient pathology.** With 256 tokens and full softmax, gradients are nearly Gaussian — AdamW's `√v_t + ε` normalization is effectively the exact optimum.
3. **The spectral control regularizer is weakly informative** on such tiny matrices (`64×64` attention) — singular spectra are dominated by init noise, not learning dynamics.
4. **The sketched SVD refresh** (`tSk=8`) introduces basis-jitter noise that hurts small-step-count training.
5. **Hyperparameters are unvalidated.** `rank`, `tau`, `rho`, `lambdaPerp` were picked from the design document defaults. A matched LR + hyperparameter sweep per optimizer would tighten the gap.

## Follow-up experiments

Listed in priority order:

1. **Wire VESTA into the GPU training path** (lines ~9755–9828 of `sgd_transformer.cpp`, mirroring `gpu::atlas_gpu_update`). Should bring wall-clock parity with AdamW.
2. **Scale up to `dModel=256, 8 layers`** with 10× more tokens. This is the smallest regime where VESTA's low-rank state starts costing less than AdamW's diagonal state.
3. **VESTA hyperparameter sweep**: `rank ∈ {8, 16, 32}`, `tau ∈ {0, 0.05, 0.1, 0.2}`, `tSk ∈ {4, 16, 64}`.
4. **Matched LR sweep per optimizer** so the AdamW baseline isn't advantaged by a favorable default.
5. **Heavy-tailed gradient stress test**: add per-token gradient noise with α-stable distribution (α∈[1.3, 1.8]) and observe whether VESTA's deterministic denominator holds up while AdamW's `√v_t` estimate diverges.

## Reproduce

```bash
cd unit-tests && sh .configure.sh cuda
bash test.sh vesta-sweep 2>&1 | grep -v "^\[i\]2026" | tee sweep.log
```

Deterministic: given fixed seeds `{101, 202, 303, 404, 505}` the numbers above should reproduce within float32 rounding (the transformer RNG honors `net.setSeed(seed)`).
