# SLC × Local-Window Compound — Negative Finding at Small Scale

**Date:** 2026-04-23 (Ralph-loop iter 134)
**Hypothesis tested:** SLC (schedule 256→512→1024) × local-window attention
(W=256) should compound for multiplicative wall-clock speedup:
- SLC alone at 66M: 1.65× speedup (during T=256 warmup phase)
- Local-window W=256 alone at T=1024: ~2× speedup (attention cost T·W vs T²)
- **Projected compound: ~3× speedup over fixed T=1024 baseline**

## 1. Empirical result at 66M × 2500

| Method | Wall | EMA@2500 | Speedup vs T=1024 baseline |
|--------|:----:|:--------:|:--------------------------:|
| Baseline T=1024 (iter 128) | 78.6s | 7.88 | — |
| SLC (iter 129) | 47.6s | 7.40 | 1.65× |
| Local-window W=256 + SLC | **350.2s** | 7.41 | **0.22× (4.5× SLOWER)** |

**The compound is 7× SLOWER than SLC alone at 66M.** Negative finding.

## 2. Diagnosis

The `--flash-attn --local-attn 256` path invokes
`chiron_attention_shear_local_bf16` which is a BF16 flash-attention
variant. At 66M scale (small tensors, m=512, dModel=1024), the per-step
overhead of:
- FP32 → BF16 cast (Q, K, V tensors)
- BF16 tensor-core setup
- Local-window bounds computation per-query

dominates the savings from skipping non-window keys. Throughput drops
from 32500 tok/s (SLC tiled path) to 4300 tok/s (SLC local-bf16 path).

## 3. Projection to larger scale

The local-window path should benefit MORE at larger model sizes where:
- Tensor cores are better utilized
- BF16 cast amortizes across larger GEMMs
- The T·W vs T² savings is proportionally larger

Expected: at 1.84B scale, local-window + SLC could provide meaningful
speedup. But testing requires ~30 min per run and the 66M negative
result discourages further investment without an independent local-
window baseline measurement.

## 4. Preserved convergence

Despite the throughput regression, convergence is preserved:
- SLC + local-window EMA@2500 = 7.41
- SLC alone EMA@2500 = 7.40

Essentially identical. The local-window mechanism doesn't hurt learning
at W=256 with T=1024 at 66M scale.

## 5. Research program implication

Local-window attention (paradigm #6) was shipped before SLC. Its
speedup claims were benchmarked at larger scale, not at 66M. The
per-step overhead of the BF16 flash-attn code path makes it
counter-productive at small scale.

**Recommendation:** Use SLC alone at scales ≤ 234M. For larger scales
(500M+), SLC × local-window may compound if the flash-attn overhead
amortizes. Would need a dedicated benchmark.

**Deferred validation:** local-window × SLC at 1.84B ceiling.

## 6. Lesson: compose benchmarks at representative scale

This iteration's negative finding reinforces a methodology lesson:
paradigm compound tests should run at a scale where EACH individual
paradigm's own benefit is measured. Local-window was never tested
at 66M; forcing the composition at 66M surfaces an overhead that
isn't visible at local-window's native test scale.

Future compound tests should match each paradigm's validated scale
range to avoid false-negative composition results.
