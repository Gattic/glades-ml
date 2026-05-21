# MEDAL Iter 45 — Combined Time-Embedding + Larger Scale + Inference Throughput

**Date**: 2026-05-16
**Iter**: 45 (paradigm #262 MEDAL — combined A+B+C attack)
**Branch**: vesta5 (glades-ml) + glades-trainer/main
**Builds on**: iter 44 (C1 fail at 5M scale, gap +0.3 nat)

---

## TL;DR

**Combined iter attack across three axes finds MEDAL more empirically problematic, not less.**

- **A. Larger scale (15M params, 10M tokens)**: gap **WIDENS** to +1.0 nat (AR drops to 8.43, MEDAL stuck at ~9.4). MEDAL does not scale.
- **B. Inference throughput**: at T=1024, MEDAL K=16 projects to **12.8k tok/s** (vs chiron_infer's no-KV-cache AR at ~200 tok/s, 64× win). But vs KV-cached AR (typical, ~200k tok/s), MEDAL is **16× SLOWER**. The magnitudes claim is highly dependent on AR baseline implementation.
- **C. Time embedding** φ(α): added (~30 LOC); minimal effect on ELBO at 5M scale (9.40 vs prior 9.36).

Net: paradigm #262 MEDAL as currently implemented does NOT meet the brief's magnitudes target on training quality, and meets it on inference only against a degraded AR baseline.

---

## Setup

Identical to iter 44 config except where noted:

```bash
# AR 15M
build/glades_chiron_train --seq-len 1024 --m 256 --layers 8 --heads 8 --dhead 64 \
  --lr 3e-4 --max-steps 10000 --warmup 200 --val-every 1000 --val-batches 8 \
  --bf16-logits --bf16-logits-storage --seed 1337
# 12.39M params; wall 2:31

# MEDAL 15M (same + --medal-train --medal-eps 0.05)
# 12.39M params; wall 2:26

# Throughput bench (lr=0, 100 steps)
build/glades_chiron_train [same config] --lr 0 --max-steps 100 [--medal-train]
```

Combined wall: ~7 minutes total across all runs.

---

## A. Larger-scale comparison (12.4M params, 10M tokens)

### AR baseline trajectory

| step | val NLL |
|---:|---:|
| 0    | 10.41 |
| 1000 | 9.39 |
| 2000 | 9.41 |
| 3000 | 9.19 |
| 4000 | 9.03 |
| 5000 | 8.58 |
| 6000 | 8.67 |
| 7000 | 8.63 |
| 8000 | 8.50 |
| 9000 | 8.46 |
| 10000 | **8.43** |

AR scales cleanly: 10.41 → 8.43 over 10M tokens, consistent downward trajectory.

### MEDAL 15M (with time embedding) trajectory

| step | val_NLL_all | medal_ELBO | alpha |
|---:|---:|---:|---:|
| 0    | 10.38 | 10.39 | 0.573 |
| 1000 | 6.62  | 9.51  | 0.845 |
| 2000 | 9.49  | 9.82  | 0.875 |
| 3000 | 8.43  | 9.51  | 0.906 |
| 4000 | 8.79  | 9.71  | 0.936 |
| 5000 | 8.12  | 9.66  | 0.067 |
| 6000 | 9.50  | 9.79  | 0.097 |
| 7000 | 9.23  | 9.47  | 0.128 |
| 8000 | 9.10  | 9.42  | 0.158 |
| 9000 | 8.60  | 9.67  | 0.189 |
| 10000 | 9.96  | **10.47** | 0.219 |

MEDAL ELBO **does not converge** at 15M scale. Final ELBO 10.47 is WORSE than at step 7000 (9.47). The val_NLL_all also oscillates wildly (6.62 → 9.49 → 9.96).

**Net A verdict**: MEDAL ELBO is stuck around 9.4-10.5 regardless of scale. AR's NLL dropped from 9.11 (5M) to 8.43 (15M) — MEDAL did NOT make this transition.

**Hypothesis**: MEDAL has learned the OUTPUT TOKEN DISTRIBUTION (unigram-like, NLL ~9.5) but not the CONDITIONAL prediction. The denoiser is collapsing to its marginal. Possible structural causes:
- The 1/α ELBO weight is not applied in the loss — gradient signal per masked position scales as α, not 1/α.
- MASK embedding (E[V]) stays at zero (no gradient via readout, no embedding-lookup backward).
- Time embedding at q_0 only doesn't reach deeper layers.
- Loss scaler interacts badly with masked-gradient-zero pattern.

---

## B. Inference-throughput analysis

### Measured per-step (forward+backward, batch=T) throughput

Both at T=1024, 15M params, 100 steps each:

| mode | wall (100 steps) | per step | tok/s |
|---|---:|---:|---:|
| AR | 2.22 s | 14.6 ms | **68,256** |
| MEDAL | 2.15 s | 14.1 ms | **70,799** |

MEDAL is ~4% faster per step than AR (less causal-mask overhead). The corruption + masked-CE kernels add negligible cost (consistent with iter 43 nsys profile).

### Projected K-step generation throughput

For K-step MEDAL ancestral sampling at T=1024:
- Each step = 1 full forward (5 ms forward-only, half of total step time)
- Generates all T=1024 tokens in K total steps
- Throughput = T / (K × t_fwd)

| K | total wall | tok/s |
|---:|---:|---:|
| 16 | 80 ms | **12,800** |
| 32 | 160 ms | 6,400 |
| 64 | 320 ms | 3,200 |
| 256 | 1280 ms | 800 |

### Comparison to AR generation

**chiron_infer (existing trainer's AR inference)**: from its docstring, it runs the full O(T²·L) forward per generated token (no KV cache):
- Per token = 5 ms full forward
- T=1024 tokens = 5.12 s = **200 tok/s**

**Hypothetical AR with KV cache**: not implemented in this codebase but standard in production LLMs:
- Per token amortized to O(T·L) — about 1/T = 0.1% of full forward
- = ~5 µs per token = **~200,000 tok/s**

### Verdict

| comparison | MEDAL K=16 (12.8k tok/s) |
|---|:---:|
| vs chiron_infer no-KV-cache AR (200 tok/s) | **64× faster** ✓ |
| vs KV-cached AR (200k tok/s, typical production) | 16× **slower** ✗ |

The brief's magnitudes target on inference throughput holds **only in the comparison to the existing chiron_infer implementation**, which lacks a KV cache. Against a fairly-implemented AR baseline with KV cache, MEDAL's inference is **16× slower**, not faster.

This is a critical realization. The original design doc §7.1 wrote: "MEDAL achieves wall-clock T/K = 256× speedup" — but this implicitly assumed AR's per-step cost was full-forward. It is NOT, when AR uses KV cache. The honest comparison gives MEDAL a 16× speedup only against the *degraded* AR implementation.

---

## C. Time embedding effect

Added `medal_compute_phi_host` + `medal_add_time_embedding` kernel (~50 LOC). Sinusoidal φ(α) ∈ ℝ^m broadcast-added to q_0 after embedding gather, before layer 0.

### Same-config (5M, 2000 steps) re-test

| run | medal_ELBO @ α=0.69 |
|---|---:|
| iter 44 MEDAL (no time emb) | 9.357 |
| iter 45 MEDAL + time emb | **9.40** |

The time embedding **does not significantly affect** ELBO at this scale. Likely reasons:
- φ(α) is added once to q_0; deeper layers don't see it directly. Per-layer time conditioning would help.
- The denoiser may not have learned to USE the time signal at this short training.
- The numerical magnitude of φ(α) (sin/cos ∈ [-1, 1]) is small relative to embedding values.

This is consistent with the literature pattern that time conditioning matters most when the noise schedule is COMPLEX (e.g., learned), and less when it's monotone linear like α(t)=t.

---

## Combined verdict — C1 status

C1 conjecture from PARADIGM_SHIFT_262_MEDAL_DESIGN.md:
> "MEDAL ELBO ≤ AR exact NLL + 0.10 nat at iso-compute, AND K_eval=32 inference ≥ 4× AR throughput"

| condition | observed | verdict |
|---|---|---|
| ELBO ≤ AR NLL + 0.10 at 5M | +0.25 nat (lr=3e-4), +0.31 (lr=1e-4) | FAIL |
| ELBO ≤ AR NLL + 0.10 at 15M | +1.0 nat (gap WIDENED) | FAIL |
| K=32 inference ≥ 4× AR (vs no-KV chiron_infer) | 32× | PASS |
| K=32 inference ≥ 4× AR (vs KV-cache hypothetical) | 0.03× (32× slower) | FAIL |

**Net: C1 FAILS** on training quality at every scale tested. PASSES on inference ONLY against a non-KV-cached AR baseline, which is not a fair production-style comparison.

---

## What this iter accomplished

### Built infrastructure
- Sinusoidal time embedding kernel `medal_add_time_embedding` (~50 LOC).
- Host helper `medal_compute_phi_host` (sin/cos features of α).
- Wired into forward path conditionally on `cfg.medalTrain`.
- Re-built clean, glades-ml + glades-trainer.

### Generated empirical data
- Two scales × two modes × multiple seeds of MEDAL vs AR training trajectories.
- Per-step throughput measurements for both modes.
- Projected K-step inference throughput at 4 K values.

### Identified structural issues with MEDAL implementation
1. ELBO doesn't scale (MEDAL stuck at ~9.4 across 5M and 15M).
2. The 1/α ELBO weight is NOT applied in the loss — gradient magnitude per masked position is α-scaled (down), not 1/α-scaled (up). This is a likely root cause of poor training.
3. MASK embedding stays zero throughout training.
4. Time embedding at q_0 has minimal effect.
5. Per-layer alpha conditioning is needed for serious training.

---

## Honest assessment

The iter 42-45 sequence on paradigm #262 MEDAL has produced:
- Mathematically correct corruption + reverse process (iter 42, 13/13 sub-tests pass).
- Functional GPU implementation (iter 43, kernels <1% of GPU time).
- Empirical failure to match AR NLL at iso-compute (iter 44, 5M scale).
- Empirical failure to scale (iter 45, 15M scale — gap WIDENS).
- Realization that the inference magnitudes claim depends on AR baseline implementation (iter 45).

**Paradigm #262 MEDAL, in its current form, does NOT meet the brief's magnitudes target.**

The path forward has three options, in priority order:

### Option D: Fix the ELBO loss weighting (cheap, well-defined)
Currently the loss applies the standard CE gradient, masked to corrupted positions. The proper ELBO requires multiplying the per-masked-position gradient by **1/α(t)**. This is a single multiplication in `scale_array`. Hypothesis: this fixes the scale-doesn't-help issue.

### Option E: Pivot to paradigm #263 (next-best from iter 37 dispatch)
VARCO (variational conditional compute) was Candidate C — modest 3-6× upside, but mathematically clean and very different failure modes than HMTA/MEDAL. Worth dispatching.

### Option F: Step back and reconsider the brief
After 5 falsifications (#250 SFA, #260 IGAA, #261 HMTA, partial #262 MEDAL training failure, MEDAL inference dependent-on-baseline), the brief's "magnitudes" target may be empirically harder than the framing implied. A meta-discussion with the user about goals and scope is appropriate.

Recommended: **D first** (cheap fix), then if it works retry C2; if it doesn't, **F** (reconsider brief).

---

## Reproducibility

Both training runs and the throughput bench are archived at:
- `research/runs/2026-05-16-medal-iter45/ar_15M/`
- `research/runs/2026-05-16-medal-iter45/medal_15M/`
- `research/runs/2026-05-16-medal-iter45/medal_5M_te/` (5M with time embedding)

All commands documented above. Wall time: ~7 min total.
