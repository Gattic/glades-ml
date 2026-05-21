# HMTA Gate-0 Iter 39 — Noise-Floor and Sink+Local Stress Tests

**Date**: 2026-05-16
**Iter**: 39
**Branch**: vesta5
**Builds on**: iter 38 (HMTA basis study; p=24 supported, basis-flexibility falsified)
**Prototype**: `research/hmta_forward_v3.cpp`

---

## TL;DR

**Iter 39 falsifies the noise-floor hypothesis of iter 38**. Forward L2 error at p=8 is **constant at 0.218** across synthetic Gaussian noise levels in `[0.001, 0.10]`, only worsening above noise σ=0.30. The 22% L2 floor is not driven by noise — it is driven by **rank-p projection of V** (the value vectors), which discards (d−p)/d of V's variance for any far-pair attention contribution.

**On sink+local synthetic data** (mimicking real LLM attention with attention-sink tokens), HMTA is **much harder**: forward L2 = 0.50 at p=8, dropping only to 0.30 at p=24. The sink-induced spectral concentration means the per-leaf rank-p basis under-represents the "background" tokens that occasionally need to be attended.

**Joint verdict**: C0a at p=8 fails on both Gaussian (0.22) AND sink+local (0.50). HMTA as-designed needs either **larger p** (iter 38's p=24 fallback, valid for Gaussian regime) **or separate p_K and p_V** (small p_K for similarity scoring, larger p_V to preserve V information — new iter-40 candidate) **or pivot**.

---

## Setup

Two new synthetic-data generators in `research/hmta_forward_v3.cpp`:

1. **Gaussian model** (same as iter 38): `q_i = U_q c_i + ξ_i`, `k_j = U_k d_j + ξ_j`. We sweep `noise_sd ∈ {0.001, 0.01, 0.03, 0.10, 0.30, 1.0}` at fixed rank `r = 8`.

2. **Sink+local model** (new): a few "sink" positions per sequence have K vectors with `10×` the magnitude of background tokens. Q + K share a rank-2 position-embedding component that creates local-decay. This produces attention matrices with a few dominant rows/columns (sinks) plus a fading local tail — the empirical pattern documented in *Attention-Sink* / *StreamingLLM* / *H2O* literature.

All other config matches iter 38: T=512, d=64, s_0=64, η=2, basis = K-only and V-only separate SVDs per leaf, 5-seed median.

---

## Result A — Noise-level sweep at rank r=8 (Gaussian model)

| noise_sd | forward L2 at p=8 | median rank-p retention |
|---:|---:|---:|
| 0.001 | **0.2178** | 1.000 |
| 0.010 | **0.2178** | 1.000 |
| 0.030 | **0.2178** | 1.000 |
| 0.100 | **0.2178** | 0.983 |
| 0.300 | 0.2195 | 0.810 |
| 1.000 | 0.3140 | 0.487 |

The forward L2 error is **invariant under noise** from σ=0.001 to σ=0.10 — a 100× range — at exactly 0.218. The rank-p retention is essentially **1.0** (perfect capture of the logit submatrix) over the same range. Yet the output L2 is 22%.

**This rules out the iter-38 noise-floor hypothesis entirely**. The error is not in the K-side projection (Λ retention is 1.0); it must be in the V-side.

### Mechanistic explanation

The HMTA forward replaces, for far pairs, the exact V[j] with V_hat[j] = U_v U_v^T V[j]. For V drawn from an isotropic Gaussian, the projection error is

$$
\|V[j] - V_\hat[j]\|^2 / \|V[j]\|^2 \;=\; (d - p)/d \;=\; (64 - 8)/64 \;=\; 0.875.
$$

That is, **87.5% of each V vector's energy is discarded** per far-pair contribution. The output Y[i] is a softmax-weighted sum over j's; for far-pair j's, each V[j] is replaced by V_hat[j] which has only 12.5% of the right direction. If far pairs carry ~37% of the attention budget (typical at η=2 with 8 clusters), the output error fraction is approximately `0.37 × sqrt(0.875) ≈ 0.35` — close to the observed 0.22 (after softmax normalization tempers things).

**No matter how good the K basis is — even perfect — the V projection loss is unavoidable** with a single rank-p subspace per leaf. The only way to reduce it is **larger p** or **a separate, larger p_V**.

---

## Result B — Sink+local synthetic (4 sinks per T=512, sink scale 10×)

| p | forward L2 | rank-p retention |
|---:|---:|---:|
| 8  | **0.4996** | 0.480 |
| 16 | 0.4350 | 0.744 |
| 24 | 0.3023 | 0.892 |

The sink+local model is **dramatically harder** than the Gaussian model:
- At p=8: forward L2 doubles (0.50 vs 0.22 for Gaussian).
- At p=24 (iter 38's recommended default): forward L2 is 0.30 — still 2× above the 0.15 threshold.
- Retention degrades correspondingly (0.48 at p=8).

### Mechanistic explanation

Sinks have K vectors with 10× the norm of background tokens. The per-leaf SVD captures the sink directions in the top singular values, leaving little room for background-token directions in a rank-p basis. When a "background" query token attends to its corresponding background key, the rank-p projection has poorly-represented that key (because most of the leaf's spectral energy was the sinks), and the resulting attention output is biased.

Real LLM attention has **exactly this pattern** (documented in multiple recent papers: attention sinks in initial positions, position-decaying local attention). So **iter 38's "p=24 clears C0a" conclusion is optimistic** — it holds for the Gaussian model but not for the sink+local model that better approximates real attention.

---

## Result C — Iter-39 Gate-0 verdict

| Test | forward L2 at p=8 | target ≤ 0.15 | verdict |
|---|---:|:---:|:---:|
| Gaussian noise_sd=0.03 | 0.2178 | | **FAIL** |
| Sink+local | 0.4996 | | **FAIL** |

**Joint verdict: C0a-iter-39 FALSIFIED.**

The hypothesis that "lower noise OR realistic attention spectrum clears C0a at p=8" is empirically rejected on both fronts. The Gaussian regime is noise-independent at p=8; the sink+local regime is *worse*, not better.

---

## What this tells us about HMTA's design

The iter sequence has now established three layered facts:

1. **Iter 37** (kernel-correctness): HMTA arithmetic is correct (SDPA recovery 1.6e-7); FLOP scaling is real (213× at p=8); forward L2 at p=8 is 0.22 (marginal C0a fail).
2. **Iter 38** (basis study): basis choice is irrelevant (joint vs separate K/V differ <1%); p=24 clears C0a on Gaussian (forward L2 = 0.13); FLOP ratio drops to 29.6× at T=16384, still above 10× floor.
3. **Iter 39** (noise + sink+local): the 0.22 floor is **rank-p V-projection loss**, not noise. Sink+local synthetic is dramatically harder. Even iter 38's p=24 fallback gives 0.30 on sink+local — failing C0a by 2×.

The HMTA design's V-projection step is a fundamental bottleneck. It cannot be fixed by basis flexibility (iter 38) or by reducing noise (iter 39); it can only be addressed by:

- **(F1) Larger p_V** (separate from p_K). Keep p_K small (cheap, sufficient for similarity scoring); raise p_V to 32 or 64 (preserves V variance).
- **(F2) Hybrid attention**: handle sinks separately. BigBird-style "global tokens" — sinks attend to / from all positions, exempt from the multipole compression.
- **(F3) Per-query basis** (i.e., abandon per-leaf basis). Each query gets its own optimal V projection. Defeats the multipole compression goal but might unlock the trainability question.
- **(F4) Pivot**: discard HMTA, retry with Candidate B (MEDAL discrete-diffusion LM) or Candidate C (VARCO variational compute), or design a new paradigm #262.

### F1 cost analysis (separate p_K, p_V)

At production T=16384, d=64, s_0=64, η=2:

| (p_K, p_V) | per-leaf K cost | per-leaf V cost | total per-layer FLOPs | FLOP ratio vs SDPA |
|---|---:|---:|---:|---:|
| (8, 8)   | 65K | 65K | 162M | 213× |
| (8, 32)  | 65K | 262K | 380M | 91× |
| (8, 64)  | 65K | 524K | 658M | 52× |

At (p_K=8, p_V=32) the FLOP ratio is still **91×** — well above the 10× target. The expected forward L2: p_K=8 captures the similarity matrix accurately (retention ~1.0 in Gaussian, ~0.48 in sink+local), and p_V=32 leaves only 50% of V variance discarded (much better than 87.5% at p_V=8). On Gaussian, this should drop forward L2 from 0.22 to roughly `0.37 × sqrt(0.50) ≈ 0.26 × 0.71 ≈ 0.19` — still slightly above 0.15. At (p_K=8, p_V=48), the V loss drops to 25%, predicting forward L2 around `0.37 × sqrt(0.25) = 0.185` — also marginal.

Even with separate p_V, clearing C0a at p=8 on sink+local appears unlikely without additional mechanisms (F2 or F3).

---

## Honest interpretation

The brief asks for **magnitudes** (10×+ compute reduction). Across iters 37–39 the empirical landscape is:

- HMTA's FLOP advantage is real (29.6× at p=24, validated at T=16384).
- HMTA's **forward accuracy** on synthetic-Gaussian is acceptable at p=24 (0.13 forward L2 < 0.15 threshold).
- HMTA's forward accuracy on synthetic-sink+local (closer to real LLM attention) is **much worse** at any p tested in this iter (0.30 at p=24).

This is a **conditional caveat**, not an outright falsification. To know whether HMTA works on REAL attention, we need either:
- **iter 40: real-attention SV-retention probe** on the post-fix flagship (the conclusive test).
- OR an iter that tests **F1 (separate p_V)** on sink+local synthetic, then proceeds to flagship probe.

Both options are tractable in one iter each. The flagship probe is the more binding.

---

## Reproducibility

```bash
g++ -std=c++98 -O2 -Wall -Wextra \
    research/hmta_forward_v3.cpp \
    -o research/hmta_forward_v3
./research/hmta_forward_v3 42
```

Wall: ~2 s. Deterministic per seed.

---

## Next-iter recommendation (iter 40)

**Option A (preferred)**: Real-attention SV-retention probe on the post-fix flagship.
- Add a `--dump-attn-svd` mode to `glades_chiron_train` or build a standalone analyzer.
- Run forward on 1 val batch (`pretok-data/val`), dump attention logit matrices at layers {6, 12, 18, 22} for ~100 random cluster pairs.
- Compute median rank-p retention for p in {8, 16, 24, 32}.
- **Pass-condition for HMTA viability**: median retention ≥ 0.85 at p ≤ 24.
- **Falsification condition**: median retention < 0.50 at p = 24 on real flagship attention. If this occurs, HMTA is empirically falsified at scale.

**Option B**: Separate p_K, p_V prototype.
- Quick C++ change: encode K at rank p_K, V at rank p_V (independently per leaf).
- Sweep (p_K, p_V) ∈ {(8, 32), (8, 48), (8, 64)} on sink+local synthetic.
- If (p_K, p_V) = (8, 32) clears C0a (forward L2 ≤ 0.15) on sink+local AND retains FLOP ratio ≥ 10× at T=16384, this is a viable design refinement.
- Otherwise, falls to F2 (hybrid attention with global sinks) or F4 (pivot).

If both options fit one iter (they likely don't — flagship integration is multi-iter), prioritize A; if not, do B first.
