# ORION vs Adam at production scale — 30k Adam complete + 300k ORION partial (90k logical)

**Date**: 2026-05-18
**Hardware**: NVIDIA RTX 4080 SUPER, 15.56 GB VRAM
**Model**: 870.94 M params (T=16384 m=2048 dModel=4096 L=24 nH=16 dH=256 V=32000)
**Stack**: `--bf16-weights --bf16-grads --bf16-residual-p --scfa --scfa-bf16-inner --scfa-bf16-outer --scfa-fuse-streams --scfa-reln-opt --scfa-checkpoint-inner-bf16 --bf16-logits --bf16-logits-storage --int8-adam --no-fuse-attn --fuse-attn-reln`
**Recipe**: lr=1e-4, warmup=500, grad-clip=0.5, seed=1337
**ORION additions**: `--orion --orion-r 1 --orion-K 10 --orion-m-subspace 0 --orion-fd-eps 1e-3 --orion-hvp-refresh 100 --orion-bf16-anchor --orion-int8-v --orion-reduced-adam --orion-e-only`

## Verdict

**Do not promote ORION into the default flagship.** ORION beats Adam by 0.15–0.22 nat in the first 1–2 k gradient evaluations, but the advantage decays to ≤ 0.05 nat by 6 k grad-evals and ties by 8 k. The 5k-calibration headline (0.52 nat at 500 grad-evals) does not compound — it shrinks.

## Iso-grad-evaluation comparison

Adam step N corresponds to ORION step 10N (same gradient evaluations, K=10).

| Grad evals | Adam val NLL | ORION val NLL | Δ (ORION − Adam) |
|---:|---:|---:|---:|
| 500 | 8.41 | 7.89* | −0.52* |
| 1 000 | 7.71 | 7.56 | −0.15 |
| 2 000 | 6.27 | 6.05 | −0.22 |
| 3 000 | 5.41 | 5.33 | −0.08 |
| 4 000 | 5.35 | 5.30 | −0.05 |
| 5 000 | 4.77 | 4.72 | −0.05 |
| 6 000 | 4.77 | 4.76 | −0.01 |
| 7 000 | 5.42 | 5.41 | tie |
| 8 000 | 4.60 | 4.58 | −0.02 |
| 9 000 | ~4.55 | 4.57 | tie |

\* 500-grad-eval result is from the 5k calibration runs (separate seeds; Adam's value comes from the 5k Adam run not the 30k Adam run). The 30k Adam trajectory matched the 5k pattern through step 8 k.

Adam at 30 k grad-evals (full run): val NLL **4.5360** final, **4.15** best intermediate at step 29 001. ORION run killed at step 90 k logical (9 k grad-evals), val NLL **4.5691** — slightly worse than Adam at the same grad-eval count.

## Why the early advantage doesn't compound

ORION's design is "1 gradient evaluation, then K closed-form α-space steps." The α-space integration is informative when:
- gradients are large (early training)
- the loss surface is approximately quadratic in the V subspace
- the FD-HVP basis is fresh

By 6 k+ grad-evals, gradients are smaller, second-order behavior is non-stationary, and Adam's running m,v second-moment estimates encode more useful curvature than ORION's r=1 fixed V basis. ORION's structural prior pays off less and less as training proceeds.

## The "step density" caveat

At iso-grad-eval, ORION's E embedding gets **10 update steps per gradient evaluation** (1 Adam + 9 α-space) while Adam's E gets 1. The 0.15–0.22 nat early advantage is achieved with that 10× E-update-density bundled in. It is not "Adam plus a free bonus"; the bonus costs ~7% wall overhead per gradient evaluation (the FD-HVP probe, projection, and lift-back). So ORION at iso-grad-eval is "Adam plus an expensive bonus that helps for the first few k grad-evals."

Iso-step comparisons across optimizer paradigms with different per-step compute (Adam vs K-step methods like ORION) are meaningless. Iso-grad-eval is the floor of fairness; iso-wall is a closer match to "what budget do I have"; iso-compute *adjusted for the extra cheap updates* would be the most rigorous test but isn't standard.

## What was right and what was wrong

**Right (this session):**
- The bf16-residual-p L=24 regression fix (`e24f6ef`) — production stack at L=24 was silently broken since iter 65; now repaired.
- The threshold-based val/log triggers in `chiron_main.cpp` — old modular `step % valEvery == 0` was incompatible with K>1 anchor cadence.
- Identifying that `--fuse-attn-per-layer` + SCFA at L=24 T=16384 produces grad spikes 10⁹–10¹⁰; switching to `--no-fuse-attn --fuse-attn-reln` recovers stable training.
- Diagonal-Adam preconditioner on ORION reduced step (per `orion_reduced_step_host`) — bare-SGD reduced step NaN'd within 200 steps; preconditioner stable through 90 k.

**Wrong (prior session, retracted):**
- The 9.15× wall-speedup headline in the deleted `ORION_PRODUCTION_T16384_L24_BENCHMARK_FIXED.md` was iso-step framing where ORION did 1/10 the gradient evaluations. The 1.9× NLL drop number was at that same iso-step axis, with ORION getting 10× the E-update-density per grad-eval. Neither was a fair production claim.

## What ORION still might be useful for

- **Short fine-tunes** where the early 0.15–0.22 nat advantage matters more than long-horizon parity. If you have a 1–2 k grad-eval budget, ORION wins.
- **Larger r** (r > 1) with rank-controlled HVP refresh might preserve the advantage further. r=1 is a 16 GB compromise; on 24+ GB hardware, full-rank ORION on attention W matrices would be testable.
- **Hybrid recipes** — use ORION for the first ~2 k grad-evals, switch to pure Adam thereafter. Code path doesn't exist; would need a new `--orion-fade-out` schedule.

## Recommendation

ORION remains a clean opt-in experimental optimizer in the trainer. It should NOT be the default. The flagship recipe remains pure Adam with the new mitigation flags:

```
--bf16-weights --bf16-grads --bf16-residual-p
--scfa --scfa-bf16-inner --scfa-bf16-outer --scfa-fuse-streams
--scfa-reln-opt --scfa-checkpoint-inner-bf16
--bf16-logits --bf16-logits-storage --int8-adam
--no-fuse-attn --fuse-attn-reln
--lr 1e-4 --warmup 500 --grad-clip 0.5
```

This 30 k run validates the stack as a working flagship recipe at L=24 T=16384 (val NLL 4.54 in 5.6 h wall). It's the right baseline for any future optimizer experiment.
