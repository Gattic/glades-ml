# SCFA T=4096 at 1B — stability validation, 2026-05-13

## TL;DR

At T=4096, m=2048, L=24, the default `--fuse-attn-per-layer` causes
divergent gradient growth (52k → 249k in 250 steps post-warmup) — the
trainer's pre-existing warning was correct.  Switching to single-layer
fuse (`--no-fuse-attn --fuse-attn-reln`) keeps gradients bounded and
delivers the best CHIRON 1B loss trajectory recorded so far: ema=5.91
at 5000 steps / 20.48M tokens, 4.37× the throughput of the T=512
baseline, and **−0.99 nat at matched token budget** (5.12M).

This config is now the recommended long-context default.

## Bench config (both attempts)

```
T=4096 m=2048 L=24 nH=16 dH=256 V=32000     (870M params)
--int8-adam --bf16-weights --bf16-grads --bf16-attn
--grad-clip 0.5 --lr 1e-4 --warmup 500
--accum 1 --scfa --scfa-compression-ratio 16
--max-steps 5000 --log-every 250
--save research/runs/2026-05-13-scfa-T4096-1B[-reln]/chiron_1B_T4096[_reln]
--save-every 1000 --save-keep-last 3 --save-full
seed=1337  (fresh init both runs)
```

## Attempt 1: default `--fuse-attn-per-layer` — DIVERGENT

```
step    1: loss=10.87 ema=10.87 ‖g‖=    7.1   warmup
step  250: loss=10.78 ema=10.77 ‖g‖=   51.4   warmup
step  500: loss=10.33 ema=10.39 ‖g‖=52225.4   <-- full lr reached, runaway begins
step  750: loss=11.43 ema=11.31 ‖g‖=249209.4  <-- 5× growth in 250 steps; ema climbing
```

Killed at step 750.  ‖g‖ growing exponentially; the grad clip prevented
the update from blowing up but produced an effective zero-update regime
in which the model can only accumulate noise.

The trainer prints this exact warning on startup when `--scfa +
--fuse-attn-per-layer + L=24 m=2048` is active.  This run confirms it
empirically — the warning is load-bearing.

## Attempt 2: `--no-fuse-attn --fuse-attn-reln` — STABLE, FULL 5000 STEPS

```
step    1: loss=10.42 ema=10.42 ‖g‖=3.4   acc=0.0002  warmup
step  250: loss= 9.14 ema= 9.13 ‖g‖=3.5   acc=0.028
step  500: loss= 8.31 ema= 8.49 ‖g‖=4.6   acc=0.081   full lr, clean post-warmup
step 1000: loss= 7.72 ema= 8.06 ‖g‖=2.8   acc=0.090   first save
step 1250: loss= 7.42 ema= 7.65 ‖g‖=4.4   acc=0.097   <-- 5.12M tokens
step 1500: loss= 7.00 ema= 7.13 ‖g‖=1.7   acc=0.105
step 2000: loss= 6.63 ema= 6.79 ‖g‖=1.8   acc=0.106
step 2500: loss= 6.70 ema= 6.72 ‖g‖=2.3   acc=0.100   minor noise uptick
step 3000: loss= 6.20 ema= 6.37 ‖g‖=2.1   acc=0.113   third save
step 3500: loss= 6.39 ema= 6.26 ‖g‖=2.5   acc=0.114
step 4000: loss= 6.56 ema= 6.35 ‖g‖=3.4   acc=0.113   fourth save; best=5.52
step 4500: loss= 5.86 ema= 5.97 ‖g‖=4.9   acc=0.129   <-- ema broke 6.0
step 5000: loss= 5.94 ema= 5.91 ‖g‖=2.7   acc=0.115   final; best=5.52@3592
```

- Wall: 1477.9 s = 24.6 min
- Tokens: 20.48M
- Sustained throughput: 14,039 tok/s (4.37× baseline T=512 at 3,208)
- ‖g‖ after warmup: range [1.7, 4.9], no spikes
- Best single-step loss: 5.52 at step 3592

## Why the divergence happens

CHIRON's `--fuse-attn-per-layer` does, at every layer:

```
p_l = attn_shear(q, p, l)              (writes p)
q   = q + α · reln_p(p)                where α = 1/√L
q   = reln(q, l)
```

At backward, dq accumulates contributions from every layer's
reln_p(p)-via-α-axpy.  With SCFA at T=4096 m=2048 L=24, the inner
attention's backward chain reduces the per-channel variance compared
to standard attention (because SCFA does the matmul at compressed
length k=256), so the standard α=1/√L scaling under-damps the
contribution.  The resulting dWq/dWk/dWv gradients grow by ~√k per
layer instead of the assumed O(1) scale, and the L=24 cascade
amplifies that into a 10⁵+ ‖g‖ pulse the moment full lr is reached.

Single-layer fuse (`--fuse-attn-reln` only) applies q←q+p exactly once
at layer L-1 instead of at every layer, breaking the cascade.  The
gradient stays bounded throughout training.

## Comparison vs the 50k T=512 baseline

The 50k baseline at `research/runs/2026-05-13-chiron-1B-save/` used
T=512 with `--fuse-attn-per-layer` and ran for ~2.75 hr / 25.6M tokens:

| Token budget | T=512 baseline ema | T=4096 SCFA + single-fuse ema | Δ      |
|-------------:|-------------------:|------------------------------:|-------:|
| 5.12M        | 8.64 (best stable) | 7.65                          | −0.99  |
| 20.48M       | 10.55 (oscillating)| 5.91                          | −4.64  |
| 25.6M        | 11.62 (.final peak)| (not run yet)                 | —      |

The T=4096 run never approached the oscillation regime that dominates
the T=512 baseline's late training.  Plausible mechanisms:

1. Longer context provides more in-context signal per gradient step.
2. Single-layer fuse removes the L-amplified gradient noise that drove
   the T=512 baseline's oscillations.
3. Wider effective batch (4096 tokens/step vs 512) reduces variance.

We can't disentangle (1) from (2) from this run alone — that needs a
matched T=512 + single-fuse comparison (open question).

## Output

Checkpoints at `/home/robert/dev/glades-ml/research/runs/2026-05-13-scfa-T4096-1B-reln/`:

```
chiron_1B_T4096_reln.step3000   3.5 GB   ema≈6.37
chiron_1B_T4096_reln.step4000   3.5 GB   ema≈6.35  (best=5.52 already)
chiron_1B_T4096_reln.step5000   3.5 GB   ema=5.91  (best=5.52@3592)
chiron_1B_T4096_reln.final      3.5 GB   = step5000
```

**`chiron_1B_T4096_reln.step5000` is the lowest-NLL CHIRON 1B checkpoint
to date.**  Use as inference target, DISTILL-FORWARD teacher (paradigm
#56), or warm-start for an extended run.

## Recommended new long-context default

```
--seq-len 4096 --m 2048 --layers 24 --heads 16 --dhead 256 --vocab 32000
--int8-adam --bf16-weights --bf16-grads --bf16-attn
--grad-clip 0.5 --lr 1e-4 --warmup 500
--accum 1 --scfa --scfa-compression-ratio 16
--no-fuse-attn --fuse-attn-reln
```

For the next 50k production run, extend `--max-steps 50000` and add
`--lr-decay` for cosine schedule.

## Open follow-ups

1. **Long-horizon stability.**  The 50k T=512 baseline started
   oscillating around step 35k.  Need a matched 50k run at T=4096 +
   single-fuse to confirm long-horizon behavior.
2. **T=512 + single-fuse comparison.**  Disentangle whether the NLL
   win is from T (longer context) or fuse mode (single-layer).
3. **T=8192 / T=16384.**  SCFA's design regime extends past T=4096;
   does VRAM allow + does stability hold?
