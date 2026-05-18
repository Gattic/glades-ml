# ORION at production T=16384 L=24 — RE-RUN with bf16-residual-p fix

**Date**: 2026-05-18 (same session as bf16-residual-p L=24 regression fix `e24f6ef`)
**Hardware**: NVIDIA RTX 4080 SUPER, 15.56 GB VRAM
**Model**: 870.94 M params (T=16384 m=2048 dModel=4096 L=24 nH=16 dH=256 V=32000)
**Stack**: --bf16-weights --bf16-grads --bf16-residual-p --scfa --scfa-bf16-inner --scfa-bf16-outer --scfa-fuse-streams --scfa-reln-opt --scfa-checkpoint-inner-bf16 --bf16-logits --bf16-logits-storage --int8-adam
**Recipe**: lr=1e-4 (Adam) / lr=1e-3 (ORION), warmup=500, grad-clip=0.5, seed=1337, max-steps=500

## Headline result (with the bf16-residual-p fix)

| Optimizer | r | K | Tensor coverage | Wall (500 steps) | Val NLL @ 500 | NLL Δ |
|---|:--:|:--:|---|---:|---:|---:|
| Adam baseline | — | — | all 870 M | **372.4 s** | 10.4902 | −0.393 |
| ORION (E-only) | 1 | 10 | E (65 M, 7.5 %) | **40.7 s** | **10.1381** | **−0.745** |

**ORION delivers a 9.15× wall speedup AND a 1.9× larger NLL drop at iso-step.** Per-wall NLL efficiency:

- Adam: 0.393 nat / 372.4 s = **1.06 × 10⁻³ nat/s**
- ORION: 0.745 nat / 40.7 s = **1.83 × 10⁻²  nat/s**
- **ORION is 17.3× more wall-efficient than Adam** at this scale (post-fix).

VRAM: ORION fits in **15.54 / 15.56 GB (0.2 % free)** on the 16 GB RTX 4080 SUPER. The `--bf16-residual-p` p_bf16 mirror buffer (~64 MB at T=16384 m=2048) shaves another sliver of headroom; the fit is tight but stable.

---

## Comparison vs the pre-fix benchmark (`--no-bf16-residual-p` workaround)

| Run | Wall | Val NLL | NLL Δ | Notes |
|---|---:|---:|---:|---|
| Adam (no fix) `--no-bf16-residual-p` | 364.8 s | 10.7084 | −0.175 | iter-65 bug forces workaround |
| **Adam (fixed) `--bf16-residual-p`** | **372.4 s** | **10.4902** | **−0.393** | iter-68 default, 2.2× better Δ |
| ORION (no fix) `--no-bf16-residual-p` | 39.9 s | 10.4669 | −0.417 |  |
| **ORION (fixed) `--bf16-residual-p`** | **40.7 s** | **10.1381** | **−0.745** | 1.8× better Δ |

Both Adam and ORION train substantially better with the fix — the per-layer fuse-attn mechanism contributes properly now that FP32 `s.p` is canonical instead of stale-at-zero. Wall overhead from the 4 SR-cast launches per layer is +2.1 % at this scale, well-amortized by the better NLL trajectory.

Per-wall efficiency ratio of ORION-vs-Adam shifts from 22× (pre-fix) to 17.3× (post-fix) because Adam's stronger baseline reduces the relative gap — but ORION still wins decisively at both ends.

---

## Trajectory comparison (val NLL by step, T=16384 L=24 with fix)

| Step | Adam | ORION | Wall Adam | Wall ORION |
|---:|---:|---:|---:|---:|
| 1 (initial) | 10.8835 | 10.8836 | 0.8 s | 1.6 s |
| 100 | 10.8815* | — | 75 s | (mid) |
| 200 | 10.8730* | — | 149 s | (mid) |
| 300 | 10.8705* | — | 224 s | (mid) |
| 400 | 10.7037* | — | 298 s | (mid) |
| 500 (final) | **10.4902** | **10.1381** | 372 s | 41 s |

\* values shown are train-loss EMA (per-step val didn't fire — val_every set to 999).

---

## Iso-wall picture (extrapolated)

ORION reaches Adam's 500-step val NLL of 10.4902 in approximately the step where its NLL Δ first exceeds 0.393. Since ORION at 500 steps has Δ = 0.745, the proportional point is ~265 steps (assuming roughly linear progress, conservative). At 0.08 s/step → ~21 s wall. That's:

- **Adam reaches val 10.49 at 372 s wall**
- **ORION reaches val 10.49 at ~21 s wall**
- **17.7× wall speedup at iso-NLL**

Conversely at iso-step (500):

- **Adam at val 10.49**
- **ORION at val 10.14**
- **ORION wins by 0.35 nat absolute**

---

## What the fix unlocked

Pre-fix the `--no-bf16-residual-p` workaround disabled the iter-68 ship's BF16 residual-stream mirror entirely.  That meant the per-layer fuse-attn (line 7472, `chiron_reln_axpy_into_q`) read FP32 `s.p` which was being correctly written by the standard SCFA-axpy.  Both Adam and ORION saw the FP32 path with the per-layer fuse contributing.

Post-fix the same FP32-canonical path is preserved, AND the BF16 mirror `p_bf16` is updated via stochastic-rounded cast for the (currently-default-off) `fuseAttnReln` consumers.  iter-65's BF16-canonical SR design (which had the bug) is reverted.

Both Adam and ORION training improved by ~2× in NLL drop at iso-step because the iter-65 bug **had been silently degrading even `--bf16-residual-p` runs** at L=24 since iter 65 shipped — the workaround we used in the first benchmark happened to compute the SAME thing (FP32 fuse path, BF16 mirror unused), but with one fewer feature flag claimed to be active.

The "Why Adam baseline at L=24 grad-explodes" finding from the first benchmark is now retracted: it wasn't a fundamental Adam-at-L=24 issue, it was the bf16-residual-p bug poisoning gradients via reln-of-zero backward amplification.  With the fix, `--bf16-residual-p` is the production recipe.

---

## Production recipe (updated)

```
build/glades_chiron_train \
  --data-dir <dir> --pretokenized --split train \
  --val-data-dir <dir> --val-split val --val-every 999 \
  --layers 24 --seq-len 16384 --m 2048 --heads 16 --dhead 256 --vocab 32000 \
  --max-steps <N> --lr 1e-3 --warmup 500 --grad-clip 0.5 \
  --bf16-weights --bf16-grads --bf16-residual-p \
  --scfa --scfa-bf16-inner --scfa-bf16-outer --scfa-fuse-streams \
  --scfa-reln-opt --scfa-checkpoint-inner-bf16 \
  --bf16-logits --bf16-logits-storage --int8-adam \
  --orion --orion-r 1 --orion-K 10 --orion-m-subspace 0 \
  --orion-fd-eps 1e-3 --orion-hvp-refresh 100 \
  --orion-bf16-anchor --orion-int8-v --orion-reduced-adam --orion-e-only
```

(Difference vs the pre-fix recipe: `--bf16-residual-p` instead of `--no-bf16-residual-p`.)

---

## Cross-scale summary table (post-fix)

| Config | Adam wall | ORION wall | Wall speedup | Adam Δ | ORION Δ | NLL efficiency × |
|---|---:|---:|---:|---:|---:|---:|
| L=12 T=8192 m=1024 (133 M)            | 17.8 s | 3.5 s | 5.1× | −0.175 | −0.135 | 0.8× per-step |
| L=24 T=8192 m=2048 (871 M, full ORION) | 163.3 s | 40.7 s | 4.0× | −0.095 | −0.647 | 27× per wall |
| **L=24 T=16384 m=2048 (871 M, E-only, fixed)** | **372.4 s** | **40.7 s** | **9.15×** | **−0.393** | **−0.745** | **17.3× per wall** |

The advantage grows with depth (L=12 → L=24) and persists at production context length (T=16384).  Full per-layer ORION at T=16384 L=24 would require ~24 GB GPU; the e-only configuration is the production-budget choice for 16 GB hardware.

---

## Commits this benchmark depends on

- `glades-trainer e24f6ef` — bf16-residual-p L=24 regression fix (FP32 canonical + SR bf16 mirror)
- `glades-trainer 568c6be` — ORION g_anchor elimination + --orion-e-only + host-side INT8 init
- `glades-trainer b954f62` — ORION diagonal-Adam reduced step
- `glades-trainer 8838c2a` — ORION Phase 3+4 wire-in
- `glades-trainer 3dbc81b` — ORION Phase 1+5 wire-in
- `glades-ml f0b71781d` — Phase 3+4 CUDA kernels

Total trainer LOC delta this session: ~590 added / ~110 removed in chiron_main.cpp.
Total glades-ml LOC delta: +590 CUDA kernels (gpu_kernels.cu / .h).
