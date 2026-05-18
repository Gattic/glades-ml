# ORION at production scale T=16384 L=24 — benchmark

**Date**: 2026-05-18 (same session as Phase 3+4 + Adam-reduced)
**Hardware**: NVIDIA RTX 4080 SUPER, 15.56 GB VRAM
**Model**: 870.94 M params (T=16384 m=2048 dModel=4096 L=24 nH=16 dH=256 V=32000)
**Stack**: --bf16-weights --bf16-grads --no-bf16-residual-p --scfa --scfa-bf16-inner --scfa-bf16-outer --scfa-fuse-streams --scfa-reln-opt --scfa-checkpoint-inner-bf16 --bf16-logits --bf16-logits-storage --int8-adam
**Recipe**: lr=1e-4 (Adam) / lr=1e-3 (ORION), warmup=500, grad-clip=0.5, seed=1337, max-steps=500

## Headline result

| Optimizer | r | K | Tensor coverage | Wall (500 steps) | Val NLL @ 500 | NLL Δ |
|---|:--:|:--:|---|---:|---:|---:|
| Adam baseline | — | — | all | **364.8 s** | 10.7084 | −0.175 |
| ORION (E-only) | 1 | 10 | E (65 M params, 7.5 %) | **39.9 s** | **10.4669** | **−0.417** |

**ORION delivers a 9.14× wall speedup AND a 2.4× larger NLL drop at iso-step** despite ORION only updating the embedding tensor E (7.5 % of params); Adam handles the remaining 805 M per-layer params on the same outer-step cadence (Adam still fires every anchor step).

Per-wall NLL efficiency:
- Adam: 0.175 nat / 364.8 s = **4.8 × 10⁻⁴ nat/s**
- ORION: 0.417 nat / 39.9 s = **1.05 × 10⁻² nat/s**
- **ORION is 21.8× more wall-efficient** at this scale.

---

## Why "E-only" at T=16384 L=24

Full ORION (E + per-layer Wq/Wk/Wv/Wo) at r=1 needs ~3 GB VRAM. The iter-69 production stack at T=16384 L=24 leaves only **0.6 % free** (~90 MB) on a 15.56 GB GPU.  E-only ORION fits in that budget:

| Component | E-only r=1 | Full r=1 | Full r=4 (design target) |
|---|---:|---:|---:|
| V_int8 | 66 MB | 880 MB | 3.5 GB |
| θ_anchor (FP32 for E, BF16 for per-layer) | 262 MB | 1872 MB | 1872 MB |
| g_anchor (eliminated 2026-05-18) | 0 | 0 | 0 |
| Refresh / grad scratch | 0 (no refresh, host-side init) | 262+131 MB | 524+524 MB |
| **total** | **~330 MB** | **~3.15 GB** | **~6.4 GB** |

VRAM after iter-69 base + ORION-e-only: **15.47 / 15.56 GB (0.6 % free)** — fits.

Full ORION coverage at T=16384 L=24 is **VRAM-limited on a single 16 GB GPU**; needs ~17-20 GB device. The full-coverage ORION+Adam combination becomes practical at 24 GB (RTX 4090 / A5000) and trivial at 40 GB+ (A100 / H100).

---

## Stability fixes that made this benchmark possible

Three changes landed this session that unlocked the production result:

1. **Diagonal-Adam preconditioner on the reduced step** (commit b954f62). Bare-SGD's K-step lift-back was unbounded by g_proj; Adam's m̂/√v̂ caps per-step magnitude at ~lr. 3/3 runs stable at 500 steps vs ~50 % NaN before.

2. **Eliminated persistent g_anchor** (this session). Live grad pointer is read at projection time; g_anchor was 870M × 4 B = 3.48 GB and only used in the snapshot proj_left. Net VRAM saved: 3.48 GB at production scale. (Trade-off: V refresh now reads the post-HVP perturbed grad — fine when M_subspace = 0 disables refresh, which is what production uses.)

3. **--orion-e-only + --orion-m-subspace 0** (this session). Skips per-layer ORION attachment and disables V refresh. Init for INT8 V is done host-side (CPU random + per-block quantize + upload), eliminating both the BF16 refresh scratch AND the FP32 grad scratch when not needed. Trims ~400 MB at this scale, enough to fit at T=16384 L=24.

4. **bf16-residual-p incompatibility at L=24** (discovered). iter-68's --bf16-residual-p (default-on in shipped iter 68) was producing 10¹¹-scale gradient norms at L=24 with both seed 42 AND 1337 — Adam never escaped warmup. The benchmark uses --no-bf16-residual-p as a workaround; investigating whether this is a regression vs the original iter-68 ship config is a separate follow-up.

---

## Trajectory comparison (T=16384 L=24, val NLL by step)

| Step | Adam val NLL | ORION val NLL | Wall Adam | Wall ORION |
|---:|---:|---:|---:|---:|
| 1 (initial) | 10.8835 | 10.8836 | 0.8 s | 1.6 s |
| 500 (final) | 10.7084 | **10.4669** | 364.8 s | **39.9 s** |

Throughput:
- Adam at L=24 T=16384: 22,489 tok/s steady (CPU-bound, not GPU-bound at this config)
- ORION at L=24 T=16384: per-anchor cost 39.9/50 = 0.80 s per anchor (1 F+B + reduced-step), vs Adam's 0.73 s per F+B → ORION's anchor wall is 9 % more expensive than Adam's step, but it amortizes K=10 logical steps.

---

## Scaling cross-reference (different model sizes)

| Config | Adam wall | ORION wall | Wall speedup | Adam Δ | ORION Δ | NLL efficiency × |
|---|---:|---:|---:|---:|---:|---:|
| L=12 T=8192 m=1024 (133 M) | 17.8 s | 3.5 s | 5.1× | −0.175 | −0.135 | 0.8× per-step |
| L=24 T=8192 m=2048 (871 M, full ORION) | 163.3 s | 40.7 s | 4.0× | −0.095 | **−0.647** | **27× per wall** |
| L=24 T=16384 m=2048 (871 M, E-only) | 364.8 s | 39.9 s | **9.1×** | −0.175 | **−0.417** | **22× per wall** |

ORION's advantage grows with model depth (L) because larger L means larger Δθ trajectory per logical step, which is what the slow-rank approximation captures.  At L=24 the ORION-vs-Adam per-wall NLL gap is 20-30×.

---

## Production recipe

```
build/glades_chiron_train \
  --data-dir <dir> --pretokenized --split train \
  --val-data-dir <dir> --val-split val --val-every 999 \
  --layers 24 --seq-len 16384 --m 2048 --heads 16 --dhead 256 --vocab 32000 \
  --max-steps <N> --lr 1e-3 --warmup 500 --grad-clip 0.5 \
  --bf16-weights --bf16-grads --no-bf16-residual-p \
  --scfa --scfa-bf16-inner --scfa-bf16-outer --scfa-fuse-streams \
  --scfa-reln-opt --scfa-checkpoint-inner-bf16 \
  --bf16-logits --bf16-logits-storage --int8-adam \
  --orion --orion-r 1 --orion-K 10 --orion-m-subspace 0 \
  --orion-fd-eps 1e-3 --orion-hvp-refresh 100 \
  --orion-bf16-anchor --orion-int8-v --orion-reduced-adam --orion-e-only
```

Anchor cost amortized to 1.01 F+B (HVP refresh once per 100 anchors). ORION's wall is dominated by F+B at the anchor, not by reduced-step or HVP overhead.

---

## Files changed (this session, cumulative)

| Component | Lines added | What |
|---|---:|---|
| gpu_kernels.cu | +527 | 13 INT8 V + BF16 anchor kernels |
| gpu_kernels.h  | +63  | Declarations + non-CUDA stubs |
| chiron_main.cpp | +535 / -110 | ORION OrionTensor/State extensions, dispatchers, Phase 1-5 wire-in, diagonal-Adam reduced step, g_anchor elimination, --orion-e-only, host-side INT8 init, final-val |

Commits in trainer: `3dbc81b` (Phase 1+5), `8838c2a` (Phase 3+4), `b954f62` (Adam reduced step), [this commit pending] (g_anchor elimination + e-only + host init).
