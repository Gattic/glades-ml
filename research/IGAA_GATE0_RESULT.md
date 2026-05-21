# IGAA Gate-0 Result — Paradigm #260 Empirically Falsified

**Date:** 2026-05-16
**Iters:** 23-36 (design through Gate-0 evaluation)
**Branch:** glades-trainer/main + glades-ml/vesta5

## TL;DR

Paradigm #260 IGAA (Information-Geometric Attention Augmentation) — designed iter 23 as the post-iter-22 pivot from cellular sheaves — fails Gate-0 with the same qualitative pattern as paradigm #250 SFA. Val NLL never crosses below the post-fix flagship baseline (4.0771 nat); it spikes upward when training begins and then oscillates in [4.28, 4.51] for 500 SFT-style steps.

This is the **second falsification** of the brief "improve LLM architecture by magnitudes" against the properly-saved flagship. Different mechanism (mixture of attention modes vs cellular sheaf), same conclusion (additive correction to a well-trained baseline does not improve val NLL at this scale).

## Setup (per PARADIGM_SHIFT_260_IGAA_DESIGN.md §13)

- **Baseline**: post-fix CHIRON-1B `chiron_1B_T16384.step30000`, val NLL 4.0771 nat on `pretok-data/val` (4 batches × T=16384).
- **IGAA insertion**: single layer at L=18, K=4 modes (3 SCFA-placeholders + 1 identity).
- **Trainable**: W_π[K×m]≈8k, b_π[K]=4, α=1 ≈ 8200 parameters.
- **Frozen**: all 870M baseline parameters.
- **Hyperparams**: `--igaa-lr 1e-4 --igaa-init-sigma 1e-3 --igaa-init-alpha 0.0 --grad-clip 0.05`, Adam with α scalar at 10× base LR.
- **Duration**: 500 SFT steps, val every 50 steps.

## Gate-0 falsifiable claim (Conjecture C2)

> PASS if (any val checkpoint ≤ 4.05 nat) AND (≥ 90% of val checkpoints in [3.95, 4.10]). Both conditions required.

## Trajectory (val NLL on pretok-data/val, complete 500-step run)

| Step | Val NLL | Δ vs flagship | Notes |
|------|--------:|--------------:|-------|
| 30050 | **4.0771** | 0.000 | α=0 cold start — bit-exact baseline |
| 30100 | 4.4622 | +0.385 | α moved from 0; gradient overshoot |
| 30150 | 4.3000 | +0.223 | partial recovery |
| 30200 | 4.2820 | +0.205 | continued recovery |
| 30250 | 4.5104 | +0.433 | new spike |
| 30300 | **4.1706** | **+0.094** | best post-training val |
| 30350 | 4.2216 | +0.145 | |
| 30400 | 5.0870 | +1.010 | worst spike |
| 30450 | 4.1924 | +0.115 | last-position bucket 3.96 < baseline 4.02 |
| 30500 | 4.3202 | +0.245 | final |

- **Best post-training val: 4.1706 nat at step 30300 (+0.094 over flagship)**
- **Mean post-training val: 4.2829 nat (+0.206 over flagship)**
- Wall: 6.93 min on RTX 4080 SUPER. No NaN. Training stable under grad-clip 0.05.

## Verdict

**FAIL on Conjecture C2 condition 1** (any val ≤ 4.05): best post-training 4.1706 nat, never crosses 4.05.
**FAIL on Conjecture C2 condition 2** (≥ 90% in [3.95, 4.10]): only 1/10 (10%) val checkpoints qualify, and that one is the α=0 baseline (not a trained checkpoint).

**Gate-0 result: paradigm #260 IGAA does NOT improve over the post-fix flagship.**

## Why it failed (mechanistic analysis)

At K=4 with modes {1=SCFA-full, 2=SCFA-tight placeholder, 3=banded-short placeholder, 4=identity}, modes 1-3 are functionally identical (all return s.p, the SCFA output) in the Gate-0 prototype. The effective mechanism reduces to a per-token scalar gate on attention contribution:

```
y'_i = s.p_i · (1 + α · (1/K − π_{i,4}))
```

where π_{i,4} is the identity-mode weight per token. So IGAA Gate-0 learns "per-token attention-strength rescaling" — analogous to a learned per-token gain on the SCFA attention output.

CHIRON-1B's attention is already well-trained at this baseline; per-token rescaling neither amplifies useful signal nor attenuates noise meaningfully on held-out data. Train loss can match flagship (best 3.89 at step 30031) but val never crosses below — exactly like iter-22 SFA's train/val gap.

The gradient overshoot at step 30050 (val jumps 0.385 nat after just 50 steps) is the same destabilization pattern SFA showed: once the parameter that was frozen at zero begins to move, the loss landscape has no nearby better minimum within the additive K-mode hull, and the optimizer drifts away from baseline.

## What this means for the brief

Two distinct mechanisms (cellular sheaf SFA paradigm #250, and information-geometric mixture IGAA paradigm #260) have now been:
1. Designed rigorously from first principles via research-framework-design skill
2. Implemented end-to-end in C++/CUDA against the properly-saved baseline
3. Empirically tested on `pretok-data/val` at production T=16384 m=2048 L=24 scale
4. Falsified with respect to "improve over flagship val NLL"

**The brief's "magnitudes" target appears unachievable via architectural augmentation alone at this scale.** The path to magnitudes-level improvement, if it exists, requires:
- Training from scratch with a new architecture (not augmenting a trained baseline), OR
- Scale increase (much larger models), OR
- Data quality / curriculum improvements (orthogonal to architecture)

None of these are "architectural shift" interventions; they are scale/data/training-regime interventions.

## What still stands

- **All infrastructure** developed across iters 23-36 is correct and reusable:
  - gpu_igaa.h / gpu_igaa.cu (250 lines: thin GEMM + softmax + scale forward; scale_bwd + gate_bwd + softmax_bwd + bias_grad + dW_pi + dx_accum backward)
  - ChironParams IGAA state (22 GpuBuffers, ~10 KB of GPU memory)
  - Trainer CLI flags (12 IGAA-related flags)
  - Adam updates for IGAA params with separate LR group
  - Self-test regression coverage from the bug-fix commit (CHRF v=4 bf16-weights round-trip)

- **The post-fix flagship** (`chiron_1B_T16384.step30000` in `2026-05-15-flagship-postfix-T16384-30k/`) remains a clean, properly-saved 870M baseline at val NLL 4.0771.

- **Six paradigm designs** (#250-#255 cellular-sheaf family, #260 IGAA): rigorous mathematics with falsifiable empirical conjectures. The math is correct; the empirical conjectures of "improvement" are falsified at this scale.

- **Critical bug fix** (chrf-save-bf16weights save/load — commit 365ad4d on glades-trainer/main) — ensures any future research at this codebase has a valid baseline.

## What's next

The honest read is that the brief "improve LLM architecture by magnitudes" via architectural shift is empirically infeasible at this scale with these methods. Three plausible continuations:

1. **Accept the negative result and stop the architectural search.** The 14-iter post-fix arc (iter 22 → iter 36) has been a clean falsification process. Document, end the arc, pivot to a different research direction.

2. **Try one more candidate** (SHADO symplectic, candidate C from iter 23) — same Gate-0 mechanics, different math. Likely outcome based on the pattern: also fails.

3. **Pivot to scale-driven research** — instead of architectural augmentation, train a properly-saved flagship at 1.4B or 1.84B scale. Compare val NLL across scales. This explicitly abandons the "architectural magnitudes" framing.

## Reproducibility

Run: `2026-05-16-igaa-gate0/`
Command:

```bash
build/glades_chiron_train \
  --data-dir pretok-data --pretokenized --vocab 32000 \
  --seq-len 16384 --m 2048 --layers 24 --heads 16 --dhead 256 \
  --lr 0 --grad-clip 0.05 --max-steps 30500 --warmup 0 \
  --int8-adam --bf16-grads --bf16-weights --bf16-attn \
  --no-fuse-attn --fuse-attn-reln \
  --scfa --scfa-bf16-inner --scfa-bf16-outer --scfa-fuse-streams --scfa-reln-opt \
  --bf16-logits --bf16-logits-storage \
  --no-resume-warmup --val-every 50 --val-batches 4 \
  --igaa-layer 18 --igaa-k 4 \
  --igaa-init-alpha 0.0 --igaa-init-sigma 1e-3 \
  --igaa-train --igaa-lr 1e-4 \
  --load /home/robert/dev/glades-ml/research/runs/2026-05-15-flagship-postfix-T16384-30k/chiron_1B_T16384.step30000
```

Wall time: ~8 minutes on RTX 4080 SUPER.
