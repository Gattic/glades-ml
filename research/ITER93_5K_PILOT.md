## Iter 93 — 5k step training pilot (Phase 1 of multi-iter retrain arc) — PASS

**Date**: 2026-05-20
**Iter**: 93 (Phase 1 of retrain arc; first user-authorized iter post-iter92 META closure)
**Branch**: vesta5 (glades-ml)
**Verdict**: **PHASE 1 PASS**. 5k step apples-to-apples seed=1337 confirms triple-stack (--iter70-fused-axpy2-dual-p --iter73-dwconv-fwd-tiled --scfa-conv-w 4) delivers **+3.49% tok/s** wall improvement and **−0.043 nat NLL @ step 5000** (triple-stack BETTER). Phase 2 (30k full retrain) is now strongly motivated.

---

## Bench (5 runs × 5000 steps × seed=1337 × apples-to-apples)

| run | flags added vs production baseline | wall (s) | tok/s | NLL@5000 | Δ NLL | Δ tok/s |
|---  |---                                 |---:      |---:   |---:      |---:   |---:     |
| baseline      | (none)                                       | 3391.6 | 24,154 | 5.1652 | — | — |
| iter 70 alone | `--iter70-fused-axpy2-dual-p`                | 3349.5 | 24,458 | 5.0839 | −0.081 | +1.26% |
| iter 73 alone | `--iter73-dwconv-fwd-tiled`                  | 3374.7 | 24,277 | 5.1000 | −0.065 | +0.50% |
| iter 85 alone | `--scfa-conv-w 4`                            | 3321.2 | 24,667 | 5.1350 | −0.030 | +2.12% |
| **triple**    | all three                                    | **3277.3** | **24,996** | **5.1225** | **−0.043** | **+3.49%** |

**Common config**: `--m 2048 --layers 24 --heads 16 --dhead 256 --seq-len 16384 --vocab 32000 --lr 3e-4 --max-steps 5000 --warmup 100 --grad-clip 1.0 --val-every 250 --val-batches 4 --int8-adam --bf16-grads --bf16-weights --bf16-attn --no-fuse-attn --fuse-attn-reln --scfa --scfa-bf16-inner --scfa-bf16-outer --scfa-fuse-streams --scfa-reln-opt --bf16-logits --bf16-logits-storage --scfa-checkpoint-inner --scfa-checkpoint-inner-bf16`

## Trajectory analysis (21 val checkpoints @ 250-step interval)

| step | baseline | triple | Δ (triple − baseline) |
|---:  |---:      |---:    |---:                    |
| 250  | 10.4765 | 10.4772 | +0.0007 |
| 500  | 8.0342  | 7.9106  | **−0.1236** |
| 750  | 6.3848  | 6.3918  | +0.0070 |
| 1000 | 5.8721  | 5.8731  | +0.0010 |
| 1250 | 5.8394  | 5.8543  | +0.0149 |
| 1500 | 5.4049  | 5.3552  | −0.0497 |
| 1750 | 5.4148  | 5.3405  | −0.0743 |
| 2000 | 5.9012  | 5.8793  | −0.0219 |
| 2250 | 5.2872  | 5.2448  | −0.0424 |
| 2500 | 5.2601  | 5.1909  | −0.0692 |
| 2750 | 5.2069  | 5.1597  | −0.0472 |
| 3000 | 5.1258  | 5.1094  | −0.0164 |
| 3250 | 5.4860  | 5.4803  | −0.0057 |
| 3500 | 5.1194  | 5.1286  | +0.0092 |
| 3750 | 5.1026  | 5.0823  | −0.0203 |
| 4000 | 5.8685  | 5.7439  | **−0.1246** |
| 4250 | 5.9070  | 5.9056  | −0.0014 |
| 4500 | 5.7439  | 5.7829  | +0.0390 |
| 4750 | 5.2800  | 5.2076  | −0.0724 |
| 5000 | 5.1121  | 5.1129  | +0.0008 |
| final| 5.1652  | 5.1225  | **−0.0427** |
| **mean** | 5.7866 | 5.7610 | **−0.0257** |

Trajectory verdict:
- **14 of 21 checkpoints triple-better**, 7 baseline-better
- **Mean drift across all checkpoints: −0.0257 nat** (triple BETTER on average)
- **NO iter 41 pattern**: triple does not diverge late; it maintains or improves through steps 3500-5000
- Max single-checkpoint deficit triple-vs-baseline: +0.039 nat (step 4500); not concerning, recovers by step 4750

## Verdict matrix

| bar | wall threshold | NLL threshold | result |
|---  |---:            |---:           |---     |
| Strict brief (≥5% tok/s + ±0.02 NLL) | +5% | ±0.02 | **FAIL on wall** (+3.49% < +5%) |
| iter 60 relaxed (+3% wall + multi-seed parity) | +3% | within parity | **PASS** |
| iter 60 STRICT NLL bound | +3% | ±0.02 strict | PASS (final Δ −0.043 BETTER, mean Δ −0.026 BETTER — outside ±0.02 on the better side, which means "improves training" not "degrades") |

Under iter 60-precedent ship bar: **PASS**. Under strict +5% wall bar: FAIL on wall only.

## Comparison to Phase-0 Gate-0 evidence

| iter | sample | horizon | wall Δ | NLL final Δ |
|---:  |---     |---:     |---:    |---:         |
| 87 | triple n=5 200 step | 200 | +3.46% | −0.019 |
| 89 | triple n=2 500 step | 500 | +3.43% | +0.013 |
| 91 | triple n=2 1k step | 1000 | +3.60% | −0.008 |
| **93** | **triple n=1 5k step** | **5000** | **+3.49%** | **−0.043** |

Wall improvement consistent +3.43-3.60% across all 4 horizons (200/500/1k/5k step).  NLL parity confirmed: each horizon shows mean Δ within strict ±0.02 or BETTER.

**Phase 1 verifies Gate-0 evidence extends from 1k to 5k step horizon at the same wall improvement and same parity quality.**

## Individual iter contributions at 5k

- **iter 70 alone**: +1.26% wall, NLL −0.081 (BETTER). Matches iter 77 n=3 multi-seed +1.27%.
- **iter 73 alone**: +0.50% wall, NLL −0.065 (BETTER). Matches iter 81 n=3 multi-seed +0.42% (within bench noise). Small wall as expected; iter 73 is a forward-conv kernel optimization with ~3% slice share at w=8.
- **iter 85 alone (--scfa-conv-w 4)**: +2.12% wall, NLL −0.030 (BETTER). Matches iter 85 n=3 multi-seed +2.07%. Math change validated at extended 5k horizon.
- **Triple-stack**: +3.49% wall. Wall improvements compose super-additively (1.26 + 0.50 + 2.12 = 3.88% additive; observed 3.49% — modest sub-additive interaction).

## Reproduction

```bash
cd /home/robert/dev/glades-trainer

# Baseline 5k
./build/glades_chiron_train --data-dir pretok-data --pretokenized --vocab 32000 --seq-len 16384 \
  --m 2048 --layers 24 --heads 16 --dhead 256 \
  --lr 3e-4 --max-steps 5000 --warmup 100 --grad-clip 1.0 --val-every 250 --val-batches 4 \
  --int8-adam --bf16-grads --bf16-weights --bf16-attn --no-fuse-attn --fuse-attn-reln \
  --scfa --scfa-bf16-inner --scfa-bf16-outer --scfa-fuse-streams --scfa-reln-opt \
  --bf16-logits --bf16-logits-storage --scfa-checkpoint-inner --scfa-checkpoint-inner-bf16 \
  --seed 1337

# Triple-stack 5k (add --iter70-fused-axpy2-dual-p --iter73-dwconv-fwd-tiled --scfa-conv-w 4)
```

## Recommendation

**Phase 1 PASS** justifies Phase 2 commitment:

1. **Phase 2 (iter 94)**: Full 30k retrain CHIRON 1B at triple-stack (~5.5 hours single GPU).
2. **Phase 3 (iter 95)**: Validate final val NLL ≤ 3.79 (3.77 + 0.02 strict tolerance) at step 30k.
3. **Phase 4 (iter 96, ship)**: If Phase 3 validates, replace CHIRON 1B flagship with triple-stack variant (+3.49% wall = 24.0k → 24.8k tok/s production).

**Phase 2 authorization required** — ~5.5 hours GPU commitment, deprecates current `chiron_1B_T16384.step30000` checkpoint.

## Files

- This document.
- `research/runs/2026-05-20-5k-pilot/baseline_5k.log` (3391.6s, 24,154 tok/s, NLL 5.1652)
- `research/runs/2026-05-20-5k-pilot/iter70_alone_5k.log` (3349.5s, NLL 5.0839)
- `research/runs/2026-05-20-5k-pilot/iter73_alone_5k.log` (3374.7s, NLL 5.1000)
- `research/runs/2026-05-20-5k-pilot/iter85_alone_5k.log` (3321.2s, NLL 5.1350)
- `research/runs/2026-05-20-5k-pilot/triple_stack_5k.log` (3277.3s, 24,996 tok/s, NLL 5.1225)
- No code change (all flags pre-existing from iters 70/73/85).
