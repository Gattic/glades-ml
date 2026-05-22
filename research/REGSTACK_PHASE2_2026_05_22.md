# Regstack Phase 2 — Z-loss + QK-Norm Pilot Arc (2026-05-22)

**STATUS: IN PROGRESS — pilot arc running.** Final verdict (PASS/FAIL)
will be set once B4 5k pilot completes and the §3.3 decision tree
selects B5.

This doc reports the regularization-stack validation arc per
`docs/superpowers/specs/2026-05-22-chiron-1b-regularization-stack-design.md`
(spec) and `docs/superpowers/plans/2026-05-22-chiron-1b-regularization-stack.md`
(plan).

## Scope Note: MTP Deferred

The original spec called for B3 (MTP, depth=1, λ_mtp=0.1) as a third
per-mechanism pilot and B4 as the three-way stack. **MTP was deferred
from this arc** because at the production shape (T=16384, V=32000,
m=2048, L=24) the MTP scratch buffers (`logitsMtp`, `probsMtp`,
`dLogitsMtp` at T·V FP32 ≈ 2 GB each, plus `hMtp`/`dHmtp` at T·dModel
≈ 256 MB each) total ~2.6 GB on top of the v5+FP8 ship's 14.97 GB
working-set, which exceeds the 15.6 GB available on the RTX 4080
SUPER. A chunked-T or sparse-T MTP implementation is needed for
production-shape inclusion; that's a follow-up arc, not part of this
Phase 2.

Pilot arc therefore runs **B0 / B1 / B2 / B4 only**, with B4 testing
Z-loss + QK-Norm stacked (the two-way stack, not three-way).

The decision tree at spec §3.3 is adapted accordingly:
- if Δ4 ≥ max(Δ1, Δ2): B5 := B4 (stacked, additive or super-additive)
- elif Δ4 ≥ 0.02: B5 := B4 (sub-additive but clears gate)
- elif max(Δ1, Δ2) ≥ 0.02: B5 := single best
- else: no B5 — publish negative result and close.

## Trainer Wiring Fixes (Pre-Arc)

Three production-recipe bugs surfaced and were fixed as preconditions
of the pilot arc:

1. **QK-Norm bypassed on `--scfa-bf16-inner`** (glades-trainer
   `6276932`). The existing FP32-inner QK-Norm path in `chiron_main.cpp`
   was unreachable on the flagship recipe — `useBf16Inner=true` took
   the monolithic `chiron_attention_shear_bf16w_tiled` branch before
   the QK-Norm check.

   Fix: added a new BF16-inner + QK-Norm branch in both forward
   (`scfa_attention_forward`) and backward (`scfa_attention_backward`)
   that decomposes the monolithic kernel into BF16-TC Q/K/V projections
   + `flash_attention_cublas_tiled_bf16` + BF16-TC output projection,
   inserting `qknorm_forward_gpu` + `scale_q_per_head` between
   projection and attention core.

   Backward recomputes qNorm/kNorm via BF16-TC projection +
   `qknorm_forward_gpu` (saves the L·k·dModel storage that would
   otherwise need ~768 MB at production shape) and snapshots qNorm
   in a k·dModel temp overlay on `scfa_qpar` for the γ-grad step.

2. **Z-loss bypassed on `--bf16-logits-storage`** (glades-ml
   `7f183ba1f`, glades-trainer `c6ab2d2`). The existing Z-loss path
   used `softmax_forward_with_lse` and `softmax_cross_entropy_bwd_zloss`
   which operate on FP32 logits/probs/dlogits. The flagship recipe
   uses `--bf16-logits-storage` — `chiron_main.cpp`'s BF16 readout
   path returned BEFORE reaching the Z-loss FP32 branch, so `--zloss-coef`
   was a silent no-op on production.

   Fix: added `softmax_forward_bf16_with_lse` and
   `softmax_cross_entropy_bwd_bf16_zloss` library kernels (mirrors of
   the FP32 versions on BF16 inputs/outputs + FP32 logZ sidecar), and
   wired them into the `--bf16-logits-storage` branches in chiron_main.

3. **`softmax_forward_with_lse` helper missing in glades-ml**
   (glades-ml `2b0818db5`). The Task 4A Z-loss port to chiron_main
   needed a softmax-with-LSE variant that wasn't in the library.

## Pre-Arc Propagation Verification

20-step seed=1337 flagship-recipe smoke (after fixes), measured against
the unmodified baseline OFF (10.5461 / 10.4013 at step 1/5):

| Config | step 1 loss | step 5 loss | Δ vs B0 step 5 |
|---|---:|---:|---:|
| B0 baseline | 10.5461 | 10.4013 | 0 |
| B1 Z-loss 1e-4 | 10.5572 | 10.4121 | +0.0108 nat |
| B2 QK-Norm γ=14 | 10.7612 | 10.6621 | +0.2608 nat |
| B4 stacked | 10.7728 | 10.6734 | +0.2721 nat |

The B1 shift matches `zlossCoef · log²(V) = 1e-4 · log²(32000) ≈
0.011 nat` (the per-token zloss penalty term at uniform logits).
The B2 shift reflects γ_init = log₂(T) = log₂(16384) = 14, which
scales Q by 14·√(dHead) = 14·16 = 224 at initialization — a large
attention temperature change that produces ~0.26 nat extra loss
initially but is expected to recover during training (the spec
designed γ_init this way for SCFA's compressed inner attention).
B4 is approximately additive (B4 − B1 ≈ B2 − B0 within rounding).

All four configs: no NaN/inf, no OOM, tok/s within 2% of baseline.

## 5k Pilot Results

**Status: pending (in flight at ~$(date))**

Pilots launched via `run_regstack_pilots.sh` (sequential single-seed
flagship recipe). Save scratch under `/tmp/regstack_b{0,1,2,4}_*`;
logs in `logs/regstack_b{0,1,2,4}.log`.

| ID | Steps | Config | Val NLL @ 5k | Δ vs B0 | tok/s | Peak VRAM |
|---|---:|---|---:|---:|---:|---:|
| B0 | 5000 | baseline | TBD | 0 | TBD | TBD |
| B1 | 5000 | --zloss-coef 1e-4 | TBD | TBD | TBD | TBD |
| B2 | 5000 | --qk-norm | TBD | TBD | TBD | TBD |
| B4 | 5000 | B1 + B2 | TBD | TBD | TBD | TBD |

### Per-mechanism 5k gate evaluation

(filled in once pilots complete)

- **B1 (Z-loss)**: TBD
- **B2 (QK-Norm)**: TBD
- **B4 (stacked)**: TBD

### Decision tree application

(filled in once pilots complete)

## B5 — 30k Phase 2 Retrain

**Status: pending (gated on pilot completion + decision tree)**

| Step | Val NLL | tok/s | Peak VRAM |
|---:|---:|---:|---:|
| 5000 | TBD | TBD | TBD |
| 10000 | TBD | TBD | TBD |
| 15000 | TBD | TBD | TBD |
| 20000 | TBD | TBD | TBD |
| 25000 | TBD | TBD | TBD |
| 30000 | TBD | TBD | TBD |

### Gate evaluation (spec §3.2)

- Val NLL @ 30k ≤ 4.1517: TBD
- tok/s ≥ 27,500: TBD
- Peak VRAM ≤ 15.72 GB: TBD

## Verdict

**PENDING.** Will be updated to PASS / PARTIAL PASS (subset shipped) /
FAIL based on §3.2 gate evaluation.

## Reproduce

```bash
# Pilot arc (4 × 5k single-seed):
cd ~/dev/glades-trainer && bash run_regstack_pilots.sh

# B5 30k retrain (config TBD per decision tree):
cd ~/dev/glades-trainer && sh run.sh flagship \
    --steps 30000 --seed 1337 \
    <decision-tree-flags> \
    --save database/checkpoints/chiron_1B_T16384_regstack_phase2/chiron_1B_T16384
```

## Related Commits

### glades-ml (chiron2 branch)
- `2b0818db5` Add softmax_forward_with_lse helper required by chiron_main Z-loss port
- `7f183ba1f` Add BF16-storage Z-loss kernels: softmax_forward_bf16_with_lse + bwd_zloss

### glades-trainer (main branch)
- `6276932` Wire QK-Norm into --scfa-bf16-inner attention path
- `c6ab2d2` Wire Z-loss into --bf16-logits-storage forward/backward path
- `e8e1002` Add regstack 5k pilot arc driver (B0/B1/B2/B4)
