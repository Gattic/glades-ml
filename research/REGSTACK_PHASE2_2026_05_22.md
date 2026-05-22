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

Pilots ran 2026-05-22 11:32 → 14:46 EDT (3h 14m total) via
`run_regstack_pilots.sh`. Sequential single-seed on flagship recipe.
Logs at `~/dev/glades-trainer/logs/regstack_b{0,1,2,4}.log`.

### Aggregate val NLL + Δ vs B0

| ID | Config | Val NLL @ 5k | Δ vs B0 | acc1 | acc5 | acc10 |
|---|---|---:|---:|---:|---:|---:|
| B0 | baseline | 4.9140 | 0 | 0.1204 | 0.4645 | 0.6850 |
| B1 | --zloss-coef 1e-4 | 4.9207 | **+0.0067** | 0.1206 | 0.4624 | 0.6832 |
| B2 | --qk-norm (γ=log₂T=14) | 3.9412 | **−0.9728** | 0.1258 | 0.5025 | 0.7766 |
| B4 | B1 + B2 stacked | 3.9396 | **−0.9744** | 0.1263 | 0.5029 | 0.7775 |

### Position-stratified val NLL (8 buckets across T=16384)

| Bucket | Positions | B0 | B1 | B2 | B4 | Δ B2 vs B0 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | 0–2047 | 4.89 | 4.89 | 3.81 | 3.81 | −1.08 |
| 1 | 2048–4095 | 4.85 | 4.88 | 3.84 | 3.84 | −1.01 |
| 2 | 4096–6143 | 4.92 | 4.98 | 3.92 | 3.92 | −1.00 |
| 3 | 6144–8191 | 4.85 | 4.85 | 3.93 | 3.93 | −0.92 |
| 4 | 8192–10239 | 4.87 | 4.87 | 3.93 | 3.94 | −0.94 |
| 5 | 10240–12287 | 4.85 | 4.84 | 4.00 | 4.00 | −0.85 |
| 6 | 12288–14335 | 4.96 | 4.96 | 4.07 | 4.07 | −0.89 |
| 7 | 14336–16383 | 5.12 | 5.10 | 4.02 | 4.02 | **−1.10** |

B0 shows the characteristic late-T degradation (4.85 plateau → 5.12 at
bucket 7), the failure mode QK-Norm targets. B2 flattens this and shifts
the entire profile down by ~1 nat, with deepest gains at the two extremes
(bucket 0 and bucket 7). B4 is bit-similar to B2 (≤ 0.01 nat per bucket).

### Per-mechanism 5k gate evaluation (spec §3.2)

- **B1 (Z-loss 1e-4): PASS.** Δ = +0.0067 nat. Within the ±0.02 nat
  "no regression" gate. Effectively no signal at 5k single-seed —
  consistent with the spec's R-RegStack-1 risk note (Z-loss at 1e-4
  is below the 5k noise floor, mechanism shows up at longer training).
  Propagation verified: 20-step smoke showed +0.011 nat training-loss
  shift matching `zlossCoef · log²(V)`.
- **B2 (QK-Norm γ=14): PASS by huge margin.** Δ = −0.9728 nat. This is
  ~10× the literature-reported QK-Norm gain at standard-T (typically
  −0.03 to −0.08 nat). All metrics improve in lockstep: acc1 +0.54pp,
  acc5 +3.8pp, acc10 +9.16pp. Position-stratified read confirms both
  the long-context fix (bucket 7: −1.10) and a broad training-dynamics
  improvement at all positions.
- **B4 (stacked): PASS, additive at the QK-Norm level.** Δ = −0.9744 nat.
  Z-loss contributes ~−0.0016 nat additional, essentially noise. The
  spec's "Δ4 ≥ max(Δ1, Δ2)" condition holds, just barely.

### Decision tree application (spec §3.3)

Computing Δi = main-head val NLL improvement vs B0:
- Δ1 = −0.0067 (Z-loss; technically negative but within ±0.02 noise)
- Δ2 = +0.9728 (QK-Norm; massive)
- Δ4 = +0.9744

Test: Δ4 ≥ max(Δ1, Δ2)?
  max(−0.0067, +0.9728) = +0.9728; Δ4 = +0.9744 ≥ +0.9728 ✓

→ **B5 := B4 (stacked Z-loss + QK-Norm).**

Note on the decision: B4 wins B2 by only 0.0016 nat (noise-level).
Picking B5 := B4 retains Z-loss for the 30k retrain on the assumption
that it may grow past noise at longer training. If B5 fails the 30k
gate while B2-only would have passed, a follow-up 30k retrain at the
B2-only config may be needed (acknowledged as an arc extension).

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
