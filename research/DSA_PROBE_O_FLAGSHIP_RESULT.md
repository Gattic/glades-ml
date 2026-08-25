# DSA Probe O at Flagship Scale — Conjecture 12 FALSIFIED

**Date:** 2026-05-15
**Iter:** Ralph-loop iter 13
**Branch:** vesta5 (glades-ml), main (glades-trainer)
**Run dir:** `glades-trainer/research/runs/2026-05-15-dsa-probe-o-200step/`

## TL;DR

Probe O of paradigm #255 DSA executed on actually-trained Σ values at flagship scale (L=24, T=16384, 1B params).  **Result: Conjecture 12 is FALSIFIED.**  The commutation defect ε is essentially flat across token positions; it does NOT correlate with the Phase 8b per-position NLL gain pattern.

The Phase 8b cocycle gain (−0.60 nat mean) is **real**, but **not driven by per-position Σ non-triviality** as the DSA design assumed. The cocycle structure that produces the gain lives elsewhere in the SFA parameter set (most likely U_i frames or the residual-stream interaction with the Tikhonov solve).

This rules out the DSA gating mechanism *as designed* and forces a refinement of paradigm #255.

## Configuration

```
glades_chiron_train \
  --pretokenized --data-dir pretok-data --split train \
  --seq-len 16384 --m 2048 --layers 24 --heads 16 --dhead 256 --vocab 32000 \
  --scfa --scfa-bf16-inner --scfa-bf16-outer --scfa-fuse-streams --scfa-reln-opt \
  --bf16-grads --bf16-weights --int8-adam --bf16-logits-storage \
  --max-steps 30200 --log-every 50 --warmup 0 --grad-clip 0.50 --lr 0.0 \
  --load .../chiron_1B_T16384.step30000 \
  --no-resume-warmup --seed 42 \
  --sfa-swap-layer 18 --sfa-d-s 8 --sfa-r 4 --sfa-w 128 --sfa-n-sinks 8 \
  --sfa-solver-iters 30 --sfa-solver-step 0.5 --sfa-lambda 0.01 --sfa-gamma 0.5 \
  --sfa-train --sfa-lr 1e-3 \
  --val-every 0 --val-position-buckets 8 \
  --sfa-defect-stat
```

200 SFA training steps starting from the flagship checkpoint (step 30000 → 30200).
Wall time: 196.7 s. tok/s: 16,654. Same setup as Phase 8b but shorter and with the
defect-stat flag enabled (no val loop).

## Raw result

```
[chiron-train] done: steps=30200 total_tokens=3276800 wall=196.7s

[sfa-defect-stat] layer=18 T=16384 r=4
mean(eps) per pos-bucket [8]:  0.9248  0.9155  0.9160  0.9201  0.9224  0.9220  0.9212  0.9202

[sfa-defect-stat] probe-O:
  late/early = 1.00x
  Pearson r(eps, |dNLL|) = -0.187
  (pass: ratio>=2.0 AND r>=0.5)  →  FAIL on both criteria
```

## Per-position defect table vs Phase 8b NLL gain

| Pos bucket | ε (trained) | \|ΔNLL\| (Phase 8b) |
|-----------:|------------:|---------------------:|
| 0 | 0.9248 | 0.07 |
| 1 | 0.9155 | 1.15 |
| 2 | 0.9160 | 0.23 |
| 3 | 0.9201 | 1.68 |
| 4 | 0.9224 | 0.54 |
| 5 | 0.9220 | 1.27 |
| 6 | 0.9212 | 0.95 |
| 7 | 0.9202 | 0.82 |

ε ranges from 0.9155 to 0.9248 — a spread of **0.01** across all positions. The Phase 8b NLL pattern shows huge per-position variation (0.07 → 1.68). The two are **essentially uncorrelated** (r = -0.187, statistically indistinguishable from zero at N=8).

## What this means mechanically

The defect formula is

```
ε_i = sqrt( Σ_β ( Σ_e[β]^2 − 1 )^2 )    where e is the predecessor edge of i.
```

Empirically, ε_i ≈ 0.92 across all positions means **Σ_e^2 − 1 has roughly the same magnitude per edge regardless of which token it terminates at**. In other words, the trained Σ values are non-trivial but their non-triviality is uniform — not concentrated at the positions where SFA's NLL gain shows up.

This rules out two parts of paradigm #255 §3.1:

1. **"Σ-divergence at pos 3-7 vs Σ ≈ 1 at pos 0-1"** — FALSE. Σ-divergence is uniform across positions.
2. **"The position-stratified NLL pattern reflects a position-stratified cocycle pattern"** — FALSE *at the Σ level*. The cocycle gain is real but its mechanism is not per-position Σ structure.

The Phase 8b cocycle gain (−0.60 nat) is preserved as a fact. Only the proposed *explanation* (per-position Σ defect) is falsified.

## Where the cocycle gain likely lives

Two candidates remain consistent with both Phase 8b and this Probe O result:

### Candidate 1: Per-position stalk frames U_i

U_i ∈ R^{d_s × r} varies per position. The trained model could be learning subspaces such that **adjacent stalks (U_{i-1}, U_i) align well for early positions but disagree for late positions**. The "cocycle obstruction" then lives in the SUBSPACE rotation, not in Σ.

Proposed alternative defect:

```
ε^U_i  =  ‖ U_i^T U_{i-1} − I_r ‖_F     (off-diagonal mass of the adjacent-frame Gram product)
```

If U_{i-1} and U_i span similar subspaces, ε^U_i is small. If they diverge (e.g., late positions adapt to different content), ε^U_i is large. This is testable with the same trainer infrastructure (one new CUDA kernel + flag).

### Candidate 2: Per-position residual-stream interaction with the Tikhonov solve

The source `b_i = U_i U_i^T P_q q_i + γ P_v v_i` is position-dependent through q_i and v_i. The Tikhonov solve `(L_F + λI)^{-1} b` smears information across the token graph. The per-position NLL gain at SFA could come from **how the residual-stream q_i and v_i interact with the spectral filter**, not from any property of L_F itself.

In this view, the cocycle expressivity gain is in the *coupling* between SFA and the residual stream — not in L_F's intrinsic structure. The DSA gating would need to be driven by a residual-stream signal (e.g., per-position attention entropy or per-position projection norm), not by Σ-level structure.

## Implications for paradigm #255

DSA's design is now **partially falsified**:

- ❌ §2.1 ε_i formula (Σ_e^2 − 1) — empirically uninformative at the per-position level.
- ❌ §3.1 Conjecture 12 (defect-NLL correspondence) — Pearson r = −0.19, target was ≥ 0.6.
- ❌ §3.1 Phase-8b-aligned synthesis test — was a sanity check on the formula; the formula is internally consistent but operates on the wrong signal.
- ✓ §3.2 Position-1 regression motivation — still valid; uniform SFA still has a position-1 problem regardless of defect explanation.
- ✓ §4.1 Cost model — still valid; the kernel is cheap (196.7 s for 200 steps WITH the kernel call adds <1% overhead).

DSA can be **rescued with a different defect signal**. The two candidates above (U-frame divergence ε^U, or residual-stream-norm ε^q) are concrete next experiments. Both are testable with the existing trainer + new kernel.

## The "magnitudes" goal is unaffected

Phase 8b's measured cocycle gain (−0.60 nat mean / −1.50 peak) is unchanged. The mechanism question is open but the empirical magnitude is real. The paradigm program's projected stacking still applies — DSA is one of six paradigms, and its specific *gate-driver* is what's been falsified, not the underlying SFA mechanism.

## Next iter

1. **Refine the defect formula** — implement ε^U_i (U-frame divergence) as a candidate. Add a `--sfa-defect-mode {sigma,frame,residual}` flag.
2. **Re-run Probe O** with mode=frame.
3. If frame defect correlates with Phase 8b NLL → paradigm #255 design corrected. If not → consider candidate 2 (residual-stream).
4. If both fail → DSA's central premise (defect-driven gating) is fundamentally wrong, and the gate must be learned end-to-end from the loss (still possible but loses the interpretability of a closed-form signal).

## How to apply

- The defect-stat flag is shipped and validated (kernel works, parity test PASSES, flag wires correctly to the trainer). The infrastructure is right; only the FORMULA needs rethinking.
- The unit test `sfa-defect-parity` continues to pass — the kernel matches the CPU reference. The reference is correct per the math derivation. The math derivation just doesn't map to the right cocycle-relevant quantity in practice.
- Future paradigms should **always run Probe O on real trained parameters** before claiming a signal correlates with NLL. Synthetic Phase-8b-aligned tests are NECESSARY but NOT SUFFICIENT.
- This is a successful negative result — exactly what Gate-0 is designed to catch cheaply (10 min trainer run, <1% overhead).

## Files

- `glades-trainer/research/runs/2026-05-15-dsa-probe-o-200step/train.log` — full trainer log (~50 lines).
- `glades-ml/research/PARADIGM_SHIFT_255_DESIGN.md` — to be updated with this falsification note.
- `glades-trainer/trainer/chiron_main.cpp` — `--sfa-defect-stat` flag implementation.
- `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_sfa.{h,cu}` — `sfa_defect_step1_fp32` kernel.

Reproducer: see configuration above.
