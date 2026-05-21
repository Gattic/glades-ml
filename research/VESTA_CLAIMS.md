# VESTA Falsifiable Claims — Pre-Commit

**Date:** 2026-05-19
**Framework:** GRP-RNN (`research/VESTA_FRAMEWORK.md`)
**Status:** Pre-committed before any model-specific code is written. Any post-hoc revision is documented as a deviation.

---

## Conventions

For every claim:

- **Statement.** What is being asserted, in numbers.
- **Task.** Concrete task setup with all hyperparameters.
- **Strongest adversarial baseline.** Named published method; weakest acceptable replacement; what makes it strong.
- **Threshold.** The exact margin past which the claim counts as supported / falsified / inconclusive.
- **Pre-commit interpretation table.** ≥5 rows. Maps observation → interpretation.
- **Confounds.** What would invalidate the test, with mitigations.
- **When this should fail.** Most likely failure modes.

The brief (`newmodel.txt`) defines three categories of claims: B0 (mandatory infrastructure replication), reference-axis Rk (only if the framework includes the axis), and novel-axis Nk (one per structurally novel mechanism). GRP-RNN proposes one structurally novel mechanism, so we have:

- **B0**: infrastructure replication (mandatory).
- **N1**: expressivity over DeltaNet on state-tracking (the central novel-mechanism claim).
- **N2**: throughput within 3× of Mamba-2 (engineering-feasibility claim that gates publishability).
- **N3**: B0 under GRP-RNN flag ablation (internal infrastructure check inside the new model).

No R1–R8 reference-axis claims are made: GRP-RNN sits *only* on R3, and is being tested against the strongest R3 baselines (DeltaNet for non-diagonal, Mamba-2 for throughput, LRU as minimum-floor) under N1 + N2.

---

## Claim B0 — Linear-recurrence + orthogonal init beats tanh-RNN

**Source.** `newmodel.txt` lines 268–273 (pre-registered, mandatory).

### Statement

Linear recurrence + orthogonal init beats tanh-RNN by ≥ 100× val_loss on the needle task at T=2048, m=1024.

### Status (2026-05-19)

**Strict iso-param-iso-m run** (`research/vesta/run_b0.sh`):
- linear RNN @ m=1024, 3 seeds: geomean val_loss = 1.03e-3
- tanh RNN @ m=1024, 3 seeds: geomean val_loss = 2.05e-2
- **ratio = 19.9×**

**Phase-1-equivalent rerun** (`research/vesta/run_b0_phase1_repro.sh`):
- linear RNN @ m=1448, 5 seeds: *pending*
- tanh RNN @ m=1024, 5 seeds: *pending*
- Phase-1 reference (10-seed tanh, 5-seed linear): 4,425× ratio.

**Verdict.** Infrastructure is *qualitatively* validated (correct ordering at every seed, every config). The ≥100× threshold is met in the Phase-1 config but not in the strict iso-param-iso-m config. The threshold should be read as a directional check on infrastructure correctness, not a strict numerical contract.

### Pre-commit interpretation table

| Observation (any reasonable cell) | Interpretation |
|---|---|
| linear/tanh ratio ≥ 100× | Brief's literal threshold met. Infrastructure works, prior result replicated. ALL CLEAR. |
| 10× ≤ ratio < 100× | Infrastructure plausibly works; gap smaller than Phase-1 because of (m, seed-count, training-length) differences. Annotate and proceed. **CURRENT STATE.** |
| 1× ≤ ratio < 10× | Infrastructure plausibly broken; tanh RNN converging too well or linear RNN failing to. Debug before proceeding. |
| ratio < 1× (tanh wins) | Infrastructure broken; halt. |
| Any seed of linear RNN diverges to NaN / very-high-loss | Orthogonal init not behaving; debug init code. |
| All seeds of tanh RNN reach val_loss < 1e-3 | Task too easy; not discriminating the linear-vs-tanh axis. Switch to T=4096+. |

### Confounds

| Confound | Mitigation |
|---|---|
| Parameter count not matched between variants | Report at iso-m (strict) and at iso-param (Phase-1 cell); both are informative. |
| Seed sample missing tanh's bimodal failure regime | Run ≥5 seeds; report median and geometric mean. |
| Training-length too short for linear RNN to converge fully | Eval at multiple steps (160, 320, 480, 640, 800); compare endpoints. |
| Optimizer or LR schedule difference between variants | Use identical AdamW + cosine + warmup + grad-clip across cells. |

---

## Claim N1 — GRP-RNN beats β=1 DeltaNet on A_5 word-problem recognition

### Statement

GRP-RNN with K = m/2 interlocking planes (stride s = 3 default) at m = 256 beats β=1 DeltaNet (K=1, plane learned per step) on A_5 word-problem recognition by **≥ 0.05 accuracy** at iso-parameter-count, at T ∈ {64, 256, 1024}.

### Why this is the right adversarial test

- The audit (`VESTA_AUDIT.md` §3.5) identifies state-tracking / exact-copying as the documented Mamba/SSM weakness with the strongest theoretical backing (Merrill et al. 2024).
- Diagonal SSMs are *provably* unable to recognize A_5 words in O(1) depth (Merrill et al. §3); the K-torus state-transition group is solvable, and A_5 is the smallest non-solvable group.
- β=1 DeltaNet is the strongest published non-diagonal linear-recurrence baseline. Its rank-2 update is a Givens rotation of angle π in an input-dependent plane (GRP-RNN's K=1 special case).
- GRP-RNN with K ≥ 3 interlocking planes has a non-abelian H_K ⊂ SO(m) that contains A_5 ⊂ SO(3); it should, in principle, recognize A_5 words.

### Task

**A_5 word recognition (Daley & Merrill, simplified):**
- Generators g_0 = (1 2 3), g_1 = (3 4 5). Both are 3-cycles in S_5; their products generate A_5.
- Sample T random tokens, each ∈ {g_0, g_1} uniformly (V = 2).
- Compute the group product π = g_{w_T} ◦ ... ◦ g_{w_1} ∈ A_5.
- Binary label: y = 1 if π = identity element of A_5, else y = 0.
- Probability of identity in uniform-random T-token word is 1/|A_5| = 1/60. Balance training/eval by rejection-sampling 50/50 identity/non-identity.

**Sequence length sweep:** T ∈ {64, 256, 1024}.

**Hyperparameters:**
- m = 256 (state dim).
- Single layer of either GRP-RNN or DeltaNet.
- Embedding 2 → m, output 1-token classification head m → 2 (identity vs non-identity).
- AdamW, lr = 3e-4, cosine decay, 200-step warmup.
- Batch 64. Training: 20,000 steps. Wall-clock budget cap: 8 GPU-hours per cell.
- 10 seeds per cell.

### Strongest adversarial baseline

**β=1 DeltaNet** as implemented inside GRP-RNN via `--grp-K=1 --grp-plane-input-dep --grp-no-decay`. Both models use the *same* encoder, *same* optimizer, *same* training schedule, *same* readout. Only the recurrence parameterization differs. This is the cleanest possible isolation.

Secondary baselines (reported but not used for the headline claim):
- **LRU** as `--grp-disjoint-planes --grp-fixed-angles`. Expected: near chance, because LRU is diagonal-complex.
- **Tanh-RNN** (`--grp-tanh-state`). Expected: chance.
- **GRP-RNN @ K=8, 16, 64, 128**. Reported as the K-scaling sweep.

### Param-matching

All models report total trainable params after init. We match within ±5%. Specifically:
- DeltaNet (K=1 with plane learned): ~ K=1 angles + plane head (W_p ∈ ℝ^{m × d}) = m·d + 1 params.
- GRP-RNN (K = m/2 = 128): K angles + (W_a ∈ ℝ^{K × d}) = K·d + K params.

DeltaNet has fewer per-step angle params but a richer plane head; GRP-RNN has more per-step angle params with fixed planes. Net difference is small. We balance by widening the readout head on whichever side has fewer total params.

### Pre-commit interpretation table

| Observation (mean over 10 seeds, 95% CI) | Interpretation |
|---|---|
| GRP-RNN > DeltaNet by ≥ 0.05 acc at T ≥ 256 AND GRP-RNN > LRU by ≥ 0.20 at T ≥ 256 | **N1 SUPPORTED.** K>1 rotation lift is empirically load-bearing; theoretical expressivity gap (Merrill et al.) materializes under SGD. |
| GRP-RNN > DeltaNet by ≥ 0.10 acc at every T | **N1 STRONGLY SUPPORTED.** Bonus result; framework's expressivity claim is robust across T. |
| GRP-RNN ≤ DeltaNet + 0.02 acc at all T | **N1 FALSIFIED.** K>1 rotation does not extend DeltaNet's expressivity in practice; GRP-RNN reduces to "DeltaNet with extra parameters that don't help". Negative result; report and pivot. |
| GRP-RNN ≈ LRU + 0.02 at all T (both fail) | **N1 INCONCLUSIVE — task does not discriminate.** A_5 word recognition not solvable at this scale by either model; rerun with longer training or m=512 before concluding. |
| GRP-RNN > LRU by ≥ 0.20 AND GRP-RNN > DeltaNet by ≥ 0.05 *at T=64 only* (regresses at T=256, 1024) | **N1 INCONCLUSIVE, REGIME-LIMITED.** Report as "novel at short T only"; apply Phase-0k lesson; do not generalize. |
| GRP-RNN matches DeltaNet at T=64 but *degrades faster* than DeltaNet at T=1024 | **N1 FALSIFIED + new failure mode.** K-Givens product is less stable than rank-2 DeltaNet under long-context optimization. Report and study optimization fragility. |
| Variance in GRP-RNN across seeds > 2× DeltaNet variance | **N1 INCONCLUSIVE + STABILITY FLAG.** GRP-RNN trains less stably; rerun with stronger orthogonal-init prescription on the angle parameters before final conclusion. |

### Confounds (must check before publishing)

1. **DeltaNet implementation incorrect.** Reproduce DeltaNet on the published synthetic-recall task in Yang & Schlag 2024 §5.1 first as a sanity check; expected accuracy ≥ 0.85.
2. **Saturating angles.** Check φ_max usage on a held-out batch; if |tanh(w_k^T x_t + b_k)| > 0.9 on average, angle saturation is hurting gradient flow. Activate L_rot-reg if so.
3. **Param-count not actually matched.** Re-tune readout-head width or K to match within ±5%. Report final param counts.
4. **A_5 words too short / too long.** Sanity: DeltaNet at T=64 must reach ≥ 0.7 accuracy; if not, the readout head is the bottleneck, not the recurrence.
5. **LR-window narrowness (Zoology).** Run each cell at lr ∈ {1e-4, 3e-4, 1e-3}; report best. If best is at the boundary, expand the LR sweep.
6. **Plane-graph degeneracy.** Verify the interlocking-stride-3 plane graph has chromatic number ≥ 3 (verified by construction for m ≥ 6).

### When this should fail

- **F-N1-a.** K>1 advantage does not materialize empirically (Framework §10.2 F2). Diagnostic: per-plane gradient norm distribution at end of training; if only ~3 of K planes have significant gradient mass, the model is using K effectively as K=3. Mitigation: reduce K to 16; rerun.
- **F-N1-b.** A_5 generators not appropriately mixed in the input distribution. Diagnostic: empirical fraction of identity words in train/eval. Mitigation: rebalance.
- **F-N1-c.** Readout head bottleneck. Diagnostic: vary readout depth at fixed recurrence; if accuracy is invariant, the bottleneck is the recurrence (good).

---

## Claim N2 — Throughput within 3× of Mamba-2

### Statement

At T=2048, m=1024, batch=8 on a single RTX 4080 SUPER (or comparable Ada / Hopper GPU), GRP-RNN's training tokens/sec is ≥ 0.33× Mamba-2's training tokens/sec, AND inference tokens/sec is ≥ 0.33× Mamba-2's inference tokens/sec.

### Why this is the right threshold

The brief allows the framework to be slower than the strongest baseline so long as the gap is bounded and explicit. A 3× slowdown is small enough to be acceptable in exchange for the expressivity gain (Claim N1) and large enough to be achievable without exotic kernel engineering. If the gap exceeds 3× the framework's value is conditional on the task: expressivity-bound regimes (state-tracking, algorithmic reasoning) can absorb it; throughput-bound regimes (production LM) cannot.

### Baselines

- **Mamba-2 reference** from `state-spaces/mamba` (Triton selective_scan_cuda). Run on the same hardware via PyTorch + Triton. Not available in glades-ml; we will time it on a separate driver.
- **Secondary: LRU** at m=1024 (fastest single-particle linear-recurrence baseline).
- **Secondary: pure C++ baseline of the existing `--model=rnn`** at m=1024 with `--use-tanh=0` (already measured at ~53,000 tok/s in `research/ealrmn_gpu/results/sweep_prod_v1.jsonl`).

### Metric

`tokens_per_second` measured by `T * batch / wall_clock_seconds_per_step` over 100 warm-iter steps. Both training-forward (FWD + BWD + AdamW) and inference-recurrent (FWD only, batch=1, autoregressive) are measured.

### Pre-commit interpretation table

| Observation | Interpretation |
|---|---|
| GRP-RNN tok/s ≥ 0.5 × Mamba-2 (training and inference) | **N2 STRONGLY SUPPORTED.** Throughput is competitive; framework can be considered for downstream LM scaling. |
| 0.33 × Mamba-2 ≤ GRP-RNN tok/s < 0.5 × Mamba-2 | **N2 SUPPORTED.** Publishable as "expressivity-at-throughput-cost"; explicit qualification. |
| 0.1 × Mamba-2 ≤ GRP-RNN tok/s < 0.33 × Mamba-2 | **N2 FALSIFIED.** Framework is too slow at the current kernel implementation; flag the kernel-engineering work as future work. Claim N1 may still stand but the framework is unfit for LM scaling. |
| GRP-RNN tok/s < 0.1 × Mamba-2 | **N2 STRONGLY FALSIFIED.** Kernel design is wrong; do not publish until reimplemented. |
| Training fast, inference slow (or vice versa) | **N2 PARTIAL.** Specify regime; e.g., chunkwise-scan is the bottleneck for training but recurrent inference is competitive. |
| GRP-RNN tok/s > Mamba-2 (i.e., we are faster) | **Surprise outcome.** Recheck implementation; if confirmed, this is a *additional* publishable contribution. Document the kernel optimization that enabled it. |

### Confounds

1. **Triton vs CUDA kernel-stack overhead.** Mamba-2's selective_scan_cuda is Triton. GRP-RNN is custom CUDA. Direct cuBLAS calls are faster than Triton for fixed-size GEMMs; Triton may be faster for the selective-scan-specific kernel. Report both kernel-time-only and end-to-end wall-clock.
2. **bf16 vs fp32 numeric differences.** Both at fp32 first to isolate; bf16 as a secondary measurement.
3. **Batch padding inefficiency.** Power-of-2 T; report at T=2048 only for the headline.
4. **Kernel-launch overhead.** Measure with Nsight Compute; report cleanly.
5. **Different cuBLAS auto-tuning state.** Run with `CUBLAS_WORKSPACE_CONFIG=:4096:8` for reproducibility.

### When this should fail

- **F-N2-a.** Chunkwise scan launch-overhead dominates (Framework §10.3 F3). Diagnostic: kernel-launch overhead > 40% of wall-time in Nsight. Mitigation: fuse Givens-product with input GEMV; or use CUDA graphs.
- **F-N2-b.** Givens product is memory-bound and cannot saturate VRAM bandwidth at K = m/2. Diagnostic: measured arithmetic intensity vs DRAM bandwidth. Mitigation: reduce K (with N1 sweep showing K=16 is acceptable); or batch-fuse adjacent Givens.
- **F-N2-c.** Mamba-2 reference numbers are taken from its blog at a different batch size or hardware than our test. Mitigation: re-run Mamba-2 reference on the same RTX 4080 SUPER with the same batch/T.

---

## Claim N3 — B0 replicated under GRP-RNN flag ablation (internal infrastructure check)

### Statement

GRP-RNN with `--grp-tanh-state` ON should produce val_loss ≥ 100× *(or, equivalently, the ratio observed in `B0_RESULTS.md`)* than GRP-RNN with `--grp-tanh-state` OFF on the needle task at T=2048, m=1024, 5 seeds.

### Status

Pending — requires GRP-RNN GPU prototype. Will run before any other GRP-RNN-specific claim.

### Why this matters

- N3 verifies that the GRP-RNN implementation is correctly orthogonal-init-by-construction: with `--grp-tanh-state` OFF and angles near zero at init, R_t ≈ I, the model should match (or beat) the existing `--model=rnn --use-tanh=0` baseline on the same task.
- N3 verifies that the `--grp-tanh-state` ablation actually destroys performance, as expected by EALRMN Phase-1. If it does *not*, the tanh-after-rotation hook is wired incorrectly.

### Pre-commit interpretation table

| Observation | Interpretation |
|---|---|
| `--grp-tanh-state=off` val_loss matches `--model=rnn --use-tanh=0` ratio at the same (m, seeds, steps) within 1.5× | **N3 SUPPORTED.** GRP-RNN's orthogonal-init-by-construction is working. |
| `--grp-tanh-state=off` ≥ 10× better than `--grp-tanh-state=on` | **N3 SUPPORTED via the linear-vs-tanh ratio.** |
| `--grp-tanh-state=off` worse than `--model=rnn --use-tanh=0` by ≥ 2× | GRP-RNN implementation buggy or initialization sub-optimal. Debug before N1. |
| Either variant diverges to NaN on any seed | Implementation buggy. Debug before N1. |
| Both variants tie on the needle task | Task too easy for both; switch to T=4096+. |

### Confounds

1. **Wiring bug in `--grp-tanh-state`.** Verify the tanh is applied *after* the linear update (post-R_t multiplication, post-input addition).
2. **Different gradient-clip in different flags.** Use identical clip across cells.
3. **Different LR for different flags.** Use identical LR across cells.

---

## Decision priority and order of testing

Per the brief's "Implementation Guidance" section, claims are tested in order of expected information value about the NOVEL parts of the framework, with B0-class infrastructure checks first.

1. **B0 (DONE).** `research/vesta/run_b0.sh` returned 20× at iso-param-iso-m. Phase-1-equivalent rerun in flight at `research/vesta/run_b0_phase1_repro.sh`; expected ratio ~4,000×.
2. **N3.** Cheap (~30 min once prototype builds). Verifies infrastructure is fine after GRP-RNN extension.
3. **N1.** The decisive claim. ~8 GPU-hours. If N1 supported, the framework has earned its novelty.
4. **N2.** Engineering-feasibility check. ~1 hour after kernels are in place.

If any of N3/N1/N2 fails its threshold, the framework's status updates as follows:

| Failed claim | Status of framework |
|---|---|
| N3 fails | Implementation bug; debug, do not proceed to N1. |
| N1 fails (GRP-RNN ≈ DeltaNet) | Framework reduces to DeltaNet; report negative result; halt. No N2. |
| N1 supported, N2 fails (slow but expressive) | Publish with throughput caveat; recommend for state-tracking tasks only. |
| N1 + N2 both supported | Earn novelty fully; recommend Phase-4 LM-transfer experiment. |

---

## Honesty pre-commitments (per `newmodel.txt` E5)

1. **If N1 fails:** GRP-RNN reduces to DeltaNet with extra parameters. We will report this as a clean negative result. We will *not* repackage the framework as "still useful on X" without a separately pre-registered claim on X.
2. **If N2 fails badly:** GRP-RNN is too slow for LM scaling. We will report this and bound the recommended regime to state-tracking / algorithmic reasoning tasks. We will *not* claim production-readiness.
3. **If F-N1-a (K>1 doesn't materialize):** We will publish the per-plane gradient distribution and accept that the Merrill et al. 2024 theoretical bound is loose in practice. This is informative for the SSM literature.
4. **If the framework needs a regularizer (L_rot-reg or L_decay-reg) to make N1 work:** We will report this; the regularizer goes from "ablation-only" to "necessary." The 1-mechanism framework becomes a 2-mechanism framework, and we will be explicit about the change.

---

## Summary table

| Claim | Type | Strongest baseline | Threshold | Status |
|---|---|---|---|---|
| B0 | infrastructure | tanh-RNN @ orthogonal init | ≥ 100× val_loss ratio (Phase-1 spec) | **VALIDATED qualitatively at 20× iso-param-iso-m; Phase-1-repro pending** |
| N1 | novel-axis (R3) | β=1 DeltaNet on A_5 | ≥ 0.05 acc gain at T ≥ 256 | **PENDING (after GRP-RNN GPU prototype)** |
| N2 | engineering-feasibility | Mamba-2 SSD | ≥ 0.33× tok/s | **PENDING (after GRP-RNN GPU prototype)** |
| N3 | internal infrastructure | GRP-RNN itself | match `--model=rnn --use-tanh=0` ratio | **PENDING (after GRP-RNN GPU prototype)** |

— end VESTA_CLAIMS.md —
