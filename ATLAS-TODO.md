# ATLAS Optimizer — Future Work

## Recently Completed

- [x] GPU Fisher diagonal collapse fix
  - The CUDA Fisher reduction bug that pinned Fisher to its epsilon floor was fixed.
  - A focused CUDA regression now checks that `atlas_gpu_fisher_update()` returns the expected mean-squared value on a simple all-ones input.

- [x] CUDA stream-ordering / blocking-buffer fix
  - Host-side reads of `sigma2`, `mu`, diagnostics, and refresh safety checks now synchronize the compute stream before consuming reduction results.
  - Blocking `GpuBuffer::{upload,download,zero}` operations once again provide ordered behavior after the compute/transfer stream split.

- [x] GPU consistency coverage
  - `glades-unit-tests atlas` now passes again.
  - GPU-CPU equivalence and GPU-to-CPU state transfer tests both return to near-exact agreement after the ordering fixes.

These fixes invalidate any adaptive-rank conclusions drawn from pre-fix logs. Use only post-fix
`atlas_gpu_step` diagnostics for rank tuning or subspace-analysis work.

## Immediate Next Step

Run at least one full post-fix training epoch with `atlas_gpu_step` diagnostics enabled and archive
per-weight summaries for:

- `baseline_rate`
- `fisher_ratio`
- `sigma2_fisher_ratio`
- `effective_rank`
- `spectral_efficiency`
- `top1_concentration`

That data is the gating input for the adaptive-rank items below. Do not tune rank from the old
Fisher-collapsed runs.

### How to Analyze Post-Fix Diagnostics

Use epoch-level summaries, not one-off diagnostic lines. For each weight tag, aggregate averages
after the first 2 refreshes so initialization transients do not dominate the signal.

Recommended interpretation:

- **Pathology, do not tune rank yet**
  - `fisher_mean` remains near the epsilon floor
  - `sigma2_fisher_ratio` is still enormous (`> 1e6`)
  - `fisher_ratio ~= 1`, `spectral_efficiency ~= 1`, and `top1_concentration ~= 1/r`
  - This means the curvature signal is still effectively missing or flat; fix the optimizer path first.

- **Overprovisioned rank, candidate for reduction**
  - `spectral_efficiency < 0.10` for most of the epoch
  - `effective_rank_avg * 1.5 < current_rank`
  - Stronger signal if `top1_concentration > 0.20` or `top10_concentration > 0.80`
  - Action: reduce toward `ceil(effective_rank_avg * 1.5)`, clamped to `[16, subDim/4]`

- **Healthy rank, leave it alone**
  - `0.10 <= spectral_efficiency <= 0.80`
  - `fisher_ratio > 3`
  - `sigma2_fisher_ratio` is elevated but not pathological
  - Action: keep current rank and collect more runs before changing anything

- **Rank-starved, candidate for increase**
  - `spectral_efficiency > 0.80` for most of the epoch
  - `fisher_ratio > 10` (there is real curvature contrast left to capture)
  - `top10_concentration < 0.50` or the Fisher mass is still spread broadly across the active rank
  - Action: increase rank, but only up to `subDim/4`

- **Uniform but not obviously broken**
  - `spectral_efficiency > 0.80` but `fisher_ratio <= 2`
  - This usually means the active subspace is close to isotropic, not necessarily under-ranked
  - Action: do not increase rank automatically; investigate whether the layer simply lacks strong curvature anisotropy

Also flag outliers in `baseline_rate`. A tensor whose baseline rate is orders of magnitude below
peer tensors may still deserve separate investigation even if the rank metrics look nominal.

## Adaptive Rank (Phase 2)

The spectral efficiency diagnostics (effective_rank, spectral_efficiency, top1/top10_concentration)
are now logged at each diagnostic step. The following features require observation data from
these diagnostics before implementation.

### Per-Weight Rank Configuration

**Problem**: All weight matrices currently share the same `--atlas-rank` value (256).
Different layers have different intrinsic gradient dimensionality.

**Implementation**:
- Add `std::unordered_map<std::string, unsigned int> perWeightRank` to `ATLASConfig`
- CLI: `--atlas-rank-override tr.tokE=128,tr.W1=64` (comma-separated tag=rank pairs)
- In `atlas_gpu_update`: if tag has an override, pass it instead of `ac.rank`
- Fallback to `ac.rank` for unspecified tags

**When to implement**: After one full epoch with post-fix spectral diagnostics.
Look for layers where `spectral_efficiency < 0.1` consistently (candidates for rank reduction)
and layers where `spectral_efficiency > 0.8` (candidates for rank increase).

### Auto-Tuning Rank Between Epochs

**Problem**: Manually setting per-weight ranks from log analysis is tedious.

**Implementation**:
- At the end of each epoch, compute average spectral_efficiency per weight tag
- Write a `rank_profile.json` to the checkpoint directory:
  ```json
  {
    "tr.tokE": {"rank": 256, "effective_rank_avg": 47.3, "suggested_rank": 64},
    "tr.Wq": {"rank": 256, "effective_rank_avg": 210.5, "suggested_rank": 256}
  }
  ```
- On next run, if `--atlas-rank-auto` is set, load the profile and apply suggested ranks
- Suggested rank formula: `next_r = clamp(ceil(effective_rank_avg * 1.5), r_min, r_max)`
  - 1.5x headroom ensures we capture slightly more than the estimated effective dimensionality
  - `r_min = 16` (below this, preconditioning is barely useful)
  - `r_max = subDim / 4` (dimension-proportional cap)

**When to implement**: After validating that spectral_efficiency is a reliable predictor
of optimal rank across multiple training runs.

### Fisher-Weighted Power Iteration

**Problem**: Standard power iteration finds the top singular vectors by gradient variance.
But ATLAS cares about *curvature* (Fisher), not just variance. Directions with high variance
but low curvature are less useful for preconditioning than directions with moderate variance
but high curvature.

**Implementation**:
- During refresh, after computing gz in the current subspace, weight the power iteration
  seed by the Fisher diagonal: `Q_seed[:,c] *= sqrt(f_c)` before the first iteration
- This biases the subspace toward directions that have both high gradient variance AND
  high curvature contrast
- Requires Fisher to contain real signal (skip until after the first nontrivial Fisher update)

**When to implement**: After confirming that the subspace captures the *right* directions on
post-fix runs.
If `fisher_ratio` is high (>100) but `spectral_efficiency` is low (<0.1), the subspace
is already finding diverse curvature — Fisher-weighting would help. If both are low,
the subspace itself needs work first.

## Optimizer Improvements

### Replace Fisher Diagonal with Per-Element Adam Moments in Subspace

**Problem**: ATLAS tracks 1 scalar (Fisher diagonal) per subspace direction.
SOAP tracks m*n Adam moments in the rotated space. Per-element moments capture
finer-grained adaptation within each direction.

**Implementation**:
- Add `GpuBuffer<float> gz_m1` (first moment EMA, size outerDim * r) to state
- Add `GpuBuffer<float> gz_v2` (second moment EMA, size outerDim * r) to state
- Replace Fisher diagonal update with per-element Adam-style update:
  ```
  gz_m1 = beta1 * gz_m1 + (1-beta1) * gz
  gz_v2 = beta2 * gz_v2 + (1-beta2) * gz^2
  gPred = gz_m1 / (sqrt(gz_v2) + eps)  (Adam update in subspace)
  ```
- The correction becomes: `W += U * (corrScale * gPred)` where corrScale is based
  on the *mean* of gz_v2 per direction (replacing Fisher diagonal)

**Memory cost**: +2 * outerDim * r floats per weight. For tokE (32000x1024, r=256):
+2 * 32000 * 256 * 4 = 64MB. Significant but feasible on 16GB GPU with fewer layers.

**When to implement**: After establishing a head-to-head benchmark against SOAP/Adam
to determine if the Fisher diagonal is the bottleneck for step efficiency.

### Head-to-Head Benchmark vs Muon and SOAP

**Setup**:
- GPT-2 124M on OpenWebText (standard benchmark)
- Compare: ATLAS, AdamW, Muon, SOAP (if implementable)
- Metrics: validation loss at {1B, 5B, 10B} tokens, wall-clock time, peak GPU memory
- Hardware: RTX 4080 Super 16GB (our current setup)

**When to implement**: After the current training run completes and we have
stable ATLAS results to compare against.

## Infrastructure

### Cosine Schedule Bug

- [x] The step-based cosine restart bug was fixed.
- [x] Add a small regression test that checks `lr_mult` decays smoothly within an epoch and does not restart unexpectedly.

The step-based cosine schedule fix uses `s / seqCount` for progress within the epoch.
The regression now covers fractional intra-epoch decay, step clamping, and the `totalStepsInEpoch == 0` fallback.
Monitor `lr_mult` in logs to confirm no more restarts.

### Checkpoint ATLAS Config + Resume Validation

**Status**: Partially done.

Already in place:
- ATLAS checkpoint persistence writes and reads `U`, `fisherDiag`, `prevGz`, `mu`, `sigma2`, and `step`
- `ATLAS Test 24` verifies resume equivalence for a checkpointed ATLAS training run
- `ATLAS Test 24B` verifies ATLAS resume equivalence for a checkpointed transformer encoder run and checks restored transformer overrides

Remaining work:
- [x] Persist ATLAS config in the checkpoint manifest so callers do not have to manually restore fields like `rank` and `tSub` after `loadCheckpoint()`
- [x] Fail fast on rank/config mismatches instead of reconstructing ATLAS state from the current runtime config
- [x] Add a transformer-focused resume test so checkpoint coverage is not limited to the DFF path
- Decide whether GPU-resident resume needs dedicated coverage beyond the current checkpoint load path
