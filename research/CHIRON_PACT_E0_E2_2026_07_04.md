# CHIRON PACT — E0/E1/E2 Result: PASS + a field-scaling refinement (2026-07-04)

**Status:** E0 **PASS** (flag-off bit-parity), E1 **PASS** (chiron-pact unit suite, all
invariants), E2 **PASS** (λ calibrated, mechanism stable). One design refinement surfaced and
applied: the field is **increment-energy scaled**, not scale-free (see §3). Next gate: **E3**
(matched 2500-step fresh pair, ~7 GPU-hr).
**Ladder:** design `docs/superpowers/specs/2026-07-04-chiron-pact-anti-cancellation-design.md`;
M0 `research/CHIRON_PACT_M0_2026_07_04.md`; plan `docs/superpowers/plans/2026-07-04-chiron-pact-e0-e2.md`.

## 1. Implementation summary

Library (glades-ml, commit `766428d71` + this session's refinement):
- CPU refs (`transformer_chiron_ops.h`): damping table, sign-coherence gate χ, Huber, the
  energy-scaled field, and profile helpers.
- CUDA kernels (`gpu_chiron.cu`): `chiron_pact_cos_row` + `chiron_pact_damp_finalize` (suffix
  products), `chiron_pact_sigma_update` (detached scale EMA from the mass accumulator),
  `chiron_scfa_axpy2_masked_dual_p_pact` (fused commit + A/M accumulation, p/p_bf16 bit-identical
  to the non-PACT kernel), `chiron_incdrop_scale_copy_dual_pact` (fused dy field injection +
  monitor stats).
- Unit suite `test.sh chiron-pact` (4 tests): gate goldens, exact orthogonality `Σ_l D_l g_l = 0`
  (5.7e-7), FD gradient correspondence (4.2e-5), GPU==CPU field bit-exact, Huber clamp count
  exact, degree(+1) scale covariance, coef=0 numeric identity. chiron-pied / chiron-whisc
  regressions green.

Trainer (glades-trainer, `28c1d7b` + refinement): `--pact-coef` / `--pact-gate` / `--pact-clamp`
(default off); interlock requires the PIED fused-commit + dual-dy flagship path; forward builds
D from rot_phi, zeros A/M, routes the masked commit to the PACT variant, updates the detached
σ EMA (first-touch on the first forward); backward fuses the field into the dual-dy hand-off
(`dy = η·dp + g` on the clean increment, `scfa_ypar` dst==a read-before-write alias); `[pact]`
monitor (value, field/dy RMS, clamp rate) at the `[sorc]` cadence; run.sh flag routing.

## 2. E0 / E1 results

- **E0 (flag-off bit-parity):** the eval protocol on `chiron_1B_pied_e4.final` with no PACT flag
  reproduces the M0 baseline val NLL **exactly — 1.0891 / 1.1174 / 1.1909** (8-batch windows).
  The forward path is byte-identical with the flag off. PACT training-path code is all gated on
  `cfg.pactCoef != 0.0f`. *PASS.*
- **E1 (units):** `chiron-pact` all green (0 asserts failed). Key invariants verified on GPU and
  CPU: exact orthogonality (field ⟂ damping direction, unclamped), FD gradient = analytic field,
  GPU field bit-exact vs CPU ref, Huber clamp count exact, gate = 0 on sign-coherent/one-hot
  profiles, dead-zone release, coef=0 numeric identity to the non-PACT dy kernel. *PASS.*

## 3. E2 — the field-scaling refinement (the load-bearing finding)

**The design's field was scale-free, and a scale-free penalty has a scale-DIVERGENT gradient.**
The original penalty `ϕ = χ·‖P_⊥Y‖²/(Dsq·σ²)` is O(1) (dimensionless, "scale-free" per F3), so
its gradient scales as **1/σ** and blows up as increments → 0. Measured in vivo at fresh init:
`field/dy = 5.4` (the field was **540%** of the task gradient — it would dominate optimization).

**Fix (applied):** drop the `1/σ²` from the weight (`ŵ = 1/Dsq`), making the penalty
increment-**energy** scaled, `ϕ ~ ‖P_⊥Y‖²`. The field becomes
`g = coef·χ·σ·clamp(res/σ, ±κ)/Dsq ~ O(res) ~ O(‖increment‖)` — bounded, vanishing as
increments → 0 (safe at init), and still ρ-free (σ is the increment scale, not a p/q ratio).
σ is retained purely as the **Huber knee** (defining "large residual" relative to the increment
scale). All E1 invariants re-verified: orthogonality `Σ_l D_l g_l = 0` preserved (the field is
still along the null-component direction), FD correspondence holds, and the field is now
degree(+1) homogeneous in the increment scale (it scales *with* the increments, like the task
gradient — the correct behavior). The gate, the anti-cancellation content, and the exact
function-preservation are unchanged.

Also corrected in E2: the trainer's penalty normalization from `2λ/(Tm)` to **`2λ/T`** (token-mean,
matching NLL's `1/T`), so λ is O(1e-3–1e-2) rather than requiring an absurd ~1e5. The unit tests
pass `coef` directly, so the kernel/CPU-ref math is unaffected by either change.

Both changes are documented in the code (`transformer_chiron_ops.h` field comment,
`gpu_chiron.cu` field line, the trainer coef site) as the **E2 refinement of record**. The design
spec's §5.2–5.3 formulas are superseded on this one point (energy scaling vs scale-free); the
mechanism — gated anti-cancellation, orthogonal function-preserving field, T1 diagonal-blindness
motivation — is intact.

## 4. E2 — calibration and stability

λ = 3e-3, energy-scaled field. `field/dy` = RMS(field) / RMS(task-dy), monitored per log window.

**Fresh-init run** (random init, flagship recipe + `--pact-coef 3e-3`, seed 1337 — the regime
E3/E4 use):

| step | value | field/dy | clampRate |
|---:|---:|---:|---:|
| 1  | 0.0024 | 1.10% | 0.16% |
| 11 | 0.036  | 2.45% | 2.97% |
| 21 | 0.023  | 2.18% | 1.69% |

field/dy ramps into ~2–2.5% early (loss still high in warmup) and is expected to reach the
design's 3–5% target as loss descends mid-training. **0 grad-skips, 0 NaN.**

**Resume-from-converged run** (documented as the worst case / non-calibration regime): field/dy
climbs 4e-4 → 7.6e-2 across 30 steps because the task gradient collapses as the optimizer
overfits the already-converged flagship's tiny residual loss. This regime is pathological for a
*ratio* target, but the mechanism stayed safe throughout: **0 grad-skips**, clampRate ≤ 2.5%,
the Huber clamp bounding the per-element field regardless of how small task-dy became. This is
direct evidence that the κ=4 clamp is the effective safety bound (the OBSD lesson).

**λ decision:** adopt **λ* = 3e-3** for E3 (the design's lower pre-registered value; puts
field/dy in the 3–5% band mid-training, conservative). Fallback **λ = 1e-3** if E3 shows any
instability; **λ = 1e-2** is ruled out by these measurements (would land field/dy ~7–8% early,
above the target). Since the field scales linearly with λ, the ratio retargets analytically.

**Two documented deviations from the design §5.1/§5.4** (recorded per the plan):
1. σ̂ is the EMA (rate = `whiscEma` = 0.05) of `mean_t(M_{t,i})/D1_i` — a mean-|damped-increment|
   scale read from the already-materialized M buffer (avoids per-element atomics; detached,
   scale-free, F3-clean). First-touch (β=1) on the first PACT forward before any backward reads σ.
2. A, M accumulators are FP32 (2×128 MiB at flagship shape, within the ~0.86 GB headroom), not
   BF16 — for exactness; BF16 remains the perf fallback if VRAM ever binds.

## 5. Perf / VRAM (measured — two perf passes, 2026-07-04)

VRAM peak ~15.66 GB at flagship shape with PACT on (A/M add 128 MiB) — within the practical
ceiling.

**Wall was A/B'd at flagship shape (T=16384, clean back-to-back same-binary runs), and the first
measurement exposed a severe regression that two perf passes fixed:**

| state | tok/s | vs no-PACT | note |
|---|---:|---:|---|
| no-PACT baseline | 25,231 | — | same binary |
| PACT, initial | 14,900 | **−40%** | shipped-then-caught |
| + warp-shuffle reduction (pass 1) | 22,538 | −10.7% | glades-ml `523a82745` |
| + gate stats to log cadence (pass 2) | **23,510** | **−6.8%** | glades-trainer `2f4418c` |

- **Root cause (nsys):** the field kernel `chiron_incdrop_scale_copy_dual_pact` took **17.8 ms/call
  (39.3% of GPU time)** vs the commit kernel's 1.87 ms for the same [T×m] shape. The `stats4`
  monitor reduction had all 256 threads/block `atomicAdd` doubles to 4 **shared** addresses;
  shared double-atomics are CAS-emulated, so same-address contention serialized the kernel
  ~quadratically.
- **Pass 1** — warp-shuffle the reduction (in-register per warp → 1 global atomic/block). Field
  output bit-identical; kernel 17.8 → ~1.35 ms. **−40% → −10.7%.**
- **Pass 2** — the monitor is read only every `logEvery` steps, so gate `stats4` to log steps
  (NULL otherwise; the kernel skips the stats block on NULL). Field bit-identical; `field/dy` at
  step 1 reproduced exactly (1.1032e-02). Monitor normalization moved to per-log-step
  (`perUstep = accumSteps`). **−10.7% → −6.8%.**
- **Residual −6.8%** is inherent **FP32 A/M [T×m] accumulation traffic**: the PACT commit RMWs A
  and M (1.87 ms vs the non-PACT commit's 0.96 ms) and the field reads them. BF16 A/M would ~halve
  it (toward ~−3.5%) but **changes the field numerics** (§4 chose FP32 for exactness), so it is a
  mechanism change requiring E1 re-validation + an E-gate — deferred, not a bit-identical perf pass.
- For context: WhiSC-D shipped at −9.7% wall, PIED at −1.87%. PACT at −6.8% (training-only;
  inference unchanged) is between them.
- **These passes do not touch the field math** (field output bit-identical), so any efficacy data
  measured on the pre-perf binary (the interrupted E3 arms, `logs/pact_e3_*.log`) remains valid —
  the val gap is field-driven and the field is unchanged.
- Harness: `scripts/pact_perf_probe.sh` (resolved-flag probe) in glades-trainer.

## 6. Disposition

- **E0/E1/E2 PASS.** PACT is implemented, unit-verified, stable in vivo, and calibrated
  (λ*=3e-3). Default-off, no checkpoint delta, no serving change; E0 bit-parity confirmed.
- **Next gate: E3** — matched 2500-step **fresh same-binary pair** (flagship recipe ±
  `--pact-coef 3e-3`, seed 1337, ~7 GPU-hr). Bars (design §13): 0 grad-skips; ‖g‖ within the
  baseline envelope; penalty value falling ≥30% from step-100; val@2500 ≤ baseline + 0.01 and
  the gap shrinking; clampRate < ~3%; `[whisc]` nominal; wall ≤ +2%. Monitor field/dy stays in a
  safe band (< ~0.1). Kill → drop to λ=1e-3 or close.
- **E4** (matched 30k **paired** arms, ~44 GPU-hr) remains an owner spend decision.

## 7. Artifacts

- glades-ml: `766428d71` (E1 kernels/refs/suite) + this session's field-scaling refinement (to
  be committed with this record).
- glades-trainer: `28c1d7b` (plumbing/monitor) + normalization fix + run.sh routing (to be
  committed).
- Logs: `logs/pact_e0_*.log` (E0), `logs/pact_smoke_on_*.log` (machinery), `logs/pact_cal_*.log`
  (converged worst-case), `logs/pact_fresh2_*.log` (fresh calibration).
- Scratch checkpoints cleaned (flagship `chiron_1B_pied_e4.final` mtime unchanged, verified).
