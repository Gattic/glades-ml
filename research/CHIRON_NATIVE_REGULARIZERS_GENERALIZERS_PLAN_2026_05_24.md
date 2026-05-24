# CHIRON-Native Regularizers and Generalizers — Research Plan (2026-05-24)

**Status:** research plan / pre-registration seed. No production behavior change.
**Baseline:** CHIRON 1B regstack Phase 2, checkpoint
`chiron_1B_T16384_regstack_phase2.final`.
**Recipe:** `cd ~/dev/glades-trainer && sh run.sh flagship --zloss-coef 1e-4 --qk-norm`.
**Baseline metrics:** val NLL **3.5734** @ 30k, ~**28,072 tok/s** @ T=16384,
~14.97 GB VRAM.

## Implementation status as of 2026-05-24

This document is a research plan plus pre-registration seed. It should not be
read as claiming an active production-loss integration for SIRA.

Implemented and verified so far:

- `glades-ml` has default-off SIRA configuration fields in
  `TrainingConfig`, with config validation and CPU reference helpers for the
  proposed phase/action loss terms.
- `glades-ml` unit tests cover SIRA defaults, disabled parity, enabled math,
  and edge cases for energy-only and action-weight paths.
- `glades-ml` has targeted unit-test selectors `chiron-sira` / `sira`.
- `glades-trainer` accepts SIRA CLI/config flags and has `--sira-config-smoke`
  paths proving flag propagation into `glades::TrainingConfig` without
  starting training.
- Verified commands included `./build/glades-unit-tests chiron-sira`,
  trainer `scripts/sira_config_smoke.sh`, direct `glades_pile_train` and
  `glades_chiron_train` config-smoke invocations, and the aggregate
  `./build/glades-unit-tests chiron` after independent CHIRON test
  stabilization.

Not implemented yet / still speculative:

- SIRA is not added to the production training objective; `siraCoef > 0` does
  not yet affect CE training loss in the trainer.
- No CUDA production kernels, analytic gradient injection, persistent phase
  diagnostics, probe-layer schedule, or position-bucket instrumentation exist
  yet for SIRA.
- `--sira-probe-layers`, `--sira-position-buckets`, PHS, and PTOC remain
  research proposals in this document, not runnable features.
- No 1B SIRA pilot has been run, and there is no NLL/throughput result to
  compare against `chiron_1B_T16384_regstack_phase2.final`.

Relevant code commits:

- `glades-ml`: `48aec4e1e Add default-off SIRA config and tests`
- `glades-trainer`: `2b6c791 Add SIRA CLI smoke plumbing`
- `glades-ml`: `5545a1ba8 Stabilize CHIRON tests` (test-only stabilization,
  not SIRA behavior)

This document starts the next research direction after the Phase-3
regularization sub-program closed. The prior failure pattern is treated as a
constraint, not an inconvenience: naive residual-transformer ports are out of
scope. The mechanisms below are native to CHIRON's paired phase-state update

```text
p_{l+1} = p_l + S_l(q_l)      # q-driven attention / shear update into p
q_{l+1} = R_l(q_l)            # ReLN / q-state evolution
```

where `p` acts as a momentum-like accumulator and `q` drives the shear.

---

## 1. Executive summary

The recommended program is **phase-space homeostasis**: regularize and/or
control CHIRON by measuring the actual `(p, q)` trajectory rather than by
adding generic residual-transformer tricks.

Three materially different formulations were considered:

1. **SIRA — Symplectic-Invariant Regularized Action.** A train-time auxiliary
   loss on phase energy, p/q balance, and layerwise action smoothness.
2. **PTOC — Phase-space Tangent Operator Consistency.** Sampled finite-
   difference checks that constrain local layer gain/curvature under small
   `(p,q)` perturbations.
3. **PHS — Phase-Homeostatic Servo Curriculum.** A data/schedule controller
   that uses CHIRON diagnostics to adapt sample mix and position weights.

**Selected first arc:** SIRA, with PHS as the follow-up generalizer and PTOC
as a diagnostic/robustness suite. SIRA is the best first pilot because it is
CHIRON-native, does not require extra prediction heads, does not perturb
inference, and can be introduced behind a single default-off coefficient.

---

## 2. Candidate formulations

### 2.1 Candidate A — SIRA: Symplectic-Invariant Regularized Action

**Primitive objects** for layer `l`, token or position bucket `b`:

- `p_{l,b}, q_{l,b}`: phase-state slices or bucket means.
- `u_{l,b} = S_l(q_l)_b`: q-driven shear added into `p`.
- `dq_{l,b} = q_{l+1,b} - q_{l,b}`: q-evolution displacement.
- `pbar_{l,b} = p_{l,b} + 0.5 u_{l,b}`: midpoint momentum proxy.

**Core quantities:**

```text
E_{l,b} = 1/2 * ( ||p_{l,b}||^2 / (m * sg(sigma_p,l^2) + eps)
                + ||q_{l,b}||^2 / (m * sg(sigma_q,l^2) + eps) )

B_{l,b} = 1/2 * log( (||p_{l,b}||^2 / (m * sg(sigma_p,l^2)) + eps)
                   / (||q_{l,b}||^2 / (m * sg(sigma_q,l^2)) + eps) )

A_{l,b} = dot(pbar_{l,b}, dq_{l,b})
          / (sqrt(m) * sg(sigma_pbar,l * sigma_dq,l) + eps)
```

where `sg(.)` means stop-gradient normalization. SIRA penalizes centered
layer drift:

```text
D^E_{l,b} = log(E_{l+1,b}) - log(E_{l,b})
D^B_{l,b} = B_{l+1,b} - B_{l,b}
D^2A_{l,b} = A_{l+1,b} - 2 A_{l,b} + A_{l-1,b}
```

using pseudo-Huber `rho_tau(z) = tau^2 * (sqrt(1 + (z/tau)^2) - 1)`:

```text
L_SIRA = lambda_sira * [
    w_E * mean rho_tau(D^E_{l,b} - mean_b D^E_{l,b})
  + w_B * mean rho_tau(D^B_{l,b} - mean_b D^B_{l,b})
  + w_A * mean rho_tau(D^2A_{l,b})
]
```

**Strengths:** directly targets CHIRON trajectory smoothness; no stochastic
layer dropping; no extra LM heads; no inference cost.

**Main failure mode:** over-conservation can suppress useful q-to-p shears and
hurt rare-token learning.

---

### 2.2 Candidate B — PTOC: Phase-space Tangent Operator Consistency

For sampled layer `l`, detach `x = (p_l, q_l)`, sample normalized phase
perturbation `u`, and compute ordinary layer forward evaluations:

```text
y0 = F_l(x)
y+ = F_l(x + eps * u)
y- = F_l(x - eps * u)
d_eps = (y+ - y-) / (2 eps)
```

Penalties:

```text
L_cycle = ||y+ - 2y0 + y-||^2 / (||y+ - y-||^2 + eta)

r = ||d_eps|| / (||u|| + eta)
L_gain = [log r - log beta]_+^2 + nu * [log alpha - log r]_+^2
```

Optionally measure local projected phase-area defect with
`Omega(a,b)=<a_p,b_q>-<a_q,b_p>` on same-token/head projections.

**Strengths:** explicitly controls local Lyapunov spikes and phase-flow
curvature.

**Main failure mode:** shadow layer evaluations can exceed the 5% wall budget
or improve robustness diagnostics without improving val NLL.

---

### 2.3 Candidate C — PHS: Phase-Homeostatic Servo Curriculum

Maintain detached EMA diagnostics over data group `g` and position bucket `b`:

```text
r_{g,b} = log(rms(p) / rms(q))             # p/q imbalance
s_{g,b} = rms(shear(q)) / rms(p)           # shear magnitude
a_{g,b} = cos(p, shear(q))                 # shear alignment
c_{g,b} = ReLN / q-state outlier measure
T_{g,b} = QK-Norm gamma / logit-temp proxy
ell_{g,b} = unweighted CE/NLL
```

Every `K` steps, update a bounded controller logit:

```text
need_{g,b}     = [ell_{g,b} - median(ell)]_+ + alpha * undercoverage_{g,b}
overload_{g,b} = norm([ |r|-tau_r, s-tau_s_hi, tau_s_lo-s, c-tau_c, |T|-tau_T ]_+)

u_{g,b} <- clip((1-rho)u_{g,b} + rho*(k_n*need - k_o*overload + k_i*I),
                -u_max, u_max)
```

Use `u_{g,b}` to adjust sample/crop distribution and token loss weights, with
sampling floors and mean-one normalization.

**Strengths:** data-side generalizer; default-off; can improve long-context
coverage without altering the CHIRON block.

**Main failure mode:** controller can Goodhart diagnostics, downweight truly
hard data, or introduce domain drift.

---

## 3. Selection rationale

| Criterion | SIRA | PTOC | PHS |
|---|---:|---:|---:|
| CHIRON-native geometry | **High** | **High** | High |
| No auxiliary prediction target | **Yes** | **Yes** | Yes |
| No inference cost | **Yes** | **Yes** | Yes |
| Expected implementation complexity | Medium | High | Medium/high in trainer |
| Expected wall overhead | **1–3%** | 3–8% unless sparse | <5% if diagnostics sparse |
| Direct NLL-improvement path | **Medium/high** | Medium | Medium |
| Risk of repeating failed ports | **Low** | Low | Low |
| Diagnostic value if it fails | High | **Very high** | High |

**Selected first arc:** SIRA. It gives the cleanest differentiable
CHIRON-native regularizer with the least change to the training task. PTOC is
better as a robustness diagnostic before it becomes an objective. PHS is a
promising generalizer, but it depends on trainer-side data instrumentation and
should follow once the phase diagnostics are available.

---

## 4. Formal problem statement

Find a default-off training mechanism `R_theta` such that

```text
L_train(theta) = CE(theta) + zloss(theta) + R_theta
```

improves unweighted held-out next-token NLL relative to regstack Phase 2, while
respecting:

1. `R_theta = 0` is bit-identical to current production defaults.
2. No new same-position denoising, MTP head, or stochastic layer skip.
3. Wall regression <= 5% unless a pilot explicitly justifies more.
4. Production comparison baseline is `chiron_1B_T16384_regstack_phase2.final`.
5. Gates are decided on main CE validation NLL, not on auxiliary-loss value.

---

## 5. Core selected framework: SIRA

SIRA assumes CHIRON generalizes best when the depth trajectory is neither
chaotic nor collapsed in phase space. The model may globally reshape energy
across depth, but token/position buckets should not suffer uncontrolled
relative spikes in:

- phase energy `E`,
- p/q imbalance `B`,
- discrete action curvature `D^2A`.

The key design choice is **centered drift**, not absolute conservation. This
avoids fighting useful learned depth schedules.

---

## 6. Objective function derivation

Let `b` index position buckets. Define layerwise normalized energy `E_{l,b}`.
Raw conservation would penalize `log E_{l+1,b} - log E_{l,b}`. That is too
strong because CHIRON may intentionally change global magnitude with depth.
SIRA instead penalizes only the bucket-relative component:

```text
L_E = mean_{l,b} rho_tau(D^E_{l,b} - mean_b D^E_{l,b})
```

The same centered construction applies to p/q balance:

```text
L_B = mean_{l,b} rho_tau(D^B_{l,b} - mean_b D^B_{l,b})
```

Action curvature needs no centering if it is already a second difference:

```text
L_A = mean_{l,b} rho_tau(A_{l+1,b} - 2A_{l,b} + A_{l-1,b})
```

Total training objective:

```text
L = CE + lambda_z * zloss + lambda_sira * (w_E L_E + w_B L_B + w_A L_A)
```

Initial coefficient proposal:

```text
lambda_sira = 3e-4
w_E = 1.0
w_B = 0.25
w_A = 0.5
tau = 0.2
warmup = 1000 steps
```

A safer first pilot may set `w_A=0` for the first 1k steps, then ramp to `0.5`.

---

## 7. Optimization algorithm

### SIRA training loop sketch

```text
for step in training:
    run normal CHIRON forward with qknorm + zloss

    if sira_coef > 0 and step >= sira_warmup:
        for probe layer l in configured probe_layers:
            collect p_l, q_l, shear_l(q_l), q_{l+1} bucket reductions
        compute E, B, A and L_SIRA
        add analytic SIRA gradients into resident phase-state gradients

    run normal backward / optimizer step
```

### Probe-layer schedule

Prototype should use a small set of layers first:

```text
probe_layers = {0, 4, 8, 12, 16, 20, 23}
position_buckets = 8 initially, then 16 if overhead permits
```

Only after a stable pilot should SIRA cover all 24 layers.

---

## 8. Temporal dynamics formulation

Treat depth `l` as discrete time. CHIRON is a learned phase flow:

```text
x_l = (p_l, q_l)
x_{l+1} = Phi_l(x_l)
```

SIRA is a weak prior that the flow has bounded nonuniform action curvature:

```text
Delta_l A = A_{l+1} - A_l
Delta_l^2 A = A_{l+1} - 2A_l + A_{l-1}
```

This is analogous to regularizing jerk rather than velocity: the model can
move through phase space, but abrupt layer-local deviations must be justified
by CE gradients.

---

## 9. Theoretical analysis

### Theorem-level statements

1. **Default-off identity.** If `lambda_sira = 0`, SIRA contributes no forward
   values, gradients, RNG draws, or kernel launches. The training path can be
   bit-identical to regstack Phase 2 if guarded before any work is performed.
2. **Scale invariance under stop-gradient normalization.** Multiplying all
   `p_l` or `q_l` in a layer by a constant leaves the normalized diagnostic
   ratios approximately unchanged, so the model cannot trivially minimize
   SIRA by global rescaling.

### Sketch-level arguments

1. **Conditioning:** bounding relative energy/action spikes limits layerwise
   gradient variance induced by rare position buckets, reducing long-context
   instability.
2. **Generalization:** centered phase smoothness acts as a geometry-aware
   capacity prior: it discourages brittle per-position phase trajectories
   without reducing the model to a lower-dimensional bottleneck.
3. **Compatibility with QK-Norm:** QK-Norm regularizes attention temperature
   before the shear; SIRA regularizes the post-shear phase trajectory. The
   mechanisms are adjacent but not redundant.

### Conjectures

1. SIRA will reduce bucket-to-bucket NLL variance at T=16384, even if aggregate
   NLL gain is small.
2. The action-curvature term is useful only after early warmup; applying it at
   step 1 will likely slow representation formation.
3. A small SIRA coefficient improves 30k NLL by >=0.01 nat; a large coefficient
   regresses by suppressing useful shear spikes.

---

## 10. Computational tradeoffs

Expected costs:

- Bucket reductions over selected probe layers: ~1–2% wall.
- Analytic gradient injection kernels: ~1% wall.
- No new persistent `T*m*L` buffers.
- No inference cost.

Hard limits:

- 5k pilot wall regression must be <=5%.
- Throughput ship floor: >=26,668 tok/s (95% of 28,072).
- VRAM must remain within existing 15.6 GB practical ceiling.

If overhead exceeds budget, fallbacks are:

1. fewer probe layers,
2. 8 buckets instead of 16/32,
3. compute SIRA every `K=2` or `4` steps with scaled coefficient,
4. energy/balance-only mode (`w_A=0`).

---

## 11. Comparison to failed/existing mechanisms

- **Not LayerDrop:** no stochastic depth, no train/inference distribution gap,
  no dropped `(shear, reln)` pairs.
- **Not UL2:** no same-position denoising or task mixture; CE remains the
  training task.
- **Not MTP:** no extra token prediction head or readout scratch.
- **Complements QK-Norm:** QK-Norm stabilizes attention logits; SIRA stabilizes
  the resulting phase trajectory.
- **Complements Z-loss:** Z-loss regularizes readout partition function; SIRA
  regularizes hidden phase dynamics.

---

## 12. Failure modes and mitigations

| Failure | Signal | Mitigation |
|---|---|---|
| Over-conservation / underfit | train CE lags baseline; all buckets worse | reduce `lambda_sira`, disable `L_A`, lengthen warmup |
| Fighting ReLN | p/q balance penalty dominates | lower `w_B`; make balance diagnostic-only |
| Rare-token suppression | acc1/rare buckets regress | Huber lower sensitivity; bucket by position not token identity |
| Wall overhead >5% | tok/s below 26.7k | fewer probe layers, every-K-step loss |
| Numerical instability | NaN/inf, grad norm spikes | coefficient ramp, gradient norm cap on SIRA branch |
| Null signal | NLL within noise, diagnostics improve only | close as diagnostic win; do not ship |

---

## 13. Minimal prototype

### Phase 0 — diagnostics only

Add default-off logging of:

- layer/bucket `rms(p)`, `rms(q)`, `rms(shear)`;
- p/q balance `B`;
- action proxy `A`;
- bucketed CE NLL.

No training objective change. This de-risks overhead and identifies whether
Phase 2 has any remaining phase irregularity.

### Phase 1 — energy + balance SIRA

Flags:

```text
--sira-coef 3e-4
--sira-energy-weight 1.0
--sira-balance-weight 0.25
--sira-action-weight 0.0
--sira-warmup 1000              # alias: --sira-warmup-steps
--sira-probe-layers 0,4,8,12,16,20,23
--sira-position-buckets 8
```

### Phase 2 — add action curvature

If Phase 1 is stable and not regressive:

```text
--sira-action-weight 0.5
```

### Minimum run matrix

| ID | Config | Steps | Purpose |
|---|---|---:|---|
| S0 | Phase 2 baseline rerun | 5k | fresh comparator |
| S1 | diagnostics only | 1k | overhead + phase map |
| S2 | energy+balance | 5k | first real pilot |
| S3 | energy+balance+action | 5k | only if S2 passes stability |

---

## 14. Full research program

1. **SIRA arc:** differential phase regularizer; decide on 5k gates.
2. **PHS arc:** reuse diagnostics for data/control generalization after SIRA
   instrumentation exists.
3. **PTOC arc:** use sampled tangent diagnostics to explain failures; only make
   PTOC a loss if SIRA/PHS indicate local gain spikes are the dominant issue.
4. **Combined policy:** if SIRA improves hidden trajectory but not NLL, use PHS
   to target data/position buckets where diagnostics remain unhealthy.

---

## 15. Open conjectures and validation criteria

### 5k continue gate

Continue beyond 5k only if all hold:

- main CE val NLL <= baseline + 0.02 nat, and preferably <= baseline - 0.02;
- no position bucket regresses by >0.03 nat;
- wall regression <=5%;
- no persistent grad norm spikes >2x baseline;
- SIRA loss contribution remains <1% of CE after warmup.

### 30k ship gate

Ship only if:

- final val NLL <= **3.5534** (>=0.020 nat better than 3.5734), or a
  pre-registered multi-seed result shows a statistically clear smaller gain;
- throughput >= **26,668 tok/s**;
- VRAM remains under practical ceiling;
- bucket-7 / long-context NLL does not regress;
- all new flags default off and disabled path is bit-identical.

### Retire criteria

Retire SIRA if:

- 5k regression >0.05 nat;
- stable coefficients are all too small to change diagnostics;
- improvements occur only in auxiliary diagnostics and not in unweighted CE;
- implementation cannot meet the wall budget without making the objective too
  sparse to matter.
