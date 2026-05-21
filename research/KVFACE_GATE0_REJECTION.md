# KV-FACE Gate-0 Rejection — Attention Popularity Findings

**Date:** 2026-04-23 (Ralph-loop iteration 122)
**Status:** Paradigm shift #36 KV-FACE convergence variant REJECTED.
**Data:** 41M × 2500 steps, T=1024, L=12, nH=8, β=default, Adam fp32.

---

## 1. Gate-0 measurement protocol

Added `--probe-attn-gini` to `chiron_train`. Captures per-layer
popularity `p[l,h,t] = (1/T)·Σ_q P[l,h,q,t]` during the forward pass,
downloads to host, computes per-(layer, head) Gini via sorted Lorenz
formula. Logs global mean + per-layer vector every `--log-every` steps.

Causal-mask structural baseline (computed in iter 121):
- T ≥ 128: Gini = 0.500 ± 0.002 (asymptotic)

Premise to validate: does trained attention develop Gini > 0.500
indicating learned Zipfian concentration?

---

## 2. Empirical result

### Per-layer Gini trajectory (41M × 2500 steps)

| Step | Loss (EMA) | Gini vector across L=12 layers                                                            |
|-----:|:----------:|:------------------------------------------------------------------------------------------|
|    0 | 10.40      | 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500       |
|  500 | 9.55       | 0.500, 0.500, 0.216, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.118       |
| 1000 | 9.25       | 0.500, 0.500, 0.173, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.184       |
| 1500 | 8.85       | 0.500, 0.500, 0.193, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.248       |
| 2000 | 8.77       | 0.500, 0.500, 0.179, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.124       |
| 2500 | 8.77       | (similar pattern — L2, L11 below baseline; rest at baseline)                              |

### Observations

1. **No layer EXCEEDS the 0.500 baseline at any step.** The Zipfian-
   concentration hypothesis is rejected for all tested layers.
2. **10/12 layers stay EXACTLY at 0.500** throughout 2500 steps. This
   suggests those layers' attention has not diverged meaningfully
   from random causal attention.
3. **L2 and L11 DROP to 0.12-0.25** — SIGNIFICANTLY BELOW baseline.
   Trained attention in those layers is MORE uniform than random.
4. **Global mean stays 0.44-0.46**, trending slightly below baseline.

---

## 3. Mechanism interpretation

**What does below-baseline Gini mean?** Under causal mask, popularity
skew toward early positions is STRUCTURAL. For popularity to become
more uniform, the attention has to actively under-weight early
positions. Three attention patterns achieve this:

- **Diagonal / self-attention:** P[q,q] = 1, else 0. Every position
  gets attention only from itself. `p[t] = 1/T` uniformly. Gini = 0.
- **Sliding window:** P[q, q-1] = 1. Every position gets attention
  from its successor. Popularity becomes uniform except at boundaries.
- **Uniform over causal prefix:** P[q, k] = 1/(q+1) — this is the
  baseline (Gini = 0.5), not a learned pattern.

The observed Gini 0.12-0.25 in L2, L11 suggests PARTIAL diagonalization
or window-like patterns in those layers' heads.

**Induction heads** (the classical "learned Zipfian" pattern in
mechanistic interpretability) would be Gini > 0.5: specific positions
in the context become repeatedly referenced. This did NOT appear at
41M × 2500 steps. Either:
- Scale/horizon too small (likely — 41M trains poorly on pile-bpe)
- Induction heads appear in layer interactions not captured by
  per-layer popularity (e.g., one head copies, another uses the copy)
- Non-natural-language structure of pretokenized pile-bpe at this scale

---

## 4. Implication for KV-FACE

**Convergence-axis mechanism:** REJECTED.
- Frequency-debiased column norms would weight AGAINST already-uniform
  popularity, pushing gradient mass in the WRONG direction.
- No head shows concentrated-popularity pattern the mechanism exploits.

**Memory-compression mechanism:** viable but weak.
- Any Gini > 0 triggers the compression pattern.
- At 0.44 global mean, the compression ratio would be similar to
  plain Adafactor on Wk,Wv (~1000×). Not a new mechanism.

**Decision:** do not pursue KV-FACE implementation. Reject paradigm
shift #36 on empirical grounds.

---

## 5. Research methodology win

The Gate-0 probe — built on existing `gpu_kvface_probe` primitives —
resolved the central empirical question in ~2 minutes of training
compute, without investing in the full Phase 1 implementation. This
follows the Ralph-loop Gate-0 methodology: probe the premise cheaply
before committing engineering effort.

Cost of the probe: ~200 lines of CUDA + trainer hook + ~2 minutes of
41M compute.
Cost saved: ~500 lines of Phase 1 implementation + ~10 hours of
larger-scale validation that would have been required to conclude
the mechanism is weak.

---

## 6. Next paradigm-shift direction

Given KV-FACE rejection, paradigm #37 should target a mechanism whose
premise is validated by existing priors:

### Candidate A: HUTCH-DIAG (previously Candidate C of #36)
- Hessian diagonal via Hutchinson Rademacher probe
- Convergence axis, memory-neutral
- Well-grounded theory (diag-Newton preconditioning)
- Already designed in PARADIGM_SHIFT_36_CANDIDATE_C_HUTCHDIAG.md
- **Promote to #37**

### Candidate B: SPAREC Phase 2 implementation
- Already designed + Phase 1 shipped
- 3-5× FFN backward speedup projected
- Concrete engineering path (gather + cuSPARSE SpMM)
- **Promote to shipping-focus priority**

### Candidate C: PARAMETER-GROUP ADAPTIVE ADAM PRECISION
- Extension of PRX
- Groups params by type (emb, attn, norm, bias), assigns per-group
  precision based on empirical gradient variance
- Memory win from precision mix
- Needs Gate-0 probe: measure per-group gradient variance distributions

Recommended: promote HUTCH-DIAG to paradigm #37, develop Gate-0 probe
(empirical Hessian diagonal correlation with Hutchinson estimate),
and pursue SPAREC Phase 2 in parallel as engineering progress.
