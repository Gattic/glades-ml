# Paradigm Shift #37 — HUTCH-DIAG: Hutchinson Diagonal-Hessian Preconditioner

**Date:** 2026-04-23 (Ralph-loop iter 123, post-KV-FACE rejection)
**Status:** Promoted from paradigm #36 Candidate C. Full design in
`PARADIGM_SHIFT_36_CANDIDATE_C_HUTCHDIAG.md` (572 lines).
**Gate-0 probe:** pending (see §4 below).

---

## 1. Why HUTCH-DIAG for paradigm #37

After paradigm #36 KV-FACE rejection (iter 122), the design space of
"attention popularity compression" is closed — trained attention does
not develop the Zipfian concentration that would justify the mechanism.
The remaining candidates from #36's three-way split are:

| | Candidate | Axis | Fate |
|-|-----------|------|------|
| A | TAIL-CE | V-dim softmax (CE compute) | Deferred until V ≥ 128k or rare-token LMs |
| B | KV-FACE | Attention popularity EMA | REJECTED at Gate-0 (iter 122) |
| C | **HUTCH-DIAG** | **Hessian-diagonal Adam v_t replacement** | **PROMOTE to #37** |

HUTCH-DIAG's strengths for the Ralph-loop brief ("magnitudes less memory,
magnitudes faster"):

1. **Magnitudes faster (convergence).** Diag-Newton preconditioning
   delivers 2-5× wall-clock convergence speedup in published Hessian-aware
   optimizers (K-FAC, Shampoo, PyHessian).  Stacked with FACE's -0.67 nat
   advantage, expected compound: 3-10× over dense Adam.
2. **Memory-neutral.** Replaces v_t with v_hutch of identical size.  Fits
   in the existing BF16-Adam memory footprint.
3. **Compatible with 1.84B ceiling.** No new tensors — no ceiling regression.
4. **Unbiased estimator.** E[v⊙Hv] = diag(H) exactly under Rademacher v.
5. **Cheap amortization.** 1/K extra backward passes at K=16 = 6.25% overhead.

---

## 2. Mechanism summary (from Candidate C doc)

Per-parameter update rule:
```
m_t      ← β₁ m_{t-1} + (1-β₁) g_t
v_hutch  ← β₂ v_hutch  + (1-β₂) |v ⊙ Hv|   [every K steps]
θ_{t+1}  ← θ_t − lr · m_t / (√v_hutch + ε)
```

where `v ∈ {-1, +1}^d` is a Rademacher probe vector, and `Hv` is computed
via Pearlmutter's double-backward trick (one extra backward pass).

**Composition:**
- FACE (#28) on embedding: **keeps** (embedding unaffected — FACE owns E's state)
- MFIO (#11) on attention: **keeps** (MFIO preconditioner lives in its own σ, no collision)
- BF16 stack: **compatible** (v_hutch can be bf16-stored like v)
- HUTCH-DIAG applies to **non-FACE, non-MFIO Adam params** (Wo, FFN, norms, biases)

## 3. Expected throughput cost

Per every K=16 steps:
- 1 additional backward pass on Rademacher probe `v · (∇L)` → O(step backward cost)
- Amortized: +6.25% overhead

Per every step:
- `v_hutch` EMA update: O(d) elementwise — negligible vs Adam update
- Adam step itself: unchanged memory pattern

Expected net: **6-10% throughput overhead** for **2-5× convergence speedup**.
Net wall-clock speedup: **2-4× to reach a target loss**.

## 4. Gate-0 probe design

**Research question.** Does the Hutchinson diagonal estimate `v ⊙ Hv` for
a single Rademacher probe v correlate with the TRUE diagonal of the
Hessian at trained-CHIRON parameters?

**Why this matters.** The unbiasedness `E[v⊙Hv] = diag(H)` guarantees the
estimator is correct IN EXPECTATION. But at single-probe resolution, the
variance can be enormous if the Hessian has strong off-diagonal structure.
The relevant quantity is the ratio:

  ρ(v_hutch, diag(H_true)) = Cov(v_hutch_i, diag(H)_i) / sqrt(Var·Var)

If ρ < 0.3, a single probe is too noisy and K=16 averaging may not be
enough. If ρ > 0.6, the estimator is good enough for direct replacement
of v_t.

**Probe algorithm:**

1. Run CHIRON training to a checkpoint (e.g., 66M × 500 steps).
2. Sample a subset S of 10,000 parameters (stratified across layer types).
3. For each parameter θ_i ∈ S:
   - Compute true `H_ii = ∂²L/∂θ_i²` via FINITE DIFFERENCES:
     H_ii ≈ [∇L(θ + ε e_i)_i − ∇L(θ)_i] / ε
     (requires 10,000 extra gradient evaluations — parallelizable)
4. Compute Hutchinson estimate per Rademacher probe:
   v_hutch_i = v_i · (Hv)_i  for a batch of N_probes ∈ {1, 4, 16} probes.
5. Compute per-probe correlation ρ(v_hutch, H_ii) on S.

**Accept criteria:**
- N=1 probe: ρ ≥ 0.3 → single-probe Hutchinson is directly usable at K=1
- N=16 probes: ρ ≥ 0.6 → K=16 averaging is strong signal
- N=16 probes: ρ < 0.4 → REJECT: variance too high, mechanism weak

**Probe cost:** ~2 GPU-hours for a 66M model. Within a single Ralph-loop iteration.

## 5. Phase 1 implementation plan

Gated on Gate-0 PASS:

- `gpu_hutchdiag.h/.cu` — 3 primitives:
  1. `hutchdiag_rademacher_probe(seed, theta_shape, v_out)` — generate Rademacher ±1
  2. `hutchdiag_hvp_backward(theta, grad, v, Hv_out)` — Pearlmutter HVP
  3. `hutchdiag_update_vema(v_hutch, v, Hv, beta)` — EMA update of v_hutch
- Unit tests:
  - Parity: `hvp_backward(θ, ê_i) ≈ (∇L(θ+ε ê_i) - ∇L(θ))/ε` to 1e-3
  - Convergence: on a 2-layer MLP, HUTCH-DIAG vs Adam → ≥ 1.5× faster at fixed final loss
- Trainer flag: `--hutch-diag K` (K = probe period in steps)

## 6. Phase 2-3 validation plan

**3-gate validation:**
1. Primitive parity (≤ 1e-4 error on HVP)
2. Trainer smoke (≥ 100 steps no divergence)
3. Multi-scale long-horizon (66M, 234M, 500M × 2500 steps, Δ vs FACE+Adam baseline)

**Success criterion:**
- Convergence: ≥ 0.1 nat sustained advantage (stacked with FACE)
- Throughput: ≤ 10% overhead
- Memory: ≤ 1% overhead (just probe buffers)

## 7. Failure-mode hedge

If Gate-0 passes but phase-2/3 shows no convergence improvement:
- Mechanism reduces to Adam (v_hutch ≈ v after EMA smoothing)
- Throughput loss acceptable; no scale regression
- Research outcome: "empirical Hessian diagonal ≈ empirical Fisher diagonal
  at trained state" — a negative but publishable finding

If Gate-0 fails (low correlation):
- Pivot to K-FAC-LITE (Kronecker-factored Hessian block)
- Or: pursue SPAREC Phase 2 (throughput axis, already validated in Phase 1)

## 8. Composition with shipped stack

```
./build/glades_chiron_train --face 1 --hutch-diag 16 \
    --mfio 2 --bf16-adam --bf16-weights --bf16-grads
```

All orthogonal. Expected stacking:
- FACE: embedding Adam state compression + convergence
- MFIO: attention Adam state compression
- HUTCH-DIAG: non-FACE non-MFIO Adam v_t → Hessian diagonal
- bf16 stack: uniform precision compression

## 9. Summary

HUTCH-DIAG is the natural paradigm #37:
- Premise (Hessian information helps convergence) backed by 30 years of optimization literature
- Memory-neutral → no ceiling regression
- Composable with all shipped paradigms
- Cheap Gate-0 probe (~2 GPU-hours on 66M)

**Next iteration action:** implement Gate-0 probe (compute finite-difference
diag(H) at a 66M checkpoint + compare to single/multi-probe Hutchinson
estimate). If ρ ≥ 0.3, proceed to Phase 1.
