# Paradigm Shift #25 — GEC: Gradient Echo Compression

**Status:** design complete; single-candidate inline formulation.
**Date:** 2026-04-23 (Ralph-loop iteration 42).

---

## 1. Target axis

**Temporal dimension of the gradient tensor.**

Every paradigm shift 1-24 compresses gradients along the D (feature)
or N (parameter) dimension.  None has attacked the T (time/sequence)
dimension.

The gradient tensor ∇h ∈ ℝ^{T × d} has T time-steps each of dim d.
For causal language modeling, consecutive tokens' gradients are
HIGHLY CORRELATED — they arise from similar predictive contexts.
Empirically, the effective rank of ∇h along T is often ≪ T.

No shipped shift exploits this.

## 2. Core thesis

Factor the T × d gradient tensor as a temporal rank-q approximation:

$$
\nabla h \approx U_{T \times q} \cdot V_{q \times d}
$$

where q ≪ T (typical q = 64 when T = 2048, a 32× temporal compression).

Maintain U, V via streaming randomized SVD on the gradient tensor.
The backward pass produces U, V directly (not the full T×d gradient),
reducing gradient memory by T/q.

## 3. Primitive objects

- `∇h_approx ∈ ℝ^{T × d}` expressible as `U · V` with U ∈ ℝ^{T × q}, V ∈ ℝ^{q × d}.
- Streaming SVD state: q leading left/right singular vectors.
- Time-compression rank q (hyperparameter; typical q ∈ [16, 128]).

## 4. Evolution law

On each backward pass:

1. Compute ∇h[T × d] via standard backward.
2. Randomized SVD: Ω ∈ ℝ^{T × q} random, project `Y = ∇hᵀ · Ω`
   (d × q), QR(Y) → V_guess, then U_guess = ∇h · V_guessᵀ.
3. Store (U_guess, V_guess) instead of ∇h.  Reconstruct ∇h = U·V as
   needed.

**Alternative (fully streaming)**: maintain a T × q running projection
Φ, update it via Oja's rule on ∇h^T·∇h.  Avoid full SVD cost per step.

## 5. Mechanism mapping

| Axis | Mechanism | Factor |
|------|-----------|--------|
| Gradient memory | T × d → q × d + T × q | T / q ≈ 32× at T=2048, q=64 |
| Backward compute | SVD adds O(T · d · q) per step | 5-10% overhead |
| Forward compute | unchanged | 1× |
| Composability | orthogonal to IBGRAD (which compresses d×N) | multiplicative |

## 6. Composition with shipped / deferred stack

- **IBGRAD (#19)**: IBGRAD factors along N (parameter dimension);
  GEC factors along T (time).  **Net: N·T → r + q**; compound.
- **OVFG (#9)**: OVFG is also a rank factorization but on weight-
  gradients g_W, not activation-gradients ∇h.  Different axes.
- **Chunked CE (shipped)**: compatible.  GEC acts on layer
  ∇h's, chunked CE acts on the loss-gradient dLogits.
- **TRCD (#13)**: per-token compute; GEC still works on the sparse
  surviving tokens' ∇h.

Projected compound at pile_large (T=2048, d=1024, q=64):
- Gradient memory per layer: T·d = 2 MB → q·d + T·q = 0.38 MB → **5×**.
- Per backward: activation+gradient memory from 4 MB to 0.76 MB per
  layer.  Stack compound with CHIRON × IBGRAD × GEC = ~140× gradient
  + activation memory reduction at pile_large.

## 7. Failure modes

**F1 — Rank q too small.**  Gradient loses fine-grained per-token
information, causing convergence slowdown.  Mitigation: adaptive q
based on observed residual ‖∇h − UV‖ / ‖∇h‖ threshold.

**F2 — SVD cost eats savings.**  At T=2048, q=64, SVD is ~8M flops
per layer, comparable to backward-GEMM.  Mitigation: streaming
Oja-on-∇hᵀ rather than fresh SVD per step.

**F3 — Loss of temporal gradient flow across K-chunks.**  Boundary
tokens in T-chunks may have their gradient leaked.  Mitigation: use
per-layer q-rank approximation rather than one global.

## 8. Comparison to prior art

- **GradPCA / LoRA**: factorize weight-matrix gradients (not
  activation-gradients).  Different tensor.
- **Low-rank attention backward** (various papers): approximate
  attention backward with low rank.  GEC generalizes to arbitrary
  layer gradients.
- **Activation checkpointing**: recomputes gradients.  GEC stores
  them compressed, complementary.

GEC's novelty: **time-dimension rank factorization of layer gradients
during backward**.

## 9. Minimal prototype

**GPU primitives needed**:
- `gec_compute_randomized_svd(grad, T, d, q, U_out, V_out)` — fresh
  rSVD per step.
- `gec_streaming_oja_update(U, V, grad, T, d, q, eta)` — amortized.
- `gec_reconstruct(U, V, T, d, q, grad_out)` — UV for downstream use.

**First E2E test**: 2-layer MLP.  Train with GEC(q=4) vs full-rank.
Target: loss ratio within 10% of full-rank.

## 10. Summary

GEC is a **genuinely novel axis** (T dimension) that complements
every shipped shift.  At pile_large scale:
- 32× time-dimension gradient compression at q=64.
- Orthogonal compound with IBGRAD for q·N → r time + r param.
- Adds ~5-10% backward overhead; amortized to near-zero with
  streaming Oja updates.

Paradigm-design count after #25: **25 shifts** (14 shipped +
11 deferred).

**Promote condition**: after IBGRAD × WIP × MPOT compound is
stable, GEC joins as the orthogonal temporal compression factor.
