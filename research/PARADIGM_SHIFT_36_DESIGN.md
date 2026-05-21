# Paradigm Shift #36 — KV-FACE: FACE-structure preconditioner for attention K,V projections

**Date:** 2026-04-23 (Ralph-loop iter 120, post-FACE 1.4B validation)
**Status:** design complete; Gate-0 probe required before Phase 1.
**Selected over:** TAIL-CE (candidate A), HUTCH-DIAG (candidate C).
**Parent:** paradigm shift #28 FACE — same mechanism, transplanted from embedding → attention.

---

## 0. Candidate selection rationale

Three materially different candidates were developed in parallel (see
`PARADIGM_SHIFT_36_CANDIDATE_{A,B,C}_*.md`):

| Candidate | Memory win | Speed win | Gate-0 cost | Risk | Ceiling |
|-----------|:----------:|:---------:|:-----------:|:----:|:-------:|
| **A TAIL-CE** (V-dim softmax compression) | 40× on CE stage activation | ~15-40% end-to-end | low (1 fwd pass + top-K mass) | medium (tail mass may be thicker than expected at warmup) | modest speedup on a 20%-of-step chunk |
| **B KV-FACE** (FACE transplant on Wk,Wv) | **~1200× on attention Adam state** | convergence: ~0.2-1.0 nat (speculative) | **trivial (1 fwd pass + Gini of p[t])** | high — premise may fail if attention is uniform at trained state | **next "FACE" (disrupting)** |
| **C HUTCH-DIAG** (Hessian-diag via Rademacher probe) | 0× (memory-neutral) | 2-5× convergence (solid theory) | medium (need hours of HVPs to validate) | medium — negative eigenvalues require |·| | strong convergence, no memory axis |

**Selection: KV-FACE.** Rationale:

1. **Highest ceiling on the DISRUPTING criterion.** The Ralph-loop brief
   demands BOTH magnitudes less memory AND magnitudes faster. FACE is the
   only shipped disrupting shift. KV-FACE is the only candidate with
   analogous 1000×-scale compression potential AND a convergence claim.
2. **Cheapest Gate-0 probe.** A single 1-second forward pass on a trained
   500M checkpoint, measuring the Gini coefficient of the position-popularity
   distribution `p[t] = (1/T)·Σ_q A[q,t]`, settles the central empirical
   question. If Gini ≥ 0.4 (clearly non-uniform), the premise holds.
3. **Graceful degradation on failure.** If attention is uniform, KV-FACE
   reduces to Adafactor on Wk,Wv — a known, valid method. Rejection does
   not destroy all prior compute.
4. **Mechanism reuse.** 50% of the CUDA primitives are shared with
   `gpu_face.cu`; engineering cost is dominated by the 3 new kernels for
   attention-popularity EMA + induced-column-popularity.
5. **Orthogonal axis to FACE.** FACE compresses embedding (V × m) Adam
   state. KV-FACE compresses attention (m × dModel) Adam state. They stack
   multiplicatively across the largest single tensor families in the
   model.

HUTCH-DIAG rejected: memory-neutral (zero memory axis) → cannot alone
satisfy the disrupting criterion. May be re-promoted as paradigm #37
after KV-FACE's outcome resolves.

TAIL-CE rejected: the CE stage is a real compute chunk but 15-40%
end-to-end speedup does not clear the "magnitudes faster" bar. May be
promoted after the next LLM-scale vocab upgrade (V ≥ 128k).

---

## 1. Target axis

**Attention K, V weight-matrix Adam state.** At 1.4B CHIRON
(m = 1536, dModel = 2048, L = 44), each of Wk, Wv is 1536 × 2048 = 3.15 M
floats per layer. Across 44 layers: 277 M floats per matrix type × 2
matrices = 554 M floats. Adam state (m + v) = 2 × 554 M = 1.1 B floats
= **4.2 GB**. This is the single largest non-embedding Adam-state
consumer at this scale.

FACE already compresses the embedding (V × m) Adam state by 1984× at 1.4B.
The attention K,V Adam state is the next-largest untapped reserve.

---

## 2. Core thesis

**Hypothesis H36.** During transformer attention at trained state, the
distribution of per-position attention-popularity
`p[t] = (1/T)·Σ_q A[q,t]` is non-uniform (Zipfian-like, Gini coefficient
≥ 0.4). This non-uniformity is inherited by the column-popularity signal
`f̃_att[j] = Σ_t p[t]·(X^T·dK)[t, j]`, which plays the role that token
frequency plays in FACE.

**Corollary.** If H36 holds, the FACE mechanism (row-unconditional EMA +
frequency-debiased column EMA) applied to Wk,Wv Adam state should deliver:
- ~1000-2000× Adam state compression (matching FACE's embedding ratio)
- ~0.2-1.0 nat convergence advantage (analogous to FACE's on embedding)
- ≤0.3% throughput overhead (3-kernel per-step cost, same pattern as FACE)

**Falsification.** If the 500M-checkpoint Gate-0 probe returns Gini < 0.3
(near-uniform attention), the mechanism cannot deliver beyond plain
Adafactor; promote-condition fails.

---

## 3. Primitive objects

For each attention layer ℓ:
- `Wk^{(ℓ)}, Wv^{(ℓ)} ∈ ℝ^{m × dModel}` — projection matrices
- `A^{(ℓ)} ∈ ℝ^{nHeads × T × T}` — attention probabilities (row-stochastic, causal)
- `p^{(ℓ)}[t] = (1/(nHeads·T))·Σ_{h,q} A[h,q,t]` — position popularity, head-averaged
- `X ∈ ℝ^{T × m}` — layer input (same for Wk and Wv)
- `dK, dV ∈ ℝ^{T × dModel}` — upstream gradients from attention backward

Per-layer KV-FACE state:
- `zn̄_k[m], zn̄_v[m]` — per-input-channel running EMA of `Σ_j dW[i,j]²`
- `dn̄_k[dModel], dn̄_v[dModel]` — per-output-column running EMA
- `p̄[T]` — per-position popularity EMA (shared across k, v)
- `f̃_bar_k[dModel], f̃_bar_v[dModel]` — induced column popularity EMA
- `gF̄_k, gF̄_v` — scalar global frequency averages for normalization

Total state per layer: `2·m + 2·dModel + T + 2·dModel + 2`
At 1.4B (m=1536, dModel=2048, T=1024): `3072 + 4096 + 1024 + 4096 + 2`
= **12,290 floats per layer** (49 KB).

Across L=44 layers: **541 KB total** vs **4.2 GB dense Adam** →
**~8000× compression**.

---

## 4. State space

Per-layer state tuple: `S^{(ℓ)} = (zn̄_k, dn̄_k, zn̄_v, dn̄_v, p̄, f̃_k, f̃_v, gF̄_k, gF̄_v)`.
Integer step counter `t_step` for bias correction.

---

## 5. Evolution law

Per gradient step, for each attention layer ℓ:

1. **Attention popularity capture** (during forward pass):
   `p[t] = (1/(nHeads·T))·Σ_{h,q} A^{(ℓ)}[h,q,t]`
   `p̄[t] ← β_pop · p̄[t] + (1 - β_pop) · p[t]` (β_pop = 0.98)

2. **Induced column popularity** (during attention backward):
   `f̃_k[j] = Σ_t p[t] · (X^T · dK)[t, j]`
   `f̃_bar_k[j] ← β_col · f̃_bar_k[j] + (1 - β_col) · f̃_k[j]`  (β_col = 0.98)
   (and analogously for `f̃_v`)

3. **Row/column EMAs** (FACE structure):
   `zn̄_k[i] ← β_row · zn̄_k[i] + (1 - β_row) · Σ_j dWk[i,j]²`
   `dn̄_k[j] ← β_col · dn̄_k[j] + (1 - β_col) · Σ_i dWk[i,j]²`
   (and for Wv)

4. **Global frequency average** (FACE's q̂):
   `gF̄_k ← β_col · gF̄_k + (1 - β_col) · (1/dModel)·Σ_j f̃_bar_k[j]`

5. **Preconditioned update** (FACE σ formula):
   `σ_k[i,j] = 1 / √(zn̄_k[i]·dn̄_k[j] / (q̂ · gF̄_k) + ε²)`
   `Wk[i,j] ← Wk[i,j] - lr · dWk[i,j] · σ_k[i,j]`
   `q̂ = f̃_bar_k[j] / gF̄_k` (per-column frequency-debias factor)

Scale-aware tuning (from FACE validation):
- `< 500M`: β_row = 0.999
- `500M - 1B`: β_row = 0.99
- `≥ 1B`: β_row = 0.98

Default β_pop = 0.98 (same timescale as FACE's β_col).

---

## 6. Mechanism mapping

| Requirement | Realization |
|-------------|------------|
| Per-token activation sparsity exploitation | Per-position attention-popularity `p[t]` encodes sparsity |
| GPU-implementable | Reuses `gpu_face.cu` primitives + 3 new kernels: `attn_popularity`, `induced_col_pop`, `kv_face_apply_update` |
| Trainer flag | `--kv-face 1` in `chiron_train` |
| Composable with MFIO | MFIO currently covers Wq/Wk/Wv. New default: `--mfio 3` = MFIO on Wq only, KV-FACE on Wk,Wv |
| Composable with WIP | WIP operates on Wo, orthogonal — no conflict |
| Composable with FACE | Different weight-matrix target — stacks multiplicatively |
| Runs at 2-30B scale | Memory-positive; only enables larger scales |

---

## 7. Objective

The FACE objective is minimizing a regularized NLL with an implicit
Zipfian prior on token frequency. KV-FACE's objective is analogous:
implicit regularization on attention-popularity distribution. The
preconditioner `σ_k ∝ 1/√(zn̄·dn̄/(q̂·gF̄))` damps updates to columns
that correspond to rarely-attended positions (small `q̂`), preserving
their learned direction and preventing overshoot from high-variance
sparse updates.

---

## 8. Stability / expressivity

**Stability.** The ε² floor in the denominator prevents division-by-zero
when `q̂·gF̄ → 0` (at initialization or for a newly unpopular position).
Analogous to FACE's stability which held at 1.4B.

**Expressivity.** KV-FACE introduces no bottleneck rank; Wk and Wv remain
full-rank. Only the preconditioner geometry changes. Therefore KV-FACE
cannot reduce expressivity relative to dense Adam — only change the
convergence trajectory.

**Worst-case degeneracy.** If `p[t]` is uniform (Gini → 0), then
`f̃_att[j] → (1/T)·Σ_t (X^T·dK)[t, j] = (1/T)·dn_k[j]`, so `q̂ → 1/dModel`
constant. The σ_k formula reduces to plain Adafactor (FACE without the
frequency-debias). No catastrophic failure, just loss of the unique
mechanism.

---

## 9. Computational trade-offs

Per-step overhead (per layer):
1. `attn_popularity` kernel: O(nHeads·T²) reduce, negligible vs attention's O(nHeads·T²)
2. `induced_col_pop` kernel: O(T·dModel) per matrix (Wk, Wv), 2× that
3. Row/col EMA: O(m·dModel + m + dModel) per matrix
4. Preconditioned update: O(m·dModel) per matrix

All kernels are memory-bandwidth-bound and fuse within 3 launches per
layer. Empirical projection: ≤0.3% throughput overhead (matching FACE).

Memory:
- State: 541 KB total at 1.4B (4.2 GB Adam savings per attention weight pair)
- Cross-layer: linear in L

---

## 10. Prior art comparison

| Method | Axis | KV-FACE's distinction |
|--------|------|----------------------|
| Adafactor (Shazeer+Stern 2018) | row/col factoring | KV-FACE adds attention-popularity-weighted column norm (frequency debias) |
| MFIO v2 (Glades #11) | attention σ | MFIO is symmetric; KV-FACE adds popularity weighting through the chain rule |
| K-FAC (Martens+Grosse 2015) | Kronecker-factored Hessian | K-FAC is second-order; KV-FACE is a first-order preconditioner |
| FACE (Glades #28) | embedding freq-debias | KV-FACE is FACE transplanted to attention K,V with attention-popularity as the frequency signal |
| LoRA (Hu+ 2021) | low-rank adapters | Orthogonal — KV-FACE compresses Adam state, LoRA compresses trainable delta |

No prior method uses attention-popularity as the frequency signal for a
per-column preconditioner on Wk, Wv.

---

## 11. Failure modes + mitigations

1. **Attention becomes uniform at trained state** (dominant failure).
   Mitigation: Gate-0 probe on 500M checkpoint. If Gini < 0.3, reject.
2. **Popularity signal is dominated by causal-mask bias** (early tokens
   always popular). Mitigation: measure popularity across multiple batches,
   report per-head Gini (induction heads will be more popular).
3. **Composition conflict with MFIO on Wk,Wv.** Mitigation: split MFIO to
   Wq only (introduce `--mfio 3`).
4. **Instability at β_row = 0.98 at small scale**. Mitigation: inherit
   scale-aware β_row recipe from FACE validation.
5. **Attention popularity EMA lags the true distribution at phase
   transitions.** Mitigation: warmup with higher β_pop for first 500
   steps, then transition to steady-state β_pop = 0.98.

---

## 12. Minimal prototype

**Phase 1** (parity + smoke test):
- `gpu_kvface.h/.cu` — 3 new primitives:
  - `kvface_attn_popularity(A, p, nHeads, T)` — popularity reduction
  - `kvface_induced_col_pop(X, dK, p, f_att, m, dModel, T)` — popularity-weighted column sum
  - `kvface_apply_update(W, zn, dn, f_att, gF, lr, eps, ...)` — FACE-structure preconditioned update
- Host wrapper: `sparec_backward_dense_reference_cpu` analogue
- CHIRON unit tests: parity to FP32 precision with dense Adam baseline
- Trainer flag: `--kv-face 1`

**Phase 2** (trainer wire-in):
- Modify `sgd_transformer.cpp` attention backward to route dK, dV through
  `kvface_induced_col_pop` before applying FACE-structure update
- Add `--kv-face-beta-pop`, `--kv-face-beta-row` CLI flags

**Phase 3** (validation):
- Gate-0 probe at 500M (measure Gini, publish GO/NO-GO)
- If GO: 66M, 234M, 500M scale-sweep with 3-gate validation
- If NO-GO: document mechanism failure, promote paradigm #37

---

## 13. Composition with shipped stack

Full 4-shift compound at small-medium scale (< 500M):
```
./build/glades_chiron_train --mfio 3 --wip-K 4 --face 1 --kv-face 1 \
    --face-beta-row 0.999 --kv-face-beta-row 0.999
```

Full large-scale compound (≥ 1B):
```
./build/glades_chiron_train --face 1 --kv-face 1 --face-beta-row 0.98 \
    --kv-face-beta-row 0.98 --bf16-adam --bf16-weights --bf16-grads
```

Expected compound compression:
- FACE: 1984× on embedding
- KV-FACE: 1000-2000× on Wk,Wv (if H36 holds)
- Combined attention + embedding Adam state: ~1000-2000× across both.

---

## 14. Summary + promote condition

KV-FACE is the natural extension of FACE to the next-largest unclaimed
Adam-state region: attention K,V projections. The mechanism is proven at
the embedding level; the hypothesis question reduces to a single
measurable property of trained attention (Gini of per-position popularity).

**Promote condition (Gate-0 probe):**
- On any trained 500M+ checkpoint, measure per-layer-per-head
  `p[t] = (1/T)·Σ_q A[q,t]` and compute the Gini coefficient G.
- **Causal-mask structural baseline (iter 121 finding).** Causal masking
  alone creates popularity skew with Gini = **0.500 ± 0.002 at any T ≥ 64**
  (computed from `p[t] = (1/T)·Σ_{q≥t} 1/(q+1)`). This is the floor
  below which the trained attention cannot go (structural lower bound).
  Table of measured baselines:

  | T    | Uniform-causal Gini | True-uniform Gini |
  |------|:-------------------:|:------------------:|
  |   32 | 0.4844              | 0.0000             |
  |   64 | 0.4922              | 0.0000             |
  |  128 | 0.4961              | 0.0000             |
  |  256 | 0.4980              | 0.0000             |
  |  512 | 0.4990              | 0.0000             |
  | 1024 | 0.4995              | 0.0000             |
  | 2048 | 0.4998              | 0.0000             |

- **Pass (memory):** any measured G confirms compression works (column
  norms are non-trivial for any G > 0.3).
- **Pass (convergence):** differential `G_trained − 0.500 ≥ 0.05`
  → learned attention adds useful structure; proceed to Phase 1
  expecting full convergence benefit.
- **Marginal:** `G_trained − 0.500 ∈ [0, 0.05]` → mechanism may
  reduce to Adafactor on the convergence axis; pursue memory-only
  variant.
- **Fail:** `G_trained < 0.50` → impossible under causal mask; indicates
  a measurement error or non-standard attention mechanism.

**Infrastructure for the probe (shipped iter 121):**
- `gpu_kvface_probe.{h,cu}` primitives: `kvface_probe_compute_popularity`,
  `kvface_probe_gini_device`, `kvface_probe_reduce_stats`,
  `kvface_probe_gini_host` (host reference).
- Unit test `CHIRONKvfaceProbeParityTest` validates kernel bit-parity
  against host reference (pop_err 2.98e-07, gini_err 2.98e-08) and
  confirms Zipf vs uniform-causal separation.
- `chiron_train --probe-attn-gini` flag: measures last-layer Gini
  every `--log-every` steps, logs mean/min/max per nHeads.

**Preliminary Gate-0 result (iter 121):**
41M × 2500 steps, T=1024, L=12, nH=8, tiled attention:

| Step | Loss (EMA) | Gini mean | Gini min | Gini max | Valid |
|-----:|:----------:|:---------:|:--------:|:--------:|:-----:|
|    0 | 10.400     | 0.4995    | 0.4994   | 0.4998   | 8/8   |
|  250 | 9.712      | 0.4995    | 0.4995   | 0.4995   | 8/8   |
|  500 | 9.550      | 0.4995    | 0.4995   | 0.4995   | 8/8   |
| 1000 | 9.252      | 0.4995    | 0.4995   | 0.4995   | 8/8   |
| 2500 | 8.775      | 0.4995    | 0.4995   | 0.4996   | 8/8   |

**Interpretation.** The LAST layer's per-head Gini stays identically
at the causal-mask baseline (0.4995) over a 1.6-nat loss improvement.
This is a MARGINAL Gate-0 result:
- Memory compression mechanism: triggered (any G > 0 suffices) ✓
- Convergence mechanism: premise not validated — learned Zipfian
  attention does NOT emerge at this scale in the LAST layer ✗

**Caveats.** (1) Only the last layer probed. Induction heads
typically emerge in EARLIER/MIDDLE layers; a per-layer probe may
reveal heterogeneous structure. (2) 41M is small. At 500M+ scale
the last-layer attention may develop richer structure. (3) 2500 steps
is short-horizon; long-horizon 10k+ steps may show divergence.

**Revised path forward.** Expand the probe to measure ALL layers
(not just last). If any layer shows G > 0.55, KV-FACE's convergence
premise holds for at least that layer. If all layers stay at 0.500,
pursue the memory-only variant of KV-FACE (compression without
convergence claim).

If passed Gate-0, 3-gate validation follows:
1. Primitive parity (≤ 1e-4 error vs host)
2. Trainer smoke (100+ steps no divergence)
3. Multi-scale long-horizon (66M, 234M, 500M × 2500 steps, validated Δ vs MFIO-only baseline)

Success criterion for paradigm-shift status:
- Memory: ≥ 500× compression on Wk,Wv Adam state (vs dense Adam)
- Convergence: ≥ 0.15 nat sustained advantage at 500M × 2500 steps
- Throughput: ≤ 0.5% overhead

If both criteria clear, KV-FACE becomes paradigm shift #36 "disrupting"
alongside FACE, extending the compression envelope from 1984× (embedding
only) to ~1000× across both embedding AND attention Adam state.
