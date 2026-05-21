# Paradigm Shift #36 Candidate B — KV-FACE (FACE extended from embedding to attention K, V projections)

**Status:** candidate design; one of three parallel proposals for shift #36.
**Date:** 2026-04-23 (post-FACE 1.4B validation, Ralph-loop iter 120).
**Axis:** optimizer-state memory + convergence for attention K/V weight matrices.
**Name:** **KV-FACE** — Frequency-Aware Column-normalized optimizer for attention K, V projections.
**Parent shift:** paradigm #28 FACE (on embedding). KV-FACE transplants the same mechanism onto Wk, Wv.

---

## 0. Elevator pitch

FACE's validated mechanism (unconditional column EMA + conditional row EMA + frequency-debiased column norm) was proven to exploit Zipfian token-frequency on the embedding `[V × m]`. KV-FACE asks: if attention-**position** popularity is Zipfian-like (some key-positions attended heavily, others sparsely), the same mechanism should compress Wk, Wv Adam state by 100-1000× with analogous convergence benefit. The empirical premise — is attention popularity Zipfian at trained state? — is the single Gate-0 question. Memory upside at 1.4B: ~2.5 GB → ~1 MB Adam state. Throughput cost: near-zero (same 3-kernel structure as FACE). Primary risk: learned attention may be uniform (maximum entropy), killing the Zipf premise; candidate provides a cheap 500M-checkpoint probe.

---

## 1. Primitive objects

The central question is **what axis of Wk, Wv carries a Zipfian frequency signal.** Wk has shape `[m × dModel]` where `dModel = nHeads · dHead`. Two axes exist:

- **Row axis (m)**: input d_model channels. Every channel is active every batch — NO sparsity, NO Zipf. Plain MFIO already handles this.
- **Column axis (dModel, partitioned into heads)**: each column is one slot of one head's key/value vector. Different head slots carry different linguistic features; empirically (from mechanistic interp work e.g. Elhage+22) some heads are "induction heads" dominating attention mass, others are near-dead. This IS a Zipfian-like axis.

But there's a THIRD, more powerful axis that the naive FACE transplant misses:

- **Sequence-position axis (T)**: after computing `K = X · Wk ∈ ℝ^{T × dModel}`, the rows of K are sequence-position keys. Each position `t` receives attention weight `Σ_q A[q, t]` from all queries. This column-sum of the attention matrix is empirically **highly non-uniform** (causal masking alone biases early positions). This is the ATTENTION-POPULARITY axis the hypothesis names.

The attention-popularity signal must therefore be routed BACK to the Wk, Wv gradient through the chain rule. Define:

- `A ∈ ℝ^{T × T}` — attention probability matrix (row-stochastic, causal-masked).
- `p[t] = (1/T) · Σ_q A[q, t]` — position `t`'s **attention popularity** (fraction of total attention mass it absorbs). Satisfies `Σ_t p[t] = 1`.
- `f_att ∈ ℝ^T` — the position-frequency vector analogous to token-frequency in FACE; `f_att[t] = p[t]`.
- `dK = dY · Wv^T` (where `dY` is attention output gradient) and `dV = A^T · dY`. Both have shape `[T × dModel]`.
- `dWk[i, j] = Σ_t X[t, i] · dK[t, j]`, `dWv[i, j] = Σ_t X[t, i] · dV[t, j]`.

The gradient `dWk` **inherits** the position-popularity weighting through `dK[t, j]`: unpopular positions (small `p[t]`) contribute proportionally smaller rows to the outer product. This is the Zipfian signal routed into the weight gradient.

- `zn_k[i] = Σ_j dWk[i, j]²` — per-input-channel squared norm.
- `dn_k[j] = Σ_i dWk[i, j]²` — per-output-column squared norm.
- `f̃_att[j] = Σ_t p[t] · ⟨dWk-contribution-of-position-t at column-j⟩` — the **induced column popularity** derived from attention weights. Specifically: `f̃_att[j] = Σ_t p[t] · (Σ_i X[t, i] · dK[t, j])` which rolls up to `Σ_t p[t] · (X^T · dK)[t, j]`.

In FACE's embedding case, rows are tokens and columns are m hidden-dims. Here the "token" analogue is a sequence-position's contribution; the "column popularity" is the attention-popularity-weighted projection.

## 2. State space

For each attention layer ℓ, maintain EMA state on Wk and Wv independently:

```
  S_k^{(ℓ)} = ( zn̄_k[m], dn̄_k[dModel], p̄[T_max], f̃_att_bar[dModel], gF̄_k )
  S_v^{(ℓ)} = ( zn̄_v[m], dn̄_v[dModel], p̄[T_max], f̃_att_bar[dModel], gF̄_v )
```

- `zn̄_k[m]` — row-EMA on Wk input-channel squared norms.
- `dn̄_k[dModel]` — column-EMA on Wk output squared norms (unconditional).
- `p̄[T_max]` — attention-popularity EMA across training (position-indexed, dim T_max = max seq len).
- `f̃_att_bar[dModel]` — EMA of the induced column popularity.
- `gF̄_k` — scalar EMA of `‖dWk‖_F²`.

Total state per Wk matrix: `m + dModel + T_max + dModel + 1` floats ≈ `m + 2·dModel + T_max` floats.

At 1.4B scale (m=2048, dModel=2048, T_max=1024): state per Wk ≈ 7169 floats = 28 KB.
Dense Adam on Wk: `2 · m · dModel` = 8.4M floats = 33.6 MB.
**Per-matrix compression: 33.6 MB / 28 KB ≈ 1200×.**

With 40 layers × 2 matrices (Wk + Wv): total KV-FACE state = 40 · 2 · 28 KB ≈ 2.2 MB.
Dense Adam on Wk+Wv across 40 layers: 40 · 2 · 33.6 MB ≈ **2.69 GB → 2.2 MB (~1200× compression)**.

## 3. Evolution law / update rule

Analogous to FACE's 3-phase cycle, specialized for KV position-popularity:

**Phase 1 (stats).** At the end of the backward pass, with `dWk, dWv, A, X` available:
```
  p[t]       = (1/T) · Σ_q A[q, t]                             (per layer, per head — averaged)
  zn_k[i]    = Σ_j dWk[i, j]²                                  (dWk row norms)
  dn_k_raw[j]= Σ_i dWk[i, j]²                                  (dWk col norms)
  f̃_att[j]  = Σ_t p[t] · (X[t, :] · dK[t, j])                 (popularity-weighted col stat)
  gF_k       = Σ_{i,j} dWk[i, j]²
```

`f̃_att` is a NEW primitive KV-FACE introduces; it weights columns by the attention-popularity of the positions that feed them. Computed as a single fused reduction; O(T·dModel) cost, cheaper than existing attention-weight reduction.

**Phase 2 (EMA update).** All EMAs are UNCONDITIONAL (Wk, Wv don't have the sparse-row structure of embeddings — every batch touches every row). Row EMA reduces to standard EMA:
```
  zn̄_k[i]   ← β_row · zn̄_k[i] + (1-β_row) · zn_k[i]
  dn̄_k[j]   ← β_col · dn̄_k[j] + (1-β_col) · dn_k_raw[j]
  p̄[t]      ← β_pop · p̄[t] + (1-β_pop) · p[t]                 (NEW — position popularity EMA)
  f̃_att_bar[j] ← β_col · f̃_att_bar[j] + (1-β_col) · f̃_att[j]
  gF̄_k      ← β_col · gF̄_k + (1-β_col) · gF_k
```

**Phase 3 (apply).** Preconditioner combines the FACE row·col structure with the popularity-debiased column norm:
```
  σ_{ij} = 1 / √( zn̄_k[i] · (dn̄_k[j] / f̃_att_bar[j]) / gF̄_k + ε² )
  Wk[i, j] ← Wk[i, j] − η · σ_{ij} · dWk[i, j]
```

The key design choice: divide `dn̄_k[j]` by `f̃_att_bar[j]` rather than by `q̂` (FACE's active-row count). The popularity-weighted normalization corrects for the fact that popular positions dominate `dn_k_raw`. This is the direct attention-axis analogue of FACE's token-frequency debiasing.

β defaults, scale-aware (inheriting FACE's recipe):
- `β_row = 0.98` (small models) … `0.98` (large).
- `β_col = 0.95`.
- `β_pop = 0.99` (popularity distribution changes slowly).

## 4. Mechanism mapping

| FACE (embedding)                     | KV-FACE (attention Wk, Wv)                    |
|--------------------------------------|-----------------------------------------------|
| Token i appears with frequency f_i   | Position t is attended with popularity p[t]   |
| Zipfian: rank-1 token absorbs ~α×total | Zipfian (hypothesized): popular positions absorb most mass |
| Rare rows → sparse gradients         | Unpopular positions → small contribution in dWk |
| Per-row cond EMA preserves staleness | Per-row uncond EMA (no sparsity for Wk rows) |
| Col norm / q gives mean per active row | Col norm / f̃_att gives popularity-debiased mean |
| σ_{ij} frequency-independent LR       | σ_{ij} popularity-independent LR              |
| **Effect: rare tokens get same LR**   | **Effect: rare-attended positions' gradients get same LR**

Why this should help convergence (if the Zipf premise holds):

1. Under dense Adam, the column `v[:, j]` of Wk-Adam is dominated by gradients from popular positions. Unpopular-position gradients' variance is under-estimated → their `1/√v` is too small → they learn too slowly.
2. This exactly parallels the rare-token bias in embedding Adam that FACE repairs.
3. KV-FACE's `σ_{ij}` replaces `1/√v` with a popularity-invariant form. Unpopular-position contributions are properly amplified when they do arise.

The nontrivial difference: FACE's popularity signal (token frequency) is **persistent** across training (Zipf law doesn't drift). Attention popularity `p[t]` is **dynamic**: early in training it's near-uniform (random attention); late in training it concentrates (learned induction heads, early-token bias, etc.). The `β_pop` EMA tracks this drift but only correctly if the concentration happens — if the network learns uniform attention, Zipf never develops and KV-FACE degenerates to standard row-col Adafactor.

## 5. Formal analogy to FACE with key differences

| Property                      | FACE (paradigm #28)              | KV-FACE (this candidate)                       |
|-------------------------------|-----------------------------------|------------------------------------------------|
| Target matrix                 | Embedding E `[V × m]`            | Wk, Wv `[m × dModel]` per layer                |
| Sparsity axis                 | V (tokens) — strong Zipf         | T (positions) — hypothesized Zipf              |
| Sparsity type                 | Per-batch (most V unused)        | Always-on (every batch sees all positions)     |
| Row EMA type                  | Conditional on activity          | Unconditional                                   |
| Popularity signal             | Token count (batch → lifetime)   | Attention column-sum Σ_q A[q, t] (batch-dynamic) |
| Popularity persistence        | Stable (Zipf law)                 | Drift: uniform → concentrated during training  |
| Debiasing denominator         | `q̂` (active-row count)          | `f̃_att_bar` (popularity-weighted column stat)  |
| State size                    | 2V + m + 2                        | m + 2·dModel + T_max + 1 per matrix             |
| Composition                   | Orthogonal to MFIO, WIP           | DIRECT OVERLAP with MFIO on Wk/Wv (§9)         |
| Validation corpus mattered    | Yes (uniform kills it, iter 76)  | Predicted: yes — early-training uniform kills it too |

Key difference to highlight: **FACE's popularity signal is static (Zipf is a corpus property); KV-FACE's popularity signal is emergent (attention shape is a learned property).** This is the mechanism-level bet — does attention concentrate enough to provide useful signal, and does it concentrate soon enough to matter for the optimization phase that KV-FACE operates in?

## 6. Compression calculation at 1B, 1.4B, 2.23B scales

| Scale   | m    | dModel | L  | T_max | Wk size (floats) | Dense Adam on Wk+Wv (all layers) | KV-FACE state | Compression |
|--------:|:----:|:------:|:--:|:-----:|:----------------:|:--------------------------------:|:-------------:|:-----------:|
| 234M    | 1024 | 1024   | 24 | 1024  | 1.05M            | 100 MB                           | 256 KB        | ~390×       |
| 500M    | 1536 | 1536   | 24 | 1024  | 2.36M            | 227 MB                           | 348 KB        | ~650×       |
| 1B      | 1536 | 1536   | 48 | 1024  | 2.36M            | 453 MB                           | 696 KB        | ~650×       |
| 1.25B   | 1792 | 1792   | 44 | 1024  | 3.21M            | 564 MB                           | 760 KB        | ~740×       |
| **1.4B**| 2048 | 2048   | 40 | 1024  | 4.19M            | **2.69 GB** (with FP32)          | **~2.2 MB**   | **~1200×**  |
| 2.23B   | 2560 | 2560   | 48 | 2048  | 6.55M            | 5.03 GB                          | 4.0 MB        | ~1280×      |

At 1.4B the 2.69 GB figure assumes FP32 Adam. With `--bf16-adam` already enabled in CHIRON (halves state), the raw memory saved is ~1.35 GB, still 600× compression. The scaling matches FACE: compression grows linearly with d_model.

**Upper-bound case with BF16 Adam at 2.23B:**
- Dense: 5.03 GB Adam state on attention Wk+Wv with FP32 → 2.5 GB with BF16.
- KV-FACE: 4 MB.
- Net saved: ~2.5 GB — fits the "magnitudes less memory" brief if the mechanism works.

## 7. Stability / expressivity

Three stability levers:

**7.1 ε-floor on preconditioner denominator.** Early in training or on low-popularity positions, `f̃_att_bar[j]` may be near zero. Without the ε² in `σ = 1/√(· + ε²)`, σ blows up, replicating surprise #10 (iter 71). ε = 1e-8 (FACE default) should suffice; needs empirical verification for KV-FACE's denominator structure.

**7.2 Warmup window.** Inherit FACE's 100-step warmup: use dense Adam on Wk, Wv for the first 100 steps to let `p̄`, `f̃_att_bar`, and `zn̄, dn̄` stabilize before handoff to the preconditioner. Without warmup, step-1 σ is numerically catastrophic.

**7.3 Causal-mask bias in `p`.** For causal-masked training, `p[t]` is strongly biased toward early positions (position 0 is attended by all T queries → p[0] ≈ 1/T; position T-1 is attended only by query T-1 → p[T-1] ≈ 1/T² effectively). This is a SYSTEMATIC bias independent of learned attention patterns — and KV-FACE must avoid confusing it with the Zipfian-mechanism signal. Mitigation: divide `p[t]` by the number of queries that can attend to t (= T - t for causal) before forming `f̃_att`, isolating the learned-pattern component.

**Expressivity.** KV-FACE cannot express a preconditioner that is not of outer-product form `row × col / popularity`. Full per-weight Adam's v has `m · dModel` degrees of freedom; KV-FACE has `m + dModel + T_max` — a dimension reduction of `(m · dModel) / (m + dModel + T_max) ≈ m/3` at m=dModel, matching Adafactor's compression theorem. Any pathology where the TRUE optimal preconditioner requires interaction between specific (row, col) pairs (e.g., low-rank correlated noise) cannot be captured.

## 8. Throughput

FACE's per-step overhead is ~0% (iter 93 at 500M: 3932 vs 3933 tok/s). KV-FACE's cost structure:

- Phase 1 stats: 3 fused reductions per Wk matrix (same as FACE's 3 kernels). Extra: `p[t]` from attention matrix (one column-sum over `A ∈ ℝ^{T×T}`, already materialized during forward). `f̃_att` reduction is `O(T · dModel)`, subsumed in the existing `dK` reduction cost.
- Phase 2 EMA: 4 elementwise kernels, trivially fast.
- Phase 3 apply: single elementwise kernel identical to `face_apply_preconditioned_update`.

Per-step cost: `~2 μs × (L layers) × (2 matrices)` — at 40 layers, ~160 μs added per step. Step time at 1.4B is ~600 ms → **<0.03% overhead**.

**Predicted throughput: identical to dense Adam within noise** (matches FACE empirical result).

## 9. Composition with MFIO (potential conflict)

MFIO v2 mode 2 (`--mfio 2`) currently applies Adafactor-style row/col preconditioning to Wq, Wk, Wv in the trainer. KV-FACE also targets Wk, Wv. Analysis:

- **MFIO v2 formula**: `σ_{ij} = 1/√(zn[i] · dn[j] · β / (T · T) + ε²)` — row/col product ONLY, no popularity term.
- **KV-FACE formula**: `σ_{ij} = 1/√(zn̄_k[i] · dn̄_k[j] / (f̃_att_bar[j] · gF̄_k) + ε²)` — row × popularity-debiased col.

These are **not compatible** as written; both overwrite Adam-v on the same weight matrix. Three composition options:

1. **Replacement**: KV-FACE REPLACES MFIO on Wk, Wv. MFIO continues to handle Wq (no popularity mechanism needed; queries generate attention, they don't receive it). New flag: `--mfio 3` = MFIO on Wq + KV-FACE on Wk, Wv. This is the recommended mode if KV-FACE validates.
2. **Blended preconditioner**: `σ = σ_MFIO · (dn̄_k[j] / f̃_att_bar[j])^γ` with γ ∈ [0, 1] a mixing parameter. Technically sound but adds hyperparameter.
3. **Rejection if MFIO already wins**: if MFIO v2's existing memory reduction on Wk, Wv is already "enough" (it's 2000× as noted in gpu_mfio.h), KV-FACE must justify its extra machinery via convergence gain, not memory.

Key insight from FACE's history (iter 65): MFIO on embedding DEGRADED loss by +1 nat despite compressing state 1008×. FACE's popularity-debiasing was what rescued it. If attention has analogous Zipfian structure, the MFIO→KV-FACE transition should show the same repair effect on Wk, Wv. If attention is near-uniform, MFIO alone is already the right answer and KV-FACE is redundant.

## 10. Composition with FACE (orthogonal axis)

FACE targets Embedding E [V × m]; KV-FACE targets Wk, Wv [m × dModel]. Zero matrix overlap. Both share the gpu_face.cu primitive library (reduction + EMA + preconditioned-update), but operate on different tensors with different popularity signals.

**Expected compound**: `--face 1 --kv-face 1 --mfio 3` (MFIO on Wq only) could deliver:
- 1008-1984× compression on embedding (FACE).
- ~1200× compression on Wk, Wv across all layers (KV-FACE).
- ~2000× compression on Wq (MFIO v2).
- Combined embedding + attention Adam state: ~3-5 MB total across all ~4 B attention+embedding parameters at 2.23B scale.

**Convergence stacking**: FACE's 0.4-1.7 nat advantage is already validated. KV-FACE's *potential* convergence advantage (if Zipf holds) would stack multiplicatively on a different mechanism axis — no mutual interference. FACE fixes rare-TOKEN learning; KV-FACE fixes rare-POSITION learning. The Ralph-loop research philosophy of orthogonal-axis stacking (§5.1 of FACE disrupting paradigm doc) supports this composition directly.

## 11. Failure modes + mitigations

**F1: Attention is uniform at trained state.** If learned attention distributes mass evenly across positions (entropy maximization), `p[t] ≈ 1/T` for all t, `f̃_att_bar[j]` becomes approximately constant, and KV-FACE degenerates to plain row-col Adafactor (== MFIO v2). No harm, but no benefit — candidate fails the "magnitudes faster" brief.
**Mitigation**: Gate-0 probe BEFORE committing (§13). Cheap to test.

**F2: Attention is Zipfian EARLY but collapses LATE.** Induction heads form at ~step 500-2000, attention sharpens, then some heads may degenerate or attention re-broadens with very long training. If mechanism is transient, KV-FACE's advantage could reverse at long horizons. FACE saw advantage GROW with horizon (§3.5 up to 5000 steps); KV-FACE might saw-tooth.
**Mitigation**: Build in a dense-Adam fallback triggered by `max_j(f̃_att_bar[j]) / mean_j(f̃_att_bar[j]) < τ_uniform` (e.g., τ=2). If attention goes uniform, fall back to MFIO or dense. Easy runtime check.

**F3: Causal-mask bias dominates learned popularity.** If the only source of `p[t]` variance is positional masking, not learned attention, `p̄` reflects a trivial position prefix and `f̃_att_bar` is uninformative.
**Mitigation**: popularity-normalization by (T-t) for causal case, as §7.3. This isolates learned attention variance.

**F4: Per-head variation is lost.** KV-FACE reduces over the heads×dHead flat dimension; but per-head attention patterns can be wildly different (induction head vs positional head vs null head). A single `f̃_att_bar[j]` vector mixes these.
**Mitigation**: PER-HEAD KV-FACE variant — maintain `f̃_att_bar^h[dHead]` for each head h separately. Costs nHeads × dHead = dModel floats (same as current design, just organized per-head). Implementation: shape `f̃_att_bar` as `[nHeads × dHead]`, compute per-head popularities, apply per-head σ. This is recommended.

**F5: BF16-Adam interaction.** FACE has been validated with `--bf16-adam + --bf16-weights` at 1.4B. KV-FACE's `f̃_att_bar` and `p̄` EMAs are small enough (few KB per layer) to keep in FP32 without memory pressure.
**Mitigation**: Keep KV-FACE state in FP32; only the weight update output inherits the current weight dtype.

**F6: MFIO is already "close enough" to the optimum on Wk,Wv.** FACE only won over MFIO on embedding because sparse-per-row gradients broke MFIO's mean. Wk gradient is dense (every batch sees all rows). KV-FACE's distinct mechanism — popularity-weighted column norm — is only helpful if column-popularity is highly non-uniform. May not be.
**Mitigation**: Gate-0 probe (§13) measures column-popularity ratio directly.

## 12. Minimal prototype

Re-use `gpu_face.cu` primitives by treating Wk as a "virtual embedding" but with different popularity signal:

```cpp
// In sgd_transformer.cpp attention-block backward, after dWk is materialized:
//   1. Compute attention popularity p[t] from A[q,t] (fused reduce).
//   2. Compute f̃_att[j] = Σ_t p[t] * (X[t,:] · dK[t,j])   — one extra reduction.
//   3. Compute zn_k[i], dn_k_raw[j], gF_k via reused face_compute_sparse_stats
//      (but mark as non-sparse so q count is skipped).
//   4. EMA update via face_update_emas specialized for unconditional rows.
//   5. Apply via face_apply_preconditioned_update with f̃_att_bar substituted for q̂.
```

New GPU primitives to add (small):
- `gpu_kvface_compute_popularity(A, p_out, T, nHeads)` — column-sum attention matrix per head.
- `gpu_kvface_compute_weighted_col_stat(X, dK, p, f_att_out, T, m, dModel)` — fused reduction.
- Everything else reuses `gpu_face.*` with a flag for unconditional-row mode (trivial extension).

Host-side trainer changes (~50 lines):
- Allocate per-layer `p̄`, `f̃_att_bar`, `zn̄_k`, `dn̄_k`, `gF̄_k` buffers.
- Add `--kv-face 1` flag, mutually exclusive with `--mfio 1/2` on Wk, Wv (new mode `--mfio 3`).
- In `adam_step` Wk, Wv branch: call KV-FACE primitives instead of `adam_one`.

Estimated code footprint: 300 lines of CUDA + 200 lines of trainer integration. Half of FACE's footprint due to primitive reuse.

## 13. Gate-0 probe

**Single cheap probe to validate the Zipfian premise BEFORE implementing anything.** From a 500M CHIRON checkpoint (iter 93 baseline exists):

```
  # pseudo-script (50 lines of C++ using existing inference machinery):
  load 500M checkpoint
  run 1000 validation tokens through forward pass
  for each layer ℓ, for each head h:
    compute attention matrix A^{(ℓ,h)} [T × T]
    compute column-sum p^{(ℓ,h)}[t] = Σ_q A^{(ℓ,h)}[q, t]
    normalize p^{(ℓ,h)} so Σ_t = 1
    sort p^{(ℓ,h)} descending → p_sorted
    measure: top-10% mass (Σ of top T/10 p_sorted values)
              top-1% mass (Σ of top T/100 p_sorted values)
              Gini coefficient of p^{(ℓ,h)}
              ratio max(p)/median(p)
  report per-(layer, head) distribution of these stats
```

**Acceptance criteria:**
- Top-10% of positions absorb ≥60% of attention mass (Zipfian threshold). → GO
- Top-10% absorb 10-30% (flat). → FAIL; abandon KV-FACE candidate.
- Gini coefficient > 0.4 per head on average → STRONG GO.
- Gini coefficient < 0.2 → NO-GO.

**Probe cost**: one forward pass on 1000 tokens. At 500M, ~1 second on GPU. Zero training cost, zero risk, pure diagnostic.

**Expected outcomes (informed prior):**
- **Induction heads** (late layers) — highly concentrated (Gini > 0.7) → KV-FACE helps these heads.
- **Positional heads** (early layers) — concentrated but by causal mask, not learning → KV-FACE's (T-t)-normalization isolates the learned component.
- **Dead heads** — may be uniform; KV-FACE's popularity signal is moot there.
- **Average over all heads**: if ≥40% of heads show Gini > 0.4, candidate is viable.

**Cheaper alternative probe**: during any existing training run, instrument one backward step to dump p_sorted for all heads. Total overhead: <1 MB I/O + 1 step of bookkeeping. Can be run opportunistically.

## 14. Strengths vs weaknesses

**Strengths:**
- Extends a **mechanism-validated** paradigm (FACE's Zipf-regularization theorem) to a new tensor axis.
- Orthogonal to FACE on embedding, orthogonal to MFIO on Wq → clean compound recipe.
- Massive memory upside (~1200× on Wk,Wv Adam state at 1.4B+); largest single-tensor win outside embedding.
- Near-zero throughput cost (inherits FACE's cheap-reduction structure).
- Primitive reuse: 50% of implementation is `gpu_face.cu` specialized for unconditional rows.
- Gate-0 probe is cheap and decisive (1 second of GPU time, full risk mitigation before investment).
- Implementation cost low: ~500 lines vs FACE's ~1000. Can ship in 2-3 Ralph-loop iterations.
- Composes with bf16-Adam (KV-FACE state is small enough to stay FP32 without memory pressure).

**Weaknesses:**
- **Load-bearing empirical assumption** — Zipfian attention popularity at trained state. If false, candidate degenerates. FACE had this same risk (token Zipf) but that was a well-known corpus property; attention popularity is emergent and less predictable.
- **Dynamic popularity** — attention concentrates OVER TRAINING, so `p̄` must track non-stationary signal. FACE's token frequency is static across training; KV-FACE's is learned. Warmup + `β_pop` handles this but adds tuning surface.
- **Potential redundancy with MFIO v2** — MFIO v2 already compresses Wk,Wv Adam state by ~2000×. KV-FACE's additional compression (~1200 vs 2000) is comparable; the pitch must rest on CONVERGENCE advantage (like FACE's 0.4-1.7 nat), not memory. But that advantage is exactly what the Gate-0 probe tests.
- **Per-head vs flat-head decision** — designing for per-head adds implementation complexity (nHeads × dHead buffers, per-head reductions). Skipping it loses precision; including it doubles implementation effort.
- **Non-trivial composition with `--bf16-grads`** — the `f̃_att` reduction requires `X` and `dK` in FP32 (or careful mixed-precision); if both are BF16, precision of the popularity-weighted stat may suffer.
- **Causal-mask bias** — systematic non-uniformity from masking alone may dominate learned signal in early layers, requiring the (T-t) normalization which adds subtlety.
- **Mechanism validation corpus** — FACE was validated by a UNIFORM-corpus ablation (iter 76 §3.7). KV-FACE's analog would be a "uniform-attention" ablation, harder to construct synthetically (would need forced attention dropout / temperature-max regularizer to force uniform attention), more expensive to run.

---

## Summary for coordinating agent

1. **Claim:** KV-FACE applies FACE's row-EMA + frequency-debiased column-EMA preconditioner to attention Wk, Wv weight matrices, using per-position attention popularity `p[t] = (1/T) Σ_q A[q,t]` as the Zipfian signal analogue of token frequency.
2. **Key novel mechanism:** a popularity-weighted column statistic `f̃_att[j] = Σ_t p[t] · (X^T · dK)[t, j]` replaces FACE's active-row-count divisor, routing dynamic attention-concentration structure into the preconditioner via the chain rule; per-head variants isolate per-head attention patterns.
3. **Expected speedup + memory impact:** if attention-popularity is Zipfian at trained state (the single Gate-0 question), ~1200× compression on Wk,Wv Adam state (2.5-5 GB → 2-4 MB at 1.4-2.23B) with near-zero throughput cost and an expected ~0.2-1.0 nat convergence advantage stacked orthogonally with FACE; if attention is uniform, candidate degenerates to plain Adafactor and must be rejected — probe is 1 second of GPU time on an existing 500M checkpoint.
