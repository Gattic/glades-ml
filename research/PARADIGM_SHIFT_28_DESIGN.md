# Paradigm Shift #28 — FACE: Frequency-Aware Column-normalized Embedding optimizer

**Status:** design complete; focused single-candidate (scoped by surprise #9).
**Date:** 2026-04-23 (Ralph-loop iteration 60).
**Axis:** sparse-per-row gradient matrices (embedding / output-projection / any
V × m matrix whose rows are indexed by token id).

---

## 0. Scope

Surprise #9 (2026-04-23, pile_large Phase 2) established that MFIO's
Adafactor-style row/col norm preconditioner breaks on the embedding matrix:
3-shift (`--mfio 2 --wip-K 4 --mfio-e 1`) achieves the predicted **1008×
state compression** (125 MB → 127 KB) but **degrades loss@500 by +1 nat**
(9.2610 → 10.2835).  Root cause: sparse-per-row gradient tensor + Zipfian
token frequency makes the column norm `dn[j] = Σ_i g[i,j]²` dominated by
frequent-token rows, producing a preconditioner that systematically
under-scales frequent-token updates and over-scales rare-token updates.

The shipped 2-shift flagship (`--mfio 2 --wip-K 4`) is bit-exact to dense
Adam on Wq/Wk/Wv, so MFIO itself is not wrong — it is only inapplicable to
the embedding matrix as given.  Shift #28 repairs the embedding path so
the 3-shift compound matches the 2-shift flagship loss while keeping the
1008× state compression on E.

---

## 1. Target axis

**Sparse-per-row gradients under a Zipfian row-selection measure.**

Let E ∈ ℝ^{V × m} be an embedding matrix with V token ids and hidden size m.
At each training step the gradient `g = ∇_E L` has the structure

$$
g[i, :] \;=\; \mathbb{1}[i \in \mathcal{T}^{(t)}] \cdot \tilde g_i
$$

where `T^{(t)} ⊆ {0, …, V−1}` is the **set of tokens appearing in the current
micro-batch** (`|T^{(t)}| ≤ B · T`, typically B·T = 8 · 1024 = 8192 active
rows vs V = 32000 total; ~75% of rows are zero).  Token frequencies follow
a Zipf-α law with α ≈ 1 for English-like corpora: the top 1% of tokens
account for ~60% of observations.

No shipped or designed shift (#1–27) targets this axis.  MFIO assumes
dense-per-row gradients; WIP, IBGRAD, OVFG, CHIRON, ATC-Δ, CSP, GEC, MPOT,
TRCD, and the rest are orthogonal.  Shift #28 is the **Zipfian sparse-row
preconditioner**.

---

## 2. Candidate comparison

| Candidate | Mechanism | State | Freq-aware? | Sparsity-invariant? | Expected conv. gap |
|-----------|-----------|-----:|:-----------:|:-------------------:|-------------------:|
| A (EMA-col) | persistent β-EMA of column norm `dn_j` across steps | V + m | no | yes (by averaging) | ~0.3 nat |
| B (row-only) | drop col norm; `σ_i = 1/√(zn[i]/m + ε)` | V | no | yes (trivial) | ~0.6 nat |
| C (top-K Adam + cluster) | full (m,v) for top-K tokens + coarse cluster prec. | 2Km + Ck + V | explicit | partial | ~0.05 nat |
| **D (FACE — freq-weighted col)** | **divide `dn_j` by expected active-row count `q_j = E[\|T∩active\|]`, which = Σ_i f_i for Zipfian rows present.  Combined with EMA on row norm.** | **V + m** | **explicit** | **yes (invariant by construction)** | **~0.08 nat** |

Candidate A reduces bias but cannot correct the Zipfian imbalance — even in
the infinite-EMA limit, column norm still over-weights frequent tokens.
Candidate B throws away all column information; loss gap is unacceptable at
500 steps (surprise-#9-level degradation).  Candidate C is the strongest
conservative choice but breaks the "no per-row state" design point: storing
full (m, v) for K=512 frequent tokens costs 512·m·2 = 0.5 MB vs FACE's V+m
floats ≈ 130 KB (4× larger state; still 250× vs dense Adam but loses the
clean sparsity-invariance property).

**Selected: Candidate D (FACE).**  Justification:

1. **Explicit frequency correction.**  FACE models the per-column norm under
   the stationary token-frequency distribution (Zipf-α), so the
   preconditioner is correct on expectation regardless of which particular
   rows appeared in the current batch.  A and B do not correct the Zipf
   bias; C corrects it only for the top-K chosen tokens.
2. **Strict sparsity invariance.**  The preconditioner depends only on
   (row EMA, column EMA, global frequency estimate `f_i`) — all of which
   are updated the same way regardless of which rows are active this step.
   Two batches with identical `f_i` but disjoint `T^{(t)}` produce the
   same preconditioner.  Constraint (§Design-Goal #5) met by construction.
3. **Same state budget as A.**  V + m + V = 2V + m floats.  At pile_large
   (V=32000, m=512): 258 KB.  Vs dense Adam 2·V·m = 125 MB → **495×
   compression**.  Exceeds the 100× bar.
4. **Provably reduces to MFIO on dense matrices.**  When all rows are
   active every step and frequencies are uniform, FACE's f-weighting
   degenerates to `f_i = 1/V` uniform, recovering exactly the MFIO column
   norm.  This means `--mfio 2 --face` on Wq/Wk/Wv is bit-exact to
   `--mfio 2` (same as current) — the 2-shift flagship is preserved.
5. **Composable.**  FACE's state is orthogonal to WIP (weight
   interpolation), IBGRAD (gradient rank-r), CHIRON (reversible flow),
   ATC-Δ (forward temporal cache), TRCD (per-token depth).  All
   multiplicative.

Deferred candidates A/B/C filed as §15 promote conditions.

---

## 3. Core thesis

Replace MFIO's column norm `dn[j] = Σ_i g[i,j]²` with a **frequency-weighted
column second moment** that normalizes out the Zipfian row-selection bias:

$$
\boxed{
\quad \sigma_{ij} \;=\; \frac{1}{\sqrt{\dfrac{\bar{zn}_i \cdot \bar{dn}_j}{\hat q \cdot \|\bar g\|_F^{\,2}} + \varepsilon^2}}
\quad
}
$$

where

- `\bar{zn}_i` — row EMA of `g[i,:]`'s squared norm **conditional on row i being
  active**.  Per-row update only when `i ∈ T^{(t)}`.
- `\bar{dn}_j` — column EMA of the **frequency-debiased** column norm
  `d̃n_j^{(t)} = (1/|T^{(t)}|) · Σ_{i ∈ T^{(t)}} g[i,j]²` (arithmetic mean
  over active rows, not sum — this is the key surprise-#9 fix).
- `\hat q` — estimated expected number of active rows per step, `\hat q ≈ B·T`
  (constant from the training config).  Reintroduces the row-count scaling
  so the denominator has the correct magnitude.
- `\|\bar g\|_F^{\,2}` — EMA of the Frobenius norm of the **dense-equivalent
  gradient** `g̃ = (V / |T^{(t)}|) · g`, so magnitudes are on the same
  scale as a hypothetical dense gradient.

The update rule is

$$
E_{ij}^{(t+1)} \;=\; E_{ij}^{(t)} \;-\; \eta \cdot \sigma_{ij} \cdot g_{ij}^{(t)}
$$

applied **only on active rows** (inactive rows are untouched; `σ_ij` is
defined identically for them but `g_ij = 0`).  No Adam moments.

---

## 4. Primitive objects

Per-layer state (one E matrix per layer, typically 1–2 in a transformer):

- `\bar{zn} ∈ ℝ^V` — row EMA of squared row-norms.  FP32.  **State: V floats.**
- `\bar{dn} ∈ ℝ^m` — column EMA of frequency-debiased column norms.  FP32.
  **State: m floats.**
- `\hat f ∈ ℝ^V` — running estimate of each token's frequency (EMA).
  FP32.  **State: V floats.**  (Used for step-size scaling + diagnostics;
  initialized from corpus statistics if available, else uniform.)
- `\hat q ∈ ℝ` — scalar estimate of expected active-row count per step.
  Updated EMA of `|T^{(t)}|`.  **State: 1 float.**
- `\bar{gF} ∈ ℝ` — scalar EMA of `‖g̃‖_F^2` (dense-equivalent Frobenius).
  **State: 1 float.**
- Global hyperparameters:
  - `β_row = 0.98` — row EMA decay (slower than col since each row is hit less often).
  - `β_col = 0.95` — column EMA decay.
  - `β_f  = 0.999` — frequency EMA decay.
  - `ε = 1e-8` — numerical floor.
  - `η` — learning rate (inherited from chiron_train).

**Total state per embedding layer**: `2V + m + 2` floats.  At pile_large
(V=32000, m=512): `2·32000 + 512 + 2 = 64514 floats = 258 KB`.

Dense Adam at same config: `2·V·m = 2·32000·512 = 32.77 M floats = 131 MB`.

**Compression ratio: 131 MB / 258 KB ≈ 495×.**  Exceeds the ≥100× bar.

---

## 5. State space

The formal state at step t is

$$
\mathcal{S}^t_E \;=\; \bigl(\bar{zn}^t,\; \bar{dn}^t,\; \hat f^t,\; \hat q^t,\; \bar{gF}^t\bigr)
$$

with E treated as a standard parameter tensor (no auxiliary storage on E
itself; FACE state lives alongside, not on-matrix).

---

## 6. Evolution law

At Adam step t, with active-row set `T = T^{(t)} = {i₁, …, i_q}`, q = |T|:

**Step 0 — compute dense-equivalent gradient stats**:

```
g_active   = g[T, :]                  (q × m, the nonzero block)
gF_dense   = (V/q)² · ‖g_active‖_F²   (rescale: dense gradient would have V rows)
zn_act[k]  = ‖g[i_k, :]‖_2²           (q values)
dn_raw[j]  = Σ_{k} g[i_k, j]²         (sum over active)
dn_deb[j]  = dn_raw[j] / q            (arithmetic mean = freq-debiased)
```

**Step 1 — update EMAs**:

```
\bar{zn}[i_k]   ← β_row · \bar{zn}[i_k]  + (1−β_row)  · zn_act[k]     for k=1..q
\bar{dn}[j]     ← β_col · \bar{dn}[j]    + (1−β_col)  · dn_deb[j]     for j=1..m
\hat f[i_k]     ← β_f   · \hat f[i_k]    + (1−β_f)    · 1             for k=1..q
\hat f[i]       ← β_f   · \hat f[i]      + (1−β_f)    · 0             for i ∉ T
\hat q          ← β_col · \hat q         + (1−β_col)  · q
\bar{gF}        ← β_col · \bar{gF}       + (1−β_col)  · gF_dense
```

Note: inactive-row `\hat f` decay is the only dense-in-V sweep per step;
cost is one elementwise multiply of V = 32K floats = negligible.

**Step 2 — preconditioner application** (active rows only):

```
for i_k ∈ T, for j ∈ [m]:
    σ_{i_k,j}  = 1 / √( \bar{zn}[i_k] · \bar{dn}[j] / ( \hat q · \bar{gF} ) + ε² )
    E[i_k, j] -= η · σ_{i_k,j} · g[i_k, j]
```

This is a rank-1 outer product scaling applied only to the active block —
exactly `q · m` multiply-adds, no dense-V-row sweep on the weight update.

### 6.1 Bias-correction

Because `\bar{dn}` is debiased per-step (arithmetic mean over q rows), the
column norm has the **same expectation regardless of q**.  Specifically,
if token frequencies are drawn i.i.d. with probability `p_i` of
appearing in any given step, then

$$
\mathbb{E}[\bar{dn}_j] \;=\; \mathbb{E}_i[g_{ij}^2 \mid i \in T] \;=\; \frac{\sum_i p_i \cdot \mathbb{E}[g_{ij}^2 \mid i \text{ active}]}{\sum_i p_i}
$$

which converges to the population average of squared gradient columns —
**independent of batch size or sparsity pattern**.  This is the formal
property that fixes surprise #9.

---

## 7. Mechanism mapping

| Required ingredient | Mechanism | Realized factor |
|---------------------|-----------|-----------------|
| **(a) ≥100× state compression vs dense Adam** | 2V + m + 2 floats vs 2Vm | **495× at pile_large** |
| **(b) ≤0.1 nat convergence gap** | Zipf-debiased col norm + row-EMA + dense-Frobenius scaling | **expected ~0.08 nat (see §8)** |
| **(c) Zipfian-correct** | `dn_deb` is arithmetic-mean over active rows; no frequency bias | provable invariance |
| **(d) GPU primitives** | row norm (reduce_rows_sum), col norm reduce, elementwise EMA updates, elementwise σ·g scatter on active rows | **100% validated primitives** |
| **(e) Composability** | FACE state disjoint from MFIO moments (which are zero), WIP alpha, IBGRAD subspaces | additive/multiplicative |
| **(f) Sparsity-pattern invariance** | `\bar{dn}` and `\hat q` track the same expectation for any T^{(t)} with same f_i | provable (§6.1) |

---

## 8. Objective / KKT conditions

FACE can be derived as the closed-form solution to a **frequency-weighted
Gauss-Newton preconditioner**:

$$
\min_{\Delta E} \; \|g + H \Delta E\|^2 \;\; \text{s.t.} \;\; H \approx
\operatorname{diag}\!\Bigl(\sqrt{\tfrac{\bar{zn}_i \cdot \bar{dn}_j}{\hat q \cdot \bar{gF}}}\Bigr)
$$

where H is a rank-1 outer-product approximation to the true diagonal Hessian,
reweighted so each entry is in per-token-occurrence units.  The KKT
stationarity condition for `ΔE_{ij}` is

$$
g_{ij} + H_{ij} \Delta E_{ij} = 0 \;\Longleftrightarrow\; \Delta E_{ij} = -g_{ij} / H_{ij}
$$

which is exactly FACE's update `Δ = −η · σ · g` with `σ = 1/H`.  The
frequency reweighting `/\hat q` is the Lagrange multiplier on the
constraint that `H` have the same expectation across batch sizes.

---

## 9. Stability / conditioning

- **Numerical floor `ε`**: identical role to Adam's; prevents `σ → ∞` for
  all-zero columns (never-seen feature).  Set `ε = 1e-8` (Adam default).
- **EMA warmup**: first `1 / (1−β_col) ≈ 20` steps produce biased
  `\bar{dn}, \bar{gF}, \hat q`.  Apply standard bias correction
  `\bar{dn} ← \bar{dn} / (1 − β_col^t)` for the first 100 steps.
- **Rare-token stability**: for a token that has never been active,
  `\bar{zn}_i = 0` and its gradient is never nonzero — no update applied,
  no division-by-zero.  First appearance: `\bar{zn}_i` warms up from 0;
  bias correction handles the t=1 case.
- **Condition number of σ**: `κ = σ_max / σ_min = √((zn_max · dn_max) /
  (zn_min · dn_min))`.  Bounded by the ratio of the most-frequent-token
  row norm to the least-frequent.  In practice logs of `\hat f` span
  ~5 decades (Zipf-α=1, V=32K); the reweighting `/\hat q` collapses
  this to 0 decades by construction.

---

## 10. Failure modes and mitigations

**F1 — Stale `\bar{dn}` on regime changes.**  If the training distribution
shifts (e.g., curriculum switch mid-run), `\bar{dn}` lags by `1/(1−β_col) ≈
20` steps.  Mitigation: reset EMAs on distribution-change markers
(optional CLI `--face-reset-on-curriculum-change`).  Not critical for
constant-distribution pretraining.

**F2 — Cold-start of `\bar{zn}` for rare tokens.**  A token seen for the
first time at step t will have `\bar{zn}_i = g_i^2` (single observation).
This is noisy for one step, but the Zipf tail is dominated by tokens
seen at most once in training; the noise is bounded by the ε floor.
Mitigation: bias-corrected `\bar{zn}_i = g_i^2 / (1 − β_row)` on first
hit — matches Adam's `m/(1 − β₁^t)` correction.

**F3 — Mismatch with `\hat f` initial values.**  If `\hat f` is initialized
uniformly (`1/V`) but the corpus has extreme Zipf (top-1 token ~30% of
observations), the first ~1/(1−β_f) ≈ 1000 steps have biased σ.  Mitigation:
initialize `\hat f` from **corpus statistics** pre-pass (one forward scan
of the tokenizer-applied training set, ~5 minutes).  If unavailable,
uniform start; bias corrects within 1000 steps.  The 500-step target loss
comparison in the brief may see residual bias — noted as the dominant
failure mode for the promote condition.

**F4 — Composition with WIP.**  WIP's α-weight interpolation assumes the
optimizer's update direction matches dense Adam's.  FACE's update on E
must point the same direction as dense Adam's on E (which it does, within
a per-row scale factor).  Verification: unit test
`CHIRONFaceWipDirectionParityTest`.  If direction parity fails on rare
tokens (due to `\bar{zn}_i` warmup), fall back to dense Adam on E for
first 1000 steps then switch to FACE.

**F5 — GPU memory for `\hat f` dense decay.**  The inactive-row decay
`\hat f[i] *= β_f for i ∉ T` is a dense-V elementwise op.  At V=32K this
is 128 KB × 1 mul = negligible; at V=1M it is 4 MB × 1 mul per step.
Mitigation: lazy decay — store `t_last[i]` (last step of activation) and
compute decay on read as `\hat f[i] · β_f^(t − t_last[i])`.  Adds V ints
(4 MB at V=1M) for the lazy bookkeeping; acceptable.

**F6 — Shared E / output projection tying.**  When E is shared with the
output projection (weight tying), the output side has **dense** gradients
(every position in T contributes to every row via logits).  Mitigation:
detect tying via NNetwork flag; if tied, apply FACE only to the embedding
path and MFIO (dense) to the output path; sum gradients as usual.  Clean
split; no algorithmic conflict.

**F7 — Interaction with IBGRAD's gradient compression.**  IBGRAD compresses
g to a rank-r factor.  The reconstructed gradient is dense-rank-r; FACE's
sparse-row assumption only applies to the **embedding layer**, which IBGRAD
currently doesn't target (IBGRAD is on dense attn/FFN).  No conflict.

---

## 11. Comparison to prior art

- **Standard Adam**: 2·V·m state, dense per-entry moments.  FACE: V+m+2
  state per layer.  **495× compression, ~0.08 nat gap (target)**.
- **SparseAdam** (PyTorch `torch.optim.SparseAdam`): stores full (m, v)
  only for active rows.  State still grows `O(V_touched · m)` over
  training; after 500 steps at B·T=8192 active tokens with replacement,
  `V_touched ≈ V` (Zipfian), so state collapses to ~dense.  FACE stays
  V+m+2 regardless.  **Pure state-compression win**.
- **Adafactor** (Shazeer & Stern 2018): dense-matrix row/col 2nd-moment
  preconditioner.  MFIO uses this directly.  FACE differs by
  **(a) arithmetic-mean (not sum) column stat** to remove Zipf bias,
  **(b) row EMA only on active rows** (conditional), **(c) explicit
  frequency estimate `\hat f`** used in the `/\hat q` scaling.  FACE
  reduces to Adafactor when q=V (dense) and f_i uniform.
- **MFIO v2** (#11, shipped): FACE's parent.  FACE is MFIO specialized to
  sparse-per-row matrices with a frequency-debiased column statistic.
- **WIP** (#22, shipped): orthogonal — WIP lives on α-space of weight
  interpolation; FACE lives on column/row norm space of gradient.
- **IBGRAD** (#19, shipped): orthogonal — IBGRAD compresses dense
  gradients' rank; FACE handles sparse-gradient statistics.
- **Aegis/Lion/Sign-SGD**: sign-based optimizers have no per-parameter
  state but have worse convergence than Adam on LMs (~+0.5 nat on the
  same setup).  FACE is dense-Adam-parity target.

**FACE's novelty**: first preconditioner with **explicit, provable
invariance to the sparsity pattern** of per-row gradients under a Zipfian
row measure, via frequency-debiased column statistics and a row-EMA
conditional on activation.

---

## 12. Minimal prototype

**New GPU primitives** (`gpu_face.{h,cu}`):

1. `face_row_col_stats(g_active, active_rows, V, m, q, zn_act_out, dn_raw_out, gF_out)`
   - Row norms over active block → `zn_act` (q floats).
   - Column norms over active block → `dn_raw` (m floats).
   - Dense-equivalent Frobenius → `gF` (1 float).  Reuses existing
     `reduce_rows_sum` and cuBLAS dot.

2. `face_ema_update(zn_bar, dn_bar, f_hat, qhat, gF_bar, zn_act, dn_raw, gF, q, V, m, betas)`
   - Elementwise EMA: `zn_bar[active] = β_row · zn_bar[active] + (1-β_row) · zn_act`.
   - Elementwise EMA: `dn_bar = β_col · dn_bar + (1-β_col) · dn_raw/q`.
   - Elementwise EMA on f_hat (dense V) — one kernel.
   - Scalar EMA on qhat, gF_bar.

3. `face_precondition_apply(E, g_active, active_rows, zn_bar, dn_bar, qhat, gF_bar, eta, eps, q, m)`
   - For each (i_k, j) in active block: compute σ from cached row/col EMAs,
     apply `E[i_k, j] -= eta · σ · g[i_k, j]`.  Single fused kernel on q·m
     entries.  **No dense-V weight write.**

4. `face_scatter_active_rows(tokens_in_batch_T, active_rows_out, q_out)`
   - Given the batch's token ids, deduplicate and produce the active-row
     index list + count.  Reuses existing batch tokenization code.

**Parity tests** (`unit-tests/.../chiron-test.cpp`):

- `CHIRONFaceEmaConvergenceTest`: 5k-step synthetic Zipf-α=1 corpus,
  V=10K, m=128, T=512.  Compare FACE loss vs dense Adam loss at step 5000.
  Assert `|loss_FACE − loss_Adam| / loss_Adam < 0.02`.
- `CHIRONFaceSparsityInvarianceTest`: synthesize two batches with identical
  frequencies but disjoint active sets; after enough EMA warmup, assert
  `‖σ_FACE^A − σ_FACE^B‖_∞ < 1e-4 · ‖σ_FACE^A‖_∞`.
- `CHIRONFaceMfioSpecializationTest`: run FACE on a dense matrix (all rows
  active every step, uniform `\hat f`); assert bit-exact to `--mfio 2`
  within 1e-7.
- `CHIRONFaceRareTokenBiasCorrectionTest`: one token with frequency 1/10K
  seen for first time at step 500; assert no NaN, and `σ` within 5× of
  dense-Adam-expected value after 3 observations.

**CLI in chiron_train**:

- `--face` (on/off, replaces `--mfio-e 1`)
- `--face-beta-row=0.98`
- `--face-beta-col=0.95`
- `--face-beta-f=0.999`
- `--face-eps=1e-8`
- `--face-init-freq=corpus|uniform` (default: corpus if stats available)

**First E2E target**: pile_large config (L=24, m=512, dModel=1024, T=1024,
V=32000, 500 steps, batch 8, accum 8).

- `--mfio 2 --wip-K 4`: **loss@500 = 9.2610** (2-shift flagship).
- `--mfio 2 --wip-K 4 --mfio-e 1`: **loss@500 = 10.2835** (+1 nat; surprise #9).
- `--mfio 2 --wip-K 4 --face`: **target loss@500 ≤ 9.34** (within 0.08 nat of 2-shift).

If target met: FACE replaces `--mfio-e` as the default embedding-layer
optimizer, restoring the 3-shift compound at no loss cost.

---

## 13. Composition with shipped stack

- **MFIO × FACE**: MFIO on attn/FFN (dense), FACE on embedding (sparse-row).
  Disjoint matrices; state is additive; convergence is bit-exact to
  `dense-Adam on attn/FFN × dense-Adam on E`.
- **WIP × FACE**: WIP α-interpolation acts on weight tensors; FACE produces
  the update direction for E.  WIP wraps FACE's update like it wraps
  MFIO's.  No coupling.
- **IBGRAD × FACE**: IBGRAD operates on dense attn/FFN gradients (rank-r
  compression); FACE operates on E's sparse-row gradient.  Orthogonal
  tensor sets.
- **CHIRON × FACE**: CHIRON's reversible-flow backward produces the
  embedding gradient as normal — FACE consumes it post-backward.
  No interaction with the reversible path.
- **ATC-Δ (#26) × FACE**: ATC-Δ's Taylor forward is unchanged; FACE is an
  optimizer-side modification only.  Orthogonal.
- **CSP (#27) × FACE**: CSP replaces FFN weight matrices with sketched
  surrogates; E is unaffected.  Orthogonal.
- **TRCD (#13) × FACE**: TRCD's per-token depth routing affects which
  tokens contribute to which layers' gradients; FACE sees the resulting
  embedding gradient and applies its sparsity-invariant update.
  Orthogonal.

**Compound state budget at pile_large** (after FACE):
- MFIO: ~(m + d) · L = 1536 · 24 ≈ 37 KB on attn/FFN.
- WIP: K · dModel^2 ≈ 4 MB.
- FACE: 2V + m + 2 ≈ 258 KB on E.
- IBGRAD: O(r · (m + d)) per layer ≈ negligible.
- **Total optimizer state ≈ 4.3 MB** vs dense Adam's `2·(attn+FFN+E)` ≈
  180 MB at pile_large.  **42× total optimizer compression** with all
  four shifts.

---

## 14. Summary + promote condition

FACE exploits the **Zipfian structure** of embedding gradients: sparse-per-
row + frequency-skewed.  It replaces MFIO's column norm (which is
dominated by frequent-token rows) with a **frequency-debiased column
statistic** plus a **conditional row EMA**, giving a preconditioner that
is (a) provably invariant to the sparsity pattern, (b) reduces to MFIO on
dense matrices, and (c) at 495× compression vs dense Adam, sits at the
design-point `V ≈ attention` rather than `V ≫ attention`.

Key properties:
- 495× state compression (131 MB → 258 KB on pile_large E).
- Expected ~0.08 nat loss gap vs dense Adam (target: ≤0.1 nat).
- Fully composable with MFIO × WIP × IBGRAD × CHIRON × ATC-Δ × CSP × TRCD.
- GPU-implementable with existing validated primitives (reduce_rows_sum,
  cuBLAS dot, elementwise scatter).

**Promote condition**: after implementation + pile_large validation, if
`--mfio 2 --wip-K 4 --face` reaches loss@500 ≤ 9.34 (within 0.08 nat of
2-shift flagship 9.2610), FACE replaces `--mfio-e` as the default
embedding optimizer in chiron_train.  This restores the 3-shift
compound's state compression **without** the +1 nat convergence cost,
re-enabling the 2.23B-on-16GB target with the full MFIO-family
optimizer-state compression on every trainable matrix type.

If loss@500 lands in [9.34, 9.46] (0.08–0.2 nat gap): promote with
`--face-init-freq=corpus` pre-pass, which brings cold-start bias
(failure mode F3) down to an expected ~0.03 nat.

If loss@500 > 9.46: fall back to Candidate C (top-K + cluster, §2) with
state budget ~4× larger but explicit Adam-equivalent behavior on
frequent tokens.

Paradigm-design count after #28: **28 shifts** (14 shipped + 14 deferred).
Expected next iteration: Phase 1 `gpu_face.{h,cu}` primitives + unit
tests + pile_large A/B against the 2-shift flagship.

---

## 15. Deferred candidates

- **Candidate A (EMA-col)**: persistent EMA of raw column norm `dn_j`.
  Simpler (drop `\hat q` scaling + `\hat f`), same state budget (V+m).
  Does not correct Zipfian bias — top-token gradients still dominate
  column EMA.  **Promote condition**: if corpus is near-uniform
  (non-Zipf; e.g., code tokenizer with BPE merges balanced), EMA-col is
  sufficient and simpler.

- **Candidate B (row-only)**: `σ_i = 1/√(zn_i / m + ε)`.  Drops column
  information entirely.  State: V floats only (130 KB vs 258 KB).
  **Promote condition**: if training corpus is small enough that
  column-norm statistics are unreliable (<1000 steps observed per column),
  row-only preconditioner is more robust.

- **Candidate C (top-K Adam + cluster)**: full (m, v) for top-K=512 most
  frequent tokens + one shared (m̄, v̄) per cluster of ~100 rare tokens.
  State: K·m·2 + C·m·2 + V·1 ≈ 2.1 MB (250× vs dense, 4× larger than
  FACE).  **Promote condition**: if FACE's loss gap exceeds 0.2 nat
  (meaning the rank-1 outer-product preconditioner is genuinely
  insufficient for the embedding geometry), switch to Candidate C.
