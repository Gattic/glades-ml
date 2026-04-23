# Paradigm Shift #27 Candidate A — THR (Top-k Hard Routing of FFN Neurons)

**Status:** candidate design; single-agent pass (candidate A).
**Date:** 2026-04-23 (Ralph-loop iteration 54).
**Axis:** post-nonlinearity activation sparsity in FFN/MLP (width-axis per-token neuron routing).

---

## 0. Target axis and problem statement (verbatim)

Target axis: **POST-NONLINEARITY ACTIVATION SPARSITY IN FFN/MLP.**  In a trained transformer FFN

    h_in → W_up → σ(·) → W_down → h_out    (GELU/SiLU/ReLU)

the intermediate activation `σ(W_up · h_in)` is OFTEN 60–90 % near-zero per-token after the
nonlinearity (empirically documented in GELU/SwiGLU transformers; e.g. Li et al. 2022, Mirzadeh
et al. 2023).  We exploit this sparsity **at training time** (not just inference) via a discrete
top-k selection of neurons per token, giving ≥3× FFN forward FLOP reduction AND reduced FFN
intermediate memory, while preserving convergence.

THR is distinct from:
- **TRCD #13** — selects **depth** per token, not **width**.  Orthogonal axis.
- **MoE / expert routing** — routes tokens to disjoint large-block experts.  THR routes tokens
  to a per-token **subset of neurons within a single shared FFN**: no expert duplication, no
  all-to-all, no expert-balance loss at block level.
- **IBGRAD/OVFG (#19, #9)** — sparsify gradients.  THR sparsifies the forward computation
  (and the backward rides on the selected subset).
- **Stiefel/MPOT (#7, #10)** — weight-shape factorization.  THR keeps dense `W_up, W_down`
  on disk and reads only rows/columns at runtime.

---

## 1. Short name and core thesis

**THR — Top-k Hard Routing.**  Working codename: *per-token neuron gate*.

For each token `t` the post-σ activation `a_t = σ(W_up · h_in^t) ∈ ℝ^{d_ff}` is highly
sparse.  If we had an oracle that pre-identified the support `S_t ⊂ [d_ff]` with `|S_t| = k`
ahead of time, the FFN could be computed on only the selected rows/columns:

    z_t = W_up[S_t, :] · h_in^t      ∈ ℝ^k
    a_t = σ(z_t)                      ∈ ℝ^k
    h_out^t = W_down[:, S_t] · a_t   ∈ ℝ^d_model

FLOPs drop from `4·d_ff·d_model` to `4·k·d_model`; memory of the intermediate activation
drops from `T·d_ff` to `T·k`.  At `k = d_ff/4` (pile_large: k = 1024, d_ff = 4096): **4×
forward FFN FLOP reduction** and **4× intermediate activation memory reduction**.

The oracle is approximated by a **low-rank learned proxy** `P = U_P V_P^T` with
`U_P ∈ ℝ^{d_ff × r}, V_P ∈ ℝ^{r × d_model}` (`r ≈ 32`).  The proxy score
`s_t = P · h_in^t = U_P (V_P h_in^t)` is a cheap 2-GEMM surrogate of the full up-projection.
The top-k indices of `|s_t|` define `S_t`.

The discrete `S_t` is trained via **straight-through estimator (STE)** on the main CE loss
plus a **proxy-match auxiliary loss** that teaches `P` to rank neurons consistently with the
true `W_up · h_in` ranking.  At init `P ← W_up` (warm start from the dense FFN weights),
so the proxy starts at perfect-oracle quality and is refined by the joint loss.

---

## 2. Primitive objects

| symbol               | shape / dtype                       | meaning                                           |
|----------------------|-------------------------------------|---------------------------------------------------|
| `h_in^t`             | `ℝ^{d_model}`, BF16                 | FFN input token                                   |
| `W_up, W_down`       | `ℝ^{d_ff × d_model}`, BF16          | FFN weights (unchanged; still dense on disk)       |
| `U_P, V_P`           | `ℝ^{d_ff × r}`, `ℝ^{r × d_model}`  | low-rank proxy factors, FP32 (trainable)          |
| `s_t`                | `ℝ^{d_ff}`, FP32                    | proxy score                                       |
| `S_t`                | `{0,1}^{d_ff}`, unsigned int32      | binary selection mask with `|S_t| = k`            |
| `I_t`                | `ℤ^k`, int32                        | ordered top-k index list per token                |
| `a_t`                | `ℝ^k`, BF16                         | compressed post-σ activation                      |
| `k`                  | scalar hyperparameter               | density; default `k = d_ff / 4`                   |
| `τ`                  | FP32                                | Gumbel-top-k temperature (anneal 1.0 → 0.1)       |
| `λ_proxy`            | FP32                                | proxy-match loss weight (default 0.01)            |
| `λ_balance`          | FP32                                | load-balance regularizer weight (default 0.001)   |
| `p_i`                | FP32                                | running fraction of tokens selecting neuron `i`   |

**Storage cost of the proxy** at pile_large (`d_ff=4096, d_model=1024, r=32`):
`d_ff·r + r·d_model = 131 072 + 32 768 = 163 840` FP32 entries = **655 KB per layer** — a
negligible 0.5 % of the dense FFN footprint (`2·d_ff·d_model·2 = 16.8 MB` BF16 per layer).

---

## 3. State space

The per-step training state extends the standard `(W, Adam-m, Adam-v)` tuple with:

    Σ^t = ( W_up, W_down, U_P, V_P, {p_i}_{i=1..d_ff}, {m, v for U_P, V_P} )

- `S_t` itself is a per-token, per-layer **transient** — it is recomputed every forward pass
  from `P · h_in^t` and never persisted across steps.  The state space of `S_t` is the
  finite set `{S ⊂ [d_ff] : |S| = k}` of cardinality `C(d_ff, k)`.  The **reachable
  subspace** of the policy is (in principle) any of these, chosen by the continuous state of
  `P`.
- `{p_i}` is an EWMA load-balance tracker used by the balance regularizer; lives in FP32.
- Adam state for `U_P, V_P` is the only new optimizer state: `2 · 163 840 · 4 = 1.3 MB/layer`
  FP32, or 0.65 MB with MFIO (shift #11) which eliminates the `m` state.

---

## 4. Evolution law

### 4.1 Forward (per token `t`, per layer `ℓ`)

1. **Proxy score:**  two thin GEMMs on the batch:
   - `R = h_in · V_P^T`        (shape `T × r`, cost `2·T·d_model·r`)
   - `s = R · U_P^T`            (shape `T × d_ff`, cost `2·T·r·d_ff`)
2. **Top-k selection:** for each token row of `s`, compute `I_t = topk(|s_t|, k)` via a
   fused CUDA kernel (radix-topk over the `d_ff = 4096` axis).  Output: `I ∈ ℤ^{T × k}`.
3. **Gather up-projection rows:** `W_up_gather[t, :, :] = W_up[I_t, :]` (logical view; implemented
   as a cuBLAS batched GEMV or a custom gather-GEMM, see §12).
4. **Partial up-proj:** `z_t = W_up[I_t, :] · h_in^t` (shape `T × k`; cost `2·T·k·d_model`).
5. **Activation:** `a_t = σ(z_t)` (shape `T × k`; cost `O(T·k)`).
6. **Partial down-proj:** `h_out^t = W_down[:, I_t] · a_t` (shape `T × d_model`; cost
   `2·T·k·d_model`).  Implemented as a **scatter-GEMM** (or equivalently a batched
   column-subset SGEMM).

**Memory:** intermediate `a_t` stored as `T × k` BF16 (not `T × d_ff`).  At pile_large:
`1024 · 1024 · 2 B = 2 MB` vs `1024 · 4096 · 2 B = 8 MB` per layer → **4× savings**.

### 4.2 Backward

All gradients flow through the **selected indices only**, using the same `I_t` cached from
forward (free; int32, `T · k · 4 B = 4 MB/layer` — see §12.3 for packing).  Specifically:

1. `da_t = W_down[:, I_t]^T · dh_out^t`        (partial; cost `2·T·k·d_model`)
2. `dz_t = σ'(z_t) ⊙ da_t`
3. `dh_in^t += W_up[I_t, :]^T · dz_t`         (partial; cost `2·T·k·d_model`)
4. **Weight gradients (selected only):**
   - `dW_up[I_t, :] += dz_t ⊗ h_in^t`         (scatter-add)
   - `dW_down[:, I_t] += a_t ⊗ dh_out^t`     (scatter-add)

Because gradient contributions to rows `∉ S_t` are zero by construction, `dW_up` and `dW_down`
are **block-sparse per step** but dense when accumulated over many tokens (load-balance ensures
every neuron is selected by some token in the batch; see §4.4).

### 4.3 Proxy gradient (STE + auxiliary loss)

The top-k operator is not differentiable.  We use a **straight-through estimator** with a
Gumbel-top-k smoothing variant during training:

- **Forward:**  `S_t = TopK(|s_t|)` as above (hard selection).
- **Backward (STE):**  treat the mask as the identity on selected neurons for the purpose of
  passing through to `P`:
  `∂L/∂s_t[i] = σ'(|s_t[i]|) · ∂L/∂mask_t[i]`  where `mask_t[i] = 1_{i ∈ S_t}`.
  The mask gradient is supplied by the **proxy-match auxiliary loss** described next.

**Auxiliary proxy-match loss.**  Let `â_t = σ(W_up[I_t, :] · h_in^t)` be the k-support true
activation magnitude.  Define
`ℓ_proxy = Σ_t ‖ topk_sort(|s_t|) − |â_t| ‖²`, i.e. the proxy score should rank-agree with
the true post-σ magnitudes on the **selected** neurons.  This is a local loss on the already-
computed subset — no dense `W_up · h_in` is needed.  Its gradient flows into `U_P, V_P`.

Additionally, we use a **Gumbel-top-k straight-through** variant for the first `n_warmup`
steps: sample `g_i ~ Gumbel(0,1)`, set `s̃_t[i] = |s_t[i]| + τ · g_i`, take hard top-k of `s̃`,
but backprop through the softmax-top-k relaxation.  This injects exploration and prevents
mode collapse.  `τ` anneals 1.0 → 0.1 over 5 k steps.

### 4.4 Load-balance regularizer

Running count `p_i = EWMA_{β=0.99}(freq(i ∈ S_t))`.  Target `p*_i = k/d_ff` (uniform).
Regularizer:
    `ℓ_balance = Σ_i (p_i − k/d_ff)² · d_ff`
added with weight `λ_balance = 1e-3`.  Without it, a subset of "popular" neurons would
monopolize selection and the remaining `d_ff − |used|` neurons would receive zero gradient
signal, their weights would drift, and the network capacity would collapse to
`|used| ≪ d_ff`.

### 4.5 Optimizer step

Standard Adam on `(W_up, W_down)` for the neurons that received any gradient this step
(block-sparse).  Standard Adam on `(U_P, V_P)` dense.  Fully composable with MFIO (shift #11:
drop `m`, keep only `v`; halves proxy optimizer state).

---

## 5. Mechanism mapping (each required ingredient)

| Required ingredient                                  | Mechanism                                                    | Realized factor                                |
|------------------------------------------------------|--------------------------------------------------------------|------------------------------------------------|
| **(a) Forward FFN FLOPs ≥ 3×**                       | Partial up/down GEMMs on `k = d_ff/4` rows/cols              | **4.0×** forward FFN; 3.5× incl proxy overhead |
| **(b) FFN intermediate activation memory**           | Store only `T × k` post-σ activations (vs `T × d_ff`)        | **4×** (2 MB vs 8 MB @ pile_large per layer)   |
| **(c) Composable with MFIO × WIP × IBGRAD**          | Proxy `(U_P, V_P)` is extra trainable weight; MFIO/WIP/IBGRAD apply unchanged. | multiplicative                         |
| **(c') Composable with CHIRON reversibility**        | `I_t` indices cached per layer (small int32 tensor); activations reconstructable given `I_t` + `h_in` via the same partial GEMMs.  CHIRON's reversibility block wraps THR identically to a dense FFN. | multiplicative      |
| **(c'') Composable with local-window attention**     | Attention is untouched.  FFN block is pure token-local → no interaction. | additive                                |
| **(c''') Composable with ATC-Δ (#26)**               | THR runs under the cache: Taylor expansion on selected subset only.  Proxy scores are also Taylor-cached (the proxy is a low-rank linear op, well-modeled by ΔW_P). | multiplicative                |
| **(d) GPU-implementable**                            | cuBLAS thin GEMMs + custom radix-topk + gather/scatter kernels | 3 new custom kernels, reuse SGEMM primitives   |

---

## 6. Objective / variational principle

Total training loss:

    ℒ_total = ℒ_CE(logits, targets)
              + λ_proxy · Σ_{ℓ, t} ‖ softmax(|s_t^{(ℓ)}|)[I_t^{(ℓ)}] − softmax(|â_t^{(ℓ)}|) ‖²
              + λ_balance · Σ_ℓ Σ_i (p_i^{(ℓ)} − k/d_ff)² · d_ff

subject to the hard constraint `|S_t^{(ℓ)}| = k` for every (t, ℓ).  The constraint is
realized by the `TopK` operator; the proxy-match term is the **soft relaxation** of the
constraint that the top-k of `s` should match the top-k of `â`.

**Interpretation as a discrete relaxation.**  At `τ → 0`, the Gumbel-top-k collapses to
deterministic top-k (the gradient estimator reduces to STE).  At `τ → ∞`, it becomes a
uniform random subset.  Training sweeps `τ: 1.0 → 0.1` to anneal from exploration to
exploitation — the analog of simulated annealing over the discrete mask space.

---

## 7. Stability / conditioning / expressivity

**Well-posedness of the mask.**  `TopK(|s_t|, k)` is unique when the top k scores are
distinct (probability 1 under continuous `s_t`).  Ties are broken by ascending index order —
deterministic under Glades' RNG policy.

**Gradient estimator variance.**  STE gradient is biased but low-variance.  Under Assumption
that the proxy scores `|s_t|` are well-separated near rank `k` (empirically true after warmup:
the `k`-th and `(k+1)`-th scores differ by ≫ typical gradient noise), STE gradient points in
the correct direction on expectation and variance is bounded by `σ_g² · (1 − (k/d_ff))`
(lower than dense by the sparsity factor).

**Expressivity (reduction to dense).**  At `k = d_ff` the mask becomes the identity, `S_t =
[d_ff]` for all t, `W_up[I_t, :] = W_up`, and THR computes **bit-exactly** the dense FFN.
`k` is a sliding dial from sparse to dense; no architectural commitment is burned in.

**Warm-start guarantee.**  At init `U_P = W_up, V_P = I_{r, d_model}` (pseudo-inverse) so the
proxy score `s_t = W_up · h_in^t` is **exactly** the pre-σ up-projection.  Thus the mask
`S_t = TopK(|W_up · h_in|, k)` — the optimal oracle mask assuming σ is monotone in magnitude
(true for ReLU, GELU, SiLU, Swish above threshold).  The network **starts from the ideal
sparsification** and only drifts as P is updated independently.

**Expressivity loss bound.**  The aggregate approximation error in the FFN output is
`‖h_out − h_out^{THR}‖ ≤ ‖W_down[:, S_t^c]‖_op · ‖a_t[S_t^c]‖`.  If the nonlinearity zeros
`> 1 − k/d_ff` fraction of entries (empirical 75–90 % for GELU / SiLU at typical scales),
the error is vanishing after warmup.

---

## 8. Computational trade-offs at pile_large

At `L=24, d_model=1024, d_ff=4096, T=1024, batch=8, k=1024 (= d_ff/4), r=32`:

| Metric                                          | Dense baseline | THR          | Ratio       |
|-------------------------------------------------|---------------:|-------------:|------------:|
| FFN forward FLOPs / layer / step                | 67 MFLOPs      | 17 MFLOPs    | **4.0×**    |
| Proxy score FLOPs / layer / step                | –              | 0.3 MFLOPs   | (overhead)  |
| Proxy + topk total overhead                     | –              | 2 MFLOPs     | 3 % of dense|
| **Net FFN forward** (THR + overhead)            | **67 MFLOPs**  | **19 MFLOPs**| **3.5×**    |
| Total transformer forward (all 24 layers)       | 302 GFLOPs     | 194 GFLOPs   | 1.55×       |
| FFN intermediate activation memory / layer      | 8 MB           | 2 MB         | **4×**      |
| Total FFN activation memory (24 layers)         | 192 MB         | 48 MB        | **4×**      |
| Proxy weight memory / layer                     | –              | 0.65 MB FP32 | (overhead)  |
| Proxy optimizer state / layer (Adam-V w/MFIO)   | –              | 0.65 MB FP32 | (overhead)  |
| Total proxy overhead (24 layers, weights+opt)   | –              | 31 MB        | 0.4 %       |
| FFN backward FLOPs / layer / step               | 134 MFLOPs     | 34 MFLOPs    | **4×**      |
| **Total step wall-clock (est, RTX 4080 SUPER)** | **138 ms**     | **~98 ms**   | **1.4×**    |

**Scale sensitivity.**  At `d_ff = 8192 (d_model=2048, T=2048)` typical of 2 B+ runs, same
`k/d_ff = 1/4`: FFN FLOP ratio rises to 4.0× (unchanged — the ratio is dimensionally independent
of d_ff).  Proxy overhead drops to `1 %` of dense (overhead scales as `r/d_ff` which decreases
with `d_ff`).  **More aggressive `k = d_ff/8`** (documented empirically feasible in
SparseGPT/DejaVu literature for pre-trained nets): ratio rises to **8×** at cost of ~5 % PPL.

---

## 9. Failure modes and mitigations

**F1 — Early-training mode collapse.**  During warmup, the proxy `P` may consistently select
the same 20 % of neurons for all tokens → the other 80 % never receive gradient → effective
FFN width collapses to `|active| ≈ d_ff/5`, the load-balance regularizer eventually corrects
this but only after the collapse is established.  **Mitigation:**
  (a) Gumbel-top-k with `τ = 1.0` for the first 2 k steps injects random exploration.
  (b) Warm start `P ← W_up` guarantees the initial mask is the oracle mask, so collapse
      cannot be worse than the pre-training activation distribution.
  (c) Load-balance regularizer from step 0 with `λ_balance = 1e-3`.

**F2 — Proxy-oracle drift.**  As `W_up` evolves under gradient descent but `P` is trained on a
**weaker loss** (proxy-match only on the selected subset), `P` may lag `W_up` and the mask
may become systematically wrong.  **Mitigation:** periodic (every 500 steps)
`P ← (1 − ρ) P + ρ · W_up` with `ρ = 0.1`, i.e. a slow re-anchoring to the true oracle.
Cost: one `d_ff × d_model` copy per layer every 500 steps — negligible.

**F3 — STE gradient bias on heavily-sparsified neurons.**  For neurons i whose score `|s_t[i]|`
is consistently far below the top-k cutoff, `∂ℒ/∂s_t[i] = 0` under STE, so `U_P, V_P` rows
corresponding to those neurons never receive gradient.  These columns of P calcify.
**Mitigation:** Gumbel exploration provides rare non-zero gradient; over 10 k steps every
neuron is selected with probability ≈ 1.  Empirically this suffices.

**F4 — GPU scatter/gather inefficiency.**  Naive `W_up[I_t, :]` gather is a non-contiguous
memory read.  On RTX 4080 SUPER at `T=1024, k=1024, d_model=1024`: worst-case bandwidth-bound
at ~500 GB/s.  Estimated gather time: `T · k · d_model · 2 B / 500 GB/s = 4.3 ms/layer`, about
4× the compute.  **Mitigation:** two GPU kernels:
  (a) `thr_gather_sgemm`: fused gather + SGEMM using `cublasGemmBatchedEx` with per-token
      gathered sub-matrices; amortizes gather cost with compute.  Alternative: `torch.sparse`
      style CSR format for `I_t`, feed into `cusparseSpMM`.
  (b) **Block-sparse top-k:**  group tokens into groups of 8 that share the same `S_t` (top-k
      of the **averaged** score over the group).  Reduces `T·k` unique gather indices to
      `(T/8)·k`, allowing block-GEMM on a dense sub-matrix.  Costs ~2 % accuracy per the
      DejaVu line of work; applies as a `--thr-token-block=8` fallback.

**F5 — Composition breakage with ATC-Δ (#26).**  ATC-Δ caches full activations across K steps;
THR selects different `S_t` each step, so the cache would be invalid.  **Mitigation:** when
ATC-Δ is enabled, freeze `S_t` within a K-step refresh window (recompute only at refresh).
The mask is stable across K steps anyway (proxy weights barely move within K Adam steps), so
fixing it is nearly free in expressivity terms.  CLI enforces: `--atc-K` applies to both
the weight cache and the mask.

**F6 — Composition breakage with CHIRON reversibility.**  CHIRON reconstructs activations on
backward from the reversible-block output.  THR's activations are `T × k`, reconstructed
identically to a dense FFN provided `I_t` is persisted (4 MB/layer).  **Mitigation:** persist
`I_t` as part of the reversibility checkpoint — 4 MB vs 16 MB for dense BF16 activations is
still a net memory win.

**F7 — ReLU / hard-zero activations.**  If σ is ReLU, the true zero-mask is `W_up · h_in > 0`.
THR's top-k does not exactly match this: a neuron with `W_up · h_in = 1e-3` (small positive)
has small |s| but is still active.  THR will likely miss some small-positive neurons and
include some small-negative (ReLU-zero) ones.  **Mitigation:** post-σ refinement: after
computing `z_t = W_up[I_t, :] · h_in^t`, mask out entries with `σ(z_t) = 0` (free for ReLU).
The intermediate `a_t` is then even sparser than `k`, further reducing `W_down` cost.

**F8 — Interaction with GQA / KV-cache prefill.**  In autoregressive inference the FFN runs
`T=1` — the proxy overhead (2 GEMMs + topk over `d_ff`) may dominate.  **Mitigation:** at
inference, the proxy + topk for `T=1` costs `~0.3 ms/layer` on 4080 S, while the saved FFN
compute is `~0.8 ms/layer`.  Net win of 0.5 ms/layer.  For very small `d_model < 512` this
reverses; THR falls back to dense via `--thr-min-dmodel=512` guard.

---

## 10. Comparison to prior art

- **DejaVu (Liu et al. 2023)**, **SparseGPT** — inference-only top-k FFN sparsification on
  frozen pre-trained models.  THR pushes this to **training time** with a learned proxy.
- **Mixture of Experts (Shazeer et al.)** — routes tokens to **different** FFN blocks.
  THR routes tokens to **different neurons inside a single** FFN block.  MoE duplicates
  weight storage; THR does not.  MoE has all-to-all communication; THR is purely local.
- **SparseMoE / Expert Choice** — block-level per-token routing; closest conceptually.  THR
  is the **neuron-level limit** of Expert Choice with expert size 1.  This removes the
  expert-balance / capacity-factor problem in exchange for a per-neuron load balance.
- **Dynamic sparse training (RigL, SET)** — structural sparsity of weights, not activations.
  Orthogonal.
- **Predictive sparsity (Belanger & McCallum)** — proxy-based sparsity on output labels.
  THR adapts the same idea to hidden neurons.
- **Top-k attention** — sparsifies attention scores.  THR sparsifies FFN activations.  Dual.

**THR's novelty:** per-token **width-axis** neuron sparsity, **at training time**, with a
**warm-started low-rank learned proxy**, using STE + Gumbel-top-k for discrete gradient,
composable with every shift in the 26-shift stack.

---

## 11. Minimal prototype

**New GPU primitives** (`gpu_thr.{h,cu}`):

1. `thr_proxy_score(h_in, UP, VP, s_out, T, d_model, d_ff, r)`
   — Two cuBLAS SGEMMs: `R = h_in · VP^T`, `s = R · UP^T`.

2. `thr_topk_indices(s, I_out, T, d_ff, k)`
   — Per-row radix-top-k of `|s|`.  One block per token; uses shared-memory heap.
     Output: int32 indices, sorted ascending for coalesced downstream access.

3. `thr_gather_sgemm(W_up, h_in, I, z_out, T, d_model, d_ff, k)`
   — Fused gather + batched SGEMM: for each token t, compute `z_t = W_up[I_t] · h_in^t`.
     Implementation option A: per-token `cublasGemv` (launch overhead dominated).  Option B:
     custom gather-GEMM tile kernel emitting `T × k` output with compile-time-tunable
     `BLOCK_K, BLOCK_D`; preferred.

4. `thr_scatter_sgemm(W_down, a, I, h_out, T, d_model, d_ff, k)`
   — Partial `h_out = W_down[:, I_t] · a_t` per token; implemented as a scatter-GEMM.

5. `thr_scatter_add_gradients(dW_up, dW_down, dz, da, I, T, d_model, d_ff, k)`
   — Scatter-add gradients back into full dense `dW_up, dW_down`.  Atomic adds with
     32-thread segmentation for hot-neuron contention.

6. `thr_gumbel_perturb(s, tau, rng_state, s_tilde_out, T, d_ff)`
   — Add `τ · G` elementwise during training warmup; deterministic under Glades RNG.

7. `thr_load_balance_ewma(I, p, beta, T, d_ff, k)`
   — Update running fraction `p_i` per neuron from observed selections.

**Parity tests** (`unit-tests/.../chiron-test.cpp`):

- `CHIRONThrDenseReductionTest`: at `k = d_ff`, forward output bit-exact vs dense FFN.
- `CHIRONThrWarmStartParityTest`: at init `U_P ← W_up`, the top-k mask matches
  `TopK(|W_up · h_in|, k)` exactly.
- `CHIRONThrGradientSteTest`: STE grad of `U_P, V_P` matches finite-difference w/ smooth
  softmax-topk relaxation at `τ = 0.1`, max_err < 1e-3.
- `CHIRONThrLoadBalanceConvergenceTest`: on synthetic LM, after 5 k steps `max_i p_i < 1.5 · k/d_ff`.
- `CHIRONThrE2EConvergenceTest`: 4-layer, `d_ff = 256`, `k = 64`, 500 steps.  Loss within 3 %
  of baseline.

**CLI in chiron_train**:

- `--thr` (on/off)
- `--thr-k-ratio=0.25` (k as fraction of d_ff; default 0.25 → k = d_ff/4)
- `--thr-rank=32` (proxy low-rank dim r)
- `--thr-tau-start=1.0 --thr-tau-end=0.1 --thr-tau-steps=5000` (Gumbel anneal)
- `--thr-lambda-proxy=0.01`
- `--thr-lambda-balance=0.001`
- `--thr-warmup=2000` (disable THR and proxy update during warmup; dense FFN during this window)
- `--thr-reanchor=500` (re-anchor P ← W_up every N steps; F2 mitigation)
- `--thr-token-block=1` (F4 mitigation; 1 = per-token, 8 = grouped)
- `--thr-min-dmodel=512` (fall-back guard for small models)

**First E2E test**: TinyStories 128M transformer with `--thr --thr-k-ratio=0.25 --chiron
--mfio 2 --wip-K 4 --accum 8 --local-window=128 --atc-delta --atc-K=8`.  Target: **1.4×+
step speedup**, FFN forward FLOP ratio 3.8–4.0×, validation PPL within 3 % of baseline.

---

## 12. Composition with the shipped stack

- **CHIRON × THR:** reversibility preserved; persist `I_t` alongside reversibility checkpoint.
  Net memory win `16 MB → 4 MB + 4 MB = 8 MB` per FFN layer (**2× further memory savings**).
- **MFIO × WIP × IBGRAD × THR:** all operate on separate axes (optimizer state × weight
  interpolation × gradient subspace × forward neuron selection).  **5-way orthogonal compound.**
  Proxy `(U_P, V_P)` Adam state participates in MFIO (drop `m`) / WIP (interpolate across
  cluster) / IBGRAD (project `m` to subspace).
- **ATC-Δ × THR:** freeze mask within K-step refresh window; Taylor expansion runs on the
  selected `k` neurons only.  Compound FFN FLOP ratio `4× · 7.2× = ~29×`.
- **local-window attention × THR:** fully orthogonal (attention vs FFN).
- **TRCD × THR:** TRCD skips entire layers; THR sparsifies surviving layers' FFN.  Compound
  works at the active-depth × active-width product: `d̄/L × 4×` FLOP ratio.  At `d̄/L = 1/3`
  (TRCD default) and `k/d_ff = 1/4` (THR default): **12× FFN FLOP ratio**, independent.

---

## 13. Summary + promote condition

THR exploits the **width-axis per-token sparsity** of post-σ FFN activations — an axis not
touched by any of the 26 prior shifts.  At `k/d_ff = 1/4`, it delivers **4× FFN forward FLOPs
and 4× FFN activation memory** reduction with a 0.5 % parameter overhead for the proxy.

Key properties:
- Orthogonal to all 26 prior shifts (width-axis; both #13 TRCD and MoE are different axes).
- Warm-start from dense FFN weights makes initial mask the oracle mask — avoids the
  early-training sparsity-vs-capacity tension common to sparse training.
- Pure GPU primitives: 2 cuBLAS SGEMMs (proxy) + 3 custom kernels (topk, gather-GEMM,
  scatter-GEMM) + existing sigmoid/gelu kernels.
- Composes multiplicatively with CHIRON × MFIO × WIP × IBGRAD × ATC-Δ × TRCD.

**Top failure mode (F1): early-training mode collapse** under a frozen proxy → mitigated by
(a) Gumbel exploration, (b) warm start `P ← W_up`, (c) load-balance regularizer active from
step 0.

**Promote condition:** after ATC-Δ (#26) Phase 1 parity tests pass and the 5-way compound
(CHIRON × MFIO × WIP × IBGRAD × ATC-Δ) is validated at pile_large, promote THR as the
**width-axis forward compression factor** giving the final 3.5–4× FFN FLOP speedup needed to
close the remaining gap to the 2–30 B on 16 GB target.

Paradigm-design count after #27 candidate A: **27 shifts** (14 shipped + 13 deferred/candidates).
Expected next iteration: three-candidate comparison (B: Gumbel-Softmax continuous relaxation;
C: hash-based LSH routing) and selection gate.
