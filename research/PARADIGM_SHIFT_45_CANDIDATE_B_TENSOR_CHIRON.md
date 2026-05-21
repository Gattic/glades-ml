# Paradigm Shift #45 Candidate B — TENSOR-CHIRON (Feature-Axis Tensor Parallelism)

**Status:** candidate-B design, single-formulation. Companion to #45-A (HYDRA / pipeline parallelism, layer-axis) and #45-C (TBD).
**Date:** 2026-05-08.
**Tagline:** *Megatron-style tensor parallelism applied to CHIRON: split each weight matrix's feature axis across `n_gpu` GPUs, all-reduce attention output and FFN output per layer. Synchronous (no bubble), simple to implement, NVLink-bound. 4× model-size scaling on `n_gpu = 4` NVLink workstation; standard ML systems technique, not CHIRON-native.*

---

## 0. Executive summary

After paradigm shifts #1-#44 (108× single-GPU compute reduction; 18 B parameter ceiling on 16 GB via MELT TT-FFN), the only remaining axis to scale is distribution. There are two structurally distinct distribution candidates for paradigm #45: **layer-axis** (HYDRA / pipeline parallelism, candidate A) and **feature-axis** (TENSOR-CHIRON / tensor parallelism, this document).

TENSOR-CHIRON splits each weight matrix's column or row axis across `n_gpu` GPUs (Megatron-style), replicates activation buffers, and synchronizes the per-layer shear output via one all-reduce per layer per direction. Each GPU holds `1/n_gpu` of weights and `1/n_gpu` of compute. **No pipeline bubble** (synchronous), but communication every layer, not just at stage boundaries.

**Headline (`n_gpu = 4` on NVLink):**

| Quantity | Single-GPU 18 B (MELT) | TENSOR-CHIRON 4-GPU |
|---|---|---|
| Per-GPU weights | infeasible at 18 B | 4.5 B per GPU |
| Per-GPU compute / step | 1× | 1/4 × + per-layer all-reduce |
| Comm / step | n/a | 1.27 GB (no SCFA) → 80 MB (with SCFA) |
| Comm time NVLink (600 GB/s) | n/a | 2.1 ms → 0.13 ms |
| Comm time PCIe 4.0 (32 GB/s) | n/a | 40 ms (regression) → 2.5 ms (with SCFA) |
| Bubble | n/a | **0** |
| Effective scaling | 1× | **3.5–3.97×** on NVLink |

**Composition with #42 + #43 + #44.** SCFA's compressed `(k, m)` projections shrink TP all-reduce volume 16×. MELT TT cores split bond-axis (`ρ → ρ/n_gpu`) or mode-axis (default; avoids σ-commutation issue). ORION's slow basis `V_*` shards on the `d` axis; reduced coordinate `α ∈ ℝ^r` is replicated. All three compose multiplicatively.

**Honest gap.** TENSOR-CHIRON is a **standard ML systems technique** (Megatron, Shoeybi 2019) applied to CHIRON. It does not exploit any CHIRON-specific structure. CHIRON's reversibility — uniquely beneficial for HYDRA's activation-memory story — *adds 50% comm overhead* under TP (the inverse-walk all-reduces, Theorem B). TENSOR-CHIRON is the less synergistic of the two distribution candidates. It earns a paradigm-#45 candidacy because: (a) mathematically clean and ~600 LOC to implement vs HYDRA's ~2000, (b) sweet spot is the cheap NVLink workstation tier (2-4 GPU) where HYDRA's bubble is worse, (c) composes with HYDRA long-term (PP × TP hybrid endgame).

Mathematical risk: essentially zero. Empirical risk: PCIe-only multi-GPU is infeasible at default `(T, m, L) = (1024, 2048, 53)` without SCFA; NVLink mandatory or SCFA mandatory.

---

## 1. Primitive objects

| Symbol | Type | Definition |
|---|---|---|
| `n_gpu` | `ℕ` | number of GPUs participating in tensor-parallel group, default 4 |
| `g ∈ {0, …, n_gpu−1}` | rank | this GPU's rank in the TP group |
| `L` | `ℕ` | total CHIRON layers, 53 shipped, 96 production target |
| `T, m, n_H, d_H` | `ℕ⁴` | sequence length, model dim, heads, head dim. Constraint: `n_H mod n_gpu = 0` (heads divide cleanly) |
| `H_g := n_H / n_gpu` | `ℕ` | per-GPU head count, e.g. `n_H = 16, n_gpu = 4 → H_g = 4` |
| `(q, p)` | `ℝ^{T×m}²` | CHIRON paired state, **replicated** on every GPU |
| `W_Q, W_K, W_V` | `ℝ^{m × H_g d_H}` per GPU | per-GPU column-shard of the Q/K/V projection (each GPU owns its `H_g` heads' projections) |
| `W_O` | `ℝ^{H_g d_H × m}` per GPU | per-GPU row-shard of the output projection |
| `b_O` | `ℝ^m` per GPU | replicated on all GPUs (added once after all-reduce by rank 0 only, to avoid double-add); see §4.1 |
| `(G_1^{(g)}, G_2^{(g)})` | TT cores | bond-axis-sharded MELT cores, ρ_local = ρ / n_gpu (§7) |
| `Y_g(q)` | `ℝ^{T×m}` | this GPU's local contribution to the per-layer shear correction |
| `AllReduceSum(·)` | NCCL primitive | tree-reduction (deterministic), sums tensors element-wise across ranks |

**Invariants.**
- `n_H mod n_gpu = 0` (or alternatively, `m mod n_gpu = 0` — for the GQA case where KV-heads differ from Q-heads, both `n_H` and `n_KV_H` must individually divide cleanly).
- `(q, p)` is byte-identical on every rank at every layer boundary.
- Adam state lives on the same shard as the parameters: `(m_local, v_local)` for each GPU's `W_Q, W_K, W_V` are `m × H_g d_H` slices.
- TENSOR-CHIRON does **not** introduce any new persistent per-parameter state; it only partitions existing parameters.

---

## 2. State space

Persistent per GPU `g`:

```
S_persistent,g := ( W^{(g)},  Adam(m^{(g)}, v^{(g)} + Kahan c),  rng_state_replicated )
```

`W^{(g)}` includes all layers' sharded `{W_Q, W_K, W_V, W_O, G_1, G_2, …}`. Per-GPU weight footprint: `(model_size) / n_gpu`.

Active per step, **replicated** on every GPU:

```
S_active,g := ( (q, p),  (q_anchor, p_anchor),  Y_g_local,  Y_global,  ∂L/∂(q,p) )
```

**Crucial: activation memory is *not* divided.** The full `(q, p, q_anchor, p_anchor)` lives on every GPU. CHIRON's reversibility gives each GPU O(1)-in-depth activation memory *independently*, but the per-GPU activation cost equals single-GPU. **TENSOR-CHIRON distributes weights and compute, not activations.**

Per-GPU memory at 18 B / 4 GPU on 16 GB ceiling:

| Component | Bytes |
|---|---:|
| Weights bf16 (sharded) | 9.0 GB |
| Adam m, v + Kahan (sharded, FACE/MFIO trims applied) | 4.5 GB |
| Activations (replicated) | 0.5 GB |
| KV-cache + CUDA/NCCL workspace | 2.0 GB |
| **Total** | **16.0 GB** (zero headroom) |

**Honest scaling claim.** TENSOR-CHIRON multiplies the *single-GPU MELT ceiling* by `n_gpu`. At `n_gpu = 4`, the joint ceiling is ≈ 18 B (not 72 B), because per-GPU memory is dominated by activations + KV + workspace once weights shrink to 9 GB. **TENSOR-CHIRON delivers `n_gpu × ` model size, not `n_gpu² × `.**

---

## 3. Evolution law

For a single CHIRON layer ℓ on GPU `g`. Both attention and MLP shears follow the same TP pattern: column-shard the input projection, row-shard the output projection, all-reduce on the way back into `p`.

### 3.1 Attention shear forward

```
Input: q, p ∈ ℝ^{T×m} (replicated on all GPUs)

(1) Column-sharded Q/K/V projection on GPU g:
    Q_g = q · W_Q^{(g)}, K_g = q · W_K^{(g)}, V_g = q · W_V^{(g)}    ∈ ℝ^{T × H_g d_H}
    Cost: 3 · T · m · H_g d_H FLOPs   ← 1/n_gpu of full

(2) Local attention on GPU g's H_g heads:
    For h ∈ heads_g: O_h = softmax((Q_g_h K_g_h^⊤)/√d_H + M) · V_g_h
    O_g = concat_h(O_h)                                              ∈ ℝ^{T × H_g d_H}
    Cost: H_g · T² d_H FLOPs           ← 1/n_gpu of full

(3) Row-sharded output projection on GPU g:
    Y_g_local = O_g · W_O^{(g)}                                      ∈ ℝ^{T × m}

(4) AllReduceSum_g(Y_g_local) → Y_global                             ∈ ℝ^{T × m}
    Comm: 1 all-reduce, T·m·2B = 4 MB at (T=1024, m=2048).

(5) Y = Y_global + b_O    (b_O replicated)
(6) p ← p + Y             (replicated update; q unchanged)
```

**Correctness:** `Σ_g Y_g_local = Σ_g (concat_h O_h) · W_O^{(g)} = O · W_O = Y`, exact in real arithmetic.

### 3.2 MLP shear forward (with MELT TT cores)

MELT's two TT cores G_1, G_2 with bond axis ρ have two natural sharding choices: **bond-axis** (split ρ) or **mode-axis** (split m_1 or n_2). Bond-axis fails the σ-commutation test: the activation σ between G_1 and G_2 acts on `Σ_ρ G_1[ρ]·q`, which means the ρ-sum must complete *before* σ — hence bond-axis would require a pre-σ all-reduce, two per FFN.

**Default: mode-axis sharding** on `n_2`. Each GPU holds `G_2^{(g)}[all_ρ, :, n_2/n_gpu, 1]`; σ commutes; one all-reduce per FFN forward.

```
Input: q ∈ ℝ^{T×m} (replicated)

(1) z = G_1 · q           (replicated; G_1 small enough to replicate per layer)
(2) x = σ(z)              (replicated)
(3) y_g_local = G_2^{(g)} · x   (sharded on n_2; produces partial sum over n_2 slice)
(4) AllReduceSum_g(y_g_local) → Y_global
(5) p ← p + Y_global + b_FFN
```

Bond-axis details and the σ-commutation analysis are deferred to §7 / Theorem C.

### 3.3 Inverse shear

CHIRON's inverse `(q', p') ↦ (q', p' − Y(q'))` needs `Y(q')`. Each GPU recomputes its local `Y_g(q')`, all-reduces, subtracts from p'. **Each inverse-walk shear requires its own all-reduce.** This is a CHIRON-specific cost of TP that non-reversible transformers don't pay.

### 3.4 Backward pass

```
g_Y = g_p                          (replicated input gradient)
g_O_g = g_Y · W_O^{(g)⊤}           (no comm; row-shard's transpose gives column-shard)
... attention backward in heads_g, producing local g_q contribution
g_q_g_local = dQ_g · W_Q^{(g)⊤} + dK_g · W_K^{(g)⊤} + dV_g · W_V^{(g)⊤}
g_q_global = AllReduceSum_g(g_q_g_local)        ← backward dual of forward §3.1 all-reduce
```

**Comm budget per step:**
- Forward: 2L all-reduces (attention + FFN, one each).
- Backward: 2L all-reduces (gradient duals).
- Inverse walks (CHIRON-specific): 2L additional all-reduces.

**Grand total: 6L all-reduces × T·m·bf16 = 6 · 53 · 4 MB = 1.27 GB / step** at default config.

Note: a non-reversible transformer with TP only pays 4L (no inverse-walk overhead). **CHIRON's reversibility increases TP comm by 50%** — the prompt's 868 MB estimate omitted this; the honest figure is 1.27 GB.

---

## 4. Bias, LayerNorm, embedding handling

**Bias `b_O`** is replicated; added on every GPU after the all-reduce. `Y = AllReduce(Y_g) + b_O`. Correct because every rank computes the same Y identically.

**ReLN** acts on replicated `q`; computed independently per GPU; stored stats `(μ, log σ)` are replicated. No comm.

**Embedding/unembedding** — replicate the `V × m` token matrix (~200 MB bf16 at vocab=50k, m=2048). Saves all-gather complexity at modest memory cost. Loss head similarly replicated; final per-step loss uses one small `r=1` AllReduce on the scalar loss.

---

## 5. Theorems

### Theorem A (forward correctness)

**Claim.** TENSOR-CHIRON forward `Y_TP(q) = Y_baseline(q)` exactly in real arithmetic; in bf16, `‖Y_TP − Y_baseline‖ ≤ log_2(n_gpu) · ε_bf16` per element under tree reduction.

**Proof.** Attention: `Y(q) = O · W_O = (concat_h O_h) · W_O = Σ_g (concat_{h ∈ heads_g} O_h) · W_O^{(g)} = Σ_g Y_g_local`, by column/row sharding identity. Tree-reduction adds `n_gpu` partial sums in `log_2(n_gpu)` rounds; each round adds one bf16 ulp of error. ∎

NCCL's `ncclAllReduce` with `ncclTree + ncclProtoSimple` is deterministic given fixed topology, so TP is bit-deterministic across runs at fixed rank assignment.

### Theorem B (CHIRON inverse walk correctness, with comm-cost corollary)

**Claim.** The CHIRON inverse `(q', p') ↦ (q', p' − Y(q'))` is exact under TP iff an all-reduce on `Y(q')` precedes the subtraction.

**Proof.** Each GPU holds `q'` replicated (exact from upstream) and weight shard `W^{(g)}`. Local `Y_g(q')` is recomputed identically. Without an all-reduce, each GPU would subtract `Y_g_local`, leaving residual `p − Y_g_local + (Y − Y_g_local) ≠ p`. **The inverse walk requires its own all-reduce per shear.** ∎

**Corollary.** TENSOR-CHIRON's per-step all-reduce count is `4L` for forward+backward of a *non-reversible* transformer, plus `2L` for CHIRON's inverse walks = **6L total**. CHIRON's reversibility imposes a 50% comm overhead under TP — the cost of pairing reversibility with feature-axis distribution.

### Theorem C (MELT TT-core sharding correctness)

**Claim (bond-axis).** `Σ_{ρ_1=0}^{ρ-1} G_1[ρ_1] · G_2[ρ_1] = Σ_g Σ_{ρ_1' ∈ [0, ρ/n_gpu)} G_1^{(g)}[ρ_1'] · G_2^{(g)}[ρ_1']` exactly, by the bond-axis partition.

**Proof.** Direct from `[0, ρ) = ⊔_g [g·ρ_g, (g+1)·ρ_g)`. Each GPU computes its slice's contribution; AllReduceSum reconstructs the full ρ-sum. ∎

**Caveat.** Bond-axis sharding fails the σ-commutation test: in MELT-FFN, σ acts between G_1 and G_2 on `Σ_ρ G_1[ρ]·q`. Each GPU only has `σ(z_g)` (local sum), not `σ(Σ_g z_g)` (global sum). Mitigation: pre-σ all-reduce on z. Then bond-axis costs **two** all-reduces per FFN forward — vs **one** for mode-axis sharding (split m_1 or n_2; σ commutes element-wise with mode shards).

**Default: mode-axis sharding** for MELT × TENSOR-CHIRON. Bond-axis preserved as a research alternative for very-large-ρ regimes (not our default ρ=8).

---

## 6. Composition with #42 SCFA

SCFA's per-layer basis `B_ℓ ∈ ℝ^{T×k}` is small (`k=64`, ~130 KB / layer) and learned per-layer. **Replicate `B_ℓ` on every GPU.**

After spectral projection `q̂ = B^⊤ q ∈ ℝ^{k×m}`, dense projections `W_Q, W_K, W_V` are tensor-parallel exactly as in §3.1, but operate on `k×m` not `T×m`:
- All-reduce on spectral-space output `Ô_local ∈ ℝ^{k × m}` = **256 KB**, vs full TP's 4 MB. **16× smaller comm.**
- Lift `y_∥ = B · Ô_global` runs locally per GPU after the all-reduce.

**Combined comm per step at `L = 53, n_gpu = 4`:**

| Configuration | Comm / step |
|---|---:|
| TP without SCFA | 1.27 GB (6L × 4 MB) |
| TP with SCFA | 80 MB (6L × 256 KB) |

At NVLink 600 GB/s, 80 MB transit = 0.13 ms — utterly negligible. **SCFA × TP is the recommended composition; PCIe-only deployment requires it.**

---

## 7. Composition with #43 ORION

ORION's persistent state shards naturally:
- `V_* ∈ ℝ^{d × r}` — sharded on `d` axis (matches weight shard); per-GPU `V^{(g)} ∈ ℝ^{d/n_gpu × r}`.
- `θ_⊥*` — same shard pattern.
- `α ∈ ℝ^r`, `H_∥* ∈ ℝ^{r×r}`, `A_∥*` — **replicated** (tiny).
- `g_∥*` — computed via `r`-AllReduce: GPU g forms `V^{(g)⊤} g^{(g)} ∈ ℝ^r` (local inner product), AllReduce sums (`r·4 = 16 B`, free).

**K-window closed-form** runs replicated on every GPU; each computes the same `α_K` arithmetic; **zero comm between anchors.** Per-rank lift-back `θ^{(g)} = θ_⊥*^{(g)} + V^{(g)} α` updates only that rank's weight shard — no cross-GPU sync.

**Anchor cost under TP:** `(3 + 2r)` full F+B passes with TP overhead. At K=20, r=4: anchor amortized to one full-comm step in 22 SGD-equivalent ones. **Steady-state TP comm per effective step ≈ original / K** — ORION reduces TP's comm pressure by an order of magnitude.

---

## 8. Composition with #44 MELT

Two sharding modes:

| Mode | Axis | All-reduces / FFN | σ-commutation |
|---|---|---:|---|
| **Mode-axis (default)** | n_2 | 1 | OK (σ element-wise) |
| Bond-axis | ρ | 2 | needs pre-σ all-reduce |

Mode-axis: G_1 replicated per layer (130 KB total); G_2^{(g)} sharded on n_2 (8 KB / GPU at ρ=8). Memory ratio vs dense: 205× MELT × n_gpu = **820× at n_gpu=4**.

**Combined compute speedup vs dense FFN:** MELT 3.2× × TP n_gpu × (1 − comm_frac) ≈ 3.2 × 4 × 0.95 = **12× per-FFN compute** at n_gpu=4 NVLink.

---

## 9. Performance analysis

Reference: single-GPU 18 B step ≈ 60 ms (extrapolating from 1.84B's ~50 ms × 10× param × MELT-stack 108×).

### 9.1 Step time at 18 B / 4 GPU

| Configuration | Compute / GPU | Comm / step | Total step | Effective scaling |
|---|---:|---:|---:|---:|
| TP NVLink, no SCFA | 15 ms | 2.1 ms | 17.1 ms | 3.5× |
| TP NVLink, with SCFA | 15 ms | 0.13 ms | 15.1 ms | **3.97×** |
| TP PCIe 4.0, no SCFA | 15 ms | 40 ms | 55 ms | **0.55× regression** |
| TP PCIe 4.0, with SCFA | 15 ms | 2.5 ms | 17.5 ms | 3.43× |

**Critical finding.** Without SCFA, PCIe-only TENSOR-CHIRON is a regression. **SCFA is mandatory for PCIe deployment.**

### 9.2 Scaling ceiling at `n_gpu = 8`

| Topology | Comm time | Total step | Effective scaling |
|---|---:|---:|---:|
| NVSwitch 900 GB/s | 1.4 ms | 8.9 ms | **6.7×** |
| NVLink 600 GB/s ring | 4.2 ms | 11.7 ms | 5.1× |

**TENSOR-CHIRON's effective scaling saturates near `n_gpu = 8` on NVLink** because all-reduce time grows as `(n_gpu − 1) / n_gpu × M / B` while per-GPU compute drops as `1/n_gpu` — the curves cross. HYDRA scales further (deeper L permits larger n_gpu).

---

## 10. Material distinctness vs HYDRA — full comparison

| Axis | HYDRA (#45-A) | TENSOR-CHIRON (#45-B) |
|---|---|---|
| Split direction | Layer-axis (depth) | Feature-axis (width) |
| Memory per GPU | (model size) / n_gpu | (model size) / n_gpu |
| Activation memory per GPU | `O(T · m)` (CHIRON anchor at segment input) | `O(T · m)` (CHIRON anchor — full state, replicated) |
| Comm pattern | Sparse (boundary only, per microbatch) | Dense (every layer per direction) |
| Comm volume / step | μ · 8 MB ≈ 0.26 GB at μ = 32 | 6L · 4 MB ≈ 1.27 GB at L = 53 |
| Comm volume × SCFA | μ · 8 MB (unchanged) | 6L · 256 KB ≈ 80 MB (16× smaller) |
| Bubble fraction at μ = 4n_gpu | 0.18 (82% efficient) | 0 (synchronous) |
| Comm overlap with compute | possible (per microbatch) | partial (ring-AR can overlap with last shear's compute) |
| Bandwidth requirement | NVLink for n_gpu ≥ 4 | NVLink for n_gpu ≥ 2 |
| Scaling ceiling | `n_gpu = 16+` feasible (smaller bubble at deeper L) | `n_gpu ≈ 8` on NVLink (comm-bound beyond) |
| CHIRON-synergy | **Strong** — reversibility uniquely eliminates per-stage activation memory; CHIRON is uniquely well-suited for PP among transformers | **None special** — TENSOR-CHIRON works the same way for any transformer; CHIRON's reversibility *adds 50% comm* via inverse-walk |
| Implementation complexity | High (~2000 LOC: scheduler, NCCL, state machine) | Medium (~600 LOC: NCCL all-reduce, sharded weight access, bias handling) |
| Hardware sweet spot | 8-GPU cluster ($5k-$10k commodity) | 2-4 GPU NVLink workstation ($2k-$5k) |
| Determinism risk | Higher (NCCL Send/Recv ordering, bf16 across stages) | Lower (single tree-AR per layer, deterministic) |
| Latency increase / step | ~1.25× (warmup + cooldown) | ~1.0× (synchronous, no pipeline depth) |
| Engineering risk | Higher (state machine bugs, microbatch ID confusion) | Lower (pure synchronous SPMD) |
| Composes with the other | **Yes** — PP × TP hybrid is the standard approach for very large models | Yes (same axis composition) |

**Bottom line.** HYDRA is more synergistic with CHIRON (reversibility specifically) and scales further; TENSOR-CHIRON is simpler to implement and addresses a smaller, cheaper hardware tier. They are *complementary*, not redundant. A future PP × TP hybrid (each pipeline stage internally tensor-parallel) is the natural endgame for true 100B+ deployments.

---

## 11. Failure modes

| ID | Failure mode | Mitigation |
|---|---|---|
| F1 | PCIe bandwidth-bound stall | Mandate SCFA composition for PCIe; prefer NVLink hardware |
| F2 | bf16 all-reduce reduction-order non-determinism | Pin NCCL to ncclTree algo + ncclProtoSimple; document acceptable nondeterminism band |
| F3 | n_H not divisible by n_gpu (e.g., n_H = 16, n_gpu = 6) | Pad n_H or warn-and-degrade to n_gpu = gcd(n_gpu, n_H) |
| F4 | Inverse-walk bf16 drift compounded by all-reduce error | Each per-GPU inverse re-uses same all-reduce path as forward; drift bounds are symmetric and same as single-GPU CHIRON; Theorem B holds |
| F5 | NCCL initialization deadlock on shared-host multi-GPU dev box | Use NCCL `cudaIpc` transport; bootstrap via TCP for cross-node; ensure CUDA context order matches rank order |
| F6 | KV cache during inference on TP rank | KV-cache shards along the head axis (matching attention); inference all-reduce per-token is `m bf16 = 4 KB` — negligible |
| F7 | Composition with FACE (paradigm #28) on tensor-parallel embedding | FACE embeds `m`-axis Adam state compression; under TP the embedding is replicated, FACE applies independently per GPU; **no interaction** |
| F8 | Composition with RLG (paradigm #39) | RLG inserts new layers mid-training; TP partition recomputed for each new layer (one-shot, ≈ 0 cost). New layer's W initialized as identity (Wo=0); shards of `Wo=0` are zero-init shards. **Composes cleanly.** |
| F9 | Composition with SAS (paradigm #40) | SAS skips per-layer with probability α; if some ranks skip and others don't, replication breaks. **Mitigation: sync SAS RNG across ranks; all GPUs make the same skip decision.** |
| F10 | Single-GPU regression risk | TENSOR-CHIRON activates only when n_gpu ≥ 2; n_gpu = 1 path is identical to baseline (no-op) |

Most serious: F1 (PCIe bandwidth) — strictly avoidable by hardware choice or by composing with SCFA. F9 (SAS sync) — a CHIRON-specific concern; trivial to fix.

---

## 12. Concrete primitives

```cpp
// Backend/Machine Learning/Networks/cuda/gpu_nccl_tp.h
namespace glades { namespace gpu { namespace tp {
void initTPGroup(int rank, int worldSize, const std::string& bootstrapAddr);
int  worldSize();
bool isTPActive();
void allReduceSumBF16(__nv_bfloat16* buf, size_t count, cudaStream_t s);
void allReduceSumF32(float* buf, size_t count, cudaStream_t s);
}}}

// Backend/Machine Learning/Networks/tp_layer.h
namespace glades { namespace tp {
struct ColumnShardedLinear {  // W shard [m, H_g d_H]
    GpuBuffer<__nv_bfloat16> W_shard, b_replicated;
    void forward(const GpuBuffer<__nv_bfloat16>& X, GpuBuffer<__nv_bfloat16>& Y_local, cudaStream_t) const;
    void backward(...);
};
struct RowShardedLinear {  // W shard [H_g d_H, m]; caller all-reduces output
    GpuBuffer<__nv_bfloat16> W_shard, b_replicated;
    void forward(const GpuBuffer<__nv_bfloat16>& X_local, GpuBuffer<__nv_bfloat16>& Y_local, cudaStream_t) const;
    void backward(...);
};
void runAttentionShearTP(const TPLayerWeights&, q, p, scratch, stream);
void runFFNShearTP_MELT(const TPLayerWeightsMELT&, q, p, scratch, stream);
void runAttentionShearTP_Inverse(...);
void runFFNShearTP_MELT_Inverse(...);
}}
```

Driver:

```cpp
for (int l = 0; l < L; ++l) {
    glades::tp::runAttentionShearTP(layer[l].attn, q, p, scratch, stream);  // 1 all-reduce
    glades::tp::runFFNShearTP_MELT(layer[l].ffn, q, p, scratch, stream);    // 1 all-reduce
    glades::reln_forward(layer[l].ln, q, scratch.lnStats[l], stream);       // no comm
}
glades::tp::allReduceSumF32(&loss_local, 1, stream);
loss_global = loss_local / glades::gpu::tp::worldSize();
```

**Total LOC: ≈ 600** (tp_layer.h/cpp, gpu_nccl_tp.h/cu, integration). Compare HYDRA's ~2000.

---

## 13. Gate-0: 2-GPU correctness + bandwidth measurement

**Hypothesis.** TP-CHIRON on 2 NVLinked GPUs produces gradients within `1e-4` bf16 tolerance of single-GPU; per-step wall-clock within 10% of `1/n_gpu × T_single` (with SCFA).

**Setup.** 2× RTX 4080 SUPER, NVLink bridge. Tiny CHIRON `L=4, T=64, m=128, n_H=4, n_gpu=2, H_g=2`. Rank 0 owns heads {0,1}, rank 1 owns {2,3}.

**Procedure.**
1. Reference: single-GPU CHIRON, save layer gradients `g_W` per shear.
2. TP run: same seed/input; each rank produces shard gradients.
3. Compare: F-norm relative error per layer's W_Q/K/V/O slice < `1e-4`.
4. Loss match: < `1e-4`.
5. Profile: NVLink throughput ≥ 400 GB/s on 4 KB / 256 KB tensors.

**Triage.** `1e-4 ≤ err < 1e-2`: bf16 reduction order — pin ncclTree. `≥ 1e-2`: shard indexing bug. `≥ 1e-1`: scheduler bug.

**PCIe sub-Gate-0:** rerun on PCIe 4.0; verify ≥ 25 GB/s throughput. If yes → PCIe is at-budget with SCFA; if no → mandate SCFA.

**Cost:** Phase 1+2 (~400 LOC) ≈ 2 weeks before Gate-0.

---

## 14. Implementation roadmap

≈ 600 LOC over 4 phases, ≈ 3 weeks focused engineering (vs HYDRA's 6 weeks / 2000 LOC):

| Phase | LOC | Time | Deliverable |
|---|---:|---|---|
| 1: NCCL TP primitives | 150 | 1 wk | `gpu_nccl_tp.h/cu`; 2-GPU AllReduce smoke test |
| 2: TP layer infra | 250 | 1 wk | `tp_layer.h/cpp`; column/row sharded linear; Gate-0 correctness |
| 3: chiron_main integration | 150 | 0.5 wk | Env-var rank, sharded checkpoint, TP forward path |
| 4: Compose #42/#43/#44 | 50 | 0.5 wk | SCFA basis replication; ORION r-AllReduce; MELT mode-axis |

---

## 15. Honest gaps

1. **Standard technique, not CHIRON-native.** Megatron-style TP (Shoeybi 2019) applied to CHIRON. Does not exploit CHIRON-specific structure. Reversibility — uniquely beneficial for HYDRA's activation memory — *adds 50% comm under TP* (Theorem B). **TP is applied to CHIRON, not native to it.**
2. **PCIe-bound without SCFA.** 1.27 GB / step on PCIe 4.0 = 40 ms transit ≈ regression vs single-GPU. NVLink mandatory or SCFA mandatory.
3. **Scaling ceiling at `n_gpu = 8`.** Comm vs compute crossover saturates further scaling on NVLink. HYDRA scales further (deeper L permits deeper pipelines).
4. **Activation memory not distributed.** CHIRON reversibility helps each GPU O(1)-in-depth, but per-GPU activation cost matches single-GPU. Weight scaling = n_gpu×; model size at 16 GB ceiling capped accordingly.
5. **SAS sync required.** Stochastic per-layer skipping must use replicated RNG; trivial fix but a CHIRON gotcha.
6. **RLG repartition.** New layers from #39 RLG need sharding; identity-init makes shard zero-init trivial.
7. **Determinism within single host only.** NCCL bf16 is deterministic at fixed topology; cross-host deployment may differ.
8. **No multi-host without RDMA.** Cross-node TP requires InfiniBand/RoCE; this doc scopes single-host n_gpu ≤ 8.
9. **HYDRA dominates on synergy.** Anyone weighing #45-A vs #45-B picks A if reversibility integration is the goal. TP's niche is the cheap-engineering / NVLink-workstation tier.
10. **PP × TP hybrid is paradigm #46.** Nested HYDRA + TENSOR-CHIRON for 100B+ models is out of scope here.

---

## 16. Decision summary

TENSOR-CHIRON is the paradigm-#45 candidate that **trades CHIRON-specificity for engineering simplicity.** Right candidate if:
- Target hardware: 2-4 GPU NVLink workstation ($2k-$5k tier).
- Engineering budget: ≤ 1 month (vs HYDRA's ~6 weeks).
- Composition with SCFA brings comm into NVLink/PCIe-comfortable regime.

**Risk profile.** Mathematical: very low (textbook TP + clean derivations). Engineering: low (~600 LOC, mature NCCL). Bandwidth: bounded by SCFA composition or NVLink hardware.

**Headline.** 4× model size at n_gpu=4 NVLink; 3.97× effective scaling with SCFA; 6.7× at n_gpu=8 NVSwitch. Combined wall-clock × parameter scaling vs single-GPU 1.84B baseline: **108 × 3.97 ≈ 430×** at n_gpu=4 (with effective 3.6× model capacity at the per-GPU 16 GB ceiling).

**Honest disposition.** TENSOR-CHIRON is the **less synergistic** of the two distribution candidates for #45. HYDRA leverages CHIRON's reversibility uniquely and scales further; TENSOR-CHIRON does not, and adds 50% comm overhead via inverse-walks. It is mathematically clean, engineering-cheap, and addresses a real (NVLink workstation) hardware tier — but it is not a CHIRON breakthrough; it is tensor parallelism applied to CHIRON.

For a project that has spent 44 shifts maximally exploiting CHIRON's structural properties, **HYDRA is the stronger paradigm-#45 selection.** TENSOR-CHIRON is the stronger selection for a project prioritizing distributed scaling with minimal new engineering and willing to use a standard ML systems technique.
