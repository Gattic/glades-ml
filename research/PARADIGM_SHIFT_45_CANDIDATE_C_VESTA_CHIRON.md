# Paradigm Shift #45 Candidate C — VESTA-CHIRON (ZeRO-3 State-Sharded CHIRON)

**Status:** candidate-C design, single-formulation. Companion to #45-A (HYDRA, pipeline-parallel layer split) and #45-B (TENSOR-CHIRON, tensor-parallel feature split).
**Date:** 2026-05-08.
**Axis:** distributed CHIRON via *state* (weights, gradients, optimizer state) sharding rather than compute sharding.
**Tagline:** *Each GPU computes the full layer graph but holds only `1/n_gpu` of every parameter tensor. Broadcast a layer's shard before its forward; reduce-scatter its gradient before its update. Linear memory scaling, zero compute scaling, NVLink-bound.*
**Materially distinct from:**
- **HYDRA (#45-A, pipeline parallel):** HYDRA splits *layers* across GPUs; each layer lives on one GPU. VESTA splits *parameters within each layer* — every GPU owns 1/n of every layer.
- **TENSOR-CHIRON (#45-B, tensor parallel):** TENSOR splits feature dimensions of W and AllReduces partial activation outputs (`O(T·m)`); VESTA replicates activations and broadcasts entire weight shards (`O(m²)`).
- **DDP (shipped):** DDP replicates all state and AllReduces gradients. VESTA replicates only activations and shards state — orthogonal to DDP, complementary in a 3-D parallelism stack.
- **FACE/MFIO/Kahan-v:** single-GPU per-parameter compression; VESTA partitions the *number of parameters touched per GPU*. Compositional.

VESTA-CHIRON is a faithful adaptation of **ZeRO Stage 3** (Rajbhandari et al., SC 2020) to CHIRON. **Honest framing: this is standard ZeRO-3 with a CHIRON-specific addendum for the inverse walk.** Most of the mathematics is well-established; the new content is (a) how ZeRO-3 broadcasts compose with the reversible inverse walk, (b) how it composes with #42/#43/#44, and (c) its honest cost relative to HYDRA on the same hardware.

---

## 0. Executive summary

After paradigm shifts #42 (SCFA) + #43 (ORION) + #44 (MELT), CHIRON's flagship is at **108× wall-clock speedup over the iter-1 baseline** on a single 16 GB RTX 4080 SUPER, with a single-GPU model-size ceiling of **~18 B parameters**. Pushing beyond 18 B on consumer hardware requires distribution: the per-GPU memory budget is exhausted, even after FACE (1008× embedding compression), MFIO (682× attention compression), MELT (205× FFN-weight compression), and Kahan-v.

VESTA-CHIRON answers: *don't shrink state — shard it across GPUs.* Each layer's parameter tensor `W_i ∈ ℝ^{12m² + bias}` is partitioned into `n_gpu` shards `W_i^{(g)} = W_i[g·shard : (g+1)·shard]`. Each GPU `g` persistently stores only `W_i^{(g)}, m_i^{(g)}, v_i^{(g)}` for every layer `i`. To compute layer `i`'s forward, GPU `g` issues a **broadcast** of its shard to all peers, every GPU now holds the *full* `W_i` transiently, all GPUs compute layer `i` forward on their own copy of activations, then non-owner GPUs **drop** the full `W_i`. The same pattern reverses on backward (broadcast `W_i`, compute, **reduce-scatter** the gradient back to the shard owner).

**Headline at `n_gpu = 4`, NVLink-equipped 4× RTX 4080 SUPER (≈ $5 k cluster):**

| Quantity | Single-GPU (#42+43+44) | VESTA-CHIRON `n_gpu=4` | HYDRA `n_gpu=4` (companion #45-A) |
|---|---|---|---|
| Total addressable model | 18 B | **72 B** | 72 B (same memory linear scaling) |
| Per-GPU weights (BF16 + MELT trims) | 12 GB | 9 GB | 12 GB (full layer, fewer layers) |
| Per-GPU Adam state (FACE+MFIO+MELT) | 1.2 GB | 0.225 GB | 1.2 GB |
| Per-GPU activation memory | O(T·m) ≈ 0.2 GB | O(T·m) ≈ 0.2 GB | O(T·m) per stage ≈ 0.2 GB |
| Per-step cross-GPU traffic | n/a | **28 GB/step** (bandwidth-bound) | 3.6 GB/step (bandwidth-comfortable) |
| NVLink-3.0 / 4.0 utilisation | n/a | 30–50 % step time | 8–12 % step time |
| PCIe-4.0 viability | n/a | **infeasible (1.1 s/step)** | marginal (200 % step time without overlap) |
| Bubble fraction | n/a | 0 % (no pipeline) | 18 % at μ = 16 |
| CHIRON-specific architectural lever | inverse walk (full) | inverse walk (broadcast pattern repeats) | inverse walk (segment-local) |
| Implementation complexity | baseline | **medium-high** (overlap comm/compute) | high (1F1B scheduler + boundary protocol) |
| Effective compute per GPU | 1× | **1×** (no compute speedup) | `n_gpu · (1 − β) ≈ 3.3×` |

**Honest claim, three numbers:**

1. **Memory:** VESTA-CHIRON enables **72 B at n_gpu = 4** with NVLink, full single-step semantics (no pipeline bubble, no staleness).
2. **Throughput:** **30–50 % comm overhead under NVLink 3.0/4.0**; effectively 0.5–0.7× single-GPU throughput per GPU on a 4× larger model — net `≈ 2.5×` wall-clock speedup. **Significantly less compute scaling than HYDRA.**
3. **Hardware:** NVLink **required**; PCIe-4.0 is structurally infeasible (28 GB / 25 GB·s⁻¹ = 1.12 s/step exceeds the entire un-accelerated single-GPU step time).

**Why anyone would pick VESTA over HYDRA:** no pipeline-depth bubble penalty (HYDRA's β → 0.5 at `n_gpu ≥ 16`); no microbatch-uniformity constraint; checkpoint resharding is a parameter-level operation. **In production, VESTA + HYDRA + DDP form a 3-D parallelism stack** (ZeRO-3 + PP + DP, the Megatron-DeepSpeed architecture). VESTA is most valuable as a *complement* to HYDRA, not as an exclusive choice.

The candidate is included in #45 because (a) it is materially different in axis, (b) its math is rigorous, (c) it provides the strongest pure-memory scaling. It is **not** the recommended standalone #45 selection if HYDRA is feasible — HYDRA gives both memory and compute scaling.

---

## 1. Primitive objects

| Symbol | Type | Definition |
|---|---|---|
| `n_gpu` | `ℕ`, ≥ 2 | number of GPUs in VESTA group; default `n_gpu = 4` |
| `g ∈ {0, …, n_gpu−1}` | rank | this GPU's index |
| `L` | `ℕ` | total CHIRON layers (53 shipped; up to 96 production target) |
| `m, T, n_H, d_H` | as in CHIRON | embedding/sequence/head dims; unchanged by VESTA |
| `W_i` | `ℝ^{P_i}` | layer-`i` parameter vector; `P_i ≈ 12 m² + bias` |
| `W_i^{(g)}` | `ℝ^{P_i / n_gpu}` | GPU `g`'s persistent shard of `W_i`; flat-vector partitioning |
| `Ŵ_i` | `ℝ^{P_i}` | **transient** full-layer weight buffer, materialised only during layer `i`'s active compute window |
| `m_i^{(g)}, v_i^{(g)}` | `ℝ^{P_i / n_gpu}` | Adam EMA shards; same partitioning as `W_i^{(g)}` |
| `g_i` | `ℝ^{P_i}` | **transient** full-layer gradient, computed locally on every GPU during backward |
| `g_i^{(g)}` | `ℝ^{P_i / n_gpu}` | persistent gradient shard owned by GPU `g`, formed by reduce-scatter of `g_i` |
| `(q, p)` | `ℝ^{T×m} × ℝ^{T×m}` | CHIRON paired state; **fully replicated** on every GPU |
| `B_g` | NCCL primitive | broadcast: `B_g(W_i^{(g)}) → Ŵ_i` on all peers |
| `RS_g` | NCCL primitive | reduce-scatter: `RS_g(g_i) → g_i^{(g)}` on owner |
| `s_W` | `ℕ` | weight-stream id for async broadcast pipeline |
| `s_C` | `ℕ` | compute-stream id for kernel launches |

**Sharding rule (flat-vector, layer-aligned).** Each layer's parameter vector is partitioned by contiguous slabs of `P_i / n_gpu` floats; Adam shards align with the weight shard. **No structural awareness** — VESTA treats `W_i` as a flat blob (matches ZeRO-3 design).

**Determinism.** NCCL bf16 reductions are non-deterministic across rank counts unless the reduction tree is fixed; we pin `NCCL_ALGO=Tree` and accept rank-fixed determinism. **VESTA breaks the bit-exact determinism guarantee** of `DETERMINISM_AND_CONCURRENCY.md` across rank counts (deterministic for a given `n_gpu`, not portable).

---

## 2. State space

GPU `g`'s **persistent** state:

```
S_persistent,g := { W_i^{(g)}, m_i^{(g)}, v_i^{(g)}  : i ∈ {0, …, L−1} }
                  ∪ { (q, p) replicas (current minibatch) }
                  ∪ { rng_state (replicated, deterministic across ranks) }
```

GPU `g`'s **transient** state during layer `i` active window (one layer at a time):

```
S_transient,g,i := { Ŵ_i (full layer weights, freshly broadcast),
                     ∂Ŵ_i := g_i (full layer grad, accumulated locally on backward),
                     scratch (LN residuals, attn probs, …) }
```

The transient state's lifetime is one layer's compute window; it is **discarded immediately** after the gradient is reduce-scattered. At any instant, only one (or two, with comm overlap) layer's `Ŵ_i` exists on the GPU.

**Memory budget (per GPU at `n_gpu = 4, m = 4096, dFFN = 16384, L = 96` flagship 72 B target):**

Per layer `P_i = 4m² + 2m·dFFN ≈ 201·10⁶` params; per-shard `P_i/n_gpu ≈ 50·10⁶` params (100 MB BF16). Adam shards with FACE/MFIO/Kahan-v trims: ≈ 10 MB per layer.

| Component | Per-GPU bytes |
|---|---:|
| Weight shards (BF16) all 96 layers | 9.6 GB |
| Adam state shards (FACE+MFIO+Kahan-v) | 0.96 GB |
| Embedding shards (FACE-compressed) | 0.5 GB |
| **Persistent sub-total** | **11.06 GB** |
| `Ŵ_i` transient full-layer buffer | 0.4 GB |
| `g_i` transient full-layer grad | 0.4 GB |
| `(q, p)` replicas | 0.067 GB |
| Activation / LN / attn scratch | 0.5 GB |
| CUDA workspace | 1.5 GB |
| **Transient sub-total** | **2.87 GB** |
| Overlap pipeline (extra `Ŵ_i + g_i`) | 1.6 GB |
| **Total worst-case (overlapped)** | **15.53 GB / 16 GB** |

**Headroom 0.47 GB tight.** For 72 B target, default to **single-layer pipeline (no overlap)** at 13.93 GB, accepting throughput cost. Overlap enabled only at smaller models (e.g. 36 B at `n_gpu = 4`).

---

## 3. Evolution law (per training step)

A VESTA-CHIRON training step processes one minibatch. Below, `g ∈ {0, …, n_gpu−1}` is the local rank; operations are performed by every GPU unless tagged "owner only".

### 3.1 Forward pass

```
for i = 0, 1, …, L−1:
    ŴLayerBroadcast(i):                  # Issue: B_{owner(i)}(W_i^{(owner)}) → Ŵ_i on all GPUs
        owner = i mod n_gpu             # round-robin shard owner; in practice every GPU owns 1/n of every layer
        Ŵ_i ← Broadcast(W_i^{(g)} if g == owner else recv-buf, root=owner, stream=s_W)

    Synchronize(s_W → s_C)              # wait for Ŵ_i to land

    LayerForward(i, q, p) → (q', p'):
        # Standard CHIRON layer: shear (q, p) ↦ (q, p + Y(q; Ŵ_i))
        Y ← attention(q; Ŵ_i.{Q,K,V,O}) + MLP(q; Ŵ_i.{FFN_in, FFN_out})
        p' ← p + Y
        q' ← q  # CHIRON: q passes through this shear; symmetry on next layer

    Discard Ŵ_i  on GPUs g ≠ owner
    Owner-only: keep W_i^{(g)} (it never left the persistent store)

    (q, p) ← (q', p')
```

**Sharding implementation note.** Every GPU owns *one shard of every layer*: GPU `g` owns `W_i[g·P_i/n_gpu : (g+1)·P_i/n_gpu]` for all i. The "owner" of layer `i` is not a single GPU; the broadcast is **all-gather** — every GPU contributes its slab and `Ŵ_i = concat(W_i^{(0)}, …, W_i^{(n_gpu−1)})` lands on every GPU. The "broadcast" notation is shorthand for `ncclAllGather`.

### 3.2 Backward pass

CHIRON backward proceeds *layer-by-layer in reverse*. For each layer, three things happen: (a) re-materialise `(q_i, p_i)` via the inverse walk; (b) compute layer backward yielding `∂L/∂Ŵ_i` and `(dq_i, dp_i)`; (c) reduce-scatter the gradient to the shard owner.

```
# Initial state: (q_L, p_L), (dq_L, dp_L) from loss
for i = L−1, L−2, …, 0:
    AllGather(W_i):  Ŵ_i ← AllGather(W_i^{(0..n_gpu−1)})

    # Inverse walk recomputes layer-i input from its output
    LayerInverse(i, q_{i+1}, p_{i+1}; Ŵ_i) → (q_i, p_i)

    # Layer backward (analytic shear-Jacobian; see CHIRON_framework Theorem 4)
    LayerBackward(i, q_i, p_i, dq_{i+1}, dp_{i+1}; Ŵ_i)
        → (dq_i, dp_i, ∂L/∂Ŵ_i)

    # ZeRO-3 gradient reduction: each GPU's local ∂L/∂Ŵ_i is partial; sum + scatter
    ReduceScatter(∂L/∂Ŵ_i):  g_i^{(g)} ← (1/n_gpu) ∑_{g'} ∂L/∂Ŵ_i (slab g)
    # Note: each GPU ends up holding only the gradient for its own shard

    Discard Ŵ_i and ∂L/∂Ŵ_i (full-layer transients)
    (dq_{i+1}, dp_{i+1}) ← (dq_i, dp_i)
```

**Why ReduceScatter, not AllReduce.** Every GPU has a partial gradient for the full `Ŵ_i` (because every GPU saw the full `Ŵ_i` and a different microbatch slice of `(q_i, p_i)`). We need to sum across GPUs *and* scatter so each GPU keeps only its shard's gradient — `ncclReduceScatter` does both in one call.

**Data parallelism is mandatory.** Activations are replicated, so getting batch-scaling from `n_gpu` requires each GPU to consume a *different* microbatch (DP). VESTA-CHIRON without DP is wasted compute. **The correct framing: VESTA = DDP with sharded state**, where the per-rank DP shard *also serves as the optimiser-state shard owner*.

### 3.3 Optimizer step (Adam)

Each GPU updates only its own shard:

```
for i = 0, 1, …, L−1:
    m_i^{(g)} ← β_1 m_i^{(g)} + (1 − β_1) g_i^{(g)}
    v_i^{(g)} ← β_2 v_i^{(g)} + (1 − β_2) (g_i^{(g)})²
    ĥ_i^{(g)} ← m_i^{(g)} / (√v_i^{(g)} + ε)
    W_i^{(g)} ← W_i^{(g)} − lr · ĥ_i^{(g)}
    # Apply Kahan compensation on v_i^{(g)} (surprise-#17 fix), FACE/MFIO compression as shipped
```

**No cross-GPU sync during Adam.** Each GPU's shard update is independent. This is one of ZeRO-3's principal wins relative to fully-sharded compute (e.g. tensor parallel needs cross-GPU operations during backward; ZeRO-3 keeps the optimiser step entirely local).

### 3.4 Async overlap — critical for throughput

Without overlap, forward serialises broadcast + compute per layer; throughput collapses to (broadcast time + compute time) per layer. Overlap pipelines layer `i+1`'s broadcast against layer `i`'s compute:

```
# Issue layer 0 broadcast at step start
Broadcast(W_0) on stream s_W

for i = 0, 1, …, L−2:
    Wait(Ŵ_i landed)                # synchronize s_W → s_C
    Issue Broadcast(W_{i+1}) on s_W  # overlap with compute below
    LayerForward(i; Ŵ_i) on s_C

LayerForward(L−1; Ŵ_{L−1}) on s_C  # last layer: compute only

# Symmetric overlap on backward
```

With overlap, per-layer wall-clock = max(broadcast time, compute time). At `m = 4096, n_gpu = 4`:
- Broadcast time per layer: `400 MB / 25 GB·s⁻¹` (PCIe) = **16 ms** (infeasible) | `400 MB / 300 GB·s⁻¹` (NVLink-3) = **1.3 ms** (feasible).
- Compute time per layer (post-#42+#43+#44): ~2.5 ms.

Overlap regime under NVLink: compute-bound (1.3 ms < 2.5 ms), comm hidden. Step time ≈ L · 2.5 ms = 240 ms per step at L = 96.

Overlap regime under PCIe-4.0: comm-bound (16 ms > 2.5 ms), comm visible. Step time ≈ L · 16 ms = 1530 ms per step at L = 96 — **unacceptable**.

---

## 4. Theorem A — sharded-forward correctness

**Statement.** For deterministic NCCL AllGather (fixed reduction tree) with bf16 transport, VESTA-CHIRON forward produces `(q_L, p_L)` bit-identical to single-GPU CHIRON forward.

**Proof sketch.** AllGather is pure data movement (no floating-point reduction). With fixed shard partition, the reconstructed `Ŵ_i` on every rank equals single-GPU `W_i` byte-for-byte. Subsequent layer forward is deterministic by CHIRON_framework Theorem 1. Therefore output matches single-GPU. ∎

**Caveat: rank-count dependence.** NCCL's algorithm choice depends on world size; bit-exact under fixed `n_gpu` but not portable across `n_gpu`. **VESTA breaks rank-portable determinism** — an acknowledged regression vs single-GPU and HYDRA.

---

## 5. Theorem B — CHIRON inverse walk under sharding

**Statement.** When each layer's full weights are reconstructed via AllGather before inversion, the inverse walk produces `(q_i, p_i)` byte-identical to single-GPU inversion modulo `O(L · ε_{bf16})` forward-trajectory drift (same bound as single-GPU; **no additional drift from sharding**).

**Proof.** CHIRON's symplectic shear inversion is `p_i := p_{i+1} − Y(q_{i+1}; W_i); q_i := q_{i+1}`. By Theorem A, reconstructed `Ŵ_i` is bit-identical to single-GPU `W_i`. The forward output `(q_{i+1}, p_{i+1})` is replicated on every GPU. Therefore the inversion arithmetic is identical and `(q_i, p_i)` is bit-identical to single-GPU. The inverse walk does not introduce additional drift (exact arithmetic on the same bf16 values, `O(1)` ops per layer). ∎

**Consequence.** VESTA-CHIRON's gradient is bit-identical to single-GPU's modulo (a) bf16 reduction in `ReduceScatter` (sub-leading), and (b) rank-count-dependent NCCL algorithm choice.

---

## 6. Composition with #42 SCFA

SCFA replaces per-layer attention `O(T²m)` with `O(T·k·m + k²·d_H·n_H)` via a per-layer sequence-spectral basis `B_ℓ ∈ ℝ^{T×k}`, `k=64`. `B_ℓ` is small: `T·k = 65 k floats ≈ 130 KB BF16` per layer.

**Sharding decision: replicate `B_ℓ` on every GPU.** Sharding 130 KB would cost a per-layer mini-AllGather for negligible memory benefit. Run the basis-update optimizer step on every rank in lockstep with a global AllReduce of `B_ℓ`'s gradient.

**Critical finding: SCFA flips VESTA into comm-bound regime.** SCFA cuts per-layer compute from 2.5 ms to 1.0 ms; under NVLink-3 comm time 1.3 ms now exceeds compute. SCFA's 2.5× single-GPU speedup is **clipped to 1.9× under VESTA at NVLink-3**. NVLink-4 (0.44 ms comm) restores the compute-bound regime.

**Net headline.** VESTA × SCFA at NVLink-3: 1.9 × 4 = **7.6× effective scaling**. HYDRA × SCFA: 6.5 × 2.3 = **15×** — HYDRA is 2× more efficient under SCFA at the same hardware.

---

## 7. Composition with #43 ORION

ORION introduces slow-mode basis `V_t ∈ Stiefel(d, r)`, `r=4`, anchored every `K=20` steps. Off-anchor cost: `O(r²)`. On-anchor: `(3 + 2r) F` for full F+B + `r` HVPs.

**Sharding decision: shard V analogously to weights.** At `d = 18·10⁹, r = 4`: `V_t` is 72 GB BF16 — must shard regardless. Shard `V_t^{(g)}` partitioning matches weight partitioning (every parameter slot's `r` `V`-entries belong to the same shard owner). Reduced coordinate `α = V^T θ ∈ ℝ^r` is 16 floats; replicate via tiny `r`-AllReduce.

**Off-anchor step is pure-local: no cross-GPU traffic.** Each GPU updates its slice of `θ_⊥` from its local reduced coordinate. **Steps K−1 = 19 of every 20 avoid all cross-GPU comm.** This is where ORION × VESTA shines.

**Net headline at NVLink-3:** VESTA × ORION × SCFA = `1.9 × 5.5 × 4 ≈ 42×`. HYDRA × ORION × SCFA = `6.5 × 5.5 × 2.3 ≈ 82×`. **HYDRA still ahead for compute scaling, but the gap closes** because ORION's K-window amortises VESTA's heavy comm.

---

## 8. Composition with #44 MELT

MELT factors FFN weights as TT cores `G_1, G_2`, `ρ=8`: 1.3 MB per layer × 96 layers = 125 MB total.

**Sharding decision: replicate, do not shard.** Sharding TT cores would force a per-layer mini-AllGather; fixed-cost overhead (~50 µs/launch) dominates the actual transfer (~1.1 µs). Replicate on all GPUs and use a global AllReduce on the TT-core gradient (DDP pattern). 0.125 GB total fits within the 0.47 GB headroom.

**Compute composition.** MELT cuts per-layer compute from 1.0 ms to 0.66 ms (post-SCFA). Comm time 1.3 ms remains the bottleneck at NVLink-3.

**Memory composition.** MELT's 205× FFN compression frees 3.55 GB; this is exactly what enables the 72 B target at `n_gpu = 4` to fit.

**SCFA × MELT independence.** SCFA assumes low-rank sequence subspace; MELT assumes low-rank FFN weight. Independent assumptions, multiplicative composition.

**Net composition table (all four at NVLink-3):**

| Stack | Memory ceiling | Per-step throughput vs single-GPU |
|---|---:|---:|
| Single-GPU + #42 + #43 + #44 | 18 B | 30× |
| **VESTA × #42 × #43 × #44 at n_gpu=4** | **72 B** | **≈ 80–110× honest** (raw 168×, ×0.5 comm overhead) |
| HYDRA × #42 × #43 × #44 at n_gpu=8 | 144 B | 250–400× |

**HYDRA is 2.5–3× more compute-efficient.** VESTA wins only when model-size scaling dominates the wall-clock budget.

---

## 9. Performance analysis at typical configs

Per-layer comm = `400 MB / B_link`; per-layer compute = `0.66 ms` (post-#42+#43+#44).

| Config | Per-layer comm | Per-layer wall | Step time (anchor) | Effective per-step (ORION K=20) | vs single-GPU |
|---|---:|---:|---:|---:|---:|
| NVLink-3 (300 GB/s), m=4096, L=96 | 1.33 ms (comm-bound) | 1.33 ms | 389 ms | 24.2 ms | **1.6×** |
| NVLink-4 (900 GB/s), m=4096, L=96 | 0.44 ms | 0.66 ms (compute-bound) | 192 ms | 14.4 ms | **2.7×** |
| PCIe-4.0 (25 GB/s), m=4096, L=96 | 16 ms | 16 ms | 3 070 ms | n/a | **0.003× — infeasible** |
| NVLink-3, m=2048, L=53 (shipped) | 0.33 ms | 0.33 ms (comm-bound) | 35 ms | 2.7 ms | **1.4×** |

**Two conclusions:**

1. **PCIe-4.0 is structurally infeasible** at flagship `m=4096, L=96`: 320× slower than single-GPU; no software mitigation can recover this.
2. **VESTA-CHIRON is most efficient at very large `m`** because broadcast scales `O(m²)` while compute scales `O(T·m²)` — compute/comm ratio grows linearly with `T`. This aligns with its purpose (training extremely large models on long sequences).

---

## 10. Comparison with HYDRA and TENSOR-CHIRON

| Axis | HYDRA (#45-A) | TENSOR-CHIRON (#45-B) | VESTA-CHIRON (#45-C) |
|---|---|---|---|
| Parallelism dimension | layers | features within layers | parameters within layers |
| Compute scaling per GPU | n_gpu (linear via PP) | n_gpu (linear via TP) | **1× (no compute scaling)** |
| Memory scaling per GPU | n_gpu (each GPU has 1/n layers) | n_gpu (each GPU has 1/n features) | **n_gpu (each GPU has 1/n parameters)** |
| Per-layer comm volume | O(T·m) (boundary activations) | O(T·m) per layer (AllReduce on activations) | **O(m²) per layer (broadcast on weights + reduce-scatter on grads)** |
| Per-step comm volume at flagship | 3.6 GB | ≈ 4–5 GB | **28 GB** |
| Bubble fraction | 18 % at μ=4n_gpu | 0 % | 0 % |
| NVLink-3 step time at 72 B | 75 ms | 100 ms | 389 ms |
| PCIe-4.0 viability | marginal | marginal | **infeasible** |
| CHIRON synergy | **strong** (segment-local inverse walk; segment activation memory unchanged from single-GPU) | weak (TP doesn't exploit reversibility; activations partitioned but inverse walk needs full activations) | weak (replicated activations + sharded weights ≠ a CHIRON-specific lever; same as ZeRO-3 on a non-reversible transformer) |
| Implementation complexity | high (1F1B + boundary protocol + per-stage Adam partition) | high (every kernel needs TP-aware reshape + AllReduce) | **medium-high** (AllGather/ReduceScatter wrappers + async overlap; existing DDP infra reusable) |
| Determinism | bit-exact (same as single-GPU) | bit-exact modulo TP-partition order | **rank-fixed only** (NCCL algorithm depends on world size) |
| Failure mode | one stage dies → pipeline hangs | one rank dies → step aborts | one rank dies → step aborts |
| Honest "best at" | maximum compute scaling at fixed N | maximum kernel-fusion efficiency | **maximum model size at fixed N** (when comm budget is plentiful) |

**Key qualitative claim.** VESTA-CHIRON is the **simplest** of the three to implement (existing DDP infrastructure provides AllGather/ReduceScatter primitives) but has the **weakest CHIRON-specific synergy** and the **heaviest comm volume**. It wins on memory scaling and loses on compute scaling.

**Selection priority:** (1) HYDRA if pipeline depth ≤ 16; (2) TENSOR-CHIRON for very large `T`; (3) VESTA-CHIRON when model-size dominates *and* NVLink is available.

**In production, the three compose: VESTA + HYDRA + DDP = 3-D parallelism** (Megatron-DeepSpeed architecture). VESTA is most valuable as a *component* of a 3-D stack, not as a standalone choice.

---

## 11. Failure modes and mitigations

| ID | Failure mode | Mitigation |
|---|---|---|
| F1 | **PCIe-4.0 cluster infeasible** (1.1 s/step at flagship; 320× slower than single-GPU) | NVLink-3 or NVLink-4 hardware required; non-negotiable; do not deploy on PCIe |
| F2 | NVLink-3 30–50 % comm overhead | Async overlap via 2 CUDA streams; expect 0.5–0.7× per-GPU throughput |
| F3 | NCCL bf16 reduction non-determinism across ranks | Document as accepted regression (vs single-GPU determinism); pin `NCCL_ALGO=Tree`, `NCCL_PROTO=Simple` |
| F4 | Adam-state shard misalignment after RLG layer-grow | Pause + re-shard on growth event; one-time cost; checkpoint immediately |
| F5 | Single-GPU failure aborts step | Checkpoint every 200 steps; resume from last checkpoint; ORION's K-window absorbs single-step loss |
| F6 | ORION basis `V` shard bf16 drift | Periodic full-precision re-anchor (fp32 V for one anchor pass every 1k steps); same mitigation as ORION single-GPU |
| F7 | Mixed bf16 reductions in ReduceScatter cause loss curve drift | Use `NCCL_ALLGATHER_ALGO=Tree`; profile with iter-185 EMA divergence guard |
| F8 | Communication-compute overlap stream synchronisation deadlock | Single comm stream (s_W) + single compute stream (s_C), event-based hand-off; deterministic launch order |
| F9 | Weight-shard slab boundary cuts a tensor mid-row | Acceptable; ZeRO-3 deliberately ignores tensor structure for simplicity; the full `Ŵ_i` is reconstructed by AllGather before any compute |
| F10 | Memory headroom tight (0.47 GB at 72 B target) | Disable comm-compute overlap for headroom; accept ~30 % throughput regression; this is the recommended default for 72 B |

**Most serious: F1.** PCIe-4.0 infeasibility is structural; no software mitigation can compensate for the bandwidth-deficit. This sharply limits VESTA-CHIRON's deployment surface to NVLink-equipped servers (DGX, HGX, or 4080 SUPER/H100 with NVLink bridges). On commodity 4×4080 PCIe rigs, **HYDRA is the only viable distributed candidate.**

---

## 12. Concrete primitives — NCCL wrappers and async overlap

### 12.1 Module layout

```
Backend/Machine Learning/Networks/
├── vesta/
│   ├── vesta_config.h, vesta_shard.h/.cpp     — partition map, shard arithmetic
│   ├── vesta_state.h/.cpp                     — per-rank persistent W/m/v shard storage
│   └── vesta_sched.h/.cpp                     — per-step orchestrator
└── cuda/gpu_nccl.h/.cu                        — NCCL primitives (extends #45-A)
```

### 12.2 Headers and primitives

```cpp
// VestaState: holds persistent shards + transient full-layer buffers
class VestaState {
public:
    VestaState(int n_gpu, int rank, const std::vector<int64_t>& layer_param_counts);
    GpuBuffer<uint16_t>& W_shard(int i);       // persistent, bf16
    GpuBuffer<uint16_t>& m_shard(int i);
    GpuBuffer<uint16_t>& v_shard(int i);
    GpuBuffer<uint16_t>& W_full();             // transient, sized to max layer
    GpuBuffer<uint16_t>& g_full();
    const ShardLayout& layout(int i) const;
};

// NCCL wrappers
namespace glades::gpu::nccl {
    void allGatherBf16(GpuBuffer<uint16_t>& shard, GpuBuffer<uint16_t>& full,
                       size_t numel_shard, cudaStream_t s);
    void reduceScatterBf16(GpuBuffer<uint16_t>& full, GpuBuffer<uint16_t>& shard,
                           size_t numel_shard, cudaStream_t s);          // SUM op, optional 1/n pre-scale
    void allReduceFp32(float* buf, size_t count, cudaStream_t s);        // for ORION α
}
```

### 12.3 Async-overlap forward orchestrator

```cpp
void VestaScheduler::runForward(NNetwork* net, GpuBuffer<float>& q, GpuBuffer<float>& p) {
    cudaStream_t s_W = streams_.commStream(), s_C = streams_.computeStream();
    cudaEvent_t  ev  = events_.layerLanded();

    glades::gpu::nccl::allGatherBf16(state_.W_shard(0), state_.W_full(),
                                     state_.layout(0).numel_shard, s_W);
    cudaEventRecord(ev, s_W);

    for (int i = 0; i < L_ - 1; ++i) {
        cudaStreamWaitEvent(s_C, ev, 0);
        glades::gpu::nccl::allGatherBf16(state_.W_shard(i + 1), state_.W_full_next(),
                                         state_.layout(i + 1).numel_shard, s_W);
        cudaEventRecord(ev, s_W);
        net->runLayerForward(i, state_.W_full(), q, p, s_C);
        state_.swapFullBuffers();
    }
    cudaStreamWaitEvent(s_C, ev, 0);
    net->runLayerForward(L_ - 1, state_.W_full(), q, p, s_C);
}
// Backward symmetric: AllGather W_i + LayerInverse + LayerBackward + ReduceScatter g_i.
// Adam: per-rank shard update only, no cross-GPU sync.
```

### 12.4 Implementation roadmap (~1 800 LOC, ~5 weeks)

| Phase | Scope | LOC | Validation |
|---|---|---|---|
| 1 | Shard layout + persistent state | 400 | unit: partitioning, byte-counts |
| 2 | NCCL primitives (AllGather, ReduceScatter, AllReduce) | 350 | 2-GPU smoke: round-trip byte-equality |
| 3 | Single-stream orchestrator (no overlap) | 500 | 2-GPU `L=4, m=512` matches single-GPU |
| 4 | Async overlap (s_W + s_C streams) | 250 | 2-GPU throughput within 1.4× single-GPU |
| 5 | Composition #42+#43+#44 + 4-GPU validation | 300 | 4-GPU 100-step loss matches within 1e-4 |

**Phase 1+2 are also prerequisites for "VESTA inside HYDRA" 3-D parallelism**, so this work is not lost if HYDRA is selected first.

---

## 13. Honest gap and selection guidance

**VESTA-CHIRON has the weakest CHIRON-specific synergy of the three #45 candidates.** The inverse walk under VESTA is just a broadcast pattern repeated during inversion — it does not exploit any structural property of the symplectic shears. Replace CHIRON with a non-reversible transformer and VESTA's design changes by zero lines of code. **VESTA is essentially CHIRON-agnostic** — standard ZeRO-3 bolted onto whatever happens to be running.

By contrast: HYDRA exploits CHIRON's segment-local invertibility for zero per-stage activation memory; TENSOR-CHIRON's feature-axis split requires per-layer activation AllReduce that interacts with reversibility constraints. **VESTA's lack of architectural lever is its principal shortcoming** for a "CHIRON-specific paradigm shift."

**This is honest framing, not a defect.** ZeRO-3 is celebrated *because* it is architecture-agnostic; VESTA-CHIRON inherits that universality. But for #45's purpose of leveraging CHIRON, VESTA underperforms.

**Selection recommendation:**

1. **#45 selection: HYDRA** — compute + memory scaling, strongest CHIRON synergy.
2. **Future #46 or beyond: VESTA-on-HYDRA** for 3-D parallelism (sharded state within pipeline stages, DDP across replicas). VESTA's universality becomes a feature in this composition.

**Position:** viable standalone #45 candidate, materially distinct in axis from HYDRA/TENSOR, but a stronger future-#46 composition candidate on top of #45-A HYDRA than as a standalone choice.

---

## 14. Summary

VESTA-CHIRON is ZeRO-3-style state sharding for CHIRON. Each GPU holds `1/n_gpu` of every layer's parameters, AllGather-gathers full layer weights before forward, computes the full layer on its replicated activation slice, and ReduceScatter-returns gradient shards on backward. Adam steps run locally per rank.

**Three honest claims:**

1. **72 B at n_gpu = 4 with NVLink-3.** Linear memory scaling; full single-step semantics; zero bubble.
2. **30–50 % comm overhead under NVLink-3.** Per-GPU throughput 0.5–0.7× single-GPU; net wall-clock ≈ 2.5× vs single-GPU. PCIe-4.0 is structurally infeasible.
3. **HYDRA is 2–3× more compute-efficient** at the same hardware. VESTA wins only when model size dominates and pipeline depth would be excessive.

**Composition smooth and multiplicative** with #42 SCFA, #43 ORION, #44 MELT; ORION's K-window amortises VESTA's heavy comm because off-anchor steps avoid all cross-GPU traffic.

**Roadmap position.** Standalone, third behind HYDRA and TENSOR. As composition partner: VESTA-on-HYDRA in a 3-D parallelism stack (post-#45 paradigm shift) becomes the natural extension once cluster size demands ZeRO-3-style amortization. **Recommend HYDRA for #45; revisit VESTA at #46 for 3-D composition.**
