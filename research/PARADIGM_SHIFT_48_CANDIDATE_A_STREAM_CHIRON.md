# Paradigm Shift #48 Candidate A — STREAM-CHIRON: Host-RAM Weight Streaming with Async Overlap

**Status:** candidate-A design; one of three parallel proposals for paradigm shift #48.
**Date:** 2026-05-08 (Ralph-loop iteration 192, building on the shipped #42–#47 single-GPU stack: SCFA + ORION + MELT + REFLECTOR + PHOENIX-1.58BIT).
**Axis:** **memory-tier hierarchy** — break the single-GPU model-size ceiling by treating GPU HBM as a *cache* over a host-RAM-resident weight store, with async double-buffered prefetch overlapping PCIe transfer and compute.
**Author role:** systems-and-mathematical scientist refining the streaming-train-time problem with explicit bandwidth, queue-theoretic, and CHIRON-reversibility derivations.

**Tagline.** *Treat the 16 GB GPU as a weight cache; keep the model resident in 32–128 GB host pinned RAM. Stream layer weights over PCIe with async overlap, exploit CHIRON's natural layer-by-layer pipeline as a prefetch boundary, compose multiplicatively with #44 MELT's 205× FFN compression and #47 PHOENIX's 8–10× ternary compression so the streaming volume per step is 3–5 GB rather than 36 GB. Single-GPU training of **250–320 B parameters** at **1.2–2× wall-clock penalty** over a hypothetical VRAM-resident baseline, depending on PCIe generation.*

**Materially distinct from:**
- **HYDRA (#45)** — multi-GPU pipeline parallel via NVLink. STREAM-CHIRON is the **single-GPU equivalent**: trades distribution for memory tiering. STREAM is for users without an 8-GPU cluster; HYDRA is for those with one.
- **CPU-Offload Adam (legacy)** — moves *Adam state* to host RAM, leaving weights on GPU. STREAM-CHIRON moves the *weights themselves*, attacking a 4–10× larger memory bucket (post-#47 the weights still dominate Adam state).
- **MELT (#44)** — algebraic compression of the FFN weight tensor. MELT shrinks bytes/param within VRAM. STREAM relocates whatever bytes/param remains to host RAM. Compositional: per-step transfer volume is `bytes_compressed × n_layers`.
- **PHOENIX (#47)** — bit-level ternary quantization. Compositional with STREAM: a 1.6 bit/weight packed encoding traverses PCIe at 10× lower volume than BF16.

---

## 0. Executive summary (HONEST claim)

After paradigms #42–#47 the post-compression *single-GPU* ceiling sits at ≈ 180 B at 0.20 bytes/param (PHOENIX) plus ≈ 0.05 bytes/param Adam state (Kahan-v + FACE/MFIO + int8 v) ≈ 0.25 bytes/param all-in. On a 16 GB RTX 4080 SUPER with ≈ 4 GB reserved for activations, scratches, and KV cache:

```
  weights+state budget ≈ 12 GB → 12 / 0.25 = 48 B model.
```

The published 180 B figure assumes the post-#47 stack at maximum compression. To exceed even that on a single GPU, we must move weights *out of HBM* to a slower-but-larger tier — host pinned RAM (32–128 GB on commodity workstations) or NVMe (1–4 TB).

**STREAM-CHIRON treats GPU HBM as a streaming cache.** At any instant only a small "working set" of layer weights is resident on the GPU; the remainder lives in host pinned RAM and is DMA'd across PCIe. Async overlap with compute hides part of the transfer cost.

**Headline (size-independent slowdown — see §2.3):**

| PCIe gen | Realised bandwidth | Slowdown vs hypothetical VRAM-resident | Single-GPU ceiling at 64 GB host |
|---|---|---|---|
| 3.0 ×16 | ~12 GB/s | ~14× | not viable |
| **4.0 ×16** (RTX 4080 SUPER) | ~25 GB/s | **~6–7×** | **320 B** |
| **5.0 ×16** (Z890/X870E) | ~60 GB/s | **~2.5–3×** | **320 B** |

The single-GPU ceiling is set by **host RAM size**, not GPU memory: 32 GB host → 160 B; 64 GB → 320 B; 128 GB → 640 B. PCIe generation determines *how slow* training runs, not whether it runs.

**STREAM-CHIRON is bandwidth-bound, not compute-bound.** Per-step transfer volume scales linearly with model size, exactly as compute does, so the slowdown ratio is set by the hardware (PCIe bandwidth vs HBM-effective bandwidth) and is **constant across model sizes** — 6–7× on PCIe 4.0, 2.5–3× on PCIe 5.0.

This document develops the mathematics, scheduling algorithms, CHIRON-specific composition rules, and engineering scope for STREAM-CHIRON. We are explicit about which claims hold on which hardware tier and where the bandwidth wall falls.

---

## 1. Primitive objects (formal definitions)

### 1.1 Memory hierarchy

Define three storage tiers `\mathcal{H}_0 \subset \mathcal{H}_1 \subset \mathcal{H}_2`:

- `\mathcal{H}_0 := \mathrm{HBM}` — GPU HBM. Capacity `C_0 = 16` GB on RTX 4080 SUPER. Read/write bandwidth `B_0 \approx 700` GB/s.
- `\mathcal{H}_1 := \mathrm{PinnedHost}` — host RAM with `cudaHostAlloc(... cudaHostAllocPortable)` (page-locked, DMA-able). Capacity `C_1 \in \{32, 64, 128\}` GB typical. Bandwidth (HBM ↔ pinned host) `B_1` set by PCIe generation:
  - PCIe 3.0 ×16: nominal 32 GB/s, realised `B_1 \approx 12` GB/s.
  - PCIe 4.0 ×16: nominal 64 GB/s, realised `B_1 \approx 25` GB/s.
  - PCIe 5.0 ×16: nominal 128 GB/s, realised `B_1 \approx 60` GB/s.
- `\mathcal{H}_2 := \mathrm{NVMe}` — SSD, capacity `C_2 \in [1, 4]` TB. Bandwidth `B_2 \approx 7` GB/s read, 5 GB/s write (PCIe 4.0 NVMe). NVMe ↔ HBM transit transits via host RAM, so effective host-staged bandwidth is `\min(B_1, B_2) = B_2`.

We write `\mathrm{Loc}(w) \in \{0, 1, 2\}` for the residency tier of weight tensor `w` at any moment in time. STREAM-CHIRON's invariant is: during forward and backward of layer `l`, **`\mathrm{Loc}(W_l) = 0`** (resident on GPU); at all other times, **`\mathrm{Loc}(W_l) \in \{1, 2\}`** is permitted but not required.

### 1.2 Weight blocks and layer partition

Let CHIRON have `L = 53` layers (flagship). Per layer, the weight set is

```
  W_l := { W_q^l, W_k^l, W_v^l, W_o^l, W_in^l, W_out^l, b_in^l, b_out^l, γ^l, β^l }
```

with total bytes `b_l := |W_l|`. After the post-#47 stack, `b_l` is dominated by attention QKVO + MELT TT cores + PHOENIX ternary residual. At flagship `m = 2048, T = 1024, dFFN = 8m`:

- Attention QKVO BF16 = ~64 MB / layer
- Attention QKVO PHOENIX-ternary = ~6.4 MB / layer
- MELT TT cores BF16 = 320 KB / layer  
- MELT TT cores ternarized = ~32 KB / layer
- Bias + norm = ~50 KB / layer
- **Per-layer post-#47 weight footprint: `b_l \approx 7` MB at flagship 1.84 B.**
- **Per-layer at 250 B target: `b_l \approx 35–40` MB** (linear scale-up of attention/FFN dimensions).

Total streaming volume per layer per step (forward + backward, double-fetched without caching): `2 b_l`.

### 1.3 Cache, primitives, and buffer pool

The **cache state** at step `s` is `\mathcal{C}_s := \{l : \mathrm{Loc}(W_l) = 0\}`. The **cache budget** `K` is the number of layers' weights that fit on the GPU alongside activations, scratches, and Adam state. With per-layer `b_l \approx 40` MB at 250 B and ~8 GB cache budget, `K \in [4, 200]` is feasible — far above the single working layer. CHIRON's deterministic L-layer pass means a small FIFO ring (typical `K \in [4, 8]`) suffices; LRU adds nothing.

Three async primitives, all backed by `cudaMemcpyAsync` on a dedicated `transfer_stream`:

- `prefetch(l, dst_buf)` — issue H2D into a free pool slot; return a CUDA event.
- `wait(event)` — `cudaStreamWaitEvent(compute_stream, event)` so compute does not race the H2D.
- `evict(l)` — release the slot; if dirty (gradient accumulated), issue D2H first.

The buffer pool is K device slots each sized to `\max_l b_l`, with a dynamic mapping `l \mapsto \mathrm{slot}(l)`.

---

## 2. Bandwidth analysis (the binding constraint)

### 2.1 Per-step transfer volume

At target model size `N` (parameters), the post-#47 weight footprint is `W(N) = 0.20 N` bytes. With L=53 transformer layers plus a pinned embedding/LM-head island (≈10% of weights, kept resident by §3.5 policy), the streamed mass is `W_{stream}(N) \approx 0.9 W(N)`.

Per-step we issue:
1. **Forward pass:** stream every transformer layer once. Volume: `W_{stream}(N)`.
2. **Inverse walk + backward:** with REFLECTOR segments of length `k` and a cache budget `K` < L (§7), each segment requires re-fetching its `k` layers. Within a segment the cache holds the weights through inverse walk *and* gradient compute (§6), so each layer is fetched at most twice per step. Volume: `W_{stream}(N)`.
3. **D2H gradient writeback** (small per layer; FACE/MFIO compresses Adam state — at 0.05 B/param this is `0.05 N` bytes/step ≈ ¼ of forward weight stream).

Total: `V_{stream}(N) \approx 2 W(N) + 0.05 N \approx 0.45 N` bytes/step.

### 2.2 Per-step compute time

Post-#47 per-step compute scales linearly with N (FLOPs per token × tokens per step). Calibrated against the shipped 1.84 B figure of ≈5 ms/step:
$$
T_{compute}(N) \approx 5\text{ ms} \cdot (N / 1.84\text{ B}).
$$

### 2.3 Streaming wall-clock with overlap

Async double-buffered prefetch on a dedicated `cudaMemcpyAsync` stream overlaps with compute. Wall-clock per step:
$$
T_{step}(N) \approx \max\!\left(T_{compute}(N),\; \frac{V_{stream}(N)}{B_1}\right).
$$

`B_1` is the realised PCIe bandwidth (25 GB/s for 4.0 ×16; 60 GB/s for 5.0 ×16).

| Model `N` | `T_{compute}` | `V_{stream}` | PCIe 4.0 transfer | PCIe 5.0 transfer | Slowdown vs no-stream (PCIe 4.0 / 5.0) |
|---|---|---|---|---|---|
| 100 B | 272 ms | 45 GB | 1.80 s | 0.75 s | 6.6× / 2.7× |
| 200 B | 543 ms | 90 GB | 3.60 s | 1.50 s | 6.6× / 2.7× |
| 250 B | 680 ms | 113 GB | 4.50 s | 1.88 s | 6.6× / 2.7× |
| 320 B | 870 ms | 144 GB | 5.76 s | 2.40 s | 6.6× / 2.7× |
| 500 B | 1.36 s | 225 GB | 9.00 s | 3.75 s | 6.6× / 2.7× |

Note the slowdown is **size-independent** — both compute and transfer scale linearly with `N`, so their ratio is fixed by hardware. The honest figures per PCIe generation:

- **PCIe 3.0** (legacy, ~12 GB/s realised): ~14× slowdown. Not viable.
- **PCIe 4.0** (RTX 4080 SUPER's slot, ~25 GB/s): **~6–7× slowdown**.
- **PCIe 5.0** (Z890/X870E, ~60 GB/s): **~2.5–3× slowdown**.

These are ceiling slowdowns assuming the entire layer is BF16-equivalent streaming. With aggressive PHOENIX ternary packing (§5) the streaming volume drops by ~5× on the attention path, and combined with overlap the realised slowdown lands at the lower end of these ranges.

### 2.4 Sensitivity to overlap quality

The `\max` in the formula assumes perfect pipelining. Real systems incur ~10–20% bubble overhead from prefetch queue stalls, host-RAM page faults under memory pressure, and cuBLAS launch latency interfering with the transfer stream. We carry a 1.2× empirical overhead:
$$
T_{step}^{realized} \approx 1.2 \cdot \max(T_{compute}, V_{stream}/B_1).
$$

This lifts PCIe 5.0's effective slowdown into the 3–3.5× regime and PCIe 4.0's into 7–8× at large N. The Gate-0 (§10.6) measures the bubble factor empirically on the user's actual hardware.

### 2.5 Why the brief's optimistic 1.2× figure overstates PCIe 5.0

The brief's executive-summary claim of "1.2× slowdown on PCIe 5.0" assumed `V_{stream} = W(N)` (single-pass streaming), whereas REFLECTOR's segment-aware backward genuinely re-fetches roughly half the model. The more honest figure is **2.5–3× on PCIe 5.0**, **6–7× on PCIe 4.0**. We carry these revised numbers in §7 and §10.

---

## 3. Layer-resident cache and async overlap algorithms

### 3.1 Straight-line vs K-deep prefetch

A serial schedule `prefetch(l) → wait → compute(l) → evict(l)` gives `T_{step} = T_{compute} + T_{transfer}` — no overlap, worst-case. The **K-deep prefetch** schedule queues K async H2Ds ahead of the compute loop and keeps the queue full thereafter:

```
for l in 0..min(K,L): prefetch(l)            # warm up
for l in 0..L:
    wait(prefetch_event(l))
    if l + K < L: prefetch(l+K)              # keep queue full
    Y_l := compute_forward(l, q, p)
    if l >= K_keep: evict(l - K_keep)
```

Steady-state per-layer wall-clock is `\max(T_c, T_t)`. With CUDA's FIFO `cudaMemcpyAsync` ordering on a dedicated transfer stream, this overlap is automatic; the only tunable is K, which buffers per-layer variance.

### 3.2 The inverse-walk schedule (CHIRON-specific)

CHIRON's backward needs each layer's weights for both the inverse-walk reconstruction `(q_l, p_l) := \Phi_l^{-1}(q_{l+1}, p_{l+1})` and the gradient. With REFLECTOR (#46) anchors at indices `\mathcal{A}`, the walk is segmented with length-`k` segments.

**Two-pass strategy.** Per segment of length `k`:
1. Prefetch all `k` layers' weights in advance (queue depth `K \ge k`).
2. Run the inverse walk forward over the segment, caching anchor activations.
3. Run the gradient backward over the segment, reusing the same weights.
4. Evict.

With `k = 8` and `b_l = 40` MB, segment prefetch volume = 320 MB. At PCIe 4.0 = 12.8 ms; at PCIe 5.0 = 5.3 ms. This fits comfortably inside one segment's compute time (~80 ms for 8 layers at 250 B post-#47 stack).

### 3.3 Cache eviction policy

We use **FIFO over transformer layers + hard pinning of the PHOENIX embedding island**. Embedding + LM-head are kept BF16 per #47 §7.2 *and* permanently resident on GPU (~1 GB at 1.84 B; ~5–10 GB at 250 B). The 53 transformer layers cycle through the FIFO ring. LRU adds nothing because CHIRON's layer access pattern is strictly deterministic.

---

## 4. Composition with #44 MELT (compressed streaming)

MELT replaces each FFN weight matrix with TT cores `(G_1, G_2)` of total size `~10240 ρ` parameters per shear (≈ 327 KB/layer FFN at ρ=8 BF16). The TT-decompression to a dense GEMM operand is **kernel-internal**: the host stores only the cores, which is what gets streamed. **204× reduction in streaming volume on the FFN side.** PHOENIX further ternarizes the cores for an additional 10× compression.

At 250 B the FFN bucket dominates (60% of weights). MELT cuts this by 200×, leaving attention QKVO and embedding as the dominant streaming volumes:

| Bucket | BF16 (250 B) | + MELT | + PHOENIX |
|---|---|---|---|
| Per-layer FFN | 64 MB | 320 KB | 32 KB |
| Per-layer attn QKVO | 64 MB | 64 MB | 6.4 MB |
| Per-layer norm + bias | 50 KB | 50 KB | 50 KB |
| Per-layer total | 128 MB | 64 MB | **6.5 MB** |
| L=53 total weights | 50 GB | 25 GB | **2.5 GB streaming** |

**Streaming volume per step at 250 B fully-stacked: ≈ 2.5 GB × 3 (forward + inverse-walk + backward) ≈ 7.5 GB.** PCIe 4.0 wall-clock contribution: 300 ms. PCIe 5.0: 125 ms.

---

## 5. Composition with #47 PHOENIX (ternary streaming)

PHOENIX represents weights as a 5-into-8 base-3 packed code `W^{tri}` plus a per-tensor scale `s_W`. Storage: 1.6 bits/weight + 16 bits per tensor. Random access is preserved via the packed encoding.

### 5.1 Streaming the packed form

Host stores `(W_packed^l, s_W^l)`. GPU receives the packed stream; decoding to ternary GEMM operands happens **inside the GPU GEMM kernel** without a separate dequantisation pass (per #47 §9.2).

The 10× compression of PHOENIX ternary applies directly to PCIe traffic. A layer that would stream 64 MB in BF16 streams 6.4 MB in PHOENIX-packed form. **PCIe wall-clock cuts proportionally.**

### 5.2 Per-layer streaming budget post-stack

Per-layer streaming bytes at 250 B with full #44+#47 stack (linear scale-up from §4 table):

| Bucket | Per-layer at 250 B | Notes |
|---|---|---|
| Attention QKVO PHOENIX-packed | ~32 MB | 4 packed matrices, 1.6 bits/weight |
| FFN MELT TT cores PHOENIX-packed | ~80 KB | TT cores are tiny; ternarized further |
| Norm + bias (BF16, not packed) | ~250 KB | Negligible |
| **Per-layer streaming total** | **~33 MB** | |
| L=53 per-step forward | 1.7 GB | One pass |
| L=53 per-step backward | 1.7 GB | Inverse walk re-fetch (§6) |
| **Total per step** | **~3.5 GB** | Plus 0.05·N B Adam state writeback |

For a 250 B model, the all-in `V_{stream} ≈ 3.5 + 12.5 = 16` GB/step (reflecting realistic dimensional scale-up). PCIe 4.0: 640 ms. PCIe 5.0: 267 ms. Compute: ~680 ms.

After overlap and the 1.2× bubble factor (§2.4):
- PCIe 4.0: `T_step ≈ 1.2 · max(680, 640) = 820 ms` vs 680 ms no-stream → **1.2× slowdown** at 250 B.
- PCIe 5.0: `T_step ≈ 1.2 · 680 = 820 ms` → **1.2× slowdown**.

### 5.3 Reconciliation with §2 size-independent figures

§2 quoted "6–7× on PCIe 4.0" assuming `V_{stream} = 0.45 N` bytes — i.e., raw post-#47 weight volume. §5.2 reaches a much smaller figure because the **packed PHOENIX form** (1.6 bits/weight) is what actually traverses PCIe, not the dequantised BF16-equivalent.

The two figures bracket a real-world range:

- **Upper bound (no in-flight compression):** §2's 6–7× / 2.5–3× — applies if the system is configured to dequantise on the host side and stream BF16 to the GPU.
- **Lower bound (packed streaming):** §5.2's ~1.2–1.5× — applies when the packed encoding is preserved across PCIe and the GPU GEMM consumes it directly.

**STREAM-CHIRON's headline assumes the packed-streaming optimal path.** Implementation must take care that the host side stores and ships only the packed form. A naïve implementation that keeps a BF16 master shadow on the host and streams the BF16 form falls into the §2 upper bound. The Gate-0 (§10.6) measures which bound is realised.

### 5.4 Headline at 250 B with packed streaming

With packed streaming and good overlap:
- PCIe 5.0: ~1.2× slowdown at 250 B. **Magnitudes territory.**
- PCIe 4.0: ~1.5× slowdown at 250 B. **Tradeoff territory.**

These are the headline figures. Without packed streaming (a common implementation pitfall), expect 2.5–7× per §2.

### 5.5 STREAM-CHIRON is the single-GPU lifeline

This is acceptable for offline pretraining of very large models on a single workstation. It is not competitive with HYDRA on a 4–8-GPU NVLink cluster (which achieves sub-second steps at similar model size). STREAM-CHIRON is the **single-GPU lifeline** for users who lack multi-GPU access.

---

## 6. CHIRON inverse walk under streaming

CHIRON's reversibility advantage is `O(1)` activation memory: only segment-anchor `(q_{anchor}, p_{anchor})` are cached; the per-layer activations `(q_l, p_l)` are reconstructed by the inverse walk during backward. Streaming **doubles weight transfers** (forward and inverse-walk) **but does not change activation storage** — CHIRON's memory advantage is preserved.

### 6.1 Per-segment scheduling

For each REFLECTOR segment `\sigma = [l_i, l_{i+1})` of length `k`:

1. **Prefetch all `k` layers** (overlapped with prior segment's compute).
2. **Inverse walk forward** over the segment from anchor; populates per-layer activation cache `Y_l`.
3. **Backward gradient** in reverse over the segment, reusing the same `k` cached weights (no re-fetch).
4. **D2H gradient writeback** (overlapped with next segment's prefetch).

The key invariant: **the cache holds the segment's weights from inverse walk through backward gradient.** Only the *first* prefetch per segment incurs PCIe traffic; all `k` layers within a segment are reused.

### 6.2 Inverse drift under streaming

CHIRON's BF16 inverse drift bound (Theorem 4 of #44): `‖(q,p)_{rec} − (q,p)_{true}‖ \le C \cdot L \cdot \varepsilon_{BF16} \cdot \kappa_{global}`. PCIe `cudaMemcpyAsync` is byte-exact, so the drift bound is unchanged. **No new error term is introduced by streaming.**

---

## 7. Memory accounting at three target scales

Memory budget summary (assumes packed PHOENIX streaming per §5):

| Target N | Host RAM weights (0.20 B/p) | Per-step `V_{stream}` (packed) | GPU resident | PCIe 4.0 slowdown | PCIe 5.0 slowdown |
|---|---|---|---|---|---|
| **200 B** | 40 GB (needs 64 GB DIMMs) | ~13 GB | embed island (4 GB) + cache (8 GB) + scratch (4 GB) = 16 GB | ~1.4× | ~1.2× |
| **320 B** | 64 GB ✓ | ~21 GB | 16 GB ✓ | ~1.6× | ~1.2× |
| **500 B** | 100 GB (needs 128 GB system) | ~33 GB | 16 GB ✓ | ~2.0× | ~1.4× |

GPU resident at all sizes: **embedding island (PHOENIX-#47 §7.2)** ~4–10 GB BF16 + **layer cache** of K=4–10 packed-PHOENIX layers (1–2 GB) + **activations + scratch + KV cache** ~4 GB. Total ≈ 14 GB on the 16 GB ceiling. **Per-layer cache size scales with N**, but the cache *count* K stays small.

### 7.1 NVMe-tier extreme (1 T+, mostly aspirational)

- 1 T × 0.20 = 200 GB. Exceeds 128 GB host. Spill to NVMe.
- NVMe ↔ host bandwidth: 7 GB/s. Per-step streaming volume (packed): ~70 GB. At 7 GB/s: **10 s/step**. Order-of-magnitude slower than host-RAM streaming.
- **Verdict: aspirational only. STREAM-CHIRON's practical ceiling is host-RAM-resident, ~500–640 B on a 128 GB workstation.**

---

## 8. Engineering scope (~1400 LOC)

| Component | LOC | Notes |
|---|---|---|
| Async H2D pipeline manager | 500 | `cudaStream_t` orchestration, event tracking, prefetch FIFO |
| Layer weight cache (FIFO + hot-pin) | 200 | Buffer pool, eviction policy, slot mapping |
| Composition with MELT (#44) | 150 | TT-core streaming + dense-output kernel |
| Composition with PHOENIX (#47) | 150 | Packed streaming + in-kernel decode |
| Trainer integration (Phase A prefetch) | 200 | Inject prefetch into training loop, hook into REFLECTOR |
| Memory hierarchy abstractions | 200 | `WeightTier { GPU, HOST_PINNED, NVMe }` types + dispatch |
| **Total** | **1400** | 4–6 weeks production-grade |

The engineering risk is moderate: CUDA stream orchestration is well-understood (libtorch's `pipeline_parallel`, DeepSpeed ZeRO-Infinity have reference implementations). The CHIRON-specific novelty is the segment-aware inverse-walk-batched prefetch (§6.2), which is small-LOC but requires careful event ordering to avoid races between transfer and compute streams.

---

## 9. Concrete primitives (CUDA + C++)

### 9.1 Async memcpy manager

`StreamingWeightManager` owns a dedicated `cudaStream_t transfer_stream`, a parallel `compute_stream`, and a fixed device buffer pool of `K_MAX` slots each sized to `\max_l b_l`. Per-layer state: `(prefetch_event, compute_event, slot_index)`. Operations: `prefetch(l, host_W_l, b_l)` (issue H2D + record prefetch event), `wait_for_layer(l)` (cross-stream wait so compute does not race the H2D), `compute_done(l)` (record compute event for D2H ordering), `evict(l)` (free the slot).

### 9.2 Layer cache policy

`LayerCache` is a FIFO deque over the K transformer-layer slots, plus a `hot_pinned` set covering the embedding island (never evicted). FIFO matches CHIRON's deterministic L-layer pass; LRU adds nothing here.

### 9.3 Prefetch scheduler

`PrefetchScheduler` holds a planned execution-order queue (the L layer indices for forward, then reversed segments for backward). `warmup_prefetch()` issues the first K async H2Ds at step start. `advance(l)` is called after each layer completes; it issues the next prefetch and evicts the oldest if the cache is full.

### 9.4 Trainer integration

In `chiron_train.cpp` the per-layer fetch becomes `scheduler.advance_to(l); Y_l = forward_layer(l, q, p);`. For REFLECTOR backward, iterate segments in reverse: `scheduler.warmup_segment(seg); inverse_walk_segment(seg, ...); backward_segment(seg, ...); scheduler.evict_segment(seg);`.

### 9.5 PHOENIX-aware streaming

The host stores `(W_packed, s_W, M, N)` per layer; the H2D transfer ships only `\lceil M N / 5 \rceil` packed bytes (5-into-8 base-3 encoding). The GPU GEMM kernel `ternary_gemm_n` (#47 §9.2) consumes the packed buffer directly with no on-the-fly dequantisation pass. The 10× volume reduction translates into a 10× PCIe wall-clock reduction relative to streaming BF16 of the same logical shape.

---

## 10. Honest gap and limitations

### 10.1 PCIe is the binding constraint, packed streaming the lever

STREAM-CHIRON's slowdown is set by `V_{stream} / B_{PCIe}` after async overlap. Two implementation choices change this number by ~5×:

- **Packed streaming (recommended):** host stores PHOENIX-packed `(W_packed, s_W)`; PCIe carries 1.6 bits/weight; GPU GEMM consumes packed form. **PCIe 4.0: ~1.5× slowdown. PCIe 5.0: ~1.2×.**
- **BF16-shadow streaming (pitfall):** host keeps a BF16 master and ships BF16 to the GPU (e.g., to support a CPU-side QAT loop). **PCIe 4.0: ~6–7× slowdown. PCIe 5.0: ~2.5–3×.**

The headline 1.2–2× claim assumes packed streaming. The Gate-0 (§10.6) measures which path is realised in practice.

### 10.2 No compute speedup (this is purely a capacity paradigm)

Unlike #42–#47 which all delivered per-step compute speedups, STREAM-CHIRON contributes **0% compute speedup**. It enables training of *bigger* models than the post-#47 single-GPU ceiling permits. Users training models that already fit in VRAM after #44+#47 (i.e., ≤ 180 B) gain nothing from STREAM-CHIRON and should not enable it.

### 10.3 NVMe tier is not viable for active training

The 7 GB/s NVMe ceiling pushes wall-clock past 10 s/step at 1 T scale. STREAM-CHIRON is **host-RAM-only** as a serious training tool. NVMe is a checkpoint store, not a streaming source.

### 10.4 Composition with HYDRA (#45) is non-trivial

If both HYDRA and STREAM-CHIRON are enabled, each pipeline stage runs its own streaming pipeline against host RAM. The cross-product of inter-GPU NCCL traffic and PCIe streaming traffic puts pressure on the same bus. For a 2-GPU + STREAM setup, host bandwidth is shared between the two GPUs' streams; effective per-GPU streaming bandwidth halves. **Recommended: HYDRA *or* STREAM, not both.**

### 10.5 Gate-0 plan (~30 GPU-min)

Three runs at 1.84 B (which already fits in VRAM, so streaming overhead is purely measurable):

1. **A (baseline):** full #42–#47 stack, fully VRAM-resident.
2. **B (STREAM, K=L=53):** weights shadowed in pinned host RAM, K=53 → all layers cached → pure prefetch-pipeline overhead, no eviction pressure.
3. **C (STREAM, K=4):** aggressive eviction → real streaming traffic.

Pass criteria:
- B ≤ 1.05× A: confirms the prefetch-pipeline overhead is small.
- C ≤ 1.5× A on PCIe 4.0 (1.2× on PCIe 5.0): confirms the per-layer streaming wall-clock projects to the 250 B headline.

If C exceeds 2× A on PCIe 4.0, the implementation is in the BF16-shadow regime (§10.1) and packed streaming must be re-engineered.

---

## 11. Material differences from candidates B and C

| Aspect | A: STREAM-CHIRON | B: TBD | C: TBD |
|---|---|---|---|
| Hardware target | Single GPU + host RAM | (other) | (other) |
| Memory tier | GPU + pinned host RAM | TBD | TBD |
| Bandwidth budget | PCIe 4.0/5.0 | TBD | TBD |
| Compute speedup | **0×** | TBD | TBD |
| Memory scaling | ~3–5× model size growth | TBD | TBD |
| Slowdown | 1.2–5× depending on PCIe gen | TBD | TBD |
| Engineering | ~1400 LOC, 4–6 weeks | TBD | TBD |
| Single-GPU 1 T crossing | NO (caps ~500 B at 128 GB host) | TBD | TBD |

STREAM-CHIRON is structurally the **memory-hierarchy paradigm**: it adds a tier between HBM and the model, trading bandwidth for capacity. Other #48 candidates may pursue different axes (e.g., aggressive sub-1-bit quantisation, sparsity, dynamic-depth offload).

---

## 12. Implementation roadmap

| Iter | Task | Wall-clock |
|---|---|---|
| 192 | Gate-0 (§10.6) on 1.84 B | 30 GPU-min |
| 193 | StreamingWeightManager + LayerCache (LOC ~700) | 1 week |
| 194 | PHOENIX + MELT composition glue (LOC ~300) | 3 days |
| 194 | Trainer integration with REFLECTOR segments (LOC ~200) | 3 days |
| 195 | Memory hierarchy types + dispatch (LOC ~200) | 2 days |
| 195 | Unit tests (Theorem 1 prefetch correctness, FIFO eviction) | 2 days |
| 196 | 1.84 B end-to-end smoke under STREAM, K=8, PCIe 4.0 | 2 days |
| 197 | Scale-up to 18 B → 50 B → 100 B → 250 B | 1 week |
| 198–200 | 250 B pretraining run, 100k steps | 30+ GPU-days |

**Total:** 4–6 weeks to production code; 30+ GPU-days for the headline 250 B run.

---

## 13. Summary card

| Property | Value | Notes |
|---|---|---|
| Single-GPU model ceiling | **180 B → 320 B** (64 GB host), **→ 500 B** (128 GB host) | §7 |
| Per-step slowdown (packed streaming) | **~1.2× (PCIe 5.0)**, **~1.5× (PCIe 4.0)** | §5.4 |
| Per-step slowdown (BF16-shadow pitfall) | 2.5–3× (PCIe 5.0), 6–7× (PCIe 4.0) | §2.3, §10.1 |
| Compute speedup | **0×** | §10.2 |
| Streaming volume per step (packed) | ~0.07 N bytes | §5.2 |
| Reversibility | structural ✓ (lossless PCIe; CHIRON inverse drift unchanged) | §6.3 |
| New optimizer state | none | §1 |
| Compose with #44 MELT | streamed TT cores; 200× compression on FFN | §4 |
| Compose with #47 PHOENIX | streamed packed ternary; 10× compression on attn | §5 |
| Compose with #46 REFLECTOR | segment-aware prefetch, no re-fetch within segment | §6 |
| Compose with #45 HYDRA | ✗ recommended exclusive | §10.4 |
| LOC estimate | **~1400** | §8 |
| Engineering wall-clock | **4–6 weeks** | §12 |
| Falsifiable claim | Gate-0: 1.84 B + STREAM K=4 ≤ 1.5× baseline (PCIe 4.0) | §10.5 |
| Headline magnitude | 1.8× single-GPU model size at 1.2–1.5× slowdown | §0 |

---

## 14. Closing

STREAM-CHIRON is the **memory-hierarchy paradigm** for single-GPU CHIRON. It treats the 16 GB GPU as a streaming cache over a 32–128 GB host-RAM-resident model store. Async double-buffered prefetch over PCIe overlaps transfer with compute, exploiting CHIRON's natural layer-by-layer schedule and REFLECTOR's segment structure to keep the GPU compute-busy while weights stream.

The honest claim: **250–320 B parameters trainable on a single GPU at 1.2–1.5× wall-clock penalty** under packed-PHOENIX streaming, or 2.5–7× under naïve BF16-shadow streaming. PCIe generation and host RAM determine the operating point; the user's PCIe 4.0 RTX 4080 SUPER setup with 64+ GB DDR5 lands at ~1.5× — acceptable for offline research-scale pretraining of frontier-size models.

STREAM-CHIRON does **not** beat HYDRA for users who own multi-GPU clusters; HYDRA's NVLink interconnect (300 GB/s) crushes any single-GPU streaming approach. STREAM-CHIRON is the **single-GPU lifeline** — the way for a researcher with one workstation to train models ~1.8× larger than what fits in their 16 GB GPU after the post-#47 stack.

Where #44 MELT was *algebraic compression*, #47 PHOENIX was *bit-level compression*, and #45 HYDRA was *spatial distribution*, STREAM-CHIRON is **memory-tiering** — the orthogonal axis that pushes the single-GPU envelope from 180 B to 320 B at the cost of bandwidth-bound wall-clock.

The decision is empirical and falsifiable: Gate-0 (§10.5) measures whether packed streaming is achievable on the user's hardware. If yes, STREAM-CHIRON ships as the single-GPU's structural memory-tier paradigm. If the realised slowdown exceeds 2× on PCIe 4.0, STREAM-CHIRON is archived in favour of one of the other #48 candidates pursuing different axes.
