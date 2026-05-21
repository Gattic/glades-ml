# CPU-Offloaded Adam — Design Document

## Motivation

Current CHIRON + int8 Adam ceiling on 16 GB RTX 4080 SUPER: 1.38 B params.
Memory breakdown:

| Category           | Size at 1.38 B |
|--------------------|---------------:|
| Weights (FP32)     | 5.27 GB        |
| Gradients (FP32)   | 5.27 GB        |
| Adam m, v (int8)   | 2.66 GB        |
| Scratch (attn + logits) | ~2 GB     |
| **Total**          | **15.32 GB**   |

Adam state (2.66 GB) is the biggest compressible chunk left *that doesn't
require new backward-kernel precision work*.  Moving it to CPU pinned
memory frees that 2.66 GB on the GPU without changing any forward /
backward math.  Projected new ceiling: ~1.7 B params on the same card.

Beyond that, the same CPU-offload mechanism extends to weights and
gradients for the BEYOND_CHIRON.md #3 paradigm — 10 B+ on 16 GB.

## Minimal implementation (Phase-1: Adam state only)

Goal: store Adam m, v on CPU host pinned memory; keep param + grad on GPU.
Each Adam step: download grad → CPU computes Adam step → upload param back.

### ChironParams changes

Replace `glades::gpu::GpuBuffer<int8_t>` / `uint8_t` / `float` vectors
for Adam state with host-side pinned buffers.  Use `cudaMallocHost`
(or `cudaHostAlloc` with `cudaHostAllocPortable`) for pinned allocation —
enables true async DMA without CPU copy staging.

```cpp
struct HostAdamState {
    float* m;      // FP32 on CPU
    float* v;      // FP32 on CPU  (could go int8 with host-side kernel if needed)
    size_t n;      // element count
    // ...
};
// Per param group, owned by ChironParams when --cpu-adam is set.
```

### adam_step changes

For each param group:
1. Allocate a temporary host pinned `grad_host` buffer (or reuse a rolling
   max-size scratch).
2. `cudaMemcpyAsync(grad_host, grad_device, ...)` on the transfer stream.
3. `cudaMemcpyAsync(param_host_scratch, param_device, ...)` for a working
   copy (CPU never keeps param resident; only staging).
4. Synchronize transfer stream.
5. On CPU thread pool (OpenMP — 8-32 cores amortises well on 1.38 B):
   ```c
   for i in 0..n:
     m[i] = beta1 * m[i] + (1-beta1) * grad_host[i]
     v[i] = beta2 * v[i] + (1-beta2) * grad_host[i]^2
     m_hat = m[i] / bc1
     v_hat = v[i] / bc2
     param_host_scratch[i] -= lr * m_hat / (sqrt(v_hat) + eps)
     if wd != 0:  param_host_scratch[i] -= lr * wd * param_host_scratch[i]
   ```
6. `cudaMemcpyAsync(param_device, param_host_scratch, ...)` back.
7. Synchronize transfer stream — the next forward needs updated params.

### Bandwidth accounting

At 1.38 B params:
- Grad download: 5.27 GB
- Param download + upload: 10.54 GB (only on --cpu-adam; regular path stays GPU-resident)

Actually the cleanest protocol is: **CPU keeps a persistent copy of param
too.**  Then:
- Download: grad (5.27 GB)
- Upload: param (5.27 GB)   ← much smaller
- Total per Adam step: 10.54 GB

At PCIe Gen4 x16 (25 GB/s realised): 420 ms per Adam step for transfers,
plus ~150 ms for CPU Adam arithmetic on 1.38 B params (8 cores × 6 GFLOPS/core
≈ 48 GFLOPS; Adam is ~10 FLOPs/param → 288 ms ... actually slower).  Call
it 500-600 ms per Adam step.

Current GPU Adam step at 1.38 B: ~20 ms (dominated by kernel launches).

So CPU-offload Adam adds ~580 ms of overhead per Adam step.  At accum=4,
the forward+backward cycle is ~2 s — overhead is 30 %.  **Acceptable for
the 24 % param-count headroom gained (1.38 → 1.7 B)**.  At accum=16+,
overhead is <10 % — much better tradeoff.

### Phase-2 optimizations (deferred)

- **Double-buffer the transfers with CPU compute**: while CPU is
  applying Adam to layer L's grads, start downloading layer L+1's grads
  (the backward finished earlier).  Halves transfer latency.
- **Overlap with the next step's forward**: params for layer 0 can
  start forward while CPU Adam is still finishing layers 40+.  Needs
  careful synchronization to ensure forward doesn't read a half-updated
  weight tensor.
- **Int8 Adam on CPU**: move the asymmetric-int8 encoding to CPU side.
  Halves the CPU memory footprint at minor CPU-compute cost.
- **Async stream integration**: transfers on `transferStream()`, compute
  on `computeStream()`, with events to serialize dependencies.

## Interaction with CHIRON's per-layer inverse cadence

CHIRON's backward does `chiron_attention_shear_backward_tiled` layer by
layer in reverse order.  After layer L's backward, its grads are complete
and WOULD be ready for Adam.  But currently we wait for all L layers
to finish, then run Adam as a block.

The per-layer interleaving point is already natural:
```cpp
for ll in 0..L:
    l = L - 1 - ll
    reln_inverse(...)
    shear_inverse(...)
    reln_backward(...)
    shear_backward(...)   // writes dWq[l], dWk[l], ..., dgamma[l], dbeta[l]

    // NEW: kick off Adam for layer l NOW — its grads are complete.
    async_download_grads_for_layer(l)
    schedule_cpu_adam_for_layer(l)
    // by the time we finish layer 0's backward, the CPU Adam thread
    // has processed most of layers L..L/2; CPU-GPU transfers overlap
    // with the continuing backward compute.
```

This is BEYOND_CHIRON.md direction #3 "interleaved with CHIRON's inverse
cadence" realised concretely.

## Risks & mitigations

- **Pinned memory allocation failure**: host pinned memory is scarce.
  Mitigation: fallback to pageable memory (with a perf warning).
- **CPU Adam numerical drift**: CPU Adam is FP32, matches GPU Adam
  byte-for-byte.  No precision difference.
- **Lock contention on m, v access**: serialize the Adam step (single
  thread or OpenMP with per-element parallelism — no shared state across
  elements).  No issue.
- **Out-of-order forward reads**: the current forward reads weights
  AFTER the Adam step completes.  With async pipelining we need event
  barriers to guarantee the updated weights are visible.  Use
  `cudaEventRecord` after the Adam-upload and `cudaStreamWaitEvent`
  before the next forward.

## Implementation estimate

- Phase 1 (blocking CPU Adam): **300 LOC**.
- Phase 2 (pipelined with CHIRON inverse): **+200 LOC**, non-trivial
  event sync.
- CPU Adam kernel (OpenMP-parallel): 50 LOC.
- Pinned host-memory allocator wrapper: 30 LOC.
- Trainer plumbing (new --cpu-adam mode, destructors, etc): 100 LOC.

Total for Phase 1 + 2: ~700 LOC.  The biggest chunk of work yet in the
trainer, but each part is well-scoped and testable independently.
