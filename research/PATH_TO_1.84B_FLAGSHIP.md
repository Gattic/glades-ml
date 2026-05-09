# Path to a 1.84B flagship head-to-head against CHIRON

**Date:** 2026-05-09
**Status:** plan + foundation — full implementation pending
**Goal:** train a 1.84B-param flagship that matches or beats CHIRON 1.84B
on wall-clock and NLL while fitting in 16 GB GPU.

## Memory budget at 1.84B (m=2048, L=53, dFF=5632, T=1024)

| Component | FP32 baseline | After current work | Target |
|---|---|---|---|
| BF16 weights | 3.7 GB | 3.7 GB ✓ | 3.7 GB |
| Adam state (int8 + FACE on tokE) | 14.7 GB | **2.0 GB ✓** (committed) | 2.0 GB |
| Gradients | 7.4 GB FP32 | 7.4 GB FP32 | **3.7 GB BF16** (work to do) |
| Activation stash (10× L·T·d) | 4.4 GB | 4.4 GB | **0.6 GB sqrt-checkpoint** (work to do) |
| ff1 scratch (MLP) | 1.2 GB | 1.2 GB ✓ (`--ffn-mlp`) | 1.2 GB |
| Other scratch | 1.5 GB | 1.5 GB | 1.5 GB |
| **TOTAL** | **32.9 GB** ❌ | **20.2 GB** ❌ | **12.7 GB** ✓ |

The current work landed in this branch (int8 Adam, FACE Adafactor,
`--ffn-mlp`, GPU init) drops the requirement from 32.9 GB to 20.2 GB —
a 12.7 GB reduction, about 2/3 of the way there. The remaining 7.5 GB
gap is closed by two things:

1. **BF16 grads** (saves ~3.7 GB) — paradigm port from CHIRON
2. **Activation gradient checkpointing** (saves ~3.8 GB) — standard transformer technique

Both required. Either alone leaves the run at 16-17 GB on a 16 GB card.

## Design 1: BF16 grads (foundation committed; per-call-site refactor pending)

### Mechanism

Replace the persistent FP32 grad buffers (one per weight tensor) with
persistent BF16 grad buffers + a single shared FP32 scratch buffer
(sized to the widest weight tensor).  Each backward GEMM writes into
the scratch FP32 buffer; immediately afterward `bf16_accum_axpy`
commits the result into the persistent BF16 grad buffer.  Adam reads
the BF16 grad directly via new kernel variants.

### What's committed

- `MixedPrecisionConfig.gradStorageBf16` (new bool) wired through
- `GpuTransformerWeights` BF16 grad fields (`gWq_bf16`, `gWk_bf16`,
  …, `gW1_bf16`, `gW2_bf16`, plus globals `gTokE_bf16`, `gWIn_bf16`,
  `gWOut_bf16`, plus MLA `gWdkv_bf16`, `gWuk_bf16`, `gWuv_bf16`)
- `allocate(..., bool gradStorageBf16)` signature wired through

### What's not yet wired

1. **The allocate flow itself** — when `gradStorageBf16=true`,
   allocate the BF16 buffers and skip the FP32 grad buffers.  Add a
   `gradScratchFp32` buffer to `GpuTransformerScratch` sized to the
   widest weight tensor (`max(V·dModel, dFF·dModel, dModel·dModel)`).
2. **The backward-pass refactor** — every `gpu_gemm_*(..., 1.0f, gXX.data(), ...)`
   call site needs to be transformed to:
   ```cpp
   gpu_gemm_*(..., 0.0f, gradScratchFp32.data(), ...);
   bf16_accum_axpy(gXX_bf16.data(), gradScratchFp32.data(), 1.0f, 1.0f, n);
   ```
   ~30 sites in `sgd_transformer.cpp`.  Mechanical but must be done
   carefully.  A helper macro can compress each site to ~3 lines.
3. **New Adam kernel variants** that read BF16 grads:
   - `adam_update_bf16_state_bf16grad_kernel(param, grad_bf16, m_bf16, v_bf16, ...)`
   - `adam_update_int8_state_bf16grad_kernel(param, grad_bf16, m_int8, v_uint8, scales, ...)`
   Same body as the existing kernels but `g = bf16_load_as_f32(grad_bf16[idx]) * gradScale`
   instead of `g = grad[idx] * gradScale`.
4. **Dispatch macros** in `sgd_transformer.cpp` extended with bf16-grad
   variants (`GLADES_BF16_ADAM_BIG_BF16GRAD`, `GLADES_INT8_ADAM_BIG_BF16GRAD`)
   selected when `gradStorageBf16` is on.
5. **Trainer flag** `--grad-bf16` plumbing through `cfg.mixedPrecision.gradStorageBf16`.

### Validation plan

- 165M smoke: NLL parity vs FP32-grads baseline within 0.01 nat over
  1000 steps (acceptable BF16 rounding noise).
- 700M smoke: confirm 3-4 GB GPU memory savings vs FP32 grads.
- 1B+ run: previously OOM at FP32 grads + activation stash; should now
  fit if combined with checkpointing (Design 2).

## Design 2: Activation gradient checkpointing (sqrt(L) scheme)

### Mechanism

Save the per-block hidden state `hAfterFF[layer]` only at every
K = ⌊√L⌋ layer (the "checkpoint" boundaries).  On backward, before
processing each segment of K layers, recompute the forward pass for
that segment from the prior checkpoint, filling the per-step scratch.
Then run backward over the segment using the freshly recomputed
intermediates.

For L=53: K=7, so we save 53/7 ≈ 8 checkpoints (~8× T × dModel = 65 MB
each at d=2048).  Per-step scratch holds K layers' worth of
activations during the segment recompute (~600 MB).  Total: ~1.1 GB
vs the 4.4 GB full stash.

### Implementation sketch

In `transformerGpuTrainEpoch` (`sgd_transformer.cpp`):

```cpp
// Forward — replace per-layer activation slot indexing with single scratch:
for (unsigned int li = 0; li < nLayers; ++li) {
    layer_forward(li, x_in, x_out, /*scratch slot=*/li % K);
    if (li % K == 0) {
        // Save checkpoint: copy x_out into the global checkpoint array
        cudaMemcpyAsync(checkpoints[li / K], x_out, T * dModel * sizeof(float), ...);
    }
}

// Backward — replace plain backward loop with segment-by-segment:
for (int segStart = (nLayers / K) * K; segStart >= 0; segStart -= K) {
    // 1. Recompute forward for layers [segStart, segStart+K)
    //    starting from checkpoints[segStart / K]
    cudaMemcpy(x, checkpoints[segStart / K], T * dModel * sizeof(float), ...);
    for (int li = segStart; li < min(segStart + K, nLayers); ++li) {
        layer_forward(li, x, x_next, /*scratch slot=*/li - segStart);
        x = x_next;
    }
    // 2. Backward over the segment (scratch slots [0..K-1] now hold
    //    the recomputed activations for this segment)
    for (int li = min(segStart + K, nLayers) - 1; li >= segStart; --li) {
        layer_backward(li, dY, dX, /*scratch slot=*/li - segStart);
        dY = dX;
    }
}
```

Per-layer scratch buffers shrink from `[L × T × dModel]` to `[K × T × dModel]`
where K = √L.  At L=53 K=7: ~7× smaller activation stash.

### What's not yet wired

- Modify `GpuTransformerScratch::allocate()` to size per-layer buffers
  at K instead of L when checkpointing enabled.  Or: keep them at L
  but only USE K slots at a time (simpler, no allocation change).
- Add `gradCheckpoint` config knob.
- Refactor the forward loop in `transformerGpuTrainEpoch` to:
  1. Replace `[layer]` indexing with `[layer % K]` indexing.
  2. Save hAfterFF copy at every Kth layer.
- Refactor the backward loop to:
  1. Iterate by segments of K.
  2. Recompute forward for the segment.
  3. Backward over the segment.
- Trainer flag `--grad-checkpoint` plumbing.
- Validation: gradient parity vs no-checkpointing baseline (sub-ε
  difference on each grad tensor; hash check on Adam updates).

### Compute cost

Standard sqrt(K) checkpointing: backward does one extra forward pass
worth of compute over the entire model (each segment is recomputed
once).  Net throughput cost: ~1.5× per backward step (forward already
done once normally, recompute is a second forward).

For wall-clock: at 1.84B, current backward dominates (~70% of step
time).  1.5× backward → 1.35× step time.  Net wall increase: ~35%.

In exchange, fits 1.84B that doesn't otherwise fit.  Acceptable.

## Run plan once both designs land

```bash
# 1.84B head-to-head with all the work
cd /home/robert/dev/glades-trainer
./build/glades_pile_train --gpu --mp \
    --face-embedding --adam-state-int8 --grad-bf16 --grad-checkpoint \
    --ffn-mlp \
    --dmodel 2048 --layers 53 --heads 16 --dff 5632 \
    --seq-len 1024 --tbptt 1024 \
    --max-tokens 50000000 \
    --attn-sinks 4 --local-attn 256 \
    --mla-dc 128 --binary-ffn \
    --lr 1e-4 --grad-clip 1.0 --warmup-steps 1000 \
    --model-name flagship_184B \
    --no-auto-resume
```

Expected: ~12.7 GB peak GPU, sustained throughput ~800-1200 tokens/sec
(slower per-token than CHIRON's 1665 due to the 1.35× checkpointing
cost, but at 1.84B same params count, comparable total wall to
similar-token-count CHIRON run).

The "improve over CHIRON" claim then rests on:
- Same params (1.84B)
- Same hardware (16 GB)
- Same NLL trajectory (within 0.05 nat)
- Comparable or better total wall-clock at fixed token budget

If wall is comparable (within 30%) and NLL parity holds, that's a
draw on convergence with cheaper per-step memory.  If wall is faster,
that's a clear win.  CHIRON's reversibility advantage on activation
memory is matched by checkpointing here, so the comparison reduces
to: which compute-side paradigm stack (MLA + binary FFN + sinks vs
reversible shears + ReLN) is faster per token at 1.84B.

## Engineering effort estimate

Honest accounting, ignoring development time as instructed:

- Design 1 (BF16 grads): foundation in this branch.  Remaining work
  is ~600 LOC of mechanical refactors + 2 new kernels + dispatch wiring
  + smoke-test validation cycle.
- Design 2 (gradient checkpointing): ~400 LOC of new training-loop
  logic + scratch allocation changes + gradient-parity validation.
- Combined integration test at 1.84B: 1-2 days of probing + tuning lr.

After both land, the head-to-head can be run in a single ~30-min
training session per the brief.
