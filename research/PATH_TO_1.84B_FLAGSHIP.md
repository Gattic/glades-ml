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

### What's committed (Phase-1 — kernel correctness, no memory savings)

- `MixedPrecisionConfig.gradStorageBf16` bool wired through (commit `627967b33`)
- `GpuTransformerWeights` BF16 grad fields + `GpuTransformerScratch.gradScratchFp32`
- `allocate(..., bool gradStorageBf16)` flow allocates BF16 mirrors + scratch alongside
  FP32 grads (Phase-1 keeps both)
- Two new Adam kernel variants that read BF16 grads:
  - `adam_update_bf16_state_bf16grad`
  - `adam_update_int8_state_bf16grad` (reuses existing int8 kernel via
    `cast_bf16_to_f32` to scratch)
- 4-way per-tensor Adam dispatch in `sgd_transformer.cpp` at tokE, WIn, WOut and
  per-block Wq/Wk/Wv/Wo/W1/W2 sites: `bf16` / `int8` / `bf16+bf16grad` / `int8+bf16grad`
- FP32→BF16 cast pass on all 9 large weight grad buffers right before each Adam
  dispatch when `gradStorageBf16` set (exercises the new bf16grad kernels)
- Trainer flag `--grad-bf16` (commit `afcf2e6` in glades-trainer)
- `sum_squared_accumulate_bf16` kernel for the Phase-2 grad-norm pass (commit `e29b03a46`)

### Phase-1 verification (165M smoke, 2026-05-09)

T=1024, 64K tokens, `--face-embedding --adam-state-int8 --ffn-mlp` ± `--grad-bf16`:

| Metric            | Baseline    | + `--grad-bf16` | Δ           |
|-------------------|------------:|----------------:|------------:|
| NLL @ seq 63      |     10.6178 |         10.6178 |       0.000 |
| Final epoch loss  |   10.618756 |       10.618745 | -1.1e-5 nat |
| acc_top1          |   0.001527% |       0.001527% |   identical |
| Targets/sec       |    17,286.4 |        17,351.6 |       +0.4% |

Δ matches BF16 cast round-off magnitude.  Phase-1 dispatch path verified end-to-end.

### Phase-2 status (2026-05-09 → 2026-05-10)

Phase-2 backward refactor for the 6 per-block weight grads + gWIn + gWOut
is shipped (commits `573d63b7b`, `a8c8b3b7b`).  165M smoke parity at +1.1e-4
nat (BF16 round-off floor).  Phase-3 v2 explored extending Phase-2 to gTokE
via an `embedding_scatter_add_bf16` kernel (atomicCAS-on-uint32); rejected
empirically at -58 mnat NLL drift due to bf16 round-off accumulation in
per-element atomic adds (verifiable in `research/runs/grad_bf16_ph3/ph3.log`).

**Phase-2 alloc-retire infrastructure landed (`bd85b3e7f`, `dd09b2d30`) but
disabled by safety guard.**  When the FP32 alloc for ANY per-block tensor
is retired (via `GLADES_BF16_PH2_RETIRE` env var), 165M smoke drifts -9 to
+200 mnat from baseline.  Per-tensor bisect confirms the missed reader is
SYSTEMIC, not tensor-specific.  `d_adamGrads` pointer table verified clean
(big tensors excluded when `useBf16AdamState=true`).  Investigation pending —
likely candidates: a sync / download / serialization path that walks ALL
FP32 grad buffers without phase-2 awareness.

Until the missed reader is identified, Phase-2 runs with FP32 grads still
allocated alongside BF16 mirrors.  This means memory cost is currently
**worse** than Phase-1 (FP32 grads + BF16 mirrors, vs Phase-1's FP32 grads
alone).  The projected -3.65 GB at 1.84B is unrealized.

### What's not yet wired (Phase-2 — actual memory savings)

Phase-1 ships the kernel-correctness path: backward still writes FP32 grads, then
casts to BF16 right before Adam.  Peak memory unchanged because FP32 grads remain
allocated and live across the whole step.  Phase-2 retires the FP32 grad buffers:

1. **Allocate flow when `gradStorageBf16=true`** — allocate the BF16 mirrors but
   *not* the FP32 grad buffers (currently we allocate both).  Skip the FP32 weight
   grad allocations in `allocate()` when the flag is set.
2. **Backward-pass refactor** — every `gpu_gemm_*(..., 1.0f, gXX.data(), ...)` call
   site (9 distinct large-weight sites: gTokE, gWIn, gWOut, plus per-block
   gWq/gWk/gWv/gWo/gW1/gW2) is transformed to:
   ```cpp
   gpu_gemm_*(..., 0.0f, gradScratchFp32.data(), ...);     // overwrite scratch
   bf16_accum_axpy(gXX_bf16.data(), gradScratchFp32.data(),
                   /*alpha*/1.0f, /*beta*/firstMicroBatch ? 0.0f : 1.0f, n);
   ```
   The bf16 mirrors are zeroed at the start of each Adam window; `beta=1` thereafter
   accumulates correctly across both layer order and micro-batches.
3. **Grad-norm pass** at lines 11159-11189 switches to `sum_squared_accumulate_bf16`
   (kernel landed; just needs the call-site swap).
4. **Drop the Phase-1 cast pass** at line 12317 onward — backward already commits
   straight to BF16, no cast needed.
5. **MLA-specific paths** (`gWdkv`, `gWuk`, `gWuv`) — same pattern.

### Validation plan (Phase-2)

- 165M smoke: NLL parity vs Phase-1 + FP32-grads baseline within 0.005 nat (Phase-2
  introduces no NEW round-off vs Phase-1 — both go through bf16 by Adam-time).
- 700M smoke: confirm 3-4 GB GPU memory savings vs Phase-1 (peak grad footprint
  drops from 7.36 GB FP32 + 3.68 GB BF16 to 3.68 GB BF16 + 31 MB scratch).
- 1B-class run: previously OOM at FP32 grads + activation stash; should now fit
  with Phase-2 alone for inference/forward, but training still needs Design 2
  (activation checkpointing).
- 1.84B run: needs Phase-2 + Design 2 stacked.

### Memory delta at 1.84B from Phase-1 → Phase-2

| Component       | FP32 grads (Phase-0) | + BF16 mirrors (Phase-1) | BF16 only (Phase-2) |
|-----------------|---------------------:|-------------------------:|--------------------:|
| Weight grads    |              7.36 GB |          7.36 + 3.68 GB |             3.68 GB |
| Adam scratch    |                    – |                        – |               31 MB |
| Net cost        |              7.36 GB |                 11.04 GB |             3.71 GB |

Phase-2 saves **3.65 GB** at 1.84B (vs Phase-0) — close to half of what FP32 grads
cost.  Stacked with int8 Adam state and FACE embedding (already shipped), this
brings 1.84B inside the budget *for the parameter side*.  Activations remain the
final blocker → Design 2.

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
