# Beyond CHIRON — Research Directions for Further Paradigm Shifts

CHIRON delivered the first axis: **O(1) activation memory in depth** via
block reversibility. The standalone CHIRON trainer demonstrates the
empirical win — 1.2 B parameters training on a single 16 GB RTX 4080
SUPER, whereas a baseline transformer with stored activations caps out
around 250 M on the same hardware.

The binding constraint has shifted. At 1.2 B params, GPU memory breaks
down as:

| Category          | FP32  | Best-available today         |
|-------------------|------:|------------------------------|
| Weights           | 4.8 GB| 4.8 GB (FP32 master required)|
| Gradients         | 4.8 GB| 2.4 GB with BF16 grads (not yet wired) |
| Adam m, v         | 9.6 GB| 4.8 GB with `--bf16-adam`     |
| Activations       |     — | 0.05 GB — **CHIRON wins here** |
| Scratch (attn, logits) | ~1 GB | ~0.5 GB with chunked readout |

**Total at 1.2 B, current best: ~12 GB.**
Hardware: 16 GB. Headroom: 4 GB. Ceiling: ~1.5–2 B.

To get to 10 B on the same card, the remaining axes (weights +
gradients + optimizer state) each need an order-of-magnitude squeeze.

## Three candidate next-paradigm directions

### 1. **NF4-quantized weights with FP32 master** (memory-axis squeeze)

*Hypothesis:* replace FP32 forward/backward weight reads with NF4 (4-bit
non-uniform quantization) per-matrix, keep an FP32 master for the Adam
update. Weights read 8× faster per token, and the stored forward-compute
weight set is 8× smaller.

*Memory impact at 1 B params:*
- FP32 master on GPU:   4 GB  (unchanged)
- NF4 forward weights:  0.5 GB  (re-derived from master each update)
- **Net saving: nil if FP32 master stays on GPU.**
- **Net saving: 3.5 GB if FP32 master lives in host pinned memory**
  and we stream-in for the Adam update (ZeRO-offload variant).

Engineering cost: ~400 LOC of CUDA (NF4 quantize/dequantize + per-tile
scale table) plus host-offload plumbing. Known-good recipe from
bitsandbytes — mostly adaptation, not invention.

### 2. **Stream-weighted CHIRON** (both memory and speed; architectural novelty)

*Hypothesis:* use the reversibility of CHIRON blocks to **share the same
physical weight tensors across a group of layers**, with the block index
mixed in as a low-rank conditional modulation. A 48-"logical-layer"
network uses the physical storage of 8 "super-layers" plus a per-layer
LoRA-style adapter (rank ≤ 16).

Effective depth is still 48 (the reversible flow composes), but the
stored parameter count drops 6× for the trunk.

*Memory impact at 1.2 B logical params:*
- Shared trunk weights: 200 M  (800 MB FP32)
- Per-layer adapters: 48 × 1 M = 48 M  (192 MB)
- Total weight state: ~1 GB  (4.8× compression)

*Open math question:* does reversibility still hold when the same
weight matrix is reused at multiple block indices? The symplectic-shear
structure of CHIRON is invariant under weight permutation at the
*inverse-composition* level, so yes — the inverse block at depth `l`
just reverses the forward block at depth `l` with the same `Wq_l,...`.
But **convergence under weight sharing** is an empirical open question;
it may or may not reach baseline loss.

Engineering cost: ~200 LOC in the trainer. Scientific cost: a risk that
the shared-weight scheme simply doesn't train.

### 3. **Gradient-streaming + Adam-offload** (fully decoupled memory axis)

*Hypothesis:* keep only the currently-backwarding layer's gradients on
the GPU. As soon as layer `l`'s backward finishes, (a) async-copy its
grads to host pinned memory, (b) the CPU thread applies Adam (FP32)
and writes updated weights back. The GPU sees only layer `l+1`'s
gradients at any time.

*Memory impact at 1.2 B params:*
- Weights on GPU: 4.8 GB (full)
- Grads on GPU: ~200 MB (one layer's worth, rotating)
- Adam m, v on host: 9.6 GB  (unused GPU memory)
- **Net GPU saving: ~14 GB of Adam state + 4.6 GB of grad — 18.6 GB.**
- But CPU-GPU bandwidth (PCIe 4.0: ~25 GB/s) becomes the bottleneck.
  At 2 ms/layer compute, we can move ~50 MB per layer without
  stalling. 1.2 B / 48 layers = 25 M params = 100 MB per layer — 2×
  over bandwidth budget, so this only works if we overlap Adam with
  the *next* layer's backward compute. Doable with pinned memory and
  async-copy, but intricate.

This is the classic "ZeRO-offload" recipe from DeepSpeed — novel
contribution here would be the **interleaving with CHIRON's inverse
reconstruction**, which has natural per-layer cadence.

## Recommendation

**Pursue #3 (gradient-streaming + Adam-offload) first.** It's the one
that unlocks *>10 B params on a single consumer GPU*, matching the
paradigm-shift brief. #1 is worth adding later as a compounding multiplier
(NF4 forward weights × offloaded FP32 master × CHIRON activations =
three independent memory wins). #2 is the most *scientifically novel* but
carries convergence risk.

Current CHIRON progress bounds the risk for each: with O(1) activation
memory already shipped, we know the remaining savings are realizable on
the optimizer/weight/gradient axes — they don't need new theory, only
careful engineering. A 10 B LLM on a 16 GB consumer card in the next
iteration horizon is a concrete, credible target.

## 2026-04-21 update — progress on direction #3

Shipped:
  - Asymmetric int8 Adam state (signed m, unsigned v) — 4× smaller than FP32
  - CPU-offloaded Adam Phase 1 (blocking) + Phase 2 (transferStream async)
    — 1.78 B GPU-only ceiling @ 307 tok/s
  - BF16 gradient accumulators via bf16_accum_axpy — halves gradient VRAM
    — 1.68 B GPU-only ceiling @ 1484 tok/s (preferred production path)

Now shipping the foundation for direction #4 — stochastic-rounded BF16
weights.  The `cast_f32_to_bf16_stochastic` kernel is in place and
unit-tested (50/50 halfway rounds, 1.00× expected on sub-ULP
accumulation).  Once the trainer wires BF16 master weights end-to-end,
the FP32 weight storage (currently 3.36 GB at 1.38 B) drops to BF16
(1.68 GB), unlocking the projected 2.2-2.5 B GPU-only ceiling.

## Direction #4 — Stochastic-rounded BF16 weights (new)

*Hypothesis:* eliminate the FP32 master weights entirely.  Adam reads
the BF16 weight, decodes to FP32, applies the update in FP32, and stores
back with stochastic rounding — the expected value of the stored BF16
weight equals the true FP32 update exactly, so sub-ULP updates
accumulate correctly across steps (instead of being quantized to zero
by deterministic round-to-nearest-even).

*Memory impact at 1.38 B (CHIRON + int8 Adam + BF16 grads baseline):*
  - FP32 weights persistent:  3.36 GB  →  BF16 weights: 1.68 GB.  Save 1.68 GB.
  - Shared FP32 weight scratch: one layer's worth (~108 MB) — not
    persistent, reused before each layer's forward / backward / Adam.
  - Net saving: 1.57 GB → new ceiling ~2.2 B GPU-only.

*Risks:*
  - BF16 storage doubles rounding noise on the weight values (~0.2%)
    vs FP32 master.  Training should still converge — BF16-master
    training is an established technique in production LLMs.
  - Cast-before-forward overhead: ~5-10 ms per step (~1-2 % of a 500 ms
    step at 1.4 B).  Acceptable.

*Engineering path:*
  1. kernel (SHIPPED): `cast_f32_to_bf16_stochastic`
  2. kernel: BF16-weight-reading Adam variant (thin wrapper: cast,
     adam_update, cast back with stochastic round)
  3. trainer: --bf16-weights flag; BF16 weight buffers per layer;
     shared FP32 weight scratches; cast-before-compute in forward/backward
  4. validation: unit-test end-to-end training convergence matches FP32
     weights within acceptable margin
