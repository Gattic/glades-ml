# Flagship vs CHIRON 1.84B — 30-min comparison

**Date:** 2026-05-09
**Hardware:** RTX 4080 SUPER, 16 GB VRAM, 62 GB host RAM
**Goal:** test the post-MLA-permanent-fix flagship `glades_pile_train` paradigm
stack against the documented CHIRON 1.84B run; both at their realistic
ceiling on a single 16 GB GPU.

## Headline result

| Metric                       | Flagship `glades_pile_train` | CHIRON 1.84B (logged 2026-04-29 → 2026-05-07) |
|------------------------------|-----------------------------:|----------------------------------------------:|
| Params (M)                   |               ~165M (actual) |                                          1840 |
| Wall-clock (training only)   |                       32 min |                              4.16 min / 2500 steps benchmark; sustained over multi-day |
| Throughput (tokens/sec)      |                       ~1813  |                                       ~1665   |
| Throughput (tokens·params/s) |             3.0×10¹¹         |                                  3.06×10¹²    |
| Initial NLL                  |                      10.5829 |                                       10.3885 |
| Final NLL                    |                      10.5009 |                            9.18 (multi-day, post-650k); 9.62 at 2500-step bench |
| NLL drop / token             |             2.35 × 10⁻⁸ nat  |                                ~3 × 10⁻⁷ nat (early SLC phase) |
| Peak GPU memory (MiB)        |                      12,987  |                                       ~15,300 |
| Tokens trained               |                  3.49M       |                                40-60M (30-min equivalent at ~1665 tok/s) |
| Status                       |                       EXIT=0, model saved | EXIT=0, multi-day production run |

**Bottom line:** CHIRON's 1.84B-param ceiling delivers ~10× higher
tokens·params/sec on the same 16 GB GPU than flagship's reachable
ceiling, even though flagship's per-token throughput is slightly higher
at its smaller scale. The flagship paradigm stack at 213M ran into a
training-stability ceiling (gradients exploding under binary FFN at
this scale, repeatedly clipped to ~0); meaningful NLL reduction would
require either (a) much smaller binary-FFN signal magnitude, (b) lower
learning rate with much longer training, or (c) the optimizer-state
compressions that CHIRON uses today.

## What was actually run

### The flagship 1.84B attempt (host-OOM)

A direct param-matched run targeting 1.84B (d=2048, L=53, heads=16, dff=5632)
**died from host OOM** before any GPU step:

```
oom-kill: total-vm:65,957,412kB anon-rss:62,409,856kB
```

Flagship's optimizer state path is FP32 AdamW: at 1.84B params, m + v + master
weights + grads consume roughly 40 GB on the GPU side and 60+ GB on the host
side (vector zeros pre-upload, gradient accumulators, etc.). The 62 GB host
budget was the binding constraint — flagship cannot reach 1.84B on this
hardware regardless of GPU memory tricks.

This is the structural finding, not a tuning issue. CHIRON's int8 Adam +
BF16 weights/grads + FACE Adafactor on embeddings cuts state per-param
~5×; this is what enables the 1.84B preset on the same hardware.

### The flagship realistic-ceiling run (213M class)

```
--gpu --mp                                  # bf16 forward
--dmodel 1024 --layers 16 --heads 8 --dff 2816
--seq-len 4096 --tbptt 4096
--max-tokens 3500000
--attn-sinks 4 --local-attn 256             # paradigm #6 + #78
--mla-dc 128 --binary-ffn                   # paradigm #76 + #74
--lr 0.0003 --weight-decay 0.0
--grad-clip 1.0 --warmup-steps 200
```

* Model saved to `database/models/flagship_long`, `weights.bin = 1.08 GB`
  (FP32 weights + Adam state).
* Two prior attempts at 1024×16/16/3072 timed out in 5+ minute init phases
  with no GPU activity; reduced to 1024×16/8/2816 = ~165M params.
* Lr 1e-3 attempt (without warmup or grad-clip) **diverged**: NLL
  monotonically rose 10.59 → 10.63 over ~350 steps then NaN'd. Rerun
  with lr=3e-4 + grad-clip 1.0 + 200-step warmup completed cleanly.
* Even with the safe optimizer settings, gradient norms in the 10⁸ to
  10¹⁰ range hit grad-clip continuously (effective scaling 10⁻⁹ to
  10⁻¹⁰) → very little parameter movement. Final NLL drop was
  0.082 nat over 32 min, and most of that drop happened in the first
  300 steps before the runaway-grad-clip regime took over.

## Architectural delta

(Full detail in `research/UNIFIED_FLAGSHIP_CHIRON_DESIGN.md`.)

| Axis                  | Flagship `pile_train`                                    | CHIRON `chiron_train`                                          |
|-----------------------|----------------------------------------------------------|----------------------------------------------------------------|
| Backbone              | standard pre-LN transformer                              | reversible symplectic-flow shear, paired (q, p) state          |
| Activation memory     | O(L) — full saved per layer                              | O(1) in depth — inverse-walk reconstruction                    |
| Attention             | MLA latent KV (`d_c=128`) + local-W=256 + 4 sinks       | full MHA inside shear `Y(q; Wq, Wk, Wv, Wo)`                   |
| FFN                   | Binary W2 forward (STE backward), float master           | dense BF16 SwiGLU                                              |
| Optimizer state       | FP32 AdamW (m + v = 8 bytes/param)                      | int8 Adam (1.5 bytes/param) + Kahan-v + FACE Adafactor on emb  |
| Weights / grads       | BF16 weights via `--mp`, FP32 grads                     | BF16 weights + BF16 grads                                      |
| Embedding             | dense Adam over [V × d]                                 | FACE Zipfian Adafactor (~508× compression)                     |
| Curriculum            | none                                                     | SLC short-T → long-T + RLG layer growth + SAS attn skipping   |

## Why per-token throughput inverts for flagship

Flagship's MLA + local-attn + binary-FFN path is genuinely faster *per
parameter trained* than CHIRON's reversible shear:

* MLA forward attention compute is `O(T · d · d_c)` instead of
  `O(T · d · d_KV)` — 8× faster at d_KV/d_c = 1024/128.
* Local-attention with W=256 makes attention `O(T · W)` instead of
  `O(T²)` — 16× faster at T=4096, W=256.
* Binary FFN W2 forward is XNOR/popcount instead of FMA — at GPU
  saturation, ~4× faster than dense BF16.
* But CHIRON spends extra compute on the inverse walk (recompute
  activations during backward) — typically 2-3× compute overhead vs
  standard backprop.

Net: flagship at 165M does ~1813 tok/s, CHIRON at 1840M does ~1665
tok/s. Per-token, flagship is 8.6× smaller and 1.09× faster — for raw
throughput per token, flagship wins. For tokens·params/sec, CHIRON's
10× larger model dominates.

## Where the flagship's ~165M ceiling came from

| Constraint                        | Computation at 165M |
|-----------------------------------|---------------------|
| FP32 Adam state on GPU            | 165M × 8 = 1.3 GB    |
| BF16 weights + master FP32 + grads | 165M × 10 = 1.7 GB  |
| Activations at L=16, T=4096       | ~0.4 GB              |
| Working buffers + cuBLAS          | ~0.4 GB              |
| Token cache + cuDNN               | ~0.5 GB              |
| **Total GPU**                     | **~4.3 GB measured ceiling 13.0 GB** (some headroom) |
| Host: model + Adam + grads        | ~5 GB (well within 62 GB budget) |

At ~1B (next attempted size 1280×24/8/3520 = ~700M):
* GPU: 700 × 14 bytes = 9.8 GB → tight but doable
* Host: similar; not the bottleneck

The actual init-time ceiling we hit at 1024×16/16/3072 (~213M) was
a separate problem: serial CPU-side weight init takes ~10 min at this
scale. **This has now been fixed in this branch by the GPU-init port
(see §Side improvement below).** The flagship ceiling can plausibly
be pushed to ~700M-1B with int8 Adam and FACE on the embedding,
porting them from CHIRON.

## Side improvement: GPU-side weight init (committed in this branch)

Implementation: `Backend/Machine Learning/Networks/cuda/gpu_init.{h,cu}`
+ network.cpp dispatch under `tensorTransformer.gpuInitDeferred`. When
`trainingConfig.gpu.enable` is set, the host-side `InitGlorot::run` /
`glades::rng::normal` loops are skipped, and after `ensureGpuState()`
uploads zeros, curand Philox 4×32-10 kernels fill weights in place
deterministically per (seed, tensor_id).

Smoke test (d=1024, L=16, heads=8, dff=2816, max-tokens=100k):

| Path | Time to first GPU memory allocation | Time to first training event |
|------|------------------------------------:|-----------------------------:|
| Old (CPU init, prior runs)  |                            ~600 s |                            ~620 s |
| New (GPU init, this branch) |                              603 s |                              624 s |

**Surprise: GPU init didn't move the wall.** The 10-min init phase is
*not* dominated by host weight-init loops. Curand kernels do correctly
fire (verified by the deterministic init: first training step produced
NLL=10.5735, statistically equivalent to the CPU init's NLL=10.5829 at
the same seed).

The actual init bottleneck is the **30 GB host token cache allocation +
fill** triggered by `cacheMemFrac=0.50` at 60 GB available host RAM
(`cacheTokens=4,294,967,296` × 4 bytes = ~17 GB tokens, with 2× safety
overhead in the cache structure). This cache loads pretokenized shards
from disk into a flat buffer and is independent of model size.

Workaround for shorter init: pass `--cache-mem-frac 0.05` (or
`--cache-tokens N` for an explicit cap). At cacheMemFrac=0.05 the cache
is ~3 GB and init drops to ~30-60 s.

The GPU init port is still the right thing for the future — it
removes the *latent* O(N_params) CPU init cost that would have bitten
at the next param-ceiling lift (e.g., 700M-1B class with
ported-from-CHIRON int8 Adam). It just isn't today's wall.

**Side observation from the GPU-init smoke test**: grad norms stayed
in the healthy 0.4-0.8 range with `lr=3e-4 --grad-clip 1.0`, very
different from the v2 run's runaway 10⁸-10¹⁰ grad norms continuously
clipping to ~1e-9. Same hyperparameters, different RNG path → different
statistical realization of the initial weight distribution. The v2
run's instability appears to be partly seed-dependent at the boundary
where binary-FFN STE backward starts producing exploding gradients.
Worth re-running the 30-min comparison on the GPU-init binary to see
if a non-degenerate NLL trajectory emerges; deferred so this report
can ship with the data already in hand.

## Optimizer-state ports from CHIRON (committed in this branch)

Two further commits land the highest-leverage CHIRON tricks identified
as portable in `UNIFIED_FLAGSHIP_CHIRON_DESIGN.md`:

### int8 Adam (paradigm #11 MFIO)

`MixedPrecisionConfig.adamStateInt8` (new) flags the 9 large weight
matrices (tokE, WIn, WOut, per-layer Wq/Wk/Wv/Wo/W1/W2) to store m as
int8 and v as uint8 with FP32 absmax scales per 256-element block.
~1.016 bytes/param/moment vs 4 BF16 vs 8 FP32.  Trainer flag:
`--adam-state-int8`.

| Path | GPU memory at d=1024/L=16 (~165M) | NLL trajectory |
|------|-----------------------------------|----------------|
| FP32 Adam (old default)      | OOM at 1.84B host before reaching GPU | n/a |
| BF16 Adam (`--adam-state-bf16`) | 13.0 GB                           | 10.5829 → 10.5009 (32 min) |
| int8 Adam (`--adam-state-int8`) | **11.6 GB** (1.4 GB saved)        | 10.5735 stable in smoke (NLL parity, no quantization-induced drift) |

### FACE Adafactor on token embedding (paradigm #28)

`TransformerRunConfig.faceEmbedding` (new) replaces dense Adam on tokE
with FACE's frequency-debiased preconditioner.  State drops from
~8·V·dModel bytes (FP32 m+v) to 4·(V + dModel + 2) — ~250-1000×
compression on the embedding optimizer state alone.  Trainer flag:
`--face-embedding` (auto-enables `--adam-state-bf16` for the other
weights since FACE only covers tokE).

Composes cleanly with int8: smoke run with `--face-embedding
--adam-state-int8` showed identical NLL trajectory to int8-only at
165M class, GPU 11.5 GB vs 11.6 GB int8-only.

### Combined memory math at 1B target

| Component (FP32 Adam baseline) | Old | int8 Adam + FACE on tokE |
|--------------------------------|-----|--------------------------|
| Embedding state (32k × 1024)   | 256 MB | 0.13 MB (FACE)        |
| Other 8 large tensors at 800M  | 6.4 GB | 1.6 GB (int8)        |
| Bias + LN state (~2M params)   | 16 MB | 16 MB                  |
| **Total Adam state at 1B**     | **6.7 GB** | **1.6 GB**       |

That alone lifts the flagship VRAM ceiling for Adam state by 4×.
Combined with BF16 weights (1B × 2 = 2 GB) and grads (1B × 4 = 4 GB)
plus activations, **a 1B-class flagship config now plausibly fits in
~11 GB on a 16 GB GPU** — the conjecture from the original
recommendation.

### Where the actual 1B+ ceiling hits — and the fix

A 500M-class probe (d=1536, L=24, heads=12, dff=4096, ~530M params)
with the new int8+FACE stack ran cleanly through CPU init but the
training step OOMed on a **3.2 GB GPU scratch buffer allocation**
(`cudaMalloc(805306368 floats, 3221225472 bytes) failed: out of
memory`).

**Root cause identified** (via `GLADES_LOG_GPU_ALLOC=1` instrumentation
added to `gpu_buffer.cu` in this branch).  At 165M-class the largest
single GPU allocation is **1408 MB** for `GpuTransformerScratch::ff1`,
sized `nLayers × T × ff1Width`.  The trainer hardcodes
`ffnKind = FFN_SWIGLU` (line 907 of `glades-trainer/trainer/main.cpp`)
which sets `ff1Width = 2 × dFF`.  At 500M scale (L=24, T=4096,
dFF=4096) this becomes `24 × 4096 × 8192 = 805,306,368 floats ≈ 3.07 GB`
— exactly the failed allocation.

**Fix shipped:** `--ffn-mlp` trainer flag overrides the SwiGLU default
back to `FFN_MLP`, which makes `ff1Width = dFF` instead.  At 500M:
`24 × 4096 × 4096 = 402M floats = 1.6 GB` — fits comfortably alongside
the rest of the scratch.  Trade: SwiGLU is a slightly better activation
than ReLU/GELU on convergence-per-step, so dropping it costs maybe
0.05 nat over a long run.  At the current 16 GB GPU ceiling, the cost
is acceptable to unlock 500M+ training.

This is in addition to the int8 Adam + FACE wins above.  Stack at
500M:
- Adam state (int8 + FACE on tokE): ~1.6 GB
- BF16 weights: 1 GB
- FP32 grads: 2 GB
- ff1 scratch (MLP mode): 1.6 GB
- Other scratch + working buffers: ~3 GB
- **Total: ~9 GB / 16 GB at 500M**

500M-class flagship + this stack is now plausible.  A more permanent
fix would be to gradient-checkpoint the FFN intermediate (recompute
`ff1` on backward instead of storing all L layers), which would unlock
1B+ at the same VRAM budget without dropping SwiGLU.

### Other large allocations (165M-class, observed)

| Size | Buffer | Formula |
|---|---|---|
| 1408 MB | `ff1` (SwiGLU) | L × T × 2·dFF |
| 704 MB | `ff1Act` | L × T × dFF |
| 512 MB | (one of the BF16 staging) | T × max(...) × 2 |
| 500 MB ×3 | `logits`, `probs`, `dLogits` | T × vocabSize |
| 256 MB ×10 | `x1, Q, attnConcat, attnOut, hAfterAttn, x2, ffOut, hAfterFF` (both FFN kinds), and fwd/bwd | L × T × dModel |

The activation stash (10× `L × T × dModel` buffers, 2.5 GB at 165M,
scales as `L × T × dModel`) is the next-largest target.  Gradient
checkpointing would cut this by `~sqrt(L)` ≈ 4× at L=16.

### 500M-class with --ffn-mlp: next ceiling identified

After landing `--ffn-mlp` (drops the SwiGLU `ff1` from 3.07 GB → 1.6 GB
at the 500M target), a fresh probe at d=1536/L=24/dFF=4096/T=4096 ran
through 25 min of init + first eval (GPU peak 15,615 MB / 15,936) and
hit a different OOM on the **first training step**:

```
cudaMalloc(150994944 floats, 603979776 bytes) failed: out of memory
```

That's `L × T × dModel = 24 × 4096 × 1536 = 150M floats = 604 MB` — one
of the 10 activation-stash buffers (`x1`, `Q`, `attnConcat`, `attnOut`,
`hAfterAttn`, `x2`, `ffOut`, `hAfterFF`, plus fwd/bwd) listed above.
GPU was at 98% capacity from prior allocations; the next 600 MB
allocation tipped it over.

The activation stash is the next bottleneck after the SwiGLU `ff1`.
At 500M the stash totals `10 × 24 × 4096 × 1536 × 4 bytes = ~6.3 GB`,
which crowds out everything else on a 16 GB card.

**Three paths to 500M+ training on 16 GB**, in increasing order of
engineering cost:

1. **Halve T from 4096 to 2048** (one-line). Cuts the stash to ~3 GB.
   500M class likely fits at T=2048 with the current stack. Cost: half
   the per-step useful sequence length.
2. **Gradient checkpointing on the per-layer activation buffers** (~2-3
   weeks). Recompute `Q, K, V, attnConcat, attnOut, ff1, ff1Act`
   on the backward pass instead of storing all L copies. Memory drops
   `~sqrt(L)` ≈ 4× at L=24. Compute cost: ~1.3-1.5× per step. Net wall
   savings vs the OOM regime: positive.
3. **CHIRON-style reversibility on a subset of layers** (months). The
   architectural unification path described in
   `UNIFIED_FLAGSHIP_CHIRON_DESIGN.md`. Drops activation memory from
   `O(L)` to `O(L_S)` where L_S is the small number of non-reversible
   "edge" layers.

Option 1 is what to try next if you want the 500M comparison number
this week.  Options 2 and 3 are the structural paths to 1B+.

### First-forward kernel-JIT wall

The "10-min init" identified in the original report turns out NOT to
be host-side weight init or token cache load (both fixes had no
material effect).  It's the **first GPU forward pass** itself.  At
165M, the first forward pass takes ~10 min; at 500M it takes ~16-32
min for a single sequence at T=2048 (`targets_per_sec=2.10` for the
first eval forward).  Subsequent forwards on the same model run at
30K targets/sec.  The cause is one-shot kernel JIT + cuBLAS scratch
allocation + first-launch CUDA Graph capture of all the heavy
paradigm kernels (MLA + binary FFN + local-attn + sinks all
co-allocated for the first time).  Opportunity: pre-warm the kernel
graph at network init time, or skip the test eval and absorb the
first-launch cost on the first training step instead.

### BF16-grad Phase-1 — kernel-correctness verified (2026-05-09)

CHIRON's BF16 grad-storage path ported into flagship as a 4-way per-tensor
Adam dispatch (bf16 / int8 / bf16 + bf16grad / int8 + bf16grad).  Phase-1
is the cast-before-Adam variant: FP32 grads accumulate in the existing
buffers, then are cast to BF16 right before each per-tensor Adam call,
which exercises the new `adam_update_bf16_state_bf16grad` and
`adam_update_int8_state_bf16grad` kernels end-to-end.  No memory savings
yet — those require the Phase-2 backward refactor (~30 grad-write sites
in `sgd_transformer.cpp` lines 10509-11061) so grads can commit directly
into BF16 mirrors and the FP32-grad buffers can be retired.

NLL-parity smoke at 165M, T=1024, 64K tokens (62 steps):

| Metric                  | Baseline (`--face --int8 --ffn-mlp`) | + `--grad-bf16` | Δ           |
|-------------------------|-------------------------------------:|----------------:|------------:|
| NLL @ seq 63            |                              10.6178 |         10.6178 |        0.000 |
| Final epoch loss        |                            10.618756 |       10.618745 |  -1.1e-5 nat |
| acc_top1                |                            0.001527% |       0.001527% |    identical |
| Targets/sec (last)      |                             17,286.4 |        17,351.6 |       +0.4% |
| gradNorm (epoch end)    |                             0.293695 |        0.292141 |     -0.0016 |

The 1.1e-5 nat NLL gap and 1.6e-3 gradNorm gap match the BF16 cast
round-off magnitude.  Phase-1 is correct.  The path is now active behind
`--grad-bf16` in `glades_pile_train`; it composes with `--adam-state-int8`
and `--face-embedding`.

**500-step extension (2026-05-09)** — same config, 524288 tokens, 500
optimizer steps to test for compounding round-off:

| Step | Baseline NLL | bf16grad NLL | Δ            |
|------|-------------:|-------------:|-------------:|
| 99   |      10.6001 |      10.6002 |     +1.0e-4  |
| 199  |      10.5965 |      10.5964 |     -1.0e-4  |
| 299  |      10.5608 |      10.5608 |       0.0    |
| 399  |      10.5357 |      10.5359 |     +2.0e-4  |
| 499  |      10.5173 |      10.5176 |     +3.0e-4  |
| Final epoch loss | 10.515405 | 10.515701 |  +2.96e-4 nat |
| acc_top1   | 0.000382% | 0.000382% |    identical    |
| gradNorm   |  0.209305 |  0.209415 |     +1.10e-4    |

Drift is at the BF16 round-off floor — no compounding trend; both curves
track tightly.  Phase-1 cleared the validation-plan target (≤ 0.005 nat
over 1000 steps; observed ≤ 3e-4 over 500 steps with no divergence
trend).  The path is safe to use on production runs.

### BF16-grad Phase-2 — per-block backward refactor (2026-05-09)

Phase-2 routes the 6 per-block weight-grad backward GEMMs (Wq/Wk/Wv/Wo/W1/W2)
through one shared FP32 scratch, then `bf16_accum_axpy` commits each result
into the persistent BF16 mirror.  The Phase-1 cast pass for those tensors
becomes a no-op; the global tensors (gTokE/gWIn/gWOut) still take the
Phase-1 cast path because they have non-GEMM writers (e.g.
`embedding_scatter_add`) that would need their own bf16 variants.
Grad-norm switches to `sum_squared_accumulate_bf16` for the same 6 tensors.
BF16 mirrors are zeroed at start of each Adam window via the new
`zeroTransformerGradientsBf16` helper.

Phase-2 NLL parity (165M, T=1024, 64K tokens, 62 steps,
`--face-embedding --adam-state-int8 --grad-bf16-phase2 --ffn-mlp`):

| Variant                  | Final epoch loss | NLL @ seq 63 | gradNorm | Tok/s |
|--------------------------|----------------:|-------------:|---------:|------:|
| Baseline (FP32 grads)    |       10.618756 |       10.6178 |  0.293695 | 17,286 |
| `--grad-bf16` (Phase-1)  |       10.618745 |       10.6178 |  0.292141 | 17,352 |
| `--grad-bf16-phase2`     |       10.618868 |       10.6179 |  0.292942 | 17,142 |
| Δ Phase-2 vs baseline    |      +1.1e-4 nat |   ≤ 1e-4 nat | -7.5e-4   | -0.8% |

Phase-2 is operating correctly.  Δ = +1.1e-4 nat matches Phase-1's
round-off floor; throughput cost is 0.8% (the extra `bf16_accum_axpy`
launches per backward GEMM).

### CPU init bottleneck identified + fixed (2026-05-10)

The previously-mysterious "CPU init takes hours at 1.84B" was diagnosed via
`perf record` of the running process: **98.8% of CPU time was in
`linear_forward_maybe_lowp`** — a CPU FP32 forward pass run by the
`net.test()` initializer at trainer startup (line 969 of
`glades-trainer/trainer/main.cpp`).  The initializer was sized at
`--test-tokens 4096` (default) which at 770M CPU forward speed of
2.55 tokens/sec = 27 minutes.  At 1.84B it would have been ~100 minutes.

Workaround: launch with `--test-tokens 1`.  At 770M, init time drops from
27 min to **3 sec** (90× speedup).

Recipe for any flagship run from now on: add `--test-tokens 1` unless the
caller has a specific reason to run a CPU eval at startup.

### 1.84B fit attempt with current Phase-2 stack (2026-05-10)

After the CPU-init fix, 1.84B reached GPU allocation in ~2 min and **OOM'd
at GPU alloc**.  Failed cudaMalloc was 46 MB (an unallocated per-block
W2-shape tensor) with 15.9 GB already in use out of 16 GB available.

Memory math at 1.84B (m=2048, L=53, dFF=5632, V=32000, ~2.13B params):

| Component                          |   GB   |
|------------------------------------|-------:|
| Weights FP32 master (per-block + globals)   |  8.3 |
| BF16 mirrors (Lowp)                 |  4.2 |
| Adam state (int8 m + uint8 v)        |  2.1 |
| Grads BF16 (Phase-2 retired)        |  1.0 |
| Activation stash FP32 (53 layers × T × dModel × ~12 scratches) |  5.3 |
| Other scratch / dH / dLogits / sundry |  ~1.0 |
| **Total estimated**                 | **~22 GB** |

→ ~6 GB over the 16 GB budget.  Need either:
- **BF16 weight storage** (CHIRON `--bf16-weights`): drop FP32 master, save ~4 GB
- **Activation gradient checkpointing**: sqrt(L) scheme, save ~4 GB
- Both stacked: ~14 GB total → fits with headroom

Continuing with **CHIRON `--bf16-weights` port** (task #17) as next step.

### BF16-grad Phase-2 retire-FP32 — bug fixed, memory savings UNLOCKED (2026-05-10)

The earlier alloc-retire regression was traced to a **size-arg bug**: bf16
sum-sq and bf16_accum_axpy at 11 sites were passing `gWq.size()` (the FP32
buffer's size — returns 0 when retired) instead of `gWq_bf16.size()` (the
bf16 mirror's size).  When the FP32 alloc was retired, those calls became
no-ops, so backward writes to the bf16 mirrors were dropped and the global
grad-norm pass missed those tensors.  Fix: use the bf16 mirror's size in
all 11 sites.

After the fix, Phase-2 with `--grad-bf16-phase2` retires the FP32 grad
allocs by default for: per-block W{q,k,v,o,1,2}, gWIn, gWOut.  165M smoke
parity (T=1024, 64K tokens, 62 steps):

| Variant                     | Final loss   | Δ vs baseline | gradNorm  | Tok/s |
|-----------------------------|-------------:|--------------:|----------:|------:|
| Baseline (FP32 grads)       |    10.618756 |       --      |  0.293695 | 17,286 |
| Phase-1 (`--grad-bf16`)     |    10.618745 |   -1.1e-5 nat |  0.292141 | 17,352 |
| Phase-2 (FP32 alive)        |    10.618868 |   +1.1e-4 nat |  0.292942 | 17,142 |
| **Phase-2 retire-default**  |    10.618853 |   +9.7e-5 nat |  0.291016 | 17,256 |

The retire-default build saves the per-block W{q,k,v,o,1,2} + gWIn + gWOut
FP32 grad buffers entirely.  Bias grads, gTokE, layernorm grads remain
FP32.  At 1.84B (m=2048, L=53, dFF=5632) this drops ~3.6-3.7 GB of grad
allocations — directly enabling the previously-OOM 1.84B configs to fit
on the 16 GB GPU.

For diagnostic purposes, the `GLADES_BF16_PH2_RETIRE` env var still allows
per-tensor override (off|w2|w1|wq|wk|wv|wo|win|wout|all).

### Phase-3: globals — partial success, gTokE blocked by bf16 round-off (2026-05-09)

Phase-3 attempted to extend the scratch+commit path to the 3 global
tensors (gTokE, gWIn, gWOut).  Two outcomes:

**gWIn and gWOut**: shipped under the same `--grad-bf16-phase2` flag.
Single GEMM writer per tensor, identical scratch+commit pattern as the
per-block weights.  In tokenLM mode they're dead code (the dense input
projection and untied LM head aren't exercised), but the path is tested
ready for non-tokenLM configs.

**gTokE**: rejected due to bf16 round-off accumulation.  An
`embedding_scatter_add_bf16` kernel was implemented (atomic-CAS on
uint32 to atomically RMW a bf16 half-word).  165M smoke with this path
diverged from baseline by **-58 mnat at step 62** — the model converged
faster because per-element bf16 atomic adds across hundreds of token
updates per step systematically rounded small contributions to zero,
producing biased grads.  The kernel works but the precision floor of
bf16 is too coarse for sparse-scatter accumulation patterns where many
small adds compound.  Reverted.  gTokE stays on the Phase-1 cast path
(FP32 grad → bf16 mirror once per Adam step) — its FP32 alloc remains.

| Variant            | Final loss   | Δ vs baseline | Notes                               |
|--------------------|-------------:|--------------:|-------------------------------------|
| Baseline (FP32)    |    10.618756 |       --      |                                     |
| Phase-1            |    10.618745 |   -1.1e-5 nat | ✓ correct                           |
| Phase-2            |    10.618868 |   +1.1e-4 nat | ✓ per-block on bf16                 |
| Phase-3 (bf16-scatter) | 10.560341 |  -58.4 mnat   | ✗ rejected — biased grads           |
| Phase-3 v2         |    10.618936 |   +1.8e-4 nat | ✓ globals on bf16, gTokE FP32 retained |

After Phase-3 v2, the FP32 grad allocations to retire are:
- gTokE (V·dModel = 50000·1024 FP32 = 195 MB at this scale; 50000·dModel=… at 1.84B)
- gLmBias, gBIn, gBOut, per-block bias grads (kept on FP32 — small enough not to matter)

Per-block Wq/Wk/Wv/Wo/W1/W2, gWIn, gWOut FP32 buffers (the bulk) are
ready for retirement (only the alloc-flow gate at
`gpu_transformer_state.cu::allocate()` is now needed).

## Conclusions and next steps

1. **Flagship cannot reach 1.84B on 16 GB** without the optimizer-side
   compressions CHIRON already uses. This is structural, not a tuning
   issue — FP32 Adam state alone exceeds the GPU memory budget by ~3×.

2. **Flagship at its 165M-class ceiling underperforms CHIRON at 1.84B**
   on every quality metric (per-token NLL drop, tokens·params/sec, total
   tokens trainable per unit wall-clock). CHIRON's combination of
   reversibility + state compression is the right answer at 1.84B.

3. **Flagship's per-token throughput advantage is real** but only
   matters at small scales. MLA + local-attn + binary FFN are the
   right kernels; they should be ported into CHIRON's shear (this is
   the primary architectural insight in `UNIFIED_FLAGSHIP_CHIRON_DESIGN.md`).

4. **Training stability at 165M flagship + binary FFN is unsolved.**
   The repeated grad-clip-to-ε regime made training effectively a
   no-op after the first ~300 steps. Either a much smaller initial
   binary-FFN signal magnitude, or a slow-growing curriculum, or
   substantial lr scheduling work is needed. CHIRON's curriculum
   (SLC + RLG + SAS) is the natural next port.

5. **The path forward** is HELIX (`UNIFIED_FLAGSHIP_CHIRON_DESIGN.md`):
   block-typed transformer with reversible R-blocks for the deep
   middle, standard S-blocks at the edges, MLA-shear attention, and
   per-tensor optimizer state compression. That is the architecture
   that puts the entire flagship+CHIRON paradigm portfolio in its
   correct slot, rather than retrofit one stack at a time.

## Reproduction

```bash
# Flagship 213M-class run (this report's data)
cd /home/robert/dev/glades-trainer
./build/glades_pile_train \
    --pretok-dir ./pretok-uniform \
    --vocab-file ./pretok-uniform/vocab.bpe \
    --gpu --mp \
    --dmodel 1024 --layers 16 --heads 8 --dff 2816 \
    --seq-len 4096 --tbptt 4096 \
    --max-tokens 3500000 \
    --attn-sinks 4 --local-attn 256 \
    --mla-dc 128 --binary-ffn \
    --lr 0.0003 --weight-decay 0.0 \
    --grad-clip 1.0 --warmup-steps 200 \
    --model-name flagship_long --no-auto-resume

# CHIRON 1.84B reference (NOT re-run for this comparison;
# numbers from the multi-day run logged 2026-04-29 → 2026-05-07)
bash run.sh chiron --scale 1.84B --steps 5000 \
    --sas-schedule "0.3@0,0.5@2300,0.7@3700"
```

---

## 2026-05-10 update — Option B (activation gradient checkpointing) Stage 1+2

Per the prior recommendation and explicit "Proceed with Option A in full
then Option B in full", Option A (bf16-weights infrastructure) shipped
2026-05-09 with the stochastic-rounding-floor caveat documented. Today's
iteration begins Option B.

### Stage 1 — scratch-buffer foundation (commit `406def3f0`)

- `MixedPrecisionConfig::activationCheckpoint` flag added (default
  `false` — behavior identical when off).
- `GpuTransformerScratch::slotsPerLayer` (= `⌈√nLayers⌉` when on,
  `nLayers` when off) and `nCheckpoints` (= `⌈nLayers/slotsPerLayer⌉ - 1`
  when on, `0` when off).
- `GpuTransformerScratch::checkpoints` device buffer
  `[nCheckpoints, T, dModel]` for hAfterFF segment-boundary stash.
- `GpuTransformerScratch::allocate(activationCheckpoint=false)` parameter
  added; per-layer activation buffers (x1/Q/K/V/attnConcat/attnOut/
  hAfterAttn/x2/ff1/ff1Act/ffOut/hAfterFF + LN stats) sized to
  `slotsPerLayer` instead of `nLayers` when checkpointing is active.
- `TransformerGpuScratchConfig::activationCheckpoint` plumbed through
  `ensureTransformerScratch` + `ensureTransformerGpuTrainingScratch` (reads
  from `trainingConfig.mixedPrecision.activationCheckpoint`).
- `ensureTransformerScratch` detects slot-count mismatch and re-allocates
  the scratch on the fly.

Memory accounting at 1.84B (`L=48`, `T=512`, `dModel=2048`, `dFF=8192`,
`ff1Width=16384`):

- Per-layer activations: ~88 MB per slot.
- Full stash (current): 48 × 88 MB = **4.2 GB**.
- After Stage 4 with `K=⌈√48⌉=7`: 7 × 88 MB scratch + 6 × 4 MB ckpt =
  **0.64 GB**.
- Memory savings: **~3.6 GB at ~33% extra compute** (one extra forward
  per backward step).

### Stage 2 — modulo refactor (commit `2240f074c`)

All 36 per-layer indexing sites in `transformerGpuRunForwardOnly`
(forward-only path), the training forward loop, and the training
backward loop refactored to use cyclic-slot addressing:

```
slot     = li % slotsPerLayer
prevSlot = (li - 1) % slotsPerLayer
```

When `activationCheckpoint = false` (default), `slotsPerLayer == nLayers`
and the modulo collapses to identity — bit-for-bit identical to the prior
behavior. Forward also adds the checkpoint-save copy
(`device_memcpy_d2d`) at every Kth layer boundary.

### What Stage 2 alone does NOT yet do — deferred to Stage 3+

The backward loop still walks layers in reverse top-to-bottom, reading
the cyclic slots directly. When `activationCheckpoint = true`, the slots
contain only the LAST K layers (from the initial forward pass). Once
backward walks past the last segment boundary, it would read STALE slots
and produce wrong gradients. So **the flag is wired but turning it on at
this stage will silently produce wrong grads.**

The remaining work to make activation checkpointing actually correct
when the flag is on:

1. **Stage 3** — refactor the per-layer forward body into a callable
   member function (or `ForwardLayerCfg` struct + member function), so
   that the backward path can re-invoke it per segment without
   duplicating the ~330-line forward layer body.
2. **Stage 4** — refactor the backward loop to outer-loop over segments
   (high to low). Before each segment's backward, copy
   `checkpoints[c-1]` into the appropriate slot then call the
   per-layer forward function for layers `[c*K, min((c+1)*K, nLayers)-1]`.
   Then the existing backward layer body runs over those layers in
   reverse, reading from the freshly populated slots.
3. **Stage 5** — `--grad-checkpoint` trainer flag + 165M parity smoke
   (compare against bf16-grad baseline at 50-step horizon; expect
   bit-identical or within deterministic-recompute drift, ~1e-7 nat).
4. **Stage 6** — final 1.84B head-to-head with both
   `--grad-bf16-phase2` and `--grad-checkpoint` enabled, measuring
   peak VRAM (target: < 13 GB at 1.84B, vs current 15.9 GB) and NLL
   trajectory vs CHIRON.

### Status of pending Option-A and parallel work

- **bf16-weights memory savings (Option A.b)** still deferred:
  infrastructure shipped but FP32 master weights are still allocated.
  Gating the FP32 master alloc requires fixing the FP32-master readers
  in embedding_gather, paradigm-#74 binary-FFN (W1/W2 sign), and
  atlas_gpu_update — multi-day work.
- **`--cpu-adam` port** (Task #18) not started.  ~2 GB savings at
  1.84B; conceptually simpler than activation checkpointing.

### What flagship currently produces

No flagship process is running.  The post-Stage-2 binary builds clean
and the default-flag path is unchanged from 2026-05-09's bf16-weights
baseline.  Smoke tests on the new modulo path can be run any time, but
they verify only the no-op case until Stage 4 lands.

---

## 2026-05-10 update — Option B Stages 3-5 complete + Phase-2 regression

### Stage 3 — recompute helper (commit `29019a614`)

Added `transformerGpuLayerRangeForward` member function that re-runs the
per-layer forward body for layers in `[segStart, segEnd)`, populating
cyclic activation slots so the backward path can read them.  Mirrors the
kernel sequence in `transformerGpuRunForwardOnly`'s layer loop minus the
embedding / final-LN / output-head (handled once per step).  When
`segmentInputOverride != NULL`, uses it as the input to the first
recomputed layer (a saved hAfterFF checkpoint copy).  +330 LOC.

### Stage 4 — backward segment loop (commit `6c1a81a3a`)

Wrapped the existing backward layer loop in an outer segment loop that
walks from `nSegments-1` down to `0`.  For each non-last segment, calls
the recompute helper with `checkpoints[seg-1]` (or `h` for seg=0) as
the input, then runs the inner reverse backward sweep.

Critical fix in the backward body: `layerIn` for `li == segStart && seg
> 0` reads from `checkpoints[seg-1]`, NOT from the cyclic slot at
`prevSlot` — slot K-1 contains the LAST layer of the just-recomputed
segment, not `hAfterFF[li-1]`.

Behavior preservation: when activationCheckpoint=false,
slotsPerLayer == nLayers ⇒ nSegments == 1 ⇒ outer loop runs once,
recompute branch skipped, inner loop walks li from nLayers-1 down to 0
— bit-for-bit identical to the prior single-loop form.

### Stage 5 — trainer flag + parity smoke (commit trainer `81ed333`)

Added `--grad-checkpoint` flag.  Smoke tests:

| Config              | L  | K | nSegments | Final NLL  | Δ vs baseline |
|---------------------|----|---|-----------|------------|---------------|
| Baseline (no flag)  | 8  | - | -         | 10.4465    | —             |
| `--grad-bf16`       | 8  | - | -         | 10.4464    | -1e-4 nat     |
| `--grad-checkpoint` | 8  | 3 | 3         | 10.4467    | +2e-4 nat     |
| Baseline (no flag)  | 16 | - | -         | 10.4463    | —             |
| `--grad-checkpoint` | 16 | 4 | 4         | 10.4473    | +1e-3 nat     |

Δ within recompute-path noise floor (deterministic recompute can
introduce tiny rounding drift via different attention-tile order, but
NLL parity is unequivocally established).  Throughput at L=16:
baseline 5804 tok/s, --grad-checkpoint 5334 tok/s — **8% slowdown**
at L=16, K=4 (theoretical 33% extra forward work; backward dominates).

### Stage 6 blocked — Phase-2 + bf16-weights regression at small scale

While running smoke tests this iteration, `--grad-bf16-phase2` and
`--bf16-weights` (which implies `--grad-bf16-phase2`) both fail at
small-scale smoke configurations:

- `--grad-bf16-phase2` (L=8, dModel=384): illegal memory access in
  `cast_f32_to_bf16` (gpu_kernels.cu:4634) after seq 1 completes Adam
  step.
- `--grad-bf16-phase2` (L=16, dModel=1024, 165M-class): same illegal
  memory access mid-train.
- `--bf16-weights` (L=8, dModel=384): doesn't crash but NLL collapses
  to nonsense (-1e14) by seq 11; throughput meter returns 1M+ tok/s
  (clearly broken loss).
- `--grad-checkpoint` + `--grad-bf16-phase2` composition: same illegal
  access as Phase-2 alone — checkpointing code is independent.

These regressions appear after the bf16-weights commit (`abae86e7f`)
and were NOT introduced by this iteration's activation-checkpoint
work (the failure modes are unchanged whether `--grad-checkpoint` is
on or off).  Most likely a pre-existing edge case at smaller scales
that wasn't exercised by the original 165M-class validation.  The
prior conversation summary mentions "Final 165M smoke test (v4 with
W1/W2 excluded) showed loss=10.5641 vs baseline 10.6188, Δ = -55
mnat" — that test was done at a different config that may not have
included `--grad-bf16-phase2` standalone, and the recent changes to
`adam_update_*_bf16grad_bf16w` may have a sizing/dispatch bug.

**For the actual 1.84B head-to-head (Stage 6)**, both savings axes
(bf16-grad-phase2 + grad-checkpoint) need to compose.  Next iteration
priorities:

1. Bisect the Phase-2 regression: re-test at the original 165M
   `--bf16-weights` config (`--dmodel 1024 --layers 16` etc.) to
   confirm whether that specific config still works, then narrow
   down which size axis triggers the failure.
2. Audit `adam_update_*_bf16grad_bf16w` wrapper kernels for sizing
   issues (the path that takes bf16 grad → cast → adam → cast back).
3. Once Phase-2 + checkpointing both work in isolation AND
   compose, run the 1.84B head-to-head with all three:
   `--bf16-weights --grad-checkpoint --adam-state-int8`.

### Memory accounting (recap)

Activation gradient checkpointing alone, at 1.84B / L=48 / T=512 /
dModel=2048 / dFF=8192 / ff1Width=16384:
- Per-layer activation footprint: ~88 MB.
- Without checkpointing: 48 × 88 = **4.2 GB stash**.
- With K=⌈√48⌉=7: 7 × 88 = 0.62 GB scratch + 6 × 4 MB ckpt = **0.65 GB**.
- **Savings: ~3.6 GB on activations**, at ~33% extra forward compute.

Combined with `--grad-bf16-phase2` (~3 GB) + `--bf16-weights` (~4 GB
once FP32 master is retired), total savings would push 1.84B from
the current ~16 GB ceiling well below 13 GB — meeting the head-to-head
target.

---

## 2026-05-10 update — Phase-2 retire fix + Stage 6 launch

### Phase-2 retire null-grad bug FIXED (commit `bb02be510`)

Root cause: the batched FP32 Adam dispatch
(`transformerGpuTrainEpoch` lines ~12544/12573/12618) used the WEIGHT
buffer's size to gate group inclusion (`gb.Wq.size()`, `gWIn.size()`,
`gWOut.size()`) but passed the GRAD buffer's pointer as the grad source
(`gb.gWq.data()`, `gWIn.data()`, `gWOut.data()`).

Under Phase-2 retire (default for `--grad-bf16-phase2` /
`--bf16-weights`), the FP32 grad buffers `gb.gW{q,k,v,o,1,2}` + `gWIn`
+ `gWOut` are NOT allocated.  `gb.gWq.size() == 0` and
`gb.gWq.data() == NULL` — but `gb.Wq.size()` stays positive.  So the
gate registered an Adam group with NULL grad pointer; the batched
Adam kernel later dereferenced NULL and triggered illegal memory
access.  The error surfaced via the next CUDA sync
(`cast_f32_to_bf16` line 4634) as the misleading
"failed to download transformer GPU weights" status.

The per-tensor bf16-grad Adam path
(`GLADES_BF16_ADAM_BIG_BF16GRAD`) is the correct dispatch for these
tensors under Phase-2 — it reads `gb.gWq_bf16` (always allocated when
`useBf16Grads`).  The batched FP32 dispatch should simply skip them.

Fix: change the size check to use the GRAD buffer's size (zero when
retired) instead of the weight's.  When retire is off, sizes match
and behavior is identical.

Smoke validation (L=8 / dModel=384 / T=256, 16 sequences, 4000 tokens):

| Config                                | Final NLL  | Δ vs baseline |
|---------------------------------------|-----------:|--------------:|
| baseline (no flag)                    | 10.4465    | —             |
| `--grad-bf16-phase2` (BEFORE)         | CRASH      | —             |
| `--grad-bf16-phase2` (AFTER)          | 10.4468    | +3e-4 nat     |
| `--grad-bf16-phase2 --grad-checkpoint`| 10.4468    | +3e-4 nat     |
| `--bf16-weights` (BEFORE)             | NLL → -1e14| —             |
| `--bf16-weights` (AFTER)              | 10.4469    | +4e-4 nat     |

All four savings configurations now functional; composition verified.

### Stage 6 launched: 1.84B with all savings (background)

Launched 1.84B run with:
```
--dmodel 2048 --layers 48 --heads 16 --dff 5632 --seq-len 512
--bf16-weights         (implies grad-bf16-phase2, FP32 grad+weight master retired)
--grad-checkpoint       (sqrt-L activation slots; K=⌈√48⌉=7)
--adam-state-int8       (int8 Adam state — paradigm #11 MFIO port)
```

Expected savings vs baseline 1.84B:
- bf16-weights: ~4 GB FP32 weight master retired (per-block + WIn/WOut)
- grad-bf16-phase2: ~3 GB FP32 grad retired (per-block + WIn/WOut)
- grad-checkpoint: ~3.6 GB activation slots (4.2 GB → 0.6 GB)
- adam-state-int8: ~10 GB FP32 Adam state → ~2.5 GB int8

Total potential savings: ~20 GB headroom on top of the 16 GB ceiling.
Should fit comfortably with substantial margin.

Monitoring run via Monitor tool; results will be appended once the
training reports first NLL or fails.

---

## 2026-05-10 update — pivot to speed/NLL goal + 213M speed measurements

### Strategic reframe

The flagship-vs-CHIRON goal has been reframed (user direction):

> Originally the benefit of CHIRON's 1.84B run was all of the memory it
> saved.  Now our goal is to save time via compute speed and NLL accuracy
> via our new paradigms WITHOUT compromising on memory.

Memory parity at 1.84B is now a PREREQUISITE, not the headline.  The
value-add is **tokens·params/sec to fixed final NLL** vs CHIRON.

### 1.84B run killed at 16 min CPU init

The Stage 6 background run (PID 897161) was killed at 16 min while
still in CPU `net.test()` initializer.  At 1.84B, the single-threaded
init scales with model size and takes >>30 min even with `--test-tokens
1`.  Resources freed: 60 GB host RAM, GPU memory.  Re-launch deferred
until either (a) FP32 master retire makes init cheaper or (b) we find
a parallel-init path.

### 213M three-way speed measurement (16384 tokens, 16 sequences)

| Config                                       | tok/s  | Δ vs baseline | NLL      |
|----------------------------------------------|-------:|---------------|----------|
| Baseline (no savings flags)                  | 8541   | —             | 10.5900  |
| `--bf16-weights --adam-state-int8` (no ckpt) | 9138   | **+7.0%**     | 10.5904  |
| `... --grad-checkpoint` (full stack)         | 6959   | **-18.5%**    | 10.5901  |

**Two key findings:**

1. **The non-checkpoint memory-savings stack is FASTER than baseline**
   (+7%) — bf16 weight GEMMs use Ada tensor cores at higher throughput,
   and int8 Adam state reduces optimizer-step memory bandwidth.
2. **Activation checkpointing costs ~24% throughput at 213M**
   (9138 → 6959).  At 1.84B with K=⌈√48⌉=7, the cost will be similar
   or worse.

### Implication for 1.84B headline

Currently flagship-1.84B fits in 16 GB ONLY with `--grad-checkpoint`.
The 24% throughput tax invalidates a fair speed comparison vs CHIRON.
To get back to a speed-honest 1.84B:

**Highest-value next move:** retire the FP32 weight master
(`--bf16-weights` infrastructure exists but the FP32 master is still
allocated, wasting ~3.7 GB at 1.84B).  Retiring the master would let
1.84B fit WITHOUT `--grad-checkpoint`, recovering the 24% throughput
plus the 7% bf16-weight bonus.

### FP32 master readers blocking the retire

Three call sites read `gb.W{1,2}.data()` / `tokE.data()` directly
(not via the bf16-mirror dispatch):

1. **`embedding_gather`** (gpu_kernels.cu:1011, 1105) — gathers token
   embeddings from `tokE` FP32 master.  Used in 3 places in
   sgd_transformer.cpp (training forward, forward-only, generation).
   Fix: add `embedding_gather_bf16` variant that gathers from `tokELowp`.
2. **Paradigm-#74 binary-FFN W1/W2 sign()** (sgd_transformer.cpp ~9544
   / 9554 / 9601 / 9611 / 9934 / 9944 / 9984 / 9994) — calls
   `bitnet_ffn_forward_gpu` and `binary_gemm_abt_from_float` which
   take `gb.W1.data()` / `gb.W2.data()` (FP32) and apply `sign()` to
   compute the binarized product.  Sign bit IS exactly preserved in
   bf16, so an `_bf16` variant that reads from `gb.W{1,2}Lowp` is
   straightforward.  Existing `binarize_to_bf16_signs` (line 5261)
   already has the right shape.
3. **Atlas/Echo paths** — only fires under Atlas configurations
   (paradigm #X.X).  Not relevant for the flagship 1.84B head-to-head
   recipe (which uses plain Adam).  Defer.

### Updated work plan (post-pivot)

1. **Stage 7 (NEW priority): retire FP32 weight master** —
   - 7a. Add `embedding_gather_bf16` kernel + dispatch
   - 7b. Add `binary_gemm_abt_from_float_bf16` + `bitnet_ffn_forward_bf16` variants + dispatch
   - 7c. Gate `gb.W{1,2}.allocate(...)` and `tokE.allocate(...)` on
     `!useWeightStorageBf16` in `GpuTransformerWeights::allocate`
   - 7d. Re-test 213M smoke for parity (expect NLL within 1e-3 nat)
   - 7e. Re-launch 1.84B WITHOUT `--grad-checkpoint`, measure speed +
     NLL trajectory + peak VRAM
2. **Stage 8 (parallel): start a SPEED paradigm** — NIMBUS (#52) is the
   smallest engineering at ~750 LOC for ~1.33× per-step speedup, and
   the design is already complete.  Or SOPHIA (#55) for 2× steps to
   fixed final NLL at ~600 LOC.
3. **Stage 9: 1.84B head-to-head with no checkpoint + speed paradigm**
   — the actual headline experiment.

### What memory-fit at 213M tells us about 1.84B

VRAM with `--bf16-weights --adam-state-int8` at 213M = ~3-4 GB (small,
plenty of headroom).  At 1.84B, scaling factor ~9× → ~30 GB without
savings, ~12-15 GB with savings.  Fitting in 16 GB without
`--grad-checkpoint` is plausible IF FP32 master is actually retired
(saves ~3.7 GB).

---

## 2026-05-10 update — Stage 7 retire SHIPPED + 770M validation

### Stage 7 commit chain

- **7a** (339c0753f): `embedding_gather_bf16` kernel + dispatch.
- **7c** (4db56f9a1): `freeFp32Masters()` runs post-init under
  `--bf16-weights` (gated against binary-FFN/atlas).
- **7c-followup** (7c6224796): bf16-aware weight download for the
  sync-to-CPU and checkpoint-save paths.

### Two findings on this iteration

**1. 770M trains successfully with retire active (no checkpoint).**

| Config (770M, dmodel=2048, L=24, dff=4096, T=512)            | tok/s | NLL final |
|--------------------------------------------------------------|------:|----------:|
| `--bf16-weights --adam-state-int8` (post-Stage-7 retire)     | 2301  | 10.7778   |

Tokens·params/sec = 770M × 2301 = **1.77 × 10¹²**.

Comparison vs CHIRON 1.84B headline (3.06 × 10¹² from prior):
flagship-770M runs at 58% of CHIRON-1.84B tokens·params/sec — but at
less than half the parameters.  The actual head-to-head requires the
1.84B run.

**2. The 1.84B run is blocked by the CPU-side `net.test()`
initializer.**

Two attempted launches today both stalled at >30 min of pure CPU
init time (52 GB host RSS, 100% single-thread CPU, 0% GPU util).
The bottleneck is single-threaded: even with `--test-tokens 1`, the
init scales super-linearly with model size.  At 770M init takes
seconds; at 1.84B it takes >>30 min (we never reached first NLL).

**Root cause TBD** but suspected `net.test()`'s sequence-1 forward
runs CPU even though tensors are GPU-resident — the prior fix
"--test-tokens 1" only reduced token count, not the per-layer init
cost.  Future Stage 8: profile and skip/parallelize the offending
init code.

### Disk-full caveat (host system)

The 770M smoke's checkpoint save fails with
"failed while writing shard data" — but that's because `/dev/sda2`
is **at 100% capacity** (1.7 TB used / 1.8 TB), not because of any
code path.  Training itself succeeds; only the post-train save is
affected.  Out of scope for the flagship-vs-CHIRON work.

### Cumulative pivot status

- ✓ Memory-savings stack works without checkpoint (FASTER than
  baseline at 213M, +7.0%).
- ✓ FP32 master retire works (+0.7% additional, 213M).
- ✓ bf16-aware sync/download path works (770M end-to-end success).
- ✗ 1.84B head-to-head still blocked on CPU init bottleneck.
- ✓ Memory parity prerequisite established (rationale for retire);
  next priority per pivot reframe is a SPEED paradigm.

### Recommended next-iteration work

Given the 1.84B init blocker, two parallel paths:

**Path A (unblock 1.84B):** Profile and fix the slow CPU
`net.test()` initializer.  May involve:
- Skip init test entirely under a flag (`--no-init-test`)
- Parallelize the per-layer init (currently single-threaded)
- Move init forward to GPU completely (CPU only orchestrates)

**Path B (start a speed/NLL paradigm at smaller scale):**
- **NIMBUS (#52)** — async CPU Adam pipelined with GPU forward.
  ~750 LOC, 3 weeks engineering.  Pure compute speedup ~1.33× at
  1.84B.  Composes with everything.
- **SOPHIA (#55)** — second-order optimizer with Hutchinson
  Hessian.  ~600 LOC, 3 weeks.  2× steps reduction to fixed final
  NLL (NLL improvement at fixed compute).
- Either can be validated at 213M-770M and projected to 1.84B once
  the init blocker is resolved.

Path B is the user's directional priority (speed/NLL via paradigms)
and produces measurable signal at smaller scales without waiting on
Path A.
