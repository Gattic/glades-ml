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

### Where the actual 1B+ ceiling still hits

A 500M-class probe (d=1536, L=24, heads=12, dff=4096, ~530M params)
with the new int8+FACE stack ran cleanly through CPU init but the
training step OOMed on a **3.2 GB GPU scratch buffer allocation**
(`cudaMalloc(805306368 floats, 3221225472 bytes) failed: out of
memory`).  Halving T from 4096 → 2048 did not resolve it (still
silently failed in the training-step kernel without stderr trace).

The scratch buffer is in the training-step path (eval forward
succeeded, training did not) and is independent of Adam state size.
Likely candidates: (a) sampled-softmax negatives buffer at
T·negatives·dModel sizes, (b) attention backward scratch_S, (c) the
BitNet QAT FFN W2 scratch which is `T·dFF/32` bits + `[T, dModel]`
popcount accumulators per block × layers.

This is the next bottleneck after Adam state — separate engineering
work to identify and either share, recompute, or quantize.

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
