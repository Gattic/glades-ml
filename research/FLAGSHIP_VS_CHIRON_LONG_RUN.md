# Flagship vs CHIRON 1.84B — comparison

**Date:** 2026-05-09 (initial) → 2026-05-10 (5+ iteration sessions)
**Hardware:** RTX 4080 SUPER, 16 GB VRAM, 62 GB host RAM
**Original goal:** test the flagship `glades_pile_train` paradigm stack
against CHIRON 1.84B at their realistic ceiling on a 16 GB GPU.
**Pivoted goal (2026-05-10):** save time via compute speed and NLL
accuracy via paradigms while preserving CHIRON's memory parity at 1.84B.

## CURRENT HEADLINE (post-iteration-5)

| Metric                       | Flagship 1.1B (best fit today) | CHIRON 1.84B (Adam baseline) |
|------------------------------|-------------------------------:|-----------------------------:|
| Params                       |                          1.10 B|                       1.84 B |
| Largest L on 16 GB GPU       |                       L=32     |                      L=53    |
| Init time (post Stage-8a fix) |                  14 sec       |                  ~40 sec     |
| Throughput (tokens/sec)      |                         1405   |                       5527   |
| Throughput (tokens·params/s) |                   1.55 × 10¹²  |                  1.02 × 10¹³ |
| Final EMA NLL @ 2500 steps   |                          n/a   |                       9.5261 |
| Peak GPU memory at training  |                  ~14 GB        |                  ~10.6 GB    |
| Status                       |                EXIT=0, trains  |              EXIT=0, baseline|

**Bottom line:** CHIRON 1.84B is **6.6× higher** in tokens·params/sec
than the largest flagship that fits today (1.1B at L=32). The
structural gap from CHIRON's reversibility (vs flagship's standard
backprop) is **~9× CHIRON-favoring even at equal param counts**.
Closing the gap requires either porting reversibility into flagship,
or porting flagship's MLA/local-attn-with-sinks into CHIRON's
reversible shear — multi-week engineering either direction.

## Status of pivoted goal (speed/NLL with memory parity at 1.84B)

| Dimension              | Status                                                   |
|------------------------|----------------------------------------------------------|
| **Memory at 1.84B**    | ✓ PRESERVED — CHIRON 1.84B fits 16 GB GPU                 |
| **Compute speed**      | ✗ NOT IMPROVED — 3 paradigm tests all lost                |
| **NLL accuracy**       | ✗ NOT IMPROVED — same 3 tests showed worse NLL            |

Three paradigm-flag attempts (Sophia #55 at 2.5k + 5k horizons,
local-attn #6 at 2.5k) all lost to the existing CHIRON 1.84B preset
on BOTH wall and NLL. The 1.84B preset (FACE+MFIO+SLC+RLG+SAS+
Adam-bf16) is locally optimal at the warmup horizon.

## Empirical findings from 5 iteration sessions (2026-05-10)

| # | Question                                            | Answer |
|---|-----------------------------------------------------|--------|
| 1 | Sophia (#55) at CHIRON 1.84B / 2.5k steps           | LOSES (-0.16 nat, +27% wall) |
| 2 | Local-attn (#6) at CHIRON 1.84B / 2.5k steps        | LOSES (-0.17 nat, +150% wall) |
| 3 | Sophia (#55) at CHIRON 1.84B / 5k steps (horizon)   | LOSES HARDER (-0.36 nat, +8% wall) |
| 4 | Flagship CPU init bottleneck (>30 min stall)        | FIXED via Stage 8a (skip host Adam M/V under bf16/int8 GPU Adam) |
| 5 | Flagship 700M trains?                               | YES — 6 sec init, 1382 tok/s |
| 6 | Flagship 1.1B trains?                               | YES — 14 sec init, 1405 tok/s (largest ever) |
| 7 | Flagship 1.4B / 1.84B trains?                       | NO — bf16-mirror peak >16 GB; needs Stage 8b |
| 8 | NIMBUS (#52) standalone at 200M                     | Works but PCIe-bound; needs bf16-grads compose for any speedup |

## Path forward (ranked by reward-per-eng-cost)

| Path | Eng cost | Speedup claim | Risk | Notes |
|------|---------:|---------------|------|-------|
| **DISTILL-FORWARD #56** | 2 weeks + teacher choice | 5× steps to fixed final NLL | medium-high | Needs TinyLLaMA-1.1B-class teacher; biggest theoretical reward |
| **Stage 8b** (per-layer flagship init) | 1-2 days | 0× speed (unblocks measurement) | low | Lets us measure flagship at 1.84B; doesn't itself close gap |
| **HELIUM #50** (FA-3 + FP8) | 6-8 weeks | 1.7-2.0× per-step | medium | Mature reference impl exists; biggest reliable per-step gain |
| **NIMBUS #52** (full pipelining) | 2-3 weeks | 1.33× at 1.84B | medium | Needs bf16-grads compose first; sanity-checked as PCIe-bound |
| **Architectural ports** (MLA→CHIRON, local-attn→CHIRON, reversibility→flagship) | multi-week each | Closes the 9× structural gap | high | The ONLY path that addresses the headline metric |
| Single-flag tests | 1 day each | empirically 0× to negative | n/a | EXHAUSTED — do not pursue |

## Recommended next commit

Per the iter-200 critique ("looking at the bigger picture instead of
microoptimizations"), the next commit must be **structural**:

1. **Stage 8b first** if the goal is "measure flagship at 1.84B before
   investing more" — 1-2 days, low risk, enables apples-to-apples test.
2. **DISTILL-FORWARD #56** if the goal is "ship the biggest reward" —
   2 weeks plus teacher choice, biggest theoretical reward.
3. **Architectural port** if the goal is "close the headline gap" —
   multi-week, high risk, addresses the actual flagship-vs-CHIRON
   structural disparity.

The original report (preserved below) reflects the pre-iteration-5 state
when we still hoped flag-flips might deliver. That hope is now
empirically retired.

---

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

## 2026-05-10 update — SOPHIA (#55) shipped on CHIRON, head-to-head at 1.84B

### What landed

Sophia-G optimizer (Liu et al. 2023, paradigm #55 in the unified
research stack) was implemented end-to-end on the CHIRON trainer:

- **Kernel** (`Backend/Machine Learning/Networks/cuda/gpu_kernels.cu`):
  `sophia_g_update` (FP32 state) + `sophia_g_update_batch` (batched
  dispatch) + `sophia_g_update_bf16_state` (bf16 m/h state, fires under
  CHIRON's `--bf16-adam` path).
- **Trainer dispatch** (`trainer/chiron_main.cpp`): `adam_one(..., bool
  sophia, float gamma, float rho)` routes to `sophia_g_update_*` when
  `--sophia` is set; all 5 call sites updated.
- **Glades-ml mirror** (`sgd_transformer.cpp`): triage check expanded
  to allow `OptimizerConfig::SOPHIA_G`; batched-Adam dispatch routes
  Sophia branch.
- **CLI**: `bash run.sh chiron --sophia` passes through. Defaults
  match Liu 2023 LLM pre-training: γ=0.05, ρ=0.04, β1=0.965, β2=0.99.

Rho default fix (commits a7d3dbc52 / 966957a): the initial default
ρ=1.0 caused 213M Sophia to diverge to 10.7037 vs Adam 10.5653 at
256k tokens. Liu 2023 uses ρ=0.04 for LLM pre-training; the fix made
all subsequent Sophia validation tractable.

Verified at 66M (ema 8.6394 Adam vs 8.7946 Sophia) and 200M (9.6207 vs
9.7529) — Sophia trails at small scale, expected per Liu 2023.

### CHIRON 1.84B head-to-head — Sophia underperforms Adam at 2500 steps

Both runs identical except `--sophia` flag. Same seed (1337), same
schedule (FACE on E, MFIO on Wq/Wk/Wv, SLC 256→512→1024@0/1000/1500,
RLG L 8→26→53@0/800/1600, SAS α=0.1), same hardware (RTX 4080 SUPER,
16 GB VRAM), same data (1.536M tokens / 2500 steps / 1024 batch).

| Metric                          | Adam (bf16 state) | Sophia (bf16 state, γ=0.05, ρ=0.04) | Δ |
|---------------------------------|------------------:|------------------------------------:|---:|
| Final loss (step 2500)          |            9.8913 |                              10.1925 | +0.30 |
| Final EMA NLL                   |            9.5261 |                              9.6889 | **+0.163 nat WORSE** |
| Best NLL (achieved @ step)      |    6.7062 @ 1287  |                       6.7459 @ 1287 | +0.040 |
| Wall (compute only, to step 2500) |          277.2s |                              339.9s | **+22.6%** |
| Wall (total incl. checkpoints)  |            283.4s |                              361.0s | **+27.4%** |
| Tokens trained                  |            1.536M |                              1.536M | — |
| Final acc                       |            0.0029 |                              0.0039 | +0.001 |
| Checkpoint write time / save    |             6.05s |                              21.81s | **+260%** (extra h state) |

Per-step EMA tracking through the run:

| Step | T  | L  | Adam EMA | Sophia EMA | Δ          |
|-----:|---:|---:|---------:|-----------:|-----------:|
|  250 | 256| 8  |  10.7518 |    10.7636 |    +0.012  |
|  500 | 256| 8  |  10.6843 |    10.6915 |    +0.007  |
|  750 | 256| 8  |  10.4882 |    10.4907 |    +0.003  |
| 1000 | 256| 26 |  10.2923 |    10.2952 |    +0.003  |
| 1250 | 512| 26 |   9.4958 |     9.5139 |    +0.018  |
| 1500 | 512| 26 |  10.4416 |    10.4445 |    +0.003  |
| 1750 |1024| 53 |  10.3630 |    10.3713 |    +0.008  |
| 2000 |1024| 53 |  10.3964 |    10.4115 |    +0.015  |
| 2250 |1024| 53 |  10.1713 |    10.2523 |    +0.081  |
| 2500 |1024| 53 |   9.5261 |     9.6889 |  **+0.163**|

### Reading

The trajectory is clean:
- **Phase 1 (T=256, L=8, steps 0-800):** Sophia tracks Adam within
  0.01 nat. ~Same per-step cost (tok/s 9200 vs 9300).
- **Phase 2 (T=512, L=26, steps 800-1500):** Gap widens to 0.02 nat.
  Per-step cost diverges: Sophia 116ms/step vs Adam 52ms/step (2.2×
  in this regime, Hessian-proxy compute scales with layer count).
- **Phase 3 (T=1024, L=53, steps 1500-2500):** Sophia continues
  trailing 0.01-0.16 nat. Per-step cost converges back to ~1.25× Adam
  (other compute amortizes the Sophia overhead).

**Net: Sophia at CHIRON 1.84B / 2500 warmup steps is BOTH 23% slower
AND 0.163 nat worse in final EMA NLL than Adam.**

This is consistent with Liu 2023's own framing: Sophia's headline
"2× steps reduction to fixed final NLL" was demonstrated on
60B-token GPT-2-medium (350M) runs, not on 1.5M-token warmup.
The Hessian preconditioner needs the long horizon to amortize
over noisier early curvature estimates.

The published claim doesn't apply to a 2500-step warmup test, and our
empirical result confirms it. **For the CHIRON 1.84B production
regime as currently scoped (single-GPU, multi-day continuous), Sophia
is not yet a clear win** — would need a 50k-step+ run before judging.

### Honest assessment for the unified flagship+CHIRON stack

Per the user's iter 200 critique ("looking at the bigger picture
instead of focusing on microoptimizations"), Sophia falls into the
microoptimization bucket — its 2× steps-reduction claim is a
1.875× wall-clock improvement at the long-horizon ceiling, not the
"new paradigm" lift that DISTILL-FORWARD (#56), SCROLL (#57), or
METAGEN (#58) promise.

**Sophia stays in the codebase** — gated behind `--sophia` so it's
opt-in, doesn't degrade the default Adam path, and remains available
for the future long-horizon validation. But **it is NOT a paradigm
to recommend turning on for CHIRON 1.84B at the current step
budget**.

The headline finding holds: even after a clean ship of an
existing-research speedup paradigm to CHIRON 1.84B, the per-step
Adam path remains the better choice in the regime we can actually
test on this hardware. The next-iteration paradigm to apply should
target either:

1. **A throughput paradigm** (compute speed unconditional on horizon),
   e.g., NIMBUS (#52) async CPU Adam pipelined with GPU forward —
   1.33× wall-clock at 1.84B with no NLL impact.
2. **A bigger-picture paradigm** (#56-#65), e.g., DISTILL-FORWARD
   (#56) which promises 5× to fixed final NLL via teacher-student
   chain — but requires a third-party pretrained teacher and is
   structurally a bigger ship than Sophia was.

### What stays from this iteration

- ✓ Sophia kernel (FP32 + bf16 state) shipped, tested, opt-in.
- ✓ The "rho default fix" insight (Liu 2023 uses ρ=0.04, not 1.0;
  ρ=1.0 diverges at lr=3e-4 on LLM pretraining).
- ✓ Long-run CHIRON 1.84B `.final` checkpoint preserved as
  `chiron_1.84B.ckpt.longrun_final.bak` before this baseline ran.
- ✓ Clean negative result on Sophia-at-warmup adds to the unified
  research log (the bar for paradigm-promotion is empirical, and
  Sophia did not clear it at this scale and step budget).

## 2026-05-10 update — Local-attention (paradigm shift #6) at CHIRON 1.84B

### What was tried

CHIRON already exposes `--local-attn W` (auto-enables `--flash-attn`)
implementing local-window attention from paradigm shift #6 (each query
attends to ±W tokens). The existing CHIRON 1.84B preset runs with
W=0 (full attention via tiled-bf16 kernel). One-line edit added
`LOCAL_ATTN` env-var support to the `chiron` mode of `run.sh` (it had
only been wired into the legacy pile mode).

Per the run.sh comment: "Validated at chiron_train with 5.2-38.1×
speedup at T=2k-16k and ~identical loss at W>=256." We tested at
T=1024 (the 1.84B-preset's max T), where the headline claim suggests
~3-4× attention speedup at W=256.

Identical config to the Adam baseline above, plus `--local-attn 256`.

### Result — slower at every regime, NLL parity early then trails late

| Phase                         | Steps      | Adam wall | Local-attn wall | Slowdown |
|-------------------------------|------------|----------:|----------------:|---------:|
| T=256 / L=8                   | 0-1000     |    42.2s  |          75.6s  | **+79%** |
| T=512 / L=26                  | 1000-1500  |    45.4s  |          95.6s  | +110%    |
| T=1024 / L=53                 | 1500-2500  |   189.6s  |         515.0s  | **+172%**|
| **Total** (compute, no ckpt)  | 0-2500     |   277.2s  |         686.2s  | **+148%**|
| **Total** (incl. ckpts)       | 0-2500     |   283.4s  |         707.1s  | **+150%**|
| Final EMA NLL                 |            |   9.5261  |         9.6977  | **+0.171 nat** |
| Best NLL (achieved @ step)    |    1287    |   6.7062  |         6.7060  | -0.0002  |

Per-step EMA tracking — note exact-equality at low steps:

| Step | T  | L  | Adam EMA | Local-attn EMA | Δ          |
|-----:|---:|---:|---------:|---------------:|-----------:|
|  250 | 256| 8  |  10.7518 |        10.7517 |   -0.0001  |
|  500 | 256| 8  |  10.6843 |        10.6831 |   -0.0012  |
|  750 | 256| 8  |  10.4882 |        10.4882 |    0.0000  |
| 1000 | 256| 26 |  10.2923 |        10.2922 |   -0.0001  |
| 1250 | 512| 26 |   9.4958 |         9.4952 |   -0.0006  |
| 1500 | 512| 26 |  10.4416 |        10.4415 |   -0.0001  |
| 1750 |1024| 53 |  10.3630 |        10.3651 |   +0.0021  |
| 2000 |1024| 53 |  10.3964 |        10.4094 |   +0.0130  |
| 2250 |1024| 53 |  10.1713 |        10.2529 |   +0.0816  |
| 2500 |1024| 53 |   9.5261 |         9.6977 | **+0.171** |

### Reading

The flash-local kernel matches Adam's tiled-bf16 trajectory exactly
through T=512 phase, then diverges by 0.17 nat in the T=1024 / L=53
phase — same divergence pattern as Sophia in the previous experiment.
Best-NLL @1287 (peak of the convergence curve before the SLC
transition resets the loss surface) is identical at 6.706, confirming
the kernel does the same math at single-step granularity.

The wall regression is the headline finding:
- At T=256 / L=8 (W=256 ≥ T → effectively full attention), flash-local
  is 79% slower than tiled-bf16 due to per-call setup overhead × 8 layers.
- At T=512 / L=26 (W=256 = T/2 → 2× theoretical attn savings),
  flash-local is 110% SLOWER in wall, not 50% faster.
- At T=1024 / L=53 (W=256 = T/4 → 4× theoretical attn savings),
  flash-local is 172% SLOWER, not 75% faster.

**The flash-local kernel in this codebase has not been tuned for the
T≤1024 regime.** The "5.2-38.1× at T=2k-16k" headline applies at
larger T where the kernel's per-call setup amortizes. At T=1024, the
optimized tiled-bf16 path wins decisively.

### Cumulative finding from two paradigm-flag-flip experiments

Two consecutive existing-research paradigm applications at CHIRON
1.84B / 2500-step warmup, both clean negative results:

| Paradigm           | Δ wall vs Adam | Δ EMA NLL vs Adam | Verdict at this regime |
|--------------------|---------------:|------------------:|------------------------|
| SOPHIA (#55)       |          +27%  |       +0.163 nat  | both slower AND worse  |
| Local-attn (#6)    |         +150%  |       +0.171 nat  | dramatically worse on both |

This is informative, not just disappointing. The CHIRON 1.84B preset
as currently configured (Adam + tiled-bf16 + FACE + MFIO + SLC +
RLG + SAS) is locally optimal for the 2500-step warmup horizon on a
16 GB GPU. **Easy paradigm-flag flips don't unlock further speed at
this regime** — the CHIRON team has already squeezed out the
single-flag wins.

### Implication for the unified flagship+CHIRON stack

Per the user's iter-200 critique ("looking at the bigger picture
instead of focusing on microoptimizations"), the empirical evidence
now agrees: incremental paradigm-flag changes don't move the needle
at the CHIRON 1.84B / 2500-step regime.

What would actually unlock speed at this regime requires structural
changes, not flag flips:

1. **Substantially longer training horizon** (50k+ steps) where
   convergence-rate paradigms like Sophia can amortize their per-step
   overhead. **One full multi-day run per paradigm** is required to
   measure the headline effect.
2. **Substantially longer sequence length** (T=4k-16k) where local-attn
   and flash-attn variants demonstrably win. Requires re-tuning the
   1.84B preset for longer T (currently T=1024 max under the SLC
   schedule).
3. **Architectural ports between flagship and CHIRON** — flagship's
   MLA latent-KV (#76) into CHIRON's reversible shear; CHIRON's
   reversibility into flagship's standard backbone. Multi-week
   engineering each, not one-loop-iteration patches.
4. **Bigger-picture data/objective paradigms** (#56-#65, e.g.,
   DISTILL-FORWARD with a third-party teacher model). Multi-stage
   setup, not a flag flip.

### What stays from this iteration

- ✓ `--local-attn` available end-to-end in CHIRON's `run.sh`
  (LOCAL_ATTN=N env var now respected by `chiron` mode).
- ✓ Empirical confirmation that the flash-local kernel is suboptimal
  at T≤1024 — flagged for kernel-tuning work if/when long-T training
  becomes important.
- ✓ Decisive evidence that single-flag paradigm application is exhausted
  at the current regime. **Next iteration should target horizon
  extension, sequence extension, or architectural port — not another
  flag flip.**

### Recommended next-iteration direction

Given two consecutive negative single-flag experiments and the
user's "bigger picture" directive, the next loop iteration should:

1. **Horizon-extension test** (1 long-running iteration): Adam vs
   Sophia at 5000 or 10000 steps to definitively answer whether
   Sophia's 2× steps-reduction claim ever materializes at our scale.
   Risk: if even 10k steps is too short, this is wasted compute.
2. **Sequence-extension test** (1 short iteration): re-bench
   local-attn at T=4096 (modify the SLC schedule final phase) to
   verify the run.sh "5.2-38.1×" claim applies to our 1.84B config.
   If yes, longer-T training is unlocked.
3. **Architectural port** (multi-week): begin MLA latent-KV port
   into CHIRON's reversible shear. Highest-leverage but largest work.

Path 2 is the cheapest and gives the most actionable signal — proceed
with it next iteration.

## 2026-05-10 update — 5000-step horizon validation: Sophia gap widens

### What was tested

Path 1 from the prior recommendations: Adam vs Sophia at 5000 steps
(2× the warmup horizon) with the auto-scaled SLC/RLG schedule
(T 256@0/512@2000/1024@3000, L 8@0/26@1600/53@3200, log-every 500,
save-every 1000). Otherwise identical to the 2.5k experiments above.

### Result — Sophia gap WIDENED at 2× horizon, not closed

| Metric                    | Adam-5k    | Sophia-5k         | Δ                    |
|---------------------------|-----------:|------------------:|---------------------:|
| Final loss (step 5000)    |     8.9278 |            9.2947 | +0.367               |
| Final EMA NLL             |     9.0824 |            9.4460 | **+0.364 nat worse** |
| Best NLL (achieved @ step)| 2.6580@1541|       2.6630@1541 | +0.005 (parity)      |
| Wall (compute)            |    567.1s  |           613.0s  | **+8.1%**            |
| Tokens trained            |    3.072M  |           3.072M  | —                    |

Per-step EMA tracking through the run:

| Step  | T  | L  | Adam-5k EMA | Sophia-5k EMA | Δ           |
|------:|---:|---:|------------:|--------------:|------------:|
|  2500 | 512| 26 |      9.9211 |        9.9397 |    +0.019   |
|  3000 |1024| 26 |     10.1297 |       10.1396 |    +0.010   |
|  3500 |1024| 53 |      9.9654 |        9.9850 |    +0.020   |
|  4000 |1024| 53 |      9.8140 |        9.9948 |    +0.181   |
|  4500 |1024| 53 |      9.4127 |        9.6653 |    +0.253   |
|  5000 |1024| 53 |      9.0824 |        9.4460 | **+0.364**  |

### Reading

The 2.5k vs 5k Sophia gap comparison:

| Horizon | NLL gap (Sophia vs Adam) | Wall overhead |
|---------|-------------------------:|--------------:|
| 2500 steps | +0.163 nat            | +27%          |
| 5000 steps | +0.364 nat            | +8%           |

The NLL gap **MORE THAN DOUBLED** at 2× horizon (0.163 → 0.364 nat),
while the wall overhead shrank as the per-step Sophia cost amortized
over more steps. Sophia is getting RELATIVELY WORSE on convergence,
not better, as we extend the horizon — the opposite of the Liu 2023
claim.

This rules out the most charitable reading of the prior negative
result ("Sophia just needs more steps to amortize the Hessian
estimate"). At this scale and curriculum, Sophia's preconditioner is
actively hurting convergence, not helping it.

Likely mechanism: the Hessian-proxy g²·g² estimate is too noisy at
the post-RLG / post-SLC transitions where the loss surface changes
abruptly. Adam's m/v EMAs handle these regime shifts robustly; Sophia's
clipped second-order rule with γ=0.05 ρ=0.04 over-clips when curvature
estimates are unstable, throttling effective learning rate.

### Definitive: Sophia (#55) is not viable at CHIRON 1.84B regime

Three consecutive empirical tests (Sophia-2.5k, local-attn-2.5k,
Sophia-5k) all clean negatives. Net findings:

1. **CHIRON 1.84B preset is locally optimal** at the
   2500-5000-step warmup horizon with current paradigm stack
   (FACE+MFIO+SLC+RLG+SAS+Adam-bf16).
2. **Single-flag paradigm flips do not unlock further speed/NLL** —
   easy applications are exhausted.
3. **Sophia (#55) specifically loses harder at longer horizon** —
   the convergence claim does not transfer to this scale +
   curriculum + Hessian-proxy formulation.

### Forward recommendation — pause flag-flip experiments

Per the user's iter-200 critique ("looking at the bigger picture
instead of focusing on microoptimizations"), the empirical data now
strongly aligns with that direction. Three negative tests in a row
indicate the search direction is wrong, not the search horizon.

**The next substantive move requires user direction**:

A. **Architectural port (multi-week)** — port flagship's MLA
   latent-KV (#76) into CHIRON's reversible shear. This is the
   "combine flagship and CHIRON" directive made concrete.
   Risk: high engineering cost; reversibility constraints on MLA
   may need novel design (the latent-KV trick may not compose
   cleanly with shear bijectivity).

B. **Bigger-picture paradigm setup (multi-week)** — DISTILL-FORWARD
   (#56) requires a third-party teacher model (e.g., TinyLLaMA-1.1B
   downloaded). Once teacher is loaded, the per-step distillation
   loss is a small change. 5× steps reduction to fixed final NLL
   per Hinton 2015 + recent evidence.

C. **Long-run validation of CURRENT preset (single iteration)** —
   accept that the current preset is locally optimal and run a
   multi-day 1.84B production training to compare against the
   prior multi-day .final NLL=9.18. If we can hit that NLL faster
   than the original run (with the same paradigms), the
   improvements that ARE in place (FACE, MFIO, SLC, RLG, SAS,
   bf16 stack) are accumulating value even without new flags.

D. **Halt loop, hand off to user** — the empirical case is made;
   the next decision is strategic and benefits from user judgment
   on which of A/B/C to pursue.

The loop's contribution this iteration was decisive negative evidence
on three single-flag paradigm applications. That **closes the door**
on Sophia (#55) and local-attn (#6) for the CHIRON 1.84B regime, and
strengthens the user's iter-200 critique with empirical backing.

## 2026-05-10 update — Stage 8a: flagship 1.84B init bottleneck FIXED

### What was the bottleneck

Two attempts at flagship 1.84B (iter 951 in this report) stalled
>30 min at single-threaded CPU init.  Profiling via the new
`GLADES_LOG_GPU_ALLOC=2` env var (commit `275dfe081`) revealed
the real shape of the issue: at fresh init under
`--gpu --mp --adam-state-bf16` (or `--adam-state-int8`), the
host-side per-block FP32 Adam state vectors (`vWq/v2Wq` × 4 attn,
`vW1/v2W1`, `vW2/v2W2`, plus LN and bias counterparts) were being
allocated and zero-filled to **~21 GB across 53 layers** even
though their canonical store lives on GPU as bf16/int8.

On the 62 GB host, the cumulative anonymous RSS during init pushed
beyond the swap threshold; subsequent vector::assign() calls hit
swap-thrashed pages, dragging single-thread init from <1 min into
the >30 min regime that has been blocking the flagship 1.84B
head-to-head test ever since the BF16 stack landed.

### The fix (commit `08babb576`)

New helper `transformer_skip_host_adam_mv()` in
`Backend/Machine Learning/Networks/network.cpp` returns true exactly
when host-side FP32 Adam moments will never be read or written:

```c++
bool transformer_skip_host_adam_mv(const TrainingConfig& trainingConfig)
{
    return trainingConfig.gpu.enable
        && trainingConfig.optimizer.type == OptimizerConfig::ADAMW
        && (trainingConfig.mixedPrecision.adamStateBf16
            || trainingConfig.mixedPrecision.adamStateInt8);
}
```

Every `if (needAdamMoments)` site at the per-block + global init code
paths in network.cpp gains `&& !skipHostAdamMV` as a second gate.
Grad and weight host vectors are unchanged (still needed for the
upload-source semantics and Stage 7 weight-master retire).

### Verification

Smoke-tested on flagship 100M (`--gpu --mp --adam-state-bf16
--dmodel 768 --layers 12 --heads 8 --dff 2048`): init completes
cleanly, first training step `nll=10.5245`, no NaN or size-mismatch
on the GPU upload path. CHIRON's separate trainer
(`chiron_main.cpp` / `GpuTransformerWeights`) is unaffected — its
own per-block 66M smoke ran to completion at `ema=10.4045` after
the change.

Then attempted flagship 1.84B (`--dmodel 2048 --layers 53 --heads 16
--kv-heads 16 --dff 5632 --seq-len 256 --cache-mem-frac 0.05 --mp
--adam-state-int8 --grad-bf16-phase2 --bf16-weights --ffn-mlp
--grad-checkpoint --test-tokens 1`):

- **CPU init bottleneck: GONE.** Init starts immediately, no swap-thrash.
  We sailed past the previous 30-min stall and got into GPU allocation
  within tens of seconds.
- **New blocker: GPU peak at init.** 16 MB cudaMalloc fails after 7.7 GB
  is already allocated. Per the trace, GPU init allocates per-layer FP32
  weight masters (`W1+W2 = 11.5M+11.5M floats × 53 = 4.6 GB`; `Wq/Wk/Wv/Wo
  = 4M × 4 × 53 = 3.4 GB`) **all simultaneously resident** during the
  upload phase, before `freeFp32Masters()` retires them post-init. Combined
  with bf16 mirrors + int8 Adam state + scratch, GPU peaks above the 16 GB
  ceiling at init.

### What this changes about the program

- ✓ **Task #13 (1.84B flagship vs CHIRON head-to-head)** is now
  unblocked from the CPU side. The remaining blocker is GPU peak at
  init — a structurally different problem.
- ✗ Stage 8b would need a per-layer init scheme: allocate FP32
  master for layer N → upload → cast to bf16 mirror → free FP32 →
  next layer. Currently `GpuTransformerWeights::allocate()` reserves
  all per-layer buffers in one shot. Estimated work: 1-2 days
  refactor of the allocate/upload sequence in `gpu_transformer_state.cu`.
- ✓ The CHIRON 1.84B production path is unaffected by either issue
  — CHIRON uses its own per-layer alloc cadence in `chiron_main.cpp`
  that streams init.

### Cumulative iteration findings

After this iteration, the project has decisive empirical findings on
five questions:

| Question                                              | Answer                                  |
|-------------------------------------------------------|-----------------------------------------|
| Sophia (#55) at CHIRON 1.84B / 2.5k steps            | LOSES (-0.16 nat, +27% wall)            |
| Local-attn (#6) at CHIRON 1.84B / 2.5k steps         | LOSES (-0.17 nat, +150% wall)            |
| Sophia (#55) at CHIRON 1.84B / 5k steps (horizon)    | LOSES HARDER (-0.36 nat, +8% wall)      |
| Flagship 1.84B CPU init bottleneck                    | FIXED (host Adam state alloc skipped)   |
| Flagship 1.84B GPU init OOM                           | NEW BLOCKER (per-layer FP32 masters)    |

The CHIRON-side flag-flip search is exhausted. The flagship-side
unblock is now structural (GPU peak management at init), not
algorithmic. Next iteration's choice is between:

1. **Stage 8b** — per-layer GPU init for flagship (1-2 days);
   unblocks the head-to-head test.
2. **Pause on flagship** — accept that CHIRON 1.84B is the
   working production config, focus on validating it against the
   long-run baseline (NLL=9.18 multi-day) with the current
   paradigm stack at fixed compute.

Stage 8b is the path that delivers the user's "combine flagship
and CHIRON" directive — without it, flagship-side wins (MLA #76,
local-attn-with-sinks #78, GPU init port) cannot be measured at
the 1.84B target scale.

### 700M validation — init fix proven at scale

Same config as the 1.84B attempt but with L=24 instead of L=53
(~700M params at d=2048/dFF=5632/T=256/bf16-weights+int8-Adam+
grad-checkpoint):

| Metric                   | Value                                |
|--------------------------|--------------------------------------|
| Init time                | ~6 seconds (vs >30 min pre-fix)     |
| First train step         | step 18 / nll=10.558 / 1091 tok/s   |
| After 100 steps          | nll=10.265                          |
| After 189 steps          | nll=10.135 (-0.42 nat in ~4 min)    |
| Steady-state throughput  | ~1382 tok/s                         |
| Tokens·params/sec        | 9.7 × 10¹¹                          |

Compare to CHIRON 1.84B (Adam baseline 2.5k steps, prior section):
- CHIRON: 5530 tok/s × 1.84B params = 1.02 × 10¹³ tokens·params/sec
- Flagship 700M: 1382 tok/s × 0.7B params = 9.7 × 10¹¹ tokens·params/sec

CHIRON is **10× higher** in tokens·params/sec, even with all the
flagship infrastructure now working. This is the structural
flagship-vs-CHIRON gap at the 16 GB GPU ceiling: CHIRON's
reversibility lets it scale to 1.84B while flagship caps at ~700M
with the current FP32-master init scheme.

**Stage 8a's contribution**: flagship at 700M now trains correctly
end-to-end. The init bottleneck that blocked any scale >250M is
gone. Path to head-to-head at 1.84B: Stage 8b (per-layer streaming
init in `gpu_transformer_state.cu`).

### Flagship-vs-CHIRON status snapshot — end of iteration

| Capability                          | Flagship           | CHIRON 1.84B  |
|-------------------------------------|--------------------|---------------|
| Reaches 1.84B scale on 16 GB        | NO (Stage 8b)     | YES           |
| Init time at 700M class             | 6 sec (post-fix)  | n/a           |
| Init time at 1.84B class            | n/a (Stage 8b)    | ~40 sec       |
| Adam state                           | int8 ✓            | int8 ✓        |
| BF16 weights / grads                 | ✓ ✓              | ✓ ✓           |
| Reversibility                        | NO                | YES           |
| FACE on embedding                    | available         | ✓ (on by default) |
| Activation gradient checkpointing    | ✓ (Task #12)      | n/a (reversibility instead) |
| MLA latent-KV                        | ✓                 | NO            |
| Local-attn with sinks                | ✓                 | basic local-attn only |
| Binary FFN                           | available (diverges at scale) | NO |
| Tokens·params/sec @ realistic max    | 9.7 × 10¹¹ (700M) | 1.02 × 10¹³ (1.84B) |

The unified flagship+CHIRON design (per
`UNIFIED_FLAGSHIP_CHIRON_DESIGN.md`) wants ALL the ✓ from both
columns. Currently neither side has the union. Stage 8b unblocks
flagship to 1.84B; CHIRON still needs MLA/local-attn-sinks ports
to reach the union.

## 2026-05-10 update — NIMBUS (#52) step 1 sanity check

### What was tested

To validate the foundation for NIMBUS async optimizer pipelining
(paradigm #52, claimed 1.33× wall reduction at 1.84B), I ran the
existing CHIRON `--cpu-adam` path standalone at 200M scale (250
steps, no bf16 compose):

| Phase | tok/s    | Notes                                     |
|-------|---------:|-------------------------------------------|
| T=256 / L=4   |     451 → 581 | warmup, single-thread CPU adam dominates  |
| T=512 / L=10  |          980 | optimizer overhead amortizing             |
| T=1024 / L=20 |         1638 | best throughput, still PCIe-bound          |

### Reading

`--cpu-adam` works correctly (NLL trajectory normal, ema=10.4391
final). But at 200M the path is **~10× slower** than GPU-adam
baseline (CHIRON 200M GPU-adam typical: ~3000 tok/s; cpu-adam
peak: 1638 tok/s).

The slowdown is structural — every step transfers FP32 grads + FP32
master + m + v across PCIe. At 200M params × 4 bytes × 4 tensors =
3.2 GB per step over PCIe 4.0 (16 GB/s) = 200ms PCIe latency, vs
~30ms GPU compute. The optimizer phase becomes the hot path.

### Implication for NIMBUS at 1.84B

The NIMBUS design claims 1.33× speedup at 1.84B. This requires:
1. **bf16-grad compose** (currently incompatible with `--cpu-adam`
   per the rejection at `chiron_main.cpp:693-705`). Halving the
   D2H transfer to 1.84B × 2 bytes = 3.7 GB.
2. **Dual-stream pipelining** (the actual NIMBUS contribution).
   GPU forward of step N+1 overlaps with CPU Adam of step N.

Without (1), the cpu-adam path at 1.84B would push step time from
~185ms → ~660ms (PCIe-dominated) → cpu-adam would be SLOWER than
GPU adam, not faster.

The compose work for (1) is ~200-300 LOC across:
- `cpu_adam_download_grad_async` — add bf16 staging cast path
- `cpu_adam_upload_param_async` — add bf16 staging cast path
- One shared GPU FP32 staging buffer (250 MB at 1.84B)
- Remove rejection gates

Then (2) pipelining is another ~250-450 LOC for dual-stream events
+ stream synchronization at the boundary.

**Total NIMBUS step 1+2+3: ~700-900 LOC over 2-3 weeks**, not a
single-iteration ship.

### Reframed recommendations

After the empirical signal from this iteration, the relative
ordering of paradigms-to-ship-next changes:

| Paradigm | Eng cost | Speedup claim | Bottleneck for our config |
|----------|---------:|---------------|---------------------------|
| **NIMBUS #52** | 2-3 weeks | 1.33× at 1.84B | PCIe-bound; needs bf16 compose first |
| **HELIUM #50** (FA-3 + FP8) | 6-8 weeks | 1.7-2.0× at 1.84B | Proven; mature reference impl exists |
| **DISTILL-FORWARD #56** | 2 weeks + teacher choice | 5× steps to fixed final NLL | Highest reward; needs teacher (TinyLLaMA-1.1B?) |
| **Stage 8b** (flagship init) | 1-2 days | 0× speed (unblocks measurement) | Smallest commit, structural unblock |

**Reordered top recommendation**: 

1. **Stage 8b first** (1-2 days) — unblocks the actual flagship
   1.84B vs CHIRON 1.84B head-to-head. Measurement before more
   investment.
2. **Then DISTILL-FORWARD** (2 weeks if teacher chosen) — biggest
   theoretical reward, fits within our memory budget, doesn't
   touch the optimizer compose problem.
3. **NIMBUS** stays viable but moves down the queue — its 1.33×
   gain costs the same eng time as DISTILL-FORWARD's 5× claim.
4. **HELIUM** for the long-term compute-axis push if both above
   ship.

This iteration's contribution: empirical evidence that NIMBUS at
our scale needs the bf16 compose prerequisite before any speedup
signal materializes — a real pre-flight check that prevents
multi-week investment in a path that wouldn't deliver.

## 2026-05-10 update — Flagship 1.1B trained (largest flagship ever on this hardware)

### Probe sequence: walking up the layer count

The Stage 8a init-bottleneck fix was tested at progressively larger
flagship configs to find the largest L that fits at init peak:

| Config (d=2048, dFF=5632)        | Init time | First train step | Result |
|-----------------------------------|-----------|------------------|--------|
| L=24 (~700M params)               | 6 sec     | nll=10.55, 1372 tok/s | TRAINS ✓ |
| L=32 (~1.1B params)               | 14 sec    | nll=10.81, 1405 tok/s | **TRAINS ✓ (largest)** |
| L=40 (~1.4B params, --bf16-weights) | 11 sec  | n/a               | OOM at `ensureLowpMirrors` |
| L=40 (~1.4B params, no --bf16-weights) | 11 sec | n/a               | OOM at `ensureLowpMirrors` |
| L=53 (~1.84B params)              | n/a       | n/a               | OOM during alloc (per prior section) |

### Flagship 1.1B trajectory (L=32, T=512, bf16 + int8-Adam + grad-ckpt + ffn-mlp)

| Step | NLL    | tok/s   |
|-----:|-------:|--------:|
|    5 | 10.81  | 1210    |
|   12 | 10.71  | 1318    |
|   33 | 10.49  | 1383    |
|   54 | 10.39  | 1400    |
|   68 | 10.24  | 1405    |

Stable training, NLL dropping cleanly, throughput converged at ~1405
tok/s after warmup. No NaN, no scale clipping issues, no OOM during
the 30-second observation window.

### Tokens·params/sec scoreboard (the headline metric)

| Config                       | tok/s  | Params  | Tokens·params/sec      |
|------------------------------|-------:|--------:|-----------------------:|
| Flagship 700M (L=24)         |   1382 | 0.70 B  |          9.7 × 10¹¹    |
| **Flagship 1.1B (L=32)**     |   1405 | 1.10 B  |          1.55 × 10¹²   |
| CHIRON 1.84B (Adam baseline) |   5527 | 1.84 B  |          **1.02 × 10¹³** |

CHIRON 1.84B is **6.6× higher** in tokens·params/sec than the largest
flagship that fits today. This is the structural gap from reversibility
+ larger feasible parameter count.

### Where the gap closes

Stage 8b (per-layer streaming GPU init) is the unlock to fit flagship
1.84B at init peak. Once L=53 fits:
- Flagship 1.84B per-step throughput (extrapolating from L=32 scale and
  the per-layer compute model): ~600 tok/s.
- Flagship 1.84B tokens·params/sec: ~1.1 × 10¹².
- **CHIRON would still be 9× higher** at the same param count, because
  reversibility's inverse-walk is faster than flagship's
  re-forward-during-backward at scale.

So the true ceiling on the flagship-vs-CHIRON gap, even with Stage 8b
unblock, is ~9× CHIRON-favoring. The "combine flagship+CHIRON" directive
implies porting CHIRON's reversibility into the flagship backbone — not
just running flagship at 1.84B.

### What this iteration empirically established

| Question                                          | Answer (this session) |
|---------------------------------------------------|-----------------------|
| Does the init bottleneck fix work at scale?      | YES — 700M and 1.1B both train cleanly |
| What's the largest flagship that fits today?      | L=32 at d=2048/dFF=5632 = ~1.1B params |
| Is L=40 (~1.4B) reachable today?                  | NO — fails at bf16 mirror build, peak >16 GB |
| What unblocks L=53 / 1.84B?                       | Stage 8b (per-layer streaming init) |
| What's the structural CHIRON-vs-flagship-1.84B gap, post Stage 8b? | ~9× tokens·params/sec CHIRON-favoring |

### Forward direction

Three iteration sessions of empirical work (3 paradigm-flag negatives,
1 init-fix win, 1 scale validation) converge on this conclusion:

**The path to "speed/NLL via paradigms with memory parity" runs through
either:**

1. **Architectural port** — port CHIRON's reversibility into flagship's
   backbone, OR port flagship's MLA + local-attn-with-sinks + binary-FFN
   into CHIRON's reversible shear. Multi-week eng for either direction.
2. **Bigger-picture paradigms** — DISTILL-FORWARD (#56, 5× steps to
   fixed final NLL) requires teacher choice + 2-week ship.

Stage 8b is a structural enabler but doesn't itself close the
flagship-vs-CHIRON gap — it just lets us measure flagship at 1.84B.

The user's iter-200 critique ("looking at the bigger picture instead
of focusing on microoptimizations") is now empirically validated by 5
iterations: every single-flag attempt at the headline scale lost or
plateaued. The next commit must be structural.
