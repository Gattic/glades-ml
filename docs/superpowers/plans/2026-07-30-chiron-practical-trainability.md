# CHIRON Practical Trainability Plan

**Date:** 2026-07-30

**Status:** terminal P4 NO-GO (2026-07-30); P0-P3 passed, P4 failed, and P5-P7 were not authorized

## Final outcome

Correctness and exact no-slide decode engineering passed. The frozen native-geometry
24,576-token cache gate reached 221.086851 tokens/s at a sampled 7,744 MiB peak on the
RTX 4080 SUPER, with an aggregate trajectory identical to the independent-cache baseline.

The small-scale trainability gate then stopped the plan. CHIRON passed all one-document
runs, but its eight-document free-greedy accuracy was 0.932692 for seed 1337 and 0.911538
for seed 2024, below the required 0.95 per-seed bar. The repaired conventional Transformer
passed all three seeds with 1.0 free-greedy accuracy and 8/8 exact documents. Therefore no
P5 recipe arms, P6 pilot, P7 scale-up, or ARREST reopening was run. Durable evidence is in
`glades-trainer/research/generation-aware/CHIRON_PRACTICAL_P3_CACHE_GATE_2026_07_30.md`
and `CHIRON_PRACTICAL_P4_MEMORIZATION_GATE_2026_07_30.md`.

## 1. Practical-v1 contract

The next CHIRON qualification lineage targets a native `T=2048` context. A model trained at
`T=16384` must not be served at reduced geometry and called equivalent: the frozen 305k sentinel proved
that `2048/128` changes sampled tokens and raw hazard mass relative to `16384/1024`.

Scale one axis at a time:

1. deterministic tiny fixtures;
2. a small parameter-matched CHIRON/Transformer control;
3. a 100–300M native-context pilot;
4. 1B only after the prior gates pass;
5. context extension only after native-context quality is established.

The minimal baseline is causal CHIRON plus CE and only required numerical stabilizers. SIRA, ECHO,
ARREST, PIED, FP8 readout, experimental FFNs, and other auxiliary objectives remain disabled until an
individual preregistered gate enables one of them. The repaired conventional pre-norm Transformer is the
mandatory control.

### Global resource and safety bars

- peak VRAM `<=15.0 GiB`, leaving at least 0.5 GiB headroom on the 16 GiB RTX 4080 SUPER;
- 500-step smoke `<=0.35 GPU-hours` and 2,500-step qualification `<=1.75 GPU-hours`;
- zero OOMs, nonfinite values, true gradient skips, or corrupted checkpoints;
- no logged global gradient above 10;
- uninterrupted and resumed training agree under the frozen deterministic/tolerance contract;
- free-generation quality is load-bearing; held-out NLL alone cannot qualify a model.

## 2. P1 — Correctness and measurement

### Engineering

1. Land synchronized row-only observed generation and permanent race/parity tests.
2. Land adjacent-pair CUDA RoPE and initialize the training RoPE cache.
3. Fix BF16 validation reduction so every scored position contributes exactly once.
4. Keep exact `T+1` record boundaries from the trainer's `record_stride` contract.
5. Freeze trainer/serving precision differences. BF16 readout is the correctness baseline; FP8 must earn
   re-entry through a separate parity/quality gate.
6. Freeze full checkpoint, optimizer, RNG, and data-cursor resume semantics.

### Tests

- observed generation: repeated row/token identity, first/final row equality against independent full
  download, sink/RNG parity, and observer-failure no-commit behavior;
- RoPE: CPU/CUDA adjacent-pair parity for single and fused Q/K kernels, partial RoPE dimensions, and
  forward/inverse round trip;
- validation reduction: `T={1,63,64,65,128,16384}`, padded/unpadded targets, one/multiple CUDA blocks,
  exact valid counts, and CPU/GPU NLL parity;
- records: exact `T+1`, non-divisible tail, empty/undersized input, train/test parity, and schedule
  incompatibility;
- resume: uninterrupted 50 steps versus 25 + save/load + 25, including cursor, optimizer step,
  next-step loss, checkpoint sections, and stochastic tolerance.

**Exit:** all tests and the full build pass; no denominator, boundary, or resume ambiguity remains.

## 3. P2 — Exact no-slide prefill/decode

First prove bounded online state for every causal serving operator: completed SCFA block summaries,
current partial block, per-layer attention state, depthwise-convolution history, reversible `q/p` state,
positional state, QK-Norm, and static WhiSC values.

Proposed API:

```cpp
struct ChironDecodeCache;
bool chiron_decode_prefill(/* model, config, prompt */, ChironDecodeCache&, /* logits */);
bool chiron_decode_step(/* model, config, token */, ChironDecodeCache&, /* logits */);
bool chiron_decode_reset(ChironDecodeCache&);
bool chiron_decode_clone(const ChironDecodeCache&, ChironDecodeCache&);
```

The first version supports only prompt-plus-horizon sequences that fit within `T`. Unsupported sliding
must fail explicitly.

### Tests

Compare cached decoding with synchronized full-forward decoding for prompt lengths
`1,15,16,17,127,128,T-256`, horizons `1,15,16,17,256`, raw greedy/nucleus/production decoding, and
seeds `1337,2024,4242`.

Require identical sampled tokens and top-1 choices, raw-q error `<=2e-3`, bit equality where arithmetic
order is unchanged (otherwise a preregistered BF16 tolerance), identical observer/RNG behavior,
byte-identical WhiSC state, deterministic reset/clone, and unchanged checkpoint hashes.

## 4. P3 — Batched decode economics

Batch independent contexts and decoder branches while sharing immutable prompt-cache state. Profile
prefill, decode kernels, transfers, sampling, VRAM, and host idle gaps separately before tuning kernels.
Only after caching/batching may the implementation consider fused single-token kernels, CUDA graphs,
inference-only INT8/BF16, or exact speculative decoding.

### Tests and bars

- batched output equals serial output;
- batch order and mixed prompt lengths cannot change outputs;
- cloned branches are independent after divergence;
- seeds and cache state cannot leak across rows;
- allocation/destruction returns VRAM to baseline;
- the frozen 24,576-token full collection reaches at least 90 aggregate tokens/s including amortized
  prefill; target 220 tokens/s; peak VRAM remains within the global limit.

Failure to reach 90 tokens/s keeps ARREST closed.

## 5. P4 — Small-scale trainability

Use exact aligned `T+1` records with equal exposure per document. Run one-document and eight-document
fixtures for three seeds at 256/512/1024 exposures using both CHIRON and the repaired conventional
Transformer.

Every CHIRON seed must reach one-document exact generation and, on eight documents, teacher-forced NLL
`<=0.10`, accuracy `>=0.95`, free-greedy token accuracy `>=0.95`, at least 7/8 exact documents, and zero
nonfinite/skip/gradient>10 events.

- Transformer passes and CHIRON fails: stop CHIRON scaling and localize architecture/optimizer weakness.
- Both fail: repair shared fixture/optimizer infrastructure.
- CHIRON passes: proceed to matched corpus training.

## 6. P5 — Minimal recipe selection

Start with CE, causal SCFA, QK-Norm, required WhiSC state, BF16 inner attention, and BF16 readout. Run
matched 500-step and then 2,500-step arms, changing at most two factors in one preregistered experiment.
The repaired Transformer sees identical records, tokens, parameter budget, and evaluator.

Measure held-out NLL/accuracy, synchronized free generation, median/p10 tokens/s, peak VRAM, gradient
p50/p95/p99, clipping, skips, and resume parity. Advance only if NLL is within 0.10 nat of the control,
free generation is noninferior under the frozen margin, p99 gradient is `<=1.15x` control, throughput is
within 5% of the fastest correct CHIRON arm, and CHIRON provides a material memory/throughput advantage
if quality is merely tied.

## 7. P6 — Consumer-GPU pilot

Run one selected 100–300M native-context recipe for approximately 50M tokens with dense early
checkpoints. At each checkpoint evaluate corrected held-out NLL, 128 frozen in-domain prefixes, factual
and multilingual prompts, reference-continuation likelihood, synchronized/cached raw and production
rollouts, repetition/diversity/entropy/margin, semantic rubric, gradients, VRAM, throughput, and wall
clock.

Pass requires no OOM/nonfinite/skip/gradient>10 event, stable throughput, improving held-out NLL, no
teacher-forcing/free-generation divergence, successful resume, and a next-scale projection inside the
agreed consumer-GPU budget.

## 8. P7 — Scale and promotion

Scale model size or context one axis at a time and repeat the correctness, overfit, control, cached
parity, stability, quality, wall-time, and VRAM gates at every rung. Reopen ARREST only if a qualified
base model still has population-level degeneration and cached collection passes both fidelity and cost
gates.

Hard stops: CHIRON fails aligned memorization while the Transformer passes; quality plateaus while
semantics remain at floor; exact decode stays below 90 tokens/s; gradients exceed 10; speed/memory gains
lose quality; or projected runtime exceeds the single-GPU budget.
