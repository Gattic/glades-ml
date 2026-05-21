# EALRMN Phase-1 GPU Prototype — Design

**Date:** 2026-05-19. **Hardware:** RTX 4080 SUPER (16 GB, compute 8.9, CUDA 12.0). **Status:** Implementing Option C from `research/EALRMN_WRITEUP.md` — production-scale GPU verification of the EALRMN architecture against multi-head Transformer and RNN baselines.

## Motivation

The CPU Phase-0 sequence (0a-0k) culminated in a negative result at small scale (m≤64, T≤256): RNN beats EALRMN-attmem decisively at every T≥128. Three interpretations remain:

- (a) **Honest small-scale negative** — production scale (m=1024+, T=4096+) might compound the mechanism gains.
- (b) **Structural negative** — 4-slot fixed-decay memory has same O(m) scaling as RNN state; no asymptotic advantage.
- (c) **Hypothesis-level negative** — mechanism stacking does not produce additive gain.

Phase-1 tests interpretation (a) directly with GPU implementation at production scale.

## Configuration matrix

| Axis | Values |
|------|--------|
| m (hidden dim) | 256, 512, 1024 |
| T (context) | 2048, 4096, 16384 |
| Seeds | 0, 1, 2, 3, 4 (5 seeds) |
| Models | ealrmn_attmem, rnn, transformer_1L, transformer_2L |
| Tasks | needle, hmm, syntheticlm |

Full Cartesian: 3 × 3 × 5 × 4 × 3 = 540 runs. We will down-select after smoke-test correctness verification.

## Architectural specifications

### EALRMN (ealrmn_attmem)
- **Embedding**: V → m (vocab size 32 for needle, 8 for hmm, 256 for syntheticlm)
- **Encoder linear (optional)**: m → m
- **Recurrence**: s_t = K·s_{t-1} + W_in·z_t, K with spectral radius < 1 (init scale 0.95/√m)
- **Memory**: 4 slots, M_t[j] = (1-λ_j)·M_{t-1}[j] + λ_j·gate_j(s_t)·z_t, decays λ = [0.5, 0.1, 0.01, 0.001]
- **Attention readout**: q = W_q·s_T, α = softmax(q·M[j]/√m), r = Σ α_j·M[j], feat = concat(s_T, r)
- **Readout**: W_out·feat → logits

Parameter count at m=1024, V=32: ~5.3M (embedding 32×1024=32K; encoder 1024×1024=1M; recurrence K+W_in 2M; memory gates 4×1024=4K; attention readout W_q 1M; output 2048×n_classes).

### RNN baseline
- Same encoder
- **Recurrence**: s_t = tanh(W_h·s_{t-1} + W_in·z_t) (standard RNN with tanh)
- **Readout**: W_out·s_T → logits
- Param count ~3.3M at m=1024

### Transformer baseline (transformer_1L, transformer_2L)
- Sinusoidal positional encoding
- L layers of: MHA(num_heads=8) + LayerNorm + MLP(4m) + LayerNorm
- Pre-norm residual stream
- **Readout**: mean-pool x[T-1] → W_out → logits (or [CLS] if applicable)
- Param count ~12M (1-layer) / 25M (2-layer) at m=1024

## Tasks

### needle (extended)
- T tokens. Marker M_k at random position, followed by value V_k 2 tokens later. Query token Q_k at position T-1.
- Goal: predict V_k given Q_k.
- Filler tokens drawn uniformly from filler vocab.
- Variants: needle_short (T=2048), needle_med (T=4096), needle_long (T=16384).

### hmm (scaled)
- 8-state HMM, transition matrix sampled per task instance.
- Observation = state with 20% noise injection.
- Goal: predict hidden state at T-1.
- T = 2048 / 4096 / 16384.

### syntheticlm
- Markov chain with 256 states, order-2 dependencies (state_t depends on state_{t-1}, state_{t-2}).
- Cross-entropy loss over per-token predictions.
- Standard LM training: predict next token.
- T as above.

## Training

- Optimizer: AdamW, β1=0.9, β2=0.95, ε=1e-8, weight_decay=0.01
- LR: 3e-4 default, with cosine decay
- Warmup: 200 steps
- Batch size: 8 for T=2048, 4 for T=4096, 2 for T=16384
- Grad accumulation: 1
- Max steps: 5000 (needle, hmm), 10000 (syntheticlm)
- Eval interval: every 500 steps on held-out batch (16 sequences)
- Grad clip: 1.0 global norm
- BF16 forward, FP32 master weights, FP32 gradients (standard mixed precision)

## Memory budget

At m=1024, T=16384, B=2:
- Embeddings forward: B·T·m·4 = 128 MB (FP32) or 64 MB (BF16)
- Recurrence states saved: B·T·m·4 = 128 MB
- Memory saved per step: B·T·4·m·4 = 512 MB (concern!)
- Attention scores (Transformer 2L 8H): B·H·T·T·2 = 8 GB (concern, need streaming)

**Mitigation:** 
- Gradient checkpointing for recurrence states (recompute in backward)
- Block-streamed attention (FlashAttention-2 style) for Transformer at T≥4096
- Memory M_t can use sliding window save (every K steps) and recompute

## Determinism

- cuRAND for seeded init, fixed seed per (model, task, m, T, seed) tuple
- cuBLAS handle per stream
- Deterministic atomic-free reductions where possible (use block-level reductions)
- Same hyperparameters across baselines (lr, batch, steps, optimizer)

## Falsification criteria

The hypothesis "EALRMN scales to compound mechanism gains" is **supported** if:
- (S1) EALRMN beats RNN on ≥2 of 3 tasks at m=1024 with ≥0.1 nat margin, multi-seed CI excludes overlap
- (S2) EALRMN matches or beats Transformer_1L at iso-param-count on ≥1 task
- (S3) EALRMN's advantage grows with T (positive scaling slope)

The hypothesis is **falsified** if:
- (F1) RNN matches or beats EALRMN at all tested m, T (≥2/3 tasks)
- (F2) EALRMN-RNN gap does not grow with T (zero or negative slope)
- (F3) Transformer baselines dominate at all scales

## Implementation plan

1. `kernels.cuh` — all forward/backward CUDA kernels
2. `tensors.cuh` — tensor allocation helpers, save/restore
3. `model_ealrmn.cu` — EALRMN model class
4. `model_rnn.cu` — RNN baseline
5. `model_transformer.cu` — Transformer baseline (1L, 2L)
6. `tasks.cu` — needle, hmm, syntheticlm generators
7. `optimizer.cu` — AdamW
8. `main.cu` — training loop, sweep driver, eval, logging
9. `gradcheck.cu` — gradient-check unit tests for every backward

## Build

Single nvcc compile:
```
nvcc -std=c++17 -O3 -arch=sm_89 -lineinfo \
     -lcublas -lcurand \
     research/ealrmn_gpu/*.cu -o research/ealrmn_gpu/ealrmn_gpu
```

## Output format

Each run writes a JSONL line: `{model, task, m, T, seed, step, train_loss, val_loss, val_acc, tok_per_sec, gpu_mem_mb}`.

Sweep aggregator computes mean ± 95% CI across seeds and emits a markdown table.
