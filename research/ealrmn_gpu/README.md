# EALRMN Phase-1 GPU prototype

Implementation of Option C from `research/EALRMN_WRITEUP.md` — production-scale GPU verification of the EALRMN architecture against RNN and multi-head Transformer baselines.

## Layout

| File | Purpose |
|------|---------|
| `DESIGN.md` | Phase-1 design memo (architecture, sweep matrix, falsification criteria) |
| `common.cuh` | Tensor allocation, RNG (splitmix64-seeded xorshift), error checks, activations |
| `kernels.cuh` | cuBLAS GEMM (row-major) wrappers, embedding, softmax+CE, layer-norm |
| `recurrence_kernels.cuh` | Memory update, attention readout, gate sigmoid backward, concat/split |
| `model_ealrmn.cuh` | EALRMN-attmem (Phase-0g architecture) full forward + BPTT |
| `model_rnn.cuh` | Standard tanh RNN baseline with matched param budget |
| `model_transformer.cuh` | Multi-head Transformer (1L, 2L) with causal mask + GELU MLP |
| `tasks.cuh` | needle-in-haystack, HMM hidden-state, syntheticlm generators |
| `optimizer.cuh` | AdamW with decoupled weight decay + global grad clip |
| `main.cu` | Entry point: train, gradcheck modes; per-model train loops; argparse |
| `aggregate.cpp` | JSONL → per-cell mean ± stddev table |
| `md_table.cpp` | JSONL → markdown table emitter (per-T, gap analyses) |
| `build.sh` | Single-command nvcc build |
| `run_sweep.sh` | Sweep driver (smoke / focused / prod_v1 / scale / long / tasks / full) |
| `build_final_doc.sh` | Aggregator + table generation helper |

## Quick start

```bash
# Build (requires CUDA 12.0+, compute capability ≥ sm_70; tuned for sm_89)
./build.sh

# Verify backward passes via numeric gradient check (FP32-noise-aware tolerance)
./ealrmn_gpu --mode=gradcheck --model=ealrmn_attmem --task=needle
./ealrmn_gpu --mode=gradcheck --model=rnn --task=needle
./ealrmn_gpu --mode=gradcheck --model=transformer_1l --task=needle --H=2

# Small training run (~10s)
./ealrmn_gpu --mode=train --model=ealrmn_attmem --task=needle \
    --m=128 --T=256 --steps=300 --batch=8 --seed=0

# Production sweep (~60 min on RTX 4080 SUPER)
./run_sweep.sh prod_v1

# Generate aggregated table (mean ± stddev)
g++ -std=c++17 -O2 aggregate.cpp -o aggregate
./aggregate results/sweep_prod_v1.jsonl
```

## Models

All models share an integer-token input pipeline (vocab V, length T, batch B). All produce a single classification head over `n_classes` taken from the last timestep's representation.

### EALRMN-attmem
Architecture identical to Phase-0g CPU prototype, scaled up:
- Embedding E (V × m)
- Koopman linear recurrence: s_t = K · s_{t-1} + W_in · z_t, K initialized via Gram-Schmidt orthogonal × 0.95 (spectral radius < 1)
- 4-slot gated EMA memory: M_t[j] = (1-λ_j) M_{t-1}[j] + λ_j · g_t[j] · z_t with λ = [0.5, 0.1, 0.01, 0.001]
- Per-slot scalar gate: g_t[j] = σ(W_g[j] · s_t + b_g[j])
- Attention readout at t=T: q = W_q · s_T + b_q; α_j = softmax(q · M[j] / √m); r = Σ α_j · M[j]
- Output: feat = concat(s_T, r); logits = W_out · feat + b_out

### RNN baseline
- Same embedding
- Standard tanh: s_t = tanh(W_h · s_{t-1} + W_in · z_t + b_h)
- Output: logits = W_out · s_T + b_out

### Transformer baseline
- Embedding + sinusoidal positional encoding
- L ∈ {1, 2} stacked layers: pre-norm causal MHA (H=8 default) → MLP (4m intermediate, GELU)
- Final LN + last-token pooling → linear head

## Gradient check details

Tolerance: relative error < 5% OR absolute error < 5e-4 (accommodates FP32 noise floor on small-magnitude gradients).

Config: B=2, T=8, m=8, 6 random indices per parameter tensor, eps = 5e-3, central differences.

Results: all three models pass at 100%.

## Memory bounds

At m=1024 on RTX 4080 SUPER (16 GB):
- EALRMN: T=16384 works at B=1 (~308 MiB total).
- RNN: T=16384 works at B=1.
- Transformer (naive O(T²) attention): T=4096 works at B=1; T=8192 OOMs.

For Transformer at T>4096 a streaming/FlashAttention variant would be needed. Phase-1 takes the OOM as data: the bounded-memory architectures (RNN, EALRMN) trivially handle the long-context regime that dense Transformer cannot.

## Configurations

The sweep driver supports these tags:

| Tag | Description | Wall clock |
|-----|-------------|------------|
| `smoke` | EALRMN+RNN at m=128 T=256, 2 seeds, 200 steps | ~30s |
| `prod_v1` | m=1024, T∈{2048, 4096, 16384}, EALRMN+RNN(+Transformer at T=2048), 2-3 seeds | ~60 min |
| `focused` | m=512, T∈{2048, 4096}, EALRMN+RNN+Transformer, 3 seeds | ~90 min |
| `scale` | m=1024 at T∈{2048, 4096}, EALRMN+RNN, 3 seeds, longer steps | ~3 hr |
| `long` | T=16384 production-scale, EALRMN+RNN, 2 seeds | ~4 hr |
| `tasks` | HMM + syntheticlm at T=2048 m=512 | ~1 hr |
| `full` | All of the above sequentially | ~12 hr |

## Determinism

- Each (model, task, m, T, seed) tuple is fully reproducible across runs.
- HostRng uses splitmix64 hash to ensure distinct seed values yield distinct streams (a naive xorshift bug — seed=0 collapsing to seed=1 — was fixed early in development).
- cuBLAS operations use default deterministic algorithms.
- The `eval_rng` stream is seeded as `splitmix64(seed + 17)` so the eval set is independent of the training data stream but deterministic given the seed.

## Output format

The JSONL log emits one line per eval point:
```json
{"model":"ealrmn_attmem","task":"needle","m":1024,"T":2048,"seed":0,"step":160,
 "train_loss":1.19428,"val_loss":0.07,"val_acc":1.0,"tok_per_sec":38706.1,
 "n_params":3257356,"wall_s":33.86,"tag":"prod_v1"}
```

The `aggregate` and `md_table` tools group by (model, task, m, T, step) and emit per-cell mean ± stddev across seeds.

## Implementation notes

- Single-file headers for each concept — no separate .cu compilation unit. nvcc handles inline kernels and template instantiation in one pass.
- Row-major GEMM wrappers around column-major cuBLAS calls (`gemm_nn`, `gemm_nt`, `gemm_tn`, plus strided-batched variants for attention).
- All kernels are deterministic in the per-block sense; cross-block atomicAdd may introduce slight non-determinism but does not affect convergence comparisons.
- BPTT for both RNN and EALRMN saves all forward states (s_all of size (T+1, B, m)). Gradient checkpointing was not implemented since at B=2 T=16384 m=1024 the saved tensors fit in <1 GB.

## Related

- `research/EALRMN_WRITEUP.md` — Phase-0 final writeup (8-phase CPU sequence)
- `research/EALRMN_DESIGN.md` — original mathematical framework design memo
- `research/EALRMN_PHASE0K_RESULTS.md` — CPU scale-up final phase
- `research/EALRMN_PHASE1_GPU_RESULTS.md` — Phase-1 GPU results (this prototype)
