# Flash Attention — CHIRON Design Document

## Motivation

Current cuBLAS-tiled attention materializes two O(nH · T²) FP32 scratch
tensors per layer (`scratch_P` and `scratch_dP`) to hold the softmax
probabilities and the backward intermediate.  Memory cost:

| Config                     | scratch_P + dP |
|----------------------------|---------------:|
| T=1024, nH=16              |        128 MB  |
| T=2048, nH=16              |        512 MB  |
| T=4096, nH=16              |       2048 MB  |
| T=8192, nH=16              |       8192 MB  ← **prohibitive** |

Paired with the 2.23 B GPU-only ceiling at T=1024, longer-context
training hits the `scratch_P` allocation limit before it hits the
weights / grads / Adam ceilings.  Flash attention eliminates the
materialization entirely: softmax probabilities are computed block-wise
in shared memory + registers, never written to HBM.

## Targets

Primary:
- **Forward flash attention** for CHIRON's `chiron_attention_shear_bf16_tiled`.
  Input: Q, K, V in BF16.  Output: O in FP32.  Scratch: none (ideally).
- **Backward flash attention** — same pattern, recompute on-the-fly.

Secondary (not in Phase 1):
- Adjoint flash attention with gradient accumulation in BF16.
- Long-context benchmarks at T=4096, T=8192, T=16384 to establish new
  ceiling.

## Algorithm (forward)

Standard flash attention v2 formulation.  Per attention head h:

```
# Q, K, V : [T, dH]
# O       : [T, dH] (init zero)
# ℓ       : [T]     (softmax denom, init zero)
# m       : [T]     (running max, init -inf)

for j in block_indices(K):                 # outer loop over key blocks
    K_j = K[j*BK:(j+1)*BK, :]              # [BK, dH]
    V_j = V[j*BK:(j+1)*BK, :]              # [BK, dH]
    for i in block_indices(Q):             # inner loop over query blocks
        Q_i = Q[i*BQ:(i+1)*BQ, :]          # [BQ, dH]

        # Score tile (never written to HBM)
        S_ij = Q_i @ K_j.T * (1/sqrt(dH))  # [BQ, BK]
        if causal: apply_causal_mask(S_ij, i, j)

        # Online softmax update
        m_new   = max(m[i*BQ:], row_max(S_ij))
        ℓ_new   = exp(m[i*BQ:] - m_new) * ℓ[i*BQ:] + row_sum(exp(S_ij - m_new[:, None]))
        O[i*BQ:] = (exp(m[i*BQ:] - m_new)[:, None]) * O[i*BQ:]
                   + exp(S_ij - m_new[:, None]) @ V_j
        m[i*BQ:] = m_new
        ℓ[i*BQ:] = ℓ_new

# Final normalization
O /= ℓ[:, None]
```

Key invariants:
- `m`, `ℓ` are O(T) scratch (not O(T²) — negligible).
- S_ij and `exp(S_ij - m_new)` live only in shared memory / registers.
- One kernel launch per (head × query block); K/V blocks iterated inside.

## Tile sizing

For RTX 4080 SUPER (Ada, SM 8.9, 164 KB shared memory per SM):
- BQ = 128, BK = 64, dH ≤ 256 — standard flash-attn v2 sizing.
- Register usage per thread: ~64 FP32 registers — fits within 255 limit.
- Shared memory per block: Q_i (BQ*dH=32k FP32 = 128 KB) is too large.
  Use BQ = 64, BK = 64 to fit Q_i + K_j + V_j + S_ij in 48 KB shared.

## Backward

Same pattern, recomputing softmax probs on-the-fly.  Key addition: we
need to recompute the full forward softmax given Q, K, V to produce the
backward intermediates (dQ, dK, dV).  Flash attention v2's backward
keeps `m`, `ℓ` from forward to avoid a second softmax pass.

Phase 1: single-pass backward without the forward-state reuse (recompute
m and ℓ).  Slower by a factor of ~2 but simpler to verify.  Phase 2 adds
the shared m/ℓ and matches v2 performance.

## Implementation plan

### Phase 1 — forward-only, BF16 inputs, FP32 output
- `flash_attention_forward_bf16(Q, K, V, T, nH, dH, dModel, causal, O, T_buf)`
  where T_buf is the per-thread scratch for m, ℓ (2T floats).
- One CUDA kernel, ~200 LOC.
- Parity test vs `flash_attention_cublas_tiled_bf16` (expect |err| < 1e-2
  at BF16 precision).

### Phase 2 — backward
- `flash_attention_backward_bf16(Q, K, V, O, dO, m_fwd, ℓ_fwd, ...)`
  where `m_fwd`, `ℓ_fwd` are the [T]-sized tensors saved from forward.
- Two kernels (dQ separate from dK/dV for better occupancy).
- Parity test vs `flash_attention_backward_cublas_tiled`.

### Phase 3 — trainer integration
- New `--flash-attn` flag in the CHIRON trainer.
- Wire via a new chiron primitive variant
  `chiron_attention_shear_flash_bf16(...)` that calls the flash kernels
  instead of the cuBLAS-tiled path.
- Validate identical training loss trajectory.

### Phase 4 — long-context benchmarks
- T=4096, T=8192, T=16384 at ~200 M params.
- Compare step time vs cuBLAS-tiled (when tiled still fits) and
  against baseline.

## Risk / complexity

- **Numerical precision**: the online softmax update is sensitive to FP16
  underflow on `exp(m - m_new)`.  Use FP32 accumulators for m, ℓ, O.
- **Correctness of backward**: the reverse-mode with shared m/ℓ is
  subtle.  Pair with a small-scale unit test (T=32, dH=32) doing
  analytic diff vs a reference PyTorch-equivalent implementation.
- **Bank conflicts**: Q_i tile in shared memory must be strided to
  avoid 32-way bank conflicts on dH-stride accesses.

## Projected impact

At T=8192 on the 2.23 B config:
- Current: `scratch_P + scratch_dP` = 16 * 8192² * 4 * 2 = **8 GB** — OOM.
- Flash:   scratch_P / dP = 0.  m + ℓ = 16 * 8192 * 4 * 2 = **1 MB**.

This unlocks **T=8192+ context on 2.23 B params** and on all smaller
configs.  For CHIRON specifically, this means the reversible flow's
activation advantage (O(1) in depth) finally extends cleanly to O(1) in
the attention context as well — closing the last activation-memory hole
in the framework.
