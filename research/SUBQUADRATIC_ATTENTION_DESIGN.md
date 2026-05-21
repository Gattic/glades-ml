# Sub-Quadratic Attention within CHIRON — Framework-Design Exercise

## Problem statement

CHIRON currently inherits standard softmax attention's O(T² · dH) compute
cost per block.  At the 2.23 B ceiling with T=1024 this is 9 GFLOP per
layer per forward — roughly 1 ms/layer on cuBLAS-tiled BF16.  At T=16384
the same kernel is 256× more work (2304 GFLOP/layer), impractical even
with flash attention's memory savings.

**The frontier we have not yet crossed: attention *compute* complexity.**
Activations (via CHIRON), weights (BF16), grads (BF16), and optimizer
state (int8) are all compressed on the memory axis; attention speed is
accelerated by TC-tiled BF16 kernels but remains fundamentally O(T²).

## Candidate formulations

Required constraint: whatever attention we use, the shear `(q, p) → (q,
p + Y(q))` must remain a symplectic shear so CHIRON's inverse-based
activation-recovery stays valid.  This is satisfied by *any* deterministic
function `Y(q) = f(q, Wq, Wk, Wv, Wo)` — the sub-quadratic attention
choice changes `f`'s internal complexity but not its algebraic form as
a shear.

### Candidate A — Local-window attention

Restrict each token's attention to a window of ±W tokens.  Complexity:
O(T · W · dH) per layer.

*Primitive objects:* a per-token sliding window, a fixed window size
W ∈ {64, 128, 256}.
*Evolution:* attention score `S[t, j] = Q[t] · K[max(0, t-W) + j] /
√dH` for `j ∈ [0, W)`.  Softmax is local.  Output is a weighted
sum of V over the window.
*Expected properties:* perfect for local-context tasks (language);
fails cross-document or long-range copy.  Compute is T · W · dH which
at W=128 is 128× reduction over full attention at T=16384.

### Candidate B — Sparse / strided attention (BigBird-style)

Fixed local window + global "anchor" tokens + random sparse links.
Complexity: O(T · (W + G + R) · dH) where G = anchors, R = random
connections per token.

*Primitive objects:* a sparse pattern mask `M ∈ {0, 1}^{T×T}` with
≤ (W + G + R) · T ones; stays fixed across layers and batches.
*Evolution:* compute scores only at `M[t, j] = 1`; pack into a dense
sub-tensor; softmax + output same as local attention.
*Expected properties:* provably approximates dense attention under mild
regularity conditions (BigBird proof).  Needs careful memory layout
for the sparse pattern to avoid irregular memory access.

### Candidate C — Random-feature / linear attention

Replace `softmax(QK^T) V` with `φ(Q) (φ(K)^T V)` where `φ : R^dH → R^r`
is a random Gaussian feature map that satisfies `φ(q) · φ(k) ≈
exp(q · k / √dH)`.  Compute: `K' = φ(K)^T V ∈ R^{r × dV}` (one matmul,
O(r · T · dV)), then `O[t] = φ(Q[t]) · K' / Z[t]` where `Z` is a
normalizer.  Total O(T · r · (dH + dV)) ≈ O(T · dH²) at r ≈ dH.

*Primitive objects:* a fixed random projection matrix `W_φ ∈ R^{r×dH}`
sampled once at init (or learned).
*Evolution:* Q' = φ(q · Wq), K' = φ(q · Wk); K_accum = K'^T · V; out
= Q' · K_accum.
*Expected properties:* approximates full softmax attention only in
expectation (variance scales with 1/r).  Preserves the causal-mask
structure only if `φ` is strictly positive and we track the
normalizer token-by-token.

## Framework selection rationale

For the **immediate next paradigm shift** we want:
- preserves exact-softmax attention semantics (no approximation),
- easiest integration with CHIRON's existing shear + BF16 tensor-core
  kernels,
- biggest win at long-context training (T ≥ 4096),
- least risk to training convergence at the ceiling we've already shown
  works (2.23 B FP32-equivalent softmax).

**Candidate A (local-window)** wins on all four fronts for the first
iteration.  It changes attention's algorithmic complexity from O(T²)
to O(T · W) without touching the shear's algebraic form — the flash-
attention kernel already sort-of supports it (causal mask can be
extended to a `|i-j| < W` mask cheaply).

Candidate B (sparse) is a natural Phase 2 — adds global anchors once
the local kernel is stable, recovers most of the expressivity
improvement at slightly higher complexity.

Candidate C (linear) is the most novel but highest risk — BF16 softmax
approximation fidelity is already a concern, adding Gaussian-feature
variance on top compounds it.

## Selected formulation — local-window softmax attention

### State space

For each head h, token t, and window offset j ∈ [-W, +W]:
- `S[h, t, j] = (Q[h, t] · K[h, t+j]) / √dH` for t+j ∈ [0, T)
- `P[h, t, j] = exp(S[h, t, j] - max_j(S[h, t, :])) / Σ_j exp(...)`
- `O[h, t] = Σ_j P[h, t, j] · V[h, t+j]`

For causal training: restrict j ∈ [-W, 0] (no attention to the future).

### Compute budget

Total work per layer: `nH · T · W · dH` (vs `nH · T² · dH` for full attn).
At T=16384, W=256, nH=14, dH=320:
  Full:   14 · 16384² · 320 · 2 ≈ 2.4 PFLOP (per forward per layer)
  Local:  14 · 16384 ·  256 · 320 · 2 ≈ 37 GFLOP   — **65,000× reduction**

Per-step at T=16384, L=12: saves ~29 PFLOP → at 25 TFLOP/s BF16 TC, that's
~1200 seconds per step saved.  Unlocks genuine long-context training.

### Memory budget

No materialized O(nH · T²) scratch (same advantage as flash attention).
Plus no dense P: the local kernel computes `W`-wide strips only, which
fit in shared memory + registers per block.

## Implementation roadmap

### Phase 1 — kernel + parity test

1. New CUDA kernel `chiron_local_attention_bf16(Q, K, V, T, nH, dH,
   dModel, W, causal, O)` — block per (head, query row), inner loop
   over W key positions.
2. Uses shared memory for the W-wide slice of K, V, and a running
   online-softmax.
3. Parity test against cuBLAS-tiled BF16 with a local mask applied
   to S (same math, different implementation).

### Phase 2 — trainer wire-in

4. `chiron_attention_shear_local_bf16` — wraps the kernel with the
   FP32 Q/K/V projections + Wo output projection (unchanged from
   existing shear).
5. `--local-attn W` flag on the trainer.  When set, attention uses
   the window-W kernel; W=0 disables (falls back to full attention).

### Phase 3 — backward

6. Local-attn backward: same local pattern, but `dQ`, `dK`, `dV` are
   computed block-wise.  Online softmax state (m, ℓ) saved at
   forward time for the backward's P recompute (flash-v2 style).

### Phase 4 — long-context benchmarks

7. Targets: T=8192, T=16384, T=32768 at the 2.23 B config (m=2368,
   L=48) with W ∈ {128, 256, 512}.  Verify throughput scales
   linearly with T (not quadratically) and verify training
   convergence on a few hundred steps.

## Expected ceiling impact

Memory: unchanged (CHIRON + BF16 stack already O(1) on all axes except
weights — and local attn doesn't touch weights).

Speed at long context:
- T=4096:  ~4× faster attention  (W=256: 4× reduction)
- T=16384: ~64× faster attention (W=256: 64× reduction)
- T=65536: ~256× faster attention

Combined with CHIRON + BF16 weights + int8 Adam, this unlocks
*megatoken* context training on a 16 GB consumer card.  The 2.23 B
param budget at 2.23 B @ T=65536 (effective batch 65k tokens/step)
becomes compute-feasible for the first time.

## Risks & open questions

- **Quality vs full attention**: literature (Longformer, BigBird)
  shows local-only loses 1-3% accuracy on held-out val; need to
  verify at our scale.
- **Boundary handling**: tokens near t=0 and t=T-1 have shorter
  windows — the softmax normalizer needs to be computed per-token,
  not from a fixed W.
- **Interaction with CHIRON reversibility**: the shear is still
  exactly invertible (the attention output `Y(q)` changed but its
  input is still deterministic in q), so CHIRON's O(1) activation
  memory continues to hold.
- **Gradient accuracy at local-only**: unclear whether the reduced
  information flow slows optimizer convergence.  Mitigation: Phase 2
  adds sparse global anchors.
