# Hierarchical Reversible Token Compression (HRTC)

Proposed paradigm shift #8 for the CHIRON training program.

## Problem statement

The 7 paradigm shifts shipped so far compress the **model** dimension
(CHIRON activation memory via reversibility; Stiefel weight factorization;
BF16/int8 for weights/grads/optimizer) or accelerate the **attention**
dimension (TC-tiled kernel, local-window sub-quadratic compute).

One axis remains untouched: the **sequence length T itself**.

For a model with `dModel` hidden state running at `T` tokens, every
CHIRON layer still stores a `[T × m]` pair (q, p).  At the current 2.23 B
config (T=1024, m=2368, L=48):

    Activation memory per layer = 2 · T · m · 4 bytes = 19.4 MB
    Total reversible activation = O(1) in depth but O(T · m) per layer

At the 2.23 B ceiling, activations are a small fraction of VRAM.  But at
**long context** (T = 8k, 16k, 32k) activations grow linearly and become
the binding constraint.  Local-window attention (shift #6) cuts compute
to O(T·W) but does not shrink activations; the q, p tensors still have T
rows.

The research frontier we have not yet crossed: **sequence-axis
compression with CHIRON-style exact reversibility**.

## Core idea

Partition the sequence into contiguous blocks of `k` tokens.  Replace
each block with a single "super-token" via a deterministic, invertible
pooling operator `Φ_k : R^{k×m} → R^{m}`, then run the transformer stack
on `T/k` super-tokens.  At the output, invert the pooling to recover
per-token predictions.

```
   T tokens (q, p)            T/k super-tokens (q', p')         T predictions
   ─────────────▶   [Φ_k]  ─────────────▶   [L layers]  ─────────────▶  [Φ_k^{-1}]  ────▶
   per-layer O(T·m)                         per-layer O((T/k)·m)
```

Because Φ_k is invertible, the mapping composes with CHIRON's reversible
flow: per-block `(q_block, p_block)` is recoverable from each
super-token `(q'_i, p'_i)` via the per-block inverse.  Activation memory
in the middle of the stack drops by factor `k` and attention compute
drops by factor `k²`.

## Candidate formulations

### Candidate A — Wavelet-style reversible pooling

`Φ_k` is a `k×k` orthogonal transform (Haar / DCT / learned) applied per
block on each of the `m` feature dimensions.  The first output component
is the "super-token" (a weighted sum of the `k` tokens); the remaining
`k−1` are stored as compact residuals.

**Invertibility**: exact, by applying the inverse orthogonal transform.
**Residuals storage**: `(k−1)·m` scalars per block; same byte count as
  the un-compressed block, just rearranged.
**Savings**: attention and compute operate on the lead component only
  (`T/k` super-tokens).  Residuals stay idle in the middle of the stack
  and are only re-mixed at the tail.

Relation to existing work: DWT / Mallat multiresolution decomposition.
Novel here is the *reversible composition with CHIRON shears*.

### Candidate B — Attention-pool super-tokens with explicit residual stream

`Φ_k` is a learned per-block attention pool (k-token self-attention with
a single learned query vector), producing one super-token per block.  A
residual stream carries per-token corrections.  Invertibility via the
residual: token `t` of block `b` is recovered as `supertoken_b ⊕
residual_t`.

**Invertibility**: exact as long as the residual is retained.
**Residuals storage**: `T·m` scalars — *same as the original q, p*.
  The win is ONLY in attention compute (pool-based attention sees
  `T/k` super-tokens, O((T/k)²) work), NOT in activation memory.
**Savings**: attention compute drops ~k²×.  Activations unchanged.

This is essentially long-context-attention-with-landmark-tokens but made
reversible.  Weaker than Candidate A on memory.

### Candidate C — Causal-coherent coarse stream

Maintain *two parallel streams*: a fine stream at resolution T and a
coarse stream at resolution T/k.  Only the coarse stream is processed
through the deep middle of the stack; the fine stream is processed
through a shallow "bypass" path.  At output, the coarse stream is
up-sampled and combined with the fine-stream result via a reversible
shear.

**Invertibility**: exact via reversible shear at the merge point.
**Savings**: depth × k factor on activation memory for the deep path;
  compute similarly scaled.  Fine path cost is O(T·m) but over only a
  few shallow layers.
**Caveat**: the coarse stream cannot see fine-grained token structure in
  the middle layers, which may hurt expressivity for tasks requiring
  precise token interactions.

Relation: closest to recent "hierarchical transformer" work (HRNet /
Swin-transformer families) but applied at the sequence axis with
reversible flow.

## Framework selection rationale

**Select Candidate A (wavelet-style reversible pooling)** as the
near-term ship.  Reasons:

1. **Biggest memory win** — activation memory drops by factor `k` across
   all middle layers (candidate B has no memory win; candidate C has the
   win but only on the deep path).
2. **Deterministic and side-effect-free** — Φ_k is a fixed orthogonal
   transform with closed-form inverse.  No learned parameters, no
   convergence questions.  Orthogonal transforms are O(k²·m) per block
   and k² is small (k=4: 16 ops per block element).
3. **Composes cleanly with CHIRON, Stiefel, BF16, local-attn** — the
   super-token sequence is still a standard `(q', p')` pair; all existing
   primitives apply unchanged.  Residual storage can itself be BF16 or
   Stiefel-factored.
4. **Clear failure-mode story** — at k=1 it reduces to identity;
   expressivity is controlled by `k`.  Expected regime: k=4 for aggressive
   context extension, k=2 for conservative.

## Quantitative expectations

At 2.23 B config (m=2368, L=48) with `k=4`:

|                               | T=1024 | T=8192 | T=32768 |
|-------------------------------|-------:|-------:|--------:|
| Activations / layer (dense)   | 19.4 MB| 156 MB |   622 MB|
| Activations / layer (HRTC k=4)|  4.9 MB|  39 MB |   156 MB|
| Per-layer compression         |    4×  |   4×   |     4×  |
| Attention compute (full)      | O(T²)  | O(T²)  |  O(T²)  |
| Attention compute (HRTC)      | O((T/k)²)| O((T/k)²) | O((T/k)²)|
| Attention speedup             |   16×  |   16×  |     16× |

At T=16384 (the configuration that motivated local-window attention):
- Dense activation memory: 312 MB per layer × 48 = **15 GB** (exceeds
  total VRAM)
- Local-window ceiling: compute-reduced, activation unchanged
- **HRTC k=4**: 78 MB per layer × 48 = **3.7 GB** (fits comfortably)

HRTC at k=4 turns T=16384 training from **OOM** into a comfortable run
on the existing 2.23 B config.  Combined with Stiefel (shift #7) at
ρ=0.25, projected ceiling reaches **8 B params @ T=16384** on 16 GB —
a qualitative jump in the "long-context training on consumer GPU"
capability.

## Theoretical analysis — well-posedness

Let `W ∈ R^{k×k}` be an orthogonal matrix (e.g., Haar).  Define
`Φ_k : R^{k×m} → R^{k×m}` by applying `W` across the k-token axis
independently for each feature dimension:
`Φ_k(X)[i, j] = Σ_t W[i, t] · X[t, j]`.

The first output row `Φ_k(X)[0, :]` is the super-token; rows 1..k−1
are residuals.  Inverse: `Φ_k^{-1} = Φ_k^T` (because W is orthogonal).

**Key property**: `Φ_k` is a **bit-exact** operator at FP32 (for Haar:
integer multiplies scaled by 1/√k; for DCT: orthogonal floats with
well-conditioned inverses).  Bit-exact round-trip is critical for
CHIRON reversibility.

**Composition with CHIRON shear**: the shear `(q, p) → (q, p + Y(q))`
is a function of `q` only.  After HRTC:
- `q'_i = Φ_k(q[k·i:k·(i+1), :])[0, :]`
- `p'_i = Φ_k(p[k·i:k·(i+1), :])[0, :]`
- The inner shear computes `p' += Y(q')`, which depends only on `q'`.
- Inverting via `Φ_k^{-1}` recovers per-token `(q, p)` exactly.

So CHIRON's inverse-walk activation recovery composes with HRTC
unchanged.

## Implementation path (minimal prototype)

1. **New primitive**: `hrtc_pool_k4(X, block_size=k)` and
   `hrtc_unpool_k4(X_pooled, residuals)`.  Single fused CUDA kernel, ~30
   LOC, bit-exact Haar transform.
2. **Trainer flag**: `--hrtc-k K` in `chiron_main.cpp`.  When set:
     - After embedding, apply `hrtc_pool_k` to both q and p.  Store
       residuals in a scratch buffer.
     - Run all L CHIRON shears on the pooled `(q', p')` of length T/k.
     - Before the output head, apply `hrtc_unpool_k` using the retained
       residuals.
3. **Parity test**: at `k=1`, HRTC must be a no-op — loss trajectory must
   match the dense baseline exactly.  At `k=2`, forward + backward +
   inverse-walk round-trip must recover `(q, p)` bit-exactly.

Estimated code cost: **~200 LOC** total (1 kernel pair + 1 trainer flag
+ 2 parity tests).  Smaller than Stiefel because no Adam changes, no
retraction, no tangent-space projection.

## Failure modes / risks

1. **Per-block information loss if shears mix across super-tokens
   non-linearly**.  CHIRON's shear `p += Y(q)` operates per-token
   (feature-axis); there is no cross-token mixing inside the shear
   itself — all cross-token mixing happens in attention.  Since attention
   is the ONLY cross-token op and HRTC preserves exact per-token recovery
   via residuals, this is not a concern.
2. **Attention quality**: the pooled attention operates over super-tokens
   that are linear combinations of k underlying tokens.  For causal
   attention, super-token `i` sees super-tokens `0..i`, which corresponds
   to original tokens `0..k·(i+1)−1` — causality preserved.
3. **Convergence**: Haar is a known-good orthogonal transform; the model
   effectively sees a coarser sequence for most of the stack.
   Empirical: expect slight loss penalty (5–10%) at a given step count,
   amortized over 16× attention speedup.
4. **Residual-storage overhead**: per-block residuals are `(k−1)·m`
   scalars — same byte count as the original block.  So residuals are
   *stored*, just not *computed through* in the middle layers.  For
   activation memory this still wins: residuals can be held in
   BF16 scratch outside the reversible flow.

## Relation to prior work

- **Wavelet pre-processing** (signal processing, old) — applies wavelets
  to inputs once; our proposal applies per-block within the network and
  inverts at the tail.
- **Reformer / Performer / Linformer** — reduce attention to linear but
  maintain T-length activations.
- **Hourglass / Funnel Transformer** — downsamples mid-stack but not
  reversibly.  HRTC is the first reversibility-preserving variant.
- **Landmark / SuperToken transformers** — add few learned super-tokens
  but keep the full T sequence; HRTC replaces the full sequence.

The novel contribution is **exact reversibility** of the pooling, which
preserves CHIRON's O(1)-in-depth activation memory guarantee at the
compressed T/k resolution.

## Why this is a paradigm shift

Each of the prior 7 shifts attacked a *fixed* axis of memory or
compute.  HRTC expands the **model's reach** on the sequence axis: at
fixed VRAM, achievable context length scales by factor `k` (at k=4,
from T=16384 OOM to T=16384 comfortable).  Combined with local-window
(shift #6), the two jointly make T=65536 training plausible on 16 GB
for a multi-billion-parameter CHIRON model.

This would be the **first demonstration of multi-billion-parameter
training at 64k context on a single consumer GPU** — a capability
distinctly absent from the current open-source ecosystem.
