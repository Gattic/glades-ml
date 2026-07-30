# CHIRON Exact Incremental Decode Design

Date: 2026-07-30
Status: implementation contract for practical-v1
Scope: causal SCFA CHIRON, native checkpoint geometry, no sliding window

## Claim boundary

This design shows that the current causal CHIRON forward has bounded online
state. It does not claim that the legacy global-DCT SCFA path, dense-attention
fallback, or a sliding-window eviction rule has equivalent state. The first
implementation must reject those modes rather than silently changing model
semantics.

## Geometry

For sequence length `T` and `k` compressed rows, block `b` is

```
begin(b) = floor(b*T/k)
end(b)   = floor((b+1)*T/k)
B_b      = end(b) - begin(b)
```

Practical-v1 fixes `T=2048`; its selected SCFA geometry must satisfy the same
causal-block contract as the checkpoint. The implementation must use these
integer boundaries, not assume equal blocks even though current production
geometries use width 16.

## Operator-by-operator online state

### Embedding and reversible `q/p`

The input state for token `t` is `q=E[token_t]`, `p=0`. Within a token, `p`
is carried through layers exactly as in full forward. Neither tensor requires
cross-token state by itself.

### Causal SCFA compression and delayed lift

For layer input `q_t`, completed block summary `c_b` is

```
c_b = sum(q_t, t in block b) / sqrt(B_b).
```

The lift seen by every token in block `b>0` is

```
q_parallel = c_(b-1) / sqrt(B_(b-1)).
```

Therefore only the completed summaries and the running rows/sum of the current
block are required. A partial block is never exposed early. Completing a block
commits exactly one immutable compressed row.

### Residual causal depthwise convolution

`q_perp_t = q_t - q_parallel_t` and

```
y_perp_t = sum(D[:,i] * q_perp_(t-i), i=0..w).
```

The sufficient state is the last `w` residual rows plus the current row. The
cache must preserve chronological order and zero-prefix behavior.

### Inner compressed causal attention

Compressed output row `z_b` is causal attention over `c_0..c_b`. It becomes
visible only in block `b+1`:

```
y_parallel_t = z_(b-1) / sqrt(B_(b-1)), t in block b.
```

Correctness-first code may recompute the existing full-`k` causal attention
kernel when a block commits and retain row `b`; future compressed rows are
causally masked. The optimized implementation stores per-layer projected
K/V rows and evaluates only the new query row. Both implementations must
produce the same committed `z_b` within the registered numeric tolerance.

### FFN, fuse, WhiSC-D, and ReLN

The reversible FFN, final/per-layer fuse, fixed-checkpoint WhiSC-D coupling,
and ReLN are row-local once the current token's `q/p` are known. They run with
`T=1` during decode. WhiSC-D is cacheable only because serving uses persisted
`whisc_a`; recalibrating it from a decode window would violate causality and is
forbidden.

### Position-dependent state

Current causal SCFA serving has no separate absolute position transform in the
forward. Position still determines the block boundary and convolution history.
QK-Norm gamma is fixed per layer/head and remains model state, not mutable cache
state.

## Cache invariant

After consuming `position` tokens, each layer cache contains:

1. exact committed `q_compr[0..completedBlocks-1]` and their projected K/V rows;
2. the most recently committed `y_compr` row (older rows are never read again);
3. layer-input `q` rows for the uncommitted current block, in order;
4. the most recent `min(position,w+1)` `q_perp` rows, in order;
5. no value derived from a token at index `>= position`.

Global cache metadata stores model geometry, `position`, and readiness. Model
weights and checkpoint tensors are immutable and are never copied into or
mutated by the cache.

## API contract

- `allocate`: bind storage to one exact model/config geometry.
- `reset`: zero mutable storage and return to `position=0` without touching
  model weights.
- `prefill`: consume a non-empty prompt with `prompt_len < T` and return logits
  for its last token.
- `step`: consume exactly one next token, advance position by one, and return
  that token's logits.
- `clone`: deep-copy mutable cache state for correctness-first branching.
- Any request with `position >= T`, dense attention, non-causal SCFA, changed
  geometry/config, or requested sliding eviction returns failure explicitly.

The no-slide limit is semantic, not merely an allocation limit: evicting old
blocks changes compressed attention and delayed-lift state. Sliding remains
unsupported until an explicit equivalence rule is designed and tested.

## Correctness gates

For prompts ending inside a block, on a boundary, and immediately after a
boundary:

- prefill logits match full-forward logits at the same row;
- each cached step matches a fresh full-forward prefix;
- greedy token trajectories match;
- stochastic trajectories and RNG advancement match when sampling consumes the
  same logits/context;
- reset reproduces the original result;
- clone branches independently and initially matches its source;
- WhiSC on/off and QK-Norm on/off are covered;
- checkpoint/model buffers hash identically before and after cache use;
- attempts to exceed `T` or request sliding fail without mutating cache state.

## Performance staging

1. Correctness-first: bounded outer state and existing compressed attention
   kernel at block commits.
2. Fast prefill: one full causal forward with exact per-layer cache capture.
3. Fast boundary commit: projected compressed K/V cache and one-query attention.
4. Shared-prefix branching: immutable prefix pages with copy-on-write tails.

Only stages that pass the correctness gates may be used in generation or
rollout collection.
