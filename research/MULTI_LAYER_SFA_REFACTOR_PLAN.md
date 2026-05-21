# Multi-Layer SFA Refactor Plan

**Status:** Plan (iter 15, 2026-05-15). Implementation deferred to iter 16+.
**Goal:** Enable `--sfa-swap-layers 12,15,18,21` (or similar comma-separated
list) so N SFA layers can be active simultaneously in the same training
run. Required for testing the compounding claim that drives the "magnitudes"
projection in `CELLULAR_SHEAF_ATTENTION_PROGRAM.md` §9.

## Why this is the right next step

Single-layer SFA at L=18 gives −0.60 nat val improvement (Phase 8b,
2026-05-15). The "magnitudes" claim in `PARADIGM_SHIFT_250_DESIGN.md` and
`CELLULAR_SHEAF_ATTENTION_PROGRAM.md` assumes this **compounds** when SFA
is stacked across multiple layers. The compounding claim is untested.

If 4 layers gives ~−2.4 nat, the claim holds and we're at a real magnitude
(perplexity drops 10×). If it saturates at ~−1.0 nat, the claim fails and
the program needs a different attack toward magnitudes.

## Existing single-layer architecture (chiron_main.cpp summary)

- **Config**: `int sfaSwapLayer` (line 448). Default -1 = off; else layer index in [0, L).
- **Parser**: `--sfa-swap-layer N` (line 928).
- **Bookkeeping** (~25 scalar `GpuBuffer` fields, lines 2890-2925):
  - Static (config): `sfa_T, sfa_d_s, sfa_d_h, sfa_r, sfa_W, sfa_n_sinks, sfa_lambda, ...`
  - Edge data: `sfa_edge_src, sfa_edge_tgt, sfa_out_off, sfa_out_e, sfa_in_off, sfa_in_e`
  - Params (trained): `sfa_U, sfa_Sigma, sfa_Pq, sfa_Pv, sfa_Po`
  - Scratches: `sfa_b, sfa_s, sfa_y, sfa_tmp_Ls, sfa_tmp_res, sfa_diag, sfa_Dinv`
  - Backward grads: `sfa_dPo, sfa_dPq, sfa_dPv, sfa_dU, sfa_dSigma`
  - Adam state (m, v): `sfa_*_m, sfa_*_v` for each of the 5 trained params
- **Allocation** (lines 2585-2820): one block per swap layer; ~200 lines.
- **Dispatch**: fwd at line 5725, bwd at line 6929 — both check
  `sfa_active && l == sfa_swap_layer` for the current layer index `l`.
- **Adam update** (line 7800+): per-param updates using `sfa_lr`.
- **Defect-stat** (line 9423): post-training kernel call.

## Multi-layer design

### What goes vector, what stays scalar

The cellular sheaf substrate has TWO classes of state:

**A. Static graph data (same across layers)**:
- T, d_s, d_h, r, W, n_sinks (already scalar)
- Edge set: `sfa_edge_src, sfa_edge_tgt, sfa_out_off, sfa_out_e, sfa_in_off, sfa_in_e`

These depend only on (T, W, n_sinks) and are **shared across all swap layers**.
Keep them scalar.

**B. Per-layer trained params + scratches**:
- Trained: U, Σ, P_q, P_v, P_o
- Adam state: 2 buffers per trained param × 5 trained params = 10 buffers per layer
- Grads: 5 buffers per layer
- Forward scratches: b, s, y, tmp_Ls, tmp_res, diag, Dinv (7 buffers per layer)

These must be **per-layer**. Convert each to `std::vector<GpuBuffer<T>>` indexed
by `idx_within_swap_layers` (NOT by absolute layer index l).

### New Config + parser

```cpp
// chiron_main.cpp Config struct, alongside existing sfaSwapLayer:
std::vector<int> sfaSwapLayers;   // 2026-05-15: multi-layer extension.
                                   // If empty + sfaSwapLayer >= 0, use {sfaSwapLayer}.
                                   // Else use this vector directly.

// Parser:
else if (streq(a, "--sfa-swap-layers") && i + 1 < argc)
{
    cfg.sfaSwapLayers.clear();
    const char* csv = argv[++i];
    // Tokenize on ',' and atoi each.
    const char* p = csv;
    while (*p)
    {
        while (*p == ',' || *p == ' ') ++p;
        if (!*p) break;
        char* end;
        long v = std::strtol(p, &end, 10);
        if (end > p && v >= 0) cfg.sfaSwapLayers.push_back((int)v);
        p = (*end == ',') ? end + 1 : end;
    }
}
```

After arg parse:
```cpp
// Normalize: if sfaSwapLayers is empty but sfaSwapLayer set, populate it.
if (cfg.sfaSwapLayers.empty() && cfg.sfaSwapLayer >= 0)
    cfg.sfaSwapLayers.push_back(cfg.sfaSwapLayer);
// Conversely if sfaSwapLayers nonempty, set sfaSwapLayer to first entry
// for back-compat with code that reads it.
if (!cfg.sfaSwapLayers.empty()) cfg.sfaSwapLayer = cfg.sfaSwapLayers[0];
```

### ChironParams field changes

```cpp
// Currently scalar:
glades::gpu::GpuBuffer<float> sfa_U;
// Becomes:
std::vector<glades::gpu::GpuBuffer<float>> sfa_U;

// One vector per per-layer buffer field. Edge buffers stay scalar.
```

To map current-layer `l` to vector index: build a lookup at allocate time:

```cpp
// 0-indexed position in sfaSwapLayers if l is a swap layer; -1 otherwise.
std::vector<int> sfa_swap_idx_for_layer;  // size L, default -1

// In allocate():
sfa_swap_idx_for_layer.assign(L, -1);
for (int i = 0; i < (int)cfg.sfaSwapLayers.size(); ++i)
{
    int ll = cfg.sfaSwapLayers[i];
    if (ll < 0 || ll >= L) { /* error */ }
    sfa_swap_idx_for_layer[ll] = i;
}
```

### Allocation loop

```cpp
const int N_swap = (int)cfg.sfaSwapLayers.size();
sfa_U.resize(N_swap);
sfa_Sigma.resize(N_swap);
... // all per-layer buffer vectors

for (int idx = 0; idx < N_swap; ++idx)
{
    sfa_U[idx].allocate(T * d_s * r);
    sfa_Sigma[idx].allocate(E * r);
    // ... allocate all per-layer buffers
    // Edge data is allocated once outside this loop.
}

// Edge allocation: only once.
sfa_edge_src.allocate(E);
sfa_edge_tgt.allocate(E);
// ...
```

### Init (Σ init, U init, etc.)

Same code as current, just inside the per-layer loop. Each layer's U and Σ
get independent random seeds (use seed + idx so reproducible per layer).

### Forward dispatch (line 5725)

Currently:
```cpp
const bool sfa_dispatch = W.sfa_active && (l == W.sfa_swap_layer);
if (sfa_dispatch) { ...use W.sfa_U.data() etc... }
```

Becomes:
```cpp
int sfa_idx = W.sfa_swap_idx_for_layer[l];
const bool sfa_dispatch = (sfa_idx >= 0);
if (sfa_dispatch)
{
    auto& U     = W.sfa_U[sfa_idx];
    auto& Sigma = W.sfa_Sigma[sfa_idx];
    auto& P_q   = W.sfa_Pq[sfa_idx];
    // ... use these instead of W.sfa_U.data() etc.
}
```

### Backward dispatch (line 6929)

Same pattern — replace single-buffer references with indexed-vector
references.

### Adam update (line 7800+)

Currently one update block per trained param. Becomes a loop:

```cpp
for (int idx = 0; idx < N_swap; ++idx)
{
    apply_adam_int8(W.sfa_U[idx], W.sfa_dU[idx], W.sfa_U_m[idx], W.sfa_U_v[idx], ...);
    apply_adam_int8(W.sfa_Sigma[idx], W.sfa_dSigma[idx], ...);
    // ... for each trained param
}
```

### Defect-stat (line 9423)

Loop over swap layers, run defect kernel on each, log per-layer per-position
bucket means:

```cpp
for (int idx = 0; idx < N_swap; ++idx)
{
    int l = cfg.sfaSwapLayers[idx];
    // call sfa_defect_step1_fp32 with W.sfa_Sigma[idx] ...
    // call sfa_defect_frame_step1_fp32 with W.sfa_U[idx] ...
    // log with layer=l label
}
```

## Estimated diff size

- Config + parser: ~40 lines
- ChironParams field type changes: ~30 lines (vector<> conversions)
- Allocation loop: ~150 lines (existing block wrapped in for loop)
- Init code: ~80 lines (loop wrapper around existing init)
- Forward dispatch: ~40 lines (re-bind references inside if)
- Backward dispatch: ~40 lines (same)
- Adam update: ~30 lines (for loop around 5 update blocks)
- Defect-stat: ~30 lines (loop around existing single-layer call)

**Total: ~400-500 lines of changes** to chiron_main.cpp.

Risk areas:
1. **VRAM budget**: 4 layers × 200 MB SFA state = 800 MB. At T=16384 on
   16 GB VRAM that's manageable (current usage is 14.16/15.56 GB single
   layer, so 4 layers might exceed budget). Mitigation: smaller `d_s` or
   `r` per layer, or fewer layers per run.
2. **CSR cost**: edge structure is shared, so no multiplicative cost there.
3. **Performance**: each extra SFA layer adds ~7% per-step wall time
   (per Phase 8b: 16,575 tok/s with 1 SFA vs ~16,665 baseline). 4 layers ≈ 28%
   slower per step. Acceptable for the magnitudes test.

## First experiment after refactor lands

1. **4-layer SFA at L = {12, 15, 18, 21}**, 2000 steps each, --sfa-train, lr=0.
2. Compare val NLL to Phase 8b's single-layer (-0.60 nat mean / -1.50 peak).
3. If 4-layer gives ≥ -1.8 nat mean: **STRONG SUPPORT for compounding** — the magnitudes claim is on track.
4. If 4-layer gives ~-0.8 to -1.2 nat: PARTIAL compounding (sub-linear); revisit which layers help most.
5. If 4-layer gives ≤ -0.7 nat: **COMPOUNDING FAILS**; pivot to non-stacking approaches.

Total iter-16 wall: ~33 min (one 4-layer training run).

## Implementation order (iter 16+)

1. **iter 16**: Config + parser + ChironParams field changes (compile but no runtime change in single-layer case).
2. **iter 17**: Allocation + init loop (runtime tests single-layer == prior behavior).
3. **iter 18**: Forward + backward dispatch (runtime tests parity).
4. **iter 19**: Adam loop + defect-stat loop.
5. **iter 20**: First 4-layer experiment.

Tight but feasible. Each iter is one logical chunk with compile/test verification.

## Alternative: shared-params multi-layer

A simpler refactor would SHARE params across swap layers: same U, Σ, P_q,
P_v, P_o applied at multiple layers. This is the TSR variant of paradigm
#253 (tied sheaf renormalization) — it tests whether stacking the SAME
sheaf at multiple depths compounds. Smaller refactor (~100 lines), but
also tests a weaker hypothesis (the sheaf-as-renormalization claim, not
the per-layer-specialization claim).

Choice: do the FULL refactor (independent params per layer). It's the
right thing for testing magnitudes. Shared-params can come later if
needed.

## Files

- `glades-trainer/trainer/chiron_main.cpp` — the refactor target.
- `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_sfa.{h,cu}` — no
  changes; kernels are stateless and operate on whatever buffer is passed in.

## Honest note on iter-15 progress

This plan is the iter-15 deliverable. The 2000-step Probe O run in
parallel may sharpen the U-frame defect signal, but even a clean DSA
result doesn't address the compounding question. The multi-layer
refactor is required for the magnitudes claim either way.
