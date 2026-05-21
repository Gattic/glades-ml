# ORION HVP reduction — staged design

**Date**: 2026-05-17 (during Gate-0 probe wait)
**Scope**: Reduce anchor-step compute from (1+r) F+B passes to (1+r/2) or (1) F+B by eliminating finite-difference HVPs.

---

## Current FD-HVP cost (in ORION v3)

At each anchor step, the trainer computes the reduced Hessian H_∥ ∈ ℝ^{r×r} via finite differences:

```
For k in [0, r):
    θ_pert  ←  θ_anchor + ε · V[:, k]
    Run forward + backward → g_pert
    HVP_k   ←  (g_pert − g_anchor) / ε
    H_∥[:, k] ←  V^T HVP_k
    Restore θ ← θ_anchor
```

Per HVP: 1 extra F+B pass.  Per anchor: r HVPs ⇒ **r extra F+B**.

Total anchor cost: 1 (gradient) + r (HVPs) = **(1+r) F+B** per anchor.

At K=20 r=4: each anchor costs 5 F+B, then 19 free reduced-ODE steps → effective cost (5/20) = **0.25 F+B per logical step** → 4× speedup ceiling.

---

## Three reductions, ordered by EV per implementation effort

### Option A: Drop HVPs entirely (highest EV, lowest cost)

ORION's reduced ODE update uses the projected Hessian H_∥ to refine α. But the **r=1, r=2 case** can use a constant scalar / 2×2 H_∥ that's stable across many anchors. Two observations:

1. The diagonal Adam preconditioner `A_∥_diag` already captures most of the second-order signal in low-dim subspaces.
2. The dominant slow eigenvalue is roughly constant after warmup (the slow manifold's curvature stabilizes).

**Proposal**: refresh H_∥ only every M_h = 100-500 anchors instead of every anchor. Between refreshes, reuse the last H_∥. Refresh cost amortizes over M_h × K = thousands of logical steps.

Cost reduction:
- Per anchor: 1 F+B (vs 5 at K=20 r=4)
- Speedup ceiling at K=20: K/1 = **20×** (vs 4× currently)
- Per-M_h-anchors overhead: 1 refresh × r FD-HVPs = r F+B once every 100 anchors

Net: anchor cost drops to ~1 F+B + tiny refresh amortization → ~5× improvement vs current ORION.

**Implementation**: add `orionHvpRefresh` flag (default 100). Track `O.anchorCount` (already exists). Skip the FD-HVP loop unless `(O.anchorCount % cfg.orionHvpRefresh) == 0`.

**Risk**: stale H_∥ may drift if curvature changes during training (e.g., L-schedule transitions). Mitigation: force refresh on RLG L-change events.

### Option B: Central-difference HVP (same cost, 2× accuracy)

Replace forward-difference with central-difference:

```
HVP_k ≈ (∇L(θ + ε·V[:, k]) − ∇L(θ − ε·V[:, k])) / (2ε)
```

Cost: 2 F+B per HVP (vs 1 in current FD).  Anchor: **(1 + 2r) F+B** → 9 F+B at r=4 → speedup ceiling 20/9 = 2.2×.

**Strictly worse than current**. Only useful if FD-HVP's O(ε) bias is the bottleneck (it's not — we've never seen that as a constraint).

**Verdict: don't pursue.**

### Option C: Pearlmutter R-operator (exact HVP in 1 F+B)

The mathematically optimal HVP. Pearlmutter (1994) shows:

```
Hv = ∇_θ (v^T ∇_θ L) = R{∇L}
```

where R is the R-operator (forward-mode-through-backward).  Implementation: propagate "tangent" variables alongside the normal forward and backward passes.

Cost: 1 F+B per HVP (down from FD's 1 extra F+B per HVP) — **eliminates the +r passes entirely**. Anchor cost becomes 1 F+B for gradient (with tangent slots carrying HVP info too) → speedup ceiling K/1 = K.

#### What the R-operator requires (the scope of work)

For every operation in CHIRON's forward/backward graph, we need a tangent variant.  This means writing a parallel "tangent" kernel for every kernel that consumes a parameter.

CHIRON's parameter-consuming kernels:
- E·x (embedding lookup + matmul) — readout + initial token-embedding (2 places)
- Per-layer Wq·q, Wk·q, Wv·q (attention projections) — 3 per layer × 24 layers
- Per-layer Wo·sV (attention output projection)
- LayerNorm γ, β (24 layers)
- SCFA's inner-shear cuBLAS GEMMs (216/step × 9 inner GEMMs)

**Minimum kernel work** for Pearlmutter:
- A new "JVP" variant of every cuBLAS GEMM (cuBLAS doesn't expose JVP; would need custom)
- A new "JVP" variant of LayerNorm forward + backward
- A new "JVP" variant of SCFA's shear (heavy — SCFA is a custom kernel with intricate FMA chains)
- A new "JVP" variant of the activation functions (GELU/SiLU)
- Tangent storage allocations parallel to activation storage (~equal VRAM as activations)

#### Engineering estimate

Roughly **2-4 weeks of focused work** to write JVP variants for every parameter-consuming kernel.

The benefit (8× → 20× speedup ceiling) is substantial — but the work is substantial. Doesn't pass the iter-236 1-day Gate-0 mandate.

**Verdict**: defer Pearlmutter HVP. Pursue Option A (drop HVPs / amortize refresh) first — it captures 80 % of the benefit at 1 % of the effort.

---

## Recommended sequencing (assuming Gate-0 passes)

1. **Stage now**: this document, Option A's flag wiring (~20 lines in chiron_main.cpp)
2. **Phase 1** (after Gate-0 PASS): combine INT8 V (other staged port) + BF16 anchor (other staged port) + Option A HVP-amortization. ~50 lines total. Run benchmark, target 8-10× ceiling at K=20 r=4.
3. **Phase 2** (if Phase 1 ships): consider Option C Pearlmutter if the 8-10× isn't enough.

## Code change for Option A (HVP amortization)

`Config`:
```cpp
int orionHvpRefresh;  // refresh reduced H_∥ every N anchors (default 100; 0 = every anchor)
```

`OrionState`:
```cpp
bool h_proj_valid;          // whether H_proj contains a valid (recent) refresh
int last_hvp_refresh;       // anchorCount at last refresh
```

`orion_anchor()`:
```cpp
bool need_hvp = (cfg.orionHvpRefresh <= 0)
             || (O.anchorCount - O.last_hvp_refresh >= cfg.orionHvpRefresh)
             || !O.h_proj_valid;

if (need_hvp) {
    // existing FD-HVP loop (r passes)
    for (int k = 0; k < r; ++k) { ... }
    O.last_hvp_refresh = O.anchorCount;
    O.h_proj_valid = true;
}
// else: reuse t->H_proj from last refresh
```

## VRAM impact

Option A: 0 additional VRAM. Pure compute optimization.

Option C: +1 tangent activation storage per layer ≈ equal to forward activations (substantial — would need T-axis BF16 storage like iter 65). At T=16384: ~2-3 GB additional VRAM.

---

## Combined speedup projection (with all three ports)

| variant | per-anchor F+B cost | speedup ceiling at K=20 r=4 | persistent VRAM at 1B (vs naive ORION 13.95 GB) |
|---|---:|---:|---:|
| current ORION v3 | 1 + 4 = 5 | 4× | 13.95 GB |
| + INT8 V | 5 | 4× | 10.47 GB |
| + BF16 anchor | 5 | 4× | 12.20 GB |
| + INT8 V + BF16 anchor | 5 | 4× | 8.72 GB |
| + INT8 V + BF16 anchor + Opt-A HVP-amort | ~1.04 (amortized) | **~19×** | 8.72 GB |
| + Opt-C Pearlmutter (instead of Opt-A) | 1 (exact) | 20× | 8.72 + ~2.5 = 11.22 GB |

**Combined Pareto front**: Option-A path gives 19× speedup at 8.72 GB persistent VRAM — fits comfortably at T=16384 (base 13 GB + 8.72 = 21.72 GB still over but with selective scope reductions can fit).
