# Deferred Paradigm Designs — Closure Record

**Date:** 2026-04-24 (Ralph-loop iter 139)
**Purpose:** Formally close out deferred paradigm designs that are no longer
worth pursuing given the validated FACE + SLC stack.

---

## Decision framework

A deferred design is CLOSED when:
1. Its target axis is covered by an already-validated disrupting paradigm,
   OR
2. Its implementation cost exceeds its expected marginal benefit,
   OR
3. Its mechanism premise is weak given empirical findings.

---

## Closed: Paradigm #10 MPOT (Matrix Product Operator weight decomposition)

**Target axis:** weight compression via tensor-network factoring.
**Closure reason:** Memory-compression axis SATURATED by validated stack:
- FACE: 1984× embedding Adam compression
- MFIO: 2730× attention Adam compression  
- bf16: 2× precision compression
- Total: ~4000× compression enables 1.84B on 16 GB consumer GPU.

MPOT would further compress WEIGHTS (not Adam state), with unclear
additional benefit given the current ceiling is already VRAM-limited
by scratch_P, not weights. Implementation cost: substantial (weights
stored as Matrix Product Operators requires rewriting all forward/
backward kernels). Not worth pursuing.

**Status:** DEFERRED (design-only), do not implement.

---

## Closed: Paradigm #13 TRCD (Token-Routed Conditional Depth)

**Target axis:** per-token dynamic depth via Gumbel-routed gating.
**Projected gain:** 2.96× speedup at d̄=L/3.
**Shipped:** Phases 1-2 GPU primitives, bit-exact parity tests pass.
**Closure reason:** SLC (#38) delivers 1.50-1.68× throughput via
sequence-length curriculum — simpler mechanism, same result axis.
TRCD's additional speedup would require multi-iteration wire-in effort
(routing logic in forward + backward, per-token execution paths,
compute-skip scaffolding in sgd_transformer.cpp equivalents).

Given SLC's validated ~1.65× speedup is already shipped and composable,
TRCD's projected additional 1.8× (= 2.96/1.65) is not worth ~3-5
iterations of engineering work. Phase 1-2 primitives remain available
in the library for future use.

**Status:** DEFERRED (Phase 1-2 shipped, Phase 3 wire-in not planned).

---

## Closed: Paradigm #16 LCP (Lattice Compute Pooling)

**Target axis:** per-token compute pooling via LSH token clustering.
**Shipped:** LSH gather/scatter primitives, roundtrip parity tests pass.
**Closure reason:** Same as TRCD — SLC covers the throughput axis.
LCP's mechanism (cluster similar tokens, compute once per cluster) is
more complex than SLC's T-scaling and hasn't been wired into the
trainer. The "detail correction" path for reconstruction adds further
complexity.

**Status:** DEFERRED (primitives shipped, trainer wire-in not planned).

---

## Closed: #38 Wire chunked-CE into glades_pile_train

**Closure reason:** Chunked cross-entropy is only relevant for V ≥ 65k
vocabularies. Current research uses V = 32k pile-bpe. Not needed for
any pending work.

**Status:** CLOSED (not needed at current vocab size).

---

## Closed: #15 WMMA Stage 2 — custom BF16 attention kernel

**Closure reason:** cuBLAS BF16 tensor-core path (shipped iters 27, 39, 54)
already delivers BF16 attention at acceptable performance. A custom WMMA
kernel would improve marginally at substantial maintenance burden.
Was explicitly deferred at design time.

**Status:** CLOSED (cuBLAS BF16 path is sufficient).

---

## Closed: #121 HUTCH-DIAG Option #37-A test

**Closure reason:** Iter 124 synthetic Gate-0 showed ρ(N=16) = 0.38,
below the 0.6 accept threshold. The proposed rescue (K=1 probe + β=0.9999
EMA) would yield ρ≈0.95 but costs 50-100% throughput overhead. Pearlmutter
HVP implementation requires CHIRON's backward to become double-
differentiable — a massive rewrite.

The research direction is not viable within reasonable engineering effort.
HUTCH-DIAG remains a documented candidate for future research if
double-backward support becomes available.

**Status:** CLOSED (mechanism too noisy, rescue too expensive).

---

## Summary

Six deferred designs closed this iteration. The research program's active
scope narrows to:

**Shipped and validated (disrupting):**
- FACE (#28) — convergence + memory
- SLC (#38) — throughput

**Shipped supporting paradigms (composable):**
- MFIO v2 (#11), WIP (#22), IBGRAD (#19), bf16 stack (#3-5), CHIRON (#1),
  local-window (#6), HRTC (#8), CLPS (#24), GEC (#25), PRX (#20)

**Open for future research:**
- Paradigm #39 on a genuinely unattacked axis
- SLC long-horizon stability fix (iter 138 divergence)
- 3B+ ceiling via gradient checkpointing

The closure consolidates the research program and frees future iteration
effort for novel paradigm exploration rather than finishing legacy work
with diminishing returns.
