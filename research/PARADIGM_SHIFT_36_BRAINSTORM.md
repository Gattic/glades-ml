# Paradigm Shift #36 — Brainstorm of Unattacked Axes

**Date:** 2026-04-23 (post-FACE 1.4B publication, Ralph-loop iter 120).
**Purpose:** Scan genuinely unattacked axes for the next paradigm shift.
**Guiding question:** what gives BOTH magnitudes less memory AND magnitudes faster?

---

## Attacked-axis census (updated)

| # | Category | Shipped/designed shifts |
|---|----------|------------------------|
| 1 | Activation memory | CHIRON (#1) · HRTC (#8) |
| 2 | Attention compute | TC-tiled (#2) · Local-window (#6) |
| 3 | Optimizer state memory | int8 Adam (#3) · BF16 grads (#4) · SR BF16 weights (#5) · MFIO (#11) · IBGRAD (#19) · WIP (#22) |
| 4 | Optimizer state convergence | **FACE (#28)** — DISRUPTING |
| 5 | Weight shape | Stiefel×Σ (#7) · MPOT (#10) |
| 6 | Gradient shape | OVFG (#9) · GEC (#25) |
| 7 | Per-token depth | TRCD (#13) |
| 8 | Per-token compute | LCP (#16) |
| 9 | Cross-step forward | ATC-Δ (#26) |
| 10 | FFN intermediate forward | CSP (#27) |
| 11 | FFN backward sparsity | **SPAREC (#35)** — design shipped |
| 12 | Backward alternative | DFA (#12) |
| 13 | Per-param precision | PRX (#20) · PFE (#21) |
| 14 | Cross-layer sharing | CLPS (#24) |
| 15 | Implicit equilibrium | IED (#14) |
| 16 | Curriculum | EDT (#23) — deferred |
| 17 | Layer surrogate | SGS (#18) |
| 18 | Selective updates | GFIB (#17) |

## Genuinely unattacked axes (Ralph-loop iter 120)

### Axis A — LOSS-FUNCTION COMPUTE / V-DIM SOFTMAX

CE loss is ~20% of step compute at V=32k; grows to ~50% at V=128k or character-level
V≥256k.  Sampled-softmax and hierarchical-softmax are existing; what's NEW is:

**Candidate A1 (TAIL-CE)**: split logits into TOP-K (full precision) + TAIL
(uniform-proxy).  ∇-target loss bias-corrected via importance sampling.  Memory:
O(T·K) instead of O(T·V).  Speedup: O(V/K) forward + backward CE.  Mechanism:
Zipfian-tail distribution of softmax mass (related to FACE's premise).

**Candidate A2 (SPECTRAL-CE)**: decompose V-dim logits via SVD of the unembed-matrix,
compute CE on top-r singular components only.  Linear approx to softmax in span.

### Axis B — PER-LAYER ADAPTIVE OPTIMIZATION

Current LR/β/wd is one value across all L layers.  Research shows: early layers need
slow decay, late layers need aggressive updates.

**Candidate B1 (LAYERWISE-ADAPT)**: per-layer β_2 scheduled from gradient variance;
per-layer LR from gradient norm quantile.  Adam state unchanged but adaptive recipe.

**Candidate B2 (MIXED-OPT)**: some layers use Adafactor (memory), others use Adam
(convergence), decided by layer's gradient statistics.  Compositional.

### Axis C — HESSIAN-INFORMED LOW-OVERHEAD PRECONDITIONERS

Adam's v_t is a diagonal approx to Hessian.  True diagonal Hessian via Hutchinson:
single extra backward pass with Rademacher probe every K steps.

**Candidate C1 (HUTCH-DIAG)**: maintain Hutchinson diagonal Hessian estimate, use
it in place of Adam v_t.  Memory: identical to Adam.  Convergence: second-order
information adds 0.2-0.5 nat.

**Candidate C2 (K-FAC-LITE)**: Kronecker-factored approx of Hessian blocks.  v_t
replaced by two small factors per weight matrix.  Memory reduction + second-order.

### Axis D — ATTENTION K,V CACHE ZIPFIAN COMPRESSION (FACE-for-attention)

FACE succeeds on EMBEDDINGS because V-dim rows have Zipfian access frequency.
Analogue question: do K,V keys at attention have Zipfian visit frequency across
queries?  If the same K-slot is "popular" (attended-to by many Q-slots), its
corresponding gradient is dense.  Rare K-slots have sparse gradients.

**Candidate D1 (KV-FACE)**: apply FACE-style row-EMA + column-freq-debias to the
K, V projections of attention.  Compresses their Adam state the same way FACE
compresses embedding state.  Mechanism: attention-popularity is Zipfian-like.

### Axis E — CROSS-STEP SOFTMAX DELTA CACHE (attention-specific ATC-Δ)

Softmax at step t+1 differs from step t by a small delta under small LR.  Exploit
via sparse-delta update to cached attention probabilities.

**Candidate E1 (ATTN-DELTA)**: cache P_{t-1} = softmax(QK^T/√d).  At step t,
compute ΔP incrementally.  Memory cost (1 cached P per head per layer) vs compute
savings.

### Axis F — SEQUENCE-POSITION-CONDITIONED DEPTH / COMPUTE

TRCD gates based on TOKEN CONTENT.  Unattacked: gate based on absolute SEQUENCE
POSITION.  Mid-sequence tokens typically carry more information than edges; edge
tokens can skip deep layers.

**Candidate F1 (POS-DEPTH)**: layers 20+ only run on tokens in positions
[0.25·T, 0.75·T].  No learned router; fixed schedule.

## Prioritization

**Highest expected payoff:**
1. **Axis A (TAIL-CE)**: targets a large compute fraction (CE = 20-50% of step).
   Mechanism is grounded in FACE-validated Zipfian structure.  Memory + speed dual win.
2. **Axis D (KV-FACE)**: extends validated FACE mechanism.  Proven Zipfian premise.
   Low implementation cost (mostly reuses gpu_face.cu).
3. **Axis C (HUTCH-DIAG)**: second-order methods are classic but under-deployed.
   Strong theoretical grounding.

**Recommended for next iteration:**
Axis A (TAIL-CE) as paradigm shift #36.  Rationale:
- Zipfian mechanism proven by FACE
- CE is a large compute fraction at LLM scale
- Bias correction via importance sampling is rigorous
- Composable with FACE (both exploit Zipf on different axes)
- Gate-0 probe available: measure softmax-mass concentration in top-K logits

## Gate-0 probe for Axis A

Measure: for a trained 500M checkpoint, what fraction of softmax mass is in top-K
logits per token?  If ≥ 95% at K=512 (V=32k → 64× speedup), promote.  If <90%,
reject.  Probe cost: 1 forward pass over 1000 validation tokens.
