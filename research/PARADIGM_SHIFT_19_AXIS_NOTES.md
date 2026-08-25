# Paradigm Shift #19 — Candidate Axis Scouting

Notes for the NEXT paradigm shift's axis selection.  Not yet formalized
into 3-candidate design via the research-framework-design skill — that
dispatch happens in a later iteration.

## Axes attacked by shifts 1-18

Shipped:     1 CHIRON activation; 2 TC-tiled attn; 3 int8 Adam; 4 BF16 grads;
             5 SR BF16 weights; 6 local-window attn; 7 Stiefel weights;
             9 OVFG factored grads; 10 MPOT weights; 11 MFIO opt-free;
             12 DFA backprop-free; 13 TRCD per-token depth;
             16 LCP per-token pooling.
Deferred:   14 IED implicit depth; 15 TPW trajectory-predictive;
            17 GFIB per-param update gate; 18 SGS per-layer substitution.

## Genuinely unattacked axes

### Axis R1: Precision heterogeneity
No shift personalizes bit-width per parameter.  Shift #5 is uniform BF16;
#3 is uniform int8 on Adam.  A heterogeneous precision axis:  critical
params at FP16, rarely-updated at int4 or even 1-bit sign.  Compose with
GFIB (#17): low-F̂ params get low-precision storage.

### Axis R2: Effective sequence compression
No shift compresses the T-axis itself (LCP dedups compute, but keeps T
tokens in the loss computation).  An axis that merges consecutive similar
tokens into virtual tokens, reducing effective T for main compute.

### Axis R3: Predictive forward emulation
SGS (#18) emulates per-layer; TPW (#15) predicts weight trajectory.
Neither predicts the MAIN FORWARD PASS end-to-end — a small "mirror
network" trained to emulate the full network, used as a cheap oracle
during training to decide whether to compute the full pass.

### Axis R4: Retrieval-based activation reuse
Maintain a cache of (input_hash → activation) from recent steps.  When
an input matches a cached hash, reuse its activation modulo a stochastic
correction.  Distinct from LCP: LCP operates within a batch; R4 operates
across steps.

### Axis R5: Per-parameter learning-rate schedule
We use one global η schedule.  Each parameter should have its own,
inferred from its gradient trajectory's volatility statistic.  Composes
with MFIO (#11) — per-layer σ IS a crude per-parameter η.

## Leading candidate: Axis R1 (precision heterogeneity)

Rationale:
- Direct memory win: bit-width × param-count.  At 2.23B, moving 50% of
  params from BF16 (2 bytes) to int4 (0.5 bytes) saves 1.7 GB of weights.
- Composable with EVERY shipped shift (every shift has a "what's the
  storage precision" axis that can be tuned per-param).
- The HARD problem: how to decide which params get which precision —
  that's where the research contribution lies.
- GFIB's (#17) Fisher-magnitude criterion is a natural partner.

## Leading secondary: Axis R3 (predictive forward emulation)

Rationale:
- If ANY network can emulate the main forward at 10% of its cost, we save
  90% of the forward compute we can skip.
- The "mirror network" idea is not new (knowledge distillation), but
  distillation is TEACHER→STUDENT with a fixed teacher.  Here, the mirror
  network is trained ONLINE to match the current network, and used as an
  ORACLE to decide whether to run the full forward.
- Failure: mirror network drifts; needs periodic recalibration.

## Decision criterion for next iteration's dispatch

When scheduling the 3-candidate dispatch:
- A: Axis R1 (precision heterogeneity — PRX: per-param precision)
- B: Axis R3 (predictive forward emulation — PFE: mirror-guided skip)
- C: Axis R4 or R5 (to be determined — the "wild card" axis)
