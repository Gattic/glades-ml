# Paradigm shift #12 — design brief

**Date**: 2026-04-22.
**Status**: design phase.  Implementation deferred.

---

## Context — what's left after shifts #1-#11

Shifts #1-#11 address memory and compute along every axis inside the
standard gradient-descent framework:

| axis                 | attacked by                           |
|----------------------|---------------------------------------|
| activation memory    | #1 (CHIRON reversibility)             |
| weight memory        | #5, #7, #10                           |
| gradient memory      | #4 (BF16), #9 (OVFG factored)         |
| optimizer memory     | #3, #9, #11 (MFIO — zero per-param!)  |
| loss/logits memory   | chunked CE                            |
| attention compute    | #2, #6                                |
| sequence length      | #8 HRTC                               |

Every one of these shifts still requires **gradient backpropagation**
— and backprop itself is a non-trivial cost:
- 2× the forward compute (roughly) every training step,
- requires storing activations for the backward (offset partly by
  #1, but the pre-reversible footprint is real),
- couples the loss at the output to every layer, limiting parallelism.

## Target for shift #12

**Train LLMs WITHOUT BACKPROPAGATION.**  If we can replace the global
gradient-backward chain with per-layer local objectives, we eliminate:
- The 50% of step time currently spent on the backward pass,
- The inter-layer sequential dependency (enables depth-wise
  parallelism),
- The "long gradient path" instability that plagues very deep
  transformers at low precision.

Target: 2× training throughput at the same final quality, OR same
throughput at meaningfully larger model capacity.

## Three candidate formulations

### Candidate A — Forward-Forward (FF) training

Hinton 2022.  Replace backprop with two forward passes:
- **Positive pass**: real training data; each layer maximizes a
  "goodness" function of its activation (e.g., ‖h_ℓ‖² − threshold).
- **Negative pass**: corrupted/adversarial data; each layer minimizes
  the same goodness function.

Each layer has its own local objective; no gradient flows across
layers.  Weights are updated with one gradient step per layer using
only the LOCAL activation and its gradient — computable from a tiny
forward-only chain at each layer.

**Compute**: 2 forward passes per step (no backward).  If forward is
cheaper than backward (typical ratio ~1:2), net speed is (2·F)/(F+B)
= 2/(1+2) = 2/3 — about 33% slower, NOT faster.

Verdict: fails the speed target on transformer architectures where
backward is more expensive than forward due to attention.

### Candidate B — Direct Feedback Alignment (DFA) with local updates

Nøkland 2016, adapted.  Use a RANDOM FIXED backward matrix instead of
the true transposed forward weights.  Each layer's weight gradient is
computed from (activation, random-projected output error) — no
true-gradient propagation.

Each layer's update is independent: compute forward, sample output
error (either from a locally computed local objective or from a
random projection of global loss), update with local rule.

**Compute**: 1 forward pass + 1 random-projection broadcast of the
output error = ~1.1 forward passes per step.  That's 2× faster than
forward+backward.  Speed target HIT.

Known limitations: DFA convergence is quality-matched to backprop
only on ≤10-layer networks in prior art.  Scaling to 48-layer
transformers unproven.

### Candidate C — Layerwise synthetic gradients (LSG)

Jaderberg 2017.  Each layer has a small auxiliary "critic" network
that PREDICTS the downstream gradient from the layer's own activation.
Forward pass runs normally.  Each layer does a local update using the
critic's predicted gradient instead of the true backprop gradient.

Critic networks are trained (online, asynchronously) against the
actual backprop gradient — but DON'T need to run synchronously.  The
main path is forward-only.

**Compute**: 1 forward + 1 critic forward + (asynchronous, amortized)
critic training = ~1.2 forward passes per step on the critical path.
Speed target hit.

Risk: critic accuracy bounds the quality.  Prior art shows LSG works
at moderate depth but lags backprop at LLM scale.

## Selection: **Candidate B (DFA) — with an LLM-specific enhancement**

### The LLM-specific enhancement: "Local-Attention-Aware DFA"

Standard DFA uses a fixed random projection for all layers.  For LLMs
with attention this is suboptimal because the attention mechanism has
STRUCTURE that a random matrix discards.

Proposal: each attention block uses a **content-aware backward
projection** derived from its OWN attention weights (available from
forward).  Each MLP block uses standard random DFA.

Update rules:
- For a Wo projection in layer ℓ with output error e_ℓ:
  dWo_ℓ ∝ (attention_weights_ℓ^T · e_global_output) · activation_ℓ
- For Wq, Wk, Wv: similar, using the attention's own content
  projection as the "backward" matrix.
- For MLP weights: use fixed random R_ℓ to project e_global_output.

### Memory impact

No added persistent state beyond what #9 (OVFG) already needs.  The
fixed random matrices R_ℓ are ~O(d_out × d_out_final) per layer —
small compared to the weights themselves.  Compose with:
- MFIO for optimizer state (zero optimizer storage): stacks cleanly.
- OVFG for gradient factored storage: stacks cleanly; the "gradient"
  in DFA is just the random-projected output error.

### Minimal prototype (≤ 3 weeks)

1. **Week 1**: `gpu_dfa.{h,cu}` with:
   - `dfa_random_matrix(L, d_per_layer, seed, R_out)` — generate fixed
     per-layer random backward matrices at init.
   - `dfa_layer_update(θ_l, z_l, e_global, R_l, η)` — single-layer
     local update rule.
2. **Week 2**: CHIRONDfaDescentTest and CHIRONDfaMLPTest, comparing
   DFA vs Adam at L=2, 4, 8 on ReLU MLPs.
3. **Week 3**: wire into chiron_main.cpp behind `--dfa` flag, run
   pile_large smoke test (small model) for 500 steps, verify loss
   trajectory within 2× of backprop baseline.

### Open questions

1. **Does DFA converge on transformers at all?**  Prior art tops out
   at ~10 layers of MLP/ConvNet.  Our 48-layer transformer case is
   substantially deeper and relies heavily on attention for long-
   range credit assignment.  The LLM-specific content-aware
   enhancement is the hypothesis that bridges this — empirically
   untested at LLM scale.
2. **Interaction with pre-LN vs post-LN**: the normalization layer's
   gradient is critical for stability.  DFA's random projection may
   not respect LN's variance constraints.  Mitigation: use true
   backprop through LN layers only, DFA for linear projections.
3. **Stiefel + DFA**: Stiefel tangent projection assumes access to
   the true gradient.  Does DFA's random-projected error preserve
   the tangent structure enough to admit Riemannian retraction?
   Open.

## Failure modes and mitigations

1. **Non-convergence at L > 10**: historical DFA ceiling.  Mitigation:
   hybrid (true backprop through every k-th layer, DFA elsewhere).
   Retains most speed win if k ≥ 4.
2. **Quality gap vs Adam**: DFA is a less-accurate estimator of the
   gradient; final loss may plateau higher.  Mitigation: use DFA for
   the BULK of training, fine-tune the final 10% of steps with true
   backprop.
3. **LN instability**: see above — true-backprop through LN.

## Tracked as task #36 (to be created on implementation start)
