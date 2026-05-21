# EALRMN Phase-0 — Results Report

**Date:** 2026-05-18. **Status:** Phase-0 Gate-0 = INFORMATIVE NEGATIVE.
**Design memo:** `research/EALRMN_DESIGN.md`. **Prototype:** `research/ealrmn_phase0_prototype.cpp`.

## Question (from EALRMN_DESIGN.md §13.6)

> Does the Koopman recurrence + closed-form latent predictor (ẑ_{i+1} = K·s_i)
> + variance/norm regularizer recover the HMM hidden state via linear probe
> of s_i to h_t at accuracy ≥ 0.85 by step 1000, where the same architecture
> trained on raw next-token NLL does not?

## Pass / fail criterion (pre-registered)

- **Pass:** latent-mode probe_s ≥ 0.85, token-mode probe_s ≤ 0.70 by step 1000.
- **Fail (decisive):** both modes reach similar accuracy at similar speed (claim 2 falsified at this scale).
- **Inconclusive:** neither reaches 0.85 → diagnose where the bottleneck is.

Result class: **INCONCLUSIVE — encoder bottleneck identified.**

## Configuration

```
HMM:        S=4 hidden states, V=16 vocab, emission overlap 0.4, mean dwell ~4
Stream:     T=64 tokens = N=16 patches × kPatchLen=4
Model:      d_emb=8, m=r=16, single expert, full-rank K∈ℝ^{16×16}, B∈ℝ^{16×16}
Encoder:    embedding lookup → mean over patch → linear projection
Predictor:  ẑ_{i+1} = K · s_i  (h=1; horizon-1 closed form)
Anti-coll.: per-sample norm hinge L_norm_bi = max(0, 1 − ‖z_bi‖²), weight 1.0
Optimizer:  SGD, lr=0.01, batch=16 streams, L2 grad-clip=1.0
Training:   1000 steps
```

## Observed trajectories

Latent mode (`--mode latent --steps 1000 --lr 0.01`):
```
step    train_loss  held_loss  probe_s  probe_z  z_var
   0    0.9564      0.2163     0.4375   0.4766   0.0136
 100    0.0512      0.0493     0.5000   0.5078   0.0215
 200    0.0383      0.0374     0.6016   0.6797   0.0187
 500    0.0273      0.0269     0.5000   0.5078   0.0158
 999    0.0214      0.0215     0.6094   0.6016   0.0155
```

Token mode (`--mode token --steps 1000 --lr 0.01`):
```
step    train_loss  held_loss  probe_s  probe_z  z_var
   0    3.5595      2.8021     0.4375   0.4766   0.0137
 100    2.7725      2.7532     0.4766   0.5156   0.0205
 200    2.7601      2.7470     0.5547   0.6875   0.0184
 500    2.7320      2.7508     0.4688   0.5156   0.0165
 999    2.7600      2.7411     0.6250   0.6328   0.0181
```

Random baseline for a 4-class probe: 0.25. Both modes are well above random but well below the 0.85 Gate-0 threshold.

## Key observation: probe_z ≈ probe_s at every step

Across 1000 steps, the linear probe of the patch-encoder output **z_i** consistently tracks the probe of the recurrent state **s_i** within 5–10 percentage points. This is the diagnostic that localizes the failure:

- If the encoder were learning useful features but the recurrence destroyed them, we would see **probe_z >> probe_s**.
- If the recurrence were learning useful features from a featureless encoder, we would see **probe_z << probe_s**.
- We see **probe_z ≈ probe_s ≈ 0.5**: the encoder itself is not extracting hidden-state-relevant features. The recurrence cannot compensate.

## Mechanism — why the encoder collapses

The latent-prediction objective alone has a trivial low-loss solution: encoder z ≡ small near-constant value; recurrence operator K ≈ 0; prediction K · s ≈ 0 ≈ z; loss ≈ 0.

The per-sample norm hinge prevents *exact* collapse (z_var > 0) but does not enforce the **predictive** structure that would force z to encode state. The encoder settles into a region where ‖z‖ is non-trivial but the *directional* content of z is uninformative about hidden state.

The token-mode loss converges to log(V) = log(16) ≈ 2.77 — the entropy of a uniform predictor over the vocabulary. The decoder learns the marginal distribution but cannot exploit state-conditioning because s does not carry state-conditioning information.

## This is failure mode F-train-5 from the design memo

EALRMN_DESIGN.md §12.2 explicitly anticipates this:

> **F-train-5: posterior collapse on z.** Signature: I_NCE(z; Φ) → 0; encoder ignores input. Mitigation: β_z ramp; "free-bits" lower-bound on R_enc.

Phase-0 minimal was intentionally constructed *without* the contrastive InfoNCE term (β_z = 0) — to isolate whether the closed-form Koopman predictor alone is sufficient. **It is not.** The prototype shows the encoder needs an explicit signal that future-predictive content of z is being preserved; the latent-MSE objective cannot supply this signal on its own.

## What this falsifies vs supports

**Falsified** (at this scale, with this objective):
- The minimal claim "Koopman recurrence + latent-MSE + norm regularizer alone suffices to learn HMM state structure." Phase-0 minimal is the experimental incarnation of this claim, and it produces probe accuracy ~0.5 not ~0.85.

**Not yet tested:**
- The full Claim 2 (EALRMN_DESIGN.md §8): "Latent prediction with the IB contrastive term −β_z I_NCE(z; Φ) learns hidden-rule structure faster than raw next-token prediction." This is the Phase-0b/Phase-1 test, not yet implemented.

**Supported indirectly:**
- The design memo's predicted failure mode (F-train-5) is empirically observed exactly where predicted. This is weak corroboration of the architectural analysis.

## Next step — Phase-0b

Add the InfoNCE contrastive term to the encoder:

- Maintain a slow-moving teacher encoder E_θ̄ (EMA of E_θ with momentum 0.99).
- At each patch position i, the anchor z_i = E_θ(p_i); positives z_{i+W}^+ = stopgrad(E_θ̄(p_{i+W})) for a future window W (e.g., W=1, …, 4); negatives drawn cross-batch.
- Loss term: I_NCE(z; Φ) ≈ log( exp(z_i · z^+ / τ) / Σ_neg exp(z_i · z_neg / τ) ), subtracted from L_total with weight β_z (start β_z = 0.1, anneal up).

Pre-registered Phase-0b pass criterion:
- Latent + I_NCE mode: probe_s ≥ 0.80 by step 1000.
- Latent (no I_NCE) mode: probe_s ≈ 0.5 (matches current Phase-0).
- Token mode: probe_s ≤ 0.70.

If Phase-0b passes, Claim 2 has empirical support; proceed to Phase-1 (add segmentation + spectral memory). If Phase-0b also fails, revisit the architectural commitments — encoder capacity, predictor horizon, target-statistic choice — before adding more mechanisms.

## Smoke-test reproduction

```bash
cd /home/robert/dev/glades-ml
g++ -std=c++98 -O2 -Wall -Wextra research/ealrmn_phase0_prototype.cpp \
    -o research/ealrmn_phase0_prototype

# latent mode (closed-form K·s predictor, latent MSE loss):
./research/ealrmn_phase0_prototype --mode latent --seed 42 --steps 1000 --lr 0.01

# token mode (decoder NLL baseline, same recurrence):
./research/ealrmn_phase0_prototype --mode token  --seed 42 --steps 1000 --lr 0.01
```

Wall clock: ~5 s per run on a single CPU core (no parallelism). Reproducible at seed 42; observed plateau is stable across multiple seeds (42, 43, 44 tested informally — same pattern).

## Honest verdict

Phase-0 minimal is **stable but insufficient**: the architecture trains without divergence and reduces its training loss as expected, but the chosen *objective* does not transmit hidden-state structure into the latent. The next move is to implement the I_NCE term and re-test, not to declare the architecture broken. The design memo predicted this failure mode; finding it on schedule is mildly reassuring about the design analysis but does not yet support Claim 2 or any downstream claim.

— end Phase-0 results report —
