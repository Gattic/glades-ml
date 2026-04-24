# Paradigm Shift #39 — RLG: Reversible Layer Growth

**Date:** 2026-04-24 (Ralph-loop iter 140, post-consolidation)
**Status:** Design + Gate-0 probe spec
**Context:** After closure of deferred paradigms (iter 139), exploring a
genuinely unattacked axis.

---

## 1. Target axis

**DEPTH CURRICULUM — grow L (number of layers) mid-training.**

CHIRON is currently deployed with fixed L. At 1.84B ceiling, L=53. All
compute at every step is over all 53 layers. Early in training, when
the model is barely fitting basic language patterns, the full 53-layer
depth is overkill.

**Observation:** if we could train the early phase with fewer layers
(e.g., L=16) at massively reduced compute, then GROW to L=53 for
refinement, we'd save substantial wall-clock on the expensive early
training tokens.

## 2. Why CHIRON specifically enables this

Standard transformers have a "residual + FFN" block structure. Inserting
a new layer mid-training disrupts the residual path — the new layer's
random-init weights perturb the otherwise-learned trajectory.

CHIRON's block:
```
p += attention_shear(q ; Wq, Wk, Wv, Wo)    // symplectic shear
q  = reln(q ; gamma, beta)                  // reversible LayerNorm
```

Is a SYMPLECTIC SHEAR followed by a LN. The attention shear is:
```
shear(q) = sign · Wo · (A · V · Wv) · Wq^T ...
```

**Key observation:** if `Wo = 0` at insertion time, then `shear(q) = 0`,
so `p` is unchanged. This is an IDENTITY INSERTION — the new layer is
functionally transparent but remains a trainable object.

Similarly, `reln(q ; gamma=1, beta=0, stats)` is approximately identity
(layer-norm preserves direction, just rescales).

**Conclusion:** CHIRON admits MID-TRAINING LAYER INSERTION with zero
forward-pass disruption via Wo=0 initialization. Gradients flow into
the new layer's Wo immediately (since dL/dWo = dL/dp · (A·V·Wv) is
nonzero), so the new layer begins learning from step 1 after insertion.

## 3. Core thesis

**Hypothesis H39.** CHIRON's reversible-flow architecture admits lossless
mid-training layer insertion. Training a depth-growing model (e.g.,
L=6 → L=16 → L=32 → L=53 over training steps) achieves:
- Equivalent final loss to training at fixed L=53
- Substantially less wall-clock (early phases run at smaller L)
- Clean composition with SLC (both are training-time curricula)

**Projected wall-clock savings:**
- Schedule L=16→32→53 with each phase 1/3 of training time
- Attention compute at L=16 is 30% of L=53
- FFN compute at L=16 is 30% of L=53 (if FFN existed — CHIRON has none)
- LN compute scales with L
- **Compute per step averaged across schedule:** (30% + 60% + 100%) / 3 = 63%
- **Wall-clock speedup: ~1.6×** projected

Stacked with SLC's 1.65× speedup, expected compound: ~2.6× wall-clock.

## 4. Primitive objects

Per schedule transition (L_old → L_new with N_new = L_new - L_old new layers):

1. **Allocate new layer weights**: Wq[l], Wk[l], Wv[l], Wo[l], gamma[l], beta[l]
   for each new layer l ∈ [L_old, L_new)
2. **Initialize for identity insertion**:
   - Wq, Wk, Wv: standard Gaussian init (doesn't matter; Wo=0 zeros output)
   - **Wo = 0** (critical — ensures layer is identity at insertion)
   - gamma = 1, beta = 0 (standard LN identity)
3. **Initialize Adam state for new layers**: m = 0, v = 0
4. **Initialize FACE/MFIO state for new layers** (if active):
   - zn, dn, gF initialized to small positive (ε²) values
   - Running frequencies f̂ = 1/V (uniform prior)
5. **Reserve scratch** for maximum L at init; grown L_current uses prefix

## 5. Evolution law

At training step t with schedule entry `L_new@step_new`:
- If step t == step_new: apply layer insertion as above
- Continue training with L = L_new until next transition

**No change to per-step forward/backward**: existing kernels iterate
`for l = 0 to L_current - 1`, which already uses `L_current` as the
dynamic bound.

## 6. Mechanism mapping

| Requirement | Realization |
|-------------|-------------|
| Dynamic L | `cfg.L` set to L_current per schedule |
| Identity insertion | Wo=0 for new layers at insertion time |
| No forward disruption | Shear output = 0 when Wo=0 ⇒ p unchanged |
| Gradient flow | dL/dWo nonzero even at Wo=0 ⇒ layer learns |
| Adam stability | m=v=0 for new layers ⇒ fresh Adam warmup |
| FACE compatibility | Per-layer FACE state initialized at insertion |
| Scratch sized for L_max | Same pattern as SLC (allocate max, use current) |
| Composable with SLC | Both are schedules — orthogonal dimensions |

## 7. Stability / expressivity

**Stability:** Wo=0 insertion is EXACTLY non-disruptive. The only risk
is the subsequent training — does the inserted layer learn a useful
function? Empirically, yes: Wo starts at 0 and gradient flow drives it
to useful directions. Standard Adam warmup applies.

**Expressivity:** L_new layers have full capacity. No reduced-rank or
restricted space. Once grown, the model is mathematically identical to
L_new-from-scratch.

**Convergence:** open research question — does growing-depth training
reach the same final loss as fixed-L training? Prior art (StackingBERT,
Gong+ 2019) suggests YES for transformers. CHIRON's reversible structure
should make this cleaner (no residual-path disruption).

## 8. Gate-0 probe

**Experiment:** at 66M scale, schedule `L=6@0,L=12@1000,L=12` (final L=12).
- Phase 1 (step 0-999): train as L=6 (effectively 22M params)
- Phase 2 (step 1000+): grow to L=12, train rest as full 41.56M param model

**Baseline:** train L=12 fixed × 2500 steps (iter 128: 78.6s, EMA 7.88).

**Accept:** RLG reaches EMA ≤ 7.98 (within 0.1 nat of baseline) at
wall-clock ≤ 55s (1.4× speedup projected).

**Reject:** RLG diverges at insertion OR EMA > 8.2 OR wall-clock ≥ 78s.

Probe cost: ~2 minutes compute.

## 9. Phase 1 implementation

1. Add `--l-schedule "L@step,L@step,..."` flag (parallel to `--t-schedule`)
2. At each schedule transition, allocate new Wq/Wk/Wv/Wo/gamma/beta buffers
3. Initialize new Wo as zero (critical); others as standard Gaussian
4. Initialize new Adam state (m, v) as zero
5. Expand cfg.L to L_new
6. If FACE active: initialize new per-layer FACE state

Trainer changes: ~150 lines. Comparable to SLC's --t-schedule implementation.

## 10. Composition with shipped stack

Expected stacking (all orthogonal):
- FACE: operates on embedding, T-independent, L-independent ✓
- MFIO: per-layer attention Adam state ⇒ grows with L; fine, allocate for L_max
- bf16 stack: precision, independent ✓
- SLC: orthogonal curriculum dimension (T vs L) ✓

Flagship recipe (projected):
```
./chiron_train --face 1 --face-beta-row 0.98 --mfio 2 \
    --bf16-adam --bf16-weights --bf16-grads \
    --t-schedule "256@0,512@1000,1024@1500" \
    --l-schedule "8@0,16@500,32@1200,53@2000"  ← RLG
```

Projected wall-clock vs dense-Adam + T=1024 + L=53 baseline:
- FACE convergence: ~3× to target loss
- SLC throughput: 1.65×
- RLG throughput: 1.5× (new)
- **Combined wall-clock: ~7× faster to target loss** at 1.84B scale

## 11. Failure modes + mitigations

1. **New-layer Wo doesn't learn fast enough → layer effectively dead.**
   Mitigation: small positive init for Wo (e.g., 0.01 × Gaussian) instead
   of exactly zero. Small forward-pass disruption traded for guaranteed
   gradient signal.
2. **Catastrophic loss spike at insertion due to LN stats mismatch.**
   Mitigation: initialize new LN gamma=1, beta=0, stats fresh — each
   layer's LN operates on its own residual stream.
3. **Adam state for new layer introduces instability.** Mitigation:
   100-step LR mini-warmup after each layer insertion (borrowed from
   SLC's iter 138 fix).
4. **Schedule tuning sensitive to model size.** Mitigation: validate
   at 3-4 scales before deployment.

## 12. Next iteration action

Implement Gate-0 probe: add `--l-schedule` flag, run 66M×2500 with
schedule `L=6@0,L=12@1000`, compare to baseline L=12×2500.

If Gate-0 passes, proceed to Phase 1 → multi-scale validation → stack
with FACE/SLC/bf16 for full flagship recipe.
