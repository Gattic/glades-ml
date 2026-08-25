# Paradigm Shift #253 — SLR: Specialised Layer Roles

**Status:** designed (Ralph-loop iter 6, 2026-05-14). Builds on PSA's layer taxonomy (paradigm #252).
**Date:** 2026-05-14.
**Branch:** vesta5.
**Predecessors:** #250 SFA (per-token perspective), #251 SRA (per-query focus), #252 PSA (multi-layer cohomology → layer-role taxonomy).
**Axis:** Specialise the per-layer attention *configuration* (not the *mechanism*) based on the layer's role in the PSA-derived taxonomy. Each role (feature-extractor / synthesizer / consensus-builder) gets a tuned (d_s, r, P, λ) configuration matching its computational need.
**Magnitude target:** 1.2–1.5× wall-clock at iso-NLL on top of SFA+SRA+PSA stack, via role-matched compute allocation. Stack total: **~12–15× over SCFA flagship**.
**Caveat:** This is a *configuration paradigm*, not a *mechanism paradigm*. It does not introduce a new mathematical primitive — it uses existing #250+#251+#252 machinery with role-specific hyperparameters. The value is operational, not theoretical.

---

## 0. Executive summary

PSA (paradigm #252) gives each layer a role in the persistence-diagram taxonomy: feature-extractor (high birth, low death), synthesizer (medium both), consensus-builder (low birth, high death), or inert. PSA's operational use of this taxonomy was binary: *prune the inert layers, keep the rest uniform*.

SLR generalises: each of the three active roles gets a **specialised configuration** of the SFA/SRA attention layer, matching its computational need:

| Role | Need | SLR configuration |
|---|---|---|
| Feature-extractor | Introduce new representational primitives — rich per-token expressivity | SFA at d_s = 64, r = 8, single pole z = i √λ (small λ) |
| Synthesizer | Combine features into mid-level representations — balanced per-token + per-query | SRA at d_s = 32, r = 4, P = 1 |
| Consensus-builder | Consolidate into stable global representations — high-precision focus on persistent content | SRA at d_s = 16, r = 2, P = 4 (multi-pole) |
| Inert | (PSA-pruned) | — |

The mathematical mechanisms are unchanged from #250-#252; only the hyperparameters vary per layer.

The magnitude gain comes from two effects:

1. **Compute reduction on consensus-builder layers**: d_s = 16 vs uniform d_s = 64 gives 4× FLOP reduction per consensus-builder. With ~8 consensus-builder layers in a 24-layer stack, this saves ~25% of attention compute.

2. **Quality gain via multi-pole on consensus-builders**: P = 4 enables 4 disjoint spectral peaks for capturing different aspects of consensus content (entity identity, syntactic agreement, discourse coherence, etc.). Conjectured 0.01–0.02 nat NLL improvement per layer.

Net at iso-quality: ~1.2-1.5× wall-clock speedup.

---

## 1. Why this is paradigm #253 — and why it's narrower than #250-#252

The three preceding paradigms each introduced a new mathematical primitive:

- #250 SFA: cellular sheaves with per-token stalks (substrate primitive).
- #251 SRA: complex-pole resolvent (focus primitive).
- #252 PSA: persistence module / layer-wise cohomology (depth primitive).

SLR introduces no new primitive. It is a **configuration recipe** applied to the existing #250+#251+#252 machinery. This is honest:

- New mathematical content: minimal (uses existing machinery).
- New empirical content: moderate (the per-role configurations are testable).
- New engineering content: moderate (per-role hyperparameter scheduling, layer-type tagging).
- New mathematical content from PSA's taxonomy: substantial (the taxonomy itself was paradigm #252).

SLR is in the "exploration-of-existing-paradigm-space" category, not the "new-primitive" category. Paradigms #250-#252 carry the framework's load; SLR fine-tunes it.

This narrowness is **deliberate**: not every paradigm needs to be a primitive. Configuration paradigms have value when the configuration space is non-obvious and the design choices have empirical consequences. SLR's per-role configuration is non-obvious — it requires PSA's taxonomy to be defined.

---

## 2. Per-role configurations

### 2.1 Feature-extractor layers (early stack, high birth)

**Role**: introduce new representational primitives from raw token embeddings + positional encoding. These layers need to "see" each token in fine-grained detail.

**Configuration**:

| Parameter | Value | Reason |
|---|---|---|
| Attention type | SFA (paradigm #250) | No per-query pole needed; per-token expressivity dominates |
| d_s (stalk dim) | 64 | Match d_h for full-dimensional per-token expressivity |
| r (stalk rank) | 8 | Larger rank → richer per-token frames (capture more aspects of each token) |
| Pole / Tikhonov | z = i √λ, λ = 10^{-3} | Conservative regulariser; SFA real form |
| Edge set W | 128 | Standard sliding window |
| Sinks | 8 | Standard |
| Chebyshev M | 16 | Higher precision needed since per-token signal is high-frequency |
| Cost per token | ~16 K ops | (Per §8.2 of #250 design, d_s=64, r=8) |

**Why no per-query pole**: at early layers, the residual stream is still close to raw token embeddings. Per-query focus (which selects spectral location) is less informative than per-token expressivity (which extracts features). The SFA real-pole form suffices.

**Identification at training time**: layers where PSA's `birth_ℓ / total_active_ℓ > 0.5` for at least 50% of training steps.

**Estimated count**: ~30% of L (i.e., 7-8 layers in L=24).

### 2.2 Synthesizer layers (mid-stack, balanced)

**Role**: combine multiple feature-extractor outputs into mid-level representations (phrase-level, clause-level structure). Need both per-token and per-query expressivity.

**Configuration**:

| Parameter | Value | Reason |
|---|---|---|
| Attention type | SRA (paradigm #251) | Per-query focus matters at mid-stack |
| d_s (stalk dim) | 32 | Balanced — half of feature-extractor d_s |
| r (stalk rank) | 4 | Standard SFA rank |
| P (poles) | 1 | Single pole; mid-stack doesn't need multi-modal focus yet |
| Edge set W | 128 | Standard |
| Sinks | 8 | Standard |
| Chebyshev M | 8 | Standard |
| Cost per token | ~6 K ops | (Per closed-form SRA estimate) |

**Why d_s = 32 not 64**: at mid-stack, the residual stream already has rich features from feature-extractors. The synthesizer needs to *combine*, not *extract*. Lower d_s is sufficient.

**Identification at training time**: layers where PSA's `|birth_ℓ - death_ℓ| ≤ 0.1 · total_active_ℓ` (balanced birth/death).

**Estimated count**: ~40% of L (i.e., 9-10 layers in L=24).

### 2.3 Consensus-builder layers (late stack, high death)

**Role**: consolidate features into coherent global representations. These layers need high-precision focus on persistent content (long-bar harmonic sections) and the ability to *combine multiple consensus dimensions* (entity tracking + syntactic agreement + discourse coherence).

**Configuration**:

| Parameter | Value | Reason |
|---|---|---|
| Attention type | SRA with multi-pole (paradigm #251) | Multi-pole captures multiple consensus aspects |
| d_s (stalk dim) | 16 | Low — by late stack, individual tokens matter less; global consensus matters more |
| r (stalk rank) | 2 | Compact — consensus is low-rank by definition |
| P (poles) | 4 | Multi-pole: each pole tracks one aspect of consensus content |
| Edge set W | 256 (wider) | Larger window — late layers integrate over longer ranges |
| Sinks | 16 (more) | More sinks for global content broadcast |
| Chebyshev M | 8 | Standard (closed-form for SRA, no Chebyshev needed at d_s=16) |
| Cost per token | ~2 K ops | (Per closed-form SRA at small d_s) |

**Why P = 4**: experimental evidence in transformer attention suggests "attention heads specialise in different patterns" (e.g., positional, semantic, syntactic). SLR makes this explicit at the consensus-building layer by allocating 4 disjoint spectral peaks per query.

**Identification at training time**: layers where PSA's `death_ℓ / total_active_ℓ > 0.5` for at least 50% of training steps.

**Estimated count**: ~30% of L (i.e., 7-8 layers in L=24, *after* pruning inert layers).

### 2.4 Inert layers (PSA-pruned)

Per #252 §5.1, layers with low `birth_ℓ + death_ℓ` are pruned. SLR inherits this — there are no SLR-active configurations for inert layers.

### 2.5 Cost summary

Per L=24 layer stack (post-PSA pruning of 30%, so 17 active layers):

| Role | Count | Cost/token | Total per token |
|---|---|---|---|
| Feature-extractor | 5 | 16 K | 80 K |
| Synthesizer | 7 | 6 K | 42 K |
| Consensus-builder | 5 | 2 K | 10 K |
| **Total** | **17** | — | **132 K ops/token** |

Compare to uniform SFA at d_s=64, r=4 across 24 layers: ~24 · 8 K = **192 K ops/token**.

**Speedup: 192 / 132 = 1.45×** at iso-quality (assuming SLR's role-matched configuration preserves NLL).

---

## 3. Identification of layer roles — algorithm

Role assignment is a function of the trained model's PSA persistence diagram. The procedure:

```
Step 1: Train the L=24 stack with uniform SFA/SRA configuration for K_burn = 5000 steps.
Step 2: Compute PSA persistence diagram PD^0.
Step 3: For each layer ℓ:
        birth_frac = birth_ℓ / total_active_ℓ
        death_frac = death_ℓ / total_active_ℓ
        Assign role:
            if birth_frac > 0.5 and death_frac < 0.3: role = "feature-extractor"
            elif death_frac > 0.5 and birth_frac < 0.3: role = "consensus-builder"
            elif |birth_frac - death_frac| < 0.1:       role = "synthesizer"
            else:                                       role = "inert" (prune)
Step 4: Re-configure the stack per role.
Step 5: Fine-tune for additional K_finetune = 5000 steps with the role-matched configuration.
```

The 5000-step burn-in is necessary because PSA's persistence diagram only stabilises after the model has trained enough to develop a meaningful sheaf structure. Empirical experiments will determine the optimal K_burn.

### 3.1 Role-conditioned learning rate

After role assignment, different roles may benefit from different learning rates:

- Feature-extractors: standard LR (these layers need stable feature representations).
- Synthesizers: slightly higher LR (1.5×) — these layers are reconfigured the most.
- Consensus-builders: lower LR (0.7×) — these layers are most fragile (small changes can disrupt global consensus).

Tested at Gate-0 Probe K below.

---

## 4. Composition with prior paradigms

| Paradigm | Composes? | Mechanism |
|---|---|---|
| #42 SCFA | ✓ Inherited | SCFA basis B is the d_s=1 limit of SFA stalk frames; SLR inherits SCFA-recovery at d_s=1 |
| #250 SFA | ✓ Inherited (feature-extractor role) | Feature-extractor layers use SFA real-pole form |
| #251 SRA | ✓ Inherited (synthesizer, consensus-builder roles) | Synthesizer + consensus-builder layers use SRA |
| #252 PSA | ✓ Required | SLR depends on PSA's layer taxonomy. Cannot exist standalone. |
| #46 REFLECTOR | ✓ Inherited | Backward through Chebyshev/resolvent solves |
| #78 Sinks | ✓ Inherited | Edge set E (consensus-builder uses larger |S_sink|=16) |
| #38 SLC | ✓ Compatible | T-curriculum is orthogonal to per-layer role configuration |
| #39 RLG | ✓ Compatible | Layer growth → new layer needs role assignment after K_burn |

**Stack projection** at T=16384, L=24 (post-PSA pruning to 17 active layers):

| Stack | Per-step | Steps-to-target-NLL | Total |
|---|---|---|---|
| SCFA flagship | 1× | 1× | 1× (baseline) |
| + SFA (#250) | ~1.01× | 0.5-0.7× | 1.4-2× |
| + SRA (#251) | 4.3× | 0.5-0.7× | 6-8.6× |
| + PSA pruning (#252) | 1.4× | inherited | 8.4-12× |
| + SLR (#253, this paradigm) | 1.2-1.5× | inherited (or slightly better) | **~10-18× total** |

The headline magnitude with all paradigms stacked: **10-18× wall-clock at iso-NLL over SCFA flagship**.

---

## 5. Gate-0 falsification

In addition to inheriting paradigms #250+#251+#252's 11 probes (A-J):

**Probe K (new, SLR-specific)**: After running PSA on the burned-in model:

1. Assign roles per §3 algorithm.
2. Re-configure layers per role.
3. Fine-tune 5000 steps.
4. Compare final NLL to uniform-config baseline.

**Pass criterion**: Final NLL ≤ uniform baseline's NLL + 0.02 nat AND wall-clock per step ≤ 0.7 × uniform's wall-clock.

**Fail criterion**: NLL regression > 0.05 nat (role-matching disrupts existing representations) OR wall-clock not reduced (the per-role cost reduction doesn't materialise).

**Cost**: ~30 min on 66M reference (5000 step fine-tune at 66M scale).

### 5.1 Decision tree

- Probe K pass → SLR is operational; promote to flagship.
- Probe K fail on NLL: per-role configuration disrupts quality. Reduce role-specialisation (smaller d_s difference between roles; remove multi-pole on consensus-builders).
- Probe K fail on wall-clock: per-role compute reduction doesn't materialise. Investigate kernel-level inefficiencies.

---

## 6. Failure modes and mitigations

| Failure mode | Detection | Mitigation |
|---|---|---|
| Role assignment unstable across training | Track role-assignment changes per step | Use moving-average PSA over last K=1000 steps |
| Feature-extractor layers over-fit (high d_s = high capacity = over-fitting risk) | Per-layer training-vs-val NLL divergence | Add per-layer weight decay; lower d_s for feature-extractor → 32 if regression observed |
| Consensus-builder multi-pole collapses to single effective pole | Per-query β_q^{(p)} distribution: 1 dominant + 3 near-zero | Add regulariser: `Σ_q Σ_p ‖β_q^{(p)} - mean‖² ≥ threshold` |
| Inert layer pruning is too aggressive (KO mostly-active layers as inert) | Per-layer NLL contribution test | Use PSA threshold `birth + death > 0.1` as cutoff (instead of bottom 30%) |
| Role-specific learning rates destabilise | Per-layer gradient norm tracking | Disable role-specific LRs; use uniform LR with optional per-role LR multipliers (1.0, 1.2, 0.8) |

---

## 7. Implementation roadmap

After paradigms #250+#251+#252's Phase 4 (1B validation) completes:

**Phase 0 — SLR Gate-0** (1 GPU-hour). Probe K.

**Phase 1 — Role identification module** (2 iterations).
- Implement PSA-driven role assignment per §3.
- Add `--slr-burn-steps 5000 --slr-finetune-steps 5000` flags.

**Phase 2 — Per-role configuration injection** (2 iterations).
- Per-layer config table loaded post-burn-in.
- Stalk-dim d_s and rank r per layer (requires per-layer NNetwork primitives to support varying shapes).

**Phase 3 — Validation** (1-2 iterations).
- 66M test: SLR vs uniform NLL parity check.
- 1B test: SLR vs SFA+SRA+PSA wall-clock comparison.

**Phase 4 — Production** (1 iteration).
- Default flag: `--slr` (requires `--psa`).

**Total**: ~6-8 iterations after paradigms #250+#251+#252.

---

## 8. Open questions

1. **Optimal role-assignment threshold**: §3 uses (>0.5, <0.3) thresholds — these are educated guesses. Cross-validation on 66M model would determine empirical optimum.

2. **Role schedule stability**: Once roles are assigned at step K_burn, do they remain stable throughout training? Or do layers shift roles as the model trains? If shift, SLR may need periodic role-reassignment (e.g., every 10K steps).

3. **Hierarchical roles**: a feature-extractor at layer 3 might be qualitatively different from a feature-extractor at layer 8 (early vs late feature extraction). Should we sub-divide roles further? Empirical question.

4. **Multi-modal poles in consensus-builders**: are P=4 poles always optimal, or does the optimal P depend on the data distribution (text vs code vs scientific text)? Could be domain-adaptive.

5. **Compatibility with Curriculum-T (SLC #38)**: when T grows during training, the persistence diagram structure shifts. Does the role assignment hold across T-transitions, or does it need re-computation?

These are empirical questions; the SLR design is the framework for asking them.

---

## 9. Summary

SLR is a configuration paradigm built on PSA's layer taxonomy. Each of three roles (feature-extractor / synthesizer / consensus-builder) gets a specialised hyperparameter configuration matching its computational need:

- Feature-extractors: SFA at high d_s, r (rich per-token).
- Synthesizers: SRA at medium d_s, r, single pole.
- Consensus-builders: SRA at compact d_s, r, multi-pole P=4.

Magnitude claim: 1.2-1.5× wall-clock at iso-NLL over uniform-config SFA+SRA+PSA stack. Combined with the full prior stack: ~10-18× total over SCFA flagship.

Gate-0 (~30 min): Probe K validates that role-matching preserves NLL while reducing wall-clock.

This is a *narrower* paradigm than #250-252: it introduces no new mathematical primitive, only a configuration recipe. Its value is operational rather than theoretical. The framework can produce additional research questions (hierarchical roles, role stability over training, domain-adaptive role assignment) for future iterations.

**Predecessor**: PSA (#252) for the layer taxonomy.
**Successor**: paradigm #254 (Dynamic Depth, sketched in CELLULAR_SHEAF_ATTENTION_PROGRAM.md §7) — runtime layer-insertion driven by commutation defect. #254 is *complementary* to SLR (#254 controls layer count at runtime; SLR controls layer configuration at training-time).
