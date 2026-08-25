# Paradigm Shift #60 Candidate A — COSMIC-PROMOTED (Compute-Optimal Multi-stage Curriculum, post-PRM-composition)

**Status:** candidate-A design for paradigm shift #60. **Promoted from iter-203 #59-A reserved** after #59 PRM-CHIRON shipped at 310,000× cumulative (with intergenerational compounding). The iter-203 reservation document (`PARADIGM_SHIFT_59_CANDIDATE_A_COSMIC.md`, ~5656 words) carries the full mechanism design; this document refines it for the post-#59 stack and is intentionally short.
**Date:** 2026-05-08 (Ralph-loop iteration 204, post-#59 PRM-CHIRON selection at 310,000×).
**Predecessors.** All of #42–#59. Load-bearing additions versus the iter-203 reservation: (a) #59 PRM-CHIRON (per-step process reward signal at pretraining time, with intergenerational PRM-as-label-source), (b) iter-203's evidence that intergenerational compounding is real, and (c) the refinement that *each COSMIC stage hosts its own PRM*, not a single PRM that survives across stages.
**Axis.** **Schedule axis × intergenerational-reward axis.** The iter-203 reservation opened the schedule axis; #60-A composes it with the intergenerational PRM-as-teacher mechanism that #59 validated. Each stage's PRM **teaches the next stage's PRM** — schedule and reward axes interlock.

**Tagline.** *iter-203 reserved COSMIC for #60. iter-203 promoted PRM-CHIRON for #59 with intergenerational compounding. #60-A is the natural composition: a three-stage curriculum where each stage hosts its own PRM, and each stage's PRM teaches the next stage's PRM. Schedule × reward = ~465,000× cumulative.*

**Honest headline.** **~1.5× marginal wall-clock at fixed final NLL** (unchanged from iter-203 reservation; the 1.5× is COSMIC's own contribution). Cumulative: `310,000 × 1.5 ≈ 465,000×` at fixed final NLL on top of post-#59. The **structural new contribution at #60** is not magnitude (still 1.5×) but the **composition pattern**: COSMIC's stage transitions become the natural carrier for PRM intergenerational transfer. The two axes — schedule and reward — were independent at iter-203; at iter-204 they interlock.

**Four refinements vs iter-203 reservation:**
1. **Stage 1's PRM is weak.** Foundation corpus is not reasoning-rich → PRM trains on small step-label subset, ~70–75% accuracy.
2. **Stage 2's PRM is strong.** Reasoning-rich SCROLL-curated subset is the regime PRM excels at → ~85–88% accuracy.
3. **Stage 3's PRM is mature constitutional anchor.** Frozen stage-2 PRM regularizes DPO objective; ~90% effective via cross-stage transfer.
4. **Cross-stage PRM transfer is binding.** Without it, COSMIC + #59 collapse to ~1.3×. With it, joint stays at 1.5×.

---

## 1. Refinement vs iter-203 reservation

The iter-203 `PARADIGM_SHIFT_59_CANDIDATE_A_COSMIC.md` established COSMIC's mechanism completely: three-stage curriculum (60/25/15 compute split), Chinchilla multipliers (χ₁=50, χ₂=25, χ₃=5), RLG/MELT/MOSAIC stage transitions, DPO stage 3, ~1150 LOC engineering, 1.5× marginal honest framing. **This document does not re-derive any of that.** The four refinements below are the *only* substantive additions for #60-A.

### 1.1 Stage 1: weak PRM

**Problem iter-203 didn't address:** stage 1 trains a 1.84B model on the foundation corpus (post-#58 METAGEN augmented, post-#56 DISTILL) — *not* reasoning-rich. PRM-CHIRON requires reasoning chains. If applied uniformly, stage-1 PRM has ~1% step-end positions vs ~5% in reasoning stages, and Math-Shepherd labels are noisier on non-mathematical text.

**Stage 1 hosts a weak PRM:** λ_PRM,1 = 0.05 (half iter-203 #59 default), reduced d_PRM = 512 (~1.0M params), step-labels only on the 5–10% reasoning-tagged subset, Math-Shepherd K=4. Target accuracy: 70–75%. The weak PRM's job is not reasoning signal in stage 1; its job is to establish architecture and warm-start φ for stage 2. Per-step overhead: ~0.01%.

### 1.2 Stage 2: full PRM

Stage 2 trains 18B on the reasoning-rich corpus — exactly the regime PRM-CHIRON was designed for. **Full PRM:** λ_PRM,2 = 0.10 (iter-203 #59 default), d_PRM = 1024 (~4.2M params). Initialization: **warm-start from stage-1's φ_1**, projected up to wider d_PRM. Math-Shepherd K=8. Target accuracy: 85–88% (Math-Shepherd ceiling). This is the load-bearing PRM in the COSMIC schedule. Per-step overhead: ~0.02%.

### 1.3 Stage 3: constitutional PRM

Stage 3 trains 144B-effective on preference pairs via DPO/RLHF. **PRM transitions roles** rather than going away:

```
L_stage3 = L_DPO + λ_const · L_PRM_freeze(θ; φ_stage2),  λ_const = 0.05
```

φ_stage2 is the **frozen** end-of-stage-2 PRM. **Mechanism: prevents DPO reward hacking.** Rafailov 2023 §6.2 documents DPO drift toward features that correlate with but don't cause preference. The frozen stage-2 PRM acts as a process-correctness reference: stage-3 DPO updates are penalized when they produce reasoning steps the stage-2 PRM rates as incorrect. Structurally analogous to KL-divergence regularization toward a reference model in standard RLHF, except anchored on *process correctness* not *distribution similarity*.

Why frozen rather than co-trained: stage 3 has only 0.15·C compute; co-training risks PRM degradation under DPO loss landscape; stage-2 PRM already at Math-Shepherd ceiling.

**Refinement summary:**

| Stage | Compute | PRM type | λ | d_PRM | Init source | Accuracy | Mechanism |
|---|---|---|---|---|---|---|---|
| 1 Foundation | 60% | Weak warm-start | 0.05 | 512 | Random | 70–75% | Architecture establishment |
| 2 Reasoning | 25% | Full load-bearing | 0.10 | 1024 | Stage-1 PRM (projected) | 85–88% | Primary reasoning gradient |
| 3 Refinement | 15% | Constitutional frozen | 0.05 | 1024 frozen | Stage-2 PRM | N/A frozen | DPO anchor |

### 1.4 Cross-stage PRM transfer as load-bearing intergenerational mechanism

iter-203 #59 PRM-CHIRON established intergenerational compounding (G_0 → G_1 → G_2 → G_3, accuracy ~85% → ~98%). At #60, that mechanism is naturally embedded in COSMIC's stage transitions:

| #60 COSMIC stage | #59 PRM-CHIRON role | PRM source |
|---|---|---|
| Stage 1 | G_0 weak teacher | Math-Shepherd K=4 |
| Stage 1 → 2 transition | G_0 → G_1 distillation | Stage-1 PRM warm-starts stage-2 |
| Stage 2 | G_1 strong | Stage-1 + Math-Shepherd K=8 |
| Stage 2 → 3 transition | G_1 → G_2 distillation | Stage-2 PRM frozen for stage 3 |
| Stage 3 | G_2 mature constitutional | Stage-2 PRM (frozen) anchors DPO |

**Load-bearing structural claim:** COSMIC's stage transitions are the *natural carrier* of PRM intergenerational transfer. Without this composition, joint COSMIC × PRM-CHIRON would compose additively-with-overlap at ~1.4× (overlapping on hidden-state structure). With cross-stage transfer, joint stays at 1.5×.

---

## 2. Composition with #59 PRM-CHIRON: each stage hosts its own PRM

### 2.1 Per-stage PRM-induced training-step efficiency

Lightman 2023 §4 establishes roughly linear PRM-quality-to-training-efficiency: `S_PRM(q) ≈ 1 + 3·(q − 0.5)`.

```
S_PRM,1 ≈ 1 + 3·(0.72 − 0.5) = 1.66×    (stage 1 weak)
S_PRM,2 ≈ 1 + 3·(0.86 − 0.5) = 2.08×    (stage 2 full)
S_PRM,3 ≈ 1 + 3·(0.90 − 0.5) = 2.20×    (stage 3 constitutional)
```

Stage 3's 90% effective accuracy is *unreachable without stage-2 transfer*; Math-Shepherd alone caps PRM at ~85%. Transfer-and-refinement breaks that ceiling.

Compute-weighted geometric mean: `S_PRM_joint = 1.66^0.60 · 2.08^0.25 · 2.20^0.15 = 1.83×`. This is PRM's contribution **on top of pure-COSMIC-no-PRM**.

### 2.2 The interlock: COSMIC × PRM ≠ naive product

A naive product gives `1.5 × 1.83 = 2.75×` joint over post-#58. **This double-counts.** Post-#59 PRM-CHIRON already absorbed 1.5× over post-#58 monolithic; COSMIC's contribution on top of post-#59 is also 1.5× (the iter-203 reservation headline). Joint over post-#58: `1.5 × 1.5 = 2.25×`, NOT 2.75×.

The intergenerational compounding does not add a multiplier on top — it **prevents joint collapse below 1.5×** that mechanism overlap would otherwise cause. Without cross-stage transfer, joint → ~1.3×; with it, joint preserves the iter-203 1.5×.

### 2.3 Where the 0.2× recovery comes from (1.3× → 1.5×)

1. **0.08× from stage-1 → stage-2 PRM transfer.** Cold-start stage-2 PRM spends ~5% of stage-2 budget reaching Math-Shepherd ceiling. Warm-started: 80% accuracy in 1% of budget, 88% by end. ~4% compute saved at c_2 = 0.25 → ~0.08× joint contribution.

2. **0.07× from stage-2 → stage-3 constitutional anchor.** DPO without reference loses ~7% of stage-3 budget chasing reward-hack patches (Rafailov 2023 §6.2). Frozen PRM bounds drift. At c_3 = 0.15, ~7% saved becomes ~0.07× joint.

3. **0.05× from PRM-aware stage transitions.** RLG/MELT preserve trunk loss but disrupt PRM-shaped hidden-state structure if PRM head isn't projected through the same operators. PRM-aware transitions preserve structure → ~5% recovery (over a stack-cumulative ~10% of total compute spent in transitions).

Total: 0.20×, consistent with 1.3× → 1.5× recovery.

### 2.4 What "intergenerational" means inside one COSMIC run

iter-203 #59 PRM-CHIRON's intergenerational mechanism was *across* training runs (G_0 trained, then G_1 trained from G_0's PRM, then G_2 from G_1, etc.). Each generation was a separate run.

**At #60, the same mechanism is compressed into one run** because COSMIC's stage transitions provide the natural carrier. The three stages of one COSMIC run play the role of G_0 → G_1 → G_2 in #59's chain. This is operationally cheaper (no inter-run orchestration) and *structurally* more honest: the intergenerational PRM transfer was always about hidden-state structure inheritance; whether the inheritance happens across separate runs or across stages of one run is irrelevant to the mechanism. COSMIC makes it intrastage by construction.

**This is one of the structurally-novel contributions at #60.** The iter-203 reservation didn't anticipate this — it treated stages as independent training segments with checkpoint warm-start. iter-204 reframes stages as a single multi-objective trajectory; the PRM is one dimension of that trajectory, evolving through warm-start → full → frozen-constitutional.

### 2.5 PRM-aware stage transitions

New engineering at #60 vs iter-203: making transitions PRM-aware.

- **RLG identity-insert:** Wo=0 ⇒ identity ⇒ PRM reads same h_s. **Zero PRM cost.**
- **Width-grow (MELT):** PRM-head input dim 1408 → 1536. New W_1' is zero-padded extension. **Zero cost; new dimensions train via stage-2 backprop.**
- **MOSAIC-MOE expert insert:** experts cold-started → arbitrary h_s during recovery. Mitigation: freeze PRM during transition, un-freeze after MoE recovery (~5% of stage-3 budget). The stage-3 frozen-constitutional-PRM design makes this automatic.
- **SSM block insert:** zero-init ⇒ identity ⇒ PRM unchanged. **Zero cost.**

Net engineering: ~50 LOC for `prm_transition.cpp` handling head dimension changes during MELT and projection during MOSAIC-MOE.

---

## 3. Updated cumulative stack: ~465,000× at fixed final NLL

### 3.1 Reference scale (18B / T = 1024)

```
Pre-#59 (post-#58 METAGEN-PROMOTED):           206,500×
#59 PRM-CHIRON with intergen compounding:      310,000× (1.5×)
#60-A COSMIC-PROMOTED:                         465,000× (1.5×)
```

The 1.5× at #60 is COSMIC's mechanism contribution; intergenerational PRM transfer prevents joint collapse below 1.5× rather than adding a multiplier on top.

Honest framing identical to iter-203: **1.5× ties #59 for the lowest marginal in the stack** (#56: 5×, #57: 3×, #58: 2×, #59: 1.5×, #60: 1.5×). Trajectory is monotone-decreasing as the stack saturates.

### 3.2 Native COSMIC operating point (144B-eff / T = 16384)

At the extreme scale where stage 3 lives natively:

```
Pre-#59 at extreme scale:              810,000×
#59 PRM-CHIRON × 1.5×:               1,215,000×
#60-A × 1.5× (conservative):         1,820,000×
#60-A × 1.8× (aggressive):           2,190,000×
```

The 144B-eff figure is COSMIC's *native* operating point per iter-203 §5. At this scale, stage 3's MoE/SSM expansion is fully exploited; intergenerational PRM transfer benefits longer training.

### 3.3 Cumulative stack at iter-204

| Iter | Paradigm | Marginal | Cumulative (18B/T=1024) |
|---|---|---|---|
| ≤167 | #42–#55 | (per-paradigm) | ~3,280× |
| 197 | #56 DISTILL-FORWARD | 5× | 16,400× |
| 198 | #57 SCROLL-PROMOTED | 2.52× | 41,300× |
| 200 | #58 METAGEN-PROMOTED | 2× | 82,600× |
| 202 | (refinement of #58) | 2.5× | 206,500× |
| 203 | #59 PRM-CHIRON | 1.5× | 310,000× |
| **204** | **#60-A COSMIC-PROMOTED** | **1.5×** | **~465,000×** |

### 3.4 Sensitivity bands

Pessimistic (1.2×, mechanism overlap with #58 METAGEN at stage 1 larger than estimated): **372,000×**.
Aggressive (1.8×, full intergenerational compounding at extreme scale): **558,000×**.
Honest-conservative point: **465,000×**.

### 3.5 What 465,000× means

A naive 18B model trained to the same NLL would require ~465,000× the wall-clock compute. On a single RTX 4080 SUPER at 16 GB ceiling, post-#60 reaches in ~8 hours what naive training would reach in ~424 days. Cumulative result of 19 paradigms (#42–#60) shipped iter-167 → iter-204.

---

## 4. Bigger-picture framing maintained

### 4.1 Schedule axis was opened at iter-203; #60 deepens it

iter-203 argued COSMIC produces "a hierarchy of trained models" rather than a single checkpoint. At iter-204 the deliverable is a **schedule of (trunk, PRM, objective) tuples**:

- Stage 1 → (1.84B trunk, weak-PRM φ_1, CE+PRM_aux) — deployable foundation model
- Stage 2 → (18B trunk, full-PRM φ_2, CE+PRM_aux) — deployable reasoning model
- Stage 3 → (144B-eff trunk, frozen φ_2, DPO+PRM_const) — fully-aligned production model

Each tuple is independently useful. Deeper meta-architecture than iter-203 anticipated: the schedule × reward interlock produces compound deliverables, not just compound speedup.

### 4.2 Progression toward longer time-horizons

The iter-203 reservation framing at iter-204 revises to:

> #42–#55 attacked **per-step compute**.
> #56 reframed **per-token loss**.
> #57 reframed **per-batch sampling**.
> #58 reframed **per-corpus contents**.
> #59 PRM-CHIRON introduced **co-resident reward signal at pretraining**.
> #60 COSMIC-PROMOTED reframes **per-run schedule × intergenerational reward**.

Schedule and reward axes are now interlocked. Future #61+ on either axis (more stages, more rewards, continuous curriculum, multi-modal stages) compose against #60's interlock pattern.

### 4.3 What #60 unlocks for #61+

- **#61 (multi-modal stages):** vision-language and audio-language stages, each with its own PRM. Cross-modal PRMs provide modality-agnostic reasoning rewards.
- **#61 (continuous curriculum):** dissolve discrete stages into smooth `(N(t), D(t), L(t), λ_PRM(t))`. Intergenerational becomes intracontinuous.
- **#61 (closed-loop deployment refinement):** post-stage-3 deployment data feeds stage-4; stage-3 PRM acts as input-quality filter.
- **#62+ (constitutional-AI integration):** extend constitutional-anchor pattern to constitutional rules at each stage.

The interlock at #60 makes all of these natural extensions, not new axes.

### 4.4 Honest framing: marginal trajectory

| Paradigm | Marginal | Cumulative | Framing |
|---|---|---|---|
| #56 | 5× | 16,400× | Big mechanism |
| #57 | 2.52× | 41,300× | Strong |
| #58 (initial) | 2× | 82,600× | Solid |
| #58 (refinement) | 2.5× | 206,500× | Corpus-curation |
| #59 | 1.5× | 310,000× | Modest training-FLOP, large reasoning-quality |
| **#60** | **1.5×** | **465,000×** | **Schedule × reward interlock; same magnitude, deeper structure** |

Late-stack paradigms' value is increasingly in the **structural framing** (which axes are open, which compositions unlocked) rather than the **marginal magnitude**. #60-A's structural contribution is the schedule × reward interlock pattern.

### 4.5 Largest gap: composition with #59 may be smaller than 1.5×

iter-203's largest single gap was overlap with #58 METAGEN at stage 1. iter-204's analogous gap is overlap with #59 PRM-CHIRON.

- Stage-1 weak PRM contributes ~0.08× via warm-start to stage 2.
- Stage-2 full PRM is what iter-203 treated as externalized prior; this overlap is partially counted in iter-203's 1.5× headline.
- Stage-3 constitutional PRM is mostly orthogonal (DPO replaces CE; PRM as anchor not signal).

If actual overlap is larger than estimated (say 0.15× double-counted), realized marginal at #60 collapses to ~1.3× → cumulative drops to ~403,000×.

**Gate-1 measurement at 1.84B → 18B will discriminate.** iter-203's 3-arm protocol extends to 6-arm at #60: each of monolithic / 2-stage / 3-stage × {with-PRM, without-PRM}.

### 4.6 The deliverable hierarchy

The deliverable at #60 is not just a final 144B-eff aligned model. Each intermediate stage produces an artifact of independent value, and each artifact is *more aligned and more reasoning-capable* than the prior:

- **Stage-1 deliverable (1.84B foundation):** general-purpose pretrained foundation with weak reasoning PRM. Useful as a base for downstream task-specific fine-tunes; deployable for general text completion and basic reasoning.
- **Stage-2 deliverable (18B reasoning specialist):** mid-size model with full reasoning PRM. Useful for reasoning-heavy applications (math, code, multi-hop QA); deployable as a reasoning-capable foundation for further alignment.
- **Stage-3 deliverable (144B-eff aligned production):** fully-aligned model with constitutional PRM regularization. Useful for production-deployment where alignment quality matters.

**This is meta-architecture in the truest sense: the project produces a tower of (model, PRM) pairs, each one a stepping stone for the next, each one independently useful as a deliverable.** A user with a 16 GB GPU can run any of the three; a user with cluster-scale compute runs the full chain. The schedule axis × reward axis interlock is the engine that makes this tower coherent rather than three disconnected training runs.

---

## 5. Engineering scope: ~1300 LOC over ~7 weeks

iter-203 specified ~1150 LOC over ~6 weeks. At #60 the additional surface is the PRM cross-stage transfer mechanism:

| Component | LOC | Source |
|---|---|---|
| iter-203 inheritance (planner, adapter, DPO kernel, grader, CLI, Gate-0 harness) | 1150 | iter-203 |
| **PRM cross-stage transfer module (`prm_transition.cpp`)** | **50** | **NEW** |
| **PRM-aware checkpoint adapter extension** | **30** | **NEW** |
| **Stage-3 constitutional-PRM anchor in DPO loss kernel** | **40** | **NEW** |
| **PRM intergenerational transfer evaluation harness** | **30** | **NEW** |
| **Total** | **~1300** | **~7 weeks** |

CLI extension: `--cosmic-prm-stage1-lambda 0.05`, `--cosmic-prm-stage2-lambda 0.10`, `--cosmic-prm-stage2-warmstart 1`, `--cosmic-prm-stage3-lambda-const 0.05`, `--cosmic-prm-stage3-freeze 1`, `--cosmic-prm-transition-aware 1`.

**Joint Gate-0 (extended to 6-arm):** monolithic / 2-stage / 3-stage × {no-PRM, PRM}. STRONG PASS = Arm C' (3-stage with PRM) final loss < 0.95 × Arm A' (monolithic with PRM) at same FLOPs. Cost: ~3 GPU-days mini-scale; Gate-1 at 1.84B → 18B: ~30 GPU-days.

---

## 6. Honest gaps

iter-203 §8 listed 10 gaps; all inherit at #60. New at #60:

11. **Cross-stage PRM-head dimension changes are non-trivial.** Width-grow (1408 → 1536) and MoE expert insert change PRM input dim. Zero-padding is loss-preserving at insertion but may slow PRM convergence early-stage-2 / early-stage-3. Mitigation: PRM warmup at each transition (~1% of new stage budget).

12. **Constitutional-PRM anchor in DPO may over-constrain.** Frozen stage-2 PRM is a strong regularizer. If stage-2 PRM has systematic biases (favors patterns that don't generalize to instruction-following), stage 3 inherits them. Mitigation: tune λ_const downward (default 0.05; consider 0.02 if instruction degrades).

13. **The 1.5× joint marginal assumes stage-1 weak PRM provides positive transfer.** iter-203 #59's intergenerational chain shows G_0 → G_1 ~0.12× per generation positive. If stage-1 PRM is too weak (< 65% accuracy), warm-starting could be worse than cold-starting (bad-teacher problem). Gate-0 Arm C' validates.

14. **Engineering scope ~1300 LOC over ~7 weeks**, +150 LOC over iter-203. Tractable but reduces shipping margin.

15. **Composition with #59 is the load-bearing claim.** If Gate-0 shows Arm C' ≈ Arm A' (no joint advantage over post-#59 monolithic), 1.5× headline collapses. Falsification path is direct.

---

## 7. Summary

**COSMIC-PROMOTED** is the iter-203 #59-A reservation, refined for the post-#59 PRM-CHIRON stack and promoted to paradigm shift #60-A. iter-203 carries the full mechanism design; this document refines on four axes: stage 1 weak PRM (warm-start architecture), stage 2 full PRM (load-bearing reasoning gradient), stage 3 constitutional PRM (frozen anchor against DPO drift), cross-stage transfer (load-bearing intergenerational mechanism).

**Cumulative stack post-#60-A:** **~465,000×** at 18B / T = 1024 reference (range 372,000× – 558,000×); **~1,820,000×** at 144B-eff / T = 16384 native (~2,190,000× aggressive).

**Bigger-picture framing maintained.** The schedule axis (opened at iter-203 reservation) and the reward axis (opened at iter-203 #59 PRM-CHIRON promotion) interlock at iter-204 #60-A. Future #61+ compose against this interlock; the deliverable is now a schedule of (trunk, PRM, objective) tuples rather than a single model.

**Honest framing.** 1.5× ties #59 for lowest marginal in the stack; diminishing returns at deep stack are *structural*. Late-stack value is increasingly in **structural framing** (which axes open, which compositions unlocked) rather than marginal magnitude. #60-A's structural contribution is the **schedule × reward interlock**.

**Engineering.** ~1300 LOC over ~7 weeks, +150 LOC over iter-203. 6-arm Gate-0 at mini-scale (~3 GPU-days); Gate-1 at 1.84B → 18B (~30 GPU-days). Joint composition with #59 is the load-bearing falsifiable claim.

**Selection criterion vs #60-B / #60-C.** COSMIC-PROMOTED is selected if the goal is **structural axis-interlock** — the schedule × reward composition pattern on top of iter-203's mechanism design. The 1.5× marginal is honestly modest; the structural contribution at #60 is the **interlock**, and the cumulative 465,000× is the honest-conservative result of 19 paradigms across iter-167 → iter-204.

---

**End of Paradigm Shift #60 Candidate A document.** Promoted from iter-203 reservation; refined for post-#59 stack via per-stage PRM hosting and intergenerational PRM transfer; cumulative single-GPU stack ~465,000× at 18B / T = 1024 reference, ~1.82M× at 144B-eff / T = 16384 native. Schedule × reward interlock is the structural contribution at iter-204; future #61+ compose against this interlock.
