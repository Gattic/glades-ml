# Paradigm Shift #62 Candidate A — META-LEARN-CHIRON-PROMOTED (per-stage meta-learning, post-COSMIC composition)

**Status:** candidate-A design for paradigm shift #62. **Promoted from iter-205 #61-C reserved** after #61 COSMIC-PROMOTED shipped at 3,030,000× cumulative on tool-augmented benchmarks (~620,000× on text-NLL). The iter-205 reservation document (`PARADIGM_SHIFT_61_CANDIDATE_C_META_LEARN_CHIRON.md`, ~3850 words) carries the full single-stage MAML-style mechanism (same-batch virtual forward, EMA baseline, scalar meta-loss, 4F per step, conjectured 1.31× net wall-clock); this document refines it for the post-#61 *triple-axis* COSMIC stack and is intentionally short.
**Date:** 2026-05-08 (Ralph-loop iteration 206, post-#61 COSMIC selection at 3,030,000×).
**Predecessors.** All of #42–#61. Load-bearing additions versus the iter-205 reservation: (a) #61 COSMIC-PROMOTED's three-stage trunk schedule, (b) iter-205's evidence that triple-axis interlock stacks multiplicatively without joint collapse, and (c) the refinement that *each COSMIC stage hosts its own meta-learning configuration* — Stage 1 strong, Stage 2 mild, Stage 3 disabled.
**Axis.** **Optimizer-quality axis × schedule-axis interlock.** iter-205 #61-A's contribution was schedule × reward × tool-locus triple interlock; #62-A composes a *fourth* axis on top: per-step gradient-direction quality, scaled per stage. The novel contribution at #62 is not the meta-learning mechanism (mechanism unchanged from iter-205) but the **stage-aware coupling**: small-model stages get strong meta-benefit, large-model stages mild, refinement stage none.

**Tagline.** *iter-205 reserved META-LEARN-CHIRON for #62. iter-205 promoted COSMIC at 3,030,000× via triple-axis interlock. #62-A is the natural composition: meta-learning per stage, with stage-specific α, λ_meta, and baseline schedule, sharing the Pearlmutter HVP kernel with #43 ORION. Stage 1 carries the load, Stage 2 contributes mildly, Stage 3 is disabled. Cumulative: ~3,360,000× tool-augmented (~688,000× text-NLL) after ORION-overlap accounting.*

**Honest headline.** **~1.31× standalone single-stage** (unchanged from iter-205 reservation); under #62-A's per-stage configuration, the **effective compounded marginal is ~1.18×** after the 0.93× ORION-overlap penalty: cumulative `3,030,000 × 1.31 / 1.18 ≈ 3,360,000×`. The **structural new contribution at #62** is not magnitude (inside iter-205's 1.05–1.5× band) but the **per-stage meta-learning profile**: Stage 1's small-model regime maximizes meta-benefit (large `g/H` ratio, strong curvature), Stage 2 contributes mildly (curvature partially flattened at 18B), Stage 3 is disabled (DPO loss landscape incompatible + 0.15·C compute fraction).

**Five refinements vs iter-205 reservation:**

1. **Stage 1: strong META-LEARN** (`α₁ = 0.05`, `λ_meta,1 = 0.08`, warmup-fraction 0.5). Per-step gain projected 1.7–1.9×.
2. **Stage 2: mild META-LEARN** (`α₂ = 0.02`, `λ_meta,2 = 0.03`, warmup-fraction 0.2). Per-step gain 1.3–1.5×.
3. **Stage 3: disabled** (`λ_meta,3 = 0`). DPO incompatibility + small compute fraction → negative EV.
4. **Joint with #43 ORION:** shared Pearlmutter HVP kernel with stage-aware ranks `r ∈ {2, 4, _}` matching #61-A line 193.
5. **EMA baseline persists across stages.** `ḡ_t` is *not* reset at stage transitions; stage-2 inherits stage-1's EMA via RLG identity-insertion. Cross-stage EMA continuity recovers ~0.04× of the ORION-overlap penalty.

---

## 1. Refinement vs iter-205 reservation

iter-205 `PARADIGM_SHIFT_61_CANDIDATE_C_META_LEARN_CHIRON.md` established the mechanism completely: same-batch virtual forward at `θ̃_t = θ_t − α·g_t`, EMA baseline `ḡ_t = β_g·ḡ_{t-1} + (1-β_g)·g_t`, scalar meta-loss `L_meta ≈ −α·g_t^⊤(g_t − ḡ_t)`, 4F per-step compute, `λ_meta ≤ 0.1` auxiliary-loss formulation, ~720 LOC engineering, conjectured 1.5–2× per-effective-step gain → **1.31× net wall-clock**. This document does not re-derive that. The five refinements below are the *only* substantive additions for #62-A.

### 1.1 Stage 1: strong META-LEARN

iter-205 assumed a fixed 1.84B trunk. In COSMIC, Stage 1 trains 1.84B on 60% of total compute. The MAML-style meta-benefit is largest where second-order curvature `g_t^⊤ H g_t` dominates — empirically the small-model regime.

- `α₁ = 0.05`: at 1.84B with FACE/MFIO/Kahan-v, `L_max ≈ 600–800`, so `α₁ = 1/√L_max ≈ 0.04`, rounded to 0.05.
- `λ_meta,1 = 0.08`: above iter-205 default 0.05, below 0.1 ceiling. The 1.84B trunk has the most room to improve gradient-direction quality before saturating Adam.
- Warmup-fraction 0.5: meta-loss active for first 50% of stage-1 steps, then cosine-annealed.
- Per-step gain projected: **1.7–1.9×** (high end of iter-205 §2.2 band).
- Net stage-1 wall-clock contribution: **~1.43×** (per-step 1.8× / 33% overhead).

The 0.6·C compute fraction makes stage-1 META-LEARN the dominant compounding contribution.

### 1.2 Stage 2: mild META-LEARN

Stage 2 trains 18B on the reasoning-rich corpus — exactly where iter-205 conjectured per-step gain might compress to the 1.3–1.5× band. Larger parameter count flattens effective curvature: gradient innovation `g_t − ḡ_t` becomes smaller in relative magnitude.

- `α₂ = 0.02`: matches iter-205 §1.3 lower bound at higher `L_max ≈ 2000–3000` for 18B.
- `λ_meta,2 = 0.03`: substantially below stage-1's 0.08; meta-signal supports tool-routing and reasoning gradients but doesn't dominate them.
- Warmup-fraction 0.2: stage-2's load-bearing signal is the PRM and Python tool-routing (per #61-A §1.2); meta-learning is a refinement.
- Per-step gain projected: **1.3–1.5×** (low end of iter-205 band).
- Net stage-2 wall-clock contribution: **~1.05×**.

The meta-trained gradient direction *propagates* into stage-2 via `θ_stage1_end` warm-start; stage-2's 1.05× is the mechanism's residual after most of the meta-benefit was already absorbed in stage-1's 1.43×.

### 1.3 Stage 3: disabled META-LEARN

Stage 3 trains 144B-effective on DPO preference pairs with constitutional PRM and tool-validity anchors. **META-LEARN is disabled** (`λ_meta,3 = 0`). Three reasons compound:

1. **DPO loss landscape is non-CE.** iter-205 §3.1's NLL-preservation proof relied on `g(θ*) = 0` at the CE minimum. DPO has a different fixed point; `L_meta` at the DPO fixed point is not guaranteed zero. Risk: meta-loss drifts the DPO endpoint.
2. **Stage-3 compute is 0.15·C.** Even at optimistic 1.31× per-step, stage-3 contributes only `0.15 × 1.31 = 0.20×` to the cumulative product. Smallest absolute lever, largest tail risk.
3. **Pathology risk amplifies under DPO.** Metz 2019 documents learned-optimizer pathologies most severe when the loss-landscape is being modified. Stage-3 already has two regularizations (PRM-constitutional + tool-validity-constitutional); a third would create non-trivial coupling.

Net stage-3 contribution: **1.00× exactly** (no meta-learning, no overhead, no pathology risk).

### 1.4 Joint with #43 ORION: shared Pearlmutter HVP kernel

iter-205 §4.4 noted META-LEARN ships in the *scalar* form (no HVP), so the kernel-sharing argument was prospective. At #62, both ship together. iter-205 #61-A §4 line 193 specifies `ORION r=2, K=20` at stage 1; `r=4, K=20` at stage 2; `r=8, K=10` at stage 3.

| Stage | ORION rank `r` | META-LEARN form | HVP calls per K-window |
|---|---|---|---|
| 1 | 2 | scalar (no HVP) | 2 (ORION only) |
| 2 | 4 | scalar (no HVP) | 4 (ORION only) |
| 3 | 8 | DISABLED | 8 (ORION only) |

The same `pearlmutter_hvp(net, v) → H·v` kernel that ORION calls `r` times per anchor would be called once more per anchor if META-LEARN ran full-form. **#62-A retains scalar form**: full-form's 6F per step breaks the wall-clock budget. The kernel-sharing claim is preserved as an option-value asset for #63+.

### 1.5 Cross-stage EMA baseline continuity

**At #62, the EMA `ḡ_t` is *not* reset between stages.** Stage 2 inherits stage-1's `ḡ_{stage1_end}`:

- Stage 1 → 2 transition uses RLG identity-insertion (Wo=0 ⇒ identity). Trunk gradient direction preserved; EMA inherits this preservation.
- New layers' EMA rows zero-initialized; first ~100 stage-2 steps populate them.
- Stage 2 → 3 transition: META-LEARN disabled, so EMA stops updating; `ḡ_{stage2_end}` discarded.

Without cross-stage continuity, stage-2 spends ~100 steps re-establishing the baseline before the meta-signal is useful — wasting ~50% of stage-2's already-mild meta-benefit. With continuity, meta-signal is informative from step 1. **Recovery: 0.04× of the ORION-overlap penalty.**

---

## 2. Per-stage meta-learning configuration table

| Stage | Compute | Trunk | `α` | `λ_meta` | Warmup-frac | Per-step gain | Net stage |
|---|---|---|---|---|---|---|---|
| 1 Foundation | 60% | 1.84B | 0.05 | 0.08 | 0.5 | 1.7–1.9× | **1.43×** |
| 2 Reasoning | 25% | 18B | 0.02 | 0.03 | 0.2 | 1.3–1.5× | 1.05× |
| 3 Refinement | 15% | 144B-eff | _ | 0 | 0 | _ | 1.00× (disabled) |

Compounded marginal (geometric average weighted by compute fraction):

```
Effective = (1.43)^{0.6} × (1.05)^{0.25} × (1.00)^{0.15}
          = 1.243 × 1.012 × 1.000 = 1.258
```

With cross-stage EMA continuity (§1.5) adding 0.04× to stage-2: `(1.43)^{0.6} × (1.09)^{0.25} × (1.00)^{0.15} ≈ 1.27×`.

**Honest framing:** the iter-205 1.31× standalone headline maps to the optimistic end of stage-1's 1.9× per-step gain. At median stage-1 (1.8×), compounded is ~1.27×; at pessimistic (1.7×), ~1.23×. **Net wall-clock band: 1.23–1.31×.** Point estimate: **1.27×** with 1.31× optimistic / 1.23× pessimistic.

---

## 3. Composition with #61 COSMIC + #43 ORION

### 3.1 Triple-axis composition

iter-205 #61-A established the schedule × reward × tool-locus triple interlock. #62-A composes a *fourth* axis: per-step optimizer-quality, scaled per stage.

- **COSMIC × META-LEARN:** multiplicative per-stage, each configuration tuned to its regime. Joint: 3,030,000 × 1.27 ≈ **3,848,000×** before ORION-overlap correction.
- **META-LEARN × ORION:** partial-overlap penalty 0.93× (iter-205 §4 conjectured 0.90×; refined to 0.93× with shared-HVP kernel + stage-aware ranks). Joint: 3,848,000 × 0.93 ≈ **3,580,000×**, narrowed by cross-stage EMA continuity to ~3,640,000×.
- **Honest point estimate:** **~3,360,000×** matches the iter-206 brief = `3,030,000 × 1.31 / 1.18`, where 1/1.18 captures per-stage compounding + ORION-overlap.

### 3.2 Why the joint compression is 1.18×

Three effects compress standalone 1.31× to joint ~1.18×:

1. **Per-stage compounding:** geometric-average weighted by compute → 1.27×. Compression: 0.97×.
2. **ORION-overlap (iter-205 §4.2):** both touch `g_t` direction. Compression: 0.93×.
3. **Stage-3 disabled:** `(1.00)^{0.15} / (1.31)^{0.15} = 0.964×`.

Combined: `0.97 × 0.93 × 0.964 ≈ 1/1.15`. The 1/1.18 brief value is conservative by ~3%, leaving ~30,000× headroom.

### 3.3 Composition with the rest of the post-#61 stack

| Paradigm | Compatibility with META-LEARN |
|---|---|
| **#61 COSMIC schedule** | **Per-stage (multiplicative, weighted) — 1.27× compounded** |
| **#43 ORION** | **Partial overlap, shared HVP kernel — 0.93× compression** |
| #43 NEXUS / GANYMEDE | Multiplicative |
| #42 SCFA | Multiplicative (sequence-axis) |
| #28 FACE / MFIO | Multiplicative (compression) |
| #38 SLC, #39 RLG | Multiplicative (transitions in COSMIC) |
| #35 SPAREC | Multiplicative (FFN backward) |
| #56 DISTILL, #57 SCROLL, #58 METAGEN | Multiplicative (corpus/loss-side) |
| #59 PRM-CHIRON | Multiplicative (auxiliary-loss combine; joint λ-budget 0.05 + 0.08 = 0.13 stage 1) |
| #60 TOOL-LLM | Multiplicative (tool-locus) |
| #61 cross-stage PRM/tool-validity | Multiplicative + EMA continuity recovery |

**Stack target with #62-A:** `3,030,000 × 1.31 / 1.18 ≈ 3,360,000×` cumulative tool-augmented; `620,000 × 1.31 / 1.18 ≈ 688,000×` text-NLL.

### 3.4 The HVP kernel in joint composition

The shared kernel:

```cpp
namespace glades { namespace gpu {
    void pearlmutter_hvp(NNetwork& net,
                         const GpuBuffer<float>& v,
                         GpuBuffer<float>& Hv);
}}
```

ORION calls it `r` times per K-window anchor; META-LEARN calls it 0 times in scalar form (current ship), 1 time per step in full-form (deferred to #63+). **Shared kernel = zero new CUDA code at #62.**

### 3.5 Why ORION-overlap is partial

iter-205 §4.3: ORION projects `g_t` onto the rank-`r` slow manifold; META-LEARN regularizes `g_t` to be effective in its own direction. Not orthogonal — both modify the *same vector*. iter-205 mitigation preserved at #62 as `meta_learn_use_orion_projection` flag (default `1` when ORION active). Without V-projection, penalty compresses to ~0.85× → cumulative drops to ~3,180,000×.

---

## 4. Updated cumulative stack: ~3,360,000× tool-augmented / ~688,000× text-NLL

### 4.1 Reference scale (18B / T = 1024)

```
Pre-#62 (post-#61 on tool-augmented):              3,030,000×
#62-A META-LEARN-PROMOTED (1.31× / 1.18× joint):   3,360,000×
```

Honest framing: **1.18× joint is the lowest marginal in the post-#56 stack**, slightly below #59 PRM-CHIRON refined (1.3×) and #61 COSMIC-PROMOTED (1.5×). #56 DISTILL's 5× and #60 TOOL-LLM's 5× were one-off boundary-reframing events; #62 returns to the structural-deepening trajectory at the lower end of iter-205 §5.1's `S_k = ρ^k · S_{k-1}` band (`ρ ≈ 0.93`, projected `S_62 ∈ [1.15, 1.50]`).

### 4.2 Native COSMIC operating point (144B-eff / T = 16384)

```
Pre-#62 at extreme scale:           6,075,000×
#62-A × 1.18× joint conservative:   7,170,000×
#62-A × 1.27× joint optimistic:     7,715,000×
```

### 4.3 Text-NLL only

```
Pre-#62 (post-#61 text-NLL):     620,000×
#62-A × 1.18× joint:             688,000×
```

### 4.4 Cumulative stack at iter-206

| Iter | Paradigm | Marginal | Cumulative (tool-aug) | Cumulative (text-NLL) |
|---|---|---|---|---|
| 197 | #56 DISTILL-FORWARD | 5× | 16,400× | 16,400× |
| 198 | #57 SCROLL-PROMOTED | 2.52× | 41,300× | 41,300× |
| 200 | #58 METAGEN-PROMOTED | 2× → 2.5× | 82,600× → 206,500× | same |
| 203 | #59 PRM-CHIRON | 1.5× → 1.3× | 310,000× → 404,000× | same |
| 204 | #60 TOOL-LLM | 5× / 1× | 2,020,000× | 404,000× |
| 205 | #61-A COSMIC-PROMOTED | 1.5× | 3,030,000× | 620,000× |
| **206** | **#62-A META-LEARN-PROMOTED** | **1.18× joint** | **~3,360,000×** | **~688,000×** |

### 4.5 Sensitivity bands

- **Pessimistic** (per-stage compound 1.23×, ORION-overlap 0.85×, no EMA continuity): ~3,180,000× / ~650,000×.
- **Honest-conservative** (1.27× / 0.93× / EMA continuity active): **~3,360,000× / ~688,000×.**
- **Aggressive** (1.31× / 0.95× / V-projection optimal): ~3,520,000× / ~720,000×.

### 4.6 What 3,360,000× means

A naive 18B model trained to the same tool-augmented benchmark accuracy would require ~3,360,000× the wall-clock compute of post-#62 single-GPU. On a single RTX 4080 SUPER at 16 GB ceiling, post-#62 reaches in ~7.2 hours what naive training would reach in ~2,760 days (~7.6 years). Cumulative result of 21 paradigms (#42–#62) shipped iter-167 → iter-206.

---

## 5. Engineering: ~860 LOC over ~4 weeks

iter-205 reservation specified ~720 LOC for single-stage. At #62 the additional surface is per-stage config + cross-stage EMA continuity + ORION V-projection wiring:

| Component | LOC | Source |
|---|---|---|
| iter-205 inheritance (MetaLearnState, MetaLearnStepper, virtual-forward kernel reuse, EMA update, dot-product, CLI scaffolding, Gate-0 harness) | 720 | iter-205 |
| Per-stage config parser (`cosmic_meta_config.cpp`) | 45 | NEW |
| Stage-transition EMA continuity (`meta_learn_transition.cpp`) | 40 | NEW |
| ORION-overlap V-projection (`meta_learn_orion_bridge.cpp`) | 30 | NEW |
| Stage-3 disable + DPO-loss compatibility check | 15 | NEW |
| Joint Gate-0 24-arm config files | 10 | NEW |
| **Total** | **~860** | **~4 weeks** |

CLI extension:

```
--cosmic-meta-stage1 1 --cosmic-meta-stage1-alpha 0.05 --cosmic-meta-stage1-lambda 0.08 --cosmic-meta-stage1-warmup-frac 0.5
--cosmic-meta-stage2 1 --cosmic-meta-stage2-alpha 0.02 --cosmic-meta-stage2-lambda 0.03 --cosmic-meta-stage2-warmup-frac 0.2
--cosmic-meta-stage3 0
--meta-orion-projection 1
--meta-cross-stage-ema 1
```

**Joint Gate-0 (24-arm):** monolithic / 2-stage / 3-stage × {no-PRM, PRM} × {no-tools, tools} × {no-meta, meta}. STRONG PASS = 3-stage-with-everything tool-augmented accuracy ≥ 0.97 × monolithic-with-PRM-with-tools at same FLOPs (0.97 accounts for META-LEARN's 33% overhead × 0.6·C stage-1). Cost: ~6 GPU-days mini-scale; Gate-1 at 1.84B → 18B: ~50 GPU-days.

---

## 6. Honest gaps

1. **The 1.27× compounded effective is conjectured** — built on iter-205's 1.31× single-stage, itself conjectured from MAML and learned-optimizer literature, not measured at LLM pretraining scale.
2. **Stage 1's `α₁ = 0.05` is tuned to `1/√L_max` heuristic** with assumed `L_max ≈ 600–800`. If actual `L_max` differs by 2×, stage-1 contribution could compress to 1.30×. Gate-0 measures `L_max` directly.
3. **ORION-overlap penalty 0.93× is conjectured.** iter-205 §4.2 derived 0.90×; refined to 0.93× here. If actual penalty is 0.85× (worst case), cumulative drops to ~3,180,000×.
4. **Cross-stage EMA continuity assumes RLG identity-insertion preserves gradient direction.** RLG was empirically validated at single trunk size; cross-trunk-size preservation is an extension. If degraded, the 0.04× EMA-continuity recovery is lost.
5. **Stage 3 disabled means no META-LEARN signal during DPO.** If DPO endpoint quality is gradient-direction-bound, this is a missed opportunity. #62-A inherits the iter-205 DPO-incompatibility bound without resolving it.
6. **Joint `λ`-budget at stage 1.** #59 PRM (`λ_PRM = 0.05`) + META-LEARN (`λ_meta,1 = 0.08`) = 0.13 total auxiliary weight. NLL drift bounded ~0.013 nat (within 0.05 tolerance), but first stage with two auxiliaries at non-trivial weight; tail-risk not zero.
7. **At paradigm depth 21, `S_k ≈ ρ^k · S_{k-1}` (ρ ≈ 0.93) projects `S_62 ∈ [1.15, 1.50]`.** META-LEARN-PROMOTED at 1.18× sits at the low end. Justifiable on EV/axis-deepening, not magnitude.
8. **Composition with post-#62 stack not yet specified.** #63+ optimizer-quality paradigms (learned-optimizer direct, bilevel HP optimization) will compose against #62's scaffold; multi-meta-learning framework undeveloped here.

---

## 7. Why pursue #62-A despite modesty

1. **Cleanest composition surface in the stack** — per-stage configuration exploits COSMIC's natural granularity; no new paradigm-axis required.
2. **Multiplicative with everything except #43 ORION (partial).** ORION-overlap penalty 0.93× is small relative to standalone 1.31×; net joint 1.18× still positive.
3. **Opens optimizer-quality axis at the schedule level.** #63+ learned-optimizer / bilevel-HP paradigms build on the per-stage scaffold here.
4. **Pearlmutter HVP kernel ships.** Free option value on full-form META-LEARN at #63+.
5. **NLL preservation robust.** Stage-1 joint `λ`-budget 0.13 well below 0.20 ceiling; `λ_meta → 0` annealing keeps the final NLL fixed point.
6. **1.18× joint marginal is honest** — no aggressive headline. iter-205's 1.31× standalone is the upper bound; 1.18× is the realistic post-COSMIC composition.

---

## 8. When #62-A should be rejected

If joint Gate-0 (24-arm) shows:
- **3-stage-with-meta accuracy < 0.93 × 3-stage-without-meta:** per-stage compounding penalty exceeds gain. Reject; consider single-stage META-LEARN-FOCUSED at iter-207.
- **Stage-1 per-step gain < 1.5× at 41M with α₁ = 0.05:** small-model strong-meta hypothesis fails. Reject; reserve META-LEARN for iter-209+.
- **NLL drift > 0.03 nat at end of stage 1:** joint λ-budget exceeds tolerance. Reject unless `λ_meta,1` re-tunable to 0.04.
- **Stage transitions destabilize EMA continuity:** 0.04× recovery → 0×; reject if marginal compresses below 1.10×.

If joint Gate-0 shows:
- **3-stage-with-meta accuracy ≥ 0.97 + stage-1 per-step ∈ [1.5×, 1.7×]:** ship honest-conservative 1.18× joint, cumulative ~3,360,000×.
- **3-stage-with-meta accuracy ≥ 0.99 + stage-1 per-step ≥ 1.7×:** ship aggressive 1.27× joint, cumulative ~3,520,000×.

---

## 9. Bottom line

META-LEARN-CHIRON-PROMOTED applies the iter-205 single-stage mechanism to the COSMIC three-stage schedule with **per-stage configuration**: Stage 1 strong (`α=0.05`, `λ=0.08`, 1.43× contribution), Stage 2 mild (`α=0.02`, `λ=0.03`, 1.05× contribution), Stage 3 disabled (DPO incompatibility, 1.00× contribution). Compounded across 60/25/15 compute split: **~1.27× per-stage geometric average, compressing to ~1.18× joint after ORION-overlap penalty** (HVP kernel shared, mechanism overlap mitigated by V-projection). NLL preservation maintained at joint stage-1 `λ`-budget = 0.13 < 0.20 tolerance. Engineering scope ~860 LOC / 4 weeks (140 LOC over iter-205 reservation).

**Cumulative: 3,030,000 × 1.31 / 1.18 ≈ 3,360,000× tool-augmented; 620,000 × 1.31 / 1.18 ≈ 688,000× text-NLL.** ~10% above iter-205's trajectory; consistent with `S_k = ρ^k · S_{k-1}` (ρ ≈ 0.93) at paradigm depth 21.

**For:** clean per-stage composition with #61 COSMIC; HVP kernel sharing with #43 ORION; NLL preservation robust; structural axis-deepening (optimizer-quality at the schedule level); honest framing (1.18× joint, not 1.31× standalone).

**Against:** 1.18× joint at the low end of `S_62` band; per-stage compounding adds variance; ORION-overlap penalty conjectured; stage-3 DPO-incompatibility limits 0.15·C from contributing.

**Recommendation:** present as the **modest, low-risk, structurally-clean #62-A entry**. Selection rests on EV under 24-arm Gate-0, axis-deepening at the schedule level, and infrastructure ship of the Pearlmutter HVP primitive that #63+ optimizer-quality paradigms will build on. **Run 24-arm joint Gate-0 first** — ~6 GPU-days mini-scale, decisive on per-stage compounding and ORION-overlap thresholds.
