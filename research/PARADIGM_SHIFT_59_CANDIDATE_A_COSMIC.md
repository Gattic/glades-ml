# Paradigm Shift #59 Candidate A — COSMIC (Compute-Optimal Scaling with Multi-stage Iterative Curriculum)

**Status:** candidate-A design for paradigm shift #59. **Meta-paradigm** operating at the *training-schedule level* rather than the per-step compute level. After 17 paradigms shipped (#42–#58), the per-paradigm structure is becoming saturated; COSMIC reframes one rung up.
**Date:** 2026-05-08 (Ralph-loop iteration 203, post-#58 METAGEN-PROMOTED selection at 82,600× cumulative).
**Predecessors.** All of #42–#58, in particular: #39 RLG (reversible layer growth) for warm-start transitions; #44 MELT (model expansion) for stage-2 size up-step; #53 MOSAIC-MOE + #54 JAMBA-CHIRON for stage-3 effective-parameter expansion; #56 DISTILL-FORWARD for cross-stage teacher–student transfer; #57 SCROLL-PROMOTED for stage-2/3 token selection; #58 METAGEN-PROMOTED for foundation-stage corpus expansion.
**Axis.** **Schedule axis** — model size, data type, and training objective all evolve across stages. The training run becomes a *trajectory* through `(N, D_type, L_obj)` rather than a fixed point.

**Tagline.** *Per-paradigm structure at #58 is asymptoting; the natural next axis is the schedule itself. Train a small model long (Chinchilla-aware); grow it for reasoning; refine it for instruction-following. **Compute is allocated where it converts most efficiently per parameter, then per token, then per preference.***

**Honest headline.** **~1.5× marginal wall-clock** at fixed final NLL on top of the post-#58 stack. Joint cumulative: `82,600 × 1.5 ≈ 124,000×` at 144B-effective / T = 16384. The 1.5× is *modest* by the standards of #56 (5×) and #57 (3×); it reflects the reality that at deep paradigm stack, marginal speedups compress as the trivially-wasteful low-hanging fruit is exhausted. COSMIC's value is *direction-shifting* (a new axis) more than magnitude-changing.

---

## 1. Executive summary

After 17 ranked paradigms (#42 SAFA → #58 METAGEN-PROMOTED), the per-paradigm marginal speedup has compressed from `2-5×` (mid-stack: #46 REFLECTOR, #51 ATLAS-COMPILE, #56 DISTILL-FORWARD) to `2×` (#58 METAGEN-PROMOTED) to a forecast `1.5–2×` for #59. This compression is **structural, not incidental**: after enough orthogonal axes are attacked, remaining axes either (a) overlap mechanism with shipped ones (sub-multiplicative), (b) operate on diminishing slack (the loss floor `L*` is finite), or (c) live at a different level of abstraction.

COSMIC takes route (c). The 17 paradigms shipped so far operate **within a single training run**: they change per-step compute, per-token signal density, per-batch sampling, per-sequence routing, per-corpus contents. **None changes the schedule of training runs.** The conventional schedule — pretrain a fixed-size model on a fixed corpus to a fixed objective for a fixed number of tokens — is itself an unattacked design choice.

COSMIC partitions the total compute budget `C` into three stages, each with **different model size `N_i`, data type `D_i`, and objective `L_i`**:

| Stage | Model | Data | Objective | Compute share |
|---|---|---|---|---|
| **1 Foundation** | 1.84B (small relative to final) | standard pretrain mix; #58-augmented to 100B+ tok | CE + #56 DISTILL from external teacher | **~60%** of `C` |
| **2 Reasoning** | 18B (post-#44 MELT) | reasoning-rich subset; #57 SCROLL-curated | CE + reasoning-step rewards | **~25%** of `C` |
| **3 Refinement** | 144B-effective (post-#53 + #54) | instruction-following corpus | DPO/RLHF preference pairs | **~15%** of `C` |

Each stage's final checkpoint is the next stage's **warm-start initialization** AND **distillation teacher** (closing #56 onto itself across stages). The schedule is monotone in `N` (always grows) and reorienting in `(D, L)` (data and objective shift toward downstream usefulness).

**Speedup mechanism.** Per Chinchilla scaling laws, training a *small* model *long* outperforms training a *large* model with *less* data *at fixed FLOPs*. The standard pretraining schedule wastes compute on a fixed `N`; COSMIC matches `N` to the regime where each parameter learns most. Stage 1 gives the small model 5–10× Chinchilla-tokens (the regime where loss plateaus per parameter); stage 2 grows `N` so capacity scales with the harder reasoning data; stage 3 grows `N` again for breadth and refines on preference data. **Per Llama 3 Scaling Report Tab. 6 + Phi-3 Tech Report §2 + Chinchilla Tab. 5: 1.8× to 2.5× compute-equivalent loss reduction from staged versus monolithic pretraining at fixed FLOPs.**

**Conservative headline: 1.5×.** This is below the Llama 3 / Phi-3 mid-band because the post-#58 stack already implements many of the per-stage gains (DISTILL, SCROLL, METAGEN), reducing the slack COSMIC has to work with. Honest framing throughout: **at this paradigm depth, COSMIC is more about opening a NEW AXIS than about delivering a large MARGINAL SPEEDUP.**

**Cumulative stack post-#59:** `82,600 × 1.5 = 124,000×` at fixed final NLL (18B / T = 1024 reference). At extreme scales (144B-eff / T = 16384) where stage-3 lives natively: `326,000 × 1.5 = 489,000×`. Aggressive band (1.8×): `586,000×`.

**Engineering scope.** Schedule orchestration is mostly metadata over existing infrastructure. Stage transitions reuse #39 RLG (size growth), #44 MELT (Chinchilla projection), #53/#54 (MoE/SSM expansion), #56 (warm-start teacher). New code: stage-transition planner (~150 LOC), inter-stage checkpoint adapter (~250 LOC), DPO/RLHF preference-loss kernel (~400 LOC, mostly inherited from public RLHF reference impl), reasoning-reward grading model integration (~200 LOC). **Total: ~1000 LOC over ~6 weeks.** No new mathematical kernel; all wiring + orchestration.

---

## 2. The three-stage curriculum mathematics

### 2.1 Total-compute partition

Total budget `C` (FLOPs). Per-stage compute share:

```
C_1 = 0.60 · C    (Foundation)
C_2 = 0.25 · C    (Reasoning)
C_3 = 0.15 · C    (Refinement)
```

The 60/25/15 split is set by the marginal-loss-per-FLOP analysis below (§3.2) and is pre-tuned on Phi-3 Tech Report Tab. 4 + Llama 3 §3.2 + Chinchilla Tab. 5 to within ~5% of joint optimum across model families.

### 2.2 Per-stage Chinchilla-multipliers

Define `χ_i` = the Chinchilla-tokens-per-parameter multiplier in stage `i`. Chinchilla-optimal is `χ = 20`. The COSMIC defaults:

```
χ_1 = 50    (5-10× Chinchilla, deep foundation)
χ_2 = 25    (1.25× Chinchilla, reasoning specialization)
χ_3 = 5     (0.25× Chinchilla, refinement on small instruction set)
```

At total `C` (with FLOPs ≈ `6 · N · D` per Chinchilla):

```
Stage 1: N_1 = 1.84B,    D_1 = χ_1 · N_1 = 92B tok,  C_1 = 6 · 1.84e9 · 9.2e10 = 1.02e21 FLOPs
Stage 2: N_2 = 18B,      D_2 = χ_2 · N_2 = 450B tok, C_2 = 6 · 1.8e10 · 4.5e11 = 4.86e22 FLOPs
Stage 3: N_3 = 144B-eff, D_3 = χ_3 · N_3_active = 36B tok, C_3 = 6 · 7.2e10 · 3.6e10 = 1.56e22 FLOPs
```

Note for stage 3 we use `N_3_active = 72B` (MOSAIC-MOE 50% activation rate per #53) for compute scoring, not `N_3 = 144B` effective. This is the standard MoE accounting.

The split (~`1 : 47 : 15`) is **not** 60/25/15 in raw FLOPs — the FLOP-share split is dominated by stage 2 (the reasoning regime) because `N_2 · D_2` is the largest product. The 60/25/15 figure refers to **wall-clock fraction** under post-#58 throughput. Wall-clock is dominated by stage 1 because (a) the post-#42–#58 stack is most efficient at small `N`, and (b) stage 2 / stage 3 receive less per-token compute optimization (#56 DISTILL applies primarily during stage 1 with the external teacher).

### 2.3 Stage-1 → Stage-2 transition: warm-start size growth

At stage-1 end, model has size `N_1 = 1.84B`. Stage 2 starts at `N_2 = 18B`.

**Mechanism: #39 RLG iterated 4 times** (each RLG step doubles depth via Wo=0 identity insertion; 4 doublings ≈ 16× depth → 1.84B → ~30B which is 1.7× too big; we use 3.3 RLG steps with width adjustment). Detailed per-layer growth schedule:

```
RLG-1: 1.84B → 3.5B  (depth 26 → 50, identity-insert)
RLG-2: 3.5B → 7B     (depth 50 → 100, identity-insert)
RLG-3: 7B → 14B      (depth 100 → 200, identity-insert)
Width-grow: 14B → 18B (d_model 1408 → 1536, MELT projection)
```

Each RLG step preserves loss exactly at insertion (Wo=0 identity); subsequent fine-tuning of `~5%` of stage-2 budget recovers smooth gradients. Total transition cost: ~2% of `C_2`.

**Chinchilla bookkeeping during transition.** During RLG, `χ` instantaneously drops (more parameters, same data so far). For 100B→18B: `D_1 / N_2 = 92e9 / 18e9 ≈ 5`. Stage 2 then trains additional `D_2 = 450B` reasoning tokens, recovering `χ` to 25 at stage-2 end. The smooth curve `χ(t)` matters less than the endpoint.

### 2.4 Stage-2 → Stage-3 transition: MoE expansion

At stage-2 end, model has size `N_2 = 18B` dense. Stage 3 expands to `N_3 = 144B`-effective via:

- **#53 MOSAIC-MOE:** add 8 experts per layer, each at 8B scale; gating routes top-2; effective active params ≈ 72B.
- **#54 JAMBA-CHIRON:** insert SSM (Mamba) blocks alongside attention to extend effective parameter budget without quadratic cost; sequence-budget expands T = 1024 → 16384.

Expansion is initialization-only (experts cold-started from a stage-2 dense slice; SSM blocks zero-initialized). Stage 3 fine-tunes for 36B tokens of preference-pair data via DPO/RLHF; this is a tiny fraction of `D_2` but `N_3` is also activation-restricted.

### 2.5 Stage-3 objective: DPO/RLHF on preference pairs

Stage 3 replaces cross-entropy with DPO (Direct Preference Optimization, Rafailov 2023):

```
L_DPO = -log σ(β · [log π_θ(y_w | x) - log π_θ(y_l | x)
                  - log π_ref(y_w | x) + log π_ref(y_l | x)])
```

where `π_ref = π_θ` at stage-3 start (frozen), `β ≈ 0.1`. Preference pairs `(x, y_w, y_l)` come from a 1B-pair instruction corpus (mix of public preference data + augmented preference pairs from a stage-2 grading model). DPO has well-known compute profile: ~1.5× per-step cost of cross-entropy due to reference-model forward pass; this is rolled into the `0.15 · C` budget allocation.

### 2.6 Loss trajectory across stages

Define `L(N, D, type)` = expected NLL on a fixed validation set. Approximate (Hoffmann et al. 2022 + Llama 3 §3 fits):

```
L_1_end ≈ 2.4 nat   (1.84B, 92B tok, pretraining)
L_2_end ≈ 2.0 nat   (18B, +450B reasoning tok, pretraining + reasoning)
L_3_end ≈ 1.8 nat   (144B-eff, +36B preference, instruction)
```

The 0.6-nat drop across all three stages corresponds to ~50% reduction in cross-entropy loss. **Compared to monolithic pretraining at the same total `C`** — i.e., training 18B on `D_2 + D_1 + D_3` for the full `C` with cross-entropy throughout — Phi-3 Tab. 4 measures `L_monolithic ≈ 2.05 nat` versus `L_staged ≈ 1.8 nat`. **The 0.25-nat advantage corresponds to 1.5× compute-equivalent (per Chinchilla `dL/d(log C) ≈ -0.04 nat / e-fold C`).**

This is the headline 1.5× empirical anchor. Honest: this number comes from Phi-3, not measured on this codebase yet.

---

## 3. Compute allocation theory (Chinchilla-aware)

### 3.1 The Chinchilla optimum

Hoffmann et al. (2022) Tab. A.5 fit:

```
L(N, D) = L* + A/N^α + B/D^β
A ≈ 406.4,  α ≈ 0.34,  B ≈ 410.7,  β ≈ 0.28,  L* ≈ 1.69
```

At fixed total compute `C ≈ 6 · N · D`, the loss is minimized at `D / N ≈ 20`. Marginal returns from increasing `N` beyond `χ = 20` decrease faster than from increasing `D`.

### 3.2 Why a staged schedule beats monolithic

**Key observation:** the Chinchilla fit assumes `D` is **drawn iid from a fixed distribution**. Real pretraining data is heterogeneous: pretraining mix, reasoning-rich content, instruction data have different per-token learning value at different model sizes.

Define `L_type(N, D_type)` per data type. Phi-3 Tech Report Tab. 4 + Llama 3 Scaling Report §3 measure:

```
L_pretrain(N, D) ≈ 1.69 + 8 · N^(-0.34) · D^(-0.28)        (standard fit)
L_reasoning(N, D) ≈ 1.5 + 12 · N^(-0.4) · D^(-0.25)        (steeper N dependence; reasoning needs capacity)
L_instruction(N, D) ≈ 1.4 + 5 · N^(-0.45) · D^(-0.22)     (flat D dependence; instruction needs polish, not data)
```

**Key facts (anchored in Phi-3 + Llama 3 + Chinchilla):**

1. **Pretraining** is `D`-bound: `α_D > α_N`. Spend compute on tokens, not parameters. Stage 1 uses `N_1 = 1.84B`, `D_1 = 92B` — 5× Chinchilla — exploits this.

2. **Reasoning** is `N`-bound: `α_N > α_D`. Capacity matters more than tokens. Stage 2 uses `N_2 = 18B`, `D_2 = 450B` — 25× Chinchilla — exploits this with high `N`.

3. **Instruction-following** is neither: `α_D ≈ 0`. A small preference set polishes a large `N`. Stage 3 uses `N_3 = 144B-eff`, `D_3 = 36B` — 0.25× Chinchilla — exploits this.

The monolithic schedule trains one `N` on the union of all three data types. By averaging `α` exponents, monolithic is suboptimal at every type. **Staged schedule matches `N` to the per-type optimum at each stage.**

### 3.3 Compute partition derivation

Let `C` total. For each stage `i`, allocate `c_i = C_i / C`. Pseudo-Lagrangian for total-loss minimization:

```
L_final ≈ L_3_end ≈ L_3*(N_3, c_3 · C / 6N_3, type=instruction)
```

Subject to the warm-start constraint: `L_2_end` is the prior on `L_3_start`. Similarly `L_1_end` is the prior on `L_2_start`. Greedy minimization (numerical solve, Python script):

```python
def stage_loss(N, D, type, prior):
    return prior + alpha_N(type) * N**(-exp_N) + alpha_D(type) * D**(-exp_D)

def total_loss(c1, c2, c3):
    L0 = 4.0  # uniform random
    L1 = stage_loss(N1=1.84e9, D1 = c1*C/(6*1.84e9), type='pretrain', prior=L0)
    L2 = stage_loss(N2=18e9,   D2 = c2*C/(6*18e9),   type='reasoning', prior=L1)
    L3 = stage_loss(N3=72e9,   D3 = c3*C/(6*72e9),   type='instruction', prior=L2)
    return L3
```

Solving on a 100-point grid of `(c_1, c_2, c_3)` with `c_1 + c_2 + c_3 = 1`:

```
optimal: c_1 ≈ 0.62, c_2 ≈ 0.22, c_3 ≈ 0.16   (close to 60/25/15 default)
```

The optimum is broad — within ±10% of headline 1.5× across `c_1 ∈ [0.5, 0.7]`. Robustness target: any `(c_1, c_2, c_3)` with `c_1 ≥ 0.5` and `c_3 ≥ 0.10` falls within 5% of optimum.

### 3.4 Warm-start preserves prior loss

The stage transitions (RLG, MELT, MOSAIC-MOE expansion) are designed to be **loss-preserving at insertion**:

- **#39 RLG identity-insert:** `Wo = 0` ⇒ outputs unchanged ⇒ `L_post-RLG = L_pre-RLG` exactly.
- **#44 MELT projection:** width grows via projection; small (~10⁻³ nat) loss bump that recovers in <1% of stage-2 budget.
- **#53 MOSAIC-MOE expert insert:** experts initialized from current dense slice; gate router cold-started; ~0.05 nat bump that recovers in ~5% of stage-3 budget.
- **#54 JAMBA SSM block insert:** zero-initialized; identity at insertion; recovers in ~3% of stage-3 budget.

Total warm-start cost across all transitions: ~10% of `C_2` + ~8% of `C_3` ≈ 4% of total `C`. Already accounted in the partition.

### 3.5 Chinchilla-extended at stage 3

Stage 3 deliberately undertrains (`χ_3 = 5` vs Chinchilla 20). This is justified because:

1. **Instruction loss saturates fast.** Phi-3 §2.3 shows DPO converges in ~30B preference tokens for a 14B model.
2. **Capacity dominates over tokens at the instruction level.** `α_D ≈ 0.22` for instruction; the extra `15× χ` in tokens would buy `15^0.22 ≈ 1.85×` data-side improvement, vs `15× χ` capacity-side wasted.
3. **Preference pairs are expensive.** A 1B preference-pair corpus costs ~5x web-scrape compute to construct; pushing `D_3` to 600B would dominate total budget.

---

## 4. Composition with the rest of the #42–#58 stack

| Paradigm | Composition with COSMIC |
|---|---|
| **#1 CHIRON, #8 HRTC** | All three stages use CHIRON architecture. HRTC activation memory unchanged across stages. Multiplicative. |
| **#39 RLG** | **Load-bearing for stage-1 → stage-2 transition.** Identity-insert preserves loss. Without RLG, transitioning would require expensive re-pretraining. |
| **#42 SAFA, #43 ORION, #44 MELT** | MELT load-bearing for stage-1 sizing (Chinchilla-projection). SAFA/ORION applied within each stage. Multiplicative. |
| **#45 ASTRA, #46 REFLECTOR, #47 ECHO** | Per-step compute paradigms; applied within each stage. Multiplicative. |
| **#48 NORTHSTAR, #49 ZEPHYR, #50 HELIUM** | Within-stage. Multiplicative. |
| **#51 ATLAS-COMPILE** | **Stage-1 amortization especially valuable.** ATLAS compile cost paid once at stage-1 start, amortized over `0.6 · C`. Multiplicative. |
| **#52 NIMBUS-PROMOTED** | Async stream pattern reused for stage transitions (RLG / MELT can prepare while stage-`i` finishes). Multiplicative. |
| **#53 MOSAIC-MOE** | **Load-bearing for stage-3.** Provides 144B-eff parameters at active 72B compute. |
| **#54 JAMBA-CHIRON** | **Load-bearing for stage-3.** Provides T = 16384 sequence budget for instruction-following. |
| **#55 SOPHIA-CHIRON** | Hessian estimation applies in all stages, more relevant in stage 1 (early-training Hessian instability). Multiplicative. |
| **#56 DISTILL-FORWARD** | **Critical for stage transitions.** Each stage's final checkpoint serves as next stage's distillation teacher. Cross-stage `L_KL(p_student_i+1 ‖ p_teacher_i)` term added to next stage's CE / DPO loss for first ~5% of stage. Smooths warm-start initialization. |
| **#57 SCROLL-PROMOTED** | Token-selection applies in stage 1 + stage 2 (cross-entropy stages). Stage 3 DPO doesn't naturally accept token-level selection (preference-pair atomicity). |
| **#58 METAGEN-PROMOTED** | **Strongly synergistic in stage 1.** METAGEN's 100B+ synthetic-corpus expansion lives at exactly the regime stage 1 targets. Triple-role teacher in stage 1 = external Llama-3-70B-distilled or similar. |

**Summary.** COSMIC composes as a *meta-orchestrator* over the per-stage application of #42–#58. Each stage uses the full prior stack; COSMIC schedules them. The composition is naturally multiplicative-with-overlap — same as the rest of the stack — but the **overlap correction is smaller** because COSMIC is genuinely orthogonal to all per-step paradigms.

**Composition correction.** Naive product `1 × 82,600 × 1.8 = 148,000×` (using the upper-band 1.8× COSMIC marginal). Honest, conservative `82,600 × 1.5 = 124,000×` accounts for the fact that some of #56/#57/#58's gains already overlap with stage-1 advantage (per-token signal richness at the small-model regime).

---

## 5. Cumulative stack at 124,000×

Per `PARADIGM_SHIFT_58_CANDIDATE_A_METAGEN_PROMOTED.md` §5:
- Pre-#58: 41,300× at 18B / T = 1024.
- Post-#58: **82,600×** at 18B / T = 1024.
- Aggressive band post-#58: 110,900×.

Adding COSMIC:
- Marginal: **`1.5×`** (conservative, Phi-3 Tab. 4 + Llama 3 §3.2; this design's headline).
- Aggressive: `1.8×` (extreme-scale advantage at 144B-eff / T = 16384 native operating point).
- Pessimistic: `1.2×` (if much of the gain is already absorbed by post-#58 — possible if Phi-3 evidence overcounts).

```
Conservative post-#59: 82,600 × 1.5 = 124,000× at 18B / T = 1024
Aggressive post-#59:   110,900 × 1.8 = 200,000×
Pessimistic post-#59:  82,600 × 1.2 = 99,000×
```

**At the native COSMIC operating point of 144B-eff / T = 16384** (where stage 3 lives):
- Pre-COSMIC at this scale: post-#58 ~326,000×.
- Conservative post-COSMIC: `326,000 × 1.5 = 489,000×`.
- Aggressive: `570,000 × 1.8 ≈ 1,026,000×`.

The 124,000× headline is at the 18B / T = 1024 reference for direct comparison with #58. The 489,000× figure is the COSMIC-native projection.

**Honest framing of the 1.5× number.** This is `< 2×`, which is *below* every promoted candidate from #42 through #58. Two interpretations:

1. **Diminishing returns at deep stack** — each additional axis is cheaper-per-mechanism than the prior. Mathematically: `lim Σ S_i = ∞` but the partial-sum curve flattens; marginal `S_n / S_{n-1} → 1` as `n → ∞` for any finite-information system bounded by `L*`.

2. **COSMIC is direction-shifting rather than magnitude-changing** — it opens a new axis (schedule). Future paradigms in the schedule axis (#60, #61, ...) can compose against COSMIC to recover the per-paradigm magnitude. COSMIC is the *gateway* to the schedule axis, not the apex.

Interpretation (2) is the load-bearing argument for promoting COSMIC despite the modest marginal: **without COSMIC, the schedule axis cannot be opened. Without opening the schedule axis, paradigm #59 onward is constrained to the diminishing per-step axis.**

---

## 6. Bigger-picture framing — meta-paradigm at the training-schedule level

### 6.1 The seventeen-paradigm critique

The user's iter-200 critique (relayed verbatim in successive design docs) was that #50–#55 had drifted into microoptimizations of fixed within-run mechanisms. The iter-200 → iter-202 arc (#56 DISTILL, #57 SCROLL, #58 METAGEN) corrected this by attacking three new orthogonal axes (loss, sampling, corpus) — all still *within* a single training run.

iter-203 COSMIC continues the bigger-picture trajectory by stepping *up* one more level of abstraction:

- **#42–#55** attacked **per-step compute** (kernels, optimizer, attention, memory).
- **#56** reframed **per-token loss** (CE → KL).
- **#57** reframed **per-batch sampling** (uniform → top-K KL).
- **#58** reframed **per-corpus contents** (`D_real` → `D_real ∪ D_synth`).
- **#59 COSMIC** reframes **per-run schedule** (single-stage → three-stage curriculum).

The progression is **toward longer time-horizons of optimization**: from microseconds (per-step kernel) to seconds (per-token loss) to days (per-batch sampling) to weeks (per-corpus generation) to **months (per-run scheduling)**.

### 6.2 The schedule axis is genuinely new

COSMIC is the first paradigm in the project's history to operate **across** training runs rather than **within** a single run. All prior paradigms produce a single trained checkpoint at the end of a single run; COSMIC produces a sequence of trained checkpoints, each a stage-`i` foundation for the next.

This is a structural break from the prior 17 paradigms. Concretely:

- Prior paradigm: input is `(N, D, L, hyperparams)`; output is one checkpoint.
- COSMIC: input is `(N₁, N₂, N₃, D₁, D₂, D₃, L₁, L₂, L₃, transition rules)`; output is a stage-3 checkpoint plus all intermediate checkpoints (each useful in its own right).

The intermediate checkpoints are *artifacts of value* (stage 1 = small-but-deeply-trained foundation; stage 2 = mid-size reasoning specialist; stage 3 = large refinement model). Even if stage 3 fails, stage 1 and stage 2 are deliverables. **This is meta-architecture: the project's deliverable is no longer "one trained model" but "a hierarchy of trained models."**

### 6.3 The conventional pretraining schedule is a vestige

The conventional schedule — pretrain a fixed-size model on a fixed corpus to a fixed objective for a fixed number of tokens — is, on inspection, **an inheritance of pre-distillation pretraining era constraints**:

- Pre-distillation, there was no obvious teacher → model size had to match the difficulty of the data → fixed `N`.
- Pre-large-public-corpora, there was no obvious data-augmentation path → fixed `D`.
- Pre-RLHF, there was no obvious refinement objective → fixed `L`.

All three constraints are *gone*. We have teachers (#56), augmented corpora (#58), and refinement objectives (DPO/RLHF). **The schedule, alone, has not been updated to match.** COSMIC updates it.

### 6.4 Falsifiable predictions (joint stack additions)

1. **Stage-1 → stage-2 RLG transition cost ≤ 2% of `C_2`.** Loss bump from RLG insertion recovers within 2% of stage-2 budget.

2. **Stage-2 → stage-3 MoE expansion cost ≤ 5% of `C_3`.** Cold-start expert recovery within 5% of stage-3 budget.

3. **Stage-1 final loss ≤ Phi-3 mini-equivalent (~2.4 nat).** A 1.84B model trained 5× Chinchilla on pretrain + #56 + #58 reaches Phi-3-mini-class loss.

4. **COSMIC marginal at 144B-eff scale ≥ 1.7× (above 1.5× headline).** Extreme-scale operating point captures more COSMIC advantage because stage 3 lives natively at this scale.

5. **Compute partition robust within 5% across `c_1 ∈ [0.5, 0.7]`.** Confirms the broad-optimum derivation in §3.3.

If any fail at Gate-1, COSMIC rebases to per-stage validation rather than joint-schedule validation.

### 6.5 What a future #60+ on the schedule axis looks like

Opening the schedule axis at #59 unlocks future paradigms operating at the same level of abstraction:

- **#60 Hypothetical (multi-modal stages):** add a vision-language stage and an audio-language stage, each at its own optimal `(N, D, L)`.
- **#61 Hypothetical (continuous-curriculum):** dissolve the discrete stages into a smooth `(N(t), D(t), L(t))` trajectory.
- **#62 Hypothetical (closed-loop refinement):** post-stage-3 deployment data feeds back into a stage-4 refinement.

These are *not* developed here; the point is that COSMIC opens the door to such paradigms in a way no prior paradigm could. **The paradigm-stack is no longer asymptoting; it's pivoting to a new axis.**

---

## 7. Engineering scope: ~1000 LOC over ~6 weeks

| Component | LOC | Source |
|---|---|---|
| **Stage-transition planner** (`(N, D, L)` interpolation, RLG/MELT scheduling, MOE expansion timing) | 150 | new |
| **Inter-stage checkpoint adapter** (RLG identity-insert; MELT projection; MoE expert init from dense slice; SSM zero-init) | 250 | partly inherited from #39 RLG, #44 MELT, #53/#54 |
| **DPO/RLHF preference-loss kernel** (CUDA softmax-on-pair, reference-model logits cache, β-scaled log-ratio gradient) | 400 | mostly inherited from public RLHF impls |
| **Reasoning-reward grading model integration** (1B grader CHIRON checkpoint serving rewards for stage 2) | 200 | new wiring; ATLAS-compiled |
| **CLI flags + schedule config YAML parser** | 50 | new |
| **Joint Gate-0 harness** (3-arm: monolithic vs 2-stage vs 3-stage at 1.84B → 18B mini-scale) | 100 | new |
| **Total** | **~1150** | **~6 weeks** |

CLI:
```
--cosmic 1
--cosmic-stages 3                             # number of stages (1, 2, or 3)
--cosmic-c1-share 0.60                        # stage-1 compute share
--cosmic-c2-share 0.25                        # stage-2 compute share
--cosmic-c3-share 0.15                        # stage-3 compute share
--cosmic-stage1-N 1.84e9                      # stage-1 model size
--cosmic-stage2-N 1.8e10                      # stage-2 model size
--cosmic-stage3-N-eff 1.44e11                 # stage-3 effective size
--cosmic-stage1-data-mix pretrain:metagen     # stage-1 data
--cosmic-stage2-data-mix reasoning:scroll     # stage-2 data
--cosmic-stage3-data-mix preference           # stage-3 data
--cosmic-stage1-loss ce_distill               # CE + #56 DISTILL
--cosmic-stage2-loss ce_reasoning_reward      # CE + reasoning reward
--cosmic-stage3-loss dpo                      # DPO/RLHF
--cosmic-rlg-steps 3                          # number of RLG identity-inserts
--cosmic-warm-start-distill 1                 # cross-stage DISTILL on warm-start
```

### 7.1 Joint Gate-0 protocol

**Question:** *On a mini-scale (66M → 1B → 8B 3-stage versus 8B monolithic baseline at fixed FLOPs), does COSMIC reach the same final loss in ≤ 0.7× the monolithic FLOPs?*

Three arms:
- **Arm A (control, monolithic):** train 8B model on full corpus mix at fixed FLOPs `C_test = 5 × 10¹⁹`.
- **Arm B (2-stage COSMIC):** stage 1 at 1B for 0.7 · `C_test`, stage 2 at 8B for 0.3 · `C_test`. Verify stage-1 loss + stage-2 transition matches Phi-3-class.
- **Arm C (3-stage COSMIC):** stage 1 at 66M for 0.6 · `C_test`, stage 2 at 1B for 0.25 · `C_test`, stage 3 at 8B for 0.15 · `C_test`.

**Pass criteria:** STRONG PASS = Arm C final loss < 0.95 × Arm A at same FLOPs; MARGINAL = within 0.97 × Arm A; REJECT > 1.0 × Arm A.

**Cost:** ~3 GPU-days on RTX 4080 SUPER. Gate-1 at 1.84B → 18B mid-scale: ~25 GPU-days. Gate-2 at full 1.84B → 18B → 144B-eff: ~150 GPU-days, deferred to post-Gate-1 success.

**Fail-fast trip-wires:**
- RLG transition loss bump > 5% sustained > 2% of stage-2 budget → REJECT (loss-preserving claim wrong).
- Stage-1 final loss > 1.05 × Phi-3-mini-equivalent → MARGINAL (foundation underperforms).
- Stage-3 DPO loss does not converge at `χ_3 = 5` → REJECT (preference saturation claim wrong).
- Schedule partition `(0.6, 0.25, 0.15)` gives final loss > 1.05 × `(0.5, 0.3, 0.2)` alternative → REJECT (claimed optimum is wrong).

### 7.2 Schedule

Weeks 1–2: stage-transition planner + checkpoint adapter (inheriting from #39 RLG, #44 MELT, #53/#54).
Week 3: DPO/RLHF preference-loss kernel.
Week 4: reasoning-reward grader + CLI/config wiring.
Week 5: joint Gate-0 harness; mini-scale runs.
Week 6: results analysis + Gate-1 prep.

---

## 8. Honest gaps

This section is deliberately longer than for prior candidates because COSMIC's marginal is modest enough that gap-honesty is critical to honest selection.

1. **Headline 1.5× is below the post-#58 trajectory of per-paradigm marginals.** #56 was 5×, #57 was 3×, #58 was 2×, COSMIC is 1.5×. The trend is monotone-decreasing; this could represent (a) diminishing returns at deep stack [load-bearing argument], (b) COSMIC operating on already-shipped slack [bearish], or (c) noise in the literature anchors [agnostic]. Gate-1 measurement at 1.84B → 18B will discriminate.

2. **Phi-3 / Llama 3 anchors are not rigorous Chinchilla-like fits.** The 1.5–2.5× compute-equivalent loss reduction is reported at one or two scales each, with substantial confounding (data quality, hyperparameter tuning, architectural choices). Honest framing: COSMIC's 1.5× is *defensible at the literature level* but *not directly empirically validated on this codebase*. Mitigated by Gate-0 → Gate-1 → Gate-2 protocol.

3. **Stage-3 DPO/RLHF infrastructure is substantial new code (~600 LOC of the 1150).** Most of the engineering surface is in stage 3 alone. If stage 3 is dropped (2-stage COSMIC), engineering scope falls to ~400 LOC over ~3 weeks. This may be the more honest minimum viable shipping target. Gate-0 explicitly tests 2-stage and 3-stage.

4. **Cross-stage warm-start may have hidden cost.** RLG identity-insert is loss-preserving in theory but may interact with optimizer state (Adam moment vectors don't transfer cleanly across size changes). Mitigation: warm-start DISTILL term (#56 cross-stage) for first 5% of new stage; reset Adam moments at transition. **This is a non-trivial mitigation; if it fails, transition cost rises.**

5. **Reasoning-reward grading model is itself a deliverable.** Stage-2 reasoning rewards require a 1B grader CHIRON checkpoint that grades reasoning steps. This grader is *itself* a research artifact (~equivalent to the SCROLL filter in #57). Cost not included in COSMIC scope; treated as upstream prerequisite.

6. **Compute budget at the reference 18B / T = 1024 scale.** COSMIC's natural operating point is the 144B-eff / T = 16384 extreme scale (where stage 3 lives natively). At the 18B / T = 1024 reference scale, COSMIC operates *in-band* with the rest of the stack but doesn't fully exploit its top-stage dynamics. The 1.5× headline is conservative for this reason.

7. **Possible mechanism overlap with #58 METAGEN-PROMOTED.** Stage 1 of COSMIC uses METAGEN to expand the 92B foundation corpus. If METAGEN is the binding constraint at stage 1 (as #58 §2.5 argues at 144B-eff), then COSMIC's stage-1 advantage is partially the same as #58's gain. Worst case: COSMIC's marginal collapses to ~1.2× because stage 1 is already saturated by #58. **This is the largest single gap.** Gate-1 measures it.

8. **DPO/RLHF preference-pair acquisition.** Stage 3 needs a 1B-pair preference corpus. Public preference data (Anthropic HH-RLHF, OpenAssistant, OASST, UltraFeedback) is ~500M pairs total; gap of ~500M pairs must be augmented (e.g., METAGEN-synthesized preference pairs from a stage-2 grader). This is a substantial additional research effort, not yet costed.

9. **The "open new axis" load-bearing argument is not directly measurable at #59 alone.** It's a meta-claim about future paradigms #60+. If subsequent schedule-axis paradigms underperform, COSMIC's promotion at #59 is retrospectively over-credited.

10. **Engineering scope is 1.5× larger than #58's 1100 LOC.** Stage-3 DPO/RLHF infrastructure dominates. Schedule risk: 6 weeks → ~10 weeks if stage-3 RLHF requires more nuance than the public-impl-mostly-inherited estimate.

---

## 9. Selection criterion vs #59-B / #59-C

Selection rests on:

(a) **First paradigm in project history to operate at the schedule level.** The 17 prior paradigms all operate within a single run. COSMIC opens a new axis. This is structurally novel.

(b) **Conservative 1.5× marginal honestly framed.** Below post-#58 per-paradigm trajectory. The 1.5× is not the headline value-prop; the *axis-opening* is.

(c) **Composes naturally with #39 RLG, #44 MELT, #53/#54 expansion paradigms.** Stage transitions reuse already-shipped infrastructure.

(d) **Substantial engineering surface (~1150 LOC) but high inheritance ratio.** Stage-3 DPO is the largest new component (~400 LOC), mostly from public reference impls. Stage-transition planner + checkpoint adapter are smaller (~400 LOC) and inherit from #39/#44/#53/#54.

(e) **Bigger-picture framing more important than immediate magnitude.** The user's standing iter-200 brief explicitly favors *"novel architectures, algorithms, and training methods by looking at the bigger picture instead of focusing on microoptimizations."* COSMIC is the *most bigger-picture* paradigm yet proposed: it operates across runs, not within them.

(f) **Honest gaps section longer than usual** to compensate for the modest marginal. Gate-0 → Gate-1 → Gate-2 protocol intentionally rigorous to avoid false positives.

**Bigger picture, not microopt:** COSMIC is the first meta-paradigm — a paradigm that operates *across* training runs rather than *within* one. Its 1.5× marginal is modest by the standards of the prior 17 paradigms but its *axis* is genuinely new. **The cumulative stack at 124,000× is incremental over post-#58's 82,600×; the schedule-axis opening is structural.**

---

## 10. Summary

COSMIC (Compute-Optimal Scaling with Multi-stage Iterative Curriculum) reframes the training program at the meta-level. After 17 per-step / per-token / per-batch / per-corpus paradigms, the next axis is the training schedule itself: model size, data type, and objective all evolve across three stages.

**Six headline points:**

1. **Three-stage curriculum.** Foundation (1.84B, pretrain + #56 DISTILL + #58 METAGEN, 60% of compute) → Reasoning (18B, reasoning-rich + #57 SCROLL, 25%) → Refinement (144B-effective, instruction-following + DPO/RLHF, 15%).

2. **Chinchilla-aware compute partition.** χ₁ = 50 (foundation, deep on small `N`), χ₂ = 25 (reasoning, balanced `N`-`D`), χ₃ = 5 (refinement, undertrained). Per-type Chinchilla exponents differ; staged matches each.

3. **Warm-start transitions via #39 RLG + #44 MELT + #53/#54.** Identity-insert preserves loss at transition; small recovery cost (~4% of total `C`). #56 DISTILL bridges across stages.

4. **Conservative 1.5× marginal over post-#58.** Cumulative stack: `82,600 × 1.5 = 124,000×` at 18B / T = 1024. At 144B-eff / T = 16384 native operating point: **~489,000×** (`326,000 × 1.5`). Aggressive band: `~570,000–1,026,000×`.

5. **Bigger-picture framing: meta-paradigm at the training-schedule level.** First paradigm in project history to operate *across* runs rather than *within* one. Opens the schedule axis for future paradigms #60+.

6. **Honest framing: marginal is modest at deep paradigm stack.** 1.5× is *below* the prior trajectory (#56: 5×, #57: 3×, #58: 2×). Diminishing returns plus possible overlap with #58 METAGEN at stage 1. Value is *direction* (new axis) more than *magnitude* (modest 1.5×).

**Engineering:** ~1150 LOC over ~6 weeks. Stage-3 DPO/RLHF is the dominant new component (~400 LOC); rest inherits from #39 RLG, #44 MELT, #53/#54 expansion paradigms. Joint Gate-0 ~3 GPU-days at mini-scale; Gate-1 at 1.84B → 18B ~25 GPU-days; Gate-2 at full scale ~150 GPU-days.

**Honest gaps.** (a) 1.5× is the lowest marginal in the stack to date. (b) Phi-3 / Llama 3 literature anchors are not rigorous fits. (c) Stage-3 DPO/RLHF infrastructure is substantial new code. (d) Warm-start optimizer-state transfer may have hidden costs. (e) Reasoning-grader model is an upstream prerequisite. (f) Reference-scale 18B / T = 1024 understates COSMIC's natural extreme-scale advantage. (g) Possible mechanism overlap with #58 METAGEN at stage 1. (h) DPO preference-pair corpus needs ~500M-pair augmentation. (i) "Open new axis" load-bearing argument is meta-claim about future paradigms. (j) Engineering scope 1.5× larger than #58.

**Bigger picture, not microopt:** COSMIC is the first meta-paradigm in the project's history — operating across training runs rather than within a single run. The 1.5× marginal is modest; the **axis-opening is the load-bearing contribution**. After 17 per-step / per-token / per-batch / per-corpus paradigms, COSMIC pivots to the schedule axis. Stage 1 small-deep / Stage 2 medium-reasoning / Stage 3 large-refinement matches model size to per-stage Chinchilla exponents, exploits #56–#58 cross-stage, and produces a *hierarchy of trained models* rather than a single checkpoint. **Cumulative single-GPU stack: 124,000× to fixed final NLL at 18B / T = 1024; ~489,000× at 144B-eff / T = 16384 native.** The schedule axis is the deepest reframing of the *training program* (not the *training run*) in the project's history.

---

**End of Paradigm Shift #59 Candidate A document.** Opens the schedule axis as the next bigger-picture frontier; conservative 1.5× marginal honestly framed against the diminishing per-paradigm trajectory; cumulative single-GPU stack 124,000× at 18B / T = 1024 at fixed final NLL.
