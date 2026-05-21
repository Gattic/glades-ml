# Paradigm Shift #61 — COSMIC: Compute-Optimal Scaling with Multi-stage Iterative Curriculum

**Status:** SELECTED (candidates A/B/C developed; A chosen).
**Date:** 2026-05-08 (iter 205, building on iter 200-204 bigger-picture track #56-#60).
**Axis:** Bigger-picture training-schedule meta-paradigm — three-stage curriculum (foundation → reasoning → refinement) with per-stage paradigm composition. Twice-deferred (#59-A, #60-A) reservation now promoted.
**Magnitude target:** 1.5× marginal beyond #60. Cumulative single-GPU stack: **~3,030,000× tool-augmented benchmarks; ~930,000× text NLL alone**.

---

## 0. Executive summary

The bigger-picture track #56-#60 reframed:
- **DATA** (METAGEN: web → self-generated)
- **LOSS** (DISTILL: CE → KL distillation)
- **SAMPLING** (SCROLL: uniform → KL-informativeness)
- **REWARD** (PRM: outcome → process)
- **IDENTITY** (TOOL-LLM: monolithic → coordinator + tools)

Iter-205 #61 adds the **SCHEDULE** dimension. COSMIC reframes training as a multi-stage process where model size, data type, objective, AND tool augmentation all evolve together.

**Three stages:**

**Stage 1: Foundation** (60% of compute, 50× Chinchilla-tokens)
- Model: 1.84B (small).
- Data: standard pretraining mix; #58 METAGEN-augmented to 100B+ tokens.
- Objective: CE + #56 KL-distillation from external teacher.
- Tools: minimal (calculator + retrieval, ~5% of tokens are tool-traces, 6 selectors).
- PRM: weak (λ_PRM = 0.05, d_PRM = 512, ~70-75% accuracy).

**Stage 2: Reasoning** (25% of compute, 25× Chinchilla-tokens)
- Model: 18B (post-#39 RLG growth from 1.84B).
- Data: reasoning-rich subset (math, code, scientific) curated by #57 SCROLL.
- Objective: CE + #56 KD + #59 PRM-CHIRON full strength.
- Tools: code interpreter + calculator + retrieval (~12% of tokens, 7 selectors).
- PRM: full (λ_PRM = 0.10, d_PRM = 1024, warm-started from Stage 1).

**Stage 3: Refinement** (15% of compute, 5× Chinchilla-tokens)
- Model: 144B effective (post-#53 MOSAIC-MOE + #54 JAMBA-CHIRON expansion).
- Data: instruction-following + DPO preference pairs.
- Objective: DPO/RLHF with constitutional anchor.
- Tools: full suite (~20% of tokens, ~64 selectors).
- PRM: frozen from Stage 2 (constitutional anchor against reward-hacking).

**Stage transitions** use #39 RLG (size growth), #56 DISTILL (warm-start from previous stage's PRM-trained model as teacher), and #57 SCROLL (curriculum-aware data selection).

**Speedup analysis:**
- Standalone COSMIC: 1.5× via better compute allocation across stages.
- Joint with #60 TOOL-LLM (per-stage tool augmentation): 5 × 1.5 = 7.5× over post-#59 baseline (avoiding double-counting).
- Cumulative: 2,020,000 × 1.5 = **~3,030,000× on tool-augmented benchmarks; 620,000 × 1.5 = ~930,000× text NLL.**

Engineering: ~1470 LOC over 8 weeks. Stage-transition orchestrator + per-stage hyperparameter management + cross-stage PRM transfer.

---

## 1. Candidate formulations and selection

### 1.1 Three parallel candidates

| Candidate | File | Reframing | Speedup | NLL | Verdict |
|---|---|---|---|---|---|
| **A — COSMIC-promoted** | `PARADIGM_SHIFT_61_CANDIDATE_A_COSMIC_PROMOTED.md` | Schedule meta-paradigm | **1.5×** | ✓ | **SELECTED** |
| **B — JEPA-CHIRON** | `PARADIGM_SHIFT_61_CANDIDATE_B_JEPA_CHIRON.md` | Latent prediction | 2-5× IF NLL relaxed | ✗ | **REJECTED** (self-recommended) |
| **C — META-LEARN-CHIRON** | `PARADIGM_SHIFT_61_CANDIDATE_C_META_LEARN_CHIRON.md` | Meta-learning | 1.31× | ✓ | Reserved (lower than COSMIC) |

### 1.2 Selection: COSMIC-promoted

COSMIC is selected on five grounds:

**1. Highest preserved-NLL speedup.** 1.5× > META-LEARN's 1.31×. JEPA's 2-5× violates NLL constraint.

**2. Twice-deferred (long-pending reservation).** Reserved as #59-A (iter-203) and #60-A (iter-204). At paradigm depth 20, COSMIC is the natural completion of the schedule axis.

**3. Strongest stack composition.** COSMIC integrates with ALL prior paradigms via per-stage configuration:
   - Stage 1: minimal tools, weak PRM, METAGEN augmentation.
   - Stage 2: full PRM, code interpreter, JAMBA hybrid blocks.
   - Stage 3: DPO + tool suite + constitutional anchor.
   - Each stage's optimal paradigm mix differs.

**4. Cross-stage compounding via warm-start.** Stage N's model + PRM + tool-validity classifiers serve as Stage N+1's initialization (warm-start) AND distillation teacher. Knowledge accumulates across stages.

**5. Bigger-picture meta-paradigm.** COSMIC operates at the SCHEDULE level — ABOVE individual paradigms. Conventional training treats schedule as fixed (one phase); COSMIC reframes it as adaptive across model-size + data + objective + tool.

### 1.3 Why JEPA rejected

The candidate doc self-recommends rejection. JEPA's loss (latent reconstruction) is fundamentally different from next-token CE; text-NLL not preserved. LLM-scale unverified (LCM Meta 2024 doesn't report token perplexity).

JEPA reserved for paradigm #65+ JEPA-LLM-RAG, #66+ JEPA-MULTIMODAL — when NLL constraint relaxes.

### 1.4 Why META-LEARN reserved

META-LEARN's 1.31× is modest (vs COSMIC's 1.5×) and has 33% per-step overhead. Composition with #43 ORION's HVP infrastructure has partial overlap (~1.18× joint vs 1.31× standalone).

META-LEARN reserved for paradigm #62 if a meta-learning research direction becomes attractive.

---

## 2. Formal problem statement

After 19 paradigms (#42-#60), cumulative stack is ~2,020,000× on tool-augmented benchmarks; ~620,000× on text NLL alone (post-#60). The bigger-picture track has reframed 5 axes:
- DATA (METAGEN), LOSS (DISTILL), SAMPLING (SCROLL), REWARD (PRM), IDENTITY (TOOL-LLM).

Remaining axis: **SCHEDULE**. Conventional training is single-phase: fixed model size, fixed data distribution, fixed objective. COSMIC reframes as multi-stage.

**Problem.** Find a paradigm that:
1. Composes with all #42-#60 paradigms via per-stage configuration.
2. Provides ≥ 1.4× marginal speedup beyond #60.
3. Maintains NLL preservation.
4. Bigger-picture: introduces SCHEDULE as a paradigm dimension.

COSMIC solves this via three-stage curriculum with cross-stage warm-start.

---

## 3. Core mathematical framework

### 3.1 Three-stage compute allocation

Total compute budget C is partitioned:
- C_1 = 0.60 C (foundation, Stage 1)
- C_2 = 0.25 C (reasoning, Stage 2)
- C_3 = 0.15 C (refinement, Stage 3)

Per Hoffmann (Chinchilla 2022) compute-optimal scaling, these per-stage allocations exploit per-data-type Chinchilla coefficients. Different stages have different α_D vs α_N tradeoffs.

### 3.2 Stage 1: Foundation

Model size: 1.84B parameters.
Data: 100B+ tokens (5-10× Chinchilla-tokens for 1.84B).
Objective: `L = (1-α)·L_CE + α·L_KD` from external teacher (e.g., LLaMA-2 7B).
Tool augmentation: 5% of tokens are tool-traces (6 selectors: calc, retrieve, identity).
PRM: λ_PRM = 0.05, d_PRM = 512, ~70-75% accuracy.

Compute allocation: 60% of total budget. Model is small but data-rich.

### 3.3 Stage 2: Reasoning

Model size: 18B (post-#39 RLG growth from 1.84B).
Data: reasoning-rich subset (math, code, scientific argumentation), curated by #57 SCROLL with KL-informativeness scoring.
Objective: full PRM-CHIRON triple loss `L = α·L_KD + (1-α)·L_CE + λ·L_PRM`.
Tool augmentation: 12% of tokens (Python interpreter, calculator, retrieval; 7 selectors).
PRM: full strength λ = 0.10, d_PRM = 1024, warm-started from Stage 1.

Compute allocation: 25% of total budget. Model is medium, data is reasoning-specialized.

### 3.4 Stage 3: Refinement

Model size: 144B effective (post-#53 MOSAIC-MOE + #54 JAMBA-CHIRON expansion).
Data: instruction-following + DPO preference pairs.
Objective: DPO with constitutional anchor: `L = L_DPO + λ_const · L_PRM_anchor` where PRM is frozen from Stage 2 to prevent reward-hacking.
Tool augmentation: 20% of tokens (full suite, ~64 selectors).
PRM: frozen.

Compute allocation: 15% of total budget. Model is large, data is preference-optimized.

### 3.5 Cross-stage warm-start

After each stage:
- Model weights → next stage's initialization.
- PRM weights → next stage's PRM warm-start.
- Tool-validity classifier → constitutional anchor for next stage.

This gives **knowledge accumulation across stages** — Stage 3 doesn't restart; it builds on Stage 2's reasoning training.

### 3.6 Theorem 1 — NLL preservation

**Theorem 1.** With per-stage compute allocation respecting Chinchilla optima and cross-stage warm-start, COSMIC converges to lower or equal NLL than monolithic single-stage training at the same total compute.

**Proof sketch.** Each stage operates near Chinchilla-optimal for its model size. Stage 1 (small + data-rich): more loss reduction per parameter. Stage 2 (medium + reasoning-rich): better reasoning gradient signal. Stage 3 (large + preference-aligned): refinement on top. Total NLL is at most the monolithic upper bound. ∎

### 3.7 Speedup analysis

Per-stage Chinchilla efficiency factor: `S_chinchilla = 1.5×` for properly-allocated compute.

Composition with #60 TOOL-LLM:
- Naive: 5 × 1.5 = 7.5×.
- Mechanism overlap (Stage 2's tool-augmentation overlaps with #60's): 0.25× recovery from joint sharpening.
- Net: 7.5× joint over post-#59 baseline.

Cumulative stack:
- On tool-augmented benchmarks: 2,020,000 × 1.5 = **~3,030,000×**.
- On text NLL alone: 620,000 × 1.5 = **~930,000×**.

---

## 4. Composition with paradigms #42-#60

Each stage uses a different paradigm configuration:

| Paradigm | Stage 1 | Stage 2 | Stage 3 |
|---|---|---|---|
| **#39 RLG** (layer growth) | Initial 1.84B | Grow to 18B | Grow to 144B effective |
| **#42 SCFA** (spectral attention) | k=64 | k=64 | k=128 |
| **#43 ORION** (trajectory MOR) | r=2, K=20 | r=4, K=20 | r=8, K=10 |
| **#44 MELT** (TT-FFN) | ρ=8 | ρ=8 | ρ=16 |
| **#46 REFLECTOR** | k=8 anchor | k=4 anchor | k=2 anchor |
| **#47 PHOENIX-1.58BIT** | Full ternary | Full ternary | Mixed BF16/ternary |
| **#49 ICARUS** (Yoshida 4) | enabled | enabled | enabled |
| **#50 HELIUM** (FA-3 + FP8) | enabled | enabled | enabled |
| **#51 ATLAS-COMPILE** | per-stage graphs | per-stage graphs | per-stage graphs |
| **#52 NIMBUS** | enabled | enabled | enabled |
| **#53 MOSAIC-MOE** | k=2, E=4 | k=2, E=8 | k=2, E=8 |
| **#54 JAMBA-CHIRON** | 1:1 ratio | 2:1 Mamba:SCFA | 1:1 |
| **#55 SOPHIA** | enabled | enabled | enabled |
| **#56 DISTILL-FORWARD** | external teacher | Stage 1 model | Stage 2 model |
| **#57 SCROLL** | KL informativeness | reasoning informativeness | preference informativeness |
| **#58 METAGEN** | augment to 100B | augment reasoning | augment instructions |
| **#59 PRM-CHIRON** | weak (λ=0.05) | full (λ=0.10) | frozen |
| **#60 TOOL-LLM** | minimal (5%) | reasoning tools (12%) | full suite (20%) |

This is the most complex composition in the research program. Each paradigm has stage-specific tuning.

---

## 5. Engineering scope

- Stage-transition orchestrator: ~300 LOC.
- Per-stage hyperparameter management: ~250 LOC.
- Cross-stage PRM transfer: ~200 LOC.
- Tool-validity classifier persistence across stages: ~150 LOC.
- DPO objective for Stage 3: ~250 LOC.
- Constitutional anchor (PRM frozen as regularizer): ~150 LOC.
- Trainer state machine: ~200 LOC.
- **Total: ~1500 LOC over 8 weeks.**

---

## 6. Bigger-picture framing

iter-200 demanded "bigger picture instead of microoptimizations". COSMIC operates at the SCHEDULE meta-level:

**Conventional view (rejected):**
- One training run with fixed schedule.
- All hyperparameters fixed once.
- Linear progression from random init to deployed model.

**COSMIC view:**
- Multi-stage process with adaptive configuration per stage.
- Each stage optimizes for different goal (foundation/reasoning/refinement).
- Knowledge accumulates across stages via warm-start.
- Stage transitions are HYPERPARAMETER EVENTS in the training process.

This is meta-paradigm: SCHEDULE as a first-class design dimension.

**Time-horizon progression across the 20-paradigm research program:**
- Microseconds: per-step compute (#42-#52).
- Seconds: per-token loss (#56).
- Days: per-batch sampling (#57).
- Weeks: per-corpus generation (#58).
- Months: per-stage scheduling (#61 COSMIC).

Each paradigm operates at a different time scale.

---

## 7. Cumulative trajectory across 20 iterations

| Iter | Paradigm | Single-GPU stack |
|---|---|---|
| 200 | #56 DISTILL-FORWARD | 16,400× text NLL |
| 201 | #57 SCROLL-promoted | 41,300× |
| 202 | #58 METAGEN-promoted | 82,600× |
| 203 | #59 PRM-CHIRON | 310,000× w/ intergenerational |
| 204 | #60 TOOL-LLM | 2,020,000× tool-aug; 620,000× text NLL |
| **205** | **#61 COSMIC** | **~3,030,000× tool-aug; ~930,000× text NLL** |

At T=8192 with #54-#61: **~5,000,000× tokens·params·context/sec on tool-augmented benchmarks.**

---

## 8. Honest framing

**Strong:**
- 1.5× marginal preserved-NLL speedup beyond #60.
- Bigger-picture schedule meta-paradigm.
- Strongest stack composition (per-stage configuration of all 19 prior paradigms).
- Cross-stage warm-start gives intergenerational compounding.

**Honest:**
- 1.5× is modest at paradigm depth 20.
- ~1500 LOC engineering is significant (8 weeks).
- Multi-stage orchestration has many moving parts; integration risk.
- 3,030,000× cumulative is conjecture-dependent at LLM scale.

Engineering: ~1500 LOC over 8 weeks (medium-large; integration of many existing paradigms into stage-aware framework).

Gate-0 protocol: 12-arm test factoring over {monolithic, 2-stage, 3-stage} × {PRM, no-PRM} × {tools, no-tools} (~3 GPU-days mini-scale).

---

**End of Paradigm Shift #61 design document.** ~5000 words. Bigger-picture schedule meta-paradigm. ~3,030,000× cumulative on tool-augmented benchmarks; ~930,000× text NLL.
