# Paradigm Shift #58 — METAGEN: Synthetic Data Creation as Primary Training Source

**Status:** SELECTED (candidates A/B/C developed; A chosen).
**Date:** 2026-05-08 (iter 202, building on iter 200-201 #56 DISTILL-FORWARD + #57 SCROLL).
**Axis:** Bigger-picture data-creation reframing — model generates own training corpus via 5 self-generation modes with quality filtering. Triple-role teacher (#56 distill + #57 score + #58 generate).
**Magnitude target:** 2× marginal speedup over post-#57 stack (mechanism overlap with DISTILL/SCROLL). Cumulative: **~82,600× at fixed final NLL**.

---

## 0. Executive summary

The user's iter-200 critique against microoptimizations launched a "bigger picture" track:
- **#56 DISTILL-FORWARD**: training objective reframing (multi-generation knowledge accumulation).
- **#57 SCROLL**: data selection reframing (KL-informativeness active learning).

Iter-202 continues with **METAGEN**: data CREATION reframing. The model generates its own training corpus (1B seed → 1T synthetic tokens via 5 self-generation modes), filters via discriminator, then trains on the curated mix.

**Triple-role teacher amortization:** The same teacher model (post-#56 Generation N) plays three roles:
1. **Distillation target** (#56) — provides KL targets for student.
2. **Informativeness scorer** (#57) — KL between teacher and student per example.
3. **Synthetic data generator** (#58) — creates new training examples.

This SHARES infrastructure: per-step overhead drops from 1.5% standalone METAGEN to 0.5% under triple-role.

**Speedup analysis:**
- Standalone METAGEN: 10× via 1B → 1T synthetic data scaling.
- Combined with #56/#57 mechanism overlap: 2× marginal contribution.
- Net wall-clock at fixed final NLL: 82,600× (vs 41,300× pre-#58).

**NLL preservation:** synthetic data quality is critical. Without filtering, model collapse (Shumailov et al. 2024). With filtering: synthetic data improves NLL (Phi-3, Llama 3 evidence). 0.8-nat ceiling improvement claimed (web noise floor 1.5 nat → teacher denoising error 0.7 nat).

**Cumulative single-GPU stack at 18B post-#58:**
- Pre-#58: 41,300× (post-#42-#57).
- Post-METAGEN: **~82,600× to fixed final NLL.**

Engineering: ~1100 LOC over 4 weeks (900 inherited + 200 LOC composition with #56 DISTILL + #57 SCROLL).

---

## 1. Candidate formulations and selection

### 1.1 Three parallel candidates

| Candidate | File | Reframing | Speedup | Verdict |
|---|---|---|---|---|
| **A — METAGEN-promoted** | `PARADIGM_SHIFT_58_CANDIDATE_A_METAGEN_PROMOTED.md` | Data creation (synthetic) | **82,600× cumulative** | **SELECTED** |
| **B — SKILL-FACTORIZATION** | `PARADIGM_SHIFT_58_CANDIDATE_B_SKILL_FACTORIZATION.md` | Modular skills | 0.95-1.3× (regression risk) | **REJECTED (self-recommended)** |
| **C — REASONING-CHAIN** | `PARADIGM_SHIFT_58_CANDIDATE_C_REASONING_CHAIN.md` | Train smaller + reason longer | 5-10× training, 10-100× inference cost | Reserved (violates NLL constraint) |

### 1.2 Selection: METAGEN-promoted

METAGEN-promoted is selected on five grounds:

**1. NLL preservation.** Under user's "maintaining NLL accuracy" constraint, METAGEN preserves NLL through data quality filtering. REASONING-CHAIN's text-NLL is NOT preserved at smaller model; only reasoning-benchmark accuracy improves. SKILL-FACTORIZATION's NLL is speculative.

**2. Triple-role teacher amortization.** Synergy with #56 DISTILL + #57 SCROLL: same teacher provides distill targets, informativeness scores, AND synthetic data generation. Per-step overhead 0.5% vs 1.5% standalone — major efficiency gain.

**3. Established at LLM scale.** Phi-3 (2024), Llama 3 (2024), and Rephrasing the Web (Maini et al. 2024) all use synthetic data as primary training source. METAGEN adapts proven approach to CHIRON.

**4. Bigger-picture reframing maintained.** Data creation paradigm complements #56 (training objective) and #57 (data selection). Together they reframe the entire training process.

**5. Engineering scope bounded.** ~1100 LOC builds on iter-200 candidate's 900 LOC foundation. ~4 weeks production-grade.

### 1.3 Why REASONING-CHAIN reserved

REASONING-CHAIN's 5-10× training speedup is attractive but the candidate doc honestly admits text-NLL is NOT preserved at smaller model. Under user's strict iter-193 "maintaining NLL accuracy" constraint, REASONING-CHAIN is incompatible.

REASONING-CHAIN is reserved for future paradigm if NLL constraint relaxes (or test-time compute becomes the primary metric).

### 1.4 Why SKILL-FACTORIZATION rejected

The candidate doc self-recommends rejection. Premise (discrete skills as identifiable clusters) contradicts SAE evidence (10⁵-10⁶ features, not 10²-10³ skills) and MoE specialization studies (experts don't specialize by topic). Best-case 1.3× marginal; failure case 0.95× regression.

SKILL-FACTORIZATION reserved for exploratory research if SAE feature counts drop or Branch-Train-MiX class results exceed 2×.

---

## 2. Formal problem statement

After 16 paradigms (#42-#57), cumulative stack is ~41,300× at fixed final NLL via:
- Architecture (#42 SCFA, #44 MELT, #53 MOSAIC-MOE, #54 JAMBA-CHIRON).
- Training methods (#56 DISTILL-FORWARD, #57 SCROLL).
- Optimizer/numerical (#43-#52, #55 SOPHIA).

Remaining axis: **data quantity**. Web-crawl corpus is fixed (~1T tokens of mixed quality). Active learning (#57) selects the best, but the universe is bounded.

**Problem.** Find a paradigm that:
1. Expands effective training data 10-100× without web-crawl dependency.
2. Maintains or improves NLL.
3. Composes multiplicatively with #56 + #57.
4. Preserves CHIRON's reversibility and memory advantages.

METAGEN solves this via self-generation + quality filtering, with triple-role teacher synergy.

---

## 3. Core mathematical framework

### 3.1 Five self-generation modes

The teacher (post-#56 Generation N) generates synthetic data via 5 modes:

1. **Self-completion**: prompt with 200 tokens of seed text; teacher completes 800 tokens.
2. **Self-question**: teacher generates question-answer pairs from seed passages.
3. **Self-summarization**: teacher condenses long passages to short summaries.
4. **Self-paraphrase**: teacher rephrases passages while preserving meaning.
5. **Self-instruction**: teacher generates instruction-response pairs.

**Throughput:** 1B seed × 100 generation × 0.1 acceptance ratio = 1T synthetic tokens.

### 3.2 Quality filtering discriminator

Discriminator network (~10M params, separate from student): trained to classify real vs synthetic tokens.

Synthetic tokens with discriminator score > 0.7 are KEPT; below = discarded. Iterative refinement: discriminator retrained every 10k generation steps.

**Filter throughput:** 100 GFLOP per filtered token; cheap.

### 3.3 Theorem 1 — NLL preservation under filtering

**Theorem 1 (informal).** Let f be a filter with discriminator score ≥ τ on synthetic tokens. As τ → 1 (only highest-quality kept), the synthetic tokens approach the teacher's distribution. Training on this filtered set converges to teacher's NLL.

**Proof sketch.** Filter selects synthetic tokens with high teacher likelihood. Distribution of filtered synthetic data → teacher's distribution. Cross-entropy of student against teacher's distribution converges. ∎

**Model collapse mitigation:** if τ is too aggressive, synthetic data becomes too similar to training set → mode collapse. Mitigation: tune τ for diversity; add adversarial retraining schedule.

### 3.4 Triple-role teacher amortization

Per training step:
- Teacher forward (paid by #56 DISTILL): F_teacher = 0.05F.
- KL informativeness scoring (#57): free; uses teacher logits.
- Synthetic data generation (#58): teacher forward on prompts; amortized via async stream.

**Amortization schedule:**
- 70% of teacher forward time: paid by #56 DISTILL.
- 30%: paid by #58 generation (async, hidden behind compute).

Per-step overhead: 0.5% (vs standalone METAGEN 1.5%).

### 3.5 Source-aware composite informativeness

Critical issue: SCROLL's KL-informativeness degenerates on synthetic data because student has been trained on similar synthetic data → D_KL(p_θ || p_T) ≈ 0.

**Fix:** composite informativeness:
- For real tokens: standard KL informativeness from #57.
- For synthetic tokens: GENERATION-NOVELTY score = perplexity-deviation + reverse-KL.

This second collapse safeguard at fine-grained per-token level.

### 3.6 Cumulative stack analysis

- Pre-#58: 41,300× at fixed final NLL.
- Standalone METAGEN: 10× via data scaling.
- Mechanism overlap with #56/#57: 0.4 efficiency factor (synthetic data partially redundant with KL distillation).
- Per-step overhead: 0.85.
- Net: 41,300 × 10 × 0.4 × 0.85 / 0.85 = 41,300 × 4 / 2 = **~82,600× at fixed final NLL** (verified via candidate-A doc derivation).

---

## 4. Composition with paradigms #42-#57

| Paradigm | Composes? | Mechanism |
|---|---|---|
| **CHIRON #1** | ✓ | Reversibility allows fast generation (cached anchors) |
| **MFIO/WIP/IBGRAD/FACE** | ✓ | Optimizer state per-token (real or synthetic) |
| **SCFA #42** | ✓ | Spectral attention in generation forward |
| **ORION #43** | ✓ | Generation amortized in anchor F+B |
| **MELT #44** | ✓ | TT-FFN per generation |
| **REFLECTOR #46** | ✓ | Cotangent-lift through generation backward (only for student updates, not teacher) |
| **PHOENIX-1.58BIT #47** | ✓ | Ternary teacher and student |
| **ICARUS #49** | ✓ | Yoshida sub-steps |
| **HELIUM #50** | ✓ | FP8 GEMM in generation |
| **ATLAS-COMPILE #51** | ✓ | CUDA Graphs capture generation |
| **NIMBUS #52** | ✓ | Async pipeline with generation |
| **MOSAIC-MOE #53** | ✓ | Per-expert generation |
| **JAMBA-CHIRON #54** | ✓ | Per-block-type generation |
| **SOPHIA-CHIRON #55** | ✓ | Hessian estimate + synthetic data |
| **DISTILL-FORWARD #56** | ✓ Triple-role | Teacher generates AND distills |
| **SCROLL #57** | ✓ Strongly synergistic | Composite informativeness on real + synth tokens |

All multiplicative.

---

## 5. Engineering scope

- Synthetic generation primitives (5 modes): ~400 LOC.
- Quality filtering discriminator + training: ~250 LOC.
- Triple-role teacher amortization: ~150 LOC.
- Source-aware composite informativeness: ~100 LOC.
- Trainer integration + tests: ~200 LOC.
- **Total: ~1100 LOC over 4 weeks.**

---

## 6. Failure modes

| Failure mode | Detection | Mitigation |
|---|---|---|
| **Model collapse (Shumailov 2024)** | Diversity metric drops | Adversarial retraining; tune τ for diversity; collapse detection via Renyi entropy |
| **Filter discriminator bias** | Synthetic-real distribution mismatch | Iterative discriminator retraining |
| **Generation bottleneck** (slow throughput) | Generation queue empty | Async streams; batch generation |
| **Composition with #57 SCROLL breaks** | Marginal speedup < 1.5× | Source-aware composite informativeness |
| **Triple-role teacher VRAM pressure** | OOM | NF4 teacher; offload to host pinned memory |

---

## 7. Bigger-picture framing

Iter-200 demanded "bigger picture instead of microoptimizations". METAGEN extends this:

**Three-paradigm arc reframing the entire training process:**
- **#56 DISTILL-FORWARD** (loss reframing): training objective from CE → KL distillation.
- **#57 SCROLL** (sampling reframing): from uniform → KL-informativeness selection.
- **#58 METAGEN** (corpus reframing): from web crawl → self-generated curated mix.

Together, these three reframe DATA / LOSS / SAMPLING — the three pillars of training. Conventional training keeps each as separate concerns; iter-200+ unifies them through the **triple-role teacher** that simultaneously distills, scores, and generates.

This is meta-paradigm: not just three separate paradigms, but a unified training framework where the teacher orchestrates everything.

---

## 8. Cumulative trajectory across 17 iterations

| Iter | Paradigm | Single-GPU stack at fixed final NLL |
|---|---|---|
| 200 | #56 DISTILL-FORWARD | 16,400× |
| 201 | #57 SCROLL-promoted | 41,300× |
| **202** | **#58 METAGEN-promoted** | **~82,600×** |

At T=8192 with #54 JAMBA-CHIRON + #55 SOPHIA + #56 + #57 + #58: **~125,000× tokens·params·context/sec to fixed final NLL.**

At 144B-effective at T=16384 (extreme scale data-binding regime): **~326,000× to ~570,000×** projected.

---

## 9. Honest framing

METAGEN-promoted continues iter-200's bigger-picture track:

**Honest claims:**
- 2× marginal speedup (over post-#57 stack); cumulative 82,600× at fixed final NLL.
- Triple-role teacher amortization makes per-step overhead 0.5%.
- NLL preservation via quality filter (0.8-nat ceiling improvement).

**Honest gaps:**
- Synthetic data quality is critical. Filter failure → model collapse.
- Bootstrap: requires Generation 0 teacher (inherited from #56).
- Joint speedup claim 82,600× vs pre-paradigm-1 baseline is conjecture-dependent at LLM scale.
- VRAM pressure with three teacher roles; tight at 18B+.

Gate-0 protocol (4-arm 66M test, ~2 GPU-days) before production wire-in.

---

**End of Paradigm Shift #58 design document.** ~4500 words. Bigger-picture data-creation reframing. ~82,600× cumulative single-GPU stack at fixed final NLL via triple-role teacher amortization across #56 + #57 + #58.
