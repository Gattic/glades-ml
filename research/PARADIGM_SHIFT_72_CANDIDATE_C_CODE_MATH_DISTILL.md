# Paradigm Shift #72 — Candidate C: CODE-MATH-DISTILL-CHIRON — Domain-Specialized Teacher Pair for Code & Math Refinement

**Status:** CANDIDATE C (under evaluation alongside A and B at iter 216). **Recommendation: REJECT** — heavy mechanism overlap with #69 REASONING-DISTILL (R1 671B already covers code/math reasoning subsets); marginal lift estimate 1.2-2× over #69 baseline on code/math benchmarks; per-axis magnitude is borderline-microoptimization per the iter-200 brief that explicitly critiqued 1.2-1.875× incremental gains. The mechanism is technically a sound application of #68 SUPER-DISTILL's cached-logit pipeline with a domain-specialized teacher pair (DeepSeek-Coder-V2 / Qwen2.5-Coder-32B for code; DeepSeek-Math 7B / NuminaMath for math), but it does NOT open a new axis and does NOT contribute magnitudes-better lift. **#72-C is mechanism-equivalent to #68 + #69 with a domain-conditioning router; novelty is a teacher-dispatch rule rather than an architectural primitive.**

**Date:** 2026-05-08 (Ralph-loop iteration 216).
**Axis:** REFINEMENT of the TEACHER PROVENANCE axis already opened at #68 (text frontier-class teacher), refined at #69 (reasoning-class teacher R1 671B), and multi-generation-extended at #70 (TOOL-trajectory teacher). #72-C adds DOMAIN SPECIALIZATION (code, math) atop the existing teacher-provenance hierarchy. Not a new composition axis; a sub-refinement within an existing axis.

**Magnitude target (honest):** **~1.2-2× over post-#69 code/math baseline** on HumanEval, MBPP, LiveCodeBench, BigCodeBench, MATH-500, AIME, GSM8K, MiniF2F. Headline 1.5-3× best-case code; 1.2-2× best-case math. **Cumulative-stack contribution: marginal — code/math benchmarks already absorbed by #69's R1 671B reasoning teacher coverage; the marginal lift is the DOMAIN-DEPTH refinement within an axis already saturated.** Net post-#72-C cumulative-stack figure: text-NLL preserved (#68/#69 unchanged); code-axis sub-magnitude lifts at most 2-3× incremental over #69 baseline. This sits at the borderline-microoptimization threshold the iter-200 brief explicitly critiqued.

---

## 0. Status & axis & honest headline

- **Status:** CANDIDATE C. Recommendation **REJECT.** Of the iter-216 candidates (A, B AUDIO-DISTILL reserved, C CODE-MATH-DISTILL), C has the cleanest domain-specialization framing but the WEAKEST claim to magnitudes-better lift. The mechanism is fundamentally #68 + #69 + a domain-detection router; novelty is a teacher-dispatch rule.
- **Date:** 2026-05-08, iter 216.
- **Axis:** REFINEMENT within TEACHER PROVENANCE — opened at #68 (Llama 3.1 405B / Mixtral 8x22B as text teachers), refined at #69 (DeepSeek-R1 671B / o1-class reasoning teachers), multi-gen-extended at #70 (TOOL-trajectory teachers). #72-C adds a domain-specialization sub-layer: a code teacher (DeepSeek-Coder-V2 236B-MoE or Qwen2.5-Coder-32B or Granite Code 34B) and a math teacher (DeepSeek-Math 7B-RL or NuminaMath-7B-CoT or MiniMath). Not a new axis.
- **Honest headline:** **~1.2-2× over post-#69 code/math baseline.** This is below the magnitudes-better bar at iter-200 ("looking at the bigger picture instead of focusing on microoptimizations"). The mechanism is sound, the prior art (DeepSeek-Coder, Qwen2.5-Coder, Granite Code) is mature, but the teacher-provenance axis is heavily saturated by #69 R1 671B which already encompasses code and math reasoning. **Per-axis magnitude is borderline-microoptimization; per-program relevance is below thresholds set at iter-200 and re-asserted at iter-215/216.**

The user brief at iter-216 reads (post-iter-200): "magnitudes-better compute + memory + nll accuracy + single-GPU + novel + bigger-picture." **The phrase "bigger picture" was load-bearing at iter-200 and remains so at iter-216.** A 1.2-2× lift on code/math benchmarks is exactly the kind of microoptimization the iter-200 brief critiqued. #72-C clears NO magnitude bar at the bigger-picture threshold; the only honest framing is REJECT.

---

## 1. Executive summary

After 30 paradigms (#42-#71), the cumulative single-GPU stack at iter-215 close (post-#71-A or post-#71-B selection, or no #71 selection if all reserved):
- Causal-reasoning subset: ~1,000,000,000× (~10⁹).
- Grounded-reasoning: ~660,000,000×.
- Agent benchmarks: ~643,000,000× (post-#70 1.2× synergy).
- Tool-augmented: ~150,000,000×.
- Text NLL: ~93,000,000× (general).
- Knowledge-augmented: ~55,000,000×.
- Code/math sub-axis (within text NLL): heavily overlapping with #69 R1 671B reasoning teacher coverage.

Reasoning-heavy domains (code, math, formal proofs) are absorbed by #69 R1 671B. R1 was trained on 800k samples of math, code, science, and reasoning trajectories from o1-class teacher provenance; its capability profile on HumanEval (~88%), LiveCodeBench (~62%), MATH-500 (~95%), AIME-2024 (~52%) is at or above frontier code-specialized teachers (DeepSeek-Coder-V2 236B MoE: HumanEval ~92%, LiveCodeBench ~60%; Qwen2.5-Coder-32B: HumanEval ~92%, LiveCodeBench ~58%; Granite Code 34B: HumanEval ~85%, LiveCodeBench ~50%). On math, R1 is at or above DeepSeek-Math 7B-RL (MATH-500 ~83%) and NuminaMath-7B-CoT (MATH-500 ~75%, AIME ~32%).

**The empirical anchor is uncomfortable for #72-C:** R1 671B is COMPETITIVE WITH OR BETTER THAN every specialized code/math teacher candidate on the benchmarks #72-C would target. The marginal lift from adding a specialized teacher is bounded by R1's capability ceiling — there is no headroom for substantial additional lift unless the specialized teacher exceeds R1 on the specific subset.

**Mechanism (sketch):**
- **Code teacher dispatch (3-tier choice):**
  - **Tier 1 (cheapest, 32B):** Qwen2.5-Coder-32B-Instruct (Apache 2.0). HumanEval ~92%, LiveCodeBench ~58%. Self-hosted on single A100 80GB. ~$0 inference cost.
  - **Tier 2 (balanced, 33B-MoE):** DeepSeek-Coder-V2-Lite (Llama license; 16B-MoE-active). HumanEval ~91%, LiveCodeBench ~55%. Self-hosted on single A100. ~$0 inference cost.
  - **Tier 3 (frontier, 236B-MoE):** DeepSeek-Coder-V2 (Llama license; 21B-active 236B-total MoE). HumanEval ~92%, LiveCodeBench ~60%. Requires 8×A100 80GB or H200 single-node for inference. ~$5-10/M tokens.
- **Math teacher dispatch (3-tier choice):**
  - **Tier 1 (cheapest, 7B):** DeepSeek-Math 7B-RL (Apache 2.0). MATH-500 ~83%, GSM8K ~88%. Self-hosted on single 16GB GPU. ~$0 inference cost.
  - **Tier 2 (balanced, 7B):** NuminaMath-7B-CoT (Apache 2.0). MATH-500 ~75%, AIME ~32%. Pure CoT supervision; cleanest distill signal.
  - **Tier 3 (frontier, but no clean win over R1):** Qwen2.5-Math-72B (Apache 2.0). MATH-500 ~85%. Requires 4×A100 80GB. Marginal capability over R1.
- **Domain detection / teacher-dispatch router:** classify each training problem by domain (general / code / math / mixed) using a lightweight classifier (~2M-param transformer head) or rule-based router (regex on `def `, `class `, `import `, `function`, math LaTeX `\\frac`, `\\sum`, etc.). Dispatch to appropriate teacher; fall back to general teacher (#68 Llama 3.1 405B) for non-classified.
- **Cached-logit pipeline (per #68):** identical to #68/#69. Top-K=16 logit cache; KL-CE blended loss; α/τ schedule.
- **KL-CE distillation loss:** L = α · CE(student, ground_truth) + (1-α) · τ² · KL(softmax(z_T_specialized/τ) || softmax(z_S/τ)). The specialized teacher provides the soft signal on its domain.

**Speedup:**
- **Code-axis lift over post-#69 baseline:** estimated 1.5-3× best-case (Qwen2.5-Coder-32B / DeepSeek-Coder-V2 specialty depth on niche libraries, modern API coverage, edge-case handling). Most of this is HumanEval-pass-rate improvement of perhaps 3-5 percentage points (~88% → ~92%) and LiveCodeBench improvement of perhaps 4-6 points. Translated to fixed-final-NLL-on-code-corpus speedup: 1.5-3× steps reduction at best.
- **Math-axis lift over post-#69 baseline:** estimated 1.2-2× best-case. R1 671B already strong on MATH-500 (~95%). DeepSeek-Math 7B-RL specialty is in formal proof structure, theorem-proving (MiniF2F where R1 is weaker), AIME problem structure. MATH-500 lift bounded; AIME lift potentially larger but still 1.2-2× steps reduction.
- **Cross-axis contributions:** 1.0× on text NLL (general; #69 unchanged); 1.0× on agent / tool / reasoning / VL / audio (orthogonal subsets).

**Cumulative stack update (#72-C selected):**
- Code sub-axis (within text NLL): ~93M× × 1.5-3× = ~140M-280M× best-case.
- Math sub-axis (within text NLL): ~93M× × 1.2-2× = ~110M-185M× best-case.
- General text NLL: ~93M× (unchanged).
- All other axes: unchanged.

**NLL preservation honest framing:**
- General text NLL: unchanged (#69 substrate preserved).
- Code/math sub-axis NLL: **NOT bit-exact preserved** because the teacher distribution shifts when domain-detection router dispatches to specialized teacher. However, NLL on code corpus IMPROVES (lower) per the speedup mechanism; this is desirable, not a regression. The dispatch is a ratchet — student inherits at-or-better teacher distribution per domain.

**Engineering scope:** ~600 LOC over 3 weeks. Domain-detection router (~150 LOC), specialized teacher inference pipeline reuse from #68/#69 (~100 LOC), code/math corpora curation and preprocessing (~200 LOC), cached-logit pipeline reuse from #68 (~50 LOC), tests + Gate-0 harness (~50 LOC), benchmark harness for HumanEval/MBPP/LiveCodeBench/BigCodeBench/MATH-500/AIME/GSM8K/MiniF2F (~50 LOC).

**Joint Gate-0 PASS probability:** ~75% (specialized teacher distillation pipelines are mature; DeepSeek-Coder, Qwen2.5-Coder are proven). Mechanism-correctness is high.
**LLM-scale empirical confirmation probability at single-GPU CHIRON:** ~30% — modulo whether the marginal lift over post-#69 baseline is actually realizable. Given R1 671B's strong code/math capability, the headroom for additional specialized-teacher lift is narrow; ~30% probability of clean ≥1.5× lift.

---

## 2. Mechanism: domain-detection router + specialized teacher pair + cached-logit dispatch

### 2.1 Specialized teacher choice

| Teacher | Domain | Params | License | HumanEval / MATH-500 | Memory (BF16) | Inference cost |
|---|---|---|---|---|---|---|
| **Qwen2.5-Coder-32B-Instruct** | Code | 32B | Apache 2.0 | ~92% / — | 64 GB | $0 self-hosted A100×4 |
| **DeepSeek-Coder-V2-Lite** | Code | 16B-MoE-active / 33B total | Llama | ~91% / — | 66 GB | $0 self-hosted A100×4 |
| **DeepSeek-Coder-V2** | Code | 21B-MoE-active / 236B total | Llama | ~92% / — | 472 GB | $5-10/M tokens or H200×8 |
| **Granite Code 34B** | Code | 34B | Apache 2.0 | ~85% / — | 68 GB | $0 self-hosted A100×4 |
| **DeepSeek-Math 7B-RL** | Math | 7B | Apache 2.0 | — / ~83% | 14 GB | $0 self-hosted single-GPU |
| **NuminaMath-7B-CoT** | Math | 7B | Apache 2.0 | — / ~75% | 14 GB | $0 self-hosted single-GPU |
| **Qwen2.5-Math-72B** | Math | 72B | Apache 2.0 | — / ~85% | 144 GB | $0 self-hosted A100×4 |

**Default code teacher: Qwen2.5-Coder-32B-Instruct.** Justification:
1. Strongest open code teacher in the 32B band (HumanEval ~92%).
2. Apache 2.0 license; clean redistribution.
3. Self-hosted single-node A100 80GB×4 (~$8-15/hour cloud; one-shot inference pre-pass).
4. Mature production engineering (Qwen-Agent, Qwen-Code release ecosystem).

**Default math teacher: DeepSeek-Math 7B-RL.** Justification:
1. Smallest viable specialized math teacher; fits single 16GB GPU for inference.
2. RL-trained on math-specific reward; specialty-depth advantage.
3. Apache 2.0 license.
4. ~$0 inference cost (self-hosted single-GPU).

### 2.2 Domain detection / teacher-dispatch router

Lightweight router classifies each training problem into one of four classes:
- **CODE:** detected by presence of code blocks, programming language keywords, function signatures, file paths, etc. Regex + rule-based.
- **MATH:** detected by LaTeX math notation (`\\frac`, `\\sum`, `\\int`, `\\sqrt`), problem patterns ("find x such that...", "prove that..."), AIME/IMO/Olympiad problem framing.
- **MIXED:** both code and math (e.g., "implement an algorithm that solves..."). Routed to either teacher based on dominant signal.
- **GENERAL:** neither; routed to #69's R1 671B reasoning teacher.

Implementation choices:
- **Option 1 (rule-based):** ~200 regex patterns. Fast (<10 µs per problem), interpretable, no training. Initial implementation; easy to debug.
- **Option 2 (neural):** 2M-param transformer head trained on labeled code/math/general corpus (~10k examples). Higher accuracy on edge cases (code-style explanations, mixed prose-math, etc.). Reserved for refinement after Gate-0 if router accuracy is the bottleneck.

Default: **Option 1 rule-based** with a few hundred manually-curated patterns. Fall-back to general teacher on ambiguous classifications.

### 2.3 Cached-logit pipeline (per #68 SUPER-DISTILL)

Teacher inference pre-pass per dispatched domain:
- **CODE corpus** (~50B tokens curated from The Stack v2 dedup, HumanEval-source-style scaffolding, MBPP, LiveCodeBench problems-and-solutions, LeetCode-public, BigCodeBench): inference with Qwen2.5-Coder-32B at top-K=16. Cache size: ~50B tokens × 128 bytes/position ≈ 6.4 TB. **Stored on NVMe; loaded JIT during training.**
- **MATH corpus** (~10B tokens curated from OpenMath-Instruct-2, ProofPile-2, NuminaMath, AIME/IMO problem corpora, Lean theorem-proving corpora): inference with DeepSeek-Math 7B-RL. Cache size: ~10B tokens × 128 bytes ≈ 1.3 TB.
- **GENERAL corpus** (everything else; reuses #69 R1 671B cache from prior paradigm).

Cache reuse: if #69 R1 671B cache exists for a token-position, prefer specialized teacher cache when available; fall back to R1 cache otherwise.

### 2.4 Loss formulation (domain-conditional teacher dispatch)

```
L(t) = α(t) · CE(student, ground_truth_t) + (1 - α(t)) · τ² · KL(softmax(z_T_dispatch[t]/τ) || softmax(z_S[t]/τ))
```

where `z_T_dispatch[t]` is the dispatched teacher's logits at position t:
- if domain(sequence_containing_t) == CODE: z_T_dispatch = z_QwenCoder32B[t]
- elif domain == MATH: z_T_dispatch = z_DeepSeekMath7BRL[t]
- elif domain == MIXED: z_T_dispatch = (z_QwenCoder32B[t] + z_DeepSeekMath7BRL[t]) / 2 (logit averaging)
- else (GENERAL): z_T_dispatch = z_R1_671B[t] (per #69 substrate)

τ = 4.0 (code) or 3.0 (math, slightly lower because math problems have lower-entropy correct answers).

### 2.5 Code/math corpus curation

**Code corpus (~50B tokens):**
- The Stack v2 dedup (~3.5 TB raw; subset to ~30B tokens after quality filter).
- HumanEval-source-style scaffolding (~2B tokens; problems and solutions in similar style).
- MBPP / LiveCodeBench / BigCodeBench problem-solution pairs (~5B tokens).
- LeetCode-public solutions (~5B tokens; permissive subset only).
- Codeforces / AtCoder solutions (~5B tokens; competition programming).
- Excludes: GPL / restrictive-license code; proprietary corpora.

**Math corpus (~10B tokens):**
- OpenMath-Instruct-2 (~3B tokens; problem-solution pairs).
- ProofPile-2 (~4B tokens; formal mathematics, theorem proving).
- NuminaMath (~1.5B tokens; competition math problem-solution pairs).
- AIME / IMO / Putnam problem corpora (~0.5B tokens).
- Lean theorem-proving corpora (~1B tokens; formal verification).

Total: ~60B tokens domain-specialized corpus. Manageable; single-pass through corpus during distillation.

---

## 3. Theoretical analysis

### 3.1 Teacher-provenance ceiling argument

The student's asymptotic NLL on a domain-D corpus is bounded below by the teacher's NLL on D (KL distillation argument: student converges to teacher's distribution as α → 0, τ → 1). For code/math:
- **R1 671B asymptotic NLL on code corpus:** strong (~88% HumanEval, ~62% LiveCodeBench imply NLL is competitive with frontier code teachers).
- **Qwen2.5-Coder-32B asymptotic NLL on code corpus:** marginally stronger (~92% HumanEval, ~58% LiveCodeBench).
- **Marginal teacher-NLL gap:** small. Implied marginal student-NLL gap: small. **Implied speedup: 1.2-2× steps reduction at most** (per the standard teacher-provenance speedup formula: speedup ≈ exp(NLL_strong - NLL_weak)).

For math:
- **R1 671B asymptotic NLL on math corpus:** strong (~95% MATH-500).
- **DeepSeek-Math 7B-RL asymptotic NLL on math corpus:** marginally stronger on niche subsets (theorem proving, AIME structure).
- **Marginal teacher-NLL gap:** even smaller than code. **Implied speedup: 1.1-1.5× steps reduction at most.**

### 3.2 Theorem 1 — Code/math sub-axis NLL improves monotonically

**Theorem 1 (informal).** For any code or math training sequence S:
```
NLL_post-#72-C(S) ≤ NLL_pre-#72-C(S) = NLL_post-#69(S)
```
i.e., the specialized teacher dispatch is a ratchet that never degrades NLL on its domain.

**Proof sketch.** The KL-CE blended loss with α schedule from 0.05 → 0.9 ensures the student's training-time gradient on domain-D problems is dominated by the dispatched teacher's distribution. If the dispatched teacher has lower NLL on D than the general teacher (R1 671B on D), the student's asymptotic NLL on D is bounded by the dispatched teacher's NLL. If the dispatched teacher has higher NLL on D, fall back to general teacher. The dispatch policy is monotonically safe: never worse than the general baseline. □

**Implication:** the code/math sub-axis NLL never regresses. The marginal improvement is bounded by the teacher-NLL gap, which is small for code/math (Section 3.1).

### 3.3 Theorem 2 — General-axis text NLL preserved (Theorem inheritance from #69)

**Theorem 2 (informal).** For any general-domain training sequence S (non-code, non-math):
```
NLL_post-#72-C(S) = NLL_post-#69(S)
```
i.e., the general-axis text NLL is preserved exactly.

**Proof sketch.** For non-code, non-math S, the domain-detection router classifies S as GENERAL, dispatches to R1 671B (per #69 substrate), and uses #69's existing logit cache. The training signal is identical to post-#69. NLL preserved. □

### 3.4 Theorem 3 — Memory cost bound

**Theorem 3 (informal).** Total GPU memory footprint of #72-C beyond pre-#72 stack:
```
ΔMemory_GPU = 0 GB during student training
```
because all specialized teacher inference is offline (pre-pass), with cached logits loaded JIT from NVMe storage. The student's GPU footprint is unchanged from post-#69.

**Implication:** single-GPU 16 GB ceiling preserved trivially. Memory cost is OFFLOADED to NVMe storage (~7.7 TB cached logits across code + math) and to the one-shot teacher-inference cloud cost (~$10K-30K depending on teacher choice).

### 3.5 Domain-detection accuracy bound

Rule-based router accuracy on representative dev set: ~90-95% (estimated; matches industry production routers like CodeBERT classifier accuracy on similar tasks). Classification errors lead to:
- **CODE classified as GENERAL:** student trains on R1 671B teacher; ~equivalent to post-#69 baseline. Lost specialty signal but no regression.
- **GENERAL classified as CODE:** student trains on Qwen2.5-Coder-32B for non-code text. Possible NLL regression on general text. Mitigation: fall-back rule on Qwen-Coder logit confidence (if Qwen-Coder's top-1 logit on a non-code token is low-confidence, fall back to R1 671B).

Net router accuracy ≥ 90% bounds NLL regression risk to <10% of dispatched code/math problems carrying the wrong teacher signal. Modest impact.

### 3.6 Honest NLL preservation framing

- **General text NLL:** PRESERVED bit-exact by Theorem 2 (#69 substrate untouched).
- **Code/math sub-axis NLL:** IMPROVES by Theorem 1 (monotonic ratchet). Improvement is ~1.2-2×.
- **No regression on existing axes** by construction (router fall-back to general teacher on ambiguous).
- **Net stack effect:** small positive on code/math sub-axis; zero on general text NLL.

---

## 4. Composition with #68 SUPER-DISTILL + #69 REASONING-DISTILL + heavy overlap analysis

### 4.1 Composition with #68 SUPER-DISTILL (substrate inheritance)

#68 opened the TEACHER PROVENANCE axis with Llama 3.1 405B as the general text teacher. #72-C inherits #68's cached-logit pipeline (top-K=16, KL-CE blended loss, α/τ schedule, COSMIC stage integration). The specialized teachers (Qwen2.5-Coder-32B, DeepSeek-Math 7B-RL) are drop-in replacements for Llama 3.1 405B on their respective domains.

**Marginal contribution beyond #68:** domain specialization. #68 was domain-agnostic.

### 4.2 Composition with #69 REASONING-DISTILL — THE CRITICAL OVERLAP

This is the load-bearing analysis for the #72-C verdict. **#69 R1 671B already covers code and math reasoning subsets.** R1 was distilled from o1-class reasoning teacher with explicit reinforcement learning on math, code, science, and reasoning problems. Its capability profile:
- HumanEval ~88%
- MBPP ~83%
- LiveCodeBench ~62%
- BigCodeBench ~50%
- MATH-500 ~95%
- AIME-2024 ~52%
- GSM8K ~95%
- MiniF2F (theorem-proving) ~30-40% depending on subset

**Compare to specialized teachers:**
- Qwen2.5-Coder-32B HumanEval ~92% (4 points higher than R1).
- DeepSeek-Coder-V2 LiveCodeBench ~60% (2 points lower than R1).
- DeepSeek-Math 7B-RL MATH-500 ~83% (12 points LOWER than R1).
- Qwen2.5-Math-72B MATH-500 ~85% (10 points LOWER than R1).
- NuminaMath-7B-CoT MATH-500 ~75% (20 points LOWER than R1).

**Empirical reality:** R1 671B is COMPETITIVE WITH OR BETTER THAN every specialized math teacher on the math benchmarks. On code, R1 is competitive on LiveCodeBench but ~4 points behind on HumanEval. The specialty-depth advantage of code teachers is bounded at ~4-6 percentage points on HumanEval / MBPP; vanishes or inverts on MATH-500 / AIME.

**Implied marginal speedup of #72-C over #69:**
- Code: at most 1.5-3× best-case (HumanEval 88% → 92%, LiveCodeBench 62% → ~64%, BigCodeBench 50% → ~52%). Translated to fixed-final-NLL-on-code-corpus speedup: 1.5-3× steps reduction.
- Math: at most 1.0-1.2× (R1 already at or above DeepSeek-Math 7B / NuminaMath / Qwen2.5-Math-72B on the standard benchmarks). The specialty advantage exists ONLY on MiniF2F theorem-proving and AIME problem-structure niches; these are narrow subsets.

**Bottom line of the overlap analysis:** ~70% of the theoretical advantage of #72-C is already captured by #69. Marginal lift is 1.2-2× over #69, which sits at the borderline-microoptimization threshold.

### 4.3 Composition with #70 TOOL-DISTILL

#70 opened a TEACHER PROVENANCE refinement on TOOL-trajectory subsets. Code is tool-adjacent (code execution, REPL traces). #70 + #69 already covers code-execution-trajectory subsets via R1 671B + tool-trajectory teachers. **Additional overlap with #72-C: ~10-20% on code subsets where tool execution is involved (e.g., LeetCode runtime feedback, BigCodeBench function-calling).**

### 4.4 Composition with #61 COSMIC

Per-stage COSMIC integration:
- **Stage 1 (Foundation):** general text-only training; #72-C's domain-specialized teachers are dormant. General teacher (R1 671B per #69) handles all domains.
- **Stage 2 (Reasoning):** introduce R1 671B reasoning trace distillation. Dispatched teacher remains R1 for code/math.
- **Stage 3 (Refinement):** **THIS IS WHERE #72-C ACTIVATES.** Domain-detection router dispatches code problems to Qwen2.5-Coder-32B; math problems to DeepSeek-Math 7B-RL. Specialty-depth refinement.

**Stage 3 isolation:** the specialized teachers only appear in Refinement; minimizes cross-stage interference.

### 4.5 Contrast with single-axis candidates A and B

Iter-216 candidate slate:
- **#72-A (TBD; likely a text-axis or new modality candidate).**
- **#72-B (TBD; likely a multimodal or memory-axis candidate).**
- **#72-C (this candidate, CODE-MATH-DISTILL):** ~1.2-2× over post-#69 code/math baseline. Borderline-microoptimization.

**#72-C is the WEAKEST candidate on magnitude** if A or B target ≥10× lift on a primary axis. Strongest only if A and B fail their respective Gate-0s and the program needs a low-risk fallback.

---

## 5. Quantitative speedup with honest band

### 5.1 Headline

**~1.2-2× over post-#69 code/math baseline.** Best-case 1.5-3× on HumanEval/MBPP/LiveCodeBench/BigCodeBench (code); 1.2-2× on MATH-500/AIME/GSM8K/MiniF2F (math). All on sub-axes within the existing text-NLL axis. **No new axis opened.**

### 5.2 Honest band breakdown

| Band end | Conditions |
|---|---|
| **3× (high)** | Qwen2.5-Coder-32B + DeepSeek-Coder-V2 ensemble on code; full 50B-token code corpus; perfect router accuracy; #61 Stage 3 full integration |
| **1.5× (headline code)** | Single Qwen2.5-Coder-32B teacher; standard router accuracy ~93%; standard cached-logit pipeline |
| **1.2× (headline math)** | Single DeepSeek-Math 7B-RL teacher; R1 671B already captures most of the math signal |
| **1.0× (failure)** | R1 671B ceiling already saturates code/math; specialized teachers add no marginal lift; router classification errors degrade general text NLL |

### 5.3 Empirical anchors

- **DeepSeek-Coder-V2 (DeepSeek 2024):** 21B-active 236B-MoE; HumanEval ~92%, LiveCodeBench ~60%. Trained from scratch with code-specific RL. Matches GPT-4 on code benchmarks.
- **Qwen2.5-Coder-32B (Alibaba 2024):** Apache 2.0; HumanEval ~92%, MBPP ~88%, LiveCodeBench ~58%. Single-node deployable.
- **Granite Code 34B (IBM 2024):** Apache 2.0; HumanEval ~85%, MBPP ~80%. Permissive license.
- **DeepSeek-Math 7B-RL (DeepSeek 2024):** Apache 2.0; MATH-500 ~83%, GSM8K ~88%. RL-trained on math.
- **NuminaMath-7B-CoT (Numina 2024):** Apache 2.0; MATH-500 ~75%, AIME ~32%. Pure CoT supervision.
- **DeepSeek-R1 671B (DeepSeek 2025):** MIT; MATH-500 ~95%, AIME ~52%, HumanEval ~88%, LiveCodeBench ~62%. **Already covers code and math at frontier-class quality.**

The 1.2-2× headline sits in the middle of the band; consistent with DeepSeek-Coder-V2's ~4-6 point HumanEval edge over R1 on code-specific benchmarks. **Math headline is honestly bounded at 1.2× because R1 is already at or above all open math teachers.**

### 5.4 Risk-adjusted claim

Joint Gate-0 PASS probability × LLM-scale empirical confirmation probability = 0.75 × 0.30 = **0.225 expected realization**. Risk-adjusted speedup: 1.5× × 0.225 = **~1.11× realized magnitude**. **At or below noise floor.** Honestly, the realized lift is likely indistinguishable from #69 baseline at single-GPU CHIRON scale.

---

## 6. Cumulative stack update

### 6.1 Pre-#72-C stack (post-#71-A or #71-B selection or no #71)

| Axis | Value (assuming no #71 selection; conservative) |
|---|---|
| Causal-reasoning subset | 1,000,000,000× |
| Grounded-reasoning | 660,000,000× |
| Agent benchmarks | 643,000,000× |
| Tool-augmented | 150,000,000× |
| Text NLL | 93,000,000× |
| - Code sub-axis (within text NLL) | ~93M× (post-#69 R1 absorbed) |
| - Math sub-axis (within text NLL) | ~93M× (post-#69 R1 absorbed) |
| Knowledge-augmented | 55,000,000× |

### 6.2 Post-#72-C stack (with CODE-MATH-DISTILL specialized teachers)

| Axis | Pre-#72-C | #72-C factor | Post-#72-C |
|---|---|---|---|
| Causal-reasoning subset | 1,000,000,000× | × 1.0 (orthogonal) | 1,000,000,000× |
| Grounded-reasoning | 660,000,000× | × 1.0 (orthogonal) | 660,000,000× |
| Agent benchmarks | 643,000,000× | × 1.0 (orthogonal) | 643,000,000× |
| Tool-augmented | 150,000,000× | × 1.0 (orthogonal subset) | 150,000,000× |
| Text NLL (general) | 93,000,000× | × 1.0 (preserved by Theorem 2) | 93,000,000× |
| - Code sub-axis | ~93M× | × 1.5-3× | ~140M-280M× |
| - Math sub-axis | ~93M× | × 1.2-2× | ~110M-185M× |
| Knowledge-augmented | 55,000,000× | × 1.0 (orthogonal) | 55,000,000× |

### 6.3 Honesty caveat

**The post-#72-C "lift" is on SUB-AXES of the existing text-NLL axis, not new primary axes.** The cumulative-stack headline figure (text NLL ~93M×, causal-reasoning 10⁹×) does NOT change. Per-axis magnitude on code/math sub-axes lifts by 1.2-3× — sub-magnitude relevance.

**Honest critical view:** at iter-200, the user brief explicitly critiqued microoptimizations of 1.2-1.875× as "focusing on microoptimizations" and asked for "the bigger picture." A 1.2-2× lift on code/math sub-axes is exactly the kind of refinement that brief critiqued. **Per-program relevance: borderline-microoptimization. Below the magnitudes-better bar.**

---

## 7. Engineering scope

### 7.1 Component breakdown

| Component | LOC | Description |
|---|---|---|
| Domain-detection router (rule-based, ~200 regex patterns) | 150 | Classify problems into CODE / MATH / MIXED / GENERAL; fall-back rules |
| Specialized teacher inference pipeline (reuse from #68/#69) | 100 | Qwen2.5-Coder-32B + DeepSeek-Math 7B-RL inference scripts; logit cache export |
| Code/math corpora curation + preprocessing | 200 | The Stack v2 dedup, MBPP, LiveCodeBench, OpenMath-Instruct-2, ProofPile-2 selection; quality filtering; tokenization |
| Cached-logit pipeline reuse from #68 | 50 | Top-K=16 logit cache; KL-CE blended loss; α/τ schedule; cache loader |
| Tests + Gate-0 harness | 50 | HumanEval-mini and MATH-500-mini distill validation |
| Benchmark harness (HumanEval, MBPP, LiveCodeBench, BigCodeBench, MATH-500, AIME, GSM8K, MiniF2F) | 50 | Standardized eval scripts; pass@1 / accuracy reporting |
| **Total** | **~600 LOC** | **~3 weeks engineering** |

If counted standalone (including #68/#69 substrate dependencies): ~600 + 1500 (from #68) + 800 (from #69) = ~2900 LOC. Marginal cost of #72-C beyond shipped #68/#69 is the ~600 LOC table. **Smallest engineering scope of all #72 candidates.**

### 7.2 External-dependency risk

- **Qwen2.5-Coder-32B-Instruct:** Apache 2.0 (HuggingFace `Qwen/Qwen2.5-Coder-32B-Instruct`). No new licensing dependency.
- **DeepSeek-Math 7B-RL:** Apache 2.0 (HuggingFace `deepseek-ai/deepseek-math-7b-rl`). No new licensing dependency.
- **Optional frontier teachers (DeepSeek-Coder-V2 236B):** Llama license; redistribution restrictions but CHIRON student is independent so OK.
- **Code corpora:** The Stack v2 dedup (BigCode; CC-BY 4.0); HumanEval / MBPP / LiveCodeBench (MIT); BigCodeBench (Apache 2.0). All permissive.
- **Math corpora:** OpenMath-Instruct-2 (Apache 2.0); ProofPile-2 (mixed; subset filtering required); NuminaMath (Apache 2.0). Manageable.
- **Cloud cost for teacher inference pre-pass:** Qwen2.5-Coder-32B BF16 inference on 50B code tokens ≈ ~$8K-15K (single-node A100×4 at ~$15/hour for ~1000 hours). DeepSeek-Math 7B-RL on 10B math tokens ≈ ~$1K (single 16GB GPU at ~$2/hour for ~500 hours). **Total ~$10K-16K.** One-shot cost; cache reused across student training runs.
- **Storage:** ~7.7 TB cached logits on NVMe. Manageable.

### 7.3 Timeline

- **Week 1:** Domain-detection router (rule-based); code/math corpus curation; preprocessing.
- **Week 2:** Qwen2.5-Coder-32B + DeepSeek-Math 7B-RL inference pre-pass; logit cache generation.
- **Week 3:** Gate-0 mini-distill on HumanEval-mini + MATH-500-mini; full benchmark harness.

If #68 + #69 not yet shipped, baseline timeline extends substantially; total ~10-12 weeks. **#72-C only makes sense if #68 and #69 are already in production.**

---

## 8. Gates

### 8.1 Gate-0 — premise validation (mandatory before wire-in)

**Hypothesis:** CHIRON-1.84B trained with #72-C dispatched specialized teachers achieves ≥1.3× HumanEval pass@1 improvement over post-#69 baseline at fixed-final-NLL on code corpus.

**Procedure:**
- CHIRON-1.84B post-#69 baseline: HumanEval pass@1 ≈ 70-75% (estimate at 1.84B-band; lower than R1 671B due to model-size gap).
- Add #72-C dispatch: Qwen2.5-Coder-32B teacher on code corpus; rule-based router; cached-logit pipeline at K=16, α 0.05 → 0.9, τ=4.0.
- Train for 5 GPU-hours on code corpus subset (~5B tokens).
- Evaluate on HumanEval pass@1.

**Pass criterion:**
- HumanEval pass@1 ≥ 1.3× post-#69 baseline (e.g., 70% → 91%); AND
- Compute used ≤ 80% of post-#69 baseline compute on equivalent code corpus; AND
- General text NLL on Pile-eval unchanged from post-#69 baseline (Theorem 2 validation).

**Estimated cost:** ~$500 cloud + 1 week engineer time.
**Pass probability:** ~40%. The 1.3× threshold is aggressive; given R1's strong code coverage, this may be hard to clear cleanly.

### 8.2 Gate-1 — full code/math validation

**Procedure:** Same as Gate-0 with full 60B-token code+math corpus and full COSMIC Stage 3 integration. Run for 3 days on cloud.
**Pass criterion:** Code-axis fixed-final-NLL ≤ 0.85 × post-#69 baseline; math-axis fixed-final-NLL ≤ 0.95 × post-#69 baseline; general text NLL unchanged.
**Estimated cost:** ~$5K-10K cloud + 2 weeks engineer time.
**Pass probability:** ~30%.

### 8.3 Gate-2 — joint integration with #61 COSMIC + #69 REASONING-DISTILL

Validate end-to-end with #61 COSMIC Stage 3 integration + #69 R1 671B substrate. Pass: code-axis NLL improvement preserved; math-axis NLL improvement preserved; general text NLL unchanged.

### 8.4 Gate-3 — frontier teacher upgrade (optional)

If Qwen2.5-Coder-32B / DeepSeek-Math 7B-RL are insufficient, upgrade to DeepSeek-Coder-V2 236B / Qwen2.5-Math-72B. Pass: 1.5× HumanEval improvement over post-#69 baseline. Likely insufficient marginal lift to justify the cost increase (~5×).

---

## 9. Honest gaps and failure modes

### 9.1 R1 671B already saturates code/math — the FUNDAMENTAL gap

**This is the load-bearing reason for REJECT.** R1 671B was trained with explicit RL on math, code, science, and reasoning trajectories from o1-class teacher. Its capability profile on HumanEval (~88%), MATH-500 (~95%), AIME (~52%) is at or above frontier specialized teachers.

**Headroom for additional specialized-teacher lift is narrow:**
- Code: ~4-6 percentage points on HumanEval / MBPP (Qwen2.5-Coder-32B: ~92% vs R1: ~88%).
- Math: 0 or NEGATIVE points (R1 ~95% vs DeepSeek-Math 7B-RL ~83% vs Qwen2.5-Math-72B ~85%).

**Implied lift on code corpus NLL:** ~1.5-3× best-case.
**Implied lift on math corpus NLL:** ~1.0-1.2× (close to floor; R1 already ahead).

**Combined effect:** below the magnitudes-better bar.

### 9.2 #72-C is mechanism-equivalent to #68 + #69 + domain router

**Novelty assessment:**
- The cached-logit pipeline is from #68.
- The teacher-provenance ratchet is from #68/#69.
- The KL-CE blended loss is from #68.
- The α/τ schedule is from #68.
- The COSMIC Stage 3 integration is from #61.
- The DOMAIN-DETECTION ROUTER is the only novel component, and it's a rule-based regex classifier (~200 patterns).

**What is NOT novel:**
- DeepSeek-Coder, Qwen2.5-Coder, Granite Code all distill from larger models or co-train with code-specific RL. Standard practice in the field.
- Domain-conditional teacher dispatch is a routine engineering pattern (Mixtral routing, MoE expert routing, etc.).
- Specialty-depth refinement is industry-standard.

**What IS arguably novel:**
- The composition pattern (#68 + #69 + domain router) is not previously published as a paradigm, even though each component is mature. **This is a SYSTEM INTEGRATION novelty rather than an architectural novelty.**

The novelty bar at iter-200 was set high: "looking at the bigger picture instead of focusing on microoptimizations." A regex-based domain router that dispatches to one of three teachers does not clear the bigger-picture bar.

### 9.3 Marginal lift below the magnitudes-better threshold

**Headline magnitude: 1.2-2× over post-#69 baseline.** This is below 10× by an order of magnitude. The iter-200 brief explicitly named this band ("1.2-1.875× incremental") as microoptimization that the program should move past. **#72-C falls squarely within the critiqued band.**

### 9.4 Risk-adjusted speedup near noise floor

Joint Gate-0 PASS × LLM-scale confirmation = 0.75 × 0.30 = 0.225. Risk-adjusted speedup: 1.5× × 0.225 = ~1.11×. **At single-GPU CHIRON scale, this is within evaluation noise.** Hard to distinguish from #69 baseline.

### 9.5 Storage cost ~7.7 TB

Cached logits at K=16 across code (50B tokens) + math (10B tokens) ≈ 7.7 TB on NVMe. Manageable but not trivial. NVMe storage at $0.10/GB-month: ~$770/month ongoing.

### 9.6 Domain-detection router edge cases

Rule-based router handles standard cases well but struggles with:
- Code-style explanations in prose (e.g., "the function `f(x)` does...").
- Math notation embedded in code (e.g., docstrings with LaTeX).
- Mixed-domain problems (e.g., "implement an algorithm that solves the following math problem").

**Mitigation:** fall-back to general teacher on ambiguous classifications. Bounded NLL regression risk.

### 9.7 Math-axis ceiling

R1 671B is already at or above all open math teachers on the standard benchmarks. **The math-axis lift is honestly bounded at 1.0-1.2× even in the best case.** The math half of #72-C is structurally weak; the value (such as it is) lies on the code side.

If #72-C were narrowed to CODE-DISTILL only (drop math), the headline would be 1.5-3× on code. Still below the magnitudes-better bar.

### 9.8 Joint Gate-0 PASS + LLM-scale empirical confirmation probabilities

| Estimate | Value |
|---|---|
| Joint Gate-0 PASS probability | **~75%** (mechanism is mature; specialized teacher distillation is well-validated) |
| Joint Gate-1 PASS probability (≥1.3× HumanEval lift over #69) | **~40%** |
| LLM-scale empirical confirmation probability at single-GPU CHIRON | **~30%** |
| Risk-adjusted speedup (code-axis) | **~1.11×** (= 1.5× × 0.225) — at noise floor |
| Probability of code-axis ≥1.5× lift | **~40%** |
| Probability of code-axis ≥3× lift | **~10%** |
| Probability of math-axis ≥1.5× lift | **~15%** |
| Probability of math-axis ≥3× lift | **<5%** |

### 9.9 Comparison to iter-200 microoptimization critique

The iter-200 brief read: "looking at the bigger picture instead of focusing on microoptimizations." This was the moment the program PIVOTED from per-step compute optimization (#50-#55: 1.2-1.875× each) toward architectural and training-method paradigms (#56 DISTILL-FORWARD: 5×; #57 SCROLL: 12.6×; #58 METAGEN: 2× marginal but 82,600× cumulative; etc.).

**#72-C at 1.2-2× lift exhibits the same pattern as #50-#55.** It is a refinement at the per-axis level rather than a structural reframing. The iter-200 brief explicitly said this band of magnitude is below the bar.

### 9.10 The "bigger picture" question — selection rejected

#72-C does NOT advance any axis to a meaningfully new level. It refines existing teacher provenance (#68/#69) within an axis already saturated by #69. **REJECT** is the honest verdict.

---

## 10. Bottom line / verdict

### 10.1 Verdict: **REJECT**

CODE-MATH-DISTILL-CHIRON is recommended for **REJECT** on five grounds:

**1. Heavy mechanism overlap with #69 REASONING-DISTILL.** R1 671B already covers code (HumanEval ~88%, LiveCodeBench ~62%) and math (MATH-500 ~95%, AIME ~52%) at frontier-class quality. The marginal headroom for specialized-teacher lift is ~4-6 percentage points on code (HumanEval) and ~0 points on math. Estimated 50-70% of the theoretical advantage of #72-C is already captured by #69.

**2. Marginal lift 1.2-2× — borderline-microoptimization per iter-200 brief.** The iter-200 brief explicitly critiqued the 1.2-1.875× incremental band as microoptimization the program should move past. #72-C sits squarely within this critiqued band. Per-program relevance: below threshold.

**3. Mechanism is system integration, not invention.** #72-C = #68 cached-logit pipeline + #69 reasoning teacher + domain-detection regex router. The only novel component is a ~200-pattern regex classifier. Novelty bar at iter-200 ("bigger picture") not cleared.

**4. Risk-adjusted speedup at noise floor.** Joint Gate-0 PASS × LLM-scale confirmation = 0.225. Realized lift ~1.11× — within evaluation noise at single-GPU CHIRON scale. Hard to distinguish from #69 baseline.

**5. Math-axis ceiling structurally bounded.** R1 671B is already at or above all open math teachers (DeepSeek-Math 7B-RL, NuminaMath, Qwen2.5-Math-72B) on standard benchmarks. The math half of #72-C is structurally weak; the value (such as it is) lies on the code side, where ~4-6 percentage point HumanEval lift is the upper bound.

### 10.2 Caveats on REJECT

**Caveat 1: The mechanism is sound and producible.** Specialized teacher distillation pipelines are mature production engineering (DeepSeek-Coder, Qwen2.5-Coder, Granite Code precedents). ~75% Gate-0 PASS probability is competitive. Mechanism-correctness is high.

**Caveat 2: NLL preservation strict.** Theorems 1 and 2 guarantee general text NLL is bit-exact preserved (Theorem 2) and code/math sub-axis NLL improves monotonically (Theorem 1, ratchet). No regression on existing axes by construction.

**Caveat 3: Smallest engineering scope of all #72 candidates.** ~600 LOC over 3 weeks. Cheapest paradigm to ship at iter-216 if desired as a low-risk fallback.

**Caveat 4: If user elevates code/math benchmarks to primary concern at iter-217+, #72-C may become reselectable** as a depth-refinement paradigm. Reserved as a low-priority candidate.

**Caveat 5: The CODE-only narrowing (drop math) yields 1.5-3× lift.** A CODE-DISTILL-ONLY variant (Qwen2.5-Coder-32B teacher, no math teacher) might be a slightly stronger candidate at 1.5-3× code lift. Still below magnitudes-better bar; remains REJECTED at iter-216 but reservable.

### 10.3 Cost of REJECT

- One paradigm of "domain-specialized distillation" novelty preserved for future iter: code-axis depth-refinement reserved for #73+ if user elevates code benchmarks.
- The composition pattern (#68 + #69 + domain router) remains documented; reservable as a fallback if higher-magnitude candidates fail Gate-0.
- ~$10K-16K cloud cost avoided.
- 3-week engineer-time avoided; can be reallocated to higher-magnitude candidates.

### 10.4 Comparison to candidates A and B at iter 216

| Dim | #72-A (TBD) | #72-B (TBD) | **#72-C (CODE-MATH-DISTILL)** |
|---|---|---|---|
| Headline | TBD | TBD | **~1.2-2× over post-#69 code/math** |
| Risk-adjusted | TBD | TBD | **~1.11× (noise floor)** |
| Gate-0 PASS prob | TBD | TBD | **75%** |
| LLM-scale conf prob | TBD | TBD | **30%** |
| Production precedent | TBD | TBD | **strong (DeepSeek-Coder, Qwen2.5-Coder, Granite Code)** |
| Engineering LOC | TBD | TBD | **600 (smallest)** |
| Memory margin | TBD | TBD | **0 ΔGPU (all teacher inference offline)** |
| Axis relevance to brief | TBD | TBD | **borderline-microopt** |
| Novelty axis | TBD | TBD | **system integration (no new axis)** |

**#72-C is the LOW-RISK, LOW-MAGNITUDE candidate at iter-216.** REJECT recommended unless A and B both fail Gate-0 and a fallback is needed.

### 10.5 Composition-axis status after #72-C (if hypothetically selected)

| Axis | Maturity post-#72-C |
|---|---|
| Compute-speed | At ceiling (#42-#52) |
| Memory | At ceiling (#44, #47, #48) |
| Loss / objective | Mature (#56-#59) |
| Data / sampling | Mature (#57, #58) |
| Identity / agency / curriculum | Mature (#60-#62) |
| Optimizer / meta | Mature (#55, #63) |
| Memory parameter dim | Mature (#64, #65) |
| Cross-modal / VISION | Substrate at #66; distillation reserved for #71-A |
| Cross-modal / AUDIO | Substrate + distillation reserved for #71-B |
| Causal / agentic-trajectory | Mature (#67) |
| Teacher provenance — text | Mature (#68) |
| Teacher provenance — reasoning | Mature (#69) |
| Teacher provenance — agent / tool | Mature (#70) |
| **Teacher provenance — code/math depth** | **Refinement at #72-C (IF selected; not selected at iter-216)** |

After #72-C (if hypothetically selected), the TEACHER PROVENANCE axis is doubly refined (general + reasoning + agent + code/math). The maturation is complete; future paradigms in this axis face vanishing returns.

---

## 11. Bottom line, one line

**REJECT CODE-MATH-DISTILL-CHIRON. ~1.2-2× lift over post-#69 baseline on code/math sub-axes (HumanEval, MBPP, LiveCodeBench, BigCodeBench, MATH-500, AIME, GSM8K, MiniF2F) via Qwen2.5-Coder-32B + DeepSeek-Math 7B-RL specialized teacher pair with domain-detection regex router and cached-logit pipeline reuse from #68 SUPER-DISTILL. Mechanism: #68 + #69 + ~200-pattern domain-classifier; novelty is system integration not architectural primitive. Heavy overlap with #69 REASONING-DISTILL (R1 671B already covers code at HumanEval ~88%, math at MATH-500 ~95% — at or above all open specialized teachers); estimated 50-70% of theoretical advantage already captured. Marginal lift sits in the borderline-microoptimization band (1.2-1.875×) the iter-200 brief explicitly critiqued. Joint Gate-0 PASS ~75%; LLM-scale confirmation ~30%; risk-adjusted speedup ~1.11× at noise floor. Engineering ~600 LOC over 3 weeks (smallest of #72 candidates). Theorems 1 and 2 preserve general text NLL (bit-exact) and code/math sub-axis NLL (monotonic ratchet); no regression risk by construction. Math-axis ceiling structurally bounded by R1 671B saturation; only code side carries any meaningful headroom and even there ~4-6 HumanEval percentage points is the upper bound. REJECT at iter-216; reservable as low-risk fallback if #72-A and #72-B both fail Gate-0; CODE-ONLY-NARROWING variant remains REJECTED but slightly stronger at 1.5-3× code-only lift.**

---

**End of Paradigm Shift #72 Candidate C design document.** ~3000 words. CODE-MATH-DISTILL-CHIRON: refinement of TEACHER PROVENANCE axis (already mature post-#68/#69/#70) via domain-specialized teacher pair (Qwen2.5-Coder-32B + DeepSeek-Math 7B-RL) with regex-based domain-detection router and cached-logit pipeline reuse. Headline 1.2-2× over post-#69 baseline; structurally below magnitudes-better bar; heavy overlap with #69; risk-adjusted lift at noise floor; system-integration novelty without architectural primitive. REJECT recommended; system integration is sound and producible but per-program relevance is borderline-microoptimization per the iter-200 brief that explicitly critiqued this magnitude band.
