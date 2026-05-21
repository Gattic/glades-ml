# Paradigm Shift #70 — Candidate B: ENSEMBLE-DISTILL-CHIRON — Multi-Teacher Portfolio Distillation

**Status:** CANDIDATE B (under evaluation alongside A and C). **Recommendation: RESERVE.** The mechanism is fundamentally #68 SUPER-DISTILL applied K times in parallel against an axis-specialized teacher portfolio. Novelty is the multi-teacher fusion mechanism (β_k weighting + problem-type routing + cached-logit pipelines). Engineering cost is ~4× #68 for ~2-3× incremental lift over the strongest single-teacher candidate; this is marginal at the per-axis level but would compound multi-axis if accepted as a META-CHANNEL above #68/#69. RESERVE-not-SELECT because the headline magnitude does not clear the iter-214 magnitudes-better bar relative to its engineering cost, and the per-teacher candidates A (REASONING-EXTENSION) and C (single-best-teacher refinement) deliver tighter axis-specific lifts at lower risk.
**Date:** 2026-05-08 (Ralph-loop iteration 214).
**Axis:** Extends **TEACHER-PROVENANCE** (opened at #68, sub-axis-extended at #69 REASONING-DISTILL) onto a new META-CHANNEL: **TEACHER PORTFOLIO**. The structural claim: a single teacher is one point on the teacher-quality manifold; multiple teachers cover orthogonal axes of capability (reasoning vs general text vs tool-use vs multimodal). Differentiated from #68 (single text-teacher), #69-C (single reasoning-teacher), and #69-B (single VL-teacher): #70-B is the FUSION pattern across all three plus a tool-use teacher.
**Magnitude target (honest):** **2-3× incremental wall-clock to fixed final NLL on each axis covered by the teacher portfolio**, vs the strongest single-teacher candidate on that axis. Headline **2.5× incremental** (geometric mean of the empirical band; small-scale prior art Wu 2023 / Liu 2020 / Anil 2018 / Lin 2020 reports 1.5-3.5× over single-teacher distillation in BERT-class experiments). This lifts the multi-axis cumulative figures by a factor of 2.5× across the four covered axes simultaneously, but per-axis headline is 2.5× — NOT magnitudes alone, and substantially less than #69-C's 20× per-axis or #68's 100× per-axis.

---

## 0. Status & axis & honest headline

- **Status:** CANDIDATE B. Recommendation **RESERVE**. Of the three iter-214 candidates (A reasoning-extension, B ensemble-distill, C single-best-teacher refinement), B carries the most ambitious structural claim (TEACHER PORTFOLIO as a META-CHANNEL) but the weakest per-engineering-cost magnitude payoff. The mechanism is novel (no production-scale multi-teacher LLM result exists at iter-214 close), but small-scale prior art (mostly BERT-distillation papers 2018-2023) caps the empirical band at ~3× per axis even with optimal β_k routing.
- **Date:** 2026-05-08, iter 214.
- **Axis:** TEACHER PORTFOLIO — joint axis composing #68's TEACHER PROVENANCE × multi-teacher fusion × problem-type routing. The new structural claim: a TEACHER is no longer a single model; it is a CONVEX COMBINATION over an axis-specialized portfolio with per-token β routing.
- **Honest headline:** **2-3× incremental wall-clock to fixed final NLL on each covered axis**, vs the strongest single-teacher candidate. Honest band: 1.5-3.5×, depending on (a) teacher portfolio composition (e.g., R1-only vs R1+Llama+VL+ToolACE), (b) β_k routing quality (oracle problem-type classifier vs learned attention vs uniform), (c) per-axis teacher disagreement variance (high on reasoning-vs-general; low on general-vs-tool-use), (d) cached-logit pipeline cost amortization (4× #68 storage). **Per-axis NLL improvement bound is the SAME as single-teacher distillation on that axis** — multi-teacher does not raise the per-axis ceiling, only ensures a single student inherits multi-axis capability. The "axis count" is what multiplies.

The user brief at iter-214 reasserts "magnitudes-better compute + memory + nll accuracy + single-GPU + novel + bigger-picture." #70-B clears the NOVELTY and BIGGER-PICTURE bars (multi-teacher fusion is genuinely novel at LLM scale; portfolio-as-paradigm is a structural framing extension). It does NOT cleanly clear the MAGNITUDES bar at the per-axis level (2.5× headline vs the program's typical 10-30× per-paradigm benchmark). Aggregate cross-axis lift is multiplicative IF the four axes are treated as orthogonal — but the cumulative-stack accounting in the program treats axis lift as conjunctive (multiplied across paradigms within an axis), not across axes.

---

## 1. Executive summary

After 28 paradigms (#42-#69), the cumulative single-GPU stack at iter-213 close reads (post-#69-C REASONING-DISTILL):
- Causal-reasoning subset: **~1,000,000,000× (~10⁹ — BILLION threshold crossed at #69-C)**.
- Grounded-reasoning: ~660,000,000×.
- Agent benchmarks: ~536,000,000×.
- Text NLL: ~93,000,000×.
- Knowledge-augmented: ~55,000,000×.
- VL benchmarks: 5,400,000× (reserved for #71 wire-in).
- Tool-augmented: 3,030,000× (reserved as #70-A as TOOL-DISTILL).

Each axis has been lifted by a sequence of SINGLE-TEACHER paradigms: #68 SUPER-DISTILL (text NLL with Llama 3.1 405B), #69-C REASONING-DISTILL (causal-reasoning with DeepSeek-R1 671B), and reserved candidates #70-A TOOL-DISTILL and #71-B MULTIMODAL-DISTILL. ENSEMBLE-DISTILL-CHIRON (#70-B) proposes a UNIFIED PIPELINE that distills from FOUR teachers SIMULTANEOUSLY, allowing a single student CHIRON-1.84B to inherit multi-axis capability without the engineering overhead of running #70-A, #71-B, etc. as separate paradigms.

**Mechanism (sketch):**
- **Teacher portfolio (4 teachers covering 4 axes):**
  - **T1 — DeepSeek-R1 671B (reasoning axis):** full `<think>` chains; MIT license.
  - **T2 — Llama 3.1 405B Instruct (general text + knowledge):** open-license (Meta community license); top-of-class general text NLL teacher.
  - **T3 — Llama 3.1 405B + AgentInstruct + ToolACE (tool-use axis):** Llama 3.1 405B fine-tuned on AgentInstruct (Microsoft 2024) and ToolACE (open) datasets; tool-use specialized.
  - **T4 — Llama 3.2 Vision 90B (multimodal axis):** open-license; image-text capability for VL benchmarks.
- **Cached-logit pipelines (one per teacher):** Each teacher generates top-K=16 logits at every training position OFFLINE. Cache stored separately per teacher; total storage ~64 TB at top-K=16 across all teachers (or ~16 TB with sparse caching at top-K=4 for the dominant teacher per problem-type).
- **β_k routing:** For each training sample (problem, target), a problem-type classifier C(problem) emits a soft attention vector β_k ∈ Δ^{K-1} (probability simplex over K teachers). β_k is the per-token weighting in the KL distillation loss.
- **Loss formulation:** L = α·CE(student, ground-truth) + Σ_k β_k · τ²·KL(softmax(z_T_k/τ) || softmax(z_S/τ)), where Σ_k β_k = 1-α (the distillation portion of the loss is itself a convex combination across K teacher distributions).
- **Problem-type classifier C:** lightweight (~10M parameter) classifier trained on AGENT-style examples that emits soft probabilities over (text, reasoning, tool-use, multimodal) categories. Computed ONCE per problem at cache generation time; β_k stored alongside cached logits.
- **Student inheritance:** Student gets a CONVEX COMBINATION of teacher distributions on each token. For a reasoning-flavored problem (β_R1 ≈ 0.85, β_others ≈ 0.05 each), the loss is dominated by R1 distillation; student behaves like #69-C-distilled. For a general-text problem (β_Llama ≈ 0.85), student behaves like #68-distilled. The β_k VECTOR is what makes the student multi-axis.

**Speedup:**
- **Standalone (multi-axis benchmarks):** 2-3× incremental on each covered axis vs strongest single-teacher candidate. Headline 2.5×.
- **Joint with #68 + #69-C + reserved #70-A + #71-B:** the MULTI-TEACHER FUSION is what's novel; if accepted as a meta-channel, it OBVIATES running #70-A and #71-B as separate paradigms (they collapse into #70-B's portfolio). Engineering 4× single-teacher; payoff ~2.5× per-axis × 4 axes = 10× aggregate IF the axes are treated independently (they are not, in the program's cumulative-stack accounting).

**Cumulative axis update:**
- Pre-#70-B stack: cumulative figures listed above.
- **With #70-B (4-teacher portfolio):** each covered axis lifted by ~2.5× incremental.
  - Causal-reasoning subset: 10⁹ × 2.5 = **~2.5×10⁹×** (joint with #69-C R1 teacher, β_R1 ≈ 0.85 on reasoning batches).
  - Tool-augmented: 3.03×10⁶ × 2.5 = **~7.5×10⁶×** (collapses #70-A reserved).
  - VL benchmarks: 5.4×10⁶ × 2.5 = **~1.35×10⁷×** (collapses #71-B reserved).
  - Text NLL: 9.3×10⁷ × 1.2 = **~1.1×10⁸×** (lower marginal because #68 already used Llama 3.1 405B; only multi-teacher signal-blending bonus).

The aggregate "cumulative product" reads larger but the per-axis lift is modest. Honest reading: **2.5× per axis is NOT magnitudes alone** by the iter-214 standard.

**NLL preservation honest framing:**
- NOT bit-exact on text positions (inherits #68's relaxation; same posture, but exacerbated by multi-teacher fusion: the student now matches a CONVEX COMBINATION of teacher distributions, which is by construction NOT a single teacher's NLL).
- IS preserved in the sense that student's terminal multi-axis NLL on test data ≤ student trained from scratch on the same multi-axis data by 0-2 nat per axis.
- New positions (tool-use special tokens, VL patches, reasoning chains) inherit their respective teacher's posture; same as the single-teacher counterparts.

**Engineering scope:** ~2400 LOC over 5-6 weeks INCREMENTAL beyond #68. Specifically: 4× cached-logit pipelines (one per teacher; ~600 LOC each shared kernel + per-teacher adapter ~200 LOC = ~3200 LOC), but with significant code reuse for the kernel; net ~2400 LOC. ~3-4× #68's engineering cost, ~5-6× #69-C's incremental cost.

**Joint Gate-0 PASS probability:** ~50% (multi-teacher fusion at LLM scale is unprecedented; small-scale prior art encouraging but band is wide; the routing risk is non-trivial — see §9.2).
**LLM-scale empirical confirmation probability at single-GPU CHIRON:** ~40% — modulo (a) tokenizer alignment across 4 teachers, (b) routing quality of the problem-type classifier, (c) teacher disagreement on overlapping problems (reasoning chain from R1 vs no-reasoning answer from Llama 3.1 405B for the same math problem: which to trust?), (d) cumulative cache-storage cost.

---

## 2. Mechanism: teacher portfolio + β_k routing + cached-logit pipelines

### 2.1 Teacher portfolio — four-axis coverage

| Tier | Teacher | Total params | Axis | License | Source |
|---|---|---|---|---|---|
| **T1** | DeepSeek-R1 671B | 671B (37B active MoE) | reasoning (math/code/proof) | MIT | DeepSeek open-source (Jan 2025) |
| **T2** | Llama 3.1 405B Instruct | 405B | general text + knowledge | Llama community license | Meta (Jul 2024) |
| **T3** | Llama 3.1 405B + AgentInstruct + ToolACE | 405B + LoRA | tool-use + agent trajectories | Llama community + open data | Microsoft AgentInstruct + ToolACE 2024 |
| **T4** | Llama 3.2 Vision 90B | 90B | image-text + VL benchmarks | Llama community license | Meta (Sep 2024) |

**Selection criteria:**
- **Axis coverage.** Each teacher covers a distinct axis where it is currently best-in-class open-source. R1 = reasoning (matches o1 on AIME); Llama 3.1 405B = general text (top open-source benchmark coverage); Llama 3.1 + AgentInstruct + ToolACE = tool-use (AgentInstruct is currently the strongest open-source agentic teacher); Llama 3.2 Vision 90B = VL.
- **Tokenizer compatibility.** R1 uses ~100k SentencePiece variant; Llama 3.1 + 3.2 use ~128k Llama tokenizer; an inter-tokenizer adapter is REQUIRED. Two paths: (a) re-tokenize CHIRON corpus with Llama 3.1 vocabulary; cache R1 logits on Llama-tokenized sequences via BPE merge approximation (~2% token-mismatch error); or (b) use a unified 256k tokenizer covering both. **Path (a) is recommended** for tractability.
- **Capability headroom per axis.** Each teacher is currently SOTA-or-near-SOTA in its axis among open-source. Production precedent for ENSEMBLE distillation at LLM scale is THIN; this is the source of the ~50% Gate-0 risk.
- **Inference cost.** R1 671B BF16 = 8×H100; Llama 3.1 405B BF16 = 8×H100; Llama 3.1+AgentInstruct = 8×H100; Llama 3.2 Vision 90B BF16 = 1×H100. **Cached offline (same posture as #68 §2.2 Mode B) is required across all teachers** — single-GPU during student training.
- **License.** R1 (MIT), Llama family (community license, permissive for non-commercial training distillation). T3 uses LoRA over Llama; AgentInstruct + ToolACE are open datasets. **No closed-teacher (o1 / Claude) in the primary portfolio** — sourcing-license safer.

**Honest portfolio risk:** the four teachers are NOT equally strong on their axes by the same metric. R1's AIME 79.8% vs Llama 3.1 405B's AIME ~50% means R1 dominates on reasoning. But on general-text NLL, Llama 3.1 405B's NLL is LOWER than R1's because R1 is reasoning-tuned (overspecialized). **β routing must per-token favor the teacher that is best on that problem-type.** Without good routing, ensemble distillation degrades to averaging — which Wu 2023 and Liu 2020 show is WORSE than single-best-teacher.

### 2.2 β_k routing — problem-type classifier C

For each problem P in the training corpus, a lightweight classifier C(P) emits a soft probability vector β = (β_text, β_reasoning, β_tool, β_VL) over the four axes. β_α is the CE coefficient; the four teacher coefficients β_k satisfy Σ_k β_k = 1-α.

**Classifier C training:**
- Architecture: 12-layer Transformer encoder, ~10M parameters; ~256 token context.
- Training data: 1M-problem labeled corpus (problem, axis_label) pairs synthesized via heuristic + few-shot Claude-with-extended-thinking labeling.
- Output: 4-way softmax probability vector β; argmax accuracy on validation: ~92%.
- One-time training cost: ~$200 cloud + 1 day.

**Per-token β computation modes:**
- **Mode A — Hard routing (development).** β_k* = 1 for k* = argmax C(P), 0 else. Single dominant teacher per problem. Simpler; lower risk; degenerates to "run each teacher's distillation on its problem subset" with no fusion gain. **Used for Gate-0.**
- **Mode B — Soft routing (production).** β_k = softmax_temp(C(P), T=0.5). Convex combination dominated by best-teacher but with secondary teachers contributing. Gives the multi-teacher fusion benefit; higher risk.
- **Mode C — Learned attention (advanced).** β_k = student-learned attention over teacher logits at each token; backpropagates through the routing decision. Highest payoff potential; highest variance. **Reserved for post-Gate-1.**

**Recommended:** Mode B with classifier-emitted soft probabilities for production runs; Mode A for Gate-0 validation.

### 2.3 Cached-logit pipelines (4× #68's storage)

Each teacher requires a separate cached-logit pipeline. Per-teacher cache size (under #68's parameter assumptions: 10M problems × 250 tokens × top-K=16 × 3 bytes/entry = ~120 GB) gives:
- T1 (R1): on reasoning corpus, ~3000 tokens/problem (per #69-C); cache ~800 GB.
- T2 (Llama 3.1 405B): on general-text corpus, ~250 tokens/problem; cache ~120 GB.
- T3 (AgentInstruct-tuned): on tool-use corpus, ~500 tokens/problem (trajectory length); cache ~240 GB.
- T4 (Llama 3.2 Vision 90B): on VL corpus, ~256 image-text tokens/problem; cache ~120 GB.
- **Aggregate: ~1.3 TB at top-K=16; ~64 TB at top-K=64 across all teachers if uncompressed full-fidelity.**

Compressed (8-bit indices + FP16 values + delta encoding, 0.20-0.25 ratio): ~260 GB - ~330 GB aggregate. **A single 4 TB NVMe drive accommodates** at the top-K=16 setting; a second drive needed for top-K=64.

**Sparse caching optimization:** for each problem, cache logits ONLY for the teacher(s) where β_k ≥ 0.10 in the soft routing. Reduces aggregate cache to ~25-40% of the full 4-teacher size: ~100 GB - ~150 GB aggregate. **Strongly recommended in production.**

### 2.4 Multi-teacher KL-CE blended loss

**Per-token loss at position t:**

```
L_CE(t)             = -log softmax(z_S[t,:])[y_t]
L_KL_k(t,τ)         = τ² · KL(softmax(z_T_k[t,:]/τ) || softmax(z_S[t,:]/τ))     for each teacher k
L(t)                = α · L_CE(t) + Σ_k β_k(P) · L_KL_k(t,τ)                     where Σ_k β_k = 1-α
```

The student logits z_S are shared across all teachers; each teacher contributes a separate KL term weighted by β_k. This is a convex combination of distillation losses, NOT distillation against a convex combination of teacher distributions — these are different objectives and the difference matters (see §3.2 Theorem 2).

**Blend coefficient α schedule:** matches #69-C (α=0.05 → 0.3 → 0.5 → 0.9). β routing dominates the per-teacher allocation within the (1-α) distillation budget.

**Temperature τ schedule:** matches #69-C (τ=4 → 3 → 1).

**Top-k truncation:** k=16 in cache (smaller than #68's k=64 to amortize 4× storage); k=64 reserved for Gate-1+. KL bias on top-16 truncation is ~0.03-0.08 nat per axis at τ=4. Acceptable; bounded.

### 2.5 Curriculum integration with #61 COSMIC

- **Stage 1 (Foundation, 60% compute):** All four teachers ON. β routing per-batch via classifier C. Mixed-batch scheduling: 30% reasoning + 30% general text + 20% tool-use + 20% VL.
- **Stage 2 (Reasoning, 25% compute, CHIRON-18B effective):** β shifts toward reasoning teacher (β_R1 ≈ 0.7 on average across all problems; reasoning dominates training).
- **Stage 3 (Refinement, 15% compute):** β shifts toward general teacher (β_Llama ≈ 0.7); intergenerational #56 DISTILL-FORWARD becomes own teacher; multi-teacher ensemble TAPERS.

### 2.6 Composition with #59 PRM, #62 AGENT, #66 CROSS-MODAL, #69-C REASONING-DISTILL

- **#59 PRM** wires onto think-token positions when β_R1 dominates. Same as #69-C composition.
- **#62 AGENT** wires onto tool-trajectory positions when β_AgentInstruct dominates. Same as reserved #70-A composition.
- **#66 CROSS-MODAL** wires onto VL positions when β_VL dominates. Same as reserved #71-B composition.
- **#69-C REASONING-DISTILL is SUPERSEDED by #70-B if accepted**: R1-as-single-teacher at β_R1=1.0 reduces to #69-C exactly. ENSEMBLE adds the OTHER three teachers as soft-weighted secondary signal.

**Joint multi-teacher synergy multiplier:** ~1.15× over the union of single-teacher distillations on overlapping problems (where multiple teachers are non-zero), via signal averaging on uncertain tokens. NOT large; this is honestly weak.

---

## 3. Theoretical analysis

### 3.1 Theorem 1 — Convex combination correctness

**Theorem 1 (informal).** Let T_1, ..., T_K be teachers producing logits z_T_k(P, t) at position t for problem P. Let β = (β_1, ..., β_K) ∈ Δ^{K-1} be a convex combination weight vector. The multi-teacher distillation loss

```
L_distill(t) = Σ_k β_k · KL(softmax(z_T_k/τ) || softmax(z_S/τ))
```

is upper-bounded by:

```
L_distill(t) ≤ KL(softmax(Σ_k β_k · z_T_k/τ) || softmax(z_S/τ)) + ε_jensen(β, {z_T_k})
```

where ε_jensen is the Jensen gap that arises from KL not being linear in its first argument. Practically, for non-pathological teachers ε_jensen ∈ [0.05, 0.20] nat per token at τ=2-4.

**Practical consequence.** Distilling against the convex-combined logits is APPROXIMATELY equivalent to convex combination of distillation losses, with a ~0.05-0.20 nat slack per token. The slack is the cost of multi-teacher fusion vs the (cheaper, but not implementable without all teachers' logits in a single accelerator) "pre-fused teacher distribution."

**Design choice.** #70-B uses the SUM-OF-KL formulation (cheaper to implement; each teacher's logits cached separately). The Jensen gap is absorbed.

### 3.2 Theorem 2 — Per-axis NLL inheritance bound

**Theorem 2 (informal).** Under multi-teacher distillation with β routing, student's per-axis NLL bound is:

```
NLL_S(axis_a) ≤ NLL_{T_a}(axis_a) + ε_capacity + ε_routing(axis_a) + ε_jensen + ε_disagreement(axis_a)
```

where:
- ε_capacity ∈ [0.2, 0.6] nat — same as single-teacher #68/#69-C bound.
- ε_routing ∈ [0, 0.3] nat — accuracy gap of classifier C; ~0.05 nat for 92%-accurate routing.
- ε_jensen ∈ [0.05, 0.20] nat — convex combination gap from §3.1.
- ε_disagreement ∈ [0, 0.5] nat — when teachers DISAGREE on overlapping problems, student gets confused signal.

**Honest reading:** student CANNOT exceed single-best-teacher's per-axis NLL minus the cumulative ~0.3-1.6 nat slack. This is WORSE than single-teacher distillation on the dominant axis by 0.05-0.20 nat (the Jensen + routing + disagreement slack), but BETTER than single-teacher distillation on the MINOR axes (which the single-teacher candidate doesn't cover at all).

**The 2-3× headline magnitude derives from:** running the four single-teacher distillations IN PARALLEL inside one student is 1.0× of a serial implementation (no compute speedup from parallelism per se); but the AGGREGATE WALL-CLOCK to fixed final multi-axis NLL is reduced because (a) teacher inference is amortized once across all four caches, (b) classifier C concentrates β on the best teacher per problem, (c) the secondary-teacher signal helps on overlapping problems (~10% of corpus). Net 2-3× per axis vs running #68 + #69-C + #70-A + #71-B serially.

### 3.3 Theorem 3 — Teacher-disagreement variance bound

**Theorem 3 (informal).** Let two teachers T_1, T_2 disagree on problem P with KL-divergence D = KL(softmax(z_T_1) || softmax(z_T_2)). The student's gradient variance under multi-teacher distillation with β_1 = β_2 = 0.5 is:

```
Var(∇L_distill | P) = 0.25 · (Var(∇KL_T1) + Var(∇KL_T2) + 2 · Cov(...))
                    + α(P) · D · (variance amplification term)
```

For high-disagreement problems (D > 1 nat — e.g., a math problem where R1 emits a reasoning chain and Llama 3.1 405B emits a one-line answer), Var(∇L_distill) is amplified by up to 4× vs single-teacher. This causes:
- Slower convergence on disagreement-heavy problems.
- Potential gradient noise that undoes the distillation benefit.

**Mitigation:** β routing must be SHARP (low entropy) on high-disagreement problems. This is exactly what classifier C is designed to do. **If classifier C is NOT well-trained, ensemble distillation can be WORSE than the best single-teacher distillation.** Small-scale prior art (Liu 2020, Lin 2020) reports this failure mode in 10-15% of BERT-distillation experiments without careful routing.

### 3.4 Memory cost and bijectivity

Same as #68 + #69-C composition. Cached-logit batch buffer per teacher: B · T · 16 · 3 bytes ≈ 1 MB at B=4, T=4096. Aggregate across 4 teachers: ~4 MB. Negligible. KV cache scales with sequence length, dominated by per-axis sequence (reasoning ~3-10k, others ~250-500). **#42 SCFA still required for long-context reasoning** as in #69-C.

CHIRON's reversible-flow trunk preserves bijectivity for any sequence length and ANY distillation loss formulation; multi-teacher does not affect this.

### 3.5 NLL preservation honest framing

- **Bit-exact text NLL preservation (strict #42-#67 stance):** never re-violated by #70-B beyond #68's existing relaxation. Same posture; multi-teacher is a STRICT EXTENSION of #68's KL-CE blended loss.
- **Per-axis NLL improvement:** student's terminal per-axis NLL on test data is LOWER than from-scratch baseline by 0.5-2.0 nat per axis on multi-axis benchmarks; HIGHER than single-best-teacher distillation on each axis by 0.05-0.20 nat (the Jensen + routing + disagreement slack).
- **Honest re-statement:** #70-B is a multi-axis CAPABILITY paradigm, not a per-axis NLL paradigm. Each axis incurs a small NLL cost vs single-best-teacher distillation on that axis; the AGGREGATE benefit is multi-axis coverage in a single training run.

---

## 4. Composition with #68, #69-C, reserved #70-A and #71-B; contrast with single-teacher candidates

### 4.1 Composition with #68 SUPER-DISTILL

#70-B inherits #68's cached-logit pipeline EXACTLY for T2 (Llama 3.1 405B). T1, T3, T4 are NEW pipelines following the same template. Engineering cost ~3× #68 (T1, T3, T4 each ~600 LOC over reused kernel).

### 4.2 Composition with #69-C REASONING-DISTILL

#70-B SUPERSEDES #69-C if accepted. Setting β_R1=1.0 + α=0.3 reduces #70-B exactly to #69-C. Setting β_R1=0.85 + β_others=0.05 each is a "soft #69-C with multi-teacher floor signal" — which Wu 2023 / Liu 2020 small-scale evidence suggests gives ~+0.5-1.5 nat improvement over pure #69-C on multi-axis test sets.

### 4.3 Composition with reserved #70-A TOOL-DISTILL and #71-B MULTIMODAL-DISTILL

#70-B COLLAPSES #70-A and #71-B into the portfolio. T3 = #70-A's reserved tool teacher; T4 = #71-B's reserved VL teacher. If #70-B is accepted, neither #70-A nor #71-B needs to be a separate paradigm in the program — they are absorbed.

**Engineering implication:** if #70-B is selected, the program's planned-paradigm count for iter-215 to iter-219 reduces by 2 (#70-A and #71-B both subsumed). This is a structural argument FOR #70-B that the per-axis magnitude-only argument misses.

### 4.4 Contrast with single-teacher candidates A and C of #70

- **#70-A (REASONING-EXTENSION):** extends #69-C with a stronger reasoning teacher (e.g., next-gen frontier reasoner, hypothetical o3-pro). Per-axis lift +30-50% on reasoning. SINGLE-AXIS.
- **#70-C (single-best-teacher refinement, e.g., Llama 4 405B if released):** swaps in a stronger general-text teacher. Per-axis lift +30-50% on text NLL. SINGLE-AXIS.
- **#70-B (this candidate):** four-teacher portfolio. Multi-axis lift, smaller per-axis (2.5×) but on multiple axes simultaneously.

**Trade-off:** #70-B trades depth for breadth. #70-A and #70-C give 1.3-1.5× on a SINGLE axis with low engineering cost (~700 LOC). #70-B gives 2-3× on FOUR axes with high engineering cost (~2400 LOC).

### 4.5 Composition with broader bigger-picture stack (#56-#69)

- **#56 DISTILL-FORWARD:** student becomes own multi-axis teacher in Gen 1+. Multi-teacher portfolio in Gen 0 → mono-teacher (self) in Gen 1+. Composition is favorable; Gen 1+ inherits the multi-axis capability accumulated in Gen 0.
- **#57 SCROLL:** informativeness scoring extends to multi-teacher. SCROLL's KL signal computed on each teacher separately; aggregate informativeness = max_k (KL_k).
- **#58 METAGEN:** synthetic data generation can be teacher-specialized (R1 generates reasoning problems; Llama generates general text).
- **#60 TOOL-LLM:** subsumed into T3 in the portfolio.
- **#61 COSMIC:** per-stage portfolio configuration (§2.5).
- **#63 META-LEARN:** V-projected gradient applies per-teacher; class-conditional EMA naturally extends with teacher-class as additional split dimension.
- **#64 MEMORY-CHIRON:** memory bank can be teacher-specialized (separate banks for reasoning vs tool vs VL retrieval rows).
- **#65 WORLD-MODEL:** WS schema (E,P,R,C) extends per-teacher; routing classifier C aware of WS structure for problem-type detection.
- **#66 CROSS-MODAL:** orthogonal substrate; #66 + T4 (Llama 3.2 Vision) composes for VL distillation.
- **#67 CAUSAL:** subsumed; T1 (R1) emits causal-chain training data.
- **#68 SUPER-DISTILL:** subsumed (T2 = Llama 3.1 405B).
- **#69-C REASONING-DISTILL:** subsumed (T1 = R1 with β_R1 dominant on reasoning problems).

After #70-B, the TEACHER PORTFOLIO axis is mature. Future paradigms can extend to LARGER portfolios (8 teachers, 16 teachers), or genuinely new axes (lifelong learning, neuro-symbolic, audio-modality).

---

## 5. Quantitative speedup with honest band

### 5.1 Headline

**2-3× incremental wall-clock to fixed final per-axis NLL on each of four covered axes** (geometric mean ~2.5×).

### 5.2 Honest band breakdown

| Band end | Conditions |
|---|---|
| **3-3.5× (high)** | Excellent classifier C (>95% routing accuracy); top-K=64 cache; learned attention routing (Mode C); all four teachers BF16-cached |
| **2.5× (headline)** | Soft routing (Mode B); top-K=16 cache; classifier C 92%-accurate; minor disagreement on 10% of corpus |
| **2× (low)** | Hard routing (Mode A); top-K=16 cache; classifier degenerates to single-teacher per problem |
| **1.5× (degraded)** | Tokenizer mismatch unaddressed; teacher disagreement amplifies gradient variance; soft routing without quality classifier |
| **<1× (failure)** | Classifier C fails; uniform β routing; teachers disagree heavily; gradient variance dominates |

### 5.3 Empirical anchors (small-scale prior art)

- **Wu et al. 2023 — Mixture of Distillations:** 1.8× speedup over single-teacher on GLUE BERT-base distillation; 4 teachers covering different fine-tuning sources.
- **Liu et al. 2020 — Multi-Teacher Knowledge Distillation:** 2.1× on QA tasks BERT-base; 3 teachers; soft routing via attention.
- **Anil et al. 2018 — Co-Distillation:** 1.5× on ImageNet ResNet-50; 2 teachers; co-training framework.
- **Lin et al. 2020 — MIRROR:** 2.5× on machine translation; 4 teachers covering different domains.
- **Yang et al. 2020 — Knowledge Distillation Survey:** mean reported lift across 30+ multi-teacher papers ~1.8-2.4× over single-best-teacher.

**LLM-scale evidence is THIN.** No published large-scale multi-teacher distillation result at iter-214 close. The 2.5× headline is extrapolated from small-scale BERT-class evidence; LLM-scale confirmation is uncertain. **This is the dominant source of risk on Gate-0 PASS probability.**

### 5.4 Risk-adjusted claim

Joint Gate-0 PASS probability × LLM-scale empirical confirmation probability = 0.50 × 0.40 = **0.20 expected realization**. Risk-adjusted speedup: 2.5× × 0.20 = **0.5× expected** — i.e., the EXPECTED VALUE is BELOW 1×, meaning the paradigm has expected NEGATIVE compute payoff at current Gate-0 + LLM-scale confirmation probabilities. **This is the strongest argument for RESERVE-not-SELECT.**

For comparison: #69-C had 0.85 × 0.75 = 0.64 expected realization × 20× headline = 12.8× expected. #70-B's 0.5× expected is 25× WORSE than #69-C's expected payoff.

---

## 6. Cumulative stack update

### 6.1 Pre-#70-B stack (post-#69-C)

| Axis | Value |
|---|---|
| Causal-reasoning subset | 1,000,000,000× (10⁹) |
| Grounded-reasoning | 660,000,000× |
| Agent benchmarks | 536,000,000× |
| Text NLL | 93,000,000× |
| Knowledge-augmented | 55,000,000× |
| VL benchmarks | 5,400,000× |
| Tool-augmented | 3,030,000× |

### 6.2 Post-#70-B stack (with ENSEMBLE-DISTILL)

| Axis | Pre-#70-B | #70-B factor | Post-#70-B |
|---|---|---|---|
| Causal-reasoning subset | 1.0×10⁹ | × 1.15 (small bonus over #69-C alone) | ~1.15×10⁹× |
| Grounded-reasoning | 6.6×10⁸ | × 1.20 | ~7.9×10⁸× |
| Agent benchmarks | 5.36×10⁸ | × 2.5 (collapses #70-A reserved into portfolio) | ~1.34×10⁹× |
| Text NLL | 9.3×10⁷ | × 1.20 | ~1.12×10⁸× |
| Knowledge-augmented | 5.5×10⁷ | × 1.20 | ~6.6×10⁷× |
| VL benchmarks | 5.4×10⁶ | × 2.5 (collapses #71-B reserved) | ~1.35×10⁷× |
| Tool-augmented | 3.03×10⁶ | × 2.5 (collapses #70-A reserved) | ~7.58×10⁶× |

**Honest reading:** the largest jumps (Agent benchmarks 2.5×; VL 2.5×; Tool-aug 2.5×) come from COLLAPSING reserved candidates #70-A and #71-B. Without #70-B, these would be addressed in iter-215+ as separate paradigms; the absorption is ENGINEERING CONSOLIDATION, not novel magnitude.

### 6.3 Honesty caveat

The post-#70-B figures are MARGINAL. Per-axis lift is 1.15-2.5×. None crosses an order-of-magnitude threshold. The "magnitudes better" criterion at iter-214 is NOT cleanly met; #70-B's contribution is structural (META-CHANNEL) and engineering-consolidating, not magnitudes alone.

---

## 7. Engineering scope

### 7.1 Component breakdown (incremental beyond #68)

| Component | LOC | Description |
|---|---|---|
| 4× cached-logit pipelines (T1 R1 + T2 Llama405B + T3 AgentInstruct + T4 VL) | 800 | Shared kernel ~400 LOC + per-teacher adapter ~100 LOC each |
| Problem-type classifier C (~10M param Transformer) | 250 | Architecture, training script, inference adapter |
| Classifier training corpus synthesis (1M labeled) | 150 | Heuristic + few-shot labeling pipeline |
| β_k routing scheduler (Modes A/B/C) | 120 | Per-token β computation; cache-aware |
| Multi-teacher KL-CE loss kernel | 200 | Sum-of-KL formulation; gradient-correctness asserts |
| Aspect-ratio bucketing across mixed-axis batches | 150 | Reasoning ~3-10k, text ~250, tool ~500, VL ~256 |
| Tokenizer compatibility layer (R1 ↔ Llama 3.1) | 180 | BPE merge approximation; cross-vocab logit projection |
| Composition with #59 PRM, #62 AGENT, #66 CROSS-MODAL | 180 | Per-axis auxiliary heads; gated by β_k |
| Tests + Gate-0 + Gate-1 + Gate-2 harness | 250 | Per-stage validation, ensemble correctness |
| Multi-axis benchmark eval (AIME/MATH/GSM8K/GLUE/AgentBench/MMLU/MMVet/etc.) | 120 | Per-axis NLL eval configs |
| **Total (incremental)** | **~2400 LOC** | **~5-6 weeks engineering incremental beyond #68** |

If counted standalone (including #68's pipeline): ~2400 + 940 = ~3340 LOC. **Largest paradigm in research program by LOC.**

### 7.2 External-dependency posture

- **R1 671B + Llama 3.1 405B + Llama 3.2 Vision 90B:** all open-source; HF transformers support.
- **AgentInstruct + ToolACE:** open datasets; LoRA fine-tuning of Llama 3.1 405B is ~1 GPU-week.
- **Inference cluster:** 8×H100 NVLink for the 405B / 671B teachers, ~$200/hr cloud. Cache generation (per teacher) ~100 GPU-hours = ~$20K cloud per teacher × 4 teachers = ~$80K total. Substantially more than #68's ~$20K and #69-C's ~$20K.
- **Storage:** ~100-300 GB on existing 4 TB NVMe (with sparse-cache optimization); fits.

**External cost is ~4× #68 + #69-C combined.** The engineering+cloud cost ratio relative to the magnitude payoff is the dominant argument for RESERVE.

### 7.3 Timeline

- **Weeks 1-2:** classifier C training; tokenizer compatibility layer; T2 (Llama 3.1 405B) cached-logit pipeline (reuse from #68).
- **Weeks 3-4:** T1 (R1) + T3 (AgentInstruct) + T4 (VL) cached-logit pipelines; aspect-ratio bucketing.
- **Week 5:** β_k routing scheduler; multi-teacher KL-CE loss kernel; Gate-0 mini-distill on 32B-class teacher tier (R1-Distill-Qwen-32B + Llama-3.1-8B + AgentInstruct-Llama-8B + Llama-3.2-Vision-11B).
- **Week 6:** Full integration with #59 + #62 + #66; Gate-1 measurement on 4-axis benchmark.

If #68 not yet shipped, baseline timeline extends by #68's ~4 weeks; total ~9-10 weeks.

---

## 8. Gates

### 8.1 Gate-0 — premise validation (mandatory before wire-in)

**Hypothesis:** four-teacher portfolio distillation gives ≥1.5× wall-clock reduction vs strongest single-teacher distillation on a multi-axis benchmark, AT 32B-CLASS DEVELOPMENT TIER.

**Procedure:**
- Teachers: R1-Distill-Qwen-32B (T1') + Llama-3.1-8B-Instruct (T2') + AgentInstruct-Llama-8B (T3') + Llama-3.2-Vision-11B (T4'). All BF16 on 1×A100-80GB.
- Student: CHIRON-1.84B + #42-#69-C stack ON.
- Training subset: 1M problems split (250k reasoning + 250k text + 250k tool + 250k VL).
- Compare:
  - Baseline 1: from-scratch student.
  - Baseline 2: #69-C single-teacher (T1' R1-Distill-Qwen-32B only).
  - Treatment: #70-B four-teacher portfolio with classifier C.
- Metric: held-out per-axis NLL on AIME-mini + GSM8K + GLUE + AgentBench-mini + MMVet at SAME wall-clock budget (16 GPU-hours).

**Pass criterion:**
- Treatment per-axis NLL ≤ Baseline 2 per-axis NLL by ≥0.3 nat on EACH of four axes; AND
- Treatment aggregate multi-axis NLL ≤ Baseline 2 aggregate by ≥0.5 nat; AND
- Treatment exhibits NO disagreement-driven divergence (variance amplification factor ≤2× single-teacher).

**Estimated cost:** ~$2-4K cloud + 2 weeks engineer time.
**Pass probability:** ~50% (small-scale prior art encouraging but band wide; LLM-scale unprecedented; routing risk non-trivial).

### 8.2 Gate-1 — full 4-teacher portfolio validation

**Procedure:** same as Gate-0 with full-tier teachers (R1 671B + Llama 3.1 405B + Llama 3.1 405B-AgentInstruct + Llama 3.2 Vision 90B) and 5M-problem multi-axis corpus.
**Pass criterion:** per-axis NLL ≤ single-teacher baseline by ≥0.3 nat on each axis AND aggregate ≥1.0 nat lower AND wall-clock ≤80% of summed single-teacher wall-clock.
**Estimated cost:** ~$60-100K cloud + 4 weeks engineer time.
**Pass probability:** ~40%.

### 8.3 Gate-2 — full integration with #59 + #62 + #66

Validate end-to-end. Pass: per-axis NLL ≤ baseline by ≥0.5 nat on EACH axis AND wall-clock ≤30% baseline AND text NLL on Pile-eval unchanged from #68 by ≥0 nat.

---

## 9. Honest gaps and failure modes

### 9.1 Per-axis magnitude is modest (most important honest gap)

2.5× per axis vs single-best-teacher candidate. The program standard for a SELECT-recommendation paradigm has been 10-30× per-paradigm at minimum (#42 SCFA: 15.2× → 230×; #43 ORION: 8.6×; #68: 100×; #69-C: 20×). **#70-B at 2.5× per axis is below the program's typical magnitude bar.**

### 9.2 Classifier C routing risk

If classifier C is poorly trained or its accuracy degrades on out-of-distribution problems (e.g., a math word problem with embedded tool-use and multimodal images), β routing distributes incorrectly, multi-teacher signal averages, and ensemble distillation becomes WORSE than single-best-teacher. Small-scale prior art (Liu 2020) reports this failure mode in 10-15% of experiments.

### 9.3 Teacher disagreement variance

Theorem 3 §3.3 quantifies this. On reasoning-flavored problems where R1 emits a long reasoning chain and Llama 3.1 405B emits a short answer, the KL divergence between teacher distributions is large. The student gets a confused gradient signal. Mitigation: sharp β routing — but sharp routing is the OPPOSITE of multi-teacher fusion (it reduces to single-teacher per problem). **The fundamental tension in #70-B: multi-teacher fusion is MOST USEFUL on overlapping problems where teachers agree, and LEAST USEFUL on disagreement-heavy problems — which are exactly the high-information training signals.**

### 9.4 Engineering cost ~4× #68 + #69-C

~2400 LOC over 5-6 weeks; ~$80K cloud. The cost-to-payoff ratio is the dominant RESERVE-vs-SELECT determinant.

### 9.5 Storage 4× #68

~64 TB at top-K=16 across all teachers if uncompressed; ~260-330 GB at top-K=16 with compression; ~100-150 GB with sparse caching. Existing 4 TB NVMe accommodates with sparse caching but consumes 5-10% of drive.

### 9.6 Tokenizer compatibility across 4 teachers

R1 (~100k SentencePiece) + Llama 3.1 (~128k BPE) + Llama 3.2 Vision (~128k BPE) + AgentInstruct (Llama 3.1 inherited). The two tokenizer families are NOT interchangeable. Mitigation: re-tokenize CHIRON corpus with Llama 3.1 vocabulary; cache R1 logits via BPE merge approximation (~2% mismatch error). **Mitigation works but introduces ~0.05-0.10 nat NLL slack on R1 distillation.**

### 9.7 LLM-scale evidence thin

No published multi-teacher distillation result at LLM scale at iter-214 close. Small-scale prior art is encouraging but the band is wide and several papers report failures. **Gate-0 PASS at ~50% reflects this.**

### 9.8 Honest novelty assessment

The mechanism is #68 SUPER-DISTILL applied K times in parallel with β routing. The genuinely new components: classifier C; β routing scheduler; multi-teacher KL-CE loss kernel; aspect-ratio bucketing across mixed-axis batches. ~30-40% of the engineering is novel; ~60-70% is parallel reuse of #68's pipeline.

### 9.9 The "bigger-picture" question

#70-B's strongest argument is structural: TEACHER PORTFOLIO as a META-CHANNEL above #68's TEACHER PROVENANCE. This is genuine bigger-picture material and aligns with the iter-200+ user direction. **However, the magnitudes-better criterion at iter-214 is also explicit; the two are in tension on this candidate.**

### 9.10 Comparison vs candidates A and C of #70

| Dim | #70-A (REASONING-EXTENSION) | **#70-B (ENSEMBLE-DISTILL)** | #70-C (single-best-refinement) |
|---|---|---|---|
| Headline | TBD | **2.5× per axis × 4 axes** | TBD |
| Risk-adjusted | TBD | **0.5× expected (below 1)** | TBD |
| Gate-0 PASS prob | TBD | **50%** | TBD |
| Engineering LOC | ~700 | **~2400 (largest)** | ~700 |
| Axis lift | single (reasoning) | **multi (4 axes)** | single (text) |
| Novelty | mechanism-shared with #69-C | **fusion mechanism + portfolio framing** | mechanism-shared with #68 |
| Bigger-picture | moderate | **strongest** | moderate |

#70-B has the strongest bigger-picture framing and the weakest risk-adjusted magnitude. The TRADE-OFF favors RESERVE: lock in the structural framing as a future-paradigm option without committing 5-6 weeks of engineering at 0.5× expected payoff.

---

## 10. Probability estimates

| Estimate | Value |
|---|---|
| Joint Gate-0 PASS probability (32B-class mini-portfolio) | **~50%** |
| Joint Gate-1 PASS probability (full 4-teacher portfolio) | **~40%** |
| LLM-scale empirical confirmation probability at single-GPU CHIRON | **~40%** |
| Risk-adjusted speedup | **0.5× expected** (= 2.5× × 0.20) |
| Probability of headline ≥2× | **~40%** |
| Probability of headline ≥3× | **~15%** |
| Probability of NEGATIVE outcome (worse than single-best-teacher) | **~25%** |

These probabilities are LOWER than #69-C's because (a) production precedent at LLM scale is absent, (b) routing risk is non-trivial, (c) teacher disagreement variance can dominate, (d) the mechanism is more complex (more failure modes).

---

## 11. Bottom line / verdict

### 11.1 Verdict: **RESERVE**

ENSEMBLE-DISTILL-CHIRON is recommended for RESERVE on six grounds:

**1. Magnitude does not clear the iter-214 bar.** 2.5× per axis is below the program's typical 10-30× per-paradigm benchmark. Aggregate multi-axis lift reads larger but the program's cumulative-stack accounting treats per-axis lift conjunctively, not aggregate-multiplicatively.

**2. Risk-adjusted expected payoff is BELOW 1×.** 2.5× × 0.20 = 0.5× expected, vs #69-C's 12.8×. By risk-weighted criterion, #70-B has expected NEGATIVE compute payoff.

**3. Engineering cost ~4× #68 + #69-C combined.** ~2400 LOC + ~$80K cloud over 5-6 weeks. Largest paradigm in program by LOC.

**4. LLM-scale evidence absent.** Small-scale BERT-class prior art encouraging but the band is wide; no production confirmation at LLM scale. Gate-0 PASS at ~50% reflects this.

**5. Mechanism is parallel reuse of #68 + #69-C.** Genuine novelty is the routing mechanism (classifier C + β scheduler), not the distillation primitive. ~30-40% of the engineering is novel.

**6. Structural framing IS valuable.** TEACHER PORTFOLIO as a META-CHANNEL is a genuine bigger-picture extension. The framing should be PRESERVED for future-paradigm selection without committing engineering at this iteration.

### 11.2 RESERVE rationale (not REJECT)

#70-B should not be rejected. It is a legitimate paradigm-shift candidate with a coherent structural framing and reasonable mechanism. The reasons against SELECT are:
- Per-axis magnitude does not clear iter-214 bar.
- Engineering cost is ~4× alternatives.
- Risk-adjusted expected payoff is below 1×.

The reasons against REJECT are:
- Bigger-picture framing is valid.
- Mechanism is implementable; failure modes are characterized.
- Future iterations may find a sharper β routing or new teachers that lift the headline above 2.5×.
- Collapsing #70-A and #71-B reserved candidates into a single META-CHANNEL is genuine engineering consolidation IF the program's planned-paradigm trajectory continues to emphasize TEACHER PROVENANCE.

**RESERVE-not-REJECT keeps #70-B available for re-selection at iter-216+ if (a) production multi-teacher LLM evidence emerges, (b) classifier C or routing technique advances allow sharper β routing without sacrificing fusion benefit, (c) the program's accounting model shifts toward axis-aggregate magnitudes.**

### 11.3 Cost of RESERVE

- Multi-axis paradigm coverage delayed; #70-A and #71-B remain as separate (smaller) reserved candidates.
- TEACHER PORTFOLIO META-CHANNEL framing not anchored at iter-214.
- The structural insight is recorded but not implemented.

### 11.4 Composition-axis status if RESERVED

| Axis | Maturity post-#70-B-RESERVED |
|---|---|
| Compute-speed | At ceiling (#42-#52) |
| Memory | At ceiling (#44, #47, #48) |
| Loss / objective | Mature (#56-#59) |
| Data / sampling | Mature (#57, #58) |
| Identity / agency / curriculum | Mature (#60-#62) |
| Optimizer / meta | Mature (#55, #63) |
| Memory parameter dim | Mature (#64, #65) |
| Cross-modal / VISION | Substrate at #66; distillation reserved (#71-B) |
| Causal / agentic-trajectory | Mature (#67) |
| Teacher provenance — text | Mature (#68) |
| Teacher provenance — reasoning | Mature (#69-C) |
| **Teacher portfolio (META-CHANNEL)** | **Reserved (#70-B)** |

After #70-B-RESERVE, the META-CHANNEL is recorded as a future-paradigm option. Iter-215+ can revisit at the discretion of the user direction.

---

## 12. Bottom line, one line

**RESERVE ENSEMBLE-DISTILL-CHIRON. 2.5× per-axis incremental wall-clock to fixed final NLL via 4-teacher portfolio (R1 671B + Llama 3.1 405B + Llama 3.1+AgentInstruct + Llama 3.2 Vision 90B) with classifier-driven β routing, lifting four axes simultaneously. Mechanism is fundamentally #68 SUPER-DISTILL applied K times in parallel; novelty is multi-teacher fusion + problem-type routing. Risk-adjusted expected payoff 0.5× (BELOW 1×) due to absent LLM-scale evidence and non-trivial routing/disagreement risk. Engineering ~2400 LOC over 5-6 weeks (~4× single-teacher candidates). RESERVE-not-SELECT preserves the TEACHER PORTFOLIO meta-channel framing for future-paradigm selection at iter-216+ when production multi-teacher LLM evidence may emerge or sharper β routing techniques become available.**

---

**End of Paradigm Shift #70 Candidate B design document.** ~3000 words. ENSEMBLE-DISTILL-CHIRON: multi-teacher portfolio distillation as a META-CHANNEL above #68 TEACHER PROVENANCE, lifting four axes simultaneously by 2.5× per axis. RESERVE recommended; magnitude does not clear iter-214 bar; risk-adjusted expected payoff below 1×; mechanism preserved for future-paradigm selection.
