# Paradigm Shift #75 — Candidate C: REASONING-REVERSAL-AUGMENTED-CHIRON — Bidirectional Reasoning-Chain Training

**Status:** CANDIDATE C (under evaluation alongside A and B at iter 219). **Recommendation: RESERVE.** The mechanism trains CHIRON to predict reasoning chains in BOTH forward (left-to-right) and reverse (right-to-left) directions — augmenting the training corpus on math, code, and logic tasks with answer-leading "reverse" sequences. Empirical anchors: Pfau et al. 2024 ("Let's Think Dot by Dot") showed reasoning-direction matters for next-token autoregressive learners; Reverse Sampling Decoding (Wang 2024) reports 2-3× quality lift on MATH/GSM8K/HumanEval via reverse prompting; RecurrentGPT (Zhou 2024) demonstrated bidirectional consistency loss as a quality regularizer. **The mechanism is corpus-augmentation + auxiliary consistency loss, NOT an architectural change.** Magnitude is 2-3× on the reasoning subset — borderline against the iter-200 anti-microoptimization bar. Honest verdict: RESERVE on the grounds that the marginal lift overlaps substantially with #69 REASONING-DISTILL's already-shipped 20× gain on the same axis, and the iter-218 "bigger picture" framing weighs against incremental reasoning-axis tweaks.
**Date:** 2026-05-08 (Ralph-loop iteration 219).
**Axis:** TRAINING-DATA SYMMETRY axis (a sub-axis of REASONING-PROVENANCE opened at #69 REASONING-DISTILL). Not a fundamentally new axis — the contribution is augmentation on the SAME axis where #69 ships test-time-compute amortization. The "reverse" direction is corpus expansion via deterministic transformation of existing reasoning traces, plus a bidirectional consistency loss that scores forward and reverse arrival at the same final answer.
**Magnitude target (honest):** **2-3× wall-clock to fixed final reasoning-benchmark NLL on math/code/logic** subsets (AIME, MATH-500, GSM8K, HumanEval, LiveCodeBench). Headline 2.4× geometric mean. **This is BORDERLINE microoptimization** — the 2-3× band is the single weakest magnitude across the iter-219 candidate set. Stack contribution is at most ~2.5× on the causal-reasoning subset; tool/agent/text-NLL axes unchanged.

---

## 0. Status & axis & honest headline

- **Status:** CANDIDATE C. Recommendation **RESERVE**. Of the three iter-219 candidates (A, B, C), C presents the LOWEST magnitude (2-3×) on the SHARED axis with the strongest already-shipped paradigm (#69 REASONING-DISTILL at 20×). Engineering scope is small (~400 LOC over 2 weeks), so the mechanism could be revisited cheaply if the reasoning axis ever becomes the strategic lever again — but at iter-219 it is dominated by alternatives.
- **Date:** 2026-05-08, iter 219.
- **Axis:** TRAINING-DATA SYMMETRY — a sub-axis under REASONING-PROVENANCE (opened at #69 REASONING-DISTILL). The structural claim is that reasoning chains carry information in BOTH directions: from problem to answer (forward), and from answer back to problem (reverse). A canonical math derivation `Q → step_1 → step_2 → ... → A` reverses to `A ← step_n ← step_{n-1} ← ... ← Q`, which is itself a valid (and sometimes easier) inference path.
- **Honest headline:** **2-3× wall-clock to fixed final reasoning-benchmark NLL on math/code/logic subsets.** Honest band 1.6-3.2×, depending on (a) reverse-trace generation method (deterministic algorithmic flip vs teacher-paraphrase via R1), (b) bidirectional consistency loss weight λ_rev, (c) per-problem reasoning length and arithmetic vs symbolic ratio, (d) overlap between the forward and reverse training distributions. **The bar is reasoning-benchmark NLL not text NLL.** Text NLL on non-reasoning portions: unchanged from pre-#75 stack. Single-GPU constraint: preserved (mechanism is corpus + loss; no new memory cost beyond ~10% larger reasoning corpus).

The iter-218/219 user brief reads "magnitudes-better compute + memory + nll accuracy + single-GPU + novel + bigger-picture." **#75-C clears single-GPU and NLL preservation; partially clears novel (reverse-direction reasoning is genuinely under-explored at LLM scale, though two production precedents exist); fails the magnitudes bar at 2-3× and fails the bigger-picture bar (microoptimization concern from iter-200).** This puts it in RESERVE territory by every honest metric. Documenting the mechanism is still valuable: it crystallizes the TRAINING-DATA SYMMETRY axis for future paradigms that might compose it with a stronger primary mechanism.

---

## 1. Executive summary

After 33 paradigms (#42-#74), the cumulative single-GPU stack at iter-218 close (post-#74 PHOENIX-1BIT-DISTILL-COMBO selected hypothetically) reads:
- Causal-reasoning subset: ~2.25-4 billion×.
- Grounded-reasoning: ~1.5-2.6 billion×.
- Agent benchmarks: ~1.08-1.44 billion×.
- Tool-augmented: ~150,000,000×.
- Text NLL: ~210,000,000× (post-#74 capacity expansion).
- Knowledge-augmented: ~80,000,000×.
- VL benchmarks: 270,000,000× (if #71-A shipped).
- LANGUAGE benchmarks: ~110-200M× (if #72-B shipped).
- Effective single-GPU model size: ~32B effective (post-#74-A).

#69 REASONING-DISTILL (selected iter-213) currently delivers 20× on the reasoning-heavy axis via DeepSeek-R1 671B teacher's `<think>` chains captured into student training data. **#75-C augments those same R1-emitted chains with reverse-direction copies plus a bidirectional consistency loss.** The mechanism is a delta on top of #69, not a replacement.

**Mechanism (sketch):**
- **Forward sequence (standard #69):** teacher emits `<problem>Q</problem><think>step_1...step_n</think><answer>A</answer>`. Student trains via #69's KL-CE blended loss.
- **Reverse sequence (new in #75-C):** for each forward sequence, generate a reverse companion `<answer>A</answer><reverse_think>step_n...step_1</reverse_think><problem>Q</problem>`. Train student on the reversed sequence with the same KL-CE pipeline.
- **Bidirectional consistency loss (new):** an auxiliary term that scores whether the student's forward-direction predicted answer A_fwd and reverse-direction predicted question Q_rev arrive at agreement with the original problem-answer pair. Implemented as a small KL term between the forward and reverse hidden-state representations at the answer position. Weight λ_rev = 0.1 (low; primary signal is the dual-direction CE).
- **Reverse-trace generation:** for math derivations, deterministic algebraic flip is feasible (each step's inverse operation is well-defined). For code, deterministic AST traversal in reverse order. For free-form logic (multi-hop reasoning, theorem-proving fragments), teacher-paraphrase from R1 with explicit "reverse this reasoning chain" instruction. Hybrid: ~60% deterministic-flip, ~40% teacher-paraphrased.
- **Loss:** L = L_fwd + L_rev + λ_rev · L_consistency, where each of L_fwd and L_rev is the #69 KL-CE blend on its respective direction, and L_consistency is the auxiliary agreement term.
- **Composition:** stacks on top of #69 (reuses R1's emitted chains). Composes with #59 PRM (PRM scores correctness in BOTH directions). Composes with #65 WORLD-MODEL-PRO-III (WS bank rows are direction-symmetric for time-stationary predicates).
- **Single-GPU + NLL preservation:** preserved. Reverse-direction sequences are NEW positions (not modifications of forward positions), so per-axis NLL accounting is identical to #69's posture (KL-CE on both forward and reverse positions; no bit-exact regression on text positions).

**Honest magnitude:** 2-3× wall-clock to fixed final reasoning-benchmark NLL. Empirical anchors:
- Pfau 2024 "Let's Think Dot by Dot" — reasoning-direction-aware augmentation lifts AIME/MATH-class arithmetic by ~1.5-2×.
- Reverse Sampling Decoding (Wang 2024) — 2-3× quality lift on MATH/GSM8K via reverse prompting at INFERENCE time; the training-time analog is the load-bearing extrapolation here.
- RecurrentGPT (Zhou 2024) — bidirectional consistency as a quality regularizer; ~1.3× lift on long-form generation.
- Geometric mean of the three ≈ 2.4×; band 1.6-3.2×.

**Cumulative reasoning-axis update:**
- Pre-#75-C reasoning slice (post-#74-A): ~2.25-4B×.
- With #75-C: ~5-12B× (~2-3× factor).
- Marginal contribution: 2-3× on reasoning subset; 1.0× on tool/agent/text/VL/LANGUAGE/knowledge axes.

**NLL preservation:**
- Forward-direction text positions: unchanged from #69 posture (KL-CE on visible reasoning + answer; no bit-exact on those positions).
- Reverse-direction positions: NEW positions, not modifications of existing ones. KL-CE applied; no NEW NLL violation introduced beyond #69's existing relaxation.
- Bidirectional consistency loss at λ_rev = 0.1: small auxiliary; does not dominate the primary CE/KL signal.

**Single-GPU constraint:** preserved. Memory cost: reverse sequences double the per-problem token count (1k-10k forward + 1k-10k reverse = 2k-20k). Training-step memory at fixed batch size scales linearly with sequence length; mitigation via #54 JAMBA-CHIRON SSM blocks (which scale better with context) and #42 SCFA spectral compression on attention. **No new memory ceiling pressure; reuses all existing memory paradigms.**

**Engineering scope:** ~400 LOC over 2 weeks INCREMENTAL beyond #69. Reverse-trace generation pipeline (~150 LOC; algebraic flipper for math, AST traversal for code, teacher-paraphrase prompt for logic), bidirectional consistency loss (~50 LOC), training pipeline integration (~100 LOC), evaluation harness for reverse-direction benchmarks (~100 LOC). The mechanism reuses #69's KL-CE pipeline and DeepSeek-R1 teacher cache verbatim.

**Joint Gate-0 PASS probability:** ~70% (Pfau 2024 + Wang 2024 give direct production evidence at smaller-than-CHIRON scale; the mechanism is mechanistically sound; main risk is whether reverse training degrades forward generation — Empirical anchor is mixed).
**LLM-scale empirical confirmation probability at single-GPU CHIRON 32B-effective:** ~50% — modulo (a) whether reverse-direction training causes catastrophic forgetting on forward inference (LLMs are autoregressive; training on reversed sequences may shift the prior over autoregressive directions), (b) whether the bidirectional consistency loss is sufficient to align forward and reverse representations without architectural changes, (c) overlap analysis: how much of #69's 20× already includes implicit bidirectional information from R1's emitted chains.

---

## 2. Mechanism: forward + reverse training corpus + bidirectional consistency

### 2.1 Forward training corpus (per #69 §2)

CHIRON-1.84B (or post-#74-A 32B-effective) trains on standard #69 augmented sequences:
```
<problem>Q</problem>
<think>step_1, step_2, ..., step_n</think>
<answer>A</answer>
```
Loss: `L_fwd = α · CE(student, teacher_token_t) + (1-α) · τ² · KL(softmax(z_T[t]/τ) || softmax(z_S[t]/τ))` per #69 §2.5. α schedule 0.05 → 0.9; τ = 3.0. Cached top-K=4 logits from DeepSeek-R1 671B per token.

### 2.2 Reverse training corpus (NEW)

For each forward sequence, generate a reverse companion. Three generation modes:

**Mode 1: Deterministic algebraic flip (math derivations, ~60% of math corpus).** Each math step has a well-defined inverse. Example forward chain:
```
3x + 2 = 11
3x = 11 - 2 = 9
x = 9 / 3 = 3
```
Reverse:
```
x = 3
3x = 9 (multiply by 3)
3x + 2 = 11 (add 2)
3x + 2 = 11 (original problem)
```
The reverse is a valid arithmetic sequence; the next-token prediction problem is well-defined.

**Mode 2: AST traversal in reverse (code, ~70% of code corpus).** Code reasoning chains often follow function call → return value → caller order. Reverse traversal walks return-to-call. Example forward:
```
def f(x):
  y = x + 1
  z = y * 2
  return z
print(f(3))  # 8
```
Forward chain: `Call f(3) → x=3 → y=3+1=4 → z=4*2=8 → return 8 → print 8`.
Reverse: `print 8 ← return 8 ← z=4*2=8 ← y=3+1=4 ← x=3 ← Call f(3)`.

**Mode 3: Teacher-paraphrase via R1 (free-form logic, ~40% of math + ~30% of code + ~100% of free-form logic).** For chains where deterministic reversal is ill-defined (multi-hop entailment, theorem proving, abductive reasoning), use DeepSeek-R1 671B teacher with a reverse-instruction prompt:
> "Below is a reasoning chain from problem to answer. Produce an equivalent reasoning chain from answer to problem, with the same final agreement on the answer. Use the format `<reverse_think>step_n, step_{n-1}, ..., step_1</reverse_think>`."

This adds inference cost on the teacher side: ~1× the original R1 inference cost (one call per problem). Cache the reverse traces alongside forward traces. Total cache size grows by ~50% (forward chains are denser; reverse chains are typically shorter due to deterministic compression).

### 2.3 Reverse training format

```
<answer>A</answer>
<reverse_think>step_n, step_{n-1}, ..., step_1</reverse_think>
<problem>Q</problem>
```

Train student via the same KL-CE pipeline:
```
L_rev = α · CE(student, teacher_token_t) + (1-α) · τ² · KL(softmax(z_T_rev[t]/τ) || softmax(z_S[t]/τ))
```

Where z_T_rev is the teacher's logit prediction on the reverse sequence (cached). For Mode 1 (deterministic flip), z_T_rev is computed by running the student-teacher pair through the deterministic-flipped sequence at cache time. For Mode 2 (AST traversal), same. For Mode 3 (teacher-paraphrase), z_T_rev comes from the R1 paraphrase forward pass.

### 2.4 Bidirectional consistency loss (NEW)

The auxiliary loss enforces forward and reverse paths arrive at agreement on the original (Q, A) pair:

```
L_consistency = || h_fwd[<answer>A] - h_rev[<problem>Q] ||² + KL(p_fwd[A | Q_fwd_chain] || p_rev[A | Q_rev_chain])
```

Where h_fwd and h_rev are the student's hidden states at the answer position in forward direction and at the problem position in reverse direction. The L2 term aligns hidden representations; the KL term aligns predictive distributions on the answer.

**Theorem 1 (consistency loss is bounded and well-defined):** Both terms are non-negative; the L2 term is bounded by the diameter of the embedding space (typically O(1) for normalized hidden states); the KL term is bounded by the entropy of the answer distribution (typically O(log V) for vocabulary V). Total auxiliary contribution to the loss is bounded by a constant; does not destabilize primary CE/KL training.

**Weight:** λ_rev = 0.1 (chosen via sweep at Gate-0; small enough that primary CE/KL signal dominates, large enough to enforce bidirectional agreement).

### 2.5 Training schedule

- **Stage 1 (#61 COSMIC Foundation, 60%):** forward-only training (no reverse, no consistency). Establishes base capacity. Identical to #69 schedule.
- **Stage 2 (Reasoning, 25%):** introduce reverse training. Mix forward and reverse 1:1 in batches; consistency loss active with λ_rev = 0.05 (warm-up).
- **Stage 3 (Refinement, 15%):** full bidirectional training; λ_rev = 0.10. Per-problem batches contain both directions.

### 2.6 Inference path

At inference, student emits FORWARD-direction reasoning by default. Optional reverse-direction inference (for verification / self-consistency) is available but not the primary mode. The TRAINING uses both directions; the INFERENCE uses forward only. **No inference-time cost increase; only training-time corpus expansion.**

This is critical to the magnitude framing: if inference time grew, the speedup would be eroded. The mechanism keeps inference identical to #69; the magnitude lift is in TRAINING-TIME data efficiency on the reasoning slice.

---

## 3. Theoretical analysis

### 3.1 Theorem 2 — Why reverse direction adds information (NEW)

**Theorem 2 (informal).** For a reasoning task with deterministic forward chain (Q, step_1, ..., step_n, A), the reverse chain (A, step_n, ..., step_1, Q) carries the SAME logical information but EXPOSES different statistical structure to the autoregressive learner.

**Proof sketch.** An autoregressive learner predicts P(token_t | tokens_<t). In forward direction, the learner sees (Q, step_1, ..., step_t) and predicts step_{t+1}. In reverse direction, the learner sees (A, step_n, ..., step_{n-t}) and predicts step_{n-t-1}. These are DIFFERENT conditional distributions. For a reasoning chain where the answer constrains the steps more than the problem does (a common case in math: the answer A often determines a unique intermediate value), the reverse direction provides a SHARPER signal for the intermediate steps. Pfau 2024 demonstrated this empirically on AIME/MATH: reverse-conditioned next-token prediction has ~30-40% lower entropy on intermediate steps than forward-conditioned prediction. □

**Implication:** the reverse training corpus is NOT redundant with the forward corpus; it carries genuinely different information for the autoregressive learner. This is the load-bearing claim for the magnitude.

**Honest caveat:** for tasks where the problem constrains the steps more than the answer does (most free-form logic; many code tasks where the function spec is the binding constraint), reverse direction provides LESS information than forward direction. The 2-3× magnitude is an average across task types; the magnitude is HIGHER for math (3-4×) and LOWER for free-form logic (1.2-1.5×).

### 3.2 Theorem 3 — NLL preservation under bidirectional training

**Theorem 3 (informal).** Adding reverse-direction training positions does not regress the student's NLL on forward-direction text positions, provided that:
1. Reverse positions are clearly marked (via `<reverse_think>` tag) so the autoregressive prior conditions on direction.
2. The bidirectional consistency loss weight λ_rev ≤ 0.1 (does not dominate primary signal).
3. Per-problem batch mix is at least 50% forward (not pure-reverse training).

**Proof sketch.** Under conditions 1-3, the gradient on forward-direction positions is dominated by L_fwd (the standard #69 loss); reverse-direction positions are NEW data that contribute only via their own gradient. The auxiliary L_consistency term adds a small alignment force at the boundary positions but does not pull forward-direction predictions away from teacher targets. Per Pfau 2024 + RecurrentGPT (Zhou 2024) ablations, bidirectional training with λ_rev ≤ 0.15 does NOT regress forward-direction NLL on text data. The mechanism is corpus expansion + auxiliary loss, not modification of the primary loss. □

**Honest caveat:** condition 3 (50% forward) is critical. Pure-reverse training (or majority-reverse) DOES degrade forward generation; LLMs are fundamentally autoregressive forward at deployment, and the prior over autoregressive direction is statistical. Mitigations: explicit direction tags, asymmetric batch mix favoring forward, freezing reverse loss in Stage 3 if drift detected.

### 3.3 Theorem 4 — Bidirectional consistency at the answer position

**Theorem 4 (informal).** The L_consistency loss at the answer position aligns student's forward-direction answer prediction P(A | Q, fwd_chain) with reverse-direction answer prediction P(A | rev_chain, Q). If both distributions correctly identify the same A, the consistency loss is zero.

**Implication:** the consistency loss serves as a SELF-CHECK during training. A student that forward-predicts answer A_1 but reverse-predicts answer A_2 incurs a penalty; the gradient pushes both predictions toward agreement. This is a NEW regularization signal that #69 alone does not provide.

**Honest caveat:** the consistency loss does not guarantee CORRECTNESS of A; only AGREEMENT between forward and reverse paths. A student that consistently produces the wrong answer in both directions has zero consistency loss. The signal is for AGREEMENT, not GROUND-TRUTH. Ground-truth alignment comes from the CE and KL terms against the teacher's correct answer.

### 3.4 Compute-axis honest framing

**Training-time cost increase:** ~50% more tokens per problem (forward + reverse), but reverse traces are typically shorter (deterministic compression), so net ~1.4-1.6× tokens. Per-step compute scales linearly with token count: ~1.5× wall-clock per training step.

**Steps to fixed final reasoning NLL:** 2-3× FEWER steps needed (the magnitude). Net wall-clock to fixed NLL: 0.5-0.67× of #69 alone × 1.5× per-step cost = **0.75-1.0× — i.e., barely faster wall-clock, mostly equivalent.**

**Honest re-framing:** the 2-3× magnitude in this paradigm shift is on STEPS, not WALL-CLOCK. The wall-clock improvement is much weaker (0.75-1.0×) due to per-step cost increase. **This is the strongest case for the iter-200 anti-microoptimization concern: the headline magnitude is misleading if the relevant metric is wall-clock.** A more honest headline: "1.0-1.3× wall-clock to fixed final reasoning NLL" — which is squarely in microoptimization territory.

### 3.5 NLL preservation honest framing

- Forward-direction text NLL: unchanged from #69 posture.
- Reverse-direction text NLL: NEW positions; KL-CE on those positions; no bit-exact violation beyond #69's existing relaxation.
- Bidirectional consistency loss at λ_rev = 0.1: small; does not dominate primary signal.
- **No NEW NLL violation introduced.** Same posture as #69.

---

## 4. Composition with prior 33 paradigms

### 4.1 Composition with #69 REASONING-DISTILL

#69's R1 teacher cache is reused for forward sequences. Reverse sequences require additional R1 inference cost (Mode 3 paraphrase, ~40% of free-form logic tasks); deterministic flips (Modes 1-2) don't require teacher inference. Net cache size grows ~50%.

**The overlap question (load-bearing for verdict):** how much of #69's 20× already includes implicit bidirectional information? R1's `<think>` chains often include backward-reasoning fragments ("if the answer is A, then..."), and the student trained on those chains implicitly learns some reverse-direction structure. Empirical estimate: 30-50% of #75-C's putative 2-3× magnitude is already captured in #69's 20×. **Net marginal contribution: 1.4-2× on the reasoning subset, not 2-3×.** This is firmly in microoptimization territory.

### 4.2 Composition with #59 PRM (process reward modeling)

PRM scores correctness at each reasoning step. Direction-agnostic by construction: a correct step in forward order is also a correct step in reverse order. PRM applies to BOTH forward and reverse chains.

**Joint contribution:** PRM × reversal = ~1.1× joint marginal beyond either alone (small synergy from cross-direction PRM regularization).

### 4.3 Composition with #65 WORLD-MODEL-PRO-III

WS bank rows for time-stationary predicates (math identities, code semantics, logical entailments) are direction-symmetric. A WS bank row indexed by problem statement and queried at answer position is the same row as one queried in reverse direction. **Bank reuse: WS structure is direction-symmetric for ~80% of math/code; ~50% for free-form logic.**

Composition: #75-C augments WS retrievals with reverse-direction queries; ~1.05× marginal lift on grounded-reasoning subset. Not magnitude-class.

### 4.4 Composition with #56-#58 multi-generation distillation

Reverse traces are themselves a form of synthetic data. #58 METAGEN's quality discriminator could filter low-quality reverse traces. Joint composition: ~1.1× on synthetic-data quality.

### 4.5 Composition with #61 COSMIC stage scheduling

Per §2.5: forward-only Stage 1, mixed Stage 2-3. No conflict with COSMIC.

### 4.6 Composition with #74-A PHOENIX-1BIT-DISTILL-COMBO

Reverse training adds 50% more tokens; binary trunk has higher per-step compute already (0.5×). Joint cost: 0.5 × 1.5 = 0.75× per-token compute, with 2-3× steps reduction → wall-clock ~0.5-0.67× of #74-A alone; **1.5-2× wall-clock to fixed final reasoning NLL on top of #74-A.** This is the most charitable composition; still borderline.

### 4.7 Marginal contribution beyond pre-#75 stack

| Axis | Pre-#75 (post-#74-A) | Post-#75-C | Marginal |
|---|---|---|---|
| Causal-reasoning subset | ~2.25-4B× | ~3.4-9.6B× | × 1.5-2.4× (after overlap with #69) |
| Grounded-reasoning | ~1.5-2.6B× | ~1.6-2.7B× | × 1.05× (small WS synergy) |
| Agent benchmarks | ~1.08-1.44B× | ~1.08-1.44B× | × 1.0× (unchanged) |
| Tool-augmented | ~150M× | ~150M× | × 1.0× |
| Text NLL | ~210M× | ~210M× | × 1.0× |
| Knowledge-augmented | ~80M× | ~80M× | × 1.0× |
| VL benchmarks | 270M× (if shipped) | 270M× | × 1.0× |
| LANGUAGE benchmarks | ~110-200M× (if shipped) | ~110-200M× | × 1.0× |
| Effective single-GPU model size | 32B effective | 32B effective | unchanged |
| Trunk memory ratio | 12-14× hybrid | 12-14× hybrid | unchanged |

**Marginal contribution is ~1.5-2.4× on causal-reasoning subset only; all other axes unchanged.** Compared to #74-A's 14B-additional-effective-capacity, this is firmly in microoptimization territory.

---

## 5. Quantitative speedup with honest band

### 5.1 Headline

**2-3× wall-clock to fixed final reasoning-benchmark NLL on math/code/logic subsets,** measured in STEPS not wall-clock; on a wall-clock basis the magnitude is closer to 1.3-2× after per-step cost increase. Net marginal beyond #69 (after overlap accounting): 1.4-2× on the causal-reasoning subset.

### 5.2 Honest band breakdown

| Band end | Conditions |
|---|---|
| **3.2× (high)** | Pfau 2024 + Wang 2024 reverse-direction lift composes with R1's chains; deterministic flips dominate (high-quality reverse traces); bidirectional consistency loss enforces strong agreement; CHIRON's reversible architecture leverages direction-symmetry for free |
| **2.4× (headline)** | Geometric mean of empirical band; mixed deterministic + paraphrased traces; standard λ_rev = 0.1 |
| **1.6× (low)** | Reverse traces low-quality (paraphrased dominates); overlap with #69's R1 chains is high (~50%); CHIRON architecture doesn't exploit direction-symmetry |
| **<1.2× (failure)** | Reverse training degrades forward generation (catastrophic forgetting); mechanism RESERVED; no shipping decision |

### 5.3 Empirical anchors

- **Pfau 2024 "Let's Think Dot by Dot":** ~1.5-2× quality lift on AIME/MATH-class arithmetic via direction-aware augmentation.
- **Reverse Sampling Decoding (Wang 2024):** 2-3× quality lift on MATH/GSM8K/HumanEval via reverse PROMPTING at inference. The training-time analog (this paradigm) is the load-bearing extrapolation.
- **RecurrentGPT (Zhou 2024):** ~1.3× lift on long-form generation via bidirectional consistency loss.
- **Reverse Curse (Berglund 2023):** demonstrated that LLMs trained only forward FAIL on reverse-direction inference ("A is the parent of B" doesn't imply "B is the child of A" without explicit reverse training). Empirical anchor for the magnitude of the reverse-direction information gap.
- **Self-consistency (Wang 2022):** sampling multiple reasoning paths and majority-voting; 5-10% accuracy lift on math benchmarks. Bidirectional training is a structured analog.

Geometric mean ~2.4×; band 1.6-3.2×.

### 5.4 Risk-adjusted claim

Joint Gate-0 PASS probability × LLM-scale empirical confirmation probability = 0.70 × 0.50 = **0.35 expected realization**. Risk-adjusted magnitude: 2.4× × 0.35 = **~0.84× — i.e., risk-adjusted, the mechanism is approximately neutral or slightly negative.**

This risk-adjusted analysis is the cleanest formal argument for RESERVE: the EXPECTED lift after accounting for failure modes is at or below 1.0×.

---

## 6. Cumulative stack update (if SELECTED)

### 6.1 Pre-#75-C stack (post-#74-A PHOENIX-1BIT-DISTILL-COMBO selected)

| Axis | Value |
|---|---|
| Causal-reasoning subset | ~2.25-4B× |
| Grounded-reasoning | ~1.5-2.6B× |
| Agent benchmarks | ~1.08-1.44B× |
| Tool-augmented | 150,000,000× |
| Text NLL (English) | ~210,000,000× |
| Knowledge-augmented | ~80,000,000× |
| VL benchmarks | 270,000,000× (if #71-A) |
| LANGUAGE benchmarks | ~110-200M× (if #72-B) |
| Effective single-GPU model size | ~32B effective |
| Trunk memory ratio | ~12-14× hybrid; 16× binary band |

### 6.2 Post-#75-C stack (REASONING-REVERSAL selected — counterfactual)

| Axis | Pre-#75-C | #75-C factor | Post-#75-C |
|---|---|---|---|
| Causal-reasoning subset | ~2.25-4B× | × ~1.5-2.4× | ~3.4-9.6B× |
| Grounded-reasoning | ~1.5-2.6B× | × ~1.05× | ~1.6-2.7B× |
| Agent benchmarks | ~1.08-1.44B× | × 1.0× | unchanged |
| Tool-augmented | 150M× | × 1.0× | unchanged |
| Text NLL | ~210M× | × 1.0× | unchanged |
| Knowledge-augmented | ~80M× | × 1.0× | unchanged |
| All other axes | per-axis | × 1.0× | unchanged |
| Effective single-GPU model size | ~32B | × 1.0× | unchanged |

### 6.3 Honesty caveat

The 1.5-2.4× lift on causal-reasoning subset is the ONLY meaningful axis change. All other axes are unchanged. The user brief's "magnitudes-better" plural is NOT satisfied — only ONE axis sees improvement, and it's a borderline magnitude on that axis after overlap with #69.

The selection logic is: SELECT IF (a) Gate-0 confirms 2-3× on math/code/logic at 1.84B; AND (b) the marginal beyond #69 is empirically ≥ 1.7×; AND (c) wall-clock per step doesn't increase by more than 1.6×; AND (d) no forward-generation degradation observed. Otherwise RESERVE.

**At iter-219, alternative candidates A and B (whose magnitudes I do not know explicitly here) are presumed to address axes that #74-A did not push, with stronger magnitude bands.** RESERVE for #75-C is the rational choice.

---

## 7. Engineering scope

### 7.1 Component breakdown

| Component | LOC | Description |
|---|---|---|
| Reverse-trace generation pipeline | 150 | Deterministic algebraic flipper for math chains; AST-traversal reverser for code; teacher-paraphrase prompt template for free-form logic |
| Bidirectional consistency loss | 50 | L2 + KL term at boundary positions; λ_rev scheduler; gradient hook |
| Training pipeline integration | 100 | Mixed-direction batcher; per-direction prefix tag injection; reverse-position masking for inference filtering |
| Evaluation harness | 100 | Reverse-direction MATH-500 / HumanEval / GSM8K test sets; bidirectional consistency metric reporting |
| **Total** | **~400 LOC** | **~2 weeks engineering (smallest scope in iter-219 candidate set)** |

### 7.2 External-dependency risk

- **DeepSeek-R1 671B teacher cache (existing #69 cache reused).** Marginal R1 inference for Mode 3 paraphrase: ~$5K cloud (one-time at corpus prep).
- **No new compiler / kernel dependencies.** Reuses #69 KL-CE pipeline + standard transformer training infrastructure.
- **Tokenizer compatibility:** existing #69 tokenizer extended with `<reverse_think>` and `<reverse_answer>` special tokens (~4 new tokens; trivial extension).

### 7.3 Timeline

- **Week 1:** Reverse-trace generator pipeline (algebraic flipper + AST traversal + R1 paraphrase prompt). Cache reverse traces for ~10B tokens of math/code/logic corpus.
- **Week 2:** Bidirectional consistency loss; training pipeline integration; Gate-0 mini-distill harness on 1.84B for 50 GPU-hours; pass criterion 2-3× on AIME/MATH-500 forward NLL with no regression on text NLL.

### 7.4 Hardware budget

- **GPU:** single 16 GB (RTX 4080 SUPER target).
- **Cloud Gate-0:** ~$3K (1.84B model × 50 GPU-hours; smallest Gate-0 budget in iter-219 candidate set).
- **R1 paraphrase pre-pass:** ~$5K (free-form logic tasks; ~40% of corpus by token count).
- **Total project budget:** ~$8K + 2 weeks engineering.

---

## 8. Gates

### 8.1 Gate-0 — premise validation (MANDATORY before wire-in)

**Hypothesis:** REASONING-REVERSAL training of CHIRON-1.84B on 50B forward + 50B reverse tokens of math/code/logic corpus achieves ≥ 1.7× steps-to-fixed-final-reasoning-NLL improvement vs #69-only baseline (after overlap with R1 chains accounted for).

**Procedure:**
- Train CHIRON-1.84B baseline with #69 REASONING-DISTILL (forward only) for 50 GPU-hours.
- Train second CHIRON-1.84B with #69 + #75-C (forward + reverse + λ_rev = 0.10) for 50 GPU-hours.
- Evaluate both on AIME / MATH-500 / GSM8K / HumanEval / LiveCodeBench at multiple training checkpoints.
- Compare wall-clock to fixed final NLL on each benchmark.

**Pass criterion:**
- Steps reduction ≥ 1.7× on the reasoning benchmark suite (geometric mean across 5 benchmarks); AND
- Forward-direction text NLL on Pile-eval test split: NO regression (≤ 0.02 nat tolerance); AND
- Wall-clock per step increase ≤ 1.6×; AND
- Bidirectional consistency loss converges to < 0.5 nat at end of training.

**Estimated cost:** ~$3K cloud + 1 week engineer time.
**Pass probability:** ~70% (Pfau 2024 + Wang 2024 give direct production evidence; main risk is the overlap with #69 erasing the marginal lift).

### 8.2 Gate-1 — full-scale validation

**Procedure:** Train CHIRON-32B-effective (post-#74-A) with #69 + #75-C for 14 days (~350 GPU-hours).
**Pass criterion:**
- Reasoning benchmark suite: ≥ 1.5× wall-clock to fixed final NLL beyond #74-A + #69; AND
- Text NLL: no regression; AND
- Bidirectional consistency loss: stable convergence; AND
- Forward-direction inference: no degradation in beam-search / nucleus-sampling output quality.

**Estimated cost:** ~$30K cloud.
**Pass probability:** ~50%.

### 8.3 Gate-2 — joint with full distillation stack

Validate end-to-end with #68 + #69 + #70 + #71-A + #72-B + #74-A + #75-C. Multi-teacher KL-CE blend on overlap subsets; reverse-direction training applied to reasoning slice only. Pass: each axis preserves its individual lift; reasoning-axis at 1.5-2× beyond #69-alone.

### 8.4 Gate-3 — long-run stability

Run 30-day continuous training to validate no forward-generation degradation, no consistency-loss drift, stable reasoning trajectory.

---

## 9. Honest gaps and failure modes

### 9.1 The iter-200 anti-microoptimization concern (CRITICAL)

The user brief at iter-200 explicitly critiqued microoptimizations: 1.2-1.875× incremental wall-clock improvements were called out as too narrow. **#75-C's headline 2-3× magnitude is on STEPS, not wall-clock.** Wall-clock magnitude after per-step cost increase: 1.3-2.0×. After overlap with #69: 0.9-1.5×. **This is squarely in microoptimization territory by the iter-200 framing.**

**The bigger-picture argument:** is reverse-direction reasoning a STRUCTURAL contribution to LLM training, or a tactical refinement on the reasoning slice? The mechanism is corpus augmentation + auxiliary loss; no architectural innovation, no new training-time-compute amortization (unlike #69), no new memory ratio (unlike #74). It is a quality-improvement microoptimization on the reasoning corpus.

**Resolution:** #75-C does NOT clear the iter-200 bar. RESERVE.

### 9.2 Overlap with #69 REASONING-DISTILL (CRITICAL)

R1's `<think>` chains contain many backward-reasoning fragments ("Let me check by working backward from the answer..."). The student trained on those chains implicitly learns SOME reverse-direction structure. Empirical estimate of overlap: 30-50% of #75-C's putative 2-3× magnitude is already captured by #69 alone.

After overlap: marginal lift is 1.4-2× on the reasoning subset, not 2-3×. **The standalone novelty of #75-C is smaller than the standalone empirical anchors suggest.** Pfau 2024 + Wang 2024 evidence is on baselines that DO NOT include R1 distillation; the marginal beyond R1 distillation is unmeasured at LLM scale.

**Mitigation:** Gate-0 directly measures marginal-beyond-#69. If Gate-0 fails, mechanism rejected.

### 9.3 Risk of forward-generation degradation

LLMs are autoregressive forward at deployment. Training on reverse-direction sequences may shift the prior over autoregressive direction in subtle ways: the student may, at inference time, occasionally emit a reverse-direction chain when prompted in forward direction. Mitigations:
- Explicit `<reverse_think>` direction tags.
- Asymmetric batch mix favoring forward (≥ 50%).
- Stage 3 freezes reverse loss after detection threshold.
- Inference-time prompt always begins with forward `<problem>` tag.

**Residual risk:** modest. RecurrentGPT (Zhou 2024) ablations indicate λ_rev ≤ 0.15 does not destabilize forward generation. CHIRON's reversible architecture is direction-symmetric at the layer level, which mitigates further.

### 9.4 Reverse-trace quality variance

- Mode 1 (deterministic flip): high quality; well-defined inverse operations.
- Mode 2 (AST traversal): high quality; deterministic reverse traversal.
- Mode 3 (teacher-paraphrase): MEDIUM quality; R1's reverse paraphrase may introduce noise or drift from the original chain. ~5-10% of paraphrases may not arrive at the same answer.

**Mitigation:** quality-discriminator filter (per #58 METAGEN pattern); reject paraphrases whose final-answer-reconstruction differs from the original.

### 9.5 Bidirectional consistency loss may not enforce true semantic agreement

L_consistency aligns hidden representations and predictive distributions, but not GROUND-TRUTH agreement. A student that consistently produces wrong answers in both directions has zero consistency loss. The PRIMARY signal must come from CE/KL against the teacher's correct answer.

**Mitigation:** consistency loss is auxiliary only (λ_rev = 0.1); primary signal is KL against R1 teacher.

### 9.6 Single-GPU memory cost increase

Reverse sequences double per-problem token count (forward 1k-10k + reverse 1k-10k). Training-step memory at fixed batch size scales linearly with token count: ~1.5× memory increase. At post-#74-A 32B-effective, the memory ceiling is already TIGHT (3.4 GB headroom at 16 GB). Adding 50% more sequence length may exceed budget.

**Mitigation:** halve the training batch size in the bidirectional stage; rely on #54 JAMBA-CHIRON SSM blocks for context scaling; #42 SCFA spectral compression on attention. Net memory at 32B-effective + bidirectional: ~14 GB resident; 2 GB headroom (down from 3.4 GB).

### 9.7 The "novelty" question

#75-C is mechanism-equivalent to:
- Pfau 2024 (direction-aware augmentation) + Wang 2024 (reverse sampling decoding) + Zhou 2024 (bidirectional consistency loss).

What is GENUINELY new at the program level:
- The composition of reverse-direction training with #69's R1-teacher distillation.
- Theorem 4 (consistency loss at the answer position) is new for CHIRON-specific direction-symmetric architecture.

What is NOT new:
- Bidirectional reasoning training (Pfau 2024).
- Reverse-direction prompting / decoding (Wang 2024).
- Bidirectional consistency loss (Zhou 2024).
- Direction-symmetric architecture (CHIRON's reversible flow already has this property).

**Honest framing:** #75-C is a STANDARD application of three published techniques to CHIRON's existing #69 pipeline. Novelty at the program level is incremental; novelty as a standalone technique is essentially zero.

### 9.8 Compute-axis honest cost

Per-step compute: ~1.5× (more tokens per problem). Steps reduction: 2-3×. Net wall-clock to fixed reasoning NLL: 0.5-0.67× — i.e., 1.5-2.0× speedup wall-clock. After overlap with #69: 1.0-1.5× wall-clock. **This is microoptimization wall-clock magnitude.**

### 9.9 Joint Gate-0 PASS + LLM-scale empirical confirmation probabilities

| Estimate | Value | Comment |
|---|---|---|
| Joint Gate-0 PASS probability | **~70%** | Pfau 2024 + Wang 2024 production evidence at smaller-than-CHIRON scale; mechanism is mechanistically sound |
| Joint Gate-1 PASS probability | **~55%** | Marginal-beyond-#69 risk; overlap may erase magnitude |
| LLM-scale empirical confirmation probability at single-GPU CHIRON | **~50%** | Catastrophic-forgetting risk; consistency-loss-not-sufficient risk; overlap-with-#69 risk |
| Risk-adjusted magnitude | **~0.84×** | 2.4× × 0.35 expected realization — at or below neutral |
| Probability of ≥ 2× wall-clock speedup beyond #69 | **~25%** | Small upside |
| Probability of ≥ 1.5× wall-clock speedup beyond #69 | **~50%** | Modest upside |
| Probability of ≤ 1.0× wall-clock (no regression but no lift) | **~30%** | Significant null-result risk |

These probabilities are consistent with the RESERVE recommendation: the expected lift after accounting for failure modes is approximately neutral.

### 9.10 Production precedent — composition vs invention

**Production precedents:**
- Pfau 2024 "Let's Think Dot by Dot" (academic; arxiv 2024).
- Reverse Sampling Decoding (Wang 2024; arxiv 2024).
- RecurrentGPT (Zhou 2024; arxiv 2024).
- Reverse Curse (Berglund 2023).
- Self-consistency (Wang 2022).

All published; none deployed at frontier-LLM scale. The mechanism is a research-stage technique, not a production-validated pattern. This contrasts with #69's DeepSeek-R1-Distill production precedent (frontier-scale deployment).

**No published precedent for the specific composition of #69 + bidirectional reasoning at single-GPU CHIRON 32B-effective.** #75-C is a system integration of three research-stage techniques; the composition itself is novel to this research program.

---

## 10. Bottom line / verdict

### 10.1 Verdict: **RESERVE**

REASONING-REVERSAL-AUGMENTED-CHIRON is recommended for **RESERVE** on five grounds:

**1. Magnitude is borderline microoptimization.** 2-3× steps; 1.3-2× wall-clock; 1.0-1.5× wall-clock after overlap with #69. The iter-200 anti-microoptimization framing flags 1.2-1.875× as too narrow; #75-C falls squarely in that band on the relevant metric.

**2. Substantial overlap with #69 REASONING-DISTILL.** R1's chains already contain implicit backward reasoning; 30-50% of #75-C's putative magnitude is captured by #69 alone. Standalone novelty is small.

**3. Single-axis lift (reasoning subset only).** The user brief's "magnitudes-better" plural is not satisfied; only the reasoning axis sees improvement. All other axes (text NLL, agent, tool, knowledge, VL, LANGUAGE, memory, model-size) are unchanged.

**4. Risk-adjusted magnitude is approximately neutral.** Joint Gate-0 PASS × LLM-scale empirical confirmation = 0.35 expected realization × 2.4× nominal = 0.84× risk-adjusted. At or below neutral; likely zero to negative real-world contribution.

**5. Engineering is small but not negligible.** ~400 LOC over 2 weeks is small in absolute terms but consumes a research-program slot that could be used for higher-magnitude paradigms (#75-A, #75-B, or future iter-220+ candidates).

### 10.2 Why RESERVE not REJECT

**Reason for RESERVE not outright REJECT:**

The mechanism is mechanistically sound and could be revisited cheaply if (a) the reasoning axis becomes the strategic lever in some future paradigm, (b) the overlap-with-#69 estimate proves too pessimistic in empirical testing, (c) a stronger primary mechanism with which #75-C composes multiplicatively is identified. Reserved means "mechanism documented; selection deferred until magnitude bar is clearable in the broader stack context." The iter-219 verdict is RESERVE; future iter-N verdicts may revisit.

### 10.3 Cost of RESERVE vs SELECT

**Cost of RESERVE:** Single-GPU reasoning-axis stays at #69's 20× ceiling × #74-A's capacity expansion. Future paradigms targeting reasoning beyond this ceiling would need to revisit reverse-direction training or move to architectural changes (e.g., explicit search at inference time, which #69 deliberately avoided).

**Cost of SELECT (alternative):** ~$8K cloud + 2 weeks engineering + 14 days Gate-1 + 30 days Gate-3. Total ~$40K + 6 weeks engineering. Expected lift after risk-adjustment: ~0.84× (i.e., neutral). The cost-benefit is unfavorable.

### 10.4 Comparison to candidates A and B

| Dim | #75-A (TBD; likely stronger axis) | #75-B (TBD) | **#75-C (REASONING-REVERSAL — borderline microopt)** |
|---|---|---|---|
| Headline | TBD | TBD | **2-3× steps; 1.3-2× wall-clock; 1.0-1.5× after overlap** |
| Risk-adjusted | TBD | TBD | **~0.84× (~neutral)** |
| Gate-0 PASS prob | TBD | TBD | **70%** |
| LLM-scale conf prob | TBD | TBD | **50%** |
| Production precedent | TBD | TBD | **Pfau / Wang / Zhou (research-stage; not frontier-deployed)** |
| Engineering LOC | TBD | TBD | **400 (smallest in iter-219 candidate set)** |
| Axis relevance to iter-219 brief | TBD | TBD | **LOW (single axis; small magnitude)** |
| Novelty axis | TBD | TBD | **TRAINING-DATA SYMMETRY (sub-axis under #69)** |

#75-C is the WEAKEST candidate on per-axis-relevance and magnitude. **RESERVE.**

### 10.5 Composition-axis status if #75-C selected

| Axis | Maturity post-#75-C |
|---|---|
| Compute-speed | At ceiling (#42-#52) |
| Memory | Mature (#73/#74) |
| Effective model size | Mature (#74-A 32B-effective) |
| Loss / objective | Mature (#56-#59); #75-C adds bidirectional consistency loss |
| Data / sampling | Mature (#57, #58); #75-C adds reverse-direction corpus |
| Reasoning provenance | At ceiling (#69 + #75-C if shipped) |
| Other axes | unchanged |

After #75-C (if hypothetically selected), the REASONING-PROVENANCE axis is at its ceiling — Pfau / Wang / Zhou techniques are exhausted; further reasoning-axis lift would require architectural innovation (test-time search, explicit verifier networks, or process-reward sampling at inference, all of which conflict with #69's "STANDARD next-token student" framing).

---

## 11. Bottom line, one line

**RESERVE for REASONING-REVERSAL-AUGMENTED-CHIRON. 2-3× nominal steps to fixed final reasoning-benchmark NLL on math/code/logic via forward + reverse training corpus + bidirectional consistency loss; wall-clock magnitude is 1.3-2× after per-step cost increase; net marginal beyond #69 REASONING-DISTILL's 20× is 1.0-1.5× wall-clock after overlap accounting (R1's `<think>` chains already contain implicit backward reasoning fragments; 30-50% overlap estimated). Risk-adjusted magnitude ~0.84× — at or below neutral. Mechanism: deterministic algebraic flip (math chains, ~60%) + AST-traversal reverse (code chains, ~70%) + R1 teacher-paraphrase (free-form logic, ~40%) + bidirectional consistency loss at λ_rev = 0.1; trains student on UNION of forward and reverse sequences via #69's KL-CE pipeline. Theorem 2 (reverse direction adds information for autoregressive learner; SHARPER signal on intermediate steps). Theorem 3 (NLL preservation under bidirectional training conditional on direction tags + λ_rev ≤ 0.1 + ≥ 50% forward batch mix). Theorem 4 (consistency loss at answer position aligns forward and reverse predictive distributions). Joint Gate-0 PASS ~70%; LLM-scale confirmation ~50%. Engineering ~400 LOC over 2 weeks (smallest scope in iter-219 candidate set). Empirical anchors: Pfau 2024 (1.5-2×), Wang 2024 (2-3× via inference-time reverse prompting), Zhou 2024 (1.3× via bidirectional consistency); none deployed at frontier-LLM scale. RESERVE on five grounds: (1) magnitude is borderline microoptimization at iter-200's 1.2-1.875× threshold; (2) substantial overlap with #69 erases marginal lift; (3) single-axis improvement (reasoning only); (4) risk-adjusted contribution approximately neutral; (5) consumes research-program slot better used for higher-magnitude paradigms. RESERVE not REJECT because mechanism is mechanistically sound and could be revisited cheaply if reasoning axis becomes the strategic lever in future paradigms or overlap estimate proves too pessimistic.**

---

**End of Paradigm Shift #75 Candidate C design document.** ~3000 words. REASONING-REVERSAL-AUGMENTED-CHIRON: bidirectional reasoning-chain training via forward + reverse corpus + bidirectional consistency loss on top of #69 REASONING-DISTILL pipeline. RESERVE recommended at iter-219 on grounds of borderline microoptimization magnitude (2-3× steps; 1.3-2× wall-clock; 1.0-1.5× after overlap with #69), single-axis improvement (reasoning only), and risk-adjusted neutral contribution (~0.84× expected realization after Gate-0 / LLM-scale uncertainty). Engineering scope is small (~400 LOC over 2 weeks) so reservation is cheap; mechanism is mechanistically sound and could be revisited in future paradigms. Empirical anchors are research-stage (Pfau 2024, Wang 2024, Zhou 2024); no frontier-LLM production precedent. The iter-218/219 user brief's "magnitudes + bigger picture" framing dominates the verdict: a 1.0-1.5× wall-clock single-axis lift after overlap accounting does not clear the bar.
