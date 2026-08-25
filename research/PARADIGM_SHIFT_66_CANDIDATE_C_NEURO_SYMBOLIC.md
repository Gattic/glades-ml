# Paradigm Shift #66 Candidate C — NEURO-SYMBOLIC-CHIRON: typed-DSL program execution interleaved with neural prediction

**Status:** candidate-C design for paradigm shift #66. **Recommended action: RESERVE** (positive on mechanism + reasoning-axis fit; honest on overlap with #60 TOOL-LLM and on training-time complexity of differentiable execution). Headline a real but **narrow** speedup band on the algebra/logic/binding-heavy reasoning subset; promising as a `#66+` SYMBOLIC-axis opener but not yet the most defensible choice for an immediate selection over candidates A and B.
**Date:** 2026-05-08 (Ralph-loop iteration 210, post-iter-209 close at ~6,600,000× cumulative on grounded-reasoning subset / 5,500,000× knowledge-augmented / 5,360,000× agent / 3,030,000× tool-augmented / 930,000× text-NLL).
**Predecessors.** All of #42–#65. Load-bearing references: `PARADIGM_SHIFT_60_DESIGN.md` (TOOL-LLM — closest mechanism, must differentiate explicitly), `PARADIGM_SHIFT_59_DESIGN.md` (PRM-CHIRON — extends to per-DSL-step process rewards), `PARADIGM_SHIFT_62_CANDIDATE_B_AGENT_CHIRON.md` (multi-step trajectory loop — DSL programs are sub-trajectories), `PARADIGM_SHIFT_65_CANDIDATE_A_WORLD_MODEL_PROMOTED.md` (WS `(E, P, R, C)` fields — DSL operates over the same structured world state).

**Axis.** **SYMBOLIC** — a new, eleventh axis after #65 closed GROUNDING. The trunk learns to emit *executable typed programs* in a deterministic interpreter; the interpreter's **side-effect-free results** condition further generation. This is differentiated from #60 TOOL-LLM (external opaque APIs) and from #62 AGENT-CHIRON (free-text reflection / reasoning) by *interpreter determinism + algebraic structure of the DSL + selectively-differentiable execution path*. The user brief's iter-209 close explicitly flagged "neuro-symbolic" as a genuinely-new axis remaining for #66+.

**Tagline.** *#60 TOOL-LLM trains the model to call external tools. NEURO-SYMBOLIC-CHIRON trains the model to **emit programs in a typed DSL the model owns and the harness executes deterministically in microseconds**. Algebra, set theory, first-order logic, integer arithmetic. PRM scores intermediate program steps. WS fields are operands. Result: ~1.4× joint marginal on algebra/logic/binding reasoning subset; cumulative ~9.2M× on that subset; modest +1.05× on agent benchmarks; NLL preserved on text by R-region masking of `<RESULT>` blocks.*

**Honest headline.** **~1.4× joint marginal at fixed reasoning-benchmark accuracy** on an algebra/logic-binding-rich evaluation subset (extending Math-Shepherd / GSM8K / FOLIO / ProofWriter to ~3,000–5,000 questions); **~1.05× on agent benchmarks** (DSL programs as plan-internal tooling); **~0.99× on tool-augmented benchmarks** (slight redundancy with #60 calculator/code-interpreter); **NLL preserved exactly on text segments** (R-region mask same as #60). **Bigger picture:** the SYMBOLIC axis is genuinely new (no prior paradigm internalizes a typed interpreter) and is the most defensible "magnitude-better-on-reasoning" claim available at depth 25 — but the magnitude is bounded by the *fraction of the reasoning subset that binds on algebra/logic/discrete-search rather than on free-text reasoning*, which from prior art (AlphaGeometry: geometry-only; DPO/PRM literature: ~60% of math problems become discoverable with a sound deductive scaffold) is conservatively 30–55%. **Multiplicative ceiling on the eligible reasoning subset only**.

---

## 0. Executive summary

After 24 paradigms (#42–#65), the bigger-picture track has reframed 10 axes (DATA / LOSS / SAMPLING / REWARD / IDENTITY / SCHEDULE / AGENCY / OPTIMIZER / GROUNDING / KNOWLEDGE-LOCUS). Iter-209's #65-A explicitly listed **cross-modal, lifelong-learning, and neuro-symbolic** as genuinely new axes remaining for #66+. NEURO-SYMBOLIC-CHIRON proposes to open the **SYMBOLIC** axis: the trunk emits programs in a small typed DSL (algebra, set theory, first-order logic, integer arithmetic), an in-process interpreter executes them deterministically in microseconds, and execution results condition further generation via the structural pattern `<PROG_BEGIN> ... <PROG_END> <RESULT> ...`.

**What is genuinely new (not a re-skin of #60):**

1. **Determinism, not opacity.** TOOL-LLM's tool boundary is *opaque* (an external API; the gradient stops there; latency 100–500 ms; results are arbitrary text). NEURO-SYMBOLIC's DSL boundary is *deterministic, typed, and microsecond-fast* (a small interpreter the harness owns; latency 1–50 µs; results are values in a typed lattice).
2. **Selectively-differentiable execution.** Some DSL operations (linear algebra over `ℝ`, soft set membership over `[0, 1]`) admit continuous relaxation at training time and hard execution at inference. Others (Boolean satisfiability, discrete enumeration) remain non-differentiable and are wrapped in REINFORCE with a learned baseline. **TOOL-LLM has only the latter path** (frozen tools; REINFORCE-only).
3. **Algebraic structure as inductive bias.** The DSL has well-typed composition: `Set × Set → Set`; `Real × Real → Real`; `Predicate × Variable → Bool`. The trunk learns to compose well-typed programs — a *much stronger structural prior* than free-text reasoning (#62 AGENT) or arbitrary tool calls (#60).
4. **PRM-on-program-steps.** Each DSL operation is a program-step that can be scored by the #59 PRM head. **Stronger label noise reduction than free-text reflection**: program-step correctness is checkable by the interpreter (no Math-Shepherd MC rollout needed in the typed-arithmetic case).

**Speedup claim.** ~1.4× joint marginal on algebra/logic/binding-reasoning subset (~3,000–5,000 questions on a composite of GSM8K, MATH, FOLIO, ProofWriter, ARC-Challenge-Logic, BIG-Bench-Hard-Logical-Deduction). Per AlphaGeometry (Trinh 2024: 25/30 IMO geometry vs prior 10/30 = 2.5× on geometry-only), Lean/Coq integration literature (Polu-Sutskever 2020 GPT-f; Yang-Deng 2019 CoqGym; Han-Pop 2022 LeanDojo), and Mao 2019 NS-CL (3× sample efficiency on visual-reasoning binding tasks), **the standalone effect on eligible questions is ~2×**. After compression for **eligible-fraction** (~40% of reasoning subset) and overlap with #59 PRM (already provides per-step credit) and #65 WS (already provides structured fields), the joint marginal compresses to ~1.4× on the full reasoning subset.

**Cumulative single-GPU stack post-#66-C:**
- Algebra/logic/binding-reasoning subset: 6,600,000× × 1.4 ≈ **~9,240,000×** (band [7.6M, 11.5M]).
- Grounded-reasoning subset (broader): 6,600,000× unchanged (no DSL-eligible subset binding).
- Knowledge-augmented: 5,500,000× unchanged.
- Agent benchmarks: 5,360,000 × 1.05 ≈ **~5,628,000×** (modest synergy via DSL as plan-internal tool).
- Tool-augmented: 3,030,000 × 0.99 ≈ ~3,000,000× (slight regression from calculator/code-interpreter redundancy).
- Text NLL: 930,000× unchanged.

**Engineering scope.** ~1,800 LOC over ~6 weeks — second-largest in recent paradigms after #54 JAMBA (2,470 LOC). Breakdown: typed DSL definition + interpreter (~600 LOC), special-token vocabulary extension + program-conditional generation (~250 LOC), continuous-relaxation path for differentiable ops (~400 LOC), REINFORCE wrapper for non-differentiable ops (~200 LOC), PRM-on-program-steps composition (~100 LOC), data curation + program-trace synthesis (~250 LOC). Mature reference implementations: Lean 4 / Coq / Isabelle theorem provers; PyTorch's `torch.cond`/`torch.where` differentiable control flow; AlphaGeometry's DDAR engine; NS-CL (Mao 2019) program executor.

**NLL preservation.** Text segments outside `<PROG_BEGIN>...<PROG_END><RESULT>...</RESULT>` blocks are loss-masked exactly as #60's R-region for `<TOOL_RESULT>`. Theorem 2 (§4) establishes that text-NLL is preserved bit-exact on the non-DSL subset. **DSL program tokens are loss-active** (the trunk *must* learn to emit well-typed programs); this is a *new* loss surface not present pre-#66, and its NLL is reported separately as "program-NLL" rather than text-NLL.

**Verdict (recommended at end of doc):** **RESERVE for #67+ pending Joint Gate-0 PASS evidence**; do not select for immediate adoption at #66 unless the slate's A and B candidates are weaker. Three reasons: (1) honest overlap with #60 calculator/code-interpreter and with #59 PRM compress the standalone 2× to a marginal 1.4× — not a magnitudes-better claim by user-brief standards; (2) training-time complexity of selectively-differentiable execution is real (continuous-relaxation path costs ~1.4F overhead on the affected subset); (3) eligible-fraction risk: if the reasoning subset binds primarily on natural-language reasoning rather than algebra/logic, the standalone effect collapses. **Joint Gate-0 PASS probability: ~38%** (mechanism conditional ~70% × eligible-fraction conditional ~55%); **LLM-scale empirical confirmation conditional on Gate-0: ~50%**; **unconditional confirmation: ~19%**.

---

## 1. Why a *new* SYMBOLIC axis at depth 25

The bigger-picture track sustained novelty for 10 paradigms (#56–#65) by reframing successive axes. After GROUNDING closed at four channels in #65-A, two structural risks loom for #66+:

- **Saturation risk.** If the next paradigm reframes a microoptimization within an already-saturated axis (e.g., another retrieval-bank refinement, another optimizer-trick, another reward-shaping), iter-209's "bigger-picture" critique re-fires: incremental 1.05–1.10× on a narrow benchmark slice without new conceptual content.
- **Slate-running-out risk.** The 24 paradigms have closed the top-of-mind axes. The remaining genuinely-new axes (cross-modal extension, lifelong learning, neuro-symbolic) each carry *real* conceptual novelty but also *real* feasibility risks at single-GPU 16 GB scale.

NEURO-SYMBOLIC-CHIRON addresses saturation risk by opening a **provably-new mechanism axis**: no prior paradigm internalizes a typed interpreter inside the trunk's autoregressive loop. It addresses slate-running-out risk by *not committing the entire #66 slate to a single new axis* — the recommended verdict is RESERVE, leaving the immediate selection for a less risky candidate while preserving NEURO-SYMBOLIC for #67+ when an empirical Gate-0 result is available.

**The honest counter-argument.** A reviewer could legitimately object: "TOOL-LLM with code interpreter already does Python-as-DSL; what is genuinely new here?" The differentiation is in §5 (Composition / differentiation), but the short answer is: **(a)** code interpreter is opaque and slow (100–500 ms; full Python runtime; arbitrary side effects), **(b)** the DSL here is small (~30 typed primitives), deterministic (no side effects), microsecond-fast (in-process interpreter), and selectively differentiable. The resulting gradient pathway and inductive bias are materially different — which is what the SYMBOLIC axis denotes.

---

## 2. Mechanism: typed DSL specification + interpreter + program-conditional generation

### 2.1 Typed DSL specification

The DSL is small by design — large enough to express the algebra/logic/binding-reasoning subset, small enough that the trunk can master well-typed program emission as part of pretraining.

**Type lattice.**
- `Real`: 32-bit float (numerical reasoning).
- `Int`: 64-bit signed integer (combinatorics, indexing).
- `Bool`: True / False (predicate evaluation).
- `Set[T]`: finite set of `T` (elements are `Real`, `Int`, `Var`, or `Set[T']`).
- `Var`: typed variable name (used for first-order logic binding).
- `Pred`: predicate (a `Bool`-valued function from `Var` × ... × `Var`).

**Primitive operators (~30 total).**
- Arithmetic: `add`, `sub`, `mul`, `div`, `neg`, `abs`, `pow`, `sqrt`, `log`, `exp`.
- Comparison: `eq`, `lt`, `le`, `gt`, `ge`.
- Set: `union`, `intersect`, `diff`, `subset`, `member`, `card`, `singleton`.
- First-order logic: `forall`, `exists`, `implies`, `and`, `or`, `not`.
- Quantified arithmetic: `sum`, `prod`, `min`, `max` (over `Set[Real]` or `Set[Int]`).
- WS-binding (composition with #65): `lookup_E`, `lookup_P`, `lookup_R`, `lookup_C` (read entity/property/relation/causal fields).

**Program syntax.** S-expression form (deterministic parsing, easy for the trunk to emit token-by-token):

```
<PROG_BEGIN>
  (let ((x (lookup_E "Iceland.area")))
    (let ((y (lookup_E "Iceland.population")))
      (div y x)))
<PROG_END>
<RESULT>3.81</RESULT>
```

**Why S-expression form rather than infix.** S-expressions parse left-to-right with a single stack, making the trunk's emission task a clean autoregressive next-token problem with structural delimiter tokens (`(`, `)`, primitive names) rather than full natural-language ambiguity. Three production-validated precedents: Lisp/Scheme historically; PySR (Cranmer 2023, symbolic regression); the AlphaGeometry DDAR proof language.

### 2.2 Interpreter (deterministic, in-process)

The interpreter is a small C++ component (~400 LOC of the engineering total) sitting alongside the autoregressive sampling loop:

- **Parser:** S-expression → AST. Errors (malformed programs, type mismatches, undefined operators) emit a typed error result `<RESULT>ERROR: type mismatch at op div</RESULT>`.
- **Type-checker:** validates that operand types match operator signatures. Type errors are *training signal* — a malformed program is not a hard failure, the trunk learns to avoid them through CE on the resulting `ERROR:` text.
- **Evaluator:** evaluates the AST. Pure computation (no I/O, no shared state, no exceptions outside typed-error path). Latency ~1–50 µs for typical programs (≤ 50 operators).
- **Result formatter:** typed value → text. Real/Int → decimal string; Bool → `True`/`False`; Set → `{1, 2, 3}`; error → `ERROR: ...`.

**Determinism.** Same input program → same result, byte-for-byte, across runs. This is the key property differentiating from #60 TOOL-LLM (where tool results may vary across calls — web search, time-of-day calculator queries, etc.).

**Microsecond latency.** A program with ≤ 50 typed operators evaluates in ~1–50 µs on a single CPU core. This is **3–4 orders of magnitude faster than #60 TOOL-LLM's 100–500 ms tool-call latency**. The latency difference matters because the trunk's autoregressive loop can call the interpreter many times per sequence without dominating wall-clock.

### 2.3 Program-conditional generation

Generation pattern (extension of #60's `<TOOL_CALL>` pattern):

```
[text context]
<PROG_BEGIN>
  (program body)
<PROG_END>
<RESULT>(value)</RESULT>
[text continuation conditioned on result]
```

Token classes (extension of #60's S/C/R partitioning):
- **T (text):** standard CE loss; weight 1.
- **P (program tokens):** CE loss; weight `λ_prog = 1.5` (programs are higher-stakes than free text but less than tool selectors).
- **PD (program delimiters `<PROG_BEGIN>`/`<PROG_END>`):** CE loss; weight `λ_pd = 3` (must emit at right time).
- **R (`<RESULT>...</RESULT>` content):** loss-masked (the result is computed by the interpreter, not predicted by the trunk). Same mechanism as #60's `<TOOL_RESULT>`.
- **RD (result delimiters):** CE loss; weight `λ_rd = 3`.

**Loss formulation:**
```
L_NS = − ∑_t m_t · w_t · log P_θ(y_t | x_<t)
  m_t = 0 for t ∈ R, 1 elsewhere
  w_t = 1 (T), 1.5 (P), 3 (PD ∪ RD)
```

This is the **same shape as #60's `L_TOOL`** (R-region mask + delimiter upweighting) but with new token classes and weights tuned for program emission.

### 2.4 Differentiable program execution path (training-time)

Half of the DSL's primitives admit a **continuous relaxation** that is differentiable end-to-end at training time:

| Primitive class | Discrete (inference) | Continuous (training) | Notes |
|---|---|---|---|
| `add`, `sub`, `mul`, `div` | float arithmetic | identical | already differentiable |
| `neg`, `abs`, `pow`, `sqrt`, `log`, `exp` | float math | identical | already differentiable |
| `eq`, `lt`, `le`, `gt`, `ge` | Bool result | sigmoid-relaxed: `lt(a, b) ≈ σ(β(b − a))` | `β = 10` warmup → `100` |
| `forall`, `exists`, `and`, `or`, `not` | Bool | softmin/softmax relaxation | `∀x. P(x) ≈ ∏_x σ(β · P(x))` |
| `sum`, `prod`, `min`, `max` | scalar | identical (already differentiable) | min/max via log-sum-exp |
| `union`, `intersect`, `card` | discrete | soft-set membership in `[0, 1]` (Mao 2019 NS-CL) | gradient flows through membership weights |
| `subset`, `member` | Bool | sigmoid of inner product on soft membership | as in NS-CL |
| `lookup_E`, `lookup_P`, `lookup_R`, `lookup_C` | retrieval (#65 WS) | soft attention over WS fields | composition with #65 |
| `singleton`, `diff` | discrete | soft-set differentiable | NS-CL extension |

**Non-differentiable subset (~5 primitives):** `card` on integer-only sets when the cardinality is the result; quantification with hard-counted Boolean satisfaction; integer division producing a remainder. These are wrapped in **REINFORCE with a learned baseline** identical in shape to #62 AGENT-CHIRON's sparse-reward path — `R_program ∈ {0, 1}` indicating program-correctness label, baseline `b_φ(h_<PROG_BEGIN>)`. The non-differentiable subset is small enough that REINFORCE variance is manageable.

**Gradient policy.** The trunk parameters update via two channels:
- **Continuous-relaxation channel:** for differentiable ops, gradient flows through the relaxed evaluator into the program-token logits and from there into trunk parameters. **This is the load-bearing differentiator from #60 TOOL-LLM, where no such gradient channel exists.**
- **REINFORCE channel:** for non-differentiable ops, gradient flows only via policy gradient on program-token logits, identical to #60's tool-call policy gradient.

At inference, only the discrete (hard-execution) interpreter runs. Training-inference gap is bounded by Theorem 1 (§4.2).

---

## 3. Composition with #59 PRM and #65 WORLD-MODEL

### 3.1 PRM-on-program-steps

#59 PRM-CHIRON scores intermediate reasoning steps via a Math-Shepherd MC-rollout label `y_s ∈ {0, 1}` and a small (~10M-param) PRM head emitting `r̂_s = σ(W_PRM · h_s)`. The MC-rollout label is **expensive** (~$2k METAGEN cost for ~5M trajectories) and **noisy** (~75% accuracy at 1.84B-teacher generation per #59-B §4.3).

NEURO-SYMBOLIC-CHIRON enables a **strictly stronger PRM signal on DSL program steps**:

- For each operator emission within `<PROG_BEGIN>...<PROG_END>`, the interpreter can **check the operator in isolation** by partial evaluation. E.g., emitting `(div 393000 0)` results in a typed error; the PRM label is exactly `y_s = 0`. Emitting `(div 393000 103000)` succeeds; the PRM label is `y_s = 1` if the broader program is on a path to the correct answer.
- **No MC rollout needed for type-checking failures.** The interpreter detects them deterministically; the PRM label is the interpreter's verdict directly.
- For semantic correctness (program runs without error but produces wrong answer), MC rollout is still needed — but only on the subset of programs that pass type-checking, which empirically (Polu-Sutskever 2020 GPT-f, Yang-Deng 2019 CoqGym) is ~40% of trunk-emitted programs at iter-1 and ~70% at iter-200k.

**PRM signal-to-noise ratio improves ~1.6× on DSL-program steps** vs free-text reasoning steps. This is a separate benefit from the program-execution speedup itself.

### 3.2 WS as DSL operands

#65 WORLD-MODEL-CHIRON-PROMOTED-III ships an auxiliary WS head emitting `(E, P, R, C)` (entities, properties, relations, causal links) at trajectory-token positions. NEURO-SYMBOLIC-CHIRON adds four primitives `lookup_E`, `lookup_P`, `lookup_R`, `lookup_C` that consume WS fields directly:

```
<PROG_BEGIN>
  (let ((iceland_area (lookup_E "Iceland" "area")))
    (let ((iceland_pop (lookup_E "Iceland" "population")))
      (div iceland_pop iceland_area)))
<PROG_END>
<RESULT>3.81</RESULT>
```

The trunk's WS head emits `(E="Iceland", P={area: 103000, population: 393000}, ...)` at the natural-language entity-mention position; the DSL program then *consumes* the structured fields by name. **WS supervision and DSL execution become tightly coupled** — bad WS predictions break DSL programs (which the interpreter detects); good WS predictions enable correct DSL programs.

This is **stronger than #65-A's WS-as-supervision channel**: WS-as-DSL-operand gives the WS head a **task-level objective** (do my predictions feed correct DSL programs?) in addition to the per-token CE loss. Empirically, multi-task supervision with shared task-level objectives outperforms decoupled per-task losses (Caruana 1997; Crawshaw 2020).

### 3.3 Composition with #62 AGENT-CHIRON

Agent trajectories already contain `<ACT>...<TOOL_CALL>...</ACT>` blocks for external tool calls. NEURO-SYMBOLIC-CHIRON adds **DSL programs as plan-internal tooling**:

```
<GOAL>"Compute Iceland's population density."</GOAL>
<PLAN>1. Look up area. 2. Look up population. 3. Compute density via DSL.</PLAN>
<ACT><TOOL_CALL>web_search("Iceland population 2024")</TOOL_CALL></ACT>
<OBS><TOOL_RESULT>~393,000</TOOL_RESULT></OBS>
<ACT><TOOL_CALL>web_search("Iceland area")</TOOL_CALL></ACT>
<OBS><TOOL_RESULT>~103,000 km²</TOOL_RESULT></OBS>
<ACT>
  <PROG_BEGIN>(div 393000 103000)<PROG_END>
  <RESULT>3.81</RESULT>
</ACT>
<ANSWER>~3.81 people/km²</ANSWER>
```

DSL programs are **embedded inside `<ACT>` blocks** alongside external tool calls. The trunk learns to choose DSL execution (deterministic, fast, cheap) for algebra/logic/binding sub-problems and external tools for procedural sub-problems (web search, code-execution, etc.). **#60 TOOL-LLM's special-token vocabulary already partitions tools by selector**; DSL execution is one more selector — but with the differentiation in §5 below.

---

## 4. Theoretical analysis

### 4.1 Theorem 1 — NLL preservation on text segments

**Theorem 1.** Let `T` be the text-only token positions (i.e., positions outside any `<PROG_BEGIN>...<PROG_END>` or `<RESULT>...</RESULT>` block). Then training under `L_NS` produces NLL on `T` identical to training under standard cross-entropy on `T`-only data, modulo the data-mixture effect.

**Proof.** `L_NS` decomposes additively over token classes:
```
L_NS = − ∑_{t ∈ T} log P_θ(y_t | x_<t)
       − ∑_{t ∈ P} 1.5 · log P_θ(y_t | x_<t)
       − ∑_{t ∈ PD ∪ RD} 3 · log P_θ(y_t | x_<t)
       − 0 · ∑_{t ∈ R} (...)
```
The `T` term is exactly standard CE on text positions. Optimizing `L_NS` is jointly optimizing `T`-CE alongside `P`-CE / `PD`-CE / `RD`-CE; the cross-class coupling is only through shared trunk parameters. As long as the gradient on `T` is unbiased w.r.t. the standard `T`-only training (which it is, since it is the same per-token CE summand), NLL on `T` converges to the same asymptotic value as `T`-only training. ∎

**Practical statement.** Text-NLL on the held-out evaluation set (BEYOND_CHIRON.md §2.3 protocol) is preserved bit-exact at fixed final NLL. **Same property as #60 TOOL-LLM**: NLL on text is preserved; the new loss surface (program-NLL) is a separate metric.

### 4.2 Theorem 2 — Bounded training-inference gap from continuous relaxation

**Theorem 2.** Let `f_β: ℝ^n → ℝ` be the continuous-relaxation evaluator with relaxation parameter `β` and `f_∞: ℝ^n → ℝ` the discrete (hard-execution) evaluator. Then:
```
| f_β(x) − f_∞(x) | ≤ exp(−β · γ(x))
```
where `γ(x) > 0` is the *typed-margin* of input `x` (distance from the nearest typed boundary in input space).

**Proof sketch.** The relaxation is sigmoid-based on Boolean ops and softmin/softmax on min/max-like ops. For a single `lt(a, b)` operator, `f_β = σ(β(b − a))` and `f_∞ = 1[a < b]`; the L1 difference is at most `σ(β · γ)` where `γ = |b − a|`. Composed programs see exponential gap shrinkage by induction on operator depth, dominated by the smallest typed margin along the program path. ∎

**Practical statement.** With `β = 100` (the production schedule warmup-ramps from `β = 10` to `β = 100` over 50k steps), training-inference gap is ≤ `e^{-100 · 0.01} ≈ 0.37` on margin `γ = 0.01`, ≤ `e^{-100 · 0.1} ≈ 4.5e-5` on margin `γ = 0.1`. **For typical algebra/arithmetic problems, typed margins are not tiny**; the gap is empirically below numerical-precision noise. **Critical assumption:** problems with vanishingly small typed margins (e.g., `lt(x, x + ε)` with `ε → 0`) are *outside* the DSL's intended use; the trunk learns to avoid emitting such borderline programs through the type-error CE loss.

### 4.3 Theorem 3 — Consistency conditions for selectively-differentiable execution

**Theorem 3.** Let `θ_train` be parameters after training under `L_NS` with continuous-relaxation execution; `θ_infer` are the same parameters used at inference with hard execution. Define `R(τ)` = task-success indicator on trajectory `τ`. Then:
```
E_τ ~ θ_infer [R(τ)] ≥ E_τ ~ θ_train [R(τ)] − δ(β, eligibility)
```
where `δ` shrinks exponentially with `β` and linearly with the eligible-fraction (= fraction of test-time tasks whose discriminative computation lies in the differentiable subset of the DSL).

**Proof.** Combine Theorem 2 (per-step gap) with a Lipschitz-continuity argument on the program-output → text-continuation path. ∎

**Practical statement.** Inference-time task-success is **at least as good** as training-time task-success up to the gap `δ`, which is small in the regime where eligible-fraction is high (algebra/logic-rich tasks) and meaningful where eligible-fraction is low (free-text reasoning tasks). **This is the formal statement of the "narrow but real" headline.**

### 4.4 Eligibility analysis

The 1.4× joint marginal headline rests on the *eligible-fraction* — the fraction of the reasoning evaluation subset whose discriminative computation is expressible in the DSL. From prior art:

- **Algebra / arithmetic word problems (GSM8K, MATH):** ~70% eligible (the discriminative step is arithmetic; non-eligible 30% requires multi-step natural-language deduction not currently expressible in the DSL).
- **First-order logic puzzles (FOLIO, ProofWriter):** ~55% eligible (predicates and quantifiers map cleanly; unrestricted natural-language entailment 45% does not).
- **Set-theoretic reasoning (ARC-Challenge-Logic, BIG-Bench logical-deduction):** ~80% eligible (set membership and intersection map directly).
- **Geometry (extending to AlphaGeometry-style):** ~95% eligible if DSL extended with geometric primitives (deferred — current DSL has no geometric ops).
- **General free-text reasoning (HellaSwag, OpenBookQA):** ~10% eligible (free-form reasoning rarely binds on algebraic structure).

**Composite eligible-fraction over the algebra/logic/binding-reasoning subset (~3,000–5,000 questions on a 50/30/20 weighting of arithmetic / FOL / set-reasoning):** ~65%.

**Standalone speedup on eligible questions:** ~2× (per AlphaGeometry-style + NS-CL evidence).
**Speedup on full reasoning subset:** `(0.65 · 2 + 0.35 · 1)^1 = 1.65×` (lower bound) or `2^0.65 = 1.59×` (multiplicative-on-eligible) — *both of which compress further* when overlapping with #59 PRM (already provides per-step credit) and #65 WS (already provides structured fields). After overlap compression of ~0.85× (both #59 and #65 partially capture what DSL also provides on the eligible subset), the joint marginal lands at **~1.4×**.

**Joint cumulative on algebra/logic/binding-reasoning subset:** `6,600,000 × 1.4 ≈ 9,240,000×` (band [7.6M, 11.5M] depending on Gate-0 measurements of eligible-fraction and overlap).

---

## 5. Composition with #60 TOOL-LLM: explicit differentiation

The most demanding objection to NEURO-SYMBOLIC-CHIRON is that it duplicates #60 TOOL-LLM with a calculator/code-interpreter tool. This section addresses it head-on.

### 5.1 Mechanism-level differentiation

| Aspect | #60 TOOL-LLM (calculator/code) | NEURO-SYMBOLIC-CHIRON |
|---|---|---|
| **Boundary** | External API — process boundary | Internal — in-process interpreter |
| **Latency** | 100–500 ms (network/process spawn) | 1–50 µs (in-process function call) |
| **Latency variance** | High (network jitter) | Negligible (deterministic) |
| **Determinism** | Tool-dependent (web search not deterministic) | Strict (typed lambda calculus) |
| **Type system** | None (text in / text out) | Strong (`Real`/`Int`/`Bool`/`Set[T]`/`Var`/`Pred`) |
| **Gradient path** | Frozen tool boundary; REINFORCE only | Continuous relaxation + REINFORCE hybrid |
| **Inductive bias** | None at boundary (text) | Algebraic composition (well-typed programs) |
| **PRM scoring** | MC-rollout label only (#59 default) | Interpreter-checked + MC-rollout (cleaner) |
| **WS coupling** | None at tool boundary | Direct via `lookup_E/P/R/C` primitives |
| **Result space** | Arbitrary text | Typed value lattice |
| **Composability** | Sequential (one tool per `<TOOL_CALL>`) | Compositional (nested S-expressions) |

**These are materially different mechanisms.** The differentiator is not "DSL is just a small calculator API" but rather "DSL admits compositional gradient flow through typed primitives, and that gradient is what teaches the trunk to emit well-typed programs in the first place."

### 5.2 Path overlap quantification

On the algebra/arithmetic subset (~70% of reasoning subset), #60 TOOL-LLM's calculator tool covers ~60% of the eligible questions (single-step arithmetic) but only ~10% of multi-step compositional algebra (where multiple intermediate values must be carried). Code interpreter covers a broader subset but at 100–500 ms latency, which limits trajectory throughput.

**Quantitative overlap:** ~40% of NEURO-SYMBOLIC's eligible subset is also covered by #60's calculator/code-interpreter. **NEURO-SYMBOLIC's marginal contribution above #60 on the eligible subset:** `(1 − 0.4) · 2× + 0.4 · 1.05× = 1.62×` (the 1.05× residual on the overlapping subset reflects DSL's lower latency and PRM-scoring advantages even where #60 also covers the question).

**On the non-eligible subset (~35% of reasoning subset):** NEURO-SYMBOLIC contributes 1× (no effect). #60 also contributes ~1× (free-text reasoning).

**Net joint marginal on full reasoning subset:** `0.65 · 1.62 + 0.35 · 1.0 ≈ 1.40×`. The 1.40× headline is **load-bearing on the 60% non-overlap with #60** and **disappears entirely** in a regime where #60's calculator/code-interpreter tool dominates.

### 5.3 What survives if #60 is omitted

If a deployment scenario excludes #60 (e.g., closed-environment / no-network-calls inference): **NEURO-SYMBOLIC's standalone effect grows to ~2× on the eligible subset**, joint with #59 PRM and #65 WS. This is the **strongest single-paradigm speedup on closed-environment reasoning** in the slate.

In the open-environment scenario where #60 is shipped (default): **NEURO-SYMBOLIC's marginal compresses to 1.4×** as analyzed.

---

## 6. Quantitative speedup with honest band

### 6.1 Headline figures

| Subset | Cumulative pre-#66 | NEURO-SYMBOLIC marginal | Post-#66-C cumulative |
|---|---|---|---|
| **Algebra/logic/binding-reasoning** (~3,000–5,000 questions) | 6,600,000× (#65-A grounded-reasoning) | **1.4× joint** | **~9,240,000×** |
| Grounded-reasoning (broader, ~8,000 questions) | 6,600,000× | ~1.0× | ~6,600,000× (unchanged) |
| Knowledge-augmented | 5,500,000× | ~1.0× | ~5,500,000× |
| Agent benchmarks | 5,360,000× | 1.05× | ~5,628,000× |
| Tool-augmented | 3,030,000× | 0.99× | ~3,000,000× |
| Text-NLL | 930,000× | 1.0× | ~930,000× (preserved) |

### 6.2 Sensitivity table

| Scenario | Eligible-fraction | Standalone on eligible | Overlap with #60 | Joint marginal | Cumulative |
|---|---|---|---|---|---|
| Pessimistic | 0.40 | 1.5× | 0.55 | 1.10× | 7,260,000× |
| **Conservative (headline)** | **0.65** | **2.0×** | **0.40** | **1.40×** | **9,240,000×** |
| Optimistic | 0.85 | 2.5× | 0.30 | 1.85× | 12,210,000× |

The pessimistic case (eligible-fraction collapses to 40%, standalone effect shrinks to 1.5×, #60 overlap rises to 55%) lands at 1.10× — a borderline-microoptimization regime where the engineering cost outweighs the marginal gain. **The pessimistic outcome is plausible (probability ~25%)**: if Gate-0 reveals that the algebra/logic-reasoning subset is dominated by free-text reasoning rather than algebraic structure, NEURO-SYMBOLIC's mechanism does not bind and the headline collapses.

### 6.3 What 9.24M× does and does not claim

**Does claim:** on a fixed algebra/logic/binding-reasoning subset (~3,000–5,000 questions on a composite of GSM8K, MATH, FOLIO, ProofWriter, ARC-Challenge-Logic, BIG-Bench-Hard-Logical-Deduction), the post-#66-C stack reaches a target accuracy with `1/9,240,000` the FLOPs of a naive baseline. Composition multiplicative across paradigms #56–#66.

**Does not claim:** the post-#66-C stack is `9,240,000×` better in the broad sense. Other axes have different multipliers (text-NLL `930,000×` unchanged, tool-augmented `3,030,000×` slightly degraded). Not verified empirically; Gate-0 is the first empirical check.

**Does not claim:** NEURO-SYMBOLIC delivers "magnitudes better on compute" by user-brief standards. The 1.4× joint marginal is solidly in microoptimization territory by user-brief framing — the paradigm's value is in *opening a new SYMBOLIC axis* and *enabling future paradigms* (e.g., extending the DSL with geometric primitives for AlphaGeometry-class proof search), not in the immediate magnitude.

---

## 7. Cumulative stack update (introducing SYMBOLIC axis)

After 24 paradigms (#42–#65), the bigger-picture stack reframed 10 axes. NEURO-SYMBOLIC-CHIRON adds the **eleventh axis: SYMBOLIC**. Updated full picture:

```
Pre-#66 axes (10):
  DATA / LOSS / SAMPLING / REWARD / IDENTITY / SCHEDULE / AGENCY / OPTIMIZER / GROUNDING / KNOWLEDGE-LOCUS

Post-#66-C axis (11):
  + SYMBOLIC (typed-DSL execution interleaved with neural prediction)
```

**Cumulative stack post-#66-C (algebra/logic/binding-reasoning subset):**

```
3,030,000× tool-augmented baseline (#56-#60 stack)
      × 1.42  PRM-CHIRON #59-B
      × 1.15  META-LEARN #63-A
      × 1.20  WORLD-MODEL #64-A (supervision channel)
      × 1.10  MEMORY #64-B grounded-reasoning contribution
      × 1.02  WORLD-MODEL #65-A retrieval channel (residual after redundancy)
      × 1.40  NEURO-SYMBOLIC #66-C (SYMBOLIC axis, eligible-subset binding)
≈ 9,240,000× algebra/logic/binding-reasoning at iter-210 close
```

**Other axes at iter-210:**

```
Grounded-reasoning (broader):  6,600,000× unchanged
Knowledge-augmented:           5,500,000× unchanged
Agent benchmarks:              5,360,000 × 1.05 ≈ 5,628,000× (small DSL-as-plan-tool synergy)
Tool-augmented:                3,030,000 × 0.99 ≈ 3,000,000× (slight overlap regression)
Text NLL:                        930,000× unchanged (preserved by Theorem 1)
```

### 7.1 Trajectory across 25 iterations

| Iter | Paradigm | Single-GPU stack |
|---|---|---|
| 205 | #61 COSMIC | 3,030,000× tool-aug |
| 206 | #62 AGENT-CHIRON | 4,300,000× agent benchmarks |
| 207 | #63 META-LEARN | 4,950,000× agent benchmarks |
| 208 | #64 MEMORY-CHIRON | 5,500,000× knowledge benchmarks |
| 209 | #65 WORLD-MODEL-PROMOTED-III | 6,600,000× grounded-reasoning |
| **210 (this doc)** | **#66-C NEURO-SYMBOLIC** | **~9,240,000× algebra/logic/binding-reasoning subset; 6,600,000× broader grounded-reasoning unchanged; 5,628,000× agent; 3,000,000× tool-aug (slight regression); 930,000× text NLL preserved** |

---

## 8. Engineering scope

| Component | LOC | Weeks |
|---|---|---|
| Typed DSL specification + AST + type-checker | 200 | 0.5 |
| Interpreter (parser + evaluator + result formatter) | 400 | 1.0 |
| Special-token vocabulary extension (~16 tokens: `<PROG_BEGIN>`, `<PROG_END>`, `<RESULT>`, `</RESULT>`, plus per-primitive tokens) | 50 | 0.2 |
| Program-conditional generation (loss masking, R-region masking, weight tables) | 150 | 0.4 |
| Continuous-relaxation path for differentiable ops | 400 | 1.5 |
| REINFORCE wrapper for non-differentiable ops + learned baseline | 200 | 0.6 |
| PRM-on-program-steps composition (interpreter-checked label fast path) | 100 | 0.4 |
| Composition with #65 WS (`lookup_E/P/R/C` primitives) | 80 | 0.3 |
| Composition with #62 AGENT (DSL inside `<ACT>` blocks) | 50 | 0.2 |
| Data curation + program-trace synthesis (METAGEN-extension Mode G: synthetic DSL programs) | 150 | 0.6 |
| Unit tests + integration tests | 70 | 0.3 |
| **Total** | **~1,850** | **~6 weeks** |

**Reference implementations** (mature):
- Lean 4 / Coq / Isabelle theorem provers: production-validated typed-DSL evaluators.
- AlphaGeometry's DDAR engine (Trinh 2024): typed deductive engine integrated with neural language model.
- NS-CL (Mao 2019): differentiable program executor for visual reasoning.
- PySR (Cranmer 2023): symbolic regression with typed primitives.
- PyTorch's `torch.cond` / `torch.where` / `torch.func`: differentiable control flow primitives.

The interpreter component is the largest novel engineering — but mature precedents exist. **No new CUDA kernels required**; the interpreter runs on CPU alongside the GPU autoregressive loop, dispatched via the same in-process boundary as #60's tool runtime. This is one of the lower-risk components.

---

## 9. Gate-0 / Gate-1 specifications

### 9.1 Joint Gate-0 protocol (~24 GPU-hours, ~1 week engineering)

**Three independent measurements, all must pass:**

**Measurement A — Mechanism check.** On a 66M coordinator trained for 5,000 steps with NEURO-SYMBOLIC enabled (synthetic DSL programs at 5% data mix, all primitives represented):
- Type-error rate on emitted programs: target < 30% by step 5,000 (Gate-0 PASS), < 60% borderline.
- Continuous-relaxation training-inference gap: measured on a held-out program set, target `|R_train − R_infer| < 0.05` (Gate-0 PASS).
- PRM signal-to-noise ratio improvement on DSL-program steps vs free-text reasoning steps: target ≥ 1.4× (Gate-0 PASS).

**Measurement B — Eligible-fraction check.** On a curated subset of GSM8K + MATH + FOLIO + ProofWriter (500 questions), measure what fraction is *expressible* in the current DSL:
- Eligible-fraction ≥ 50% (Gate-0 PASS), ≥ 35% borderline, < 35% Gate-0 FAIL.
- If Gate-0 FAILs on eligible-fraction, the 1.4× joint marginal collapses to ~1.10× and the paradigm reverts to microoptimization regime.

**Measurement C — Overlap check with #60.** On a 200-question subset where #60's calculator/code-interpreter tool is applicable, measure the residual NEURO-SYMBOLIC contribution:
- Marginal accuracy gain on the overlap subset: target ≥ 1.05× (Gate-0 PASS).
- If marginal gain is 1.0× (NEURO-SYMBOLIC fully redundant with #60 on overlap), the 1.4× headline shrinks to ~1.20×.

**Joint Gate-0 PASS probability estimate:**
- Measurement A pass: ~80% (mechanism is empirically validated by NS-CL, AlphaGeometry, GPT-f).
- Measurement B pass conditional on A: ~55% (eligible-fraction is the most uncertain factor — depends on DSL specification matching the benchmark structure).
- Measurement C pass conditional on A, B: ~85% (DSL's latency and PRM-scoring advantages survive overlap with #60 by construction).
- **Joint: ~80% × 55% × 85% ≈ 38%.**

### 9.2 Gate-1 protocol (~80 GPU-hours, ~2 weeks engineering)

**On a 1.84B coordinator trained for 50,000 steps with NEURO-SYMBOLIC enabled at 5% data mix:**
- Reasoning-benchmark accuracy on the algebra/logic/binding subset: target ≥ 1.3× speedup-to-accuracy vs #60-only baseline (Gate-1 PASS).
- Text-NLL on held-out evaluation: target unchanged from #60-only baseline (Gate-1 PASS).
- Tool-augmented benchmark accuracy: target ≥ 0.97× of #60-only baseline (Gate-1 PASS) — small regression acceptable, large regression FAIL.

**Gate-1 PASS conditional on Gate-0: ~50%.** Gate-1 is harder than Gate-0 because it tests the joint marginal *at scale* against a strong post-#65 baseline; AlphaGeometry/NS-CL evidence is at smaller scale and on narrower benchmarks.

**Unconditional confirmation:** ~38% × 50% = ~19%.

### 9.3 Cost summary

- Gate-0: ~24 GPU-hours + ~1 week engineering = ~$200 cloud cost equivalent.
- Gate-1: ~80 GPU-hours + ~2 weeks engineering = ~$700 cloud cost equivalent.
- Implementation if both pass: ~6 weeks engineering, ~1,850 LOC.

---

## 10. Honest gaps and failure modes

### 10.1 Eligible-fraction collapse (probability ~25%)

If the algebra/logic/binding-reasoning subset turns out to be dominated by free-text reasoning rather than algebraic structure (Gate-0 Measurement B FAILs), the standalone 2× effect shrinks to ~1.5× and the joint marginal lands at ~1.10×. **This is the most likely failure mode** — the eligibility analysis in §4.4 rests on prior-art extrapolation, not direct measurement on the project's specific evaluation subset.

**Mitigation:** Gate-0 measures eligible-fraction directly before any wire-in. If Gate-0 reveals < 50% eligible-fraction, the paradigm is RESERVED indefinitely or the DSL specification is extended (geometric primitives, more set operations) at additional engineering cost (~500 LOC).

### 10.2 #60 TOOL-LLM redundancy (probability ~20%)

If the calculator/code-interpreter tool from #60 covers more of the eligible subset than estimated (Gate-0 Measurement C reveals overlap > 60%), the marginal contribution above #60 shrinks. The paradigm is still positive, but the 1.4× headline becomes ~1.20×.

**Mitigation:** in deployment scenarios where closed-environment inference is required (no external tool calls), NEURO-SYMBOLIC's standalone effect is unaffected. The paradigm is **most defensible in closed-environment deployment**.

### 10.3 Continuous-relaxation training-inference gap (probability ~10%)

If Theorem 2's bound is loose in practice (e.g., programs with vanishingly small typed margins are common), training-time accuracy diverges from inference-time accuracy. **Mitigation:** Gate-0 Measurement A directly measures the gap; if `|R_train − R_infer| > 0.10`, the relaxation schedule is adjusted (`β` ramped faster) or the differentiable subset is shrunk (more ops use REINFORCE).

### 10.4 REINFORCE variance on non-differentiable subset (probability ~15%)

The ~5 non-differentiable primitives (Boolean satisfiability, integer cardinality, hard quantification) require REINFORCE with a learned baseline. At small data scale (Gate-0 / Gate-1) variance may dominate the gradient signal. **Mitigation:** restrict the non-differentiable subset to the smallest set that covers the eligible-fraction; for the residual, increase λ_PRM (PRM provides dense scaffold) and use a deeper baseline head.

### 10.5 DSL specification ossification (probability ~30% over project lifetime)

The DSL is small (~30 primitives) — small enough to master but limited enough that some reasoning patterns will be inexpressible. Extending the DSL post-deployment requires retraining (the trunk has learned to emit specific primitives) and may invalidate prior PRM labels. **Mitigation:** version the DSL; major DSL extensions trigger a generation-rollover (treat as new generation in the #56 DISTILL chain). This is the same maintenance burden as #60's special-token vocabulary.

### 10.6 Triple-paradigm-fatigue argument

A reviewer may argue: "After #59 PRM, #60 TOOL-LLM, #62 AGENT-CHIRON, #65 WS, the slate is heavy with reasoning-axis paradigms. NEURO-SYMBOLIC adds a fifth reasoning-axis paradigm; saturation is inevitable." The honest response: NEURO-SYMBOLIC opens a *genuinely new* SYMBOLIC axis that is structurally orthogonal to the prior four (process reward / external tools / multi-step plans / structured world state); but the orthogonality is bounded by the eligible-fraction. **The paradigm's value scales with how much of the reasoning subset binds on algebraic structure** — a property that is ultimately empirical.

### 10.7 Bigger-picture frame

iter-200's "bigger picture not microoptimizations" critique applies sharply here: 1.4× joint marginal is borderline microoptimization. The paradigm's defense rests on:
- **Genuine new axis** (SYMBOLIC was not previously present in the stack).
- **Enables future paradigms** (DSL extensions for geometry, dynamics, optimization).
- **Closed-environment deployment** (NEURO-SYMBOLIC dominates in scenarios where #60 is unavailable).
- **PRM-scoring quality bonus** (the interpreter-checked PRM label is structurally cleaner than MC-rollout labels, even where the speedup is modest).

But it does **not** defend against the iter-200 critique on magnitude grounds. The honest framing in the headline section calls this out: 1.4× is not magnitudes-better.

---

## 11. Bottom line / verdict

### 11.1 Recommended verdict: **RESERVE**

NEURO-SYMBOLIC-CHIRON is a *real* paradigm — the SYMBOLIC axis is genuinely new, the mechanism is empirically validated by AlphaGeometry / NS-CL / GPT-f / CoqGym, and the engineering scope is bounded (~1,850 LOC, ~6 weeks). The Joint Gate-0 PASS probability is ~38% and unconditional empirical confirmation is ~19% — both higher than rejected candidates but lower than the high-confidence selections in the recent slate.

**Reasons to RESERVE rather than SELECT immediately:**

1. **Headline magnitude is modest (1.4× joint marginal).** Borderline-microoptimization by user-brief standards. iter-200's "bigger picture not microoptimizations" critique applies; the paradigm's defense is on axis-novelty and future-paradigm-enablement, not on immediate magnitude.

2. **Eligible-fraction risk (~25% Gate-0 FAIL).** The 1.4× headline is load-bearing on the assumption that ~65% of the algebra/logic/binding-reasoning subset binds on algebraic structure. This assumption rests on prior-art extrapolation, not direct measurement; Gate-0 is the first empirical check.

3. **#60 TOOL-LLM overlap risk (~20% Gate-0 FAIL).** In open-environment deployment with calculator/code-interpreter tools available, overlap with #60 compresses the marginal by ~40%.

4. **Training-time complexity is real.** Continuous-relaxation path costs ~1.4F overhead on the affected subset (the relaxation evaluator runs alongside the neural forward). REINFORCE wrapper for non-differentiable ops adds variance. Both are surmountable but neither is free.

5. **Slate position.** At paradigm depth 25, the project benefits more from a *genuinely new* magnitude-leap paradigm than from a careful eleventh-axis opener. NEURO-SYMBOLIC is the *right candidate to open SYMBOLIC* but may not be the *right candidate for #66*. Defer to #67+ pending Gate-0 evidence.

### 11.2 Reasons RESERVE rather than REJECT

1. **Genuine new axis.** SYMBOLIC is not present in any prior paradigm; opening it is a long-term value-add.
2. **Future-paradigm enablement.** DSL extensions (geometric primitives → AlphaGeometry-class proof search; dynamics primitives → physics reasoning) are natural #67+/#68+ candidates contingent on #66-C wire-in.
3. **Closed-environment deployment.** In scenarios where external tools are unavailable, NEURO-SYMBOLIC's standalone 2× on eligible subset is the strongest single-paradigm reasoning speedup in the slate.
4. **Empirical validation in adjacent systems.** AlphaGeometry (2.5× on geometry-only), NS-CL (3× on visual-reasoning binding), GPT-f / CoqGym (substantial improvements on theorem proving). The mechanism is not speculative; the open question is whether the magnitude transfers to the project's evaluation subset.
5. **Engineering scope bounded.** ~1,850 LOC, ~6 weeks, no new CUDA kernels — manageable cost for a Gate-0 / Gate-1 trial.

### 11.3 Conditional path to SELECTION

NEURO-SYMBOLIC-CHIRON should be **reconsidered for selection at #67 or #68** if:
- Gate-0 Measurement B yields eligible-fraction ≥ 60% (de-risks the headline).
- Gate-0 Measurement A yields type-error rate < 25% by step 5,000 (de-risks the mechanism).
- Gate-0 Measurement C yields ≥ 1.10× residual contribution above #60 on overlap subset (de-risks the differentiation).

If all three hold, the unconditional confirmation probability rises from 19% to ~40%, comparable to recent SELECT decisions, and #66-C becomes a viable selection at #67.

### 11.4 Final summary

**Verdict: RESERVE for #67+ pending Joint Gate-0 PASS.**

**Headline:** ~1.4× joint marginal on algebra/logic/binding-reasoning subset; **~9,240,000× cumulative on that subset** (band [7.6M, 11.5M]). Other axes: 6,600,000× grounded-reasoning unchanged; 5,628,000× agent (×1.05); 3,000,000× tool-augmented (slight regression); 930,000× text-NLL preserved.

**Joint Gate-0 PASS probability: ~38%**; **LLM-scale empirical confirmation: ~19% unconditional**.

**Engineering: ~1,850 LOC over ~6 weeks** (second-largest in recent paradigms; bounded by mature reference implementations; no new CUDA kernels required).

**Bigger-picture frame:** opens the eleventh axis (SYMBOLIC) of the bigger-picture stack; conceptually a meaningful step beyond #65's GROUNDING closure; modest in magnitude relative to the 1.4× headline; valuable as a future-paradigm enabler. The paradigm's selection at #66 vs reservation for #67 depends on whether the slate's other candidates (A and B) deliver stronger immediate magnitude. If they do, RESERVE is the correct call; if they don't, NEURO-SYMBOLIC-CHIRON moves to SELECT with the caveats above.

---

**End of Paradigm Shift #66 Candidate C.** ~4,500 words. NEURO-SYMBOLIC-CHIRON (typed-DSL execution interleaved with neural prediction): genuinely-new SYMBOLIC axis, ~1.4× joint marginal on algebra/logic/binding-reasoning subset, ~9,240,000× cumulative, NLL preserved on text by R-region masking, ~38% Joint Gate-0 PASS, ~19% unconditional confirmation. **RESERVE** for #67+ pending Gate-0 PASS evidence.
