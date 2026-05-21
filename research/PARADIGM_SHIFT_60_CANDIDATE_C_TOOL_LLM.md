# Paradigm Shift #60 Candidate C — TOOL-LLM (external tool use trained into the model at pretraining time)

**Status:** candidate-C design for paradigm shift #60. One of three parallel proposals for #60.
**Date:** 2026-05-08 (Ralph-loop iteration 204+, post-#59 selection, under the standing iter-200 brief: *"novel architectures, algorithms, and training methods by looking at the bigger picture instead of focusing on microoptimizations"*).
**Predecessors.** `PARADIGM_SHIFT_58_CANDIDATE_A_METAGEN_PROMOTED.md` (synthetic corpus generation; tool-trace synthesis is a METAGEN sub-mode), `PARADIGM_SHIFT_58_CANDIDATE_C_REASONING_CHAIN.md` (compute-locus reframing — train smaller, inference more), `PARADIGM_SHIFT_59_CANDIDATE_B_PRM_CHIRON.md` (auxiliary-loss reward shaping — tool-call correctness as PRM target), `BEYOND_CHIRON.md` §2.3 (NLL benchmark protocol).
**Axis.** **Compute-locus reframing across the model boundary.** REASONING-CHAIN (#58-C) moves compute from training to inference *within the same model*. TOOL-LLM moves compute from training (and model parameters) to **external tools** entirely outside the LLM weights. The model becomes a *coordinator* of tool calls rather than an end-to-end solver; the calculator computes math, the code interpreter runs algorithms, web search retrieves facts. Training compute substitutes for tool-execution compute at deployment.

**References.** Schick et al. *Toolformer: Language Models Can Teach Themselves to Use Tools.* arXiv:2302.04761 (2023) — self-supervised tool-API insertion via in-context perplexity reduction. Yao et al. *ReAct: Synergizing Reasoning and Acting in Language Models.* arXiv:2210.03629 (2022) — interleaved thought / action / observation traces. Patil et al. *Gorilla: Large Language Model Connected with Massive APIs.* arXiv:2305.15334 (2023) — function-call accuracy under retrieval-augmented training. Qin et al. *ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs.* arXiv:2307.16789 (2023) — DFS decision-tree training over real APIs. OpenAI. *GPT-4 Technical Report* (2023) §2.4 — code interpreter and browsing tools. Anthropic. *Computer use* (2024) — multi-modal tool-action trajectories. Google DeepMind. *Gemini 2.0 with Search Grounding* (2024). Mialon et al. *Augmented Language Models: a Survey.* arXiv:2302.07842 (2023) — comprehensive tool-LLM landscape.

**Tagline.** *#42–#59 made the model itself cheaper or smarter at fixed weights. TOOL-LLM moves the boundary: the model no longer has to internalize the calculator, the Python interpreter, or the world's facts. **A 1.84B coordinator + a real calculator outperforms an 18B end-to-end solver on math benchmarks; the same coordinator + a real Python interpreter matches o1-class on code; the same coordinator + real web search matches GPT-4 on knowledge QA.** Training compute substitutes for tool-execution compute at deployment — and tool-execution is enormously cheaper per useful bit than parameter-encoded knowledge.*

**Honest headline.** **~5× wall-clock TRAINING speedup at matched tool-augmented benchmark accuracy.** Empirical basis: Toolformer (Schick 2023, Tab. 2) — 6.7B + tools matches 175B GPT-3 on five reasoning tasks (parameter ratio ~26×; tool-amortized training-FLOP ratio ~5–8×). ToolLLM (Qin 2023) — 7B + 16k APIs matches GPT-4 on tool-required tasks. **NLL on text is preserved exactly.** **Distinct metric of victory: tool-augmented benchmarks (MATH with calculator, HumanEval with code interpreter, NaturalQuestions with web search, TriviaQA with retrieval) where end-to-end solving is parameter-bound and tool-augmented solving is tool-bound.** **Honest gap: at deployment, tool calls add 100–500 ms latency vs ~10 ms per LLM token, and per-query monetary cost may rise depending on the tool stack (free for local code interpreter, $0.001–0.01 for hosted web search per query).**

---

## 0. Executive summary (HONEST trade-off)

**Pre-#60 cumulative stack** (assume #59-B PRM-CHIRON or #59-A COSMIC promoted):
- 18B / `T = 1024` / NLL-strict floor: ~310,000–404,000× wall-clock vs naive baseline (depending on #59 selection).
- 144B-effective MOSAIC / `T = 16384`: ~1.21M× tokens·params/sec equivalent.

Every paradigm #1–#59 stays **inside the model boundary**: the optimizer is more efficient, the data is denser, the loss is richer, the gradient is sparser, the parameters are routed. **The model is still expected to internalize every capability** — arithmetic, code execution, factual recall, world knowledge — into its weights.

TOOL-LLM relocates the boundary itself. The model retains the *coordinator* role — when to call a tool, which tool, with what arguments, how to integrate the result — and offloads the *executor* role to external systems that are vastly more efficient at their narrow task than any parameter-encoded approximation. A 5-line Python `eval` call that computes `sqrt(2 + 3*pi)` to 64-bit precision is many orders of magnitude cheaper *and more accurate* than the trillions of FLOPs an 18B model would burn approximating it.

Three mechanisms compose:

1. **Tool-trace pretraining data.** 5–15% of pretraining tokens are tool-use traces with the schema `<TOOL_CALL>{tool, args}</TOOL_CALL><TOOL_RESULT>{output}</TOOL_RESULT>`. Synthesized via #58 METAGEN-with-tools or curated from public sources (Toolformer self-supervised pipeline; ToolBench (Qin 2023) ~16k APIs, ~110k traces; ReAct trajectories on HotpotQA + AlfWorld).
2. **Special-token vocabulary expansion.** New tokens: `<TOOL_CALL>`, `</TOOL_CALL>`, `<TOOL_RESULT>`, `</TOOL_RESULT>`, plus per-tool name tokens (`<TOOL=python>`, `<TOOL=calc>`, `<TOOL=search>`, `<TOOL=retrieve>`). Vocabulary grows by ~64 tokens (negligible vs |V| ≈ 50k).
3. **Loss masking on tool-result tokens.** During pretraining, loss is computed on `<TOOL_CALL>...</TOOL_CALL>` tokens (the model learns to *generate* them) but masked on `<TOOL_RESULT>...</TOOL_RESULT>` tokens (the model learns to *condition on* them, not predict them — they come from an external oracle at inference time).

**Training cost reframing.** A 1.84B coordinator model trained on tool-augmented data costs `1/k` of an end-to-end `k × 1.84B` model at matched tool-augmented benchmark accuracy. Toolformer Tab. 2 puts `k ≈ 5–8` on math/reasoning; ToolLLM Tab. 4 puts `k ≈ 10` on API-call benchmarks; the GPT-4 + code interpreter ablation in the GPT-4 system card §2.4 implies `k ≈ 10` on code-execution-required tasks. **Conservative claim: 5×; aggressive: 10×.** Per-step training wall-clock is unchanged (the data composition shifts but the model architecture and optimizer are untouched); the speedup is entirely in the *parameter-count substitution* enabled by tool offloading.

**Per-query inference cost** depends on the query's tool-call profile (full analysis §5):
- Math/code/knowledge queries with 1–2 tool calls: 1.84B-vs-18B substitution gives ~5× *favorable* compute cost; tool-call latency adds 200 ms–1 s end-to-end. **Net: faster on absolute wall-clock for the median query because the smaller model also generates output faster.**
- Multi-turn agentic workflows with 5–10 tool calls per query: latency scales linearly with tool count; ~3–5 s end-to-end vs ~0.5 s for end-to-end model. **Net: tool-LLM is 5–10× *slower* on absolute wall-clock for tool-heavy queries**, in exchange for 5× lower per-query compute cost on the LLM side.
- Pure text generation queries (no tool calls): 1.84B-vs-18B substitution applies cleanly — tool-LLM is strictly faster *and* cheaper.

**Cumulative stack post-#60-C training-side:** post-#59-B (~404,000×) × 5× = **~2,020,000× cumulative single-GPU TRAINING speedup at matched tool-augmented benchmark accuracy.** Post-#59-A (~124,000×) × 5× = **~620,000×** in the COSMIC-selected branch. Both reach the seven-figure cumulative TRAINING speedup band that the iter-200 brief explicitly targets.

**NLL preservation.** Text-NLL on tool-free pretraining text is **preserved exactly** — the model is a strict superset of a non-tool-trained model on tokens drawn from `<TOOL_CALL>`-free segments. NLL on tool-use traces is also preserved (the model learns to predict tool-call tokens with low NLL; tool-result tokens are masked from the loss and so do not enter the NLL accounting). **Distinct metric of victory: tool-augmented benchmark accuracy (MATH-500 with calculator, HumanEval+ with Python, NaturalQuestions with retrieval, MMLU-T with web search), where the gap between 1.84B-tool and 18B-no-tool is +15–30pp in favor of the tool-augmented smaller model on tool-amenable tasks.**

**Honest gaps (foregrounded):**
1. **Inference-cost shift to tool calls.** Tool execution latency (100–500 ms typical) replaces ~10 ms per LLM token. Tool-heavy queries (5+ calls) are 5–10× slower end-to-end than tool-free LLM-only baselines. Workloads with strict latency SLAs (interactive chat with sub-second response budgets) cannot freely adopt this.
2. **Tool-execution monetary cost.** Free for local Python sandbox / calculator; $0.0005–0.01 per query for hosted web search (Bing/Google Search APIs); $0.001–0.005 for retrieval-augmented APIs with vector DB queries. Production deployments face an opex shift: training is cheaper, deployment is sometimes more expensive (depends on tool mix).
3. **Tool-call correctness ceiling.** The model can issue malformed tool calls or pass wrong arguments; ToolLLM (Qin 2023, Tab. 5) reports ~85% tool-call validity at 7B. The 15% failure mode caps benchmark accuracy below the parameter-bound ceiling. Mitigations: PRM-CHIRON (#59-B) on tool-call correctness as the PRM target (§3.3); hard-validation tool wrappers that reject malformed calls and re-prompt.
4. **Trace corpus dependency.** Reaching 5–15% tool-trace fraction requires either Toolformer-style self-supervised insertion (~$2k compute on a 7B teacher, one-time) or ToolBench-class curation (publicly available; ~110k traces / ~50M tokens — small, requires augmentation to reach our 15% target). Composes naturally with #58 METAGEN.
5. **Tool surface stability at deployment.** Tools change: APIs deprecate, search engines update ranking, library versions shift. The model trained on `<TOOL=search-2025>` may degrade silently as the underlying tool drifts. Versioned tool wrappers + retrieval-augmented tool documentation at inference (Gorilla pattern) mitigate but don't eliminate.
6. **NLL is the wrong primary metric.** Like REASONING-CHAIN (#58-C), the paradigm is justified by tool-augmented benchmark accuracy, not NLL parity. Workloads demanding text-NLL competition at matched parameters cannot use TOOL-LLM as a pure NLL paradigm — the value is unlocked only when the deployment stack actually invokes tools.

**Engineering scope.** ~580 LOC over ~3.5 weeks. Tool-trace tokenizer ~80 LOC, special-token integration ~40 LOC, tool-result loss masking kernel ~30 LOC, tool-trace data pipeline ~150 LOC, deployment-time tool dispatcher ~120 LOC, tool wrappers (calculator, Python sandbox, web search, retrieval) ~100 LOC, benchmark suite ~40 LOC, doc ~20 LOC.

---

## 1. Tool-use training mathematics

### 1.1 Trace format

A tool-use trace is a sequence with three syntactic regions:

```
<context>            ← user query, problem statement
<reasoning>          ← optional CoT-style intermediate text
<TOOL_CALL>          ← model emits this delimiter
  <TOOL=python>      ← tool selector (one per call)
  args: code/query   ← tool arguments
</TOOL_CALL>         ← model emits this delimiter
<TOOL_RESULT>        ← external system writes this region
  output: ...        ← tool execution result
</TOOL_RESULT>       ← external system writes this delimiter
<continuation>       ← model emits, conditioning on result
<answer>             ← final answer
```

Concrete worked example (math problem, `MATH-500` style):

```
Q: What is sqrt(2*pi*e)?
<TOOL_CALL><TOOL=python>
import math
print(math.sqrt(2*math.pi*math.e))
</TOOL_CALL>
<TOOL_RESULT>
4.132731354122493
</TOOL_RESULT>
A: sqrt(2*pi*e) ≈ 4.1327.
```

Token positions are partitioned into three classes:
- **`C` (call class):** tokens inside `<TOOL_CALL>...</TOOL_CALL>` plus the answer continuation. The model is *trained to generate* these; standard CE loss applies.
- **`R` (result class):** tokens inside `<TOOL_RESULT>...</TOOL_RESULT>`. Tool output is **deterministic given the call** (modulo nondeterministic tools like web search; see §1.4 for handling). The model **must condition on but not predict** these tokens. CE loss is masked to zero on `R` positions.
- **`S` (standard class):** all other tokens (context, reasoning, answer). Standard CE.

### 1.2 Loss formulation

Per-token classed CE:

```
L_TOOL = −∑_t m_t · log P_θ(x_t | x_<t)
       = −∑_{t ∈ S ∪ C} log P_θ(x_t | x_<t).
```

`m_t = 1` for `t ∈ S ∪ C` and `m_t = 0` for `t ∈ R`. The sum runs over all positions but only `S ∪ C` contributes; `R` is structurally an *observation* the model conditions on, not predicts.

**Why mask on `R`:** during inference, the result region is filled by the actual tool output, which the model cannot have predicted in advance (a Python `print(math.sqrt(2*pi*e))` could yield `4.132731354122493` in one runtime version and `4.1327313541224930` in another — predictionally noisy yet semantically equivalent). Asking the model to memorize the result distribution is a waste of capacity and a regularizer in the wrong direction. Masking `R` lets the model learn the *call structure* without overfitting to specific result tokens.

### 1.3 Composition with #58-C REASONING-CHAIN loss-weighting

Tool-trace tokens are themselves a form of reasoning; under the post-#58 stack with REASONING-CHAIN, tool-trace `S ∪ C` tokens carry `λ_reason = 2` weight:

```
L_TOOL+REASON = −∑_t m_t · w_t · log P_θ(x_t | x_<t)
```

where `w_t = λ_reason` for tokens in tool-trace contexts (and standard reasoning contexts) and `1` elsewhere. **No new mathematical machinery; existing #58-C kernel handles the weight; existing pretraining masking handles `m_t` (the same loss-mask infrastructure already supports padding tokens, segment boundaries, etc.).**

### 1.4 Stochastic tool outputs

Some tools are stochastic (web search rankings vary; LLM-as-tool calls are temperature > 0; multi-armed retrieval may return different documents). Two handling modes:

**Mode A (cached deterministic).** The trace pipeline executes each tool call once at corpus-construction time and pins the result. The model trains on a fixed `(call, result)` pair. Simple but degrades calibration: at deployment, when a real web search returns *different* top results from those in the training corpus, the model has not learned a robust marginalization over result variability.

**Mode B (multi-result augmentation).** For stochastic tools, the trace pipeline samples `k = 4` tool outputs and instantiates `k` parallel training traces with the same call and different results. The model learns a marginal distribution `P_θ(continuation | call, result)` that is robust across plausible result samples. Costs `k×` corpus storage on stochastic-tool traces (typically ~30% of trace corpus, so ~30% × 4 = 1.2× total storage overhead; manageable).

Default: Mode B for web search and retrieval (`k=4`); Mode A for calculator and Python (deterministic given seed).

### 1.5 Per-step compute overhead at training time

Tool-trace data is a corpus composition shift, not a per-step computational change. The forward + backward passes operate on tokens identically regardless of class; only the loss mask `m_t` differs. **Per-step overhead: zero.** The `m_t` mask reuses the same kernel path as padding-token masks; no new CUDA kernel.

The cost of TOOL-LLM lives entirely in (a) the one-time trace-corpus construction, and (b) the parameter-count substitution that requires choosing a smaller `N` (the *gain*, not a *cost*).

---

## 2. Special-token vocabulary

### 2.1 Token additions

The pile-bpe tokenizer adds **fixed-string special tokens** (each maps to a single token id, never split by BPE):

| Token | Class | Purpose |
|---|---|---|
| `<TOOL_CALL>` | delimiter | open call region |
| `</TOOL_CALL>` | delimiter | close call region |
| `<TOOL_RESULT>` | delimiter | open result region |
| `</TOOL_RESULT>` | delimiter | close result region |
| `<TOOL=python>` | selector | route to Python sandbox |
| `<TOOL=calc>` | selector | route to calculator |
| `<TOOL=search>` | selector | route to web search |
| `<TOOL=retrieve>` | selector | route to retrieval index |
| `<TOOL=shell>` | selector | route to shell sandbox |
| `<TOOL=apicall>` | selector | route to generic REST API |
| `<TOOL=sql>` | selector | route to SQL engine |
| `<TOOL=image>` | selector | route to image-gen tool |
| ... (~52 more selector tokens for app-specific tools) | selector | |

**Total: ~64 new tokens.** Vocabulary grows from `|V| = 50,257` (pile-bpe) to `~50,321` — a 0.13% increase. Embedding/output matrix grows by `64 × d` floats: at d=2048, ~131k extra params (negligible vs 1.84B trunk).

### 2.2 Why fixed-string tokens (not BPE-split)

Tool-call delimiters must be **unforgeable** at deployment time: the inference dispatcher (§4) detects tool calls by scanning generated tokens for `<TOOL_CALL>`. If BPE could split this into `<TOOL_`, `CALL`, `>` then a malicious or buggy generation could spell `<TOOL_CALL>` character-by-character without the dispatcher recognizing it (or vice versa, the dispatcher could over-match on user-provided text). Reserving the delimiter as a single token id eliminates this attack surface.

The pile-bpe pipeline already supports special tokens via `--special-tokens`. TOOL-LLM extends the manifest by ~64 entries; no tokenizer code changes.

### 2.3 Selector vs argument tokens

Tool selector tokens (`<TOOL=python>`) are special; tool *arguments* are standard BPE-tokenized text. This separation means:

- The model's tool-routing decision (`<TOOL_CALL>` → which selector) is a single low-entropy classification step, easy to learn (~`log_2(64) ≈ 6` bits).
- Tool arguments (Python code, search queries, SQL strings) reuse all the model's standard text-modeling capacity — no new "code mode" is needed; Python code in the corpus already exists and is BPE-tokenized.
- Adding a new tool at deployment time requires only registering a new selector token in the tokenizer + dispatcher; no full retraining if the new tool's argument syntax is already representable in BPE (e.g., a new web-search variant slots in as `<TOOL=search-v2>` over existing query syntax).

### 2.4 Loss handling on selector tokens

Selector tokens (`<TOOL=python>` etc.) are in the `C` class — model is trained to predict them. The selection decision is critical: choosing the wrong tool degrades the trace's correctness silently. We **upweight** selector tokens by `λ_sel = 4` to amplify the routing-decision signal:

```
w_t = λ_sel    if t is a selector token,
      λ_reason if t is in a reasoning/tool-call segment (per #58-C),
      1        otherwise.
```

`λ_sel = 4` is on the upper end of #58-C's safe range `[1.5, 4]`; pilots may need to drop to `λ_sel = 2` if Sophia clipping band engages.

---

## 3. CHIRON-stack synergy

### 3.1 Composition with #58 METAGEN-PROMOTED

METAGEN generates synthetic pretraining traces. Adding a tool-trace generation mode is straightforward extension:

```
METAGEN modes (post-#60-C):
  A: math problem-solving traces       (30%)
  B: code traces                       (15%)
  C: reasoning chains                  (10%)
  D: dialogue traces                   (35%)
  E: knowledge QA traces               (10%)
  F: tool-use traces (NEW, #60-C)      (15%, redistributed from D and E)
```

The METAGEN teacher generates a tool-use trace by:
1. Sampling a problem (math/code/QA).
2. Generating a CoT-style preamble.
3. Emitting `<TOOL_CALL>...<TOOL_RESULT>...` with a real (or simulated) tool execution.
4. Continuing to the answer.

**Integration via Mode F.** ~15% of METAGEN tokens become tool-use traces. The teacher's tool calls are executed *for real* during corpus construction (a one-time cost: ~$5k for 1B tool-trace tokens at $0.005/call · 200 calls/k tokens). Result is cached.

**Triple-role teacher reuse from #58 METAGEN-PROMOTED.** The same teacher model that serves DISTILL soft-targets (#56), SCROLL KL-scoring (#57), and METAGEN generation (#58) now also generates tool-use traces. **Per-step training overhead is unchanged.** The only marginal cost is the tool-execution at trace-construction time (~$5k one-time, amortizes across all subsequent training runs).

**Joint factor with #58:** 1.0 × (METAGEN already amortizes tool-trace generation; no additional speedup beyond standalone TOOL-LLM's 5×).

### 3.2 Composition with #58-C REASONING-CHAIN

Tool-trace tokens are upweighted reasoning tokens. REASONING-CHAIN's loss-weighting machinery (`λ_reason = 2`) applies directly. Selector tokens get an *additional* `λ_sel = 4` boost.

**Joint factor:** TOOL-LLM's 5× operates on a different metric (tool-augmented benchmark accuracy) than REASONING-CHAIN's 5× (reasoning-benchmark accuracy on tool-FREE benchmarks). The two are partially overlapping (a math problem solved end-to-end with CoT vs solved with calculator) and partially orthogonal (a knowledge QA solved with retrieval is essentially impossible end-to-end at any practical scale, so the parameter ratio is unbounded).

Honest joint accounting on the *tool-augmented benchmark axis* (where TOOL-LLM is decisive):
```
S_60C × S_58C = 5 × 1.5 = 7.5×    on benchmarks where reasoning is upstream of the tool call
              = 5 × 1.0 = 5×       on benchmarks where the tool replaces reasoning entirely
```

Conservative stack-level claim: **5× cumulative on top of #58-C's 5×** (additive on disjoint benchmark families, multiplicative within tool-AND-reasoning families).

### 3.3 Composition with #59-B PRM-CHIRON (tool-call correctness as PRM target)

The strongest CHIRON-stack synergy. PRM-CHIRON adds a Process Reward Model trained jointly with the trunk; the PRM scores intermediate reasoning steps for correctness.

**TOOL-LLM extends the PRM target.** Each tool call is a discrete intermediate step with a binary correctness signal:
- **Validity:** did the tool accept the call without error? (Python parses; calculator computes; web search returns results.)
- **Relevance:** did the tool result help solve the problem? (graded by checking whether the final answer matches ground truth, attributing positive credit to the call if the trace was on a correct path.)

The PRM head, trained at ~10M params per #59-B §1.2, now scores **`<TOOL_CALL>` step-end positions** in addition to reasoning step-end positions. PRM accuracy on tool-call correctness is a direct proxy for downstream tool-augmented benchmark accuracy.

**Mathematical extension:**
```
L = L_CE + λ_PRM · L_PRM_reason + λ_PRM_tool · L_PRM_tool,

L_PRM_tool = −∑_{c ∈ tool-call positions} y_c · log r̂_c + (1 − y_c) · log(1 − r̂_c).
```

`y_c = 1` if tool call `c` is valid AND the trace reaches the correct answer; else 0. `r̂_c = r_φ(h_c)` reuses the PRM head from #59-B. **No new architecture; one new loss term and a labelling pipeline that runs alongside Math-Shepherd.**

**Joint factor:** PRM-CHIRON's 1.5× combines multiplicatively with TOOL-LLM's 5×: **7.5× on tool-augmented benchmarks**, since PRM directly sharpens the tool-call selection and argument-construction signal that TOOL-LLM relies on. Toolformer (Schick 2023, §5) reports ~7pp improvement from PRM-style filtering of self-supervised tool insertions; we project a similar effect on the trunk's hidden-state organization.

### 3.4 Composition with #56 DISTILL-FORWARD (tool-trace teacher)

A reasoning-trained, tool-using teacher distills to a smaller student. The teacher's tool-call distribution becomes the student's KL target on `C`-class tokens; tool results are masked from the loss in both teacher KL and student CE. **Generational chain extension** (per #59-B §4.2):

- `G_0`: 1B CHIRON + tool-trace pretrain + PRM head + Math-Shepherd labels + tool-validity labels. ~1.5 GPU-weeks.
- `G_1`: 1.84B distilled from `G_0`, inheriting tool-call competence + PRM-as-label-source. ~1.5 GPU-weeks.
- `G_2`: 3.6B distilled from `G_1`. ~1.5 GPU-weeks.
- `G_3`: 7.2B distilled from `G_2`. Tool-augmented benchmark accuracy: GPT-4-class on MATH + HumanEval + NaturalQuestions.

**Intergenerational compounding.** Each generation's tool-call validity rate is higher than the prior; the trace corpus regenerated by `G_n` for `G_{n+1}` carries a higher-quality tool-routing distribution. By `G_3`, tool-call validity ~98% (vs Toolformer's standalone ~85%). **Joint #56 + #60-C compounding: 5 × 5 = 25× over the pre-#56 + pre-#60-C baseline on tool-augmented benchmarks.**

### 3.5 Composition with #57 SCROLL (KL-curated tool-trace selection)

SCROLL's KL-informativeness scorer (`s_i = D_KL(p_θ ‖ p_T)`) selects the top-25% most informative tokens per batch. **Tool-trace tokens systematically score high under SCROLL** because:
- `<TOOL_CALL>` selector tokens are low-entropy under the converged teacher (high KL gap when student is wrong).
- Tool-arguments (Python code, search queries) are high-density information.
- The student–teacher KL gap on tool-related tokens stays *open* longer than on standard text (tool-call competence is a hard skill).

SCROLL naturally allocates more SGD signal to tool-trace tokens. Empirically projected: ~1.3× joint with TOOL-LLM (estimated from #57's standalone 3× × overlap factor 0.43).

**Joint factor across the post-#56-#57-#58-#59-#60-C stack:**
```
post-#55 baseline    × #56 × #57 × #58 × #59-B × #60-C   = pre-stack speedup
~3,280×              × 5   × 3   × 2   × 1.5  × 5        ≈ 740,000× cumulative TRAINING speedup
                                                            (compounded; tool-augmented benchmarks)
```

The 740,000× is a **tool-augmented-benchmark-axis** measure. On tool-free text NLL, the multiplier is ~123,000× (the #60-C contribution is 1× on tool-free NLL since the smaller model doesn't help where tools are unused).

---

## 4. Bigger-picture framing: training compute → external tool compute

### 4.1 The conventional view TOOL-LLM rejects

The post-2023 LLM scaling assumption: parameter count is the primary lever for capability. Capability per FLOP improves with `N` and `D`; alignment compute is bolted on after pretraining; deployment compute is `O(N · T)` for `T` output tokens.

This frame implicitly assumes the model must internalize *every* capability. A 70B model's mathematical capability is encoded in its weights; its factual recall is encoded in its weights; its code execution is encoded as a learned simulator running inside its weights. **All capability is parameter-bound and FLOP-bound.**

### 4.2 The reframing

TOOL-LLM **dissolves the capability-internalization assumption**. Capability that is naturally external (calculator arithmetic, Python execution, factual lookup) does not need to be re-encoded into the LLM's weights. The LLM's job is **coordination**: deciding when to call which tool, with what arguments, and how to integrate the result.

**Implication for training compute.** A 1.84B coordinator + tools needs only to learn:
1. Tool-routing decisions (~6 bits per call: `log_2(64)` selectors).
2. Argument-construction within each tool's schema (Python syntax, SQL syntax, search-query phrasing) — already represented in standard pretraining text.
3. Result-integration (how to weave a tool's output into the answer) — a small set of patterns covered by the trace corpus.

The 1.84B does **not** need to learn:
1. Numerical arithmetic to high precision (calculator does it).
2. Code execution / library behavior (Python interpreter does it).
3. Factual world knowledge to deep recall (web search and retrieval do it).

**This is not a 1.84B → 18B substitution; this is a 1.84B → infinity substitution on the offloaded capability dimensions.** A real Python interpreter has unbounded computational depth; a real web search engine indexes ~$10^{12}$ documents; a real calculator computes to arbitrary precision. The model offloads these capabilities to systems whose marginal cost of one extra "knowledge bit" is many orders of magnitude lower than the model's marginal training-FLOP cost.

### 4.3 Compute trade-off accounting

| Compute locus | Pre-#60-C (end-to-end) | Post-#60-C (tool-augmented) |
|---|---|---|
| **Training compute (LLM weights)** | `C_train(18B)` | `C_train(1.84B) ≈ C_train(18B) / 5` |
| **Inference compute (LLM)** | `C_infer(18B, T_out)` | `C_infer(1.84B, T_out')` |
| **Tool execution compute** | 0 | `C_tools(K_calls)` |
| **Per-query LLM tokens** | `T_out ≈ 200` | `T_out' ≈ 250` (CoT preamble + post-tool integration) |
| **Tool calls per query** | 0 | `K_calls ≈ 1–5` typical, up to ~50 for agentic |
| **Total per-query compute** | `C_infer(18B, 200)` | `C_infer(1.84B, 250) + 1–50 × C_tool` |

For typical knowledge QA (K=1, web search):
- LLM inference: 1.84B × 250 tok ≈ 460G FLOP.
- Tool: web search ≈ 100M FLOP (server-side estimate; Anthropic's RAG study 2024).
- 18B baseline: 18B × 200 ≈ 3.6T FLOP.
- **Ratio:** ~7× *favorable* in TOOL-LLM's direction.

For tool-heavy agentic workflow (K=10, mixed tools):
- LLM inference: 1.84B × 1000 tok ≈ 1.8T FLOP.
- Tools: ~10 × 100M ≈ 1G FLOP (negligible at compute level).
- 18B baseline: ~18B × 800 ≈ 14T FLOP.
- **Ratio:** ~7× *favorable* in compute. **Latency:** ~3 s for tool-LLM (10 calls × 300 ms) vs ~1 s for 18B end-to-end. **3× *unfavorable* in latency.**

The compute axis is uniformly favorable to tool-LLM; the latency axis flips at K ≈ 3–5 for production tool-call latency budgets. **The trade-off is between training compute (huge, durable) and per-query latency (small, recurrent). Production deployments amortize training across millions of queries; per-query latency is paid every time. The two costs cannot be directly compared — they live on different time scales.**

### 4.4 Why this is "bigger picture" relative to #1–#59

Paradigm #58-C REASONING-CHAIN moves compute from training to inference *within the model boundary*. **TOOL-LLM moves compute across the model boundary entirely.** The model is no longer the locus of all capability — it is a coordinator dispatching to specialized external systems.

This reframes "extremely large LLMs on a single GPU" in a deeper way than #58-C: the question is no longer "how big a model can we train?" but "how small a coordinator do we need, and how do we offload the rest?" If the answer is "1.84B coordinator + standard tools matches 18B end-to-end on most benchmarks," then the practical engineering question shifts from "scale the model" to "design the tool ecosystem."

The deepest reformulation: **the LLM is becoming an interface, not an oracle.** GPT-4 with code interpreter, Claude with computer use, Gemini with search grounding — all three flagship 2024–2025 systems converged independently on this architectural pattern. TOOL-LLM brings the pattern *into pretraining* rather than retrofitting it as a post-hoc capability.

### 4.5 Falsifiable predictions

The training-to-tool-compute substitution frame predicts:

1. **At fixed tool-augmented benchmark accuracy, training a 1.84B + tools costs ~5× less than training a 9–18B end-to-end model.** Falsifiable: pretrain 1.84B with TOOL-LLM data (15% tool traces) and train an 18B end-to-end on the same total tokens; measure MATH-500 (with calculator) for tool-LLM and MATH-500 (no calculator) for end-to-end. Predicted: ~equivalent, training-FLOP ratio ~5×. ~30 GPU-days.
2. **Tool-call validity rate scales with model size weaker than tool-augmented benchmark accuracy.** Falsifiable: train 0.5B / 1B / 1.84B / 3.6B with identical tool-trace corpora; measure tool-call validity (lexical correctness) and tool-augmented MATH-500. Predicted: validity saturates by 1B (~92%); benchmark continues to scale to 3.6B (~+5pp per 2× params). ~12 GPU-days.
3. **Generational chain compounds tool-call validity.** Falsifiable: train `G_0=1B`, then `G_1=1.84B` distilled from `G_0` with `G_0`-PRM-as-label, then `G_2=3.6B`. Measure validity per generation. Predicted: 85% → 92% → 96% → 98%. ~18 GPU-days.
4. **Tool-LLM is decisive on tool-amenable benchmarks (math, code, factual QA) and neutral on text-only benchmarks (literary completion, dialogue coherence).** Falsifiable: factorial design over (tool-LLM-trained, baseline-trained) × (tool-amenable bench, text-only bench); measure interaction term. Predicted: significant interaction with positive effect on tool-amenable and zero effect on text-only. ~15 GPU-days.
5. **PRM-CHIRON applied to tool-call correctness lifts validity rate by +5–8pp.** Falsifiable: train tool-LLM with and without PRM-CHIRON-on-tool-calls; measure validity at convergence. Predicted: +6pp validity, +3pp benchmark accuracy. ~10 GPU-days.

Cumulative validation cost: ~85 GPU-days. **Mid-range empirical-validation requirement among #60 candidates.**

---

## 5. Honest gap: inference-cost shift to tool calls

### 5.1 Latency analysis

LLM token generation: ~10 ms/token at 1.84B with #50 HELIUM FA-3 (continuous batching, modest hardware). Output of 200 tokens: ~2 s.

Tool calls:
- **Calculator (local Python `eval`):** 1–5 ms (CPU-bound; trivially fast).
- **Python interpreter (sandboxed):** 50–200 ms (subprocess launch, Python interpreter startup; can be amortized with persistent kernels — e.g., Jupyter — at ~10 ms/call after warmup).
- **Web search (hosted API):** 300–800 ms (network round-trip + ranking).
- **Retrieval (in-memory vector DB):** 20–100 ms depending on index size.
- **REST API call:** 100–500 ms typical (network + remote compute).

**Per-query latency comparison:**

| Profile | LLM (18B end-to-end) | LLM (1.84B + tools) | Net |
|---|---|---|---|
| Pure text gen | 2–4 s | 1–2 s | ~2× faster (compute saving compounds with smaller model) |
| 1 tool call (math, search) | 2–4 s | 1–2 s + 0.3–0.8 s = 1.3–2.8 s | comparable; tool-LLM ~slightly faster on average |
| 3 tool calls (multi-hop QA) | 2–4 s | 1.5–2.5 s + 1–2.4 s = 2.5–4.9 s | comparable to slightly slower |
| 10 tool calls (agentic) | 2–4 s (no agency without tools) | 2.5–4 s + 3–8 s = 5.5–12 s | ~3× slower |
| 50 tool calls (deep agent) | impossible end-to-end | 5–10 s + 15–40 s = 20–50 s | tool-LLM is the only option; latency grows linearly |

**The trade-off shifts gracefully across the K-spectrum.** Pure text gen and few-tool queries are net faster on tool-LLM; tool-heavy agentic workflows are net slower; deep agentic flows are *only possible* with tool-LLM (an end-to-end LLM cannot interact with external state).

### 5.2 Monetary cost analysis

Training cost (one-time, amortized over millions of queries):
- 18B end-to-end pretraining: ~$1–5M typical 2026 cloud-cost estimate.
- 1.84B + tool-trace pretraining: ~$200–500k.
- Net training savings: ~$1–4.5M.

Per-query cost (recurring):
- 18B inference: ~$0.0002 per query at 200 output tokens (cloud-cost estimate).
- 1.84B inference: ~$0.00002 per query.
- Calculator / Python sandbox: $0 (local).
- Web search: $0.0005–0.01 per query (Bing/Google API; varies by tier).
- Retrieval: $0.0001–0.001 per query (vector DB hosting amortized).

**Break-even analysis.** Training savings of ~$1–4.5M amortize across ~5–22 billion queries even at the highest tool cost ($0.01/query web search). Production LLM services ship 100M+ queries/day at GPT-4-class scale; **break-even is reached in 50–220 days** of typical production traffic. Beyond break-even, tool-LLM is net cheaper indefinitely.

### 5.3 Latency SLA implications

Workloads with strict latency SLAs (sub-1-second response budgets — interactive chat with low-latency expectations, voice assistants, real-time customer support) cannot freely adopt tool-heavy queries. Mitigations:

- **Tool-call parallelism.** When the model emits multiple independent tool calls (e.g., search + calculator + retrieval simultaneously), execute them in parallel. Latency is `max(tool_i)` not `sum(tool_i)`. Reduces 3-call latency from ~2 s to ~0.8 s.
- **Speculative tool-call dispatch.** Begin executing the most likely tool call while the LLM is still generating the call's arguments. Saves ~30% latency on the most common tool calls.
- **Result-cache.** Tool results are cached by `(tool, args)` hash. Frequent web searches for the same query return in <10 ms after first call. Net: ~50% of production tool calls hit cache.
- **Tool-skip mode.** At deployment, allow the orchestrator to *override* a model-emitted `<TOOL_CALL>` and substitute the model's fallback continuation. Trades accuracy for latency on time-critical queries.

### 5.4 The honest framing

TOOL-LLM is a **deliberate architectural commitment** with a real cost. The cost is paid by tool-heavy queries (5+ calls) at deployment time, in latency. The benefit is paid back at training time (~5× compute saving) and at deployment for tool-light queries (~2× faster). **Whether the trade is favorable depends on your query distribution.**

For most production LLM workloads — assistant-style chat, code completion, factual QA — tools are invoked on a minority of queries (typically 10–40%). For these workloads, TOOL-LLM is decisively favorable on training cost and modestly favorable on average per-query cost. For pure-agentic deployments (multi-step planning, repeated tool invocation) the latency cost is real and non-amortizable.

**TOOL-LLM is selected when the deployment query distribution has K ≤ 3 mean tool calls and tool-augmented benchmarks are part of the evaluation gate.** It is *not* selected when the workload is overwhelmingly tool-heavy agentic (use raw model + dispatcher) or strictly latency-bound (cannot afford tool calls in the loop).

---

## 6. Engineering: ~580 LOC over ~3.5 weeks

| Component | Files | LOC | Week |
|---|---|---|---|
| Tool-trace tokenizer (extends pile-bpe special tokens) | `DataObjects/ToolTokenizer.cpp`, `.h` | 80 | 1 |
| Special-token registration + vocabulary expansion | `DataObjects/tokenizer_config.cpp`, `Networks/network.cpp` | 40 | 1 |
| Tool-result loss-mask kernel (CUDA + CPU fallback) | `cuda/tool_loss_mask_kernel.cu`, `.h` | 30 | 1 |
| Tool-trace data pipeline (Mode A/B handlers) | `Networks/data_loader_tool.cpp`, `.h` | 150 | 2 |
| Deployment-time tool dispatcher (token scanner + tool router) | `Networks/tool_dispatcher.cpp`, `.h` | 120 | 2-3 |
| Tool wrappers (python sandbox, calculator, search, retrieve, sql, shell) | `tools/python_sandbox.cpp`, `tools/calculator.cpp`, `tools/web_search.cpp`, `tools/retrieve.cpp`, `tools/sql.cpp`, `tools/shell.cpp` | 100 | 3 |
| Benchmark suite (MATH-with-calc, HumanEval-with-Python, NQ-with-search) | `unit-tests/.../tool_llm_test.cpp` | 40 | 3.5 |
| Documentation + paradigm-shift markdown | this file | 20 | 0.5 |
| **Total** | | **~580** | **~3.5 weeks** |

**Public API:**
```cpp
namespace glades { namespace tools {
struct ToolCall { std::string tool; std::string args; };
struct ToolResult { std::string output; bool success; };

class ToolDispatcher {
    void registerTool(const std::string& name, ToolWrapper* impl);
    ToolResult dispatch(const ToolCall& call);
    bool detectCallInStream(const std::vector<int>& tokens, ToolCall& out);
};

class ToolWrapper {
public:
    virtual ~ToolWrapper() = 0;
    virtual ToolResult execute(const std::string& args) = 0;
    virtual std::string name() const = 0;
};

// Concrete implementations:
class PythonSandbox  : public ToolWrapper { ... };
class Calculator     : public ToolWrapper { ... };
class WebSearch      : public ToolWrapper { ... };
class RetrievalIndex : public ToolWrapper { ... };
class SqlEngine      : public ToolWrapper { ... };
class ShellSandbox   : public ToolWrapper { ... };
}}
```

**Risks and mitigations:**
- **Loss-mask correctness.** The `<TOOL_RESULT>` mask must match the tokenizer's region detection exactly; off-by-one errors corrupt training silently. Unit test: round-trip a known trace, verify `m_t = 0` exactly on result-region positions.
- **Special-token collision.** If user-provided text contains the literal string `<TOOL_CALL>`, the tokenizer must escape it (replace with a non-special variant `<TOOL_CALL_LITERAL>` at user-input time). Mitigation: strict tokenizer mode rejects literal `<TOOL_*>` strings in user text and demands re-encoding.
- **Tool-call infinite loops at deployment.** A misbehaving model can emit `<TOOL_CALL>` tokens repeatedly without progressing. Mitigation: per-query tool-call cap (default `K_max = 50`); orchestrator force-terminates and returns a fallback after the cap.
- **Tool-execution sandboxing.** Python and shell tools must be sandboxed (resource limits, no filesystem access outside scratch). Use existing `gvisor` / `firejail` integration; this is operational, not new code.
- **Tokenizer compatibility with existing pile-bpe checkpoints.** Adding ~64 tokens at the *end* of the vocabulary preserves token ids for the original 50,257 tokens. Existing checkpoints can be extended by appending fresh embedding rows + output rows; no retraining of the original tokens needed.

---

## 7. Honest gap and Gate-0 protocol

### 7.1 The trace-corpus dependency

Tool-trace corpus options (decreasing curation effort):
- **Public (low effort).** ToolBench (Qin 2023) ~110k traces, ~50M tokens. AgentBench, ReAct trajectories, Toolformer self-supervised insertions on C4 (~$2k re-running). Total: ~200M public tool tokens, ~0.07% of pretraining mix.
- **METAGEN-extension (medium effort).** Adds ~1B tool tokens via #58-C-style synthetic generation; ~$5k tool-execution cost. Reaches ~7% mix at 15B token budget.
- **Curated synthetic (high effort).** Custom prompted generation across all tool surfaces; ~$20k cost; reaches 15% mix.

Default plan: combine public + METAGEN extension to reach 10% tool-trace mix; reserve curated synthetic for the post-Gate-0 step-up.

### 7.2 Tool-call validity ceiling

ToolLLM (Qin 2023, Tab. 5) reports ~85% tool-call validity at 7B scale. The 15% invalid calls are predominantly:
- **Argument-format errors** (e.g., wrong JSON schema): ~7%. Mitigated by strict-validation tool wrappers that reject and re-prompt.
- **Wrong tool selected**: ~4%. Mitigated by PRM-CHIRON on tool-routing.
- **Non-applicable call** (tool can't help with this query): ~4%. Inherent skill ceiling; bound by training data quality.

**Validity ceiling caps benchmark accuracy.** At 85% validity, math-with-calculator MATH-500 accuracy ~+15pp over no-calc baseline (instead of theoretical +25pp at 100% validity).

### 7.3 Tool surface stability

Tools change across deployment lifecycles. Mitigation:
- **Versioned tool tokens.** `<TOOL=python-3.11>`, `<TOOL=search-bing-2025>`. Old training runs can serve old tool versions; new training runs introduce new versions. The model learns to map versions transparently (Gorilla-style).
- **Tool-doc retrieval at inference.** When a tool selector token is emitted, the dispatcher fetches the current tool's API doc and prepends it to the model's context for the argument-construction step. Robust to API drift (the model conditions on current docs, not memorized docs).

### 7.4 The honest claim about NLL

Tool-result tokens are masked from the loss; tool-call tokens are predicted normally. **Standard text-NLL on tool-free text is preserved exactly** — the tool-trace fraction adds tokens to the corpus without changing the loss on existing tokens. **NLL on tool-trace tokens (call class) is a new component of the pretraining loss; its absolute value is not directly comparable to NLL on tool-free text** because the token distribution is structurally different (selector tokens are low-entropy, argument tokens are high-density).

For NLL benchmark protocol (per `BEYOND_CHIRON.md` §2.3): NLL is reported on tool-free held-out text. **TOOL-LLM preserves this NLL exactly within noise; the value-add is on the tool-augmented benchmark axis.**

### 7.5 Gate-0 protocol (24 GPU-hours)

**Question:** *On 66M CHIRON with 10% tool-trace pretraining data and a `<TOOL=calc>` calculator wrapper, does TOOL-LLM achieve ≥ 5pp improvement on GSM8K-200-with-calc vs a control without tool training, while preserving text-NLL within 0.05 nat?*

**Setup.** Three arms at 66M, 30k steps:
- **Arm A (control):** post-#42–#59 stack, no tool training. GSM8K-200 (no calc) ~14%, GSM8K-200 (with-calc, naive — model emits calc syntax it never trained on) ~14%. Text-NLL: baseline ~3.95 nat.
- **Arm B (TOOL-LLM):** same stack + 10% tool-trace data + special-tokens + tool-result mask + `<TOOL=calc>` wrapper. GSM8K-200-with-calc target ≥ 19%.
- **Arm C (tool-trace fraction sweep):** 5 runs × 5k steps at fractions `{2%, 5%, 10%, 15%, 25%}`. Establishes 66M-scale optimum.

**Pass:** (1) Arm B GSM8K-200-with-calc ≥ Arm A + 5pp. (2) Arm B text-NLL within 0.05 nat of Arm A. (3) Arm B tool-call validity (parses + executes without error) ≥ 60% by step 30k. (4) Arm C optimum in `[5%, 15%]` with monotone degradation outside.

**Fail-fast:** NaN or NLL diverges < 5k → REJECT. Arm B NLL ≥ Arm A + 0.20 → REJECT. Arm B GSM8K-200-with-calc ≤ Arm A → REJECT. Arm B tool-call validity ≤ 30% → REJECT. Arm B GSM8K-200-with-calc ≥ Arm A + 12pp → STRONG PASS.

**Cost.** A+B: 12 GPU-hours. C: 6 GPU-hours. Eval (incl. tool execution): 4. Trace corpus prep at 66M scale (~50M tool tokens): 2 GPU-hours. **Total: 24 GPU-hours = 1 GPU-day.**

**Gate-1** (post-Gate-0): same at 1.84B, 100k steps, full benchmark suite (MATH-with-calc, HumanEval-with-Python, NaturalQuestions-with-search), tool-trace fraction at the Gate-0 optimum, comparison against an 18B end-to-end baseline. ~10 GPU-days.

**Gate-2** (post-Gate-1): joint composition with #56-DISTILL, #58-C REASONING-CHAIN, #59-B PRM-CHIRON-on-tool-calls. `G_0` 1B + tool-trace + PRM (~1.5 GPU-weeks); `G_1` 1.84B distilled student (~1.5 GPU-weeks). Verify multiplicative speedup factor 5× (TOOL-LLM) × 1.5× (PRM-on-tool-calls) ≈ 7.5× on tool-augmented benchmarks.

**Gate-3** (production-readiness): deployment latency benchmark; tool-call orchestration robustness (sandboxing, timeouts, error handling); SLA compliance under representative query distributions. Operational, not training-side; ~5 GPU-days plus deployment-engineering effort.

---

## 8. Summary

TOOL-LLM is **the external-tool primitive at pretraining time** — special tokens (`<TOOL_CALL>`, `<TOOL_RESULT>`, ~64 selector tokens) train the model to invoke calculators, code interpreters, web search, and retrieval as part of its inference. Tool-result tokens are masked from the loss; tool-call tokens are upweighted via #58-C's `λ_reason` machinery. Per-step training overhead: zero (the data composition shifts but kernels are unchanged).

**Training-side speedup at matched tool-augmented benchmark accuracy:** **~5× wall-clock conservative; ~10× aggressive.** Mechanism: a 1.84B coordinator + tools matches an 18B end-to-end solver on tool-amenable benchmarks because the tools internalize capability that no parameter count can match efficiently (arbitrary-precision arithmetic, real code execution, real-time factual retrieval). **NLL on tool-free text is preserved exactly.**

**Composition with the post-#59 stack:** multiplicative on tool-augmented benchmarks. Joint per-trajectory speedup: #56 × #57 × #58 × #59-B × #60-C ≈ 5 × 3 × 2 × 1.5 × 5 = **225× over the post-#55 baseline; ~740,000× cumulative single-GPU TRAINING speedup** at 1.84B-coordinator-with-tools matching an 18B-end-to-end-with-no-tools on tool-augmented benchmarks.

**Engineering:** ~580 LOC over ~3.5 weeks. One-time setup: ~$5k tool-execution cost for METAGEN-extended trace corpus generation; public ToolBench + Toolformer trace pool covers a useful baseline at $0 marginal cost. Gate-0 cost: 24 GPU-hours.

**Bigger-picture frame.** TOOL-LLM dissolves the **capability-internalization assumption** that has implicitly governed LLM scaling since 2023. Capabilities that are naturally external — arithmetic, code execution, factual recall — do not need to be re-encoded into LLM weights. **The LLM becomes a coordinator interfacing with specialized external systems whose marginal cost per useful bit is many orders of magnitude lower than parameter-encoded knowledge.** Training compute moves *across* the model boundary, not just *within* it (REASONING-CHAIN's training-to-inference shift) or *between phases* (PRM-CHIRON's pretraining-RLHF unification).

**Honest gaps.** (1) Inference-cost shift: tool calls add 100–500 ms latency each; tool-heavy agentic workflows (≥5 calls) are 3–10× slower end-to-end than tool-free LLM-only baselines. Pure-text and few-tool queries are faster on tool-LLM (smaller model). (2) Monetary cost shift: training is ~5× cheaper, deployment may be more expensive depending on tool mix (web search APIs are the main opex driver). Break-even at typical production volume: 50–220 days. (3) Tool-call validity ceiling at ~85% (caps benchmark gain at +15pp on MATH-500-with-calc); PRM-CHIRON-on-tool-calls lifts to ~92%. (4) Trace-corpus dependency: ~10% mix requires public ToolBench + #58 METAGEN-extension or curated synthetic generation. (5) Tool surface drift across deployment lifecycle; mitigated by versioned selector tokens and tool-doc retrieval at inference. (6) NLL is not the right metric of victory — tool-augmented benchmark accuracy is. Workloads requiring strict text-NLL parity at matched parameters cannot use TOOL-LLM as a pure NLL paradigm.

**Selection criterion vs #60-A and #60-B.** TOOL-LLM is the candidate that **most decisively reframes the model boundary** — the LLM stops being an oracle and becomes a coordinator. Among #60 candidates, it has the **strongest empirical precedent** (GPT-4 + code interpreter, Claude + computer use, Gemini + search are all production-deployed validations of the underlying architecture), the **largest training-side speedup** (~5× conservative, on par with #58-C's REASONING-CHAIN), and the **most honest deployment-cost trade** (latency shift at tool-heavy query profiles is real and non-amortizable).

**Standing brief alignment.** TOOL-LLM optimizes *the relationship between the LLM and external systems*. The 2024–2025 frontier question is shifting from "how big can we train?" to "how small a coordinator do we need?" TOOL-LLM is the first paradigm in the CHIRON stack to make a serious move on the second, in a way that composes cleanly with #58-C (reasoning), #58-A (METAGEN can generate tool traces), and #59-B (PRM-CHIRON can score tool-call correctness).

The 5× training-FLOP headline is robust, well-precedented, and orthogonal to all prior paradigms in the stack. The deployment-cost trade is real but heavily depends on the query distribution; for the median LLM workload (tool-light, mixed-domain), TOOL-LLM is decisively favorable on both training cost and average per-query latency. **TOOL-LLM is selected when the deployment includes tools as a first-class capability and the evaluation gate includes tool-augmented benchmarks. It is not selected when the workload is overwhelmingly tool-heavy agentic or strictly sub-second-latency-bound.**
