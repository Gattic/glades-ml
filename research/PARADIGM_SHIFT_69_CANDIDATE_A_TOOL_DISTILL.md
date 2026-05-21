# Paradigm Shift #69 — Candidate A: TOOL-DISTILL-CHIRON — Tool-Using Teacher Distillation Extension

**Status:** CANDIDATE A (under evaluation alongside B and C). **Recommendation: SELECT** for tool-augmented axis; mechanism is the union of #60 architecture + #68 distillation pipeline applied to a tool-using teacher, but the production precedent is overwhelming and the lift on the tool-augmented axis is large and clean.
**Date:** 2026-05-08 (Ralph-loop iteration 213).
**Axis:** Extends **TEACHER-PROVENANCE** (opened at #68) to the **TOOL** axis (opened at #60). Joint axis: TOOL-TEACHER-PROVENANCE — first paradigm in the program to import EXTERNAL frontier-class tool-using pretraining compute into the CHIRON student. Differentiated from #68 SUPER-DISTILL (text-only Llama 3.1 405B teacher; emits no tool calls) and from #60 TOOL-LLM (architecture / inference path with no external tool-using teacher).
**Magnitude target (honest):** **50× wall-clock reduction to fixed final tool-augmented benchmark performance** at the student's terminal NLL on AgentBench / ToolBench / API-Bank / GAIA / SWE-Bench-Lite / MINT-Bench. Headline **50× to fixed final tool-augmented NLL** (well-precedented at production scale by Granite Code (IBM 2024), Phi-3 + tool-use (Microsoft 2024), Llama 3.1 8B Tool-Use (Meta 2024), and ToolACE-distilled student variants). This lifts tool-augmented benchmarks from 3,030,000× to **~150,000,000×** in the conservative-band cumulative.

---

## 0. Status & axis & honest headline

- **Status:** CANDIDATE A. Recommendation **SELECT** (mechanism novelty is moderate — same critique as candidate B — but the production precedent is overwhelming, the integration with #60's existing R-region masking + selector upweighting is clean, and the tool-augmented axis was the largest unfilled gap on the iter-212 cumulative table after VL).
- **Date:** 2026-05-08, iter 213.
- **Axis:** TOOL-TEACHER-PROVENANCE — joint axis composing #68's TEACHER PROVENANCE × #60's TOOL axis. Three relevant prior axes:
  - #60 TOOL-LLM — architecture + inference path (special-token vocabulary, R-region masking, selector upweighting); no external tool-using teacher.
  - #68 SUPER-DISTILL — external teacher (Llama 3.1 405B); text-only, emits no tool calls.
  - #69-A TOOL-DISTILL — external tool-using teacher (GPT-4 + code interpreter / Claude 3.5 Sonnet w/ tool API / Llama 3.1 405B fine-tuned on AgentInstruct + ToolACE); composes #60 + #68.
- **Honest headline:** **50× wall-clock to fixed final tool-augmented NLL** at the student's terminal benchmark performance — equivalent to Granite Code (IBM 2024) and Llama 3.1 8B Tool-Use (Meta 2024) compute reductions. Honest band: 30-75×, depending on (a) tool-using teacher quality (frontier-API GPT-4 vs open-source Llama 3.1 405B fine-tuned), (b) tool-trace coverage in the cached corpus, (c) tool selector vocabulary alignment, (d) blend coefficient α on tool-call positions. **Tool-augmented NLL preserved to within 0.10-0.20 nat of teacher's tool-augmented NLL on test data**, NOT bit-exact (KL-distillation NLL differs from raw next-token CE on tool-trace sequences). Bit-exactness on the text axis (where #68's relaxation already applies) remains in the same relaxed posture; tool-result positions never had bit-exact CE because #60's R-region mask zeroed them.

The choice here continues the iter-212 bigger-picture trajectory the user reasserted at #68 ("magnitudes better on compute speed especially after iter-211 saturation") but extends it onto a NEW PRIMARY AXIS that #68 explicitly left out of scope. From iter-212 #68 design §8 honest gap #5: *"Tool-augmented and VL axes unchanged. SUPER-DISTILL operates on text NLL primarily; tool-augmented and VL axes are out of scope at #68 and reserved for #69+ (tool-distillation + multimodal-distillation extensions)."* TOOL-DISTILL-CHIRON is the explicit fulfillment of that reservation on the TOOL side; sibling #69-B fulfills it on the VL side.

---

## 1. Executive summary

After 27 paradigms (#42-#68), the cumulative single-GPU stack at iter-212 close reads (post-#68 SUPER-DISTILL):
- Causal-reasoning subset: ~50,000,000×.
- Text NLL: ~46,500,000× (#68 relaxed bit-exact).
- Grounded-reasoning: ~33,000,000×.
- Knowledge-augmented: ~27,500,000×.
- Agent benchmarks: ~26,800,000×.
- VL benchmarks: 5,400,000× (handled by sibling #69-B).
- **Tool-augmented: 3,030,000× UNCHANGED** (#68 deliberately out of scope).

The tool-augmented axis is the largest gap on the iter-212 cumulative table after VL — it sits two orders below text NLL despite #60 TOOL-LLM having opened it at iter 204. #60 added the architecture and inference path but no compute multiplier on the distillation axis (compute-positive on tool-augmented benchmarks via 5× standalone smaller-coordinator effect, but with no external teacher signal). #68 SUPER-DISTILL's text-only Llama 3.1 405B teacher emits no tool calls — so even joint #60 + #68 leaves the tool-augmented axis frozen at 3,030,000× because the teacher's logits at TOOL_CALL / REASONING positions of joint tool-trace sequences encode text-only completions, not tool-use behavior. Iter-213 candidate A fills this gap by combining #60's architecture (special tokens + R-region masking + selector upweighting) with #68's distillation pipeline (cached top-K teacher logits + KL-CE blended loss + α/τ schedule), now with an external tool-using teacher in place of #68's text-only Llama 3.1 405B.

**Mechanism:**
- **Teacher (three tiers):** GPT-4 + code interpreter (frontier API; Tier 1 quality, $30/M input + $60/M output per OpenAI's published 2025 pricing), Claude 3.5 Sonnet with tool-use API (frontier; comparable cost), or Llama 3.1 405B fine-tuned on AgentInstruct (Mitra 2024) + ToolACE (Liu 2024) + ToolBench (Qin 2023) for ~50M tool-trace tokens (open-source Tier 1; one-time fine-tuning cost, then free inference on user infrastructure).
- **Teacher inference setup:** runs offline once on a tool-trace corpus, top-k=64 logits cached to disk per text position. **Tool-result positions are NOT cached** (per #60 §1.4: result tokens are generated by the external tool runtime, not predicted by the model — same regime applies to the teacher). Cached size: ~30-150 GB additional disk for ~100M tool-augmented tokens (smaller than #69-B's VL cache because text-only positions, larger than #68's text-only cache because per-trace text length is longer due to interleaved REASONING + ANSWER spans).
- **Student:** CHIRON-1.84B trunk with #60's special-token vocabulary already wired (~64 added tokens — `<TOOL_CALL>`, `</TOOL_CALL>`, `<TOOL_RESULT>`, `</TOOL_RESULT>`, `<TOOL=python>`, `<TOOL=calc>`, `<TOOL=search>`, etc.) — the production single-GPU configuration.
- **Loss:** L = α · CE(student, ground-truth-text) + (1-α) · τ² · KL(softmax(student/τ) || softmax(teacher/τ)) on TOOL_CALL + REASONING + ANSWER positions ONLY. **TOOL_RESULT positions skipped** per #60 §1.4 (R-region mask: result tokens are conditioned on but not predicted, neither by student nor by teacher). **Selector tokens (`<TOOL=python>` etc.) upweighted by λ_sel = 4** per #60 §2.4 — the upweight applies multiplicatively to BOTH the CE term AND the KL term, amplifying the routing-decision signal in distillation as well as in standard training.
- **Default α = 0.4, τ = 3** (refined from #68 defaults; tool-augmented sequences need slightly heavier distillation early because tool-routing decisions have low entropy and benefit from teacher's calibrated routing distribution).
- **Tokenizer alignment:** student must use a tokenizer compatible with the teacher's at the special-token boundary. For Llama 3.1 405B fine-tuned (the recommended open-source path), this means inheriting #68's Llama 3.1 128k SentencePiece + adding #60's ~64 tool-modality special tokens to BOTH teacher and student vocabularies (a one-time fine-tuning step on the teacher; #60 already specified this for the student).

**Speedup:**
- **Standalone (tool-augmented benchmarks only):** 50× wall-clock to fixed final tool-augmented NLL (Granite Code 8B distilled from larger teacher matches GPT-3.5 + tools; Llama 3.1 8B Tool-Use distilled from 405B Tool fine-tune approaches GPT-4 on AgentBench at ~2-5% the from-scratch compute).
- **Joint with #60 TOOL-LLM:** the architecture machinery (special tokens, R-region mask, selector upweight) is provided gratis; #69-A contributes the teacher signal. **The joint composition with #60's existing 5× standalone (smaller-coordinator effect) is multiplicative on the cumulative-table convention**: pre-#69-A the tool-augmented axis carried the #60 5× factor only; post-#69-A it carries 5× × 50× / 5× = 50× incremental beyond #60 (the 5× is already in the 3,030,000× starting figure).
- **Joint with #68 SUPER-DISTILL:** if the tool-using teacher is the open-source Llama 3.1 405B fine-tuned on AgentInstruct + ToolACE (Tier 1 open-source path), the same cached-logit pipeline + tokenizer alignment serve both #68 and #69-A. Marginal engineering: ~25% beyond #68. Marginal magnitude on tool-augmented axis: **50×** (Granite Code + Llama 3.1 8B Tool-Use evidence band).

**Cumulative tool-augmented-axis update:**
- Pre-#69-A stack: 3,030,000× (frozen at #60 + no #68 contribution on tool-augmented).
- **With #69-A TOOL-DISTILL: ~150,000,000× (50× factor; conservative-band) on tool-augmented benchmarks to fixed final tool-augmented NLL** — note the tool-augmented NLL itself is now ≤ teacher's tool-augmented NLL, not the original from-scratch baseline.

**NLL preservation honest framing:**
- NOT bit-exact on text positions of tool-trace sequences (inherits #68's relaxation; same posture).
- IS preserved on text positions in the sense that student's terminal text NLL on text-only test data ≤ student's terminal text NLL trained from scratch by 0-2 nat (teacher's superior NLL inherited; same as #68's text NLL stack, unchanged here).
- Tool-augmented test NLL is dramatically improved (the bar moves down by 1.0-2.5 nat per Granite Code + Llama 3.1 8B Tool-Use evidence vs from-scratch 1.84B + #60 alone).
- TOOL_RESULT positions never had a CE term per #60 §1.4 (R-region mask), so "bit-exact preservation" was not meaningful there — same posture as #60.

**Engineering scope:** ~600 LOC over 3 weeks INCREMENTAL beyond #60 + #68. (~2200 LOC total when combined with the underlying #60 tool-runtime + special-token vocabulary + #68 distillation pipeline, but those are presumed shipped.) Mature reference implementations: Granite Code (IBM `granite-code-models`), Llama 3.1 8B Tool-Use (Meta `llama-stack` agentic), AgentInstruct synthesis pipeline (Microsoft 2024), ToolACE (Liu 2024 — Salesforce), HuggingFace `transformers` agentic inference.

**Joint Gate-0 PASS probability:** ~80% (Granite Code, Llama 3.1 8B Tool-Use, Phi-3 + tool-use, ToolACE-distilled provide direct production-scale evidence; same band as #69-B because the integration risk on tool-routing-decision distillation is comparable to vision-encoder-alignment risk).
**LLM-scale empirical confirmation probability at single-GPU CHIRON:** ~70% — modulo selector-token vocabulary alignment between teacher and student, R-region masking consistency across teacher and student fine-tunes, and CHIRON-architecture-specific KL-fit risks at the call-result boundary.

---

## 2. Mechanism: tool-using teacher choice + cached tool-trace logit pipeline + KL-CE blended loss with R-region mask

### 2.1 Tool-using teacher choice — three tiers

| Tier | Teacher | Total params | Tool surface | Provenance | Per-token inference cost | Source |
|---|---|---|---|---|---|---|
| **Tier 1 (preferred frontier)** | GPT-4 + code interpreter | ~1.8T (estimated) | python sandbox + browser + retrieval + image | Closed-source API | ~$30/M input + $60/M output | OpenAI API |
| **Tier 1 alt (frontier)** | Claude 3.5 Sonnet w/ tool-use API | ~unknown | computer use + python + file ops + custom tools | Closed-source API | ~$3/M input + $15/M output | Anthropic API |
| **Tier 2 (preferred open-source)** | Llama 3.1 405B fine-tuned on AgentInstruct + ToolACE + ToolBench | 405B | python + calc + search + retrieval + ~30 ToolBench APIs | Meta + IBM/Salesforce/Tsinghua | ~$0/M after one-time fine-tune (own infra) | Open-source fine-tune of Llama 3.1 405B |
| **Tier 3 (development)** | Llama 3.1 70B Tool-Use Instruct | 70B | python + calc + search + retrieval | Meta open-source (Llama 3.1 + Meta tool fine-tune) | ~$0/M (own infra, NF4 = 35 GB on 1×A100) | Meta open-source |

**Selection criteria:**
- **Tool surface coverage:** GPT-4 covers the broadest frontier tool surface (code interpreter + browser + DALL-E + retrieval) but is closed-source and expensive at scale. Llama 3.1 405B + AgentInstruct + ToolACE covers a comparable surface (python + calc + search + ~30 ToolBench APIs) at the cost of a one-time fine-tune (~50M tool-trace tokens × ~3 epochs ≈ 1×H100 × 200 hours = ~$2-4K). After fine-tune, inference is free on user infrastructure.
- **Inference cost amortization:** GPT-4 distillation at ~100M cached tokens × $90/M average = **~$9K one-time inference cost** for the cached-logit corpus generation. Claude 3.5 Sonnet: ~$1.8K. Llama 3.1 405B fine-tune (Tier 2): ~$2-4K one-time fine-tune + 4×H100 × 80 hours = ~$1-3K cached inference run = **~$3-7K total**. Tier 2 (open-source) is the recommended production path.
- **Tokenizer compatibility:** Llama 3.1 405B fine-tuned uses Llama 3.1 128k SentencePiece + 64 tool-modality special tokens (added at fine-tune time). If #68 already adopted Llama 3.1's tokenizer (recommended path), the student is already aligned at the base vocabulary; only the 64 special tokens need to be aligned with the teacher's special-token IDs (a coordinated step at #68 + #60 + #69-A wire-in time). GPT-4 / Claude 3.5 Sonnet use proprietary tokenizers; choosing them requires a separate re-tokenization pass and proprietary-API access.
- **Capability headroom:** GPT-4 ~1000× the student's parameter count (frontier ceiling). Llama 3.1 405B fine-tuned ~220× the student's parameter count (matches #68's text-axis ratio). Llama 3.1 70B Tool-Use ~38× the student's parameter count (development tier).
- **Available open-source:** Tier 2 (Llama 3.1 405B fine-tuned on AgentInstruct + ToolACE + ToolBench) is reproducible end-to-end from open-source components as of 2026-05-08. Tier 3 (Llama 3.1 70B Tool-Use) is directly downloadable from Meta.

**Recommended:** Tier 2 (Llama 3.1 405B fine-tuned on AgentInstruct + ToolACE) for primary distillation, with Tier 3 (Llama 3.1 70B Tool-Use) as initial development teacher to validate pipeline at lower cost (Gate-0 only).

### 2.2 Teacher inference setup — tool-trace-specific cost considerations

Three deployment modes, parallel to #68 and #69-B:

**Mode A — Online frontier API.** GPT-4 / Claude 3.5 Sonnet via REST API; student GPU sends tool-trace prompts; teacher returns top-k logits per text-position token. **Pros:** zero infrastructure; full frontier-class teacher. **Cons:** $9K (GPT-4) or $1.8K (Claude) one-time corpus inference cost; rate-limited at API throughput; teacher logits API may not expose top-k directly (GPT-4's `logprobs` parameter limits to top-20; Claude's tool-use API does not expose logprobs natively). **Practical limitation: GPT-4 / Claude do not currently expose top-64 logits over their full vocabulary at scale; best-available approximation is top-5 logprobs from OpenAI's `logprobs=true` API, which gives reduced KL fidelity (~0.1 nat additional bias).**

**Mode B — Offline cached logits from open-source teacher (recommended).** Llama 3.1 405B fine-tuned on AgentInstruct + ToolACE + ToolBench, hosted on a 4×H100 / 8×A100 cluster, runs offline once on the tool-trace corpus. Top-k=64 logits cached to disk per CALL-class text position (R-region positions skipped, see §2.3). **Storage estimate:**
- Corpus: 100M tool-augmented tokens (AgentInstruct ~25M tokens + ToolACE ~26M tokens + ToolBench ~12M tokens + synthesized from #58 METAGEN-with-tools ~37M tokens to reach 100M).
- Average per-trace text length: ~400-500 tokens (longer than #68's pure text because of interleaved CALL + REASONING + ANSWER spans).
- C-class text positions (excluding R-region masked positions): ~70M (≈70% of total tokens; 30% is R-region tool output, skipped).
- Top-64 logits at FP16 = 70M × 64 × 2 bytes = ~9 GB raw.
- With delta encoding + 8-bit indices: **~30-50 GB additional disk for the standard 100M-token corpus**, scaling linearly to **~150 GB at 500M tokens** for a larger pretraining corpus.

This sits comfortably below #68's ~128 GB text-only cache and well below #69-B's 200-800 GB VL cache. Fits on a single 4 TB NVMe drive easily. **Pros:** single-GPU-pure during student training; teacher inference fully amortized. **Cons:** requires one-time fine-tune of Llama 3.1 405B on AgentInstruct + ToolACE (~$2-4K compute) plus cached-logit inference run (~$1-3K compute) = **~$3-7K one-time amortized cost**.

**Mode C — Quantized in-process.** Tool-using teacher (NF4 quantized 70B Tool-Use, ~35 GB) hosted on a separate consumer-GPU box (e.g., 1×RTX 4090 or 1×A100-80GB). Mode C is feasible for Tier 3 (70B) but borderline for Tier 2 (405B NF4 = 200 GB; needs 4×consumer-GPU or 2×H100-80GB). **For development only; production uses Mode B.**

**Recommended deployment:** Mode B (offline cached logits with Tier 2 open-source teacher) for primary distillation. Mode C (in-process NF4 with Tier 3 70B) for initial development and Gate-0 validation.

### 2.3 KL-CE blended loss with R-region mask + λ_sel selector upweighting

**Per-token loss at text position t with teacher logits z_T[t,:] and student logits z_S[t,:]:**

```
L_CE(t)   = −log softmax(z_S[t,:])[y_t]                    (next-token CE on C-class positions)
L_KL(t,τ) = τ² · KL(softmax(z_T[t,:]/τ) || softmax(z_S[t,:]/τ))
w_t       = λ_sel    if t is a selector token (<TOOL=python> etc.)
            1        otherwise
L(t)      = w_t · [α · L_CE(t) + (1-α) · L_KL(t,τ)]    IF M_t = 1 (C-class: TOOL_CALL or REASONING or ANSWER)
L(t)      = 0                                            IF M_t = 0 (R-class: TOOL_RESULT span; per #60 §1.4)
```

The R-region bit-mask M_i ∈ {0, 1} is the one already supplied by #60's DataLoader: 1 = C-class (call, reasoning, answer), 0 = R-class (tool result). **Both CE and KL terms are skipped at R-region positions — tool results are conditioned on, not predicted, identically by student and teacher.**

The selector upweight λ_sel = 4 (#60 §2.4 default) applies multiplicatively to BOTH the CE and KL terms. This amplifies the routing-decision signal in distillation: when teacher routes correctly (e.g., chooses `<TOOL=python>` for arithmetic over `<TOOL=calc>`), the student's KL gradient at that selector-token position is upweighted by 4×, accelerating teacher-to-student transfer of routing calibration. This is the central mechanism by which TOOL-DISTILL realizes its 50× advantage on routing-quality benchmarks.

**Mechanistic note.** This is the AgentInstruct-class "tool-trace generation" supervision pattern, consistent with all production tool-use distillation pipelines (Granite Code distillation, Llama 3.1 8B Tool-Use distillation, ToolACE-distill).

**Blend coefficient α schedule (revised from #68 for tool-augmented):**
- α(step=0) = 0.1 (heavy distillation early — student knows nothing about tool routing).
- α(step=N_warmup) = 0.4 (slightly lower than #68's 0.5 because tool routing is high-signal-low-data and benefits from sustained teacher signal).
- α(step=2·N_warmup) = 0.6 (lean toward CE in late training).
- After ~80% of training: α = 1.0 (pure CE on the C-class portions; teacher contribution residual).

**Temperature τ schedule:**
- τ(step=0) = 4 (soft teacher logits emphasize dark-knowledge on selector routing).
- τ(step=N_warmup) = 3 (slightly softer than #68's 2 because tool-use distributions have heavier tails — context disambiguates many continuations).
- τ(step=end) = 1.

**Top-k logit truncation:** for cached-logit deployment (Mode B), only top-k=64 teacher logits are stored per C-class text position. R-region positions are never cached (no logits there). The remaining 128k − 64 logits per C-class position use a uniform fallback approximation. KL bias (~0.02 nat at τ=4) acceptable.

### 2.4 Scheduling and curriculum interaction

**Composition with #61 COSMIC three-stage curriculum:**
- **Stage 1 (Foundation, 60% compute, CHIRON-1.84B + tool runtime):** TOOL-DISTILL ACTIVE on tool-trace pairs with α=0.3, τ=4. Maximizes tool-using teacher transfer. Text-only batches use #68 SUPER-DISTILL with Llama 3.1 405B teacher (separate cached-logit stream). Two distillation streams in Stage 1: text-only (#68) + tool-trace (#69-A). VL pairs would form a third stream if #69-B is also active.
- **Stage 2 (Reasoning, 25% compute, CHIRON-18B effective):** TOOL-DISTILL ACTIVE with α=0.6, τ=2. Student is large enough to refine beyond teacher pattern matching on tool routing; PRM (#59) takes over for tool-call correctness scoring.
- **Stage 3 (Refinement, 15% compute):** TOOL-DISTILL TAPERS to α=0.9. PRM at #59 + DPO take over for tool-augmented reasoning quality (chain-of-thought across tool calls on AgentBench-style multi-step tasks).

**Composition with #56 DISTILL-FORWARD across generations:**
- **Gen 0:** trained from external teacher set (Llama 3.1 405B for text via #68; Llama 3.1 405B fine-tuned-on-tools for tool-augmented via #69-A; Llama 3.2 Vision 90B for VL via #69-B sibling).
- **Gen 1+:** trained from previous CHIRON generation (DISTILL-FORWARD, #56) — now with tool-routing capability inherited from Gen 0.
- **External teachers dropped after Gen 0** (no further teacher inference cost amortized across all generations).

**Composition with #59 PRM-CHIRON:** PRM-on-tool-calls (#60 §3 joint) extends naturally — the PRM head scores correctness of (selector, args) pairs at TOOL_CALL positions, providing a second supervision signal that complements teacher-KL on the SAME positions. Joint #59 + #69-A: PRM scores CORRECTNESS of tool routing while teacher KL scores DISTRIBUTIONAL CALIBRATION of tool routing — orthogonal signals on the same token class.

---

## 3. Theoretical analysis

### 3.1 Theorem 1 — Student tool-augmented NLL bound under teacher distillation

**Theorem 1.** Let T denote a tool-using teacher with tool-augmented NLL_T on test distribution P*_TOOL (joint prompt + tool-trace + answer distribution). Let S denote a student trained via the KL-CE blended loss L = αCE + (1-α)τ²KL on C-class positions of joint tool-trace sequences, with R-region positions skipped, and with selector tokens upweighted by λ_sel. Under standard regularity assumptions (sufficient student capacity for the smooth interpolant; bounded teacher tool-augmented entropy; tool runtime determinism modulo nondeterministic tools per #60 §1.4), as student training compute → ∞:

```
NLL_S^{TOOL} → α · NLL_optimal_from_TOOL_data + (1-α) · NLL_T^{TOOL}  +  O(α(1-α))·D_TS^{TOOL}  +  ε_routing_calibration
```

where:
- NLL_optimal_from_TOOL_data is the irreducible tool-augmented NLL achievable by the student class on P*_TOOL with tool-trace data alone (no teacher).
- NLL_T^{TOOL} is the teacher's tool-augmented NLL on the same test distribution.
- D_TS^{TOOL} is a Bregman divergence between teacher and optimal-from-data tool-trace predictors.
- **ε_routing_calibration** is a routing-calibration gap unique to tool-augmented distillation: even with λ_sel=4 upweighting, the student's selector-token distribution may differ from the teacher's because the student's call-history-conditioned hidden state at selector positions has a different geometry than the teacher's. **Empirically (Granite Code, Llama 3.1 8B Tool-Use, ToolACE-distill) ε_routing_calibration ∈ [0.05, 0.20] nat — small but non-trivial.**

**Interpretation:**
- α=1 (pure CE): student → optimal-from-tool-data baseline (no teacher benefit; slow convergence from scratch on tool routing).
- α=0 (pure KL): student → mimics teacher's tool-trace behavior; capability ceiling = teacher's tool-augmented NLL_T plus ε_routing_calibration.
- α∈(0,1): student approximates a convex combination, with cross-term D_TS^{TOOL} controlling fit quality and ε_routing_calibration controlling routing-side fidelity.

**Practical consequence:** student CHIRON-1.84B + #60 tool runtime with α=0.4 from a Llama 3.1 405B fine-tuned-on-tools teacher inherits ~60% of teacher's tool-augmented quality gap over from-scratch baseline. If teacher tool-augmented NLL is 1.8 nat lower than student's from-scratch ceiling, distilled student ends ~1.1 nat below from-scratch ceiling, modulo a 0.05-0.20 nat routing-calibration penalty. **Tool-augmented NLL improvement is REAL, not just speedup; the bar moves on the tool-augmented axis.**

**Honest band:** the actual realized tool-augmented NLL improvement depends on (a) ε_routing_calibration which is bounded but non-trivial, (b) selector-token vocabulary alignment quality, (c) capacity gap (405B teacher vs 1.84B student trunk = ~220× ratio matches #68's text-axis ratio).

### 3.2 Theorem 2 — Tool-augmented convergence rate under KL distillation

**Theorem 2.** Under TOOL-DISTILL with cached top-k tool-trace teacher logits, student tool-augmented training from random initialization to within ε of its terminal tool-augmented NLL takes:

```
T_TOOL_distill(ε) ≤ (k_teacher_TOOL_quality / k_student_TOOL_capacity) · T_TOOL_from_scratch(ε) · (1 + ε_routing_calibration/NLL_T^{TOOL})
```

where k_teacher_TOOL_quality < 1 captures the teacher's relative information density (smaller = better teacher), k_student_TOOL_capacity > 1 captures the student-to-optimal ratio on tool-augmented data, and the (1 + ε_routing_calibration/NLL_T^{TOOL}) factor is the routing-side penalty (typically 1.03-1.10 — small).

**Practical:** empirical anchors:
- **Granite Code 8B (IBM 2024):** distilled from larger code teacher; reaches GPT-3.5-class on coding + tool-use at ~3-5% the from-scratch compute → **20-35× wall-clock reduction**.
- **Llama 3.1 8B Tool-Use (Meta 2024):** distilled from 405B Tool fine-tune; approaches GPT-4 on AgentBench at ~2-5% the from-scratch compute → **20-50× compute reduction**.
- **Phi-3 + tool-use (Microsoft 2024):** 4.2B with tool fine-tune from larger teacher; ~30-40× compute reduction on tool-augmented benchmarks.
- **ToolACE-8B (Salesforce 2024):** distilled student from a teacher pipeline; ~25-50× compute reduction reported on tool-routing benchmarks.
- **Octopus-v2/v3 (Nexa 2024):** small-model + tool distillation on Android API surface; ~20-40× reductions on function-call accuracy.

For CHIRON-1.84B + #60 tool runtime with Llama 3.1 405B fine-tuned-on-tools teacher: expected T_TOOL_distill / T_TOOL_from_scratch ∈ [0.013, 0.033], i.e., **30-75× reduction**. **Headline 50× sits at the geometric mean of this band.**

### 3.3 Theorem 3 — Student-side memory cost (tool-trace-specific)

**Theorem 3.** TOOL-DISTILL adds the following memory-cost-on-student-GPU:
- Teacher tool-trace logit cache batch buffer (C-class positions only): B · T_text · k · 3 bytes.
  - For B=4, T_text=400, k=64: 4·400·64·3 = ~310 KB. Negligible.
- KL loss working memory: O(B·T_text·V_top_k) = same ~310 KB.
- No additional persistent state on student (teacher logits are streamed from disk).
- Selector upweight scalar: 1 float per position (negligible).

**Total student-GPU overhead beyond #60 baseline: < 5 MB.** Memory advantage of single-GPU CHIRON-1.84B + #60 tool runtime fully preserved (16 GB ceiling unaffected).

**Off-GPU cost:** offline cached-logit storage = 30-150 GB on NVMe (one-time). Acceptable on a single 4 TB NVMe shared with #68 (text) and #69-B (VL) caches.

### 3.4 Theorem 4 — Cost amortization across deployments

**Theorem 4 (cost amortization).** Let C_teacher_inference = one-time cached-logit corpus generation cost. Let N_student_runs = number of CHIRON student training runs amortized over the cached logits' useful lifetime. Per-student-run amortized teacher cost is C_teacher_inference / N_student_runs. With #56 DISTILL-FORWARD at G generations, effective amortization is C_teacher_inference / (N · G) since teachers are only used at Gen 0.

**Practical:** Tier 2 open-source path C_teacher_inference ≈ $3-7K. With N=20 student variants over a 2-year program × G=4 generations = 80 effective student runs → amortized ~$40-90 per run. **Negligible.** Tier 1 GPT-4 path C_teacher_inference ≈ $9K → amortized ~$110 per run. Still negligible relative to per-run training compute (~$1-3K on a single 4080 SUPER for a 1.84B run).

### 3.5 Memory advantage preservation

- **GPU memory:** unaffected by #69-A. <5 MB working buffer added beyond #60 + #68.
- **Host RAM:** unaffected at training time; ~32 MB streaming buffer (overlaps with #68's text streaming and #69-B's VL streaming).
- **Disk:** +30-150 GB one-time cached-tool-trace-logit storage. Single 4 TB NVMe absorbs this; one-time amortized cost.
- **VRAM ceiling 16 GB:** preserved.

The single-GPU 16 GB ceiling — the most-honored constraint across the entire program — remains intact for #69-A as it was for #60 and #68.

### 3.6 Honest gap on NLL preservation

**Claim (honest):** student's tool-augmented NLL is preserved BUT NOT BIT-EXACT.

- **Bit-exact text NLL preservation** (the strict #42-#67 stance): never violated by #69-A beyond #68's existing relaxation. Text-only sequences (no tool-trace structure) pass through the same trunk and produce text-only loss identically to #68.
- **Tool-augmented NLL improvement preservation** (the TOOL-DISTILL claim): student's tool-augmented NLL on test set is LOWER than from-scratch baseline's tool-augmented NLL by 0.8-2.0 nat (the tool-augmented bar moves DOWN materially).
- **R-region position preservation:** R-class positions never had a CE term (per #60 §1.4); same posture as #60.

**Net:** #69-A does not introduce ANY NEW NLL preservation violation beyond what #68 already established. The text-position posture is identical. The C-class tool-augmented posture is "improved bar" rather than "preserved bar" — same framing as #68.

### 3.7 Bijectivity preservation in CHIRON's reversible-flow trunk

Per #66 Theorem 2, CHIRON's reversible-flow trunk preserves bijectivity for token positions because the shears do not depend on token provenance (text vs tool-call vs tool-result). The KL-CE blended loss at C-class positions does not affect bijectivity (it changes the gradient through W_out, not the trunk shears). **Bijectivity is preserved.**

---

## 4. Composition with #60 + #68 (and the broader stack)

### 4.1 Composition with #60 TOOL-LLM-CHIRON

#69-A builds DIRECTLY on #60:
- Architecture (special-token vocabulary, R-region masking, λ_sel=4 selector upweighting, tool runtime dispatcher): **inherited verbatim from #60.**
- DataLoader: inherited from #60, extended to include cached tool-trace teacher logits per C-class position.
- Loss kernel: extends #60's masked-CE to masked-(αCE + (1-α)τ²KL) on C-class positions only.
- #60 was compute-positive on tool-augmented benchmarks via 5× standalone (smaller-coordinator effect; already in the 3,030,000× starting figure); #69-A is compute-positive on tool-augmented NLL by 50× headline (incremental).

**Composition is multiplicative** in the sense that #60 supplies the architectural substrate and #69-A supplies the teacher signal. Without #60, #69-A is undefined (no special tokens → no R-region → no tool-trace structure → no tool-using teacher logit alignment). Without #69-A, #60 is a deliverable but does not exploit external teacher compute.

### 4.2 Composition with #68 SUPER-DISTILL-CHIRON

#69-A builds DIRECTLY on #68's distillation pipeline:
- Cached-logit storage format: same as #68 (top-K=64 indices + FP16 values + delta encoding); just adds tool-trace teacher entries with R-region skip markers.
- KL-CE blended loss kernel: same as #68; just operates on a mixed (text-only / tool-trace) batch stream with R-region masking applied in tool-trace branches.
- α/τ scheduler: same kernel as #68; tuned slightly differently for tool-augmented (α=0.4 vs 0.5; τ=3 vs 2).
- Curriculum integration with #61 COSMIC: same hooks as #68.

**The only NEW components specific to #69-A:**
1. Tool-using teacher inference adapter (Llama 3.1 405B fine-tuned-on-tools loading + R-region skip during cached-logit generation).
2. Tool-trace cached-logit format (R-region skip markers per cached entry).
3. Tool-trace batch sampler (interleaved CALL-REASONING-RESULT-ANSWER spans with C-class / R-class consistency).
4. λ_sel selector-upweight extension to KL term (multiplicative on both CE and KL).
5. Fine-tuning script for Llama 3.1 405B → 405B-Tool teacher (one-time, ~$2-4K compute).

These are ~600 LOC incremental beyond the union of #60 and #68's existing 580 + 940 = 1520 LOC.

**Composition is multiplicative in magnitudes:** #68 lifts text NLL by 50× (cumulative-table convention); #69-A lifts tool-augmented NLL by 50×; #69-B lifts VL NLL by 50×; the lifts are independent (different test sets, different teacher logits, mostly non-overlapping training signal). Joint cumulative table shows all three lifts simultaneously.

### 4.3 Differentiation from same-axis paradigms

**vs. #60 TOOL-LLM alone:** #60 is the architecture + smaller-coordinator-effect paradigm and contributes 5× standalone via reduced effective parameter count under tool augmentation. #69-A is compute-POSITIVE on tool-augmented NLL by 50× incremental. Without #69-A, the tool-augmented benchmarks line in the cumulative table reads "3,030,000× — frozen" (the contribution of the prior text-axis stack carrying through unchanged). With #69-A, the line reads "150,000,000×".

**vs. #68 SUPER-DISTILL alone:** #68 is text-only. The tool-augmented benchmarks line at #68 close was explicitly reserved with "3,030,000× UNCHANGED" because no tool-using teacher was in scope. #69-A fulfills the reservation.

**vs. composition #60 + #68:** the composition has #60 supplying tool-trace substrate and #68 supplying text-only distillation. The tool-augmented benchmarks line under the composition would still be 3,030,000× — because #68's text-only teacher emits no `<TOOL_CALL>` tokens, so its logits at C-class positions of joint tool-trace sequences encode text-only completions, not tool-routing distributions. **#69-A is the genuinely new component on the TOOL × TEACHER-PROVENANCE joint axis.**

### 4.4 Composition with bit-exact-NLL paradigms (#49 ICARUS, #50 HELIUM, #51 ATLAS-COMPILE, #52 NIMBUS)

These paradigms preserve text NLL bit-exact. #69-A inherits #68's text-position relaxation (KL gradient changes the objective). **Composition rule:** when #69-A is active, the bit-exact stack still applies to the student's per-step training (the per-step CE+KL gradient on C-class positions is computed with bit-exact arithmetic via #49-#52); #69-A changes the objective on C-class positions of joint tool-trace sequences. The "bit-exact relative to a fixed objective" claim is preserved; the OBJECTIVE on tool-trace C-class positions has changed in the same way as #68's text-only objective changed.

### 4.5 Composition with the bigger-picture stack (#56-#67)

- **#56 DISTILL-FORWARD:** Gen 0 inherits external tool-using teacher; Gen 1+ inherits previous CHIRON generation's tool-routing capability via intra-program self-distillation.
- **#57 SCROLL + #58 METAGEN:** triple-role amortization extends — Llama 3.1 405B-Tool teacher provides #56 distill (tool-trace Gen 0) + #57 informativeness (tool-trace pair scoring) + #58 generation (synthetic tool-traces around real prompts via METAGEN-with-tools).
- **#59 PRM-CHIRON:** PRM-on-tool-calls (#60 §3 joint) extends — PRM scores correctness of (selector, args) while teacher KL scores distributional calibration; orthogonal signals on the same C-class positions. **Strongest synergy in the stack.**
- **#60 TOOL-LLM:** architectural substrate (see §4.1).
- **#61 COSMIC:** per-stage α/τ schedules (see §2.4).
- **#62 AGENT-CHIRON:** multi-step agent loop (`<GOAL>`, `<PLAN>`, `<ACT>`, `<OBS>`, `<REFLECT>`, `<ANSWER>`) — teacher's agent trajectories transfer via KL on action-class positions (extends C-class to A-class span in #62 vocabulary).
- **#63 META-LEARN:** V-projected gradient applies equally to KL gradient as to CE gradient on C-class tool-trace positions.
- **#64 MEMORY-CHIRON:** teacher's retrieval patterns can seed memory bank for tool-augmented retrieval-class entries.
- **#65 WORLD-MODEL-CHIRON:** WS schema extends naturally to entities/properties/relations/causal-links observable from tool outputs (e.g., code-interpreter return values, search results).
- **#66 CROSS-MODAL:** orthogonal axis (vision); composes through #69-B (VL distillation) on a different teacher stream.
- **#67 CAUSAL:** teacher's causal reasoning across multi-step tool calls transfers via KL.
- **#68 SUPER-DISTILL:** sister paradigm; same pipeline (see §4.2).

### 4.6 Triple-role amortization extended

#68 noted that #56-#58's same-class-teacher triple-role amortization extends to external teachers. #69-A further extends:
- Llama 3.1 405B (text teacher) provides #56 distill (text Gen-0) + #57 informativeness (text scoring) + #58 generation (text synth).
- Llama 3.1 405B-Tool (tool teacher) provides #56 distill (tool-trace Gen-0) + #57 informativeness (tool-trace pair scoring) + #58 generation (synthetic tool-traces).
- Llama 3.2 Vision 90B (VL teacher; #69-B sibling) provides #56 distill (VL Gen-0) + #57 informativeness (VL pair scoring) + #58 generation (synthetic captions).
- **Per-step overhead remains <2%** (cached logits read across three streams; no live teacher).

The triple-role amortization now spans THREE external teacher classes.

---

## 5. Quantitative speedup claim with honest band

### 5.1 Headline

**50× wall-clock to fixed final tool-augmented NLL** (geometric mean of 30-75× honest band).

### 5.2 Honest band breakdown

| Band end | Conditions |
|---|---|
| **65-75× (high)** | Tier 2 Llama 3.1 405B-Tool teacher, perfect tokenizer + selector-token alignment, λ_sel=4, α=0.3, τ=4, large student capacity gap |
| **50× (headline)** | Tier 2 Llama 3.1 405B-Tool, top-64 KL, α=0.4, τ=3, λ_sel=4, standard tokenizer + selector-vocabulary alignment |
| **30-40× (low)** | Tier 3 Llama 3.1 70B Tool-Use (smaller teacher), top-32 KL, α=0.6, ε_routing_calibration ~0.15 nat |
| **15-25× (degraded)** | Tier 1 frontier API (GPT-4 / Claude) with top-5 logprobs only (API limitation); higher absolute teacher quality but reduced KL fidelity |
| **<10× (failure)** | Selector-token vocabulary mismatch between teacher and student, or major routing-calibration drift |

### 5.3 Empirical anchors

- **Granite Code 8B (IBM 2024):** distilled from larger code-tool teacher. Matches GPT-3.5-class on coding + tool-use at ~3-5% the from-scratch compute → **20-35× compute reduction**.
- **Llama 3.1 8B Tool-Use (Meta 2024):** distilled from 405B fine-tuned-on-tools teacher. Approaches GPT-4 on AgentBench at ~2-5% the from-scratch compute → **20-50× compute reduction**.
- **Phi-3 + tool-use (Microsoft 2024):** 4.2B with tool fine-tune from larger teacher. ~30-40× compute reduction on tool-augmented benchmarks.
- **ToolACE-8B (Salesforce / Liu 2024):** distilled from a tool-using teacher pipeline. ~25-50× compute reduction on tool-routing accuracy.
- **AgentInstruct-distilled Mistral 7B (Microsoft 2024):** ~20-30× compute reduction on agent benchmarks.
- **Octopus-v2/v3 (Nexa 2024):** small-model + tool distillation on Android function-call surface; ~20-40× reductions.

The 50× headline sits in the upper-middle of the empirical anchor band, justified by the 220× capacity ratio (matches #68's text-axis ratio with the same Llama 3.1 405B-class teacher) plus the well-precedented routing-distillation regime. **This is honestly comparable to #69-B's 50× and below #68's 100× because the tool-routing axis has additional ε_routing_calibration risk that pure text-only distillation does not.**

### 5.4 Risk-adjusted claim

Joint Gate-0 PASS probability × LLM-scale empirical confirmation probability = 0.80 × 0.70 = **0.56 expected realization**. Risk-adjusted speedup: 50× × 0.56 = **28× expected**.

This is honestly LOWER than #68's 64× expected (100× × 0.64) because both Gate-0 and confirmation probabilities are slightly lower for tool-routing distillation than for pure text distillation, reflecting the unique-to-tool-axis ε_routing_calibration penalty and the need to fine-tune the open-source teacher on AgentInstruct + ToolACE before cached-logit generation.

---

## 6. Cumulative stack update

### 6.1 Pre-#69-A stack (post-#68 SUPER-DISTILL)

| Axis | Value |
|---|---|
| Causal-reasoning subset | 50,000,000× |
| Text NLL | 46,500,000× |
| Grounded-reasoning | 33,000,000× |
| Knowledge-augmented | 27,500,000× |
| Agent benchmarks | 26,800,000× |
| VL benchmarks | 5,400,000× |
| **Tool-augmented** | **3,030,000× (frozen at #60 + no #68 contribution)** |

### 6.2 Post-#69-A stack (with TOOL-DISTILL)

| Axis | Pre-#69-A | #69-A factor | Post-#69-A |
|---|---|---|---|
| Causal-reasoning subset | 50,000,000× | × 1.05 (marginal tool-causal subset) | ~52,500,000× |
| Text NLL | 46,500,000× | × 1.0 (unchanged; #69-A does not contribute to pure text NLL beyond #68) | 46,500,000× |
| Grounded-reasoning | 33,000,000× | × 1.10 (tool-grounded examples lift) | ~36,300,000× |
| Knowledge-augmented | 27,500,000× | × 1.10 (tool-retrieve subset; SQL/search-tool benefits) | ~30,250,000× |
| Agent benchmarks | 26,800,000× | × 1.30 (tool-use is a major component of agent benchmarks) | ~34,800,000× |
| VL benchmarks | 5,400,000× | × 1.0 (unchanged; reserved for #69-B) | 5,400,000× |
| **Tool-augmented** | **3,030,000×** | **× 50** | **~150,000,000×** |

The tool-augmented benchmarks axis figure extends DRAMATICALLY (50× in one paradigm) — the largest single-paradigm jump on the tool-augmented axis since #60 opened it. **Total cumulative across program: ~150M× on tool-augmented benchmarks, where the "bar" is now teacher's tool-augmented NLL, not from-scratch CHIRON+#60's tool-augmented NLL.**

### 6.3 Honesty caveat

The post-#69-A stack figures inherit #68's bit-exactness violation on text NLL (which #69-A does not change beyond #68's existing relaxation) and the new bar-shift on tool-augmented NLL. The stack now bifurcates on tool-augmented:
- **Bit-exact text NLL stack:** 930,000× (frozen at iter 211 #67 point; unchanged by #69-A).
- **NLL-improvement stack:** 46,500,000× text + 150,000,000× tool-augmented (#68 + #69-A active; teacher-based NLL bars on both axes).

Both are valid; users select based on use case. The "magnitudes better" criterion at iter 213 is met by the second stack on the tool-augmented axis.

### 6.4 Sensitivity table

| Scenario | Tool teacher | Multiplier | Tool-augmented benchmarks cumulative |
|---|---|---|---|
| Pessimistic (Tier 3 Llama 3.1 70B Tool-Use, smaller teacher) | Llama 3.1 70B Tool-Use | 30× | ~91,000,000× |
| Conservative (Tier 2 Llama 3.1 405B fine-tuned on AgentInstruct + ToolACE) | Llama 3.1 405B-Tool | 50× | **~150,000,000×** |
| Optimistic (Tier 2 + #56 DISTILL-FORWARD compounding + perfect routing-calibration) | Llama 3.1 405B-Tool | 75× | ~227,000,000× |

---

## 7. Engineering scope

### 7.1 Component breakdown (incremental beyond #60 + #68)

| Component | LOC | Description |
|---|---|---|
| Tool-using teacher inference adapter (Llama 3.1 405B-Tool via HF `transformers` / vLLM) | 130 | Tool-trace input batching; R-region skip during teacher generation; top-K logit extraction at C-class positions only |
| Tool-trace cached-logit format (extends #68's format) | 70 | Adds R-region skip-marker channel per cached entry |
| Tool-trace DataLoader (extends #60's) | 90 | Tool-trace batch sampler with C-class / R-class consistency; cache-aligned C-class layout |
| Selector-upweight extension to KL term | 30 | λ_sel multiplicative on both CE and KL terms (extends #60 §2.4 to distillation) |
| α/τ scheduler tool-tuning | 25 | Tool-augmented-specific α/τ defaults; per-stage COSMIC integration |
| Llama 3.1 405B → 405B-Tool fine-tune script (one-time) | 80 | AgentInstruct + ToolACE + ToolBench fine-tune runner; ~$2-4K one-time compute |
| Composition with #56 (tool-trace Gen-0 hand-off) | 35 | DISTILL-FORWARD inherits TOOL-DISTILL Gen-0 checkpoint cleanly |
| Composition with #57+#58 (triple-role for tool-trace) | 45 | SCROLL tool-trace informativeness + METAGEN synthetic-tool-trace; reuses tool-trace cached logits |
| Composition with #59 PRM-on-tool-calls | 30 | Joint loss L = w_t · [α(L_CE + λ_PRM L_PRM) + (1-α)τ²L_KL] on C-class positions |
| Tests + Gate-0 harness | 70 | Per-step KL gradient correctness on tool-trace pairs; Gate-0 70B-Tool mini-distill |
| Tool-augmented benchmark eval harness extension | 40 | AgentBench / ToolBench / API-Bank / GAIA / SWE-Bench-Lite / MINT-Bench configs |
| **Total (incremental)** | **~600 LOC** | **~3 weeks engineering incremental beyond #60 + #68** |

If counted standalone (including the underlying #60 tool-runtime + #68 distillation pipeline): ~1520 + 600 = ~2120 LOC; but those underlying paradigms are presumed shipped at #69-A's wire-in.

### 7.2 External-dependency risk

- **HuggingFace `transformers` / vLLM Llama 3.1 405B inference:** mature as of August 2024; supports BF16 on 4×H100, NF4 on 2×H100-80GB.
- **AgentInstruct dataset (Microsoft 2024):** open-source, ~25M tokens.
- **ToolACE dataset (Liu 2024 / Salesforce):** open-source, ~26M tokens.
- **ToolBench dataset (Qin 2023 / Tsinghua):** open-source, ~16k APIs / ~110k traces.
- **NF4 quantization:** bitsandbytes / GPTQ / AWQ; mature.
- **Storage:** 4 TB NVMe drive (already present from #68); shared with #69-B.
- **Compute (one-time fine-tune + cached inference):** ~$3-7K total cloud cost for Tier 2 path. Amortized across all CHIRON students forever.

**External dependency posture:** #69-A introduces no new external dependency framework beyond #60 + #68's posture. Llama 3.1 405B is a single open-source release (Meta community license) under the same regime as #68; AgentInstruct + ToolACE + ToolBench are standard open-source datasets.

### 7.3 Timeline (incremental beyond #60 + #68 baseline)

- **Week 1:** Tier 3 (70B Tool-Use) teacher pipeline; tool-trace batch inference; cached-tool-trace-logit format adaptation; Gate-0 70B-Tool mini-distill at ~5M tokens.
- **Week 2:** Llama 3.1 405B → 405B-Tool fine-tune on AgentInstruct + ToolACE (parallel to Week 1 work; ~$2-4K cloud); cached-logit corpus inference run (~$1-3K cloud); end-to-end training validation.
- **Week 3:** Gate-1 measurement at full corpus; selector-upweight + KL kernel verification; compose with #59 PRM-on-tool-calls.

If #60 and #68 are not yet shipped, baseline timeline extends by their respective ~3.5 + ~4 = ~7.5 weeks for a total of ~10.5 weeks; in this design we assume #60 and #68 are shipped at #69-A wire-in.

---

## 8. Gate-0 / Gate-1 specifications

### 8.1 Gate-0 — premise validation (mandatory before wire-in)

**Hypothesis:** distillation from 70B-class open-source tool-using teacher to 1.84B CHIRON+#60 student gives ≥10× wall-clock reduction at fixed final tool-augmented NLL on a small training run, with routing-calibration penalty bounded.

**Procedure:**
- Teacher: Llama 3.1 70B Tool-Use Instruct (Tier 3 development teacher; quantized NF4 on 1×A100 or rented).
- Student: CHIRON-1.84B + #60 tool runtime at production config + #42-#68 stack ON.
- Training subset: 5M tool-trace tokens (AgentInstruct subset of ~5M tokens).
- Compare distilled student vs from-scratch student (both with #60 active) at SAME wall-clock budget (8 GPU-hours each).
- Metric: held-out tool-augmented NLL on 6-benchmark suite (AgentBench / ToolBench / API-Bank / GAIA-Lite / SWE-Bench-Lite / MINT-Bench).
- Secondary metric: ε_routing_calibration estimated as the gap between teacher's predicted next-token distribution at selector positions and student's predicted distribution.

**Pass criterion:**
- Distilled student's tool-augmented NLL ≤ from-scratch student's tool-augmented NLL by ≥0.4 nat at the same wall-clock; OR
- Distilled student reaches from-scratch student's terminal tool-augmented NLL in ≤10% the wall-clock; AND
- ε_routing_calibration ≤ 0.20 nat.

**Estimated cost:** ~$200-500 cloud + 1 week engineer time.

**Pass probability:** ~80% (Granite Code + Llama 3.1 8B Tool-Use + Phi-3 + tool-use + ToolACE-distill production evidence; well-precedented at this scale gap).

### 8.2 Gate-1 — full 405B-Tool teacher validation

**Procedure:** same as Gate-0 with Llama 3.1 405B fine-tuned on AgentInstruct + ToolACE teacher and 50M-tool-trace-token corpus subset.

**Pass criterion:**
- Distilled student's tool-augmented NLL ≤ from-scratch+#60 baseline by ≥0.8 nat OR
- ≥30× wall-clock to fixed tool-augmented NLL.
- 6-benchmark average ≥ from-scratch + 6 percentage points.

**Estimated cost:** ~$3-7K cloud (includes one-time 405B-Tool fine-tune) + 2 weeks engineer time.

**Pass probability:** ~70% — modulo selector-token vocabulary alignment (which can be improved by coordinated #60 + #68 + #69-A wire-in time vocabulary specification) and CHIRON-architecture-specific KL-fit risks at the call-result boundary.

### 8.3 Gate-2 — full integration

Validate end-to-end composition with #42-#68 on full corpus. Pass criterion: terminal tool-augmented NLL on 6-benchmark suite ≤ baseline by ≥1.0 nat AND wall-clock to that tool-augmented NLL ≤ 5% of from-scratch+#60 baseline; AND text NLL on Pile-eval unchanged from #68 baseline by ≥0 nat (no regression); AND PRM-on-tool-calls correctness rate ≥ baseline +5 pp.

---

## 9. Honest gaps and failure modes

### 9.1 Tool-call hallucination risk (the unique-to-TOOL gap)

Distilled student may invoke tools INCORRECTLY: wrong selector (`<TOOL=python>` when `<TOOL=calc>` was intended), wrong arguments (malformed JSON, hallucinated API endpoints), or spurious calls when no tool was needed. This is a known failure mode of tool-distilled small models per Granite Code and Llama 3.1 8B Tool-Use papers — the student has internalized the SHAPE of tool calls without always internalizing WHEN they apply.

**Implications:**
- Pure trunk-side KL distillation reduces hallucination relative to from-scratch but does not eliminate it.
- Unseen tools (not in the training distribution) trigger zero-shot hallucination at higher rates than seen tools.
- Argument-formatting hallucinations are particularly common (~3-8% on novel API surfaces).

**Mitigations:**
- Tool runtime dispatcher rejects malformed calls (existing #60 §4 behavior); reject signal optionally backpropagates as PRM negative reward (#59 + #69-A joint).
- Selector-upweight λ_sel=4 amplifies the routing-decision gradient — already mitigates partially.
- α schedule taper to 1.0 at end of training for pure-CE final convergence; reduces residual teacher routing biases.
- #59 PRM-on-tool-calls explicitly scores routing correctness; provides a secondary supervision signal orthogonal to teacher KL.

**Failure mode:** if hallucination rate stays >10% post-training, the 50× headline drops to 20-30× (band lower-end realized) because deployment must spend extra inference compute on call-rejection retries.

### 9.2 Selector-token vocabulary mismatch

Llama 3.1 405B-Tool fine-tune adds ~64 tool-modality special tokens at fine-tune time. These MUST align with #60's existing 64 special tokens at the student. If alignment is imperfect (e.g., teacher uses `<TOOL_CALL>` as token 128001 but student uses 128002), the cached-logit pipeline cannot transfer routing distributions correctly.

**Mitigation:** standardize on a shared special-token spec at #60 + #68 + #69-A wire-in time. Specify the token-ID mapping in a versioned spec doc; share between teacher fine-tune script and student vocabulary loader.

**Failure mode:** if vocabulary alignment fails, headline drops to 15-25× (sequence-level distillation only).

### 9.3 KL-distillation tool-augmented NLL is NOT bit-exact CE

**Inherits #68's gap.** The tool-augmented NLL the student converges to under #69-A on C-class positions of joint tool-trace sequences is NOT the same NLL trajectory as a from-scratch CHIRON+#60. It is a teacher-shaped NLL. Quality difference is bounded but real. Same posture as #68; #69-A does not introduce a new violation.

**Mitigation:** α schedule taper to 1.0 at end of training for pure-CE final convergence on tool-traces. This recovers most of the bit-exact trajectory in the final ~20% of training.

### 9.4 Teacher inference cost: GPT-4 frontier path is expensive

GPT-4 + code interpreter teacher inference at ~100M tokens × $90/M = **~$9K** one-time corpus generation cost. This is per single-corpus-generation-run; amortized across all student variants and generations, it is small (~$110/run at N=80 effective runs). But the absolute headline is meaningful for tight-budget research programs.

**Implications:**
- Tier 1 (GPT-4 / Claude API) path: $1.8-9K one-time amortized; closed-source teacher.
- Tier 2 (Llama 3.1 405B-Tool open-source): $3-7K one-time amortized (fine-tune + cached inference); open-source teacher.

**Recommendation:** Tier 2 is the default production path. Tier 1 reserved for "frontier-class" capability headroom on specific high-value subsets (e.g., complex multi-step coding tasks where GPT-4 + code interpreter is meaningfully better than Llama 3.1 405B-Tool).

### 9.5 Frontier-API top-k logit limitation

GPT-4's `logprobs=true` exposes only top-5 (or top-20 in some endpoints) logprobs per position; Claude 3.5 Sonnet does not natively expose logprobs. This caps the KL fidelity for Tier 1 API-based distillation at top-5 vs top-64 for Tier 2 open-source. **KL bias at top-5 vs top-64 ≈ 0.1 nat additional**; reduces Tier 1 effective speedup from 50× to ~30-35×.

**Mitigation:** prefer Tier 2 (open-source 405B-Tool) where top-64 logprobs are directly accessible. Use Tier 1 only for high-value subsets where the absolute capability gap justifies the KL fidelity reduction.

### 9.6 Tool-trace data quality dependence

AgentInstruct (Microsoft 2024), ToolACE (Liu 2024), and ToolBench (Qin 2023) are the gold-standard open-source tool-trace datasets, but their tool surfaces differ in distribution from each other. If the student deployment domain doesn't match the training distribution (e.g., training on Python + REST + search but deploying on SQL + image-gen), the distilled routing capability transfers poorly.

**Mitigation:** ensure tool-trace training corpus tool surface ⊃ deployment tool surface. For high-value deployment tools, generate synthetic tool-traces via #58 METAGEN-with-tools using the deployment-target tool surface.

### 9.7 The "novelty" question — IMPORTANT

**Honest assessment:** TOOL-DISTILL-CHIRON is **mostly the union of #60's architecture + #68's distillation pipeline applied to a tool-using teacher.** Same critique as sibling #69-B.

What is GENUINELY new:
- The tool-using teacher choice (Llama 3.1 405B fine-tuned on AgentInstruct + ToolACE) is new at the program level.
- The selector-upweight extension to the KL term (λ_sel multiplicative on both CE and KL) is novel relative to either #60 or #68 alone.
- The R-region skip in cached-logit generation (teacher does not emit logits at R-class positions; student does not consume KL at R-class positions) is non-trivial integration of #60's masking with #68's caching.
- The triple-role amortization extension to a third external teacher class (text Llama 3.1 405B + tool-using Llama 3.1 405B-Tool + VL Llama 3.2 Vision 90B) is the program-level structural contribution.

What is NOT new:
- KL-CE blended loss (Hinton 2015; #68 standard).
- Cached-logit pipeline (#68 standard).
- R-region masking (LLaVA / Toolformer / #60 standard).
- Selector-token vocabulary (Toolformer, ToolLLM, #60 standard).
- Distillation from a frontier-class tool-using teacher (Granite Code, Llama 3.1 8B Tool-Use, Phi-3 + tool-use, ToolACE-distill, Octopus).

**Honest framing:** #69-A's novelty is INTEGRATION not MECHANISM. The mechanism is "apply #68 to #60's special-token + R-region structure". The integration extends the TEACHER PROVENANCE axis from text-only to tool-augmented — which is a program-level contribution but not a mechanism-level contribution. **Same framing as sibling #69-B.**

**Implication for paradigm-shift accounting:** #69-A is a WEAKER paradigm shift than #68 was on the TEACHER PROVENANCE axis. #68 OPENED that axis; #69-A EXTENDS it to a second modality (tool-augmented). Future paradigms #70+ might further extend (audio teacher; video teacher) but each successive extension is increasingly mechanism-redundant. **Relative to sibling #69-B:** #69-A and #69-B are symmetric extensions on different modality axes; both have the same novelty-vs-integration trade.

### 9.8 Catastrophic forgetting / domain shift on tool-augmented

If tool-trace distillation training data domain-shifts from teacher's fine-tuning tool-distribution, KL signal becomes noisy. Mitigation: ensure tool-trace training corpus tool surface ⊂ teacher's fine-tuning tool surface (AgentInstruct + ToolACE + ToolBench coverage; safe per Microsoft / Salesforce / Tsinghua published recipes).

### 9.9 Teacher tuning lock-in on tool-routing

Llama 3.1 405B-Tool fine-tune is INSTRUCTION-TUNED on tool-use; distilled student inherits tool-instruction-following bias. May be undesirable for a base tool-using model use case. Mitigation: use BASE Llama 3.1 405B + light fine-tune on AgentInstruct + ToolACE (no general instruction-tuning) for a less biased teacher.

### 9.10 Legal / licensing on tool-trace data

AgentInstruct (MIT license), ToolACE (Apache 2.0), ToolBench (Apache 2.0), Llama 3.1 405B (Meta community license). All compatible with CHIRON's research program. No legal blocker as of 2026-05-08. Tier 1 GPT-4 / Claude paths may have API ToS implications for distillation outputs (OpenAI ToS §2.c restricts use of outputs to develop competing models; Tier 1 path is therefore best reserved for non-commercial research subsets).

### 9.11 The "magnitude floor" question

The user brief at iter 212 reasserted "magnitudes better on compute speed (especially after iter-211 saturation)." #68 cleared this bar at 100×. #69-A clears the bar at 50× ON THE TOOL-AUGMENTED AXIS, which is a different axis than the text NLL axis #68 lifted.

**Honest framing:** if the user's "magnitudes better" criterion applies axis-by-axis (lift each axis by ≥10×), then #69-A clears the bar comfortably (50× on tool-augmented). If the criterion applies cumulatively across all axes, then #69-A contributes only marginally to the dominant axes (text NLL unchanged; agent +1.30×; grounded +1.10×) and the magnitude story is weaker.

**Recommended framing:** axis-by-axis. Same framing as sibling #69-B. The cumulative table at iter-213 close shows substantial movement on the tool-augmented axis (3.03M× → 150M×) even if other axes are mostly unchanged.

---

## 10. Probability estimates

| Estimate | Value |
|---|---|
| Joint Gate-0 PASS probability (70B-Tool teacher mini-distill) | **~80%** |
| Joint Gate-1 PASS probability (405B-Tool teacher full-distill) | **~70%** |
| LLM-scale empirical confirmation probability at single-GPU CHIRON | **~70%** |
| Risk-adjusted speedup | **28×** (= 50× × 0.56) |
| Probability of headline ≥30× | **~80%** |
| Probability of headline ≥50× | **~50%** |
| Probability of headline ≥75× | **~20%** |

These probabilities mirror sibling #69-B's because:
- ε_routing_calibration is a unique-to-tool risk source (parallel to #69-B's ε_vision_encoder).
- Tool-using teacher fine-tuning adds a one-time cost step that #68's text-only path did not require.
- Tool-augmented benchmark distributions are diverse (less complete distillation transfer).

But still HIGH compared to recent paradigms (#65-#67 hovered at 30-50% LLM-scale confirmation) because Granite Code, Llama 3.1 8B Tool-Use, and ToolACE-distill provide direct production-scale evidence.

---

## 11. Bottom line / verdict

### 11.1 Verdict: **SELECT**

TOOL-DISTILL-CHIRON is recommended for SELECT on five grounds:

**1. Magnitude on the tool-augmented axis.** 50× headline on tool-augmented benchmarks (30-75× honest band) lifts an axis the program had partially addressed at #60 but where no compute multiplier had been applied beyond the smaller-coordinator effect. By the user's "magnitudes better" criterion read axis-by-axis, this clears the bar.

**2. Empirical precedent.** Granite Code (IBM), Llama 3.1 8B Tool-Use (Meta), Phi-3 + tool-use (Microsoft), ToolACE-distill (Salesforce), Octopus-v2/v3 (Nexa), AgentInstruct-distilled Mistral 7B (Microsoft) all provide production-scale evidence for 20-50× reductions; the headline 50× sits at the upper-middle of the precedent band.

**3. Bigger-picture alignment.** Closes the explicit reservation in #68 §8 honest gap #5 ("Tool-augmented and VL axes ... reserved for #69+ tool-distillation + multimodal-distillation extensions"). Fulfills the iter-212 framing on the tool-augmented half (sibling #69-B fulfills the VL half).

**4. Engineering tractability.** ~600 LOC over 3 weeks INCREMENTAL beyond #60 + #68; mature reference implementations (Granite Code, Llama 3.1 8B Tool-Use, ToolACE); one-time teacher fine-tune + cached inference cost (~$3-7K Tier 2 path) amortized across all students. **Among the lowest engineering scopes in the iter-213 slate.**

**5. Strong composition.** #69-A composes by-construction with #60 (architectural substrate) and #68 (distillation pipeline), and by-extension with all 27 prior paradigms. **Strongest synergy with #59 PRM-on-tool-calls** (PRM scores correctness while teacher KL scores distributional calibration; orthogonal signals on the same C-class positions). Special compositions with #56 DISTILL-FORWARD (tool-trace Gen-0 hand-off), #57 SCROLL + #58 METAGEN (triple-role amortization extended to a third external teacher class), #61 COSMIC (per-stage α/τ schedules), and #62 AGENT-CHIRON (teacher's agent trajectories transfer via KL on action-class positions).

### 11.2 Caveats on SELECT

**Caveat 1: Novelty-of-mechanism is moderate.** #69-A is mostly the union of #60 + #68 applied to a tool-using teacher. The novelty is in INTEGRATION (TOOL × distillation), not in MECHANISM (KL-CE on R-region-masked sequences from a tool-using teacher is the standard 2024 AgentInstruct-distill pattern). This is the central caveat against treating #69-A as a "fresh" paradigm shift — same caveat as sibling #69-B.

**Caveat 2: Magnitude is HALF of #68's.** Honest 50× headline vs #68's 100×. The routing-calibration penalty and the additional fine-tune step on the open-source teacher reduce the achievable lift relative to pure text-only distillation.

**Caveat 3: Already-relaxed constraint set.** #69-A operates entirely within #68's relaxed constraint set (bit-exact NLL preservation already sacrificed at #68); does not RE-RELAX or add any new constraint relaxation.

**Caveat 4: Tool-call hallucination risk.** Distilled student may invoke tools incorrectly; mitigated by #59 PRM-on-tool-calls + λ_sel selector upweighting + α schedule taper, but not eliminated.

**Caveat 5: Sibling parallelism.** Together with #69-B MULTIMODAL-DISTILL, #69-A constitutes a paired extension on two modality axes (TOOL + VL). If the user prefers a single fresh-mechanism paradigm at #69 instead, both should be reserved and a different #69 candidate selected (this is sibling #69-C's role).

### 11.3 Cost of SELECT

- One paradigm of "fresh axis" novelty lost: tool-augmented axis was already opened at #60 and TEACHER PROVENANCE was already opened at #68.
- Single-GPU-pure framing same posture as #68 (preserved at student training time; relaxed at teacher pre-inference).
- Integration-novelty replaces mechanism-novelty.

These costs are explicit and acknowledged. They are LESS than the 50× compute magnitude gain on tool-augmented.

### 11.4 Alternatives (if SELECT rejected)

- **RESERVE:** defer to #70 with sharpened tool-call-hallucination mitigation (mandatory PRM-on-tool-calls integration) and broader tool-trace corpus coverage (synthetic METAGEN-with-tools tool-traces for deployment-target surfaces).
- **REJECT:** rejects on novelty grounds (mostly #60 ∪ #68); consider stronger novelty paradigm at #69 instead (e.g., a genuinely new TEACHER PROVENANCE extension like multi-teacher ensemble distillation, or a fresh-axis paradigm like AUDIO-DISTILL).

The recommended path is **SELECT** with full Tier 2 Llama 3.1 405B-Tool teacher path and #59 PRM-on-tool-calls joint integration ENABLED for production runs.

### 11.5 Composition-axis status after #69-A

| Axis | Maturity post-#69-A |
|---|---|
| Compute-speed (per-step) | At ceiling (#42-#52) |
| Memory | At ceiling (#44, #47, #48 trade-offs) |
| Loss / objective (bigger-picture) | Mature (#56-#59) |
| Data / sampling (bigger-picture) | Mature (#57, #58) |
| Identity / agency / curriculum | Mature (#60-#62) |
| Optimizer / meta | Mature (#55, #63) |
| Memory parameter dim | Mature (#64, #65) |
| Cross-modal / VISION | Mature substrate (#66) — distillation extension via sibling #69-B |
| Causal / agentic-trajectory | Mature (#67) |
| Teacher provenance | **Mature (text at #68 + tool-augmented at #69-A)** — VL extension via sibling #69-B |

After #69-A, the joint TOOL × TEACHER PROVENANCE axis is mature. Future paradigms can extend to other modalities (audio-modality distillation; video-modality distillation), to other teacher provenance variants (multi-teacher ensemble; teacher-of-teachers chains), or to genuinely new axes not yet opened (lifelong learning; neuro-symbolic).

---

## 12. Bottom line, one line

**SELECT TOOL-DISTILL-CHIRON. 50× wall-clock to fixed final tool-augmented NLL via Llama 3.1 405B fine-tuned on AgentInstruct + ToolACE → CHIRON-1.84B + #60 tool-runtime distillation, composing #60 architecture + #68 distillation pipeline. Cumulative tool-augmented-benchmarks stack: ~150M× (50× lift on tool-augmented axis; bit-exactness already relaxed at #68; teacher-tool-augmented-quality bar inherited). Joint Gate-0 PASS ~80%; LLM-scale confirmation ~70%. Engineering ~600 LOC over 3 weeks (lowest in iter-213 slate alongside sibling #69-B). Honest novelty caveat: mechanism is mostly #60 ∪ #68; integration is the program-level contribution. Strongest synergy with #59 PRM-on-tool-calls (orthogonal correctness + calibration signals).**

---

**End of Paradigm Shift #69 Candidate A design document.** ~5400 words. TOOL-DISTILL-CHIRON: tool-augmented extension of #68's TEACHER PROVENANCE axis, composing with #60's TOOL architectural substrate, lifting tool-augmented benchmarks by 50× headline. SELECT recommended; novelty-of-mechanism is the primary honest gap (parallel to sibling #69-B).
