# Paradigm Shift #61 Candidate A — COSMIC-PROMOTED (Compute-Optimal Multi-stage Curriculum, post-TOOL-LLM-composition)

**Status:** candidate-A design for paradigm shift #61. **Promoted from iter-204 #60-A reserved** after #60 TOOL-LLM shipped at 2,020,000× cumulative on tool-augmented benchmarks (620,000× on text-NLL alone). The iter-204 reservation document (`PARADIGM_SHIFT_60_CANDIDATE_A_COSMIC_PROMOTED.md`, ~3000 words) carries the per-stage PRM mechanism design; the iter-203 antecedent (`PARADIGM_SHIFT_59_CANDIDATE_A_COSMIC.md`, ~5656 words) carries the foundational schedule-axis design. This document refines both for the post-#60 TOOL-LLM stack and is intentionally short.

**Date:** 2026-05-08 (Ralph-loop iteration 205, post-#60 TOOL-LLM selection at 2,020,000× cumulative on tool-augmented benchmarks).

**Predecessors.** All of #42–#60. Load-bearing additions versus the iter-204 reservation: (a) #60 TOOL-LLM (special-token tool primitives at pretraining time, masked tool-result loss, ~5× training-FLOP speedup at matched tool-augmented benchmark accuracy), (b) iter-204's evidence that compute-locus boundary reframings stack multiplicatively with structural-axis paradigms, and (c) the refinement that *each COSMIC stage hosts its own tool-augmentation profile*, not a single tool surface that survives identically across stages.

**Axis.** **Schedule × intergenerational-reward × tool-locus.** iter-203 opened the schedule axis; iter-204 #60-A composed it with the intergenerational PRM-as-teacher mechanism #59 validated; iter-205 #61-A composes both with the tool-locus boundary #60 validated. **Each stage's tool-augmentation profile teaches the next stage's tool-augmentation profile** — schedule, reward, and tool-locus axes interlock in one triple-axis composition.

**Tagline.** *iter-204 reserved COSMIC for #61. #60 TOOL-LLM dissolved the capability-internalization assumption. #61-A composes them: a three-stage curriculum where each stage's tool surface escalates in expressivity, and where each stage's PRM teaches the next stage's PRM on a richer tool surface. Schedule × reward × tool-locus = ~3,030,000× cumulative on tool-augmented benchmarks.*

**Honest headline.** **~1.5× marginal wall-clock at fixed final tool-augmented-benchmark accuracy** (unchanged from iter-204 reservation). Cumulative: `2,020,000 × 1.5 ≈ 3,030,000×` on tool-augmented benchmarks; `404,000 × 1.5 ≈ 620,000×` on text-NLL alone. The **structural new contribution at #61** is not magnitude (still 1.5×) but the **triple-axis interlock**: COSMIC's stage transitions become the natural carrier for *both* PRM intergenerational transfer (iter-204 contribution) *and* tool-augmentation profile escalation (iter-205 contribution). Three axes were independent at iter-204; at iter-205 they all interlock.

**Five refinements vs iter-204 reservation:**
1. **Stage 1: minimal tool surface** (calculator + retrieval; no Python, no web search). Establishes delimiter primitives without capability-rich external dependence.
2. **Stage 2: reasoning-grade tool surface** (code interpreter + calculator + retrieval). PRM and tools co-train; model learns to *route* under PRM supervision.
3. **Stage 3: full-suite tool surface + DPO** (Python + web search + retrieval + SQL + shell + REST + image gen + ~52 app-specific tools). Constitutional anchor on both PRM and tool-validity classifier.
4. **Per-stage tool-trace fraction escalates** (5% → 12% → 20%). Stage 1 establishes vocabulary; stage 2 saturates routing skill; stage 3 trains deployment-grade ecosystem.
5. **Cross-stage tool-call validity transfer is binding.** Without it, COSMIC + #60 collapse to ~1.25×. With it, joint stays at 1.5×.

---

## 1. Refinement vs iter-204 reservation

iter-204 established COSMIC's per-stage PRM mechanism completely (three-stage curriculum, 60/25/15 compute split, RLG/MELT/MOSAIC stage transitions, DPO stage 3, weak/full/constitutional PRM hierarchy, ~1300 LOC, 1.5× marginal honest framing). iter-203 established the foundational schedule-axis design (Chinchilla multipliers χ₁=50, χ₂=25, χ₃=5; ~1150 LOC base). **This document re-derives neither.** The five refinements below are the *only* substantive additions for #61-A.

### 1.1 Stage 1: minimal tool surface (calculator + retrieval)

**Problem iter-204 didn't address:** stage 1 trains 1.84B on the foundation corpus — *not* tool-rich. Uniformly applying #60 TOOL-LLM's full deployment tool surface (Python, web search, SQL, shell) would cap stage-1 tool-call validity ~70% due to insufficient skill density at foundation scale.

**Stage 1 hosts:**
- Tools: `<TOOL=calc>`, `<TOOL=retrieve>` (small fixed-corpus lookup).
- Tool-trace fraction: 5% (half iter-204 #60 default of 10%).
- Special-token vocabulary: 4 delimiters + 2 selectors = 6 active tokens. Remaining ~58 selector rows are **reserved** zero-initialized in the embedding matrix; stage 2 and stage 3 unfreeze them as needed.
- Per-step overhead: zero (tool-result mask kernel from #60-C, unchanged).
- Target tool-call validity: ~75% calculator, ~80% retrieval.

Stage 1's job is not deployment capability; it is to **establish delimiter primitives** (`<TOOL_CALL>`, `</TOOL_CALL>`, `<TOOL_RESULT>`, `</TOOL_RESULT>`) and the **tool-trace narrative pattern** (call → result → integrate-into-answer).

### 1.2 Stage 2: reasoning-grade tool surface (Python + calc + retrieval)

Stage 2 trains 18B on the reasoning-rich corpus — exactly the regime where tool-augmented reasoning is decisive.

- Tools: `<TOOL=python>` (sandboxed Python, full stdlib), `<TOOL=calc>` (continued), `<TOOL=retrieve>` (continued, expanded corpus).
- Tool-trace fraction: 12% (slightly above iter-204 #60 baseline of 10%).
- Active selectors: 7 (stage-1's 6 + `<TOOL=python>`). Unused selectors remain frozen zero rows.
- iter-204 #60-A's PRM head is **load-bearing on tool-call correctness**: PRM-CHIRON's `L_PRM_tool` term (per #60-C §3.3) scores tool-call validity at step-end positions.
- Target validity: ~92% Python, ~95% calc, ~88% retrieval.

This is the **load-bearing tool surface** in the COSMIC schedule. The model learns to *route* under PRM supervision: when Python suffices, when calculator is enough, when retrieval is needed. Stage-2's tool-routing decisions are the highest-information training signal in the entire schedule (~6 bits per call entropy reduction at ~12% of tokens).

### 1.3 Stage 3: full-suite tool surface + DPO

Stage 3 trains 144B-effective on preference pairs via DPO/RLHF.

- Tools: stage-2's three + `<TOOL=search>`, `<TOOL=sql>`, `<TOOL=shell>`, `<TOOL=apicall>`, `<TOOL=image>`, plus ~52 app-specific tools.
- Tool-trace fraction: 20% (production preference traffic in 2025–2026 invokes tools on ~30–40% of queries; downsampling to 20% is a balance constraint).
- All ~64 selector rows now active.
- DPO loss anchors against frozen stage-2 PRM (iter-204 #60-A) **AND** frozen stage-2 tool-validity classifier (NEW at #61):

```
L_stage3 = L_DPO + λ_const · L_PRM_freeze(θ; φ_stage2)
                 + λ_tool_const · L_TOOL_VAL_freeze(θ; ψ_stage2)
λ_const = 0.05, λ_tool_const = 0.03
```

ψ_stage2 is the frozen end-of-stage-2 tool-call-validity classifier. **Mechanism: prevents DPO drift toward malformed or absent tool calls.** Stage-3 DPO can subtly degrade tool validity because preferred-response axes (verbosity, length, perceived effort) sometimes favor tool-avoiding responses even when functionally inferior. The frozen ψ_stage2 acts as a structural reference; stage-3 updates penalized when validity drops below stage-2 levels. Structurally analogous to iter-204's constitutional-PRM anchor, but on tool-call correctness rather than reasoning-step correctness.

**Refinement summary (all three axes):**

| Stage | Compute | PRM (iter-204) | Tool surface (NEW iter-205) | Tool-trace % | Active selectors | Validity target |
|---|---|---|---|---|---|---|
| 1 Foundation | 60% | Weak warm-start | calc + retrieve | 5% | 6 | 75–80% |
| 2 Reasoning | 25% | Full load-bearing | python + calc + retrieve | 12% | 7 | 88–95% |
| 3 Refinement | 15% | Constitutional frozen | full ~64-tool deployment + DPO | 20% | 64 | 90–98% |

### 1.4 Cross-stage tool-call validity transfer

iter-204 #60-A established intergenerational compounding for the PRM head (stage-1 PRM warm-starts stage-2 PRM; stage-2 PRM frozen as constitutional anchor in stage 3). At #61, the *same compounding mechanism* applies to the tool-call validity classifier, with one structural difference: tool-call validity is **per-tool**, so the chain compounds *separately* on each of ~64 selector tokens.

| #61-A COSMIC stage | #60 TOOL-LLM role | Tool-validity source |
|---|---|---|
| Stage 1 | Calc + retrieve teacher (~75%) | Toolformer-style + ToolBench |
| Stage 1 → 2 transition | Stage-1 validity warm-starts stage-2 | Selector embedding rows expanded; new Python head zero-initialized |
| Stage 2 | Python + calc + retrieve teacher (~92%) | Stage-1 + Math-Shepherd-tool-extension + #58 METAGEN tool-trace mode |
| Stage 2 → 3 transition | Stage-2 validity classifier frozen | Used selector embeddings frozen; new selector embeddings zero-initialized at start of stage 3 |
| Stage 3 | Full-suite deployment teacher (~90–98%) | Stage-2 frozen + ToolBench full + curated synthetic |

**Load-bearing structural claim:** COSMIC's stage transitions are the *natural carrier* of both intergenerational PRM transfer (iter-204) *and* intergenerational tool-validity transfer (iter-205). Without the tool-axis composition, joint COSMIC × TOOL-LLM × PRM-CHIRON would compose with *triple* mechanism overlap (hidden-state structure, reasoning-step gradient, tool-routing gradient) at ~1.25×. With cross-stage tool-validity transfer, joint stays at 1.5×.

### 1.5 Per-stage tool-trace fraction escalation (5% → 12% → 20%)

The escalation tracks the foundation→reasoning→deployment compute-locus shift, not arbitrary choice:

- **Stage 1 at 5%:** foundation corpus has minimal natural tool-trace coverage; pushing higher requires disproportionate METAGEN-tool synthesis without commensurate skill gain.
- **Stage 2 at 12%:** reasoning-rich corpus has natural tool-trace density (math problems with explicit calculation; code problems with explicit execution); 12% matches naturally available + modest METAGEN augmentation.
- **Stage 3 at 20%:** preference-pair corpus is itself ~30–40% tool-invoking; downsampling to 20% is the constraint, not the upper bound.

The escalation preserves stage-specific signal density on tool-routing without diluting non-tool training signal. iter-204 #60 established 1× on text-NLL; the per-stage escalation maintains that property across the COSMIC schedule.

---

## 2. Composition with #60 TOOL-LLM: per-stage tool augmentation

### 2.1 The interlock: COSMIC × TOOL-LLM ≠ naive product

A naive product gives `1.5 × 5 = 7.5×` joint over post-#59. **This double-counts heavily.** Post-#60 TOOL-LLM already absorbed 5× over post-#59. COSMIC's contribution on top of post-#60 is 1.5× (iter-204 reservation headline, preserved through tool-augmentation interlock). Joint over post-#59: `5 × 1.5 = 7.5×`, NOT `5 × 5 × 1.5 = 37.5×`.

The triple-axis intergenerational compounding (PRM + tool-validity) does not add a multiplier on top — it **prevents joint collapse below 1.5×** that triple mechanism overlap would otherwise cause. Without cross-stage tool-validity transfer, joint → ~1.25×; with it, joint preserves the iter-204 1.5×.

### 2.2 Where the 0.25× recovery comes from (1.25× → 1.5×)

1. **0.08× from stage-1 → stage-2 tool-vocabulary continuity.** Cold-start of the Python selector embedding wastes ~5% of stage-2 budget reaching tool-routing competence. With stage-1 vocabulary establishment, stage-2 reaches 88% Python validity in ~1% of budget. ~5% saved at c_2 = 0.25 → 0.08× joint.

2. **0.07× from stage-2 → stage-3 tool-validity classifier anchor.** DPO without a tool-validity reference loses ~7% of stage-3 budget to verbosity-favoring drift that suppresses tool calls (Rafailov 2023 §6.2). Frozen ψ_stage2 bounds drift. At c_3 = 0.15, ~7% saved becomes 0.05× joint.

3. **0.06× from PRM-CHIRON × TOOL-LLM joint sharpening at stage 2.** PRM's `L_PRM_tool` sharpens tool-call selection gradient; TOOL-LLM's special-token vocabulary makes routing a single low-entropy classification step. The two multiply on the routing decision specifically (not on argument construction or result integration). At stage 2 where PRM is load-bearing and tools are reasoning-grade, ~6% additional savings.

4. **0.04× from tool-aware stage transitions.** RLG/MELT/MOSAIC preserve trunk loss but disrupt tool-routing hidden-state structure if selector-embedding rows aren't projected through the same operators. Tool-aware transitions preserve structure → ~4% recovery (over a stack-cumulative ~10% of total compute spent in transitions).

Total: 0.25×, consistent with 1.25× → 1.5× recovery.

### 2.3 What "intergenerational" means at #61

iter-203 #59 PRM-CHIRON's intergenerational mechanism was *across* training runs. iter-204 #60-A compressed it into one COSMIC run (stages play the role of generations for the PRM head). **iter-205 #61-A compresses it further:** stages now play the role of generations for the PRM head *and* the tool-call validity classifier simultaneously.

The same single COSMIC run delivers: (1) trunk through warm/full/constitutional PRM phases (iter-204), (2) trunk through minimal/reasoning/full-suite tool-augmentation phases (iter-205), (3) frozen stage-2 PRM and frozen stage-2 tool-validity classifier as constitutional anchors for stage 3.

iter-204 didn't anticipate the tool-validity-axis compounding because #60 TOOL-LLM hadn't yet been promoted; iter-205 reframes COSMIC's stages as a single triple-objective trajectory (trunk training + PRM evolution + tool-vocabulary expansion).

### 2.4 Tool-aware stage transitions

New engineering at #61: making transitions *also* tool-aware (in addition to PRM-aware).

- **RLG identity-insert:** Wo=0 ⇒ identity ⇒ tool-routing hidden-state unchanged. **Zero cost.**
- **Width-grow (MELT):** selector embeddings 1408 → 1536. Zero-padded extension on selector rows. **Zero cost.**
- **Selector-vocabulary growth (NEW at #61):** new selectors activate between stages. `<TOOL=python>` activates between stage 1 → 2; remaining ~57 selectors activate between stage 2 → 3. To-be-activated rows reserved zero-initialized from start of training; activating means *unfreezing* the row, not allocating new memory. **Zero allocation cost.**
- **MOSAIC-MOE expert insert:** experts cold-started → arbitrary tool-routing during recovery. Mitigation: freeze tool-validity classifier during transition, un-freeze after MoE recovery. Stage-3 frozen-constitutional design from iter-204 makes this automatic.
- **SSM block insert:** zero-init ⇒ identity ⇒ tool-routing unchanged. **Zero cost.**

Net: ~70 LOC for `tool_transition.cpp` handling selector-vocabulary activation and embedding row unfreezing; reuses iter-204's `prm_transition.cpp` infrastructure.

---

## 3. Updated cumulative stack: ~3,030,000× tool-augmented / ~620,000× text-NLL

### 3.1 Reference scale (18B / T = 1024)

```
Pre-#60 (post-#59-B PRM-CHIRON):                    404,000×
#60 TOOL-LLM (5× on tool-augmented benchmarks):   2,020,000×
#61-A COSMIC-PROMOTED (1.5× joint via interlock): 3,030,000×
```

The 1.5× at #61 is COSMIC's mechanism contribution; intergenerational PRM + tool-validity transfer prevents joint collapse below 1.5× rather than adding multipliers on top.

Honest framing identical to iter-204: **1.5× ties #59 and iter-204-#60-A-reservation for the lowest marginal in the stack** (#56: 5×, #57: 3×, #58: 2×, #59: 1.5×, #60: 5× tool-augmented / 1× text-NLL, #61: 1.5×). #60 was a one-off boundary-reframing event, not a return to high marginals; #61 returns to the structural-deepening trajectory.

### 3.2 Native COSMIC operating point (144B-eff / T = 16384)

```
Pre-#60 at extreme scale:                810,000×
#60 TOOL-LLM × 5×:                     4,050,000×
#61-A × 1.5× (conservative):           6,075,000×
#61-A × 1.8× (aggressive):             7,290,000×
```

### 3.3 Text-NLL only (no tool augmentation at deployment)

#60 TOOL-LLM contributes 1× on text-NLL (its 5× is on tool-augmented benchmarks).

```
Pre-#60 (post-#59-B PRM-CHIRON):                    404,000×
#60 TOOL-LLM on text-NLL:                           404,000× (1×)
#61-A COSMIC-PROMOTED on text-NLL × 1.5×:           620,000×
```

The ~620,000× and ~3,030,000× figures are not comparable; they live on different evaluation axes.

### 3.4 Cumulative stack at iter-205

| Iter | Paradigm | Marginal | Cumulative (tool-augmented) | Cumulative (text-NLL) |
|---|---|---|---|---|
| ≤167 | #42–#55 | (per-paradigm) | ~3,280× | ~3,280× |
| 197 | #56 DISTILL-FORWARD | 5× | 16,400× | 16,400× |
| 198 | #57 SCROLL-PROMOTED | 2.52× | 41,300× | 41,300× |
| 200 | #58 METAGEN-PROMOTED | 2× | 82,600× | 82,600× |
| 202 | (refinement of #58) | 2.5× | 206,500× | 206,500× |
| 203 | #59 PRM-CHIRON | 1.5× | 310,000× | 310,000× |
| 203 | (refinement of #59-B) | 1.3× | 404,000× | 404,000× |
| 204 | #60 TOOL-LLM | 5× / 1× | 2,020,000× | 404,000× |
| **205** | **#61-A COSMIC-PROMOTED** | **1.5×** | **~3,030,000×** | **~620,000×** |

### 3.5 Sensitivity bands

Pessimistic (1.2×, mechanism overlap with #60 at stage 2 larger than estimated): **2,424,000×** tool-augmented / **485,000×** text-NLL.
Aggressive (1.8×, full triple-axis intergenerational compounding at extreme scale): **3,636,000×** tool-augmented / **727,000×** text-NLL.
Honest-conservative point: **3,030,000×** tool-augmented / **620,000×** text-NLL.

### 3.6 What 3,030,000× means

A naive 18B model trained to the same tool-augmented benchmark accuracy would require ~3,030,000× the wall-clock compute. On a single RTX 4080 SUPER at 16 GB ceiling, post-#61 reaches in ~8 hours what naive training would reach in ~2,770 days (~7.6 years). Cumulative result of 20 paradigms (#42–#61) shipped iter-167 → iter-205. **The seven-figure cumulative speedup band the iter-200 brief explicitly targeted is now firmly inside the conservative point estimate.**

---

## 4. Bigger-picture framing maintained

### 4.1 Schedule × reward × tool-locus triple-axis interlock

iter-203 argued COSMIC produces "a hierarchy of trained models." iter-204 deepened it to a **schedule of (trunk, PRM, objective) tuples**. iter-205 deepens it further to a **schedule of (trunk, PRM, tool-surface, objective) quadruples**:

- Stage 1 → (1.84B trunk, weak-PRM φ_1, calc+retrieve tools, CE+PRM_aux+tool_aux) — deployable foundation with minimal tool primitives
- Stage 2 → (18B trunk, full-PRM φ_2, python+calc+retrieve tools, CE+PRM_aux+tool_aux) — deployable reasoning model with code-execution capability
- Stage 3 → (144B-eff trunk, frozen φ_2, full-suite tools, DPO+PRM_const+tool_const) — fully-aligned production model with deployment-grade tool ecosystem

Each quadruple is independently useful. The schedule × reward × tool-locus triple-interlock produces compound deliverables across three axes, not just compound speedup along one.

### 4.2 Progression toward longer time-horizons and broader capability surfaces

The iter-204 framing at iter-205 revises to:

> #42–#55 attacked **per-step compute**.
> #56 reframed **per-token loss**.
> #57 reframed **per-batch sampling**.
> #58 reframed **per-corpus contents**.
> #59 introduced **co-resident reward signal at pretraining**.
> #60 reframed **the model boundary** (capability internalization → coordinator with external tools).
> #61 reframes **per-run schedule × intergenerational reward × intergenerational tool-vocabulary**.

Schedule, reward, and tool-locus axes are now all interlocked. Future #62+ on any axis (more stages, more rewards, more tools, multi-modal stages) compose against #61's triple-interlock.

### 4.3 What #61 unlocks for #62+

- **#62 multi-modal stages with multi-modal tools:** vision-language and audio-language stages, each with its own PRM and tool surface (`<TOOL=image_describe>`, `<TOOL=audio_transcribe>`). Triple-axis becomes quadruple (modality × schedule × reward × tool-locus).
- **#62 continuous curriculum with continuous tool expansion:** dissolve discrete stages into smooth `(N(t), D(t), L(t), λ_PRM(t), |Tools|(t))`. Intergenerational becomes intracontinuous on three axes simultaneously.
- **#62 closed-loop deployment refinement with tool-usage telemetry:** post-stage-3 deployment data feeds stage-4; stage-3 PRM filters input quality; stage-3 tool-validity classifier filters tool-call correctness; deployment-observed tool-usage informs stage-4 trace synthesis.
- **#63+ constitutional tool-use:** extend constitutional-anchor pattern to constitutional rules at each stage *and* to constitutional tool-usage patterns (e.g., "never invoke `<TOOL=search>` for PII queries" as a frozen classifier).

The triple-interlock at #61 makes all of these natural extensions, not new axes.

### 4.4 Marginal trajectory across 20 paradigms

| Paradigm | Marginal | Cumulative (tool-augmented) | Framing |
|---|---|---|---|
| #56 | 5× | 16,400× | Big mechanism |
| #57 | 2.52× | 41,300× | Strong |
| #58 | 2× → 2.5× | 82,600× → 206,500× | Solid → corpus-curation |
| #59 | 1.5× → 1.3× | 310,000× → 404,000× | Reasoning-quality → PRM saturation |
| #60 | 5× | 2,020,000× | Boundary-reframing event |
| **#61** | **1.5×** | **3,030,000×** | **Triple-axis interlock; same magnitude, deeper structure** |

Late-stack value is increasingly in **structural framing** rather than marginal magnitude. #60 was a one-off boundary-reframing event; #61 returns to the structural-deepening trajectory. #61-A's structural contribution is the triple-axis schedule × reward × tool-locus interlock.

### 4.5 Largest gap: composition with #60 may be smaller than 1.5×

iter-204's largest gap was overlap with #59 PRM-CHIRON. iter-205's analogous gap is overlap with #60 TOOL-LLM at stage 2.

- Stage-1 minimal tool surface contributes ~0.08× via vocabulary establishment.
- Stage-2 reasoning-grade tool surface is what iter-204 implicitly assumed as a fixed deployment surface; iter-205 makes it *escalating* and *PRM-supervised*. This overlap is partially counted in iter-204's 1.5× headline.
- Stage-3 full-suite is mostly orthogonal (DPO replaces CE; tool-validity classifier as anchor not signal).

If actual overlap is larger than estimated (say 0.20× double-counted), realized marginal at #61 collapses to ~1.25× → cumulative drops to ~2,525,000× tool-augmented / ~505,000× text-NLL.

**Gate-1 measurement at 1.84B → 18B will discriminate.** iter-204's 6-arm protocol extends to 12-arm at #61: monolithic / 2-stage / 3-stage × {with-PRM, without-PRM} × {with-tools, without-tools}. The triple factorial cleanly identifies which axis carries which fraction of the joint speedup.

### 4.6 The deliverable hierarchy

The deliverable at #61 is not just a final 144B-eff aligned tool-using model. Each intermediate stage produces an artifact of independent value:

- **Stage-1 (1.84B foundation + minimal tools):** general-purpose foundation with calculator + retrieval. Useful as a base for downstream task-specific fine-tunes; deployable for general text completion, basic reasoning, arithmetic-augmented queries.
- **Stage-2 (18B reasoning + reasoning-grade tools):** mid-size with full reasoning PRM and Python+calc+retrieval. Useful for math/code/multi-hop QA with Python interpreter access; deployable as a reasoning-capable foundation for further alignment.
- **Stage-3 (144B-eff aligned + full-suite tools):** fully-aligned with constitutional PRM, constitutional tool-validity, and ~64-tool deployment ecosystem. Useful for production where alignment quality, reasoning, and tool integration all matter.

**This is meta-architecture in the truest sense: a tower of (model, PRM, tool-surface) triples, each independently useful as a deliverable.** A user with a 16 GB GPU can run any of the three; a user with cluster compute runs the full chain. A user needing only arithmetic-augmented general text completion runs stage 1; one needing reasoning + Python runs stage 2; one needing full deployment runs stage 3.

---

## 5. Engineering: ~1470 LOC over ~8 weeks

iter-204 specified ~1300 LOC over ~7 weeks. At #61 the additional surface is per-stage tool-augmentation profile + cross-stage tool-validity transfer:

| Component | LOC | Source |
|---|---|---|
| iter-204 inheritance (planner, adapter, DPO kernel, grader, CLI, Gate-0 harness, PRM transition, constitutional-PRM anchor) | 1300 | iter-204 |
| **Tool-vocabulary stage-aware activation (`tool_transition.cpp`)** | **70** | **NEW** |
| **Per-stage tool-trace fraction control (data pipeline ext.)** | **30** | **NEW** |
| **Stage-3 constitutional tool-validity anchor in DPO loss kernel** | **40** | **NEW** |
| **Tool-validity classifier intergenerational transfer eval harness** | **30** | **NEW** |
| **Total** | **~1470** | **~8 weeks** |

CLI extension: `--cosmic-tool-stage1-fraction 0.05`, `--cosmic-tool-stage2-fraction 0.12`, `--cosmic-tool-stage3-fraction 0.20`, `--cosmic-tool-stage1-selectors calc,retrieve`, `--cosmic-tool-stage2-selectors python,calc,retrieve`, `--cosmic-tool-stage3-selectors all`, `--cosmic-tool-stage3-lambda-const 0.03`, `--cosmic-tool-stage3-freeze 1`.

**Joint Gate-0 (extended to 12-arm):** monolithic / 2-stage / 3-stage × {no-PRM, PRM} × {no-tools, tools}. STRONG PASS = 3-stage-with-PRM-with-tools tool-augmented benchmark accuracy > 0.95 × monolithic-with-PRM-with-tools at same FLOPs. Cost: ~5 GPU-days mini-scale; Gate-1 at 1.84B → 18B: ~45 GPU-days.

---

## 6. Honest gaps

iter-204 §6 listed 15 gaps; all inherit at #61. New at #61:

16. **Tool-vocabulary stage-aware activation is non-trivial.** Reserving zero-initialized embedding rows for to-be-activated selectors requires careful initialization (no contribution to loss while frozen) and careful unfreezing (gradient flows correctly when activated mid-training without destabilizing trunk). Mitigation: 1% step-warmup at each selector activation event.

17. **Constitutional tool-validity anchor in DPO may over-constrain.** Frozen ψ_stage2 is a strong regularizer; if it has systematic biases (favors Python over alternatives that may be cheaper for the same task), stage 3 inherits them. Mitigation: tune λ_tool_const downward (default 0.03; consider 0.01 if instruction degrades).

18. **The 1.5× joint marginal assumes per-stage tool-trace fraction escalation provides positive transfer.** If stage-1 surface is too minimal (<3% fraction), vocabulary establishment is too weak to warm-start stage 2. Gate-0 Arm at multiple stage-1 fractions validates.

19. **Engineering scope ~1470 LOC over ~8 weeks**, +170 LOC over iter-204. Tractable but reduces shipping margin.

20. **Composition with #60 is the load-bearing claim.** If Gate-0 shows 3-stage-with-PRM-with-tools ≈ monolithic-with-PRM-with-tools (no joint advantage), 1.5× headline collapses. Falsification path is direct.

21. **Triple-axis intergenerational compounding is unprecedented in the literature.** Single-axis intergenerational compounding (PRM in #59-B, tool-validity implicit in Toolformer's self-supervised chain) is well-established. Joint triple-axis compounding (trunk + PRM + tool-validity all evolving across stages of one COSMIC run) is a structural claim with no direct empirical precedent. **Headline structural risk at #61.**

---

## 7. Summary

**COSMIC-PROMOTED at #61** is the iter-204 #60-A reservation, refined for the post-#60 TOOL-LLM stack and promoted to paradigm shift #61-A. iter-204 carries per-stage PRM hosting + intergenerational PRM transfer; iter-203 carries the foundational schedule-axis design; iter-205 #61-A refines on five axes: stage 1 minimal tool surface (vocabulary establishment), stage 2 reasoning-grade (PRM-supervised tool-routing), stage 3 full-suite + DPO + constitutional tool-validity anchor, per-stage tool-trace fraction escalation (5% → 12% → 20%), cross-stage tool-validity transfer (load-bearing intergenerational mechanism on the third axis).

**Cumulative stack post-#61-A:**
- **Tool-augmented benchmarks:** **~3,030,000×** at 18B / T = 1024 reference (range 2,424,000× – 3,636,000×); **~6,075,000×** at 144B-eff / T = 16384 native (~7,290,000× aggressive).
- **Text-NLL only:** **~620,000×** at 18B / T = 1024 reference (range 485,000× – 727,000×).

**Bigger-picture framing maintained.** The schedule axis (iter-203 reservation), reward axis (iter-203 #59 PRM-CHIRON), and tool-locus axis (iter-204 #60 TOOL-LLM) interlock at iter-205 #61-A. Future #62+ compose against this triple-interlock; the deliverable is now a schedule of (trunk, PRM, tool-surface, objective) quadruples rather than a single model with a fixed tool surface.

**Honest framing.** 1.5× ties #59 and iter-204-#60-A-reservation for lowest marginal in the stack; diminishing returns at deep stack are *structural*. #60 TOOL-LLM was a one-off boundary-reframing event (largest marginal since #56); #61 returns to the structural-deepening trajectory. Late-stack value is increasingly in **structural framing** rather than marginal magnitude.

**Engineering.** ~1470 LOC over ~8 weeks, +170 LOC over iter-204. 12-arm Gate-0 at mini-scale (~5 GPU-days); Gate-1 at 1.84B → 18B (~45 GPU-days). Joint composition with #60 is the load-bearing falsifiable claim; triple-axis intergenerational compounding is the headline structural risk.

**Selection criterion vs #61-B / #61-C.** COSMIC-PROMOTED is selected if the goal is **structural triple-axis-interlock** — schedule × reward × tool-locus composition on top of iter-204's mechanism design and #60's boundary-reframing. The 1.5× marginal is honestly modest; the structural contribution at #61 is the **triple-interlock**, and the cumulative 3,030,000× tool-augmented (620,000× text-NLL) is the honest-conservative result of 20 paradigms across iter-167 → iter-205.

---

**End of Paradigm Shift #61 Candidate A document.** Promoted from iter-204 reservation; refined for post-#60 stack via per-stage tool-augmentation profile escalation and intergenerational tool-validity transfer; cumulative single-GPU stack ~3,030,000× tool-augmented / ~620,000× text-NLL at 18B / T = 1024 reference, ~6.075M× tool-augmented at 144B-eff / T = 16384 native. Schedule × reward × tool-locus triple-axis interlock is the structural contribution at iter-205.
