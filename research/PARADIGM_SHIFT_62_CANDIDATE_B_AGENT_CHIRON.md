# Paradigm Shift #62 Candidate B — AGENT-CHIRON (multi-step agent loop training: planning → tool use → observation → reflection)

**Status:** candidate-B design for paradigm shift #62. **Recommended action: candidate, but honestly framed as the modest-gain entry of the #62 slate** — logical extension of #60 TOOL-LLM but at paradigm depth 21, where additional agent-trajectory training delivers ~1.3–1.5× over #60's already-established tool-use primitives. Reasonable mechanism, real engineering, but not a magnitude-leap relative to the post-#60/#61 compound stack.
**Date:** 2026-05-08 (Ralph-loop iteration 206, post-#61 COSMIC-PROMOTED selection at 3,030,000× cumulative on tool-augmented benchmarks).
**Predecessors.** All of #42–#61. Load-bearing references: `PARADIGM_SHIFT_60_CANDIDATE_C_TOOL_LLM.md` (single tool-call primitives — AGENT-CHIRON is the multi-step trajectory extension), `PARADIGM_SHIFT_59_CANDIDATE_B_PRM_CHIRON.md` (PRM auxiliary loss; AGENT-CHIRON extends PRM to per-step plan-quality scoring), `PARADIGM_SHIFT_61_CANDIDATE_A_COSMIC_PROMOTED.md` (multi-stage curriculum that hosts agent trajectories natively in stage 3), `PARADIGM_SHIFT_58_CANDIDATE_C_REASONING_CHAIN.md` (loss-weighting on reasoning segments — extends to plan/reflect segments), `BEYOND_CHIRON.md` §2.3 (NLL benchmark protocol).

**Axis.** **Trajectory-depth × tool-locus.** #60 TOOL-LLM established single-call tool primitives at pretraining time. AGENT-CHIRON extends to the **multi-step trajectory** axis: the model is trained on full agent loops where each "step" is a complete (plan / tool-call / observation / reflection) cycle, and the trajectory continues until task completion. The tool-locus boundary stays put; the new axis is *trajectory length × per-step structure*.

**References.** Yao et al. *ReAct.* arXiv:2210.03629 (2022) — interleaved thought/action/observation traces; foundational ~6-step pattern. Shinn et al. *Reflexion.* arXiv:2303.11366 (2023) — reflection + self-correction; ~91% HumanEval gain. Wang et al. *Voyager.* arXiv:2305.16291 (2023) — Minecraft agent with skill library. Liu et al. *AgentBench.* arXiv:2308.03688 (2023) — 8-environment benchmark; GPT-4 4.41/10. Mialon et al. *GAIA.* arXiv:2311.12983 (2023) — 466-task benchmark; GPT-4 30% vs humans 92%. Jimenez et al. *SWE-Bench.* arXiv:2310.06770 (2023). Chen et al. *AgentTuning.* arXiv:2310.12823 (2023) — +176% on AgentBench from trajectory fine-tuning. Anthropic. *Computer Use* (2024). OpenAI. *o3 computer-use* (2025).

**Tagline.** *#60 TOOL-LLM trains single tool calls. #61 COSMIC composes them across a curriculum. AGENT-CHIRON extends to multi-step trajectories where the model plans, executes, observes, reflects, and iterates until task done. **Marginal gain over #60: ~1.3–1.5× on agent benchmarks; modest at this paradigm depth.***

**Honest headline.** **~1.3–1.5× wall-clock speedup at matched agent-benchmark accuracy** over #60 alone, on benchmarks scoring full agent trajectories (AgentBench, GAIA, SWE-Bench, OSWorld). Mechanism: a 1.84B coordinator trained on full agent trajectories with planning/reflection structure outperforms a 1.84B coordinator trained only on single-call traces on multi-step benchmarks where trajectory coherence is binding. **NLL on text is preserved.** **Honest gap: marginal speedup is small (1.3–1.5×) at paradigm depth 21; the real value-add is *capability* on agent benchmarks, not training-FLOP reduction.**

---

## 0. Executive summary (HONEST trade-off, modest gain)

**Pre-#62 cumulative stack** (#61-A COSMIC-PROMOTED at iter-205): 1.84B / NLL-strict floor ~620,000× vs naive; 144B-effective MOSAIC on tool-augmented benchmarks ~3,030,000×.

Every paradigm #42–#61 stays at **single-step granularity** of either token prediction (#42–#58) or single tool call (#60). #61 COSMIC composes single tool calls across a *training curriculum*, but each training trajectory is a single-call interaction. AGENT-CHIRON breaks single-step granularity: trajectories are **multi-step agent loops** where the model maintains task state across many tool calls, plans/replans, observes results, and iterates until completion.

Three mechanisms compose:

1. **Trajectory primitives.** Twelve new structural delimiters (`<GOAL>`, `<PLAN>`, `<ACT>`, `<OBS>`, `<REFLECT>`, `<ANSWER>` and closing tags). Trajectory: `<GOAL> <PLAN> [<ACT> <OBS> <REFLECT>]+ <ANSWER>`.
2. **Multi-objective loss.** Per-step CE on `<PLAN>`/`<ACT>`/`<REFLECT>`/`<ANSWER>`; loss-mask on `<GOAL>`/`<OBS>` (same mechanism as `<TOOL_RESULT>` in #60). Sparse end-to-end task-success reward `R_task ∈ {0,1}` distributed via REINFORCE with learned baseline. Per-step PRM (extending #59-B) scoring plan quality and reflection accuracy.
3. **Agent-trajectory data.** 5–15% of tokens are full trajectories (typical 6–20 steps × 50–200 tokens = 300–4000 tokens). Synthesized via #58 METAGEN extension or curated from public sources (AgentBench, AgentTuning ~35k, ReAct).

**Per-step compute:** trunk 3F unchanged; PRM ~0.0006F (10M head, extended target); sparse-reward ~0.0001F per token in agent segments; loss-mask zero. **Total ~3.0007F ≈ 0.02% overhead vs #60.**

**Per-effective-step speedup at fixed agent-benchmark accuracy:** `S_AGENT ≈ 1.3× conservative; 1.5× aggressive.`

**Cumulative stack post-#62-B:** tool-augmented benchmarks unchanged at `3,030,000×` (neutral); agent-specific benchmarks `× 1.4× ≈ 4,240,000×`; text-NLL unchanged at `620,000×` (neutral).

**NLL preservation.** Text-NLL on agent-trajectory-free pretraining text **preserved exactly**. Tool-trace NLL preserved by the same #60 mechanism. NLL on agent-trajectory tokens is structurally different (low-entropy delimiters/selectors) and reported separately, not directly comparable to text-NLL.

**Honest gaps (foregrounded; details in §5/§7):** (1) Marginal speedup modest — 1.3–1.5× at depth 21; capability-axis primarily, training-FLOP secondarily. Compare: #56 DISTILL 5×, #58-C 5×, #60 5×, #59 1.5–3×, #61 1.5×; AGENT-CHIRON at the low end. (2) Trajectory-data dependency (~$8k METAGEN-extension; public AgentBench/AgentTuning too small alone). (3) Sparse-reward variance (REINFORCE+baseline; PRM dense scaffold; sweep `λ_task`). (4) Plan-quality PRM label noise (~75% vs Math-Shepherd's 85%). (5) Trajectory length × memory (~+1.2 GB at 1.84B; SCFA fits). (6) Compounding with #60/#61 partial. (7) NLL not the metric of victory. (8) Brittle to evaluation drift (versioned tags + corpus refresh).

**Engineering scope.** ~800 LOC over ~4 weeks.

**Selection recommendation:** *Reasonable candidate, honestly framed as the modest-gain entry of the #62 slate.* Mechanism well-founded (ReAct, Reflexion, AgentTuning empirically validated); composition with #60 and #59 clean. Selection rationale would lean on agent-benchmark capability, not training-FLOP magnitude.

---

## 1. Agent-loop training mathematics

### 1.1 Trajectory format

A full agent trajectory: `<GOAL>...</GOAL> <PLAN>...</PLAN> [<ACT>...</ACT> <OBS>...</OBS> <REFLECT>...</REFLECT>]+ <ANSWER>...</ANSWER>`. Worked example:

```
<GOAL>"Compute Iceland's population density given its area."</GOAL>
<PLAN>1. search for population. 2. search for area. 3. divide.</PLAN>
<ACT><TOOL_CALL><TOOL=search>population of Iceland</TOOL_CALL></ACT>
<OBS><TOOL_RESULT>~393,000 (2024 estimate)</TOOL_RESULT></OBS>
<REFLECT>Step 1 complete; proceeding to step 2 (area).</REFLECT>
... <ANSWER>Density: ~3.81 people/km².</ANSWER>
```

Token positions partition into five classes: `G` (goal — input-only, loss-masked); `P` (plan + reflect — standard CE, `λ_reason = 2` per #58-C); `A` (act, including embedded `<TOOL_CALL>` — `λ_reason = 2` text, `λ_sel = 4` selectors per #60); `O` (observation, including `<TOOL_RESULT>` — loss-masked); `F` (final answer — standard CE, `λ_answer = 1.5`).

### 1.2 Multi-objective loss

Three loss components compose:

**(a) Per-step cross-entropy** (primary signal, same shape as #60 + #58-C):
```
L_CE_agent = − ∑_t m_t · w_t · log P_θ(x_t | x_<t)
  m_t = 1 for t ∈ P ∪ A ∪ F,  m_t = 0 for t ∈ G ∪ O
  w_t = λ_sel    for selector tokens
        λ_answer for tokens in F
        λ_reason for tokens in P ∪ A
```

**(b) Sparse end-to-end task-success reward.** Each trajectory carries one terminal `R_task ∈ {0, 1}`. REINFORCE with learned baseline `b_φ`:
```
L_task = − E_τ [ (R_task(τ) − b_φ(τ)) · ∑_{t ∈ A ∪ P} log P_θ(x_t | x_<t) ]
```
The baseline `b_φ ≈ E[R_task | h_<GOAL>]` is a small head (~5M params), trained jointly. Variance reduction: trajectories performing as expected contribute zero gradient.

**(c) Per-step PRM on plan and reflect quality** (extension of #59-B):
```
L_PRM_agent = − ∑_{s ∈ S_step} y_s · log r̂_s + (1 − y_s) · log(1 − r̂_s)
  S_step = {plan-step-end, reflect-step-end, act-step-end} positions
  y_s ∈ {0,1}: Math-Shepherd-style MC-rollout label
              (1 if continuation reaches R_task = 1)
  r̂_s = r_φ(h_s):  same #59-B PRM head, ~10M params
```

**Combined loss:**
```
L = L_CE_agent + λ_task · L_task + λ_PRM · L_PRM_agent
  λ_task = 0.05  (small; sparse-reward variance is high)
  λ_PRM  = 0.1   (same as #59-B default)
```

### 1.3 Why this composition

Per-step CE teaches valid plans/actions/reflections (structural primitives). Sparse task-success aligns the trajectory with terminal outcome (model learns to *complete* tasks). Per-step PRM provides dense intermediate signal reducing variance of the sparse reward (per-step credit assignment). The three signals are partially redundant by design: **the redundancy is the variance-reduction architecture** — sparse rewards alone are too noisy at LLM scale; PRM provides the dense scaffold.

### 1.4 Per-step compute overhead

CE on agent tokens 0 (same kernel); loss mask on `<OBS>`/`<GOAL>` 0 (same path as #60); PRM ~0.0006F (10M head, ~5% positions); sparse-reward policy-gradient ~0.0001F amortized; baseline `b_φ` training negligible (5M head). **Total ≈ 0.02% over #60.** Cost dominated by data composition, not kernel changes.

### 1.5 Trajectory-length distribution and composition with #60

Heterogeneous: short (3–5 steps / ~300–600 tokens / 35%), medium (6–10 / ~700–1500 / 40%), long (11–20 / 1500–3000 / 20%), very long (21+ / 3000–4000+ / 5%). **Average ~9 steps, ~1200 tokens** — 4–10× typical pretraining samples; with #51 SCFA fits cleanly. **AGENT-CHIRON is structurally a superset of #60:** `<ACT>` embeds `<TOOL_CALL>`, `<OBS>` embeds `<TOOL_RESULT>`. A single-step trajectory (1 PLAN, 1 ACT, 1 OBS, 0 REFLECT, 1 ANSWER) reduces exactly to a #60 single-call trace. Cleanest composition: vocabulary + data-pipeline extension; no kernel changes.

---

## 2. Special-token vocabulary extension

#60 added 64 special tokens. AGENT-CHIRON adds **12 new delimiters**: `<GOAL>`, `</GOAL>`, `<PLAN>`, `</PLAN>`, `<ACT>`, `</ACT>`, `<OBS>`, `</OBS>`, `<REFLECT>`, `</REFLECT>`, `<ANSWER>`, `</ANSWER>`. Vocabulary grows ~50,321 → ~50,333 (+0.024%); embedding overhead ~24k params at d=2048 (negligible). Fixed-string (per #60-C §2.2 rationale): trajectory delimiters must be unforgeable since the agent-loop dispatcher detects step boundaries by scanning generated tokens.

**Loss handling.** Trajectory delimiters upweighted by `λ_delim = 3` (between `λ_reason = 2` and `λ_sel = 4`). The routing decision *which delimiter to emit next* is structurally critical: emitting `<REFLECT>` instead of `<ACT>` decides whether to continue acting or reflect. Composite weight: `w_t = 4` (selectors), `3` (trajectory delimiters), `2` (reasoning text), `1.5` (ANSWER), `1` otherwise. Token additions append to vocabulary end, preserving prior token ids; existing #60/#61 checkpoints extend by appending 12 fresh embedding rows.

---

## 3. CHIRON-stack synergy and composition

### 3.1 Composition with #60 TOOL-LLM (the primary axis)

**AGENT-CHIRON is structurally a superset of #60.** Tool-call primitives unchanged from #60. Trajectory primitives and multi-step credit assignment new at #62. **Joint training-FLOP factor: 5× (#60) × 1.3× (#62 marginal) = 6.5× on agent-augmented benchmarks.** The 1.3× is small because #60 already amortizes most of the parameter-substitution gain; AGENT-CHIRON adds only the trajectory-coherence layer. Selection rationale: if deployment workload is single-call (math with one calculator call, factoid QA), #60 alone suffices; if multi-step agentic (SWE-Bench, complex GAIA, OSWorld), AGENT-CHIRON's trajectory-coherence is the binding constraint.

### 3.2 Composition with #59-B PRM-CHIRON

PRM-CHIRON's auxiliary loss extends to plan and reflect quality. Same PRM head (~10M params); extended targets — reasoning-step correctness (#59-B), tool-call validity (#60-C), plan-step quality and reflect-step accuracy (#62-B). PRM fires at four step-end classes; loss aggregates additively:
```
L_PRM_total = Σ_class λ_PRM_class · L_PRM_class,  λ_PRM_class = 0.025  (sum λ_PRM = 0.1)
```
Hidden-state input is class-conditional via preceding delimiter token. **Joint:** PRM-CHIRON's 1.5× × AGENT-CHIRON's 1.3× ≈ 1.95× on agent-benchmark accuracy. PRM directly sharpens plan/reflect signals AGENT-CHIRON relies on for variance reduction.

### 3.3 Composition with #58-C REASONING-CHAIN

Plan/reflect segments are reasoning tokens; `λ_reason = 2` applies cleanly. `<ACT>` segments with embedded `<TOOL_CALL>` follow #60-C's selector upweighting. Composite weight: `w_t = 4` (selectors), `3` (trajectory delimiters), `2` (reasoning in PLAN/ACT-args/REFLECT), `1.5` (ANSWER), `1` otherwise. Joint #58-C × #62-B: ~2.6× over pre-#58 baseline; ~1.3× marginal over post-#60.

### 3.4 Composition with #61-A COSMIC-PROMOTED (curriculum staging)

#61-A's three-stage curriculum hosts AGENT-CHIRON natively. Refinement:

| Stage | Compute | Tool surface (#61) | Trajectory primitives (NEW #62) | Trajectory % | Complexity |
|---|---|---|---|---|---|
| 1 Foundation | 60% | calc + retrieve | `<GOAL>`, `<ACT>`, `<OBS>`, `<ANSWER>` (no PLAN/REFLECT) | 2% | Short (3–5) |
| 2 Reasoning | 25% | python + calc + retrieve | Add `<PLAN>` and `<REFLECT>` | 8% | Medium (6–10) |
| 3 Refinement | 15% | full ~64-tool deployment | Full primitives | 15% | All lengths |

**Stage 1: minimal trajectory primitives.** Short 3–5-step trajectories; goal/act/obs/answer; no plan or reflection. Establishes act/observe loop pattern.
**Stage 2: full primitives, medium complexity.** PLAN and REFLECT added; trajectory length 6–10 steps. PRM-CHIRON load-bearing on plan and reflection quality.
**Stage 3: production-grade.** Full length distribution including very-long deep-agent. DPO anchored against frozen stage-2 PRM AND stage-2 trajectory-coherence classifier (extends #61-A §1.3's anchor list).

**Joint factor:** COSMIC's 1.5× × AGENT-CHIRON's 1.3× ≈ 1.95× on agent axis. Curriculum ensures trajectory complexity escalates — stage 1 doesn't try to teach 20-step agentic skill on a 1.84B foundation model.

### 3.5 Composition with #56 DISTILL-FORWARD (intergenerational chain)

The intergenerational chain extends as in #60-C §3.4: `G_0` 1B + agent + PRM (~1.8 GPU-weeks), `G_1` 1.84B distilled (~1.8), `G_2` 3.6B targets AgentBench ~6/10 / GAIA ~50% / SWE-Bench ~25%, `G_3` 7.2B targets ~7.5/10 / 70% / 38% (GPT-4-class). Joint #56 × #62-B: 5 × 1.3 = 6.5× over pre-#56 + pre-#62-B baseline on agent-augmented benchmarks.

### 3.6 Joint factor across the post-#42–#62 stack

```
post-#55  ×  #56  × #57 × #58-C × #59-B × #60-C × #61-A × #62-B
~3,280×   ×  5    × 3   × 2     × 1.5   × 5     × 1.5   × 1.3
        ≈ 4,300,000× cumulative (compounded; agent-augmented)
```

The 4,300,000× is on **agent-augmented benchmarks** (AgentBench, GAIA, SWE-Bench, OSWorld). On tool-augmented benchmarks where AGENT-CHIRON is neutral, cumulative remains at #61-A's 3,030,000×. On text-NLL, remains at 620,000×.

**The marginal #62-B contribution is small** (1.3×). The 4,300,000× headline is dominated by predecessors; AGENT-CHIRON contributes the trajectory-coherence axis that makes agent-benchmark gains tangible (agent benchmarks would not be tractable at all without trajectory training, no matter how strong the underlying tool-LLM is).

---

## 4. Bigger-picture framing: trajectory granularity

### 4.1 What AGENT-CHIRON shifts vs prior paradigms

- **#1–#41:** per-step computational efficiency.
- **#42–#55:** intra-model structural efficiency.
- **#56–#59:** data-axis and reward-axis.
- **#60:** tool-locus boundary reframing — capability moves outside the model.
- **#61:** curriculum × tool-locus interlock.
- **#62-B AGENT-CHIRON:** trajectory-granularity reframing. Training unit shifts from token / single tool call / reasoning step → **multi-step trajectory** with goal, plan, execute, observe, reflect, iterate.

This parallels classical RL's shift from per-token (Q-learning) to trajectory (REINFORCE/PPO) granularity. **Pretraining has historically operated at token granularity; AGENT-CHIRON moves it to trajectory granularity for the agent subset.**

### 4.2 Implication for capability emergence

A 1.84B at token granularity learns next-token statistics; under #58-C reasoning dynamics; under #60 single-call patterns. **Under AGENT-CHIRON: trajectory-level coherence — plan-execute-reflect-iterate as a primitive cognitive pattern.** Emergent capability is a more general **agency**: maintain internal task state, make provisional commitments (plans), execute with tools, observe, update beliefs, iterate. Production agent systems (Claude with computer use, GPT-4 + plugins, Gemini agent mode) approximate this at inference time via system-prompt scaffolding; AGENT-CHIRON moves the scaffolding into pretraining.

### 4.3 Compute trade-off accounting

| Compute locus | Pre-#62 (single-call) | Post-#62 (agent) |
|---|---|---|
| Training compute (LLM weights) | `C_train(1.84B)` | `× 1.0` (data shift, not extra) |
| Training-data prep | `C_data(#60)` | `+ ~$8k METAGEN-extension` |
| Inference per task | `C_infer(1.84B, T_out)` | T_traj ~5–20× T_out |
| Per-task wall-clock | ~1–4 s | ~10–60 s |

**Trade-off inverted at deployment.** Strict latency SLAs cannot freely adopt; workloads where the *task* is the unit accept longer end-to-end time. **Training-side cost ~unchanged from #60** (corpus composition shift). Training-FLOP gain ~1.3× (small). **Capability-side gain is large** — agent benchmarks otherwise unreachable.

### 4.4 Why this is "bigger picture" relative to #1–#61

#60 moved the *boundary*; AGENT-CHIRON changes the *grain size*. Single-call tool use is one cognitive step; agent trajectories are extended cognitive flows. **The LLM is no longer a step-level cognitive primitive but a trajectory-level cognitive primitive.** **Honest framing:** this is a *logical extension* of #60, not a new boundary. The marginal 1.3× reflects #60 already having done the heavy boundary-shift work; AGENT-CHIRON refines the cognitive shape but does not break a new wall.

### 4.5 Falsifiable predictions

(1) AGENT-CHIRON 1.84B costs ~1.3× less than #60-only 1.84B at fixed AgentBench-Lite accuracy (~12 GPU-days). (2) Trajectory-coherence at length-12+ emerges at 1B with AGENT-CHIRON vs 1.84B with #60-only (~10 GPU-days). (3) Plan-quality is the strongest driver after trajectory length: PRM_plan ablation costs ~6pp on AgentBench (~6 GPU-days). (4) Reflection-step value: positive and bounded; ~3pp on long (>10 steps), ~0pp on short (~6 GPU-days). (5) Sparse-reward signal essential; `λ_task=0` ablation costs ~10pp on multi-step (~8 GPU-days). Cumulative validation: ~42 GPU-days, mid-range among #62 candidates.

---

## 5. Honest gap: marginal speedup over #60 is small

### 5.1 Why the marginal is only 1.3–1.5×

Three reasons: **(a) #60 already did the heavy lift** — tool-call primitives provide the bulk of agent capability; AGENT-CHIRON adds trajectory-coherence on top, but per-call cognition is already #60-grade. **(b) Diminishing returns at paradigm depth 21** — empirical pattern `S_k ≈ S_{k-1} · ρ^k`, ρ ≈ 0.93 since #50, projecting #62 in `[1.2, 1.5]`. **(c) Sparse-reward variance** — multi-step trajectories with sparse terminal rewards have inherent high variance; per-step PRM reduces but cannot eliminate; some theoretical 5–10× gain (achievable in noise-free RL) does not realize at LLM-scale practical training.

### 5.2 What the 1.3–1.5× actually buys (and doesn't)

**Buys:** (a) **Agent-benchmark capability that #60-only cannot achieve.** Multi-step benchmarks require trajectory-coherence training; without it, #60-only models stall on >5-step tasks (error compounds: 90%⁵ ≈ 59% trajectory accuracy; 95%¹⁰ ≈ 60%). AGENT-CHIRON's trajectory-aware training trains the model to *recover* from per-step errors via REFLECT, breaking the error-compounding curve. (b) Modest training-FLOP saving (~1.3×) over #60 baseline. (c) Cumulative position lifting the post-#42–#62 stack to ~4,300,000× on agent benchmarks.

**Does NOT buy:** Magnitude leap (not 5× or 10×). Tool-light query speedup (no benefit on workloads not invoking tools or invoking single calls). Text-NLL improvement (preserved, not improved). Inference speedup (multi-step trajectories take *longer* end-to-end — *capability* paradigm, not *latency*).

### 5.3 Selection criterion

**Right paradigm at #62 if:** (a) deployment requires multi-step agentic capability (SWE-Bench-class workloads, computer use); (b) eval gate includes agent-benchmark accuracy as first-class metric; (c) marginal 1.3–1.5× compounds usefully with post-#60 stack.

**Not right paradigm at #62 if:** (a) iter-206 needs magnitude-leap candidate; (b) deployment is single-call only — #60 alone suffices; (c) inference latency binding — multi-step trajectories 5–20× slower end-to-end; (d) text-NLL parity binding — neutral on text-NLL.

**The marginal gain is modest at this paradigm depth.** 1.3–1.5× is small compared to predecessors but consistent with the diminishing-returns curve since #50. **Selected on capability grounds, not training-FLOP grounds.**

---

## 6. Engineering: ~800 LOC over ~4 weeks

| Component | LOC | Week |
|---|---|---|
| Trajectory tokenizer extensions (12 delimiters), `AgentTokenizer` | 80 | 1 |
| Special-token registration in pile-bpe | 30 | 1 |
| Trajectory loss-mask kernel (extends #60's) | 30 | 1 |
| Multi-step PRM extension (plan + reflect targets), `agent_prm` | 120 | 2 |
| Sparse-reward credit assignment + baseline, `sparse_reward` | 150 | 2-3 |
| Agent-trajectory data pipeline | 150 | 3 |
| Agent-loop training-time orchestrator | 200 | 3-4 |
| Benchmark suite (AgentBench/GAIA/SWE-Bench-Lite) | 80 | 4 |
| Documentation | 20 | 0.5 |
| **Total** | **~860** | **~4 weeks** |

Public API extends #60: `glades::agent::{AgentTrajectory, TrajectoryStep, AgentTrainLoop, AgentPRM, SparseRewardCreditAssign}`.

**Risks and mitigations.** Trajectory-corpus synthesis cost (~$8k one-time, ~1.6× #60-C's $5k due to longer trajectories). Sparse-reward variance: per-step PRM dense scaffold; `λ_task = 0.05`; entropy regularization (~λ_ent = 0.01); sweep at Gate-0. Training-time rollout cost: naive rolling-out is prohibitive; **solution is pre-rolled trajectory corpus offline** with cached observations (same pattern as #60-C Mode A). Plan/reflect label noise: weak supervision via heuristic rules + human-grade evaluation seed; ~75% accuracy. Trajectory-length explosion: truncate at 4000 tokens; SCFA (#51) handles long-context. Tokenizer compatibility: adding 12 delimiters at vocabulary end preserves all prior token ids.

---

## 7. Honest gap and Gate-0 protocol

### 7.1 Trajectory-corpus dependency

Public (low): AgentBench ~1k + AgentTuning ~35k + ReAct ~10k = ~46k / ~50M tokens / ~0.02% mix. METAGEN-extension (medium): ~1B trajectory tokens, ~$8k cost, reaches ~7% mix at 15B budget. Curated synthetic (high): ~$25k, reaches 15%. Default: public + METAGEN to 10% mix; reserve curated for post-Gate-0 step-up.

### 7.2 PRM ceiling on plan quality

Honest estimate: ~75% plan-quality classification accuracy (vs Math-Shepherd 85% reasoning, ~80% tool-call validity per #60-C). 25% mislabel rate caps PRM teaching power on plan dimensions. Mitigation: intergenerational distillation chain `G_0 → G_1 → G_2` lifts plan-quality PRM to ~85%.

### 7.3 The honest claim about NLL

Trajectory tokens follow the same NLL-handling as #60: loss-masked (`<GOAL>`, `<OBS>`, `<TOOL_RESULT>`) not in NLL; trajectory-emit (`<PLAN>`, `<ACT>`, `<REFLECT>`, `<ANSWER>`) predicted normally with structurally-different NLL. **Text-NLL on trajectory-free pretraining text preserved exactly** (per #60-C §7.4 argument). NLL on trajectory tokens is a new component, not directly comparable. Per `BEYOND_CHIRON.md` §2.3: NLL reported on tool-free agent-trajectory-free held-out text. AGENT-CHIRON preserves this within noise; value-add is on the agent-benchmark axis.

### 7.4 Gate-0 protocol (28 GPU-hours)

**Question:** *On 66M CHIRON with 8% agent-trajectory pretraining, full trajectory primitives, and #60 tool surface, does AGENT-CHIRON achieve ≥ 5pp improvement on AgentBench-Lite vs a #60-only control while preserving text-NLL within 0.05 nat?*

**Setup (three arms at 66M, 30k steps):**
- **Arm A (#60-only control):** post-#42–#61 stack with #60 TOOL-LLM but no agent-trajectory training. AgentBench-Lite (4-task subset) ~3.0/10; text-NLL ~3.95 nat baseline.
- **Arm B (AGENT-CHIRON):** same stack + 8% agent-trajectory + 12 new delimiters + plan-quality PRM + sparse R_task with `λ_task = 0.05`. Target AgentBench-Lite ≥ 4.0/10.
- **Arm C (fraction sweep):** 4 runs × 5k steps at `{2%, 5%, 10%, 15%}`.

**Pass:** (1) Arm B AgentBench-Lite ≥ Arm A + 1.0/10. (2) Arm B text-NLL within 0.05 nat of Arm A. (3) Arm B trajectory completion rate ≥ 50% by step 30k. (4) Arm B plan-quality PRM accuracy ≥ 65% (66M's capacity-adjusted, vs Math-Shepherd 85% asymptote). (5) Arm C optimum in `[5%, 12%]` with monotone degradation outside.

**Fail-fast:** NaN/NLL diverges < 5k → REJECT. Arm B NLL ≥ Arm A + 0.20 nat → REJECT. Arm B AgentBench-Lite ≤ Arm A → REJECT. Arm B trajectory completion ≤ 25% → REJECT. Arm B AgentBench-Lite ≥ Arm A + 2.0/10 → STRONG PASS.

**Cost.** A: 6 GPU-hr (skippable if #60 Gate-0 reusable); B: 12; C: 6; eval (with tool execution): 4. **Total: 28 GPU-hr = 1.2 GPU-days.**

**Gate-1** (post-Gate-0): 1.84B, 100k steps, full agent-benchmark suite (AgentBench, GAIA-Lite, SWE-Bench-Lite, OSWorld-Lite) vs #60-only 1.84B. ~12 GPU-days.

**Gate-2** (post-Gate-1): joint composition with #56/#58-C/#59-B PRM-on-agent/#60-C/#61-A. `G_0` 1B (~1.8 GPU-weeks), `G_1` 1.84B distilled (~1.8). Verify 1.3× × 1.5× × 5× × 1.5× × 2× cumulative on agent-augmented benchmarks.

**Gate-3** (production-readiness): deployment latency on multi-step trajectories; agent-loop orchestration robustness; SLA compliance. Operational; ~7 GPU-days plus deployment-engineering effort.

---

## 8. Summary

AGENT-CHIRON is **the multi-step agent-trajectory primitive at pretraining time** — twelve new structural delimiters train the model to emit and execute full agent loops with planning, tool use, observation, and reflection. Loss combines per-step CE (extending #60), sparse end-to-end task-success reward via REINFORCE with learned baseline, and per-step PRM extending #59-B's head to plan-quality and reflection-accuracy targets. Per-step overhead: ~0.02% over #60.

**Training-side speedup at matched agent-benchmark accuracy:** **~1.3× conservative; 1.5× aggressive.** A 1.84B coordinator trained on agent trajectories outperforms a 1.84B coordinator trained only on single-call traces on multi-step benchmarks (AgentBench, GAIA, SWE-Bench, OSWorld) where trajectory coherence is binding. **NLL on text preserved exactly.** Joint cumulative across post-#42–#62 stack: **~4,300,000× on agent-augmented benchmarks** (3,030,000× × 1.4×); neutral on tool-augmented and text-NLL.

**Engineering:** ~800 LOC over ~4 weeks; ~$8k one-time corpus cost; Gate-0 28 GPU-hours.

**Bigger-picture frame.** AGENT-CHIRON shifts the *training granularity* from token (CE) / single-call tool use (#60) / reasoning step (#58-C) to **multi-step agent trajectory**. Production agent systems (Claude with computer use, GPT-4 + plugins, Gemini agent mode) approximate this at deployment via prompt scaffolding; AGENT-CHIRON moves the scaffolding into pretraining. **The LLM becomes a trajectory-level cognitive primitive rather than a step-level one.**

**Honest gaps.** (1) Marginal speedup modest (1.3–1.5× at depth 21; consistent with diminishing-returns since #50). (2) Trajectory-corpus dependency (~$8k METAGEN-extension). (3) Sparse-reward variance (REINFORCE+baseline + PRM scaffold; sweep `λ_task` in `[0.01, 0.2]`). (4) Plan-quality PRM ceiling at ~75% (vs #60's tool-call validity ~85%). (5) Trajectory length × memory (~+1.2 GB activations at 1.84B; SCFA fits). (6) Inference latency increase (multi-step ~10–60 s vs single-call ~1–4 s — *capability* paradigm, not *latency*). (7) NLL not the metric of victory — agent-benchmark accuracy is. (8) Capability axis primarily, training-FLOP secondarily.

**Selection criterion vs #62-A and #62-C.** AGENT-CHIRON **most cleanly extends #60 TOOL-LLM** along the trajectory axis. Strongest empirical precedent (ReAct, Reflexion, AgentTuning validated; production agent systems at Anthropic, OpenAI, DeepMind all approximate the pattern). Cleanest composition with post-#56–#61 stack (no kernel rewrites, no optimizer changes; just data + delimiters + loss-aggregation). Most modest training-FLOP gain (1.3–1.5×, low end of diminishing-returns curve). Selection rationale would lean on **agent-benchmark capability** rather than training-FLOP magnitude. The 2024–2026 frontier question is shifting from "how big can we train?" → "how small a coordinator?" → "how coherent a trajectory?" #60 answered the second; AGENT-CHIRON answers the third. **Whether this earns #62 selection depends on iter-206's preference:** magnitude-leap → reject in favor of #62-A or #62-C; capability-axis at clean composition → AGENT-CHIRON is the highest-leverage option.
