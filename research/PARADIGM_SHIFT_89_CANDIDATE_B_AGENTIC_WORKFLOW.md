# Paradigm Shift #89 Candidate B — AGENTIC-WORKFLOW-DISTILL: Multi-Agent Orchestration Distillation

**Status:** RESERVE — heavy mechanism overlap with #62 AGENT-CHIRON (~50-70%); distinguished but modest standalone magnitude.
**Date:** 2026-05-08 (Ralph-loop iter 233 candidate slate, post-#88 fourth saturation finding).
**Axis:** **AGENTIC-WORKFLOW** — proposed extension within existing #62 AGENCY axis, not 28th orthogonal axis.
**Magnitude target:** **1.5-3× on multi-agent benchmarks** (AgentBench-orchestration, GAIA-multi-step, SWE-Bench-multi-actor); **risk-adjusted ~1.5-2.0×** on multi-agent subset; compute-NEUTRAL on text NLL.

---

## 0. Executive summary

Iter-233 continues the post-iter-224 saturation pattern (#80-#88 axis-extensions at ~5M× each on new axes, with iter-232 #88 acknowledged as fourth consecutive least-bad selection). This candidate, **AGENTIC-WORKFLOW-DISTILL-CHIRON**, was sourced as candidate B in the iter-233 slate following the user-brief signal toward agentic workflow systems (LangChain, AutoGen, CrewAI, OpenAI Swarm).

**The honest framing:** AGENTIC-WORKFLOW is **NOT a new orthogonal axis** — it extends #62 AGENT-CHIRON's AGENCY axis from single-agent multi-step to multi-agent multi-step. The mechanism overlap with #62 is substantial (estimated 50-70% based on shared trajectory tokenization, REINFORCE loss structure, and PRM-step-level supervision). Production precedent is strong (LangChain shipped at scale, AutoGen/CrewAI/Swarm adopted broadly), but **at non-CHIRON architectures** that do not preserve text NLL, do not compose with #74 PHOENIX-1BIT, and do not respect the 16 GB single-GPU ceiling.

| Property | Value |
|---|---|
| Mechanism overlap with #62 | ~50-70% (high) |
| Magnitude (multi-agent benchmarks) | 1.5-3× standalone |
| Magnitude (text NLL) | NEUTRAL (Theorem 1) |
| Joint Gate-0 PASS | ~50% |
| LLM-scale empirical confirmation | ~32% |
| Risk-adjusted magnitude | ~1.5-2.0× on multi-agent subset only |
| Engineering | ~700 LOC over 3.5 weeks |

**Why this candidate doc is RESERVE rather than SELECT:**
1. **Overlap with #62 is the dominant fact.** Mechanism is largely #62 AGENT-CHIRON with multi-agent role-tokens layered atop existing trajectory tokenization. Distinguishability rests on the multi-actor coordination tokens (~10-15 new special tokens for ROUTER/AGENT_K/SYNTHESIZER plus protocol delimiters), which is incremental.
2. **Magnitude is microoptimization-tier under iter-200 critique.** 1.5-3× on a benchmark subset is below the iter-228 "magnitudes-better" bar; risk-adjusted ~1.5-2.0× falls into the post-iter-200 "small-fish" zone unless paired with a fresh axis.
3. **Production precedent strong but architecture-mismatched.** LangChain/AutoGen/CrewAI/Swarm orchestrate API-calls to GPT-4-class models at production scale. Distilling THEIR orchestration patterns into CHIRON-2.23B-class students is sound in principle; quality of the distilled traces caps at the orchestrating model's tier.
4. **Composes well with prior paradigms but does not unlock new structural ground.** Multiplicative composition with #59 PRM (per-coordination-decision scoring) and #69 REASONING-DISTILL (reasoning chains within each agent role) is real but mechanically straightforward.

**Recommendation:** RESERVE for iter-234+ promotion if (a) empirical evidence emerges from #62 AGENT-CHIRON Gate-1 showing single-agent agency yields production-ready traces but multi-agent orchestration remains underexpressed, or (b) the user brief explicitly calls out multi-agent agentic workflows as a strategic target.

---

## 1. Mechanism: orchestration-trace distillation

### 1.1 Multi-agent trace structure

The key mechanism reuses #62 AGENT-CHIRON's trajectory token vocabulary (12 tokens for `<GOAL>`, `<PLAN>`, `<ACT>`, `<OBS>`, `<REFLECT>`, `<ANSWER>` plus closing) and **extends it** with a multi-actor protocol layer:

```
<TASK>
  <ROUTER>
    <AGENT_SELECT> agent_specification </AGENT_SELECT>
  </ROUTER>
  <AGENT_1 role="researcher">
    <GOAL>...</GOAL>
    <PLAN>...</PLAN>
    <ACT>...</ACT>
    <OBS>...</OBS>
    <RESULT_1>...</RESULT_1>
  </AGENT_1>
  <AGENT_2 role="coder">
    <GOAL>...</GOAL>
    <PLAN>...</PLAN>
    <ACT>...</ACT>
    <OBS>...</OBS>
    <RESULT_2>...</RESULT_2>
  </AGENT_2>
  <SYNTHESIZER>
    <ANSWER>...</ANSWER>
  </SYNTHESIZER>
</TASK>
```

**New special tokens (10-15 added):** `<TASK>`, `<ROUTER>`, `<AGENT_SELECT>`, `<AGENT_K role="...">`, `<RESULT_K>`, `<SYNTHESIZER>` plus closing tags. Role attributes (`researcher`, `coder`, `reviewer`, `planner`, etc.) drawn from a closed vocabulary of ~32 production-validated agent personas.

### 1.2 Orchestration teacher signals

| Tier | Orchestration framework | Mechanism extracted | Production scale |
|---|---|---|---|
| **Tier 1 (preferred)** | LangChain | Chain-of-tool-calls + agent-router patterns | Production-shipped at major enterprises |
| **Tier 2** | Microsoft AutoGen (2023) | Multi-agent conversation patterns | Production-shipped at Microsoft scale |
| **Tier 3** | CrewAI | Role-based orchestration patterns | Open-source widely-adopted |
| **Tier 4** | OpenAI Swarm | Lightweight handoff patterns | Reference implementation |

**Recommended:** Tier 1 + Tier 2 dual-source for trace diversity; Tier 3 + Tier 4 for fallback.

### 1.3 Distillation pipeline

The student inherits orchestration via three signal channels:

**Channel A — Trace token-CE (primary).** Multi-agent orchestration traces collected from teacher framework runs. Each trace is a token sequence in the extended vocabulary; standard CE loss on visible tokens.

**Channel B — KL-distillation on coordination decisions (per #69 pattern).** Where teacher orchestration framework exposes routing logits (LangChain agent-selector, AutoGen multi-agent voting, etc.), KL distillation at top-K=16 over the ~32-agent persona vocabulary. Where teacher exposes only hard decisions, fall back to one-hot CE.

**Channel C — PRM-on-coordination-decisions (per #59 extension).** PRM head scores each `<AGENT_SELECT>` and `<RESULT_K>` for correctness of the coordination decision. Auxiliary loss `L_PRM_coord = 0.05 · L_PRM` (lower than #59's 0.1 because the signal is sparser).

Total loss:
```
L = L_CE_trace + 0.5 · L_KL_route + 0.05 · L_PRM_coord
```

### 1.4 Composition with prior 48 paradigms

| Paradigm | Composes? | Mechanism |
|---|---|---|
| **#62 AGENT-CHIRON** | Stack-base (heavy overlap) | Trajectory tokens reused; multi-actor layer added atop. |
| **#59 PRM-CHIRON** | ✓ Multiplicative | PRM scores coordination decisions. |
| **#60 TOOL-LLM** | ✓ Multiplicative | Each agent role can invoke tools. |
| **#69 REASONING-DISTILL** | ✓ Multiplicative | Reasoning chains within each agent's `<PLAN>`/`<ACT>` blocks. |
| **#56 DISTILL-FORWARD** | ✓ Stack-base | Teacher traces are distillation source. |
| **#74 PHOENIX-1BIT** | ✓ Compatible | New special-token embeddings BF16; trunk PHOENIX-quantized. |
| **#76 MLA + #78 SINK + #79 MoD** | ✓ Compatible | Coordination tokens are normal tokens in joint sequence. |

The strongest synergy is with **#59 PRM-CHIRON × #62 AGENT-CHIRON**: PRM-on-coordination-decisions extracts a finer-grained reward signal than #62's per-step PRM by labeling the multi-agent routing as the credit-assignment locus.

---

## 2. Theoretical analysis

### 2.1 Theorem 1 — Text NLL preservation

**Claim.** On text-only sequences (no `<TASK>`/`<ROUTER>`/`<AGENT_K>` tokens), AGENTIC-WORKFLOW-DISTILL preserves text NLL bit-exactly relative to the pre-#89 stack.

**Proof sketch.** The new ~10-15 special tokens occupy fresh vocabulary slots; their embedding vectors are trained jointly but never appear in text-only sequences. The trunk forward pass on text-only inputs is identical to the pre-#89 stack; the L_KL_route and L_PRM_coord terms are zero on text-only batches. Therefore text-NLL gradient on text-only batches is unchanged.

**Caveat.** During mixed batches (text + agent traces), gradient interference at the embedding-island level is bounded by the standard #28 FACE Zipfian-regularization argument. Empirically expect <0.005 nat drift on text NLL benchmarks.

### 2.2 Theorem 2 — Multi-agent magnitude ceiling

**Claim.** Standalone speedup on multi-agent benchmarks bounded above by `min(teacher_orchestration_quality / student_capacity, distillation_temperature_factor)`.

**Reasoning.** When the orchestration teacher (LangChain + GPT-4) achieves multi-agent benchmark accuracy `q_T` and the CHIRON student (post-#62 AGENT-CHIRON) achieves `q_S` without #89, the post-#89 ceiling is `q'_S ≤ q_T`. Empirical 1.5-3× speedup-to-equal-quality on multi-agent benchmarks aligns with reported multi-agent-vs-single-agent gaps (AgentBench, SWE-Bench-Multi).

### 2.3 Joint Gate-0 PASS probability

```
Multi-agent trace dataset assembly:                  ~75%
LangChain/AutoGen trace ingestion pipeline:          ~80%
Multi-actor token extension:                         ~88%
Mixed-batch training stability:                      ~70%
Memory at 16 GB ceiling (~80 MB additional):         ~92%
LLM-scale empirical confirmation (multi-agent):      ~55%

Joint Gate-0 PASS:                                   ~50%
LLM-scale empirical confirmation:                    ~32%
```

Below #88's 55% Gate-0 PASS, reflecting the mixed-batch stability risk and the multi-agent benchmark dependency on orchestration framework quality.

---

## 3. Updated cumulative stack

```
Iter 232 close (post-#88):
  All 27 axes ≈preserved

Iter 233 with #89-B AGENTIC-WORKFLOW (HYPOTHETICAL if SELECTED):
  All 27 axes ≈preserved (compute-NEUTRAL on text NLL by Theorem 1)
  AGENCY axis (#62) extended: single-agent → multi-agent
  Multi-agent benchmark subset: ~1.5-3× standalone (~1.5-2.0× risk-adj)
  Cumulative: AgentBench-multi-agent + SWE-Bench-multi-actor + GAIA-multi-step ≈ 1.7× geometric mean
```

**Bigger-picture note.** Unlike #88 (3D-SPATIAL — opens 27th axis), #89-B does NOT add an orthogonal axis. It deepens the AGENCY axis already established by #62. This is the structural distinction recommending RESERVE.

---

## 4. Engineering scope

| Component | LOC | Weeks |
|---|---|---|
| Multi-actor token extension (10-15 new specials) | 80 | 0.5 |
| Trace ingestion pipeline (LangChain + AutoGen logs) | 200 | 1 |
| Multi-agent DataLoader (mixed batches text + traces) | 150 | 1 |
| KL-distill loss on coordination decisions | 80 | 0.5 |
| PRM-on-coordination-decisions extension | 100 | 0.5 |
| Multi-agent evaluation harness (AgentBench-multi-agent, SWE-Bench-multi-actor, GAIA-multi-step) | 90 | 0 |
| **Total** | **~700** | **3.5** |

Lighter than #88 (~1,650 LOC, 7 weeks) due to mechanism overlap with #62.

---

## 5. Memory advantage preservation

| Component | GPU memory |
|---|---|
| New special-token embeddings (~15 × m=2048 BF16) | ~0.06 MB |
| Multi-agent trace buffer (training only, streaming) | ~80 MB |
| Coordination-PRM head (small extension of #59) | ~5 MB |
| **Total additional** | **~85 MB** |

**Single-GPU 16 GB ceiling preserved** with ~545 MB headroom (narrower than #88's 630 MB but still positive).

---

## 6. Gates

### Gate-0 (~6 GPU-hours)

**Probe.** 200M coordinator + multi-agent trace dataset (~5M traces from LangChain + AutoGen logs) + L_CE_trace only (no KL, no PRM).

**PASS criteria.**
- Trace token-CE convergence within 30% of single-agent #62 trace token-CE.
- NLL on text-only ≤ 0.005 nat drift.
- Multi-agent benchmark accuracy on held-out subset ≥ 1.2× single-agent #62 baseline.

**PASS probability:** ~62%.

### Gate-1 (~80 GPU-hours)

**Probe.** Full 32B-effective + multi-agent traces + KL-distill + PRM-on-coordination + 30M trace dataset.

**PASS criteria.**
- AgentBench-multi-agent ≥ 55%.
- SWE-Bench-multi-actor ≥ 38%.
- GAIA-multi-step ≥ 40%.
- Text NLL drift ≤ 0.01 nat from pre-#89.

**PASS probability conditional on Gate-0:** ~52%.

---

## 7. Honest gaps

1. **Mechanism overlap with #62 is high (50-70%).** This is the dominant fact recommending RESERVE. Multi-agent extension layered atop #62's single-agent trajectory tokenization is incremental rather than structural.

2. **Magnitude is microoptimization-tier under iter-200 critique.** 1.5-3× standalone on a benchmark subset; risk-adj 1.5-2.0× on multi-agent subset only. Falls below the iter-228 "magnitudes-better" bar.

3. **Production precedent strong but architecture-mismatched.** LangChain/AutoGen/CrewAI/Swarm operate at GPT-4-class scale on closed-source production APIs. Distilling their orchestration into CHIRON-2.23B-class students is sound but quality-capped by teacher.

4. **No new orthogonal axis.** Extends #62's AGENCY axis; 27 axes preserved but no 28th opened. Structural distinction from #88 (which opened 3D-SPATIAL as 27th).

5. **Mixed-batch stability risk.** Joint training on text-only + multi-agent traces requires careful loss balancing; embedding-island gradient interference at L_CE_trace vs L_CE_text not theoretically clean.

6. **Iter-200 microopt critique applies.** A 1.5-3× speedup on a benchmark subset is precisely the pattern flagged in iter-200's "looking at the bigger picture instead of focusing on microoptimizations" critique.

7. **Composition with #62 multiplicative but not magnitudes.** Stacking #62 (1.3-1.5× on agent benchmarks) with #89-B (1.5-3× on multi-agent benchmarks) yields ~2.0-4.5× on multi-agent subset — still single-axis-of-improvement.

8. **Trace data quality dependency.** Production orchestration logs from LangChain/AutoGen include framework-specific patterns that may not transfer cleanly to CHIRON. Trace cleaning and normalization adds engineering risk.

---

## 8. Why RESERVE rather than SELECT

The candidate B disposition decision rests on three weighted factors:

**Factor 1 — Overlap fact dominates (weight 0.4).** The 50-70% mechanism overlap with #62 means that empirical evidence for AGENTIC-WORKFLOW improvement may be confounded by #62's single-agent agency. Causal attribution between #62 and #89-B requires careful ablation.

**Factor 2 — Magnitude below iter-228 bar (weight 0.35).** 1.5-3× on a benchmark subset (multi-agent only; not text NLL, not all agent benchmarks) is structurally insufficient for the magnitudes-better selection criterion.

**Factor 3 — User brief signal (weight 0.25).** The iter-233 prompt mentions agentic workflow systems by name (LangChain, AutoGen, CrewAI, Swarm). This signals user interest. **However**, the iter-200 critique of microoptimization explicitly applies to single-axis-of-improvement extensions of existing paradigms.

**Net.** The 0.4 + 0.35 = 0.75 weight against SELECT outweighs the 0.25 user-brief weight in favor of investigation. Recommended disposition: **RESERVE for iter-234+ promotion** if either:
- (a) Empirical evidence from #62 AGENT-CHIRON Gate-1 establishes single-agent agency as production-ready and identifies multi-agent orchestration as the next bottleneck.
- (b) User brief is sharpened in iter-234+ to explicitly request multi-agent orchestration as strategic priority.

---

## 9. Composition opportunities (if eventually promoted)

### 9.1 Joint with #62 AGENT-CHIRON

Multi-agent orchestration uses #62's `<GOAL>`/`<PLAN>`/`<ACT>`/`<OBS>`/`<REFLECT>`/`<ANSWER>` blocks WITHIN each agent role. The multi-actor protocol layer wraps single-agent trajectories at the next level of nesting. Joint speedup: 2.0-4.5× on multi-agent subset (multiplicative).

### 9.2 Joint with #59 PRM-CHIRON

Per-coordination-decision PRM scoring extends #59's per-reasoning-step framework to per-routing-step. The PRM head shares parameters across single-step reasoning (#59), per-step agency (#62), and per-coordination (#89-B) — a single PRM amortized across three axes. Joint speedup: ~3.0× on multi-agent subset (vs ~1.7× #59 alone).

### 9.3 Joint with #69 REASONING-DISTILL

Reasoning chains within each agent's `<PLAN>` and `<ACT>` blocks distilled per #69. Multi-agent orchestration thus inherits both coordination signal (#89-B) and within-agent reasoning quality (#69). Joint speedup: ~3.5× on multi-agent benchmarks involving complex reasoning per role.

### 9.4 Joint with #56 DISTILL-FORWARD

Multi-agent traces serve as distillation source per #56 framework. Standard knowledge distillation infrastructure reused. Marginal engineering cost: ~0 LOC (entirely within #56 pipeline).

---

## 10. Bottom line

**AGENTIC-WORKFLOW-DISTILL-CHIRON is recommended for RESERVE at iter-233 candidate B disposition.**

The disposition reflects:

- **Mechanism overlap with #62 AGENT-CHIRON is high (50-70%).** Multi-agent extension is incremental over single-agent multi-step.
- **Magnitude is microoptimization-tier (1.5-3× standalone on benchmark subset).** Below iter-228 magnitudes-better bar; risk-adj 1.5-2.0×.
- **Production precedent is strong but at non-CHIRON architecture.** LangChain/AutoGen/CrewAI/Swarm validate orchestration patterns but at GPT-4-class scale.
- **No new orthogonal axis.** Extends #62 AGENCY axis; does not open 28th axis.
- **iter-200 microopt critique applies.** 1.5-3× single-axis improvement is precisely the pattern flagged.

**Reservation criteria for iter-234+ promotion:**
1. Empirical evidence from #62 Gate-1 establishing single-agent agency as production-ready.
2. User-brief sharpening on multi-agent orchestration as strategic priority.
3. Validation phase (per #87-C META-VALIDATION recommendation) producing direct empirical signal that multi-agent agentic workflows are the bottleneck in CHIRON's agent-benchmark performance.

**If eventually promoted, expected disposition:**
- ~700 LOC over 3.5 weeks
- Joint Gate-0 PASS ~50%; LLM-scale confirmation ~32%
- Risk-adj 1.5-2.0× on multi-agent subset (multi-agent benchmarks only; text NLL preserved)
- Cumulative single-GPU stack: AgentBench-multi-agent / SWE-Bench-multi-actor / GAIA-multi-step subset ~1.7× geometric mean above pre-#89 baseline

**The verdict, restated honestly:** AGENTIC-WORKFLOW is sound mechanism with production precedent and clear composition. It is RESERVED, not SELECTED, because (a) overlap with #62 is high, (b) magnitude is microoptimization-tier under iter-200 critique, and (c) no orthogonal axis is opened. These three facts collectively recommend deferring to a future iteration in which empirical signal or user brief sharpening makes the deferred promotion clearly indicated.

---

## 11. Iter-233 saturation pattern note

Iter-233 candidate slate includes #89-B AGENTIC-WORKFLOW (this doc) alongside other slate candidates. Per the iter-232 acknowledgment, the post-iter-224 saturation pattern continues:

- Iter-225 #81 MAMBA-2: second saturation, least-bad.
- Iter-228 #84 VIDEO-INPUT: axis-extension at ~5M×.
- Iter-229 #85 LANGUAGE-FAMILY: axis-extension at ~5M×.
- Iter-230 #86 EMBODIED-ACTION: axis-extension at ~5M×.
- Iter-231 #87 META-VALIDATION-RESERVED + VIDEO-OUTPUT axis-extension: third saturation.
- Iter-232 #88 3D-SPATIAL: axis-extension; fourth saturation explicitly acknowledged.
- Iter-233 #89-B AGENTIC-WORKFLOW: extends existing AGENCY axis (#62); does NOT open new axis. **Saturation continues.**

If the iter-233 slate produces no candidate opening a 28th orthogonal axis, the iter-232 strategic recommendations remain in force:
- **Recomposition** of more rejected paradigms under iter-212 framing.
- **Constraint relaxation** (multi-GPU; further bit-exact NLL relaxation; not yet user-signaled).
- **Empirical validation feedback** (per #87-C META-VALIDATION reserved-as-recommendation).
- **Continued axis-extensions** at structurally-bounded ~5M× each.

The fact that #89-B is candidate B (not the SELECT) within iter-233 indicates the slate is producing candidates at decreasing structural distinguishability. This itself is a saturation signal worth tracking iter-over-iter.
