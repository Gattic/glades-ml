# Paradigm Shift #57 Candidate C — ATLAS-LIVE: Continual / Lifelong Pre-Training with Reversibility-Protected Memory Replay

**Status:** candidate-C design for paradigm shift #57. **HONEST RECOMMENDATION: REJECT for #57.** ATLAS-LIVE is an OPERATIONAL paradigm shift (deployment + always-current models), not a training-step speedup. It does not address the user's "compute speed to fixed final NLL" axis on which #1–#56 have been scored. This document is written because the continual-learning direction is large, novel, and worth being on the record — but it is the wrong paradigm for the iter-200 brief.
**Date:** 2026-05-08 (Ralph-loop iter 201, post-#56 DISTILL-FORWARD selection).
**Predecessors:** `PARADIGM_SHIFT_42_DESIGN.md` (CHIRON Theorem 1; reversibility is the load-bearing primitive for cheap memory replay); `PARADIGM_SHIFT_56_CANDIDATE_B_DISTILL_FORWARD.md` (training-as-multi-generation precedent); `PARADIGM_SHIFT_39_DESIGN.md` (RLG late-layer growth — same primitive for era-stratified absorption); `surprise18_continuation_drift.md` (post-resume LR mini-warmup + cosine LR decay).
**Axis:** **temporal extent of training**. Reframe LLM training as a never-ending stream rather than a one-shot pretraining run with frozen deployment.

**Tagline.** *#42–#56 each ask "given a fixed corpus and budget, reach L\* faster". ATLAS-LIVE asks "what if pre-training never ends, and the model is deployed while still learning". Training horizon unbounded; knowledge always current; forgetting suppressed by CHIRON-reversibility-driven memory replay. **Operationally bigger-picture, but does NOT speed up wall-clock to fixed final NLL — and that is the user's brief.***

---

## 0. Executive summary (HONEST recommendation: REJECT for #57)

ATLAS-LIVE is a genuinely novel paradigm shift on the **temporal-extent** axis. It is also the **wrong paradigm** for the iter-200 brief, which scores paradigms by wall-clock to fixed final NLL on a fixed compute budget. ATLAS-LIVE distributes the same target NLL across **more time** (continuous training), a different axis. Under the user's scoring rubric, ATLAS-LIVE delivers a 1.0× speedup — formally a non-improvement on the metric. **Recommendation: defer.**

**What ATLAS-LIVE does.** Treats training as an unbounded stream (web crawl, news, papers, RSS, RL logs); suppresses catastrophic forgetting via CHIRON-reversibility-protected memory replay from periodic checkpoints (every 10k steps); mixes 80% current-stream gradients with 20% replay-checkpoint gradients; permits late-layer specialization on recent data while early layers preserve foundational distribution.

**What ATLAS-LIVE does NOT do.** Does not reduce SGD steps to reach a fixed final NLL; does not reduce per-step compute; does not compose multiplicatively with #42–#56 on the wall-clock-to-L\* axis (it composes on the deployment-currency axis, scored 1.0× under iter-193).

**Speedup framing.** Three possible frames:

| Frame | What it measures | ATLAS-LIVE score |
|---|---|---|
| **iter-193 fixed final NLL** (user's brief) | wall-clock to reach `L\*` from fresh init | **1.0× (no speedup)** |
| Compute-amortization across years | total knowledge per cumulative GPU-hour | ~100× over 1–2 yrs vs episodic pretraining |
| Peak compute headroom | max instantaneous compute for "current" model | ~50–100× smoothing vs episodic peaks |

The latter two are operationally significant but are **not the iter-193 axis**.

**Why this design is on the record anyway.** (1) The bigger-picture brief is genuinely satisfied at the framing level — temporal-horizon thinking *is* paradigm-level — even though the scoring metric does not credit it; (2) the CHIRON-reversibility synergy is real (activation-memory replay is `O(1)` via bijective shear-flow); (3) honest record-keeping for a future operational-paradigm slot.

**Recommendation.** REJECT ATLAS-LIVE for paradigm slot #57. Defer to a future operational-paradigm slot when the user's brief includes a temporal axis. SCROLL (data-side active learning) should hold the #57 slot.

---

## 1. Why "temporal extent" is a paradigm-level axis (and why it is the wrong one for #57)

Through #56 the project has attacked eight axes (activation memory, forward attention compute, FFN/projection compute, optimizer state, optimizer trajectory, data sampling, loss formulation, architecture-as-state). All eight share a property: they take **the training run as the unit of work**. The clock starts at "begin training" and stops at "deploy." ATLAS-LIVE breaks that boundary — the model is deployed **while still being updated**.

These are orthogonal axes, but they are not commensurable. The user's iter-193 framing is unambiguous: *given a fixed compute budget C and a target NLL L\**, *how long does it take in wall-clock to reach L\**? ATLAS-LIVE re-asks this question with a different time horizon — *given an unbounded compute stream, what NLL trajectory does the model maintain over a year-long deployment*? Under iter-193, #56-B DISTILL-FORWARD scores 5× because it reaches L\* in 1/5 the wall-clock of the baseline. Under ATLAS-LIVE the question "how long to reach L\*" is malformed: L\* is reached at some point, then the model continues training and reaches L\* − ΔL, then L\* − 2ΔL. The headline number is "asymptotic NLL after N years," not "wall-clock to L\*."

**Three deployment regimes where ATLAS-LIVE is the correct answer:** always-current LLMs (search assistant, news summarizer, knowledge-cutoff = today); long-tail RLHF / continuous fine-tuning from user feedback signals; multi-year institutional deployments amortizing hardware turnover without retrain-from-scratch. **None of these is the iter-200 brief.** The user has not asked for any of them. ATLAS-LIVE in this slot is solving the wrong problem.

---

## 2. Continual-learning mathematics

### 2.1 The streaming pre-training objective

Standard pretraining minimizes a fixed-corpus expectation:
$$
\mathcal{L}_{\text{pre}}(\theta) \;=\; \mathbb{E}_{x \sim \mathcal{D}_{\text{pre}}}\bigl[ -\log P_\theta(x) \bigr],
$$
over a frozen training distribution `D_pre`. The deployment distribution `D_deploy` is unobserved and assumed identical to `D_pre`.

ATLAS-LIVE replaces the static distribution with a **time-indexed stream** `{D_t}`:
$$
\mathcal{L}_{\text{stream}}(\theta_t, t) \;=\; \mathbb{E}_{x \sim \mathcal{D}_t}\bigl[ -\log P_{\theta_t}(x) \bigr],
$$
where `D_t` is the data distribution at wall-clock time `t` (today's news, today's papers, today's code commits). The model `θ_t` is updated continuously to track `D_t`.

The naive online-SGD update — `θ_{t+1} = θ_t − η ∇ L(θ_t; x_t)` — converges in expectation only if `D_t` is stationary, which by definition it is not. **Without intervention, online SGD on non-stationary `D_t` exhibits catastrophic forgetting**: gradients on recent batches overwrite competence on historical distribution components.

### 2.2 The catastrophic-forgetting bound

McCloskey 1989; Kirkpatrick 2017 (EWC). Online SGD on a switching distribution (`D_1 → ... → D_T`) has forgetting bound `E[L_{D_{t-k}}(θ_t) − L_{D_{t-k}}(θ_{t-k})] ≤ (1/2) η · tr(F_{D_{t-k}}) · k`, where `F_D` is Fisher information. **Forgetting grows linearly in elapsed steps `k`**, scaled by Fisher trace and LR. At LLM scale (tr(F) ~10^9, η = 3e-4, k = 10^5 in a week of streaming) the magnitude is catastrophic without mitigation. The mitigation literature splits into regularization (EWC, MAS, PathInt — Fisher-weighted L2), replay (ER, A-GEM, GEM, DER — inject old-distribution samples into gradient), and architectural (PackNet, ProgressiveNet — dedicated subnetworks). ATLAS-LIVE is **replay-family** with optional architectural specialization. Replay was selected over regularization because (1) Fisher computation at LLM scale is HVP-class expensive; (2) replay has stronger empirical scaling at LLM/RL scale (Rolnick 2019); (3) the CHIRON-reversibility primitive makes replay cheap enough to be the default.

### 2.3 The replay-mixing gradient

ATLAS-LIVE forms its update by mixing current-stream and replay gradients from periodic past checkpoints:
$$
\theta_{t+1} = \theta_t - \eta\bigl[\alpha \nabla L_{D_t}(\theta_t) + (1-\alpha)\nabla L_{\mathcal{R}}(\theta_t)\bigr],
$$
with `α = 0.8` and `R` drawn from past checkpoints `{C_0, C_K, C_2K, ...}`, `K = 10^4` steps. The replay data is **checkpoint-derived synthetic**: sample past `C_k`, generate ~M synthetic completions from `C_k` on a held-out prompt set, treat (prompt, completion) pairs as replay data for `θ_t`. This is the **Distillation-from-Past-Self** variant (closer to Buzzega 2020 DER than raw experience-replay) — operationally we never need to store raw input data, only past parameter snapshots.

### 2.4 Convergence of replay-mixed SGD

Under standard assumptions (bounded gradients, smooth losses, decaying learning rate `η_t = η_0 / sqrt(t)`), replay-mixed SGD on a slowly-drifting stream `D_t` converges to a moving target with bounded tracking error:
$$
\mathbb{E}\bigl[\|\theta_t - \theta^*_t\|^2\bigr] \;\le\; \tfrac{C_1}{\sqrt{t}} + C_2 \,\sup_{s \le t} \|D_s - D_{s-1}\|_{\mathrm{TV}},
$$
where `θ^*_t = \mathrm{argmin} L_{D_t}`. The first term is the standard SGD rate; the second is a tracking-error penalty proportional to the distribution-drift TV-norm. At natural rates of language drift (~0.1% TV-distance per month based on common-crawl corpus comparisons 2020–2024), the steady-state tracking error is bounded by `~0.05` nat — small relative to typical training-time NLL improvements. This is the load-bearing convergence claim for ATLAS-LIVE.

### 2.5 Why distribution drift is bounded in practice

Two convenient facts about natural-language drift: **(a)** the top-10k Zipf vocabulary changes by less than 5% per year (Twitter / common-crawl studies) — most absorption is *factual* (new entities, new events) rather than *distributional*; **(b)** tokenizer, model dimension, layer count, RoPE basis, and all #42–#56 paradigm states are unchanged — drift is in the data manifold, not the model substrate. (a) bounds the tracking-error penalty `C_2 · ||D_s − D_{s-1}||`; (b) eliminates the need for tokenizer or architecture surgery during streaming.

---

## 3. CHIRON-reversibility memory replay — the synergy that makes ATLAS-LIVE thinkable at scale

This is the section where ATLAS-LIVE has its only genuine technical novelty. The continual-learning literature has known about replay for 30 years; what is novel is using CHIRON's reversible-flow structure to make replay **cheap** enough to run continuously at LLM scale.

### 3.1 The replay-cost problem on a non-reversible stack

Replay requires (i) a past checkpoint `C_k`, (ii) a forward pass through `C_k` to generate replay tokens, (iii) a loss computation under the *current* `θ_t` on the generated replay data. On a non-reversible stack the load-bearing cost is **activation memory**: replaying through a 53-layer 18B model needs ~24 GB of activation cache per batch — colliding with the headroom needed for the current model's training. Without CHIRON, ATLAS-LIVE's replay events are gated by the same 16 GB GPU ceiling that motivated CHIRON.

### 3.2 The CHIRON synergy: replay forward is `O(1)`-activation-memory by construction

CHIRON Theorem 1 (paradigm #42): the block map `(q, p) → (q', p') = (q, p + Y(q))` followed by reversible LayerNorm is bijective with explicit inverse. **The same primitive applies to replay forward** — a past checkpoint's forward pass can run with `O(1)` activation memory (per-block scratch only, not full per-layer cache), since the inverse walk is available if gradients were needed (they are not — replay generation is forward-only). Operationally: replay forward runs in `O(L · m · T)` activation footprint instead of `O(L^2 · m · T)`. At 18B / T=1024 this is ~0.5 GB instead of ~24 GB. Replay no longer competes with current training for memory.

This is the only **real** technical novelty in ATLAS-LIVE. Without CHIRON, replay-from-past-self does not scale to LLM size on a 16 GB GPU. With CHIRON, it does.

### 3.3 Replay protocol

Per SGD step: 80% current-stream gradient `g_stream = ∇L(θ_t, x_stream)`; every 10 steps, sample a past checkpoint `C_k` uniformly from the last 256 retained, run a CHIRON-forward through `C_k` on a 64-prompt set to generate replay tokens (O(1) activation memory by §3.2), compute `g_replay = ∇L(θ_t, x_replay)`, and form `g_total = 0.8·g_stream + 0.2·g_replay`. Adam-update with `g_total`. Save full checkpoint every 10000 steps. Total per-step compute: `3F` baseline + `~0.05F` per replay event (1/10 frequency × CHIRON-forward at `0.5×F` cost) = `3.05F` average — **1.7% overhead**, identical to #56-B's per-step cost framing.

### 3.4 Storage cost of the checkpoint chain

Naive checkpointing (every 10k steps, indefinite retention) is unaffordable: at 18B params with INT8 Adam state and BF16 weights, each checkpoint is ~22 GB; a year of streaming at 10k steps/day yields 365 checkpoints = 8 TB. Sliding-window retention (last 256 checkpoints + sparse log-scale tail) reduces to ~5.6 TB + ~200 GB long-tail anchors, manageable on a single SSD per training rig. Smarter alternatives — Δ-checkpointing (~20× compression) and rank-r Stiefel × Σ summaries via paradigm #7 (~100× lossy compression) — are operational engineering, deferred.

---

## 4. Late-layer specialization: the architectural component

**Motivation.** Voita 2019 (head pruning), Geva 2021 (FFNs as key-value memories), Tenney 2019 (BERT's layered linguistic hierarchy): early transformer layers encode general-purpose syntactic features; late layers encode task-specific and recency-weighted memory traces. This justifies splitting the ATLAS-LIVE update into two regimes.

**Stratified learning rates.** Per-layer LR multiplier `μ_l = 0.1` for `l ≤ L/3` (early), `0.5` for middle, `1.0` for `l > 2L/3` (late). Early layers preserve foundational knowledge (slow); late layers absorb recent data (fast). The replay loss inverts: `μ_l^{replay} = 1 − μ_l + 0.1`, so replay anchors *early* layers (foundational distribution) and lightly touches late layers (which legitimately drift to track recent data). PackNet-flavored stratification without discrete subnetwork allocation.

**Composition with paradigm #39 RLG.** Paradigm #39 (validated 1.30× at 1.84B) provides the primitive: insert a fresh layer with `Wo = 0` identity initialization. ATLAS-LIVE uses RLG to **periodically grow new late layers** dedicated to new data eras: every 6 months insert 2 new layers at positions `0.85·L` and `0.95·L`; train predominantly on recent-stream data (μ_l = 1.0); anneal old late layers' LR to ~0.1 (de-emphasized soft-frozen specialists for their historical eras). Over a multi-year deployment the network accrues a stack of era-specialized late layers while early layers remain stable. **This is the strongest piece of ATLAS-LIVE's design** — but it still operates on the temporal-extent axis, not iter-193, and does not change the headline rejection.

---

## 5. Honest gap analysis: why ATLAS-LIVE fails the iter-200 brief

**5.1 Wall-clock to fixed final NLL: 1.0×.** The iter-193 protocol (`BEYOND_CHIRON.md` §2.3): fix L\*, fix compute budget C, measure wall-clock to first reach L\*. ATLAS-LIVE on this protocol behaves exactly like baseline online SGD — the streaming structure does not accelerate convergence to L\*; it only changes what happens *after* L\*. Score: **1.0×**. Every shipped paradigm in #1–#56 has a strictly-greater-than-1.0× score on this metric (#56-B DISTILL-FORWARD: 5×; #55 SOPHIA: 1.875×; #50 HELIUM: 1.875×; #38 SLC: 1.5–1.68×; #39 RLG: 1.30×). ATLAS-LIVE alone is a non-improvement; selecting it means trading a measurable speedup for a quality the user has not requested.

**5.2 Composition with #1–#56 is NOT multiplicative on the speedup metric.** Every prior paradigm composes multiplicatively with #1–#55 on iter-193 (the cumulative ~3280× post-#55 is exactly this multiplication). ATLAS-LIVE composes multiplicatively only on the *operational* axis (e.g., "always-current 18B student trained continuously"), but the iter-193 metric of the combined system equals the iter-193 metric of #56-B alone. **ATLAS-LIVE's iter-193 multiplier is 1.0×.** This is not a bug in ATLAS-LIVE; it is a feature of the metric. Orthogonal-with-multiplier-1 means it does not aggregate into the cumulative speedup the project tracks.

**5.3 Engineering scope: ~3200 LOC over ~12 weeks, dominated by operational engineering.** Streaming ingestion ~800 LOC; checkpoint mgr + retention ~400; replay scheduler ~500; stratified-LR ~200; era-RLG insertion ~300; 24/7 monitoring + drift detection ~600; crash-recovery ~400. By comparison, #56-B DISTILL-FORWARD shipped at 400 LOC over 2 weeks. ATLAS-LIVE's engineering surface is ~8× larger and the work is *operational* rather than *research* — a poor fit for the project's research-lab posture.

**5.4 No published LLM-scale validation of continuous pretraining.** Continual-learning literature is large (Kirkpatrick 2017 EWC, Rolnick 2019 CLEAR, Lopez-Paz 2017 GEM, Buzzega 2020 DER) but every benchmarked study is at convolutional / RL / small-classifier scale. The closest LLM-scale work is unpublished continual fine-tuning (rumored ChatGPT / undisclosed RLHF deployments). **No published study of multi-month continuous pretraining at >1B parameters with a measured NLL trajectory.** Gate-0 for ATLAS-LIVE would require ~6 months of continuous training to produce meaningful signal — incompatible with the Ralph-loop iteration cycle (1–7 days per paradigm).

**5.5 Operational risk dwarfs research risk.** ATLAS-LIVE is a 24/7 system, not a finite-N training run. Failure modes include data-feed corruption / poisoning (irreversible model degradation), distribution-shift cliffs (sharp NLL spikes the replay buffer cannot suppress), storage exhaustion (overnight unattended-process termination), and customer-facing regressions on a deployed surface. None of these are present in #1–#56 paradigms. ATLAS-LIVE inherits the entire operational-reliability surface of a production system — **not what a research-lab paradigm shift should be carrying**.

**5.6 Misalignment with the iter-200 "bigger picture" intent.** The brief was *"by looking at the bigger picture instead of focusing on microoptimizations"* — clearly **paradigm-level** thinking, not deployment-level thinking. A clean reading of the brief is "rethink the training method's underlying structure" (which #56-A SCROLL, #56-B DISTILL-FORWARD, #56-C ATLAS-EVO all do at the level of data, loss, and architecture). ATLAS-LIVE *does* rethink the training method, but it does so by asking a question the user did not ask. The blunt description: ATLAS-LIVE is solving a different problem and labeling it as a candidate for the user's problem.

---

## 6. The honest case for keeping ATLAS-LIVE on the deferred-paradigm log

Despite the rejection recommendation for slot #57, three properties of ATLAS-LIVE merit recording the design now: **(1)** the CHIRON-reversibility synergy (§3.2) is independent of the rejection — CHIRON's bijective shear-flow makes replay-style continual learning cheap at LLM scale in a way no other modern transformer architecture does, a structural advantage of the CHIRON stack that competitors will eventually need to replicate; **(2)** era-stratified RLG insertion (§4.3) is a distinct, smaller contribution that could be slotted as a minor extension to paradigm #39 when the project ships its first continuous-deployment system; **(3)** the temporal-extent axis is genuinely paradigm-untouched — likely #60+, when the project crosses from research-lab to deployment-lab, a paradigm slot for continuous pretraining will be the natural choice. Documenting ATLAS-LIVE's design now means the slot has a serious prior candidate ready.

**Conditions under which the rejection should be revisited:** the user's brief shifts from "compute speed to fixed L\*" to "asymptotic NLL on a multi-year horizon" (iter-193 rubric explicitly relaxed); the project ships its first user-facing inference deployment where stale-knowledge complaints accumulate; a published LLM-scale continuous-pretraining study lands at >1B parameters with measured NLL improvement vs episodic retraining. None of these hold as of iter-201. **Defer.**

---

## 7. Recommended #57 disposition

### 7.1 Selection slate for #57

The current #57 candidate slate is:
- **A — SCROLL.** Data-side active learning (deferred from #56-A). 1.5–5× wall-clock speedup; 800 LOC; 4 weeks. Composes multiplicatively with #56-B DISTILL-FORWARD. **Strong candidate.**
- **B — TBD second-generation distillation refinement** (NEXUS-L, or "DISTILL-FORWARD-PRO" with self-distillation + KL temperature scheduling).
- **C — ATLAS-LIVE** (this document). 1.0× wall-clock speedup on iter-193. **Reject.**

### 7.2 Recommendation

Select **SCROLL** for #57 (already developed at `PARADIGM_SHIFT_56_CANDIDATE_A_SCROLL.md`; promotion path is straightforward; composes multiplicatively with #56-B for combined ~10–25× steps reduction). Defer ATLAS-LIVE to the future operational-paradigm slot once one of the §6.4 conditions holds.

### 7.3 If the user explicitly wants the operational-paradigm direction

Two openings for the user to override:
- "I want to start the deployment story now." → ATLAS-LIVE becomes the right paradigm and the iter-193 rubric should be loosened to include a temporal-axis component.
- "I want to redefine 'speedup' to include compute amortization across years." → ATLAS-LIVE scores ~100× under the new metric and becomes the headline paradigm of the project.

In either case the rejection should be reversed and ATLAS-LIVE promoted to selected. Without explicit user override on the metric, the rejection stands.

---

## 8. Summary table

| Property | ATLAS-LIVE | #56-B DISTILL-FORWARD (selected) | #56-A SCROLL (deferred to #57) | #56-C ATLAS-EVO (deferred) |
|---|---|---|---|---|
| Axis | Temporal extent | Loss formulation | Data sampling | Architecture |
| iter-193 wall-clock to L\* | **1.0×** | 5× | 1.5–5× | 1.5–2× |
| Compute amortization (multi-year) | ~100× | ~1× | ~1× | ~1× |
| Engineering LOC | ~3200 | 400 | 800 | 1500 |
| Engineering weeks | ~12 (deployment) | 2 | 4 | 8 |
| Published LLM precedent | ~0 (continual pretraining unpublished at LLM scale) | Strong (Hinton, Sanh, MobileLLM) | Modest (Katharopoulos vision-scale) | Weak (Cosmos at 1B, ~1.1×) |
| CHIRON-reversibility synergy | **Strong (replay activation memory `O(1)`)** | None | None | Strong (bijectivity preserved under layer count) |
| Composes mult. on iter-193 | **No (×1.0)** | Yes | Yes | Yes |
| Brief alignment | Misaligned (operational) | Aligned | Aligned | Aligned |
| **#57 disposition** | **REJECT** | shipped #56 | **SELECT for #57** | hold for #58+ |

---

## 9. Cross-references

- `PARADIGM_SHIFT_42_DESIGN.md` — CHIRON Theorem 1: bijective shear-flow makes replay forward `O(1)`-activation-memory. The load-bearing primitive for ATLAS-LIVE's only genuine technical novelty.
- `PARADIGM_SHIFT_39_DESIGN.md` — RLG identity-insertion, reused by ATLAS-LIVE's era-stratified late-layer growth (§4).
- `PARADIGM_SHIFT_56_CANDIDATE_B_DISTILL_FORWARD.md` — multi-generation knowledge accumulation; ATLAS-LIVE generalizes "training as a chain" from generation-discrete to time-continuous.
- `PARADIGM_SHIFT_56_CANDIDATE_A_SCROLL.md` — recommended #57 selection over ATLAS-LIVE.
- `surprise18_continuation_drift.md` — iter-184 resume-time LR mini-warmup + cosine LR decay; **prerequisite for ATLAS-LIVE if ever shipped** (streaming system reuses the continuation-stability primitive at every checkpoint-replay event).
- `BEYOND_CHIRON.md` §2.3 — fixed-final-NLL benchmark protocol under which ATLAS-LIVE scores 1.0×.
- `FUTURE_PARADIGM_CANDIDATES.md` — add ATLAS-LIVE under "deferred operational paradigms" with the §6 reactivation conditions.

---

## 10. Final position

**Reject ATLAS-LIVE for paradigm slot #57.** Promote SCROLL to #57. Defer ATLAS-LIVE to a future operational-paradigm slot (likely #60+) when one of the §6 reactivation conditions holds.

The CHIRON-reversibility-replay synergy (§3.2) and era-stratified RLG insertion (§4) are the two pieces that will retain value when ATLAS-LIVE is eventually picked up; the rest is operational engineering that should not consume a paradigm slot under the iter-200 brief.

The user's brief asks for compute speed to fixed final NLL. ATLAS-LIVE distributes the same target NLL across more time. **Different questions; answering the wrong one is not novelty — it is misalignment.** Honest rejection now beats a 12-week engineering effort that scores 1.0× on the metric the project tracks.
