# Paradigm Shift #64 Candidate C — COMPUTE-ALLOCATOR (per-token learned joint allocator over experts, tools, and halt)

**Status:** candidate-C design for paradigm shift #64. **Recommended action: candidate, but honestly framed as a speculative entry in the #64 slate with substantial mechanistic overlap with three already-shipped or reserved paradigms.** Marginal contribution at paradigm depth 23 is small.
**Date:** 2026-05-08 (Ralph-loop iter 208, post-#63 META-LEARN-PROMOTED selection, paradigm depth 23 in the bigger-picture track #56–#63).
**Predecessors.** All of #42–#63. Load-bearing references: `PARADIGM_SHIFT_53_CANDIDATE_B_MOSAIC_MOE.md` (per-token expert routing inside the FFN shear; the *width-side* analog of the expert-axis decision), `PARADIGM_SHIFT_60_CANDIDATE_C_TOOL_LLM.md` (in-vocabulary tool-call special tokens; the *capability-boundary* analog of the tool-axis decision), `PARADIGM_SHIFT_49_CANDIDATE_C_AURORA.md` (rejected; heuristic-thresholded ACT-style halt; the *depth-axis* analog the allocator re-frames as a learned routing decision).
**Axis.** **Per-token joint compute-allocation.** A small allocator network takes the token's hidden state and emits a *joint distribution* over three independent decisions: (1) which `k` of `E` MOSAIC experts to activate, (2) whether to emit a tool-call special token (and which tool), (3) whether to halt (skip subsequent layers). Decisions are sampled jointly via Gumbel-softmax (training) or arg-top-k (inference) and back-propagated through the trunk's primary CE.

**Tagline.** *#53 MOSAIC-MOE makes the per-token expert decision. #60 TOOL-LLM makes the per-token tool decision. #49-rejected AURORA tries to make the per-token halt decision via a hand-tuned threshold and falls below the magnitude bar. COMPUTE-ALLOCATOR makes all three decisions JOINTLY, on the hypothesis that the axes interact — a token calling a tool likely needs fewer layers; a token whose expert routing concentrates is "easy" in the depth sense too. The headline is the joint allocation, not any individual axis. Value-add at depth 23 is the **single locus** for compute-budget enforcement that all three decisions answer to.*

**Honest headline.** **~1.2× wall-clock speedup at fixed final NLL (conservative); ~1.5× aggressive if the cross-axis interaction hypothesis holds and the allocator generalizes.** NLL preservation: yes (the allocator is auxiliary; primary loss is CE; decisions are differentiable via Gumbel-softmax + straight-through estimator). The 1.2× claim reflects that the bigger-picture track has independently shipped or reserved paradigms covering each of the three axes.

---

## 0. Executive summary (HONEST claim — modest gain; substantial overlap)

The pre-#64 cumulative stack has shipped #53 MOSAIC-MOE, #60 TOOL-LLM, #62 AGENT-CHIRON, and #63 META-LEARN-CHIRON; #49-C AURORA was rejected at iter-193 because conservative-threshold ACT delivered only ~1.2–1.3×, below the magnitude bar. The dominant per-token routing paradigms in this stack are precisely those the allocator most overlaps with — MOSAIC routes per-token over `E` experts; TOOL-LLM routes per-token over `K_tools` tools via special tokens; AURORA tried to route per-token over `L` depths via a halt threshold. **COMPUTE-ALLOCATOR's "joint" claim is that the three decisions co-occur on the same token-state vector and a single allocator amortizes them; its independent contribution is the coupling, not any individual axis.**

The three sub-mechanisms:

1. **Per-layer allocator head.** ~5M-param network `g_φ : q ∈ ℝ^m → z ∈ ℝ^{E + K_tools + 2}` (two MLP layers + GELU + three per-axis projection heads, mirroring the #59 PRM-CHIRON pattern). Joint loss `L = L_CE + λ_alloc · L_alloc + λ_cost · L_compute`. Per-step trunk overhead `≤ 0.4%`.
2. **Joint Gumbel-softmax sampling.** Three axes sampled independently with temperatures `(τ_E, τ_T, τ_H)` annealed from 2.0 → 0.5 at step `0.5·N`. Joint sample decoded into top-`k` expert selection (compatible with #53), tool-call gating (compatible with #60), halt (compatible with #49). Gradients via straight-through estimator.
3. **Compute-cost regularizer.** `L_compute = ∑_t (FLOPs_t − target_FLOPs)^2 / target_FLOPs^2`. The budget knob is the headline-1.2× lever — without it, the allocator has no incentive to save compute.

**Per-effective-step speedup (conjectured):** `1.10–1.30×` conservative; `1.30–1.60×` aggressive contingent on the cross-axis interaction hypothesis. **Headline conservative: 1.2× over post-#63 stack.**

**Honest gaps (foregrounded).** Three largest:

1. **Mechanistic overlap with #53, #60, and rejected #49.** Each axis is structurally identical to a paradigm already-shipped or reserved-rejected. Independent contribution is restricted to the **cross-axis interaction** and **single-locus compute-budget regularizer**. Neither is large.
2. **AURORA's rejection precedent.** #49-C was rejected because conservative-threshold ACT delivered only 1.2–1.3× standalone; the allocator's halt axis inherits the same speedup ceiling. The lift must come from cross-axis interaction.
3. **Allocator generalization at small param count is empirically risky.** Recent per-token routing work (Raposo 2024 *Mixture-of-Depths*; Du 2024 *GLaM*; Lewis 2021 *BASE-Layers*) reports small heads are sensitive to corpus composition, Gumbel schedule, and budget regularizer weight. The allocator must reach ≥ 0.75 per-axis routing accuracy AND ≥ 1.10× cross-axis speedup; if accuracy plateaus at ≤ 0.65 on any axis the paradigm reduces to a soft-routing auxiliary with negligible compute savings.

**Engineering scope.** ~750 LOC over ~4 weeks.

---

## 1. Allocator architecture

### 1.1 Head architecture

```
COMPUTE-ALLOCATOR head, shared across L layers:

  q_{ℓ,t} ∈ ℝ^m       (token state, post-LayerNorm, pre-shear)
       ↓
  W_1 ∈ ℝ^{m × d_alloc},  GELU
       ↓
  W_2 ∈ ℝ^{d_alloc × d_alloc},  GELU
       ↓
  ┌──────────────────────────────────┐
  │  Three layer-specific heads:     │
  │   H_E : d_alloc → E              │   expert logits
  │   H_T : d_alloc → K_tools + 1    │   tool logits ("no tool" included)
  │   H_H : d_alloc → 2              │   halt logits (continue, halt)
  └──────────────────────────────────┘
       ↓
  z_{ℓ,t} = (z_E, z_T, z_H) ∈ ℝ^{E + K_tools + 1 + 2}
```

Parameters at flagship 1.84B (`m = 2048`, `d_alloc = 512`, `E = 8`, `K_tools = 8`): backbone `(W_1, W_2)` ~1.31M shared globally; per-layer heads `(H_E, H_T, H_H)` ~9.7K × 53 layers ≈ 0.5M. **Total ~1.8M, well within the 5M-parameter budget.** The head fires once per token per layer; total allocator forward FLOPs per training step ≈ 4 GFLOPs at flagship vs ~5 TFLOPs trunk forward — **wall-clock cost ≤ 0.4% of trunk wall-clock**.

### 1.2 Joint loss

```
L(θ, φ) = L_CE(θ) + λ_alloc · L_alloc(θ, φ) + λ_cost · L_compute(φ),

L_CE(θ)        = −∑_t log P_θ(x_t | x_<t),                                 [primary]
L_alloc(θ, φ)  = ∑_{ℓ,t} D_KL(GS(z_{ℓ,t}; τ) ∥ a*_{ℓ,t}),                  [routing imitation]
L_compute(φ)   = ∑_t (FLOPs_t(g_φ) − target_FLOPs)² / target_FLOPs²,        [budget]
```

`GS(z; τ)` is the Jang–Gu–Poole 2017 Gumbel-softmax estimator. `a*_{ℓ,t}` is a teacher-routing target derived from one of three sources (§1.4). `FLOPs_t(g_φ)` is the per-token compute estimator computed differentiably from the soft Gumbel-softmax samples. `target_FLOPs` defaults to `0.8 × dense-trunk FLOPs/token` (1.25× speedup target). `λ_alloc = 0.05` and `λ_cost = 0.10`.

Per-axis gradient norm bounded by `||∂L_alloc/∂q|| ≤ 0.025 · λ_alloc` per axis, total `≤ 0.075 · λ_alloc = 0.00375`, well below the CE gradient norm; no LR retuning.

### 1.3 Joint Gumbel-softmax sampling

The three logit groups parameterize three independent Gumbel-softmax distributions:

```
e ~ GS(z_E; τ_E)  ∈ Δ^{E-1}        (top-k decoded)
t ~ GS(z_T; τ_T)  ∈ Δ^{K_tools}    (argmax decoded)
h ~ GS(z_H; τ_H)  ∈ Δ^1            (binary continue-vs-halt)
```

The three samples are **independent given `q`**. This is a deliberate simplification: a fully-joint distribution over `8 × 9 × 2 = 144` outcomes is too high-dimensional for a 5M-param head to learn well. **Cross-axis coupling lives in the shared backbone** `(W_1, W_2)` — the three heads see the same intermediate representation and learn correlated decisions implicitly. This is structurally identical to the multi-head pattern in PRM-CHIRON's §1.2 (one trunk feeds two heads).

### 1.4 Routing-target supervision

`L_alloc` requires per-axis routing targets `a*_{ℓ,t}`:

- **Expert axis target.** When the post-#53 MOSAIC router exists in the same forward pass, `a*_E := softmax(W_r^T q_{ℓ,t})` is the MOSAIC router's distribution. **The allocator imitates the production router.** This reduces the allocator's expert axis to a parameter-efficient re-parameterization of MOSAIC.
- **Tool axis target.** Where tool-call special tokens are present in the training corpus (#60 data), `a*_T` is one-hot at the tool-class label (or the all-zero "no tool" class otherwise). The allocator predicts which tokens *would* emit a tool call.
- **Halt axis target.** Heuristic: `a*_H := one-hot(continue)` for tokens above the median per-batch CE, `one-hot(halt)` below. Confidently-predicted tokens tolerate halt; uncertain tokens need full depth. Alternative heuristics (`gnorm`, `ent`) admitted via `--alloc-halt-target` flag at Gate-0.

**The "imitate the production router" frame for the expert axis means COMPUTE-ALLOCATOR's marginal contribution there is essentially zero.** The contribution comes from the *other two axes plus cross-axis interaction*.

### 1.5 Three-phase curriculum

- **Phase 0** (`0 → 0.05·N`): `λ_alloc = λ_cost = 0`; allocator is a no-op, gradient only via routing-target supervision.
- **Phase 1** (`0.05·N → 0.3·N`): `λ_alloc` ramps 0 → 0.05; `τ` ramps 2.0 → 1.0; soft sampling.
- **Phase 2** (`0.3·N → 0.7·N`): `λ_cost` ramps 0 → 0.10; `τ` ramps 1.0 → 0.5; **hard sampling via straight-through**; halt/tool decisions become discrete; compute savings accrue.
- **Phase 3** (`0.7·N → N`): all parameters at final values; allocator in production mode.

If `target_FLOPs` is set too aggressively the allocator halts too aggressively in Phase 2 and CE diverges. Mitigation: anneal `target_FLOPs` from `1.0` at `0.3·N` to `0.8` at `0.7·N`. Hard floor `min_t halt_depth_t ≥ ⌈0.5·L⌉` for the first `0.3·N` steps prevents halt-everywhere init.

---

## 2. Composition with #53 MOSAIC-MOE, #60 TOOL-LLM, #49-rejected AURORA

This is the most important honest-accounting section. COMPUTE-ALLOCATOR has substantial mechanistic overlap with each of three pre-existing paradigms; the marginal contribution must be defended carefully.

### 2.1 vs #53 MOSAIC-MOE

**MOSAIC-MOE:** per-layer router `r_t = softmax(W_r^T q_{ℓ,t})` selects top-`k` of `E` experts; LoRA-shared FFN backbone; bijective shear preserved because the router is a deterministic function of `q`. Per-token compute reduction: 4× at FFN-level when `k/E = 1/4`.

**Allocator expert axis:** the `H_E` head outputs an expert distribution structurally identical to MOSAIC's router. The decision pipeline (top-`k` decode, renormalization, dispatch to LoRA experts) is unchanged from #53.

**Distinction.** The allocator's `H_E` head uses a *shared backbone* across layers (saving parameters) but otherwise outputs the same logit space MOSAIC's router does. There is **no expert-axis savings independent of MOSAIC**.

**Realistic joint multiplier on the expert axis: 1.0× (parity) to 1.05× (slight win from shared-backbone regularization).**

### 2.2 vs #60 TOOL-LLM

**TOOL-LLM:** in-vocabulary special tokens (`<TOOL_CALL>`, `<TOOL=python>`, etc.) train the trunk to emit tool-call sequences as part of its output. The routing decision is a deterministic function of the trunk's output distribution at each token position; no separate routing head exists.

**Allocator tool axis:** the `H_T` head outputs a per-token distribution over tool classes, *parallel to* the trunk's special-token output — the allocator predicts "this token *should* emit a tool call" while the trunk separately predicts the actual special-token sequence.

**Distinction.** The allocator's tool axis is **architecturally redundant** with TOOL-LLM at training time: the trunk's output distribution at any token position already specifies whether a tool call is being emitted. **One small win:** the allocator's tool axis fires at *intermediate layers*, allowing speculative tool-call dispatch — the dispatcher pre-warms the tool kernel as soon as the allocator predicts a tool call at layer `L/2`. Latency saving at deployment: ~50–150 ms per tool call.

**Realistic joint multiplier on the tool axis: 1.0× at training time; 1.05–1.10× at deployment latency.** The deployment win is genuine but does not compose multiplicatively with the training-side speedup the #64 slate is selecting for.

### 2.3 vs #49-rejected AURORA

**AURORA (rejected #49-C):** ACT-style halt-mass `P_{ℓ,t} = ∑_{ℓ' ≤ ℓ} h_{ℓ',t}`; halt fires when `P_{ℓ,t} ≥ 1 − ε_halt`; the threshold is hand-tuned (`τ_halt = 0.95`, `ℓ_min = 0.7·L`) to preserve NLL within 0.01 nat. Standalone speedup: 1.2–1.3× under the conservative threshold. **Rejected** at iter-193: below the magnitude bar.

**Allocator halt axis:** the `H_H` head outputs a binary halt logit per layer per token. Decision is *learned*, not threshold-calibrated; supervised via the per-token CE heuristic and gated by the compute-cost regularizer's pressure on average per-token depth.

**Distinction.** The allocator's halt axis replaces AURORA's hand-tuned threshold with a learned classifier. **This does not change AURORA's speedup ceiling** — the underlying mechanism (skip subsequent layers for tokens whose representation is converged) is the same; only the decision rule differs. AURORA's §6 Theorem 3 gives a deterministic NLL bound under the conservative threshold; the allocator's learned halt has no equivalent guarantee, only an empirical bound from Phase-2 monitoring.

The learned classifier can be either better or worse than AURORA's threshold. Better: adapt to within-token-class structure (different halt depth for nouns vs verbs; AURORA's threshold cannot). Worse: a 5M-param allocator on `T × L ≈ 50,000` decisions per batch may underfit, producing noisier halt decisions than the calibrated threshold.

**Realistic joint multiplier on the halt axis: 1.10–1.30× (matches AURORA's standalone band).** Below the magnitude bar in isolation; the cross-axis interaction is the only path to a paradigm-magnitude claim.

### 2.4 Net independent contribution: cross-axis interaction

The three axes individually reproduce paradigms already in or rejected from the stack. **The marginal contribution of COMPUTE-ALLOCATOR is the cross-axis interaction term** — the hypothesis that a token whose expert routing is concentrated has a different optimal halt depth than one with diffuse routing, and that a token emitting a tool call has different downstream-layer requirements than one that does not. If this interaction is real and learnable, the joint allocator captures gains the three axes cannot capture independently.

Three concrete interaction patterns the allocator can in principle exploit:

1. **Tool-call → halt.** A token emitting a tool call has its computation effectively delegated to the tool; the trunk's subsequent layers can halt. Standalone TOOL-LLM does not exploit this. Standalone AURORA has no tool signal. **The joint allocator learns "tool token ⇒ halt soon".**
2. **Expert-confidence → halt.** A token whose expert routing concentrates on one expert (low routing entropy) is "easy" in the FFN sense; it may be "easy" in the depth sense too. **The joint allocator learns "low routing entropy ⇒ halt soon".**
3. **Expert × tool.** Different experts may correlate with different tool-call probabilities (a "math expert" with `<TOOL=calc>`; a "code expert" with `<TOOL=python>`). **The joint allocator learns `P(t | e)`.**

**Realistic interaction-axis multiplier: 1.10–1.20× over the additive sum of the three axes.** This is the only thing on the COMPUTE-ALLOCATOR axis genuinely novel relative to #42–#63.

### 2.5 Net contribution at paradigm depth 23

vs the post-#53 + #60 + (rejected) #49 + #62 + #63 combined baseline:

```
Expert axis    : 1.0–1.05×    (overlap with #53)
Tool axis      : 1.0× train    (overlap with #60); 1.05–1.10× deployment
Halt axis      : 1.10–1.30×    (overlap with rejected #49)
Cross-axis     : 1.10–1.20×    (the genuinely novel headline mechanism)
```

Joint speedup as the geometric mean: **`1.10–1.30×`**. **Headline `1.2×` is the conservative center.** The 1.5× aggressive claim requires the cross-axis multiplier at its upper bound AND the halt axis at its upper bound.

---

## 3. Honest gap and Gate-0 protocol

### 3.1 The three big honest risks

**Risk 1 — Mechanistic overlap is the dominant constraint.** The depth-23 reality is that the bigger-picture track has saturated the per-token routing axis on all three of COMPUTE-ALLOCATOR's component decisions. Independent contribution is bounded by the cross-axis interaction term, which is hypothetical and untested at LLM scale. **If joint speedup at Gate-1 falls below 1.10×, COMPUTE-ALLOCATOR should be rejected as a paradigm shift and reduced to a deployment-time speculative-dispatch feature.**

**Risk 2 — AURORA's rejection precedent.** Paradigm #49-C was rejected because conservative-threshold ACT could not deliver a paradigm-magnitude lift. COMPUTE-ALLOCATOR's halt axis inherits the same speedup ceiling. **If the Gate-0 halt-only ablation reproduces AURORA's 1.2–1.3× and no more, the cross-axis interaction must carry the headline alone — a much higher empirical bar.**

**Risk 3 — Allocator generalization at small parameter count.** Recent per-token-routing literature (Mixture-of-Depths, GLaM, BASE-Layers) reports sensitivity to corpus composition, temperature schedule, and budget regularizer weight. The allocator must reach ≥ 0.75 per-axis routing accuracy AND ≥ 1.10× cross-axis speedup. If accuracy plateaus at ≤ 0.65 on any axis, that axis's gating is too noisy to drive useful compute savings.

### 3.2 Gate-0 protocol (~36 GPU-hours)

**Question.** *On 66M CHIRON × 30k steps, does the COMPUTE-ALLOCATOR achieve ≥ 0.70 per-axis routing accuracy and ≥ 1.10× wall-clock speedup at fixed validation NLL vs the post-#63 baseline?*

Five arms × 30k steps at 66M:

- **A (control)** = post-#63 stack, no allocator, MOSAIC + TOOL-LLM disabled. Target NLL ≈ 3.95 nat.
- **B (halt-only)** = stack + allocator with `H_H` enabled, `H_E` and `H_T` disabled. Tests whether learned halt matches AURORA's calibrated halt.
- **C (joint, MOSAIC + TOOL-LLM enabled)** = stack + MOSAIC + TOOL-LLM + full allocator. Tests the full mechanism.
- **D (joint, allocator with no `L_compute`)** = arm C with `λ_cost = 0`. Tests whether the budget regularizer is necessary.
- **E (joint, allocator with random init at Phase-2 entry)** = arm C with heads re-initialized at step `0.3·N`. Tests whether routing-target supervision in Phases 0–1 is necessary.

**Pass criteria:** (1) Per-axis routing accuracy `≥ 0.70` by step 20k. (2) Arm B speedup over A: `≥ 1.10×` at NLL parity within 0.05 nat. (3) Arm C speedup over A: `≥ 1.20×` at NLL parity. (4) Cross-axis interaction (C ÷ B): `≥ 1.10×`. (5) Arm D speedup over A: `≤ 1.05×` (confirms `L_compute` necessity).

**Fail-fast:** Per-axis accuracy `≤ 0.55` at step 20k → REJECT. Arm C NLL ≥ A + 0.10 nat → REJECT (gating disrupts CE convergence). Arm C cross-axis interaction `≤ 1.02×` → REJECT (joint hypothesis fails). Arm C speedup `≥ 1.40×` → STRONG PASS; proceed to Gate-1.

**Cost.** ~36 GPU-hours = 1.5 GPU-day (parallelizable across ≥ 4 GPUs).

**Gate-1 (post-pass):** 1.84B / 100k steps with full MOSAIC + TOOL-LLM stack. Target ≥ 1.20× speedup, no NLL drift over 5 cycles. ~7 GPU-days. **Gate-2:** joint composition with full post-#63 stack. Target marginal ≥ 1.10×; below this, downgrade to deployment feature.

---

## 4. Engineering: ~750 LOC over ~4 weeks

| Component | LOC | Week |
|---|---|---|
| Allocator head (forward + backward, three projection heads, shared backbone) | 200 | 1 |
| Joint Gumbel-softmax CUDA kernel (three axes, per-axis temperature, straight-through gradient) | 150 | 1 |
| Compute-cost regularizer + per-token FLOPs estimator + budget enforcement | 80 | 2 |
| MOSAIC-MOE integration (allocator's `H_E` overrides MOSAIC router when both active) | 100 | 2 |
| TOOL-LLM integration (allocator's `H_T` parallel to trunk output; speculative dispatch) | 80 | 3 |
| Halt-bookkeeping (per-token halt-depth tensor; reuses AURORA scaffolding) | 70 | 3 |
| Routing-target supervision pipeline (CE-heuristic, MOSAIC-router, TOOL-LLM-special-token) | 70 | 3 |
| CLI flags, monitoring, Gate-0 harness | 100 | 4 |
| **Total** | **~750** | **~4 weeks** |

**Public API.** Mirrors PRM-CHIRON: `glades::alloc::ComputeAllocator` (forward / backward / `routing_accuracy_eval` / save / load) + `glades::alloc::ComputeBudget` (`enforce(allocator_outputs, current_step)` returns the per-token compute mask).

**Risks.** Per-axis gradient bounded by `λ · σ'(z) · ||W_2||_F ≤ 0.025 · λ`; auto-reduce `λ_alloc, λ_cost` if `||∇L_aux|| / ||∇L_CE|| > 0.5`. Missing checkpoint falls back to `λ_alloc = λ_cost = 0` (post-#63 baseline). 7 new hyperparameters; defaults from MOSAIC-MOE / Gumbel-softmax literature (not CHIRON-tuned). Post-Gate-0 tuning ≈ 24 GPU-hours.

---

## 5. Summary

COMPUTE-ALLOCATOR is **the joint per-token compute-allocation primitive at training time**: a small (~5M-param) shared-across-layers head emitting joint distributions over (a) which `k` of `E` MOSAIC experts to activate, (b) whether to emit a tool-call special token, (c) whether to halt for the remaining layers. Decisions are sampled via Gumbel-softmax with annealing temperature, trained via routing-target supervision plus a compute-cost regularizer, and back-propagated through CE via straight-through estimator.

**Training-side speedup at fixed final NLL: 1.2× conservative; 1.5× aggressive contingent on the cross-axis interaction hypothesis.** Per-step overhead ~0.4%; engineering ~750 LOC / ~4 weeks (re-uses ~30% of #53 routing, ~20% of #60 dispatch, ~15% of rejected-#49 halt-bookkeeping).

**Honest overlap.** §2 details substantial overlap with three pre-existing paradigms: #53 MOSAIC-MOE (expert axis is structurally identical to MOSAIC's router; joint `1.0–1.05×`), #60 TOOL-LLM (tool axis is architecturally redundant at training time; joint `1.0×` train, `1.05–1.10×` deploy), #49-rejected AURORA (halt axis re-implements ACT with a learned classifier; joint `1.10–1.30×` matching AURORA's rejected band). Geometric-mean composition yields the ~1.2× headline. **The only paradigm-novel mechanism is the cross-axis interaction term (`1.10–1.20×`).** If the cross-axis hypothesis is empirically weak, COMPUTE-ALLOCATOR reduces to a parameter-efficient re-parameterization of three pre-existing or rejected mechanisms with no paradigm-magnitude lift.

**Honest empirical risks.** (1) AURORA's rejection precedent caps the halt axis's standalone contribution at ~1.3×. (2) Allocator generalization at ~5M params on three-axis decisions across `T × L ≈ 50,000` decisions per batch is not guaranteed; per-token-routing literature reports sensitivity to corpus composition, temperature schedule, and budget regularizer weight. (3) The joint-allocation hypothesis is intuitively plausible but empirically untested at LLM scale; if any cross-axis correlation is weak, the headline shrinks toward the geometric-mean lower bound (1.10×).

**Speculative due to overlap; reasonable but not magnitude-leap** — matching the user's design call-out for #64 candidate C. The depth-23 reality: the bigger-picture track has saturated the per-token routing axis. COMPUTE-ALLOCATOR's contribution is the **single locus** for compute-budget enforcement that all three decisions answer to. Whether this is paradigm-magnitude is the Gate-0 question.

**Recommended action.** Develop through Gate-0 to obtain empirical per-axis routing accuracies and the cross-axis interaction measurement. **Strong-pass** (per-axis ≥ 0.80, cross-axis ≥ 1.20×, joint ≥ 1.40×) → promote to Gate-1. **Marginal-pass** (per-axis 0.65–0.75, cross-axis 1.05–1.15×, joint 1.10–1.20×) → downgrade to deployment-feature (the speculative-dispatch latency win is genuine and ships at zero training cost). **Fail** (per-axis ≤ 0.65 on any axis, cross-axis ≤ 1.05×, joint ≤ 1.05×) → reject; reserve the joint-allocation primitive for a later paradigm where cross-axis correlations are stronger or per-axis components are not pre-saturated.

The 1.2× headline is modest. The paradigm depth is 23. **COMPUTE-ALLOCATOR is the candidate that most clearly admits the overlap — three of three component axes were addressed by other paradigms before #64 — and offers the cross-axis interaction term as the single defensible paradigm-novel contribution. The contribution is real but bounded; the user's framing as "speculative due to overlap; reasonable but not magnitude-leap" is the honest selection criterion.**
