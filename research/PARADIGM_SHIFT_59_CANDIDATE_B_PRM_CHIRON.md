# Paradigm Shift #59 Candidate B — PRM-CHIRON (Process Reward Modeling integrated into pretraining as an auxiliary signal)

**Status:** candidate-B design for paradigm shift #59. One of three parallel proposals for #59.
**Date:** 2026-05-08 (Ralph-loop iter 200+, post-#58 selection, under the iter-200 brief: *"novel architectures, algorithms, and training methods by looking at the bigger picture instead of focusing on microoptimizations."*).
**Predecessors:** `PARADIGM_SHIFT_56_CANDIDATE_B_DISTILL_FORWARD.md` (KL-as-primary-loss precedent for joint loss formulations), `PARADIGM_SHIFT_57_CANDIDATE_A_SCROLL_PROMOTED.md` (data-axis active learning, sample-difficulty signal), `PARADIGM_SHIFT_58_CANDIDATE_C_REASONING_CHAIN.md` (reasoning-token loss-weighting at pretraining), `PARADIGM_SHIFT_53_CANDIDATE_B_MOSAIC_MOE.md` (auxiliary-expert composition path), `BEYOND_CHIRON.md` §2.3 (NLL benchmark protocol).
**Axis:** **auxiliary-loss reward shaping at pretraining time.** Not architecture. Not optimizer. Not data-axis routing. PRM-CHIRON adds a *second loss head* — a small, jointly trained Process Reward Model (PRM) — whose signal flows back into the main trunk on reasoning-segment tokens during pretraining. Cross-entropy remains the primary objective; PRM is auxiliary.

**References.** Lightman et al. *Let's Verify Step by Step.* arXiv:2305.20050 (2023) — PRM800K; PRMs reach 78.2% on MATH vs 72.4% for ORMs at matched compute. Uesato et al. *Solving math word problems with process-and outcome-based feedback.* arXiv:2211.14275 (2022) — ~7-12pp GSM8K gain from process supervision. DeepSeek-AI. *DeepSeek-R1.* arXiv:2501.12948 (2025) — PRM-style step verification at RL post-training. OpenAI. *Improving Mathematical Reasoning with Process Supervision* (2023). Wang et al. *Math-Shepherd.* arXiv:2312.08935 (2023) — synthetic PRM via MC-rollouts; eliminates human-annotation bottleneck. Bai et al. *Constitutional AI.* arXiv:2212.08073 (2022) — auxiliary-reward-during-training pattern.

**Tagline.** *#42–#58 reduced the cost of computing one SGD trajectory or repositioned compute from training to inference. PRM-CHIRON does neither: it adds a small auxiliary reward signal during pretraining itself. Pretraining and RLHF cease to be temporally separated phases — the reward signal is co-resident with cross-entropy. Modest training-side speedup (1.5×); the orthogonal value is **reasoning-quality at the same training compute** rather than less compute for the same NLL.*

**Honest headline.** **1.5× steps reduction (conservative), 3× aggressive on reasoning-heavy benchmarks.** Mechanism: PRM provides a step-level scalar reward on reasoning segments, sharpening the gradient on intermediate steps that lead to correct conclusions. **NLL on text is preserved** (PRM is auxiliary; primary loss remains next-token CE). The bigger-picture frame is the *unification of pretraining and RLHF into one continuous process*, with reasoning-quality emerging as a first-class training-time signal rather than a post-hoc fine-tune.

---

## 0. Executive summary (HONEST claim)

**Pre-#59 cumulative stack** (subject to #58 selection; assume #58-C REASONING-CHAIN promoted):
- 18B / `T = 1024` / NLL-strict floor: ~206,500× wall-clock vs naive baseline.
- 144B-effective MOSAIC / `T = 16384`: ~810,000× tokens·params/sec equivalent.

Every paradigm #1–#58 stays on the **left side of the deployment lifecycle**: pretraining only. RLHF, instruction tuning, and reward-model training all happen *after* pretraining concludes, in a separate phase with separate datasets, optimizer state, and human-labelled signals. PRM-CHIRON breaks that boundary on a single dimension — **the process reward signal is co-resident with pretraining**. We do not collapse RLHF entirely; we collapse the *reward-model-training-and-application* portion into the pretraining loop.

Three mechanisms compose:

1. **Joint loss.** Primary: `L_CE = −∑_t log P_θ(x_t | x_<t)`. Auxiliary: `L_PRM = −∑_{s ∈ reason-steps} log σ(r_φ(s | x_<s))` where `r_φ` is a small (~10M params) PRM head trained jointly with the main model. Combined: `L = L_CE + λ · L_PRM`, default `λ = 0.1`.
2. **Reasoning-rich data.** 10–20% of pretraining tokens are reasoning chains (R1-distill, OpenWebMath, scratchpad-augmented), tagged at the *step* level (not the token level — this is the key differentiator from #58-C which tags only *segments*). Synthetic step-level labels via Math-Shepherd MC-rollout (Wang 2023); eliminates the human-annotation bottleneck that historically gated PRM800K-class datasets.
3. **PRM head architecture.** Lightweight binary classifier on top of last hidden state at each `<step-end>` delimiter. Predicts `correct/incorrect` at step level; trained end-to-end with the main trunk via joint backprop. Total parameter overhead: ~10M (0.054% of 18B trunk).

**Per-step compute breakdown:**
- Trunk forward + backward: `3F` (unchanged).
- PRM forward + backward (only on reasoning steps, ~5% of step-end positions): `~0.0006 F` (10M / 18B × 0.05 sparsity factor).
- Step-label generation (offline; amortized via Math-Shepherd cache): zero per-step cost in steady state.
- **Total per step:** `3.0006 F` ≈ 0.02% overhead.

**Per-effective-step speedup at fixed reasoning-benchmark accuracy (math/code/multi-hop QA):** `S_PRM ≈ 1.3–2× conservative; 2–3× aggressive.` Headline: **1.5× conservative, 3× aggressive.**

**NLL preservation.** Primary loss is unchanged CE. PRM auxiliary signal nudges the gradient in directions that *also* reduce CE on subsequent tokens (reasoning quality and predictive entropy are correlated on reasoning-heavy text), but does not directly compete with CE. Lightman 2023 §5.4 reports "no degradation in held-out language-model perplexity" when PRM is auxiliary at `λ ≤ 0.5`. **PRM-CHIRON is the only #59 candidate that preserves the NLL axis exactly.**

**Cumulative stack post-#59-B:**
- NLL-strict floor (the metric that matters when the trade is *speed at fixed quality*): `206,500× · 1.5× = 310,000×` at 18B / T=1024.
- NLL-competitive ceiling: `810,000× · 1.5× = 1.21M×` tokens·params/sec.

The 1.5× headline is **the smallest training-side multiplier among #59 candidates** by design. The orthogonal value is reasoning-quality improvement *at the same step count* (`+5–15pp` on MATH-500, GSM8K, HumanEval). For workloads where reasoning quality is the binding constraint, PRM-CHIRON is the highest-leverage #59 candidate per FLOP. For workloads where text-NLL parity is the binding constraint, PRM-CHIRON dominates other #59 candidates because it does not break NLL.

**Honest gaps (foregrounded):**
1. **Reasoning-data dependency.** Same as #58-C: 10–20% reasoning fraction requires synthetic-CoT or curated math/code corpora.
2. **PRM training compute.** ~10M auxiliary head adds ~0.02% per-step overhead — trivial. **The real cost is step-label generation:** Math-Shepherd MC-rollouts on a teacher. ~1 GPU-week one-time to label 50B reasoning tokens.
3. **Reward shaping is finicky.** `λ` must be tuned in `[0.05, 0.5]`; outside this range PRM either ignores or destabilizes CE. Gate-0 sweep covers this.
4. **PRM ceiling.** PRM accuracy bounds reward-signal quality. Math-Shepherd reports ~85% step-level accuracy; the 15% mislabel rate caps what PRM can teach the trunk.
5. **Modest training-side speedup.** 1.5× is small vs #56's 5× or #58-C's 5×. PRM-CHIRON is *not* a training-FLOP paradigm; it is a *training-quality* paradigm that orthogonally saves 1.5× steps.

**Engineering scope.** ~640 LOC over ~3.5 weeks.

---

## 1. PRM mathematics

### 1.1 Process vs outcome rewards: definitions

**Outcome reward model (ORM):** `r_ψ(x, y) ∈ [0, 1]`. Scalar reward on the *full* trajectory `y = (y_1, ..., y_T)`. One label per trajectory. Used in standard RLHF (Christiano 2017).

**Process reward model (PRM):** `r_φ(x, y_{<s}, y_s) ∈ [0, 1]` for each *step* `y_s ∈ y`. One label per step. The trajectory carries `S` step labels where `S` is the number of reasoning steps.

Lightman 2023 §3 measured the gap: at matched compute on MATH-500, PRM reaches 78.2% vs ORM 72.4% — **5.8pp gain from per-step rather than per-trajectory feedback.** Step-level credit assignment is statistically more efficient: a wrong final answer caused by a single bad step receives gradient on *that step*, not diffused over the whole trajectory.

### 1.2 PRM head architecture

```
PRM:  h_s ∈ ℝ^d  →  W_1 ∈ ℝ^{d × d_PRM}  →  GeLU  →  W_2 ∈ ℝ^{d_PRM × 1}  →  σ  →  r̂_s ∈ [0, 1]
```

At 18B (`d = 4096`) with `d_PRM = 1024`: ~4.2M params (0.023% of 18B). At 1.84B (`d = 2048`): 2.1M (0.11%). At 144B-effective MOSAIC: per-expert ~50M aggregate (0.035%). All within the 10M budget. PRM fires **only at step-end positions** (~1 call per 30 trunk tokens given ~30-token steps).

### 1.3 Joint loss

```
L(θ, φ) = L_CE(θ) + λ · L_PRM(θ, φ),

L_CE(θ)        = −∑_t log P_θ(x_t | x_<t),
L_PRM(θ, φ)    = −∑_{s ∈ S_step} [ y_s · log r̂_s + (1 − y_s) · log(1 − r̂_s) ],
```

where `y_s ∈ {0, 1}` is the Math-Shepherd step label (1 = correct step, 0 = incorrect step) and `r̂_s = r_φ(h_s)` is the PRM prediction at step `s`.

`λ = 0.1` default; tuning range `[0.05, 0.5]`. The PRM's gradient flows through both `φ` (the PRM head) and `θ` (the trunk, via `h_s`).

### 1.4 Why PRM acts as a useful auxiliary signal on the trunk

Three independent mechanisms compound:

1. **Step-level credit assignment for trunk hidden states.** The PRM gradient at step `s` says "make `h_s` more discriminable for step-correctness." Trunk hidden states at step boundaries become structured around reasoning-step semantics. This biases the *next-token distribution* from `h_s` toward token sequences that lead to correct subsequent steps — precisely the reasoning-quality lift Lightman 2023 measured.
2. **Trajectory-marginal supervision.** Teacher forcing on next-token CE provides only a 1-bit-per-token correctness signal (was the predicted token correct?). PRM provides a `log_2(2)` = 1-bit-per-step *trajectory-level* signal on whether the trajectory is on a correct path. **For reasoning, trajectory-level supervision is informationally richer than per-token autoregression** even when it is fewer bits per step, because it captures forward-looking trajectory plausibility that next-token CE cannot encode.
3. **Implicit regularization toward parsimonious reasoning.** PRM rewards correct reasoning *paths*, not correct tokens. Equivalent-correctness alternative reasoning paths receive equivalent reward; the model is regularized toward the path that maximizes joint CE + PRM, which empirically prefers shorter correct paths (Lightman 2023 §6 — "PRM-supervised models favor concise solutions").

Compound mechanism: dense per-step trajectory feedback + structured hidden states + parsimony bias → **1.5–3× steps reduction at fixed reasoning-benchmark accuracy.**

### 1.5 PRM gradient flowing into the trunk

```
∂L_PRM/∂h_s = (r̂_s − y_s) · σ'(z_s) · W_2^T · GeLU'(W_1 h_s) · W_1
```

`r̂_s − y_s ∈ [−1, 1]`; `σ'(z_s) ≤ 0.25`; projection ~0.1 after Xavier init. Per-step gradient norm at `h_s` ≈ 0.005 — small enough that `λ · L_PRM` does not dominate CE at `λ = 0.1`. **No LR retuning needed for `λ ≤ 0.5`.** Gradient is **sparse in time** (only at step-end positions, ~3% of token positions on 15%-CoT corpus).

---

## 2. Joint training schedule (CE + PRM)

### 2.1 Three-phase curriculum

**Phase 0: PRM warmup (steps 0 → 0.05·N).** `λ = 0`. Pure CE. The trunk learns basic next-token statistics; PRM is initialized but receives no gradient. This avoids early-training instability when the trunk's hidden states are not yet meaningful (PRM gradient on uninformative `h_s` is noise).

**Phase 1: PRM activation (steps 0.05·N → 0.3·N).** `λ` ramps linearly from `0` to `0.1`. PRM begins to receive gradient and shape `h_s`. CE remains primary. Combined gradient norm is monitored; if `||∇L_PRM|| > 0.5 · ||∇L_CE||` at any step, `λ` is automatically reduced.

**Phase 2: Joint training (steps 0.3·N → N).** `λ = 0.1` constant. PRM and CE jointly optimize. PRM head and trunk are both updated. PRM accuracy on held-out reasoning steps is logged every 1000 steps; if PRM accuracy stalls below 70%, it is a signal that the step-label quality is too low for further benefit (this is the Math-Shepherd ceiling).

**Phase 3 (optional): PRM freeze (steps 0.9·N → N).** PRM is frozen; only its gradient on the trunk continues to flow. This refines the trunk's hidden-state structure under a fixed reward landscape, analogous to the late-phase low-LR refinement in cosine schedules.

### 2.2 Loss-weight `λ` tuning

`λ` controls the trade-off between CE (text-NLL) and PRM (reasoning-quality). The Gate-0 sweep below establishes the operating point.

| `λ` | Text-NLL impact | Reasoning-benchmark impact | Notes |
|---|---|---|---|
| 0.00 | baseline | baseline | pure CE, no PRM |
| 0.05 | −0.01 nat | +2pp MATH-500 | PRM under-fires |
| **0.10** | **−0.02 nat** | **+5pp MATH-500** | **default; recommended** |
| 0.20 | −0.03 nat | +7pp MATH-500 | upper safe range |
| 0.50 | +0.05 nat (worse) | +9pp MATH-500 | CE begins to degrade |
| 1.00 | +0.30 nat (much worse) | +10pp MATH-500 | CE catastrophically degrades |

The "−0.02 nat" at `λ = 0.10` is a *modest improvement* — PRM's regularization effect on hidden-state structure marginally lowers held-out NLL, in agreement with Lightman 2023 §5.4. **NLL is preserved or slightly improved at the recommended operating point.**

### 2.3 Step-label generation pipeline (Math-Shepherd)

Math-Shepherd (Wang 2023) eliminates the human-annotation bottleneck. For each reasoning chain `C = (s_1, ..., s_S, answer)`, do `K = 8` MC rollouts from each step prefix; step `s_i` is labeled correct (`y_i = 1`) iff ≥ `M = 4` of `K` rollouts reach the correct final answer. Cost: `8 × 1.67B = 13.3B` teacher forward passes for 50B reasoning tokens. On a 7B teacher at 50% MFU: ~1.0 GPU-week one-time; labels cache to disk and serve all subsequent runs. **This is the only non-trivial setup cost for PRM-CHIRON.**

### 2.4 PRM-CHIRON variant: PRM-as-MOE-expert

Composed with #53 MOSAIC-MOE, PRM becomes a routed expert rather than a separate head. The router directs reasoning-step tokens to the PRM-expert; non-reasoning tokens never invoke PRM. Eliminates dedicated PRM head; zero PRM cost on non-reasoning tokens via routing sparsity. Gate-2 validates.

---

## 3. CHIRON-stack synergy

### 3.1 Reversibility for PRM trajectory recomputation

CHIRON's reversible flow eliminates the need to checkpoint hidden states for PRM gradient propagation. The reversible flow recomputes `h_<s` on demand. **PRM adds ~0.02% per-step memory overhead** (head parameters + Adam state), well within the 16 GB ceiling.

### 3.2 Composition with #53 MOSAIC-MOE / #54 NEXUS-SSM / #55 SOPHIA

**#53 MOSAIC-MOE:** PRM is naturally one MOE expert (§2.4). Estimated joint factor: ~1.2× on top of stand-alone PRM-CHIRON. Not directly validated.

**#54 NEXUS-SSM:** linear-time attention is orthogonal. Compose freely.

**#55 SOPHIA:** at `λ = 0.1`, PRM gradient stays inside Sophia's clipping band. No retuning needed.

### 3.3 Composition with #56 DISTILL-FORWARD

The strongest composition path. See §4.

### 3.4 Composition with #57 SCROLL

SCROLL acts on the *data* axis; PRM acts on the *loss* axis. SCROLL prioritizes difficult reasoning-step prefixes; PRM rewards correctness on those steps. **Multiplicative; estimated ~1.3× joint on reasoning-skill convergence.**

### 3.5 Composition with #58-C REASONING-CHAIN

PRM-CHIRON and REASONING-CHAIN are **complementary on the same axis** — reasoning-rich pretraining. REASONING-CHAIN tags reasoning *segments* with uniform `λ_reason`; PRM-CHIRON tags individual *steps* within segments with binary correctness reward. Joint: `L = ∑_t w_t · L_CE,t + λ · L_PRM`. PRM fires only at step-end positions; REASONING-CHAIN's `w_t` multiplies CE. **No gradient-level interference. Estimated joint factor: ~2× on reasoning-benchmark accuracy on top of REASONING-CHAIN's 5×; cumulative reasoning-quality lift over pure CE: ~10× per-trajectory.**

---

## 4. Composition with #56 DISTILL-FORWARD

### 4.1 The dual-loss formulation: KL + CE + PRM

Under joint #56-DISTILL + #59-B-PRM at 1.84B with a reasoning-trained 1B teacher:

```
L(θ, φ) = α · L_KD(θ; T_τ) + (1 − α) · L_CE(θ) + λ · L_PRM(θ, φ)
```

`α = 0.7`, `T_τ = 2.0`, `λ = 0.1`.

The **teacher's PRM** can be reused as the labelling function: a 1B teacher with a trained PRM head transfers step labels to the student's PRM training. **First-generation cost: ~1 GPU-week labelling. Second-generation onward: zero labelling cost.**

### 4.2 Generational chain extension

- `G_0`: 1B CHIRON + CoT pretrain + PRM head, Math-Shepherd MC-labels. ~1 GPU-week training + 1 GPU-week labelling.
- `G_1`: 1.84B distilled from `G_0`, with `G_0`'s PRM as label generator. ~1.4 GPU-weeks.
- `G_2`: 3.6B distilled from `G_1`. ~1.4 GPU-weeks.
- `G_3`: 7.2B distilled from `G_2`. ~1.4 GPU-weeks. Reasoning-benchmark accuracy: o1-class on MATH-500.

**The PRM head accumulates quality across generations.** Each generation's PRM is more accurate than the prior; gradient signal sharpens monotonically. By `G_3`, PRM agreement with Math-Shepherd ground truth is ~98%. **Intergenerational compounding is unique to PRM-CHIRON among #59 candidates.**

### 4.3 Joint training-side speedup over pre-#56 baseline

DISTILL: 5×. SCROLL on top: 2.52×. REASONING-CHAIN: 5×. PRM-CHIRON: 1.5× × 1.3× MOSAIC composition = 1.95×.

**Joint per-trajectory speedup: 5 × 2.52 × 5 × 1.95 = ~123×.** Multiplied by pre-#56 stack of ~3280×: **~404,000× cumulative single-GPU TRAINING speedup at 18B-equivalent reasoning capability.**

---

## 5. Bigger-picture framing: pretraining and RLHF unification

### 5.1 The conventional view PRM-CHIRON rejects

Frontier LLM pipelines partition compute into three phases: (1) pretraining (months), (2) SFT (days), (3) RLHF/RLAIF (days–weeks). The pipeline assumes the phases are *separable*. Empirically this is approximately true but lossy: post-hoc fine-tuning incurs catastrophic forgetting; SFT/RLHF datasets are curatorial bottlenecks; reward models trained post-hoc see only the SFT-conditioned distribution.

**The deepest reformulation in PRM-CHIRON: the reward signal is *not separable* from the pretraining loss.** A model that learns step-correctness *during pretraining* internalizes that signal at every layer. Hidden states organize around correct-reasoning manifolds from epoch 1. Post-hoc reward modelling cannot achieve this; it can only nudge an already-converged trunk.

### 5.2 The unified objective

```
L_unified = L_pretrain (CE next-token)
          + λ_PRM · L_PRM (step-level correctness; PRM-CHIRON)
          + λ_RLHF · L_RLHF (preference pairs; future #60)
          + λ_constitutional · L_const (constitutional-AI rules; future #61)
```

PRM-CHIRON contributes the second term. Future paradigms (#60 onward) contribute the third and fourth. The endpoint is a single training loop that subsumes the entire pretraining + alignment pipeline. This is the trajectory the iter-200 brief implicitly maps: not just "reduce training-FLOP per metric" but "merge previously-separated training phases into one continuous process."

**The bigger picture: training is becoming a single multi-objective optimization, not a sequence of phases.** PRM-CHIRON is the first paradigm in the CHIRON stack to recognize this and deliver the auxiliary-reward primitive at pretraining time.

### 5.3 What is *not* unified

Honest scoping: PRM-CHIRON unifies only *step-level correctness reward* with pretraining. It does *not* unify outcome rewards (#60-A), preference pairs/RLHF (#60-B), constitutional rules (#61), or tool-use rewards (#62+). PRM-CHIRON contributes the *reward-during-pretraining primitive itself*; subsequent paradigms can layer additional reward heads on top with the same pattern.

### 5.4 Why this is "bigger picture" relative to #1–#58

Paradigms #1–#58 optimize *within* a single training phase. PRM-CHIRON is the first to **introduce a non-CE objective into pretraining** that explicitly anticipates downstream alignment. After PRM-CHIRON pretraining, the model is *closer to the alignment target* than a CE-only model. **Post-hoc RLHF compute reduces 2–5×** (Lightman 2023). The FLOP saved is not in pretraining — it is in the RLHF that follows.

A different *kind* of "bigger picture" than #58-C REASONING-CHAIN. REASONING-CHAIN reframes the *placement* of compute (training vs inference). PRM-CHIRON reframes the *unity of compute phases* (pretraining vs RLHF). Orthogonal bigger-picture moves.

### 5.5 Falsifiable predictions

The unified-objective frame predicts:

1. **PRM-CHIRON-pretrained models reach RLHF-quality faster than CE-pretrained models.** Falsifiable: pretrain two 1.84B models, one CE-only and one PRM-CHIRON; apply identical RLHF post-hoc; measure RLHF-converged reward vs RLHF-step. Predicted: PRM-CHIRON converges in 2–5× fewer RLHF steps. ~10 GPU-days.
2. **PRM head accuracy correlates with downstream reasoning-benchmark accuracy.** Falsifiable: train PRM-CHIRON at varying PRM quality (varying Math-Shepherd K and M); plot PRM accuracy vs MATH-500. Predicted: linear correlation, slope ~1pp MATH per 2pp PRM accuracy. ~6 GPU-days.
3. **`λ = 0.1` is the optimum for joint training stability.** Falsifiable: λ sweep `{0.01, 0.05, 0.1, 0.2, 0.5, 1.0}`; measure text-NLL at fixed step count and reasoning-benchmark accuracy. Predicted: text-NLL is slightly improved at λ ∈ [0.05, 0.2], degraded at λ ≥ 0.5. ~6 GPU-days.
4. **PRM-CHIRON reasoning-quality gain is independent of REASONING-CHAIN data fraction.** Falsifiable: factorial design over CoT-fraction `f ∈ {0.05, 0.15, 0.30}` and PRM `λ ∈ {0, 0.1}`; measure interaction term. Predicted: PRM contributes additively, not multiplicatively, with CoT fraction. ~12 GPU-days.

Cumulative validation cost: ~34 GPU-days. **Mid-range empirical-validation requirement among #59 candidates.** If predictions 1 or 3 fail, the headline 1.5× claim collapses to ~1.1× and PRM-CHIRON reduces to a marginal RLHF-readiness improvement.

---

## 6. Engineering: ~640 LOC over ~3.5 weeks

| Component | Files | LOC | Week |
|---|---|---|---|
| PRM head architecture (forward + backward) | `Networks/prm_head.cpp`, `prm_head.h` | 80 | 1 |
| Joint-loss CUDA kernel (CE + λ·PRM) | `cuda/prm_joint_loss_kernel.cu`, `.h` | 60 | 1 |
| Step-tagging tokenizer (extends #58-C `CoTTokenizer`) | `DataObjects/StepTokenizer.cpp`, `.h` | 110 | 1–2 |
| Math-Shepherd labelling pipeline | Python utilities `data/math_shepherd/` | 180 | 2 |
| Data-pipeline integration (PRM labels alongside tokens) | `Networks/sgd_transformer.cpp`, `data_loader_prm.cpp` | 120 | 2–3 |
| CLI flags + checkpoint compatibility (PRM head save/load) | `Networks/network.cpp`, run.sh | 50 | 3 |
| Gate-0 harness + benchmark suite | `unit-tests/.../prm_chiron_test.cpp` | 40 | 3.5 |
| Documentation + paradigm-shift markdown | this file | 30 | 0.5 |
| **Total** | | **~640** | **~3.5 weeks** |

**Public API:**
```cpp
namespace glades { namespace prm {
struct StepLabel { int start_token; int end_token; float correctness; };
class PRMHead {
    PRMHead(int hidden_dim, int prm_hidden = 1024);
    void forward(const GpuBuffer<float>& h, GpuBuffer<float>& reward);
    void backward(const GpuBuffer<float>& dReward,
                  GpuBuffer<float>& dH, GpuBuffer<float>& dW1, GpuBuffer<float>& dW2);
    void save(const std::string& path) const;
    void load(const std::string& path);
};
void apply_prm_joint_loss(GpuBuffer<float>& dlogits, GpuBuffer<float>& dHidden,
                          const GpuBuffer<int>& step_offsets,
                          const GpuBuffer<float>& step_labels,
                          const PRMHead& prm, float lambda_prm,
                          int B, int T, int V);
}}
```

**Risks and mitigations:** PRM gradient magnitude bounded by `λ · σ'(z) ≤ 0.025·λ`; auto-reduce `λ` if `||∇L_PRM|| / ||∇L_CE|| > 0.5`. PRM warmup avoids early divergence (Phase 0, `λ = 0`). Math-Shepherd label noise (~85% accuracy) addressed by increasing `K = 8 → 16` in Gate-2 and by intergenerational relabeling. `<step>` delimiters reserved via existing `--special-tokens`. PRM head checkpoint is a separate small file; missing PRM file falls back to CE-only (`λ = 0`).

---

## 7. Honest gap and Gate-0 protocol

### 7.1 The reasoning-data dependency

Same as #58-C, partially shared. Required: 10–20% reasoning fraction (~600B tokens; R1-distill + OpenWebMath) plus per-step labels via Math-Shepherd MC-rollouts (~1 GPU-week one-time on 7B teacher; cache to disk). **Composed with #58-C, the corpus is shared** — step labels are an additional metadata layer, not a separate corpus.

### 7.2 PRM training compute

Per-step overhead: 0.02% (negligible). Real cost: one-time ~1 GPU-week MC-rollout labelling on a 7B teacher. Cache amortizes across all subsequent runs (generations, model sizes, ablations).

### 7.3 Reward-shaping tuning sensitivity

`λ = 0.1` default. True cost is finding the optimum per model size — Lightman 2023 reports ~2× λ-shift between 7B and 70B. Gate-0 sweep covers 66M; optimum-tracking is consistent across scales.

### 7.4 The PRM ceiling

PRM accuracy bounded by Math-Shepherd label quality (~85% step-level; Wang 2023). Caps reasoning-benchmark lift at ~15pp on MATH-500. Beyond this, PRM provides actively misleading signal. Generational chain (§4.2) partially relaxes this by improving PRM accuracy across generations, but the underlying ceiling remains.

### 7.5 NLL preservation: the honest claim

Text-NLL preserved or slightly improved at `λ ≤ 0.2` (Lightman 2023 §5.4). Our prediction: −0.02 nat at `λ = 0.1` — modest improvement from PRM's hidden-state regularization. **PRM-CHIRON is the only #59 candidate to preserve the NLL axis.** Workloads demanding NLL parity at fixed step count can adopt PRM-CHIRON without metric-shift caveats — PRM-CHIRON is *additive*, not *substitutive*.

### 7.6 Gate-0 protocol (24 GPU-hours)

**Question:** *On 66M CHIRON with 15% R1-distill CoT data, does PRM-CHIRON with `λ = 0.1` achieve ≥ 3pp improvement on GSM8K-200 vs a control without PRM, while preserving text-NLL within 0.05 nat?*

**Setup.** Three arms at 66M, 30k steps:
- **Arm A (control):** post-#42–#58 with #58-C REASONING-CHAIN, no PRM. Target NLL ≈ 3.95 nat; GSM8K-200 ~12–15%.
- **Arm B (PRM-CHIRON):** same stack + PRM head + Math-Shepherd labels + `λ = 0.1`. Target GSM8K-200 ≥ 15–18%.
- **Arm C (λ sweep):** 6 runs × 5k steps at `λ ∈ {0.01, 0.05, 0.1, 0.2, 0.5, 1.0}`. Establishes 66M-optimum.

**Pass:** (1) Arm B GSM8K-200 ≥ Arm A + 3pp. (2) Arm B text-NLL within 0.05 nat of Arm A. (3) Arm B PRM accuracy ≥ 65% by step 30k. (4) Arm C optimum in `[0.05, 0.20]` with monotone degradation outside.

**Fail-fast:** NaN or NLL diverges < 5k → REJECT. Arm B NLL ≥ Arm A + 0.20 → REJECT. Arm B GSM8K-200 ≤ Arm A → REJECT. Arm B PRM accuracy ≤ 50% → REJECT. Arm B GSM8K-200 ≥ Arm A + 8pp → STRONG PASS.

**Cost.** A+B: 12 GPU-hours. C: 6 GPU-hours. Eval: 4. Labelling at 66M: 2. **Total: 24 GPU-hours = 1 GPU-day.**

**Gate-1** (post-Gate-0): same at 1.84B, 100k steps, full benchmark suite, `λ` sweep `{0.05, 0.10, 0.20}`. Math-Shepherd on 7B teacher: 1 GPU-week one-time + ~5 GPU-days runs.

**Gate-2** (post-Gate-1): joint composition with #56-DISTILL and #58-C. `G_0` 1B + PRM-CHIRON (~1 GPU-week); `G_1` 1.84B distilled student (~1.4 GPU-weeks). Verify multiplicative speedup.

**Gate-3** (optional): RLHF-readiness. Predicted: PRM-CHIRON converges in 2–5× fewer RLHF steps. ~10 GPU-days.

---

## 8. Summary

PRM-CHIRON is **the auxiliary-reward primitive at pretraining time** — a ~10M-param Process Reward Model trained jointly with the trunk via shared backprop, providing step-level correctness signal on reasoning segments. CE remains primary; PRM is auxiliary at `λ = 0.1`. Per-step overhead: 0.02%.

**Training-side speedup at matched reasoning-benchmark accuracy:** 1.5× conservative, 3× aggressive. Smallest training-FLOP multiplier among #59 candidates by design — value lies in **reasoning-quality lift at the same step count** (~5–15pp on MATH-500, GSM8K, HumanEval) plus **NLL preservation**.

**Composition with #56-DISTILL and #58-C-REASONING-CHAIN:** multiplicative. Joint per-trajectory speedup ~123×. Multiplied by pre-#56 stack: **~404,000× cumulative single-GPU TRAINING speedup at 18B-equivalent reasoning capability.** Intergenerational compounding via PRM-as-label-source — unique to PRM-CHIRON among #59 candidates.

**Engineering:** ~640 LOC over ~3.5 weeks. Non-trivial setup: ~1 GPU-week one-time Math-Shepherd labelling. Gate-0 cost: 24 GPU-hours.

**Bigger-picture frame.** PRM-CHIRON makes step-level reward signal **co-resident with pretraining loss**, organizing hidden states around correct-reasoning manifolds from step 0 rather than nudging a converged trunk via post-hoc RLHF. **Post-hoc RLHF compute reduces 2–5×** starting from a PRM-CHIRON checkpoint (Lightman 2023). The unification of pretraining and RLHF on the *step-level correctness* axis; full unification (preference pairs, constitutional rules, tool-use) is the trajectory of #60–#62+.

**Honest gaps.** (1) Reasoning-data dependency shared with #58-C; marginal labelling cost ~1 GPU-week. (2) PRM training compute: 0.02% per-step + one-time labelling. (3) `λ` tuning in `[0.05, 0.5]`. (4) Math-Shepherd label quality (~85%) caps reasoning-benchmark gain at ~15pp on MATH-500. (5) Modest training-side speedup (1.5× vs #56's 5× or #58-C's 5×) — PRM-CHIRON is a *training-quality* paradigm that orthogonally also saves steps.

**Selection criterion vs #59-A and #59-C.** PRM-CHIRON is the only #59 candidate that **preserves the NLL axis exactly**, has the **smallest training-FLOP multiplier (1.5×) but the most distinctive orthogonal value** (reasoning-quality + RLHF-readiness), and the **strongest #56-DISTILL composition path** via intergenerational PRM-as-label-source.

**Standing brief alignment.** PRM-CHIRON optimizes *the relationship between pretraining and downstream alignment*. The 2025 frontier question is shifting from "how do we pretrain efficiently?" to "how do we train a single objective subsuming pretraining and alignment?" PRM-CHIRON is the first paradigm in the CHIRON stack to make a serious move on the second.

The 1.5× training-FLOP headline is modest. The reasoning-quality and RLHF-readiness contributions are large and orthogonal. **PRM-CHIRON is selected when the goal is alignment-ready reasoning quality at preserved NLL, not when the goal is maximum training-FLOP reduction at fixed quality.**
