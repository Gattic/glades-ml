# Paradigm Shift #60 Candidate B — SELF-PLAY-CHIRON (competitive self-improvement via debate / two-copy adversarial training)

**Status:** candidate-B design for paradigm shift #60. **Recommended action: REJECT for #60; reserve for a future safety/alignment paradigm (#62+).**
**Date:** 2026-05-08 (Ralph-loop iter 200+, post-#59 selection, under the iter-200 brief).
**Predecessors:** `PARADIGM_SHIFT_59_CANDIDATE_B_PRM_CHIRON.md` (auxiliary-reward-during-pretraining precedent), `PARADIGM_SHIFT_58_CANDIDATE_C_REASONING_CHAIN.md`, `PARADIGM_SHIFT_56_CANDIDATE_B_DISTILL_FORWARD.md` (multi-model joint training infrastructure), `BEYOND_CHIRON.md` §2.3 (NLL benchmark protocol).
**Axis:** **competitive multi-copy training dynamics.** Two model copies generate opposing arguments on reasoning prompts; a discriminator judges; both copies update on the discriminator's signal; periodic weight-merging prevents divergence. The training signal is disagreement-resolution, **not** autoregressive likelihood. NLL preservation is **not** a property of the formulation.

**References.** Silver et al. *Mastering the game of Go without human knowledge.* Nature 550 (2017) — AlphaGo Zero. Irving, Christiano, Amodei. *AI safety via debate.* arXiv:1805.00899 (2018). Bai et al. *Constitutional AI.* arXiv:2212.08073 (2022) — self-critique at fine-tuning. Burns et al. *Weak-to-Strong Generalization.* arXiv:2312.09390 (2023). Singh et al. *ReST^EM.* arXiv:2312.06585 (2023). Yuan et al. *Self-Rewarding Language Models.* arXiv:2401.10020 (2024) — model-as-its-own-judge at fine-tuning scale.

**Tagline.** *AlphaGo defeated Lee Sedol via self-play; the analogy to LLM pretraining is tempting and largely wrong. Games admit a closed adversarial structure with a lossless ground-truth oracle (rules + win/loss); language modeling does not. SELF-PLAY-CHIRON is highly speculative at LLM pretraining scale and breaks NLL preservation. Recommend rejection for #60.*

**Honest headline.** **Speculative 2-3× reasoning-convergence speedup, conditional on a discriminator that does not exist; NLL preservation violated; no reproducible LLM-scale precedent.** The training signal is disagreement-resolution, not next-token CE — NLL drift is expected and unbounded. Empirical evidence at LLM scale is **thin to absent** (mostly games and post-hoc fine-tuning). **Recommend REJECT for #60.**

---

## 0. Executive summary (HONEST claim — recommendation: REJECT)

Pre-#60 cumulative stack (assume #59-B PRM-CHIRON promoted): 18B / `T = 1024` / NLL-strict floor ~310,000× wall-clock vs naive baseline.

SELF-PLAY-CHIRON proposes a fourth axis beyond data (#57 SCROLL), loss (#56 DISTILL, #59-B PRM), and sampling (#58-C REASONING-CHAIN): the **competitive-dynamics axis**. Two copies `θ_A, θ_B` are trained jointly; a discriminator `ψ` judges their competing answers; both receive gradient on the disagreement-resolution signal.

The paradigm is structurally analogous to GAN training (#43 GANYMEDE) and AlphaGo policy-network self-play, but **the analogy is load-bearing in ways that do not transfer to language modeling**:

1. **Games have a lossless ground-truth oracle.** Go's rules + win/loss are perfect, free, exhaustive. **Language has no such oracle.**
2. **Adversarial games are closed systems** with a fixed-point convergence theorem. Language has no analogous closed structure; the "opponent" is the underlying linguistic distribution, which does not respond to the model's policy.
3. **AlphaGo's discriminator is the game itself.** SELF-PLAY-CHIRON requires a *learned* discriminator at least as capable as the generators — the same problem as training a strong LLM in the first place. Circular dependency.
4. **NLL preservation is not provided.** The training signal is disagreement-resolution, not next-token likelihood. Lightman 2023-style "auxiliary reward preserves CE" guarantees do **not** apply because the signal is *primary*, not auxiliary.

**Per-step compute:** ~6F vs 3F for single-copy CE. **2× FLOP cost minimum.** Per-effective-step speedup: speculative 2-3× with no LLM-scale empirical floor. Net wall-clock after 2× FLOP overhead: 1-1.5× *if* the conjecture holds.

**NLL preservation: NOT provided.** Expected drift over 30k steps at recommended discriminator confidence: `+0.10 to +0.30 nat` — well outside the `0.05 nat` tolerance #59-B achieves.

**Honest gaps (rejection rationale):**

1. **No LLM-scale empirical precedent for self-play *pretraining* acceleration.** Validated for *games* (AlphaGo, AlphaZero) and post-hoc *fine-tuning* (Constitutional AI, ReST^EM, Self-Rewarding LMs). **No published work demonstrates self-play acceleration of pretraining at LLM scale.**
2. **Discriminator capability paradox.** A discriminator that judges argument quality must be at least as capable as the generators — the same problem as training a strong LLM. Weak-to-strong bootstrapping (Burns 2023) is research-frontier, not engineering.
3. **NLL not preserved by construction.**
4. **Mode collapse is structurally likely.** Copies near-identical at init; without strong regularization they collapse to a single mode (same dynamics class as GAN collapse). Periodic weight-merging mitigates but defeats the purpose.
5. **Speculative C1.** 2-3× conjectured by analogy to AlphaGo. Closest LLM-scale data: Yuan 2024 reports +5-9% on AlpacaEval-2 — a quality lift at fixed compute, not a steps reduction.
6. **The legitimate use is alignment, not pretraining.** Forcing SELF-PLAY-CHIRON-class machinery into pretraining acceleration conflates training axes.

**Engineering scope:** ~1100 LOC / ~6 weeks; major rework risk if Gate-0 fails. **Recommended action: REJECT for #60.** Reserve for the safety/alignment lane (#62+).

---

## 1. Self-play mathematics (debate paradigm, two-copy training)

### 1.1 Notation and dynamics

Let `θ_A, θ_B ∈ ℝ^P` be two trunk copies sharing the CHIRON architecture; let `ψ ∈ ℝ^Q` be a discriminator (typically `Q ≈ 0.05P`). For each training prompt `x`:

```
y_A ~ P(· | x; θ_A)            [Copy A's argument; 100-500 generated tokens]
y_B ~ P(· | x; θ_B)            [Copy B's argument; different sampling seed]
r̂ = ψ(x, y_A, y_B) ∈ [0, 1]   [Discriminator: P(A more correct than B)]
y* ∈ {0, 1}                    [Ground-truth label, IF available]
```

`r̂` is the **only** training signal for the trunks; `y*` is sparse (often unavailable on synthetic CoT) and trains the discriminator only.

### 1.2 The disagreement-resolution loss

REINFORCE-style policy-gradient with discriminator reward:

```
L_A = -E_{y_A ~ P(·|x; θ_A)} [ log P(y_A | x; θ_A) · (r̂ - 0.5) ]
L_B = -E_{y_B ~ P(·|x; θ_B)} [ log P(y_B | x; θ_B) · (0.5 - r̂) ]
L_ψ = -E_{(x, y_A, y_B, y*)} [ y* · log r̂ + (1 - y*) · log(1 - r̂) ]

min_{θ_A, θ_B, ψ}  L_A + L_B + L_ψ
```

**Critical structural difference from #59-B PRM-CHIRON:** `log P(y_A | x; θ_A)` is CE on the *copy's own sampled output*, not on the training corpus. In PRM-CHIRON, CE on corpus tokens remains primary; in SELF-PLAY-CHIRON, CE on *self-generated* tokens is the only autoregressive signal, weighted by a reward the corpus does not constrain.

### 1.3 Periodic synchronization

Without regularization the copies drift apart, eventually losing the disagreement signal (one always wins). Every `K = 1000` steps:

```
θ̄ = 0.5 · θ_A + 0.5 · θ_B,            [EMA average]
θ_A ← θ̄ + ε_A,   θ_B ← θ̄ + ε_B,       ε ~ N(0, σ²)   [re-perturb, σ ≈ 0.01]
```

Heuristic only; **no theoretical guarantee that EMA + re-perturbation preserves the disagreement signal across many cycles.** The long-horizon dynamics are not understood.

### 1.4 Why the AlphaGo analogy fails for language modeling

| Property | AlphaGo / Go | LLM pretraining |
|---|---|---|
| Ground-truth oracle | Rules + win/loss: lossless, free, exhaustive | None: human judgment is sparse and noisy |
| Closed adversarial structure | Two-player zero-sum with well-defined value function | Language is not a game; no zero-sum structure |
| Reward signal density | Every game terminates with ±1 | Rewards require a learned discriminator at every prompt |
| Self-play stationary distribution | Provably converges to Nash via Q-learning | No analogous convergence theorem |
| Discriminator vs generator capability | Game rules are *infinitely* more capable | Discriminator at best matches generator |

**AlphaGo's discriminator is the game itself, which is free.** SELF-PLAY-CHIRON requires a *learned* discriminator on language — a circular dependency AlphaGo does not face. The only resolution is bootstrapping from a weak discriminator (Burns 2023), but weak-to-strong is research-frontier, not engineering, and it is not clear the discriminator can improve faster than the generators it judges.

### 1.5 Mode collapse and the two-copy degeneracy

GANs collapse when the generator produces a single mode that fools the discriminator. SELF-PLAY-CHIRON's two-copy structure has the dual problem: the copies collapse when both produce identical outputs (`y_A ≡ y_B`), at which point `r̂ ≡ 0.5` and gradient vanishes. Three structurally-likely failures: (1) **two-copy convergence** — both copies converge to the same policy; disagreement vanishes; training stalls; (2) **discriminator overfit** — the discriminator memorizes the small set of generated arguments and gives a perfect signal that does not generalize; (3) **adversarial drift** — one copy finds a high-probability-but-wrong argument the discriminator misjudges as correct; both chase it; both diverge from the language-modeling objective.

#43 GANYMEDE addressed some of these via spectral normalization, but those techniques transferred poorly outside small-scale image-GAN settings; effectiveness on language self-play is unestablished.

---

## 2. Quality discriminator design

The discriminator is the load-bearing component; its design is more difficult than the trunks themselves.

### 2.1 Discriminator architecture options

- **A: Pairwise classifier on frozen external teacher.** Requires a teacher *better than the trunk* — circular at SELF-PLAY-CHIRON's intended 18B scale; useful only sub-7B.
- **B: Co-trained from scratch.** Capability bounded by labelled data and trunk capability — circular. Empirically fails in vanilla GAN settings when both nets train from scratch.
- **C: Bootstrapped from #59-B PRM head.** Leverages step-level correctness. PRM is per-step; SELF-PLAY-CHIRON requires per-argument pairwise judgment — different output structure. Needs a non-trivial pairwise-aggregation head.
- **D: Constitutional discriminator (Bai 2022).** Rule-based scoring. Less circular but covers only a narrow slice of argument quality; reasoning correctness on MATH-500 is not rule-checkable.

**Recommended (least bad): D + C composition.** Rule-based for stylistic correctness, bootstrapped PRM for step-level reasoning. This is essentially Constitutional AI fine-tuning (Bai 2022 §4) — the only design with **any** LLM-scale precedent. Crucially, that precedent is **fine-tuning, not pretraining.**

### 2.2 The capability ceiling

Typical regime: `Q ≈ 0.05P` (1B discriminator at 18B trunk; 100M at 1.84B). **Once the trunks exceed the discriminator's judgment capability, the self-play signal becomes noise.** In AlphaGo this never bites (game rules are infinitely capable). In SELF-PLAY-CHIRON it bites at exactly the point where the trunks would otherwise be improving fastest.

### 2.3 Where do discriminator labels come from?

(1) Human pairwise preferences — gold but ~$5/comparison, ~10-100k needed. (2) Synthetic ground-truth from MATH/GSM8K/HumanEval — free but restricted to checkable answers (~1B tokens). (3) MC-rollout consensus (Math-Shepherd) — restricted to step-level reasoning. (4) Self-consistency (Wang 2022) — noisy and circular at high trunk capability.

**No source provides a discriminator capable of judging a strong-LLM argument it has not seen before.** This is the fundamental problem.

---

## 3. The honest NLL preservation gap

### 3.1 SELF-PLAY-CHIRON does not preserve NLL by construction

#59-B PRM-CHIRON preserves NLL because PRM is *auxiliary*: `L = L_CE + λ · L_PRM`, with `L_CE` always present on corpus tokens. SELF-PLAY-CHIRON's loss does not include any term equal to `L_CE` on training-corpus tokens; instead it has CE on *self-generated* tokens, weighted by a reward signal. **The corpus next-token-prediction objective is absent.** Whatever held-out NLL the model achieves is incidental, not optimized.

### 3.2 Expected NLL drift across confidence regimes

1. **Conservative (rare disagreement, low `r̂` confidence).** When `r̂ ≈ 0.5` most of the time, trunks receive near-zero gradient and NLL stays near initialization. **Drift ~0 nat, but no learning either.** Defeats the purpose.
2. **Moderate (occasional confident disagreement).** When `r̂ ≈ 0.7-0.9` on ~10% of prompts. **Expected drift +0.10 to +0.30 nat over 30k steps.** Drift is positive (NLL worsens) because the optimization is no longer minimizing corpus likelihood.
3. **Aggressive (frequent confident disagreement).** When `r̂ ≈ 0.9+` on most prompts, the model rapidly adapts to discriminator preferences. **Expected drift +0.50 to +2.00 nat.** Catastrophic NLL failure.

**No regime preserves NLL within the ≤0.05 nat tolerance #59-B achieves.**

### 3.3 Hybrid CE + SP formulation

A natural fix retains CE as auxiliary: `L_hybrid = L_CE + λ_SP · (L_A + L_B)` with `λ_SP = 0.1`. This (in principle) preserves NLL. *But:* the disagreement-resolution gradient is now small relative to CE; the speedup is also small; the FLOP overhead (2× from running two copies) is not offset; **net wall-clock is slower than CE-only.**

The hybrid formulation **either preserves NLL and provides no speedup, or provides speedup and violates NLL**. There is no setting that satisfies both #60 selection criteria. At that point the paradigm reduces to "PRM-CHIRON with two model copies and a more expensive discriminator" — strictly dominated by #59-B.

---

## 4. The speculative speedup conjecture C1

**C1:** SELF-PLAY-CHIRON accelerates reasoning convergence by 2-3× via richer training signal vs single-copy CE pretraining at fixed step count.

### 4.1 Evidence for C1: analogy to games

AlphaGo Zero achieved superhuman Go in 40 days vs years of pretraining + self-play in original AlphaGo — ~10× wall-clock acceleration (Silver 2017). C1 imports this 10× as a 2-3× lower bound for LLM self-play.

**Problems with the analogy:** AlphaGo's "training" is policy-iteration; LLM pretraining is supervised learning on a corpus. AlphaGo's signal is binary win/loss; LLM signal is a learned probability. AlphaGo's data distribution is determined by self-play; LLM data distribution is the corpus, and self-play does not generate new corpus data — it generates new *opinions on existing data*, a fundamentally smaller signal.

### 4.2 LLM-scale precedents are post-hoc fine-tuning, not pretraining

| Paper | Setting | Reported gain | Pretraining or FT? | Mechanism |
|---|---|---|---|---|
| Yuan 2024 (Self-Rewarding LMs) | Llama-2-70B FT | +5-9% on AlpacaEval-2 | Fine-tuning (DPO) | Model judges its own outputs |
| Singh 2023 (ReST^EM) | PaLM 2-L FT | +1-3% on math | Fine-tuning | Self-distillation on filtered samples |
| Bai 2022 (Constitutional AI) | Anthropic models | No quantitative speedup | Fine-tuning (RLAIF) | Self-critique + RL on critique |
| Burns 2023 (Weak-to-Strong) | GPT-4 ⇐ GPT-2 | "PGR" 20-80% | FT analog | Weak supervisor; relevant to discriminator paradox |
| **Pretraining-scale acceleration** | — | — | — | **None published** |

### 4.3 Evidence against C1

(1) Constitutional AI's gains are post-pretraining alignment, not perplexity. (2) **GAN training (the structural analog) is notoriously slow** — vision GANs typically need 5-10× more compute than direct supervised training; **#43 GANYMEDE in our own stack reported ~1.5× *slowdown* before careful tuning.** Two-copy adversarial structure is a known compute-multiplier. (3) Mode collapse + discriminator overfitting appear in every adversarial training setting; LLM scale would not be exceptional.

### 4.4 Honest range estimate

If C1 holds: 2-3× per-effective-step → **net wall-clock 1-1.5× after 2× FLOP overhead**. Mode collapse: **1× (no gain)**. Catastrophic: **<1× (slows training)**. Expected value: ~1.0-1.2× wall-clock with high probability of net loss. Below threshold for a #60 paradigm shift.

---

## 5. Engineering scope (if implemented despite recommendation)

| Component | LOC |
|---|---|
| Two-copy training infrastructure (forward × 2, backward × 2, sync) | 250 |
| Discriminator architecture (pairwise head; bootstrapped from PRM) | 150 |
| REINFORCE policy-gradient kernel (CUDA) | 150 |
| Sample-generation + discriminator-judgment harness | 280 |
| Periodic synchronization + mode-collapse detector | 170 |
| CLI flags + checkpoint compatibility + Gate-0 harness | 100 |
| **Total** | **~1100 LOC / ~6 weeks** |

**Doubled scope vs PRM-CHIRON (640 LOC / 3.5 weeks).** Major rework risk if Gate-0 fails — most likely failure mode is discriminator collapse, requiring §2 redesign.

**Memory cost:** 2× trunk copies under bf16 at 18B exceeds the 16 GB ceiling — would force aggressive sharding or drop scale to 7-9B. **This alone disqualifies SELF-PLAY-CHIRON from the production stack.**

---

## 6. Why SELF-PLAY-CHIRON should be REJECTED for #60

### 6.1 Selection-criteria scoring

| Criterion | SELF-PLAY-CHIRON status |
|---|---|
| 1. Training speedup ≥1.5× | **CONJECTURED 2-3× per-effective-step; ~1.0-1.5× wall-clock after 2× FLOP overhead. Marginal at best.** |
| 2. NLL preservation ≤0.05 nat | **VIOLATED by construction; expected drift +0.10 to +0.30 nat at recommended settings.** |
| 3. LLM-scale empirical foundation | **ABSENT at pretraining scale; only post-hoc FT precedents.** |
| 4. Multiplicative composition with stack | **UNCLEAR; #59-B PRM-CHIRON overlaps the discriminator design.** |
| 5. Engineering scope ≤4 weeks | **VIOLATED; ~6 weeks with high rework risk.** |

**Fails 3 of 5 criteria.**

### 6.2 The structural mismatch

**SELF-PLAY-CHIRON is not a pretraining-acceleration paradigm.** It is a *judgment-distillation* paradigm — the goal is to teach a model to reason like the discriminator, not to maximize corpus likelihood. This is the right tool for **alignment**, not pretraining acceleration. Legitimate uses:

- **Safety/alignment.** Distilling rules or human preferences (Constitutional AI, Anthropic HH-RLHF). Discriminator paradox is bounded because it judges alignment, not factual correctness across all domains.
- **Reasoning-quality fine-tuning.** Self-Rewarding LMs (Yuan 2024), ReST^EM (Singh 2023). Post-pretraining lift on benchmarks.
- **Debate-style verification (Irving 2018).** Adversarial protocols at inference time for high-stakes outputs.

None of these are pretraining-acceleration. **Forcing SELF-PLAY-CHIRON into a #60 pretraining slot conflates training axes** and produces a worse paradigm than direct alternatives like #60-A or #60-C.

### 6.3 Reserved for #62+ in the safety/alignment lane

Natural homes for SELF-PLAY-CHIRON-class machinery:

- **#62 (hypothetical) SELF-PLAY-ALIGN.** Two copies generate opposing answers on alignment-relevant prompts; constitutional discriminator judges; both update toward the constitutionally-preferred response. NLL preservation is **not** the criterion; alignment-benchmark accuracy is. Discriminator paradox bounded by alignment-rule scope.
- **#63 (hypothetical) DEBATE-VERIFY.** Irving 2018's debate protocol as a *post-training* verification layer for high-stakes outputs (medical, legal, scientific). Inference-time only; no training cost.

These are legitimate paradigm-shift candidates in the alignment track, not pretraining-acceleration. The #60 selection should not be diluted by retrofitting them into the wrong axis.

### 6.4 What can be salvaged for #60

If the #60 axis is "competitive training dynamics," a strictly-superior alternative exists outside the candidate-A/B/C set:

**SELF-DISTILL-CHIRON (hypothetical #60-D).** One model copy `θ`; generate K candidate continuations; rank with PRM (already trained in #59-B); train on the highest-ranked continuation with auxiliary CE on corpus tokens. **No two-copy overhead; no discriminator design problem; NLL preserved via auxiliary `λ` weighting.** Essentially Self-Rewarding LMs (Yuan 2024) at pretraining scale, retrofitted into existing #59-B PRM infrastructure. ~400 LOC vs ~1100.

But this is a separate proposal; **SELF-PLAY-CHIRON itself remains rejected.** The salvageable kernel is the simpler single-copy ranking formulation, not the two-copy adversarial structure.

---

## 7. Gate-0 protocol (for completeness; predicted to confirm rejection)

If validation is requested despite the rejection recommendation, a 3-arm Gate-0 at 66M CHIRON / 15% R1-distill CoT / 30k steps establishes an empirical floor.

- **Arm A (control):** post-#59-B PRM-CHIRON, single copy. Target NLL ≈ 3.93 nat; GSM8K-200 ~17-20%.
- **Arm B (SELF-PLAY-CHIRON):** two 66M copies + bootstrapped PRM disc + EMA-sync every 1000 steps.
- **Arm C (hybrid CE + λ_SP·SP):** single copy, K=2 samples per prompt, `L_CE + 0.1·L_SP`.

**Pass:** Arm B GSM8K-200 ≥ A + 2pp; NLL within 0.10 nat of A (relaxed from 0.05 given known violation); no mode collapse (`θ_A,θ_B` cosine in [0.7, 0.99]); discriminator accuracy ≥ 60%.

**Fail-fast:** mode collapse (cosine > 0.99 for >5k steps) → REJECT. NLL drift > 0.30 nat → REJECT. Discriminator stuck ~50% → REJECT. Arm B GSM8K-200 ≤ A → REJECT.

**Predicted outcome:** **fail-fast on mode collapse or NLL drift within 5k-15k steps**, confirming structural mismatch. ~36 GPU-hours total. **Gate-1 not recommended.**

---

## 8. Summary

SELF-PLAY-CHIRON is **competitive multi-copy training dynamics** — two trunk copies, learned discriminator, disagreement-resolution gradient, periodic EMA-sync. Structurally analogous to AlphaGo self-play and Irving 2018 debate, but **the analogy is load-bearing in ways that fail for LLM pretraining**.

**Training-side speedup:** speculative 2-3× per-effective-step; net wall-clock 1-1.5× after 2× FLOP overhead; high probability of net loss via mode collapse or discriminator drift.

**NLL preservation:** **VIOLATED by construction.** Expected drift +0.10 to +0.30 nat at recommended settings. Hybrid CE+SP either preserves NLL with no speedup or provides speedup with NLL violation.

**Empirical foundation:** **ABSENT at LLM pretraining scale.** Closest precedents (Self-Rewarding LMs, ReST^EM, Constitutional AI) are post-hoc fine-tuning. AlphaGo's analogy fails on three structural properties: no ground-truth oracle, no closed adversarial structure, no discriminator-as-game-rules.

**Engineering scope:** ~1100 LOC over ~6 weeks. 2× memory cost disqualifies the paradigm from the 16 GB ceiling at 18B.

**Honest gaps:** (1) No LLM-scale precedent for pretraining acceleration via self-play. (2) Discriminator capability paradox. (3) NLL not preserved. (4) Mode collapse structurally likely. (5) Speculative 2-3× conjecture has no data. (6) Legitimate use case is alignment, not pretraining.

**Recommendation: REJECT for #60.** Fails 3 of 5 selection criteria and is structurally mismatched to the pretraining-acceleration axis. The right home for SELF-PLAY-CHIRON-class machinery is the **safety/alignment lane**, where the discriminator paradox is bounded by alignment-rule scope and NLL preservation is not the criterion. **Reserve as a future #62+ candidate (SELF-PLAY-ALIGN, DEBATE-VERIFY)** in the alignment paradigm track.

If the #60 selection insists on a competitive-dynamics axis candidate, **SELF-DISTILL-CHIRON** (single copy, K-sample ranking via #59-B PRM, top-1 retraining with auxiliary CE) strictly dominates SELF-PLAY-CHIRON on every criterion — but it is a separate proposal. The salvageable kernel of SELF-PLAY-CHIRON's idea is exactly that simpler formulation, not the two-copy adversarial structure.

**Standing brief alignment.** The iter-200 brief asks for *bigger picture* moves. SELF-PLAY-CHIRON is "bigger picture" in the wrong direction — it imports a games/alignment paradigm into a pretraining role where it is structurally mismatched. The honest assessment is that **the bigger picture for pretraining is multi-objective unification of CE + auxiliary signals (#59-B PRM-CHIRON) + smarter sampling (#58-C REASONING-CHAIN) + smarter data (#57 SCROLL)**, not adversarial dynamics. Self-play earns its place in the alignment lane, not the pretraining lane.

**Final recommendation: REJECT for #60. Reserve for #62+ alignment paradigm.**
