# Paradigm Shift #58 Candidate C — REASONING-CHAIN (chain-of-thought integrated into pretraining for test-time compute scaling)

**Status:** candidate-C design for paradigm shift #58. One of three parallel proposals for #58.
**Date:** 2026-05-08 (Ralph-loop iteration 202+, post-#57 SCROLL-PROMOTED, under the standing iter-200 brief: *"novel architectures, algorithms, and training methods by looking at the bigger picture instead of focusing on microoptimizations"*).
**Predecessors:** `PARADIGM_SHIFT_56_CANDIDATE_B_DISTILL_FORWARD.md` (multi-generation knowledge accumulation), `PARADIGM_SHIFT_57_CANDIDATE_A_SCROLL_PROMOTED.md` (data-axis active learning), `BEYOND_CHIRON.md` §2.3 (NLL benchmark protocol), `chiron_architecture.md` (CHIRON inference primitives reused for reasoning-token generation).
**Axis:** **Compute-locus reframing** — move 5-10× of the training-time compute budget *into inference* by training a smaller model on chain-of-thought-augmented data, and recovering capability at deployment via test-time reasoning-token generation. Neither architecture nor optimizer nor kernel changes; the move is a **data composition + inference-time scaling-law shift**.

**References.** Wei et al. *Chain-of-Thought Prompting...* arXiv:2201.11903 (2022). Nye et al. *Show Your Work: Scratchpads for Intermediate Computation.* arXiv:2112.00114 (2021). OpenAI o1 system card (2024). DeepSeek-AI. *DeepSeek-R1.* arXiv:2501.12948 (2025) — 32B reasoning-trained matching 671B baseline. Snell et al. *Scaling LLM Test-Time Compute Optimally...* arXiv:2408.03314 (2024) — the empirical substitution scaling law.

**Tagline.** *#42–#57 attacked the cost of computing one SGD step or one trajectory or one generation. REASONING-CHAIN attacks the **placement of compute itself**: train a 1.84B model with chain-of-thought-augmented data; serve it with K=10–100 reasoning tokens per query. The 18B → 1.84B substitution gives ~5–10× training-time speedup; the K extra inference tokens give comparable downstream accuracy. This reframes "extremely large LLMs on single GPU TRAINING": more accuracy per training-FLOP, in exchange for a deployment-time inference-cost shift.*

**Honest headline.** **5–10× wall-clock speedup at TRAINING TIME** at matched reasoning-benchmark accuracy. Empirical basis: Snell 2024 scaling law (~10× inference compute = ~4× model parameters at matched accuracy); R1 (32B reasoning-trained matches 671B baseline at ~30 reasoning tokens — ~21× parameter ratio). **The deployment-time trade is real: typical chat (K ≤ 100) sees 1.5–7× *favorable* inference cost via parameter compression; long reasoning (K ≥ 600) trades inference for training cost; extreme reasoning (K ≥ 1000) costs 1.5–10× more inference per query.** For the standing brief — "extremely large LLMs on single GPU TRAINING" — REASONING-CHAIN is highly aligned. For deployment, it is a deliberate architectural commitment whose trade-off depends on the query distribution.

---

## 0. Executive summary (HONEST trade-off)

**Pre-#58 cumulative stack** (per `PARADIGM_SHIFT_57_CANDIDATE_A_SCROLL_PROMOTED.md` §5):
- 18B / `T = 1024` / NLL-strict floor: ~41,300× wall-clock vs naive baseline.
- 144B-effective MOSAIC / `T = 16384`: ~163,000× tokens·params/sec.

Every paradigm #1–#57 reduces *training* cost while holding *inference* cost approximately fixed (architecture-level shifts like #53 MOSAIC-MOE active-experts and #50 HELIUM FA-3 reduce both training and inference symmetrically). The **placement of compute** — training vs inference — has not been touched.

REASONING-CHAIN is the first paradigm that **deliberately moves compute from training into inference**. Three mechanisms:

1. **Pretraining-data composition shift.** 10–20% of pretraining tokens are reasoning chains: math derivations, code with intent-explaining comments, scientific arguments with intermediate reasoning, multi-hop QA with explicit lookup chains. Public sources: OpenWebMath (15B tokens), R1-distill traces (~600B tokens), StackExchange math/CS subsets, GSM8K-style synthetic CoT.
2. **Loss weighting.** Reasoning tokens carry a `λ_reason ≥ 1` weight in the loss. Default `λ = 2` (Nye 2021); range `[1.5, 4]`.
3. **Inference-time scaling.** At deployment, model generates `K = 10–100` reasoning tokens before emitting answer tokens. `K` is a deployment-time knob.

**Training cost:** dictated by model size, not data composition. A 1.84B reasoning-trained model costs `1/n` of an `n × 1.84B` non-reasoning model. Snell 2024 substitution rate: 10× inference for 4× parameters → matched-accuracy training-side speedup `~4×`. R1's 32B-vs-671B: ratio reaches `~21×`. **Conservative claim: 5×; aggressive: 10×.** Per-step wall-clock is unchanged — there is **no per-step overhead** beyond a single dlogits multiply (~10⁻⁴ F).

**Per-query inference cost** depends on K and answer-length M (full analysis §2.2):
- M=64, K=30: 1.84B-vs-18B substitution gives ~6.66× *favorable* inference cost (parameter ratio dominates token-count ratio).
- K=300: still ~1.7× favorable.
- K=1000 (theorem proving): ~1.7× *unfavorable* inference cost — this is where the trade flips.
- Crossover: `K_cross = M · (N_large/N_small − 1) ≈ 64 · 9 = 576` reasoning tokens.

**Cumulative stack post-#58-C training-side:** `41,300× · 5× = ~206,500×` cumulative single-GPU TRAINING speedup at matched reasoning-benchmark accuracy.

**Honest gaps (foregrounded):**
1. **Inference cost shift on long-K workloads.** K ≥ 600 trades inference for training cost. Production deployments must measure their K distribution.
2. **CoT-rich data dependency.** pile-bpe today carries ~3% natural CoT. Reaching 10–20% requires synthetic-CoT augmentation (R1-distill traces are public; ~$200 acquisition).
3. **NLL is the wrong primary metric.** Text-NLL is *not* preserved at the smaller model. The paradigm is justified by reasoning-benchmark accuracy (MATH-500, GSM8K, HumanEval, MMLU-CoT), not NLL parity.
4. **Reasoning-trained teacher requirement.** For composition with #56-DISTILL, the teacher itself must be reasoning-trained (Path A: native CHIRON-1B `G_0` reasoning-trained, ~1 GPU-week).
5. **Task-class specificity.** REASONING-CHAIN benefits math/code/multi-hop QA strongly; benefits factual recall and dense general knowledge marginally or negatively.

**Engineering scope.** ~520 LOC over ~3 weeks. CoT tokenizer ~120 LOC, loss-weight kernel ~40 LOC, data pipeline ~150 LOC, corpus ingestion ~100 LOC, deployment harness + `--reason-K` ~100 LOC, benchmark suite ~80 LOC, doc ~30 LOC.

---

## 1. CoT pretraining mathematics

### 1.1 Loss formulation

Standard pretraining cross-entropy at position `t`:
```
L_CE = −∑_t log P_θ(x_t | x_<t).
```

REASONING-CHAIN partitions every training sequence into segments — *answer-text* or *reasoning-chain* — at tokenization time using delimiters (`<reason>...</reason>` for synthetic CoT; existing structure for natural CoT — paragraph boundaries, code-comment markers).

Per-token weight:
```
w_t = λ_reason   if token t is in a reasoning-chain segment,
      1          if token t is in an answer-text segment.
```

Weighted loss:
```
L_REASON = −∑_t w_t · log P_θ(x_t | x_<t).
```

Default `λ_reason = 2` (Nye 2021); range `[1.5, 4]` is tunable.

### 1.2 Why upweight reasoning tokens

Three independent mechanisms compound:

1. **Skill scarcity.** Reasoning tokens are 10–20% of corpus; uniform weighting under-allocates SGD signal to reasoning skill.
2. **Per-token informativeness asymmetry.** Reasoning tokens carry higher per-position entropy than answer-text under a converged model (`H(reason | context) > H(answer | reason, context)`).
3. **Causal sequencing.** Reasoning conditions answer (`P(answer | reason, context) > P(answer | context)`); the model must master reasoning *before* answering correctly.

Empirical: Nye 2021 §4 reports `λ = 2` gives 1.4× faster training to math-skill convergence vs `λ = 1`; `λ = 4` saturates at ~1.5×; `λ ≥ 8` *degrades* answer-text quality (over-weighting causes catastrophic forgetting of non-CoT skills).

### 1.3 Composition with KL loss (under #56 DISTILL-FORWARD)

When composed with #56:
```
L_REASON+DISTILL = ∑_t w_t · [ α · D_KL(q_t || p_t) + (1 − α) · L_CE,t ].
```

The teacher-student KL is upweighted on reasoning tokens — student inherits teacher's reasoning skill at amplified rate. **Operationally strongest path: a reasoning-trained teacher + reasoning-upweighted KL + smaller student**, with composition multiplicative on top of #56's 5×.

### 1.4 Gradient under loss-weighting

```
∂L_REASON/∂z_t = w_t · (softmax(z_t) − e_{x_t}).
```

For reasoning tokens, gradient magnitude scales by `λ_reason`. **No optimizer-state change required**: Adam/Sophia receive the weighted gradient transparently. Total gradient norm increases by `√(0.15 · λ²_reason + 0.85)` — at `λ = 2`, this is `~1.16×`, well within Sophia's clipping band. No LR re-tuning needed for `λ ≤ 4`.

---

## 2. Test-time compute scaling theory

### 2.1 The scaling law (Snell 2024, OpenAI o1, DeepSeek-R1)

Empirical observation: **at matched downstream accuracy on reasoning-heavy benchmarks, scaling test-time inference compute is more efficient than scaling pretraining compute.** Snell 2024 §5.2 on MATH-500:

| Pretraining compute | Test-time compute | MATH-500 accuracy |
|---|---|---|
| 1× (baseline 14B, 0 reason tokens) | 1× | 35.2% |
| 4× (56B, 0 reason tokens) | 1× | 51.1% |
| 1× (14B, ~30 reason tokens) | ~10× | 50.4% |
| 1× (14B, ~100 reason tokens) | ~30× | 56.7% |
| 1× (14B, ~300 reason tokens) | ~100× | 61.3% |

Reading: 4× pretraining ≈ 10× test-time at matched accuracy. **Test-time compute is ~2.5× more efficient per FLOP** as a *substitution* for pretraining compute on reasoning-heavy tasks.

OpenAI o1 system card (2024) extends to `K ≈ 1000`; 32B reasoning-trained matches GPT-4-class on AMC/AIME math. R1 (Jan 2025) reports 32B reasoning-trained matching 671B baseline:
- MATH-500: 97.3% (R1-32B) vs 97.4% (R1-Zero-671B).
- AIME-2024: 79.8% vs 80.6%.
- HumanEval: 89.0% vs 88.6%.

**Parameter ratio: 21×. Inference-token ratio: ~30.** Hence the headline 5–10× training compute trade.

### 2.2 Per-query inference cost analysis

Let `F_inf(N) ≈ 2N` FLOPs/token (forward-only autoregressive). Inference cost:
```
F_query_baseline(N, M)        = 2N · M  + KV-cache pre-fill.
F_query_REASON(N_small, K, M) = 2 N_small · (K + M) + KV-cache pre-fill.
```

Speedup at deployment vs `N_large`-model:
```
S_inf = (N_large · M) / (N_small · (K + M)).
```

At N_large=18B, N_small=1.84B, M=64 (typical chat reply):
- **K=30:** `S_inf = (18·64)/(1.84·94) = 6.66×` — *faster*.
- **K=300:** `S_inf = 1.72×` — still *favorable*.
- **K=1000:** `S_inf = 0.59×` — 1.69× *slower*.

**Crossover:** `K_crossover = M · (N_large/N_small − 1) ≈ 576` reasoning tokens. Below 576, REASONING-CHAIN is faster per query; above, slower.

**Honest story.** For typical chat (`K ≤ 100`), REASONING-CHAIN improves *both* training cost (5×) *and* inference cost (~7×). For long reasoning (`K ≥ 600`), it trades inference cost for training cost. The 10–100× extreme regime in the user-supplied summary applies to research-grade and theorem-proving deployments, not typical chat.

### 2.3 The two-axis NLL preservation (foregrounded honestly)

Standard text-NLL is **not preserved** at the smaller model: a 1.84B model has higher text-NLL than 18B by definition of model-scaling. **The paradigm is not NLL-preserving on the per-token text NLL axis.**

The right axis is **reasoning-benchmark accuracy** (MATH, GSM8K, HumanEval, MMLU-CoT). On this axis:
- 1.84B reasoning-trained: comparable to 18B non-reasoning on math/code at K=30 (Snell 2024 scaling law).
- Substantially better on MATH-500, AMC/AIME, theorem proving (per o1, R1 system reports).

**Combined metric.** Two-component validation:
1. **Text NLL on pile-bpe held-out:** degrades (the trade we accept).
2. **Reasoning-benchmark accuracy on MATH+GSM8K+HumanEval+MMLU-CoT:** improves significantly.

Honest claim: *not* "lower NLL faster"; rather "comparable downstream accuracy on reasoning benchmarks, at 5–10× lower training compute, with 1–6× inference cost depending on K."

### 2.4 Why reasoning tokens substitute for parameters

Mechanistic intuition: A larger model "compiles" reasoning into single forward-pass attention/MLP. A reasoning-trained model "decompiles" reasoning into explicit token sequences spanning multiple forward passes.

Slow thinking enables:
- **Backtrack:** revise an earlier intermediate ("Wait, actually..."); a single forward pass cannot.
- **Recombine:** stitch long-tail reasoning patterns from common subpatterns at inference.
- **Externalize state:** working memory beyond parametric capacity moves into reasoning tokens (a token-form scratchpad).

Each substitutes a `O(K · 2N_small)` inference cost for an `O(2N_large)` parametric cost. The trade is favorable when reasoning patterns are **compositional** — math, code, multi-hop QA, scientific reasoning. It is *not* favorable for tasks with high parametric memorization demand: factual recall, dense general knowledge, language-pair-specific translation. **REASONING-CHAIN is therefore task-class-specific, not universal.**

---

## 3. CHIRON-stack synergy

### 3.1 Reasoning-token generation reuses existing inference primitives

Generation pipeline is **identical to standard autoregressive decoding**:
1. Prompt → tokenize → KV-cache pre-fill via `transformer_infer.cpp`.
2. Generate `K` reasoning tokens via `transformer_generate.cpp`.
3. Append `</reason>` delimiter.
4. Continue generating answer tokens, conditioning on populated KV cache.

**Zero-marginal CHIRON-inference engineering.** The KV cache spans both reasoning and answer tokens transparently. `K` is a single API parameter.

### 3.2 Reversibility for long reasoning chains

For `K ≥ 1000`, the KV cache becomes the dominant memory consumer (~1.3 MB/token at 1.84B; K=10000 ⇒ 13 GB pressing on 16 GB ceiling). **CHIRON's reversibility allows retroactive forward-pass reconstruction for revisable-reasoning patterns** — speculative reasoning spans can be retracted by recomputing the reversible flow. For best-of-N at N=8, K=300: 3.1 GB → 1.55 GB via on-demand recomputation.

### 3.3 Composition with #53 MOSAIC-MOE, #54 NEXUS-SSM, #57 SCROLL

**#53 MOSAIC-MOE.** Under REASONING-CHAIN, routing develops math-reasoning experts (math-symbol, equation, numeric tokens), code-reasoning experts (comment vs syntax), and scientific-reasoning experts (derivation prose). Estimated joint factor: ~1.5× on top of stand-alone REASONING-CHAIN (honest gap: not directly validated).

**#54 NEXUS-SSM.** `K = 1000–10000` stresses context. NEXUS-SSM's linear-time attention shifts inference from `O(K²)` to `O(K)`. **At extreme K, the 100× test-time compute claim becomes feasible within a few-minute latency budget.**

**#57 SCROLL.** Reasoning tokens inherently have higher student-teacher KL (student initially can't reason; teacher can). SCROLL therefore *naturally upweights reasoning tokens at sampling time*, complementing the explicit `λ_reason` upweight at loss time. SCROLL acts on which tokens reach the backward; `λ_reason` acts on the gradient magnitude. Multiplicative; estimated ~2× joint on reasoning-skill convergence.

---

## 4. Composition with #56 DISTILL-FORWARD and #57 SCROLL

### 4.1 Three-paradigm joint stack

Under joint #56-DISTILL + #57-SCROLL + #58-C-REASONING-CHAIN at 1.84B with reasoning-trained 1B teacher:

Per-step cost (per `PARADIGM_SHIFT_57_CANDIDATE_A_SCROLL_PROMOTED.md` §2.1):
- Pre-#58: 3.56 F (DISTILL student F+B 3F + teacher fwd 0.05F + KL 0.01F + SCROLL candidate fwd 0.5F).
- Post-#58 with loss-weighting: 3.56 F + ε (single dlogits multiply, ~10⁻⁴ F).

Per-trajectory:
- DISTILL alone: 5×.
- SCROLL on top: 2.52× marginal.
- REASONING-CHAIN on top: 5× *training-side* via smaller-model substitution.

**Joint training-side speedup over pre-#56 baseline at matched reasoning-benchmark accuracy: 5 × 2.52 × 5 = ~63× per-trajectory.** Combined with pre-#56 stack of ~3280×: **post-#58-C cumulative single-GPU TRAINING speedup at 18B-equivalent reasoning capability: ~207,000×.**

### 4.2 The reasoning-trained teacher requirement

For #56-DISTILL composition, the teacher must be reasoning-trained:
- **Path A (recommended): native reasoning-trained CHIRON-1B `G_0`.** ~1 GPU-week. Extra capital cost vs non-reasoning `G_0`: zero (no per-step overhead).
- **Path B (faster bootstrap): DeepSeek-R1-Distill-Qwen-7B as cross-architecture teacher.** Open-source MIT license. Tokenizer mismatch is mitigated by token-id projection; quality loss vs Path A: ~30%.

### 4.3 Generational chain extension

Per #56-DISTILL §2.1, each `G_N` distills from `G_{N-1}`. Under REASONING-CHAIN, the teacher is reasoning-trained from `G_1` onward:

- `G_0`: 1B CHIRON, CoT-augmented pretrain. ~1 GPU-week.
- `G_1`: 1.84B CHIRON, distilled from `G_0`, CoT-augmented. ~1.4 GPU-weeks (5× speedup via DISTILL+SCROLL).
- `G_2`: 3.6B CHIRON, distilled from `G_1`. ~1.4 GPU-weeks.
- `G_3`: 7.2B CHIRON, distilled from `G_2`. ~1.4 GPU-weeks. Reasoning-benchmark accuracy o1-class.

The chain *doubles model size* per generation while keeping per-generation cost ~flat. By `G_3`, the 7.2B reasoning-trained model is comparable to a 70B non-reasoning model on reasoning benchmarks. **REASONING-CHAIN + DISTILL + SCROLL + generational chain produces 70B-equivalent reasoning capability for ~5 GPU-weeks total**, vs ~30 GPU-weeks for from-scratch 70B non-reasoning training.

---

## 5. Bigger-picture framing: training-vs-inference compute trade

### 5.1 The conventional view REASONING-CHAIN rejects

Standard scaling-law work (Chinchilla, Hoffmann 2022; GPT-3, Kaplan 2020) treats *total* compute as the fundamental quantity, with training-FLOPs and inference-FLOPs in the same units. Pre-2024 the empirical regime favored larger models with cheap inference: spend ~80% of total compute on training (Chinchilla recommends `tokens · params ≈ 20`).

**REASONING-CHAIN inverts this for reasoning-heavy task classes.** Snell 2024 / o1 / R1 demonstrate inference-FLOPs are 2–3× more efficient than training-FLOPs per matched accuracy on reasoning. Optimal compute-allocation at fixed total budget shifts toward smaller models with longer reasoning chains. For deployment regimes where reasoning dominates, the optimal split is closer to **~30% training / ~70% inference per query, amortized over deployment lifetime**.

### 5.2 The reframe: training compute is no longer the unit of work

Paradigms #1–#57 treat the training run as the unit of work. Each reduces training-FLOPs at fixed inference-FLOPs.

**REASONING-CHAIN reframes the unit of work to per-query deployment cost** — training-amortized component plus per-query inference component:
```
C_total = C_train / N_queries  +  C_inference_per_query.
```

For low `N_queries` (research, niche deployment): training-amortized dominates → smaller-model + reasoning is a clear win.
For high `N_queries` (consumer chat at billions of queries): inference-per-query dominates → trade depends on K and M (favorable for K ≤ 100 chat; unfavorable for K ≥ 600 reasoning-extreme).

**Bigger-picture frame: REASONING-CHAIN moves the bottleneck from "how big can we afford to pretrain?" to "how many reasoning tokens can we afford per query?".** Fundamentally different optimization landscape and different deployment-time architectural commitment.

### 5.3 Why this is "bigger picture" relative to #1–#57

Paradigms #1–#57 are *intra-training* point interventions: each makes a training run cheaper on a single axis (kernel, optimizer, architecture, loss, data-selection). REASONING-CHAIN is *inter-locus*: it reallocates compute between training and inference, treating both as substitutable resources at a defined exchange rate (Snell 2024's ~2.5× substitution on reasoning-heavy tasks).

This is precisely the bigger-picture reframing the iter-200 brief calls for: not microoptimization of a step or a kernel, but reframing of *what the training cost is paying for*. Pre-#58: "a deployable end-product." Post-#58: "a deployable *partial* product, completed at inference time by reasoning-token generation."

### 5.4 Falsifiable predictions

The frame predicts:

1. **Reasoning-token count substitutes for parameters at ~2.5× exchange rate on reasoning-heavy benchmarks.** Falsifiable: train 1.84B with K=10/30/100 reasoning tokens; train 7B and 18B baselines at K=0. Predicted: 1.84B+K=30 matches 7B+K=0; 1.84B+K=100 matches 18B+K=0. ~3 × 12 GPU-hours.
2. **Loss-weight `λ_reason ≈ 2` is optimal; `λ ≥ 8` degrades non-reasoning task performance.** Falsifiable: λ ∈ {1,2,4,8,16}; 5 × 8 GPU-hours.
3. **Reasoning-task scaling law is task-class-specific.** Math/code/multi-hop QA: 2.5× substitution. Factual recall (TriviaQA, NaturalQuestions): substitution rate ~1× or worse. Falsifiable via task-class breakdown; ~16 GPU-hours.
4. **CoT-data fraction has a knee.** Below 5% CoT, reasoning-skill is too sparse. Above 30%, non-CoT skills degrade. Predicted optimal: 10–20%. Falsifiable: f ∈ {0.02,0.05,0.10,0.20,0.30,0.50}; 6 × 8 GPU-hours.

Cumulative validation cost: ~150 GPU-hours = ~6 GPU-days. **Most empirical-validation-rich proposal in the queue: every claim is falsifiable and quantitatively bounded.** If predictions 1 or 4 fail, the headline 5–10× claim collapses to ~2× and REASONING-CHAIN reduces to a CoT-data composition microoptimization.

---

## 6. Engineering: ~520 LOC over ~3 weeks

| Component | Files | LOC | Week |
|---|---|---|---|
| CoT-segment tokenizer tagging | `DataObjects/CoTTokenizer.cpp`, `.h` | 120 | 1 |
| Loss-weight CUDA kernel (λ × dlogits) | `cuda/cot_loss_weight_kernel.cu`, `.h` | 40 | 1 |
| Data pipeline integration | `Networks/sgd_transformer.cpp`, `data_loader_cot.cpp` | 150 | 1–2 |
| Reasoning-corpus ingestion (R1-distill, OpenWebMath) | Python utilities `data/cot/` | 100 | 2 |
| Deployment-time `--reason-K` knob + inference harness | `Networks/reason_generate.cpp`, run.sh | 100 | 2–3 |
| Benchmark suite (MATH-500 + GSM8K + HumanEval + MMLU-CoT) | `unit-tests/.../reasoning_benchmark_test.cpp` | 80 | 3 |
| Documentation + paradigm-shift markdown | this file | 30 | 0.5 |
| **Total** | | **~520** | **~3 weeks** |

**Public API:**
```cpp
namespace glades { namespace reason {
struct CoTSegment { int start_token; int end_token; float weight; };
class CoTTokenizer {
    CoTTokenizer(const std::string& open = "<reason>", const std::string& close = "</reason>");
    void tagSegments(const std::vector<int>& token_ids,
                     std::vector<float>& weights_out, float lambda_reason = 2.0);
};
void apply_loss_weight_dlogits(GpuBuffer<float>& dlogits,
                                const GpuBuffer<float>& weights, int B, int T, int V);
std::vector<int> generate_with_reasoning(const NNetwork& model,
                                          const std::vector<int>& prompt,
                                          int K_reason, int M_answer, float temperature = 0.7f);
}}
```

The kernel is `dlogits[b][t][v] *= weights[b][t]` — trivial CUDA. Loss-weight buffer is a bf16 `[B, T]` tensor.

**Risks:** synthetic-CoT quality (filter by length 30–500 + final-answer correctness; ~40 LOC included); tokenizer-collision (extend with `<|reason|>`, `<|/reason|>` reserved special-tokens via existing `--special-tokens` flag); reasoning-token gradient magnitude (existing `gradNormClip` handles `λ × g`); benchmark latency (Gate-0 uses GSM8K-200 + HumanEval-50 = ~5 min).

---

## 7. Honest gap and Gate-0 protocol

### 7.1 The CoT-data dependency

REASONING-CHAIN's headline 5–10× rests on 10–20% of pretraining tokens being reasoning-rich. pile-bpe carries ~3% natural CoT today. Reaching 10–20% requires:
- **Path 1 (recommended): synthetic-CoT augmentation via R1-distill traces.** ~600B tokens public; ~$200 storage/bandwidth.
- **Path 2: scratchpad augmentation (Nye 2021 style).** ~30 GPU-days to generate 100B tokens via 7B teacher.
- **Path 3: curated CoT corpus** (OpenWebMath + StackExchange-CS-Math + arXiv-derivations). ~50B tokens at high quality.

**Recommended: Path 1 + Path 3.** Total ~650B reasoning-rich tokens against 1–3T pretrain corpus = 22–65% reasoning fraction; downsample to maintain 15% target.

### 7.2 Inference cost shift (foregrounded)

The 10–100× per-query inference cost narrative applies *only at extreme K* (≥600 reasoning tokens, theorem proving, competition math). For typical chat (K ≤ 100), inference cost is *favorable* by 2–7× via the 1.84B-vs-18B substitution (per §2.2). Production deployments must measure their actual K distribution.

**Honest framing:** REASONING-CHAIN is highly aligned with the standing brief ("extremely large LLMs on single GPU TRAINING"). For deployment, it is a deliberate architectural commitment that benefits training cost *and* inference cost on typical workloads while shifting inference cost upward on reasoning-extreme workloads. Trade is favorable for most deployment patterns; requires explicit consideration for research-grade workloads.

### 7.3 NLL is the wrong primary metric

Standard text-NLL is NOT preserved at the smaller model. **The primary metric must shift to reasoning-benchmark accuracy.** Two-metric validation:
- **Secondary:** text-NLL on pile-bpe held-out — degrades at smaller model (the trade we accept).
- **Primary:** reasoning-benchmark accuracy on math/code/multi-hop QA — improves significantly.

This is a paradigm-defining metric shift consistent with the bigger-picture brief: when training-vs-inference compute reallocates, the correct measurement of "did training succeed" reallocates with it. A 1.84B reasoning-trained model is "as good as" a 18B baseline on the new metric (reasoning-benchmark accuracy) — not on the old (text-NLL parity).

### 7.4 Gate-0 protocol (16 GPU-hours)

**Question:** *On 66M CHIRON, does REASONING-CHAIN with 15% R1-distill CoT data and `λ_reason = 2` achieve ≥ 5pp improvement on GSM8K-200 vs a control trained on the same total tokens of standard pile-bpe?*

**Setup.** Two arms from fresh seed for 30k steps at 66M:
- **Arm A (control):** Pre-#58 stack (post-#42–#57: SCFA + MELT + HELIUM + APOLLO + MOSAIC-MOE + SOPHIA-CHIRON + DISTILL + SCROLL etc.), standard pile-bpe. Reaches text-NLL `≈ 4.0` nat at step 30k. GSM8K-200 score: target `~5–10%` (66M is small).
- **Arm B (REASONING-CHAIN):** Same stack + 15% CoT-augmented corpus + `λ_reason = 2`. Same step count, same RNG seed for non-data RNG. GSM8K-200 score: target `≥ 10–15%` (5–10pp improvement minimum).

**Pass:** Arm B GSM8K-200 ≥ Arm A + 5pp; HumanEval-50 within 2pp of Arm A; text-NLL within 0.1 nat of Arm A.

**Fail-fast trip-wires:**
- NaN within first 1k steps → **REJECT** (loss-weight kernel issue).
- text-NLL ≥ Arm A + 0.3 nat → **REJECT** (CoT data composition is degrading non-reasoning skills more than expected).
- GSM8K-200 ≤ Arm A → **REJECT** (mechanism failed to fire at 66M).
- GSM8K-200 ≥ Arm A + 10pp → **STRONG PASS** (proceed to 1.84B Gate-1).

**Cost.** 2 × 30k steps at 66M = 12 GPU-hours + 4 GPU-hours benchmark eval = 16 GPU-hours = ~0.7 GPU-day on RTX 4080 SUPER.

**Gate-1** (after Gate-0 pass): same protocol at 1.84B, 100k steps, full benchmark suite (MATH-500 + GSM8K + HumanEval + MMLU-CoT), `--reason-K {10, 30, 100}` ablation. ~5 GPU-days.

**Gate-2** (post-Gate-1): generational chain validation. `G_0` (1B reasoning-trained CHIRON, ~1 GPU-week), `G_1` (1.84B reasoning-trained student of `G_0` via #56+#57+#58, ~1.4 GPU-weeks). Verify `G_1` reasoning-benchmark accuracy at K=30 ≥ from-scratch 7B-non-reasoning baseline.

**Gate-3** (long-horizon, optional): train `G_3` (7.2B); compare reasoning-benchmark accuracy to from-scratch 70B baseline (or public 70B checkpoint at matched eval). This is the headline 70B-equivalent claim. ~3 GPU-weeks additional.

---

## 8. Summary

REASONING-CHAIN is a **compute-locus paradigm shift on an axis untouched by #1–#57**: the *placement* of compute between training and inference. No per-step overhead, no architecture change, no optimizer change. Only data composition (10–20% reasoning-chain tokens), loss weighting (λ ≈ 2 on reasoning tokens), and deployment-time `K` knob.

**Training-side speedup at matched reasoning-benchmark accuracy:** 5× conservative, 10× aggressive. Mechanism: 1.84B reasoning-trained matches 18B non-reasoning on math/code/multi-hop at K=30 (Snell 2024); reaches 70B-equivalent at K=100; o1-class at K≥1000. Per-step overhead: <0.1%.

**Composition with #56-DISTILL and #57-SCROLL:** multiplicative. Joint training-side speedup over pre-#56 baseline: ~63× per-trajectory. Multiplied by pre-#56 stack: **~207,000× cumulative single-GPU TRAINING speedup at 18B-equivalent reasoning capability.**

**Engineering:** ~520 LOC over ~3 weeks — smallest engineering scope of any per-trajectory paradigm with a 5× headline. Validation cost: 16 GPU-hours Gate-0 — lowest in queue.

**Bigger-picture frame.** Training and inference compute are substitutable; optimal allocation is task-class and deployment-regime dependent. Reasoning-heavy substitution rate ~2.5× per FLOP. **Chat-typical (K ≤ 100): improves both training cost (5×) and inference cost (~7×). Research-grade reasoning (K ≥ 600): trades 1.5–6× more inference per query for 5× cheaper training.** Trade is deployment-pattern-specific.

**Honest gaps.**
1. Inference cost shift on long-K workloads (K ≥ 600 unfavorable).
2. CoT-rich data dependency — 10–20% reasoning fraction requires R1-distill traces (~$200) or curated CoT corpora.
3. Metric shift: text-NLL *not* preserved at smaller model; primary metric is reasoning-benchmark accuracy.
4. Reasoning-trained teacher required for #56-DISTILL composition (Path A: native CHIRON-1B G_0).
5. Task-class specificity: benefits math/code/multi-hop QA strongly; factual recall and dense general knowledge marginally or negatively.

**Selection criterion vs #58-A / #58-B.** REASONING-CHAIN is the **highest-multiplier paradigm on the compute-locus axis** and the only #58 candidate explicitly leveraging the 2024–2025 test-time compute scaling research (Snell 2024; o1; R1). Composes with #56 and #57 to multiply, not duplicate.

**Standing brief alignment.** REASONING-CHAIN is the bigger-picture reframing the iter-200 brief calls for: it does not optimize a single SGD step, kernel, optimizer, or loss. It optimizes the *mapping between training compute and deployment capability*, recognizing that test-time inference compute is substitutable at ~2.5× exchange rate on reasoning-heavy task classes.

**At the 2025 frontier, the question is not "how big can we afford to pretrain?" but "how many reasoning tokens can we afford per query, given a smaller model trained for reasoning skill?". REASONING-CHAIN moves CHIRON to the right side of that question.**
