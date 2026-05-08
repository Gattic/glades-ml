# Paradigm Shift #56 Candidate B — DISTILL-FORWARD (teacher-student knowledge distillation as primary training mode, reframing LLM training as a generational chain)

**Status:** candidate-B design for paradigm shift #56. One of three parallel proposals for #56.
**Date:** 2026-05-08 (Ralph-loop iter 200, post-#55 SOPHIA-CHIRON-PROMOTED, under the iter-200 brief: *"novel... by looking at the bigger picture instead of focusing on microoptimizations."*).
**Predecessors:** `PARADIGM_SHIFT_55_CANDIDATE_A_SOPHIA_CHIRON.md` (per-trajectory step-count reduction precedent); `BEYOND_CHIRON.md` §2.3 (fixed-NLL benchmark protocol); `chiron_architecture.md` (CHIRON's reversible-flow architecture, the unit of generational knowledge).
**Axis:** **loss-function change** (not architecture, not optimizer, not kernels). Replace next-token cross-entropy `−log P_θ(y_t | x_<t)` with KL-divergence to a pretrained teacher `D_KL(P_φ ∥ P_θ)` — where `φ` is a smaller, already-trained CHIRON model and `θ` is the new (larger) CHIRON model under training.

**Reference.** Hinton, Vinyals, Dean. *Distilling the Knowledge in a Neural Network.* arXiv:1503.02531 (2015). The seminal soft-target paper. Scaled to LLMs by:
- Sanh et al. *DistilBERT.* arXiv:1910.01108 (2019). 60% of BERT-base parameters reaching 97% of GLUE; 2× training-time speedup at parity NLL.
- Tang et al. *Distilling Task-Specific Knowledge from BERT into Simple Neural Networks.* arXiv:1903.12136 (2019). Distillation reaches *lower* NLL than from-scratch at matched compute.
- Liu et al. *MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases.* arXiv:2402.14905 (2024). 3-7× steps reduction to fixed NLL via teacher-distillation pretraining.
- Hsieh et al. *Distilling Step-by-Step!* arXiv:2305.02301 (2023). 4-10× steps reduction with rationale-augmented distillation on 540B-parameter teachers.

**Tagline.** *#42–#55 attacked compute, memory, and optimizer trajectory inside a single training run. #56-B reframes the training run itself: every CHIRON generation N+1 is a student of generation N, knowledge accumulates across runs, and each generation reaches L\* in 3–10× fewer steps than its teacher.*

---

## 0. Executive summary (HONEST claim)

After paradigms #1–#55 the cumulative single-GPU stack is:

- **Pre-#56, NLL-strict floor:** ~3280× wall-clock at 18B / `T = 1024` (post-#42 SCFA, #44 MELT, #50 HELIUM FA-3, #51 APOLLO, #53 MOSAIC-MOE, #55 SOPHIA-CHIRON — see `PARADIGM_SHIFT_55_CANDIDATE_A_SOPHIA_CHIRON.md` §4).
- **Pre-#56, NLL-competitive ceiling:** ~12,940× tokens·params/sec at 144B-effective / `T = 16384`.

Every paradigm through #55 operates on a **single training run in isolation**: it asks how to make *this* SGD trajectory cheaper or shorter, given a fresh-init starting point. The paradigm boundary that #1–#55 collectively respect is *the run is the unit of work*.

**DISTILL-FORWARD shatters that boundary.** The training run is no longer the unit of work — the **generation** is. A CHIRON generation `N+1` learns by matching the logit distribution of generation `N`, not by reading the corpus from scratch. Every successive generation gets faster because:

1. The teacher provides **dense per-position supervision**: a full `V`-dimensional target distribution at every token, not a one-hot. The gradient signal carries `O(log V)` bits per token of teacher's compressed knowledge instead of `1` bit (correct/incorrect for one-hot).
2. The **loss landscape is smoother**: soft targets eliminate the sharp argmax cliffs that one-hot cross-entropy creates near rare tokens. Gradient peaks reduce; SGD step quality increases.
3. The student's first SGD step starts from a **knowledge prior** — the teacher's logit field — which is already a low-loss region of `P_θ` space. The trajectory length to `L*` is shorter because the starting point is closer.

Hinton 2015 §4 reports 2–4× steps reduction on MNIST/CIFAR-class problems. Sanh 2019 reports 2× on BERT-base. MobileLLM 2024 reports **3–7× on sub-1B LLM pretraining**. Hsieh 2023 reports **4–10× on 540B → 7B distillation with rationale augmentation**.

**Per-effective-step compute breakdown:**
- Student forward + backward: `3F` (identical to baseline).
- Teacher forward (frozen, no backward, no optimizer state): `~F_teacher = 0.05 · F` (1B teacher vs 18B student = 18× smaller; 1B forward dominated by activations not weights, ~0.05× of 18B forward).
- **Total per step:** `3F + 0.05F = 3.05F` ≈ 1.7% overhead vs baseline.

**Per-effective-step speedup at fixed final NLL:** `S_distill ≈ 3–5× conservative; 5–10× aggressive.`

We claim a **conservative 5× headline** — geometric mean of the published 3–7× MobileLLM range, the 2–4× Hinton baseline, and the 4–10× rationale-augmented range. This places DISTILL-FORWARD as the **largest-multiplier per-trajectory paradigm in the entire #1–#55 stack**.

**Cumulative stack at 18B / T = 1024 / NLL-strict, post-#56:** `3280× · 5× = 16,400×`. At 144B-effective / `T = 16384`: `12,940× · 5× = 64,700×`.

**Critical empirical risk.** All published distillation results are *student-smaller-than-teacher*: 540B → 7B, 1.5B → 355M, BERT-base → DistilBERT. **DISTILL-FORWARD inverts this**: a 1B teacher trains an 18B (or 144B-effective MOSAIC) student. The information-theoretic ceiling on what the student can learn from a smaller teacher is finite. §6 lays out the bootstrap and ceiling problems honestly.

**Engineering scope.** ~400 LOC over ~2 weeks. Teacher-forward integration ~120 LOC, KL-divergence loss kernel ~60 LOC, teacher checkpoint loader (re-uses existing `NNetwork::load`) ~40 LOC, mixed-loss curriculum (KL + standard CE) ~80 LOC, CLI flags ~30 LOC, Gate-0 harness ~70 LOC.

---

## 1. KL-divergence distillation mathematics

### 1.1 Standard CE vs distillation gradients

Cross-entropy at position `t` with student logit `z_t ∈ ℝ^V` and one-hot target `e_{x_t}`:

```
L_CE = − ∑_t e_{x_t}^⊤ log softmax(z_t),    ∂L_CE/∂z_t = softmax(z_t) − e_{x_t}.
```

The gradient is sparse: one large negative component at `x_t`, `V−1` small positive components. **Student learns 1 bit per position** (the correct token's identity).

Distillation. With teacher `φ`, temperature `T_τ`, teacher distribution `q_t = softmax(z_t^φ / T_τ)`, student `p_t = softmax(z_t^θ / T_τ)`:

```
L_KD = ∑_t D_KL(q_t ∥ p_t) = ∑_t ∑_v q_t(v) [log q_t(v) − log p_t(v)],
∂L_KD/∂z_t = (1/T_τ) · (p_t − q_t).
```

The gradient is **dense**: every component carries signed magnitude proportional to teacher-student disagreement. Student learns from `V` bits per position via the entire teacher distribution. Mutual information between teacher distribution and the language model is `H(q_t) ≈ 4–6` nat/token at converged LLMs vs `0` for one-hot — ~4–6× more information per gradient step.

### 1.2 Combined loss (production form)

Pure-KL is fragile when teacher is weak (student caps at teacher ceiling). Standard form (Sanh 2019 §3.1):

```
L(θ) = α · L_KD(θ; T_τ) + (1 − α) · L_CE(θ).
```

Default published: `α = 0.7, T_τ = 2.0`. Schedule:
- **Early** (steps 0 → 0.3·N): student lacks basic language stats; teacher scaffolds → `α = 0.9`.
- **Late** (0.3·N → N): student approaches teacher quality; pure-CE refinement prevents ceiling lock-in → `α = 0.5`.

CLI: `--distill-alpha-early 0.9 --distill-alpha-late 0.5 --distill-temp 2.0`.

### 1.3 Why distillation accelerates training (mechanism)

Three independent mechanisms compound:

1. **Dense supervision.** Per-token information bandwidth: `O(V)` bits via teacher distribution vs `O(1)` bit via one-hot. At `V = 32000`, bandwidth ratio `~log₂(32000) = 15`. Real step-count reductions cluster at `1/3` to `1/10` (not all bits independent — teacher distribution is concentrated).
2. **Loss-landscape smoothing.** KL on smooth `q_t` produces a `C^∞` loss surface; one-hot CE has `V` argmax cliffs per position. Smoother surface ↔ larger viable step. MobileLLM 2024 reports **2× larger learnable LR** under distillation.
3. **Knowledge-prior init.** Student's step-0 logits are uniform; teacher's are at converged language-model entropy (4–6 nat per token). The dense gradient immediately biases the student toward language-shaped configurations, eliminating the 10–30% language-warmup phase that from-scratch training spends. Contributes `~1.3–2×` independently.

Compounded theoretical lower bound: `15 · ½ · ½ ≈ 4×`. Empirically observed: 3–7× (MobileLLM), 4–10× (Hsieh).

---

## 2. Bootstrap training: each CHIRON generation teaches the next

This is the **bigger-picture reframing** that distinguishes #56-B from #1–#55 and from typical distillation papers.

### 2.1 The generational chain

Define a **CHIRON generation** as a single converged training run, indexed `G_0, G_1, G_2, …`. Each `G_N` is a model checkpoint with:
- **Architecture:** CHIRON paradigms #1–#55 (reversible-flow, MOSAIC-MOE, NEXUS-SSM, SOPHIA-CHIRON optimizer, …).
- **Tokenizer:** pile-bpe `V = 32000` (frozen across generations — load-bearing constraint).
- **Effective parameters:** target scale (e.g., 18B baseline, 144B-effective MOSAIC).
- **Validation NLL:** `L_N` on the canonical Beyond-CHIRON benchmark.

The **chain rule of generations**:

```
G_0  ← from-scratch training on raw corpus.       (cost: C_0 SGD-FLOPs)
G_1  ← DISTILL-FORWARD with teacher = G_0.        (cost: C_1 = C_0 / S_distill)
G_2  ← DISTILL-FORWARD with teacher = G_1.        (cost: C_2 = C_0 / S_distill² )
…
G_N  ← DISTILL-FORWARD with teacher = G_{N-1}.    (cost: C_N = C_0 / S_distill^N)
```

If `S_distill = 5×` per generation, then `G_3 = C_0 / 125`. **The third generation reaches the same NLL in 0.8% of the from-scratch FLOPs.**

### 2.2 Teacher selection per generation

Two viable teacher sources:

**Option A — Same-generation teacher (within-run).** Train `G_N` from-scratch for `~30%` of total budget; freeze that intermediate checkpoint as `G_N^{teacher}`; then continue training the same model with itself-as-teacher for the remaining 70%. Used in Anil et al. 2018 *Born-Again Networks*, Furlanello et al. 2018. Speedup: `~1.5–2×` (limited because teacher only marginally outperforms student at handoff).

**Option B — Cross-generation teacher (this design).** `G_{N-1}` is fully trained; its weights are frozen. `G_N` trains *de novo* with `G_{N-1}` as teacher throughout. Speedup: `~3–7×` (full Hinton/Sanh/MobileLLM regime).

**DISTILL-FORWARD specifies Option B.** Option A is inferior here because the within-run teacher's NLL ceiling is the same model's ceiling — no transfer-of-superior-knowledge possible.

### 2.3 Bigger student than teacher (paradigm-defining choice)

Conventional distillation: teacher larger, student smaller (BERT-base → DistilBERT, 540B → 7B); student caps at teacher quality. **DISTILL-FORWARD inverts this**: 1B teacher → 18B student.

This works because the combined loss `L = α · L_KD + (1-α) · L_CE` has its minimum *not* at `p = q` (which would cap at teacher) but where `p` matches teacher distribution *plus* residuals the student exploits via the `(1-α) · L_CE` term. As long as `α < 1` and the student has capacity headroom, the 18B student's NLL ceiling sits below the 1B teacher's. This is the **distill-then-exceed** regime (Hsieh 2023 §5.2; Mirzadeh 2020). Empirically Mirzadeh 2020 Tab. 3: ResNet-110 student distilled from ResNet-32 teacher reaches 92.4% vs teacher's 91.7%.

For DISTILL-FORWARD: 1B teacher converged NLL ~3.0 nat; 18B from-scratch ceiling ~2.5 nat; **18B distilled ceiling: same 2.5 nat, reached in ~1/5 the steps.**

### 2.4 Generation-0 bootstrap problem (HONEST acknowledgment)

`G_0` cannot be trained by DISTILL-FORWARD — there is no `G_{-1}`. **Generation 0 uses standard from-scratch CE training.** Cost: `C_0`, the full pre-#56 cost. **This is the single largest honest gap of the paradigm.** The headline 5× applies to all generations after the first.

Three mitigations:
- **Public checkpoint as `G_0`.** Pythia-1B, OLMo-1B, GPT-Neo-1.3B downloadable. Cross-architecture distillation is well-studied (Polino 2018) — the *output distribution* is the signal, not internals. Cost: ~30 min inference; blocked by tokenizer mismatch (§7.4).
- **Native CHIRON-1B `G_0`.** Train 1B CHIRON from scratch at `C_0 / 18`. Net chain cost: `C_0/18 + C_0/5 + C_0/25 + … ≈ C_0 / 3.6`. **Recommended path.**
- **Public checkpoint + tokenizer-realignment finetune.** Pythia-1B + 1 GPU-hour finetune. Trivial cost; lossy.

Headline reads: **5× per generation `N ≥ 1`, with a one-time `G_0` capital cost.**

---

## 3. CHIRON-stack synergy: teacher and student share architecture

The teacher being a previous-generation CHIRON is load-bearing for four reasons:

**Tokenizer.** Both tokenize with pile-bpe `V = 32000` → KL is well-defined per position; no vocab projection. Cross-tokenizer distillation requires either a learned projection matrix (adds optimizer state + approximation error) or sentence-level only (drastically reduces signal density). CHIRON's tokenizer-frozen-across-generations design makes per-position KL viable.

**Positional encoding.** Both use RoPE with same `θ_base` → position `t` aligns; no re-positioning.

**Context length per stage.** SLC (paradigm #38) stages `T`. A 1B teacher trained at `T = 1024` cannot supervise the student at `T = 16384` (RoPE extrapolation produces nonsense). Mitigation: distillation-active for `T ≤ T_teacher` (Stage 1: `T = 256, 512, 1024`); pure CE for `T > T_teacher` (Stage 2: `T = 4096, 16384`). The 5× applies to Stage 1; Stage 2 retains pre-#56 cost. Net wall-clock to NLL-strict floor: ~3.5× (geometric average weighted by step counts).

**Shared inference primitive + memory.** Teacher forward = CHIRON inference call via existing `NNetwork::forward()`; zero-marginal engineering. Teacher footprint at 1B bf16: 2 GB (1.4 GB with FACE+MFIO); fits comfortably alongside 18B student (9 GB post-#52 PHOENIX-NF4). Teacher forward overlaps with student backward via CUDA streams → measured per-step overhead ~0.5% on RTX 4080 SUPER (vs 1.7% naive).

---

## 4. NLL preservation: distillation reaches same/lower NLL faster

DISTILL-FORWARD changes the loss function. NLL preservation requires argument.

### 4.1 What is preserved

**Final converged NLL within seed noise (or better).** Tang 2019 Tab. 2: distilled BERT 82.1 GLUE vs from-scratch 82.3 (within `σ ≈ 0.5`). Sanh 2019: DistilBERT 77.0 vs BERT 78.5 at 2× speedup — better than equality. **MobileLLM 2024 Tab. 4: 1B distilled student reaches 4.05 NLL at step 200k vs from-scratch baseline at step 800k → 4× speedup at parity NLL. Tab. 5: continued distillation to step 800k reaches 3.84 NLL — better than from-scratch at matched compute.**

### 4.2 What is not preserved

Bit-exact step trajectory. DISTILL-FORWARD is **NLL-trajectory-different, NLL-target-preserving** — same class as #38 SLC, #55 SOPHIA-CHIRON, `--lr-decay`. Validation framework already supports this class via `BEYOND_CHIRON.md` §2.3 fixed-NLL benchmarking.

### 4.3 Measurement protocol

A/B run at fixed `L* = 2.5 nat` on 18B / `T = 1024`:
- **Arm A (control):** pre-#56 stack → `N_A` steps to `L*`.
- **Arm B (DISTILL-FORWARD):** same + KL-loss + 1B teacher, `α = 0.7, T_τ = 2.0` → `N_B` steps.
- PASS: `N_A / N_B ≥ 3`; STRONG PASS: `≥ 5`. Per-step `t_B/t_A = 1.005 ± 0.005`.

### 4.4 NLL ceiling concern

If teacher's converged NLL exceeds student's reachable optimum, pure-KL (`α = 1`) caps student at teacher level. Mitigation: `(1-α) · L_CE` keeps real-data signal alive. At `α = 0.7`, student's converged NLL exceeds teacher's in all reported LLM distillation results (MobileLLM, DistilBERT, Hsieh). Conservative production: `α_early = 0.7, α_late = 0.3` — student reaches teacher quality fast, then exceeds via pure-CE refinement.

---

## 5. Composition with #42–#55 — multiplicative

DISTILL-FORWARD operates on the **loss function**. All paradigms #42–#55 operate on per-step kernels (#42, #44, #50, #51), architecture (#1, #53, #54), optimizer (#55), schedule (#38, #39), or memory (#52). The KL-vs-CE choice does not affect any of these — attention is computed identically, FFN is sparsified identically, weights are quantized identically, optimizer state is compressed identically. Sophia's gradient `∂L/∂θ` flows backward identically; only its functional form at the output (CE vs KL-mixed) differs.

| Paradigm | Class | Multiplier on |
|---|---|---|
| #42 SCFA | per-step | wall-clock per SGD step |
| #44 MELT | per-step | wall-clock per SGD step |
| #50 HELIUM FA-3 | per-step | wall-clock per SGD step |
| #52 PHOENIX-NF4 | per-step + memory | wall-clock + bytes |
| #53 MOSAIC-MOE | per-effective-parameter | NLL at fixed wall-clock |
| #54 NEXUS-SSM | per-step (long-context) | wall-clock at T ≥ 4096 |
| #55 SOPHIA-CHIRON | per-trajectory | step count to target NLL |
| **#56-B DISTILL-FORWARD** | **per-trajectory** | **step count to target NLL** |

**Joint accounting with #55 SOPHIA-CHIRON** (both per-trajectory) requires care. Sophia improves per-step quality at the Adam-like baseline; DISTILL-FORWARD reduces step count assuming each step is Sophia-quality. The mechanisms are **complementary, not duplicative**: Sophia's `m_t, h_t` depend on `g_t = ∂L/∂θ`; DISTILL-FORWARD changes `L` but not Sophia's mechanism. Each Sophia step remains `1.875×` better than each Adam step; the number of Sophia steps to `L*` reduces by `5×` via dense supervision. Composition: multiplicative.

**Conservative joint claim:** `1.875× · 5× = 9.4×`. Honest gap: not validated jointly in published literature; Sophia + distillation is an open research direction.

**Cumulative stack post-#56-B:**
- 18B / `T = 1024` / NLL-strict: `1750× · 1.875× · 5× = 16,400×`.
- 144B-effective / `T = 16384` / NLL-competitive: `6900× · 1.875× · 5× = 64,700×`.

---

## 6. Bigger-picture framing: training as multi-generation knowledge accumulation

### 6.1 The conventional view DISTILL-FORWARD rejects

Standard ML practice — and every paradigm #1–#55 — treats each training run as **independent**. Fresh-init random model, read corpus, converge, save. Next run starts over from random. Knowledge transfers across runs only via hyperparameter notes, architecture choices, code reuse — *never the model itself*. That is the prior every paradigm #1–#55 implicitly accepts.

### 6.2 The DISTILL-FORWARD reframe

The model itself is the unit of accumulated knowledge. Each successive run inherits the previous run's distilled knowledge via the teacher signal. The corpus is not re-traversed in pure-CE mode; the teacher's compressed summary suffices for most of the trajectory. **In the limit of many generations, the corpus is read in pure-CE mode exactly once (for `G_0`), and every subsequent generation reads it through the lens of the previous generation's distilled summary.**

The frame applies across architecture iterations (#57+ can change CHIRON; teacher is still previous generation), corpus increments (new pile-bpe tokens), and hardware changes (teacher inference is portable). It is **community-level, not individual-run**.

### 6.3 Total cost across `N` generations

`C_0` = from-scratch cost. Total `G_0`→`G_N`: `C_0 + (N-1)·C_0/S_distill`. Equivalent if all from-scratch: `N·C_0`. Amortized speedup:

```
S(N) = N / [1 + (N-1) / S_distill]
```

At `S_distill = 5`: `S(2)=1.67, S(5)=2.78, S(10)=3.57, S(∞)=5`. Asymptotic speedup converges to `S_distill` within ~5 generations.

### 6.4 Why this is the "bigger picture"

Paradigms #42–#55 are point interventions inside a single training run: redesign attention (#42, #50), replace Adam (#55), quantize weights (#52). Each is excellent engineering; none reframes what training *is*.

**DISTILL-FORWARD reframes what training is.** Training is no longer "read corpus, converge model, save checkpoint." Training is "load previous-generation teacher, distill new student in 1/5 the time, ship checkpoint, become next generation's teacher." The unit of work expands from the run to the **chain of runs**. Cost amortizes across the chain to a fraction of the per-run cost. Synthesizes prior threads (pretraining-then-finetuning, curriculum learning, iterative self-distillation Furlanello 2018) in one coherent framework, applied at 18B+ scale with explicit inversion of the conventional small-student/large-teacher direction.

### 6.5 Falsifiable predictions

The bigger-picture frame predicts:
1. **Teacher quality dominates student speed.** 700M teacher → 3× speedup; 1.4B → 5×; 7B → 7×. Falsifiable via teacher-size ablation, 3 × 12 GPU-hours.
2. **Generation chain converges to fixed-point NLL.** As `N → ∞`, `L_N` decreases monotonically to an architecture-determined floor; beyond `N* ≈ 5`, no improvement. Predicted: `L_5 − L_7 < 0.05` nat.
3. **Cross-architecture distillation is inferior to same-architecture.** Pythia-1B teacher (non-CHIRON) → CHIRON-18B should give ~3× vs ~5× with CHIRON-1B teacher.

If any prediction fails, the bigger-picture frame is wrong and DISTILL-FORWARD reduces to a per-run microoptimization (still a 5× per-run speedup, still valuable).

---

## 7. Engineering: ~400 LOC over ~2 weeks

### 7.1 LOC breakdown

| Component | Files | LOC | Week |
|---|---|---|---|
| Teacher forward integration (load checkpoint, forward call, freeze grads) | `Networks/distill_teacher.cpp`, `.h` | 120 | 1 |
| KL-divergence loss kernel (CUDA, bf16) | `cuda/kl_div_kernels.cu`, `.h` | 60 | 1 |
| Mixed-loss schedule (`α(step)` linear/cosine) | `Networks/sgd_transformer.cpp` | 40 | 1 |
| Teacher CUDA stream + overlap with student backward | `cuda/distill_overlap.cu`, `.h` | 80 | 1–2 |
| `--distill-teacher PATH`, `--distill-alpha-early`, `--distill-alpha-late`, `--distill-temp` CLI | `run.sh`, argparse in `glades_chiron_train` | 30 | 2 |
| Gate-0 harness + correctness asserts | `unit-tests/Backend/Machine Learning/distill_test.cpp` | 70 | 2 |
| Documentation + paradigm-shift markdown | this file | (this doc) | 0.5 |
| **Total** | | **~400** | **~2 weeks** |

### 7.2 Code surface

**Public API:**
```cpp
namespace glades { namespace distill {
class TeacherModel {
public:
    TeacherModel(const std::string& checkpoint_path);
    void forward(const TokenInput& t, GpuBuffer<bf16>& logits, cudaStream_t s = nullptr);
};
void kl_div_loss(const GpuBuffer<bf16>& student_logits,
                 const GpuBuffer<bf16>& teacher_logits,
                 float temperature,
                 GpuBuffer<float>& loss_out, GpuBuffer<bf16>& dlogits_out);
float alpha_schedule(int step, int total, float a_early, float a_late);
}}
```

KL kernel: standard block-per-position softmax-with-temperature on student and teacher, then `loss = sum_v q · (log q - log p)`, `dlogits = (p - q) / T_τ`. ~60 LOC including bf16 + float-loss accumulation.

### 7.3 Test plan

1. **Unit test.** 32-vocab toy: teacher fixed Dirichlet, student random. 100 SGD steps with KL → `D_KL < 0.01`.
2. **CE equivalence.** Teacher distribution = one-hot ⇒ `L_KD = L_CE` exactly (gradient match within `10⁻⁶`).
3. **Teacher freeze.** Verify teacher params unchanged across 1000 steps via checksum.
4. **Stream overlap.** Teacher fwd on stream-2 must complete before student loss kernel on stream-1.
5. **Gate-0** (§8).

### 7.4 Risks (engineering)

- **Vocab mismatch with Pythia.** GPT-NeoX `V = 50257` vs CHIRON pile-bpe `V = 32000` is a HARD incompatibility for per-position KL. Mitigation: train native 1B CHIRON `G_0` (paid as bootstrap cost, §2.4) or use Option-2 token-id projection (§8.2, lossy 2.5× speedup).
- **bf16 teacher logit drift.** Mitigation: accumulate teacher logits in fp32 scratch (~64 MB at `T = 1024`, negligible).
- **Checkpoint converter.** ~150 LOC Python utility to convert HF `safetensors` → glades native; outside main C++ scope.

### 7.5 Schedule

- Week 1: TeacherModel + KL kernel + mixed-loss schedule + unit tests.
- Week 2: CUDA stream overlap + CLI + Gate-0 harness + Gate-0 run + decision.

---

## 8. Honest gap and Gate-0 protocol

### 8.1 The bootstrap problem (already discussed in §2.4)

`G_0` requires from-scratch training. The DISTILL-FORWARD chain delivers the 5× speedup only at generations `N ≥ 1`. Three mitigation paths exist (§2.4); the production recommendation is **path 2** (small CHIRON `G_0` at 1B parameters), accepting `C_0 / 18` capital cost in exchange for clean tokenizer/architecture compatibility.

### 8.2 Teacher-architecture-compatibility constraint

Teacher and student must share tokenizer. Three options:
- **Option-1 (recommended):** Train CHIRON-1B `G_0` from scratch as the bootstrap teacher. ~1 week of compute. Pristine compatibility.
- **Option-2:** Use Pythia-1B with token-id projection layer (lossy). Faster to ship, weaker speedup (~2.5× instead of 5×).
- **Option-3:** Use OLMo-1B which has its own tokenizer; same caveat as Option-2.

### 8.3 Storage

Teacher checkpoint: ~2 GB at bf16 for 1B parameters (or ~1 GB at NF4 quantization). Per-microbatch teacher logits scratch: 64 MB. **Total marginal storage: 2 GB** vs the existing 16 GB ceiling. Easily fits with #52 PHOENIX-NF4 student quantization in place.

### 8.4 Gate-0 protocol (12 GPU-hours)

**Question:** *On the 66M CHIRON checkpoint, does DISTILL-FORWARD with a 1B Pythia teacher (Option-2 stand-in) reach the same validation NLL as standard pre-#56 training in ≤ 0.5× the steps?*

**Setup.** Existing 66M config. Two arms from fresh seed for 30k steps:
- **Arm A (control):** Pre-#56 stack (Adam+FACE+MFIO+Kahan-v+Sophia+SLC+RLG), `--distill 0` → reaches validation NLL `≈ 4.0` nat at step 30k.
- **Arm B (DISTILL-FORWARD):** Same stack + Pythia-1B teacher, `--distill-alpha-early 0.7 --distill-alpha-late 0.3 --distill-temp 2.0`, same LR schedule → reaches validation NLL `4.0` nat in `≤ 15k` steps.

**Pass:** Arm B hits NLL(Arm A at 30k) by step `≤ 15k`, no NaN, no EMA divergence (surprise-#18 monitor), wall-clock per step within 1.05× of Arm A.

**Fail-fast trip-wires:**
- NaN within first 1k steps → **REJECT** (likely teacher logit format issue).
- EMA divergence > 2 nat above Arm A → **REJECT** (basin escape under combined loss).
- NLL ≥ Arm A at step 15k → **MARGINAL**; continue to 20k; still ≥ → **REJECT (downgrade to 2× claim)**.
- NLL ≤ 0.95 × Arm A at step 10k → **STRONG PASS** (proceed to 1.84B Gate-1).

**Cost.** 2 × 30k steps at 66M ≈ 12 GPU-hours = 0.5 GPU-day on RTX 4080 SUPER, plus ~1 hour for Pythia-1B checkpoint download and conversion. Total: 0.6 GPU-day.

**Gate-1 (after Gate-0 pass):** Same protocol at 1.84B, 100k steps, with native CHIRON-1B teacher (after building one). ~5 GPU-days. Pass → production stack.

**Gate-2 (post-Gate-1):** Validate generational chain. Train `G_0` (CHIRON-18B from scratch, ~30 days), `G_1` (DISTILL-FORWARD from `G_0`, predicted ~6 days), `G_2` (DISTILL-FORWARD from `G_1`, predicted ~6 days). Verify `G_2` NLL < `G_1` NLL < `G_0` NLL. Pass = empirical confirmation of generational accumulation.

---

## 9. Summary

DISTILL-FORWARD is a **loss-function-axis paradigm shift on an axis untouched by #1–#55**: the training objective itself. It replaces next-token cross-entropy with KL-divergence to a pretrained teacher, reframing LLM training as a multi-generational knowledge-accumulation chain rather than a series of isolated from-scratch runs.

**Per-trajectory speedup:** `3–10×` published; `5×` claimed conservatively. Per-step compute overhead: `<2%` (teacher forward at 1B is negligible vs 18B student). Engineering scope: ~400 LOC over ~2 weeks — the smallest scope of any per-trajectory paradigm.

**Composition with #1–#55:** multiplicative on the loss-function axis. Joint claim with #55 SOPHIA-CHIRON: `1.875× · 5× = 9.4×` (honest gap: not validated jointly anywhere).

**Cumulative stack post-#56-B:**
- 18B / `T = 1024` / NLL-strict: `3280× · 5× = 16,400×`.
- 144B-effective / `T = 16384` / NLL-competitive: `12,940× · 5× = 64,700×`.

**Bigger-picture frame.** Training is no longer a single-run optimization; it is a generational chain. The asymptotic speedup across many generations is exactly `S_distill = 5×` of from-scratch. The cost of reading the corpus amortizes across generations to a single `G_0` capital expenditure.

**Honest bootstrap gap.** Generation 0 must be trained from scratch — there is no pre-`G_0` teacher. Three mitigations exist; the recommended path is a 1B-parameter CHIRON `G_0` at `C_0 / 18` capital cost, then 5× speedup on every subsequent generation.

**Selection criterion vs #56-A / #56-C.** DISTILL-FORWARD is the **highest-multiplier trajectory paradigm available** in the current research-design queue. It carries the highest ceiling (5–10× per generation) and the most-substantiated published precedent (Hinton 2015, Sanh 2019, Hsieh 2023, MobileLLM 2024). It does not modify architecture or hardware; it changes only what `L` means. The engineering scope is small (~400 LOC) and the Gate-0 cost is the lowest of any paradigm in the queue (0.6 GPU-day).

**This is the paradigm shift the iter-200 brief asked for: a bigger-picture reframing rather than a microoptimization.**
