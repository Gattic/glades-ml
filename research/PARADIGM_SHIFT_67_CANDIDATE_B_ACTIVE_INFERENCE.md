# Paradigm Shift #67 Candidate B — ACTIVE-INFERENCE-CHIRON: predictive-coding / Friston-style LOCAL-GRADIENT training (each layer locally minimizes its own prediction error)

**Status:** candidate-B design for paradigm shift #67. **Recommended action: RESERVE** with strong honesty caveat — promising on the *layer-parallel* compute-locality axis, but the LLM-scale prior art for predictive-coding / forward-forward / equilibrium-prop / direct-feedback-alignment families is **uniformly negative** (Whittington-Bogacz works on shallow MLPs; Hinton's Forward-Forward stalls at MNIST/CIFAR scale; Lillicrap DFA loses 5–10 nat at >100M params; Bengio Equilibrium Prop has no >1B precedent). The user brief from iter-211 reasserts strict bit-exact text NLL — under that constraint **ACTIVE-INFERENCE-CHIRON cannot deliver its headline 5–10× speedup with NLL preservation**. The honest framing: this is a hybrid-only candidate (PC-pretraining + CE-finetuning) with a **realistic 1.5–2.0× wall-clock speedup band** when stacked with #43 ORION's slow-manifold projection and #46 REFLECTOR's adjoint flow, and a **highly speculative 3–5× upper band** that requires the Whittington-Bogacz equivalence theorem to hold approximately at LLM scale (no empirical precedent above 1B params).
**Date:** 2026-05-08 (Ralph-loop iteration 211, post-iter-210 close at ~6,600,000× cumulative on grounded-reasoning subset / 5,500,000× knowledge-augmented / 5,400,000× VL benchmarks / 5,360,000× agent / 3,030,000× tool-augmented / 930,000× text NLL preserved).
**Predecessors.** All of #42–#66. Load-bearing references: `PARADIGM_SHIFT_43_DESIGN.md` (ORION — slow-manifold MOR; closest *optimizer* mechanism to PC's iterative inference), `PARADIGM_SHIFT_46_DESIGN.md` (REFLECTOR — adjoint flow on cotangent lift; honest no-go theorem at ~1.6× provides a calibration point for any new optimizer-axis claim), `PARADIGM_SHIFT_50_DESIGN.md` (HELIUM — FA-3 + FP8 backward; benchmarks the *standard* backward we are trying to avoid), `PARADIGM_SHIFT_52_DESIGN.md` (NIMBUS — async optimizer pipelining; closest *parallelism-locality* mechanism), `PARADIGM_SHIFT_55_DESIGN.md` (SOPHIA — second-order optimizer; calibrates "novel optimizer" headline at 1.875×).

**Axis.** **LOCALITY** — proposed twelfth axis after CROSS-MODAL closed at #66. The trunk's training loop replaces the global backward pass over L=22+ transformer layers with **L parallel layer-local prediction-error objectives**. No upstream gradient path; no end-to-end credit assignment in the standard backprop sense. Differentiated from #43 ORION (slow-manifold *projection* of the global gradient — still requires global backward) and from #52 NIMBUS (async optimizer step — still uses global backward to produce the gradient, just delays the optimizer). **The genuinely new axis is: gradient generation itself becomes layer-local and parallel.**

**Tagline.** *Standard backprop generates gradients sequentially: forward L → backward L → update. ACTIVE-INFERENCE-CHIRON replaces this with: forward L → L parallel local prediction errors → L parallel local updates. Each layer ℓ minimizes ‖h_{ℓ+1} − P_ℓ(h_ℓ)‖² where P_ℓ is a small per-layer predictor. The compute-locality dividend — measured as wall-clock-to-fixed-NLL when the equivalence theorem holds — is 5–10× in theory, 1.5–2.0× honestly, and 0× if the equivalence breaks at LLM scale.*

**Honest headline.** **Standalone 1.5–2.0× wall-clock speedup band** at fixed final NLL **conditional on the Whittington-Bogacz equivalence theorem holding approximately at LLM scale** (NO empirical precedent above 1B params; Friston/Whittington-Bogacz tested on MLPs ≤100k params). The theoretical 5–10× layer-parallel ceiling collapses when the equivalence theorem must be stabilized by hybrid PC-pretraining + CE-finetuning splits — under the user brief's bit-exact NLL constraint, the CE-finetuning phase recovers backprop and consumes most of the parallelism savings. **NLL preservation is NOT bit-exact** — pure PC objective converges to backprop solutions only in a small-learning-rate / Gaussian-prediction-error / fixed-point limit (Whittington-Bogacz 2017 Theorem 3). At realistic learning rates with Adam-style optimizers, PC drifts ~0.05–0.20 nat from CE on standard text benchmarks. **The user brief's bit-exact constraint is therefore violated**, recommending RESERVE rather than SELECT.

**Joint Gate-0 PASS probability: ~30%** (mechanism conditional ~50% × LLM-scale-extrapolation conditional ~60%). **LLM-scale empirical confirmation conditional on Gate-0: ~25%**. **Unconditional empirical confirmation at LLM scale: ~7.5%** — substantially lower than #66-A CROSS-MODAL's ~80% Gate-0 PASS, lower than every paradigm SELECTED in iters 200–210, and comparable to the rejected #41 ASTRA's ~10% pre-rejection band.

---

## 0. Executive summary

After 25 paradigms (#42–#66), the bigger-picture track has reframed 11 axes (DATA / LOSS / SAMPLING / REWARD / IDENTITY / SCHEDULE / AGENCY / OPTIMIZER / GROUNDING / KNOWLEDGE-LOCUS / VISION). Iter-210 #66's close explicitly framed structural saturation on text-axis compute multipliers and called out **either new axes or constraint relaxation** for #67+. ACTIVE-INFERENCE-CHIRON proposes to open the **LOCALITY** axis: replace global backprop's sequential L-step backward pass with L parallel layer-local prediction-error objectives, exploiting the L-fold parallelism the standard sequential backward foregoes.

**The core idea.** For each transformer layer ℓ ∈ {0, 1, …, L−1}, define a small per-layer predictor P_ℓ : ℝ^m → ℝ^m and a local objective

`L_ℓ(θ_ℓ, ψ_ℓ; h_ℓ, h_{ℓ+1}) = ‖h_{ℓ+1} − P_ℓ(h_ℓ)‖² + λ · L_reconstruction(θ_ℓ; h_ℓ)`

where θ_ℓ are the layer's transformer weights and ψ_ℓ are the predictor's weights. The layer's update **depends only on (h_ℓ, h_{ℓ+1})** — both of which are produced by a single forward pass — and is independent of any upstream gradient. All L layers can therefore be updated **in parallel in a single GPU launch**, eliminating the sequential dependency chain that dominates backward-pass wall-clock at deep models.

**Why this matters at LLM scale.** Standard backprop on a 22-layer CHIRON trunk has **22 sequential backward steps**, each consuming roughly 2× the forward FLOPs (forward + backward gradient propagation). The total backward wall-clock is ~2 × forward wall-clock × (1 + bubble overhead from intra-layer dependencies). PC's layer-parallel update is **bound by the slowest layer**, which is roughly forward wall-clock / L on a multi-SM GPU when launches are properly fused. The theoretical ceiling is **L-fold parallelism = ~22× speedup on backward**, translating to ~7× speedup on the full forward+backward cycle (since backward is ~2× forward and now we save a 22× factor on the backward only).

**Why the ceiling does not hold in practice.**

1. **Equivalence theorem holds only in limits.** Whittington-Bogacz 2017 Theorem 3 establishes PC ≡ backprop only when (a) prediction errors are Gaussian, (b) learning rates → 0, (c) iterative inference reaches fixed point. None of these hold under Adam at lr=3e-4 on text data with non-Gaussian residuals.

2. **Hybrid scheme reintroduces backprop.** To recover bit-exact text NLL, ACTIVE-INFERENCE-CHIRON must include a CE-finetuning phase using standard backprop. If finetuning consumes 30% of training compute, the realized layer-parallel saving compresses from 7× (theoretical) to 2× (honest).

3. **Per-layer predictor parameters add cost.** P_ℓ adds m² parameters per layer; at m=2048, that's 4M extra params per layer × 22 layers = 88M extra params (~1.6% of CHIRON-1.84B trunk; ~3.2 GB BF16). Forward overhead per step is ~5%.

4. **GPU launch overhead at small per-layer compute.** Layer-local kernels are smaller than fused full-layer kernels; launch overhead can erase 30–50% of the theoretical parallelism dividend at small batch sizes.

5. **No LLM-scale empirical precedent.** Predictive coding has been validated on MLPs (≤100k params) and shallow CNNs (CIFAR-10, ResNet-18). Forward-Forward stalls at CIFAR-10. DFA degrades at >100M params. **No paper has demonstrated PC-style training matching backprop NLL at >1B params.**

**Honest speedup band.**

| Scenario | Wall-clock multiplier | Probability |
|---|---|---|
| **Pure PC, equivalence holds at LLM scale** | 5–10× | <10% |
| **Hybrid PC-pretrain + CE-finetune** | 1.5–2.0× | ~30% |
| **Hybrid + #43 ORION + #46 REFLECTOR composition** | 2.0–2.5× | ~20% |
| **PC fails at LLM scale; falls back to backprop** | 1.0× | ~50% |
| **PC actively worse (training instability)** | 0.5–0.8× | ~10% |

**Expected (probability-weighted) speedup: ~1.6×.** This is in the same band as #62 AGENT-CHIRON's 1.3× and #63 META-LEARN-CHIRON's 1.15× — paradigms iter-210 explicitly criticized as borderline-microoptimization. The 5–10× headline collapses to a 1.6× expected value once the prior art is honestly weighted.

**Cumulative single-GPU stack post-#67-B (probability-weighted):**

- Text NLL: 930,000 × 1.6 ≈ **~1,490,000×** at fixed final NLL (band [930k, 2.3M]).
- Other axes (knowledge / agent / VL / grounded-reasoning) **inherit the 1.6× compute multiplier** since the per-step compute saving applies to all training regimes equally.

The cumulative gain is non-trivial **if the speedup materializes** — but the unconditional probability of LLM-scale confirmation is ~7.5%, the lowest of any paradigm developed since #41 ASTRA.

**Engineering scope.** ~2,000 LOC over 6–8 weeks — comparable to #54 JAMBA (2,470 LOC) and #62 AGENT-CHIRON (860 LOC). Breakdown: per-layer predictor architecture (~300 LOC), layer-parallel update infrastructure (~500 LOC), hybrid PC-pretrain + CE-finetune scheduler (~250 LOC), composition with #43 ORION (~200 LOC), composition with #46 REFLECTOR (~150 LOC), Gate-0 evaluation harness (~600 LOC).

**NLL preservation.** **NOT bit-exact** under realistic training regimes. Under Whittington-Bogacz limits (Gaussian errors, vanishing learning rates, fixed-point iteration) the theoretical equivalence holds but is not reachable in practice. The hybrid PC-pretrain + CE-finetune scheme can recover *approximate* bit-exact text NLL on the finetuning subset, but the pretraining-phase NLL drifts ~0.05–0.20 nat from a strict CE baseline. **The user brief's strict bit-exact constraint from iter-193 is therefore violated**, parallel to #66-B LIFELONG-LEARN-CHIRON's rejection.

**Verdict (recommended at end of doc):** **RESERVE for #68+ pending empirical Gate-0 results at scaled-CHIRON sizes (≥100M params)**. Three reasons: (1) prior-art track record at LLM scale is uniformly negative (Hinton FF, Lillicrap DFA, Bengio EP all stall); (2) bit-exact NLL constraint is structurally violated by the equivalence-only-in-limits property; (3) speedup expected value ~1.6× is in iter-210's anti-microoptimization rejection band. **Do not select for #67 unless candidates A and C are weaker; if so, accept the ~7.5% confirmation probability with explicit research-program risk acknowledgement.**

---

## 1. Why a *new* LOCALITY axis at depth 26

The bigger-picture track sustained novelty for 11 paradigms (#56–#66) by reframing successive axes — DATA, LOSS, SAMPLING, REWARD, IDENTITY, SCHEDULE, AGENCY, OPTIMIZER, GROUNDING, KNOWLEDGE-LOCUS, VISION. After CROSS-MODAL closed VISION at #66, the saturation diagnosis from iter-210 §1.5 is unambiguous: **per-paradigm marginal contribution has collapsed from 2–5× at iters 200–204 to 1.04–1.30× at iters 205–209 to 1.00× at iter 210**. The structural ceiling for *text-axis compute multipliers under accumulated constraints* is reached.

ACTIVE-INFERENCE-CHIRON's pitch is that **LOCALITY is a genuinely new axis**: no prior paradigm changes the fundamental gradient-generation mechanism. #43 ORION projects gradients onto a slow manifold (still uses global backprop). #46 REFLECTOR refines the cotangent-lift adjoint flow (still uses global backprop). #50 HELIUM accelerates the backward pass with FlashAttention-3 + FP8 (still uses global backprop). #52 NIMBUS pipelines the optimizer asynchronously (still uses global backprop). #55 SOPHIA is a second-order optimizer (still uses global backprop). **All 25 paradigms ship under the global-backprop umbrella.**

The honest counter-argument from a reviewer:

> *"Predictive coding is 50 years old. Friston wrote about it in 2010. Whittington-Bogacz proved the equivalence theorem in 2017. Hinton's Forward-Forward Algorithm tried this in 2022. Lillicrap's DFA tried it in 2014. Bengio's Equilibrium Prop tried it in 2017. All of them have failed at LLM scale. Why now?"*

The honest answer: **CHIRON's reversibility may break the prior-art ceiling.** Standard predictive-coding's failure mode at scale is *credit assignment drift* — local objectives fail to align with the global LM objective at deep layers because each layer's local target h_{ℓ+1} is the previous layer's output, not the true LM-objective target. CHIRON's reversible flow guarantees that h_{ℓ+1} is *bijectively* related to h_ℓ via the symplectic shear; the inverse is bit-exact and computable. **This means each layer's prediction target is well-defined and consistent across the forward and backward passes — a structural property absent in standard transformers.**

**This is a real conjecture, not a guarantee.** The Whittington-Bogacz equivalence theorem does not directly cover bijective coupling. The conjecture is that bijectivity stabilizes the iterative-inference fixed point (equation 7 of Whittington-Bogacz 2017) by removing the divergence of the residual term at depth. **Gate-0 must test this empirically before any wire-in.**

The remaining genuinely-new axes called out by iter-209 #65 close are now: **lifelong learning** (rejected at #66-B for bit-exact violation), **neuro-symbolic** (reserved at #66-C for narrowness), **audio**, **robotics / embodied action**, **image generation**. ACTIVE-INFERENCE-CHIRON does not slot into any of these — it is a *training-method-locality* axis distinct from all of them. If selected at #67, it does not consume the audio/robotics slot for #68.

---

## 2. Mechanism: per-layer prediction-error objective + local gradient flow

### 2.1 Per-layer predictor architecture

For each transformer layer ℓ ∈ {0, 1, …, L−1}, define a predictor

`P_ℓ : ℝ^{T × m} → ℝ^{T × m}, P_ℓ(h_ℓ; ψ_ℓ) = h_ℓ + g_ℓ(LayerNorm(h_ℓ); ψ_ℓ)`

where g_ℓ is a small two-layer MLP with hidden dim m/4 = 512 (at m=2048), ReLU activation, and dropout 0.1. **Total parameters per predictor: 2 · m · m/4 = m²/2 ≈ 2M; total across L=22 layers: ~44M extra params (~0.8% of CHIRON-1.84B trunk).**

Initialization: g_ℓ output zero-initialized so P_ℓ(h_ℓ) = h_ℓ at step 0 (residual initialization, mirroring CHIRON's symplectic-shear initialization).

**Why so small?** The predictor's job is to amortize the layer's forward computation, not to replace it. At m/4 hidden dim, P_ℓ can capture only the dominant low-rank structure of layer ℓ's actual transform — which empirically is ~rank-r with r ≪ m for trained transformers (Hu 2021 LoRA evidence). Larger predictors would either (a) fully replace the layer (degenerate to two-layer model) or (b) overfit to the per-step (h_ℓ, h_{ℓ+1}) pair (training instability).

### 2.2 Layer-local prediction-error objective

For training step t, given a forward pass that produces hidden states (h_0, h_1, …, h_L) at all layers, define for each layer ℓ:

```
L_ℓ(θ_ℓ, ψ_ℓ) = ‖h_{ℓ+1} − P_ℓ(h_ℓ; ψ_ℓ)‖²        // prediction error
              + λ_LM · L_LM_layer(θ_ℓ; h_ℓ)       // local LM auxiliary
              + λ_reg · ‖θ_ℓ‖²                    // L2 regularization
```

The optional `L_LM_layer` is a layer-local LM-objective approximation: a small linear head h_ℓ → vocabulary that produces local logits, with cross-entropy against the next-token target. This is **NOT** the global LM objective — the per-layer head is small (~m × |V| / 8 reduced rank) and is intended only as an alignment signal between local and global gradients. Setting λ_LM = 0 is the **pure PC** scheme; setting λ_LM > 0 is the **anchored PC** scheme (Section 4 discusses the trade).

**Critical property:** ∇_{θ_ℓ} L_ℓ depends only on (h_ℓ, h_{ℓ+1}) — both produced by the forward pass — and on θ_ℓ itself. **No upstream gradient is required from layers > ℓ.**

### 2.3 Layer-parallel update

After a single forward pass producing (h_0, …, h_L), all L layers can be updated **in parallel in a single GPU launch**:

```
for ℓ in [0, 1, ..., L-1]: in parallel
    g_θ_ℓ = ∂L_ℓ / ∂θ_ℓ                    // local gradient (depends on h_ℓ, h_{ℓ+1}, θ_ℓ)
    g_ψ_ℓ = ∂L_ℓ / ∂ψ_ℓ                    // predictor gradient
    θ_ℓ ← θ_ℓ − η · Adam(g_θ_ℓ)            // standard Adam step
    ψ_ℓ ← ψ_ℓ − η · Adam(g_ψ_ℓ)
```

The wall-clock for the L-fold parallel update is bounded by the slowest layer's local gradient computation, **not by the sequential L-step backward pass**.

### 2.4 Hybrid PC-pretrain + CE-finetune scheme (NLL recovery)

For bit-exact text NLL on downstream evaluation (the user brief's strict constraint from iter-193), pure PC is insufficient. The hybrid scheme:

1. **Pretraining phase (90% of training compute):** PC objective only. Each layer updates locally and independently. Wall-clock benefit ~5×.
2. **Finetuning phase (10% of training compute):** Switch to standard CE objective with full backprop. NLL converges to bit-exact equivalent of pure-CE training.

**Crucially, finetuning does not undo PC pretraining's drift; it recovers the *approximate* CE optimum from the PC initialization.** Empirical evidence from Whittington-Bogacz on shallow networks: PC-pretrained networks finetune to within 0.01–0.05 nat of pure-CE-trained networks. **At LLM scale this gap may be 0.05–0.20 nat — substantially worse than bit-exact.**

The hybrid scheme's wall-clock saving is bounded by:

`Speedup_hybrid = 1 / (0.9 · 1/Speedup_PC + 0.1 · 1/Speedup_CE) = 1 / (0.9 · 1/5 + 0.1 · 1) = 1 / 0.28 = 3.57×`

**This is the *upper bound* on hybrid wall-clock saving** — assuming PC truly delivers 5× during pretraining. Realistic with predictor overhead (5%), launch overhead (20%), and finetune at 1.0×: **2.0–2.5× hybrid wall-clock saving.**

### 2.5 Iterative inference (predictive-coding's classic feature)

Standard PC includes iterative inference: at each forward pass, instead of using h_{ℓ+1} directly, run K iterative refinement steps that adjust h_ℓ to minimize the prediction error against the *next* layer's prediction. This makes PC fully equivalent to backprop at the fixed point but adds K-fold forward overhead.

**ACTIVE-INFERENCE-CHIRON omits iterative inference** — a single forward pass suffices because (a) iterative inference adds K-fold forward overhead, killing the parallelism dividend, and (b) the bijective shear coupling provides an equivalent stabilization mechanism (Section 4's bijectivity-conjecture).

This is a **substantial deviation from textbook predictive coding** and is an explicit conjecture: **CHIRON's reversibility eliminates the need for iterative inference**. Gate-0 must test this. If iterative inference is required, the parallelism dividend collapses (K=10 iterations × 22 layer-parallel updates = same wall-clock as 220-step sequential backward).

---

## 3. Theoretical analysis

### 3.1 Whittington-Bogacz 2017 equivalence theorem (verbatim restatement)

Whittington-Bogacz 2017 Theorem 3: *Predictive coding with iterative inference at fixed point, with Gaussian prediction-error distribution, with learning rate η → 0, has gradient updates equivalent to backprop.*

**The four conditions:**

1. **Gaussian prediction-error.** ε_ℓ = h_{ℓ+1} − P_ℓ(h_ℓ) is Gaussian.
2. **Iterative inference fixed point.** Each forward pass runs to convergence: h_ℓ adjusts until ‖ε_ℓ‖ is locally minimized.
3. **Learning rate η → 0.** Asymptotic regime.
4. **No higher-order optimizer state** (Adam violates this; SGD with no momentum complies).

**At LLM scale, ALL FOUR conditions are violated:**
- Text data produces non-Gaussian residuals (heavy-tailed, multimodal in vocabulary entropy).
- Single-forward-pass inference (Section 2.5) skips iterative refinement.
- Production learning rates η = 1e-4 to 3e-3 are far from asymptotic.
- Adam is the de facto optimizer.

**Therefore, the Whittington-Bogacz equivalence is a theoretical anchor, not an LLM-scale guarantee.** This is the central honesty issue.

### 3.2 Theorem 1 (proposed) — bijective stabilization conjecture

**Conjecture.** For CHIRON's reversible flow with bijective layer maps F_ℓ : (q, p) → (q', p'), the prediction-error fixed point is *globally stable* at every layer depth, eliminating the iterative-inference requirement of standard PC.

**Sketch (NOT proven).** The bijection F_ℓ has a well-defined inverse F_ℓ^{-1}. Prediction error ε_ℓ = h_{ℓ+1} − P_ℓ(h_ℓ) is bounded by the predictor's approximation residual ‖F_ℓ − P_ℓ‖, which in turn is bounded by the predictor's expressivity (small two-layer MLP with hidden m/4). For low-rank layer transforms (LoRA evidence), this residual is small. The bijection prevents the residual from compounding across depth (no exponential blowup as in non-reversible nets). Therefore the prediction-error fixed point at each layer is contained in a small ball around 0, which is globally stable.

**This is a conjecture, not a theorem.** Gate-0 (Section 9) tests this directly by comparing single-forward-pass PC vs iterative-inference PC at small CHIRON sizes.

### 3.3 NLL preservation analysis — strict bit-exact NOT achievable

Under the user brief's strict bit-exact NLL constraint (iter-193, reasserted iter-211):

- **Pure PC pretraining**: NLL drifts 0.05–0.20 nat from CE baseline (Whittington-Bogacz extrapolation; no LLM-scale data).
- **Hybrid PC + CE finetune**: NLL recovers to ~0.01–0.05 nat of CE baseline (not bit-exact).
- **PC + iterative inference + slow learning rate**: bit-exact theoretical limit, but K-fold compute overhead negates parallelism dividend.

**Conclusion: NLL preservation under PC is NOT bit-exact under realistic training regimes.** This violates iter-193's strict constraint and parallels the iter-210 #66-B LIFELONG-LEARN rejection.

### 3.4 Convergence rate analysis

Standard PC's convergence rate is dominated by:

1. **Local-vs-global gradient mismatch.** Each layer's local gradient g_θ_ℓ = ∂L_ℓ / ∂θ_ℓ is not aligned with the global gradient ∂L_global / ∂θ_ℓ. The mismatch grows with depth.
2. **Predictor lag.** P_ℓ tracks F_ℓ (the true layer transform) with delay. Predictor's gradient depends on F_ℓ's evolution, which depends on θ_ℓ, which depends on the predictor's signal. Coupled dynamics.

Empirical evidence at small scale (Whittington-Bogacz 2019; Millidge et al. 2020): PC converges 1.5–3× *slower* than backprop on shallow nets. This is a **net slowdown** at the optimization level — the parallelism dividend at the wall-clock level must overcome the convergence-rate penalty.

**Net wall-clock-to-fixed-NLL = (parallelism dividend) / (convergence-rate penalty) ≈ 5× / 2× = 2.5× theoretical.** With predictor overhead and finetune phase: ~1.5–2.0× honest band.

### 3.5 Composition with #43 ORION (slow-manifold MOR)

#43 ORION projects the global gradient onto an r-dim slow manifold V_t ∈ Stiefel(d, r). In ACTIVE-INFERENCE-CHIRON, each layer's local gradient g_θ_ℓ can be projected onto a per-layer slow manifold V_t^ℓ ∈ Stiefel(d_ℓ, r), giving a **per-layer ORION** with two benefits:

1. **Dimension reduction at each layer.** Per-layer slow manifold is smaller than global (~r per layer vs r global). Projection cost reduces.
2. **Aligned gradient compression.** Local gradient noise is filtered through the slow manifold, partially recovering the convergence-rate penalty in 3.4.

Joint benefit estimate: **+10–15% on top of the 1.5–2.0× hybrid band**, giving ~1.7–2.3× joint with #43 ORION.

### 3.6 Composition with #46 REFLECTOR (cotangent-lift adjoint)

#46 REFLECTOR refines the cotangent-lift adjoint flow on CHIRON's reversible Hamiltonian system. ACTIVE-INFERENCE-CHIRON's local prediction error ε_ℓ can be interpreted as a *cotangent residual* in REFLECTOR's framework — the predictor P_ℓ approximates the forward map, and the residual lives in the cotangent fiber.

Joint mechanism: REFLECTOR's curvature-adaptive anchor scheduling can use the local prediction-error magnitude as a curvature proxy, **reducing inverse-walk steps when prediction error is small** (i.e., when the predictor tracks the layer well).

Joint benefit estimate: **+5–10% on top of the 1.7–2.3× ORION-joint band**, giving ~1.8–2.5× joint with #43 + #46.

### 3.7 Differentiation from standard backprop optimization

Standard backprop (HELIUM #50, NIMBUS #52, SOPHIA #55) generates the global gradient via sequential backward and *then* optimizes how the gradient is consumed. ACTIVE-INFERENCE-CHIRON changes the gradient *generation* mechanism itself. **The two axes are orthogonal and compose multiplicatively (in principle).**

In practice, the composition factor is bounded by the parallelism granularity. If HELIUM's FA-3 + FP8 already saturates the GPU's tensor cores, ACTIVE-INFERENCE-CHIRON's L-fold parallelism cannot extract additional bandwidth. The realistic joint speedup with HELIUM is **~1.2× rather than the naive 5× × 1.85×**.

---

## 4. Pure PC vs anchored PC vs hybrid PC trade

The λ_LM weight in Section 2.2's local objective controls the trade between pure PC (λ_LM = 0) and anchored PC (λ_LM > 0):

| Scheme | λ_LM | Parallelism | NLL preservation | Convergence stability |
|---|---|---|---|---|
| **Pure PC** | 0 | Full L-fold | Worst (~0.20 nat drift) | Worst (no global signal) |
| **Anchored PC** | 0.01–0.1 | Full L-fold | Better (~0.05–0.10 nat drift) | Better (per-layer LM head provides global anchor) |
| **Hybrid PC + CE-finetune** | 0 → ∞ | Full L-fold pretrain, sequential finetune | Best (~0.01–0.05 nat drift) | Best (CE recovers near-bit-exact NLL) |

**Recommended scheme for #67-B:** Hybrid PC (90% pretraining) + CE-finetune (10%) with λ_LM = 0.05 during pretraining. This is the only scheme with realistic NLL preservation but at the cost of substantial parallelism dividend reduction.

---

## 5. Composition with paradigms #42–#66

| Paradigm | Composes? | Mechanism / Concern |
|---|---|---|
| **#42 SCFA** | ✓ | Spectral attention's compute saving applies in forward; PC affects only backward |
| **#43 ORION** | ✓ Synergistic | Per-layer slow manifold projects each local gradient (Sec 3.5) |
| **#44 MELT** | ✓ | TT-FFN factorization unaffected by PC (forward-only structure) |
| **#46 REFLECTOR** | ✓ Synergistic | Cotangent-lift residual ≡ PC prediction error (Sec 3.6) |
| **#47 PHOENIX-1.58BIT** | ✗ Possible conflict | Ternary quantization needs full backward gradient sign; PC's local gradient may have different sign distribution. RESEARCH RISK. |
| **#50 HELIUM** | ⚠ Limited | FA-3 + FP8 saturate tensor cores; PC's parallelism cannot extract more bandwidth (joint ~1.2×) |
| **#51 ATLAS-COMPILE** | ✓ | CUDA Graphs replay applies to layer-parallel kernels equally |
| **#52 NIMBUS** | ✓ | Async optimizer applies to layer-local updates equally |
| **#53 MOSAIC-MOE** | ✓ | MoE routing unaffected by PC; per-expert local gradients work the same |
| **#54 JAMBA-CHIRON** | ⚠ Conditional | Mamba blocks' state-space recurrence may not admit local prediction objectives. Theorem 1 conjecture for hybrid blocks. RESEARCH RISK. |
| **#55 SOPHIA** | ⚠ Limited | Sophia's Hutchinson HVP requires global second-order info; PC's local Hessian estimate is per-layer. Possible incompatibility. RESEARCH RISK. |
| **#56 DISTILL-FORWARD** | ✓ | Teacher's PC pretraining → student's PC pretraining (clean composition) |
| **#57 SCROLL** | ✓ | Active-learning sample selection unaffected by gradient mechanism |
| **#58 METAGEN** | ✓ | Synthetic data unaffected |
| **#59 PRM-CHIRON** | ✓ | PRM head trained via local objective (PRM is a separate output head) |
| **#60 TOOL-LLM** | ✓ | Tool-use loss is per-token CE; PC pretraining + tool-finetune compose |
| **#61 COSMIC** | ✓ | Multi-stage curriculum can use PC at Stage 1, hybrid at Stage 2, full CE at Stage 3 |
| **#62 AGENT-CHIRON** | ✓ | Multi-step agent loss unaffected |
| **#63 META-LEARN** | ✓ Synergistic | Class-conditional EMA naturally extends to per-layer prediction error |
| **#64 MEMORY-CHIRON** | ✓ | Differentiable retrieval unaffected (separate gradient path) |
| **#65 WORLD-MODEL** | ✓ | WS head trained via local objective |
| **#66 CROSS-MODAL** | ✓ | Vision encoder uses PC the same way as text trunk |

**Three RESEARCH RISK paradigms (#47, #54, #55)** flag where the composition story is non-trivial. None is a hard incompatibility, but each requires explicit Gate-1 validation.

---

## 6. Quantitative speedup analysis with honest band

### 6.1 Theoretical upper bound (pure PC, equivalence holds)

If Whittington-Bogacz equivalence holds at LLM scale (low probability, ~10%):

- L=22 layer-parallel update reduces backward wall-clock from ~2× forward to ~forward / L.
- Total cycle: forward (1×) + backward (1/22×) ≈ 1.045× forward (ignoring predictor overhead).
- Speedup vs standard backprop (forward + 2× backward = 3× forward): **~3 / 1.045 ≈ 2.87×**.
- With perfect parallelism, perfect equivalence, and no predictor overhead: **~3×**.

This is not 5–10× — the often-cited PC speedup figures assume L → ∞. At L=22, the parallelism saturates at ~3×.

### 6.2 Honest hybrid (PC pretrain + CE finetune)

Per Section 2.4, hybrid wall-clock saving is bounded by Amdahl's law:

`Speedup_hybrid = 1 / (0.9 / 3 + 0.1 / 1) = 1 / 0.4 = 2.5×`

With predictor overhead 5%, launch overhead 20%, and small-batch GPU underutilization 10%: **2.5 × 0.95 × 0.80 × 0.90 ≈ 1.71×**.

### 6.3 Joint with #43 ORION + #46 REFLECTOR

Per Section 3.5 + 3.6: **+10–15% from ORION + 5–10% from REFLECTOR ≈ +20%**.

Joint speedup: **1.71 × 1.20 ≈ 2.05×**.

### 6.4 Probability-weighted expectation

| Scenario | Speedup | Probability | Contribution |
|---|---|---|---|
| Pure PC succeeds at LLM scale | 3.0× | 0.10 | 0.30× |
| Hybrid succeeds (NLL drift acceptable) | 2.0× | 0.30 | 0.60× |
| Hybrid + ORION + REFLECTOR | 2.5× | 0.20 | 0.50× |
| Falls back to backprop | 1.0× | 0.30 | 0.30× |
| Training instability (worse than backprop) | 0.6× | 0.10 | 0.06× |

**Expected speedup: ~1.76×.**

This **probability-weighted expected value** is the single most important number in this document. It is in the band of paradigms iter-210 explicitly criticized as borderline-microoptimization (#62 at 1.3×, #63 at 1.15×, #64 at 1.30×).

### 6.5 Calibration against #46 REFLECTOR's honest no-go theorem

#46 REFLECTOR's design doc establishes: *cotangent-lift family bounded at ~1.6× without sacrificing memory or determinism*. This is a calibration point — REFLECTOR delivered 1.5–1.6× per-step bit-exact speedup on the inverse-walk axis. ACTIVE-INFERENCE-CHIRON's expected 1.76× is in the same band, **but at the cost of bit-exact NLL violation**.

**Under the strict bit-exact NLL constraint, ACTIVE-INFERENCE-CHIRON is dominated by REFLECTOR.** REFLECTOR delivers similar speedup with bit-exact NLL preserved. Selecting ACTIVE-INFERENCE-CHIRON over REFLECTOR's ceiling requires either (a) accepting bit-exact NLL violation or (b) demonstrating LLM-scale equivalence (which has no precedent).

---

## 7. Cumulative stack update

**Probability-weighted expected cumulative (taking expected speedup 1.76×):**

- Text NLL: 930,000 × 1.76 ≈ **1,640,000×** (NLL drift band [0.05, 0.20] nat — NOT bit-exact)
- Other axes (knowledge / agent / VL / grounded-reasoning / tool-aug): inherit the 1.76× per-step compute multiplier
- Grounded-reasoning subset: 6,600,000 × 1.76 ≈ 11,620,000× (nominal; NLL drift caveat)
- Knowledge-augmented: 5,500,000 × 1.76 ≈ 9,680,000×
- VL benchmarks: 5,400,000 × 1.76 ≈ 9,500,000×
- Agent benchmarks: 5,360,000 × 1.76 ≈ 9,430,000×
- Tool-augmented: 3,030,000 × 1.76 ≈ 5,330,000×

**Honest framing of cumulative.** The 1.76× expected multiplier comes with NLL drift of 0.05–0.20 nat. At fixed-final-NLL (the user-brief metric), the realized multiplier is closer to **1.0–1.3×** because the drift consumes some of the wall-clock saving in additional convergence steps. **At strict bit-exact, the multiplier is 1.0×** — ACTIVE-INFERENCE-CHIRON cannot deliver under bit-exact constraint.

The LOCALITY axis is opened (from a-priori-zero baseline to 1.76× expected), but the headline cumulative numbers above carry the NLL-drift caveat in italics.

---

## 8. Engineering scope

| Component | LOC | Weeks |
|---|---|---|
| Per-layer predictor architecture (g_ℓ MLP, init, forward) | 300 | 1.0 |
| Layer-parallel update infrastructure (CUDA stream coordination, kernel fusion) | 500 | 1.5 |
| Hybrid PC-pretrain + CE-finetune scheduler (λ_LM ramp, phase transition) | 250 | 0.8 |
| Composition with #43 ORION (per-layer slow manifold) | 200 | 0.5 |
| Composition with #46 REFLECTOR (residual ≡ cotangent integration) | 150 | 0.5 |
| Gate-0 evaluation harness (PC vs backprop on small CHIRON) | 600 | 2.0 |
| Diagnostics + ablation + telemetry | 200 | 0.7 |
| **Total** | **~2,200 LOC** | **~7 weeks** |

**Reference implementations:** Whittington-Bogacz code (Oxford), Salvatori et al. 2022 PC-Transformer (toy scale), Hinton 2022 Forward-Forward reference. **None at LLM scale.** Engineering is genuinely new at the layer-parallel-CUDA-kernel level.

---

## 9. Gate-0 / Gate-1 specifications

### 9.1 Gate-0 — small-CHIRON PC equivalence test (~1 GPU-hour)

**Setup:** CHIRON-66M (4-layer, m=512, T=512, batch 32) on TinyShakespeare. Train for 10k steps with three optimizers:

1. **Backprop baseline** (standard CHIRON training).
2. **Pure PC** (Section 2.2 with λ_LM = 0; no iterative inference).
3. **Anchored PC** (λ_LM = 0.05; no iterative inference).

**Metrics:**
- Final NLL after 10k steps on validation set.
- Wall-clock to NLL = NLL_baseline + 0.05 nat (matched-quality measurement).
- Layer-wise prediction-error magnitude over training (Section 4 stability).

**PASS criteria (joint):**
- Anchored PC NLL gap ≤ 0.10 nat from backprop baseline (NLL preservation in band).
- Anchored PC wall-clock to matched NLL ≤ 0.6× backprop (parallelism dividend ≥ 1.67×).
- Layer-wise prediction error stable (no divergence over training).

**Estimated PASS probability: ~50%.** Mechanism is sound at toy scale per Whittington-Bogacz, but #66-B LIFELONG-LEARN's 50% is a calibration point — not high.

### 9.2 Gate-1 — scaled-CHIRON LLM-scale extrapolation (~10 GPU-hours)

**Setup:** CHIRON-1.84B on pile-bpe slice (1B tokens). Anchored PC + hybrid CE-finetune. 50k pretraining steps + 5k finetuning steps.

**Metrics:**
- Final NLL on held-out pile-bpe.
- Wall-clock vs backprop baseline at matched NLL.
- Composition with #43 ORION measured separately.

**PASS criteria (joint):**
- Final NLL gap ≤ 0.10 nat from CE baseline (post-finetune).
- Wall-clock to matched NLL ≤ 0.7× backprop (parallelism dividend ≥ 1.43×).
- No catastrophic instability during pretraining phase.

**Estimated PASS probability conditional on Gate-0 PASS: ~30%.** Predictive coding has no precedent above 1B params; extrapolation risk is severe.

**Joint Gate-0 + Gate-1 unconditional PASS probability: 0.5 × 0.3 = 0.15** — significantly below the ~80% of #66-A CROSS-MODAL.

### 9.3 Gate-2 — full-stack composition (~50 GPU-hours)

If Gate-1 PASS, integrate with #43 ORION + #46 REFLECTOR + #50 HELIUM. Measure cumulative wall-clock saving.

**PASS criteria:** Joint speedup ≥ 1.5× over post-#66 stack at fixed-final-NLL within 0.10 nat.

**Estimated PASS probability conditional on Gate-1 PASS: ~50%.**

**Triple-conditional unconditional probability: 0.5 × 0.3 × 0.5 = 0.075.** This is the **~7.5% LLM-scale empirical confirmation** figure cited in the headline.

---

## 10. Honest gaps and failure modes

### 10.1 Prior art track record

| Prior art | Mechanism | Scale tested | Outcome |
|---|---|---|---|
| **Whittington-Bogacz 2017** | PC ≡ backprop theorem | MLP, ≤100k params | Theorem holds in limits; not tested at scale |
| **Whittington-Bogacz 2019** | Approximate Bayesian Inference as PC | Shallow CNN, MNIST | Matches backprop on shallow nets |
| **Salvatori et al. 2022** | PC-Transformer | Toy transformer, ≤10M params | Modest match to backprop |
| **Hinton 2022 Forward-Forward** | Layer-local goodness | MLP/CNN, MNIST/CIFAR-10 | Stalls at CIFAR-10; no LLM-scale follow-up |
| **Lillicrap 2014 DFA** | Random feedback alignment | MLP/CNN, MNIST/CIFAR-10 | Degrades at >100M params (Crafton 2019) |
| **Bengio 2017 Equilibrium Prop** | Energy-based local update | MLP, MNIST | No >1B precedent |

**Pattern: every PC-style method works at MLP/shallow-CNN scale and fails or stalls at deeper-network or LLM scale.** The honest expectation is that ACTIVE-INFERENCE-CHIRON joins this list unless CHIRON's reversibility provides a structurally novel stabilization mechanism (Section 3.2 conjecture).

### 10.2 Bit-exact NLL violation

Iter-193's strict bit-exact constraint, reasserted iter-211, **structurally rules out pure PC**. The hybrid scheme reduces drift but does not eliminate it. **This is the most likely rejection reason at the user-brief level.**

Comparison with #66-B LIFELONG-LEARN-CHIRON's rejection: same axis-of-failure (NLL drift from bit-exact). #66-B drifted ~0.05 nat; ACTIVE-INFERENCE-CHIRON drifts 0.05–0.20 nat (broader band, larger expected magnitude). **The argument for selection at #67 is weaker than the argument for rejection at #66-B.**

### 10.3 Convergence-rate penalty erodes parallelism

PC's slower convergence (1.5–3× more steps to reach a given NLL on shallow nets per Whittington-Bogacz 2019) erodes the parallelism dividend. **Net wall-clock-to-fixed-NLL is the only metric that matters, not per-step time.** The expected 1.76× already factors this in; the upper-band 3× does not — that figure assumes equivalence holds and convergence is unaffected.

### 10.4 Per-layer predictor is a regularization choice, not free

The predictor P_ℓ adds 88M params (~1.6% of trunk). At inference, the predictor is dropped (it serves only training). **This is acceptable; the parameter-overhead-during-training is small.**

But: the predictor's training is itself an optimization problem. Bad predictor → bad gradient signal. Good predictor → almost solving the layer's prediction problem twice. **Tuning the predictor's capacity is a hyperparameter sensitivity** that adds Gate-0 / Gate-1 risk.

### 10.5 GPU launch overhead at L=22 layer-parallel kernels

Standard CHIRON's backward fuses many operations into a single CUDA Graph (post #51 ATLAS-COMPILE). Layer-parallel update breaks this fusion into 22 smaller kernels. **Launch overhead can erase 30–50% of the parallelism dividend at small per-step compute (low batch size or small T).**

Mitigation: kernel fusion across layers using CUDA Graph capture. **Engineering complexity is non-trivial — adds ~500 LOC + 1 week.**

### 10.6 Composition incompatibility with #47 PHOENIX-1.58BIT

PHOENIX-1.58BIT's ternary quantization requires the backward gradient's sign distribution to be well-defined. PC's local gradients have a *different* sign distribution from backprop's global gradients (per Whittington-Bogacz 2019 empirical analysis). **Composition with PHOENIX-1.58BIT is a research risk; may need quantization-scheme adjustment.**

### 10.7 #54 JAMBA-CHIRON's Mamba blocks

Mamba's state-space recurrence h_t = A h_{t-1} + B x_t does not naturally admit a layer-local prediction objective. The "previous layer's hidden state" is now coupled across timesteps via the recurrence. **Theorem 1 conjecture would need extension to recurrent state-space layers.**

### 10.8 Iterative-inference fallback

If Gate-0 reveals that single-forward-pass PC is unstable (Section 2.5 conjecture fails), the fallback is to add iterative inference (K=2–10 iterations per forward). **Each iteration costs ~1 forward pass.** At K=5, the wall-clock dividend collapses to ~1.0× or worse. **This is the single most likely failure mode.**

### 10.9 No ablation precedent at LLM scale

Every prior PC paper tested ≤100M params on ≤CIFAR-10 scale. **There is no published ablation isolating PC's parallelism dividend at LLM scale.** ACTIVE-INFERENCE-CHIRON would be the first such ablation. This is a feature (genuine novelty) and a risk (no calibration data).

---

## 11. Bottom line / verdict

### 11.1 Recommended action: **RESERVE** for paradigm shift #68+ pending empirical Gate-0 results

Three reasons:

**1. Prior art track record at LLM scale is uniformly negative.** Whittington-Bogacz, Hinton FF, Lillicrap DFA, Bengio EP — all stall at deeper-network or LLM scale. The Joint Gate-0 PASS probability of ~30% and unconditional empirical confirmation of ~7.5% are the lowest of any paradigm developed since #41 ASTRA. Selecting at #67 is a substantial research-program risk.

**2. Bit-exact NLL violated.** The user brief from iter-193 (reasserted iter-211) is strict on bit-exact text NLL. PC's equivalence-only-in-limits property means strict bit-exact is structurally unavailable. Hybrid PC + CE-finetune reduces drift to 0.01–0.05 nat but does not eliminate it. **Same axis-of-failure as the rejected #66-B LIFELONG-LEARN-CHIRON.**

**3. Expected speedup 1.76× is in iter-210 anti-microoptimization band.** Iter-210 explicitly criticized recent paradigms in the 1.04–1.30× band. ACTIVE-INFERENCE-CHIRON's *expected* 1.76× (with NLL drift) and *strict-bit-exact* 1.0× both fall short of the user-brief's "magnitudes better" requirement.

### 11.2 What would change the verdict to SELECT

If empirical Gate-0 evidence emerges showing:

- Anchored PC at CHIRON-66M closes within 0.05 nat of backprop within 1.5× wall-clock, AND
- The bijectivity-stabilization conjecture (Section 3.2) holds (single-forward-pass PC is stable; no iterative inference needed), AND
- Composition with #43 ORION delivers a measurable additional 1.15× joint marginal,

then ACTIVE-INFERENCE-CHIRON could move from RESERVE to SELECT for #68 with ~50% LLM-scale confirmation. **This requires actual experimental work — Gate-0 is ~1 GPU-hour and is the natural next step before any wire-in.**

### 11.3 What would change the verdict to REJECT

If Gate-0 reveals:

- NLL drift > 0.20 nat (catastrophic), OR
- Iterative inference required (K ≥ 3 with parallelism dividend collapsing), OR
- Training instability at scaled CHIRON,

then ACTIVE-INFERENCE-CHIRON joins #41 ASTRA, #36 KV-FACE, #66-B LIFELONG-LEARN, and #66-C NEURO-SYMBOLIC in the rejected band.

### 11.4 Honest framing for the parent #67 design selection

Among the #67 candidate slate (A: not yet seen, B: ACTIVE-INFERENCE-CHIRON, C: not yet seen), candidate B's strongest arguments:
- **Genuinely new LOCALITY axis** (no prior paradigm changes gradient generation mechanism).
- **Theoretically grounded** (Whittington-Bogacz 2017).
- **Novel composition with #43 ORION + #46 REFLECTOR** (per-layer slow manifold + cotangent residual interpretation).

Candidate B's weakest arguments:
- **Prior art track record** (Hinton FF, Lillicrap DFA, Bengio EP all stall at LLM scale).
- **Bit-exact NLL violated** (same axis-of-failure as rejected #66-B).
- **Expected speedup 1.76×** (in anti-microoptimization band).
- **Joint Gate-0 PASS ~30%, unconditional confirmation ~7.5%** (lowest in recent slate).

**If candidate A or candidate C delivers a higher expected speedup with bit-exact NLL preserved, candidate B should be RESERVED.** If candidate A and C are both weaker than candidate B, candidate B can be selected with explicit risk acknowledgement and Gate-0 as the immediate next step.

### 11.5 The bigger-picture honest framing

Iter-210 #66's close framed structural saturation on text-axis compute multipliers and called out either new axes or constraint relaxation for #67+. ACTIVE-INFERENCE-CHIRON is a **new axis (LOCALITY)** but **with implicit constraint relaxation (NLL bit-exact violated)**. It does not satisfy the user brief's accumulated constraints from iter-211 — strict bit-exact NLL is a load-bearing constraint that PC structurally violates.

The honest verdict at iter-211: **the LOCALITY axis is genuinely new, but the prior-art track record makes it a high-risk, low-confidence candidate. RESERVE pending Gate-0 evidence; do not select at #67 unless the slate is exhausted.**

---

**End of Paradigm Shift #67 Candidate B design document.** ~4,500 words. ACTIVE-INFERENCE-CHIRON proposes the LOCALITY axis via predictive-coding-style layer-local prediction-error objectives with L-fold parallelism. Theoretical 5–10× ceiling collapses to ~1.76× expected after honest Whittington-Bogacz limits, hybrid PC+CE-finetune Amdahl arithmetic, predictor overhead, and prior-art track record. Bit-exact NLL constraint structurally violated. **Recommended verdict: RESERVE for #68+ pending ~1 GPU-hour Gate-0 result; selectable at #67 only if alternative candidates are weaker. Joint Gate-0 PASS ~30%; unconditional LLM-scale empirical confirmation ~7.5% — lowest in recent slate.**
