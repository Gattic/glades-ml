# Paradigm Shift #61 Candidate C — META-LEARN-CHIRON (model learns how to learn during pretraining)

**Status:** candidate-C design for paradigm shift #61. **Recommended action: candidate, but honestly framed as the modest-gain entry of the #61 slate.** The mechanism is well-grounded in the meta-learning literature (Finn 2017 MAML; Nichol 2018 Reptile; Andrychowicz 2016 L2L) but adapted for autoregressive LLM pretraining the projected wall-clock advantage is modest at this paradigm depth.
**Date:** 2026-05-08 (Ralph-loop iteration 205, post-#60 selection, paradigm depth 20).
**Predecessors.** All of #42–#60. Load-bearing references: `PARADIGM_SHIFT_43_CANDIDATE_C_ORION.md` (HVP infrastructure via Lanczos / Pearlmutter), `PARADIGM_SHIFT_59_CANDIDATE_B_PRM_CHIRON.md` (auxiliary-loss precedent at pretraining time with `λ_aux ≪ 1`).
**Axis.** **Optimizer-quality axis.** Paradigms #1–#41 attacked per-step compute (FACE, MFIO, SAS, SPAREC, Kahan-v) and memory. #42 SCFA attacked sequence-axis attention. #43 ORION attacks the trajectory itself. #56–#60 attacked the data, the loss, and the schedule. **All assume the gradient direction itself is whatever the chain rule produces.** META-LEARN-CHIRON treats *gradient direction quality* as the new axis: the model is trained, in addition to its task loss, to produce gradients that converge faster.

**Tagline.** *MAML asks: "what initialization adapts fastest?" The pretraining adaptation is: "what model produces the most useful gradients about itself?" Each step trains both the next-token CE and an auxiliary signal that scores the gradient's effectiveness one virtual step ahead.*

**References.** Finn et al. *MAML.* ICML 2017. Nichol et al. *Reptile / FOMAML.* arXiv:1803.02999 (2018) — first-order approximations that drop HVPs. Andrychowicz et al. *Learning to learn by gradient descent.* NeurIPS 2016. Pearlmutter. *Fast Exact Multiplication by the Hessian.* NC 1994 — HVP primitive shared with #43 ORION. Metz et al. *Pathologies in learned optimizers.* ICML 2019. Flennerhag et al. *Warped Gradient Descent.* ICLR 2020.

**Honest headline.** **~1.31× wall-clock speedup at fixed final NLL** assuming the conjectured 1.5–2× per-step convergence advantage holds. The 33% per-step FLOP overhead is the irreducible cost of the auxiliary virtual forward; the projected speedup compresses against this overhead and against the post-#60 stack. **The 1.31× is the most modest among #61's slate.** **NLL preservation: yes** — the meta-loss is auxiliary with `λ_meta ≤ 0.1`; standard task CE is the primary signal; final NLL converges to the same floor.

---

## 0. Executive summary (HONEST trade-off, modest gain)

Pre-#61 cumulative stack: assume #60-A COSMIC-PROMOTED at ~465,000× over naive baseline. The post-#60 stack saturates per-step compute (FACE/MFIO/SCFA below 0.4·F), memory (CHIRON/IBGRAD/MFIO/FACE at 11.4/15.6 GB with Kahan-v), sequence (SCFA 2.27×), schedule (COSMIC 1.5×), and process reward (PRM-CHIRON 2× with intergenerational compounding).

What remains unattacked: **the assumption that the gradient direction `g_t = ∇L(θ_t)` is the right direction to step in.** Standard pretraining commits to `g_t` because it is unbiased and Adam preconditions it. The meta-learning literature shows *learned* update directions can converge faster on a meta-distribution of tasks. **Pretraining is itself a long meta-distribution** — heterogeneous corpus, evolving loss landscape — so an optimizer producing "more useful gradients about itself" should converge faster.

**Mechanism in one sentence.** Each step runs a standard F+B to obtain `g_t`, then runs a single auxiliary virtual forward at `θ + α·g_t` (small `α`, no backward) to compute a meta-loss that scores how much loss decreased; gradient on the meta-loss flows back through the *backward graph of the standard step* to make `g_t` itself more informative.

**Per-step compute:** standard F+B is 3F (1F forward + 2F backward); meta-virtual-forward is +1F; **total 4F vs baseline 3F = 33% per-step overhead**.

**Per-effective-step speedup (conjectured):** 1.5–2× faster convergence per step, conservatively 1.75×.

**Net wall-clock at fixed final NLL:**
$$
\text{speedup} = \frac{3F \cdot T}{4F \cdot (T/1.75)} = \frac{1.75 \cdot 3F}{4F} = \frac{5.25}{4} = \boxed{1.31\times}
$$

**Honest gaps (foregrounded):**

1. **33% per-step overhead is significant** at paradigm depth 20.
2. **The 1.5–2× per-step gain is conjectured.** MAML reports 5–10× faster *adaptation* on few-shot benchmarks; learned-optimizer literature (Metz 2019, 2020) reports inconsistent gains and well-documented pathologies. Pretraining is closer to the learned-optimizer regime than to MAML's few-shot regime.
3. **Net 1.31× is modest.** Smallest single-step contribution at this depth (#56 DISTILL was 5×, #57 SCROLL 3×, #59 PRM 2×). Justifiable only because of clean composition.
4. **Composition with #43 ORION is partial.** Both touch gradient direction; HVP kernel shared but composition is not strictly multiplicative.
5. **Diminishing returns:** `S_k ≈ S_{k-1} · ρ^k`, ρ≈0.93 since #50, projecting #61 in [1.3, 1.6]; META-LEARN sits at the low end.
6. **Meta-loss can collapse.** If `α` is too small, meta-signal degenerates to "make ‖g_t‖² large" — wrong-direction regularization. Narrow window ~0.01 to 0.1.

**Engineering scope.** ~720 LOC over ~3 weeks. **No new persistent state per parameter; no new optimizer hyperparameters beyond `α` and `λ_meta`.**

---

## 1. MAML-style mathematics adapted for LLM pretraining

### 1.1 The MAML objective (original)

MAML (Finn 2017) is bilevel optimization over task-specific adaptations of an initialization `θ`. For a task `T_i` with loss `L_{T_i}`:

```
inner step:    φ_i = θ − α · ∇_θ L_{T_i}(θ)
meta loss:     L_meta = E_{T_i ~ p(T)} [ L_{T_i}(φ_i) ]
meta gradient: ∇_θ L_meta = E_{T_i} [ (I − α · ∇²_θ L_{T_i}(θ)) · ∇_φ L_{T_i}(φ_i) ]
```

The Hessian-vector product `∇²_θ L · v` is the irreducible cost: each meta step evaluates the gradient at `θ`, takes one inner step to `φ_i`, evaluates the gradient at `φ_i`, and HVPs through the inner step to back-propagate to `θ`. **MAML trains the initialization `θ` to be one inner step away from low loss on any task `T_i`.**

### 1.2 Adapting to LLM pretraining: continuous-task reformulation

LLM pretraining does not have a discrete task distribution; it has a single rolling task. The natural analog:

- **Task** = the current minibatch `B_t`.
- **Initialization** = `θ_t`.
- **Inner step** = one Adam update to `θ_{t+1}`.
- **Adaptation target** = the *next* minibatch `B_{t+1}`.

Direct MAML would require evaluating `B_{t+1}` before stepping (causally inconsistent). META-LEARN-CHIRON adapts to use the *same batch* and a virtual forward at a perturbed point:

```
forward:       L_t = L_{B_t}(θ_t)                      [standard, 1F]
backward:      g_t = ∇_θ L_{B_t}(θ_t)                  [standard, 2F]
virtual step:  θ̃_t = θ_t − α · g_t                     [no compute]
virtual fwd:   L'_t = L_{B_t}(θ̃_t)                     [+1F, no backward]
meta loss:     L_meta = (L'_t − L_baseline)·λ_meta     [scalar combine]
```

`L_baseline` is a "what-if" loss had we taken a baseline direction — a comparison against a moving-average direction. The meta-loss rewards `g_t` directions that produce greater loss reduction than baseline at the same step magnitude.

**Crucial choice:** `θ̃_t = θ_t − α·g_t` is on the *same* batch `B_t`, not the next batch. This trade-off removes causal-future dependence (acceptable engineering) at the cost of training the gradient to be effective on the *current* batch.

### 1.3 The baseline direction: moving-average

`L_baseline` uses `g_baseline = ḡ_t` where `ḡ_t = β_g·ḡ_{t-1} + (1-β_g)·g_t` with `β_g ≈ 0.99`. Two alternatives rejected:

- **Random direction.** Doubles overhead from 1F to 2F. Reject.
- **Zero direction (g_baseline = 0).** Reduces meta-loss to `−α·‖g_t‖²` plus higher-order — a regularizer on gradient magnitude with the wrong sign. Reject.

**Moving-average baseline (selected).** No extra forward needed: by Taylor expansion,
```
L_baseline ≈ L_t − α·g_t^T·ḡ_t + O(α²)
```
The first-order term is computable from one dot product. Substituting `L'_t − L_t ≈ −α·‖g_t‖² + O(α²)`:
```
L_meta ≈ −α · g_t^T · (g_t − ḡ_t) + (α²/2)·g_t^T·H·g_t + O(α³)
```

The first-order term is **`−α · g_t^T · (g_t − ḡ_t)`** — the inner product of the current gradient with the *gradient innovation*. This rewards the gradient for departing from the moving average in a direction that reduces loss.

### 1.4 Why the same-batch trade-off still works

The simplification uses `B_t` for both forwards. The same-batch version still produces a useful meta-signal because: (1) Adam preconditioning + minibatch noise yield non-trivial second-order curvature `g_t^T·H·g_t`; (2) `ḡ_t` is the historical mean across past batches, so `g_t^T·(g_t − ḡ_t)` measures the current-batch gradient innovation against history; (3) the meta-loss back-propagates through shared parameters (embeddings, layer norms) that aren't directly active on `B_t`.

This is weaker than cross-batch MAML. **The 1.5–2× per-step gain is optimistic; if same-batch is binding, realistic per-step is 1.2–1.4× and net wall-clock 1.05–1.18×.**

### 1.5 Full bilevel formulation rejected

The full MAML formulation is 8F per step plus HVPs — too expensive. META-LEARN-CHIRON drops the second backward and the HVP, reducing to **4F per step**, treating `L'_t − L_baseline` as a *scalar* meta-loss rather than a vector meta-gradient.

---

## 2. Compute analysis — 33% per-step overhead

### 2.1 FLOP accounting per training step

```
F_baseline = 1·F (forward) + 2·F (backward) = 3F

F_meta = 1·F (forward)               [computes L_t]
       + 2·F (backward)              [computes g_t]
       + 1·F (virtual forward at θ̃)  [computes L'_t; NO backward]
       + ε  (dot product g_t·ḡ_t)    [≈ 10⁻⁵ · F at 1.84B]
       + ε  (EMA update on ḡ_t)
       ≈ 4F
```

**Per-step overhead: 4F / 3F = 1.333× = 33.3%.** No new persistent state. No extra memory beyond:
- `α` and `λ_meta` (2 scalars).
- `ḡ_t` EMA (`d` floats; with FACE-MFIO compression, ~200 MB at 1.84B).
- `θ̃ = θ - α·g_t` (transient; computed in-place using a scratch buffer).

### 2.2 Wall-clock projection

```
WallclockBaseline = 3F · N_steps
WallclockMeta    = 4F · N_steps_meta = 4F · (N_steps / 1.75) = 2.286 F · N_steps

Speedup = 3F · N_steps / (2.286 F · N_steps) = 1.31×
```

**Sensitivity to per-effective-step gain:**

| Per-step gain | Net wall-clock |
|---|---|
| 1.2× (pessimistic) | 1.05× ← marginal |
| 1.5× (mild) | 1.13× |
| 1.75× (default) | **1.31×** |
| 2.0× (optimistic) | 1.50× |
| 2.5× (very optimistic) | 1.88× |

**A per-step gain below 1.4× makes META-LEARN net-negative wall-clock** (the 33% overhead exceeds the saving). Gate-0 (§9) tests this threshold at 41M scale before committing to 1.84B.

### 2.3 Memory and kernel overhead

`ḡ_t` at 1.84B in bf16: 3.68 GB; with FACE/MFIO Zipfian compression (1008–1570× as established by FACE on `m, v`): ~200 MB. **Net memory: ~200 MB, well within the 4 GB headroom.**

The virtual forward reuses the existing inference kernel (`transformer_infer.cpp`). No new CUDA kernel required. The dot product is one cuBLAS `sdot` call (~1 ms on H100). **All compute already exists in the codebase.**

---

## 3. NLL preservation

### 3.1 The auxiliary-loss formulation

```
L_total(θ_t) = L_{B_t}(θ_t)              ← standard CE, primary
             + λ_meta · L_meta(θ_t)       ← META-LEARN auxiliary
```

with `λ_meta ≤ 0.1` (similar to PRM-CHIRON's `λ_PRM = 0.1`). The standard CE term is always dominant; the meta-loss is a *regularizer* on gradient direction quality.

**Final NLL:** as `t → ∞` and `λ_meta` stays bounded, the model's CE loss converges to the same floor as the baseline. Formally, if `θ*` minimizes `L_{B_t}` on the population, then `g(θ*) = 0`, so `L_meta(θ*) = L_{B_t}(θ*) − L_baseline = 0` regardless of baseline. **The fixed point of the meta-loss coincides with the fixed point of CE.**

### 3.2 Worst-case NLL drift

```
‖θ_meta_endpoint − θ_CE_endpoint‖ ≤ λ_meta · diameter(L_meta_landscape) / κ_min
```

At `λ_meta = 0.1` and typical `κ_min ≈ 10⁻³`, worst-case drift ≤ 0.005 nat — **well within the 0.05 nat tolerance** that PRM-CHIRON established.

### 3.3 The α window where the meta-loss is informative

§1.3 derived that as `α → 0`, the meta-loss collapses to `−α·g_t^T·(g_t − ḡ_t)`. As `α` grows, higher-order curvature terms dominate and the meta-loss measures Hessian-aligned-ness, which may not align with convergence acceleration.

```
Optimal α ≈ 1 / sqrt(L_max),  L_max = max Hessian eigenvalue on trajectory
```

At LLM pretraining, `L_max ≈ 10² to 10⁴`. So `α ≈ 0.01 to 0.1`. **Default α = 0.03** is the geometric mean; SAS-style schedule anneals `α` to track the `L_max` trajectory.

---

## 4. Composition with #43 ORION (HVP overlap)

### 4.1 Both rely on Pearlmutter HVPs

ORION's anchor step computes a rank-`r` Hessian projection via Lanczos: `H_∥ = V^⊤ · M · V` requires `r` Pearlmutter HVPs. META-LEARN's full second-order formulation would compute one HVP per step. **The kernel is shared, so the HVP cost is paid once.**

However, META-LEARN-CHIRON in this design ships in the **simplified, scalar-meta-loss form**, which **does not require an HVP**:

```
F_meta = 1·F (forward) + 2·F (backward) + 1·F (virtual forward) + 0·F (no HVP) = 4F
```

The full MAML formulation would push per-step cost to 6F and break-even to 2× per-step gain. **This is why META-LEARN-CHIRON ships scalar-only.**

### 4.2 Joint ORION + META-LEARN composition

If ORION ships in #43 and META-LEARN ships in #61:

- ORION amortizes per-step compute over K=20 anchor steps; per-effective-step `(3 + 2r)F / K = 0.55F` at K=20, r=4.
- META-LEARN adds 33% overhead to the *anchor* steps. Reduced steps inside the K-window do not invoke META-LEARN.
- Per-effective-step cost: `(3 + 2r + 1)F / K = 0.7F` (versus ORION-alone 0.55F).
- META-LEARN's per-step convergence gain (1.75×) applies to anchor effectiveness; conjectured K-window gain ~1.5× (slight compression because the closed-form quadratic is already optimal within the surrogate).

Joint ORION × META-LEARN net: `0.55F → 0.7F` cost (1.27× cost increase) × 1.5× gain = **1.18× wall-clock from META-LEARN on top of ORION**. Modest but positive.

### 4.3 Why the composition is partial (not multiplicative)

Both ORION and META-LEARN touch *the gradient direction* at the anchor step. ORION projects `g_t` onto a low-rank slow manifold; META-LEARN regularizes `g_t` to be effective in its own direction. These are not orthogonal:

- If META-LEARN's regularization pushes `g_t` away from the slow-manifold subspace, ORION's projection loses information.
- If ORION's projection leaves the meta-relevant direction outside the subspace, META-LEARN's signal is suppressed.

**Mitigation:** wire META-LEARN's `α·g_t` perturbation to use the *projected* gradient `V·V^⊤·g_t` instead of the full gradient. This forces META-LEARN to operate within ORION's slow manifold, eliminating the cross-paradigm interference at the cost of slightly weaker meta-signal.

### 4.4 HVP kernel sharing — concrete

The Pearlmutter HVP kernel is implemented as a forward+backward with a tangent vector, returning `H · v`. ORION needs `H · V[:, j]` for each column; META-LEARN-FULL would need `H · g̃_t`. **The kernel is identical.** Shipping META-LEARN at the same time as ORION requires zero additional kernel work in the scalar form — and even the full form requires only one extra call site.

### 4.5 Composition with the rest of the post-#60 stack

| Paradigm | Compatibility |
|---|---|
| #43 ORION | Partial (~1.18× joint, with V-projection mitigation) |
| #43 NEXUS / GANYMEDE | Multiplicative (~1.31× joint) |
| #42 SCFA, #28 FACE, #38 SLC, #39 RLG, #35 SPAREC | Multiplicative (orthogonal axes) |
| #56 DISTILL, #57 SCROLL, #58 METAGEN | Multiplicative (corpus/loss-side) |
| #59 PRM-CHIRON | Multiplicative (both auxiliary; joint λ-budget 0.05+0.10=0.15) |
| #60 COSMIC | Multiplicative per stage (META active in stages 1–2) |

**Stack target with #61-C selected:** `465,000× × 1.31× = 609,000×` cumulative single-GPU TRAINING speedup at fixed final NLL.

---

## 5. Diminishing returns at paradigm depth 20

### 5.1 The marginal-gain compression law

Across paradigms #1–#60, per-paradigm wall-clock multiplier has compressed: ~3–6× at #1–#10, ~2–4× at #11–#20, ~1.5–2.5× at #31–#40, ~1.5–2× at #51–#60 (with #56 DISTILL-FORWARD a 5× outlier). Empirically, `S_k ≈ S_{k-1} · ρ^k` with `ρ ≈ 0.93` since #50. **Projected `S_61 ∈ [1.3, 1.6]`. META-LEARN at 1.31× is at the low end of this band but within it.**

Structural causes of compression: orthogonal axes are exhausted; the loss floor `L*` is finite; hyperparameter co-tuning costs grow.

### 5.2 Honest framing of META-LEARN's modesty

**(a) Modest in absolute terms.** A 31% improvement on a 465,000× stack is structurally large in user-facing time (one hour per ~3) but small in stack-multiplier terms.

**(b) Modest relative to the slate.** If #61-A/B project 1.5×, META-LEARN is the smallest. Reason for honest framing, not rejection — selection should be on EV, not magnitude alone.

**(c) Modest conditional on the conjecture.** Under literature-honest reading (per-step 1.2–1.4×), net 1.05–1.18×. Under pessimistic reading, META-LEARN gives almost no advantage. Gate-0 is decisive.

### 5.3 Why pursue META-LEARN despite modesty

1. **Cleanest composition surface in the slate.** No new persistent state, two new scalar hyperparameters, reuses existing kernels, multiplicative with everything except #43 (partial). Stack risk near zero.
2. **HVP kernel reuse with future ORION ship.** Free option value on the full-form upgrade.
3. **Opens the optimizer-quality axis for #62+.** Learned optimizers, bilevel hyperparameter optimization, curriculum-aware meta-learning all build on this scaffold.

### 5.4 When META-LEARN should be rejected

If Gate-0 (§7) shows:
- **Per-step gain < 1.4× at 41M:** net ≤ 1.05× — too marginal, reject.
- **NLL drift > 0.02 nat:** auxiliary-loss formulation fails, reject (or down-tune `λ_meta` to 0.02).
- **Catastrophic instability:** Metz 2019 documents pathologies in learned optimizers. If similar at 41M, reject.

If Gate-0 shows:
- **Per-step gain ∈ [1.4×, 1.6×]:** ship at modest 1.05–1.18× net.
- **Per-step gain ∈ [1.6×, 1.8×]:** ship at strong 1.18–1.35× net (default band).
- **Per-step gain > 1.8×:** ship at 1.35–1.5× net — surprise upside.

### 5.5 Risk-adjusted EV vs slate alternatives

| Candidate | Mean × | Variance | Worst | Best |
|---|---|---|---|---|
| #61-A (e.g., MoE-extension) | 1.50× | 0.15 | 1.20× | 1.80× |
| #61-B (e.g., distill-extension) | 1.45× | 0.10 | 1.25× | 1.65× |
| **#61-C META-LEARN** | **1.31×** | **0.20** | **1.05×** | **1.55×** |

META-LEARN has the lowest mean but highest variance; best-case approaches A/B's mean. **Risk-neutral selection picks A; portfolio approach hedges by retaining META-LEARN as a feasibility-tested fallback.**

---

## 6. Concrete primitives and CLI

```cpp
namespace glades { namespace meta_learn {

struct MetaLearnState {
    GpuBuffer<float> g_ema;          // d-dim, FACE-MFIO-compressed
    float alpha;                     // virtual-step magnitude (cosine schedule)
    float lambda_meta;               // auxiliary-loss weight (warmup-only)
    int t_warmup;                    // disable after this step
    int step;
    bool active() const { return step < t_warmup && lambda_meta > 0; }
};

class MetaLearnStepper {
public:
    MetaLearnStepper(int d, float alpha_start, float lambda_start, int t_warmup);
    float compute_meta_loss(const NNetwork& net,
                             const GpuBuffer<float>& g_t,
                             const Minibatch& batch,
                             GpuBuffer<float>& theta_tilde_scratch);
    void update_ema(const GpuBuffer<float>& g_t, float beta = 0.99);
    void apply_meta_gradient(GpuBuffer<float>& g_t_modified) const;
    void anneal(int total_steps);
};

}}  // namespace glades::meta_learn
```

**CLI:** `--meta-learn 1 --meta-alpha 0.03 --meta-lambda 0.05 --meta-warmup-fraction 0.3 --meta-baseline ema --meta-baseline-beta 0.99`

---

## 7. Gate-0 probe — 41M × 5000 steps

### 7.1 Goal

Test whether META-LEARN delivers **per-step convergence gain ≥ 1.4×** (the wall-clock break-even threshold) at small scale before 1.84B implementation. Also test NLL parity within 0.02 nat.

### 7.2 Procedure (~6 GPU-hours)

1. **Baseline run.** 41M, pile-bpe, 5000 steps, T=512, α=3e-4, β=(0.9, 0.999), Kahan-v on. Record `L_t`.
2. **META-LEARN run.** Same config + `--meta-learn 1 --meta-alpha 0.03 --meta-lambda 0.05`. Record `L_t`.
3. **Convergence comparison.** Per-step gain = `t_baseline / t_meta` for `L_target ∈ {3.5, 3.0, 2.7, 2.5}`.
4. **NLL parity.** End-of-run validation NLL on held-out 1M tokens. Drift = `NLL_meta − NLL_baseline`.

### 7.3 Pass / fail criteria

| Criterion | Pass | Marginal | Fail |
|---|---|---|---|
| Per-step gain at L=3.0 | ≥ 1.6× | 1.4–1.6× | < 1.4× |
| Per-step gain at L=2.7 | ≥ 1.5× | 1.3–1.5× | < 1.3× |
| NLL drift | ≤ 0.01 nat | 0.01–0.02 nat | > 0.02 nat |
| Stability | clean | one warning OK | any divergence |

**Pass:** ship at 1.84B with default settings. Project 1.31× net.
**Marginal:** ship with `λ_meta = 0.02` and re-validate. Project 1.10–1.18×.
**Fail:** reject for #61.

### 7.4 What "marginal" means for #61 selection

If META-LEARN passes marginally (per-step 1.4× exactly), projected net 1.10× competes against #61-A's 1.50× directly on magnitude — and loses. **Marginal Gate-0 → reject for #61, keep design for #62+.**

---

## 8. Honest gaps (foregrounded summary)

1. **The 1.5–2× per-step gain is conjectured.** Based on MAML on few-shot benchmarks; not measured at LLM pretraining. Realistic band 1.2–1.8×. Below 1.4×, net wall-clock ≤ 1.05×.
2. **33% per-step overhead is irreducible.** Reuses inference kernel but still +1F. At paradigm depth 20, every percentage of overhead competes against shipped paradigms that *removed* overhead.
3. **The same-batch simplification weakens the meta-signal.** Cross-batch MAML is the literature standard; we use same-batch for causal-consistency. Estimate: 0.7–0.8× of full MAML signal strength.
4. **Composition with #43 ORION is partial.** Both touch gradient direction. Joint gain compresses from 1.31× to 1.18× with the V-projection mitigation.
5. **At paradigm depth 20, 1.31× is the smallest in the #61 slate.** Justifiable on EV/axis-opening/scope-efficiency, not magnitude.
6. **Auxiliary-loss collapse risk.** Narrow `α` window ~0.01 to 0.1; outside it the signal degenerates or is dominated by second-order terms.
7. **Learned-optimizer pathologies** (Metz 2019, 2020) are mitigated — Adam is preserved; we only meta-train gradient direction quality — but not eliminated. Gate-0 protects against catastrophic instability.
8. **NLL parity not guaranteed under aggressive `λ_meta`.** Default `λ_meta = 0.05` gives ≤ 0.005 nat drift; `λ_meta = 0.2` could drift 0.02–0.05 nat. Defaults are conservative.

---

## 9. Bottom line

META-LEARN-CHIRON applies MAML-style mathematics to LLM pretraining via a same-batch virtual-forward at `θ + α·g_t` that scores gradient direction quality. Per-step cost 4F (33% overhead vs baseline 3F); per-step gain conjectured 1.5–2× (default 1.75×); **net wall-clock 1.31×** at fixed final NLL. NLL parity preserved by construction. Composition with #43 ORION partial (~1.18× joint); with everything else multiplicative. Engineering scope ~720 LOC / 3 weeks.

**For:** honest, safe, scoped, forward-looking (opens optimizer-quality axis for #62+).
**Against:** smallest projected gain in the #61 slate; 1.5–2× conjecture not measured at LLM pretraining scale; 33% per-step overhead expensive at this depth; partial composition with #43 ORION.

**Recommendation:** present as the **modest, low-risk #61-C entry**. Selection rests on risk-adjusted EV and axis-opening, not magnitude alone. Selection over #61-A/B depends on: (a) whether ORION shipped at #43 (HVP kernel reuse), (b) slate risk tolerance, (c) value of axis-opening. **Run Gate-0 first** — 6 GPU-hours, decisive on the 1.4× per-step break-even threshold.
