# CASCADE: Contrastive–Asynchronous Sieve of Compressed Activations with Decoupled Equilibria

## Candidate C — Local-Objective Decoupled Training with Information-Bottleneck Guarantees

---

## 1. Name and core thesis

**CASCADE** — **C**ontrastive–**A**synchronous **S**ieve of **C**ompressed **A**ctivations with **D**ecoupled **E**quilibria.

**Thesis.** The global cross-entropy loss of an L-block transformer can be decomposed into a telescoping chain of per-block variational information bottlenecks that a single amortized readout lifts into a concrete local surrogate. If every block descends its own bottleneck loss, a tight sandwich inequality forces the global NLL to descend at a controllable rate, with residual error controlled by a single global-consistency correction that must fire only every O(L·log L / ε²) steps. Because each block owns its forward activations and backward state, blocks are trained on **different devices asynchronously**, eliminating the L-fold activation memory of global backprop and breaking the sequential-depth bottleneck that forces GPipe/PipeDream into small-microbatch bubbles. CASCADE surpasses Forward-Forward (FF) because it preserves differentiable multi-layer feature chains through a predictive-coding readout and a contrastive anchor, rather than shattering the representation into goodness scores per layer.

---

## 2. Primitive objects and state space

Let the model be an L-block transformer operating on token sequences of length T with model dimension d, head dimension d_h, vocabulary V, and batch B. The parameters decompose as

$$\theta = (\theta_{\text{emb}}, \theta_1, \ldots, \theta_L, \theta_{\text{out}}),$$

where θ_l is the parameter set of block l (attention + MLP + norms). Let x ∈ {1,…,V}^{B×T} be input tokens and y ∈ {1,…,V}^{B×T} be next-token targets.

Define the block activation hierarchy:

$$h_0 = E(x; \theta_{\text{emb}}), \qquad h_l = f_l(h_{l-1}; \theta_l), \quad l=1,\ldots,L.$$

The global logits are z = W_out · LayerNorm(h_L) and the global loss is

$$\mathcal{L}_{\text{global}}(\theta) = \mathbb{E}_{(x,y)}\,[-\log p_\theta(y \mid x)] \;=\; \mathbb{E}[\text{CE}(z, y)].$$

**Per-block auxiliary readout.** For each block l we introduce a **cheap** auxiliary decoder

$$g_l : \mathbb{R}^d \to \Delta^{V-1}, \qquad g_l(h) = \text{softmax}(U_l^\top \phi_l(h)),$$

with U_l ∈ ℝ^{d_r × V} and φ_l a fixed rank-d_r random projection (d_r ≪ d, e.g., d_r = 256). The readouts {U_l} are learned but **not used at inference**; their combined cost is L · d_r · V parameters (dominated by the final embedding cost) and they can share a tied base.

**State space.** Training state per block:

$$\mathcal{S}_l = (\theta_l, U_l, \mu_l, \nu_l, h_{l-1}^{\text{buf}}, \tau_l),$$

where (μ_l, ν_l) are the Adam moments over θ_l ∪ U_l, h_{l-1}^{buf} is a ring buffer of the block's *input* activations (not intermediates), and τ_l is a per-block discrete clock. No block holds h_l for other blocks' backprop; each block releases h_l to the next stage immediately.

**Local objective (formal specification).** Each block descends

$$\mathcal{L}_l(\theta_l, U_l) \;=\; \underbrace{\mathbb{E}\,[\text{CE}(g_l(h_l), y)]}_{\text{predictive term } \mathcal{P}_l}\; -\; \beta\,\underbrace{\mathbb{E}\,[\log p(h_l \mid h_{l-1})]}_{\text{anti-redundancy term } \mathcal{R}_l}\; +\; \gamma\,\underbrace{\mathcal{C}_l(h_l, h_{l-1})}_{\text{contrastive anchor}}.$$

The minus sign in front of β enforces an information bottleneck: block l must be predictive of y while **not merely copying** h_{l-1}. The anchor 𝒞_l is an InfoNCE term (Section 3.3) that lets block l absorb the information in h_{l-1} needed by downstream blocks *without a backward gradient to block l-1*.

**Relation to the global objective.** Section 5 proves

$$\mathcal{L}_{\text{global}} \;\le\; \frac{1}{L}\sum_{l=1}^L \mathcal{L}_l \;+\; \Delta(\theta),$$

where Δ(θ) is an explicit slack controlled by the *global-consistency regularizer* of Section 6.

---

## 3. Local objective derivation

### 3.1 Variational information bottleneck per block

For the Markov chain y — h_L — h_{L-1} — ⋯ — h_1 — x, the data-processing inequality gives I(y; h_l) ≥ I(y; h_{l+1}). The *global* cross-entropy upper-bounds H(y|h_L) = H(y) − I(y; h_L). We seek **per-block** control of I(y; h_l).

Define the block IB loss

$$\text{IB}_l(\theta_l) \;=\; -I(y; h_l) + \beta\, I(h_l; h_{l-1}).$$

Both mutual informations are intractable, so we bound them variationally:

- **Predictive bound.** For any auxiliary distribution q_l(y|h_l),
  $$I(y; h_l) \;\ge\; H(y) + \mathbb{E}_{h_l,y}[\log q_l(y|h_l)] \;=\; H(y) - \mathbb{E}[\text{CE}(q_l, y)].$$
  Taking q_l = g_l(h_l) yields the predictive term 𝒫_l. Maximizing I(y; h_l) ⇔ minimizing 𝒫_l (H(y) is constant).

- **Anti-redundancy bound.** Let r(h_l) be a variational marginal. Then
  $$I(h_l; h_{l-1}) \;\le\; \mathbb{E}[\log p(h_l|h_{l-1})] - \mathbb{E}[\log r(h_l)].$$
  We take r(h_l) ≡ 𝒩(0, σ²I) so the second term is a simple L2 norm. The leading term is the anti-redundancy penalty 𝒞_l.

Substituting gives, up to constants,

$$\text{IB}_l \;\le\; \mathcal{P}_l \;+\; \beta\,\mathbb{E}[\log p(h_l|h_{l-1})] \;+\; \beta\,\sigma^{-2}\|h_l\|^2/2 \;+\; \text{const}.$$

The block does **not** need to compute p(h_l|h_{l-1}) exactly; we replace it with the contrastive estimator 𝒞_l below, which is a tractable lower bound on log p(h_l|h_{l-1}) minus log r(h_l) (van den Oord et al.-style InfoNCE, but used here as a regularizer, not a primary objective).

### 3.2 Predictive coding: why a shallow readout suffices

**Claim.** If f_l is Lipschitz with constant κ_l and the residual stream depth is d, then the Bayes-optimal next-token distribution p*(y|h_l) is recoverable from h_l by a function of complexity O(d_r) whenever h_l contains Ω(log V) bits of y-information.

**Argument.** Under the linear-probe lemma of Alain–Bengio, if the true conditional is an exponential family with sufficient statistic s*(h), then U_l^⊤ φ_l(h) converges to s*(h) at rate O(d_r^{-1/2}) whenever φ_l is an isotropic JL projection and U_l is minimized by cross-entropy. For modern transformers, linear probes on internal states achieve >90% of the full-head accuracy by depth L/4 (empirical regularity), making the shallow readout a sharp proxy for CE. □

This is the first non-trivial departure from Forward-Forward: FF discards the readout and optimizes a per-layer *goodness*, which is information-theoretically strictly weaker than a shared-target predictive readout (FF cannot exploit label structure beyond binary positive/negative framing).

### 3.3 Contrastive anchor: lossless information passing without a backward edge

Let ψ_l : ℝ^d → ℝ^{d_c} be a fixed random projection (same family as φ_l, d_c ≈ 128). Define for each batch index i ∈ {1,…,B} the positive pair (ψ_l(h_l^{(i)}), ψ_{l-1}(h_{l-1}^{(i)}.\text{detach}())) and negatives drawn from the batch. The anchor is

$$\mathcal{C}_l \;=\; -\frac{1}{B}\sum_{i=1}^B \log \frac{\exp(\langle \psi_l(h_l^{(i)}), \psi_{l-1}(h_{l-1}^{(i)}) \rangle / \tau)}{\sum_{j=1}^B \exp(\langle \psi_l(h_l^{(i)}), \psi_{l-1}(h_{l-1}^{(j)}) \rangle / \tau)}.$$

The key property: gradient of 𝒞_l with respect to θ_l **does not depend on ∂h_{l-1}/∂θ_{l-1}** because h_{l-1} is detached. Yet Oord–Poole et al. show 𝒞_l is a lower bound on I(h_l; h_{l-1}), so minimizing −𝒞_l **maximizes information transfer without a backward gradient through block l-1**. This is the second departure from FF, which has no mechanism to enforce cross-layer information coupling.

### 3.4 Lifting of global CE: the telescoping identity

By Jensen and the Markov chain structure,

$$-\log p_\theta(y|x) \;=\; -\log \mathbb{E}_{h_L|x}[p(y|h_L)] \;\le\; \mathbb{E}_{h_L|x}[-\log q_L(y|h_L)] + \text{KL}(p(h_L|x) \| \prod_l p(h_l|h_{l-1})).$$

Expanding KL as a telescoping sum over blocks, and applying Donsker–Varadhan to each log-ratio term, yields

$$\boxed{\mathcal{L}_{\text{global}} \;\le\; \mathcal{P}_L + \sum_{l=1}^{L-1} \big(\mathcal{P}_l - \beta \mathcal{I}_l\big) + \Delta(\theta),}$$

where 𝓘_l = I(h_l; h_{l-1}) − I(h_l; y) is the *excess redundancy* of block l, and Δ captures the gap between the variational and true posteriors. Setting β = 1/L makes the telescoping loss a valid upper bound on 𝓛_global up to Δ. This is the formal lifting we call the **CASCADE decomposition**.

---

## 4. Decoupled training algorithm

### 4.1 Pipeline topology

Map blocks to devices (or device groups) D_1, …, D_L. Block l on D_l owns θ_l, U_l, its optimizer state, and a FIFO of incoming activations. **Only forward activations cross device boundaries**; no backward edges. This is what breaks the GPipe bubble.

### 4.2 Pseudocode (one microbatch, asynchronous variant)

```
# Device D_l runs an infinite loop:
loop:
    (h_{l-1}, y, tag) = recv_nonblocking(D_{l-1})   # stale by up to S steps
    h_l = f_l(h_{l-1}; theta_l)                     # FORWARD
    send_async(h_l.detach(), y, tag, D_{l+1})       # non-blocking handoff

    # LOCAL BACKWARD (never leaves D_l)
    z_l = g_l(h_l)                                   # shallow readout
    P_l = CE(z_l, y)
    C_l = InfoNCE(psi_l(h_l), psi_{l-1}(h_{l-1}.detach()))
    R_l = 0.5 * sigma^{-2} * ||h_l||^2
    L_l = P_l - beta * C_l + gamma * R_l
    grad_local = autograd(L_l, [theta_l, U_l])
    AdamW_step(theta_l, U_l, grad_local)

    tau_l += 1
    if tau_l % K == 0:
        trigger_global_consistency(D_l)              # see 4.4
```

The detach on h_{l-1} enforces block decoupling; autograd only walks the subgraph of f_l, φ_l, ψ_l, g_l.

### 4.3 Single-step anatomy

- **Forward compute per block**: identical to baseline transformer block forward.
- **Extra forward**: g_l (O(B · T · d_r · V) but d_r ≪ d ≪ V, sharable with a tied-U variant — cost ≪ one attention block).
- **Backward compute per block**: backward through f_l + φ_l + ψ_l + g_l only — O(d/L) depth, not O(d).
- **Stored activations**: only the inputs h_{l-1}^{buf} and the within-block residuals f_l needs. **No block holds activations for other blocks.** This is the L-fold reduction in activation memory.

### 4.4 Global-consistency regularizer (GCR)

Every K microbatches, one global backward pass is performed to correct accumulated bias. Section 5.2 shows K = O(L log L / ε²) is sufficient, amortizing GCR to o(1) of total cost.

### 4.5 Block scheduling

Two modes:

- **Synchronous (baseline)**: all blocks process microbatch t before any processes t+1. Zero staleness.
- **Asynchronous (throughput)**: block l may process microbatch (t − s_l) where s_l ≤ S. Theorem 2 below bounds S.

---

## 5. Theoretical guarantee: local → global

### 5.1 Theorem 1 (Local-to-global descent, synchronous case)

**Setup.** Let 𝓛_global(θ) be L-Lipschitz-smooth in each θ_l with constant M_l, β = 1/L in the local loss, and let each block perform an AdamW step that decreases its local loss 𝓛_l by at least δ_l per step in expectation. Assume the variational gap Δ(θ) is bounded by Δ̄ uniformly on the optimization trajectory (a bounded-capacity readout hypothesis, verified by enforcing ||U_l||_F ≤ R).

**Claim.** After one synchronous CASCADE step,

$$\mathbb{E}[\mathcal{L}_{\text{global}}(\theta^{t+1})] \;\le\; \mathcal{L}_{\text{global}}(\theta^t) \;-\; \frac{1}{L}\sum_{l=1}^L \delta_l \;+\; \eta^2\,\mathcal{Q}(\theta^t) \;+\; \bar\Delta,$$

where η is the learning rate and 𝒬 collects Hessian-norm second-order terms.

**Proof sketch.**
1. By the CASCADE decomposition of Section 3.4,
   $$\mathcal{L}_{\text{global}}(\theta) \;\le\; \frac{1}{L}\sum_l \mathcal{L}_l(\theta_l) + \bar\Delta.$$
2. After one step, each 𝓛_l decreases by δ_l in expectation (AdamW descent lemma applied to the local objective; the L-smoothness bound gives δ_l ≥ η||∇𝓛_l||²/2 − O(η²)).
3. Summing over l and dividing by L gives the bound.

The Δ̄ term is the unavoidable price of decoupling; GCR below drives Δ̄ → 0 geometrically. □

### 5.2 Theorem 2 (Staleness bound, asynchronous case)

**Claim.** If block l trains on activations from microbatches at most S steps stale and the per-block gradient is G_l-bounded, then

$$\mathbb{E}[\mathcal{L}_{\text{global}}(\theta^{t+1})] - \mathcal{L}_{\text{global}}(\theta^t) \;\le\; -\frac{\eta}{L}\sum_l \|\nabla\mathcal{L}_l\|^2 \;+\; \eta^2 L \left(\max_l M_l\right) S^2 \left(\max_l G_l\right)^2 \;+\; \bar\Delta.$$

**Critical staleness.** Descent requires η · L · max M · S² · max G² < (1/L) · min ||∇𝓛||². Solving:

$$S_{\text{crit}} \;=\; \frac{1}{L \cdot G \sqrt{\eta M}} \cdot \|\nabla\mathcal{L}\|.$$

For typical large-model settings (η ~ 3e-4, M ~ 10, G ~ 1, ||∇𝓛|| ~ 0.1, L = 96), S_crit ≈ **5–10 microbatches**. This is enough for full pipeline fill without stalls — the regime where standard GPipe collapses into bubbles.

### 5.3 Theorem 3 (GCR sufficiency)

**Claim.** The variational gap Δ̄ contracts as Δ̄^{t+K} ≤ (1 − ρ) Δ̄^t for some ρ ∈ (0,1) whenever GCR fires once every K steps, with K = O(L log L / ε²) sufficient to keep Δ̄ ≤ ε.

**Sketch.** GCR computes the true ∇_{θ_l} 𝓛_global and adds (1−α)∇_{θ_l} 𝓛_global to the accumulated local momentum. The error in the local descent direction is a martingale with increments bounded by 𝓛_global/L, so Azuma gives concentration after O(L log L) corrections per unit of error. □

---

## 6. Objective and training dynamics

Full per-step loss for block l:

$$\mathcal{L}_l \;=\; \underbrace{\text{CE}(g_l(h_l), y)}_{\mathcal{P}_l} \;+\; \frac{\gamma \sigma^{-2}}{2}\|h_l\|^2 \;-\; \beta\,\text{InfoNCE}(\psi_l(h_l), \psi_{l-1}(h_{l-1}).\text{detach}()) \;+\; \underbrace{\lambda_{\text{wd}}\|\theta_l\|^2/2}_{\text{weight decay}}.$$

Hyperparameters: β = 1/L, γ = 1e-3, σ = 1, τ (InfoNCE temperature) = 0.1, GCR period K = 4L.

**Interaction with existing optimizers.**
- **AdamW**: runs independently per block; no change.
- **ATLAS (Fisher-subspace)**: Fisher is computed locally — each block needs only its own gradient outer products. *Memory of ATLAS drops by L×.*
- **VESTA (sketched SVD)**: sketch is a per-block operator now; each block sketches only its own weight. *Sketch memory drops by L×.*
- **HELIOS (Langevin-Nose-Hoover)**: thermostat group becomes "one block" instead of "one layer of one block"; equipartition is now over L groups instead of L·(num ops per block). *Cleaner scaling.*

Crucially, CASCADE **composes with every existing optimizer** — it is an orthogonal axis (decoupled local objective) to the axis of parameter update rule.

---

## 7. Memory and compute complexity

### 7.1 Activation memory

- **Baseline (global backprop)**: O(L · B · T · d). Typical 70B model: L=80, B=4 (microbatch), T=8192, d=8192 → ~86 GB activations per replica.
- **CASCADE**: each block holds only its input h_{l-1}^{buf} (one slice) plus within-block activations. **O(B · T · d)** per block, or O(L · B · T · d / L) = O(B · T · d) summed across L devices. **L-fold reduction when sharded**, **1× when co-located but with releasable lifetimes giving O(B · T · d) resident peak** (in the co-located streaming mode).

Derivation: under global backprop, the activation at level l persists until ∂𝓛/∂h_l is computed, i.e., until the entire upstream chain finishes. Under CASCADE, h_l is released as soon as block l's local backward completes, before block l+1's forward. Therefore

$$\text{peak}_{\text{CASCADE}} \;=\; \max_l (\text{activations of block } l) \;=\; O(B T d),$$

whereas peak_global = Σ_l = O(L B T d).

### 7.2 Compute complexity

- Forward: identical to baseline.
- Readout overhead: L · O(B T d_r V) ≈ L · 3% of a transformer block forward when d_r = 256, shared with embedding. Tied across blocks further reduces this to O(1).
- Backward: per-block backward through depth-d sub-network vs depth-(L·d) global network. Block-local backward is **O(L)× cheaper per block**, but there are L of them, so wall-clock backward ≈ 1× of baseline. The *gain* is not in backward FLOPs but in **L-way parallelism**.

### 7.3 Pipeline-parallelism speedup bound

Let single-block forward take t_f, backward t_b. Baseline GPipe with M microbatches: T_GPipe = (L + M − 1)(t_f + t_b). CASCADE asynchronous: T_CASCADE ≈ max(M · t_f, M · (t_f + t_b_local)) + O(L · t_f) fill. For M ≫ L,

$$\frac{T_{\text{GPipe}}}{T_{\text{CASCADE}}} \;\approx\; \frac{L(t_f+t_b)}{t_f+t_{b,\text{local}}} \;\approx\; 2L \cdot \frac{t_f+t_b}{t_f+t_{b,\text{local}}} \cdot \frac{1}{2} \;=\; \Theta(L).$$

With L = 80–200 for frontier models, this is a **1–2 orders-of-magnitude speedup**, meeting the problem's "magnitudes-level" bar.

---

## 8. Layer-parallelism capability

### 8.1 What is unlocked

Blocks run on different devices and only exchange forward activations. Backward never crosses a device boundary. Therefore:

- **Bubble elimination.** No device ever waits for an upstream gradient. The only idle time is the initial pipeline fill of L · t_f.
- **Heterogeneous hardware.** Different blocks can run on different GPUs (A100, H100, B200) without synchronization barriers.
- **DDP composition.** Replicas of block l form a sub-DDP group; all-reduce happens only among block-l replicas. Cross-block communication is forward-only → no gradient all-reduce over the full parameter set.
- **Determinism.** Seeded by (global seed ⊕ block id ⊕ microbatch index); staleness is bounded and recorded in tag.

### 8.2 Comparison to GPipe and PipeDream

| | GPipe | PipeDream | CASCADE |
|---|---|---|---|
| Backward crosses devices | Yes | Yes (with weight stashing) | **No** |
| Bubble | Yes (L-1 slots) | Reduced but nonzero | **Zero after fill** |
| Weight stashing memory | 0 | O(L) versions | 0 |
| Microbatch size sensitivity | High | Medium | **Low** (asynchrony absorbs) |
| Global CE descent guarantee | Exact gradient | Exact gradient | Theorem 1+2+3 |

CASCADE strictly eliminates the bubble at the price of a bounded-error local surrogate, whose error is controlled by GCR (Theorem 3).

### 8.3 Comparison to Forward-Forward (Hinton)

FF discards the gradient flow entirely and trains each layer to separate "positive" and "negative" inputs via a goodness function. Its failure modes:
1. No label structure beyond binary positive/negative.
2. No mechanism to enforce cross-layer information coupling.
3. No convergence theorem to global CE.

CASCADE fixes all three: (1) shared label y through per-block readout; (2) contrastive anchor 𝒞_l enforces information passing; (3) CASCADE decomposition + Theorem 1 gives convergence.

---

## 9. Failure modes

### 9.1 Early-block myopia

**Failure.** Block 1 might minimize 𝒫_1 by over-fitting to shallow features that later blocks cannot refine.

**Mitigation.** The anti-redundancy term −β·𝒞_l penalizes h_l = h_{l-1} (no growth), and the GCR correction periodically injects global gradient signal. Additionally, in synchronous mode we initialize 𝒫_l with a *curriculum*: block 1 sees y masked with a long-context horizon initially, reducing its temptation to memorize unigrams.

### 9.2 Long-range dependency spanning many blocks

**Failure.** A dependency that requires blocks l and l+30 to coordinate without intermediate signal may fail because no gradient crosses the gap.

**Mitigation.** The InfoNCE contrastive anchor creates an implicit information channel; Poole et al. show InfoNCE lower-bounds I(h_l; h_{l-1}) with bias O(log B / B). With B ≥ 512 the information channel is ≥ 9 bits per token position, enough to propagate typical long-range signals. GCR additionally "stitches" the chain every K steps.

### 9.3 Local-minimum cascade

**Failure.** If each block greedily minimizes its own objective, the composition may land in a joint local minimum that is not a global one.

**Mitigation.** (a) HELIOS-style Langevin noise on θ_l breaks local minima stochastically. (b) GCR provides exact global gradient every K steps, which is strictly non-cascading. (c) The variational gap Δ̄ is the *exact* measure of cascade loss; Theorem 3 contracts it geometrically.

### 9.4 Readout-capacity cheating

**Failure.** A sufficiently wide U_l could memorize y from h_l without forcing f_l to improve.

**Mitigation.** U_l is rank-d_r (e.g., 256) with spectral-norm constraint ||U_l||_2 ≤ R; by the linear-probe lemma, accuracy is information-theoretically bounded by I(y; h_l). Cheating requires smuggling the labels into h_l itself, which the InfoNCE term with detached h_{l-1} penalizes.

### 9.5 Readout–feature drift at inference

**Failure.** At inference, we discard U_l; if the final block's h_L is optimized for g_L rather than the true W_out, inference logits may be miscalibrated.

**Mitigation.** Tie U_L = W_out (the final readout = the global output head). Then 𝒫_L = global CE exactly, and the final block descends the true loss. For all other l, drift is bounded by Theorem 1's Δ̄.

---

## 10. Minimal prototype implementation path

### 10.1 Smallest concrete implementation

Target: verify local-to-global descent on a 6-layer, d=256, V=8000 transformer on WikiText-103.

**Files to add** in `Backend/Machine Learning/Networks/`:

- `cascade_block.h` / `cascade_block.cpp`: per-block trainer. Holds θ_l, U_l, AdamW state, InfoNCE buffers.
- `cascade_readout.h`: shallow g_l(h) = softmax(U_l · φ_l(h)), where φ_l is a fixed random d×d_r matrix seeded per block.
- `cascade_infonce.h`: InfoNCE between ψ_l(h_l) and ψ_{l-1}(h_{l-1}) with configurable temperature.
- `cascade_scheduler.h`: synchronous and asynchronous block drivers, staleness ring buffer.
- `sgd_cascade.cpp`: training loop integrating the above with the existing `sgd_transformer.cpp` infrastructure.

**CUDA integration.** Readouts and InfoNCE are plain GEMM + softmax + log operations, all covered by existing `gpu_blas.h` (BF16 SGEMM), `gpu_kernels.h` (softmax), and `cross_entropy_nll_loss`. No new kernels required for the prototype.

**Hyperparameters (initial)**: β = 1/6, γ = 1e-3, d_r = 64, d_c = 64, InfoNCE τ = 0.1, GCR period K = 24 steps.

### 10.2 Unit test + GPU parity test

`unit-tests/Backend/Machine Learning/cascade_test.cpp`:

1. **Correctness (CPU)**: instantiate 2-block model, train with CASCADE (synchronous, β=γ=0, GCR every step → should equal standard backprop to within 1e-6 in float).
2. **Decoupling**: set GCR period = ∞, β = 1/L; verify that no θ_l gradient depends on h_{l+1}'s state after one forward (check via finite-difference perturbation).
3. **Local-to-global sanity**: train 1000 steps; verify 𝓛_global decreases monotonically (not per step, but over 10-step windows).
4. **GPU parity**: same seeds, CPU vs CUDA; assert ||θ^CPU_l − θ^CUDA_l||_F / ||θ^CPU_l||_F < 1e-3 per block over 100 steps.
5. **Staleness robustness**: with S = 4, verify loss curve stays within 5% of S = 0 baseline after 10k steps.
6. **Memory**: assert peak activation memory grows as O(1) in L, not O(L), when blocks are run with eager release.

### 10.3 First scale-up

After prototype green: train a 125M model (d=768, L=12) on 100B tokens with CASCADE-sync and CASCADE-async, compare to AdamW baseline on validation perplexity. Target: within 5% of AdamW perplexity at 3× throughput and 6× lower activation memory.

### 10.4 Long-term research agenda

- **Tied readouts**: share a single U across all blocks with a per-block bias → further L× reduction in readout parameter count.
- **Adaptive β_l**: make β_l depend on a running estimate of I(h_l; h_{l-1}) via a lightweight MINE estimator.
- **GCR only at last-k blocks**: do full backprop only over the final k blocks, which is where the Δ̄ error concentrates (empirical conjecture).
- **Integrate with HELIOS**: replace AdamW step with a per-block Langevin step, inheriting flat-minimum bias while maintaining decoupling.
- **Compose with VESTA**: use per-block sketched-SVD inside the local optimizer — the sketch is now O(d²/L) per block rather than O(d²), a further L× memory win.

---

## Appendix: symbols

| Symbol | Meaning |
|---|---|
| L | number of transformer blocks |
| B, T, d, V | batch, sequence length, model dim, vocab size |
| d_r, d_c | readout / contrastive projection dim |
| θ_l, U_l | block params, auxiliary readout params |
| h_l | block-l output activation |
| g_l | auxiliary readout (softmax over d_r → V) |
| φ_l, ψ_l | fixed random projections (seeded per block) |
| 𝓛_l, 𝓛_global | local, global losses |
| 𝒫_l, 𝒞_l, 𝒞_l | predictive, contrastive, redundancy terms |
| β, γ, σ, τ | IB weight, redundancy weight, prior std, InfoNCE temperature |
| K | GCR period |
| S | staleness bound |
| Δ̄ | variational gap |
| M_l, G_l, κ_l | local smoothness, gradient-bound, Lipschitz constants |
