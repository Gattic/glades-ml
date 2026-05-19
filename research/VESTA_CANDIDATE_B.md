# VESTA Candidate B — MuRe (μ-Recurrence): Multi-Particle Belief Recurrence with Causal Information-Bottleneck

**Date:** 2026-05-19
**Status:** Candidate design (parallel with Candidate A and Candidate C).
**Audit prerequisite:** `research/VESTA_AUDIT.md`.
**Phase-0a postmortem:** `research/EALRMN_PHASE0_RESULTS.md`, `research/EALRMN_PHASE0B_RESULTS.md`.

---

## 0. One-paragraph summary

We propose **MuRe** (μ-Recurrence): a sequence model whose hidden state at time *t* is an *empirical probability measure* μ_t over a learned latent space ℒ ⊂ ℝ^d_z, represented as a *k*-particle ensemble {(w_t^{(j)}, z_t^{(j)})}_{j=1..k}. The update rule is a discrete-time approximation of a Bayesian filter on a controlled Markov chain whose transition kernel and likelihood are both learned. The training objective is a **causal information bottleneck** with a non-collapsing latent-prediction term: each particle predicts a *cross-stream* future statistic (not a same-stream future state), which structurally breaks the Phase-0a bootstrap-circularity failure. The *k=1* reduction is a linear-RNN / LRU baseline; we pre-register that this reduction must replicate the 131× tanh→linear win (Claim N3 / B0). Particles are batched on GPU; per-particle Python loops are forbidden. The framework is motivated by the observation in `VESTA_AUDIT.md` that pure SSMs (Mamba, RWKV, S4) have a documented unimodal-state weakness on multi-query associative recall (MQAR; Based +32.2 acc over Mamba), and that the R2 (latent prediction) axis has no published clean win — the niche where novelty has the most room.

---

## 1. Motivation and position in the audit

### 1.1 The unimodal-state cliff

`VESTA_AUDIT.md` documents two quantitative weaknesses of the SSM / linear-attention family that share a common cause:

1. **Multi-query associative recall (Zoology, Arora et al. 2024).** Based, by adding a linear-attention term, gains +32.2 acc on MQAR and +10.36 acc on real recall slices of Pile vs Mamba. The published explanation is that compressing the past into a fixed-size vector loses the ability to keep *multiple* candidate matches alive.
2. **Exact copying / induction at long T (Jelassi et al. 2024, arXiv 2410.03810).** Mamba's degradation on copy/induction grows with T at a rate that a per-head fixed-rank state cannot avoid.

Both behaviours are explained, mathematically, by the recurrent state being a **point estimate** of an underlying latent rather than a **distribution**. A point estimate, however large, commits to a single hypothesis at each step. When the past supports multiple plausible continuations — multiple candidate keys, multiple grammar paths, multiple speakers — the point-estimate state must either average them (losing discriminability) or arbitrarily collapse to one (losing the rest).

A natural fix is to carry a *belief* — a probability measure — over the latent. Below we make this concrete.

### 1.2 Why a measure-valued state is not a trivial restatement of "make the state bigger"

A bigger flat state vector still represents one hypothesis with more channels per hypothesis. The empirical-measure state μ_t = (1/k) Σ_j w_t^{(j)} δ_{z_t^{(j)}} represents *k hypotheses simultaneously*, with weights that compete under the Bayesian update. Information theoretically:

- Flat state at width m: representable hypotheses ≈ Vol(state-manifold) — capacity scales with m.
- Empirical-measure state with k particles in ℝ^d_z: jointly representable hypotheses = k, each at full d_z resolution, *plus* a relative-weighting prior over those k.

This is structurally similar to how a Gaussian mixture model differs from a Gaussian: the GMM is genuinely multi-modal, the Gaussian is not — and no rescaling of a Gaussian recovers the GMM. Mamba's selective-scan, S5's MIMO diagonal, RWKV-6/7's matrix-valued state are all the "make the Gaussian wider/non-isotropic" branch. MuRe is the "make it a mixture" branch.

### 1.3 Why this attacks R2 specifically

The Phase-0a failure of EALRMN's latent-prediction objective (predict z_{t+h} from s_t under InfoNCE+latent-MSE) was *bootstrap circularity*: the encoder, the teacher EMA, and the recurrence all updated together with no external grounding, and so they collapsed to constants or to the random-init basin (`EALRMN_PHASE0B_RESULTS.md` §"Two distinct failure modes"). The reconstruction-bootstrap fix at α_recon=1 worked at probe_z but failed to propagate to probe_s.

The multi-particle state opens a *third* type of latent prediction: each particle, at each step, predicts a *future statistic conditioned on its own hypothesis being correct*. The aggregation at the model level then reduces to a posterior expectation — this is the standard SMC particle filter score. Crucially, this objective has a non-trivial Bayes-optimal solution (the true posterior mean of the future statistic) that is *not* a constant in z, and so does not collapse the way Phase-0a's same-stream-z prediction did. We make this precise in §6.

---

## 2. Primitive objects

We define every symbol used in the rest of the document.

**Sequence and observations.**
- *x* = (x_1, ..., x_T) ∈ 𝒱^T: token sequence, vocabulary size |𝒱| = V.
- *e_t* = E[x_t] ∈ ℝ^d_emb: embedded token, E ∈ ℝ^{V × d_emb}.

**Latent space and particles.**
- ℒ ⊂ ℝ^d_z: latent space, dimension d_z (typically d_z = m/4..m/2 where m is total model width).
- k ∈ {1, 4, 16, 64}: particle count. Pre-registered sweep points.
- Particle: z_t^{(j)} ∈ ℒ for j ∈ {1, ..., k}.
- Weight: w_t^{(j)} ∈ [0, 1] with Σ_j w_t^{(j)} = 1, for j ∈ {1, ..., k}.
- Empirical measure: μ_t = Σ_j w_t^{(j)} δ_{z_t^{(j)}}.

**Learned operators (per-layer, parameters θ).**
- Encoder ϕ: ℝ^{d_emb} → ℝ^{d_z}: maps input embedding to a latent observation. ϕ_t = ϕ(e_t) ∈ ℝ^{d_z}.
- Transition kernel τ_θ(z' | z, e): factored as deterministic drift A(e) z + B(e) plus stochastic perturbation ε ∼ 𝒩(0, Σ(e)). Equivalently a stochastic linear-RNN cell with input-conditional A, B, Σ.
- Likelihood ℓ_θ(e | z): scalar score for "z explains e", parameterized as exp(-‖ϕ(e) - C z‖²/2σ²) with learned C ∈ ℝ^{d_z × d_z} and σ.
- Readout head r_θ(μ_t) → ℝ^m: maps the empirical measure to a per-token model output. r is permutation-invariant in particle index j (DeepSets-style), so the head sees the *measure*, not a numbered list.
- Future-statistic head g_θ(μ_t) → ℝ^{d_z}: maps the measure to a prediction of a learned cross-stream future statistic ψ. Distinct from r.

**Cross-stream future statistic ψ (the R2 target — defined §6).**
- ψ: 𝒱^T × ℕ → ℝ^{d_z}: a *fixed* (non-trainable) function of a *paired* future stream that summarizes the latent structure we want s_t to predict.

**Notation conventions.**
- Subscripts t, t+h denote time. Superscripts (j) denote particle index.
- ⟨f, μ⟩ := Σ_j w^{(j)} f(z^{(j)}) is the expectation of f under μ.
- ‖·‖ without subscript is L2.
- All sums over j run 1..k unless stated.

---

## 3. State representation

The hidden state of the recurrence at time t is the empirical measure μ_t, stored on device as two tensors:

- `Z` ∈ ℝ^{B × k × d_z}: particle locations, B = batch size.
- `W` ∈ ℝ^{B × k}: particle weights, with the constraint Σ_j W[:, j] = 1 enforced after each update.

For batched GPU work, we additionally maintain:

- `logw` ∈ ℝ^{B × k}: log-weights, for numerically stable softmax-normalization.
- An optional `Sigma` ∈ ℝ^{B × k × d_z}: per-particle isotropic spread (Σ_j Gaussian-component variance). Set Sigma = 0 to recover the pure particle (Dirac-mixture) variant; set k = 1 to recover a Gaussian recurrence.

The state is a **non-parametric** Monte-Carlo representation when Sigma = 0 (a particle filter, e.g., Doucet, de Freitas, Gordon 2001); a **parametric Gaussian mixture** when Sigma > 0 (a Gaussian-sum filter, Anderson & Moore 1979 §10.3); and a **single Gaussian** when k = 1 — which reduces, as we show in §5, to a linear-RNN / LRU baseline.

Memory cost: O(B · k · d_z). At d_z=64, k=16, B=32, this is 32·16·64·4 = 128 KiB per layer per timestep on FP32 — negligible vs activations.

### 3.1 Capacity argument (multi-particle vs flat-state)

For an audit-grade comparison we need a concrete capacity statement, not a metaphor. Consider two state representations with equal *total* parameter budget P:

- (Flat-state) A single state vector s_t ∈ ℝ^P with transition s_t = A s_{t-1} + B e_t, A ∈ ℝ^{P × P}.
- (k-particle) k particles each in ℝ^{d_z}, with k · d_z = P/2 (half the budget; the other half is the shared A, B, C). At k=16, d_z=P/32.

Information-theoretically, both representations can store ~ P bits of state at infinite numerical precision. But the *structure* of representable posteriors differs:

- The flat state encodes a *single* point estimate (or, if we read s as a Gaussian mean, a single unimodal Gaussian with covariance fixed by A's spectrum).
- The k-particle state encodes an *empirical measure*: at d_z resolution, up to k distinct candidate hypotheses simultaneously.

Concretely, consider the multi-key recall sub-problem: at step t the model has seen k distinct (key, value) pairs and must retain *all* of them in state. A flat linear-RNN state can carry at most ~ P/d_key key-value pairs before mutual interference. The k-particle state can store one key-value pair per particle without interference (each particle's z encodes one (key, value)). At P=2048, d_key=64, k=16: flat state ~32 pairs; particle state ~16 pairs but with *zero mutual interference*. This is the regime where Based's sliding-window attention wins on MQAR; we claim particles get there by a different mechanism.

This is not a proof of an expressivity gap (a P-dim flat state can in principle encode any k-mode distribution under a sufficiently clever encoding). It is an argument about *what gradients can find*: particle ensembles make multi-modal posteriors a *first-class object*, so SGD does not have to discover the multi-modal encoding from scratch.

### 3.2 Why not Gaussian-component covariances (Sigma > 0 default)

We pre-register Sigma = 0 (Dirac particles) as the default. The reason: a Gaussian-component variant with per-particle Σ^{(j)} requires gradient flow through (a) the matrix Σ^{(j)} itself, (b) Σ^{(j)} in the likelihood denominator. Numerical conditioning is brittle when Σ^{(j)} → 0. The pure-particle variant with shared Σ in the predict step is more robust. Sigma > 0 is a Phase 3 ablation, not a Phase 0 default.

---

## 4. Update rule

### 4.1 One-step Bayesian filter

Given μ_{t-1} = Σ_j w_{t-1}^{(j)} δ_{z_{t-1}^{(j)}} and input x_t:

**Step 1 (predict / drift).** Each particle propagates under the input-conditional Markov kernel:

$$
\tilde z_t^{(j)} = A(e_t) \, z_{t-1}^{(j)} + B(e_t) + \xi_t^{(j)}, \quad \xi_t^{(j)} \sim \mathcal{N}(0, \Sigma(e_t))
$$

A is parameterized **diagonally** in a learned basis to match LRU / S5 (Orvieto et al. 2023; Smith, Warrington, Linderman 2023). Specifically, A(e_t) = U · diag(λ(e_t)) · U^* where U ∈ ℂ^{d_z × d_z} is shared across t (fixed orthogonal, complex-valued) and λ(e_t) ∈ ℂ^{d_z} is input-conditional with |λ| ≤ 1 enforced via the LRU stability parameterization λ = exp(-exp(ν) + i·exp(φ)).

**Step 2 (correct / update weights).** The likelihood of x_t under each propagated particle:

$$
\ell_t^{(j)} = \exp\left( -\frac{1}{2 \sigma^2} \| \phi(e_t) - C \tilde z_t^{(j)} \|^2 \right)
$$

Update weights:

$$
w_t^{(j)} \leftarrow \frac{w_{t-1}^{(j)} \cdot \ell_t^{(j)}}{\sum_{j'} w_{t-1}^{(j')} \cdot \ell_t^{(j')}}, \quad z_t^{(j)} = \tilde z_t^{(j)}
$$

Numerically computed in log-space:

$$
\log w_t^{(j)} = \log w_{t-1}^{(j)} - \tfrac{1}{2\sigma^2}\|\phi_t - C \tilde z_t^{(j)}\|^2 - \mathrm{logsumexp}_{j'}\left[\log w_{t-1}^{(j')} - \tfrac{1}{2\sigma^2}\|\phi_t - C \tilde z_t^{(j')}\|^2\right]
$$

**Step 3 (resampling / bounded-k maintenance).** A pure weighted-particle update suffers *particle degeneracy*: after O(d_z) steps almost all weight concentrates on one particle (Doucet, Godsill, Andrieu 2000; Cappé, Godsill, Moulines 2007). Without resampling, the effective sample size ESS = 1/Σ_j (w^{(j)})² → 1 and the multi-particle state degrades to a single-particle Gaussian.

We use **differentiable systematic resampling** (Corenflos et al., ICML 2021, arXiv 2102.07850), which provides an unbiased low-variance estimator with gradient flow through the resampling operation:

```
ESS_t = 1 / Σ_j (w_t^{(j)})²
if ESS_t < k/2:
    z_new = OptimalTransportResample(z_t, w_t)     # Corenflos et al.
    w_new = 1/k for all j
    z_t, w_t = z_new, w_new
```

Optimal-transport resampling (Corenflos et al.) replaces the standard categorical-multinomial resample with a *Sinkhorn transport plan* from the weighted empirical measure to a uniform empirical measure on a fresh particle set; the plan is differentiable in (z, w) for fixed transport-plan iterates. Cost: O(k² · n_sinkhorn) per resample, n_sinkhorn ≈ 5–10. Resampling is triggered ~ T/4 times per sequence in practice (ESS half-life is sub-linear in T under our likelihood scale σ).

For pure inference (no gradient required) we fall back to standard stratified resampling (Kitagawa 1996), which is O(k).

**Sinkhorn-iteration details.** The OT plan is computed by entropy-regularized Sinkhorn. Given particles z^{(j)} with weights w^{(j)} (the marginal before resampling) and a uniform target marginal u^{(j)} = 1/k:

1. Cost matrix C_{jj'} = ‖z^{(j)} − z^{(j')}‖² (the L2-squared transport cost).
2. Initialize log-potentials α^{(j)} = 0, β^{(j')} = 0.
3. For n_sinkhorn iterations: alternating updates α ← log w − logsumexp_{j'}(β − C/ε), β ← log u − logsumexp_{j}(α − C/ε), with regularization ε ≈ 0.05·median(C).
4. Transport plan: T_{jj'} = exp(α^{(j)} + β^{(j')} − C_{jj'}/ε).
5. Resampled particles: z_new^{(j')} = Σ_j T_{jj'} z^{(j)} · k (row-marginalized barycenter).
6. New weights: 1/k.

The Sinkhorn updates are differentiable in (z, w); the implicit-function-theorem trick from Corenflos et al. §3.2 gives the gradient back through the fixed-point (α*, β*) without unrolling the iteration loop, saving memory and compute.

**Step 4 (readout).** The per-token model output:

$$
o_t = r_\theta(\mu_t) = \mathrm{MLP}\Big(\textstyle\sum_j w_t^{(j)} z_t^{(j)},\ \sum_j w_t^{(j)} (z_t^{(j)} \otimes z_t^{(j)}),\ -\sum_j w_t^{(j)} \log w_t^{(j)}\Big)
$$

The three features fed to the readout MLP are: posterior mean, posterior covariance (flattened), and entropy of the weights. A DeepSets-style permutation-invariant readout that uses these three moments is sufficient to be a *universal approximator* on bounded-support measures (Zaheer et al. 2017; for measures specifically, Maron, Litany, Chechik, Fetaya 2020). We pre-register the moment-readout as the default; an alternative attention-readout (`r = softmax(Q·Kj/√d_z) · z_j`) is a §8 ablation.

### 4.2 Continuous-time interpretation

In the limit Δt → 0 the prediction step becomes a **drift-diffusion SDE** controlled by the input:

$$
dz = (A(e) z + B(e)) \, dt + \Sigma(e)^{1/2} \, dW
$$

and the empirical measure μ_t evolves under the corresponding **Fokker-Planck equation** with a likelihood-correction term (a Zakai equation for the unnormalized filtering density, Zakai 1969; equivalently a McKean-Vlasov SDE with the likelihood as the interaction term, Crisan, Lyons 1999). In the k → ∞ limit the empirical-measure update is consistent with the continuous-time nonlinear filter μ_t = p(z_t | e_1..t).

This places MuRe in the same mathematical class as **neural SDEs** (Li et al. 2020, arXiv 2001.01328) and **particle-filter RNNs** (Karkus, Hsu, Lee 2018, arXiv 1805.11122), but with two distinctions:

- LRU-style stability parameterization on A, so the recurrence is *linear in z* and inherits the linear-RNN / SSM gradient stability that EALRMN Phase-1 confirmed is the dominant axis.
- A causal information-bottleneck training objective (§6) rather than next-token CE alone.

### 4.3 Why the resampling step is the load-bearing engineering choice

Particle degeneracy is the dominant failure mode of naive particle-filter RNNs. Without resampling, k>1 is *exactly equivalent to k=1* after a few hundred tokens (Doucet, Godsill, Andrieu 2000): a single particle carries almost all the weight, the rest carry numerical noise. Standard categorical resampling breaks the gradient (the resampled index set is a discrete sample). The Corenflos et al. (2021) OT-resampling is the first differentiable resampling with both low variance and tractable gradients; without it, end-to-end training of a particle filter is brittle. **We pre-register that ablating OT-resampling to plain multinomial resampling (with straight-through gradients) will degrade multi-needle recall substantially; this is Claim N1's secondary check.**

---

## 5. Reduction to k=1: linear-RNN / LRU baseline

When k=1 the empirical measure μ_t = δ_{z_t} collapses to a single Dirac. The update becomes:

- Predict: z_t = A(e_t) z_{t-1} + B(e_t) + ξ_t.
- Correct: the weight update is trivial (single particle, weight = 1).
- Resample: never triggers (ESS = 1 always equals k = 1).
- Readout: o_t = MLP(z_t, z_t ⊗ z_t, 0) — the moment-readout reduces to a deterministic function of z_t.

Setting Σ = 0 (no diffusion) and A diagonal in a fixed complex basis with the LRU stability parameterization gives **exactly the LRU recurrence** (Orvieto et al. 2023). With Σ = 0 and A real-orthogonal we recover the linear-recurrence + orthogonal-init baseline that EALRMN Phase-1 found responsible for the entire EALRMN win (`research/EALRMN_PHASE1_GPU_RESULTS.md`).

**Pre-registered prediction (Claim N3 = B0).** k=1, Σ=0, A diagonal-LRU-orthogonal init, trained on the needle task at T=2048, m=1024 (d_z=512), 5 seeds, will reach the same 131× gap over a tanh-RNN matched-param baseline that EALRMN Phase-1 reported. **If this fails, the implementation is broken; debug before proceeding to k>1.**

The k=1 reduction also gives us a free k=1 vs Mamba-via-LRU comparison: LRU is the audit's strongest *minimal* linear-RNN baseline. We use it as the reference floor.

---

## 6. Information-bottleneck objective

### 6.1 Lagrangian

We train MuRe with a causal information-bottleneck objective:

$$
\mathcal{L}_{\text{MuRe}}(\theta) = \alpha \cdot \mathcal{L}_{\text{NTP}} + \beta \cdot \mathcal{L}_{\psi} + \gamma \cdot \mathcal{L}_{\text{ESS}} + \delta \cdot \mathcal{L}_{\text{surprise}} + \eta \cdot \mathcal{L}_{\text{bound}}
$$

with α, β, γ, δ, η all individually ablatable (set to 0 to remove the term and verify nothing else breaks).

#### Term L_NTP — standard next-token CE

$$
\mathcal{L}_{\text{NTP}} = -\sum_t \log p_\theta(x_{t+1} \mid x_{\leq t}) = -\sum_t \log \mathrm{softmax}(W_{\text{out}} o_t)_{x_{t+1}}
$$

This is the audit-mandated baseline objective. α is pinned to 1 in all primary runs; the comparison is what the *extra* terms add.

#### Term L_ψ — cross-stream future-statistic prediction (the R2 term)

This is the term whose design avoids the Phase-0a failure. The design choice is:

**Predict a future statistic of a paired stream, not of the same stream.**

In Phase-0a, the prediction target was z_{t+h} from the *same* stream as z_t (`EALRMN_DESIGN.md` §8). With encoder ϕ, teacher EMA ϕ̄, and recurrence all updating together, the system has a degenerate solution: ϕ ≡ const, ϕ̄ ≡ const, K · const = const. This is the constant-output collapse documented in `EALRMN_PHASE0B_RESULTS.md` §"Two distinct failure modes (ii)".

We avoid this by using a **paired-stream construction**:

1. Each minibatch contains *pairs* of streams (x, x') sampled from a known joint distribution. For our R2 tasks (§7.N2 below), x' is a perturbation of x that preserves the underlying latent (e.g., paraphrased copy on the copy task, or a synthetic HMM emission sampled from the same hidden state sequence).
2. ψ(x', t+h) is a *fixed* function of the paired stream — *not* trainable. Specifically, on the synthetic Markov-chain task (§7.N2), ψ(x', t+h) = one-hot encoding of the hidden state of x' at time t+h, *as a deterministic function of x'* once x' has been sampled.
3. The model predicts ψ(x', t+h) from its own state μ_t at time t (which has seen only x_{1..t}, not x' at all):

$$
\mathcal{L}_\psi = \sum_t \mathbb{E}_{x' \sim p(\cdot \mid x)} \, \| g_\theta(\mu_t) - \psi(x', t+h) \|^2
$$

Why this does not collapse:

- ψ is not a function of the model's parameters. It depends only on x'.
- The expectation E_{x' | x} ψ(x', t+h) is a *non-trivial* function of x_{1..t} (specifically, the posterior expectation of the hidden state at t+h given x_{1..t} — a quantity that *requires* the recurrence to actually filter the latent).
- Constant g_θ ≡ c does *not* minimize ‖g_θ(μ_t) - ψ‖² unless ψ is itself constant in expectation — which on a non-trivial latent it is not.
- The Bayes-optimal predictor for L_ψ is g*(μ_t) = E[ψ(x', t+h) | x_{1..t}], a non-constant function of the input history. So the loss has a unique non-degenerate minimum.

This is structurally identical to the **noise-contrastive density-ratio** approach in CPC (van den Oord et al. 2018, arXiv 1807.03748), but with a key difference: CPC contrasts the *same* sequence's future against negatives, leading to the bootstrap-circularity Phase-0a observed; we contrast against an *independently generated paired stream*, breaking the circularity by injecting a non-learnable signal.

**Important caveat.** L_ψ requires a paired-stream dataset. On natural-language pretraining corpora, paired streams are not free. Three options:

1. (Pre-registered for VESTA Phase-0) Use *synthetic* tasks where the latent is observable and pairs can be generated cheaply (§7.N2). This is the cleanest test.
2. Use *augmentation pairs* (e.g., paraphrased sentences, or noisy versions of the same passage). Standard self-supervised LM augmentation; the latent must be invariant under the augmentation.
3. Use *cross-document pairing under same topic* (e.g., two news articles on the same event). Topic is the implicit latent.

For Claim N2 we commit to option 1.

#### Term L_ESS — effective-sample-size regularization

Particle filters degenerate when ESS_t = 1/Σ_j (w_t^{(j)})² collapses to 1. We add a hinge:

$$
\mathcal{L}_{\text{ESS}} = \sum_t \max(0, \, \tau_{\text{ESS}} - \mathrm{ESS}_t / k)
$$

with τ_ESS = 0.25 by default (target ESS ≥ k/4). This term pushes the particle distribution to stay diverse; without it the system can learn to collapse weights as a shortcut. In SMC terms, this is a soft adaptive-resampling threshold.

#### Term L_surprise — surface-entropy regularization (β·I(s; surface entropy) in the bottleneck)

The information-bottleneck Lagrangian is

$$
\max_\theta \, I(\mu_t; F_{t+h}) - \beta \, I(\mu_t; X_t)
$$

where F_{t+h} is the future-statistic random variable (the target of L_ψ) and X_t is the surface input. The β term penalizes mutual information between the state and the *surface* observation — equivalently, penalizes the state for remembering input details that don't matter for predicting F.

Operationally we approximate this with a variational upper bound (Alemi et al. 2017, arXiv 1612.00410; Achille, Soatto 2018):

$$
I(\mu_t; X_t) \leq \mathbb{E}_{x_t} \mathrm{KL}(p_\theta(\mu_t \mid x_t) \| p_\theta(\mu_t))
$$

The marginal p_θ(μ_t) is approximated by a moving-average over the batch (a Gaussian fit to the per-batch particle cloud). Per-timestep cost: O(k · d_z). This term realizes the "discard surface entropy" part of the bottleneck and gives MuRe its name's "causal information-bottleneck" identity.

#### Term L_bound — particle-bound regularization

A hinge that penalizes particles drifting outside a ball of radius R:

$$
\mathcal{L}_{\text{bound}} = \sum_t \sum_j w_t^{(j)} \max(0, \|z_t^{(j)}\|^2 - R^2)
$$

Standard latent-space regularization, prevents unbounded growth under unstable LRU dynamics (the LRU |λ| ≤ 1 parameterization makes this redundant in theory but useful in practice during early training).

### 6.2 Why this objective is honest about the EALRMN failure

EALRMN's design memo claimed "latent prediction beats raw token prediction" (E5 in `newmodel.txt`). Phase-0a failed and Phase-0b onward silently returned to token prediction. The MuRe Lagrangian is honest about this:

- α (token prediction) is pinned to 1. We do *not* claim L_ψ alone suffices.
- β (the R2 latent-prediction term, cross-stream) is the *additional* term whose contribution we will measure.
- The pre-registered Claim N2 (§7) is: at iso-encoder-params and iso-α, adding L_ψ improves the model on a task where the latent is identifiable, by ≥ 0.10 nat. If it doesn't, we report that L_ψ contributes nothing — we do not silently drop the term.

This is the discipline E5 in `newmodel.txt` calls for.

### 6.3 Variational SMC derivation for the L_ψ + L_NTP combination

The L_NTP and L_ψ terms can be derived as variational lower bounds on two distinct mutual-information quantities. This places MuRe in the *variational SMC* family (Maddison et al. 2017; Naesseth et al. 2018).

**Bound on I(X_t; X_{t+1} | X_{<t}).** Standard CE: the next-token prediction is exactly the KL-anchored variational bound on H(X_{t+1} | X_{≤t}). No further analysis needed.

**Bound on I(F_{t+h}; μ_t | X_{≤t}).** Where F = ψ(X', t+h) is the cross-stream future statistic. By the data-processing inequality, I(F; μ_t | X_{≤t}) ≤ I(F; X_{≤t}); the closer μ_t is to a sufficient statistic of X_{≤t} for F, the tighter the bound. The MSE bound

$$
\mathbb{E}\|g_\theta(\mu_t) - F\|^2 \geq \mathbb{E}\|\mathbb{E}[F | X_{\leq t}] - F\|^2 = \mathrm{Var}(F | X_{\leq t})
$$

with equality at g* = E[F | X_{≤t}]. So minimizing L_ψ pushes g_θ ∘ μ_t to be a sufficient statistic of X_{≤t} for F. This is the *predictive sufficiency* notion (Bahadur 1954; Lehmann 1959) and is exactly the goal of the IB framework (Tishby, Pereira, Bialek 1999).

**Combined Lagrangian as a generalized IB.** With L_NTP capturing surface predictive sufficiency and L_ψ capturing latent predictive sufficiency, and L_surprise penalizing I(μ_t; X_t) (surface-entropy memorization), the full objective is:

$$
\max_\theta \, I(\mu_t; X_{t+1}) + \beta \, I(\mu_t; F_{t+h}) - \delta \, I(\mu_t; X_t)
$$

— the IB Lagrangian with a *causal* twist (the constraint set is causal: μ_t is a function of X_{≤t} only) and a *cross-stream R2 target* F_{t+h} added.

### 6.4 Comparison: what ABLATING each term does

The pre-registered ablation matrix for Phase 3 (only if Phase 1 + 2 pass) is:

| Setting | α | β | γ | δ | η | Expected behavior | Tests |
|---|---|---|---|---|---|---|---|
| MuRe-full | 1 | 0.5 | 0.1 | 0.05 | 0.01 | Best on N1+N2 by hypothesis | both |
| MuRe-NTP-only | 1 | 0 | 0.1 | 0 | 0.01 | Equals Mamba+OT-resample analog | N1 isolation |
| MuRe-NTP+ESS-only | 1 | 0 | 0.1 | 0 | 0.01 | Tests whether L_ESS alone helps N1 | N1 isolation |
| MuRe-ψ-only | 0 | 0.5 | 0.1 | 0 | 0.01 | Should *fail* — readout MLP has no gradient signal toward NTP. Sanity check that L_ψ is *not* a complete training objective. | sanity |
| MuRe-no-IB | 1 | 0.5 | 0.1 | 0 | 0.01 | δ=0 — tests whether the surface-entropy penalty matters | N2 isolation |
| MuRe-no-ESS | 1 | 0.5 | 0 | 0.05 | 0.01 | γ=0 — tests whether ESS hinge is required for stability | F1 measurement |
| MuRe-no-bound | 1 | 0.5 | 0.1 | 0.05 | 0 | η=0 — tests whether bound regularizer is required | F3 isolation |
| MuRe-no-resample | 1 | 0.5 | 0.1 | 0.05 | 0.01 | Disable OT resampling — particles will degenerate | F1 isolation |
| MuRe-categorical-resample | 1 | 0.5 | 0.1 | 0.05 | 0.01 | Categorical-resample-with-straight-through instead of OT | Resampling-ablation |

Each row is a single flag in the implementation. Compile-once, run-each.

---

## 7. Pre-committed falsifiable claims

Three claims, each with a 5-row pre-committed interpretation table and a confound-invalidation list.

### 7.N1 — Multi-query associative recall (the R4 attack)

**Claim N1 (preregistered).** On the MQAR synthetic task (Arora et al. 2024, arXiv 2402.18668, the Zoology suite as packaged in the Based repo), MuRe with k ≥ 16 particles, iso-param-count with **Based** (the strongest published linear-attention recall baseline), reaches MQAR accuracy ≥ 0.85 at the configuration (T = 4096, n_keys = 64, distractor_ratio = 4). Based's published numbers at the closest configuration (T = 2048, n_keys = 64) are 0.86; the Mamba baseline at the same is 0.46.

**Why this is the right adversarial test.** The audit identifies Based as the strongest linear-attention-class MQAR baseline. Mamba is a strawman here (already documented to fail by 32 acc points). The k=1 MuRe reduction is an LRU and will also fail MQAR. The interesting question is whether k=16 MuRe **closes the gap to Based**, which has a fundamentally different mechanism (sliding-window attention + Taylor linear attention) for the same purpose (multi-key recall).

**Threshold and interpretation.**

| Observation | Interpretation |
|---|---|
| MuRe@k=16 MQAR acc ≥ 0.85 AND ≥ Based acc − 0.03 | MuRe's multi-particle state is competitive with explicit sliding-window attention for MQAR; novelty earned on the recall axis. |
| MuRe@k=16 acc in [0.65, 0.85) | Multi-particle helps over k=1 LRU (~0.30 expected) but underperforms Based; partial. Look at k=64 and at OT-resample ablation. |
| MuRe@k=16 acc < 0.65, MuRe@k=64 acc < 0.65 | Multi-particle state does not solve MQAR; the unimodal-state-cliff hypothesis as the *root* cause is partially refuted. |
| MuRe@k=16 acc ≥ 0.85 but MuRe@k=1 acc also ≥ 0.85 | LRU at this scale already solves MQAR; the particle mechanism contributes nothing. Falsify N1, fall back to LRU. |
| MuRe @ Based-iso-FLOPs is ≥ 4× slower wall-clock | A wall-clock-loss is fine but must be explicit. If wall-clock-equivalent MuRe loses, novelty is on the expressivity axis only. |

**Confounds.** (1) Param count: MQAR is sensitive to capacity; we will match Based on total params. (2) Particle count vs head count: Based with H heads vs MuRe with k particles is not iso-FLOP; we will report both iso-param and iso-FLOPs separately. (3) Test set leakage: use Zoology's held-out test set, no LR sweep on test.

### 7.N2 — Cross-stream latent prediction (the R2 attack)

**Claim N2 (preregistered).** On a synthetic **paired-HMM task** with hidden-state-cardinality S = 16, vocabulary V = 64, emission-overlap 0.4, T = 1024, paired streams (x, x') sharing the *same* hidden-state sequence (z_t = z'_t for all t), MuRe with the L_ψ term active (β = 0.5, ψ = one-hot hidden state of x' at t+h, h = 8) achieves:

- L_NTP-only-val improves over the MuRe α=1, β=0 baseline by ≥ 0.10 nat.
- Linear probe of μ_t for hidden state z_t improves by ≥ 0.10 absolute acc.

**Why this is the right adversarial test.** EALRMN Phase-0b tried this with a same-stream construction and found probe_z = 0.84 vs probe_s = 0.76 — meaning the encoder learned z but the recurrence didn't propagate the latent. Our cross-stream construction breaks the constant-output equilibrium *structurally*, not via reconstruction hacks. The strongest "baseline" here is **MuRe-itself-with-β=0**, which is honest — no published paper has won this comparison cleanly at iso-encoder-params (`VESTA_AUDIT.md` §R2).

The task is designed to be *fair* in three specific ways relative to Phase-0a:

1. **The latent is genuinely identifiable from the input.** Phase-0a's T=64 16-patch task at S=4 had a near-uniform Bayes-optimal probe ceiling. We move to S=16, T=1024 so the posterior over z_t is non-trivial and the future statistic carries real information.
2. **The future-statistic horizon h=8 is long enough that next-token CE alone has a weak gradient on z.** At h=1, predicting x_{t+1} is dominated by emission noise; predicting z_{t+1} via cross-stream ψ is a meaningfully different signal.
3. **The paired stream is generated from the same z but independent emissions.** This means ψ(x') depends on z_{t+h}, not on x_{t+h} — exactly the latent-prediction signal Phase-0a was trying to extract.

**Threshold and interpretation.**

| Observation | Interpretation |
|---|---|
| L_NTP-val improvement ≥ 0.10 nat AND probe acc improvement ≥ 0.10 | L_ψ supports a measurable latent-prediction contribution beyond NTP. Claim N2 supported. |
| L_NTP-val improvement in [0.03, 0.10) nat | Small but real; report as marginal. May reflect a weaker version of the same effect; investigate β and h sweep. |
| L_NTP-val improvement < 0.03 nat AND probe acc improvement ≥ 0.10 | L_ψ improves the *latent* but not the *language model*. This is interesting but does not support N2 as stated. |
| L_NTP-val improvement < 0.03 nat AND probe acc improvement < 0.03 | L_ψ contributes nothing. Falsify N2. The cross-stream construction does not rescue R2 either. Honest negative result. |
| L_NTP improves but seed-variance > 0.10 nat | Optimization fragility (cf. Mamba's narrow optimal-LR window in Zoology). Report variance, do not over-claim. |

**Confounds.** (1) The paired-stream baseline gets *additional supervision* the unpaired baseline does not; we must match total gradient signal. We control by reporting at iso-gradient-steps, not iso-data-samples. (2) ψ might leak into the test set; we use disjoint pair-generation seeds for train and test. (3) The reconstruction-bootstrap that fixed EALRMN Phase-0b might be doing the work in MuRe too; we ablate reconstruction explicitly (γ = 0 vs γ > 0 in a side experiment).

### 7.N3 — Baseline replication (B0, validates infrastructure)

**Claim N3 (preregistered).** MuRe at k=1, Σ=0, A diagonal-LRU, orthogonal init, on the needle-in-haystack task at T=2048, m=1024, 5 seeds, beats a tanh-RNN baseline at the same param count by ≥ 100× val_loss ratio. This replicates EALRMN's Phase-1 result.

**Threshold and interpretation.**

| Observation | Interpretation |
|---|---|
| MuRe@k=1 / tanh-RNN val_loss ratio ≥ 100×, all seeds | B0 replicated, infrastructure validated, proceed to N1/N2. |
| Ratio in [10×, 100×) | Some signal but weaker than Phase-1. Debug optimization before proceeding. |
| Ratio < 10× | Infrastructure broken or tanh-RNN unrepresentative. Debug. |
| MuRe@k=1 diverges to NaN on ≥ 1 seed | Stability bug. Orthogonal init not actually orthogonal, or LRU λ parameterization wrong. Debug. |
| MuRe@k=1 beats tanh by ≥ 100× AND k=4 beats k=1 by ≥ 0.05 nat | Bonus: the resampling machinery is *not* hurting at k>1 even on a single-needle task. Pre-flight check for N1. |

**Confounds.** (1) The tanh-RNN baseline must use the *same* B, ϕ, C, readout — only A is different. (2) Reading out the readout MLP from a k=1 state vs from a k=16 state has different param counts; we match the readout MLP param count by widening k=1 to compensate.

---

## 8. Implementation sketch

### 8.1 File layout (target ~1500 LOC C++ + ~600 LOC CUDA)

```
research/mure_phase0/
  mure_phase0.cpp                        # CPU prototype, single file, ~800 LOC
                                         # Tests: k=1 reduction (Claim N3 floor),
                                         # k=4 / k=16 on toy MQAR-style task.
  mure_phase1/
    main.cpp                             # GPU driver
    kernels/
      mure_predict.cu                    # k-particle drift step (batched)
      mure_likelihood.cu                 # log-likelihood per particle
      mure_logsumexp_norm.cu             # weight normalization (numerically stable)
      mure_ess.cu                        # ESS reduction per batch element
      mure_ot_resample.cu                # OT resampling (Corenflos et al.)
                                         #   Sinkhorn matrix-scaling iterations
      mure_readout.cu                    # DeepSets moment readout
      mure_loss_ntp.cu                   # next-token CE (existing kernel)
      mure_loss_psi.cu                   # paired-stream MSE on g_θ(μ) - ψ
      mure_loss_ib.cu                    # variational IB upper bound
    Makefile
  mure_phase2/                           # if N1 + N2 pass; integration into glades-ml
    Backend/Machine Learning/Networks/
      mure_network.cpp                   # full integration
      mure_network.h
      sgd_mure.cpp                       # SGD training loop integration
```

### 8.2 Kernel inventory (the load-bearing part)

**Per-step kernels (called T times per sequence):**

1. `mure_predict_kernel<<<B, k * d_z>>>`: A(e) z + B(e) + ξ. Each block handles one batch element, threads parallelize over (particle, latent dim). Fully batched, no per-particle loop. Cost: O(B · k · d_z) elementary ops per step.

2. `mure_likelihood_kernel<<<B, k>>>`: ‖ϕ(e) - C z‖². One thread per particle per batch element. Cost: O(B · k · d_z).

3. `mure_logsumexp_norm_kernel<<<B, 1>>>`: per-batch-element softmax over k. Cost: O(B · k).

4. `mure_ess_kernel<<<B, 1>>>`: 1/Σ_j w². Cost: O(B · k).

5. `mure_ot_resample_kernel`: triggered when ESS < k/2. Sinkhorn for n_sinkhorn iters. Cost: O(B · k² · n_sinkhorn). At k=16, n_sinkhorn=10: O(B · 2560). Only triggered ~ T/4 times per sequence. Amortized cost: O(B · k² · n_sinkhorn / 4) per step.

6. `mure_readout_kernel<<<B, m>>>`: posterior mean + posterior covariance + entropy → MLP. Cost: O(B · k · d_z + B · d_z²) for the moments, O(B · m · (d_z² + m)) for the MLP.

**Per-sequence kernels (called once at end):**

7. `mure_loss_ntp_kernel`: existing kernel from glades-ml transformer stack.
8. `mure_loss_psi_kernel`: ‖g_θ(μ) - ψ‖². Trivial.
9. `mure_loss_ib_kernel`: variational IB upper bound. KL between per-batch particle marginal and a fitted Gaussian. Cost: O(B · k · d_z + d_z²) for the Gaussian fit.

**Backward kernels:** every forward kernel has a matching backward. OT-resample backward uses the implicit-function-theorem trick from Corenflos et al. §3.2 to avoid differentiating through the Sinkhorn iterates explicitly. Approximation: hold Sinkhorn permutation matrix fixed at backward time. Empirically validated in their Section 5.

### 8.3 C++ cost estimate

| Module | LOC | Difficulty | New ops vs glades-ml |
|---|---|---|---|
| CPU prototype | ~800 | Easy | None — pure C++98 |
| Per-particle GPU kernels (predict, likelihood, normalize, ESS) | ~250 | Easy | All elementwise / reductive, batchable |
| OT-resample kernel + backward | ~200 | Hard | Sinkhorn iterations; needs careful numerics in fp32 |
| Readout (moment-based) | ~100 | Easy | DeepSets, standard |
| Loss kernels (ψ + IB) | ~150 | Easy | MSE + KL, both standard |
| Integration into glades-ml Networks/ | ~400 | Moderate | New `TYPE_MURE` enum, integration with existing optimizer / trainer |
| Tests (unit + integration) | ~300 | Easy | Use existing `unit-tests/` framework |
| **Total** | **~2200** | | |

This is comparable to the EALRMN Phase-1 GPU implementation (~3000 LOC at `research/ealrmn_gpu/`). One major risk: the OT-resample backward is the only genuinely novel kernel and the highest-defect-rate component. We pre-allocate ~30% of implementation time to that one kernel.

### 8.4 Compatibility with existing glades-ml infrastructure

- Builds on the existing `glades::rng::*` deterministic-RNG facility (CLAUDE.md). Particle stochasticity uses `glades::rng::normal(seed)`.
- Adds `TYPE_MURE = 7` to `Backend/Machine Learning/Networks/network.h`.
- Adds `sgd_mure.cpp` analogous to `sgd_rnn.cpp` / `sgd_lstm.cpp`.
- Reuses the existing transformer-stack `softmax_f32`, `dot_f32`, `axpy_f32` kernels from `transformer_kernels.h` for the readout MLP and the input encoder.
- Test suite uses the `ASSERT` macro from `unit-tests/unit-test.h`. New test name candidate: `mure` (or `partssm`).

---

## 9. Failure modes (the three most likely)

### 9.1 F1 — Particle degeneracy at long T (most likely failure)

**Symptom.** ESS_t drops to 1 within a few hundred tokens despite the L_ESS hinge and OT resampling. Multi-particle MuRe behaves indistinguishably from k=1. Claim N1 fails by mechanism, not threshold.

**Diagnostic.** Log ESS_t per layer per batch element across training. If mean ESS / k < 0.3 at step 1000, F1 is realized.

**Mitigation.** (a) Increase τ_ESS hinge weight γ. (b) Reduce likelihood-sharpness 1/σ² (more permissive likelihood keeps weights more uniform). (c) Switch from systematic to *adaptive* resampling on every step (cheaper since it's stratified, not OT, on most steps). (d) Add a particle-diversity term that explicitly penalizes Σ_{j≠j'} similarity(z^{(j)}, z^{(j')}).

**Falsification cost.** If F1 cannot be mitigated by 3-day search across (γ, σ, resample-rule), conclude that particle filters in the autoregressive-LM regime are fundamentally unstable, and the framework's expressivity advantage cannot be realized. This is a substantively different result than EALRMN's null — it would point to a *known* SMC pathology, not a novel one, but its discovery in this regime would still be informative.

### 9.2 F2 — L_ψ does not transfer to natural language (likely failure)

**Symptom.** Claim N2 passes on the synthetic paired-HMM task but on any natural-language task (even paraphrased pairs), L_ψ improvement collapses to ≤ 0.01 nat.

**Diagnostic.** Train a small MuRe on a paraphrase-paired Wikipedia/C4 slice and compare to MuRe with β = 0.

**Interpretation if realized.** The cross-stream construction works only when the latent is *exactly* identifiable from a clean simulator. Natural-language paraphrases share semantics but not a Markov-chain hidden state in any clean sense; the L_ψ target ψ(x', t+h) becomes too noisy to predict reliably. This would mean R2-as-stated is a synthetic-only claim, not a natural-language claim. This is the same kind of regime-specificity warning F3 in `newmodel.txt` calls for. We will report it as such.

**Honest framing if F2 realizes.** "MuRe's cross-stream latent-prediction term provides a measurable improvement on simulator-clean latent tasks (Claim N2 supported), but does not generalize to natural-language paraphrase pairs at the budget tested. This bounds the regime of applicability of the R2 axis to controlled-simulator settings; the natural-language version remains open."

### 9.3a F3 — k=1 reduction is itself broken, masking Claim N3 failure (most consequential failure)

**Symptom.** N3 reports ≥ 100× tanh-RNN gap as expected, but it is being driven by a bug in tanh-RNN baseline rather than by MuRe@k=1's correctness. Or, MuRe@k=1 ≠ LRU because of a subtle bug in the diagonal-A parameterization.

**Diagnostic.** Compare MuRe@k=1 weight trajectory to a known-good LRU implementation (port from NicolasZucchet/minimal-LRU, ~200 LOC). Compare weights bit-by-bit at step 0 (init), step 1, step 100.

**Why this matters.** EALRMN's prior program shipped 0.5 days of compute before discovering that its baseline was wrong. A reproducible LRU port is the *first* deliverable in Phase-0; we will not run N1 or N2 until N3 has bit-identical MuRe@k=1-vs-port behavior to 1e-5 absolute error per step on the needle task.

**Mitigation.** Port `minimal-LRU` (JAX) to C++ as a sanity-check baseline. ~200 LOC, ~2 hours. Bit-compare at every step on a fixed-seed run.

**Falsification cost.** Discovering this bug after N1 and N2 have been run would invalidate both. The cost of preventing it is ~2 hours; the cost of recovering is ~2 days. We pre-register the LRU bit-compare as a Phase-0 *blocker*.

---

### 9.4 Additional failure-mode considerations (briefly)

**F4 — Wall-clock cost dominates expressivity wins.** Even if k=16 MuRe is more *accurate* than Based on MQAR at iso-param, if it is 4-8× slower per token, it loses on any compute-aware benchmark. We commit to reporting wall-clock alongside accuracy. The OT-resample step is the single most expensive operation and the most likely culprit.

**F5 — Optimization fragility (Zoology pattern).** Mamba and Hyena have very narrow optimal-LR windows (Eyuboglu et al. 2023). MuRe inherits Mamba's input-dependent A and adds particle dynamics + OT resampling. The optimization surface may be more fragile, not less. We commit to LR sweeps at every (k, T) cell.

**F6 — Cross-stream pairing leaks at training time.** If the paired-stream sampler has any train-test correlation (e.g., same seed, same HMM realization), L_ψ trivially overfits. We commit to disjoint pair-seeds for train/val/test and to a held-out-HMM evaluation set.

---

## 10. Relationship to existing methods

The audit warns: a mechanism that is novel only in presentation, not in effect, is not novel. We compare MuRe to the closest published work.

**vs. Mamba / S6 (Gu, Dao 2023).** Mamba's selective scan makes A, B, C input-dependent on a single-particle state. MuRe shares the input-dependence but extends the state from single to k particles. At k=1, MuRe with full input-dependence on (A, B, Σ) is approximately Mamba with an additional learned noise term. The novelty axis is k>1 — Mamba does not have this.

**vs. RWKV-7 "Goose" (Peng et al. 2025).** RWKV-7 introduces an *expressive dynamic state evolution* with a matrix-valued state. Matrix-valued state ≠ particle-mixture state: RWKV-7 carries a richer single hypothesis; MuRe carries multiple lower-dim hypotheses. The mathematical character is distinct (rank-1 vs rank-k mixture).

**vs. Based (Arora et al. 2024).** Based combines linear attention (Taylor approximation of softmax) with a small sliding-window attention. Multi-modal recall comes from the sliding-window component, which holds an explicit KV cache of size W tokens. MuRe's multi-modal recall comes from the particle ensemble, which holds an implicit k-component posterior. Cost: Based pays O(W·d) per step; MuRe pays O(k·d_z) per step. At W ≈ 128 and k = 16 with d_z = d/4, MuRe is cheaper per step by ~ d/d_z = 4× on the recall path — *if* the particle ensemble actually captures the recall variability.

**vs. Particle-Filter RNN / PF-RNN (Karkus, Hsu, Lee 2018).** PF-RNN uses particle filtering for state estimation in nonlinear RNNs but: (i) uses categorical resampling (non-differentiable; uses straight-through estimator); (ii) uses tanh-RNN dynamics (which EALRMN Phase-1 proved is the dominant fragility); (iii) has no information-bottleneck objective. MuRe uses differentiable OT resampling (Corenflos et al. 2021), LRU-stable linear dynamics, and the IB Lagrangian. The mathematical structure is the same family; the engineering and objective are distinct.

**vs. Variational SMC (Naesseth et al. 2018, Maddison et al. 2017).** VSMC trains generative models via SMC bounds on the marginal likelihood. MuRe inherits the variational-SMC framework but applies it to *autoregressive sequence modeling* (not generative latent-variable modeling), and adds a non-NTP supervised term L_ψ that doesn't exist in pure VSMC. The novelty is in the regime, not in the variational-SMC primitive.

**vs. Mixture-of-Gaussians SSM (Bayesian Filtering).** Classical Gaussian-sum filters (Anderson, Moore 1979; Sorenson, Alspach 1971) have used Gaussian mixtures for nonlinear state estimation since the 1970s. MuRe is mathematically a Gaussian-sum filter with k components, learned A/B/Σ/C, and a deep-learning training loop. The mathematical structure is over 50 years old; what is new is (a) the integration with LRU stability, (b) the differentiable OT resampling at scale on GPU, (c) the IB + cross-stream-ψ objective.

**Honest assessment.** The framework is novel as an *integration*. The component primitives — LRU dynamics, OT resampling, IB objectives, Gaussian-sum filtering — are all from existing literature. The claim of novelty is therefore in the *combination + the autoregressive-LM application + the empirically-measured wins on MQAR / R2 tasks*. If N1 and N2 both fail, the framework reduces to "a Gaussian-sum filter wrapped around LRU" — which has academic interest but is not a contribution.

---

## 11. Pre-registered experiment plan (phased, with kill criteria)

We commit to running phases in the following order. Each phase has a pre-registered kill criterion; if the criterion triggers, we publish the negative result and do not run subsequent phases of MuRe (we may continue with other VESTA candidates).

### Phase 0 — Infrastructure (CPU prototype, 2 days)

1. Implement the LRU bit-compare (port NicolasZucchet/minimal-LRU). 2 hours.
2. Implement MuRe@k=1 in C++. 1 day.
3. Run Claim N3 (B0 replication). 5 seeds, T=2048, m=1024.
4. **Kill criterion.** If MuRe@k=1 vs tanh-RNN ratio < 100× val_loss on the needle task, debug. If unable to fix within 1 day, abandon — implementation is broken.
5. Implement OT-resample (Corenflos et al.) on CPU at k ∈ {4, 16}. 1 day.

### Phase 1 — MQAR (GPU, 3 days)

1. Port Phase-0 to GPU. 1 day.
2. Run Claim N1 (MQAR) at k ∈ {1, 4, 16, 64}. 1 day for all four cells with 3 seeds.
3. Compare to **Based** (audit-mandated baseline) at iso-param and iso-FLOPs.
4. **Kill criterion.** If MuRe@k=16 MQAR < 0.65 AND MuRe@k=64 < 0.65, the particle mechanism does not solve MQAR; the unimodal-state-cliff hypothesis is partially refuted. Publish negative.
5. **Pass criterion.** If MuRe@k=16 MQAR ≥ 0.85 AND ≥ Based − 0.03, proceed to Phase 2.

### Phase 2 — R2 cross-stream (CPU + GPU, 3 days)

1. Implement paired-HMM task and ψ generator. 0.5 day.
2. Implement L_ψ + L_IB. 1 day.
3. Run Claim N2: paired-HMM at S=16, T=1024, h=8. β sweep ∈ {0, 0.1, 0.5, 1.0}. 5 seeds per cell.
4. **Kill criterion.** If L_NTP improvement < 0.03 nat AND probe acc improvement < 0.03 across all β > 0, L_ψ contributes nothing. Publish negative on R2 (matches `VESTA_AUDIT.md` §R2: "no clean baseline exists" — our negative would be one more data point against R2-as-stated).
5. **Pass criterion.** Threshold N2-row-1.

### Phase 3 — Ablations and stress tests (3 days)

If both N1 and N2 pass:

1. OT-resample vs categorical-resample-with-straight-through. Pre-registered: OT should win by ≥ 0.05 nat on MQAR.
2. k=1 vs k=4 vs k=16 vs k=64 scaling.
3. β ablation: L_ψ alone (β=1, α=0). Pre-registered: should fail catastrophically (no NTP gradient on the readout MLP). This is a *sanity check*, not a competitive run.
4. L_IB ablation: β=0.5 with η=0 vs η=1.0. Tests whether the IB term contributes.

### Total: ~11 days CPU + GPU compute, ~2200 LOC.

### Smoke-test commands (Phase 0)

```bash
# Build CPU prototype
cd /home/robert/dev/glades-ml
g++ -std=c++98 -O3 -march=native -Wall -Wextra \
    research/mure_phase0/mure_phase0.cpp \
    -o research/mure_phase0/mure_phase0

# Claim N3 floor: MuRe@k=1 vs tanh-RNN at needle T=2048 m=1024 (5 seeds, ~3 min each)
for s in 1 2 3 4 5; do
    ./research/mure_phase0/mure_phase0 --mode mure --k 1 --T 2048 --m 1024 --seed $s --task needle
    ./research/mure_phase0/mure_phase0 --mode tanh --k 1 --T 2048 --m 1024 --seed $s --task needle
done

# LRU bit-compare: MuRe@k=1 against ported minimal-LRU on a fixed-seed run
./research/mure_phase0/mure_phase0 --mode mure --k 1 --bit-compare-lru --seed 42
# Expect: max-absolute-diff < 1e-5 at every step for the first 100 steps.

# Multi-particle warm-up: MuRe@k=4 on a small MQAR-style task
./research/mure_phase0/mure_phase0 --mode mure --k 4 --T 512 --task mqar --n-keys 8 --seed 42
```

### Decision-table template (to be filled per phase)

| Phase | Claim | Status | Regime where supported | Notes |
|---|---|---|---|---|
| 0 | N3 (B0) | TBD | T=2048, m=1024 | If <100×, debug; if ≥100×, proceed |
| 0 | LRU bit-compare | TBD | full trajectory | Hard kill if max-diff > 1e-5 |
| 1 | N1 (MQAR) | TBD | T=4096, k≥16 | Compare vs Based at iso-param + iso-FLOPs |
| 2 | N2 (R2) | TBD | S=16 paired-HMM | β sweep ∈ {0, 0.1, 0.5, 1.0} |
| 3 | OT-resample ablation | TBD | conditional on N1 pass | OT vs categorical |
| 3 | β ablation | TBD | conditional on N2 pass | β ∈ {0.1, 0.5, 1.0} |
| 3 | k scaling | TBD | conditional on N1 pass | k ∈ {1, 4, 16, 64} |

---

## 12. What this framework is *not* claiming

In the spirit of `newmodel.txt`'s discipline E5 (do not silently drop the central claim):

- **We are not claiming MuRe beats Mamba-2 at scale.** We are claiming it closes the MQAR gap at small scale (Claim N1), at iso-param with Based. Whether the multi-particle mechanism survives to 7B parameters is unknown and deliberately not pre-registered.
- **We are not claiming L_ψ improves natural-language pretraining.** We claim it improves a synthetic paired-HMM task (Claim N2) with an identifiable latent. F2 enumerates the failure mode where this does not generalize.
- **We are not claiming the particle ensemble is the unique answer to multi-modal recall.** Based (sliding-window) and DeltaNet (delta-rule on non-diagonal state) are alternative mechanisms in the audit. MuRe is one of several candidates; the comparison is empirical.
- **We are not claiming the C++ cost is small.** ~2200 LOC + 1 novel CUDA kernel (OT-resample) is a real engineering commitment. We pre-commit it to deliver Phase 0 and Phase 1; Phase 2 onward only if Phase 1 passes.

---

## 13. Open questions deliberately not answered here

1. **Is the IB Lagrangian's β learned or fixed?** We pre-register fixed β ∈ {0, 0.1, 0.5, 1.0} sweeps. Annealing schedules are deferred.
2. **What is the right d_z?** We pre-register d_z = m/4 as default; m/2 and m/8 are §3 ablations.
3. **Does the readout need to be DeepSets-moments or attention-over-particles?** We pre-register moments; attention-over-particles is a §8 ablation.
4. **How does MuRe interact with layer stacking?** All claims are pre-registered at L=1. L=4, L=8 are scaling experiments deferred to Phase 3+.

---

## 14. Adversarial-baseline comparison summary

For audit-traceability, the full strongest-baseline table for MuRe's three claims:

| Claim | Strongest published baseline | Audit-section justification | Our beat threshold | Comparison protocol |
|---|---|---|---|---|
| N1 (MQAR / R4) | **Based** (Arora et al. 2024) | `VESTA_AUDIT.md` §R4 — explicitly named as the strongest linear-attention recall baseline (+32.2 acc over Mamba on MQAR). | MuRe@k=16 MQAR ≥ 0.85 AND ≥ Based − 0.03 acc | Zoology held-out test set; iso-param AND iso-FLOPs both reported |
| N2 (R2 latent prediction) | **None published** (`VESTA_AUDIT.md` §R2: "no clean baseline exists"). Internal control is **MuRe with β=0** (NTP only). | Audit explicitly flags absence of baseline; comparison is against the iso-architecture iso-loss-budget MuRe-NTP. | NTP-val improvement ≥ 0.10 nat AND probe acc improvement ≥ 0.10 | Same model, same encoder params, same gradient steps, different β |
| N3 (B0 / R3 floor) | **Vanilla linear-RNN with orthogonal init** (EALRMN Phase-1; LRU as published reference) | `VESTA_AUDIT.md` §R3 — LRU is named "strongest minimal baseline; any novel recurrence must beat LRU before claiming anything beyond it" | MuRe@k=1 / tanh-RNN val_loss ratio ≥ 100× | Needle task T=2048 m=1024, 5 seeds |

**What we explicitly do not claim against:**

- **Mamba-2 (SSD).** Audit's strongest pure-recurrence baseline. We do not pre-register a beat against Mamba-2 at scale because the Phase-0 compute budget cannot run Mamba-2 at credible param counts. We commit to evaluating MuRe-at-scale against Mamba-2 *only if* Phase 0-2 pass and additional compute is granted.
- **Mixtral / MoE.** Different axis (R5, sparse routing). MuRe makes no R5 claim.
- **MambaByte.** Different axis (R1, tokenization). MuRe makes no R1 claim.
- **MoD / MoR.** Different axis (R7, compute-adaptive). MuRe makes no R7 claim.

---

## 15. Pre-committed honesty markers

Following `newmodel.txt` E5 (do not silently drop the central claim):

1. **If Claim N1 fails (MQAR < 0.65 at k=16 and k=64)**: We will report MuRe's multi-particle mechanism as falsified on the recall axis, and update the audit table accordingly. We will *not* repackage the framework as "still useful for X" without a separately pre-registered claim on X.

2. **If Claim N2 fails (no L_NTP improvement from β > 0)**: We will report R2-via-cross-stream as falsified at the synthetic-task tier — joining the EALRMN Phase-0a/0b prior in saying that the R2 axis cannot be unlocked by the obvious moves. We will *not* silently drop L_ψ from subsequent experiments; if L_ψ contributes nothing, we will set β=0 going forward and explicitly mark the change.

3. **If Claim N3 fails (B0 not replicated)**: We will treat this as an infrastructure bug, not a falsification of the framework. We will publish the bug, fix it, and re-run N3 before proceeding.

4. **If F1 (particle degeneracy) is realized despite mitigations**: We will publish the diagnostic ESS trajectories. We will not claim "multi-particle helps" if all measured cases show ESS → 1 within the first 1/3 of T.

5. **If the wall-clock cost is prohibitive (F4 realized)**: We will report wall-clock cost transparently and acknowledge MuRe as an expressivity-axis contribution rather than a throughput contribution. The audit's existing wall-clock leaderboard (Mamba-2 6× FA2 at T=16k) is not what we are challenging.

6. **If natural-language transfer fails (F2 realized)**: We will explicitly bound MuRe's R2 contribution to "synthetic-task tier" and *not* generalize it to LM pretraining. The framework's value in that case is methodological (cross-stream pairing as a way to defeat bootstrap-circularity), not architectural.

These six honesty markers are pre-committed before any code is written.

---

## 16. Sources

Inline references:

- Anderson, Moore (1979). *Optimal Filtering.* Gaussian-sum filters §10.3.
- Sorenson, Alspach (1971). Recursive Bayesian estimation using Gaussian sums. *Automatica.* Original Gaussian-sum filter.
- Doucet, Godsill, Andrieu (2000). On sequential Monte Carlo sampling methods for Bayesian filtering. *Statistics and Computing.*
- Doucet, de Freitas, Gordon (eds., 2001). *Sequential Monte Carlo Methods in Practice.* Springer.
- Zakai (1969). On the optimal filtering of diffusion processes. *Z. Wahrscheinlichkeitstheorie.* The Zakai equation.
- Crisan, Lyons (1999). A particle approximation of the solution of the Kushner-Stratonovich equation. *Prob. Theory Rel. Fields.*
- Kitagawa (1996). Monte Carlo filter and smoother for non-Gaussian nonlinear state-space models. *J. Comp. Graph. Stat.* Stratified resampling.
- Cappé, Godsill, Moulines (2007). An overview of existing methods and recent advances in sequential Monte Carlo. *Proc. IEEE.*
- Corenflos, Thornton, Deligiannidis, Doucet (2021). Differentiable particle filtering via entropy-regularized optimal transport. *ICML 2021,* arXiv 2102.07850. The differentiable resampling.
- Karkus, Hsu, Lee (2018). Particle filter networks: end-to-end probabilistic localization from visual observations. arXiv 1805.11122. PF-RNN.
- Maddison, Lawson, Tucker, Heess, Norouzi, Mnih, Doucet, Whye Teh (2017). Filtering variational objectives. *NeurIPS.* VSMC.
- Naesseth, Linderman, Ranganath, Blei (2018). Variational sequential Monte Carlo. *AISTATS.*
- Li, Wong, Chen, Duvenaud (2020). Scalable gradients for stochastic differential equations. arXiv 2001.01328.
- Alemi, Fischer, Dillon, Murphy (2017). Deep variational information bottleneck. *ICLR 2017,* arXiv 1612.00410.
- Achille, Soatto (2018). Emergence of invariance and disentanglement in deep representations. *JMLR.*
- van den Oord, Li, Vinyals (2018). Representation learning with contrastive predictive coding. arXiv 1807.03748.
- Zaheer, Kottur, Ravanbakhsh, Poczos, Salakhutdinov, Smola (2017). Deep Sets. *NeurIPS 2017.*
- Maron, Litany, Chechik, Fetaya (2020). On learning sets of symmetric elements. *ICML 2020.* Permutation-invariant readouts for measures.
- Orvieto, Smith, Gu, Fernando, Gulcehre, Pascanu, De (2023). Resurrecting recurrent neural networks for long sequences. *ICML 2023,* arXiv 2303.06349. LRU.
- Smith, Warrington, Linderman (2023). Simplified state space layers for sequence modeling. *ICLR 2023,* arXiv 2208.04933. S5.
- Gu, Dao (2023). Mamba: linear-time sequence modeling with selective state spaces. arXiv 2312.00752.
- Arora, Eyuboglu, Zhang, Timalsina, Alberti, Zinsley, Zou, Ré (2024). Simple linear attention language models balance the recall-throughput tradeoff. *ICML 2024,* arXiv 2402.18668. Based, MQAR.
- Yang, Schlag, Hofmann, Liu, Ge, Stanić, Jaeger, Schmidhuber (2024). Parallelizing linear transformers with the delta rule over sequence length. arXiv 2406.06484. DeltaNet.
- Jelassi, Brandfonbrener, Kakade, Malach (2024). Repeat after me: Transformers are better than state space models at copying. arXiv 2410.03810.
- Eyuboglu, Arora, Zhang, Ré (2023). Zoology: measuring and improving recall in efficient language models. Stanford HazyResearch blog post.
- Peng et al. (2025). RWKV-7 "Goose" with expressive dynamic state evolution. OpenReview.
- Peng, Alcaide, Anthony, et al. (2023). RWKV: Reinventing RNNs for the Transformer era. *EMNLP,* arXiv 2305.13048.

---

## Appendix A — Pseudo-code for one MuRe forward step (single layer)

```
def mure_forward_step(x_t, z_prev, w_prev, theta):
    # x_t: (B,) token ids
    # z_prev: (B, k, d_z), w_prev: (B, k)

    e_t = E[x_t]                                  # (B, d_emb)
    phi_t = phi_theta(e_t)                        # (B, d_z) -- linear projection

    # Step 1: predict (drift)
    lam = lambda_theta(e_t)                       # (B, d_z) complex, |lam| <= 1
    bias = B_theta(e_t)                           # (B, d_z)
    sigma2 = sigma2_theta(e_t)                    # (B, d_z) positive
    xi = randn(B, k, d_z) * sqrt(sigma2)[:, None] # (B, k, d_z)
    z_tilde = einsum('bd,bkd->bkd', lam, z_prev) + bias[:, None] + xi

    # Step 2: correct (likelihood weights)
    C_z = einsum('de,bke->bkd', C_theta, z_tilde) # (B, k, d_z)
    diff = phi_t[:, None] - C_z                   # (B, k, d_z)
    logL = -0.5 / sigma_lik**2 * sum(diff**2, -1) # (B, k)
    log_w_new = log(w_prev) + logL
    log_w_new = log_w_new - logsumexp(log_w_new, -1, keepdims=True)
    w_new = exp(log_w_new)                        # (B, k)

    # Step 3: maybe resample
    ess = 1.0 / sum(w_new**2, -1)                 # (B,)
    needs_resample = ess < (k / 2)                # (B,)
    if any(needs_resample):
        z_resampled, w_resampled = ot_resample(
            z_tilde[needs_resample],
            w_new[needs_resample],
            n_sinkhorn=10)
        z_tilde[needs_resample] = z_resampled
        w_new[needs_resample] = w_resampled

    z_new = z_tilde

    # Step 4: readout
    mean_z = einsum('bk,bkd->bd', w_new, z_new)
    var_z = einsum('bk,bkd,bke->bde', w_new, z_new, z_new) - mean_z[:, :, None] * mean_z[:, None]
    H_w = -sum(w_new * log_w_new, -1)             # (B,)
    o_t = MLP_theta(concat([mean_z, var_z.reshape(B, -1), H_w[:, None]], -1))

    return o_t, z_new, w_new


def mure_loss(x, x_paired, theta):
    # x: (B, T) primary stream
    # x_paired: (B, T) cross-stream pair (same hidden state, different emissions)

    z, w = init_state(B, k, d_z)
    losses = []
    for t in range(T):
        o_t, z, w = mure_forward_step(x[:, t], z, w, theta)
        losses.append(o_t)

    L_NTP = next_token_ce(stack(losses), x)               # (B, T)

    # L_psi: predict cross-stream hidden state at t+h
    psi_target = hidden_state_one_hot(x_paired)           # (B, T, S)
    g_pred = stack([g_theta(z_t) for z_t in trajectory])  # (B, T, S)
    L_psi = mse(g_pred[:, :-h], psi_target[:, h:])

    # L_ESS, L_surprise, L_bound as defined in §6.1
    ...

    return alpha * L_NTP + beta * L_psi + gamma * L_ESS + delta * L_surprise + eta * L_bound
```

---

## Appendix B — Why this is *not* a Mamba in disguise

A reasonable skeptic will ask: isn't this just Mamba with extra fluff? The answer is concrete:

1. **Mamba is k=1.** Its hidden state at each layer is a single d-dimensional vector (well, 16 vectors at d=16 SSD; still single per channel). MuRe at k=1 is LRU, not Mamba. The Mamba-equivalent in MuRe-language would be "k=1 with input-dependent (A, B, Σ) plus diagonal-A". That's a strict special case.

2. **MuRe's particles do not have channels split across them.** Mamba's state has channel-wise (or head-wise) parallelism: different channels of the state vector correspond to different "channels" of representation. MuRe's particles are *alternative complete hypotheses* over the same latent space — they share d_z.

3. **The IB term + L_ψ has no Mamba analogue.** Mamba's training is pure NTP. The bottleneck-with-cross-stream-target is a separate dimension entirely; it is the R2 attack the audit identified as having no clean published baseline.

4. **The OT-resample step is *not* in any SSM/RWKV/Based/DeltaNet.** It is the operation that keeps the particle ensemble multi-modal. Without it, k>1 collapses to k=1 after a few hundred tokens (the Doucet et al. 2000 result).

The honest claim: MuRe at k=1, β=0, η=0 is approximately Mamba-without-input-dependent-A (i.e., S4 or LRU). At k=1, β=0, η=0, with input-dependent A, it is approximately Mamba. At k>1 with OT resampling and the IB+ψ Lagrangian, it is structurally distinct. The novelty axis is therefore (k>1, OT-resample, IB+ψ); we measure all three.

---

End of MuRe candidate framework.
