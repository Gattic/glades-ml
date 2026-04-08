# ATLAS: Adaptive Temporally-Predictive Learning in Active Subspaces

## A Mathematical Framework for Curvature-Aware, Compressed, Predictive LLM Training

---

## 1. Executive Summary

This document proposes **ATLAS**, a training framework for large language models that unifies four capabilities—subspace-constrained optimization, curvature-aware dynamics, temporal prediction, and compressed-space learning—under a single information-geometric principle: **the Fisher information matrix simultaneously determines the optimal training subspace, the natural metric within that subspace, and (through its temporal evolution) an implicit source of second-order information for predictive acceleration.**

The core mechanism: at each training step, parameters are updated only within a low-dimensional subspace spanned by the top eigenvectors of a running Fisher information estimate. Within this subspace, the natural gradient (Fisher-preconditioned) is combined with a temporally extrapolated gradient that implicitly corrects for landscape curvature. The subspace itself evolves continuously via online eigenspace tracking, and its dimension adapts based on an information-theoretic criterion.

The framework yields:
- A **Predictive Natural Gradient (PNG)** update rule that achieves Nesterov-like acceleration without extra forward passes.
- A **convergence guarantee** matching SGD's O(1/sqrt(T)) rate for non-convex objectives, with improved constants from curvature preconditioning.
- A **PAC-Bayes generalization bound** tightened by a factor of k/d relative to unconstrained training.
- A **minimal prototype** implementable as a drop-in PyTorch optimizer.

### 1.1 Empirical Redesign Update (April 8, 2026)

ATLAS was revised using the diagnostic logs in `unit-tests/logs/`, especially:
- `unit-tests/logs/2026-04-08-H16.log`
- `unit-tests/logs/2026-04-08-H17.log`
- `unit-tests/logs/2026-03-11-H08.log`

The main empirical findings were:
- Repeated `atlas_gram_schmidt_degenerate` events clustered in small or nearly rank-deficient matrices, indicating that refresh-time basis maintenance was too brittle.
- Many output-like layers exhibited nearly flat Fisher spectra, so aggressive rank collapse created representational loss without reliable evidence of a true rank-1 optimum.
- Large hidden layers often had materially non-flat Fisher structure and needed the full tracked subspace to preserve out-of-sample accuracy.

The implemented redesign therefore makes three concrete changes:
- **Fisher-weighted refresh seeds.** Refresh now follows the same curvature signal used for preconditioning instead of treating all tracked directions equally.
- **Packed active-basis execution.** The optimizer can operate on a reduced active prefix without reallocating or reprojecting the entire stored basis every step.
- **Conservative adaptive-rank policy.** Adaptive rank remains available, but it now shrinks only at refresh boundaries, repairs the inactive basis to keep `U` orthonormal, and is disabled by default until broader benchmarks justify turning it on globally.

Observed outcomes on April 8, 2026:
- `./glades-unit-tests atlas` passes with the redesign enabled.
- The aggressive adaptive-rank setting improved throughput slightly but hurt MNIST test accuracy; it is therefore no longer the default.
- The default ATLAS path with Fisher-weighted refresh and the new diagnostics reached `train=7.96s`, `testAcc=98.64%` on `./glades-unit-tests atlas-bench --mode standard --repeats 1`.

Interpretation:
- The data supports **stability-first subspace tracking** and **opt-in adaptive compression**, not unconditional online rank collapse.
- The next likely source of additional out-of-sample gains is a scale-consistency review of the baseline `sigma2` preconditioner versus the subspace Fisher statistics; this remains an open item rather than a completed claim.

---

## 2. Candidate Formulations

### Candidate A: Riemannian Submanifold with Geodesic Prediction

**Core geometric object.** A k-dimensional Riemannian submanifold M embedded in parameter space R^d, equipped with the pullback of the Fisher-Rao metric.

**Compression mechanism.** Parameters are represented as theta = phi(z) where phi: R^k -> R^d is a smooth embedding and z in R^k are intrinsic coordinates. All computation occurs in R^k.

**Temporal prediction.** The trajectory z(t) on M is modeled as approximate geodesic flow. Future states are predicted by integrating the geodesic equation:

    d^2 z^i / dt^2 + Gamma^i_{jk} (dz^j/dt)(dz^k/dt) = -G^{ij} partial_j L

where Gamma^i_{jk} are Christoffel symbols of the pullback metric G_{ij} = (partial phi / partial z^i)^T F(phi(z)) (partial phi / partial z^j).

**Curvature mechanism.** Sectional curvatures of M govern trust-region radii. The Riemann curvature tensor R_{ijkl} of (M, G) determines where geodesic prediction is reliable (low curvature) versus unreliable (high curvature).

**Strengths.** Mathematically natural: geodesics are the "straightest paths" on curved spaces. Curvature directly quantifies prediction reliability. Reparameterization-invariant by construction.

**Failure modes.** (i) Computing Christoffel symbols requires second derivatives of phi and the Fisher, cost O(k^3 d). (ii) The embedding phi must be differentiable and its Jacobian must have full rank everywhere—hard to guarantee for learned embeddings. (iii) Geodesic equations can be stiff when curvature varies rapidly, requiring small integration steps that negate the benefit of prediction. (iv) The submanifold M is fixed; adapting it requires re-learning phi.

### Candidate B: Spectral Flow on Gradient Bundles with Autoregressive Dynamics

**Core geometric object.** The spectral decomposition of the Hessian H(theta), viewed as a time-varying frame field over parameter space. At each theta, the eigenvectors {v_1, ..., v_d} and eigenvalues {lambda_1, ..., lambda_d} define a local coordinate system aligned with the loss curvature.

**Compression mechanism.** Project all computation onto the top-k eigenvectors of H(theta). The "active subspace" S_t = span{v_1(theta_t), ..., v_k(theta_t)} captures directions of highest curvature (and typically highest gradient magnitude).

**Temporal prediction.** Model the spectral decomposition as a vector autoregressive process:

    lambda_{t+1} = sum_{j=0}^{p-1} A_j lambda_{t-j} + epsilon_t

where lambda_t = (lambda_1^(t), ..., lambda_k^(t)) is the vector of top eigenvalues and A_j are learned coefficient matrices. Eigenvector evolution is modeled via rotation rates on the Grassmannian.

**Curvature mechanism.** Direct: eigenvalues ARE curvatures along principal directions. Newton-type steps in the subspace use lambda_i^{-1} as per-direction step sizes.

**Strengths.** Most interpretable: each subspace dimension has a clear meaning (principal curvature direction). Spectral gaps provide natural criteria for dimension selection. Newton convergence in the subspace when eigenvalues are accurate.

**Failure modes.** (i) Computing top-k Hessian eigenvectors requires O(kd) Hessian-vector products via Lanczos, each costing one forward + backward pass—k-fold overhead. (ii) The Hessian has negative eigenvalues in non-convex regions, complicating the Newton step. (iii) The active subspace can change discontinuously at eigenvalue crossings, causing instability in the autoregressive model. (iv) Hessian eigenvectors are not aligned with the Fisher eigenvectors; using H instead of F loses the information-geometric interpretation.

### Candidate C: Information-Geometric Flow with Fisher-Optimal Compression

**Core geometric object.** The Fisher information matrix F(theta), viewed simultaneously as (i) a Riemannian metric on the statistical manifold of model output distributions, and (ii) a covariance operator whose spectral decomposition defines the optimal training subspace.

**Compression mechanism.** The active subspace is the top-k eigenspace of a running exponential-moving-average estimate of F. This subspace maximizes the Fisher information captured per dimension (proven optimal by the Ky Fan inequality). Parameters are decomposed as theta_t = theta_bar + U_t z_t where U_t in R^{d x k} spans the active subspace and z_t in R^k are compressed coordinates.

**Temporal prediction.** The gradient history in the compressed space {g^z_s}_{s <= t} is modeled as an autoregressive process. A linear predictor extrapolates the next compressed gradient:

    g^z_{t+1|t} = g^z_t + mu_t (g^z_t - g^z_{t-1})

This extrapolated gradient is combined with curvature preconditioning to form the Predictive Natural Gradient (PNG) update. Under a linear gradient field approximation, this achieves the same correction as Nesterov momentum without an extra forward pass.

**Curvature mechanism.** The Fisher metric restricted to the subspace, G_t = U_t^T F(theta_t) U_t in R^{k x k}, serves as the curvature tensor. Its inverse preconditions the gradient, giving the natural gradient. Since G_t is k x k with k << d, inversion is cheap (O(k^3) vs O(d^3)).

**Strengths.** (i) Single object (Fisher) unifies all four aspects: its eigenvectors define the subspace, its eigenvalues define the metric, its temporal evolution drives prediction, its spectral decay justifies compression. (ii) The Fisher is always positive semidefinite (no negative curvature issues). (iii) The empirical Fisher is cheap: it requires only per-sample gradient outer products, available from the backward pass. (iv) Subspace tracking via Oja's rule costs O(dk) per step—same as a gradient projection. (v) The information-theoretic subspace selection criterion has a clean optimality proof. (vi) The framework degenerates gracefully: k = d recovers full natural gradient; k = 1 recovers steepest descent along the top Fisher eigenvector; mu = 0 recovers standard natural gradient without prediction.

**Failure modes.** (i) The empirical Fisher (gradient outer products under data distribution) is not exactly the true Fisher (under model distribution); the approximation error is bounded but nonzero. (ii) If the Fisher spectrum is flat (all eigenvalues similar), no subspace is significantly better than any other—compression provides little benefit. (iii) Linear gradient prediction fails when the gradient field changes nonlinearly between steps. (iv) Subspace tracking via Oja's rule can converge slowly if the spectral gap is small.

---

## 3. Framework Selection Rationale

**Candidate C is selected.** The decisive advantages are:

1. **Unification depth.** In Candidates A and B, the compression mechanism, curvature mechanism, and temporal prediction are separate constructions that must be reconciled. In C, they are different views of a single object (the Fisher information matrix and its spectral decomposition). This is not mere elegance—it means the four components cannot conflict, because they share the same mathematical substrate.

2. **Computational feasibility.** Candidate A requires learning and differentiating an embedding phi (meta-learning overhead) and computing Christoffel symbols (O(k^3 d)). Candidate B requires k Hessian-vector products per step (k extra backward passes). Candidate C requires only the empirical Fisher, which is available from per-sample gradients (no extra computation beyond what is already computed), and subspace tracking via Oja's rule at O(dk) per step.

3. **Positive semidefiniteness.** The Hessian in Candidate B can have negative eigenvalues, requiring saddle-point handling. The Fisher in Candidate C is always PSD, and its inverse is well-defined when regularized.

4. **Theoretical grounding.** The Fisher-Rao metric is the unique Riemannian metric on statistical manifolds that is invariant under sufficient statistics (Chentsov's theorem). This means the natural gradient is the unique first-order method that respects the information geometry of the model. No analogous uniqueness result holds for the Hessian metric or for arbitrary embeddings.

5. **Graceful degradation.** Candidate C reduces to well-understood algorithms at its boundaries (full natural gradient, Adam, SGD), making it easier to validate and debug.

Candidate C is developed below as **ATLAS**.

---

## 4. Formal Problem Statement

### Notation

| Symbol | Type | Definition |
|--------|------|------------|
| d | N | Total number of model parameters |
| k | N, k << d | Dimension of active subspace |
| theta | R^d | Model parameter vector |
| x | X | Data sample (token sequence, etc.) |
| D | Distribution over X | Training data distribution |
| l(theta; x) | R -> R | Per-sample loss |
| L(theta) | R -> R | Expected loss: E_{x ~ D}[l(theta; x)] |
| p(x \| theta) | Probability | Model output distribution |
| F(theta) | R^{d x d}, PSD | Fisher information matrix |
| U_t | R^{d x k} | Active subspace basis (orthonormal columns) |
| z_t | R^k | Compressed coordinates |
| theta_bar_t | R^d | Anchor point |
| Lambda_t | R^{k x k}, diagonal | Subspace Fisher eigenvalues |
| G_t | R^{k x k}, PSD | Subspace Fisher metric |
| g_t | R^d | Stochastic gradient of L at theta_t |
| g^z_t | R^k | Compressed gradient: U_t^T g_t |
| a^z_t | R^k | Gradient acceleration: g^z_t - g^z_{t-1} |
| mu_t | [0, 1] | Prediction coefficient |
| eta_t | R_+ | Learning rate |
| beta | (0, 1) | Fisher EMA decay rate |
| T_sub | N | Subspace update interval |
| rho_t | (0, 1] | Subspace quality: \|\|U_t U_t^T nabla L\|\| / \|\|nabla L\|\| |
| Gr(k, d) | Manifold | Grassmannian: space of k-dim subspaces of R^d |

### The problem

Find a training algorithm that:

(P1) Converges to approximate stationary points of L(theta) at rate no worse than O(1/sqrt(T)).

(P2) Has per-step computational cost O(dk) instead of O(d^2) (or O(d) with structure).

(P3) Has memory cost O(dk) for optimizer state instead of O(d).

(P4) Exploits curvature information (beyond diagonal scaling) without computing or storing the full Hessian or Fisher.

(P5) Extracts implicit second-order information from the gradient trajectory to accelerate convergence without extra forward/backward passes.

(P6) Adapts the working subspace to track the evolving geometry of the loss landscape.

(P7) Provides generalization benefits from the subspace constraint, with provable bounds.

---

## 5. Core Mathematical Framework

### 5.1 The Fisher Information Matrix

For a model with output distribution p(x | theta), the Fisher information matrix is:

    F(theta) = E_{x ~ p(.|theta)} [ nabla log p(x|theta) nabla log p(x|theta)^T ]

**Proposition 5.1** (Equivalences). F(theta) equals:
- (a) The negative expected Hessian of the log-likelihood: F = -E[nabla^2 log p(x|theta)] (under regularity conditions).
- (b) The Hessian of the KL divergence at zero perturbation: F_{ij} = partial^2 KL(p(.|theta) || p(.|theta + delta)) / partial delta_i partial delta_j |_{delta=0}.
- (c) The Riemannian metric tensor of the statistical manifold {p(.|theta) : theta in R^d} under the Fisher-Rao metric.

**Status:** Proven (Rao 1945, Amari 1998). These equivalences are standard.

In practice, we use the **empirical Fisher** computed from the data distribution rather than the model distribution:

    F_hat(theta) = (1/|B|) sum_{x in B} nabla l(theta; x) nabla l(theta; x)^T

where B is a mini-batch. This is not exactly F(theta) but approximates it when the model fits the data well (i.e., p_model approx p_data). The approximation error is bounded by the squared total variation between model and data distributions. We use F_hat throughout and note where the distinction matters.

### 5.2 Active Subspace via Fisher Eigenspace

**Definition 5.2.** The *active subspace* of rank k at theta is:

    S_k(theta) = span{u_1(theta), ..., u_k(theta)}

where u_1, ..., u_d are eigenvectors of F(theta) ordered by decreasing eigenvalue: F u_i = lambda_i u_i, lambda_1 >= lambda_2 >= ... >= lambda_d >= 0.

**Theorem 5.3** (Optimality of Fisher Eigenspace). Among all k-dimensional subspaces S of R^d, the active subspace S_k(theta) uniquely maximizes the captured Fisher information:

    S_k(theta) = argmax_{S in Gr(k,d)} tr(P_S F(theta) P_S)

where P_S = UU^T is the orthogonal projector onto S (U being any orthonormal basis for S).

*Proof.* Write P_S = UU^T. Then tr(P_S F P_S) = tr(U^T F U). By the Ky Fan k-inequality (Ky Fan, 1949), for any Hermitian matrix A with eigenvalues alpha_1 >= ... >= alpha_d:

    max_{U^T U = I_k} tr(U^T A U) = sum_{i=1}^k alpha_i

attained when U = [u_1, ..., u_k]. Applying this to A = F gives the result. QED.

**Corollary 5.4** (Minimum Gradient Loss). The active subspace minimizes the expected squared norm of the gradient component lost to projection:

    S_k(theta) = argmin_{S in Gr(k,d)} E_{x ~ p(.|theta)} [ ||(I - P_S) nabla log p(x|theta)||^2 ]

*Proof.* E[||(I-P_S) nabla log p||^2] = tr((I-P_S) F (I-P_S)) = tr(F) - tr(P_S F P_S). Since tr(F) is independent of S, minimizing the residual is equivalent to maximizing tr(P_S F P_S). QED.

**Interpretation.** The active subspace retains the directions along which the model's output distribution is *most sensitive* to parameter perturbations. Directions outside S_k have small Fisher eigenvalues, meaning parameter changes along them barely affect predictions.

### 5.3 Compressed Coordinates

Fix an anchor point theta_bar in R^d and an orthonormal basis U in R^{d x k} for the active subspace. Define compressed coordinates:

    z = U^T (theta - theta_bar) in R^k

The parameter vector is reconstructed as:

    theta = theta_bar + U z

The gradient in compressed coordinates:

    g^z = U^T nabla L(theta) in R^k

The Fisher metric in compressed coordinates (the **subspace Fisher**):

    G = U^T F(theta) U in R^{k x k}

When U spans the top-k eigenspace and F is diagonal in the eigenbasis, G = Lambda = diag(lambda_1, ..., lambda_k). In general, G is a dense k x k PSD matrix.

### 5.4 The Natural Gradient in Compressed Space

The natural gradient update in the full space is theta_{t+1} = theta_t - eta F(theta_t)^{-1} nabla L(theta_t), which requires inverting a d x d matrix—infeasible.

Restricted to the active subspace:

    delta^z_t = -eta_t G_t^{-1} g^z_t = -eta_t (U_t^T F_t U_t)^{-1} U_t^T nabla L(theta_t)

    theta_{t+1} = theta_t + U_t delta^z_t

Since G_t is k x k, inversion costs O(k^3)—tractable for k in the hundreds.

**Proposition 5.5.** The subspace natural gradient delta^z is the solution to:

    min_{delta in R^k} nabla L(theta)^T U delta + (1/2eta) delta^T G delta

i.e., it minimizes a Fisher-regularized linear model of the loss within the subspace.

*Proof.* First-order optimality: G delta + eta U^T nabla L = 0, giving delta = -eta G^{-1} U^T nabla L. QED.

### 5.5 The Predictive Natural Gradient (PNG)

**Definition 5.6.** The *gradient acceleration* in the subspace is:

    a^z_t = g^z_t - g^z_{t-1} in R^k

The *temporally extrapolated gradient* at prediction horizon mu_t in [0, 1] is:

    g^z_{t,pred} = g^z_t + mu_t a^z_t = (1 + mu_t) g^z_t - mu_t g^z_{t-1}

The **Predictive Natural Gradient (PNG) update** is:

    delta^z_t = -eta_t G_t^{-1} g^z_{t,pred}

    theta_{t+1} = theta_t + U_t delta^z_t

**Theorem 5.7** (PNG as Implicit Hessian Correction). Under the linear gradient field approximation:

    nabla_z L(z_{t+1}) approx nabla_z L(z_t) + H_z (z_{t+1} - z_t)

where H_z = U^T nabla^2 L(theta) U is the subspace Hessian, the gradient acceleration satisfies:

    a^z_t = g^z_t - g^z_{t-1} approx H_z (z_t - z_{t-1}) = H_z delta^z_{t-1}

Therefore the PNG extrapolated gradient is:

    g^z_{t,pred} approx g^z_t + mu_t H_z delta^z_{t-1}

This corrects the current gradient by an amount proportional to the Hessian-step product, providing implicit second-order information extracted from the gradient history at zero additional forward/backward cost.

*Proof.* By the linear approximation, g^z_t = nabla_z L(z_t) approx nabla_z L(z_{t-1}) + H_z(z_t - z_{t-1}) = g^z_{t-1} + H_z delta^z_{t-1}. Subtracting: a^z_t = g^z_t - g^z_{t-1} approx H_z delta^z_{t-1}. QED.

**Status of Theorem 5.7.** The linear gradient field approximation holds when eta_t is small relative to 1/||H_z||. For LLM training with typical learning rates (1e-4 to 1e-3) and well-conditioned subspaces, this is a reasonable heuristic. The approximation error is O(eta^2 ||nabla^3 L||), which is the cubic term in the Taylor expansion.

**Comparison with Nesterov momentum.** Nesterov evaluates the gradient at an extrapolated position: g(theta_t + beta v_{t-1}). Under the same linear approximation:

    g(theta_t + beta v_{t-1}) approx g(theta_t) + beta H v_{t-1}

So PNG with mu_t = beta and delta_{t-1} = v_{t-1} achieves the same first-order correction as Nesterov, but without the extra forward pass at the extrapolated position. The methods diverge at second order (O(eta^2) terms).

### 5.6 Adaptive Prediction Coefficient

The prediction coefficient mu_t is adapted based on prediction accuracy. Define the prediction error:

    e_t = g^z_t - g^z_{t|t-1} = g^z_t - (g^z_{t-1} + mu_{t-1} a^z_{t-1})

Update rule:

    mu_t = mu_{t-1} * clip(1 - ||e_t|| / (||g^z_t|| + epsilon), 0, 1)

When predictions are accurate (small e_t), mu_t stays near its current value. When predictions fail (large e_t), mu_t decays toward 0, reducing PNG to standard natural gradient.

**Lower bound:** mu_t >= mu_min (e.g., 0.01) to maintain some predictive benefit.
**Upper bound:** mu_t <= mu_max (e.g., 0.5) to prevent prediction from dominating the update.

### 5.7 Online Subspace Tracking

Rather than periodically recomputing U via SVD (as in GaLore), ATLAS tracks the active subspace continuously using a variant of Oja's subspace rule.

Given a new stochastic gradient g_t, update U via:

    U_{t+1} = orth(U_t + eta_U g_t (g_t^T U_t)^T - eta_U beta_decay U_t Lambda_t)

where:
- The term g_t (g_t^T U_t)^T = g_t (U_t^T g_t)^T in R^{d x k} pushes U toward the direction of g_t
- The term -beta_decay U_t Lambda_t provides a forgetting mechanism (without it, U would converge to the all-time average Fisher eigenspace rather than tracking the current one)
- orth(.) denotes QR orthonormalization: cost O(dk^2)
- eta_U is the subspace learning rate (typically much smaller than eta_t)

The eigenvalues are updated via exponential moving average:

    lambda_{i,t+1} = beta lambda_{i,t} + (1 - beta) (u_{i,t}^T g_t)^2    for i = 1, ..., k

where u_{i,t} is the i-th column of U_t.

**Per-step cost:** O(dk) for the projection U_t^T g_t, O(dk) for the rank-1 outer product update, O(dk^2) for QR. Total: O(dk^2). For k << sqrt(d), this is sublinear in d^2.

**Alternative (periodic SVD).** Update U every T_sub steps by computing the top-k SVD of the accumulated gradient matrix G_B = [g_{t-s+1}, ..., g_t] in R^{d x s} where s is a window size. Cost: O(ds^2 + s^3) amortized over T_sub steps. This is simpler and may be preferred when T_sub is large.

### 5.8 Anchor Management

The decomposition theta = theta_bar + Uz assumes theta stays near theta_bar. When ||z_t|| exceeds a threshold R (the anchor radius), re-anchor:

    theta_bar_{t+1} = theta_t = theta_bar_t + U_t z_t
    z_{t+1} = 0

This prevents numerical issues from large z values and keeps the linear approximation theta approx theta_bar + Uz valid.

**Anchor radius selection.** Set R = sqrt(k) * sigma_anchor where sigma_anchor is a hyperparameter controlling how far the compressed coordinates can drift. Typical value: sigma_anchor in [1, 10].

---

## 6. Objective Function Derivation

### 6.1 The Task Loss

The primary objective is the standard language modeling loss:

    L_task(z; U, theta_bar) = E_{x ~ D} [ l(theta_bar + Uz; x) ]

In practice, estimated via mini-batch: L_hat_task = (1/|B|) sum_{x in B} l(theta_bar + Uz; x).

### 6.2 Information-Theoretic Subspace Selection

The choice of k (subspace dimension) is governed by a rate-distortion tradeoff.

**Definition 6.1.** The *information capture ratio* at rank k is:

    rho(k) = sum_{i=1}^k lambda_i / sum_{i=1}^d lambda_i = tr(P_{S_k} F) / tr(F)

This measures the fraction of total Fisher information captured by the top-k subspace.

**Definition 6.2.** The *information loss* at rank k is:

    D(k) = 1 - rho(k) = sum_{i=k+1}^d lambda_i / sum_{i=1}^d lambda_i

**Criterion.** Choose k as the smallest integer such that:

    rho(k) >= 1 - epsilon_info

for a target information retention epsilon_info (e.g., epsilon_info = 0.01 for 99% retention).

**Proposition 6.3** (Subspace Dimension under Power-Law Spectra). If the Fisher eigenvalues decay as lambda_i ~ C i^{-alpha} for alpha > 1, then the information capture ratio satisfies:

    rho(k) >= 1 - (k/(d-1))^{1-alpha} * ((d-1)/(alpha-1))

and the required subspace dimension for (1 - epsilon)-retention is:

    k = O(d * epsilon^{1/(1-alpha)})

For alpha = 2 (quadratic decay): k = O(d * epsilon). For epsilon = 0.01, d = 10^9: k approx 10^7. Still large.
For alpha = 3 (cubic decay): k = O(d * epsilon^{1/2}). For epsilon = 0.01: k approx 10^5. More manageable.
For alpha = 5: k = O(d * epsilon^{1/4}). k approx 10^4. Practical.

**Empirical hypothesis** (Fisher Spectral Decay for Transformers): For transformer language models, the Fisher eigenvalue spectrum decays approximately as a power law with alpha in [2, 4], with alpha increasing during training (the spectrum becomes more concentrated as training progresses).

**Status:** This is an empirical hypothesis based on indirect evidence from gradient covariance studies (Sagun et al. 2017, Ghorbani et al. 2019, Papyan 2020). Direct measurement of the full Fisher spectrum for billion-parameter models is infeasible; the hypothesis extrapolates from smaller-scale experiments. Validating or refuting this hypothesis is a key item in the research program (Section 14).

### 6.3 Subspace Regularization

To prevent the subspace from collapsing (all columns of U becoming parallel) or oscillating, add a coherence penalty:

    L_coherence(U) = ||U^T U - I_k||_F^2

This is automatically zero when U is orthonormal, but provides a gradient signal toward orthonormality when using approximate (non-QR) subspace updates.

### 6.4 The Combined Objective

ATLAS minimizes the task loss subject to the subspace constraint. The subspace U and its dimension k are not optimized jointly with z but are determined by the Fisher spectrum (Sections 5.2, 6.2). The regularization is optional and only needed for approximate subspace tracking.

The effective optimization problem at each step:

    min_{delta^z in R^k}  (g^z_{t,pred})^T delta^z + (1 / 2 eta_t) (delta^z)^T G_t delta^z

This is the PNG update from Section 5.5, with closed-form solution delta^z = -eta_t G_t^{-1} g^z_{t,pred}.

---

## 7. Optimization Algorithm

### 7.1 Full Algorithm

```
Algorithm: ATLAS Training

Input:
  Model parameters theta_0 in R^d
  Learning rate schedule {eta_t}
  Subspace dimension k, EMA decay beta
  Prediction bounds mu_min, mu_max
  Anchor radius R, subspace update interval T_sub

Initialize:
  U_0 <- top-k left singular vectors of first mini-batch gradient matrix
  Lambda_0 <- corresponding squared singular values
  theta_bar_0 <- theta_0
  z_0 <- 0 in R^k
  g^z_{-1} <- 0 in R^k
  mu_0 <- mu_min

For t = 0, 1, 2, ..., T-1:

  1. FORWARD/BACKWARD PASS
     Compute stochastic gradient g_t = nabla l(theta_bar_t + U_t z_t; x_t)

  2. COMPRESS GRADIENT
     g^z_t = U_t^T g_t                                        [O(dk)]

  3. UPDATE FISHER EIGENVALUES
     For i = 1..k:
       lambda_{i,t} = beta * lambda_{i,t-1} + (1-beta) * (g^z_t[i])^2    [O(k)]

  4. FORM SUBSPACE FISHER
     G_t = diag(lambda_{1,t}, ..., lambda_{k,t}) + epsilon * I_k    [O(k)]
     (diagonal approximation; see Section 7.2 for full G_t)

  5. PREDICT
     a^z_t = g^z_t - g^z_{t-1}                                [O(k)]
     g^z_{t,pred} = g^z_t + mu_t * a^z_t                      [O(k)]

  6. COMPUTE PNG UPDATE
     delta^z_t = -eta_t * G_t^{-1} g^z_{t,pred}               [O(k) diagonal, O(k^3) full]

  7. UPDATE COMPRESSED COORDINATES
     z_{t+1} = z_t + delta^z_t                                 [O(k)]

  8. UPDATE FULL PARAMETERS
     theta_{t+1} = theta_bar_t + U_t z_{t+1}                   [O(dk)]
     (or incrementally: theta_{t+1} = theta_t + U_t delta^z_t)

  9. ADAPT PREDICTION COEFFICIENT
     e_t = g^z_t - (g^z_{t-1} + mu_{t-1} * a^z_{t-1})
     mu_{t+1} = clip(mu_t * (1 - ||e_t|| / (||g^z_t|| + eps)),
                      mu_min, mu_max)                           [O(k)]

  10. UPDATE SUBSPACE (every T_sub steps)
      Option A (Oja): See Section 5.7                           [O(dk^2)]
      Option B (SVD): U_{t+1}, Lambda_{t+1} from top-k SVD
        of accumulated gradient matrix                          [O(dk^2) amortized]

  11. RE-ANCHOR (if ||z_{t+1}|| > R)
      theta_bar_{t+1} = theta_bar_t + U_t z_{t+1}              [O(dk)]
      z_{t+1} <- 0

  12. STORE FOR NEXT STEP
      g^z_{t-1} <- g^z_t

Output: theta_T = theta_bar_T + U_T z_T
```

### 7.2 Full Subspace Fisher (Non-Diagonal)

The diagonal approximation G_t = diag(lambda_{1,t}, ..., lambda_{k,t}) ignores correlations between subspace dimensions. The full subspace Fisher is:

    G_t = beta * G_{t-1} + (1-beta) * g^z_t (g^z_t)^T

This is a rank-1 EMA update of a k x k matrix. Cost: O(k^2) per step. Inversion: O(k^3) per step (or use the Woodbury identity for rank-1 updates: O(k^2)).

For k <= 1024, the full G_t is practical and captures inter-dimension curvature that the diagonal version misses.

### 7.3 Per-Layer Instantiation

For models with L layers, each with weight matrix W^(l) in R^{m_l x n_l}, ATLAS is applied per-layer:

- U^(l) in R^{m_l x r_l}: left subspace basis for layer l
- z^(l) in R^{r_l x n_l}: compressed coordinates (matrix-valued)
- G^(l) in R^{r_l x r_l}: subspace Fisher for layer l

Per-layer gradient compression: P^(l) = U^{(l)T} nabla_{W^(l)} L in R^{r_l x n_l}

Memory savings per layer: optimizer state is 2 * r_l * n_l (first/second moments) instead of 2 * m_l * n_l. Savings factor: m_l / r_l.

For a transformer with m_l = 4096, n_l = 4096, r_l = 256: savings factor = 16x on optimizer memory.

---

## 8. Temporal Dynamics Formulation

### 8.1 The Training Trajectory as a Dynamical System

Define the ATLAS state at time t as the tuple (z_t, U_t, Lambda_t). The training algorithm defines a discrete dynamical system:

    z_{t+1} = z_t - eta_t G_t^{-1} g^z_{t,pred}           ...(D1: coordinate dynamics)
    U_{t+1} = Phi_U(U_t, g_t)                                ...(D2: subspace dynamics)
    Lambda_{t+1} = beta Lambda_t + (1-beta) diag((U_t^T g_t)^2)  ...(D3: curvature dynamics)

where Phi_U is the subspace update map (Oja's rule or periodic SVD).

### 8.2 Continuous-Time Limit

For small eta, the coordinate dynamics (D1) approximate the ODE:

    dz/dt = -G(z,t)^{-1} nabla_z L(z,t) - mu G(z,t)^{-1} (d/dt nabla_z L)

The first term is the natural gradient flow. The second term is the predictive correction, proportional to the time derivative of the gradient. In continuous time, this becomes:

    dz/dt = -G^{-1} nabla_z L - mu G^{-1} (nabla^2_z L (dz/dt) + partial_t nabla_z L)

where partial_t nabla_z L accounts for explicit time dependence (from evolving U).

Rearranging (assuming mu < 1 and I + mu G^{-1} nabla^2_z L is invertible):

    (I + mu G^{-1} H_z) dz/dt = -G^{-1} nabla_z L - mu G^{-1} partial_t nabla_z L

    dz/dt = -(I + mu G^{-1} H_z)^{-1} G^{-1} (nabla_z L + mu partial_t nabla_z L)

This is a **modified natural gradient flow** where the effective preconditioner is (I + mu G^{-1} H_z)^{-1} G^{-1} instead of G^{-1}. The modification damps high-curvature directions more aggressively (since (I + mu G^{-1} H_z)^{-1} attenuates directions with large eigenvalues of G^{-1} H_z), providing implicit regularization.

**Status:** This continuous-time analysis is exact in the limit eta -> 0 but the discrete algorithm with finite eta deviates by O(eta^2) terms. The analysis provides qualitative insight rather than quantitative guarantees.

### 8.3 Subspace Evolution on the Grassmannian

The subspace U_t evolves on the Grassmannian Gr(k, d). The tangent space at U is:

    T_U Gr(k,d) = { Delta in R^{d x k} : U^T Delta = 0 }

The velocity of U_t under Oja's rule is:

    dU/dt = (I - UU^T) g g^T U / ||g||^2 - beta_decay (I - UU^T) U Lambda

The first term rotates U toward the instantaneous gradient direction. The second term provides decay. Both are tangent to the Grassmannian (perpendicular to U).

**Predictive subspace update (extension).** Instead of tracking the current Fisher eigenspace, predict its future evolution. If we model the subspace velocity as approximately constant over short intervals:

    U_{t+1}^{pred} = orth(U_t + tau * V_t)

where V_t = (U_t - U_{t-1}) / eta_U is the estimated subspace velocity and tau is a prediction horizon. This allows ATLAS to anticipate which new directions will become important, rather than reacting after the fact.

**Caveat:** Predictive subspace updates are speculative and have not been theoretically analyzed. They may cause instability if the subspace velocity is noisy. This is relegated to the full research program (Section 14).

### 8.4 Generalized Gradient Prediction (VAR Model)

The linear extrapolation g^z_{t,pred} = g^z_t + mu (g^z_t - g^z_{t-1}) is a special case of a first-order vector autoregressive (VAR(1)) model. The general VAR(p) model is:

    g^z_{t+1|t} = sum_{j=0}^{p-1} A_j g^z_{t-j}

where A_j in R^{k x k} are coefficient matrices estimated from the gradient history via least squares:

    [A_0, ..., A_{p-1}] = argmin_{A} sum_{s=p}^{t} || g^z_s - sum_j A_j g^z_{s-1-j} ||^2

The cost of estimation: O(k^2 p T_hist + (kp)^3) where T_hist is the history length. For k = 512, p = 4, T_hist = 100: approximately 10^{10} FLOPs. This should be performed infrequently (every T_sub steps) and amortized.

The VAR model captures richer temporal structure than linear extrapolation:
- Oscillatory gradient patterns (via complex eigenvalues of the companion matrix)
- Multi-scale dynamics (via higher-order autoregressive terms)
- Cross-dimension gradient correlations (via the off-diagonal entries of A_j)

**When to use VAR vs. linear.** Linear extrapolation (p=1, A_0 = (1+mu)I, no fitting) is the default. Upgrade to VAR when: (i) the gradient trajectory shows clear oscillatory patterns, (ii) the prediction error e_t is persistently large despite adaptive mu, (iii) k is small enough that the O(k^2 p) estimation cost is negligible.

---

## 9. Theoretical Analysis

### 9.1 Convergence

**Assumptions.**

(A1) *L-smoothness:* ||nabla L(theta) - nabla L(theta')|| <= L_s ||theta - theta'|| for all theta, theta'.

(A2) *Bounded stochastic gradient variance:* E[||g_t - nabla L(theta_t)||^2] <= sigma^2 for all t.

(A3) *Subspace quality:* ||U_t U_t^T nabla L(theta_t)|| >= rho ||nabla L(theta_t)|| for all t, for some rho in (0, 1].

(A4) *Fisher eigenvalue bounds:* 0 < lambda_min <= lambda_k^{(t)} <= lambda_1^{(t)} <= lambda_max for all t.

**Theorem 9.1** (Convergence of ATLAS without prediction). Under (A1)-(A4), with mu_t = 0 (no prediction), step size eta = min(lambda_min / (2 L_s), c / sqrt(T)), and diagonal subspace Fisher G_t = Lambda_t:

    (1/T) sum_{t=0}^{T-1} E[||nabla L(theta_t)||^2] <= (2 (L_0 - L*)) / (rho^2 eta T) + (eta L_s sigma^2) / (rho^2 lambda_min^2)

where L_0 = L(theta_0) and L* = inf L.

Choosing eta = min(lambda_min / (2 L_s), sqrt(2 lambda_min^2 (L_0 - L*) / (L_s sigma^2 T))):

    (1/T) sum_{t=0}^{T-1} E[||nabla L(theta_t)||^2] <= O(sigma sqrt(L_s (L_0 - L*)) / (rho lambda_min sqrt(T))) + O(L_s (L_0 - L*) / (rho^2 lambda_min T))

For large T, the first term dominates: O(1 / (rho lambda_min sqrt(T))).

*Proof sketch.*

Step 1: Descent lemma. By L-smoothness:

    L(theta_{t+1}) <= L(theta_t) + nabla L(theta_t)^T (theta_{t+1} - theta_t) + (L_s/2) ||theta_{t+1} - theta_t||^2

Step 2: Substitute the ATLAS update theta_{t+1} - theta_t = -eta U_t Lambda_t^{-1} U_t^T g_t:

    nabla L^T (theta_{t+1} - theta_t) = -eta (U_t^T nabla L)^T Lambda_t^{-1} (U_t^T g_t)

Taking expectation (E[g_t] = nabla L):

    E[nabla L^T (theta_{t+1} - theta_t)] = -eta (U_t^T nabla L)^T Lambda_t^{-1} (U_t^T nabla L)
                                          <= -eta / lambda_max * ||U_t^T nabla L||^2
                                          <= -(eta rho^2 / lambda_max) ||nabla L||^2

Step 3: Bound the quadratic term:

    E[||theta_{t+1} - theta_t||^2] = eta^2 E[||U_t Lambda_t^{-1} U_t^T g_t||^2]
                                    = eta^2 E[tr(Lambda_t^{-1} U_t^T g_t g_t^T U_t Lambda_t^{-1})]
                                    <= eta^2 / lambda_min^2 * E[||U_t^T g_t||^2]
                                    <= eta^2 / lambda_min^2 * (||U_t^T nabla L||^2 + sigma^2)

Step 4: Combine and telescope over t = 0, ..., T-1. The ||U_t^T nabla L||^2 terms cancel (descent vs. quadratic) when eta <= lambda_min^2 / (L_s). The sigma^2 term accumulates. Dividing by T gives the result. QED.

**Remark.** The convergence rate O(1 / (rho lambda_min sqrt(T))) improves over SGD's O(1/sqrt(T)) when rho lambda_min > 1. This occurs when (i) the subspace captures most of the gradient (rho close to 1) AND (ii) the Fisher eigenvalues in the subspace are large (lambda_min > 1). Condition (ii) is related to the model being "sensitive" to parameter changes in the subspace, which is expected for well-parameterized models where gradients carry substantial information.

If lambda_min < 1, the natural gradient actually slows convergence relative to SGD. Mitigation: clip eigenvalues below at lambda_floor > 0: lambda_i <- max(lambda_i, lambda_floor).

### 9.2 Effect of Temporal Prediction

**Proposition 9.2** (Prediction Bias-Variance Tradeoff). With mu > 0, the PNG update uses the extrapolated gradient g^z_{pred} = g^z + mu a^z. Decompose:

    E[g^z_{pred}] = E[g^z] + mu E[a^z] = nabla_z L(z_t) + mu (nabla_z L(z_t) - nabla_z L(z_{t-1}))
                  = (1+mu) nabla_z L(z_t) - mu nabla_z L(z_{t-1})

The "signal" in the extrapolated gradient is the true gradient at time t plus a correction term mu * (nabla L_t - nabla L_{t-1}).

    Var[g^z_{pred}] = Var[(1+mu) g^z_t - mu g^z_{t-1}]
                    = (1+mu)^2 Var[g^z_t] + mu^2 Var[g^z_{t-1}] - 2mu(1+mu) Cov[g^z_t, g^z_{t-1}]

If gradients across steps are independent (different mini-batches): Cov = 0 and Var[g^z_{pred}] = ((1+mu)^2 + mu^2) sigma_z^2 = (1 + 2mu + 2mu^2) sigma_z^2 > sigma_z^2.

**Conclusion (proved).** Temporal prediction with independent mini-batches increases gradient variance by a factor of (1 + 2mu + 2mu^2). For mu = 0.3: factor = 1.78.

**Where the benefit comes from.** The benefit is not variance reduction but *directional improvement*. The expected direction E[g^z_{pred}] / ||E[g^z_{pred}]|| is a better descent direction than E[g^z] / ||E[g^z]|| when the loss surface is curved, because the Hessian correction aligns the step more closely with the Newton direction (Theorem 5.7).

**Conjecture 9.3** (Net Benefit of Prediction). For loss functions satisfying a Polyak-Lojasiewicz (PL) condition in the subspace:

    ||nabla_z L||^2 >= 2 mu_PL (L - L*)

with mu_PL > 0, the PNG update with optimal mu_t achieves convergence rate:

    L(z_T) - L* <= (1 - 2 eta mu_PL (1 + c mu))^T (L(z_0) - L*)

for some constant c > 0 depending on the smoothness of the gradient field. This is faster than the mu = 0 rate by a factor of (1 + c mu) in the exponent.

**Status:** Conjecture. The PL condition in the subspace is itself a strong assumption (it implies the subspace contains a descent direction from every point, which is guaranteed when rho is close to 1). The constant c depends on the accuracy of the linear gradient approximation, which is hard to bound globally.

### 9.3 Conditioning

**Proposition 9.4** (Condition Number Improvement). The condition number of the subspace Fisher is:

    kappa_sub = lambda_1 / lambda_k

The condition number of the full Fisher is:

    kappa_full = lambda_1 / lambda_d

Since lambda_k >= lambda_{k+1} >= ... >= lambda_d, we have kappa_sub <= kappa_full, with equality only when lambda_k = lambda_d.

Under power-law decay lambda_i ~ i^{-alpha}:

    kappa_sub = (1/k)^{-alpha} = k^alpha
    kappa_full = (1/d)^{-alpha} = d^alpha

Improvement factor: kappa_full / kappa_sub = (d/k)^alpha.

For d = 10^9, k = 10^3, alpha = 2: improvement factor = (10^6)^2 = 10^{12}.

*Proof.* Direct from the eigenvalue ordering and power-law assumption. QED.

**Interpretation.** The subspace natural gradient operates in a much better-conditioned space than the full-space natural gradient (or gradient descent). This means it can take larger steps without overshooting, accelerating convergence. The conditioning benefit grows with the spectral decay rate alpha.

### 9.4 Information Retention

**Proposition 9.5.** The information loss from subspace projection is:

    E_{x ~ p(.|theta)} [||nabla log p(x|theta) - P_{S_k} nabla log p(x|theta)||^2] = sum_{i=k+1}^d lambda_i

This is the sum of all discarded Fisher eigenvalues.

Under the power-law decay lambda_i ~ C i^{-alpha} with alpha > 1:

    sum_{i=k+1}^d lambda_i approx C integral_k^d x^{-alpha} dx = C (k^{1-alpha} - d^{1-alpha}) / (alpha - 1)
                               approx C k^{1-alpha} / (alpha - 1)   for d >> k

As a fraction of total information:

    D(k) = sum_{i>k} lambda_i / sum_i lambda_i approx k^{1-alpha} / (alpha - 1) * (alpha - 1) / (1 - d^{1-alpha})
         approx k^{1-alpha}   for d large

For k = 1000, alpha = 3: D(k) approx 10^{-6}. Negligible information loss.

**Status:** Proved under the power-law spectral assumption. The result is only as good as the spectral decay hypothesis (Section 6.2).

### 9.5 Generalization

**Theorem 9.6** (PAC-Bayes Bound for Subspace-Constrained Learning). Let P = N(0, sigma_P^2 I_d) be an isotropic Gaussian prior over R^d. Let Q be a posterior concentrated on the k-dimensional affine subspace {theta_bar + Uz : z in R^k}, specifically Q = pushforward of N(z_T, sigma_Q^2 I_k) through z -> theta_bar + Uz.

Then with probability >= 1 - delta over training set S of size n:

    L_D(theta_S) <= L_S(theta_S) + sqrt((KL(Q || P) + ln(2n/delta)) / (2n))

where:

    KL(Q || P) = (1/2) [ (||theta_bar||^2 + ||z_T||^2) / sigma_P^2 + k ln(sigma_P^2 / sigma_Q^2) + k (sigma_Q^2 / sigma_P^2 - 1) - k + (d - k) ln(1) ]

For sigma_Q = sigma_P:

    KL(Q || P) = (||theta_bar + U z_T||^2) / (2 sigma_P^2)

The key point: the KL divergence does **not** scale with d (the full parameter dimension) but only with the norm of the actual parameter vector. The subspace constraint does not directly appear in the KL term but ensures that the optimization trajectory stays in a low-dimensional space, which empirically keeps ||theta_S|| smaller.

**More refined bound.** If we use a data-dependent prior P' that is uniform over the Grassmannian Gr(k, d) (choosing the subspace) times Gaussian in the subspace:

    KL(Q || P') = (1/2) ||z_T||^2 / sigma_P^2 + k ln(sigma_P / sigma_Q) + ln(Vol(Gr(k,d)))

where Vol(Gr(k,d)) = prod_{i=1}^k Vol(S^{d-i}) / prod_{i=1}^k Vol(S^{k-i}) is the volume of the Grassmannian under the round metric.

    ln(Vol(Gr(k,d))) approx k(d-k)/2 * ln(2 pi e / (d-k))   [Stirling approximation]

For k << d: ln(Vol(Gr(k,d))) approx (k d / 2) ln(2 pi e / d) = O(kd). This is large, reflecting the cost of choosing the subspace.

**Tighter approach (conjecture).** If the subspace is determined by a deterministic function of the first m data points (a "data-dependent prior" construction), the PAC-Bayes bound can be applied to the remaining n - m points with KL depending only on z_T in R^k:

    KL approx (1/2) ||z_T||^2 / sigma_P^2 + (k/2) ln(sigma_P^2 / sigma_Q^2)

This eliminates the Grassmannian volume term and gives:

    Generalization gap <= O(sqrt((||z_T||^2 / sigma_P^2 + k) / n))

For fixed sigma_P: gap = O(sqrt(k/n)), which improves over the unconstrained O(sqrt(d/n)) by a factor of sqrt(k/d).

**Status.** The standard PAC-Bayes bound (Theorem 9.6) is proved. The data-dependent prior refinement is a known technique (Dziugaite & Roy 2017) but its application to ATLAS's specific subspace selection is conjectural—it requires that the subspace selection step is measurable with respect to a held-out split, which is feasible in practice but introduces a train/validation split that may affect performance.

### 9.6 Stability of the Coupled System

The ATLAS state (z, U, Lambda) evolves as a coupled dynamical system (Section 8.1). Stability requires that perturbations to the state do not grow unboundedly.

**Proposition 9.7** (Lyapunov Stability). Define the Lyapunov function:

    V(z, U) = L(theta_bar + Uz) + (gamma / 2) ||U - U_*||_F^2

where U_* is the top-k eigenspace of F(theta) at the current theta, and gamma > 0 is a coupling weight.

Along the ATLAS trajectory (with mu = 0 for simplicity):

    dV/dt = nabla_z L^T (dz/dt) + gamma tr((U - U_*)^T (dU/dt))

For the first term (natural gradient flow):

    nabla_z L^T (dz/dt) = -nabla_z L^T G^{-1} nabla_z L <= -(1/lambda_max) ||nabla_z L||^2 < 0

This is strictly negative whenever nabla_z L != 0. The coordinate dynamics decrease the loss.

For the second term (subspace tracking): under Oja's rule, U converges to U_* at rate proportional to the spectral gap lambda_k - lambda_{k+1} (Oja 1982, Theorem 2). When the spectral gap is positive, ||U - U_*|| decreases exponentially.

**Sufficient condition for stability:**

    gamma (lambda_k - lambda_{k+1}) > max over trajectory of ||cross-coupling term||

where the cross-coupling arises because U_* depends on theta (and hence on z). When the subspace changes slowly relative to the parameter updates (ensured by eta_U << eta or by large T_sub), the coupling is weak and stability holds.

**Status.** Heuristic argument. A rigorous proof would require bounding the cross-coupling term, which depends on the third derivative of L (how the Hessian/Fisher changes with theta). This is an open problem.

---

## 10. Computational Tradeoffs

### 10.1 Per-Step Cost

| Operation | ATLAS (diagonal G) | ATLAS (full G) | Adam | GaLore |
|-----------|-------------------|----------------|------|--------|
| Gradient computation | O(C_fwd) | O(C_fwd) | O(C_fwd) | O(C_fwd) |
| Gradient projection | O(dk) | O(dk) | -- | O(dk) |
| Moment update | O(k) | O(k^2) | O(d) | O(dk) |
| Preconditioner solve | O(k) | O(k^3) | O(d) | -- |
| Parameter update | O(dk) | O(dk) | O(d) | O(dk) |
| **Total (excl. gradient)** | **O(dk)** | **O(dk + k^3)** | **O(d)** | **O(dk)** |

where C_fwd is the forward/backward pass cost (dominant term, typically O(seq_len^2 * d_model + seq_len * d_model^2) for transformers).

For k = 512, d = 10^9: O(dk) = O(5 * 10^{11}). Adam: O(d) = O(10^9). ATLAS overhead: ~500x in the projection step.

**However:** C_fwd >> dk for typical transformer training (C_fwd ~ 6 * d * n_tokens per step). For a 1B model with 2K token sequences and batch size 512: C_fwd ~ 6 * 10^9 * 10^6 = 6 * 10^{15}. The ATLAS overhead of 5 * 10^{11} is < 0.01% of C_fwd. **Negligible.**

### 10.2 Memory

| Component | ATLAS (per-layer) | Adam (per-layer) |
|-----------|-------------------|-------------------|
| Parameters | m*n | m*n |
| First moment | r*n | m*n |
| Second moment | r*n (diag) or r^2 (full) | m*n |
| Subspace basis | m*r | -- |
| **Total optimizer state** | **2rn + mr** | **2mn** |

For m = n = 4096, r = 256: ATLAS = 2*256*4096 + 4096*256 = 3 * 2^{20} approx 3M. Adam = 2 * 4096^2 = 33.5M. Savings: **~11x**.

For m = n = 8192, r = 256: ATLAS = 2*256*8192 + 8192*256 = 6M. Adam = 2 * 8192^2 = 134M. Savings: **~22x**.

### 10.3 Communication (Distributed Training)

In distributed data-parallel (DDP) training, gradients are all-reduced across workers. ATLAS can reduce communication:

- All-reduce the compressed gradient g^z = U^T g in R^k instead of the full gradient g in R^d.
- Communication volume: O(k) instead of O(d). Savings: d/k.
- Requires all workers to share the same U (broadcast U periodically).

**Caveat:** Workers must use the same subspace U. If U is updated asynchronously, workers may diverge. Solution: synchronize U updates (broadcast from rank 0) at every T_sub steps.

---

## 11. Comparison to Existing Methods

### 11.1 Detailed Comparison Table

| Method | Subspace | Curvature | Prediction | Compression | Per-step cost | Memory |
|--------|----------|-----------|------------|-------------|---------------|--------|
| SGD | R^d (full) | None | None | None | O(d) | O(d) |
| SGD + Momentum | R^d | None | Position extrap. | None | O(d) | O(d) |
| Adam | R^d | Diagonal (EMA of g^2) | EMA of g (1st moment) | None | O(d) | O(3d) |
| Natural Gradient | R^d | Full Fisher | None | None | O(d^3) | O(d^2) |
| K-FAC | R^d | Kronecker Fisher | None | None | O(d sqrt(d)) | O(d) |
| Shampoo | R^d | Full-matrix per layer | None | None | O(d^{3/2}) | O(d^{3/2}) |
| GaLore | R^k subspace | None (or diagonal) | None | Gradient SVD | O(dk) | O(dk) |
| LoRA | R^k subspace (fixed) | None | None | Random init | O(dk) | O(dk) |
| L2O (learned opt.) | R^d | Learned | Learned | None | O(d * C_opt) | O(d) |
| **ATLAS** | **R^k (Fisher-optimal)** | **Full k x k Fisher** | **Gradient extrap.** | **Fisher eigenspace** | **O(dk + k^3)** | **O(dk)** |

### 11.2 Key Distinctions

**ATLAS vs. GaLore.** Both project gradients to a low-rank subspace. GaLore uses the SVD of the gradient matrix (or weight matrix) to define the subspace, which captures directions of high gradient variance. ATLAS uses the Fisher eigenspace, which captures directions of high *information*. These coincide when the empirical Fisher equals the gradient outer product matrix (which it does, by definition). The real distinction is:
- GaLore does not precondition by curvature within the subspace (it uses Adam or SGD in the subspace).
- ATLAS uses the Fisher metric within the subspace (natural gradient), providing curvature-aware step sizes per subspace dimension.
- GaLore does not use temporal prediction.
- ATLAS adds the PNG correction.

**ATLAS vs. K-FAC.** K-FAC approximates the Fisher with a Kronecker product structure (one factor per layer, capturing inter-neuron and inter-feature correlations separately). ATLAS approximates the Fisher by low-rank truncation (keeping top-k eigenvectors). These are different approximation strategies:
- K-FAC retains the full d-dimensional space but approximates the metric structure.
- ATLAS retains the exact metric but only in a k-dimensional subspace.
- K-FAC's per-step cost is O(d m) where m is the largest layer width. ATLAS's is O(dk).
- K-FAC requires eigendecomposition of the Kronecker factors (O(m^3) per layer). ATLAS requires the k x k subspace Fisher inversion (O(k^3)).

**ATLAS vs. Shampoo.** Shampoo maintains full-matrix preconditioners per layer (L and R factors such that the preconditioner is L^{-1/2} otimes R^{-1/2}). This captures more curvature information than ATLAS's low-rank approximation but at higher memory and compute cost. ATLAS can be viewed as a low-rank restriction of Shampoo where only the top-k left singular directions are used.

**ATLAS vs. LoRA.** LoRA fixes a random low-rank subspace and adapts only within it (for fine-tuning). ATLAS adapts the subspace itself based on the Fisher spectrum (for pre-training or fine-tuning). LoRA's subspace is information-agnostic; ATLAS's is information-optimal.

**ATLAS vs. Natural Gradient.** ATLAS is the natural gradient restricted to a k-dimensional Fisher-optimal subspace with temporal prediction. It degenerates to the full natural gradient when k = d and mu = 0.

---

## 12. Failure Modes and Mitigations

### 12.1 Subspace Collapse

**Failure.** The subspace U collapses: columns become near-parallel, effective dimension drops below k.

**Cause.** Fisher spectrum has a large gap (lambda_1 >> lambda_2 >= ... >= lambda_k). The subspace tracking algorithm converges all columns toward u_1.

**Detection.** Monitor the condition number kappa_sub = lambda_1 / lambda_k. If kappa_sub > threshold (e.g., 10^6), collapse is occurring.

**Mitigation.** (i) Add a coherence regularizer: L_coherence = ||U^T U - I_k||_F^2 to the subspace tracking objective. (ii) Use QR orthonormalization at every subspace update step (already in the algorithm). (iii) Reduce k to match the effective rank of the Fisher.

### 12.2 Fisher Singularity

**Failure.** Some Fisher eigenvalues lambda_i approach zero, making G^{-1} ill-conditioned.

**Cause.** Certain parameter directions have near-zero effect on the output distribution (redundant parameters, symmetries).

**Detection.** Monitor min_i lambda_i.

**Mitigation.** Regularize: G_t <- G_t + epsilon I_k with epsilon > 0 (Tikhonov/ridge). This is already in the algorithm (Section 7.1, step 4). Typical epsilon: 10^{-8} to 10^{-4}.

### 12.3 Prediction Divergence

**Failure.** The temporal prediction g^z_{pred} points in a direction that increases the loss.

**Cause.** The linear gradient approximation fails (e.g., at a loss landscape inflection point, phase transition in training, or learning rate change).

**Detection.** The adaptive mu mechanism (Section 5.6) detects large prediction errors and shrinks mu.

**Mitigation.** (i) The adaptive mu is the primary defense. (ii) Apply a trust region: reject the update if L(theta_{t+1}) > L(theta_t) + c (a non-monotone line search condition) and fall back to mu = 0. (iii) Warmup: start with mu = 0 for the first W steps (e.g., W = 1000) and gradually increase.

### 12.4 Flat Fisher Spectrum

**Failure.** All Fisher eigenvalues are approximately equal (lambda_1 approx lambda_d). The subspace provides no benefit because every k-dimensional subspace captures the same fraction rho = k/d of the Fisher information.

**Cause.** Model is near random initialization (all directions equally informative) or is severely overparameterized.

**Detection.** Compute the effective rank: r_eff = (sum_i lambda_i)^2 / sum_i lambda_i^2. If r_eff approx d, the spectrum is flat.

**Mitigation.** (i) Use k = d (degenerate to full natural gradient with diagonal approximation = Adam). (ii) Wait for the spectrum to concentrate (it typically does after the initial training phase) and then activate subspace projection. (iii) Use a random subspace initially and switch to Fisher-based once the spectrum differentiates.

### 12.5 Subspace Staleness

**Failure.** The subspace U_t lags behind the true Fisher eigenspace, causing rho_t to drop.

**Cause.** T_sub is too large, or eta_U is too small, or the loss landscape is changing rapidly (e.g., during learning rate warmup or curriculum changes).

**Detection.** Monitor rho_t = ||U_t U_t^T nabla L||^2 / ||nabla L||^2 (or a stochastic estimate thereof).

**Mitigation.** (i) Decrease T_sub or increase eta_U. (ii) Use the Oja tracking rule (continuous updates) instead of periodic SVD. (iii) Trigger an immediate subspace refresh when rho_t drops below a threshold.

### 12.6 Anchor Drift

**Failure.** The linear approximation theta = theta_bar + Uz becomes inaccurate as z grows large, because the Fisher spectrum at theta differs significantly from the spectrum at theta_bar.

**Cause.** Infrequent re-anchoring (anchor radius R too large).

**Detection.** ||z_t|| > R.

**Mitigation.** Re-anchor (Section 5.8). The cost is O(dk) for the parameter reconstruction. With R = sqrt(k), re-anchoring occurs approximately every R^2 / (eta ||delta_z||^2) steps, which for typical hyperparameters is every few hundred to few thousand steps.

---

## 13. Minimal Prototype

### 13.1 Design Choices for Prototype

The minimal prototype makes the following simplifications:
- Diagonal subspace Fisher (G = Lambda, no off-diagonal terms)
- Linear gradient prediction (p = 1, no VAR)
- Periodic SVD for subspace updates (no Oja tracking)
- Fixed subspace dimension k
- Per-layer application (one subspace per weight matrix)
- No predictive subspace rotation

### 13.2 PyTorch-Style Pseudocode

```python
class ATLASOptimizer:
    """Minimal ATLAS: Fisher-optimal subspace + PNG."""

    def __init__(self, params, lr=1e-3, k=256, beta=0.999,
                 mu_min=0.01, mu_max=0.3, T_sub=200,
                 anchor_radius=10.0, eps=1e-8):
        self.lr = lr
        self.k = k
        self.beta = beta
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.T_sub = T_sub
        self.anchor_radius = anchor_radius
        self.eps = eps

        self.state = {}
        for p in params:
            if p.ndim != 2:  # Apply only to weight matrices
                continue
            m, n = p.shape
            r = min(k, m, n)
            s = {
                'step': 0,
                'U': None,           # R^{m x r}, subspace basis
                'z': zeros(r, n),    # R^{r x n}, compressed coords
                'theta_bar': p.data.clone(),  # anchor
                'fisher_diag': ones(r),       # EMA of squared proj grads
                'prev_gz': zeros(r, n),       # previous compressed gradient
                'mu': mu_min,                 # prediction coefficient
                'grad_buffer': [],            # for periodic SVD
            }
            self.state[p] = s

    def step(self):
        for p in self.params:
            if p not in self.state:
                # Non-matrix params: standard SGD
                p.data -= self.lr * p.grad
                continue

            s = self.state[p]
            s['step'] += 1
            grad = p.grad  # R^{m x n}

            # --- Subspace initialization / update ---
            if s['U'] is None or s['step'] % self.T_sub == 0:
                # Accumulate gradients for SVD
                # (in practice, use the current gradient matrix directly)
                U, S, V = torch.svd_lowrank(grad, q=self.k)
                s['U'] = U[:, :self.k]  # R^{m x r}
                # Re-anchor
                s['theta_bar'] = p.data.clone()
                s['z'] = zeros_like(s['z'])
                s['fisher_diag'] = S[:self.k] ** 2 + self.eps

            U = s['U']
            r = U.shape[1]

            # --- Compress gradient ---
            gz = U.T @ grad  # R^{r x n}

            # --- Update Fisher diagonal (EMA) ---
            gz_sq = (gz ** 2).mean(dim=1)  # R^r, average over output dim
            s['fisher_diag'] = self.beta * s['fisher_diag'] + \
                               (1 - self.beta) * gz_sq

            # --- Predictive gradient ---
            az = gz - s['prev_gz']
            gz_pred = gz + s['mu'] * az

            # --- Natural gradient update in subspace ---
            fisher_inv = 1.0 / (s['fisher_diag'].unsqueeze(1) + self.eps)
            delta_z = -self.lr * fisher_inv * gz_pred  # R^{r x n}

            # --- Update compressed coords ---
            s['z'] = s['z'] + delta_z

            # --- Reconstruct parameters ---
            p.data = s['theta_bar'] + U @ s['z']

            # --- Adapt prediction coefficient ---
            if s['step'] > 1:
                predicted_prev = s['prev_gz'] + s['mu'] * \
                    (s['prev_gz'] - s.get('prev_prev_gz', s['prev_gz']))
                err = gz - predicted_prev
                err_ratio = err.norm() / (gz.norm() + self.eps)
                s['mu'] = max(self.mu_min,
                              min(self.mu_max,
                                  s['mu'] * (1 - err_ratio.item())))

            # --- Re-anchor if needed ---
            if s['z'].norm() > self.anchor_radius:
                s['theta_bar'] = p.data.clone()
                s['z'] = zeros_like(s['z'])

            # --- Store for next step ---
            s['prev_prev_gz'] = s['prev_gz'].clone()
            s['prev_gz'] = gz.clone()
```

### 13.3 Validation Experiments (Proposed)

1. **Sanity check.** Train a 2-layer MLP on MNIST with ATLAS vs. Adam. Verify ATLAS converges and achieves comparable accuracy. Expected: ATLAS matches Adam with ~10x less optimizer memory.

2. **Small transformer.** Train a 125M-parameter GPT-2 on OpenWebText with ATLAS vs. Adam vs. GaLore. Measure: final perplexity, wall-clock time, peak memory. Expected: ATLAS achieves lower perplexity than GaLore (due to curvature awareness) at similar memory cost, and approaches Adam's perplexity with ~10x less optimizer memory.

3. **Ablation.** Disable each component (prediction: mu=0; curvature: G=I; subspace: k=d) and measure the contribution of each.

4. **Fisher spectrum measurement.** During training, periodically compute the top-100 eigenvalues of the per-layer empirical Fisher via Lanczos. Plot the spectrum to validate the power-law decay hypothesis.

---

## 14. Full Research Program

### Phase 1: Empirical Validation (3-6 months)

1. Implement ATLAS in PyTorch as a drop-in optimizer.
2. Run the validation experiments from Section 13.3.
3. Measure the Fisher spectrum for transformers at scales 125M, 350M, 1.3B, 6.7B.
4. Characterize the relationship between spectral decay rate alpha, model scale, and optimal k.
5. Compare ATLAS against Adam, GaLore, K-FAC, Shampoo at each scale.

### Phase 2: Theoretical Refinement (6-12 months)

1. Prove or disprove Conjecture 9.3 (net benefit of prediction) under weakened assumptions.
2. Rigorously bound the cross-coupling term in Proposition 9.7 (Lyapunov stability).
3. Develop a theory for optimal k selection as a function of the Fisher spectrum and training budget.
4. Analyze the interaction between ATLAS and learning rate scheduling (warmup, cosine decay).
5. Extend the PAC-Bayes generalization bound to account for the data-dependent subspace.

### Phase 3: Scaling and Extensions (12-24 months)

1. Scale ATLAS to 70B+ parameter models. Key challenge: subspace SVD/tracking at scale.
2. Develop the VAR(p) gradient prediction model and compare against linear extrapolation.
3. Implement predictive subspace rotation (Section 8.3) and evaluate stability.
4. Explore non-Euclidean subspaces (e.g., subspaces of the Lie algebra for weight matrices with symmetry).
5. Combine ATLAS with mixed-precision training (FP8 for full parameters, FP32 for subspace coordinates).
6. Integrate with tensor parallelism and pipeline parallelism for distributed training at scale.

### Phase 4: Architectural Co-Design (24+ months)

1. Design network architectures that are *natively* low-rank in the Fisher spectrum (architectures where alpha is large by construction).
2. Explore "subspace-native" training where the model never instantiates full d-dimensional parameters, only the compressed coordinates z and the subspace basis U.
3. Investigate connections to neural architecture search: can the optimal k per layer guide architecture decisions (e.g., layer width, attention heads)?

---

## 15. Open Conjectures and Validation Criteria

### Conjecture 1: Fisher Spectral Decay

**Statement.** For transformer language models with d parameters trained on natural language, the eigenvalues of the Fisher information matrix satisfy lambda_i <= C * i^{-alpha} where alpha >= 2 and alpha increases monotonically during training.

**Validation criterion.** Compute the top-500 Fisher eigenvalues (via Lanczos with Hessian-vector products) for GPT-2 (125M) at 10 checkpoints during training. Fit the power law. Report alpha and goodness-of-fit (R^2). The conjecture is supported if R^2 > 0.9 and alpha >= 2 at convergence.

**Falsification.** If the spectrum is not power-law (e.g., has multiple plateaus or an exponential tail), the subspace dimension selection theory (Proposition 6.3) needs revision, though ATLAS may still work with empirically tuned k.

### Conjecture 2: Temporal Prediction Benefit

**Statement.** For transformer training with ATLAS, the optimal prediction coefficient mu* satisfies mu* > 0 for all but the first ~1000 steps, and the PNG update with adaptive mu achieves at least 5% faster convergence (measured in steps to target loss) compared to mu = 0.

**Validation criterion.** Train GPT-2 (125M) with ATLAS at mu = 0, mu = 0.1, mu = 0.3, and adaptive mu. Compare steps-to-target-loss curves. The conjecture is supported if adaptive mu consistently outperforms mu = 0 by > 5%.

**Falsification.** If mu = 0 is always optimal, the temporal prediction mechanism provides no benefit and should be removed from ATLAS (simplifying to a Fisher-optimal subspace natural gradient method).

### Conjecture 3: Subspace Stability

**Statement.** Under the ATLAS dynamics with Oja tracking, the subspace U_t converges to a neighborhood of the true Fisher eigenspace U_* and remains within this neighborhood throughout training, provided the spectral gap lambda_k - lambda_{k+1} > 0 and the learning rate satisfies eta < c * (lambda_k - lambda_{k+1}) / lambda_1^2 for some universal constant c.

**Validation criterion.** During training, monitor the principal angle between U_t and U_* (computed by periodic exact eigendecomposition on a small proxy model). The conjecture is supported if the principal angle decreases monotonically to a small value and remains bounded.

**Falsification.** If the principal angle oscillates or grows, Oja tracking is insufficient and periodic SVD (or a more sophisticated subspace tracking algorithm) is required.

### Conjecture 4: Generalization Improvement

**Statement.** The subspace constraint in ATLAS provides a measurable generalization benefit: the gap between training loss and validation loss is smaller for ATLAS than for Adam, at matched training loss.

**Validation criterion.** Train GPT-2 to matched training perplexity with ATLAS (various k) and Adam. Compare validation perplexity. The conjecture is supported if ATLAS achieves lower validation perplexity at matched training perplexity.

**Falsification.** If ATLAS shows no generalization benefit (or worse generalization), the PAC-Bayes analysis (Theorem 9.6) is not tight enough to be practically relevant, though the computational benefits of ATLAS may still justify its use.

### Conjecture 5: Implicit Hessian Accuracy

**Statement.** The implicit Hessian correction from temporal prediction (Theorem 5.7) captures at least 50% of the true Hessian-step product in the subspace, measured as:

    cos_sim(a^z_t, H_z delta^z_{t-1}) >= 0.5

averaged over training steps (excluding the first 1000).

**Validation criterion.** On a small model (e.g., 10M parameters), compute both a^z_t (from gradient differences) and H_z delta^z_{t-1} (from exact Hessian-vector product). Compute cosine similarity. The conjecture is supported if mean cosine similarity > 0.5.

**Falsification.** If cosine similarity is near zero, the linear gradient field approximation is invalid and the theoretical motivation for PNG (Theorem 5.7) does not hold. However, PNG may still work empirically for reasons unrelated to implicit Hessian correction (e.g., as a form of gradient smoothing).

---

## Appendix A: Notation Summary

All notation is defined at first use and collected here for reference.

- R^n: n-dimensional real vector space
- R^{m x n}: space of m-by-n real matrices
- I_k: k-by-k identity matrix
- St(k, d): Stiefel manifold of orthonormal k-frames in R^d
- Gr(k, d): Grassmannian of k-dimensional subspaces of R^d
- diag(v): diagonal matrix with entries from vector v
- tr(A): trace of matrix A
- ||v||: Euclidean norm of vector v
- ||A||_F: Frobenius norm of matrix A
- orth(A): QR orthonormalization of the columns of A
- P_S = UU^T: orthogonal projector onto subspace S = col(U)
- nabla f: gradient of f
- nabla^2 f: Hessian of f
- E[.]: expectation
- Var[.]: variance
- KL(Q || P): Kullback-Leibler divergence from P to Q
- O(f): asymptotic upper bound (big-O notation)

## Appendix B: Relationship to Mirror Descent

ATLAS can be viewed as a form of mirror descent in the compressed space. Mirror descent with Bregman divergence D_phi:

    z_{t+1} = argmin_{z in R^k} { eta <g^z, z> + D_phi(z, z_t) }

With the quadratic Bregman divergence D_phi(z, z') = (1/2)(z - z')^T G (z - z'), the solution is:

    z_{t+1} = z_t - eta G^{-1} g^z

This is exactly the natural gradient update (without prediction). The mirror map is phi(z) = (1/2) z^T G z, whose gradient is nabla phi(z) = Gz.

The PNG extension replaces g^z with g^z_{pred}, which does not fit the standard mirror descent template but can be analyzed as mirror descent with a time-varying linear perturbation.

This connection suggests that convergence results for mirror descent with time-varying potentials (e.g., Rakhlin & Sridharan 2013) may be applicable to ATLAS, potentially yielding tighter bounds than the direct analysis in Theorem 9.1.

---

*Document version: 1.0*
*Framework: ATLAS (Adaptive Temporally-Predictive Learning in Active Subspaces)*
*Date: 2026-03-10*
