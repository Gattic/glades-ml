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
- **Projection-consistent covariance closure.** `sigma2` is no longer estimated on a separate accumulated-gradient scale; the optimizer now tracks the normalized covariance trace `tr((1/n) H H^T)` on the same update scale as the subspace Fisher statistics and derives the complement scalar by trace closure.
- **Residual complement block prototype.** `complementRank` now controls a dense low-rank residual block `V R V^T` instead of a single scalar direction. `complementRank=1` reproduces the earlier sector path; `complementRank>1` enables a denser FC-style residual closure.
- **FC-gated adaptive residual rank.** Tagged hidden FC-style layers now treat `complementRank` as a cap, keep a runtime `activeComplementRank`, and promote/demote residual modes only at ATLAS control boundaries. Untagged/unit-test calls keep the fixed-block semantics so the core math stays directly testable.
- **Scout-driven birth criterion.** Residual-rank birth no longer waits for the dense-block EMA alone. It now uses the current residual-block sample as a scout and scores birth with a Kelly-style edge fraction relative to the isotropic tail, while deaths remain EMA-based.
- **Kelly-style probationary acceptance.** A born residual mode is no longer promoted immediately. Stable low-uncertainty modes can still graduate into the active complement block, but fresh scout-only modes now remain on probation until their excess-return estimate clears a Kelly-style risk charge based on sample/EMA disagreement and directional alignment.
- **Transported `q=2` scout quality filter.** Birth proposals now come from a two-mode scout subspace rather than a single raw sample eigenvector. The controller projects that scout subspace onto the next transported EMA residual mode, penalizes overlap with already-retained complement directions, and only hands the resulting projected support to the Kelly probation gate.
- **Generalized quotient scout geometry.** On the CPU ATLAS control-boundary path (`tSub>0`), residual proposals now use a separate scout basis on the quotient complement and score directions with a generalized eigenvalue against tail, innovation, and retained-block contamination penalties. The older `tSub=0` controller path remains as a focused fallback for the math tests.
- **Complement-specific damping controls.** The residual block uses its own nominal lr scale and `kappaMax` cap instead of inheriting the more aggressive active-space settings. The current default keeps the complement path conservative while the richer closure is still benchmarked.

Observed outcomes on April 8, 2026:
- `./glades-unit-tests atlas` passes with the redesign enabled.
- `./glades-unit-tests atlas-controller` now passes, covering four focused controller cases: stable modes promote after one probation window, fresh scout spikes stay on probation, rotated two-mode scout subspaces remain promotable, and misaligned scout directions are rejected outright.
- The aggressive adaptive-rank setting improved throughput slightly but hurt MNIST test accuracy; it is therefore no longer the default.
- The trace-closed baseline path with `complementRank=0` reached `train=8.06s`, `testAcc=98.14%` on `./glades-unit-tests atlas-bench --mode standard --repeats 1 --atlas-complement-rank 0`.
- The one-sector residual closure with `complementRank=1` reached `train=8.00s`, `testAcc=97.72%` on `./glades-unit-tests atlas-bench --mode standard --repeats 1 --atlas-complement-rank 1`.
- After retuning the one-sector path to `complementLrScale=0.25` and `complementKappaMax=0.5`, the same benchmark reached `train=8.17s`, `testAcc=97.86%`. This is better than the undamped one-sector path but still below the `complementRank=0` baseline.
- The new dense residual block with `complementRank=4` reached `train=7.98s`, `testAcc=98.10%`. It captures more FC complement trace than the scalar path and is slightly faster than the isotropic baseline, but it still loses a small amount of out-of-sample accuracy.
- The first FC-gated adaptive residual-rank path with `complementRank=4` reached `train=8.21s`, `testAcc=97.82%` on the same benchmark. By step 200 the hidden FC block was still parked at `complement_active_rank=0`, so that controller was conservative enough to avoid over-correction but not strong enough to recover the dense block’s lost accuracy.
- After switching births to the scout-driven Kelly-style criterion, the same benchmark still reached only `train=8.28s`, `testAcc=97.82%`. The hidden FC layer now promoted to `complement_active_rank=1` at step 200 with `birth_scout≈5.62` and `birth_kelly≈0.73`, but the extra modeled residual trace (`sector_trace_capture≈0.0099`) still did not improve out-of-sample accuracy.
- After adding the Kelly-style probationary gate, the standard benchmark with `complementRank=4` still reached only `train=8.40s`, `testAcc=97.82%`, versus the `complementRank=0` baseline at `train=8.39s`, `testAcc=98.14%`. At the hidden FC layer’s step-200 control boundary the candidate residual mode stayed inactive (`complement_active_rank=0`) because the new quality metrics were explicitly negative (`complement_trial_alignment≈0.021`, `complement_trial_return≈-0.172`).
- After replacing the single scout vector with the transported `q=2` scout filter, the standard benchmark still reached only `train=8.21s`, `testAcc=97.82%`, versus the refreshed `complementRank=0` baseline at `train=8.27s`, `testAcc=98.14%`. The hidden FC layer’s step-200 controller state was materially cleaner because the scout now measured projected support and retained-block contamination separately, but the candidate residual mode still stayed inactive and did not recover the baseline accuracy.
- After upgrading the real control-boundary path to the generalized quotient scout, the focused `atlas-controller` suite still passed: stable modes promoted after probation, rotated scout subspaces remained promotable, and misaligned scout directions stayed rejected. This confirmed the scout geometry change did not regress the controller invariants, even though the benchmark still needed to decide whether the stronger scout was enough to justify residual activation on MNIST.
- On April 9, 2026, the CPU generalized-quotient scout path with `complementRank=4` reached `train=8.30s`, `testAcc=97.94%` on the standard MNIST benchmark, versus the same-day `complementRank=0` baseline at `train=8.25s`, `testAcc=98.10%`. The hidden FC controller signal was materially cleaner: `complement_scout_lambda≈1.79e-4`, `complement_trial_alignment≈0.132`, `complement_trial_return≈-0.017`, and the residual mode stayed inactive instead of being over-promoted.
- On the hidden FC layer at step 200 of the standard benchmark, the sector-specific damping reduced the logged `sector_rate` from about `0.1609` in the undamped path to `0.0100`, confirming that the retune directly addressed the residual-block over-correction mechanism.
- On the hidden FC layer at step 200 with `complementRank=4`, the dense block captured about `0.7%` of total trace beyond the active subspace (`sector_trace_capture≈0.007`), but the active-plus-block model still left a very large closure gap (`closure_gap≈161.9`), so the isotropic tail remains the dominant modeled mass.

Interpretation:
- The data supports **stability-first subspace tracking** and **opt-in adaptive compression**, not unconditional online rank collapse.
- The scale-consistency issue between `sigma2` and the subspace Fisher statistics is addressed in the implementation by sharing one normalized covariance model.
- A richer residual closure alone still does **not** beat the `complementRank=0` MNIST baseline. The dense block is a cleaner structural test than the one-sector path, but both the fixed and adaptive `complementRank=4` results remain accuracy-negative.
- The scout-driven controller fixes the original “no birth” defect, but MNIST still does not reward the added residual freedom. The problem is no longer lack of activation; it is that a rank-1 residual birth still leaves the closure gap overwhelmingly dominated by the isotropic tail.
- The Kelly-style probationary gate fixes the opposite defect: ATLAS now correctly refuses the low-alignment FC scout direction that previously activated at step 200. That rejection improves controller discipline, but it still does not recover the isotropic-baseline accuracy, so the remaining issue is not acceptance logic alone.
- The transported `q=2` scout improves direction-quality measurement, but not enough to change the MNIST decision boundary: the hidden FC scout is now described more faithfully, yet the benchmark still prefers `complementRank=0`.
- The generalized quotient scout is a better geometric probe than the old raw scout, but the benchmark must still decide whether the hidden FC complement contains a persistent super-tail mode worth activating. If the generalized eigenvalue path still clusters near the tail, complement modeling is not the right lever for this MNIST setting.
- The April 9, 2026 benchmark now points the same way: the generalized scout reduced false activation pressure, but it still left the hidden FC residual mode below the acceptance bar and did not beat the isotropic baseline. That makes the next research step a retained-mode keep/drop rule or a different benchmark, not another looser birth threshold.
- The next likely gains are from stronger scout geometry or a validation-free keep/drop rule on retained residual modes, not from further scalar damping alone.

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

## Implementation Note: PRISM Prototype (April 9, 2026)

An opt-in PRISM prototype was implemented in the CPU ATLAS path with two concrete mechanisms:

- an active-space memory correction using up to two lags of compressed-gradient history,
- a predictive-edge gate that suppresses adaptive complement activation when the residual complement has subcritical lagged edge.

Focused controller coverage now includes:

- a PRISM predictive-edge gate regression that confirms stale complement modes stay rejected,
- a PRISM active-memory regression that confirms the active correction energy shrinks on aligned histories,
- checkpoint manifest/state coverage for the new PRISM config and history state.

Standard MNIST benchmark results on **April 9, 2026**:

- `cRank=0`, `prism=0`: `train=8.27s`, `testAcc=97.84%`
- `cRank=0`, `prism=1`: `train=8.34s`, `testAcc=97.58%`
- `cRank=4`, `prism=1`: `train=8.30s`, `testAcc=97.74%`

Interpretation:

- The PRISM gate behaves as intended: on the FC hidden layer, the residual predictive edge remains subcritical rather than producing a useful retained complement mode.
- The active-memory correction is measurable in diagnostics, but on this benchmark it degrades out-of-sample accuracy instead of improving it.
- So the PRISM hypothesis is informative but not yet beneficial on MNIST: the benchmark still does not justify explicit complement modeling, and the current memory-only correction is not a default-worthy improvement.

The most defensible conclusion from this prototype is that the residual bulk on this benchmark is not just poorly modeled geometry; it is also not yielding enough short-horizon predictive utility to pay for the extra correction.

### RESOLVE Prototype Update

Minimal RESOLVE was implemented on **April 9, 2026** as:

- a transfer-edge gate on the complement controller using lagged active-history cross structure,
- a stable scalar lag-kernel fit on the active compressed gradients,
- checkpoint-persisted lag history and benchmark/config plumbing.

Focused validation:

- `atlas-controller` passes, including RESOLVE gate and active-memory tests.
- `atlas` passes after the RESOLVE config/state persistence updates.

Standard MNIST benchmark results on **April 9, 2026**:

- `cRank=0`, `resolve=0`: `train=8.42s`, `testAcc=97.84%`
- `cRank=0`, `resolve=1`: `train=8.65s`, `testAcc=97.68%`
- `cRank=4`, `resolve=1`: `train=8.68s`, `testAcc=97.70%`

Interpretation:

- RESOLVE is measurable in both diagnostics and runtime, but it does not improve the benchmark.
- On the hidden FC block, the transfer-edge gate can report large edge scores while the retained residual mode still fails to deliver better out-of-sample behavior.
- That is further evidence that the benchmark problem is not a missing threshold or missing lag term. The compressed statistics are still not isolating a complement mode with robust predictive utility.

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

## Appendix C: HERO Prototype Note

On April 9, 2026, a minimal CPU-only HERO prototype was added on top of the existing RESOLVE path:

- explicit complement activation is now gated by a whitened Hankel-style edge score built from the current active compressed gradient and lagged scout-history slices,
- the HERO memory fallback reuses the stable active-space pole fit already used by RESOLVE, but applies the separate `heroMemoryScale`,
- lagged scout history is transported through scout-basis refresh overlap so the gate remains basis-consistent across control boundaries.

The first benchmark result did not justify a default change on standard MNIST:

- baseline `--atlas-complement-rank 0` remained stronger than the HERO-enabled paths,
- the hidden FC residual stayed subcritical under the HERO gate,
- the memory-only fallback also failed to recover the isotropic baseline.

Interpretation: the benchmark still supports the broader research conclusion that the unresolved FC complement behaves more like weak predictive bulk than a stable explicit geometric mode family.

## Appendix D: COBALT Prototype Note

On April 9, 2026, a minimal CPU-only COBALT prototype was added as a transfer-weighted variant of the existing active-memory fallback:

- explicit complement activation is now gated by a stacked active-plus-scout transfer-edge score computed from the lagged `resolveGzHistory` and transported scout history,
- the active-space memory fallback is scaled by the same transfer strength estimate (`cobaltSigma`) rather than a fixed residual-edge heuristic,
- the implementation reuses the existing ATLAS compressed history buffers, config plumbing, benchmark flags, and checkpoint config persistence.

Focused validation:

- `atlas-controller` passes, including COBALT transfer-gate and active-memory tests.
- `atlas` passes after the COBALT config and controller changes.

Standard MNIST benchmark results on **April 9, 2026**:

- `cRank=0`, `cobalt=0`: `train=8.21s`, `testAcc=97.84%`
- `cRank=0`, `cobalt=1`: `train=10.04s`, `testAcc=97.62%`
- `cRank=4`, `cobalt=1`: `train=10.44s`, `testAcc=97.64%`

Interpretation:

- COBALT is measurable and the controller diagnostics are coherent, but it is materially slower than the isotropic baseline on this benchmark.
- The hidden FC block can show positive transfer-edge scores while still failing to deliver a useful optimizer intervention.
- That reinforces the same research conclusion reached by PRISM, RESOLVE, and HERO: on this benchmark, the unresolved FC complement still behaves more like broad weak bulk than a stable, optimizer-useful explicit mode family.

## Appendix E: BIRCH Prototype Note

On April 9, 2026, a minimal CPU-only BIRCH prototype was added as a memory-only Hankel-transfer fallback:

- it computes a local whitened Hankel-style transfer score from stacked active-history slices plus optional transported scout-history slices,
- it exposes that score through `birch_edge` and `birch_sigma`,
- it keeps explicit complement activation disabled in the minimal prototype and only uses the supercritical part of the Hankel score to scale an active-space memory kernel.

Focused validation:

- `atlas-controller` passes after the BIRCH integration.
- `atlas` passes, including the new BIRCH controller regressions and the expanded checkpoint-config coverage.

Standard MNIST benchmark results on **April 9, 2026**:

- `cRank=0`, `birch=0`: `train=8.36s`, `testAcc=97.84%`
- `cRank=0`, `birch=1`: `train=9.69s`, `testAcc=97.68%`
- `cRank=4`, `birch=1`: `train=10.00s`, `testAcc=97.66%`

Interpretation:

- BIRCH produces coherent transfer diagnostics and a measurable memory fallback, but it is slower than the isotropic baseline on this benchmark.
- The hidden FC block can show supercritical local Hankel scores while still failing to improve out-of-sample behavior.
- That strengthens the overall ATLAS conclusion: on this benchmark, even a more structured transfer-oriented summary still does not isolate a residual component worth paying to model.

## Appendix F: GHOST Prototype Note

On April 9, 2026, a minimal CPU-only GHOST prototype was added as a quotient-horizontal, memory-only transfer fallback:

- it computes an approximate gauge-horizontal projection in the compressed active/scout state by removing the component aligned with the current compressed weight image,
- it extracts a rank-1 left/right transfer mode from the whitened cross-covariance between the current horizontal active state and lagged horizontal active/scout history,
- it exposes that mode through `ghost_edge`, `ghost_sigma`, and `ghost_horizontal_ratio`,
- it keeps explicit complement activation disabled in the minimal prototype and uses the retained transfer mode only to shape an active-space memory correction.

Focused validation:

- `atlas-controller` passes after the GHOST integration.
- `atlas` passes, including the new GHOST controller regressions and the expanded checkpoint-config coverage.

Standard MNIST benchmark results on **April 9, 2026**:

- `cRank=0`, `ghost=0`: `train=8.26s`, `testAcc=97.84%`
- `cRank=0`, `ghost=1`: `train=9.89s`, `testAcc=97.94%`
- `cRank=4`, `ghost=1`: `train=10.53s`, `testAcc=97.72%`

Interpretation:

- GHOST is the first ATLAS-side prototype that explicitly mixes approximate quotienting with a biorthogonal transfer mode rather than another PSD complement block.
- The prototype is intentionally local and conservative: it uses a single approximate gauge direction and a rank-1 memory correction, not a full quotient-balanced semigroup model.
- On this benchmark, the memory-only `cRank=0` path is the first recent ATLAS-side transfer prototype to beat the same-day isotropic baseline in a single run, but the gain is small (`97.94%` vs `97.84%`) and comes with a large throughput penalty (`9.89s` vs `8.26s`).
- Re-enabling complement capacity under GHOST still makes the result worse (`cRank=4`, `testAcc=97.72%`), so the old conclusion remains intact: richer explicit complement modeling is still not paying for itself on standard MNIST.
- As with HERO, COBALT, and BIRCH, the standard MNIST benchmark remains the falsifier: if the hidden FC residual still fails to clear a useful post-quotient transfer edge, then the remaining ATLAS improvement path on this workload is unlikely to come from richer complement modeling.

## Appendix G: Late-Prototype Benchmark Ledger (April 9, 2026)

For the late April 9 prototypes, the relevant comparison is each prototype against its same-day isotropic `complementRank=0` anchor on the branch snapshot where it was run. The exact baseline timing shifts slightly between snapshots, but the qualitative ranking is stable.

Recorded standard-MNIST outcomes:

- **PRISM**: `cRank=0`, `prism=1` -> `train=8.34s`, `testAcc=97.58%`; `cRank=4`, `prism=1` -> `train=8.30s`, `testAcc=97.74%`.
- **RESOLVE**: `cRank=0`, `resolve=1` -> `train=8.65s`, `testAcc=97.68%`; `cRank=4`, `resolve=1` -> `train=8.68s`, `testAcc=97.70%`.
- **HERO**: `cRank=0`, `hero=1` -> `train=8.51s`, `testAcc=97.68%`; `cRank=4`, `hero=1` -> `train=8.53s`, `testAcc=97.70%`.
- **COBALT**: `cRank=0`, `cobalt=1` -> `train=10.04s`, `testAcc=97.62%`; `cRank=4`, `cobalt=1` -> `train=10.44s`, `testAcc=97.64%`.
- **BIRCH**: `cRank=0`, `birch=1` -> `train=9.69s`, `testAcc=97.68%`; `cRank=4`, `birch=1` -> `train=10.00s`, `testAcc=97.66%`.
- **GHOST**: `cRank=0`, `ghost=1` -> `train=9.89s`, `testAcc=97.94%`; `cRank=4`, `ghost=1` -> `train=10.53s`, `testAcc=97.72%`.
- **SPARROW**: `cRank=0`, `sparrow=1` -> `train=8.75s`, `testAcc=97.76%`; `cRank=2`, `sparrow=1` -> `train=8.76s`, `testAcc=97.88%`; `cRank=4`, `sparrow=1` -> `train=8.86s`, `testAcc=98.00%`.
- **ORBIT-Lite**: stabilized CPU last-layer prototype, `cRank=0`, `orbit=1` -> `train=8.25s`, `testAcc=97.36%`; `cRank=4`, `orbit=1` -> `train=8.36s`, `testAcc=97.40%`.
- **QBRT**: `cRank=0`, `qbrt=1` -> `train=10.00s`, `testAcc=97.78%`; `cRank=4`, `qbrt=1` -> `train=9.97s`, `testAcc=97.78%`.
- **QRC**: `cRank=0`, `qrc=1` -> `train=9.90s`, `testAcc=97.84%`; `cRank=4`, `qrc=1` -> `train=10.05s`, `testAcc=97.80%`.
- **RIFT**: `cRank=0`, `rift=1` -> `train=17.80s`, `testAcc=97.80%`; `cRank=4`, `rift=1` -> `train=17.93s`, `testAcc=97.84%`.

Cross-run synthesis:

- These late prototypes consistently improved controller discipline and made the diagnostics more interpretable, but they almost never justified explicit complement activation on standard MNIST.
- With the lone exception of SPARROW, `cRank>0` remained accuracy-negative across the late transfer/memory prototypes listed above.
- GHOST was the first late prototype to produce a small same-day single-run accuracy lift over its isotropic anchor, but it paid a large runtime cost and did not overturn the broader conclusion.
- SPARROW kept the GHOST lesson while removing most of that transfer-model tax. Its memory-only `cRank=0` path was still accuracy-negative, but with larger scout capacity it recovered part of the lost accuracy: `cRank=2` reached `97.88%` and `cRank=4` reached `98.00%` while staying materially faster than GHOST (`8.76-8.86s` vs `9.89-10.53s`).
- ORBIT-Lite kept the quotient-focused lesson but moved the memory path to the output head. After stabilizing the edge score and making the correction truly memory-only, it avoided the earlier output-head blow-up, but both `cRank=0` and `cRank=4` remained below the same-day isotropic baseline (`97.36-97.40%` vs `97.84%`).
- QBRT pushed the same transfer-first lesson toward a small quotient-balanced ARX-style controller. It produced clear bounded transfer diagnostics, but on standard MNIST both the memory-only and `cRank=4` paths landed at `97.78%` while costing about `9.97-10.00s`, so the extra balanced-transfer machinery did not pay for itself on this workload either.
- QRC distilled the same transfer-memory lesson into a tiny reduced robust controller on quotient-active coordinates. That made the controller logic cleaner, but it did not improve the tradeoff: the memory-only `cRank=0` path only matched the same-day isotropic accuracy (`97.84%`) while slowing to `9.90s`, and `cRank=4` slipped back to `97.80%` at `10.05s`.
- RIFT tested whether the remaining signal was pathwise rather than purely transfer-linear by adding a second-level signature-style observer on quotient-horizontal active/scout histories. On standard MNIST that richer path statistic did not preserve the SPARROW gain: `cRank=0` fell to `97.80%` and `cRank=4` only recovered to `97.84%`, while both runs roughly doubled training time (`17.80-17.93s`).
- Even so, SPARROW did not beat the same-day isotropic baseline of `97.84%` decisively enough, or cheaply enough, to justify a default change. The best SPARROW point (`cRank=4`) is still slower than the plain isotropic path (`8.86s` vs `8.47s`) and still trails the stronger April 8 reference snapshot.
- A follow-up FC-heavy MLP benchmark was then added to test the obvious escape hatch: move the same SPARROW idea to a workload with much more fully connected structure. That benchmark did not rescue the result. Across three repeats on April 9, 2026, the FC-heavy ATLAS baseline landed at `12.32 +/- 0.24s`, `96.56 +/- 0.22%`, while SPARROW remained flat or slightly worse: `cRank=0` -> `13.74 +/- 0.51s`, `96.52 +/- 0.22%`; `cRank=2` -> `13.61 +/- 0.10s`, `96.51 +/- 0.22%`; `cRank=4` -> `13.03 +/- 0.35s`, `96.55 +/- 0.29%`.
- Two materially different task classes were then added through `atlas-alt-bench`. The small autoregressive token-LM benchmark was a clean negative result: across three repeats on April 9, 2026, AdamW remained strongest at `train=1.33 +/- 0.00s`, `testNLL=4.32010 +/- 0.02348`, `testPPL=75.217 +/- 1.751`, while isotropic ATLAS reached `train=2.10 +/- 0.01s`, `testNLL=4.19541 +/- 0.00789`, and SPARROW only matched or slightly worsened that ATLAS result at much lower throughput (`train=3.77 +/- 0.09s`, `testNLL=4.20153 +/- 0.01115`, `testPPL=66.793 +/- 0.743`).
- The planted teacher-student benchmark was the first materially different task where the transfer-memory branch clearly paid off. Across three repeats on April 9, 2026, AdamW landed at `testMSE=0.07863 +/- 0.00278`, `testR2%=3.737 +/- 3.398`, isotropic ATLAS improved that to `testMSE=0.05933 +/- 0.01585`, `testR2%=27.366 +/- 19.406`, and ATLAS-SPARROW improved again to `testMSE=0.04640 +/- 0.00142`, `testR2%=43.200 +/- 1.733`, with a moderate throughput penalty (`1.67s` vs `1.28s` for isotropic ATLAS).
- A later rank-2 SPARROW follow-up showed that the transfer-memory branch is real, but not uniformly improved by adding a second streaming mode. On the canonical planted teacher cases, rank-2 won the strongest signal case (`signal-win`: base `0.13618`, rank-1 `0.10486`, rank-2 `0.10431`), lost the neutral anchor (`default-anchor`: base `0.05800`, rank-1 `0.05847`, rank-2 `0.06964`), and gave back the earlier rank-1 advantage in the failure band (`failure-band`: base `0.17332`, rank-1 `0.12703`, rank-2 `0.17300`). On latent-state forecasting, rank-1 and rank-2 were effectively tied on held-out accuracy while rank-2 was slightly slower: rank-1 `train=2.53 +/- 0.01s`, `testMSE=0.00266 +/- 0.00017`, `testR2%=34.980 +/- 4.159`; rank-2 `train=2.60 +/- 0.03s`, `testMSE=0.00266 +/- 0.00017`, `testR2%=34.979 +/- 4.158`.
- An automatic second-mode gate then narrowed that conclusion. With `modeRankCap=2`, `autoGate=1`, `secondEdgeThreshold=0.10`, and `secondEdgeFraction=0.50`, the canonical planted cases shifted to: `signal-win` base `0.13618`, rank-1 `0.10486`, auto `0.10432`; `default-anchor` base `0.05800`, rank-1 `0.05847`, auto `0.06964`; `failure-band` base `0.17332`, rank-1 `0.12703`, auto `0.17311`. On latent-state forecasting, rank-1 and auto-gated cap-2 were effectively identical on both speed and held-out accuracy: rank-1 `train=2.53 +/- 0.01s`, `testMSE=0.00266 +/- 0.00017`, `testR2%=34.980 +/- 4.159`; auto-gated cap-2 `train=2.53 +/- 0.01s`, `testMSE=0.00266 +/- 0.00017`, `testR2%=34.981 +/- 4.159`. The practical reading is that a gated second mode is a safer experimental cap than unconditional rank-2, but it still does not replace rank-1 as the default SPARROW setting.
- A final nonlinear latent-dynamics follow-up tested whether the positive branch survives once the latent process stops being purely linear. It did not. On a switching/tanh partially observed forecasting task, isotropic ATLAS still beat SPARROW: ATLAS-BSRP reached `testMSE=0.00143 +/- 0.00009`, `testR2%=21.498 +/- 4.802`, while SPARROW rank-1 fell to `0.00159 +/- 0.00014`, `12.295 +/- 7.634`. Auto-gated cap-2 materially activated the second streaming mode (`activeModes=1.69 +/- 0.35`, `mode2Frac=0.685 +/- 0.346`, `edge2/1=0.662 +/- 0.166`) but did not improve the held-out result over rank-1. That is the strongest current evidence that the remaining bottleneck is the observable family, not just mode count or controller gating.
- The stronger historical reference remains the earlier April 8 isotropic branch snapshot: `cRank=0` reached `train=8.06s`, `testAcc=98.14%`. None of the later transfer-oriented memory prototypes surpassed that earlier benchmark.

## Appendix H: SPARROW Prototype Note

On April 9, 2026, a minimal CPU-only SPARROW prototype was added as a higher-throughput descendant of GHOST:

- it keeps the approximate quotient-horizontal projection used by GHOST,
- it replaces lag-stack balanced-mode extraction with a streaming rank-1 left/right transfer observer,
- it fits a single stable latent pole online and uses that latent state only for a memory-style active correction,
- it keeps explicit complement activation disabled in the minimal prototype, but still allows a larger scout-capacity budget to improve the observer.

Focused validation:

- `atlas-controller` passes after the SPARROW integration, including new memory-only controller tests.
- `atlas` passes, including the expanded checkpoint-config coverage and SPARROW controller regressions.

Standard MNIST benchmark results on **April 9, 2026**:

- baseline `cRank=0`, `sparrow=0`: `train=8.47s`, `testAcc=97.84%`
- memory-only `cRank=0`, `sparrow=1`: `train=8.75s`, `testAcc=97.76%`
- scout budget `cRank=2`, `sparrow=1`: `train=8.76s`, `testAcc=97.88%`
- larger scout budget `cRank=4`, `sparrow=1`: `train=8.86s`, `testAcc=98.00%`

Interpretation:

- SPARROW is materially faster than GHOST while preserving the same core idea: quotient-horizontalization plus directional transfer memory instead of explicit complement geometry.
- The cheap `cRank=0` SPARROW path did not help on standard MNIST, so the useful part of the signal is not captured by the smallest possible observer.
- Increasing the scout budget to `cRank=2` and `cRank=4` improved out-of-sample accuracy again without reverting to the full GHOST runtime tax, which suggests the useful information is still transfer-directional but not well represented by the smallest scout sketch.
- The best SPARROW point in this pass (`cRank=4`, `98.00%`) remains slower than the same-day isotropic baseline and still does not surpass the stronger April 8 isotropic reference (`98.14%`), so the prototype should stay opt-in.

Later follow-up on **April 9, 2026**:

- A rank-2 SPARROW extension was added as a direct test of whether the positive task-class results were still rank-1 limited.
- The result was regime-dependent rather than uniformly better. Rank-2 slightly improved the strongest planted teacher case, regressed the neutral anchor, and erased the earlier rank-1 advantage in a known failure band.
- On latent-state forecasting, rank-1 and rank-2 were effectively tied on held-out accuracy, with rank-2 slightly slower (`2.60s` vs `2.53s`).
- So the practical follow-up conclusion is narrower than “increase mode rank”: rank-2 should remain task-conditioned or opt-in, not treated as a new default SPARROW setting.

## Appendix I: ORBIT-Lite Prototype Note

On April 9, 2026, a minimal CPU-only ORBIT-Lite prototype was added as a final-layer, quotient-function-space descendant of the GHOST/SPARROW line:

- it is restricted to small output heads rather than hidden FC blocks,
- it removes common-logit row-mean motion before scoring a mode,
- it uses a bounded trace-normalized class-space spike rather than an unregularized generalized eigenvalue,
- it applies a true memory-only correction by shifting the current signal into the next-step latent state instead of feeding it back immediately,
- it keeps explicit complement activation off in the minimal path and only perturbs the active correction.

Focused validation:

- `atlas-controller` passes after the ORBIT-Lite integration, including the new output-head memory-only controller coverage.
- `atlas` passes, including the expanded checkpoint-config coverage and ORBIT controller regressions.

Standard MNIST benchmark results on **April 9, 2026**:

- baseline `cRank=0`, `orbit=0`: `train=8.23s`, `testAcc=97.84%`
- memory-only `cRank=0`, `orbit=1`: `train=8.25s`, `testAcc=97.36%`
- mixed path `cRank=4`, `orbit=1`: `train=8.36s`, `testAcc=97.40%`

Representative output-head diagnostics at the logged step-200 control boundary after stabilization:

- memory-only `cRank=0`: `orbit_edge=0.315021`, `orbit_sigma=0.315021`, `orbit_pole=0.95`, `orbit_memory_gain=0.0111588`
- mixed `cRank=4`: `orbit_edge=0.231249`, `orbit_sigma=0.231249`, `orbit_pole=0.95`, `orbit_memory_gain=0.00763152`

Interpretation:

- The stabilized ORBIT-Lite path did what it was supposed to mechanically: the output-head observer stopped saturating, the edge score became bounded and interpretable, and the memory gain stayed small.
- Even after that correction, ORBIT-Lite remained accuracy-negative on standard MNIST. So the failure is no longer numerical instability; it is that this benchmark still does not reward the output-head quotient-memory path enough to beat the isotropic ATLAS baseline.
- Unlike the earlier naive ORBIT attempt, the final recorded result is a meaningful negative result rather than a broken prototype. That is useful: it narrows the remaining headroom and suggests that moving further toward explicit function-space quotienting will need either a stronger benchmark or richer output-space observables than this minimal last-layer sketch.

## Appendix J: QBRT Prototype Note

On April 9, 2026, a minimal CPU-only QBRT prototype was added as a quotient-balanced, transfer-first descendant of the GHOST/SPARROW line:

- it reuses the existing compressed active/scout histories rather than adding another explicit complement basis family,
- it forms a small quotient-style horizontal past/future transfer problem on the compressed state,
- it extracts a bounded rank-1 balanced transfer score and left/right memory mode,
- it fits a stable latent pole online and uses that latent state only for an active-space memory correction,
- it leaves explicit complement activation effectively suppressed in the minimal path, even when `cRank=4` is allowed.

Focused validation:

- `atlas-controller` passes after the QBRT integration, including the new balanced-transfer controller coverage.
- `atlas` passes, including the expanded checkpoint-config coverage and QBRT controller regressions.

Standard MNIST benchmark results on **April 9, 2026**:

- baseline `cRank=0`, `qbrt=0`: `train=8.28s`, `testAcc=97.84%`
- memory-only `cRank=0`, `qbrt=1`: `train=10.00s`, `testAcc=97.78%`
- mixed path `cRank=4`, `qbrt=1`: `train=9.97s`, `testAcc=97.78%`

Representative hidden-FC diagnostics at the logged step-200 control boundary:

- memory-only `cRank=0`: `qbrt_edge=0.761053`, `qbrt_sigma=1.76251`, `qbrt_pole=0.0719671`, `qbrt_horizontal_ratio=0.998086`, `qbrt_memory_gain=0.0379798`
- mixed `cRank=4`: `qbrt_edge=0.884316`, `qbrt_sigma=1.88637`, `qbrt_pole=0.00783868`, `qbrt_horizontal_ratio=0.997686`, `qbrt_memory_gain=0.0441135`

Interpretation:

- QBRT confirmed the main late-prototype lesson rather than overturning it: a more structured balanced-transfer score can be measured cleanly, but on standard MNIST that signal still does not convert into a better ATLAS update.
- Unlike the older complement experiments, QBRT did not fail because its controller was obviously misfiring. The recorded transfer edge and memory gain are both finite and nontrivial, yet the benchmark still lands slightly below the isotropic baseline.
- That makes QBRT a useful negative result. It narrows the likely cause further: on this workload, the residual transfer signal is either too weak, too noisy, or too misaligned with generalization to justify the extra memory machinery.
- In practical terms, QBRT should remain opt-in. It adds overhead comparable to the heavier transfer prototypes while failing to beat the simpler isotropic path or the cheaper SPARROW variant.

## Appendix K: RIFT Prototype Note

On April 9, 2026, a minimal CPU-only RIFT prototype was added as a path-signature descendant of the GHOST/SPARROW line:

- it keeps quotient-horizontal projection in the compressed active/scout state,
- it replaces the rank-1 transfer observer with an order-2 signature-style feature map over recent control-boundary history,
- it fits a stable memory-only latent state from that path feature block,
- it keeps explicit complement activation suppressed even when `cRank=4` is allowed.

Focused validation:

- `atlas-controller` passes after the RIFT integration, including the new signature-path controller coverage.
- `atlas` passes, including the expanded checkpoint-config coverage and the RIFT controller regressions.

Standard MNIST benchmark results on **April 9, 2026**:

- baseline `cRank=0`, `rift=0`: `train=8.34s`, `testAcc=97.84%`
- memory-only `cRank=0`, `rift=1`: `train=17.80s`, `testAcc=97.80%`
- mixed path `cRank=4`, `rift=1`: `train=17.93s`, `testAcc=97.84%`

Representative step-200 diagnostics from the final benchmark runs:

- hidden FC, `cRank=0`: `rift_edge=0.0196265`, `rift_sigma=1`, `rift_pole=0.000334218`, `rift_horizontal_ratio=0.997851`, `rift_area_energy=0.000386859`, `rift_memory_gain=0`
- output head, `cRank=0`: `rift_edge=0.0656473`, `rift_sigma=1`, `rift_pole=0.0593547`, `rift_horizontal_ratio=0.991745`, `rift_area_energy=0.00438161`, `rift_memory_gain=0.000823541`
- hidden FC, `cRank=4`: `rift_edge=0.0216972`, `rift_sigma=1`, `rift_pole=0.00100023`, `rift_horizontal_ratio=0.997683`, `rift_area_energy=0.000472958`, `rift_memory_gain=0`

Interpretation:

- RIFT is a clean negative result, not a broken prototype. The signature-path observer is bounded, restorable, and test-covered.
- On this workload, the extra path-statistic machinery is expensive and almost entirely inactive at the hidden FC layer. The measured level-2 area share stays tiny, and the resulting memory gain is usually zero at the layer that originally motivated the complement work.
- Allowing `cRank=4` does not recover the lost runtime or produce a better result than the same-day isotropic baseline; it only climbs back to a statistical tie (`97.84%`) while staying much slower.
- Relative to SPARROW, the lesson is narrow but useful: richer pathwise features did not help because the remaining signal on standard MNIST is too weak to justify second-level signature machinery. The surviving positive clue is still the cheaper transfer-directional observer, not a heavier path-geometry model.

## Appendix L: QRC Prototype Note

On April 9, 2026, a minimal CPU-only QRC prototype was added as a reduced robust-control descendant of the SPARROW/QBRT line:

- it reuses the existing compressed active/scout histories rather than introducing another complement-basis family,
- it forms a small quotient-active transfer state and extracts a bounded scalar edge, pole, and control gain at ATLAS control boundaries,
- it keeps the path memory-only by applying a reduced active-space correction through a left transfer mode and latent state,
- it suppresses explicit complement activation when `qrc=1`, even if `cRank=4` is allowed.

Focused validation:

- `atlas-controller` passes after the QRC integration, including the new reduced-control controller coverage.
- `atlas` passes, including the expanded checkpoint-config coverage and the QRC controller regressions.

Standard MNIST benchmark results on **April 9, 2026**:

- baseline `cRank=0`, `qrc=0`: `train=8.21s`, `testAcc=97.84%`
- memory-only `cRank=0`, `qrc=1`: `train=9.90s`, `testAcc=97.84%`
- mixed path `cRank=4`, `qrc=1`: `train=10.05s`, `testAcc=97.80%`

Representative hidden-FC diagnostics at the logged step-200 control boundary:

- memory-only `cRank=0`: `qrc_edge=0.840688`, `qrc_sigma=1.84868`, `qrc_pole=0.0907657`, `qrc_horizontal_ratio=0.998124`, `qrc_control_gain=0.0832129`, `qrc_memory_gain=0.0034978`
- mixed `cRank=4`: `qrc_edge=0.760746`, `qrc_sigma=1.76548`, `qrc_pole=0.0664062`, `qrc_horizontal_ratio=0.997948`, `qrc_control_gain=0.062271`, `qrc_memory_gain=0.00236862`

Interpretation:

- QRC is a clean negative result rather than a broken controller. The reduced transfer edge, pole, and gain all stay bounded and interpretable.
- The prototype confirms the narrow late-stage lesson: if there is any proceed path left on standard MNIST, it is a small active-space transfer correction, not richer complement geometry.
- Even so, the QRC controller does not buy new headroom on this workload. The memory-only path only matches the same-day isotropic baseline on accuracy while paying roughly a 20% runtime penalty, and allowing `cRank=4` makes the result slightly worse.
- Relative to SPARROW, the practical conclusion is unfavorable. The extra reduced-control machinery did not improve the accuracy/runtime tradeoff, so QRC should remain opt-in rather than replacing the cheaper streaming observer.

## Appendix M: FC-Heavy MLP Follow-Up

On April 9, 2026, the `atlas-bench` harness was extended with a dedicated `fc-heavy` mode to test the narrowest remaining proceed hypothesis:

- keep the dataset fixed to MNIST,
- replace the LeNet-style CNN with a DFF MLP `784 -> 512 -> 256 -> 128 -> 10`,
- keep the SPARROW controller as the cheapest surviving transfer-memory branch,
- ask whether the ATLAS signal strengthens when the workload is dominated by fully connected layers rather than conv blocks.

The FC-heavy mode uses a smaller default run budget than the CNN benchmark so repeated CPU runs stay practical:

- train split cap `2000`,
- test split cap `1000`,
- epochs `3`,
- batch size `128`.

Three-repeat standard results on **April 9, 2026**:

- baseline `cRank=0`, `sparrow=0`: `train=12.32 +/- 0.24s`, `testAcc=96.56 +/- 0.22%`
- memory-only `cRank=0`, `sparrow=1`: `train=13.74 +/- 0.51s`, `testAcc=96.52 +/- 0.22%`
- scout budget `cRank=2`, `sparrow=1`: `train=13.61 +/- 0.10s`, `testAcc=96.51 +/- 0.22%`
- larger scout budget `cRank=4`, `sparrow=1`: `train=13.03 +/- 0.35s`, `testAcc=96.55 +/- 0.29%`

Interpretation:

- This benchmark was the most obvious “maybe ATLAS still has room” follow-up after the standard-MNIST CNN falsifier. It gives the method more fully connected structure without changing the data domain.
- The result is still negative. SPARROW does not produce a statistically meaningful accuracy lift over the FC-heavy isotropic ATLAS baseline, and every tested SPARROW setting is slower.
- The `cRank=4` point is the least bad variant in this pass, but it only recovers to a statistical tie with baseline (`96.55 +/- 0.29%` vs `96.56 +/- 0.22%`) while remaining slower (`13.03s` vs `12.32s`).
- That materially weakens the “just move to a more FC-heavy workload” argument. On this repo’s current MNIST-family workloads, the remaining ATLAS headroom looks narrow enough that further complement/transfer variants are hard to justify without changing the observable family or the task class much more aggressively.

## Appendix N: Alternate Task-Class Follow-Up

On April 9, 2026, a new `atlas-alt-bench` harness was added to test materially different proceed paths:

- an autoregressive next-token benchmark with a small CPU transformer decoder and synthetic order-2 recurrence data,
- a planted teacher-student regression benchmark with a low-rank teacher signal plus nuisance bulk.
- a partially observed latent-state forecasting benchmark with windowed DFF prediction on top of a stable latent linear system.

The goal was to stop guessing from MNIST-family image tasks and ask two sharper questions:

- does ATLAS/SPARROW help on a task where sequential transfer is intrinsic?
- does it help on a task where the low-rank transfer signal is explicitly planted?

### N.1 Autoregressive token-LM benchmark

Setup:

- decoder-only transformer, `vocab=65`, `dModel=48`, `dFF=192`, `layers=2`, `heads=4`,
- `seqLen=32`, `trainSeqs=128`, `testSeqs=32`, `epochs=6`,
- compared variants: AdamW, isotropic ATLAS-BSRP (`cRank=0`), and ATLAS-SPARROW (`cRank=4`).

Three-repeat results on **April 9, 2026**:

- AdamW: `train=1.33 +/- 0.00s`, `tok/s=18409.1 +/- 33.8`, `trainNLL=3.45656 +/- 0.00956`, `trainPPL=31.709 +/- 0.303`, `testNLL=4.32010 +/- 0.02348`, `testPPL=75.217 +/- 1.751`
- ATLAS-BSRP: `train=2.10 +/- 0.01s`, `tok/s=11688.1 +/- 30.2`, `trainNLL=4.01758 +/- 0.00167`, `trainPPL=55.567 +/- 0.093`, `testNLL=4.19541 +/- 0.00789`, `testPPL=66.383 +/- 0.522`
- ATLAS-SPARROW: `train=3.77 +/- 0.09s`, `tok/s=6520.0 +/- 160.1`, `trainNLL=4.01383 +/- 0.00655`, `trainPPL=55.360 +/- 0.362`, `testNLL=4.20153 +/- 0.01115`, `testPPL=66.793 +/- 0.743`

Interpretation:

- This is not a rescue for the ATLAS proceed path. SPARROW does not improve over isotropic ATLAS here and is substantially slower.
- The task is still useful as a falsifier: it shows that “sequence memory exists” is not enough by itself. The ATLAS transfer-memory branch still needs the right observable signal, not just any sequential workload.
- AdamW remains best on raw likelihood/perplexity in this small-token-LM regime.

### N.2 Planted teacher-student benchmark

Setup:

- student DFF regressor with input dimension `32`, hidden widths `64 -> 32 -> 1`,
- synthetic teacher with planted low-rank signal `rank=4` plus nuisance bulk `rank=12`,
- `train=4096`, `test=1024`, `batch=64`, `epochs=20`, `bulkScale=0.20`,
- compared variants: AdamW, isotropic ATLAS-BSRP (`cRank=0`), and ATLAS-SPARROW (`cRank=4`).

Three-repeat results on **April 9, 2026**:

- AdamW: `train=1.20 +/- 0.00s`, `samples/s=68323.7 +/- 80.5`, `trainMSE=0.07765 +/- 0.00382`, `trainR2%=4.202 +/- 4.708`, `testMSE=0.07863 +/- 0.00278`, `testR2%=3.737 +/- 3.398`
- ATLAS-BSRP: `train=1.28 +/- 0.00s`, `samples/s=63966.7 +/- 62.3`, `trainMSE=0.05721 +/- 0.01654`, `trainR2%=29.426 +/- 20.405`, `testMSE=0.05933 +/- 0.01585`, `testR2%=27.366 +/- 19.406`
- ATLAS-SPARROW: `train=1.67 +/- 0.00s`, `samples/s=49122.5 +/- 13.9`, `trainMSE=0.04376 +/- 0.00109`, `trainR2%=46.009 +/- 1.341`, `testMSE=0.04640 +/- 0.00142`, `testR2%=43.200 +/- 1.733`

Interpretation:

- This is the first materially different workload in the entire ATLAS exploration where the transfer-memory branch shows a clear positive result rather than a tie or a clean negative.
- The ranking is coherent with the planted construction: when the task really does contain a recoverable low-rank transfer structure, isotropic ATLAS helps, and SPARROW helps more.
- The cost is throughput, not correctness. SPARROW is slower than isotropic ATLAS, but the gain is large enough here to justify calling it a real signal.

### N.3 Latent-state forecasting benchmark

Setup:

- partially observed latent linear dynamical system with `latentDim=6`, `obsDim=4`, `window=8`, `seqLen=40`,
- `trainSeqs=96`, `testSeqs=24`, `epochs=18`,
- student DFF forecaster with hidden widths `96 -> 64 -> 4`,
- compared variants: AdamW, isotropic ATLAS-BSRP (`cRank=0`), and ATLAS-SPARROW (`cRank=4`).

Three-repeat results on **April 9, 2026**:

- AdamW: `train=2.06 +/- 0.00s`, `windows/s=26799.5 +/- 52.2`, `trainMSE=0.00400 +/- 0.00013`, `trainR2%=15.352 +/- 2.791`, `testMSE=0.00351 +/- 0.00013`, `testR2%=14.154 +/- 3.233`
- ATLAS-BSRP: `train=2.19 +/- 0.00s`, `windows/s=25284.0 +/- 49.9`, `trainMSE=0.00313 +/- 0.00025`, `trainR2%=33.767 +/- 5.382`, `testMSE=0.00284 +/- 0.00022`, `testR2%=30.541 +/- 5.448`
- ATLAS-SPARROW: `train=2.52 +/- 0.01s`, `windows/s=21977.9 +/- 53.8`, `trainMSE=0.00297 +/- 0.00024`, `trainR2%=37.133 +/- 5.165`, `testMSE=0.00266 +/- 0.00017`, `testR2%=34.980 +/- 4.159`

Interpretation:

- This is the second positive result for the transfer-memory branch and the first one that sits between the fully planted teacher-student task and the negative token-LM result.
- The ranking again matches the proceed hypothesis: isotropic ATLAS helps relative to AdamW, and SPARROW helps more, but with a modest throughput tax.
- That makes the line more credible than the teacher-student result alone. SPARROW is no longer winning only on an explicitly planted static regression problem; it is also helping on a modest dynamical forecasting problem with partial observability.

### N.4 Teacher-sweep regime map

To avoid overfitting the proceed story to one planted point, the alternate harness was also extended with a `teacher-sweep` mode. The first smoke run intentionally used a bounded quick slice:

- command: `./glades-unit-tests atlas-alt-bench --mode teacher-sweep --repeats 1 --teacher-sweep-limit 4`
- axes exercised in that bounded slice: `teacherRank=2`, `bulkRank in {0, 12}`, `bulkScale in {0.0, 0.2, 0.8}`, `cRank in {0, 4}`

Observed quick-slice results on **April 9, 2026**:

- 4 base teacher/bulk configurations were evaluated,
- across the resulting 8 SPARROW-vs-baseline rows, SPARROW improved test MSE in 6 cases,
- the best observed delta was `-0.01631` test MSE at `teacherRank=2`, `bulkRank=0`, `bulkScale=0.20`, `cRank=4`,
- the worst observed delta in the bounded slice was `+0.01551` at `teacherRank=2`, `bulkRank=0`, `bulkScale=0.80`, `cRank=4`.

Interpretation:

- The positive teacher-student result is not a single isolated point; there is already a visible regime map.
- SPARROW benefits are strongest when the planted signal is present but not completely dominant. When the nuisance/bulk structure is absent or the scale regime shifts, the right `cRank` changes and the branch can still regress.
- That is exactly the behavior needed to justify the next stage: a real phase diagram on teacher-student first, then a transition benchmark such as latent-state forecasting.

A larger repeated quick-profile sweep was then run on **April 9, 2026**:

- command: `./glades-unit-tests atlas-alt-bench --mode teacher-sweep --repeats 3 --teacher-sweep-profile quick`
- axes: `teacherRank in {2, 4, 8}`, `bulkRank in {0, 12, 24}`, `bulkScale in {0.0, 0.2, 0.8}`, `cRank in {0, 4}`

Observed repeated quick-profile summary:

- `baseConfigs=27`, producing `54` SPARROW-vs-baseline rows,
- SPARROW improved test MSE in `30 / 54` rows,
- the best observed delta was `-0.03162` test MSE at `teacherRank=8`, `bulkRank=24`, `bulkScale=0.00`, `cRank=4`,
- clear failure bands remained, including `+0.09112` at `teacherRank=8`, `bulkRank=12`, `bulkScale=0.20`, `cRank=4`.

Interpretation:

- The proceed path is now supported by an actual regime map, not just a single planted win.
- SPARROW is not uniformly beneficial. Its gains depend materially on the teacher/bulk structure, and `cRank=4` is not universally better than `cRank=0`.
- That is a healthy result: it implies the transfer-memory branch is responding to task structure rather than just adding generic regularization or noise.

### N.5 Latent-state tuning note

Because the latent-state forecasting task is the first non-planted dynamic task where SPARROW helped, a small knob sweep was run around the default SPARROW settings on **April 9, 2026**:

- baseline rerun: `cRank=0`, `memoryScale=0.05`, `edge=0.10`, `tSub=64`
- variants tested:
  - `memoryScale=0.03`
  - `memoryScale=0.08`
  - `edge=0.05`
  - `edge=0.15`
  - `tSub=32`

Observed result:

- all tested variants stayed effectively tied within noise on held-out accuracy,
- the repeated baseline rerun with `cRank=0` already matched the earlier latent SPARROW result: `testMSE=0.00266 +/- 0.00017`, `testR2%=34.977 +/- 4.162`,
- none of the tested one-knob perturbations produced a meaningful improvement over that point.

Interpretation:

- The latent positive result appears structural, not a fragile threshold accident.
- On this benchmark, the default SPARROW controller is already close to the local optimum within the easy one-knob tuning surface.
- That shifts the next research step away from threshold fiddling and toward broader benchmark transfer or richer reduced-state models only if a new task justifies them.

### N.6 Rank-2 SPARROW follow-up

Because the teacher-student and latent-state tasks were the first genuinely positive settings for the transfer-memory branch, a direct rank-2 SPARROW follow-up was run on **April 9, 2026**.

Canonical planted teacher cases:

- command: `./glades-unit-tests atlas-alt-bench --mode teacher-canonical --repeats 3`
- compared variants: isotropic ATLAS-BSRP, SPARROW rank-1, and SPARROW rank-2

Observed results:

- `signal-win` (`teacherRank=8`, `bulkRank=24`, `bulkScale=0.00`, `cRank=4`):
  - base `testMSE=0.13618`
  - SPARROW rank-1 `testMSE=0.10486`
  - SPARROW rank-2 `testMSE=0.10431`
  - winner: `sparrow-r2`
- `default-anchor` (`teacherRank=4`, `bulkRank=12`, `bulkScale=0.20`, `cRank=4`):
  - base `testMSE=0.05800`
  - SPARROW rank-1 `testMSE=0.05847`
  - SPARROW rank-2 `testMSE=0.06964`
  - winner: `base`
- `failure-band` (`teacherRank=8`, `bulkRank=12`, `bulkScale=0.20`, `cRank=4`):
  - base `testMSE=0.17332`
  - SPARROW rank-1 `testMSE=0.12703`
  - SPARROW rank-2 `testMSE=0.17300`
  - winner: `sparrow-r1`

Direct latent-state comparison:

- command: `./glades-unit-tests atlas-alt-bench --mode latent-forecast --repeats 3 --atlas-complement-rank 4 --atlas-sparrow-mode-rank {1,2}`
- rank-1:
  - `train=2.53 +/- 0.01s`
  - `testMSE=0.00266 +/- 0.00017`
  - `testR2%=34.980 +/- 4.159`
- rank-2:
  - `train=2.60 +/- 0.03s`
  - `testMSE=0.00266 +/- 0.00017`
  - `testR2%=34.979 +/- 4.158`

Interpretation:

- The added streaming mode is not a general improvement. It helps in the strongest planted signal regime, but it can regress neutral or mixed regimes.
- On the first non-planted dynamic task where SPARROW worked, rank-2 appears effectively redundant relative to rank-1.
- So the correct reading is not “SPARROW wants higher mode rank.” It is “SPARROW is now credible enough that mode rank becomes a regime variable rather than a monotone upgrade.”

Bottom line:

- On MNIST-family tasks, ATLAS complement/transfer work remains mostly a dead end.
- On token-LM, the same branch still does not help.
- On planted teacher-student and latent-state forecasting tasks, the SPARROW transfer-memory branch works.
- Rank-2 SPARROW is regime-dependent, not a new default.
- Auto-gated cap-2 SPARROW is a safer experimental variant than unconditional rank-2, but rank-1 remains the default-worthy setting.
- On a harder nonlinear latent forecasting task, the second mode activates but does not help, which points to an observable-family limit rather than a simple rank-selection problem.
- So the global conclusion is now narrower and more actionable than “ATLAS is dead”: the current ATLAS observable family appears to need tasks with genuinely strong low-rank transfer structure or partially observed dynamical transfer before its transfer-memory branch pays off.

### N.7 Auto-gated second-mode follow-up

Because unconditional rank-2 SPARROW was clearly regime-dependent, an automatic second-mode gate was added and evaluated on **April 9, 2026**.

Gate settings:

- `modeRankCap=2`
- `autoGate=1`
- `secondEdgeThreshold=0.10`
- `secondEdgeFraction=0.50`

Canonical planted teacher cases:

- command: `./glades-unit-tests atlas-alt-bench --mode teacher-canonical --repeats 3`
- compared variants: isotropic ATLAS-BSRP, SPARROW rank-1, and SPARROW auto-gated cap-2

Observed results:

- `signal-win` (`teacherRank=8`, `bulkRank=24`, `bulkScale=0.00`, `cRank=4`):
  - base `testMSE=0.13618`
  - SPARROW rank-1 `testMSE=0.10486`
  - SPARROW auto-gated cap-2 `testMSE=0.10432`
  - winner: `sparrow-auto`
- `default-anchor` (`teacherRank=4`, `bulkRank=12`, `bulkScale=0.20`, `cRank=4`):
  - base `testMSE=0.05800`
  - SPARROW rank-1 `testMSE=0.05847`
  - SPARROW auto-gated cap-2 `testMSE=0.06964`
  - winner: `base`
- `failure-band` (`teacherRank=8`, `bulkRank=12`, `bulkScale=0.20`, `cRank=4`):
  - base `testMSE=0.17332`
  - SPARROW rank-1 `testMSE=0.12703`
  - SPARROW auto-gated cap-2 `testMSE=0.17311`
  - winner: `sparrow-r1`

Direct latent-state comparison:

- command: `./glades-unit-tests atlas-alt-bench --mode latent-forecast --repeats 3 --atlas-complement-rank 4 --atlas-sparrow-mode-rank 2 --atlas-sparrow-auto-mode-gate 1`
- rank-1:
  - `train=2.53 +/- 0.01s`
  - `testMSE=0.00266 +/- 0.00017`
  - `testR2%=34.980 +/- 4.159`
- auto-gated cap-2:
  - `train=2.53 +/- 0.01s`
  - `testMSE=0.00266 +/- 0.00017`
  - `testR2%=34.981 +/- 4.159`

Interpretation:

- The gate does what it was designed to do. It preserves the strong planted-case improvement without forcing a universally active second mode.
- On the first non-planted dynamic task where SPARROW works, the gate collapses cleanly to rank-1 behavior rather than adding extra cost.
- It does not rescue the neutral anchor or known planted failure band, so it is not a global default upgrade.
- The correct policy is therefore asymmetric:
  - rank-1 remains the default SPARROW setting,
  - auto-gated cap-2 is the safer experimental option when a task is believed to contain a second strong transfer mode,
  - unconditional rank-2 should remain opt-in only.

### N.8 Mode-usage instrumentation and nonlinear latent follow-up

Because the next question after the auto-gated pass was not “does mode 2 exist?” but “does it actually activate on harder dynamic tasks?”, a small read-only ATLAS runtime summary was added on **April 9, 2026** and threaded into `atlas-alt-bench`.

Tracked SPARROW diagnostics:

- mean retained mode count per epoch,
- fraction of epochs with retained mode count at least `2`,
- mean raw `mode-2 / mode-1` singular proxy ratio,
- mean retained SPARROW edge,
- mean memory gain,
- mean horizontal ratio.

Linear latent-state forecasting, rank-1 baseline:

- command: `./glades-unit-tests atlas-alt-bench --mode latent-forecast --repeats 3 --atlas-complement-rank 4 --atlas-sparrow-mode-rank 1`
- ATLAS-SPARROW:
  - `train=2.53 +/- 0.01s`
  - `testMSE=0.00266 +/- 0.00017`
  - `testR2%=34.980 +/- 4.159`
  - `activeModes=1.00 +/- 0.00`
  - `mode2Frac=0.000 +/- 0.000`
  - `edge2/1=0.000 +/- 0.000`
  - `edge=0.524 +/- 0.017`
  - `mem=0.024 +/- 0.001`
  - `horiz=0.999 +/- 0.000`

Interpretation:

- On the first non-planted dynamic task where SPARROW clearly helped, the controller is genuinely rank-1. The second mode is not merely “inactive by policy”; it is absent in the retained dynamics.

Nonlinear latent-state forecasting, rank-1:

- command: `./glades-unit-tests atlas-alt-bench --mode nonlinear-forecast --repeats 3 --atlas-complement-rank 4 --atlas-sparrow-mode-rank 1`
- AdamW:
  - `testMSE=0.00182 +/- 0.00005`
  - `testR2%=0.178 +/- 2.807`
- ATLAS-BSRP:
  - `train=4.00 +/- 0.00s`
  - `testMSE=0.00143 +/- 0.00009`
  - `testR2%=21.498 +/- 4.802`
- ATLAS-SPARROW rank-1:
  - `train=4.39 +/- 0.00s`
  - `testMSE=0.00159 +/- 0.00014`
  - `testR2%=12.295 +/- 7.634`
  - `activeModes=1.00 +/- 0.00`
  - `mode2Frac=0.000 +/- 0.000`
  - `edge2/1=0.000 +/- 0.000`
  - `edge=0.676 +/- 0.008`
  - `mem=0.032 +/- 0.000`
  - `horiz=0.999 +/- 0.000`

Nonlinear latent-state forecasting, auto-gated cap-2:

- command: `./glades-unit-tests atlas-alt-bench --mode nonlinear-forecast --repeats 3 --atlas-complement-rank 4 --atlas-sparrow-mode-rank 2 --atlas-sparrow-auto-mode-gate 1`
- ATLAS-SPARROW auto-gated cap-2:
  - `train=4.38 +/- 0.00s`
  - `testMSE=0.00159 +/- 0.00014`
  - `testR2%=12.295 +/- 7.633`
  - `activeModes=1.69 +/- 0.35`
  - `mode2Frac=0.685 +/- 0.346`
  - `edge2/1=0.662 +/- 0.166`
  - `edge=0.768 +/- 0.069`
  - `mem=0.037 +/- 0.004`
  - `horiz=0.999 +/- 0.000`

Interpretation:

- The instrumentation is informative rather than decorative: on the nonlinear task, the second SPARROW mode really does activate often and with nontrivial strength.
- But the held-out result stays unchanged relative to rank-1 and still trails isotropic ATLAS.
- So the next bottleneck is not “mode 2 is unavailable” or “the gate is too strict.” It is that the current compressed observable family does not translate that extra transfer complexity into better prediction.
- That sharply narrows the next step. If the line continues, it should move to a different observable family such as hidden-state or output-space transfer, not another parameter-space SPARROW variant.

### N.9 HELM-Lite hidden/output observable follow-up

Because the nonlinear latent result suggested that parameter-space transfer modes were no longer the right observable family, a first hidden/output probe was added on **April 9, 2026**:

- `HELM-Lite` tracks the last hidden layer and the output residual,
- builds a rank-1 hidden-to-output predictive observer,
- and applies only a bounded output-head memory correction,
- with no explicit complement activation.

Linear latent-state forecasting:

- command: `./glades-unit-tests atlas-alt-bench --mode latent-forecast --repeats 3`
- AdamW:
  - `train=2.20 +/- 0.01s`
  - `testMSE=0.00351 +/- 0.00013`
  - `testR2%=14.154 +/- 3.233`
- ATLAS-BSRP:
  - `train=2.31 +/- 0.17s`
  - `testMSE=0.00284 +/- 0.00022`
  - `testR2%=30.541 +/- 5.448`
- ATLAS-SPARROW:
  - `train=2.52 +/- 0.01s`
  - `testMSE=0.00266 +/- 0.00017`
  - `testR2%=34.980 +/- 4.159`
  - `activeModes=1.00 +/- 0.00`
  - `edge=0.524 +/- 0.017`
  - `mem=0.024 +/- 0.001`
- ATLAS-HELM:
  - `train=2.19 +/- 0.00s`
  - `testMSE=0.00290 +/- 0.00027`
  - `testR2%=29.105 +/- 6.579`
  - `edge=0.000 +/- 0.000`
  - `sigma=0.000 +/- 0.000`
  - `predR2=0.014 +/- 0.009`
  - `mem=0.000 +/- 0.000`
  - `pole=0.950 +/- 0.000`

Nonlinear latent-state forecasting:

- command: `./glades-unit-tests atlas-alt-bench --mode nonlinear-forecast --repeats 3`
- AdamW:
  - `train=4.05 +/- 0.38s`
  - `testMSE=0.00182 +/- 0.00005`
  - `testR2%=0.178 +/- 2.807`
- ATLAS-BSRP:
  - `train=4.26 +/- 0.39s`
  - `testMSE=0.00143 +/- 0.00009`
  - `testR2%=21.498 +/- 4.802`
- ATLAS-SPARROW:
  - `train=4.37 +/- 0.00s`
  - `testMSE=0.00159 +/- 0.00014`
  - `testR2%=12.295 +/- 7.634`
  - `activeModes=1.00 +/- 0.00`
  - `edge=0.676 +/- 0.008`
  - `mem=0.032 +/- 0.000`
- ATLAS-HELM:
  - `train=3.99 +/- 0.00s`
  - `testMSE=0.00144 +/- 0.00012`
  - `testR2%=20.596 +/- 6.720`
  - `edge=0.000 +/- 0.000`
  - `sigma=0.000 +/- 0.000`
  - `predR2=0.020 +/- 0.007`
  - `mem=0.000 +/- 0.000`
  - `pole=0.950 +/- 0.000`

Interpretation:

- HELM-Lite is a meaningful observable-family probe rather than a broken prototype. It stays bounded, adds almost no runtime overhead, and cleanly falls back toward the isotropic ATLAS path when the hidden-to-output edge is subcritical.
- On the linear latent task, that fallback is not good enough to beat SPARROW. HELM underperforms both ATLAS-BSRP and SPARROW on held-out error, which means the minimal last-hidden/output-head observer is too weak to recover the planted predictive mode that SPARROW still captures indirectly.
- On the nonlinear latent task, HELM is materially better than SPARROW and nearly matches isotropic ATLAS while running slightly faster than both ATLAS baselines. That is the first evidence that the observable-family move itself is directionally correct even though the minimal rank-1 HELM instantiation is still too weak.
- The diagnostic pattern is the important part: `edge≈0`, `sigma≈0`, `mem≈0`, and a pole pinned at its stability cap mean HELM is effectively voting “no usable hidden/output transfer spike found.” In other words, the prototype is not discovering a bad mode and overfitting it; it is mostly declining to act.

### N.10 HELM-v2 stacked-hidden rank-2 follow-up

Because HELM-Lite mostly behaved like a clean no-op, a stronger hidden/output probe was added and rerun on **April 9, 2026** after a full backend rebuild:

- `HELM-v2` stacks the last two hidden layers instead of only the final hidden layer,
- raises the hidden/output observer cap to `modeRank=2`,
- keeps the correction output-head-only and memory-style,
- and logs HELM mode usage in the same style as the SPARROW instrumentation.

The full `atlas-controller` and `atlas` suites both passed after the rebuild, which also resolved a stale-build ABI mismatch caused by the expanded HELM state layout.

Linear latent-state forecasting, rebuilt HELM-v2:

- command: `./glades-unit-tests atlas-alt-bench --mode latent-forecast --repeats 3`
- AdamW:
  - `train=2.03 +/- 0.00s`
  - `testMSE=0.00351 +/- 0.00013`
  - `testR2%=14.154 +/- 3.233`
- ATLAS-BSRP:
  - `train=2.16 +/- 0.00s`
  - `testMSE=0.00284 +/- 0.00022`
  - `testR2%=30.541 +/- 5.448`
- ATLAS-SPARROW:
  - `train=2.50 +/- 0.01s`
  - `testMSE=0.00266 +/- 0.00017`
  - `testR2%=34.980 +/- 4.159`
  - `activeModes=1.00 +/- 0.00`
  - `mode2Frac=0.000 +/- 0.000`
  - `edge2/1=0.000 +/- 0.000`
  - `edge=0.524 +/- 0.017`
  - `mem=0.024 +/- 0.001`
  - `horiz=0.999 +/- 0.000`
- ATLAS-HELM-v2:
  - `train=2.18 +/- 0.00s`
  - `testMSE=0.00290 +/- 0.00027`
  - `testR2%=29.105 +/- 6.579`
  - `activeModes=0.00 +/- 0.00`
  - `mode2Frac=0.000 +/- 0.000`
  - `edge2/1=0.526 +/- 0.121`
  - `edge=0.000 +/- 0.000`
  - `sigma=0.000 +/- 0.000`
  - `predR2=0.015 +/- 0.010`
  - `mem=0.000 +/- 0.000`
  - `pole=0.950 +/- 0.000`

Nonlinear latent-state forecasting, rebuilt HELM-v2:

- command: `./glades-unit-tests atlas-alt-bench --mode nonlinear-forecast --repeats 3`
- AdamW:
  - `train=4.76 +/- 0.01s`
  - `testMSE=0.00182 +/- 0.00005`
  - `testR2%=0.178 +/- 2.807`
- ATLAS-BSRP:
  - `train=4.92 +/- 0.10s`
  - `testMSE=0.00143 +/- 0.00009`
  - `testR2%=21.498 +/- 4.802`
- ATLAS-SPARROW:
  - `train=4.69 +/- 0.47s`
  - `testMSE=0.00159 +/- 0.00014`
  - `testR2%=12.295 +/- 7.634`
  - `activeModes=1.00 +/- 0.00`
  - `mode2Frac=0.000 +/- 0.000`
  - `edge2/1=0.000 +/- 0.000`
  - `edge=0.676 +/- 0.008`
  - `mem=0.032 +/- 0.000`
  - `horiz=0.999 +/- 0.000`
- ATLAS-HELM-v2:
  - `train=4.27 +/- 0.39s`
  - `testMSE=0.00144 +/- 0.00012`
  - `testR2%=20.596 +/- 6.720`
  - `activeModes=0.00 +/- 0.00`
  - `mode2Frac=0.000 +/- 0.000`
  - `edge2/1=0.078 +/- 0.054`
  - `edge=0.000 +/- 0.000`
  - `sigma=0.000 +/- 0.000`
  - `predR2=0.023 +/- 0.009`
  - `mem=0.000 +/- 0.000`
  - `pole=0.915 +/- 0.025`

Interpretation:

- The stronger hidden/output observer does **not** solve the main HELM bottleneck. Even with a two-layer hidden stack and a rank-2 cap, the gate still stays shut: `activeModes=0`, `mode2Frac=0`, `edge≈0`, `sigma≈0`, and `mem≈0` on both dynamic tasks.
- That means the failure is no longer “rank-1 was too small” or “the last hidden layer was too narrow.” The current HELM observable family still does not expose a supercritical hidden-to-output transfer spike under this lightweight linear observer.
- The nonlinear latent result remains the useful clue. HELM-v2 stays much closer to ATLAS-BSRP than SPARROW while also running faster than both of those baselines on this task. So the observable-family move still looks directionally correct even though the present HELM gate never truly opens.
- The practical next step is therefore not more threshold tuning inside HELM-v2. It is a stronger hidden/output observable family, likely a reduced state-space or output-space transfer model, if this branch continues at all.

### N.11 HELM-v3 output-space transport observable follow-up

Because HELM-v2 still used a large raw stacked hidden observable, the next step on **April 9, 2026** was to replace that state with a compact output-space transport observable:

- the tracked hidden layers are still the last `helmHiddenStackDepth` hidden activations,
- but each layer is now transported into output space through the current downstream weights before entering the HELM predictor,
- so the past signal dimension shrinks from raw hidden width plus residual width to `outputDim * hiddenStackDepth + outputDim`,
- while the rest of the HELM controller, gating, and output-head-only correction stays the same.

This was intended to answer a narrower question than HELM-v2: was the real bottleneck the raw hidden observable itself, rather than the controller wrapped around it?

Verification:

- command: `cmake --build /home/robert/dev/glades-ml/build -j4`
- command: `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
- command: `./glades-unit-tests atlas-controller`
- command: `timeout 120s ./glades-unit-tests atlas`

Linear latent-state forecasting, HELM-v3 transport observable:

- command: `./glades-unit-tests atlas-alt-bench --mode latent-forecast --repeats 3`
- ATLAS-BSRP:
  - `train=2.17 +/- 0.01s`
  - `testMSE=0.00284 +/- 0.00022`
  - `testR2%=30.541 +/- 5.448`
- ATLAS-SPARROW:
  - `train=2.54 +/- 0.03s`
  - `testMSE=0.00266 +/- 0.00017`
  - `testR2%=34.980 +/- 4.159`
- ATLAS-HELM-v3:
  - `train=2.34 +/- 0.22s`
  - `testMSE=0.00290 +/- 0.00027`
  - `testR2%=29.105 +/- 6.579`
  - `activeModes=0.00 +/- 0.00`
  - `mode2Frac=0.000 +/- 0.000`
  - `edge2/1=0.526 +/- 0.121`
  - `edge=0.000 +/- 0.000`
  - `sigma=0.000 +/- 0.000`
  - `predR2=0.015 +/- 0.010`
  - `mem=0.000 +/- 0.000`
  - `pole=0.950 +/- 0.000`

Nonlinear latent-state forecasting, HELM-v3 transport observable:

- command: `./glades-unit-tests atlas-alt-bench --mode nonlinear-forecast --repeats 3`
- ATLAS-BSRP:
  - `train=3.94 +/- 0.00s`
  - `testMSE=0.00143 +/- 0.00009`
  - `testR2%=21.498 +/- 4.802`
- ATLAS-SPARROW:
  - `train=4.42 +/- 0.06s`
  - `testMSE=0.00159 +/- 0.00014`
  - `testR2%=12.295 +/- 7.634`
- ATLAS-HELM-v3:
  - `train=3.98 +/- 0.00s`
  - `testMSE=0.00144 +/- 0.00012`
  - `testR2%=20.596 +/- 6.720`
  - `activeModes=0.00 +/- 0.00`
  - `mode2Frac=0.000 +/- 0.000`
  - `edge2/1=0.078 +/- 0.054`
  - `edge=0.000 +/- 0.000`
  - `sigma=0.000 +/- 0.000`
  - `predR2=0.023 +/- 0.009`
  - `mem=0.000 +/- 0.000`
  - `pole=0.915 +/- 0.025`

Interpretation:

- This is a clean negative result. The output-space transport observable is more principled and much smaller than the raw HELM-v2 stacked hidden state, but it does **not** create a usable HELM mode on either dynamic task.
- In practice the numbers are almost unchanged from HELM-v2, including the most important diagnosis: `activeModes=0`, `edge≈0`, `sigma≈0`, and `mem≈0`.
- So the bottleneck is not merely “the hidden observable was too wide.” The present HELM line still fails earlier than that: the lightweight linear hidden-to-output predictor is not exposing a supercritical transfer spike, even after the hidden state is explicitly transported into output space.
- The nonlinear latent task still shows the same directional clue as before. HELM remains much closer to ATLAS-BSRP than SPARROW there, but the improvement comes from a branch that is still effectively declining to act.
- That narrows the next step further. If this branch continues, it should move to a **true reduced state-space or output-space transfer model**, not another hidden-observable repackaging inside the current HELM gate.
- That narrows the next step further. If this line continues, it should not go back to parameter-space complement geometry or more SPARROW mode tuning. It should strengthen the hidden/output observable family itself, for example with multi-layer hidden stacks, richer hidden-state targets, or a reduced hidden-to-output state-space realization.

### N.12 ASTER-Lite output-space state-space follow-up

The next step on **April 9, 2026** was to stop treating the hidden/output branch as another covariance gate and replace it with a true reduced output-space transfer model:

- `ASTER-Lite` transports the last two hidden layers into output space and treats them as controls,
- tracks the whitened batch-mean output innovation as the observed signal,
- fits a tiny reduced ARX / innovation map on `[prevResidual, prevControl, currentControl]`,
- extracts up to 2 output-space transfer modes,
- and applies only a bounded output-head correction from the predicted innovation.

This is the first prototype in this line that is genuinely different in kind from HELM. It is not another hidden observable reshaping inside the same gate; it is a reduced state-space predictor with explicit retained modes and latent poles.

Verification:

- command: `cmake --build /home/robert/dev/glades-ml/build -j4`
- command: `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
- command: `./glades-unit-tests atlas-controller`
- command: `timeout 120s ./glades-unit-tests atlas`

Linear latent-state forecasting, ASTER-Lite:

- command: `./glades-unit-tests atlas-alt-bench --mode latent-forecast --repeats 3`
- AdamW:
  - `train=2.11 +/- 0.01s`
  - `testMSE=0.00351 +/- 0.00013`
  - `testR2%=14.154 +/- 3.233`
- ATLAS-BSRP:
  - `train=2.23 +/- 0.00s`
  - `testMSE=0.00284 +/- 0.00022`
  - `testR2%=30.541 +/- 5.448`
- ATLAS-SPARROW:
  - `train=2.69 +/- 0.16s`
  - `testMSE=0.00266 +/- 0.00017`
  - `testR2%=34.980 +/- 4.159`
  - `activeModes=1.00 +/- 0.00`
  - `mode2Frac=0.000 +/- 0.000`
  - `edge=0.524 +/- 0.017`
  - `mem=0.024 +/- 0.001`
- ATLAS-HELM-v3:
  - `train=2.41 +/- 0.20s`
  - `testMSE=0.00290 +/- 0.00027`
  - `testR2%=29.105 +/- 6.579`
  - `activeModes=0.00 +/- 0.00`
  - `edge=0.000 +/- 0.000`
  - `mem=0.000 +/- 0.000`
- ATLAS-ASTER:
  - `train=2.27 +/- 0.00s`
  - `testMSE=0.00267 +/- 0.00016`
  - `testR2%=34.845 +/- 3.902`
  - `activeModes=1.94 +/- 0.00`
  - `mode2Frac=0.944 +/- 0.000`
  - `edge2/1=0.601 +/- 0.058`
  - `edge=0.277 +/- 0.027`
  - `sigma=0.323 +/- 0.024`
  - `predR2=0.036 +/- 0.043`
  - `mem=0.001 +/- 0.001`
  - `pole=0.871 +/- 0.112`

Nonlinear latent-state forecasting, ASTER-Lite:

- command: `./glades-unit-tests atlas-alt-bench --mode nonlinear-forecast --repeats 3`
- AdamW:
  - `train=3.86 +/- 0.01s`
  - `testMSE=0.00182 +/- 0.00005`
  - `testR2%=0.178 +/- 2.807`
- ATLAS-BSRP:
  - `train=4.07 +/- 0.01s`
  - `testMSE=0.00143 +/- 0.00009`
  - `testR2%=21.498 +/- 4.802`
- ATLAS-SPARROW:
  - `train=4.46 +/- 0.00s`
  - `testMSE=0.00159 +/- 0.00014`
  - `testR2%=12.295 +/- 7.634`
  - `activeModes=1.00 +/- 0.00`
  - `mode2Frac=0.000 +/- 0.000`
  - `edge=0.676 +/- 0.008`
  - `mem=0.032 +/- 0.000`
- ATLAS-HELM-v3:
  - `train=4.11 +/- 0.00s`
  - `testMSE=0.00144 +/- 0.00012`
  - `testR2%=20.596 +/- 6.720`
  - `activeModes=0.00 +/- 0.00`
  - `edge=0.000 +/- 0.000`
  - `mem=0.000 +/- 0.000`
- ATLAS-ASTER:
  - `train=4.11 +/- 0.00s`
  - `testMSE=0.00140 +/- 0.00014`
  - `testR2%=23.220 +/- 7.514`
  - `activeModes=1.43 +/- 0.34`
  - `mode2Frac=0.537 +/- 0.189`
  - `edge2/1=0.691 +/- 0.049`
  - `edge=0.147 +/- 0.019`
  - `sigma=0.180 +/- 0.021`
  - `predR2=0.114 +/- 0.081`
  - `mem=0.002 +/- 0.002`
  - `pole=0.950 +/- 0.000`

Interpretation:

- This is the first hidden/output branch that both stays bounded **and** activates nonzero retained modes on the positive dynamic tasks. Unlike HELM, ASTER is no longer voting “no observable state found.”
- On the linear latent task, ASTER is effectively tied with SPARROW on held-out error while running materially faster. It does not clearly beat SPARROW there, but it narrows the old observable-family gap to essentially zero.
- On the nonlinear latent task, ASTER is the strongest ATLAS-family result so far. It beats SPARROW clearly and edges past isotropic ATLAS-BSRP on held-out MSE while running at essentially the same speed as BSRP.
- The most important result is diagnostic, not just metric: `activeModes > 0`, nontrivial `mode2Frac`, positive `predR2`, and small but nonzero `mem` mean the output-space state-space observable is finally exposing a real predictive mode family instead of collapsing to a no-op.
- That changes the proceed path. The branch is now no longer “find a better parameter-space summary” or “tune HELM harder.” It is “strengthen ASTER,” for example with better reduced realization, better innovation filtering, or slightly richer output-space state models.

### N.13 ASTER-v2 explicit innovation-state update

The next step on **April 9, 2026** was to turn `ASTER-Lite` into a more explicit innovation state-space update:

- keep the existing output-space transfer-mode extraction,
- replace the old one-step latent recurrence with a true predict / innovate / filter / predict cycle,
- fit a small latent transition on `[x_{k-1}, u_{k-1}, u_k]`,
- fit a separate innovation gain from output residual surprise,
- and keep the actuation path output-head-only.

Implementation status after the profiling / root-cause follow-up:

- focused correctness checks still pass:
  - command: `./unit-tests/build/glades-unit-tests atlas-controller`
  - command result: `pass`
- the broader ATLAS suite also completes again:
  - command: `timeout 120s ./unit-tests/build/glades-unit-tests atlas`
  - command result: `pass`

Performance investigation result:

- the apparent ASTER-v2 “timeout regression” turned out **not** to be in the innovation-state update itself.
- isolated ASTER timing was added inside the DFF loop and the hot path was split into:
  - setup,
  - hidden-to-output transport,
  - transfer fit,
  - state fit,
  - innovation fit,
  - apply.
- isolated runs showed ASTER-local overhead is tiny:
  - `nonlinear-forecast`: `boundaryMs≈0.043`, `transportMs≈0.032`, `transferMs≈0.006`, `stateMs≈0.003`
  - `latent-forecast`: `boundaryMs≈0.026`, `transportMs≈0.016`, `transferMs≈0.006`, `stateMs≈0.003`
- `perf stat -d` on isolated `nonlinear-forecast` base vs ASTER agreed: ASTER adds only about `1-2%` task-clock with essentially unchanged IPC, branch-miss, and cache-miss behavior.
- the real regression was in the alternate benchmark harness: the regression-network factory was still returning `NNetwork` **by value**, and after the recent object/layout churn that path started throwing `std::bad_array_new_length` on the regression tasks.
- fixing the harness to use the same explicit owner pattern as the token benchmark restored the measured ASTER runs.

Repeated dynamic benchmark status after the harness fix:

- command: `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode latent-forecast --repeats 3 --variant all`
- command result:
  - `ATLAS-BSRP`: `train=2.38 +/- 0.20s`, `testMSE=0.00284 +/- 0.00022`
  - `ATLAS-SPARROW`: `train=2.58 +/- 0.01s`, `testMSE=0.00266 +/- 0.00017`
  - `ATLAS-ASTER`: `train=2.28 +/- 0.00s`, `testMSE=0.00267 +/- 0.00016`
- command: `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode nonlinear-forecast --repeats 3 --variant all`
- command result:
  - `ATLAS-BSRP`: `train=4.33 +/- 0.35s`, `testMSE=0.00143 +/- 0.00009`
  - `ATLAS-SPARROW`: `train=4.48 +/- 0.00s`, `testMSE=0.00159 +/- 0.00014`
  - `ATLAS-ASTER`: `train=4.14 +/- 0.00s`, `testMSE=0.00140 +/- 0.00014`
  - ASTER usage remained nontrivial: `activeModes=1.43 +/- 0.34`, `mode2Frac=0.537 +/- 0.189`, `predR2=0.056 +/- 0.057`
- command: `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode teacher-canonical --repeats 3`
- command result:
  - `signal-win`: `base=0.13618`, `sparrow-r1=0.10486`, `sparrow-auto=0.10432`
  - `default-anchor`: `base=0.05800`, `sparrow-r1=0.05847`, `sparrow-auto=0.06964`
  - `failure-band`: `base=0.17332`, `sparrow-r1=0.12703`, `sparrow-auto=0.17311`

Interpretation:

- ASTER-v2 is now a **validated** dynamic-task branch, not an unvalidated throughput-regression branch.
- On `latent-forecast`, ASTER remains effectively tied with SPARROW on held-out error while running materially faster.
- On `nonlinear-forecast`, ASTER remains the strongest branch so far: it beats both isotropic `ATLAS-BSRP` and SPARROW on held-out MSE while running at essentially the same speed as BSRP.
- The ASTER innovation/state update is not the practical runtime bottleneck; if an ASTER-side perf pass is ever needed, the only local hotspot worth touching is hidden-to-output transport.

Current recommendation:

- make `ASTER` the primary dynamic-task research branch,
- keep `SPARROW` as the control branch on `latent-forecast` and planted teacher cases,
- stop treating ASTER as a perf problem,
- and spend the next iteration on ASTER accuracy / robustness rather than new framework churn.

### N.14 ASTER-T transformer token-LM port

On **April 9, 2026**, the first transformer-side `ASTER` port was added to the decoder token-LM path:

- observable family:
  - fixed logit-space sketch over LM-head residual rows,
  - hidden-state transport from the final hidden state and the last decoder blocks,
  - small retained ASTER state on that sketched output process.
- actuation path:
  - tied LM head only (`tokE` / `lmBias`),
  - no parameter-space complement modeling,
  - no extra forward / backward pass in the minimal prototype.
- implementation status:
  - transformer-side ASTER state and diagnostics were added to `TensorTransformerState`,
  - token-LM backward now accumulates hidden/output ASTER statistics,
  - `atlas-alt-bench` token-LM runs now include `ATLAS-ASTER`,
  - a transformer token-LM ASTER smoke test was added to `atlas-test.cpp`.

Validation status:

- command: `cmake --build /home/robert/dev/glades-ml/build -j4`
  - command result: `pass`
- command: `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
  - command result: `pass`
- command: `./unit-tests/build/glades-unit-tests atlas-controller`
  - command result: `pass`
- command: `timeout 120s ./unit-tests/build/glades-unit-tests atlas`
  - command result: `timeout`, but this reproduces a broader long-running ATLAS-suite issue unrelated to the transformer ASTER path.

Because the default token-LM benchmark configuration is too heavy for short interactive runs, a reduced token-LM smoke benchmark was also added via new harness knobs for `seqLen`, `dModel`, `dFF`, `layers`, and `heads`.

Reduced token-LM smoke comparison on **April 9, 2026**:

- command:
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm --repeats 1 --token-train-seqs 4 --token-test-seqs 2 --token-epochs 1 --token-seq-len 16 --token-dmodel 16 --token-dff 32 --token-layers 1 --token-heads 1 --variant all`
- command result:
  - `AdamW`: `trainNLL=4.16915`, `testNLL=4.17474`
  - `ATLAS-BSRP`: `trainNLL=4.18018`, `testNLL=4.18314`
  - `ATLAS-SPARROW`: `trainNLL=4.17168`, `testNLL=4.16570`
  - `ATLAS-ASTER`: `trainNLL=4.17808`, `testNLL=4.16877`
  - ASTER usage:
    - `activeModes=2.00`
    - `mode2Frac=1.000`
    - `edge2/1=0.673`
    - `edge=0.168`
    - `sigma=0.203`
    - `predR2=0.331`
    - `mem=0.007`
    - `pole=-0.015`

Interpretation:

- This is the first transformer-side result showing that the ASTER-T port is not a no-op: it activates nonzero retained modes and produces bounded output-space state usage on token LM.
- On this tiny transformer smoke configuration, `SPARROW` still has the best held-out NLL, but `ASTER` is close and clearly better than isotropic `ATLAS-BSRP`.
- The important result is observability, not the exact ranking on this small smoke case: unlike the earlier parameter-space token-LM branch, ASTER-T exposes a real retained output-state process instead of collapsing to zero modes.
- The default token-LM harness remains too expensive for interactive repeated sweeps, so any serious transformer follow-up should either:
  - use the new reduced token-model knobs for iterative work, or
  - move to offline repeated runs on the default decoder benchmark.

### N.15 Larger token-LM transformer preset

To bridge the gap between the tiny smoke case and the still-expensive default token-LM benchmark, a dedicated larger transformer preset was added on **April 9, 2026**:

- mode:
  - `token-lm-large`
- preset:
  - `vocab=97`
  - `dModel=24`
  - `dFF=96`
  - `layers=2`
  - `heads=4`
  - `seqLen=24`
  - `trainSeqs=8`
  - `testSeqs=4`
  - `epochs=1`

This is intentionally larger than the ASTER-T smoke configuration but still small enough to complete quickly in the unit-test harness.

Observed one-repeat comparison on **April 9, 2026**:

- command:
  - `timeout 90s ./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-large --repeats 1 --variant all`
- command result:
  - `AdamW`: `testNLL=4.56885`, `testPPL=96.433`
  - `ATLAS-BSRP`: `testNLL=4.59081`, `testPPL=98.574`
  - `ATLAS-SPARROW`: `testNLL=4.57604`, `testPPL=97.129`
  - `ATLAS-ASTER`: `testNLL=4.56730`, `testPPL=96.284`
  - ASTER usage:
    - `activeModes=0.00`
    - `edge=0.090`
    - `sigma=0.107`
    - `predR2=0.793`
    - `mem=0.000`

Interpretation:

- On this first larger token-LM preset, `ASTER` is the strongest ATLAS-family variant on held-out NLL and slightly edges AdamW in this single run.
- Unlike the smaller ASTER-T smoke case, the larger preset lands in a near-threshold regime: `predR2` remains strong, but `edge` sits just below the current activation threshold, so the ASTER correction mostly stays gated off.
- That makes this preset useful as the next transformer-side tuning target: it is large enough to separate `ASTER` from `ATLAS-BSRP`, but still fast enough for iterative work.

Recommended next steps:

- Treat `token-lm-large` as the primary interactive transformer benchmark.
- Keep `ATLAS-BSRP` and `SPARROW` as controls, but make `ASTER` the main transformer research branch.
- Run repeated `token-lm-large` comparisons next, not more one-off single runs:
  - `repeats=3`
  - `variant all`
  - fixed seed ladder
- Center the next ASTER-T tuning pass on the near-threshold regime exposed here:
  - `asterEdgeThreshold` slightly below `0.10`
  - `asterMemoryScale` around the current default
  - no architecture churn until the repeated result is stable.
- Do not spend more time on the old default token-LM benchmark interactively; it is too expensive for the current harness budget and adds less information per iteration than `token-lm-large`.

### N.16 ASTER-T threshold / memory sweep on token-lm-large

Because the first repeated `token-lm-large` result placed ASTER just below the activation threshold (`edge≈0.096` vs threshold `0.10`), a small ASTER-only sweep was run on **April 9, 2026** to test whether the remaining gap was mostly gating or mostly model quality.

Observed repeated results:

- baseline repeated `token-lm-large`:
  - `ATLAS-ASTER` with `edgeThreshold=0.10`, `memoryScale=0.05`
  - `testNLL=4.57903 +/- 0.00832`
  - `activeModes=0.33 +/- 0.47`
  - `mem=0.000 +/- 0.000`

- lower threshold:
  - command:
    - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-large --repeats 3 --variant aster --atlas-aster-edge-threshold 0.08`
  - result:
    - `testNLL=4.57903 +/- 0.00832`
    - `activeModes=1.00 +/- 0.00`
    - `mem=0.006 +/- 0.002`

- lower threshold + lower memory gain:
  - command:
    - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-large --repeats 3 --variant aster --atlas-aster-edge-threshold 0.08 --atlas-aster-memory-scale 0.02`
  - result:
    - `testNLL=4.57903 +/- 0.00832`
    - `mem=0.002 +/- 0.001`

- lower threshold + higher memory gain:
  - command:
    - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-large --repeats 3 --variant aster --atlas-aster-edge-threshold 0.08 --atlas-aster-memory-scale 0.10`
  - result:
    - `testNLL=4.57903 +/- 0.00832`
    - `mem=0.012 +/- 0.003`

Interpretation:

- Lowering the threshold does exactly what it should operationally: ASTER activates consistently on `token-lm-large`.
- Varying `memoryScale` also changes the applied ASTER gain in the expected direction.
- But the held-out NLL stays numerically unchanged across the whole sweep.
- That means the remaining bottleneck is no longer “ASTER does not turn on” or “ASTER is too weak.” The bottleneck is the quality of the current transformer ASTER observable/state model.

Updated recommendation:

- Stop tuning `asterEdgeThreshold` and `asterMemoryScale` on this preset.
- Keep `token-lm-large` as the main interactive transformer benchmark.
- The next transformer-side work should change the ASTER-T state/observable model itself:
  - richer logit sketch,
  - different hidden transport summary,
  - or a stronger reduced innovation model.

### N.17 Structural ASTER-T bundle-plus-support revision

On **April 9, 2026**, the transformer ASTER-T branch was revised structurally rather than by further scalar tuning.

Revision:

- kept the persistent global logit sketch as the low-rank bundle observable,
- added exact support channels for:
  - the target residual,
  - the top hard-negative residual ranks inside each token step,
- kept hidden-state transport from the last decoder blocks,
- changed the ASTER-T state model to run on the combined observation:
  - `bundle sketch + exact support roles`,
- changed the ASTER correction path to split into:
  - bundle backprojection over the sketched residual bulk,
  - exact sparse support-row corrections for touched target / hard-negative rows.

Implementation notes:

- transformer ASTER state now tracks:
  - support-channel count,
  - last-block raw hidden means as well as sketch means,
  - sparse touched support ids plus per-role counts,
  - per-role exact hidden means for the output-head correction.
- the current unit-test harness did not need new CLI flags because the revision stayed under the existing `ATLAS-ASTER` variant.

Validation status:

- command:
  - `cmake --build /home/robert/dev/glades-ml/build -j4`
  - result: `pass`
- command:
  - `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
  - result: `pass`
- command:
  - `./unit-tests/build/glades-unit-tests atlas-controller`
  - result: `pass`
- command:
  - `timeout 120s ./unit-tests/build/glades-unit-tests atlas`
  - result: `pass`

Observed benchmark results after the structural revision:

- reduced token-LM smoke:
  - command:
    - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm --repeats 1 --token-train-seqs 4 --token-test-seqs 2 --token-epochs 1 --token-seq-len 16 --token-dmodel 16 --token-dff 32 --token-layers 1 --token-heads 1 --variant all`
  - result:
    - `AdamW`: `testNLL=4.17474`
    - `ATLAS-BSRP`: `4.18314`
    - `ATLAS-SPARROW`: `4.16570`
    - `ATLAS-ASTER`: `4.16876`
    - ASTER usage:
      - `activeModes=2.00`
      - `edge=0.377`
      - `sigma=0.409`
      - `predR2=0.910`
      - `mem=0.033`

- `token-lm-large`, repeated:
  - command:
    - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-large --repeats 3 --variant all`
  - result:
    - `AdamW`: `testNLL=4.57676 +/- 0.00637`
    - `ATLAS-BSRP`: `4.59059 +/- 0.00348`
    - `ATLAS-SPARROW`: `4.58331 +/- 0.01515`
    - `ATLAS-ASTER`: `4.57902 +/- 0.00832`
    - ASTER usage:
      - `activeModes=1.00 +/- 0.00`
      - `edge=0.537 +/- 0.141`
      - `sigma=0.544 +/- 0.139`
      - `predR2=0.968 +/- 0.009`
      - `mem=0.039 +/- 0.003`
      - `boundaryMs=0.691 +/- 0.026`

Interpretation:

- The structural revision did what the threshold sweep could not:
  - ASTER is no longer stuck in the near-threshold regime on `token-lm-large`.
  - Mode activation and applied gain are now clearly nonzero across repeats.
- But the ranking did **not** invert:
  - `ASTER` remains stronger than `ATLAS-BSRP`,
  - `ASTER` remains materially stronger than the old thresholded-no-op regime,
  - but `AdamW` is still slightly best on held-out NLL on this preset,
  - and `SPARROW` remains slightly better than `ASTER` on the tiny smoke case.

Updated recommendation:

- Keep `token-lm-large` as the primary interactive transformer benchmark.
- Keep `ATLAS-ASTER` as the main transformer research branch.
- Keep `ATLAS-BSRP` and `SPARROW` as controls.
- Do **not** go back to threshold / memory sweeps on transformer ASTER-T.
- The next useful transformer-side step, if the branch continues, should now target:
  - stronger sequence state,
  - regime conditioning,
  - or a richer output-space realization,
  not more scalar gate tuning.

### N.18 Lag-2 ASTER-T sequence-state revision

On **April 9, 2026**, the transformer ASTER-T branch was revised again to test a stronger sequence/state model rather than another observable or scalar-gate tweak.

Revision:

- expanded the ASTER-T transfer feature from a lag-1 ARX-style state to a lag-2 packed history:
  - past residual sketch at lags 1 and 2,
  - transported hidden controls at lags 0, 1, and 2,
- expanded the latent transition feature similarly:
  - latent state at lags 1 and 2,
  - transported controls at lags 0, 1, and 2,
- kept the bundle-plus-support observable family unchanged,
- kept output-head-only ASTER correction unchanged.

Implementation note:

- the token benchmark harness was also hardened to set the transformer runtime knobs explicitly
  (`kvCacheDType`, `tokenLmLossKind`, `layerNormEps`, dropout rates) instead of relying on inherited defaults.
  This avoided a harness-local `setTrainingConfig: unknown kvCacheDType` failure and did not change the intended benchmark semantics.

Validation status:

- command:
  - `cmake --build /home/robert/dev/glades-ml/build -j4`
  - result: `pass`
- command:
  - `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
  - result: `pass`
- command:
  - `./unit-tests/build/glades-unit-tests atlas-controller`
  - result: `pass`
- command:
  - `timeout 120s ./unit-tests/build/glades-unit-tests atlas`
  - result: `timeout`
  - note: this is the same long-running logging-heavy DFF path seen earlier; it did not block the focused transformer benchmark.

Observed benchmark result after the lag-2 state revision:

- `token-lm-large`, repeated:
  - command:
    - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-large --repeats 3 --variant all`
  - result:
    - `AdamW`: `testNLL=4.57676 +/- 0.00637`
    - `ATLAS-BSRP`: `4.59059 +/- 0.00348`
    - `ATLAS-SPARROW`: `4.58331 +/- 0.01515`
    - `ATLAS-ASTER`: `4.57903 +/- 0.00832`
    - ASTER usage:
      - `activeModes=1.00 +/- 0.00`
      - `edge=0.438 +/- 0.135`
      - `sigma=0.443 +/- 0.133`
      - `predR2=0.971 +/- 0.019`
      - `mem=0.036 +/- 0.004`
      - `boundaryMs=2.318 +/- 0.217`

Interpretation:

- The lag-2 state revision is **valid**:
  - ASTER still activates consistently,
  - predictive quality remains high,
  - the branch still beats `ATLAS-BSRP` cleanly on this preset.
- But it is **not** a ranking change:
  - held-out NLL is effectively unchanged from the prior structural ASTER-T branch,
  - `AdamW` still leads on `token-lm-large`,
  - `SPARROW` remains behind ASTER here.
- So the first stronger sequence-state upgrade did not unlock additional loss-critical signal.

Updated recommendation:

- Keep the bundle-plus-support ASTER-T branch as the transformer ATLAS baseline.
- Treat lag-2 ASTER-T as a neutral result, not a new default.
- If transformer ASTER continues, the next structural step should be:
  - regime conditioning,
  - or a richer output-space realization,
  not further linear lag expansion or scalar threshold tuning.

### N.19 Hard-bucket regime-conditioned ASTER-T

On **April 9, 2026**, the next transformer ASTER-T revision tested the simplest practical form of regime conditioning rather than a richer global linear state.

Revision:

- added `4` hard output regimes per token:
  - low entropy / high margin,
  - low entropy / low margin,
  - high entropy / high margin,
  - high entropy / low margin,
- accumulated ASTER bundle, support, hidden-transport, and latent-state statistics separately for each regime,
- selected the dominant regime at each ATLAS control boundary and ran the existing ASTER-T update on that regime-local slice,
- kept the bundle-plus-support output observable and output-head-only actuation unchanged.

Implementation note:

- the first benchmark run after the code change threw `std::bad_alloc`, but that turned out to be a stale `glades-unit-tests` binary after the transformer ASTER state layout change.
- after relinking `glades-unit-tests` against the rebuilt backend, the benchmark path was stable again.

Validation status:

- command:
  - `cmake --build /home/robert/dev/glades-ml/build -j4`
  - result: `pass`
- command:
  - `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
  - result: `pass`
- command:
  - `./unit-tests/build/glades-unit-tests atlas-controller`
  - result: `pass`
- command:
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm --repeats 1 --token-train-seqs 4 --token-test-seqs 2 --token-epochs 1 --token-seq-len 16 --token-dmodel 16 --token-dff 32 --token-layers 1 --token-heads 1 --variant all`
  - result: `pass`
- command:
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-large --repeats 3 --variant all`
  - result: `pass`

Observed benchmark results after the hard-bucket regime revision:

- reduced token-LM smoke:
  - result:
    - `AdamW`: `testNLL=4.17474`
    - `ATLAS-BSRP`: `4.18314`
    - `ATLAS-SPARROW`: `4.16570`
    - `ATLAS-ASTER`: `4.16877`
    - ASTER usage:
      - `activeModes=2.00`
      - `edge=0.373`
      - `sigma=0.388`
      - `predR2=0.875`
      - `mem=0.032`

- `token-lm-large`, repeated:
  - result:
    - `AdamW`: `testNLL=4.57676 +/- 0.00637`
    - `ATLAS-BSRP`: `4.59059 +/- 0.00348`
    - `ATLAS-SPARROW`: `4.58331 +/- 0.01515`
    - `ATLAS-ASTER`: `4.57903 +/- 0.00832`
    - ASTER usage:
      - `activeModes=1.00 +/- 0.00`
      - `edge=0.438 +/- 0.135`
      - `sigma=0.443 +/- 0.133`
      - `predR2=0.971 +/- 0.019`
      - `mem=0.036 +/- 0.004`
      - `boundaryMs=2.164 +/- 0.186`

Interpretation:

- The minimal hard-bucket regime conditioning is a **valid implementation**, not a no-op:
  - ASTER still activates cleanly,
  - bounded per-boundary cost remains small,
  - the branch still beats `ATLAS-BSRP` on the larger token-LM preset.
- But it is **not** a quality breakthrough:
  - held-out NLL on `token-lm-large` is effectively unchanged from the prior lag-2 / global ASTER-T branch,
  - `AdamW` still leads,
  - the current dominant-regime approximation does not unlock additional loss-critical signal.

Updated recommendation:

- Keep global bundle-plus-support ASTER-T as the transformer ATLAS baseline.
- Treat this first hard-bucket regime-conditioned ASTER as a neutral result, not a new default.
- If transformer ASTER continues, the next structural step should be:
  - a true multi-expert mixture instead of dominant-regime selection,
  - or a richer output-space realization inside each regime,
  not more threshold tuning.

### N.20 True multi-expert MOSAIC-ASTER mixture

On **April 10, 2026**, the dominant-regime ASTER-T approximation was replaced by the first true multi-expert mixture pass.

Revision:

- kept the same `4` hard output regimes:
  - low entropy / high margin,
  - low entropy / low margin,
  - high entropy / high margin,
  - high entropy / low margin,
- kept the same regime-local state slices and bundle-plus-support observable family,
- changed the ASTER boundary update from:
  - pick the single dominant regime and ignore the rest,
  to:
  - run every nonempty regime expert,
  - apply every regime-local correction,
  - aggregate ASTER diagnostics across regimes instead of reporting only the dominant one.

Implementation note:

- the transformer ASTER timing buckets were also adjusted so setup / transport / fit times remain meaningful under the per-regime loop instead of double-counting elapsed time from the outer boundary start.

Validation status:

- command:
  - `cmake --build /home/robert/dev/glades-ml/build -j4`
  - result: `pass`
- command:
  - `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
  - result: `pass`
- command:
  - `./unit-tests/build/glades-unit-tests atlas-controller`
  - result: `pass`
- command:
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm --repeats 1 --token-train-seqs 4 --token-test-seqs 2 --token-epochs 1 --token-seq-len 16 --token-dmodel 16 --token-dff 32 --token-layers 1 --token-heads 1 --variant all`
  - result: `pass`
- command:
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-large --repeats 3 --variant all`
  - result: `pass`

Observed benchmark results after the true mixture revision:

- reduced token-LM smoke:
  - result:
    - `AdamW`: `testNLL=4.17474`
    - `ATLAS-BSRP`: `4.18314`
    - `ATLAS-SPARROW`: `4.16570`
    - `ATLAS-ASTER`: `4.16877`
    - ASTER usage:
      - `activeModes=2.00`
      - `edge=0.373`
      - `sigma=0.388`
      - `predR2=0.875`
      - `mem=0.032`
      - `boundaryMs=0.561`

- `token-lm-large`, repeated:
  - result:
    - `AdamW`: `testNLL=4.57676 +/- 0.00637`
    - `ATLAS-BSRP`: `4.59059 +/- 0.00348`
    - `ATLAS-SPARROW`: `4.58331 +/- 0.01515`
    - `ATLAS-ASTER`: `4.57903 +/- 0.00832`
    - ASTER usage:
      - `activeModes=1.00 +/- 0.00`
      - `edge=0.438 +/- 0.135`
      - `sigma=0.443 +/- 0.133`
      - `predR2=0.971 +/- 0.019`
      - `mem=0.036 +/- 0.004`
      - `boundaryMs=2.149 +/- 0.203`

Interpretation:

- The first true multi-expert MOSAIC-ASTER implementation is a **valid systems result**:
  - all experts run cleanly,
  - the aggregated diagnostics stay bounded,
  - the transformer ASTER branch still beats `ATLAS-BSRP`.
- But it is **not** a quality breakthrough:
  - the held-out `token-lm-large` result is numerically unchanged from the dominant-regime approximation,
  - `AdamW` still leads,
  - the extra expert coverage does not convert the already strong ASTER predictive signal into better next-token loss.

Updated recommendation:

- Treat the regime-conditioning hypothesis as largely falsified in its current hard-bucket / linear-expert form.
- Keep the current ASTER-T mixture as a documented neutral result, not a new default.
- If transformer ASTER continues, the next useful structural step should be:
  - a richer output-space realization inside each regime,
  - or a token-conditioned / switched state model with a meaningfully different observable family,
  not more threshold tuning and not more copies of the same linear expert.

### N.21 Token-lm-large 10-repeat significance pass

On **April 10, 2026**, the larger transformer preset was rerun with `repeats=10` before making another ASTER-T architecture change.

Command:

- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-large --repeats 10 --variant all`

Observed result:

- `AdamW`: `testNLL=4.58178 +/- 0.00977`
- `ATLAS-BSRP`: `4.58258 +/- 0.01477`
- `ATLAS-SPARROW`: `4.57641 +/- 0.01124`
- `ATLAS-ASTER`: `4.58045 +/- 0.00787`

Interpretation:

- the earlier `repeats=3` picture was too optimistic for ASTER-T,
- on the current interactive transformer preset, `SPARROW` is the best branch,
- `ASTER` is still better than `BSRP`, but it no longer looks like the lead transformer branch on this benchmark.

Updated recommendation:

- use this 10-repeat result as the transformer significance baseline,
- require any future ASTER-T revision to beat `ATLAS-SPARROW`, not just `ATLAS-BSRP`,
- stop treating the tiny gaps to AdamW or ASTER’s activation diagnostics as sufficient evidence on their own.

### N.22 Margin-state ASTER-T revision

On **April 10, 2026**, the next ASTER-T structural revision replaced the exact support-residual channels with a more loss-aligned support state:

- kept the bundle residual sketch for low-rank output bulk,
- replaced support residual channels by exact support logits,
- added explicit target-minus-negative margin channels derived from those exact support logits,
- kept the multi-expert regime mixture and output-head-only actuation unchanged.

The intention was to make ASTER predict target-margin dynamics directly rather than approximate them through residual support roles.

Validation status:

- command:
  - `cmake --build /home/robert/dev/glades-ml/build -j4`
  - result: `pass`
- command:
  - `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
  - result: `pass`
- command:
  - `./unit-tests/build/glades-unit-tests atlas-controller`
  - result: `pass`
- command:
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm --repeats 1 --token-train-seqs 4 --token-test-seqs 2 --token-epochs 1 --token-seq-len 16 --token-dmodel 16 --token-dff 32 --token-layers 1 --token-heads 1 --variant all`
  - result: `pass`
- command:
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-large --repeats 10 --variant all`
  - result: `pass`

Observed benchmark result after the margin-state revision:

- reduced token-LM smoke:
  - `AdamW`: `testNLL=4.17474`
  - `ATLAS-BSRP`: `4.18314`
  - `ATLAS-SPARROW`: `4.16570`
  - `ATLAS-ASTER`: `4.16878`
  - ASTER usage:
    - `edge=0.186`
    - `sigma=0.216`
    - `predR2=0.563`
    - `mem=0.013`
    - `boundaryMs=0.906`

- `token-lm-large`, repeated:
  - `AdamW`: `testNLL=4.58178 +/- 0.00977`
  - `ATLAS-BSRP`: `4.58258 +/- 0.01477`
  - `ATLAS-SPARROW`: `4.57641 +/- 0.01124`
  - `ATLAS-ASTER`: `4.58045 +/- 0.00787`
  - ASTER usage:
    - `edge=0.258 +/- 0.066`
    - `sigma=0.265 +/- 0.064`
    - `predR2=0.896 +/- 0.079`
    - `mem=0.026 +/- 0.004`
    - `boundaryMs=3.626 +/- 0.251`

Interpretation:

- This is a clean negative result for the margin-state hypothesis in its current lightweight form.
- Held-out NLL is effectively unchanged from the pre-revision ASTER-T significance run.
- The revision reduced ASTER’s internal edge/sigma signal and made it slower, without improving ranking.
- So replacing support residual roles with exact support logits plus explicit margins did **not** recover the missing transformer-side loss signal.

Updated recommendation:

- stop local ASTER-T tuning on `token-lm-large`,
- keep `SPARROW` as the best current transformer-side ATLAS control on this preset,
- only resume ASTER-T transformer work if the next step is materially different:
  - token-conditioned or attention-state observables,
  - a larger / more realistic LM benchmark,
  - or a richer function-side model that is not another small linear observable tweak.

### N.23 Structured-context transformer benchmark

On **April 10, 2026**, the alternate benchmark harness was extended with a new transformer LM mode, `token-lm-context`, to replace the old "larger preset on the same order-2 recurrence" escape hatch with a genuinely different sequence structure.

Implementation:

- added a new `atlas-alt-bench` mode:
  - `token-lm-context`
- added a dedicated preset:
  - `vocab=129`
  - `dModel=32`
  - `dFF=128`
  - `layers=2`
  - `heads=4`
  - `seqLen=56`
  - `trainSeqs=24`
  - `testSeqs=8`
  - `epochs=2`
- replaced the generator for this mode with a structured token program made of repeated segments containing:
  - topic markers,
  - delayed summary recall markers,
  - delayed anchor recall markers,
  - local continuation tokens,
  - and separator/store control tokens.

The point of the new mode is not "more tokens." It is to force the decoder to mix:

- short-horizon local prediction,
- marker-conditioned recall,
- topic-conditioned content generation,
- and cross-segment memory.

Validation:

- command:
  - `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
  - result: `pass`
- command:
  - `./unit-tests/build/glades-unit-tests atlas-controller`
  - result: `pass`
- command:
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-context --repeats 3 --variant all`
  - result: `pass`

Observed benchmark result:

- `AdamW`: `testNLL=4.22251 +/- 0.01795`
- `ATLAS-BSRP`: `4.81371 +/- 0.01429`
- `ATLAS-SPARROW`: `4.79139 +/- 0.00754`
- `ATLAS-ASTER`: `4.80455 +/- 0.01402`

Diagnostics:

- `ATLAS-SPARROW`:
  - `edge=0.477 +/- 0.018`
  - `mem=0.021 +/- 0.001`
- `ATLAS-ASTER`:
  - `activeModes=1.83 +/- 0.24`
  - `mode2Frac=0.833 +/- 0.236`
  - `edge=0.474 +/- 0.103`
  - `sigma=0.510 +/- 0.100`
  - `predR2=0.892 +/- 0.047`
  - `mem=0.034 +/- 0.004`
  - `boundaryMs=4.748 +/- 0.078`

Interpretation:

- This benchmark is materially harsher than `token-lm-large`; the old near-tie between AdamW and the best ATLAS branch disappears here.
- On this more realistic structured-context task, `AdamW` is clearly best.
- Within the ATLAS family, `SPARROW` is stronger than `ASTER`, and both are materially better than isotropic `ATLAS-BSRP`.
- ASTER still activates real modes here, so the result is not "ASTER failed to turn on." It is "the current ASTER transformer observable/control still does not beat the cheaper SPARROW branch once the task demands stronger context use."

Updated recommendation:

- use `token-lm-context` as the primary interactive transformer benchmark going forward,
- keep `token-lm-large` as a lighter continuity/control preset,
- keep `SPARROW` as the best current transformer-side ATLAS branch on the available token-LM tasks,
- stop treating transformer ASTER as the lead branch unless a materially different observable family beats SPARROW on `token-lm-context`.

### N.24 AEGIS minimal fusion optimizer

On **April 10, 2026**, the first minimal implementation of the proposed unified optimizer, **AEGIS** (`Adaptive Evidence-Gated Integrated Subspaces`), was added as a new ATLAS-family branch.

Implementation scope:

- AEGIS is not a separate optimizer stack. It is a narrow fusion branch inside the existing ATLAS path.
- The minimal prototype keeps:
  - `ATLAS-BSRP` as the spatial base,
  - `SPARROW` enabled on the active/scout parameter-space channel,
  - `ASTER` enabled on the output-space innovation channel.
- The first fusion rule is intentionally simple:
  - compute ASTER's usual output-space gain,
  - measure prior SPARROW evidence on the same head/update path,
  - attenuate ASTER's applied gain when predictive parameter-space evidence is already stronger.
- This gives a falsifiable first fusion branch without introducing a separate optimizer state stack beyond the existing ATLAS/SPARROW/ASTER components.

Implementation notes:

- new ATLAS config keys:
  - `training.atlas.aegisEnabled`
  - `training.atlas.aegisPredictiveScale`
  - `training.atlas.aegisOutputScale`
- checkpoint parse/write/mismatch validation was extended to include those keys.
- `atlas-alt-bench` now supports:
  - `--variant aegis`
  - `ATLAS-AEGIS` in `--variant all`

Validation:

- command:
  - `cmake --build /home/robert/dev/glades-ml/build -j4`
  - result: `pass`
- command:
  - `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
  - result: `pass`
- command:
  - `./unit-tests/build/glades-unit-tests atlas-controller`
  - result: `pass`
- command:
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode teacher-student --repeats 3 --variant all`
  - result: `pass`
- command:
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode latent-forecast --repeats 3 --variant all`
  - result: `pass`
- command:
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode nonlinear-forecast --repeats 3 --variant all`
  - result: `pass`
- command:
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-large --repeats 10 --variant all`
  - result: `pass`
- command:
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-context --repeats 10 --variant all`
  - result: `pass`

Observed benchmark result:

- `teacher-student`, repeated:
  - `AdamW`: `testMSE=0.07863 +/- 0.00278`
  - `ATLAS-BSRP`: `0.05933 +/- 0.01585`
  - `ATLAS-SPARROW`: `0.04640 +/- 0.00142`
  - `ATLAS-AEGIS`: `0.05842 +/- 0.01678`

- `latent-forecast`, repeated:
  - `AdamW`: `testMSE=0.00351 +/- 0.00013`
  - `ATLAS-BSRP`: `0.00284 +/- 0.00022`
  - `ATLAS-SPARROW`: `0.00266 +/- 0.00017`
  - `ATLAS-ASTER`: `0.00267 +/- 0.00016`
  - `ATLAS-AEGIS`: `0.00267 +/- 0.00015`

- `nonlinear-forecast`, repeated:
  - `AdamW`: `testMSE=0.00182 +/- 0.00005`
  - `ATLAS-BSRP`: `0.00143 +/- 0.00009`
  - `ATLAS-SPARROW`: `0.00159 +/- 0.00014`
  - `ATLAS-ASTER`: `0.00140 +/- 0.00014`
  - `ATLAS-AEGIS`: `0.00138 +/- 0.00011`

- `token-lm-large`, repeated:
  - `AdamW`: `testNLL=4.58178 +/- 0.00977`
  - `ATLAS-BSRP`: `4.58258 +/- 0.01477`
  - `ATLAS-SPARROW`: `4.57641 +/- 0.01124`
  - `ATLAS-ASTER`: `4.58045 +/- 0.00787`
  - `ATLAS-AEGIS`: `4.57398 +/- 0.01308`

- `token-lm-context`, repeated:
  - `AdamW`: `testNLL=4.24529 +/- 0.03627`
  - `ATLAS-BSRP`: `4.80616 +/- 0.01737`
  - `ATLAS-SPARROW`: `4.79550 +/- 0.01327`
  - `ATLAS-ASTER`: `4.80677 +/- 0.01414`
  - `ATLAS-AEGIS`: `4.79785 +/- 0.02182`

Diagnostics:

- On `token-lm-large`, AEGIS keeps the strong SPARROW signal but only a small ASTER correction:
  - `SPARROW edge=0.701 +/- 0.020`
  - `SPARROW mem=0.034 +/- 0.001`
  - `ASTER edge=0.194 +/- 0.037`
  - `ASTER predR2=0.932 +/- 0.029`
  - `ASTER mem=0.005 +/- 0.002`
- On `token-lm-context`, AEGIS again retains both channels but with ASTER attenuated relative to standalone ASTER:
  - `SPARROW edge=0.473 +/- 0.036`
  - `SPARROW mem=0.021 +/- 0.002`
  - `ASTER edge=0.389 +/- 0.059`
  - `ASTER predR2=0.889 +/- 0.057`
  - `ASTER mem=0.013 +/- 0.003`
- On `latent-forecast` and `nonlinear-forecast`, the fusion mostly leaves SPARROW intact while shrinking ASTER to a very small or zero applied gain. That is exactly the intended conservative behavior of the first AEGIS prototype.

Interpretation:

- This is the first integrated optimizer branch in the ATLAS line that shows nontrivial wins across more than one task family.
- AEGIS is **not** universally better:
  - it is clearly worse than SPARROW on the planted `teacher-student` benchmark,
  - essentially tied with the best existing branch on `latent-forecast`,
  - slightly better than ASTER on `nonlinear-forecast`,
  - best ATLAS-family branch on `token-lm-large`,
  - but still worse than AdamW on `token-lm-context`.
- The main lesson is that the design premise is correct:
  - `SPARROW` and `ASTER` should be treated as complementary sensors, not mutually exclusive optimizers.
- The minimal evidence gate is already strong enough to prevent ASTER from hurting the transformer branch as much as it did standalone, while preserving enough output-space signal to help on nonlinear dynamics.

Updated recommendation:

- keep `AEGIS` as the main **unified** research branch,
- keep `SPARROW` as the best single-branch control on planted / easier dynamic / current structured-context transformer tasks,
- keep `ASTER` as the nonlinear dynamic specialist branch and output-space control,
- do **not** yet replace AdamW as the robustness anchor on harder transformer-context tasks,
- next AEGIS work should improve evidence calibration and channel selection, not reopen standalone ASTER threshold tuning.

### N.25 AEGIS-v2 delayed evidence calibration

On **April 10, 2026**, the minimal AEGIS fusion rule was upgraded to **AEGIS-v2**, replacing the original one-shot ASTER attenuation with an explicit delayed-calibration scheme.

Implementation scope:

- kept the optimizer family fixed:
  - `AdamW`-style base stability,
  - `ATLAS-BSRP` spatial channel,
  - `SPARROW` predictive channel,
  - `ASTER` output-space channel.
- added explicit bounded channel precisions:
  - `lambdaSpatial`
  - `lambdaPredictive`
  - `lambdaOutput`
- updated those precisions from delayed calibration error rather than raw edge alone:
  - previous predictive/output evidence is stored,
  - next-boundary realized evidence is compared against it,
  - predictive/output error EMAs are updated,
  - channel precisions are normalized before applying ASTER gain.
- extended runtime diagnostics and the benchmark harness to report:
  - lambda means,
  - predicted vs realized predictive/output evidence,
  - predictive/output calibration errors,
  - channel disagreement.

Validation:

- command:
  - `cmake --build /home/robert/dev/glades-ml/build -j4`
  - result: `pass`
- command:
  - `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
  - result: `pass`
- command:
  - `./unit-tests/build/glades-unit-tests atlas-controller`
  - result: `pass`
- command:
  - `timeout 120s ./unit-tests/build/glades-unit-tests atlas`
  - result: `pass`
- command:
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode teacher-student --repeats 3 --variant all`
  - result: `pass`
- command:
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode latent-forecast --repeats 3 --variant all`
  - result: `pass`
- command:
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode nonlinear-forecast --repeats 3 --variant all`
  - result: `pass`
- command:
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-large --repeats 10 --variant all`
  - result: `pass`
- command:
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-context --repeats 10 --variant all`
  - result: `pass`

Observed benchmark result:

- `teacher-student`, repeated:
  - `AdamW`: `testMSE=0.07863 +/- 0.00278`
  - `ATLAS-BSRP`: `0.05933 +/- 0.01585`
  - `ATLAS-SPARROW`: `0.04640 +/- 0.00142`
  - `ATLAS-AEGIS`: `0.05841 +/- 0.01678`

- `latent-forecast`, repeated:
  - `AdamW`: `testMSE=0.00351 +/- 0.00013`
  - `ATLAS-BSRP`: `0.00284 +/- 0.00022`
  - `ATLAS-SPARROW`: `0.00266 +/- 0.00017`
  - `ATLAS-ASTER`: `0.00267 +/- 0.00016`
  - `ATLAS-AEGIS`: `0.00267 +/- 0.00015`

- `nonlinear-forecast`, repeated:
  - `AdamW`: `testMSE=0.00182 +/- 0.00005`
  - `ATLAS-BSRP`: `0.00143 +/- 0.00009`
  - `ATLAS-SPARROW`: `0.00159 +/- 0.00014`
  - `ATLAS-ASTER`: `0.00140 +/- 0.00014`
  - `ATLAS-AEGIS`: `0.00138 +/- 0.00011`

- `token-lm-large`, repeated:
  - `AdamW`: `testNLL=4.58178 +/- 0.00977`
  - `ATLAS-BSRP`: `4.58258 +/- 0.01477`
  - `ATLAS-SPARROW`: `4.57641 +/- 0.01124`
  - `ATLAS-ASTER`: `4.58045 +/- 0.00787`
  - `ATLAS-AEGIS`: `4.57398 +/- 0.01308`

- `token-lm-context`, repeated:
  - `AdamW`: `testNLL=4.24529 +/- 0.03627`
  - `ATLAS-BSRP`: `4.80616 +/- 0.01737`
  - `ATLAS-SPARROW`: `4.79550 +/- 0.01327`
  - `ATLAS-ASTER`: `4.80677 +/- 0.01414`
  - `ATLAS-AEGIS`: `4.79782 +/- 0.02182`

Diagnostics:

- On planted and DFF dynamic tasks, AEGIS-v2 remains strongly predictive-dominant:
  - `teacher-student`:
    - `lambdaSpatial=0.088 +/- 0.022`
    - `lambdaPredictive=0.889 +/- 0.040`
    - `lambdaOutput=0.023 +/- 0.018`
  - `latent-forecast`:
    - `lambdaSpatial=0.057 +/- 0.004`
    - `lambdaPredictive=0.919 +/- 0.008`
    - `lambdaOutput=0.024 +/- 0.004`
  - `nonlinear-forecast`:
    - `lambdaSpatial=0.065 +/- 0.011`
    - `lambdaPredictive=0.931 +/- 0.014`
    - `lambdaOutput=0.005 +/- 0.003`
- On `token-lm-large`, AEGIS-v2 becomes a genuinely balanced two-sensor fusion:
  - `lambdaSpatial=0.081 +/- 0.010`
  - `lambdaPredictive=0.461 +/- 0.022`
  - `lambdaOutput=0.457 +/- 0.028`
  - predictive calibration:
    - `predicted=0.675 +/- 0.043`
    - `realized=0.639 +/- 0.044`
    - `error=0.087 +/- 0.004`
  - output calibration:
    - `predicted=0.168 +/- 0.033`
    - `realized=0.181 +/- 0.038`
    - `error=0.007 +/- 0.002`
- On `token-lm-context`, both non-spatial channels remain active and reasonably calibrated:
  - `lambdaSpatial=0.046 +/- 0.004`
  - `lambdaPredictive=0.570 +/- 0.056`
  - `lambdaOutput=0.384 +/- 0.058`
  - predictive calibration:
    - `predicted=0.508 +/- 0.046`
    - `realized=0.507 +/- 0.046`
    - `error=0.016 +/- 0.002`
  - output calibration:
    - `predicted=0.338 +/- 0.067`
    - `realized=0.345 +/- 0.067`
    - `error=0.014 +/- 0.002`

Interpretation:

- AEGIS-v2 is a **real calibration improvement**, but not yet a broad frontier shift.
- The topline ranking is basically unchanged from the minimal AEGIS prototype:
  - still clearly worse than SPARROW on `teacher-student`,
  - still tied with the best branch on `latent-forecast`,
  - still best on `nonlinear-forecast`,
  - still best ATLAS-family branch on `token-lm-large`,
  - still behind `AdamW` and slightly behind SPARROW on `token-lm-context`.
- The important new lesson is diagnostic, not just numeric:
  - on DFF tasks, AEGIS correctly learns that the predictive channel is the only one worth trusting,
  - on `token-lm-large`, AEGIS now shows that the predictive and output channels are comparably credible,
  - on `token-lm-context`, both channels are well calibrated but still do not close the much larger gap to `AdamW`.
- That means the remaining transformer problem is no longer "bad evidence weighting." It is the quality of the underlying observable/state models on the harder structured-context task.

Updated recommendation:

- keep `AEGIS-v2` as the main unified optimizer branch,
- treat the new lambda diagnostics as the primary signal for deciding which channel family is actually contributing,
- stop tuning scalar AEGIS weights locally,
- if transformer work continues, change the transformer-side observable/state model inside AEGIS rather than changing the fusion math again,
- keep `SPARROW` as the planted/easier-dynamics control and `AdamW` as the robustness anchor on `token-lm-context`.

### N.26 Transformer AEGIS attention-state observable revision

On **April 10, 2026**, the transformer-side ASTER sensor inside `AEGIS-v2` was structurally widened rather than retuned.

Implementation summary:

- kept the `AEGIS-v2` fusion math and delayed evidence calibration unchanged,
- added a second transformer control stream for the last tracked decoder blocks:
  - existing stream: hidden-state transport,
  - new stream: attention-output transport,
- expanded transformer ASTER control construction so the output-space sensor sees both hidden and attention summaries at each boundary,
- kept the correction head-only on the tied LM head.

Code surface:

- transformer ASTER state buffers:
  - [transformer_model_state.inc](/home/robert/dev/glades-ml/Backend/Machine%20Learning/Networks/transformer_model_state.inc)
- transformer ASTER initialization:
  - [network.cpp](/home/robert/dev/glades-ml/Backend/Machine%20Learning/Networks/network.cpp)
- transformer ASTER/AEGIS control accumulation and boundary update:
  - [sgd_transformer.cpp](/home/robert/dev/glades-ml/Backend/Machine%20Learning/Networks/sgd_transformer.cpp)

Validation:

- command:
  - `cmake --build /home/robert/dev/glades-ml/build -j4`
  - result: `pass`
- command:
  - `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
  - result: `pass`
- command:
  - `./unit-tests/build/glades-unit-tests atlas-controller`
  - result: `pass`
- command:
  - `timeout 120s ./unit-tests/build/glades-unit-tests atlas`
  - result: `pass`
- command:
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-large --repeats 10 --variant all`
  - result: `pass`
- command:
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-context --repeats 10 --variant all`
  - result: `pass`

Observed benchmark result:

- `token-lm-large`, repeated:
  - `AdamW`: `testNLL=4.58178 +/- 0.00977`
  - `ATLAS-BSRP`: `4.58258 +/- 0.01477`
  - `ATLAS-SPARROW`: `4.57641 +/- 0.01124`
  - `ATLAS-ASTER`: `4.58045 +/- 0.00787`
  - `ATLAS-AEGIS`: `4.57398 +/- 0.01308`

- `token-lm-context`, repeated:
  - `AdamW`: `testNLL=4.24529 +/- 0.03627`
  - `ATLAS-BSRP`: `4.80616 +/- 0.01737`
  - `ATLAS-SPARROW`: `4.79550 +/- 0.01327`
  - `ATLAS-ASTER`: `4.80678 +/- 0.01417`
  - `ATLAS-AEGIS`: `4.79780 +/- 0.02182`

Diagnostics:

- `token-lm-large`:
  - `AEGIS` channel trust:
    - `lambdaSpatial=0.085 +/- 0.009`
    - `lambdaPredictive=0.494 +/- 0.017`
    - `lambdaOutput=0.420 +/- 0.019`
  - calibration:
    - predictive:
      - `predicted=0.676 +/- 0.046`
      - `realized=0.648 +/- 0.048`
      - `error=0.086 +/- 0.004`
    - output:
      - `predicted=0.142 +/- 0.024`
      - `realized=0.154 +/- 0.030`
      - `error=0.006 +/- 0.003`

- `token-lm-context`:
  - `AEGIS` channel trust:
    - `lambdaSpatial=0.049 +/- 0.004`
    - `lambdaPredictive=0.612 +/- 0.048`
    - `lambdaOutput=0.338 +/- 0.048`
  - calibration:
    - predictive:
      - `predicted=0.508 +/- 0.046`
      - `realized=0.507 +/- 0.045`
      - `error=0.016 +/- 0.001`
    - output:
      - `predicted=0.244 +/- 0.040`
      - `realized=0.249 +/- 0.042`
      - `error=0.010 +/- 0.001`

Interpretation:

- This is a **clean neutral result**.
- The added attention-state observable did **not** move the benchmark ranking:
  - `AEGIS` remains best on `token-lm-large`,
  - `AEGIS` remains behind `SPARROW` and far behind `AdamW` on `token-lm-context`.
- The new transformer sensor also did not unlock a hidden output-channel surge:
  - on `token-lm-large`, the two non-spatial channels remain balanced,
  - on `token-lm-context`, the predictive channel is still dominant even after exposing attention-state transport.
- So the remaining transformer gap is not explained by "missing attention-state information" alone.
- The most important result is narrowing:
  - `AEGIS-v2` fusion is already stable,
  - adding last-block attention-output transport is not enough,
  - the next transformer advance would need a materially richer token-conditioned or attention-pattern observable, not another small global control-stream expansion.

Updated recommendation:

- keep the current transformer `AEGIS-v2` implementation as the documented baseline,
- stop local transformer observable tweaks of the same class,
- keep `token-lm-large` as the light repeated control and `token-lm-context` as the primary transformer stress benchmark,
- if transformer work continues, move to a materially different observable family:
  - token-conditioned output observables,
  - attention-pattern / key-value state observables,
  - or a larger, more realistic LM benchmark before more optimizer-side refinements.

### N.27 Transformer AEGIS token-conditioned output + attention-pattern revision

On **April 10, 2026**, the transformer-side `AEGIS-v2` sensor was widened again, this time with a more explicitly token-conditioned observable and a separate attention-pattern control stream.

Implementation summary:

- kept the `AEGIS-v2` fusion math unchanged,
- added a small token-conditioned tail to the transformer ASTER/AEGIS output observable,
- accumulated exact support-role residual mass into that token-conditioned tail,
- added a third transformer ASTER control stream for attention-pattern summaries from the last tracked decoder blocks,
- introduced a harsher transformer benchmark, `token-lm-context-large`, to test whether the richer observable helps under longer-context delayed recall pressure.

Code surface:

- transformer ASTER / AEGIS state:
  - [transformer_model_state.inc](/home/robert/dev/glades-ml/Backend/Machine%20Learning/Networks/transformer_model_state.inc)
- transformer ASTER / AEGIS initialization:
  - [network.cpp](/home/robert/dev/glades-ml/Backend/Machine%20Learning/Networks/network.cpp)
- transformer token-conditioned residual / attention-pattern accumulation and boundary update:
  - [sgd_transformer.cpp](/home/robert/dev/glades-ml/Backend/Machine%20Learning/Networks/sgd_transformer.cpp)
- transformer alternate benchmark harness:
  - [atlas-alt-bench.cpp](/home/robert/dev/glades-ml/unit-tests/Backend/Machine%20Learning/atlas-alt-bench.cpp)

Validation:

- command:
  - `cmake --build /home/robert/dev/glades-ml/build -j4`
  - result: `pass`
- command:
  - `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
  - result: `pass`
- command:
  - `./unit-tests/build/glades-unit-tests atlas-controller`
  - result: `pass`
- command:
  - `timeout 120s ./unit-tests/build/glades-unit-tests atlas`
  - result: `timeout (existing long-running DFF/logging path)`
- command:
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-large --repeats 10 --variant all`
  - result: `pass`
- command:
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-context --repeats 10 --variant all`
  - result: `pass`
- command:
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-context-large --repeats 3 --variant all`
  - result: `pass`

Observed benchmark result:

- `token-lm-large`, repeated:
  - `AdamW`: `testNLL=4.58178 +/- 0.00977`
  - `ATLAS-BSRP`: `4.58258 +/- 0.01477`
  - `ATLAS-SPARROW`: `4.57641 +/- 0.01124`
  - `ATLAS-ASTER`: `4.58045 +/- 0.00787`
  - `ATLAS-AEGIS`: `4.57398 +/- 0.01308`

- `token-lm-context`, repeated:
  - `AdamW`: `testNLL=4.24529 +/- 0.03627`
  - `ATLAS-BSRP`: `4.80616 +/- 0.01737`
  - `ATLAS-SPARROW`: `4.79550 +/- 0.01327`
  - `ATLAS-ASTER`: `4.80681 +/- 0.01416`
  - `ATLAS-AEGIS`: `4.79780 +/- 0.02183`

- `token-lm-context-large`, repeated:
  - `AdamW`: `testNLL=4.26843 +/- 0.04316`
  - `ATLAS-BSRP`: `5.14022 +/- 0.04002`
  - `ATLAS-SPARROW`: `5.15660 +/- 0.00482`
  - `ATLAS-ASTER`: `5.15688 +/- 0.01281`
  - `ATLAS-AEGIS`: `5.16129 +/- 0.01511`

Diagnostics:

- `token-lm-large`:
  - `AEGIS` channel trust:
    - `lambdaSpatial=0.093 +/- 0.011`
    - `lambdaPredictive=0.532 +/- 0.032`
    - `lambdaOutput=0.376 +/- 0.038`
  - `ASTER` timing:
    - `boundaryMs=50.054 +/- 3.405`

- `token-lm-context`:
  - `AEGIS` channel trust:
    - `lambdaSpatial=0.051 +/- 0.004`
    - `lambdaPredictive=0.641 +/- 0.051`
    - `lambdaOutput=0.307 +/- 0.052`
  - `ASTER` timing:
    - `boundaryMs=60.330 +/- 4.807`

- `token-lm-context-large`:
  - `SPARROW`:
    - `edge=0.463 +/- 0.069`
    - `mem=0.020 +/- 0.004`
  - `ASTER`:
    - `activeModes=1.50 +/- 0.00`
    - `mode2Frac=0.500 +/- 0.000`
    - `edge=0.234 +/- 0.017`
    - `sigma=0.270 +/- 0.013`
    - `predR2=0.907 +/- 0.016`
    - `mem=0.008 +/- 0.002`
    - `boundaryMs=61.300 +/- 0.937`
  - `AEGIS`:
    - `lambdaSpatial=0.051 +/- 0.004`
    - `lambdaPredictive=0.616 +/- 0.071`
    - `lambdaOutput=0.334 +/- 0.067`
    - predictive calibration:
      - `predicted=0.436 +/- 0.086`
      - `realized=0.431 +/- 0.086`
      - `error=0.010 +/- 0.000`
    - output calibration:
      - `predicted=0.207 +/- 0.019`
      - `realized=0.211 +/- 0.018`
      - `error=0.007 +/- 0.002`

Interpretation:

- This is a **measured neutral-to-negative result** for the new transformer sensor family.
- The richer token-conditioned residual sketch plus attention-pattern stream does **not** improve the established transformer benchmarks:
  - `token-lm-large` stays unchanged in held-out NLL,
  - `token-lm-context` stays effectively unchanged and still trails `SPARROW`,
  - `token-lm-context-large` is harsher still and leaves `AdamW` clearly ahead of every ATLAS-family branch.
- The internal optimizer evidence is real but not sufficient:
  - `AEGIS` still calibrates both predictive and output channels sensibly,
  - `ASTER` still opens real retained modes,
  - but those signals are not translating into better held-out language-model loss on the harder structured-context tasks.
- The cost picture also worsens:
  - transformer ASTER / AEGIS boundary time rose materially on the new sensor path,
  - especially on `token-lm-context` and `token-lm-context-large`.

Updated recommendation:

- keep the current transformer `AEGIS-v2` branch as the documented baseline,
- treat `token-lm-context-large` as a stronger negative-control stress benchmark,
- stop local transformer observable expansion of this same class,
- keep `SPARROW` as the best current ATLAS-family control on the available transformer LM tasks,
- keep `AdamW` as the robustness anchor and topline baseline,
- if transformer optimizer research continues, move to a materially different observable family or a more realistic larger LM setup:
  - direct token-conditioned attention / KV-state observables,
  - richer function-side sequence state,
  - or a substantially more realistic LM benchmark before more optimizer-side refinement.

### N.26 KAPPA-AEGIS compressed KV-retrieval observable

On **April 10, 2026**, I implemented the first `KAPPA-AEGIS` prototype: a compressed KV-retrieval observable added to the existing transformer-side `AEGIS-v2` output channel.

The implementation:

- kept the `AEGIS-v2` fusion and calibration math unchanged,
- added an opt-in KAPPA observable with:
  - `kappaEnabled`
  - `kappaHeads`
  - `kappaLagBuckets`
  - `kappaRank`
- tracked a small projected retrieval summary from the first `kappaHeads` attention heads using lag buckets `{1, 2, 4, 8}`,
- appended that retrieval summary to the transformer ASTER/AEGIS observation and control streams,
- left actuation head-only.

Important implementation note:

- the first test pass exposed a shared `NNetwork::clean()` regression in the `TrainingConfig` reset path, not a KAPPA logic fault,
- fixing that lifetime/reset bug restored `atlas-controller` and the small token-LM harness before the KAPPA benchmark pass.

Focused validation:

- `atlas-controller`
- tiny `token-lm` smoke runs for `AdamW` and `AEGIS` with `--atlas-kappa-enabled 1`
- `token-lm-large --repeats 10 --variant all --atlas-kappa-enabled 1`
- `token-lm-context --repeats 10 --variant all --atlas-kappa-enabled 1`
- `token-lm-context-large --repeats 3 --variant all --atlas-kappa-enabled 1`

Results:

- `token-lm-large`, repeated:
  - `AdamW`: `4.58178 +/- 0.00977`
  - `ATLAS-BSRP`: `4.58258 +/- 0.01477`
  - `ATLAS-SPARROW`: `4.57641 +/- 0.01124`
  - `ATLAS-ASTER`: `4.58045 +/- 0.00787`
  - `ATLAS-AEGIS`: `4.57398 +/- 0.01308`

- `token-lm-context`, repeated:
  - `AdamW`: `4.24529 +/- 0.03627`
  - `ATLAS-BSRP`: `4.80616 +/- 0.01737`
  - `ATLAS-SPARROW`: `4.79550 +/- 0.01327`
  - `ATLAS-ASTER`: `4.80683 +/- 0.01419`
  - `ATLAS-AEGIS`: `4.79781 +/- 0.02185`

- `token-lm-context-large`, repeated:
  - `AdamW`: `4.26843 +/- 0.04316`
  - `ATLAS-BSRP`: `5.14022 +/- 0.04002`
  - `ATLAS-SPARROW`: `5.15660 +/- 0.00482`
  - `ATLAS-ASTER`: `5.15665 +/- 0.01296`
  - `ATLAS-AEGIS`: `5.16122 +/- 0.01521`

Diagnostics:

- `token-lm-large`:
  - `AEGIS`:
    - `lambdaSpatial=0.096 +/- 0.010`
    - `lambdaPredictive=0.549 +/- 0.034`
    - `lambdaOutput=0.355 +/- 0.037`
  - `ASTER`:
    - `edge=0.177 +/- 0.030`
    - `predR2=0.759 +/- 0.111`
    - `boundaryMs=143.831 +/- 0.867`

- `token-lm-context`:
  - `AEGIS`:
    - `lambdaSpatial=0.055 +/- 0.004`
    - `lambdaPredictive=0.684 +/- 0.050`
    - `lambdaOutput=0.261 +/- 0.050`
  - `ASTER`:
    - `edge=0.302 +/- 0.025`
    - `predR2=0.766 +/- 0.093`
    - `boundaryMs=160.069 +/- 6.996`

- `token-lm-context-large`:
  - `AEGIS`:
    - `lambdaSpatial=0.060 +/- 0.003`
    - `lambdaPredictive=0.745 +/- 0.133`
    - `lambdaOutput=0.195 +/- 0.133`
  - `ASTER`:
    - `edge=0.339 +/- 0.059`
    - `predR2=0.540 +/- 0.301`
    - `boundaryMs=148.377 +/- 1.027`

Interpretation:

- This is a **clean negative result for KAPPA-Lite**.
- The compressed KV-retrieval observable is implemented and active, but it does **not** improve the transformer rankings:
  - `token-lm-large` stays effectively unchanged from the established `AEGIS-v2` result,
  - `token-lm-context` still leaves `SPARROW` ahead of `AEGIS`,
  - `token-lm-context-large` remains a strong negative-control case where `AdamW` is clearly best and `AEGIS` is now the weakest ATLAS-family branch in the repeated run.
- The evidence still supports the earlier conclusion:
  - transformer-side `AEGIS` is real on the easier LM preset,
  - but local observable expansion of the same family is no longer yielding progress on context-heavy LM tasks.

Updated recommendation:

- keep `token-lm-large` as the light repeated transformer control,
- keep `token-lm-context` and `token-lm-context-large` as the stress benchmarks,
- keep `SPARROW` as the best current ATLAS-family control on the harder transformer LM cases,
- keep `AEGIS-v2` as the unified branch for dynamic non-transformer work and the light transformer preset,
- stop local KAPPA-style retrieval observable tuning,
- if transformer optimizer research continues, move next to a materially different family or benchmark:
  - richer token-conditioned attention/KV observables with stronger state structure,
  - a more realistic larger LM setup,
  - or a broader training regime where optimizer differences are not dominated by the tiny synthetic harness.

### N.27 CITADEL-v3-lite delayed-evidence anchor pass

On **April 10, 2026**, I replaced the earlier hard-regime `CITADEL-Lite` anchor with a softer **delayed-evidence CITADEL-v3-lite** pass.

Important scope note:

- This is still **not** the full research-design ideal of a literal `AdamW` prior plus fully modular `BSRP` / `SPARROW` / `ASTER` residual proposal APIs.
- In the current codebase, the practical step is still to refine the existing `AEGIS-v2` shell:
  - keep `SPARROW` and `ASTER` embedded in the ATLAS path,
  - keep `AEGIS` delayed benefit/error tracking,
  - replace the old hard regime anchor with a smoother anchor driven by **delayed trust EMAs**, **channel disagreement**, and **predictive-vs-output dominance** rather than raw hard-regime mass alone.

Implementation summary:

- core config and trust plumbing:
  - [training_config.h](/home/robert/dev/glades-ml/Backend/Machine%20Learning/Networks/training_config.h)
  - [atlas_optimizer.h](/home/robert/dev/glades-ml/Backend/Machine%20Learning/Networks/atlas_optimizer.h)
  - [atlas_optimizer.cpp](/home/robert/dev/glades-ml/Backend/Machine%20Learning/Networks/atlas_optimizer.cpp)
- runtime state and diagnostics:
  - [network.h](/home/robert/dev/glades-ml/Backend/Machine%20Learning/Networks/network.h)
  - [transformer_model_state.inc](/home/robert/dev/glades-ml/Backend/Machine%20Learning/Networks/transformer_model_state.inc)
  - [network.cpp](/home/robert/dev/glades-ml/Backend/Machine%20Learning/Networks/network.cpp)
- checkpoint/config persistence:
  - [checkpoint_persistence.cpp](/home/robert/dev/glades-ml/Backend/Machine%20Learning/Networks/checkpoint_persistence.cpp)
- DFF and transformer training paths:
  - [sgd_dff.cpp](/home/robert/dev/glades-ml/Backend/Machine%20Learning/Networks/sgd_dff.cpp)
  - [sgd_transformer.cpp](/home/robert/dev/glades-ml/Backend/Machine%20Learning/Networks/sgd_transformer.cpp)
- benchmark/test harness:
  - [atlas-alt-bench.cpp](/home/robert/dev/glades-ml/unit-tests/Backend/Machine%20Learning/atlas-alt-bench.cpp)
  - [atlas-test.cpp](/home/robert/dev/glades-ml/unit-tests/Backend/Machine%20Learning/atlas-test.cpp)

Focused verification:

- build:
  - `cmake --build /home/robert/dev/glades-ml/build -j4`
  - `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
- focused tests:
  - `./unit-tests/build/glades-unit-tests atlas-controller`
  - `timeout 120s ./unit-tests/build/glades-unit-tests atlas`
- focused benchmark ladder:
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode teacher-student --repeats 3 --variant all`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode latent-forecast --repeats 3 --variant all`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode nonlinear-forecast --repeats 3 --variant all`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-large --repeats 10 --variant all`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-context --repeats 10 --variant all`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-context-large --repeats 3 --variant all`

Results:

- `teacher-student`, repeated:
  - `SPARROW`: `testMSE=0.04640 +/- 0.00142`
  - `AEGIS`: `0.05841 +/- 0.01678`
  - `CITADEL`: `0.04906 +/- 0.00089`
  - diagnostics:
    - `anchor=0.301 +/- 0.047`
    - `hard=0.000 +/- 0.000`
    - `sparrowTrust=0.636 +/- 0.042`
  - interpretation:
    - CITADEL still **materially repairs** the AEGIS regression,
    - but still does **not** beat `SPARROW`.

- `latent-forecast`, repeated:
  - `SPARROW`: `testMSE=0.00266 +/- 0.00017`
  - `AEGIS`: `0.00267 +/- 0.00015`
  - `CITADEL`: `0.00280 +/- 0.00018`
  - diagnostics:
    - `anchor=0.387 +/- 0.050`
    - `hard=0.000 +/- 0.000`
    - `sparrowTrust=0.561 +/- 0.039`
  - interpretation:
    - CITADEL is still a **clear regression** here.
    - Even with the softer anchor, it is still suppressing useful predictive structure on a task where predictive memory is already the right answer.

- `nonlinear-forecast`, repeated:
  - `BSRP`: `0.00143 +/- 0.00009`
  - `ASTER`: `0.00140 +/- 0.00014`
  - `AEGIS`: `0.00138 +/- 0.00011`
  - `CITADEL`: `0.00135 +/- 0.00004`
  - diagnostics:
    - `anchor=0.459 +/- 0.031`
    - `hard=0.000 +/- 0.000`
    - `sparrowTrust=0.501 +/- 0.029`
  - interpretation:
    - CITADEL remains the **best branch so far** on this benchmark.
    - This is still the strongest positive result for the anchoring idea.

- `token-lm-large`, repeated:
  - `AdamW`: `testNLL=4.58178 +/- 0.00977`
  - `SPARROW`: `4.57641 +/- 0.01124`
  - `AEGIS`: `4.57398 +/- 0.01308`
  - `CITADEL`: `4.58191 +/- 0.01244`
  - diagnostics:
    - `anchor=0.000 +/- 0.000`
    - `hard=1.000 +/- 0.000`
    - `sparrowTrust=0.753 +/- 0.007`
  - interpretation:
    - The delayed-evidence anchor **fixes the pathological over-anchoring** from the earlier `CITADEL-Lite` pass.
    - But it still does **not** beat `AEGIS` or `SPARROW` on the lighter transformer preset.

- `token-lm-context`, repeated:
  - `AdamW`: `testNLL=4.24529 +/- 0.03627`
  - `SPARROW`: `4.79550 +/- 0.01327`
  - `AEGIS`: `4.79780 +/- 0.02183`
  - `CITADEL`: `4.80268 +/- 0.01102`
  - diagnostics:
    - `anchor=0.028 +/- 0.013`
    - `hard=1.000 +/- 0.000`
    - `sparrowTrust=0.886 +/- 0.006`
  - interpretation:
    - The new anchor no longer saturates here either,
    - but the branch is still a **regression** against `SPARROW` and remains far behind `AdamW`.

- `token-lm-context-large`, repeated:
  - `AdamW`: `testNLL=4.26843 +/- 0.04316`
  - `SPARROW`: `5.15660 +/- 0.00482`
  - `AEGIS`: `5.16129 +/- 0.01511`
  - `CITADEL`: `5.15084 +/- 0.03278`
  - diagnostics:
    - `anchor=0.028 +/- 0.013`
    - `hard=1.000 +/- 0.000`
    - `sparrowTrust=0.896 +/- 0.004`
  - interpretation:
    - CITADEL stays the **best ATLAS-family branch** on the hardest transformer stress case,
    - but it is still **far behind `AdamW`**.

Overall interpretation:

- CITADEL-v3-lite is a **better-calibrated version** of the earlier CITADEL anchor pass.
- It fixes the worst failure mode of `CITADEL-Lite`:
  - the old branch saturated at `anchor≈0.95` on transformer tasks,
  - the new branch does **not**.
- But it is still **not** a universal optimizer upgrade.
- The updated picture is:
  - CITADEL remains useful where anchoring truly helps:
    - `teacher-student` relative to `AEGIS`,
    - `nonlinear-forecast`,
    - `token-lm-context-large` within the ATLAS family.
  - CITADEL still hurts where predictive structure should dominate:
    - `latent-forecast`,
    - `token-lm-large`,
    - `token-lm-context`.

Updated recommendation:

- Keep **AEGIS-v2** as the documented unified branch.
- Keep **CITADEL-v3-lite** as a recorded experimental branch, but do **not** promote it to the default optimizer.
- The delayed-evidence anchor is a real improvement over the old hard heuristic, but it still does not solve the transformer gap to `AdamW`.
- If this line is revisited, the next version should:
  - use the full `AdamW`-prior posterior-fusion formulation,
  - learn anchor strength from delayed benefit more directly than the current heuristic trust EMAs,
  - and improve transformer observability rather than relying on anchor logic alone.
- Until then:
  - keep `SPARROW` as the best planted/easier-dynamics control,
  - keep `AEGIS-v2` as the main unified research branch,
  - keep `CITADEL-v3-lite` as evidence that **soft backbone anchoring can help**, but not yet as a broad replacement.

### N.28 RAMPART-lite covariance-aware residual posterior

On **April 10, 2026**, I implemented the first practical **RAMPART-lite** pass:

- `RAMPART` in research-design form is an `AdamW`-centered residual posterior with correlated `BSRP` / `SPARROW` / `ASTER` sensors and an explicit residual trust region.
- The current codebase does **not** yet expose fully modular optimizer proposals around a literal `AdamW` prior.
- So the practical implementation is a **minimal ATLAS-family approximation** inside the existing `AEGIS` shell:
  - add a covariance-aware three-channel posterior over `{spatial, predictive, output}`,
  - add an explicit residual budget,
  - keep `SPARROW` and `ASTER` as embedded ATLAS residual channels,
  - and record `tau`, `budget`, covariance, and effective predictive trust as runtime diagnostics.

Implementation summary:

- config and persistence:
  - [training_config.h](/home/robert/dev/glades-ml/Backend/Machine%20Learning/Networks/training_config.h)
  - [checkpoint_persistence.cpp](/home/robert/dev/glades-ml/Backend/Machine%20Learning/Networks/checkpoint_persistence.cpp)
- runtime diagnostics and state:
  - [network.h](/home/robert/dev/glades-ml/Backend/Machine%20Learning/Networks/network.h)
  - [transformer_model_state.inc](/home/robert/dev/glades-ml/Backend/Machine%20Learning/Networks/transformer_model_state.inc)
  - [network.cpp](/home/robert/dev/glades-ml/Backend/Machine%20Learning/Networks/network.cpp)
- DFF / transformer training paths:
  - [sgd_dff.cpp](/home/robert/dev/glades-ml/Backend/Machine%20Learning/Networks/sgd_dff.cpp)
  - [sgd_transformer.cpp](/home/robert/dev/glades-ml/Backend/Machine%20Learning/Networks/sgd_transformer.cpp)
- benchmark / tests:
  - [atlas-alt-bench.cpp](/home/robert/dev/glades-ml/unit-tests/Backend/Machine%20Learning/atlas-alt-bench.cpp)
  - [atlas-test.cpp](/home/robert/dev/glades-ml/unit-tests/Backend/Machine%20Learning/atlas-test.cpp)

Verification:

- build:
  - `cmake --build /home/robert/dev/glades-ml/build -j4`
  - `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
- tests:
  - `./unit-tests/build/glades-unit-tests atlas-controller`
  - `timeout 120s ./unit-tests/build/glades-unit-tests atlas`
- repeated benchmark ladder:
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode teacher-student --repeats 3 --variant all`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode latent-forecast --repeats 3 --variant all`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode nonlinear-forecast --repeats 3 --variant all`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-large --repeats 10 --variant all`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-context --repeats 10 --variant all`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-context-large --repeats 3 --variant all`

Results:

- `teacher-student`, repeated:
  - `SPARROW`: `testMSE=0.04640 +/- 0.00142`
  - `AEGIS`: `0.05841 +/- 0.01678`
  - `CITADEL`: `0.04906 +/- 0.00089`
  - `RAMPART`: `0.04803 +/- 0.00313`
  - diagnostics:
    - `tau=2.301 +/- 0.048`
    - `budget=0.181 +/- 0.011`
    - `cov=0.065 +/- 0.021`
    - `sparrowTrust=0.177 +/- 0.014`
  - interpretation:
    - RAMPART materially improves over `AEGIS`,
    - edges out `CITADEL`,
    - but still does not recover `SPARROW`.

- `latent-forecast`, repeated:
  - `SPARROW`: `testMSE=0.00266 +/- 0.00017`
  - `AEGIS`: `0.00267 +/- 0.00015`
  - `CITADEL`: `0.00280 +/- 0.00018`
  - `RAMPART`: `0.00274 +/- 0.00025`
  - diagnostics:
    - `tau=2.635 +/- 0.052`
    - `budget=0.161 +/- 0.008`
    - `cov=0.015 +/- 0.005`
    - `sparrowTrust=0.160 +/- 0.008`
  - interpretation:
    - RAMPART is a regression versus both `SPARROW` and `AEGIS`.
    - The current posterior is still suppressing the predictive channel too aggressively on a task where predictive memory is the right answer.

- `nonlinear-forecast`, repeated:
  - `ASTER`: `testMSE=0.00140 +/- 0.00014`
  - `AEGIS`: `0.00138 +/- 0.00011`
  - `CITADEL`: `0.00135 +/- 0.00004`
  - `RAMPART`: `0.00130 +/- 0.00001`
  - diagnostics:
    - `tau=2.530 +/- 0.081`
    - `budget=0.172 +/- 0.009`
    - `cov=0.024 +/- 0.032`
    - `sparrowTrust=0.170 +/- 0.007`
  - interpretation:
    - This is the strongest positive result of the branch.
    - RAMPART becomes the **best result so far** on the hardest DFF dynamic benchmark.

- `token-lm-large`, repeated:
  - `AdamW`: `testNLL=4.58178 +/- 0.00977`
  - `SPARROW`: `4.57641 +/- 0.01124`
  - `AEGIS`: `4.57398 +/- 0.01308`
  - `CITADEL`: `4.58191 +/- 0.01244`
  - `RAMPART`: `4.58103 +/- 0.00835`
  - diagnostics:
    - `tau=1.601 +/- 0.047`
    - `budget=0.247 +/- 0.010`
    - `cov=0.116 +/- 0.029`
    - `sparrowTrust=0.177 +/- 0.011`
  - interpretation:
    - RAMPART loses the current `AEGIS` transformer win.
    - It is slightly better than `AdamW` and `CITADEL` here, but clearly worse than `SPARROW` and `AEGIS`.

- `token-lm-context`, repeated:
  - `AdamW`: `testNLL=4.24529 +/- 0.03627`
  - `SPARROW`: `4.79550 +/- 0.01327`
  - `AEGIS`: `4.79780 +/- 0.02183`
  - `CITADEL`: `4.80268 +/- 0.01102`
  - `RAMPART`: `4.80289 +/- 0.01105`
  - diagnostics:
    - `tau=1.233 +/- 0.115`
    - `budget=0.433 +/- 0.015`
    - `cov=0.180 +/- 0.043`
    - `sparrowTrust=0.327 +/- 0.015`
  - interpretation:
    - RAMPART does not close the transformer context gap.
    - It is effectively tied with `CITADEL`, still behind `SPARROW`, and far behind `AdamW`.

- `token-lm-context-large`, repeated:
  - `AdamW`: `testNLL=4.26843 +/- 0.04316`
  - `SPARROW`: `5.15660 +/- 0.00482`
  - `AEGIS`: `5.16129 +/- 0.01511`
  - `CITADEL`: `5.15084 +/- 0.03278`
  - `RAMPART`: `5.15856 +/- 0.01328`
  - diagnostics:
    - `tau=1.178 +/- 0.130`
    - `budget=0.453 +/- 0.015`
    - `cov=0.182 +/- 0.044`
    - `sparrowTrust=0.337 +/- 0.014`
  - interpretation:
    - RAMPART is not the best ATLAS-family branch on the hardest transformer stress case.
    - `CITADEL` still holds that position, though both remain far behind `AdamW`.

Overall interpretation:

- RAMPART-lite is a **real optimizer branch**, not a no-op:
  - it improves over `AEGIS` and `CITADEL` on `teacher-student`,
  - and it sets the best result so far on `nonlinear-forecast`.
- But it is **not** a new default:
  - it regresses on `latent-forecast`,
  - loses the `token-lm-large` win that `AEGIS` currently holds,
  - and does not solve the context-heavy transformer gap.
- The current posterior is still leaning too hard toward the spatial channel on tasks that want predictive dominance.

Updated recommendation:

- Keep **AEGIS-v2** as the main unified branch.
- Keep **RAMPART-lite** as a documented experimental branch, specifically because:
  - it is now the best result on `nonlinear-forecast`,
  - and it shows that covariance-aware residual budgeting can help on some dynamic regimes.
- Do **not** promote RAMPART-lite to the default optimizer.
- If this line continues, the next version should:
  - move closer to the full research-design optimizer with a literal `AdamW` prior and modular residual proposals,
  - improve the way predictive trust is preserved on easy/linear dynamic tasks,
  - and avoid spending more time on transformer-side posterior math until the observable family improves.

## N.30 MERIT-lite geometry-only residual posterior

Date: April 10, 2026

Goal:

- test the next structural optimizer update after `RAMPART-lite`
- keep `AdamW` as the backbone,
- treat `BSRP` as geometry only instead of a competing residual sensor,
- and let only `SPARROW` and `ASTER` provide residual evidence

Implementation:

- added `MERIT-lite` config/state/diagnostics in the optimizer/runtime path
- wired a new `ATLAS-MERIT` benchmark variant into `atlas-alt-bench`
- added a transformer MERIT smoke test
- fixed a benchmark harness regression in the DFF alt-bench path by explicitly initializing transformer enum defaults so `setTrainingConfig` no longer fails with `unknown ffnKind`

Verification:

- `cmake --build /home/robert/dev/glades-ml/build -j4`
- `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
- focused benchmarks:
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode teacher-student --repeats 3 --variant all`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode latent-forecast --repeats 3 --variant all`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode nonlinear-forecast --repeats 3 --variant all`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-large --repeats 10 --variant all`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-context --repeats 10 --variant all`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-context-large --repeats 3 --variant all`
- note:
  - `./unit-tests/build/glades-unit-tests atlas-controller` still fails in the existing controller probation test, outside the new MERIT path
  - `timeout 120s ./unit-tests/build/glades-unit-tests atlas` still runs into the long logging-heavy DFF path and is not a useful regression signal here

Results:

- `teacher-student`, repeated:
  - `SPARROW`: `testMSE=0.04640 +/- 0.00142`
  - `CITADEL`: `0.04906 +/- 0.00089`
  - `RAMPART`: `0.04803 +/- 0.00313`
  - `MERIT`: `0.06339 +/- 0.01503`
  - diagnostics:
    - `tau=0.760 +/- 0.113`
    - `budget=0.268 +/- 0.015`
    - `cov=0.011 +/- 0.006`
    - `sparrowTrust=0.263 +/- 0.016`
    - `geom=1.000 +/- 0.000`
  - interpretation:
    - MERIT is a clear regression on a task that wants predictive dominance.
    - Treating spatial structure as pure geometry did not recover SPARROW-like behavior here.

- `latent-forecast`, repeated:
  - `SPARROW`: `testMSE=0.00266 +/- 0.00017`
  - `AEGIS`: `0.00267 +/- 0.00015`
  - `RAMPART`: `0.00274 +/- 0.00025`
  - `MERIT`: `0.00273 +/- 0.00026`
  - diagnostics:
    - `tau=0.724 +/- 0.028`
    - `budget=0.292 +/- 0.005`
    - `cov=0.001 +/- 0.001`
    - `sparrowTrust=0.292 +/- 0.005`
    - `geom=1.000 +/- 0.000`
  - interpretation:
    - MERIT is slightly better than RAMPART-lite but still worse than SPARROW and AEGIS.
    - The geometry-only reinterpretation does not fix the predictive-task regression.

- `nonlinear-forecast`, repeated:
  - `ASTER`: `testMSE=0.00140 +/- 0.00014`
  - `AEGIS`: `0.00138 +/- 0.00011`
  - `CITADEL`: `0.00135 +/- 0.00004`
  - `RAMPART`: `0.00130 +/- 0.00001`
  - `MERIT`: `0.00148 +/- 0.00017`
  - diagnostics:
    - `tau=0.722 +/- 0.021`
    - `budget=0.287 +/- 0.002`
    - `cov=0.019 +/- 0.006`
    - `sparrowTrust=0.280 +/- 0.006`
    - `geom=1.000 +/- 0.000`
  - interpretation:
    - This is the decisive negative result for MERIT-lite.
    - The branch loses the strongest RAMPART-lite win instead of preserving it.

- `token-lm-large`, repeated:
  - `AEGIS`: `testNLL=4.57398 +/- 0.01308`
  - `SPARROW`: `4.57641 +/- 0.01124`
  - `MERIT`: `4.57922 +/- 0.01099`
  - `RAMPART`: `4.58103 +/- 0.00835`
  - `AdamW`: `4.58178 +/- 0.00977`
  - diagnostics:
    - `tau=1.083 +/- 0.037`
    - `budget=0.104 +/- 0.009`
    - `cov=0.094 +/- 0.019`
    - `sparrowTrust=0.074 +/- 0.005`
    - `geom=0.216 +/- 0.039`
  - interpretation:
    - MERIT loses the current AEGIS transformer win.
    - It is better than RAMPART-lite and slightly better than AdamW, but still behind both AEGIS and SPARROW.

- `token-lm-context`, repeated:
  - `AdamW`: `testNLL=4.24529 +/- 0.03627`
  - `SPARROW`: `4.79550 +/- 0.01327`
  - `AEGIS`: `4.79780 +/- 0.02183`
  - `MERIT`: `4.80676 +/- 0.00960`
  - diagnostics:
    - `tau=0.749 +/- 0.080`
    - `budget=0.228 +/- 0.015`
    - `cov=0.164 +/- 0.029`
    - `sparrowTrust=0.173 +/- 0.004`
    - `geom=0.163 +/- 0.025`
  - interpretation:
    - MERIT does not help on the main structured-context transformer benchmark.
    - It is worse than SPARROW and AEGIS and remains far behind AdamW.

- `token-lm-context-large`, repeated:
  - `AdamW`: `testNLL=4.26843 +/- 0.04316`
  - `ATLAS-BSRP`: `5.14022 +/- 0.04002`
  - `CITADEL`: `5.15084 +/- 0.03278`
  - `MERIT`: `5.15037 +/- 0.00973`
  - `SPARROW`: `5.15660 +/- 0.00482`
  - `RAMPART`: `5.15856 +/- 0.01328`
  - diagnostics:
    - `tau=0.612 +/- 0.043`
    - `budget=0.258 +/- 0.007`
    - `cov=0.196 +/- 0.019`
    - `sparrowTrust=0.186 +/- 0.003`
    - `geom=0.144 +/- 0.010`
  - interpretation:
    - MERIT edges out CITADEL and RAMPART on the hardest transformer stress case.
    - But it still does not beat plain ATLAS-BSRP there, and it remains far behind AdamW.

Overall interpretation:

- MERIT-lite is a **real structural test**, not a no-op.
- But it is **not a promotion candidate**:
  - it regresses badly on `teacher-student`,
  - stays behind SPARROW/AEGIS on `latent-forecast`,
  - loses the strongest `RAMPART-lite` nonlinear-dynamics win,
  - and does not improve the main transformer context benchmark.
- The one modest positive sign is `token-lm-context-large`, where MERIT slightly improves over `CITADEL`/`RAMPART`, but that is not enough to outweigh the broader regressions.

Updated recommendation:

- Keep **AEGIS-v2** as the main unified branch.
- Keep **RAMPART-lite** as the best nonlinear-dynamics experimental branch.
- Keep **MERIT-lite** only as a documented structural falsifier:
  - it shows that moving `BSRP` from sensor to geometry does not, by itself, solve the branch conflicts.
- Do **not** promote MERIT-lite to the default optimizer.
- The remaining work should go to better transformer observability, not more fusion-geometry rearrangements of the same sensor family.

## N.31 STRATA-lite sparse regime-conditioned residual control

On **April 10, 2026**, I implemented the first practical **STRATA-lite** branch:

- `STRATA` in research-design form is a sparse `AdamW`-backed residual controller with four residual modes:
  - `null`
  - `predictive`
  - `output`
  - `coupled predictive+output`
- In the minimal implementation, it reuses the existing `SPARROW` and head-only `ASTER` sensors, but replaces dense always-on fusion with a hard dominant-mode controller plus a bounded residual budget.

Implementation summary:

- added `STRATA-lite` config, checkpoint, runtime diagnostics, and persistent mode state
- implemented the sparse mode controller in both the DFF and transformer optimizer paths
- wired a new `ATLAS-STRATA` benchmark variant into `atlas-alt-bench`
- added a transformer STRATA smoke test
- fixed the alt-bench banner strings so `ATLAS-STRATA` is shown in the printed `Optimizers:` header alongside the already-running variant rows

Verification:

- `cmake --build /home/robert/dev/glades-ml/build -j4`
- `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
- focused benchmarks:
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode teacher-student --repeats 3 --variant all`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode latent-forecast --repeats 3 --variant all`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode nonlinear-forecast --repeats 3 --variant all`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-large --repeats 10 --variant all`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-context --repeats 10 --variant all`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-context-large --repeats 3 --variant all`
- note:
  - `./unit-tests/build/glades-unit-tests atlas-controller` still fails in the existing controller probation test, outside the new STRATA path
  - `timeout 120s ./unit-tests/build/glades-unit-tests atlas` still runs into the long logging-heavy DFF path and is not a useful regression signal here

Results:

- `teacher-student`, repeated:
  - `SPARROW`: `testMSE=0.04640 +/- 0.00142`
  - `STRATA`: `0.04614 +/- 0.00158`
  - diagnostics:
    - `null=0.000 +/- 0.000`
    - `pred=0.983 +/- 0.024`
    - `out=0.000 +/- 0.000`
    - `coupled=0.017 +/- 0.024`
    - `budget=0.486 +/- 0.044`
  - interpretation:
    - STRATA recovers the expected predictive-dominant behavior.
    - It slightly edges out SPARROW on point estimate, though well within repeated-run noise.

- `latent-forecast`, repeated:
  - `SPARROW`: `testMSE=0.00266 +/- 0.00017`
  - `AEGIS`: `0.00267 +/- 0.00015`
  - `STRATA`: `0.00257 +/- 0.00003`
  - diagnostics:
    - `null=0.000 +/- 0.000`
    - `pred=1.000 +/- 0.000`
    - `out=0.000 +/- 0.000`
    - `coupled=0.000 +/- 0.000`
    - `budget=0.537 +/- 0.007`
  - interpretation:
    - This is the strongest positive STRATA result.
    - Sparse predictive-only control clearly beats the current dense fusion branches here.

- `nonlinear-forecast`, repeated:
  - `CITADEL`: `testMSE=0.00135 +/- 0.00004`
  - `RAMPART`: `0.00130 +/- 0.00001`
  - `STRATA`: `0.00135 +/- 0.00006`
  - diagnostics:
    - `null=0.000 +/- 0.000`
    - `pred=1.000 +/- 0.000`
    - `out=0.000 +/- 0.000`
    - `coupled=0.000 +/- 0.000`
    - `budget=0.546 +/- 0.012`
  - interpretation:
    - STRATA does not preserve the `RAMPART-lite` nonlinear win.
    - It remains competitive with `CITADEL`, but the controller is still over-selecting predictive mode on a task that wants stronger output-side structure.

- `token-lm-large`, repeated:
  - `AEGIS`: `testNLL=4.57398 +/- 0.01308`
  - `STRATA`: `4.57630 +/- 0.00844`
  - `SPARROW`: `4.57641 +/- 0.01124`
  - diagnostics:
    - `null=0.000 +/- 0.000`
    - `pred=0.000 +/- 0.000`
    - `out=0.800 +/- 0.400`
    - `coupled=0.200 +/- 0.400`
    - `budget=0.382 +/- 0.026`
  - interpretation:
    - STRATA is clearly better than the weaker fusion branches here.
    - It nearly matches SPARROW and stays close to AEGIS, but it does not recover the current AEGIS win.

- `token-lm-context`, repeated:
  - `AdamW`: `testNLL=4.24529 +/- 0.03627`
  - `SPARROW`: `4.79550 +/- 0.01327`
  - `AEGIS`: `4.79780 +/- 0.02183`
  - `STRATA`: `4.80881 +/- 0.01186`
  - diagnostics:
    - `null=0.000 +/- 0.000`
    - `pred=0.000 +/- 0.000`
    - `out=0.000 +/- 0.000`
    - `coupled=1.000 +/- 0.000`
    - `budget=0.446 +/- 0.012`
  - interpretation:
    - STRATA is a clear regression on the main context-heavy transformer stress case.
    - The sparse controller is locking into coupled mode where the correct fallback is still much closer to `AdamW`.

- `token-lm-context-large`, repeated:
  - `AdamW`: `testNLL=4.26843 +/- 0.04316`
  - `ATLAS-BSRP`: `5.14022 +/- 0.04002`
  - `MERIT`: `5.15037 +/- 0.00973`
  - `CITADEL`: `5.15084 +/- 0.03278`
  - `STRATA`: `5.15574 +/- 0.00228`
  - diagnostics:
    - `null=0.000 +/- 0.000`
    - `pred=0.000 +/- 0.000`
    - `out=0.000 +/- 0.000`
    - `coupled=1.000 +/- 0.000`
    - `budget=0.446 +/- 0.006`
  - interpretation:
    - STRATA improves over `AEGIS`, `RAMPART`, `SPARROW`, and `ASTER` on the hardest transformer stress case.
    - But it still trails `BSRP`, `MERIT`, and `CITADEL`, and remains far behind `AdamW`.

Overall interpretation:

- STRATA-lite is a **real optimizer branch**, not a no-op.
- It is the first sparse-controller branch here that cleanly recovers the expected predictive mode on the planted/light-dynamics tasks.
- It produces the best current result on `latent-forecast` and is effectively tied for best on `teacher-student`.
- But it is **not a new default**:
  - it loses the `RAMPART-lite` nonlinear-dynamics win,
  - it does not recover the `AEGIS-v2` `token-lm-large` win,
  - and it regresses on the main transformer context benchmark.

Updated recommendation:

- Keep **AEGIS-v2** as the main unified branch.
- Keep **RAMPART-lite** as the best nonlinear-dynamics experimental branch.
- Keep **STRATA-lite** as the best sparse predictive-control experimental branch:
  - especially because it wins `latent-forecast`,
  - and because it shows sparse mode selection is better than dense fusion on the light predictive regimes.
- Do **not** promote STRATA-lite to the default optimizer.
- The next work should combine:
  - sparse controller structure like STRATA,
  - with better transformer observability and stronger null/AdamW fallback on context-heavy transformer regimes.

## N.32 STRATA-v2 delayed-benefit null calibration

On **April 10, 2026**, I implemented **STRATA-v2** as a direct refinement of `STRATA-lite`:

- kept the same sparse controller modes `{null, predictive, output, coupled}`
- replaced the old mode score with a delayed-benefit controller based on per-mode excess-gain EMAs
- added explicit STRATA diagnostics for:
  - per-mode realized benefit
  - selected excess gain vs the backbone
  - switch rate

Implementation summary:

- extended `STRATA` runtime state with delayed-benefit EMAs and per-step realized benefit fields
- updated both the DFF and transformer STRATA controllers to score modes from delayed excess gain rather than raw instantaneous trust alone
- extended runtime aggregation and the alt-bench printout with STRATA benefit diagnostics

Verification:

- `cmake --build /home/robert/dev/glades-ml/build -j4`
- `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
- focused benchmarks:
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode teacher-student --repeats 3 --variant all`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode latent-forecast --repeats 3 --variant all`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode nonlinear-forecast --repeats 3 --variant all`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-large --repeats 10 --variant all`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-context --repeats 10 --variant all`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-context-large --repeats 3 --variant all`

Results:

- `teacher-student`, repeated:
  - `SPARROW`: `testMSE=0.04640 +/- 0.00142`
  - `STRATA-v2`: `0.04600 +/- 0.00144`
  - diagnostics:
    - `null=0.017 +/- 0.024`
    - `pred=0.983 +/- 0.024`
    - `budget=0.256 +/- 0.064`
    - `bPred=0.310 +/- 0.095`
  - interpretation:
    - STRATA-v2 preserves the predictive planted-task win.
    - The new controller introduces a small null occupancy without hurting the result.

- `latent-forecast`, repeated:
  - `SPARROW`: `testMSE=0.00266 +/- 0.00017`
  - `STRATA-v2`: `0.00257 +/- 0.00003`
  - diagnostics:
    - `null=0.000 +/- 0.000`
    - `pred=1.000 +/- 0.000`
    - `budget=0.306 +/- 0.012`
    - `bPred=0.416 +/- 0.022`
  - interpretation:
    - STRATA-v2 cleanly preserves the best current `latent-forecast` result.
    - This remains the strongest evidence that sparse predictive mode selection beats dense fusion on light predictive regimes.

- `nonlinear-forecast`, repeated:
  - `RAMPART`: `testMSE=0.00130 +/- 0.00001`
  - `CITADEL`: `0.00135 +/- 0.00004`
  - `STRATA-v2`: `0.00135 +/- 0.00006`
  - diagnostics:
    - `pred=1.000 +/- 0.000`
    - `out=0.000 +/- 0.000`
    - `bPred=0.471 +/- 0.056`
    - `bCoupled=0.007 +/- 0.070`
  - interpretation:
    - STRATA-v2 still routes the nonlinear task through predictive mode.
    - So the delayed-benefit change does not recover the `RAMPART-lite` nonlinear win.

- `token-lm-large`, repeated:
  - `AEGIS`: `testNLL=4.57398 +/- 0.01308`
  - `SPARROW`: `4.57641 +/- 0.01124`
  - `STRATA-v2`: `4.57630 +/- 0.00844`
  - diagnostics:
    - `out=1.000 +/- 0.000`
    - `budget=0.155 +/- 0.056`
    - `bOut=0.445 +/- 0.107`
  - interpretation:
    - STRATA-v2 is cleaner than STRATA-lite here:
      - it chooses pure output mode instead of mixing in coupled mode.
    - But the benchmark result is effectively unchanged, still behind `AEGIS-v2`.

- `token-lm-context`, repeated:
  - `AdamW`: `testNLL=4.24529 +/- 0.03627`
  - `SPARROW`: `4.79550 +/- 0.01327`
  - `AEGIS`: `4.79780 +/- 0.02183`
  - `STRATA-v2`: `4.80880 +/- 0.01186`
  - diagnostics:
    - `null=0.000 +/- 0.000`
    - `out=1.000 +/- 0.000`
    - `budget=0.386 +/- 0.031`
    - `bOut=0.680 +/- 0.044`
  - interpretation:
    - This is the decisive negative result for STRATA-v2.
    - The delayed-benefit controller still does not abstain on the main transformer context benchmark.

- `token-lm-context-large`, repeated:
  - `AdamW`: `testNLL=4.26843 +/- 0.04316`
  - `ATLAS-BSRP`: `5.14022 +/- 0.04002`
  - `MERIT`: `5.15037 +/- 0.00973`
  - `CITADEL`: `5.15084 +/- 0.03278`
  - `STRATA-v2`: `5.15589 +/- 0.00225`
  - diagnostics:
    - `null=0.000 +/- 0.000`
    - `out=1.000 +/- 0.000`
    - `budget=0.403 +/- 0.016`
    - `bOut=0.645 +/- 0.039`
  - interpretation:
    - STRATA-v2 is more internally coherent than STRATA-lite, but it still does not beat the stronger ATLAS-family transformer branches here.

Overall interpretation:

- STRATA-v2 is a **cleaner controller**, not a stronger optimizer.
- It improves controller interpretation:
  - predictive tasks now show explicit positive predictive excess,
  - `token-lm-large` cleanly selects output mode,
  - and switch rate collapses to zero once a regime is identified.
- But it does **not** solve the main problem:
  - it still fails to abstain on the context-heavy transformer tasks,
  - and it still does not recover the nonlinear-dynamics win.

Updated recommendation:

- Keep **AEGIS-v2** as the main unified branch.
- Keep **RAMPART-lite** as the best nonlinear-dynamics branch.
- Keep **STRATA-v2** as the best sparse predictive-control branch.
- Do **not** spend more time on controller calibration alone.
- The next work should shift to better transformer observability and stronger `AdamW`-relative null evidence, because the current sensor family is still telling STRATA to act where it should stay close to the backbone.

## N.33 Transformer Margin-Shortfall Observable Pass

Objective:

- Change the transformer ASTER/AEGIS observable family without touching controller or fusion math.
- Add explicit regime-level target-margin shortfall features relative to running transformer baselines.

Implementation:

- widened the transformer token-conditioned observable tail from `4` to `8` dimensions in [network.cpp](/home/robert/dev/glades-ml/Backend/Machine%20Learning/Networks/network.cpp)
- kept the original hashed support-residual sketch in the first 4 token-conditioned slots
- used the extra 4 slots for explicit regime-level features:
  - target-margin shortfall vs running EMA
  - hard-negative logit pressure vs running EMA
  - hard-regime shortfall vs running EMA
  - rate of tokens whose target margin fell below the running baseline
- threaded the same explicit features into the pattern/control stream, while leaving STRATA / AEGIS / CITADEL / RAMPART math unchanged

State additions:

- per-regime EMAs:
  - `targetMarginEma`
  - `hardNegativeLogitEma`
  - `hardMarginShortfallEma`
- per-batch regime summaries:
  - `batchTargetMarginSum`
  - `batchHardNegativeLogitSum`
  - `batchBaselineWorseSum`

Validation:

- `cmake --build /home/robert/dev/glades-ml/build -j4`
- `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
- `./unit-tests/build/glades-unit-tests atlas-controller`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-large --repeats 10 --variant all`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-context --repeats 10 --variant all`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-context-large --repeats 3 --variant all`

Results:

- `token-lm-large`, repeated:
  - `AEGIS`: `testNLL=4.57398 +/- 0.01308`
  - `SPARROW`: `4.57641 +/- 0.01124`
  - `STRATA-v2`: `4.57630 +/- 0.00844`
  - interpretation:
    - no material change from the pre-observable-pass ordering
    - the lighter transformer preset still favors `AEGIS-v2`

- `token-lm-context`, repeated:
  - `AdamW`: `testNLL=4.24529 +/- 0.03627`
  - `SPARROW`: `4.79550 +/- 0.01327`
  - `AEGIS`: `4.79789 +/- 0.02180`
  - `STRATA-v2`: `4.80879 +/- 0.01183`
  - interpretation:
    - the explicit margin-shortfall features do not close the transformer context gap
    - STRATA still selects pure output mode and still does not abstain

- `token-lm-context-large`, repeated:
  - `AdamW`: `testNLL=4.26843 +/- 0.04316`
  - `BSRP`: `5.14022 +/- 0.04002`
  - `MERIT`: `5.15034 +/- 0.00971`
  - `CITADEL`: `5.15112 +/- 0.03287`
  - `STRATA-v2`: `5.15589 +/- 0.00229`
  - `AEGIS`: `5.16168 +/- 0.01503`
  - interpretation:
    - this is another clean falsification of the current transformer observable family
    - explicit margin-shortfall signals sharpen the observable story, but they do not change the benchmark ranking

Conclusion:

- The transformer issue is not just “missing margin information.”
- Even after adding explicit regime-level margin-shortfall observables, the ATLAS-family branches still fail to approach `AdamW` on the context-heavy transformer tasks.
- So the remaining bottleneck is likely a deeper transformer observability problem:
  - retrieval failure
  - context compression failure
  - or benchmark mismatch between these ATLAS observables and the true next-token error geometry

Updated recommendation:

- Keep **AEGIS-v2** as the main unified branch.
- Keep **RAMPART-lite** as the best nonlinear-dynamics branch.
- Keep **STRATA-v2** as the best sparse predictive-control branch.
- Treat the margin-shortfall observable pass as a documented negative result.
- Stop local observable/controller tinkering of this same class.
- If transformer-side work continues, move next to a materially different observable family or a larger/more realistic LM benchmark.

## N.34 Larger / More Realistic LM Benchmark: `token-lm-document`

Objective:

- Add a larger transformer benchmark that is more document-like than the current synthetic recurrence and structured-context presets, without depending on an external corpus.

Implementation:

- added a new alt-bench mode:
  - `token-lm-document`
- new preset:
  - `vocab=257`
  - `dModel=48`
  - `dFF=192`
  - `layers=3`
  - `heads=6`
  - `seqLen=96`
  - `trainSeqs=48`
  - `testSeqs=12`
  - `epochs=2`
- added a new pseudo-document generator in [atlas-alt-bench.cpp](/home/robert/dev/glades-ml/unit-tests/Backend/Machine%20Learning/atlas-alt-bench.cpp):
  - article-style sections
  - topic markers
  - entity / place / year / verb / object pools
  - cross-paragraph recall of summaries, objects, places, years, and speakers
  - mixed topical and global detail tokens

Validation:

- `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-document --repeats 3 --variant all`

Results:

- `AdamW`: `testNLL=4.71194 +/- 0.03683`
- `ATLAS-SPARROW`: `5.44572 +/- 0.00859`
- `ATLAS-ASTER`: `5.44583 +/- 0.00438`
- `ATLAS-RAMPART`: `5.44799 +/- 0.01566`
- `ATLAS-CITADEL`: `5.45769 +/- 0.01389`
- `ATLAS-MERIT`: `5.45772 +/- 0.00891`
- `ATLAS-STRATA`: `5.46286 +/- 0.00668`
- `ATLAS-AEGIS`: `5.46353 +/- 0.00718`
- `ATLAS-BSRP`: `5.46411 +/- 0.01637`

Interpretation:

- The new benchmark is materially harsher and more realistic than `token-lm-large`.
- `AdamW` remains the only strong transformer-side answer here.
- The ATLAS-family variants cluster tightly together far behind it.
- Within the ATLAS family:
  - `SPARROW`, `ASTER`, and `RAMPART` are effectively tied for best
  - `AEGIS-v2` does not retain its `token-lm-large` advantage on this harder document-style task

Updated recommendation:

- Keep `token-lm-document` as the new larger / more realistic transformer stress benchmark.
- Keep `token-lm-large` as the light control and `token-lm-context` / `token-lm-context-large` as structured-recall controls.
- Treat the optimizer-side transformer story as stable for now:
  - `AdamW` is still the robustness anchor
  - ATLAS-family branches are not yet competitive on the more realistic LM-style benchmarks
- If transformer work continues, the next step should change the observable family or use an even more realistic corpus-like benchmark, not another controller or fusion tweak.

## N.35 Larger / More Realistic LM Benchmark: `token-lm-document`

Objective:

- Add a larger benchmark that is closer to document-style language modeling than the recurrence and structured-context synthetic tasks, while keeping the harness self-contained.

Implementation:

- added new mode:
  - `token-lm-document`
- added new preset in [atlas-alt-bench.cpp](/home/robert/dev/glades-ml/unit-tests/Backend/Machine%20Learning/atlas-alt-bench.cpp):
  - `vocab=257`
  - `dModel=48`
  - `dFF=192`
  - `layers=3`
  - `heads=6`
  - `seqLen=96`
  - `trainSeqs=48`
  - `testSeqs=12`
  - `epochs=2`
- added a pseudo-document sequence generator:
  - article-style sections
  - topic markers
  - entity / place / year / verb / object pools
  - cross-paragraph recall of summaries, speakers, places, years, and objects
  - mixed topic-local and globally shared detail tokens

Validation:

- `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-document --repeats 3 --variant all`

Results:

- `AdamW`: `testNLL=4.71194 +/- 0.03683`
- `ATLAS-SPARROW`: `5.44572 +/- 0.00859`
- `ATLAS-ASTER`: `5.44583 +/- 0.00438`
- `ATLAS-RAMPART`: `5.44799 +/- 0.01566`
- `ATLAS-CITADEL`: `5.45769 +/- 0.01389`
- `ATLAS-MERIT`: `5.45772 +/- 0.00891`
- `ATLAS-STRATA`: `5.46286 +/- 0.00668`
- `ATLAS-AEGIS`: `5.46353 +/- 0.00718`
- `ATLAS-BSRP`: `5.46411 +/- 0.01637`

Interpretation:

- `token-lm-document` is materially harsher than `token-lm-large`.
- `AdamW` remains clearly best on the more realistic transformer-side benchmark.
- The ATLAS-family variants are tightly clustered far behind it.
- Within the ATLAS family, the current ordering is:
  - `SPARROW ≈ ASTER ≈ RAMPART`
  - then `CITADEL / MERIT`
  - then `STRATA / AEGIS / BSRP`
- So the `AEGIS-v2` advantage on `token-lm-large` does not survive the harder document-like task.

Updated recommendation:

- Keep `token-lm-document` as the primary larger transformer benchmark.
- Keep `token-lm-large` as the quick control and `token-lm-context*` as structured-recall controls.
- Keep `AdamW` as the transformer default.
- Keep only `SPARROW`, `ASTER`, and `RAMPART` as the main transformer-side ATLAS controls.
- Do not spend more time on controller/fusion tweaks before a materially different observable family or a genuinely corpus-backed benchmark is available.

## N.36 Next-Generation AdamW Replacement Prototypes: `AURORA-lite`, `SEAM-lite`, `QUASAR-lite`

Objective:

- Implement minimal practical prototypes of the three next-generation AdamW-replacement frameworks proposed in the research design pass:
  - `AURORA`
  - `SEAM`
  - `QUASAR`
- Test them on the fixed mixed ladder before doing any deeper architectural investment.

Implementation:

- added new ATLAS-family variant flags in [training_config.h](/home/robert/dev/glades-ml/Backend/Machine%20Learning/Networks/training_config.h):
  - `auroraEnabled`
  - `seamEnabled`
  - `quasarEnabled`
- added tuning parameters:
  - `auroraHorizonBlend`
  - `auroraBudgetMax`
  - `seamMirrorStep`
  - `seamBudgetMax`
  - `quasarTemperature`
  - `quasarBudgetMax`
- implemented minimal runtime branches in:
  - [sgd_dff.cpp](/home/robert/dev/glades-ml/Backend/Machine%20Learning/Networks/sgd_dff.cpp)
  - [sgd_transformer.cpp](/home/robert/dev/glades-ml/Backend/Machine%20Learning/Networks/sgd_transformer.cpp)
- added benchmark harness variants and CLI routing in:
  - [atlas-alt-bench.cpp](/home/robert/dev/glades-ml/unit-tests/Backend/Machine%20Learning/atlas-alt-bench.cpp)

Prototype interpretation:

- `AURORA-lite`
  - horizon-smoothed predictive/output residual fusion around the existing ATLAS state.
- `SEAM-lite`
  - mirror-descent-style coordinate reweighting over spatial / predictive / output channels.
- `QUASAR-lite`
  - entropy-regularized mode distribution over null / predictive / output / coupled residual modes.

These are deliberately minimal practical instantiations, not full literal implementations of the original mathematical idealizations.

Validation:

- `cmake --build /home/robert/dev/glades-ml/build -j4`
- `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
- `./unit-tests/build/glades-unit-tests atlas-controller`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode teacher-student --repeats 3 --variant all`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode latent-forecast --repeats 3 --variant all`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode nonlinear-forecast --repeats 3 --variant all`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-large --repeats 3 --variant all`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-context --repeats 3 --variant all`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-document --repeats 3 --variant all`

Results by case:

- `teacher-student`
  - `AURORA`: `0.05823 +/- 0.01678`
  - `SEAM`: `0.05727 +/- 0.01730`
  - `QUASAR`: `0.04955 +/- 0.00292`
  - interpretation:
    - only `QUASAR` is competitive here, but it still trails `SPARROW 0.04640` and `STRATA-v2 0.04600`

- `latent-forecast`
  - `AURORA`: `0.00257 +/- 0.00001`
  - `SEAM`: `0.00284 +/- 0.00021`
  - `QUASAR`: `0.00288 +/- 0.00026`
  - interpretation:
    - `AURORA-lite` is the strongest of the three and ties the best current ATLAS-family result on this case

- `nonlinear-forecast`
  - `AURORA`: `0.00135 +/- 0.00007`
  - `SEAM`: `0.00154 +/- 0.00008`
  - `QUASAR`: `0.00134 +/- 0.00005`
  - interpretation:
    - `QUASAR-lite` is the strongest of the three, but it does not beat `RAMPART-lite 0.00130`

- `token-lm-large`
  - `AURORA`: `4.58315 +/- 0.00276`
  - `SEAM`: `4.57783 +/- 0.00226`
  - `QUASAR`: `4.58082 +/- 0.01226`
  - interpretation:
    - `SEAM-lite` is clearly the best of the three on the light transformer preset
    - but it still trails `AEGIS-v2 4.56779`

- `token-lm-context`
  - `AURORA`: `4.80784 +/- 0.00140`
  - `SEAM`: `4.78930 +/- 0.00997`
  - `QUASAR`: `4.80071 +/- 0.00903`
  - interpretation:
    - `SEAM-lite` is the best of the three and edges out the prior ATLAS-family front (`SPARROW 4.79139`)
    - but all ATLAS-family branches remain far behind `AdamW 4.22251`

- `token-lm-document`
  - `AURORA`: `5.44241 +/- 0.01185`
  - `SEAM`: `5.44980 +/- 0.00931`
  - `QUASAR`: `5.45924 +/- 0.01230`
  - interpretation:
    - `AURORA-lite` is the best of the three and becomes the best current ATLAS-family branch on this larger document-style benchmark
    - but it still remains far behind `AdamW 4.71194`

Overall interpretation:

- The three new frameworks are all real, functioning branches, not no-ops.
- They separate into distinct niches:
  - `AURORA-lite`: strongest on `latent-forecast` and the best new branch on `token-lm-document`
  - `SEAM-lite`: strongest on the lighter transformer presets and the best ATLAS-family branch on `token-lm-context`
  - `QUASAR-lite`: strongest of the three on `teacher-student` and `nonlinear-forecast`
- None of the three is dominant enough to replace the current role map:
  - `STRATA-v2` still owns the sparse predictive niche
  - `RAMPART-lite` still owns the nonlinear niche
  - `AEGIS-v2` still owns `token-lm-large`
  - `AdamW` still dominates the realistic transformer benchmarks

Updated recommendation:

- Keep `AURORA-lite`, `SEAM-lite`, and `QUASAR-lite` as documented experimental branches.
- Do not promote any of them to the default optimizer.
- If this line continues, the most promising follow-up is:
  - `SEAM`-style coordinate control on transformer-light tasks
  - `AURORA`-style horizon fusion on document-like tasks
  - but only after materially improving transformer observability
- The main blocker remains unchanged:
  - on realistic LM-style benchmarks, observability is still weaker than the optimizer-control law

## N.37 AURORA-v2 Transformer Observability Pass + `token-lm-corpus`

Objective:

- Follow the post-`AURORA-lite` recommendation:
  - keep only `AURORA` as the active next-generation replacement candidate
  - upgrade its transformer path before inventing another optimizer family
  - add a more corpus-like benchmark instead of relying only on synthetic recurrence/document generators

Implementation:

- upgraded the transformer `AURORA` branch in [sgd_transformer.cpp](/home/robert/dev/glades-ml/Backend/Machine%20Learning/Networks/sgd_transformer.cpp):
  - `AURORA-v2` now consumes retrieval-aware transformer observables already accumulated by the ASTER path:
    - lag-pattern signal from `batchLayerPatternSum`
    - optional compressed KV signal from `batchLayerKappaSum` when enabled
    - margin shortfall / hard-negative pressure / baseline-failure rate
  - these signals now affect:
    - adjusted predictive trust
    - adjusted output trust
    - AURORA covariance
    - AURORA residual budget
    - ASTER output-memory gain inside the AURORA branch
- added a new token benchmark mode in [atlas-alt-bench.cpp](/home/robert/dev/glades-ml/unit-tests/Backend/Machine%20Learning/atlas-alt-bench.cpp):
  - `token-lm-corpus`
  - checked-in small public-domain prose excerpts
  - shared train/test vocabulary
  - separate train and test document pools
  - contiguous sequence windows drawn from each split

Validation:

- `cmake --build /home/robert/dev/glades-ml/build -j4`
- `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
- `./unit-tests/build/glades-unit-tests atlas-controller`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-large --repeats 3 --variant all`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-context --repeats 3 --variant all`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-document --repeats 3 --variant all`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-corpus --repeats 3 --variant all`

Results:

- `token-lm-large`
  - `ATLAS-AURORA`: `4.58315 +/- 0.00276`
  - unchanged in practice versus the earlier `AURORA-lite` result
  - still behind `ATLAS-AEGIS 4.56779`

- `token-lm-context`
  - `ATLAS-AURORA`: `4.80784 +/- 0.00140`
  - essentially unchanged
  - still behind `ATLAS-SEAM 4.78930`
  - still far behind `AdamW 4.22251`

- `token-lm-document`
  - `ATLAS-AURORA`: `5.44242 +/- 0.01184`
  - effectively unchanged from the prior `AURORA-lite` result
  - still the best ATLAS-family branch on this pseudo-document benchmark
  - still far behind `AdamW 4.71194`

- `token-lm-corpus`
  - `AdamW`: `6.13689 +/- 0.02343`
  - `ATLAS-BSRP`: `6.11350 +/- 0.01552`
  - `ATLAS-AEGIS`: `6.13132 +/- 0.01645`
  - `ATLAS-SEAM`: `6.13035 +/- 0.02041`
  - `ATLAS-AURORA`: `6.13745 +/- 0.00790`
  - `ATLAS-SPARROW`: `6.14640 +/- 0.01959`
  - `ATLAS-RAMPART`: `6.14782 +/- 0.01119`
  - interpretation:
    - this benchmark changes the ordering materially
    - unlike `token-lm-document`, the ATLAS family no longer uniformly trails `AdamW`
    - the best branch here is actually `BSRP`, with `AEGIS` / `SEAM` close behind
    - `AURORA-v2` is competitive but not leading

Interpretation:

- The `AURORA-v2` observability pass did **not** materially improve the existing synthetic/document transformer stress cases.
- So the retrieval-aware trust rewrite, by itself, is not the missing ingredient.
- The new `token-lm-corpus` benchmark is still useful because it reveals a different regime:
  - it does **not** reproduce the strong `AdamW` dominance seen on `token-lm-document`
  - it suggests the benchmark family matters as much as the optimizer family
- So the transformer optimizer picture is now split:
  - pseudo-document benchmark: `AdamW` clearly best
  - small checked-in corpus benchmark: `BSRP / AEGIS / SEAM` are competitive and `AdamW` is no longer dominant

Updated recommendation:

- Keep `token-lm-document` as the primary harsher transformer stress benchmark.
- Keep `token-lm-corpus` as a secondary realism check, not yet the new primary decision benchmark.
- Keep `AURORA-v2` as the only active next-generation replacement candidate, but do not promote it.
- Do not create another optimizer family until one of the following is true:
  - `AURORA` materially narrows the `token-lm-document` gap to `AdamW`
  - or the corpus benchmark is expanded enough to become more trustworthy than the pseudo-document generator
- The main unresolved issue remains:
  - transformer observability and benchmark realism are still entangled

## N.38 `token-lm-corpus-large` + Reduced 10-Repeat Corpus/Document Ladder

Objective:

- Follow the next benchmark-first recommendation:
  - freeze the transformer optimizer set to `AdamW`, `BSRP`, `AEGIS`, `SEAM`, and `AURORA`
  - add a harder corpus benchmark before creating another optimizer family
  - rerun `token-lm-document` and `token-lm-corpus` at `repeats=10`
  - decide whether the corpus-side ATLAS advantage survives scale

Implementation:

- added a new benchmark mode in [atlas-alt-bench.cpp](/home/robert/dev/glades-ml/unit-tests/Backend/Machine%20Learning/atlas-alt-bench.cpp):
  - `token-lm-corpus-large`
  - more checked-in public-domain prose documents
  - longer windows
  - larger decoder preset
  - larger train/test split than `token-lm-corpus`
- the mode uses:
  - `vocab=769`
  - `dModel=56`
  - `dFF=224`
  - `layers=4`
  - `heads=8`
  - `seqLen=112`
  - `trainSeqs=64`
  - `testSeqs=16`
  - `epochs=2`

Validation:

- `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
- `./unit-tests/build/glades-unit-tests atlas-controller`
- smoke:
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-corpus-large --repeats 1 --variant all`
- reduced ladder:
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-document --repeats 10 --variant adamw`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-document --repeats 10 --variant base`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-document --repeats 10 --variant aegis`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-document --repeats 10 --variant seam`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-document --repeats 10 --variant aurora`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-corpus --repeats 10 --variant adamw`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-corpus --repeats 10 --variant base`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-corpus --repeats 10 --variant aegis`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-corpus --repeats 10 --variant seam`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-corpus --repeats 10 --variant aurora`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-corpus-large --repeats 10 --variant adamw`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-corpus-large --repeats 10 --variant base`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-corpus-large --repeats 10 --variant aegis`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-corpus-large --repeats 10 --variant seam`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-corpus-large --repeats 10 --variant aurora`

Results:

- `token-lm-document`
  - `AdamW`: `4.72491 +/- 0.04037`
  - `ATLAS-BSRP`: `5.45081 +/- 0.01650`
  - `ATLAS-AEGIS`: `5.45397 +/- 0.01089`
  - `ATLAS-SEAM`: `5.45420 +/- 0.01070`
  - `ATLAS-AURORA`: `5.45528 +/- 0.01526`

- `token-lm-corpus`
  - `AdamW`: `6.19049 +/- 0.06828`
  - `ATLAS-BSRP`: `6.12795 +/- 0.02145`
  - `ATLAS-SEAM`: `6.13086 +/- 0.01381`
  - `ATLAS-AURORA`: `6.13373 +/- 0.01353`
  - `ATLAS-AEGIS`: `6.13630 +/- 0.01229`

- `token-lm-corpus-large`
  - `AdamW`: `6.27182 +/- 0.10700`
  - `ATLAS-BSRP`: `6.35582 +/- 0.01716`
  - `ATLAS-AURORA`: `6.38183 +/- 0.02088`
  - `ATLAS-SEAM`: `6.39078 +/- 0.02419`
  - `ATLAS-AEGIS`: `6.39683 +/- 0.02174`

Interpretation:

- The old `token-lm-document` conclusion survives 10 repeats:
  - `AdamW` is still decisively best
  - the ATLAS-family branches remain clustered far behind it
- The small-corpus anomaly is real enough to matter:
  - after `repeats=10`, `BSRP` still beats `AdamW` on `token-lm-corpus`
  - `SEAM`, `AURORA`, and `AEGIS` also remain close
- But the scale-up resolves the ambiguity:
  - on `token-lm-corpus-large`, `AdamW` becomes clearly best again
  - the ATLAS-family advantage from `token-lm-corpus` does **not** survive the harder corpus setting
- So the current picture is now consistent across the harder LM-style benchmarks:
  - `token-lm-document`: `AdamW` clearly best
  - `token-lm-corpus-large`: `AdamW` clearly best
  - `token-lm-corpus`: useful secondary realism check, but too small to drive optimizer strategy by itself

Updated recommendation:

- Keep `token-lm-document` as the primary harsh transformer benchmark.
- Keep `token-lm-corpus-large` as the secondary corpus-style benchmark.
- Demote `token-lm-corpus` to a small realism sanity check, not a decision benchmark.
- Freeze the transformer optimizer comparison set to:
  - `AdamW`
  - `ATLAS-BSRP`
  - `ATLAS-AEGIS`
  - `ATLAS-SEAM`
  - `ATLAS-AURORA`
- Do not create another optimizer family until one of those branches materially narrows the `token-lm-document` gap.
- The practical research bottleneck remains:
  - transformer observability and benchmark realism, not optimizer controller math

---

## April 11, 2026: AdamW gap decomposition on the hard transformer benchmarks

Implemented a focused transformer diagnostic pass instead of another optimizer branch.

What changed:

- Added benchmark-side grouped transformer parameter snapshots so the optimizer-profile comparison is measured as a before/after training displacement, not as a hot-path per-apply copy inside `sgd_transformer.cpp`.
- Kept only the cheap runtime diagnostics in the training loop:
  - train/test target margin
  - hard-negative logit
  - sampled apply-time

Verification:

- `cmake --build /home/robert/dev/glades-ml/build -j4`
- `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
- `./unit-tests/build/glades-unit-tests atlas-controller`
- reduced hard-benchmark comparison:
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-document --repeats 3 --variant adamw`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-document --repeats 3 --variant base`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-document --repeats 3 --variant aurora`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-corpus-large --repeats 3 --variant adamw`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-corpus-large --repeats 3 --variant base`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-corpus-large --repeats 3 --variant aurora`

Results:

- `token-lm-document`
  - `AdamW`: `testNLL=4.71194 +/- 0.03683`
    - proxy profile: `blk=[2.617 2.703 2.966] final=0.736 head=2.441 hShare=0.203 applyMs=0.851`
    - margins: `train=-1.567`, `test=-1.363`
  - `ATLAS-BSRP`: `5.46411 +/- 0.01637`
    - proxy profile: `blk=[0.412 0.291 0.349] final=0.019 head=0.142 hShare=0.051 applyMs=3.825`
    - margins: `train=-0.400`, `test=-0.460`
    - profile cosine vs `AdamW`: `0.954`
  - `ATLAS-AURORA`: `5.44242 +/- 0.01184`
    - proxy profile: `blk=[0.401 0.361 0.395] final=0.026 head=0.146 hShare=0.045 applyMs=135.101`
    - margins: `train=-0.399`, `test=-0.477`
    - profile cosine vs `AdamW`: `0.961`

- `token-lm-corpus-large`
  - `AdamW`: `testNLL=6.19167 +/- 0.05172`
    - proxy profile: `blk=[4.103 3.494 3.320 3.176] final=1.109 head=10.019 hShare=0.661 applyMs=1.548`
    - margins: `train=-2.076`, `test=-3.119`
  - `ATLAS-BSRP`: `6.34855 +/- 0.01436`
    - proxy profile: `blk=[0.614 0.459 0.368 0.434] final=0.057 head=0.348 hShare=0.119 applyMs=6.600`
    - margins: `train=-0.945`, `test=-1.410`
    - profile cosine vs `AdamW`: `0.819`
  - `ATLAS-AURORA`: `6.38980 +/- 0.01715`
    - proxy profile: `blk=[0.569 0.432 0.424 0.426] final=0.052 head=0.313 hShare=0.101 applyMs=147.017`
    - margins: `train=-0.771`, `test=-1.189`
    - profile cosine vs `AdamW`: `0.807`

Interpretation:

- On both hard transformer benchmarks, the ATLAS-family branches are not merely mis-scaled copies of `AdamW`.
- `BSRP` and `AURORA` have reasonably high coarse profile cosine on `token-lm-document`, but they act at dramatically smaller magnitude and with far less head-share than `AdamW`.
- The divergence becomes stronger on `token-lm-corpus-large`, where the ATLAS-family profile departs more from `AdamW` and still loses badly in test NLL.
- `AURORA` is especially informative here:
  - it is directionally similar enough to be considered a real transformer-side control branch
  - but its overhead is massive (`applyMs ~135-147 ms`) and it still does not close the loss gap
- So the current transformer gap is not just a learning-rate or trust-budget problem.
- The stronger diagnosis is:
  - `AdamW` allocates much more update mass to the head and final decoder path on the hard LM tasks
  - the current ATLAS-family branches do not generate enough high-impact head-side correction
  - and `AURORA` pays heavy control cost without buying margin or NLL gains

Updated recommendation:

- Freeze the transformer comparison set to:
  - `AdamW`
  - `ATLAS-BSRP`
  - `ATLAS-AURORA`
- Keep this new gap decomposition pass for future transformer checks.
- Do not build another optimizer family yet.
- If transformer optimizer work continues, the next serious step should target:
  - head-dominant actuation
  - or materially richer transformer observability
- If neither can materially narrow the `token-lm-document` gap, stop optimizer-side replacement work and treat `AdamW` as the effective transformer default.

## April 11, 2026: AURORA head-dominant actuation pass is a clean negative result

Implemented a narrow transformer-only AURORA split-budget pass:

- added explicit AURORA knobs:
  - `auroraHeadGain`
  - `auroraBodyTrustScale`
- exposed them in the alt-bench harness as:
  - `--atlas-aurora-head-gain`
  - `--atlas-aurora-body-trust-scale`
- changed the transformer AURORA path to:
  - amplify ASTER/AURORA head correction when the retrieval/margin signal is strong
  - attenuate non-head SPARROW trust for `WIn/WOut` and block weights
  - leave `tokE` on the full transformer-side predictive trust so the head can absorb more of the residual budget

Default test values used in the benchmark harness:

- `auroraHeadGain=3.0`
- `auroraBodyTrustScale=0.60`

Verification:

- `cmake --build /home/robert/dev/glades-ml/build -j4`
- `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
- `./unit-tests/build/glades-unit-tests atlas-controller`
- focused hard-benchmark comparison:
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-document --repeats 3 --variant adamw`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-document --repeats 3 --variant aurora`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-corpus-large --repeats 3 --variant adamw`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-corpus-large --repeats 3 --variant aurora`

Results:

- `token-lm-document`
  - `AdamW`: unchanged at `testNLL=4.71194 +/- 0.03683`
  - `ATLAS-AURORA`: unchanged at `5.44242 +/- 0.01184`
    - proxy profile stayed effectively the same:
      - `blk=[0.401 0.361 0.395]`
      - `final=0.026`
      - `head=0.146`
      - `hShare=0.045`
      - `applyMs=138.088`

- `token-lm-corpus-large`
  - `AdamW`: unchanged at `testNLL=6.19167 +/- 0.05172`
  - `ATLAS-AURORA`: unchanged at `6.38980 +/- 0.01715`
    - proxy profile also stayed effectively the same:
      - `blk=[0.569 0.432 0.424 0.426]`
      - `final=0.052`
      - `head=0.313`
      - `hShare=0.101`
      - `applyMs=142.260`

Interpretation:

- The hard transformer gap is not being driven by a simple lack of head weighting inside the current AURORA controller.
- Even a targeted split-budget pass that explicitly boosts the head and suppresses the body does not materially change:
  - test NLL
  - head share
  - or AURORA's large runtime cost
- So the remaining bottleneck is not a small controller-allocation mistake inside the current observable family.
- The stronger conclusion is:
  - current AURORA observables do not contain enough high-value transformer-side signal
  - and the current head-local correction is too weak in substance, not just too weak in nominal gain

Updated recommendation:

- Keep the current AURORA implementation as a documented falsified branch extension.
- Do not spend more time on local AURORA budget/head-allocation tuning.
- If transformer optimizer work continues, it should now focus on:
  - materially richer observability
  - or a fundamentally different transformer actuation path
- Until one branch closes the `token-lm-document` gap, treat `AdamW` as the effective transformer default.

## April 11, 2026: corrected AdamW-backed AURORA schedule is a clean negative result

The first AdamW-backed AURORA prototype changed the transformer update kernel but still inherited the ATLAS token learning rate in the alt-bench harness. That made the initial result ambiguous. I corrected the benchmark path so:

- `ATLAS-AURORA` uses `cfg.token.adamLR` whenever `auroraAdamwBackbone=1`
- the token benchmark header now reports the effective AURORA token learning rate correctly

Verification:

- `cmake --build /home/robert/dev/glades-ml/build -j4`
- `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
- `./unit-tests/build/glades-unit-tests atlas-controller`
- focused AURORA reruns:
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-large --repeats 3 --variant aurora`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-document --repeats 3 --variant aurora`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-corpus-large --repeats 3 --variant aurora`

Corrected results with `ATLAS-AURORA(lr=0.0010, adamwBackbone=1)`:

- `token-lm-large`
  - `ATLAS-AURORA`: `testNLL=4.58343 +/- 0.00283`
  - proxy profile:
    - `blk=[0.002 0.001]`
    - `final=0.000`
    - `head=0.000`
    - `hShare=0.021`
    - `applyMs=40.241`

- `token-lm-document`
  - `ATLAS-AURORA`: `testNLL=5.54653 +/- 0.00819`
  - proxy profile:
    - `blk=[0.030 0.024 0.023]`
    - `final=0.001`
    - `head=0.007`
    - `hShare=0.024`
    - `applyMs=132.445`

- `token-lm-corpus-large`
  - `ATLAS-AURORA`: `testNLL=6.64533 +/- 0.00463`
  - proxy profile:
    - `blk=[0.042 0.027 0.022 0.020]`
    - `final=0.001`
    - `head=0.013`
    - `hShare=0.050`
    - `applyMs=151.493`

Interpretation:

- The earlier near-neutral AdamW-backed AURORA result was a harness artifact caused by the wrong token learning-rate path.
- Once AURORA actually runs with an AdamW-scale transformer schedule, it gets materially worse on every transformer benchmark that matters.
- The update profile also collapses:
  - even less head share
  - very small parameter displacement
  - still large AURORA controller cost
- So the conclusion is now stronger than before:
  - the current AURORA controller/observable family does not produce a useful AdamW replacement on transformer LM
  - and simply swapping the backbone kernel to AdamW does not rescue it

Updated recommendation:

- Treat AdamW-backed AURORA as a falsified transformer branch.
- Do not spend more time on local AURORA backbone/schedule tuning.
- Keep:
  - `AdamW` as the transformer default
  - `AEGIS-v2` as the best ATLAS-family light-transformer control
  - `AURORA` only as a documented next-generation research branch that failed this transformer replacement test
- If transformer optimizer work continues, change observability or actuation path, not the local AURORA learning-rate/backbone configuration.

## April 11, 2026: retrieval-distance observables are another clean negative result

I used a materially different transformer observability patch rather than another controller or schedule change.

What changed:

- kept the corrected AdamW-backed AURORA schedule
- widened the AURORA-only token-conditioned observation size from `8` to `12`
- added generic sequence-retrieval observables in the transformer ASTER/AURORA path:
  - whether the target token appeared earlier in the current context
  - inverse distance to the previous target occurrence
  - whether the top hard negative appeared earlier in the current context
  - inverse distance to the previous hard-negative occurrence
- injected those features into:
  - the per-layer pattern summary stream
  - the residual/control token-conditioned observation slots
  - the AURORA observation-strength / recall-pressure computation

This was intentionally generic:

- no benchmark-specific token-role hacks
- no new optimizer family
- no new learning-rate or budget tuning

Verification:

- `cmake --build /home/robert/dev/glades-ml/build -j4`
- `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
- `./unit-tests/build/glades-unit-tests atlas-controller`
- focused AURORA reruns:
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-large --repeats 3 --variant aurora`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-document --repeats 3 --variant aurora`
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-corpus-large --repeats 3 --variant aurora`

Results:

- `token-lm-large`
  - `ATLAS-AURORA`: `testNLL=4.58343 +/- 0.00283`
  - effectively unchanged

- `token-lm-document`
  - `ATLAS-AURORA`: `testNLL=5.54653 +/- 0.00819`
  - exactly unchanged at benchmark precision

- `token-lm-corpus-large`
  - `ATLAS-AURORA`: `testNLL=6.64533 +/- 0.00463`
  - exactly unchanged at benchmark precision

Interpretation:

- The failure is now harder to blame on missing generic retrieval-distance summaries.
- The new observables are real and they do flow through the AURORA controller, but they do not change held-out NLL on the transformer stress cases.
- So the bottleneck is unlikely to be:
  - simple backbone schedule choice
  - local head-budget choice
  - or missing generic target/hard-negative repeat-distance information
- The remaining transformer gap is therefore likely to require one of:
  - a fundamentally different observable family
  - a fundamentally different actuation path
  - or accepting that the current ATLAS/AURORA line is not competitive for realistic LM optimization

Updated recommendation:

- Stop local AURORA transformer tuning on the current branch family.
- Keep:
  - `token-lm-document` as the primary harsh transformer benchmark
  - `token-lm-corpus-large` as the secondary corpus-style benchmark
  - `AdamW` as the transformer default
- If work continues, it should be a qualitatively new line, not another AURORA observable/controller refinement.

## April 11, 2026: GEODE root cause and first real transformer results

I implemented a transformer-only `ATLAS-GEODE` branch as the first geometry-first Adam-replacement candidate:

- diagonal Adam-style moment backbone
- ATLAS active subspace refresh and low-rank geometry
- optional cheap predictive correction in active coordinates
- no ASTER/AURORA-style output controller in the hot path

### Root cause of the initial failure

The first GEODE runs failed immediately on the token embedding update with:

- `SGDHelper_Transformer: GEODE tokE update entered NaN recovery`

That was not a numerical-stability failure in the low-rank solve. The actual mechanism was a transformer state-construction bug:

- GEODE reuses Adam-style diagonal moment buffers (`v*`, `v2*`) for its backbone
- transformer initialization in `network.cpp` only allocated those moment buffers for:
  - plain `AdamW`
  - `ATLAS-AURORA` with `auroraAdamwBackbone=1`
- so GEODE entered its first `tokE` update with:
  - `tokE.size() = gTokE.size() > 0`
  - `vTokE.size() = v2TokE.size() = 0`
- the update helper correctly rejected the buffer-size mismatch and surfaced the generic GEODE recovery error

Fix:

- extend the transformer `needAdamMoments` allocation condition to include `trainingConfig.atlas.geodeEnabled`

After that fix, GEODE ran cleanly end-to-end.

### Verification

- `cmake --build /home/robert/dev/glades-ml/build -j4`
- `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
- `./unit-tests/build/glades-unit-tests atlas-controller`

Focused benchmark reruns:

- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-large --repeats 3 --variant geode`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-document --repeats 3 --variant adamw`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-document --repeats 3 --variant base`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-document --repeats 3 --variant geode`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-corpus-large --repeats 3 --variant adamw`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-corpus-large --repeats 3 --variant base`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-corpus-large --repeats 3 --variant geode`

### Results

- `token-lm-large`
  - `AdamW`: `testNLL=4.57676 +/- 0.00637`
  - `ATLAS-BSRP`: `4.59059 +/- 0.00348`
  - `ATLAS-GEODE`: `4.56488 +/- 0.00594`

- `token-lm-document`
  - `AdamW`: `testNLL=4.71194 +/- 0.03683`
  - `ATLAS-BSRP`: `5.46411 +/- 0.01637`
  - `ATLAS-GEODE`: `4.74741 +/- 0.01612`

- `token-lm-corpus-large`
  - `AdamW`: `testNLL=6.19167 +/- 0.05172`
  - `ATLAS-BSRP`: `6.34855 +/- 0.01436`
  - `ATLAS-GEODE`: `6.30823 +/- 0.10459`

### Interpretation

- GEODE is the first geometry-first transformer branch here that remains competitive after the AURORA/HELM/controller failures.
- It is clearly better than the old BSRP transformer control on all three tested transformer cases.
- It appears to beat `AdamW` on the light control `token-lm-large`.
- On the harder realistic transformer cases:
  - it nearly closes the `token-lm-document` gap
  - but does not beat `AdamW` on `token-lm-corpus-large`
- Its update profile also looks materially healthier than prior ATLAS transformer branches:
  - higher head share than BSRP on the hard cases
  - still much lower controller complexity than AURORA-like branches

Updated recommendation:

- keep `ATLAS-GEODE` as the new primary AdamW-replacement candidate for transformers
- compare `AdamW` vs `GEODE` next on:
  - `token-lm-document`
  - `token-lm-corpus-large`
  with `repeats=10`
- if the `token-lm-document` near-match survives at `repeats=10`, GEODE becomes the main replacement line
- if it collapses, stop AdamW-replacement work on this branch and conclude the remaining gap still needs a different signal or larger-scale benchmark

## April 11, 2026: GEODE 10-repeat acceptance pass and predictive ablation

I ran the narrow acceptance pass exactly as planned:

- `AdamW` vs `ATLAS-GEODE` at `repeats=10` on:
  - `token-lm-document`
  - `token-lm-corpus-large`
- `ATLAS-GEODE` ablation with `--atlas-geode-predictive-scale 0` on the same two hard benchmarks

### Verification

- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-document --repeats 10 --variant adamw`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-document --repeats 10 --variant geode`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-corpus-large --repeats 10 --variant adamw`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-corpus-large --repeats 10 --variant geode`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-document --repeats 10 --variant geode --atlas-geode-predictive-scale 0`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-corpus-large --repeats 10 --variant geode --atlas-geode-predictive-scale 0`

### Results

- `token-lm-document`
  - `AdamW`: `testNLL=4.72491 +/- 0.04037`
  - `ATLAS-GEODE`: `4.74337 +/- 0.04541`
  - `ATLAS-GEODE (pred=0)`: `4.75211 +/- 0.04429`

- `token-lm-corpus-large`
  - `AdamW`: `testNLL=6.27182 +/- 0.10700`
  - `ATLAS-GEODE`: `6.27839 +/- 0.07367`
  - `ATLAS-GEODE (pred=0)`: `6.29094 +/- 0.06443`

### Interpretation

- The `token-lm-document` near-match survives the 10-repeat pass.
  - GEODE does not beat AdamW there.
  - But it remains very close, and far stronger than the older ATLAS transformer controls.
- On `token-lm-corpus-large`, GEODE is effectively near-tied with AdamW at this benchmark scale, but still not clearly better.
- The predictive term is helping, but only modestly:
  - removing it worsens both hard benchmarks
  - the branch still mostly lives or dies by the geometry backbone rather than the cheap predictive extrapolation
- So the correct conclusion is:
  - GEODE is a legitimate replacement candidate
  - but it has not yet cleared the promotion bar over AdamW on the hard transformer benchmarks

Updated recommendation:

- keep `ATLAS-GEODE` as the only active AdamW-replacement line for transformers
- stop creating new top-level optimizer families
- if GEODE work continues, focus next on:
  - runtime reduction
  - geometry schedule / rank ablations
  - GPU viability
- do not return to controller-style branches unless GEODE is clearly falsified

## April 11, 2026: GEODE rank and refresh-cadence ablations

I ran the next narrow ablation pass on GEODE rather than changing the optimizer family again.

Goal:

- identify whether GEODE still has a useful local tuning lever
- distinguish between:
  - active rank
  - subspace refresh cadence (`tSub`)

Method:

- screen on `token-lm-document`
- carry only the best setting to `token-lm-corpus-large`

### Document-side screen

Baseline reference from the 10-repeat pass:

- `ATLAS-GEODE` (`rank=16`, `tSub=64`): `testNLL=4.74337 +/- 0.04541`

Screen results on `token-lm-document`:

- `rank=8`, `tSub=64`:
  - `testNLL=4.73145 +/- 0.03785`
  - `applyMs=4.461 +/- 0.221`

- `rank=24`, `tSub=64`:
  - `testNLL=4.76557 +/- 0.03422`
  - `applyMs=15.396 +/- 0.599`

- `rank=16`, `tSub=32`:
  - `testNLL=4.75609 +/- 0.03446`
  - `applyMs=8.681 +/- 0.337`

- `rank=16`, `tSub=128`:
  - `testNLL=4.74693 +/- 0.03428`
  - `applyMs=8.644 +/- 0.243`

Interpretation:

- rank is the real lever
- lower rank helps both speed and held-out NLL on the harsh document benchmark
- `tSub` changes are small and do not beat the `rank=8` setting

### Corpus-large confirmation

I carried the best screened setting to `token-lm-corpus-large`:

- `ATLAS-GEODE` baseline (`rank=16`, `tSub=64`):
  - `testNLL=6.27839 +/- 0.07367`
  - `applyMs=33.374 +/- 1.517`

- `ATLAS-GEODE` (`rank=8`, `tSub=64`):
  - `testNLL=6.28702 +/- 0.08777`
  - `applyMs=15.717 +/- 0.198`

Interpretation:

- `rank=8` is effectively accuracy-neutral on `token-lm-corpus-large`
- but it cuts GEODE application cost by more than 2x
- so `rank=8` is the better operating point for the current branch

### Updated recommendation

- treat `rank=8` as the new preferred GEODE operating point for transformer experiments
- do not spend more time on `tSub` tuning
- if GEODE continues, the next step should be:
  - runtime / implementation efficiency
  - especially GPU viability
- the remaining question is no longer “is there a better small scalar schedule?”
- it is “can the rank-8 GEODE branch beat AdamW on time-to-target once the implementation cost is reduced?”

## N.39 GEODE Rank-8 Runtime Cleanup

I followed the recommendation above and ran a narrow runtime pass on the `rank=8` GEODE implementation instead of changing optimizer shape again.

Goal:

- preserve the accepted `rank=8` accuracy point
- reduce GEODE application overhead on the CPU transformer path
- decide whether the next step should be more CPU cleanup or a GPU viability pass

Method:

- profile `token-lm-document --variant geode --rank 8`
- optimize only the GEODE hot path
- rerun `token-lm-document` and `token-lm-corpus-large`

Profile finding:

- the dominant user-space hotspot was `Geode::update_weight(...)`
- `invert_small(...)` was not the problem
- the main issue was repeated scratch construction and redundant temporary materialization inside the per-weight update path

Implementation change:

- moved GEODE scratch storage into persistent `atlas::WeightState` buffers
- removed repeated per-update heap allocations in `Geode::update_weight(...)`
- projected the active gradient through the packed basis using the GEMM helper path instead of rebuilding equivalent temporaries
- kept the GEODE math unchanged

Post-change results:

- `token-lm-document`, `ATLAS-GEODE` (`rank=8`, `tSub=64`):
  - before:
    - `testNLL=4.73145 +/- 0.03785`
    - `applyMs=4.461 +/- 0.221`
  - after:
    - `testNLL=4.73145 +/- 0.03785`
    - `applyMs=3.929 +/- 0.027`

- `token-lm-corpus-large`, `ATLAS-GEODE` (`rank=8`, `tSub=64`):
  - before:
    - `testNLL=6.28702 +/- 0.08777`
    - `applyMs=15.717 +/- 0.198`
  - after:
    - `testNLL=6.28702 +/- 0.08777`
    - `applyMs=15.408 +/- 0.684`

Interpretation:

- the cleanup is behavior-preserving
- the document benchmark shows a meaningful CPU-side runtime win
- the larger corpus benchmark still improves, but only modestly
- this is enough to keep `rank=8` GEODE as the active replacement candidate
- it is not enough to justify more CPU micro-tuning as the main path

Updated recommendation:

- keep `ATLAS-GEODE rank=8` as the only active AdamW-replacement line
- stop small schedule and CPU micro-tuning after this pass
- make the next engineering gate:
  - GPU viability
  - or explicit time-to-target measurement against `AdamW`
- if GEODE cannot win on wall-clock efficiency after the runtime path is cleaned up, stop the replacement line
- if it can, then it becomes the first branch worth promoting beyond research-control status

## N.40 Explicit Wall-Clock Time-to-Target: GEODE Rank-8 vs AdamW

I ran the next gate directly instead of using more proxy metrics: explicit time-to-target sweeps for `AdamW` and `ATLAS-GEODE` (`rank=8`) on the two hard transformer benchmarks.

Method:

- single-core pinned runs with `taskset -c 2`
- `repeats=5`
- sweep `--token-epochs 1 2 3 4`
- compare measured `Train(s)` against matched held-out `TestNLL`

Commands:

- `taskset -c 2 ./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-document --token-epochs {1,2,3,4} --repeats 5 --variant adamw`
- `taskset -c 2 ./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-document --token-epochs {1,2,3,4} --repeats 5 --variant geode --rank 8`
- `taskset -c 2 ./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-corpus-large --token-epochs {1,2,3,4} --repeats 5 --variant adamw`
- `taskset -c 2 ./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-corpus-large --token-epochs {1,2,3,4} --repeats 5 --variant geode --rank 8`

### `token-lm-document`

`AdamW`:

- 1 epoch:
  - `Train(s)=0.53 +/- 0.00`
  - `TestNLL=5.08055 +/- 0.00853`
- 2 epochs:
  - `Train(s)=1.05 +/- 0.00`
  - `TestNLL=4.73422 +/- 0.03952`
- 3 epochs:
  - `Train(s)=1.58 +/- 0.01`
  - `TestNLL=4.22666 +/- 0.06805`
- 4 epochs:
  - `Train(s)=2.10 +/- 0.00`
  - `TestNLL=3.58111 +/- 0.08071`

`ATLAS-GEODE` (`rank=8`):

- 1 epoch:
  - `Train(s)=0.70 +/- 0.00`
  - `TestNLL=5.08007 +/- 0.01350`
- 2 epochs:
  - `Train(s)=1.39 +/- 0.00`
  - `TestNLL=4.73145 +/- 0.03785`
- 3 epochs:
  - `Train(s)=2.08 +/- 0.01`
  - `TestNLL=4.22376 +/- 0.11238`
- 4 epochs:
  - `Train(s)=2.78 +/- 0.01`
  - `TestNLL=3.61555 +/- 0.16011`

Time-to-target interpretation:

- At the loose target `TestNLL≈5.08`, `AdamW` gets there in `0.53s`; `GEODE` takes `0.70s`.
- At the matched target `TestNLL≈4.73`, `AdamW` reaches it in `1.05s`; `GEODE` takes `1.39s`.
- At the matched target `TestNLL≈4.22`, `AdamW` reaches it in `1.58s`; `GEODE` takes `2.08s`.

So on `token-lm-document`, GEODE does not win time-to-target anywhere on the measured curve. It tracks AdamW’s quality closely, but it is consistently about `1.3x` slower in wall-clock to hit the same NLL.

### `token-lm-corpus-large`

`AdamW`:

- 1 epoch:
  - `Train(s)=1.60 +/- 0.01`
  - `TestNLL=6.40277 +/- 0.02125`
- 2 epochs:
  - `Train(s)=3.21 +/- 0.00`
  - `TestNLL=6.25672 +/- 0.09305`
- 3 epochs:
  - `Train(s)=4.81 +/- 0.01`
  - `TestNLL=6.40044 +/- 0.08325`
- 4 epochs:
  - `Train(s)=6.44 +/- 0.01`
  - `TestNLL=6.61852 +/- 0.08902`

`ATLAS-GEODE` (`rank=8`):

- 1 epoch:
  - `Train(s)=2.04 +/- 0.01`
  - `TestNLL=6.36924 +/- 0.01752`
- 2 epochs:
  - `Train(s)=4.08 +/- 0.01`
  - `TestNLL=6.32884 +/- 0.07451`
- 3 epochs:
  - `Train(s)=6.11 +/- 0.01`
  - `TestNLL=6.35228 +/- 0.13351`
- 4 epochs:
  - `Train(s)=8.16 +/- 0.01`
  - `TestNLL=6.64143 +/- 0.15692`

Time-to-target interpretation:

- At the loose target `TestNLL≈6.40`, `AdamW` gets there in `1.60s`; `GEODE` takes `2.04s`.
- At the tighter target `TestNLL≈6.35`, `AdamW` reaches it between epochs 1 and 2 and is still faster than GEODE’s 2-epoch point.
- `GEODE` never reaches AdamW’s best measured point `TestNLL=6.25672` within 4 epochs.

So on `token-lm-corpus-large`, GEODE also fails the explicit time-to-target gate. It can be directionally competitive at loose early targets, but AdamW reaches every meaningful target faster, and GEODE does not catch up at stricter targets.

### Conclusion

- The explicit wall-clock measurement is decisive.
- `ATLAS-GEODE rank=8` remains the only transformer-side replacement candidate that is close in quality.
- But it still does **not** beat `AdamW` on time-to-target on either `token-lm-document` or `token-lm-corpus-large`.

Updated recommendation:

- do not promote GEODE as an AdamW replacement
- stop local GEODE tuning as a transformer replacement line unless the next step is:
  - a genuine GPU implementation change, or
  - a materially different benchmark regime
- for the current CPU transformer path and current benchmark set, `AdamW` remains the correct default
- GEODE is now a documented near-match control, not a successor

## N.41 GPU-Native GEODE-v2 Wiring

I followed the next recommendation and implemented the minimal GPU-native GEODE-v2 path instead of continuing CPU-only tuning.

Goal:

- keep `GEODE` as the only active AdamW-replacement line
- move the replacement question to the only remaining viable axis: GPU execution
- avoid another controller or observability branch

Implementation:

- the transformer harness now accepts:
  - `--gpu-enable 0|1`
  - `--gpu-device N`
- transformer GPU weight allocation now keeps Adam-style moment buffers when the optimizer is `ATLAS-GEODE`
- GPU ATLAS gained a residual-only entrypoint that:
  - reuses ATLAS subspace/Fisher tracking
  - skips decoupled weight decay
  - skips the full-space baseline step
  - applies only the low-rank ATLAS correction
- transformer GPU `GEODE` now runs as:
  - Adam-style batched GPU backbone update on all parameters
  - plus residual-only ATLAS geometry correction on matrix weights

This is the practical GPU analogue of the GEODE research direction:

\[
\delta_t \approx \delta_t^{\text{Adam}} + \delta_t^{\text{low-rank residual}}
\]

with the residual applied only to matrix weights and the Adam-style backbone retained everywhere.

Files changed:

- `Backend/Machine Learning/Networks/cuda/gpu_atlas.h`
- `Backend/Machine Learning/Networks/cuda/gpu_atlas.cu`
- `Backend/Machine Learning/Networks/network.cpp`
- `Backend/Machine Learning/Networks/sgd_transformer.cpp`
- `unit-tests/Backend/Machine Learning/atlas-alt-bench.cpp`

Verification:

- `cmake --build /home/robert/dev/glades-ml/build -j4`
- `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
- `./unit-tests/build/glades-unit-tests atlas-controller`
- CPU GEODE smoke:
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm --token-epochs 1 --repeats 1 --variant geode --rank 8`
- GPU-requested smoke:
  - `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm --token-epochs 1 --repeats 1 --variant geode --rank 8 --gpu-enable 1`

Result:

- the new GPU GEODE-v2 path builds cleanly
- the benchmark harness now requests CUDA correctly
- on this machine, the CUDA request reports:
  - `No CUDA devices found (err=100, count=0)`
- the run then falls back to CPU and preserves the prior GEODE result

Interpretation:

- the implementation work is complete enough to evaluate on a real CUDA host
- but this environment cannot answer the actual GPU viability question
- so the only honest conclusion here is:
  - GPU GEODE-v2 is wired and benchmarkable
  - real GPU time-to-target measurement is still outstanding

Updated recommendation:

- keep `GEODE rank=8` as the only active AdamW-replacement line
- do not make more CPU-side optimizer-shape changes
- run the new `--gpu-enable 1` GEODE benchmark ladder on a machine with a working NVIDIA driver
- use the same explicit gate as before:
  - `token-lm-document`
  - `token-lm-corpus-large`
  - matched-NLL wall-clock time-to-target against `AdamW`

### April 11, 2026: `BiMAP-lite` true blockwise matrix-preconditioner prototype

I followed the next replacement recommendation and implemented the minimal practical `BiMAP-lite` branch instead of adding another controller family.

Goal:

- test a true matrix-native preconditioner rather than another low-rank residual shell
- keep Adam-style robustness on vectors/biases
- replace transformer matrix-weight updates with a two-sided row/column preconditioner

Implementation:

- added `ATLAS-BIMAP` config/plumbing in:
  - `Backend/Machine Learning/Networks/training_config.h`
  - `Backend/Machine Learning/Networks/checkpoint_persistence.cpp`
  - `Backend/Machine Learning/Networks/transformer_model_state.inc`
  - `Backend/Machine Learning/Networks/network.cpp`
  - `unit-tests/Backend/Machine Learning/atlas-alt-bench.cpp`
- added persistent `BiMAPWeightState` and `bimapUpdate(...)` in:
  - `Backend/Machine Learning/Networks/atlas_optimizer.h`
  - `Backend/Machine Learning/Networks/atlas_optimizer.cpp`
- wired transformer matrix weights through `BiMAP` in:
  - `Backend/Machine Learning/Networks/sgd_transformer.cpp`

Current `BiMAP-lite` update:

\[
\Delta W_t \approx -\eta_t \frac{\widetilde M_t}{(\sqrt{\hat V_t}+\epsilon)\odot S_r \odot S_c}
\]

where:

- `\widetilde M_t` is Adam first moment plus a bounded secant-style predictive blend
- `S_r` is a row anisotropy scale from an EMA of mean row gradient squares
- `S_c` is a column anisotropy scale from an EMA of mean column gradient squares

This is a real two-sided blockwise preconditioner, but still the minimal diagonal-row/column-factor version, not the full low-rank SPD factor design.

Harness note:

- token alt-bench was initially blocked by a harness regression:
  - `setTrainingConfig: unknown positionalEncoding`
- root cause was the benchmark constructing token/regression configs from `net->getTrainingConfig()` instead of a fresh `TrainingConfig`
- switching the harness to explicit fresh config construction restored valid transformer enum defaults

Verification:

- `cmake --build /home/robert/dev/glades-ml/build -j4`
- `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
- `./unit-tests/build/glades-unit-tests atlas-controller`
- focused transformer ladder:
  - `token-lm-large --repeats 3 --variant adamw|geode|bimap`
  - `token-lm-document --repeats 3 --variant adamw|geode|bimap`
  - `token-lm-corpus-large --repeats 3 --variant adamw|geode|bimap`

Results:

- `token-lm-large`
  - `AdamW`: `4.57676 +/- 0.00637`, `applyMs=0.056`
  - `GEODE`: `4.56488 +/- 0.00594`, `applyMs=5.211`
  - `BiMAP`: `4.57120 +/- 0.00552`, `applyMs=0.172`
- `token-lm-document`
  - `AdamW`: `4.71194 +/- 0.03683`, `applyMs=0.892`
  - `GEODE`: `4.74741 +/- 0.01612`, `applyMs=11.216`
  - `BiMAP`: `4.72577 +/- 0.05053`, `applyMs=1.610`
- `token-lm-corpus-large`
  - `AdamW`: `6.19167 +/- 0.05172`, `applyMs=1.748`
  - `GEODE`: `6.30822 +/- 0.10461`, `applyMs=39.520`
  - `BiMAP`: `6.27305 +/- 0.09779`, `applyMs=2.883`

Interpretation:

- `BiMAP-lite` is the first true matrix-preconditioner branch in this line that is both:
  - materially cheaper than `GEODE`
  - still reasonably close to `AdamW` on the hard transformer benchmarks
- but it still does not beat `AdamW` on either `token-lm-document` or `token-lm-corpus-large`
- so it is a valid next-generation branch, not a replacement winner yet

Updated recommendation:

- keep `BiMAP-lite` as the active true matrix-preconditioner branch
- treat `GEODE` as the older low-rank-residual control
- if optimizer replacement work continues, the next serious step should be:
  - full low-rank row/column SPD factors
  - GPU-first implementation
  - explicit time-to-target gate against `AdamW`

### April 11, 2026: `BiMAP-v2` low-rank factor prototype and bounded micro-benchmark

Implementation:

- extended `BiMAPWeightState` with low-rank row/column factor buffers and capture diagnostics in:
  - `Backend/Machine Learning/Networks/atlas_optimizer.h`
- added a low-rank `BiMAP-v2` path in:
  - `Backend/Machine Learning/Networks/atlas_optimizer.cpp`
  - using alternating row/column subspace iteration plus Woodbury-style left/right inverse application
- added `bimapLowRankEnabled` plumbing in:
  - `Backend/Machine Learning/Networks/training_config.h`
  - `Backend/Machine Learning/Networks/checkpoint_persistence.cpp`
  - `unit-tests/Backend/Machine Learning/atlas-alt-bench.cpp`
- added a bounded transformer micro-benchmark entrypoint in:
  - `unit-tests/Backend/Machine Learning/atlas-test.cpp`
  - `unit-tests/Backend/Machine Learning/atlas-test.h`
  - `unit-tests/main.cpp`

Runtime note:

- the hard transformer alt-bench loop is no longer a practical local CPU iteration path after this change
- even `AdamW` timed out locally on:
  - `token-lm-large --token-epochs 1 --repeats 1 --variant adamw` at `60s`
  - `token-lm-document --token-epochs 1 --repeats 1 --variant adamw` at `180s`
- so the local verification gate for this pass was reduced to:
  - build
  - `atlas-controller`
  - `atlas-bimap-micro`

Verification:

- `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
- `./unit-tests/build/glades-unit-tests atlas-controller`
- `./unit-tests/build/glades-unit-tests atlas-bimap-micro | rg '^(AdamW|BiMAP)'`

Micro-benchmark result:

- `AdamW`: `TrainNLL=2.82223`, `TrainPPL=16.81433`, `0.002s`
- `BiMAP-lite`: `TrainNLL=2.82151`, `TrainPPL=16.80225`, `0.002s`
- `BiMAP-v2-0`: `TrainNLL=2.81736`, `TrainPPL=16.73262`, `0.002s`
- `BiMAP-v2`: `TrainNLL=2.80104`, `TrainPPL=16.46183`, `0.001s`

Interpretation:

- the low-rank factor path is functionally live and directionally better than `BiMAP-lite` on the tiny transformer sanity check
- but the hard benchmark gate remains unresolved on this CPU path
- the next real decision still has to come from GPU-side `token-lm-document` / `token-lm-corpus-large`, not from more local CPU tuning

### April 11, 2026: `BiMAP` GPU gate and acceptance verdict

Ran the GPU-side `BiMAP` gate with:

- `scripts/run_bimap_gpu_gate.sh`
- `scripts/run_bimap_gpu_gate.sh --skip-build --acceptance`

Artifacts:

- `artifacts/bimap_gpu_gate_20260411-083847/epoch_sweep_summary.tsv`
- `artifacts/bimap_gpu_gate_20260411-084520/epoch_sweep_summary.tsv`
- `artifacts/bimap_gpu_gate_20260411-084520/acceptance_summary.tsv`

Acceptance summary (`repeats=10`, default epoch count):

- `token-lm-document`
  - `AdamW`: `Train(s)=0.10 +/- 0.06`, `TestNLL=4.87531 +/- 0.05549`
  - `BiMAP-lite`: `0.69 +/- 0.01`, `4.73995 +/- 0.03700`
  - `BiMAP-v2-0`: `0.72 +/- 0.00`, `4.74647 +/- 0.03648`
  - `BiMAP-v2`: `0.76 +/- 0.00`, `4.73995 +/- 0.03700`
- `token-lm-corpus-large`
  - `AdamW`: `0.19 +/- 0.07`, `6.29585 +/- 0.07842`
  - `BiMAP-lite`: `2.05 +/- 0.02`, `6.27656 +/- 0.08591`
  - `BiMAP-v2-0`: `2.06 +/- 0.01`, `6.28066 +/- 0.08327`
  - `BiMAP-v2`: `2.21 +/- 0.01`, `6.27656 +/- 0.08591`

Epoch-sweep readout:

- `token-lm-document`
  - `AdamW` remains much faster at every point
  - `BiMAP-lite` / `BiMAP-v2` produce lower `TestNLL` by epoch count, but not by wall-clock
  - for example:
    - `AdamW`: `4.84283` at `0.13s`
    - `BiMAP-lite`: `4.73404` at `0.69s`
    - `BiMAP-v2`: `4.73404` at `0.77s`
  - and `AdamW` already reaches `4.72663` by `0.21s`
- `token-lm-corpus-large`
  - `AdamW` dominates the time-to-target frontier
  - best measured `AdamW`: `6.24807` at `0.21s`
  - `BiMAP-lite`: `6.26362` at `2.01s`
  - `BiMAP-v2-0`: `6.26652` at `2.04s`
  - `BiMAP-v2`: `6.26362` at `2.20s`

Conclusion:

- `BiMAP` is a real matrix-preconditioner line, not a dead branch
- but it does **not** replace `AdamW` on the actual promotion metric:
  - GPU wall-clock time-to-target
- `BiMAP-v2` also failed to justify itself over `BiMAP-lite`
  - it is slightly slower
  - it does not produce a consistent accuracy gain on the hard transformer benchmarks

Updated recommendation:

- freeze `AdamW` as the transformer default
- keep `BiMAP-lite` as the only matrix-preconditioner control worth retaining
- stop `BiMAP-v2` low-rank-factor refinement on this line
- if optimizer replacement work continues, change matrix-preconditioner family rather than tuning `BiMAP`

### April 11, 2026: `PACT` prototype (`Promoted Adaptive Compressed Tensor-preconditioner`)

Implemented a compute-aware blockwise promotion prototype that keeps exact `AdamW` fallback on demoted or unsupported blocks and only activates a two-sided matrix preconditioner when a local gain proxy beats an analytical overhead proxy.

Main code:

- config surface:
  - `Backend/Machine Learning/Networks/training_config.h`
  - `Backend/Machine Learning/Networks/checkpoint_persistence.cpp`
- optimizer core:
  - `Backend/Machine Learning/Networks/atlas_optimizer.h`
  - `Backend/Machine Learning/Networks/atlas_optimizer.cpp`
- transformer integration:
  - `Backend/Machine Learning/Networks/transformer_model_state.inc`
  - `Backend/Machine Learning/Networks/network.cpp`
  - `Backend/Machine Learning/Networks/sgd_transformer.cpp`
- harness and dedicated tests:
  - `unit-tests/Backend/Machine Learning/atlas-alt-bench.cpp`
  - `unit-tests/Backend/Machine Learning/atlas-test.cpp`
  - `unit-tests/Backend/Machine Learning/atlas-test.h`
  - `unit-tests/main.cpp`

Key mechanics:

- `PACT` keeps standard Adam moments on every matrix block
- builds optional row/column geometry from diagonal-plus-low-rank block factors
- adds a bounded secant-style predictive transport term
- computes both:
  - an exact Adam-style fallback step
  - a promoted two-sided preconditioned step
- promotes only when
  - predicted preconditioned gain
  - minus analytical cost penalty
  exceeds the block EMA threshold
- otherwise falls back exactly to `AdamW`

Verification passed:

- `cmake --build /home/robert/dev/glades-ml/build -j4`
- `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
- `./unit-tests/build/glades-unit-tests atlas-controller`
- `./unit-tests/build/glades-unit-tests atlas-pact-core`
- `./unit-tests/build/glades-unit-tests atlas-pact-micro`

Dedicated PACT tests:

- `atlas-pact-core`
  - confirms exact Adam-style fallback when promotion/geometry are disabled
  - confirms promotion engages and changes the update on anisotropic blocks
- `atlas-pact-micro`
  - `AdamW`: `TrainNLL=2.80569`, `TrainPPL=16.53852`
  - `BiMAP-lite`: `2.82781`, `16.90845`
  - `PACT-lite`: `2.80709`, `16.56161`
  - `PACT-v2-0`: `2.82075`, `16.78942`
  - `PACT-v2`: `2.81469`, `16.68794`

Minimal end-to-end transformer harness readout (`epochs=1`, `repeats=1`):

- `token-lm-large`
  - `AdamW`: `TestNLL=4.57676`
  - `ATLAS-PACT`: `4.56867`
  - `applyMs`: `AdamW 0.849`, `PACT 2.135`
- `token-lm-document`
  - `AdamW`: `Train(s)=0.39`, `TestNLL=5.06561`
  - `ATLAS-PACT`: `0.47`, `5.07358`
  - `applyMs`: `AdamW 0.849`, `PACT 11.914`
- `token-lm-corpus-large`
  - `AdamW`: `Train(s)=0.93`, `TestNLL=6.40179`
  - `ATLAS-PACT`: `1.25`, `6.39441`
  - `applyMs`: `AdamW 1.720`, `PACT 23.691`

Conclusion:

- `PACT` is a real working branch with exact Adam fallback and live promotion logic
- it survives correctness testing and bounded transformer sanity checks
- on the current hard transformer harness it shows mixed early-quality behavior:
  - slightly better than `AdamW` on `token-lm-large`
  - slightly worse on `token-lm-document`
  - slightly better on `token-lm-corpus-large`
- but the systems cost is still too high to argue for it as an `AdamW` replacement
- the next serious gate, if this line continues, should be explicit GPU time-to-target rather than more local CPU tuning

### April 11, 2026: `PACT-lite` minimal GPU path

Implemented the first real GPU-capable `PACT-lite` transformer path.

Main code:

- CUDA optimizer/state:
  - `Backend/Machine Learning/Networks/cuda/gpu_atlas.h`
  - `Backend/Machine Learning/Networks/cuda/gpu_atlas.cu`
  - `Backend/Machine Learning/Networks/cuda/gpu_transformer_state.h`
- transformer runtime:
  - `Backend/Machine Learning/Networks/sgd_transformer.cpp`
- benchmark gate:
  - `scripts/run_pact_gpu_gate.sh`

Design:

- keep the existing batched GPU `AdamW` update as the exact backbone
- remove the old blanket GPU rejection for `pactEnabled`
- run `PACT-lite` only as a residual on top of the already-applied GPU Adam step
- use only diagonal row/column anisotropy on GPU
  - no low-rank PACT factors
  - no predictive transport
- refresh promotion stats only on the configured cadence
- score each block by:
  - predicted preconditioned gain
  - minus Adam gain
  - minus analytical cost penalty
- fall back exactly to `AdamW` whenever the block is demoted

Verification passed locally:

- `cmake --build /home/robert/dev/glades-ml/build -j4`
- `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
- `./unit-tests/build/glades-unit-tests atlas-controller`
- `./unit-tests/build/glades-unit-tests atlas-pact-core`
- `./unit-tests/build/glades-unit-tests atlas-pact-micro`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm --token-epochs 1 --repeats 1 --variant pact --gpu-enable 1 --gpu-device 0`

Important scope note:

- this sandbox still cannot execute the real CUDA training path
- the local `--gpu-enable 1` smoke only verifies that the new branch compiles, routes correctly, and degrades cleanly when a CUDA device is unavailable
- the actual decision gate is now:
  - `scripts/run_pact_gpu_gate.sh --gpu-device 0`
  - `scripts/run_pact_gpu_gate.sh --gpu-device 0 --acceptance`

Current recommendation:

- `PACT-v2` stays dropped
- `PACT-lite` is now the only active PACT line
- the next decision is purely empirical:
  - whether GPU `PACT-lite` can beat `AdamW` on explicit wall-clock time-to-target on `token-lm-document` or `token-lm-corpus-large`

### April 11, 2026: `PACT-lite` GPU gate verdict

Ran the real GPU gate on a CUDA host with:

- `scripts/run_pact_gpu_gate.sh --gpu-device 0`
- `scripts/run_pact_gpu_gate.sh --gpu-device 0 --skip-build --acceptance`

Artifacts:

- `artifacts/pact_gpu_gate_20260411-104605/epoch_sweep_summary.tsv`
- `artifacts/pact_gpu_gate_20260411-105654/epoch_sweep_summary.tsv`
- `artifacts/pact_gpu_gate_20260411-105654/acceptance_summary.tsv`

Acceptance summary:

- `token-lm-document`
  - `AdamW`: `Train(s)=0.10 +/- 0.05`, `TestNLL=4.87318 +/- 0.04835`
  - `PACT-lite`: `0.12 +/- 0.05`, `4.91364 +/- 0.10252`
- `token-lm-corpus-large`
  - `AdamW`: `0.18 +/- 0.06`, `6.28685 +/- 0.06764`
  - `PACT-lite`: `0.22 +/- 0.05`, `6.30785 +/- 0.11172`

Epoch-sweep readout:

- `token-lm-document`
  - `PACT-lite` showed a small per-epoch quality signal late in the sweep
  - but not a stable wall-clock advantage
  - example:
    - `AdamW`: `4.71289` at `0.20s`
    - `PACT-lite`: `4.72488` at `0.24s`
- `token-lm-corpus-large`
  - `AdamW` kept the stronger frontier throughout
  - example:
    - `AdamW`: `6.25968` at `0.20s`
    - `PACT-lite`: `6.32338` at `0.24s`

Conclusion:

- `PACT-lite` is a real GPU branch, not a dead prototype
- but it does **not** beat `AdamW` on the actual decision metric:
  - wall-clock time-to-target at matched or better validation NLL
- the document-benchmark hint from the 5-repeat sweep does not survive the 10-repeat acceptance pass

Updated recommendation:

- freeze `AdamW` as the transformer default
- keep `PACT-lite` only as a control branch
- stop `PACT` refinement on this line
- if optimizer replacement work continues, change family again rather than tuning `PACT-lite`

### April 11, 2026: `ATLAS-KRON` block-factor preconditioner prototype

Implemented a new transformer-only matrix-preconditioner control branch, `ATLAS-KRON`.

Design:

- exact `AdamW` fallback when `kronGeometryScale=0`
- full row/column covariance EMA on each matrix block
- two-sided inverse-square-root factor apply on Adam-normalized momentum
- bounded secant transport inside the momentum signal
- no controller shell, no output-side sidecar, no dense fusion

Files touched:

- `Backend/Machine Learning/Networks/atlas_optimizer.h`
- `Backend/Machine Learning/Networks/atlas_optimizer.cpp`
- `Backend/Machine Learning/Networks/training_config.h`
- `Backend/Machine Learning/Networks/checkpoint_persistence.cpp`
- `Backend/Machine Learning/Networks/network.cpp`
- `Backend/Machine Learning/Networks/transformer_model_state.inc`
- `Backend/Machine Learning/Networks/sgd_transformer.cpp`
- `unit-tests/Backend/Machine Learning/atlas-alt-bench.cpp`
- `unit-tests/Backend/Machine Learning/atlas-test.cpp`
- `unit-tests/Backend/Machine Learning/atlas-test.h`
- `unit-tests/main.cpp`

Verification passed:

- `cmake --build /home/robert/dev/glades-ml/build -j4`
- `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
- `./unit-tests/build/glades-unit-tests atlas-controller`
- `./unit-tests/build/glades-unit-tests atlas-kron-core`
- `./unit-tests/build/glades-unit-tests atlas-kron-micro`

Correctness checks:

- `KRON` fallback matches exact Adam-style updates when `kronGeometryScale=0`
- anisotropic test blocks diverge from Adam fallback and record nontrivial block conditioning

Micro-benchmark:

- `AdamW`: `TrainNLL=2.81549`
- `BiMAP-lite`: `2.80846`
- `KRON-0`: `2.81233`
- `KRON`: `2.82362`

Focused transformer smoke (`epochs=1`, `repeats=1`):

- `token-lm-large`
  - `AdamW`: `TestNLL=4.56885`, `Train(s)=0.00`, `applyMs=0.077`
  - `KRON`: `4.57232`, `0.03`, `13.732`
- `token-lm-document`
  - `AdamW`: `5.06561`, `0.36s`, `applyMs=0.807`
  - `KRON`: `5.06384`, `2.90s`, `250.510`
- `token-lm-corpus-large`
  - `AdamW`: `6.40179`, `0.91s`, `applyMs=1.633`
  - `KRON`: `6.39331`, `10.24s`, `626.725`

Interpretation:

- `KRON` is not a cosmetic variant; it is a genuinely different approximation class from `BiMAP`/`PACT`
- but the systems-cost defect is already dominant on the bounded transformer smoke
- it buys only tiny 1-epoch quality changes while introducing two to three orders of magnitude more apply-time overhead

Verdict:

- keep `ATLAS-KRON` only as a falsification/control branch
- do **not** spend a GPU-kernel round on it in the current form
- the branch failed the minimal local viability gate before the real GPU time-to-target gate
- if replacement work continues, the next family should be even more compute-disciplined than `KRON`, not a fuller version of it

## April 11, 2026: `ATLAS-MUON-lite` selective orthogonalized-momentum prototype

Implemented a new matrix-only optimizer branch, `ATLAS-MUON-lite`, as the next family after `KRON`:

\[
\Delta W = -\eta \left[(1-\gamma)\,D^{-1}\hat m + \gamma\,s\,\operatorname{Polar}(D^{-1}\hat m)\right]
\]

with exact `AdamW` fallback when:

- `muonGeometryScale = 0`
- the matrix block is too rectangular: `max(m,n) / min(m,n) > muonMaxAspect`
- the block is too small: `min(m,n) < muonMinDim`

Scope:

- no GPU path yet
- only matrix blocks use the MUON update
- embeddings, biases, norms, and ineligible matrix blocks stay on exact Adam-style updates
- bounded secant transport is optional and small (`muonPredictiveScale`)

Files touched:

- `Backend/Machine Learning/Networks/training_config.h`
- `Backend/Machine Learning/Networks/checkpoint_persistence.cpp`
- `Backend/Machine Learning/Networks/network.cpp`
- `Backend/Machine Learning/Networks/transformer_model_state.inc`
- `Backend/Machine Learning/Networks/atlas_optimizer.h`
- `Backend/Machine Learning/Networks/atlas_optimizer.cpp`
- `Backend/Machine Learning/Networks/sgd_transformer.cpp`
- `unit-tests/Backend/Machine Learning/atlas-alt-bench.cpp`
- `unit-tests/Backend/Machine Learning/atlas-test.cpp`
- `unit-tests/Backend/Machine Learning/atlas-test.h`
- `unit-tests/main.cpp`

Verification passed:

- `cmake --build /home/robert/dev/glades-ml/build -j4`
- `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
- `./unit-tests/build/glades-unit-tests atlas-controller`
- `./unit-tests/build/glades-unit-tests atlas-muon-core`
- `./unit-tests/build/glades-unit-tests atlas-muon-micro`

Correctness checks:

- `MUON` fallback matches exact Adam-style updates when `muonGeometryScale=0`
- eligible square blocks diverge from Adam fallback on the second step once moment history exists

Micro-benchmark:

- `AdamW`: `TrainNLL=2.80163`
- `MUON-0`: `2.80407`
- `MUON`: `2.81027`

Focused transformer smoke (`epochs=1`, `repeats=1`):

- `token-lm-large`
  - `AdamW`: `TestNLL=4.56885`, `Train(s)=0.01`, `applyMs=0.063`
  - `ATLAS-MUON`: `4.57659`, `0.01`, `0.927`
- `token-lm-document`
  - `AdamW`: `5.06561`, `0.30s`, `applyMs=0.849`
  - `ATLAS-MUON`: `5.04163`, `0.83s`, `9.823`
- `token-lm-corpus-large`
  - `AdamW`: `6.40179`, `1.01s`, `applyMs=1.701`
  - `ATLAS-MUON`: `6.21710`, `2.24s`, `18.769`

Interpretation:

- `MUON-lite` is directionally alive on the hard transformer cases in a way `KRON` was not: it materially improves bounded 1-epoch test NLL on `token-lm-document` and `token-lm-corpus-large`
- but its CPU systems cost is still far too high to make it a wall-clock replacement candidate
- the branch is therefore worth keeping alive, but only as the new leading replacement family for a real GPU time-to-target gate

Verdict:

- keep `ATLAS-MUON-lite` as the only active post-`KRON` replacement candidate
- do **not** infer an `AdamW` win from these bounded CPU results
- the next serious step, if this line continues, is a GPU-first implementation and explicit `T_\epsilon` measurement on:
  - `token-lm-document`
  - `token-lm-corpus-large`

## April 11, 2026: minimal GPU MUON-lite residual path and gate script

Implemented a minimal GPU `MUON-lite` route by reusing the existing batched Adam
backbone and adding a host-assisted orthogonalized residual on eligible matrix
blocks. This is intentionally a bridge implementation:

- exact AdamW backbone remains the hot path
- GPU `MUON-lite` only applies a residual correction after Adam
- all unsensed or ineligible blocks degenerate exactly to AdamW
- no claim of fused optimality yet; the purpose is to make the real gate runnable

Files touched:

- `Backend/Machine Learning/Networks/cuda/gpu_atlas.h`
- `Backend/Machine Learning/Networks/cuda/gpu_atlas.cu`
- `Backend/Machine Learning/Networks/cuda/gpu_transformer_state.h`
- `Backend/Machine Learning/Networks/sgd_transformer.cpp`
- `scripts/run_muon_gpu_gate.sh`

Verification passed:

- `cmake --build /home/robert/dev/glades-ml/build -j4`
- `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
- `./unit-tests/build/glades-unit-tests atlas-controller`
- `./unit-tests/build/glades-unit-tests atlas-muon-core`
- `./unit-tests/build/glades-unit-tests atlas-muon-micro`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm --token-epochs 1 --repeats 1 --variant muon --gpu-enable 1 --gpu-device 0`

Local sweep artifact:

- `artifacts/muon_gpu_gate_20260411-120536/epoch_sweep_summary.tsv`

Important caveat:

- this sandbox still does **not** provide a stable real CUDA gate
- the raw `AdamW` logs in that sweep explicitly report `No CUDA devices found`
- the resulting frontier must therefore be treated as CPU fallback / directional only, not as a valid GPU `T_\epsilon` decision

Directional fallback-only sweep summary:

- `token-lm-document`
  - `AdamW`
    - `1 epoch`: `0.30s`, `TestNLL=5.08055`
    - `2 epochs`: `0.61s`, `4.73422`
    - `3 epochs`: `0.92s`, `4.22666`
    - `4 epochs`: `1.22s`, `3.58111`
  - `ATLAS-MUON`
    - `1 epoch`: `0.70s`, `4.99930`
    - `2 epochs`: `1.42s`, `4.35125`
    - `3 epochs`: `2.12s`, `3.52163`
    - `4 epochs`: `2.83s`, `2.97537`
- `token-lm-corpus-large`
  - `AdamW`
    - `1 epoch`: `0.89s`, `TestNLL=6.40277`
    - `2 epochs`: `1.77s`, `6.25672`
    - `3 epochs`: `2.66s`, `6.40044`
    - `4 epochs`: `3.56s`, `6.61852`
  - `ATLAS-MUON`
    - `1 epoch`: `2.09s`, `6.25034`
    - `2 epochs`: `4.19s`, `6.27432`
    - `3 epochs`: `6.28s`, `6.53348`
    - `4 epochs`: `8.37s`, `6.74450`

Interpretation:

- `MUON-lite` remains interesting as a quality-per-epoch / quality-per-step family
- the current local sweep does **not** establish a GPU wall-clock win
- the only valid next decision is to run `scripts/run_muon_gpu_gate.sh` on a host with stable CUDA visibility

Verdict:

- GPU `MUON-lite` is now implemented and benchmarkable
- no valid promotion or rejection decision should be made from the sandbox sweep
- keep the line alive only until a real GPU host runs the gate

## April 11, 2026: fixed real GPU alt-bench transformer init crash; ran valid MUON GPU gate

The real CUDA crash in `atlas-alt-bench` was not a generic transformer GPU
allocator failure. The same shapes ran cleanly in
`transformer-gpu-bench`. The crash was specific to the alt-bench path enabling
benchmark-only optimizer-gap diagnostics before the GPU fast path was selected.

Root-cause fix:

- moved the `captureOptimizerGapDiagnostics` reset block in
  `Backend/Machine Learning/Networks/sgd_transformer.cpp` to run only after the
  GPU fast path declines and the CPU fallback is actually taken
- restored `atlas-alt-bench` token-LM configs to keep
  `captureOptimizerGapDiagnostics = true`

Why this is the right fix:

- the gap-reset block is benchmark-only host bookkeeping
- it has no role in the GPU epoch path itself
- `atlas-alt-bench` no longer crashes on real CUDA for:
  - `token-lm --variant adamw --gpu-enable 1`
  - `token-lm-document --variant adamw --gpu-enable 1`

Verification:

- `cmake --build /home/robert/dev/glades-ml/build -j4`
- `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
- `./unit-tests/build/glades-unit-tests atlas-controller`
- real CUDA smoke:
  - `stdbuf -oL ./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm --token-epochs 1 --repeats 1 --variant adamw --gpu-enable 1 --gpu-device 0`
  - `stdbuf -oL ./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-document --token-epochs 1 --repeats 1 --variant adamw --gpu-enable 1 --gpu-device 0`

Real GPU MUON gate artifact:

- `artifacts/muon_gpu_gate_20260411-130403/epoch_sweep_summary.tsv`

Real GPU frontier summary:

- `token-lm-document`
  - `AdamW`
    - `1 epoch`: `0.08s`, `TestNLL=5.09990`
    - `2 epochs`: `0.12s`, `4.84727`
    - `3 epochs`: `0.16s`, `4.68956`
    - `4 epochs`: `0.20s`, `4.75030`
  - `ATLAS-MUON`
    - `1 epoch`: `0.49s`, `5.04205`
    - `2 epochs`: `0.98s`, `4.79342`
    - `3 epochs`: `1.41s`, `4.64563`
    - `4 epochs`: `1.91s`, `4.60492`
- `token-lm-corpus-large`
  - `AdamW`
    - `1 epoch`: `0.13s`, `TestNLL=6.34261`
    - `2 epochs`: `0.20s`, `6.20495`
    - `3 epochs`: `0.29s`, `6.30748`
    - `4 epochs`: `0.37s`, `6.50887`
  - `ATLAS-MUON`
    - `1 epoch`: `1.36s`, `6.24589`
    - `2 epochs`: `2.70s`, `6.30244`
    - `3 epochs`: `4.03s`, `6.39071`
    - `4 epochs`: `5.36s`, `6.58369`

Interpretation:

- `MUON-lite` buys real quality-per-epoch on `token-lm-document`
- it still fails the actual gate because it is far slower in wall-clock
- on `token-lm-corpus-large`, it helps only at the loosest 1-epoch point and
  loses the frontier afterward

Verdict:

- the alt-bench GPU gate is now valid again
- `ATLAS-MUON-lite` is falsified as an `AdamW` replacement on explicit GPU
  time-to-target
- keep it only as a control if needed; do not spend more time refining it

## April 11, 2026: Nsight Systems profile shows MUON-lite is host/orchestration bound, not kernel bound

I added a reusable profiling script:

- `scripts/run_muon_nsys_profile.sh`

The important robustness fix is that the script no longer depends on Nsight
auto-importing `.qdstrm` traces to `.nsys-rep`; it explicitly invokes
`QdstrmImporter` before running `nsys stats`.

Profiled pair:

- `token-lm-document`
- `epochs=1`
- `AdamW` vs `ATLAS-MUON-lite`
- real CUDA host

Artifact:

- `artifacts/muon_nsys_20260411-132148`

Observed training result for the profiled run:

- `AdamW`: `Train(s)=0.49`, `TestNLL=5.08509`
- `ATLAS-MUON-lite`: `Train(s)=0.97`, `TestNLL=5.06455`

So the profile is representative of the earlier gate:

- `MUON-lite` buys a small quality improvement
- but still roughly doubles wall-clock

Nsight summary totals:

- `AdamW`
  - CUDA API time: `62.532 ms`
  - GPU kernel time: `44.182 ms`
  - GPU memory op time: `1.353 ms`
- `ATLAS-MUON-lite`
  - CUDA API time: `93.238 ms`
  - GPU kernel time: `46.218 ms`
  - GPU memory op time: `5.087 ms`

Key breakdown:

- kernel time is almost unchanged:
  - `44.182 ms` -> `46.218 ms`
- the slowdown is mostly outside the core math:
  - CUDA API time rises by about `49%`
  - GPU memory op time rises by about `3.8x`
  - `cudaMemcpy` calls jump from `306` to `2370`
  - `cudaStreamSynchronize` calls/total time rise sharply
  - OS runtime waiting roughly doubles

Important kernel detail:

- `muon_lite_apply_residual_kernel` is present and small:
  - `1.045 ms` total across `576` launches
- the dominant transformer kernels remain the same flash-attention and GEMM
  kernels as `AdamW`

Interpretation:

- the current `MUON-lite` slowdown is not primarily device-side linear algebra
- it is primarily host-assisted orchestration, extra synchronization, and
  device<->host movement
- therefore a fully device-native `MUON-v2` remains the only technically
  defensible continuation of this line

Decision:

- do not tune `MUON-lite`
- only continue if willing to replace the host-assisted residual path with a
  fully device-native implementation
- otherwise stop optimizer-replacement work and keep `AdamW` as the transformer
  default

## April 11, 2026: fully device-native `MUON-lite` still loses the real CUDA gate

I replaced the old host-assisted GPU MUON bridge with a fully device-native
path.

Main implementation changes:

- `Backend/Machine Learning/Networks/cuda/gpu_atlas.h`
- `Backend/Machine Learning/Networks/cuda/gpu_atlas.cu`

What changed structurally:

- removed per-step host downloads of Adam moments for MUON blocks
- removed host-side orthogonalization and residual assembly
- moved predictive trust and signal scaling fully onto the device
- reused the existing tiny on-device Cholesky path for the MUON core solve
- kept exact `AdamW` fallback for ineligible blocks

Verification:

- `cmake --build /home/robert/dev/glades-ml/build -j4`
- `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
- `./unit-tests/build/glades-unit-tests atlas-controller`
- `./unit-tests/build/glades-unit-tests atlas-muon-core`
- `./unit-tests/build/glades-unit-tests atlas-muon-micro`
- real CUDA smoke:
  - `atlas-alt-bench --mode token-lm-document --token-epochs 1 --repeats 1 --variant adamw --gpu-enable 1`
  - `atlas-alt-bench --mode token-lm-document --token-epochs 1 --repeats 1 --variant muon --gpu-enable 1`

Real CUDA sweep artifact:

- `artifacts/muon_gpu_gate_20260411-134926/epoch_sweep_summary.tsv`

True device-native frontier:

- `token-lm-document`
  - `AdamW`
    - `1 epoch`: `0.08s`, `TestNLL=5.09820`
    - `2 epochs`: `0.12s`, `4.85065`
    - `3 epochs`: `0.16s`, `4.68609`
    - `4 epochs`: `0.20s`, `4.75096`
  - `ATLAS-MUON`
    - `1 epoch`: `0.48s`, `5.07453`
    - `2 epochs`: `0.92s`, `4.85126`
    - `3 epochs`: `1.38s`, `4.78971`
    - `4 epochs`: `1.82s`, `4.61171`
- `token-lm-corpus-large`
  - `AdamW`
    - `1 epoch`: `0.12s`, `TestNLL=6.34269`
    - `2 epochs`: `0.20s`, `6.28071`
    - `3 epochs`: `0.28s`, `6.33483`
    - `4 epochs`: `0.36s`, `6.53199`
  - `ATLAS-MUON`
    - `1 epoch`: `1.14s`, `6.27144`
    - `2 epochs`: `2.22s`, `6.32167`
    - `3 epochs`: `3.31s`, `6.41135`
    - `4 epochs`: `4.44s`, `6.71464`

Interpretation:

- this is now the clean comparison the previous host-assisted MUON path could
  not provide
- device-native MUON still does not beat `AdamW` on explicit GPU
  wall-clock time-to-target
- on `token-lm-document`, it only shows a small loose-target quality advantage
  while remaining roughly `6x` to `9x` slower
- on `token-lm-corpus-large`, it helps only at the loosest 1-epoch point and
  then loses both quality and wall-clock

Verdict:

- `ATLAS-MUON-lite` is now falsified as an `AdamW` replacement under a true
  device-native CUDA comparison
- keep it only as a control if needed
- stop MUON refinement on this branch family

## April 11, 2026: `RACER-lite` implements risk-adjusted sparse promotion, but the first CPU transformer pass is not a replacement win

I implemented `RACER-lite` as a new transformer-side ATLAS branch with:

- exact `AdamW` fallback on all unsensed or demoted blocks
- BiMAP-style two-sided row/column preconditioning as the promoted action
- delayed stable-signal EMA inside each matrix block
- explicit promotion score:
  stable reward minus curvature proxy minus noise penalty minus cost penalty
- explicit CPU-only status on the transformer path for now; GPU RACER is not
  implemented and is rejected rather than silently measured as fallback

Main implementation changes:

- `Backend/Machine Learning/Networks/training_config.h`
- `Backend/Machine Learning/Networks/atlas_optimizer.h`
- `Backend/Machine Learning/Networks/atlas_optimizer.cpp`
- `Backend/Machine Learning/Networks/checkpoint_persistence.cpp`
- `Backend/Machine Learning/Networks/network.cpp`
- `Backend/Machine Learning/Networks/transformer_model_state.inc`
- `Backend/Machine Learning/Networks/sgd_transformer.cpp`
- `unit-tests/Backend/Machine Learning/atlas-alt-bench.cpp`
- `unit-tests/Backend/Machine Learning/atlas-test.cpp`
- `unit-tests/Backend/Machine Learning/atlas-test.h`
- `unit-tests/main.cpp`

Verification:

- `cmake --build /home/robert/dev/glades-ml/build -j4`
- `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
- `./unit-tests/build/glades-unit-tests atlas-controller`
- `./unit-tests/build/glades-unit-tests atlas-racer-core`
- `./unit-tests/build/glades-unit-tests atlas-racer-micro`

Core correctness outcome:

- fallback test: exact Adam-style update when RACER geometry is disabled and
  promotion is blocked
- promotion test: anisotropic matrix block diverges from exact Adam fallback
  and records a finite promotion margin

Transformer micro benchmark:

- `AdamW`: `TrainNLL=2.79707`
- `BiMAP-lite`: `2.82039`
- `RACER-0`: `2.81524`
- `RACER-lite`: `2.81332`

Bounded 1-epoch transformer smoke:

- `token-lm-large`
  - `AdamW`: `TestNLL=4.56885`, `applyMs=0.066`
  - `ATLAS-RACER`: `4.57231`, `applyMs=0.346`
- `token-lm-document`
  - `AdamW`: `5.06561`, `0.56s`, `applyMs=0.871`
  - `ATLAS-RACER`: `5.06481`, `0.56s`, `applyMs=2.488`
- `token-lm-corpus-large`
  - `AdamW`: `6.40179`, `1.17s`, `applyMs=1.841`
  - `ATLAS-RACER`: `6.39173`, `1.39s`, `applyMs=5.123`

Interpretation:

- `RACER-lite` is now a real named branch, not a paper design
- the stable-signal / risk-adjusted promotion logic is internally consistent
- early transformer quality is mixed but real:
  slightly worse on `token-lm-large`, roughly tied/slightly better on
  `token-lm-document`, slightly better on `token-lm-corpus-large`
- cost is still too high even in this narrow CPU pass, so there is no
  replacement case against `AdamW` yet

Verdict:

- keep `RACER-lite` as a valid falsification/control branch
- do not promote it as an `AdamW` replacement
- if this line continues, the next honest gate is explicit GPU
  wall-clock time-to-target, not more CPU-side optimizer-shape tuning

## April 11, 2026: GPU `RACER-lite` is now wired through the real transformer epoch path

I removed the old transformer-side `GPU RACER-lite path not implemented`
rejection and added a minimal GPU residual branch:

- exact batched `AdamW` backbone remains unchanged
- GPU RACER keeps row/column second-moment EMAs resident on device
- GPU RACER keeps `prevMhat` and `stableMhat` resident on device
- promotion is still cheap and scalar-scored:
  predicted stable reward minus risk penalty minus cost penalty
- only the promoted residual difference from `AdamW` is applied

Main CUDA/runtime changes:

- `Backend/Machine Learning/Networks/cuda/gpu_atlas.h`
- `Backend/Machine Learning/Networks/cuda/gpu_atlas.cu`
- `Backend/Machine Learning/Networks/cuda/gpu_transformer_state.h`
- `Backend/Machine Learning/Networks/sgd_transformer.cpp`
- `scripts/run_racer_gpu_gate.sh`

Verification:

- `cmake --build /home/robert/dev/glades-ml/build -j4`
- `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
- `./unit-tests/build/glades-unit-tests atlas-controller`
- `./unit-tests/build/glades-unit-tests atlas-racer-core`
- `./unit-tests/build/glades-unit-tests atlas-racer-micro`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-document --token-epochs 1 --repeats 1 --variant adamw`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-document --token-epochs 1 --repeats 1 --variant racer`
- `bash -n ./scripts/run_racer_gpu_gate.sh`

Sandbox-only GPU-requested smoke:

- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm --token-epochs 1 --repeats 1 --variant racer --gpu-enable 1 --gpu-device 0`
- result: no more RACER-specific unsupported-path error
- this sandbox still reports `No CUDA devices found (err=100, count=0)`,
  so the run falls back and is not a valid GPU benchmark

Interpretation:

- GPU RACER is now benchmarkable on a real CUDA host
- local correctness and CPU behavior still hold after the CUDA wiring
- no promotion/rejection claim should be made until
  `./scripts/run_racer_gpu_gate.sh --gpu-device 0`
  is run on a machine with a visible NVIDIA device

## April 11, 2026: zero-overhead `AdamW-Group` branch is live, but bounded CPU results do not justify promotion

Instead of adding more geometry or controller state, I implemented a
groupwise scalar modulation on top of the exact `AdamW` hot path:

- optimizer type remains `AdamW`
- extra state is only one previous-step RMS scalar per parameter group
- per-group scale is computed from:
  - step stability,
  - moment SNR,
  - update-to-weight ratio
- small groups fall back exactly to `AdamW`
- GPU path keeps batched Adam and only adds one batched group-scale kernel

Main code paths:

- `Backend/Machine Learning/Networks/training_config.h`
- `Backend/Machine Learning/Networks/checkpoint_persistence.cpp`
- `Backend/Machine Learning/Networks/model_persistence.cpp`
- `Backend/Machine Learning/Networks/transformer_model_state.inc`
- `Backend/Machine Learning/Networks/cuda/gpu_transformer_state.h`
- `Backend/Machine Learning/Networks/cuda/gpu_transformer_state.cu`
- `Backend/Machine Learning/Networks/cuda/gpu_kernels.h`
- `Backend/Machine Learning/Networks/cuda/gpu_kernels.cu`
- `Backend/Machine Learning/Networks/sgd_transformer.cpp`
- `unit-tests/Backend/Machine Learning/atlas-alt-bench.cpp`
- `unit-tests/Backend/Machine Learning/atlas-test.cpp`
- `scripts/run_groupadam_gpu_gate.sh`

Verification:

- `cmake --build /home/robert/dev/glades-ml/build -j4`
- `cmake --build /home/robert/dev/glades-ml/unit-tests/build -j4 --target glades-unit-tests`
- `./unit-tests/build/glades-unit-tests atlas-controller`
- `./unit-tests/build/glades-unit-tests atlas-groupadam-micro`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-large --token-epochs 1 --repeats 1 --variant adamw`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-large --token-epochs 1 --repeats 1 --variant adamw-group`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-document --token-epochs 1 --repeats 1 --variant adamw`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-document --token-epochs 1 --repeats 1 --variant adamw-group`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-corpus-large --token-epochs 1 --repeats 1 --variant adamw`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm-corpus-large --token-epochs 1 --repeats 1 --variant adamw-group`

Bounded CPU results:

- micro benchmark
  - `AdamW`: `TrainNLL=2.81271`
  - `AdamW-Group`: `2.80571`

- `token-lm-large`
  - `AdamW`: `TestNLL=4.56885`, `applyMs=0.069`
  - `AdamW-Group`: `4.57867`, `applyMs=0.146`

- `token-lm-document`
  - `AdamW`: `TestNLL=5.06561`, `0.53s`, `applyMs=0.843`
  - `AdamW-Group`: `5.10043`, `0.58s`, `applyMs=1.388`

- `token-lm-corpus-large`
  - `AdamW`: `TestNLL=6.40179`, `1.25s`, `applyMs=2.523`
  - `AdamW-Group`: `6.34366`, `1.28s`, `applyMs=3.506`

Interpretation:

- the branch is real and zero-overhead in structure, not another ATLAS sidecar
- local quality effects are mixed:
  - slightly worse on `token-lm-large`
  - worse on `token-lm-document`
  - slightly better on `token-lm-corpus-large`
- even this cheaper family still pays extra apply cost on CPU
- there is no bounded-case argument for promotion over `AdamW`

Recommendation:

- run the real GPU gate next:
  - `./scripts/run_groupadam_gpu_gate.sh --gpu-device 0`
  - `./scripts/run_groupadam_gpu_gate.sh --gpu-device 0 --acceptance`
- only continue the branch if it beats `AdamW` on GPU `T_epsilon`
  on `token-lm-document` or `token-lm-corpus-large`

## April 11, 2026: real GPU gate rejects `AdamW-Group` as an `AdamW` replacement

GPU artifacts:

- sweep:
  - `artifacts/groupadam_gpu_gate_20260411-152722/epoch_sweep_summary.tsv`
- acceptance:
  - `artifacts/groupadam_gpu_gate_20260411-152722/acceptance_summary.tsv`

5-repeat sweep:

- `token-lm-document`
  - `AdamW` dominates at every measured point
  - `1 epoch`: `AdamW 5.09979 @ 0.08s`, `AdamW-Group 5.12598 @ 0.08s`
  - `2 epochs`: `AdamW 4.85514 @ 0.12s`, `AdamW-Group 4.91335 @ 0.12s`
  - `3 epochs`: `AdamW 4.70822 @ 0.16s`, `AdamW-Group 4.80596 @ 0.16s`
  - `4 epochs`: `AdamW 4.62929 @ 0.20s`, `AdamW-Group 4.75050 @ 0.21s`

- `token-lm-corpus-large`
  - there is a weak loose-target signal, but not a clean frontier win
  - `1 epoch`: `AdamW 6.34327 @ 0.12s`, `AdamW-Group 6.28580 @ 0.13s`
  - `2 epochs`: `AdamW 6.26259 @ 0.20s`, `AdamW-Group 6.25732 @ 0.21s`
  - `3 epochs`: `AdamW 6.30994 @ 0.28s`, `AdamW-Group 6.27191 @ 0.30s`
  - `4 epochs`: `AdamW 6.53131 @ 0.37s`, `AdamW-Group 6.45454 @ 0.38s`

Acceptance pass:

- `token-lm-document`
  - `AdamW`: `TestNLL=4.87007 @ 0.10s`
  - `AdamW-Group`: `4.89741 @ 0.10s`

- `token-lm-corpus-large`
  - `AdamW`: `6.30836 @ 0.19s`
  - `AdamW-Group`: `6.31116 @ 0.19s`

Interpretation:

- the document benchmark is a clean negative
- the corpus-large loose-target hint does not survive the acceptance pass
- `AdamW-Group` is a valid low-overhead control, not a promotion candidate

Verdict:

- keep `AdamW` as the transformer default
- keep `AdamW-Group` only as a control
- stop refining this branch

## April 11, 2026: device-native `BiMAP-lite` changes the BiMAP verdict

After wiring `BiMAP-lite` onto the transformer GPU fast path, I reran the focused
and hard-gate suites with real CUDA instead of the earlier CPU-fallback timing path.

Artifacts:

- focused acceptance:
  - `artifacts/bimap_focus_acceptance_20260411-184621/acceptance_summary.tsv`
  - `artifacts/bimap_focus_acceptance_20260411-184621/ctx_delta_bimap_lite_vs_adamw.md`
- hard-gate sweep:
  - `artifacts/bimap_extended_suite_20260411-185207/epoch_sweep_summary.tsv`
  - `artifacts/bimap_extended_suite_20260411-185207/ctx_delta_bimap_lite_vs_adamw.md`

Focused acceptance (`BiMAP-lite late-head` vs `AdamW`):

- `token-lm-large`
  - `AdamW`: `4.58257 @ 0.03s`
  - `BiMAP-lite`: `4.57542 @ 0.02s`
- `token-lm-context`
  - `AdamW`: `4.27857 @ 0.04s`
  - `BiMAP-lite`: `4.34805 @ 0.05s`
- `token-lm-context-large`
  - `AdamW`: `4.30782 @ 0.07s`
  - `BiMAP-lite`: `4.45575 @ 0.08s`

This overturns the earlier context-family read. On the true GPU path:

- `BiMAP-lite` keeps a small `token-lm-large` niche
- it no longer wins `token-lm-context`
- it clearly loses `token-lm-context-large`

The CTX bucket deltas explain the reversal:

- on the true GPU path, `BiMAP-lite` still helps some `topic` / `recall` / `anchor`
  buckets
- but it now hurts the structural buckets that mattered before:
  - `query`
  - `marker`
  - `sep`

Hard-gate sweep:

- `token-lm-document`
  - `AdamW`
    - `e2`: `4.84733 @ 0.13s`
    - `e3`: `4.70200 @ 0.17s`
    - `e4`: `4.64279 @ 0.21s`
  - `BiMAP-lite`
    - `e2`: `4.92967 @ 0.14s`
    - `e3`: `4.79302 @ 0.20s`
    - `e4`: `4.66989 @ 0.25s`
  - `BiMAP-v2`
    - `e2`: `4.72748 @ 0.68s`
    - `e3`: `4.22233 @ 1.02s`
    - `e4`: `3.59063 @ 1.37s`

- `token-lm-corpus-large`
  - `AdamW`
    - `e1`: `6.34148 @ 0.13s`
    - `e2`: `6.27363 @ 0.21s`
  - `BiMAP-lite`
    - `e1`: `7.23378 @ 0.14s`
    - `e2`: `7.23858 @ 0.23s`
  - `BiMAP-v2`
    - `e1`: `6.39099 @ 0.98s`
    - `e2`: `6.26658 @ 1.95s`

Interpretation:

- device-native `BiMAP-lite` is not a replacement candidate
- `BiMAP-lite` now looks like a narrow `token-lm-large` control, not a broader
  context-family specialist
- `BiMAP-v2` can still buy much lower `TestNLL` by epoch count on
  `token-lm-document`, but its wall-clock cost is far too large to beat `AdamW`
  on `T_epsilon`
- `token-lm-corpus-large` is negative for every BiMAP branch that was tested

Updated verdict:

- keep `AdamW` as the transformer default
- keep `BiMAP-lite` only as a specialized control
- keep `BiMAP-v2` / `BiMAP-v2-0` retired as replacement lines
- stop `BiMAP` refinement as an `AdamW` replacement family

Follow-up implementation on April 12, 2026:

- built a native transformer GPU path for `BiMAP-v2`
- removed the `bimapLowRankEnabled` fast-path rejection in
  `sgd_transformer.cpp`
- the CUDA path now keeps:
  - row / column second moments resident on device
  - low-rank row / column bases resident on device
  - predictive trust and two-sided Woodbury residual application resident on
    device
- local verification passed:
  - `cmake --build build -j4`
  - `cmake --build unit-tests/build -j4 --target glades-unit-tests`
  - `./unit-tests/build/glades-unit-tests atlas-controller`
  - `./unit-tests/build/glades-unit-tests atlas-bimap-micro`
  - bounded route smoke:
    `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm --token-epochs 1 --repeats 1 --variant bimap --atlas-bimap-low-rank 1 --atlas-bimap-scope late-head --gpu-enable 1 --gpu-device 0`
- this sandbox still cannot execute the real CUDA benchmark because it reports
  `No CUDA devices found`, so the actual time-to-target verdict for native
  `BiMAP-v2` still has to come from the GPU host

## April 12, 2026: `ECHO` is the strongest post-AdamW branch, but still not a clean replacement winner

I implemented the `ECHO` family as a geometry-harvesting Adam replacement that
keeps the useful row/column geometry signal but removes the expensive
post-gradient matrix machinery used by `BiMAP`:

- geometry comes from operand-side row / column second moments
- exact Adam-style backbone remains the base update
- the GPU path was then progressively cleaned up:
  - removed per-step host stats downloads
  - removed hot-path `cudaStreamSynchronize` for scalar readback
  - built device-resident row / column metric vectors
  - fused ECHO metric application into the shared batched Adam path
- finally added scope ablations:
  - `all`
  - `large-only`
  - `late-head`
  - `late-head-large`

Key files:

- `Backend/Machine Learning/Networks/atlas_optimizer.cpp`
- `Backend/Machine Learning/Networks/cuda/gpu_atlas.cu`
- `Backend/Machine Learning/Networks/sgd_transformer.cpp`
- `Backend/Machine Learning/Networks/training_config.h`
- `unit-tests/Backend/Machine Learning/atlas-alt-bench.cpp`
- `unit-tests/Backend/Machine Learning/atlas-test.cpp`
- `scripts/run_echo_gpu_gate.sh`
- `scripts/run_echo_geometry_sweep.sh`
- `scripts/run_echo_schedule_sweep.sh`
- `scripts/run_echo_scope_sweep.sh`
- `scripts/run_echo_nsys_profile.sh`
- `scripts/run_echo_scaleup_gate.sh`
- `scripts/run_echo_scaleup_nsys_profile.sh`

Verification across the implementation stages included:

- `cmake --build build -j4`
- `cmake --build unit-tests/build -j4 --target glades-unit-tests`
- `./unit-tests/build/glades-unit-tests atlas-echo-core`
- `./unit-tests/build/glades-unit-tests atlas-echo-micro`
- `./unit-tests/build/glades-unit-tests atlas-alt-bench --mode token-lm --repeats 1 --token-epochs 1 --variant echo`
- GPU-host verification and profiling then came from the saved artifacts below

### Small hard-gate result after the ECHO systems rewrites

Artifacts:

- initial hard gate:
  - `artifacts/echo_gpu_gate_20260412-063748/epoch_sweep_summary.tsv`
  - `artifacts/echo_gpu_gate_20260412-063748/acceptance_summary.tsv`
- schedule sweep:
  - `artifacts/echo_schedule_sweep_20260412-070736/epoch_sweep_summary.tsv`
  - `artifacts/echo_schedule_sweep_20260412-070736/acceptance_summary.tsv`
- post-fusion hard gate:
  - `artifacts/echo_gpu_gate_20260412-091735/epoch_sweep_summary.tsv`
  - `artifacts/echo_gpu_gate_20260412-091735/acceptance_summary.tsv`
- scope sweep:
  - `artifacts/echo_scope_sweep_20260412-103017/epoch_sweep_summary.tsv`
  - `artifacts/echo_scope_sweep_20260412-103017/acceptance_summary.tsv`
- narrowed `late-head` gate:
  - `artifacts/echo_gpu_gate_20260412-104339/epoch_sweep_summary.tsv`
  - `artifacts/echo_gpu_gate_20260412-104339/acceptance_summary.tsv`

Final small-gate read:

- `ECHO 1.0 late-head` is the only serious live ECHO configuration
- `token-lm-document` is genuinely positive for ECHO by quality:
  - acceptance: `AdamW 4.87837 @ 0.10s`, `ECHO 4.87258 @ 0.11s`
  - sweep `e3`: `AdamW 4.69635 @ 0.16s`, `ECHO 4.67906 @ 0.16s`
  - sweep `e4`: `AdamW 4.65802 @ 0.20s`, `ECHO 4.59229 @ 0.21s`
- `token-lm-corpus-large` remains the blocker:
  - acceptance: `AdamW 6.28952 @ 0.18s`, `ECHO 6.27975 @ 0.19s`
  - sweep `e2`: `AdamW 6.27894 @ 0.20s`, `ECHO 6.29306 @ 0.21s`
  - sweep `e3`: `AdamW 6.29129 @ 0.28s`, `ECHO 6.42601 @ 0.29s`

Interpretation:

- ECHO finally extended the `token-lm-document` quality frontier
- but it still did **not** produce a broad `AdamW` replacement win on the two
  hard transformer benchmarks
- `ECHO 1.0 large-only` is useful only as a document-quality control, not as
  the mainline optimizer candidate

### Hybrid borrowing experiments: mathematically interesting, but not a better default

I then tried to import the cheap parts of `GEODE` and `BiMAP` into ECHO without
reintroducing the heavy matrix machinery:

- trust-gated geometry
- bounded predictive blending
- grouped structural factors
- then a continuous non-selector simplex blend around plain ECHO

Artifacts:

- bundled hybrid gate:
  - `artifacts/echo_gpu_gate_20260412-131706/epoch_sweep_summary.tsv`
  - `artifacts/echo_gpu_gate_20260412-131706/acceptance_summary.tsv`
  - `artifacts/echo_gpu_gate_20260412-131954/epoch_sweep_summary.tsv`
  - `artifacts/echo_gpu_gate_20260412-131954/acceptance_summary.tsv`
- ablation gate:
  - `artifacts/echo_ablation_gate_20260412-133100/epoch_sweep_summary.tsv`
  - `artifacts/echo_ablation_gate_20260412-133100/acceptance_summary.tsv`
- continuous simplex blend:
  - `artifacts/echo_gpu_gate_20260412-140958/epoch_sweep_summary.tsv`
  - `artifacts/echo_gpu_gate_20260412-140958/acceptance_summary.tsv`
- late-contracted simplex blend:
  - `artifacts/echo_gpu_gate_20260412-141611/epoch_sweep_summary.tsv`
  - `artifacts/echo_gpu_gate_20260412-141611/acceptance_summary.tsv`
- 20-repeat confidence rerun:
  - `artifacts/echo_gpu_gate_20260412-142826/epoch_sweep_summary.tsv`
  - `artifacts/echo_gpu_gate_20260412-142826/acceptance_summary.tsv`

Read:

- the bundled `trust + predictive + structural` branch was real but not broad:
  - it sometimes helped `token-lm-document`
  - it consistently hurt or destabilized `token-lm-corpus-large`
- the ablation result was clear:
  - plain `echo_base` beat `echo_trust`
  - plain `echo_base` beat `echo_trust_pred`
  - plain `echo_base` beat `echo_trust_struct` as the default branch
- the continuous simplex blend was mathematically cleaner than the naive bundle,
  and the late-contraction variant repaired some of the late-run damage on the
  5-repeat gate
- but the 20-repeat confidence pass falsified the promotion case:
  - `token-lm-document`: `AdamW 4.87030 @ 0.10s`, contracted blend
    `ECHO 4.86820 @ 0.10s`
  - `token-lm-corpus-large`: `AdamW 6.29929 @ 0.18s`, contracted blend
    `ECHO 6.31763 @ 0.19s`

Interpretation:

- the borrowed `GEODE` / `BiMAP` terms are scientifically useful diagnostics
- they did **not** beat plain `ECHO 1.0 late-head` as a checked-in default
- the live ECHO branch therefore returned to plain `late-head` with
  `trust=0`, `pred=0`, `struct=0`

### Nsight result: the ECHO systems problem was mostly solved

Artifacts:

- `artifacts/echo_nsys_20260412-072200`
- `artifacts/echo_nsys_20260412-083501`
- `artifacts/echo_nsys_20260412-091044`
- `artifacts/echo_nsys_20260412-104701`
- `artifacts/echo_nsys_20260412-145837`
- `artifacts/echo_nsys_20260412-163804`
- `artifacts/echo_nsys_20260412-165635`

The progressive CUDA rewrites materially changed the cost model:

- early ECHO was losing mainly on:
  - host stats downloads
  - `cudaStreamSynchronize`
  - extra memcpy / launch overhead
- after ECHO-v2 and ECHO-v3:
  - host round-trips were removed from the hot path
  - metric finalization moved fully on device
  - metric application was fused into the shared batched Adam update

Scoped late-head Nsight (`artifacts/echo_nsys_20260412-104701`) showed that the
remaining overhead on the small benchmark was already close to AdamW:

- `token-lm-document`
  - `cudaLaunchKernel`: `39.8ms / 9984 calls` for `AdamW` vs
    `44.5ms / 10656 calls` for scoped ECHO
  - `cudaMemcpy`: `1.89ms / 306` vs `2.02ms / 334`
  - `cudaStreamSynchronize`: `1.05ms / 609` vs `1.12ms / 679`

So by that point the remaining difference was no longer “bad implementation.”
It was the actual optimizer tradeoff.

Two later systems passes tightened the hot path further:

- lazy allocation of ECHO-only GPU buffers so plain `AdamW` no longer touched
  unused ECHO metadata during the benchmark-path allocation chain
- static per-group base LR / WD metadata on device, with only a scalar
  `lrScale` applied in the fused Adam kernel

The latest valid Nsight comparison (`artifacts/echo_nsys_20260412-165635`
vs `artifacts/echo_nsys_20260412-163804`) showed that this last runtime cut
removed the remaining per-step LR/WD transfer overhead without changing the
launch envelope:

- `token-lm-document`, `ECHO fixed`
  - `cudaMemcpyAsync`: `400 -> 306`
  - `cudaStreamSynchronize`: `722 -> 674`
  - `cudaLaunchKernel`: unchanged at `10080`
- `token-lm-corpus-large`, `ECHO fixed`
  - `cudaMemcpyAsync`: `528 -> 402`
  - `cudaStreamSynchronize`: `928 -> 864`
  - `cudaLaunchKernel`: unchanged at `17792`

So the remaining runtime question is no longer copy/sync churn. The next real
systems levers, if ECHO continues, are launch compression (`CUDA Graphs`) or
deeper kernel fusion.

### Scale-up study: ECHO only becomes interesting when the model is larger

To test whether the remaining overhead would amortize, I froze the candidate set
to:

- `AdamW`
- `ECHO 1.0 late-head`
- `ECHO 1.0 large-only`

and ran a larger-model study with:

- `token-lm-document`:
  - `dModel=96`
  - `dFF=384`
  - `layers=6`
  - `heads=8`
  - `seqLen=128`
  - `trainSeqs=64`
  - `testSeqs=16`
- `token-lm-corpus-large`:
  - `dModel=112`
  - `dFF=448`
  - `layers=6`
  - `heads=8`
  - `seqLen=128`
  - `trainSeqs=80`
  - `testSeqs=20`

Artifacts:

- first scale-up pass:
  - `artifacts/echo_scaleup_gate_20260412-111322/epoch_sweep_summary.tsv`
  - `artifacts/echo_scaleup_gate_20260412-111322/acceptance_summary.tsv`
- 20-repeat confidence rerun:
  - `artifacts/echo_scaleup_gate_20260412-112911/epoch_sweep_summary.tsv`
- scale-up Nsight:
  - `artifacts/echo_scaleup_nsys_20260412-111722`

The first scale-up pass looked promising:

- `token-lm-document` acceptance
  - `AdamW`: `4.95290 @ 0.36s`
  - `ECHO late-head`: `4.93254 @ 0.37s`
- `token-lm-corpus-large` acceptance
  - `AdamW`: `6.68770 @ 0.51s`
  - `ECHO late-head`: `6.64396 @ 0.52s`

and the scale-up Nsight profile showed the overhead had largely amortized:

- `token-lm-document`
  - `cudaLaunchKernel`: `110.6ms / 26240 calls` for `AdamW` vs
    `112.6ms / 27136 calls` for `ECHO late-head`
  - `cudaMemcpyAsync`: `55.5ms / 517` vs `57.0ms / 773`
  - `cudaStreamSynchronize`: `3.29ms / 1057` vs `3.43ms / 1141`
- `token-lm-corpus-large`
  - `cudaLaunchKernel`: `132.3ms / 32800 calls` vs
    `140.4ms / 33920 calls`
  - `cudaMemcpyAsync`: `102.1ms / 645` vs `105.4ms / 965`
  - `cudaStreamSynchronize`: `4.60ms / 1121` vs `5.02ms / 1205`

So the scale-up result made one thing clear:

- ECHO is the first branch where the overhead really does approach AdamW at
  larger model sizes

### 20-repeat confidence pass: not stable enough to promote

The larger-model confidence rerun (`artifacts/echo_scaleup_gate_20260412-112911`)
did not confirm a stable replacement win.

Key points:

- `token-lm-document`, `e1`
  - `AdamW`: `4.86875 @ 0.18s`
  - `ECHO late-head`: `4.86677 @ 0.19s`
- `token-lm-document`, `e2`
  - `AdamW`: `4.99376 @ 0.35s`
  - `ECHO late-head`: `5.01022 @ 0.36s`

- `token-lm-corpus-large`, `e1`
  - `AdamW`: `6.49298 @ 0.26s`
  - `ECHO late-head`: `6.54522 @ 0.26s`
- `token-lm-corpus-large`, `e2`
  - `AdamW`: `6.60706 @ 0.50s`
  - `ECHO late-head`: `6.60237 @ 0.51s`

Interpretation:

- `ECHO late-head` is near AdamW, not clearly better
- the effect is too small and too unstable to justify promotion
- the larger-model benchmarks also show odd test-NLL behavior by epoch for both
  optimizers, so they are not yet strong enough to support a default-optimizer
  decision on their own

### Current ECHO verdict

Updated ranking:

- `ECHO 1.0 late-head` is the strongest post-AdamW research branch in the repo
- `ECHO 1.0 large-only` is only a document-quality probe
- `AdamW` remains the transformer default

Latest gate:

- `artifacts/echo_gpu_gate_20260412-171040/epoch_sweep_summary.tsv`
- `artifacts/echo_gpu_gate_20260412-171040/acceptance_summary.tsv`

Current read:

- the live branch is plain `ECHO late-head`
  - `geom=1.0`
  - `cadence=1`
  - `trust=0`
  - `pred=0`
  - `struct=0`
- acceptance:
  - `token-lm-document`: `AdamW 4.87532 @ 0.10s`, `ECHO 4.87433 @ 0.10s`
  - `token-lm-corpus-large`: `AdamW 6.29052 @ 0.18s`, `ECHO 6.27868 @ 0.19s`
- full sweep:
  - `token-lm-document`: ECHO is ahead at all four epoch points and materially
    ahead late (`e4: 4.57435 @ 0.20s` vs `AdamW 4.66519 @ 0.20s`)
  - `token-lm-corpus-large`: ECHO is only good early (`e1`), while `AdamW`
    remains better from `e2` onward

Interpretation:

- ECHO remains the strongest post-AdamW branch in the repo
- it is now close enough to AdamW that the remaining difference is the real
  optimizer tradeoff, not obvious implementation waste
- it still does **not** justify replacing `AdamW` as the repo-wide default,
  because `token-lm-corpus-large` remains inconsistent
- on document-style transformer tasks, ECHO is a real quality branch worth
  keeping as a research control

Final recommendation:

- stop optimizer-replacement work as a practical repo goal
- keep `ECHO late-head` as the best research branch and documented control
- if ECHO continues at all, run it as a separate retuned larger-model research
  program rather than as “one more tweak” on the checked-in small gates
- for practical training work in this repo, use tuned `AdamW` as the default

## April 13, 2026: MUON-lite CUDA optimization campaign materially changed the practical MUON verdict

The April 11 device-native MUON result was scientifically correct, but it was
not the end of the systems story. On April 13, 2026 I ran an iterative,
Nsight-guided CUDA optimization campaign across:

- `artifacts/muon_nsys_20260413-*`
- `artifacts/muon_gpu_gate_20260413-*`

Starting point on the first April 13 exact-path gate
(`artifacts/muon_gpu_gate_20260413-064809/acceptance_summary.tsv`):

- `token-lm-document`
  - `AdamW`: `4.86803 @ 0.10s`
  - `MUON-lite`: `4.81667 @ 0.92s`
- `token-lm-corpus-large`
  - `AdamW`: `6.28885 @ 0.19s`
  - `MUON-lite`: `6.30417 @ 2.19s`

So MUON still had attractive quality, but the wall-clock cost was
unacceptable.

The optimization campaign then progressively removed the real bottlenecks:

- host-side eligibility/orchestration overhead on ineligible blocks
- per-matrix small-core factorization launches
- per-matrix triangular solves
- per-matrix MUON bookkeeping kernels
- per-matrix Gram formation
- the custom small-core Cholesky bottleneck, by switching to batched cuSOLVER
  `potrf`

The high-ROI systems passes were:

- host-side MUON eligibility gate plus one batched gradient clear
- batched same-shape small-core factorization
- batched `TRSM`
- batched MUON prep/apply housekeeping
- batched Gram GEMM
- cuSOLVER batched Cholesky for the grouped small-core exact solve

By the later exact-path plateau, MUON was no longer a `6x` to `12x` slower
branch. Representative later exact gates:

- `artifacts/muon_gpu_gate_20260413-114641/acceptance_summary.tsv`
  - `token-lm-document`: `AdamW 4.88145 @ 0.10s`, `MUON-lite 4.80562 @ 0.11s`
  - `token-lm-corpus-large`: `AdamW 6.31030 @ 0.18s`, `MUON-lite 6.29686 @ 0.19s`
- `artifacts/muon_gpu_gate_20260413-123247/acceptance_summary.tsv`
  - `token-lm-document`: `AdamW 4.87028 @ 0.10s`, `MUON-lite 4.83742 @ 0.11s`
  - `token-lm-corpus-large`: `AdamW 6.30071 @ 0.18s`, `MUON-lite 6.26461 @ 0.20s`

Late in the same exact-path frontier (`artifacts/muon_gpu_gate_20260413-123247/epoch_sweep_summary.tsv`):

- `token-lm-document`, `e4`
  - `AdamW`: `4.65370 @ 0.20s`
  - `MUON-lite`: `4.57286 @ 0.22s`
- `token-lm-corpus-large`, `e4`
  - `AdamW`: `6.44646 @ 0.36s`
  - `MUON-lite`: `6.68020 @ 0.39s`

Final systems finding:

- the April 13 campaign moved MUON from obviously impractical to genuinely
  competitive on wall-clock
- the remaining fixed tax appears to be mostly library-floor overhead rather
  than obvious repo-side waste
- at the plateau Nsight still showed one extra async-copy group per MUON batch:
  - `token-lm-document`: `343` `cudaMemcpyAsync` calls for `MUON-lite` vs `295`
    for `AdamW`
  - `token-lm-corpus-large`: `455` vs `391`
  - see `artifacts/muon_nsys_20260413-122803/stats/*_cudaapisum.csv`

Updated MUON verdict:

- the old April 11 “stop MUON refinement immediately” verdict was too
  pessimistic about systems headroom
- exact-path `MUON-lite` is now a real research branch on GPU
- it is strongest on `token-lm-document`
- it is still inconsistent on `token-lm-corpus-large`, especially late in the
  sweep
- do **not** replace repo-default `AdamW` with MUON on the basis of the current
  two-benchmark evidence alone

## April 13, 2026: approximate MUON orthogonalization fast path failed and was reverted

After the exact MUON path reached the apparent library floor, I tested a more
algorithmic speed substitute: a grouped small-core approximate orthogonalization
path using a 2-step cubic Newton-Schulz iteration on the normalized step.

Artifacts:

- `artifacts/muon_nsys_20260413-132659`
- `artifacts/muon_gpu_gate_20260413-132936/acceptance_summary.tsv`
- `artifacts/muon_gpu_gate_20260413-132936/epoch_sweep_summary.tsv`

Result:

- `token-lm-document`
  - `AdamW`: `4.87070 @ 0.10s`
  - approximate `MUON-lite`: `4.88507 @ 0.11s`
- `token-lm-corpus-large`
  - `AdamW`: `6.30796 @ 0.18s`
  - approximate `MUON-lite`: `6.43808 @ 0.19s`

Interpretation:

- the approximate branch did **not** remove the fixed MUON copy overhead
- it materially damaged quality, especially on `token-lm-corpus-large`
- this is not a promotion candidate

Decision:

- keep the grouped exact solve as the checked-in MUON path
- treat the approximate orthogonalization branch as a falsified research dead
  end for the current benchmark family

## April 13, 2026: clean same-codebase optimizer reranking updates the cross-branch verdict

To remove the “different day / different codebase” ambiguity, I added
`scripts/run_optimizer_clean_ranking.sh` and ran a unified GPU rerank on the
same checked-in codebase for:

- `AdamW`
- live exact-path `MUON-lite`
- live `ECHO late-head`
- `BiMAP-lite`
- `BiMAP-v2`

Artifacts:

- `artifacts/optimizer_clean_ranking_20260413-141452/acceptance_summary.tsv`
- `artifacts/optimizer_clean_ranking_20260413-141452/epoch_sweep_summary.tsv`
- `artifacts/optimizer_clean_ranking_20260413-141452/acceptance_rank_by_nll.tsv`
- `artifacts/optimizer_clean_ranking_20260413-141452/epoch4_rank_by_nll.tsv`

Acceptance ranking by benchmark:

- `token-lm-document`
  - `MUON-lite`: `4.81613 @ 0.11s`
  - `BiMAP-lite`: `4.86633 @ 0.13s`
  - `AdamW`: `4.86955 @ 0.10s`
  - `ECHO`: `4.87305 @ 0.10s`
  - `BiMAP-v2`: `4.88624 @ 0.15s`
- `token-lm-corpus-large`
  - `ECHO`: `6.29683 @ 0.19s`
  - `AdamW`: `6.30220 @ 0.18s`
  - `BiMAP-lite`: `6.31839 @ 0.22s`
  - `MUON-lite`: `6.32128 @ 0.20s`
  - `BiMAP-v2`: `6.52011 @ 0.26s`

Epoch-4 ranking by benchmark:

- `token-lm-document`
  - `MUON-lite`: `4.51939 @ 0.22s`
  - `ECHO`: `4.56536 @ 0.20s`
  - `BiMAP-lite`: `4.59567 @ 0.25s`
  - `AdamW`: `4.64589 @ 0.20s`
  - `BiMAP-v2`: `4.65474 @ 0.30s`
- `token-lm-corpus-large`
  - `AdamW`: `6.50439 @ 0.36s`
  - `BiMAP-lite`: `6.56006 @ 0.44s`
  - `ECHO`: `6.58692 @ 0.37s`
  - `BiMAP-v2`: `6.63715 @ 0.52s`
  - `MUON-lite`: `6.64363 @ 0.39s`

Interpretation:

- the old “`ECHO` is the strongest post-AdamW branch” statement is no longer a
  clean repo-wide summary after the April 13 rerank
- the current ranking is benchmark-specific:
  - `token-lm-document`: `MUON-lite` is now the strongest checked-in branch
  - `token-lm-corpus-large`: `ECHO` is best at acceptance, but `AdamW` is best
    from `e2` onward and at `e4`
- `BiMAP-lite` is competitive but not leading
- `BiMAP-v2` remains a raw-NLL / high-cost branch, not a practical optimizer

Superseding practical verdict:

- for practical default training in this repo, keep tuned `AdamW`
- keep `MUON-lite` as the strongest document-style research branch
- keep `ECHO late-head` as a strong control, especially for corpus-acceptance
  comparisons
- keep `BiMAP-lite` as a secondary control
- keep `BiMAP-v2` retired as a practical replacement line

## April 13, 2026: `MATRA` is now fully implemented, parity-validated, and ready for the same-codebase GPU rerank

I completed the first full `MATRA` implementation pass as a real optimizer line
in the repo, not just a framework note:

- CPU `MATRA` lives in `atlas_optimizer.cpp` / `atlas_optimizer.h`
- GPU `MATRA` lives in `gpu_atlas.cu` / `gpu_atlas.h`
- transformer dispatch, checkpoint/config plumbing, and state allocation are
  wired through the normal transformer training path
- unit coverage includes:
  - `atlas-matra-core`
  - `atlas-matra-parity`

Important implementation finding:

- the hard GPU parity failure was **not** ultimately a scratch-buffer issue and
  not primarily an orthogonalizer-quality issue
- the real step-2 drift came from the GPU decomposition of the `MATRA` update:
  GPU was applying plain Adam first and then only the geometry / orthogonal
  residuals, while CPU `MATRA` forms its Adam prior **after** the bounded
  predictive transport
- the final fix was to keep the pre-applied plain Adam backbone step separate on
  GPU and add the missing correction from backbone Adam to the predictive
  `MATRA` Adam prior before the geometry and orthogonal residual terms

Current implementation verdict:

- `MATRA` now has a checked-in CPU implementation
- `MATRA` now has a checked-in GPU implementation
- GPU parity now passes on real CUDA hardware
- `scripts/run_optimizer_clean_ranking.sh` was extended to include `MATRA` in
  the same-codebase benchmark gate, using the checked-in defaults:
  - `matraGeometryScale=1.0`
  - `matraOrthogonalScale=0.5`
  - `matraPredictiveScale=0.05`
  - `matraTrustRadius=0.50`
  - `matraMetricCadence=1`
  - `matraMaxAspect=1.50`
  - `matraMinDim=8`
  - `matraDamping=0.01`

What is **not** claimed yet:

- no repo verdict should rank `MATRA` against `AdamW` / `MUON-lite` / `ECHO` /
  `BiMAP-lite` until the updated clean GPU rerank is actually run
- as of this entry, `MATRA` is implementation-complete and test-valid, but not
  yet performance-ranked in the research log

Decision:

- keep `MATRA` as a live research branch
- do the next comparison with the same-codebase clean ranking gate, not with
  ad-hoc single-branch runs
- if `MATRA` fails that matched-codebase gate, treat the framework as a
  research falsification rather than spending another loop on systems polish

## April 13, 2026: clean same-codebase GPU rerank falsifies `MATRA` as a practical optimizer default

I ran the updated clean ranking gate with `MATRA` included on the same checked-in
codebase:

- `artifacts/optimizer_clean_ranking_20260413-165114/acceptance_summary.tsv`
- `artifacts/optimizer_clean_ranking_20260413-165114/epoch_sweep_summary.tsv`
- `artifacts/optimizer_clean_ranking_20260413-165114/acceptance_rank_by_nll.tsv`
- `artifacts/optimizer_clean_ranking_20260413-165114/epoch4_rank_by_nll.tsv`

Acceptance result:

- `token-lm-document`
  - `MUON-lite`: `4.81375 @ 0.11s`
  - `BiMAP-lite`: `4.86184 @ 0.13s`
  - `AdamW`: `4.87394 @ 0.10s`
  - `ECHO`: `4.87874 @ 0.10s`
  - `BiMAP-v2`: `4.89054 @ 0.15s`
  - `MATRA`: `4.89311 @ 0.69s`
- `token-lm-corpus-large`
  - `MATRA`: `6.26683 @ 1.92s`
  - `AdamW`: `6.29584 @ 0.18s`
  - `ECHO`: `6.31077 @ 0.19s`
  - `MUON-lite`: `6.31584 @ 0.20s`
  - `BiMAP-lite`: `6.31940 @ 0.22s`
  - `BiMAP-v2`: `6.53076 @ 0.26s`

Epoch-4 result:

- `token-lm-document`
  - `MUON-lite`: `4.53920 @ 0.22s`
  - `ECHO`: `4.59427 @ 0.20s`
  - `BiMAP-lite`: `4.62747 @ 0.25s`
  - `AdamW`: `4.63484 @ 0.20s`
  - `BiMAP-v2`: `4.70897 @ 0.31s`
  - `MATRA`: `4.80251 @ 1.42s`
- `token-lm-corpus-large`
  - `AdamW`: `6.47953 @ 0.36s`
  - `MATRA`: `6.51687 @ 3.76s`
  - `MUON-lite`: `6.57568 @ 0.39s`
  - `ECHO`: `6.57945 @ 0.37s`
  - `BiMAP-lite`: `6.61691 @ 0.44s`
  - `BiMAP-v2`: `6.64413 @ 0.51s`

Interpretation:

- `MATRA` does show a real acceptance-quality signal on `token-lm-corpus-large`
  and is the best branch there by raw acceptance `TestNLL`
- that gain is bought at roughly a `10x` wall-clock penalty versus `AdamW`, so
  it is not a practical optimizer recommendation
- on `token-lm-document`, `MATRA` is dominated on both speed and quality by the
  existing live branches
- `MATRA` therefore failed the practical matched-codebase gate that was defined
  at the end of the implementation/parity phase

Updated `MATRA` verdict:

- keep `MATRA` as a scientifically interesting corpus-large quality probe
- do **not** promote `MATRA` as a repo-default optimizer
- do **not** replace `MUON-lite` / `ECHO` / `BiMAP-lite` controls with `MATRA`
  for normal optimizer studies
- if `MATRA` continues, the next step must be an explicit systems or algorithmic
  cost-reduction program with a hard matched-wall-clock gate, not further
  conceptual expansion

## April 13, 2026: the first `MATRA` systems campaign recovered most of the gap, then hit a micro-optimization plateau

I ran a focused CUDA optimization loop on the checked-in `MATRA` branch using:

- `artifacts/matra_nsys_20260413-171249/`
- `artifacts/matra_gpu_gate_20260413-171528/`
- through
- `artifacts/matra_nsys_20260413-184311/`
- `artifacts/matra_gpu_gate_20260413-184559/`

Main systems findings:

- the original `MATRA` GPU implementation was dominated by exact orthogonal-core
  housekeeping and per-group launch overhead
- moving the orthogonal path fully on-device and batching the small exact solves
  cut acceptance wall clock from the original rerank result of:
  - `token-lm-document`: `0.69s`
  - `token-lm-corpus-large`: `1.92s`
  down to the stable post-optimization band of about:
  - `token-lm-document`: `0.12s`
  - `token-lm-corpus-large`: `0.21s`
- after that large gain, further micro-passes only shaved small amounts of
  kernel / memcpy / synchronize overhead without moving the actual acceptance
  frontier in a meaningful way

Current trustworthy frontier from the post-optimization branch:

- `artifacts/matra_gpu_gate_20260413-181005/acceptance_summary.tsv`
  - `token-lm-document`: `MATRA 4.87063 @ 0.12s`
  - `token-lm-corpus-large`: `MATRA 6.25067 @ 0.21s`
- `artifacts/matra_gpu_gate_20260413-182150/acceptance_summary.tsv`
  - `token-lm-document`: `MATRA 4.87351 @ 0.12s`
  - `token-lm-corpus-large`: `MATRA 6.23707 @ 0.21s`

Latest plateau confirmation:

- `artifacts/matra_gpu_gate_20260413-184559/acceptance_summary.tsv`
  - `token-lm-document`: `MATRA 4.87413 @ 0.12s` vs `AdamW 4.87464 @ 0.10s`
  - `token-lm-corpus-large`: `MATRA 6.27135 @ 0.21s` vs `AdamW 6.30253 @ 0.18s`
- `artifacts/matra_nsys_20260413-184311/stats/`
  showed that the packed batch-recording pass reduced some blocking host
  bookkeeping on `token-lm-corpus-large`
  - `cudaMemcpy`: `478 -> 454` calls
  - `cudaStreamSynchronize`: `1312 -> 1238` calls

Final structural systems result:

- `artifacts/matra_gpu_gate_20260413-190506/acceptance_summary.tsv`
  - `token-lm-document`: `MATRA 4.86999 @ 0.12s` vs `AdamW 4.86686 @ 0.10s`
  - `token-lm-corpus-large`: `MATRA 6.25424 @ 0.21s` vs `AdamW 6.29996 @ 0.18s`
- `artifacts/matra_nsys_20260413-190142/stats/`
  showed that the persistent MATRA batch-descriptor cache removed the fixed
  async-copy surplus almost completely:
  - document `cudaMemcpyAsync`: `343 -> 296` calls for `MATRA`, versus `AdamW 295`
  - corpus-large `cudaMemcpyAsync`: `455 -> 392` calls for `MATRA`, versus `AdamW 391`
- despite that, wall clock still stayed in the same practical band:
  - document remained `0.12s`
  - corpus-large remained `0.21s`
- this means the remaining gap is no longer explained by MATRA batch descriptor
  uploads; it is now dominated by MATRA’s extra launch count and residual kernel
  stack versus the simpler `AdamW` path

Latest measured confirmation before the next structural pass:

- `artifacts/matra_gpu_gate_20260413-191843/acceptance_summary.tsv`
  - `token-lm-document`: `MATRA 4.87570 @ 0.12s` vs `AdamW 4.88271 @ 0.10s`
  - `token-lm-corpus-large`: `MATRA 6.26440 @ 0.20s` vs `AdamW 6.30597 @ 0.18s`
- `artifacts/matra_nsys_20260413-191606/stats/`
  confirmed that the fixed metadata-copy tax was effectively gone:
  - document `cudaMemcpyAsync`: `296` calls for `MATRA`, versus `AdamW 295`
  - corpus-large `cudaMemcpyAsync`: `392` calls for `MATRA`, versus `AdamW 391`
  while launch count was still materially higher:
  - document `cudaLaunchKernel`: `MATRA 12000`
  - corpus-large `cudaLaunchKernel`: `MATRA 20545`
- interpretation:
  - the remaining cost is no longer MATRA descriptor upload or obvious host-side
    metadata plumbing
  - the remaining gap is mostly the MATRA launch stack plus solver / library
    overhead relative to `AdamW`

Checked-in next structural pass, pending rerun:

- a dedicated geometry-only MATRA fused batch path now skips the separate
  prepare/apply residual sequence for non-orth groups
- the MATRA batch path also folds trust-budget finalization into the stats
  finalization kernel to remove another control launch
- those changes were built and `atlas-matra-core` passed locally, but they still
  require a fresh GPU rerun before any new speed claim is valid

Follow-up measurement of that fused geometry-only path:

- `artifacts/matra_gpu_gate_20260413-192828/acceptance_summary.tsv`
  - `token-lm-document`: `MATRA 4.87217 @ 0.11s` vs `AdamW 4.87222 @ 0.10s`
  - `token-lm-corpus-large`: `MATRA 6.26947 @ 0.20s` vs `AdamW 6.30198 @ 0.18s`
- `artifacts/matra_nsys_20260413-192506/stats/`
  - document `cudaLaunchKernel`: `12000 -> 11808`
  - corpus-large `cudaLaunchKernel`: `20545 -> 20289`
  - `cudaMemcpyAsync` remained effectively at parity with `AdamW`
- interpretation:
  - the geometry-only fused path did remove a small amount of fixed launch tax
  - it modestly improved the document acceptance frontier
  - the broader `MATRA` verdict did not change; the remaining gap is still
    mostly launch / solver overhead rather than metadata-copy traffic

Falsification of the over-fused geometry-only kernel:

- `artifacts/matra_gpu_gate_20260413-194142/acceptance_summary.tsv`
  - `token-lm-document`: `MATRA 4.86972 @ 0.14s` vs `AdamW 4.87594 @ 0.10s`
  - `token-lm-corpus-large`: `MATRA 6.24540 @ 0.28s` vs `AdamW 6.30433 @ 0.18s`
- `artifacts/matra_nsys_20260413-193709/stats/`
  showed why that branch failed:
  - the compact geometry-only kernel became the dominant MATRA hotspot
  - document: about `18.6 ms` total across `96 + 48` instances
  - corpus-large: about `44.4 ms` total across `128 + 64` instances
  - `cudaLaunchKernel` counts did go down further
    - document: `11808 -> 11088`
    - corpus-large: `20289 -> 19329`
  - but `cudaStreamSynchronize` exploded
    - document: `0.94 ms -> 18.47 ms`
    - corpus-large: `1.24 ms -> 44.83 ms`
- interpretation:
  - collapsing the entire geometry-only MATRA path into one heavy kernel was the
    wrong tradeoff
  - the kernel reduced launch count but replaced it with a much larger per-group
    execution cost dominated by reductions / atomics / low-occupancy work
  - that branch was therefore rolled back locally, restoring the earlier
    geometry-only path as the last known-good baseline

Rollback confirmation:

- `artifacts/matra_gpu_gate_20260413-195800/acceptance_summary.tsv`
  - `token-lm-document`: `MATRA 4.88542 @ 0.11s` vs `AdamW 4.87553 @ 0.10s`
  - `token-lm-corpus-large`: `MATRA 6.26701 @ 0.20s` vs `AdamW 6.29907 @ 0.18s`
- `artifacts/matra_nsys_20260413-195523/stats/`
  confirmed that the rollback returned MATRA to the earlier good systems band:
  - document `cudaLaunchKernel`: back to `11808`
  - corpus-large `cudaLaunchKernel`: back to `20289`
  - document `cudaStreamSynchronize`: back down to about `1.23 ms`
  - corpus-large `cudaStreamSynchronize`: back down to about `2.17 ms`
- interpretation:
  - the regression was isolated to the compact geometry-only kernel
  - the rollback restores the previous MATRA systems frontier
  - no further evidence currently suggests another high-ROI repo-side
    micro-optimization remains

Interpretation:

- the optimization campaign was successful in making `MATRA` operationally
  competitive enough to study under matched wall clock
- `MATRA` still preserves a real quality signal on `token-lm-corpus-large`
- the remaining gap now looks like a mixture of launch-count tax and library /
  solver behavior, not an obvious repo-side metadata or host-copy bug
- I would stop `MATRA` micro-optimization at this point

Updated action rule:

- only continue `MATRA` work if the next step is structural, for example:
  - a dedicated geometry-only fast path
  - a tighter fusion with the batched Adam backbone update
  - or a new algorithmic simplification of the orthogonal branch

## April 14, 2026: clean same-codebase rerank on the restored `MATRA` branch updates the cross-optimizer picture again

After the April 13 `MATRA` systems work and the rollback of the failed
over-fused geometry-only kernel, I reran the clean same-codebase ranking on the
restored stable branch using:

- `artifacts/optimizer_clean_ranking_20260414-041121/acceptance_summary.tsv`
- `artifacts/optimizer_clean_ranking_20260414-041121/epoch_sweep_summary.tsv`
- `artifacts/optimizer_clean_ranking_20260414-041121/acceptance_rank_by_nll.tsv`
- `artifacts/optimizer_clean_ranking_20260414-041121/epoch4_rank_by_nll.tsv`

Acceptance ranking by benchmark:

- `token-lm-document`
  - `MUON-lite`: `4.81535 @ 0.11s`
  - `BiMAP-lite`: `4.86350 @ 0.13s`
  - `MATRA`: `4.87195 @ 0.12s`
  - `ECHO`: `4.87300 @ 0.10s`
  - `AdamW`: `4.87636 @ 0.10s`
  - `BiMAP-v2`: `4.89802 @ 0.15s`
- `token-lm-corpus-large`
  - `MATRA`: `6.25805 @ 0.20s`
  - `AdamW`: `6.30330 @ 0.18s`
  - `ECHO`: `6.30588 @ 0.19s`
  - `MUON-lite`: `6.31724 @ 0.20s`
  - `BiMAP-lite`: `6.33291 @ 0.22s`
  - `BiMAP-v2`: `6.52259 @ 0.26s`

Epoch-4 ranking by benchmark:

- `token-lm-document`
  - `MUON-lite`: `4.59419 @ 0.22s`
  - `ECHO`: `4.61979 @ 0.20s`
  - `BiMAP-v2`: `4.64928 @ 0.30s`
  - `AdamW`: `4.65121 @ 0.20s`
  - `BiMAP-lite`: `4.65899 @ 0.26s`
  - `MATRA`: `4.73031 @ 0.23s`
- `token-lm-corpus-large`
  - `MATRA`: `6.47817 @ 0.40s`
  - `AdamW`: `6.49354 @ 0.36s`
  - `ECHO`: `6.55310 @ 0.37s`
  - `BiMAP-lite`: `6.59928 @ 0.44s`
  - `BiMAP-v2`: `6.64940 @ 0.52s`
  - `MUON-lite`: `6.68172 @ 0.39s`

What changed relative to the April 13 clean rerank:

- `MATRA` is no longer a high-cost outlier branch; the systems work moved it
  from `0.69s / 1.92s` acceptance into the same practical wall-clock band as
  the live branches
- on `token-lm-corpus-large`, `MATRA` is now the best checked-in branch at both
  acceptance and `e4`
- on `token-lm-document`, `MATRA` improved enough to be competitive at
  acceptance, but it is still the weakest branch by `e4`

Updated practical verdict after the April 14 rerank:

- `AdamW` remains the safest general default for broad large-LLM training in
  this repo because it is still the most balanced cross-benchmark optimizer and
  gives nearly the same corpus-large quality as `MATRA` at lower wall clock
- `MUON-lite` remains the strongest document-style branch
- `MATRA` is now the strongest corpus-large research branch and the best current
  option when that benchmark family is the actual target
- `ECHO` remains a useful near-AdamW control, but it is no longer the strongest
  corpus-side branch after the stable `MATRA` systems work
- `BiMAP-lite` remains competitive but secondary, and `BiMAP-v2` remains a
  high-cost specialist rather than a practical default

## April 14, 2026: `MATRA` exact-orth cadence sweep shows the best corpus-large tradeoff is every 2 steps

I then tested the first structural performance recommendation for `MATRA`: keep
predictive + geometry active every step, but only run the exact orthogonal
branch every `N` steps.

Implementation notes:

- added `matraOrthCadence` to the ATLAS config, checkpoint persistence, CLI
  surface, and CPU / GPU MATRA paths
- semantics:
  - step `0` still allows the orth branch
  - after that, exact orth is only eligible when `step % matraOrthCadence == 0`
  - on skipped steps, `MATRA` keeps the same predictive + geometry update and
    only suppresses the exact orth solve
- local verification:
  - `atlas-matra-core` passed
  - `atlas-matra-parity` remains the GPU-host verification check

Sweep runs, in execution order:

- `artifacts/matra_gpu_gate_20260414-050503/`
  - `matraOrthCadence = 1`
- `artifacts/matra_gpu_gate_20260414-050618/`
  - `matraOrthCadence = 2`
- `artifacts/matra_gpu_gate_20260414-050719/`
  - `matraOrthCadence = 4`

Acceptance comparison:

- `token-lm-document`
  - cadence `1`: `MATRA 4.85875 @ 0.12s`
  - cadence `2`: `MATRA 4.87179 @ 0.11s`
  - cadence `4`: `MATRA 4.87572 @ 0.11s`
  - `AdamW` reference in the same runs stayed around `4.87 @ 0.10s`
- `token-lm-corpus-large`
  - cadence `1`: `MATRA 6.26262 @ 0.20s`
  - cadence `2`: `MATRA 6.24490 @ 0.20s`
  - cadence `4`: `MATRA 6.26325 @ 0.20s`
  - `AdamW` reference in the same runs stayed around `6.30 @ 0.18s`

Epoch-4 comparison:

- `token-lm-document`
  - cadence `1`: `MATRA 4.78005 @ 0.23s`
  - cadence `2`: `MATRA 4.81819 @ 0.22s`
  - cadence `4`: `MATRA 4.79618 @ 0.22s`
- `token-lm-corpus-large`
  - cadence `1`: `MATRA 6.45442 @ 0.40s`
  - cadence `2`: `MATRA 6.45465 @ 0.39s`
  - cadence `4`: `MATRA 6.46282 @ 0.39s`

Interpretation:

- the exact orth branch is still useful often enough that sparsifying it
  aggressively to every `4` steps hurts both document and corpus quality
- `matraOrthCadence = 2` is the best large-LLM / corpus-large tradeoff:
  - same practical acceptance speed as cadence `1`
  - clearly better corpus-large acceptance NLL
  - essentially tied corpus-large `e4`
  - slightly better `e4` wall clock
- `matraOrthCadence = 1` remains preferable only if the target is
  document-style acceptance quality specifically

Updated practical rule:

- for large-LLM-style / corpus-large MATRA runs, prefer:
  - `matraOrthCadence = 2`
- for document-style MATRA experiments where acceptance quality matters more
  than the last `0.01s`, keep:
  - `matraOrthCadence = 1`
- do not use:
  - `matraOrthCadence = 4`
    because it gives up quality without buying meaningful additional speed

This does not overturn the broader optimizer recommendation:

- `AdamW` remains the safest overall default
- `MATRA` becomes a slightly better corpus-large branch when configured with
  `matraOrthCadence = 2`
- the next clean rerank should use `MATRA` with that cadence setting if the
  goal is a corpus-large-centered comparison

## April 14, 2026: clean rerank with `MATRA matraOrthCadence=2` keeps MATRA best at corpus acceptance but no longer clearly best at `e4`

I reran the clean same-codebase ranking with the new preferred corpus-side
`MATRA` setting:

- `artifacts/optimizer_clean_ranking_20260414-051611/acceptance_summary.tsv`
- `artifacts/optimizer_clean_ranking_20260414-051611/epoch_sweep_summary.tsv`
- `artifacts/optimizer_clean_ranking_20260414-051611/acceptance_rank_by_nll.tsv`
- `artifacts/optimizer_clean_ranking_20260414-051611/epoch4_rank_by_nll.tsv`

Acceptance ranking by benchmark:

- `token-lm-document`
  - `MUON-lite`: `4.81874 @ 0.11s`
  - `MATRA`: `4.86823 @ 0.11s`
  - `BiMAP-lite`: `4.86996 @ 0.13s`
  - `AdamW`: `4.87212 @ 0.10s`
  - `ECHO`: `4.88157 @ 0.10s`
  - `BiMAP-v2`: `4.89044 @ 0.15s`
- `token-lm-corpus-large`
  - `MATRA`: `6.26854 @ 0.20s`
  - `MUON-lite`: `6.29803 @ 0.19s`
  - `AdamW`: `6.30592 @ 0.18s`
  - `ECHO`: `6.30698 @ 0.18s`
  - `BiMAP-lite`: `6.33056 @ 0.22s`
  - `BiMAP-v2`: `6.52154 @ 0.26s`

Epoch-4 ranking by benchmark:

- `token-lm-document`
  - `MUON-lite`: `4.54773 @ 0.22s`
  - `ECHO`: `4.58823 @ 0.20s`
  - `BiMAP-lite`: `4.62711 @ 0.26s`
  - `BiMAP-v2`: `4.62801 @ 0.30s`
  - `AdamW`: `4.66382 @ 0.20s`
  - `MATRA`: `4.74992 @ 0.22s`
- `token-lm-corpus-large`
  - `AdamW`: `6.47005 @ 0.36s`
  - `MATRA`: `6.47006 @ 0.39s`
  - `ECHO`: `6.59047 @ 0.37s`
  - `BiMAP-lite`: `6.59653 @ 0.44s`
  - `BiMAP-v2`: `6.61974 @ 0.52s`
  - `MUON-lite`: `6.64958 @ 0.39s`

Interpretation:

- the cadence-2 rerank preserves the main corpus-side result:
  - `MATRA` is still the best acceptance branch on `token-lm-corpus-large`
- but it weakens the earlier stronger claim that MATRA is also clearly the best
  late-horizon corpus branch:
  - at `e4`, `AdamW` is effectively tied and wins by a hair
  - `6.47005` vs `6.47006` is not a meaningful practical separation
- on `token-lm-document`, the new cadence setting improves MATRA acceptance
  enough to place it second, but it remains weak by `e4`

Updated practical verdict:

- `AdamW` remains the safest general default for large-LLM training in this repo
- `MUON-lite` remains the strongest document-style branch
- `MATRA` with `matraOrthCadence = 2` is now the best checked-in corpus-large
  acceptance branch
- `MATRA` is no longer clearly better than `AdamW` by late corpus-large
  training horizon; they should be treated as effectively tied there
- `ECHO` remains a useful control but not the frontier branch
---

*Document version: 1.31*
*Framework: ATLAS (Adaptive Temporally-Predictive Learning in Active Subspaces)*
*Date: 2026-04-14*
