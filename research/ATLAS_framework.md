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

---

*Document version: 1.3*
*Framework: ATLAS (Adaptive Temporally-Predictive Learning in Active Subspaces)*
*Date: 2026-04-09*
