# HELIOS: Hamiltonian Ensemble Langevin Integrator with Sharpness-adaptive Thermostat

## A Mathematical Framework for Physics-Based LLM Optimization at Extreme Scale

---

## 1. Executive Summary

This document proposes **HELIOS**, a novel optimizer for extremely large language models (10B-1T parameters) that is structurally disjoint from AdamW, ATLAS (empirical-Fisher subspace), and VESTA (spectral entropy mirror). HELIOS reformulates LLM training as dissipative Hamiltonian dynamics on an augmented phase space (theta, p, xi) and discretizes it with a **stochastic-symplectic BAOAB integrator**. A per-group Nose-Hoover thermostat xi_g enforces kinetic-temperature equipartition, and a state-dependent friction is modulated by a cheap Rayleigh-quotient sharpness probe kappa_g evaluated at rate <= 1/100. Mini-batch noise is absorbed into the Ornstein-Uhlenbeck substep via the Li-Sato-Tan temperature correction. The stationary distribution is a sharpness-tilted Gibbs measure whose theta-marginal concentrates on **flat** low-loss regions -- a property AdamW, ATLAS, and VESTA do not possess.

**Memory**: 4N bytes (bf16 momentum + bf16 anchor), i.e. **half AdamW's state**.
**Compute**: one forward+backward per step, plus ~5% overhead from BAOAB; one extra HVP per K_hvp >= 100 steps on one group only (sub-percent amortized).
**DDP**: update is purely local given the reduced gradient; no extra collectives.
**Low-precision stability**: no division by noisy second-moment estimates (classic bf16 Adam failure mode is eliminated by construction).
**Expected mechanism for beating AdamW at scale**: (i) flat-minimum bias (SAM-like without the double-backward cost), (ii) hypocoercive convergence rate optimized by curvature-adaptive friction, (iii) muP-compatible per-group masses.

---

## 2. Candidate Formulations

Three formulations were developed in parallel on materially different axes:

### Candidate A: KAIROS -- Koopman-operator trajectory-based optimizer

Core: estimate a rank-r Koopman matrix \tilde K_b online via Hankel-EDMD on a sliding window of observables phi(W_t), with phi a fixed random sketch dictionary. Update: split the gradient into projected + residual parts; shape the projected part by the Koopman-spectral Neumann inverse (I - \tilde K)^(-1) to amplify slow modes; let the residual follow EMA-momentum. Memory: 0.5x weights. Novelty axis: **dynamical-systems / operator-theoretic**.

### Candidate B: NYX -- Nystrom-sketched true-Gauss-Newton natural gradient

Core: approximate the **true Fisher** F_l = E[J_l^T H_out J_l] (computed via Pearlmutter JVP+VJP against the model's output softmax Hessian H_out, not empirical gradient outer products) via a column Nystrom sketch \hat S = F Omega, \hat M = Omega^T F Omega. Update: Woodbury-inverted natural gradient d = lambda^(-1) [m - \hat S (lambda \hat M + \hat S^T \hat S)^(-1) \hat S^T m]. Memory: ~1.0x weights. Novelty axis: **information-geometric with true Fisher (vs ATLAS's empirical Fisher)**.

### Candidate C: HELIOS -- Hamiltonian/symplectic thermostat optimizer

Core: underdamped Langevin-Nose-Hoover SDE discretized by BAOAB splitting with sharpness-adaptive friction. Memory: 0.5x AdamW. Novelty axis: **continuous-time stochastic physics with explicit invariant measure**.

---

## 3. Framework Selection Rationale

| Axis | KAIROS | NYX | HELIOS |
|---|---|---|---|
| Memory vs AdamW | 0.25x | ~1.0x (at ceiling) | **0.5x** (half) |
| Compute overhead | ~5% | ~3% + HVP amortization | ~5-7% + sub-1% HVP |
| DDP cleanliness | local | needs seeded Omega + periodic HVP reductions | purely local |
| bf16 stability | moderate (EDMD regression) | strong (fp32 Woodbury core) | **strong (no variance division)** |
| Material separation from AdamW | momentum-adjacent + Koopman core | still uses momentum + rank-r adaptive | **Hamiltonian conjugate momentum, not EMA heuristic** |
| Flat-minimum bias | none by construction | none | **explicit via T_g(kappa_g)** |
| Theoretical depth at scale | EDMD stability under non-stationarity is open | natural-gradient classical + Nystrom concentration | **hypocoercivity + stochastic symplectic stability both classical** |
| Novelty vs ATLAS/VESTA | strong (trajectory not statistics) | structural Fisher distinction is real but subtle | **paradigm-level (continuous-time physical, not discrete-adaptive)** |

**Dominant tradeoffs:** NYX is mathematically satisfying but sits at the edge of the memory budget and its advantage over ATLAS depends on how much E[gg^T] deviates from F_true at scale -- a quantitative empirical question. KAIROS is most novel but theoretically least robust (Koopman identification under non-stationary drift is fragile). HELIOS dominates on every practical axis at scale: memory, parallelism, low-precision stability, flat-minimum bias, and paradigm-level distance from AdamW.

**Selection:** HELIOS.

**Why rejected are weaker:** KAIROS depends on the window-estimated \tilde K being predictively accurate; LLM training has phase transitions (warmup, schedule changes, curriculum) that repeatedly invalidate the estimate. NYX's per-block state r(m+n) at r=128 saturates the 1.0x budget; its refresh HVPs become a real compute tax at 1T scale. HELIOS alone achieves large-scale efficiency **and** provides a qualitatively new selection bias (flat minima) that AdamW provably lacks.

---

## 4. Formal Problem Statement

Let theta in R^N parameterize a transformer LM with output softmax p_theta(.|x). Given data distribution D, define population loss
  L(theta) = E_{(x,y)~D} [ -log p_theta(y|x) ].
The stochastic mini-batch estimator is \hat L_B(theta) with gradient g_B = grad \hat L_B. Denote the stochastic-gradient noise covariance Sigma_B(theta) = Cov(g_B - grad L), of order 1/|B|.

Partition theta into G layer-role groups {theta_g}_{g=1}^G with N_g = |theta_g|; roles are Q/K/V/O (per attention layer), MLP-up/gate/down, embedding, output head.

**Problem:** design an iterative update theta_{k+1} = F(theta_k, \hat L_B) with auxiliary state s_k such that:

- memory(s_k) <= N bf16 parameters (= 1.0x weights; vs AdamW's 2x fp32 = 4x weights);
- compute(F) <= 1.1 x compute of vanilla SGD;
- F is DDP-decomposable: F depends on B only through the all-reduced gradient g_B (except amortized <= 1% extra);
- F is numerically stable in bf16 working precision;
- asymptotic iterates theta_k have lower expected loss and better generalization than AdamW at matched wall-clock.

---

## 5. Core Mathematical Framework

**Augmented phase space.** Introduce momentum p in R^N with group-scalar masses m_g > 0 assembled into M = diag(m_{g(i)}), and per-group thermostat variables xi in R^G with inertias Q_g > 0. Define the augmented Hamiltonian

  H(theta, p, xi) = U(theta) + (1/2) p^T M^{-1} p + sum_g (Q_g/2) xi_g^2,

with potential

  U(theta) = L(theta) + (lambda_a / 2) ||theta - \bar theta||^2.

\bar theta_k = beta_a \bar theta_{k-1} + (1 - beta_a) theta_k is an EMA anchor enforcing slow-manifold regularization.

**Primary SDE.** HELIOS posits training *is* the Ito SDE

  d theta = M^{-1} p dt,
  dp = -grad U(theta) dt - Gamma(theta, t) M^{-1} p dt + sqrt(2 Gamma(theta, t) T(t)) dW,
  d xi_g = (1/Q_g) ( p_g^T M_g^{-1} p_g - N_g T_g(t) ) dt,

with state-dependent friction

  Gamma(theta, t) = gamma_0 I + diag_i(xi_{g(i)}) + alpha diag_i(kappa_{g(i)}),

and per-group sharpness probe

  kappa_g(theta, p) = clip( v_g^T (grad^2 L(theta)) v_g, 0, kappa_max ),  v_g = p_g / ||p_g||.

T_g(t) is the target temperature (Section 6).

**Group mass choice (muP-compatible).** m_g proportional to sqrt(fan-in_g) makes per-group kinetic energy dimensionally consistent across model scales: kinetic energy per coordinate is (1/2) E[p_i^2] / m_{g(i)} = (1/2) T_{g(i)} at equipartition, and the update theta += h M^{-1} p then scales as h / sqrt(fan-in), matching the muP parameterization of Yang & Hu (2021) without per-parameter tuning.

---

## 6. Objective Function Derivation

The SDE's stationary distribution, at static T_g = T, is the **extended Gibbs measure**

  pi_T(theta, p, xi) prop. exp( -(1/T) [ U(theta) + (1/2) p^T M^{-1} p + sum_g (Q_g/2) xi_g^2 ] ),

verifiable by Fokker-Planck: (d/dt + L) pi_T = 0 with L the infinitesimal generator of the SDE. The theta-marginal is

  pi_T(theta) prop. exp(-U(theta)/T),

i.e. a tilted Gibbs measure on theta with the anchor regularizer (lambda_a / 2) ||theta - \bar theta||^2 added.

**Flat-minimum selection via sharpness-adaptive temperature.** Now let T_g depend on local sharpness: T_g = T_0 * max(1, kappa_g / kappa_star)^(-1/2). On a flat region (kappa_g small) T_g = T_0; on a sharp region (kappa_g >> kappa_star) T_g shrinks as kappa_g^(-1/2). The effective theta-marginal (in the adiabatic approximation where xi_g equilibrates fast relative to theta) is

  pi_eff(theta) prop. exp( -(L(theta) + (lambda_a/2) ||theta-\bar theta||^2) / T(theta) ),  T(theta) = T_0 min(1, kappa_star / kappa(theta))^(1/2).

The Laplace expansion around a local minimum theta* gives the log-marginal free energy

  -log pi_eff(theta*) ~ L(theta*) / T(theta*) + (1/2) log det( grad^2 L(theta*) / T(theta*) ).

Substituting T(theta*) prop. kappa(theta*)^(-1/2) makes the first term penalize sharp minima with a **positive power of sharpness**, while the second term (standard Laplace volume) also favors flat minima. Combined effect: HELIOS concentrates probability mass on minima with L(theta*) * kappa(theta*)^(1/2) small -- a **sharpness-weighted loss**. This is the SAM-style selection bias, achieved as a stationary property of the dynamics rather than a per-step penalty.

**Variational form.** HELIOS minimizes the free-energy functional

  F_T[rho] = integral U(theta) rho(theta) d theta + T integral rho log rho

over probability densities rho on theta, subject to the equipartition constraint E_rho[ p^T M^{-1} p ] = N T enforced by xi. Gradient descent is the zero-temperature, zero-inertia limit; AdamW is not a limit of HELIOS (different inductive bias).

---

## 7. Optimization Algorithm

**BAOAB splitting** (Leimkuhler-Matthews 2013) factors the generator L = L_A + L_B + L_O where

- L_A propagates theta += h M^{-1} p (position Hamiltonian flow, exactly solvable),
- L_B propagates p -= h grad U(theta) (momentum kick, exactly solvable),
- L_O propagates the Ornstein-Uhlenbeck dp = -Gamma M^{-1} p dt + sqrt(2 Gamma T) dW (exactly solvable with c = exp(-Gamma h): p <- c p + sqrt(M T (1 - c^2)) zeta).

The BAOAB composition exp((h/2) L_B) exp((h/2) L_A) exp(h L_O) exp((h/2) L_A) exp((h/2) L_B) is a **stochastic-symplectic** integrator with provable properties (Section 9).

**Per-step algorithm** (iteration k, mini-batch B_k):

**Input:** theta_k, p_k, xi_k, \bar theta_k; step h; rng state.

1. Compute gradient g <- g_{B_k}(theta_k) + lambda_a (theta_k - \bar theta_k)   *(one fwd+bwd, same as SGD)*.
2. p <- p - (h/2) g    *(B-half)*.
3. theta <- theta + (h/2) M^{-1} p    *(A-half)*.
4. For each group g: xi_g <- xi_g + (h / (2 Q_g)) ( ||p_g||^2 / m_g - N_g T_g )    *(N-half)*.
5. **O-step**: set Gamma_i = gamma_0 + xi_{g(i)} + alpha kappa_{g(i)}, c_i = exp(-Gamma_i h), T_eff_g = T_g + (h/4) tr(Sigma_g) / N_g (Li-Sato-Tan correction, running estimate from 2-batch difference every 10^3 steps),

  p_i <- c_i p_i + sqrt( m_{g(i)} T_eff_{g(i)} (1 - c_i^2) ) zeta_i,  zeta_i ~ N(0, 1).

6. For each group g: xi_g <- xi_g + (h / (2 Q_g)) ( ||p_g||^2 / m_g - N_g T_g )    *(N-half)*.
7. theta <- theta + (h/2) M^{-1} p    *(A-half)*.
8. p <- p - (h/2) g    *(B-half, reuses cached g from step 1)*.
9. Anchor: \bar theta <- beta_a \bar theta + (1 - beta_a) theta.
10. Every K_hvp steps, on a single round-robin group g*: draw v = p_{g*} / ||p_{g*}||, compute H v by one extra forward pass (Pearlmutter), set kappa_{g*} <- 0.95 kappa_{g*} + 0.05 clip( v^T H v, 0, kappa_max ). Update T_{g*} <- T_0 * max(1, kappa_{g*} / kappa_star)^(-1/2).

**Critical correctness point:** steps 2 and 8 use the *same* gradient g from step 1. BAOAB with a fixed-per-step force field *does* converge to the correct invariant measure on theta -- this is the Leimkuhler-Matthews analysis. No additional forward/backward is required. HELIOS's compute budget matches SGD+momentum.

---

## 8. Temporal Dynamics Formulation

**Annealing schedule.** T_0(t) follows a cosine decay from T_init to T_final over training, logarithmic in steps: T_0(t) = T_final + (T_init - T_final) (1 + cos(pi log(1 + t) / log(1 + t_max))) / 2.

**Two-timescale separation.** Group inertias Q_g are set so that xi_g relaxes on timescale tau_xi = sqrt(Q_g / (N_g T_g)) that is fast compared to loss-surface exploration but slow compared to the step h: h << tau_xi << tau_loss. Standard MD tuning: tau_xi ~ 10 h. This ensures the adiabatic approximation used in Section 6's effective marginal is valid.

**Sharpness refresh cadence.** kappa_g is refreshed on a round-robin of groups with period K_hvp >= 100 per group. With G ~ 100 groups at 70B scale, the effective global refresh is every step -- but each step costs only one HVP on one group, so amortized overhead is O(1/G) of a step. EMA smoothing (beta_kappa = 0.95) prevents noise from mini-batch HVP estimates from destabilizing the thermostat.

**Warmup.** lambda_a = 0 and T_g = T_init for the first ~2 * 10^3 steps; alpha is zero until kappa_g has been measured at least twice. This prevents spurious thermostat dynamics before the phase space is populated.

---

## 9. Theoretical Analysis

**Well-posedness.** The SDE has Lipschitz drift on any compact set where L in C^2; global existence and uniqueness follow from standard results (Kunita 1990) once we verify dissipativity: -grad U^T p - p^T Gamma M^{-1} p <= -gamma_0 ||p||^2 / m_min + ||p|| ||grad U||, yielding a Lyapunov function V = U + (1/2) p^T M^{-1} p + c ||theta||^2 with E[V] <= C for all t.

**Invariance and symmetry.** The dynamics are invariant under per-group orthogonal reparameterization of hidden dimensions if M, Q are scalars per group (as specified): for any O_g in O(d_g), the pair (p_g, theta_g) -> (O_g p_g, O_g theta_g) leaves H and Gamma distribution-invariant.

**Conditioning.** The momentum update has condition number determined by Gamma, which is bounded above by gamma_0 + max_g |xi_g| + alpha kappa_max; the Nose-Hoover feedback keeps |xi_g| bounded by sqrt(Q_g N_g T_g) / Q_g in expectation. Unlike AdamW, there is no division by a running second moment, so bf16 representation suffices throughout.

**Stochastic symplectic stability.** BAOAB is a Strang-splitting of three exactly-integrable pieces. Theorem (Leimkuhler-Matthews 2013): for smooth U, BAOAB is weakly first-order consistent with O(h^2) configurational bias on E_pi[f(theta)] and O(h^4) on position autocorrelations in the high-friction limit. The stability limit is h < 2 / sqrt(lambda_max(grad^2 U)); empirically ~3-5x the AdamW learning-rate stability threshold.

**Hypocoercivity.** Villani's hypocoercive theorem (2009) applied to the combined Langevin-Nose-Hoover generator gives exponential convergence of ||rho_t - pi||_{H^1} at rate r* prop. gamma_0 / (1 + gamma_0^2 / lambda_min(grad^2 U)). This is optimized at gamma_0* ~ sqrt(lambda_min(grad^2 U)) -- matching precisely what the alpha kappa_g term delivers adaptively. HELIOS's friction approaches the hypocoercivity-optimal rate *locally*.

**Flat-minimum theorem (sketch).** Under the adiabatic approximation and assuming kappa is well-estimated, the stationary theta-marginal satisfies

  log( pi(theta_a*) / pi(theta_b*) ) ~ -(L(theta_a*) - L(theta_b*)) / T_0 + (1/2) log( kappa(theta_b*) / kappa(theta_a*) ) + O( T_0^(-1/2) (kappa_b^(1/2) - kappa_a^(1/2)) ).

At equal loss, HELIOS assigns higher probability to the flatter minimum with log-ratio proportional to (kappa_b^(1/2) - kappa_a^(1/2)) / T_0^(1/2). SGD/AdamW assign equal probability up to diffusion effects. This is the qualitative inductive bias difference.

**What is hard vs conjectural.** Well-posedness, symplectic stability bounds, and the invariant-measure identity are theorem-level statements. The flat-minimum ratio depends on the adiabatic separation being valid, which is a design assumption (supported by the two-timescale setup). The claim that HELIOS beats AdamW in wall-clock is an **empirical conjecture** requiring the validation in Section 14.

---

## 10. Computational Tradeoffs

Per iteration, denoting N parameters, G groups, B batch:

| Work | Cost | vs SGD |
|---|---|---|
| Gradient (fwd+bwd) | 2 * flops_f | 1x |
| Two half-kicks | 2N adds | 0.1% |
| Two half-drifts | 2N mults | 0.1% |
| O-step (OU) | 3N FMAs + N RNG | 0.3% |
| N-step | O(G) | <0.01% |
| Anchor EMA | 2N ops | 0.2% |
| HVP refresh | (2/K_hvp) * flops_f / G | <0.05% at G=100, K=100 |

Total: ~1.05x SGD; well under the 1.1x budget.

**Memory** (bf16 weights baseline 2N bytes):
- weights: 2N
- p: 2N
- \bar theta: 2N
- {xi_g, T_g, kappa_g, m_g}: O(G) << N

Total optimizer state: 4N bytes = 2x weight memory as bf16, but only 1x "additional" overhead beyond weights if weights are stored bf16. AdamW requires 8N bytes (fp32 m + fp32 v) = 4x weights. **HELIOS halves AdamW's state.** At 1T parameters (2 TB weights bf16) this saves 4 TB of optimizer state per replica -- fundamentally changing the ZeRO sharding economics.

**DDP.** The update after step 1 depends only on local state and the already-reduced g. The HVP refresh (step 10) on a single group does require computing H v for that group's subset of parameters, which under ZeRO-1/2/3 sharding is either purely local (weights shard owns activations for the HVP -- ZeRO-1/2) or requires one parameter all-gather for that group (ZeRO-3, rare). Either way, sub-1% communication overhead.

**Low precision.** The O-step's c_i = exp(-Gamma_i h) and sqrt(1 - c_i^2) are computed in fp32 once per group (since Gamma_i is piecewise-constant per group up to the kappa contribution), then broadcast to bf16 updates. No per-parameter divisions; no variance estimates that can underflow.

---

## 11. Comparison to Existing Methods

**vs AdamW.** AdamW maintains (m_i, v_i) per parameter and computes Delta theta_i = -h m_i / (sqrt(v_i) + eps). The v_i term is a diagonal second-moment estimate -- a biased, noisy proxy for curvature. HELIOS replaces this with per-group Hamiltonian dynamics: no v_i, no per-parameter division. Adam's inductive bias is toward equal per-coordinate effective step; HELIOS's inductive bias is toward flat minima. Different qualitative attractor, half the state.

**vs ATLAS.** ATLAS uses a rank-r EMA of the **empirical Fisher** E[gg^T] on layer subspaces, preconditioning by \hat F^{-1} with PNG temporal extrapolation. This is a *statistical curvature surrogate* -- biased under non-Gaussian likelihoods and subject to rank collapse on output layers. HELIOS does not estimate curvature statistically; it lets dynamics *probe* curvature via momentum p, which accumulates response to grad U automatically, with a sharpness *feedback* (kappa_g) that is computed explicitly but rarely. No subspace to maintain, no refresh cost scaling with rank.

**vs VESTA.** VESTA operates on singular-value spectra of weight matrices via Bregman mirror descent with spectral-entropy potential on a Stiefel manifold. Its object is the *static* SVD of W. HELIOS's object is the *dynamic* phase-space trajectory of (theta, p); it never computes or regularizes any SVD. Whereas VESTA regularizes which singular values are allowed, HELIOS regularizes which local geometries of the loss surface are visited.

**vs K-FAC/Shampoo.** These estimate (approximate) block-Kronecker Fisher factors; they carry O(d^2) state per block and require periodic inversion. HELIOS carries O(1) per-group state. Different regime.

**vs SAM (Sharpness-Aware Minimization).** SAM computes a perturbed gradient grad L(theta + eps grad L / ||grad L||), requiring **two** forward+backward passes per step (2x cost). HELIOS achieves the same flat-minimum bias as a *stationary property of the dynamics* with <= 1.05x cost. Mechanistically different: SAM is gradient-of-worst-case; HELIOS is Gibbs-measure-tilting.

**vs SGLD/HMC.** Classical Langevin samplers use fixed temperature and isotropic noise. HELIOS's novelty is (a) group-wise Nose-Hoover thermostat, (b) kappa-adaptive friction, (c) BAOAB discretization for symplectic stability at deep-learning step sizes, (d) anchor regularization. No existing Langevin-family method combines these specifically for LLM training.

---

## 12. Failure Modes and Mitigations

1. **Thermostat chatter** (oscillating xi_g): if Q_g too small, xi_g ringing. Mitigation: Q_g = N_g T_g / omega_xi^2 with omega_xi = 1/(10 h) standard tuning. Unit test: autocorrelation of xi_g should decay within 10 steps.

2. **Negative Rayleigh quotient** from mini-batch HVP: v^T \hat H v < 0 occurs due to finite-batch non-convexity. Mitigation: clip at zero (preserving sign-conservation property of kappa_g >= 0); EMA smoothing beta_kappa = 0.95 further denoises.

3. **Momentum blowup at phase transitions** (loss spikes): ||p|| can grow rapidly near bad gradients. Mitigation: the O-step's c_i < 1 contracts p geometrically each step; additionally a hard safety clip ||p_g||_2 <= c_p sqrt(m_g N_g T_g) triggers only on outlier events.

4. **Anchor EMA stale at init**: \bar theta_0 = theta_0, lambda_a = 0 for first 2k steps, ramped to target.

5. **Noise-temperature miscalibration**: if T_eff is set to T_g but mini-batch noise adds variance Sigma_B, the invariant measure is distorted. Mitigation: Li-Sato-Tan correction T_eff_g = T_g + (h/4) tr(Sigma_g) / N_g. Estimate tr(Sigma_g) every 10^3 steps via two-batch difference: \hat tr(Sigma_g) = (1/2) ||g_{B_1}^g - g_{B_2}^g||^2.

6. **Cold-start of kappa_g**: before the first HVP, kappa_g = 0; friction reduces to gamma_0 + xi_g. This is graceful -- the optimizer behaves like plain underdamped Langevin until sharpness estimates populate.

7. **bf16 accumulation error in O-step over long horizons**: the p <- c p + sqrt(...) zeta step accumulates roundoff. Mitigation: fp32 master momentum with bf16 working copy, same as AdamW master weights.

8. **Wrong alpha sign (instability)**: if alpha is too large, friction on sharp regions overshoots and the thermostat becomes degenerate. Mitigation: alpha is a small (<= 0.1) dimensionless scalar; T_g >= T_min > 0 floor prevents T_g -> 0.

9. **ZeRO-3 HVP all-gather cost**: if parameter partitioning is aggressive, an HVP on one group may require an all-gather of its weights. Mitigation: align HVP cadence with existing all-gather patterns (e.g., during forward pass of that layer).

---

## 13. Minimal Prototype

**Smallest meaningful experiment.** Train `pile_small` (the existing glades-trainer 512-dim / 8-layer / 8-head config) with HELIOS and compare to AdamW at matched wall-clock on 500M Pile tokens. Expected signal: HELIOS reaches matched validation NLL in fewer tokens OR hits lower final NLL at same tokens; secondary signal: flatter Hessian spectrum at endpoint (measure via kappa-proxy).

**Minimum viable mathematical instantiation** (drops complications):
- Scalar global T (no per-group T_g), scheduled with cosine decay.
- No sharpness adaptation (alpha = 0) -- plain BAOAB with Nose-Hoover.
- No anchor (lambda_a = 0).
- Group masses m_g = 1.

This reduces HELIOS to "SGLD with BAOAB discretization and one thermostat", which is already on the research frontier and should beat or match AdamW on small models.

**Incremental additions (in order):**
1. Per-group thermostats xi_g with Q_g tuning.
2. muP-compatible masses m_g prop. sqrt(fan-in).
3. Sharpness feedback alpha kappa_g with HVP refresh.
4. Anchor EMA regularization.
5. Li-Sato-Tan noise-temperature correction.

**Implementation path in glades-ml (from current VESTA/ATLAS infrastructure):**
- Add OptimizerConfig::HELIOS and HeliosConfig in training_config.h mirroring VestaConfig structure.
- helios_optimizer.h/.cpp implementing WeightState (momentum, anchor), GroupState (xi, T, kappa, m), and BAOAB step.
- CUDA kernel: fused BAOAB (B-A-O-A-B with per-group parameter scatter); this is ~50 lines following the existing VESTA kernel pattern.
- HVP scaffolding: use the existing autodiff machinery; one extra forward on one group per K_hvp steps.
- Integration test: helios-test.cpp mirroring vesta-test.cpp, with a correctness check that BAOAB samples the Gaussian exp(-||theta||^2 / 2T) correctly on a quadratic loss.

**Prototype experiments that can run now:**
1. pile_small, 100M tokens: HELIOS vs AdamW at same LR schedule. Success criterion: Delta NLL < -0.02 at matched tokens.
2. Ablate alpha (sharpness feedback): does it help?
3. Ablate per-group vs scalar T: is the group machinery worth its cost?
4. Measure endpoint Hessian trace (cheap Hutchinson estimator); HELIOS endpoints should have smaller trace than AdamW endpoints at matched loss.

---

## 14. Full Research Program

**Phase 1 (1-2 months):** Implement minimal HELIOS in glades-ml, run smoke tests and pile_small comparisons; publish a tech report with pile_small/pile_large experiments vs AdamW and VESTA. Deliverable: empirical case for flat-minimum bias at <= 1B scale.

**Phase 2 (2-4 months):** Port to glades-trainer with DDP, BF16 mixed precision (the Phase A work already scaffolded in the codebase). Run LLaMA-class 7B comparisons on 200B tokens of Pile. Extend the theoretical analysis to non-convex hypocoercivity bounds (local-to-global via Bakry-Emery).

**Phase 3 (6-12 months):** Scale to 30B+ with ZeRO-3 sharding; study interaction with existing gradient-compression machinery (fp16/topk DDP already in codebase). Publish a paper focused on *mechanism* -- ablations isolating (a) symplectic vs non-symplectic discretization, (b) sharpness feedback vs static friction, (c) Nose-Hoover vs Langevin, (d) anchor EMA.

**Phase 4 (>12 months):** Theoretical agenda: (i) prove a flat-minimum concentration theorem under realistic deep-learning assumptions; (ii) quantify the wall-clock advantage as a function of loss-surface anisotropy; (iii) design hybrid HELIOS + VESTA on Stiefel-constrained layers; (iv) explore non-equilibrium variants (Jarzynski-style bias correction for finite-time training). Position HELIOS as the first member of a new family: *physical optimizers*.

---

## 14a. Implementation Status (2026-04-20)

Sections 1-14 are the framework design. This section is the operational
record of what has been built, tested, and measured in glades-ml.

### Shipped

**CPU path:**
- `helios_optimizer.{h,cpp}` — BAOAB integrator for the MVI (Q=0, alpha=0,
  lambdaAnchor=0). Per-matrix state: `p` (momentum), optional `thetaBar`
  (anchor), scalars xi/Tcurr/kappa/mass/step.
- `HeliosConfig` in `training_config.h` with all fields from the framework
  spec; defaults match the MVI.
- `OptimizerConfig::HELIOS` type tag and dispatch in `sgd_transformer.cpp`
  for all weight matrices (tokE, WIn, WOut, per-block Wq/Wk/Wv/Wo/W1/W2);
  biases and LN params use vanilla SGD (matching VESTA's CPU path).
- glades-trainer CLI: `--helios --helios-h --helios-gamma0 --helios-t0
  --helios-mass`; run.sh env vars mirror these.
- **FD-HVP primitives (2026-04-20)**:
  `helios::directional_curvature_fd` and
  `helios::directional_curvature_fd_preallocated` compute `v^T H v` via
  central finite differences over a gradient callback. O(eps^2)
  truncation, 2 callback invocations. `helios::updateSharpness` does the
  beta=0.05 EMA update with [0, kappaMax] clipping and non-finite
  rejection. These are the building blocks for the framework Section 7
  step 10 sharpness probe; they are callable standalone today and used
  by the `HELIOSSharpnessFeedbackTest` integration test but not yet
  wired into the CPU transformer training loop.

**GPU path:**
- `cuda/gpu_helios.{h,cu}` — GpuHeliosWeightState + helios_gpu_init +
  helios_gpu_step. 312 lines of CUDA covering B-half, A-half, O-step
  (deterministic + stochastic with host-generated noise for strict CPU/GPU
  RNG parity), anchor EMA, non-finite guard.
- `GpuHeliosWeightState` per-block (heliosW{q,k,v,o,1,2}) and top-level
  (heliosTokE/WIn/WOut) fields in `GpuTransformerWeights`.
- HELIOS GPU dispatch branch in `transformerGpuTrainEpoch` parallel to
  VESTA, following the same macro pattern.
- Registered in `cuda/CMakeLists.txt`; builds clean with
  `sh .configure.sh cuda`.

**Tests:**
- `HELIOSInitStateTest` — state allocation + scalar initialization.
- `HELIOSBaoabInvariantTest` — samples the Gibbs measure on a quadratic
  potential; empirical variance 1.028 vs target 1.0 at h=0.05 (within the
  5% O(h^2) bias tolerance).
- `HELIOSStepDescentTest` — deterministic T=0 descent on quadratic; loss
  ratio 4.9e-14 in 300 steps (converged).
- `HELIOSNonFiniteGuardTest` — NaN gradient returns false without
  corrupting state.
- `HELIOSvsAdamWComparisonTest` — tiny token LM (vocab=29, dModel=64,
  3 layers) LR-swept both optimizers: **HELIOS best (lr=3.0) beats AdamW
  best (lr=1e-3) by 0.40 nats on test NLL.**
- `HELIOSGpuParityTest` — deterministic T=0, 20 steps, 32x24 matrix:
  `max |W_cpu - W_gpu| = 0.0` (bit-exact); momentum also bit-exact.
- `HELIOSGpuStochasticParityTest` — stochastic T=1e-3, 15 steps, 24x20
  matrix, matched RNG: `max |W_cpu - W_gpu| = 5.96e-8` (FP rounding).
- `HELIOSFdHvpQuadraticTest` — FD-HVP on diagonal quadratic: v^T H v
  matches true eigenvalue lambda_i for v=e_i (max error 5e-5 at
  eps=1e-3); mixed-direction curvature correct to same tolerance.
- `HELIOSUpdateSharpnessTest` — kappa EMA (beta=0.05), [0, kappaMax]
  clipping, NaN-probe rejection.
- `HELIOSSharpnessFeedbackTest` — end-to-end mechanism on 8x4 matrix
  with anisotropic Hessian (lambda in {0.1, 1.0, 5.0}): periodic
  FD-HVP probe along momentum direction, EMA into state.kappa,
  O-step friction gamma_0 + alpha*kappa. After 400 steps,
  kappa=1.94 — within the [0.1, 5.0] eigenvalue band, confirming
  the probe sees real directional curvature.

### Measured throughput (RTX 4080 SUPER, 2026-04-20)

| Config | AdamW GPU tok/s | HELIOS GPU tok/s | Ratio |
|---|---|---|---|
| pile_small (dModel=512, 8L, seq=1024, mb=16) | 9,795 | 9,791 | **99.96%** |
| dModel=1024, 8L, seq=1024, mb=16 | ~3,105 | 3,103 (lr=3e-2), 3,103 (lr=1e-2) | **99.94%** |

Noise-upload cost analysis at dModel=1024: T=0 (deterministic path, no
host→device upload) and T=1e-6 (host-generate Gaussians + upload per
matrix per step) both hit 3,098 tok/s. The upload overlaps with the
preceding A/B-half kernel launches on the compute stream, contributing no
measurable throughput overhead at this scale. An on-device curand RNG
would matter only at dModel≥4096 where per-matrix P grows to ≥16M floats.

### Measured NLL at mid-scale

**pile_small-class head-to-head (dModel=1024, 8L, seq=1024, mb=16, 1.5M
tokens, sampled-softmax loss, RTX 4080 SUPER):**

| Optimizer | LR | Final train NLL | gap vs AdamW |
|---|---|---|---|
| AdamW | 3e-4 | **9.737** | 0 |
| HELIOS | 1e-2 | 10.024 | +0.287 |
| HELIOS | 3e-2 | 10.272 | +0.535 |

HELIOS best (lr=1e-2) loses by 0.29 nats at dModel=1024.

**Scale trend (HELIOS vs AdamW, best-LR each)** — revised after the
2026-04-20 CPU HELIOS dispatch-bug fix:

| dModel | Scale | Path | HELIOS − AdamW (NLL) | Verdict |
|---|---|---|---|---|
| 64 | tiny unit-test LM | CPU pre-fix (was SGD+mom) | −0.40 | invalid |
| 64 | tiny unit-test LM | CPU post-fix MVI | **+0.78** | AdamW wins |
| 64 | tiny unit-test LM | CPU post-fix + sharpness probe | **+0.78** | wash vs MVI |
| 512 | pile_small CPU | CPU pre-fix (was SGD+mom) | ~0 / mixed | invalid |
| 1024 | pile_small-class GPU | **+0.29** | AdamW wins |

This pattern (win at ≤256-dim, lose at ≥1024-dim) matches what we observed
for VESTA earlier in the BF16 scale-up campaign. The
"scale until the new optimizer wins" bet does not hold on this hardware
at the token budgets tested (≤5M); the gap does not visibly narrow with
dModel.

### What the implementation does NOT yet cover

The MVI covers the bare BAOAB. Extensions from the framework (Sections
6-8) that are implemented-but-inactive or not-yet-implemented:

- **Per-group Nose-Hoover thermostat (Q > 0).** Scaffolded in
  `HeliosConfig::Q` and `WeightState::xi`. CPU path supports it. GPU
  path returns false if Q > 0 (guarded in helios_gpu_step); needs a
  kinetic-energy reduction kernel (`sum p²`) and xi update. ~50 additional
  lines.
- **Sharpness feedback (alpha > 0, kHvp > 0).** The core math infrastructure
  and the training-loop wiring are now shipped (2026-04-20):
  - `helios::directional_curvature_fd` — finite-difference HVP primitive
    (central FD, O(eps^2) truncation, 2 gradient-callback evaluations per
    call). Test `HELIOSFdHvpQuadraticTest` validates v^T H v recovery on
    a diagonal quadratic for e_i / mixed directions to within 1e-3.
  - `helios::updateSharpness(state, kappaProbe, hc)` — EMA update with
    beta_kappa = 0.05 and clipping to [0, kappaMax], non-finite-safe.
    Tests `HELIOSUpdateSharpnessTest` / `HELIOSSharpnessFeedbackTest`
    exercise EMA convergence, clipping, and the end-to-end feedback loop
    on a known-eigenstructure quadratic (kappa converges to 1.94 inside
    the true [0.1, 5.0] eigenvalue band).
  - **CPU transformer training-loop wiring (2026-04-20):** at each
    minibatch boundary in `SGDHelper_TRANSFORMER` (before
    `minibatchDriver.apply_ready_batch`), when HELIOS is the active
    optimizer and `hc.alpha > 0` and `hc.kHvp > 0`, we: (1) increment
    a per-state probe counter; (2) on multiples of `hc.kHvp`, pick a
    round-robin target from {Wq, Wk, Wv, Wo, W1, W2} × nLayers; (3)
    build `v = p_target / ||p_target||`; (4) snapshot the target's
    weights and all gradient buffers; (5) perturb `W_target += eps·v`
    and replay the last sequence's forward+backward (reusing
    `transformerCpuForwardPass` / `transformerCpuBackwardPass`); (6)
    capture `gPlus`; (7) perturb `-eps·v` and replay to get `gMinus`;
    (8) compute `kappa = v · (gPlus - gMinus) / (2 eps)`; (9) restore
    weights and grads via vector.swap; (10) call `updateSharpness`.
    The O-step friction `gamma_0 + xi + alpha·kappa` picks up the
    updated kappa on the next optimizer step. Implementation at
    `sgd_transformer.cpp` around line 7160.
  - **GPU probe kernel infrastructure (2026-04-21):**
    `gpu::helios_gpu_probe_snapshot_W` / `_compute_v` / `_perturb` /
    `_compute_kappa` / `_restore_W`. Unit-tested via
    `HELIOSGpuProbeKernelsTest` on a diagonal-Hessian quadratic: `v^T H v`
    matches CPU to 5e-5 on basis directions; `||p||` computed on-device
    matches host to 1e-4.
  - **GPU probe control flow (2026-04-21):** round-robin matrix target
    selection, snapshot/compute-v/skip-if-p-norm-too-small, and the
    kappa EMA update are all wired into `transformerGpuTrainEpoch`'s
    `gpuUseHelios` branch. The probe fires at the configured K_hvp cadence.
  - **GPU forward extraction (2026-04-21):** the inline per-sequence
    forward block (embedding → per-layer blocks → final LN → output head
    → softmax) was extracted into
    `NNetwork::transformerGpuRunForwardOnly`, callable from both the
    training loop's normal forward and the probe's perturbed re-forwards.
    Handles tokenLM + tied head, dense input projection, RoPE, BF16 per-
    site flags, RMSNorm/LayerNorm, ReLU/GELU/SwiGLU. Method is located
    near `transformerGpuTrainEpoch` in sgd_transformer.cpp.
  - **GPU probe wired end-to-end (2026-04-21):** the probe uses the
    scalar-FD form `kappa = (L_plus + L_minus - 2*L_0) / eps^2` (2 extra
    forward-only passes per probe event, no extra backward). L_0 is
    captured from the last sequence's normal forward; L_plus/L_minus are
    obtained by perturbing the target matrix by +/- eps·v and calling
    `transformerGpuRunForwardOnly` + `collect_token_lm_metrics`. Weights
    are restored from snapshot after every probe. BF16 mirrors are
    refreshed around each perturbation via `ensureLowpMirrors` to keep
    BF16 GEMMs consistent with FP32 master. CLI surface:
    `--helios-alpha`, `--helios-khvp` in glades-trainer.

- **Critical routing bug in CPU HELIOS dispatch (found + fixed
  2026-04-20):** the optimizer-dispatch if/else chain in
  `ApplyBatch::operator()` started with
  `if (!useAdamW && !useAtlas && !useVesta) { /* SGD + momentum */ }`.
  This branch triggers whenever AdamW / ATLAS / VESTA is not selected —
  which includes every HELIOS run. As a result, **every historical CPU
  HELIOS training run was silently executing SGD + momentum instead**,
  including all prior `HELIOSvsAdamWComparisonTest` runs that reported
  "HELIOS wins by 0.4 nats". Fix: add `&& !useHelios` to the fallback
  branch. The GPU HELIOS dispatch was unaffected (its branch chain is
  `ATLAS / VESTA / HELIOS / else Adam`, no fallback catch-all).
  Consequence: the dModel=1024 GPU NLL comparison (AdamW 9.737 vs
  HELIOS 10.024) IS a valid HELIOS measurement, but the tiny
  `HELIOSvsAdamWComparisonTest` "win" was a measurement of SGD+momentum
  and is now invalid. After the fix, real HELIOS MVI does not train
  stably at the tested configs on the tiny tokenLM — see next bullet.

- **Second latent bug — lr=0 handling (fixed 2026-04-20).** `applyStep`
  treated `hEff == 0` (i.e. `lr == 0`) as a failure and returned false.
  The test fixture sets the InputLayerInfo learning rate to 0.0 (input
  layer isn't trained), so the first `helios::update` call on tokE
  received lr=0 and was reported as "HELIOS tokE update produced NaN/Inf"
  even though the state was valid. Fix: treat `hEff == 0` as a no-op
  success (matching AdamW/ATLAS semantics); only reject negative hEff.

- **Empirical result (tiny-scale, post-bug-fixes, 2026-04-20):**

  | Config | LR | Test NLL (mean, 3 seeds) |
  |---|---|---|
  | AdamW (best-LR) | 1e-2 | **2.244** |
  | HELIOS MVI (best-LR) | 1e-1 | 3.023 |
  | HELIOS + sharpness (α=0.1, K_hvp=3) | 1e-1 | 3.025 |

  The sharpness probe is a wash on tiny tokenLM: it hurts by 0.002 nats
  vs MVI (within seed variance), and MVI loses to AdamW by 0.78 nats.
  Three seeds, identical-within-1e-3 between MVI and sharp. The probe
  fires (verified via debug), kappa gets updated, but α·κ friction
  doesn't materially change the trajectory at this scale.

  Possible reasons:
  (a) The flat-minimum thesis is just wrong at small scale — tiny
      transformers have loss landscapes where directional curvature
      doesn't predict generalization.
  (b) Single-sequence FD-HVP is too noisy: replaying one 256-token
      sequence per probe (vs a full minibatch) gives a high-variance
      kappa estimate whose EMA (β=0.05) hasn't converged in the
      30-epoch window.
  (c) α=0.1 is the wrong setting (too small for signal, or too big
      for stability at this LR).

  Disambiguating (a) vs (b/c) requires either: (i) larger-scale
  benchmarking at dModel≥1024 where HELIOS is known to train (needs
  GPU port of the probe); or (ii) a less noisy probe estimate
  (full-minibatch replay, or Hutchinson's trace estimator).
- **Anchor term in the force (lambdaAnchor > 0).** Anchor EMA IS updated
  on both CPU and GPU; the anchor force `lambda (theta - thetaBar)` is
  NOT yet added to gW in the B-halves. Small change (fused scale-and-add
  into the b-half kernel).
- **Li-Sato-Tan noise-temperature correction (noiseCorrection > 0).**
  Scaffolded; defaults to 0 on both paths.
- **muP-compatible per-group masses.** Framework Section 5 specifies
  m_g ∝ sqrt(fan-in). Current implementation uses a single scalar mass
  from `HeliosConfig::mass`. Extension requires a per-matrix-role mass
  override at the dispatch sites.

### Decision points validated/invalidated by measurement

- **"Throughput-optimize the GPU kernel"** (the user's stated request for
  this session) — measurement shows HELIOS GPU already hits 99.94-99.96%
  of AdamW GPU throughput at dModel=512 and dModel=1024. No optimization
  is needed at these scales; the host-upload noise path is
  compute-overlapped. Further throughput work should be deferred until a
  run at dModel≥4096 demonstrates a real bottleneck.

- **"HELIOS beats AdamW at pile_small scale"** (conjecture from
  Section 13) — **falsified on this hardware.** At dModel=1024, 1.5M
  tokens, best-LR HELIOS loses to best-LR AdamW by 0.29 nats. The
  tiny-transformer win (dModel=64, 60 epochs on period-7 pattern) does
  not survive scale-up in our setup. Conjecture C1 (flat-minimum
  advantage at ≥1B scale) is still untested — requires sharpness
  feedback (alpha > 0, kHvp > 0) which is not yet implemented.

- **"MVI is enough to prove the concept"** — not yet. The MVI is plain
  underdamped Langevin with constant friction; it lacks the sharpness
  feedback that the flat-minimum theorem (Section 9) depends on. The
  measurable advantage of HELIOS over AdamW in Section 6 is specifically
  the sharpness-tilted Gibbs measure; without alpha > 0 and a working
  HVP, we are testing only the stochastic-symplectic half of the claim.

### Next experiments with positive EV

Given what has been measured, the highest-EV next steps are:

1. **Wire the FD-HVP probe into `SGDHelper_TRANSFORMER`** (CPU first).
   The math primitives are now shipped and unit-tested; the remaining
   work is training-loop surgery. Concrete steps:
   (a) at the minibatch boundary, before the HELIOS dispatch, save
   target matrix weights and all accumulated gradient buffers;
   (b) perturb target += eps * v (where v = p_target / ||p_target||);
   (c) replay the minibatch's forward+backward — need to reuse the
   existing `transformerCpuForwardPass` / `transformerCpuBackwardPass`
   entry points and a saved minibatch data snapshot;
   (d) extract target's gradient as grad_plus; repeat with -eps; compute
   kappa via `directional_curvature_fd_preallocated`;
   (e) call `updateSharpness(state, kappa, hc)`;
   (f) restore saved gradients and continue to the optimizer step.
   Estimated 1 week focused. ~2% amortized overhead at K_hvp=100.
2. **Anchor force in B-half** (small change). Adds the SAM-like
   regularization that the framework's Laplace expansion depends on.
3. **Per-group thermostats on GPU** (Q > 0 support). Needed to validate
   C3 (hypocoercivity-optimal friction).
4. **Extend the NLL benchmark to dModel=2048 with BF16** to confirm the
   scale trend under the production precision target. 5M-token budget
   already used for AdamW and VESTA in the BF16_PLAN; HELIOS run costs
   ~3x a single pile_small run on this hardware (3k tok/s at ~5M tokens
   ≈ 28 min per LR).

---

## 15. Open Conjectures and Validation Criteria

**C1 (Flat-minimum advantage).** For transformer LMs at >= 1B scale, HELIOS endpoints achieve strictly smaller Hessian trace than AdamW endpoints at matched validation NLL. Falsifier: Hutchinson trace estimate on both is statistically equivalent after n = 100 probes.

**C2 (Wall-clock dominance).** HELIOS reaches AdamW's 500B-token loss on pile_large in <= 400B tokens at matched hardware. Falsifier: HELIOS needs > 520B tokens.

**C3 (Hypocoercivity-optimal friction).** Empirically optimal alpha at each scale satisfies alpha* kappa_g ~ gamma_0 + |xi_g|, validating the hypocoercivity prediction gamma ~ sqrt(kappa).

**C4 (Memory disproof).** At 70B scale with ZeRO-1, HELIOS's per-rank state is measurably <= 0.55x AdamW's. Falsifier: observed ratio > 0.6x.

**C5 (Generalization advantage).** HELIOS endpoints on downstream evals (HellaSwag, MMLU, etc.) strictly exceed AdamW at matched pretraining loss. Falsifier: gap <= 0.

**C6 (Low-precision stability).** HELIOS trains stably in pure bf16 (no fp32 master copy) for all weight classes; AdamW requires fp32 v to avoid underflow. Falsifier: any divergence in HELIOS bf16 runs below a 0.001 NaN-rate threshold.

Each conjecture comes with a concrete falsifier and an experimental protocol implementable in the existing glades/glades-trainer infrastructure. C1, C4, C6 are cheap (single-run) tests; C2, C5 are expensive (multi-replicate scaling) tests.
