# EALRMN-v1 Design Memo
## Entropy-Adaptive Latent Recurrent Memory Network

**Status:** Research design — not yet implemented. Date: 2026-05-18.

**One-line summary.** A finite-rank Koopman operator acts on an entropy-segmented latent stream; its spectral decomposition is the bounded associative memory; switching between learned operators is the sparse-expert mechanism; adaptive spectral truncation is compute-adaptivity; training is a single rate-distortion Lagrangian augmented with a predictive-information bottleneck.

**What this memo is.** A first-principles mathematical framework that can be falsified or supported piece-by-piece on controlled synthetic tasks before any claim of advantage on natural language.

**What this memo is not.** A claim that any of these mechanisms beats dense Transformers at scale. The point is to construct the test, not to declare the verdict.

---

## 0. Table of contents

1. Executive summary
2. Candidate formulations and selection
3. Formal problem statement and hypothesis
4. EALRMN-v1: core mathematical framework
5. Training objective (rate-distortion Lagrangian)
6. Complexity analysis
7. Limiting cases (Transformer / RNN / SSM / VQ-VAE)
8. Falsifiable claims with experiments
9. Synthetic datasets (generative specifications)
10. Ablation matrix
11. Metrics
12. Failure modes — when this hypothesis should fail
13. Minimal prototype (what to build first)
14. Full research program
15. Open conjectures and validation criteria

---

## 1. Executive summary

Dense autoregressive Transformers minimize $L_\text{LM}(\theta) = -\sum_t \log p_\theta(x_t \mid x_{<t})$ over raw surface streams. Five structural inefficiencies are postulated:

(I1) **Surface redundancy.** Many sequences $x_{1:T}$ map to the same latent meaning; the LM objective spends parameters distinguishing equivalent forms.

(I2) **Uniform per-token compute.** Each token activates the full forward pass irrespective of informational content.

(I3) **Quadratic attention.** Self-attention costs $O(L^2 d)$ per layer; the KV cache costs $O(N_\text{layers} \cdot L \cdot d)$.

(I4) **Entangled storage.** Mutable facts and procedural skills share the same parameter pool; updating one perturbs the other.

(I5) **No bounded long-range carrier.** Long-range information is reconstructed through context rather than stored compactly.

EALRMN-v1 hypothesizes that these inefficiencies are *separable* and *individually testable*. The framework introduces eight mechanisms, each with a specific mathematical signature, and trains them jointly through a single rate-distortion Lagrangian
$$
\mathcal{L}_\text{total} \;=\; R_\text{total} \;+\; \beta\, D_\text{total} \;-\; \beta_z\, I_\text{NCE}(z; \Phi).
$$
Each mechanism contributes a labelled term to $R_\text{total}$ or $D_\text{total}$, so every claim has a corresponding ablation: turn off the term, observe what changes.

**Core mathematical object.** The latent stream $\{z_t\}$ is acted on by a finite-rank switching Koopman operator $\hat K_{g_t}$:
$$
s_{t+1} \;=\; \hat K_{g_t}\, s_t \;+\; \hat B_{g_t}\, z_t.
$$
$s_t$ is the recurrent state, $z_t$ is the encoded patch, $g_t$ is the routed expert. The eigen-pairs $(\lambda_k, v_k)$ of $\hat K$ form the spectral memory; truncating to top-$r$ modes is compute-adaptive inference; rank-1 updates to $\hat K$ are memory writes; the residual of the orbit projection drives entropy-adaptive segmentation.

**Falsifiable claims (six).**

1. (Segmentation) Surprisal-driven segmentation strictly improves accuracy/compute Pareto over fixed-length segmentation, on tasks with non-uniform information density.
2. (Latent prediction) Predicting $\hat z_{t+h} = \hat K^h s_t$ in closed form learns latent rule structure faster than raw next-token prediction on synthetic finite-state processes.
3. (Bounded memory) Spectrally decomposed bounded memory of capacity $K \ll T$ matches a Transformer's KV cache on long-range retrieval up to a precise information-theoretic threshold; beyond that threshold it should fail.
4. (Sparse experts) Switching operators improve loss per active parameter compared to dense recurrence of equal total parameter count, on data with multi-regime dynamics.
5. (Selective recurrence) Operator-based recurrence with $|\lambda| \to 1$ retains long-range information better than a small Transformer of equal memory budget on persistence tasks.
6. (Write regularization) A nonzero write penalty improves generalization by forcing selective retention; an excessively large penalty collapses memory.

Each claim has a defined experiment, a controlled baseline, an expected positive outcome, and an expected negative outcome (Section 8).

**Honest limits.** Where the underlying token dynamics have continuous-spectrum or high effective Koopman rank — e.g., adversarial cryptographic streams, exact long-literal recall — EALRMN-v1 is *expected to lose* to dense Transformers (Section 12). This is the architectural bet: that natural language, code, and structured signals have *low effective Koopman rank* on a learned dictionary. The framework's purpose is to make this bet falsifiable.

---

## 2. Candidate formulations and selection

Three materially-different formulations were developed in parallel.

**Candidate A — SSVIB** (Stochastic State-Space + Variational Information Bottleneck). The framework is a joint measure $P_\theta$ on $(x_{1:T}, z_{1:N}, s_{1:N}, M_{1:N})$ with variational posterior $Q_\phi$; the training objective is a modified ELBO with an IB penalty $-\beta_z\, I(z; \Phi)$. Each mechanism is a KL divergence in the ELBO. Strengths: principled probabilistic foundation, mature variational machinery, clean posterior-collapse diagnostics. Weaknesses: posterior collapse risks for $z$, $s$, $M$ simultaneously; contrastive estimators saturate at $\log N_\text{neg}$; reparameterization-gradient noise on discrete write decisions; does not natively produce a closed-form multi-step predictor.

**Candidate B — KOSM** (Koopman / Operator-theoretic with Spectral Memory). Sequence dynamics are viewed as a (typically nonlinear) shift map $F$ whose Koopman operator $K$ is linear on observables $\varphi$. The learned object is a finite-rank $\hat K$ acting on a parametric dictionary realized by an encoder $E_\theta$. The recurrent state is $s_t = \hat K^t s_0 + \text{input drive}$; spectral memory is the eigen-decomposition of $\hat K$; experts are a partition of latent space across which $\hat K$ switches (SLDS form); segmentation is residual-driven; compute-adaptive inference is spectral truncation. Strengths: closed-form multi-step predictor $\hat z_{t+h} = \hat K^h s_t$; each mechanism has a precise linear-algebraic signature; spectral truncation is natively compute-adaptive; rank-1 operator updates are natively memory writes. Weaknesses: linearity-on-observables is the architectural bet; EDMD identifiability requires full-rank covariance; non-normal operators have pseudo-spectral blow-up risk.

**Candidate C — RDPC** (Rate-Distortion + MDL Coupled Coding). Every mechanism is a channel with a code-length. The training objective is the rate-distortion Lagrangian $\mathcal{L} = R_\text{total} + \beta D_\text{total}$ with $R_\text{total}$ summing the seven per-mechanism rates. Strengths: single Lagrangian unifies all eight mechanisms; β-sweep directly traces the Pareto front; explicit MDL two-part code interpretation; an achievability theorem (Theorem 7.1) establishes the operational lower bound. Weaknesses: VQ training is finicky (codebook collapse, dead codewords); the formulation does not specify *how* the encoder, predictor, or memory are computed at the architectural level — these are left abstract.

**Selection rationale.** KOSM is the lead formulation because:

(R1) The user's architectural template $z_t = E(p_t)$, $s_t = U(s_{t-1}, z_t, r_t)$, $\hat z_{t+h} = P(s_t, M_t)$ is *structurally a finite-rank Koopman system*. The math and code align directly without an intermediate abstraction layer.

(R2) KOSM is the most original of the three: the spectral memory as one-pole IIR filters per eigenmode, the residual-norm signal driving simultaneously segmentation / halting / memory-write, the SLDS-style expert switching at *every* recurrence step (not as a final MoE layer) are concrete novel pieces.

(R3) The closed-form readout $\hat z_{t+h} = \hat K^h s_t$ provides multi-step latent prediction without iterative sampling — this is the mathematical realization of the user's "latent future prediction" requirement.

(R4) Adaptive spectral truncation $r_\text{active}(t) = \#\{k : |\lambda_k|^h \cdot |c_k(t)| > \epsilon\}$ is the natural compute-adaptive inference primitive — distinct from depth pruning.

(R5) The linear-algebra core (eigendecomposition, matrix powers, rank-1 updates, projections) maps directly onto C++98 numerical code that is straightforward to implement on existing glades-ml infrastructure.

But KOSM alone is incomplete in two ways that the sister candidates address better:

(R6) KOSM's training-objective skeleton is a sum of bespoke losses ($\mathcal{L}_\text{lat}, \mathcal{L}_\text{op}, \mathcal{L}_\text{var}, \mathcal{L}_\text{write}, \dots$); the *meaning* of these losses as a unified efficiency objective is unclear. **Borrow from RDPC**: cast every KOSM loss as a rate or distortion term in a single $R + \beta D$ Lagrangian. Then β-sweep directly traces the Pareto front the user asked for.

(R7) KOSM's anti-collapse mechanism (eigenvalue spread of $\Sigma_z$) is generic and indirect — it does not directly target the user's stated principle of "predictive information per unit compute". **Borrow from SSVIB**: add a contrastive lower bound $I_\text{NCE}(z; \Phi)$ on the mutual information between the latent and a predictive sufficient statistic of the future. This is the precise information-bottleneck term the user named.

**Final formulation = KOSM (architecture) + RDPC (objective skeleton) + SSVIB ($I_\text{NCE}$ regularizer on $z$).** Hereafter "EALRMN-v1".

Rejected pure SSVIB because it is the most familiar and least mathematically novel of the three; pure KOSM is more original. Rejected pure RDPC because it specifies an objective but not a concrete realization; KOSM IS the realization that RDPC abstracts over.

---

## 3. Formal problem statement and hypothesis

### 3.1 Setting

Let $\mathcal{X}$ be a discrete or continuous symbol alphabet. A *causal stream* is $x_{1:T} \in \mathcal{X}^T$ drawn from an unknown distribution $P_\text{data}$. A causal *learner* is a family of parameterised distributions $\{p_\theta(\cdot)\}_{\theta \in \Theta}$ trained to maximise a predictive-performance functional on held-out streams.

The standard learner is the autoregressive language model with parameters $\theta_\text{LM}$ and objective
$$
L_\text{LM}(\theta_\text{LM}) \;=\; -\,\mathbb{E}_{P_\text{data}}\Big[\sum_{t=1}^T \log p_{\theta_\text{LM}}(x_t \mid x_{<t})\Big]. \tag{3.1}
$$
Inefficiencies (I1)–(I5) above are claims about $\theta_\text{LM}$ structure.

### 3.2 Hypothesis (formal)

**H0 (null).** No factorisation of $p_\theta$ involving latent codes $z_t$, bounded recurrent state $s_t$, bounded memory $M_t$, sparse experts, entropy-adaptive segmentation, write regularization, and adaptive compute *jointly* improves the predictive-information-per-unit-resource Pareto frontier over a dense autoregressive Transformer trained to convergence on the same data and compute budget.

**H1 (alternative).** There exists at least one task class $\mathcal{T}$ — characterised by (i) low effective Koopman rank, (ii) variable per-token information density, (iii) bounded long-range dependency rank — on which EALRMN-v1 achieves a strictly better $(R, D)$ Pareto front than a dense Transformer at matched total parameter count.

H0 vs H1 is *task-dependent*. The hypothesis is not "EALRMN dominates everywhere" (it does not — Section 12 identifies regimes where it loses). The hypothesis is that the Pareto-front improvement is achievable in well-identified regimes.

### 3.3 Predictive information per unit resource

Define the *predictive information rate* of the source as
$$
I_\infty \;:=\; \lim_{T \to \infty} \frac{1}{T}\, I(x_{1:T/2}; x_{T/2+1:T}). \tag{3.2}
$$
A learner with $R$ bits of internal representation and $D$ bits of residual distortion achieves the operational predictive performance $I_\text{op}(R) = I_\infty - D(R)$, where $D(R)$ is the achievable rate-distortion curve. The *efficiency functional* is
$$
\mathcal{E} \;:=\; \frac{I_\text{op}}{R + C \cdot \text{FLOPs}_\text{infer} + W \cdot \text{writes} + B \cdot \text{mem-bytes}}, \tag{3.3}
$$
with weights $(C, W, B)$ chosen by the deployment scenario. Pareto-dominance in $\mathcal{E}$ means: for any choice of $(C, W, B)$ within a stated range, EALRMN-v1 produces a point in $(I_\text{op}, R + C\text{FLOPs} + \dots)$-space that no Transformer at matched total parameter count can match.

(3.3) explicitly prohibits the trivial reduction to a single scalar.

### 3.4 Independently testable mechanisms

The eight EALRMN-v1 mechanisms map to eight independent terms in the training objective:

| # | Mechanism | Objective term | Section |
|---|-----------|----------------|---------|
| 1 | Entropy-adaptive segmentation | $\mathcal{R}_\text{seg}$ | 4.2, 5.4 |
| 2 | Latent-state prediction | $\mathcal{D}_\text{lat}$ | 4.5, 5.2 |
| 3 | Selective recurrent / state-space memory | $\mathcal{R}_\text{rec}$ | 4.3, 5.5 |
| 4 | Bounded associative memory | $\mathcal{R}_\text{mem}$ | 4.4, 5.6 |
| 5 | Sparse conditional computation | $\mathcal{R}_\text{exp}$ | 4.6, 5.7 |
| 6 | Memory-write regularization | $\lambda_w \cdot \mathcal{R}_\text{write}$ | 4.4, 5.8 |
| 7 | Compute-adaptive inference | $\eta_c \cdot \mathcal{R}_\text{compute}$ | 4.7, 5.9 |
| 8 | Optional raw decoding | $\mathbf{1}_\text{decode} \cdot \mathcal{R}_\text{dec}$ | 4.8, 5.10 |

Each term has a single multiplier ($\lambda_\text{seg}, \beta, C_s, \dots$); turning a multiplier to 0 ablates the mechanism. This is the structural commitment to "every term independently testable".

---

## 4. EALRMN-v1: core mathematical framework

### 4.1 Primitive objects and notation

We index input positions by $t \in \{1, \dots, T\}$ and patches (post-segmentation) by $i \in \{1, \dots, N\}$ with $N \leq T$. The segmentation produces stopping times $0 = t_0 < t_1 < \dots < t_N \leq T$ and patches $p_i := x_{t_{i-1}+1:t_i}$ of variable length $\ell_i := t_i - t_{i-1}$.

| Symbol | Type | Meaning |
|--------|------|---------|
| $x_t$ | $\mathcal{X}$ | raw symbol at position $t$ |
| $p_i$ | $\mathcal{X}^{\ell_i}$ | $i$-th patch |
| $z_i$ | $\mathbb{C}^m$ | lifted observable vector for patch $i$ |
| $s_i$ | $\mathbb{C}^r$ | recurrent state (top-$r$ modes), $r \leq m$ |
| $\hat K_j$ | $\mathbb{C}^{r \times r}$ | learned Koopman operator for expert $j$ |
| $\hat B_j$ | $\mathbb{C}^{r \times m}$ | input-coupling for expert $j$ |
| $V \in \mathbb{C}^{m \times r}$ | basis | right eigenframe (Schur form) of the spectral memory |
| $U \in \mathbb{C}^{m \times r}$ | basis | left eigenframe; $U^* V = I_r$ |
| $\Pi_r := V U^*$ | projector | rank-$r$ spectral projector |
| $\lambda_k$ | $\mathbb{C}$ | $k$-th eigenvalue of $\hat K$ |
| $c_k(i)$ | $\mathbb{C}$ | modal amplitude $c_k(i) := u_k^* z_i$ |
| $M_i$ | rank-$K$ subspace | spectral memory = top-$K \leq r$ active modes + their amplitudes |
| $g_i$ | $\{1, \dots, J\}$ | expert pick |
| $\pi_i \in \Delta^{J-1}$ | router | soft routing distribution |
| $b_t \in \{0,1\}$ | seg. flag | patch boundary indicator at position $t$ |
| $\eta_t = -\log \hat p(x_t | x_{<t})$ | scalar | per-position surprisal under model's own running predictor |
| $\Phi_t$ | sufficient statistic | predictive sufficient statistic of $x_{>t}$ |

Constants $m, r, K, J, h, T_\text{max}$ are architecture hyperparameters. $\beta, \beta_z, \lambda_w, \lambda_\text{seg}, \lambda_\text{exp}, \lambda_\text{ent}, \lambda_\text{lb}, \eta_c, \alpha_\text{lat}, \alpha_\text{recon}$ are training hyperparameters.

### 4.2 Encoder and entropy-adaptive segmentation (M1)

The encoder $E_\theta : \mathcal{X}^* \to \mathbb{C}^m$ is a *causal byte-level* network (e.g., 1-D depthwise + pointwise convolution stack) that emits a lifted observable $z_i$ once per patch. Crucially $E_\theta$ is *patch-local* — it sees only the current patch $p_i$ and a small carry from $s_{i-1}$:
$$
z_i \;=\; E_\theta(p_i; s_{i-1}). \tag{4.1}
$$

The segmenter is a Bernoulli channel on input positions
$$
b_t \mid x_{1:t}, s_{i(t)-1} \;\sim\; \mathrm{Bernoulli}\big(\sigma(\alpha \eta_t + \beta_\text{seg})\big), \tag{4.2}
$$
with $\eta_t$ the model's own one-step surprisal and $\sigma$ the logistic. The parameters $(\alpha, \beta_\text{seg})$ are learned. A patch ends exactly when $b_t = 1$.

**Why this avoids the "fixed bucket" failure mode.** The Bernoulli rate $\sigma(\alpha \eta_t + \beta_\text{seg})$ is monotone in $\eta_t$: high-surprisal positions are more likely to be boundaries. The expected patch length is $\mathbb{E}[\ell] \approx 1/\sigma(\beta_\text{seg} + \alpha \cdot \mathbb{E}[\eta])$; under a stationary stream this is well-defined and finite. Patches are explicitly *not* equal-length in input positions; they are approximately equal in *information content* once $\alpha$ is tuned (a derivable result under iid bits, Theorem 4.1 below).

**Theorem 4.1 (rate-equalising property; derivable).** Under (4.2) with the segmenter trained to minimise $\mathbb{E}[\,(L_{\text{enc},i} - r_\text{target})^2\,]$ over a stationary input stream and a fixed encoder $E_\theta$, the segmenter is Bayes-optimal in the class of causal Bernoulli stopping rules iff each patch carries exactly $r_\text{target}$ nats of encoder-rate in expectation.
*Proof sketch.* This is the optimality of the stopping rule for the rate-budget problem: choose stopping time $\tau$ minimising $(\sum_{t \leq \tau} \eta_t - r_\text{target})^2$; the optimal causal rule is to stop the first time the running sum crosses $r_\text{target} - \delta$ (one-sided) for an appropriate $\delta$. Bernoulli relaxation (4.2) is a smoothed version. ∎

**Differentiability.** $b_t$ is binary; gradients propagate through a Gumbel-softmax / straight-through relaxation. The relaxation temperature $\tau_b$ is annealed.

### 4.3 Operator-theoretic recurrence (M3 + M5)

The recurrent state evolves under the routed Koopman operator
$$
s_i \;=\; \hat K_{g_i} s_{i-1} \;+\; \hat B_{g_i} z_i. \tag{4.3}
$$
Each operator $\hat K_j \in \mathbb{C}^{r \times r}$ has spectral decomposition $\hat K_j = V_j \Lambda_j U_j^*$ with $U_j^* V_j = I_r$. (Or, for stability, the Schur form $\hat K_j = Q_j T_j Q_j^*$ with $T_j$ upper-triangular and $Q_j$ unitary.)

The router emits
$$
\pi_i \;=\; \mathrm{softmax}(W_\pi[s_{i-1}; z_i] / \tau_\pi) \in \Delta^{J-1}, \qquad g_i \;=\; \mathrm{TopK}(\pi_i, k_\text{top}). \tag{4.4}
$$
Hard routing in the forward pass; soft (mixture) routing only used in the gradient through the router itself.

**Why this is "selective" recurrence in the SSM sense.** The diagonal form of (4.3) in eigen-coordinates is
$$
c_k(i) \;=\; \lambda_k(g_i) \cdot c_k(i-1) \;+\; (\hat B_{g_i} z_i)_k. \tag{4.5}
$$
Each mode is a one-pole IIR filter with pole $\lambda_k$. Selectivity arises from (i) the expert switch changing the operator (and so the pole locations) and (ii) the input drive $\hat B_{g_i} z_i$ being input-dependent. This recovers Mamba / S4-style selective SSMs as a limit (Section 7.3).

**Spectral stability constraint.** All operators satisfy $\|\hat K_j\|_2 \leq 1 + \varepsilon$ in operator norm. Equivalently each $|\lambda_k(j)| \leq 1 + \varepsilon$. The slow modes ($|\lambda_k| \approx 1$) carry persistent state; transient modes ($|\lambda_k| \ll 1$) carry fast-decaying signal. This is enforced as a penalty (Section 5.5).

**Identifiability.** Given a fixed expert pick $g$, the operator $\hat K_g$ is identifiable from the trajectory $\{z_i\}_{g_i = g}$ iff the empirical covariance $\Sigma_z^{(g)}$ has full rank — the EDMD identifiability condition. The anti-collapse regulariser (Section 5.3) enforces this.

### 4.4 Spectral / associative memory (M4 + M6)

The memory $M_i$ at patch $i$ is the *spectral decomposition* of the running mixture-of-operators applied to past inputs. Specifically, in mode-amplitude coordinates,
$$
M_i \;:=\; \big( \{(\lambda_k(j), v_k(j), c_k(i))\}_{k \leq r, j \leq J} : \text{active set } \mathcal{A}_i \big), \tag{4.6}
$$
with active set $|\mathcal{A}_i| \leq K \leq r \cdot J$ a fixed capacity. The mode amplitude $c_k(i)$ is updated by (4.5); the *write* operation is a rank-1 update of $\hat K_{g_i}$:
$$
\hat K_{g_i} \;\leftarrow\; \hat K_{g_i} \;+\; a_i\, b_i^*, \qquad a_i, b_i \in \mathbb{C}^r, \tag{4.7}
$$
with $a_i, b_i$ learned heads that read off $s_i$ and $z_i$. By the Bunch–Nielsen–Sorensen rank-1 update formulas, (4.7) is a controlled perturbation of the spectrum of $\hat K_{g_i}$.

A write event is the indicator $w_i := \mathbf{1}[\|a_i\|_2 \cdot \|b_i\|_2 > \tau_w]$; the rate of writes is
$$
\rho_\text{write} \;:=\; \mathbb{E}_i[\, w_i \,]. \tag{4.8}
$$

**Why this is not "a longer KV cache".** A KV cache stores $L$ raw $(k_\tau, v_\tau)$ pairs in a list of length $L \approx T$. The spectral memory $M_i$ stores at most $K$ eigen-modes, each a $(\lambda_k, v_k)$ pair where $v_k$ is a *direction in observable space* (not a per-position value vector), and the running mode amplitudes $c_k(i)$ are computed by the recurrence (4.5). The capacity $K$ is fixed *a priori* and independent of $T$. Mathematically $M_i$ lives in the Grassmannian $\mathrm{Gr}(K, m)$ — a manifold whose dimension is $K(m-K)$, independent of $T$.

**Read.** A query $q_i = W_q s_i$ is read against memory by attention in the *modal basis*:
$$
\alpha_k \;\propto\; \exp(\langle q_i, v_k \rangle / \sqrt{m}), \qquad r_i \;=\; \sum_k \alpha_k\, c_k(i)\, v_k. \tag{4.9}
$$
The output $r_i$ is fed back to the encoder $E_\theta$ for the next patch and to the predictor (Section 4.5).

### 4.5 Latent-state prediction (M2)

The closed-form $h$-step latent predictor is
$$
\hat z_{i+h} \;=\; V_{g_i}\, \mathrm{diag}(\lambda_1^h, \dots, \lambda_r^h)\, U_{g_i}^*\, s_i \;+\; \underbrace{\sum_{j=1}^{h-1} V_{g_i}\, \mathrm{diag}(\lambda^{h-j})\, U_{g_i}^*\, \hat B_{g_i} \hat z_{i+j}}_{\text{input-driven term}}. \tag{4.10}
$$
For autonomous prediction ($\hat B = 0$) this collapses to $\hat z_{i+h} = \hat K_{g_i}^h s_i$, a single $r \times r$ matrix power — $O(r^2 \log h)$ via repeated squaring or $O(r)$ if diagonalised. **This is the operational benefit of the Koopman lift: multi-step prediction is closed-form, not iterative sampling.**

The primary supervisory signal is the latent multi-step MSE:
$$
\mathcal{D}_\text{lat}(\theta) \;=\; \mathbb{E}_i\Big[ \sum_{h=1}^{h_\text{max}} w_h\, \|\hat z_{i+h} - \mathrm{stopgrad}(z_{i+h})\|_2^2 \Big], \qquad w_h = \gamma^{h-1}. \tag{4.11}
$$
The stop-gradient on the target $z_{i+h}$ prevents the trivial collapse $z \equiv \text{const}$.

### 4.6 Sparse conditional computation (M5)

The expert switch $g_i$ at every recurrence step is the sparse-computation primitive. Activated experts contribute their operator $\hat K_{g_i}$ and coupling $\hat B_{g_i}$; non-activated experts cost zero. The total parameter count of $\{\hat K_j, \hat B_j\}_{j=1}^J$ is $J \cdot (r^2 + r m)$, but the *active* parameter count per step is $k_\text{top} \cdot (r^2 + r m) \ll J \cdot (r^2 + r m)$ when $k_\text{top} = 1$ or $2$.

Load balance is enforced by
$$
\mathcal{R}_\text{lb} \;=\; \lambda_\text{lb}\, \mathrm{KL}\Big( \tfrac{1}{N}\sum_i \pi_i \,\Big\|\, \mathrm{Uniform}_J \Big), \tag{4.12}
$$
and the router-entropy floor
$$
\mathcal{R}_\text{ent} \;=\; \lambda_\text{ent}\, \big(\, H_\text{floor} - \mathbb{E}_i[H(\pi_i)] \,\big)_+. \tag{4.13}
$$
Both terms are MDL-form penalties (Section 5).

### 4.7 Compute-adaptive inference (M7)

At inference, mode $k$ is *active* at step $i$ iff
$$
|\lambda_k|^h_{pred} \cdot |c_k(i)| \;>\; \varepsilon_\text{tol} / r. \tag{4.14}
$$
The *active spectrum size* $r_\text{active}(i) := |\{k : (4.14) \text{ holds}\}|$ is the per-step compute footprint. The recurrence (4.5) and predictor (4.10) only iterate over active modes; inactive modes contribute zero. This is the operator-theoretic realisation of compute-adaptivity *without* depth pruning: depth is fixed, but the *effective spectrum size* shrinks on easy tokens.

**Important contrast with depth pruning.** Depth pruning reduces $N_\text{layers}$, which decreases representational depth. Spectral truncation reduces the *active mode count*, which decreases the rank of the linear evolution operator but preserves depth. These are orthogonal compute axes.

### 4.8 Optional raw decoding (M8)

A side decoder $D_\theta : \mathbb{C}^r \to \Delta^{|\mathcal{X}|-1}$ produces a distribution over the next raw token:
$$
\hat p(x_{t_i + 1} \mid s_i) \;=\; D_\theta(s_i). \tag{4.15}
$$
$D_\theta$ is invoked only on (i) training positions sampled with probability $p_\text{decode-train} \in (0, 1]$ and (ii) inference positions where downstream actually needs $x$. Otherwise it is dormant. This is the operator-theoretic realisation of *lossy compression that retains predictive sufficiency*: the latent $s_i$ carries enough information to predict future $z$'s and to recover $x$ when needed, but full-history reconstruction is not required.

---

## 5. Training objective (rate-distortion Lagrangian)

### 5.1 Form of the objective

The unified EALRMN-v1 training objective is
$$
\boxed{\mathcal{L}_\text{total}(\theta) \;=\; \mathcal{R}_\text{total}(\theta) \;+\; \beta\, \mathcal{D}_\text{total}(\theta) \;-\; \beta_z\, I_\text{NCE}(z; \Phi).} \tag{5.1}
$$

We expand each term.

### 5.2 Distortion terms

$$
\mathcal{D}_\text{total} \;=\; \alpha_\text{lat} \mathcal{D}_\text{lat} \;+\; \alpha_\text{recon} \mathcal{D}_\text{recon} \;+\; \alpha_\text{task} \mathcal{D}_\text{task}. \tag{5.2}
$$
- $\mathcal{D}_\text{lat}$: multi-step latent MSE (4.11).
- $\mathcal{D}_\text{recon}$: optional raw-token NLL on decode-invoked positions:
$$
\mathcal{D}_\text{recon} \;=\; \mathbb{E}\big[\, \mathbf{1}_{\text{decode},t} \cdot (-\log p_{D_\theta}(x_t \mid s_{i(t)})) \,\big]. \tag{5.3}
$$
- $\mathcal{D}_\text{task}$: task-specific distortion (cross-entropy on labels, retrieval-accuracy proxy, etc.); set to zero in pretraining.

### 5.3 Encoder rate and the predictive information bottleneck

The encoder rate is the per-patch description length of $z_i$ under its prior $p(z_i)$:
$$
\mathcal{R}_\text{enc} \;=\; \sum_i \mathbb{E}\big[\, -\log p(z_i) \,\big]. \tag{5.4}
$$

The **predictive information bottleneck** lower-bounds $I(z_i; \Phi_{t_i})$ via InfoNCE with $N_\text{neg}$ negatives drawn cross-batch:
$$
I_\text{NCE}(z; \Phi) \;=\; \mathbb{E}\Big[\,\log \frac{f_\xi(z_i, \Phi_{t_i})}{\frac{1}{N_\text{neg}}\sum_{j=1}^{N_\text{neg}} f_\xi(z_i, \Phi_{t_j})}\,\Big], \tag{5.5}
$$
with $f_\xi$ a learned bilinear or MLP scorer. $\Phi_{t_i}$ is a *frozen* slow-moving teacher's embedding of $x_{t_i+1:t_i+W}$ for a future window $W$. The contrastive term is *subtracted* from the loss with weight $\beta_z$, so its lower-bounding role drives $I(z; \Phi)$ *up*. This is the SSVIB-style IB on the encoder.

**Anti-collapse role of $I_\text{NCE}$.** The trivial collapse $z_i \equiv 0$ achieves $I_\text{NCE} = 0$; an informative encoder achieves $I_\text{NCE} \to \log N_\text{neg}$. The gap is the encoder's predictive information content. Combined with the encoder rate (5.4), the IB Lagrangian is
$$
\mathcal{L}_\text{IB}(z) \;=\; \mathcal{R}_\text{enc} \;-\; \beta_z\, I_\text{NCE}(z; \Phi), \tag{5.6}
$$
which is exactly the rate-distortion problem $\min I(z; x)$ subject to $I(z; \Phi) \geq r$, with $r$ controlled by $\beta_z$.

### 5.4 Segmentation rate

The Bernoulli segmenter (4.2) has rate
$$
\mathcal{R}_\text{seg} \;=\; \mathbb{E}\Big[\, \sum_t \big( -b_t \log \pi_t - (1-b_t)\log(1-\pi_t) \big) \,\Big]. \tag{5.7}
$$
The rate-equalising target (Theorem 4.1) gives an *additional* loss
$$
\mathcal{L}_\text{seg-target} \;=\; \lambda_\text{seg} \cdot \mathbb{E}\big[\, (L_{\text{enc},i} - r_\text{target})^2 \,\big]. \tag{5.8}
$$

### 5.5 Recurrence rate (channel-capacity bound)

Under the variational Gaussian formulation of the recurrence channel (cf. SSVIB-eq.6), the per-step information cost of updating $s_{i-1} \to s_i$ via input $z_i$ is upper-bounded by
$$
\mathcal{R}_\text{rec} \;=\; \mathbb{E}\big[\, \mathrm{KL}(p_\psi(s_i \mid s_{i-1}, z_i) \,\|\, p(s_i \mid s_{i-1})) \,\big] \;\leq\; \mathbb{E}[I(s_{i-1}, z_i; s_i)] \;\leq\; C_s. \tag{5.9}
$$
The bound $C_s$ is the *selectivity capacity* — small $C_s$ forces sparse channel use (selective gating); large $C_s$ allows dense recurrence. The Koopman parameterisation (4.3) is a deterministic limit of the Gaussian channel as variance → 0.

The spectral stability penalty enforces $|\lambda_k(j)| \leq 1 + \varepsilon$:
$$
\mathcal{R}_\text{stab} \;=\; \lambda_\text{stab} \sum_{k,j} (\,|\lambda_k(j)| - (1+\varepsilon)\,)_+. \tag{5.10}
$$

### 5.6 Memory rate

The memory update has rate
$$
\mathcal{R}_\text{mem} \;=\; \mathbb{E}_i\big[\, \|\Delta \hat K_i\|_* \,\big] \;=\; \mathbb{E}_i\big[\, \|a_i\|_2 \cdot \|b_i\|_2 \,\big], \tag{5.11}
$$
the nuclear-norm cost of the rank-1 update (4.7). This is the precise "bits added to memory" cost, *not* a slogan.

### 5.7 Routing rate

$$
\mathcal{R}_\text{exp} \;=\; \mathcal{R}_\text{lb} \;+\; \mathcal{R}_\text{ent}, \tag{5.12}
$$
from (4.12)–(4.13).

### 5.8 Write penalty

The write rate (4.8) contributes
$$
\mathcal{R}_\text{write} \;=\; \lambda_w \cdot \rho_\text{write} \;=\; \lambda_w \cdot \mathbb{E}_i[\, w_i \,]. \tag{5.13}
$$
This is the MDL hyperprior on memory complexity: a $\mathrm{Bernoulli}(\epsilon)$ write prior gives $-\log \epsilon$ nats per write; $\lambda_w = -\log \epsilon$.

### 5.9 Compute penalty

$$
\mathcal{R}_\text{compute} \;=\; \eta_c \cdot \mathbb{E}_i[\, r_\text{active}(i) \,] \;+\; \eta_\text{seg} \cdot \mathbb{E}[N]. \tag{5.14}
$$
Includes both the per-step active-mode count (4.14) and the total patch count $N$ (more patches = more compute).

### 5.10 Decoder rate

$$
\mathcal{R}_\text{dec} \;=\; \mathbb{E}\big[\, \mathbf{1}_{\text{decode},t} \cdot (-\log p_{D_\theta}(x_t \mid s_{i(t)})) \,\big] \;=\; \mathcal{D}_\text{recon}. \tag{5.15}
$$
(Same quantity as the recon distortion, contributed to the total only when decode is invoked.)

### 5.11 Assembled objective

$$
\begin{aligned}
\mathcal{L}_\text{total}(\theta) \;=&\; \underbrace{\mathcal{R}_\text{enc} + \mathcal{R}_\text{seg} + \mathcal{R}_\text{rec} + \mathcal{R}_\text{mem} + \mathcal{R}_\text{exp} + \mathcal{R}_\text{write} + \mathcal{R}_\text{compute} + \mathcal{R}_\text{stab}}_{\mathcal{R}_\text{total}(\theta)} \\
&\;+\; \beta \cdot \big(\, \alpha_\text{lat} \mathcal{D}_\text{lat} + \alpha_\text{recon} \mathcal{D}_\text{recon} + \alpha_\text{task} \mathcal{D}_\text{task} \,\big) \\
&\;+\; \lambda_\text{seg} \mathcal{L}_\text{seg-target} \\
&\;-\; \beta_z \cdot I_\text{NCE}(z; \Phi).
\end{aligned} \tag{5.16}
$$

Each term has its own multiplier; setting a multiplier to 0 *cleanly removes* the corresponding mechanism (Section 10 ablation matrix).

### 5.12 Pareto sweep

Varying $\beta \in [0, \infty)$ traces a Pareto front in $(\mathcal{R}_\text{total}, \mathcal{D}_\text{total})$-space. $\beta \to 0$: rate minimisation; all latents collapse. $\beta \to \infty$: pure distortion minimisation; rate is unconstrained (and effectively limited only by the bounded-capacity constraints on $K$, $C_s$, codebook sizes). The operational regime is $\beta = O(1)$, with annealing
$$
\beta(t) \;=\; \beta_\text{min} + (\beta_\text{max} - \beta_\text{min}) \cdot \sigma\big((t - t_0)/\tau\big), \tag{5.17}
$$
or — preferred — dual ascent on $\beta$ targeting a fixed distortion level $\mathcal{D}^*$.

---

## 6. Complexity analysis

### 6.1 Per-patch FLOPs

| Component | EALRMN-v1 | Dense Transformer (per-token) |
|-----------|-----------|-------------------------------|
| Encoder $E_\theta$ | $O(\ell_i \cdot d_\text{enc} \cdot d_\text{enc}')$ amortized over patch | $O(d^2)$ per token |
| Recurrence (4.3) | $O(r^2 + r m)$ per patch | n/a |
| Memory read (4.9) | $O(K m)$ | n/a |
| Spectral memory update (4.7) | $O(r^2)$ per write | n/a |
| Multi-step predictor (4.10) | $O(r^2 \cdot h)$ closed form | $O(L_\text{ctx} \cdot d^2)$ per future step (KV-cache attn) |
| Attention (full self-) | n/a | $O(L_\text{ctx}^2 d)$ per layer |
| KV-cache memory | n/a | $O(N_\text{layers} L_\text{ctx} d)$ |
| Spectral memory (modes) | $O(K m)$ | n/a |

**Per-stream FLOPs scaling.** Suppose the entropy-adaptive segmenter compresses $T$ raw tokens into $N = T / \bar\ell$ patches with average length $\bar\ell$. EALRMN-v1 total FLOPs are
$$
\text{FLOPs}_\text{EALRMN} \;\approx\; T \cdot d_\text{enc} d_\text{enc}' \;+\; N \cdot (r^2 + r m + K m + r^2 h) \;=\; O(T d_\text{enc}^2 + (T/\bar\ell) \cdot d_\text{model}^2). \tag{6.1}
$$

The Transformer baseline costs
$$
\text{FLOPs}_\text{Tx} \;\approx\; N_\text{layers} \cdot T \cdot (d^2 + L_\text{ctx} d). \tag{6.2}
$$
For long contexts ($L_\text{ctx} \to T$), the Transformer's attention term $T \cdot L_\text{ctx} d = O(T^2 d)$ dominates. EALRMN-v1's effective sequence length is $T/\bar\ell$, eliminating the quadratic factor *if* $\bar\ell$ does not shrink as $T$ grows (which holds for stationary streams under (4.2)).

### 6.2 Activation memory

EALRMN-v1 stores only $s_i \in \mathbb{C}^r$ and the memory $M_i$ of capacity $K m$. Activation memory scales as $O(N \cdot r + K m)$ per training step (or $O(N \cdot r + K m)$ if backprop-through-recurrence is used).

Transformer activation memory scales as $O(N_\text{layers} \cdot T \cdot d)$ for the residual stream plus the KV cache $O(N_\text{layers} \cdot T \cdot d)$.

**Asymptotic ratio.** At fixed model dimension $d$,
$$
\frac{\text{Activation}_\text{EALRMN}}{\text{Activation}_\text{Tx}} \;=\; O\Big(\frac{T/\bar\ell \cdot r + K m}{N_\text{layers} T d}\Big). \tag{6.3}
$$
For $r = d$, $\bar\ell = O(\log T)$, $K = O(\text{const})$, this is $O((\log T)^{-1} / N_\text{layers})$ — a factor $N_\text{layers} \log T$ savings.

### 6.3 Parameter counts (total vs active)

| Component | Total params | Active params per step |
|-----------|--------------|------------------------|
| Encoder $E_\theta$ | $d_\text{enc}^2 \cdot k_\text{layers}^\text{enc}$ | full |
| Operators $\hat K_j, \hat B_j$ | $J \cdot (r^2 + r m)$ | $k_\text{top} \cdot (r^2 + r m)$ |
| Router $W_\pi$ | $J \cdot (r + m)$ | full |
| Predictor (matrix power) | shared with $\hat K$ | $k_\text{top} \cdot r^2 \log h$ |
| Spectral memory $V, U$ | $r \cdot m$ | $r_\text{active} \cdot m$ |
| Decoder $D_\theta$ | $r \cdot |\mathcal{X}|$ | only when invoked |

For $k_\text{top} = 1$ and $J = 8$ experts, total operator parameters are 8× the per-step active count. This is the sparse-capacity claim.

### 6.4 Memory writes

Per-stream write count is $T/\bar\ell \cdot \rho_\text{write}$ where $\rho_\text{write} = \mathbb{E}[w_i]$ is the trained write rate. With $\lambda_w$ chosen to target $\rho_\text{write} = 0.05$ and $\bar\ell = 4$, the write rate is $\approx T / 80$ — one write per ~80 raw input tokens. Memory bandwidth is dominated by the rank-1 updates rather than per-token cache appends.

### 6.5 Pareto comparison summary

| Metric | EALRMN-v1 advantage regime | When it loses |
|--------|----------------------------|---------------|
| FLOPs/token | low-information regions, large $\bar\ell$ | $\bar\ell \to 1$ |
| KV/state memory | constant in $T$ | tasks needing all-pairwise interactions |
| Active params | $J k_\text{top} / J$ savings | $J = 1$ ablation |
| Memory writes | $\rho_\text{write} \cdot T/\bar\ell$ small | high-novelty streams |
| Wall-clock training | depends on op implementation | spectral updates not vectorised |
| Long-range carrier | $|\lambda| \to 1$ modes preserve $O(1)$-bit/step indefinitely | needs more than $r$ independent long-range channels |

---

## 7. Limiting cases

EALRMN-v1 contains the following standard architectures as parameter limits. These give *fair comparison axes* (set the relevant parameters and the model becomes the baseline).

### 7.1 Dense Transformer limit

Take: no segmentation ($b_t \equiv 1$, so each "patch" is one token; $\bar\ell = 1$); $J = 1$ (single expert); $\hat K \to I_r$ (operator is the identity); spectral memory $K = T$ (unbounded); the predictor (4.10) is replaced by attention-over-memory; $\beta_z = 0$; $\beta \to \infty$ on $\mathcal{D}_\text{recon}$.

In this limit:
- $s_i = s_{i-1} + \hat B z_i = \sum_\tau \hat B z_\tau$ (running sum of lifted tokens; equivalent to a residual-stream Transformer).
- $M_i = \{z_\tau, c_k(\tau)\}_{\tau \leq i}$ collects all past lifted observations (the KV cache).
- Memory read (4.9) becomes scaled-dot-product attention over the running cache.

The KOSM-style argument that *this* is exactly a (single-head, single-layer) Transformer is constructive and derivable. Stacking $N_\text{layers}$ via repeated refinement (4.10 applied iteratively) recovers a multi-layer dense Transformer.

### 7.2 Dense RNN limit

Take: $K = 0$ (memory disabled); $J = 1$; no segmentation; $\bar\ell = 1$; $\hat K = $ full-rank $r \times r$ matrix; predictor horizon $h = 1$. Then (4.3) reduces to $s_i = \hat K s_{i-1} + \hat B z_i$ — a linear RNN. Adding a nonlinearity inside the encoder gives Elman-style RNN; $\beta_z = 0$ and decode-every-token gives the standard RNN-LM objective.

### 7.3 Selective-SSM (Mamba/S4) limit

Take: $\hat K_j$ diagonal-real for all $j$; $J$ small; expert switch driven by gated input projection; no spectral memory (the modes themselves are the memory). Then (4.5) becomes the diagonal SSM
$$
c_k(i) = a_k(j) \cdot c_k(i-1) + b_k(j) (\hat B z_i)_k, \tag{7.1}
$$
with $a_k, b_k$ scalars depending on the routed expert. This is precisely the form of S4/S5/Mamba SSMs. EALRMN-v1 strictly generalises this case by allowing (a) complex-valued $\lambda_k$ (oscillatory modes), (b) full $r \times r$ operators (off-diagonal coupling), (c) rank-1 updates to $\hat K$ (the spectral memory).

### 7.4 VQ-VAE limit

Take: encoder outputs are restricted to a finite codebook $z_i \in V \subset \mathbb{C}^m$ with $|V| = K_z$; $\hat K = 0$ (no recurrence); decoder always active. Then (5.16) reduces to the VQ-VAE training objective, with the encoder/decoder distortion-rate-tradeoff parameterised by $\beta$.

### 7.5 Why these matter

For each falsifiable claim (Section 8), the baseline is a *parameter ablation* of EALRMN-v1 itself, not an external architecture. This eliminates the confound of differing initialisation, optimiser, and training recipe — the only difference between the test and the baseline is the EALRMN multiplier under test.

---

## 8. Falsifiable claims with experiments

Each claim has: (i) statement, (ii) test architecture, (iii) baseline, (iv) metric, (v) expected positive outcome, (vi) expected negative outcome, (vii) confounders.

### Claim 1 — Entropy-adaptive segmentation reduces effective sequence length while preserving task-relevant information

**Statement.** On streams with non-uniform information density, the surprisal-driven segmenter (4.2) achieves equal task accuracy as a fixed-length segmenter at strictly fewer expected patches.

**Test.** Two EALRMN-v1 runs differing only in segmenter:
- A: trained $(\alpha, \beta_\text{seg})$ in (4.2).
- B: $\alpha = 0$, $\beta_\text{seg}$ set so $\bar\ell = $ same as A's expected value.

Both target $r_\text{target} = $ same. Same encoder, recurrence, memory, predictor.

**Baseline.** B (fixed-length segmentation at matched mean patch length).

**Metric.** Predictive accuracy at 1, 4, 16-step latent prediction horizons, averaged over a held-out set. Also the *patch-count-to-accuracy* curve: at a given accuracy threshold, how many patches did each segmenter use?

**Synthetic task.** Compression-burst task (Section 9.5): long low-entropy stretches interrupted by short high-entropy events. Information density varies by factor of ~50 across positions.

**Positive outcome.** A uses 30–60% fewer patches than B at matched accuracy. The patch-count distribution of A is bimodal (long patches in low-entropy regions, short patches at events).

**Negative outcome.** A and B perform identically — surprisal is uncorrelated with task-relevant information, OR the segmenter does not converge to a surprisal-tracking solution.

**Confounders.** (i) The model's own surprisal estimator $\hat p$ may be miscalibrated early in training, causing the segmenter to lag. (ii) The rate-target $r_\text{target}$ in (5.8) interacts with segmenter behaviour; sweep it. (iii) If the encoder is too small, A and B may both saturate.

### Claim 2 — Latent prediction learns hidden-rule structure faster than raw next-token prediction

**Statement.** On streams generated by a hidden finite-state process, EALRMN-v1 trained with $\mathcal{D}_\text{lat}$ (4.11) and $\beta_z I_\text{NCE}$ (5.5) recovers the hidden-state structure faster (in training compute) than the same architecture trained only with $\mathcal{D}_\text{recon}$ (raw next-token prediction).

**Test.** Two runs:
- A: $\alpha_\text{lat} = 1$, $\alpha_\text{recon} = 0.1$, $\beta_z = 1$.
- B: $\alpha_\text{lat} = 0$, $\alpha_\text{recon} = 1$, $\beta_z = 0$.

**Baseline.** B (pure next-token prediction; raw-token language modelling).

**Metric.** Two metrics: (i) *latent decode accuracy* — given $s_i$, predict the true hidden state via a frozen linear probe; (ii) *training steps to reach 95% latent decode accuracy*.

**Synthetic task.** Hidden-Markov sequence (Section 9.2): finite-state automaton over a 4-state hidden process; observations are noisy emissions from each state.

**Positive outcome.** A reaches 95% latent decode accuracy in 2–10× fewer training steps than B; A's $s_i$ explicitly clusters by true hidden state.

**Negative outcome.** A and B are equally good — the IB and latent-MSE terms don't actually push $s_i$ toward sufficient statistics; OR the hidden structure is too easy and both saturate.

**Confounders.** (i) $\beta_z$ choice matters; sweep. (ii) The frozen teacher $\Phi$ for $I_\text{NCE}$ may itself be uninformative; warm up from a slow-moving teacher. (iii) Latent decode probes may not be linearly identifiable.

### Claim 3 — Bounded associative memory at finite $K$ matches Transformer KV on retrieval up to a precise information-theoretic threshold

**Statement.** On long-context retrieval tasks where the relevant information rank $\rho_\text{rel}$ satisfies $\rho_\text{rel} \leq K$, EALRMN-v1 with memory capacity $K$ achieves retrieval accuracy within $\epsilon$ of a dense Transformer with full KV cache. For $\rho_\text{rel} > K$, retrieval accuracy degrades smoothly.

**Test.** Vary $K \in \{4, 16, 64, 256\}$ at fixed task. Measure retrieval accuracy. Compare to dense Transformer baseline.

**Baseline.** Small Transformer with KV cache of size $L_\text{ctx} = T$.

**Metric.** Retrieval accuracy: fraction of inserted needle-tokens correctly retrieved at the query position. Also the *information-theoretic capacity ratio*: empirical $\rho_\text{rel}$ vs $K$.

**Synthetic task.** Long-context key-retrieval (Section 9.1): a few key-value pairs inserted at random positions in a long stream; query at the end asks for one value.

**Positive outcome.** Smooth degradation curve in $K$; sharp transition near $K = \rho_\text{rel}$; below this threshold EALRMN matches Transformer.

**Negative outcome.** EALRMN strictly worse than Transformer at every $K$ — write mechanism doesn't actually capture needles; OR write penalty too high.

**Confounders.** (i) $\lambda_w$ sweep; too-strong write penalty prevents needle storage. (ii) Read attention (4.9) must be strong enough to actually retrieve. (iii) Spectrum stability constraint may bias the memory away from sharp needles.

### Claim 4 — Sparse experts improve loss per active parameter

**Statement.** EALRMN-v1 with $J$ experts and $k_\text{top} = 1$ achieves equal or lower loss-per-active-parameter than EALRMN-v1 with a single expert at $J \cdot r$ total dimensions.

**Test.** Two runs at matched total parameter count:
- A: $J = 8$, $r = r_0$, $k_\text{top} = 1$. Active params $\approx r_0^2 + r_0 m$.
- B: $J = 1$, $r = r_0 \sqrt{8} \approx 2.83 r_0$. Active params $\approx (r_0 \sqrt 8)^2 + r_0 \sqrt 8 m = 8 r_0^2 + 2.83 r_0 m$.

A's total params ≈ B's active params (the "iso-capacity" comparison).

**Baseline.** B (dense with 8× the operator dimension).

**Metric.** Held-out loss at matched compute (FLOPs); ratio of held-out loss per active parameter; routing entropy and expert utilisation.

**Synthetic task.** Multi-regime dynamics (Section 9.4): a sequence whose generating distribution switches among $J' = 8$ regimes; the model must route to the right operator.

**Positive outcome.** A matches B's held-out loss at $1/J$ the active-param count; routing converges to per-regime specialisation.

**Negative outcome.** A is worse than B — experts collapse (one always-active) or routing fails to specialise; load-balance breaks.

**Confounders.** (i) Load-balance multiplier $\lambda_\text{lb}$ critical. (ii) Router temperature annealing schedule. (iii) Regime-switch frequency vs $J$ — if regimes are too short, experts cannot stabilise.

### Claim 5 — Selective recurrence preserves long-range state more efficiently than small Transformer

**Statement.** On long-context persistence tasks, EALRMN-v1 with $\lambda \to 1$ for one mode preserves a bit of information indefinitely; a Transformer of equal state-memory budget cannot.

**Test.** EALRMN-v1 with $r = 32$ vs Transformer with $L_\text{ctx} = 32 \cdot d$ (same memory budget). Both must remember a bit through $T = 10^4$ tokens.

**Baseline.** Equal-memory-budget small Transformer.

**Metric.** Bit-retention accuracy at $T = 10^4$ for varying context length.

**Synthetic task.** Single-bit persistence: a sequence with a known position carrying a bit; the query at position $T$ asks for that bit. Vary $T$ and the position of the bit.

**Positive outcome.** EALRMN retains the bit at $T \to \infty$ provided one mode satisfies $|\lambda| \approx 1$; the Transformer fails when $T > L_\text{ctx}$.

**Negative outcome.** EALRMN also fails — operator drift or write thrashing destroys the bit; OR the bit is encoded into a fast mode and decays.

**Confounders.** (i) Spectral stability multiplier $\lambda_\text{stab}$ controls slow modes. (ii) Initial spectrum spread. (iii) Mode-allocation: if all $r$ modes are needed for transient signal, no slow mode is available.

### Claim 6 — Memory-write penalty improves generalisation

**Statement.** Sweeping the write penalty $\lambda_w$ produces an inverted-U curve in test loss: too-low $\lambda_w$ encourages overfitting via memorisation; too-high $\lambda_w$ collapses memory. An interior optimum exists.

**Test.** Sweep $\lambda_w \in \{0, 0.01, 0.1, 1, 10, 100\}$. Measure held-out loss.

**Baseline.** $\lambda_w = 0$ (no penalty).

**Metric.** Held-out loss as a function of $\lambda_w$; write rate $\rho_\text{write}$ as a function of $\lambda_w$; generalisation gap (train − held-out).

**Synthetic task.** Episode-recall with distractors (Section 9.6): a few salient episodes interspersed in long irrelevant context; the model must remember the salient ones but not the distractors. Overfitting = memorising distractors.

**Positive outcome.** U-shape: lowest held-out loss at intermediate $\lambda_w$; high $\lambda_w$ pushes $\rho_\text{write} \to 0$ and recall accuracy → 0; low $\lambda_w$ overfits.

**Negative outcome.** Monotone curve in either direction — penalty has no useful regime.

**Confounders.** (i) Other regularisers (dropout, weight decay) interact. (ii) Memory capacity $K$ interacts with $\lambda_w$.

---

## 9. Synthetic datasets (generative specifications)

All datasets are *generated synthetically* from explicit processes. No natural language is used in initial Gate-0 experiments. This isolates mathematical claims from data-side confounders.

### 9.1 Long-context key-retrieval (`needle-in-haystack`)

**Generative process.**
1. Stream length $T \in \{1024, 4096, 16384\}$.
2. Vocabulary $\mathcal{X} = \{0, 1, \dots, V-1\}$ with $V = 256$.
3. Insert $K_\text{ins} \in \{1, 4, 16\}$ key-value pairs at uniform-random positions: a key token $k_j \in \mathcal{X}$ followed by a value token $v_j \in \mathcal{X}$.
4. Fill remaining positions with iid uniform samples from $\mathcal{X} \setminus (\text{keys} \cup \text{values})$.
5. At position $T$, append a query: a copy of one randomly-chosen key $k_j$.
6. The label is $v_j$.

**Latent variable.** The mapping $\{k_j \to v_j\}_j$ — a small lookup table.

**Why a token-level baseline is inefficient.** The Transformer must scan the entire context to find the matching key; its KV cache is $O(T \cdot d)$. EALRMN-v1 should write each $(k_j, v_j)$ to memory at the time it sees it; subsequent positions ignore it; the query reads the table.

**Required performance.** Retrieval accuracy as a function of $K_\text{ins}$ vs $K$ (memory capacity).

### 9.2 Hidden finite-state Markov chain

**Generative process.**
1. Hidden state $h_t \in \{1, \dots, S\}$ for $S = 4$ states, transitions $h_t \mid h_{t-1} \sim P_\text{trans}(\cdot \mid h_{t-1})$ a learned random stochastic matrix.
2. Observation $x_t \mid h_t \sim P_\text{emit}(\cdot \mid h_t)$ — a noisy emission distribution over $\mathcal{X} = \{0, 1, \dots, 31\}$ where each hidden state has a distinct emission profile but the profiles overlap (Bhattacharyya overlap 0.3).
3. Stream length $T = 2048$.

**Latent variable.** $h_t$.

**Why a token-level baseline is inefficient.** The HMM has a 4-dimensional sufficient statistic ($\Pr(h_t \mid x_{<t})$); the next-token distribution is just $\sum_h \Pr(h_t = h \mid x_{<t}) P_\text{emit}(\cdot \mid h)$. A token-level model must reconstruct the sufficient statistic each step; a latent-state model with $r = 4$ has it directly.

**Required performance.** Latent-decode accuracy = correctness of a linear probe from $s_i$ to $h_t$.

### 9.3 Noisy surface equivalence

**Generative process.**
1. Underlying *operation* $o_t \in \{1, \dots, O\}$ — a categorical sequence with finite Markov structure.
2. Surface emission: $x_t \mid o_t \sim $ a noisy multi-token emission of variable length 2–8 tokens that encodes $o_t$ with random fillers and reorderings.

For instance, $o_t = $ "ADD-3" might emit any of `(+, 3)`, `(plus, three)`, `(3, +, _filler_)`, `(add, 3)`. The vocabulary distinguishes these surface forms but they're operationally equivalent.

**Latent variable.** $o_t$.

**Why a token-level baseline is inefficient.** Many surface forms produce the same $o_t$; the model wastes capacity distinguishing them. A latent-equivalence model collapses them.

### 9.4 Multi-regime dynamics

**Generative process.**
1. Regime $r_t \in \{1, \dots, R\}$ for $R = 8$ regimes, with regime persistence governed by a geometric distribution of mean 64.
2. Within regime $r$, $x_t$ evolves according to a regime-specific dynamics $f_r(x_{<t})$.
3. Stream length $T = 4096$.

**Latent variable.** $r_t$.

**Why a token-level baseline is inefficient.** A single dense model must learn a meta-function $f^* = \sum_r \mathbf{1}[r_t = r] f_r$; an expert-routed model can learn each $f_r$ separately.

### 9.5 Compression-burst stream

**Generative process.**
1. Stream alternates between "calm" segments (length 50–200, iid uniform from a small alphabet) and "burst" segments (length 1–5, iid uniform from a large alphabet).
2. Information density: burst regions have $\log V$ nats/token; calm regions have $\log V_\text{calm}$ nats/token with $V_\text{calm} \ll V$.

**Latent variable.** segment phase (calm/burst) at each position.

**Why a token-level baseline is inefficient.** Uniform per-token compute is wasted on calm regions. Entropy-adaptive segmentation should yield long patches in calm regions, short patches in burst regions.

### 9.6 Episode recall with distractors

**Generative process.**
1. Insert $K_\text{ep}$ episodes at random positions in a long stream of length $T$.
2. Each episode is a short content-bearing snippet (a key-value pair).
3. Between episodes: iid uniform distractors.
4. Query at the end: which episode contained the key $k^*$?

**Latent variable.** which episode positions are content-bearing.

**Why a token-level baseline is inefficient.** Same as 9.1, but with a generalisation twist: distractors are statistically similar but not identical to episodes.

---

## 10. Ablation matrix

Each row is a model variant; each column is a hyperparameter or mechanism toggle. Numbered ablation IDs allow reproducible reference in later experiments.

| ID | Variant | Seg | Lat-pred | Recurr. | Memory | Experts | Write reg | Comp-adaptive | IB |
|----|---------|-----|----------|---------|--------|---------|-----------|---------------|----|
| A0 | Full EALRMN-v1 | learned | $h=4$ | $r=32$ | $K=64$ | $J=8, k=1$ | $\lambda_w=0.1$ | learned | $\beta_z=1$ |
| A1 | No segmentation | fixed $\bar\ell=4$ | $h=4$ | $r=32$ | $K=64$ | $J=8, k=1$ | $\lambda_w=0.1$ | learned | $\beta_z=1$ |
| A2 | No latent prediction | learned | $\alpha_\text{lat}=0, \alpha_\text{recon}=1$ | $r=32$ | $K=64$ | $J=8, k=1$ | $\lambda_w=0.1$ | learned | $\beta_z=0$ |
| A3 | No memory | learned | $h=4$ | $r=32$ | $K=0$ | $J=8, k=1$ | n/a | learned | $\beta_z=1$ |
| A4 | No write penalty | learned | $h=4$ | $r=32$ | $K=64$ | $J=8, k=1$ | $\lambda_w=0$ | learned | $\beta_z=1$ |
| A5 | No experts (dense) | learned | $h=4$ | $r=32$ | $K=64$ | $J=1$ | $\lambda_w=0.1$ | learned | $\beta_z=1$ |
| A6 | No compute adaptive | learned | $h=4$ | $r=32$ | $K=64$ | $J=8, k=1$ | $\lambda_w=0.1$ | fixed $r_\text{active}=r$ | $\beta_z=1$ |
| A7 | No IB ($\beta_z=0$) | learned | $h=4$ | $r=32$ | $K=64$ | $J=8, k=1$ | $\lambda_w=0.1$ | learned | $\beta_z=0$ |
| A8 | Small RNN baseline | n/a (token-level) | $h=1$ | $r=32$ | $K=0$ | $J=1$ | n/a | fixed | $\beta_z=0$ |
| A9 | Small Transformer baseline | n/a | n/a (causal LM) | n/a | KV cache | n/a | n/a | fixed | n/a |

Each ablation tests a single claim from Section 8. A0 vs A1 → Claim 1. A0 vs A2 → Claim 2. A0 vs A3 with varying $K$ → Claim 3. A0 vs A5 → Claim 4. A0 vs A8 (or A0 vs A9 at matched memory) → Claim 5. Sweep over $\lambda_w$ for A4-family → Claim 6.

---

## 11. Metrics

Every experiment reports a *fixed dashboard* of metrics so cross-experiment comparison is well-defined. Categories:

**Predictive performance**
1. Held-out NLL (per-token, raw alphabet).
2. Held-out latent-NLL (per-patch, on $z$).
3. Multi-step prediction MSE at $h \in \{1, 4, 16\}$.
4. Task-specific accuracy (retrieval, latent-decode, regime-classify).

**Information retention**
5. Linear-probe accuracy from $s_i$ to known latent (hidden state, regime, episode tag).
6. Mutual information $I_\text{NCE}(z; \Phi)$ estimate (the IB target).
7. Spectral memory utilisation: $\#\{k : |c_k(i)| > \epsilon\}$.

**Compression**
8. Effective sequence length $T/\bar\ell$.
9. Total rate $\mathcal{R}_\text{total}$ in nats per raw token (broken down by mechanism).
10. Compression ratio $T_\text{raw}/N_\text{patch}$.

**Memory usage**
11. Memory write rate $\rho_\text{write}$.
12. Memory read entropy (per-query Shannon entropy of read attention).
13. Memory utilisation: fraction of slots active in any step.
14. Overwrite rate: writes per unique slot per unit time.

**Routing / sparsity**
15. Active expert count per step (should equal $k_\text{top}$ by construction; report for sanity).
16. Routing entropy $H(\pi_i)$.
17. Expert load balance: stddev of expert usage frequency.
18. Active parameter count per step.

**Compute**
19. Training FLOPs / sample.
20. Inference FLOPs / token.
21. Wall-clock training rate (tokens/sec).
22. Active spectrum size $r_\text{active}$ distribution.

**Generalisation**
23. Held-out vs train loss gap.
24. Length-extrapolation generalisation (test on $T > T_\text{train}$).
25. Distribution-shift generalisation (test on data with novel regime).

---

## 12. Failure modes — when this hypothesis should fail

This section is required and important. The user explicitly asked for it.

### 12.1 When EALRMN-v1 is plausibly worse than a dense Transformer

**F-task-1: continuous-spectrum dynamics.** If the underlying token dynamics have a continuous Koopman spectrum (e.g., quasi-periodic with irrational frequency ratios; arbitrary nonlinear maps), the finite-rank approximation $\hat K_r$ inevitably fails. The empirical signature: residual norm $\|r_t\|$ (4.9) cannot be driven below a positive constant regardless of $r$. The hypothesised remedy (more experts, larger $r$) fails because the spectrum has positive Lebesgue measure on the unit circle.

**F-task-2: high-rank long-range dependence.** If the effective rank of the predictive sufficient statistic $\Phi_t$ grows as $\Theta(T)$ (no low-dimensional manifold ever forms), the $K$-bounded memory is fundamentally insufficient. Information-theoretically: $K \cdot (\log r + d_k + d_v)$ bits $< $ required Kolmogorov information. The Transformer's $O(T \cdot d)$ KV cache can absorb the required bits at $O(T^2)$ compute cost.

**F-task-3: exact long literal recall.** Tasks requiring bit-exact reproduction of arbitrary stretches (e.g., copy a 1000-character literal) are precisely the high-rank regime. EALRMN-v1 compresses via $z$ and discards surface information; reconstruction errors compound.

**F-task-4: cryptographic / adversarial streams.** Streams with Kolmogorov complexity equal to length (random bit streams, encrypted text) cannot be compressed. The IB target $I(z; \Phi)$ is then close to zero for any compressed $z$.

### 12.2 Training-time failure modes

**F-train-1: spectral collapse.** Signature: spectrum of $\hat K$ concentrates in a single cluster (diameter < $\delta$). Mitigation: log-Coulomb repulsion penalty $-\lambda_\text{spread} \sum_{k \neq \ell} \log |\lambda_k - \lambda_\ell|$.

**F-train-2: expert collapse.** Signature: routing $\pi_i \to e_{j^*}$ for some single $j^*$; other operators receive zero gradient. Mitigation: load-balance term $\mathcal{R}_\text{lb}$ (4.12) + Gumbel-softmax router temperature schedule + periodic dead-expert reset.

**F-train-3: switching thrashing.** Signature: $g_i$ flips between experts every step. Mitigation: temporal-smoothness penalty $-\lambda_\text{smooth} \log \pi(g_{i-1} \mid \cdot)$.

**F-train-4: memory thrashing.** Signature: $\rho_\text{write} \to 1$; slot contents oscillate. Mitigation: raise $\lambda_w$; lower the slot-rewrite learning rate $\rho$ (4.9 in SSVIB sense).

**F-train-5: posterior collapse on $z$.** Signature: $I_\text{NCE}(z; \Phi) \to 0$; encoder ignores input. Mitigation: $\beta_z$ ramp; "free-bits" lower-bound on $\mathcal{R}_\text{enc}$.

**F-train-6: $\beta$ collapse.** Signature: $\beta$ saturates at either limit during annealing; rate or distortion collapses. Mitigation: dual ascent on $\beta$ targeting fixed $\mathcal{D}^*$; clip $\beta \in [\beta_\text{min}, \beta_\text{max}]$.

**F-train-7: dictionary inadequacy.** Signature: residual $\|r_i\|$ never reduces. Mitigation: enlarge $m$; reinit encoder; random-Fourier-feature augmentation.

**F-train-8: non-normal pseudo-spectral blow-up.** Signature: $\hat K$ has well-bounded spectrum but the $V_r U_r^*$ basis has $\kappa(V_r) \gg 1$; intermediate states explode despite $\|\hat K\|$ bounded. Mitigation: enforce Schur form $\hat K = Q T Q^*$ with $Q$ unitary; penalise off-diagonal of $T$.

### 12.3 Halting / compute degeneracies

**F-comp-1: aggressive truncation.** $r_\text{active}(i) = 1$ on every token; model becomes constant-rate. Mitigation: floor $r_\text{active} \geq r_\text{min}$.

**F-comp-2: no truncation.** $r_\text{active}(i) = r$ always; no compute savings. Mitigation: increase $\eta_c$; the tolerance $\varepsilon_\text{tol}$ in (4.14) is loosened.

### 12.4 The honest summary

EALRMN-v1 explicitly bets that natural-stream dynamics have:
- Low effective Koopman rank (a few dozen modes suffice).
- Low rank of $\Phi_t$ (most "useful" past information is captured in a small subspace).
- Non-uniform information density (so segmentation has work to do).
- Multi-regime structure (so experts have work to do).

For tasks not satisfying these conditions, dense Transformers will dominate. This is by mathematical construction — not a bug.

---

## 13. Minimal prototype (what to build first)

The implementation must not anticipate later mechanisms. The smallest meaningful slice is:

### 13.1 Phase-0 build (the prototype)

Components to add to glades-ml:

**P0-1.** A `KoopmanCore` C++ class implementing (4.3) with one expert and one operator:
```cpp
// research/EALRMN_prototype.cpp (single-file C++98 prototype)
struct KoopmanCore {
    int r;            // operator dimension
    int m;            // dictionary dimension
    Matrix K;         // r x r operator
    Matrix B;         // r x m input coupling
    Vector s;         // current state (r)
    void step(const Vector& z);          // s := K*s + B*z
    Vector predict(int h) const;          // returns K^h * s, closed form
};
```

**P0-2.** A `PatchEncoder` that takes a patch of raw tokens and emits a $m$-dimensional vector. Initial choice: a 2-layer 1-D convolution + global average pooling. Implementable on existing glades-ml convolution primitives.

**P0-3.** A `Segmenter` that emits boundaries via (4.2). Initial choice: fixed surprisal threshold (learned $\alpha, \beta_\text{seg}$ come in Phase-1).

**P0-4.** A `LatentPredictor` that computes $\hat z_{i+h} = K^h s_i$ in closed form and the latent MSE (4.11).

**P0-5.** A `Trainer` driver that runs Phase-0 training on synthetic dataset 9.2 (HMM).

Components explicitly NOT in Phase-0:
- Spectral memory (P1).
- Experts (P2).
- IB / $I_\text{NCE}$ (P3).
- Write penalty (P4).
- Compute-adaptive truncation (P5).
- Decode head (P6).

**What Phase-0 answers.** Does the closed-form latent predictor $K^h s_i$ + the patch encoder + a trivial segmenter learn to track the HMM hidden state on synthetic data 9.2? This is Claim 2 in its narrowest form.

### 13.2 Phase-1 build

Add: spectral memory (4.7), $I_\text{NCE}$ contrastive term (5.5), surprisal-driven segmentation (4.2). Run datasets 9.1 + 9.5. Tests Claim 1 and Claim 3.

### 13.3 Phase-2 build

Add: experts (4.3 with $J > 1$ and router 4.4), write penalty, compute-adaptive truncation. Run datasets 9.4 + 9.6. Tests Claims 4 and 6.

### 13.4 Phase-3 build

Add: optional decode head, full ablation matrix, length-extrapolation experiments. Tests Claim 5 + generalisation.

### 13.5 Smoke-test commands

```bash
# Phase-0 prototype build
cd unit-tests/build && sh .configure.sh
make ealrmn_phase0

# Phase-0 smoke test on HMM synthetic
./ealrmn_phase0 --dataset hmm --T 2048 --r 4 --m 16 --steps 1000 --seed 42

# Expected output: linear-probe accuracy s_i → h_t reaches > 0.85 by step 1000.
# If it stays at random (0.25 for 4-state HMM), something is wrong.
```

### 13.6 First Gate-0 experiment (Phase-0 success criterion)

**Question.** Does $\hat z_{i+h} = K^h s_i$ + $\mathcal{D}_\text{lat}$ + a trivial encoder track the latent hidden state of an HMM faster than $-\log p(x_{i+1} \mid s_i)$ (the next-token baseline)?

**Pass criterion.** Linear-probe accuracy of $s_i$ for the true HMM hidden state $h_t$ reaches $\geq 0.85$ by training step 1000 with the latent-prediction objective and **does not** with the raw-next-token objective alone.

**Fail criterion.** Both objectives reach similar accuracy at similar speed → Claim 2 falsified at this scale; redesign required.

This is the single decision point that gates everything downstream. If Phase-0 fails on the HMM, the whole architecture concept needs revisiting before adding mechanisms.

---

## 14. Full research program

Beyond Phase-3 (the minimal slice), the framework extends through:

**Long-term direction 1: scale validation.** Increase $r, m, J$ to compare against actual small-scale Transformers on natural language (TinyStories, Wikipedia subset). Test whether the synthetic-task wins persist at scale and whether the underlying assumption (low effective Koopman rank) is empirically supported on natural text.

**Long-term direction 2: nonlinear lift.** Replace the linear-on-observables $\hat K z$ with a low-rank-perturbation nonlinear $f_\theta(z, s)$ where $f$ is *near-linear* (Jacobian close to a sparse matrix). This relaxes the Koopman bet.

**Long-term direction 3: learned dictionaries.** Replace the fixed encoder $E_\theta$ with a dictionary-learning step that jointly finds $\{\varphi_i\}$ and $\hat K$ to minimise the Galerkin residual.

**Long-term direction 4: continuous-time formulation.** Replace the discrete recurrence (4.3) with a continuous-time SDE $ds = (\hat K_g s + \hat B_g z) dt + \sigma dW$; segmentation becomes a stopping-time problem on the SDE.

**Long-term direction 5: hierarchical memory.** Multi-level spectral memory where slow modes feed into a separate operator acting on slower time scales.

These extensions are explicitly Phase-4+ and not part of the current memo's commitments.

---

## 15. Open conjectures and validation criteria

The following are stated as **conjectures** with explicit validation criteria. The first three are central and should be the focus of early experiments.

**Conjecture C1 (low effective Koopman rank of natural streams).** For natural-stream data (English text, code, audio, sensor logs), there exists $r^* \ll T$ and an encoder $E_\theta$ such that the rank-$r^*$ Koopman approximation $\hat K_{r^*}$ achieves residual $\|r_t\|/\|z_t\| < 0.1$ on average.

*Validation.* Train EALRMN-v1 on natural data at varying $r$; measure residual. Falsified if residual stays high at any reasonable $r$.

**Conjecture C2 (variance reduction by IB).** The contrastive IB term $-\beta_z I_\text{NCE}(z; \Phi)$ strictly improves convergence rate of the latent predictor compared to the same architecture without it.

*Validation.* Ablation A0 vs A7. Falsified if A0 doesn't train measurably faster than A7 on Claim 2's HMM task.

**Conjecture C3 (spectral memory matches KV cache up to rank threshold).** On retrieval tasks with relevant-information-rank $\rho_\text{rel}$, spectral memory of capacity $K \geq \rho_\text{rel}$ matches Transformer KV-cache retrieval accuracy within 5%.

*Validation.* Claim 3. Falsified if EALRMN underperforms Transformer at $K \geq \rho_\text{rel}$.

**Conjecture C4 (sparse-expert capacity bonus).** Total parameters $J \cdot (r^2 + r m)$ provide effective capacity proportional to $J^\alpha$ for some $\alpha \in (0.3, 1)$, not $\sqrt{J}$ as a naive iso-loss bound would predict.

*Validation.* Train EALRMN-v1 with $J \in \{1, 2, 4, 8, 16\}$ at matched active-parameter count; measure held-out loss. Fit $J^\alpha$ to the loss curve.

**Conjecture C5 (rate-distortion Pareto-dominance).** On synthetic tasks where data has multi-regime structure + non-uniform info density + bounded long-range rank, EALRMN-v1 traces an $(R, D)$ Pareto front that strictly dominates a similarly-parameterised dense Transformer.

*Validation.* Sweep $\beta$ on both architectures; plot $(R, D)$ curves; check for dominance.

**Conjecture C6 (write-rate sparsity).** A well-trained EALRMN-v1 with $\lambda_w$ tuned to the data has $\rho_\text{write} \ll 1/\bar\ell$ — most patches do not write.

*Validation.* Measure $\rho_\text{write}$ post-training; expected $0.01 \leq \rho_\text{write} \leq 0.1$.

---

## Appendix A — Quick reference: symbol table

| Symbol | Meaning | First defined |
|--------|---------|---------------|
| $T$ | raw stream length | §3.1 |
| $N$ | patch count, $N \leq T$ | §4.1 |
| $\ell_i$ | $i$-th patch length | §4.1 |
| $\bar\ell$ | expected patch length | §6.1 |
| $z_i$ | lifted observable, $\in \mathbb{C}^m$ | §4.1 |
| $s_i$ | recurrent state, $\in \mathbb{C}^r$ | §4.1 |
| $\hat K_j$ | $j$-th expert's operator | §4.1 |
| $\hat B_j$ | $j$-th expert's input coupling | §4.1 |
| $\lambda_k$ | $k$-th eigenvalue of $\hat K$ | §4.3 |
| $V, U$ | right/left Schur frames | §4.1 |
| $\Pi_r = V U^*$ | rank-$r$ spectral projector | §4.1 |
| $c_k(i) = u_k^* z_i$ | $k$-th modal amplitude | §4.1 |
| $M_i$ | spectral memory at patch $i$ | §4.4 |
| $K$ | memory capacity (slot count) | §4.4 |
| $J$ | number of experts | §4.1 |
| $g_i$ | expert pick at patch $i$ | §4.1 |
| $\pi_i$ | router soft-distribution | §4.1 |
| $b_t$ | patch boundary indicator | §4.1 |
| $\eta_t$ | per-position surprisal | §4.1 |
| $\Phi_t$ | predictive sufficient statistic of $x_{>t}$ | §4.1 |
| $h$ | latent prediction horizon | §4.5 |
| $r_\text{active}(i)$ | compute-active mode count | §4.7 |
| $\beta$ | rate-distortion dual | §5.1 |
| $\beta_z$ | IB multiplier | §5.3 |
| $\lambda_w$ | write penalty | §5.8 |
| $\eta_c$ | compute penalty | §5.9 |
| $\mathcal{R}_\text{total}$ | total rate (sum of 8) | §5.11 |
| $\mathcal{D}_\text{total}$ | total distortion | §5.2 |

---

## Appendix B — What is and is not in this memo

**In scope.**
- Formal problem statement (§3).
- Formal hypothesis H0 vs H1 (§3.2).
- Operator-theoretic state-space framework (§4).
- Rate-distortion + IB Lagrangian (§5).
- Complexity vs Transformer/RNN (§6).
- Limiting cases (§7).
- Six falsifiable claims with explicit experiments (§8).
- Six synthetic-dataset generative specifications (§9).
- Ten-row ablation matrix (§10).
- 25-metric reporting dashboard (§11).
- Twelve failure modes (§12).
- Phase-0 → Phase-3 incremental build path (§13).
- Phase-0 Gate-0 success criterion (§13.6).
- Six labelled conjectures (§15).

**Out of scope (deferred).**
- Implementation code (Phase-0 will produce it).
- Choice of nonlinear lift beyond linear-on-observables.
- Continuous-time formulation.
- Scale-up beyond synthetic Phase-0.
- Comparison with named external architectures (Mamba, RetNet, GLA, etc.) — these will arise naturally once Phase-0 limits are tested.
- Claims about deployment efficiency on production hardware.

**Honest status.**
- §4 (architecture), §5 (objective), §6 (complexity), §7 (limits) — these are derivable claims under stated assumptions, mathematically grounded.
- §8 claims and §15 conjectures — these are empirically testable predictions, none yet verified.
- §13 minimal prototype — proposed, not yet implemented.

The next step is to build Phase-0 (§13.1) and run the Gate-0 experiment (§13.6). If it passes, proceed to Phase-1. If it fails, the architecture concept needs reconsideration before adding mechanisms.
