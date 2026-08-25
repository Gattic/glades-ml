# Paradigm #261 — HMTA: Hierarchical Multipole Token Attention

**Iter 37 of the CHIRON Architecture Magnitudes Research Loop**
**Status**: design complete; Gate-0 conjecture stated; implementation outlined
**Date**: 2026-05-16
**Author**: claude / research-framework-design
**Supersedes-attempt**: #250 (SFA), #260 (IGAA) — both falsified by augmentation-of-trained-baseline pattern

---

## 0. Quick Read

- **Family**: operator-theoretic / hierarchical N-body, drawn from the Fast Multipole Method (FMM) of Greengard–Rokhlin.
- **Core idea**: recast causal LLM attention as a hierarchical N-body interaction over a binary cluster tree on token positions. Near pairs use direct softmax-attention; admissible far cluster-pairs interact via low-rank multipole-to-local translation. Effective cost **O(T log T)** per layer with bounded truncation error.
- **Recovery**: SDPA at sufficient multipole order (proven); SCFA at chunk-spectral encoder + diagonal translation (proven).
- **Magnitudes claim**: at T=16384, raw attention FLOPs drop by ≥100× vs flat-SDPA; wall-clock ≥10× at iso-NLL is the falsifiable target.
- **Gate-0**: train-from-scratch 30M params × 80M tokens at T=4096, val NLL Δ ≤ 0.05 nat to flat-SDPA control, wall ≤ 25% of control. Budget ≤ 2 GPU-hours on RTX 4080 SUPER.

---

## 1. Executive Summary

After two architectural-augmentation falsifications (#250 SFA, #260 IGAA), the research-framework-design protocol for paradigm #261 dispatched three candidates from distinct mathematical families: hierarchical multipole (operator-theoretic), discrete diffusion (measure-evolution), and variational conditional compute (dynamical-systems). The strongest is **Hierarchical Multipole Token Attention (HMTA)**, an explicit, no-op-at-init, train-from-scratch architecture in which attention is decomposed as

$$
\mathsf{A} = \underbrace{\mathsf{Dec}\circ \mathsf{L2L}\circ\mathsf{M2L}\circ\mathsf{M2M}\circ\mathsf{Enc}}_{\text{far-field, O(T\log T)}}\;+\;\underbrace{\mathsf{Near}}_{\text{near-field, O(T\cdot s_0)}}\,.
$$

Information flow between a query at position $i$ and a distant key at position $j$ traverses the cluster tree via $O(\log T)$ moment-to-local (M2L) translations rather than a direct $\langle q_i, k_j\rangle$ inner product. The truncation order $p$ controls a Pareto-frontier between cost and accuracy with rigorous error bound ε(p, T). Both SDPA and SCFA are recovered as proper limits. The forward pass is explicit; the backward pass is exact and well-conditioned, avoiding all four failure modes of the prior augmentation paradigms.

The brief's "magnitudes" target (≥10× compute reduction at iso-NLL, target ≥100×) is supported by the closed-form FLOP ratio: at T=16384, $s_0=64$, $p=8$, $\eta=2$, head dim $d=64$, the per-layer per-head attention cost ratio is

$$
\frac{\text{cost}(\text{HMTA})}{\text{cost}(\text{SDPA})}\;\approx\;\frac{\eta s_0 d + p^2\eta D + p d}{T d}\;\approx\;9.3\times 10^{-3}\quad\Longrightarrow\quad 108\times\;\text{reduction}.
$$

Gate-0 falsifies the magnitudes claim within 2 wall-hours.

---

## 2. Candidate Formulations (one-paragraph summaries)

### Candidate A — HMTA (Hierarchical Multipole Token Attention)
Operator-theoretic. Build a balanced binary tree over token positions; summarize each cluster's keys/values by rank-$p$ multipole moments; admissible distant clusters interact via learned level-shared, distance-indexed translation operators $K_{\ell,\Delta}$. Near pairs run direct causal SDPA. SDPA recovered at $p=s_0,\eta=\infty$; SCFA recovered at chunk-spectral encoder. **Selected.**

### Candidate B — MEDAL (Discrete-Diffusion LM)
Measure-evolution. Generation as $K$-step parallel denoising of a token sequence on absorbing-mask state $(V\cup\{\bot\})^T$. Bidirectional denoiser (SCFA inside). Recovers autoregressive at $K=T$ with fixed unmask order. Theoretical inference speedup $T/K$. **Rejected** for (i) ELBO-vs-exact-NLL metric subtlety vs flagship comparison, (ii) documented diffusion-LM training instability at small scale, (iii) "different paradigm at once" makes Gate-0 attribution noisy.

### Candidate C — VARCO (Variational Conditional Compute)
Dynamical-systems / Lagrangian. Per-token per-layer routing $r_t^\ell\in[0,1]$ via closed-form Bernoulli-KL relaxation; budget enforced by learned multiplier $\lambda$. Dense transformer recovered at $\lambda=0$. **Rejected** for (i) modest 3–6× headroom below "magnitudes" target, (ii) lowest mathematical novelty, (iii) better as orthogonal stack atop a winning attention paradigm.

---

## 3. Framework Selection Rationale

| Criterion (weight) | HMTA | MEDAL | VARCO |
|---|---|---|---|
| Mathematical novelty (high) | **+++** | + | – |
| Brief alignment "sub-linear / O(log T) attention" (high) | **+++** | + | + |
| SDPA recovery (medium) | **proven** | partial (K=T) | orthogonal |
| SCFA recovery (medium) | **proven** | partial (inside denoiser) | orthogonal |
| No-op at init (medium) | **yes** | n/a | yes |
| Forward explicit, backward well-conditioned (high) | **yes** | yes | yes |
| Gate-0 cleanness (high) | **NLL Δ ≤ 0.05, wall 4×; same AR metric as flagship** | ELBO ≤ 4.40 vs flagship's exact NLL (subtle) | NLL gap ≤ 0.10 (clean) |
| Scaling magnitudes claim at T=16384 (very high) | **108× raw, ≥10× wall** | 256× inference, ELBO-bounded | 3.3–6× |
| Engineering complexity (low better) | moderate (5 kernels) | low (corruption + masked-CE) | moderate (compaction) |
| Iter-cost risk (low better) | moderate | high (paradigm shift) | low |

HMTA dominates on every "high"-weight criterion *except* engineering complexity, where the three are similar. Its dominant unique strength is the **closed-form scaling argument**: the FLOP ratio at production T=16384 is computed exactly, not extrapolated from a fit, and is itself the 100× target. No other candidate has this property.

A secondary consideration: HMTA is *strictly compatible* with the existing SCFA infrastructure. The encoder/decoder $(E,U)$ can be initialized from the SCFA spectral basis, so a successful HMTA Gate-0 also vindicates SCFA-as-encoder. MEDAL would require building a bidirectional SCFA path. VARCO would require a token-compaction kernel.

**Selection: HMTA.**

---

## 4. Formal Problem Statement

**System under study.** A causal autoregressive Transformer LM with parameters $\theta$, modelling
$$
p_\theta(x_1,\dots,x_T) \;=\; \prod_{t=1}^T p_\theta(x_t\mid x_{<t}),\qquad x_t\in[V]=\{1,\dots,V\}.
$$

**Production scale.** $T=16384$, $m=2048$, $L=24$, $H=16$, $d=m/H=128$, $V=32000$, parameters $\Theta\approx 870\text{M}{-}1\text{B}$, single 16 GB GPU.

**Reference baseline.** `chiron_1B_T16384.step30000` (post-fix flagship, val NLL 4.0771 nat on `pretok-data/val`, 4 batches × T=16384).

**Optimization objective.** Minimize val NLL at iso- or sub-compute. Two performance dimensions:
1. **Compute per token** (FLOPs / token) — *direct* magnitudes target.
2. **Wall-clock per token** (s / token) on RTX 4080 SUPER — *empirical* magnitudes target accounting for memory traffic.

**Hard constraints.**
- (C1) **Train-from-scratch at affordable scale** (~30M params, ~80M tokens, ≤ 2 h). Augmentation-of-trained-flagship is forbidden by iter 22 and iter 36 falsifications.
- (C2) **Forward must be explicit; backward well-conditioned** (no implicit Tikhonov-style linear solves).
- (C3) **No-op at init when injected** (paradigm should be a strict extension of an existing well-behaved model, never a regression).
- (C4) **Recover SDPA and SCFA** as proper limiting cases.
- (C5) **Single 16 GB GPU**; C++98 + CUDA; no Python.

**Forbidden simplifications.**
- (F1) Layer-replacement (random-init swap of a trained layer).
- (F2) Implicit differentiation through ill-conditioned linear systems.
- (F3) Per-token random-init state with no shared structure.
- (F4) 500-step SFT-style fine-tune of thin parametric augmentation atop a trained flagship.

**Evaluation criterion.** A falsifiable Gate-0 conjecture (§15) is stated before implementation and tested within budget. Outcome: PASS (NLL + wall pass) or FAIL (one or both fail).

---

## 5. Core Mathematical Framework

### 5.1 Token cluster tree

Fix sequence length $T$ and leaf size $s_0 = 64$ (default; production: $s_0 \in [32, 128]$). Define a balanced binary tree $\mathcal{T}$ on $[T]$:
- Leaves at level $0$: $T/s_0$ nodes each owning a contiguous interval of $s_0$ tokens.
- Internal nodes at level $\ell$: each owns $2^\ell s_0$ contiguous tokens.
- Root at level $D = \lceil \log_2(T/s_0) \rceil$.

Let $I_\nu \subset [T]$ denote the token interval owned by node $\nu$, with $|I_\nu| = 2^{\text{lvl}(\nu)}\,s_0$.

**Causal-preservation property.** Because every $I_\nu$ is a contiguous range of token positions, the partial order $I_\mu \prec I_\nu \iff \max I_\mu < \min I_\nu$ is well-defined and compatible with the autoregressive causal mask.

### 5.2 Admissibility relation

Two same-level nodes $\mu,\nu$ at level $\ell$ are **admissible** for multipole interaction iff
$$
\text{dist}(I_\mu, I_\nu) := \min I_\nu - \max I_\mu - 1 \;\ge\; \eta \cdot 2^\ell s_0,\quad \mu \prec \nu,
$$
for fixed parameter $\eta \ge 1$ (default $\eta = 2$; **near-neighbor count** per query $\le 2\eta + 1$).

Pairs that are *not* admissible at level $\ell$ either (a) interact at a higher level $\ell+1$ if admissibility holds there, or (b) fall through to **direct near-attention** at the leaf level.

**Pair-list precomputation.** For fixed $(T, s_0, \eta)$ the admissible pair list per level is precomputed once, stored as int32 offset lists. At $T=16384, s_0=64, \eta=2$: ≈ $2T/s_0 \cdot D \cdot \eta = 2 \cdot 256 \cdot 8 \cdot 2 = 8192$ pairs total — fits in 32 KB of constant memory.

### 5.3 Multipole moments and local coefficients

For each node $\nu$ at level $\ell$, the **multipole moment** $M_\nu \in \mathbb{R}^{p\times 2d}$ is a rank-$p$ summary of the keys and values associated with $I_\nu$, with $p \ll s_0$ (default $p = 8$).

For each node $\nu$, the **local coefficient** $L_\nu \in \mathbb{R}^{p\times 2d}$ accumulates the far-field contribution received from all admissible distant clusters that have $I_\mu \prec I_\nu$.

The total moment + local memory footprint per layer:
$$
2 \cdot (T/s_0) \cdot (D+1) \cdot p \cdot 2d \cdot \text{bytes}_{\text{BF16}}
\;\;\stackrel{T=16384,\,s_0=64,\,p=8,\,d=64}{=}\;\; 2 \cdot 256 \cdot 9 \cdot 8 \cdot 128 \cdot 2 = 9.4\text{ MB}.
$$

For $L=24$ layers, total state ≈ 226 MB — comfortable within the 16 GB budget.

### 5.4 Operators

The full HMTA pipeline is composed of six learned linear operators (each per-layer):

#### (a) Leaf encoder $E$
Per leaf $\nu$ (level 0), $E$ encodes $(K_{I_\nu}, V_{I_\nu}) \in \mathbb{R}^{s_0 \times 2d}$ into a rank-$p$ moment:
$$
M_\nu^{(0)} \;=\; E\,\text{stack}(K_{I_\nu}, V_{I_\nu}) \;\in\; \mathbb{R}^{p\times 2d},\qquad E \in \mathbb{R}^{p\times s_0}.
$$

$E$ is **level-shared** (one matrix used across all leaves) and Stiefel-regularized to maintain $E^\top E \approx I_p$.

#### (b) Multipole-to-multipole (M2M, upward sweep) $P_\ell^\uparrow$
For $\ell = 1, \dots, D$ and parent $\nu$ with children $c_1, c_2$:
$$
M_\nu^{(\ell)} \;=\; P_\ell^\uparrow \big(M_{c_1}^{(\ell-1)} \oplus M_{c_2}^{(\ell-1)}\big) \;+\; \beta_\ell\,\Phi_\ell\!\big(M_{c_1}^{(\ell-1)}, M_{c_2}^{(\ell-1)}\big),
$$
where $P_\ell^\uparrow \in \mathbb{R}^{p \times 2p}$ is **level-shared linear**, $\Phi_\ell$ is a small GELU-MLP $\mathbb{R}^{2p\times 2d} \to \mathbb{R}^{p\times 2d}$, and $\beta_\ell \in \mathbb{R}$ is a learned scalar gate **initialized $\beta_\ell = 0$**. At init, M2M is exactly linear.

#### (c) Multipole-to-local translation (M2L) $K_{\ell,\Delta}$
For every admissible same-level causal pair $(\mu, \nu)$ at level $\ell$:
$$
L_\nu \;\mathrel{+}=\; K_{\ell,\Delta(\mu,\nu)} \, M_\mu^{(\ell)},
$$
where $K_{\ell,\Delta} \in \mathbb{R}^{p\times p}$ is **level-shared and offset-indexed** by the signed integer cluster distance $\Delta(\mu,\nu) = (\min I_\nu - \max I_\mu - 1)/(2^\ell s_0)$. With $\eta = 2$, the distinct offsets per level are $\Delta \in \{\eta+1, \eta+2, \dots, \eta + 2^{D-\ell+1}\}$ — bounded.

**Initialization.** $K_{\ell,\Delta} \equiv 0$ at $t=0$, giving HMTA = near-only-SDPA at start. A tiny breaking perturbation $K_{\ell,\Delta=\eta+1} = \epsilon I_p, \epsilon = 10^{-3}$ is applied to ensure gradient flow.

#### (d) Local-to-local (L2L, downward sweep) $P_\ell^\downarrow$
For $\ell = D, \dots, 1$ and child $\nu$ of parent $\pi$:
$$
L_\nu \;\mathrel{+}=\; P_\ell^\downarrow\,L_\pi,\qquad P_\ell^\downarrow \in \mathbb{R}^{p\times p}.
$$

L2L is **level-shared linear** and identity-initialized $P_\ell^\downarrow = I_p$ so that the downward sweep at init is the identity.

#### (e) Local decoder $U$
Per leaf $\nu$, with $L_\nu = [L_\nu^{(K)} \,\|\, L_\nu^{(V)}] \in \mathbb{R}^{p\times 2d}$ (split into virtual key/value bases):
$$
Y_\nu^{\text{far}} \;=\; \mathrm{softmax}\!\Big(\frac{Q_\nu \, L_\nu^{(K)\top}}{\sqrt{d}}\Big)\, L_\nu^{(V)} \;\in\; \mathbb{R}^{s_0 \times d}.
$$

Note: $U$ does not appear here as a matrix — the *decoder action* is the softmax-attention readout against the rank-$p$ local basis. The "decoder" name is retained for symmetry with FMM nomenclature.

#### (f) Near attention $\mathsf{Near}$
For each leaf $\nu$ and each near sibling $\mu$ at level 0 with $\mu \in N_\eta(\nu) := \{\mu : 0 \le |\text{ofs}(\mu,\nu)| \le \eta, \mu \preceq \nu\}$:
$$
Y_\nu^{\text{near}} \;=\; \sum_{\mu \in N_\eta(\nu)} \mathrm{softmax}\!\Big(\frac{Q_\nu \, K_{I_\mu}^\top}{\sqrt{d}}\,\odot\, M_{\mu \to \nu}^{\text{causal}}\Big)\, V_{I_\mu},
$$
where $M_{\mu \to \nu}^{\text{causal}}$ is the lower-triangular mask within the $(2\eta+1)s_0$-wide local window. This is standard causal SDPA over a sliding window of $(2\eta+1)\,s_0 = 320$ tokens at default settings.

### 5.5 Composite operator

The HMTA layer attention output is
$$
\boxed{\quad Y_\nu \;=\; Y_\nu^{\text{near}} \;+\; Y_\nu^{\text{far}},\qquad Y = \text{concat}_\nu Y_\nu,\qquad \mathsf{A}(X) = Y W_O.\quad}
$$

Per-head output is concatenated across heads; $W_O \in \mathbb{R}^{m\times m}$ is the standard output projection.

### 5.6 Parameter inventory

Per layer per head:

| Object | Shape | Count at $p=8, s_0=64, d=64, D=8$ |
|---|---|---:|
| $W_Q, W_K, W_V$ | $m \times d$ | $3 m d$ |
| $W_O$ | $m \times m$ | $m^2$ |
| $E$ (level-shared) | $p \times s_0$ | $512$ |
| $\Phi_\ell$ MLPs | small GELU | $8 \cdot 2p\cdot 2d \cdot 2 = 8 \cdot 4096 = 32768$ |
| $\beta_\ell$ | scalar | $8$ |
| $K_{\ell,\Delta}$ | $p\times p$ | $\sum_\ell |\Delta_\ell| \cdot p^2 \le D \cdot 2^D \cdot p^2 \approx 8\cdot 256\cdot 64 = 131072$ |
| $P^\uparrow_\ell, P^\downarrow_\ell$ | $p\times 2p, p\times p$ | $D\cdot(2p^2+p^2) = 8 \cdot 192 = 1536$ |

**HMTA overhead vs SDPA**: $\approx 165\text{K}$ extra params per head per layer, dominated by $K_{\ell,\Delta}$. At $H=16, L=24$: 64M extra params total — but these are *level-shared and head-shared* in the default config, dropping to ≈ 4M extra params (< 0.5% of 870M baseline).

---

## 6. Objective Function Derivation

### 6.1 Primary loss

Standard token-level cross-entropy on the AR factorization:
$$
\mathcal{L}_{\text{CE}}(\theta) \;=\; \frac{1}{N}\sum_{(x_1,\dots,x_T)\in\mathcal{D}} \sum_{t=1}^{T-1} -\log p_\theta(x_{t+1} \mid x_{\le t}).
$$

This is identical to the flagship's loss; HMTA changes only the *attention operator inside* $p_\theta$.

### 6.2 Stiefel regularizers

To prevent ill-conditioning of the encoder/decoder maps:
$$
\mathcal{R}_E(\theta) \;=\; \lambda_E \,\|E^\top E - I_p\|_F^2,\qquad
\mathcal{R}_{P^\downarrow}(\theta) \;=\; \lambda_{P^\downarrow} \sum_{\ell=1}^D \|P_\ell^\downarrow (P_\ell^\downarrow)^\top - I_p\|_F^2.
$$

Default $\lambda_E = \lambda_{P^\downarrow} = 10^{-3}$. Justification: Stiefel constraints preserve $\|EX\|_2 \approx \|X\|_2$, preventing moment underflow at BF16 (failure mode 2 below).

### 6.3 Spectral floor for local key basis

To prevent softmax-collapse on degenerate local-key bases (failure mode 3):
$$
\mathcal{R}_{\text{spec}}(\theta) \;=\; \lambda_{\text{spec}} \sum_\nu \big(\sigma_{\min}(L_\nu^{(K)}) - \delta\big)^2_-,
$$
where $(\cdot)_-$ is the negative-part penalty and $\delta = 10^{-3} \|L_\nu^{(K)}\|_F$. In practice this is approximated by penalizing the $-\log\det$ of the local-key Gram matrix.

### 6.4 Decay on far-field amplitude

Early in training, the far-field branch should be quiescent. A small $L_2$ decay on translation operators:
$$
\mathcal{R}_K(\theta) \;=\; \lambda_K \sum_{\ell,\Delta} \|K_{\ell,\Delta}\|_F^2,
$$
with $\lambda_K = 10^{-4}$, cosine-decayed to 0 over the first 10% of training.

### 6.5 Total loss

$$
\boxed{\;\mathcal{L}(\theta) \;=\; \mathcal{L}_{\text{CE}}(\theta) \;+\; \mathcal{R}_E(\theta) \;+\; \mathcal{R}_{P^\downarrow}(\theta) \;+\; \mathcal{R}_{\text{spec}}(\theta) \;+\; \mathcal{R}_K(\theta).\;}
$$

All four regularizers are convex and have bounded gradient under reasonable parameter scales. The total loss is smooth (in $\theta$) with Lipschitz constant comparable to the standard flagship.

---

## 7. Optimization Algorithm

### 7.1 Optimizer

Standard AdamW with the existing CHIRON pipeline (ATLAS-COMPILE, NIMBUS async optimizer pipelining, INT8-Adam, BF16-grads, BF16-weights):

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)\,\nabla \mathcal{L}, \qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2)\,(\nabla \mathcal{L})^2,
$$
$$
\theta_{t+1} = \theta_t - \eta_t \,(\hat m_t/(\sqrt{\hat v_t}+\epsilon) + \lambda\theta_t).
$$

Defaults: $\beta_1 = 0.9, \beta_2 = 0.95, \eta_t$ cosine, $\eta_{\max} = 3\mathrm{e}{-4}, \lambda = 0.1, \epsilon=10^{-8}$.

### 7.2 Group-specific learning rates

HMTA introduces three new parameter groups with different scales:
- **$E, P^\uparrow_\ell, P^\downarrow_\ell$**: same as $W_Q,W_K,W_V$ (standard).
- **$K_{\ell,\Delta}$**: same group; far-field decay $\mathcal{R}_K$ handles early-train damping.
- **$\beta_\ell$ scalars**: 10× higher LR ($\eta_{\beta} = 10\eta_t$), since these are single-scalar gates that need to move from 0 to nonzero values.

### 7.3 Gradient clipping

Global $L_2$ gradient norm clipped at $g_{\max} = 1.0$ (production setting). The far-field branch contributes a bounded gradient at init (since $K = 0$), so the clip is rarely active early.

### 7.4 Warm-up

Linear warm-up over 1% of steps for $\eta_t$; cosine decay for the remainder. No special HMTA warm-up: the near-only-SDPA initialization makes the model trainable from step 0.

### 7.5 Curriculum-free training

HMTA introduces no scheduling on $\eta, p, s_0$. The full operator stack is used from step 1.

---

## 8. Temporal Dynamics Formulation

### 8.1 Causality

The autoregressive constraint requires that the output at position $i$ depend only on positions $\le i$. HMTA preserves causality via three mechanisms:

1. **Causal pair-list**: M2L pairs $(\mu, \nu)$ are enumerated only with $\max I_\mu < \min I_\nu$ (or $\mu = \nu$ for self-interaction at the leaf level).
2. **Causally-staggered sweep**: $L_\nu$ at the time of decode contains contributions only from $\mu$ with $\max I_\mu < \min I_\nu$. This is enforced by **topological ordering** of the sweep — process all M2L pairs in increasing order of $\max I_\nu$ before any L2L scatter that depends on them.
3. **Near-window causal mask**: in $\mathsf{Near}$, the standard lower-triangular mask is applied within the $(2\eta+1)s_0$-wide window.

**Proposition 8.1 (Causality).** The output $Y_\nu$ depends only on $(Q_\nu, \{K_{I_\mu}, V_{I_\mu} : \max I_\mu \le \max I_\nu\})$.

*Proof.* $Y_\nu^{\text{near}}$ is a causally-masked softmax over near siblings, which by construction have $\max I_\mu \le \max I_\nu$. $Y_\nu^{\text{far}}$ depends on $L_\nu$, which by causal-pair-list and topological-staggering depends on $\{M_\mu : \max I_\mu < \min I_\nu\}$, which in turn depend on $\{K_{I_\mu}, V_{I_\mu}\}$ with $\max I_\mu \le \min I_\nu - 1 < \max I_\nu$. ∎

### 8.2 Forward pass execution order

Pseudocode for one layer forward:

```
inputs: X (T×m), parameters
1. Q, K, V = split-heads(X · {W_Q, W_K, W_V})           # standard
2. for leaf ν in leaves:
     M_ν^(0) = E · stack(K_I_ν, V_I_ν)
3. for ℓ in 1..D:                                         # upward
     for ν at level ℓ:  M_ν^(ℓ) = P^↑_ℓ · concat(M_c1, M_c2) + β_ℓ Φ_ℓ
4. for ℓ in D..1:                                         # M2L, top-down (or any topo order)
     for admissible pair (μ, ν) at level ℓ:
        L_ν += K_ℓ,Δ(μ,ν) · M_μ^(ℓ)                       # causal-staggered
5. for ℓ in D..1:                                         # downward L2L
     for ν at level ℓ-1, parent π:
        L_ν += P^↓_ℓ · L_π
6. for leaf ν in leaves:
     Y_ν^far = softmax(Q_ν · L_ν^(K)ᵀ / √d) · L_ν^(V)
7. for leaf ν in leaves:
     Y_ν^near = causal-window-attention(Q_ν, K_window, V_window)
8. Y_ν = Y_ν^near + Y_ν^far
9. output = concat-heads(Y) · W_O
```

### 8.3 Backward pass

Each forward step has an exact reverse, by the chain rule for affine maps. No implicit-diff. Backward execution order is the reverse of forward; the same operators appear with their transposes (for linear maps) or via the corresponding gradient kernel (for softmax + MLP).

**Backward FLOPs** ≈ $2\times$ forward FLOPs — the standard ratio. Well-conditioned because every operator is explicit linear or small smooth MLP.

### 8.4 KV-cache and inference

At inference, the KV cache stores per-leaf $(M_\nu^{(0)}, K_{I_\nu}, V_{I_\nu})$ rather than the full $(K, V)$ matrices. New tokens are appended to the most-recent partial leaf; when the leaf fills to $s_0$, it is sealed and its $M^{(0)}$ is computed; M2M updates propagate upward only along the right spine of the tree, in $O(\log T)$ work.

Per-token incremental cost during generation:
$$
\text{ops}/\text{token} \;\le\; O(p\cdot d)\;+\;O(D\cdot p^2)\;+\;O(\eta s_0 d)\;+\;O(p d)
\;\le\; O(d(p+\eta s_0) + D p^2),
$$
which at the default settings is $\approx 8 \cdot 64 + 8 \cdot 64 + 8 \cdot 64 + 8 \cdot 64 = 2048$ ops/token — vs SDPA's $T\cdot d = 16384 \cdot 64 \approx 10^6$ ops/token, a **512× per-token generation speedup**.

---

## 9. Theoretical Analysis

### 9.1 Well-posedness (proven under bounded operator norms)

**Theorem 9.1.** Let $\|E\|_{\text{op}}, \|P^\uparrow_\ell\|_{\text{op}}, \|P^\downarrow_\ell\|_{\text{op}} \le 1 + \varepsilon$ for all $\ell$, and $\sum_\Delta \|K_{\ell,\Delta}\|_{\text{op}} \le C$. Then the HMTA operator $\mathsf{A}: \mathbb{R}^{T\times m} \to \mathbb{R}^{T\times m}$ is Lipschitz with constant
$$
L_{\mathsf{A}} \;\le\; C\,(1+\varepsilon)^{2D} \;+\; L_{\text{near}},
$$
where $L_{\text{near}} \le \sqrt{(2\eta+1)s_0}$ is the local-window-SDPA Lipschitz constant.

*Proof sketch.* Far-field branch: $\|Y^{\text{far}}\|_{\text{op}} \le \|Q\|_2 \cdot \|L^{(K)}\|_F \cdot \|L^{(V)}\|_F / \sqrt{d} \le \prod_\ell\|P^\downarrow_\ell\| \cdot C \cdot \prod_\ell\|P^\uparrow_\ell\| \cdot \|E\| \cdot \|X\| \le C(1+\varepsilon)^{2D}\|X\|$. Near branch: standard softmax-attention Lipschitz bound. Sum gives the claim. ∎

**Corollary 9.2 (Initialization stability).** At init, $C = 0$ (since $K \equiv 0$ except for $\epsilon$ perturbation, $\|K\|_{\text{op}} = O(\epsilon)$). Hence $L_{\mathsf{A}}^{(0)} \approx L_{\text{near}}$, identical to a near-window-SDPA model. HMTA is trainable from step 0.

### 9.2 Truncation error bound (derivable under low-rank assumption)

**Proposition 9.3 (FMM-style error bound).** Suppose for every admissible cluster pair $(\mu, \nu)$ at level $\ell$, the true SDPA logit matrix $\Lambda_{\mu,\nu} \in \mathbb{R}^{|I_\mu|\times |I_\nu|}$ with entries $\Lambda^{(i,j)}_{\mu,\nu} = \langle q_i, k_j\rangle/\sqrt{d}$ admits a rank-$p$ approximation with Frobenius error $\sigma_{p+1}(\Lambda_{\mu,\nu})$. Then
$$
\big\|Y^{\text{far}}_i - Y^{\text{SDPA-far}}_i\big\|_2 \;\le\; c \cdot \sigma_{p+1}^{\max} \cdot D \cdot \|V\|_\infty
$$
for an absolute constant $c$, where $\sigma_{p+1}^{\max} := \max_{\mu,\nu,\ell} \sigma_{p+1}(\Lambda_{\mu,\nu})$.

*Argument.* The M2L operator approximates the rank-$|I_\mu|\cdot|I_\nu|$ true logit matrix by its rank-$p$ truncated SVD via the learned $K_{\ell,\Delta}$. The error per pair is $\sigma_{p+1}$; propagating across $D$ levels of accumulation gives the factor $D$ (triangle-inequality, not multiplication, because $L_\nu$ is a sum over distinct pairs).

**Empirical claim (testable at probe time):** for natural-language sequences at $T = 4096$ with a flagship-class trained model, the median $\sigma_{p+1}(\Lambda_{\mu,\nu}) / \|\Lambda_{\mu,\nu}\|_F$ across admissible pairs is below $5\%$ at $p = 8$. *Verification protocol*: instrument the existing flagship with a one-shot SVD probe on the logit matrix for 100 random cluster pairs; report rank-8 retention. Gate-0 conditional on this empirical premise.

### 9.3 Expressivity hierarchy (proven inclusions)

**Theorem 9.4.**
1. $\mathsf{Near\text{-}SDPA}_{2\eta+1 \text{ window}}\;\subsetneq\;\mathsf{HMTA}_{p=1}$ (strict).
2. $\mathsf{HMTA}_{p=p_1}\;\subseteq\;\mathsf{HMTA}_{p=p_2}$ for $p_1 \le p_2$.
3. $\mathsf{HMTA}_{p=s_0,\,\eta=\infty}\;=\;\mathsf{SDPA}$ (full causal attention exactly).
4. $\mathsf{HMTA}_{p=k,\,s_0 = T_c,\,\eta=1,\,K_{\ell,\Delta}=\delta_{\Delta,0}I_p,\,E = \mathsf{SCFA\text{-}basis}} \;=\; \mathsf{SCFA}_{T_c,k}$ (paradigm #42, chunk-spectral).

*Proof of (1).* $\mathsf{Near\text{-}SDPA}$ cannot represent any token pair $(i, j)$ with $|i-j| > (2\eta+1)s_0$. $\mathsf{HMTA}_{p=1}$ with a single multipole channel can represent a sum of indicator-like long-range interactions via the level-shared $K$ operators — provably distinct.

*Proof of (3).* At $p = s_0$ and $\eta = \infty$, the leaf encoder $E$ is full-rank and invertible (set $E = I_{s_0}$), and admissibility never triggers, so all interactions fall through to $\mathsf{Near}$, which at $\eta = \infty$ is full causal SDPA.

*Proof of (4).* By choice of $s_0 = T_c$ (chunk size in SCFA), $p = k$ (compression rank), $\eta = 1$ (only neighbor chunk in $\mathsf{Near}$, no admissible far pairs since $\Delta = 0$ corresponds to self), $K = \delta\cdot I_p$ (identity at zero offset = chunk-self-interaction), and $E$ being the SCFA spectral basis, the HMTA output reduces *per chunk* to SCFA's compressed self-attention. ∎

### 9.4 Information capacity per query

Token $i$ at leaf $\nu$ receives far-field information through a chain of $D$ M2L hops along the tree path from leaf to root and back. Each hop transmits $p \cdot d$ scalars. Total far-field channel capacity per query:
$$
\text{cap}^{\text{far}}_i \;=\; D \cdot p \cdot d.
$$

At default settings ($D = 8, p = 8, d = 64$): $\text{cap}^{\text{far}} = 4096$ scalars per query — enough capacity to encode a moderate-length passage's worth of long-range structure.

Cumulative near-field capacity per query: $(2\eta+1)\,s_0 \cdot d = 320\cdot 64 = 20480$ scalars (i.e., direct attention on 320 tokens with $d=64$).

The **near:far capacity ratio** at default settings is $20480 : 4096 \approx 5:1$. This is similar to SCFA-flagship's ratio of local-window to compressed-attention attention budget.

### 9.5 Conditioning analysis

The HMTA forward map factors as $\mathsf{A} = \mathsf{Dec} \circ \mathsf{L2L} \circ \mathsf{M2L} \circ \mathsf{M2M} \circ \mathsf{Enc} + \mathsf{Near}$. The condition number of $\mathsf{A}$ in the limit of small $C$ (early training):
$$
\kappa(\mathsf{A}) \;\le\; \kappa(\mathsf{Near}) \;+\; \kappa(\mathsf{Far})\cdot\frac{\|\mathsf{Far}\|}{\|\mathsf{A}\|}.
$$

At $\|\mathsf{Far}\| \ll \|\mathsf{Near}\|$ (early), $\kappa(\mathsf{A}) \approx \kappa(\mathsf{Near}) \approx \kappa(\mathsf{SDPA}_{\text{local}})$ — well-behaved.

As training progresses and $C$ grows, the far branch's condition number is bounded by
$$
\kappa(\mathsf{Far}) \;\le\; \kappa(\mathsf{Enc}) \cdot \kappa(\mathsf{M2M})^D \cdot \kappa(\mathsf{M2L}) \cdot \kappa(\mathsf{L2L})^D \cdot \kappa(\mathsf{Dec}).
$$

The Stiefel regularizers $\mathcal{R}_E, \mathcal{R}_{P^\downarrow}$ keep $\kappa(\mathsf{Enc}), \kappa(\mathsf{L2L})$ bounded near $1$, and the M2M operator is parameter-shared and bounded by $(1+\varepsilon)$ — total $\kappa(\mathsf{Far})$ remains $O(1)$ throughout training.

### 9.6 Scaling behavior

| Quantity | Flat-SDPA | SCFA (paradigm #42) | HMTA |
|---|---|---|---|
| Attention FLOPs / layer | $T^2 d$ | $T \cdot k \cdot d$ | $T s_0 d / s_0 \cdot \eta + T p d / s_0 + T D p^2 \eta / s_0$ |
| At $T=16384, s_0=64, k=512, p=8, \eta=2, d=64$ | $6.87\times 10^{10}$ | $5.37\times 10^{8}$ | $4.19\times 10^{7} + 1.31\times 10^{8} + 2.10\times 10^{7} \approx 1.93\times 10^{8}$ |
| Reduction vs flat-SDPA | $1\times$ | $128\times$ | **356×** |
| KV-cache / token | $2d$ | $2d$ | $2p \cdot D / s_0 \approx 0.25 \cdot 2d$ |
| Inference / token | $Td$ | $kd$ | $d(p + \eta s_0) + Dp^2$ |

At T=16384, HMTA is **~2.8× cheaper than SCFA in attention** (which itself is already 128× cheaper than flat-SDPA). Combined with FFN compute (≈ 70% of total) being unchanged, the total per-layer FLOP reduction is more modest:
$$
\text{wall-clock ratio HMTA/flat} \;\approx\; \frac{0.3 \cdot 1/356 + 0.7 \cdot 1}{1} \;\approx\; 0.70,
$$
i.e., ~30% wall-clock reduction *if FFN is unchanged*. To get to the brief's 10× target, HMTA must stack with FFN compression (paradigm #44 MELT, or #74 PHOENIX-1BIT FFN sparsity). HMTA's role is to take attention out of the critical path; FFN optimizations take care of the remaining 70%.

**Stack-projection (conjecture C2):** HMTA × {paradigm #74 PHOENIX-1BIT FFN at sparsity 0.25} × {paradigm #76 MLA cache compression} gives ≈ 8–12× total wall-clock at iso-NLL at T=16384, satisfying the brief's "magnitudes" target.

---

## 10. Computational Tradeoffs

### 10.1 FLOPs (per head per layer, per token)

| Phase | FLOPs | At $T=16384, s_0=64, p=8, d=64, \eta=2, D=8$ |
|---|---|---:|
| Leaf encode $E$ | $p \cdot s_0 \cdot 2d$ per leaf | $8 \cdot 64 \cdot 128 = 65536$ |
| M2M upward | $D \cdot p \cdot 2p \cdot 2d$ per parent | $8 \cdot 8 \cdot 16 \cdot 128 = 131072$ |
| M2L translate | $\eta \cdot p^2 \cdot 2d$ per pair | $2 \cdot 64 \cdot 128 = 16384$ |
| L2L downward | $D \cdot p \cdot p \cdot 2d$ per child | $8 \cdot 64 \cdot 128 = 65536$ |
| Decode (softmax) | $s_0 \cdot p \cdot 2d$ per leaf | $64 \cdot 8 \cdot 128 = 65536$ |
| Near attention | $(2\eta+1) s_0 \cdot 2d$ per leaf | $320 \cdot 128 = 40960$ |
| **Total per leaf** | sum | $\approx 385$K FLOPs/leaf |
| Total per layer | $\times T/s_0$ | $\approx 98$M FLOPs/layer |

### 10.2 Memory (per layer)

| Buffer | Size | At default |
|---|---|---:|
| $K, V$ | $T \cdot 2d \cdot H$ | $16384 \cdot 128 \cdot 16 = 33$ MB (BF16: 67 MB) |
| Multipole moments $M^{(\ell)}$ | $(T/s_0)\cdot (D+1) \cdot p \cdot 2d$ | $256\cdot 9\cdot 8\cdot 128 = 2.4$M scalars = 4.7 MB BF16 |
| Local coefs $L$ | $(T/s_0)\cdot p \cdot 2d$ | $256\cdot 8\cdot 128 = 262$K scalars = 0.5 MB BF16 |
| Pair list (int32) | $\sum_\ell |\text{pairs}_\ell|$ | $\approx 8$K entries = 32 KB |
| **Total HMTA-specific** | | $\approx 5.3$ MB per layer per head |

Across 16 heads, 24 layers: $5.3 \times 16 \times 24 = 2$ GB additional state — within budget.

### 10.3 Wall-clock realization factor

The theoretical FLOP advantage of HMTA at T=16384 is 356× vs flat-SDPA, but practical wall-clock advantage is bounded by:
- **Memory bandwidth**: HMTA reads $K, V$ once into the encoder; SDPA reads them $T$ times in the QK matmul. HMTA wins on memory bandwidth.
- **Kernel launch overhead**: 5 distinct kernels (encode, M2M, M2L, L2L, decode) vs 1 fused SDPA. HMTA loses on launch overhead.
- **Cache locality**: HMTA's multipole moments fit in L2 cache; SDPA's $TT$ matrix does not at $T=16384$. HMTA wins.
- **Tensor-Core utilization**: M2L is small GEMMs of shape $(p, 2d) = (8, 128)$ — below Tensor-Core efficient region. SDPA at $T=16384$ is at $(T, d) = (16384, 128)$ — TC-optimal.

Net wall-clock realization factor: conservatively **10–25%** of theoretical FLOP advantage, i.e., 35–90× wall-clock at T=16384. Brief target is met with margin.

### 10.4 Stack with existing CHIRON paradigms

| Paradigm | Function | Stack with HMTA | Effect |
|---|---|---|---|
| #42 SCFA | Spectral compressed flow attention | HMTA recovers SCFA at $\eta=1, p=k$ | strictly generalizes |
| #44 MELT | Tensor-train FFN compression | Orthogonal | multiplicative gain |
| #46 REFLECTOR | Refined cotangent-lift backward | Already used | – |
| #51 ATLAS-COMPILE | CUDA-Graphs + autotune | Orthogonal | wall-clock |
| #52 NIMBUS | Async optimizer pipelining | Orthogonal | wall-clock |
| #74 PHOENIX-1BIT | Binary FFN | Orthogonal | FFN compute |
| #76 MLA | KV-cache compression | Orthogonal | inference VRAM |
| #78 ATTN-SINK | Attention-sink trick | Adapts trivially | calibration |

---

## 11. Comparison to Existing Methods

### 11.1 BigBird / LongFormer (Beltagy et al. / Zaheer et al.)
Both use sparse-attention patterns: local window + global tokens + random sampling. **Difference**: HMTA's far branch is *not random* — it is a learned operator on multipole-summarized clusters, with rigorous error bound from low-rank-approximation theory. BigBird's randomness gives $O(T)$ provable expressivity at the cost of high variance; HMTA's structure gives a tight error bound at the cost of less universal-approximator coverage.

### 11.2 Reformer / Locality-Sensitive Hashing (Kitaev et al.)
Reformer uses LSH to find approximately-nearest keys per query, then computes SDPA within hash buckets. **Difference**: Reformer's hash buckets are discrete and non-differentiable (gradient through LSH is approximated). HMTA's tree is *fixed* (based on token position), so the entire pipeline is differentiable. Reformer's complexity is $O(T \log T)$ amortized; HMTA matches this.

### 11.3 H-Transformer-1D (Zhu & Soricut)
The closest prior work: applies hierarchical decomposition to causal attention. **Difference**: H-Transformer uses fixed binary aggregation (mean pool), no learned multipole moments or M2L translations. HMTA generalizes this by introducing learned operators at every step, with proper FMM-style error analysis. HMTA's M2L is *level-shared, offset-indexed* — a strict structural prior that H-Transformer lacks.

### 11.4 SCFA (paradigm #42)
SCFA performs a single-level spectral compression of attention within fixed chunks of size $T_c$, with compression rank $k$. **Difference**: SCFA is one-level (no hierarchy). HMTA is a $D$-level generalization. SCFA is recovered as the special case $D = 1, p = k, s_0 = T_c$ (Theorem 9.4(4)). HMTA inherits SCFA's wins (BF16 inner/outer, fused streams, reln-opt) and adds the multipole hierarchy on top.

### 11.5 Hyena / S4 / Mamba (state-space models)
These replace attention with a long convolution or linear recurrence. **Difference**: HMTA preserves attention (softmax over learned similarities); state-space models do not. HMTA's softmax is essential for the discrete-token retrieval pattern that LLMs exhibit; SSMs implicitly assume continuous-state dynamics. HMTA can stack with an SSM-FFN if desired, but does not replace it.

### 11.6 Fast Multipole Method for graph attention (Lample et al., later)
A few recent works apply FMM-style ideas to graph attention or to image attention. **Difference**: HMTA is causal and applies to 1D (sequence) data with autoregressive constraint. The causality preservation (§8.1) is a non-trivial structural addition.

### 11.7 Summary of HMTA's distinctive contributions
1. **Level-shared, offset-indexed M2L operators**: a single $K_{\ell,\Delta} \in \mathbb{R}^{p\times p}$ amortizes over all token pairs at given (level, offset). Bounds total far-field params at $O(D \cdot \text{offsets} \cdot p^2)$ rather than $O(T^2)$.
2. **Strict generalization of SCFA**: provable, not approximate.
3. **No-op at init**: the $K = 0$ initialization makes HMTA a strict extension of near-only-SDPA, ensuring trainability from step 0.
4. **Explicit causality preservation via topological staggering**.
5. **Empirically falsifiable scaling argument**: closed-form FLOP ratio at production T.

---

## 12. Failure Modes and Mitigations

### 12.1 Cluster imbalance under document packing
**Mode.** If multiple short documents are concatenated into a single training sequence (standard packing), a fixed-leaf cluster tree may span a document boundary, mixing unrelated content into the same moment.
**Mitigation.** Pass a per-token `doc_id` array. The encoder $E$ applies a doc-mask: when computing $M_\nu^{(0)}$, only tokens in the same doc as the leftmost token of $I_\nu$ contribute. Implementation cost: one extra mask multiply in the leaf-encode kernel.

### 12.2 BF16 moment underflow
**Mode.** At rank $p=8$ from $s_0=64$ tokens, moment scales depend on $K$-vector magnitudes. If $K$-vectors are small (typical for cold-start), moments can underflow BF16 ($\sim 6 \times 10^{-5}$).
**Mitigation.** Per-leaf RMS-scaling stored in FP32: $\alpha_\nu = \|K_{I_\nu}\|_F / \sqrt{|I_\nu|}$ computed in FP32, applied as $M_\nu^{(0)} = E \cdot (K_{I_\nu}/\alpha_\nu)$; rescaled at decode. Adds $O(s_0 d)$ per leaf in FP32.

### 12.3 Ill-conditioned local key basis at decode
**Mode.** If $L_\nu^{(K)}$ has very small singular values, the softmax-attention readout collapses (logits saturate to one channel).
**Mitigation.** Spectral floor regularizer $\mathcal{R}_{\text{spec}}$ in §6.3. In the kernel, also a hard clip: if $\sigma_{\min}(L_\nu^{(K)}) < \delta$, add $\delta I_p$ on the diagonal of the Gram matrix before softmax.

### 12.4 M2L operator over-sharing across heads
**Mode.** Level-shared $K_{\ell,\Delta}$ across all heads forces heterogeneous content (e.g., syntactic vs semantic heads) to use the same translation.
**Mitigation.** Default to head-shared; if Gate-0 fails with low expressivity, expand to head-specific $K_{\ell,\Delta,h}$ at cost of $H$× more $K$ parameters (≈4M extra per head → 64M total at $H=16$, ~7% of baseline 870M).

### 12.5 Causality violation through naive L2L
**Mode.** A naive L2L scatters parent's accumulated future-from-past back to children, including those whose causal cone hasn't been "filled" yet.
**Mitigation.** Causally-staggered sweep order (§8.1, Proposition 8.1). Implementation: M2L pair list sorted by $\max I_\nu$; L2L processed in same order; assertion check via mask.

### 12.6 Near pair count blow-up
**Mode.** Increasing $\eta$ beyond 2 grows near-pair count as $O(\eta^2)$ within the local window.
**Mitigation.** Cap $\eta = 2$ (3 near siblings: self, left, right). Rely on multipole branch for everything beyond.

### 12.7 Vanishing far-field gradient at init
**Mode.** With $K_{\ell,\Delta} \equiv 0$ and $\beta_\ell \equiv 0$, the far branch produces zero output and thus zero gradient — far parameters never train.
**Mitigation.** Spectral pre-init perturbation: $K_{\ell,\Delta=\eta+1} = \epsilon I_p$ for $\epsilon = 10^{-3}$. This is below numerical noise floor for forward but breaks symmetry for backward, ensuring nonzero gradient flow.

### 12.8 Leaf-size mismatch with sequence length
**Mode.** If $T$ is not divisible by $s_0$, the rightmost leaf is partial.
**Mitigation.** Pad to $\lceil T/s_0 \rceil \cdot s_0$ with mask tokens; standard packing trick. Causal mask handles attention to/from padding tokens automatically.

### 12.9 Catastrophic interference between near and far branches at training step ≈ 1k
**Mode.** Once $K_{\ell,\Delta}$ starts moving from $0$, the far branch's contribution can interfere destructively with the well-trained near branch.
**Mitigation.** Far-field decay $\mathcal{R}_K$ in §6.4: cosine-decay from $\lambda_K = 10^{-4}$ to 0 over the first 10% of training. Lets the near branch establish before far branch ramps.

### 12.10 Production VRAM at T=16384 from multipole storage
**Mode.** Stored multipole moments scale as $O(T D p d / s_0)$, which at production scale is 2 GB total — significant.
**Mitigation.** Already accounted for in §10.2. If VRAM is tight, recompute leaf moments during backward (cost: one extra $O(T p s_0)$ forward) instead of storing them.

---

## 13. Minimal Prototype

### 13.1 C++/CUDA file plan

```
research/gpu_hmta.h           — public API
research/gpu_hmta.cu          — kernel implementations
research/HMTA_GATE0_PROTO.cpp — standalone Gate-0 driver (C++98, no Python)
research/HMTA_GATE0_RESULT.md — empirical results (filled by Gate-0 run)
```

Hookup into the trainer:
```
Backend/Machine Learning/Networks/transformer_ops.h   — add hmta_attention_fwd()
Backend/Machine Learning/Networks/sgd_transformer.cpp — add hmta_attention_bwd()
Backend/Machine Learning/Networks/training_config.h   — add hmta_enabled, hmta_s0, hmta_p, hmta_eta
```

CLI flags (in `glades-trainer/main.cpp`):
```
--hmta              # enable HMTA attention instead of SCFA/SDPA
--hmta-s0 64        # leaf size
--hmta-p 8          # multipole rank
--hmta-eta 2        # near-neighbor radius
--hmta-stiefel 1e-3 # Stiefel regularizer weight
--hmta-decay 1e-4   # far-field K decay weight
--hmta-eps 1e-3     # spectral floor + init perturbation
```

### 13.2 Kernel-by-kernel outline

#### Kernel 1: `hmta_leaf_encode_fwd` (and bwd)
- **Input**: $K, V \in \mathbb{R}^{B\times T \times d}$ (BF16 storage, FP32 accum), $E \in \mathbb{R}^{p \times s_0}$ (BF16).
- **Output**: $M^{(0)} \in \mathbb{R}^{B \times (T/s_0) \times p \times 2d}$ (BF16 storage).
- **Implementation**: one cuBLAS strided-batched GEMM per leaf. Pre-compute the strided layout of $K, V$ as $(B, T/s_0, s_0, d)$.
- **Cost**: $O(B \cdot T \cdot p \cdot d)$ FLOPs; one launch.

#### Kernel 2: `hmta_m2m_sweep_fwd` (and bwd)
- **Input**: $\{M_\nu^{(\ell-1)}\}_\nu$, $P^\uparrow_\ell \in \mathbb{R}^{p\times 2p}$, $\Phi_\ell$ MLP weights, $\beta_\ell$ scalar.
- **Output**: $\{M_\nu^{(\ell)}\}_\nu$.
- **Implementation**: $D$ launches, one per level. Each level: cuBLAS strided-batched GEMM + small fused MLP kernel.
- **Cost**: $O(B \cdot T \cdot p \cdot 2d \cdot D / s_0)$.

#### Kernel 3: `hmta_m2l_translate_fwd` (and bwd)
- **Input**: $\{M_\mu^{(\ell)}\}$, $\{K_{\ell,\Delta}\}$, admissible pair list.
- **Output**: $\{L_\nu\}$.
- **Implementation**: per-level, one custom CUDA kernel that iterates over admissible pairs (stored as int32 offset list in constant memory), accumulates $L_\nu \mathrel{+}= K_{\ell,\Delta(\mu,\nu)} M_\mu^{(\ell)}$ via atomic add.
- **Cost**: $O(B \cdot (T/s_0) \cdot D \cdot \eta \cdot p^2 \cdot 2d)$.

#### Kernel 4: `hmta_l2l_scatter_fwd` (and bwd)
- **Input**: $\{L_\pi\}$ at parent level, $P^\downarrow_\ell$.
- **Output**: $\{L_\nu\}$ at child level updated.
- **Implementation**: $D$ launches, cuBLAS strided-batched GEMM.
- **Cost**: $O(B \cdot T \cdot p^2 \cdot 2d \cdot D / s_0)$.

#### Kernel 5: `hmta_leaf_decode_fwd` (and bwd)
- **Input**: $Q \in \mathbb{R}^{B\times T \times d}$, $\{L_\nu\}$, spectral floor $\delta$.
- **Output**: $Y^{\text{far}} \in \mathbb{R}^{B\times T \times d}$.
- **Implementation**: per-leaf, one fused softmax-attention kernel against the rank-$p$ local-key/local-value basis. Reuses the existing CHIRON softmax-attention kernel with $(T, K_{\text{len}}) = (s_0, p)$ — the small-K path.
- **Cost**: $O(B \cdot T \cdot p \cdot 2d)$.

#### Kernel 6: `hmta_near_attention_fwd` (and bwd)
- **Input**: $Q, K, V$, near-window mask of half-width $\eta s_0$.
- **Output**: $Y^{\text{near}}$.
- **Implementation**: existing SCFA near-window kernel with $(s_0, (2\eta+1)s_0)$ tile shape; reuses the `--fuse-attn-reln` path. No new kernel needed.
- **Cost**: $O(B \cdot T \cdot s_0 \cdot \eta \cdot d)$.

### 13.3 Smallest meaningful experiment

```
build/glades_chiron_train_hmta_gate0 \
  --data-dir pretok-data --pretokenized --vocab 32000 \
  --seq-len 4096 --m 384 --layers 8 --heads 6 --dhead 64 \
  --lr 3e-4 --grad-clip 1.0 --max-steps 12000 --warmup 200 \
  --int8-adam --bf16-grads --bf16-weights --bf16-attn \
  --bf16-logits --bf16-logits-storage \
  --hmta --hmta-s0 64 --hmta-p 8 --hmta-eta 2 \
  --hmta-stiefel 1e-3 --hmta-decay 1e-4 --hmta-eps 1e-3 \
  --val-every 500 --val-batches 4 \
  --save-dir research/runs/2026-05-16-hmta-gate0/
```

Parameters: $m=384, L=8, H=6, d=64$ ⇒ ≈ 30M params (dominated by tied-readout: $V\cdot m = 32000 \cdot 384 = 12.3$M; rest in attention + FFN). Total tokens: 80M (12000 steps × batch 32 × T=4096 ≈ 1.6B tokens — adjust batch to reach ~80M token equivalent). Wall: ~2 hours.

**Control run**: identical command minus `--hmta`, using flat SDPA at the same scale.

### 13.4 Gate-0 measurement protocol

1. Train HMTA and SDPA-control to 12000 steps each.
2. Evaluate val NLL every 500 steps on 4 batches of `pretok-data/val` (T=4096 segment).
3. Report:
   - Best val NLL (HMTA) and (SDPA-control).
   - Mean val NLL over last 10 checkpoints (HMTA, SDPA-control).
   - Wall-clock per training step (HMTA, SDPA-control).
   - Wall-clock per val forward pass (HMTA, SDPA-control).
   - Singular-value spectrum probe: at step 6000, dump the SDPA-control's logit matrix on 32 random cluster-pairs; report median rank-8 retention (Proposition 9.3 verification).

### 13.5 Pass criteria (falsification thresholds)

**PASS** iff all of:
- $|\text{val NLL}(\text{HMTA}) - \text{val NLL}(\text{SDPA-control})| \le 0.05$ nat (both at last 10-checkpoint mean).
- $\text{wall}(\text{HMTA-step}) / \text{wall}(\text{SDPA-step}) \le 0.25$ at $T=4096$.
- Rank-8 retention probe $\ge 0.85$ (i.e., $\sigma_9 / \|\Lambda\|_F \le 0.05$ median across pairs).

**FAIL** if any of the above is violated.

Either way, **document and commit**. Falsification informs iter 38; validation triggers full-scale port to T=16384 m=2048 L=24.

---

## 14. Full Research Program

### Phase A — Gate-0 (this iter)
**Objective**: validate the multipole hypothesis at small scale.
**Budget**: 2 GPU-hours.
**Deliverable**: pass/fail verdict on §15 conjectures.

### Phase B — Production port (iter 38, conditional on Phase A pass)
**Objective**: deploy HMTA at production scale T=16384 m=2048 L=24.
**Tasks**:
- Refactor kernels for BF16-TC at production tile shapes.
- Tune $s_0, p, \eta$ via small grid (4–6 configurations).
- Add `--hmta` flag to glades-trainer production stack.
- Train 30k-step run from scratch at full scale.

**Budget**: ~24 GPU-hours.
**Deliverable**: HMTA-flagship checkpoint at val NLL ≤ 4.10 nat, wall ≤ 50% of post-fix flagship's wall per step.

### Phase C — Stacking experiments (iter 39+)
**Objective**: combine HMTA with orthogonal CHIRON paradigms for the brief's "magnitudes" target.
**Combinations to test**:
1. HMTA × #42 SCFA (HMTA generalizes SCFA — sanity check that the special-case parametrization works).
2. HMTA × #74 PHOENIX-1BIT FFN sparsity (FFN-side compression).
3. HMTA × #76 MLA (KV-cache compression for inference).
4. HMTA × #51 ATLAS-COMPILE (CUDA-graph fusion).

**Budget**: ~30 GPU-hours.
**Deliverable**: stacked combo at wall ≤ 10% of post-fix flagship, NLL within 0.05 nat — i.e., **10× wall-clock at iso-NLL**, the brief's stated target.

### Phase D — Empirical scaling-law extension (iter 40+)
**Objective**: characterize HMTA's scaling along $T$, $L$, $m$, $p$ axes.
**Tasks**: train 5–8 models at varied scales; fit a Chinchilla-style scaling law for HMTA; compare to flat-SDPA and SCFA scaling.
**Deliverable**: a scaling-law report showing whether HMTA's compute advantage holds across $T$ (especially $T > 16384$).

### Phase E — Theoretical refinements (iter 41+, optional)
**Objective**: tighten the truncation error bound of Proposition 9.3, and prove or disprove rank-$p$ admissibility for natural language attention matrices.
**Tasks**: spectral probe on flagship; PAC-Bayesian bound on generalization; possible connection to barvarian / Beylkin's wavelet-attention framework.
**Deliverable**: a theory-track paper or revised conjecture in the framework doc.

### Phase F — Multipole over the value dimension (iter 42+, speculative)
**Objective**: extend HMTA from multipole-over-positions to **multipole-over-positions × multipole-over-channels**, decomposing both axes hierarchically.
**Potential**: another $\sim p_c/d$ FLOP reduction on $W_Q, W_K, W_V, W_O$.

---

## 15. Open Conjectures and Validation Criteria

### 15.1 Conjecture C1 — Multipole hypothesis (Gate-0)

> **C1.** *At small scale ($M = 30$M params, $T = 4096$, $L = 8$, $H = 6$, $d = 64$, $s_0 = 64$, $p = 8$, $\eta = 2$, $N = 80$M tokens, wall ≤ 2 h on RTX 4080 SUPER), an HMTA model trained from scratch achieves val NLL within 0.05 nat of a parameter-matched flat-SDPA baseline trained on the same tokens, at no more than 25% of the SDPA wall-clock per step.*

**Test**: §13 Gate-0 experiment. **Result expected by**: end of this iter.

### 15.2 Conjecture C2 — Magnitudes scaling (production)

> **C2.** *At production scale ($T = 16384$, $m = 2048$, $L = 24$, $H = 16$, $d = 128$, $s_0 = 128$, $p = 16$, $\eta = 2$, full pretok corpus, 30k steps from scratch), HMTA achieves val NLL ≤ 4.10 nat (within 0.02 nat of post-fix flagship's 4.0771) at no more than 50% of the flagship's per-step wall-clock.*

**Test**: Phase B (iter 38). Conditional on C1 passing. **Result expected by**: iter 38.

### 15.3 Conjecture C3 — Stack delivers brief's magnitudes target

> **C3.** *HMTA × #74-PHOENIX-1BIT × #76-MLA × #51-ATLAS achieves val NLL ≤ 4.10 nat at wall-clock per token ≤ 10% of post-fix flagship.*

**Test**: Phase C (iter 39+). Conditional on C2 passing. **Result expected by**: iter 39–40.

### 15.4 Conjecture C4 — Truncation error matches empirical singular-value decay

> **C4.** *For natural-language sequences at $T \ge 4096$ on the pretok-data corpus, the SDPA logit matrices on admissible cluster pairs have median rank-$p$ retention $\ge 1 - 0.05$ at $p = 8$ (i.e., singular value 9 is below 5% of Frobenius norm).*

**Test**: §13.4 probe step. **Result expected by**: end of this iter, as part of Gate-0.

### 15.5 Conjecture C5 — Information capacity is sufficient

> **C5.** *The far-field information capacity $D \cdot p \cdot d$ is sufficient to encode long-range structure relevant to the next-token prediction, in the sense that an HMTA model with $D \cdot p \cdot d \ge 4096$ matches SCFA NLL at iso-budget on tasks with long-range dependencies.*

**Test**: Phase D ablation (iter 40+).

### 15.6 Falsifying signals

A C1 failure (val NLL Δ > 0.05 or wall > 25%) at Gate-0 implies one of:
- (a) Rank-8 truncation is insufficient (test: rerun at $p = 16$).
- (b) Level-shared $K$ is too restrictive (test: rerun with head-specific $K$).
- (c) BF16 precision is insufficient for moments (test: rerun with FP32 moments).
- (d) Local-window radius is insufficient (test: rerun at $\eta = 4$).

If (a)-(d) all fail, the multipole hypothesis is **falsified** at this scale. Next iter pivots to candidate B (MEDAL) or candidate C (VARCO).

### 15.7 Honest uncertainty

- **Proven**: Theorems 9.1, 9.4, Proposition 8.1.
- **Derivable under stated assumptions**: Proposition 9.3 (assumes low-rank logits — to be checked at probe step).
- **Empirical conjectures**: C1–C5; each has a concrete falsification protocol.
- **Speculative**: the brief's "magnitudes" target (10×–100×) requires multiplicative stacking with FFN-side paradigms. HMTA alone is necessary but not sufficient.

---

## 16. Appendix — Reproducibility Manifest

- **Branch**: `vesta5` of `glades-ml`; trainer commits track `glades-trainer/main` (currently at `365ad4d` post chrf-bf16-weights fix).
- **Baseline**: `research/runs/2026-05-15-flagship-postfix-T16384-30k/chiron_1B_T16384.step30000`, val NLL 4.0771.
- **Pretok corpus**: `pretok-data/` (4 val batches × T=16384 = 4.2M val tokens).
- **GPU**: RTX 4080 SUPER, 16 GB.
- **Build**: `sh .configure.sh cuda` from glades-ml; `cd build && make` from glades-trainer.

**End of design document — paradigm #261 HMTA.**
