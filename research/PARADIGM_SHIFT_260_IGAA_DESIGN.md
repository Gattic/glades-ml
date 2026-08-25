# Paradigm Shift #260 — IGAA: Information-Geometric Attention Augmentation

**Iter 23 of "Focused attention with perspective"** — first iteration after the iter-22 empirical falsification of the cellular-sheaf family (paradigms #250-#255). Designed against the **properly-saved** post-fix CHIRON-1B flagship at val NLL 4.0771 nat.

## 1. Executive summary

IGAA augments a trained baseline attention layer with a **per-token mixture over K structurally distinct attention modes**, aggregated in natural-parameter space (Euclidean convex combination on an exponential family) and additively injected through a triply-gated zero-at-init residual tap.

The mechanism cleanly factors the brief's two named ingredients:

- **Perspective** = the discrete catalogue of K modes M = {m_1, ..., m_K}, each a deterministic attention operator with different effective receptive width, similarity kernel, or positional decay.
- **Focus** = the per-token simplex weights π_i ∈ Δ^{K-1}, produced by a softmax over learned per-token logits. Concentration (low entropy) on one mode = focused attention; uniform mass = no-op.

The full layer-ℓ update is the additive correction

```
y'_ℓ = y^{SCFA}_ℓ + α_ℓ · (Σ_{k=1..K} π_{i,k}^{(ℓ)} · θ^{(k,ℓ)}_i − (1/K) Σ_k θ^{(k,ℓ)}_i)
```

with α_ℓ = 0, W_π^{(ℓ)} = 0, and τ^{(ℓ)} = 1 at initialization. Both gates independently zero the correction.

**Honest empirical conjecture (Gate-0).** A 1-layer, K=4 IGAA inserted at layer 18 of the post-fix flagship, with 500 SFT-style steps training only (W_π, b_π, α, τ) at lr=1e-4 and grad-clip=0.05, achieves **val NLL ≤ 4.05 nat** (improvement ≥ 0.03 over 4.0771) at any single checkpoint, with ≥ 90% of val measurements in [3.95, 4.10]. Gate-0 PASSES if both conditions hold simultaneously and FAILS otherwise.

**Magnitudes question.** Honestly: improbable. The mechanism's expressivity is bounded by conv(M), the K-mode convex hull. Any token's representable attention is a mixture of K fixed patterns. Magnitudes-level improvement (>2.3 nat) would require K-mode aggregation to dominate the trained-from-scratch baseline by a large factor, which is not what architecture changes typically deliver. Honest target: 0.05–0.30 nat val improvement. The "magnitudes" target remains an open question that this iteration cannot resolve.

## 2. Candidate formulations (compact)

Three formulations were developed in parallel; only the selected one is detailed in §5+.

**Candidate A — COMET** (operator-theoretic): a learned residual correction operator built from 4 structured subalgebras (DCT-circulant, banded Toeplitz, diagonal content gate, low-rank similarity), mixed by a sparsemax-over-lenses gate. ~110M new parameters total. Strongest at containing SCFA as a limiting case via explicit operator-family inclusion.

**Candidate B — IGAA** (information-geometric, **selected**): per-token mixture over K fixed attention modes, aggregated as α-barycenter in natural parameter space, additively injected via triply-gated correction. ~800k new parameters.

**Candidate C — SHADO** (dynamical / symplectic): augmenting tokens with a learned momentum p_i and evolving (q,p) via Hamiltonian leapfrog flow with K subspace projectors. Symplecticity gives volume preservation across layers. ~1.7M new parameters.

## 3. Framework selection rationale

See the inline selection table above. Summary tradeoffs:

- **A** is most expressive on paper (multi-layer Toeplitz+circulant generates dense operators) but has the largest parameter footprint and the most kernel work to implement.
- **C** has the strongest stability theorem (Liouville) but introduces a per-token momentum vector that — even with low-rank lift — adds optimization-state state and the leapfrog backward chain is unfamiliar territory in this codebase.
- **B** has the smallest empirical risk profile (least parameters, triple-redundant no-op, simplest backward), the clearest mapping of "perspective" and "focus" to discrete K modes and a simplex distribution respectively, and the cleanest Gate-0 prototype: it can reuse existing SCFA infrastructure as one of the K modes, and the new code is only (a) K-batched mode evaluation and (b) a per-token softmax mixture.

The selection is conservative on purpose. Iter 22 demonstrated that the failure mode of "architecturally interesting but empirically destabilizing" is real and expensive. Picking the candidate with the smallest perturbation surface — that still admits the required ingredients — gives Gate-0 the best chance of a clean signal.

## 4. Formal problem statement

Let f_θ : V^T → ℝ^{T×V} be the post-fix CHIRON-1B flagship (parameters θ ∈ ℝ^{870M}, vocab V=32000, sequence length T=16384, residual width m=2048, layer count L=24, attention path = SCFA paradigm #42 with k=1024 compression and w=8 per-layer convolution).

The empirically measured baseline is

```
NLL_val(θ) = 4.0771 nat on pretok-data/val (4 batches × T=16384 = 4.2M tokens)
```

We seek **augmentation parameters** ψ ∈ Ψ_IGAA (defined in §5) such that the augmented model f_{θ,ψ} satisfies:

- **(C1) no-op preservation**: ψ has a canonical zero point ψ* with f_{θ,ψ*} = f_θ exactly (bit-exact bf16 round-trip).
- **(C2) explicit forward**: f_{θ,ψ} is computable as a finite acyclic graph of bounded-norm linear-algebra primitives.
- **(C3) bounded backward**: ‖∂f_{θ,ψ} / ∂ψ‖_op is bounded for ψ in a neighborhood of ψ*.
- **(C4) stackability**: applying IGAA at multiple layers gives output norm bounded by Σ_ℓ |α_ℓ| · O(‖x‖).
- **(C5) limiting cases**: ψ admits a state that recovers SCFA exactly (K=1 mode chosen as SCFA) and approximately recovers SDPA (one mode chosen as full softmax attention, π one-hot on it).

Subject to C1–C5, we want a ψ that minimizes

```
NLL_val(θ, ψ) = E_{(x,y) ∈ D_val} [-log p_{θ,ψ}(y | x)]
```

with the **honest minimum target**: NLL_val(θ, ψ*) − NLL_val(θ, ψ̂) ≥ 0.30 nat on the same val split for the SFT-trained ψ̂ after stable training, while never violating the C1–C5 constraints during training.

## 5. Core mathematical framework

### 5.1 Per-token mixture state

For each token i ∈ {1, ..., T} at each layer ℓ ∈ {1, ..., L_IGAA} ⊂ {1, ..., L}, we attach a categorical distribution

```
π_i^{(ℓ)} ∈ Δ^{K-1} = {π ∈ ℝ^K : π_k ≥ 0, Σ_k π_k = 1}
```

over K modes. K is shared across all tokens and layers. Production target K=16; Gate-0 prototype K=4.

The simplex Δ^{K-1} carries the **Fisher–Rao metric** g_{ij}(π) = δ_{ij}/π_i + 1/π_K (in interior coordinates), or equivalently the pullback of the round metric on S^{K-1} under √π. We work primarily in the **e-coordinates** (logit space) because softmax-from-logits is the e-affine chart and the e-flat structure makes natural-parameter aggregation trivially Euclidean.

### 5.2 Perspective modes

Fix a catalogue M = {m_1, ..., m_K} where each m_k is a deterministic, parameterized attention operator

```
A_k : ℝ^{T×m} → ℝ^{T×m},   A_k(X) = Attn(X; κ_k, w_k, s_k, φ_k)
```

with mode-k hyperparameters: κ_k (similarity kernel — dot-product, exp-decay, ReLU(cos)²), w_k (effective receptive width), s_k (temperature), φ_k (positional bias profile — ALiBi slope or none).

**Gate-0 instantiation (K=4)** — chosen for cheap, diverse, and SCFA-orthogonal coverage:

| k | mode name | kernel | effective width | implementation |
|---|-----------|--------|-----------------|----------------|
| 1 | SCFA-full | DCT-II + per-layer conv | T (full) | reuse existing SCFA at layer ℓ |
| 2 | SCFA-tight | DCT-II at k'=256 | T (full but low rank) | SCFA recompiled with reduced k |
| 3 | banded-short | dot-product softmax | 64 tokens | new banded SDPA kernel |
| 4 | identity | x ↦ x (no attention) | self only | trivial |

Modes are **fixed at instantiation** (no learnable parameters in the modes themselves — all learning is in the mixture). This is the "shared structure" constraint #3 from §4.

**Production instantiation (K=16)** — adds:

| k | mode name | comment |
|---|-----------|---------|
| 5 | banded-medium | w=256 |
| 6 | banded-long | w=4096 |
| 7 | ALiBi-shallow | slope=0.05 |
| 8 | ALiBi-medium | slope=0.10 |
| 9 | ALiBi-steep | slope=0.30 |
| 10 | self-only-decay | exp(-|i-j|/τ_10) × identity |
| 11 | retroactive | only attend backward in window |
| 12-16 | learned-low-rank | rank-r adaptive heads |

Modes 12-16 have a few hundred trainable parameters each (slow-rate group); modes 1-11 are entirely deterministic.

### 5.3 Mode outputs as natural parameters

The K mode outputs θ_i^{(k, ℓ)} := A_k(X^{(ℓ)})_i ∈ ℝ^m are interpreted as the **natural parameter** of an isotropic Gaussian on the residual stream: p_i^{(k,ℓ)}(y) ∝ exp(θ_i^{(k,ℓ)} · y − ½‖y‖²). Under this identification the mode-output manifold is e-flat (Amari, *Information Geometry and Its Applications*, §3.7).

### 5.4 Mixture logits

For each layer ℓ where IGAA is active, we have a small linear head W_π^{(ℓ)} ∈ ℝ^{K×m}, bias b_π^{(ℓ)} ∈ ℝ^K, and a layer-local temperature τ^{(ℓ)} ∈ ℝ_+. The per-token mixture is

```
z_i^{(ℓ)} = W_π^{(ℓ)} x_i^{(ℓ)} + b_π^{(ℓ)}                 ∈ ℝ^K
π_i^{(ℓ)} = softmax(z_i^{(ℓ)} / τ^{(ℓ)})                   ∈ Δ^{K-1}
```

**Initialization.** W_π^{(ℓ)} = 0, b_π^{(ℓ)} = 0, τ^{(ℓ)} = 1, so π_i^{(ℓ)} = (1/K, ..., 1/K) for all tokens.

### 5.5 Aggregation in natural-parameter space

The α-barycenter of {θ_i^{(k)}} with weights π_i is, in natural coordinates of an e-flat exponential family,

```
θ̄_i^{(ℓ)} = Σ_{k=1..K} π_{i,k}^{(ℓ)} · θ_i^{(k, ℓ)}        ∈ ℝ^m
```

independent of the α-connection (the natural-parameter manifold is e-affine for any α). This is the **explicit forward** — a per-token convex combination with no implicit solve. Backward is the chain of softmax and linear-mixture Jacobians, both bounded (§9).

### 5.6 Centered correction and gated injection

Define the **uniform-mixture reference**

```
θ̄_i^{unif, (ℓ)} = (1/K) Σ_k θ_i^{(k, ℓ)}
```

and the **centered correction**

```
Δy_i^{(ℓ)} = θ̄_i^{(ℓ)} − θ̄_i^{unif, (ℓ)} = Σ_k (π_{i,k}^{(ℓ)} − 1/K) · θ_i^{(k, ℓ)}
```

The full augmented layer output is

```
y_i^{IGAA, (ℓ)} = y_i^{SCFA, (ℓ)} + α_ℓ · Δy_i^{(ℓ)}                       (★)
```

with α_ℓ ∈ ℝ a layer-local scalar gate, α_ℓ = 0 at initialization.

**No-op verification.** At ψ* := (W_π = 0, b_π = 0, τ = 1, α = 0):
1. π = uniform → Δy ≡ 0 (mean subtraction zeroes the centered weights).
2. α = 0 → the addition contributes nothing.

Either condition alone gives bit-exact recovery of f_θ. The conjunction is robust to a single bf16 rounding error.

### 5.7 Recovery of baselines

**SCFA recovery (exact)**: K=1 forces Δ^0 = {1}, π trivially uniform, mean-subtraction yields 0, IGAA(layer) ≡ SCFA(layer) for any α.

**SDPA recovery (exact in limit)**: set mode 1 = full SDPA. Construct W_π^{(ℓ)} so that z_{i,1} → +∞ and z_{i,k} → −∞ for k ≠ 1 across all tokens (achievable by W_π_1 = c · w, b_π = c · e_1 with c → ∞). Then π_i → e_1 uniformly, θ̄_i → θ_i^{(1)}, and the correction Δy_i → θ_i^{(1)} − (1/K) Σ_k θ_i^{(k)}. With α_ℓ = 1 and discarding the constant offset (1/K) Σ_k θ_i^{(k)} (which is absorbed by SCFA's residual norm), the model output approaches SDPA(X)_i.

**SCFA paradigm #42 augmentation (the production target)**: mode 1 = SCFA(k=1024, w=8), mode 2 = SCFA(k=256, w=8), other modes structurally distinct. IGAA expresses any per-token convex mixture of K SCFA-like patterns and selectively chooses among them via π_i.

## 6. Objective function derivation

Let θ denote the frozen baseline parameters, ψ = (W_π^{(ℓ)}, b_π^{(ℓ)}, τ^{(ℓ)}, α^{(ℓ)})_{ℓ ∈ L_IGAA} the IGAA parameters. The training objective is

```
L_IGAA(ψ) = L_LM(θ, ψ) + β_foc · R_foc(Π) + β_drift · R_drift(Π) + β_decay · ‖α‖²
```

where:

**LM term.**
```
L_LM(θ, ψ) = -(1/N) Σ_n log p_{θ,ψ}(y_n | x_{<n})
```
standard next-token NLL on pretok-data/train.

**Focus regularizer.** Tsallis-2 entropy weighted by a learned per-token confidence gate:
```
R_foc(Π) = (1/(T · L_IGAA)) Σ_{ℓ, i} c_i^{(ℓ)} · S_2(π_i^{(ℓ)})
S_2(π) = 1 − Σ_k π_k²
c_i^{(ℓ)} = σ(u^{(ℓ)} · x_i^{(ℓ)})  ∈ (0, 1)
```
High-confidence tokens (c large) pay a penalty proportional to π's entropy → π concentrates. Low-confidence tokens are free to remain near-uniform. u^{(ℓ)} ∈ ℝ^m is a learned confidence direction.

**Cross-layer drift regularizer.** Symmetric KL between consecutive layers' π:
```
R_drift(Π) = (1/(T · (L_IGAA − 1))) Σ_{i, ℓ ≥ 2} ½[KL(π_i^{(ℓ)} ‖ π_i^{(ℓ-1)}) + KL(π_i^{(ℓ-1)} ‖ π_i^{(ℓ)})]
```
Penalizes abrupt depth-wise perspective shifts. Critical for stackability.

**Gate decay.** ‖α‖² = Σ_ℓ α_ℓ² keeps gates small, satisfying §5.7's stackability bound.

**Hyperparameters (Gate-0).** β_foc = 0.01, β_drift = 0.01, β_decay = 0.001. These are conservative; tuning is part of the post-Gate-0 program.

**Natural gradient on π.** The softmax-from-logits parameterization gives, for any scalar loss L depending on π via π = softmax(z),
```
∂L/∂z_k = π_k · (∂L/∂π_k − E_π[∂L/∂π])
```
which is exactly the **policy-gradient-with-baseline** form — also the **Fisher–Rao natural gradient** on Δ^{K-1} in the e-chart, up to scaling. So Adam on z is, modulo step-size effects, a natural-gradient method on π. No extra machinery needed.

## 7. Optimization algorithm

**Setup.**
- Freeze baseline θ.
- Initialize ψ at ψ* (canonical zero).
- Use AdamW with β1=0.9, β2=0.999, eps=1e-8, weight_decay=0.001.
- LR: 1e-4 for (W_π, b_π, u), 1e-5 for τ, 1e-3 for α. (Gates learn fastest because they are the activation switch; τ moves slowly to avoid oscillation.)
- Gradient clip: global L2 norm 1.0 (much looser than SFA's 0.05 because IGAA's Jacobians are bounded by softmax+linear, not by an implicit-diff solve).
- Batch: 1 sequence at T=16384 per step (effective batch = 16384 tokens, matching SCFA training).
- Schedule: linear warmup 50 steps from 0 to peak LR, then constant.

**Per-step.**

```
1. forward:
     for each IGAA layer ℓ in {18} (Gate-0) or {18, 12, 6} (Gate-1):
         compute K mode outputs θ^{(k,ℓ)} via mode-specific attention ops
         compute mixture logits z = W_π x + b_π, π = softmax(z/τ)
         compute Δy = Σ_k (π_k − 1/K) θ^{(k)}
         inject α · Δy additively into y^{SCFA}
2. loss = LM_NLL + β_foc · R_foc + β_drift · R_drift + β_decay · ‖α‖²
3. backward:
     gradients flow through softmax, linear mixture, mode outputs (mode outputs use the existing SCFA backward when mode == SCFA)
4. AdamW step.
```

**Val cadence.** Every 50 steps, evaluate val NLL on 4 batches × T=16384.

**Early stopping (Gate-0).** Stop if val NLL exceeds 4.10 nat at any val checkpoint after step 100. Stop if val NLL drops below 4.05 nat for 3 consecutive checkpoints (PASS).

## 8. Temporal dynamics formulation

The "temporal" axis here is **depth ℓ** (not autoregressive time). The system is a sequence of L_IGAA per-token distributions π_i^{(1)}, ..., π_i^{(L_IGAA)} on Δ^{K-1}.

**Continuous-time view.** As L_IGAA → ∞ with α_ℓ → 0 at rate 1/L_IGAA, the depth-wise dynamics on π_i has the form

```
∂π_i / ∂ℓ = − Σ_k (∂L_LM / ∂z_{i,k}) · (J_softmax)_{i,k}    + (regularizer gradients)
```

where J_softmax = diag(π) − π π^⊤ is the softmax Jacobian. In the high-confidence limit (c large), R_foc adds a drift toward simplex vertices; R_drift adds a depth-smoothing term that resembles graph diffusion on the simplex.

This is **not** the same as a "Hamiltonian flow" (candidate C) — it's a **gradient flow** in the (depth × token) bundle Δ^{K-1} × T × L_IGAA. The dynamics is **dissipative** (gradient flow contracts under the regularizers), which gives:

**Proposition (informal).** Under R_drift > 0, the trajectory ℓ ↦ π_i^{(ℓ)} is Lipschitz in ℓ with constant O(1/√β_drift). Hence cumulative cross-layer drift is bounded.

This bounds the per-token attention pattern from changing too rapidly with depth — analogous to how SCFA's per-layer kernel can't be too different from the previous layer's. A direct consequence: IGAA cannot abruptly switch from one mode to another between consecutive layers, which would otherwise risk a similar damage-compounding effect to SFA's multi-layer instability.

## 9. Theoretical analysis

### 9.1 Well-posedness

The forward map is the composition of: linear projection W_π, softmax (smooth on ℝ^K), K-batched attention (smooth in X if each mode is smooth), affine convex combination, and addition. Each is C^∞ where defined. The whole composition is differentiable and bounded for bounded inputs.

### 9.2 Invariances

- **Permutation of modes.** Swapping (k, k′) in M corresponds to permuting columns of W_π and entries of π. The output is invariant. So Ψ_IGAA has a built-in S_K symmetry that doesn't affect any observable.
- **Temperature/scale duality.** Doubling W_π and halving τ produces the same π. Hence (W_π, τ) is identifiable only up to scalar gauge. Practically resolved by initializing τ=1 and letting W_π absorb scale freely.

### 9.3 Conditioning

The Jacobian of the IGAA correction wrt ψ at ψ* (where α=0):

```
∂Δy_i / ∂α_ℓ = Δy_i = Σ_k (π_{i,k} − 1/K) θ_i^{(k)}
```

At ψ*, π = uniform → Δy_i = 0 → ∂Δy/∂α_ℓ = 0. So **α has zero gradient at strict init** — symmetry-breaking is needed.

Three options for breaking the symmetry without spoiling the no-op:
1. **Initialize W_π with tiny random noise** ε ~ N(0, σ² I) with σ ~ 10^{-3}. π differs from uniform by O(σ), Δy is O(σ‖θ‖), and ∂L/∂α is O(σ · ‖∂L/∂y‖) — non-zero, controllable. Val NLL at step 0 differs from baseline by O(α · σ²) = 0 (since α=0).
2. **Warmup with α = ε_α := 10^{-4}** for the first 100 steps. Same property: gradient signal exists, val perturbation is bounded.
3. **Initialize b_π non-uniform**: b_π = (b_1, ..., b_K) with the b_k chosen to encode a prior preference for mode k=1 (SCFA-full). E.g., b_1 = 1, b_{>1} = 0. This makes initial π non-uniform but still gives Δy non-zero gradient.

We adopt **option (1)**: random W_π initialization with σ = 10^{-3}. The resulting initial val NLL perturbation from the no-op baseline is below 10^{-6} nat (within bf16 noise), measurable in practice.

The full Jacobian ∂y^{IGAA} / ∂ψ has operator norm bounded by

```
‖∂y / ∂(W_π, b_π, τ, α)‖_op ≤ ‖α‖_∞ · K · max_k ‖θ^{(k)}‖_∞ · ‖J_softmax‖_op · ‖x‖_∞ + ‖Δy‖_∞
```

with ‖J_softmax‖_op ≤ max π_k ≤ 1 and ‖x‖_∞ bounded by the baseline pre-norm (~1 by RMS-normalization). So the Jacobian norm is **O(1)** uniformly in T, m, L. No exploding gradients possible.

### 9.4 Stability

**Stackability bound.** Let β_ℓ = |α_ℓ| · sup_x ‖Δy_ℓ‖ / ‖x‖. Cumulative perturbation across L_IGAA layers:

```
‖x_L − x_L^{baseline}‖ ≤ (e^{Σ β_ℓ} − 1) · ‖x_0‖
```

(same bound as candidate A, derived the same way.) Enforce Σ_ℓ β_ℓ ≤ log 2 ≈ 0.69 via the β_decay regularizer. Then the cumulative IGAA perturbation never exceeds ‖x_0‖. Combined with the Lipschitz-in-depth bound from §8, **multi-layer IGAA is unconditionally stable in a neighborhood of ψ\***.

### 9.5 Expressivity

The set of representable correction outputs at layer ℓ is

```
{Δy : Δy_i ∈ conv({θ^{(k,ℓ)}_i − mean}, k=1..K)}
```

— the K-vertex convex polytope hull. With K=4 the polytope has 4 vertices; with K=16, 16 vertices. **Expressivity scales linearly in K**.

**Upper bound (Conjecture E1)**: the optimal val NLL achievable by IGAA at K modes and L_IGAA layers is bounded below by min(NLL_SCFA, NLL_SDPA) − ε(K, L_IGAA), with ε → 0 as K → ∞ and L_IGAA → L. Unproven but plausible by approximation theory of mixtures.

**Lower bound (Conjecture E2)**: at K=4 with the Gate-0 mode choice, the achievable val NLL improvement over SCFA-only is bounded above by ~0.3 nat. (Heuristic: SDPA on small models typically beats SCFA by 0.1–0.3 nat; IGAA at K=4 cannot exceed the best of its modes plus a small mixing benefit.)

### 9.6 Generalization

The training-val gap for IGAA depends on the parameter count of ψ. With ~33k parameters per IGAA layer and a single layer at K=4, total ψ size is ~10k. This is **much smaller than the SFA failure scale** (where the SFA module added ~30M trainable parameters and overfit the training batch). By standard learning-theory rules (VC-style or PAC-Bayes), the generalization gap should scale as √(|ψ| log(1/δ) / N_train), and at N_train > 10^9 tokens this is below 0.01 nat for our parameter count.

### 9.7 Comparison to SFA failure modes

| Failure mode (SFA) | IGAA structural defense |
|---------------------|--------------------------|
| Layer replacement destroys 4-7 nat on insertion | Additive injection; α=0 → bit-exact baseline preservation |
| Implicit-diff backward ill-conditioned | Explicit softmax + linear mixture; bounded Jacobian uniformly |
| Per-token random state compounds gradients | All learnable state is shared (W_π is m × K, ~33k params); no per-token random init |
| Multi-layer compounds damage | Bounded cumulative perturbation via β_decay; depth Lipschitz via R_drift |
| Training oscillates upward | Gradient signal on α requires symmetry-breaking via small W_π noise; loss landscape is locally convex around ψ* |
| Train-val gap (~0.6 nat) | Drastically reduced parameter count (~30k vs ~30M); generalization gap should be ~0.01 nat |

## 10. Computational tradeoffs

**Memory.** K mode outputs of shape (T, m) at FP32 cost 4·T·m·K bytes. At T=16384, m=2048, K=4, FP32: 512 MB per IGAA layer; BF16: 256 MB. With L_IGAA = 1 (Gate-0), this fits comfortably in the 2 GB SCFA headroom.

For L_IGAA = 24 (full-stack production), BF16 mode-output storage is 6 GB. Strategy: **compute modes sequentially**, not in parallel, accumulating into a single (T, m) buffer. Constant 256 MB BF16 working set across all layers. This is the production VRAM budget.

**Time.** K SCFA-like forward passes per layer ≈ K× SCFA forward cost. At K=4: ~4× SCFA forward, plus negligible softmax+mixture overhead.

For Gate-0: one layer of K=4 IGAA adds 3× one-layer SCFA cost (mode 1 reuses existing SCFA). Total ~12% overhead for the whole network. Acceptable.

For K=16 production: ~15× single-layer SCFA cost at L_IGAA = 24 → roughly doubles total forward time. Acceptable for research; would need optimization for serving.

**Parameter storage.** W_π: K · m = 4 · 2048 = 8192 floats per IGAA layer. b_π: K = 4. τ: 1. α: 1. u (confidence direction): m = 2048. Total ~10.2k floats per layer. At L_IGAA = 24 production: 244k parameters. Three orders of magnitude smaller than the 870M baseline.

## 11. Comparison to existing methods

**Multi-Head Attention (Vaswani et al. 2017).** Standard MHA has H independent heads with their own (W_Q, W_K, W_V) per layer, then concatenated and projected. Each head computes a different attention pattern. IGAA differs in that:
- Modes are *fixed* (not learned Q/K/V projections); the diversity comes from structural choices (kernel, range, decay), not from gradient descent.
- The mixture weight π_i is *per-token, learned*; MHA gives uniform 1/H weight per head via concatenation.
- IGAA is *additive on top of an existing attention layer*; MHA is a *layer in its own right*.

IGAA is closer in spirit to **Mixture-of-Attentions (MoA, Peng et al. 2020)** and **Switch attention (Du et al. 2022)** which mix attention patterns. The novel contributions of IGAA over those:
1. **Information-geometric formulation**: the convex-combination aggregation is justified as a barycenter in e-flat exponential family geometry, not an ad-hoc mixture.
2. **Triple-redundant no-op-at-init**: α + W_π=0 + mean-subtraction. MoA does not preserve baseline exactly.
3. **Multi-scale via explicit fixed modes** at known receptive widths {64, 256, 1024, 4096, 16384}, with the per-token mixture choosing the effective range. This is a sharper "perspective" mechanism than learned MoA where modes drift during training.
4. **Stackability theorem** via β_decay and R_drift.

**SCFA (paradigm #42).** IGAA contains SCFA as the K=1 limit. SCFA's spectral compression provides one perspective; IGAA adds K-1 more.

**SDPA.** IGAA approaches SDPA as π collapses to the (full SDPA) mode and α=1.

**Cellular sheaf attention (paradigms #250-#255, falsified iter 22).** IGAA differs structurally: no per-token sheaf with random-init stalks, no Tikhonov solve, no implicit diff. The "perspective" mechanism is discrete (K modes) vs continuous (per-token U matrix).

**Mixture-of-Experts (MoE) on FFN.** MoE routes tokens to different FFN experts. IGAA does the same conceptually for attention. The differences: (a) IGAA modes are deterministic and structurally constrained, MoE experts are learned MLPs; (b) IGAA uses soft (simplex) routing, MoE typically uses hard top-k.

## 12. Failure modes and mitigations

(Per the subagent output, integrated and refined.)

**F1 — Mode collapse.** π_i → e_{k*} globally. *Diagnosis*: bar{S_2(π)} → 0 across the batch. *Mitigation*: load-balancing penalty Σ_k (bar π_k − 1/K)² with weight 0.001, anneal β_foc upward from 0.

**F2 — Mode merging to uniform.** π → 1/K everywhere → Δy → 0 (no-op). *Diagnosis*: |α| stops moving. *Mechanism*: this is the trivial trap. *Mitigation*: option (1) from §9.3 — initialize W_π with σ=10^{-3} noise to break symmetry; if still stuck, add small entropy regularizer at the START of training and remove it after 200 steps.

**F3 — Redundancy with SCFA.** Modes that resemble SCFA contribute nothing (gate α stays 0). *Diagnosis*: monitor correlation(Δy, y^{SCFA}). *Mitigation*: structurally exclude SCFA's k=1024 spectral window from any mode k ≥ 2; use modes 3, 4 = banded/identity which are SCFA-orthogonal.

**F4 — Cross-layer drift instability.** Different ℓ produces wildly different π. *Diagnosis*: R_drift grows linearly in ℓ. *Mitigation*: the drift regularizer is the primary fix; secondary safeguard is to share W_π^{(ℓ)} ≡ W_π globally (single matrix, ~8k parameters total), enforcing depth-wise consistency.

**F5 — Symmetry-breaking failure (gradient stuck at zero).** Per §9.3, α has zero gradient at strict init. *Mitigation*: σ=10^{-3} W_π noise. Validate by checking ‖∂L/∂α‖ > 10^{-8} at step 0.

**F6 — Mode degeneracy via large logits.** ‖z‖ → ∞ if one mode dominates by unbounded margin. *Mitigation*: clip ‖z‖∞ to 10; pre-norm x before W_π.

**F7 — VRAM overrun at K=16, L_IGAA=24.** Per §10, parallel mode storage is 6 GB. *Mitigation*: sequential mode computation with single 256 MB BF16 buffer.

**F8 — Magnitude target miss (likely).** Per §1, magnitudes-level improvement is implausible. IGAA's expressivity is bounded by conv(M). *Mitigation*: be honest in evaluation — accept 0.05–0.30 nat as the real target, not "magnitudes".

## 13. Minimal prototype (Gate-0)

### 13.1 Scope

Single IGAA layer at layer 18, K=4 modes, frozen modes, frozen baseline θ. Train only (W_π, b_π, τ, α) ∈ ℝ^{10242} for 500 steps. Measure val NLL every 50 steps. Pass criterion: any val checkpoint with NLL ≤ 4.05 (improvement ≥ 0.03) AND ≥ 90% of checkpoints in [3.95, 4.10].

### 13.2 Implementation (C++98)

New files in `glades-trainer/trainer/igaa/`:
- `igaa_kernels.h` / `igaa_kernels.cu` — K-batched mode forward, softmax mixture, additive injection
- `igaa_backward.cu` — softmax + mixture backward (existing SCFA backward handles mode 1; modes 2-4 are new but simple)

Trainer flags:
- `--igaa-layer L` — layer at which to insert IGAA (default 18)
- `--igaa-k K` — number of modes (default 4)
- `--igaa-modes 1,2,3,4` — comma-separated mode IDs (1=SCFA-full, 2=SCFA-tight, 3=banded-short, 4=identity)
- `--igaa-train` — enable IGAA parameter training (mirrors `--sfa-train`)
- `--igaa-lr` — learning rate for IGAA params (default 1e-4)
- `--igaa-init-sigma` — symmetry-breaking init noise (default 1e-3)
- `--igaa-beta-foc`, `--igaa-beta-drift`, `--igaa-beta-decay` — regularizer weights

### 13.3 Gate-0 falsifiable claim

**Claim**: After 500 training steps of the above setup, val NLL on pretok-data/val will satisfy NLL ≤ 4.05 nat at some checkpoint AND NLL ∈ [3.95, 4.10] at ≥ 90% of checkpoints.

**If claim PASSES**: IGAA architecture is empirically validated as a stable additive improvement. Proceed to Gate-1 (K=16, L_IGAA={6,12,18,22}).

**If claim FAILS (either condition violated)**: 
- If val stays at 4.0771 (no improvement): the K=4 mode set is insufficient. Try K=8 with broader scale coverage.
- If val rises (training destabilizes): IGAA does not survive at this baseline. Document, retract, consider Candidate A or C.

### 13.4 Estimated wall time

500 steps × (~1.1 × baseline step time) ≈ 7 min on the RTX 4080 SUPER. Single experiment fits comfortably.

## 14. Full research program

**Gate-1 — K=16 multi-layer**: insert IGAA at layers {6, 12, 18, 22} with K=16 modes including the structured catalogue (banded variants, ALiBi variants, learned low-rank). Train 2000 steps. Target: ≥ 0.10 nat improvement over Gate-0 best.

**Gate-2 — Joint training**: unfreeze θ alongside ψ for 200 fine-tuning steps after IGAA converges. Target: ≥ 0.20 nat improvement over Gate-1.

**Gate-3 — Scale**: rerun at 1.84B parameter scale (if obtainable). Test whether IGAA's mixture mechanism scales.

**Theoretical extensions**:
- Prove or refute Conjecture E1 (asymptotic optimality as K → ∞).
- Sharpen the bound in Conjecture E2 (max achievable improvement at finite K).
- Extend to **causal mixture flow**: π_i^{(ℓ)} depends on π_i^{(ℓ-1)} via a learned transition, not just on x_i^{(ℓ)}. Gives a Markov chain on the simplex at each token. May improve depth-wise coherence.
- **Adaptive K**: per-layer K_ℓ that grows with depth (deeper layers see more modes). Test against fixed K.

## 15. Open conjectures and validation criteria

**Conjecture C1 (No-op fidelity)**: At ψ = ψ*, val NLL = 4.0771 ± 10^{-5} nat. *Validation*: run flagship-pure equivalent with IGAA inserted at ψ* and confirm bit-exact match.

**Conjecture C2 (Gate-0 falsifiable, primary)**: With the §13 setup, val NLL improves by ≥ 0.03 nat at some checkpoint within 500 training steps. *Validation*: §13.3.

**Conjecture C3 (Stability at K=4)**: Training runs stably for 5000 steps with no NaN, no oscillating divergence. *Validation*: extend Gate-0 to 5000 steps.

**Conjecture C4 (Layer-stack compounding)**: IGAA at L_IGAA={6,12,18,22} achieves ≥ 1.3 × the improvement of IGAA at L_IGAA={18}, demonstrating positive compounding. *Validation*: Gate-1.

**Conjecture C5 (Mode necessity)**: Removing any one of the K=4 modes (ablation: force π_k=0 for one k) loses ≥ 30% of the val NLL improvement. *Validation*: post-Gate-0 ablation, ~15 min wall.

**Conjecture C6 (Magnitudes implausibility)**: IGAA at K=16 and L_IGAA=24 achieves val NLL improvement < 1 nat. *Honest prediction*: this conjecture will be confirmed, not refuted. Documenting it now so we don't post-hoc redefine "magnitudes" downward.

---

**Status: design complete; Gate-0 prototype implementable in ~1-2 days of C++ kernel work; the empirical answer of whether the brief's "magnitudes" target is achievable becomes a measurable claim within ~1 week of focused effort. Conservative honest expectation: 0.05–0.30 nat improvement, well short of magnitudes but a real validated addition to the post-fix baseline.**
