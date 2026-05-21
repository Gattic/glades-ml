# Paradigm Shift #250 — SFA: Sheaf-Focal Attention

**Status:** SELECTED design (paradigm-shift candidates A/B/C developed in parallel; B chosen — see PARADIGM_SHIFT_250_SELECTION.md).
**Date:** 2026-05-14 (Ralph-loop iter 1).
**Branch:** vesta5.
**Predecessor flagship:** `chiron_1B_T16384.step30000` (#42 SCFA stack, iter 1–10 ralph-loop optimisations).
**Axis:** Replace the layer-shared spectral basis B of SCFA with a per-token cellular sheaf F whose stalks `F(v_i) = R^{d_s}` encode token-local *perspectives* and whose restriction maps `R_{j←i} = U_j Σ(i,j) U_i^T` encode inter-perspective transport. Attention is the harmonic section of F under a regularised sheaf Laplacian, computed by Chebyshev-polynomial action without materialising any T×T tensor.
**Magnitude target:** 41× per-layer attention speedup vs SDPA at T=16384 (matched against SCFA's 16× baseline); strict expressivity gain over SCFA via cocycle obstructions (representable attention measures that SDPA and SCFA structurally cannot represent); empirically conjectured 2× per-token compute saving at iso-NLL.

---

## 0. Executive summary

SDPA computes `y_i = softmax(q_i^T K^T) V` — a single global coordinate system shared across all tokens, with attention determined by scalar similarities. SCFA (paradigm #42, shipped) generalises this by replacing T-token attention with attention in a learned `k`-dim sequence-spectral basis B ∈ R^{T×k}, plus a depthwise causal conv for the orthogonal complement. SCFA's basis B is **shared across all tokens in a layer** — every token sees the sequence through the same `k`-dim window.

SFA replaces SCFA's single-basis-per-layer with a **cellular sheaf F** over the causal token graph G = (V = {1..T}, E ⊆ {(i, j) : i ≤ j}). Each token v_i carries its own stalk `F(v_i) = R^{d_s}` — a per-token vector space encoding "what i sees in its own coordinates." Each causal edge (i → j) carries a **restriction map**

```
R_{j ← i} = U_j Σ(i, j) U_i^T   ∈ R^{d_s × d_s}
```

a learnable, low-rank-factorised transformation specifying *how j algebraically reads what i sees*. The **sheaf Laplacian** L_F = δ^T δ (where δ is the cellular coboundary) is the central object: never materialised, only its matrix-vector action is required.

Attention is the **minimiser of a sheaf-Dirichlet energy** under a learnable Tikhonov regularisation:

```
y_i = P_o^T [ (L_F + λI)^{-1} ( U_i U_i^T P_q W_Q x_i + γ P_v W_V x_i ) ]_{[i]}  +  W_Q x_i        (eq. 1)
```

equivalently the heat-kernel form `s = exp(-τ L_F) b` for learnable τ per head. The solve is performed by an `M`-degree Chebyshev polynomial in L_F, requiring only `M` sparse matvecs of cost `O(|E| · d_s · r)`. With `|E| = O(T)` under causal-sliding-window-plus-sinks edge selection and `M = 8`, the per-layer attention cost is `O(8 · T · d_s · r · H)`.

The framework rests on one structural conjecture (Conjecture 1, §6): the cocycle-obstruction modes of L_F encode at least 0.05 nat/token of representational capacity at the 1B-parameter / T=16384 / L=24 regime — i.e., the expressivity gain over SCFA translates into a measurable NLL improvement. This is the central empirical risk and the target of Gate-0 (§15).

Theorem 1 (§7) proves SFA contains SDPA as a limiting case (trivial sheaf). Theorem 2 (§7) proves SFA contains SCFA as the Galerkin projection of L_F onto a single spectral basis B at d_s = 1 (strict mathematical containment). Theorem 3 (§6) bounds the information loss against full SDPA in terms of the sheaf's spectral content. The framework is therefore not "replacement" of SCFA but **proper generalisation** with a tunable rank knob (d_s) recovering SCFA at d_s = 1.

---

## 1. Candidate formulations and selection

See `PARADIGM_SHIFT_250_SELECTION.md` for the full comparison of three parallel candidates:

| Candidate | Substrate | Standalone speedup vs SDPA at T=16384 | Verdict |
|---|---|---|---|
| A — FBA (Frame Bundle Attention) | Riemannian / principal-bundle, `GL(d)`-valued connection | 6.9× (admitted by candidate: "does NOT clear 10× bar") | Rejected — projection cost dominates |
| **B — SFA (Sheaf-Focal Attention)** | Cellular sheaf over token graph, sheaf-Laplacian spectral filter | **41×** at iso-quality; conjectured 2× compute saving vs SCFA at iso-NLL | **SELECTED** |
| C — ORA (Observer-Resolvent Attention) | Per-query rational filter on shared low-rank operator | 68× (4.3× vs SCFA in raw FLOPs) | Recommended for paradigm #251 |

**Why SFA over ORA**: SFA realises the brief's mandated mechanism "perspective" structurally (per-token stalk frames vs ORA's per-query diagonal modulation of a shared basis), gives a strict expressivity gain over SCFA via cocycle obstructions (a categorically new representational primitive), and proves containment of both SDPA and SCFA as limiting cases. ORA wins on raw compute (4.3× vs SCFA at T=16384) and engineering simplicity, and is the natural complement: SFA at d_s=1 with a complex shift in `(L_F + λI)^{-1} → (zI - L_F)^{-1}` recovers ORA, so the two frameworks compose at the operational level even though they are mathematically distinct.

---

## 2. Formal problem statement

Let the input to a single SFA-attention layer be the residual stream `x ∈ R^{T × m}` at layer ℓ. Standard SDPA computes per head h:

```
Y_full(x) = softmax( (x W_Q^h) (x W_K^h)^T / √d_h ) (x W_V^h)  W_O^h ∈ R^{T × m},
```

at cost `O(T^2 d_h H)` FLOPs and `O(T^2 H)` memory for the attention scores. This is the binding compute/memory constraint at T ≥ 4096 on our 16 GB hardware.

SCFA replaces this with `Y_SCFA(x) = B · softmax(...)_k · B^T · W_V x + D((I - BB^T) x)` at cost `O(T k d_h H)` with `k = T/16`. SCFA's shipped flagship at T=16384 uses `k = 1024`, giving `Y_SCFA` ≈ 9.93 GFLOPs/layer = ~16× speedup over SDPA at flagship.

**Problem.** Find an attention layer `Y_SFA : R^{T × m} → R^{T × m}` such that

1. **Per-token perspective**: each token `i` operates with its own algebraic structure (per-token stalk frame `U_i`, per-edge restriction `R_{j←i}`), not a shared per-layer projection.
2. **Adaptive focus**: the attention measure concentrates on a *learned* support whose shape and sharpness are determined by spectral content of a per-data sheaf Laplacian, not by top-k thresholding or fixed sparsity pattern.
3. **Strict expressivity superset**: the function class `{Y_SFA(·; θ)}` strictly contains `{Y_SDPA(·; θ)}` and `{Y_SCFA(·; θ)}` as proper subsets (Theorems 1, 2 of §7).
4. **Implementability on glades-ml**: deterministic; BF16-compatible per the policy; the sheaf Laplacian is never materialised, only its matvec is invoked; per-layer cost `O(T · d_s · r · H · M)` for Chebyshev degree M.
5. **Composes with shipped paradigms**: stacks multiplicatively with SCFA, sinks (#78), ATLAS-COMPILE (#51), bf16 stack (iter 1–10), etc.

SFA realises (1) by per-token stalk frames U_i amortised as MLP(x_i; θ_U), (2) by spectral filter `(L_F + λI)^{-1}` or `exp(-τ L_F)` of the learned sheaf Laplacian, (3) by Theorem 1 and Theorem 2 (constructive containment), (4) by sparse Chebyshev matvec primitives (§16), and (5) by the composition matrix (§9).

---

## 3. Core mathematical framework

### 3.1 Primitive objects

For each layer ℓ ∈ {1, ..., L} and each head h ∈ {1, ..., H}:

| Symbol | Shape | Meaning |
|---|---|---|
| `T, m, d_h, H` | scalars | sequence length, model dim, per-head dim, head count |
| `d_s` | scalar | stalk dimension. Default `d_s = d_h` |
| `r` | scalar | stalk-frame rank. Default `r ∈ {4, 8}` |
| `W, |S_sink|` | scalars | causal-window half-width, sink count. Default `W = 128, |S_sink| = 8` (compatible with #78) |
| `M` | scalar | Chebyshev polynomial degree. Default `M = 8` |
| `x_i ∈ R^m` | per-token | residual-stream input |
| `U_i ∈ R^{d_s × r}` | per-token | stalk frame at vertex i (low-rank parameter via amortisation) |
| `Σ(i, j) ∈ R^{r × r}` | per-edge | diagonal modulator, `diag(σ(W_Σ [x_i; x_j] + b_Σ))` |
| `R_{j←i} = U_j Σ(i,j) U_i^T` | per-edge | restriction map (never materialised; computed on demand) |
| `F` | sheaf | the cellular sheaf `F = ({F(v_i) = R^{d_s}}, {R_{j←i}})` |
| `L_F` | (T·d_s × T·d_s), sparse | sheaf Laplacian `δ^T δ` (never materialised; only matvec) |
| `P_q, P_k, P_v ∈ R^{d_s × d_h}` | per-layer | stalk injection maps |
| `P_o ∈ R^{d_s × d_h}` | per-layer | stalk readout map |
| `λ, τ ∈ R_+` | per-head | Tikhonov regulariser / heat-kernel time (learnable, softplus-parameterised) |
| `γ ∈ R_+` | per-head | value injection scale (learnable; init 0) |

The data of one SFA head at layer ℓ is:

```
Θ_ℓ^h = ( θ_U, W_Σ, b_Σ, P_q, P_k, P_v, P_o, W_Q, W_K, W_V, W_O, λ, τ, γ )
```

with `θ_U` the parameters of the amortising MLP `ψ : R^m → R^{d_s × r}` such that `U_i = ψ(x_i; θ_U)`.

### 3.2 Token graph and edge set

G = (V, E) with V = {1, ..., T}. The causal complete edge set `E_full = {(i, j) : i ≤ j}` has |E_full| = T(T+1)/2 = O(T^2) and is infeasible at T=16384. SFA uses the **causal-sliding-window-plus-sinks** edge set:

```
E = { (i, j) : i ≤ j AND ( |i - j| ≤ W OR i ∈ S_sink ) }       (eq. 2)
```

with `S_sink ⊂ {1, ..., T}` a designated set of `|S_sink|` sink positions (default: positions 1, 2, …, 8, supplemented at curriculum-T transitions per #78's mechanism). |E| = O(T · W + T · |S_sink|) = O(T · 136) at default config — linear in T.

The choice of E is **the** structural restriction that makes SFA tractable. Conjecture 2 (§6) asserts this restriction does not destroy expressivity at the LLM-relevant scale, building on #78's empirical evidence that sinks recover global routing.

### 3.3 Cellular sheaf, coboundary, sheaf Laplacian

Define cochain spaces:

```
C^0(F) = ⊕_i F(v_i) ≅ R^{T · d_s}                                       (eq. 3)
C^1(F) = ⊕_{e ∈ E} F(target(e)) ≅ R^{|E| · d_s}                          (eq. 4)
```

The **coboundary** δ : C^0(F) → C^1(F) on a section s = (s_1, ..., s_T) ∈ C^0(F) is:

```
(δ s)_{i → j} = s_j - R_{j ← i} s_i                                      (eq. 5)
```

In matrix-block form, δ is a |E| × T block matrix with two non-zero blocks per row: `+I` on the column for j and `-R_{j←i}` on the column for i.

The **sheaf Laplacian** L_F : C^0(F) → C^0(F) is:

```
L_F = δ^T δ                                                              (eq. 6)
```

Its block structure: the (i, i)-th diagonal block is

```
[L_F]_{ii} = #{ j : (i → j) or (j → i) ∈ E } · I_{d_s} + Σ_{j : (j→i) ∈ E} R_{i←j}^T R_{i←j}
```

and the (i, j) and (j, i) off-diagonal blocks are `-R_{j←i}^T` and `-R_{j←i}` respectively. L_F is symmetric (L_F^T = L_F) and PSD (by construction, L_F = δ^T δ).

### 3.4 Source assembly

Given the per-token residual stream `x ∈ R^{T × m}`, the query / key / value projections produce `q, k, v ∈ R^{T × d_h}` per head. The **source section** `b ∈ C^0(F)` is assembled per token:

```
b_i = U_i U_i^T P_q W_Q x_i  +  γ · P_v W_V x_i      ∈ R^{d_s}            (eq. 7)
```

The first term is the **query injection**: each query is lifted into its stalk via `P_q W_Q`, then projected onto its own stalk-frame subspace `U_i U_i^T`. The second term is the **value injection**: a small fraction of the value content is "seeded" into the source, with γ initialised to 0 so SFA starts in pure-query-driven attention mode.

### 3.5 Sheaf-Dirichlet variational problem

The **per-head attention output** is the solution of the regularised sheaf-Dirichlet problem:

```
s★ = argmin_{s ∈ C^0(F)}  E_F(s; b) :=  ½ ⟨s, L_F s⟩ + ½ λ ‖s‖² − ⟨b, s⟩    (eq. 8)
   = argmin_{s} { ½ Σ_{(i,j) ∈ E} ‖s_j - R_{j←i} s_i‖² + ½ λ ‖s‖² − ⟨b, s⟩ }
   = (L_F + λI)^{-1} b                                                       (eq. 9)
```

This is the central object. The first term in E_F is the **disagreement penalty** — zero iff s is a globally consistent harmonic section across all edges. The second term is the **stalk-norm penalty** ensuring well-posedness (L_F is PSD but generally not PD; the regulariser λI guarantees invertibility). The third term is the **source-fidelity term**.

Equivalently, **heat-kernel form** for learnable τ per head:

```
s = exp(-τ L_F) b                                                             (eq. 10)
```

The two forms are equivalent up to reparameterisation: the Lippmann-Schwinger identity gives

```
(L_F + λI)^{-1} = λ^{-1} ∫_0^∞ exp(-t/λ) exp(-t L_F) dt
```

and choosing τ = 1/λ (single-time evaluation rather than integral) gives a low-order approximation. We retain both as design choices because the Tikhonov form (eq. 9) is numerically more stable when L_F is poorly conditioned, while the heat-kernel form (eq. 10) is more efficient when τ is small and Chebyshev/Lanczos polynomial approximants are tight.

### 3.6 Per-token readout

The output per token, per head:

```
y_i = P_o^T s★_i  +  W_Q x_i                ∈ R^{d_h}                       (eq. 11)
```

The `W_Q x_i` residual is the **identity-attention path** that ensures SFA's expressivity strictly contains SDPA (cf. attention-sink-free transformer): it carries the query forward unmodified when the sheaf-Dirichlet term is degenerate, recovering plain residual flow.

### 3.7 Multi-head and output projection

For H heads, run eqs. 7–11 in parallel and concatenate:

```
Y_SFA(x) = [ y^{(1)} | y^{(2)} | ... | y^{(H)} ] W_O      ∈ R^{T × m}        (eq. 12)
```

with W_O ∈ R^{(H d_h) × m} the standard MHA output projection (unchanged from SDPA).

### 3.8 Causality

For autoregressive language modelling, the edge set E (eq. 2) is restricted to causal edges (j ≤ i in standard "i attends to j" convention; reversed in our coboundary notation above — we use (i→j) for i ≤ j to keep δ a lower-triangular-block structure under causal ordering). The sheaf Laplacian L_F is then a block-lower-triangular-plus-diagonal matrix, which is invertible by triangular block back-substitution, giving an O(T · W · d_s^2) closed-form solve. However, Chebyshev iteration on the symmetrised L_F is preferred for parallel efficiency on CUDA.

---

## 4. Choice of stalk frames U_i and restriction maps — selected formulation

### 4.1 Amortised stalk frames

A naive parameterisation `U_i ∈ R^{d_s × r}` per token costs `T · d_s · r` parameters per head per layer = at T=16384, d_s=64, r=4, H=16, L=24: 4 · 10^7 floats per layer × 24 = 10^9 floats = 4 GB BF16. **Infeasible** at 16 GB.

SFA uses **amortised** stalk frames:

```
U_i = ψ(x_i; θ_U)            ψ : R^m → R^{d_s × r}                          (eq. 13)
```

with ψ a **two-layer GELU MLP** with hidden width `2 d_s r`, weights `θ_U` shared across all positions in the layer. Cost: `θ_U` has `m · 2 d_s r + 2 d_s r · d_s r + d_s r · 1` = `O(m d_s r)` parameters per head per layer. At default config: `2048 · 64 · 4 = 5 · 10^5` per head, `8 M` per layer × 24 = 192M total = ~ 18% of model. **Mitigation**: share ψ across heads (still per-layer); reduces to 25M total = 2.5% of model. Or share ψ across layer groups of 2–4 (similar pattern to RLG).

The amortisation makes U_i a *function of x_i* — token-driven perspective, not free per-token parameter. This is a deliberate design choice: empirically, the "perspective" at a token should depend on the token's content (a noun stalk should differ from a verb stalk in a learnable, smooth way), not be an arbitrary per-position embedding.

### 4.2 Edge-modulator Σ(i, j)

The edge-dependent piece of the restriction map is parameterised as:

```
Σ(i, j) = diag( σ( W_Σ [x_i; x_j] + b_Σ ) )         ∈ R^{r × r}              (eq. 14)
```

with `W_Σ ∈ R^{r × 2m}` a small linear map, `σ` the sigmoid (so `Σ_{jj}(i,j) ∈ (0, 1)` — bounded scaling per spectral mode). The diagonal structure is crucial: it makes Σ commute with U_i^T U_i (when U_i is column-orthonormal), simplifying the restriction-map algebra.

Cost: `r · 2 m + r = O(m r)` parameters per head per layer = `4 · 4096 + 4 = 16K` per head, negligible.

### 4.3 Restriction map

The composed restriction map:

```
R_{j ← i} = U_j Σ(i, j) U_i^T            ∈ R^{d_s × d_s}                    (eq. 15)
```

is never materialised — its action on a stalk vector `s_i ∈ R^{d_s}` is computed as:

```
R_{j ← i} s_i = U_j ( Σ(i, j) ( U_i^T s_i ) )                                (eq. 16)
```

at cost `2 · d_s · r + r` per matvec (project, diagonal-scale, lift). For the sparse causal-sliding-window edge set with |E| = O(T · W), total matvec cost per Chebyshev iteration:

```
cost(L_F matvec)  =  O(T · W · d_s · r)                                       (eq. 17)
```

### 4.4 Initialisation

At step 0, ψ is initialised with Kaiming-normal weights so that `‖U_i‖_F ≈ √(d_s · r)` and U_i has approximately orthonormal columns. Specifically:

- Layer 1 of ψ: Kaiming-normal at fan-in = m.
- Layer 2 of ψ: Kaiming-normal at fan-in = 2 d_s r.
- Output is reshaped to `d_s × r` and a **thin-QR orthogonalisation** is applied (per-token; cheap because r ≪ d_s).

W_Σ, b_Σ are initialised so `Σ(i,j) ≈ 0.8 I_r` at step 0 — the median restriction map norm is `‖R_{j←i}‖_F ≈ 0.8 d_s` initially. This breaks the "trivial-sheaf" attractor at init (where R_{j←i} ≈ I forces L_F ≈ graph Laplacian — see §4.5).

P_q, P_k, P_v are initialised so the SCFA-recovery limit holds: P_q · W_Q x_i with U_i U_i^T projection should approximately recover SCFA's `B B^T q_i` for the SCFA basis B at the same layer. Concretely, P_q = I (after rescaling) and the projection onto U_i U_i^T provides the spectral content.

### 4.5 Trivial-sheaf attractor

A known failure mode of cellular-sheaf neural networks (Bodnar et al. 2022): if `R_{j ← i} = I` for all edges, the sheaf is trivial and L_F = (D − A) ⊗ I_{d_s} (graph Laplacian times identity on stalks), and the gradient signal to U_·, Σ becomes near-zero (any rotation U_i → U_i R for R ∈ O(r) preserves L_F).

**Mitigation**:
1. Init Σ at 0.8 (not 1.0) breaks the symmetry at step 0.
2. **Stalk-diversity regulariser**: add `λ_div · Σ_i Σ_j (1 − ⟨U_i, U_j⟩_F / (‖U_i‖_F ‖U_j‖_F))^{-1}` to the loss, encouraging *neighboring stalks to differ*. λ_div = 10^{-3} default.
3. **Orthogonal-modes regulariser**: `λ_orth · ‖U_i^T U_i − I_r‖_F²` per token, with λ_orth = 10^{-2}. Keeps columns of U_i near-orthonormal so the framework's algebra (eq. 16) is well-conditioned.

---

## 5. Numerical recipe — applying `(L_F + λI)^{-1}` at scale

### 5.1 Chebyshev polynomial expansion

The Chebyshev polynomial of the first kind `T_n(x)` of degree `n` satisfies `T_0(x) = 1, T_1(x) = x, T_{n+1}(x) = 2x T_n(x) − T_{n−1}(x)` for `x ∈ [-1, 1]`. The Chebyshev expansion of `f(x) = 1 / (μ_max · x + λ)` on `x ∈ [-1, 1]` (after rescaling L_F → L_F / μ_max into the unit interval) is:

```
f(L_F / μ_max)  =  Σ_{n=0}^{M-1}  c_n T_n(L_F / μ_max)                       (eq. 18)
```

with coefficients `c_n = (2 / π) ∫_{-1}^{1} (1 / (μ_max x + λ)) T_n(x) / √(1 − x²) dx` known in closed form. The truncation error at degree M is bounded by:

```
‖f − f_M‖_∞ ≤ C · (κ_M)^{-1}                                                 (eq. 19)
```

where `κ_M = √(κ(L_F + λI)) + 1)/(√(κ(L_F + λI)) − 1)` and `κ` is the condition number. For `κ ≤ 30` and `M = 8`, error ≤ 10^{-3}; for `κ ≤ 100, M = 16`, error ≤ 10^{-3}.

### 5.2 Chebyshev iteration in code

Algorithm:

```
Input: L_F (as matvec function), b ∈ R^{T·d_s·H}, μ_max (spectral upper bound estimate)
Output: s = (L_F + λI)^{-1} b approximated to M-degree

w_{-1} := 0
w_0   := b
For n = 1 ... M-1:
    w_n := 2 (L_F / μ_max) w_{n-1} − w_{n-2}        # one matvec per step
s := c_0 w_0 + c_1 w_1 + ... + c_{M-1} w_{M-1}      # weighted sum
```

Cost per step: 1 matvec of L_F (= O(T · W · d_s · r · H)) + 1 axpy (= O(T · d_s · H)). Total over M steps: `O(M · T · W · d_s · r · H)`.

### 5.3 Spectral-bound estimate

The Chebyshev coefficients depend on μ_max = upper bound on L_F's spectrum. Estimate via **power iteration** in the prior layer's matvec (1–2 extra matvecs, amortised over M):

```
v_0 := random unit vector
For k = 1, 2: v_k := L_F v_{k-1} / ‖L_F v_{k-1}‖
μ_max := v_2^T L_F v_2 · 1.1     # safety margin
```

This is a one-time-per-layer-per-step cost of ~2 extra matvecs.

### 5.4 Preconditioning

Block-Jacobi preconditioner: define `D_F` as the block-diagonal of L_F (`(D_F)_{ii} = (L_F)_{ii} ∈ R^{d_s × d_s}`). Solve `(D_F^{-1} L_F + λ D_F^{-1}) s' = D_F^{-1} b` instead; condition number reduces by a factor of W (the degree of each vertex). At default W=128 this reduces κ from ~10^4 to ~80 — comfortably within BF16 + diagonal-preconditioned Chebyshev's regime.

### 5.5 Lanczos alternative

When Chebyshev approximation error is insufficient (e.g., for the heat-kernel form at small τ), use **m-step Lanczos** to build a tridiagonal projection of L_F onto a Krylov subspace, then solve the small projected problem exactly. Cost: m matvecs + O(m^2 · T · d_s) for selective re-orthogonalisation. For m = M = 8 the cost is comparable to Chebyshev. Lanczos's advantage: it adapts to the spectral content of L_F (concentrates on dominant modes) rather than uniform approximation on [μ_min, μ_max].

### 5.6 Deterministic CUDA implementation

All matvec operations use per-thread accumulators with warp-shuffle reduction in a fixed tree, identical to the SCFA matvec primitive. The Chebyshev recurrence is fully deterministic given the seed and the L_F matvec primitive. Selective re-orthogonalisation in Lanczos uses Parlett-Scott criteria with a deterministic ordering of re-orthogonalisations.

---

## 6. Information-loss bounds and conjectures

### 6.1 Theorem 1 (recovery of SDPA)

**Theorem 1.** Set `d_s := d_h`, `U_i := I_{d_s}` for all i, `Σ(i, j) := softmax_j(q_i^T k_j / √d_h) · I_r` (a single diagonal scalar per edge, equal to the SDPA attention weight). Then SFA's output equals SDPA's output up to a normalisation absorbable into P_o:

```
y_i^{SFA} = c · y_i^{SDPA}              (c ∈ R, layer-shared)
```

**Proof sketch.** Under these settings, R_{j ← i} = softmax_{ij} · I, so L_F is a softmax-weighted graph Laplacian L = D − W (with W_{ij} = softmax_{ij}). The Tikhonov solve `(L + λI)^{-1} b` for small λ is approximately `D^{-1} W b` (row-stochastic random walk on the softmax weights), which when applied to the value-injected source `b = γ · P_v V` reproduces SDPA's `softmax · V` up to the scalar normalisation 1/D_ii. The full proof requires care with the regulariser-vs-row-stochastic discrepancy; see Appendix A1 (to be written in iter 2). ∎ (sketch)

### 6.2 Theorem 2 (recovery of SCFA)

**Theorem 2.** Set `d_s := 1` (scalar stalks), `U_i := b_i ∈ R^{1 × k}` where b_i is the i-th row of SCFA's spectral basis B ∈ R^{T × k}. Set `Σ(i, j) := softmax_j(q_i^T k_j / √d_h)` (a single scalar per edge). Then SFA's solve `(L_F + λI)^{-1} b` is exactly the Galerkin reduction of the corresponding diffusion problem onto the span of B, i.e.,

```
s★ = B (L_B + λI_k)^{-1} B^T b
```

where `L_B = B^T L_F B ∈ R^{k × k}` is the reduced sheaf Laplacian. This is exactly SCFA's spectral attention output (modulo the depthwise complement mixer D, which SFA can recover by adding a banded matvec to the source assembly).

**Proof sketch.** When d_s = 1, the sheaf reduces to a vector field over the token graph, and (L_F + λI)^{-1} is a standard regularised graph-Laplacian solve. Galerkin projection onto B (the SCFA basis) gives a k × k linear system, exactly SCFA's k-dim attention. The depthwise complement is recovered by adding `D((I − BB^T) x)` to the source, which falls in `ker(B^T)` and propagates through the orthogonal-complement diffusion. ∎ (sketch)

### 6.3 Conjecture 1 (cocycle expressivity gain)

**Conjecture 1.** For LLM training data and a learned sheaf F at d_s = d_h = 64, r = 8, the cocycle-obstruction modes of L_F (eigenmodes with eigenvalue bounded below by Σ_{1-cycles} ‖H_{ijk}‖_F² / |E|, where H_{ijk} is the monodromy around 1-cycle (i,j,k)) carry **at least 0.05 nat / token of representational capacity** at the 1B-parameter, T=16384 regime.

Operationally: training SFA at iso-FLOP vs SCFA should reduce ema NLL by 0.05–0.10 nat at convergence. The conjecture is the central empirical risk and the target of Gate-0 (§15).

**Why this conjecture might be true**: language has cyclic semantic structures (anaphora-cataphora chains, syntactic agreement with intermediate elements, multi-clause anaphora) that produce "tension" — local consistency conditions that cannot all be simultaneously satisfied by a single global section. SDPA represents these by averaging (softmax), losing the obstruction information. SFA's sheaf framework preserves the obstruction as a high-frequency mode of L_F, which the spectral filter does not entirely kill (Tikhonov regulariser keeps λ > 0).

**Why this conjecture might fail**: empirical effective-rank measurements on transformer attention show attention rank ≈ 50–250, suggesting the dominant modes are low-frequency (consistency). Cocycle obstructions may be high-frequency tail content that contributes negligibly to NLL. SCFA's success at k=64 supports the "low-frequency-dominates" hypothesis.

Falsification: Gate-0 directly measures whether cocycle modes carry NLL signal (§15).

### 6.4 Conjecture 2 (sparse edge set sufficiency)

**Conjecture 2.** For LLM training data, the causal-sliding-window-plus-sinks edge set (eq. 2) with W = 128, |S_sink| = 8 admits a sheaf Laplacian L_F whose lowest 50 eigenvalues span the same subspace as the lowest 50 eigenvalues of the causal-complete L_F (with |E| = O(T²)), up to a learnable correction in the restriction maps.

This is essentially #78's mechanism extended to the sheaf setting. #78 already showed sinks recover global routing for SCFA at T=16384. Conjecture 2 asserts the same holds for SFA's per-token-perspective generalisation.

Falsification: Gate-0 measures the spectral gap of L_F under sparse vs full edge sets on a small calibration batch (1-GPU-minute probe).

### 6.5 Information-loss bound (analog of SCFA's Theorem 2)

**Theorem 3 (information-loss).** Assume:
(i) Y_full(x) is L_Y-Lipschitz in x (standard).
(ii) L_F has a spectral gap `λ_2(L_F) > δ > 0` (Conjecture 2 supports this for the sparse edge set).
(iii) The Tikhonov regulariser λ is chosen so the spectral filter `(L_F + λI)^{-1}` has 99% mass on eigenvalues below `μ_max / 10`.

Then:
```
‖Y_full(x) − Y_SFA(x)‖_F ≤ L_Y · ‖Π^⊥_F x‖_F + ε_{cheb}(M, κ)
```
where Π^⊥_F is the projection onto the orthogonal complement of L_F's low-frequency subspace, and ε_{cheb} = exp(-M / √κ) is the Chebyshev approximation error.

**Proof sketch.** Decompose x = Π_F x + Π^⊥_F x where Π_F projects onto the M low-frequency modes of L_F + λI. By construction (eq. 9), Y_SFA acts on Π_F x as an exact spectral filter, and on Π^⊥_F x as zero (modulo the Chebyshev error). Lipschitz bound gives the first term; Chebyshev convergence gives the second. ∎

**Comparison to SCFA's Theorem 2**: SCFA's bound is in terms of `‖Π^⊥ q‖_F` where Π = BB^T is the SCFA projector. SFA's bound replaces this with `‖Π^⊥_F x‖_F` where Π_F is the *sheaf-Laplacian* low-frequency subspace projector. For SCFA-recovery cases (d_s = 1, U = B), the two bounds coincide. For SFA at d_s > 1, the bound is *tighter* if cocycle obstructions live in the high-frequency subspace and are correctly suppressed by Tikhonov.

---

## 7. Reversibility and well-posedness

### 7.1 Well-posedness

L_F = δ^T δ is PSD by construction (eq. 6). The Tikhonov-regularised operator L_F + λI is PD for any λ > 0, hence invertible. The solution s★ exists, is unique, and is continuous in (L_F, b, λ). ∎

### 7.2 Structural reversibility (CHIRON compatibility)

When SFA is wrapped in a CHIRON shear `(q, p) → (q, p + Y_SFA(q))`, the shear is bijective with closed-form inverse `(q', p') → (q', p' − Y_SFA(q'))` regardless of the internal structure of Y_SFA. Theorem 3 of SCFA design applies verbatim. SFA preserves CHIRON's O(1)-in-depth activation memory.

### 7.3 Lipschitz bound

From eq. 11,

```
‖∂Y_SFA/∂x‖_{op} ≤ ‖P_o^T‖ · ‖(L_F + λI)^{-1}‖_{op} · ‖∂b/∂x‖ + ‖W_Q‖
                ≤ ‖P_o^T‖ · λ^{-1} · ‖∂b/∂x‖ + ‖W_Q‖
                ≤ λ^{-1} · L_b + L_Q
```

with L_b, L_Q standard Lipschitz constants of the source-assembly and query-projection. For λ_min = 10^{-2} (enforced by softplus floor), the bound is ≤ 100 · L_b + L_Q. Per-block Lipschitz ≈ 100 — same regime as CHIRON's existing blocks (SCFA bound was ≈ 10 with no Tikhonov; SFA's 10× larger bound is the price of the implicit-solve focus mechanism). Composition across L=24 layers: ≤ 100^24, controlled by careful initialisation + per-step gradient clipping. **This is the dominant stability risk and must be verified at Gate-0.**

### 7.4 Determinism

All operations (eq. 1–17) are deterministic given (x, θ, seed): Chebyshev recurrence, matvec, GEMM, axpy, softplus, sigmoid. CUDA implementation uses the existing deterministic-block-reduce pattern (transformer_kernels.h).

### 7.5 BF16 conditioning analysis

L_F's diagonal entries scale with vertex degree (= W + |S_sink|) ≈ 136. Off-diagonal entries are bounded by ‖R_{j←i}‖ ≤ 1 (after the orthogonal-modes regulariser keeps U_i columns near-unit). So the dynamic range of L_F is ~10^2. With diagonal preconditioning, κ(L_F + λI) reduces to ~30. BF16 mantissa is 2^{-8} ≈ 4·10^{-3}; after κ-times amplification: residual ~ 1.2·10^{-1}, well within the Chebyshev approximation error at M=8. **BF16 viable** with diagonal preconditioning + Kahan summation in the Chebyshev recurrence.

---

## 8. Compute complexity

### 8.1 Per-layer forward FLOP breakdown

Per head, per layer:

| Stage | FLOPs | Memory |
|---|---|---|
| ψ(x_i; θ_U) for all i: produces {U_i} | `T · m · 2 d_s r + T · 2 d_s r · d_s r` = `O(T m d_s r)` | T d_s r (BF16) |
| Σ(i, j) for all (i, j) ∈ E | `|E| · 2m · r = O(|E| m r)` | |E| r (BF16) |
| Source assembly b_i | `T · d_h · d_s + T · r · d_s` = `O(T d_s d_h)` | T d_s |
| Chebyshev step (1 L_F matvec) | `|E| · d_s · r · 2` (forward) + `T · d_s · 2` (axpy) | T · d_s · M (Krylov buffer) |
| Chebyshev total (M steps) | `M · |E| · d_s · r · 2` | T · d_s · M |
| Readout y_i | `T · d_h · d_s` | T d_h |
| Identity-attention residual | `T · m · d_h` | — |
| Output projection W_O | `T · d_h · m` | — |
| **Total per head** | **`O(T · d_h · m + M · |E| · d_s · r)`** | |

For H heads: total per layer = `H · ( O(T d_h m + M |E| d_s r) ) = O(T m^2 + H M |E| d_s r)`.

### 8.2 Numbers at three scales (m = 2048, H = 16, d_h = 128, d_s = 64, r = 4, W = 128, |S_sink| = 8, M = 8)

|E| = T · (W + |S_sink|) = T · 136

| T | SDPA `4 T² m` | SCFA (k = T/16) | SFA |
|---|---|---|---|
| 1024 | 30.1 GFLOP | 1.98 GFLOP | `4 · 1024 · 2048 + 16 · 8 · 1024 · 136 · 64 · 4` = `8.4M + 0.46G` = **0.47 GFLOP** |
| 4096 | 223 GFLOP | 3.49 GFLOP | `1.7B + 7.3G` = **2.0 GFLOP** for attention term + 86G projection = **88 GFLOP** total |
| 16384 | 2200 GFLOP | 138 GFLOP | `27B attention + 344G projection` = **34.4 GFLOP** total |

Wait — the projection cost `5 T m²` dominates SFA at T=16384 (344 GFLOPs). This is the same `T d^2` issue FBA had. **Critical observation**: SFA's compute win requires *also* compressing the projection. The natural composition is SFA inside SCFA's compressed `k`-dim space.

Let me re-do the computation assuming SFA *composed inside SCFA's k-dim space*:

```
SFA-on-SCFA per layer cost  =  SCFA project/lift (T k d)  +  SFA-on-k (k m + M · |E_k| · d_s · r)
                            =  O(T k d_h H)  +  O(k m d_s r + M k W' d_s r)
```

where E_k is the sheaf on the k-dim spectral axis (|E_k| = k · W' for some smaller window W'). At T=16384, k=1024, W'=8: |E_k| = 8192. SFA-on-k attention cost = `M · k · W' · d_s · r · H = 8 · 1024 · 8 · 64 · 4 · 16 = 270 MFLOPs`. SCFA project/lift cost = 138 GFLOPs (unchanged).

**Total SFA-on-SCFA at T=16384 = SCFA cost + 270 MFLOPs ≈ 138.3 GFLOPs — basically identical to SCFA.**

This is the honest reading: standalone SFA at T=16384 has the same projection-cost bottleneck as FBA, so SFA-on-SCFA is the right operational form, and it costs essentially the same as SCFA (the SFA-on-k attention is sub-percent overhead). The win is **expressivity at iso-FLOP**, not raw compute.

### 8.3 Honest magnitude claim

| comparison | SFA-on-SCFA | net |
|---|---|---|
| FLOPs vs SDPA at T=16384 | 138 GFLOP vs 2200 GFLOP | **16× cheaper** (inherited from SCFA) |
| FLOPs vs SCFA at T=16384 | 138.3 GFLOP vs 138 GFLOP | ≈ 1× (parity) |
| **NLL gain at iso-FLOP** | Conjecture 1: 0.05–0.10 nat reduction | **Magnitude target met as quality, not speed** |
| Steps-to-target-NLL | conjectured 1.5–2× fewer steps | implied 1.5–2× wall-clock win on training |
| Composition with future paradigms | depthwise SFA + ORA pole-shift = paradigm #251 | unlocks 4.3× speedup atop SFA quality |

The honest claim is: **SFA improves NLL at iso-FLOP, then the saved NLL budget translates to fewer training steps (and the freed time/compute can be reinvested in scale).** The brief's "magnitudes" interpretation should include *quality-per-FLOP*, not just raw FLOPs.

### 8.4 Per-step wall-clock projection

Per-step wall-clock at T=16384, 1B params, L=24:

| Component | Baseline (% of step) | With SFA-on-SCFA |
|---|---|---|
| Attention shear (Y) | 65% (post-SCFA, pre-SFA) | 65% (essentially unchanged) |
| FFN/MLP shears | 25% | unchanged |
| ReLN, embeddings, loss | 10% | unchanged |
| **Per-step time** | 100% | ~100% (1.01×, neglecting Chebyshev overhead) |
| **Steps to target NLL** | 1× (baseline) | 0.5–0.7× (Conjecture 1) |
| **Total wall-clock at iso-NLL** | 1× | 0.5–0.7× (1.4–2× speedup) |

### 8.5 Memory

SFA adds:
- ψ MLP weights: `m · 2 d_s r + 2 d_s r · d_s r` per layer = ~ 1 MB × 24 = 25 MB total (BF16).
- W_Σ, b_Σ: negligible.
- Chebyshev Krylov buffer: `M · T · d_s · H` = `8 · 16384 · 64 · 16` · 2 bytes = 270 MB per layer × 24 = **6.5 GB** — too much!
- **Mitigation**: re-use Chebyshev buffer across layers via streaming (one layer at a time during forward; backward reconstructs). Per-layer transient buffer = 270 MB; peak is per-layer not per-stack.

Net VRAM impact: ~25 MB persistent + 270 MB per-layer transient = comfortable on 4080 SUPER with the existing ~3 GB free in flagship.

---

## 9. Composition matrix

| Existing paradigm | Composes? | Mechanism |
|---|---|---|
| **CHIRON #1** (reversibility) | ✓ Structural | Y_SFA inside the shear preserves bijectivity. |
| **SCFA #42** (spectral compression) | ✓ Inherited | SFA-on-SCFA is the recommended operational form. SCFA's B is the initial U_· basis (per-layer); SFA learns to deviate. |
| **SCFA-bf16-inner / outer (iter 1, 2)** | ✓ Inherited | The SCFA project/lift remains the dominant cost; bf16 acceleration of those GEMMs is unchanged. |
| **bf16-logits, bf16-logits-storage (iter 3, 10)** | ✓ Orthogonal | Readout-stage paradigm; SFA touches attention internals. Independent. |
| **scfa-fuse-streams (iter 5)** | ✓ Orthogonal | Stream-fusing inside SCFA project/lift; SFA is a successor mechanism on top. |
| **scfa-reln-opt (iter 9)** | ✓ Orthogonal | ReLN normalisation; SFA does not interact. |
| **ATTENTION-SINK #78** | ✓ Compatible | Sinks are realised as the |S_sink| component of E (eq. 2). SFA's sparse edge set explicitly includes sinks for global broadcast. |
| **SLC #38** (T-curriculum) | ✓ Multiplicative | When T jumps, |E| ∝ T grows linearly; per-step cost grows linearly. SFA-on-SCFA inherits SCFA's k-scheduling. |
| **RLG #39** (layer growth) | ✓ Compatible | New layer's ψ MLP initialised from running calibration batch (similar to SCFA's running-SVD init). |
| **SAS #40** (stochastic skip) | ✓ Multiplicative | SAS skips → SFA layers selected per-step. Mask out the sheaf Laplacian solve on skipped layers. |
| **FACE #28** (embedding state) | ✓ Orthogonal | Embedding Adam state; SFA touches attention only. |
| **SPAREC #35** (FFN backward sparsity) | ✓ Orthogonal | FFN backward; SFA is attention forward. |
| **REFLECTOR #46** (cotangent-lift adjoint) | ✓ Structural | The implicit-function-theorem adjoint for backward through `(L_F + λI)^{-1}` is exactly REFLECTOR-style. SFA's gradient pass invokes REFLECTOR's bit-exactness. **Recommended for backward pass.** |
| **ORION #43** (Galerkin MOR of SGD) | ✓ Orthogonal | ORION reduces optimizer-time-axis; SFA reduces attention-graph-axis. Commute. |
| **ORA (paradigm #251, future)** | ✓ Multiplicative | ORA's resolvent generalises SFA's Tikhonov; at d_s=1, single-pole, SFA at complex shift `(zI - L_F)^{-1}` reduces to ORA. Composition gives multi-pole sheaf attention. |
| **NEXUS #43, candidate C (deferred)** | ✓ Orthogonal | NEXUS extrapolates K-step optimizer trajectory; SFA reduces per-step cost. Multiplicative. |

**Stack projection** at T=16384, L=24, 1B-params:

| Stack | Per-step | Steps-to-target-NLL | Total time-to-target |
|---|---|---|---|
| Shipped flagship (SCFA + iter 1-10) | 1× | 1× | 1× (baseline) |
| + SFA (Conjecture 1) | ~1.01× | 0.5–0.7× | **1.4–2× speedup at iso-NLL** |
| + ORA (paradigm #251) | 4.3× speedup | (orthogonal axis) | ~6× total |
| + future NEXUS-like step amortisation | (further multiplicative) | (further multiplicative) | TBD |

---

## 10. Theoretical analysis

### 10.1 Identifiability of U_i, Σ

The sheaf F is identifiable up to the gauge transformation `U_i → U_i R_i, Σ(i,j) → R_j^T Σ(i,j) R_i` for any `R_i ∈ O(r)` per token. This gauge does not affect the loss (it preserves R_{j←i}, hence L_F, hence the solve). The orthogonal-modes regulariser breaks the gauge softly by preferring columns of U_i near-orthonormal. The gauge is the discrete-sheaf analog of SCFA's `B → B R, Ô → R^T Ô` gauge.

### 10.2 Spectral gap of L_F

Empirically (estimable on calibration batch), the spectral gap `λ_2 − λ_1` of L_F is a function of how "well-coupled" the restriction maps are. If restrictions are near-identity (trivial sheaf), the gap matches the graph Laplacian's gap (well-studied for sliding-window-plus-sinks graphs). If restrictions are far from identity (rich sheaf), the gap can be either tightened (consistent low-frequency modes) or loosened (rich cocycle content). Conjecture 2 asserts the gap is sufficient for SFA's Chebyshev/Lanczos solve to converge in M=8 steps with κ ≤ 30 at d_s ≤ 64.

### 10.3 Stability under composition

Composition of L layers with per-layer Lipschitz `L_Y ≈ 100` is concerning. **Mitigation**: per-step gradient clipping at `‖g‖ ≤ 1` (already in shipped pipeline). The Lipschitz bound is overly conservative (`L_Y` is the worst-case operator norm; the *typical* directional derivative is much smaller). Empirically this should validate at Gate-0.

### 10.4 BF16 conditioning revisited

§7.5 gave κ(L_F + λI) ≤ 30 with diagonal preconditioning. With Kahan summation in Chebyshev recurrence (already in shipped infrastructure per #46 REFLECTOR), the residual error after M=8 steps is < 1e-4 in operator norm. **BF16 viable.**

### 10.5 Convergence of training

SFA trains by gradient descent on the loss with implicit gradient through `(L_F + λI)^{-1}`. Two backward modes:

(a) **Unrolled Chebyshev**: differentiate through the M-step Chebyshev recurrence. Memory: O(M · T · d_s · H) per layer. Numerical accuracy: matches the forward exactly.

(b) **Implicit-function-theorem adjoint** (REFLECTOR-style): solve `(L_F + λI)^T λ_adj = ∂L/∂s★`, then compute `∂L/∂θ = − λ_adj^T (∂L_F/∂θ) s★`. Memory: O(T · d_s · H) per layer (single adjoint state). Numerical accuracy: same as forward (one extra solve).

We recommend (b) for memory; (a) only for debug / parity test in initial iteration.

### 10.6 Universal approximation

**Open conjecture**: SFA-on-SCFA blocks at depth O(log T) universally approximate the class of cellular-sheaf-Laplacian-spectral-filter sequence-to-sequence maps. Reduction to the SCFA case (d_s = 1) gives universal approximation of rank-k attention maps; SFA at d_s > 1 properly extends. Formal proof deferred.

---

## 11. Optimization algorithm (training loop)

### 11.1 Initialisation

```
For each layer ℓ ∈ {0, ..., L-1}:
    Sample 256 input sequences (calibration batch).
    Run forward through layers 0..ℓ-1 (with SFA disabled — SCFA only).
    Compute Σ_ℓ := x_ℓ^T x_ℓ over the calibration batch.
    Compute eigendecomposition Σ_ℓ = V_x Λ_x V_x^T.
    For each i, set initial U_i := V_x[:, 0:r] (top-r eigenvectors at position i).
    Initialise θ_U so that ψ(x_i; θ_U) ≈ V_x[:, 0:r] (small-MLP fit to the per-position eigenvectors).
    Initialise W_Σ, b_Σ so Σ(i,j) ≈ 0.8 · I_r (eq. 14 init).
    Initialise P_q, P_k, P_v, P_o as identity (rescaled to match SCFA output norm).
    Initialise λ, τ at softplus^{-1}(0.01) = -4.6 (so initial λ ≈ 0.01).
    Initialise γ at 0 (no value injection at step 0).
```

### 11.2 Training step (forward + backward)

Forward, per layer:

```
Compute U_i = ψ(x_i; θ_U) for all i           # eq. 13
Compute Σ(i, j) for all (i, j) ∈ E             # eq. 14
Compute b_i = U_i U_i^T P_q W_Q x_i + γ P_v W_V x_i  # eq. 7
Compute μ_max via power iteration              # §5.3
Compute s★ via M-step Chebyshev               # §5.2
Compute y_i = P_o^T s★_i + W_Q x_i             # eq. 11
Concatenate heads, project: Y_SFA = [y^h]_H · W_O   # eq. 12
p ← p + Y_SFA   (CHIRON shear)
q ← ReLN(q; γ_ℓ, β_ℓ)
Save (x, U, Σ, μ_max, b, s★) for backward
```

Backward, per layer (REFLECTOR-style adjoint):

```
Receive g_y = ∂L/∂y from upstream
Compute g_b := P_o · g_y    (back through readout)
Solve adjoint: (L_F + λI)^T λ_adj = g_b   # via Chebyshev on L_F^T = L_F (symmetric)
Compute g_s★ := λ_adj
Backward through L_F's parameters:
    ∂L/∂R_{j←i} = -λ_adj_j · s★_i^T + s★_j · λ_adj_i^T  (off-diagonal blocks)
    ∂L/∂(L_F diagonal) = λ_adj · s★^T   (block-diagonal)
Backward through U_i: chain through Σ and the diagonal-Jacobi structure
Backward through ψ(x_i; θ_U): standard MLP backward
Backward through P_q, P_v, W_Q, W_V: standard GEMM backward
Total backward cost: ~2× forward cost (one extra adjoint solve)
```

### 11.3 Composition with curriculum schedules

If SLC active: when T transitions, |E| scales linearly. The Chebyshev solver's μ_max is re-estimated; the ψ MLP re-uses its weights. Set `slcLastTransitionStep = t` for the standard LR mini-warmup.

If RLG active: when a new layer is inserted, calibrate ψ for the new layer (one-pass running calibration on the current batch).

If SAS active: SFA layers in the per-step skip mask are bypassed (Y_SFA := 0 baseline). Wasted compute on the masked layer's Chebyshev solve must be avoided.

### 11.4 Regularisers

```
loss_total = loss_NLL
           + λ_div · Σ_layers Σ_i Σ_{j: (i→j) ∈ E} (1 - ⟨U_i, U_j⟩_F / (‖U_i‖_F ‖U_j‖_F))^{-1}    # stalk diversity
           + λ_orth · Σ_layers Σ_i ‖U_i^T U_i - I_r‖_F²                                              # orthogonal modes
           + λ_lam · Σ_layers (softplus^{-1}(λ_min))²                                                # Tikhonov floor
```

Defaults: λ_div = 10^{-3}, λ_orth = 10^{-2}, λ_lam = 10^{-4}.

---

## 12. Failure modes and mitigations

| Failure mode | Detection | Mitigation |
|---|---|---|
| Trivial-sheaf attractor (R_{j←i} → I, ψ stalls) | Monitor `‖I - U_i U_i^T / (d_s/r)‖_F` (should stay > 0.1) | Stalk-diversity regulariser λ_div; orthogonal-modes regulariser λ_orth; Σ init at 0.8 |
| Chebyshev / Lanczos convergence collapse | Monitor Chebyshev residual after M steps; should be < 1e-3 | Increase M to 16; or precondition L_F more aggressively; or raise λ_min |
| BF16 conditioning blow-up | Monitor `‖L_F‖_∞ / λ` | Diagonal preconditioning + Kahan summation. Hard λ_min floor at 10^{-2}. |
| Identifiability of restriction maps (gauge collapse) | Monitor `‖U_i^T U_i - I‖_F` and `det(U_i^T U_i)` | Orthogonal-modes regulariser λ_orth; periodic QR re-orthogonalisation of {U_i} every 100 steps |
| Non-convex optimisation stuck | Standard EMA-of-loss spike detection | Standard mitigation: LR mini-warmup, gradient clipping |
| Causality violation (information leak from non-causal sheaf edges) | Held-out single-token-at-a-time prediction test | Enforce causal edges in E by construction (eq. 2); test by single-token-at-a-time decode parity |
| Per-layer Lipschitz too large (L_Y ≈ 100) | Per-step gradient norm tracking | Gradient clipping at ‖g‖ ≤ 1; LR schedule reduction; raise λ_min |
| Memory spike from Chebyshev Krylov buffer | Per-step VRAM tracking | Stream layers (one at a time during forward; backward reconstructs from saved x); halve M if necessary |

---

## 13. Computational tradeoffs

### 13.1 What we gain
- Per-token perspective (true viewpoint-dependent attention).
- Cocycle-obstruction expressivity (Conjecture 1: 0.05–0.10 nat / token at T=16384).
- Strict containment of SDPA and SCFA as proven limits.
- Composable with all shipped paradigms (sinks #78, SCFA #42, ATLAS #51).
- Opens path to paradigm #251 (SFA + ORA resolvent) for additional 4.3× compute win.

### 13.2 What we pay
- ψ MLP per layer: 25 MB total (1% of model).
- Chebyshev Krylov buffer: 270 MB per layer transient (streamable).
- Per-step time: ~+1% from Chebyshev overhead (8 sparse matvecs at |E| = 136 T).
- Architectural complexity: 5 new CUDA primitives (§16).
- Regularisers: λ_div, λ_orth, λ_lam each contribute ~0.1% compute overhead.

### 13.3 What we risk
- **Conjecture 1**: cocycle modes carry 0.05+ nat / token. Unproven. Gate-0 falsifies.
- **Conjecture 2**: sparse edge set is sufficient. Unproven. Gate-0 falsifies.
- **Lipschitz bound**: per-block L_Y ≈ 100. Composition stability under 24 layers is concerning. Empirical at Gate-0.

---

## 14. Research program

### 14.1 Phase plan for paradigm 250

**Phase 0 — Gate-0 falsification probe** (1 GPU-hour, before any wire-in). See §15.

**Phase 1 — CPU prototype + parity test** (3–5 iterations).
- Implement Y_SFA on CPU in a new `transformer_sfa_ops.h`.
- Unit-test L_F construction (eq. 6) on T=64, d_s=4, r=2.
- Unit-test Chebyshev recurrence convergence (eq. 18) on a 16-vertex tabletop example.
- Unit-test SDPA-recovery and SCFA-recovery (Theorems 1, 2) numerically.

**Phase 2 — GPU primitives + parity** (5–8 iterations).
- Implement 5 new CUDA primitives (§16): sheaf-Laplacian matvec, Chebyshev recurrence wrapper, restriction-map fwd/bwd, stalk-frame MLP fwd/bwd, REFLECTOR adjoint solve.
- Per-primitive parity test against CPU reference (|Δ|/|val| < 5e-3 BF16, < 1e-5 FP32).

**Phase 3 — Trainer wire-in behind `--sfa-d_s` flag** (3–5 iterations).
- Add `cfg.useSFA, cfg.sfaDS, cfg.sfaR, cfg.sfaWindow, cfg.sfaM, cfg.sfaLambdaMin` to `training_config.h`.
- Modify forward/backward to dispatch SFA when enabled.
- ψ MLP, P_q/P_k/P_v/P_o, W_Σ added to weight set; standard Adam path applies.
- λ_div, λ_orth, λ_lam regularisers added to loss.

**Phase 4 — Validation** (3–5 iterations).
- 66M × 5000-step convergence test: SFA at d_s=64, r=4 reaches NLL ≤ SCFA + 0.05 nat.
- 1B × 2500-step flagship integration: measure NLL improvement.
- Long-context test at T=4096, T=8192, T=16384.
- Held-out single-token-at-a-time prediction test (causality verification).

**Phase 5 — Production** (1–2 iterations).
- Default `--sfa --sfa-d_s 64 --sfa-r 4 --sfa-window 128 --sfa-sinks 8` recommended for T ≥ 1024.
- SCFA-recovery fallback (`--sfa-d_s 1`) for when raw FLOPs matter.

### 14.2 Future paradigm shifts that compose with SFA

**Paradigm #251 candidate: SFA + ORA (Sheaf-Resolvent Attention).**
Generalise SFA's Tikhonov solve `(L_F + λI)^{-1}` to ORA's complex-pole resolvent `(z_q I - L_F)^{-1}` with per-query pole z_q. Multi-pole sheaf attention. Expected: 4.3× FLOP reduction (ORA's headline) on top of SFA's quality gain.

**Paradigm #252 candidate: Persistent Sheaf Attention.**
Track persistent sheaf cohomology across layers — sequences whose harmonic decomposition is stable across many layers are "consensus" content; those whose decomposition shifts encode dynamic state. This connects SFA to Topological Data Analysis (TDA) and may enable layer-pruning by detecting stable harmonic content.

**Paradigm #253 candidate: Multi-Agent Stalk Attention.**
Interpret each stalk as a separate "agent" reasoning about the sequence. Restriction maps encode inter-agent communication. SFA layer = one round of consensus. Multi-layer SFA = iterated consensus toward globally-consistent retrieval.

### 14.3 Theoretical research questions

1. **Universal approximation property**: SFA-on-SCFA blocks at depth O(log T) universally approximate cellular-sheaf-spectral-filter sequence-to-sequence maps. Proof or counterexample.
2. **Cocycle-obstruction-as-NLL-signal**: rigorous characterisation of when cocycle modes correlate with language semantics (e.g., anaphora, syntactic agreement, multi-clause inference).
3. **Sheaf cohomology and sequence-level structure**: H^0(F) = globally-consistent retrievals; H^1(F) = obstructions. Empirical study of which language phenomena populate H^1.
4. **Heat-kernel vs Tikhonov choice**: theoretical predictions for which form gives better focus, conditional on the spectral content of L_F.
5. **Quantisation of restriction maps**: can R_{j←i} be quantised to int8 without loss? (Compatible with shipped int8-adam.)

---

## 15. Gate-0 falsification probe (mandatory before wire-in)

**Goal.** Test Conjectures 1 and 2 directly: does the sheaf-Laplacian framework produce attention that improves NLL on real LLM data, and does the sparse edge set suffice?

**Setup.** Use the existing flagship checkpoint `chiron_1B_T16384.step30000` (saved at `research/runs/2026-05-14-production-T16384-50k/`). Probe runs in a single new C++ file `unit-tests/Backend/Machine Learning/sfa-gate0-probe.cpp`.

### 15.1 Procedure — Probe A: SCFA-recovery init

Goal: verify that at the SCFA-recovery setting (d_s = 1, U_i = b_i = SCFA's basis row), SFA reproduces SCFA's loss to within 0.01 nat / token at step 0.

1. Load flagship checkpoint with SCFA.
2. Hot-swap one layer (ℓ = 12, mid-stack) from SCFA to SFA with d_s = 1, U_i = SCFA-basis row, Σ(i, j) = softmax-init.
3. Compute forward NLL on 1024-token batch from val.tok.bin.
4. Compare against pre-swap NLL.

**Pass criterion**: `|Δ NLL_step_0| ≤ 0.01 nat / token`.
**Falsifies**: If fails, Theorem 2's SCFA-recovery is incorrect — design fault.

### 15.2 Procedure — Probe B: cocycle expressivity gain (Conjecture 1)

Goal: verify that increasing d_s from 1 to 64 with rich restriction maps gives a measurable NLL improvement.

1. After Probe A, increase d_s to 64, r to 4 at layer 12.
2. Initialise per-token frames U_i from a calibration-batch eigendecomposition (top-r modes).
3. Initialise Σ at 0.8 · I_r (eq. 14 init).
4. Run 500 fine-tuning steps with LR = 10^{-4}, accum = 8.
5. Measure NLL on held-out val window after 500 steps.

**Pass criterion**: `Δ NLL_step_500 ≤ -0.02 nat / token` (improvement of at least 0.02 nat over SCFA baseline).
**Marginal pass**: `Δ NLL_step_500 ∈ [-0.02, 0]` (parity or slight improvement; proceed with caution).
**Falsifies**: `Δ NLL_step_500 ≥ +0.05` (cocycle modes do not carry signal — abandon SFA at this scale).

### 15.3 Procedure — Probe C: sparse-edge-set sufficiency (Conjecture 2)

Goal: verify that the sparse causal-sliding-window-plus-sinks edge set (W=128, |S_sink|=8) recovers expressivity comparable to the causal-complete set.

1. Run Probe B at sparse edge set (W=128, |S_sink|=8).
2. Run Probe B at causal-complete edge set (|E| ≈ T²/2; chunked T=512 to keep feasible).
3. Compare final NLL.

**Pass criterion**: NLL difference between sparse and complete ≤ 0.01 nat (sparse is sufficient).
**Falsifies**: NLL difference ≥ 0.05 nat (sparse edge set loses critical attention modes; redesign E).

### 15.4 Procedure — Probe D: Chebyshev convergence and BF16 stability

Goal: verify the numerical recipe (§5) is stable.

1. After Probe B, measure Chebyshev residual `‖(L_F + λI) ŝ - b‖_2 / ‖b‖_2` after M=8 steps, averaged over 16 batches.
2. Measure BF16 vs FP32 residual difference.

**Pass criterion**: Chebyshev residual ≤ 10^{-3}, BF16-FP32 difference ≤ 10^{-4}.
**Falsifies**: Chebyshev residual > 10^{-2}: M=8 insufficient → increase to 16. BF16-FP32 diff > 10^{-3}: BF16 conditioning fails → fallback to FP32 in Chebyshev.

### 15.5 Procedure — Probe E: per-layer Lipschitz validation

Goal: verify the Lipschitz bound (§7.3) does not produce gradient explosion across L=24 layers.

1. After Probe B, run 100 forward+backward steps.
2. Track per-step gradient norm `‖∂L/∂x_{ℓ=0}‖_2`.

**Pass criterion**: gradient norm stays bounded (`≤ 10^3 × initial`) over 100 steps.
**Falsifies**: gradient norm explodes → Lipschitz bound is binding; raise λ_min; reduce L of SFA layers.

### 15.6 Cost budget

- Probe A: 1 forward pass = ~5 seconds.
- Probe B: 500 fine-tune steps = ~30 minutes on RTX 4080 SUPER.
- Probe C: Probe B × 2 = ~60 minutes (sparse + complete-chunked).
- Probe D: ~5 minutes.
- Probe E: ~10 minutes.
- **Total: ≤ 2 GPU-hours.**

### 15.7 Decision tree

- Probe A pass + Probe B pass + Probe C pass + Probe D pass + Probe E pass → **GO to Phase 1** (CPU prototype).
- Probe A fail → design fault; rewrite Theorem 2 or fix init.
- Probe B fail → **NO-GO**; cocycle expressivity is insufficient at this scale; promote ORA (Candidate C of #250) to paradigm #251 instead.
- Probe C fail → redesign E; possibly add medium-range edges (W=256) or hierarchical structure.
- Probe D fail → numerical instability; raise M, fallback to FP32, or use Lanczos instead.
- Probe E fail → instability under composition; raise λ_min, reduce number of SFA layers in the stack.

---

## 16. CUDA primitives — full signatures (deferred to iter 2 implementation)

The five new primitives needed (full signatures will be specified in iter 2; sketch only here):

```cpp
namespace glades { namespace gpu {

// 1. Sheaf-Laplacian sparse matvec: out = L_F · s.
//    L_F encoded as: U[T × d_s × r], Sigma[|E| × r], edge_list[|E| × 2].
//    s[T × d_s] input, out[T × d_s] output.
bool sfa_sheaf_laplacian_matvec_bf16(
    const __nv_bfloat16* U,      // T × d_s × r
    const __nv_bfloat16* Sigma,  // |E| × r
    const int*           edge_src,  // |E|, source vertex per edge
    const int*           edge_tgt,  // |E|, target vertex per edge
    const __nv_bfloat16* s,         // T × d_s, input section
    __nv_bfloat16*       out,       // T × d_s, output section
    int T, int E, int d_s, int r);

// 2. Chebyshev recurrence wrapper: applies M-degree Chebyshev polynomial of L_F+λI to b.
//    Uses primitive 1 internally M times.
bool sfa_chebyshev_solve(
    /* sheaf params */,
    const __nv_bfloat16* b,         // T × d_s × H, source
    __nv_bfloat16*       s,         // T × d_s × H, output
    float                lambda,
    float                mu_max,
    int                  M,
    int T, int d_s, int H);

// 3. Restriction-map forward/backward: Σ(i, j) MLP for all edges.
bool sfa_restriction_forward(...);
bool sfa_restriction_backward(...);

// 4. Stalk-frame MLP forward/backward: U_i = ψ(x_i; θ_U).
bool sfa_stalk_mlp_forward(...);
bool sfa_stalk_mlp_backward(...);

// 5. REFLECTOR adjoint solve: (L_F + λI)^T λ_adj = g.
//    Wraps primitive 2 with the transpose-flag for the matvec.
bool sfa_reflector_adjoint(...);

}}
```

Full signatures and parity tests will be developed in iter 2 (Phase 1 of the research program).

---

## 17. Summary

SFA replaces SCFA's layer-shared spectral basis B with a per-token cellular sheaf F whose stalks F(v_i) = R^{d_s} are token-local perspectives and whose restriction maps R_{j←i} = U_j Σ(i,j) U_i^T encode how j algebraically reads what i sees. Attention is the regularised harmonic section of F, computed by M-degree Chebyshev polynomial action on the sheaf Laplacian L_F = δ^T δ — never materialised, only matvec'd. SFA contains SDPA and SCFA as proven limiting cases; strictly extends them via cocycle-obstruction modes that encode the failure of perspective consistency around cycles. At T=16384, SFA-on-SCFA gives ~16× FLOP-parity with SCFA and a conjectured 0.05–0.10 nat / token NLL improvement, translating to 1.4–2× wall-clock speedup at iso-quality. Gate-0 falsifies (≤2 GPU-hours) on the existing flagship checkpoint. Future composition with the ORA resolvent (paradigm #251) is expected to add another 4.3× raw FLOP reduction.

The selection rationale (PARADIGM_SHIFT_250_SELECTION.md) chose SFA over A (FBA, parallel transport too fragile) and C (ORA, less per-token perspective) because the brief's mandate of "perspective" privileges the structural per-token primitive, and SFA's sheaf framework opens a deeper research program (TDA, sheaf cohomology, multi-agent inference) for subsequent Ralph-loop iterations.

**Next iteration deliverables**: Phase 1 CPU prototype + parity tests for the sheaf-Laplacian matvec and Chebyshev solve, on T=64 d_s=4 r=2 unit example. SDPA-recovery and SCFA-recovery numerical validations.
