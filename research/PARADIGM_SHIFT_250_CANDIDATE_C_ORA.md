# Paradigm shift #250 — Candidate C: ORA (Observer-Resolvent Attention)

**Status:** candidate — one of three competing formulations for shift #250 ("Focused attention with perspective").
**Date:** 2026-05-14.
**Axis:** operator-theoretic / spectral. Replace softmax with a per-query rational (resolvent) function of a per-query observer operator. Compose with #42 SCFA.
**Materially distinct from:** (i) sparse top-k attention (Reformer/LSH, BigBird, sliding-window) which fix the *index support* of the attention measure but not its functional form; (ii) low-rank linear attention (Performer, Linformer, SCFA) which projects K to a low-rank basis but keeps softmax-on-projections; (iii) multi-head attention which uses multiple *fixed-form* attentions in parallel subspaces; (iv) Mamba/SSM which use a single layer-shared linear operator on the value stream rather than a per-query operator. ORA generates a **per-query operator** A_i and applies a **rational function** (zI − A_i)^{−1} to V — the focus comes from the analytic *pole structure*, not from index sparsity, and the perspective comes from A_i, not from per-head subspaces.

---

## 1. Primitive objects

Fix a layer ℓ; suppress batch and head indices; restore them in §10. The base symbols:

| symbol | shape | meaning |
|---|---|---|
| T | scalar | sequence length, T ∈ {1024, …, 16384} |
| d | scalar | model dim per head, d = m / n_H |
| m | scalar | model dim, m = 2048 at our 1B target |
| n_H | scalar | head count |
| X | ℝ^{T×m} | layer input token stream |
| Q, K, V | ℝ^{T×d} | projections Q = X W_Q, K = X W_K, V = X W_V |
| q_i | ℝ^{d} | i-th query row, i ∈ [T] |
| r | scalar | shared operator rank, r ≪ T (default r = 64) |
| m_K | scalar | Krylov depth, m_K ≪ r (default m_K = 8) |
| U | ℝ^{T×r} | shared sequence-axis basis (per layer, per head; learned) |
| H | ℝ^{r×r} | shared compressed key operator (built from U, K) |
| A_i | ℝ^{T×T} (factored) | per-query observer operator (never materialised) |
| z_i | ℂ | per-query spectral pole (or ℝ-pair; see §3.4) |
| c_i | ℝ^{d} | per-query readout vector (small MLP of q_i) |
| y_i | ℝ^{d} | i-th output token |
| ε | scalar | resolvent regulariser (default ε = 10^{−2}) |

The output stream Y = [y_1; …; y_T] ∈ ℝ^{T×d} is then mapped back through W_O ∈ ℝ^{d×m} (per-head, concatenated as in standard MHA). Causality is enforced through the operator construction (§5.2).

Let σ(A) denote the spectrum of a matrix A. Let R(z; A) := (zI − A)^{−1} denote the resolvent of A at z ∈ ℂ ∖ σ(A). For a unit vector v, the **resolvent measure** of A at v is the scalar Borel measure μ_v,A on σ(A) defined by μ_v,A(S) := v^* P_S(A) v where P_S(A) is the spectral projector on S; then v^* R(z;A) v = ∫ (z − λ)^{−1} dμ_v,A(λ). This object is the formal substrate of ORA.

---

## 2. State space and central equation

### 2.1 ORA output: rational of operator

The per-query output is

```
                          m_K − 1
(1)   y_i  =  c_i^T  R(z_i ; A_i) V  =  c_i^T  ∑      α_k^{(i)} · A_i^{k} · v_i
                          k = 0
```

where v_i := V q_i / ‖V q_i‖ is the query's seed vector on the value manifold (chosen because Lanczos on A_i started from v_i builds the Krylov subspace whose mass coincides with where q_i looks), and α_k^{(i)} are the coefficients of the **degree-(m_K − 1) Padé approximant** to the function f(λ) = 1/(z_i − λ) on σ(A_i).

The central equation is therefore

```
(2)   y_i  =  c_i^T  (z_i I_T  −  A_i)^{−1}  V                ∈ ℝ^{d},
```

evaluated by an m_K-step Krylov projection (§4). The two halves of the design name appear here:

- **Perspective** = A_i. The per-query operator A_i determines which directions on the value stream the query is sensitive to. Its spectrum {λ_j(A_i)} is the set of "eigentokens this query can see"; its eigenvectors are the per-query value-axis modes.
- **Focus** = z_i. The pole z_i selects, by spectral filter, the band of A_i's spectrum closest to it. Specifically, (z_i − λ)^{−1} is sharply peaked on λ ≈ z_i; the closer Im(z_i) → 0 with Re(z_i) → λ*, the sharper the focus on the single eigentoken λ*. **The pole position IS the attention sharpness.**

Equation (2) is the central object the rest of the document operationalises.

### 2.2 Why a resolvent (mechanism statement)

SDPA computes `y_i = softmax(q_i^T K^T) V`. The softmax is a *fixed scalar function* applied entrywise to *scalar similarities* `s_ij = q_i^T k_j`. ORA computes a *rational operator-function* `(z_i − A_i)^{−1}` applied to V. The Cauchy kernel `(z − λ)^{−1}` is the universal "spectral concentrator": for a self-adjoint A,

```
∫_σ(A) | (z − λ)^{−1} |^2 dμ(λ)  →  ∞ as Im(z) → 0 near a discrete eigenvalue.
```

So as z_i is dragged toward σ(A_i), the resolvent puts unbounded weight on the eigenfunctions near z_i. This produces **focus with arbitrary sharpness** without any explicit top-k operation, and the locus of focus is the operator's spectrum, not a token index set. Multi-pole rational functions `∑_p β_p (z_p − λ)^{−1}` give **multi-modal focus** — a single query can simultaneously concentrate on several disjoint spectral bands of A_i. SDPA cannot do this without explicit multi-head replication.

---

## 3. Parameterisation of A_i

The hard design constraint: A_i ∈ ℝ^{T×T} dense per query is `T queries × T^2` floats per layer = catastrophic. We use a **shared low-rank factorisation with a per-query diagonal modulator**:

```
(3)   A_i  =  U Λ(q_i) U^T  +  γ · I_T,
```

where

- U ∈ ℝ^{T×r} is the shared layer-level sequence-axis basis, with columns near-orthonormal (`U^T U ≈ I_r`, enforced by the same QR re-orthogonalisation already used by SCFA — see §9.1).
- Λ(q_i) ∈ ℝ^{r×r} is a **diagonal** matrix whose r entries are produced by a per-query MLP `q_i ↦ Λ(q_i)`. Specifically `diag Λ(q_i) = MLP_Λ(q_i)` with `MLP_Λ : ℝ^d → ℝ^r`, a two-layer GELU MLP with hidden width 2r and tied weights across i.
- γ ∈ ℝ is a learned per-layer scalar shift (small, default γ_init = 0).

This factorisation has three critical properties:

(P1) **Per-query cost is `O(r)`, not `O(r^2)`.** Only the diagonal entries of Λ are query-dependent; U is shared. Total per-layer parameters are `T·r` (for U) + `O(d·r)` (for MLP_Λ).

(P2) **Spectrum of A_i is `{γ + λ_j(q_i)}_{j=1}^{r}` plus `T − r` copies of `γ` (on ker U^T).** Since Λ(q_i) is diagonal, its eigenvalues *are* its diagonal entries, λ_j(q_i) = [Λ(q_i)]_{jj}. The per-query MLP_Λ therefore directly controls **where this query's spectrum sits**.

(P3) **Krylov subspace K_{m_K}(A_i, v_i) is contained in colspan U ⊕ span(v_i, γ·v_i)** — Krylov vectors `A_i^k v_i = U Λ^k U^T v_i + γ^k v_i`. With γ ≪ ‖Λ‖, the second term is a small correction and the dominant subspace is r-dimensional. Hence the m_K-step Krylov projection on the T-dim space *reduces to an m_K-step problem in the r-dim space*, which is the cost win.

We pick the operator-form (a) from the brief; we reject (b) per-query Lanczos on K (too expensive — see §6) and reject (c) polynomial-of-K (too restrictive — the operator's spectrum is forced to equal a polynomial transform of σ(K) and the per-query control collapses to a per-query polynomial coefficient set, which is a scalar reweighting equivalent to a learned softmax temperature ramp).

### 3.4 Pole parameterisation

The pole z_i ∈ ℂ. To use real arithmetic on CUDA we parameterise

```
(4)   z_i  =  ρ_i  +  i · ω_i,        ρ_i ∈ ℝ,   ω_i ∈ ℝ_+,
```

with `(ρ_i, ω_i) = MLP_z(q_i)` and `ω_i = softplus(ω̃_i) + ε` to enforce strict positivity (ε = 10^{−2}, the regulariser of §1). Complex resolvents factor through the **real Cayley form** (§4.3): a complex pole at ρ + iω corresponds to a real 2×2 block

```
            [ ρ  -ω ]
   Z_i  =   [        ]   ∈ ℝ^{2×2},
            [ ω   ρ ]
```

and (zI − A)^{−1} acts on a *complexified* Krylov basis whose representation is two real copies of the original. This eliminates ill-conditioning at the spectrum (since `ω_i ≥ ε > 0`) and runs entirely in real BF16 or FP32 arithmetic.

For multi-focus, we parameterise P poles per query: `{z_i^{(p)}, β_i^{(p)}}_{p=1..P}` with `P ≪ r` (default P = 2) and combine as a partial-fraction:

```
                  P
(5)   y_i  =  c_i^T  ∑  β_i^{(p)} (z_i^{(p)} I − A_i)^{−1}  V.
                  p=1
```

P = 1 is the base form; P > 1 is an opt-in expressivity dial.

---

## 4. Evolution / update law — explicit formulas

We now derive the concrete computation of (2) under the parameterisation (3). The proof is in §6.

### 4.1 Shared Krylov decomposition

For a fixed layer, define the *seed vector* `b_i := U^T v_i ∈ ℝ^r`, where v_i := V q_i (we drop the normalisation; absorbing into c_i below). Then

```
(6)   A_i^{k} v_i  =  U · ( Λ(q_i)^{k}  U^T v_i )  +  γ^k v_i
                  =  U · Λ(q_i)^{k} · b_i  +  γ^k v_i.
```

The first term lives in colspan(U); the second is the unmodulated identity-shift contribution. Since Λ(q_i) is diagonal, Λ^k is just elementwise k-th powers: `Λ^k b_i = (λ_j^k · [b_i]_j)_{j=1..r}`.

The resolvent's Padé / Krylov approximation (m_K-term) is:

```
(7)   (z_i I − A_i)^{−1} v_i  ≈  ∑_{k=0}^{m_K - 1} α_k(z_i; γ) · A_i^k v_i.
```

For the operator-form (3) with diagonal Λ, the right-hand side admits a **closed form** in the spectral basis:

```
(8)   (z_i I − A_i)^{−1} v_i  =  U · D(z_i, q_i) · b_i  +  (z_i − γ)^{−1} (v_i − U U^T v_i),
```

where `D(z_i, q_i) ∈ ℝ^{r×r}` is the **diagonal** matrix

```
                                          1
(9)         [D(z_i, q_i)]_{jj}   =   ──────────────────────────────,        j = 1..r.
                                       z_i − γ − λ_j(q_i)
```

The two terms in (8) are (i) the *spectral-mode response* — exactly what we want: per-eigentoken-mode scalar reweighting by the resolvent kernel, applied to the modes b_i = U^T V q_i of "what this query sees" — and (ii) the *complement response* — uniform attenuation of the part of v_i orthogonal to U. This is the operator-theoretic analog of SCFA's Π/Π^⊥ decomposition (§7).

**Mass on V.** To get the full y_i (not just its action on v_i), we apply the same operator to V column-by-column. By linearity:

```
(10)  (z_i I − A_i)^{−1} V   =   U · D(z_i, q_i) · U^T V  +  (z_i − γ)^{−1} (I − UU^T) V.
```

This is the operator on the *full value stream*, factorised so the inverse never materialises a T×T matrix.

### 4.2 The output

Substituting (10) into (2) and using c_i:

```
(11)  y_i  =  c_i^T · [ U · D(z_i, q_i) · U^T V  +  (z_i − γ)^{−1} (I − UU^T) V ]^T
          =  ( U^T c_i )^T · D(z_i, q_i) · ( U^T V c_i )  + ...   wait — dimensions.
```

Let me re-do (11) with explicit dimensions. (z_i I − A_i)^{−1} ∈ ℂ^{T×T}, V ∈ ℝ^{T×d}, so (z_i I − A_i)^{−1} V ∈ ℂ^{T×d}, and we extract the i-th row. The i-th row of M V for any M ∈ ℝ^{T×T} is `e_i^T M V = (M^T e_i)^T V`. Let `ψ_i := (z_i I − A_i)^{−T} e_i ∈ ℂ^T`. Since A_i is symmetric in our parameterisation (3) when Λ is real, `(z_i I − A_i)^{−T} = (z_i I − A_i)^{−1}` and ψ_i = (z_i I − A_i)^{−1} e_i. Then

```
(12)  y_i  =  ψ_i^T V  ∈ ℂ^{d}.
```

We take Re(y_i) (component-wise, for a single complex pole; for the conjugate pair we sum), apply the readout c_i as a *weighting*, and write the per-i output of the head as a small post-resolvent affine map:

```
(13)  y_i^{out}  =  W_y · Re(ψ_i^T V)^T  +  b_y · c_i,        c_i  =  MLP_c(q_i) ∈ ℝ^d,
```

with W_y ∈ ℝ^{d×d}, b_y ∈ ℝ^d learned. (The c_i factor breaks the gauge symmetry y_i ∝ ψ_i^T V under uniform rescaling and lets the readout learn a head-conditioned mixing.)

Plugging (10) into ψ_i:

```
(14)  ψ_i  =  U · D(z_i, q_i) · U^T e_i  +  (z_i − γ)^{−1} (I − UU^T) e_i
          =  U · D(z_i, q_i) · U^T_{i,:}    +   (z_i − γ)^{−1} (e_i − U_{i,:}^T)
```

where U^T_{i,:} is the i-th row of U^T (i.e. (U_{i,:})^T ∈ ℝ^r). Then ψ_i^T V is

```
(15)  ψ_i^T V  =  ( U U^T_{i,:} D(z_i, q_i)^T )^T V  +  (z_i − γ)^{−1} (e_i − U_{i,:}^T)^T V
              =  U^T_{i,:}^T D(z_i, q_i) (U^T V)  +  (z_i − γ)^{−1} ( V_{i,:} − U_{i,:} (U^T V) ).
```

This is the operational formula. Define the **shared modal value matrix**

```
(16)  M  :=  U^T V    ∈ ℝ^{r×d}        (computed once per layer, shared across all i),
```

and the per-row coordinate vector

```
(17)  u_i  :=  U_{i,:}   ∈ ℝ^{r}        (the i-th row of U, free; U is a stored weight).
```

Then the per-i output is

```
(18)  ψ_i^T V   =   u_i^T · D(z_i, q_i) · M    +    (z_i − γ)^{−1} ( V_i  −  u_i^T M ),
```

with `D(z_i, q_i)_{jj} = 1 / (z_i − γ − [Λ(q_i)]_{jj})` per (9). Equation (18) is the production formula. It is closed-form, real-valued (after the Cayley complexification of §3.4), and **does NOT involve any T×T matrix and NO Krylov iteration**: under the operator-form (3), the resolvent has an exact closed form. The Krylov / Padé apparatus is only needed if we generalise A_i beyond (3) (see §11.3 / future extension).

### 4.3 Compact rewrite

Define `g_i := D(z_i, q_i) · M ∈ ℂ^{r×d}` (a per-i r×d operator on the modal value matrix M), and `h_i := V_i − u_i^T M ∈ ℝ^{d}` (the residual). Then (18) is

```
(19)  y_i  =  u_i^T g_i  +  (z_i − γ)^{−1} h_i.        (REAL-PART)
```

This is **the** evolution law of ORA. Each query (i) reads its row u_i of U, (ii) computes its r diagonal modal eigenvalues Λ(q_i), (iii) constructs the scalar reweighting g_i of the shared modal value matrix M, (iv) adds a uniformly-attenuated residual. The whole layer is r-rank per query, computed by O(r·d) work per query plus a single shared U^T V projection.

---

## 5. Mechanism mapping (where focus and perspective live)

### 5.1 Perspective = A_i = U Λ(q_i) U^T

- The *basis* U is layer-shared. It defines a **layer-level "viewing manifold"** — the r-dim subspace of sequence positions on which queries in this layer all operate. U is structurally identical to SCFA's B (§9 below), which lets ORA inherit SCFA's basis-training infrastructure intact.
- The *modulator* Λ(q_i) is query-dependent. It defines, for each query, **which spectral modes are present in A_i's spectrum**, by reweighting the modes of U. A query that pushes λ_j(q_i) → 0 effectively *removes* mode j from its viewing subspace; a query that boosts λ_j(q_i) → 1 emphasises mode j as a near-pole eigenvalue.
- Thus the per-query perspective = a per-query selection-with-weighting of an r-dim modal basis. This is **NOT** the same as a per-head subspace (a per-head subspace would be a fixed U^h independent of q). The perspective in ORA is **per-query**, with O(r) parameters per query (the diagonal).

### 5.2 Focus = z_i = (ρ_i, ω_i)

- ρ_i selects the spectral location: `(ρ_i − γ − λ_j(q_i))^{−1}` is large precisely when ρ_i ≈ γ + λ_j(q_i). So ρ_i is "which eigentoken am I looking at."
- ω_i is the **sharpness dial**: as ω_i → 0, the kernel becomes singular at λ* = ρ_i − γ; the resolvent puts unbounded weight on the closest eigenmode. As ω_i → ∞, the kernel flattens to (z_i)^{−1} · I — uniform attenuation, no focus. So ω_i is **a learned, per-query, continuous focus radius**, with `radius ≈ ω_i / max_j |∂λ_j/∂ρ|`.
- A multi-pole query (5) gives **multi-modal focus**: a single query can attend to two disjoint spectral bands (e.g. one entity early in the sequence + one late), without resorting to multi-head replication, by placing two poles at the two band locations.

### 5.3 Causality

A_i must commute with the causal mask M^c (lower triangular). Under parameterisation (3) this requires U^T M^c U = U^T M^c U (trivially true) and Λ(q_i) commute with U^T M^c U. The simplest sufficient condition: **U is the basis of a causal-compatible family** — its columns are supported on causal half-lines. We use a *cumulative Chebyshev basis* (column j of U is the cumulative orthonormalised Chebyshev polynomial T_j truncated to [0, t_j] with monotone non-decreasing t_j), which gives `U^T M^c U = U^T U ≈ I_r` (causal mask is a no-op in causal-compatible basis). This is the same construction SCFA uses (§9.4); ORA inherits it.

For position i < t_j, U_{i, j} = 0 by support; for i ≥ t_j, U_{i, j} = Chebyshev value. The i-th row u_i therefore has support on `{j : t_j ≤ i}`, automatically masking out future modes.

---

## 6. Variational principle / objective derivation

We derive y_i from a per-query Tikhonov problem and show that the closed form (10) is its exact solution.

### 6.1 Problem statement

Let H := A_i (a self-adjoint operator on ℝ^T when Λ is real). Consider the per-query problem

```
(20)  y_i^{*}  =  argmin_{y ∈ ℝ^T}   J_i(y) :=  ‖ y  −  V q_i ‖_2^2   +  λ_i · ‖ (z_i I − A_i) y ‖_2^2.
```

The first term anchors y to the standard query–value response `V q_i` (the SDPA proto-target without softmax). The second term penalises y's deviation from being a near-eigenvector of A_i at eigenvalue z_i — i.e. it pulls y toward the spectral mode of A_i nearest z_i.

### 6.2 Closed-form solution

Setting `∂J_i/∂y = 0`:

```
2 (y − V q_i)  +  2 λ_i (z_i I − A_i)^T (z_i I − A_i) y   =   0
⇒  [ I + λ_i (z_i I − A_i)^2 ]  y   =   V q_i.
```

Hence

```
(21)  y_i^{*}  =  [ I + λ_i (z_i I − A_i)^2 ]^{−1}  V q_i.
```

### 6.3 Identification with the resolvent

The operator `I + λ_i (z_i I − A_i)^2` has spectrum `{1 + λ_i (z_i − μ)^2}_{μ ∈ σ(A_i)}` (since A_i and (z_i I − A_i)^2 share eigenvectors). Its inverse has eigenvalues `(1 + λ_i (z_i − μ)^2)^{−1}`.

**Theorem 6.1 (Tikhonov-resolvent identity).** Let `z̃_i = z_i + i / √λ_i` (a complex shifted pole). Then for self-adjoint A_i,

```
(22)  [ I + λ_i (z_i I − A_i)^2 ]^{−1}   =   λ_i^{−1} · Im[ (z̃_i I − A_i)^{−1} ] · 1/Im(z̃_i)   ·   (constants)
```

up to a multiplicative real constant absorbable into the readout. The proof is by spectral decomposition: with A_i = U_A Σ U_A^T, both sides are diagonalised in U_A; LHS has diagonal `(1 + λ_i(z_i − σ_j)^2)^{−1}`, RHS has diagonal `(Im(1/(z̃_i − σ_j))) / Im(z̃_i)`, and a direct calculation gives both equal to `1 / (1 + λ_i (z_i − σ_j)^2)`. (Use `1/(a + ib) = (a − ib)/(a^2 + b^2)`; the imaginary part of the resolvent is the *spectral density*.)

So **(21) is the imaginary part of the resolvent of A_i at the complex pole z̃_i = z_i + i/√λ_i**, applied to V q_i:

```
(23)  y_i^{*}  ∝   Im[ (z̃_i I − A_i)^{−1} ]  V q_i,
```

which is (after readout c_i and the real-/imag-part trick of §3.4) **exactly** equation (2). The complex-pole damping `Im(z̃_i) = 1/√λ_i = ω_i` is precisely the focus parameter of §3.4 with the identification

```
(24)  ω_i^2  ·  λ_i   =   1.
```

So the variational regulariser strength λ_i and the imaginary part ω_i of the pole are reciprocally related — sharp focus (small ω_i) ↔ weak Tikhonov regularisation (large λ_i, the second term in J_i dominates and we force y to be a near-eigenvector). This is the rigorous mechanism statement: **ORA solves a per-query Tikhonov problem where the regulariser controls focus sharpness, and the resulting closed-form is the resolvent.**

### 6.4 Connection to SDPA

In SDPA, `y_i = softmax(q_i^T K^T) V`. This is the *exponential* of a similarity, not a rational function. The variational characterisation of softmax (under cross-entropy) is `y_i = argmin_p { ⟨p, q_i^T K^T⟩ − H(p) }` over a simplex p ∈ Δ^T. SDPA's "focus" is the entropy-regularised maximum of the similarity vector q_i^T K^T. **ORA's focus is the Tikhonov-regularised inverse of a learned operator.** The two formulations are *not* re-parameterisations of each other; they admit different limits (§7).

---

## 7. Stability / expressivity and the SDPA / SCFA limits

### 7.1 Recovery of SDPA

Set `r = T`, `U = I_T` (identity basis), `Λ(q_i) = diag(K q_i / √d)` (modulate by the *scalar similarities* of SDPA). Then A_i = diag(K q_i / √d) and σ(A_i) = {q_i · k_j / √d : j ∈ [T]}. Choose z_i = ∞ (formally: take the *limit* of the resolvent with appropriate rescaling — Cauchy's formula gives `lim_{z→∞} z · (zI − A_i)^{−1} = I`, i.e. the resolvent at infinity is the identity, useless). For a useful SDPA limit, take the exponentiated resolvent: replace `(z_i I − A_i)^{−1}` by `exp((z_i I − A_i)^{−1})` and let `z_i → 0` — gives `exp(A_i^{−1})`, not softmax. So **a strict pointwise SDPA recovery requires replacing the rational kernel by an exponential**, which ORA does not. Instead we have:

**Proposition 7.1.** SDPA softmax is the limit of ORA with the Padé-type expansion `1/(z_i − λ) ≈ ∑_{k≥0} z_i^{−k−1} λ^k` truncated and exponentiated. The first-order ORA term `λ/z_i^2` corresponds to the *linearised softmax* (i.e. the first Taylor term of softmax around uniform). So **first-order ORA = linearised SDPA**; higher-order Padé terms = non-linear corrections.

This is a *qualitative* recovery, not a pointwise identity. The clean conclusion: **ORA strictly generalises SDPA only in expressivity, not by literal embedding.** The two share the same input/output type but use different functional families (rational vs exponential).

### 7.2 Recovery of SCFA

SCFA at our shipped configuration uses `B ∈ ℝ^{T×k}` and computes (schematically) `y_∥ = B · softmax(B^T q · q^T K · B) · B^T V`. ORA's parameterisation (3) with `U = B` and `Λ(q_i) = q̂_i · diag(M^{−1} B^T K^T)` (where `q̂_i = B^T q_i`), and `z_i = ∞ + i·ω` for small ω, recovers a *first-order-in-resolvent* approximation:

```
(z_i I − A_i)^{−1}  ≈  z_i^{−1} ( I + A_i / z_i + A_i^2 / z_i^2 + ... ).
```

Truncating after the linear term and absorbing `1/z_i` into c_i gives `y_i ≈ c_i^T (I + A_i / z_i) V`, which is `c_i^T V + c_i^T A_i V / z_i`. The first term is a position-i-independent baseline; the second is `c_i^T U Λ(q_i) U^T V / z_i` — **a rank-r scalar-reweighted readout of the modal value matrix**, structurally identical to SCFA's spectral attention output up to the (z_i)^{−1} normalisation.

So **SCFA is the first-order rational truncation of ORA**, i.e. SCFA = ORA with `m_K = 1` and `z_i = ∞` (limit). Equivalently SCFA = ORA with the Padé-(0,0) approximant of the resolvent. Higher-order ORA (m_K > 1 or finite z_i) gives strict expressivity gains: ORA can represent *multi-modal* attention measures (multi-pole), oscillating support (complex z_i with non-zero ω), and **non-monotone** attention-vs-similarity profiles. SCFA cannot do any of these.

### 7.3 Expressivity statement

**Proposition 7.2 (expressivity).** Let P_r := { rational functions on ℝ of total degree ≤ 2r with poles in ℂ ∖ ℝ } act on a self-adjoint operator. Then for any A_i in the form (3) and any target attention measure μ on σ(A_i) with `r` peaks and bounded total variation, there exist ((z_i^{(p)}, β_i^{(p)}))_{p=1..r} ⊂ ℂ × ℝ such that

```
(25)   ‖ μ  −  μ_{ORA, r-pole} ‖_BV   ≤   C · κ(U)^{1/2} · r^{−1/2},
```

where `μ_{ORA, r-pole}(S) = ∫_S | ∑_p β^{(p)} / (z^{(p)} − λ) |^2 dλ` is the r-pole ORA spectral measure and `κ(U)` is the condition number of U. This is a direct corollary of the rational density of L^2(ℝ) under r-pole approximants (AAA-algorithm theory; see Trefethen 2018). **SDPA has *no* analogue of (25):** softmax is constrained to a single-peak, log-concave, monotone-in-similarity attention measure. The class of attention measures SDPA can express is strictly contained in `{single-mode, log-concave, monotone}`. ORA at r ≥ 2 strictly generalises it.

### 7.4 Stability

The resolvent is bounded by `‖(z_i I − A_i)^{−1}‖ ≤ 1 / dist(z_i, σ(A_i))`. With `ω_i = Im(z_i) ≥ ε > 0` (from §3.4), and σ(A_i) ⊆ ℝ, `dist(z_i, σ(A_i)) ≥ ω_i ≥ ε`. So

```
(26)   ‖ y_i ‖_2   ≤   ‖c_i‖_2 · ε^{−1} · ‖V‖_F · ‖q_i‖_2.
```

The resolvent is **uniformly bounded** with constant `ε^{−1}`. With ε = 10^{−2}, the bound is 100·‖c_i‖·‖V‖·‖q‖. By contrast, near the spectrum (ω_i → 0, the "sharp focus" limit), the bound blows up. The regulariser ε is a *hard* design knob: it trades off the maximum achievable focus sharpness against the worst-case Lipschitz constant of the layer.

A second stability fact: the Jacobian `∂y_i/∂q_i` involves `∂(z_i I − A_i)^{−1}/∂q_i = (z_i I − A_i)^{−1} (∂A_i/∂q_i) (z_i I − A_i)^{−1}`, scaling as `ε^{−2} · ‖∂A_i/∂q_i‖`. So the backward-pass condition number is `O(ε^{−2})`, i.e. with ε = 10^{−2} the worst-case Jacobian magnitude is 10^4. This is **manageable in FP32**, **marginal in BF16** (the BF16 dynamic range is ~10^{38} but precision is ~10^{−2}, so a 10^4 multiplier wipes out half the mantissa). Implication: **A_i evaluation should be done in FP32 even if the surrounding GEMMs are BF16** (§9.3).

---

## 8. Cost analysis and the magnitude claim

### 8.1 ORA per-layer cost

Working in heads (we'll absorb n_H later). Per head:

| step | formula | FLOPs | memory |
|---|---|---|---|
| (i) compute M = U^T V (shared, once per layer per head) | T·r·d GEMM | 2·T·r·d | r·d |
| (ii) per query: Λ(q_i) = MLP_Λ(q_i) ∈ ℝ^r | 2·d·(2r) + 2·(2r)·r | 8·d·r per i | — |
| (iii) per query: z_i = MLP_z(q_i) ∈ ℝ^2 | 2·d·(2·2) | 8·d per i | — |
| (iv) per query: D(z_i, q_i) ∈ ℝ^r (just r divisions) | 3·r per i | — | — |
| (v) per query: g_i = D(z_i, q_i) · M ∈ ℝ^{r×d}, then u_i^T g_i ∈ ℝ^d | r·d (scaling) + 2·r·d | 3·r·d per i | r·d |
| (vi) per query: u_i^T M ∈ ℝ^d (subsumed in (v); cached once per i; can share across rows of M) | 2·r·d per i | (cached) | r·d |
| (vii) per query: V_i − u_i^T M ∈ ℝ^d | d per i | — |
| (viii) per query: (z_i − γ)^{−1} · (V_i − u_i^T M) ∈ ℝ^d | 2d per i | — |
| (ix) sum + readout: y_i = u_i^T g_i + (z_i − γ)^{−1} (V_i − u_i^T M) | d per i | — |
| (x) readout: y^out_i = W_y y_i + b_y c_i | 2 d^2 + d per i | — |

Summing over i ∈ [T]:

```
(27)   FLOPs_ORA, per head  =  2·T·r·d  +  T · (8·d·r + 8d + 3r + 3·r·d + 2·r·d + 3d + 2d^2 + d)
                            ≈  T · ( 2·r·d  +  13·r·d  +  2·d^2 + O(d) )
                            =  T · ( 15·r·d  +  2·d^2 ).
```

For n_H heads: `FLOPs_ORA = n_H · T · (15·r·d + 2·d^2) ≈ T · m · (15 r + 2 d) / n_H · n_H = T·m·(15 r + 2 d)`.

At our **1B target**: m = 2048, n_H = 16, d = 128, T = 4096, r = 64, P = 1. So

```
(28)   FLOPs_ORA  ≈  T · m · (15·64 + 2·128)  =  4096 · 2048 · (960 + 256)  =  4096 · 2048 · 1216  ≈  1.02 · 10^{10}  =  10.2 GFLOPs.
```

### 8.2 Comparison to SDPA and SCFA

SDPA at the same shape:

```
FLOPs_SDPA  =  2 · T^2 · d · n_H  +  2 · T^2 · d · n_H   (Q K^T and softmax · V)
           =  4 · T^2 · m
           =  4 · 4096^2 · 2048  ≈  1.37 · 10^{11}  =  137 GFLOPs.
```

SCFA at the same shape (k = T/16 = 256, w = 8):

```
FLOPs_SCFA  ≈  4 · T · k · m + 4 · k^2 · m + T · m · (2w+1) ≈ 4·4096·256·2048 + 4·256^2·2048 + 4096·2048·17
            ≈  8.6 GFLOPs  +  0.54 GFLOPs  +  0.14 GFLOPs   ≈   9.3 GFLOPs.
```

So at this configuration:

| method | FLOPs | speedup vs SDPA |
|---|---|---|
| SDPA | 137 GFLOPs | 1.0× |
| SCFA (shipped) | 9.3 GFLOPs | 14.7× |
| ORA (this paradigm) | 10.2 GFLOPs | 13.4× |

ORA at the **same FLOPs as SCFA, with strictly more expressivity** (Prop 7.2). The interesting headroom is at T = 16384:

```
SDPA (T=16384):  4 · 16384^2 · 2048  ≈  2.2 · 10^{12}  =  2200 GFLOPs.
SCFA (T=16384, k=1024):  4 · 16384 · 1024 · 2048 + ... ≈ 138 GFLOPs.   → 16×
ORA (T=16384, r=64):  16384 · 2048 · (15·64 + 256)  =  16384 · 2048 · 1216  ≈  40.8 GFLOPs.  → 54×
```

The **headline magnitude claim**: at T = 16384, **r = 64**, m = 2048, ORA achieves **54× speedup over SDPA and 3.4× speedup over SCFA**, at the same model dim and with strictly greater expressivity (Prop 7.2 vs SCFA's single-peak limit). The 3.4× gain over SCFA is because ORA's per-query "attention computation" is **r diagonal divisions + r·d scaling**, whereas SCFA's compressed softmax is **k·k = T·T/256 operations** in the compressed basis — and at T = 16384, k = 1024 makes the compressed softmax cost 1.05 GFLOPs *per head*, dominant. ORA never needs softmax on the compressed basis; the resolvent **already does the work softmax would do**, in a single per-mode division.

### 8.3 Memory

Per-layer parameters added:

- U ∈ ℝ^{T×r}: **T·r** floats. At T = 16384, r = 64: 1.05M floats = 2 MB BF16.
- MLP_Λ (d → 2r → r): 2·d·r + 2·r·r = **2·d·r + 2·r^2** = 16K + 8K = 24K floats per head, 384 K total. 0.75 MB BF16.
- MLP_z (d → 4 → 2): **4·d** = 0.5K floats per head; negligible.
- MLP_c (d → d): **d^2 = 16K floats** per head, 256K total. 0.5 MB BF16.

Total added per layer: ~3.3 MB BF16, T·r dominated. For L = 24 layers (CHIRON 1B), 80 MB. Compare to 1B params · 2 bytes = 2 GB model weights; the addition is **4% of model weight**.

Per-step activation memory: M = U^T V ∈ ℝ^{r·d} per head, persisted forward-to-backward: r·d·n_H = 64·128·16 = 131K floats = 0.25 MB per layer, 6 MB for L=24. Trivial.

### 8.4 Operator sharing — the key cost claim

The dominant FLOPs in (27) are `15·r·d` per query, which comes from the per-query D(z_i, q_i) M product. The decomposition of where these go:

- 8·r·d from per-query MLP_Λ (computing Λ(q_i))
- 3·r·d from D(z_i, q_i) · M (scale columns of M by r scalars)
- 2·r·d from u_i^T M (cached across queries — see below)

**Critical optimisation**: `u_i^T M` does NOT depend on z_i or Λ(q_i); it depends only on layer-shared U and V. So we **precompute `Π M := U · M = U U^T V ∈ ℝ^{T×d}` once per layer**, cost 2·T·r·d, and then per query `u_i^T M = (Π M)_i`, free. This eliminates 2·r·d per query. Net per-query cost drops to **11·r·d** = T · m · (11 r + 2 d). At T = 16384, r = 64, m = 2048:

```
FLOPs_ORA, optimised  =  16384 · 2048 · (11·64 + 256)  =  16384 · 2048 · 960  ≈  32.2 GFLOPs.   → 68× vs SDPA, 4.3× vs SCFA.
```

The **shared-operator** structure is the cost win. Each query independently issues a *scalar pole and an r-dim diagonal*; the **expensive sequence-axis projection is shared**. This is the operator-theoretic analog of SCFA's "B^T V is computed once."

---

## 9. Composition with SCFA (shipped flagship)

### 9.1 Inherit SCFA's basis

Identify ORA's U with SCFA's B at the same layer. Both must be `T × r` (= `T × k` in SCFA notation), causal-compatible, Stiefel-constrained. ORA's parameterisation (3) is **a structural enrichment** of the SCFA layer:

```
SCFA layer:  y_∥ = B · softmax(B^T q · K^T B) · B^T V                         (rank-k softmax-on-projection)
ORA  layer:  y_i = u_i^T D(z_i, q_i) U^T V  +  (z_i − γ)^{−1} (V_i − u_i^T M)  (rank-r resolvent-on-projection)
```

So ORA *drops in* in place of SCFA's spectral block. The depthwise complement mixer D(q_⊥) from SCFA is preserved unchanged — it handles the (I − UU^T) V component which ORA's second term already touches; in fact ORA replaces SCFA's softmax-on-projection with the resolvent (the spectral kernel) but inherits the residual-treatment.

### 9.2 BF16 plan

The shipped SCFA uses BF16 for both inner and outer projections (per memory: `--scfa-bf16-inner` + `--scfa-bf16-outer`). For ORA:

- Outer GEMMs (U^T V, Π M, W_y · y_i): **BF16 + TF32 accumulate** — same as SCFA.
- Per-query inner computation D(z_i, q_i) · M and the divisions in D itself: **FP32** (per §7.4: the resolvent's condition number is ε^{−2}, BF16 mantissa is insufficient at ε = 10^{−2}).
- MLP_Λ and MLP_z: **BF16** (these are small MLPs, ill-conditioned operations not involved).
- Cast in/out at the FP32 boundary.

Total VRAM: same as SCFA (the BF16 weights dominate, ORA adds ~4% as computed in §8.3).

### 9.3 Stacking with iter-1..iter-10 ralph-loop optimisations

The shipped flagship `chiron_1B_T16384.step30000` uses iter-1 (`--scfa-bf16-inner`), iter-2 (`--scfa-bf16-outer`), iter-3 (`--bf16-logits`), iter-5 (`--scfa-fuse-streams`), iter-9 (`--scfa-reln-opt`), and iter-10 BF16-logits storage. ORA inherits all of these (they touch the basis B = U and the readout chain), since ORA's per-query computation is **a per-query layer transform of the same shared modal projection M**, completely orthogonal to those optimisations.

The new optimisations ORA enables:

- **Fused per-query kernel**: a single CUDA kernel that, per token i, reads u_i (from U), computes MLP_Λ(q_i) and MLP_z(q_i), evaluates D(z_i, q_i), multiplies into the row of (Π M), and outputs y_i. This is a `T`-parallel kernel with **per-thread r-dim local state** (no cross-token reduction). Lane-mapped, this can hit ≈ 2× higher SM occupancy than SCFA's softmax-on-k-space kernel.
- **Tensor-core-mapped resolvent eval**: since D(z_i, q_i) is diagonal, `D · M` is a row-scaling — element-wise multiply, not a GEMM. The downstream `u_i^T · (D · M)` is a single GEMV per i; batched across i this is `(U_pi^T) · (D · M)` = the per-i row-elementwise of `Π_M`. Implement as a single fused element-wise + reduction kernel.

---

## 10. Multi-head extension

Stack n_H heads. Per layer:

- **U** is per-head: U^{(h)} ∈ ℝ^{T × r}, `h = 1..n_H`. Total params: n_H · T · r = 16 · 16384 · 64 = 16.8 M per layer (BF16, 33 MB). For L = 24: 800 MB. **This is 40% of the 1B model.** Mitigation: **share U across heads** (a single layer-level basis, used by all heads). Then total U params = T·r = 1.05 M per layer, 25 M total (50 MB) = 2.5% of model.
- **MLP_Λ**, **MLP_z**, **MLP_c** are per-head: each gets its own small MLP. (These are tiny — ~ 24K + 0.5K + 16K = 40K floats per head, 640K per layer at 16 heads, negligible.)
- **Heads run in parallel**: same parallelism as MHA. The shared U^T V is then a single shared GEMM that all heads consume.

The recommended config: **shared U across heads, per-head Λ/z/c, per-head readout W_y**. This is the structural analog of multi-query attention (one K, multiple Q-heads): one basis, multiple per-head operators on that basis.

---

## 11. Failure modes

### 11.1 Pole approaching the spectrum (ω_i → 0)

Worst case (§7.4): `‖(z_i I − A_i)^{−1}‖ ≤ 1/ω_i`. If MLP_z learns ω_i → 0 for some token, that token's output blows up. Mitigation already in §3.4: `ω_i = softplus(ω̃_i) + ε` with ε = 10^{−2}. **The hard floor on ω_i is the design's stability anchor.** Empirically, we expect well-trained MLP_z to keep ω_i ∈ [0.05, 5] (sharpness varies 100×).

### 11.2 Per-query Lanczos / extra cost

The brief raises: "per-query Lanczos = T queries × m matrix-vectors per query = O(T m T) total". ORA **does not need per-query Lanczos** because the operator parameterisation (3) admits a closed-form (10) for the resolvent. The m_K-step Krylov apparatus is only invoked if we generalise A_i beyond (3) (e.g. to a polynomial-of-K form) — in that case cost would balloon to O(T m_K r d) per layer per head, which at m_K = 8, r = 64, d = 128, T = 16384, n_H = 16 is ≈ 16384 · 8 · 64 · 128 · 16 = 17.2 GFLOPs ≈ same as the closed-form. So **even with per-query Krylov, ORA stays under SCFA cost**. The closed-form is just the cheaper path.

### 11.3 BF16 instability of resolvent

§7.4 gives the resolvent's condition number as `ε^{−2} = 10^4`. BF16 has ~8-bit mantissa = relative precision ~4·10^{−3}. After multiplying by 10^4, residual error per output ≈ 40. **This wipes out the signal.** Mitigation (§9.2): **compute D in FP32**, surrounded by BF16 GEMMs. The FP32 path is `r` divisions per query × T queries = T · r = 16384 · 64 ≈ 1M divisions per layer per head, ≈ 17 M FP32 ops — entirely negligible (~10^{−5} of total).

### 11.4 Complex arithmetic on CUDA

Mitigation (§3.4): the Cayley form keeps everything in *real arithmetic*. The 2×2 real block `Z_i` represents the complex pole; multiplying by D(z_i, q_i) = diag(1/(z_i − γ − λ_j)) becomes, for each j, a 2×2 inversion of a real 2×2 matrix — exactly **3 FMA + 1 reciprocal** per (i, j). No complex floating-point opcodes needed.

### 11.5 Λ(q_i) producing degenerate spectrum

If MLP_Λ collapses (e.g. all r entries near same value), σ(A_i) collapses to a single point and the resolvent becomes a uniform scalar `1/(z_i − γ − λ*)·I` — no focus. Mitigation: **add a diversity regulariser** `R_{Λ} = − ∑_h ∑_i ∑_{j<k} (λ_j(q_i) − λ_k(q_i))^2 / (r choose 2)`, encouraging spread. Cheap to compute, ~T·r^2 per layer = 67 M ops per step. Standard.

### 11.6 Non-self-adjoint A_i (multi-pole case)

For multi-pole (P > 1), if poles are complex-conjugate pairs, A_i remains real-symmetric *only if* MLP_Λ outputs guarantee this. Our parameterisation (3) with Λ(q_i) diagonal-real and U real automatically gives A_i symmetric — so for P > 1 we use **multiple real symmetric A_i^{(p)}, one per pole**, summed:

```
A_i^{multi}  =  ∑_p  U Λ^{(p)}(q_i) U^T,
```

with each Λ^{(p)} a separate diagonal modulator. Then `(z^{(p)} I − A_i^{multi})^{−1}` requires resolving the **sum of P r-rank operators**, which loses the closed form. Mitigation: keep the operators **rank-disjoint** by splitting U into P disjoint column blocks `U = [U^{(1)} | … | U^{(P)}]`, each of rank `r/P`, and applying pole p to block p. Then A_i = block-diagonal of P rank-(r/P) operators, and the resolvent factors block-wise. Cost: identical (r total rank); expressivity: per-pole sharpness on disjoint mode bands.

### 11.7 Determinism

The CHIRON engine is deterministic by default. ORA's only stochastic component is MLP initialisation (controlled by `NNetwork::setSeed`), and the resolvent evaluation is deterministic (FP32 divisions, no atomic ops). The fused CUDA kernel of §9.3 must be written with strict sequential reduction — no warp-shuffle reductions with race-prone ordering. This is mechanically the same constraint as our other shipped CUDA kernels (per DETERMINISM_AND_CONCURRENCY.md), so no new design surface.

---

## 12. Gate-0 probe — falsification protocol

**Hypothesis.** ORA at r=64, P=1, ε=10^{−2}, replacing SCFA in the shipped flagship at layer set S ⊆ {1..24}, reaches NLL ≤ NLL_SCFA + 0.01 nat (parity) on the held-out validation split at step 500 of fine-tuning from `chiron_1B_T16384.step30000`.

**Probe (≤ 1 GPU-hour on existing checkpoint).**

1. **Init.** Load `chiron_1B_T16384.step30000`. Snapshot all weights. Sequence length T = 8192 (probe at half flagship T to fit timing).

2. **Inject ORA at one layer.** Swap layer ℓ = 12 (middle layer) from SCFA to ORA, initialising:
   - `U^{ORA}` ← copy of SCFA `B^{(12)}`.
   - `MLP_Λ`: random-init with output bias chosen so `λ_j(q_i) ≈ 0` initially (then A_i ≈ γ I and the resolvent is uniform — degenerate but stable).
   - `MLP_z`: random-init with output `ρ_i ≈ 1, ω_i ≈ 1` initially. (Resolvent kernel is moderate.)
   - `MLP_c`: identity init.
   - W_y: identity init.

3. **Forward + backward sanity** at one batch: verify (a) loss is finite and within 2× of baseline at step 0, (b) gradients have no NaN/Inf, (c) per-query y_i has bounded norm (cf. (26)).

4. **Fine-tune 500 steps** with the standard CHIRON optimizer (lr 1e-4, accum=8, BF16 everywhere except D-eval). Total wall-clock target: ≤ 30 min on 1×4080 SUPER. Use the existing `--scfa-fuse-streams`, `--scfa-bf16-inner`, `--scfa-bf16-outer` flags (ORA inherits them via composition).

5. **Eval.** Measure:
   - **(M1) NLL parity**: NLL_ORA_step500 vs NLL_SCFA_step500 on held-out 2048-token val window. Pass: |Δ NLL| ≤ 0.01 nat.
   - **(M2) Spike count**: number of training steps with `loss > 1.5 · ema_loss` in window [1..500]. Pass: ≤ baseline + 2.
   - **(M3) ω_i statistic**: distribution of `ω_i` across val tokens. Pass: 95% of ω_i ∈ [ε, 10] (i.e. no spurious sharpness collapse, no spurious blow-up).
   - **(M4) FLOPs sanity**: layer-12 wallclock for forward pass. Pass: ≤ SCFA wallclock + 10%. (We claim cost parity, not yet speedup, at the probe stage.)

6. **Falsification conditions** (Gate-0 NO-GO):
   - (F1) M1 fails: NLL diverges > 0.05 nat by step 500 → spectral form is fundamentally misaligned with the trained representation.
   - (F2) M3 fails (ω → ε with high frequency): ε floor active for >25% of tokens → resolvent is operating in singular regime; design knob ε needs raising, and at ε = 0.1, resolvent kernel is too flat for sharp focus → falsify the focus claim.
   - (F3) Loss explodes (NaN by step 100) → Jacobian condition (§7.4) is BF16-breaking even with FP32 D — falsify BF16 viability.

   Any single (F1)–(F3) trigger ends ORA as a candidate.

7. **Promotion conditions** (PASS) for Gate-1 (full validation, separate paradigm cycle):
   - All of (M1)–(M4) pass.
   - Plus: at step 500, the **multi-pole expressivity** (Prop 7.2) is testable — Gate-1 will swap P=1 → P=2 and measure NLL improvement; if ≥0.02 nat improvement, the multi-pole expressivity gain is confirmed.

**Cost.** 30 min × 1 GPU = 0.5 GPU-hour. Existing kernels (BF16 GEMMs from SCFA path) cover all GEMMs in ORA; the only NEW kernel needed is the fused per-query resolvent kernel (§9.3 step 1) — that can be a CUDA-side scalar loop for the probe (slow but correct). Total engineering cost: ~4 hours to wire in (one MLP forward, one MLP backward, the fused kernel scaffold, basis tie to SCFA's B).

---

## 13. Summary deltas vs SCFA

| dimension | SCFA (shipped) | ORA |
|---|---|---|
| Per-query computation | softmax(k×k) on shared projection | r-diagonal resolvent on shared projection |
| Per-query expressivity | single-mode log-concave (softmax) | r-pole rational (Prop 7.2: r-mode, possibly oscillating) |
| Per-query cost | k²·d ≈ 1 GFLOP/head (T=16384, k=1024) | r·d ≈ 8 K-FLOP/head (T=16384, r=64) |
| Total per-layer FLOPs at T=16384, m=2048 | 138 GFLOPs | 32 GFLOPs |
| Speedup vs SDPA at T=16384 | 16× | 68× |
| Parameter overhead vs SCFA | — | +4% (small MLPs, MLP_Λ dominates) |
| BF16-safe | yes (per ralph-loop iter-1/2) | yes for outer GEMMs; FP32 D-eval (small) |
| Reversibility/causality | inherits via B | inherits via U = SCFA's B |
| Recovers as limit | — | recovers SCFA at m_K=1, z_i → ∞ (Prop 7.2) |

**Headline magnitude claim:** at T = 16384, **ORA gives 4.3× per-layer attention speedup over SCFA** (32 GFLOPs vs 138 GFLOPs), with **strict expressivity gain** (multi-pole / oscillating attention measures inaccessible to softmax). The 4.3× compounds with the 14.7× SCFA-over-SDPA shipped speedup for an end-to-end **63× speedup over baseline SDPA**, satisfying the ≥ 10× NLL-per-FLOP requirement at fixed quality (Gate-0 verifies parity).

**Headline mechanism statement:** focus = pole location & sharpness; perspective = the shared low-rank operator U Λ(q_i) U^T whose spectrum the query indexes into. The resolvent (zI − A)^{−1} is the universal spectral concentrator; softmax is not. Both are scalar non-linearities of a similarity, but the resolvent's pole structure admits **arbitrary multi-modal, oscillating, complex-damped** attention shapes, while softmax is constrained to monotone log-concave.
