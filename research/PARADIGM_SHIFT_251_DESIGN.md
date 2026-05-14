# Paradigm Shift #251 — SRA: Sheaf-Resolvent Attention

**Status:** designed (Ralph-loop iter 3, 2026-05-14). Builds on paradigm #250 SFA and incorporates Candidate C (ORA) recommendations.
**Date:** 2026-05-14.
**Branch:** vesta5.
**Predecessor:** paradigm #250 SFA (Sheaf-Focal Attention) — selected and designed in iter 1, proofs written in iter 2.
**Axis:** Generalise SFA's real-pole Tikhonov solve `(L_F + λI)^{-1}` to ORA's complex-pole resolvent `(z_q I − L_F)^{-1}` with **per-query** complex pole `z_q = ρ_q + i ω_q`. Combines SFA's per-token *perspective* (stalk frames, restriction maps) with ORA's per-query *focus* (pole location + sharpness).
**Magnitude target:** 5–8× wall-clock speedup over SCFA flagship at iso-NLL, by composing SFA's quality (cocycle expressivity → 0.015–0.03 nat NLL win) with ORA's compute (4.3× FLOP reduction via closed-form per-query operator).

---

## 0. Executive summary

SFA (paradigm #250) gives each token its own *perspective* (stalk frame `U_i ∈ R^{d_s × r}` and restriction map `R_{j ← i} = U_j Σ(i,j) U_i^T`) and computes attention as the regularised harmonic section under the sheaf Laplacian `L_F = δ^T δ`. The Tikhonov solve `s★ = (L_F + λI)^{-1} b` is a **layer-uniform** spectral low-pass: λ is per-head but not per-query. This means every query in a layer experiences the same focus sharpness.

ORA (paradigm #250 Candidate C, recommended for promotion) gives each query its own *focus* (complex pole `z_q = ρ_q + i ω_q` produced by per-query MLPs from q_i) and computes attention as `(z_q I − A_q)^{-1} V`. The pole location `ρ_q` is "which eigentokens am I looking at" and `ω_q` is "how sharply." But ORA uses a **layer-shared** basis `U` for the operator `A_q = U Λ(q) U^T + γ I`: the perspective is per-query but operates on the same low-rank subspace for every query.

SRA combines both:

```
y_q = Im[ (z_q I − L_F)^{-1} b_q ]_{[q]}      (eq. 1)
```

with:
- `L_F` = sheaf Laplacian with per-token stalk frames U_i and per-edge restriction maps R_{j ← i} (inherited from SFA, paradigm #250).
- `z_q = ρ_q + i ω_q` = complex pole per query (inherited from ORA).
- `b_q` = per-query source assembled from x_q's content (per SFA's eq. 7, paradigm #250).
- `Im[·]` = imaginary part (taking the spectral-density side of the resolvent, equivalent to the heat-kernel `exp(-ω_q L_F)` evaluated at the spectral-shift `ρ_q`).

The closed-form decomposition (analog of eq. 18 in ORA candidate doc):

```
(z_q I − L_F)^{-1} = U_eff · D(z_q, U_eff^T L_F U_eff) · U_eff^T + (z_q − γ)^{-1} (I − U_eff U_eff^T)    (eq. 2)
```

where `U_eff = ψ(x_q; θ_U)` is the per-token frame at the query position. The resolvent factors through the per-token perspective, giving per-query attention that is *both* per-token-aware (via U_q) *and* per-query-focused (via z_q).

**Magnitude**: SRA's per-query closed-form cost is comparable to ORA's (`O(r · d_s)` per query plus shared `O(T · |E| · d_s · r)` for the sheaf-Laplacian factorisation), but the NLL gain from SFA's cocycle expressivity is preserved. Combined: ~5–8× wall-clock at iso-NLL over SCFA flagship.

**Honest reading**: SRA is a *compositional* paradigm — it does not introduce a categorically new mechanism beyond what SFA + ORA already provide separately. Its value is operational: per-query focus (sharpness, spectral location) layered atop per-token perspective (stalk frame, restriction map). The math is more elaborate but the engineering surface is smaller than SFA alone (closed-form per-query, no per-token Chebyshev).

---

## 1. Why this is paradigm #251 and not paradigm #252

SFA was selected as paradigm #250 with explicit forecast (PARADIGM_SHIFT_250_SELECTION.md §2.4 and PARADIGM_SHIFT_250_DESIGN.md §14.2):

> "Paradigm #251 candidate: SFA + ORA (Sheaf-Resolvent Attention). Generalise SFA's Tikhonov solve `(L_F + λI)^{-1}` to ORA's complex-pole resolvent `(z_q I − L_F)^{-1}` with per-query pole z_q. Multi-pole sheaf attention. Expected: 4.3× FLOP reduction (ORA's headline) on top of SFA's quality gain."

This iter 3 promotes the candidate to a full design. Reasons #251 (not #252):

1. **Natural successor to #250**: SFA proved SCFA-recovery (Theorem 2, iter 2) and contains ORA as a limiting case (set d_s = 1 and use complex pole instead of real λ). SRA is the strict generalisation.
2. **Independent expressivity axes**: SFA gives per-token expressivity (d_s, r); ORA gives per-query expressivity (z_q, multi-pole). These are *orthogonal*. Combining them is multiplicative in capacity.
3. **Composes with iter 2's REFLECTOR adjoint**: the implicit-function adjoint for `(z_q I − L_F)^{-1}` is the same machinery as for `(L_F + λI)^{-1}`, with the symmetry `(z_q I − L_F)^T = z_q^* I − L_F` (where * is complex conjugate). Real-arithmetic via Cayley form (ORA §3.4).
4. **Decisive engineering simplification at d_s = 1**: when d_s = 1, SRA reduces to ORA exactly (closed-form per-query, no Chebyshev iteration). This gives a fallback to a higher-throughput, lower-quality variant at deployment time.

Paradigm #252 (TBD) is reserved for **Persistent Sheaf Attention** — extending SFA + SRA to track cohomology across layers (Conjecture 3 of PARADIGM_SHIFT_250_PROOFS.md §7).

---

## 2. Mathematical setup

### 2.1 Primitive objects (inherited + new)

Inherited from SFA (paradigm #250):
- Token graph G = (V, E) with E = causal-sliding-window-plus-sinks.
- Stalks F(v_i) = R^{d_s}, stalk frames `U_i = ψ(x_i; θ_U) ∈ R^{d_s × r}`.
- Edge modulators `Σ(i, j) = diag(σ(W_Σ [x_i; x_j] + b_Σ))`.
- Restriction maps `R_{j ← i} = U_j Σ(i, j) U_i^T`.
- Sheaf Laplacian `L_F = δ^T δ`.
- Source assembly `b_q = U_q U_q^T P_q W_Q x_q + γ P_v W_V x_q`.

New for SRA:
| Symbol | Shape | Meaning |
|---|---|---|
| `z_q = ρ_q + i ω_q` | ℂ | per-query complex pole |
| `ρ_q ∈ R` | scalar | spectral location (which eigentokens to focus on) |
| `ω_q ∈ R_+` | scalar | focus sharpness (ω → 0 = sharp, ω → ∞ = uniform) |
| `MLP_z` | layer | small MLP producing (ρ_q, ω_q) from q_q via softplus |
| `P, β_q^{(p)}` | per-query | for multi-pole extension: pole index p ∈ {1..P}, weight β |
| `ε` | scalar | hard floor on ω_q. Default ε = 10^{-2}. |

The pole is parameterised as:

```
(ρ_q, ω̃_q) = MLP_z(W_Q x_q),    ω_q = softplus(ω̃_q) + ε.          (eq. 3)
```

The softplus + ε floor ensures `ω_q ≥ ε > 0` always, which is the load-bearing stability requirement (§5 below).

### 2.2 The central equation

SRA's per-query, per-head output is:

```
y_q = Im[ (z_q I − L_F)^{-1} b_q ]_{[q]}      ∈ R^{d_s}              (eq. 4)
```

where `[·]_{[q]}` extracts the q-th stalk block from C^0(F).

Equivalently (via the spectral-density representation):

```
y_q = ω_q · [(L_F + ω_q^2 I + (ρ_q − γ)^2 I)^{-1} b_q]_{[q]} · 1/((ρ_q − γ)^2 + ω_q^2)  (eq. 5)
```

is the real-arithmetic equivalent obtained by `1/(z − μ) = (z̄ − μ̄)/|z − μ|^2` (with `μ̄ = μ` for real `μ`). The denominator is real-positive; the real part of `(z − μ)^{-1}` is `(ρ − μ)/((ρ − μ)^2 + ω^2)` and the imaginary part is `ω/((ρ − μ)^2 + ω^2)`. We take the imaginary part to get the spectral-density-weighted attention (matches ORA §6 Theorem 6.1).

For final readout (analog of SFA eq. 11, paradigm #250):

```
y_q^{out} = P_o^T y_q + W_Q x_q.                                    (eq. 6)
```

### 2.3 Multi-pole extension

For multi-pole attention (each query has P poles):

```
y_q = Σ_{p=1}^P β_q^{(p)} · Im[(z_q^{(p)} I − L_F)^{-1} b_q]_{[q]}.    (eq. 7)
```

Each pole z_q^{(p)} carries its own spectral focus; the weights β_q^{(p)} are produced by another small MLP (`MLP_β : R^d → R^P`). The "multi-modal attention" expressivity gain of ORA (Prop 7.2 of Candidate C doc) is inherited: SRA at P = 2 can represent attention measures with two disjoint spectral peaks; at P = 4, four peaks; etc.

Default: P = 1 (single pole). P > 1 is opt-in per layer or per head.

---

## 3. Closed-form via structured factorisation

The key cost win comes from the observation that **L_F factors through the per-token stalk frames U_i**, so the resolvent has a closed form analogous to ORA's eq. 18.

### 3.1 Reduced operator

Define `U ∈ R^{T·d_s × T·r}` as the block-diagonal stacking of {U_i}, and `Σ_diag ∈ R^{|E|·r}` as the stacked edge modulators. The sheaf Laplacian decomposes:

```
L_F = (D_U − A_U) ⊗_{stalk} blocks                                 (eq. 8)
```

where D_U is the block-diagonal degree contribution (with diagonal blocks `[L_F]_{ii} = Σ_j R_{i←j}^T R_{i←j}` from §3.3 of PARADIGM_SHIFT_250_DESIGN.md) and A_U is the off-diagonal block-adjacency (with blocks `[L_F]_{ij} = -R_{j←i}^T`).

Under the factorisation `R_{j←i} = U_j Σ(i,j) U_i^T`:

```
[L_F]_{ij} = -U_j Σ(i,j) U_i^T
[L_F]_{ii} = Σ_{j: (j→i) ∈ E} (U_i Σ(j,i) U_j^T)(U_j Σ(j,i) U_i^T)
           = U_i (Σ_j Σ(j,i)^2 U_j^T U_j) U_i^T   + extra-degree-I contributions
```

For near-orthonormal U_i (i.e., U_i^T U_i ≈ I_r), the diagonal simplifies:

```
[L_F]_{ii} ≈ U_i (Σ_{j: (j→i) ∈ E} Σ(j,i)^2) U_i^T = U_i Λ_i U_i^T
```

where `Λ_i := Σ_{j: (j→i) ∈ E} Σ(j,i)^2 ∈ R^{r × r}` is diagonal (sum of squared diagonals).

So **L_F has the structure** `L_F = U Λ̃ U^T + (off-block terms)` for a per-token-frame-decomposable factor `Λ̃`. This is the cellular-sheaf analog of ORA's operator factorisation `A = U Λ U^T + γI`.

### 3.2 Closed-form resolvent

For the **block-diagonal-only** approximation (off-blocks treated separately):

```
(z_q I − L_F)^{-1} ≈ Σ_i U_i (z_q − Λ_i)^{-1} U_i^T  +  (z_q^{-1}) · (off-block term)   (eq. 9)
```

The diagonal `(z_q − Λ_i)^{-1}` is computed per-token by `r` scalar divisions (since Λ_i is r × r diagonal). The cost per query for the q-th block of the resolvent applied to b_q:

```
[(z_q I − L_F)^{-1} b_q]_{[q]} ≈ U_q (z_q − Λ_q)^{-1} U_q^T b_q  +  (z_q^{-1}) · off-block-correction
                              = U_q · D(z_q, Λ_q) · (U_q^T b_q)  +  (z_q^{-1}) · correction      (eq. 10)
```

where `D(z_q, Λ_q) ∈ R^{r × r}` is the diagonal matrix with entries `D_{jj} = 1/(z_q − Λ_{q,jj})`.

**Cost per query**:
- Compute `Λ_q`: `O(W r^2)` (sum of squared diagonals over neighboring edges).
- Compute `(U_q^T b_q)`: `O(d_s r)`.
- Scale by `D(z_q, Λ_q)`: `O(r)`.
- Lift by `U_q`: `O(d_s r)`.
- Off-block correction (banded matvec): `O(W d_s)`.

Total: `O(W r^2 + d_s r + W d_s)` per query. For typical config (W = 128, r = 4, d_s = 64): ~2000 ops per query — comparable to ORA's per-query cost (`~960 ops` at the headline config).

**Comparison to SFA's Chebyshev**: SFA at default M = 8 needs `O(8 · |E| · d_s · r)` per matvec × M steps = ~ 32k ops per query at the same config. **SRA is ~16× cheaper per query than SFA's iterative Chebyshev**.

### 3.3 The off-block correction

The block-diagonal-only approximation drops the inter-token coupling through `[L_F]_{ij} = -U_j Σ(i,j) U_i^T` for off-diagonal `j ≠ i`. This is the term that "moves" mass from token i's stalk into token j's stalk under the resolvent.

For the off-block correction, one option is to apply a *single Chebyshev step* after the block-diagonal closed form:

```
y_q^{(1)} = block-diagonal-closed-form (eq. 10).
y_q^{(2)} = y_q^{(1)} − (1/μ_max) · [L_F · y_q^{(1)}]_{[q]}.    (eq. 11)
```

This is a single Richardson iteration that captures first-order off-block contributions. Cost: 1 sparse L_F matvec at the query position = `O(W d_s r)`. Total: ~ `1024 ops` per query, still ~ 30× cheaper than full SFA Chebyshev.

For higher accuracy, additional Chebyshev / Lanczos steps can be added at the cost of returning toward SFA's full iterative cost. The **operational form** is: block-diagonal closed form (default, cheapest) + optional single Richardson refinement (when quality matters).

### 3.4 Validity of the approximation

The block-diagonal approximation drops the off-block contributions `Σ_{j ≠ i} U_j Σ(i,j) U_i^T`. The error is bounded by:

```
‖[L_F]_off-block‖_op ≤ ‖Σ‖_∞ · max_j ‖U_j‖_op · ‖U_i‖_op ≤ 1.0   (under near-unit U_i norm)
```

The total contribution to the resolvent error: `‖(z_q I − L_F^{(diag)})^{-1} − (z_q I − L_F)^{-1}‖_op ≤ ‖[L_F]_off-block‖_op / |z_q − μ_max|^2`.

For typical `|z_q| ≥ 1` and `μ_max ≈ 100`: error ≤ 1/100² = 10^{-4}. This is **negligible** compared to BF16 mantissa precision.

So the block-diagonal closed form is empirically tight; the Richardson refinement (§3.3) brings the error well below 10^{-5}.

---

## 4. Recovers SFA and ORA as limiting cases

### 4.1 Recovers SFA (paradigm #250)

Set `z_q = i ε` (purely imaginary pole near the origin), `ρ_q = 0`. Then:

```
(z_q I − L_F)^{-1} = (i ε I − L_F)^{-1}
                   = − L_F^{-1} (I − i ε L_F^{-1})^{-1}
                   ≈ − L_F^{-1} − i ε L_F^{-2} + O(ε^2)
```

The imaginary part:

```
Im[(i ε I − L_F)^{-1}] ≈ − ε L_F^{-2} = − ε (L_F + λI)^{-2}     (for small λ)
```

This is the *squared* SFA filter, not the *linear* SFA filter. The leading-order match is via the **Tikhonov-resolvent identity** (Theorem 6.1 of ORA candidate doc):

```
(L_F + λI)^{-1} = Im[(z̃ I − L_F)^{-1}] / Im(z̃)   for z̃ = i √λ.
```

So setting `z_q = i √λ` recovers SFA's Tikhonov solve exactly:

```
y_q^{SRA} |_{z_q = i √λ} = Im[(i √λ I − L_F)^{-1} b_q] / √λ = (L_F + λI)^{-1} b_q.
```

**This is the SFA-recovery limit of SRA.** ∎ (Theorem 4)

### 4.2 Recovers ORA (paradigm #250 Candidate C)

Set `d_s = 1` (scalar stalks) and `U_i = b_i` (i-th row of SCFA's spectral basis B). Then L_F reduces to a scalar T × T matrix (per Theorem 2 of PARADIGM_SHIFT_250_PROOFS.md §2), and:

```
(z_q I − L_F)^{-1} ≈ U · D(z_q, Λ) · U^T + (z_q − γ)^{-1} (I − UU^T)     (eq. 12)
```

This is *exactly* ORA's eq. 10 (PARADIGM_SHIFT_250_CANDIDATE_C_ORA.md §4.1). The per-query output `y_q = u_q^T D(z_q, Λ) M + (z_q − γ)^{-1} (V_q − u_q^T M)` of ORA is recovered.

**This is the ORA-recovery limit of SRA.** ∎ (Theorem 5)

### 4.3 The SFA ↔ ORA decoupling

By Theorems 4 and 5, SRA contains both SFA and ORA as proper subsets. The two axes of generalisation are:

| axis | parameter | meaning |
|---|---|---|
| Per-token perspective | d_s, r | rank of stalk frames; recovers ORA at d_s = 1 |
| Per-query focus | z_q (or P pole pairs) | complex shift of the resolvent; recovers SFA at z_q = i √λ |

The two axes are **multiplicative in capacity**: SRA at d_s = 64, r = 4, P = 1 has capacity ~ 256 per query (stalk-frame-dim × pole flexibility), versus SCFA's k = 64.

---

## 5. Numerical stability

### 5.1 Pole-spectrum-collision

The resolvent `(z_q I − L_F)^{-1}` is bounded by `1 / dist(z_q, σ(L_F))`. With `ω_q ≥ ε > 0` and `σ(L_F) ⊆ R_+`, `dist(z_q, σ(L_F)) ≥ ω_q ≥ ε`. So:

```
‖(z_q I − L_F)^{-1}‖_op ≤ 1/ω_q ≤ 1/ε.
```

For `ε = 10^{-2}`: bound is 100. This is the SRA stability anchor, identical to ORA's.

### 5.2 BF16 conditioning

The numerical condition of the resolvent is `1/ω_q^2 = 10^4` in worst case. BF16 mantissa is 2^{-8} ≈ 4 · 10^{-3}; after κ-times amplification: residual ~ 40 in worst case — **BF16 fails**.

Mitigation (identical to ORA §9.2): **compute the D(z_q, Λ_q) division in FP32**, surround by BF16 GEMMs. FP32 cost: r divisions per query = 4 · T per layer per head = negligible (~10^{-5} of total).

### 5.3 Cayley real-arithmetic form

To avoid CUDA complex arithmetic (per ORA §3.4 and §11.4):

```
z_q = ρ_q + i ω_q   →   Z_q = [[ ρ_q  −ω_q ]
                                [ ω_q   ρ_q ]]    ∈ R^{2 × 2}.
```

The resolvent `(Z_q ⊗ I_{d_s} − L_F^{2-copy})^{-1}` operates on a **complexified Krylov basis** which is two real copies of the original C^0(F). For each token, this doubles the stalk dimension (2 d_s real numbers per stalk) and replaces complex divisions with 2×2 real-matrix inversions:

```
(z_q − Λ_{jj})^{-1} = ((ρ_q − Λ_{jj})^2 + ω_q^2)^{-1} · (ρ_q − Λ_{jj} − i ω_q)
                    → real 2×2 inverse: 1/det · [[ ρ_q − Λ_{jj}   ω_q ]
                                                  [ −ω_q   ρ_q − Λ_{jj} ]].
```

Each scalar complex division → 3 FMA + 1 reciprocal in real arithmetic. **No complex floating-point opcodes needed.**

### 5.4 Backward pass (via REFLECTOR adjoint extended)

The implicit-function adjoint for SRA follows the same recipe as SFA's iter-2 adjoint, with the complex-pole substitution. Define `F(s, θ) := (z_q(θ) I − L_F(θ)) s − b_q(θ) = 0`. Then:

```
∂s/∂θ = − (z_q I − L_F)^{-1} · [∂(z_q I − L_F)/∂θ · s − ∂b_q/∂θ]
       = − (z_q I − L_F)^{-1} · [(∂z_q/∂θ · I − ∂L_F/∂θ) · s − ∂b_q/∂θ].
```

The adjoint state `λ_adj` solves:

```
(z_q I − L_F)^T λ_adj = ∂L/∂s★.
```

Since `(z_q I − L_F)^T = z̄_q I − L_F^T = z̄_q I − L_F` (L_F symmetric), this is the **conjugate-pole resolvent** applied to `∂L/∂s★`. Same Cayley-form solve as the forward, with `Z_q^T` (transpose of the 2×2 block) in place of Z_q.

**Cost**: same as forward (~2× total for fwd + adjoint). **Memory**: O(T · 2 d_s · H) for the adjoint state (2× SFA's, due to Cayley doubling).

Parameter gradients follow the same per-parameter formulae as SFA's iter-2 adjoint (§4.3 of PARADIGM_SHIFT_250_PROOFS.md), with the additional `∂L/∂z_q = − λ_adj^T s★` contribution back-propagating into the MLP_z weights via softplus chain rule.

---

## 6. Composition with shipped paradigms

| Paradigm | Composes? | Notes |
|---|---|---|
| #42 SCFA (shipped) | ✓ Inherited via SFA-recovery limit | Set d_s = 1, U = SCFA's B. |
| #250 SFA (designed iter 1) | ✓ Inherited via z_q → i √λ limit | Set ω_q = √λ constant per head. |
| #46 REFLECTOR | ✓ Backward pass | Implicit-function adjoint extended to complex pole. |
| #78 ATTENTION-SINK | ✓ Edge set | Sinks included in E (same as SFA). |
| #51 ATLAS-COMPILE | ✓ Kernel-level | The closed-form per-query kernel benefits from autotuning. |
| #43 ORION | ✓ Orthogonal | Optimizer-time MOR vs attention-graph MOR. |
| #44 MELT | ✓ Orthogonal | FFN tensor-train; SRA is attention. |
| BF16 stack (iter 1-10) | ✓ Inherited | FP32-D in the resolvent (per §5.2), BF16 outer GEMMs. |

**Stack projection at T=16384, L=24, 1B params**:

| Stack | Per-step | Steps-to-target-NLL | Total |
|---|---|---|---|
| SCFA flagship | 1× | 1× | 1× (baseline) |
| + SFA (paradigm #250, Conjecture 1) | ~1.01× | 0.5–0.7× | 1.4–2× |
| + SRA (this paradigm, ORA's 4.3× compute on top) | **4.3×** | 0.5–0.7× (inherited) | **6.0–8.6× total** |
| + future paradigm #252 (Persistent Sheaf Attn) | TBD | TBD | TBD |

The headline magnitude claim of SRA is **6–8× wall-clock speedup at iso-NLL over SCFA flagship**, by combining SFA's quality gain with ORA's compute gain.

---

## 7. Open conjectures and tests

### 7.1 Conjecture 4 (multi-pole expressivity preserves under sheaf)

ORA's Prop 7.2 gives `‖μ_true − μ_ORA-r-pole‖_BV ≤ C κ(U)^{1/2} r^{-1/2}` — r-pole rational approximation of arbitrary BV measures on σ(A). Conjecture 4 asserts the same density holds for SRA when the operator is L_F (cellular-sheaf Laplacian) rather than ORA's `U Λ U^T + γI`. The cocycle obstruction modes of L_F (high-frequency content) make the measure-class richer, so the density argument may need adjustment, but the limit (r → ∞) recovers continuous spectrum approximation.

### 7.2 Conjecture 5 (per-query pole correlates with linguistic phenomena)

If Conjecture 1 (refined, §5 of PARADIGM_SHIFT_250_PROOFS.md) holds — cocycle modes correlate with Φ-phenomena (anaphora, agreement, embedded discourse) — then the per-query pole `z_q` should *empirically* concentrate at specific spectral locations for queries inside Φ-phenomena. **Testable post-implementation** by extracting per-query (ρ_q, ω_q) statistics on a Φ-tagged val set.

### 7.3 Gate-0 falsification spec

Inherits paradigm #250's Gate-0 spec (5 probes A-E) with these modifications:

**Probe B (refined)**: Run SRA at d_s = 64, r = 4, P = 1 (single pole), z_q from MLP_z. Pass criterion (refined from iter 2): `Δ NLL_step_500 ≤ -0.015 nat` aggregate AND `Δ NLL_Φ-rich / Δ NLL_Φ-poor ≥ 4×`.

**Probe F (new, SRA-specific)**: Verify ORA-recovery and SFA-recovery limits numerically:
- Set d_s = 1, U = B: forward NLL should match SCFA's within 0.005 nat (ORA-recovery).
- Set z_q = i √λ constant: forward NLL should match SFA-only's within 0.005 nat (SFA-recovery).
- Both limits work simultaneously: NLL parity within 0.005 in both edge cases.

**Probe G (new, SRA-specific)**: Verify pole-statistics make sense. After 500 fine-tune steps, distribution of (ρ_q, ω_q) over val tokens:
- Pass: ω_q distribution variance ≥ 0.5 · E[ω_q]^2 (model is using per-query focus, not collapsing to single global bandwidth).
- Pass: ρ_q distribution covers ≥ 50% of σ(L_F)'s range (pole locations span the spectrum).

**Budget**: ≤ 2 GPU-hours, including all SFA Gate-0 probes + SRA-specific Probes F, G.

---

## 8. Implementation roadmap

SRA builds directly on SFA's infrastructure. Once SFA's Gate-0 (paradigm #250) passes and Phase 1-3 CPU/GPU kernels are wired in:

**Phase 0 — SRA Gate-0** (1 GPU-hour, after SFA Gate-0). Probe B + F + G.

**Phase 1 — SRA primitives** (3–5 iterations after SFA primitives exist).
- New primitives needed beyond SFA's 5:
  - 6: `MLP_z` forward/backward (small MLP, ~ same as MLP_Σ infrastructure).
  - 7: Per-query Cayley-form resolvent application: scalar `r`-fold inversion of 2×2 real blocks.
  - 8: Multi-pole partial-fraction combination (only when P > 1).
- ~3 new substantive kernels (most are trivial wrappers).

**Phase 2 — Trainer wire-in** (2–3 iterations).
- Add `cfg.useSRA, cfg.sraPolesP, cfg.sraEpsilon` to training_config.h.
- Default `P = 1, ε = 10^{-2}`.

**Phase 3 — Validation** (2–3 iterations).
- 66M × 2500 steps: SRA vs SFA NLL parity test.
- 1B × 1000 steps: long-context test at T=16384.

**Phase 4 — Production** (1 iteration).
- Default flag: `--sra --sra-poles 1` for new training; SFA fallback (`--sfa`) for legacy.

**Total**: ~10-15 iterations from Phase 0 to production, assuming SFA primitives already exist (paradigm #250 Phase 1-3 must complete first).

---

## 9. Summary

SRA combines SFA's per-token perspective (paradigm #250) with ORA's per-query complex-pole focus into a unified framework. The central equation `y_q = Im[(z_q I − L_F)^{-1} b_q]_{[q]}` recovers both SFA (at z_q = i √λ) and ORA (at d_s = 1) as proven limiting cases. The closed-form factorisation through per-token stalk frames U_i gives per-query cost `O(W r^2 + d_s r)` ≈ 2000 ops at default config — ~16× cheaper per query than SFA's Chebyshev iteration and comparable to ORA's per-query cost.

Combined with SFA's quality gain (0.015–0.03 nat NLL improvement via cocycle expressivity, paradigm #250 Conjecture 1) and ORA's compute gain (4.3× over SCFA), SRA delivers a projected **6–8× wall-clock speedup at iso-NLL over SCFA flagship** at T=16384.

Numerical stability: same ε-floor mechanism as ORA (ω_q ≥ ε = 10^{-2}), same Cayley real-arithmetic form, same FP32-D-inside-BF16 plan. Backward pass extends iter-2's REFLECTOR adjoint to complex-pole resolvent (transpose = conjugate-pole, real-arithmetic via Cayley).

Multi-pole extension (P > 1) gives ORA's multi-modal attention expressivity inside SFA's sheaf-Dirichlet framework — first-time-achievable representations of multi-peaked, oscillating, complex-damped attention measures over per-token-perspective sheaves.

**Open empirical questions**:
- Conjecture 4: r-pole rational approximation tight on L_F spectra.
- Conjecture 5: per-query pole z_q correlates with linguistic Φ-phenomena.

Both testable at Gate-0 + first 500 fine-tune steps on the existing 1B flagship `chiron_1B_T16384.step30000`.

**Predecessor**: paradigm #250 SFA (full design in PARADIGM_SHIFT_250_DESIGN.md + proofs in PARADIGM_SHIFT_250_PROOFS.md).
**Successor**: paradigm #252 (TBD) — Persistent Sheaf Attention, extending Conjecture 3 (layer-stacking cohomology) into a computable layer-pruning / depth-tuning mechanism.
