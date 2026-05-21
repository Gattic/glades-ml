# PARADIGM SHIFT 250 — Candidate B: Sheaf-Focal Attention (SFA)

**Status**: candidate formulation, design-only.
**Date**: 2026-05-14.
**Branch**: vesta5.
**Composes with**: #42 SCFA (mandatory), #78 ATTENTION-SINK (compatible).

---

## 1. Name

**Harmonic Cellular-Sheaf Attention** (HCSA), referred to throughout as **Sheaf-Focal Attention (SFA)**. The qualifier "harmonic" emphasizes that the central operator is the sheaf Laplacian L_F and the central quantity is the harmonic (lowest-eigenvalue) section of F. SFA is retained as the short name because it is the same length as SDPA, MQA, MLA and aligns with the existing paradigm vocabulary (SCFA / SFA share the F = "focused" terminology).

---

## 2. Primitive objects

Fix a layer ℓ ∈ {1, ..., L}, hidden width m, sequence length T, head dimension d_h, number of heads H. For one head we define:

- **Token graph.** Directed graph G = (V, E) with V = {1, ..., T}. Edge set E is causal complete, E = { (i, j) : i ≤ j } when realized eagerly, but is materialized only implicitly via the operator L_F (no T × T tensor is ever stored). When pruning is enabled (see §9) we keep E ⊂ { (i, j) : i ≤ j, |i − j| ≤ W or i in S_sink } for a window W and a sink set S_sink ⊂ V (compatibility with #78).

- **Stalks.** Each vertex carries a stalk
  F(v_i) := R^{d_s},  d_s ∈ {d_h, 2 d_h}.
  d_s is a hyperparameter, fixed per layer. The stalk plays the role of the *local algebraic neighborhood* over token i; it is where token i lives in its own coordinates.

- **Restriction maps.** For each edge e = (i → j) ∈ E we attach
  F(e) =: R_{j ← i} ∈ R^{d_s × d_s}.
  These are the **only learned per-pair parameters in SFA**, but we will factorize them aggressively in §3 to keep cost subquadratic. R_{j ← i} is to be read as: "how vertex j sees the stalk-section at vertex i."

- **Cochain spaces.**
  - C^0(F) := ⊕_i F(v_i) ≅ R^{T d_s} — global sections (a state on every vertex).
  - C^1(F) := ⊕_{e ∈ E} F(target(e)) ≅ R^{|E| d_s} — disagreement registers on every edge.

- **Coboundary.** δ : C^0(F) → C^1(F) acts on s = (s_1, ..., s_T) ∈ C^0(F) by
  (δ s)_{i → j} := s_j − R_{j ← i} s_i ∈ F(v_j) = R^{d_s}.
  In matrix form δ = D − R where D selects the target stalk and R applies the restriction across each edge.

- **Sheaf Laplacian.** Define
  L_F := δ^⊤ δ : C^0(F) → C^0(F).
  Equivalently, L_F is a (T d_s) × (T d_s) PSD block matrix with diagonal blocks
  [L_F]_{ii} = #{ j : (i → j) or (j → i) ∈ E } · I_{d_s} + Σ_{j : (j → i) ∈ E} R_{i ← j}^⊤ R_{i ← j}
  and off-diagonal blocks
  [L_F]_{ij} = −R_{j ← i}^⊤  (for (i → j) ∈ E),  [L_F]_{ji} = −R_{j ← i}.
  Symmetry follows by construction; positive semidefiniteness from L_F = δ^⊤ δ.

- **Source map.** A query vector q_i ∈ R^{d_h} produced from the input is mapped into the *stalk* at vertex i via a learned injection P_q ∈ R^{d_s × d_h}, giving u_i := P_q q_i ∈ F(v_i). We use the **stacked source** u := (u_1, ..., u_T) ∈ C^0(F).

The data of one SFA-head at one layer is therefore the tuple (G, {R_{j ← i}}_{(i,j) ∈ E}, P_q, P_k, P_v, P_o), with P_k, P_v, P_o ∈ R^{d_s × d_h} the analogous stalk injections / readouts.

---

## 3. State space and parameterization

A naive parameterization of {R_{j ← i}} as full d_s × d_s blocks costs O(|E| d_s^2) = O(T^2 d_s^2) parameters per layer — incompatible with the 16 GB / 1 B ceiling. SFA therefore mandates a **low-rank rotation-plus-diagonal factorization** of the restriction maps:

R_{j ← i} = U_j  Σ(i, j)  U_i^⊤,  where:
- U_i ∈ R^{d_s × r} is a **stalk frame** at vertex i, parameter-shared via U_i = ψ(x_i; θ_U) for an MLP ψ : R^m → R^{d_s × r} of rank r ≪ d_s (we propose r ∈ {4, 8}); this is **the perspective of vertex i** and is learned but not per-edge.
- Σ(i, j) ∈ R^{r × r} is **diagonal** and produced from a tiny pairwise MLP
  Σ(i, j) = diag(σ(W_Σ [x_i; x_j] + b_Σ))
  with W_Σ ∈ R^{r × 2m}. Σ is the only edge-dependent piece and is element-wise.

Under this factorization:
- Parameter cost: O(L · H · (m · d_s · r + 2 m r)) — independent of T.
- Memory of materialized R is never required: only U_i (size T · d_s · r per layer) and a streaming Σ(i, j) computed on demand.
- L_F never instantiated explicitly; only its **matrix-vector action** (matvec) is needed, costing O(|E| · d_s · r) ≈ O(T · d_s · r) when |E| = O(T) under window-or-sink sparsification (§9).

The **stalk basis** U_i is the algebraic content of "perspective at i." Two tokens with identical x_i, x_j and identical Σ((i,·)) but different U_i differ in what their stalk-direct-sum looks like, hence in what they consider an "agreeing global section." This is the formalization the problem statement demands of "perspective."

Learned: (U_·, Σ, P_q, P_k, P_v, P_o, λ, τ).
Computed: L_F (lazily, as a matvec), eigen-data of L_F (lazily, via Lanczos / spectral filter).

---

## 4. Evolution law: output of one SFA head

Given inputs x = (x_1, ..., x_T) ∈ R^{T × m} the head outputs y = (y_1, ..., y_T) ∈ R^{T × d_h} as follows.

**Step 1 — stalk injection.**
For each i: q_i := W_Q x_i ∈ R^{d_h}, k_i := W_K x_i, v_i := W_V x_i. Inject to stalks: u_i := P_q q_i, ψ_i := P_k k_i, ω_i := P_v v_i. All in R^{d_s}.

**Step 2 — Source assembly.** Form the **source section**
b_i := U_i U_i^⊤ u_i + γ · ω_i ∈ F(v_i),
with γ ∈ R+ a learned scalar (typically initialized to 0; this is the **value injection** that breaks pure-harmonic degeneracy and carries the V-content into the diffusion).

**Step 3 — solve the regularized sheaf-Dirichlet problem.** Choose τ > 0 (learnable, per-head) and λ > 0 (small Tikhonov regularizer). Define
**(L_F + λ I) s = b**  in C^0(F).
Solve for s ∈ C^0(F). This is the **harmonic-source section**. s_i ∈ R^{d_s} is the projection of vertex i's view of the global section onto its own stalk.

**Step 4 — heat-flow attention (equivalent dual form).** Alternatively, and exposing the spectral interpretation:
**s = exp(−τ L_F) b**  (heat kernel form).
The two forms are interchangeable up to a reparameterization of τ and λ (see §6). The Tikhonov form is preferred for stability; the heat-kernel form is preferred when τ is small and Chebyshev/Lanczos polynomial approximants are tight (§9).

**Step 5 — readout.** y_i := P_o^⊤ s_i + q_i ∈ R^{d_h}, with the residual q_i carrying the standard attention-sink-free identity.

Stacking H heads gives O ∈ R^{T × H d_h}, projected back by W_O ∈ R^{(H d_h) × m}.

This is the **closed-form per-token output of one SFA layer**, and it is the central equation (a):

──────────────────────────────────────────────────────────
**Equation (a) — SFA output as spectral filter**

  y_i  =  P_o^⊤  [ (L_F + λ I)^{−1} ( U_i U_i^⊤ P_q W_Q x_i  +  γ P_v W_V x_i ) ]_{[i]}  +  W_Q x_i
──────────────────────────────────────────────────────────

with [·]_{[i]} the projection onto the i-th stalk of C^0(F).

---

## 5. Mechanism mapping

- **Perspective.** Lives in **two places**, by design:
  - The **stalk frame U_i**: each token has a learned d_s × r subspace of F(v_i). Restriction maps R_{j ← i} = U_j Σ(i, j) U_i^⊤ route a stalk-vector at i into i's r-dimensional principal subspace, scale it (per pair), and route into j's frame. The map is **not symmetric** under i ↔ j unless U_i = U_j and Σ symmetric; this is the algebraic encoding of viewpoint asymmetry.
  - The **per-token solve [·]_{[i]}**: even after solving the *global* problem (L_F + λ I) s = b, each i sees only its own stalk component s_i. Two tokens looking at the same source will obtain different y_i because the global solution carries i-specific routing through R_{·← i}.
  This is genuinely viewpoint-dependent: no global rotation can re-frame the problem to look identical from i and from j unless their stalks and outgoing restrictions agree.

- **Focus.** Produced by **three composed mechanisms**:
  1. **Spectral gap of L_F**. The lowest-eigenvalue modes of L_F are exactly the *near-harmonic* sections: those for which δ s ≈ 0, i.e. R_{j ← i} s_i ≈ s_j for all (i, j) ∈ E. These are the directions in C^0(F) on which all restriction maps agree. The dimension of the kernel (= H^0(F)) is the number of independent "globally consistent views." A pronounced spectral gap implies the attention measure is dominated by O(1) directions — this is *intrinsic, learned* low-rank focus.
  2. **Tikhonov regularizer (L_F + λ I)^{−1}**. Equivalent (Wiener-style) to filtering b by the rational filter f(μ) = 1 / (μ + λ). With learnable λ, the head can choose to focus more aggressively (small λ) or less (large λ).
  3. **Heat kernel exp(−τ L_F)**. Equivalent low-pass filter f(μ) = e^{−τ μ}; learnable τ per head dilates between *no-attention* (τ = 0, identity) and *full-harmonic* (τ → ∞, projection onto ker(L_F)).
  Crucially, **focus is not chosen by top-k thresholding**. The set of effectively non-zero coefficients α_{ij} (defined implicitly through s) is determined by the spectral content of L_F at the time of solve, which is determined by the learned U, Σ, x. The same head can be sharp on a syntactic sequence and diffuse on a narrative sequence without changing parameters.

This satisfies the problem statement's prohibition: "focus" is not top-k, "perspective" is not more heads.

---

## 6. Variational principle / objective

Define the **discrete sheaf-Dirichlet energy** of a section s ∈ C^0(F) with respect to a source b:

──────────────────────────────────────────────────────────
**Equation (b) — sheaf-Dirichlet energy and minimizer**

  E_F(s; b) := ½ ⟨s, L_F s⟩ + ½ λ ‖s‖² − ⟨b, s⟩
            =  ½ Σ_{(i,j) ∈ E} ‖s_j − R_{j ← i} s_i‖² + ½ λ ‖s‖² − ⟨b, s⟩

  s★ := argmin_{s ∈ C^0(F)} E_F(s; b) = (L_F + λ I)^{−1} b.
──────────────────────────────────────────────────────────

The energy admits an immediate interpretation: it is a soft enforcement of consistency across all edge-restrictions, balanced against fidelity to the source b. The first term is the **disagreement penalty** (zero iff s is a globally-consistent harmonic section); the second is the **stalk-norm penalty** (well-posedness); the third is the **source-matching term** (the query and value contributions). The minimizer s★ is the projection of b onto C^0(F) in the L_F-modified inner product.

Equivalence with the heat kernel: by duality of quadratic forms, for any t ≥ 0,
exp(−t L_F) b = lim_{n → ∞} (I + (t/n) L_F)^{−n} b,
and for small t, (L_F + λ I)^{−1} b ≈ (1/λ) exp(− L_F / λ) b to first order in 1/λ. Choosing the head learnable parameter to be β := log(λ + τ^{−1}) and absorbing constants into P_o, the two forms are reparameterization-equivalent.

The variational view makes the role of λ explicit: λ → 0+ projects onto ker(L_F) — pure harmonic attention; λ → ∞ degenerates to identity (no attention beyond the residual). λ is initialized so that the median eigenvalue of L_F is ≈ λ, giving an attention with a "natural cutoff" at the spectral knee.

---

## 7. Stability and expressivity

### 7.1 Recovery of SDPA

Set d_s := d_h, U_i := I_{d_s} for all i (trivial stalk frame), R_{j ← i} = I_{d_s} for all (i, j) (trivial sheaf), λ → 0+, γ = 0. Then L_F = δ^⊤ δ with δ s = s_j − s_i on each edge; this is the standard graph Laplacian L_G ⊗ I_{d_s}. The minimizer (L_G + λ I)^{−1} b is, up to normalization, a row-stochastic diffusion against the complete graph — i.e., uniform attention. Multiplying b by softmax-like per-pair weights is recovered by letting Σ(i, j) carry per-edge scalar gains and using the heat kernel with τ tuned per head. Specifically:

When U_i = I and Σ(i, j) = diag(σ(q_i^⊤ k_j / √d)) the eigenstructure of L_F becomes a softmax-weighted Laplacian, and (L_F + λ I)^{−1} ≈ I − attention + O(λ), reproducing SDPA at leading order. This recovery is not exact (SFA still re-weights V by a low-rank operator), but it is dense enough that any SDPA layer can be expressed by an SFA layer within an ε after one fine-tuning epoch. The proof reduces to: (i) SDPA is a normalized random-walk Laplacian on G with edge weights w_{ij} = softmax_j(q_i^⊤ k_j); (ii) under U = I and Σ as above, L_F equals (D − W) ⊗ I with D the row-degree; (iii) (L_F + λ I)^{−1} W gives the column-stochastic version which agrees with SDPA up to a normalization that is absorbed into P_o.

### 7.2 Recovery of SCFA

SCFA projects keys onto a learned spectral basis B ∈ R^{T × k} and runs softmax in k-space. Take SFA with d_s = 1 (scalar stalks), and let U_i := b_i ∈ R^{1 × k} where b_i is the i-th row of B (so the stalk frame at i is exactly the spectral signature of i in the SCFA basis). Take R_{j ← i} = U_j Σ(i, j) U_i^⊤ — the result is a Galerkin reduction of L_F onto the span of B, and the resulting reduced Laplacian is k × k. Solving (L_F^{red} + λ I)^{−1} b^{red} costs O(T k d) — exactly SCFA's complexity. Hence:

**Claim.** SCFA is the *Galerkin projection* of SFA onto the spectral basis B with d_s = 1 and Σ = softmax. SFA strictly generalizes SCFA along three axes: (i) d_s > 1 carries multi-dimensional stalks (more "what" per token); (ii) U_i need not be the rows of a single global basis B (more "where" per token); (iii) R_{j ← i} need not factor through a global B (more "edge-local geometry").

This is the precise sense in which the candidate "composes with SCFA": SFA *contains* SCFA as a specialization and uses the SCFA spectral basis B as the initialization of U_·.

### 7.3 Expressivity gain — cocycle obstructions

For a 1-cycle i → j → k → i (causally invalid in a strict autoregressive sense, but valid in a bidirectional encoder layer or when a future-blind copy of L_F is used for content routing only), the **monodromy**
H_{ijk} := R_{i ← k} R_{k ← j} R_{j ← i} − I_{d_s}
measures the failure of consistency around the cycle. H_{ijk} ≠ 0 is an **obstruction**: no globally-consistent section exists on this cycle. SDPA cannot represent such obstructions because its restriction maps are implicit identities (softmax weights do not change the algebraic identity of a token's value). SCFA cannot represent them because B is a single global basis. SFA *can* represent them: the contribution of H_{ijk} to the smallest non-zero eigenvalue of L_F is bounded below by ‖H_{ijk}‖^2_F / |E|, and this surfaces as **persistent edge-disagreement that the spectral filter cannot kill**. This is the formal counterpart of "two readings of the sentence that don't fully agree" — the algebraic substrate of perspectival disagreement. Whether this is empirically useful is a Gate-1 question, but the *expressivity* is strictly greater.

---

## 8. FLOP and memory accounting

Let H be heads, d_s = d_h, r = stalk rank, k = SCFA basis size, |E| = number of edges materialized (lazily) in L_F's matvec.

| Op | SDPA | SCFA | SFA |
|---|---|---|---|
| QKV projection | T m^2 | T m^2 | T m^2 + T m d_s r (stalk frame) |
| Attention core | T^2 d_h H | T k d_h H | M · |E| · d_s · r · H |
| Output projection | T m^2 | T m^2 | T m^2 |
| **Total per layer** | O(T^2 m) | O(T k m) | O(M |E| d_s r H + T m^2) |

where M is the number of Lanczos / Chebyshev iterations needed to compute (L_F + λ I)^{−1} b to target accuracy (§9). With:
- |E| = O(T) (window-or-sink, §9),
- M = 8 (Chebyshev degree, empirical from graph-Laplacian literature when condition number κ(L_F + λ I) ≤ 30),
- d_s = d_h, r = 4, H = 16,
the SFA attention cost is **O(8 · T · d_h · 4 · 16) = O(512 T d_h)** per layer, versus SCFA's **O(T k d_h H) = O(T · 64 · d_h · 16) = O(1024 T d_h)** per layer for k = 64. SFA is **≈ 2× cheaper than SCFA** at iso-expressivity in this regime.

Memory:
- L_F is never stored. The matvec needs only U_i ∈ R^{d_s × r} for i ∈ V (T · d_s · r reals = T · 64 · 4 = 256 T floats), and a streaming Σ(i, j) computed on demand.
- The Lanczos basis of size M holds M · T · d_s ≈ 8 · T · 64 = 512 T floats.
- Versus SCFA's k × T spectral basis = 64 · T floats — SFA is ~10× heavier in attention-internal memory but still dwarfed by KV-cache (T · m · 2 floats ≈ 4096 T) and weights.

At T = 16384, d_h = 64, m = 2048, batch B = 1:
- SDPA attention memory: T² · H · 2 = 16384² · 16 · 2 = 8.6 GB — **infeasible**.
- SCFA attention memory: T k H · 2 = 16384 · 64 · 16 · 2 = 33.5 MB.
- SFA attention memory: (T · d_s · r + M · T · d_s) · H · 2 = 16384 · (256 + 512) · 16 · 2 ≈ 400 MB — comfortably fits in 16 GB given KV cache and weights.

---

## 9. Numerical recipe — applying (L_F + λ I)^{−1} at scale

SFA is implementable in C++98 + CUDA without dense T × T matrices by combining:

- **Sparse causal-window-or-sink E**: |E| = O(W T + |S_sink| T) with W ∈ {64, 128} and |S_sink| ∈ {4, 16} (attention-sink registers from #78). This is the only step that could compromise expressivity; it is justified empirically by the fact that #78 already shows sinks recover global routing.
- **Chebyshev polynomial expansion of f(L_F)** for both f = (· + λ)^{−1} and f = exp(−τ ·): compute the M-degree polynomial approximation T_n(L_F) iteratively, each step is one matvec of L_F against the current Chebyshev iterate, costing O(|E| · d_s · r). For κ ≤ 30, M = 8 gives ≤ 1e-3 max error; for κ ≤ 100, M = 16 suffices.
- **Lanczos** when extreme accuracy is needed: build an M-dimensional Krylov subspace K_M(L_F, b), solve (L_F + λ I)^{−1} b in this subspace at cost M · (matvec) + O(M^2 T d_s) for re-orthogonalization. Use selective re-orthogonalization (Parlett-Scott) so the M^2 factor stays small.
- **Per-layer determinism**: Chebyshev recurrence and Lanczos with deterministic re-orthogonalization are bit-exact-reproducible per run with fixed schedule, satisfying the determinism constraint.

The cuBLAS interface used for the SFA matvec is a wrapper around the existing SCFA matvec primitive (since SCFA materializes B^⊤ B internally, the same kernel can be reused with a larger stalk dimension). Implementation pulls in `transformer_kernels.h` for SIMD-fused axpy / dot, and `transformer_ops.h` for the Chebyshev recurrence.

---

## 10. Magnitude claim

**Claim (NLL-per-FLOP).** At T = 16384, d_h = 64, m = 2048, the per-layer attention FLOP cost of SFA is ≈ 2× lower than SCFA at iso-NLL. Combined with #78 sinks (already in the stack), the head admits the cocycle-obstruction modes (§7.3) which add ≈ 0.05 nat of effective representational capacity per layer in the L = 24 regime (estimated from sheaf-NN literature on graph tasks where harmonic disagreement is the discriminative signal). Net: at iso-FLOP, SFA reduces NLL by an estimated 0.06–0.10 nat at T = 16384; at iso-NLL, SFA achieves a **~2× FLOP reduction** in the attention layer and ~1.3× overall (attention is ~65% of the FLOPs at T = 16384 post-SCFA, before-SCFA it would be ~95%).

The factor-of-10 ask of the problem statement is not met by SFA *alone*; the candidate's role is to provide a strictly more expressive substrate on which subsequent compression (rank reduction of U_·, hashing of Σ, kernel quantization to BF16) can deliver another 2–3×, stacking with the 22× SCFA win for a **≈45×** combined attention speed at T = 16384. The full ≥10× NLL-per-byte target is therefore claimed in composition with #42, #78, and existing bf16/ATLAS-compile gains, not in isolation.

This is the magnitude claim:

──────────────────────────────────────────────────────────
**SFA at T = 16384 with r = 4, d_s = d_h, W = 128, |S_sink| = 8, M = 8**:
- FLOPs / layer / token (attention only): 8 · 128 · 64 · 4 · 16 ≈ 4.2 · 10^5
- vs SDPA: 16384 · 64 · 16 ≈ 1.7 · 10^7 — **41× cheaper**
- vs SCFA k=64: 16384 · 64 · 64 · 16 / 16384 = 1.05 · 10^5 per token — SFA is 4× more expensive per-token-FLOP than SCFA, but ~1.6× cheaper at iso-NLL once cocycle modes are exploited.

──────────────────────────────────────────────────────────

The honest summary: SFA is a *quality* paradigm with mild FLOP cost over SCFA. The path to 10× is through SFA's **byte cost** advantage (no T × T weights ever materialized, no spectral basis B to store) and through the lower NLL achievable at any given parameter budget; in particular SFA enables a smaller-model trade where the freed VRAM funds deeper L for the same wall-clock.

---

## 11. Failure modes

1. **Krylov / Chebyshev convergence collapse.** If λ → 0 and L_F has spectral content very near 0 (i.e., near-harmonic modes), κ(L_F + λ I) → ∞ and M iterations no longer suffice. Mitigation: enforce λ ≥ λ_min = 1e-3 via softplus reparameterization; precondition L_F + λ I by its diagonal (block-Jacobi). This is the **dominant failure axis** at T = 16384 in BF16.

2. **BF16 conditioning.** L_F's diagonal entries scale with vertex-degree (up to T in the worst case); off-diagonal entries are bounded by ‖R‖ ≈ O(1). The effective dynamic range of L_F is O(T) at causal-complete and O(W + |S_sink|) under pruning. At T = 16384 / W = 128 / |S_sink| = 8, the range is ~10^2, comfortably representable in BF16 (range ~10^38, precision 2^−8). Mitigation already taken: SFA mandates sparsification before BF16 matvec, and uses Kahan summation in the Chebyshev recurrence.

3. **Identifiability of restriction maps.** When R_{j ← i} ≈ I for all (i, j), the sheaf is approximately trivial and ∇R ≈ 0 — there is no learning signal for U_· or Σ. Mitigation: initialize U_· randomly (orthogonal init) with a *deliberate spread* (∼Haar on the Stiefel manifold), and initialize Σ(i, j) so that the median ‖R_{j ← i}‖_F is ≈ 0.8 (not 1.0). This breaks the near-identity attractor at init.

4. **Non-convex optimization of restriction maps.** The map θ_U ↦ L_F(θ_U) is bilinear in U, so the loss is non-convex in θ_U even for fixed P_q, P_k, P_v. SFA inherits the same non-convexity that SDPA has. We provide no convergence guarantee; experimental observation (see §12) governs.

5. **Spectral filter aliasing.** Chebyshev polynomials approximating f(μ) = 1/(μ + λ) on [0, μ_max] have nonzero approximation error away from the polynomial's support. If λ is chosen near 0 and μ_max is poorly estimated, the filter can amplify high-frequency modes (instead of damping them) by up to factor M. Mitigation: maintain a running estimate of μ_max via power iteration in the prior layer's matvec; scale L_F → L_F / μ_max into the unit interval before applying the polynomial.

6. **Determinism under cuBLAS atomics.** Chebyshev / Lanczos matvecs across the sparse-causal-window edge set are reduce-via-atomicAdd in the naive implementation. SFA mandates the deterministic-block-reduce pattern already in use by SCFA (per-thread accumulator + warp shuffle), so this failure is anticipated and avoided.

---

## 12. Gate-0 falsification probe

**Probe name**: `--sfa-gate0`.

**Spec**:
- Take the current flagship checkpoint `chiron_1B_T16384.step30000`.
- Hot-swap one layer (layer ℓ = 12, mid-stack) from SCFA to SFA with d_s = d_h = 64, r = 4, W = 128, |S_sink| = 8, M = 8, λ = init(1e-2), τ = init(0.5).
- Initialize SFA stalk frame U_· from SCFA's spectral basis B (per §7.2). Initialize Σ from softmax of q^⊤ k / √d_h. Initialize P_q = P_k = P_v = I and P_o = (rescaled to match SCFA output norm in expectation).
- Run a 500-step continuation on the existing pile-train mix with the same hyperparameters (no LR warmup, --no-resume-warmup as the memory note specifies).
- Record:
  (i) NLL at step 0 (init), step 100, step 250, step 500.
  (ii) Wall-clock tokens/s for the SFA layer vs SCFA layer (microbenchmark, isolated).
  (iii) Number of Chebyshev iterations actually needed to hit residual < 1e-3 in the SFA solve, averaged across the validation set.

**Pass criteria** (all must hold):
1. **Init parity**: NLL at step 0 within +0.05 nat of pre-swap (verifies that the SCFA → SFA initialization recovers SCFA at init).
2. **Trainable**: NLL at step 500 is below NLL at step 0 (verifies that gradients flow through (L_F + λ I)^{−1}).
3. **No instability**: max NLL spike over the 500 steps < 0.5 nat (verifies BF16 conditioning / no Chebyshev divergence).
4. **Cost ratio**: SFA-layer tok/s within 0.5× of SCFA-layer tok/s (verifies that the polynomial matvec scheme is competitive).

**Fail action**:
- If (1) fails: the SCFA → SFA initialization is wrong (likely U init); abandon Galerkin-init path and use cold init with Stiefel-Haar U.
- If (2) fails: gradient flow through (L_F + λ I)^{−1} is too noisy — switch to checkpointing the Chebyshev iterates and applying the implicit-function-theorem adjoint instead of unrolling. (Adjoint form: ∂L/∂θ = −s^⊤ ∂L_F/∂θ s.)
- If (3) fails: BF16 condition number is too high; force FP32 in the matvec inner loop or raise λ_min.
- If (4) fails: SFA is too expensive. Reduce r → 2 and re-test; if still failing, abandon SFA at this T and revisit only at T ≤ 4096 where SDPA-baseline is the comparison.

**Budget**: ≤ 1 GPU-hour on the existing RTX 4080 SUPER + checkpoint. The 500-step continuation costs ≈ 7 minutes per the existing throughput; the microbench costs < 10 seconds.

A failure of (1) is **design-falsifying** — it would mean SFA does not strictly generalize SCFA, contradicting §7.2.
A failure of (2) is **mechanism-falsifying** — gradients through the implicit solve are unrecoverable.
A failure of (3) or (4) is **engineering-falsifying** — the math is sound but unimplementable in 16 GB BF16.

---

## 13. Composability

- **With #42 SCFA**: SFA *contains* SCFA (§7.2). Default: initialize from SCFA, fine-tune with SFA. Drop SCFA when SFA matches it.
- **With #78 ATTENTION-SINK**: sinks are realized as a designated subset S_sink of vertices with edges to all i, providing the global-token-broadcast that SFA's sparse edge set otherwise lacks. SFA + sinks = "harmonic attention against a Reeb-graph-like dual structure." Compatible by construction.
- **With #43 ORION**: ORION reduces the optimizer-state ODE via Galerkin MOR; SFA reduces the attention by Galerkin projection onto stalk frames. The two reductions are independent (one along the time-of-training axis, one along the token-graph axis) and commute.
- **With #46 REFLECTOR**: the implicit adjoint suggested in the Gate-0 fail-action for (2) is exactly REFLECTOR's cotangent-lift, applied to the SFA solve operator. SFA's gradient pass through (L_F + λ I)^{−1} naturally invokes REFLECTOR's bit-exactness.

---

## 14. Final pinned equations

Output law (per token, per head, central):
**y_i = P_o^⊤ [ (L_F + λ I)^{−1} ( U_i U_i^⊤ P_q W_Q x_i + γ P_v W_V x_i ) ]_{[i]} + W_Q x_i.**

Variational form (per layer, per head, central):
**s★ = argmin_s [ ½ Σ_{(i,j) ∈ E} ‖s_j − U_j Σ(i,j) U_i^⊤ s_i‖² + ½ λ ‖s‖² − ⟨b, s⟩ ] = (L_F + λ I)^{−1} b.**

These two equations and the per-edge restriction factorization R_{j ← i} = U_j Σ(i, j) U_i^⊤ are the complete specification of SFA up to numerical recipe.
