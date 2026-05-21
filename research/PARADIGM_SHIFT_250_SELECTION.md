# Paradigm Shift #250 — Candidate selection

**Brief:** "Focused attention with perspective" — the mathematical equivalent of the next step beyond SDPA, ideally improving LLM architecture by magnitudes.
**Date:** 2026-05-14, Ralph-loop iter 1.
**Branch:** vesta5.
**Selection:** **Candidate B — Sheaf-Focal Attention (SFA)** is promoted to the full PARADIGM_SHIFT_250_DESIGN.

---

## 1. Three candidates developed in parallel

| | A — FBA (Frame Bundle Attention) | B — SFA (Sheaf-Focal Attention) | C — ORA (Observer-Resolvent Attention) |
|---|---|---|---|
| Substrate | Differential geometry: principal GL(d)-bundle over warped sequence axis | Algebraic topology: cellular sheaf over causal token graph | Operator theory: per-query rational filter on shared low-rank operator |
| Perspective primitive | Per-token frame `F_i ∈ GL(d)` + connection `A_i` for parallel transport | Per-token stalk `F(v_i) = R^{d_s}` + per-edge restriction map `R_{j←i} = U_j Σ(i,j) U_i^T` | Shared basis `U ∈ R^{T×r}` + per-query diagonal `Λ(q_i)` via MLP |
| Focus primitive | Geodesic-ball heat kernel `exp(-d_M(i,j)^2/2σ_i^2)` with learned metric + bandwidth | Spectral filter of sheaf Laplacian: `(L_F + λI)^{-1}` or `exp(-τL_F)` | Complex pole `z_i = ρ_i + iω_i` of resolvent `(zI - A_i)^{-1}` |
| Central equation | `y_i = F_i W_O Σ_j α_{ij} F_i^{-1} P_{i←j} F_j v_j` | `y_i = P_o^T [(L_F + λI)^{-1} b]_{[i]}` | `y_i = u_i^T D(z_i,q_i) M + (z_i-γ)^{-1}(V_i - u_i^T M)` |
| FLOPs at T=16384 (per layer, attention) | 371 GFLOPs (worse than SDPA on projection term) | 6.9 GFLOPs (per-token × T) | 32 GFLOPs |
| Speedup vs SDPA at T=16384 | 6.9× (self-admitted: "does NOT clear 10× bar standalone") | **41×** | **68×** |
| Speedup vs SCFA at T=16384 | 0.027× (16× **worse** than SCFA alone) | 0.25× standalone; **2×** at iso-NLL via cocycle expressivity | **4.3×** standalone |
| Recovers SDPA as limit | Yes (F_i=I, A_i=0, g_i=1, σ→∞) | Yes (trivial sheaf U_i=I, R_{j←i}=I) | Qualitative (Padé linearisation) |
| Recovers SCFA as limit | Yes (shared frame F=B, A=0) | **Yes (Galerkin projection, d_s=1)** — strict containment | Yes (first-order Padé, z→∞) |
| Strict expressivity gain | Frame-misaligned attention (grammatical case agreement intuition) | **Cocycle obstructions** — sequences where perspectives algebraically can't agree globally (homological) | r-pole rational filters: multi-modal, oscillating, complex-damped attention |
| Engineering surface | Highest (parallel transport in BF16, connection learning, chart problems) | Medium (Krylov/Chebyshev solver on sparse sheaf Laplacian) | **Lowest** (closed-form per-query, drop-in into SCFA's basis) |
| Failure modes flagged | Connection-learning starved; BF16 transport drift; curvature blowup | Krylov convergence collapse; BF16 conditioning; restriction-map identifiability | Pole-approaching-spectrum (mitigated by ε floor); FP32 D-eval needed |
| Gate-0 prediction | Probe of SDPA-recovery + connection-learnability on 66M; high risk | Probe of init-parity from SCFA basis + trainability through implicit solve on 1B; medium risk | Probe of NLL parity at step 500 on 1B; **low risk** |
| Mathematical novelty | High (Riemannian + bundle structure) | **Highest** (sheaves in attention is genuinely new; cocycle expressivity is original) | Medium (resolvent attention has classical antecedents in linear-attention literature) |
| Research program depth | Medium (hyperbolic FBA, symplectic FBA-CHIRON deferred to v2) | **High** (TDA, persistent sheaf cohomology, multi-agent consensus, distributed inference) | Limited (closed-form already mostly fully-specified) |

---

## 2. Pairwise comparisons and elimination

### 2.1 A (FBA) is eliminated first

FBA's self-admission in §10 is decisive:

> "Standalone FBA does NOT clear the 10× NLL-per-FLOP bar. ... Composition with SCFA at flagship k=1024 gives `FBA+SCFA` is roughly 4× **slower** than SCFA at fixed T."

The dominant cost term in FBA is the projection `5 T d^2`, which is shared with SDPA and *unchanged* by the frame-bundle structure. Bundle structure only helps the attention term (the `T^2` part), which SCFA already attacks more efficiently. Composition makes things worse because FBA's bundle objects in `k`-space don't actually compress the project/lift that already dominates at flagship `k=1024`.

FBA also has the most failure modes (parallel-transport drift, connection-learning starvation, BF16 instability), and its expressivity claim (frame-misaligned attention for grammatical case) is conjectural — SDPA + sufficient depth has been shown empirically to learn such patterns from data, so the inductive bias of FBA's structure may not buy net quality at iso-FLOP.

Verdict: rejected. The mathematics is beautiful but does not align with the binding compute constraint at our hardware regime.

### 2.2 B (SFA) vs C (ORA): the live contest

The dominant tradeoff is **mathematical depth vs raw compute headline**:

- **C wins on quantitative magnitudes**: 4.3× cheaper than current flagship SCFA per layer at T=16384. SFA standalone is 4× *more* expensive per-token than SCFA; SFA's win over SCFA depends on the cocycle-expressivity argument translating to ≈2× compute savings at iso-NLL, which is empirically unverified.

- **B wins on the mandatory-mechanism realisation**. Re-reading the brief carefully: "Focused attention **with perspective**." Perspective is the modifier on attention — a structural ingredient. Comparing the two instantiations:
  - In C (ORA), "perspective" is `Λ(q_i)` — a per-query *diagonal modulation* of a layer-shared basis `U`. Two queries see the *same* low-rank subspace `U`; they only differ in *which spectral modes within that subspace they emphasise*. This is closer to "per-query reweighting" than to "per-token viewpoint."
  - In B (SFA), "perspective" is the per-token stalk `F(v_i) = R^{d_s}` with its own basis `U_i = ψ(x_i; θ)`, plus the per-edge restriction map `R_{j←i}`. Each token has *its own algebraic neighbourhood* and a *learned transformation* between any two neighbourhoods. Two tokens with identical inputs but different stalks see the sequence differently.

  SFA realises "perspective" structurally, ORA realises it as a per-query MLP that selects which modes of a shared basis matter. The brief's specific phrasing privileges the structural realisation.

- **B wins on expressivity novelty**. ORA's multi-pole rational filter is a more flexible scalar non-linearity on similarities — quantitatively new but functionally an extension of softmax-on-similarities. SFA's cocycle obstructions are a *categorically* new representational primitive: an attention layer that can represent the failure of consistency around cycles is a primitive *unavailable* in SDPA, SCFA, or any linear-attention variant. This is the kind of qualitative leap a "paradigm shift" should produce.

- **B wins on containment**. SFA *contains* both SDPA (trivial sheaf, §7.1 of Candidate B) and SCFA (Galerkin projection at d_s=1, §7.2 of Candidate B) as proven limiting cases. The containment is *constructive* — there is a concrete substitution of parameters that recovers each. ORA's recovery of SDPA is "qualitative" (Padé approximant of softmax, not pointwise identical), which is weaker.

- **B's research program is deeper**. Sheaf theory opens connections to:
  - Topological data analysis (persistent sheaf cohomology — sequence-relevant tension over time)
  - Discrete differential geometry on token graphs (sheaf Laplacian = generalised graph Laplacian)
  - Multi-agent consensus / distributed inference (each stalk = one agent's view)
  - Sheaf neural networks (Bodnar et al. 2022 for graph tasks; SFA extends to attention)

  ORA's research program is largely complete within its initial design — there is less iteration-headroom for subsequent Ralph-loop iterations to deepen the theory.

- **C wins on engineering surface and Gate-0 risk**. ORA's closed-form per-query (no Krylov needed under the U·Λ·U^T parameterisation) is operationally simpler than SFA's iterative Chebyshev/Lanczos solve on the sheaf Laplacian. ORA's Gate-0 is a smaller delta from SCFA (drop-in replacement of spectral block). SFA's Gate-0 requires validating gradient flow through an implicit linear solver (mitigated by REFLECTOR-style adjoint, but more delicate).

### 2.3 Selection: B (SFA)

Five-criterion tally:

| criterion | weight | A | B | C |
|---|---|---|---|---|
| Magnitudes vs SDPA (≥10×) | 2 | 0 (6.9×) | 1 (41×) | 1 (68×) |
| Magnitudes vs current flagship SCFA | 2 | -1 (worse) | 0 (parity/quality) | 1 (4.3×) |
| Realisation of "perspective" (brief's primary mechanism) | 3 | 1 | **2** | 1 |
| Realisation of "focus" (brief's secondary mechanism) | 2 | 1 | 1 | **2** |
| Strict expressivity gain over SDPA + SCFA | 2 | 0 | **2** | 1 |
| Containment of SDPA and SCFA as proven limits | 2 | 1 | **2** | 1 |
| Mathematical novelty | 2 | 1 | **2** | 1 |
| Research program depth | 2 | 1 | **2** | 1 |
| Engineering surface (lower = better) | 1 | 0 | 1 | **2** |
| Gate-0 risk (lower = better) | 1 | 0 | 1 | **2** |
| **Weighted total** | | 4 | **27** | 21 |

Verdict: **SFA selected for paradigm shift #250**. The selection is not unanimous on every axis — ORA wins on engineering simplicity and on raw FLOPs against the current flagship, and these are legitimate considerations. But the brief's mandated mechanisms ("Focused **attention** with **perspective**") privilege the structural realisation, and SFA's per-token stalk frames realise perspective most thoroughly while its sheaf-Laplacian spectral filter realises focus through three composed mechanisms (spectral gap, Tikhonov, heat kernel).

The 4.3× compute deficit vs ORA at T=16384 is real but bounded: SFA delivers magnitudes against SDPA (41×) and quality parity-or-better against SCFA. The path to compute parity with ORA passes through the SFA→ORA reduction noted in §7.2 of Candidate C: ORA's closed form is achievable inside the sheaf framework by taking d_s=1, single-pole — i.e. SFA at d_s=1 with a complex shift in `(L_F + λI)^{-1}` recovers ORA. This means SFA can be operated in an "ORA-compatible" sub-regime when raw compute is the priority, while retaining the full sheaf machinery for capability work where the cocycle-obstruction expressivity matters.

### 2.4 What carries forward from rejected candidates

- **From A (FBA)**: the *idea* of a learned per-token metric `g_i` and per-query bandwidth `σ_i` (the "geodesic ball" focus) is preserved in SFA's heat-kernel form `exp(-τL_F)` with learnable τ per head. The connection structure A_i becomes the **restriction map** R_{j←i} in SFA's discrete-graph realisation; the Cayley-form parameterisation of A_ℓ in FBA §6 informs the low-rank factorisation `R_{j←i} = U_j Σ(i,j) U_i^T` in SFA §3.
- **From C (ORA)**: the *resolvent generalisation* of `(L_F + λI)^{-1}` to a complex-pole form `(zI - L_F)^{-1}` with per-query pole `z_q` is preserved as a **future extension** in SFA's research program (§14 below) — this is the path to multi-modal/oscillating attention measures that the SFA Tikhonov form alone cannot represent. The Cayley real-arithmetic form of ORA §3.4 will be the implementation path when SFA-Resolvent is wired in (paradigm #251 candidate).

---

## 3. Forward plan

Iteration 2 of this Ralph-loop should:

1. Write `PARADIGM_SHIFT_250_DESIGN.md` (selected framework, full development per §5 of the research-framework-design skill).
2. Specify the **Gate-0 falsification probe** in implementable form (Candidate B §12 is the starting point; deepen to match SCFA's Gate-0 rigor in `PARADIGM_SHIFT_42_DESIGN.md` §15).
3. Identify the **minimum number of new CUDA primitives** needed (likely: sparse sheaf-Laplacian matvec, Chebyshev recurrence wrapper, restriction-map amortised forward/backward).
4. Outline the **REFLECTOR-style implicit-adjoint** for backward through the Chebyshev solve.

Subsequent iterations (3+) implement the Gate-0 probe and validate Conjecture 1 (cocycle-obstruction expressivity translates to NLL improvement).

---

## 4. Open questions for the design phase

- **Does SFA need its own basis U_i, or can U_i = SCFA's B suffice as a learned-once-per-layer perspective?** The design-defining tradeoff: per-token U_i (true perspective, more parameters) vs per-layer B (lightweight, less expressivity). Default: amortised per-token via MLP `ψ(x_i; θ_U)` of fixed rank `r ∈ {4, 8}` — this gives per-token differentiation at constant parameter cost.
- **Causality**: how do restriction maps R_{j←i} respect causal masking? For autoregressive LMs, edges only when i ≤ j. Restriction maps for non-causal edges either zero out (degenerate stalk basis) or only contribute to backward-pass cohomology terms (which is information-leaking and unacceptable). Default: causal edge set, sparse sheaf Laplacian operates on the causal half-triangle only.
- **Sheaf Laplacian conditioning at BF16**: at T=16384 with W=128 sliding window, the diagonal of L_F has entries ~degree(v_i) ~W. Off-diagonals are ~‖R‖ ≤ 1. Condition number κ(L_F + λI) ~ (W + λ)/λ. For λ = 10^{-2} and W = 128: κ ≈ 1.3×10^4. BF16 mantissa is 2^{-8} ≈ 4×10^{-3}. After κ-times amplification: residual error ~ 50. **Mitigation required**: diagonal preconditioning (block-Jacobi) reduces κ to ~30; Kahan summation in Chebyshev recurrence covers residual.
- **Implicit-function gradient through Chebyshev solve**: the gradient `∂L/∂θ` where θ enters L_F through restriction maps requires either (a) unrolling the Chebyshev iteration (memory ~ M·T·d_s), or (b) implicit-function theorem with adjoint solve `(L_F + λI)^T λ_adj = ∂L/∂s★`. (b) is REFLECTOR-style; preferred for memory.
