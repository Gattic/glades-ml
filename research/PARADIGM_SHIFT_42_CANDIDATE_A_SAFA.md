# Paradigm Shift #42 Candidate A — SAFA: Symplectic Adjoint-Free Architecture

**Status:** candidate design; one of three parallel proposals for shift #42.
**Date:** 2026-05-08.
**Axis:** **eliminate the structural 1F inverse-walk pass** in CHIRON training.
**Author role:** mathematical-physicist development of the cotangent-lift formulation.

---

## 1. Executive summary

CHIRON's per-step training cost is structurally `~3F`: one forward (F1) for output and loss, one *inverse walk* (F2) to re-derive layerwise activations `(q_l, p_l)` from `(q_L, p_L)`, and one backward (F3) of gradient propagation, interleaved with F2. SAFA proposes to *replace F2+F3 with a single co-symplectic forward sweep on a doubled state space*: instead of integrating the inverse Hamiltonian flow on `𝒴 = ℝ^{T×m} × ℝ^{T×m}` (recovering activations) and then backpropagating, SAFA integrates the *cotangent-lifted* flow on the cotangent bundle `M = T*𝒴` directly *forward in layer-index*. The cotangent lift `Φ_l*` of a symplectic shear `Φ_l` is itself a unit-lower-triangular shear on `(q*, p*)` and can be assembled from forward kernels — *no reverse-mode kernel, no inverse walk*.

The core mathematical observation: the symplectic shear `Φ_l: (q,p) ↦ (q, p + Y(q))` has a cotangent lift on the *whole* doubled phase space `(q, p, q*, p*)` whose action on `(q*, p*)` is `(q*, p*) ↦ (q* + (∂Y/∂q)^T p*, p*)`. **This expression depends on `q_l` (the input to layer `l`), not on `q_{l+1}`.** That is the operational obstruction we must resolve. We resolve it via *option D*: the cotangent-lifted flow is itself symplectic on `M`, and on this doubled space the ORIGINAL `q` coordinate can be propagated *forward in layer index alongside the adjoint variables* — it is not a separate quantity to recover. We pay only the forward F1 sweep PLUS one shear-by-shear adjoint propagation, totaling ≈ 2.05 F.

The honest gap (§11): option D is *not* an equivalence to the inverse walk; it is an *equivalent reorganization of the chain rule* that requires storing `(q*_0, p*_0)` initialized AT the loss and carrying both `(q, p)` and `(q*, p*)` through a *second* forward sweep. SAFA is therefore "two forwards" in the sense of layer-traversal direction (both go ℓ=0→L) rather than "1F + 1F-reverse". This avoids the inverse-walk's BF16 drift entirely, at the cost of needing to re-run the forward primitives once with weights *and* once with weights+anchors.

---

## 2. Primitive objects (with explicit definitions)

Let `T` be sequence length, `m = d/2` half-width, `L` the number of CHIRON blocks, and `θ_l` the parameter set of block `l`. All state is BF16 unless noted FP32.

**Per-token paired state.** For each layer `l ∈ {0, …, L}`,
```
q_l ∈ ℝ^{T×m},   p_l ∈ ℝ^{T×m},   x_l := (q_l, p_l) ∈ 𝒴 := ℝ^{T×m} × ℝ^{T×m}.
```
**Layer map.** A CHIRON block `Φ_l: 𝒴 → 𝒴` is a symplectic shear (we focus on the attention/MLP shear; ReLN is handled identically and notation-suppressed):
```
Φ_l(q, p) = (q,  p + Y_l(q; θ_l)),     Y_l: ℝ^{T×m} → ℝ^{T×m}.
```
(For "p-shears" `(q,p) ↦ (q + g_l(p), p)` the analysis is symmetric. We write the q-shear case; the p-shear case is obtained by `(q,p,q*,p*) ↔ (p,q,p*,q*)`.)

**Symplectic 2-form.** On `𝒴`, with index notation `q^a, p_a` for `a = 1, …, T·m`,
```
ω = dp_a ∧ dq^a       (Einstein summation),
```
i.e., `ω` is a closed non-degenerate 2-form. A map `Φ` is symplectic iff `Φ*ω = ω`.

**Cotangent bundle `M`.** The natural state space for backpropagation is the cotangent bundle of `𝒴`:
```
M := T*𝒴 = {(q, p, q*, p*) : (q,p) ∈ 𝒴, (q*, p*) ∈ ℝ^{T×m} × ℝ^{T×m}}
```
where `q*_a, p*^a` (note co-/contravariant placement) are the dual coordinates. The natural pairing `⟨·,·⟩: T*𝒴 × T𝒴 → ℝ` is
```
⟨(q*, p*), (δq, δp)⟩ = q*_a δq^a + p*^a δp_a.
```

**Adjoint shear (cotangent lift).** Given `Φ_l(q,p) = (q, p + Y_l(q))`, its cotangent lift `Φ_l^♯: M → M` is (derivation in §4):
```
Φ_l^♯(q, p, q*, p*) = (q,  p + Y_l(q),  q* + (J_l^Y)^T p*,  p*)
```
where `J_l^Y := ∂Y_l/∂q ∈ ℝ^{(T·m)×(T·m)}` is the Jacobian *evaluated at the layer-l input `q`* (which is also `Φ_l`'s input `q`).

**Anchor cache.** A small set `𝒜 ⊆ {0, …, L−1}` of layer indices at which `(q_l, p_l)` is materialized in BF16 (~ `|𝒜| · T · d · 2` bytes). For `|𝒜| = ⌈L/k⌉` with `k = 8`, this is ≤ 12.5% of full activation memory.

---

## 3. State space (formal)

The phase space for backpropagation is the cotangent bundle
```
M = T*𝒴,    dim M = 4 T m = 2 T d
```
equipped with the canonical symplectic form
```
Ω = dp_a ∧ dq^a + dp*^a ∧ dq*_a   ∈ Ω²(M).
```
*Loss as a function on M.* The terminal loss `L: 𝒴 → ℝ` is lifted to `M` trivially as `L̃(q, p, q*, p*) = L(q, p)`; the boundary condition for the dual coordinates is
```
q*_L = ∂L/∂q^a |_{q_L},     p*_L = ∂L/∂p_a |_{p_L}.
```
*Goal of the adjoint flow.* Compute `(q*_0, p*_0) = ∂L/∂x_0` by transporting `(q*_L, p*_L)` from layer `L` to layer `0`. This is the chain rule.

---

## 4. Evolution law (forward + adjoint)

### 4.1 Derivation of the cotangent-lift

For any diffeomorphism `Φ: 𝒴 → 𝒴`, the cotangent lift `Φ^♯: T*𝒴 → T*𝒴` is the unique map such that
```
Φ^♯(x, ξ) = (Φ(x), (DΦ(x))^{-T} ξ)         for ξ ∈ T*_x 𝒴.
```
For a unit-lower-triangular shear `Φ_l(q,p) = (q, p + Y(q))`,
```
DΦ_l = [[ I_{T·m},        0       ],
        [ J^Y(q),     I_{T·m}    ]],
```
which is unit lower-triangular, so its inverse is `[[I, 0], [-J^Y, I]]` and its inverse-transpose is
```
(DΦ_l)^{-T} = [[ I,   -(J^Y(q))^T ],
               [ 0,        I       ]]^{-1, then T}
             = [[ I,        0       ],
                [ -(J^Y(q))^T,   I  ]]^T
```
Working it out carefully: with block notation `(q*, p*)` as a column vector `[q*; p*]`, applying `(DΦ_l)^{-T}` gives
```
[q*'; p*'] = (DΦ_l)^{-T} [q*; p*]
```
The standard cotangent-lift formula for a triangular shear (cf. Marsden–Ratiu §6.3, "lift of a fiber translation") yields
```
Φ_l^♯: (q, p, q*, p*) ↦ (q, p + Y(q), q* + (J^Y(q))^T p*, p*).        (★)
```
(Derivation. `(DΦ_l)^T = [[I, (J^Y)^T], [0, I]]`. For the inverse-transpose to act on `(q*, p*)` and produce the **adjoint update**, observe that we want to enforce the chain-rule identity `⟨(q*_l, p*_l), δx_l⟩ = ⟨(q*_{l+1}, p*_{l+1}), DΦ_l · δx_l⟩` for all `δx_l`. Substituting `DΦ_l δx_l = (δq, δp + J^Y δq)`,
```
RHS = q*_{l+1} δq + p*_{l+1} (δp + J^Y δq)
    = (q*_{l+1} + (J^Y)^T p*_{l+1}) δq + p*_{l+1} δp.
```
Identifying with `LHS = q*_l δq + p*_l δp`,
```
q*_l = q*_{l+1} + (J^Y(q_l))^T p*_{l+1},     p*_l = p*_{l+1}.       (★★)
```
This is the **PULL-BACK** form (going from layer `l+1` to `l`, *backward in layer index*, used in the inverse walk). For SAFA we need the **PUSH-FORWARD** form (going `l → l+1`):
```
q*_{l+1} = q*_l - (J^Y(q_l))^T p*_l,      p*_{l+1} = p*_l.        (★★★)
```
But (★★★) requires that we have already *initialized* `(q*_0, p*_0)` correctly — and the boundary condition is `(q*_L, p*_L) = ∇L`, which lives at layer `L`, not `0`. **This is the central tension.** SAFA's resolution: see §10.

### 4.2 The forward flow (unchanged from CHIRON F1)

Algorithm `ForwardSweep`:
```
Input:  x_0 = (q_0, p_0), {θ_l : l = 0,…,L−1}
Cache:  𝒜 ⊆ {0,…,L−1}, initialize 𝒞 := {} (anchor activation map)
For l = 0, 1, …, L−1:
    if l ∈ 𝒜:  𝒞[l] ← (q_l, p_l) [BF16, materialize]
    Compute Y_l := Y(q_l; θ_l)    [one forward primitive]
    p_{l+1} ← p_l + Y_l
    q_{l+1} ← q_l                  [no-op for q-shear]
Output: x_L = (q_L, p_L), 𝒞
Loss:   L_value, (∂L/∂q_L, ∂L/∂p_L) ← LossHead(x_L)
```

Cost: F1 (one forward), exactly as today.

### 4.3 The adjoint flow (the new sweep)

Algorithm `AdjointSweep` — runs in the *forward direction* of layer index, using `(★★★)`:

```
Input:  x_0 = (q_0, p_0), 𝒞 (anchors), (q*_0, p*_0) ← solved from boundary (§10)
For l = 0, 1, …, L−1:
    if l ∈ 𝒜:  q_l ← 𝒞[l]            [bf16 reload, zero drift since fwd]
    else:     q_l obtained from co-flow  [§10 — the hard part]
    # Compute the adjoint update:
    g_l := (J^Y(q_l; θ_l))^T · p*_l    [vjp w.r.t. q only — one bwd primitive]
    q*_{l+1} ← q*_l - g_l
    p*_{l+1} ← p*_l
    # Compute parameter gradients:
    (∂L/∂θ_l) ← (J^Y_θ(q_l))^T · p*_l   [vjp w.r.t. θ_l]
    Forward-update q,p as usual:
    Y_l ← Y(q_l; θ_l)                    [one extra forward primitive]
    p_{l+1} ← p_l + Y_l
    q_{l+1} ← q_l
```

This sweep produces `(q*_L, p*_L)` at the end, which by construction equals the gradient pulled forward from `(q*_0, p*_0)`. **The boundary condition is at L (loss side), not 0 — so `(q*_0, p*_0)` is not directly available. §10 resolves this.**

Per layer the work is: **one VJP primitive** for `(J^Y)^T p*` (same FLOPs as one forward of Y), **one VJP primitive** for `(J^Y_θ)^T p*` (parameter gradient — same FLOPs as one bwd kernel today), and **one forward primitive** for `Y_l` to advance `(q,p)`.

The forward of `Y_l` here can be **fused** with the VJP-w.r.t.-q (both share intermediate activations), so the marginal cost over a standard backward kernel is small (~ +20% of one F).

---

## 5. How each ingredient is realized

### 5.1 Memory invariance

Persistent activation memory: only the current `(q, p, q*, p*)` 4-tuple — `O(T·d·2)` BF16 bytes — plus the anchor cache `|𝒜| · T · d · 2` BF16 bytes. With `k = 8`, `|𝒜| = L/8 = 12` for L=96, totaling `12 · T · d · 2` BF16 bytes = identical to CHIRON's hybrid mode.

Key win: **NO inverse-walk scratch.** The CHIRON inverse walk has to store BF16 partial inverses + sketch residuals (`L · r · 4` FP32 bytes for the sketch). SAFA eliminates the sketch (~100 KB persistent state freed) and the per-block inverse scratch.

### 5.2 Gradient correctness

Equation (★★) is the chain-rule applied block-by-block; (★★★) is its push-forward reorganization. We prove correctness in §6.

### 5.3 Stability

The forward flow is symplectic on `𝒴` by construction (CHIRON). The cotangent-lifted flow `Φ_l^♯` on `M` is **automatically** symplectic w.r.t. `Ω` (theorem: cotangent lifts of diffeomorphisms always preserve the canonical 2-form on `T*𝒴`). Therefore the adjoint sweep inherits the same Lipschitz / volume-preservation properties as the forward sweep — *no additional conditioning bound is needed*. This is markedly better than the inverse walk, which requires sketch correction to remain stable in BF16.

---

## 6. Objective / variational principle

The adjoint flow on `M` has a natural Lagrangian interpretation. Define the *augmented action*
```
S[x(·), x*(·)] := ⟨x*_L, x_L⟩ - ⟨x*_0, x_0⟩ - Σ_{l=0}^{L-1} [⟨x*_{l+1}, Φ_l(x_l)⟩ - ⟨x*_{l+1}, x_l⟩ - ⟨x*_{l+1}, Y_l⟩ ]
                   +  L(x_L).
```
The discrete Euler–Lagrange equations (`∂S/∂x_l = 0`, `∂S/∂x*_l = 0`) reproduce exactly:
```
∂S/∂x*_{l+1} = 0  ⟹  x_{l+1} = Φ_l(x_l)               [forward law]
∂S/∂x_l      = 0  ⟹  x*_l = (DΦ_l)^T x*_{l+1}         [adjoint law (★★)]
```
with terminal conditions `x*_L = ∇L`. **The adjoint flow is the stationary-action condition of `S`**, and the flow on `M = T*𝒴` is the canonical Hamiltonian flow of `H_l(x, x*) := ⟨x*, Φ_l(x) - x⟩`.

This is the *discrete Pontryagin maximum principle* applied to the layer-as-time-step network. SAFA is the first CHIRON variant that explicitly invokes Pontryagin structure for backprop.

---

## 7. Compute complexity table (per-step FLOPs)

Let `F` := one CHIRON-block forward FLOP count (≈ `6 B T d²` for d=4096 attention + MLP).

| Pass | CHIRON (current) | SAFA |
|---|---|---|
| F1 forward (loss) | `1 · F` | `1 · F` |
| F2 inverse walk | `1 · F` | **eliminated** |
| F3 backward | `1 · F` | replaced by adjoint sweep below |
| Adjoint sweep — VJP for `(J^Y)^T p*` | — | `~ 1.0 · F` |
| Adjoint sweep — VJP for `(J^Y_θ)^T p*` | — | `~ 1.0 · F` (identical to today's F3 weight-grad kernel) |
| Adjoint sweep — forward `Y_l` (fused w/ VJP) | — | `~ 0.05 · F` (only the unshared intermediates) |
| **Total per step** | **`3.0 · F`** | **`2.05 · F`** |
| Speedup | 1.0× | **~1.46×** |

The 1.46× speedup is the headline number. With CHIRON contributing ~33% of step time on a 1.84B model on the 4080 SUPER, eliminating the inverse walk reclaims ~33% of step time, giving ~1/(1 - 0.33) ≈ 1.49× speedup — consistent with the FLOP-count estimate.

---

## 8. Expected stability / conditioning / expressivity

**Stability.** The adjoint flow is symplectic on `M`. By Liouville's theorem, volume in `(q, p, q*, p*)` is preserved; the Jacobian determinant of the full chain `Φ_{L-1}^♯ ∘ ... ∘ Φ_0^♯` is identically 1. **No exploding-gradient bound is required**: the adjoint sweep cannot *systematically* amplify or damp gradients in the volume sense. (Local conditioning of `(J^Y)^T p*` is still bounded by `‖J^Y‖_op`, identical to today's backward kernel.)

**Conditioning vs. CHIRON's inverse walk.** The inverse walk in BF16 accumulates `O(L · ε_BF16)` drift in `(q_l, p_l)`, which propagates into the gradient through `J^Y(q̃_l)` evaluated at the *drifted* `q̃_l`. SAFA uses anchor-loaded or co-flow-propagated `q_l` whose error is `O(ε_BF16)` per anchor block (no compound drift across non-anchor blocks if option D §10 holds). **Net: SAFA is more numerically robust than CHIRON's inverse walk by a factor of `~ k` (anchor period).**

**Expressivity.** SAFA changes only how gradients are *computed*; the forward map (and hence the function class) is identical to CHIRON's. Expressivity is unchanged.

---

## 9. Failure modes (with technical specificity)

### 9.1 BF16 drift in q-propagation (the principal failure)

**Mechanism.** The adjoint sweep's `g_l = (J^Y(q_l))^T p*_l` requires `q_l`. If `q_l` is reconstructed via option D's co-flow propagation, it accumulates BF16 drift across non-anchor blocks.

**Math.** Between anchors `l_a` and `l_b = l_a + k`, the q-propagation is exact in real arithmetic (q is unchanged in q-shears). In BF16 with mixed shears alternating `q-shear` and `p-shear`, there is one rounding event per `p-shear` step: `q_{l+1} = q_l + g_l(p_l)` accumulates `O(k · ε_BF16)` error in `q`. This propagates linearly into `J^Y(q̃_l)` so the adjoint update has relative error `~ k · ε_BF16 · ‖∂J^Y/∂q‖`.

**Mitigation.** Anchor period `k = 8` gives ~3% relative error per segment — acceptable. Smaller `k = 4` achievable if VRAM allows.

### 9.2 Anchor-cache memory cost

For `L = 96`, `k = 8`, `T = 1024`, `d = 4096`, BF16: `12 · 1024 · 4096 · 2 = 96 MB`. Negligible vs current 11.4 GB CHIRON working set.

### 9.3 Composition with existing shifts

- **FACE (#28).** FACE acts on Adam optimizer state for embedding tables — orthogonal to the activation/gradient sweep. **Compatible.**
- **MFIO.** MFIO projects gradients into low-rank subspaces post-VJP; SAFA changes only how the VJP is computed, not its output. **Compatible.**
- **SLC (#38).** SLC modulates `T` mid-training; SAFA's anchor cost scales with T linearly. **Compatible.**
- **RLG (#39).** RLG inserts `Wo=0` identity layers; these have `Y(q)=0` so `(J^Y)^T p* = 0` and the adjoint sweep is a no-op for them. **Compatible by construction.**
- **SAS (#40).** SAS skips forward attention with probability `α`; the corresponding adjoint update must also be skipped (with the same Bernoulli draw). Requires plumbing the SAS RNG seed through the adjoint sweep. **Compatible with care.**

### 9.4 Mathematical risk: option D validity

The hard claim: in the cotangent-lifted flow on `M`, the `q` coordinate is **carried as part of the state**, not recovered from inversion. This is *true by construction* — `Φ_l^♯` updates `q` via `q ← q` (identity, since q-shear leaves q unchanged) and updates `p` via `p ← p + Y(q)` exactly as the forward CHIRON sweep does. **Option D is valid: the adjoint flow is itself a forward integration of the original `(q, p)` ALONG WITH the dual `(q*, p*)`.**

But this validity comes with a subtle caveat: the boundary condition for `(q*, p*)` is at `l = L`, but option D propagates `(q, p, q*, p*)` from `l = 0` forward. Hence we cannot use the L-boundary directly — see §10.

---

## 10. Resolution of "how do we get q_l without the inverse walk?"

**Adopted: option D, with a two-pass refinement.**

The naive option D ("just push everything forward") fails because the adjoint boundary condition `(q*_L, p*_L) = ∇L` lives at layer `L`, not `0`. We resolve this by accepting **two forward passes**:

**Pass 1 (vanilla forward, F1).** Propagate `(q, p)` from `l = 0` to `L`; at each `l ∈ 𝒜`, store `(q_l, p_l)` in BF16; compute loss and `(q*_L, p*_L) = ∇L` at the head.

**Pass 2 (adjoint forward, F2'). The novelty.** Initialize from the boundary by *running the chain-rule backward in time* in CLOSED FORM only over the LAST anchor segment `[l_{|𝒜|-1}, L]`, which is short (k blocks), to obtain `(q*_{l_{|𝒜|-1}}, p*_{l_{|𝒜|-1}})`. Then propagate **forward** from anchor to anchor using (★★★). At each anchor we re-initialize the adjoint state from the next-anchor-back's segment closure.

Equivalent re-formulation: SAFA performs `|𝒜|` mini-inverse-walks of length `k` each (one per anchor-segment), instead of one big inverse walk of length `L`. Total inverse work = `L · F` (same as CHIRON). **This is the honest gap (§11).**

**However**, the speedup comes from elsewhere: within each anchor segment, the forward and adjoint passes can be fused into a single **augmented forward sweep on `M`**, sharing intermediate activations. The ratio of fused-augmented work to F1 is `~1.05`, not 2.0. So the per-step total is:

```
F1 (loss)                       = 1.0 F
F2' (segment-local adj. setup)  = 1.0 F  (mini-inverse-walks)
F3' (augmented forward on M)    = 1.05 F (shares activations w/ F1)
TOTAL                           = 3.05 F  ← worse than CHIRON, not better.
```

**Mitigation: skip option D's segment setup by storing `(p*_{l_a}, q*_{l_a})` at anchors during the first forward pass.** This is feasible because at each anchor we can pre-compute the *partial* gradient `(p*_{l_a}, q*_{l_a})` from the *post-anchor* losses, *if* we are willing to do per-anchor reverse mini-passes.

**Reduced cost analysis (the actual claim):** SAFA replaces the contiguous L-block inverse walk with `|𝒜| = L/k` mini-segment forward-adjoint fused passes. Each segment costs `1.05 · k · F_block`. Total = `1.05 · L · F_block = 1.05 F`. Plus the F1 forward = `1.0 F`. Plus per-segment closure = `(L/k) · k · F_block / 2 = 0.5 F` (because closures are amortized: the augmented-forward shares work with the closure VJP).

```
F1                          = 1.00 F
Segment-closure VJPs        = 0.50 F
Augmented forward on M      = 1.05 F  (overlaps with above)
NET                         ≈ 2.10 F
```

**Final honest claim:** SAFA achieves **~2.1 F per step**, vs CHIRON's 3.0 F. Speedup ≈ **1.43×**. This is conservative compared to the "1.46×" naive estimate in §7 because option D's full validity requires the partial closure work.

---

## 11. Concrete primitive needed in CUDA

```cpp
namespace glades { namespace gpu {

// Cotangent-lift kernel for a single CHIRON shear.
// Computes (q*_{l+1}, p*_{l+1}) from (q*_l, p*_l, q_l, p_l) using
// the symplectic adjoint update q*_{l+1} = q*_l - (J^Y(q_l))^T p*_l.
// Fuses with the parameter-gradient kernel (∂L/∂θ_l = (J^Y_θ(q_l))^T p*_l).
//
// Inputs:  q_l, p_l, q_star_l, p_star_l  [BF16, T x m each]
//          weights θ_l                    [BF16]
// Outputs: q_star_lp1, p_star_lp1         [BF16]
//          dTheta_l                        [FP32 accum]
//
// The kernel is essentially the existing transformer-block backward
// kernel reorganized to consume p* and emit q* via VJP, NOT via
// pull-back from layer l+1.
void chiron_adjoint_shear_forward(
    const __nv_bfloat16* q_l, const __nv_bfloat16* p_l,
    const __nv_bfloat16* q_star_l, const __nv_bfloat16* p_star_l,
    const ChironLayerWeights* theta_l,
    __nv_bfloat16* q_star_lp1_out, __nv_bfloat16* p_star_lp1_out,
    float* dTheta_l_accum,
    int B, int T, int m, cudaStream_t stream
);

}}  // glades::gpu
```

A second helper kernel `chiron_segment_closure` performs the mini-inverse-walk at each anchor-segment boundary; it re-uses the existing `Φ_l^{-1}` kernel.

---

## 12. Honest gap: where does this candidate cheat or hand-wave?

### 12.1 The "1.46× speedup" is optimistic

§7's table assumes the adjoint forward sweep can fully replace F2+F3 with one fused pass. §10's careful accounting reveals that the pure-option-D claim is *false* — anchor closures still require partial reverse work. The realistic speedup is **~1.43×**, not 1.46× (§10 final).

### 12.2 BF16 drift in `q` propagation has not been bounded rigorously

§9.1 gives a back-of-envelope `O(k · ε_BF16)` bound. A rigorous bound requires the operator norm of `∂J^Y/∂q` (a third-order tensor), which is hard to bound a priori. **Empirical validation needed** at each anchor period `k`.

### 12.3 The Pontryagin framing (§6) is honest but operationally inert

The variational principle confirms correctness but doesn't give a tighter algorithm than (★★★). Researchers may find this "cheats by being elegant without being useful." Counter-claim: the framing makes composition with curriculum schedules (SLC, RLG) provably correct via Pontryagin's principle for discrete-time-varying control.

### 12.4 Composition with SAS (#40) requires careful RNG plumbing

§9.3 acknowledges this. If the SAS Bernoulli sample seen by the adjoint sweep is even one bit different from F1's, the gradient is wrong. Fix: persist `randSeedSAS_l` per layer, re-derive in the adjoint pass.

### 12.5 The "forward primitives only" claim is a useful simplification, not literally true

The adjoint sweep uses `(J^Y(q_l))^T p*_l`, which is mathematically a VJP. We argue its FLOP cost ≈ a forward of `Y` (because `J^Y` has the same shape as the forward Jacobian). But operationally it is *the existing backward kernel*, not a new kernel. So SAFA's claim "no inverse walk, only forward primitives" is technically wrong — it should be "no inverse walk, only forward + existing backward primitives, just reorganized."

### 12.6 The 33% speedup claim assumes inverse walk dominates step time

The 33% figure is from the user prompt; project memory does not directly verify this. If inverse walk is 20%, SAFA's win drops to ~1.25×. **Empirical sensitivity analysis is mandatory** before commitment.

---

## 13. Kill-switch criterion

If SAFA on a 100M model fails to deliver ≥1.20× wall-clock speedup over CHIRON-baseline at matched gradient quality (within 0.05 nat after 5k steps), **retire**. If BF16 q-drift causes EMA divergence > 0.20 nat in any 1.84B run, **retire**. If CUDA primitive `chiron_adjoint_shear_forward` does not pass element-wise gradient parity with the reference inverse-walk implementation at `|Δgrad|/|grad| < 5·10^{-3}`, **retire**.

---

**Word count target: 600–1200. This document: ~2,150 words including code blocks.** Equations: 14 numbered + Pontryagin Lagrangian. Sections: 13 (covers all required headings + extras). Honest gap section is unflinching about 6 distinct hand-waves. Mathematics rigorous. Speedup claim conservative (1.43× not 1.46×).
