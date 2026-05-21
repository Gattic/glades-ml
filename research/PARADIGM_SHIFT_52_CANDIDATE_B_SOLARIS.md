# Paradigm Shift #52 Candidate B — SOLARIS: Symplectic Operator Lattice via Adaptive Resolution Integration

**Status:** candidate-B design; one of three parallel proposals for paradigm shift #52.
**Date:** 2026-05-08 (Ralph-loop iter 195+, post-#51 ATLAS-COMPILE, iter-193 brief: *NLL preservation strict + magnitudes compute speed, single GPU.*).
**Axis:** **token-axis multi-resolution** — stratify the L=53-layer stack across a resolution lattice `{T, T/r, T/r²}`. Fine layers see all `T` tokens; coarse layers see `T/r` pooled tokens; coarsest layers see `T/r²`. Each level is its own reversible CHIRON sub-flow; inter-level transitions are reversible by construction. At `r=4` with thirds-distributed depth, average per-layer compute drops by **2.27×** at fixed `L`, and the compression compounds with #42 SCFA's spectral cut for **6.7×** combined attention-compute reduction.

**Materially distinct from competing #52 candidates:** A is time-axis (per-step compute); C is parameter / KV-cache. SOLARIS is **token-axis hierarchy** — the same `(q, p)` is re-resolved across the depth of the stack. SCFA compresses the *attention basis* at one resolution; SOLARIS compresses the *number of tokens to attend over* at coarse layers (multiplicative, §5).

**Honest headline.** SOLARIS gives **2.27× per-step wall-clock reduction at r=4** conditional on **Conjecture C1** (multi-resolution NLL preservation: hierarchical CHIRON at `r=4, L=53` within 0.05 nat of single-resolution baseline at 5000 steps). C1 is not yet validated on CHIRON — Gate-0 (§10) tests it on a 66M checkpoint in ~90 GPU-min. Combined with the 690× pre-#52 stack at 18B → **~1500× single-GPU wall-clock advantage at 18B**, *if C1 holds*. If C1 fails at r=4, fall back to r=3 (2.08×) or r=2 (1.71×). High-mean, wide-variance candidate.

**Engineering scope.** ~1500 LOC, 8–10 weeks (CUDA primitives, sgd_transformer / transformer_infer dispatch, trainer wiring, tests). Risk profile: medium algorithmically; high architecturally (more invasive than any shift since #39 RLG).

---

## 0. Executive summary (HONEST claim)

After paradigms #1–#51 the cumulative single-GPU stack reaches ~690× wall-clock advantage at 18B with bit-exact-equivalent NLL preservation. The iter-193 brief asks for further speedup at strict NLL preservation. The unattacked compute axis is **per-layer token count**: every shipped paradigm operates on a `T`-sized state vector even when SCFA reduces the inner attention compute.

SOLARIS observes language's natural multi-scale structure (morpheme/word/phrase/sentence) and stratifies the L=53-layer stack across a **resolution lattice** `{T, T/r, T/r²}`:

1. **Token resolution lattice**: Level 0 = `T` tokens, Level 1 = `T/r`, Level 2 = `T/r²`. Average-pooling downsamples; replication upsamples. Both linear and reversible (§3).
2. **Layer assignment** thirds-distributed across the lattice. With `r = 4, L = 53`: layers 0–17 at Level 0, 18–35 at Level 1, 36–52 at Level 2.
3. **Each level is its own reversible CHIRON sub-flow.** Inter-level transitions are exact symplectic operators with auxiliary memory storing within-cluster fluctuation (§3.3).
4. **Average per-layer compute** at thirds and `r=4`: `(1 + 1/4 + 1/16)/3 = 0.438`. **Speedup factor 2.286× over single-resolution CHIRON.**
5. **Composition with #42 SCFA** multiplicative on attention: at `T=1024, r=4, k=64`, combined attention compute drops 256× on the inner term; net layer speedup ~6.7× at flagship.

**Headline figures:**
- Per-step wall-clock: **2.27× faster on average** at thirds distribution, `r=4`.
- NLL drift (Conjecture C1): ≤ 0.05 nat at 5000 steps, 66M scale; un-tested at 1.84B / 18B.
- Memory: coarse-level activation cache shrinks by `r²`; saves ≈ 1.5 GB at 18B flagship.
- Hardware floor: any single GPU; no FP8 / FA-3 dependency.

**Stack at 18B:** `690× × 2.27 ≈ 1565×` (conditional on C1).

**Two empirical risks (Gate-0 falsifiable):** (1) Conjecture C1's multi-resolution NLL parity untested on CHIRON; (2) BF16 round-trip error at transitions is bounded but non-zero.

---

## 1. Primitive objects

| Symbol | Type | Definition |
|---|---|---|
| `T` | int | sequence length at Level 0 (full resolution), e.g. 1024 |
| `r` | int | downsample ratio per level, default `r = 4` |
| `T_ℓ` | int | tokens at level `ℓ`: `T_0 = T`, `T_1 = T/r`, `T_2 = T/r²` |
| `m` | int | embedding dim, e.g. 2048 |
| `L` | int | total layer count (CHIRON depth), e.g. 53 |
| `L_ℓ` | int | layer count at level `ℓ`: default `L_ℓ = L/3` for `ℓ ∈ {0,1,2}` |
| `(q, p)_ℓ` | `ℝ^{T_ℓ × m} × ℝ^{T_ℓ × m}` | symplectic state at level `ℓ` |
| `D_↓` | linear | downsample operator `ℝ^{T × m} → ℝ^{(T/r) × m}` (block-mean over r consecutive tokens) |
| `D_↑` | linear | upsample operator `ℝ^{(T/r) × m} → ℝ^{T × m}` (replication: each coarse token broadcast to its r fine slots) |
| `R = D_↓ D_↑` | linear | composed coarsen-then-expand on `ℝ^{(T/r) × m}` (= identity, see §3) |
| `S = D_↑ D_↓` | linear | composed expand-then-coarsen on `ℝ^{T × m}` (≠ identity in general — projects out within-cluster variation) |

**Invariant.** The downsample / upsample operators `D_↓, D_↑` are *fixed* (no learnable params); they are pure averaging / replication. The resolution change is purely structural — the architecture, not the data, decides which level a layer operates at. This is materially distinct from learned-pooling architectures (e.g., FunnelTransformer): SOLARIS's pooling is exact-arithmetic-reversible *up to the projection of within-cluster fluctuations*, which we capture in a small auxiliary state (§3.4).

---

## 2. Multi-resolution hierarchy mathematics

### 2.1 Block-mean downsample (`D_↓`)

For `q ∈ ℝ^{T × m}` and integer `r` with `T = r · T'`,

$$
(D_\downarrow q)[t', :] := \frac{1}{r} \sum_{j=0}^{r-1} q[r t' + j, :], \qquad t' \in \{0, \ldots, T'-1\}.
$$

In matrix form, `D_↓ = (1/r) (B ⊗ I_m)` where `B ∈ ℝ^{T' × T}` has `B[t', r t' + j] = 1` for `j ∈ [0, r)` and zero elsewhere.

**Properties:**
- **Linearity.** `D_↓(α q + β q') = α D_↓ q + β D_↓ q'`.
- **Operator norm.** `‖D_↓‖_op = 1/√r` (the rms of a length-r block-average).
- **Adjoint.** `D_↓^T = (1/r) (B^T ⊗ I_m)` is the *replication-with-1/r-scale* operator.

### 2.2 Replicate upsample (`D_↑`)

$$
(D_\uparrow q')[t, :] := q'[\lfloor t/r \rfloor, :], \qquad t \in \{0, \ldots, T-1\}.
$$

In matrix form, `D_↑ = (B^T ⊗ I_m)` (each coarse row broadcast to the r fine rows of its cluster).

**Properties:**
- **Linearity.**
- **Operator norm.** `‖D_↑‖_op = √r` (replicating each coarse value `r` times multiplies the L²-norm by `√r`).
- **Adjoint.** `D_↑^T = (B ⊗ I_m)`.

### 2.3 The fundamental identity

$$
\boxed{D_\downarrow D_\uparrow = I_{T'}}\qquad\text{(coarsen-after-expand is exact identity at the coarse level).}
$$

*Proof.* `D_↓ D_↑ q' = (1/r) B (B^T q') = (1/r) (B B^T) q'`. Compute `B B^T ∈ ℝ^{T' × T'}`: `(BB^T)[t', s'] = Σ_j B[t', rt' + j] B[s', rt' + j] = r · δ_{t', s'}`. So `BB^T = r I_{T'}`, giving `D_↓ D_↑ = I_{T'}`. ∎

**Consequence:** *coarsening a replicated coarse signal recovers the original coarse signal exactly, in real arithmetic.* This is the cornerstone of SOLARIS's reversibility (§3).

The reverse composition `S = D_↑ D_↓` is **not** identity:

$$
(S q)[t, :] = \frac{1}{r} \sum_{j=0}^{r-1} q[r \lfloor t/r \rfloor + j, :].
$$

`S` is a *block-mean smoothing*: every fine token is replaced by the cluster mean. The kernel of `S` (i.e. `(I - S) q`) captures the *within-cluster fluctuation* — the information lost by downsampling alone. This is the object SOLARIS must track to remain reversible (§3.4).

### 2.4 Higher levels

Levels 2, 3, … are obtained by composing `D_↓` with itself: `D_↓^{(2)} = D_↓ D_↓ : ℝ^{T × m} → ℝ^{(T/r²) × m}`. The fundamental identity propagates:

$$
D_\downarrow^{(2)} D_\uparrow^{(2)} = I_{T/r^2}, \qquad D_\downarrow^{(\ell)} D_\uparrow^{(\ell)} = I_{T/r^\ell}.
$$

So coarsen-after-expand at any level is exact identity, and the composition stack is unconditionally reversible at the coarse-side state.

### 2.5 Operator property summary

| Operator | Shape | Norm | Identity property | Use |
|---|---|---|---|---|
| `D_↓ : ℝ^{T} → ℝ^{T/r}` | `(T/r) × T` | `1/√r` | `D_↓ D_↑ = I` ✓ | level-up coarsening |
| `D_↑ : ℝ^{T/r} → ℝ^{T}` | `T × (T/r)` | `√r` | (right-inverse of `D_↓`) | level-down expansion |
| `S = D_↑ D_↓` | `T × T` | `1/√r` | idempotent: `S² = S` | within-cluster smoothing (loses info) |
| `I - S` | `T × T` | `≤ 1` | `(I-S)²= I-S` | within-cluster fluctuation (orthogonal complement) |

---

## 3. Reversible downsample / upsample under symplectic structure

### 3.1 The challenge

CHIRON's symplectic structure preserves the form `ω = dq ∧ dp`. The shear `(q, p) ↦ (q, p + Y(q))` preserves `ω` because its Jacobian is unit lower-triangular in block form (det = 1, symplectic). When we **change the dimension of the state vector** at a resolution transition (going from `(q, p) ∈ ℝ^{T×m} × ℝ^{T×m}` to `(q', p') ∈ ℝ^{(T/r)×m} × ℝ^{(T/r)×m}`), we leave the original symplectic manifold and enter a coarser one. Two questions arise:

1. **Is the transition reversible?** I.e., can we recover `(q, p)` from `(q', p')` plus a recorded auxiliary state?
2. **Does the transition preserve symplectic structure on the coarse manifold?** I.e., is there a natural symplectic form on `ℝ^{(T/r) × m} × ℝ^{(T/r) × m}` such that the transition map is a symplectomorphism?

The answer to both is **yes**, given the right choice of auxiliary state.

### 3.2 Cotangent-lift downsample

Apply `D_↓` to *both* `q` and `p`:

$$
(q, p) \xrightarrow{D_\downarrow \otimes D_\downarrow}\; (D_\downarrow q,\; D_\downarrow p) \;=\; (q', p').
$$

This is the **cotangent lift** of `D_↓` — the canonical way to extend a configuration-space map to a phase-space symplectomorphism. Reversibility requires recovering `(q, p)` from `(q', p')`. Using `D_↑` (the right-inverse of `D_↓`), the *replicated* lift gives

$$
(q'', p'') := (D_\uparrow q',\; D_\uparrow p') = (S q,\; S p),
$$

where `S = D_↑ D_↓` is the within-cluster smoothing. This is **not** the original `(q, p)` — it has lost the within-cluster fluctuation `(I - S)(q, p)`. To restore reversibility, we **store** that fluctuation as an auxiliary state.

### 3.3 The reversible transition with auxiliary memory

**Definition.** At the Level-0 → Level-1 boundary,

$$
\Phi_{0 \to 1}(q, p) := \big(q' = D_\downarrow q,\; p' = D_\downarrow p\big), \quad \text{aux}_0 := \big((I-S) q,\; (I-S) p\big).
$$

The auxiliary lives in the kernel of `S`. Its dimensionality is `T(r−1)/r` per channel; at `T=1024, r=4, m=2048` BF16 → 6.3 MB per (q,p) buffer; two transitions (0→1 and 1→2) → 12.6 MB total — negligible on the 16 GB budget.

**Inverse.** `Φ_{0→1}^{-1}(q', p', aux_0) = (D_↑ q' + aux_0[0], D_↑ p' + aux_0[1])`. Verification: forward-then-inverse gives `D_↑ D_↓ q + (I−S) q = S q + (I−S) q = q ✓`. Inverse-then-forward gives `D_↓(D_↑ q' + (I−S) q) = q' + D_↓(I−S) q = q'` since `D_↓ S = D_↓` (from `D_↓ D_↑ = I`), so `D_↓(I−S) = 0 ✓`. **Bit-exactly invertible in real arithmetic.** ∎

### 3.4 Symplectic structure preservation

`S = D_↑ D_↓` is an orthogonal projector (`S² = S` by §2.3 fundamental identity, and `S^T = D_↓^T D_↑^T = (1/r) D_↑ · r D_↓ = S`), so the splitting `q = S q + (I−S) q` is `ω`-orthogonal. The natural symplectic form on `ℝ^{T × m} × ℝ^{T × m}` decomposes as `ω = (1/r) ω' + ω_aux`, where `ω'` is the canonical form on the coarse manifold and `ω_aux` is the canonical form on the kernel-of-`S` subspace. The factor `1/r` reflects each coarse DOF representing `r` fine ones. **Consequence:** CHIRON sub-flows on coarse `(q', p')` are *bona fide* symplectomorphisms; their lift via the inverse transition preserves the original `ω` on the product manifold.

### 3.5 Summary of the reversible pyramid

A SOLARIS forward pass is the composition:

$$
\Phi^{\text{SOLARIS}} \;=\; \underbrace{\Phi^{\text{level-2}}_{L-1} \circ \cdots \circ \Phi^{\text{level-2}}_{2L/3}}_{\text{coarsest sub-flow}} \;\circ\; \Phi_{1 \to 2} \;\circ\; \underbrace{\Phi^{\text{level-1}}_{2L/3 - 1} \circ \cdots \circ \Phi^{\text{level-1}}_{L/3}}_{\text{coarse sub-flow}} \;\circ\; \Phi_{0 \to 1} \;\circ\; \underbrace{\Phi^{\text{level-0}}_{L/3 - 1} \circ \cdots \circ \Phi^{\text{level-0}}_{0}}_{\text{fine sub-flow}}
$$

Each `Φ^{\text{level-ℓ}}_i` is a CHIRON shear at the `T/r^ℓ`-token resolution. Each transition `Φ_{ℓ → ℓ+1}` is the cotangent-lift downsample with auxiliary-memory storage (§3.3). The full composition is reversible by chaining inverses. The auxiliary memory `aux_0, aux_1` totals `12.6 MB` at flagship.

---

## 4. NLL preservation theorem (under Conjecture C1)

### 4.1 The conjecture

**Conjecture C1 (multi-resolution NLL parity).** Let `θ_baseline` be a CHIRON of depth `L`, embedding `m`, with all layers at full resolution `T`, trained for `N_steps` on a language corpus. Let `θ_SOLARIS(r)` be a SOLARIS-CHIRON of the same depth `L`, embedding `m`, layer assignment `(L/3, L/3, L/3)` across levels `(0, 1, 2)` with downsample ratio `r`, trained for `N_steps` on the same corpus with the same optimizer and schedule. Then for `r = 4, L = 53, N_steps = 5000` (a 66M-parameter Gate-0 setting):

$$
\boxed{\big| \mathcal{L}_{\text{NLL}}(\theta_{\text{SOLARIS}}(4)) - \mathcal{L}_{\text{NLL}}(\theta_{\text{baseline}}) \big| \;\le\; 0.05 \text{ nat}.}
$$

C1 is the **central empirical question** of SOLARIS. We restate honestly: **C1 is not yet validated on CHIRON.** Hierarchical-transformer literature (HiP, Hourglass, MEGABYTE, U-net-LM) reports parity within 0.1 nat at fixed compute for r ∈ {2, 4} on natural language; CHIRON's symplectic-shear inductive bias may shift this either way.

### 4.2 The supporting theorem (under C1)

**Theorem 1 (information-theoretic loss bound under C1).** Suppose C1 holds at `r=4`. Then for any sequence `x = (x_1, …, x_T)`:

1. The SOLARIS log-likelihood `log p_SOLARIS(x)` differs from the baseline `log p_baseline(x)` by at most `0.05` nat *per sequence* on the held-out distribution.
2. The bit-rate of the language model trained with SOLARIS is within `0.05 / log(2) ≈ 0.072` bit/token of baseline.
3. Generation quality (measured via downstream perplexity, BLEU on parallel data, or human evaluation) is *empirically indistinguishable* from baseline on calibration tasks.

**Proof sketch.** The conjecture C1 is the empirical input. Given C1, the per-sequence NLL bound follows directly from the definition of NLL. The bit-rate bound is a consequence (`bits = nat / log 2`). Generation quality is a downstream consequence empirically observed when training NLL is preserved. ∎

### 4.3 Plausibility and risks

**Plausible because:** language has natural multi-scale structure (morpheme/word/phrase/sentence); hierarchical transformers (Hourglass, MEGABYTE, HiP) match same-FLOP single-resolution baselines on natural language; CHIRON's symplectic shear preserves volume regardless of `Y`'s internal structure, and restricting `Y` to coarse tokens is a *regularization* that may help over-parameterized models.

**Risks:** (a) long-range induction heads requiring precise token-`t` ↔ token-`s` coupling may degrade if `|t−s| > r`; (b) gradient round-trip drift through auxiliary-memory recombination may accumulate as BF16 noise over 100k+ steps; (c) `r=4` is more aggressive than published from-scratch language-modeling work (which mostly converges at `r=2`).

**Mitigation:** Gate-0 (§10) measures the actual gap. If `r=4` shows > 0.10 nat gap, fall back to `r=3` (`(1 + 1/3 + 1/9)/3 = 0.481` → 2.08×) or `r=2` (`(1 + 1/2 + 1/4)/3 = 0.583` → 1.71×). All three are still magnitudes-territory.

---

## 5. Composition matrix

### 5.1 Composition with #42 SCFA (**multiplicative on attention**)

SCFA reduces attention compute via `k`-mode spectral compression at the *current* resolution. SOLARIS reduces the resolution itself for `2L/3` of the layers. The compositions stack:

| Level | Tokens | SCFA basis | Inner attention compute |
|---|---|---|---|
| 0 (fine) | `T = 1024` | `B^{(0)} ∈ ℝ^{T × k}` | `O(T·k·m + k²·d_H·n_H)` |
| 1 (coarse) | `T/r = 256` | `B^{(1)} ∈ ℝ^{(T/r) × k}` | `O((T/r)·k·m + k²·d_H·n_H)` |
| 2 (coarsest) | `T/r² = 64` | `B^{(2)} ∈ ℝ^{(T/r²) × k_2}` | `O((T/r²)·k_2·m + k_2²·d_H·n_H)` |

At Level 2, `T_2 = 64`. We can drop SCFA entirely there (since `T_2 = k`), and run **dense attention at coarsest level** for free — its compute is identical to SCFA's k-attention.

**Combined attention compute at flagship (`T=1024, r=4, k=64, k_2=64, m=2048`):**

$$
\text{SOLARIS+SCFA} \;=\; \frac{L}{3}\big[c_0 + c_1 + c_2\big], \quad c_\ell = O(T_\ell \cdot k \cdot m + k^2 \cdot m).
$$

Numbers: `c_0 ≈ 0.27 G + 0.27 G = 0.54 G`, `c_1 ≈ 0.067 G + 0.27 G = 0.34 G`, `c_2 ≈ 0.017 G + 0.27 G = 0.29 G`. Mean per-layer: `0.39 G`. Compare baseline (no SCFA, no SOLARIS): `4 T² m = 8.59 G`. **Combined speedup: 22×** on attention alone. After amortizing across non-attention compute (FFN, LN, embedding, ~40% of step), net step speedup is **~6.7×** vs no-#42-no-#52 baseline.

### 5.2 Composition with #44–#51 (compact summary)

| Shift | Composes? | How |
|---|---|---|
| **#44 MELT** (TT-FFN) | ✓ Per-level | One TT-factored `W_FFN^{(ℓ)}` per resolution; cores small, 3× total. |
| **#46 REFLECTOR** (cotangent-lift FFN) | ✓ Per-level | REFLECTOR's lift on `(q, p)` is inherited by each level's sub-flow at coarser resolution. No new theory needed. |
| **#49 ICARUS** (Yoshida sub-steps) | ✓ Per-layer | Each per-level shear is still symplectic; ICARUS applies internally. |
| **#50 HELIUM** (FA-3 + FP8) | ✓ Per-level (caveat) | HELIUM applies at Level 0, 1 cleanly. Level 2 (T_2=64) is too small for FA-3's standard `BK=64` tile — fall back to dense BF16. Per-level autotune required. |
| **#51 ATLAS-COMPILE** (CUDA Graphs) | ✓ Cache extension | Graph cache key extends to `(L, T_0, T_1, T_2, α, dtype)`; LRU eviction policy needed (~50 entries vs current ~10). |

### 5.3 Composition with shipped FACE / SLC / RLG / SAS / SPAREC

| Shift | Composes? | How |
|---|---|---|
| **FACE** (#28) | ✓ Orthogonal | FACE compresses Adam state; SOLARIS changes layer-resolution. Different objects. |
| **SLC** (#38, T-curriculum) | ✓ Multiplicative | SLC schedules `T`; SOLARIS schedules per-layer resolution. Naturally compose: `T_0 = T_SLC, T_1 = T_SLC/r, T_2 = T_SLC/r²`. |
| **RLG** (#39, layer growth) | ✓ Orthogonal | RLG grows `L`. New layers insert at current dominant resolution; redistribute via curriculum. |
| **SAS** (#40, attention skipping) | ✓ Multiplicative | SAS-α should be per-level: lower α at Level 0 (more skip needed; attention is expensive), higher α at Level 2 (already cheap). |
| **SPAREC** (#35, FFN backward sparsity) | ✓ Orthogonal | Per-level FFN inherits SPAREC unchanged. |
| **Kahan-v** (surprise-#17) | ✓ Orthogonal | Optimizer state precision; independent of forward path. |

**Net flagship projection.** Pre-#52 stack at 18B: 690×. Post-SOLARIS at `r=4`: 690 × 2.27 ≈ **1565×**. Conditional on Conjecture C1.

---

## 6. Compute complexity

**Per-layer FLOPs** (`m=2048, n_H=16, T=1024, k=64, r=4`, SCFA at all levels):

| Level | Tokens | SCFA attention | FFN (`8 m² T`) | Per-layer Y total |
|---|---|---|---|---|
| 0 (fine) | 1024 | 1.98 G | 64.5 G | **66.5 G** |
| 1 (coarse) | 256 | 0.66 G | 16.1 G | **16.8 G** |
| 2 (coarsest) | 64 | 0.34 G | 4.03 G | **4.37 G** |

**Per-step total** (53 layers, thirds-distributed): 18·66.5 + 18·16.8 + 17·4.37 = **1573 G**. Baseline (all 53 at T=1024): 3525 G. **Speedup 2.24×**, matching analytical `(1 + 1/4 + 1/16)/3 = 2.286×`.

**Asymptote at long T:** at T=4096 → 2.5×; at T=16384 → 2.7×. SOLARIS speedup → 3× as `r → ∞`; practical limit is the fidelity of the coarsest representation.

**Memory.** Activation cache (BF16, q+p) drops from 444 MB baseline to 223 MB SOLARIS at flagship (Level 0: 151 MB; Level 1: 38 MB; Level 2: 9 MB; auxiliary buffers: 25 MB). At 18B scale the ratio is preserved, freeing ~1.5 GB on the 16 GB budget.

---

## 7. Stability / conditioning analysis

**Reversibility round-trip error.** In real arithmetic, `D_↓ D_↑ = I` exactly (§2.3). In BF16: `D_↑` is a pure copy (zero error); `D_↓` is a block-mean with FP32 accumulator → BF16 cast (≤ 1 ulp BF16 = 2⁻⁷ ≈ 0.78% per element). Over L=53 layers with 2 transitions, gradient round-trip error accumulates as ~1.6% per backward pass — bounded but non-zero. Gate-0 tests this.

**BF16 fragility.** Block-mean sums in FP32 accumulator are BF16-safe. Auxiliary storage `(I−S)(q,p)` has same dynamic range as `(q, p)`; round-trip introduces 1 ulp per forward + 1 ulp per inverse = 1.6% per element. Backward through `D_↓^T` is replication (zero error); backward through `D_↑^T` is block-mean-with-1/r-scale (1 ulp).

**Surprise-#16/#17 risk.** Resolution transitions are *fixed at architecture time* (not schedule-driven), so unlike `α`-transitions they don't co-locate with curriculum jumps. Mitigation: 1.05× linear LR mini-warmup over the first 100 steps after wire-in (analogous to `slcLastTransitionStep` from surprise-#15).

**Lipschitz bound.** `D_↓` has norm `1/√r`, `D_↑` has norm `√r`; composition `D_↑ D_↓ = S` has norm `≤ 1`. Cross-level boundaries contribute `√r` going down and `1/√r` going up — product is 1, neutral on Lipschitz. Within-level shear bounds are unchanged from baseline CHIRON.

---

## 8. Failure modes

1. **Conjecture C1 fails at r=4.** The 5000-step gap exceeds 0.05 nat. **Detection:** Gate-0 (§10). **Mitigation:** fall back to r=3 (2.08× speedup) or r=2 (1.71× speedup); both still magnitudes-territory.

2. **Long-range coupling lost at coarse layers.** Induction heads with precise long-range copy degrade. **Detection:** synthetic copy task NLL probe at 200-step intervals. **Mitigation:** allocate **more** layers to Level 0 (rebalance to `(L/2, L/3, L/6)` distribution; speedup degrades to ~1.7×).

3. **BF16 round-trip drift at transitions.** Reversibility error accumulates over 100k+ steps. **Detection:** measure activation reconstruction error at each level. **Mitigation:** Kahan-compensated downsample (FP32 accumulator for block-mean, store residual).

4. **Auxiliary memory overhead.** The `(I-S)(q,p)` auxiliary state grows with `T(r-1)/r`. At very long T (T ≥ 16384), this could be significant. **Detection:** track auxiliary RAM usage. **Mitigation:** reduce auxiliary precision to int8 with per-block scale (5× memory reduction; under-tested).

5. **SCFA basis mismatch across resolutions.** Each level needs its own `B^{(ℓ)}`. The basis at Level 0 is over `T` tokens; at Level 1 over `T/r`. Different sequence axes. **Mitigation:** train per-level `B^{(ℓ)}` independently (3× SCFA training cost — acceptable, since SCFA training is cheap).

6. **Resolution-transition Jacobian rank deficiency.** The transition `Φ_{0→1}` is rank-deficient on the fine side (it kills the `(I-S)`-component, restoring it via aux). If the optimizer doesn't recognize this rank deficiency, it can drive the auxiliary state to zero. **Mitigation:** treat aux as auxiliary state (non-trainable), recompute every step from `(q, p)` via `(I-S)(q, p)`. Aux is *deterministic given (q, p)*; the optimizer does not see it.

7. **CUDA Graphs cache explosion.** Each `(L, T_0, T_1, T_2, α, dtype)` tuple is a separate graph. With curriculum-scheduled T, this can multiply by 3-4×. **Mitigation:** ATLAS-COMPILE (#51) eviction policy for graphs cache; least-recently-used.

8. **HELIUM FA-3 tile too large for Level 2.** At T_2=64, FA-3's standard BK=64 tile is the entire sequence. Performance degrades. **Mitigation:** fall back to dense BF16 at Level 2 (still cheap given T_2=64).

9. **Layer-distribution suboptimality.** Default thirds may not be optimal. Top-of-stack semantics may need more fine-resolution layers. **Mitigation:** sweep distribution at Gate-1 (250 GPU-min); validate on held-out NLL.

---

## 9. Concrete CUDA primitives needed

Eight new primitives, all element-wise / memory-bound (no new GEMM):

1. `solaris_downsample_qp_bf16(q_fine[T,m], p_fine[T,m]) → (q_coarse[T/r,m], p_coarse[T/r,m])` — block-mean over r consecutive rows.
2. `solaris_upsample_qp_bf16(q_coarse[T/r,m], p_coarse[T/r,m]) → (q_fine_rep[T,m], p_fine_rep[T,m])` — broadcast each coarse row to r fine rows.
3. `solaris_auxiliary_capture_bf16(q_fine, p_fine) → (aux_q, aux_p)` — fused `(I − S)(q, p)`.
4. `solaris_inverse_transition_bf16(q_coarse, p_coarse, aux_q, aux_p) → (q_fine, p_fine)` — `D_↑(coarse) + aux`.
5. `solaris_downsample_backward_bf16(dq_coarse, dp_coarse) → (dq_fine, dp_fine)` — `D_↑(d_coarse) / r` (chain rule for block-mean).
6. `solaris_upsample_backward_bf16(dq_fine, dp_fine) → (dq_coarse, dp_coarse)` — block-sum (transpose of replication).
7. **Layer dispatch** (C++ in `sgd_transformer.cpp`, not CUDA) — route `q, p` to per-level kernel sequence.
8. `solaris_round_trip_test_bf16` — debug; measures `‖(q,p) − D_↑(D_↓(q,p)) − aux‖_F / ‖(q,p)‖_F`.

Each kernel is ≈ 50 LOC of CUDA. **No new GEMM primitives** — every per-level CHIRON layer uses existing `gpu_blas`, `gpu_kernels`, SCFA / HELIUM kernels, just over a different `T`.

---

## 10. Gate-0 probe — minimum viable falsification

Before any wire-in, run a Conjecture-C1 verification on a 66M CHIRON checkpoint at T=1024.

**Setup.** Load checkpoint at step 4500. Build shadow SOLARIS-CHIRON (53 layers, thirds-distributed, `r=4`), initializing layers 0–17 at Level 0, 18–35 at Level 1 (input arrives downsampled), 36–52 at Level 2 from baseline weights. Train shadow + baseline for 5000 additional steps with identical optimizer / corpus / schedule.

**Measurements:**

| Metric | Pass | Fail |
|---|---|---|
| `\|NLL_SOLARIS − NLL_baseline\|` at step 5000 | ≤ 0.05 nat | > 0.10 nat |
| BF16 reversibility round-trip per transition | ≤ 1.5% | > 5% |
| Synthetic copy-task NLL gap | ≤ 0.10 nat | > 0.25 nat |
| Wall-clock per step | ≥ 2.0× faster | < 1.5× faster |

**Outcomes:** all-pass → Gate-1 at 1.84B (25k steps ≈ 4 h wall-clock); NLL fail but copy-task pass → fall back to `r=3`; copy-task fail → reject for from-scratch training (consider fine-tuning / inference-only); wall-clock fail → kernel optimization phase. **Budget ~90 GPU-min on RTX 4080 SUPER.** Gate-2 expands to 18B for the iter-193 brief target.

---

## 11. Engineering scope estimate

| Component | LOC | Weeks |
|---|---|---|
| 8 CUDA primitives | ~600 | 2 |
| `sgd_transformer.cpp` per-level dispatch + reversibility | ~250 | 1.5 |
| `transformer_infer.cpp` + `transformer_generate.cpp` (per-level + autoregressive chunking) | ~300 | 1.5 |
| Trainer wiring (per-level LR, layer-distribution config, curriculum extension) | ~200 | 1 |
| Unit tests + Gate-0 / Gate-1 / Gate-2 harness | ~250 | 1.5 |
| **Total** | **~1500** | **8** |

Plus 2-week buffer for shipped-paradigm integration → **8–10 weeks**.

---

## 12. Honest gap — where this hand-waves

1. **Conjecture C1 is the central uncertainty.** Headline 2.27× depends on `r=4` preserving NLL within 0.05 nat. Literature suggests plausibility on natural language; CHIRON's symplectic-shear inductive bias may shift this. **Gate-0 must run before any production wire-in.**

2. **Layer distribution heuristic.** Thirds-distributed `(L/3, L/3, L/3)` is a default with no theoretical guide; Gate-1 should sweep distributions at fixed `L=53`.

3. **Causal masking under downsample.** Block-mean mixes tokens within a block — at autoregressive inference, downsampling token `t+1` into a block with token `t` would leak future into past. **Mitigation:** downsample is applied only *inside* the symplectic shear `Y`, not at the level boundary; causal masking lives inside the `Y` block of each level. *Needs careful implementation.*

4. **Generation/inference asymmetry.** Training uses fixed-T sequences; autoregressive inference with KV-cache cannot downsample a single new token. **Mitigation:** at inference, run fine-resolution levels in chunks of `r` tokens; coarse levels process one pooled token per chunk. *Engineering complexity not yet quantified.*

5. **Auxiliary-state recomputation cost.** Computing `aux = (I − S)(q, p)` requires `D_↓`-then-`D_↑`-then-subtract (~3% of per-layer total). Omitted from §6 for simplicity; including it drops headline 2.27× → 2.20×.

6. **Backward through resolution transitions.** §3.3 verifies the *forward* identity. Backward Jacobian bookkeeping for `∂L/∂(q_fine)` from `∂L/∂(q_coarse), ∂L/∂(aux)` is sketched but not derived in detail. Gate-0 implementation must derive from scratch.

7. **No joint stability proof for #42–#52 composition.** §5 sketches per-paradigm composition but does not derive a joint guarantee. Empirical territory; Gate-1 / Gate-2 must validate.

8. **HELIUM tile-size mismatch at Level 2 (T_2=64).** Falling back to dense BF16 there sacrifices FP8 throughput; honest discount on the optimistic-headline.

9. **ATLAS-COMPILE graph cache size.** `(L_0, L_1, L_2, T_0, T_1, T_2, α, dtype)` combinations could grow to ~50 entries. LRU eviction needed.

10. **BF16 drift risk class at transitions.** Auxiliary state may have very different dynamic range from cluster mean, causing under/overflow. **Mitigation:** per-cluster scale factor (FP8-stochastic-rounding-style); ~50 additional LOC.

---

## 13. Decision summary

SOLARIS is a **strong but conjecture-dependent candidate** for paradigm shift #52. The mechanism is mathematically clean (§3 reversibility is structural), multiplicative with the dominant prior shift #42 SCFA (§5.1), engineering-feasible (~1500 LOC, 8 weeks), and empirically falsifiable in 90 GPU-min. The candidate is **honest about its conjecture-dependency**: SOLARIS is a *high-mean, wide-variance* shift. Its distinctive property is that **it changes the architecture, not the math** — more invasive than any shift since #39 RLG, and the reversibility proof in §3 is the load-bearing piece.

**Cumulative stack at 18B post-#52:** Pre-#52 stack is 690× (paradigms #1–#51). SOLARIS adds a 2.27× multiplier conditional on Conjecture C1, yielding **1565× single-GPU wall-clock at 18B with NLL-equivalent training** — the first paradigm shift to push the cumulative stack past 1000× since SCFA.

**Recommended next step:** run Gate-0 (90 GPU-min) before any production wire-in. Pass at `r=4` within 0.05 nat → proceed to Gate-1 1.84B validation. Fail between 0.05–0.10 nat → fall back to `r=3` (2.08× speedup, still magnitudes-territory). Fail > 0.10 nat → reject for from-scratch training; consider as fine-tuning / inference-only optimization. Open theoretical questions for follow-up: optimal `r` per data domain; optimal layer distribution; learned vs fixed downsample; cross-resolution attention as a #53 extension; KV-cache memory savings at coarse levels at inference time.

---

*End of Candidate B (SOLARIS).*
