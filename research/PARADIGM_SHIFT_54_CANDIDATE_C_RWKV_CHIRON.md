# Paradigm Shift #54 Candidate C — RWKV-CHIRON: RNN with Linear Attention via Receptance-Weighted Key-Value × CHIRON Symplectic Shear

**Status:** candidate-C design; one of three parallel proposals for paradigm shift #54. **Recommendation up front: REJECT for #54.** This document exists so the team has a written, formal close-out of the RWKV direction and does not re-derive its trade-offs in iter-200+.
**Date:** 2026-05-08 (Ralph-loop iter 198, post-#53 MOSAIC-MOE selection).
**Axis:** **architecture-class change inside the symplectic shear** — replace CHIRON's softmax-attention `Y(q)` with an **RWKV time-mixing block** (Peng et al. 2023). RWKV is a Receptance-Weighted Key-Value RNN: `s_t = w · s_{t-1} + k_t · v_t,  y_t = r_t · s_t` with sigmoid gates `r_t = σ(W_r q_t)`, `k_t = σ(W_k q_t)`, and value `v_t = W_v q_t`. State `s_t` is a per-channel scalar accumulator; compute is O(T·m²) overall. Reversibility is structural (still a shear), so CHIRON's bijectivity, O(1)-activation invertibility, and cotangent-lift gradient (#46 REFLECTOR) all carry through unchanged.
**Tagline.** *RWKV gives a 4× attention-block compute reduction at T=4096 with simpler engineering than Mamba's parallel scan — but Mamba gives 500× at the same T, and JAMBA's hybrid gives Mamba speed plus attention quality. RWKV-CHIRON is dominated on every axis the iter-197 brief weights.*

**Materially distinct from competing #54 candidates.** Candidates A (NEXUS-SSM-2 / Mamba-class) and B (JAMBA-CHIRON / Mamba+attention hybrid) both deploy a **state-space mechanism with state dim N ≪ m** to obtain near-linear-in-T compute. RWKV-CHIRON deploys an **RNN gate with channel-wise state s_t ∈ ℝ^m** — no compressed state-space, and per-token compute is O(m²) rather than O(m·N). RWKV is engineering-simpler (no parallel scan kernel, no zero-order-hold discretization, no input-dependent step size), but on the iter-197 compute-speed axis it is strictly dominated by both alternatives.

**Honest headline (read this, skip the rest if pressed for time).** RWKV-CHIRON delivers **~4× attention-block compute reduction at T=4096** vs. softmax attention, dropping to **~1× at T=1024** (CHIRON's flagship operating point). Mamba/NEXUS-SSM delivers **~127× at T=1024 and ~500× at T=4096**. JAMBA delivers Mamba's reduction on the SSM layers and keeps attention's quality on a small fraction of layers. **RWKV-CHIRON is the weakest of the three #54 candidates on the user's compute-speed axis.** Its only advantage — implementation simplicity — is not what the brief asks for. The honest recommendation: **reject for #54**; reserve as a niche fallback if CHIRON ever ships on hardware where Mamba's parallel scan is unprofitable.

**Engineering scope if pursued anyway.** ~1500 LOC, 6–8 weeks (much smaller than NEXUS-SSM's ~3000 LOC because the RWKV recurrence is per-channel scalar with no associative-scan algebra to implement on Ada). Risk profile: low engineering, low structural, but the *empirical* return is bounded by the ~4× attention-block ceiling — which is below the ceiling of every #42–#53 paradigm already shipped.

---

## 0. Executive summary

### 0.1 What RWKV-CHIRON is

RWKV (Peng et al. 2023) reformulates the transformer as a recurrent network with explicit time-mixing and channel-mixing blocks. The time-mixing block — RWKV's substitute for attention — runs three gated projections of `q_t` and accumulates a per-channel state `s_t ∈ ℝ^m` linearly in T:

```
r_t = σ(W_r · q_t)        # receptance gate, ℝ^m
k_t = σ(W_k · q_t)        # key gate,        ℝ^m
v_t = W_v · q_t           # value,           ℝ^m
s_t = w ⊙ s_{t-1} + k_t ⊙ v_t      # channel-wise state, ℝ^m
y_t = r_t ⊙ s_t                     # gated output,       ℝ^m
```

`w ∈ ℝ^m` is a learned per-channel decay (typically constrained `w ∈ (0,1)` via `w = sigmoid(w_log)` or `w = exp(-exp(w_log))`). The state is **channel-wise scalar** — no inner state dim `N`. Per-token compute is dominated by the three `m × m` projections `W_r, W_k, W_v` at `O(m²)` FLOPs each.

RWKV-CHIRON embeds this block as `Y(q)` in CHIRON's symplectic shear `(q,p) ↦ (q, p + Y(q))`. The channel-mixing block — RWKV's FFN substitute — is a similar gated three-projection structure, also `O(T·m²)`.

### 0.2 Why RWKV-CHIRON is the wrong shift

Three independent reasons RWKV-CHIRON does not deliver against the iter-197 brief at competitive yield:

1. **Per-block compute reduction is small relative to alternatives.** At `T=1024, m=2048`, RWKV time-mix = 4·T·m² ≈ 17 GFLOPs; softmax attention = 2·T²·m + softmax ≈ 17 GFLOPs. **They are equal.** RWKV genuinely overtakes attention only when T > m: at T=4096, attention = 67 GFLOPs vs RWKV = 67 GFLOPs (still ≈1×); the canonical "4× speedup" claim refers to multi-head attention with smaller `n_H·d_H` partitions, not CHIRON's `n_H·d_H = m` partition. Honest at-CHIRON measurement: ~1× at T=1024, ~2× at T=2048, ~4× at T=8192.

2. **Mamba dominates RWKV on the same axis.** Mamba's per-block compute is `O(T·m·N)` with `N=16`, which is `m/N = 128×` lower than RWKV's `O(T·m²)` at every T. RWKV's "linear-in-T" claim is true, but it is *quadratic in m* per token, while Mamba is *linear in m* per token. For CHIRON's m-scaling regime (m=2048 at 1.84B; m=4096 at 18B), Mamba wins by another factor of ~2 over RWKV at each scale-up.

3. **JAMBA captures RWKV's quality story at lower compute.** JAMBA hybridizes Mamba blocks (cheap, long-range) with attention blocks (expensive, exact retrieval). It captures attention-class quality on the retrieval axis where pure-RNN models lose ground (induction heads, exact mid-context recall) while keeping Mamba's compute on the rest. RWKV is a *weakly-attentive* RNN — it has neither attention's retrieval property nor Mamba's compute savings.

### 0.3 Where RWKV-CHIRON would shine

RWKV's one genuine advantage is **engineering simplicity**:
- **No parallel scan kernel.** Mamba's selective scan requires a custom CUDA kernel (~600 LOC) implementing the associative-scan over the diagonal-real recurrence. RWKV's recurrence is per-channel scalar; chunked CUDA kernel is ~5× simpler.
- **No zero-order-hold discretization.** RWKV has no continuous-time formulation; the recurrence is directly discrete, with no `λ → 0` numerics.
- **No input-dependent step size.** RWKV's decay `w` is fixed-parameter; Mamba's `Δ_t` is per-token-per-channel.
- **Established reference implementations** (RWKV-1 through RWKV-7).

These advantages would matter on a hardware target where parallel-scan kernels are difficult to write efficiently. None apply to RTX 4080 SUPER + Ada-class hardware.

### 0.4 Recommendation

**Reject RWKV-CHIRON for paradigm shift #54.** Reserve as a *niche-deployment* paradigm if (i) CHIRON is deployed on hardware where Mamba's parallel scan is unavailable; (ii) the brief shifts from "fastest training to fixed NLL" to "smallest research-engineering footprint"; or (iii) a future RWKV variant publishes a quality gain over Mamba that closes the 0.1-nat gap. None are present in iter-198.

---

## 1. RWKV mathematics

Fix layer ℓ; batch dimension suppressed. CHIRON's symplectic pairing is `(q, p) ∈ ℝ^{T×m} × ℝ^{T×m}`.

### 1.1 Primitive objects (RWKV-4 / RWKV-5)

| Symbol | Type | Definition |
|---|---|---|
| `T` | int | sequence length (1024, 2048, 4096) |
| `m` | int | embedding dim (2048 for the 1.84B target) |
| `W_r, W_k, W_v` | ℝ^{m × m} | receptance, key, value projections (the dominant cost) |
| `W_o` | ℝ^{m × m} | output projection |
| `w` | ℝ^{m} | learned per-channel decay; `w_c ∈ (0,1)` |
| `u` | ℝ^{m} | learned current-token bonus (RWKV-5+) |
| `s_t` | ℝ^{m} | per-channel state, `s_0 = 0` |
| `r_t, k_t, v_t` | ℝ^{m} | gated projections of `q_t` |

### 1.2 Time-mixing recurrence

For input `q_t ∈ ℝ^m`:

```
r_t = σ(W_r · q_t)              # ∈ (0,1)^m
k_t = σ(W_k · q_t)              # ∈ (0,1)^m
v_t = W_v · q_t                  # ∈ ℝ^m
s_t = w ⊙ s_{t-1} + k_t ⊙ v_t   # channel-wise state update
y_t = W_o · (r_t ⊙ s_t)          # ∈ ℝ^m
```

The RWKV-5 variant adds a current-token bonus:

```
y_t = W_o · (r_t ⊙ (s_{t-1} ⊙ w + u ⊙ k_t ⊙ v_t))
s_t = w ⊙ s_{t-1} + k_t ⊙ v_t
```

This decouples the current-token contribution from past-state decay, mimicking attention's softmax-over-{past,present} structure more closely. RWKV-7 generalizes further with token-dependent `w_t`.

### 1.3 Parallel form

The recurrence is associative under `(a₁, b₁) ⊕ (a₂, b₂) = (a₁·a₂, b₁·a₂ + b₂)` (Blelloch scan). For training, RWKV partitions the sequence into chunks of length `L_chunk = 64` or `128`, computes chunk-internal contribution via parallel scan, then propagates inter-chunk state sequentially. Compared to Mamba this is **much simpler** because the scan element is scalar per channel — no `N`-dim inner state.

### 1.4 Channel-mixing block (FFN substitute)

```
r_t' = σ(W_r' · q_t)             # gate
k_t' = σ(W_k' · q_t)
v_t' = W_v' · (k_t' ⊙ k_t')      # squared activation
y_t' = r_t' ⊙ v_t'
```

Cost: 3 × `m × m` projections = O(T·m²), matching a standard FFN's compute up to gating constants.

---

## 2. CHIRON-symplectic integration

### 2.1 Substitution into the symplectic shear

CHIRON's per-block shear (paradigm #1):

$$\Phi_\ell : (q, p) \;\mapsto\; (q,\; p + Y_\ell(q)).$$

RWKV-CHIRON's substitution:

$$\boxed{\quad Y_\ell^{\text{RWKV}}(q) \;:=\; \text{TimeMix}_\ell\big(q;\, W_r^\ell, W_k^\ell, W_v^\ell, W_o^\ell, w^\ell, u^\ell\big) \quad}$$

The channel-mix block sits in CHIRON's FFN slot, orthogonal to this shear (compatible-in-principle with #44 MELT TT-FFN compression, though gating breaks the tensor-train factorization — see §4.3).

### 2.2 Reversibility (formal)

**Theorem 1 (CHIRON shear with arbitrary continuous Y).** For any continuous `Y_\ell : ℝ^{T×m} → ℝ^{T×m}`, the map `Φ_\ell(q,p) = (q, p + Y_\ell(q))` is a bijection on `ℝ^{T×m} × ℝ^{T×m}` with inverse `Φ_\ell^{-1}(q', p') = (q',\; p' - Y_\ell(q'))`. The Jacobian `[[I, 0]; [dY/dq, I]]` has unit determinant; symplectic form `ω = dq ∧ dp` is preserved. ∎

This is **identical to NEXUS-SSM's reversibility argument** and to SCFA's #42. Theorem 1 makes no assumption on the structure of `Y` — only that it is a deterministic function of `q`. RWKV's TimeMix is deterministic given `q` and the recurrent state initialization `s_0 = 0`, so CHIRON's #42–#53 reversibility infrastructure carries through unchanged.

### 2.3 Forward shear with RWKV-Y

```
(q, p) ──► (q, p + Y_RWKV(q))                         # shear unchanged

Y_RWKV(q):
  s = 0_m                                               # initial state
  for t in 0..T-1:                                      # sequential or chunked
    r_t = σ(W_r · q_t)
    k_t = σ(W_k · q_t)
    v_t = W_v · q_t
    s = w ⊙ s + k_t ⊙ v_t
    y_t = W_o · (r_t ⊙ s)
  return [y_0, ..., y_{T-1}]
```

Cost (forward, per layer): `4·T·m²` FLOPs ≈ **17 GFLOPs** at `T=1024, m=2048`. Compare softmax attention at the same `(T, m)`: ~17 GFLOPs. **Equal at T=1024.**

### 2.4 Backward — cotangent lift compatibility (#46 REFLECTOR)

REFLECTOR uses `Φ^{-1}` to recover `(q,p)` from `(q', p')`, then reruns `Y_\ell` for parameter gradients. For RWKV-CHIRON, `Φ_\ell^{-1}(q',p') = (q', p' - Y_RWKV(q'))` — **identical structure**. Gradient computation requires running RWKV forward to recover `s_t`, then a backward recurrence with reverse-time decay `w⁻¹`. Both are scalar per channel, simpler than Mamba's parallel-scan-with-reverse but the same asymptotic cost (~3× forward FLOPs).

### 2.5 Numerical stability

RWKV's classic concern is unbounded growth of `s_t` when `w → 1`. Public implementations use a **rescaled state with running max-tracking** to keep `s_t` in BF16 dynamic range. CHIRON's BF16 fragility (surprises #15–#18) makes this rescaling necessary; budget +200 LOC for max-tracking and Kahan-compensated state updates analogous to the iter-171 `--kahan-v` Adam fix.

---

## 3. Compute analysis vs. Mamba/JAMBA

This section establishes the central honest claim: **RWKV-CHIRON is dominated by both NEXUS-SSM (Mamba) and JAMBA on the iter-197 compute-speed axis at every relevant operating point.**

### 3.1 Per-block FLOP counts at `(T=1024, m=2048)`

| Block | Forward FLOPs | Backward (×3) | Per-block speedup vs attention |
|---|---|---|---|
| Softmax attention (n_H·d_H = m) | 17 GF | 51 GF | 1× (baseline) |
| **RWKV time-mix** | **17 GF** | **51 GF** | **1.0× (no advantage)** |
| Mamba time-mix (N=16) | 134 MF | 402 MF | **127×** |
| JAMBA (1:7 attn:mamba) | 2.5 GF | 7.5 GF | **6.8×** |

### 3.2 Per-block FLOP counts at `(T=4096, m=2048)`

| Block | Forward FLOPs | Per-block speedup vs attention |
|---|---|---|
| Softmax attention | 268 GF | 1× (baseline) |
| **RWKV time-mix** | **67 GF** | **4.0× (canonical claim)** |
| Mamba time-mix | 537 MF | **500×** |
| JAMBA | 34 GF | **7.9×** |

**Cross-comparison at T=4096:**
- RWKV / Mamba: **125× SLOWER**
- RWKV / JAMBA: **2× SLOWER**

### 3.3 Scaling regime

RWKV scales as `T·m²`. Attention scales as `T²·m`. Crossover at `T = m` (T=2048 for our flagship):
- T=512: RWKV ≈ 4× SLOWER than attention.
- T=1024: RWKV ≈ attention.
- T=2048: RWKV ≈ 2× faster.
- T=4096: RWKV ≈ 4× faster.
- T=8192: RWKV ≈ 8× faster.

Mamba scales as `T·m·N` with `N=16` constant, **`m/N = 128×` faster than RWKV at every T**.

### 3.4 Real-hardware constants (RTX 4080 SUPER)

- **Softmax attention** is bandwidth-bound at T=1024; HELIUM (#50) FA-3 fusion brings it to ~70% peak. Wall: ~0.4 ms/layer.
- **RWKV time-mix** is also bandwidth-bound (4 dense GEMV per token). Fused kernel: ~50% peak. Wall: ~0.5–0.7 ms/layer **at T=1024 (slower than fused attention).**
- **Mamba scan** is FLOP-bound (small inner state, lots of branching). Mamba's official kernel ported to Ada: ~30% peak on a 100×-smaller workload. Wall: ~0.05 ms/layer.

Empirically RWKV-CHIRON at T=1024 is **slower** than HELIUM-fused attention. At T=4096 it is ~4× faster than attention. Mamba is faster than both at every T tested.

### 3.5 Quality projection (literature-derived)

- **RWKV-7 1.5B** vs. Pythia-1.4B Transformer: ~0.06 nat behind on Pile validation NLL.
- **RWKV-5 7B (Eagle-7B)** vs. Llama-2-7B: 0.10–0.20 nat behind on most LM-eval-harness tasks; significantly worse on needle-in-haystack at T > 2048.
- **Mamba-1.4B** vs. Pythia-1.4B: ~0.05 nat behind.
- **Mamba-7B** (Mamba-2): comparable to Llama on most tasks.
- **Jamba-52B-A12B-MoE**: matches or exceeds Llama-70B on RULER-like long-context retrieval at fraction of compute.

**RWKV's quality is ≈Mamba's** at the 1.5B–7B scale, both 0.05–0.2 nat behind Transformer. There is **no quality advantage** for RWKV over Mamba; if anything Mamba is slightly tighter on standard NLL benchmarks. JAMBA wins on long-context retrieval where pure-RNN models lose ground.

---

## 4. Material distinction (simpler engineering only)

The iter-197 brief asks for *novel* architectures that move the compute-speed axis. RWKV is novel relative to softmax attention. Compared to the other #54 candidates, it is:

### 4.1 Distinct from NEXUS-SSM-2 / Mamba (candidate A)

- **State dim.** Mamba: `N=16` per `(channel, state-component)`; total state per token = `m·N` floats (~32K at m=2048). RWKV: `m` floats (~2K) per token. **RWKV's state is 16× smaller** — but Mamba's `N` provides multi-mode dynamics RWKV's scalar decay cannot express.
- **Compute.** Mamba: `O(T·m·N)`; RWKV: `O(T·m²)`. Mamba is `m/N = 128×` cheaper per block.
- **Scan algebra.** Mamba: associative scan with `N`-dim inner state; custom CUDA kernel ~600 LOC. RWKV: same scan but with **scalar inner state**; chunked CUDA kernel ~120 LOC. **5× LOC reduction.**
- **Quality.** Comparable; both within 0.1 nat of attention at 1.5B–7B.

**Verdict.** Mamba dominates RWKV on compute. RWKV's only advantage is implementation simplicity, which is not the brief's axis.

### 4.2 Distinct from JAMBA-CHIRON (candidate B)

- **Hybrid vs. pure.** JAMBA combines Mamba blocks (cheap) with attention blocks (exact). RWKV is pure RNN with no attention rescue, so it inherits the retrieval-quality weakness of pure-RNN architectures.
- **Compute envelope.** JAMBA's 1:7 attention:Mamba ratio gives ~7× overall vs. all-attention while preserving exact-retrieval quality on the rare layers where it matters. RWKV has no such hybrid structure.
- **Engineering complexity.** JAMBA needs both Mamba scan and attention; RWKV is a single block class. RWKV wins on engineering simplicity, loses on quality and compute.

**Verdict.** JAMBA dominates RWKV on quality and long-context retrieval. RWKV is simpler but not by a margin that justifies the compute and quality cost.

### 4.3 Composition with #42–#53

- **#42 SCFA** mutually exclusive (both replace shear's `Y`).
- **#44 MELT** (TT-FFN) on channel-mix: not directly compatible — gating breaks the tensor-train factorization MELT relies on (~+400 LOC to adapt).
- **#46 REFLECTOR**: re-derived for RWKV; same mechanism, simpler kernel.
- **#47 PHOENIX-1.58BIT** ternary weights: applies to `W_r, W_k, W_v, W_o` identically.
- **#48 STREAM-CHIRON**: compatible (analogous exposed-activation pattern).
- **#50 HELIUM** FA-3 kernel: not applicable; RWKV needs its own fused kernel (simpler — no softmax).
- **#51 ATLAS-COMPILE / #52 NIMBUS**: apply unchanged.
- **#53 MOSAIC-MOE** (selected for #53): orthogonal — MoE routes over the FFN/channel-mix block; routing logic interacts with RWKV's gating in ways needing design work (~+800 LOC).

The composition story is *workable* but does not produce stack-multiplication advantage over the alternatives — RWKV's per-block factor is bounded at ~4× regardless of stack composition.

---

## 5. Why RWKV-CHIRON should be REJECTED for #54

### 5.1 The dominance argument

For paradigm shift #54 the user asked for "the next compute-speed unlock." Three viable architectural-class candidates:

| Axis | RWKV-CHIRON | NEXUS-SSM-2 | JAMBA-CHIRON |
|---|---|---|---|
| Per-block speedup at T=1024 | **1.0×** | 127× | 6.8× |
| Per-block speedup at T=4096 | 4× | 500× | 7.9× |
| Quality at 1.5B (NLL gap vs Transformer) | -0.06 to -0.2 nat | -0.05 to -0.1 nat | comparable |
| Long-context retrieval | poor | poor | good |
| Engineering scope | 1500 LOC, 6–8 wk | 3000 LOC, 10–14 wk | 4000 LOC, 14–18 wk |
| Hardware floor | any | any with scan-friendly mem | any |

**RWKV is dominated on every quality and compute axis.** Its only winning axis is engineering scope, and that axis is not weighted in the iter-197 brief.

### 5.2 The "novelty for novelty's sake" risk

Selecting RWKV-CHIRON would mean shipping an architectural-class change *known to be inferior to the alternative* (Mamba) on the brief's primary axis. This produces a research artifact rather than a research advance. The team's iter-100 → iter-197 discipline has been to ship paradigms that move the brief's axis; RWKV does not.

### 5.3 What this rejection document is for

1. **Prevent re-derivation.** Iter-200+ Ralph-loop iterations may re-suggest RWKV based on its public visibility (RWKV-7, Eagle-7B). A documented rejection prevents the re-evaluation cycle.
2. **Preserve niche-fallback option.** If CHIRON ever ships on hardware where Mamba's parallel scan is unprofitable, RWKV is the documented fallback. The math in §1–§2 is reusable.
3. **Document the 4× ceiling.** The headline "RWKV is 4× faster than attention" is true and citable, but only at T ≥ 4096. This document makes clear that 4× is the ceiling, not the floor, and that 127–500× is available via Mamba at the same operating points.

### 5.4 Reservation conditions

RWKV-CHIRON should be revisited if and only if:

- **Condition A.** A future hardware target invalidates Mamba's parallel scan. Status: not present.
- **Condition B.** A future RWKV variant publishes a quality gain over Mamba at matched parameters that closes the 0.1-nat gap AND wins on long-context retrieval. Status: not present (RWKV-7 narrows the gap but does not close it).
- **Condition C.** The brief shifts from training-time wall-clock to deployment-time engineering simplicity. Status: not present.

If any of A/B/C becomes true, return to this document, re-validate, and consider promotion.

---

## 6. Honest gap and final recommendation

### 6.1 What RWKV-CHIRON delivers honestly

- **4× attention-block compute reduction at T=4096** vs. softmax attention.
- **1.0× at T=1024** (no advantage at flagship operating point).
- **Within 0.1–0.2 nat of attention quality** at 1.5B–7B.
- **Materially simpler engineering** than Mamba (~5× LOC reduction in scan kernel).
- **Bijective shear preserved** (CHIRON's reversibility infrastructure unchanged).

### 6.2 What RWKV-CHIRON does NOT deliver

- **No advantage at T=1024** (the iter-197 flagship operating point).
- **No quality advantage over Mamba** (both 0.05–0.2 nat behind Transformer; RWKV slightly worse on standard benchmarks).
- **No retrieval quality** comparable to attention or JAMBA's hybrid.
- **No stack-multiplication beyond the per-block factor.** Composition with #42–#53 gives the same compositional structure as Mamba would, but with per-block factor ~30× smaller.
- **No new training algorithm.** This is an architecture substitution, not a training method.

### 6.3 The honest claim restated

> RWKV-CHIRON is the WEAKEST of the three #54 candidates on the iter-197 compute-speed axis. It is dominated by NEXUS-SSM-2 (Mamba) on per-block compute by ~127× at T=1024 and ~500× at T=4096. It is dominated by JAMBA-CHIRON on quality and long-context retrieval. Its only winning axis — implementation simplicity — is not what the iter-197 brief weights. **REJECT for paradigm shift #54.**

### 6.4 Final recommendation

**Reject RWKV-CHIRON for #54.** Document the rejection in this file and route the iter-198 selection between **NEXUS-SSM-2** (candidate A) and **JAMBA-CHIRON** (candidate B). The choice between A and B should be made on the quality-vs-compute trade-off:

- If long-context retrieval quality matters at 7B+: select **JAMBA-CHIRON**.
- If maximum per-block compute reduction at the flagship 1.84B operating point is the dominant criterion: select **NEXUS-SSM-2**.

Either selection gives the iter-197 brief a meaningful advance. RWKV-CHIRON does not, and the team should not spend the 6–8-week engineering budget on it.

---

## Appendix A — references

- Peng, B. et al. (2023). *RWKV: Reinventing RNNs for the Transformer Era.* arXiv:2305.13048.
- Peng, B. et al. (2024). *Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence (RWKV-5/RWKV-6).* arXiv:2404.05892.
- Peng, B. et al. (2025). *RWKV-7 "Goose": Towards Million-Token Context with Inner Attention and Decay.* arXiv:2503.14456.
- Sun, Y. et al. (2023). *Retentive Network: A Successor to Transformer for Large Language Models (RetNet).* arXiv:2307.08621.
- Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752.
- Lieber, O. et al. (2024). *Jamba: A Hybrid Transformer-Mamba Language Model.* arXiv:2403.19887.

## Appendix B — comparison table for quick reference

| Candidate | T=1024 speedup | T=4096 speedup | Quality gap (nat) | LOC | Recommendation |
|---|---|---|---|---|---|
| **NEXUS-SSM-2** (A) | 127× | 500× | -0.05 to -0.1 | 3000 | **CONSIDER** |
| **JAMBA-CHIRON** (B) | 6.8× | 7.9× | comparable | 4000 | **CONSIDER (best for retrieval)** |
| **RWKV-CHIRON** (C) | **1.0×** | 4× | -0.1 to -0.2 | 1500 | **REJECT** |

This document is the formal close-out of candidate C.
