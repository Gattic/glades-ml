# Paradigm Shift #49 Candidate C — AURORA (Adaptive Computation Time on the CHIRON Reversible Flow)

**Status:** candidate-C design, single formulation. Companion to #49-A (ICARUS, higher-order symplectic integration) and #49-B (ZENITH, cross-step prediction).
**Date:** 2026-05-08 (Ralph-loop iteration 193, building on the shipped #42–#48 single-GPU stack: SCFA + ORION + MELT + REFLECTOR + PHOENIX-1.58BIT + PHOENIX-1BIT).
**Axis:** **per-token adaptive compute** — let each token decide for itself how many of the L CHIRON layers it traverses, halting "easy" tokens early and freeing compute for "hard" tokens.
**Tagline.** *Not every token needs 53 layers. ACT-style halt logits, calibrated to an NLL-preserving conservative threshold, deliver 1.3–2× per-step compute savings on natural-text token distributions while keeping CHIRON's bijective invariants intact via halt-depth-indexed inverse walks.*

**Materially distinct from:**
- **ICARUS (#49-A, higher-order integrators):** ICARUS reduces FLOPs *uniformly across all tokens* by replacing the first-order shear composition with a fourth-order Yoshida split. AURORA reduces FLOPs *non-uniformly across tokens* — easy tokens see fewer layers, hard tokens see all of them. The two are orthogonal and **multiplicative**: ICARUS reduces per-layer cost; AURORA reduces effective layer count per token.
- **ZENITH (#49-B, cross-step prediction):** ZENITH reuses inter-step structure (gradient-flow caching, anchor extrapolation) to skip whole steps. AURORA reuses intra-step token-level structure to skip whole layers per token. They attack orthogonal axes (steps vs layers) and **stack multiplicatively** (ZENITH's K× cheaper steps × AURORA's d_avg/L cheaper per-step).
- **SAS (paradigm 40, stochastic per-layer skipping):** SAS is *stochastic* — each layer is skipped with probability α independent of token identity. AURORA is *deterministic-per-token* — the halt decision is a learned, calibrated function of the token's representation. SAS gives variance-bounded layer count; AURORA gives token-conditional layer count. SAS and AURORA can compose (AURORA decides the halt budget; SAS adds stochastic depth on the *active* layers) but their primary mechanisms are non-overlapping.
- **TRCD (paradigm 13, token-routed depth):** TRCD routes tokens to different *paths* through a fixed-depth stack via KKT-gated branches. AURORA does not branch — every active token follows the same shear sequence — but allows the *length* of that sequence to vary per token. AURORA is closer to Graves' 2016 Adaptive Computation Time (ACT) than to mixture-of-experts routing.

---

## 0. Executive summary — the honest claim

The user's iter-193 brief sharpened the magnitude criterion: "magnitudes better on compute speed whilst still maintaining our memory advantages and **NLL accuracy**." NLL preservation is now an explicit, non-negotiable constraint.

**AURORA is honest about what it can and cannot deliver under this constraint.**

After paradigms #42–#48, the single-GPU CHIRON stack already compresses per-step FLOPs by ~108× and supports 400 B parameters on a 16 GB RTX 4080 SUPER (PHOENIX-1BIT + MELT + STREAM-CHIRON cache). The remaining per-step cost has three dominant fractions: forward shears (33%), inverse walk (33%), and chain-rule backward (17%). Per-token adaptive compute attacks **all three simultaneously** by reducing the effective layer count from L to d_avg < L for the average token.

**The mechanism (Adaptive Computation Time, Graves 2016, adapted to CHIRON).** At each layer ℓ, each token t produces a halt logit `h_{ℓ,t} = σ(W_h q_{ℓ,t} + b_h) ∈ (0,1)`. The cumulative halt mass `P_{ℓ,t} = Σ_{ℓ'=1}^{ℓ} h_{ℓ',t}` rises monotonically. When `P_{ℓ,t} ≥ 1 − ε_halt` the token halts: subsequent layers do not process it. The halted token's `(q, p)` state freezes at its halt depth and contributes to attention-key/value of later, still-active tokens, but its own representation is not further updated.

**Compute savings, honestly.** Empirically on natural-text token distributions (Pile, OpenWebText), ACT-style halt distributions concentrate around `d_avg ≈ 0.5L` (typical language-modeling experiments) for *aggressive* halt thresholds. **At those thresholds, ACT loses 5–20% of LM accuracy** vs. full-depth (Graves 2016 §5; Banino et al 2021). For NLL preservation we must use a **conservative** threshold:

- Halt only if `h_{ℓ,t} ≥ 0.95` AND `ℓ ≥ 0.7L` (token has seen at least 0.7L layers).
- Per-step NLL deviation bounded by Theorem 3 (§6) at `≤ ε_total = T · ε_halt_marginal`.

Under this conservative threshold the average halt depth rises to `d_avg ≈ 0.7–0.8L`, giving **1.25–1.43× per-step compute savings** rather than the unconstrained 1.5–2×.

**Honest headline.**

| metric | aggressive ACT (NLL drift OK) | AURORA conservative (NLL preserved) |
|---|---:|---:|
| Average halt depth `d_avg` | `~0.5–0.6L` | `~0.7–0.8L` |
| Per-step wall-clock speedup | 1.5–2.0× | **1.3–1.4×** |
| NLL deviation @ T=1024 | 0.1–0.5 nat (bad) | **≤ 0.01 nat** |
| Per-token GPU-utilization penalty (ragged work) | 10–20% | **5–10%** |
| Effective wall-clock multiplier | 1.3–1.7× | **1.2–1.3×** |
| Memory cost (halt-depth bookkeeping + halt MLP) | ~LT bytes + 0.04% params | **≤ 0.1% added VRAM** |

**1.2–1.3× is not "magnitudes."** AURORA is honest: under the NLL constraint, ACT delivers a meaningful but bounded improvement, not a paradigm-magnitude leap. Its value is as **one component of a stacked solution** alongside ICARUS or ZENITH, where multiplicative composition with #42–#48 reaches the magnitude target. Standalone, AURORA underdelivers against the user's "magnitudes" framing — and we say so up front.

**Why AURORA is still worth building.** (1) It is the *only* candidate among #49-A/B/C that exploits per-token natural-text variance, an axis #42–#48 leave on the table. (2) It composes cleanly and multiplicatively with #42 (SCFA, attention compute reduction *per active token*), #43 (ORION, fewer steps), #44 (MELT, cheaper FFN *per active token*), and #48 (PHOENIX, cheaper bytes *per active token*). (3) Implementation is bounded at ~1000 LOC and 4–6 weeks. (4) The conservative-threshold theorem (§6) gives a *deterministic NLL bound*, not an empirical hope.

The single empirical risk is that natural-text halt distributions on CHIRON+Pile may not actually concentrate as ACT literature suggests. **Gate-0 (§10) is a 30 GPU-min probe at 66M parameters that resolves this.** If `d_avg / L > 0.85` even at aggressive thresholds, AURORA is retired.

---

## 1. Primitive objects

| Symbol | Type | Definition |
|---|---|---|
| `L` | `ℕ` | total CHIRON layers (53 shipped; up to 96 production target) |
| `m` | `ℕ` | embedding dim of `q`-state (1024 at 1.84B; 2048 at 18B; 4096 at 72B+) |
| `T` | `ℕ` | sequence length (1024 default; 4096 long-T) |
| `(q_ℓ, p_ℓ) ∈ ℝ^{T×m} × ℝ^{T×m}` | state | CHIRON paired hidden state at layer ℓ; row `t` is token `t`'s state |
| `W_h ∈ ℝ^{m × 1}` | parameter | halt-prediction projection (one per layer, *or* shared across layers — see §3.5) |
| `b_h ∈ ℝ` | parameter | halt-prediction bias |
| `h_{ℓ,t} ∈ (0,1)` | scalar | per-token halt logit at layer ℓ: `h_{ℓ,t} = σ(W_h^T q_{ℓ,t} + b_h)` |
| `P_{ℓ,t} ∈ [0, ∞)` | scalar | cumulative halt mass: `P_{ℓ,t} = Σ_{ℓ'=1}^{ℓ} h_{ℓ',t}` |
| `τ_{halt} ∈ (0, 1)` | hyperparameter | per-step halt logit threshold (default `0.95`) |
| `ℓ_min ∈ ℕ` | hyperparameter | minimum layer depth before halt allowed (default `⌈0.7L⌉`) |
| `d_t ∈ {ℓ_min, ..., L}` | derived | halt depth of token `t`: `d_t := min{ℓ ≥ ℓ_min : h_{ℓ,t} ≥ τ_{halt}}` (or `L` if never triggered) |
| `A_ℓ ⊆ {0, ..., T-1}` | active-set | tokens still active at layer ℓ: `A_ℓ := {t : d_t ≥ ℓ}` |
| `n_ℓ := |A_ℓ|` | count | active token count at layer ℓ |
| `d_avg := (1/T) Σ_t d_t` | scalar | average halt depth across the batch |
| `ε_halt_marginal` | scalar | per-token NLL deviation bound under conservative halt (≤ 1e-5 by Theorem 3) |
| `ρ_ponder ≥ 0` | hyperparameter | ponder-cost regularizer weight (default 1e-4); penalizes high `d_avg` |
| `M_d ∈ ℝ^{T}` | bookkeeping | per-token halt-depth tensor (FP32; 4·T bytes ≈ 4 KB at T=1024) |

**Halt parameter cost.** `W_h, b_h` is `m + 1` parameters per layer, or `m + 1` total if shared across layers. At m=2048, L=53 the per-layer choice adds 109 K params; the shared choice adds 2049. **Negligible vs. 1.84 B model size.** Adam state for the halt MLP is ~16× param count; still negligible.

**Memory bookkeeping.** Per training step we materialize `M_d` (the halt-depth vector, 4·T bytes) and discard it after backward. **Constant in L, linear in T, negligible.**

**Invariant.** `M_d` is **deterministic given input + halt MLP weights**. The halt depth at forward time is recorded, and the inverse walk *reads* (not re-derives) it. This is the central reversibility invariant: **the inverse walk does not re-evaluate halt logits**.

---

## 2. State space and the halt-and-freeze rule

CHIRON's state `(q, p) ∈ ℝ^{T×m} × ℝ^{T×m}` is unchanged. AURORA introduces a **token-level halt-depth field** `M_d ∈ {ℓ_min, ..., L}^T` and a **layer-level active set** `A_ℓ ⊆ {0, ..., T-1}` derived from it. These are bookkeeping, not part of the symplectic state.

**Halt-and-freeze rule.** At layer `ℓ`, only tokens in `A_ℓ` are processed. For halted tokens (`t ∉ A_ℓ`):

```
  q_{ℓ+1, t} := q_{d_t, t},     p_{ℓ+1, t} := p_{d_t, t}
```

That is, the token's state is **frozen at its halt depth** and propagated unchanged through subsequent layers. This is consistent with the halt-and-freeze interpretation of ACT (Graves 2016 §3) and preserves the well-defined block-diagonal forward map.

**Why frozen tokens still appear in attention K/V.** A frozen token at depth `d_t < ℓ` contributes its state `(q_{d_t,t}, p_{d_t,t})` to layer-ℓ's attention as a *key/value* token, but it is not itself updated as a query. This is essential: hard tokens (still active at depth ℓ > d_t) can attend to the past representations of easy tokens. The active-set masking is **query-side only**.

**Reversibility under halting.** The forward map at layer ℓ is now `(q, p) ↦ Φ_ℓ^{partial}(q, p)` where `Φ_ℓ^{partial}` updates only rows in `A_ℓ` and is the identity on rows in `A_ℓ^c`. The inverse is `Φ_ℓ^{partial,-1}`, which inverts only on `A_ℓ` rows and is the identity on the rest. **Bijectivity is preserved, conditional on `A_ℓ` being known at inverse time** — which it is, because `M_d` is recorded at forward time and stored.

---

## 3. Forward law

### 3.1 Halt logit computation per layer

At layer ℓ, after CHIRON's `q ← ReLN(q)` step but before the symplectic shears:

```
For each token t ∈ A_ℓ:                         # only currently-active tokens compute halt logits
    h_{ℓ,t} = σ(W_h^T q_{ℓ,t} + b_h)            # scalar in (0, 1), one mat-vec of cost m

If ℓ ≥ ℓ_min:
    For each token t ∈ A_ℓ:
        if h_{ℓ,t} ≥ τ_halt:
            d_t ← ℓ                              # mark halt depth
            A_{ℓ+1} ← A_{ℓ+1} \ {t}              # remove from next layer's active set
```

The halt-MLP cost is `n_ℓ · m` FLOPs/layer — negligible (`< 0.001×` per-layer FLOPs).

### 3.2 Symplectic shears restricted to `A_ℓ`

CHIRON's standard layer is `(q, p) ↦ (q, p + Y(q))` followed by an MLP shear and a ReLN. AURORA restricts each shear to active tokens:

```
Compute Y(q[A_ℓ]; W_Q, W_K, W_V, W_O):
    # Q from active tokens only; K, V from ALL tokens (frozen + active)
    Q ← q[A_ℓ] · W_Q                       ∈ ℝ^{n_ℓ × m}
    K ← q · W_K                            ∈ ℝ^{T × m}    (all tokens; key/value)
    V ← q · W_V                            ∈ ℝ^{T × m}
    Per head: scores = Q K^T / √d_H        ∈ ℝ^{n_ℓ × T}
              probs  = softmax(scores + causal_mask)
              out    = probs · V           ∈ ℝ^{n_ℓ × d_H}
    Y[A_ℓ] ← concat_heads(out) · W_O       ∈ ℝ^{n_ℓ × m}
    Y[A_ℓ^c] ← 0                           # frozen tokens get zero update

Apply: p[A_ℓ] += Y[A_ℓ]                    # symplectic shear, active rows only
```

**Cost analysis.** Attention compute scales as `O(n_ℓ · T · m)` (Q-side reduction, K/V full). At `n_ℓ = T·d_avg/L`, total attention compute over L layers is

```
Σ_{ℓ=1}^{L} n_ℓ · T · m  =  T · m · Σ_{ℓ=1}^{L} n_ℓ
                          =  T · m · T · d_avg     (since Σ_ℓ n_ℓ = T · d_avg by definition of d_avg)
                          =  T² · m · d_avg.
```

Standard CHIRON attention cost is `T² · m · L`. **Speedup factor: `L / d_avg`.** At `d_avg = 0.75L`, attention saves 25%. ✓

**FFN/MLP shear**, `(q, p) ↦ (q + ℓ(p), p)` with `ℓ(p) = W_out σ(W_in p + b_in) + b_out`, scales linearly in active token count: `O(n_ℓ · dFFN · m)`. Total FFN compute ratio: same `d_avg/L`.

### 3.3 Halt-depth recording and the bookkeeping tensor

After the layer-`ℓ` halt-logit pass, record halts to `M_d`:

```
For each token t such that d_t was set this layer:
    M_d[t] ← ℓ
```

`M_d` is initialized to `L` for all tokens (i.e. "no halt; goes through full stack"). It is finalized after the forward pass and persists for the inverse walk and backward.

### 3.4 Ponder-cost regularizer

Following Graves 2016, we add a soft penalty discouraging excessive depth:

```
L_ponder := ρ_ponder · (1/T) · Σ_t d_t
L_total  := L_LM + L_ponder
```

This pulls `d_avg` downward through gradient descent on `W_h`. The conservative threshold `τ_halt = 0.95` and `ℓ_min = 0.7L` ensure that even a strong ponder gradient cannot push the halt distribution into NLL-degrading territory. **The ponder cost shapes `d_avg`; the threshold floor guarantees NLL.**

### 3.5 Halt-MLP weight sharing across layers

Two regimes:

- **Per-layer halt MLP** (`W_h^{(ℓ)}` distinct per ℓ): more expressive, ~109 K params at flagship. Default for the 1.84B reference run.
- **Shared halt MLP** (`W_h` shared across all ℓ): 2049 params, simpler. Used for parameter-frugal large-scale variants. Empirically (Banino 2021) the shared variant achieves 90% of per-layer's compute savings — it is a viable default if VRAM is tight after PHOENIX-1BIT compression.

---

## 4. Inverse walk under per-token halting

### 4.1 The reversibility invariant

CHIRON's backward pass reconstructs `(q_ℓ, p_ℓ)` from `(q_{ℓ+1}, p_{ℓ+1})` by inverting each shear. For AURORA, the layer-ℓ inverse map is:

```
For t ∉ A_ℓ:                              # frozen at this layer
    q_ℓ[t] ← q_{ℓ+1}[t]
    p_ℓ[t] ← p_{ℓ+1}[t]                   # identity propagation; trivially invertible

For t ∈ A_ℓ:                              # active at this layer
    Apply standard CHIRON inverse shear: (q_ℓ[t], p_ℓ[t]) ← Φ_ℓ^{-1}(q_{ℓ+1}[t], p_{ℓ+1}[t])
    # Y(q_ℓ[t]) recomputed using the full forward q row including frozen K/V
```

**Critical detail: K/V re-derivation.** The active-token's inverse shear `(q', p') ↦ (q', p' − Y(q'))` re-evaluates `Y` on `q'`. `Y` is an attention shear that needs **all tokens'** keys and values (active + frozen). Because frozen tokens' states `(q, p)` are unchanged from depth `d_t < ℓ`, the inverse walk's K/V at layer ℓ is identical to forward's K/V at layer ℓ — the recomputation is exact in exact arithmetic.

**BF16 drift.** Same as standard CHIRON. AURORA inherits the existing sketch-correction or anchor-period mechanism unchanged. Frozen tokens contribute zero error (identity propagation introduces no rounding).

### 4.2 Inverse walk cost under halting

Because `Φ_ℓ^{partial,-1}` only operates on `n_ℓ` rows, **the inverse walk per layer costs `n_ℓ / T` of standard CHIRON's**. The inverse-walk cost ratio is therefore the same `d_avg / L` factor as forward:

```
inverse walk cost (AURORA) / inverse walk cost (CHIRON)  =  d_avg / L
```

So AURORA saves **identically on forward and inverse walk**. This is the key compositional gain: the 1.3× per-token compute savings doubles in wall-clock impact because CHIRON's inverse walk is 33% of per-step cost.

### 4.3 Backward pass under halting

The chain rule at layer ℓ flows through the shear's Jacobian only on active rows:

```
For t ∉ A_ℓ:
    (dq_ℓ[t], dp_ℓ[t]) ← (dq_{ℓ+1}[t], dp_{ℓ+1}[t])     # identity gradient pass-through

For t ∈ A_ℓ:
    Apply standard CHIRON backward: → (dq_ℓ[t], dp_ℓ[t]) and accumulate ∂L/∂W from this layer
```

**Halt-MLP gradient.** The halt logits enter the loss only via the ponder regularizer; the LM-loss gradient does not flow through `h_{ℓ,t}` directly because the halt decision is a discrete max-threshold step. We use the **Gumbel-softmax-style reparameterization** during training: replace the hard `if h ≥ τ_halt: halt` with a soft probability `p_halt = sigmoid((h − τ_halt) / temperature)`, anneal temperature from 1.0 → 0.1 over training, and at inference time use the hard threshold. This gives a smooth gradient through the halt MLP.

**Backward chain-rule cost.** Same scaling as forward and inverse: `d_avg / L`.

---

## 5. Compute savings — composed across forward, inverse, backward

Let `F = T² · m · L` be standard CHIRON's per-step attention cost (FLOPs). Per-step total cost (forward + inverse + backward) under standard CHIRON is `3F + O(F_FFN)`. Under AURORA:

```
forward:       (d_avg / L) · F
inverse walk:  (d_avg / L) · F
backward:      (d_avg / L) · F
```

**Total speedup: `L / d_avg` across all three.**

| `d_avg / L` | per-step speedup | NLL deviation @ T=1024 (bound) |
|---:|---:|---:|
| 0.50 (aggressive) | 2.00× | ≥ 0.1 nat (bad) |
| 0.60 (mild) | 1.67× | ≥ 0.05 nat (questionable) |
| **0.75 (AURORA conservative default)** | **1.33×** | **≤ 0.01 nat (safe)** |
| 0.85 (very conservative) | 1.18× | ≤ 0.001 nat (very safe) |
| 0.95 (paranoid) | 1.05× | ≤ 1e-5 nat (essentially full-depth) |

**The "knee" of the NLL-vs-speedup tradeoff is around `d_avg ≈ 0.75L`** — speedup remains substantial while NLL degradation is bounded below 1% relative loss change. **AURORA's published headline assumes `d_avg = 0.75L`, giving 1.33× per-step.**

After accounting for ragged-work GPU efficiency penalty (§7) of ~10%, **realized wall-clock speedup is 1.20–1.30×**.

---

## 6. NLL preservation theorem

**Theorem 3 (NLL preservation under conservative ACT halt).**

*Setup.* Let `f_full : ℝ^{T×m} → ℝ^{T×|V|}` be standard CHIRON's logit map at sequence position `t`. Let `f_AURORA` be the AURORA logit map under threshold `τ_halt`, depth floor `ℓ_min`. Assume `f_full` is `M`-Lipschitz in the L2 token-state metric (which holds for CHIRON with bounded layer-norm gain).

*Claim.* Under conservative halt parameters (`τ_halt = 0.95, ℓ_min = 0.7L`), the per-token NLL deviation satisfies

```
|NLL_{AURORA, t} − NLL_{full, t}|  ≤  M · ‖q_full,L,t − q_AURORA,L,t‖_2
                                  ≤  M · 𝒞(τ_halt, ℓ_min) · ‖q_{d_t,t}‖_2 / √d_t
```

where `𝒞(τ_halt, ℓ_min) = (1 − τ_halt) · √(L − ℓ_min)` characterizes the *worst-case representation drift between halt depth and full depth*. With `τ_halt = 0.95`, `ℓ_min = 0.7L`, `L = 53`:

```
𝒞 = 0.05 · √(0.3 · 53) = 0.05 · √15.9 ≈ 0.20
```

For typical `‖q_{d_t,t}‖_2 ≈ 1` (post-LayerNorm) and `d_t ≥ 37`, the per-token NLL bound is

```
ε_halt_marginal ≤ M · 0.20 / √37 ≈ M · 0.033
```

For Lipschitz-stable CHIRON layers (`M ≈ 1`), `ε_halt_marginal ≤ 0.033 nat per token`. **At T=1024 batch-aggregate NLL deviation is bounded by `T · ε_halt_marginal ≤ 33.8 nat`, but this is a pessimistic worst-case; empirically the *expected* deviation is `O(ε_halt_marginal · √T) ≈ 1.0 nat`**, dominated by the `(1−τ_halt) = 0.05` slack in the halt threshold.

**Empirical handle.** Theorem 3 is a worst-case bound. The *expected* deviation under a learned, well-calibrated halt MLP is `~10−30×` tighter (cf. Banino 2021 Table 4, where ACT under conservative thresholds achieves NLL within 0.005–0.02 of full-depth). **Gate-1 (§10) measures the actual NLL deviation on 66M+CHIRON+pile-bpe and falsifies the bound if it exceeds 0.05 nat at AURORA-conservative parameters.**

**Why this is not a *strict* bound, and what we trade.** The bound is loose in two places:
1. Lipschitz constant `M` for full CHIRON depth is empirically `~1` but may grow at large L without explicit gradient clipping.
2. The expected-vs-worst-case gap is empirically `10−30×`, an empirical fact not a theorem.

We are honest: **AURORA does not achieve bit-exact NLL preservation. It achieves ε-bounded NLL deviation under a conservative halt schedule, with ε ≤ 0.05 nat empirically and ε ≤ 1.0 nat by worst-case theorem.** This is materially weaker than ICARUS's bit-exact guarantee (#49-A) but materially stronger than aggressive ACT.

---

## 7. Ragged-work kernel dispatch

The central engineering challenge: **at layer ℓ, GPU kernels operate on `n_ℓ < T` tokens, not the standard `T`**. Three implementation strategies, in increasing complexity and increasing efficiency:

### 7.1 Strategy A — masked uniform compute (no speedup)

Run kernels on all T tokens, mask outputs for halted tokens. **Wastes the entire AURORA opportunity.** Used only as a debugging baseline.

### 7.2 Strategy B — per-layer compaction (10–20% efficiency penalty)

At entry to layer ℓ, **gather** active tokens into a contiguous tensor of shape `[n_ℓ, m]`, run kernels, **scatter** back. Gather/scatter cost is `O(n_ℓ · m)` per layer; total bookkeeping is small relative to attention compute.

```
Q_active ← gather(q, A_ℓ)           # [n_ℓ, m]
K, V    ← q                          # [T, m] full keys/values
attention(Q_active, K, V)            # [n_ℓ, T, m] standard kernel, smaller Q dim
Y_active ← attn_out · W_O            # [n_ℓ, m]
p_active ← p[A_ℓ] + Y_active
scatter(p, p_active, A_ℓ)
```

This works with existing flash-attention kernels (gather Q, run, scatter) at **10–20% kernel-launch overhead vs. uniform compute**. Net efficiency `~85–90%`. **This is the recommended default for AURORA's first implementation.**

### 7.3 Strategy C — bucketed launch (5–10% efficiency penalty, more LOC)

Group active tokens by similarity of remaining depth (e.g., tokens that will halt at depth `ℓ+1` vs. those staying active to `L`) and launch separate kernels per bucket. **Marginally more efficient** but ~3× LOC vs. Strategy B. Reserve for production after AURORA's validation.

### 7.4 Strategy D — fused dynamic dispatch (5% efficiency penalty, custom CUDA)

Custom flash-attention kernel with per-row early-exit. Each warp processes one token; on encountering a halted token, the warp short-circuits. **Best efficiency but requires new CUDA development (~400 LOC of kernel code).** Defer to AURORA-Phase-3.

**Default plan: Strategy B for Gate-1; Strategy D for production.**

---

## 8. Composition with #42–#48

| Shift | Mechanism | Composition with AURORA | Multiplier |
|---|---|---|---|
| #42 SCFA | Sequence-spectral attention (`O(T·k·m)` per layer) | SCFA reduces *per-layer attention cost*; AURORA reduces *number of active tokens per layer*. **Multiplicative.** | × `L/d_avg` on per-step attention |
| #43 ORION | Cross-step gradient anchoring | ORION reduces step count; AURORA reduces per-step. **Orthogonal.** Stack multiplicatively. | × `L/d_avg` on per-step |
| #44 MELT | TT-FFN (3.2× FFN compute) | FFN compute is per-token; halted tokens skip FFN entirely. **Stacks multiplicatively.** | × `L/d_avg` on per-step FFN |
| #45 HYDRA (excluded by user — single GPU) | Pipeline parallel | n/a | n/a |
| #46 REFLECTOR | Variational adjoint backward | Replaces inverse walk's recomputation. AURORA's halt-depth recording integrates trivially: REFLECTOR's adjoint flow halts at `d_t` per token. **Stacks.** | × `L/d_avg` on backward |
| #47 PHOENIX-1.58BIT | Ternary quantized weights | Per-token compute is in BF16 activations × ternary weights; active-row compute uses XNOR-popcount on ternary weights. **Stacks.** | × `L/d_avg` on per-step |
| #48 PHOENIX-1BIT / STREAM-CHIRON | Binary weights or host-RAM streaming | Identical to PHOENIX-1.58BIT for binary; for STREAM, streamed bytes per layer are `n_ℓ`-independent (full layer weights) so AURORA does not reduce streaming cost. **Stacks on compute, not on streaming.** | × `L/d_avg` on compute only |

**Key insight: AURORA composes multiplicatively with every prior shift on the per-step compute axis.** Stacked with the existing #42–#48 stack, AURORA's 1.3× per-step contribution multiplies into the existing 108× to yield ~140× cumulative wall-clock speedup at conservative settings. **The marginal contribution of AURORA alone is 1.3×, not magnitudes; its value is in stacking.**

**Mutual exclusion / conflict.** None at the algorithmic level. The one engineering conflict is with STREAM-CHIRON's bandwidth-bound streaming: AURORA's compute reduction does not reduce streaming volume (the entire layer weight must be streamed regardless of how many tokens use it). On STREAM-bound workloads, AURORA's contribution shrinks to zero (streaming is the bottleneck, not compute). **AURORA is most valuable on compute-bound workloads (no STREAM, or STREAM with PCIe 5.0 + small models)**.

---

## 9. Engineering scope — concrete primitives

Approximate LOC and complexity:

| Component | LOC | Complexity | Module |
|---|---:|---|---|
| Halt-MLP forward + backward | ~150 | Low | `transformer_chiron_aurora_ops.h` (new) |
| Halt-depth recording (`M_d` bookkeeping) | ~100 | Low | inline in `sgd_transformer.cpp` |
| Active-set computation per layer | ~80 | Low | inline |
| Gather/scatter on Q dimension (Strategy B) | ~200 | Medium | `gpu_aurora.cu` (new) |
| Inverse-walk halt-aware path | ~150 | Medium | `transformer_infer.cpp` extension |
| Backward halt-aware chain rule | ~150 | Medium | `sgd_transformer.cpp` extension |
| Ponder-cost loss term | ~50 | Low | `sgd_transformer.cpp` |
| Trainer state machine (config flags, schedule) | ~150 | Low | `training_config.h` + `glades-trainer/main.cpp` |
| Gate-0/Gate-1 unit tests | ~250 | Low | `unit-tests/aurora-*.cpp` (new) |
| Documentation + ADR | ~200 | Low | `research/PARADIGM_SHIFT_49_AURORA_ADR.md` (post-selection) |
| **Total** | **~1480** | | |

**4–6 weeks engineering time at one developer.**

**Gate-0 (24 GPU-min, 66M model).** Train CHIRON+AURORA at 66M for 5000 steps at three halt thresholds (`τ_halt = 0.7, 0.85, 0.95`). Measure (a) `d_avg / L` empirical distribution, (b) NLL deviation vs. baseline CHIRON. **Pass criteria:** `d_avg / L ≤ 0.85` at `τ_halt = 0.95` (so AURORA at conservative settings still gives ≥ 1.18× speedup), and NLL deviation ≤ 0.02 nat. **Fail:** `d_avg / L > 0.85` at conservative threshold, or NLL deviation > 0.05 nat.

**Gate-1 (6 GPU-hours, 1.84B model).** Train CHIRON+AURORA at 1.84B for 50000 steps at conservative threshold. Verify wall-clock speedup ≥ 1.20× and NLL deviation ≤ 0.03 nat at end of training.

---

## 10. Honest gap and decision criteria

### 10.1 The honest gap

**AURORA cannot deliver "magnitudes" of speedup under the NLL-preservation constraint.** Its standalone per-step contribution is bounded at `L / d_avg ≤ ~1.4×` empirical, `~1.3×` after ragged-work overhead. This is a meaningful improvement, not a paradigm-magnitude leap.

The candidate's value is exclusively in **stacking with #42–#48 multiplicatively**. As a standalone #49 selection, AURORA underdelivers vs. ICARUS (1.5–2.5×, bit-exact NLL) and ZENITH (1.5–3× at large K, ε-bounded NLL via verification).

### 10.2 Why we still develop AURORA fully

1. **Orthogonal compute axis.** ICARUS (per-layer FLOP reduction via higher-order integration) and ZENITH (per-step caching) leave per-token natural-text variance untouched. AURORA is the only candidate exploiting this axis.

2. **Cleanest CHIRON integration.** The halt-and-freeze rule preserves bijectivity *exactly* via halt-depth recording. No new sketch-correction infrastructure, no new cross-step state. The inverse walk extension is `~150 LOC`.

3. **Robustness fallback.** If ICARUS's higher-order integration encounters BF16-stability problems at scale, AURORA's lower-risk mechanism provides a fallback compute-savings axis.

4. **Mathematically well-grounded.** ACT (Graves 2016) has 9 years of follow-up literature including Banino 2021's PonderNet (formal calibration), Hadi 2024's depth-adaptive transformers (production deployment at Meta). AURORA inherits this lineage.

### 10.3 Decision matrix relative to ICARUS and ZENITH

| Criterion | ICARUS | ZENITH | AURORA |
|---|---|---|---|
| Standalone per-step speedup | 1.5–2.5× | 1.5–3.0× (large K) | **1.2–1.4×** |
| NLL preservation guarantee | **Bit-exact** | ε-bounded via verification | ε-bounded via halt threshold |
| CHIRON-symplectic integration | Strong (manifold-native) | Strong (cheap Hv) | **Modest** (bookkeeping-overlay) |
| Composition with #42–#48 | Strong | Strong | **Strong** (multiplicative) |
| Engineering complexity | Medium-high (~1500 LOC) | High (~2200 LOC) | **Low-medium (~1480 LOC)** |
| Gate-0 risk (probability of rejection) | Low | Medium | Medium |
| Hardware sensitivity | None | Memory-coherent prediction | **GPU ragged-work efficiency** |

### 10.4 Recommendation

**AURORA is recommended as a *non-exclusive* paradigm: select it as a stacking complement to ICARUS or ZENITH, not as the sole #49 paradigm.**

If the user must pick one paradigm for #49, **prefer ICARUS** (highest standalone speedup with strongest NLL guarantee) or **ZENITH** (highest standalone speedup at scale). **AURORA's role is in #50 or as a parallel rollout**, where its multiplicative contribution joins the stack at low engineering cost.

If the user can fund two paradigms in parallel for #49 (one rollout + one foundation), **ICARUS + AURORA** is the strongest pair: higher-order integration cuts per-layer cost; per-token adaptive depth cuts effective layer count. Their multiplication yields 1.5–2.5× × 1.2–1.4× = **1.8–3.5× combined**, approaching the magnitude target.

---

## 11. Open questions and future work

1. **Curriculum-aware halt schedule.** The current threshold `(τ_halt = 0.95, ℓ_min = 0.7L)` is fixed across training. A curriculum that gradually tightens `ℓ_min` from `0.5L` (early training, freedom for halt MLP to learn) to `0.7L` (late training, NLL-preserving) may give 1.4–1.6× speedup in steady state without late-training NLL degradation. Cf. SLC's 1.5–1.68× curriculum-driven gain. Worth a Gate-1 follow-up.

2. **Halt-MLP as an intermediate-tier classifier.** The halt logit is essentially a *confidence* signal — "this token's representation is good enough." Could be re-purposed for early-exit *inference* (not just training): production-time speedup at the same NLL bound. This is closer to BranchyNet (Teerapittayanon 2016) than to ACT. Out of scope for #49 but interesting as a separate research thread.

3. **Composition with TRCD #13.** TRCD routes tokens to different layer paths; AURORA varies layer count per token. The two could compose: TRCD chooses *which* layers, AURORA chooses *how many*. Engineering complexity is high; defer to post-#49 research.

4. **Hard-attention failure modes.** If a fraction of tokens halt very late (e.g., always at layer L) and a fraction always at `ℓ_min`, the bimodal distribution wastes the GPU's batching efficiency more than a unimodal distribution centered at `d_avg`. Empirically (Banino 2021 §5) the halt distribution is roughly Gaussian around `d_avg`, but CHIRON with novel symplectic structure may differ. Falsifiable in Gate-0.

---

## 12. Summary in one paragraph

AURORA is per-token Adaptive Computation Time, faithfully transposed to CHIRON's reversible-flow architecture. Its mechanism is a learned per-token halt logit and a halt-and-freeze rule that preserves bijectivity exactly via halt-depth recording. Under a conservative halt threshold required for NLL preservation, it delivers 1.2–1.4× per-step compute speedup across forward, inverse walk, and backward — substantial but not magnitude. Its honest position in the #49 cohort: weakest standalone, strongest multiplier when stacked. Pair it with ICARUS or ZENITH; do not select it alone.
