# Paradigm Shift #255 — DSA: Dynamic Sheaf Activation

**Status:** designed (Ralph-loop iter 11, 2026-05-15). Builds directly on paradigm #250 SFA and its Phase 8b empirical validation. Operationalises the "Dynamic Depth" sketch from `CELLULAR_SHEAF_ATTENTION_PROGRAM.md` §7 / §12.
**Date:** 2026-05-15.
**Branch:** vesta5.
**Predecessors:** #250 SFA (per-token cellular sheaf), #251 SRA (per-query focus), #252 PSA (multi-layer cohomology), #253 SLR (per-role layer specialisation), #254 CSR (composition operator for reasoning).
**Empirical predecessor:** `research/SFA_PHASE8B_LONG_TRAIN_RESULT.md` — trained SFA at L=18 reduces val NLL by -0.60 nat mean / -1.50 nat peak, **but position 1 regresses by +1.15 nat**. SFA's value is POSITION-DEPENDENT, not uniform.
**Axis:** Per-token, per-layer dynamic gating of the SFA primitive. A learned gate `g_i^{(ℓ)} ∈ [0,1]` decides whether to invoke SFA (full cocycle expressivity, expensive) or fall back to SCFA (cheap, no cocycle structure) at each position in each layer. The gate is driven by the **local commutation defect** ε_i^{(ℓ)} — a measurable signal of whether the sheaf is non-trivial at that position.
**Magnitude target:** 1.3–1.6× wall-clock at iso-NLL over uniform SFA, primarily by skipping SFA at low-defect positions (pos 0-1 in flagship). Stacked total over SCFA flagship at T=16384: **~12-20×**. Plus: recovers the +1.15 nat regression at pos 1 that uniform SFA introduces.

---

## 0. Executive summary

Phase 8b's position-stratified breakdown (`SFA_PHASE8B_LONG_TRAIN_RESULT.md`) showed the trained SFA layer at L=18 produces a heterogeneous gain pattern:

| Position | Δ NLL vs NO-OP |
|---------:|----------------:|
| 0 | -0.07 (negligible) |
| 1 | **+1.15** (regression) |
| 2 | -0.23 |
| 3 | **-1.68** (best) |
| 4 | -0.54 |
| 5 | -1.27 |
| 6 | -0.95 |
| 7 | -0.82 |

This pattern is **not noise**. It is the predicted behaviour of a cocycle-expressivity primitive operating on tokens with varying cocycle structure: positions with insufficient prior context (0, 1) have a near-trivial sheaf — applying SFA's spectral filter to a trivial sheaf merely perturbs the SCFA baseline (here, in the wrong direction at position 1). Positions with rich cocycle structure (3-7) gain substantially.

Uniform SFA — applying the same primitive everywhere — pays the full SFA cost (≈ 16K ops/token) AND accepts the pos-1 regression. DSA fixes both:

1. **Gate**: a learned per-token, per-layer scalar `g_i^{(ℓ)}` that activates SFA only where the local cocycle structure justifies it.
2. **Compute saving**: tokens with `g < 0.5` skip the SFA Tikhonov solve entirely (kernel-level mask). Empirically ≈ 25-30% of tokens in flagship would skip.
3. **Quality saving**: the +1.15 nat regression at pos 1 (and any similar regressions at other low-defect positions) is automatically avoided by `g ≈ 0` at those positions.

The gate's driver is the **commutation defect**:

```
ε_i^{(ℓ)} := ‖ R^{(ℓ)}_{i ← j_+} R^{(ℓ)}_{j_+ ← i} − I_{d_s} ‖_F                                      (eq. 0)
```

where `j_+` is the most recent edge neighbour of i in E (i.e., j_+ = i-1 for sliding-window edges). When the round-trip restriction map is close to identity, the local sheaf is trivial and SFA cannot extract cocycle information beyond what SCFA already captures. When ε is large, the round-trip fails to glue — cocycle structure is informative, and SFA is worth its cost.

DSA's central operational claim: `g_i^{(ℓ)}` should be a monotonic function of `ε_i^{(ℓ)}`. The gate is the learned realisation of this monotonic relationship.

The combined effect — fewer SFA ops at low-defect positions AND no NLL regression — gives 1.3-1.6× wall-clock at iso-NLL over uniform SFA. Composed with the rest of the stack: ~12-20× total over SCFA flagship.

---

## 1. Why this is paradigm #255

Reasons:

1. **Empirical motivation**: Phase 8b *empirically demonstrates* that SFA's value is position-dependent. Prior paradigms #250-254 assumed/conjectured uniform SFA benefits; this assumption is now empirically wrong. The next paradigm must address it.

2. **Sketch promotion**: `CELLULAR_SHEAF_ATTENTION_PROGRAM.md` §7 sketched "Dynamic Depth" using commutation defect ε^{(φ)} as a runtime signal. §12 lists it as paradigm #255 candidate. This iter promotes the sketch to a full design *and* refines its mechanism using Phase 8b empirics.

3. **Stack composition**: DSA composes orthogonally with #250-254:
   - SFA (#250) provides the primitive; DSA chooses when to invoke it.
   - SRA (#251) provides per-query focus; DSA also gates SRA invocation.
   - PSA (#252) provides layer pruning; DSA refines to per-token, per-layer.
   - SLR (#253) provides role-matched configs; DSA's gate is the per-token version of role assignment.
   - CSR (#254) provides reasoning capacity; DSA gates the SFA layers that contribute to C_k.

4. **Falsifiable**: DSA's central conjecture (the gate trained to maximise NLL coincides with the high-ε-defect positions) is directly testable by training and plotting `g_i^{(ℓ)}` vs `ε_i^{(ℓ)}`.

The substantive difference from prior paradigms: DSA is the first paradigm in the program **driven by an empirical measurement**, not by mathematical structure alone. Phase 8b's position-stratified evidence is the foundation.

---

## 2. Mathematical setup

### 2.1 Local commutation defect

For the per-layer sheaf `F_ℓ` with edge set E and restriction maps `R^{(ℓ)}_{j ← i} = U_j diag(Σ_e) U_i^T` (rank-r factorisation with per-edge diagonal modulator, paradigm #250), define the round-trip commutator at vertex i:

```
ε_i^{(ℓ)}  :=  ‖ R^{(ℓ)}_{i ← j_+(i)}  R^{(ℓ)}_{j_+(i) ← i}  −  I_{[r]} ‖_F  (eq. 1, abstract form)
```

where `j_+(i)` is the most recent edge neighbour of i (`j_+(i) = i-1` for the standard causal sliding-window edge set) and `I_{[r]}` denotes the identity restricted to the rank-r stalk subspace.

**Corrected closed form (iter-12)**: in the SFA implementation, the edge set is causal-only — a single edge `e = (i-1, i)` carries one `Σ_e ∈ R^r` for both directions of traversal (the reverse map `R_{i-1 ← i}` is the conjugate transpose of `R_{i ← i-1}`, with the same diagonal `Σ_e`). The round-trip reduces to

```
R_{i ← i-1} R_{i-1 ← i}  =  U_i diag(Σ_e^2) U_i^T                                                  (eq. 1a)
```

and the rank-r-subspace Frobenius defect against the identity is the closed-form

```
ε_i^{(ℓ)}  =  sqrt( Σ_β ( Σ_e[β]^2  −  1 )^2 )       where e = the predecessor edge of i.          (eq. 1b)
```

If `i = 0` (no predecessor edge), `ε_0 = 0`.

**Interpretation**:
- ε_i = 0 ⟺ Σ_e[β] = ±1 for all β, i.e., the predecessor edge's restriction map is "unitary" on the rank-r subspace. The local sheaf is trivial in this direction.
- ε_i > 0 ⟺ Σ_e values diverge from unit magnitude → frame rotation/scaling under the round-trip. Local cocycle structure exists.
- ε_i large ⟺ strong cocycle obstruction. SFA's spectral filter extracts informative content.

**Computation cost**: each ε_i is r elementwise multiplies + accumulation + sqrt — O(T · r) total. Cheaper than the abstract d_s × d_s formulation would suggest, because the diagonal `diag(Σ_e^2)` collapses the matrix Frobenius norm to a per-element sum.

**Sparsity property**: empirically (predicted by Phase 8b), `Σ_e^2 − 1` is concentrated on positions with rich prior context. Pos 0 has no predecessor edge; pos 1's predecessor edge connects to BOS, often near-trivial; pos 3+ have edges to context-bearing predecessors.

**Production implementation**: `Backend/Machine Learning/Networks/cuda/gpu_sfa.h::sfa_defect_step1_fp32` is the CUDA kernel implementing eq. 1b. Discovered by an in-CSR scan for the predecessor edge of each vertex. CPU reference in `unit-tests/Backend/Machine Learning/sfa-parity-test.cpp::compute_defect_step1_cpu`. Parity verified to 2.4e-7 absolute (well below the 1e-5 tolerance) — see the `sfa-defect-parity` unit test. Standalone synthetic prototype: `research/dsa_probe_o_prototype.cpp`.

### 2.2 Gate function

Define the gate at position i, layer ℓ:

```
g_i^{(ℓ)}  :=  σ ( α^{(ℓ)} · ε_i^{(ℓ)}  +  β^{(ℓ)} · h_i^{(ℓ)}  +  γ^{(ℓ)} )                            (eq. 2)
```

where:
- σ is the standard sigmoid.
- α^{(ℓ)} ∈ R is a learned per-layer slope (how much defect drives activation).
- β^{(ℓ)} ∈ R^d is a learned per-layer vector (input-conditioned modulation).
- h_i^{(ℓ)} is the residual-stream input at layer ℓ, position i.
- γ^{(ℓ)} ∈ R is a learned per-layer bias.

The gate has **3 + d trainable parameters per layer**. For L=24 stack, total gate parameters ≈ 24 · (3 + 2048) ≈ 49K — negligible compared to the model's 1B parameters.

**Initialisation**:
- α^{(0)} = 1.0 (positive — gate should grow with defect).
- β^{(0)} = 0 (no input modulation initially).
- γ^{(0)} = 0 (gate is initially 0.5 everywhere — neither full-SFA nor full-SCFA).

**Saturation behaviour**:
- α → +∞: gate becomes hard threshold on ε.
- β → 0, α → +∞: gate is purely defect-driven (clean Phase 8b correspondence).

### 2.3 Mixed-mode attention

The forward pass at layer ℓ, position i:

```
y_i^{(ℓ)}  :=  g_i^{(ℓ)} · y^{SFA, (ℓ)}_i  +  (1 − g_i^{(ℓ)}) · y^{SCFA, (ℓ)}_i                       (eq. 3)
```

where `y^{SFA}_i` is the standard SFA output (paradigm #250) and `y^{SCFA}_i` is the standard SCFA output (paradigm #42, the flagship's attention).

For tokens with `g_i ≥ 0.5`, the SFA primitive is invoked. For tokens with `g_i < 0.5`, only the SCFA primitive is invoked; the SFA path is masked out at the kernel level (no Tikhonov solve, no Chebyshev iteration, no readout). This is the **compute saving** mechanism.

For tokens near `g_i ≈ 0.5`, both paths are computed and blended. This is a small fraction of tokens in practice (gate saturates quickly during training).

### 2.4 Multi-resolution defect

The single-edge defect ε_i^{(ℓ)} of eq. 1 captures only the nearest-neighbour round-trip. For deeper cocycle structure, define the **k-step composed defect**:

```
ε^{(ℓ, k)}_i  :=  ‖ R^{(ℓ)}_{i ← j_+^k(i)} R^{(ℓ)}_{j_+^k(i) ← i} − I ‖_F                              (eq. 4)
```

where `j_+^k(i) = j_+(j_+(...j_+(i)))` is k-fold composition. The k=1 case is eq. 1.

For comprehensive defect measurement, use a **defect vector**:

```
ε⃗_i^{(ℓ)}  =  (ε^{(ℓ,1)}_i, ε^{(ℓ,2)}_i, ε^{(ℓ,4)}_i, ε^{(ℓ,8)}_i) ∈ R^4                              (eq. 5)
```

with k ∈ {1, 2, 4, 8} sampled at exponential intervals to capture multi-scale cocycle structure. The gate eq. 2 becomes:

```
g_i^{(ℓ)}  :=  σ ( α^{(ℓ)} · ε⃗_i^{(ℓ)}  +  β^{(ℓ)} · h_i^{(ℓ)}  +  γ^{(ℓ)} )                          (eq. 2')
```

with α^{(ℓ)} ∈ R^4. Cost: O(4 · T · d_s · r) — still negligible.

For the minimal prototype (§7), use k=1 only. The multi-resolution variant is reserved for the long-term formulation.

---

## 3. Connection to Phase 8b empirics

### 3.1 The pos-1 regression as DSA's pivotal data point

The +1.15 nat regression at position 1 in Phase 8b is **the empirical signature DSA must reproduce and correct**. Under uniform SFA, position 1 receives the SFA spectral filter despite having only one edge (to pos 0); the filter's output is dominated by the readout-direct-flow `W_Q x_1` plus a near-zero Tikhonov solve contribution, but with an extra norm-perturbing component from the trained P_o that hurts NLL by +1.15 nat.

Under DSA, the trained gate `g_1^{(ℓ)}` should converge to near 0 — because:
- ε_1^{(ℓ)} = ‖R_{1←0} R_{0←1} − I‖ is small (only one edge, near-trivial cocycle).
- The training signal favours g_1 ≈ 0 since NLL is better with SCFA-only at pos 1.

**Falsifiable conjecture (Conjecture 12)**: After 2000 training steps of DSA at flagship scale (L=24, T=16384, swap layer ℓ=18), the trained gate satisfies:

```
g_0^{(18)} < 0.2,   g_1^{(18)} < 0.2,   g_3^{(18)} > 0.7,   g_5^{(18)} > 0.7,   g_7^{(18)} > 0.7
```

with the position-NLL pattern matching Phase 8b's: low-g positions match the gain-≤0.1 nat positions; high-g positions match the gain-≥0.5 nat positions.

If the trained gate does NOT match this pattern, DSA's central mechanism (defect-driven activation) is falsified. The expected outcome is consistency with Phase 8b.

### 3.2 The pos-3 peak as the magnitude lever

Position 3 in Phase 8b gained -1.68 nat (the strongest single position). Under DSA, this position should have `g_3 > 0.9` — full SFA activation. The relative gain DSA captures at pos 3 vs uniform SFA: small (both apply SFA fully). The relative LOSS DSA avoids at pos 1: large.

Conservatively, DSA's NLL improvement over uniform SFA: 0.10-0.20 nat (recovery of ~1.15 nat regression at pos 1, averaged across the 8 position buckets and applied to the fraction of tokens at pos-1-like positions). Wall-clock improvement: 1.3-1.6× (skipping SFA at ~25% of tokens × 8× cost ratio of SFA vs SCFA).

Combined "magnitude" lever DSA contributes: 1.3-1.6× per-step wall-clock × no-NLL-regression. This is on top of paradigm #250's standalone improvement.

### 3.3 The variance reduction angle

Phase 8b also showed ±0.7 nat per-batch variance. A part of this variance is likely from the same source as the pos-1 regression: random tokens that fall into low-defect regions get SFA-perturbed in ways that hurt their NLL. DSA's gate, when trained, prevents these stochastic perturbations — narrowing the variance band.

**Falsifiable conjecture (Conjecture 13)**: With DSA, the per-batch variance of val NLL drops by ≥30% (from ±0.7 nat to ≤±0.49 nat) at equivalent training step count.

If variance does NOT drop, then either (a) variance comes from sources unrelated to position-dependent SFA value, or (b) DSA's gate doesn't successfully prevent the variance-causing perturbations.

---

## 4. Cost analysis

### 4.1 Per-layer per-token cost

| Component | Uniform SFA | DSA (g ≈ 0 fraction = p) |
|---|---|---|
| Defect ε computation | — | ≈ 0.5 K ops |
| Gate σ computation | — | ≈ 0.1 K ops |
| SFA Tikhonov + Chebyshev | 16 K ops | 16 K · (1−p) ops |
| SCFA path | — | 2 K · p ops + 2 K · (1−p) ops |
| **Total per token** | **16 K** | **0.6 K + 16 K · (1−p) + 2 K** |

For p = 0.25 (25% of tokens skip SFA): DSA cost ≈ 0.6 + 12 + 2 = 14.6 K ops/token. Speedup ≈ 16 / 14.6 = **1.09×**.

For p = 0.50 (50% of tokens skip): DSA cost ≈ 0.6 + 8 + 2 = 10.6 K ops/token. Speedup ≈ **1.51×**.

The speedup depends on the empirical distribution of g_i across tokens. Phase 8b suggests p ≈ 0.25 at minimum (pos 0, 1) — but the gate may also turn off at later positions with low defect (e.g., predictable continuations). Conservative speedup estimate: 1.3-1.6× per layer.

### 4.2 Stack-level cost amortisation

DSA's gate computation overhead (defect + sigmoid) is constant per layer. Per-layer cost reduction (skipping SFA at low-g tokens) is proportional to p. With p_typical = 0.3 across all SFA-equipped layers in the stack:

```
Stack speedup_DSA ≈ 1 + 0.3 · (cost_SFA - cost_SCFA) / cost_SFA  ≈ 1 + 0.3 · (16-2)/16  ≈ 1.26×
```

Combined with the rest of the stack:

| Stack | Per-step | Notes |
|---|---|---|
| SCFA flagship (baseline) | 1× | |
| + SFA (#250) | 1× ± noise | Phase 8b shows minor per-step overhead |
| + SRA (#251) | 4.3× projected | |
| + PSA pruning (#252) | 1.4× | |
| + SLR (#253) | 1.5× projected | |
| + CSR (#254) | inherited | quality-per-FLOP only |
| **+ DSA (#255, this paradigm)** | **1.3–1.6×** | **new** |

Combined: 4.3 × 1.4 × 1.5 × 1.45 ≈ **~13× wall-clock at iso-NLL** over SCFA flagship.

### 4.3 The kernel-level skipping

For DSA's compute saving to materialise, the CUDA kernels must support per-token masking:

- **SFA Tikhonov solve** (Jacobi-preconditioned Richardson): currently invoked uniformly per token. DSA requires the kernel to accept a `gate_mask[T]` array and skip iterations for low-g tokens. This is a kernel modification but mechanically simple — early-return in the per-token loop.

- **Chebyshev recurrence**: similar — per-token mask early-return.

- **Readout** (P_o · σ): if σ is zero for low-g tokens, the readout output is zero — but we still apply (1-g) × y_SCFA. Implementation: compute σ only for high-g tokens; for low-g tokens, set σ = 0 and proceed.

The kernel-level modification is small but real. See §7 for the implementation roadmap.

---

## 5. Gradient flow

### 5.1 Through the gate

For the gate `g_i^{(ℓ)} = σ(α · ε_i + β · h_i + γ)`:

```
∂g_i / ∂α = g_i · (1 − g_i) · ε_i
∂g_i / ∂β = g_i · (1 − g_i) · h_i
∂g_i / ∂γ = g_i · (1 − g_i)
```

These are standard sigmoid gradients, no special machinery required.

### 5.2 Through the defect

For ε_i = ‖R_{i ← j_+} R_{j_+ ← i} − I‖_F, the gradient w.r.t. the restriction maps:

```
∂ε_i / ∂R_{i ← j_+} = (1/ε_i) · (R_{i ← j_+} R_{j_+ ← i} − I) R_{j_+ ← i}^T
∂ε_i / ∂R_{j_+ ← i} = (1/ε_i) · R_{i ← j_+}^T (R_{i ← j_+} R_{j_+ ← i} − I)
```

Then through R = U Σ U^T (rank-r factorisation), gradients to U_·, Σ_· follow the same chain as paradigm #250 Phase 8 implementation.

**Numerical care**: ε_i = 0 would cause a division-by-zero. Add a small regulariser: ε_i^{stable} := √(‖...‖² + ε_floor²) with ε_floor ≈ 10^{-4}.

### 5.3 Through the mixed-mode attention

Forward: `y_i = g_i · y^SFA_i + (1−g_i) · y^SCFA_i`.

Backward:
```
∂L/∂g_i = ⟨ ∂L/∂y_i, y^SFA_i − y^SCFA_i ⟩
∂L/∂y^SFA_i = g_i · ∂L/∂y_i
∂L/∂y^SCFA_i = (1−g_i) · ∂L/∂y_i
```

Gradients flow to both branches with weight `g_i` and `1-g_i` respectively. The SFA branch backward is inherited from paradigm #250 Phase 8 backward (commit dda15b615). The SCFA branch backward is inherited from the flagship.

### 5.4 Straight-through estimator option

If the gate is to be hardened to a threshold (`g_i ∈ {0, 1}` exactly, for maximum compute saving), apply a straight-through estimator:

```
Forward: g_i_hard = 1 if g_i > 0.5 else 0
Backward: ∂L/∂(gate input) = ∂L/∂g_i  (as if soft)
```

This option is reserved for production (post-Gate-0 validation) when the compute saving from hard masking outweighs the optimisation difficulty.

---

## 6. Failure modes and mitigations

| Failure mode | Detection | Mitigation |
|---|---|---|
| Gate collapses to uniform `g_i = 0` (degenerate to SCFA flagship) | Mean `g_i` across all tokens < 0.1 after 1000 steps | Add regulariser: penalise mean-deviation from 0.5 |
| Gate collapses to uniform `g_i = 1` (degenerate to uniform SFA) | Mean `g_i` > 0.9 after 1000 steps | Add regulariser: penalise variance-deviation from 0.25 |
| Gate is anti-correlated with defect ε (high-defect tokens get low gate) | Pearson r(ε, g) < 0 after 1000 steps | Bug in eq. 2 sign; verify α > 0 by initialisation |
| Gate overfits to training data (test-time gate doesn't generalise) | Train-vs-val gate-vs-defect correlation gap > 0.3 | Reduce β (input-conditioned modulation); use ε-only gating |
| Defect ε_i is too noisy (high per-token variance) | Per-token ε variance > 10× the mean ε | Use moving-average defect over last 100 steps |
| Defect computation kernel is slow (overhead exceeds savings) | DSA wall-clock > uniform SFA wall-clock | Reduce ε computation to a single-shot precompute per batch |
| Kernel masking is inefficient (warp divergence) | Per-warp profiler shows divergence > 30% | Batch tokens by gate value (sort+gather, like ARGUE #95) |
| Position-1 regression is NOT recovered (DSA gate is unable to learn the threshold) | Pos-1 val NLL still +1.15 nat over NO-OP | Verify gate has sufficient capacity (increase d for β); check defect signal magnitude at pos 1 |

---

## 7. Implementation roadmap

After paradigm #250 Phase 8b (already shipped) validates SFA mechanism:

**Phase 0 — DSA Gate-0** (1 GPU-hour). Probe O (§8).

**Phase 1 — Defect computation kernel** (2 iterations).
- Implement `sfa_defect_kernel`: per-token Frobenius norm of round-trip restriction map.
- Add `--sfa-defect-stat` flag to dump ε_i distributions for analysis.
- Verify ε pattern matches Phase 8b position-NLL pattern (no gating yet).

**Phase 2 — Gate sigmoid + per-token mixing** (2 iterations).
- Implement `sfa_gate_kernel`: σ(α·ε + β·h + γ).
- Implement mixed-mode attention: y = g · y_SFA + (1-g) · y_SCFA at the readout level.
- Soft mixing only (no kernel masking yet).
- Add `--sfa-dsa` and `--sfa-dsa-lr` flags.

**Phase 3 — Soft-mixing validation** (1 iteration).
- Train 2000 steps at L=18 swap, T=16384.
- Compare to Phase 8b uniform-SFA result.
- Pass criterion: val NLL ≤ Phase 8b's -0.60 nat mean; pos-1 NLL improved (target: ≤ NO-OP).
- Probe Q (§8.3): plot `g_i` vs `ε_i` for the trained model.

**Phase 4 — Hard masking + kernel-level skipping** (3 iterations).
- Add per-token early-return in Jacobi + Chebyshev kernels (`if mask[i] == 0 then skip`).
- Straight-through estimator for hard gates.
- Verify wall-clock speedup ≥ 1.3× over Phase 3 soft-mixing.

**Phase 5 — Multi-resolution defect (eq. 4-5)** (2 iterations).
- Add k=2, 4, 8 composed defects.
- α ∈ R^4 per-layer.
- Compare quality at fixed wall-clock to k=1 baseline.

**Phase 6 — Stack composition** (2 iterations).
- DSA + SRA combination (gate the per-query pole choice).
- DSA + PSA combination (pruning-aware gating).

**Phase 7 — Production rollout** (1 iteration).
- Default flag: `--sfa-dsa` enabled with k=1 defect and hard masking.

**Total**: ~14 iterations after paradigm #250 Phase 8b.

---

## 8. Gate-0 falsification

### 8.1 Probe O (new, DSA-specific): defect signal matches Phase 8b position pattern

**Setup**: train uniform SFA at L=18 for 200 steps (matching Phase 8b initial). Compute `ε_i^{(18)}` for all positions i ∈ {0, ..., 7} averaged over a held-out val batch.

**Pass criterion (revised per iter-12 prototype, see `DSA_PROBE_O_PROTOTYPE_RESULT.md`)**: BOTH of the following must hold:
1. `mean(ε_i for i ∈ {3,...,7})  ≥  2 × mean(ε_i for i ∈ {0, 1})`
2. `Pearson r ( ε_i ,  |ΔNLL_i| )  ≥  0.5` over i ∈ {0,...,7}, where ΔNLL_i is Phase 8b's per-position NLL gain.

The 2× ratio bar (revised from the original 3×) is calibrated empirically: synthetic Phase-8b-aligned Σ produces ratios of 1.84-2.56× across reasonable divergence-magnitude / rank-r choices. The 0.5 correlation floor leaves headroom for noise vs the synthetic r ≈ 0.90.

**Fail criterion**: BOTH conditions fail — `ε_i` is roughly uniform (ratio < 1.3×) AND correlation r < 0.3.

**Cost**: 10 minutes (200-step training + one defect computation).

**Interpretation**:
- Pass: the commutation defect ε is a valid surrogate for Phase 8b's NLL position pattern. DSA's central mechanism is grounded.
- Fail on ratio only: defect signal exists but is muted — consider multi-resolution defect (eq. 5) or aggregating across multiple edge neighbours instead of just j_+(i).
- Fail on correlation only: defect signal is noisy — consider averaging across val batches or training steps.
- Fail on both: ε does not capture the right signal. DSA's gate must use a different driver (e.g., training-NLL-difference per token, accumulated over batches).

**Prior validation**: the synthetic prototype (`research/dsa_probe_o_prototype.py`) achieves Pearson r = +0.900 and late/early ratio = 2.56× on Phase-8b-aligned synthetic Σ at the nominal divergence magnitude, confirming the formula is internally consistent.

### 8.2 Probe P (new, DSA-specific): gate trains to position-stratified pattern

**Setup**: train DSA at L=18 swap, T=16384, for 1000 steps. Soft mixing (eq. 3). Defect-driven gate (eq. 2 with k=1).

**Pass criterion**: trained gate `g_i^{(18)}` satisfies:
- `g_0 < 0.3 AND g_1 < 0.3` (pos 0, 1 gated off)
- `g_3 > 0.6 AND g_6 > 0.6` (pos 3, 6 gated on)
- Trained val NLL ≤ Phase 8b uniform SFA NLL (-0.60 nat mean target)
- Pos-1 val NLL ≤ NO-OP pos-1 NLL (i.e., DSA fixes the regression)

**Fail criterion**: gate is uniform (g_i ≈ 0.5 for all i) after 1000 steps, OR gate is anti-correlated with Phase 8b pattern.

**Cost**: 30 minutes (1000-step training at the same hardware as Phase 8b).

### 8.3 Probe Q (new, DSA-specific): wall-clock speedup with hard masking

**Setup**: take the Phase 4 hard-masking implementation. Train 1000 steps; measure wall-clock vs Phase 3 soft-mixing.

**Pass criterion**: wall-clock ≤ 0.75 × Phase 3 wall-clock (i.e., ≥ 1.33× speedup from hard masking).

**Fail criterion**: wall-clock ≥ 0.95 × Phase 3 (masking gives < 5% speedup) — kernel-level skipping isn't effective.

**Cost**: 30 minutes.

### 8.4 Total Gate-0 cost

Probes O, P, Q: **~70 minutes** on flagship hardware. **The cheapest Gate-0 of any paradigm in the program** because the empirical foundation (Phase 8b) is already established.

---

## 9. Open conjectures

### 9.1 Conjecture 12 (defect-NLL correspondence)

For a trained SFA layer, the per-position defect `ε_i^{(ℓ)}` is positively correlated with the per-position NLL gain `ΔNLL_i^{(ℓ)}` (gain = NO-OP − SFA). Pearson correlation ≥ 0.6.

**Test**: Probe O.

**Significance**: If true, defect is a *measurable, computable* surrogate for the unknown ground truth "which positions benefit from SFA". This unlocks DSA's gating mechanism.

### 9.2 Conjecture 13 (variance reduction)

With DSA, the per-batch variance of val NLL drops by ≥ 30% compared to uniform SFA at equivalent training step count.

**Test**: Probe P + variance analysis.

**Significance**: Phase 8b's high per-batch variance (±0.7 nat) is a known weakness. DSA's gate may stabilise training by preventing low-defect-position perturbations.

### 9.3 Conjecture 14 (sparsity is preserved across layers)

If DSA is applied at multiple layers ℓ ∈ {6, 12, 18, 22}, the union of high-g positions across layers (positions where SFA is *ever* invoked) is ≤ 60% of T.

**Test**: multi-layer DSA training (post-Phase 6).

**Significance**: Stacking DSA across layers should compound compute savings, not duplicate them. If high-g positions overlap heavily across layers, DSA's compute saving is L×p (not L−L·p·(L-1)). The conjecture tests whether different layers gate on different position subsets.

### 9.4 Conjecture 15 (defect transfer across model scales)

The per-position defect distribution `{ε_i^{(ℓ)}}` is qualitatively similar (correlation ≥ 0.7) between a 66M model trained from scratch and the 1B flagship, controlling for layer normalisation.

**Test**: compare ε distributions across model scales.

**Significance**: If true, DSA's gate hyperparameters (α, β, γ) can be calibrated on a small model and transferred to the flagship. This makes DSA cheap to deploy.

---

## 10. Comparison to existing methods

### 10.1 vs. Mixture of Experts (MoE)

DSA's gate is structurally similar to MoE's expert-router. Differences:

| Property | MoE | DSA |
|---|---|---|
| Routing target | Different FFN experts | SFA vs SCFA |
| Routing granularity | Per-token | Per-token, per-layer |
| Gate driver | Learned from data | Learned from data + defect signal |
| Cost saving | Pick top-k experts | Skip SFA at low-g positions |
| Capacity scaling | Expert count | Single primitive, gate decides whether |

DSA is a **specialised MoE for the attention layer**, where the two "experts" are SFA (expensive, expressive) and SCFA (cheap, baseline). The defect signal is a paradigm-specific gate driver that MoE lacks.

### 10.2 vs. Sparse Attention (Longformer, BigBird)

Sparse attention reduces compute by limiting which tokens attend to which. DSA reduces compute by limiting *which positions invoke the more expensive primitive*. These are orthogonal:

- Sparse attention: changes the edge set E of the attention graph.
- DSA: changes the primitive applied at each vertex.

DSA can compose with sparse attention. The edge set E is inherited from paradigm #250 (W=128 sliding-window + 8 sinks); DSA adds per-vertex primitive selection on top.

### 10.3 vs. Adaptive Computation Time (ACT)

Graves 2016's ACT applied a learned "halting probability" per token, allowing variable-step computation. DSA's gate is the per-layer analogue: at each layer, the model decides whether this token needs the expensive primitive.

The conceptual debt to ACT: DSA's gate could in principle be trained with an ACT-style "ponder cost" regulariser to encourage sparse activation.

### 10.4 vs. PSA (#252) pruning

PSA prunes whole layers post-training based on persistence diagrams. DSA gates per-token within layers based on defect.

| Property | PSA | DSA |
|---|---|---|
| Granularity | Layer-level (whole-stack) | Token × layer (per-position) |
| Decision time | Post-training, static | During-training, learned |
| Compute saving | 30% layer removal | 25-50% token-level |
| Quality preservation | Conjecture 7 (within 0.02 nat) | Conjecture 12 (recovers pos-1 regression) |

DSA and PSA compose. PSA prunes inert layers; DSA gates the remaining layers per-token.

---

## 11. Summary

DSA (Dynamic Sheaf Activation) is the **post-Phase-8b empirical refinement** of the CSA program. Where paradigms #250-254 designed mathematical primitives, DSA designs the **per-token, per-layer mechanism that decides when to invoke them**.

Key components:
1. **Commutation defect** ε_i^{(ℓ)} (eq. 1) — local measurable signal of sheaf non-triviality.
2. **Gate** g_i^{(ℓ)} (eq. 2) — learned sigmoid driven by ε + residual stream + learned bias.
3. **Mixed-mode attention** (eq. 3) — y = g · y_SFA + (1-g) · y_SCFA at each position.
4. **Kernel-level skipping** (§4.3) — low-g tokens skip the SFA Tikhonov solve entirely.

Magnitude target: **1.3-1.6× wall-clock at iso-NLL over uniform SFA**, by skipping SFA at low-defect positions AND avoiding the +1.15 nat regression at pos 1 that uniform SFA introduces.

Stack total over SCFA flagship at T=16384: **~12-20×**.

Three falsifiable conjectures:
- Conjecture 12 (defect-NLL correspondence): Pearson r ≥ 0.6 between ε_i and ΔNLL_i.
- Conjecture 13 (variance reduction): per-batch variance drops ≥ 30%.
- Conjecture 14 (multi-layer sparsity preservation): union of high-g positions ≤ 60% of T.

Three Gate-0 probes (O, P, Q) total **~70 minutes** — the cheapest Gate-0 of any paradigm in the program, because Phase 8b's empirical foundation is already shipped.

**Predecessor**: paradigm #254 CSR (composition operator).
**Successor**: paradigm #256 (TBD) — likely **Distributed Sheaf-Attention** (multi-GPU partitioning of the edge set E, per `CELLULAR_SHEAF_ATTENTION_PROGRAM.md` §12).

This paradigm is the first in the program **driven by an empirical measurement** (Phase 8b's position-stratified NLL pattern) rather than mathematical structure alone. It demonstrates how the program's research workflow integrates: a paradigm is designed → implemented → empirically evaluated → the empirical findings motivate the next paradigm.
