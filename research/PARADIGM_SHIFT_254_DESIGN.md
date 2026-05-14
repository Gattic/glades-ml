# Paradigm Shift #254 — CSR: Compositional Sheaf Reasoning

**Status:** designed (Ralph-loop iter 7, 2026-05-14). Extends paradigms #250-253 to address multi-step reasoning capability.
**Date:** 2026-05-14.
**Branch:** vesta5.
**Predecessors:** #250 SFA (cellular sheaf substrate), #251 SRA (per-query focus), #252 PSA (multi-layer cohomology), #253 SLR (role-matched configuration).
**Axis:** Make **multi-step reasoning capability** mathematically explicit. Iterated composition of restriction maps `R_{i_k ← i_{k-1}} · ... · R_{i_1 ← i_0}` along paths through the token graph yields the **composition operator C_k** — its spectrum characterises the model's k-step reasoning capacity. CSR also operationalises this for measurement, training-time regularisation, and reasoning-task-specific architecture choices.
**Magnitude target:** Not a raw-compute paradigm. CSR addresses **reasoning capability per FLOP** — provides a principled mechanism to improve multi-step reasoning tasks (math, multi-hop QA, code) without scaling compute. Conjectured: +0.10–0.30 nat NLL improvement on reasoning-rich benchmarks at iso-FLOP.

---

## 0. Executive summary

Modern LLMs perform multi-step reasoning empirically (chain-of-thought, multi-hop QA, math). Yet the *mathematical structure* of reasoning in transformer attention is not explicit. Layer depth correlates with reasoning length (Chowdhery et al. 2022), but this is folklore — there is no formal connection between attention's mathematical form and reasoning capacity.

The CSA program (paradigms #250-#253) provides a substrate where reasoning has a *natural* mathematical formulation. Each layer's restriction map `R^{(ℓ)}_{j ← i} : F(v_i) → F(v_j)` encodes "how token j's perspective sees what token i's perspective produces." A *k-step* reasoning chain is a path `i_0 → i_1 → ... → i_k` through the token graph, with the composed restriction map:

```
R^{path}_{i_0 → i_k}  :=  R^{(ℓ_k)}_{i_k ← i_{k-1}}  ·  R^{(ℓ_{k-1})}_{i_{k-1} ← i_{k-2}}  ·  ...  ·  R^{(ℓ_1)}_{i_1 ← i_0}        (eq. 1)
```

This composition is the mathematical primitive of *transitive inference* in the sheaf framework. If `R^{(ℓ_1)}` encodes "Alice knows Bob" and `R^{(ℓ_2)}` encodes "Bob trusts Charlie", the composition `R^{(ℓ_2)} · R^{(ℓ_1)}` encodes "Alice → (knows-Bob, trusts-Charlie) → Charlie." With enough depth, the model represents arbitrary-length reasoning chains.

CSR's central object is the **composition operator** at depth k:

```
C_k  :=  E_{paths} [ R^{path}_{i_0 → i_k} ]  =  Σ_{paths of length k} weight(path) · R^{path}                (eq. 2)
```

C_k is a linear operator on R^{d_s} that summarises the "k-step reasoning capacity" of the trained model. Its spectrum determines:

- **Strong eigenvalues** (|λ_p(C_k)| ≈ 1): well-trained k-step reasoning chains.
- **Small eigenvalues** (|λ_p(C_k)| << 1): weak k-step chains — the model cannot reliably perform k-step reasoning.
- **Eigenvalue decay rate vs k**: characterises the *maximum reliable reasoning depth* of the model.

CSR provides three operational deliverables:

1. **Reasoning-capacity metric** (§4): compute `‖C_k‖_op` and `cond(C_k)` as functions of k. The max-k where these are well-behaved is the model's *reasoning ceiling*.

2. **Reasoning-task auxiliary loss** (§5): add a regulariser encouraging well-conditioned C_k at task-relevant depths. Improves k-step task performance without scaling parameters.

3. **Depth-targeting layer placement** (§6): if the target task needs k=5 reasoning, allocate layers to optimise C_5 specifically (not just the persistence diagram).

The magnitude claim is about **capability**, not raw compute. CSR is conjectured to improve LLM performance on multi-step reasoning benchmarks (GSM8K, MATH, HellaSwag, multi-hop QA) by 0.10–0.30 nat NLL improvement at iso-FLOP, with the improvement *concentrated* on reasoning-heavy samples.

---

## 1. Why this is paradigm #254 — and why it differs from #253 SLR

Paradigms #250-253 each addressed an *architectural* axis:
- #250 SFA: per-token perspective (substrate)
- #251 SRA: per-query focus (configuration)
- #252 PSA: multi-layer structure (depth)
- #253 SLR: per-role specialisation (configuration of configurations)

These are *width-x-depth* paradigms — they configure the attention layer and its placement, but the *capability* axis (what the model can do) is still emergent from training.

CSR is the first **capability paradigm** in the program. It identifies a specific computational task (multi-step reasoning) and provides:
- A mathematical structure (C_k composition operator)
- A measurement procedure (spectrum analysis)
- A training-time regulariser (auxiliary loss on C_k conditioning)
- A design knob (depth-targeted layer placement)

This is conceptually different from #250-253. SLR optimised wall-clock per token; CSR optimises *what the model can accomplish per token*.

The two are **orthogonal and compose multiplicatively**: SLR + CSR = role-matched configurations PLUS capability-targeted training.

---

## 2. Mathematical setup

### 2.1 Path-composition algebra

The token graph G = (V, E) with V = {1,...,T} has paths of length k from i_0 to i_k:

```
P_k(i_0, i_k)  :=  { (i_0, i_1, ..., i_k) :  (i_{j-1} → i_j) ∈ E  for j = 1, ..., k }       (eq. 3)
```

For each path π ∈ P_k(i_0, i_k), define:

```
R^π  :=  R^{(ℓ_k)}_{i_k ← i_{k-1}}  ·  R^{(ℓ_{k-1})}_{i_{k-1} ← i_{k-2}}  ·  ...  ·  R^{(ℓ_1)}_{i_1 ← i_0}       (eq. 4)
```

where `R^{(ℓ_j)}` is the restriction map at layer ℓ_j. We use a *layer assignment* ℓ_j ∈ {1,...,L} for each path step, allowing the same edge to be traversed using different layers' restriction maps. The total number of (path, layer-assignment) pairs grows as `|E|^k · L^k`.

### 2.2 Path-weighted composition operator

A simple all-paths sum is too expensive. Instead, we use **softmax-weighted path averaging**:

```
C_k(i_0, i_k)  :=  Σ_{π ∈ P_k(i_0, i_k)}  Σ_{layer-assignment}  w(π, ℓ_1, ..., ℓ_k) · R^π        (eq. 5)
```

with weights `w(π, ℓ_1, ..., ℓ_k) = softmax over paths and layer assignments` — i.e., a stochastic process over the set of (path, layer-assignment) pairs.

For computational tractability, two simplifications:

**Simplification A**: assume the layer assignment is *monotone* — `ℓ_1 ≤ ℓ_2 ≤ ... ≤ ℓ_k` — i.e., reasoning proceeds *deeper* through the stack as path length grows. This matches the empirical observation that deeper layers handle higher-order reasoning.

**Simplification B**: restrict to *forward-time* paths in E — `i_0 ≤ i_1 ≤ ... ≤ i_k` (no looping back in the causal sequence). This is enforced naturally by causal masking.

With both simplifications, eq. 5 becomes a tractable T × T × L^k sparse computation.

### 2.3 The composition operator C_k

Aggregate over all (i_0, i_k) pairs at fixed k:

```
C_k  :=  (1 / Z_k)  ·  Σ_{i_0, i_k}  Σ_{path π, layer ℓ}  R^π  ⊗ |i_0⟩⟨i_k|        (eq. 6)
```

where `Z_k` is the normalising constant (total path count or expected mass), and `|i⟩⟨j|` is the rank-1 operator on R^T.

C_k is an operator on `R^{T · d_s} → R^{T · d_s}` (rank-T sheaf direct sum). Its spectrum λ_1(C_k) ≥ λ_2(C_k) ≥ ... determines the k-step reasoning capacity.

### 2.4 The reasoning-capacity metric

Define:

```
ρ_k  :=  λ_1(C_k) / σ_max(C_k^T C_k)^{1/2}  =  cosine of dominant left and right singular vectors    (eq. 7)
```

`ρ_k` measures how aligned the dominant eigenvector of C_k is with its singular structure — i.e., how "consistent" the k-step compositions are.

A model with strong k-step reasoning will have `ρ_k ≈ 1` (clean spectral structure). A model that has merely *parroted* k-step patterns will have `ρ_k << 1` (noisy spectrum, no coherent reasoning pathway).

The **reasoning ceiling** of the model is:

```
k★(model)  :=  max  k  such that  ρ_k > 0.7                                                          (eq. 8)
```

This is a *measurable* quantity from a trained model (computed by power iteration on C_k in ~M matvecs per k).

---

## 3. Connection to existing paradigms

### 3.1 PSA's persistence diagram is the k → ∞ limit

C_k's spectrum as k → ∞ converges to the spectrum of the **infinite-step composition operator**, which is identified (via standard ergodic theory) with the spectrum of the **stationary measure** on the path space. This stationary measure is the *colimit sheaf F_∞* of PSA (Conjecture 3, PARADIGM_SHIFT_250_PROOFS.md §7).

Equivalently: PSA's PD^0 long bars (length ≥ L/2) correspond to dominant eigenvectors of C_{L/2}. The longer the bar, the more stable the corresponding reasoning chain across depth.

This connection gives PSA + CSR a unified mathematical structure: PSA measures *persistent* compositional patterns; CSR measures *finite-depth* compositional patterns at each k.

### 3.2 SLR's roles correspond to C_k contributions

PSA's layer-role taxonomy (feature-extractor / synthesizer / consensus-builder) maps to CSR contributions:

- **Feature-extractor layers** contribute the *initial* restriction maps `R^{(ℓ_1)}` — small `ℓ_1` in eq. 4. They populate the input to all k-step reasoning chains.
- **Synthesizer layers** contribute the *intermediate* compositions — mid-stack `ℓ_j`'s. They build up partial reasoning chains.
- **Consensus-builder layers** contribute the *final* compositions — large `ℓ_k`'s. They terminate reasoning chains with a coherent output.

This refines #253 SLR's "role" picture into a more granular "reasoning-step-position" picture. The two are compatible: SLR's role assignment determines the *configuration* of each layer; CSR's reasoning-step-position determines the *path-composition contribution*.

### 3.3 SRA's pole structure encodes step-relations

At each layer, SRA's per-query pole `z_q^{(ℓ)}` selects spectral content. In CSR, this is identifiable with selecting *which kind of relation* the layer applies to its input:

- Low Im(z_q) (sharp pole): the layer focuses on a single relation type (e.g., identity, "is-a").
- High Im(z_q) (broad pole): the layer applies many relations at once.

Multi-pole SRA (P > 1) means a single layer can apply *multiple* relation types simultaneously — captured in CSR as multiple paths sharing layer-assignment but having different effective restriction maps.

---

## 4. Operationalisation — measuring C_k

### 4.1 Spectrum estimation via stochastic Lanczos

Materialising C_k is infeasible (T · d_s ≈ 16384 · 64 = 10^6 dimension). Instead, use **stochastic Lanczos**:

```
For random initial vector v ∈ R^{T·d_s}:
    For j = 1, ..., m_Lanczos:
        w_j := apply C_k to current vector v_{j-1}
                # = sum over k-step paths and layer-assignments of R^π v_{j-1}
                # Computed via successive matvecs: u_0 := v_{j-1}; u_l := L_{F_{ℓ_l}} u_{l-1}; final = u_k
        v_j := (w_j - <v_j, v_{j-1}> v_{j-1} - <v_j, v_{j-2}> v_{j-2}) / norm
    Build tridiagonal matrix T_k ∈ R^{m_Lanczos × m_Lanczos}
    Diagonalise T_k → eigenvalue estimates of C_k.
```

Cost per Lanczos step: k matvecs of L_F (already computed in SFA forward) × L layer-assignments. For k = 8, L = 24, m_Lanczos = 32: total ≈ 32 · 8 · 24 = 6K matvecs.

At ~16K ops per matvec (default SFA config): ~100 M ops total per C_k spectrum estimate. **Cheap enough for post-training analysis** but too expensive per-step in training.

### 4.2 Reasoning-ceiling computation

Compute `ρ_k` for k = 1, 2, 4, 8, 16, 32. The first k where `ρ_k < 0.7` (eq. 8) is the model's reasoning ceiling. This takes ~100 M × 6 = 600 M ops, ~0.1 sec on RTX 4080 SUPER. **Practical for periodic monitoring.**

### 4.3 Visualisation

Plot `ρ_k` vs k on log-x axis. Different model architectures should produce distinct curves:
- **Strong-reasoning model**: ρ_k stays high (>0.7) up to k ≈ L/2.
- **Weak-reasoning model**: ρ_k drops rapidly past k ≈ 2.
- **Pathological models**: ρ_k is non-monotonic — reasoning capability emerges and then disappears at certain depths.

The plot is a *fingerprint* of the model's reasoning capability, computable post-hoc on the trained model.

---

## 5. Reasoning-task auxiliary loss

### 5.1 Conditioning regulariser

Define the auxiliary loss:

```
L_reason  :=  λ_reason  ·  Σ_{k ∈ K_target}  (1 − ρ_k)^2                                          (eq. 9)
```

where K_target = {2, 4, 8} is a chosen set of target reasoning depths (matching task expectations: GSM8K math problems average ~3-5 steps; multi-hop QA ~2-4 hops).

This regulariser penalises models whose C_k operators are poorly conditioned at task-relevant depths. Empirically (conjectured): training with L_reason should bias the model toward developing strong k-step reasoning chains.

### 5.2 Cost

ρ_k computation requires stochastic Lanczos (§4.1) at cost ~0.1 sec per gradient step. **Too expensive per-step.**

**Mitigation**: compute ρ_k periodically (every K = 100 steps). Cache the result. Use the cached value for the regulariser between updates.

Cost amortisation: 0.1 sec / 100 steps × 1 step/sec = 0.001 fraction of training time. **Negligible.**

### 5.3 Gradient flow

Backward through ρ_k requires differentiating through the Lanczos process. Use the **implicit-function adjoint** (same machinery as REFLECTOR #46 and SFA's iter-2 adjoint): for an eigenvalue λ of C_k with eigenvector v, the gradient `∂λ / ∂R` is `v^T (∂C_k/∂R) v` — local in the path structure.

For the dominant eigenvalue and σ_max (eq. 7), this gives explicit gradient formulae through the restriction-map composition. Cost ~ M_Lanczos additional matvecs per backward step.

---

## 6. Depth-targeting layer placement

Given a target task's reasoning depth k_target (e.g., GSM8K: k_target = 5), CSR suggests architecture choices:

### 6.1 Allocate layers to optimise C_{k_target}

Standard transformer stacks L layers uniformly. CSR suggests:
- Allocate `L_FE = floor(L · 0.25)` layers as feature-extractors (small ℓ_1).
- Allocate `L_SY = k_target − 2` synthesizer layers (mid-stack ℓ_2, ..., ℓ_{k_target-1}).
- Allocate `L_CB = floor(L · 0.25)` layers as consensus-builders (large ℓ_{k_target}).
- Allocate remaining layers as auxiliary refinements.

For k_target = 5, L = 24: L_FE = 6, L_SY = 3, L_CB = 6, plus 9 auxiliary. The auxiliary layers run multi-pole SRA (P = 4) for *parallel reasoning chains* — captures concurrent reasoning paths.

This is **task-aware depth allocation** — different tasks get different stack configurations matching their reasoning depth.

### 6.2 Compatibility with PSA pruning

If PSA identifies certain layers as inert (low birth/death), they can be safely pruned. But pruning must respect the k_target allocation: pruning too many synthesizer layers could break C_{k_target}.

Combined PSA+CSR pruning algorithm:
1. Identify candidates for pruning via PSA's `birth_ℓ + death_ℓ` ordering.
2. Compute the reasoning-capacity loss `(1 − ρ_k_target)` *with the prune-candidate removed*.
3. Prune only if the loss change is ≤ 0.05.

This makes pruning **capability-aware**.

---

## 7. Magnitude claim

CSR does not contribute raw-compute speedup. Its magnitude claim is **quality-per-FLOP on reasoning tasks**:

| Task | Baseline NLL (uniform stack) | NLL with CSR-trained model |
|---|---|---|
| GSM8K (math reasoning, k_target ≈ 5) | (set by SFA+SRA stack) | -0.15 to -0.25 nat (conjectured) |
| HellaSwag (commonsense inference, k_target ≈ 3) | (baseline) | -0.10 to -0.20 nat |
| Multi-hop QA (k_target ≈ 4) | (baseline) | -0.20 to -0.30 nat |
| Pure-language modelling (no specific k) | (baseline) | ±0.05 nat (no specific benefit) |

The improvements are **task-targeted**. CSR is most valuable for reasoning-heavy benchmarks; on pure-language data, the benefit is small.

This is honest: not every paradigm helps every metric. CSR's magnitude is in *capability*, not throughput.

---

## 8. Gate-0 falsification

### 8.1 Probe L (new, CSR-specific): reasoning-ceiling metric on flagship

Compute `ρ_k` for k = 1, 2, 4, 8 on `chiron_1B_T16384.step30000` (post-SFA hot-swap from Gate-0 Probe B).

**Pass criterion**: `ρ_k` is well-behaved (monotonically non-increasing in k, with `ρ_4 > 0.5`). This confirms the existence of multi-step reasoning structure in the trained model.

**Fail criterion**: `ρ_k` is degenerate (random) or non-monotonic. This suggests the reasoning-capacity metric doesn't capture useful structure — re-design ρ_k definition.

**Cost**: ~5 minutes (Lanczos × 4 k values × 0.5 sec each).

### 8.2 Probe M (new, CSR-specific): auxiliary loss improves reasoning task

Train two 66M models for 5000 steps:
- Model 1: standard CSA stack (SFA+SRA+PSA).
- Model 2: same stack + `L_reason` regulariser with K_target = {2, 4}.

Evaluate both on a simple reasoning task (e.g., a small multi-hop QA synthetic dataset).

**Pass criterion**: Model 2 achieves NLL ≤ Model 1 by ≥ 0.05 nat on the reasoning task.
**Fail criterion**: L_reason gives no benefit (or regresses) → reasoning capacity is not the bottleneck for the synthetic task.

**Cost**: ~1 GPU-hour (two 5000-step 66M training runs in parallel).

### 8.3 Probe N (new, CSR-specific): depth-allocation matters

Train three 66M models with different depth allocations targeting k = 3:
- Uniform: 24 layers, all standard SFA.
- CSR-allocated for k=3: 6 FE + 1 SY + 6 CB + 11 auxiliary multi-pole.
- CSR-allocated for k=5 (wrong target): 6 FE + 3 SY + 6 CB + 9 auxiliary.

**Pass criterion**: The k=3-allocated model performs best on k=3 reasoning, the k=5-allocated model performs best on k=5 reasoning, uniform is intermediate on both.
**Fail criterion**: Depth allocation has no impact — uniform performs equivalent — rejects the task-aware-allocation claim.

**Cost**: ~3 GPU-hours (three 66M runs).

### 8.4 Total cost

Probes L, M, N total ≈ 4 GPU-hours, **on top of paradigm #250-253's Gate-0 (~3 GPU-hours)**. Net Gate-0 budget for the full CSA stack with CSR: ~7 GPU-hours. Still feasible.

---

## 9. Open conjectures

### 9.1 Conjecture 9 (reasoning ceiling)

**Conjecture 9**: For an L-layer SFA-stacked model, `k★(model) ≤ L/2` for any reasonable training procedure. I.e., depth limits reasoning depth, and the relationship is roughly k_max = L/2.

This is conjectural; the constant 1/2 could be 1/3 or 2/3 empirically. The qualitative claim — that depth is the binding constraint on reasoning — is well-supported in transformer literature.

### 9.2 Conjecture 10 (reasoning capacity is preserved under PSA pruning)

**Conjecture 10**: PSA pruning of the bottom 30% layers preserves `ρ_k` for k ≤ L/4 within ε = 0.05.

This would mean: PSA's compute savings don't come at the cost of reasoning capability. Falsifiable by Gate-0 Probe N variant.

### 9.3 Conjecture 11 (C_k structure mirrors data-distribution structure)

**Conjecture 11**: For language model training, the spectrum of C_k is informative about the *distribution* of reasoning patterns in training data. Specifically: training on math-heavy data should produce `ρ_4` (4-step reasoning) high; training on simple-narrative data should produce `ρ_4` lower.

If true, CSR provides a *diagnostic* for what kind of reasoning the model has learned. Testable by comparing C_k spectra of differently-trained models.

---

## 10. Implementation roadmap

After paradigms #250-253's Phase 4 (1B validation) completes:

**Phase 0 — CSR Gate-0** (4 GPU-hours). Probes L, M, N.

**Phase 1 — ρ_k computation infrastructure** (2 iterations).
- Implement stochastic Lanczos for C_k spectrum estimation.
- Add `--csr-monitor-k 1,2,4,8` flag.

**Phase 2 — Reasoning auxiliary loss** (3 iterations).
- Implement L_reason gradient flow through Lanczos.
- Add `--csr-loss-weight 0.01 --csr-k-target 2,4,8` flags.

**Phase 3 — Depth-targeting allocation** (2 iterations).
- Task-aware layer allocation per §6.1.
- Add `--csr-task-k 5` flag for task-specific stacks.

**Phase 4 — Validation** (2-3 iterations).
- 66M reasoning benchmark suite (synthetic multi-hop QA, simple math).
- 1B reasoning-task evaluation.

**Phase 5 — Production** (1 iteration).
- Default flag: `--csr` enabled with K_target = {2, 4, 8}.

**Total**: ~8-10 iterations after paradigms #250-253 complete.

---

## 11. Summary

CSR addresses **multi-step reasoning capability** as an explicit mathematical structure in the CSA framework. The composition operator `C_k` aggregates path-composed restriction maps; its spectrum measures k-step reasoning capacity. CSR provides three operational deliverables:

1. **Reasoning-ceiling metric** `k★(model)` (eq. 8): the max k where ρ_k > 0.7, computable in ~0.1 sec on a trained model.
2. **Auxiliary loss** L_reason (eq. 9): regulariser encouraging well-conditioned C_k at task-relevant depths.
3. **Depth-targeting allocation** (§6): task-aware layer configuration based on k_target.

The magnitude claim is **quality-per-FLOP on reasoning tasks**: 0.10–0.30 nat NLL improvement on multi-step reasoning benchmarks at iso-FLOP, *concentrated* on reasoning-rich samples (not uniform improvement).

CSR is the program's first **capability paradigm** — focused on what the model can do, not how fast or how cheap. It connects to PSA (k → ∞ limit of C_k recovers the colimit sheaf), SLR (role assignment maps to path-step positions), and SRA (pole structure encodes relation types).

Three new conjectures (9, 10, 11) about reasoning ceiling, pruning compatibility, and data-distribution reflection. Three new Gate-0 probes (L, M, N) for ~4 GPU-hours. Combined CSA stack Gate-0 budget: ~7 GPU-hours.

**Predecessor**: SLR (#253) for layer roles.
**Successor**: paradigm #255 (TBD) — Dynamic Depth, originally sketched in CELLULAR_SHEAF_ATTENTION_PROGRAM.md §7 as #254 but renamed to #255 to accommodate CSR's primacy. Dynamic Depth uses per-input commutation defect ε^{(φ)} to runtime-insert layers when the input requires deeper reasoning.

**The program now spans seven paradigm directions**: substrate (#250), focus (#251), depth observation (#252), depth configuration (#253), reasoning capability (#254), dynamic depth (#255 sketched), and further (still open).

This is the first time the program explicitly addresses *what the model can compute* (#254) rather than *how it computes it* (#250-253). It opens a research thread distinct from raw compute — one focused on capability ceilings, reasoning structure, and task-targeted architecture.
