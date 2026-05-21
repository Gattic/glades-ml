# Paradigm Shift #252 — PSA: Persistent Sheaf Attention

**Status:** designed (Ralph-loop iter 4, 2026-05-14). Builds on paradigm #250 SFA, iter-2 Conjecture 3, and paradigm #251 SRA.
**Date:** 2026-05-14.
**Branch:** vesta5.
**Predecessors:** #250 SFA (Sheaf-Focal Attention) — per-token cellular sheaf; #251 SRA — per-query complex-pole resolvent on the sheaf.
**Axis:** Track the evolution of the per-layer cellular sheaf cohomology `H^k(F_ℓ)` across the depth-L stack of SFA/SRA layers. The persistent cohomology (in the sense of Topological Data Analysis / persistent homology) provides a principled, mathematically grounded mechanism for **layer-pruning** (removing layers that add no new cohomology), **depth-tuning** (adding layers where rich new cohomology is needed), and **layer-specialisation** (each layer's role becomes legible).
**Magnitude target:** 1.4–1.6× wall-clock speedup at iso-quality via principled layer pruning on top of SFA + SRA's combined 6–8×. Stack total: **8.4–12.8× wall-clock at iso-NLL over SCFA flagship**. Plus: legibility — each layer's role becomes computable and interpretable.

---

## 0. Executive summary

A deep transformer with L layers stacked sequentially is empirically observed to be **over-parameterised in depth**: literature (Sajjad 2023, Gromov et al. 2024) shows 30–50% of layers can be pruned with negligible perplexity loss. Yet today's pruning heuristics (gradient magnitude, attention entropy, activation norm) are *post-hoc* and *unprincipled* — they correlate with importance but don't measure it directly.

SFA (paradigm #250) attached a cellular sheaf `F_ℓ` to each layer. Iter 2's Conjecture 3 (PARADIGM_SHIFT_250_PROOFS.md §7) proposed:

> The composed transformation `Y_full = (SFA_L ∘ ... ∘ SFA_1)(x)` is the projection of a **global sheaf F_∞** onto the residual stream, where F_∞ is the colimit (direct limit) of the diagram `F_1 → F_2 → ... → F_L` with morphisms determined by the residual-stream coupling.

PSA promotes this conjecture into an operational framework. The central object is the **persistence module**:

```
H^k(F_•) := { H^k(F_ℓ) }_{ℓ=1..L}  with maps  H^k(F_ℓ) → H^k(F_{ℓ+1}) induced by φ_ℓ : F_ℓ → F_{ℓ+1}.    (eq. 1)
```

This is a persistence module in the sense of persistent homology (Edelsbrunner & Harer 2010; Carlsson 2009). Its **persistence diagram** PD^k decomposes the layer-stack into:

- **Long bars** = cohomology classes that persist across many layers. These are *globally consistent* model representations (e.g., entity identities, syntactic agreement chains).
- **Short bars** = cohomology classes specific to a small window of layers. These are *transient* processing (e.g., partial parse trees, local linguistic features in a phrase).

PSA uses the persistence diagram for three operational benefits:

1. **Principled layer pruning** (§5.1): layers contributing few new long-bar births can be removed. Targets 1.4–1.6× wall-clock by pruning 30% of layers at iso-NLL.
2. **Depth-tuning by birth/death balance** (§5.2): when a new short bar is born and dies in the same layer (single-layer-lifetime), the layer is "over-saturated" — duplicate it. When a long bar dies prematurely (mid-stack), the model is too shallow — insert a refinement layer.
3. **Layer-role legibility** (§5.3): each layer's contribution to H^0, H^1, ... is computable. Layers split naturally into "consensus-builders" (high H^0 contribution), "tension-detectors" (high H^1 contribution), "harmonic-readers" (high H^2 contribution), etc.

PSA also provides a **self-supervised auxiliary signal**: the **persistence loss** (§6) penalises models whose persistence diagrams are *too sparse* (collapse to layer-uniformity) or *too dense* (no consensus emerges). This regulariser, added to the standard NLL loss, encourages a healthy spread of bar lengths.

The honest magnitude claim: PSA *itself* contributes modest direct compute reduction (~1.4–1.6× via pruning at iso-quality). Its larger value is **architectural legibility** — a principled, computable measure of what each layer does. This unlocks future paradigm shifts (#253, #254...) that target specific layer roles.

---

## 1. Why this is paradigm #252

Forecast in:
- PARADIGM_SHIFT_250_DESIGN.md §14.2: "Paradigm #252 candidate: Persistent Sheaf Attention. Track persistent sheaf cohomology across layers."
- PARADIGM_SHIFT_250_PROOFS.md §7: Conjecture 3 on emergent global-sheaf structure across layers.
- PARADIGM_SHIFT_251_DESIGN.md §9: "Successor: paradigm #252 (TBD) — Persistent Sheaf Attention, extending Conjecture 3."

This iter promotes the candidate to a full design. Reasons #252 (not earlier or later):

1. **Logical depth ordering**: paradigm #250 introduced per-layer sheaves. Paradigm #251 enriched the per-layer attention via complex poles. Paradigm #252 *integrates across layers* — the next axis after per-layer and per-query is per-stack.
2. **TDA connection**: persistent homology (Edelsbrunner-Harer-Carlsson) provides a mathematically mature framework for tracking topological invariants under filtration. Cellular sheaves admit a natural filtration by layer ℓ. The match is exact.
3. **Concrete engineering use**: layer-pruning is a measurable engineering win (1.4–1.6×) on top of paradigms #250 + #251. The math earns its keep operationally.
4. **Falsifiable**: persistence diagrams are computable from trained model state. If PSA's prediction (pruning by short-bar layers preserves NLL) fails empirically, the framework is falsified.

---

## 2. Mathematical setup — inter-layer sheaf morphisms

### 2.1 Recap: per-layer sheaf

From paradigm #250 (and #251 if SRA is in the stack): each layer ℓ ∈ {1, ..., L} has a cellular sheaf F_ℓ over the causal token graph G = (V, E):

```
F_ℓ = ( {F_ℓ(v_i) = R^{d_s}}_i, {R^{(ℓ)}_{j ← i} = U^{(ℓ)}_j Σ^{(ℓ)}(i,j) (U^{(ℓ)}_i)^T}_{(i,j) ∈ E} )
```

with sheaf Laplacian `L_{F_ℓ} = δ^T_ℓ δ_ℓ`. The per-layer cochain space C^0(F_ℓ) ≅ R^{T · d_s}; the per-layer cohomology:

- H^0(F_ℓ) = ker(δ_ℓ) = globally-consistent sections (cohomology degree 0).
- H^1(F_ℓ) = ker(d_2) / im(δ_ℓ) where d_2 is the next coboundary (for graphs, H^1 captures cycle obstructions; for trees, H^1 = 0).

For our causal-sliding-window-plus-sinks graph, H^1(F_ℓ) is nontrivial because the graph has cycles (sinks have edges to all positions, and short-range edges form many small cycles).

### 2.2 Inter-layer morphism

The residual stream coupling between consecutive layers induces a natural map. Layer ℓ takes input `x_ℓ ∈ R^{T × m}` (the residual stream), and the source assembly `b^{(ℓ)}_i = U^{(ℓ)}_i (U^{(ℓ)}_i)^T P_q W^{(ℓ)}_Q x^{(ℓ)}_i + γ_ℓ P_v W^{(ℓ)}_V x^{(ℓ)}_i` lifts the input into C^0(F_ℓ). The output `y^{(ℓ)}_i = P^{(ℓ)}_o^T s^{★(ℓ)}_i + W^{(ℓ)}_Q x^{(ℓ)}_i` plus residual gives `x^{(ℓ+1)}_i = x^{(ℓ)}_i + y^{(ℓ)}_i + (FFN contributions)`.

Define the **inter-layer sheaf morphism** φ_ℓ : F_ℓ → F_{ℓ+1} as the composition:

```
φ_ℓ :  F_ℓ(v_i) = R^{d_s}    --(P^{(ℓ)}_o)-->  R^{d_h}    --(α_ℓ · I)-->  R^{d_h}   --(P^{(ℓ+1)}_q^T, P^{(ℓ+1)}_v)-->  F_{ℓ+1}(v_i) = R^{d_s}     (eq. 2)
```

with `α_ℓ ∈ R` a scalar weighting the layer's contribution to the residual stream (in practice typically 1, possibly scaled by a layer-norm factor). The composition expresses how layer ℓ's stalk-section is *transported* into layer ℓ+1's stalk-section through the residual-stream coupling.

For φ_ℓ to be a valid sheaf morphism, it must respect restriction maps:

```
φ_{ℓ}(R^{(ℓ)}_{j ← i} s_i)  =  R^{(ℓ+1)}_{j ← i}  φ_{ℓ}(s_i)        (commuting diagram condition)
```

In general this commuting condition does NOT hold exactly — the restriction maps R^{(ℓ)}_{j ← i} and R^{(ℓ+1)}_{j ← i} are independently learned. The **commutation defect**:

```
ε^{(φ)}_ℓ(i, j)  :=  φ_ℓ(R^{(ℓ)}_{j ← i} s_i)  −  R^{(ℓ+1)}_{j ← i}  φ_{ℓ}(s_i)        (eq. 3)
```

measures how much the inter-layer morphism *fails* to be sheaf-coherent.

**Key observation**: ε^{(φ)}_ℓ is non-zero precisely when layer ℓ → ℓ+1 *transforms* the perspective relationships. If ε^{(φ)}_ℓ = 0 for all (i, j), then layer ℓ does NOT change the perspective structure — it can be pruned without loss. This is the **first principled pruning signal** in PSA.

### 2.3 The persistence module

Define the layered diagram:

```
F_1  --φ_1-->  F_2  --φ_2-->  ...  --φ_{L-1}-->  F_L        (eq. 4)
```

Apply the H^k functor (k = 0, 1, 2, ...). H^k is a covariant functor on the category of cellular sheaves with sheaf morphisms (Curry 2014). So:

```
H^k(F_1)  --H^k(φ_1)-->  H^k(F_2)  --H^k(φ_2)-->  ...  --H^k(φ_{L-1})-->  H^k(F_L)        (eq. 5)
```

This is a **persistence module** indexed by ℓ ∈ {1, ..., L} (a finite poset).

### 2.4 The persistence diagram (decomposition theorem)

By the **structure theorem for persistence modules** (Crawley-Boevey 2015): any finite-dimensional persistence module decomposes uniquely (up to isomorphism) as a direct sum of *interval modules*:

```
H^k(F_•)  =  ⊕_α  I[b_α, d_α]                                        (eq. 6)
```

where I[b, d] is the interval module that is "1-dimensional in degrees b, b+1, ..., d-1 and zero elsewhere," with identity maps on overlapping degrees. Each interval [b_α, d_α] is a **bar** in the persistence diagram, representing a cohomology class that is *born* at layer b_α and *dies* at layer d_α.

The collection of bars `PD^k := { (b_α, d_α) }_α` is the **persistence diagram in degree k**. Long bars (d_α − b_α large) = persistent global content. Short bars = transient local content.

---

## 3. Computing the persistence diagram in practice

### 3.1 Cellular sheaf cohomology via Laplacian kernel

For a cellular sheaf F over a finite graph, the 0-th cohomology is computable via the kernel of the sheaf Laplacian:

```
H^0(F_ℓ)  =  ker(L_{F_ℓ}) ⊂ C^0(F_ℓ) ≅ R^{T · d_s}              (eq. 7)
```

This is the standard "harmonic section" characterisation. The dimension of H^0(F_ℓ) is the number of eigenvalues of L_{F_ℓ} that are zero (or, in the Tikhonov-regularised setting, below a chosen threshold `μ_th`).

For H^1, the formula is:

```
H^1(F_ℓ)  =  ker(d_2) / im(δ_ℓ)
```

For cellular sheaves over graphs (no 2-cells), d_2 = 0, so H^1(F_ℓ) = C^1(F_ℓ) / im(δ_ℓ). The dimension:

```
dim H^1(F_ℓ)  =  |E| · d_s  −  rank(δ_ℓ)  =  |E| · d_s  −  (T · d_s  −  dim H^0(F_ℓ))    (eq. 8)
```

So both H^0 and H^1 reduce to computing `dim ker(L_{F_ℓ})` (or `dim ker(δ_ℓ)`) for each layer.

### 3.2 Numerical computation

At training time and at convergence, compute per-layer:

```
For each ℓ ∈ {1, ..., L}:
    Build L_{F_ℓ} matvec (already done in SFA forward).
    Run Lanczos for m = 32 steps (cheap, one-time post-convergence).
    Extract the lowest 32 eigenvalues {μ^{(ℓ)}_1, ..., μ^{(ℓ)}_{32}}.
    Threshold: μ < μ_th = 10^{-3} ⇒ "harmonic" (counts toward H^0).
    Count: dim H^0(F_ℓ) ≈ #{ p : μ^{(ℓ)}_p < μ_th }.
    Plus the corresponding eigenvectors {v^{(ℓ)}_p}_{p in harmonic set}.
```

Cost: m = 32 Lanczos steps per layer × L layers = 32 L matvecs = ~800 matvecs at L=24. Each matvec costs O(|E| · d_s · r) ≈ 16K ops at default config; total = 13M ops one-time. **Negligible**.

### 3.3 Tracking the persistence module

The interval-module decomposition (eq. 6) requires tracking how H^0(F_ℓ) maps into H^0(F_{ℓ+1}) under φ_ℓ. Operationally:

```
For each ℓ ∈ {1, ..., L-1}:
    Compute the matrix representation of H^0(φ_ℓ): a linear map between harmonic eigenspaces.
    Use SVD: write H^0(φ_ℓ) = U_φ Λ_φ V_φ^T.
    The singular values Λ_φ tell us which directions in H^0(F_ℓ) are preserved into H^0(F_{ℓ+1}).
    Threshold: Λ_φ_jj > 0.5 ⇒ direction is preserved ("survives" into layer ℓ+1).
              Λ_φ_jj ≤ 0.5 ⇒ direction is killed ("dies" at layer ℓ+1).
```

Cost: per layer pair, m × m SVD = O(m^3) = O(32^3) = 32K ops. Per-stack: (L − 1) × 32K ≈ 730K ops. **Trivial**.

### 3.4 Persistence diagram construction

Run the standard persistent-homology bar-tracking algorithm:

```
For each ℓ = 1 to L:
    For each direction d in H^0(F_ℓ) (harmonic eigenvector):
        If d is "new" (not in image of H^0(φ_{ℓ-1})):
            Birth a new bar at layer ℓ.
        Else:
            Continue an existing bar.
    For each existing bar tracked from layer ℓ-1:
        If H^0(φ_{ℓ-1}) does not preserve this direction:
            Die the bar at layer ℓ.

Return PD^0 = { (b_α, d_α) }_α (with d_α = L + 1 for bars that "live forever").
```

Persistence diagram in degree 1 (PD^1) follows the same recipe with H^1 instead of H^0.

### 3.5 Persistence module as model introspection

At training-time, compute the persistence diagram every K = 1000 steps (or per-checkpoint). Visualise:
- **Bar length distribution**: histogram of (d_α − b_α). Healthy training has a mix of short and long bars.
- **Bar density per layer**: count of bars active at each ℓ. Layers with few active bars are "narrow"; with many are "wide."
- **Birth/death events per layer**: count of bars born/dying at each ℓ. Layers with high birth count are "feature-extractors"; high death count are "consensus-builders."

These statistics make layer roles **legible** in a mathematically grounded way, replacing the post-hoc "attention entropy" / "activation norm" heuristics.

---

## 4. The persistence loss (training-time regulariser)

To prevent degenerate persistence diagrams (all-bars-die-immediately or all-bars-persist-forever), add a **persistence loss** to the training objective:

```
L_persist  =  λ_persist · D_W( PD^0 , PD^0_target )                  (eq. 9)
```

where:
- `PD^0` = the current model's degree-0 persistence diagram.
- `PD^0_target` = a *target* diagram chosen at training-design time (e.g., the persistence diagram of a well-known reference model, or a synthetic "ideal" diagram with a healthy mix of bar lengths).
- `D_W` = the **Wasserstein distance** between persistence diagrams (a standard TDA metric; computable via optimal transport on R^2).

Cost: bottleneck-distance computation is O(B^{1.5}) where B is the number of bars; B is bounded by m·L = 32·24 = 768. So cost ≈ 21K ops per training step (if computed every step) or 21M ops if computed every K=1000 steps. **Negligible**.

### 4.1 Choice of target diagram

For language models, the *empirical* persistence diagram of well-trained transformers (estimated from trained checkpoints of GPT-2 / Pile-trained CHIRON) gives a target. Alternatively, a synthetic target with bars at scales matching the linguistic-phenomenon hierarchy:

- Very long bars (full L): syntactic identity, entity tracking.
- Medium bars (L/2): phrase-level structure, agreement chains.
- Short bars (1-2 layers): local features (token-level POS, n-gram context).

The exact target is empirical and can be tuned per data distribution.

### 4.2 Why this regulariser helps

The persistence loss provides an *architecture-level* training signal that complements NLL. NLL is a *per-token* prediction loss; L_persist is a *whole-model* structure loss. Together they encourage the model to organise its layers into a healthy persistence diagram, which empirically correlates with better generalisation (longer effective context, more robust attention patterns) — to be verified at Gate-0.

---

## 5. Operational uses of the persistence diagram

### 5.1 Principled layer pruning

A layer ℓ is a **candidate for pruning** if:

```
| birth_ℓ - death_ℓ | < pruning_threshold        (eq. 10)
```

where `birth_ℓ = #{α : b_α = ℓ}` (bars born at ℓ) and `death_ℓ = #{α : d_α = ℓ}` (bars dying at ℓ). The condition says: layer ℓ neither *births many new bars* (= introduces new representational capacity) nor *kills many bars* (= consolidates existing capacity).

Pruning algorithm:

```
For each ℓ ∈ {1, ..., L}:
    Compute net_contribution_ℓ = birth_ℓ + death_ℓ.
    Rank layers by net_contribution_ℓ.
    Prune the bottom K (default K = 0.3 L) layers.
    Re-evaluate persistence diagram on the pruned stack.
    Re-train for ~500 steps to re-stabilise.
```

**Magnitude claim**: K = 0.3 L removes 30% of layers → 1.4× wall-clock speedup. At iso-NLL (verified by re-evaluation), this is a direct compute saving.

**Conservative bound**: K = 0.2 L gives 1.25× speedup at iso-NLL. Aggressive K = 0.4 L gives 1.7× but risks NLL regression.

### 5.2 Depth-tuning by birth/death balance

A different signal: identify **bottleneck layers** where many bars die in a small window. This signals that the model is *over-consolidating* — losing distinctions that downstream layers might need. The remedy: *insert* a refinement layer just before the bottleneck.

Algorithm:

```
For each ℓ ∈ {1, ..., L}:
    Compute death_rate_ℓ = death_ℓ / total_bars_at_ℓ.
    If death_rate_ℓ > 0.3:
        Mark ℓ as a bottleneck candidate.

For each bottleneck candidate ℓ:
    Insert a new layer ℓ' between ℓ-1 and ℓ.
    Initialise its sheaf to the SVD-truncated average of F_{ℓ-1} and F_ℓ.
    Re-train for ~500 steps.
    Measure NLL change. If improved, keep; if regressed, revert.
```

**Magnitude claim**: bottleneck-driven layer insertion is conjectured to improve NLL by 0.05–0.10 nat at iso-FLOP (the added compute of one layer is offset by 0.1-nat NLL gain → translates to 5-10% effective compute saving via reduced steps-to-target).

### 5.3 Layer-role legibility

Compute per-layer:

```
Role_ℓ = ( birth_ℓ / max_layer_birth,  death_ℓ / max_layer_death,  persistent_active_ℓ / max_persistent )
```

Cluster layers by `Role_ℓ` (k-means in R^3):
- **Feature extractors** (high birth, low death): early layers; introduce new representational primitives.
- **Mid-stack synthesizers** (medium birth, medium death): integrate features into mid-level representations.
- **Consensus builders** (low birth, high death): late layers; consolidate features into a coherent global representation.
- **Inert layers** (low birth, low death): candidates for pruning (§5.1).

This taxonomy is computable, mathematically grounded (via cohomology), and gives a principled framework for *interpreting* deep-transformer organization that goes beyond ad-hoc heuristics.

---

## 6. Composition with paradigms #250 and #251

PSA does NOT replace the per-layer attention mechanism. It *observes* and *regulates* the stack of SFA / SRA layers. So composition is by addition, not substitution:

| paradigm | role in PSA |
|---|---|
| #42 SCFA | base, when SFA/SRA disabled |
| #250 SFA | per-layer attention substrate; provides L_{F_ℓ} for each layer |
| #251 SRA | per-query focus on per-layer sheaf; persistence is computed on the SRA-induced L_{F_ℓ} |
| #78 Sinks | edge set E (shared across layers); sinks may have special role in long-bar lifetimes |
| #46 REFLECTOR | backward through Chebyshev (SFA) and resolvent (SRA) solves |

**Stack projection**:

| Stack | Per-step | Steps-to-target-NLL | Layer count | Total |
|---|---|---|---|---|
| SCFA flagship | 1× | 1× | L | 1× (baseline) |
| + SFA (#250) | ~1.01× | 0.5-0.7× | L | 1.4-2× |
| + SRA (#251) | 4.3× | 0.5-0.7× | L | 6.0-8.6× |
| + PSA (#252, layer pruning at K=0.3L) | 1.4× | inherited | 0.7L | **8.4-12× total** |
| + PSA bottleneck-tuning | 1.0× (cost-neutral) | additional 0.9× | varies | **9.3-13.4× total** |

The headline magnitude claim of the SFA + SRA + PSA stack: **~10× wall-clock speedup at iso-NLL over SCFA flagship** at T=16384, with PSA contributing the final ~1.5× via principled layer pruning.

---

## 7. Open conjectures and tests

### 7.1 Conjecture 6 (persistent cohomology and language structure)

**Conjecture 6**: For LLM training data with linguistic phenomena Φ (anaphora, agreement, embedded discourse), the persistence diagram PD^0 of a trained SFA/SRA model has:
- **Bars of length ≥ L/2** corresponding to *globally consistent* representations: entity identities, syntactic agreement chains.
- **Bars of length L/4 to L/2** corresponding to *phrase-level structure*: noun phrases, verb phrases, embedded clauses.
- **Bars of length ≤ L/8** corresponding to *transient features*: position-specific embeddings, low-level n-gram patterns.

The conjecture's significance: the **persistence diagram of a trained transformer is a fingerprint of its linguistic competence**. Different training corpora (literature vs code vs scientific text) should produce distinct persistence diagrams. This is testable empirically post-implementation.

### 7.2 Conjecture 7 (layer-pruning preserves PD^0 long bars)

**Conjecture 7**: For a trained SFA/SRA stack of L layers, pruning the K = 0.3 L layers with lowest `birth_ℓ + death_ℓ` preserves the long-bar (length ≥ L/2) portion of PD^0 within Wasserstein-distance ε ≤ 0.1, and the resulting model's NLL is within 0.02 nat of the unpruned baseline.

This is the *operational* heart of PSA. If Conjecture 7 holds, PSA delivers the 1.4× wall-clock claim. If it fails, layer pruning has to fall back to ad-hoc heuristics. **Gate-0 directly tests this** (Probe H below).

### 7.3 Conjecture 8 (commutation-defect ε^{(φ)} as feature)

**Conjecture 8**: The commutation defect `ε^{(φ)}_ℓ(i, j)` (eq. 3) is empirically *informative* — its magnitude per token correlates with *linguistic transitions* in the input (e.g., clause boundaries, topic shifts).

This is the most speculative conjecture. If true, it provides a *per-token* legibility signal: each token has a magnitude indicating *how much the model's interpretation of it is changing across layers*. This is a strictly stronger interpretability signal than attention maps.

### 7.4 Gate-0 probes

In addition to inheriting paradigm #250's Probes A-E:

**Probe H (new, PSA-specific)**: Compute the persistence diagram of `chiron_1B_T16384.step30000` (post-SFA hot-swap from Gate-0 Probe B):
- Extract L_{F_ℓ} for each ℓ ∈ {1, ..., 24}.
- Run m = 32 Lanczos per layer.
- Compute PD^0 via the bar-tracking algorithm.
- **Visualise**: persistence diagram, bar length histogram, layer role clustering.

**Pass**: at least 30% of bars have length ≥ L/2 (well-developed persistent content).
**Fail**: bars cluster at very short lengths (no persistence), or all bars at length L (no layer differentiation). Both indicate degenerate cohomology.

**Probe I (new, PSA-specific)**: Layer-pruning experiment.
- Identify the 30% lowest-`birth_ℓ + death_ℓ` layers.
- Prune them.
- Re-train for 500 steps.
- Compare NLL to baseline.

**Pass**: ΔNLL ≤ +0.02 nat. **Magnitude verification**: pruned model is ~1.4× faster.
**Fail**: ΔNLL > +0.05 nat — pruning damages quality; PSA's operational claim is falsified. Layer pruning falls back to heuristics; PSA's value is reduced to interpretability only.

**Probe J (new, PSA-specific)**: Persistence loss helps training.
- Train two 66M models for 5000 steps: one with L_persist, one without.
- Compare final NLL.

**Pass**: with-L_persist achieves NLL ≤ without by ≥ 0.02 nat.
**Fail**: with-L_persist achieves NLL > without — L_persist is a useless regulariser. PSA's training-time use is falsified.

**Total Gate-0 budget**: ≤ 2 GPU-hours (inherited SFA Probes A-E + new H, I, J).

---

## 8. Implementation roadmap

PSA is *post-hoc* on top of SFA/SRA. Once paradigm #250's Phase 4 (1B validation) is complete:

**Phase 0 — PSA Gate-0** (1 GPU-hour). Probes H, I, J.

**Phase 1 — Persistence diagram computation** (3 iterations).
- Implement Lanczos eigendecomposition per-layer (uses existing SFA matvec).
- Implement bar-tracking algorithm.
- Visualise persistence diagrams in unit-tests / debugging tooling.

**Phase 2 — Layer-pruning** (2 iterations).
- Implement `--psa-prune-fraction 0.3` flag.
- Test on 66M reference: prune, fine-tune 500 steps, compare NLL.

**Phase 3 — Persistence loss** (2 iterations).
- Implement Wasserstein distance for persistence diagrams.
- Add `--psa-persist-loss-weight 1e-3` flag.
- Train 66M model with persistence loss; compare to baseline.

**Phase 4 — Depth-tuning** (2 iterations).
- Implement bottleneck-detection algorithm.
- Layer-insertion mechanism (analog of RLG #39).
- Test on 66M.

**Phase 5 — Production** (1-2 iterations).
- Default: `--psa-prune-fraction 0.3` for inference deployment (training keeps L layers).

**Total**: ~10-12 iterations after paradigm #250 Phase 4 (1B validation).

---

## 9. Summary

PSA promotes iter-2's Conjecture 3 (layer-stacking cohomology) into an operational framework. The per-layer sheaves `F_ℓ` (from SFA, paradigm #250) form a sequential diagram `F_1 → F_2 → ... → F_L`; applying the H^k functor gives a persistence module, decomposed by the structure theorem into bars of varying lifetime. The persistence diagram is computable in ~1M ops one-time and provides:

1. **Principled layer pruning** (§5.1): prune layers with low `birth_ℓ + death_ℓ`. Targets 1.4× wall-clock at iso-NLL (Conjecture 7).
2. **Depth-tuning by bottleneck detection** (§5.2): insert refinement layers where many bars die simultaneously. Targets 0.05–0.10 nat NLL improvement at one extra layer's cost (Conjecture 8).
3. **Layer-role legibility** (§5.3): cluster layers as feature-extractors / synthesizers / consensus-builders / inert by `(birth_ℓ, death_ℓ, persistent_active_ℓ)`. Computable, mathematically grounded.
4. **Persistence-loss regulariser** (§4): Wasserstein distance to a target persistence diagram. Training-time architecture-level signal complementing NLL.

Combined with paradigms #250 (SFA) and #251 (SRA): stack delivers ~10× wall-clock speedup at iso-NLL over SCFA flagship at T=16384, with PSA contributing the final ~1.5× via pruning.

Gate-0 (≤2 GPU-hours): Probes H (persistence diagram health), I (pruning preserves NLL), J (persistence loss helps training). Falsifiable on the existing flagship `chiron_1B_T16384.step30000`.

PSA's broader significance: **it makes deep-transformer architecture mathematically legible**. Each layer's role becomes computable, principled, and interpretable. This opens future paradigm shifts that target specific layer roles (e.g., paradigm #253 may specialise the consensus-builder layers; paradigm #254 may dynamically expand feature-extractor depth based on input complexity).

The connection to Topological Data Analysis is exact: persistence modules, persistence diagrams, Wasserstein distance are all standard TDA tools. PSA imports the full mathematical machinery of persistent homology into LLM architecture for the first time (modulo the connection between sheaves and persistence, which exists in the algebraic-topology literature but has not been operationalised for transformers).

**Predecessors**: #250 SFA (per-layer cellular sheaf), #251 SRA (per-query complex-pole resolvent), iter-2 Conjecture 3 (the formal substrate for PSA).
**Successor**: paradigm #253 (TBD) — Specialised Layer Roles, leveraging PSA's layer-role taxonomy to design layer-type-specific attention forms (e.g., consensus-builder layers use H^0-projection attention; tension-detector layers use H^1-augmenting attention).
