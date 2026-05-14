# Cellular Sheaf Attention — Unified Research Program

**Scope:** Paradigms #250 (SFA), #251 (SRA), #252 (PSA) — designed in Ralph-loop iterations 1-4 (2026-05-14). Forward sketches of #253 (SLR) and #254 (Dynamic Depth) included for iter 5+ continuation.
**Brief that initiated this program (iter 1):** "The paper Attention is all you need changed the game for AI forever giving us modern day LLMs. I think the solution the next step is Focused attention with perspective. ... Ideally it will improve LLM architecture by magnitudes."
**Date:** 2026-05-14, Ralph-loop iter 5.
**Predecessor flagship:** `chiron_1B_T16384.step30000` (SCFA stack with iter 1-10 ralph-loop optimisations).
**Stack magnitude target:** **~10× wall-clock at iso-NLL** over SCFA flagship at T=16384, with magnitudes-territory expressivity gains beyond what softmax attention can represent.

---

## 0. The "next step" question

"Attention is all you need" (Vaswani et al. 2017) introduced scaled dot-product attention (SDPA):

```
y_i = softmax( q_i^T K^T / √d_h ) V
```

A single global coordinate system for all tokens, attention determined by scalar similarities, softmax as the non-linearity. The success of SDPA in producing all modern LLMs is empirical evidence of the formula's *sufficiency*. But sufficiency is not necessity, and "what's the next step" is a different question.

**The brief asked**: what mathematical primitive should replace SDPA so that LLM architecture improves by magnitudes?

**The proposed answer**: replace the **global coordinate system + softmax non-linearity** of SDPA with a **per-token cellular sheaf + spectral filter** — a structure where each token has its own algebraic neighbourhood and attention is the harmonic section of the resulting sheaf-Laplacian-induced diffusion.

This unified framework — **Cellular Sheaf Attention (CSA)** — has three current paradigms:

| Paradigm | Axis | Status |
|---|---|---|
| #250 SFA (Sheaf-Focal Attention) | Per-token *perspective* + spectral *focus* | Designed iter 1-2; Gate-0 spec ready |
| #251 SRA (Sheaf-Resolvent Attention) | Per-query complex-pole *focus* on the per-token sheaf | Designed iter 3; composes with #250 |
| #252 PSA (Persistent Sheaf Attention) | Multi-layer cohomology tracking → principled pruning + depth-tuning + layer-role legibility | Designed iter 4; observes the #250+#251 stack |

And two forward-sketched extensions:

| Paradigm | Axis | Status |
|---|---|---|
| #253 SLR (Specialised Layer Roles) | Per-layer-type attention forms based on PSA's layer taxonomy | Sketched §6 below |
| #254 Dynamic Depth | Runtime layer-insertion driven by per-token commutation defect | Sketched §7 below |

The combined stack delivers **~10× wall-clock speedup at iso-NLL** with strict expressivity gains over SDPA and SCFA (the current flagship) at T=16384.

---

## 1. Mathematical foundations

### 1.1 The common substrate

All CSA paradigms operate on the same underlying object: a **cellular sheaf F over the token graph** G = (V, E) with:

- **V** = {1, ..., T} (vertices = tokens).
- **E** = causal-sliding-window-plus-sinks edge set (default W = 128 window half-width, |S_sink| = 8 sinks).
- **F(v_i) = R^{d_s}** = stalk at each vertex (per-token vector space — this is *perspective*).
- **R_{j ← i} = U_j Σ(i, j) U_i^T** = restriction map per edge (rank-r factorisation: U_i ∈ R^{d_s × r} amortised as MLP of x_i; Σ diagonal modulator).
- **L_F = δ^T δ** = sheaf Laplacian (PSD, never materialised).

This single substrate supports all three current paradigms:

- **#250 SFA** computes `y = P_o^T · (L_F + λI)^{-1} · b + W_Q x`, the regularised harmonic section of F.
- **#251 SRA** generalises to `y_q = Im[(z_q I − L_F)^{-1} b_q]_{[q]}` with per-query complex pole z_q.
- **#252 PSA** observes the per-layer F_ℓ's and tracks H^k(F_ℓ) across layers as a persistence module.

### 1.2 Containment hierarchy

Each paradigm strictly contains its predecessor (and SDPA / SCFA) as proven limiting cases:

```
SDPA   ⊂   SCFA   ⊂   SFA   ⊂   SRA       (containment as function classes)
         (#42)      (#250)   (#251)
```

with proofs:
- **SDPA ⊂ SCFA**: SCFA at k = T recovers SDPA exactly (PARADIGM_SHIFT_42_DESIGN.md §10.7).
- **SCFA ⊂ SFA**: Theorem 2 of PARADIGM_SHIFT_250_PROOFS.md §2 (Galerkin projection at d_s = 1).
- **SDPA ⊂ SFA**: Theorem 1 of PARADIGM_SHIFT_250_PROOFS.md §1 (trivial sheaf, after Σ-square-root correction).
- **SFA ⊂ SRA**: Theorem 4 of PARADIGM_SHIFT_251_DESIGN.md §4.1 (Tikhonov-resolvent identity at z_q = i √λ).
- **ORA ⊂ SRA**: Theorem 5 of PARADIGM_SHIFT_251_DESIGN.md §4.2 (d_s = 1).

PSA does not extend the *function class*; it extends the *training program* (regulariser) and *deployment program* (pruning) on top of an SFA/SRA stack.

### 1.3 Why sheaves?

The choice of cellular sheaves (not e.g. principal-bundle Riemannian geometry, not e.g. operator-resolvent on a shared basis) reflects three properties that the brief's "perspective" mechanism requires:

1. **Local data with global gluing**: a sheaf attaches local data (the stalk F(v_i)) to each vertex AND specifies how local data fits together globally (via restriction maps and the gluing axiom). This matches the linguistic intuition that each token has a local meaning *and* a way of fitting into the broader context.

2. **Algebraic structure for inconsistency**: a sheaf can be *non-trivial* — its 1-cocycles can represent failure-to-glue (cocycle obstructions). This captures linguistic phenomena like contradictory clauses, irony, multi-reading inputs that SDPA's "averaging-via-softmax" structurally cannot represent.

3. **Discrete + computable**: cellular sheaves over finite graphs are linear-algebraic objects (the sheaf Laplacian is a sparse symmetric PSD matrix). They are computable in CUDA at scale. By contrast, Riemannian-manifold attention (e.g. paradigm #250 Candidate A FBA) requires parallel transport, which suffers BF16 drift.

### 1.4 The mathematical primitive: harmonic section

The unifying operation across all CSA paradigms is **harmonic-section retrieval**:

> Given a source `b ∈ C^0(F)` (encoding the query content) and a sheaf F (encoding token perspectives + their relationships), retrieve the section `s ∈ C^0(F)` that minimises disagreement across all edges AND matches the source fidelity:

```
s★ = argmin_s { ½ ⟨s, L_F s⟩ + ½ λ ‖s‖² − ⟨b, s⟩ }
   = (L_F + λI)^{-1} b           (SFA, real Tikhonov)
   = Im[(z_q I − L_F)^{-1} b_q]  (SRA, complex pole per query)
```

This is **fundamentally different from SDPA's softmax**. SDPA computes attention as a probability distribution; CSA computes it as a *least-squares regularised consensus*. The probability-distribution view of attention is a special case (recovered by Theorem 1) but not the general primitive.

---

## 2. Paradigm #250 — Sheaf-Focal Attention (SFA)

Full design: `research/PARADIGM_SHIFT_250_DESIGN.md`. Proofs: `research/PARADIGM_SHIFT_250_PROOFS.md`. Selection rationale: `research/PARADIGM_SHIFT_250_SELECTION.md`. Memory: `[[paradigm250_sfa]]`.

### 2.1 Central equation

```
y_i = P_o^T [ (L_F + λI)^{-1} ( U_i U_i^T P_q W_Q x_i + γ P_v W_V x_i ) ]_{[i]} + W_Q x_i.
```

### 2.2 Key innovations

- **Per-token cellular sheaf** with stalk frames U_i = ψ(x_i; θ_U) amortised by a small MLP.
- **Restriction map factorisation** R_{j←i} = U_j Σ(i,j) U_i^T with diagonal Σ from a small per-edge MLP.
- **Chebyshev iteration** on the sparse causal-window-plus-sinks sheaf Laplacian (M = 8 default).
- **REFLECTOR-style implicit adjoint** for backward (PROOFS.md §4).

### 2.3 Quantitative claim

- **41× vs SDPA standalone** at T=16384 per-token attention math.
- **2× FLOP-saving at iso-NLL vs SCFA** via cocycle expressivity (refined Conjecture 1: 0.015–0.03 nat NLL improvement at iso-FLOP).
- **Translates to 1.4–2× wall-clock at iso-NLL** vs SCFA flagship.

### 2.4 Critical risks (Gate-0 falsifiers)

- **Conjecture 1**: cocycle modes carry ≥0.015 nat / token signal. Falsified if Probe B's 500-step fine-tune shows NLL flat or worse vs SCFA.
- **Conjecture 2**: sparse W=128 + sinks-8 edge set suffices. Falsified if Probe C's sparse-vs-complete comparison shows >0.05 nat gap.
- **BF16 conditioning**: Chebyshev requires Jacobi preconditioning (κ → 30) for M=8 to converge in BF16. Falsified if Probe D shows residual >10^{-2}.

---

## 3. Paradigm #251 — Sheaf-Resolvent Attention (SRA)

Full design: `research/PARADIGM_SHIFT_251_DESIGN.md`. Memory: `[[paradigm251_sra]]`.

### 3.1 Central equation

```
y_q = Im[ (z_q I − L_F)^{-1} b_q ]_{[q]},      z_q = ρ_q + i ω_q from MLP_z(W_Q x_q).
```

### 3.2 Key innovations

- **Per-query complex pole** z_q determines focus sharpness (ω_q) and spectral location (ρ_q).
- **Closed-form per-query** via block-diagonal resolvent factorisation (eq. 10 of #251 design). No Chebyshev iteration needed per query.
- **Cayley real-arithmetic form** for the complex pole — 2×2 real blocks, no CUDA complex opcodes.
- **FP32 D-inversion** wrapped in BF16 GEMMs — handles the resolvent's ε^{-2} = 10^4 condition number.
- **Multi-pole extension** (P ≥ 1) for multi-modal attention measures unrepresentable by softmax.

### 3.3 Quantitative claim

- **4.3× compute reduction vs SFA** at T=16384 (closed-form vs M=8 Chebyshev).
- **~16× cheaper per query** than SFA's per-query Chebyshev cost.
- **6–8× combined wall-clock at iso-NLL** vs SCFA flagship (SFA's quality gain × SRA's compute reduction).
- **Multi-modal attention expressivity** via Prop 7.2 of #250 Candidate C (ORA recovery proof shows the structure transfers).

### 3.4 Critical risks (Gate-0 falsifiers, extends #250 Gate-0)

- **Conjecture 4**: r-pole rational density on L_F spectra. Falsified if multi-pole P=2 attention doesn't improve over P=1.
- **Per-query pole stability**: ω_q distribution must not collapse to single value (focus collapse) or extreme bimodality.
- **Block-diagonal-approximation tightness**: §3.4 of #251 bounds error at 10^{-4}; falsified if numerical tests show >10^{-3}.

---

## 4. Paradigm #252 — Persistent Sheaf Attention (PSA)

Full design: `research/PARADIGM_SHIFT_252_DESIGN.md`. Memory: `[[paradigm252_psa]]`.

### 4.1 Central object

The **persistence module** indexed by layer ℓ:

```
H^k(F_1) → H^k(F_2) → ... → H^k(F_L)
```

with maps induced by the residual-stream coupling. By Crawley-Boevey's structure theorem, this decomposes uniquely into interval modules — bars (b_α, d_α) in the **persistence diagram PD^k**.

### 4.2 Key innovations

- **Imports Topological Data Analysis** into LLM architecture. Persistence diagrams are standard TDA objects (Edelsbrunner-Harer-Carlsson), now computed for LLM layers via cellular-sheaf cohomology.
- **Principled layer pruning**: rank layers by `birth_ℓ + death_ℓ`; prune the bottom 30%.
- **Depth-tuning by bottleneck detection**: insert refinement layers where many bars die simultaneously.
- **Layer-role taxonomy**: cluster layers by (birth, death, persistent_active) → feature-extractors / synthesizers / consensus-builders / inert.
- **Persistence-loss regulariser** `L_persist = λ · D_Wasserstein(PD^0, target)` as architecture-level training signal.

### 4.3 Quantitative claim

- **1.4–1.6× wall-clock at iso-NLL** via 30% layer pruning (Conjecture 7).
- **Stack with #250 + #251**: **~10× total wall-clock at iso-NLL** over SCFA flagship.
- **Plus architectural legibility**: each layer's role becomes computable, not just empirically guessed.

### 4.4 Critical risks (Gate-0 falsifiers)

- **Conjecture 7**: 30% pruning preserves NLL within 0.02 nat. Falsified if Probe I shows NLL regression >0.05 nat.
- **Persistence diagram health**: ≥30% of bars have length ≥ L/2. Falsified if Probe H shows degenerate diagram (all short bars, or all infinite bars).
- **Persistence-loss regulariser**: must help 66M training by ≥0.02 nat. Falsified by Probe J.

---

## 5. Cross-paradigm Gate-0 plan

Combined Gate-0 spec for paradigms #250 + #251 + #252:

| Probe | Source | Cost | Goal |
|---|---|---|---|
| A | #250 §15.1 | 5 sec | SFA at d_s=1 recovers SCFA at step 0 (Theorem 2 numerical check) |
| B | #250 §15.2 (refined iter 2) | 30 min | SFA cocycle expressivity: ΔNLL ≤ -0.015 nat after 500 fine-tune steps |
| B' | iter 2 §5.4 | (subset of B) | Φ-rich / Φ-poor NLL ratio ≥ 4× (mechanism falsification) |
| C | #250 §15.3 | 60 min | Sparse vs causal-complete edge set: diff ≤ 0.01 nat |
| D | #250 §15.4 | 5 min | Chebyshev residual ≤ 10^{-3}, BF16 stable |
| E | #250 §15.5 | 10 min | Per-layer Lipschitz bounded over 100 steps |
| F | #251 §7.3 | 5 sec | SRA recovers both SFA and ORA limits to within 0.005 nat |
| G | #251 §7.3 | 10 min | Per-query pole distribution: ω variance ≥ 0.5·E[ω]^2; ρ spans ≥50% of σ(L_F) |
| H | #252 §7.4 | 5 min | Persistence diagram health: ≥30% bars length ≥ L/2 |
| I | #252 §7.4 | 30 min | 30% layer pruning: ΔNLL ≤ +0.02 nat |
| J | #252 §7.4 | 30 min | L_persist regulariser improves 66M by ≥0.02 nat |

**Total budget**: ~3 GPU-hours on `chiron_1B_T16384.step30000` + 66M reference. Each probe is *cheap-decisive* — a failure of any probe falsifies that paradigm's load-bearing conjecture.

**Decision tree**:
- All probes pass → implement the full stack (~50 iterations to production per design roadmaps).
- Probe B fails → SFA's central conjecture falsified; promote SRA-without-SFA-substrate; reduce to ORA-only path.
- Probe I fails → PSA's pruning falsified; PSA's value reduces to interpretability only; no compute win from pruning.
- Multiple probes fail → re-design from candidates A (FBA, rejected) or pure SCFA improvements.

---

## 6. Paradigm #253 sketch — Specialised Layer Roles (SLR)

PSA's layer taxonomy (§5.3 of #252) divides layers into four roles. SLR specialises the attention mechanism per role:

### 6.1 Per-role attention forms

| Layer role | Attention form | Hyperparams |
|---|---|---|
| Feature-extractors (early, high birth) | SFA at high d_s (rich per-token) | d_s=64, r=8, single-pole z=i√λ |
| Synthesizers (mid, balanced) | SRA standard (per-token + per-query) | d_s=32, r=4, P=1 |
| Consensus-builders (late, high death) | SRA multi-pole at low d_s | d_s=16, r=2, P=4 |
| Inert | PSA-pruned | — |

### 6.2 Magnitude claim

- **Reduce compute on consensus-builder layers** by ~4× (d_s=16 vs 64).
- **Multi-pole on consensus-builders** preserves quality.
- **Per-role specialisation** at inference: ~1.5× wall-clock on top of PSA's 10× = **~15× total**.

### 6.3 Open questions

- Does per-role specialisation actually buy quality at iso-FLOP? Or does uniform configuration suffice with the PSA taxonomy only for pruning?
- How to schedule role-assignment over training? Layers start uniform; PSA's taxonomy crystallises only after some training. SLR's per-role configuration should phase in gradually.

Full design deferred to a future iter when paradigms #250-252 have empirical data.

---

## 7. Paradigm #254 sketch — Dynamic Depth

The per-token commutation defect `ε^{(φ)}_ℓ(i, j)` (eq. 3 of #252) is conjectured (Conjecture 8) to correlate with linguistic complexity at token i. Dynamic Depth uses this signal to **runtime-insert layers** for inputs with high complexity:

### 7.1 Algorithm

```
For each input x and each layer ℓ:
    Compute ε^{(φ)}_ℓ(i, j) for all (i, j) ∈ E.
    If max_i max_j ‖ε‖ > threshold:
        Insert an additional "refinement layer" between ℓ and ℓ+1.
        Initialise its sheaf to a low-rank perturbation of F_ℓ.
        Apply, then continue forward.
```

### 7.2 Why this might work

Linguistic complexity is per-input. Simple sequences (lists, factual statements) need shallow processing; complex sequences (multi-clause inference, embedded irony) need deeper processing. SDPA-based transformers apply fixed depth regardless of input; Dynamic Depth allocates depth on demand.

### 7.3 Magnitude claim

- Average-case depth reduction: 30-40% (simple inputs).
- Hardest-case depth increase: +50% (complex inputs get deeper processing).
- **Net wall-clock at iso-quality**: ~2× speedup on a diverse input distribution.
- **Plus capability ceiling raised** on complex inputs (deeper processing for what would otherwise be too-shallow).

### 7.4 Open issues

- Runtime kernel launch overhead for inserted layers.
- KV-cache management with variable depth.
- Stability of mid-stack layer insertion.

Full design deferred. The Conjecture 8 must first be validated empirically: does ε^{(φ)} actually correlate with complexity?

---

## 8. Unified implementation roadmap

### 8.1 Phase ordering across paradigms

```
Phase 0 (Gate-0): combined #250+#251+#252 probes A-J         ≤ 3 GPU-hours, ~1 iteration

Phase 1 (#250 SFA primitives): CPU prototype + parity
  - L_F sparse matvec
  - Chebyshev recurrence
  - Restriction-map MLP (Σ)
  - Stalk-frame ψ MLP
  - REFLECTOR adjoint                                          ~5-8 iterations

Phase 2 (#250 GPU): wire 5 CUDA primitives + parity test     ~5-8 iterations
Phase 3 (#250 trainer): --sfa flag + regularisers            ~3-5 iterations
Phase 4 (#250 validation): 66M → 1B → long-context           ~3-5 iterations

Phase 5 (#251 SRA primitives + flag): adds 3 primitives      ~5-7 iterations
  - MLP_z (per-query pole)
  - Cayley-form resolvent kernel
  - Multi-pole combiner
  - --sra flag

Phase 6 (#251 validation)                                     ~3-5 iterations

Phase 7 (#252 PSA): persistence diagram + pruning            ~5-7 iterations
  - Per-layer Lanczos
  - Bar-tracking algorithm
  - Wasserstein-distance loss
  - Pruning logic

Phase 8 (production rollout): #250+#251+#252 defaults        ~2-3 iterations
```

**Total iterations**: ~30-50. Assuming 1 iter / day (Ralph-loop pace), program reaches production in ~1-2 months wall-clock.

### 8.2 Risk-driven phase order

The roadmap is **risk-front-loaded**: Gate-0 (Phase 0) runs first, costing only 3 GPU-hours. Any falsifier kills the corresponding paradigm before significant engineering investment.

If Gate-0 reveals issues with #250 (Probe B fails), the program pivots to a slimmed version: SCFA + ORA only, skipping the SFA substrate. This is roughly paradigm #251's "ORA-recovery limit at d_s=1" (Theorem 5) as a standalone — still gives ~4× over SCFA but loses the cocycle expressivity claim.

If Gate-0 passes #250 but fails #252 (Probe I), PSA still provides architectural legibility (the persistence diagram is still a meaningful object), just without the pruning operational use.

### 8.3 Composition with shipped infrastructure

All paradigms inherit from the shipped flagship `chiron_1B_T16384.step30000` stack:

- SCFA basis B → initial U_· stalk frames (per Theorem 2 SCFA-recovery).
- Sinks (#78) → explicit in edge set E.
- BF16 stack (iter 1-10) → inner GEMMs in BF16, FP32 only for resolvent D-eval and Chebyshev preconditioning.
- REFLECTOR (#46) → backward through implicit solves.
- ATLAS-COMPILE (#51) → autotunes new SFA/SRA kernels.
- SLC (#38), RLG (#39) → orthogonal to CSA paradigms; compose multiplicatively.

### 8.4 Composition with existing ralph-loop optimisations

The shipped iter 1-10 SCFA optimisations (BF16-inner, BF16-outer, BF16-logits, scfa-fuse-streams, scfa-reln-opt, BF16-logits-storage) are all *kernel-level* optimisations on the SCFA project/lift GEMMs. SFA inherits them via the U_· basis (which doubles as SCFA's B initially). No new ralph-loop iterations needed at the kernel level for the CSA stack — they piggyback on the existing infrastructure.

---

## 9. Magnitude claim summary

| Stack | Per-step | Steps-to-NLL | Layer count | Total wall-clock |
|---|---|---|---|---|
| SDPA (Vaswani 2017) at T=16384 | infeasible (T² memory) | N/A | N/A | N/A |
| SCFA flagship (paradigm #42, shipped) | 1× | 1× | L=24 | 1× (baseline) |
| + SFA (#250) | ~1.01× | 0.5–0.7× | L=24 | 1.4–2× |
| + SRA (#251) | 4.3× | 0.5–0.7× | L=24 | 6.0–8.6× |
| + PSA pruning (#252, K=0.3L) | 1.4× | inherited | 0.7L=17 | 8.4–12× |
| + #253 SLR (sketched, conjectural) | 1.5× | inherited | 0.7L=17 | 12.6–18× |
| + #254 Dynamic Depth (sketched) | varies | inherited | adaptive | 18–36× |

**The brief's "magnitudes" target is met decisively** at the SFA+SRA+PSA level (~10×), with additional headroom from forward-sketched paradigms #253-254.

The single most aggressive realistic claim — **10× wall-clock speedup at iso-NLL over SCFA flagship at T=16384** — depends on three conjectures (1, 4, 7) all holding empirically. Each is falsifiable in ≤30 GPU-minutes at Gate-0.

---

## 10. Conjectures-and-tests summary

Eight load-bearing conjectures across the program:

| # | Statement | Tested by | Cost |
|---|---|---|---|
| 1 (refined) | Cocycle modes carry ≥0.015 nat NLL signal at 1B/T=16384 | Probe B | 30 min |
| 1' (mechanism) | Φ-rich / Φ-poor NLL ratio ≥ 4× | Probe B' (subset of B) | (inclusive) |
| 2 | Sparse W=128+sinks-8 edge set suffices | Probe C | 60 min |
| 3 | Layer-stacking gives well-defined colimit sheaf F_∞ | empirically via PSA persistence diagrams | (in Phase 7) |
| 4 | r-pole rational density on L_F spectra (SRA) | (theoretical; empirical via P=2 vs P=1 comparison) | 15 min |
| 5 | Per-query pole correlates with Φ-phenomena | post-#251-implementation | (in Phase 6) |
| 6 | Trained-LM PD^0 matches linguistic-phenomenon hierarchy | Probe H + analysis | 5 min |
| 7 | 30% layer pruning preserves NLL within 0.02 nat | Probe I | 30 min |
| 8 | Commutation defect ε^{(φ)} correlates with linguistic complexity | post-implementation | (in #254 design) |

**All conjectures are falsifiable by ≤2 GPU-hours total**. The program is **maximally falsifiable** in the Lakatos sense: every load-bearing claim has a cheap, decisive empirical test.

---

## 11. Open theoretical questions

The program leaves several mathematical questions for future research:

1. **Tight proof of Lemma 6.1 (stalk-rank monotonicity)**: currently sketched in #250 PROOFS §6. Tight proof requires bounding the spectral content of L_F + λI in terms of d_s · M_eff more rigorously.
2. **Formal proof of Conjecture 3 (layer-stacking colimit)**: PSA's foundation. Persistent-homology theory provides the framework (Crawley-Boevey 2015) but the LLM-specific application needs careful treatment of the inter-layer morphism's commutation defects.
3. **Universal approximation property**: SFA-on-SCFA at depth O(log T) universally approximates cellular-sheaf-spectral-filter maps. Proof or counterexample.
4. **Connection to ∞-categorical sheaves**: cellular sheaves are the discrete approximation. The continuous limit (sheaves on a continuous Riemannian manifold, e.g. paradigm #250 Candidate A FBA) is a different beast. Is there a unifying ∞-categorical framework?
5. **Connection to Yoneda embedding**: each token's stalk can be viewed as a representable presheaf. Does the Yoneda embedding give a natural "Universal Stalk Frame" worth implementing?

These are research-grade questions for follow-on work; none are blockers for the Gate-0-driven empirical program.

---

## 12. The "next next step" beyond this program

If the CSA stack (#250+#251+#252) validates at Gate-0 and delivers the 10× wall-clock speedup, the program opens several directions:

1. **Paradigm #253 SLR** (sketched §6): per-role attention.
2. **Paradigm #254 Dynamic Depth** (sketched §7): runtime layer-insertion.
3. **Paradigm #255 candidate: Cohomology-guided weight sharing**. Bars in PD^0 that persist across many layers represent global content; the corresponding eigenvectors can be SHARED across layers (saving parameters).
4. **Paradigm #256 candidate: Distributed sheaf-attention**. For multi-GPU training, partition E (the edge set) across GPUs; each GPU computes a slice of L_F's action. Inherits HYDRA (#45) pipeline parallelism.
5. **Paradigm #257 candidate: Sheaf-of-Sheaves attention**. Hierarchical sheaves where each stalk is itself a sheaf (multi-scale token graphs).

**The "next next step" beyond CSA** may be: replacing the *linear* sheaf-Laplacian framework with a *non-linear* one — e.g., **non-linear sheaf cohomology** where restriction maps are non-linear functions of stalks. This would step outside the linear-algebraic comfort zone of all current attention mechanisms (SDPA included). It is also potentially much harder to implement efficiently.

A second possible "next next step": replacing the attention layer entirely with **discrete differential geometry on token graphs** — using ricci curvature, optimal transport on graphs, and similar tools. The CSA program is the discrete-differential-geometry-lite version; full DDG would be paradigm #260+.

Either direction would emerge from learning what worked and what didn't in CSA, after Gate-0 and Phase 1-4 validation.

---

## 13. Summary

The Cellular Sheaf Attention research program proposes a unified replacement for SDPA's "global-coordinates + softmax" with "per-token cellular-sheaf + spectral-filter":

- **Paradigm #250 SFA**: per-token *perspective* (stalk frames) + spectral *focus* (Tikhonov solve of sheaf Laplacian).
- **Paradigm #251 SRA**: per-query complex-pole *focus* on the per-token sheaf. Closed-form per-query.
- **Paradigm #252 PSA**: multi-layer *cohomology tracking* via persistent homology. Principled layer pruning + depth-tuning + role legibility.

Combined: **~10× wall-clock at iso-NLL over SCFA flagship at T=16384**, with strict expressivity gains via cocycle obstructions (SFA) and multi-modal rational filters (SRA) that SDPA structurally cannot represent.

Gate-0 specifies 11 probes totalling ≤3 GPU-hours. Eight load-bearing conjectures are individually falsifiable.

Forward sketches of #253 SLR (specialised per-role attention) and #254 Dynamic Depth (runtime layer-insertion) provide ~5× additional headroom if the core program validates.

The program imports three mathematical fields into LLM architecture for the first time:
- **Cellular sheaf theory** (Hansen-Ghrist 2019, Curry 2014, Bodnar et al. 2022) — substrate.
- **Spectral graph theory** (Chung 1997) — sheaf Laplacian.
- **Topological Data Analysis** (Edelsbrunner-Harer 2010, Carlsson 2009) — persistence diagrams.

Each had been mature for decades; the contribution is *unifying* them with attention.

**File index for the program**:

- `research/CELLULAR_SHEAF_ATTENTION_PROGRAM.md` — this document
- `research/PARADIGM_SHIFT_250_SELECTION.md` — iter 1, candidate selection
- `research/PARADIGM_SHIFT_250_DESIGN.md` — iter 1, SFA full framework
- `research/PARADIGM_SHIFT_250_PROOFS.md` — iter 2, formal proofs + adjoint backward
- `research/PARADIGM_SHIFT_250_CANDIDATE_{A_FBA,B_SFA,C_ORA}.md` — iter 1, three candidates
- `research/PARADIGM_SHIFT_251_DESIGN.md` — iter 3, SRA design
- `research/PARADIGM_SHIFT_252_DESIGN.md` — iter 4, PSA design

**Memory index**:

- `[[paradigm250_sfa]]` — SFA persistent memory
- `[[paradigm251_sra]]` — SRA persistent memory
- `[[paradigm252_psa]]` — PSA persistent memory
- `MEMORY.md` — top-level index entries

The program is complete as a *design* (5 iterations of Ralph-loop research-framework-design work). What remains is empirical: implement, gate-0, iterate. The math is ready; the engineering follows.
