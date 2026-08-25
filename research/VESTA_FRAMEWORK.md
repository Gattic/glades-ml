# VESTA — Selected Framework: GRP-RNN

**Date:** 2026-05-19
**Status:** Selected candidate (A of 3). Candidate documents at `research/VESTA_CANDIDATE_{A,B,C}.md`.
**Audit:** `research/VESTA_AUDIT.md` (mandatory pre-work).
**Brief:** `newmodel.txt`.

---

## 0. One-paragraph summary

VESTA selects **GRP-RNN** (Group-Rotation Product Recurrent Network, Candidate A) as the framework for the inventive phase. The state evolves under a per-step rotation `R_t = ∏_{k=1}^K G_k(θ_k(x_t)) ∈ SO(m)`, the product of K input-dependent Givens rotations on pre-chosen interlocking planes. With disjoint planes, this reduces to LRU. With one plane and a fixed rotation angle of π, it reduces to β=1 DeltaNet. With interlocking planes and K ≥ 3, the generated subgroup of SO(m) is *non-abelian* and contains non-solvable finite groups (notably A_5 ⊂ SO(3) ⊂ SO(m)) — giving GRP-RNN a *provable* expressivity advantage over diagonal SSMs (Merrill, Petty, Sabharwal 2024). The framework has one mechanism, one loss term (next-token CE), two ablation-only regularizers, and three pre-committed falsifiable claims. Implementation cost is ~1630 LOC on top of the existing `glades-ml` infrastructure — the smallest of the three candidates.

---

## 1. Selection rationale (against Candidates B, C)

The brief (`newmodel.txt`) demands three properties of the selected framework:

1. **Adversarial isolation**: every mechanism must beat the strongest existing literature baseline on its axis.
2. **Minimal decisive experiments**: prefer one clean falsifiable claim over a stack of inter-dependent claims.
3. **Reductions to known methods**: any mechanism that overlaps with an existing method must either subsume it as a special case or be discarded.

Applied to the three candidates:

| Criterion | A: GRP-RNN | B: MuRe | C: PULSAR |
|---|---|---|---|
| Strongest adversarial baseline | DeltaNet on A_5 (provably the cleanest discriminator) | Based on MQAR | Mixture-of-Depths on T7 |
| Theoretical novelty backing | Group-theoretic theorem: solvable→non-solvable transition | Information-theoretic IB + Bayesian filter (50-yr-old primitives) | Endogenous surprisal residual (genuinely novel) |
| Single claim or stack | One mechanism → one claim (N1) | One mechanism → two claims (N1+N2 attack 2 axes) | One mechanism → unified 3-axis attack |
| Reduction matrix completeness | Subsumes LRU, DeltaNet, Mamba-2, MEGA-EMA explicitly | Subsumes LRU at k=1; new R2 task not in literature | Subsumes LRU, MoD, MambaByte |
| Bootstrap-circularity surfaces | 0 (no gates) | 1 (resampling can stall via ESS collapse) | 1 (event detector; the framework's central bet) |
| C++ cost (LOC, weeks) | 1630, 2 weeks | 2200, 3 weeks | 4250, 6 weeks |
| Failure-of-central-claim → publishable negative | Yes (clean: K>1 doesn't help → reduces to DeltaNet) | Yes (particle degeneracy = known SMC pathology) | Yes (gate fails like EALRMN Phase-0f — but this would *replicate* the prior negative) |
| Time to first decisive experiment | ~4 days | ~7 days | ~14 days (CPU prototype) |

**Decisive factors:**

1. **Theorem-grounded falsifiability.** GRP-RNN's central claim has a *theoretical* answer (Merrill et al. 2024 §3) for what the diagonal-SSM baseline *cannot* do. The empirical question is whether the K>1 lifting realizes that advantage. This is the only one of the three candidates whose claim is decisively informative either way: if A_5 is solved → expressivity earned; if A_5 fails → optimization can't realize the proven theoretical gap; either result publishable.

2. **Single mechanism, single gate-free.** Per the brief's E1 critique, EALRMN's failure was conjoining 8 mechanisms before isolation. GRP-RNN has no gates, no auxiliary objectives, no bootstrap surfaces. The L_rot-reg and L_decay-reg terms are ablation-only — primary runs are CE-only.

3. **Implementation budget.** With single-developer engineering, ~1630 LOC on existing infrastructure (`gpu_vesta.cu/.h` placeholders already exist; `gpu_stiefel.cu/.h` provides orthogonal-structure primitives) is feasible. PULSAR's ~4250 LOC + GPU sparse scatter is a strictly larger commitment that has no decisive answer until the CPU prototype completes — and the CPU prototype itself is a 3-week project.

4. **Reduction matrix.** GRP-RNN explicitly reduces to LRU under `--grp-disjoint-planes`, to β=1 DeltaNet under `--grp-K=1`, to MEGA-EMA under `--grp-K=0`, and (via the deliberately-broken `--grp-tanh-state`) to the tanh-RNN strawman that drives Claim B0. Every flag flip is an isolation experiment.

**Insights borrowed from rejected candidates (parked as future work):**

- **B's cross-stream future-statistic predictor (L_ψ)** structurally breaks Phase-0a's bootstrap circularity. If a future VESTA phase re-attacks R2, this construction is the right starting point. *Parked for Phase 4+.*
- **C's endogenous-surprisal-residual detector** (KL of default-stream prediction against EMA of full-model prediction) is the most novel idea in either rejected candidate. If a future phase requires any kind of gate, this supervision signal is the right starting point. *Parked for Phase 5+.*

These are *not* integrated into GRP-RNN. The brief explicitly warns against mechanism conjunction before isolation (E1). The MuRe paired-stream construction and the PULSAR endogenous-surprisal idea are independent contributions of the design exercise and may be tested as *separate* future programs.

---

## 2. Formal problem statement

We are given:

- Sequence model class: maps x_{1:T} ∈ V^T to logits over V.
- Compute budget: O(T · m · log m) per layer or better (linear-recurrence class).
- Memory budget: O(m) state per layer (no KV cache).
- Existing strongest baselines for recurrence type (R3): Mamba-2, Hawk, LRU. Existing strongest non-diagonal baseline: DeltaNet.

We pose:

**Q (operator-completeness).** What is the smallest structured non-diagonal recurrence operator family that (a) provably escapes the diagonal-SSM expressivity ceiling (Merrill et al. 2024) and (b) admits a parallel-scan training algorithm within 3× of Mamba-2's throughput at T=2048, m=1024?

GRP-RNN proposes: a per-step product of K Givens rotations on interlocking planes, with input-dependent angles, is sufficient.

---

## 3. Core mathematical framework

### 3.1 State, parameters, dynamics

**State.** s_t ∈ ℝ^m. The state lives in the ambient vector space, *not* on a manifold representation; the orthogonality lives in the operator.

**Parameters.** For k = 1, …, K:
- Plane indices (p_k, q_k) ∈ {0, …, m−1}² with p_k ≠ q_k. Fixed at init. Pre-chosen via the **interlocking stride** rule (p_k, q_k) = (2k − 2 mod m, (2k − 2 + 2s) mod m) with gcd(s, m/2) = 1.
- Angle parameters w_k ∈ ℝ^d, b_k ∈ ℝ. Learnable.
- Input projection B ∈ ℝ^{m×d}, readout C ∈ ℝ^{V×m}. Learnable.
- Scalar decay λ ∈ (0, 1], parameterized as λ = exp(−exp(ν)) (LRU-style) with ν ∈ ℝ. Learnable. Default λ = 1 (pure orthogonal).
- Bounded-angle scalar φ_max ∈ (0, π). Default φ_max = π/2. Learnable but typically frozen.

**Per-step rotation.**
$$
G_k(θ) = I + (\cos θ − 1) (E_{p_k p_k} + E_{q_k q_k}) − \sin θ (E_{p_k q_k} − E_{q_k p_k})
$$
where E_{ij} is the matrix unit at row i, column j. G_k(θ) is the identity except in the 2×2 block (p_k, q_k) where it is `[[cos θ, −sin θ], [sin θ, cos θ]]`.

**Angle map.**
$$
θ_k(x_t) = \tanh(w_k^T x_t + b_k) \cdot φ_\max
$$
The tanh constrains θ_k to (−φ_max, +φ_max). At init w_k ~ N(0, 1/√d), b_k = 0, so θ_k(x_t) ≈ 0 — the per-step rotation is near-identity, giving an automatic "orthogonal init" prescription matching the EALRMN Phase-1 finding.

**Per-step transition operator.**
$$
R_t = G_1(θ_1(x_t)) · G_2(θ_2(x_t)) · ... · G_K(θ_K(x_t)) ∈ \text{SO}(m)
$$
Order matters: K Givens with shared coordinates do not commute. We fix the order k = 1, 2, …, K.

**Evolution law.**
$$
s_t = λ · R_t · s_{t-1} + B x_t
$$
$$
y_t = C · s_t
$$

Note: the recurrence is **linear in s_{t−1} given x_t** — there is no nonlinearity-in-state. The nonlinearity sits in how x_t parameterizes R_t (the tanh on θ_k). Vanishing-gradient pathologies of tanh-RNN do not apply.

### 3.2 Continuous-time interpretation

With τ = t · Δt, Δt → 0:
$$
\frac{ds}{dτ} = (−μ I + Ω(x(τ))) s(τ) + B x(τ)
$$
where μ ∈ ℝ_+ corresponds to the scalar decay rate (μΔt = 1 − λ) and Ω(x) = Σ_k θ̇_k(x) E_k, with E_k ∈ so(m) the antisymmetric generator of the k-th Givens (E_{p_k q_k} − E_{q_k p_k}).

The generator Ω lies in a fixed K-dimensional subspace of so(m), spanned by {E_1, …, E_K}. This makes GRP-RNN a *structured* non-diagonal recurrence: the recurrence operator is non-diagonal in any basis that mixes planes, but the parameterization has only K + 1 free parameters per step (the K angles plus the decay).

The discrete transition (3.1) is the **Trotter splitting** of exp(Ω(x_t) Δt) into a product of Givens. Splitting is exact when generators commute (disjoint planes); a first-order approximation otherwise.

### 3.3 Reachable operator set

Let H_K ⊂ SO(m) be the closed subgroup generated by {G_k(θ) : θ ∈ ℝ, k = 1, …, K}.

**Theorem 3.1 (operator-completeness).** Let `(V, E)` be the *plane graph* with vertex set V = ⋃_k {p_k, q_k} ⊆ {0, …, m−1} and edge set E = {{p_k, q_k} : k = 1, …, K}. If `(V, E)` is connected and has chromatic number ≥ 3, then H_K = SO(|V|), i.e., the closed subgroup generated by the K Givens generators is the full special orthogonal group on the touched coordinates.

**Proof sketch.** The Lie algebra of H_K is the Lie subalgebra of so(m) generated by {E_1, …, E_K} under repeated commutator. Two Givens generators on planes sharing exactly one coordinate (say, planes {a, b} and {b, c}) have commutator [E_{ab}, E_{bc}] = E_{ac} (up to sign), a third Givens generator on a new plane. By induction on the path length in the plane graph, the Lie algebra contains E_{ij} for every pair (i, j) with i, j ∈ V — i.e., all of so(|V|). Therefore H_K, the closed connected subgroup with this Lie algebra, equals SO(|V|). □

**Corollary 3.2 (interlocking-planes choice).** With (p_k, q_k) = (2k − 2 mod m, (2k − 2 + 2s) mod m), gcd(s, m/2) = 1, K = m/2, the plane graph is connected with all m vertices and chromatic number ≥ 3 (it's a cycle of length m). Therefore H_{m/2} = SO(m).

**Corollary 3.3 (non-solvable expressivity).** For any finite group G that embeds into SO(n) for some n ≤ m, GRP-RNN with K interlocking planes can realize a single-layer recurrence whose state trajectory follows a word in G. In particular, A_5 ⊂ SO(3) ⊂ SO(m) is realizable by GRP-RNN with K ≥ 3 in a single layer.

By Merrill, Petty, Sabharwal (arXiv 2404.08819) §3: a recurrent net whose state-transition group is solvable cannot recognize word-problems on non-solvable groups in O(1) depth. Diagonal SSMs (LRU, S4D, Mamba's scalar-time-dependent decay) have a solvable (abelian K-torus) state-transition group; GRP-RNN's H_K is non-solvable for K ≥ 3 with interlocking planes.

**This is the structural expressivity claim**: GRP-RNN is operator-complete in SO(m) under the interlocking-planes choice, and therefore strictly more expressive than diagonal SSMs on a well-defined task class.

### 3.4 Reduction to existing methods

Every existing baseline named in the audit is a strict parameter-restriction of GRP-RNN:

| Method | Restriction | Effect |
|---|---|---|
| **EMA / MEGA-core** | K = 0 (no rotation) | s_t = λ s_{t-1} + B x_t |
| **LRU (Orvieto 2023)** | disjoint planes + fixed angles | s_t = diag(λ exp(iθ_k)) s_{t-1} + B x_t |
| **Input-dependent LRU** | disjoint planes only | LRU with θ_k = θ_k(x_t) |
| **β=1 DeltaNet (Yang/Schlag 2024)** | K = 1, plane learned from (u_t, v_t) | DeltaNet with rotation angle π |
| **Mamba-2 SSD core** | R_t = I, scalar input-dep decay | s_t = exp(−Δ_t α) s_{t-1} + Δ_t B_t x_t |
| **tanh-RNN (strawman)** | apply tanh after s_t (`--grp-tanh-state`) | recovers the EALRMN-Phase-1 strawman |

Each restriction is a single flag (Section 6). Every reduction is implemented as an ablation cell in the test plan; running with the flag must produce numerically equivalent output to the reference implementation up to fp32 rounding.

---

## 4. Objective

Single loss with two ablation-only regularizers:

$$
\mathcal{L}_\text{total} = α · \mathcal{L}_\text{CE} + β · \mathcal{L}_\text{rot-reg} + γ · \mathcal{L}_\text{decay-reg}
$$

where α = 1 in all primary runs, and β = γ = 0 by default. The regularizers are present *only* to be turned on if optimization pathologies appear; their default is off so that any reported gain is from the architecture, not from auxiliary objective shaping.

$$
\mathcal{L}_\text{CE} = -\sum_t \log p_θ(x_t | x_{<t}) = -\sum_t \log \text{softmax}(y_t)_{x_t}
$$

$$
\mathcal{L}_\text{rot-reg} = \frac{1}{K} \sum_k |\tanh(w_k^T \bar{x} + b_k)| \cdot φ_\max
$$
(evaluated on batch-mean input x̄; *not* per-token; prevents global angle collapse without destroying per-token specialization).

$$
\mathcal{L}_\text{decay-reg} = -\log(λ)
$$
(used only if the decay parameter ν drifts toward identity; encourages a slight contraction).

The discipline this loss commits to: **no contrastive term, no reconstruction term, no information-bottleneck term, no expert-balance term, no compute term**. Each additional term in EALRMN's 7-term Lagrangian carried a bootstrap-circularity risk; GRP-RNN refuses the same trap.

---

## 5. Optimization algorithm

Standard AdamW with the existing glades-ml optimizer (`research/ealrmn_gpu/optimizer.cuh` reusable verbatim). Hyperparameters per Phase-1 GPU recipe:

- β₁ = 0.9, β₂ = 0.95, ε = 1e-8, weight_decay = 0.01
- LR = 3e-4 (m=256), 1e-4 (m≥512), with cosine decay + 200-step linear warmup
- Global gradient clipping at 1.0
- Mixed precision: fp32 state, bf16 weights, fp32 master copy in optimizer (existing infrastructure)

No new optimizer choice. No special learning-rate schedule. The Zoology / Eyuboglu observation that linear-RNN-class models can have narrow optimal-LR windows is addressed by running each cell at 3 LRs (lr/3, lr, 3·lr) and reporting the best.

---

## 6. Ablation matrix (per-flag isolation)

| Flag | Effect | Reduces GRP-RNN to | Tests |
|---|---|---|---|
| `--grp-K=0` | no rotation | EMA / MEGA-core | "is rotation needed at all" |
| `--grp-disjoint-planes` | (p_k, q_k) = (2k−2, 2k−1) | input-dep LRU | "is non-diagonality needed" |
| `--grp-fixed-angles` | θ_k(x) ≡ θ_k (no input dep) | input-indep LRU | "is input-dependent angle needed" |
| `--grp-K=1 --grp-plane-input-dep` | one plane, plane learned per step | β=1 DeltaNet | "does K>1 help over K=1" |
| `--grp-no-decay` | Λ = I always | pure orthogonal | B0 ablation cell |
| `--grp-tanh-state` | apply tanh after linear update | tanh-RNN strawman | B0 verification (must fail) |
| `--grp-rot-reg=0` | drop L_rot-reg | α-only | default (primary runs) |
| `--grp-decay-reg=0` | drop L_decay-reg | α-only | default (primary runs) |
| `--grp-stride=s` | interlocking stride parameter | (1 = disjoint, 3 = default) | "is stride choice load-bearing" |

A1 (linear vs nonlinear), A2 (orthogonal vs Xavier), A4 (mechanism present vs removed) from the brief's mandatory ablation matrix are all expressible as single-flag swaps above.

---

## 7. Three pre-committed claims (full development in `VESTA_CLAIMS.md`)

The framework's empirical contribution rests on three falsifiable claims. The full pre-commit interpretation tables live in the claims doc; this section summarizes.

- **B0 (infrastructure replication).** Linear-recurrence + orthogonal init beats tanh-RNN by ≥100× val_loss on needle T=2048 m=1024 (pre-registered per `newmodel.txt`). Already validated by EALRMN Phase-1 ablation; VESTA re-runs in `research/vesta/run_b0.sh`.

- **N1 (expressivity over DeltaNet on A_5 state-tracking).** GRP-RNN at K=m/2 interlocking planes beats β=1 DeltaNet by ≥0.05 acc on A_5 word-problem recognition at iso-param, T ∈ {64, 256, 1024}, m=256.

- **N2 (throughput within 3× of Mamba-2).** Tokens/sec for training and inference within 0.33× of Mamba-2 reference at T=2048 m=1024.

If B0 fails: infrastructure broken; debug before anything else.
If N1 fails but B0 passes: GRP-RNN reduces to DeltaNet — clean negative result.
If N2 fails badly but B0 + N1 pass: framework is expressivity-at-throughput-cost — publishable with regime qualification.

---

## 8. Theoretical analysis

### 8.1 Well-posedness

The state s_t = λ R_t λ R_{t-1} ... λ R_1 s_0 + Σ_k λ^k (∏_{j<k} R_{t-j}) B x_{t-k+1}. With R_t orthogonal and λ ∈ (0, 1], the linear time-varying system is BIBO stable with norm bound ‖s_t‖ ≤ λ^t ‖s_0‖ + Σ_k λ^k ‖B x_{t-k+1}‖. At λ = 1 (pure orthogonal), the homogeneous term has constant norm — perfect long-range retention. This is the published-favorable case for SSMs on synthetic long-range tasks (LRA).

### 8.2 Conditioning and gradient flow

Under backprop, the gradient of `s_t` with respect to `s_0` is `(λ^t) (R_t R_{t-1} ... R_1)^T`. Norm is `λ^t`. With λ = 1, gradients are conserved exactly. With λ ∈ (0, 1) gradients decay polynomially in t. There is no vanishing-gradient pathology from nonlinear squashing (the only nonlinearity is in the angle map, which is bounded but does not appear inside the gradient chain through s).

This is the *core* explanation for why the EALRMN Phase-1 ablation found "linear recurrence + orthogonal init" responsible for the entire EALRMN win. GRP-RNN structurally maintains this property: at φ_max = π/2 and small initial w_k, angles are near zero, R_t ≈ I, gradient flow is near-isometric.

### 8.3 Expressivity (formal)

By Theorem 3.1 and Corollaries 3.2–3.3, GRP-RNN with K interlocking planes (K ≥ 3, plane graph chromatic number ≥ 3) is operator-complete in SO(m). The reachable single-step transition set is dense in SO(m), bounded only by the per-angle constraint |θ_k| ≤ φ_max.

Multi-step composition: after T steps the operator R_T R_{T-1} ... R_1 lies in SO(m). The set of reachable T-step operators is the image of {(θ_{k,t})_{k,t}} ↦ ∏_t R_t under the (T·K)-dimensional input map. For T·K ≥ m(m−1)/2, this image is generically dense in SO(m); for T·K < m(m−1)/2 it is a proper sub-manifold. In all interesting regimes (T ≥ 64, K = m/2 ≥ 64) the image is dense.

By contrast, the reachable T-step operator of a diagonal SSM is `T^K`, the K-torus — an abelian group of dimension K. The dimension gap is m(m−1)/2 − K = O(m²) parameters of expressivity that GRP-RNN buys.

### 8.4 What this analysis does *not* prove

It does not prove GRP-RNN trains faster or to lower loss on natural language. The expressivity argument is about the *set of representable functions*; whether SGD finds the useful non-diagonal points is an empirical question. F2 in §10 enumerates the case where SGD does not.

### 8.5 Identifiability

Setting aside trivial symmetries (block reordering, decay rescaling), the (θ_{k,t}) parameterization is locally identifiable at any non-degenerate configuration (i.e., where no two angles are exactly 0 or ±φ_max). Saturating angles (tanh ≈ ±1) are non-identifiable in the limit — handled by the `L_rot-reg` regularizer in §4 if observed empirically.

### 8.6 Controllability and observability

The pair (R_t, B) is controllable iff the columns of B span ℝ^m together with the action of {R_t}. With R_t ∈ SO(m) and any non-degenerate B, controllability follows. Observability under C (V × m) requires V ≥ m or C's rows to span ℝ^m; at V = 32 (needle task vocab) and m = 1024 we have V < m and the system is observation-deficient — same as every SSM-class model on this task. The task-specific readout `s_T → logits over V` is the standard SSM bottleneck and not specific to GRP-RNN.

---

## 9. Computational complexity

| Operation | Per-step cost | Per-sequence cost (T steps) | Memory |
|---|---|---|---|
| Compute θ_k(x_t) ∀k | O(K · d) GEMV | O(T · K · d) | O(K · d) params |
| Apply K Givens to s_{t-1} | O(K) per state element ⇒ O(K · m) | O(T · K · m) | O(m) state |
| Scalar decay | O(m) | O(T · m) | O(1) |
| Input projection | O(m · d) GEMV | O(T · m · d) | O(m · d) params |
| Readout | O(m · V) GEMV | O(T · m · V) | O(m · V) params |

At K = m/2, the per-step cost is O(m²/2 + m·d + m·V), comparable to Mamba's O(m·N · d) for state dim N. The dominant term is the input projection + readout, same as LRU and Mamba. The K-Givens cost (m²/2 work per step) is the marginal extra cost — equivalent to one m×m matvec per step. Memory-bound, not compute-bound, since cos/sin can be computed in registers.

**Parallel-scan training.** The Givens product is associative. Chunkwise scan with chunk size W = 64 stores the chunk-product `Q_b = R_{bW} · ... · R_{(b-1)W+1}` ∈ SO(m). Cost per chunk: W Givens applications (O(W · K · m) work) plus one m×m GEMM at chunk boundary. With T = 2048, m = 1024, K = 512, W = 64: total work ≈ 2048 · 512 · 1024 ≈ 10^9 ops, plus 32 chunk-boundary 1024×1024 GEMMs (≈ 32 · 2·10^9 = 6·10^10 ops). The chunk-boundary GEMMs dominate.

This is the same asymptotic regime as Mamba-2's SSD scan. Throughput claim N2 (within 3× of Mamba-2) is feasible under this scheme; the precise number depends on kernel-launch overhead and cuBLAS GEMM efficiency at the chosen chunk size.

**Inference.** Recurrent, per-step. Per-step cost as above. No KV cache. Memory is O(m) for state plus O(K · d + m · d + m · V) for parameters.

---

## 10. Failure modes

### 10.1 F1 — Angles collapse to zero

**Symptom.** Mean(|θ_k(x_t)|) → 0 across training; R_t → I; GRP-RNN degenerates to EMA / MEGA-core.

**Why plausible.** L_CE may prefer the identity recurrence in early training, before any non-zero rotation produces a useful predictive signal.

**Diagnostic.** Histogram of |θ_k(x_t)| every 200 steps. If mean(|θ|) decays monotonically through training, F1 is active.

**Mitigation.** Activate L_rot-reg with a *sign-flipped* coefficient (push angles *away* from zero) or initialize w_k with larger norm so initial angles are away from zero.

### 10.2 F2 — K > 1 advantage does not materialize empirically

**Symptom.** N1 result: GRP-RNN at K = m/2 ties β=1 DeltaNet on A_5 word recognition. The expressivity is correct theoretically but optimization cannot realize it.

**Why plausible.** K-Givens product factorization is "narrow" — each plane-generator is a 2D subspace of so(m), and the gradient through K compositions has K-times-amplified noise. DeltaNet's β=1 rank-2 update has better-conditioned gradients because the rank-2 perturbation is one unit.

**Diagnostic.** Per-plane gradient norm distribution at end of N1 training. If only ~3 of K planes have significant non-zero gradients, the model is using GRP-RNN as if K=3 — F2 confirmed.

**Mitigation.** Reduce K to 8 or 16 and re-run N1. If the gap appears at small K but not at large K, the expressivity claim survives at small K. If the gap doesn't appear at any K, the claim is empirically falsified (despite being theoretically correct).

**Falsification interpretation if F2 realizes.** GRP-RNN reduces operationally to DeltaNet with extra parameters that don't help. This is a publishable negative result: a *theoretical* expressivity argument that fails *empirically* due to gradient-conditioning constraints is informative — it tells the SSM community that the Merrill et al. 2024 bound is loose in practice.

### 10.3 F3 — Chunkwise scan is too slow

**Symptom.** N2 fails: GRP-RNN training tok/s < 0.1 × Mamba-2.

**Why plausible.** Chunkwise scan needs to materialize the chunk-product Q_b ∈ ℝ^{m×m} at every chunk boundary. At m = 1024 this is a 4 MB matrix per chunk. The launch overhead for the kernel chain (Givens product + scalar mul + input GEMV + chunk-end m×m GEMM) is 4–8 μs per layer per step on Ada. Per-step launch overhead may dominate.

**Diagnostic.** Nsight Compute profile: if total kernel-launch overhead > 40% of wall-time, F3 is active.

**Mitigation.** Fuse Givens-product kernel with input-GEMV into a single launch, eliminating half the launches. Or use CUDA graphs to pre-capture the per-chunk dependency graph (Mamba-2's approach).

**Falsification interpretation.** If F3 cannot be mitigated within a 1-week kernel-engineering effort, report N2 as "expressivity-at-throughput-cost" and recommend GRP-RNN only for tasks where the diagonal-SSM expressivity gap matters (state-tracking, algebraic reasoning).

### 10.4 What is *not* a failure mode

- **The framework not transferring to natural language LM.** This is explicitly *outside* the scope of N1–N3. A future Phase 4+ can attempt this. F2 already identifies the gradient-conditioning failure that would make natural-language transfer fail; we don't need a separate failure mode.

- **Bootstrap circularity.** GRP-RNN has no gates. There is no second-order circularity surface.

- **State explosion.** With orthogonal R_t and λ ≤ 1, the state norm is bounded for free.

---

## 11. Implementation plan

### 11.1 Phase order

1. **Phase-0 (B0 confirmation).** *Already in flight* — `research/vesta/run_b0.sh` is running. Validates that the testbed reproduces the 131× tanh→linear gap from EALRMN Phase-1. ETA: ~15 min wall-clock.

2. **Phase-1 (GRP-RNN GPU prototype on top of `research/ealrmn_gpu/`).** Add a new model `grp_rnn` to the existing `--model=...` switch. Reuse encoder, optimizer, eval loop verbatim. Implement Givens-product forward + backward in `model_grp_rnn.cuh`. *No* manifold representation, *no* matrix-exponential. ~600 LOC. Wall-clock: 3–4 days.

3. **Phase-2 (Claim N3 = B0 with GRP-RNN flags).** Run `--grp-tanh-state` ON vs OFF, 5 seeds, T=2048 m=1024. Expect ≥100× gap. ETA: 1 hour.

4. **Phase-3 (Claim N1 = A_5 word recognition).** Build the A_5 task generator (uniform random A_5 word, label = identity / not-identity). Run GRP-RNN at K = m/2 vs β=1 DeltaNet at K=1 at iso-param, T ∈ {64, 256, 1024}, m=256, 10 seeds each. ETA: 8 GPU-hours.

5. **Phase-4 (Claim N2 = throughput).** Benchmark GRP-RNN vs Mamba-2 reference at T=2048 m=1024. ETA: 1 hour.

6. **Phase-5 (Iteration / report).** Decision table + final report at `research/VESTA_REPORT.md`.

### 11.2 Why this fits in the existing ealrmn_gpu testbed (vs new infrastructure)

The existing `research/ealrmn_gpu/` infrastructure provides:
- Embedding, layer-norm, softmax+CE, AdamW, deterministic RNG, gradient check, JSONL logging, sweep driver, aggregator.
- Task generators for needle, HMM, syntheticlm. We add `a5_word` task.
- Model dispatcher pattern: `--model=rnn`, `--model=ealrmn_attmem`, `--model=transformer_1l`. We add `--model=grp_rnn`.
- DeltaNet reference: not present; we implement β=1 DeltaNet as a special case of GRP-RNN with `--grp-K=1 --grp-plane-input-dep`.

This avoids building a new testbed and gets us to a decisive N1 result fastest.

### 11.3 New kernels (CUDA)

| Kernel | File | LOC | Purpose |
|---|---|---|---|
| `k_givens_apply_fwd` | `model_grp_rnn.cuh` | 80 | Apply K Givens to s in (interlocking) plane order |
| `k_givens_apply_bwd` | `model_grp_rnn.cuh` | 100 | Backward pass; uses θ̄_k = −θ_k for inverse |
| `k_compute_angles` | `model_grp_rnn.cuh` | 40 | θ_k(x_t) = tanh(w_k^T x_t + b_k) · φ_max |
| `k_compute_angles_bwd` | `model_grp_rnn.cuh` | 50 | Backward through tanh + GEMV |
| `chunkscan_forward` | `model_grp_rnn.cuh` | 200 | Blelloch parallel scan over chunked R_t products |
| `chunkscan_backward` | `model_grp_rnn.cuh` | 200 | Reverse-scan backward |

**Total new CUDA: ~670 LOC.** Plus glue (model class, gradient check, smoke tests): ~330 LOC. Tracks Candidate A's 1000-LOC target.

### 11.4 What is reused unchanged

| Module | Source | Use |
|---|---|---|
| Embedding fwd/bwd | `kernels.cuh` | Same encoder for all models |
| Layer-norm fwd/bwd | `kernels.cuh` | Pre-readout layer norm |
| Softmax + CE | `kernels.cuh` | Loss |
| AdamW | `optimizer.cuh` | Optimizer |
| Deterministic RNG | `common.cuh` | Reproducibility |
| Gradient check | `main.cu` `gradcheck_mode` | Verification of new kernels |
| JSONL logging + aggregator | `main.cu`, `aggregate.cpp` | Sweep results |
| Needle / HMM task generators | `tasks.cuh` | B0 + sanity sweep |

### 11.5 New task generator

The A_5 word recognition task (`tasks.cuh::a5_word`). Per sample:
- Generators: g_0 = (1 2 3), g_1 = (3 4 5). Vocab V = 6 (5 elements of A_5 act on {1, 2, 3, 4, 5}; or use V = 2 for the two generators, with separator).
- Sample a random word w ∈ {g_0, g_1}^T uniformly. Token at position t encodes which generator is applied.
- Compute the product π = g_{w_T} ◦ ... ◦ g_{w_1} ∈ A_5.
- Label = 1 if π is the identity element, else 0. Probability of identity in a uniform random word of length T is ~1/60 → balance by rejection-resampling.

This is the simplest task that discriminates solvable from non-solvable groups (Merrill et al. 2024 §3.1).

---

## 12. Comparison to existing methods (audit-anchored)

| Existing baseline | GRP-RNN relationship | Where comparison is sharp |
|---|---|---|
| **S4 / S4D** (Gu 2021) | Subsumed (S4-diag = LRU + HiPPO λ schedule = `--grp-disjoint-planes` + tuned λ) | Long-Range Arena (LRA) — but GRP-RNN expects parity, not improvement; LRA does not discriminate H_K vs T^K |
| **Mamba (S6, SSD)** (Gu/Dao 2023, 2024) | Strict superset under R_t = I restriction | Mamba's documented strength (long-context retrieval) — GRP-RNN should match within 2× tok/s |
| **LRU** (Orvieto 2023) | Strict subset (`--grp-disjoint-planes` + `--grp-fixed-angles`) | Direct ablation: K = 0 vs K = 1 vs K = m/2 at fixed compute |
| **DeltaNet** (Yang/Schlag 2024) | β=1 DeltaNet ⊂ GRP-RNN @ K=1 | A_5 word recognition (Claim N1) |
| **RWKV-7** (Peng 2025) | Different structural choice (rank-1 matrix-valued state vs rank-K rotation) | Not directly compared; orthogonal axes |
| **Based** (Arora 2024) | Different axis (R4 explicit memory) | Not directly compared; orthogonal axes |
| **Hyena, Mega, RetNet** | Different structural choice (convolution / EMA / retention) | Not directly compared; reduces to `--grp-K=0` analog |

The audit (`VESTA_AUDIT.md` §3.5) identifies "exact copying / state-tracking expressivity" as the documented Mamba/SSM weakness with the strongest theoretical backing. GRP-RNN targets exactly this gap.

---

## 13. What GRP-RNN deliberately does NOT propose

For audit-traceability and to reject the EALRMN trap of stacking mechanisms:

- **No latent-prediction objective** (R2). Phase-0a/0b's failure is respected.
- **No bounded associative memory** (R4). Phase-0g/0k's null is respected.
- **No mixture of experts** (R5). Out of scope.
- **No compute-adaptive inference** (R7). Out of scope.
- **No entropy-adaptive chunking** (R1). Out of scope.
- **No raw-token-vs-latent decoding switch** (R8). Out of scope.
- **No paired-stream construction** (Candidate B's R2 attack). Parked.
- **No endogenous-surprisal-residual detector** (Candidate C's gate design). Parked.

GRP-RNN sits *only* on axis R3 (recurrence type) and claims novelty *only* relative to diagonal SSMs and to DeltaNet. If N1 fails, the claim is reduced to "GRP-RNN ≈ DeltaNet" — a publishable equivalence, not a Pyrrhic victory.

---

## 14. Open conjectures and validation criteria

The framework leaves the following open:

- **C-A (transferability to LM).** Conjecture: the A_5 expressivity advantage transfers to *some* natural-language tasks (induction heads, in-context learning of novel compositional functions). Validation: a Phase-4 LM experiment at scale, *only if* N1 supported.
- **C-B (saturation regime).** Conjecture: as K grows beyond ~16, marginal expressivity gain diminishes due to gradient-conditioning constraints (F2). Validation: K sweep at fixed param count, K ∈ {1, 4, 16, 64, 128, 256, 512}.
- **C-C (plane-graph topology).** Conjecture: the stride-3 interlocking plane graph is suboptimal; a random expander-graph plane choice would give faster Lie-algebra generation. Validation: plane-graph topology ablation at fixed K.
- **C-D (HiPPO integration).** Conjecture: combining the HiPPO λ-schedule (S4) with the K-Givens rotation gives the best of both worlds (long-range retention + state-tracking expressivity). Validation: Phase-4 ablation if N1 + N2 + LM transfer all pass.

These are not pre-registered claims. They are conjectures whose status will be reported, not pre-committed.

---

## 15. Phase-by-phase decision table

To be updated after each phase. Template:

| Phase | Claim | Status | Regime where supported | Notes |
|---|---|---|---|---|
| 0 | B0 (infrastructure) | IN PROGRESS | T=2048, m=1024 iso-param | `research/vesta/run_b0.sh` running |
| 1 | (build prototype) | PENDING | — | After B0 confirmed |
| 2 | N3 (B0 with GRP-RNN flags) | PENDING | — | Replicates B0 via `--grp-tanh-state` ablation |
| 3 | N1 (A_5 expressivity) | PENDING | T ∈ {64, 256, 1024} m=256 | Strongest baseline: β=1 DeltaNet |
| 4 | N2 (throughput) | PENDING | T=2048 m=1024 | Strongest baseline: Mamba-2 SSD |
| 5 | Report | PENDING | — | Final writeup |

---

## 16. Summary

**GRP-RNN** is the strongest of three candidate frameworks for the VESTA program. Its central commitment is the simplest possible structurally-novel recurrence:

> A per-step rotation `R_t = ∏_k G_k(θ_k(x_t)) ∈ SO(m)`, K Givens on interlocking planes, input-dependent angles.

Reductions to LRU, DeltaNet, Mamba-2 (SSD), MEGA-EMA, and tanh-RNN are explicit single-flag ablations. The theorem in §3.3 establishes operator-completeness in SO(m) — a *proven* expressivity gap over diagonal SSMs. The empirical claim N1 (beat β=1 DeltaNet on A_5 word recognition at iso-param) tests whether the theoretical gap manifests under SGD. The framework is implementable in ~1000 LOC of new CUDA on top of `research/ealrmn_gpu/`, has no gates and no bootstrap surfaces, and produces clean publishable results in both pass and fail outcomes.

Implementation begins after B0 confirmation completes.

— end VESTA_FRAMEWORK.md —
