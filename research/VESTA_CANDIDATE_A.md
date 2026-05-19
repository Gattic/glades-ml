# VESTA Candidate A — GRP-RNN

**Group-Rotation Product Recurrent Network.** A non-diagonal, non-LTI structured
recurrence whose state-transition matrix is, at every step, an element of
SO(2k) ⊂ SO(m) realized as a product of K input-dependent Givens (2×2 plane)
rotations. The state lives on the unit sphere S^{m-1} (or, in the augmented
variant, on a Stiefel-manifold-anchored frame); the dynamics are an
input-controlled discrete trajectory through SO(2k).

**Date:** 2026-05-19. **Author:** VESTA Candidate-A designer (parallel-3 cohort,
ONE of three; independent from Candidates B and C).

**Short name (used throughout):** **GRP-RNN**.

---

## 0. The single claim

GRP-RNN is the proposal that *the cheapest way to escape the diagonal-SSM
expressivity ceiling is to make the recurrence a product of Givens rotations
whose plane-angles are input-dependent*. Givens rotations are the smallest
non-trivial elements of SO(n); a product of K of them in fixed planes generates
a depth-K word in a finitely-generated subgroup of SO(n). For carefully chosen
plane sets, that subgroup contains non-solvable groups (e.g. A_5 ⊂ SO(3) via
the icosahedral embedding), giving a *provable* expressivity advantage over
diagonal SSMs (Merrill, Petty, Sabharwal arXiv 2404.08819).

The whole framework is one mechanism, one flag, one ablation. The rest of the
document derives it, reduces it to LRU / Mamba / DeltaNet under explicit
restrictions, states three pre-committed claims, and sketches the C++/CUDA
implementation on top of `glades-ml`.

---

## 1. Primitive objects

| Symbol | Meaning |
|--------|---------|
| `t` | discrete time index, `t = 1 .. T` |
| `m` | model dim (state dim is `m`, embedding dim is `m`) |
| `K` | number of Givens rotations per step. **Hyperparameter; K = m/2 gives full coverage of a single SO(m)-coset block; K < m/2 gives a *structured* sub-manifold.** |
| `x_t ∈ R^d` | input vector at step t (token embedding) |
| `s_t ∈ R^m` | recurrent state at step t |
| `B ∈ R^{m × d}` | input projection (fixed, learned) |
| `C ∈ R^{v × m}` | readout (fixed, learned) |
| `(p_k, q_k)` | the two coordinate axes of the k-th Givens plane. Fixed at init; **non-overlapping** for `k=1..K` so that the K rotations commute pairwise when restricted to distinct planes, *but* their composition with the input map B does not commute (this is the source of non-diagonality). |
| `θ_k(x_t) ∈ R` | rotation angle for the k-th Givens, depends on the current input. |
| `G_k(θ) ∈ SO(m)` | the m×m matrix that is identity except for the 2×2 sub-block at rows/cols (p_k, q_k), which is `[[cos θ, -sin θ], [sin θ, cos θ]]`. |
| `R_t = ∏_k G_k(θ_k(x_t)) ∈ SO(m)` | per-step rotation. K matrix-products of 2×2 blocks. |
| `λ ∈ (0, 1]` | optional scalar decay applied multiplicatively to the state to allow forgetting (without it the recurrence is energy-preserving). |
| `Λ_t ∈ R^{m×m}` | optional input-dependent **diagonal** decay; default `Λ_t = λ I`. |
| `η ∈ R` | gate scalar for the input injection step (default 1). |
| `α, β, γ` | three loss-mixture coefficients defined in §7. |

Throughout this document **the state s_t lives in the ambient R^m, not on a
manifold representation.** The orthogonality is in the *operator*, not the
state. This is why we can compute everything with standard cuBLAS GEMVs and
elementwise kernels — no manifold projection, no Riemannian SGD, no
matrix-exponential approximation.

The "non-diagonality" of the transition operator is the *cross-coupling between
distinct planes induced by the input map B*. Each Givens G_k acts on a single
plane; composed *with each other* they commute (disjoint planes), but composed
*with B + the input-dependent θ_k(x_t)* they generate non-commutative dynamics
across planes. This is the precise sense in which GRP-RNN is non-diagonal: in
the eigenbasis of the recurrence the operator is *block*-diagonal (2×2 blocks),
but the input-dependent θ_k(x_t) couples blocks through the input map.

---

## 2. State space

The state space is `R^m`. The set of *reachable* transition operators is

  G_K  :=  { ∏_k G_k(θ_k) : θ_k ∈ R }  ⊂  SO(m)

which is a K-torus T^K embedded in SO(m). With `Λ_t = λ I` (scalar decay) the
*reachable* one-step operator set is `[0,1] · G_K`, i.e. a scaled K-torus.

If we are willing to let `θ_k` *depend on x_t*, the K-torus becomes
**input-controllable** in the sense of control theory: the orbit map
`(x_1, …, x_T) ↦ R_T · R_{T-1} · … · R_1` realizes a word in the free group
F_K with letters of the form `G_k(θ)`. The set of such words modulo the
relation `G_k(θ) G_k(φ) = G_k(θ + φ)` is the path-group of the K-torus, which
strictly contains diagonal-complex linear recurrences (those are K-torus words
*with fixed θ_k*, i.e. constant-input control).

**Choice of planes.** A naïve choice is `(p_k, q_k) = (2k-2, 2k-1)` — disjoint
adjacent pairs, K = m/2. This is the "block-diagonal SO(m)" parameterization,
trivially equivalent to a complex-diagonal recurrence with shared phase. The
*expressivity-bearing* choice is **interlocking planes** with k-shift, e.g.

  (p_k, q_k) = (2k-2 mod m,  (2k-2 + 2s) mod m),  k = 1..K

with stride s coprime to m/2. This makes the K Givens rotations *not* pairwise
commuting (they share coordinates), so their product is a non-trivial element
of SO(m) that does not factor as a block-diagonal complex recurrence.

We take the interlocking-planes choice as the default. The "trivial-planes"
choice (disjoint pairs) is the *ablation* that recovers LRU as a strict special
case (see §5).

---

## 3. Evolution law

Discrete-time update:

  R_t   = ∏_{k=1}^{K}  G_k( θ_k(x_t) )                                  (3.1)

  s_t   = Λ_t · R_t · s_{t-1}  +  η · B x_t                              (3.2)

  y_t   = C · s_t                                                       (3.3)

  θ_k(x_t) = tanh( w_k^T x_t  +  b_k )  ·  φ_max                          (3.4)

where `φ_max ∈ (0, π)` is a learned but bounded scalar (default π/2), `w_k ∈ R^d`,
`b_k ∈ R` are learned. The tanh is *not* the source of nonlinearity-in-the-state;
it is a nonlinearity-in-the-*parameter* (the rotation angle is a non-linear
function of the input, but the state evolves linearly in s_{t-1} for given x_t).

This is the crucial point that distinguishes GRP-RNN from a tanh-RNN: the state
evolution is **conditionally linear** in s_{t-1} given x_t. The non-linearity
sits in how x_t maps to the transition operator, exactly as in Mamba's
selective scan and DeltaNet's delta-rule. This means:

* The recurrence is a **linear time-varying (LTV)** system, not a non-linear
  one. There is no vanishing-gradient pathology of the kind tanh-RNNs suffer.
* It can be unrolled in parallel: see §9 for the chunkwise scan algorithm
  that exploits associativity of rotation products.
* The orthogonality of R_t makes the gradient norm conserved exactly under
  the no-decay limit Λ_t = I, removing the need for orthogonal-init tricks.

**Continuous-time view.** Set τ = t · Δt with Δt → 0. Define

  Ω(x, t) = Σ_k  θ̇_k(x_t)  E_k

where `E_k ∈ so(m)` is the antisymmetric "generator" of the k-th Givens (the
2×2 matrix `[[0,-1],[1,0]]` embedded at rows/cols (p_k, q_k)). Then

  ṡ(τ) = -μ(τ) s(τ) + Ω(x(τ), τ) s(τ) + η B x(τ)                       (3.5)

with `Λ ≈ I - μ Δt`. Equation (3.5) is a *linear time-varying ODE on R^m* with
generator in so(m) ⊕ R · I. The discretization (3.1)-(3.2) is the exponential
integrator `R_t = exp(Ω(x_t) Δt)` factored as a product of Givens rotations
(the "Trotter splitting"); this is *exact* when the Givens generators
commute (disjoint planes) and a controlled first-order approximation
otherwise.

This view also clarifies why the framework is "structured non-diagonal": the
generator Ω lies in a fixed K-dimensional subspace of so(m), spanned by the
chosen plane-generators. The recurrence is non-diagonal in any basis that
mixes the planes, but is *structured* in that its generator has K free
parameters (not the m(m-1)/2 of full SO(m)).

---

## 4. Why this captures expressivity that diagonal-complex does not

A diagonal-complex linear RNN (LRU, S4D, the diagonal limit of Mamba) realizes
exactly:

  s_t  =  Diag(c_1, …, c_m) · s_{t-1}  +  B x_t,   c_j ∈ C, |c_j| ≤ 1

In real coordinates this is **block-diagonal SO(m) × diagonal R^m** — the
recurrence operator factors as a direct sum of K = m/2 independent 2×2
rotations on disjoint planes. As an operator group this is the K-torus
**T^K**, which is *abelian and solvable*.

Merrill, Petty, Sabharwal (arXiv 2404.08819) prove: a recurrent net whose
state-transition group is solvable cannot solve word-problems on non-solvable
groups in O(1) depth. The K-torus is solvable; therefore LRU/S4D/Mamba (in the
diagonal limit) cannot, in a single layer, recognize a word in the symmetric
group S_5 or the alternating group A_5 (both non-solvable).

GRP-RNN with **interlocking planes** has a transition operator that does
*not* factor as block-diagonal in any fixed basis. The generated subgroup
of SO(m), call it H_K ⊂ SO(m), is in general **non-abelian** (this is
elementary: two Givens rotations whose planes share an axis do not commute,
since restricted to the 3D coordinate subspace they are SO(3) generators in
the standard sense). For K ≥ 3 with the interlocking-stride choice, H_K
contains finite subgroups isomorphic to dihedral, tetrahedral, octahedral,
and icosahedral subgroups of SO(3). The icosahedral group **A_5 ⊂ SO(3) ⊂
SO(m)** is *non-solvable*. Therefore GRP-RNN can, in principle, realize an
A_5 word as a single-layer recurrence, which a diagonal SSM cannot in O(1)
depth.

The expressivity claim is *structural*, not "with the right training":
the dimension of the Lie algebra Ω lies in is K (the number of distinct
plane-generators E_k), so the parameter count is O(K · m) for the
input-dependent angles, but the *reachable operator set* is the universal
cover of H_K, which is non-abelian for K ≥ 3 with interlocking planes.

**This is the only expressivity claim made by GRP-RNN.** Every other
property — long-range retention, throughput, parallelizability — is
either inherited from diagonal-SSM (so does not have to be claimed as
novel) or is a tradeoff to be measured.

---

## 5. Reduction to existing methods

Under explicit parameter restrictions GRP-RNN reduces to known baselines:

**Restriction Z — diagonal-complex LRU.** Set planes to **disjoint adjacent
pairs** `(2k-2, 2k-1)`, `k=1..K`, K = m/2. Then the K Givens rotations
commute pairwise (disjoint coordinates), so their product is block-diagonal
SO(m). In the standard complex-to-real correspondence

  G_k(θ_k) ↔ multiplication by exp(iθ_k) on C

the recurrence (3.2) becomes

  s_t  =  λ · Diag(exp(iθ_1(x_t)), …, exp(iθ_K(x_t))) · s_{t-1}  +  B x_t

which is precisely **LRU with input-dependent phase, no input-dependent
magnitude.** Setting `θ_k(x_t) = θ_k` (input-independent) recovers vanilla
LRU.

**Restriction X — diagonal-real LTI.** Restriction Z + `θ_k(x_t) = 0`
(no rotation, identity transition) gives s_t = λ s_{t-1} + B x_t, the
exponentially-weighted moving average (EMA) used in MEGA and the
"forget-baseline" linear RNN.

**Restriction Y — DeltaNet special case.** DeltaNet's update is `S_t =
(I - β u_t v_t^T) S_{t-1} + β u_t v_t^T`. Take `β = 1`, choose `(u_t, v_t)`
orthonormal, and the update is `S_t = (I - u_t v_t^T - v_t u_t^T) S_{t-1}` plus
a low-rank rank-2 correction. The map `I - u_t v_t^T - v_t u_t^T` is a Givens
rotation by π in the (u_t, v_t)-plane. Choosing `β ≠ 1` gives a non-orthogonal
update *not* in GRP-RNN's set; however, for `β = 1` with rotation angle π,
DeltaNet's single-step update is exactly a **single rank-2 Givens-π rotation**
in an input-dependent plane. GRP-RNN with K=1 and input-dependent plane (not
just input-dependent angle) is then a *strict superset* of β=1 DeltaNet.
Strictly: GRP-RNN's K is the depth of plane composition, DeltaNet is K=1 with
β=1 fixed and only the plane learned.

**Mamba-2 (SSD).** Mamba's selective scan is `s_t = exp(-Δ_t A) s_{t-1} +
Δ_t B_t x_t` with A diagonal real and Δ_t scalar-per-channel. Set
`Λ_t = exp(-Δ_t diag(α))` (scalar per Givens block), `θ_k(x_t) = 0` (no
rotation), and `R_t = I`. Then GRP-RNN reduces to Mamba-2 with the SSD scalar-
times-identity-per-head structure (compare Dao & Gu §3). **Mamba-2 is a strict
special case of GRP-RNN under three restrictions: identity transition + scalar
input-dependent decay + B_t input-dependent input map.** Note GRP-RNN's
default keeps `B` time-invariant, so we are slightly *less* expressive than
Mamba-2 in the input-projection direction; we deliberately concede this and
spend the budget on rotation expressivity instead. *(This is a real tradeoff,
addressed in the failure-modes section.)*

**S4 / S5.** S4 and S5 use a HiPPO-initialized diagonal-plus-low-rank A.
GRP-RNN's default does not match the HiPPO measure approximation guarantee;
this is a non-trivial difference. If empirically necessary, one can swap the
identity-Λ for a HiPPO-style λ_k schedule without changing the rotation
structure.

**Summary subsumption table.**

| Method | Reduces from GRP-RNN under | What's removed |
|--------|-----------------------------|----------------|
| LRU (input-indep) | disjoint planes + θ_k(x) ≡ θ_k constant | input dependence of angle |
| Input-dep LRU | disjoint planes + Λ = λI | rotation across non-disjoint planes |
| EMA (MEGA core) | Z + θ ≡ 0 | rotation entirely |
| DeltaNet (β=1) | K=1 + input-dep plane choice | composition of K rotations |
| Mamba-2 (SSD) | R_t = I + Λ_t scalar-time-dep + B_t time-dep | rotation entirely (Λ does the work) |
| S4-diag | as LRU + HiPPO Λ schedule | HiPPO measure approximation |

In the parameter ablation matrix the flag `--grp-K=0` recovers EMA, `--grp-K>0
--grp-disjoint-planes` recovers input-dependent LRU, `--grp-K=1 --grp-plane-
input-dep` recovers β=1 DeltaNet, and the default `--grp-K=m/2 --grp-stride=3`
gives the novel mechanism.

---

## 6. Expressivity argument

**Theorem (informal).** Let `H_K ⊂ SO(m)` be the closed subgroup generated by
the K Givens rotations `{G_k(θ) : θ ∈ R}` with planes `{(p_k, q_k)}_{k=1}^K`.
Suppose the plane graph (vertices: coordinate indices in {0,…,m-1}; edges:
plane pairs (p_k, q_k)) is *connected* and has chromatic number ≥ 3.

Then `H_K = SO(n)` where `n = |⋃_k {p_k, q_k}|` is the number of distinct
coordinates touched by at least one plane.

**Sketch.** Each Givens generator lies in so(m). The Lie algebra generated by
{E_k} is a Lie subalgebra of so(m). For disjoint planes, this subalgebra is
the abelian K-torus algebra R^K. When two planes share a coordinate, the
commutator [E_k, E_j] is non-zero — explicitly, if planes share coordinate i,
then [E_k, E_j] is a Givens generator in a *third* plane (the
"completion" of the i-triangle). Iterated commutators of {E_k} with a
connected sharing graph saturate the so(n) Lie algebra by the standard
**3-vertex argument** for Lie-algebra generation of so(n) (any two
non-commuting plane generators in 3D plus a third linearly-independent one
generate so(3); induction on n). Connected plane graphs of chromatic number
≥ 3 ensure no obstruction to generating all of so(n). The closed subgroup
generated by the one-parameter groups exp(R · E_k) is then SO(n) by
Lie's theorem.  □

**Corollary.** With interlocking planes (`(p_k, q_k) = (2k-2, 2k-2 + 2s) mod m`,
gcd(s, m/2) = 1, K ≥ 3), H_K = SO(m). Therefore the reachable transition
operator set of GRP-RNN is **all of SO(m)** — every special orthogonal
operator can be expressed as a product of K = O(m) Givens rotations from the
chosen family (this is just QR/Givens factorization). GRP-RNN with K = m/2
interlocking planes is **operator-complete in SO(m)**, modulo the bounded-
angle constraint `|θ_k| ≤ φ_max`.

**Word-problem corollary.** For finite groups G that embed into SO(n) for some
n ≤ m (in particular, A_5 ⊂ SO(3) ⊂ SO(m) for m ≥ 3), GRP-RNN can realize G
as a product of input-dependent generators. In contrast, a diagonal SSM
(LRU/S4D/Mamba in its scalar-decay form) generates only the abelian (and hence
solvable) subgroup of SO(m). By Merrill et al. 2024, no fixed-depth diagonal
SSM can recognize words on non-solvable groups. GRP-RNN can, in a single
layer, with K = m/2.

The published baseline this targets is **DeltaNet** (Yang & Schlag 2024 arXiv
2406.06484), the strongest non-diagonal linear-recurrence model. DeltaNet is
GRP-RNN's K=1 special case with β=1; its expressivity argument (Yang & Schlag
Theorem 2) shows it can implement bounded-context-free recognition on
non-solvable monoids. GRP-RNN with K > 1 strictly extends DeltaNet's
generating set (more plane choices per step), so the expressivity floor is at
least DeltaNet's. Whether the K > 1 lifting actually wins in practice is what
the empirical test (§8) decides.

**What this expressivity argument does NOT say.** It does *not* say GRP-RNN
will outperform Mamba-2 on language modeling. It does *not* say compositional
generalization will improve. It says only: there exist tasks (word-problems on
A_5, modular arithmetic mod a non-solvable-quotient modulus, parity-like
algebraic state-tracking) where GRP-RNN has expressivity that diagonal SSMs
provably lack. The empirical test below pre-commits to one such task.

---

## 7. Objective

**One loss, three coefficients, all ablatable to zero.**

  L_total  =  α · L_CE  +  β · L_rot-reg  +  γ · L_decay-reg                (7.1)

where:

  L_CE  = - Σ_t  log p_θ(x_t | x_{<t})                                    (7.2)

is standard next-token cross-entropy on output `y_t` of (3.3).

  L_rot-reg  =  (1/K) Σ_k  | tanh(w_k^T x̄ + b_k) | · φ_max               (7.3)

  L_decay-reg  =  - log( λ )                                                (7.4)

The two regularizers each address one specific failure mode:

* **L_rot-reg** discourages saturating rotation angles (where θ_k ≈ ±φ_max,
  the tanh gradient ≈ 0, and the per-step rotation cannot adapt). It is
  evaluated on the *batch mean input* x̄, not per-token, so it does not
  destroy input-dependent specialization. **Set γ_rot = 0 to ablate.**

* **L_decay-reg** is an `entropy-of-decay` regularizer. log(λ) penalty
  attached to a learned scalar λ ∈ (0, 1] via reparameterization. With
  γ = 0 the model is free to choose pure-orthogonal Λ = I (no forgetting);
  with γ > 0 a slight contraction is preferred. **Set γ_decay = 0 to
  ablate.**

There is no contrastive term, no reconstruction term, no information-bottleneck
term, no expert-balance term, no compute term. The EALRMN-style 7-term
Lagrangian is *deliberately rejected*: every additional term carries a
bootstrap-circularity risk (the EALRMN writeup §5 documents this). If a future
phase needs a new term, it must justify itself against a single-loss baseline
*before* being added to the framework.

**Ablation positions.** Each component is removable as a single flag:

| Flag | Effect | Reduces GRP-RNN to |
|------|--------|---------------------|
| `--grp-K=0` | no rotation at all | EMA / MEGA-core |
| `--grp-disjoint-planes` | disjoint adjacent planes | input-dep LRU |
| `--grp-fixed-angles` | θ_k(x) ≡ θ_k constant | input-indep LRU |
| `--grp-no-decay` (Λ ≡ I) | pure orthogonal | "B0-replication" target |
| `--grp-tanh-state` (deliberately broken) | s_t ← tanh(R_t s_{t-1} + B x_t) | tanh-RNN (the EALRMN strawman) |
| `--grp-rot-reg=0` | drop L_rot-reg | α-only |
| `--grp-decay-reg=0` | drop L_decay-reg | α-only |

The `--grp-tanh-state` flag exists *specifically* to recreate the EALRMN
falsification target: applying a tanh to the state after the linear update
should destroy performance by 100× on the needle task, replicating Phase-1
ablation. This is **claim B0 / replication**.

---

## 8. Pre-committed claims

Three claims, three pre-commit tables. No fourth claim. If a future
experiment makes us want to add more terms, the framework as published is
falsified.

### Claim N1 — Expressivity over DeltaNet on state-tracking

**Task:** Word-problem recognition on A_5 (alternating group on 5 elements,
the simplest non-solvable group). Sequence of symbols from A_5 generators
{(1 2 3), (3 4 5)} of length T ∈ {64, 256, 1024}; label = identity element of
the word-product. Training: sample words uniformly, label balanced 50/50
identity vs non-identity. Strong-baseline: DeltaNet (β=1) at iso-param. Weak-
baseline: LRU at iso-param. Floor-baseline: 50% (chance).

**Param-matching protocol:** all models at d = 128, m = 256, 1-layer. Number
of trainable parameters reported and matched within ±5%. K is set per model so
the parameter count is constant (DeltaNet K=1; GRP-RNN K=m/2=128 default).

**Training budget:** 10 seeds, 20 000 steps each, Adam lr=3e-4 with cosine
decay, batch 64. Wall-clock budget cap: 8 GPU-hours per cell.

**Metric:** held-out accuracy on 4096 words at each T, mean over seeds.

**Pre-commit interpretation table:**

| Observation (mean over 10 seeds; 95% CI) | Interpretation |
|------------------------------------------|----------------|
| GRP-RNN > DeltaNet by ≥ 0.05 at T ≥ 256 AND GRP-RNN > LRU by ≥ 0.20 at T ≥ 256 | **N1 SUPPORTED**: GRP-RNN's K>1 rotation lift is load-bearing |
| GRP-RNN ≤ DeltaNet + 0.02 at all T | **N1 FALSIFIED**: K>1 rotation does not extend DeltaNet's expressivity in practice; GRP-RNN's novelty collapses to "DeltaNet with different notation" |
| GRP-RNN ≈ LRU + 0.02 at all T (both fail) | **N1 INCONCLUSIVE**: task does not discriminate the expressivity gap (suggests A_5 word recognition needs stronger model or longer training; rerun at scale before concluding) |
| GRP-RNN > LRU by ≥ 0.20 AND GRP-RNN > DeltaNet by ≥ 0.05 AT T=64 ONLY (regresses at T=256, T=1024) | **N1 INCONCLUSIVE, REGIME-LIMITED**: report as "novel at short T only", apply Phase-0k lesson, do not generalize |
| GRP-RNN matches DeltaNet at T=64 but degrades faster than DeltaNet at T=1024 | **N1 FALSIFIED + new failure mode**: K>1 product is *less* stable than K=1; investigate as optimization-not-expressivity issue but do not claim novelty |
| Variance in GRP-RNN across seeds > 2× DeltaNet variance | **N1 INCONCLUSIVE + STABILITY FLAG**: GRP-RNN trains less stably, novelty claim contingent on fixing this; rerun with orthogonal-init on plane parameterization |

**Confounds that would invalidate the test (must check before publishing):**
* DeltaNet implementation incorrect (chunked parallel update wrong) → reproduce DeltaNet on the published synthetic-recall task in Yang & Schlag §5.1 as a sanity check.
* GRP-RNN trained without rotation regularizer, hits saturation → check φ_max usage on a held-out batch.
* Param count not actually matched (one model has 1.2× more params) → re-tune K and m to match.
* A_5 words too short / too long for either model to learn (chance both sides) → ensure DeltaNet at T=64 reaches ≥ 0.7 (sanity).

### Claim N2 — Throughput within 3× of Mamba-2 at T=2048 m=1024

**Regime:** T=2048, m=1024, batch=8, A100 or H100 single-GPU, bf16 weights +
fp32 state, no Mamba-2 SSD-block-size tuning beyond defaults.

**Metric:** tokens/sec, training-forward and inference-recurrent separately.

**Baseline:** Mamba-2 from `state-spaces/mamba` reference implementation
(Triton selective_scan). Compared against GRP-RNN with K=512 (= m/2) and
chunkwise parallel scan (§9).

**Target:** GRP-RNN training tok/s ≥ 0.33 × Mamba-2 training tok/s.
GRP-RNN inference tok/s ≥ 0.33 × Mamba-2 inference tok/s.

**Pre-commit interpretation table:**

| Observation | Interpretation |
|-------------|----------------|
| GRP-RNN tok/s ≥ 0.5 × Mamba-2 | **N2 SUPPORTED (strong)**: throughput is competitive |
| 0.33 × Mamba-2 ≤ GRP-RNN tok/s < 0.5 × Mamba-2 | **N2 SUPPORTED (weak)**: publishable as "expressivity-bought-at-throughput-cost" |
| 0.1 × Mamba-2 ≤ GRP-RNN tok/s < 0.33 × Mamba-2 | **N2 FALSIFIED**: framework is too slow; pivot K downward or use full SO(m) generator |
| GRP-RNN tok/s < 0.1 × Mamba-2 | **N2 STRONGLY FALSIFIED**: kernel design is wrong; reimplement before re-running |
| Inference fast (≥ 0.5 ×) but training slow (< 0.1 ×) | **N2 partial**: the chunkwise scan is the bottleneck; chunk-size sweep before declaring |

**Confounds:** Triton vs CUDA kernel-stack overhead (use a kernel-time
breakdown, not just wall-clock), bf16 vs fp32 numeric differences (run both at
fp32 to isolate), batch padding inefficiency at non-power-of-2 T.

### Claim N3 — B0 replication (validates infrastructure)

**Pre-registered, mandatory by `newmodel.txt` lines 268-273.**

GRP-RNN with `--grp-tanh-state` ON should diverge or train ≥ 100× worse than
GRP-RNN with `--grp-tanh-state` OFF on the needle task (T=2048, m=1024, single
KV pair, single-layer encoder), reproducing the EALRMN Phase-1 finding that
"linear recurrence + orthogonal init beats tanh RNN by ≥ 100× val_loss."

**Baseline:** the *same* GRP-RNN code at `--grp-tanh-state` OFF.

**Sub-baseline:** Mamba-2 at iso-param on the same task (sanity).

**Pre-commit interpretation table:**

| Observation | Interpretation |
|-------------|----------------|
| `--grp-tanh-state` OFF beats `--grp-tanh-state` ON by ≥ 100× val_loss | **N3 SUPPORTED**: infrastructure works, prior result replicated |
| 10× ≤ gap < 100× | **N3 INCONCLUSIVE**: infrastructure plausibly works, but the gap is smaller than Phase-1 reported; investigate (could be hyperparam mismatch or task implementation) |
| gap < 10× | **N3 FALSIFIED**: testbed broken; debug before any other claim |
| `--grp-tanh-state` OFF diverges (NaN) | **N3 FALSIFIED + setup error**: orthogonal-init in our framework is mis-specified; fix init before re-running |
| Both diverge | **N3 FALSIFIED + task error**: needle task too hard at this scale; reduce m or T or fix task generation |

**Confounds:** seed-to-seed variance (use ≥ 5 seeds and report median + IQR),
gradient-clip mismatch (use the *same* clip in both branches), Adam ε
mismatch.

### Total-set claim count: 3.

There is no Claim N4, N5, or N6. The framework's empirical contribution
stands on N1 (expressivity-novelty) supported by N2 (throughput-not-a-deal-
breaker) supported by N3 (infrastructure-confirms). If N3 fails, abort. If
N3 passes but N1 fails, GRP-RNN reduces to DeltaNet with different notation
and we report the negative result. If N1 and N3 pass but N2 fails badly, we
report "expressivity at high cost" as the contribution and recommend the
framework only for tasks where the diagonal-SSM gap matters.

---

## 9. Implementation sketch

### 9.1 Files to add (new) and extend (existing)

**New CUDA kernels (under `Backend/Machine Learning/Networks/cuda/`):**

* `gpu_vesta.cu` — *already exists as placeholder*. Replace with the GRP-RNN
  entry point. Public functions: `vesta_grp_forward`, `vesta_grp_backward`,
  `vesta_grp_chunkscan_forward`, `vesta_grp_chunkscan_backward`. **~400 LOC.**
* `gpu_vesta_kernels.cuh` — the Givens-product per-step kernel and the
  chunkwise-product reduction. **~250 LOC.**
* `gpu_vesta_angles.cuh` — input-dependent angle computation `θ_k(x_t) =
  tanh(w_k^T x_t + b_k) · φ_max`, including its backward pass.
  **~80 LOC.**

**Existing files to extend:**

* `Backend/Machine Learning/Networks/network.h` — add `TYPE_GRP_RNN = 8` to
  the enum (line 2118 region) and the API hooks. **~20 LOC.**
* `Backend/Machine Learning/Networks/sgd_rnn.cpp` — add the `TYPE_GRP_RNN`
  branch in the SGD dispatch (Givens-product forward/backward sits next to
  RNN forward/backward; differs only in the transition kernel). **~150 LOC.**
* `Backend/Machine Learning/Structure/hiddenlayerinfo.{h,cpp}` — add fields
  `int grp_K`, `int grp_stride`, `bool grp_disjoint_planes`, `bool
  grp_fixed_angles`, `bool grp_no_decay`, `bool grp_tanh_state`. **~50 LOC.**
* `Backend/Machine Learning/main.h` and `main.cpp` — flag plumbing in
  `glades::init()` and `glades::train()`. **~30 LOC.**

**Unit tests (`unit-tests/Backend/Machine Learning/`):**

* `nn-grp-rnn.cpp` — replication-correctness tests:
  * `test_givens_product_associativity` (K=1, K=2 disjoint should commute,
    K=2 interlocking should not).
  * `test_a5_word_recognition_smoke` (T=64, m=64, K=32, 100 steps, should hit
    ≥ 60% accuracy).
  * `test_tanh_ablation` (B0 replication on needle).
  * `test_lru_reduction_bit_match` (disjoint planes + fixed angles + scalar
    decay should produce numerically identical state to a reference LRU
    implementation to 1e-5).
  **~250 LOC.**

**Total new CUDA: ~730 LOC. Total new + extended C++: ~1180 LOC.** Tracks the
brief's "~1000 LOC of new CUDA" target; slightly over (730 CUDA + 250 host
hooks); justify the overage by the framework requiring no new infrastructure
beyond existing cuBLAS GEMV pipelines.

### 9.2 The per-step kernel

A single step of GRP-RNN at fixed t involves:

1. Compute angles `θ_k(x_t)` for k=1..K. This is a GEMV (W ∈ R^{K×d}, x_t ∈
   R^d → θ_raw ∈ R^K) followed by element-wise tanh and scale. Can be
   accelerated by cuBLAS GEMV + a fused elementwise kernel for tanh*φ_max,
   or written inline. **One cuBLAS GEMV + one elementwise kernel.**
2. Apply K Givens rotations to s_{t-1} in sequence. Each Givens is a 2×2
   matrix-vector multiply on two coordinates of s. With K Givens that share
   coordinates (interlocking), the order matters; we apply them in
   k = 1, 2, ..., K order. Each plane-rotation is one fused two-element
   read/cos/sin/write. **One custom CUDA kernel; K updates of 2 elements
   each = 2K memory writes, m+2K reads if we cache cos/sin per Givens; the
   kernel is memory-bound.**
3. Apply scalar decay `Λ_t = λ I` to the rotated state: `s ← λ s`. **One
   elementwise kernel.**
4. Add `η B x_t`: standard cuBLAS GEMV. **One cuBLAS GEMV.**

Per-step compute is dominated by (4), the m×d GEMV. The K Givens cost is
O(K) work per state element, but is memory-bound (4 floats touched per
Givens). At K = m/2 the Givens phase is O(m) work vs the m×d input-projection
GEMV at O(md), so the recurrence is asymptotically as fast as a vanilla RNN
*per step* — modulo the kernel-launch overhead, which is the real concern.

### 9.3 Chunkwise parallel scan (for training throughput)

A naive sequential application is O(T) launches, which kills training at long T.
The Givens product is associative (matrix product is associative), so we can
do a Blelloch parallel scan over the per-step rotation matrices:

  S_t  =  R_t · R_{t-1} · … · R_1  ∈ SO(m)

Storing S_t in full is O(m²) memory per t — prohibitive. The trick is to
*chunk* the sequence into blocks of size W (e.g. W=64), accumulate the chunk-
product Q_b = R_{bW} · … · R_{(b-1)W + 1} once per chunk (this is O(m²W) compute
and O(m²) memory per block, then prefix-scan the Q_b's across blocks (only B/W
of them) using a standard parallel-scan. Within a chunk, the per-token states
are recomputed from the chunk-start.

The chunkwise scheme requires a 64×64 rotation matrix multiply primitive at
each chunk boundary (chunked m×m GEMM). With m=1024, W=64, the chunk-product
is a single 1024×1024 matmul per chunk — cuBLAS handles this in microseconds.
This is the same pattern Mamba-2 uses; it gets us within Mamba-2's
asymptotic-throughput regime. Inference is purely recurrent (Q_b matters only
for training-parallelism), so inference tok/s is governed by the per-step
kernel above and should be competitive with LRU.

**LOC for the chunkwise scan: ~250 in `vesta_grp_chunkscan_forward` and
matching backward.** Backward uses the property that R^T = R^{-1} for
orthogonal matrices, so the reverse scan is a forward scan with negated
angles — no extra storage of intermediate states required (constant memory
in T for backward, like Mamba-2).

### 9.4 Numerical considerations

* **Orthogonality preservation.** Each Givens rotation is exact orthogonal to
  machine precision (cos²+sin²=1 by Pythagorean identity, modulo IEEE rounding).
  Composition of K Givens rotations in fp16/bf16 will drift slightly off the
  manifold over long T. We *do not* re-orthogonalize: the drift is bounded by
  K·eps_bf16 ≈ K · 8e-4 ≈ 0.1 at K=128 over a single step, but accumulates
  per-step. For T=2048 the worst-case orthogonality drift bound is K·T·eps ≈
  0.2, which is small enough that downstream layers absorb it. If empirically
  necessary, add a QR re-projection every 1000 steps (cheap, since it's
  one-shot).
* **State range.** With Λ = I and orthogonal R, ‖s_t‖₂ is conserved exactly
  (up to numeric drift). The state stays bounded for free, no normalization
  layer needed.
* **Angle saturation.** φ_max π/2 with tanh activation means most angles are
  near zero at random init (small w_k, small x_t). The recurrence is close
  to identity at init, which is the "orthogonal init" prescription the
  EALRMN Phase-1 ablation found load-bearing. We *get this for free* — there
  is no need for orthogonal-init tricks because the parameterization is
  orthogonal by construction.

### 9.5 Smoke-test sequence

After implementation:

```bash
# 1. Unit-test build
cd unit-tests/build && sh .configure.sh cuda
cd .. && bash test.sh nn-grp-rnn

# 2. B0 replication (Claim N3): single GPU, ~30 min
./research/grp_rnn_b0 --task needle --T 2048 --m 1024 --K 512 \
    --tanh-state off --seeds 5 --steps 5000
./research/grp_rnn_b0 --task needle --T 2048 --m 1024 --K 512 \
    --tanh-state on  --seeds 5 --steps 5000
# expected: val_loss(tanh off) ≤ val_loss(tanh on) / 100

# 3. N1 (expressivity): single GPU, ~8 hr
./research/grp_rnn_n1 --task a5-word --T 256 --m 256 --K 128 \
    --model grp-rnn --seeds 10
./research/grp_rnn_n1 --task a5-word --T 256 --m 256 --K 1 \
    --model deltanet --seeds 10
./research/grp_rnn_n1 --task a5-word --T 256 --m 256 \
    --model lru --seeds 10

# 4. N2 (throughput): single GPU, ~1 hr
./research/grp_rnn_throughput --T 2048 --m 1024 --batch 8 \
    --bench grp-rnn --bench mamba2
```

---

## 10. Failure modes

Three most likely ways GRP-RNN fails its own claims, with the symptom to watch
for.

### 10.1 Failure mode F1 — angles collapse to zero

**Symptom.** Training proceeds but tanh(w_k^T x_t + b_k) → 0 across all k as
training progresses, so the rotation R_t → I, and GRP-RNN degenerates to its
EMA / MEGA-core special case. Val loss matches `--grp-K=0` ablation; A_5 word
accuracy stays near chance.

**Why it could happen.** The L_CE gradient might prefer "do nothing" (R=I)
because every non-zero angle makes the state-trajectory noisier in early
training. Without L_rot-reg to push angles *outward*, the model can collapse
into the LTI sub-manifold. EALRMN's Phase-0c found exactly this kind of
"trivial-solution attractor" for the Koopman operator: identity-reg pulled K
toward I but the gradient toward I from L_recon dominated.

**Diagnostic.** Track the histogram of `|θ_k(x_t)|` across training. If
mean(|θ|) decays toward zero, F1 is active.

**Mitigation.** L_rot-reg with sign-flipped coefficient (push angles *away*
from zero) instead of the default toward-zero. Or, parameterize the angle as
`θ_k(x_t) = θ_k0 + tanh(...)` with `θ_k0 ≠ 0` initialized.

### 10.2 Failure mode F2 — K > 1 advantage does not materialize empirically

**Symptom.** N1 result: GRP-RNN at K=m/2 ties DeltaNet at K=1 on the A_5
task. Expressivity argument is correct in theory, but optimization cannot
realize the extra capacity.

**Why it could happen.** The K-Givens product factorization is "narrow" — each
plane-generator is a 2-dimensional subspace of so(m), and the gradient through
K compositions has K-times-amplified noise. DeltaNet's β=1 rank-2 update has
better-conditioned gradients because the entire rank-2 perturbation is one
unit. If the optimization landscape is dominated by gradient SNR, GRP-RNN's
finer parameterization may not help.

**Diagnostic.** Check the per-plane gradient norm distribution at the end of
training. If only ~3 of K planes have significant non-zero gradients, the
model is using GRP-RNN as if K=3 — confirming F2.

**Mitigation.** Reduce K to 8 or 16 (architectural simplification toward
DeltaNet) and re-run N1. If the gap *appears* at K small but not at K large,
the expressivity claim survives at small K. If the gap doesn't appear at any
K, the claim is falsified empirically (despite being correct theoretically).

### 10.3 Failure mode F3 — chunkwise scan is too slow

**Symptom.** N2 result: GRP-RNN training tok/s < 0.1 × Mamba-2.

**Why it could happen.** The chunkwise scan needs to materialize the chunk-
product Q_b ∈ R^{m×m} at every chunk boundary. At m=1024 this is a 4 MB
matrix per chunk × B/W chunks. At T=2048, W=64, that's 32 chunks × 4 MB = 128
MB per sequence. Within a batch of 8, that's 1 GB of intermediate state,
which is fine, but each chunk-product is a 1024^3 = 1.07 GFLOP GEMM. At
A100 312 TFLOP/s, that's 3.4 μs per chunk-product compute time, but the
**launch overhead** for the kernel chain (Givens product + scalar mul + GEMV
for input + chunk-end GEMM, all chained) is 4-8 μs per layer per step. The
launch overhead dominates.

**Diagnostic.** Profile with Nsight Compute: if total kernel-launch overhead
> 40% of wall-time, F3 is active.

**Mitigation.** Fuse the Givens-product kernel with the input-GEMV kernel
into a single launch (sacrificing some readability but eliminating ~half the
launches). Or, use CUDA graphs to pre-capture the per-chunk dependency graph
(this is what Mamba-2's reference impl does). Worst case: report N2 as
"expressivity-at-throughput-cost", honestly disclose that GRP-RNN is not yet
production-ready, and recommend it only for tasks where the diagonal-SSM
expressivity gap matters.

---

## 11. Why this is *not* an EALRMN rerun

The EALRMN program failed because it stacked 8 mechanisms behind a single
Lagrangian and could not isolate which (if any) was contributing. Phase-1's
ablation revealed the answer was "linear recurrence + orthogonal init alone,"
already known in the literature.

GRP-RNN's structure is deliberately the opposite:

* **One mechanism** (input-dependent product-of-Givens-rotations recurrence).
* **One loss term** (CE; the two regularizers are *ablation-only*).
* **One expressivity claim** (Theorem in §6, with the parameter restriction
  reductions in §5 making it clear which sub-manifold is the "novel" part).
* **Each flag is a clean ablation.** `--grp-disjoint-planes` reduces to LRU
  (no expressivity gap claimed). `--grp-K=0` reduces to EMA. `--grp-tanh-
  state` reduces to tanh-RNN (the strawman). `--grp-fixed-angles` removes
  input dependence. `--grp-K=1 --grp-plane-input-dep` reduces to DeltaNet.
  Each flag flip changes one bit of the architecture and isolates one claim.

The contrapositive of EALRMN's error is: **before any positive result, the
ablation must specify what reducing each flag tests.** The §5 reduction table
+ §7 ablation table do this exhaustively. Every claim in §8 names which flags
must be set to recover the claim, and which flags' ablations would falsify it.

If GRP-RNN's N1 result is `GRP-RNN > LRU at the A_5 task`, the literature
already predicts this (Merrill et al. 2024 §3 word-problem). The novel claim is
specifically `GRP-RNN > DeltaNet at the A_5 task at iso-param`, because that
is the comparison where the K>1 lifting earns its keep. If GRP-RNN ties
DeltaNet, we have not invented anything — we have re-parameterized DeltaNet,
which is what Phase-1 of EALRMN revealed about the prior program.

The framework is also deliberately *less* ambitious than EALRMN. It does
not claim a Pareto improvement over Transformers on language modeling at
production scale. It claims expressivity on one specific class of tasks
(non-solvable word-problems and their generalizations to compositional state-
tracking) and reasonable throughput. Whether that expressivity *transfers* to
language modeling is a separate empirical question explicitly outside the
scope of N1-N3.

---

## 12. What this candidate does NOT propose

To be transparent about scope (and to head off scope creep in the framework-
selection step that compares Candidates A, B, C):

* **No latent-prediction objective** (R2 axis). The prior program's R2
  failure mode (Phase-0a / 0b) is documented; this candidate does not retry.
* **No bounded associative memory** (R4 axis). The EALRMN Phase-0g/0k
  finding that 4-slot memory ties / loses to vanilla RNN at scale is
  respected.
* **No mixture of experts** (R5 axis). Out of scope.
* **No compute-adaptive inference** (R7 axis). Out of scope.
* **No entropy-adaptive chunking** (R1 axis). Out of scope.
* **No raw-token-vs-latent decoding switch** (R8 axis). Out of scope.

GRP-RNN sits *only* on axis R3 (recurrence type) and claims novelty *only*
relative to diagonal SSMs and to DeltaNet. The audit (`VESTA_AUDIT.md` §3.5)
identifies state-tracking expressivity as the place where "novelty has the
most plausible path"; this candidate takes that opening.

The other two candidates (B = causal information-bottleneck multi-particle
state; C = event-driven hierarchical compute) attack different axes and may
compose with GRP-RNN orthogonally. The framework-selection step decides
whether to *combine* them or *select* one. This document argues only for
GRP-RNN as a structurally novel mechanism on R3.

---

## 13. Implementability budget summary

| Item | LOC | Files |
|------|-----|-------|
| CUDA kernels (forward, backward, chunkscan, angles) | ~730 | 3 new files under `cuda/` |
| C++ host-side trainer extension | ~150 | `sgd_rnn.cpp` (extended) |
| Layer-info / network-type registry | ~70 | `network.h`, `hiddenlayerinfo.{h,cpp}` |
| API plumbing | ~30 | `main.{h,cpp}` |
| Unit tests | ~250 | `unit-tests/Backend/Machine Learning/nn-grp-rnn.cpp` |
| Smoke-test harnesses | ~400 | `research/grp_rnn_{b0,n1,n2}.cpp` |
| **Total** | **~1630 LOC** | 10 files (4 new, 6 extended) |

A subset of the kernel inventory exists already (`gpu_vesta.cu/.h` placeholders,
`gpu_stiefel.cu` for orthogonal structures, `gpu_blas` for cuBLAS plumbing).
The actual new-kernel writing is concentrated in `vesta_grp_kernels.cuh`
(Givens application + chunkwise reduction); the rest is plumbing.

Estimated implementation calendar time: ~2 weeks of focused engineering
(matches the "moderate engineering cost" target in the brief), plus
~1 week for N1/N2/N3 experiments.

---

## 14. Summary

**GRP-RNN.** State `s_t ∈ R^m`, transition `R_t = ∏_k G_k(θ_k(x_t)) ∈ SO(m)`
where G_k are Givens rotations in pre-chosen plane (p_k, q_k), angles
`θ_k(x_t) = tanh(w_k^T x_t + b_k) · φ_max`. With interlocking planes the
generated subgroup of SO(m) is the full SO(n) for n ≤ m, allowing non-
solvable-group word recognition that diagonal SSMs provably cannot achieve
in a single layer.

Subsumes input-dep LRU (disjoint planes), DeltaNet (K=1, plane-learned), EMA
(K=0), tanh-RNN (state-tanh ablation, expected to *fail*, replicates EALRMN
Phase-1 result). One loss term, two ablation-only regularizers, three
falsifiable claims with pre-commit interpretation tables. ~1000 LOC of new
CUDA on top of glades-ml's existing infrastructure.

The single empirical question this candidate poses: **does the K > 1 Givens
lifting beat DeltaNet (K=1) on A_5 word recognition at iso-param?** If yes,
GRP-RNN earns its novelty. If no, it reduces to "DeltaNet with extra
parameters that don't help" and is reported as a negative result.

— end VESTA Candidate A —
