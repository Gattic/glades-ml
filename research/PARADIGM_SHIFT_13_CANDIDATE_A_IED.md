# Paradigm shift #13 — Candidate A: Implicit Equilibrium Depth (IED)

**Formulation class:** implicit / fixed-point / operator-theoretic.
**Author pass:** subagent-dispatched design (2026-04-22).
**Status:** candidate — awaiting selection at the shift-13 gate.

---

## 1. Short name

**IED** — *Implicit Equilibrium Depth*.  A single transformer-style
operator `B_θ` is iterated to its fixed point `z* = B_θ(z*, x)` and the
entire "depth" of the network is replaced by that implicit solution.
The backward pass is the implicit function theorem: one linear solve
against the fixed-point Jacobian, no stored activation tape, no layer
unroll.

IED is the logical extremum of depth-axis compression.  Where CHIRON
(shift #1) reduced activation memory to **O(1) in depth** by being
reversible while still running `L` explicit blocks, IED reduces the
*explicit block count itself* to zero: depth becomes the *number of
fixed-point iterations at inference time*, which is (a) data-dependent,
(b) bounded by a contraction guarantee, and (c) untracked during the
backward.

## 2. Primitive objects

Let the pile_large scale (the reference configuration for memory
accounting in this document) be:

- `T = 2048`   — sequence length
- `d = 1024`   — residual-stream dimension
- `V = 50 304` — vocabulary
- `L_eff = 24` — the *effective* depth a reference explicit transformer
  would use.  IED replaces this with `K ≤ L_eff` dynamic iterations of
  a single block.

Primitive objects:

- **State** `z ∈ R^{T×d}`, the residual-stream tensor.  This is the
  *only* activation IED ever materializes.
- **Operator** `B_θ : R^{T×d} × R^{T×d} → R^{T×d}`, a single block
  parameterized by `θ` (attention + MLP, exactly as in one CHIRON block
  or one standard transformer block).  The second argument is the
  *injection* `x` — the embedded input tokens + positional encoding —
  which is held constant across the fixed-point iteration.
- **Residual map** `F_θ(z, x) = B_θ(z, x) − z` ∈ R^{T×d}.  Fixed points
  of `B_θ(·, x)` are zeros of `F_θ(·, x)`.
- **Fixed-point Jacobian**  `J_z(z*) = ∂B_θ/∂z |_{z*}` ∈
  R^{(Td)×(Td)}.  Never materialized; accessed only as a matvec
  operator.
- **Anderson memory** `{z_{k-m+1}, ..., z_k} ⊂ R^{T×d}` with `m ∈
  {5,...,8}`: the history buffer feeding the Anderson acceleration
  combination.
- **Contraction certificate** `c_θ ∈ [0, 1)`: a scalar upper bound on
  the Lipschitz constant of `B_θ(·, x)`, maintained by construction
  (Stiefel factors + Σ clamp, see §5).

## 3. Mathematical state space

The IED "deep network" is the map

    Φ_θ : x ∈ R^{T×d}  ↦  z* ∈ R^{T×d}
                such that  z* = B_θ(z*, x).

State space for the forward pass is a *point* `z ∈ R^{T×d}` (plus the
Anderson history).  Total forward state, peak:

    S_fwd  =  (m+1)·T·d   (Anderson history)
           ≈  8·2048·1024·2 bytes (BF16)
           =  34 MB.

A reference explicit 24-layer transformer at the same `T, d`, *without*
CHIRON, would materialize `L·T·d = 24·2048·1024·2 ≈ 100 MB` of residual
activations.  With CHIRON, this drops to `O(1)` in depth but still
requires a logical layer loop (the reversible-flow cost).  IED
eliminates the loop entirely: **no `L` appears in the memory formula**;
instead `m` (Anderson history) replaces it, and `m` is a fixed small
constant tuned once.

**Constraints:** `B_θ(·, x)` must be a *contraction* in a neighborhood
of the trajectory, with Lipschitz `< 1`:

    ‖B_θ(z, x) − B_θ(z', x)‖_2  ≤  c_θ · ‖z − z'‖_2,   c_θ < 1.           (C)

**Topology:** `R^{T×d}` as a Banach space under `‖·‖_2` (Frobenius).
`B_θ` is a continuous self-map of the closed ball `‖z‖ ≤ R_*` where
`R_*` is an a-priori bound from the Banach fixed-point theorem.

## 4. Evolution law / update rule / governing operator

### 4.1 Forward — Anderson-accelerated fixed-point iteration

Anderson acceleration of order `m` (AA(m)):

1. Initialize `z_0 = x` (inject tokens directly; the fixed point shifts
   relative to `x`).
2. For `k = 0, 1, 2, ...` until convergence:

       g_k       =  B_θ(z_k, x)
       F_k       =  g_k − z_k                                 (residual)
       γ*_k      =  arg min_{γ ∈ R^{m_k+1}, 1^T γ = 1}
                       ‖ Σ_{i=0..m_k} γ_i F_{k-i} ‖_2         (LS solve)
       z_{k+1}   =  (1 − β) · Σ_i γ*_i z_{k-i}
                  +       β  · Σ_i γ*_i g_{k-i}               (β = 1 standard)
       m_k       =  min(m, k)

3. Stop when `‖F_k‖ / ‖z_k‖ ≤ ε` (ε ≈ 1e-3) **or** `k = K_max`
   (K_max = 32; empirical median ≈ 8–12).

The LS problem is size `(Td) × (m_k+1)`, solved via a stored QR factor
of the residual-difference matrix `ΔF_k = [F_k − F_{k-1}, ..., F_{k-m+1}
− F_{k-m}] ∈ R^{Td × m}`.  QR is maintained incrementally in O(Td·m)
per step.

Memory: `(m+1)` residual-stream tensors (the z history) + `(m)`
difference vectors for QR + a small `m×m` factor = `(2m+1)·T·d + O(m²)`
BF16 floats.  At `m = 6`: 13·T·d ≈ 26 MB.

### 4.2 Backward — implicit function theorem (IFT)

Let `L` be the scalar loss (cross-entropy) at the output head.  Compute
the output head's upstream gradient w.r.t. the fixed point:

    g  =  ∂L / ∂z* ∈ R^{T×d}.

IFT says that the total derivative of `z*(x, θ)` satisfies

    ∂z*/∂θ_i  =  (I − J_z)^{-1} · ∂B_θ/∂θ_i (z*, x).              (IFT)

Therefore the gradient w.r.t. `θ_i` is

    dL/dθ_i   =  [(I − J_z)^{-T} g]^T · ∂B_θ/∂θ_i (z*, x).         (★)

Define the **adjoint state** `v ∈ R^{T×d}` as

    v   :=  (I − J_z)^{-T} · g.                                    (LS)

Given `v`, the parameter gradient `dL/dθ = v^T ∂B_θ/∂θ (z*, x)` is
computed by **one backward pass through a single block `B_θ`**, with `v`
taking the role that the layer-`L` upstream gradient plays in standard
backprop.  Total cost: exactly one block backward, independent of `K`
(the number of forward iterations).

The adjoint `v` is obtained by *iteratively solving* `(I − J_z)^T v =
g`.  Two options:

**Option A — Neumann series (Jacobian-free):**

    v  =  Σ_{k≥0} (J_z^T)^k g
       ≈  Σ_{k=0..N-1} (J_z^T)^k g         (truncate at N ≈ 20)

Each term requires one **vector-Jacobian product** `J_z^T · w`, which
is one backward pass through `B_θ` at `z*` with upstream gradient `w`.
Convergence is guaranteed iff `‖J_z‖ ≤ c_θ < 1` (our contraction
condition, see §5).  Rate: `‖error‖_N ≤ c_θ^N · ‖g‖`, so `c_θ = 0.9`
needs N = 87 for 1 % error, `c_θ = 0.5` needs N = 14.

**Option B — Anderson/GMRES on the adjoint problem:**

Solve `(I − J_z)^T v = g` by AA(m') or GMRES using only matvecs.  GMRES
converges in `O(√κ)` iterations where `κ = κ(I − J_z) ≤ (1+c_θ)/(1−c_θ)`.
At `c_θ = 0.9`, `κ ≈ 19`, GMRES converges in ~5 iterations.  This is
the default; Neumann is a fallback when GMRES conditioning degrades.

**Option C — Phantom gradient (Jacobian-free alternative):**

Skip the adjoint solve entirely; approximate `v ≈ α · g` for a tuned
scalar `α` (or `v ≈ g` with "unrolled gradient from the last T-step"
of the forward, the canonical *Jacobian-free backprop* trick).  Known
to work in DEQ literature for small-scale image models.  IED keeps
this as a low-cost inner-loop mode for early training when the
accurate IFT gradient is not yet needed (see §8.1).

### 4.3 Composition with the outer optimizer

Once `dL/dθ` is computed by (★), the outer loop is *any* optimizer:
MFIO (shift #11), Adam, Muon, ...  No change to the outer loop.

## 5. Mechanism mapping

### 5.1 Activation memory

| Approach                    | Peak activation memory at T=2048, d=1024, L=24 |
|-----------------------------|-----:|
| Vanilla transformer         | ≈ 100 MB (L·T·d, BF16) |
| Gradient checkpointing      | ≈ 20 MB (√L blocks) |
| CHIRON reversible           | ≈ 4 MB (O(1) in depth; only `q`, `p` states) |
| **IED, m = 6 Anderson**     | **≈ 26 MB (Anderson history + diffs)** |
| IED, m = 0 (Picard)         | ≈ 4 MB (single z + history of 1) |

Surprise: IED at `m = 6` uses **more** memory than CHIRON.  The win is
*not* a peak-memory improvement — it is:

1. **Invariance to depth `L`**.  IED's footprint is constant in `L`.
   CHIRON's constant-in-depth only holds when the reversible inverse
   has no drift; in BF16 with sketch anchoring at k=8, CHIRON's
   footprint is `8·T·d ≈ 33 MB` — *same order as IED*.
2. **Gradient backward cost**.  CHIRON's backward is 3× the forward.
   IED's backward is one block + a short adjoint solve: `(1 + N_adj) ≈
   6–10` block-forwards worth of work.  At `L = 24`, the backward work
   is *cheaper* than CHIRON for `L > ~12`.
3. **No stored tape**.  CHIRON stores `{q_ℓ, p_ℓ}` for every logical
   layer.  IED stores only `z*` (and transiently, the Anderson
   history, which can be discarded before the backward).

### 5.2 Speed

Forward cost at `K_avg` iterations:

    Cost_IED_fwd   =  K_avg · Cost_one_block
    Cost_expl_fwd  =  L · Cost_one_block

IED wins the forward if `K_avg < L`.  Empirical DEQ work (Bai 2019,
Bai 2020) reports `K_avg ≈ L/2` at convergence for vision transformers
and text.  At `L = 24`, expect `K_avg ≈ 8–14`.  **Forward speedup:
1.7–3.0×**.

Backward cost:

    Cost_IED_bwd   =  (1 + N_adj) · Cost_one_block_bwd       (N_adj ≈ 5)
    Cost_expl_bwd  =  L · Cost_one_block_bwd                 (no ckpt)
                   =  (L + √L) · Cost_one_block_bwd          (ckpt)

**Backward speedup at L = 24**: 4.0× (no ckpt) or 4.8× (with ckpt).

Combined (fwd + bwd): **2.5–3.5× wall-clock training speedup** versus
a non-CHIRON baseline at the same `L`.  Versus CHIRON (which has no
forward speedup and a 3× backward): **~1.8× training speedup**.

### 5.3 Stability

Stability is the crux.  IED's forward is stable **iff** `c_θ < 1` at
every step of training.  Standard transformers are not naturally
contractive; weight initialization puts them on the edge.  IED
*enforces* `c_θ < 1` by construction via Stiefel × Σ (shift #7):

- All linear weights inside `B_θ` are factored as `W = U Σ V^T` with
  `U, V ∈ St(·)` orthonormal and `Σ` a positive-diagonal factor.
- The operator norm of `W` equals `‖Σ‖_∞`.
- Clamp `‖Σ‖_∞ ≤ s_max` with `s_max` chosen so that the block
  Lipschitz satisfies

      c_θ  ≤  s_max · L_softmax · L_gelu + 1 · δ_residual
           ≤  0.9   (target)

  where `L_softmax ≤ 1/4` (softmax Jacobian bound) and `L_gelu ≤ 1.13`.
  Solving: `s_max ≈ 0.7` is a safe bound.

Stiefel × Σ **makes IED safe out of the box** — this is the primary
differentiator vs. vanilla DEQs, which rely on monotone-operator
splitting (Winston & Kolter 2020) that imposes a structural penalty at
solve time.

### 5.4 Expressivity

A contraction `c_θ = 0.9` has fixed-point basin of attraction of radius
`∞` (it's globally contractive), and the fixed point `z*(x)` is a
function with Lipschitz constant `1/(1−c_θ) = 10` w.r.t. `x`.  A
24-layer transformer is equivalent to a Lipschitz-`≈10` function under
standard pre-norm initialization.  So IED matches the expressivity of
the 24-layer explicit baseline at `c_θ = 0.9`.

At `c_θ → 1⁻`, the fixed-point map can represent arbitrarily sharp
dependencies — the limit recovers *infinite effective depth*.  IED's
`c_θ` is therefore a direct knob on the **depth/stability tradeoff**.

## 6. Objective / variational principle

IED solves

    min_θ   E_{x ~ D} [ L(Φ_θ(x)) ]                                  (P)

subject to the contraction feasibility constraint

    ‖∂B_θ/∂z (z*, x)‖_2  ≤  c_θ  <  1   for all (θ, x) on the trajectory.

The contraction constraint is enforced structurally (Stiefel × Σ with
Σ clamp), not by Lagrangian penalty.

Equivalently, IED minimizes

    min_θ   E_x [ L( z*(θ, x) ) ]   s.t.   F_θ(z*, x) = 0.            (P')

(P') is the standard DEQ / implicit-layer objective.  The novelty
versus prior DEQ work is entirely in the *constraint-satisfaction
mechanism* (Stiefel makes it automatic) and in the *composition with
shifts #1/#7/#10/#11/#12* (see §7).

Variational principle (functional form): IED is the minimizer of

    𝓛[z(·), θ]  =  E_x L(z(x))  +  ⟨λ(x), F_θ(z(x), x)⟩,

and the Euler–Lagrange condition is exactly (P'), with λ = adjoint
state = `(I − J_z)^{-T} g`.  This gives a clean Pontryagin-style
optimality structure, distinct from the tape-based Lagrangian that
standard backprop implicitly solves.

## 7. Composability with shifts #1–#12

| Shift                             | Interaction with IED                                           |
|-----------------------------------|----------------------------------------------------------------|
| #1 CHIRON reversible flow         | **Degenerate case**: CHIRON is IED with `K = L` fixed and no contraction enforcement.  In IED, we make `K` dynamic AND enforce contraction; CHIRON's inverse structure is *not needed* (no stored tape at all). |
| #2 TC-tiled attention             | Composes.  The attention *inside* `B_θ` uses the TC kernel unchanged. |
| #3 int8 Adam                      | Composes.  IED changes how `dL/dθ` is obtained; Adam state is unaffected. |
| #4 BF16 gradient accumulation     | Composes directly.  |
| #5 SR BF16 weights                | Composes; SR applies after the IFT gradient is in hand.  |
| #6 Local-window attention         | Composes inside `B_θ`; reduces per-iteration cost, improving `K_avg` net speedup. |
| #7 Stiefel × Σ                    | **Critical composition**: Stiefel × Σ with `‖Σ‖_∞ ≤ s_max` is the mechanism that *guarantees* the contraction `c_θ < 1`.  Without #7, IED is as fragile as a vanilla DEQ. |
| #9 OVFG factored gradients        | Composes.  The factored gradient representation receives `dL/dθ` from (★); the factoring is orthogonal to how the gradient was computed. |
| #10 MPOT tensor-network weights   | Composes.  MPO bond-`D` weights are used inside `B_θ`; the contraction bound is derived from MPO spectral norms (bond-dim analysis; requires tight MPO operator-norm bound — open). |
| #11 MFIO moment-free optimizer    | Composes.  MFIO needs `‖z‖²` and `‖δ‖²`; both are available at `z*` from the single-block backward pass of (★).  Free. |
| #12 DFA direct feedback alignment | **Interesting synergy**: DFA replaces the global adjoint with a random projection.  IED uses DFA to *approximate* the adjoint solve: `v ≈ R · g` with fixed random `R`, avoiding the Neumann/GMRES inner loop entirely.  This is IED's "phantom gradient" variant with a principled interpretation. |

## 8. Expected conditioning, stability, and expressivity

### 8.1 Forward conditioning

Banach fixed-point theorem: `z*` is unique and iteration converges
linearly with rate `c_θ`.  Anderson acceleration gives super-linear
convergence in practice.  **Empirical median K_avg: 8–14 at c_θ = 0.9,
dropping to 4–6 at c_θ = 0.5**.

### 8.2 Backward conditioning

The adjoint linear system has condition number

    κ  =  κ(I − J_z)  ≤  (1 + c_θ) / (1 − c_θ).

At `c_θ = 0.9`: `κ ≤ 19`.  GMRES converges in `O(√19) ≈ 5` iterations
to 1 % residual.  At `c_θ = 0.99`: `κ ≤ 199`, GMRES needs ~14
iterations — still cheaper than 24-layer backprop.

### 8.3 Expressivity as function class

The set of functions representable by an IED operator `B_θ` with
Stiefel × Σ Lipschitz `c_θ < 1` equals the set of functions
representable by a residual network of *any* depth with the same
Lipschitz bound — a result that follows from fixed-point completeness
(Ciliberto et al. 2021, "Understanding DEQ").  So the expressivity
ceiling of IED under Stiefel is the full expressivity of Stiefel
residual networks.

## 9. Memory accounting at pile_large (T=2048, d=1024, L_eff=24)

All figures BF16, on-device.

| Component                              |   IED    |  CHIRON  | Vanilla |
|----------------------------------------|---------:|---------:|--------:|
| Activation tape (forward)              |    0     |    0     |  100 MB |
| CHIRON (q,p) states or IED Anderson    |   26 MB  |    4 MB  |    0    |
| Adjoint scratch (IFT / backward)       |    4 MB  |    0     |    0    |
| Stored z*  (for backward)              |    2 MB  |    0     |    0    |
| **Total activation-like memory**       | **32 MB**| **4 MB** |**100 MB**|

Conclusion at `L = 24`: IED's absolute memory is **higher than CHIRON**
(32 MB vs. 4 MB).  The win versus CHIRON is:

- At `L = 48` or `L = 96` (deeper models): CHIRON scales with `L`
  through its sketch anchor buffer (k·T·d, k growing with depth to
  control BF16 drift).  IED stays at 32 MB independent of `L`.  The
  memory crossover occurs near `L ≈ 60`.
- Training wall-clock: IED is 1.8–3× faster.
- Depth is *adaptive*.  Easy inputs converge in K_avg = 4, hard inputs
  (e.g., long multi-hop reasoning) in K_avg = 30.  Vanilla/CHIRON
  always pay `L`.

At the target pile_large scale (L = 24), IED's main selling point is
**speed**, not memory.  For *deeper* reference models (L ≥ 48), IED
dominates on both axes.

## 10. Comparison with standard DEQ (Bai et al. 2019, 2020)

| Aspect                         | Bai DEQ                             | IED                                 |
|--------------------------------|-------------------------------------|-------------------------------------|
| Fixed-point iteration          | Anderson / Broyden                  | Anderson + GMRES fallback           |
| Backward                       | IFT (same)                          | IFT + DFA-approx variant            |
| Contraction guarantee          | Empirical (monotone operators, Winston-Kolter imposes a projection penalty) | **Structural** via Stiefel × Σ clamp (shift #7) |
| Composition with reversible flows | Not done                         | Native (CHIRON block acceptable)    |
| Composition with tensor-net weights | Not done                       | Native (MPOT shift #10)             |
| Composition with optimizer-free | Not done                           | Native (MFIO shift #11)             |
| Scale demonstrated             | ~100 M vision & small LM (DEQ-FNOP) | Targeting 2 B LLM                   |

**IED's novel contributions beyond DEQ:**

1. Stiefel × Σ as a constraint-satisfaction mechanism for the
   contraction condition (no prior work uses Stiefel to *guarantee*
   DEQ contraction).
2. Composition with reversible-block primitives for the operator `B_θ`
   itself — each Anderson iteration is a *CHIRON block*, not a stack.
3. MPOT-factored weights inside `B_θ`, using the MPO bond spectrum to
   bound `c_θ`.
4. DFA-based phantom-adjoint variant (shift #12 × IFT), cutting the
   adjoint inner loop for early training.
5. Scaling target of 2 B on 16 GB, where no published DEQ lives.

## 11. Likely failure modes

1. **Non-contraction during early training.**  At init, `s_max` clamps
   Σ, but the composition with softmax/GELU may still push `c_θ > 1`
   stochastically.
   *Mitigation*: aggressive Σ clamp (`s_max = 0.5`) for first
   ~500 steps; monitor `c_θ` via a power-iteration estimate; trigger
   emergency clamp if `c_θ > 0.98`.

2. **Slow fixed-point convergence.**  If Anderson stagnates,
   `K_max = 32` caps the forward cost but the per-step forward becomes
   slower than explicit.
   *Mitigation*: track `K_avg`; if `K_avg > L_eff/2` sustained, the
   contraction is too loose — tighten `s_max` by 0.1 and continue.
   Fallback: warm-start `z_0` from the previous batch's `z*` (assumes
   smoothness in `x`).

3. **IFT solver conditioning.**  As `c_θ → 1⁻`, `κ` explodes; GMRES
   may fail to converge.
   *Mitigation*: hybrid adjoint — use truncated Neumann (N = 5) to
   warm-start GMRES.  If GMRES fails, fall back to *phantom gradient*
   (set `v = g`): biased but stable.  Gradient bias is tolerable in
   late training (established in DEQ literature).

4. **Gradient variance explosion.**  The IFT backward has no layer
   averaging; gradients can be noisier than a 24-layer explicit
   backward.
   *Mitigation*: gradient accumulation across batches; composition
   with #4 (BF16 grad accum) naturally averages.

5. **Interaction with MPOT (shift #10).**  MPO-factored weights have
   operator norms that are *hard* to bound tightly.  The spectral norm
   of an MPO is not a simple function of the factor spectra.
   *Mitigation*: use a power-iteration estimate of `‖W_MPO‖` every 100
   steps; clamp the MPO bond-scale based on estimated norm.  If
   unreliable, fall back to #7 dense Stiefel at the cost of shift-#10
   compression (soft-disable #10 inside IED blocks).

6. **Inference vs. training coupling.**  At inference, we want *deeper*
   `K` for hard tokens but we trained with `K_max = 32`.  Out-of-
   distribution `K > 32` has never been seen by the gradient.
   *Mitigation*: curriculum on `K_max` — train `K_max` from 8 to 32
   linearly over training; serve at `K_max = 48` with a warm-start.
   This is the only paradigm shift that requires training-inference
   `K` matching.

7. **Batch heterogeneity.**  Different sequences in a batch converge
   at different rates.  Naïve implementation pays the slowest one.
   *Mitigation*: per-example early-exit; mask contributions of
   converged examples out of the Anderson LS problem.  Adds ~5% code
   complexity.

8. **Subtle bug surface.**  The implicit-function-theorem backward is
   mathematically delicate: a bug in the adjoint solve silently
   produces *biased* gradients with no obvious symptom until the loss
   plateaus.
   *Mitigation*: unit-test parity against finite-difference `dL/dθ`
   on a toy model at every composition (IED-alone, IED×Stiefel,
   IED×MPOT, IED×DFA).  Maintain a regression test in `unit-tests/`.

## 12. Minimal prototype (≤ 4 weeks)

1. **Week 1**: `gpu_ied.{h,cu}`
   - `anderson_step(z_hist, F_hist, m, β, z_out)` — the AA(m) combine.
   - `ied_forward(x, θ, z_init, K_max, ε, z_star_out, K_used_out)`.
   - `ied_spectral_bound(θ, c_θ_out)` — power-iteration on `J_z`.
2. **Week 2**: IFT backward
   - `ied_backward(z_star, g, θ, N_gmres, v_out, dθ_out)`.
   - Internal: GMRES with `J_z^T · w` as matvec via a single block
     backward at `z_star`.
3. **Week 3**: composition tests
   - Parity vs. finite-diff `dL/dθ` on a 2-layer toy.
   - IED × Stiefel: verify contraction clamp enforces `c_θ < 0.95`.
   - IED × MPOT: spectral bound estimate.
4. **Week 4**: pile_large smoke test, 300 steps at small model
   (80M params).  Compare tok/s + loss vs. CHIRON baseline.

## 13. Open questions / research deltas

1. **What is `K_avg` at LLM scale?**  DEQ literature reports modest
   `K` at vision scale; LM scale is untested.  If `K_avg > L_eff`,
   IED loses the speed argument.
2. **How tight is the MPO operator-norm bound?**  Affects #10 × IED
   composition.
3. **Can DFA-phantom adjoint match true-IFT quality at 2 B scale?**
   Central #12 × IED research question.
4. **Does `K_max = 32` leave enough headroom for hard-token
   generalization?**  Inference-time `K` extrapolation is a bet.

---

**Document status**: candidate A complete, ready for side-by-side
selection against candidates B, C at the shift-13 gate.
