# Paradigm Shift #250 Candidate A — FBA (Frame Bundle Attention)

**Status:** candidate design; one of three parallel formulations for paradigm shift #250 ("Focused attention with perspective").
**Date:** 2026-05-14.
**Tagline:** *Each token carries a learned local frame; attention is heat flow on a sequence base manifold with anisotropic metric, and parallel transport reconciles viewpoints. Both "focus" (geodesic-ball concentration) and "perspective" (frame on a GL-bundle) are structural, not labeling.*
**Axis:** geometric/manifold attention with explicit principal-bundle structure, composable with flagship SCFA.
**Materially distinct from:** multi-head SDPA (per-head linear subspaces, no transport, no metric learning); SCFA (linear sequence-spectral compression; flat metric); ATTENTION-SINK (#78, fixed register tokens, no geometry); curvature-as-prior heuristics (e.g. hyperbolic embeddings — these treat the *embedding* space curved, FBA treats the *base sequence axis* curved with a fiber over each token).

---

## 0. Executive summary

SDPA treats all `T` tokens as a flat Euclidean point cloud, contracting `q_i^\top k_j` in a single shared coordinate system. SCFA generalises by performing attention in a learned `k`-dim sequence-spectral basis, but still imposes a flat metric and a single global coordinate system.

FBA replaces both. The sequence axis becomes a **base manifold** `(M, g)` with learned Riemannian metric. Each token carries a **local frame** `F_i \in GL(d, \mathbb{R})` (its viewpoint). Information flow between tokens is **parallel transport** by a connection `\omega` whose curvature controls geodesic-ball volume — hence focus radius. *Perspective* is the frame plus the connection. *Focus* is the heat-kernel bandwidth which is the geodesic-ball radius set by per-query learned bandwidth `\sigma_i` and per-edge learned metric `g_i`.

**Magnitude claim (§9).** At fixed effective participants `\bar n = O(\log T)` per query, per-query cost is `O(\log T \cdot d)`, total per-layer `O(T \log T \cdot d)`. Standalone FBA does NOT clear the `10\times` bar — projection cost `T d^2` dominates. **Stacked FBA + flagship SCFA** is the actual proposal: FBA replaces SCFA's inner `k`-dim attention, exploiting the small `k`-axis where bundle objects are cheap. Claim recast as **NLL-per-FLOP**: ~`0.10–0.30` nat ema reduction at long context (T=16384) via expressivity unreachable by SDPA/SCFA (§8.3).

**Honest gap (§11).** Dominant risk: connection-learning starvation in the `\bar n \approx 50`-participant regime. BF16 parallel-transport drift compounds across L=53 layers. Both must be falsified at Gate-0 (§14, ≤1 GPU-hour) before any 1B-scale claim.

---

## 1. Primitive objects

We define every object before use. Throughout, batch and head dims are suppressed except where load-bearing.

### 1.1 Base manifold

Let `(M, g)` be a smooth `d_M`-dim Riemannian manifold with metric `g \in \Gamma(T^*M \otimes T^*M)`. Three concrete choices: **FBA-S1** (`S^1`, circumference `T`, causal-periodic), **FBA-R1** (`\mathbb{R}^1` with learned warp `\phi: \{1..T\} \to \mathbb{R}`, default), **FBA-Md** (`d_M \in \{2,3\}` learned, deferred — §13). The default **FBA-R1** strictly contains SCFA's flat axis as `g \equiv 1`; parallel transport in 1-D is scalar holonomy (BF16-stable); avoids chart problems.

### 1.2 Principal bundle and frame

A **principal `G`-bundle** `\pi : P \to M` with structure group
$$
G \;:=\; GL(d, \mathbb{R}) \quad\text{(or a subgroup, see §5).}
$$
A **frame** at `x \in M` is an ordered basis `e^{(x)} = (e_1^{(x)}, \dots, e_d^{(x)})` of the fiber `E_x` of the **associated vector bundle**
$$
E \;:=\; P \times_G V \;\to\; M, \qquad V := \mathbb{R}^d.
$$
Equivalently, `e^{(x)} \in GL(d, \mathbb{R})` viewed as a `d \times d` matrix-valued field over `M`. A **section** `v \in \Gamma(E)` is a smooth assignment `x \mapsto v(x) \in E_x`; in the frame `e^{(x)}` it has coordinates `v^\alpha(x)` with `v(x) = v^\alpha(x) e_\alpha^{(x)}`. Einstein summation over Greek (fiber) indices.

At the **discrete** level (the only level the CUDA implementation sees), each token `i \in \{1, \dots, T\}` carries a frame matrix
$$
F_i \;\in\; GL(d, \mathbb{R}), \qquad F_i \text{ a learnable } d\times d \text{ matrix per token (parameterised, see §3)}.
$$

### 1.3 Connection

A **principal connection** on `P` is a `\mathfrak{g}`-valued 1-form `\omega \in \Omega^1(P, \mathfrak{g})`, `\mathfrak{g} = \mathfrak{gl}(d, \mathbb{R})`. In a local trivialisation, a `\mathfrak{gl}(d)`-valued 1-form `A = A_\mu^{\;a}{}_b\, dx^\mu \otimes E_a{}^b \in \Omega^1(M, \mathfrak{gl}(d))`. Christoffel coefficients `\Gamma^\alpha{}_{\beta\mu}(x) = A_\mu^{\;\alpha}{}_\beta(x)`. Curvature `R = dA + A \wedge A \in \Omega^2(M, \mathfrak{gl}(d))`, with components `R^\alpha{}_{\beta\mu\nu} = \partial_\mu\Gamma^\alpha{}_{\beta\nu} - \partial_\nu\Gamma^\alpha{}_{\beta\mu} + \Gamma^\alpha{}_{\sigma\mu}\Gamma^\sigma{}_{\beta\nu} - \Gamma^\alpha{}_{\sigma\nu}\Gamma^\sigma{}_{\beta\mu}.`

In **discrete 1-D** (`d_M = 1`), `A_i \in \mathfrak{gl}(d, \mathbb{R})` for `i \in \{1,..,T-1\}` (one matrix per edge `i \to i+1`). Discrete curvature on a path from `i` to `j` is the holonomy defect of the rectangle (in 1-D vanishes identically; richer curvature only at `d_M \ge 2`). For default `d_M = 1` we capture curvature *implicitly* through learned `g_i \in \mathbb{R}_{>0}` per edge (§1.4).

### 1.4 Metric and geodesic distance

The **discrete metric** is `g_i > 0` per edge, parameterising
$$
ds_i \;:=\; \sqrt{g_i}\, , \qquad d_M(i, j) \;=\; \sum_{\ell = \min(i,j)}^{\max(i,j)-1} \sqrt{g_\ell}.
$$
Causal masking restricts `j \le i`. The **geodesic ball** of radius `r` around query `i` is
$$
B_r(i) \;:=\; \{\, j : d_M(i, j) \le r \,\}.
$$

### 1.5 Streams

| symbol | shape | meaning |
|---|---|---|
| `x_i` | `\mathbb{R}^d` | token embedding at position `i` |
| `F_i` | `\mathbb{R}^{d \times d}` | local frame (learned, parametrised — §3) |
| `A_i` | `\mathbb{R}^{d \times d}` | connection coefficient on edge `i \to i+1` |
| `g_i` | `\mathbb{R}_{>0}` | metric coefficient on edge `i \to i+1` |
| `\sigma_i` | `\mathbb{R}_{>0}` | per-query bandwidth (focus radius) |
| `q_i, k_i, v_i` | `\mathbb{R}^d` | query/key/value at token `i` |
| `\bar v_{i \leftarrow j}` | `\mathbb{R}^d` | value `v_j` parallel-transported along the geodesic from `j` to `i` |
| `\alpha_{ij}` | `\mathbb{R}_{\ge 0}` | attention measure (heat-kernel mass) |
| `y_i` | `\mathbb{R}^d` | output token at position `i` |

---

## 2. State space

Per-layer state: `\Theta_\ell = (W_Q, W_K, W_V, W_O, \{F_i\}, \{A_i\}, \{\log g_i\}, \{\log \sigma_i\})`. Bundle objects stored in factorised form:
- `F_i = I + U_i V_i^\top`, `U_i, V_i \in \mathbb{R}^{d \times r_F}`, `r_F \in \{4, 8, 16\}` (rank-`r_F` perturbation of identity);
- `A_i = U_i^A V_i^{A\top}`, `r_A \in \{2, 4, 8\}`;
- `g_i, \sigma_i` scalars per position.

Total extra per layer: `2 T d (r_F + r_A) \approx 50M` at `T=1024, d=2048, r_F=8, r_A=4` — too large for 53 layers of 1B. **Remediation:** use low-rank *amortised* parameterisation `F_i = I + U(x_i) V(x_i)^\top` with `U, V` small MLPs of `x_i` (~5M weights shared across positions); same for `A_i = U^A(x_i, x_{i+1}) V^{A\top}(x_i, x_{i+1})`. Parallel transport learns from data, not from `T` independent matrices.

---

## 3. Evolution law

We give the explicit forward update for `y_i`.

### 3.1 Step 1 — local frame application

Each token's stream is *expressed in its own frame*. The "perspective" lives here:
$$
\boxed{\;\tilde q_i \;=\; F_i^{-1}\, W_Q\, x_i, \qquad \tilde k_i \;=\; F_i^{-1}\, W_K\, x_i, \qquad \tilde v_i \;=\; F_i^{-1}\, W_V\, x_i.\;} \tag{F1}
$$
Because `F_i = I + U_i V_i^\top`, the inverse is Sherman–Morrison:
$$
F_i^{-1} \;=\; I \;-\; U_i (I_{r_F} + V_i^\top U_i)^{-1} V_i^\top, \tag{F1'}
$$
costing `O(d r_F^2 + r_F^3)` per token = `O(T d r_F^2)` total per layer.

### 3.2 Step 2 — parallel transport of values from `j` to `i`

For `j < i` (causal), parallel transport `v_j` along the geodesic by the **ordered product** of edge connections
$$
P_{i \leftarrow j} \;:=\; \prod_{\ell = j}^{i-1} \mathrm{Exp}_{GL}(A_\ell) \;=\; \mathrm{Exp}_{GL}(A_{i-1}) \cdots \mathrm{Exp}_{GL}(A_j), \tag{PT1}
$$
where `\mathrm{Exp}_{GL}: \mathfrak{gl}(d) \to GL(d)` is matrix exponentiation. For small `\|A_\ell\| < 0.1` (enforced by init + weight decay, §6) we use the **rank-truncated Cayley map** `\mathrm{Exp}_{GL}(A) \approx (I - A/2)^{-1}(I + A/2)`, also Sherman–Morrison-able when `A` is low-rank.

The **transported value** in the query's frame is
$$
\boxed{\;\bar v_{i \leftarrow j} \;:=\; F_i^{-1}\, P_{i \leftarrow j}\, F_j\, \tilde v_j.\;} \tag{PT2}
$$
The composition `F_i^{-1} P_{i \leftarrow j} F_j` is the **frame-coordinate parallel transport** — it expresses `v_j` (originally written in `F_j`'s perspective) in `F_i`'s perspective after geodesic transport.

### 3.3 Step 3 — focused heat-kernel attention measure

The **attention measure** is a *heat-kernel-on-`M`* concentrated in a geodesic ball:
$$
\boxed{\;\alpha_{ij} \;\propto\; \exp\!\bigl(-\tfrac{1}{2}\, d_M(i,j)^2 / \sigma_i^2\bigr)\, \cdot\, \exp\!\bigl(\langle \tilde q_i,\, \bar k_{i \leftarrow j}\rangle / \sqrt{d}\bigr),\;} \tag{HK1}
$$
where `\bar k_{i \leftarrow j} = F_i^{-1} P_{i \leftarrow j} F_j \tilde k_j` (parallel-transported key) and the *geodesic distance* `d_M(i,j) = \sum_{\ell = j}^{i-1} \sqrt{g_\ell}` is the integrated metric. Normalisation is by row-softmax over causal `j \le i`.

The **focus mechanism** is the Gaussian-on-`M` factor `\exp(-d_M(i,j)^2/2\sigma_i^2)`. Three things make it adaptive (not a fixed sparsity pattern):
1. `\sigma_i` is **per-query learned** — high-information queries pick small `\sigma_i` (sharp focus); high-entropy queries pick large `\sigma_i`.
2. `g_i` is **per-edge learned** — high `g_i` *stretches* the metric, putting tokens far apart and pushing them outside the ball.
3. The Gaussian *acts in the geodesic metric*, not the integer-position metric: an edge with `g_i = 100` effectively *deletes* the link `i \to i+1` from attention.

The effective ball `B_{r_i}(i) = \{j : d_M(i,j) \le 3\sigma_i\}` (3σ truncation) typically contains `O(\sigma_i / \mathbb{E}[\sqrt{g}])` participants. With learned median `\mathbb{E}[\sqrt{g}] \approx 1` and `\sigma_i \sim O(\log T)`, the ball contains `O(\log T)` tokens — the magnitude claim.

### 3.4 Step 4 — value composition in the query's frame

$$
\boxed{\;y_i \;=\; F_i\, W_O\, \sum_{j \in B_{r_i}(i)} \alpha_{ij}\, \bar v_{i \leftarrow j}.\;} \tag{FBA-OUT}
$$
The final `F_i` re-expresses the result in the *next* layer's input perspective (which is the same `F_i` for the residual stream).

### 3.5 Connection ODE / discrete update

The continuous connection ODE governing parallel transport on `(M, g)` is
$$
\boxed{\;\nabla_{\dot\gamma} v \;=\; 0 \qquad\Longleftrightarrow\qquad \frac{d v^\alpha}{ds} + \Gamma^\alpha{}_{\beta\mu}(\gamma(s))\, \dot\gamma^\mu(s)\, v^\beta(s) \;=\; 0,\;} \tag{ODE}
$$
solved by `v(s) = P_{s \leftarrow 0} v(0)` where `P_{s \leftarrow 0}` is the path-ordered exponential of `\int_0^s \Gamma\, \dot\gamma\, du`. In discrete 1-D this reduces precisely to (PT1) with `A_\ell = \Gamma(x_\ell)\, ds_\ell = \Gamma_\ell \sqrt{g_\ell}`, so the connection coefficient absorbs the metric step length. (We learn `A_\ell` and `g_\ell` independently; consistency `A = \Gamma \sqrt{g}` is enforced softly via regulariser, §6.)

---

## 4. Mechanism mapping (the surgical answer to the brief)

The brief mandates two mechanisms be **structural, not labels**. Locating them in FBA:

### 4.1 Perspective

**Perspective** is the **frame `F_i \in GL(d, \mathbb{R})` plus the connection `A_i`**, specifically:
- `F_i` is the *local viewpoint* — each query has its own non-orthogonal basis of `\mathbb{R}^d`.
- `A_i` is *how viewpoints transform when you walk from one token to another* — i.e., the rule for converting "what `j` sees" into "what `i` sees".
- The structure group `G = GL(d, \mathbb{R})` (not `O(d)`) is critical: it allows **scaling, shearing, and projective distortions**, not just rotations. A multi-head SDPA model is at best the `G = O(d)` × diagonal scaling subgroup with `A \equiv 0` — i.e., it has *frames* (the per-head projections) but **no connection at all** and only the rotational structure group.

The perspective is **not** "per-head subspace projection" because (a) the structure group is full `GL`, not the head-product group `\prod_h O(d_H)`; (b) the *connection* — the rule for relating perspectives across tokens — is a learned object with no SDPA analogue.

### 4.2 Focus

**Focus** is the **adaptive geodesic-ball heat kernel**, specifically:
- The attention measure (HK1) is a Gaussian in the **learned geodesic distance** `d_M`, not in token indices or Euclidean distances. The metric `g_i` is learned, so the distance is data-dependent.
- The **bandwidth `\sigma_i`** is per-query learned, providing query-specific focus strength.
- **Curvature controls volume**: in dimensions `d_M \ge 2`, by Bishop's volume comparison `\mathrm{Vol}(B_r) \le \mathrm{Vol}_{\kappa}(B_r)` where `\kappa` is sectional curvature. Negative curvature *enlarges* balls (more participants, soft focus); positive curvature *shrinks* balls (sharp focus). The model can therefore *learn its focus regime* by tuning curvature alone.

Focus is **not** "top-k" because (a) it is a smooth, differentiable measure (no hard cutoff except for compute, §7); (b) the support is adaptive *per query* via `\sigma_i` and *per data* via `g_i`; (c) the *shape* of the ball can be non-isotropic when `d_M \ge 2`.

---

## 5. Choice of group `G` and base `M` — options, scored

| Option | `G` | `M` | Pros | Cons |
|---|---|---|---|---|
| α | `O(d)` | `\mathbb{R}^1` | BF16-stable Cayley; norm-preserving | Strictly less expressive than SDPA |
| β **(selected)** | `GL(d, \mathbb{R})` | `\mathbb{R}^1` (warped) | Contains SDPA + SCFA as limits (§8) | Transport drift; needs renormalisation (§6) |
| γ | `GL(d_H)^{n_H}` (per-head) | `\mathbb{R}^1` | Cheaper; recovers MH-SDPA | Loses cross-head coupling |
| δ | Heisenberg | `\mathbb{R}^1` | Uncertainty-principle bias | Deferred |
| ε | `GL(d)` | learned `d_M \ge 2` | Anisotropic balls, non-trivial curvature | Chart problem; v2 |

**Selection:** Option β + FBA-R1. Final parameterisation: `F_i = I + U_i V_i^\top` rank-`r_F = 8`, `A_i = U^A_i V^{A\top}_i` rank-`r_A = 4`, **amortised** as MLPs of `x_i` (§3.5).

---

## 6. Stability of parallel transport

Parallel transport (PT1) over `T` steps with `A_\ell = U^A_\ell V^{A\top}_\ell` could in principle accumulate drift; we need an **operator-norm bound**.

**Lemma 1 (transport drift).** If `\|A_\ell\|_{op} \le \epsilon_A` for all `\ell`, then
$$
\|P_{i \leftarrow j}\|_{op} \;\le\; \prod_{\ell=j}^{i-1} (1 + \epsilon_A) \;\le\; e^{\epsilon_A (i-j)}, \qquad \|P_{i \leftarrow j} - I\|_{op} \;\le\; e^{\epsilon_A (i-j)} - 1.
$$

**Proof.** `\mathrm{Exp}_{GL}(A) = \sum_{n \ge 0} A^n/n!`, so `\|\mathrm{Exp}_{GL}(A)\|_{op} \le e^{\|A\|_{op}} \le e^{\epsilon_A}`. The product of `(i-j)` such factors has operator norm `\le e^{\epsilon_A (i-j)}`. ∎

**Practical implication.** At `T=16384` and a typical max transport distance of `\sigma_i \cdot d_M^{1/2} \sim 100` integer steps (because the heat-kernel weight on far transports is `\le e^{-50}`), the operator-norm bound at `\epsilon_A = 0.05` is `e^{0.05 \cdot 100} = e^5 \approx 148` — **too loose**. We add a **transport renormalisation** every step:
$$
\boxed{\;\hat P_{i \leftarrow j} \;:=\; P_{i \leftarrow j} \,/\, \bigl(1 + \kappa\, (i-j)\bigr),\;} \tag{REN}
$$
where `\kappa = 10^{-3}` is a learned scalar (one per layer). Operationally `\hat P` is what is *applied*; the bound is now `\le e^{\epsilon_A (i-j)} / (1+\kappa(i-j))`, which at `i-j = 100` and `\epsilon_A = 0.01` gives `\approx e/1.1 \approx 2.5` — safe.

**BF16 numerical stability.** Each `\mathrm{Exp}_{GL}(A_\ell)` requires a low-rank Sherman–Morrison-style inverse. We need `\|V_\ell^\top U_\ell\|_{op}` bounded away from `-I` for the SMW formula to be well-conditioned. Enforce
$$
\|V^A_\ell\|_F, \|U^A_\ell\|_F \;\le\; \sqrt{r_A}/2 \tag{INIT}
$$
at init and via a (very) light per-step projection (clip each row to the norm bound — `O(T r_A)` per layer, negligible).

---

## 7. Compute complexity

We dominate by the heat-kernel ball cost. Let `\bar n` denote the *average number of effective participants* per query.

### 7.1 Per-layer forward FLOP breakdown

| Stage | Cost | Notes |
|---|---|---|
| Frame application `F_i^{-1} W \cdot x_i` (Q,K,V) | `T \cdot 3 \cdot (d^2 + d r_F^2)` | SMW |
| Parallel transport `P_{i \leftarrow j}` (online, recursive) | `T \cdot \bar n \cdot d r_A` | rank-`r_A`, see §7.3 |
| Geodesic distance `d_M(i,j)` (cumulative sum) | `T` | one prefix sum |
| Heat-kernel `\alpha_{ij}` | `T \cdot \bar n` | exponential + multiply |
| Inner product `\langle \tilde q_i, \bar k_{i \leftarrow j}\rangle` | `T \cdot \bar n \cdot d` | inner products |
| Softmax over `j \in B_{r_i}(i)` | `T \cdot \bar n` | small softmax |
| Weighted sum `\sum_j \alpha_{ij} \bar v_{i \leftarrow j}` | `T \cdot \bar n \cdot d` | reduction |
| Final `F_i W_O \cdot (\dots)` | `T \cdot (d^2 + d r_F^2)` | SMW |
| **Total per layer** | `O(T d^2) + O(T \bar n d) + O(T \bar n d r_A)` | |

The `T d^2` term is **the SDPA-projection cost** — common to all attention variants, irreducible without further compression. The `T \bar n d` cost is the **FBA attention** itself.

### 7.2 Magnitude derivation

We claim `\bar n = O(\log T)`. *Derivation:* fix metric volume `V_0 := \int_{B_r(i)} dvol_g = 2 r_i \cdot \mathbb{E}[\sqrt{g}]^{-1}` (in `d_M = 1`). If we *fix* `V_0` across `T` (a regularisation choice, §6), then `r_i = V_0 \mathbb{E}[\sqrt{g}] / 2`. The **number of participants** is the count of `j` with `d_M(i,j) \le 3\sigma_i`. Setting `\sigma_i = \alpha \log T` for some small `\alpha`, and noting that heat-kernel concentration in a `d_M = 1` walk produces a Gaussian with stdev `\sigma_i`, the effective participation count is `\bar n = 2 \sigma_i \cdot \mathbb{E}[\sqrt{g}]^{-1} = O(\log T)`. We choose `\sigma_i` initialisation so that `\bar n \approx 32` at `T = 1024` (`\alpha \approx 5`) and `\bar n \approx 50` at `T = 16384` — both `O(\log T)`.

### 7.3 Numbers at three scales

Assume `d=2048, r_F=8, r_A=4`. Total Y per layer (G FLOPs):

| `T` | `\bar n` | SDPA `4T^2d + 5Td^2` | SCFA (`k=T/16`) | FBA `5Td^2 + 4T\bar n d r_A` |
|---|---|---|---|---|
| 1024 | 32 | 30.1 | 1.88 | **22.6** (1.33× vs SDPA, 0.083× vs SCFA) |
| 4096 | 41 | 223 | 3.49 | **91.5** (2.4× vs SDPA, 0.038× vs SCFA) |
| 16384 | 50 | 2550 | 9.93 | **371** (6.9× vs SDPA, 0.027× vs SCFA) |

**Honest reading.** Standalone FBA beats SDPA by `1.3 \to 7\times` but loses to SCFA by `12 \to 37\times` because SCFA compresses the projection `5Td^2 \to 5kd^2` that dominates FBA's cost. *FBA must compose with SCFA* to recover magnitude (§10).

### 7.4 Memory

| Quantity | Bytes |
|---|---|
| Per-token frame `F_i` (low-rank: `U_i, V_i`) | `2 T d r_F \cdot 2 = 16 T d r_F` BF16 |
| Per-edge connection `A_i` (low-rank: `U^A_i, V^A_i`) | `2 T d r_A \cdot 2 = 16 T d r_A` BF16 |
| Per-edge metric `g_i, \sigma_i` | `2 T \cdot 4 = 8 T` FP32 |
| Heat-kernel weights `\alpha_{ij}` (only stored over `B_{r_i}`) | `T \bar n \cdot 4` FP32 |

At `T=4096, d=2048, r_F=8, r_A=4`: frames = 1 GB/layer, connection = 512 MB/layer. **Too much** for the 16 GB VRAM budget on a 53-layer model. The **amortised parameterisation** (§3.5) reduces this to one MLP per layer (~5 MB), recovering memory feasibility but at the cost of expressivity (the frames are functions of token embeddings, not independent per token).

---

## 8. Limiting cases — recovery of SDPA and SCFA

### 8.1 SDPA recovery

Set `F_i = I`, `A_i = 0`, `g_i = 1`, `\sigma_i = \infty`. Then `P_{i \leftarrow j} = I`, `d_M(i,j) = |i-j|`, the Gaussian factor `\to 1` uniformly, and (HK1) reduces to
$$
\alpha_{ij} \propto \exp(\langle q_i, k_j\rangle/\sqrt d),
$$
which is exactly SDPA's softmax. ✓

### 8.2 SCFA recovery

Set `F_i = B \in \mathbb{R}^{T \times k}` shared across all `i` (constant frame, rank-`k`), `A_i = 0` (flat connection), `g_i = 1`, `\sigma_i = \infty`. Then `F_i^{-1}` is `B^\top` (Moore–Penrose pseudoinverse when `B^\top B = I_k`), `\tilde q_i = B^\top W_Q x_i`, and the attention is performed in the `k`-dim subspace — i.e., SCFA's spectral attention. The depthwise complement `D` of SCFA appears as the residual `(I - F_i F_i^\top) x_i` flow through a banded mixer; FBA's geodesic-ball factor with finite `\sigma_i` *replaces* this complement with a *smooth* local-window weighting on the ambient sequence axis. ✓

### 8.3 Expressivity gain

**Proposition.** The class of attention maps representable by FBA strictly includes those representable by SDPA and SCFA. The strict containment is witnessed by **frame-misaligned attention**: a measure where token `j` contributes to query `i` *with rotation* (e.g., `\bar v_{i \leftarrow j} = R_{ij} v_j` with `R_{ij} \ne I`). SDPA cannot express this (no transport); SCFA cannot express it (linear projection commutes with itself, gives only `B^\top \cdots B`).

Concretely: FBA can represent a *language model with grammatical case agreement* — where the "subject" frame and the "object" frame are *distinct linear bases of `\mathbb{R}^d`*, and the connection learns the conjugation `R_{\text{nom} \to \text{acc}}` between them. SDPA/SCFA must learn this from gradient signal alone with no inductive bias.

---

## 9. Magnitude claim — full derivation

Total compute per layer:
$$
C_{\text{FBA}} \;=\; \underbrace{5 T d^2}_{\text{projections}} \;+\; \underbrace{4 T \bar n d r_A}_{\text{transport + attn}} \;+\; \underbrace{2 T d r_F^2}_{\text{frame inv}} \;+\; \underbrace{T \bar n}_{\text{softmax+kernel}} \;\sim\; T d^2 + T d \log T.
$$
SDPA is `\sim T^2 d`. The asymptotic ratio
$$
\frac{C_{\text{SDPA}}}{C_{\text{FBA}}} \;\sim\; \frac{T^2 d}{T d^2 + T d \log T} \;=\; \frac{T}{d + \log T} \;\sim\; \frac{T}{d} \quad (T \ll e^d)
$$
gives `0.5\times` at `T=1024, d=2048` (SDPA wins — projection dominates) and `8\times` at `T=16384, d=2048`. **Standalone FBA is `O(T/d)` better than SDPA — a magnitude only at `T \gg d`.** Composition with SCFA (§10) is required.

---

## 10. Composition with flagship SCFA

SCFA gives `Y_{\text{SCFA}}(q) = B \hat y + D(q_\perp)` with `\hat y = \text{Attn}(B^\top q)`. FBA replaces the `\text{Attn}` step with FBA-on-`k`-space:
$$
\boxed{\;\hat y \;=\; \text{FBA}(B^\top q;\, F^{(k)}, A^{(k)}, g^{(k)}, \sigma^{(k)}),\;} \tag{COMPOSE}
$$
bundle objects living on the `k`-dim spectral axis: `F^{(k)}_a \in GL(d)` per spectral mode `a \in \{1,..,k\}`, `A^{(k)}_a \in \mathfrak{gl}(d)`, scalars `g^{(k)}_a, \sigma^{(k)}_a`. Bundle parameter count drops from `O(T(r_F+r_A)d)` to `O(k(r_F+r_A)d)` — `T/k = 16\times` smaller. Memory feasible without amortisation. Depthwise residual `D(q_\perp)` unchanged.

**Magnitude under composition (T=16384, k=1024, d=2048, `\bar n=50, r_A=4, w=8`):**
$$
C_{\text{SCFA+FBA}} \;=\; \underbrace{4 T k d}_{\text{SCFA project/lift}} \;+\; \underbrace{4 k \bar n d r_A + 5 k d^2}_{\text{FBA on k}} \;+\; \underbrace{2 T m (2w+1)}_{D(q_\perp)} \;\approx\; 137 + 23 + 1.1 \;=\; 161 \mathrm{G}
$$
vs SCFA-alone `9.93 G` — composition is `16\times` *slower* in raw FLOPs because SCFA uses `k = T/16 = 1024`. To make composition cheap we either (a) shrink `k` or (b) apply FBA only at high-entropy layers (top 25% by attention-entropy on calibration). With (b), net wall-clock is `~4\times` slower than SCFA but claimed to deliver `>5\times` step-count reduction via expressivity (§8.3). **Magnitude claim recast: NLL-per-FLOP, not time-per-step.** Conjectured `0.10–0.30` nat ema reduction at T=16384. Untested — Gate-0 (§14) validates.

---

## 11. Failure modes

**11.1 Connection learning starved.** Per query only `\bar n \approx 32` participants. Gradient signal to `A_\ell` summed over `T-\ell` queries weighted by `\alpha_{ij}`. Risk: `A_\ell` updates O(100×) slower than `W_Q`. Mitigation: boosted LR `\eta_A = 10 \eta_{\text{base}}`; init `A_\ell = 0` and rely on SDPA-recovery limit (§8.1). Gate-0 (§14) probes directly.

**11.2 BF16 parallel-transport drift.** By Lemma 1, 100-step transport with BF16 `A_\ell` (`\sim 10^{-3}` per-step error) accumulates `\sim 10^{-1}` operator-norm error → exponentially blows `\alpha_{ij}`. Mitigation: store `A_\ell` in **FP32** (128 MB total at `T=4096`, negligible); transport in FP32; cast to BF16 only at the final inner product.

**11.3 Curvature blowup.** If `\|A_\ell\| \to \infty`, operator-norm bound (Lemma 1) explodes. Mitigation: norm-clip `\|A_\ell\|_{op} \le 0.05` via per-step projection (`O(T r_A^2)`, cheap).

**11.4 Metric collapse.** `g_i \to 0` (free attention) or `g_i \to \infty` (no attention). Mitigation: `\sum_i (\log g_i)^2 \le \lambda_g T` regulariser.

**11.5 Inseparability from RoPE.** `g_i` overlaps in function with positional encoding. Gradient routes through whichever has higher unit signal. Mitigation: **disable RoPE in FBA layers**.

---

## 12. Determinism (per glades-ml policy)

All randomness goes through `glades::rng::*`. The bundle initialisation (low-rank factors `U, V, U^A, V^A`) draws from `glades::rng::randn` with per-network seed. Parallel transport (PT1) is purely matrix products — deterministic given inputs. Heat kernel (HK1) is a softmax over a fixed-ball support — deterministic. The Sherman–Morrison inverse (F1') is a closed form. **No stochastic gradient estimators, no random projections at inference time.** The only randomness is in per-batch training data sampling, governed by the existing data loader RNG.

CUDA kernel layout: per-token frames `F_i` stored contiguously per layer (`T \times d \times r_F` BF16 tensor); transport `P_{i \leftarrow j}` computed *online* per query in a CUDA block, never materialised. Memory access is causal and contiguous along the `j`-axis within `B_{r_i}(i)`. Determinism within a block: standard parallel reduction with deterministic tree (existing infrastructure `transformer_kernels.h`).

---

## 13. Deferred variants

FBA-Md (base `d_M \ge 2`, requires chart atlas), Heisenberg group, hyperbolic FBA (`M = \mathbb{H}^{d_M}`), and FBA-CHIRON (bundle on `(q,p)` symplectic phase space) are reserved for v2.

---

## 14. Gate-0 probe (must run before any 1B-scale claim)

**Setup.** 66M reference checkpoint (SCFA Gate-0 baseline). Single FBA layer at depth `L/2 = 6`; `r_F = 4, r_A = 2`; amortised frame parameterisation (§3.5); `\sigma_i` init so `\bar n = 16` at `T=1024`; `A_i = 0` init (SDPA-recovery). All other layers SDPA.

**Probe 1 — SDPA recovery.** With `A_i \equiv 0, F_i \equiv I, \sigma_i = \infty`: FBA layer reproduces SDPA loss within `10^{-6}` nat over 100 steps. Pass: divergence `\le 10^{-5}`. *Falsifies implementation correctness if fails.*

**Probe 2 — Connection learning.** 2,500 steps, `\eta_A = 10 \eta_{\text{base}}`. Pass: mean `\|A_i\|_{op} > 10^{-2}` at step 2500. *Falsifies "connection learning is starved" if fails.*

**Probe 3 — NLL parity vs SDPA.** ema NLL at step 2,500 vs SDPA baseline. Pass: `\Delta \text{ema NLL} \le +0.02` nat. *Falsifies "FBA does not hurt at iso-FLOP" if fails.*

**Probe 4 — Focus activation.** `\sigma_i` distribution. Pass: variance `> 0.5 \cdot \mathbb{E}[\sigma_i]^2` (model uses per-query focus, not collapsed to a global bandwidth). *Falsifies "focus is data-adaptive" if fails.*

**Cost budget.** ≤1 GPU-hour on 66M. Failure → **reject** FBA, move to candidates B or C.

**Probe failure → redesign:** 1 → fix SMW/transport kernel; 2 → bump `\eta_A` to `\times 100` or auxiliary loss `\|A_i\|_F^2 \to 1`; 3 → abandon FBA; 4 → remove `\log \sigma_i` regulariser.

---

## 15. Conclusion

FBA places attention in a principal `GL(d, \mathbb{R})`-bundle over a learned warped sequence base manifold. The two mandatory mechanisms are surgically resolved: **perspective** as per-token frame `F_i \in GL(d)` plus connection `A_i`; **focus** as the adaptive geodesic-ball heat-kernel measure with learned `g_i, \sigma_i`. Standalone FBA does not clear the `10\times` raw-FLOP bar; the proposal is **FBA ∘ SCFA** with the magnitude claim recast as NLL-per-FLOP. Dominant risk is connection-learning starvation; Gate-0 (≤1 GPU-hour, 66M ref) falsifies or admits.
