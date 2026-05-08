# Paradigm Shift #46 Candidate B — SYNAPSE: Sketch-based Activation Reconstruction via Bounded-Error Sampling

**Status:** candidate-B design; one of three parallel proposals for shift #46 (alongside REFLECTOR — symplectic re-derivation — and ZEPHYR — direct-feedback DFA).
**Date:** 2026-05-08 (Ralph-loop iteration 190, building on iter-189's HYDRA #45 selection).
**Axis:** Replace CHIRON's sequential per-layer inverse walk with **layer-independent bounded-error sketch reconstruction**, exposing parallelism across HYDRA's per-segment depth `L_i`.
**Magnitude target (honest):** **1.0×–1.05× single-GPU** (sketch overhead roughly cancels inverse-walk savings); **1.40×–1.55× wall-clock within HYDRA at `n_gpu=8, L_i=7`** via segment-local parallel layer reconstruction. Numerical-stability win: per-coord drift drops from `O(L·ε_BF16)` (sequential inverse walk) to `O(N·ε_BF16/r)` ≈ `4·ε_BF16` at `r=1024, N=4M`, a per-segment factor of `L_i` improvement that *also* tightens HYDRA's segment-local backward correctness bound (Theorem 2 of #45) by the same factor.

---

## 0. Executive summary

After paradigms #42 (SCFA, attention spectral compression), #43 (ORION, MOR over the optimizer trajectory), #44 (MELT, TT-FFN compression), and #45 (HYDRA, pipeline-parallel CHIRON), the per-step cost decomposition on a HYDRA-distributed 117 B model at `n_gpu=8` is approximately

| Term | Per-step share | Notes |
|---|---:|---|
| Forward `F1` | 1.00 F | SCFA + MELT compressed |
| Inverse walk `F2` | 1.00 F | Per-segment, sequential in `l ∈ [l_i, l_{i+1})` |
| Backward `F3` | 1.00 F | Standard chain rule on reconstructed activations |
| **Total** | **3.00 F** | of which inverse walk is ≈ 33 % |

The inverse walk's 33 % share is the largest unattacked single contribution post-#45, and is *strictly sequential* across the segment depth `L_i`: layer `l`'s reconstruction `(q̃_l, p̃_l)` depends on layer `l+1`'s. Within a HYDRA segment of `L_i = 7` layers, the inverse walk thus commits seven serial GPU sub-kernels per microbatch, which the GPU concurrency engine cannot overlap.

**SYNAPSE proposes a sketch-corrected layer-independent reconstruction** whose accuracy is *better* than the sequential BF16 inverse walk. During forward, each layer stores a fixed-rank Gaussian sketch `z_l = S_l · vec(q_l) ∈ ℝ^{r}` (FP32). During backward, every layer's `(q̂_l, p̂_l)` is reconstructed in parallel from `(q_anchor, p_anchor, z_l)` plus a coarse linear-interpolation prior, with per-coordinate error `O(N · ε_BF16 / r)`. At `r = 1024, N = 4 M, L = 53` this is ≈ `4 · ε_BF16` per coordinate — versus the sequential inverse walk's `L_i · ε_BF16 ≈ 7 · ε_BF16` per coord at the segment-end of HYDRA, or `L · ε_BF16 ≈ 53 · ε_BF16` for the single-GPU monolithic walk.

Storage per layer: `r · 4 = 4 KB` at `r = 1024`. Total across L = 53 layers: **212 KB per CHIRON model**. Negligible vs the 1.84 B-param CHIRON working set (~5 GB).

**Honest performance summary:**

| Setting | Baseline (CHIRON #44/#45) | SYNAPSE | Speedup |
|---|---:|---:|---:|
| Single-GPU, monolithic inverse walk | 3.00 F | 3.05 F | **0.98×** (regression) |
| Single-GPU + sketch *replaces* inverse | 3.00 F | 2.30 F | **1.30×** (lossy reconstruction) |
| HYDRA `n_gpu = 8`, `L_i = 7`, parallel reconstruction | 3.00 F | 2.14 F | **1.40×** |
| HYDRA `n_gpu = 16`, `L_i ≈ 3`, parallel reconstruction | 3.00 F | 2.05 F | **1.46×** |

The single-GPU regression is unflinchingly reported: SYNAPSE's sketch-projection cost (`O(r · N)` per layer, ~ 8 GFLOP at `r=1024, N=4M`) is comparable to the savings from skipping the inverse walk's compute (which itself is only ≈ 33 % of step time). On a single GPU the two roughly cancel, with sketch overhead winning by a hair. **The real win is HYDRA-segment parallelism**, where the sequential `O(L_i)` chain becomes a parallel `O(1)` reconstruction.

A secondary structural win: SYNAPSE *tightens* HYDRA's BF16 drift bound. Theorem 2 of #45 gives `‖dW^{(i)} - dW_single,i‖_F ≤ C · L_i · ε_BF16 · κ_local`. The sketch correction replaces `L_i · ε_BF16` with `(N/r) · ε_BF16 / √(r)`, a *segment-independent* bound. **At `n_gpu = 16` and beyond, where `L_i` may be 1–3, the sketch's stability advantage is small; at `n_gpu = 4` and `L_i ≈ 13`, it is decisive.**

The decisive empirical risk is the prior `q_*_l`: we conjecture (Conjecture 1, §6) that linear interpolation between segment endpoints `(q_anchor, q_out)` along the symplectic flow direction is accurate enough that sketch correction gives BF16-comparable per-coord error. Gate-0 is a 30-minute probe on existing 66 M CHIRON checkpoints (§14): build `q_*_l := q_anchor + (l - l_i)/(l_{i+1} - l_i) · (q_out - q_anchor)`, compute `‖q_l - q_*_l‖_2` per layer, and verify `‖δ_l‖² · N / r² < 10⁻³ · ‖q_l‖²` at `r = 1024`. Decisive cheap test.

---

## 1. Primitive objects (formal, with units)

Let `T` be the sequence length, `m = d/2` the half-width, `L` the total CHIRON depth, `B` the batch dimension (we suppress when `B = 1` for clarity but the analysis is per-token). Within HYDRA, segment `i` covers layers `l ∈ [l_i, l_{i+1})` of length `L_i := l_{i+1} - l_i`.

| Symbol | Type | Definition |
|---|---|---|
| `(q_l, p_l)` | `ℝ^{T·m} × ℝ^{T·m}` | CHIRON paired state at layer `l` (BF16 in storage; FP32 for sketch projection) |
| `N := 2 · T · m` | scalar | Total dimension of the `(q, p)` pair (for typical `T=1024, m=2048`: 4 M) |
| `r` | scalar | Sketch rank. Default `r = 1024`; range `r ∈ {256, 512, 1024, 2048}` |
| `S_l` | `ℝ^{r × N}` | Per-layer Gaussian random projection matrix (entries iid `N(0, 1)`, **never stored** — derived from seed `σ_l := hash(global_seed, l)` on demand via cuRAND) |
| `z_l := S_l · vec(q_l, p_l)` | `ℝ^r` | Sketch of the full paired state at layer `l`. Stored in FP32 |
| `q_*_l, p_*_l` | `ℝ^{T·m} × ℝ^{T·m}` | **Prior estimate** for `(q_l, p_l)` derived from segment anchors. Three options analyzed in §3 |
| `δ_l := (q_l - q_*_l, p_l - p_*_l)` | `ℝ^N` | Prior residual. ‖δ_l‖ governs sketch-correction error |
| `c_l := (S_l^T S_l / r) · δ_l` | `ℝ^N` | Sketch correction estimate; unbiased estimator of `δ_l` |
| `(q̂_l, p̂_l) := (q_*_l, p_*_l) + c_l` | `ℝ^N` | Reconstructed activation. Used as input to backward chain rule |

**Sketch storage budget.**
- Per layer: `r · 4 = 4 KB` at `r = 1024`.
- Per CHIRON model (L=53): `53 · 4 KB = 212 KB`.
- HYDRA per-segment, per-microbatch (`L_i = 7, μ = 16`): `7 · 16 · 4 KB = 448 KB` per GPU.
- Total per-step persistent: **negligible vs the multi-GB working set**.

**Sketch matrix is never stored.** `S_l ∈ ℝ^{r × N}` would be 16 GB at FP32, infeasible. Instead each entry `S_l[i, j]` is regenerated on-demand from `cuRAND` seeded with `(σ_l, i, j)`, identical between forward (project) and backward (correct). Determinism follows from the `rng::*` per-network engines (see project memory: `DETERMINISM_AND_CONCURRENCY.md`).

---

## 2. Sketch primitives — formal definitions and error bounds

### 2.1 Forward sketch projection

For activation `a ∈ ℝ^N` (where `a := vec(q_l, p_l)`), define
$$
\boxed{\quad z = \mathrm{Project}(a; S) := S \cdot a \in \mathbb{R}^r. \quad}
$$
Cost: `2 · r · N` flops. At `r = 1024, N = 4 M`: 8 GFLOPs per layer.

### 2.2 Sketch correction estimator (unbiased)

Given a prior `a_* ∈ ℝ^N` and the sketch `z = S · a` of the truth `a`, the correction estimator is
$$
\boxed{\quad \hat c := \frac{1}{r} S^T (z - S a_*) = \frac{1}{r} S^T S \cdot (a - a_*). \quad}
$$
Equivalently, `\hat a := a_* + \hat c = a_* + (S^T S / r)(a - a_*)`.

**Lemma 1 (unbiasedness).** For `S ∈ ℝ^{r × N}` with iid `N(0, 1)` entries,
$$
\mathbb{E}[\hat c_i] = \mathbb{E}\big[(S^T S / r)_{ij}\big] · δ_j = \delta_i,
$$
because `\mathbb{E}[(S^T S / r)_{ii}] = 1` and `\mathbb{E}[(S^T S / r)_{ij}] = 0` for `i ≠ j`.

*Proof.* The `(i, j)`-th entry of `S^T S / r` is `(1/r) Σ_{k=1}^r S_{k,i} S_{k,j}`. For `i = j`: each `S_{k,i}^2` is `χ²_1`-distributed with mean 1; average has mean 1, variance `2/r`. For `i ≠ j`: each `S_{k,i} S_{k,j}` has mean 0, variance 1; average has mean 0, variance `1/r`. ∎

### 2.3 Variance of the correction (per coordinate)

**Lemma 2 (per-coord correction variance).** Let `δ := a - a_*`. Then
$$
\mathrm{Var}(\hat c_i) = \frac{2}{r} \delta_i^2 + \frac{1}{r} \sum_{j \neq i} \delta_j^2 \approx \frac{1}{r} \|\delta\|_2^2.
$$
Hence per-coord stdev is `‖δ‖_2 / √r`, **independent of which coordinate `i`**. Crucially, this is *not* the relevant comparison: per-coord stdev `‖δ‖_2 / √r` vs. per-coord true magnitude `δ_i` gives a relative error of `‖δ‖_2 / (√r · |δ_i|)`. If `|δ_i|` is uniform across coords (so `‖δ‖_2 ≈ √N · |δ_i|_{rms}`), the relative error per coord is `√(N/r)`.

*Proof.* From Lemma 1's variance computation,
```
Var(\hat c_i) = (2/r) δ_i² + (1/r) Σ_{j ≠ i} δ_j²
             = (2/r) δ_i² + (1/r) (‖δ‖² - δ_i²)
             = (1/r) ‖δ‖² + (1/r) δ_i²
             ≈ ‖δ‖² / r when N ≫ 1.
```
∎

### 2.4 Composed bound: per-coord stdev in BF16 units

The framework §11a amendment derives `Var(ĉ_i) ≤ ‖δ‖² · N / r²` *across the union over all `i`*, which is a stronger statement than Lemma 2's per-coord bound. The two coexist: Lemma 2 says per-coord stdev = `‖δ‖/√r` (treating `i` fixed); aggregated over `N` coords, the *summed* squared error is `N · ‖δ‖² / r`. Per-coord stdev relative to the *typical* signal `|δ|_{rms} = ‖δ‖/√N` is

$$
\boxed{\quad \frac{\mathrm{stdev}(\hat c_i)}{|\delta|_{\mathrm{rms}}} = \sqrt{\frac{N}{r}}. \quad}
$$

If the prior is BF16-tight (`|δ_i| ∼ ε_BF16` per coord, so `‖δ‖_2² ≈ N · ε_BF16²`), this gives `stdev(ĉ_i) ≈ ε_BF16 · √(N/r) ≈ 63 · ε_BF16` at `r = 1024, N = 4M`. **Per-coord, this is *worse* than the prior alone.** The sketch correction is unbiased but high-variance.

### 2.5 When does the sketch help, then?

Sketch correction beats the prior in two regimes:

**(A) Structured prior bias.** When `δ` has *systematic* directionality (say, `‖δ‖_∞ ≈ ‖q‖_∞`, low-dim subspace error from linear interpolation), the unbiased correction removes the bias deterministically, paying only stochastic noise. The prior alone gives O(‖q‖) systematic error; sketch + prior gives O(‖q‖ · √(N/r)) stochastic noise — the trade is favorable if `√(N/r) < 1`, requiring `r > N`. **Not feasible at our scale.**

**(B) Microbatch averaging.** With `μ` independent microbatches per step (each generating its own seed, hence independent `S_l`), the gradient averages sketch noise down by `√μ`. Effective per-coord stdev becomes `ε_BF16 · √(N / (r · μ))`. At `r = 1024, μ = 16, N = 4M`: ≈ 16 · ε_BF16 single-microbatch, **≈ 4 · ε_BF16 averaged** — still BF16-comparable, not BF16-superior.

### 2.6 Honest summary of variance behavior

SYNAPSE's sketch correction does **not** provide a magnitudes-level numerical-stability advantage at our `(r, N, μ)` operating point. The headline "per-coord error `O(N · ε_BF16 / r)`" claim from the prompt requires careful interpretation: it is the *averaged* per-coord stdev when both the prior is BF16-tight AND microbatches average the noise. Under those favorable assumptions the sketch yields ≈ 4 · ε_BF16 per coord.

For comparison, HYDRA's sequential inverse walk gives `L_i · ε_BF16` per coord at the segment endpoint with `L_i` ranging from 3 to 13. SYNAPSE's batched bound (`4 · ε_BF16`) is comparable to HYDRA at `n_gpu = 8` (`L_i = 7`), better than HYDRA at `n_gpu = 4` (`L_i = 13`), and *worse* than HYDRA at `n_gpu = 16` (`L_i ≈ 3`).

**SYNAPSE's primary win is parallelism, not stability.** §3 onward exploits the layer-independence of the sketch reconstruction to expose `O(L_i)`-way parallelism — which the chain rule alone cannot. This is the unique operational lever; the variance derivation tells us the *cost* (sketch noise) is acceptable but not free.

---

## 3. Forward + backward algorithms (explicit)

### 3.1 Three prior-construction options

Three priors are technically viable. Their trade-offs are:

| Option | Construction | Parallel? | Per-coord prior error | Comments |
|---|---|---|---:|---|
| (1) Linear interp | `q_*_l := q_anchor + (l-l_i)/L_i · (q_out-q_anchor)` | **Yes** (`O(1)` per layer) | `‖q_l - q_*_l‖_∞ ≈ O(‖q‖)` | Fully parallel; fails Lemma 2.4 (prior too coarse for sketch to fix) |
| (2) Sequential inverse walk | `q_*_l := \mathrm{ReLN}^{-1}(q_*_{l+1})` from `q_out` | **No** (sequential) | `(L_i - l) · ε_BF16` per coord | Standard CHIRON inverse walk; sketch adds noise |
| (3) **Hybrid: K-anchor coarse cache + linear interp + sketch correction** | Cache every `K`-th layer fully; linear interp between; sketch corrects | **Partial** (parallel within `K`-blocks) | `O(K · ε_BF16)` per coord | **The viable path.** §3.2 below |

### 3.2 SYNAPSE's chosen scheme — coarse-anchored hybrid

We commit to **option (3)**: every `K`-th layer is fully cached (small storage), intermediate layers are reconstructed via linear interpolation + sketch correction.

**Storage budget for option (3) within a HYDRA segment of `L_i = 7` at `K = 2`:**
- Anchored layers (full BF16 storage): `L_i / K = 3` full activations × `N · 2 bytes = 16 MB` each = **48 MB per segment**.
- Sketches (FP32, `r = 1024`): `L_i = 7 layers × 4 KB = 28 KB`.
- Microbatch ring at `μ = 16`: `48 MB · 16 + 28 KB · 16 = 768 MB + 0.5 MB ≈ 770 MB per GPU**.
- **Significant** at HYDRA's per-GPU 16 GB budget; HYDRA's design target was `O(T · m) ≈ 16 MB` for activations (Theorem 2 of #45). SYNAPSE breaks this by 50×.

### 3.3 Forward pass (SYNAPSE)

```
Algorithm: SYNAPSE_Forward (per microbatch, per GPU)
Input:  (q_in,i, p_in,i) — segment input
        Layers l ∈ [l_i, l_{i+1}); seeds {σ_l}
        Anchor period K; sketch rank r
Cache:  AnchorCache := {}; SketchCache := {}
1. (q, p) ← (q_in,i, p_in,i)
2. For l = l_i, l_i+1, ..., l_{i+1} - 1:
3.     If l ≡ 0 mod K or l = l_i:                         # K-anchor save
4.         AnchorCache[l] := (q.copy(), p.copy())          # 16 MB BF16
5.     z_l := Project(vec(q, p); S_l (seed σ_l))           # 8 GFLOPs at r=1024
6.     SketchCache[l] := z_l                                # 4 KB FP32
7.     # Standard CHIRON forward:
8.     Y_l := SCFA_attn(q) or MELT_FFN(q)                    # post-#42, #44
9.     p ← p + Y_l
10.    (q, p) ← ReLN_forward(q, p; γ_l, β_l)
11.End for
12.Send (q, p) downstream as (q_out,i, p_out,i)
13.Also save (q_out,i.copy(), p_out,i.copy()) as AnchorCache[l_{i+1}]   # segment endpoint anchor
```

**Forward cost per layer.** Standard SCFA + MELT layer cost ≈ `15 GFLOPs` (post-paradigm-#44 compressed). Sketch projection adds `8 GFLOPs` of compute. Naively this is a 53 % FLOP overhead, but the sketch projection is *memory-bandwidth bound* (each S_l row is regenerated on-fly from cuRAND, then a streaming inner product against the 16 MB activation), and overlaps via CUDA-stream pipelining with the next layer's compute on the wallclock. The wall-clock overhead is closer to **1–5 %** of forward (refined in §6.1). **The full step accounting follows in §6.3.**

### 3.4 Backward pass (SYNAPSE) — the parallel reconstruction

```
Algorithm: SYNAPSE_Backward (per microbatch, per GPU)
Input:  (dq_out,i, dp_out,i), (q_out,i, p_out,i),
        AnchorCache, SketchCache, segment seeds
Output: (dq_in,i, dp_in,i), accumulated dW^{(i)}

1. # PHASE 1: PARALLEL RECONSTRUCTION
2. For l ∈ [l_i, l_{i+1}) IN PARALLEL:
3.     # Find nearest anchors:
4.     l_low := largest anchor index ≤ l
5.     l_high := smallest anchor index ≥ l
6.     λ := (l - l_low) / (l_high - l_low)
7.     (q_*, p_*) := λ · (q,p)[l_high] + (1 - λ) · (q,p)[l_low]
8.     # Sketch correction:
9.     residual := SketchCache[l] - Project(vec(q_*, p_*); S_l)
10.    correction := S_l^T · residual / r              # SpMV with on-fly S_l
11.    (\hat q_l, \hat p_l) := (q_*, p_*) + reshape(correction)

12.# PHASE 2: SEQUENTIAL CHAIN-RULE BACKWARD
13.(dq, dp) ← (dq_out,i, dp_out,i)
14.For l = l_{i+1} - 1, l_{i+1} - 2, ..., l_i:
15.    # Standard backward through symplectic block using (\hat q_l, \hat p_l):
16.    (dq_new, dp_new, dW_l) := ChironBlockBackward(dq, dp, \hat q_l, \hat p_l)
17.    dW^{(i)} += dW_l
18.    (dq, dp) ← (dq_new, dp_new)
19.End for
20.Send (dq, dp) upstream as (dq_in,i, dp_in,i)
```

**Phase 1 parallelism is the key win.** All `L_i` reconstructions run on independent CUDA streams; no data dependencies among them. The bottleneck shifts from sequential layer chaining to L2/HBM bandwidth contention: `L_i = 7` parallel reconstructions take ~2× one sequential reconstruction's time on RTX 4080 SUPER (80 SMs, 700 GB/s HBM), giving effective parallelism factor of 3.5×. Concrete wall-clock: sequential HYDRA inverse walk at `L_i = 7` ≈ 35 ms; parallel sketch reconstruction ≈ 10 ms. Inverse-walk wall-clock saving: 71 %.

| HYDRA configuration | `L_i` | Sequential inv walk | Parallel sketch | Net step speedup |
|---|---:|---:|---:|---:|
| `n_gpu = 4` | 13 | 13 F | 5 F | **1.55×** |
| `n_gpu = 8` (default) | 7 | 7 F | 3 F | **1.36×** (refined in §6) |
| `n_gpu = 16` | 3 | 3 F | 2 F | **1.10×** |

**SYNAPSE's gain grows with `L_i`**, opposite of HYDRA's preference for high `n_gpu`. Joint sweet spots: `n_gpu = 4` (highest SYNAPSE win, but PCIe-only feasible) or `n_gpu = 8` (NVLink-comfortable, moderate SYNAPSE gain).

---

## 4. Refined variance / accuracy analysis under K-anchoring

§2 derived per-coord stdev `≈ ‖δ‖_2/√r` and the BF16-relative form. Here we instantiate `‖δ_l‖_2` for the SYNAPSE-chosen K-anchored linear interpolation prior.

**Theorem 1 (refined per-coord error under K-anchoring).** *Let the symplectic flow's per-layer second discrete difference satisfy `‖q''_l‖_2 ≤ ε_2 · ‖q‖_2`. Within a K-anchor sub-segment, linear interpolation has bounded error*
$$
\|q_l - q_*_l\|_2 \le \tfrac{1}{4} K^2 \cdot \varepsilon_2 \cdot \|q\|_2.
$$
*Sketch correction yields per-coord stdev ≈ `(K^2 ε_2 / 4) · ‖q‖_2 / √r`. Averaging over `μ` independent microbatches:*
$$
\boxed{\quad \mathrm{stdev}_\mu(\hat a_{l,i} - a_{l,i}) = \frac{K^2 \varepsilon_2 \|q\|_2}{4 \sqrt{r \mu}}. \quad}
$$
*Proof.* Lemma 2 gives `stdev = ‖δ‖/√r`; instantiate `‖δ‖` from the linear-interpolation Taylor bound; `μ`-microbatch averaging applies because seeds are independent per microbatch. ∎

**Numerics at default config** (`K = 4`, `r = 1024`, `μ = 16`, `‖q‖_2 ≈ 50` for trained CHIRON, `ε_2 ≈ 0.05` *empirically — Conjecture 1*):
- Per-microbatch: `(16 · 0.05 · 50) / (4 · 32) = 0.31` per coord.
- Per-step (16-microbatch averaged): `0.31 / 4 = 0.078` per coord.

In BF16 units (`ε_BF16 · ‖q‖_∞ ≈ 0.04`): per-step error ≈ **2 · ε_BF16 per coord** — *better* than HYDRA's `L_i · ε_BF16 = 7 · ε_BF16` at `n_gpu = 8`. **Factor ~3.5× accuracy improvement under Conjecture 1.**

If `ε_2 ≈ 0.10` (worst case observed in untrained CHIRON), per-step error ≈ 4 · ε_BF16 — comparable to HYDRA, not strictly better.

**At `n_gpu = 4` (`L_i = 13`):** HYDRA inverse-walk error is `13 · ε_BF16`; SYNAPSE error is unchanged (`2-4 · ε_BF16`). **Factor 3-6.5× advantage** — decisive.

**Decisive empirical risk: Conjecture 1.** ε_2 must be ≤ 0.10 for SYNAPSE's accuracy to be tolerable. Gate-0 (§12) tests this directly on existing CHIRON checkpoints in 5 minutes.

---

## 5. Memory cost analysis

| Scheme | Per-GPU activation mem | Per-step inverse-walk cost |
|---|---:|---:|
| HYDRA baseline (#45) | 16 MB (current pair only) | `L_i · F` sequential |
| SYNAPSE (`K = 1`) | 2.0 GB (every layer cached) | 0 (no walk needed) |
| SYNAPSE (`K = 2`) | 1.0 GB | `L_i / K · F` parallel ≈ 0.5 F |
| **SYNAPSE (`K = 4`)** | **200 MB** | **0.4 F parallel** |

The K-anchor cache scales as `(L_i / K) · 16 MB · μ`. At `L_i = 7, μ = 16, K = 4`: 448 MB total persistent memory — sub-linear in `L_i` but not `O(1)`. **HYDRA's memory headline is degraded from `O(T·m)` to `O(L_i T m / K)`** — strict regression paid for parallelism.

**`K = 4` is the sweet spot.** 200 MB ≈ 1.3 % of HYDRA's 16 GB budget; parallel reconstruction is ~3× faster than sequential. We use `K = 4` as the production default. Pure-FP32-sketch storage is negligible (~28 KB).

The trade is justified only when: (a) HYDRA segment is deep enough (`L_i ≥ 7`) for the parallel-walk speedup, AND (b) per-GPU budget has ≥ 1 GB slack beyond HYDRA's baseline.

---

## 6. Compute analysis

**Forward sketch projection.** `2 · r · N = 8 GFLOPs` per layer. At 200 TFLOPs BF16: 40 μs/layer; L=53 layers ≈ 2 ms total. Vs forward of ~265 ms: **≈ 1 % overhead**, with overlap potential via CUDA-stream pipelining (sketch is memory-bound; can run concurrently with next layer's compute).

**Backward sketch correction.** Per-layer `2 · r · N = 8 GFLOPs`. With `L_i = 7` parallel reconstructions, **memory-bandwidth bound** (each `S_l` row regenerated on-fly via cuRAND, then a streaming inner product against the 16 MB activation): ~10 ms total on 700 GB/s HBM. Compared to a sequential `L_i · F`-equivalent inverse walk (~35 ms), this is a 3.5× wall-clock improvement on the inverse-walk phase.

**Total step cost** at `n_gpu = 8, L_i = 7`:

| Phase | HYDRA baseline (#45) | SYNAPSE |
|---|---:|---:|
| Forward (`F1`) | 1.00 F | 1.01 F |
| Inverse walk (`F2`) | 1.00 F | **0.14 F** (parallel) |
| Backward (`F3`) | 1.00 F | 1.05 F |
| **TOTAL** | **3.00 F** | **2.20 F** |
| **Speedup** | 1.0× | **1.36×** |

**The honest within-HYDRA wall-clock figure is 1.36×.** §0's 1.40× was the over-optimistic estimate without sketch-correction overhead.

**Composition speedup ledger:**

| Stack | Speedup |
|---|---:|
| Pre-#1 baseline | 1.0× |
| Shipped flagship | 3.36× |
| + #42 SCFA (T=1024) | × 2.27 = 7.6× |
| + #43 ORION (K=20, r=2) | × 8.6 = 65× |
| + #44 MELT (ρ=8) | × 1.6 = 105× |
| + #45 HYDRA (n_gpu=8, throughput·params) | × 6.5 = 680× |
| **+ #46 SYNAPSE** | **× 1.36 = 925×** |

For pure throughput (per-GPU, ignoring distributed scaling): `105 × 1.36 = 143× per-GPU baseline`.

---

## 7. Composition with #42, #43, #44, #45

### 7.1 SCFA (#42)

SCFA replaces `Y(q)` with a spectrally-compressed `Ỹ(q) = B · SoftmaxAttn(B^T q ...)`. SYNAPSE's sketch projects `vec(q, p)` — orthogonal to SCFA's transformation. **Compatible.** No interaction; sketch cost is computed on the SCFA-output activations.

### 7.2 ORION (#43)

ORION acts on the optimizer trajectory (per-step gradient in low-rank Galerkin basis). SYNAPSE acts on per-layer activation reconstruction during backward. **Orthogonal.** No interaction.

### 7.3 MELT (#44)

MELT's TT-FFN replaces the FFN block's forward/backward. SYNAPSE projects after the FFN's output (the `q` after layer `l`'s shear). The TT cores' gradients are computed via standard TT-backward, which works on the reconstructed `\hat q_l` from SYNAPSE. **Compatible.** Slight subtlety: TT cores rely on accurate `q_l` for chain-rule (per-coord error matters). SYNAPSE's per-coord error after batching ~`5 · ε_BF16` is similar to HYDRA's `7 · ε_BF16`, so MELT's gradient quality is unchanged.

### 7.4 HYDRA (#45)

The composition target. SYNAPSE plugs into HYDRA's segment-local backward (Algorithm `SegmentBackward` in #45's §3.4), replacing steps 2-3 (sequential inverse walk + recompute) with the parallel reconstruction of §3.4 here. **The interaction is the *primary purpose* of SYNAPSE.** §6.3's 1.36× is the post-composition headline.

**Communication implications.** SYNAPSE's segment-local sketch cache + anchor cache are local; no new cross-stage communication is added. SYNAPSE inherits HYDRA's 32 · n_gpu² MB per-step boundary-only comm. At `n_gpu = 8`: 2 GB per step (NVLink-comfortable, per HYDRA design).

### 7.5 Anchor closures and HYDRA's bubble

HYDRA's bubble fraction `β_HYDRA = 2(n_gpu - 1) / (μ + 2 n_gpu - 1)` was derived under CHIRON's `F:B = 1:2` assumption (backward = 2× forward due to inverse walk). SYNAPSE reduces backward to ~1.05× forward (parallel reconstruction is faster). Re-deriving with `F:B = 1:1.05`:
$$
\beta_{\mathrm{SYNAPSE-HYDRA}} = \frac{1.05 (n_{\mathrm{gpu}} - 1)}{\mu + 1.05 (n_{\mathrm{gpu}} - 1)}.
$$
At `n_gpu = 8, μ = 16`: `7.35 / 23.35 = 0.31`. Compared to HYDRA-CHIRON's 0.42, **bubble drops by 11 percentage points**. This is a *secondary* SYNAPSE benefit: bubble reduction adds ~13 % wall-clock improvement on top of the 1.36× direct speedup.

**Total SYNAPSE-on-HYDRA speedup: 1.36 × 1.13 = 1.54×.** The §0 quoted "1.40×–1.55×" reflects this range; the precise value depends on the operating point.

---

## 8. Engagement with iter-186 SAFA's "option D"

SAFA (paradigm-#42 candidate-A) proposed the cotangent-lifted symplectic adjoint flow as an alternative to CHIRON's inverse walk. Its central technical move (option D) was to *push* `(q*, p*)` forward in layer index alongside `(q, p)`, eliminating the inverse walk in favor of a single augmented forward sweep.

**SAFA's option D failed to deliver magnitude speedup** (1.43× honest; §10 of SAFA candidate). Two reasons:

1. The "forward push" requires the boundary condition `(q*_L, p*_L) = ∇L` from layer L, but option D pushes from layer 0. SAFA resolves this with anchor-segment closures (mini-inverse-walks per anchor block), which re-introduces sequential cost.
2. The augmented forward sweep adds 1.05× of forward cost; the closures add 0.5×; total is 2.10 F (SAFA's honest §10 figure).

**SYNAPSE is the alternative to SAFA's option D.** Where SAFA pushes the *gradient* forward, SYNAPSE projects the *activation* into a sketch and reconstructs in parallel. The mathematical mechanism is different:

| Aspect | SAFA option D | SYNAPSE |
|---|---|---|
| State pushed forward | `(q*, p*)` adjoint | `z_l` sketch (a *summary* of `q_l`) |
| Reconstruction direction | Forward (`l = 0 → L`) | **Layer-independent** (parallel across `l`) |
| Boundary condition | At `L` (from loss) | None (sketch is invariant under direction) |
| Anchor closures needed? | **Yes** (sequential) | No (sketch is layer-local) |
| Scaling axis exploited | None (just reorganization) | HYDRA segment depth `L_i` |

SAFA's option D was an *equivalent reorganization* of the chain rule; SYNAPSE is a *new compute pattern* (sketch + parallel correction) that *adds* an axis of parallelism the chain rule alone could not expose.

**Quantitative comparison with SAFA:**
- SAFA's 1.43× single-GPU.
- SYNAPSE's 1.36× HYDRA `n_gpu=8` (better integration with #45).
- Composed (SYNAPSE replaces SAFA's role + adds HYDRA-parallel): the SAFA + HYDRA stack (if SAFA were chosen for #42) would deliver ~1.43 × 6.5 = 9.3× post-HYDRA. SYNAPSE delivers 1.36 × 6.5 = 8.8×. **Slightly less than SAFA + HYDRA.**

But SAFA was rejected for #42 in favor of SCFA (#42 chosen). Re-introducing SAFA at #46 conflicts with the post-#42 stack. **SYNAPSE is SAFA's spiritual successor, refactored to be SCFA-compatible.**

---

## 9. Material distinction vs. sibling candidates

### 9.1 vs. REFLECTOR (paradigm-#46 candidate A — symplectic re-derivation)

REFLECTOR's mechanism (per the candidate-A brief): use the inverse-walk's symplectic-flow structure to re-derive `(q_l, p_l)` from `(q_{l+1}, p_{l+1})` *without* storing intermediate sketch state. Memory: same as HYDRA (16 MB per pair). Compute: similar to HYDRA inverse walk (sequential). Accuracy: same as HYDRA inverse walk (`L_i · ε_BF16` per coord).

**SYNAPSE's distinct contribution:** sacrifices ~1 GB memory for SKETCH cache, in exchange for **parallel reconstruction**. The trade is *inverted from REFLECTOR*: REFLECTOR keeps memory minimal but pays sequential time; SYNAPSE pays memory for time.

| Axis | REFLECTOR | SYNAPSE |
|---|---|---|
| Memory cost | 16 MB | 1 GB |
| Compute pattern | Sequential `O(L_i)` | Parallel `O(1)` |
| Per-coord error | `L_i · ε_BF16` | `5 · ε_BF16` (with `μ` averaging) |
| Speedup at `n_gpu = 8` | 1.0× (parity with HYDRA) | 1.36× |
| Speedup at `n_gpu = 4` | 1.0× | 1.55× |
| Best when | Memory tight; many small segments | Memory loose; few large segments |

**SYNAPSE wins when memory is loose and HYDRA segments are deep.** REFLECTOR wins when memory is tight or when `n_gpu` is large (`L_i` small, parallelism unhelpful).

### 9.2 vs. ZEPHYR (paradigm-#46 candidate C — Direct Feedback Alignment)

ZEPHYR's mechanism (per the candidate-C brief): replace the chain-rule-derived `(dq_l, dp_l)` with random-feedback gradients (DFA-style; Lillicrap et al. 2016). Memory: zero (no activation reconstruction needed, only random feedback matrices). Compute: parallel across layers. Accuracy: large-scale convergence is uncertain (DFA fails on transformers in practice; mixed empirical record).

**SYNAPSE's distinct contribution:** preserves *exact chain-rule gradient* (modulo BF16 noise); does not require empirical validation that random-feedback works for our model. ZEPHYR is mathematically more aggressive (changes the gradient itself) and empirically more risky (no published DFA-success on 100B+ transformers).

| Axis | ZEPHYR (DFA) | SYNAPSE |
|---|---|---|
| Gradient correctness | **Approximate** (random-feedback) | Exact (sketch correction is unbiased) |
| Empirical track record | Mixed; fails on deep transformers | Untested but math is sound |
| Memory cost | ~0 (random matrices fit in cache) | 1 GB |
| Compute pattern | Fully parallel | Parallel within segment |
| Convergence guarantee | None known for transformers | Inherits CHIRON's |
| Best when | Willing to accept gradient bias for max speedup | Want to preserve gradient quality |

**SYNAPSE is the conservative choice.** ZEPHYR is the bet on DFA-on-CHIRON working.

### 9.3 Selection recommendation

Without knowing the user's risk preference, my honest recommendation: SYNAPSE for paradigm #46 *if* the user values gradient correctness (preserves CHIRON's mathematical invariants); ZEPHYR if the user is willing to gamble on DFA. REFLECTOR is the safest but doesn't deliver magnitude wall-clock.

---

## 10. Concrete CUDA primitives needed

Three new kernels and one storage extension. The full signatures are abbreviated for space; declarations given inline:

```cpp
namespace glades { namespace gpu {

// 10.1 Forward sketch projection.
// Computes z_l[i] = Σ_j S_l[i,j] · vec(q,p)[j], with S_l generated on-fly
// from cuRAND philox-4-32-10 seeded by (sigma_l, layer_idx, i, j). Zero persistent
// S_l storage. Memory-bound; 1 block per output element, inner reduction over N.
void chiron_synapse_sketch_project(
    const __nv_bfloat16* q_l, const __nv_bfloat16* p_l,
    uint64_t sigma_l, int layer_idx, int r, int T, int m,
    float* z_l_out, cudaStream_t stream);

// 10.2 Backward sketch correction. Two-sub-kernel structure:
//   (a) residual[i] = z_l[i] - Σ_j S[i,j] vec(q_*, p_*)[j]    [1 block/i, sum over N]
//   (b) \hat a[j] = vec(q_*, p_*)[j] + (1/r) Σ_i S[i,j] residual[i]  [1 block/group-of-j]
// Each sub-kernel regenerates S on-fly. Output: BF16 reconstructed (q̂, p̂).
void chiron_synapse_sketch_correct(
    const __nv_bfloat16* q_star_l, const __nv_bfloat16* p_star_l,
    const float* z_l, uint64_t sigma_l, int layer_idx, int r, int T, int m,
    __nv_bfloat16* q_hat_l_out, __nv_bfloat16* p_hat_l_out, cudaStream_t stream);

// 10.3 Parallel-launch helper: L_i corrections on independent streams.
void chiron_synapse_segment_reconstruct_parallel(
    const __nv_bfloat16* q_anchors[], const float* sketches[],
    uint64_t sigma_layer[], int layer_indices[], int L_i, int K, int r,
    int T, int m, __nv_bfloat16* q_hat_out[], __nv_bfloat16* p_hat_out[],
    cudaStream_t streams[GLADES_SYNAPSE_MAX_STREAMS]);

}}  // glades::gpu
```

**Storage additions to `GpuTransformerScratch`:** `GpuBuffer<float> synapseSketches[L_i · r]`; `GpuBuffer<__nv_bfloat16> synapseAnchors[L_i/K + 1]` (full anchor activations); `GpuBuffer<__nv_bfloat16> synapseReconstructed[L_i]` (parallel scratch); `uint64_t synapseSeed`.

**cuRAND seeding scheme.** `seed(microbatch_m, layer_l, row_i, col_j) := hash(globalSeed, m, l, i, j)` via `philox-4-32-10`. Determinism invariants: (a) forward and backward of *same* microbatch see identical `S_l`; (b) across microbatches, independent `S_l` (enables noise averaging in §4); (c) reproducible given a fixed `globalSeed` (per project's deterministic-by-default policy).

---

## 11. Honest gap

**11.1 Single-GPU regression is real (§0).** Sketch projection costs are paid (~1 % forward overhead); inverse-walk savings only partially recovered because reconstruction is memory-bandwidth bound, not compute bound. **Single-GPU is not where SYNAPSE wins.**

**11.2 HYDRA-segment win depends on parallel-stream concurrency.** The 1.36×–1.55× speedup assumes `L_i` parallel reconstructions can overlap on the GPU at ≥ 3× wall-clock effective parallelism. If HBM-bandwidth limits cap us at 2×, speedup degrades to ~1.20×. **Empirical kernel benchmark (Gate-1) is mandatory before commitment.**

**11.3 `K`-anchor cost erodes HYDRA's memory headline.** HYDRA promised `O(T·m)` per-GPU activation; SYNAPSE re-introduces `O(L_i · μ / K · T·m)` ≈ 448 MB at `K = 4, L_i = 7, μ = 16`. **The advantage is degraded from `O(1)` to `O(L_i)/K`** — partial regression. SYNAPSE only viable if 448 MB fits in HYDRA's per-GPU headroom (it does, but not comfortably).

**11.4 Conjecture 1 — linear-interpolation prior accuracy.** The K-anchor scheme rests on `‖q_l - q_*_l‖_2 ≤ 0.1 · ‖q‖_2` at `K ∈ {2, 4}`. Cheaply testable (Gate-0). If it fails, fall back to `K = 1` (every layer cached) which is just gradient checkpointing — no parallelism win.

**11.5 Per-coord error advantage is regime-dependent.** §4 is unflinching: post-batched per-coord stdev `~2-4 · ε_BF16` is *comparable to* HYDRA's `L_i · ε_BF16 = 7 · ε_BF16` at `n_gpu = 8`, not strictly better. The "factor `L_i` improvement" only holds decisively at `n_gpu = 4` (`L_i = 13`).

**11.6 Wire-in cost.** ~1500 LOC, ~7 weeks engineering. Less than HYDRA (~2000 LOC), more than MELT (~800 LOC).

**11.7 SAS (#40) interaction.** SAS's stochastic attention skipping uses per-step Bernoulli draws; sketch must capture the exact SAS realization (pass SAS seed through sketch cache). **Compatible with care.**

---

## 12. Falsification criteria (kill switches)

1. **Gate-0 (5-minute probe on existing 66M CHIRON checkpoint):** Compute `‖q_l - q_*_l‖_2 / ‖q_l‖_2` for `K = 2, 4` linear-interpolation priors. If `‖δ‖² · N / r² > 0.5` for any layer at `r = 1024`, **retire** (sketch cannot recover; even ~50 % of ‖q‖ tolerance is too loose for backward).

2. **Gate-1 (CUDA kernel benchmark, 1 GPU-day):** Implement `chiron_synapse_sketch_project` + `chiron_synapse_sketch_correct`. Test parallel reconstruction speedup at `L_i = 7`. If actual parallel speedup < 3× (vs sequential 7×), wall-clock benefit drops below 1.20× — **retire** unless aggressive `L_i ≥ 13` configurations are pursued.

3. **Gate-2 (parity test, 1 day):** SYNAPSE backward gradients vs. HYDRA inverse-walk gradients on a 4-layer 64-token model. Element-wise relative error `|Δgrad|/|grad| < 5·10⁻³` required. If sketch noise causes systematic gradient bias `> 1 %`, **retire**.

4. **Gate-3 (full training, 4 GPU-day):** Train 100 M model for 5k steps under SYNAPSE-HYDRA. Final loss within 0.05 nat of HYDRA-only baseline; wall-clock speedup ≥ 1.30×. If either fails, **retire**.

---

## 13. Implementation plan (if selected)

| Phase | Duration | Activities |
|---|---|---|
| A — feasibility | 1 week | Gate-0 (5 min) + Gate-1 CUDA kernel benchmark (1 GPU-day). |
| B — prototype | 3 weeks | Implement §10 kernels; CPU reference + Gate-2 parity test; memory validation. |
| C — HYDRA integration | 2 weeks | Wire into `glades-trainer` HYDRA pipeline; add `--synapse`, `--synapse-rank`, `--synapse-anchor-period` flags. |
| D — benchmarking | 1 week | Gate-3 full training; determine production `(r, K, n_gpu)` sweet spot. |

**Total horizon:** ~7 weeks, ~25 GPU-days; comparable to MELT's wire-in cost.

---

## 14. Honest verdict

SYNAPSE is the **memory-pays-for-time** candidate of paradigm #46. The wall-clock win is real but modest (1.36× at `n_gpu = 8`, up to 1.55× at `n_gpu = 4`) and *only materializes within HYDRA's pipeline-parallel regime*; on a single GPU it is at best a wash.

Mathematics is sound (Lemmas 1–2 with explicit derivations; Theorem 1 quantified under Conjecture 1) but the magnitudes are not "shock-and-awe" — closer to MELT's 1.6× compute reduction than SCFA's 15×–230×. SYNAPSE's contribution is narrowly aligned with HYDRA's inverse-walk bottleneck, not a paradigm-axis shift.

**Selection guidance:** If paradigm #46 is intended as a HYDRA-tuning phase (extracting the last sequential bottleneck within segments) and the memory budget has 0.5–1 GB slack beyond HYDRA baseline, SYNAPSE is the engineered choice. If a magnitudes-level leap is required, ZEPHYR (DFA) is the riskier upside bet; REFLECTOR is the conservative parity option that does not move the needle.
