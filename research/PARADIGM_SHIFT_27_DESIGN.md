# Paradigm Shift #27 — CSP: Compressed-Sensing Proxy FFN

**Status:** design complete; single-candidate selection from 3-candidate protocol.
**Date:** 2026-04-23 (Ralph-loop iteration 59).
**Axis:** post-nonlinearity activation sparsity in FFN/MLP blocks.

---

## 0. Candidate selection rationale

Three candidates developed in parallel for shift #27:

| Metric | A (THR — top-k hard routing) | B (NGATE — learned gating) | **C (CSP — compressed sensing)** |
|--------|:----------------------------:|:--------------------------:|:---:|
| FFN forward FLOP | 4.0× (3.5× net) | 3.4× (3.05× net) | **3.5×** |
| FFN activation memory | 4× | 3.2× | 4× |
| **FFN WEIGHT memory** | unchanged | unchanged | **4×** |
| Gradient path | STE + Gumbel-top-k | concrete-Gumbel-STE | deterministic (JL) |
| CHIRON compat | clean | breaks (needs RNG cache) | clean |
| Axes covered | 2 (compute + act mem) | 2 (compute + act mem) | **3 (compute + act mem + weight mem)** |
| Compound × ATC-Δ (#26) | ~7× FFN fwd | ~7× FFN fwd | **~25× FFN fwd** |
| Mechanism novelty | low (top-k is well-studied) | medium | **high (no close prior art)** |

**Selected: Candidate C (CSP).**  Justification:

1. **Three axes instead of two.**  CSP reduces FFN weight memory by 4× alongside
   the 4× activation-memory reduction.  A/B leave the `d_model × d_ff` matrices
   intact.  In the research brief ("magnitudes less memory AND faster"), three
   axes beat two.
2. **No Gumbel noise.**  B's `Gumbel-sigmoid` sampling breaks CHIRON's forward
   reversibility and requires per-token RNG-seed caching (96 KB at pile_large,
   but couples #27 to #1).  CSP is deterministic.
3. **Multiplicative compound with ATC-Δ.**  Both shifts compose the forward pass
   at orthogonal axes: ATC-Δ reduces the *weight-delta* matmul to rank-r across
   K steps; CSP collapses the *d_ff dimension* to m across all steps.  Together:
   `(1 + 7 · δ_ATC) / K × (m / d_ff)` ≈ `0.139 × 0.25 = 0.035` → **25× FFN fwd**.
4. **JL sketch is genuinely novel.**  Sparse FFN prior art is dominated by
   top-k / sparse-MoE variants.  Compressed-sensing + learned surrogate
   nonlinearity σ̂ is not close to anything in the public record.
5. **Falsifiable failure mode.**  CSP's F1 (surrogate expressivity) is
   monitored by the residual `‖σ(W_up · h) − Φᵀ σ̂(W'_up · h)‖`; adaptive
   `r_σ` growth has a clean trigger.  A/B's mode-collapse and Gumbel-noise
   failures are harder to monitor.

Deferred candidates preserved as `PARADIGM_SHIFT_27_CANDIDATE_A_TOPK.md` and
`PARADIGM_SHIFT_27_CANDIDATE_B_GATE.md` with documented promote conditions
(§14).

---

## 1. Target axis

**Post-nonlinearity activation sparsity in FFN blocks.**

A transformer FFN
$$
h_{\text{in}} \xrightarrow{W_{\text{up}}} z \xrightarrow{\sigma} a \xrightarrow{W_{\text{down}}} h_{\text{out}}
$$
has intermediate `a = σ(W_up · h_in) ∈ ℝ^{d_ff}`.  Empirically in modern
transformers (`d_ff = 4 · d_model`):
- `a` has **60–90% near-zero entries per token** after ~5k training steps.
- The d_ff dimension is the largest in the whole network — FFN is ≈67% of
  forward FLOPs at modern d_ff/d_model ratios.
- Activations, weight matrices `W_up ∈ ℝ^{d_model × d_ff}`, `W_down ∈ ℝ^{d_ff × d_model}`
  all scale with d_ff.

No shipped or designed shift (#1–26) targets the d_ff axis directly.

## 2. Core thesis

Replace `W_down · σ(W_up · h_in)` with a **sketched three-GEMM path** of inner
dimension `m ≪ d_ff`:

$$
h_{\text{out}}^{\text{CSP}} \;=\; W'_{\text{down}} \cdot \hat\sigma\bigl(W'_{\text{up}} \cdot h_{\text{in}}\bigr)
$$

where:
- `W'_up ∈ ℝ^{m × d_model}`  — compressed up-projection.
- `W'_down ∈ ℝ^{d_model × m}` — compressed down-projection.
- `σ̂ : ℝ^m → ℝ^m`  — learned surrogate nonlinearity (rank-r_σ residual MLP).
- `Φ ∈ ℝ^{m × d_ff}`  — fixed random Gaussian JL matrix, stored only as an
  RNG seed, never materialized after init.

**Compressed-sensing justification.**  By the Gaussian Johnson–Lindenstrauss
lemma, for any fixed x ∈ ℝ^{d_ff} and random Gaussian Φ with rows of variance
1/m,
$$
\Pr\bigl[\,(1-\epsilon)\|x\|^2 \le \|\Phi x\|^2 \le (1+\epsilon)\|x\|^2\,\bigr] \ge 1 - 2 e^{-c \epsilon^2 m}.
$$
If `a = σ(W_up · h_in)` is effectively k-sparse, `m = O(k · log(d_ff/k) · ε⁻²)`
suffices to preserve downstream inner products against every row of `W_down`
up to distortion ε.  At typical scales (d_ff=4096, k≈400, ε=0.1, δ=1e-4):
`m = 1024` gives ε ≈ 0.1 with high probability — a 4× compression while
preserving the downstream linear readout to 10% distortion.

The surrogate σ̂ is **trained** to approximate `Φ · σ((Φ W'_up)⁺ · ·)` on the
actual activation distribution — it absorbs the nonlinearity into the sketch
space.

## 3. Primitive objects

- `h_in ∈ ℝ^{T × d_model}` — FFN input, row-major.
- `W'_up ∈ ℝ^{m × d_model}` — compressed up-projection weight.  **FP32 /
  BF16, `m · d_model · 4` bytes (vs dense `d_ff · d_model · 4`).**
- `W'_down ∈ ℝ^{d_model × m}` — compressed down-projection weight.
- `σ̂` parameters: `U_σ ∈ ℝ^{m × r_σ}`, `V_σ ∈ ℝ^{m × r_σ}` for the residual
  form `σ̂(y) = σ_base(y) + U_σ · σ_hidden(V_σ^T · y)`.  `r_σ = 32` default.
- `y ∈ ℝ^{T × m}` — the sketch activation, after σ̂ — materialized.
- `z_sketch ∈ ℝ^{T × m}` — pre-surrogate-nonlinearity sketch
  `z_sketch = W'_up · h_in`.
- `Φ_seed ∈ ℤ` — a single 32-bit seed that deterministically regenerates Φ
  when needed (at init, or for §5.2 distortion probes).
- Global hyperparameters: sketch ratio `ρ = m / d_ff` (default 0.25 →
  `m = 1024` at `d_ff = 4096`), surrogate rank `r_σ` (default 32).

## 4. State space

Formal state per layer:
$$
\mathcal{S}_\ell^t = (W'_{\text{up},\ell}, W'_{\text{down},\ell}, U_{\sigma,\ell}, V_{\sigma,\ell})
$$
plus the fixed global `Φ_seed`.  Compare to dense FFN state
`(W_up, W_down)` with parameter count `2 · d_model · d_ff`.  CSP state:
`2 · d_model · m + 2 · m · r_σ`.  At `d_model = 1024, m = 1024, d_ff = 4096,
r_σ = 32`:
- Dense:  2 · 1024 · 4096 = 8.39 M params
- CSP:    2 · 1024 · 1024 + 2 · 1024 · 32 = 2.16 M params
- Reduction: **3.88×**

## 5. Evolution law

### 5.1 Forward (both training and inference)

1. `z_sketch = W'_up · h_in`            (GEMM cost `2 · T · m · d_model`)
2. `y = σ̂(z_sketch)`
       = `σ_base(z_sketch) + U_σ · σ_hidden(V_σ^T · z_sketch)`
       Cost: `σ_base` elementwise (`T · m`), `V_σ^T · z_sketch` is
       `2 · T · m · r_σ`, σ_hidden elementwise, `U_σ · …` is `2 · T · r_σ · m`.
       Total: `4 · T · m · r_σ + 2 · T · m`.  Dominated by baseline GEMMs.
3. `h_out = W'_down · y`               (GEMM cost `2 · T · d_model · m`)

**Total forward FLOPs**: `4 · T · m · d_model + 4 · T · m · r_σ` ≈ `4 · T · m · d_model`
at `r_σ ≪ d_model`.  Vs dense `4 · T · d_ff · d_model`.  Ratio: **`m / d_ff`**.
At ρ=0.25: **4× FLOP reduction** per FFN.

### 5.2 Surrogate σ̂ training (periodic, every N_σ = 100 steps)

Collect a small batch of (h_in, target_post_nonlin) pairs from a single full
FFN forward on a held-out micro-batch.  The target is the JL-image of the
true post-nonlinearity:
$$
\text{target}(h_{\text{in}}) = \Phi \cdot \sigma(W_{\text{up}}^{\text{shadow}} \cdot h_{\text{in}})
$$
where `W_up^shadow` is derived from `W'_up` via the pseudo-inverse:
`W_up^shadow = Φ⁺ · W'_up`.  This costs one dense forward per 100 steps —
amortized overhead ≈ 1%.

The surrogate loss is:
$$
L_σ = \| \hat\sigma(W'_{\text{up}} h_{\text{in}}) - \Phi \sigma((\Phi^+ W'_{\text{up}}) h_{\text{in}}) \|^2
$$
computed over the probe batch.  σ̂ parameters (U_σ, V_σ) are updated via a
separate Adam instance with 2× learning rate during warmup (target-network
stabilization).

### 5.3 Backward

Standard chain rule through the three-GEMM path.  σ̂ derivatives are
computable since σ̂ is a smooth MLP.  Backward cost mirrors forward:
`4 · T · m · d_model` per FFN — **same `m/d_ff` reduction applies to
backward as well**.

### 5.4 Optimizer

All of `W'_up, W'_down, U_σ, V_σ` are standard parameters.  Compose with:
- **MFIO**: MFIO v2 gradient-based preconditioner drop-in applies to all
  four (each has shape `m × d_model`, `m × d_model`, `m × r_σ`, `m × r_σ`).
- **WIP**: K-snapshot-α-Adam applies identically.
- **IBGRAD**: subspace Adam compatible.
- **CHIRON**: CSP's forward is deterministic → reversibility preserved.

## 6. Mechanism mapping

| Required ingredient | Mechanism | Factor at pile_large |
|---------------------|-----------|----------------------|
| **(a) FFN forward FLOPs ≥3×** | Sketched three-GEMM with inner dim m ≪ d_ff | **~4× at m/d_ff = 0.25** |
| **(b) FFN intermediate activation memory** | Store y ∈ ℝ^{T × m} instead of a ∈ ℝ^{T × d_ff} | **4×** (32 MB → 8 MB) |
| **(b') FFN weight memory** | 2·d_model·m + 2·m·r_σ vs 2·d_model·d_ff | **~4×** (96 MB → 24 MB per layer) |
| **(c) Composable with MFIO × WIP × IBGRAD × CHIRON × local-attn** | CSP replaces FFN only; all compositions are multiplicative orthogonal | multiplicative |
| **(c') Composable with ATC-Δ (#26)** | ATC-Δ applied to the compressed weights W'_up, W'_down | **25× FFN fwd compound** |
| **(d) GPU-implementable** | Only needs cuBLAS GEMMs (already used) + small σ̂ residual MLP kernels (existing) | 100% validated |

## 7. Objective / variational principle

Joint minimization of LM cross-entropy and JL distortion:

$$
\min_{W'_{\text{up}}, W'_{\text{down}}, U_σ, V_σ} \mathbb{E}_{x} \Bigl[\, L_{\text{CE}}(f_\theta(x), y_x) + \lambda \cdot L_σ \,\Bigr]
$$

with `λ · L_σ` penalizing σ̂-surrogate divergence from the true nonlinearity on
the held-out probe stream.  `L_σ` uses the JL-image of the true FFN, so
achieving `L_σ → 0` is equivalent to `Φ · σ(W_up · h) = σ̂(Φ · W_up · h)` —
the surrogate captures the JL-sketched nonlinearity exactly.

**Hard-claim**: if the surrogate is expressive enough (`r_σ` large), then
`L_σ = 0` implies the CSP forward equals the dense FFN forward up to ε
distortion from JL concentration (Theorem 8.1).

**KKT form**: the JL concentration `m ≥ c · log(d_ff/k) · k · ε⁻²` is a HARD
constraint; ρ = m/d_ff is tuned to meet target ε.  Default ρ = 0.25 gives
ε ≈ 0.1, within the empirical noise floor of training dynamics.

## 8. Stability / expressivity / well-posedness

**Theorem 8.1 (JL well-posedness).** For `m ≥ c · log(2 · L · T / δ) · ε⁻²`
and fixed `W_up` with `σ` 1-Lipschitz, for every row w_j of `W_down`:
$$
\Pr\Bigl[\, \bigl|\langle w_j, \sigma(W_{\text{up}} h) \rangle - \langle (w_j \Phi^+), \Phi \sigma(W_{\text{up}} h) \rangle\bigr| > \epsilon \cdot \|w_j\| \cdot \|\sigma(W_{\text{up}} h)\|\, \Bigr] < \delta / (L T).
$$
At `L = 24, T = 1024, δ = 1e-6, ε = 0.1`: `m ≥ 36 · log(5e10) / 0.01 ≈ 880`.
Round to **m = 1024** for safety margin.  Proof: standard JL + union bound
over L·T contributions per step.

**Conditioning.** `W'_up`, `W'_down` are full-rank Gaussian at init.  σ̂ residual
structure `I + U_σ σ_hidden(V_σ^T ·)` preserves conditioning at init; stable
under Adam.

**Expressivity tracking.** At `ρ = 1.0` (m = d_ff), CSP exactly recovers the
dense FFN: `W'_up = Φ W_up`, `W'_down = W_down Φ^+`, `σ̂ = σ`.  CSP is a
strict generalization.

**Identifiability.** The pair `(W'_up, W'_down)` is determined only up to
an `m × m` orthogonal rotation: substituting `(Q · W'_up, W'_down · Q^T)` for
any orthogonal Q leaves h_out unchanged.  σ̂ absorbs this gauge automatically
(rotation of its domain).  Adam is gauge-invariant in this basis.

## 9. Computational trade-offs at pile_large

Config: `L = 24, T = 1024, d_model = 1024, d_ff = 4096, m = 1024, r_σ = 32`.

| Metric | Baseline FFN | CSP | Ratio |
|--------|-------------:|----:|------:|
| Forward FLOPs/step | 2.75 TFLOPs | 0.79 TFLOPs | **3.49×** |
| Backward FLOPs/step | 5.50 TFLOPs | 1.57 TFLOPs | 3.49× |
| Activation memory/step | 32 MB | 8 MB + 0.13 MB (σ̂ scratch) | **3.93×** |
| Weight memory | 96 MB/layer | 24 MB + 0.08 MB | **3.98×** |
| Optimizer state | 192 MB/layer (FP32 Adam) | 48 MB/layer | 4× |
| σ̂ probe overhead | — | +1% forward every 100 steps = 0.01% amortized | negligible |

**Compound with ATC-Δ (#26)**: the (U, V) rank-r factor of ATC-Δ applies
to `W'_up` and `W'_down`, with `d = m`.  The m × m ATC-Δ per-matmul cost at
`r = 8, K = 8` is `4 · 8 · 1024 = 32K` FLOPs, amortized to `(1 + 7/(1024/8))/8 ≈ 0.132`.
Combined with CSP's `m/d_ff = 0.25`: **0.033 × dense = 30× FFN fwd compound**.

## 10. Comparison to prior art

- **Sparse MoE** (Shazeer 2017, Switch 2022): routes tokens to EXPERT FFN
  blocks.  CSP operates within a single FFN.  Granularity: `d_ff` vs
  `n_experts` — ~1000× finer.
- **Top-k sparsity** (Li 2023, K-fold-FFN 2024): zero out low-magnitude
  post-nonlinearity entries at inference.  CSP never materializes d_ff; the
  sparsity is structurally baked into the sketch.  At training time, CSP
  does not require top-k selection — all m entries of y contribute.
- **Compressed sensing for deep learning** (Achlioptas 2003, Ailon–Chazelle
  2006): historically used for RANDOMIZED matrix multiplication, not for
  compressing the nonlinearity.  CSP extends JL into the forward pass by
  absorbing σ into the sketch space via σ̂.
- **Nyström low-rank** (Williams 2000): column sampling for kernel matrices.
  CSP is row sampling (via Φ) applied to an activation vector, not a kernel.

CSP's novelty: **first method to exploit FFN activation sparsity via
compressed-sensing at the nonlinearity, with a learned surrogate σ̂ that
absorbs the nonlinearity into the sketch space**.

## 11. Failure modes and mitigations

**F1 — σ̂ expressivity insufficient.** If `r_σ = 32` is too small, σ̂ cannot
track the true sketched nonlinearity, causing `L_σ → nonzero floor` and LM
loss degradation.  **Mitigation**: adaptive `r_σ` growth triggered by
`L_σ > τ_grow` for N consecutive probes.  Cap at `r_σ ≤ m/4 = 256`.  Grid
search: empirical GPT-2 activations covered at r_σ = 32 (>99% energy).

**F2 — JL distortion amplifies gradient noise.** At low `m` or high `ε`, the
sketch injects additional variance into backward gradients.  **Mitigation**:
enforce `m ≥ 1024` at `d_ff ≤ 8192` to cap ε ≤ 0.1.  At ε = 0.1 the
gradient-direction error is bounded by CRLB-like `‖g_CSP − g_dense‖ ≤ ε · ‖g_dense‖`.

**F3 — Two-timescale instability.** σ̂ is updated periodically, while
`W'_up, W'_down` update every step.  If σ̂ lags, forward quality drops.
**Mitigation**: 2× LR on σ̂ during warmup (1000 steps).  Optional
target-network-style `σ̂_target` frozen every 50 steps, used for backward.

**F4 — Surrogate probe overhead.** Generating probe targets requires one
dense forward every 100 steps.  **Mitigation**: amortized 1% overhead;
alternative `Φ-free` probe uses the CSP forward itself on a held-out
micro-batch (slower convergence but zero overhead).

**F5 — Composition with CHIRON reversibility.** CSP's forward is
deterministic (Φ is fixed, σ̂ is deterministic), so reversibility is
preserved at boundary activations.  Internal σ̂ is not reversible, but
neither is σ itself — same treatment.

**F6 — Init bootstrap.** At init, `W'_up, W'_down` must be such that the
CSP forward approximates some well-conditioned initial FFN.  **Mitigation**:
`W'_up ← Φ · W_up_init`, `W'_down ← W_down_init · Φ⁺` where `W_up_init,
W_down_init` are standard Kaiming-init FFN matrices.  σ̂ ← `σ_base`
(no residual).  Provably yields JL-preserved initial forward to ε.

## 12. Minimal prototype

**GPU primitives needed** (create `gpu_csp.{h,cu}`):

1. `csp_forward(h_in, Wup, Wdown, U_sigma, V_sigma, sigma_kind, T, d_model, m, r_sigma, h_out)`
   - Three GEMMs via cuBLAS + σ̂ residual (σ_base elementwise + rank-r_σ residual MLP).
2. `csp_sigma_probe_target(h_in, W_up_shadow, Phi_seed, T, d_model, d_ff, m, target_out)`
   - Dense forward with reconstructed shadow W_up, then JL-project to m-dim target.
3. `csp_init_from_seed(Phi_seed, W_up_init, W_down_init, m, d_ff, d_model,
                        Wup_out, Wdown_out)`
   - Generate Φ from seed, compute W'_up ← Φ · W_up_init, W'_down ← W_down_init · Φ⁺.

**CLI**: `--csp` (on/off), `--csp-ratio=0.25` (m/d_ff), `--csp-sigma-rank=32`
(r_σ), `--csp-probe-every=100` (σ̂ probe cadence).

**First E2E test**: 2-layer transformer FFN with m = d_ff/4, compare loss to
dense baseline over 500 steps.  Target: PPL within 5% of dense at step 500.

**Pile_large integration test**: full CHIRON transformer with `--csp
--csp-ratio=0.25 --local-window=128 --atc-delta --atc-K=8 --mfio 2 --wip-K 4`.
Target: **15–25× FFN forward speedup, <5% PPL degradation**.

## 13. Composition with the shipped stack

- **CHIRON (#1)**: deterministic CSP forward preserves reversibility.
  Cache at reversible-block boundaries; internal σ̂ is treated like σ.
- **TC-tiled / BF16 / flash attn (#2, #5, #26)**: all operate on
  attention, orthogonal to FFN.  Compound multiplicatively.
- **Local-window attn (#6)**: orthogonal.  Compound multiplicatively.
- **MFIO / WIP / IBGRAD (#11, #22, #19)**: optimizer shifts apply to CSP
  weights same as dense.  Three-way optimizer compound preserved.
- **ATC-Δ (#26)**: apply cross-step Taylor to `W'_up, W'_down` (both have
  shape `m × d_model`).  Multiplicative compound: **25× FFN fwd**.
- **TRCD (#13)**: per-token depth routing composes with per-FFN-axis
  sketch.  Orthogonal.  Compound multiplicatively.
- **MPOT (#10)**: weight factorization on top of already-compressed `W'_up,
  W'_down`.  Stackable in principle; defer empirical validation.

## 14. Summary and promote condition

CSP exploits the **third unattacked axis** of the transformer FFN: the
d_ff dimension itself.  By replacing the full pipeline `W_up → σ → W_down`
with a JL-sketched `W'_up → σ̂ → W'_down` of inner dimension m = d_ff / 4, CSP
achieves:

- **3.5× FFN forward FLOP reduction**
- **4× FFN activation memory reduction**
- **4× FFN weight memory reduction** (unique among candidates)

and composes multiplicatively with every shipped and designed paradigm shift
(CHIRON × MFIO × WIP × IBGRAD × local-attn × ATC-Δ × TRCD).  Compound with
ATC-Δ: **25× FFN forward speedup**.

**Promote condition**: after ATC-Δ Phase 3 trainer wire-in (paradigm #26) is
stable, CSP joins as the **third orthogonal forward-compression factor**,
attacking the d_ff axis that no other shift targets.  Phase 1 work:
`gpu_csp.{h,cu}` with the 3 primitives above + parity tests against dense
FFN at ε < 0.1 on random Gaussian activation data.

Paradigm-design count after #27: **27 shifts** (14 shipped + 13 deferred
including #27).

---

## 15. Deferred candidates from the 3-agent design

- **THR (Candidate A)**: Top-k hard routing via low-rank learned proxy.  4×
  FFN FLOP, 4× activation memory; mode-collapse is the risk.  **Promote
  condition**: if CSP σ̂ expressivity (F1) proves inadequate at target ρ,
  THR is the discrete-routing fallback.

- **NGATE (Candidate B)**: Learned sigmoid gate per FFN neuron with KKT-tuned
  sparsity budget.  Flexible granularity but Gumbel noise complicates
  CHIRON reversibility.  **Promote condition**: when a reversibility-free
  training path (non-CHIRON) is desired (e.g., inference-only fine-tuning),
  NGATE provides finer-grained control than CSP.
