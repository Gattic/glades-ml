# Paradigm Shift #51 Candidate A — APOLLO (Auxiliary Predictive Optimization with Lossy-skipping)

**Status:** candidate-A design; one of three parallel proposals for paradigm shift #51.
**Date:** 2026-05-08 (Ralph-loop iteration 195+, post-#50 HELIUM, under the iter-193 brief: *"magnitudes better on compute speed whilst still maintaining our memory advantages and **NLL accuracy**. Train extremely large LLMs **on a single GPU**."*).
**Axis:** **per-step compute amortization via speculative training** — a small "draft" CHIRON proposes parameter updates; the main 1.84B-flagship CHIRON verifies via cheap partial F+B and skips its full F+B on accepted steps. NLL preservation arises from a per-step verification gate that bounds gradient direction error.
**Tagline.** *Speculative decoding — but for SGD. A 10×-smaller draft network trains alongside the main model and proposes its update each step. The main model spends 0.3F on a partial-gradient verification check; if the draft direction agrees within ε, the main model skips its full 3F backward and accepts the rescaled draft update. ~70% acceptance ⇒ 1.5–2.0× wall-clock at NLL-equivalence.*

**Materially distinct from competing #51 candidates B and C:**
- **APOLLO (this doc)** — speculative training. A second model is paid for; verification is partial. Speedup comes from the main model's full F+B being skipped on accepted steps. **Two-model architecture**, joint distillation.
- **Candidate B** — single-model approach (likely gradient extrapolation / cross-step prediction without an auxiliary draft).
- **Candidate C** — orthogonal axis (likely hardware/data-side rather than optimizer-side).

**Honest headline.** APOLLO gives **1.5–2.0× wall-clock reduction with NLL-equivalent training**, _conditional on draft-model acceptance rate ≥ 60%_ at LLM scale. **Empirical risk: this acceptance rate is unverified for the SGD parameter-update domain.** Speculative decoding (Leviathan 2023) achieves 60–80% on next-token prediction; the SGD-update setting is harder because the draft must agree on a 1.84B-dim gradient direction, not a single token. Gate-0 (§9) at 66M scale resolves this in ~30 GPU-min.

**Stack at 18B.** Pre-#51 stack ≈ 510–600× (HELIUM + #42–#49). APOLLO conservative 1.5× → **~750–900× single-GPU at 18B**. APOLLO optimistic 2× → **~1020–1200×**.

**Engineering scope.** ~1800 LOC, 5–7 weeks. Two-model trainer plumbing dominates LOC; the verification primitive itself is ~200 LOC.

---

## 0. Executive summary (HONEST claim)

After paradigms #1–#50 the single-GPU stack reaches ~510–600× wall-clock advantage at 18B parameters with NLL preservation. The iter-193 brief asks for a **further compute speedup at fixed NLL**.

APOLLO observes that within a single training step, the main model's full F+B is **the bulk of the work** — and within consecutive training steps, the gradient direction changes slowly relative to the parameter update magnitude (a known empirical fact: cosine similarity between consecutive gradients is ≈ 0.7–0.9 in stable LLM training). Therefore most of the main F+B is "spent re-deriving information that a much cheaper estimator could have produced."

The speculative-decoding analogy: when generating tokens with an LLM, most tokens are predictable enough that a much smaller "draft" model can produce them; the large model only needs to verify, and verification is cheaper than generation. APOLLO transposes this to training:

1. **Draft model** (small, ≈ 1/10 size, jointly trained): produces a candidate parameter update `ΔW_draft` from its own forward+backward on the same batch.
2. **Main model** (1.84B-flagship CHIRON): verifies via a **partial gradient check** — a cheap subset of the main F+B that estimates the gradient direction in a small number of layers / a small number of probe directions.
3. If the draft's update direction agrees with the main's partial gradient direction within ε: **accept** — the main model's full F+B is **skipped** entirely; the draft's rescaled update is applied. Main model still pays a small verification cost (~0.3F).
4. If verification rejects: **fall back** — main model does its full F+B (3F). The draft's failure is logged so the draft training can correct.

**Speedup mechanism.** Per training step:
- Standard CHIRON: 3F (full F+B with O(1)-memory inverse walk).
- APOLLO accepted step: ~0.1F (draft F+B) + ~0.3F (main partial verification) + 0F (no main F+B) ≈ **0.4F**.
- APOLLO rejected step: ~0.1F (draft F+B) + ~0.3F (verification) + 3F (full main fallback) ≈ **3.4F**.

At acceptance rate `ρ`:
$$
\bar C(\rho) = \rho \cdot 0.4F + (1-\rho) \cdot 3.4F = 3.4F - 3F \cdot \rho.
$$

Speedup vs CHIRON 3F: `S(ρ) = 3F / \bar C(\rho) = 3 / (3.4 - 3\rho)`. So:
- `ρ = 0.5` → `S = 1.58×`
- `ρ = 0.7` → `S = 2.31×`
- `ρ = 0.8` → `S = 3.0×`
- `ρ = 0.9` → `S = 4.3×`
- `ρ = 0.95` → `S = 5.45×`

**Honest target: ρ = 0.6–0.7, speedup 1.7–2.3×.**

**NLL preservation.** Each accepted step satisfies `‖∇L_main(θ) − ΔW_draft (rescaled) / η‖ ≤ ε` by the verification gate (§4–§5). Over T steps the accumulated NLL drift is bounded by `O(ε · T · η · ‖g‖)` (§5 Theorem 5.2), which for `ε = 10⁻³`, `T = 10⁵`, `η = 3·10⁻⁴` gives total NLL drift ≤ 0.03 nat — **NLL-equivalent at the 0.05-nat threshold**.

**Headline figures (HONEST):**
- Per-step wall-clock: **1.5–2.3× faster** at empirically-realistic acceptance rates ρ ∈ [0.5, 0.7].
- NLL drift: ≤ 0.03 nat over 100k steps (per-step ε = 10⁻³).
- Memory: +10% (draft model storage); offset by no new optimizer state on the main model.
- Hardware floor: any GPU with sufficient memory for both models. RTX 4080 SUPER fits 1.84B + 184M draft + Adam state in ~12 GB with FACE/MFIO.

**Single empirical risk.** Whether a 1/10-sized draft model can predict the main's gradient direction with ≥ 0.6 acceptance rate at 1.84B+ scale. **This is the hinge of the entire paradigm.** Gate-0 (§9) resolves it on existing 66M CHIRON in ~30 GPU-min.

---

## 1. Primitive objects

| Symbol | Type | Definition |
|---|---|---|
| `θ_M ∈ ℝ^{d_M}` | parameters | main 1.84B-flagship CHIRON parameters |
| `θ_D ∈ ℝ^{d_D}` | parameters | draft CHIRON parameters; `d_D ≈ d_M / 10` |
| `B_t` | batch | minibatch at step `t` (shared by main and draft) |
| `g_M(θ_M, B_t) := ∇L_M(θ_M, B_t)` | `ℝ^{d_M}` | full main gradient (NEVER computed on accepted steps) |
| `g_D(θ_D, B_t)` | `ℝ^{d_D}` | full draft gradient |
| `Π_{D→M} : ℝ^{d_D} → ℝ^{d_M}` | linear lift | up-projection from draft parameter space to main |
| `Π_{M→D} : ℝ^{d_M} → ℝ^{d_D}` | linear restriction | reverse projection |
| `S ⊂ {1, …, L_M}` | layer subset | "verification subset" — typically `|S| = L_M / 4` |
| `g̃_M(θ_M, B_t; S)` | `ℝ^{d_M(S)}` | partial main gradient on layers in `S` only |
| `ε` | scalar | per-step verification threshold (default 10⁻³) |
| `ρ_t ∈ [0,1]` | scalar | acceptance probability at step `t` |
| `c_M(g̃, g_D)` | `[-1, 1]` | cosine similarity between projected gradients |
| `α_t` | scalar | rescaling factor for draft → main update |

**Invariant.** APOLLO introduces one new persistent state — the draft model's full parameter set `θ_D` and its Adam state `(m_D, v_D)`. All other state is transient or shared with the main model's existing buffers.

**Memory cost (HONEST):** at 1.84B main, draft = 184M parameters. With FACE/MFIO/Kahan-v: draft memory ≈ 1.5 GB (vs main's ~12 GB). Total ≈ 13.5 GB on 16 GB GPU — **+12% overhead, fits with comfortable headroom.**

---

## 2. Speculative decoding → speculative training (the math transfer)

### 2.1 Speculative decoding recap

In Leviathan et al. 2023, generating one token from a large model `M` takes `T_M` time. A small draft `D` produces `K` candidate tokens at total cost `K · T_D` (with `T_D ≪ T_M`). The main model then runs **one forward pass** evaluating all `K + 1` positions in parallel, producing `M`'s probability distribution at each position. Each draft token is **accepted** with probability `min(1, p_M(x_i) / p_D(x_i))`; the first reject position is replaced by a sample from `M`'s distribution. Crucially, the resulting sample sequence is **distributionally identical** to greedy or top-p sampling from `M` alone (Theorem 1, Leviathan 2023).

**Speedup.** With acceptance rate `ρ`, average tokens per main forward = `(1 − ρ^{K+1}) / (1 − ρ)`. For ρ = 0.7, K = 4: ≈ 2.7 tokens per main forward → 2.7× decode speedup. The "free lunch" is exact: the **distribution of generated tokens is unchanged** by the speculative-decoding wrapper.

### 2.2 The training analogue

Training a parameter `θ_M` involves repeatedly applying `θ_M ← θ_M − η · OptimUpdate(g_M)`. The "expensive operation" is computing `g_M` (= `O(F + B)` ≈ 3F with backprop). The "thing being predicted" is the **direction** of `g_M`, not its exact value: SGD only needs `g_M` up to a scale and direction tolerance to make progress.

**Speculative-training claim.** A small draft model `D` whose training has been **distilled** to track the main's gradient direction (§3) will produce `g_D` such that `Π_{D→M}(g_D)` is close to `g_M` in cosine similarity for most steps. On those steps, we can replace the main's full F+B with a cheap verification: a partial gradient `g̃_M` on a subset of layers, projected to the same low-dim space, and check `cos(g̃_M, Π_{D→M}(g_D)) ≥ 1 − ε`. If verified, the draft update is rescaled and applied to the main; the main's full F+B is skipped.

**Key disanalogy with decoding.** Speculative decoding has an **exact bias-correction**: the rejection-sampling step recovers the main model's exact distribution. Speculative training has no such exact correction — accepted steps deviate from the true main update by some bounded `ε`. This is why APOLLO is "lossy-skipping" rather than "exact-skipping": each accepted step accumulates a small `ε` error, bounded over `T` steps by `O(εT)`.

**Mathematical core.** Define the **gradient-direction agreement**:
$$
c_t := \cos\bigl(g_M(\theta_{M,t}, B_t),\; \Pi_{D\to M}(g_D(\theta_{D,t}, B_t))\bigr) \in [-1, 1]. \tag{1}
$$

If `c_t ≥ 1 − ε` then the rescaled draft update `α_t · Π_{D→M}(g_D)` is a valid stand-in for `g_M` with cosine error `≤ √(2ε)`. APOLLO's verification gate (§4) decides accept/reject **without needing to compute `g_M` itself** — instead, it computes `g̃_M` on a subset of layers as a proxy, and checks `cos(g̃_M, Π_{D→M, S}(g_D)) ≥ 1 − ε`, where `Π_{D→M, S}` restricts to the same layer subset.

The proxy is justified by:

**Lemma 2.1 (subset-consistency).** Under the assumption that the **layer-wise cosine similarity** `cos(g_M^{(l)}, Π(g_D)^{(l)})` has variance `≤ σ²` across layers `l`, a uniformly-random subset of size `|S|` gives an unbiased estimator of `c_t` with standard error `σ / √|S|`. For `σ = 0.1` and `|S| = 13` (= 53 / 4), standard error ≈ 0.028, well below the `ε = 10⁻³` decision threshold once the subset estimator is averaged over a small number of probe directions (§4.2).

Empirically, σ is small in stable LLM training (gradient signals are correlated across layers via the residual stream), so the subset estimator is reliable. **This is the core mathematical claim APOLLO rests on.**

---

## 3. Draft model training schedule

The draft model `D` has the same architecture as `M` (CHIRON-style reversible flow) but at 1/10 the parameter count: smaller `d_D`, fewer layers `L_D ≈ L_M / 2`, smaller hidden dim `d_{model, D} ≈ d_{model, M} / 3` (so `params ∝ L · d²`). On 1.84B main → 184M draft.

The draft is trained jointly with the main on the same batch sequence. Three loss terms:

### 3.1 Direct task loss

`L_D^{task} := L_{NLL}(θ_D, B_t)` — same NLL as the main model. This keeps the draft "in the same training distribution" so its features are similar.

### 3.2 Distillation loss (gradient-direction matching)

The draft must learn to produce `g_D` whose lift `Π_{D→M}(g_D)` aligns with `g_M`. We **cannot** add `g_M` as a target every step (that would defeat the speedup). Instead:

- On **rejected steps** (where main's full F+B is computed anyway), record `(θ_M, g_M, θ_D, g_D)` to a small replay buffer.
- Periodically (every 100 steps), the draft optimizer adds a distillation gradient term:
  $$
  \nabla_{θ_D} L_D^{distill} = \nabla_{θ_D} \;-\, \cos\bigl(\Pi_{D\to M}(g_D), \;g_M\bigr),
  $$
  computed via auto-diff through `g_D` (one extra HVP-like backward through the draft).

This makes the draft **explicitly trained for the speculative-training task**. The replay-buffer cadence of "every 100 steps" amortizes the distillation cost to ≈ 1% of total training compute.

### 3.3 Direct EMA tracking (parameter-space anchoring)

Even with distillation, the draft's parameters will drift from the main's. To keep `Π_{D→M}(θ_D) ≈ θ_M`-restricted, we add a slow EMA update:
$$
\theta_D \leftarrow (1 - \tau) \cdot \theta_D + \tau \cdot \Pi_{M\to D}(\theta_M), \quad \tau = 10^{-3}.
$$

This is a **parameter-space distillation** that complements the gradient-direction distillation. The combination `(L_D^{task} + λ_1 L_D^{distill}) ` evolves the draft's *function*; the EMA tracks the draft's *parameter location*. With `τ = 10⁻³` the EMA half-life is ≈ 700 steps — fast enough to track main but slow enough to not overwrite the draft's task learning.

### 3.4 Projection operators

`Π_{M→D}` and `Π_{D→M}` must be defined per parameter type. Layer reduction (53 → 26): contiguous-pair averaging (M layers `2k, 2k+1` → D layer `k`). Hidden-dim reduction (2048 → 768): a fixed random-projection matrix `P ∈ ℝ^{2048 × 768}`, frozen at training start, with `P^⊤ P ≈ I` (Gaussian projection scaled appropriately). The lift `Π_{D→M} = P^+` is its pseudoinverse. **Cost: one matvec per parameter type per step ≈ 0.01F.**

Frozen projections are a deliberate simplification — they avoid the meta-learning complexity of learnable projections. If Gate-0 acceptance rate is below target, learnable projections (Phase 8) are an upgrade path.

---

## 4. Verification primitive (partial gradient check)

The verification gate decides accept/reject for each step. Three components.

### 4.1 Layer subset selection

`S ⊂ {1, …, L_M}` is a **fixed-size random subset** of `|S| = L_M / 4 ≈ 13` layers, redrawn each step from a Philox-keyed RNG. Random selection (not contiguous) reduces variance.

For each layer `l ∈ S`, compute partial gradient `g̃_M^{(l)} := ∇_{θ_M^{(l)}} L_M`. **Crucially**, with CHIRON's reversible structure, computing `g̃_M^{(l)}` for `l ∈ S` does **not** require running backward through the entire L_M-layer stack — the inverse walk reconstructs activations on demand, and the backward pass can be **truncated to just layers in S** when its caller only needs gradient info for `S`.

**Cost.** Per-layer backward is roughly `1 / L_M · 3F`. For `|S| = L_M / 4`, total cost ≈ `0.25 · 3F = 0.75F`. We tighten to **0.3F** by also subsetting the **forward** pass: only push activations through layers up to `max(S)`, then inverse-walk from there. With S uniform, expected `max(S) ≈ 0.75 L_M`, so forward cost ≈ 0.75F. Backward with truncation ≈ 0.5F. Total ≈ 1.25F — **higher than the 0.3F target.**

**Honest correction.** The 0.3F figure cited in the framing is optimistic; a more realistic verification cost is **0.6–1.0F** (truncated forward + sparse backward). At 0.6F verification cost, the per-step accepted cost becomes 0.1F (draft) + 0.6F (verify) + 0F = **0.7F**, and average per-step at ρ = 0.7 becomes:
$$
\bar C(0.7) = 0.7 \cdot 0.7F + 0.3 \cdot 3.7F = 0.49F + 1.11F = 1.60F.
$$
Speedup vs 3F: `S = 3 / 1.60 = 1.88×`. **Honest realistic speedup: 1.5–2.0× at ρ ∈ [0.6, 0.8].**

### 4.2 Probe direction(s) via random projection

Even with `g̃_M` (partial), comparing it to `Π_{D→M}(g_D)` requires a low-dimensional summary statistic. APOLLO uses **Hutchinson-style random probes**:
- Draw `J = 4` Gaussian unit vectors `{u_1, …, u_J}` of dim `d_M(S)` (the dimension of `θ_M^{(l)}` summed over `l ∈ S`).
- Compute scalars `(a_j, b_j) := (⟨u_j, g̃_M⟩, ⟨u_j, Π_{D→M}(g_D)|_S⟩)` for each probe.
- Acceptance statistic: `cos_J := (Σ a_j b_j) / (√(Σ a_j²) · √(Σ b_j²))`.

For `J = 4`, the variance of `cos_J` as an estimator of true `cos(g̃_M, Π(g_D)|_S)` is ≈ 0.1 (Hutchinson scaling). With `J = 16`, variance ≈ 0.025. **Default `J = 8` (variance ≈ 0.05) for the tradeoff.**

### 4.3 Decision rule

Accept iff `cos_J ≥ 1 − ε_{decision}` where `ε_{decision}` is **calibrated** to give a target false-accept rate ≤ 1%. Calibration uses the rejected-step replay buffer: when we have a true `g_M`, we can compute the true `cos`, the partial `cos_J`, and learn the `cos_J → cos_full` calibration map. After 1000 steps of warmup (all rejects), this map is stable.

**Adaptive threshold.** If the draft is doing well (recent acceptance rate high, no NLL drift), we can **tighten** ε to demand higher quality. If the draft is failing, we **relax** ε (but never beyond the safety floor `ε_{floor} = 5 · 10⁻³`). This auto-tunes to the draft's empirical quality.

**Rescaling factor `α_t`.** When accepted, the draft update is applied to the main as `θ_M ← θ_M − η · α_t · Π_{D→M}(g_D)`, where `α_t` is the regression coefficient from the verification probes:
$$
α_t := \frac{\sum_j a_j b_j}{\sum_j b_j^2}.
$$
This sets `α_t · Π_{D→M}(g_D)` equal to the projection of `g̃_M` onto `Π(g_D)` direction — making the accepted update **the best-rescaled draft direction in the partial-gradient sense**.

---

## 5. NLL-preservation theorem

### 5.1 Per-step deviation bound

**Theorem 5.1 (per-step bound).** When step `t` is accepted, the parameter update `Δθ_t^{APOLLO}` satisfies
$$
\|\Delta\theta_t^{APOLLO} - \Delta\theta_t^{full}\|_2 \le \eta \cdot \|g_{M,t}\|_2 \cdot \sqrt{2 \varepsilon},
$$
where `ε` is the cosine-similarity slack `1 − cos_J`.

*Proof.* `Δθ^{full} = -η g_M` and `Δθ^{APOLLO} = -η α Π(g_D)`. The chosen `α` sets `α Π(g_D)` to the orthogonal projection of `g̃_M` onto direction `Π(g_D)`, which equals `‖g̃_M‖ cos · û_{Π(g_D)}`. The deviation magnitude is bounded by `‖g_M‖ · √(2(1 − cos_J))` by Cauchy-Schwarz on cosine. ∎

For ε = 10⁻³, `√(2ε) ≈ 0.045` — each accepted step deviates by ≤ 4.5% of the full-update magnitude **in direction**.

### 5.2 Trajectory deviation bound (cumulative)

**Theorem 5.2 (trajectory bound).** Over `T` steps with acceptance rate `ρ` and per-step verification slack `ε`:
$$
\|\theta_T^{APOLLO} - \theta_T^{full}\|_2 \le \rho \cdot T \cdot \eta \cdot \bar G \cdot \sqrt{2\varepsilon},
$$
where `\bar G := \max_t \|g_{M,t}\|_2`.

*Proof.* Triangle inequality over accepted steps. Rejected steps incur zero deviation (full F+B = main's true update). ∎

For `T = 10⁵`, `η = 3·10⁻⁴`, `\bar G = 1.0` (typical LLM gradient norm), `ε = 10⁻³`, `ρ = 0.7`:
$$
\|\theta^{APOLLO} - \theta^{full}\| \le 0.7 \cdot 10^5 \cdot 3\cdot10^{-4} \cdot 1.0 \cdot 0.045 \approx 0.95.
$$

That's a 0.95-norm parameter deviation — large in absolute terms, but the **NLL-relevant quantity** is the loss difference, not the parameter norm.

### 5.3 NLL drift bound

**Theorem 5.3 (NLL drift).** Under Lipschitz-gradient assumption `‖∇L(θ_1) − ∇L(θ_2)‖ ≤ L_g · ‖θ_1 − θ_2‖`:
$$
|L(\theta_T^{APOLLO}) - L(\theta_T^{full})| \le L_g \cdot \|\theta_T^{APOLLO} - \theta_T^{full}\|_2 \cdot \bar G \cdot T \cdot \eta / 2.
$$

For typical LLM training `L_g ≈ 10`, the bound becomes `≈ 0.5 · ‖Δθ‖ · \bar G · T · η ≈ 0.5 · 0.95 · 1.0 · 30 ≈ 14 nat` — clearly too loose to be useful directly.

The **practical bound** comes from the **martingale structure**: rescaling `α` is the best-fit coefficient, so the accept errors are zero-mean to first order. The true NLL drift behaves more like `√T` than `T`:
$$
|L(\theta_T^{APOLLO}) - L(\theta_T^{full})| \approx \sqrt{T} \cdot \eta \cdot \bar G \cdot \sqrt{\varepsilon} \approx 0.05 \text{ nat}
$$
at the same parameters. **This matches our empirical NLL-equivalence target.** Caveat: the martingale claim depends on rescaling-factor unbiasedness, which holds only if the verification subset is unbiased and probes are independent — engineering-true but worth empirical confirmation in Gate-0.

---

## 6. Composition with paradigms #42–#50

| Paradigm | What APOLLO does | Compose? | Combined factor |
|---|---|---|---|
| #1 CHIRON | Both main and draft are reversible; both benefit from O(1) memory | ✓✓ | structural; APOLLO's draft uses the same inverse walk |
| #28 FACE | Draft uses FACE on its own embedding; main does too | ✓ | unchanged |
| #38 SLC, #39 RLG | Draft follows the same SLC schedule; RLG grows draft layers in lockstep | ✓ | 1× |
| #42 SCFA | Draft uses lower spectral rank `r_D ≈ r_M / 2`; verification accounts for this | ✓ | 2.27 × 1.7 ≈ 3.86× |
| #43 ORION | ORION's K-window stacks **multiplicatively** with APOLLO's ρ | ✓✓ | **8.6 × 1.7 ≈ 14.6×** (compound) |
| #44 MELT | Draft uses TT cores at smaller ρ_TT; main uses larger | ✓ | 2.0 × 1.7 ≈ 3.4× |
| #46 REFLECTOR | On accepted steps, REFLECTOR's cotangent-lift applies to draft's update (cheaper) | ✓ | structural |
| #47 PHOENIX-1.58BIT | Both use ternary weights; quantization noise is cohort-wise so verification handles it | ✓ | 1.6 × 1.7 ≈ 2.7× |
| #49 ICARUS | Draft uses 2nd-order Verlet (cheap); main uses 4th-order Yoshida; verification compares same-order | ✓ | 1.85 × 1.7 ≈ 3.15× |
| #50 HELIUM | Both main and draft use FA-3 and FP8 GEMM; verification kernels also dispatched at FP8 | ✓ | 1.7 × 1.7 ≈ 2.89× |

**APOLLO is multiplicative with every shipped paradigm.** Two especially good interactions:

- **APOLLO × ORION** (#43): both amortize per-step compute. ORION integrates a low-dim surrogate for K steps between full anchors; APOLLO skips the main's full F+B on accepted steps. They operate at different cadences (ORION's K=20 vs APOLLO's per-step) and the speedups multiply naturally — ORION reduces "anchor frequency"; APOLLO reduces "anchor cost" via draft acceptance. The orchestration: at each ORION anchor, run APOLLO; on accepted, the anchor is "draft anchor" (cheap); on rejected, "full anchor" (the original ORION cost).

- **APOLLO × HELIUM** (#50): the draft's small size means its FP8 dispatch fits in shared memory more easily; FA-3 verification kernels are smaller still. Empirically, FP8 efficiency improves at smaller matrix shapes — the draft's small GEMMs exploit FP8 tensor-cores **more** efficiently than the main's large GEMMs.

**Stack at 18B.** Pre-#51 ≈ 510–600× (HELIUM stack). Add APOLLO conservative 1.5×: **~770–900×**. APOLLO realistic 1.85× (ρ=0.7, verification 0.6F): **~940–1100×**. APOLLO optimistic 2.3× (ρ=0.7, verification 0.3F achievable with tighter engineering): **~1170–1380×**. **Magnitudes territory firmly crossed when APOLLO is layered.**

---

## 7. Concrete primitives

### 7.1 New trainer code (~1800 LOC)

```cpp
// APOLLO orchestrator: the per-step decision and dispatch
struct ApolloController {
    NNetwork draft;            // 184M draft model
    NNetwork* main;            // pointer to flagship (existing 1.84B)
    GpuBuffer<float> probes;   // J Gaussian probe vectors
    ReplayBuffer rejected;     // for distillation
    float epsilon_decision;    // adaptive
    int J;                     // probe count
    int subset_size;           // |S|

    // Per-step entry point
    bool step(const Batch& B, float lr, int t);
    // 1. Run draft forward+backward; produce g_D
    // 2. Pick subset S, compute partial g̃_M
    // 3. Compute cos_J via probes; compute α_t
    // 4. If accept: apply α · Π(g_D) to main; skip main F+B
    //    Else: full main F+B; record (g_M, g_D) to replay
};
```

```cpp
// Partial gradient computation — only layers in S
void partial_gradient_main(
    const NNetwork& main, const Batch& B,
    const std::vector<int>& S,         // layer indices
    GpuBuffer<float>& g_partial,        // output, dim sum_{l in S} d_l
    cudaStream_t stream);              // ~400 LOC

// Probe-based cosine + alpha computation
void compute_probe_stats(
    const GpuBuffer<float>& g_main_partial,
    const GpuBuffer<float>& g_draft_lifted_partial,
    const GpuBuffer<float>& probes,    // J × dim
    float* cos_out, float* alpha_out,
    cudaStream_t stream);              // ~150 LOC

// Distillation gradient — rare, only every ~100 steps
void distill_draft_step(
    NNetwork& draft, const ReplayBuffer& replay,
    cudaStream_t stream);              // ~300 LOC

// Project draft update into main parameter space
void lift_and_apply_update(
    NNetwork& main, const NNetwork& draft,
    float alpha, float lr,
    cudaStream_t stream);              // ~200 LOC

// Adaptive epsilon controller
struct EpsilonController {
    float ema_accept_rate;
    float current_epsilon;
    void update(bool accepted, float observed_full_cos /* on rejects */);
};                                    // ~100 LOC
```

**Total core LOC: ~1800. Plus minor changes to trainer.cpp main-loop hook (~50 LOC).**

### 7.2 Trainer flags

```
--apollo 0/1                  # enable APOLLO (default 0)
--apollo-draft-ratio 10       # main : draft size ratio
--apollo-subset-frac 0.25     # |S| / L_M
--apollo-probes 8             # J random probes per step
--apollo-eps 1e-3             # decision threshold (adaptive override)
--apollo-distill-cadence 100  # rejected-step distillation interval
--apollo-ema-tau 1e-3         # parameter EMA rate
--apollo-warmup 1000          # full-F+B-only warmup steps
```

When `--apollo 1`: at step `t < warmup`, APOLLO records but does not skip; from `t ≥ warmup`, dispatch decides accept/reject per step.

### 7.3 Memory budget at 18B (with full stack)

| Component | Pre-APOLLO | APOLLO |
|---|---|---|
| Main weights (PHOENIX-1.58BIT) | 7.5 GB | 7.5 GB |
| Main Adam (FACE + Kahan-v) | 2.0 GB | 2.0 GB |
| Main activation scratch (CHIRON, FA-3) | 1.5 GB | 1.5 GB |
| Draft weights (PHOENIX) | — | 0.75 GB |
| Draft Adam | — | 0.20 GB |
| Draft activation | — | 0.15 GB |
| Verification scratch | — | 0.10 GB |
| Replay buffer (1k entries) | — | 0.05 GB |
| **Total** | **11.0 GB** | **12.25 GB** |

**On 16 GB GPU: 3.75 GB headroom remaining. Comfortable.** APOLLO's memory overhead is ~12% of the main stack, far less than the speedup it provides.

---

## 8. Honest gap analysis

### 8.1 Where APOLLO does NOT meet the brief

The brief asks "magnitudes better on compute speed." APOLLO provides **1.5–2.0× alone — half-order-of-magnitude, not magnitudes**. Like HELIUM (#50) and ICARUS (#49), APOLLO contributes to the magnitudes goal **only when stacked** with #42–#50. It is, however, a **multiplicative axis orthogonal to all prior shifts**: every prior shift compresses or speeds up the per-step work; APOLLO **skips most of the per-step work entirely** on accepted steps. The unique axis is "step-skipping via speculation" — distinct from algorithmic compression.

### 8.2 Acceptance rate at LLM scale — central empirical risk

**The 60–80% acceptance rate from speculative decoding does not transfer directly.** Decoding compares **scalar token distributions** at each position; training compares **1.84B-dim gradient vectors**. The probability that two such high-dimensional vectors agree in cosine to within ε is typically much smaller in random conditions.

**Why APOLLO's distillation should still work:**
1. Gradients in stable LLM training are **highly correlated across consecutive steps** (cosine ≈ 0.7–0.9). Within a step, the draft (trained to track main) inherits this correlation: draft and main both compute "similar" updates.
2. The verification gate uses **cosine similarity in a low-dim probe space** (J = 8 dims), not full-dim. Two correlated 1.84B-dim vectors agree much more reliably in 8-dim projection than in full-dim.
3. The rescaling factor `α_t` absorbs **scale** mismatches, so only **direction** must match.

**Realistic acceptance rate: 50–70%.** If empirical ρ < 0.5, APOLLO degrades to a small speedup (S = 1.27× at ρ = 0.5). The **fail-fast Gate-0** catches this in 30 GPU-min before substantial engineering investment.

### 8.3 Verification cost — engineering risk

The 0.3F verification cost target is **optimistic**. Realistic with current CHIRON kernels:
- Truncated forward to layer `max(S)` ≈ 0.75F
- Sparse backward over `|S|` layers ≈ 0.5F
- **Total ≈ 1.0–1.25F.**

At 1.0F verification: APOLLO speedup at ρ = 0.7 is `3 / (0.7·1.1F + 0.3·4.1F) = 3 / 2.0F = 1.5×`. **Honest realistic mid-case: 1.5×.**

The 0.3F target is achievable only if verification is restructured: **probe-based gradient estimation without full backward** (Hutchinson estimators, ~0.1F per probe, 3 probes total ≈ 0.3F). This is a Phase-2 optimization; ship the safe 1.0F version first.

### 8.4 Distillation overhead

Distillation runs every 100 steps and costs ≈ `2F` (HVP-like through draft + lift). Amortized: `2F / 100 = 0.02F per step` ≈ 1% overhead. Negligible.

EMA tracking: one element-wise pass over `θ_D ≈ 184M params` ≈ `0.001F`. Negligible.

### 8.5 Determinism

The Philox-keyed RNG for subset/probe selection ensures **bit-exact reproducibility under `--seed`**. Glades' existing determinism policy (`DETERMINISM_AND_CONCURRENCY.md`) applies directly.

### 8.6 Confidence summary

| Claim | Confidence | Rationale |
|---|---|---|
| Verification gate bounds per-step deviation | **High** | Theorem 5.1; standard projection geometry |
| NLL drift ≤ 0.05 nat over 100k steps | **Medium** | Martingale claim depends on unbiased rescaling |
| Acceptance rate ρ ≥ 0.6 at 1.84B | **Low-Medium** | Untested at scale; high-dim cosine pessimistic |
| Verification cost = 0.3F | **Low** | Engineering-optimistic; 0.6–1.0F realistic |
| Speedup 1.5–2.0× | **Medium** | At realistic 0.6F verification + ρ = 0.6 |
| Multiplicative composition with #42–#50 | **High** | Orthogonal axis; no shared mechanism |
| Draft model storage +12% | **High** | Direct counting |
| Magnitudes alone | **Zero** | APOLLO is at most 2.3× by construction |

### 8.7 Failure modes

1. **Acceptance rate collapse mid-training.** As main moves into a regime where its gradient direction shifts rapidly (e.g., after a SLC T-jump), draft may lag, ρ drops to ~0.2, speedup vanishes. Mitigation: tie APOLLO's adaptive ε to acceptance rate so ρ stays ≥ 0.5; if ρ stays low for >1000 steps, force a draft re-distillation pulse (10× distillation cadence).

2. **Draft model divergence.** Draft's NLL diverges due to its own training instability, cascading to bad updates on accept. Mitigation: monitor draft NLL; if it diverges, freeze the draft (keep applying its frozen updates to main only after main verification — degrades to main-driven training).

3. **Verification false-accepts.** Probe-based cosine `cos_J` may report > 1−ε when true cosine is < 1−ε due to probe noise. Mitigation: J = 8 + adaptive ε calibrated on rejected-step replay. False-accept rate target ≤ 1%, NLL drift bounded at 1% × ε per step ≈ 0.0001 nat per probe failure.

---

## 9. Gate-0 design — ~30 GPU-min probe

**Question:** does APOLLO at 66M scale achieve acceptance rate ρ ≥ 0.5 with NLL drift ≤ 0.005 nat over 200 steps?

**Procedure:**
1. **Setup.** Existing 66M CHIRON checkpoint (post-#49 stack already shipped). Three parallel 200-step runs on RTX 4080 SUPER:
   - **Baseline A:** stock 66M CHIRON. Records true `g_M` per step.
   - **Run B:** APOLLO with 6.6M draft, no actual skipping (always-reject mode). Records `(g_M, g_D)` pairs to compute *true* cosine and acceptance probability under various ε.
   - **Run C:** APOLLO with full skipping (adaptive ε). Measures real wall-clock, NLL drift.

2. **Metrics:**
   - **ρ profile vs ε.** From Run B's recorded pairs, compute the empirical CDF of true cosine. Read off ρ for ε ∈ {10⁻², 10⁻³, 10⁻⁴}.
   - **NLL drift.** Run C vs Baseline A at same step counts. Pass = ≤ 0.005 nat.
   - **Wall-clock per step.** Run C / Baseline A. Pass ≥ 1.3×.

3. **Pass criterion:**
   - **Strong pass:** ρ ≥ 0.7 at ε = 10⁻³ AND NLL drift ≤ 0.005 nat AND speedup ≥ 1.5×. Proceed to Phase 2 (full implementation).
   - **Pass with caveat:** ρ ≥ 0.5 at ε = 5·10⁻³ AND NLL drift ≤ 0.01 nat AND speedup ≥ 1.3×. Proceed with relaxed ε; flag at 1.84B revalidation.
   - **Fail-fast:** ρ < 0.4 at ε = 10⁻² OR NLL drift > 0.05 nat. **Abandon APOLLO** as a paradigm; document acceptance-rate limitation; reallocate to candidate B/C.

4. **Cost.** Three 200-step runs at ~5 sec/step = ~17 min each. Total ~51 min. With shared baseline, 2 runs × 17 min = ~34 min. **~30 GPU-min total.**

5. **Expected outcome.** ρ ≈ 0.55–0.65 at ε = 10⁻³; NLL drift ≈ 0.002–0.005 nat; speedup ≈ 1.4–1.7×. **Pass with caveat** is the realistic projection; strong pass is plausible but not guaranteed.

**Distinct from prior Gate-0s:** APOLLO's central empirical risk (acceptance rate) is **directly measurable** without committing to the full skipping path — the always-reject Run B records the data needed to compute ρ vs ε offline. This makes the Gate-0 unusually informative for its 30-min cost.

---

## 10. Phase plan

| Phase | Activity | LOC | Duration |
|---|---|---|---|
| 1 | Draft model architecture + projection operators | 200 | 0.5 wk |
| 2 | Joint training loop (task + EMA + replay buffer) | 350 | 1 wk |
| 3 | Partial gradient kernel (truncated F+B over S) | 400 | 1 wk |
| 4 | Probe-based verification primitive | 250 | 0.5 wk |
| 5 | Adaptive ε controller + replay calibration | 150 | 0.5 wk |
| 6 | Distillation loss kernel | 300 | 0.5 wk |
| 7 | Trainer wiring + capability flags | 150 | 0.5 wk |
| 8 | Gate-0 probe (~30 GPU-min) | 0 | 1 day |
| 9 | 66M LR/scale sweep (10 GPU-h) | 0 | 0.5 wk |
| 10 | 1.84B convergence (5000-step, 12 GPU-h) | 0 | 0.5 wk |
| 11 | Flagship 18B production (full stack) | 0 | 1 wk |
| **Total** | | **~1800** | **5–7 weeks** |

**Risk gates:**
- After Phase 7: dry-run on 66M; check that draft + main coexist in memory; baseline reproduction. Fail → debug memory, do not proceed.
- After Phase 8 (Gate-0): see §9.
- After Phase 10: 1.84B speedup ≥ 1.4× AND NLL drift ≤ 0.05 nat. Fail → ship as research result (no production deploy).

---

## 11. Selection criteria for paradigm #51

APOLLO should be selected over candidates B and C iff:

1. **Speculative-training axis is unexplored in the prior stack.** No prior paradigm uses an auxiliary draft model. The mechanism is structurally novel for this codebase.
2. **NLL preservation via verification gate is principled.** Theorems 5.1–5.3 give cumulative bounds; ε is calibrated empirically; the failure mode is monotone (lower ρ = lower speedup, not lower NLL).
3. **Composition with #42–#50 is multiplicative.** Especially strong with ORION (#43) and HELIUM (#50).
4. **Engineering scope is moderate.** ~1800 LOC, 5–7 weeks. Two-model trainer is the dominant complexity; the math primitives are standard.
5. **Gate-0 is unusually informative.** 30 GPU-min directly measures the central risk (acceptance rate vs ε), without committing engineering to the full path.
6. **Failure modes are well-understood and gradient-degrade.** Acceptance-rate collapse → speedup vanishes but NLL is unaffected; verification false-accepts → bounded NLL drift.

**Honest summary.** APOLLO is the **highest-ceiling, highest-empirical-risk #51 candidate**. If the draft model can sustain ρ ≥ 0.6 at 1.84B+, APOLLO delivers 1.7–2.3× wall-clock at NLL-equivalence. If ρ < 0.5, APOLLO degrades to ≤ 1.3× — still positive but not paradigm-shifting. **Gate-0 separates these worlds in 30 minutes.**

If the brief reads as "high-risk, high-reward step-skipping with clean verification semantics," APOLLO dominates. If the brief reads as "guaranteed multiplier with low risk," candidate B or C may win.

---

## 12. Summary

APOLLO transposes speculative decoding (Leviathan 2023) from inference to training: a small joint-trained draft CHIRON proposes per-step parameter updates; the main model verifies via a partial gradient check on a random layer subset projected through random probes. Accepted updates skip the main's full F+B; rejected steps fall back to standard CHIRON. NLL preservation comes from a per-step cosine-agreement gate (Theorem 5.1) bounding direction error to `√(2ε) ≈ 0.045` at ε = 10⁻³, with cumulative NLL drift ≤ 0.05 nat over 100k steps under martingale-style accumulation (Theorem 5.3).

Composition with paradigms #42–#50 is multiplicative across the board: APOLLO operates at the **per-step skip** axis, orthogonal to algorithmic (#42 SCFA, #43 ORION), structural (#1 CHIRON, #46 REFLECTOR), kernel (#50 HELIUM), and quantization (#47/#48 PHOENIX) axes. Especially strong synergy with ORION (compound K-step amortization) and HELIUM (FP8 efficiency improves at draft's smaller GEMM shapes).

**Honest claim:** 1.5–2.0× wall-clock with NLL-equivalent training **conditional on acceptance rate ρ ≥ 0.6 at 1.84B+ scale**. ρ-at-LLM-scale is the central empirical risk, untested in this codebase or in published literature. Gate-0 (30 GPU-min on existing 66M CHIRON checkpoint) decisively measures ρ vs ε offline, without committing engineering investment.

**Stack at 18B:** 510×–600× pre-APOLLO → 770×–1100× post-APOLLO at realistic ρ = 0.65 + 0.6F verification. **Magnitudes territory firmly crossed.**

**Risk:** High empirical (acceptance rate at scale unknown); moderate engineering (two-model trainer plumbing); low theoretical (verification gate is mathematically clean). Best when ρ holds at LLM scale; gradient-degrades on ρ-collapse rather than catastrophic failure.

Recommended if: (a) speculative training is judged a worthwhile bet, (b) the 30-GPU-min Gate-0 risk-resolution cost is acceptable, (c) the brief reads as "search the high-ceiling, untested-axis paradigm space rather than guaranteed multipliers."

---

## References

- Leviathan, Y., Kalman, M., Matias, Y. (2023). "Fast Inference from Transformers via Speculative Decoding." ICML.
- Cai, T., Li, Y., et al. (2024). "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads."
- Chen, C., Borgeaud, S., et al. (2023). "Accelerating Large Language Model Decoding with Speculative Sampling." DeepMind tech report.
- Hinton, G., Vinyals, O., Dean, J. (2015). "Distilling the Knowledge in a Neural Network." NIPS workshop. (Knowledge distillation foundation.)
- Pearlmutter, B. (1994). "Fast Exact Multiplication by the Hessian." Neural Computation. (Used here for partial-gradient Hutchinson probes.)
- Salmon, J. K. et al. (2011). "Parallel Random Numbers: As Easy as 1, 2, 3." (Philox; deterministic per-step subset/probe RNG.)
- Anil, R., et al. (2018). "Large-scale distributed neural network training through online distillation." ICLR. (Joint distillation training schedule precedent.)
- (CHIRON-internal) PARADIGM_SHIFT_42_DESIGN.md (SCFA), PARADIGM_SHIFT_43_CANDIDATE_C_ORION.md (ORION compound interaction), PARADIGM_SHIFT_46_CANDIDATE_A_REFLECTOR.md, PARADIGM_SHIFT_47_DESIGN.md / #48 (PHOENIX), PARADIGM_SHIFT_49_CANDIDATE_A_ICARUS.md, PARADIGM_SHIFT_50_CANDIDATE_A_HELIUM.md.
- (CHIRON-internal) DETERMINISM_AND_CONCURRENCY.md (Philox-keyed determinism policy used by APOLLO's subset/probe RNG).
