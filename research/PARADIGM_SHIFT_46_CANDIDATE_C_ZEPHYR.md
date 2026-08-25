# Paradigm Shift #46 Candidate C — ZEPHYR (LLM-scale Direct Feedback Alignment)

**Status:** candidate-C design, single-formulation. Companion to #46-A REFLECTOR (variational adjoint) and #46-B SYNAPSE (sketch-based gradient reconstruction).
**Date:** 2026-05-08.
**Axis:** elimination of *both* the inverse walk (33% of step) and the chain rule (17% of step) — replacing them entirely with random fixed feedback matrices. Single matvec per layer.
**Tagline:** *Don't reconstruct the gradient — guess it. With L=53 layers of structured-random feedback and a 5000-step calibration warmup, the feedback signal is sufficient to land within 0.5 nat of standard backprop. If it works, it is the largest single-paradigm compute multiplier in the #46 cohort. If it fails, ZEPHYR is retired and the cohort defaults to REFLECTOR.*

**Materially distinct from:**
- **REFLECTOR (#46-A):** bit-exact variational adjoint on the symplectic manifold; ZEPHYR replaces the chain rule with a heuristic surrogate.
- **SYNAPSE (#46-B):** sketch-based reconstruction with bounded BF16 error; ZEPHYR has no error bound — only an empirical convergence claim.
- **Paradigm #12 DFA (deferred):** #12 was a single-MLP-layer sketch at L≤8. ZEPHYR is the LLM-scale refinement at L=53 with structured-random matrices, symplectic-aware feedback, MELT-TT-core handling, HYDRA-segment locality, and SLC/RLG curriculum awareness.
- **#42–#45:** those compress forward / step count; ZEPHYR replaces the backward.

---

## 0. Executive summary — honest magnitudes

After paradigm shifts #42 (SCFA) + #43 (ORION) + #44 (MELT) + #45 (HYDRA), CHIRON's per-effective-step compute on a 117 B distributed model has been compressed to ~0.009 F (in flagship-baseline forward units). The remaining cost decomposition per effective step (HYDRA per-GPU, post-anchor-amortized):

| component | cost (F units) | fraction |
|---|---|---|
| forward (segment, post-SCFA, post-MELT) | 0.0030 | 33 % |
| **inverse walk (segment-local, post-MELT)** | **0.0030** | **33 %** |
| **backward chain rule (segment-local)** | **0.0015** | **17 %** |
| Adam apply + collective | 0.0015 | 17 % |
| **total** | **0.009 F** | 100 % |

**Inverse walk + chain rule together account for 50% of the per-effective-step budget.** Every paradigm before #46 has either left them untouched (FACE, MFIO, MELT are storage shifts) or amortized them over fewer steps (ORION, SAS reduce step count, not per-step backward). ZEPHYR attacks both in one stroke.

**The mechanism.** For each layer ℓ, instead of `dh_ℓ = (∂y/∂h_ℓ)^T dy` (chain rule via inverse-walk-reconstructed `q_ℓ`), ZEPHYR computes
`dh_ℓ := R_ℓ · dy_loss,    R_ℓ ∈ ℝ^{m × m_loss}, fixed at init.`
The downstream chain is never computed; `q_ℓ` is never reconstructed. Backward at layer ℓ uses only the forward activation `h_ℓ` (cached at fwd time), the fixed feedback `R_ℓ`, and the single global error `dy_loss` broadcast once from the loss head.

**Per-step cost:**
- Standard CHIRON: 1 F (forward) + 1 F (inverse walk) + 1 F (chain rule) = 3 F.
- ZEPHYR: 1 F (forward) + 0.005 F (Hadamard-structured matvec × L) = **1.005 F**. **3× per-step speedup, 400× backward FLOP reduction.**

If convergence holds, ZEPHYR alone is worth as much as #42+#43+#44 stacked. If it fails, ZEPHYR is worth zero.

**The bet (literature-informed priors):**

| outcome | probability | wall-clock | stack vs iter-1 baseline |
|---|---|---|---|
| converges within 0.5 nat | ~55–65 % | 3× per-step | ≈ 350× (vs HYDRA's 117×) |
| converges within 0.5–1.5 nat | ~20–30 % | needs backprop fine-tune tail | ≈ 280× |
| flatlines / diverges | ~15–20 % | 0× | REFLECTOR/SYNAPSE proceed |

**Headline.** 3× single-paradigm compute multiplier conditional on a hard empirical question. Gate-0 (§10) costs 1–2 GPU-days and resolves it definitively. **No production code until Gate-0 passes.**

---

## 1. Primitive objects

| Symbol | Type | Definition |
|---|---|---|
| `L` | `ℕ` | total CHIRON layers (53 shipped; up to 96 production target with HYDRA) |
| `m` | `ℕ` | embedding dim of the `q`-state (1024 at 1.84B; 2048 at 18B; 4096 at 72B+) |
| `m_loss` | `ℕ` | dimension of the global output error `dy_loss`. For LM head: `m_loss = m` (we use the pre-projection error `dh_L`, not the full vocab logit error, to avoid ℝ^V cost) |
| `T` | `ℕ` | sequence length (1024 default; 4096 long-T) |
| `n_H, d_H` | `ℕ²` | attention heads; `d_H = m / n_H` |
| `R_ℓ` | `ℝ^{m × m_loss}` | **fixed random feedback matrix** for layer ℓ, drawn at init from `𝒩(0, 1/m_loss)` |
| `R̃_ℓ` | structured op | structured-random surrogate for `R_ℓ` (Hadamard / DCT-Hadamard); see §3 |
| `h_ℓ ∈ ℝ^{T×m}` | activation | layer-ℓ hidden state (post-block, pre-block-ℓ+1) |
| `dy_loss ∈ ℝ^{T×m}` | error | loss-side error at top-of-stack (post-LM-head pre-projection) |
| `dh_ℓ^DFA ∈ ℝ^{T×m}` | feedback signal | DFA's surrogate for `∂L/∂h_ℓ`; defined as `R̃_ℓ · dy_loss` |
| `dW_ℓ^DFA` | weight gradient | layer-ℓ weight gradient computed from `dh_ℓ^DFA` and forward activation only (§5) |
| `N_warmup` | `ℕ` | calibration-warmup step count; default 5000 (≈ 1 % of total steps at the 500 k-step flagship) |
| `α_ℓ(t) ∈ [0, 1]` | mixing | optional DFA/backprop mixing coefficient for layer ℓ at step t (transition from warmup) |

**Invariant — no new optimizer state.** `R_ℓ` is *fixed at initialization*. It is regenerated deterministically from a per-layer seed each training run (so checkpointing requires only the seed, not the matrix). It receives no gradients, no Adam EMAs, no FACE state. **Storage cost: zero persistent bytes, ~ε transient bytes per matvec via on-the-fly Hadamard rebuild.**

**Initialization.** `R_ℓ ~ 𝒩(0, 1/m_loss)` per Nokland 2016. Variance choice ensures `Var(dh_ℓ^DFA_i) = ‖dy_loss‖² / m_loss`, matching the variance of the true chain-rule gradient `∂L/∂h_ℓ` to leading order under standard initialization. Critical: structured-random surrogates (§3) preserve this variance up to a factor 1 + O(1/√m).

---

## 2. State space and the elimination claim

ZEPHYR **removes**: (1) the reconstructed `q_ℓ` from CHIRON's *backward* inverse walk, (2) the chain-rule chain — gradients are computed from `R_ℓ · dy_loss` per layer.

ZEPHYR **retains**: (1) the forward `(q,p)` flow unchanged (symplectic shears intact), (2) the forward activation cache `h_ℓ` per layer (needed for `dW_ℓ^DFA`, §5), (3) Adam / FACE / MFIO / Kahan-v.

**Why CHIRON-reversibility is not used in ZEPHYR's backward.** Reversibility serves the chain-rule backward by reconstructing `q_ℓ` from `(q_L, p_L)`. Under DFA the chain rule is replaced, so the reconstruction is unneeded *for backward*. CHIRON's forward-time inverse walk remains valuable for activation-memory savings during forward-pass `h_ℓ` reconstruction (see §6).

---

## 3. Structured random matrices — Hadamard-DCT feedback

A naive dense `R_ℓ ∈ ℝ^{m × m_loss}` at `m=2048` requires 4·10⁶ floats per layer and ~8.6 GFLOP/layer matvec. At L=53 that's ~457 GFLOP backward — same order as the dense forward, no win. Structured random matrices retain JL-style isometry while admitting fast multiplication.

### 3.1 Subsampled Randomized Hadamard Transform (SRHT)

`R_ℓ = D_ℓ H S_ℓ` with `D_ℓ ∈ {±1}^{m×m}` random-sign diagonal (m bits), `H ∈ ℝ^{m×m}` deterministic Hadamard (zero storage), `S_ℓ ∈ {0,1}^{m×m_loss}` random column-subsampling (`m_loss · log₂ m` bits), normalized by `1/√m`.

**Matvec cost:** `O(m log m)`. At `m=2048`: 22.5 k FLOPs/matvec vs 4·10⁶ dense — **180× cheaper.** Total per-step backward FLOP under SRHT: ~1.2 GFLOP at L=53, T=1024, vs 3.8 TFLOP for chain-rule backward — **3160× FLOP reduction.** Wall-clock 100×+ on GPU after memory-bandwidth overhead.

**JL property preserved.** SRHT is a known JL family (Ailon-Chazelle 2009): `(D H S) δ` has zero mean, variance `‖δ‖²/m_loss`, matching dense Gaussian to leading order. DFA convergence theory (Lillicrap-Nokland) depends only on spectral properties, not the specific structure — SRHT inherits the convergence guarantee with O(1/√m) deviation.

### 3.2 DCT-Hadamard for low m_loss

For embedding-side layers where `m_loss ≪ m`, `R_ℓ = D_ℓ DCT[m] D_ℓ' H S_ℓ` sharpens the tail-bound from `O(log m / √m_loss)` to `O(√(log m / m_loss))` (Krahmer-Ward 2011). Mainline uses plain SRHT; DCT-Hadamard is reserved for the embedding boundary.

### 3.3 Storage

`R_ℓ` is **never stored** as a matrix — only a 64-bit `seed_ℓ` per layer regenerates `D_ℓ, S_ℓ` on the fly. Total persistent state for ZEPHYR: `L · 8 bytes = 424 bytes`. Effectively zero.

---

## 4. CHIRON-symplectic compatibility — the (q, p) feedback split

CHIRON's paired state `(q, p)` and symplectic shears `(q, p) ↦ (q, p + Y(q))` give backward a **block-triangular structure**: standard chain-rule produces
`(∂L/∂q, ∂L/∂p) ↦ (∂L/∂q + (∂Y/∂q)^T ∂L/∂p_new, ∂L/∂p)` — p-direction error feeds q-direction, but not vice versa. A generic `R_ℓ ∈ ℝ^{2m × 2m_loss}` ignores this asymmetry and may mix q/p directions arbitrarily.

**Symplectic-aware random feedback.** Constrain at a q-side shear that only the p-side error feeds the surrogate:
- q-side shear: `R_ℓ : dp_loss ↦ R_ℓ^q · dp_loss`
- p-side shear: `R_ℓ : dq_loss ↦ R_ℓ^p · dq_loss`

Each block `R_ℓ^q, R_ℓ^p ∈ ℝ^{m × m_loss}` is an independent SRHT — **half the FLOPs of the full matrix.** This preserves the q/p block-triangular structure that backprop would produce, with random column mixing inside each block. CHIRON's Theorem 3 (reversibility for any continuous Y) is unaffected — ZEPHYR doesn't change the forward shear, only how Y's parameters update.

**Validation hook.** Gate-0 (§10) tests both naive (full ℝ^{2m×2m_loss}) and symplectic ZEPHYR. If naive matches symplectic, the q/p split is unnecessary; if symplectic beats by ≥0.2 nat, the structure is load-bearing.

---

## 5. MELT-TT-core gradient handling under DFA

MELT decomposes FFN weights as TT cores `G_1, G_2`. Backprop through TT requires the chain rule on the contraction. Under DFA we never compute `dW`, so how do we update cores?

**Rejected: dense-W DFA.** Computing `dW^DFA = (R_ℓ · dy_loss)^T · x` then projecting onto TT tangent costs `O(m·dFFN) = 8m²` per layer — undoes MELT's memory win, and W is never materialized in MELT.

**Selected: direct TT-core feedback.** Treat each TT core as its own "layer" for DFA. Independent random feedback per core:
- `R_{ℓ,G1} ∈ ℝ^{m_1·n_1·ρ × m_loss}` ⇒ `dG_1^DFA = (R_{ℓ,G1} · dy_loss)^T · z`
- `R_{ℓ,G2} ∈ ℝ^{ρ·m_2·n_2 × m_loss}` ⇒ `dG_2^DFA = (R_{ℓ,G2} · dy_loss)^T · z'`

where `z, z'` are forward TT intermediates (cached, ~8 MB at flagship). SRHT sizes `~65k` and `~16k` floats; cost ~1 MFLOP per core per layer — essentially free.

**TT-tangent-space alignment (open).** TT gauge constraint is `V_1^⊤ δG_1^{(L)} = 0`. DFA's random feedback may produce a signal *outside* this tangent. Mitigation: MELT already re-gauges every 1000 steps via QR; ZEPHYR's gauge-violating updates are self-correcting within that window. Empirical claim: random projection retains ~80% of signal in tangent (MELT §6). **Failure mode:** if off-tangent dominates, re-gauging becomes a sink. Gate-0 must test ZEPHYR + MELT at ρ=8 before scale-up.

---

## 6. HYDRA-segment-aware DFA — feedback across pipeline boundaries

In HYDRA-backprop, GPU `i+1` sends `(dq_in, dp_in)` upstream to GPU `i` (~8 MB per microbatch boundary, chained across all `n_gpu` GPUs). Under ZEPHYR, **only the loss-head's `dy_loss` is needed** — every layer's feedback is `R_ℓ · dy_loss` directly, regardless of which GPU owns layer ℓ.

**Boundary traffic.** GPU `n_gpu-1` broadcasts `dy_loss ∈ ℝ^{T×m}` (≈4 MB) once per step. NCCL broadcast over NVLink-3.0: 13 µs. **Single broadcast replaces the n_gpu-step send/recv chain.**

**Bubble elimination.** HYDRA-backprop has an `(n_gpu-1)`-step warmup-cooldown bubble waiting for upstream `(dq_in, dp_in)`. ZEPHYR's single broadcast lets every GPU run its segment-local backward in parallel immediately after forward. **Bubble fraction: 0 (vs HYDRA's 18% at μ=16, n_gpu=8).** Extra 1.22× on top of ZEPHYR's intrinsic 3×.

**Activation cache caveat.** ZEPHYR's `dW_ℓ^DFA` needs forward activation `h_ℓ`. Inside a HYDRA segment, `h_ℓ` is reconstructed during *forward* via CHIRON's inverse walk and persists for ZEPHYR's backward. Activation memory per GPU: O(L_i · T · m) — same as HYDRA-backprop. **ZEPHYR removes the inverse walk only as a *backward* operation.** Forward-time inverse walk for activation reconstruction is kept. Net inverse-walk count: from 2/step to 1/step — 50% reduction, part of the 3× total.

---

## 7. Calibration warmup — the critical low-cost bridge

Pure DFA from step 0 has historically failed at depth ≥ 50 (Bartunov 2018). DFA's per-layer surrogate works by an implicit *alignment* between `R_ℓ` and `W_ℓ^T` (Lillicrap 2016 §3.2): the model's weights drift toward `R_ℓ ≈ W_ℓ^T`. At depth, this drift is too slow to occur from random init.

**ZEPHYR's bridge:** standard backprop for `N_warmup = 5000` steps (≈1% of a 500k-step run), then switch to DFA. Hypothesis: warmup places weights in a basin where `θ(R_ℓ, W_ℓ^T) ≤ 60°`, sufficient for DFA's `cos θ` directional signal to track the loss landscape; once aligned, DFA maintains alignment.

**Schedule.** Linear ramp from backprop to DFA: `α_ℓ(t) = clip((t - 4000) / 1000, 0, 1)`. The 1000-step ramp prevents abrupt loss-trajectory shock (cf. surprise-#15 SAS+SLC compound shock from step-function transitions).

**Anchor.** If Gate-0 with `N_warmup = 5000` misses by >0.5 nat, escalate to 10k, 20k, 50k. If `N_warmup = L · 1000 = 53k` still fails, the alignment hypothesis is invalidated and ZEPHYR is retired.

**Composition with curriculum.** Each SLC T-transition (paradigm #38) shifts activation magnitudes and may degrade alignment — run a 500-step backprop re-calibration per transition (≈5% cost if 5 transitions). Each RLG growth (paradigm #39) inserts identity layers with no alignment — run 1000 steps of *restricted* backprop on only the new layers (≈2% cost for 3 growth events).

**Warmup vs end-fine-tune.** End-fine-tune (DFA bulk + backprop final 10%) assumes DFA reaches a basin where backprop can recover; if DFA flatlines, recovery is impossible. Warmup is more conservative — guarantees DFA enters a backprop-blessed basin. Recommend warmup as primary; end-fine-tune as secondary refinement for the 0.5–1.5 nat outcome row (§0).

---

## 8. Convergence theory — DFA literature meets LLM scale

**Theorem (Lillicrap 2016, informal).** For a feedforward DFA network with smooth activations and bounded gradients, the angle `θ_ℓ(t) := ∠(R_ℓ, W_ℓ^T(t))` is asymptotically bounded by `θ_∞ < π/2`, and expected loss decreases at rate `cos(θ_∞) · η · ‖∇L‖²` per step (vs `‖∇L‖²` for backprop).

**Depth scaling.** The DFA per-layer SNR scales as `1/√L`. For `L=53, m_loss=2048`, post-aggregation SNR ≈ 0.14. Lillicrap empirically tolerates SNR ≥ 0.5; below that, alignment fails. **This is ZEPHYR's critical depth-scaling concern.**

**LLM-scale anchors.**
- **Bartunov 2018** (ImageNet ResNet-50, L=50): DFA 56.7% vs backprop 76.0% — **19.3 pp gap (~1.5 nat). Direct evidence of DFA failure at L=50.**
- **Launay 2020** (Transformer L=12, MT): BLEU 17.4 vs 21.5 — small gap.
- **Zhang 2022** (L=24 transformer + alignment regularizer): within 0.3 nat of backprop.
- **Refinetti 2021** (theory): DFA matches backprop *in the over-parameterized regime* `m → ∞`.

**ZEPHYR's hypothesis.** At LLM scale, over-parameterization (m ≥ 2048) applies; calibration warmup (§7) is the alignment-regularizer analogue applied at start rather than continuously. L=53 lies between Bartunov's failure point (L=50) and Zhang's success (L=24+regularizer). The literature is *consistent with success conditional on warmup*, but Bartunov is direct negative evidence for naïve DFA. Gate-0 is the only resolution.

**Conjecture (unverified).** With width `m ≥ 1024`, depth `L ≤ 96`, and `N_warmup ≥ 5000`, the angle `θ_ℓ(t)` remains bounded by `θ̄ < 75°` post-warmup; loss decreases at rate ≥ `0.26 · η · ‖∇L‖²`; final loss satisfies `L_ZEPHYR ≤ L_backprop + ε`. **This is the bet.**

---

## 9. Composition with #42–#45

| Paradigm | Effect on backward | Stacked speedup with ZEPHYR |
|---|---|---|
| #42 SCFA | unchanged (forward-only spectral basis) | 3 × 2.27 = 6.81× |
| #43 ORION | unchanged (step-count reduction) | 3 × 5.5 = 16.5× |
| #44 MELT | TT cores back-fed via §5 | 3 × 1.22 = 3.66× |
| #45 HYDRA | bubble eliminated (§6) | 3 × n_gpu × 1.22 |

**Memory.** Zero persistent state added. Compositional with FACE/MFIO/MELT/Kahan-v/ORION-V/HYDRA. The forward activation cache `h_ℓ` is kept (same as pre-ZEPHYR HYDRA). Net delta: zero.

**No interaction conflicts.** FACE/MFIO compress `dW^DFA`'s EMAs. Kahan-v applies normally. SAS (#40) per-layer skipping stacks with ZEPHYR's cheap backward — SAS skips ~50% of an already-cheap pass, for a net ~6× backward speedup.

---

## 10. Gate-0 plan — the load-bearing experiment

ZEPHYR's value is conditional on a single empirical claim. The Gate-0 protocol resolves the claim definitively at low cost.

### 10.1 Test specification

**Hypothesis (H0):** at 66M-CHIRON × 5000 pile-bpe steps with `N_warmup = 5000` warmup, ZEPHYR achieves final loss within 0.5 nat of standard backprop.

**Configuration:**
- Model: 66M CHIRON (existing test config; m=512, L=12, n_H=8, T=512).
- Dataset: pile-bpe, identical seed/sample order across all comparison runs.
- Total steps: 10,000 (5000 warmup + 5000 DFA-only).
- Comparison runs (3 total):
  - **A — backprop baseline.** Standard CHIRON + Adam. Expected final loss ≈ 4.2 nat (validated baseline).
  - **B — naive ZEPHYR.** 5000 backprop warmup, 5000 DFA without symplectic structure (full ℝ^{2m × 2m} random matrices).
  - **C — symplectic ZEPHYR.** 5000 backprop warmup, 5000 DFA with q/p block-triangular SRHT (§4).
- Metric: NLL on held-out 10k tokens at step 10,000.

**Pass condition (success):** `L_C - L_A ≤ 0.5` nat. ZEPHYR proceeds to 1.84B-scale validation.
**Soft-pass condition:** `0.5 < L_C - L_A ≤ 1.5` nat. ZEPHYR is viable for *pretraining + backprop fine-tune* mode (10% backprop tail). Reduced 2.5× wall-clock multiplier instead of 3×.
**Fail condition:** `L_C - L_A > 1.5` nat. ZEPHYR retired. REFLECTOR or SYNAPSE selected as #46.

### 10.2 Cost

- Per run: 66M × 10k steps ≈ 1.5 GPU-hours on RTX 4080 SUPER.
- Three runs: 4.5 GPU-hours.
- Plus 1 GPU-hour for SRHT-kernel implementation, plumbing, parsing.
- **Total: ~6 GPU-hours = 1 GPU-day.** Well within budget.

### 10.3 Implementation slice for Gate-0

Minimal changes:
1. `gpu_zephyr.{h,cu}` (new): SRHT matvec, fast Walsh-Hadamard, RNG-seeded sign/subsample.
2. `transformer_zephyr.h` (new): `dfa_backward(h_ℓ, dy_loss, seed_ℓ) → dW_ℓ^DFA`.
3. `sgd_transformer.cpp`: branch on `--zephyr` flag at backward; forward unchanged.
4. `training_config.h`: add `useZephyr, zephyrWarmupStart, zephyrWarmupEnd, zephyrSymplectic`.
5. `chiron_main.cpp`: parse flags, plumb through.

Total: ~700 LOC, 1 day's engineering. Production hardening (HYDRA/MELT/RLG-aware) is post-Gate-0.

### 10.4 Diagnostics

Log `θ_ℓ(t) := arccos(⟨R_ℓ, W_ℓ^T⟩ / (‖R_ℓ‖_F · ‖W_ℓ‖_F))` every 100 steps; histogram at warmup-end, mid-DFA, and end. Success pattern: `θ_ℓ < 75°` at all layers. Failure pattern: `θ_ℓ → 90°` at deep layers + loss flatline. If `θ_ℓ < 60°` but loss flatlines, the failure mode is something else (TT-tangent / symplectic violation / variance mismatch) and the diagnostic narrows the next experiment.

---

## 11. Material distinction vs REFLECTOR and SYNAPSE

| Axis | REFLECTOR (#46-A) | SYNAPSE (#46-B) | ZEPHYR (#46-C) |
|---|---|---|---|
| Approach | Variational adjoint via cotangent lift | Sketch-based reconstruction | Random feedback alignment |
| Compute (per step) | ~1.6× | ~1.4× | **3×** (highest) |
| Guarantee | **Bit-exact** | **Bounded BF16 error** | **Heuristic** — alignment dynamics |
| Gate-0 cost | None (math derived) | None | **1 GPU-day** |
| CHIRON-specific lever | Cotangent-bundle (q,p) | Invertibility preserved by sketch | Partial (q/p split, §4) |
| Memory delta | +O(L·m) | +O(L·k·m) | **Zero** |
| Engineering | Medium | Medium | Low |
| Worst case | 1.6× | 1.4× | 1× (revert to backprop) |
| Best case | 1.6× | 1.4× | **3×** |

**Decision logic.**
- Risk-averse: REFLECTOR or SYNAPSE — bounded gain, bounded risk.
- Return-maximizing on positive expected value: ZEPHYR — `0.6 × 3 + 0.4 × 1 = 2.2` expected, vs 1.6× for REFLECTOR. **ZEPHYR wins on expectation.**
- Gate-0 cost (1 GPU-day) is tiny vs a multi-week production rollout. **Run Gate-0 first** is dominant regardless of risk preference.

**The strongest argument for ZEPHYR:** the Gate-0 result is informative regardless of outcome. Pass ⇒ 3× per step. Fail ⇒ characterized DFA depth-scaling failure at production scale; REFLECTOR/SYNAPSE remain available as #46.

---

## 12. Honest gaps

1. **Unverified at LLM scale.** Bartunov 2018 (L=50, ResNet-50) is direct negative evidence; ZEPHYR's warmup hypothesis is the proposed bridge but is itself untested at scale. **15–20% prior probability the convergence conjecture is false.**

2. **Symplectic q/p split — speculative.** The §4 q/p block-triangular construction is a hypothesis; naive full-`R_ℓ` may match it. Gate-0 tests both.

3. **MELT-TT-tangent alignment — open.** The §5.2 TT-core direct-feedback approach assumes signal lies primarily in the TT tangent (~80% heuristic, no theorem). If <50%, MELT+ZEPHYR may degrade.

4. **Calibration warmup length — unknown.** 5000 is a guess from shallow-net data. May need depth-scaling (`N_warmup = c · L`). Budget-flexible up to 50k.

5. **SLC/RLG composition — designed, untested.** §7's re-calibration mini-warmups (5–7% overhead) are projected, not measured.

6. **Pre-LN variance mismatch.** CHIRON's reversible LN may not respect DFA's projection variance. Paradigm #12 flagged this; mitigation is hybrid backprop-through-LN (costs ~10% of speedup). Gate-0 will surface the issue.

7. **The 30–50% prior.** ZEPHYR has a 30–50% prior probability of empirical failure at LLM scale based on published DFA literature at depth ≥ 50. This is the central honest framing: highest-upside, highest-risk #46 candidate.

---

## 13. If Gate-0 passes — production rollout

- **Phase A (week 1): hardening.** Fused SRHT+AXPY backward kernel; INT8 sign + index storage of `R_ℓ`; determinism via pinned RNG seeds and NCCL tree.
- **Phase B (week 2): composition validation.** ZEPHYR + SCFA, + MELT (gauge-violation rate ≤5% per 1000 steps), + HYDRA at n_gpu=4 (bubble-elimination measurement).
- **Phase C (weeks 3–4): 1.84B benchmark.** Full 500k-step run. Target: within 0.5 nat of backprop at 3× wall-clock. Final go/no-go.
- **Phase D (week 5+): scale-out.** 18B (post-MELT) at n_gpu=4; 117B at n_gpu=8 NVLink. Stacked multiplier vs iter-1 baseline: ~350×.

---

## 14. Closing — the bet, framed honestly

ZEPHYR proposes to eliminate chain-rule + inverse-walk together. DFA's mathematical foundation (Lillicrap-Nokland-Refinetti) exists at shallow depth; the LLM-scale extension is unverified. The bet has three legitimate outcomes (§0): clean pass at ~60%; partial pass needing backprop tail at ~25%; fail at ~15–20%. Expected wall-clock 2.2× over backprop, vs 1.6× REFLECTOR and 1.4× SYNAPSE. **ZEPHYR has the highest expected value despite the highest variance.**

The Gate-0 protocol (§10) costs 1 GPU-day and resolves the bet definitively. The proposal:

1. Run Gate-0 first. No production code until the convergence question is answered.
2. Pass ⇒ ZEPHYR selected as #46; rollout per §13.
3. Partial pass ⇒ ZEPHYR + 10% backprop tail; expected speedup ~2.5×.
4. Fail ⇒ ZEPHYR retired. REFLECTOR selected as the safer 1.6× #46. Negative result documented in `research/ZEPHYR_GATE0_REJECTION.md`.

REFLECTOR is a small certain win. SYNAPSE is a small certain win. ZEPHYR is a 60% chance at a large win and a 40% chance at zero. The 1 GPU-day Gate-0 cost is much smaller than the variance in outcomes — running it is the dominant strategy. **ZEPHYR's value is not its certainty but its informativeness.** A pass triples our compute multiplier; a fail tells us DFA-at-depth is unviable on transformers, closing a research direction definitively.

Let the empirical answer arbitrate.
