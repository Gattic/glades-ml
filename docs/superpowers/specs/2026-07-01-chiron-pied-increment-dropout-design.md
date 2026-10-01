# CHIRON PIED — Phase-Increment Ensemble Dropout

**Date:** 2026-07-01
**Status:** Design (approved for planning; pre-registration of the E0–E4 ladder in §13)
**Author:** research-framework-design (3-candidate parallel synthesis)
**Target:** a CHIRON-native dropout for the production flagship (CHIRON 1B WhiSC-D 30k),
ship metric = matched-30k val NLL (E4), on the WhiSC-D recipe.
**Predecessor lessons (binding):**
- **LayerDrop** (stochastic depth, the closest dropout-family port) — **FAILED** (+0.073 nat @5k,
  4 loss spikes; `research/LAYERDROP_5K_FAIL_2026_05_23.md`): hard-dropping a (shear, reln) pair
  breaks the (q,p) trajectory; 1/(1−π) compensation has no clean meaning through the nonlinear
  per-token reln; whole-layer mask bimodality misconditions Adam moments.
- **SORC** — **NO-GO**: forward norm-preservation did not bound the backward; the q-perturbation
  was O(‖p‖); `drot_phi` cascaded 2.5×/layer. Bound q-perturbations by ‖q‖, derive backward
  bounds, never assume them.
- **OBSD** — **NO-GO**: unbounded gate → ‖g‖ inflation → clip-throttled effective LR → −0.66 nat.
  Hard-bound every rate.
- **WhiSC-D** — **SHIPPED**: detached whitened frame, engaged from step 1. Scale statistics must
  be detached; there is no early p≈q window (ρ = σ^p/σ^q is trained-in to ~10³ within ~100 steps).

---

## 1. Executive summary

Classical dropout deletes units — a non-invertible, state-destroying operation that is forbidden
on CHIRON's reversible state and empirically fails when ported (LayerDrop). But CHIRON has a
structural feature no residual transformer has: **the momentum stream `p` is a linear accumulator
of attention increments that reaches the loss only through the linear terminal fold `q_L + p_L`.**
On that channel, dropout's ensemble semantics can be realized **exactly** instead of approximately:

> **PIED gates the attention increments, never the states:** `p += η ⊙ Y_l(q)` with a mean-one
> two-point mask `η ∈ {0, 1/(1−π)}`, i.i.d. per (layer, µstep, token, channel), regenerated from a
> counter-based stateless hash. Deleting a *contribution* is exactly invertible (recompute, regenerate,
> subtract); deleting a *state* is not.

Exact consequences (proven at θ_max = 0, i.e. WhiSC off; O(θ_max²·Var η) ≈ 10⁻⁴-relative
corrections at the flagship's θ_max = 0.07):

- **Unbiasedness:** E_η[logits] = noise-free logits. The deterministic inference network *is* the
  ensemble mean — no weight-scaling heuristic, nothing to compensate through reln (the exact trap
  that broke LayerDrop is structurally absent: the mask never meets a nonlinearity).
- **Exact ensemble:** softmax(E[z]) is the normalized geometric mean of the member predictives over
  2^L per-coordinate subnetworks — each member is itself a valid CHIRON flow (symplectic, det-1,
  reversible): the mask deletes increments, not structure.
- **Jensen sandwich:** NLL_inference ≤ E_η[NLL_train]. We always deploy a model at least as good
  as the one we trained.
- **Closed-form implicit regularizer:** E_η[L] = L + R_PIED + h.o.t., with
  R_PIED = ½·π/(1−π)·Σ_{l,t,i} D²_{l,i}·Y²_{l,t,i}·Var_{v∼s_t}[E_{v,i}] — a Fisher-weighted
  energy of the attention increments. Its distinctive content is **anti-cancellation across
  depth**: a mutually-cancelling increment pair (Y_a = +v, Y_b = −v) contributes nothing to the
  function but is taxed in full. Mutually-cancelling increments are the CHIRON-native form of
  co-adaptation; PIED taxes exactly this and nothing about the total. Complementary to SIRA
  (which penalizes the terminal magnitude, not the decomposition).

Safety is categorical, not asymptotic: the mechanism owns **no state, no parameters, no
statistics** — no σ^p, no ρ, no EMA appears anywhere in it, so the SORC/OBSD failure channels
have no inputs. The backward q-gain is the baseline's × η ≤ 1/(1−π) (≈1.11 at π = 0.1),
ρ-independent by inspection. Parameter-free ⇒ no checkpoint delta, no serving change.

Cost: one hash + one FMA fused into the DRAM-bound shear-commit (forward + inverse-walk) and one
masked copy replacing the existing dy hand-off in backward — ≈0.1–0.3% wall, no new [T×m] traffic.

**Honesty (pre-registered):** the flagship is under-trained (870M params, ~2B tokens, single
epoch) — "prevents overfitting" is not claimed. The claims are (1) Fisher-flatness /
anti-cancellation bias on how p_L is assembled, (2) forced redundancy across depth in the sole
inter-token channel, (3) a *secondary, non-gating* conjecture on the repetition attractor.
Modal-outcome priors: ~30–40% small win, ~35–45% neutral, ~25% tax-regression. The E3 gate
prices the Jensen tax directly before any 30k spend.

---

## 2. Candidate formulations (3-candidate parallel synthesis)

Three independent theory agents developed materially different formulations from the identical
problem statement.

### 2A — WhiSK / Whitened Stochastic Kicks (symplectic/Hamiltonian stochastic-map view)
Random elements of Sp(2,ℝ) near identity in the detached whitened frame, appended to the WhiSC
3-shear kernel: whitened kicks p̃ += ε₀·r·q̃ (r Rademacher, per (l,t,i)). Central theorem
(Cayley–Hamilton on sp(2)): **the parabolic/nilpotent (shear) cone is the unique noise direction
whose ensemble mean equals the deterministic map exactly** — exp(εN) = I + εN identically, so
E[map] = I for any zero-mean noise, all moments; elliptic noise carries an irreducible E[cos ε] < 1
contraction bias, hyperbolic an AM-GM expansion bias, and generic mixed sp(2) support has strictly
positive Lyapunov exponent (Furstenberg) — exponential depth-growth. Parameter-free; no parameter
gradient exists (the SORC-killer channel is empty); whitened-metric gain ≤ 1+ε₀/layer; kicks in one
nilpotent direction commute so the kick sub-chain gain is 1 + O(√L ε₀) exactly. Implicit
regularizer: Fisher-weighted sensitivity of logits to per-channel *relative* perturbations of
accumulated attention. Own honest prior: E3-pass ~0.9, but **E4-flat is the modal outcome
(~50–60%)** — noise-injection, not deletion; no ensemble-over-subnetworks semantic.

### 2B — PIED / Phase-Increment Ensemble Dropout (variational/information-theoretic view)
Mean-one Bernoulli gating of the attention increments at the shear commit (thesis above, §1).
The p-bus as an erasure channel: reliable prediction under rate-π erasure of (layer→terminal)
paths forces distributed codes across depth — a prior over redundant assemblies, priced directly
by E_η[L]. Granularity theorem: per-(l,t,i) masks make the mask-noise in any weight-gradient
entry ~√(Var(η)/T_eff) < 1% (T = 16384), so Adam moments see essentially baseline statistics —
the LayerDrop bimodality (its failure cause iii) is excluded by construction, and the mean penalty
is granularity-independent (no cross-token loss curvature), so fineness buys gradient-estimator
variance for free. Two fusion points, both zero-new-traffic. Symmetric two-point arm
(η ∈ {1−a, 1+a}, matched variance) separates the deletion/ensemble channel from the pure variance
channel as a mechanism experiment.

### 2C — SFD / Spectral Flow Dropout (operator-theoretic/spectral view)
Mean-one mask on the k = 1024 compressed DCT-II rows of the SCFA increment (`y_compr`), before
expansion — dropout on the (layer × token-frequency) grid of the accumulator; O(k) randomness per
layer, [k×m] traffic (16× cheaper than token-domain), 0.24% wall. Same exactness family as PIED
(linear-side placement, E[Ξ] = I, branch-bounded adjoint ≤ 1/(1−π)). Distinctive finding, honest:
the kept spectral band covers token-periods ≥ 32, while the diagnosed repetition loops have
periods 2–6 — **the naive "erase the repetition frequency" story is false**; any anti-repetition
effect is a trajectory effect. Own honest prediction: **ΔNLL ∈ [−0.02, +0.04] — NLL-neutral**;
its ship case rests on robustness/generation channels, not the ship metric. Also contributes the
G4 internal control (per-layer gating predicted to reproduce LayerDrop-style spikes — a falsifiable
test of the granularity theory) and the (l,j) placement as an upgrade path.

**Convergent findings (all three, independently):** mask increments never states; counter-based
stateless RNG (no state advance — the LayerDrop val-RNG bug class structurally impossible);
parameter-free with no checkpoint delta; mean-one two-point masks; bounded 1/(1−π)-class backward
factors; honest modal outcome is neutral-to-small. The convergence is evidence the family is right.

---

## 3. Framework selection rationale

**Selected: PIED (2B), with A's theory absorbed as justification and C's protocol absorbed as
controls/upgrades.**

- **Only PIED is literally dropout.** Deletion of contributions + exact inverted compensation +
  ensemble-of-subnetworks semantics, realized exactly where the architecture makes them exact.
  WhiSK is noise-injection (ensemble of transports — no deletion, no subnetworks); SFD is dropout
  on a coarser grid that omits the token dimension and self-predicts NLL-neutrality.
- **Strongest honest case on the ship metric.** The anti-cancellation penalty is the most
  plausible val-NLL lever: it taxes a specific, architecture-native co-adaptation (cancelling
  increment pairs on the p-bus) at zero function-space cost, complementary to SIRA. WhiSK's own
  analysis makes E4-flat modal; SFD's own analysis predicts neutrality (C-SFD-3).
- **Categorically safest.** PIED touches no state and reads no statistic — R1 is by inspection
  (the bound 1/(1−π) contains no ρ, σ, depth, or step). WhiSK writes the trunk state
  (p += κq, κ = ε₀ρ̂ raw-frame) and inherits the WhiSC telescoping/EMA-smoothness assumption;
  correct, but conditional — and it edits the production-critical, freshly perf-tuned
  `chiron_rot_*` kernels. SFD's insertion sits inside the SCFA checkpoint-cache/aliasing
  machinery (its own failure-mode #4: masked-cache inconsistency).
- **Best Adam conditioning.** Per-(l,t,i) granularity suppresses the LayerDrop failure channel
  hardest (√(Var/T_eff) < 1% vs SFD-G1's σ²/k_eff ~ 10⁻⁴ — both fine — vs whole-layer bimodality).
- **Unification (from A, adopted as theory).** PIED's perturbation δp = (η−1)⊙Y_l(q) is p-only,
  q-untouched, zero-mean — i.e. it lies in WhiSK's parabolic/shear cone, the unique bias-free
  species of phase-space noise. A's classification explains *why* increment gating is the one
  dropout that needs no compensation factor: the noise enters along the nilpotent direction where
  exp = identity + noise exactly. The two biased families (elliptic/rotation jitter, hyperbolic/
  squeeze) are exactly what a "dropout on q" or "dropout on θ" would have been — and are ruled out
  on first principles, not just engineering caution.

**Absorbed from the rejected candidates:**
- From WhiSK: the pre-registered **tax-based rate selection** (choose π by measured train-loss tax
  at E2/E3, not by convention); the **trajectory-shape rule** at E3 (the val-gap must shrink across
  checkpoints — the LayerDrop trajectory read as a leading indicator); claim-ledger discipline.
- From SFD: **position-stratified val** at E3/E4 (the LayerDrop bucket protocol); the
  **inference-time ablation-robustness probe** as a mechanism observable; the per-head pre-Wo arm
  (B2 = SFD-G3) and the spectral (l,j) placement as clearly-separated upgrades; the honest
  spectral-mismatch finding on repetition (loop periods 2–6 ≪ masked band) which demotes the
  repetition claim to trajectory-level for the whole family.

**Why the rejected are weaker for the first experiment.** WhiSK: modal outcome flat by its own
derivation; trunk-state write under a conditional bound; edits shipped kernels. SFD: predicts
itself NLL-neutral (fails the brief's primary objective a priori); (l,j) grid misses the
per-token ensemble dimension; SCFA-internal insertion carries its highest-ranked bug class. Both
remain documented upgrade paths (§14) and their theory is used above.

---

## 4. Formal problem statement

Per token t ∈ {1..T}, phase state (q_t, p_t) ∈ ℝ^m × ℝ^m (m = 2048, L = 24, T = 16384,
nH = 16, V = 32000). Grounded layer-l forward (flagship recipe, WhiSC-D on):

```
 (shear)  p ← p + Y_l(q),   Y_l(q) = Wo·SCFA(Wq q, Wk q, Wv q) = y_par + y_perp   [q unchanged]
 (WhiSC)  (q,p) ← Φ_l(q,p),  per-channel M_i = W_i⁻¹R(θ_i)W_i, |θ_i| ≤ θ_max = 0.07, detached W_i
 (ReLN)   q ← reln(q; γ,β)   per-token, reversible; backward re-derives stats from q_in (reanchor)
```

p_0 = 0, q_0 = embed; terminal fold q ← q + p at layer L−1 before logits z_t = (q+p)Eᵀ. Backward
is an exact inverse walk (reln⁻¹ → Φ⁻¹ → recompute Y_l(q), p −= Y_l(q)); nothing stored per layer.
The production shear-commit is `chiron_scfa_axpy2(p, ±1, y_par, y_perp, T·m)` (flagship runs
`--scfa-fuse-streams`, no `--bf16-residual-p`).

**Load-bearing structural facts.**
(F1) With θ_max = 0 the q-trajectory is independent of p: attention reads only q; WhiSC is the
sole p→q route and it is bounded by θ_max.
(F2) p is linear in the increments; each increment reaches the loss through the per-channel damping
D_{l,i} = Π_{l'>l} cos θ_{l',i} ∈ [cos(0.07)²³, 1] ≈ [0.945, 1] and the linear fold+readout.
(F3) ρ_i = σ^p_i/σ^q_i is trained-in to ~10³ within ~100 steps; no early p≈q window exists.

**Requirements.**
- **R1** backward q-subspace gain bounded independent of ρ (the SORC bar) — derived, §6.
- **R2** exact identity at rate 0 (bit-exact E0; default-off flag).
- **R3** exactly invertible in the inverse walk (bit-exact reconstruction; frozen per-step
  randomness).
- **R4** train/inference consistency stated exactly (inference noise-free; where E_η[fwd] differs
  from noise-free fwd, and by how much) — §6.
- **R5** safe from step 1 (σ^p → 0 at init) and at ρ ~ 4100.
- **R6** deterministic given seed; masks regenerable bit-exactly in backward (counter-based; no RNG
  state advance).
- **R7** perf ≲ 1–2% wall; elementwise only; no new GEMMs; no new [T×m] round trips.
- **R8** C++98 host; CUDA kernels in the gpu_chiron.cu style with CPU references; parameter-free
  (no checkpoint delta).
- **R9** pre-registered E0–E4 ladder with kill criteria (§13).

**Forbidden:** Bernoulli-zeroing of states; unbounded rates; raw-frame p→q mixing; stored per-layer
activations; warmup-dependent safety; learned rates in the prototype.

---

## 5. Core framework — increment-gated ensemble dropout

### 5.1 The mask field

Per accumulation micro-step, layer l, token t, channel i:

```
η_{l,t,i} = B_{l,t,i} / (1−π),   B ~ Bernoulli(1−π) i.i.d.   ⇒  E[η] = 1 exactly,  Var(η) = π/(1−π)
```

Ablation arm (`--inc-dropout-symmetric`): η ∈ {1−a, 1+a} equiprobable, a = √(π/(1−π)) — matched
variance, zero skew, no deletion. Separates the ensemble/deletion channel from the pure variance
channel (§13, mechanism-separation arm).

RNG: stateless counter hash (xorshift-mix family already shipped in `gpu_nesr.cu` /
`gpu_init.cu`), key = (seed_pied, step, µstep, layer) mixed host-side into one 32-bit key K, then
per element h = mix32(K ⊕ idx), idx = t·m + i; keep ⇔ h ≥ ⌊π·2³²⌋; η = keep·(1/(1−π)).
No RNG state ever advances; forward, inverse walk, and adjoint evaluate the identical pure
function; val/inference draws nothing. seed and step are already checkpoint-persisted ⇒ resume
reproduces the mask sequence.

### 5.2 Evolution law

**Forward (training, per layer):**

```
u = Y_l(q)                       [unchanged attention compute; u = y_par + y_perp at the commit]
p_{t,i} += η_{l,t,i} · u_{t,i}   [masked commit — one rounding for s = fl(η·u), one for fl(p+s)]
(q,p) ← Φ_l(q,p);  q ← reln(q)   [unchanged]
```

**Inference/validation (isTraining = false):** the mask branch is not dispatched; π treated as 0.

**Inverse walk:** reln⁻¹, Φ⁻¹ (unchanged); recompute u = Y_l(q) (bit-exact — q was never touched
by the shear, the same guarantee the baseline inverse already uses); regenerate η from the same
counters; p_{t,i} −= fl(η·u_{t,i}). Adding then subtracting the identical rounded value is
bit-exact recovery of p (IEEE-754), conditional only on the recompute bit-exactness the
architecture already certifies. **[proven]**

**Backward adjoint.** The masked layer's state Jacobian is unit-lower-triangular:

```
dp_in = dp_out                              [p-adjoint is EXACT identity — never masked]
dq   += (∂Y_l/∂q)ᵀ (η ⊙ dp_out)             [the SCFA backward is fed the masked upstream]
dW•   = attention weight grads with upstream (η ⊙ dp_out)
```

η is regenerated at the point of use (never stored). Implementation seam: the trainer already
hands dp to the SCFA backward through a dy buffer copy at the inverse-commit site — the masked
variant replaces that copy with dy = η ⊙ dp (zero extra traffic). The through-going dp is
untouched by construction.

**Per-realization validity (proven, and the architecture-native heart of the design):** for any
fixed draw, Ŷ_l(q) := η_l ⊙ Y_l(q) is still a function of q alone, so the noisy network is itself
an exact CHIRON network — symplectic shear, det-1, exact inverse, reanchor- and WhiSC-compatible.
Every ensemble member is a valid flow. LayerDrop deleted structure (a shear+reln pair — trajectory
surgery); PIED deletes only contributions.

### 5.3 Granularity and placement (decided, with reasons)

- **Granularity: per-(layer, µstep, token, channel), post-Wo, at the commit.** (i) The ensemble is
  over 2^L per-coordinate subsets — the finest dropout semantic; (ii) mask-noise in any weight-grad
  entry is ~√(Var(η)/T_eff) < 1% at T = 16384 ⇒ Adam m̂, v̂ see baseline statistics (the LayerDrop
  failure channel iii suppressed hardest — whole-layer masks make every entry bimodal
  {0, 1/(1−π)} per step); (iii) the mean penalty depends only on the marginal Var(η) (no
  cross-token loss curvature), so fineness buys gradient-estimator variance for free **[proven]**;
  (iv) the expansion in §7 is controlled by the per-component logit perturbation
  |Y_{l,t,i}|·‖E_{:,i}‖ — tiny at this granularity, O(‖Y_l‖) for coarse masks (which is LayerDrop
  territory: the quadratic/variational reading collapses there).
- **Placement: the whole increment (y_par + y_perp) at the single commit seam.** Simplest possible
  insertion (one masked axpy2); the theory (linearity into p) holds identically for both branches.
  The per-head pre-Wo arm (mask on sO before the Wo GEMM — the literal attention-path ensemble)
  and the spectral (l,j) placement (SFD-G1) are clearly-separated upgrades, not the prototype.
- **Depth schedule: constant π.** The accumulator is depth-transparent (D ≥ 0.945) and the penalty
  self-weights by increment energy Y²_l; a π_l schedule adds a knob with no first-order
  justification. **[derived]**

---

## 6. The R1 / R4 derivations

**R1 (backward q-gain bounded independent of ρ).** The only new factor anywhere in the backward is
the elementwise η on the increment branch:

```
‖dq_extra‖ = ‖(∂Y/∂q)ᵀ(η ⊙ dp)‖ ≤ (1/(1−π)) · ‖(∂Y/∂q)ᵀ dp‖ = (1/(1−π)) · baseline
```

Gain multiplier ≤ 1/(1−π) ≈ 1.11 at π = 0.1 — a constant. No factor of ρ, σ^p, depth, or step can
appear because the mask multiplies the increment and its adjoint, never p, never q, never a
statistic: **the mechanism has no scale inputs at all.** The forward perturbation to p is
δp = (η−1)⊙Y_l = O(‖Y_l‖), never O(‖p‖); the perturbation to q at the shear is zero (F1). The
1/(1−π) factor sits on a *branch* whose trunk block is exactly I, so depth composition has no
geometric channel (the SORC cascade was on the trunk). There is no parameter gradient (parameter-
free) — the exact object that exploded in SORC (`drot_phi`) has no analogue. dφ (WhiSC) and dγ,dβ
(reln) see only the slightly-perturbed states, bounded as above. **[proven]**

**R4 (train/inference consistency).** Inference runs noise-free. At θ_max = 0 the forward is affine
in η (F1, F2):

```
p_{L,t,i}(η) = Σ_l D_{l,i} · η_{l,t,i} · Y_{l,t,i}     (D ≡ 1 at θ_max = 0)
z_t(η) = (q_{L,t} + p_{L,t}(η)) Eᵀ                     — exactly linear in η
```

Hence exactly: (1) **E_η[z] = z_noise-free** (deterministic inference logits are the ensemble-mean
logits); (2) **softmax(E[z]) = normalized geometric mean** of member predictives; (3) **Jensen:**
NLL_inference ≤ E_η[NLL_train]. **[proven at θ_max = 0]**

With WhiSC on, the mid-stack leak: a δp crossing Φ_{l'} deposits δq with relative q-magnitude
≤ sinθ·(‖δp‖/σ^p) — the whitening makes the leak automatically q-scaled (ρ⁻¹σ^p = σ^q, a WhiSC
dividend). First-order-in-δη terms are exactly zero-mean (deterministic coefficients × zero-mean
noise), so (1)–(3) break only at second order: bias = O(Var(η)·θ_max²·(‖Y_l‖/σ^p)²) ≈ 10⁻⁴–10⁻³
relative, worst-case coherent. **Statement of record: exact at θ_max = 0; O(θ_max²·Var η)-approximate
at 0.07.** The leak is also the cross-depth composition channel — PIED additionally trains the
model's use of WhiSC coupling to be robust to missing increments. **[derived; assumes bounded
downstream Jacobians — the flagship's certified regime]**

**R5.** No scale statistic is referenced anywhere. At init (σ^p → 0) PIED gates near-zero
increments (harmless); at ρ ≈ 4100 it still only rescales increments by ≤ 1/(1−π). Safety is
unconditional in ρ, by inspection of §5.2. **[proven]**

---

## 7. Objective — the implicit regularizer, derived

Training on E_η[NLL] optimizes (Jensen) an upper bound on the mixture NLL — the fixed-posterior
ELBO reading of dropout — and §6(3) gives the deployment sandwich. Second-order expansion: since
z is affine in η (θ = 0 channel) and per-token losses depend only on their own logits
(∂²L/∂z_t∂z_{t'} = δ_{tt'}(diag s_t − s_t s_tᵀ), the softmax Fisher), the expansion in δη = η−1
has only loss-curvature terms:

```
E_η[L] = L + ½ Σ_{l,t,i} Var(η) · g_{l,t,i}ᵀ (diag s_t − s_t s_tᵀ) g_{l,t,i} + h.o.t.
       g_{l,t,i} = ∂z_t/∂η_{l,t,i} = D_{l,i} · Y_{l,t,i} · E_{:,i}
⇒  R_PIED = ½ · π/(1−π) · Σ_{l,t,i} D²_{l,i} · Y²_{l,t,i} · Var_{v∼s_t}[E_{v,i}]
```

**Reading:** the energy of every attention increment, weighted per (token, channel) by the variance
of that embedding coordinate under the model's own predictive distribution — a Fisher-metric
Tikhonov term on the *assembly* of p_L. **[derived: Gauss–Newton form; exact statement of the
quadratic term at θ = 0]**

Consequences:
- **Anti-cancellation (co-adaptation across depth, made precise).** At fixed realized function
  (fixed Σ_l D_l Y_l), the penalty Σ_l ‖Y_l‖²_Fisher is minimized by the minimal-energy
  decomposition and strictly increased by any cancelling pair — which contributes nothing to the
  function and pure variance to the output. PIED taxes exactly the cancellation, nothing about the
  total. **[derived]**
- **Relation to SIRA — complementary.** SIRA penalizes the terminal (q_L, p_L) energy/balance (the
  magnitude of the sum, last layer only, by backprop). R_PIED penalizes Σ_l ‖increment‖² at fixed
  sum (the decomposition), layer-by-layer, Fisher-weighted. They intersect only through
  ‖Σ Y_l‖² ≤ L·Σ‖Y_l‖². **[derived]**
- **Erasure-coding / IB reading.** The p-bus is the only inter-token channel and the only route to
  the loss. Rate-π erasure per (layer→terminal) path prices non-redundant codes: a prediction-
  critical bit carried by one increment is lost w.p. π; carried across k layers, w.p. π^k.
  E_η[L] pays the model to build repetition codes across depth. In the one-pass regime this is a
  prior over codes, not overfitting control.
- **Higher cumulants:** E[δη³] = π(2π−1)/(1−π)² = O(π); the truncation is honest only for small π
  and fine granularity (per-component |δz| tiny) — a stated assumption, and another reason the
  granularity is per-(l,t,i).

---

## 8. Temporal / depth dynamics

Mask effects compose additively down the accumulator with per-channel damping D ∈ [0.945, 1] — no
geometric amplification is possible in the linear channel **[proven]**; the nonlinear leak is
O(θ_max) per crossing with exactly zero-mean first order (§6). Terminal noise per (t,i):
Var[δz] = Var(η)·Σ_l D²_l Y²_l ‖E_i‖²-weighted — the same weights as the penalty, so depth
schedules are free at second order (adopted: constant π). Gradient side: E[η²] = 1/(1−π) inflates
the increment-branch second moment by ≤ 11% at π = 0.1, averaged by per-token masks into < 1%
relative on any weight-grad entry ⇒ ‖g‖ shift ≤ ~1%, far from the clip channel (OBSD lesson) —
watched at E3 anyway. WhiSC's detached EMA P̄ = E[p²] is measured on noisy p, inflated by
Var(η)·Σ_l E[Y_l²]/E[p²] — a few-percent shift in a detached conditioning input (not a correctness
input; frozen per step, so R3 unaffected); `[whisc]` monitor watches it. Reanchor is untouched:
its backward stats derive from the recomputed q_in, whose reconstruction under PIED is bit-exact.

---

## 9. Theoretical analysis (claim ledger)

- **Proven:** per-realization CHIRON-exactness of every ensemble member; bit-exact inverse
  reconstruction; R1 bound 1/(1−π), ρ-independent, per-realization, worst-case; exactness of
  unbiasedness/geometric-ensemble/Jensen-sandwich at θ_max = 0; the closed-form quadratic R_PIED;
  granularity-independence of the mean penalty (no cross-token curvature); constant-π sufficiency
  given D ≥ 0.945; absence of any parameter-gradient channel; PIED's noise lies in the parabolic
  (nilpotent) cone of sp(2) — the unique bias-free species (WhiSK's classification).
- **Derived under assumptions:** the O(θ_max²·Var η) leak bias (bounded downstream Jacobians);
  the √(Var/T_eff) Adam-noise estimate (weak cross-token correlation); the GN form of R_PIED
  (small π, fine granularity).
- **Heuristic:** Fisher-flatness/anti-cancellation → better late-training descent in the
  excess-capacity one-pass regime (the lineage's two largest wins came from removing conditioning
  pathologies, not adding capacity).
- **Conjecture (secondary endpoint only, never the ship metric):** redundancy weakens the
  repetition latch at formation. Two honest concessions: the Fisher weight vanishes where s_t
  collapses (the penalty does not fight confident end-states), and teacher-forced training never
  visits attractor states — the claim is upstream (caps single-path logit domination at formation).
  Additionally (from SFD): repetition loop periods (2–6 tokens) are far below any coarse spectral
  structure — only trajectory-level claims are admissible for this whole mechanism family.

---

## 10. Computational tradeoffs

| | PIED (prototype) | per-head arm (B2) | spectral arm (SFD-G1) |
|---|---|---|---|
| Mask domain | (l, µstep, t, i) post-Wo | (l, µstep, h, t) pre-Wo | (l, µstep, j) on y_compr |
| Randomness/layer | T·m bits (hashed, not stored) | T·nH | k = 1024 |
| New traffic | 0 (fused into commit + dy copy) | one [T×dH·nH] pass | 3 × [k×m] passes (0.24%) |
| Wall estimate | ~0.1–0.3% | < 1% | 0.24% |
| Ensemble semantic | per-coordinate layer subsets | attention paths (l,h) | (layer × frequency) cells |
| Checkpoint delta | none | none | none |

---

## 11. Comparison to existing CHIRON mechanisms

| | LayerDrop | SIRA | WhiSC-D | **PIED (this)** |
|---|---|---|---|---|
| Object | whole (shear, reln) pair | terminal (q_L,p_L) loss term | coupling map | **attention increments** |
| Action | hard skip (state surgery) | soft penalty | deterministic rotation | **mean-one Bernoulli gate** |
| Compensation | 1/(1−π), broken by reln | n/a | n/a | **exact by linearity** |
| Invertibility | broken | n/a | structural | **structural (regenerate+subtract)** |
| Backward bound | uncontrolled shifts | n/a | O(1) whitened | **≤ 1/(1−π), no scale inputs** |
| Adam conditioning | bimodal per layer | clean | clean | **< 1% (per-token masks)** |
| Params/ckpt | none | none | rot_phi + bit 1024 | **none** |
| Outcome | FAIL +0.073 | shipped | shipped | **predicted E3-pass; E4 = the experiment** |

PIED is to dropout what WhiSC was to cross-depth coupling: the same intent, rebuilt on the
architecture's own invariants (linearity of the p-bus; reversibility via increment gating;
scale-freedom instead of whitening — PIED needs no frame because it never touches a state).

---

## 12. Failure modes and mitigations (ranked)

1. **Pure Jensen tax, no offsetting benefit (most likely).** Val regresses ≈ Δ_tax at all
   horizons. *Caught:* E3 measures Δ_tax directly (two eval passes, noise-on vs noise-off at the
   same checkpoint); E4 decisive. Rate rule: π chosen so Δ_tax ∈ [0.01, 0.05] nat (predicted
   π = 0.1; fall back 0.05).
2. **Early tax never repaid (LayerDrop trajectory shape).** *Caught:* E3 trajectory rule — val-gap
   at 2500 ≤ +0.04 nat AND shrinking across the {1k, 1.5k, 2k, 2.5k} checkpoints, else no E4 at
   that π.
3. **Mechanism doesn't engage** (redundancy already saturated): NLL neutral AND the concentration
   probe unchanged. *Caught:* E4 secondary endpoint (Herfindahl concentration of per-layer
   contributions to the top-1 logit; inference-time increment-ablation robustness probe). Closes
   the arc as "lever not live" — distinct from failure 1.
4. **‖g‖/clip interaction** (residual OBSD channel). *Caught:* E3 ‖g‖ census; kill if sustained
   > 1.5× baseline.
5. **WhiSC-stat inflation** shifts a_i. *Caught:* E3 `[whisc]` monitor vs the flagship traces.
6. **Backward-branch masking bug** (masking the through-going dp instead of only the dy branch
   silently changes the adjoint). *Caught:* E1 finite-difference gradient checks with fixed masks
   at synthetic ρ = 45 — the test is designed for this exact bug class.
7. **Repetition conjecture false.** *Caught:* E4 secondary metrics; never gates the ship.

---

## 13. Minimal prototype + the pre-registered ladder

**Build = PIED, per-(l,µstep,t,i), post-Wo, mean-one Bernoulli, default-off.**
Library (glades-ml): masked commit kernel (`chiron_scfa_axpy2_masked` — η folded into the
existing axpy2; sign ±1 covers forward and inverse-walk), masked dy copy
(`chiron_incdrop_scale_copy` — replaces the existing dy hand-off memcpy at zero extra traffic),
shared uint32 hash (identical CPU/GPU), CPU references, unit tests (`test.sh chiron-incdrop`).
Trainer (glades-trainer): `--inc-dropout <π>` (default 0.0), `--inc-dropout-symmetric`
(default off), key = (seed, step, µstep, layer), gated on isTraining at every site. No new
params, no checkpoint bit, no serving change. Under `--bf16-residual-p` (default-on in the
flagship stack) the masked commit falls back to the unfused masked-axpy2 + SR-cast pair (the
fused dual_p kernel is not mask-aware) — one extra [T×m] pass per commit only while PIED is
active. Incompatible (hard-error): `--cuda-graphs` (replay would freeze the mask key),
`--sfa-swap-layer` (own commit kernels), no `--scfa`. **Build discipline:** the trainer links
the glades CUDA kernels statically — rebuild the trainer (`bash build.sh`) after any kernel
change; `make install` is not enough (and the trainer vendors the glades headers under
`include/` — sync `gpu_chiron.h` / `transformer_chiron_ops.h` after header changes).

**Validation ladder (pre-registered):**
- **E0** — flag off: step-1 bit-identical to the WhiSC-D flagship path (mask branch not
  dispatched). *Must pass.*
- **E1** — unit tests: (i) inverse-walk reconstruction at synthetic ρ = 45, π ∈ {0.1, 0.3}: q is
  0-ulp (untouched by definition); p reconstructs in the unmasked shear's tolerance class (bar
  1e-5 rel; measured 1.09e-07 — `fl(fl(p+s)−s)` is not exactly p in FP32, same as the baseline
  shear inverse); (ii) FD gradient checks with fixed masks at ρ = 45 (dq, dW•, dφ) — guards
  failure 6; (iii) E[η] = 1 to 4σ over 10⁶ draws; (iv) fwd/bwd/inverse mask-regeneration equality
  (bit-compare); (v) CPU ≡ GPU hash parity; (vi) η = 0-rate path dispatches the unmasked kernels.
- **E2** — small shape: E_η[logits] ≈ logits (exact at θ = 0; bias < 10⁻³ relative at θ = 0.07 —
  the §6 leak bound in vivo); measured Δ_tax > 0 and ≈ R_PIED at a probe checkpoint (the
  second-order formula validated numerically). *Kill if the bias violates the O(θ²) bound.*
- **E3** — matched single-seed 2500-step T = 16384, flagship WhiSC recipe + `--inc-dropout 0.1`
  (~3.4 GPU-hr). Predictions: no divergence (P < 1%), 0 grad-skips, ‖g‖ max ≤ 1.1× the WhiSC
  run's 3.54, val@2500 ∈ [2.78, 2.83] (baseline 2.78), Δ_tax ∈ [0.01, 0.05]. **Kill:** any
  grad-skip wave; ‖g‖ sustained > 1.5×; val@2500 > 2.88 (+0.10); Δ_tax > 0.10; trajectory rule
  (failure 2) violated. If only Δ_tax > 0.05: one retry at π = 0.05.
- **E4** — matched 30k vs whisc30k (seed 1337), decisive. 4-batch final-val + wide-32 confirm;
  position-stratified buckets (LayerDrop protocol). **Ship** if Δ ≤ −0.02 nat (val ≤ 1.355 vs
  1.3753) with 0 skips. **Neutral** (−0.02, +0.02): keep default-off; read the secondary
  endpoints (concentration ↓ ≥ 20%; ablation-robustness separation; gen-metrics) to decide
  whether the B2 per-head arm or the symmetric arm merits one more E3. **Kill** if Δ ≥ +0.02.
- **Mechanism-separation arm** (only if primary is promising): `--inc-dropout-symmetric` at
  matched Var(η). Same Δ ⇒ the win is the variance/curvature channel; Bernoulli better ⇒ the
  deletion/ensemble semantic is real. Either answer is knowledge.

**Pre-registered priors:** ~30–40% small win, ~35–45% neutral, ~25% tax-regression. The asymmetric
payoff (cheap E3, parameter-free, ~0.2% wall if shipped) justifies the probe; the neutral/kill
zones make a negative clean and cheap.

---

## 14. Full research program

1. **PIED E0–E4** (above) — adjudicates the family's ship question.
2. **B2 / per-head pre-Wo arm** — the literal attention-path ensemble (mask sO before the Wo
   GEMM), if PIED is neutral but the mechanism observables move.
3. **Spectral arm (SFD-G1)** — mask the k DCT rows of y_compr ((layer × frequency) ensemble,
   0.24% wall) if per-(t,i) proves too diffuse; carries its own pre-registered controls (G4
   internal control: per-layer gating predicted to reproduce LayerDrop-style spikes — a direct
   test of the granularity theory).
4. **WhiSK kicks** — the parameter-free parabolic phase-noise regularizer (ensemble of
   transports), if the *variance* channel (symmetric arm) turns out to be the live one; ~10 lines
   inside the WhiSC kernels, with its sp(2) uniqueness theory already established.
5. **Learned/scheduled rates** — only after a fixed-rate result exists; explicitly out of the
   prototype.
6. **Generation-aware training hook** — if E4 secondary gen-metrics move, PIED becomes a component
   of the deferred generation-training plan (`research/CHIRON_GENERATION_LOOP_FIX_2026_06_27.md`),
   not a perplexity ship.

---

## 15. Open conjectures and validation criteria

- **C1 (hard, falsifiable):** PIED does not destabilize the flagship recipe (E3: 0 skips,
  ‖g‖ ~ baseline). *Falsifier:* any divergence — which would refute far more than PIED, since the
  mechanism has no scale inputs; a failure here indicts the increment-recompute exactness
  assumptions and closes the family.
- **C2 (the experiment):** the anti-cancellation prior buys val NLL at 30k (E4 ship bar Δ ≤ −0.02).
  *Falsifier:* E4 kill/neutral zones with mechanism observables also flat.
- **C3 (mechanism):** per-layer contribution concentration (Herfindahl) drops ≥ 20% and
  inference-time increment-ablation robustness separates from baseline at matched NLL — the
  observable signature that the ensemble is engaging, independent of the NLL outcome.
- **C4 (separation):** Bernoulli vs symmetric arms differ ⇒ deletion semantics matter beyond
  variance; identical ⇒ the curvature penalty is the whole story (and WhiSK becomes the cheaper
  equivalent).

**Engineering discipline (project convention):** default-off, gated, no checkpoint delta, E0
bit-parity when off; kernels + CPU refs + unit tests land regardless of outcome; the E3 gate is
the cheap decision point before any 30k spend.
