# Paradigm shift #19 — Candidate B: Predictive Forward Emulation (PFE)

**Formulation class:** control-theoretic whole-network substitution with
an online-trained mirror oracle; bilevel optimization with a drift
Lagrangian.
**Author pass:** subagent-dispatched design (2026-04-22).
**Status:** candidate — awaiting selection at the shift-19 gate.

---

## 1. Short name and core thesis

**PFE — Predictive Forward Emulation.**  Working codename: *mirror
oracle with drift-priced substitution*.

Every shipped paradigm shift attacks an *interior* axis of training
compute — CHIRON recomputes activations, TC-tiled attention reshapes
attention, OVFG factors gradients, TRCD (#13) shortens per-token
depth, LCP (#16) shares per-token compute.  **All still run the full
main network f_θ forward on every step.**  The unattacked axis is the
whole-network pass itself.

PFE attacks it.  A small mirror network m_φ with |φ| ≈ 0.01·|θ| is
trained *online* to match f_θ's outputs on the current batch.  A
drift-based confidence c ∈ [0,1] — the EMA of KL(f_θ ‖ m_φ) — decides
per batch between (i) mirror step: run only m_φ, skip the main
forward/backward; (ii) calibration step: run f_θ and update both θ
(CE gradient) and φ (distillation KL).  Lagrange multiplier λ_d
enforces a drift budget.  An optional learned linear map R (the
"gradient synthesizer," a learned-DFA variant) permits θ updates on
mirror steps when R has been calibrated well enough.

Novelty: **online** (not offline) teacher–student inversion — student
*substitutes* teacher while teacher still trains — with a
*self-calibrating* trust signal and a Lagrangian step budget instead
of a heuristic threshold.  Distillation freezes the teacher;
speculative decoding is inference-only; Switch-Transformer gating
routes per token.  PFE routes per whole-network step while the
teacher evolves.

---

## 2. Primitive objects and state space

| symbol        | shape                     | meaning                                                         |
|---------------|---------------------------|-----------------------------------------------------------------|
| f_θ           | ℝ^{T×d} → ℝ^{T×V}         | main network (the one we ultimately train)                      |
| m_φ           | ℝ^{T×d} → ℝ^{T×V}         | mirror network, |φ| ≈ 0.01·|θ|                                  |
| c_k           | [0,1]                     | confidence statistic at step k                                  |
| τ_k           | [0,1]                     | substitution threshold at step k                                |
| ρ_target      | [0,1)                     | operator-set target mirror fraction (e.g., 0.7)                 |
| D_k           | ℝ_+                       | EMA of KL( f_θ(x_k) ‖ m_φ(x_k) ) over calibration steps        |
| λ_d           | ℝ_+                       | Lagrangian price on drift                                       |
| R             | ℝ^{d_m} → ℝ^{|θ|}         | optional learned gradient-synthesis map (DFA-style)            |
| s_k           | {CAL, MIR}                | step mode                                                       |
| k_cal         | integer                   | forced calibration period (every k_cal-th step is CAL)          |

**Mirror architecture.**  m_φ is a transformer of matched depth L but
reduced width d_m = d/8, so each layer is 1/64 the main FLOPs.  Depth
is matched so the residual-stream *trajectory* shape matches; width is
reduced because it preserves logit-space geometry more faithfully
than depth reduction.  With weight tying (embedding ↔ unembedding)
|φ| ≈ 0.012·|θ| — 1.2% overhead.

**State.**  (θ, φ, D_k, λ_d, τ_k, R); D_k, λ_d, τ_k are scalars
updated by the dual-ascent controller (§3.4).

---

## 3. Evolution law

### 3.1 Forward decision

Let x_k be the step-k batch.  Compute the mirror forward first:

    z_φ ← m_φ(x_k),  logit_φ = W_U · z_φ.

Extract the *mirror confidence* from its output entropy relative to a
running baseline:

    c_k = σ( β·( H̄ − H(logit_φ) )  +  γ·(1 − D̂_k) ),

where H(·) is per-token entropy, H̄ is its running mean, D̂_k = D_k /
D̄_k is the normalized drift, β and γ are fixed sigmoid gains, and σ is
the logistic.  The first term says "the mirror is confident about this
input"; the second says "the mirror has tracked f_θ well recently."
Both are necessary; either alone fails (confident-but-drifted or
tracking-but-uncertain).

Decide:
- If k mod k_cal = 0 → s_k = CAL (forced calibration).
- Else if c_k > τ_k → s_k = MIR (trust the mirror).
- Else → s_k = CAL.

### 3.2 Calibration step (s_k = CAL)

Run both forwards:

    z_θ = f_θ(x_k),  logit_θ = W_U · z_θ.

Main losses:

    L_CE(θ)  =  CE(logit_θ, y_k).
    L_KD(φ)  =  KL( softmax(logit_θ / T_d)  ‖  softmax(logit_φ / T_d) ),

with distillation temperature T_d = 2.0.  Then the joint objective:

    L_CAL  =  L_CE  +  λ_d · L_KD  +  μ · ‖φ‖²  (weight decay on mirror).

Backprop updates θ via standard rules (AdamW BF16), φ via the KD term
only (so φ never receives the CE gradient directly — it tracks θ's
*output* not the data).

Also update the drift EMA:

    D_{k+1} = (1 − η_D) D_k + η_D · KL( softmax(logit_θ) ‖ softmax(logit_φ) ),

with η_D ≈ 0.05.  This is the *inner* drift feedback.

### 3.3 Mirror step (s_k = MIR)

Only m_φ runs.  Three sub-policies:

**(a) Pure-skip (default).**  Use logit_φ as the step's prediction.
θ receives no update.  φ updates with a self-distillation regularizer
that stabilizes it in the absence of a teacher signal: the mirror is
encouraged to match its own EMA shadow (§6).  Cost: ~|φ|/|θ| ≈ 1.2%
of a full step, plus the embedding-level MLP which is the V·d term
(dominant unless V is small).  At V = 50 000, d = 1024, |θ| = 2.23 B:
|φ|/|θ| ≈ 0.012 → step cost ≈ 0.012 · full, giving **~83× saving per
mirror step** on interior compute and ~8× saving after embedding
amortization.

**(b) Synthesized-θ update (optional).**  Use the gradient synthesizer
R to produce a cheap estimate of ∇_θ L from m_φ's terminal activations:

    ĝ_θ  =  R( z_φ(y_k − softmax(logit_φ)) ).

R is a *block-diagonal* linear map (per-layer block) trained during
calibration steps to minimize ‖ĝ_θ − g_θ_true‖² on the current batch.
Analogous to DFA's random-feedback matrix but *learned*, which lets R
specialize to the network's gradient geometry.  Apply ĝ_θ with a
conservative learning rate η' = 0.1·η.

**(c) No update at all.**  θ is frozen this step; only φ's EMA shadow
evolves.  This is the strict default unless (b) is ablated in.

### 3.4 Threshold controller

The threshold τ_k is **not a hyperparameter** — it is the Lagrange
multiplier that enforces the mirror-fraction budget ρ_target.  Using
dual ascent on the constraint E[s_k = MIR] = ρ_target:

    τ_{k+1} = τ_k + η_τ · (ρ_target − 1{s_k = MIR}).

η_τ ≈ 1e-3.  The controller raises τ when the mirror is used *too
much* (to pull it back) and lowers τ when too little.  Equilibrium:
mirror is used exactly ρ_target fraction of the time, and the
statistic c_k's quantile at τ* aligns with it.

Simultaneously, λ_d evolves to enforce a drift bound D_target on the
calibration-step KL:

    λ_{d,k+1} = max(0, λ_{d,k} + η_λ · (D_k − D_target)).

If the mirror drifts too far, λ_d rises, the KD loss dominates
calibration updates, and φ is pulled back toward θ.

---

## 4. Derivation of the confidence statistic c(x)

Three candidates considered: (i) **input-based** c = σ(−H(x)) fails F2
on OOD easy-looking inputs; (ii) **output-uncertainty-based**
c = σ(−H(logit_φ)) fails F2 in reverse — an undertrained mirror is
uncertain everywhere and suppresses itself needlessly; (iii)
**drift-based** hybrid (selected).  Drift directly measures mirror
quality on the distribution we are training on — operationally the
only metric that matters.  The hybrid
c = σ(β·(H̄ − H(logit_φ)) + γ·(1 − D̂_k)) combines local (per-input
entropy) and global (recent tracking via D̂_k) signals; the local term
catches inputs the mirror *happens* to be wrong about; the global term
shuts the mirror down when aggregate drift is high regardless of
input.  Both failure modes of the simpler statistics are covered.

---

## 5. Bias analysis and optimal ρ

### 5.1 Step-count arithmetic

Run PFE for N wall-clock steps.  (1 − ρ)·N are calibration steps;
ρ·N are mirror steps.  θ receives (1 − ρ)·N gradient updates (pure
skip) or N-with-conservative-η updates (synthesized-θ).

**Pure skip.**  Per-step wall-clock: on CAL, full-forward cost F_full
plus mirror cost F_mir ≈ 0.012·F_full plus KD gradient ≈ 1.02·F_full.
On MIR, F_mir ≈ 0.012·F_full.  Total wall clock:

    T_PFE  =  N · ((1 − ρ)·1.02 + ρ·0.012) · F_full
           ≈  N · (1.02 − ρ·1.008) · F_full.

At ρ = 0.7: T_PFE ≈ 0.315 · N · F_full — **3.2× wall-clock reduction**.
At ρ = 0.8: 2.6× wall-clock * but* only 0.2·N θ updates.

### 5.2 Loss reduction per update

Assume a convex neighborhood where per-update loss drop is ΔL ≈ η · g².
Total loss drop over N wall-clock units compared to dense baseline:

    ΔL_dense(T) = N_dense · ΔL  where N_dense = T / F_full.
    ΔL_PFE(T)   = (1 − ρ)·N_PFE · ΔL     (pure skip)
                = (1 − ρ) · (T / ((1 − ρ)·1.02 + ρ·0.012)·F_full) · ΔL.

Let Q(ρ) = (1 − ρ) / ((1 − ρ)·1.02 + ρ·0.012) = (1 − ρ) /
(1.02 − 1.008·ρ).  Optimal ρ maximizes Q(ρ) under the constraint
D_k ≤ D_target (drift bound).

    dQ/dρ  =  [ −(1.02 − 1.008ρ) + 1.008·(1 − ρ) ] / (1.02 − 1.008ρ)²
          =  [ −1.02 + 1.008ρ + 1.008 − 1.008ρ ] / (...)²
          =  −0.012 / (1.02 − 1.008ρ)²  <  0.

Q decreases in ρ *monotonically* — which means at identical per-step
loss drop, skipping is never a net win in the pure-skip regime.  The
savings are purely in the wall-clock/memory axes; **loss-per-update
quality is strictly degraded** unless the synthesized-θ or some other
compensatory mechanism recovers gradient signal on mirror steps.  This
is the *critical* finding: PFE-pure-skip is a **wall-clock-vs-NLL
trade**, never a pure speedup.

### 5.3 Synthesized-θ update shifts the curve

Denote by κ ∈ [0,1] the *effective gradient quality* of an R-synthesized
update (κ = 1 means as good as true backprop, κ = 0 means a no-op).
Then ΔL_PFE(T) includes an extra ρ·N_PFE·κ·ΔL:

    Q_R(ρ, κ)  =  (1 − ρ + ρ·κ) / (1.02 − 1.008·ρ).

    dQ_R/dρ  =  [(κ − 1)(1.02 − 1.008ρ) + 1.008(1 − ρ + ρκ)] / (...)².

Set numerator = 0:
    (κ − 1)(1.02) + 1.008 + ρ·[−(κ − 1)·1.008·(−1) + 1.008·(κ − 1)] = 0.

After simplification (the ρ-terms cancel for linear Q_R):

    Q_R is constant in ρ iff κ = 1 − 0.012/1.008 ≈ 0.988.

For κ < 0.988, Q_R is decreasing in ρ → ρ* = 0 (never mirror).
For κ > 0.988, Q_R is increasing in ρ → ρ* = 1 (always mirror).

This is a **bang-bang optimum**: PFE is either a full win or a full
loss depending on whether R can recover 98.8% of the true gradient's
signal.  DFA experiments (#12) report 60–90% recovery on
similar-scale networks.  The regime κ ∈ [0.8, 0.95] is realistic; in
this regime pure-skip (ρ = 0) beats synthesized-θ per loss-per-update.

**Conclusion.**  The *wall-clock-minimizing* schedule is not the
*NLL-minimizing* schedule.  PFE should be deployed as a
*speedup-at-some-NLL-cost* lever, with ρ chosen to match the
operator's compute budget, **not as an always-on training replacement**.

### 5.4 Operational ρ policy

Given the above analysis:

- During warm-up (first ~5000 steps): ρ = 0.0 (pure dense).  Mirror is
  too cold to be trusted.
- After warm-up: ρ ramped from 0.0 to ρ_target over 10000 steps.  The
  λ_d drift controller keeps D_k ≤ D_target throughout.
- ρ_target is a budget knob, not a quality knob.  Typical values: 0.3
  (conservative, ≈1.4× speed), 0.5 (moderate, ≈1.9× speed), 0.7
  (aggressive, ≈3.2× speed with measurable NLL regression).

---

## 6. Mirror stabilization on MIR steps

Pure-skip MIR steps do not update φ from the true f_θ signal; without
regularization φ drifts.  Two cheap stabilizers (both entirely on the
mirror's 1/8-width network):

**(S1) EMA shadow.**  φ̄_k = (1 − η_EMA)·φ̄_{k−1} + η_EMA·φ_k.  On MIR
steps a weak pull L_shadow = μ_sh·‖φ − φ̄‖² prevents high-frequency
excursions that the infrequent calibration signal cannot correct.

**(S2) Adversarial self-distillation.**  On a MIR step compute both
m_φ(x) and m_{φ̄}(x) and penalize KL.  Cost 2·|φ|/|θ| ≈ 2.4% per MIR
step.  Guarantees mirror updates slow enough to be caught on the next
CAL step.

---

## 7. Composition with shifts #1–#13, #16

All #1–#11 orthogonal to PFE: they compress θ-interior axes, PFE
substitutes across networks.  Mirror uses the same compressions at
1/64-ish scale.  **#12 DFA**: strong composition — R is a *learned*
DFA matrix; MIR policy (b) is a DFA update seeded by mirror
activations.  PFE gives DFA a *conditional* deployment.  **#13 TRCD
& #16 LCP**: mirror can run TRCD/LCP internally; the interesting
composition is on CAL steps where TRCD × LCP multiplicatively shrinks
full-forward cost.

Key stack: **TRCD × LCP × PFE**.  CAL costs F_full / 12;
MIR costs 0.012·F_full.  At ρ = 0.6: 0.4·(1/12) + 0.6·0.012 =
0.040·F_full — **25× wall-clock reduction** per step, at the cost of
(1 − ρ)·12 = 4.8× reduction in θ updates per wall-clock.  Loss-per-update
is preserved on CAL steps; loss-per-second is the operative metric.
PFE × LCP × TRCD is the strongest stack the shipped set admits.

---

## 8. Relation to prior work

- **Knowledge distillation** — teacher fixed, student is the artifact.
  PFE inverts: mirror is a throwaway tool, never deployed; teacher
  still trains.
- **Speculative decoding** (Leviathan, Chen 2023) — small-model propose
  + large-model verify, inference-only.  PFE brings the pattern to
  training with a calibration-step verifier.
- **Diffusion reverse-distillation / Consistency Models** — compresses
  inference steps; again inference-only.  PFE runs during training.
- **Switch Transformer / MoE** — token-level routing to *weight
  subsets*; PFE does batch-level routing across two *entire networks*.
- **DFA (Nøkland 2016)** — random-matrix feedback.  PFE's R is a
  *learned* DFA matrix deployed *conditionally* on mirror trust.
- **Model-predictive control** — cheap-model-guided actions on a true
  system; PFE imports the pattern to pre-training.

**Novelty.**  No prior method combines (i) online mirror matched to an
evolving main network, (ii) drift-priced substitution, (iii) bang-bang
ρ analysis revealing the wall-clock/NLL trade, (iv) composition with
per-token depth/pooling (#13/#16), (v) learned-DFA gradient synthesis
used conditionally.

---

## 9. Failure modes and mitigations

**(F1) Mirror drift.**  φ diverges from θ; c_k is fooled.
*Mitigations:* λ_d dual-ascent on D_k; forced CAL every k_cal = 16
steps; EMA-shadow (S1); D_target ≈ 0.05 nats (same scale as ~100
steps' CE improvement).

**(F2) Confidence miscalibration.**  c_k high where f_θ would have
updated strongly.
*Mitigations:* hybrid statistic (§4) uses both local + global
signals; every forced-CAL batch doubles as a miscalibration probe by
measuring would-have-been loss on recent MIR inputs; raise τ_k if MIR
loss significantly exceeds CAL loss.

**(F3) Distillation doesn't reproduce θ gradient.**  KD aligns outputs,
not gradients; mirror internals aren't forced to match f_θ's.
*Mitigations:* (a) hidden-state KD ‖z_θ − P·z_φ‖² on CAL steps with
fixed random P ∈ ℝ^{d×d_m}; (b) R synthesizer with measurable quality
κ; (c) disable synthesized-θ when κ < 0.8.

**(F4) Mirror generalizes poorly off-distribution.**
*Mitigation:* **mirror is never used at inference** — deployment is an
absolute rule.  Evaluation always runs f_θ end-to-end.  Mirror is
purely an internal training throttle.

**(F5) Two-network synchronization overhead.**  CAL costs 1.02·F_full;
MIR costs 0.012·F_full; switching adds kernel-launch latency.
*Mitigation:* persistent CUDA streams per network; async mirror
forward.  At ~10 μs/launch vs ~50 ms/step on 4080 SUPER: <0.02%.

**(F6) Memory for R.**  Naive R ∈ ℝ^{|θ|×d_m} ≈ 285 GB.
*Mitigation:* block-diagonal-by-layer and rank-r factored
(R_l = U_l V_l^⊤, r = 16) gives ~0.05·|θ| ≈ 111 MB.  If even this is
tight, revert to fixed-random DFA (no R learning).

**(F7) Mirror width too small.**  d_m = d/8 below JL threshold.
*Mitigations:* test d_m ∈ {d/16, d/8, d/4}; widen if KD plateau > 0.1
nats; alternatively reduce depth L_m.

---

## 10. Implementation sketch

**Phase 1 — GPU primitives.**  `pfe_mirror_forward` (BF16 forward at
width d_m, ~400 LOC reusing existing transformer forward with a width
parameter); `pfe_kl_logits` (fused softmax-KL, ~80 LOC); `pfe_drift_ema_update`
(~40 LOC); `pfe_confidence_statistic` (~60 LOC);
`pfe_threshold_controller` (CPU dual-ascent, ~100 LOC);
`pfe_shadow_update` (~30 LOC).

**Phase 2 — Trainer integration.**  `MirrorNetwork` wrapping an NNetwork
with reduced width; `PFEScheduler` in MLState/ holding (c, τ, λ_d, D,
ρ_target); CAL/MIR branching in training loop; forced CAL every k_cal
overrides c_k.

**Phase 3 — Parity tests.**  `CHIRONPFEMirrorMatchesMainTest` (KL < 0.1
nats after 1k CAL steps); `CHIRONPFEControllerConvergenceTest` (τ, λ_d
converge to ρ_target ±0.05); `CHIRONPFEDriftBoundTest` (D_k ≤ 2·D_target
for 5k steps); `CHIRONPFECalibrationFallbackTest`;
`CHIRONPFEMirrorNeverAtInferenceTest`.

**Phase 4 — Trainer flags.**  `--pfe-enable`, `--pfe-mirror-ratio`
(default 0.012), `--pfe-rho-target` (0.5), `--pfe-drift-target` (0.05
nats), `--pfe-kcal` (16), `--pfe-synth-gradient` (off).

**Phase 5 — Scale.**  pile_small: mirror-tracks validation.  pile_large
L=24, T=2048, ρ=0.5: target ≥1.9× throughput at NLL regression ≤ 0.15
nats.  2.23 B run, L=48, ρ=0.3: ≥1.4× at NLL parity within 0.10 nats.
Compound (PFE + LCP + TRCD + CHIRON + #7): ≥300 k tok/s at NLL
regression ≤ 0.25 nats — best-case stack.

**Estimated effort.**  ~3 engineering weeks (one longer than LCP —
mirror weight management and controller stability add test surface).
Core risk: F3 gradient reproduction — long-run divergence at ρ > 0.5
must be empirically bounded.

---

## 11. Open conjectures

1. **Mirror-tracking.**  d_m = d/8 KD-trained mirror tracks f_θ to
   KL ≤ 0.05 nats throughout pre-training.  Test: log KL at every CAL
   step across 50k steps.
2. **Bang-bang ρ.**  For κ ≤ 0.95 the loss-optimal ρ is 0 or 1; mixed
   ρ is Pareto-dominated on loss-per-update.  Test: sweep
   ρ ∈ {0.0, 0.3, 0.5, 0.7, 1.0} and measure NLL-per-wall-clock.
3. **Controller stability.**  λ_d dual-ascent converges to
   D* ≈ D_target with oscillation amplitude < 0.02·D_target after 2k
   steps.
4. **Composition.**  PFE × LCP × TRCD ≥ 20× wall-clock at NLL
   regression ≤ 0.3 nats.
5. **Gradient quality.**  Rank-16 per-layer R achieves κ ≥ 0.85 after
   10k CAL steps.  Test: cos(ĝ_θ, g_θ_true) on CAL batches.
6. **Mirror-width threshold.**  Below d/8 KD plateau > 0.1 nats; above
   d/4 mirror cost dominates.  Test: sweep d_m ∈ {d/16, d/8, d/4, d/2}.

---

*End of candidate B.*
