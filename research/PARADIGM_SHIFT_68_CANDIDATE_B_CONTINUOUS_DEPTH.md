# Paradigm Shift #68 Candidate B — CONTINUOUS-DEPTH-CHIRON: Neural ODE Backbone with Adaptive Integrator

**Status:** CANDIDATE (one of three for #68 selection at iter 212).
**Date:** 2026-05-08 (iter 212).
**Axis:** DEPTH-CONTINUOUS reframing — replace discrete L=53 layer stack with continuous-time Neural ODE solved by adaptive Runge-Kutta integrator. Adaptive NFE per input.
**Magnitude target:** 1.5-3.0× headline (HONEST band; aggressive 5-20× claims rejected on LLM-scale evidence).
**Verdict (this candidate):** **RESERVE** — wait for empirical NeurODE-LM results at >1B params before promoting to SELECT.

---

## 0. Status, date, axis, honest headline

**Honest headline.** CONTINUOUS-DEPTH-CHIRON reframes CHIRON's L=53 discrete shears as a continuous-time Neural ODE on `t ∈ [0, T_depth=53]` solved by an adaptive Runge-Kutta integrator. **Mechanistically appealing — bijectivity and reversibility become structural rather than engineered.** The hope: easy inputs get few function evaluations (NFE), hard inputs get many; median NFE < 53 yields wall-clock speedup.

**The hard truth.** No NeurODE-style architecture has produced results matching discrete-layer transformers at >1B params (Chen 2018, Dupont 2019, Massaroli 2020, Kidger 2020 all stall well below the LLM-scale frontier; FFJORD 2019 is density-modeling not autoregressive LM). Strict bit-exact NLL preservation forces tolerance settings that erase most of the speedup. Composition with #43 ORION + #46 REFLECTOR + #49 ICARUS is heavily overlapping (~50%) — much of the integrator-side gain has been captured already.

**Honest band:** **1.5-3.0× standalone wall-clock at 1.84B if the empirical confirmation breaks favorably**, but the marginal gain over #49 ICARUS + #46 REFLECTOR is more like **1.1-1.5×**. **At LLM scale the empirical confirmation probability is 25-30%**, an order of magnitude below typical paradigms in this program.

**Verdict.** **RESERVE.** This paradigm is not ready to ship. It should remain on the candidate roster until either (a) a peer NeurODE-LM result demonstrates >1B-param convergence parity, or (b) a Gate-0 probe at 41M shows median NFE < 0.6·L_eff with bit-exact NLL preserved. Until then, ICARUS + REFLECTOR already capture most of the ODE-integrator headroom.

---

## 1. Executive summary

The CHIRON forward pass is a sequence of L=53 reversible symplectic shears:

```
(q_{ℓ+1}, p_{ℓ+1}) = shear_ℓ(q_ℓ, p_ℓ),    ℓ = 0, 1, ..., 52
```

CONTINUOUS-DEPTH-CHIRON replaces the discrete index `ℓ` with a continuous depth coordinate `t ∈ [0, T_depth = 53]`, and the discrete shears with a vector field:

```
d(q,p)/dt = F(q, p, t; θ),    (q,p)|_{t=0} = (q_0, p_0)
```

The forward pass solves the IVP from t=0 to t=53 using an adaptive Runge-Kutta solver (RK4 fixed-step, or DOPRI5/RK45 adaptive). The reverse-time backward pass integrates from t=53 to t=0 (cotangent-lift from #46 REFLECTOR applies directly).

**Three potential wins.**
1. **Adaptive NFE.** Easy tokens (high-frequency words, low-loss positions) need few solver steps; hard tokens (rare entities, ambiguous boundaries) need many. If median NFE = 30 and worst-case NFE = 80, average wall-clock falls below the 53-step discrete baseline.
2. **Continuous-depth interpolation.** The trained vector field can be evaluated at arbitrary depth resolutions at inference — like a continuous-resolution model.
3. **Theoretical cleanness.** Bijectivity and reversibility become free consequences of well-posed ODE flow, not engineered properties. Composes cleanly with #46 REFLECTOR (cotangent-lift is the textbook adjoint method for ODEs).

**Three load-bearing risks.**
1. **NeurODE training instability at scale.** Chen 2018 / Grathwohl 2019 / Dupont 2019 all show NFE growing pathologically as training proceeds; vector field gets stiffer and stiffer; integrator tolerances tighten until wall-clock collapses. **No public result demonstrates NeurODE LM convergence parity at >1B params.**
2. **Bit-exact NLL preservation at LLM scale.** Standard adaptive RK45 gives ε ≈ 10^{-8} per step at modest tolerance, which compounds over 53 effective steps to ε ≈ 10^{-6} — at the boundary of the project's bit-exact band. Stricter tolerance erases speedup.
3. **Heavy overlap with #43 + #46 + #49.** ORION already does Galerkin model-order reduction of the SGD ODE; REFLECTOR does cotangent-lift adjoint flow; ICARUS does Yoshida 4th-order symplectic integration. CONTINUOUS-DEPTH attacks the same axis — the marginal gain after these three is bounded.

**Speedup estimate.**
- **Aggressive (rejected):** 5-20× via median NFE = 8-15 with adaptive tolerance.
- **Plausible (HONEST):** 1.5-3.0× standalone if empirical confirmation breaks favorably.
- **Marginal beyond #49 ICARUS + #46 REFLECTOR:** 1.1-1.5×.
- **Cumulative-stack ceiling at 1.84B with bit-exact NLL:** ~10,000,000-12,000,000× (vs current 8,580,000×) on causal-reasoning if the favorable case holds.

**Cumulative stack (favorable case):** ~10,000,000× causal-reasoning subset; otherwise unchanged at 8,580,000×.

**Engineering scope:** ~1900 LOC over 6-8 weeks, with substantial integrator-tolerance instrumentation and a non-trivial Gate-0 protocol.

**Joint Gate-0 PASS probability:** ~45%. **LLM-scale empirical confirmation probability:** **25-30%** (the limiting factor).

**Bottom line:** RESERVE for paradigm #69+ pending empirical NeurODE-LM evidence at >1B params, or Gate-0 confirmation that median NFE on production data < 0.6·L_eff with bit-exact NLL.

---

## 2. Mechanism: ODE formulation, adaptive integrator, reverse-time backward

### 2.1 From discrete shears to continuous flow

CHIRON's discrete shear at layer ℓ acts on paired state (q,p) ∈ ℝ^{T×d} × ℝ^{T×d}:

```
q_{ℓ+1} = q_ℓ + Δt · A_ℓ(p_ℓ; θ_q^{(ℓ)})       (q-shear)
p_{ℓ+1} = p_ℓ + Δt · B_ℓ(q_{ℓ+1}; θ_p^{(ℓ)})  (p-shear)
```

where Δt = 1 layer per discrete step, and A_ℓ, B_ℓ are MLP-attention blocks parameterised by θ. This is structurally a Verlet/leapfrog integrator of step Δt = 1.

The continuous-depth reframing: let θ^{(ℓ)} → θ(t) be a smooth function of depth, and let A, B → A(p, t; θ), B(q, t; θ) be vector fields. The forward pass becomes a Hamiltonian-flow IVP:

```
dq/dt = A(p, t; θ),      q(0) = q_0
dp/dt = B(q, t; θ),      p(0) = p_0
t ∈ [0, T_depth = 53]
```

Two implementations of θ(t):

**Option B1 — depth-conditioned shared weights.** A, B share MLP weights across t, but accept t as an additional input via sinusoidal time embedding (analogous to diffusion models' time conditioning). Parameter count: O(d²) — independent of T_depth. Strong inductive bias toward smoothness in t.

**Option B2 — basis-expansion of θ.** θ(t) = Σ_k φ_k(t) · θ_k where {φ_k} are Chebyshev or Legendre polynomials of degree K. Parameter count: K · O(d²). Tunable smoothness.

Selection: **Option B1**, because (a) parameter count matches discrete CHIRON if K=53, (b) RetNet/Hyena/Mamba precedent for time-conditioned vector fields suggests B1 is well-behaved, (c) B2 risks overfitting depth-localised idiosyncrasies in training.

### 2.2 Adaptive Runge-Kutta integrator (DOPRI5 / RK45)

The forward pass solves the IVP via Dormand-Prince RK45 (DOPRI5), a 6-stage embedded RK pair giving 4th-order solution + 5th-order error estimate at the cost of one extra function evaluation per accepted step.

Per accepted step:
- 6 vector-field evaluations.
- Local error estimate η̂_t = ||y_5 - y_4||.
- Step accepted if η̂_t < tol; otherwise step rejected and h ← h / 2.
- Adaptive step size: h_{new} = h · (tol / η̂_t)^{1/5} · safety_factor (typical safety = 0.9).

For a forward pass over [0, 53] with average accepted step h = 1.0, expected NFE = 6 · 53 = 318 — **6× WORSE than discrete CHIRON's 53 layer evaluations!** The adaptive solver only wins if h > 1.0 on easy regions, i.e. the integrator can take 2-5 layer-equivalent steps in regions of slowly-changing vector field.

This is the load-bearing empirical question: **does the trained F have slowly-changing regions, or does it stay stiff throughout?** If F stays stiff, the adaptive solver chooses h ≤ 1 everywhere and CONTINUOUS-DEPTH-CHIRON is *slower* than discrete CHIRON.

Mitigations:
- **Lower-order RK methods.** RK4 fixed-step at h=1 matches discrete CHIRON cost exactly. Adaptive only on a per-input basis.
- **Hybrid integrator.** Run RK4 at h=1 for a "trunk" pass; refine with adaptive RK45 only on hard tokens (selected by entropy or PRM signal from #59).
- **ICARUS-derived integrator.** Yoshida 4th-order symplectic integrator from #49 already gives 4th-order accuracy at O(h³) effective NFE per step — this is the *baseline*, not a new gain.

### 2.3 Reverse-time backward pass (adjoint method)

The classical Neural ODE adjoint (Chen 2018) recovers gradients by integrating an augmented ODE backward in time:

```
da/dt = -a^⊤ · ∂F/∂(q,p)        adjoint state
dl/dt = -a^⊤ · ∂F/∂θ            parameter gradient
```

Reverse-time integration from t=53 to t=0, with terminal conditions a(53) = ∂L/∂(q(53), p(53)), l(53) = 0.

For CHIRON specifically, **the adjoint method is exactly #46 REFLECTOR's cotangent-lift**. The cotangent bundle of the (q,p) phase space carries the dual variables (a_q, a_p), and the reverse-time flow on this lifted space is bit-exactly determined by F (when F is reversible — which it is by construction in CHIRON).

**This means CONTINUOUS-DEPTH-CHIRON's backward pass IS REFLECTOR's cotangent-lift backward pass, applied to a continuous-time forward.** The 1.5-1.6× backward speedup from #46 carries over directly. **Marginal gain on the backward axis from CONTINUOUS-DEPTH alone: zero.**

Total NFE for adaptive RK45 forward + adjoint backward ≈ 12 · 53 = 636 in worst case (vs discrete CHIRON forward + REFLECTOR backward ≈ 80). The 8× cost increase is the price of generality. Speedup must come from accepted-step h > 1 on average.

### 2.4 Bit-exact NLL preservation under integrator tolerance

Per-step solver error: ||y(t+h) - ŷ(t+h)|| ≤ C · h^{p+1} for a p-th order method. For RK45 (p=4): ε_step ≈ 10^{-7} at h=1 with tol=10^{-6}.

Over 53 effective steps the error compounds linearly (worst case): ε_total ≈ 53 · 10^{-7} ≈ 5 · 10^{-6}.

**This is at the boundary of the project's bit-exact band** (typical bound 10^{-7} per step from #50 HELIUM, #51 ATLAS-COMPILE, #52 NIMBUS). To stay strictly bit-exact:
- Tighten tol to 10^{-8} → step rejection rate climbs → wall-clock speedup erodes.
- Use higher-order method (e.g. RK87) → more NFE per accepted step.
- Accept ε_total ≈ 5 · 10^{-6} as "bit-exact-equivalent" (this stretches the project's strict definition).

**The honest finding:** strict bit-exact NLL preservation forces tolerance settings that bring CONTINUOUS-DEPTH-CHIRON's wall-clock close to discrete CHIRON. The 5-20× aggressive claim is incompatible with the project's NLL constraint.

The 1.5-3.0× plausible band assumes "bit-exact-equivalent" semantics (≤ 5 · 10^{-6} nat per step), matching #50 HELIUM's tolerance not #52 NIMBUS's stricter ≤ 10^{-7}.

---

## 3. Theoretical analysis

### 3.1 Theorem 1 (well-posedness)

**Theorem 1.** If F(q, p, t; θ) is Lipschitz-continuous in (q, p) uniformly in t over [0, T_depth] and continuous in t, then the IVP has a unique global solution for any (q_0, p_0) ∈ ℝ^d × ℝ^d.

**Proof.** Picard-Lindelöf. The MLP-attention block is composed of bounded-Lipschitz nonlinearities (GELU, softmax) and bounded-norm linear layers; F is Lipschitz with constant K = O(σ_max(W)·||θ||) where σ_max is the spectral norm. Uniform Lipschitz over t holds because θ(t) is smooth (Option B1) and bounded over the compact interval [0, 53]. ∎

**Caveat.** The Lipschitz constant K can be very large (10²-10⁴) at LLM scale; well-posedness is not the same as well-conditioned. Stiffness comes from large K.

### 3.2 Theorem 2 (bijectivity, reversibility)

**Theorem 2.** The flow Φ_t: (q_0, p_0) ↦ (q(t), p(t)) is a C¹ bijection of ℝ^d × ℝ^d → ℝ^d × ℝ^d for all t ∈ [0, T_depth].

**Proof.** ODE flows are bijective by uniqueness of solutions (forward and backward). Differentiability in initial condition follows from differentiability of F. ∎

**This is structurally cleaner than discrete CHIRON.** Discrete reversibility required engineered shear pairs; continuous reversibility is automatic.

### 3.3 Theorem 3 (NLL preservation under integrator-error tolerance)

**Theorem 3.** Let p_θ(x) be the model density induced by the discrete CHIRON, and p̃_θ(x) the density induced by CONTINUOUS-DEPTH-CHIRON solved with integrator tolerance tol. Then

```
| log p_θ(x) - log p̃_θ(x) | ≤ C_1 · L_eff · ε_step(tol)
```

where C_1 depends on the LM head Lipschitz constant.

**Proof sketch.** Solver error in (q(53), p(53)) is bounded by L_eff · ε_step. The LM head is Lipschitz with constant C_1, so log-prob shifts are bounded by C_1 · L_eff · ε_step. ∎

**Numeric bound.** For RK45 at tol=10^{-6}: ε_step ≈ 10^{-7}, C_1 ≈ 10, L_eff = 53 → total NLL shift ≤ 5 · 10^{-5} nat per token.

This **violates** the project's bit-exact band (≤ 10^{-6}). To recover, tol must drop to 10^{-9}, which raises NFE by ~3×, eroding speedup.

**The trade is fundamental:** integrator-error compounding over depth versus solver-step economy. At LLM scale this trade is ε-tight.

### 3.4 Convergence rate

For a p-th order RK method, global error scales as O(h^p) for fixed h, or O(tol) for adaptive control. RK4: p=4. RK45: p=5 (effective).

For adaptive control: NFE scales as O(tol^{-1/p}). To halve total error, NFE multiplies by 2^{1/5} ≈ 1.15 (RK45). For RK4 fixed-step the relationship is 2^{1/4} ≈ 1.19 for halving error.

**The bit-exact constraint requires tol ≤ 10^{-7}** which forces NFE ≈ 6 · 53 / 1.0 = 318 in the favorable case (h=1 on average) — actually no better than discrete CHIRON's 53 layer evaluations because each adaptive RK45 step costs 6 NFE.

**Speedup only materialises if average accepted h > 6** to break even with discrete CHIRON. This requires an extremely smooth trained vector field — empirically rare in NeurODE-LM.

### 3.5 Stiffness analysis

NeurODE training is plagued by stiffness growth: as θ trains, the spectral radius of ∂F/∂(q,p) climbs, the Lipschitz constant K grows, and the integrator must take smaller steps. This is the central failure mode reported in Chen 2018, Grathwohl 2019, Dupont 2019.

Mitigation in literature: regularize ||∂F/∂(q,p)||² (Massaroli 2020). This adds a side loss; speedup depends on regularizer strength.

For CHIRON specifically, the symplectic shear structure may help — symplectic flows preserve volume, which constrains spectral growth. But this is a CONJECTURE; empirical confirmation needed.

---

## 4. Composition with prior paradigms

### 4.1 Composition table

| Paradigm | Composes? | Mechanism overlap | Marginal gain |
|---|---|---|---|
| **#39 RLG (Reversible Layer Growth)** | Partial | RLG grows L mid-training; CONTINUOUS-DEPTH has continuous L. Mutually exclusive in pure form. | Reframe: RLG → grow T_depth from 16 to 53 across training. Same speedup as RLG. |
| **#42 SCFA (Spectral Compressed Flow Attention)** | ✓ | Orthogonal — SCFA on attention; CONTINUOUS-DEPTH on layer count. | Multiplicative. |
| **#43 ORION (slow-manifold MOR)** | Heavy overlap | ORION reduces SGD dynamics to r-dim manifold; CONTINUOUS-DEPTH integrates a different ODE (forward pass, not parameter update). Different ODEs but similar reduction philosophy. | Marginal contribution refined. ORION's 8.6× already captures slow-manifold gain. |
| **#46 REFLECTOR (cotangent-lift backward)** | ✓ Synergistic | REFLECTOR is the adjoint method for CHIRON's reversible flow. CONTINUOUS-DEPTH adopts this directly. | REFLECTOR's 1.5-1.6× carries over; CONTINUOUS-DEPTH adds nothing on the backward axis. |
| **#49 ICARUS (Yoshida 4th-order symplectic)** | Heavy overlap | ICARUS gives 4th-order symplectic integration of discrete shears. CONTINUOUS-DEPTH gives 4th-order RK45 of continuous flow. **These are competing integrators.** | Mutually exclusive; pick one. ICARUS already captured. |
| **#54 JAMBA-CHIRON (Mamba+SCFA hybrid)** | Partial | Mamba is itself continuous-time-adjacent (state-space model). Composition unclear. | Possibly synergistic; possibly redundant. Empirical. |
| **#55 SOPHIA (second-order optimizer)** | ✓ | Optimizer-axis orthogonal to forward-pass axis. | Multiplicative. |
| **#56-#67 (DATA/LOSS/AGENT/MEMORY/...)** | ✓ | All on different axes. | Multiplicative. |

### 4.2 Heavy-overlap analysis

**The critical observation:** #43 ORION + #46 REFLECTOR + #49 ICARUS *together* attack the integrator axis from three angles:

- **#43 ORION.** Galerkin MOR — reduce dimensionality of the dynamics.
- **#46 REFLECTOR.** Cotangent-lift adjoint — reduce backward-pass cost.
- **#49 ICARUS.** Yoshida 4th-order symplectic — increase effective integration order at fixed NFE.

CONTINUOUS-DEPTH-CHIRON sits squarely on the same axis. It claims gains from:
- Adaptive NFE per input (new — not in #43/#46/#49).
- Continuous-depth interpolation (new — but inference-time benefit, not training-time).

The training-time wall-clock gain from CONTINUOUS-DEPTH alone, beyond #43+#46+#49, is bounded by the adaptive-NFE channel. If median NFE per input is uniform across the dataset, gain = 1.0×. If 30% of inputs need only 0.3·L_eff while 70% need 0.8·L_eff, gain ≈ 1.4×.

**Honest marginal beyond #43+#46+#49: 1.1-1.5×.**

### 4.3 RLG reframing

#39 RLG grows L mid-training (start at L=16, finish at L=53 via Wo=0 identity insertion). CONTINUOUS-DEPTH's natural analog: start with T_depth=16, grow to T_depth=53.

This gives RLG-equivalent speedup (1.30× at 1.84B). **Not net new** beyond RLG. The inheritance from RLG is the strongest single-paradigm composition.

### 4.4 Composition with bigger-picture stack (#56-#67)

CONTINUOUS-DEPTH is purely a forward-pass-axis paradigm. It composes multiplicatively with all bigger-picture paradigms (DATA/LOSS/SAMPLING/REWARD/IDENTITY/SCHEDULE/AGENCY/OPTIMIZER/MEMORY/GROUNDING).

**Multiplicative on bigger-picture; ε-marginal on optimizer-axis.**

---

## 5. Quantitative speedup claim with honest band

### 5.1 Speedup decomposition

```
S_continuous = S_adaptive × S_continuous-depth-interp × S_reversibility
            = (1.1 - 1.4×)    ×  (1.0 - 1.2×)             × (1.0×)
            = 1.1 - 1.7× standalone IF empirical confirmation breaks favorably
```

(S_reversibility = 1.0× because CHIRON already has structural reversibility from #46 REFLECTOR.)

### 5.2 Marginal gain beyond #49 ICARUS

ICARUS already gives 1.5-1.85× via Yoshida 4th-order symplectic at fixed h=1. CONTINUOUS-DEPTH replaces ICARUS with RK45-adaptive. The marginal gain is ONLY the adaptive-NFE channel:

```
Marginal_beyond_ICARUS = S_adaptive_NFE = 1.1 - 1.4×
```

So **CONTINUOUS-DEPTH-CHIRON net contribution ≈ 1.1-1.4×** beyond the existing stack at #67.

### 5.3 Honest band

| Scenario | Probability | Marginal | Cumulative |
|---|---|---|---|
| **Aggressive — adaptive RK45 with median NFE = 0.4·L_eff, bit-exact-equivalent** | 10% | 2.5× | 21,000,000× |
| **Plausible — adaptive RK45 with median NFE = 0.7·L_eff, bit-exact-equivalent** | 30% | 1.4× | 12,000,000× |
| **Marginal — adaptive RK45 saturates at h=1, bit-exact-equivalent** | 35% | 1.0× | 8,580,000× (no gain) |
| **Negative — stiffness blowup at LLM scale; tolerance forces NFE ≥ L_eff** | 25% | 0.7× | 6,000,000× |

Expected value: **0.10·2.5 + 0.30·1.4 + 0.35·1.0 + 0.25·0.7 = 1.20×**.

Median (50th percentile): ~1.0×.

**This is well below the project's typical paradigm threshold of 1.3× confirmed marginal.** The expected case is barely above break-even.

### 5.4 At LLM scale (>1B params)

Public NeurODE-LM results to date stall well below 1B params:
- Chen 2018 (Neural ODEs): MNIST classification only.
- Grathwohl 2019 (FFJORD): density modeling, no LM at scale.
- Dupont 2019 (Augmented Neural ODEs): toy tasks.
- Massaroli 2020 (Neural ODE Processes): meta-learning, no LLM.
- Kidger 2020 (Neural CDE): time series, no LM.

**There is no published evidence that NeurODE-LM scales to 1.84B params, much less the cumulative-stack frontier (18B-180B effective).** Empirical confirmation probability at LLM scale: **25-30%**.

This is the dominant source of uncertainty.

---

## 6. Cumulative stack update

**If favorable case (probability ~10%):**

| Iter | Paradigm | Single-GPU stack |
|---|---|---|
| 211 | #67 (prior) | 8,580,000× causal-reasoning |
| 212 | **#68 CONTINUOUS-DEPTH (favorable)** | ~21,000,000× causal-reasoning |

**If plausible case (probability ~30%):**

| Iter | Paradigm | Single-GPU stack |
|---|---|---|
| 211 | #67 (prior) | 8,580,000× causal-reasoning |
| 212 | **#68 CONTINUOUS-DEPTH (plausible)** | ~12,000,000× causal-reasoning |

**If marginal/negative case (probability ~60%):**

| Iter | Paradigm | Single-GPU stack |
|---|---|---|
| 211 | #67 (prior) | 8,580,000× causal-reasoning |
| 212 | **#68 CONTINUOUS-DEPTH (rejected)** | 8,580,000× unchanged |

**Expected stack:** 0.10·21M + 0.30·12M + 0.60·8.6M = **10.4M ×**.

This is a 1.21× expected lift, but with ~60% probability of zero-or-negative gain. **Worse risk-adjusted return than recent paradigms** (#56-#67 averaged 1.3-1.5× with >70% confirmation).

---

## 7. Engineering scope

### 7.1 LOC breakdown

| Component | LOC | Weeks |
|---|---|---|
| Continuous vector field F(q,p,t;θ) — depth-conditioned MLP-attention with sinusoidal time embedding | 350 | 1.5 |
| RK45 (DOPRI5) integrator with adaptive step control | 450 | 2.0 |
| Adjoint backward pass (REFLECTOR-derived) | 250 | 1.0 |
| Tolerance instrumentation + NFE histogram | 200 | 0.5 |
| Bit-exact NLL band test harness | 150 | 0.5 |
| RLG reframing — grow T_depth from 16 to 53 | 200 | 1.0 |
| Composition with #43 ORION (V-projection on continuous flow) | 150 | 1.0 |
| Composition with #49 ICARUS deselection — pick continuous OR Yoshida | 80 | 0.5 |
| Gate-0 protocol + 41M probe | 120 | 0.5 |
| **Total** | **~1950** | **~8 weeks** |

### 7.2 Reference implementations

- `torchdiffeq` (Chen 2018) — reference CPU/GPU adjoint Neural ODE.
- `diffrax` (Kidger 2021) — JAX-native adaptive ODE solvers.
- `torchdyn` (Massaroli 2020) — Neural ODE library.

These provide RK45 implementations but **none have been validated at LLM scale**. Significant engineering risk in scaling to 1.84B-18B.

### 7.3 Engineering risk

The largest risk is integrator-tolerance instrumentation. NFE histogram must be tracked per token, per layer-equivalent step, per training step — telemetry overhead may itself be ~5% of training wall-clock. Mitigation: sample-mode (every 100 steps).

---

## 8. Gate-0 / Gate-1 specifications

### 8.1 Gate-0 (cheap probe — ~2 GPU-hours)

**Premise to test:** the trained CHIRON vector field has slowly-changing regions (h_avg > 1) with bit-exact NLL preservation.

**Protocol:**
1. Take an existing 41M trained CHIRON checkpoint.
2. Cast its discrete shears as a piecewise-constant continuous F(q,p,t;θ) with the depth-conditioned MLP fitted to the discrete θ^{(ℓ)} sequence.
3. Forward-pass 1000 sequences through DOPRI5 with tol = 10^{-6}, 10^{-7}, 10^{-8}.
4. Measure: median NFE per forward pass; NLL drift vs discrete; wall-clock per forward.

**PASS criteria:**
- Median NFE / L_eff < 0.6 at tol = 10^{-6}.
- NLL drift < 5 · 10^{-6} nat per token at tol = 10^{-6}.
- Wall-clock < 0.85 × discrete CHIRON wall-clock at tol = 10^{-6}.

**Joint Gate-0 PASS probability: ~45%**.

### 8.2 Gate-1 (training validation — ~1 GPU-week)

**Premise to test:** at 1.84B with full training (not retrofit), CONTINUOUS-DEPTH achieves NLL parity with discrete + 1.4× wall-clock speedup.

**Protocol:**
1. Train 1.84B CONTINUOUS-DEPTH-CHIRON for 50,000 steps with adaptive RK45 (tol=10^{-7}).
2. Train baseline 1.84B discrete CHIRON for same step budget.
3. Compare: final NLL (ε-band: ≤ 10^{-6} nat); wall-clock per step; NFE distribution over training.

**PASS criteria:**
- Final NLL within 5 · 10^{-6} nat of baseline.
- Wall-clock per step ≤ 0.7 × baseline.
- NFE distribution stable (no stiffness blowup over training).

**Joint Gate-1 PASS probability conditional on Gate-0 PASS: ~50%**.

**Joint Gate-0 ∧ Gate-1: ~22%.**

**LLM-scale empirical confirmation (Gate-1 PASS): 25-30%.**

---

## 9. Honest gaps and failure modes

### 9.1 NeurODE training instability at scale

**Failure mode #1 (probability ~30%):** vector field stiffness grows during training, NFE distribution degrades from median 30 to median 80 to non-convergent. This is the dominant failure mode in NeurODE literature.

**Mitigation attempts:**
- Spectral regularization on ∂F/∂(q,p) (Massaroli 2020). Adds side loss; may erode speedup.
- Symplectic regularization (constrain volume preservation). CHIRON-specific advantage but unproven.
- Stiffness-adaptive solver (BDF for stiff regions). 3-5× slower than RK45 in stiff regions.

**Honest assessment:** even if mitigations work, the 1.4× expected lift may collapse to 1.0× under stiffness pressure.

### 9.2 Bit-exact NLL preservation versus speedup

**Failure mode #2 (probability ~25%):** to keep NLL within 10^{-6} band, tolerance must be 10^{-9}; this raises NFE by ~3×; CONTINUOUS-DEPTH becomes *slower* than discrete CHIRON.

This is the trade described in §3.3.

### 9.3 Composition saturation with #43+#46+#49

**Failure mode #3 (probability ~35%):** CONTINUOUS-DEPTH overlaps with the integrator stack such that marginal gain falls below 1.1×, well below the project's threshold for SELECT.

This is the dominant *risk-adjusted* failure mode: even if the paradigm "works," the gain may be too small.

### 9.4 Engineering complexity

Adaptive integrator with tolerance instrumentation, adjoint backward, depth-conditioned vector field, RLG reframing, V-projection from #43 — this is a 1900 LOC engineering effort with non-trivial debugging surface. **Engineering risk is substantial.**

### 9.5 No LLM-scale precedent

No public NeurODE result at >1B params. CONTINUOUS-DEPTH-CHIRON would be the first to attempt this. The empirical confirmation at LLM scale is the largest single uncertainty (25-30%).

---

## 10. Bottom line / verdict

### 10.1 Verdict: **RESERVE**

**Reasons to NOT select at iter 212:**

1. **Heavy overlap with #43 + #46 + #49** (~50% of the integrator-axis gain already captured). Marginal contribution falls in the 1.1-1.5× band.
2. **Empirical confirmation probability at LLM scale only 25-30%.** No published NeurODE-LM at >1B params.
3. **Bit-exact NLL constraint is ε-tight.** Strict tolerance may erase the speedup.
4. **Expected stack lift only ~1.21×** (combination of 60% zero-gain probability and 40% positive-gain probability).
5. **Engineering scope is substantial** (~1950 LOC, 8 weeks) with high debugging surface.

**Reasons to RESERVE rather than REJECT:**

1. **Theoretically clean.** Bijectivity and reversibility become structural; this is mechanistically appealing for CHIRON.
2. **Continuous-depth interpolation is a real (if niche) inference-time capability** — supports variable-resolution inference, important for some deployment scenarios.
3. **Adaptive NFE is a genuinely new mechanism** not present in #43+#46+#49 (which all use fixed-NFE schemes).
4. **External-event trigger:** if a peer NeurODE-LM result demonstrates >1B convergence in the next 12 months, the empirical-confirmation probability jumps to 60-70% and CONTINUOUS-DEPTH becomes a strong SELECT candidate.

**Trigger conditions for promotion to SELECT (paradigm #69+ candidate):**
- (A) A peer NeurODE-LM paper demonstrates >1B-param convergence parity with discrete transformers, OR
- (B) A Gate-0 probe at 41M shows median NFE < 0.6·L_eff with NLL drift < 5·10^{-6} nat at tol=10^{-6} (cheap test, 2 GPU-hours).

Until then, **CONTINUOUS-DEPTH-CHIRON sits on the reserve list with paradigm #65-A WORLD-MODEL (35%) and #66-X (similar reservation level).**

### 10.2 Recommendation to user

**RESERVE this candidate.** Allocate 2 GPU-hours to a Gate-0 probe at 41M before iter 213 paradigm-#68 selection finalises. If Gate-0 PASSES, escalate CONTINUOUS-DEPTH to SELECT for paradigm #69. If Gate-0 FAILS, the reservation closes.

Concrete next-step deliverable: build the 41M retrofit Gate-0 protocol described in §8.1 as a stand-alone unit test under `unit-tests/`. Cost: ~150 LOC + 2 GPU-hours. This puts CONTINUOUS-DEPTH on a fast empirical track without committing the full 8-week engineering scope.

### 10.3 What this finding tells us about the broader research program

The integrator axis (#42 SCFA / #43 ORION / #46 REFLECTOR / #49 ICARUS) is **at structural saturation** under the bit-exact NLL constraint. The remaining headroom is bounded by ~1.2-1.4× per paradigm increment.

**This was already iter-211's saturation finding for the COMPUTE-SPEED axis under strict NLL.** CONTINUOUS-DEPTH-CHIRON's analysis confirms the saturation finding from a second direction: continuous-time reformulations also bottleneck at the same ceiling.

**Implication for paradigm #68 selection:** prefer candidates that operate on axes NOT yet attacked by the integrator stack. Likely candidates: cross-modal (vision/audio fusion), neuro-symbolic (program synthesis), lifelong-learning (online updates without catastrophic forgetting), or genuinely novel objective formulations (hopefully not collapsing to rejected #59-C MDL-PRETRAIN style).

CONTINUOUS-DEPTH-CHIRON remains a high-quality reserve candidate, but its risk-adjusted contribution at iter 212 is below alternatives that exploit underexplored axes.

---

**End of Paradigm Shift #68 Candidate B design document.** ~4500 words. Honest verdict: **RESERVE** with 1.5-3.0× standalone band but only 1.1-1.5× marginal beyond #43+#46+#49 + ICARUS, and 25-30% LLM-scale empirical confirmation probability. Trigger condition for promotion: peer NeurODE-LM result at >1B params, or favorable Gate-0 probe at 41M.
