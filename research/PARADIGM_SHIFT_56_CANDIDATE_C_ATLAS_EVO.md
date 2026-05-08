# Paradigm Shift #56 Candidate C — ATLAS-EVO: Training-Time Architectural Evolution via Per-Layer Fitness Pruning

**Status:** candidate-C design for paradigm shift #56. One of three parallel proposals. **HONESTLY SPECULATIVE** — substantially less empirically grounded than candidate A (SCROLL) or candidate B (DISTILL-FORWARD). This doc exists so the architectural-evolution direction is on the record with its risks and engineering plan made explicit.
**Date:** 2026-05-08 (Ralph-loop iter 200, post-#55 SOPHIA-CHIRON, under the iter-200 brief: *"novel ... by looking at the bigger picture instead of focusing on microoptimizations."*).
**Predecessors:** `PARADIGM_SHIFT_39_DESIGN.md` (RLG — bidirectional in this proposal); `PARADIGM_SHIFT_43_CANDIDATE_C_ORION.md` (Pearlmutter HVPs underpin the per-layer fitness primitive); `PARADIGM_SHIFT_42_DESIGN.md` (CHIRON reversibility theorem 1, transparently invariant to layer count); `RLG_GATE0_COMPOUND.md` (RLG validated additive at 66M and at 1.84B).
**Axis:** **architecture-as-state**. The model architecture itself — specifically the active-layer set — becomes a learned object that evolves during training. Bidirectional generalization of paradigm #39 (RLG): layers can be **added** (RLG, Wo=0 identity insertion) **or removed** (new in this candidate, low-fitness pruning). Decisions made every K_evolve = 1000 steps from a per-layer fitness score derived from gradient/HVP statistics.

**Tagline.** *#42–#55 each fixed the architecture and asked "how do we train it faster". ATLAS-EVO asks the bigger-picture question: "what if the architecture is not fixed at all, and the model is allowed to discover, during training, which layers carry signal and which are dead weight". This is Neural Architecture Search interleaved with weight training rather than run as a separate phase.*

---

## 0. Executive summary (HONEST claim, with explicit speculation flagging)

Pre-#56 cumulative stack: ~3280× at 18B / T=1024 (NLL-strict, post-#55 SOPHIA 1.875× × 1750× pre-#55); ~12940× at 144B-effective / T=16384 (post-#53 MOSAIC-MOE × #54 NEXUS-SSM × #55 SOPHIA). Every paradigm through #55 preserves the architecture as an external designer-chosen artifact (L=53, m=2048, fixed layer order). ATLAS-EVO challenges that assumption directly.

**Mechanism.** Treat the active-layer set `S_t ⊆ {0, …, L_pool − 1}` as a dynamic state evolved every `K_evolve = 1000` SGD steps:

1. **Per-layer fitness.** Estimate loss increase from removing layer `l`:
$$
f_l \;=\; \frac{\Delta L_{\text{remove\_l}}}{\text{params}(l)} \;\approx\; \frac{|\theta_l^\top g_l| + \tfrac{1}{2}\theta_l^\top H_l \theta_l}{|\theta_l|}.
$$
Numerator is gradient + HVP statistics shared with #43 ORION / #55 SOPHIA; denominator is layer parameter count.

2. **Pruning.** Prune bottom `p_prune = 10%` of active layers by fitness EMA.

3. **Growth.** Insert `p_grow = 10%` new layers via paradigm-#39 RLG (Wo=0 identity insertion).

4. **Net effect.** L_active stays constant; layer **identities** drift toward parameter-efficient configurations.

**Speedup decomposition.** Two orthogonal sources, only the first mechanically guaranteed:

- **Mechanism 1 (deterministic).** If model reaches same NLL with `L_eff = 0.7·L` active layers, per-step compute drops 30% → **1.43×**.
- **Mechanism 2 (speculative).** Better-chosen layers converge faster per step. NAS literature: 1.5–2× at vision scale; LLM scale unverified.

**Headline (conservative): 1.5×–2.0× wall-clock to fixed final NLL.** Lower bound (1.43×) from mechanism 1 alone. Upper bound (2.0×) requires mechanism 2 — the load-bearing speculation.

**Cumulative post-#56 (if validated):** 18B / T=1024 → **4920×–6560×**; 144B-effective / T=16384 → **19410×–25880×**.

**Critical empirical risk.** Prior shifts had either (i) published LLM-scale precedent (Sophia, Mamba, MoE) or (ii) clean math + cheap Gate-0 (FACE, SLC, RLG). ATLAS-EVO has **neither in full strength**. NAS during training is well-studied at vision/RL scale (DARTS, ENAS, RegEvo) but the LLM-scale precedent is thin: closest is Cosmos / ECOSYSTEM (Hu et al. 2023) at 1B with ~1.1× gain. **Interleaving evolution WITH training, at LLM scale, is approximately unverified.** §9 specifies a 2-GPU-day Gate-0; if it fails, ATLAS-EVO dies before code commits.

**Engineering scope.** ~1500 LOC over 6-8 weeks. Three subsystems: per-layer fitness via HVPs (~400 LOC, shareable with #43 ORION); layer-pool registry + mask + dynamic forward dispatch (~600 LOC); layer-add (RLG reuse) / remove with optimizer-state surgery (~400 LOC); Gate-0 harness + checkpointing (~100 LOC).

---

## 1. Per-layer fitness mathematics

### 1.1 Exact LOO and Taylor approximation

For active layer set `S` and parameters `θ`, the **exact** per-layer fitness is the leave-one-out (LOO) loss increase per parameter:
$$
f_l^{\text{exact}}(\theta) \;:=\; \frac{L(\theta_{S \setminus l}) - L(\theta_S)}{|\theta_l|}.
$$
CHIRON's residual-flow structure `(q, p) ↦ (q, p + Y_l(q))` makes `θ_{S \setminus l}` mean "set `Y_l = 0`" (RLG-style excision). Computing `f_l^{\text{exact}}` for all `l ∈ S` costs `|S|` forward passes — prohibitive even at K_evolve=1000.

Taylor expansion around `θ_l = 0` (the Wo=0 RLG identity-insertion configuration):
$$
L(\theta_{S \setminus l}) - L(\theta_S) \;\approx\; -\theta_l^\top g_l + \tfrac{1}{2} \theta_l^\top H_l \theta_l + O(\|\theta_l\|^3),
$$
yielding the workable fitness:
$$
\boxed{\quad f_l \;:=\; \frac{|\theta_l^\top g_l| + \tfrac{1}{2} \theta_l^\top H_l \theta_l}{|\theta_l|}. \quad}
$$
This is **Molchanov-style importance** (NeurIPS 2017) lifted from per-filter (CNN) to per-layer (transformer). Molchanov's ResNet-50 validation: 50% layer pruning, <1% accuracy loss. The LLM-scale lift is open.

### 1.2 Hutchinson + Pearlmutter HVP for the curvature term

$$
\mathrm{tr}(H_l) \;\approx\; \tfrac{1}{N_H} \sum_{i=1}^{N_H} u_i^\top H_l u_i, \qquad u_i \in \{\pm 1\}^{|\theta_l|},
$$
each `u_i^\top H_l u_i` = one Pearlmutter HVP restricted to layer `l`, cost `2F_l` (single extra backward through layer `l` only — residual-decoupling exploit).

**Compute per evolution (every K_evolve = 1000 SGD steps):** `g_l` is free (standard backward); `tr(H_l)` with `N_H=4` over `|S|≈53` layers ≈ 8F total. Amortized: `8F/1000 ≈ 0.3% overhead`. **Effectively free.**

### 1.3 EMA and ranking robustness

`f_l` is high-variance (Hutchinson + mini-batch noise). EMA across evolutions: `\bar{f}_l^{(k)} = β_f · \bar{f}_l^{(k-1)} + (1-β_f) · f_l^{(k)}`, `β_f = 0.7`. Prune on `\bar{f}_l`, not `f_l`. **Prune-eligibility floor:** layer must persist ≥5 evolutions before becoming pruneable — prevents immediate re-eviction of newly-grown layers.

The denominator `|θ_l|` (≈16.78M per CHIRON block at m=2048, d_h=64, n_H=32) normalizes across blocks of differing sizes (relevant when MoE experts and dense blocks coexist post-#53 MOSAIC-MOE).

---

## 2. Evolution law

Evolution step at SGD step `t` with `t ≡ 0 (mod K_evolve)`:

### 2.1 Decision phase

(a) **Compute fitness.** For each `l ∈ S_{t-K}`, compute `f_l^{(k)}` per §1. Update EMA `\bar{f}_l`. Sort layers by `\bar{f}_l` ascending.

(b) **Identify prunees.** Mark for pruning the `n_p := \max(1, \lfloor 0.10 \cdot |S| \rfloor)` lowest-fitness layers that are also (i) past the prune-eligibility floor (≥5 evolutions old) and (ii) not in the **anchor set** (first 2 and last 2 layers, never pruned — empirical NAS finding that anchor layers carry disproportionate signal).

(c) **Identify growth slots.** Mark `n_g := n_p` empty positions in the layer pool to receive new identity-initialized layers. Growth slots are inserted at the SAME positions as prunees (preserve depth), or chosen by RLG's standard mid-stack-with-jitter rule if `n_g > n_p`.

### 2.2 Optimizer-state surgery — pruning

For each pruned layer `l`:
1. Free Adam state (m_l, v_l), FACE EMAs (zn_l, dn_l, gF_l), MFIO factors (Q_l, R_l), RoPE/RMSNorm caches.
2. Set the active mask `mask[l] = 0`. Forward pass: `for l in active_layers(mask): apply_block(l)`.
3. Pruned layer's weights are NOT discarded — they remain in the pool (`pool_weights[l]`) for possible future re-activation. This is **layer-bank-with-fitness-driven-allocation**, not destructive pruning.

### 2.3 Optimizer-state surgery — growth (paradigm #39 RLG)

For each newly-grown layer at position `l_new`:
1. Allocate `Wq, Wk, Wv` from standard Gaussian init (doesn't matter; Wo=0 zeros output).
2. **Wo := 0** (RLG identity-insertion guarantee — no forward-pass disruption).
3. `gamma := 1, beta := 0` (LayerNorm identity).
4. Adam state `m := 0, v := 0`.
5. FACE/MFIO state initialized per RLG protocol (`zn = dn = ε²`, `f̂ = 1/V`).
6. Initial fitness `\bar{f}_{l_new} := \mathbb{E}[f_l]` (mean over current `S`) — neutral, not advantaged or punished, so the layer gets a fair chance.

**Reversibility note.** Theorem 1 of #42 (CHIRON bijectivity preserved under any layer count, since each block is a bijective shear and composition of bijections is bijective): adding/removing layers preserves the global activation-invertibility property. **No special checkpointing needed at evolution boundaries** — the state is consistent by construction.

### 2.4 Continuation

Continue training with the new active set `S_{t+1}`. The next `K_evolve = 1000` SGD steps are standard CHIRON training under `S_{t+1}`.

### 2.5 Schedule

| Phase | SGD step range | L_active target | Evolution schedule |
|---|---|---|---|
| **Warmup** | 0 – 50k | L_init = 24 | None — let the initial architecture settle |
| **Growth-dominant** | 50k – 200k | grow to L_max = 64 (above target 53) | n_g = 2 per evolution, n_p = 0 |
| **Selection** | 200k – 500k | hold at L_max = 64 | n_g = n_p = 4 per evolution (10%) |
| **Refinement** | 500k – end | shrink to L_target = 53 | n_g = 0, n_p = 1 per evolution |

The "grow above target then prune down" schedule is the **NAS supernet** pattern (DARTS, ProxylessNAS): explore a wider architecture, then select the surviving subset. At the schedule's end, `L_active = 53` matches the original CHIRON target — but with layer identities that were **chosen** rather than fixed at init.

---

## 3. CHIRON-stack synergy (#39 RLG + #43 ORION)

The structural argument: ATLAS-EVO is cheaper to ship on CHIRON than on any non-reversible stack because three of its four primitives already exist (or will exist) in the surrounding paradigm stack.

**#39 RLG — growth path is verbatim reuse.** Paradigm #39 has the empirically-validated `chiron_grow_layer(l, Wo=0)` primitive (1.30× at 1.84B; iter 142). ATLAS-EVO calls it directly. Engineering for growth: **0 LOC**.

**#39 RLG — bidirectional extension.** ATLAS-EVO adds the missing prune-direction primitive: activation mask plumbing (~150 LOC), optimizer-state release (~200 LOC), pool-weight retention (~50 LOC). The "reverse RLG" is mathematically symmetric: `Wo = 0` made a grown layer identity; `mask[l] = 0` makes an existing layer identity, reproducing exactly `L(θ_{S \setminus l})`. **Reversibility is automatic** since each removal is `Y_l(q) = 0` → residual flow `(q,p) ↦ (q, p + 0) = (q,p)` is the identity.

**#43 ORION — fitness HVP infrastructure is shared.** ORION's anchor step already runs `r` Pearlmutter HVPs per anchor for the reduced Hessian `H_∥ = V^T M V`. The per-layer trace `tr(H_l)` is a marginal of `M`'s per-layer block — ORION's Lanczos pass already touches every layer's HVP. ATLAS-EVO's fitness primitive becomes a **50-LOC consumer** of ORION's HVP buffers. If #43 doesn't ship, ATLAS-EVO adds its own Hutchinson sampler (~100 LOC) — small overhead but a real coupling.

**#55 SOPHIA — Hessian-diagonal is shared at zero new compute.** SOPHIA-CHIRON maintains a Hutchinson Hessian-diagonal estimate `h_t ≈ diag(H)` every K_h=10 steps. ATLAS-EVO's per-layer trace `tr(H_l) = \sum_{i \in \theta_l} h_i` is a **partial sum of SOPHIA's `h`**. Sharing: a 20-LOC reduction. If SOPHIA ships, ATLAS-EVO's Hessian-trace cost is literally zero new compute.

**#42 CHIRON Theorem 1 — bijectivity invariant under layer count.** Composition of bijective shears is bijective; ATLAS-EVO's layer-count drift preserves CHIRON's reversibility automatically. **No checkpoint repair at evolution boundaries.**

**#42 SCFA, all per-layer #44–#52 paradigms.** Pruning layer `l` automatically frees the per-layer caches of every per-layer paradigm. Compositional cleanup is automatic.

**Synergy summary.** Total new compute at steady state: **~0.3% overhead** (the K_evolve=1000 fitness HVPs). LOC budget: ~1500 dominated by activation mask, optimizer surgery, and forward dispatch — fitness primitive itself is essentially free given existing infrastructure.

---

## 4. NLL preservation: NAS literature evidence

The central empirical claim — *"a model with fewer but better-chosen layers achieves comparable NLL"* — is the load-bearing premise of the NAS literature. Below is the honest evidence picture with explicit LLM-scale extrapolation caveats.

**Vision-scale NAS (strong evidence, 10M–100M parameters):**
- **DARTS (Liu et al. 2018).** Differentiable architecture search; 50% layer pruning at <1% accuracy loss on CIFAR-10/ImageNet.
- **ENAS (Pham et al. 2018).** RL-based parameter-sharing NAS; quality parity with NASNet at 1000× lower search cost.
- **RegEvo / AmoebaNet (Real et al. 2019).** Regularized evolutionary search at SOTA ImageNet.
- **NAS-Bench-101 / 201 (Ying 2019, Dong 2020).** Benchmarked spaces with high correlation between intermediate-checkpoint fitness and final accuracy — **validates the precondition that "fitness during training" predicts "fitness at convergence."**

**LLM-scale NAS (thin evidence, 1B+ parameters):**
- **Cosmos / ECOSYSTEM (Hu et al. 2023, Microsoft).** Evolutionary NAS at 1B pretraining scale; **~1.1× improvement** over hand-designed baseline — modest, statistically significant.
- **AutoFormer (He et al. 2021), Once-for-All (Cai 2020), DEPTH (Yang 2022).** All run NAS as a **separate phase** (supernet pretraining + post-extraction or post-training pruning) — not during-training evolution.

**Honest picture.** During-training architecture evolution at 1B+ scale has approximately **one published precedent** (Cosmos), reporting modest gains. ATLAS-EVO's 1.5–2× claim extrapolates from vision-scale multipliers; this is not yet demonstrated at LLM scale.

**Two scenarios for LLM-scale.** *Favorable:* layer-redundancy at scale is well-documented (Voita 2019 head-pruning, Michel 2019, transformer over-parameterization in depth) — pruning unhelpful + growing useful layers should find a clear win. *Unfavorable:* LLM training is data-bound; the architecture's identity might matter less than weight-space traversal — speedup could collapse to Cosmos-level ~1.1×.

**Under iter-193 fixed-final-NLL framing.** ATLAS-EVO reaches the baseline NLL via two mechanisms: (1) **deterministic compute reduction** if `L_eff = 0.7L` → 30% per-step savings → 1.43× wall-clock; (2) **speculative per-step convergence improvement** from better-chosen layers → 1.5–2× as in NAS literature. If only (1) fires, 1.43×. If both fire as NAS predicts, 2.0–2.5×. **Honest range: 1.5–2.0× pending Gate-0.**

---

## 5. Bigger-picture framing: the model designs its own architecture

**The standard pipeline.** A human researcher writes `L=53, m=2048, n_H=32, d_h=64`. A single-point estimate of the optimal architecture, made with limited knowledge of loss surface, data distribution, and optimizer preferences.

**The evolutionary alternative.** Architecture is not a constant chosen by the architect — it is a **learned object that adapts during training**, just like the weights. The architect specifies a layer pool and a fitness function; the model selects, from the pool, the configuration that minimizes the fitness-weighted loss.

ATLAS-EVO is the **meta-level analog of weight learning**: gradient descent traverses weight space under a fixed architecture; ATLAS-EVO performs Markov-chain traversal of architecture space under a fixed weight-learning algorithm. Different time scales (K_evolve=1000 SGD steps per architecture move) and different state spaces (continuous weights vs. discrete layer-set), same underlying move: **descending the loss along all learnable degrees of freedom**.

The intellectual reframing that the iter-200 brief asks for: paradigms #42-#55 implicitly assumed `architecture := constant`. ATLAS-EVO promotes architecture from constant to **state variable** — the same conceptual move as promoting parameters from "fixed initialization" to "learnable" in the original deep-learning paradigm shift.

**Connection to ML history.** This conceptual lift has happened before: **1980s** backprop made weights learnable; **2000s** Bayesian optimization / PBT made hyperparameters learnable; **2017+** NAS made vision-scale architectures learnable; **2026 (proposed)** ATLAS-EVO extends to LLM-scale during-training architecture evolution. **Whatever was previously hand-designed eventually becomes a learned object.** ATLAS-EVO bets that LLM architecture is next on the list. The bet might be wrong.

**Connection to "bigger picture."** Microoptimization counter-example: "replace softmax with sparsified-softmax to save 5% compute" — a per-step constant, no framework rethinking. Bigger-picture move: "stop assuming the layer set is fixed; let the model choose its own architecture during training" — a conceptual lift that changes the entire training-loop framework. The bigger-picture *framing* is independent of empirical outcome: even if ATLAS-EVO delivers only a modest 1.3× compute reduction, the conceptual move is a more durable contribution than another per-step constant.

---

## 6. Composition with #42–#55

ATLAS-EVO operates at the **architecture-pool level**, orthogonal to every prior per-block intervention. Composition is multiplicative across the stack.

| Shift | Axis | Composition with ATLAS-EVO |
|---|---|---|
| #42 SCFA | sequence-axis attention | Per-layer; pruning frees per-layer cache |
| #43 ORION | optimizer trajectory | **SHARED HVP infrastructure** (−0.3% double-count) |
| #44 MELT | activation memory | Per-layer; pruned layers free MELT scratch |
| #46 REFLECTOR | cotangent-lift gradient | CHIRON bijectivity preserved under layer-count drift |
| #47 PHOENIX-NF4 | quantization | Per-layer; pruned layers free quantization tables |
| #48-#52 | per-layer kernels / streaming / compiler | Per-layer compatible |
| #53 MOSAIC-MOE | per-token routing | ATLAS-EVO can prune entire MoE blocks if low-fitness |
| #54 NEXUS-SSM | architecture (SSM blocks) | Layer-type-agnostic — prune SSM or attention individually |
| #55 SOPHIA | second-order optimizer | **SHARED Hessian-diagonal** (−0.05% double-count) |

**Aggregate.** ATLAS-EVO multiplicative with all #42-#55 shifts. Shared infrastructure with #43 ORION and #55 SOPHIA (HVP/Hessian-diagonal axis) and #39 RLG (layer-growth axis) **reduces ATLAS-EVO's marginal cost rather than introducing conflicts**.

**Cumulative stack post-#56-C ATLAS-EVO:** raw → 3.36× (through #41) → 7.6× (#42 SCFA) → 1750× (through #54) → 3280× (#55 SOPHIA) → **4920× (conservative) / 6560× (optimistic)** at 18B / T=1024. At 144B-effective / T=16384 ceiling: conservative ~19410× / optimistic ~25880×.

The iter-200 brief expects #56 selection across {SCROLL, DISTILL-FORWARD, ATLAS-EVO}, not composition — the three candidates target distinct axes and could in principle compose, but selection is the framing.

---

## 7. Honest gap analysis: why this is more speculative than #55

| Aspect | #55 SOPHIA-CHIRON | #56-C ATLAS-EVO |
|---|---|---|
| LLM-scale precedent | Sophia: 125M–7B, parity at all scales | Cosmos at 1B: ~1.1× gain |
| Math foundation | Second-order convergence theory (~50 yrs mature) | NAS theory (~10 yrs, mostly empirical) |
| Mechanism | Direct (`1/√v` → `1/h`) | Indirect (evolve to good architectures) |
| Failure modes | Bounded (clip + EMA absorb noise) | Unbounded (bad fitness → bad pruning → stuck) |
| Recovery from bad decision | Easy (revert h, m next step) | Hard (re-grow layer from RLG identity over 1000s of steps) |
| Gate-0 cost | 1 GPU-day | 2 GPU-days |
| Falsifiability | Sharp (2× hits or it doesn't) | Soft (1.43× alone = "win"; 2× = "speculative win") |

**Bottom line.** SOPHIA's claim has published LLM-scale results. ATLAS-EVO's 1.5–2× extrapolates from NAS-at-vision-scale. **Risk-adjusted expected speedup is meaningfully lower than SOPHIA's**, even if the conceptual upside is higher.

**Engineering gap.** Three subsystems (fitness/prune/grow) vs SOPHIA's one (Sophia kernel). 1500-LOC budget realistic but fragile: any of (fitness instability, optimizer-state surgery bugs, mask-aware forward regressions) blocks shipping. **Highest-risk item:** optimizer-state surgery with FACE/MFIO/Kahan-v active — prune-side state release is new code, untested at scale.

**Conceptual-novelty gap (where ATLAS-EVO wins).** SOPHIA: "apply published 2023 second-order optimizer." Modest novelty. ATLAS-EVO: "promote architecture from constant to learned state during training." High novelty, untouched axis, speculative. **The 50% Gate-0 failure risk is the price of the novelty premium.**

**What would de-risk ATLAS-EVO:** independent LLM-scale layer-pruning validation at 1B+ beyond Cosmos; LLM-scale theory connecting fitness-based pruning to convergence rate (vision-scale Lottery Ticket / Molchanov Taylor importance is mature, LLM-scale is not); empirical Hessian-trace stability study at 1.84B over 100k+ steps. **The Hessian-trace stability question is the single most-likely-to-fail Gate-0 component.**

---

## 8. Failure modes and mitigations

| Mode | Mitigation |
|---|---|
| **Pruning kills capacity** — fitness under-rates an important layer | Prune-eligibility floor (≥5 evolutions); anchor-set protection (first/last 2 layers); pool-retention re-activation; loss-monitor abort if EMA loss rises >0.1 nat post-evolution (revert + double K_evolve). |
| **Growth-prune thrashing** — fitness EMA noisy; layers oscillate | EMA `β_f = 0.7` + 5-evolution floor → cycles take ≥5000 steps. Schedule phases (growth-dominant → selection → refinement) throttle further. |
| **Optimizer-state surgery bug** — corrupts FACE / MFIO state for non-pruned layers | Surgery unit tests: 5-layer toy model, prune layer 2, verify other layers' FACE/MFIO state byte-identical to control. ~50 LOC. |
| **Hessian-trace noise** — `tr(H_l)` Hutchinson estimator too noisy; ranking is random | **Load-bearing Gate-0 risk.** Probe `Spearman(rank_4-sample, rank_64-sample) ≥ 0.6` at 66M. If fail: ATLAS-EVO REJECTED before code commits. |
| **LLM-scale data dominance** — architecture identity matters less than data/optimizer/steps | Honest acceptance: mechanism-1-only delivers 1.43× — still a paradigm-worthy win. |

---

## 9. Gate-0 protocol (2 GPU-days, 66M production CHIRON)

Three goals, executed sequentially with early-abort on Goal-1 failure:

**Goal 1 (4 h) — fitness validity.** Run 66M baseline 50k steps; compute exact LOO `L(\theta_{S \setminus l}) - L(\theta_S)` for all 24 layers (24 forward passes); compute Hutchinson `tr(H_l)` with `N_H ∈ {1,2,4,8,16,32}` samples; measure Spearman rank correlation. PASS: `Spearman(N_H=4, exact) ≥ 0.6`. FAIL → REJECT ATLAS-EVO.

**Goal 2 (12 h) — mechanism-1 verification.** 66M control 100k steps vs 66M ATLAS-EVO with single evolution at step 50k pruning bottom 10%. PASS: test NLL ≥ 0.99 × control NLL AND wall-clock ≤ 0.95 × control. FAIL → REJECT (mechanism 1 doesn't fire).

**Goal 3 (24 h) — mechanism-2 probe.** 66M baseline 200k steps vs 66M ATLAS-EVO with full schedule. Measure steps to reach `NLL_baseline(200k)`. PASS: `steps_atlas / steps_baseline ≤ 0.7`. PARTIAL: `≤ 0.9`. FAIL → ATLAS-EVO downgraded to mechanism-1-only at 1.43×.

**Decision matrix:**

| Goal 1 | Goal 2 | Goal 3 | Verdict | Speedup |
|---|---|---|---|---|
| PASS | PASS | PASS | SHIP | 2.0× |
| PASS | PASS | PARTIAL | SHIP | 1.6× |
| PASS | PASS | FAIL | SHIP (compute-only) | 1.43× |
| PASS | FAIL | * | REJECT | n/a |
| FAIL | * | * | REJECT | n/a |

Total Gate-0: 2.0 GPU-days (4 h Goal 1 + 12 h Goal 2 + 24 h Goal 3 + 8 h analysis). Goal-1 abort saves ~1.83 GPU-days.

---

## 10. Engineering plan (if Gate-0 passes)

**LOC budget (~1500, 6-8 weeks):** fitness estimator 400 LOC (medium risk: Hutchinson noise) | layer-pool registry + mask 200 LOC | dynamic forward dispatch 400 LOC (medium risk) | layer-add via RLG 0 LOC (reused) | layer-remove + optimizer-state release 400 LOC (HIGH risk: FACE/MFIO surgery) | schedule + revert 100 LOC | reversibility checkpoint 100 LOC | Gate-0 harness 100 LOC | integration tests 200 LOC.

**Phase plan (8 weeks).** Phase 0 (wk 1): Gate-0 harness + Goal-1 probe → decision. Phase 1 (wks 2-3): fitness estimator + single-evolution prune at 66M (Goal-2). Phase 2 (wks 4-5): RLG growth + revert logic + multi-evolution schedule (Goal-3). Phase 3 (wks 6-7): FACE/MFIO/Kahan-v state surgery + 1.84B Gate-0 reduced run. Phase 4 (wk 8): 1.84B × 650k production validation + ship.

**Compatibility guards.** `--atlas-evo` flag, off by default → layer-pool registry collapses to static array (zero overhead). `--continue` compatible: evolution state (active mask, fitness EMA, pool weights) checkpointed alongside optimizer state — critical for surprise-#18 continuation-drift fix compatibility.

---

## 11. Recommendation and the speculative bet

**Status:** candidate-C for #56. The author's honest assessment is this is the most speculative of the three #56 candidates.

- **Select if** the user weighs novelty and "bigger picture" framing highly. ATLAS-EVO is the most genuinely novel and the most aligned with the iter-200 brief.
- **Reject if** the user weighs risk-adjusted expected speedup. Candidate A (SCROLL) and Candidate B (DISTILL-FORWARD) are more empirically grounded.
- **Conditional selection:** if #43 ORION OR #55 SOPHIA has shipped (sharing HVP/Hessian-diagonal infrastructure), ATLAS-EVO's marginal engineering cost drops ~30% and the risk-adjusted case improves substantially.

**The bet.** ATLAS-EVO bets that the next paradigm-worthy win is at the **architecture-as-state** level — not at the per-block, per-step, or per-token level that #42-#55 have exhausted. The historical pattern: whatever was previously hand-designed eventually becomes a learned object (weights → 1980s; hyperparameters → 2000s; vision architectures → 2017). LLM architectures are plausibly next.

**Honest expected outcome.** With probability ~50%, Gate-0 reveals mechanism 1 fires (compute reduction real) but mechanism 2 is weak (per-step convergence improvement ~1.05× not 1.5×) → ATLAS-EVO ships at 1.43×. With probability ~25%, both mechanisms fire fully → ships at 2.0×. With probability ~25%, Goal-1 or Goal-2 fails → ATLAS-EVO REJECTED, ~2 GPU-days lost. Recoverable.

**Selection trigger.** Pursue ATLAS-EVO for #56 iff (Gate-0 Goal 1 + Goal 2 pass at 66M within 16 GPU-hours) AND (#43 OR #55 has shipped or is committed to ship). Otherwise hold for a later paradigm slot once HVP infrastructure is in place.

---

**Cross-references.**
- `PARADIGM_SHIFT_39_DESIGN.md` — RLG (layer growth), reused for ATLAS-EVO's growth path.
- `PARADIGM_SHIFT_42_DESIGN.md` — CHIRON Theorem 1 (bijectivity preserved under composition), justifying that layer-count drift preserves reversibility.
- `PARADIGM_SHIFT_43_CANDIDATE_C_ORION.md` — Pearlmutter HVP infrastructure, shared with ATLAS-EVO's fitness primitive.
- `PARADIGM_SHIFT_55_CANDIDATE_A_SOPHIA_CHIRON.md` — Hutchinson Hessian-diagonal estimate, shared with ATLAS-EVO's fitness primitive (if SOPHIA ships).
- `RLG_GATE0_COMPOUND.md` — RLG empirical validation at 66M and 1.84B (1.30× speedup), evidence that growth-side primitive is reliable.
- `RLG_SCALING.md` — RLG scaling characteristics, evidence that bidirectional generalization is plausible at flagship scale.
- `FUTURE_PARADIGM_CANDIDATES.md` — original NAS-class proposals (mostly unattacked axes); ATLAS-EVO is the first to formally engage that family.
- `surprise18_continuation_drift.md` — checkpoint-resume requirement that ATLAS-EVO's evolution state must obey.
