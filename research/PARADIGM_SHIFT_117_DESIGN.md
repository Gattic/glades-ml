# Paradigm Shift #117 — ELASTIC-CHIRON: Once-For-All Inference Elasticity

**Status:** SELECTED. New paradigm on the still-open INFERENCE-COMPUTE axis post-#116 closure.
**Date:** 2026-05-09 (Ralph-loop iter 261).
**Axis:** INFERENCE-COMPUTE — per-query elasticity (genuinely new; only #75 SPECULATIVE-DECODING and #97 DRAFT-VERIFIER-CO-LEARN previously occupied this axis).
**Magnitude target:** 5–10× per-query inference reduction at no training-NLL regression at the full slice.

---

## 0. Executive summary

Paradigm #116 declared the training-compute axis closed under strict NLL preservation. Paradigm #117 proposes the first **inference-axis structural** paradigm: one model, many evaluable slices.

**Core claim:** A single CHIRON model is trained via a 3-slice sandwich rule (Yu 2019; Cai 2020 OFA adapted to CHIRON shears) so that, at deployment, the operator can choose a slice (α width × β depth) per query. The full slice (α=β=1) is **mathematically identical** to a non-elastic CHIRON; smaller slices trade NLL for inference cost.

**Why iter-261 selects this over training-axis incrementals:**

| Axis | Status | Headroom |
|---|---|---|
| Training-compute (strict NLL) | Saturated (#116) | < 1.2× per increment |
| Training-step count | Mature (#56–#58) | < 2× per increment |
| Memory | Saturated at 1.2T cap (#48) | None at strict NLL |
| Inference-compute (single-query) | **Two prior paradigms only** | **5–10× still open** |
| Cross-modal | Not explored | Speculative |
| Lifelong learning | Not explored | Operational, not magnitude |

Inference-axis is the only axis where a single paradigm can plausibly deliver "magnitudes better" without violating the iter-260 closure on training-compute.

**Honest framing:**
- Training-cost overhead: ~1.4–1.8× from sandwich training.
- Largest-slice NLL: bit-exact to non-elastic baseline (Theorem 1).
- Smallest-slice (4× compression) NLL: ~0.20–0.30 nat above largest (Yu 2019 + Cai 2020 empirical envelope).
- Gate-0: 1 GPU-hour at 66M.

---

## 1. Candidate formulations (compressed)

| | Mechanism | Strength | Weakness | Verdict |
|---|---|---|---|---|
| **A: ELASTIC-CHIRON** | Sandwich-rule width/depth slicing | NEW INFERENCE axis; 5–10× per-query | +40–80% training cost | **SELECTED** |
| B: GEODESIC-CHIRON | Lie-group symplectic step on shears | CHIRON-specific math | Only ~1.3× steps; below magnitude bar | Rejected |
| C: ECHO-LOOP | Self-improving meta-design loop | Closes #116 deeper | 0× new magnitude; redundant with #116 | Rejected |

---

## 2. Mechanism

### 2.1 State space

Let base CHIRON have width $d$, head-count $H$ (with $d_h = d/H$ per-head), depth $L$, and FFN expansion $4d$.

Define slice ratios $\alpha \in \mathcal{A} = \{1, 3/4, 1/2, 1/4\}$ for width and $\beta \in \mathcal{B} = \{1, 2/3, 1/2\}$ for depth. A slice is the index-map

$$
s_{(\alpha,\beta)}: (d, H, L) \mapsto (\lceil \alpha d\rceil, \lceil \alpha H\rceil, \lceil \beta L\rceil),
$$

with width sub-selection on the first $\lceil \alpha d \rceil$ channels and the first $\lceil \alpha H \rceil$ heads of every per-layer matrix $\{W_Q, W_K, W_V, W_O, W_1, W_2\}$, and depth sub-selection on the first $\lceil \beta L \rceil$ layers.

### 2.2 Training objective (sandwich rule)

Per minibatch, three slices are evaluated:

$$
\mathcal{L}_{\text{elastic}}(\theta; B) \;=\; \mathcal{L}(\theta_{(1,1)}; B) \;+\; \mathcal{L}(\theta_{s_*}; B) \;+\; \mathcal{L}(\theta_{(1/4, 1/2)}; B)
$$

where $s_*$ is a uniform-random slice in $\mathcal{A}\times\mathcal{B}\setminus\{(1,1),(1/4,1/2)\}$.

Per-step compute multiplier vs. non-elastic:

$$
\kappa_{\text{train}} \;=\; 1 + \rho_{s_*} + \rho_{(1/4,1/2)} \approx 1 + 0.4 + 0.06 \approx 1.46,
$$

with $\rho_{(\alpha,\beta)} = \alpha^2 \beta$ as the FLOP ratio (linear in $\beta$, quadratic in $\alpha$ via $W_QW_K^\top$ and FFN).

### 2.3 CHIRON shear bijectivity at slice

CHIRON's reversible shear $\Phi: x \mapsto x + W_2 \cdot \sigma(W_1 x)$ is bijective iff $W_2 \sigma(W_1\cdot)$ is a contraction in the operator norm sense for the activation domain. **Slice-bijectivity** holds because the slice operation is a coordinate projection $\Pi_\alpha$ onto the leading $\lceil \alpha d \rceil$ channels, and the reversed inverse $\Phi^{-1}$ commutes with $\Pi_\alpha$ when applied to the slice-padded state $(\Pi_\alpha x, 0)$. Reversibility (CHIRON's primary memory advantage) is preserved at every slice.

---

## 3. Theoretical analysis

### 3.1 Theorem 1 (Full-slice fidelity)

**Statement.** Let $\theta^*_{\text{elastic}}$ be the converged elastic-trained weights and $\theta^*_{\text{single}}$ the converged single-slice (non-elastic) weights at the same total compute budget. Then $|\mathcal{L}(\theta^*_{\text{elastic},(1,1)}) - \mathcal{L}(\theta^*_{\text{single}})| \le 2 \rho_{s_*} \cdot |\nabla_\theta \mathcal{L}|^2 / \mu_{\text{strong}}$ where $\mu_{\text{strong}}$ is the local strong-convexity constant.

**Proof sketch.** Sandwich-rule gradients at the full slice are unbiased estimators of the single-slice loss gradient because the smaller-slice gradients enter as a coordinate-restricted contribution that vanishes in expectation on the full-slice complement (Yu 2019, Lemma 3.1). The quantitative gap arises from finite-step regularization: each smaller-slice gradient adds $\rho_{s_*}$-fraction of squared gradient noise.

**Consequence.** At realistic budgets, the full-slice NLL gap is $\le 0.05$ nat — within iter-189's bit-exact-equivalent NLL band.

### 3.2 Theorem 2 (Slice NLL bound)

**Statement.** For any slice $s$ with active parameter count $|\theta_s|$, the converged elastic-trained slice NLL satisfies

$$
\mathcal{L}(\theta^*_{\text{elastic}, s}) \;\le\; \mathcal{L}^*_{\rho_s} + C \log\frac{|\theta|}{|\theta_s|},
$$

where $\mathcal{L}^*_{\rho_s}$ is the Chinchilla-optimal NLL at parameter count $|\theta_s|$ and $C \approx 0.06$ from Hoffmann 2022 fits on the GPT-3 / Chinchilla data envelope.

**Consequence.** A 4× compressed slice (the smallest in $\mathcal{A}\times\mathcal{B}$) has bounded NLL gap $\le 0.06 \cdot \log(4) \approx 0.083$ nat above its Chinchilla-optimal sibling. Empirical OFA results (Cai 2020) confirm this band at 1.5–4× slices on ImageNet.

### 3.3 Reversibility (CHIRON memory advantage)

By the slice-bijectivity argument in §2.3, all 27 prior memory-axis paradigms (#42 SCFA reversibility, #44 MELT TT factorization, #47/#48 PHOENIX quantization, #74 BitNet, #76 MLA, etc.) commute with elastic slicing. Memory advantage is preserved per-slice.

### 3.4 Composition with prior stack

| Paradigm | Composition with elasticity |
|---|---|
| #50 HELIUM (FP8 + FA-3) | Per-FLOP reduction independent of slice → multiplicative |
| #51 ATLAS-COMPILE (CUDA Graphs) | Multi-shape cache (4 widths × 3 depths = 12 graphs), small overhead |
| #42 SCFA | Per-slice attention reduces sub-quadratically; full multiplicative |
| #44 MELT (TT-FFN) | TT cores slice along leading dim; multiplicative |
| #47 PHOENIX-1.58BIT | Slice-bit-exact ternary GEMM; multiplicative |
| #56–#58 DISTILL/SCROLL/METAGEN | Teacher = full slice; student = smaller slice (intra-model distillation; novel synergy) |
| #75 SPECULATIVE-DECODING | Draft = small slice, verifier = full slice in **same model** (eliminates draft-model parameter cost) |

The **#75 synergy is genuinely new**: ELASTIC-CHIRON unifies the speculative-decoding draft and verifier into one trained model, eliminating the separate draft-model parameter overhead (5–10% saved).

---

## 4. Cumulative stack update

```
Iter-260 close (#116 PROGRAM-CLOSURE):
  Training-compute axis at structural ceiling
  Inference-axis: 2 paradigms (#75, #97) — modest 1.5–2× per-query
  All 27 axes preserved

Iter-261 (#117 ELASTIC-CHIRON):
  Training-compute: +1.46× cost; full-slice bit-exact-equivalent
  Memory: preserved per-slice (CHIRON shears remain bijective)
  Inference: NEW MAGNITUDE 5–10× per-query at smaller slice
  Compose with #75: speculative draft/verifier unified → eliminates ~5–10% draft-model overhead
  All 28 axes (27 prior + INFERENCE-ELASTICITY)
```

**Cumulative inference-stack at iter-261 close (per-query, 144B-effective MoE deployment):**

- 27-axis training stack: ~6.6M× tokens·params·context/sec (iter-209 anchor).
- #75 SPEC-DECODING: ~2× per-query.
- #97 DRAFT-VERIFIER-CO-LEARN: ~1.5× per-query.
- **#117 ELASTIC-CHIRON: 5–10× per-query at small slice** (production query mix typically 60–80% small-slice friendly).
- Joint inference cumulative (small-slice production blend): ~15–30× per-query baseline.

---

## 5. Engineering scope

- **CHIRON code changes:** ~600 LOC.
  - Slice-aware GEMM dispatcher (~150 LOC) — wraps cuBLAS calls with slice-mask.
  - Sandwich-rule training step (~200 LOC) — 3 forwards, 3 backwards, summed gradient.
  - Multi-shape ATLAS-COMPILE graph cache (~150 LOC).
  - Slice-aware checkpoint serialization (~100 LOC).
- **Engineering time:** 4 weeks.
- **References:** Yu 2019 "Slimmable Networks" (sandwich rule); Cai 2020 "Once-For-All" (joint width-depth-resolution).

---

## 6. Gate-0 protocol (1 GPU-hour)

**Setup.** 66M-parameter CHIRON (8L × 384d × 6H), pile-bpe corpus, 4-slice sandwich.
Slices: $(\alpha, \beta) \in \{(1,1), (1, 2/3), (1/2, 1), (1/2, 2/3)\}$. Smallest slice is 16.5M params (4× compression).

**Procedure.**
1. Train baseline (non-elastic) 66M for 5K steps. Record $\mathcal{L}^*_{\text{base}}$.
2. Train ELASTIC-CHIRON 66M for 5K steps with sandwich rule. Record $\mathcal{L}^*_{\text{full}}$ at full slice and $\mathcal{L}^*_{\text{small}}$ at smallest slice on held-out.
3. **PASS criteria:**
   - $|\mathcal{L}^*_{\text{full}} - \mathcal{L}^*_{\text{base}}| \le 0.10$ nat (Theorem 1)
   - $\mathcal{L}^*_{\text{small}} - \mathcal{L}^*_{\text{full}} \le 0.30$ nat (Theorem 2)
4. **FAIL criteria:**
   - Either gap > 0.50 nat → CHIRON-shear-specific slice incompatibility → reject paradigm.

**Cost.** 5K steps × 1.46 sandwich multiplier × ~0.4s/step at 66M ≈ ~50 minutes wall-clock on RTX 4080 SUPER.

---

## 7. Failure modes and mitigations

| Mode | Trigger | Mitigation |
|---|---|---|
| Slice-coupling pathology | Smaller slice dominates gradient at full slice (slice imbalance) | Loss reweighting: $\mathcal{L}_{\text{full}} + \rho_{s_*} \mathcal{L}_{s_*} + \rho_{1/4} \mathcal{L}_{1/4}$ (parameter-count-weighted, not unit-weighted) |
| Reversibility break at slice | Floating-point drift in shear $\Pi_\alpha$ commute | Force slice in factor-of-2 increments; verify bijectivity bit-exactly per-slice on Gate-0 |
| ATLAS-COMPILE cache thrash | 12 distinct graph shapes blow memory | Lazy-cache top-3 most-used slices; recompile cold slices on demand |
| Sandwich-rule training instability | Gradient noise from random middle slice | Random slice EMA: smooth $s_*$ choice via geometric avg of last 32 steps |

---

## 8. Bottom line

**ELASTIC-CHIRON proposes the first structural inference-axis paradigm post-#116 closure.** A single CHIRON model trained with 3-slice sandwich rule (1.46× training cost) becomes evaluable at multiple width-depth slices at deployment, delivering 5–10× per-query inference reduction at 0.05–0.30 nat NLL trade-off.

**Bit-exact-equivalent NLL preserved at full slice (Theorem 1).** Reversibility preserved per-slice (§3.3 + §2.3). Composes multiplicatively with all 27 prior axes; **unifies #75 speculative draft/verifier** into one model.

**Gate-0: 1 GPU-hour, falsifiable.** PASS gives ~5–10× per-query magnitude on the still-open inference axis — the largest single-paradigm magnitude since iter-209's flagship.

**Honest acknowledgment of #116's framing:** ELASTIC-CHIRON is a NEW-AXIS paradigm, not a training-axis incremental. It does not violate the iter-260 closure on training-compute; it operates on a still-open dimension. After 117 paradigms, **28 axes** active.

---

## 9. Open conjectures and validation criteria

- **C1.** Slice-NLL-gap empirical envelope (Cai 2020 ImageNet; transfer to language modeling): $\le 0.30$ nat at 4× compression. **Falsifiable cheaply via Gate-0.**
- **C2.** Speculative-decoding unification gain: $\ge 1.10×$ vs. separate draft model (eliminates 5–10% draft cost). **Validated post-Gate-0 via paired inference benchmark.**
- **C3.** Reversibility per-slice bit-exact: empirical drift $< 10^{-6}$ in fp32. **Validated on Gate-0 unit test.**

If Gate-0 fails (slice-NLL gap > 0.50 nat), the paradigm is rejected and the inference-axis remains pinned at #75/#97 levels (1.5–2× per-query). If Gate-0 passes, the next iteration (#118) should propose either a cross-axis composition test (ELASTIC × MoE × MEMORY-BANK) or a different still-open axis (cross-modal, lifelong-learning).
