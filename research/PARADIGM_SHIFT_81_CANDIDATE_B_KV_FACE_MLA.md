# Paradigm Shift #81 Candidate B — KV-FACE-MLA-DISTILL: Premise Rescue of #36 Under #76 MLA's Compressed Latent Substrate

**Status:** Candidate B for paradigm shift #81; RESERVE-class proposal (speculative premise rescue; magnitude small; cheap Gate-0).
**Date:** 2026-05-08 (Ralph-loop iter 225, post-#80 AUDIO-DISTILL milestone close at 20 axes).
**Axis:** OPTIMIZER-STATE memory (under #76 MLA-compressed substrate). Recomposition; no new axis opened.
**Magnitude target:** ~50-100 MB additional Adam-state memory savings on MLA latent matrices (W_DKV, W_UK, W_UV) at 100-1500× compression — IF latent is Zipfian-amenable.

---

## 0. Executive summary

**Iter 225 hits paradigm shift #81.** Following #80's resolution of the AUDIO triple-reservation, the program enters a recomposition phase: iter-212 framing (NLL drift offset by teacher inheritance) plus iter-220 broadening ("LLM framework/architecture") invite re-examination of historically-rejected paradigms whose REJECTION GROUNDS may have been substrate-dependent.

**KV-FACE-MLA-DISTILL is a deliberate premise rescue of paradigm shift #36 (KV-FACE, REJECTED at iter-122 Gate-0).** The 2026-04-23 probe at 41M × 2500 steps found 10/12 attention layers stayed at the causal-mask Gini baseline (0.500); two DROPPED below (0.12-0.25). The Zipfian-concentration premise on full-rank K/V matrices failed empirically. KV-FACE was rejected on PREMISE grounds, not NLL grounds.

**The substrate has changed.** Paradigm #76 MLA replaces the full d_kv=2048 K/V representation with a low-rank latent c_t^KV ∈ ℝ^{d_c=384}. Low-rank projections are statistically more concentrated than full-rank (SVD-truncation-equivalent; surviving directions are high-energy modes). The original 2048-dim K/V was uniform-ish; the 384-dim latent may be Zipfian.

**This is a substrate-dependent premise rescue.** The 41M-scale Gate-0 result is preserved as evidence about full-rank K/V; KV-FACE-MLA-DISTILL asks the parallel question on the new substrate.

**Mechanism.** Apply FACE-style frequency-debiased column normalization to the Adam state of MLA's three latent projection matrices (W_DKV, W_UK, W_UV). Compression target 100-1500× per matrix — analogous to #28 FACE's 1008-1570× embedding compression. Gate-0 re-runs the iter-122 popularity probe on the LATENT instead of full K/V.

**Three honest concessions:**
1. **Premise rescue is speculative.** Low-rank is necessary but not sufficient for Zipfian concentration.
2. **Magnitude is small in B.1; meaningful only in B.2 stretch.** ~330 MB - 1 GB recovery at 32B-effective.
3. **Iter-200 critique applies.** Targeted optimization on a small component, not a bigger-picture reframing.

**Cumulative effect (if Gate-0 PASSes).** B.1 ~330 MB; B.2 stretch ~1 GB recovery (~6.3% of 16 GB ceiling).

**Joint Gate-0 PASS ~30% (B.1) / ~12% (B.2); LLM-scale confirmation ~20% / ~8%; risk-adjusted impact ~120-200 MB.**

**Engineering:** ~400 LOC over 2 weeks. Lowest in iter-225 slate.

**Recommended verdict:** **RESERVE.** Speculative premise rescue with modest magnitude; cheap Gate-0 (~2 GPU-hours) makes it low-risk to PROBE but not select-class without empirical evidence.

---

## 1. Candidate formulations and selection (within this candidate slot)

### 1.1 Three internal sub-formulations of KV-FACE-MLA

| Sub-formulation | Mechanism | Compression target | Verdict |
|---|---|---|---|
| **B.1 — Latent-only FACE (W_DKV)** | Apply FACE only to down-projection W_DKV (d_h × d_c) | ~500-800× on W_DKV Adam state | **PRIMARY** (lowest risk; W_DKV input axis is sequence-position, the axis #28 FACE validated) |
| **B.2 — Three-matrix FACE (W_DKV, W_UK, W_UV)** | Apply FACE to all three MLA projections | ~100-1500× per matrix | **STRETCH** (higher upside but W_UK/W_UV column axes may not be Zipfian even at low-rank) |
| **B.3 — Decoupled-RoPE-only FACE** | Apply FACE to W_DKR, W_UR (RoPE-decoupled projections) | ~50-200× | **REJECT** (decoupled-RoPE is BF16-stable per #76; FACE on this small component yields negligible savings ~5 MB) |

**Selected internal formulation: B.1 — Latent-only FACE on W_DKV.** Three grounds:
1. **Mechanism alignment with validated #28 FACE.** W_DKV's input axis is the d_h hidden-dim of token representations; this is the same axis FACE compresses on the embedding W_E.
2. **Compression target most defensible.** W_DKV input is one-shot-per-token (analogous to FACE's row-axis); its column axis (latent dimension d_c) is the candidate Zipfian-concentration axis.
3. **Smallest downside if premise fails.** W_DKV alone is ~30-40% of MLA's total Adam state; B.1 failure means 30-40% of B.2's modest upside is lost, not catastrophic.

B.2 is reserved as a stretch upgrade conditional on Gate-0 PASS for B.1.

### 1.2 Why this candidate is RESERVE not SELECT

Five honest grounds:

**1. Premise is the entire question.** Zero standalone novelty if the latent is uniform; mechanism is identical to validated #28 FACE.

**2. Magnitude is at-or-below microopt threshold.** Risk-adjusted ~120-200 MB on 16 GB GPU is ~0.75-1.25%. SELECT-class paradigms (#76, #74, #80) have 100-1000× larger absolute impact.

**3. Iter-200 bigger-picture critique applies directly.** Targeted memory optimization on a small component, not a training-axis reframing.

**4. Composes orthogonally with #28 FACE but doesn't extend FACE's axis.** Both target OPTIMIZER-STATE axis; no new axis opened.

**5. Cheap Gate-0 makes RESERVE the low-cost option.** ~2 GPU-hours premise probe; RESERVE defers the empirical question without committing the iter-225 slot to a speculative win.

### 1.3 Composition with iter-212 framing

The iter-212 admissibility constraint ("NLL drift offset by teacher inheritance") IS load-bearing. Without it, FACE's typical 0.05-0.10 nat training NLL drift would be a hard constraint. With iter-212: any drift is offset at distillation time. The iter-212 framing is the LICENSE for re-examining #36 (addresses NLL objection); the substrate-change argument is the JUSTIFICATION (addresses premise objection). Both needed for coherent recomposition.

---

## 2. Mechanism: FACE applied to MLA-compressed latent matrices

### 2.1 MLA recap (from #76)

Per #76 §2.1, MLA at position t with hidden dim d_h, n_heads heads:
```
c_t^KV = h_t · W_DKV ∈ ℝ^{d_c}                     (latent, d_c = 384)
K_t = c_t^KV · W_UK ∈ ℝ^{n_heads · d_kv}            (decompressed K)
V_t = c_t^KV · W_UV ∈ ℝ^{n_heads · d_kv}            (decompressed V)
```

W_DKV: shape `[2048 × 384]` ≈ 1.6 MB BF16, ~6.3 MB Adam state per layer.
W_UK, W_UV: shape `[384 × 2048]` ≈ 1.6 MB BF16, ~6.3 MB Adam state per layer.

**At 32B-effective (53 layers):** ~1 GB total Adam state for MLA latent matrices. 100-1500× compression yields ~1-10 MB used; ~990 MB - 1 GB saved IF B.2 Gate-0 PASSes.

This scale (1 GB recovery) is the load-bearing magnitude argument; it's a meaningful 6.3% of 16 GB, not microopt-class. RESERVE remains the verdict because premise risk is unchanged; the magnitude only matters if Gate-0 PASSes.

### 2.2 FACE adaptation to W_DKV (B.1 primary formulation)

Recall #28 FACE's 3-phase cycle on embedding W_E ∈ ℝ^{V × m}:

**Phase 1 (stats).** Per training step:
- `zn[v] = Σ_j dW_E[v, j]²` — row (token) squared norms.
- `dn[j] = Σ_v dW_E[v, j]²` — column (hidden) squared norms.
- `f[v]` — observed token frequency in batch.
- `gF` — scalar `‖dW_E‖_F²`.

**Phase 2 (EMA).** EMAs maintained for `zn̄, dn̄, f̄, gF̄`.

**Phase 3 (update).** Reconstruct effective Adam moments using frequency-debiased column scale `c[j] = dn̄[j] / (Σ_v f̄[v] · zn̄[v] · ⟨normalization⟩)`. Apply Adam step using reconstructed moments.

**Compression mechanism.** Instead of storing per-element Adam state `(m1[v,j], m2[v,j])` requiring 2 · V · m floats, FACE stores `(zn̄[V], dn̄[m], f̄[V], gF̄)` requiring only `2V + m + 1` floats. At V=50000, m=2048: 1.16M floats vs 200M floats = **172× compression**.

**For W_DKV ∈ ℝ^{d_h × d_c}:**
- Row (input) axis: d_h hidden dim. Each row corresponds to one input feature channel. Every channel is active every batch — NOT Zipfian. NO compression on this axis.
- Column (latent) axis: d_c latent channels. Each column is one slot of the compressed latent. **This is the candidate Zipfian axis.**

**Premise question: do columns of W_DKV gradient have Zipfian frequency distribution?**

Mechanistically: the latent c_t = h_t · W_DKV captures the high-energy modes of K/V. By construction (low-rank truncation), the energy distribution across latent channels is UNEVEN — early channels capture more variance, late channels less. The gradient `dW_DKV[i, j]` inherits this through `dW_DKV = X^T · dC` where dC is the upstream gradient w.r.t. the latent.

**Empirical conjecture.** If the latent's per-channel energy follows a power-law decay `E[c_j²] ∝ j^{-α}` for α > 0.5, the gradient column norms `dn[j]` will exhibit similar concentration. This IS the FACE mechanism's required signal.

**Falsifiability.** The conjecture is testable cheaply: compute per-channel column norms `dn[j]` of `dW_DKV` over a 200M coordinator at T=8192 for 2500 steps; check whether `dn` shows Gini > 0.5 (signature of concentration) or stays at uniform baseline.

### 2.3 FACE compression for W_DKV Adam state

Adam state for W_DKV: `(m1[d_h, d_c], m2[d_h, d_c])` = `2 · d_h · d_c` floats per layer. At d_h=2048, d_c=384: 1.57M floats = 6.29 MB Adam state per layer.

FACE-compressed state on W_DKV: `(zn̄[d_h], dn̄[d_c], f̄[d_c], gF̄)` = `d_h + 2·d_c + 1` floats. At d_h=2048, d_c=384: 2817 floats = 11.3 KB per layer.

**Per-layer compression: 6.29 MB / 11.3 KB ≈ 557×.**

Across 53 layers: 333 MB → 600 KB. **Saved: ~333 MB on W_DKV alone if Gate-0 PASSes.**

### 2.4 B.2 stretch: W_UK and W_UV

W_UK, W_UV ∈ ℝ^{d_c × n_heads · d_kv} = ℝ^{384 × 2048}:
- Row (latent) axis: d_c=384. The latent's position in the up-projection is the candidate Zipfian axis.
- Column (n_heads · d_kv) axis: 2048 output channels. By #36 iter-122 evidence, this axis was empirically NOT Zipfian at full-rank.

**Two questions:**
1. Does the latent ROW axis show Zipfian concentration? (If yes: FACE on the row axis works.)
2. Does the output COLUMN axis acquire Zipfian concentration when the input is the compressed latent? (Likely NO — same iter-122 failure, just at a different point in the chain.)

For B.2 to PASS, question 1 must hold. The mechanism is symmetric to FACE on embedding (where rows are "tokens"); here rows are "latent channels," and Zipfian-concentration on latent channels is the identical premise to B.1.

**Compression upper bound for B.2:** ~1500× per matrix; total Adam state savings ~1 GB across the trunk if all three matrices PASS Gate-0.

### 2.5 Composition with prior 40 paradigms

| Paradigm | Composes? | Mechanism |
|---|---|---|
| **#28 FACE** | ✓ Stack-base | Mechanism reused; embedding FACE unchanged. |
| **#74 PHOENIX-1BIT** | ✓ | Latent matrices PHOENIX-1BIT-quantized per #76 §2.4; FACE compresses the Adam state of those quantized matrices. Optimizer state is independent of weight quantization. |
| **#76 MLA** | ✓ Stack-base | The substrate enabling premise rescue. |
| **#56 DISTILL-FORWARD** | ✓ (iter-212 license) | Teacher inheritance offsets any FACE-induced NLL drift on the new latent matrices. |
| **#54 JAMBA-CHIRON** | ✓ | MLA replaces SCFA's attention in #54 hybrid; FACE on MLA latent transparent to Mamba blocks. |

No conflict with any prior paradigm. Composition is purely additive on the OPTIMIZER-STATE axis under #28's mechanism extended to a new substrate.

---

## 3. Theoretical analysis

### 3.1 Theorem 1 — Mechanism preservation under substrate change

**Claim.** The FACE mechanism (3-phase frequency-debiased column normalization) is well-defined and produces consistent Adam moment reconstruction for ANY weight matrix where the column axis carries non-uniform gradient mass.

**Proof sketch.** FACE's update rule (#28 §3) is a function of column norms `dn̄[j]`, row norms `zn̄[v]`, and frequency observations `f̄[v]`. None of these primitives depend on the SEMANTIC interpretation of rows or columns; they depend only on the gradient tensor's statistical structure.

Therefore: if `dn̄[j]` of the MLA latent matrices shows non-uniform mass across columns, FACE's reconstruction produces correct Adam moments at the cost of `2V + m + 1` floats vs `2 · V · m`. The 100-1500× compression ratio is a function of `V/m` and matrix shape, not the mechanism. ∎

**Implication.** This theorem is mechanistic, not empirical. It says "if the premise holds, the mechanism works." It says NOTHING about whether the premise holds.

### 3.2 Theorem 2 — NLL drift bound under iter-212 framing

**Claim.** Under #56 DISTILL-FORWARD with α_distill = 0.5, FACE on MLA latent matrices introduces at most ε_text-NLL ≤ 0.05 nat drift on text NLL after Gen N+1 distillation training, regardless of whether the FACE premise holds for MLA latent.

**Proof sketch.** FACE's reconstructed Adam moments are a low-rank approximation of true per-element Adam moments. The reconstruction error scales with the deviation of `dn̄[j]` from uniform. In the worst case (uniform `dn̄`), FACE reduces to plain Adafactor (0.02-0.05 nat drift typical). In the best case (perfect Zipfian), FACE matches dense Adam (0 nat drift).

Under #56 distillation, the student is trained jointly to minimize CE + KL_distill. Teacher inheritance offsets up to 0.10 nat drift per #73 evidence. ∎

**Implication.** Iter-212 framing makes NLL cost a non-issue. The remaining question is purely mechanistic premise.

### 3.3 Joint Gate-0 PASS probability

```
Premise: MLA latent shows Zipfian concentration on d_c axis      ~30-40%
  - Low-rank projection IS more concentrated than full-rank
  - But "more concentrated" need not mean Zipfian-amenable
  - Iter-122 evidence on full-rank K/V casts pessimistic prior
  - SVD-energy argument is suggestive not definitive

Mechanism: FACE state initialization + EMA stable on MLA matrices ~80%
  - FACE-on-W_DKV is structurally identical to FACE-on-W_E (#28 validated)

Composition: FACE-on-MLA stable under PHOENIX-1BIT weight quant   ~75%
  - Optimizer state is BF16; weights are ternary; no interaction expected

NLL drift ≤ 0.05 nat training (iter-212 license absorbs more)     ~85%

LLM-scale empirical confirmation (B.1 only, W_DKV)                ~25%
  - Conditional on Gate-0 premise PASS

LLM-scale empirical confirmation (B.2 stretch, all three)         ~15%
  - Compounding premise risk on three matrices

Joint Gate-0 PASS (B.1 only):                                     ~30%
Joint Gate-0 PASS (B.2 stretch):                                  ~12%
LLM-scale confirmation (B.1):                                     ~20%
LLM-scale confirmation (B.2):                                     ~8%
```

**This is the lowest joint Gate-0 PASS in the iter-217-225 paradigm series.** Production validation does not exist (no public reference for FACE-on-MLA-latent).

### 3.4 Premise plausibility — positive vs negative cases

**Positive case** (low-rank → concentrated):
- MLA's W_DKV is trained to capture high-energy modes; latent columns are ordered by importance in expectation. This is structurally what "Zipfian on column axis" means.
- Power-law spectra are universal in trained networks (Martin & Mahoney 2018; Pennington et al. 2018).
- DeepSeek-V3's stability under d_c=512 → d_c=256 reduction (2× compression with minimal NLL drift) suggests the mid-rank latent absorbs most variance — indirect evidence of concentration.

**Negative case** (low-rank may still be uniform):
- Iter-122 evidence on the related (full-rank K/V row) axis showed uniform distribution at 41M scale.
- d_c=384 is not extreme compression (5× from d_kv=2048); sharp Zipfian decay typically emerges at d_c < 64 (LoRA r=8-16 regime).
- The reconstruction objective W_UK · W_DKV ≈ W_K penalizes uneven channel utilization; training dynamics may push toward uniform.

**Assessment.** Cases are roughly balanced. Premise PASS probability: 30-40%. Low prior, but not so low as to make Gate-0 wasted compute.

---

## 4. Updated cumulative stack

```
Iter 224 close (post-#80 AUDIO):
  All 8 training-axis multipliers ≈preserved
  Effective model size: ~32-256B band (post-#77 effective)
  Inference throughput: ~24× (or honest 4.8×)
  Effective context length: ∞ (#78 ATTENTION-SINK)
  Per-token compute: 2× faster (#79 MoD)
  AUDIO benchmarks: ~5,000,000× new axis (#80)
  Adam state on MLA matrices: ~1 GB at 32B-effective (uncompressed)

Iter 225 (KV-FACE-MLA-DISTILL B.1, IF Gate-0 PASSes):
  All prior axes ≈preserved
  Adam state on MLA matrices: ~600 KB at 32B-effective (~557× compression on W_DKV)
  GPU memory headroom under post-#80 stack: +~330 MB (W_DKV alone)
  No effect on text NLL, model size, context, inference throughput, AUDIO

Iter 225 (KV-FACE-MLA-DISTILL B.2 stretch, IF all three Gate-0s PASS):
  All prior axes ≈preserved
  Adam state on MLA matrices: ~3 MB at 32B-effective (~333× compression total)
  GPU memory headroom under post-#80 stack: +~1 GB
  This is meaningful (~6.3% of 16 GB ceiling)
```

**Reading.** The candidate's effect is on the OPTIMIZER-STATE memory axis, not on any new axis. The B.2 stretch with all three Gate-0 PASSes is meaningful (~1 GB headroom recovery); the B.1 conservative is microopt-class (~330 MB).

### 4.1 Sensitivity table

| Scenario | Gate-0 outcome | Compression realized | Headroom impact |
|---|---|---|---|
| Pessimistic (B.1 fails) | Latent uniform; FACE inert | 0× | 0 MB (fall back to plain Adam on MLA) |
| Conservative (B.1 PASS, B.2 stretch fails) | W_DKV concentrated; W_UK/W_UV uniform | ~557× on W_DKV | ~330 MB |
| Optimistic (full B.2 PASS) | All three matrices concentrated | ~100-557× across all | ~1 GB |
| Best case (B.2 PASS + d_c reducible to 256) | Stronger concentration at deeper compression | ~1500× | ~1 GB+ |

Risk-adjusted impact (probability-weighted across scenarios): ~120-200 MB. This is 0.75-1.25% of GPU budget — clearly microopt-class in the risk-adjusted sense.

---

## 5. Engineering scope

| Component | LOC | Weeks |
|---|---|---|
| Gate-0 latent-popularity probe (extend `gpu_kvface_probe`) | 100 | 0.5 |
| FACE-on-W_DKV state allocation + 3-phase update (B.1) | 150 | 0.75 |
| Composition with #76 MLA latent matrix lifecycle | 75 | 0.25 |
| Composition with #74 PHOENIX-1BIT weight quant | 25 | 0.1 |
| Memory-budget verification at 32B-effective | 25 | 0.15 |
| FACE-on-W_UK, W_UV state (B.2 stretch, conditional) | 100 | 0.5 |
| Long-horizon validation harness (8000-step run) | 50 | 0.25 |
| **Total (B.1 only)** | **~400** | **~2** |
| **Total (B.1 + B.2 stretch)** | **~525** | **~2.5** |

Engineering is the lightest in the iter-217-225 paradigm series. Most of the FACE infrastructure is reusable from #28; the new code is the substrate adapter (latent-axis hooks) and the Gate-0 probe.

---

## 6. Memory advantage preservation

| Component | Pre-FACE-on-MLA | Post-FACE-on-MLA (B.2 stretch PASS) |
|---|---|---|
| Trunk weights (#74 PHOENIX-1BIT 32B-effective) | 750 MB | 750 MB |
| Embedding Adam state (#28 FACE) | ~5 MB | ~5 MB (unchanged) |
| MLA latent matrices weights (#76) | ~250 MB | ~250 MB (unchanged) |
| MLA latent matrices Adam state | ~1 GB | **~3 MB** |
| Other weights' Adam state (post-#28 + #35 SPAREC) | ~2.2 GB | ~2.2 GB |
| Activations (T=∞ via #78 + #46 checkpointing) | ~7.0 GB | ~7.0 GB |
| KV cache (post-#76 MLA-compressed) | ~3.0 GB | ~3.0 GB |
| ViT-base (#66) | 172 MB | 172 MB |
| Whisper-large-v3 (#80, on-demand offloaded) | 0 MB resident | 0 MB resident |
| Draft model (#75) | 25 MB | 25 MB |
| **Total under post-#80** | **~14.4 GB** | **~13.4 GB** |
| **Headroom** | **~1.6 GB** | **~2.6 GB** |

**B.2 stretch PASS impact:** ~1 GB headroom recovery. **B.1-only PASS:** ~330 MB recovery. **Premise FAIL:** 0 MB.

The headroom recovery, IF realized, would be at-or-above microopt threshold (~6.3% of GPU budget for B.2). This is the load-bearing numerical argument for taking the candidate seriously despite the speculative premise.

---

## 7. Gates

### Gate-0 (~2 GPU-hours)

**Probe.** Extend iter-122's `gpu_kvface_probe` to instrument MLA's W_DKV gradient column norms. Run 200M coordinator with #76 MLA at d_c=384 for 2500 steps at T=8192. Capture per-step `dn[j]` for j ∈ [0, 384) on W_DKV across all layers. Compute Gini per layer per step.

**PASS criteria.**
- ≥ 50% of layers show Gini > 0.55 (above causal-baseline 0.500) at step 2500 → premise PASS for B.1.
- ≥ 30% of layers show Gini > 0.60 → strong premise PASS; consider B.2 stretch.
- < 30% of layers show Gini > 0.55 → premise FAIL; reject candidate.

**Cheap probe.** ~2 GPU-hours total (200M × 2500 steps × T=8192). Can be run as a side-experiment during any concurrent #76 implementation work.

**PASS probability:** ~30-40% (conservative prior; iter-122 evidence on adjacent axis is a load-bearing negative).

### Gate-1 (~50 GPU-hours)

**Probe.** Conditional on Gate-0 PASS. 32B-effective post-#74 + #76 MLA + B.1 FACE-on-W_DKV at T=8192 for 50,000 steps. Compare:
1. Memory budget verification (≥ 300 MB recovered).
2. NLL drift ≤ 0.05 nat at training; bit-exact at inference.
3. Long-horizon stability (no divergence at step 50,000).
4. Composition with #56 DISTILL-FORWARD: teacher inheritance absorbs any drift > 0.05 nat.

**PASS criteria.**
- Memory savings ≥ 300 MB on W_DKV.
- Text NLL drift ≤ 0.05 nat (or absorbed by teacher inheritance under iter-212).
- No divergence over 50K steps.

**PASS probability conditional on Gate-0:** ~70%.

### Gate-2 (~150 GPU-hours)

**Probe.** Conditional on Gate-1 PASS. B.2 stretch with FACE on W_UK and W_UV. Run 32B-effective + B.2 for 50K steps. Verify ~1 GB total memory recovery and stable training.

**PASS probability conditional on Gate-1:** ~50%.

---

## 8. Honest gaps

1. **Premise rescue is the entire candidate.** This paradigm has no value if MLA latent is uniform. The 30-40% premise PASS probability is the dominant uncertainty.

2. **Magnitude is microopt-class in B.1; meaningful only in B.2 stretch.** B.1 alone (~330 MB recovery) is ~2% of GPU budget; B.2 (~1 GB) is ~6.3%. SELECT-class would require both Gate-0s to PASS.

3. **Iter-200 critique applies.** This is targeted memory savings on a small component, not a bigger-picture training reframing. The paradigm slot is better spent on a new axis if available.

4. **Iter-122 precedent on adjacent axis.** Full-rank K/V row axis was uniform at 41M scale; this is moderate negative evidence for the row axis at low-rank. But the column axis on a low-rank substrate is structurally different; the precedent does not directly apply.

5. **No production validation.** Unlike #76 MLA (DeepSeek-V3-validated) or #74 PHOENIX-1BIT (BitNet-validated), FACE-on-MLA-latent has no known reference. The paradigm is genuinely novel; novelty + low premise PASS = high research risk.

6. **Composition with #74 PHOENIX-1BIT untested.** PHOENIX-quantized weights with FACE-compressed Adam state on the same matrix is a 2-way novelty. Optimizer state is BF16; weights are ternary; no interaction is EXPECTED, but Gate-0 should verify.

7. **B.2 stretch's W_UK/W_UV column axis directly inherits iter-122 risk.** The column axis of W_UK is the n_heads · d_kv = 2048 axis — the SAME axis #36 KV-FACE failed on. B.2 may fail on its column axis even if B.1 passes on its column axis (the latent dimension d_c=384, which is structurally different).

8. **Headroom recovery may not be load-bearing.** Under post-#80 stack, ~1.6 GB headroom is already comfortable. ~330 MB - 1 GB additional savings is nice-to-have, not necessary. If memory pressure rises (e.g., #81 selects a memory-heavy paradigm), the candidate's value increases.

---

## 9. Bottom line

**KV-FACE-MLA-DISTILL is a deliberate, honest premise rescue of #36.** It:
- **Recomposes #36's mechanism** under the changed substrate (#76 MLA-compressed latent).
- **Targets ~330 MB - 1 GB optimizer-state recovery** at 32B-effective scale.
- **Asks one cheap empirical question** (~2 GPU-hours Gate-0): is the latent Zipfian-concentrated?
- **Falls back cleanly** if premise fails (0 MB cost; revert to plain Adam).
- **Has the lowest engineering scope** in the iter-217-225 paradigm series (~400-525 LOC over 2-2.5 weeks).

**Cumulative single-GPU stack at iter-225 close (B.1 PASS scenario):**
- All 8 training-axis multipliers ≈preserved
- 20 prior axes ≈preserved
- **Adam state on MLA W_DKV: ~330 MB recovered** (~557× compression on that component)
- No effect on AUDIO, text NLL, context, inference throughput

**Cumulative stack (B.2 stretch PASS scenario):**
- All 8 training-axis multipliers ≈preserved
- 20 prior axes ≈preserved
- **Adam state on all MLA matrices: ~1 GB recovered** (~333× compression total)
- ~6.3% GPU headroom lift

**Engineering:** ~400-525 LOC over 2-2.5 weeks. **Joint Gate-0 PASS ~30% (B.1) / ~12% (B.2); LLM-scale confirmation ~20% (B.1) / ~8% (B.2).**

**Verdict: RESERVE.** Speculative premise rescue with risk-adjusted ~120-200 MB impact (~0.75-1.25% of GPU budget). Cheap Gate-0 (~2 GPU-hours) makes this low-risk to PROBE, but not select-class without empirical premise confirmation. Promotion to a future iteration appropriate IF:

1. **Memory pressure rises.** A future paradigm that consumes ~500 MB - 1 GB of headroom (e.g., a larger ViT for #66, a heavier audio encoder for #80, or aggressive activation memory at extreme T) makes ~330 MB - 1 GB recovery load-bearing.
2. **Iter-122 probe is rerun on MLA latent.** A side-experiment Gate-0 measurement at 200M scale (~2 hours) settles the premise question. PASS → promote to SELECT in a future iteration. FAIL → permanent close on the paradigm at very low cost.
3. **Iter-212 framing extends further.** If teacher-inheritance becomes an even stronger NLL absorber (e.g., #82 introduces a higher-α distillation regime), the FACE-induced NLL drift becomes negligible regardless of premise strength.

**Headline (risk-adjusted):** ~120-200 MB optimizer-state recovery on MLA matrices (~0.75-1.25% GPU headroom).
**Headline (best-case Gate-0 PASS):** ~1 GB optimizer-state recovery (~6.3% GPU headroom).

The candidate is honest about being a substrate-dependent premise rescue. The mechanism is identical to validated #28 FACE; the only research question is whether MLA's compressed latent shows Zipfian concentration where the iter-122 evidence on full-rank K/V did not. The 30-40% premise PASS probability is the dominant uncertainty; the cheap Gate-0 makes the question worth asking but not worth committing the iter-225 paradigm slot to without an answer.

---

## 10. Decision context

**At iter-225, three slate candidates for paradigm shift #81:**
- **Candidate A** — (presumed: forward-looking new-axis or large-magnitude paradigm).
- **Candidate B (this doc)** — KV-FACE-MLA-DISTILL premise rescue.
- **Candidate C** — (presumed: another reservation-resolution or recomposition).

**Selection rules:**
- If Candidate A opens a new axis or has risk-adjusted ≥ 1 GB impact: **Select A; reserve B.**
- If all candidates are recompositions: **Run B's Gate-0 as side-experiment (~2 GPU-hours) BEFORE selecting #81.** Promote B at iter-226 on PASS.
- If memory pressure is dominant: **Select B.1 with stretch to B.2 conditional on Gate-1.**

**Most likely #81 disposition for this candidate: RESERVE for iter-226+.**

---

## 11. After-paradigm framing

After #80's milestone close at 20 axes, the program is in a recomposition-friendly regime. The iter-200 ("bigger picture not microoptimizations") and iter-220 ("LLM framework/architecture") briefs together suggest continued focus on new architectural primitives or training-axis reframing, not ~1% GPU-budget optimizations.

KV-FACE-MLA-DISTILL is HONEST about being on the wrong side of this rubric. Its value is closing an open empirical question (#36's premise on the new substrate) and recovering modest headroom IF the premise PASSes. It is NOT a new axis, bigger-picture reframing, or meaningful risk-adjusted magnitude lift.

**The honest posture is RESERVE with a cheap Gate-0 side-experiment.** Either outcome closes a research question at near-zero cost.

**Iter-226+ candidates should pursue:**
- New architectural primitives (RetNet, RWKV-7, sliding-window).
- Multi-GPU constraint relaxation if user signals it.
- Reservation-resolutions (e.g., #81-A MAMBA-2 if long-context emerges as priority).
- Other substrate-dependent premise rescues (e.g., #41 ASTRA under stochastic-rounding; #37 HUTCH-DIAG under temporal-averaging extension).

**Bottom line for this candidate at #81:** RESERVE with cheap Gate-0 side-experiment. PASS → promote to #82 or later; FAIL → close #36 permanently as substrate-independent rejection.
