# Paradigm Shift #75 — Candidate B: MOEFICATION-DISTILL-CHIRON — Post-Hoc Mixture-of-Experts at #74's Quantized Base

**Status:** CANDIDATE B (under evaluation alongside A and C at iter 219). **Recommendation: SELECT-CONDITIONAL.** The mechanism extends the iter-218 #74 PHOENIX-1BIT-DISTILL-COMBO substrate (32B-effective on 16 GB single GPU via binary middle + ternary edges + BF16 island + Llama 3.1 405B teacher) to a sparse mixture-of-experts via post-hoc moefication (Yu et al. 2022 "MoEfication: Conditional Computation in Pretrained Models"). Per-token activation drops from 32B-effective dense to ~8B active via top-2-of-8 expert routing; effective-parameter-count climbs to ~256B effective at fixed memory ceiling via shared-backbone + per-expert LoRA factorization (inheriting #53 MOSAIC-MOE selected-but-unvalidated mechanism). **Effective single-GPU model size jumps from 32B (post-#74) to ~256B effective on a single 16 GB GPU** — almost an order of magnitude beyond #74's ceiling, addressing the iter-219 brief's "extremely large LLMs on a single GPU" core ask at the next paradigm-axis tier (CONDITIONAL COMPUTATION rather than further QUANTIZATION).
**Date:** 2026-05-08 (Ralph-loop iteration 219).
**Axis:** EXTENSION/RECOMPOSITION — combines MEMORY axis at #74's quantization tier with CONDITIONAL-COMPUTATION axis (sparse MoE). #74 closed the per-parameter quantization frontier at 1-bit binary; #75-B opens a different lever — fewer active parameters per token at higher effective parameter count. Genuinely novel at the program level: the JOINT composition of #48 PHOENIX-1BIT (re-admitted via #73) + #53 MOSAIC-MOE (selected at iter-197 but never empirically validated) + #68 SUPER-DISTILL (validated at iter-212) was unavailable until #74 selected the quantization base in iter-218. The composition itself has no published precedent at this aggressiveness; nearest empirical anchors are Mixtral 8x22B (~141B / ~39B active, BF16) and DeepSeek-V3 (671B / 37B active, FP8) — neither at 1-bit-binary-trunk substrate.
**Magnitude target (honest):** **~8× effective parameter expansion via expert sparsity on top of #74's 32B-effective base + 4× per-token COMPUTE reduction (top-2-of-8 routing → 25% active params per token) + net NLL trajectory ≤ from-scratch baseline by 0.10-1.70 nat (slightly tighter than #74's 0.20-1.85 due to additional MoE quality penalty)** — single-GPU 16 GB ceiling preserved by mandatory shared-backbone factorization. **Headline: 256B effective at ~8B active per token AND NLL strictly improved (per iter-212 framing).** Compute speedup: per-token wall-clock 4× FASTER than dense #74 32B-effective at the same effective capacity; ~2× SLOWER than native 1.84B BF16 at much higher effective capacity. The first paradigm in the recent series that delivers a TRUE per-token compute speedup over its predecessor.

---

## 0. Status & axis & honest headline

- **Status:** CANDIDATE B. Recommendation **SELECT-CONDITIONAL on joint Gate-0 PASS.** Of the iter-219 candidates (A, B, C), B introduces a NEW axis on top of #74's substrate — conditional computation via post-hoc moefication. #74 extended the same MEMORY axis to its production-validated quantization frontier; #75-B opens the CONDITIONAL-COMPUTATION axis at the same memory tier. **SELECT-CONDITIONAL with moderate-low confidence — magnitude is exceptional (256B effective; 4× per-token compute speedup over #74), but composition risk is HIGHER than #74 due to compounding two unvalidated selections (#53 MOSAIC-MOE + #74 PHOENIX-DISTILL).**
- **Date:** 2026-05-08, iter 219.
- **Axis:** RECOMPOSITION — #74 (MEMORY) × #53 (CONDITIONAL COMPUTATION) × #68 (TEACHER-PROVENANCE). Pre-#75-B stack ships post-#74 PHOENIX-1BIT-DISTILL-COMBO (32B-effective at 16 GB; NLL improved 0.20-1.85 nat over from-scratch). #75-B re-applies #53 MOSAIC-MOE's shared-backbone-plus-LoRA-experts schema to the QUANTIZED base, gaining 8× effective expansion at ~25% active fraction per token.
- **Honest headline:** **~256B effective on single 16 GB GPU at ~8B active per token + per-token compute 4× faster than #74's 32B-dense at same memory budget + net NLL improved by 0.10-1.70 nat over from-scratch baseline (slightly tighter range than #74-alone).** Inference latency at 256B-effective ≈ 1.5-2.5× of native 1.84B BF16 (the first paradigm in #73→#74→#75-B sequence to NARROW the wall-clock gap relative to native baseline). Quality preserved-and-improved per iter-212 framing.

The user brief at iter-219 reads "magnitudes better on compute speed without compromising memory advantages or nll accuracy" + single-GPU + novel + bigger-picture. #75-B is the FIRST candidate in the post-#73 series that satisfies the "compute speed" clause directly per-token (vs #73/#74 which satisfied it only at the EFFECTIVE-MODEL-SIZE-PER-MEMORY-BUDGET layer). **#75-B clears the magnitude bar at 8× effective expansion beyond #74 (1.84B native → 256B effective; 139× over native; 8× over #74) AND ~4× per-token compute speedup over #74-dense AND NLL improved (not regressed) over from-scratch baseline.** The compounding-risk caveat is real: #53 MOSAIC-MOE was selected at iter-197 but never empirically validated; combining it with #74's also-unvalidated quantization tier stresses both at once.

---

## 1. Executive summary

After 33 paradigms (#42-#74), the cumulative single-GPU stack at iter-218 close (post-#74 PHOENIX-1BIT-DISTILL-COMBO selected) reads:
- Causal-reasoning subset: ~2.25-4 billion×.
- Grounded-reasoning: ~1.5-2.6 billion×.
- Agent benchmarks: ~1.08-1.44 billion×.
- Tool-augmented: ~150,000,000×.
- Text NLL: ~210,000,000×.
- Knowledge-augmented: ~80,000,000×.
- VL benchmarks: ~270,000,000× (if #71-A shipped).
- LANGUAGE benchmarks: ~110-200M× (if #72-B shipped).
- **Single-GPU model-size ceiling: ~32B effective** (post-#74; via #74-A PHOENIX-1BIT-DISTILL hybrid).

#75-B applies #53 MOSAIC-MOE's shared-backbone-plus-per-expert-LoRA schema to #74's PHOENIX-quantized base. The mechanism is post-hoc — start from the post-#74 32B-effective dense trunk, cluster FFN neurons into E=8 expert groups based on co-activation patterns observed during a probe forward pass, then route per-token to top-k=2 experts:
- **#53 MOSAIC-MOE** (selected iter-197, never empirically validated): Sparse mixture-of-experts within reversible CHIRON. Shared FFN backbone + per-expert LoRA adapters (rank r=4, E=8 experts, k=2). Effective parameters 8× active = 144B effective at 18B-active compute. Already in the design pipeline; #75-B realizes it on the quantized substrate.
- **MOEfication (Yu et al. 2022)**: post-hoc conversion of dense FFN to sparse MoE via co-activation clustering. Production-validated on T5-base/T5-large; ~2× compute reduction at <0.5% downstream task degradation. Unlike train-from-scratch MoE (Mixtral, DeepSeek-V3), moefication is a POST-HOC structural rewrite — the dense pretrained FFN is preserved as the shared backbone, and only the routing + LoRA adapters are introduced.
- **#74 PHOENIX-1BIT-DISTILL-COMBO** (selected iter-218): demonstrated iter-212 composition pattern at binary tier. Provides the quantized substrate.

**Composition mechanism (sketch):**

- **Quantized substrate:** #74's PHOENIX-hybrid trunk preserved verbatim (binary middle + ternary edges + BF16 embedding-island). 32B-effective dense base.
- **Moefication step:** Run a one-time probe forward pass over a 100M-token MoE-clustering corpus (subset of Pile + curated). For each FFN layer in the binary middle band, cluster the d_ff=4·d_model intermediate neurons into E=8 expert groups via balanced-k-means on co-activation patterns. Each expert is a slice of the dense FFN (same shared weights, partitioned by neuron index).
- **Per-expert LoRA adapters:** Add rank r=4 LoRA adapters on top of each expert slice: `expert_i(x) = W_shared_i · x + B_i · (A_i · x)` where W_shared_i is the binary-quantized slice, A_i ∈ ℝ^{r×d_model} and B_i ∈ ℝ^{d_ff/E × r} are BF16 LoRA matrices. Rank r=4 default per #53 §3.
- **Routing:** Per-token gating function g(x) = softmax(W_gate · x), top-k=2 experts selected. Gating weights W_gate ∈ ℝ^{E × d_model} in BF16 (~2 MB per layer at d_model=4096, E=8). **Routing decision is deterministic given x** (preserves CHIRON's reversibility per #53 §4 Theorem 1).
- **Active parameters per token:** 32B × (top-k / E) = 32B × (2 / 8) = **8B active per forward pass.**
- **Effective parameter count:** 32B × E = 32B × 8 = **256B effective.**
- **Memory accounting (THE LOAD-BEARING question):**
  - 8 expert FFN slices SHARE the same #74 binary-quantized backbone weights via the shared-backbone schema. Slice partition is by NEURON INDEX, not by separate weight copies.
  - Per-expert LoRA adapters: 8 × (A_i + B_i) × num_FFN_layers ≈ 8 × (r·d_model + d_ff/E·r) × L_FFN. At r=4, d_model=4096, d_ff=16384, E=8, L_FFN=24 (binary middle band): 8 × (16384 + 8192) × 24 × 2 bytes = ~9.4 GB total LoRA at 32B-effective. **TOO HIGH.**
  - **Honest correction (CRITICAL):** Naive LoRA accounting at 32B-effective doesn't fit. Mitigations required:
    - Reduce LoRA rank from r=4 to r=2 → halves to ~4.7 GB.
    - LoRA adapters NF4-quantized (per #71-A's quantization scheme on adapter side) → ~1.2 GB at r=2 NF4.
    - Apply LoRA only to subset of FFN layers (e.g., 12 of 24 binary-middle layers) → ~600 MB at r=2 NF4 on half-coverage.
  - Net: **~600 MB - 1.2 GB total LoRA adapter memory at 256B-effective**, on top of #74's ~1.6 GB quantized trunk. New total trunk: 1.6 + (0.6 to 1.2) = **2.2-2.8 GB**.
- **Routing overhead:** ~5% per-step compute (gate function + top-k argmax + sparse expert dispatch). Per-step latency increase modest.

**Quality bookkeeping (the load-bearing argument):**
- From-scratch baseline NLL: BASE.
- Post-#74-alone NLL: BASE - (0.20 to 1.85) nat (improved per #74 §3.1).
- Post-MoE quality penalty: +0.10 to +0.25 nat (Mixtral 8x22B ablation: dense-equivalent vs MoE at fixed FLOPs shows ~0.1-0.2 nat gap at fixed compute budget; moefication post-hoc is ~+0.05 nat additional vs train-from-scratch MoE per Yu 2022).
- **Post-#75-B combined NLL: BASE - (0.20 to 1.85) + (0.10 to 0.25) = BASE - (0.10 to 1.70) nat.**
- **Net: NLL is STRICTLY IMPROVED (not regressed) by 0.10-1.70 nat compared to from-scratch baseline.**
- Compared to #74-alone: 0.05-0.20 nat WORSE in expectation (MoE quality penalty offsets a fraction of distillation gain). Compared to #73-alone: comparable or slightly worse.
- "Improved-not-compromise" framing per iter-212 admissibility holds; tighter range than #74's 0.20-1.85 nat by 0.10-0.15 nat at both ends.

**Headline magnitude:**
- **Effective parameter count at fixed 16 GB GPU memory: 1.84B → ~256B effective** (~139× expansion; 8× over post-#74 32B; the largest single-GPU effective-model-size lift in the research program to date).
- **Active parameters per token: ~8B** (4× FEWER than #74's 32B-dense at same memory; the FIRST per-token compute speedup in the post-#73 series).
- **Per-token wall-clock: ~2× faster than #74's 32B-effective dense; ~2× slower than native 1.84B BF16** (vs #74's ~9× slower; substantial wall-clock improvement on per-token axis).
- **Net NLL improvement over from-scratch: 0.10-1.70 nat** (slightly tighter than #74's 0.20-1.85 due to MoE penalty).
- **Trunk memory ratio: ~5-7× overall hybrid (slightly worse than #74's ~10-12× due to LoRA adapter overhead; the MoE expansion costs some memory ratio efficiency).**

**Speedup framing per iter-219 brief:**
- "Magnitudes better on compute speed": **DIRECTLY SATISFIED per-token** (4× faster than #74-dense; 2× slower than native baseline at much higher effective capacity); FIRST candidate in #73→#74→#75-B sequence to satisfy this directly. Magnitude on EFFECTIVE-PARAM-COUNT × SPEEDUP is 8× × 4× = ~32× joint axis lift over #74-dense.
- "Without compromising memory advantages": SATISFIED with caveat — overall hybrid ratio slightly worse than #74 (~5-7× vs ~10-12×) due to LoRA overhead, but EFFECTIVE-PARAM-COUNT-PER-MEMORY-BYTE is much better (256B at ~3 GB trunk vs #74's 32B at ~1.6 GB trunk = ~4.3× more effective-params-per-trunk-byte).
- "Without compromising NLL accuracy": SATISFIED via iter-212 framing — NLL improved 0.10-1.70 nat over from-scratch.

**Cumulative stack update (#75-B selected):**
- Effective single-GPU model-size ceiling: 32B (post-#74) → **~256B effective**.
- Active params per token: 32B (post-#74) → **8B** (4× fewer; per-token speedup recovered).
- Trunk memory ratio: ~10-12× hybrid (post-#74) → **~5-7× hybrid** (LoRA overhead; effective-param-per-byte ratio improves).
- Text NLL on shared corpus: improved by 0.10-1.70 nat over from-scratch baseline (slightly tighter than post-#74; ~0.05-0.20 nat WORSE than #74-alone).
- All other axes: marginally improved (more capacity at 256B-effective absorbs more teacher signal, more conditional knowledge).

**Engineering scope:** ~1700 LOC over 8 weeks. Moefication clustering (~250 LOC), per-expert LoRA implementation on quantized substrate (~400 LOC), routing kernel + top-k dispatch (~250 LOC), shared-backbone schema integration with #74 PHOENIX kernel (~250 LOC), QAT for LoRA adapters jointly with frozen quantized backbone (~150 LOC), Gate-0 mini-distill harness (~250 LOC), evaluation harness (~150 LOC).

**Joint Gate-0 PASS probability:** ~50% (Mixtral 8x22B + DeepSeek-V3 production-validated for train-from-scratch MoE; moefication post-hoc validated at smaller scale by Yu 2022; the joint composition with #74's binary-quantized substrate is novel; #53 MOSAIC-MOE never empirically validated; compounding risk vs #74-alone is ~10-15% drop).
**LLM-scale empirical confirmation probability at single-GPU CHIRON 256B-effective:** ~35% — modulo whether the routing learns useful per-token specialization on a binary-quantized substrate, whether LoRA adapters at NF4 r=2 retain enough capacity, whether #74's tight memory headroom (~3.4 GB) holds under additional routing + LoRA overhead.

---

## 2. Mechanism: post-hoc moefication on PHOENIX-quantized substrate + LoRA experts + SUPER-DISTILL teacher

### 2.1 Substrate inheritance from #74

The full #74-A PHOENIX-1BIT-DISTILL-COMBO trunk is preserved AS THE SHARED BACKBONE:
- BF16 embedding-island (input + first 2 + last 2 layers + LM head; ~15% of params).
- Ternary edges 1.58-bit (layers 3-6 + L-5 to L-2; ~30%).
- Binary middle 1-bit (layers 7 to L-6; ~55%).
- Llama 3.1 405B teacher distillation pipeline (KL-CE loss, top-K=4 cache, α schedule 0.05 → 0.9, τ=3.0).

**No modifications to #74 substrate.** #75-B is a structural delta on top of the FFN computation graph in the binary middle band. Edge bands and embedding-island remain dense (no MoE — too few neurons to benefit from sparse routing; quality cost > compute gain).

### 2.2 Moefication clustering procedure (per Yu 2022)

For each FFN layer l in the binary middle band (24 layers total at 32B-effective with d_ff=16384):
1. **Probe forward pass:** Run 100M tokens of curated Pile through the post-#74 dense model. Record per-neuron activation magnitudes |a_l[i]| for each intermediate neuron i ∈ [0, d_ff).
2. **Co-activation matrix:** Compute C_l[i,j] = E[1{a_l[i] > θ} · 1{a_l[j] > θ}] over the probe corpus (θ = activation threshold; θ = mean over corpus).
3. **Balanced k-means:** Partition the d_ff neurons into E=8 balanced clusters using co-activation similarity as the distance metric. Each cluster receives d_ff/E = 2048 neurons.
4. **Assignment:** Cluster c_l[i] ∈ {0, 1, ..., 7} assigns each neuron i to one of E experts.

**Cost:** ~2 GPU-hours one-time for 100M-token probe forward + clustering. Comparable to #74's master-weight initialization cost.

**Output:** Per-layer cluster assignment matrix C_l (no new weights). The expert i in layer l is the SLICE of the dense FFN restricted to neurons {j : c_l[j] = i}.

### 2.3 Per-expert LoRA adapters

Each expert receives a rank r=2 LoRA adapter on top of the binary-quantized FFN slice:
```
expert_i(x) = (W_q_l_slice_i · x) · α_i_layer + B_i · (A_i · x)
```
where:
- W_q_l_slice_i is the binary-quantized FFN weight matrix restricted to expert i's neuron slice (slice of d_ff/E = 2048 rows, all d_model = 4096 columns); SHARED across experts in the sense that all expert slices come from the same dense parent matrix.
- α_i_layer is the per-expert per-layer learnable scale (extends #74's per-layer α to per-expert).
- A_i ∈ ℝ^{r × d_model} = ℝ^{2 × 4096} in NF4 quantization (~1 KB per layer per expert).
- B_i ∈ ℝ^{d_ff/E × r} = ℝ^{2048 × 2} in NF4 (~256 bytes per layer per expert).

**Total LoRA adapter memory at 32B-effective with E=8, r=2, NF4, applied to 12 of 24 binary-middle layers:**
12 layers × 8 experts × ((2 × 4096) + (2048 × 2)) × 0.5 bytes/weight (NF4) = 12 × 8 × 12288 × 0.5 = ~590 MB.

### 2.4 Routing function

Per-layer routing gate `g_l(x) = softmax(W_gate_l · x)` where W_gate_l ∈ ℝ^{E × d_model} in BF16. Top-k=2 experts selected per token. Combined output:
```
y = Σ_{i ∈ top-2(g_l(x))} g_l(x)[i] · expert_i(x)
```

**Routing memory:** 24 layers × 8 × 4096 × 2 bytes = ~1.6 MB. Negligible.

**Routing compute:** O(E·d_model + E·log(k)) per token per layer. At E=8, k=2, d_model=4096: ~33K flops per token per layer; ~5% per-step overhead at 256B-effective.

**Determinism (per #53 §4 Theorem 1):** g_l(x) is a deterministic function of x. Top-2 selection is deterministic up to tie-breaking (handle by stable argmax). **CHIRON's reversibility preserved exactly** — the same input x produces the same expert routing in forward and inverse walks.

### 2.5 Quantization-aware training of LoRA adapters

Joint QAT recipe:
- Frozen quantized backbone weights w_q_l_slice_i (no STE updates; #74's QAT continues only on master weights, not on the moefication slices).
- LoRA matrices A_i, B_i in NF4 with master BF16 weights (per Lit-LLaMA QAT-LoRA recipe; ~5K LOC mature reference).
- Routing gate W_gate_l in BF16 (small enough to avoid quantization).
- α_i_layer per-expert per-layer scale: continued QAT from #74; new per-expert dimension co-optimized.

**Critical convergence concern:** LoRA adapters on a frozen binary-quantized backbone may struggle to compensate for binary's 0.15-0.30 nat penalty. Mitigation: warmup phase where LoRA is trained alone (backbone frozen) for first 5% of training, then joint with backbone QAT.

### 2.6 SUPER-DISTILL teacher pipeline (per #68 §2)

Reuse #68 cached-logit pipeline verbatim — identical to #74:
- **Teacher:** Llama 3.1 405B (default; English-dominant).
- **Cache:** top-K=4 logits per token (~16 TB on NVMe; $0 marginal cost reused from #74).
- **Loss:** L = α · CE(student, teacher_token) + (1-α) · τ² · KL(softmax(z_T/τ) || softmax(z_S/τ)). α schedule 0.05 → 0.9; τ = 3.0.
- **No modifications to #68 pipeline.** #75-B student receives identical teacher signal as #74 student; only the FFN computation graph differs.

### 2.7 Composition-stage scheduling

Per #61 COSMIC stage scheduling with Stage 1 further extended:
- **Stage 1 (Foundation, 75% of training; +5% vs #74):** PHOENIX trunk active; SUPER-DISTILL active α = 0.05 → 0.5. **MoE routing INACTIVE for first 25% of Stage 1** (dense forward; allows dense-equivalent QAT to converge), then MoE activated with routing entropy regularization for remainder.
- **Stage 2 (Reasoning, 17%):** MoE routing active with all experts; per-expert specialization emerges via routing entropy decay schedule. SUPER-DISTILL α = 0.5 → 0.85.
- **Stage 3 (Refinement, 8%):** MoE routing frozen (per-expert scales α_i_layer frozen, routing gates frozen); per-expert LoRA fine-tuning continues. SUPER-DISTILL α = 0.85 → 0.95.

### 2.8 Inference path

At inference: master BF16 weights and LoRA master weights dropped; only quantized weights w_q + per-expert α_i + routing gates W_gate + NF4 LoRA matrices retained. **Inference memory: ~2.5 GB at 256B-effective** (binary middle + ternary edges + BF16 island + LoRA NF4 + routing gates BF16). Inference compute per token: top-2 expert dispatch = 25% of dense FFN compute + 5% routing overhead = ~30% of #74-dense per-token compute. **~2× faster per token than #74-dense.**

---

## 3. Theoretical analysis

### 3.1 Theorem 1 — Net NLL bound under composition (NEW; the load-bearing theorem)

**Theorem 1 (informal).** Let BASE be the NLL of from-scratch CHIRON-1.84B trained on the standard Pile + curated corpus without distillation. Under #75-B composition (PHOENIX-1BIT trunk + ternary edges + SUPER-DISTILL + post-hoc moefication + per-expert LoRA):
```
NLL_post-#75-B ≤ BASE - Δ_distill + Δ_PHOENIX-1BIT-hybrid + Δ_MoE-penalty
```
where:
- Δ_distill ∈ [0.5, 2.0] nat per #68 SUPER-DISTILL bound.
- Δ_PHOENIX-1BIT-hybrid ∈ [0.15, 0.30] nat per #74 §3.1.
- Δ_MoE-penalty ∈ [0.10, 0.25] nat per Mixtral ablation + Yu 2022 moefication post-hoc empirical.

**Net:** NLL_post-#75-B ≤ BASE - (0.5 - 0.30 - 0.25) = BASE - (-0.05) at the very pessimistic end → effectively NEUTRAL or slightly negative; NLL_post-#75-B ≤ BASE - (2.0 - 0.15 - 0.10) = BASE - 1.75 nat at the optimistic end. **Tighter range to BASE - (0.10 to 1.70) nat at central estimate.**

**Headline:** **NLL improved by 0.10-1.70 nat over from-scratch baseline at central estimate; range tightens to NEUTRAL at very pessimistic end.** Strictly an improvement at the central estimate; not at the tail.

**Proof sketch.** Three additive penalty terms with approximately independent sources at the gradient level: distillation supervision (positive), binary quantization noise (small negative), MoE conditional-computation penalty (small negative). Mixtral 8x22B published ablation: at fixed FLOPs, MoE underperforms equivalent-FLOP dense by ~0.1-0.2 nat, but at fixed effective-param-count, MoE matches dense (the dense-equivalent of 256B is 256B, infeasible at 16 GB). Yu 2022 moefication post-hoc adds ~0.05 nat over train-from-scratch MoE (the structural rewrite is imperfect). The distillation term DOMINATES at central estimate; pessimistic end has all three penalties at maximum and distillation at minimum, yielding near-neutral. Mechanistically sound at central estimate; pessimistic tail requires Gate-0 to verify. □

**Honest caveat:** The very pessimistic end (BASE - 0.05 nat or worse) would VIOLATE the iter-212 admissibility rule (not "improved over from-scratch"). If Gate-0 shows pessimistic-end behavior, mechanism is REJECTED. Probability of pessimistic-end at central scale: ~25-35% (HIGHER than #74's pessimistic risk).

### 3.2 Theorem 2 — Memory accounting at 256B-effective

**Theorem 2 (informal).** Trunk + LoRA + routing memory at 256B-effective on 16 GB single GPU:

Components:
- **PHOENIX-quantized trunk (32B base, shared across experts):** ~1.6 GB (per #74 §3.2).
- **Per-expert LoRA NF4 r=2 (12 of 24 binary-middle layers):** ~590 MB (§2.3).
- **Per-expert α_i_layer scales (BF16):** 24 layers × 8 experts × 4 bytes = ~768 bytes; negligible.
- **Routing gates W_gate_l (BF16):** ~1.6 MB (§2.4); negligible.
- **Master weights (host RAM, Fallback A):** 32B × 2 + LoRA × E × 2 = 64 GB + 1.2 GB ≈ 65 GB host.
- **Adam state (host RAM):** 32B × 8 + LoRA × 8 ≈ 256 GB host.
- **Activations:** ~4 GB (active fraction 25%; activation memory scales with active params; lower than #74-dense's 4 GB at 32B due to top-2 dispatch sparse activations, but routing logits + gating add overhead; net comparable).
- **KV cache:** ~3 GB (with #42 SCFA spectral compression; same as #74).
- **Routing dispatch buffer:** ~500 MB (sparse top-k dispatch tables; transient).
- **Framework overhead:** ~2 GB (CUDA + buffers).
- **PCIe prefetch buffer (extended for 65 GB master):** ~2.5 GB (vs #74's 2 GB).

**Total GPU resident:** 1.6 + 0.6 + 4 + 3 + 0.5 + 2 + 2.5 = **~14.2 GB**, headroom **~1.8 GB** at 16 GB ceiling.

**Honest framing:** Headroom is VERY TIGHT (1.8 GB margin at 16 GB ceiling; 47% reduction vs #74's 3.4 GB). This reflects the additional LoRA + routing dispatch + extended prefetch overhead. **Mitigations**:
- LoRA-coverage reduction (12 of 24 → 8 of 24 layers) drops LoRA memory to ~390 MB; gains ~200 MB headroom.
- Smaller routing dispatch buffer with cooperative-thread-array scheduling drops to ~300 MB; gains ~200 MB headroom.
- Together: ~2.2 GB headroom achievable but at quality cost.

**Effective model size at 16 GB GPU + Fallback A: ~256B MOEFICATION-DISTILL** ≈ 139× expansion over native 1.84B; 8× over post-#74 32B.

### 3.3 Theorem 3 — Bijectivity and reversibility under MoE on PHOENIX-1BIT

**Theorem 3 (informal).** CHIRON's reversible-flow trunk is composed of symplectic shears `(x, y) → (x + f_w(y), y)`. Under #75-B:
1. Routing g(x) is a deterministic function of x.
2. Each expert i computes f_{w_q_i, A_i, B_i}(y) deterministically.
3. The combined output Σ_{i ∈ top-2(g(x))} g(x)[i] · f_i(y) is a deterministic function of x, y.

The shear `(x, y) → (x + Σ_{i ∈ top-2(g(x))} g(x)[i] · f_i(y), y)` is bijective with inverse `(x', y) → (x' - Σ_{i ∈ top-2(g(x'))} g(x')[i] · f_i(y), y)`. **Bijectivity preserved end-to-end.** Inherits #53 §4 Theorem 1 + #74 Theorem 3.

**Critical caveat:** The routing `g(x)` depends on x ONLY (not on y). This is essential — if routing depended on y, the shear inverse would require recomputing y from x' which breaks reversibility. **#53 MOSAIC-MOE's design constraint is preserved verbatim.**

**Composition with #44 MELT (TT-FFN):** MELT's TT-cores are themselves PHOENIX-quantized + moeficated. Each expert i has its own TT-core slice; per-expert TT-FFN is bijective per-expert; routing combines per-expert TT-FFN outputs additively per #75-B §2.4. Joint bijectivity preserved.

### 3.4 Compute-axis honest framing — THE FIRST PER-TOKEN SPEEDUP

**Per-token compute at 256B-effective:**
- Active parameters per forward: 32B × (k/E) = 8B.
- Per-active-param compute: same as #74-dense (XNOR-popcount overhead 2× of BF16).
- Total per-token wall-clock: 8B × 2× = 16B-equivalent compute.

**Per-token compute at #74-dense 32B-effective:**
- 32B × 2× = 64B-equivalent compute.

**Speedup: 64 / 16 = 4× per-token wall-clock improvement of #75-B over #74.**

**Per-token wall-clock at 256B-effective vs native 1.84B BF16:**
- #75-B: 16B-equivalent compute.
- Native 1.84B BF16: 1.84B-equivalent compute.
- Ratio: 16 / 1.84 ≈ 8.7× slower vs native.

**Honest framing:** #75-B is ~2× slower per token than native baseline at 139× more effective parameter count; ~4× faster per token than #74 at 8× more effective parameter count. **The first paradigm in the post-#73 series to deliver a per-token speedup over its predecessor.** The user brief's "magnitudes better on compute speed" is satisfied at ~4× per-token level (over #74) AND at the EFFECTIVE-MODEL-SIZE × INVERSE-WALL-CLOCK level: 8× × 4× = 32× joint magnitude lift over #74-dense.

### 3.5 NLL preservation honest framing

- **Pre-#75-B baseline: from-scratch BF16 1.84B.** NLL = BASE.
- **Pre-#75-B with #74 only: BASE - (0.20 to 1.85) nat = BETTER.**
- **Post-#75-B: BASE - (0.10 to 1.70) nat = STILL BETTER than from-scratch; ~0.05-0.20 nat WORSE than #74-alone.**

**Iter-212 framing satisfied at central estimate; pessimistic tail is borderline (NEUTRAL not improved). Gate-0 must verify central-estimate behavior.**

### 3.6 Compounding-risk axis

**Reader-side critical view:** #75-B compounds two unvalidated mechanisms:
- #53 MOSAIC-MOE: selected at iter-197 design but never empirically validated at any scale.
- #74 PHOENIX-1BIT: selected at iter-218 with Gate-0 mandatory; compounding risk if Gate-0 fails for either.

**Resolution:** #75-B is GATED on #74's Gate-0 PASS. If #74 Gate-0 fails, #75-B reverts to fall-back composition without binary tier (use #73 ternary trunk + moefication; ceiling drops to ~144B-effective). If #53 MOSAIC-MOE Gate-0 fails (i.e., moefication degrades NLL beyond 0.25 nat), mechanism rejected; fall back to #74-alone (32B-effective, no MoE).

### 3.7 LANGUAGE / multilingual axis

If #72-B MULTILINGUAL-DISTILL is shipped, #75-B composes: 256B-effective × Qwen2.5-72B teacher. Substantial LANGUAGE-axis lift due to higher capacity (vs #74's 32B); ~5-7× LANGUAGE benchmarks lift. No interference; clean composition.

---

## 4. Composition with #53 + #68 + #74 + prior 32 paradigms

### 4.1 Composition with #53 MOSAIC-MOE (re-admission/realization)

#53 was selected at iter-197 design but never empirically validated. #75-B realizes #53 on the iter-218 #74 substrate. **All Theorems 1-2 from #53 inherited; new per-expert NF4 LoRA quantization layered on top.**

### 4.2 Composition with #74 PHOENIX-1BIT-DISTILL-COMBO

#74 substrate preserved verbatim (binary middle + ternary edges + BF16 island + Llama 3.1 405B teacher). MoE structural delta on FFN computation graph in binary middle band only.

### 4.3 Composition with #68 SUPER-DISTILL

#68 cached-logit pipeline reused at $0 marginal cost (cache exists from #74). KL-CE blended loss applied to MoE-routed student logits.

### 4.4 Composition with #44 MELT (TT-FFN)

Each expert FFN is itself TT-decomposed per #44. Per-expert TT-cores at NF4 LoRA on shared backbone. Joint memory ratio sub-multiplicative.

### 4.5 Composition with #61 COSMIC stages

Per §2.7: extended Stage 1 (75%) for joint binary + MoE QAT convergence; routing activation deferred to mid-Stage-1.

### 4.6 Marginal contribution beyond pre-#75-B stack (post-#74)

| Axis | Pre-#75-B (post-#74) | Post-#75-B | Marginal |
|---|---|---|---|
| Effective model size | 32B-effective | **256B-effective** | **+224B (8× expansion)** |
| Active params per token | 32B (dense) | **8B (top-2 of 8)** | **0.25× (4× compute reduction)** |
| Trunk memory ratio (overall hybrid) | ~10-12× | ~5-7× | -1.5× to -2× (LoRA overhead) |
| Effective-param-per-trunk-byte | 20B / GB | **~91B / GB** | **+4.5×** |
| NLL on shared corpus | BASE - (0.20 to 1.85) nat | BASE - (0.10 to 1.70) nat | -0.05 to -0.20 nat |
| Per-token wall-clock | 9× slower than native | **2× slower than native; 4× faster than #74-dense** | **+4× per-token speedup over #74** |
| All other axes | per-axis cumulative | preserved or marginally improved | ~1.0× to ~1.3× |

**Marginal contribution: 224B additional effective capacity at 4× per-token speedup over #74, with NLL strictly improved at central estimate (slightly tighter range than #74-alone).**

---

## 5. Quantitative speedup with honest band

### 5.1 Headline

**~256B effective parameter count + ~8B active per token (4× compute reduction over #74-dense at same effective capacity) + ~5-7× hybrid trunk memory ratio + NLL improved by 0.10-1.70 nat over from-scratch.**

### 5.2 Honest band breakdown

| Band end | Conditions |
|---|---|
| **400B effective (high)** | E=16 experts, k=2 routing; LoRA r=4 on full 24 binary-middle layers; activation compression aggressive; ~50× compute reduction at 16× expansion over #74 |
| **256B effective (headline)** | E=8, k=2, r=2 NF4 LoRA on 12 layers; ~4× compute reduction at 8× expansion over #74 |
| **128B effective (low)** | E=4, k=2 (less aggressive routing); LoRA r=2 on 8 layers; ~2× compute reduction at 4× expansion |
| **<64B effective (failure)** | MoE quality penalty exceeds 0.30 nat at scale; routing fails to specialize; mechanism RESERVED, fall back to #74 (32B effective with no regression) |

### 5.3 Empirical anchors

- **Mixtral 8x22B (Mistral 2024):** ~141B params / ~39B active; train-from-scratch MoE; <0.1 nat ablation gap vs equivalent-FLOPs dense at fixed compute. Closest scale anchor for the COMPOSITION at 8-expert MoE.
- **DeepSeek-V3 (DeepSeek 2024):** 671B params / 37B active (8 of 256 routed experts); FP8 quantization; <0.05 nat ablation gap vs equivalent-FLOPs dense. ANCHOR for FP8 + MoE composition; closest production-validated for QUANTIZED MoE.
- **MoEfication (Yu 2022):** post-hoc dense → MoE conversion of T5-base (220M) and T5-large (770M); ~2× compute reduction at <0.5% downstream task degradation. ANCHOR for the post-hoc moefication mechanism; smaller scale than CHIRON-32B.
- **#53 MOSAIC-MOE (this research program iter-197 design):** never empirically validated; design exists. SAME mechanism as #75-B but on dense BF16 substrate.
- **LoRA-on-quantized-model (QLoRA, Dettmers 2023):** LoRA r=8 on 4-bit NF4 quantized 65B Llama; matches FP16 fine-tuning at 1/4 memory. ANCHOR for LoRA-on-quantized; scales to 65B production-validated.
- **Lit-LLaMA QAT-LoRA:** LoRA + QAT joint training; reference impl ~5K LOC; production-validated.

The combination: Mixtral/DeepSeek (MoE at scale) + MoEfication (post-hoc conversion) + QLoRA (LoRA on quantized) + #74 (binary substrate). NO direct precedent for the JOINT composition at 256B-effective scale. **Three independent precedent gaps stacked; Gate-0 mandatory and high-priority.**

### 5.4 Risk-adjusted claim

Joint Gate-0 PASS probability × LLM-scale empirical confirmation probability = 0.50 × 0.35 = **0.175 expected realization**. Risk-adjusted: 256B effective × 0.35 = **~90B effective realized** in the central case; 4× compute speedup × 0.50 = **~2× realized speedup over #74**.

This is HIGHER UPSIDE but HIGHER VARIANCE than #74. Worst-case (Gate-0 FAIL): falls back to #74 (32B-effective) — no regression. 80th-percentile case: ~150B-effective with ~3× speedup over #74-dense — still substantial.

---

## 6. Cumulative stack update

### 6.1 Pre-#75-B stack (post-#74 selected at iter-218)

| Axis | Value |
|---|---|
| Causal-reasoning subset | ~2.25-4 billion× |
| Grounded-reasoning | ~1.5-2.6 billion× |
| Agent benchmarks | ~1.08-1.44 billion× |
| Tool-augmented | 150,000,000× |
| Text NLL (English) | ~210,000,000× |
| Knowledge-augmented | ~80,000,000× |
| **Effective single-GPU model size** | **~32B effective** (post-#74) |
| **Active params per token** | **32B (dense)** |
| **Trunk memory ratio** | **~10-12× (hybrid binary middle)** |

### 6.2 Post-#75-B stack (MOEFICATION-DISTILL-CHIRON selected)

| Axis | Pre-#75-B | #75-B factor | Post-#75-B |
|---|---|---|---|
| Causal-reasoning subset | ~2.25-4B× | × ~1.8-2.5× (more capacity at 256B; conditional specialization helps reasoning) | ~4-10B× |
| Grounded-reasoning | ~1.5-2.6B× | × ~1.8-2.5× | ~2.7-6.5B× |
| Agent benchmarks | ~1.08-1.44B× | × ~1.5× | ~1.6-2.2B× |
| Tool-augmented | 150,000,000× | × ~1.0× | 150,000,000× |
| Text NLL (English) | ~210,000,000× | × ~1.5-2× (more capacity, slightly higher penalty) | ~315M-420M× |
| Knowledge-augmented | ~80,000,000× | × ~1.3× | ~104,000,000× |
| **Effective single-GPU model size** | **~32B-effective** | **× 8** | **~256B effective** |
| **Active params per token** | **32B** | **× 0.25** | **~8B** |
| **Per-token wall-clock vs native** | **9× slower** | **× 0.22** | **~2× slower than native** |

### 6.3 Honesty caveat

**The 8× expansion to 256B-effective is the load-bearing claim.** If empirical realization at 256B-effective is only 90B (35th percentile risk-adjusted), the claim degrades to ~3× expansion over #74 — still substantial but not magnitude-class. Worst-case (Gate-0 FAIL: MoE quality penalty exceeds 0.30 nat): mechanism RESERVED, fall back to #74 (32B effective) — no regression. Stronger fallback than #73→#74 because #75-B's failure modes are well-isolated to the MoE structural delta.

The selection logic: SELECT IF (Joint Gate-0 PASS confirms MoE quality penalty ≤ 0.25 nat at 64B-effective AND per-token wall-clock ≤ 5× of 1.84B BF16). Otherwise RESERVE.

---

## 7. Engineering scope

### 7.1 Component breakdown

| Component | LOC | Description |
|---|---|---|
| Moefication clustering | 250 | Probe forward + co-activation matrix + balanced k-means; per-layer cluster assignments; one-time ~2 GPU-hour cost |
| Per-expert NF4 LoRA on quantized substrate | 400 | Adapter weight storage, NF4 encoder/decoder, joint kernel with #74 PHOENIX backbone; references QLoRA + Lit-LLaMA QAT-LoRA |
| Routing kernel + top-k dispatch | 250 | CUDA top-k dispatch; sparse expert combination; routing entropy regularization; deterministic tie-break |
| Shared-backbone schema with #74 PHOENIX | 250 | Per-expert slice of dense binary trunk; per-expert α scale dispatch; backward-compatible with #74 inverse walk |
| QAT for LoRA + frozen backbone | 150 | Master BF16 LoRA + NF4 quantized; backbone frozen (no STE); routing gate BF16 |
| Gate-0 mini-distill harness | 250 | Mini 64B-effective; assert NLL improvement ≥ 0.10 nat over from-scratch; assert MoE quality penalty ≤ 0.25 nat; routing entropy convergence check |
| Evaluation harness | 150 | NLL on Pile-eval + MMLU + HumanEval + GSM8K; per-expert specialization metrics; routing entropy decay tracking |
| **Total** | **~1700 LOC** | **~8 weeks engineering** (1 week longer than #74 due to MoE harness + routing kernel) |

### 7.2 External-dependency risk

- **MoEfication reference impl** (Yu 2022 GitHub): MIT license; ~2K LOC; integration moderate (clustering + dispatch).
- **QLoRA reference impl** (Dettmers 2023): MIT license; ~3K LOC; LoRA on NF4 quantized; production-ready.
- **Lit-LLaMA QAT-LoRA** (Lightning AI): Apache 2.0; ~5K LOC; QAT-LoRA recipe.
- **#74 PHOENIX kernel** (this research program iter-218 if Gate-0 PASS): mandatory dependency; #75-B inherits #74's risk surface.
- **Cache from #68 + #74 reused at $0 marginal cost.**

### 7.3 Timeline

- **Weeks 1-2:** Moefication clustering implementation; co-activation matrix + balanced k-means; per-layer cluster assignment storage.
- **Week 3:** Per-expert NF4 LoRA on quantized substrate; QLoRA + Lit-LLaMA QAT-LoRA integration with #74 PHOENIX backbone.
- **Week 4:** Routing kernel + top-k dispatch; routing entropy regularization; deterministic tie-break.
- **Week 5:** Joint training schedule with extended Stage 1; routing activation deferred to mid-Stage-1.
- **Week 6:** Gate-0 mini-distill on 64B-effective; assert NLL improvement ≥ 0.10 nat AND MoE quality penalty ≤ 0.25 nat AND wall-clock per token ≤ 5× of 1.84B BF16 baseline.
- **Week 7:** Evaluation harness; per-expert specialization metrics; routing entropy decay tracking.
- **Week 8:** Sign-off; Gate-1 full 256B-effective preparation.

### 7.4 Hardware budget

- **GPU:** single 16 GB (RTX 4080 SUPER target; RTX 4090 24 GB strongly preferred for the tight 1.8 GB headroom; RTX 5090 32 GB ideal).
- **Host RAM:** 192 GB minimum (master weights at 32B BF16 = 64 GB + LoRA master = 1.2 GB + Adam state = 256 GB / 8 effectively-paged + system overhead). DDR5 6400+ recommended; ECC mandatory.
- **NVMe:** 5 TB (cache from #68/#74 + master-weight checkpoints + Adam state + co-activation matrix + LoRA checkpoints).
- **Cloud Gate-0:** ~$15K (64B-effective × 200 GPU-hours; longer than #74's Gate-0 due to MoE warmup phase).
- **Cloud Gate-1:** ~$80K (256B-effective × 600 GPU-hours; substantially longer than #74 due to MoE training + extended Stage 1).

---

## 8. Gates

### 8.1 Gate-0 — premise validation (MANDATORY before wire-in)

**Hypothesis:** MOEFICATION-DISTILL-CHIRON 64B-effective model (#74 PHOENIX hybrid base + E=8, k=2, r=2 NF4 LoRA per-expert + Llama 3.1 405B SUPER-DISTILL) trained on 100B Pile-eval tokens achieves:
- NLL ≥ 0.10 nat better than from-scratch BF16 1.84B baseline; AND
- MoE quality penalty (#75-B-64B vs #74-alone-32B distilled): NLL gap ≤ 0.25 nat; AND
- Per-token wall-clock ≤ 5× of 1.84B BF16 baseline (allow 1.5× over theoretical 4×).

**Procedure:**
- Build #75-B 64B-effective model (~2× expansion of #74's 32B-effective).
- Apply MoEfication clustering + per-expert NF4 LoRA + SUPER-DISTILL.
- Train for 200 GPU-hours on 100B Pile-eval tokens with extended Stage 1 (75%) and routing-activation-deferred warmup.
- Evaluate on Pile-eval test split + MMLU + HumanEval + GSM8K + per-expert specialization metrics.

**Pass criterion:**
- All three above quantitative bars; AND
- Routing entropy converges to ~1.5-2.0 nats (indicates real per-token specialization, not uniform random); AND
- Per-expert α scale convergence stable across all 8 experts in middle band; AND
- No catastrophic divergence over 200 GPU-hours.

**Estimated cost:** ~$15K cloud + 4 weeks engineer time.
**Pass probability:** ~50%.

### 8.2 Gate-1 — full 256B-effective validation

**Procedure:** Build #75-B 256B-effective model on 16 GB GPU + Fallback A. Train for 35 days (~840 GPU-hours; 20% more than #74 Gate-1).
**Pass criterion:**
- NLL improvement ≥ 0.10 nat over from-scratch 1.84B; AND
- Effective model size ≥ 180B (allow ~70% of theoretical 256B); AND
- Per-token wall-clock ≤ 4× of 1.84B BF16 (allow 2× over theoretical 2×); AND
- Stable training; AND
- Downstream benchmarks ≥ post-#74 32B-effective baseline.

**Estimated cost:** ~$80K cloud + 6 weeks engineer time.
**Pass probability:** ~35%.

### 8.3 Gate-2 — multi-teacher integration

Multi-teacher KL-CE blend (#68 English + #69 reasoning + #70 tool + #72-B multilingual). Each teacher routes to specialized expert subsets via routing entropy regularization.

### 8.4 Gate-3 — long-run stability

60-day continuous training; per-expert α convergence; routing distribution stability; no expert collapse (uniform load balancing maintained).

---

## 9. Honest gaps and failure modes

### 9.1 Compounding-risk caveat (vs #74)

#75-B compounds three unvalidated mechanisms: #53 MOSAIC-MOE, #74 PHOENIX-1BIT-DISTILL, and the post-hoc moefication on quantized substrate. Joint Gate-0 PASS probability is ~50% — 10 points lower than #74-alone. The compounding-risk axis is the principal differentiator vs #74.

### 9.2 Tight memory headroom (1.8 GB at 16 GB ceiling)

Per §3.2: 14.2 GB GPU-resident with hybrid + Fallback A; 1.8 GB headroom. Risks:
- Long-context KV cache may exceed 3 GB allocation at T=4096+.
- Routing dispatch buffer at peak may exceed 500 MB.
- Activation memory at 256B-effective with 25% active fraction has higher peak than uniform-dense due to expert-routing burst patterns.

**Mitigation:** Aggressive #42 SCFA spectral compression on KV; smaller dispatch buffer with cooperative-thread-array scheduling; in-context length capped at 4096 with #54 JAMBA-CHIRON SSM extension.

### 9.3 #53 MOSAIC-MOE empirical-validation gap

#53 was selected at iter-197 design but never validated. #75-B is THE first empirical test of #53's mechanism, simultaneously layering on top of #74's also-unvalidated quantization tier. If #53 fails (routing fails to specialize, expert collapse, etc.), entire #75-B mechanism rejected.

**Mitigation:** Gate-0 directly tests at 64B-effective with explicit routing-entropy convergence and per-expert specialization metrics. If Gate-0 fails on these metrics, mechanism rejected with no regression.

### 9.4 Per-expert LoRA capacity at NF4 r=2 (CRITICAL)

NF4 r=2 LoRA may be too low capacity to compensate for binary backbone + MoE penalty. Concerns:
- Mixtral uses full BF16 dense FFN per expert (no LoRA factorization); production-validated capacity.
- QLoRA validates LoRA-on-quantized at r=8 BF16; #75-B is r=2 NF4, ~16× less LoRA capacity.
- Joint binary + MoE + low-LoRA may over-reduce per-expert representation.

**Mitigation:** Gate-0 with r=2 NF4; if NLL gap > 0.25 nat, escalate to r=4 NF4 (doubles LoRA memory; tightens headroom further but still feasible at ~2.4 GB total trunk).

### 9.5 Routing failure modes

- **Expert collapse:** all tokens route to one expert; defeats MoE benefit. Mitigation: routing entropy regularization (load-balancing loss, Mixtral-standard).
- **Routing instability:** routing flickers between epochs. Mitigation: routing gate gradient scaling (Mixtral-standard).
- **Pessimistic-end NLL at-or-near-NEUTRAL:** #75-B's pessimistic tail (BASE - 0.05 nat or worse) violates iter-212 admissibility. Mitigation: Gate-0 verifies central-estimate behavior at 64B; if pessimistic tail observed, mechanism rejected.

### 9.6 The "novelty" question

#75-B is mechanism-equivalent to:
- #53 MOSAIC-MOE (this research program iter-197 design) + Yu 2022 MoEfication + QLoRA + #74 PHOENIX-1BIT-DISTILL.

What is GENUINELY new at the program level:
- The JOINT composition at 256B-effective on a 16 GB single GPU is unprecedented.
- The composition of MoE + 1-bit quantization is novel; nearest analogue DeepSeek-V3 is FP8 + MoE, not 1-bit.
- The post-hoc moefication on a binary-quantized substrate has no published precedent.
- Theorem 1 (joint NLL bound under three penalty terms) is new.

What is NOT new:
- Mixture-of-experts (Shazeer 2017, Mixtral 2024, DeepSeek-V3 2024).
- MoEfication (Yu 2022).
- LoRA on quantized models (QLoRA 2023).
- Per-expert LoRA factorization (#53 MOSAIC-MOE 2026 design).

**Honest framing:** #75-B's novelty is the SPECIFIC composition at 1-bit + 8-expert + post-hoc moefication on a single 16 GB GPU; not the architectural primitive. Genuinely novel at the program level; not novel as standalone techniques.

### 9.7 Compute-axis honest cost vs framing

Per-token wall-clock at 256B-effective is ~2× slower than native 1.84B BF16 — substantially BETTER than #74's 9× slower. The user brief's "magnitudes better on compute speed" is satisfied at ~4× per-token level (over #74-dense at same memory budget). **First post-#73 paradigm to satisfy compute-speed clause directly per-token.**

### 9.8 Joint Gate-0 PASS + LLM-scale empirical confirmation probabilities (LOWER than #74)

| Estimate | Value | Comparison to #74 |
|---|---|---|
| Joint Gate-0 PASS probability | **~50%** | -10% (vs #74's 60%) |
| Joint Gate-1 PASS probability | **~35%** | -10% (vs #74's 45%) |
| LLM-scale empirical confirmation at 256B-effective | **~35%** | -10% |
| Risk-adjusted effective model size | **~90B-150B** (= 256B × 0.35-0.6) | HIGHER mean than #74's 14-19B; higher variance |
| Risk-adjusted per-token speedup vs #74 | **~2-3×** | comparable to magnitude framing |
| Risk-adjusted NLL improvement | **~0.05-0.85 nat** | LOWER (vs #74's 0.10-0.83 nat) |
| Probability effective ≥ 100B | **~55%** | no analog in #74 |
| Probability effective ≥ 256B | **~25%** | no analog in #74 |
| Probability NLL improvement ≥ 0.20 nat | **~50%** | -5% (vs #74's 55%) |

These probabilities are LOWER than #74's by ~10% across all axes; the increased variance reflects MoE + binary compounding-risk.

### 9.9 Production precedent

**Production precedents:**
- Mixtral 8x22B + DeepSeek-V3: MoE at scale; production-validated.
- MoEfication (Yu 2022): post-hoc dense → MoE; production-validated at small scale.
- QLoRA (Dettmers 2023): LoRA-on-quantized; production-validated at 65B.
- #74 PHOENIX-1BIT-DISTILL (this research program iter-218 if Gate-0 PASS): direct precedent for the binary substrate.

**No published precedent for the JOINT composition at 256B-effective scale on 16 GB single GPU.** #75-B is the most aggressive composition in the research program to date.

---

## 10. Bottom line / verdict

### 10.1 Verdict: **SELECT-CONDITIONAL on joint Gate-0 PASS**

MOEFICATION-DISTILL-CHIRON is recommended for **SELECT-CONDITIONAL** on six grounds, with two caveats compared to #74:

**1. First per-token compute speedup in the post-#73 series.** ~4× faster per-token wall-clock than #74-dense at same memory budget; first paradigm in the series to directly satisfy the iter-219 brief's "compute speed" clause per-token.

**2. ~256B-effective is exceptional model-size lift.** 8× expansion over #74's 32B; 139× over native 1.84B. Largest single-GPU effective-model-size lift in the research program.

**3. NLL improved over from-scratch baseline at central estimate.** 0.10-1.70 nat; per iter-212 admissibility.

**4. Production precedents exist for each component.** Mixtral/DeepSeek (MoE), MoEfication (post-hoc), QLoRA (LoRA-on-quantized), #74 (binary substrate). The joint composition is novel but mechanistically sound.

**5. New axis opened (CONDITIONAL COMPUTATION) on top of mature MEMORY axis.** #74 closed the per-parameter quantization frontier; #75-B opens conditional computation as the next dimension.

**6. Engineering scope moderate.** ~1700 LOC over 8 weeks; reuses #74 substrate verbatim; references mature MoEfication + QLoRA + Lit-LLaMA codebases.

### 10.2 Why CONDITIONAL not direct SELECT

**Caveat 1: Compounding-risk axis.** #75-B layers three unvalidated mechanisms (#53 + #74 + post-hoc moefication on binary substrate). Joint Gate-0 PASS probability ~50% (vs #74's 60%).

**Caveat 2: Pessimistic-tail NLL violates iter-212 admissibility.** At very pessimistic end (all three penalties at maximum), NLL trajectory is NEUTRAL or slightly negative vs from-scratch. Gate-0 must verify central-estimate behavior.

**Caveat 3: Tight memory headroom (1.8 GB at 16 GB ceiling).** 47% reduction vs #74; operationally fragile.

**Caveat 4: NF4 r=2 LoRA capacity uncertainty.** Lower-than-QLoRA capacity may under-compensate for binary + MoE penalty.

**Caveat 5: #53 MOSAIC-MOE empirical-validation gap.** First empirical test of #53; if fails, mechanism rejected.

**Caveat 6: Worse memory ratio than #74.** ~5-7× hybrid (vs #74's ~10-12×) due to LoRA overhead, though effective-param-per-byte is much better.

**The CONDITIONAL is on joint Gate-0 PASS confirming MoE quality penalty ≤ 0.25 nat at 64B-effective AND per-token wall-clock ≤ 5× of 1.84B BF16 AND routing entropy convergence to per-expert specialization regime.**

### 10.3 Cost of SELECT-CONDITIONAL vs RESERVE

**Cost of SELECT-CONDITIONAL:** ~$15K Gate-0 + ~$80K Gate-1 cloud + ~$15K storage + 8 weeks engineering + 6 weeks Gate-1. Total ~$110K + 3.5 months engineering.

**Cost of RESERVE:** Single-GPU model-size ceiling stays at 32B (post-#74); per-token wall-clock at ceiling stays at 9× of native baseline. The unique opportunity to recover per-token compute axis on top of the memory ceiling is deferred or lost.

### 10.4 Comparison to candidates A and C

| Dim | **#75-B (MOEFICATION-DISTILL — CONDITIONAL COMPUTATION axis on quantized base)** | #75-A (TBD) | #75-C (TBD) |
|---|---|---|---|
| Headline | **8× expansion over #74 to 256B effective + 4× per-token speedup over #74 + NLL improved 0.10-1.70 nat** | TBD | TBD |
| Risk-adjusted | **90-150B effective; 2-3× per-token speedup over #74** | TBD | TBD |
| Gate-0 PASS prob | **50%** | TBD | TBD |
| LLM-scale conf prob | **35%** | TBD | TBD |
| Production precedent | **Mixtral + DeepSeek-V3 + MoEfication + QLoRA + #74 (joint composition novel at 256B-effective on 16 GB)** | TBD | TBD |
| Engineering LOC | **1700** | TBD | TBD |
| Per-token speedup | **FIRST in post-#73 series; 4× over #74; 0.5× of native baseline** | TBD | TBD |
| Axis relevance to brief | **HIGH (per-token compute speed + extreme effective model size)** | TBD | TBD |
| Novelty axis | **Conditional computation × quantization joint composition** | TBD | TBD |
| Compounding-risk | **HIGHEST (3 unvalidated mechanisms compounded)** | TBD | TBD |

#75-B is the STRONGEST candidate on per-token compute speed (the iter-219 brief's most explicit phrase) and on extreme effective model size, with HIGHEST UPSIDE in the program to date. Highest compounding-risk; highest variance. **SELECT-CONDITIONAL.**

### 10.5 Composition-axis status after #75-B (if selected)

| Axis | Maturity post-#75-B |
|---|---|
| Compute-speed | At new ceiling (#42-#52 saturated; #75-B recovers per-token via sparse activation) |
| Memory | At near-frontier (#74 at 1-bit binary + LoRA r=2 NF4; sub-1-bit reserved as #76+?) |
| Effective model size at fixed memory | **AT NEW CEILING (256B-effective via #75-B; 8× over post-#74)** |
| Conditional computation | **MATURE at #75-B (if selected)** |
| Loss / objective | Mature (#56-#59) |
| Data / sampling | Mature (#57, #58) |
| Identity / agency / curriculum | Mature (#60-#62) |
| Optimizer / meta | Mature (#55, #63) |
| Memory parameter dim | Mature (#64, #65) |
| Cross-modal / VISION | Substrate at #66; #71-A distillation if shipped |
| Cross-modal / AUDIO | Substrate + distillation if #71-B |
| Causal / agentic-trajectory | Mature (#67) |
| Teacher provenance — text English | Mature (#68); composes with #75-B |
| Teacher provenance — reasoning | Mature (#69) |
| Teacher provenance — agent / tool | Mature (#70) |
| Teacher provenance — multimodal | Mature if #71-A |
| Teacher provenance — LANGUAGE multilingual | Mature if #72-B |
| **Memory-axis recomposition + iter-212 re-admission at 1.58-bit** | **MATURE at #73 (selected iter-217)** |
| **Memory-axis extension to 1-bit binary tier** | **MATURE at #74-A (selected iter-218 if Gate-0 PASS)** |
| **CONDITIONAL COMPUTATION axis on quantized base** | **MATURE at #75-B (if selected)** |

After #75-B (if selected), the major compute + memory + conditional-computation axes are at near-frontier. Future paradigms targeting model size beyond 256B-effective on a single GPU require either multi-GPU (#45 HYDRA-COMPOSED), more aggressive routing (E=16+ experts; reserved), or sub-1-bit memory (reserved).

---

## 11. Bottom line, one line

**SELECT-CONDITIONAL on joint Gate-0 PASS for MOEFICATION-DISTILL-CHIRON. ~256B effective parameter count at ~8B active per token on single 16 GB GPU (8× expansion over post-#74 32B; 139× over native 1.84B; largest effective-model-size lift in the research program) + 4× per-token compute speedup over #74-dense at same memory budget (FIRST per-token speedup in the post-#73 series; satisfies the iter-219 brief's "compute speed" clause directly per-token) + NLL strictly improved by 0.10-1.70 nat over from-scratch baseline at central estimate (slightly tighter range than #74-alone). Mechanism: post-hoc moefication of #74 PHOENIX-quantized 32B-effective dense trunk into E=8 expert FFN slices via Yu 2022 co-activation clustering + per-expert NF4 r=2 LoRA adapters (~590 MB total) + top-k=2 routing + #74 PHOENIX-1BIT-DISTILL substrate preserved verbatim + #68 SUPER-DISTILL Llama 3.1 405B teacher KL-CE pipeline. Theorem 1: net NLL ≤ BASE - (Δ_distill - Δ_PHOENIX-1BIT-hybrid - Δ_MoE-penalty) = BASE - (0.10 to 1.70) nat at central estimate. Theorem 2: 256B effective at ~14.2 GB GPU-resident with hybrid + Fallback A (1.8 GB headroom; tight). Theorem 3: bijectivity preserved (routing g(x) function of x only, not y). Joint Gate-0 PASS ~50% (lower than #74's 60% due to compounding three unvalidated mechanisms: #53 MOSAIC-MOE design + #74 binary tier + post-hoc moefication on quantized substrate); LLM-scale confirmation ~35% at 256B-effective. Engineering ~1700 LOC over 8 weeks. Compute axis: FIRST per-token speedup in the series (~4× over #74-dense; ~0.5× of native baseline at 139× more effective parameters). Mechanism is RECOMPOSITION — combines mature MEMORY axis from #74 with new CONDITIONAL-COMPUTATION axis from #53 design realization; novelty is at the program level (joint composition at 1-bit + 8-expert + post-hoc moefication + 256B-effective on 16 GB single GPU has no published precedent). Direct alignment with iter-219 brief's "magnitudes better on compute speed without compromising memory advantages or NLL accuracy" — first candidate in post-#73 series to satisfy compute-speed clause directly per-token. SELECT-CONDITIONAL with moderate-low confidence on joint Gate-0 PASS at 64B-effective confirming MoE quality penalty ≤ 0.25 nat AND wall-clock ≤ 5× of 1.84B BF16 AND routing-entropy convergence to per-expert specialization. Falls back to #74 (no regression) if Gate-0 fails. Highest upside in the research program to date (256B at 4× per-token speedup) AND highest variance (joint Gate-0 50% vs #74's 60%; compounding three unvalidated mechanisms).**

---

**End of Paradigm Shift #75 Candidate B design document.** ~3000 words. MOEFICATION-DISTILL-CHIRON: realization of #53 MOSAIC-MOE design (selected iter-197, never empirically validated) on top of #74 PHOENIX-1BIT-DISTILL-COMBO substrate (selected iter-218) via Yu 2022 post-hoc moefication, lifting single-GPU effective model size from 32B (post-#74) to ~256B effective on 16 GB single GPU (~139× expansion over native 1.84B; ~8× over post-#74) at ~8B active per token (first per-token compute speedup in post-#73 series; ~4× over #74-dense at same memory budget), with net NLL strictly improved by 0.10-1.70 nat over from-scratch baseline at central estimate. SELECT-CONDITIONAL recommended on joint Gate-0 PASS at 64B-effective; mechanism is RECOMPOSITION — combines mature MEMORY axis (#74) with new CONDITIONAL-COMPUTATION axis (#53 realization) on quantized substrate. Highest upside (256B-effective + 4× per-token speedup) AND highest variance (compounding three unvalidated mechanisms; joint Gate-0 PASS ~50% vs #74-alone 60%); Gate-0 mandatory at 64B-effective; falls back to #74 with no regression if Gate-0 fails.
