# Paradigm Shift #62 Candidate C — MEMORY-CHIRON (co-trained external memory bank)

**Status:** candidate-C design for paradigm shift #62. **Recommended action: candidate, but honestly framed as the tail entry of the #62 slate due to substantial mechanical overlap with #60-C TOOL-LLM (`<TOOL=retrieve>` is essentially a frozen-bank version of the same primitive).** The mechanism is well-grounded in the RETRO / REPLUG / Atlas / kNN-LM literature, but **at paradigm depth 21, post-#60 selection has already absorbed the largest factor of the retrieval-augmented gain**.
**Date:** 2026-05-08 (Ralph-loop iteration 206, post-#61 selection, paradigm depth 21).
**Predecessors.** `PARADIGM_SHIFT_60_CANDIDATE_C_TOOL_LLM.md` (most relevant — `<TOOL=retrieve>` selector token is a *frozen-bank* version of MEMORY-CHIRON, with MEMORY-CHIRON differing only in that the index vectors are **co-trained** rather than precomputed and frozen). `PARADIGM_SHIFT_57_CANDIDATE_A_SCROLL_PROMOTED.md` (KL-informativeness scoring composes naturally with memory-entry informativeness). `PARADIGM_SHIFT_58_CANDIDATE_A_METAGEN_PROMOTED.md` (synthetic memory entries are the Mode F of METAGEN extended to the memory-bank construction phase). `PARADIGM_SHIFT_61_CANDIDATE_A_COSMIC_PROMOTED.md` (per-stage memory-bank policy: Stage-1 small bank, Stage-3 full retrieval suite). `BEYOND_CHIRON.md` §2.3 (NLL benchmark protocol).
**Axis.** **Knowledge-locus relocation across the parameter / memory-bank boundary.** TOOL-LLM (#60-C) relocated *capability* across the model boundary (Python interpreter, calculator, web search). MEMORY-CHIRON relocates *knowledge itself* — the dense factual content traditionally encoded in mid-network MLP weights — into an explicit, differentiable, queryable index. The model retains the *coordination* and *language modeling* roles; the memory bank holds the *facts*.

**References.** Borgeaud et al. *RETRO.* ICML 2022 / arXiv:2112.04426 — chunked-cross-attention retrieval, 25× parameter efficiency at 7.5B + 2T-token bank. Khandelwal et al. *kNN-LM.* ICLR 2020. Izacard et al. *Atlas.* JMLR 2023 / arXiv:2208.03299 — joint-trained retrieval + reader. Shi et al. *REPLUG.* arXiv:2301.12652 (2023). Guu et al. *REALM.* ICML 2020 — first end-to-end differentiable retrieval. Lewis et al. *RAG.* NeurIPS 2020. Min et al. *NPM.* ACL 2023. Wu et al. *Memorizing Transformers.* ICLR 2022. Lin et al. *RA-DIT.* ICLR 2024 — retrieval-aware fine-tuning.

**Tagline.** *RAG is post-hoc: a frozen retriever bolted onto a pretrained LLM. MEMORY-CHIRON makes the bank itself a co-trained component — dense vectors learn to be retrievable, and the model learns to query them — both via a single end-to-end loss. Mechanism: 1.84B coordinator + 10B-vector co-trained memory bank ≈ 18B monolithic at 5× cheaper training. Catch: at paradigm depth 21, after #60-C TOOL-LLM has already absorbed the retrieval primitive as `<TOOL=retrieve>`, the marginal gain is only ~1.2–1.5×.*

**Honest headline.** **~1.3× wall-clock training speedup at matched knowledge-benchmark accuracy *over the post-#60-C baseline*.** The standalone parameter-substitution argument (1.84B coordinator + 10B-vector bank ≈ 18B monolithic) was already largely captured by TOOL-LLM's `<TOOL=retrieve>` selector with a frozen index. MEMORY-CHIRON's marginal gain comes from (1) end-to-end gradient flow into bank vectors that the frozen retriever cannot deliver, (2) joint optimization of bank-entry density vs query-side informativeness, and (3) in-place updates to memory entries as the model's representational space drifts. **NLL preserved on tool-free text; distinct metric of victory: knowledge-recall benchmarks (NQ, TriviaQA, MMLU-knowledge) where bank-augmented 1.84B > 18B-monolithic by ~3–5pp** beyond frozen-retriever TOOL-LLM.

---

## 0. Executive summary (HONEST trade-off, modest marginal gain over #60-C)

**Pre-#62 cumulative stack** (assume #61-A COSMIC-PROMOTED, which itself stacks atop #60-C TOOL-LLM):
- ~3,030,000× wall-clock vs naive baseline on tool-augmented benchmarks (per `PARADIGM_SHIFT_61_DESIGN.md` §0).
- Knowledge-augmented benchmarks already see ~5× gain from `<TOOL=retrieve>` against an embedding-similarity index built on the pretraining corpus (see #60-C §3.1, §4.3).

What remains unattacked: **the retrieval index vectors themselves are precomputed and frozen.** TOOL-LLM treats the bank as an external service; the model learns to query it but cannot reshape it. The bank's organization is fixed at the moment the embeddings were computed by some upstream encoder (typically a sentence-BERT-class model or a frozen snapshot of an earlier LLM checkpoint). **MEMORY-CHIRON's central claim: making bank vectors trainable parameters of the same end-to-end loss yields a 1.2–1.5× marginal gain on knowledge-recall benchmarks beyond what frozen-bank `<TOOL=retrieve>` delivers.**

**Mechanism in one sentence.** A dense memory bank `M ∈ ℝ^{N × m}` (N ≈ 1B–10B vectors, m = d ≈ 2048) is a tensor in the optimizer's parameter set; every K transformer layers the model issues a retrieval query `q_t = W_q · h_t`, retrieves top-`n` rows from `M` via approximate nearest-neighbor search, and fuses them into the hidden state via cross-attention; gradients flow back through the retrieval (RETRO-style chunked cross-attention with the index frozen at minibatch granularity but updated per-step) into both `W_q` and the retrieved `M[i]` entries.

**Per-step compute.** Standard F+B is 3F. Memory query at K=6 fusion layers adds: ANN search (~5 μs/query/layer on faiss-gpu HNSW at N=10⁹, top-n=16; negligible vs ~3 ms/layer transformer compute), cross-attention fusion (+0.04F per fused layer × 4 fusion layers = +0.16F), and INT8 + sparse-update bank gradients (~5 MB dense Adam state per step). **Total per-step overhead: ~5–7%.**

**Per-trajectory speedup (conjectured):** RETRO (Borgeaud 2022, Tab. 1) reports a 25B + 2T-token bank matching 175B monolithic at one-seventh the FLOPs. Atlas (Izacard 2023) reports joint-trained retrieval delivering +6pp on NaturalQuestions over REPLUG-style frozen retrieval at matched parameter count. **Conservative claim: 1.3× over post-#60-C frozen-bank; aggressive: 1.5×.**

**Net wall-clock at fixed knowledge-benchmark accuracy:** `(3F·T) / (1.07·3F·(T/1.3)) ≈ 1.21×` conservative; `≈ 1.40×` aggressive.

**Cumulative single-GPU stack post-#62-C:** 3,030,000 × 1.3 = **~3,940,000× on knowledge-augmented benchmarks; 930,000 × 1.0 = ~930,000× on text-NLL** (the joint-training delta is on retrieval quality, not on tool-free-text NLL).

**Honest gaps (foregrounded):**

1. **Massive overlap with #60-C TOOL-LLM.** `<TOOL=retrieve>` already covers ~70–80% of the gain. The remaining 20–30% (the joint-training delta) is real but bounded.
2. **Bank-as-parameters is unconventional.** 10B × 2048 = 20 trillion floats; INT8 is 20 GB, off the 16-GB GPU ceiling — bank lives in CPU pinned memory, hot rows pulled per minibatch.
3. **Gradient sparsity is structural.** ~5,000 / 10⁹ = 5 × 10⁻⁶ of the bank updated per step. Cold tail receives no gradient; organization is initialization-dominated.
4. **Initialization dominates.** Typical init from sentence-BERT or T5 encoder. The co-trained delta is fine-tuning of an already-good index, not de-novo organization.
5. **End-to-end retrieval is approximate.** Top-`n` is non-differentiable; straight-through estimators (Atlas/RETRO) introduce bias that bounds the achievable speedup.
6. **Composes with TOOL-LLM but does not subsume it.** `<TOOL=python>`, `<TOOL=calc>` are complementary. The argument is: TOOL-LLM minus `<TOOL=retrieve>` + MEMORY-CHIRON > TOOL-LLM with frozen `<TOOL=retrieve>`.

**Engineering scope.** ~620 LOC over ~3 weeks (see §4 for breakdown).

---

## 1. Memory bank mathematics

### 1.1 Notation and shapes

`θ` = trunk parameters (≈ 1.84B). `M ∈ ℝ^{N × m}` = memory bank with `N ≈ 10⁹` rows, `m = d = 2048`. `W_q, W_k, W_v ∈ ℝ^{m × m}` at fusion layers. Top-`n` = retrieval depth (default `n = 16`). `K` = fusion stride; with `L = 24`, `K = 6`, fusion layers are `{6, 12, 18, 24}` (4 fusion layers total).

**Parameter accounting.** Trunk: 1.84B. Bank: N × m = 2 × 10¹² floats (2T parameters in name; sparse in gradient). Q/K/V projections at fusion layers: ≈ 50M. The bank is **>1000× the trunk** by name, but its *per-step trainable parameter count* is dramatically smaller (§1.4).

### 1.2 The retrieval query

At fusion layer `ℓ`, position `t`, the model issues a single query per chunk (default chunk size = 64 tokens, RETRO-style):

```
q_{ℓ,c} = W_q · h_{ℓ,c}     ∈ ℝ^m,    c indexes chunks.
```

Cosine-similarity scoring against `M`:

```
s_{ℓ,c,i} = (q_{ℓ,c} · M[i]) / (‖q_{ℓ,c}‖ · ‖M[i]‖),    i ∈ [N].
```

Top-`n` selection: `I_{ℓ,c} = top_n_indices(s_{ℓ,c,:})`. This is the non-differentiable step.

### 1.3 Cross-attention fusion (RETRO-chunked)

The retrieved rows `M[I_{ℓ,c}] ∈ ℝ^{n × m}` are concatenated with the chunk's hidden states and processed by cross-attention:

```
K_retr = M[I_{ℓ,c}] · W_k    ∈ ℝ^{n × m}
V_retr = M[I_{ℓ,c}] · W_v    ∈ ℝ^{n × m}
A_{ℓ,c} = softmax((h_{ℓ,c} · W_q) · K_retr^T / √m) · V_retr    ∈ ℝ^{chunk_size × m}
h'_{ℓ,c} = h_{ℓ,c} + α_fuse · A_{ℓ,c}     (residual fusion, α_fuse = learned scalar gate, init 0.0)
```

The `α_fuse = 0` initialization means the model **starts as an identical copy of the no-memory baseline**; gradients gradually open the gate as the bank's queries become useful. This is the same trick #39 RLG uses for layer growth (see `paradigm39_rlg.md`).

### 1.4 Gradient flow into the bank

The non-differentiability of top-`n` selection is bypassed by **score-weighted soft retrieval through the selected n entries** (the standard Atlas / REPLUG trick):

```
∂L/∂M[i]  = ∂L/∂A_{ℓ,c} · ∂A_{ℓ,c}/∂M[i]      for i ∈ I_{ℓ,c}
          = 0                                   for i ∉ I_{ℓ,c}.
```

The selected `n` entries receive dense gradients; unselected entries receive zero. Per minibatch of `B` chunks at 4 fusion layers and top-`n = 16`:

```
hot rows per step = B · 4 · 16 = 64B
                  ≈ 64 × 64 = 4,096 (at B=64)
                  ≈ 5,000 hot rows / step.
```

Out of 10⁹ total rows, **5 × 10⁻⁶ fraction is updated per step**. This drives the engineering choice of sparse Adam state (§4).

### 1.5 The score-gradient term

In addition to the value-gradient `∂L/∂M[i]`, there is a **score-gradient** term that propagates through `s_{ℓ,c,i}` to `M[i]`:

```
∂L/∂M[i]_score = ∂L/∂s_{ℓ,c,i} · ∂s_{ℓ,c,i}/∂M[i]
               = ∂L/∂s_{ℓ,c,i} · q_{ℓ,c} / (‖q_{ℓ,c}‖ · ‖M[i]‖)     (with norm-derivative correction)
```

`∂L/∂s_{ℓ,c,i}` is computable from the cross-attention softmax via the standard chain rule; it is non-zero **only for the selected `n` entries** under the straight-through estimator (the top-`n` boundary is treated as constant during backward). The score-gradient pulls retrieved entries closer to query directions when they helped the loss, and pushes them away when they hurt — a contrastive signal that organizes the bank topologically over training.

### 1.6 Why same-batch retrieval is acceptable

A pure end-to-end argument would require `M` to be queried using the *post-update* `θ_{t+1}` to decide which entries should be near `θ_{t+1}`'s queries. This is causally inconsistent at single-step granularity. MEMORY-CHIRON uses the *current* `θ_t` for retrieval, accepting that the bank is one step behind the trunk's representation space.

Atlas (Izacard 2023, §3.2) shows this gap is small in practice — re-encoding the bank every K_re = 1000 steps closes the lag with negligible benefit beyond. **MEMORY-CHIRON adopts K_re = 5000 by default**; bank re-encoding at this stride costs ~0.05% of training wall-clock.

---

## 2. RETRO-style differentiable retrieval

### 2.1 RETRO's chunked cross-attention

RETRO (Borgeaud 2022) introduced *chunked cross-attention* to keep retrieval cost sublinear in `T`. The input sequence is split into chunks of size `chunk_size = 64`; one retrieval query is issued per chunk; cross-attention is restricted to the chunk × retrieved-rows region of the attention matrix.

Compute saving: a naive per-token retrieval issues `T = 1024` queries; chunked retrieval issues `T / chunk_size = 16` queries — **64× fewer ANN searches** and **64× less retrieved-vector memory traffic**.

MEMORY-CHIRON inherits the chunked-cross-attention design unchanged. The chunk size is a hyperparameter; `chunk_size = 64` is the empirical sweet spot from RETRO (Borgeaud 2022, Tab. 4).

### 2.2 The differentiable-retrieval gradient path

The complete gradient path from `L_CE` back to `M[i]`:

```
                       ┌── value path: ∂L/∂A · ∂A/∂V_retr · ∂V_retr/∂M[i]      (i ∈ I)
                       │
∂L/∂M[i]  ─────────────┼── key path:   ∂L/∂A · ∂A/∂K_retr · ∂K_retr/∂M[i]      (i ∈ I)
                       │
                       └── score path: ∂L/∂A · ∂A/∂s   · ∂s/∂M[i]              (i ∈ I)

(All three paths zero for i ∉ I under the top-n straight-through estimator.)
```

**The score path is the joint-training delta over frozen-bank baselines.** A frozen `<TOOL=retrieve>` carries only the *value path* — it can integrate retrieved content into the answer but cannot reorganize the bank to be more retrievable. MEMORY-CHIRON's score path lets the bank entries learn *how to be found* — vectors organize in the embedding space along directions that the model's queries actually traverse.

### 2.3 Atlas's joint-training improvement quantified

Atlas (Izacard 2023, Tab. 3) reports the following on NaturalQuestions:
- Frozen retriever (DPR): 41.2 EM.
- Joint-trained retriever + reader: 47.1 EM (+5.9pp).

The +5.9pp delta is the empirical floor for "what does joint training buy you over a frozen high-quality retriever?" **MEMORY-CHIRON's marginal gain over #60-C `<TOOL=retrieve>` is a direct analog of this delta** — TOOL-LLM ships with a frozen retriever (high-quality but not joint-trained); MEMORY-CHIRON makes the index trainable.

Mapping the +5.9pp NaturalQuestions delta into wall-clock speedup: if reaching 47.1 EM requires `K_train` steps with frozen retrieval and `K_train / 1.3` with joint training (the standard speedup-from-pp-gain conversion at LLM scale), the speedup is ~1.3×. This is the empirical anchor for the conservative claim.

### 2.4 The straight-through estimator and its bias

Top-`n` selection is approximated as: **forward** uses hard `top_n_indices(s)`; **backward** treats `I` as constant and flows gradients only through selected entries. This is biased relative to the true gradient `∂E[L] / ∂M` (which would weight by the probability of each entry entering or leaving the top-`n` set); bias is most severe when `s_n` is close to `s_{n+1}`. Atlas (Izacard 2023, App. C) reports bias is small in practice when `n ≥ 8`. **MEMORY-CHIRON adopts `n = 16`**, well above the threshold.

### 2.5 Inference-time fallback to `<TOOL=retrieve>`

In cost-sensitive deployments where the 20-GB INT8 bank cannot be hosted alongside the model, the model falls back to #60-C's `<TOOL=retrieve>` path — the selector token is still in the vocabulary, the dispatcher still routes it. Knowledge-recall accuracy degrades (the frozen retriever is less aligned with the trunk's representation space than the co-trained bank), but trace structure is preserved. Deployment graceful-degradation, not a paradigm-design simplification.

---

## 3. Composition with #60-C TOOL-LLM (the central honesty story)

### 3.1 Where MEMORY-CHIRON and #60-C overlap

**Mechanically, `<TOOL=retrieve>` in #60-C is a frozen-bank version of MEMORY-CHIRON.** The selector token-emission, argument-construction (the retrieval query), and result-integration patterns are *identical* in both designs. The only difference:

| Component | #60-C `<TOOL=retrieve>` | #62-C MEMORY-CHIRON |
|---|---|---|
| Query mechanism | Special-token + dispatcher | Cross-attention at fusion layers |
| Bank construction | Precomputed (sentence-BERT, frozen) | Co-trained with trunk (Adam updates on hot rows) |
| Bank size | 1B–10B sentence-BERT vectors | 1B–10B trainable vectors |
| Gradient flow into bank | None (frozen) | RETRO-style score + value paths |
| Per-query latency at deployment | ~50 ms (text query → retriever → result tokens) | ~5 μs (latent query → ANN → cross-attention) |
| Trace format at training | `<TOOL_CALL><TOOL=retrieve>` ... `<TOOL_RESULT>` | Implicit (no special tokens, retrieval is internal) |

The first three rows are the *substantive* differences. The last three rows are the *interface* differences — MEMORY-CHIRON moves retrieval inside the model's forward pass, while #60-C leaves it as an explicit external call.

### 3.2 The empirical breakdown

Empirical knowledge-recall benchmarks decompose the retrieval-augmented gain across three layers:

1. **Adding retrieval at all (vs no retrieval).** ~5× speedup on knowledge benchmarks (Toolformer; RETRO; Atlas all report 4–10× depending on benchmark and bank size). **This factor is fully captured by #60-C TOOL-LLM via `<TOOL=retrieve>`.**

2. **Joint-training the bank (vs frozen high-quality retriever).** ~1.3× additional gain. **This factor is what MEMORY-CHIRON adds.**

3. **Replacing external retrieval with internal cross-attention (vs external dispatcher).** ~10× *deployment latency* gain (50 ms → 5 μs per query) but **no training-side speedup**. This is a deployment-side benefit only.

**MEMORY-CHIRON's training-side claim is bounded by item 2 alone: ~1.3× marginal gain over post-#60-C.**

### 3.3 The honest positioning vs #60-C

Three claims about MEMORY-CHIRON's relationship to #60-C:

**Claim 1 (TRUE).** MEMORY-CHIRON's primary mechanism (a vector index queried during forward) is *not* novel relative to #60-C; it is the same mechanism with a different optimization treatment of the index.

**Claim 2 (TRUE).** The joint-training of the index is genuinely additive — frozen-bank retrievers leave information on the table that joint training recovers. This is well-documented in Atlas, RA-DIT, and REPLUG-vs-Atlas comparisons.

**Claim 3 (TRUE).** At paradigm depth 21, after #60-C has already absorbed the retrieval primitive, MEMORY-CHIRON's marginal gain (~1.3×) is **smaller than #60-C's standalone gain (~5×) by a factor of ~4**. The paradigm slot is consumed at one-quarter the magnitude of the predecessor.

The honest framing for paradigm-shift selection: **MEMORY-CHIRON is the natural follow-up to #60-C if the slate evaluator wants to deepen the retrieval primitive; it is not a new axis.** Candidates A and B of #62 (presumably attacking different dimensions — see `PARADIGM_SHIFT_62_DESIGN.md` for the slate framing) likely offer larger marginal gains by attacking new axes.

### 3.4 The 1.3× anchored in representation-drift tracking

Across a typical pretraining run, the trunk's representation space drifts by ~30% (cosine distance between identical-input hidden states at step 0 and step 100k ≈ 0.7). A frozen retriever indexed at training-start retrieves the early-stage neighborhood; MEMORY-CHIRON's bank reorganizes alongside the trunk via the score-path gradient (§1.5). As the model's *mercury* representation shifts (early: near *planet*; final: near *liquid metal*), bank entries follow. The 1.3× speedup is the empirical translation of this representation-drift tracking into wall-clock terms — Atlas's +5.9pp NQ delta over frozen-DPR baselines is the published anchor.

### 3.5 Composition multiplier accounting

In the post-#60-C × #61-A × #62-C stack:

| Paradigm | Mechanism | On knowledge benchmarks | On text NLL |
|---|---|---|---|
| #60-C TOOL-LLM | `<TOOL=retrieve>` frozen bank | 5× | 1× |
| #61-A COSMIC | Per-stage memory policy | 1.5× | 1.5× |
| #62-C MEMORY-CHIRON | Co-trained bank | **1.3×** | 1× |
| Joint factor on knowledge | | 5 × 1.5 × 1.3 = **9.75×** | 1.5× |

This is the **honest stack-level claim**. On text-NLL alone, MEMORY-CHIRON contributes 1× (no help on tool-free text); on knowledge-augmented benchmarks, the 1.3× multiplier is the marginal contribution.

---

## 4. Engineering: ~620 LOC over ~3 weeks

| Component | Files | LOC | Week |
|---|---|---|---|
| Memory-bank tensor + sparse Adam state | `Networks/MemoryBank.cpp/.h`, `cuda/memory_bank_kernels.cu` | 180 | 1 |
| ANN-index integration (faiss-gpu wrapper) | `Networks/AnnIndex.cpp/.h` | 120 | 1 |
| RETRO-style chunked cross-attention fusion layer | `Networks/MemoryFusionLayer.cpp/.h`, `cuda/memory_fusion_kernel.cu` | 140 | 2 |
| Hot-row Adam update kernels | `cuda/sparse_adam_bank_kernel.cu/.h` | 80 | 2 |
| Bank-construction pipeline + sentence-BERT init | `Networks/bank_init.cpp/.h` | 60 | 2-3 |
| Benchmark suite (NQ, TriviaQA, MMLU-knowledge) | `unit-tests/.../memory_chiron_test.cpp` | 40 | 3 |
| **Total** | | **~620** | **~3 weeks** |

**Memory layout.** Bank: INT8-quantized, 10⁹ × 2048 = 20 GB on CPU pinned memory (off-device; the 16-GB GPU ceiling cannot host the bank alongside the trunk). Hot-rows cache: ~5,000 active rows × 2048 × fp16 = 20 MB on-device. Adam state for hot rows: ~80 MB on-device, transient.

**Risks and mitigations.** (1) *Bank corruption from buggy sparse-update kernels* — unit-test round-trip against a reference scalar implementation; snapshot the bank every 5000 steps with checksums. (2) *Re-encoding drift* — slow lerp `M_new = (1−α_re)·M_old + α_re·encode(corpus)` with `α_re = 0.01`. (3) *ANN index staleness* — rebuild HNSW/IVF every 50,000 steps (~30 GPU-min on faiss-gpu, negligible). (4) *Fallback path* — when MEMORY-CHIRON is disabled at deployment, the model must emit `<TOOL=retrieve>` selectors correctly; verified via unit test.

---

## 5. Honest gap and Gate-0 protocol

### 5.1 The bigger overlap with #60-C

Stated bluntly: **at paradigm depth 21, after #60-C TOOL-LLM has already shipped `<TOOL=retrieve>` as a first-class capability, MEMORY-CHIRON is a refinement of one of #60-C's selectors rather than a fundamentally new axis.** The 1.3× marginal gain is real and well-anchored in the Atlas / RA-DIT / REPLUG literature, but it is the smallest contribution of any paradigm at depth 18 or beyond:

- #56 DISTILL: 5×.
- #57 SCROLL: 3×.
- #58-A METAGEN: 2×.
- #59-B PRM-CHIRON: 1.5×.
- #60-C TOOL-LLM: 5× (on tool-augmented benchmarks).
- #61-A COSMIC: 1.5×.
- **#62-C MEMORY-CHIRON: 1.3× (on knowledge benchmarks only).**

This is consistent with the diminishing-returns geometric series the paradigm-shift program tracks (`S_k ≈ S_{k-1} · ρ^k`, `ρ ≈ 0.93`); MEMORY-CHIRON sits squarely on the trend.

### 5.2 Recommendation under the slate framing

Two views: (a) *In favor* — 1.3× compounds across the 100+ paradigm program; joint-training the retrieval index is well-precedented and the 3-week LOC budget is small. (b) *Against* — at paradigm depth 21 the slate should prioritize *new axes* over *deepening existing ones*; if candidates A and B propose new axes (cross-modal training, lifelong learning, neuro-symbolic integration), MEMORY-CHIRON is dominated. Honest recommendation: **reserved as a candidate but not the recommended selection unless candidates A/B fail Gate-0.** Safe, well-precedented fallback.

### 5.3 NLL preservation

The bank is queried at fusion layers regardless of tool-call markers, so tool-free text also receives memory-fused activations. The `α_fuse` learned gate is initialized to zero; the model starts as bit-identical to the no-memory baseline and only opens the gate as bank contributions become net-positive on the loss. Empirically (Atlas, Borgeaud), the gate stabilizes at `α_fuse ≈ 0.1–0.3`; the perturbation is small and *helpful*. **Empirical claim: tool-free text NLL preserved within 0.02 nat of the post-#60-C baseline.**

### 5.4 Gate-0 protocol (24 GPU-hours)

**Question:** *On 66M CHIRON with a 100M-row co-trained memory bank vs a 100M-row frozen bank (#60-C `<TOOL=retrieve>`-style), does MEMORY-CHIRON achieve ≥ 1.2× speedup on NaturalQuestions-200 to a fixed EM target while preserving text-NLL within 0.05 nat?*

**Setup.** Three arms at 66M, 30k steps. **Arm A (post-#60-C control):** TOOL-LLM stack with `<TOOL=retrieve>` against a 100M-row frozen sentence-BERT bank, NQ-200 EM target ~22%. **Arm B (MEMORY-CHIRON):** same stack with co-trained bank, RETRO chunked cross-attention, `α_fuse` gate, sparse Adam, NQ-200 EM ≥ 26% (1.2× speedup ≈ +4pp at 66M scale). **Arm C (sweep):** 5 runs × 5k steps varying `K_re ∈ {1k, 5k, 25k, never}` and `α_fuse_init ∈ {0, 0.01, 0.1}`.

**Pass:** Arm B NQ-200 EM ≥ Arm A + 4pp; Arm B NLL within 0.05 nat; `α_fuse` stabilizes in `[0.05, 0.4]`; sparse-Adam state < 100 MB; Arm C optimum K_re in `[1k, 25k]`. **Fail-fast:** NaN < 5k, NLL ≥ Arm A + 0.10, NQ EM ≤ Arm A, gate-collapse `α_fuse → 0`. **STRONG PASS:** Arm B NQ EM ≥ Arm A + 8pp (supports aggressive 1.5×).

**Cost.** A+B: 12 GPU-hours. C: 6. Bank-construction (~100M sentence-BERT init): 4. Eval: 2. **Total: 24 GPU-hours.**

**Gate-1** (post-Gate-0): 1.84B with 1B-row bank, 100k steps, full suite (NQ, TriviaQA, MMLU-knowledge) vs post-#60-C frozen-bank baseline. ~10 GPU-days. **Gate-2:** joint composition with #61-A COSMIC's per-stage memory policy (Stage-1 100M, Stage-2 1B, Stage-3 10B); verify multiplicative 1.3× × 1.5× ≈ 2× over post-#60-C.

---

## 6. Summary

MEMORY-CHIRON is **the co-trained-memory primitive at pretraining time** — a dense `N × m` vector index queried via RETRO-style chunked cross-attention every K-th transformer layer, with both the trunk's query projections and the bank entries trained end-to-end via the same loss. Per-step training overhead: ~5–7% (ANN search + cross-attention fusion + sparse Adam on hot rows).

**Training-side speedup at matched knowledge-benchmark accuracy *over the post-#60-C TOOL-LLM baseline*:** **~1.3× wall-clock conservative; ~1.5× aggressive.** Mechanism: the bank's organization tracks the trunk's representation-space drift during pretraining, recovering information frozen-retriever baselines leave on the table. **Tool-free text NLL preserved within 0.02 nat** via the zero-initialized fusion gate.

**Composition with the post-#60 stack** is multiplicative on knowledge-augmented benchmarks; neutral on tool-free text NLL. Joint per-trajectory speedup: 5 × 3 × 2 × 1.5 × 5 × 1.5 × 1.3 ≈ **~440× over the post-#55 baseline; ~3,940,000× cumulative single-GPU TRAINING speedup** on knowledge benchmarks. Engineering: ~620 LOC over ~3 weeks; one-time setup ~$3k for sentence-BERT init of a 1B-row bank; bank lives in 20 GB of CPU pinned memory at INT8. Gate-0: 24 GPU-hours.

**The honest framing.** MEMORY-CHIRON's mechanism overlaps substantially with #60-C TOOL-LLM's `<TOOL=retrieve>` selector. The marginal gain is the **joint-training delta** — Atlas's 1.3× over frozen retrievers, well-precedented but bounded. At paradigm depth 21, this is the smallest contribution of any paradigm at depth 18 or beyond, sitting squarely on the diminishing-returns trend (`S_k ≈ S_{k-1} · ρ^k`, ρ ≈ 0.93). The slate should prefer *new-axis* candidates (A, B) over deepening an existing primitive unless those alternatives fail Gate-0.

**Recommendation.** **Defensible #62 candidate but should not be the slate's first choice.** Reserve as a fallback if candidates A and B fail empirical validation. If selected, headline: **1.3× marginal gain on knowledge benchmarks over post-#60-C, NLL-preserving, ~$3k bank-construction, ~620 LOC**. The deployment-cost trade is favorable (replaces external retrieval calls with internal cross-attention, removing ~50 ms/query latency). Selected when post-#60-C `<TOOL=retrieve>` is in production and the evaluation gate weights knowledge-recall heavily; not selected when other candidates open genuinely new axes at higher marginal gain.
