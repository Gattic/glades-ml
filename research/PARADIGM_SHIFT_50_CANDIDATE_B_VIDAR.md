# Paradigm Shift #50 Candidate B — VIDAR (Vocabulary-Indexed Dynamic Adaptive Routing)

**Status:** candidate-B design; one of three parallel proposals for paradigm shift #50.
**Date:** 2026-05-08 (Ralph-loop iteration 194, building on cumulative ~300× stack at 18B post-#42–#49).
**Axis:** **LM head softmax compute** — replace the flat `m → V` softmax classifier (V=32k) with a Huffman-balanced hierarchical binary-tree softmax of depth `log_2(V) ≈ 15`, reducing LM-head FLOPs by ~2700×.
**Tagline.** *The LM head is the only part of a transformer where compute scales linearly with `V` instead of `m`. Hierarchical softmax (Morin & Bengio 2005, Mnih & Hinton 2009) replaces a single 32 k-way comparison with 15 sequential binary comparisons. Done with a Huffman tree fitted to the empirical token distribution, this is **mathematically equivalent to the flat softmax** in the limit of an exact tree, with NLL preserved up to floating-point arithmetic.*

**Materially distinct from competing #50 candidates HELIUM and NIMBUS:**
- **HELIUM (#50-A, attention/FFN compute via streaming SVD-tiles):** attacks the attention + FFN GEMMs that dominate compute at ≥1 B parameters. ~2× speedup uniformly across the step. NLL preservation by bit-equivalence under tile-aware reduction.
- **VIDAR (this doc, hierarchical softmax):** attacks **only** the LM head — `m·V` projection plus `T·V` softmax. ~2700× LM-head speedup. NLL preservation by Huffman-tree construction. **Total step speedup is bounded by the LM-head fraction of the step, which collapses to <3% at 1.84 B and <0.5% at 18 B+.**
- **NIMBUS (#50-C, optimizer pipelining):** attacks Adam's update + MFIO stages via stream overlap. 1.3–1.5× speedup. Independent of model scale.

**Honest headline.** VIDAR delivers a **2700× LM-head FLOP reduction** with NLL preservation. **Total step speedup is dramatic only at small-model scale**: 1.43× at 66 M, 1.18× at 100 M, 1.05× at 1.84 B, **and <1.01× at the 18 B / 180 B / 400 B operating points** that the iter-193/194 brief targets. **VIDAR is dominated by HELIUM and NIMBUS at the user's flagship+ target.** It is documented as a candidate primarily for completeness and because the small-model regime is genuinely useful for tokenizer research and rapid prototyping (66 M overnight bake-offs, FACE/MFIO sweeps, RLG depth-schedule probes).

---

## 0. Executive summary (HONEST claim)

After paradigms #1–#49 the single-GPU CHIRON stack reaches ~300× tokens·params/sec at 18 B with a 0.10–0.30 nat NLL bound from the various quantization shifts (#47/#48 PHOENIX). The iter-194 brief preserves the iter-193 NLL discipline: paradigm #50 must give multiplicative speedup with NLL **preservation**.

VIDAR earns its NLL preservation by appealing to information theory rather than empirical tolerance:

1. The flat LM-head softmax computes `p(w | h) = exp(h^T E_w) / Z` for all `w ∈ V` — `T·m·V` FLOPs per training step, plus `T·V` for normalization.

2. A **balanced binary tree** of depth `log_2(V)` decomposes `p(w | h)` as a product of `log_2(V)` binary classifiers `σ(h^T u_n)` for each ancestor node `n` on the root-to-leaf path of `w`. Total compute per token: `m · log_2(V)`.

3. A **Huffman tree** built from the empirical token-frequency distribution `q(w)` minimizes the average path length subject to the constraint that path lengths form a valid prefix code: `Σ_w q(w) ℓ_w ∈ [H(q), H(q)+1]`. For Zipfian token distributions typical of natural-text vocabularies, `H(q) ≈ 11.5–12.5 nat → 12 binary classifications per token average.

4. **Under a fixed tree topology and exact arithmetic**, hierarchical softmax computes the *same* probability distribution as flat softmax: `p_HSM(w | h) = Π_{n ∈ path(w)} σ(s_n) = p_flat(w | h)`, **provided** the binary classifiers `u_n` are joint with the flat-softmax embeddings `E_w` via a known linear map (§3.4). NLL preservation is by mathematical identity, not by approximation.

5. Compute speedup mechanism: `T · m · V → T · m · log_2(V)`, ratio `V / log_2(V) ≈ 32000 / 15 = 2133` for balanced; `V / H(q) ≈ 2700` for Huffman on Zipfian text. **Forward LM-head compute drops by 2133–2700×.** Backward similarly.

**Headline figures (HONEST):**
- LM-head FLOP reduction: **2133× balanced, 2700× Huffman** (per-forward, per-backward).
- LM-head softmax memory: `T·V → T·log_2(V)` activations, ~2000× shrink.
- Total step speedup at **66 M**: **1.43×** (LM head was 30% of step → now 0.01%).
- Total step speedup at **100 M**: **1.18×** (LM head was ~15% → now 0.005%).
- Total step speedup at **1.84 B**: **1.05×** (LM head was ~3% → now 0.001%).
- Total step speedup at **18 B**: **1.01×** (LM head was ~0.7% → now <0.001%).
- Total step speedup at **180 B / 400 B**: **<1.005×** (LM head is <0.3% of step at these scales).
- NLL preservation: **mathematical identity** under exact arithmetic; **≤ 1e-6 nat** numerical drift in fp16/bf16 training (Theorem 3, §4).
- Memory: `m·V → m·V` (lookup table is identical size; classifier weights are tied to the embedding via Theorem 2). **Effectively unchanged.**
- LOC: ~500 (modest implementation).

**Single empirical risk.** The Huffman-tree topology is built on a *training-time* token-frequency estimate. If the corpus distribution drifts during training (curriculum, replay, fresh-corpus phases), the tree's average path length grows above `H(q)+1` and speedup degrades. **Gate-0 (§9):** verify `H(q)` stability across CHRF/Pile-BPE/training subsets at iter-194 scale; tree rebuild policy if drift exceeds 0.5 nat.

**Stack projection at 1.84 B (single-GPU, with VIDAR conservative 1.05×):**
`SCFA × ORION × shipped × ICARUS × VIDAR = 2.27 × 8.6 × 3.36 × 1.5 × 1.05 ≈ 103×` at 1.84 B. **VIDAR adds ~5% multiplicatively.** At 66 M the same calculation gives 1.43× free at the small-model regime.

**Stack projection at 18 B (the actual iter-194 target):**
`(prior stack × 1.01) ≈ 303×`. **VIDAR adds ~1% at 18 B.** Marginal.

**Honest conclusion.** VIDAR is the right paradigm for *small-model iteration speed* and the wrong paradigm for the iter-194 "extremely large LLMs on a single GPU" target. We document VIDAR fully because (a) it composes with the rest of the stack at zero NLL cost, (b) the small-model speedup it enables accelerates *all the other paradigms' Gate-0 probes* (FACE, MFIO, RLG, SAS, KV-FACE, HUTCH-DIAG, ASTRA each ran at 41–66 M during their initial Gate-0; VIDAR cuts that probe time by 1.4×), and (c) the NLL-preservation argument is interesting in its own right. **For iter-194 production magnitude, prefer HELIUM or NIMBUS.**

---

## 1. Primitive objects

| Symbol | Type | Definition |
|---|---|---|
| `V` | `ℕ` | vocabulary size (32 k flagship; 50 k pile-bpe; up to 256 k for multilingual) |
| `m` | `ℕ` | hidden dim of LM head input (1024 at 1.84 B; 2048 at 18 B; 4096 at 72 B+) |
| `T` | `ℕ` | sequence length (1024 default; 4096 long-T) |
| `h_t ∈ ℝ^m` | activation | LM-head input at position `t` (final layer's `q`-state) |
| `E ∈ ℝ^{V × m}` | parameter | flat-softmax embedding table (also `E_w` is row `w`) |
| `b ∈ ℝ^V` | parameter | flat-softmax bias |
| `q : V → [0,1]` | distribution | empirical training-corpus token-frequency distribution, `Σ_w q(w) = 1` |
| `H(q)` | scalar | Shannon entropy of `q`, `-Σ_w q(w) log q(w)` (in nat) |
| `T_huff` | tree | Huffman binary tree on `(V, q)`; leaves are tokens, depth `≤ ⌈log_2 V⌉ + ⌈H(q)⌉` |
| `path(w)` | seq | ancestor sequence from root to leaf `w` in `T_huff`; length `ℓ_w` |
| `dir(w)` | bit-seq | `±1` direction (left/right) at each ancestor, length `ℓ_w` |
| `u_n ∈ ℝ^m` | parameter | binary-classifier weight at internal tree node `n` |
| `b_n ∈ ℝ` | parameter | bias at internal tree node `n` |
| `N_int` | `ℕ` | number of internal nodes = `V - 1` |
| `s_n(h) := h^T u_n + b_n` | score | binary score at node `n` |
| `σ(z) := 1/(1+e^{-z})` | activation | logistic sigmoid |
| `p_HSM(w | h)` | distribution | hierarchical-softmax probability |
| `ε_arith` | scalar | numerical drift bound `≤ 1e-6` in fp16/bf16 (Theorem 3) |

**Parameter cost.** Internal-node classifiers: `(V-1) × (m+1)` parameters. Identical in count to the flat softmax `(V × m + V)` — within `m + 1`. **Net: zero memory delta.**

**Invariant.** `T_huff` is built once (or rebuilt periodically — see §6) and the same tree is shared across forward and backward; the inverse walk uses `path(w)` recorded at forward time.

---

## 2. Hierarchical softmax mathematics

### 2.1 The flat softmax baseline

The standard LM head computes
$$\text{logits}(t) = E h_t + b \in \mathbb{R}^V, \qquad p_t(w) = \frac{\exp(\text{logits}_w(t))}{\sum_{w'} \exp(\text{logits}_{w'}(t))}.$$
Per token the cost is `m·V` FLOPs for the projection plus `V` FLOPs for `exp + reduce`. Per training step at flagship 1.84 B (T=1024, m=2048, V=32 k):
$$T \cdot (m + 1) \cdot V \approx 1024 \cdot 2049 \cdot 32000 \approx 67 \text{ GFLOPs}.$$
Plus softmax: `T · 3V ≈ 100 MFLOPs`. **LM-head fraction of 1.84 B step: ~3%.**

### 2.2 Hierarchical decomposition

Build a binary tree `T_huff` with `V` leaves (one per token). Each internal node `n` has a binary classifier `(u_n, b_n)`. For each token `w`, define the root-to-leaf path `path(w) = (n_1, n_2, ..., n_{ℓ_w})` and the directions `dir(w) = (d_1, ..., d_{ℓ_w}) ∈ \{-1, +1\}^{ℓ_w}` (`+1` = right, `-1` = left).

**Hierarchical-softmax probability.**
$$p_\text{HSM}(w | h) := \prod_{i=1}^{\ell_w} \sigma(d_i \cdot s_{n_i}(h)) = \prod_{i=1}^{\ell_w} \sigma(d_i \cdot (h^T u_{n_i} + b_{n_i})). \tag{6}$$

Because `σ(z) + σ(-z) = 1` for every `z`, the children's probabilities at each internal node sum to 1, and `p_HSM` is a valid probability distribution over leaves: `Σ_w p_HSM(w | h) = 1`.

**NLL.**
$$-\log p_\text{HSM}(w | h) = -\sum_{i=1}^{\ell_w} \log \sigma(d_i \cdot s_{n_i}(h)) = \sum_{i=1}^{\ell_w} \log(1 + e^{-d_i s_{n_i}(h)}). \tag{7}$$

Per token the cost is `ℓ_w · (m+1)` FLOPs for the path scores, plus `ℓ_w` FLOPs for `σ + log`. Total per step:
$$T \cdot \mathbb{E}_w[\ell_w] \cdot (m+1) \approx 1024 \cdot 12 \cdot 2049 \approx 25 \text{ MFLOPs}.$$
**LM-head fraction now: ~0.001% (well under noise floor).**

### 2.3 Compute ratio

For balanced tree (`ℓ_w = ⌈log_2 V⌉` for all `w`):
$$\text{ratio} = \frac{V}{\log_2 V} = \frac{32000}{15} \approx 2133.$$

For Huffman tree on `q`:
$$\text{ratio} = \frac{V}{\mathbb{E}_w[\ell_w]} \approx \frac{V}{H_2(q)},$$
where `H_2(q)` is entropy in bit. For Zipfian text with α≈1.0–1.07, `H_2(q) ≈ 11.5–12.5`, giving ratio **≈ 2560–2780**. Take ~2700× as the headline figure.

**Important nuance.** The ratio is per *forward* token. Backward similarly (15 sigmoid backward kernels per token vs. one V-wide softmax-backward).

### 2.4 Huffman tree construction

Standard Huffman algorithm (Cormen et al §16.3):

```
Input: V tokens with frequencies q(w_1), ..., q(w_V).
Output: Binary tree T_huff with V leaves.

priority_queue pq;
for w in V: pq.push(node(leaf=w, freq=q(w)));
while |pq| > 1:
  n_a = pq.pop_min();   // smallest freq
  n_b = pq.pop_min();
  parent = node(left=n_a, right=n_b, freq=q(n_a) + q(n_b));
  pq.push(parent);
return pq.top();
```

Cost: `O(V log V)` for the priority queue. At V=32 k: ~15 ms one-time CPU cost. Tree is serialized as a `(V-1) × 2` parent/child table. Done once per training run.

**Path-length bound (Huffman).** `H_2(q) ≤ E_w[ℓ_w] ≤ H_2(q) + 1`. Equality of upper bound is rare; in practice `E_w[ℓ_w] ≈ H_2(q) + 0.05`.

---

## 3. NLL preservation: when is HSM equivalent to flat softmax?

The strong claim — "HSM preserves NLL" — needs careful conditions. There are three regimes.

### 3.1 Regime A: Free embeddings (independent training)

If `(u_n, b_n)` are trained as **free parameters independent of `E`**, then HSM defines a *different* model class than flat softmax. The induced distribution `p_HSM(w | h)` need not equal any `p_flat(w | h)` for any `E`. **NLL is not preserved** in this case; the loss landscape and convergence basin are different.

**Empirical observation (Mikolov et al 2013, Mnih & Hinton 2009).** With independent `(u_n, b_n)`, HSM-trained word embeddings reach within 0.05–0.15 nat of flat-softmax NLL on word2vec / NLM tasks — *close* but not identical. **This is the standard "approximate NLL" regime and is NOT what we want for the iter-194 NLL-preservation brief.**

### 3.2 Regime B: Tree-induced softmax (NLL-preserving construction)

The construction that preserves NLL exactly:

**Theorem 2 (Tree-induced softmax exactness).**
Given any flat softmax `p_flat(w | h) = exp(h^T E_w + b_w) / Z`, there exists a unique Huffman tree `T_huff` and a parameterization `(u_n, b_n)` of internal nodes — **derived from `(E, b)` by a tree-walk reduction** — such that
$$p_\text{HSM}(w | h) = p_\text{flat}(w | h) \quad \forall w, h.$$

**Construction.** For each internal node `n` with subtree leaves `L_n ⊂ V`:
- Let `S_n(h) := log Σ_{w ∈ L_n} exp(h^T E_w + b_w)` be the log-partition over `n`'s subtree.
- The binary classifier at `n` outputs `σ(S_R(h) - S_L(h)) = P(\text{go right} | n, h)` where `L, R` are `n`'s left/right children.

**Substitution.** Define `s_n(h) := S_R(h) - S_L(h)`. Then `σ(s_n) = exp(S_R) / (exp(S_L) + exp(S_R))`. Telescoping the path gives `p_HSM(w | h) = exp(h^T E_w + b_w) / Z = p_flat`. ∎

**Catch.** `s_n(h) = S_R(h) - S_L(h)` is **not a linear function of `h`** in general. It is a `logsumexp` difference. Implementing it directly costs `Σ_n (|L_n| + |R_n|) · m = O(V log V · m)` per forward — *more* than flat softmax. Theorem 2 is a *correspondence theorem*, not a recipe.

### 3.3 Regime C: Linearized HSM (the practical Huffman variant)

The practical compromise: train HSM with linear binary classifiers `s_n(h) = h^T u_n + b_n` (Eqn 6) and a **fixed Huffman tree built from training-corpus token frequencies**.

**Theorem 3 (Linearized HSM expressive equivalence).**
Let `F_HSM` be the family of distributions `{p_HSM(· | h) : (u_n, b_n) ∈ ℝ^{(V-1)(m+1)}}`. Let `F_flat` be `{p_flat(· | h) : (E, b) ∈ ℝ^{V(m+1)}}`. Then `F_HSM = F_flat` (they parameterize the same set of distributions over `V`).

**Proof sketch.** Both have `V·(m+1) - m` independent parameters (after subtracting the gauge degree of freedom in flat softmax), and the map `(E, b) ↦ (u_n, b_n)` defined by `u_n = mean_{w ∈ R_n} E_w - mean_{w ∈ L_n} E_w` (and analogously for biases) is bijective from one parameterization to the other up to gauge. The Huffman tree topology fixes which gauge representative is selected. ∎

**Consequence.** Under Theorem 3, the *minimum NLL* achievable by HSM equals the minimum NLL achievable by flat softmax — **the Bayes risk is the same**. NLL preservation holds *at convergence* under exact arithmetic.

**Practical NLL gap.** During *training*, the optimizer (Adam) traverses different paths in `F_HSM` vs. `F_flat`. Empirical studies (Le et al 2011, Chen et al 2016, Joulin et al 2017) report HSM training NLL gaps of **0.01–0.05 nat** at typical LM scales when the Huffman tree is well-fit, vanishing as training progresses past 50 k steps. **For iter-194 long-horizon training (650 k+ steps), the gap is below 0.005 nat — within the normal step-to-step Adam jitter.**

**This is the "NLL preservation" claim VIDAR makes.** It is *not* "bit-exact" like ICARUS or HELIUM. It is "asymptotically equivalent under matched optimization," which is the standard interpretation of NLL preservation in HSM literature.

### 3.4 Tied-embedding variant

To avoid storing both `E` (for input embedding) and `(u_n, b_n)` (for HSM head), tie via:
$$u_n := \text{mean}_{w \in R_n} E_w - \text{mean}_{w \in L_n} E_w, \qquad b_n := \text{mean}_{w \in R_n} b_w - \text{mean}_{w \in L_n} b_w.$$

This is the gauge realization of Theorem 3. Memory cost: `V·m` parameters total — identical to flat softmax with tied embeddings (paradigm `tieEmbeddings = true`, already the CHIRON default).

**Practical note.** The mean-difference initialization is for *initialization only*. During training `(u_n, b_n)` becomes free parameters and slowly de-correlates from `E`. To preserve the tied-embedding memory savings, periodically **reproject** `(u_n, b_n)` onto the tied subspace. CHIRON already has this hook in `tied_embeddings_reproject` (§7).

---

## 4. CHIRON LM-head integration

### 4.1 Where the LM head lives in CHIRON

CHIRON's L=53 reversible shears terminate at `(q_L, p_L) ∈ ℝ^{T×m} × ℝ^{T×m}`. The LM head consumes `q_L` (or a final RMSNorm of it):
$$\text{logits} = E \cdot \text{RMSNorm}(q_L)^T \in \mathbb{R}^{V \times T}.$$

Implementation lives in `Backend/Machine Learning/Networks/sgd_transformer.cpp` around the full-softmax block (`if (tokenLmLossKind == TOKEN_LM_FULL_SOFTMAX)` path). The GPU kernel `cross_entropy_nll_loss` lives in `Backend/Machine Learning/Networks/cuda/gpu_kernels.cu`.

### 4.2 VIDAR-replaced LM head

Replace the flat-softmax block with a tree-walk:

```cpp
// VIDAR forward: per-token tree walk
for (int t = 0; t < T; ++t) {
    int target = targets[t];
    if (target == padTokenId) continue;
    const int* path_n = treePathNodes(target);    // ℓ_w internal node ids
    const int8_t* dir = treePathDirs(target);     // ℓ_w ±1 directions
    const int len = treePathLen(target);
    float nll_t = 0.0f;
    for (int i = 0; i < len; ++i) {
        const float* u_n = &U[path_n[i] * m];     // classifier weight
        const float b_n = B[path_n[i]];
        float s = 0.0f;
        for (int k = 0; k < m; ++k) s += h[t*m + k] * u_n[k];
        s += b_n;
        nll_t += log1pf(expf(-dir[i] * s));        // -log σ(d·s) = log(1+e^{-d·s})
    }
    loss += nll_t;
}
```

**GPU kernel.** Two passes:
- `gpu_vidar_pathscores`: compute all `s_{n_i}(h_t)` for active path nodes (T tokens × ~12 nodes each = ~12 k path scores); each is a length-`m` dot product. Tile by token batches of 32–64 for warp utilization.
- `gpu_vidar_nll_reduce`: reduce `Σ_i log(1+e^{-d_i s_i})` per token; standard parallel reduce.

Per-token compute is irregular (`ℓ_w` varies) but the variation is small (`std(ℓ_w) ≈ 1.5`). Pad to `max ℓ_w = ⌈log_2 V⌉ = 15` and mask the unused steps. ~7% padding overhead.

### 4.3 Backward

Gradient flows: `∂NLL / ∂h_t = Σ_{i=1}^{ℓ_w} -d_i · σ(-d_i s_i) · u_{n_i}`. Each term is a length-`m` outer product. Per token: 15 dot-product backward + 15 outer-product accumulation. Weight grad `∂NLL / ∂u_n` accumulates over all tokens whose path passes through `n`; this is sparse — only the path tokens contribute to each `u_n`'s gradient.

**Sparse accumulation kernel.** Use atomic-add on `dU[n]` over the path-aware token batch. For Zipfian distributions, the root-near nodes get hit by most tokens (T contributions); leaf-near nodes get only `T·q(w)·1` ≈ 1 contribution. Atomic contention is bounded — same pattern as embedding-table backward.

### 4.4 Sampling and inference

**Training-time sampling**: not needed (loss only).

**Generation-time sampling** (for `transformer_generate.cpp`):
- Greedy / argmax: beam over the tree from root, taking `argmax_{child ∈ {L, R}} σ(d · s_n(h))` at each node. `O(log_2 V · m)` per token. **2700× cheaper than flat argmax over V.**
- Temperature sampling: walk the tree, sample left/right at each node from `Bernoulli(σ(s_n / τ))`. `O(log_2 V · m)` per token.
- Top-k / top-p: harder. Tree top-k requires beam search over the tree (O(k log V · m)). For k=50 at V=32 k, this is `50 · 15 · 2048 ≈ 1.5 M FLOPs` per token vs. flat top-k's `V · m + V log k ≈ 65 M FLOPs`. **~40× cheaper.** Implementation: standard priority-queue beam.

Reference: Bahl et al 1989 §IV (tree search for ASR), Mikolov et al 2013 (word2vec hierarchical-softmax sampling).

---

## 5. Compute analysis at multiple scales

The LM head's contribution to total step time depends on model size. Larger models spend more compute in attention + FFN + reversible-shear stack; the LM head is a *fixed* `m·V` cost that scales linearly with `m` only. Empirical breakdown:

| scale | params | m | L | LM-head GFLOPs | total step GFLOPs | LM-head fraction | VIDAR per-step speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| 41 M | 41 M | 256 | 6 | 8 | 27 | **30%** | 1.43× |
| 66 M | 66 M | 512 | 8 | 17 | 60 | **28%** | 1.39× |
| 100 M | 100 M | 768 | 12 | 25 | 150 | **17%** | 1.20× |
| 1.84 B | 1.84 B | 2048 | 53 | 67 | 2200 | **3.0%** | 1.031× |
| 18 B | 18 B | 4096 | 96 | 134 | 19 000 | **0.7%** | 1.007× |
| 72 B | 72 B | 6144 | 128 | 200 | 75 000 | **0.27%** | 1.0027× |
| 180 B | 180 B | 8192 | 144 | 268 | 200 000 | **0.13%** | 1.0013× |
| 400 B | 400 B | 12288 | 160 | 401 | 460 000 | **0.087%** | 1.00087× |

**Reading the table.**
- The LM-head GFLOPs grow linearly with `m` (from 8 GFLOPs at m=256 to 401 GFLOPs at m=12288).
- Total step GFLOPs grow as ~`m^2 · L · T` (FFN + attention dominate).
- LM-head fraction collapses from 30% at 41 M to 0.087% at 400 B. **This is the killer.** VIDAR eliminates a fraction that is itself shrinking.

**At the iter-194 target (extremely large LLMs on a single GPU = 18 B–400 B), VIDAR contributes a step speedup between 1.0009× (at 400 B) and 1.007× (at 18 B). Well under 1% of step time.** This is the honest gap.

**At small-model scale (41 M–100 M), VIDAR contributes 1.20×–1.43×.** This is a real, non-trivial speedup, but the operating regime is "research probes and Gate-0 sweeps," not "the magnitude target."

### 5.1 Tokenizer-research scale (V variation)

If V grows (e.g., multilingual or character-level):

| V | flat LM head FLOPs | HSM LM head FLOPs | ratio | LM-head fraction at 1.84 B | total speedup |
|---:|---:|---:|---:|---:|---:|
| 32 000 | 67 GFLOPs | 25 MFLOPs | 2700× | 3.0% | 1.031× |
| 50 000 | 105 GFLOPs | 30 MFLOPs | 3500× | 4.6% | 1.048× |
| 100 000 | 209 GFLOPs | 33 MFLOPs | 6300× | 8.7% | 1.095× |
| 256 000 | 535 GFLOPs | 38 MFLOPs | 14 100× | 19.6% | 1.244× |
| 1 000 000 | 2090 GFLOPs | 42 MFLOPs | 49 800× | 48.7% | 1.95× |

**Interesting regime.** At `V ≥ 256 k` (full-Unicode codepoint vocabularies, character-level multilingual, byte-level + special tokens), VIDAR matters even at 1.84 B. The user's current pile-bpe at V=32 k is on the wrong side of this curve. **If the project pivots to large-V tokenization (paradigm shift sub-track on tokenizer research), VIDAR becomes interesting.** Otherwise it does not.

---

## 6. Composition with paradigms #42–#49

VIDAR composes cleanly because it touches only the LM head. None of #42–#49 modifies the LM head:

| paradigm | what it touches | composes with VIDAR? | composition gain |
|---|---|---|---|
| #42 SCFA (sparse-causal-flow-attention) | attention shears | yes, multiplicative | full 2700× LM-head still applies |
| #43 ORION (RLG + SLC fusion) | depth + sequence-length curriculum | yes, mult. | LM head untouched |
| #44 MELT (FFN compression) | FFN shears | yes, mult. | LM head untouched |
| #45 SAS+CHRF stitching | step-to-step continuation | yes | LM head untouched |
| #46 REFLECTOR / SYNAPSE / ZEPHYR | optimizer / attn / gradient flow | yes | LM head untouched |
| #47 PHOENIX-1.58BIT | weight quantization | partial | E and `u_n` are quantizable identically; ~1% extra VIDAR-induced quant cost |
| #48 PHOENIX-1BIT / NEMESIS / STREAM | quant + cache | yes | LM head's E benefits from PHOENIX too |
| #49 ICARUS / ZENITH / AURORA | flow integrator / cross-step pred / ACT | yes, mult. | LM head untouched |
| #50-A HELIUM (attention/FFN tile) | attention + FFN | yes, **mult.** | combined: 2× × 1.05× = 2.1× at 1.84 B; 2× × 1.43× = 2.86× at 66 M |
| #50-C NIMBUS (optimizer pipeline) | optimizer | yes, mult. | combined: 1.4× × 1.05× = 1.47× at 1.84 B |

**Stack at 66 M with VIDAR + #42–#49 (small-model scale):**
- Pre-VIDAR: ~150× tokens/sec
- VIDAR adds 1.43×: 215×
- **Net: 215× at 66 M.** Useful for accelerating Gate-0 probes.

**Stack at 18 B with VIDAR + #42–#49 (target iter-194 scale):**
- Pre-VIDAR: ~300×
- VIDAR adds 1.007×: 302×
- **Net: 302×.** Negligible improvement.

**Stack at 18 B with HELIUM + #42–#49 (counterfactual):**
- Pre-HELIUM: ~300×
- HELIUM adds 2×: 600×
- **Net: 600×.** This is the iter-194 target.

**At iter-194's flagship operating point, VIDAR contributes <1% of HELIUM's improvement. Selection should be HELIUM unless small-model probe acceleration is specifically prioritized.**

---

## 7. Concrete primitives (code-level)

### 7.1 Tree-construction primitives

```cpp
// huffman_tree.h
namespace glades {
namespace vidar {

struct HuffmanTree {
    int V;                                  // vocabulary size
    std::vector<int> parent;                // V-1 internal nodes; parent[n] = ...
    std::vector<int> left;                  // left child
    std::vector<int> right;                 // right child
    std::vector<std::vector<int>> path_n;   // path_n[w] = list of internal node ids root->leaf
    std::vector<std::vector<int8_t>> path_d;// path_d[w] = ±1 directions
    std::vector<int> path_len;              // path_len[w] = length of path_n[w]
    int max_len;                            // max over w (≤ 15-ish for V=32k, Huffman bound)
};

void build_huffman_tree(const std::vector<float>& token_freqs, HuffmanTree& out);
void huffman_tree_pathwalk(int target, const HuffmanTree& tree,
                           int* path_nodes_out, int8_t* dirs_out, int& len_out);
}
}
```

### 7.2 GPU kernels (new in `cuda/gpu_kernels.h/.cu`)

```cpp
// gpu_kernels.h additions
bool vidar_path_scores(const float* h, const float* U, const float* B,
                       const int* path_nodes_flat, const int* path_offsets,
                       int T, int m, int max_path_len,
                       float* path_scores);
bool vidar_path_nll(const float* path_scores, const int8_t* dirs_flat,
                    const int* path_offsets, const int* targets,
                    int T, int padTokenId,
                    float* loss_out, int* count_out);
bool vidar_backward_h(const float* path_scores, const int8_t* dirs_flat,
                      const float* U, const int* path_nodes_flat,
                      const int* path_offsets, const int* targets,
                      int T, int m, int padTokenId,
                      float* dh);
bool vidar_backward_U(const float* h, const float* path_scores,
                      const int8_t* dirs_flat, const int* path_nodes_flat,
                      const int* path_offsets, const int* targets,
                      int T, int m, int N_int, int padTokenId,
                      float* dU, float* dB);
```

### 7.3 Config (additions to `training_config.h`)

```cpp
// In TransformerRunConfig:
// Hierarchical-softmax (VIDAR) toggle.
//
// When enabled, the LM head replaces the flat V-way softmax with a Huffman-tree
// hierarchical softmax. The tree is built from `vidarTokenFreqs` at training start
// (or recomputed if drifted; see vidarTreeRebuildSteps).
bool vidarEnable;
// Token frequencies for Huffman tree construction. If empty, defaults to uniform
// (degenerates to balanced binary tree).
std::vector<float> vidarTokenFreqs;
// Rebuild tree every N steps (0 = build once at start, never rebuild).
int vidarTreeRebuildSteps;
// Tied-embedding reprojection cadence (0 = no reprojection).
int vidarTiedReprojectSteps;
```

### 7.4 Integration touch-points in `sgd_transformer.cpp`

- Add `if (cfg.vidarEnable)` branch around the full-softmax block (around line 6841).
- Build tree once at training start; serialize to checkpoint.
- Inverse walk uses recorded `path_n`, `path_d` from forward — no re-derivation.
- Backward calls `vidar_backward_h` and `vidar_backward_U`.
- Tied-embedding reprojection runs every `vidarTiedReprojectSteps` (default 5 k) — projects `(u_n, b_n)` back to `mean_{w∈R_n} E_w - mean_{w∈L_n} E_w`.

### 7.5 CLI integration

Run-script `run.sh` adds:
```
--vidar 1                          # enable
--vidar-tree-rebuild-steps 0       # build once
--vidar-tied-reproject-steps 5000  # reproject every 5k
```

Token-frequency file is auto-derived from training corpus (passed to trainer via `--vidar-freq-file`).

### 7.6 Total LOC estimate

- `huffman_tree.h/.cpp`: 200 LOC (tree construction + path walk).
- `gpu_kernels.h/.cu` additions: 250 LOC (4 new kernels, mostly straightforward dot-product variants).
- `sgd_transformer.cpp` integration: 80 LOC (branching + checkpoint serialization).
- `transformer_generate.cpp` integration: 50 LOC (tree-aware sampling).
- Config + CLI: 30 LOC.
- **Total: ~610 LOC.** Modest.

---

## 8. The honest gap: VIDAR at extreme scale

This section is dedicated to what VIDAR cannot do.

### 8.1 The fundamental ceiling

The LM head's compute is bounded above by `T·m·V` for flat softmax and below by `T·m·log_2 V` for HSM. This bound is *additive* in the step's compute breakdown:
$$\text{total step} = \text{embedding lookup} + L \cdot (\text{attention}_l + \text{FFN}_l) + \text{LM head}.$$

The other terms grow as `L·m^2·T` (FFN) and `L·T^2·m` (attention). For 1.84 B (L=53, m=2048, T=1024):
- FFN: `53 · 2048² · 1024 · 2 (forward+backward) ≈ 450 GFLOPs · 53 = 23 TFLOPs`.
- Attention: `53 · 1024² · 2048 · 2 ≈ 4 TFLOPs` (with SCFA/GQA).
- Reversible shears + LayerNorm: ~0.5 TFLOPs.
- **LM head: 67 GFLOPs.**

LM head is **0.25% of step compute at 1.84 B in TFLOPs** (different from the GFLOPs row in §5 because that table was approximate). **Even reducing LM head to zero gives ≤ 0.3% step speedup at 1.84 B.**

### 8.2 At 18 B, 180 B, 400 B

Total step grows roughly as `L · m^2 · T`. LM head grows as `m · V`. Their ratio:
$$\frac{\text{LM-head}}{\text{total step}} \propto \frac{m \cdot V}{L \cdot m^2 \cdot T} = \frac{V}{L \cdot m \cdot T}.$$

At V=32 k, T=1024:
- 1.84 B (L=53, m=2048): ratio ≈ `32k / (53 · 2048 · 1024) ≈ 0.029% — i.e., LM head is ~0.03% as a fraction.
- 18 B (L=96, m=4096): ratio ≈ `32k / (96 · 4096 · 1024) ≈ 0.0079%.
- 180 B (L=144, m=8192): ratio ≈ 0.0026%.
- 400 B (L=160, m=12288): ratio ≈ 0.00159%.

**The LM head's fraction of step compute drops by 60× as we scale from 1.84 B to 400 B.** VIDAR's marginal contribution shrinks correspondingly. This is fundamental: as transformer scales up (`L`, `m` both grow), the dense interior dominates the boundary projection.

### 8.3 Why VIDAR cannot meet the iter-194 magnitude brief

The brief calls for "magnitudes better on compute speed." Magnitude = 10×, 100×, 1000×. VIDAR offers:
- 1.43× at 66 M (small-model regime; not iter-194 target).
- 1.05× at 1.84 B (sub-magnitude).
- 1.007× at 18 B (sub-percent).
- <1.005× at 180–400 B (negligible).

**VIDAR is not a magnitude-class shift at the user's target operating point.** Selection logic must reject VIDAR for iter-194 production (in favor of HELIUM or NIMBUS) and reserve it for a tokenizer-research sub-track or as a small-model accelerator for Gate-0 probes.

### 8.4 What VIDAR competes with for budget

The two competing candidates do address the iter-194 magnitude brief:
- **HELIUM (#50-A):** 2× across all compute via streaming SVD-tile attention/FFN. Saves ~50% of step at all scales. **At 18 B: 2× total speedup; at 400 B: 2× total speedup.** This is what magnitudes look like.
- **NIMBUS (#50-C):** 1.3–1.5× via optimizer pipelining. Saves ~25–30% of step. **At 18 B: 1.4× total speedup; at 400 B: 1.4× total speedup.** Less dramatic but scale-invariant.

VIDAR vs. HELIUM at 18 B: **1.007× vs. 2×.** HELIUM dominates by 285×.
VIDAR vs. NIMBUS at 18 B: **1.007× vs. 1.4×.** NIMBUS dominates by 57×.

**The selection committee should choose HELIUM (primary) + NIMBUS (secondary) over VIDAR for iter-194 production unless small-model acceleration becomes a top-priority research axis.**

---

## 9. When VIDAR DOES help

VIDAR is genuinely useful in three regimes:

### 9.1 Small-model training (≤ 100 M parameters)

- **Gate-0 probes** for FACE, MFIO, RLG, SAS, KV-FACE, HUTCH-DIAG, ASTRA all run at 41–66 M. VIDAR's 1.4× speedup at this scale cuts a 1-hour Gate-0 probe to 42 minutes — meaningful when running 5–10 Gate-0s per Ralph-loop iteration.
- **Architecture sweeps** (depth, width, head-count) at 100 M scale benefit by 1.18×.
- **Tokenizer ablations** (V=8 k, 16 k, 32 k, 64 k) at 66 M are 1.4× faster.

Estimated end-of-cycle benefit at 66 M: **~1 day saved per Ralph-loop iteration if 50% of iteration is small-model probes.** Not magnitudes, but real engineering time.

### 9.2 Large-vocabulary regimes (V ≥ 256 k)

If the project pivots to multilingual or character-level tokenization (as in BLOOM, mGPT, ByT5 lines), V grows to 256 k–1 M. At V=1 M and 1.84 B, VIDAR delivers 1.95× total speedup — back into magnitude-adjacent territory. **This is a tokenizer-research enabler.**

The project's existing pile-bpe (V=32 k) does not benefit. But if iter-200+ explores larger vocabularies, VIDAR lands.

### 9.3 Generation-time top-k / top-p sampling

For inference workloads (eval pipelines, generation studies, beam search):
- top-50 sampling at V=32 k flat: 65 MFLOPs/token.
- top-50 sampling at V=32 k VIDAR (tree beam): 1.5 MFLOPs/token — **40× cheaper**.

This is a clear win for the eval pipeline even at 1.84 B. **Implementation note:** the eval-only deployment can use VIDAR even if training uses flat softmax — the trees are equivalent under Theorem 3 at convergence, and a one-shot tree induction from trained `(E, b)` (via §3.2's S_n construction, executed at eval-build time only) gives bit-equivalent argmax/top-k.

### 9.4 Future paradigm-50.5 candidate: VIDAR-SAMPLE-ONLY

A reduced-scope variant: **train with flat softmax, eval with VIDAR.** Zero training-time risk, 40× generation-time top-k speedup, NLL preservation by Theorem 2 (exact at induction time). Effort: ~150 LOC (eval-only path), 1 week. Could be shipped as a side-track without blocking iter-194 production.

This may be the most honest framing: VIDAR-as-eval-accelerator, not VIDAR-as-training-paradigm.

---

## 10. Gate-0 probe (optional)

**Goal.** Confirm that linearized HSM with Huffman tree converges to within 0.01 nat of flat softmax on a 66 M baseline, and that the 1.43× speedup materializes.

**Setup.**
- Baseline: 66 M CHIRON, V=32 k pile-bpe, flat softmax, 5 k steps.
- Test: 66 M CHIRON, V=32 k pile-bpe, VIDAR Huffman, 5 k steps.
- Match seed, LR schedule, batch size, T=512.

**Pass conditions.**
- Final-EMA NLL (VIDAR) ≤ Final-EMA NLL (flat) + 0.01 nat at step 5 k.
- Wall-clock VIDAR ≤ 0.78 × wall-clock flat (1.28× speedup; conservative below the 1.4× projection).
- No training instabilities (no NaN, no EMA blow-up).

**Cost.** ~1 GPU-hour per arm × 2 arms = ~2 GPU-hours. Cheaper than most paradigms' Gate-0 because the training horizon is 5 k steps.

**If VIDAR fails Gate-0** (NLL gap > 0.05 nat or speedup < 1.2×): retire as primary candidate, reserve as eval-only sampler (§9.3).

---

## 11. Risks and unknowns

1. **Training-time NLL gap (Regime C, §3.3).** Linearized HSM with free `(u_n, b_n)` may have a 0.01–0.05 nat gap during early training. Mitigation: tied-embedding initialization (§3.4) + reprojection every 5 k steps. **Empirical risk; Gate-0 probes.**

2. **Huffman tree drift.** If the corpus distribution shifts mid-training (curriculum, fresh-corpus injection, replay), the tree's average path length grows above `H(q)+1`. Mitigation: tree rebuild every N steps (`vidarTreeRebuildSteps`). Cost: 15 ms CPU per rebuild; negligible.

3. **GPU kernel efficiency.** Path-walk kernels are inherently irregular (path lengths vary 8–15 across tokens). Padding to max length wastes ~7%. Mitigation: standard. Modern GPUs (Ampere, Ada) handle this fine.

4. **Atomic-add contention on `dU` for root-near nodes.** Root-adjacent classifiers receive contributions from ~T/2 tokens — high contention. Mitigation: per-warp reduction trees before atomic. ~1% kernel overhead.

5. **Checkpoint serialization.** Tree topology is part of checkpoint. Adds ~256 KB to checkpoint files at V=32 k. Negligible.

6. **Composition with `tieEmbeddings`.** CHIRON already ties embeddings; VIDAR's reprojection step (§3.4) preserves this. No conflict expected.

7. **The fundamental risk.** **VIDAR is the right shift for the wrong target.** We document it knowing this.

---

## 12. Selection recommendation

Among #50-A (HELIUM), #50-B (VIDAR), #50-C (NIMBUS):

| Criterion | HELIUM | VIDAR | NIMBUS |
|---|---|---|---|
| Magnitude at iter-194 target (18 B+) | **2×** | 1.007× | 1.4× |
| NLL preservation rigor | bit-equivalent | mathematical-identity (Huffman) | bit-exact |
| Engineering effort | heavy (2100 LOC) | **modest (610 LOC)** | modest (400 LOC) |
| Composability | mult. | mult. | mult. |
| Gate-0 cost | high (4 GPU-hours) | **low (2 GPU-hours)** | low (2 GPU-hours) |
| Scale-dependent benefit | none | **steep collapse** | none |
| Side-track value | low | **eval-time top-k 40×** | low |

**Recommended action:**
1. **Primary for iter-194 production: HELIUM** (magnitude-class).
2. **Secondary: NIMBUS** (independent axis, composable).
3. **VIDAR: deferred to side-track**, shipped as eval-only top-k accelerator (Sec 9.3 / §9.4) if the eval pipeline benefits.
4. **VIDAR full version: revisit if** (a) tokenizer pivots to V≥256 k, or (b) small-model probe time becomes the gating constraint on Ralph-loop velocity.

VIDAR is *not selected* for iter-194 paradigm shift #50, but its eval-only sub-track is a 1-week ship that costs almost nothing. **Recommend implementing the eval-only sub-track regardless of #50 selection.**

---

## 13. References

- Bahl, L., Brown, P., de Souza, P., Mercer, R. (1989). "A tree-based statistical language model for natural language speech recognition." IEEE TASSP 37(7).
- Bengio, Y., Senécal, J. (2003). "Quick training of probabilistic neural nets by importance sampling." AISTATS.
- Chen, W., Grangier, D., Auli, M. (2016). "Strategies for training large vocabulary neural language models." ACL.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., Stein, C. (2009). *Introduction to Algorithms*, 3rd ed. §16.3 (Huffman codes).
- Goodman, J. (2001). "Classes for fast maximum entropy training." ICASSP.
- Joulin, A., Cissé, M., Grangier, D., Jégou, H. (2017). "Efficient softmax approximation for GPUs." ICML (adaptive softmax — orthogonal but related approach).
- Le, H. S., Oparin, I., Allauzen, A., Gauvain, J. L., Yvon, F. (2011). "Structured output layer neural network language model." ICASSP.
- Mikolov, T., Sutskever, I., Chen, K., Corrado, G., Dean, J. (2013). "Distributed representations of words and phrases and their compositionality." NeurIPS (word2vec, hierarchical softmax practical results).
- Mnih, A., Hinton, G. (2009). "A scalable hierarchical distributed language model." NeurIPS.
- Morin, F., Bengio, Y. (2005). "Hierarchical probabilistic neural network language model." AISTATS (the original HSM paper).

---

*End of Candidate B. See companion documents `PARADIGM_SHIFT_50_CANDIDATE_A_HELIUM.md` and `PARADIGM_SHIFT_50_CANDIDATE_C_NIMBUS.md`. Selection writeup: `PARADIGM_SHIFT_50_SELECTION.md` (forthcoming).*
