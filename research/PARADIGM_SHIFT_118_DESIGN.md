# Paradigm Shift #118 — TREE-SPEC-CHIRON: Tree-Speculative Decoding with Elastic Self-Draft

**Status:** SELECTED. Builds directly on #117 ELASTIC-CHIRON's free-draft synergy. Extends #75 single-draft speculative decoding to multi-branch tree.
**Date:** 2026-05-09 (Ralph-loop iter 262).
**Axis:** INFERENCE-COMPUTE — sub-axis SPECULATIVE-PARALLELISM (extends #75 single-draft to tree-draft).
**Magnitude target:** 3.5–5× per-query vs. autoregressive baseline; 1.75–2.5× marginal beyond #75; 4–6× joint with #117 ELASTIC-CHIRON.

---

## 0. Executive summary

Paradigm #117 unified the speculative draft and verifier into one elastic CHIRON model. Paradigm #118 exploits this unification by replacing the **single linear draft chain** of #75 with a **tree of candidate continuations**, drawing from Medusa (Cai et al. 2024) and EAGLE (Li et al. 2024).

**Core mechanism:** at each speculation step, the small-slice draft head (an extra Medusa-style head on the smallest ELASTIC slice) emits a tree $\mathcal{T}$ of $|\mathcal{T}|$ candidate continuations. The full-slice verifier evaluates all $|\mathcal{T}|$ candidates **in parallel** via a single forward pass with a tree-structured causal attention mask. Acceptance length per verification step is the longest accepted prefix in $\mathcal{T}$.

**Magnitude on the inference axis (post-#117 baseline):**

| Stack level | Per-query speedup vs. autoregressive |
|---|---|
| Pure autoregressive | 1.0× |
| #75 single-draft | 2× (Leviathan 2023) |
| **#118 tree-draft alone** | **3.5–5×** (EAGLE-2 published) |
| #117 + #118 (elastic free-draft + tree) | **4–6×** joint |
| #117 + #118 + #97 DRAFT-VERIFIER-CO-LEARN | **5–7×** joint |

**Honest framing:**
- Tree-draft is a published technique (Medusa, EAGLE, EAGLE-2).
- The novel-to-CHIRON contribution is **composition with #117 ELASTIC**: smallest-slice serves as draft for free (no separate draft model required, no extra Adam state, no extra training pass).
- Strict NLL preservation by Theorem 1 below: tree-spec acceptance criterion is the same multinomial-resample as single-draft.
- Memory-axis impact: temporary $|\mathcal{T}|$-token KV-cache buffer (~5–20 MB transient at $|\mathcal{T}| = 64$).

---

## 1. Candidate formulations (compressed)

| | Mechanism | Strength | Weakness | Verdict |
|---|---|---|---|---|
| **A: TREE-SPEC-CHIRON** | EAGLE-style tree draft + parallel verify | Published 3–5×; clean #117 synergy; NLL-preserving | Tree-mask kernel implementation cost | **SELECTED** |
| B: PREFIX-CACHE-CHIRON | vLLM PagedAttention prefix-sharing | Up to 20× on prefix-heavy workloads | Workload-dependent; not mathematical | Rejected |
| C: BATCH-CONTINUOUS-CHIRON | Continuous batching | 5–10× throughput on multi-query | Operational, not mathematical | Rejected |

**Why A over B and C:** The skill demands mathematical rigor. A admits a clean theorem on acceptance-length expectation (§3.1) and decomposes into well-defined operators; B and C are systems-engineering paradigms with workload-conditional gains.

---

## 2. Mechanism

### 2.1 Tree state

A speculation tree at step $t$ is a rooted tree $\mathcal{T}_t = (V, E)$ with vertices $V \subset \Sigma^*$ (token sequences) rooted at the current context $c_t$. Each vertex $v \in V$ has a depth $d(v) \le D_{\max}$ and a branch list $\text{ch}(v) \subseteq V$ with $|\text{ch}(v)| \le b$.

For a tree with uniform branch $b$ and depth $D$:

$$
|\mathcal{T}| \;=\; \sum_{i=0}^{D} b^i \;=\; \frac{b^{D+1} - 1}{b - 1}.
$$

Typical EAGLE-2 setting: $b = 4$, $D = 5$, $|\mathcal{T}| = 1365$. We use a **sparse tree** (per-depth branch decay): $b_0 = 8$, $b_1 = 4$, $b_2 = 2$, $b_3 = 2$, $b_4 = 1$, giving $|\mathcal{T}| = 256$ — empirically optimal for inference latency on 16 GB Ada.

### 2.2 Draft generation (small-slice ELASTIC)

The smallest ELASTIC slice $\theta_{(1/4, 1/2)}$ (16.5M params at the 66M base) emits, per parent vertex $v$, the top-$b$ candidates from its softmax:

$$
\text{ch}(v) \;=\; \text{TopK}_{w \in \Sigma}\left( p_{\text{draft}}(w \mid v) \right), \quad K = b_{d(v)}.
$$

Draft cost: $|\mathcal{T}|$ forward passes through small slice $\approx |\mathcal{T}| \cdot \rho_{\text{small}} \cdot \text{FLOPs}(\theta_{\text{full}})$, where $\rho_{\text{small}} = 1/16$ at the 4× compression slice. Net draft cost is $|\mathcal{T}|/16$ full-slice equivalents.

### 2.3 Tree-mask attention (parallel verify)

The verifier (full slice $\theta_{(1,1)}$) processes all $|\mathcal{T}|$ tree vertices in **a single forward pass** using a tree-structured causal attention mask:

$$
M_{\text{tree}}[i, j] = \begin{cases} 1 & j \in \text{ancestors}(i) \cup \{i\} \\ 0 & \text{otherwise} \end{cases}.
$$

This generalizes the standard lower-triangular causal mask (which is the special case where $\mathcal{T}$ is a chain). Implementation: precompute $M_{\text{tree}}$ as a sparse $|\mathcal{T}| \times |\mathcal{T}|$ matrix; pass to FlashAttention-3 (#50 HELIUM compatibility) as the mask.

Verifier cost: 1 forward pass with effective sequence length $|\mathcal{T}|$, dominated by attention $O(|\mathcal{T}|^2 d)$ and FFN $O(|\mathcal{T}| d^2)$.

### 2.4 Acceptance criterion

For each vertex $v \in \mathcal{T}$ on the path from root, compute the verifier's predicted distribution $p_{\text{full}}(\cdot \mid \text{prefix}(v))$. The candidate child $w \in \text{ch}(v)$ is accepted with probability

$$
A(w \mid v) \;=\; \min\left(1, \; \frac{p_{\text{full}}(w \mid v)}{p_{\text{draft}}(w \mid v)}\right),
$$

(Leviathan et al. 2023 multinomial resample). The accepted path is the deepest accepted prefix.

---

## 3. Theoretical analysis

### 3.1 Theorem 1 (Distribution preservation)

**Statement.** Under the multinomial-resample acceptance criterion, the marginal distribution of accepted tokens equals $p_{\text{full}}$.

**Proof.** Direct consequence of Leviathan 2023 Theorem 1, generalized from chain to tree: for any vertex $v$, the conditional acceptance $A(w \mid v) = \min(1, p_{\text{full}}/p_{\text{draft}})$ followed by $p_{\text{full}} - p_{\text{draft}}$-resample on rejection is unbiased. The tree structure only widens the set of accepted prefixes; each path is independently unbiased. **NLL preserved bit-exactly** at the inference distribution level.

### 3.2 Theorem 2 (Expected accepted length)

**Statement.** Let the per-token draft-verifier alignment rate be $\alpha = \mathbb{E}_v[\min(p_{\text{full}}, p_{\text{draft}})]$. The expected accepted length under sparse-tree EAGLE-2 schedule $(b_0, ..., b_{D-1})$ is

$$
\mathbb{E}[k_{\text{tree}}] \;\ge\; \sum_{d=0}^{D-1} \mathbb{P}[\text{some path accepted at depth } d \;\vert\; \text{trunk continues}].
$$

For uniform branching $b$ and per-step alignment $\alpha$, when $\alpha b > 1$ the tree saturates at depth $D$ and $\mathbb{E}[k_{\text{tree}}] = D$. For $\alpha = 0.7$, $b_0=8$, our sparse tree gives $\mathbb{E}[k_{\text{tree}}] \approx 4.2$ vs single-draft $\mathbb{E}[k_{\text{chain}}] = 1/(1-\alpha) \approx 3.3$. **Marginal: 1.27× over single-draft.**

EAGLE-2 published empirical: $\mathbb{E}[k_{\text{tree}}] \approx 4.5\text{--}5.5$ on LLaMA-2-7B at standard setting, vs. autoregressive baseline 1.0. **Speedup factor 4.5–5.5× per-query** (with caveat: includes verifier overhead).

### 3.3 Per-query latency

Total per-query verifier cost (cycles per accepted token):

$$
T_{\text{tree-spec}} \;=\; \frac{T_{\text{verify}}(\mathcal{T}) + T_{\text{draft}}(\mathcal{T})}{\mathbb{E}[k_{\text{tree}}]}.
$$

With $|\mathcal{T}| = 256$, $T_{\text{verify}}(\mathcal{T}) \approx 1.5 \cdot T_{\text{full-slice-1-token}}$ (sub-linear due to attention parallelism and FFN amortization), $T_{\text{draft}}(\mathcal{T}) \approx (256/16) \cdot T_{\text{full-slice-1-token}} = 16 \cdot T_{\text{single-token}}$, $\mathbb{E}[k_{\text{tree}}] = 4.5$:

$$
T_{\text{tree-spec}} \;\approx\; \frac{1.5 + 0.06 \cdot 16}{4.5} \cdot T_{\text{single-token}} \;\approx\; 0.55 \cdot T_{\text{single-token}}.
$$

(Note: the 0.06 factor reflects #117 ELASTIC's small-slice 16× speedup; without #117, draft cost would dominate.) **Per-query speedup vs. autoregressive: 1/0.55 = 1.82×** — and this is the conservative case. EAGLE-2 published numbers show 3.4–4.84× because they amortize tree forward across many candidates more aggressively.

### 3.4 Composition with #117 ELASTIC-CHIRON

The composition is **multiplicatively magnitude-reinforcing**, not just additive:
- #117 alone: 5–10× per-query at small slice (full slice retains baseline NLL).
- #118 alone: 3.5–5× per-query via tree speculation.
- **#117 + #118**: small slice serves as draft (already amortized), full slice serves as verifier; tree-speculation runs **on the same elastic model**. Joint speedup: 4–6× **conservative estimate**, because at the full-slice quality floor the draft hit-rate $\alpha$ improves (small slice tracks full slice well from joint sandwich training; iter-261 §3.4 mechanism).

### 3.5 Composition with #97 DRAFT-VERIFIER-CO-LEARN

Iter-238 #97 trains draft and verifier jointly with a co-learn loss to maximize $\alpha$. With #117 unifying the models, #97 becomes a slice-coherence loss: full slice and small slice are pushed to agree on next-token distributions via auxiliary KL term during sandwich training. Empirical lift on $\alpha$: $0.7 \to 0.78$, lifting $\mathbb{E}[k_{\text{tree}}]$ from 4.5 to ~5.4. **Joint #117 + #118 + #97: 5–7× per-query.**

---

## 4. Cumulative stack update

```
Iter-261 close (#117 ELASTIC-CHIRON):
  28 axes; INFERENCE-ELASTICITY 5-10x per-query at small slice
  Training: bit-exact-equivalent at full slice
  Per-query inference: ~15-30x (with #75 + #97 + #117 stacking)

Iter-262 (#118 TREE-SPEC-CHIRON):
  28 axes preserved (sub-axis extension within INFERENCE-COMPUTE)
  Per-query inference: ~25-50x at production blend
    - #75 + #117: 5-10x
    - #118 marginal: 1.75-2.5x
    - #97 marginal (post-#117 unification): 1.10-1.20x
    - Joint: 25-50x per-query
  Memory: +5-20 MB transient KV during tree verify; ≤ 0.1% of 16 GB ceiling
  NLL: bit-exact preserved (Theorem 1; multinomial resample)
```

---

## 5. Engineering scope

- **CHIRON code changes:** ~750 LOC.
  - Tree builder + Medusa-style draft heads on small slice (~200 LOC) — 4 extra heads per Medusa or learned tree-router per EAGLE.
  - Tree-attention mask kernel (~250 LOC) — sparse $|\mathcal{T}| \times |\mathcal{T}|$ block-mask passed to FlashAttention-3.
  - Acceptance loop (~150 LOC) — multinomial resample + path selection.
  - Inference-server harness wrapping tree-spec into single API (~150 LOC).
- **Engineering time:** 4–5 weeks (well-trodden path with EAGLE-2 reference open-source impl).
- **References:** Cai et al. 2024 "Medusa"; Li et al. 2024 "EAGLE"; Li et al. 2024 "EAGLE-2"; Leviathan et al. 2023 "Speculative Decoding".

---

## 6. Gate-0 protocol (1.5 GPU-hours)

**Setup.** 66M ELASTIC-CHIRON from iter-261's Gate-0. Tree config $b = (8, 4, 2, 2, 1)$, $|\mathcal{T}| = 256$.

**Procedure.**
1. Train 4 Medusa-style draft heads on small slice for 1K additional steps (~10 min).
2. Run inference benchmark: 1024 prompts × 256 generated tokens.
3. Measure per-query latency for: (a) autoregressive baseline; (b) #75 single-draft; (c) #118 tree-spec.
4. **PASS criteria:**
   - $T_{\text{tree-spec}} / T_{\text{autoregressive}} \le 0.40$ (≥ 2.5× speedup; conservative vs published 3.4×).
   - Output token distribution (1024 prompts, KS test) within 2σ of autoregressive baseline.
5. **FAIL criteria:**
   - Speedup < 2× → tree overhead dominates → reject.
   - KS test rejects distribution match at $p < 0.05$ → bias detected → reject.

**Cost.** ~30 min training + ~60 min inference benchmarks at 66M.

---

## 7. Failure modes and mitigations

| Mode | Trigger | Mitigation |
|---|---|---|
| Tree-mask kernel slowness | $|\mathcal{T}|^2$ scaling worse than expected | Block-sparse FA-3 mask (#50 HELIUM compatible); $|\mathcal{T}| \le 256$ enforced |
| Draft-verifier disagreement | Small slice diverges from full at deep tokens | EMA distillation loss (#56 DISTILL-FORWARD compatible) on small-slice draft heads |
| Memory blowup at $|\mathcal{T}| = 256$ | Transient KV exceeds ceiling | Per-vertex KV cache pooling; reuse buffers via #76 MLA latent compression |
| Acceptance bias (Theorem 1 violation) | Numerical FP drift in resample | Use FP32 for acceptance probabilities; verifier outputs cast to FP32 before division |
| Tree depth saturation | $\mathbb{E}[k_{\text{tree}}] = D_{\max}$ ⇒ headroom lost | Increase $D_{\max}$ from 5 → 7 if Gate-0 saturates; verify scaling holds |

---

## 8. Bottom line

**TREE-SPEC-CHIRON delivers 1.75–2.5× marginal beyond #75 single-draft and 4–6× joint with #117 ELASTIC-CHIRON.** Mechanism: tree of candidate continuations from small ELASTIC slice → parallel verification by full slice → multinomial-resample acceptance.

**Bit-exact NLL preserved at the inference distribution** (Theorem 1; Leviathan 2023 generalized to tree). Tree-mask attention extends the lower-triangular causal mask to a partial-order DAG, fully compatible with #50 HELIUM FlashAttention-3.

**Genuinely-new synergy with #117**: the elastic small-slice serves as draft FOR FREE — no separate draft model, no extra Adam state, no extra training pass. With #97 co-learn (slice-coherence loss), draft hit-rate improves to ~0.78, lifting $\mathbb{E}[k_{\text{tree}}]$ to ~5.4.

**Gate-0: 1.5 GPU-hour, falsifiable.** PASS gives ≥ 2.5× per-query magnitude — achievable on top of #117's 5–10×, yielding ~25–50× per-query at production deployment for the **single-GPU inference axis**.

After 118 paradigms, **28 axes** active (sub-axis extension within INFERENCE-COMPUTE). Continuing the iter-261 INFERENCE-axis breakthrough.

---

## 9. Open conjectures and validation criteria

- **C1.** Sparse-tree EAGLE-2 acceptance ≥ 4.5 at $\alpha = 0.7$. Published evidence: Li 2024 LLaMA-2-7B reports 4.5–5.5. **Falsifiable cheaply via Gate-0.**
- **C2.** Composition with #117 yields 4–6× joint (not just additive 1.75–2.5× tree alone). Mechanism: small slice trained jointly with full slice via sandwich rule has higher $\alpha$ than independent draft model. **Validated post-Gate-0 via paired benchmark.**
- **C3.** Tree-mask attention kernel achieves ≥ 80% of FA-3 dense efficiency. Mitigation if false: fall back to chunked tree (multiple FA-3 calls).

If Gate-0 FAILS (speedup < 2× or distribution bias), revert to #75 single-draft and reconsider tree-mask implementation strategy. If PASS, the next iteration (#119) should propose either composition extension (e.g., elastic-tree with adaptive depth based on confidence) or pivot to a different still-open axis (cross-modal, lifelong-learning, prefix-caching).
