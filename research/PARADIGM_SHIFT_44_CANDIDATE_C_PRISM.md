# Paradigm Shift #44, Candidate C — PRISM

**P**er-token **R**eversibility-preserving **I**ntermediate **S**parse-**M**ixture

---

## Executive Summary

PRISM replaces each CHIRON FFN shear with a sparse mixture-of-experts (MoE) FFN, where every token is routed to k of E experts via a deterministic, q-only routing function. With default config (E=8, k=2), per-FFN compute drops by a factor of 4× when scaled against an expressive-equivalent dense baseline. Combined with the existing 65× compute lift from prior paradigm shifts, the headline at 1.84B-param flagship is **~260× compute speedup**.

**The compute claim is strong (4× on the FFN bucket — the dominant 25% post-#42 SCFA). The memory claim is honest and limited:**

- **Naive PRISM (E independent full-size expert FFNs): 8× MORE FFN memory** — *worse* for "extremely large LLMs."
- **PRISM-LoRA (shared expert backbone + per-expert low-rank adapters): memory parity with dense FFN** — neutral, no compression.

PRISM does NOT enable larger models on the same hardware (unlike MELT, which gives 205× memory compression) and does NOT enable distributed scaling (unlike HYDRA, which adds 6.4× scaling at 8 GPUs). PRISM's strength is **single-GPU compute speedup on a fixed model**.

CHIRON-specific design point: the symplectic shear `(q, p) ↦ (q, p + Y(q))` is invertible iff Y(q) is deterministic and a pure function of q. PRISM routing is engineered deterministic and q-only (no p, no auxiliary state, no randomness), preserving reversibility exactly (Theorem 1).

**Honest assessment**: by the joint criterion "magnitudes compute AND extremely large LLMs," PRISM is the weaker candidate. MELT compresses memory 205× AND speeds compute 3.2×. HYDRA enables true distributed scaling. PRISM gives the highest single-paradigm raw compute factor but cannot compete on the size-scaling axis.

---

## 1. Primitive Objects

### 1.1 Routing Layer
```
W_r ∈ ℝ^{m × E}    (routing weights; m = model dim, E = experts)
s_t = W_r^⊤ q_t ∈ ℝ^E
```
Pure function of q — the position-half of the (q, p) state. No p, no state, no randomness.

### 1.2 Expert FFNs
E experts of the same architectural form:
```
FFN_e(q) = W_e^{down} · σ(W_e^{up} · q),  e = 1, ..., E
```
σ is SwiGLU (matches baseline). Each FFN_e has inner dim dFFN_e (≤ baseline dFFN).

### 1.3 Top-k Selection and Capacity
```
ε_t := top-k indices of s_t,  |ε_t| = k  (default k=2)
r_{t,e} := softmax(s_t|_{ε_t})_e for e ∈ ε_t
```
Capacity factor c=1.25 bounds per-expert tokens: `C_e = ⌊c·k·T/E⌋`. Tokens beyond capacity bypass that expert (zero contribution from e; other selected experts still apply).

### 1.4 Per-token Assignment
`A_t := (ε_t, r_{t,·}|_{ε_t})` — recomputable from q at inverse, materialized in scratch for backward only (~12 KB/layer at default config).

---

## 2. State Space

The CHIRON state `(q, p) ∈ ℝ^{T×m} × ℝ^{T×m}` is unchanged. Transient routing state at each shear:
```
R ∈ ℝ^{T×E}  — sparse, k non-zeros per row (the assignment matrix)
B_e := {t : e ∈ ε_t}  — per-expert token buckets, |B_e| ≤ C_e
```
R is consumed within the shear and discarded (or recomputed on inverse). Per-expert buckets B_e are dispatch artifacts, materialized only during forward/backward.

---

## 3. Evolution Law

**Forward shear:**
1. Routing scores: `S = q · W_r ∈ ℝ^{T×E}`.
2. Top-k selection: assignment matrix R per §1.3.
3. Build buckets B_e with capacity throttling.
4. Per-expert: `Z_e = FFN_e(q[B_e])`.
5. Scatter-gather: `Y_t = Σ_{e ∈ ε_t} R_{t,e} · Z_e[t]`.
6. Apply shear: `(q, p) ↦ (q, p + Y)`.

**Inverse shear:** q' = q (unchanged in shear). Recompute Y from q' using identical W_r, identical experts, identical deterministic routing → identical R → identical Y. Recover p = p' − Y.

The inverse uses the same forward-FFN call. CHIRON's reversibility budget already accounts for this.

---

## 4. Theorems

### Theorem 1 (Reversibility Preservation)

Let r: ℝ^m → Δ^E_{(k)} be any deterministic top-k routing function, and FFN_e deterministic for e=1..E. Define `Y_PRISM(q) := Σ_e r_e(q) · FFN_e(q)`. The shear `Φ_PRISM: (q,p) ↦ (q, p + Y_PRISM(q))` is bijective with inverse `(q', p') ↦ (q', p' − Y_PRISM(q'))`.

**Proof.** Determinism: r and FFN_e are deterministic, so Y_PRISM is. Inverse: q' = q so Y_PRISM(q') = Y_PRISM(q); thus p' − Y_PRISM(q') = p. The shear's algebraic form (modify p by a function of q only) is preserved, so symplecticity holds. ∎

**Numerical caveat.** Bit-identical reproducibility requires:
- fp32 routing arithmetic (not bf16, to avoid quantization-induced reroutes).
- Deterministic tiebreaking (lowest index wins on ties).
- fp32 softmax over the k selected experts.

Without these, single-bit perturbations could flip routing decisions, manifesting as O(1) errors in p — not the usual O(ε) drift. PRISM enforces all three.

### Theorem 2 (Compute Bound)

Per-PRISM-shear FLOPs (dominant terms):
```
C_PRISM ≈ T · k · 2 · m · dFFN_e   (k expert FFNs forward)
        + T · m · E                  (routing scores)
        + O(T · k · log E)           (top-k argmax)
```
Comparing to a dense FFN of inner dim D = E · dFFN_e (the expressivity-equivalent dense baseline):
```
C_dense = T · 2 · m · E · dFFN_e
C_PRISM / C_dense = k/E + O(E/(2·E·dFFN_e)) ≈ k/E
```
**Headline: 4× compute reduction at k=2, E=8 vs dense E·dFFN_e.** ∎

**Critical caveat.** The 4× speedup is vs *expressivity-equivalent dense* (inner dim E·dFFN_e). It is NOT 4× vs the existing CHIRON dense FFN at dim 8192. To realize the speedup, one must either:
- (a) Scale up: replace baseline dim-D dense with PRISM E experts each at dim-D — gain 4× expressivity at 1× compute (effective speedup at higher capacity).
- (b) Scale across: replace baseline dim-D dense with PRISM E=8/k=2 each at dim-D/4 — match expressivity at 0.5× compute (modest speedup, modest memory increase).

Misapplied at full per-expert dim with no expressivity target, PRISM costs *more* than dense.

### Theorem 3 (Expressivity Bound)

Let F_dense be a dense FFN of inner dim D. Let F_MoE^{(E,k)} be a k-of-E mixture, each of inner dim d.

- **Lower bound**: F_MoE^{(E,k)} expresses any dim-d dense (set all experts equal).
- **Upper bound**: parameter count E · 2md matches dense E·d. Function-space dim is bounded by E·d-dense.
- **Empirical (Switch, Mixtral)**: trained MoE achieves loss within 0.1 nat of dense E·d at compute factor k/E.

For PRISM E=8/k=2 at d = dFFN: matches dim-8·dFFN dense expressivity at 0.25× compute. ∎

---

## 5. Routing Design

### 5.1 Top-k with Deterministic Tiebreaking
```
top-k(s) := sort indices descending by (s_e, -e), take first k.
```
Lowest index wins ties. Bit-deterministic across runs.

### 5.2 Soft-top-k for Differentiable Backward

Hard top-k is non-differentiable. Standard MoE uses Gumbel-softmax for stochastic relaxation, but Gumbel introduces randomness that breaks reversibility.

**PRISM solution: forward-hard, backward-soft.** Forward selects k experts via hard top-k. Backward routes gradients only through the k selected experts and their softmax weights (Switch-style). For e ∉ ε_t, gradient is zero. This matches Switch/Mixtral practice and is fully deterministic. The router W_r updates to push scores upward for experts that produced helpful FFN outputs.

### 5.3 Capacity Throttling and Bypass

Per-expert capacity `C_e = ⌊c·k·T/E⌋`, default c=1.25. When expert e exceeds capacity, excess tokens (lowest s_{t,e} among those routing to e) bypass: their contribution from e is zero. They still receive contributions from their other selected experts. If all of t's selected experts are over capacity, t bypasses entirely (Y_t = 0; identity passthrough through residual).

At c=1.25 with uniform random routing, ~17.5% peak bypass rate. With deterministic q-dependent routing, empirical imbalance is usually lower (Conjecture C1).

### 5.4 Inverse Reproducibility

Forward stores nothing extra: inverse recomputes routing from q'. As long as routing is deterministic in q + W_r with deterministic tiebreaking, inverse is exact.

---

## 6. PRISM-LoRA Memory Mitigation

Naive PRISM (E full-size experts) costs 8× FFN memory — unacceptable for "extremely large LLMs."

**PRISM-LoRA**: each expert is a shared backbone plus a low-rank adapter:
```
FFN_e(q) = (W_base + B_e · A_e) · σ((W_base^{up} + B_e^{up} · A_e^{up}) · q)
```
with rank r=4 adapters.

**Memory accounting (m=2048, dFFN=8192, E=8, r=4):**
- Shared W_base + W_base^{up}: 16.78M params
- 8 LoRA pairs (up + down): 8 · 2 · 2 · 2048 · 4 ≈ 256k params
- Routing W_r: m · E = 16k params
- **Total: ~17.05M params per layer (vs dense FFN: 16.78M)** — memory parity.

**Compute cost.** Shared backbone computes once per token (T·m·dFFN ≈ 16M FLOPs); per-expert LoRA delta is T·k·m·r ≈ 4096 FLOPs/token. The LoRA delta is negligible. **Effective PRISM-LoRA compute ≈ dense FFN compute** — no compute speedup.

**Tradeoff:**

| PRISM variant | Compute vs dense (matched expressivity) | Memory vs dense | Notes |
|---|---|---|---|
| Naive (full-size experts) | 0.25× (4× faster) | 8× WORSE | wins on raw compute, loses memory |
| LoRA (r=4) | ~1× | ~1× (parity) | gains expert specialization at no overhead, no compute win |

**Honest framing**: PRISM is a **capacity-scaling** mechanism, not a compute-shrinking mechanism. Naive PRISM trades 8× memory for 4× faster training of an effectively-larger model. PRISM-LoRA gives expert specialization at parity but no obvious compute or memory win.

---

## 7. Conjectures

### Conjecture C1 (Load Balance)

With deterministic top-k routing and capacity factor c=1.25 on q-dependent routing scores at production scale:
```
Var(use_e) / E ≤ 0.10  (10% relative imbalance)
```
after the first 5000 training steps.

**Falsifiable**: instrument 66M PRISM run, log per-step expert counts, compute variance every 100 steps for steps 5000-10000.

**Plausibility**: standard MoE without auxiliary loss has ~30% imbalance; with capacity = 1.25, hard imbalance is bounded. Switch (with auxiliary loss) achieves ~5%; deterministic q-dependence might reach ~10-25% — PLAUSIBLE BUT NOT GUARANTEED.

### Conjecture C2 (Routing Stability)

Per-token expert assignment ε_t stabilizes after ~5000 steps: for fixed t in fixed sequence, ε_t at step 10000 differs from step 5000 in fewer than 30% of tokens.

**Falsifiable**: sample 1000 token positions at steps 5000 and 10000, compute mismatch fraction.

**Plausibility**: routing converges as W_r stabilizes; bistable assignments are possible but uncommon at k=2/E=8. PLAUSIBLE.

---

## 8. Composition Matrix

| Paradigm | Dimension | Compose with PRISM? | Notes |
|---|---|---|---|
| FACE (#28) | optimizer state | YES | W_r is small (16k params); apply FACE to embedding/output as today |
| SLC (#38) | T schedule | YES | independent of FFN routing |
| RLG (#39) | L schedule | YES | independent of FFN routing |
| SAS (#40) | attention sparsity | YES | attention vs FFN |
| MFIO | attn-weight optimizer | YES | attn vs FFN |
| SCFA (#42) | attention compute | YES — #42 enables PRISM motivation | PRISM attacks remaining 25% bucket |
| Kahan-v (surprise #17) | bf16 numerical | YES (inherited) | applies to expert weights identically |

---

## 9. Concrete Primitives

### 9.1 Sparse-MoE Forward
```cpp
// Inputs: q [T,m], W_r [m,E], experts[E]
// Step 1: scores
S = q @ W_r;                                 // [T,E]
// Step 2: top-k
for (t = 0; t < T; ++t) {
    eps[t]  = top_k(S[t], k);                // sorted indices
    r[t]    = softmax(S[t][eps[t]]);         // softmax over k
}
// Step 3: capacity throttling (deterministic; sort by score, ties by t)
for (e = 0; e < E; ++e) {
    cands = sort([(t,j): eps[t][j]==e] desc by S[t][e], ties by t);
    for (i = 0; i < min(C_e, len(cands)); ++i) B[e].push(cands[i]);
    for (i = C_e; i < len(cands); ++i) r[cands[i].t][cands[i].j] = 0;  // bypass
}
// Step 4: per-expert FFN
for (e = 0; e < E; ++e) Z[e] = FFN_e(q[B[e].tokens]);
// Step 5: scatter-gather
Y = zeros(T,m);
for (t = 0; t < T; ++t) for (j = 0; j < k; ++j) {
    e = eps[t][j];
    if (r[t][j] > 0) Y[t] += r[t][j] * Z[e][t_in_B];
}
```

### 9.2 Sparse-MoE Backward
1. `dY → dZ_e` per expert (scatter): `dZ_e[t_in_B] = r[t][j] · dY[t]`.
2. `dZ_e → dW_e^{up,down}, dq[B[e]]` via expert backward.
3. `dY → dr[t][j] = dY[t] · Z_e[t]` (router weight gradient).
4. `dr → dS` via softmax-over-k Jacobian.
5. `dS → dW_r, dq` via routing-layer backward; sum dq from steps 2 + 5.

### 9.3 Inverse Pass
```cpp
// q' = q (unchanged in shear). Recompute Y exactly:
Y = forward_PRISM_FFN(q', W_r, experts, capacity);
p_recovered = p' - Y;  // exact to bf16/fp32 precision under §4 caveats
```

### 9.4 Capacity Throttling
Determinism requires sorted bypass: highest s_{t,e} wins capacity, ties broken by token index. O(T·E·log T) per layer; ~80k ops at T=1024/E=8 — negligible.

### 9.5 Determinism Anchor
- Routing scores S in fp32.
- Top-k with explicit tiebreaking (lowest index wins).
- Capacity throttling with explicit tiebreaking.
- Softmax over k entries in fp32.

---

## 10. Comparison Matrix

| Axis | MELT | HYDRA | PRISM (naive) | PRISM-LoRA |
|---|---|---|---|---|
| Compute speedup on FFN | 3.2× | 1× per GPU | 4× (vs E·dFFN baseline) | ~1× |
| Memory impact on FFN | 205× compression | 1× per GPU | 8× WORSE | 1× (parity) |
| CHIRON synergy | algebraic invariance | distributed scaling | reversibility constrains routing form | same |
| Risk | TT-rank conjecture | engineering complexity | routing stability + load balance | rank-too-low collapse |
| Implementation horizon | 6-9 iter | 12-15 iter | 9-12 iter | 9-12 iter |
| Headline at 1.84B | 3.2× × 65× = 208× | 6.4× scaling | 4× × 65× = 260× | 65× (no compute gain) |
| Enables larger LLMs | YES (memory) | YES (distribution) | NO (worse memory) | NO (parity only) |

---

## 11. Honest Gaps

### 11.1 Weaker on the "Extremely Large LLMs" Axis
The user brief emphasized BOTH (i) magnitudes of compute AND (ii) extremely large LLMs.
- **MELT** wins on (ii): 205× memory compression directly enables larger models.
- **HYDRA** wins on (ii): distributed scaling enables linearly larger models per GPU.
- **PRISM** loses on (ii): naive variant makes memory worse; LoRA variant only neutral.

PRISM only wins on (i) — and only against an expressivity-equivalent dense baseline.

### 11.2 The 4× Speedup is Conditional
The 4× claim assumes the comparison is to dense FFN of inner dim E·dFFN_e. CHIRON's current baseline is dense at 8192, NOT 65536. So PRISM E=8/k=2 each at dFFN=8192 would COST 2× more compute and 8× more memory than current baseline. **Misapplied, PRISM is a regression.** Correct application: scale baseline FFN dim down (dFFN_e = 4096), use PRISM to recover and exceed expressivity vs current dim-8192. Savings come from never growing dense.

### 11.3 Routing Stability is Empirical
C2 says routing stabilizes by step 5000. In practice, MoE routing can be unstable (oscillating expert assignments). Standard MoE uses auxiliary balance loss for stability; PRISM eschews this for cleaner gradients. Whether deterministic + capacity is enough is empirical. If C2 fails, PRISM may converge slower or worse than dense.

### 11.4 Per-Expert Statistics
Each expert sees only ~25% of tokens (k=2/E=8). Effective per-expert batch size is reduced; gradient noise per expert is higher. Whether aggregate gradient (across experts) matches dense FFN gradient quality is empirical.

### 11.5 LoRA Variant's Hidden Cost
PRISM-LoRA reuses the same W_base across experts. If rank r=4 is too low, experts may not diverge meaningfully — degenerating to "dense FFN with dead weight." Mitigation: rank scan (r ∈ {4, 8, 16, 32}) at small scale.

---

## 12. Gate-0 Validation Plan (~30 GPU-min)

**Goal**: verify routing stability (C2) and load balance (C1) at small scale before 9-12 iteration full implementation.

**Build**: 4M-param transformer (no CHIRON shears, plain transformer for speed) with PRISM FFN. E=8, k=2, dFFN=512.

**Test 1 — load balance (~10 GPU-min):** train 2000 steps on synthetic data, log per-step expert utilization, compute Var(use_e)/E across steps 1000-2000.
- Pass: ≤ 0.20 (relaxed from C1's 0.10).
- Fail: > 0.30 (severe imbalance).

**Test 2 — routing stability (~10 GPU-min):** continue to 5000 steps. Sample 1000 tokens at step 2500 and 5000, compare ε_t.
- Pass: ≥ 60% of tokens share at least 1 of 2 experts.
- Fail: < 40% (oscillating routing).

**Test 3 — loss vs dense (~10 GPU-min):** same model size and steps; PRISM vs dense FFN of inner dim 4·512 = 2048.
- Pass: PRISM final loss within 0.2 nat of dense.
- Fail: > 0.5 nat gap.

**Decision**:
- All three pass → proceed to full PRISM implementation.
- Any fail → either pivot to PRISM-LoRA (more conservative) or REJECT.

---

## 13. Final Position vs MELT and HYDRA

**Honest recommendation**: if the goal is "extremely large LLMs," PRISM is the third choice.
- **MELT first** if memory compression (205×) matters and TT-rank risk is acceptable.
- **HYDRA second** if distributed scaling matters and multiple GPUs are available.
- **PRISM third** for single-GPU, fixed-model scenarios wanting raw FFN throughput.

PRISM's strongest application is **secondary** — layered on top of MELT or HYDRA after the primary memory/distribution win, to squeeze additional capacity from the FFN bucket. As a STANDALONE candidate for paradigm #44, MELT and HYDRA are stronger.

**PRISM standalone niche**: research-clean, fast to implement (9-12 iter vs HYDRA's 12-15), low engineering risk relative to HYDRA, no exotic math relative to MELT. If "fast to ship" matters more than "biggest possible win," PRISM is competitive.

---

**End of document.**
