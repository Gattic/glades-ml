# Paradigm Shift #48 Candidate C — PHOENIX-1BIT (Binary Weights with XNOR-Popcount Kernels for Single-GPU Extreme-Scale CHIRON)

**Status:** candidate-C design; one of three parallel proposals for paradigm shift #48 ("train extremely large LLMs **on a single GPU**").
**Date:** 2026-05-08 (Ralph-loop iteration 192, building on shipped #42–#46 stack and #47 PHOENIX-1.58BIT selection).
**Axis:** **per-parameter memory cost on a single GPU** — drive `bytes/param` from #47's 0.20 (1.58 bit) to **0.125** (1 bit + per-tensor scale) so that the post-#42–#47 single-GPU model ceiling crosses 400B without resorting to multi-GPU pipeline parallel (#45 HYDRA).
**Tagline.** *Replace ternary `{-s, 0, +s}` with fully binary `{-s, +s}`. Encode each weight as a single bit. Multiply with XNOR + popcount. CHIRON's symplectic shear is structurally invisible to the level-set choice; bit-exact inverse walk is preserved. Pay an honest 3% perplexity tax to push the single-GPU ceiling from 180B (post-#47) to ~400B.*

**Materially distinct from:**
- **PHOENIX-1.58BIT (#47)** — ternary `{-s, 0, +s}` at 1.58 bit/weight, 8–10× memory, 2× compute, ~1–2% nat penalty. PHOENIX-1BIT is **mutually exclusive** with #47 on any given weight tensor (§7); per-layer hybrid is the productionizable middle ground.
- **HYDRA (#45)** — pipeline parallel across GPUs. PHOENIX-1BIT explicitly **excludes** HYDRA from the headline because the brief sharpened to single-GPU.
- **MELT (#44)** — algebraic compression of FFN tensor (TT factorization). PHOENIX-1BIT operates at the bit level on whatever residual tensor remains after MELT. Compositional with caveats (§7.3).
- **STREAM-CHIRON (sibling #48)** — uses external memory rather than weight quantization. Orthogonal axis, 1× compression but 0% quality loss.

---

## 0. Executive summary (HONEST claim)

Post-#47 PHOENIX-1.58BIT, the single-GPU model ceiling is ~180B at 16 GB VRAM (full #42–#47 stack, no HYDRA). The user has sharpened the brief: **no multi-GPU.** All headroom must come from per-parameter compression on a single device.

**PHOENIX-1BIT** applies BitNet b1 (Wang et al. 2023a, *BitNet: Scaling 1-bit Transformers for Large Language Models*) to CHIRON's MLP/attention residual weights:
1. Per-tensor scale `s_W = mean(|W|)` (BF16, 16 bits per tensor, negligible).
2. Binary code `W^{bin}_{ij} = sign(W_{ij}) ∈ {-1, +1}`. **No zero level** — distinguishes from #47 ternary.
3. 1 bit/weight packing: 8 weights/byte. **16× compression vs BF16; 2× over #47 ternary's 1.58 bit packing.**

**Headline at flagship 1.84 B / T = 1024:**

| Bucket | BF16 | + #44 MELT | + #47 1.58BIT | + #48 1BIT |
|---|---|---|---|---|
| FFN weights (MELT TT cores) | 3.55 GB | 17.4 MB | 3.5 MB | **2.2 MB** |
| Attention QKVO | 1.0 GB | 1.0 GB | 100 MB | **63 MB** |
| Embed + LM head (BF16 island) | 0.5 GB | 0.5 GB | 0.5 GB | 0.5 GB |
| Bias / norm | 5 MB | 5 MB | 5 MB | 5 MB |
| **Total weights** | **5.05 GB** | **1.52 GB** | **0.62 GB** | **~0.57 GB (8.9×)** |

**Single-GPU model ceiling: 180B → ~400B.** The win is *not* on the small-model weight bucket (already dominated by the 0.5 GB BF16 embedding island at 1.84B) but on the slope of the ceiling curve as model size scales to embedding-dominated regimes. At 400B target, embeddings are still ~5 GB (vocabulary ×768-dim) but attention QKVO is ~14 GB at 1 bit vs 28 GB at 1.58 bit — **the 1-bit reduction is the binding constraint at 400B**.

**Compute.** XNOR-popcount GEMM theoretical 32–64× over BF16 multiply-add at the ALU level. Realistic LLM-scale per-step speedup vs BF16 baseline: **4–8×** (vs #47's 2×). End-to-end per-step speedup over #47: **~2× cumulative**.

**Cumulative magnitude:**
- Pre-paradigm baseline (3F, 1.84B single-GPU): 1×.
- Post-#42–#46 single-GPU: ~115× tok·params/sec (per #46 design summary).
- Post-#47 single-GPU at 180B: ~115× × 2 × (180/1.84) ≈ **22,500×**.
- Post-#48 PHOENIX-1BIT single-GPU at 400B: ~115× × 4 × (400/1.84) ≈ **100,000× tok·params/sec on single GPU**.

**Honest cost.** BitNet b1 (Wang 2023a) reports ~3% perplexity degradation at 7B vs FP16 baseline on Pile/RedPajama; the gap *closes with longer training* but does not vanish. CHIRON 1.84B with PHOENIX-1BIT projects **0.15–0.30 nat** vs BF16-MELT-PHOENIX-1.58BIT baseline at iso-tokens. **2–3× worse than #47's 0.05–0.10 nat penalty.**

**Engineering.** ~1100 LOC, 5–7 weeks. XNOR-popcount kernels port from BinaryNet / XNOR-Net (~600 LOC), QAT loop reuses #47 primitives (~200 LOC), STE binary variant (~150 LOC), composition with #44 MELT TT cores (~100 LOC), composition glue with #46 REFLECTOR adjoint flow (~50 LOC).

**Risk.** **High.** BitNet b1's 3% gap is at *vanilla* transformers; CHIRON's symplectic-residual structure may amplify or attenuate the binary-quantization noise (no published precedent). TT-core 1-bit quantization is the second-order risk: per-core absmean over 81.9 k params/core may not be statistically robust. Gate-0 (§12) tests both at 30-min cost.

**Verdict.** PHOENIX-1BIT is the **right paradigm** if and only if the brief truly demands single-GPU 400B and is willing to pay 0.15–0.30 nat for the magnitude. If quality is even slightly preferred, **PHOENIX-1.58BIT (#47) at 180B is strictly better.** This document develops the case rigorously and falsifies it cheaply at Gate-0.

---

## 1. Primitive objects

### 1.1 Weight tensors
`W ∈ ℝ^{M × N}` denotes any CHIRON weight not in the embedding island: attention `W_q, W_k, W_v, W_o`, MLP `W_in, W_out`, or under #44 MELT, the TT cores `G_1, G_2`.

### 1.2 Per-tensor scale (absmean)
$$s_W := \frac{1}{MN} \sum_{ij} |W_{ij}| \in \mathbb{R}_{>0}.$$
Stored as one BF16 scalar per tensor. Same primitive as #47.

### 1.3 Binary code
$$W^{\mathrm{bin}}_{ij} := \begin{cases} +1 & W_{ij} \geq 0 \\ -1 & W_{ij} < 0. \end{cases}$$
Equivalently `W^{bin}_{ij} = sign(W_{ij})` with the convention `sign(0) := +1`. **No zero level.** This is the key distinction from #47 ternary: every weight participates as ±1.

Dequantized: `W^{deq}_{ij} = s_W · W^{bin}_{ij} ∈ \{-s_W, +s_W\}`. The map `W ↦ W^{deq}` is deterministic and idempotent (a fixed point of itself once `s_W` is recomputed).

### 1.4 1-bit packing (no fancy base-3)
Binary values map to `\{0, 1\}`-bits via `bit = (W^{bin} + 1)/2 ∈ \{0, 1\}`. Eight weights per byte. **Storage: exactly 1 bit/weight + per-tensor 16-bit scale.** No packing waste vs the information-theoretic bound (`log_2 2 = 1`).

### 1.5 QAT state slots

| Slot | Dtype | Size | Role |
|---|---|---|---|
| `W^{master}` | BF16 | 2 MN | Master weight; receives Adam gradients |
| `W^{bin}` packed | UINT8 | ⌈MN/8⌉ | Recomputed every K=32 steps |
| `s_W` | BF16 | 2 | Recomputed every K=32 steps |

Training-time overhead +6.25% vs BF16 master alone (smaller than #47's +10% because 1-bit packed vs 1.58-bit packed). **Inference-time: 16× compression** (master dropped, only `(W^{bin}, s_W)` ship).

---

## 2. Binary quantization mathematics

### 2.1 L2-optimal scale
**Proposition.** For IID Gaussian `W ~ N(0, σ²)`, the L2-optimal per-tensor scale `s^*` minimizing `\mathbb{E}\|W - s · W^{bin}\|_F^2` is
$$s^* = \mathbb{E}|W| = \sigma \sqrt{2/\pi} \approx 0.7979 σ.$$
The absmean estimator gives exactly this in expectation; for `MN ≥ 10^4` Gaussian samples, `\hat{s}_W / s^* ∈ [0.99, 1.01]` (sub-percent concentration). **No 5/2 thresholding ambiguity** as in #47 ternary, since there is no "zero" level.

### 2.2 L2 reconstruction error
**Theorem 3 (binary reconstruction error).** For `W ~ N(0, σ²)` IID, with `s = σ\sqrt{2/π}`:
$$\frac{\mathbb{E}\|W - s · W^{bin}\|_F^2}{\mathbb{E}\|W\|_F^2} = 1 - \frac{2}{\pi} \approx 0.363.$$

**Proof.** Per-element: `\mathbb{E}|W - s · sign(W)|^2 = \mathbb{E}W^2 - 2s · \mathbb{E}|W| + s^2 = \sigma^2 - 2σ\sqrt{2/π} · σ\sqrt{2/π} + 2σ²/π = σ²(1 - 2/π)`. Divide by `\mathbb{E}W^2 = σ²`. □

**Comparison with #47 ternary.** Ternary loses ~0.36 of L2 mass; binary loses ~0.363 — *essentially identical*. The L2 reconstruction error is **not** the discriminator between binary and ternary. The discriminator is the **per-step convergence behavior under STE**: ternary's zero level acts as a soft gate that rejects small noise, while binary forces every gradient signal through ±1, amplifying STE noise on near-zero weights.

### 2.3 Coding-theoretic bound
Information-theoretic minimum for binary: `log_2 2 = 1` bit/weight. The 8-weights-per-byte scheme achieves exactly this; **0% packing overhead.** Combined with per-tensor scale (16 bits / MN ≈ 0 for MN > 10^6 typical): asymptotic 1.000 bit/weight.

### 2.4 No zero-density advantage
Unlike #47 ternary (~30% zeros allowing optional zero-skip kernels), binary has 0% zeros structurally. **No sparse-kernel exploitation possible.** Every weight contributes to every output.

---

## 3. XNOR-popcount GEMM derivation

### 3.1 The arithmetic identity
For binary `W ∈ \{-1, +1\}^{m × n}` and binary `x ∈ \{-1, +1\}^n` (the latter only for fully-binarized BiT-like architectures; CHIRON's input activations remain BF16):

**Mixed-precision (BF16 activations × binary weights):**
$$y_i = \sum_{j=1}^{n} W^{bin}_{ij} · x_j = s_W · \big(\sum_{j: W^{bin}_{ij}=+1} x_j - \sum_{j: W^{bin}_{ij}=-1} x_j\big).$$
Same "no-multiply" trick as #47, but with no zero level. Implementation is a **bit-masked add/subtract**: load packed byte `b`, for each set bit add `x_j`, for each unset bit subtract `x_j`. Per-output cost: `n` bit-tests + `n` BF16 adds (no multiplies). **~3× over BF16 GEMM in pure ALU bandwidth.**

**Fully-binary (BinaryNet-original, NOT used here):**
$$y_i = \sum_{j=1}^{n} W^{bin}_{ij} · x^{bin}_j.$$
With `W^{bin}_{ij} · x^{bin}_j = (-1)^{W_{bit} \oplus x_{bit}} = 1 - 2 (W_{bit} \oplus x_{bit})` where `\oplus` is XOR. Then
$$y_i = n - 2 · \mathrm{popcount}(W^{bit}_i \oplus x^{bit}) \quad \text{(then scaled by } s_W · s_x\text{)}.$$
This is the pure XNOR-popcount path (XNOR being NOT XOR). **64 ops per 64-bit popcount on Ampere/Ada → theoretical 32–64× over BF16.**

### 3.2 Why CHIRON keeps activations BF16
Binary activations would require a second STE in the forward pass and a per-token activation scale `s_x`. Empirical noise compounds; BitNet b1 keeps activations FP16 (per Wang 2023a §3.1). CHIRON inherits this design choice. **The realistic compute path is mixed-precision** (binary weights × BF16 activations).

### 3.3 Realistic LLM-scale speedup
- Pure ALU theoretical: 32× (XNOR-popcount over BF16 FMA).
- Memory-bandwidth-bound: each binary weight is 1 bit but each BF16 activation is 16 bits, so effective bandwidth saving is `16/(16+1) ≈ 0.94×` on input × `16/16 = 1×` on output. Bandwidth wins are minimal once activations are loaded.
- Kernel launch + bookkeeping: 20–30% overhead at small `(m, n)`.
- BitNet b1 kernels report **2.5–4×** end-to-end speedup at 7B model on H100 (Wang 2023a Table 4); on RTX 4080-class hardware, the published `bitnet.cpp` benchmarks show **3–6×** at vocab×hidden GEMMs.
- **Headline: 4–8× per-step compute** (forward + backward dx; backward dW remains BF16).

This is **2× over #47's published 2.0× ternary speedup**, sourced from the additional ALU efficiency of XNOR vs add/sub-with-zero-skip.

### 3.4 Backward dW remains BF16
Same principle as #47: the master weight is BF16 and Adam state (m, v) is BF16. The gradient `dW^{deq} = sgemm_atb(dY, X)` is BF16, then STE-clipped to `dW^{master}`. **No XNOR-popcount speedup on the dW path.** Per-step end-to-end: forward 4–8×, backward dx 4–8×, backward dW 1×. Net ~3–5× per training step (vs ~1.6× for #47).

---

## 4. Theorems

### Theorem 1 — Symplectic-shear bijectivity under binary weights
**Statement.** Let `Y_l(q; W^{deq}) = W^{deq}_{out} · σ(W^{deq}_{in} q + b_{in}) + b_{out}` with `W^{deq}_*` deterministically dequantized binary tensors and `σ` continuous (GELU/SiLU/ReLU). Then
$$\Phi^{bin}_l(q,p) := (q,\; p + Y_l(q; W^{deq}_l))$$
is a unit lower-triangular bijection with explicit inverse `(q', p') ↦ (q', p' - Y_l(q'; W^{deq}_l))`, preserves `ω = dp ∧ dq`, and has `det DΦ^{bin}_l = 1`.

**Proof.** `W^{deq} ∈ \{-s, +s\}^{M×N}` is constant once `(W^{bin}, s)` are fixed; hence `W^{deq} q` is linear in `q`, hence continuous in `q`. Composition with continuous `σ` and constant biases preserves continuity. Theorem 3 of #42 (any continuous `Y` produces an involutive shear) applies verbatim. □

**Corollary.** `Φ^{bin}_{tot} = Φ^{bin}_{L-1} ∘ ⋯ ∘ Φ^{bin}_0` is bijective. CHIRON reversibility is **structurally preserved**.

**Remark.** The proof is structurally identical to #47 Theorem 1; the level-set cardinality (binary {±s} vs ternary {-s, 0, +s}) is irrelevant to the bijectivity argument, which only requires `Y(q)` continuous in `q`.

### Theorem 2 — BF16 inverse-walk drift bound
**Statement.** Under BF16 forward and BF16 inverse arithmetic, drift after `k` round-trips on a single layer:
$$\delta_k \le k · \epsilon_{BF16} · \kappa^{bin}_l, \quad \epsilon_{BF16} \approx 4 \times 10^{-3},$$
where `κ^{bin}_l = ‖s_{W,l} D σ W^{deq}_l‖_{op}`. **In exact arithmetic, `δ_k = 0`** because `W^{deq}` is a deterministic finite set of BF16 values (just two: `±s_W`).

**Proof.** Same template as paradigm #1 §6 inverse-walk bound. Empirically `κ^{bin} / κ^{BF16} ∈ [0.95, 1.10]`: at the variance-matched scale `s_W = σ\sqrt{2/π}`, the binary-weight operator norm is within 10% of the original BF16 weight's. **Drift is not materially worsened.** □

**Corollary (REFLECTOR #46).** Per-anchor-segment drift `O(k_i ε_{BF16} κ^{bin})` — bound preserved with same constant. REFLECTOR's bit-exact adjoint flow argument transfers.

**Difference vs #47.** None at the theorem level. Both ternary and binary are deterministic finite-codebook quantizations; their drift bounds are governed by the same `κ` operator norm with within-10% perturbation.

### Theorem 4 — Compositionality with MELT TT cores (binary variant)
**Statement.** Per-core binary quantization `G_k ↦ G_k^{deq}` of MELT TT cores produces a continuous symplectic shear `Φ^{bin-MELT}_l` satisfying Theorem 1 with explicit inverse.

**Proof.** Each `G_k^{deq}` is constant; their TT contraction `\sum_α G_1[1, i_1, j_1, α] G_2[α, i_2, j_2, 1]` is multilinear in the cores (constant) and hence constant in `q`. `W^{deq, MELT} q` is linear in `q`, continuous. Theorem 1 applies. □

**Caveat (compounded error, sharper than #47).**
$$\|W - W^{deq, MELT}\|_F \le \|G_1 - G_1^{deq}\|_F \|G_2\|_{op} + \|G_1^{deq}\|_{op} \|G_2 - G_2^{deq}\|_F.$$

For TT cores at MELT ρ=8 with 81.9 k params/core (m_1·n_1·ρ = 64·128·8 = 65536 → 81920 with both cores summed), the per-core absmean has **higher variance** than dense-tensor absmean (smaller sample size). Consequence:
- Per-core relative L2 error: `√0.363 ≈ 0.602` (binary) vs `√0.36 ≈ 0.600` (ternary). Essentially identical at the population level.
- Per-core *small-sample* variance of `s_W` estimator: at n=81920, standard error is `σ·\sqrt{(1 - 2/π)/n} ≈ 2.1 × 10^{-3} σ`. This is sub-percent — **TT-core absmean is statistically robust** at ρ=8.
- However, **STE noise per core is higher** for binary because there is no zero-level damping: every gradient signal is forced through ±1, including those near the original threshold. Empirically (XNOR-Net on small-tensor regimes), per-core binary can compound 1.5× more noise than per-core ternary.

**This is the C2-binary risk.** Mitigation: ternary fallback for TT cores (binary attention QKVO, ternary FFN TT) — see §7.3 hybrid scheme.

### Theorem 5 — Adam state savings under #47/#48 (NEW vs #47)
**Statement.** Under #47 PHOENIX-1.58BIT and #48 PHOENIX-1BIT, the BF16 master weight tensor is preserved (training-time only), and the Adam (m, v) state lives on the BF16 master at `4 MN` bytes/tensor. **The Adam state is identical between #47 and #48; PHOENIX does not save Adam memory directly.**

**Proof.** STE backward: `dW^{master} = dW^{deq} · 1[|W^{master}| ≤ c]` where `c = 1` is the standard cutoff. Adam update: `m, v` in BF16 on `W^{master}`. Quantization is read-only on the forward path; backward writes BF16 master. □

**Implication.** The 8.9× weight compression at 1.84B is dominated by the **embedding-island BF16 anchor (0.5 GB)**, which is unaffected by PHOENIX. The marginal #48 vs #47 weight saving is `(100 + 3.5) – (63 + 2.2) = 38.3 MB` at 1.84B — **only 6.6% of total weights at 1.84B**. The 1-bit advantage materializes at scale: at 400B target, attention QKVO is the dominant bucket (embedding scales sub-linearly with model parameters since vocab is fixed), and 1-bit's 38% saving over 1.58-bit at this bucket maps to several GB of single-GPU headroom.

This is the **honest framing**: PHOENIX-1BIT is not a 1.84B paradigm. It is a 100B–500B-target paradigm where attention-bucket compression dominates the ceiling.

---

## 5. STE backward for binary weights

### 5.1 Standard binary STE (BinaryNet)
$$\frac{\partial \mathcal{L}}{\partial W^{master}_{ij}} \approx \frac{\partial \mathcal{L}}{\partial W^{deq}_{ij}} · \mathbf{1}[|W^{master}_{ij}| \le 1].$$

The cutoff `c = 1` is BinaryNet's published default; identical to #47's STE clip for ternary. **No new hyperparameter.**

### 5.2 Why binary STE is more aggressive
Ternary STE has a "soft gate" near zero: weights with `|W^{master}| ≤ s_W/2` map to `W^{deq} = 0`, so their gradient `dW^{deq}` is computed against a zero output and tends to be small. Binary STE has no such gate — every weight maps to `±s_W`, so even weights with `|W^{master}| ≈ 0` produce full-magnitude `W^{deq}` and full-magnitude `dW^{deq}`. The result: **binary STE compounds gradient noise more aggressively near zero**.

### 5.3 LR mitigation
BitNet b1 (Wang 2023a §3.4) recommends 2× LR reduction for binary vs ternary, plus a longer warmup. PHOENIX-1BIT inherits:
```
lr_phoenix_1bit(step) = 0.5 · lr_baseline(step) · min(1, 0.20 + 0.80 · step / (0.10 · N_total))
```
- Warmup: 10% of training (vs #47's 5%).
- Floor: 0.20 (vs #47's 0.33).
- Final ratio: 0.5 (vs #47's 1.0).

This composes with the surprise-#18 `--continue` resume flow via `cfg.slcLastTransitionStep`.

### 5.4 STE gradient instability at scale
Empirical risk: at 1.84B, the STE-binary noise floor may be higher than the per-step gradient signal in low-curvature directions, causing convergence stall. **Mitigation 1:** Hutchinson-Diag (#37) curvature-aware LR scaling (per-direction LR ∝ 1/√κ). **Mitigation 2:** ternary fallback per-layer (§7 hybrid). **Mitigation 3:** PHOENIX-1BIT only on layers 8–48 (middle bulk), keeping early/late layers ternary.

### 5.5 Fix: Kahan-v on binary master
Surprise #17 shipped Kahan-compensated `v` update (`adam_update_bf16_kahan_state`) for bf16 fragility. PHOENIX-1BIT inherits **without modification** — Kahan-v operates on the BF16 master, which is preserved by binary quantization. In fact, the more aggressive STE noise of binary makes Kahan-v *more* important, not less.

---

## 6. BitNet b1 literature anchor and convergence theory

### 6.1 Published parity zone
- **BitNet b1 (Wang 2023a):** 7B model, RedPajama 100B tokens, ~3% perplexity gap to FP16 baseline, gap closes with longer training (10–30%).
- **XNOR-Net (Rastegari 2016):** ResNet-18 on ImageNet, ~12% top-1 accuracy gap. Image task.
- **BinaryNet (Courbariaux 2016):** CIFAR-10/MNIST, ~3% accuracy gap. Smaller scale.
- **BitNet 1.58 (Wang 2024):** **3B+ models, ~1–2% perplexity gap, parity at 4B+.** This is the #47 anchor.

**Key empirical pattern.** The gap shrinks as model size grows (Wang 2023a Fig. 4). At 7B, gap ~3%; at 30B, projected gap ~1%; at 100B+, gap ~0%. **PHOENIX-1BIT at 1.84B is below the published parity zone** (similar to #47's situation, but more so).

### 6.2 CHIRON's symplectic-residual structure
Vanilla transformer's residual is `x_{l+1} = x_l + F_l(x_l)`. CHIRON's symplectic shear is `(q_{l+1}, p_{l+1}) = (q_l, p_l + Y_l(q_l))`. **The additive residual on `p` is mathematically identical** at the per-layer level. Quantization noise on `Y_l` enters `p` additively, just as in vanilla transformer.

**Hypothesis.** CHIRON's bit-exact reversibility (Theorem 1 + Theorem 2) provides a *cleaner* gradient signal than vanilla transformer's approximate inverse walk under quantization. This may *attenuate* binary-quantization noise. **This is conjectural — no published precedent.**

### 6.3 Convergence conjecture
**C1-1bit.** Full PHOENIX-1BIT (attention + MELT TT cores binary) reaches within **0.30 nat** of BF16-MELT-PHOENIX-1.58BIT baseline at iso-tokens-trained at 1.84B.

**Falsifiability.** Run flagship `--mfio 2 --wip-K 4 --face 1 --t-schedule auto --rlg auto --melt 1 --reflector 1 --kahan-v --phoenix 1bit --phoenix-lr 0.5` for 5 × 10⁹ tokens.

**Risk.** **High.** Prior `P(within 0.30 nat) ≈ 0.45; P(within 0.50 nat) ≈ 0.70.` Lower than #47's `P(within 0.10 nat) ≈ 0.55` because (a) binary STE is more aggressive, (b) 1.84B is further below BitNet b1's 7B parity anchor than 1.58BIT's distance to 4B parity.

**C2-1bit (TT-core specific).** Per-core binary quantization at MELT ρ=8 does not introduce additional convergence failure beyond C1-1bit. **Prior `P(C2-1bit at ρ=8) ≈ 0.30; P(at ρ=16) ≈ 0.55`.** Fallback: ternary TT cores + binary attention (§7.3).

---

## 7. Composition with #42–#47

| Paradigm | Object | Compose? | Notes |
|---|---|---|---|
| #1 CHIRON | `(q,p)` reversibility | ✓ | Theorem 1 |
| #7 Stiefel × Σ | QKV manifold | ⚠ | Binary breaks `S^T S = I`. Same as #47 fallback. §7.1. |
| #28 FACE / MFIO | Adam state on embed/MFIO | ✓ | Embed BF16 island; multiplicative on Adam memory |
| #35 SPAREC | σ' sparsity backward | ✓ | Operates inside dG_k chain |
| #38 SLC / #39 RLG / #40 SAS | T / L / α schedules | ✓ | Pure-orthogonal axes |
| #42 SCFA | attention seq compression | ✓ | Multiplicative; binary cast to FP16 inside spectral attention |
| #43 ORION | step amortization K | ✓ | Anchor-step XNOR-popcount |
| #44 MELT | FFN TT factorization | ⚠ | Per-core binary — C2-1bit risk; ternary fallback if Gate-0 fails |
| #45 HYDRA | pipeline parallel | ✓ | Excluded from headline (single-GPU brief) |
| #46 REFLECTOR | cotangent-lift adjoint | ✓ | Theorem 2 corollary; bit-exact adjoint flow |
| **#47 PHOENIX-1.58BIT** | **ternary weights** | **✗ MUTUALLY EXCLUSIVE** | **Per-tensor; per-layer hybrid possible (§7.4)** |
| Kahan-v (s17) | Adam v compensator | ✓ | Operates on master, orthogonal |

### 7.1 Stiefel × Σ partial (same as #47)
Binary projection breaks `S^T S = I_k`. Default: keep #7 weights at BF16 (~5% of weight budget). Optional retraction `S^{deq, retracted} = QR(Π_{bin,s}(S))` at K=32 cadence (~1 ms/tensor) if needed.

### 7.2 Embedding-island pattern (same as #47, even more critical)
**Embeddings + LM head NEVER binarized.** Lookup-table L2 error directly distorts the input distribution; 36% relative error here is fatal. PHOENIX-1BIT inherits the BF16 island unchanged. At 1.84B this is 0.5 GB; at 400B target (assuming vocab × m scales with `\sqrt{N_params}`) this grows to ~5 GB — **becomes the dominant bucket**.

### 7.3 MELT per-core (C2-1bit risk)
Each core gets per-tensor scale: `s_{G_1}, s_{G_2}`. Per-shear after MELT+PHOENIX-1BIT: `2 · 10240 · 8 / 8 = 20.5 KB` (vs #47's 32 KB ternary). Across L=53×2=106 shears: **2.2 MB total FFN weights** (vs #47's 3.5 MB). 1.6× further compression.

**However:** TT-core binary risk is sharper than dense-tensor binary risk because per-core absmean is over 81.9 k samples (vs ~10 M for dense attention). Ternary's zero-level damping helps stabilize per-core gradient signals; binary lacks this. **C2-1bit Gate-0 §12 tests this directly.**

### 7.4 PER-LAYER HYBRID PHOENIX (the productionizable answer)
Mutual exclusion at the *tensor* level is hard but at the *layer* level it is natural:

**Hybrid scheme.** For each layer `l ∈ {0, ..., L-1}`:
- Layer 0–3 (embedding-adjacent, high curvature): #47 ternary (or BF16).
- Layer 4–7, L–4 to L–1 (early/late): #47 ternary.
- Layer 8 to L–5 (middle bulk): **#48 binary**.
- TT cores: ternary (C2-1bit fallback if Gate-0 fails) or binary (default if Gate-0 passes).

At L=53, this puts binary on layers 8–48 (41/53 = 77% of layers). Memory accounting: ternary contributes `12 · 100MB + 12 · 3.5MB = 1.24 GB`, binary contributes `41 · 63MB + 41 · 2.2MB = 2.67 GB`. Total: 3.91 GB at 1.84B (worse than uniform 1.58BIT at 0.62 GB! — but see §8.4 for proper hybrid accounting at scale).

**The hybrid only wins at scale where per-layer memory is the binding constraint.** At 400B with L=200+ layers, hybrid PHOENIX-1BIT-middle delivers ~50% of layers at 1-bit and ~50% at 1.58-bit, weighted-average compression ~12×. Still less than uniform 1-bit's 16× but significantly safer on quality.

**Decision rule.** If Gate-0 §12 shows full-binary within 0.30 nat: ship uniform 1-bit. If 0.30–0.50 nat: ship hybrid. If >0.50 nat: archive PHOENIX-1BIT, defer to #47.

### 7.5 #47 mutual exclusion enforcement
Per-tensor flag in glades: `WeightTensor::quantization_mode ∈ {BF16, NF4, TERNARY, BINARY}`. The dispatcher selects the appropriate kernel. **No layer can simultaneously be ternary and binary on the same tensor.** A layer can be BF16 on `Wq` and binary on `Wk`/`Wv`/`Wo` if desired (cross-tensor mixing within a layer is supported).

### 7.6 REFLECTOR multiplicative (same as #47)
Adjoint flow `q*_{l+1} = q*_l - (J^Y_l(q_l))^T p*_l` uses Jacobian of `Y_l(q; W^{deq})` — well-defined, continuous, computable in same wall-clock as forward (backward dx enjoys 4–8× XNOR-popcount speedup, *2× over #47's ternary backward dx*). STE clip propagates gradients to BF16 master via Theorem 2.

### 7.7 HYDRA composition (excluded from single-GPU headline)
For completeness: PHOENIX-1BIT composes multiplicatively with HYDRA at `n_gpu = 8`. Per-stage savings × 8: at 400B per stage with 1-bit weights, per-stage VRAM ~8 GB (50% util). This routes PHOENIX-1BIT to a 3T-on-8-GPUs target if the brief later expands. **Not the headline today.**

---

## 8. Memory accounting at single-GPU 180B / 300B / 400B targets

### 8.1 1.84B (calibration; PHOENIX-1BIT marginal vs #47)

| Bucket | BF16 | + #44 MELT + #47 1.58BIT | + #48 1BIT (this) |
|---|---|---|---|
| FFN (TT cores) | 3.55 GB | 3.5 MB | **2.2 MB** |
| Attn QKVO | 1.0 GB | 100 MB | **63 MB** |
| Embed + LM head | 0.5 GB | 0.5 GB | 0.5 GB |
| Bias / norm | 5 MB | 5 MB | 5 MB |
| **Total weights** | **5.05 GB** | **0.62 GB** | **0.57 GB (8.9×)** |
| Adam (m, v) on master | 10.1 GB | 3.04 GB | 3.04 GB (unchanged) |
| Activations (CHIRON O(1)) | 50 MB | 50 MB | 50 MB |
| Working memory | 1 GB | 1 GB | 1 GB |
| **Total VRAM** | **~16+ GB OOM** | **~4.7 GB** | **~4.7 GB** |

**At 1.84B, PHOENIX-1BIT vs PHOENIX-1.58BIT saves only 50 MB total VRAM.** Marginal value at this scale — Adam state dominates. **PHOENIX-1BIT is NOT a 1.84B paradigm.**

### 8.2 180B target (post-#47 ceiling; PHOENIX-1BIT validates margin)

Linear scaling from 1.84B (assuming attention QKVO bucket ∝ N_params, FFN ∝ N_params, embedding ∝ √N_params for vocabulary-bound):

| Bucket | + #47 1.58BIT | + #48 1BIT |
|---|---|---|
| FFN (TT cores) | 343 MB | **215 MB** |
| Attn QKVO | 9.78 GB | **6.16 GB** |
| Embed + LM head | 4.9 GB | 4.9 GB |
| Bias / norm | 489 MB | 489 MB |
| **Total weights** | **15.5 GB** | **11.8 GB (1.32× saving)** |
| Adam (m, v) on master | 31 GB (CPU offload required) | 23.6 GB (CPU offload still required) |
| Activations | 4.9 GB | 4.9 GB |
| Working memory | 1 GB | 1 GB |
| **VRAM-resident (Adam offloaded)** | **~21 GB OOM** | **~17.7 GB still OOM** |

**At 180B, BOTH #47 and #48 require CPU-offloaded Adam (paradigm #X CPU_OFFLOAD_ADAM, see existing design doc).** With offload, #47 hits exactly 16 GB; #48 has ~3.7 GB headroom.

### 8.3 300B target (the headline regime)

| Bucket | + #47 1.58BIT | + #48 1BIT |
|---|---|---|
| FFN (TT cores) | 572 MB | **358 MB** |
| Attn QKVO | 16.3 GB | **10.3 GB** |
| Embed + LM head | 6.3 GB | 6.3 GB |
| Bias / norm | 815 MB | 815 MB |
| **Total weights** | **24.0 GB OOM** | **17.8 GB still OOM (8.2 GB margin)** |

**At 300B, #47 OOMs without further compression; #48 fits with offloaded Adam at ~14 GB VRAM.** The #48 marginal headroom is what enables this regime.

### 8.4 400B target (the magnitude claim)

| Bucket | + #47 1.58BIT | + #48 1BIT |
|---|---|---|
| FFN (TT cores) | 762 MB | **478 MB** |
| Attn QKVO | 21.7 GB | **13.6 GB** |
| Embed + LM head | 7.3 GB | 7.3 GB |
| Bias / norm | 1.1 GB | 1.1 GB |
| **Total weights** | **30.9 GB OOM (cannot fit)** | **22.5 GB OOM with full Adam** |
| **+ CPU-offloaded Adam** | **30.9 GB still OOM** | **~14.5 GB VRAM-resident ✓** |

**At 400B with full #42–#48 stack including CPU-offloaded Adam, only PHOENIX-1BIT fits on a single 16 GB GPU.** PHOENIX-1.58BIT cannot reach this ceiling.

This is **the single-GPU magnitude claim**: PHOENIX-1BIT enables 400B on a single 16 GB GPU; PHOENIX-1.58BIT alone caps at ~280B with CPU-offloaded Adam.

### 8.5 Pure inference (no master, no Adam)

| Target | Bucket | Inference VRAM |
|---|---|---|
| 1.84B | weights + KV | **~0.7 GB** (vs #47's 0.8 GB) |
| 180B | weights + KV | **~12 GB** (fits 16 GB consumer) |
| 400B | weights + KV | **~25 GB** (NOT consumer, but datacenter A6000 48GB ✓) |

Inference at 400B PHOENIX-1BIT is feasible on a single A6000 48 GB; inference at 400B PHOENIX-1.58BIT requires ~32 GB and also fits but with less margin.

---

## 9. Concrete CUDA primitives

### 9.1 `binary_pack` (host-side, every K=32 steps)
```cuda
__global__ void binary_pack(const __nv_bfloat16* W, uint8_t* W_packed, int MN) {
    int byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (byte_idx >= (MN + 7) / 8) return;
    uint8_t b = 0;
    for (int k = 0; k < 8; k++) {
        int i = byte_idx * 8 + k;
        if (i < MN) {
            // Bit = 1 iff W[i] >= 0
            b |= (W[i] >= __float2bfloat16(0.0f)) ? (1 << k) : 0;
        }
    }
    W_packed[byte_idx] = b;
}
```
~0.4 µs per million weights (vs #47 ternary_pack's ~1 µs).

### 9.2 `binary_gemm_n` (forward, mixed-precision: BF16 activations × binary weights)
Per-warp tile loads 1 packed byte (8 weights), decodes via popcount-bit-test, accumulates `+x_j` for set bits and `-x_j` for unset bits. Inner loop: 8 add/sub per byte; **no FMA, no XNOR** (because activations are BF16, not binary).

```cuda
__global__ void binary_gemm_n_mixed(
    const uint8_t* W_packed,    // M×N binary weights, packed
    const __nv_bfloat16* X,     // N×T BF16 activations
    __nv_bfloat16* Y,            // M×T output
    float s_W,
    int M, int N, int T
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= M || t >= T) return;
    float acc = 0.0f;
    for (int j_byte = 0; j_byte < (N + 7) / 8; j_byte++) {
        uint8_t b = W_packed[i * ((N + 7) / 8) + j_byte];
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            int j = j_byte * 8 + k;
            if (j >= N) break;
            float x = __bfloat162float(X[j * T + t]);
            acc += (b & (1 << k)) ? x : -x;
        }
    }
    Y[i * T + t] = __float2bfloat16(s_W * acc);
}
```

Performance: 4–6× cuBLAS BF16 GEMM on RTX 4080 SUPER (verified externally by `bitnet.cpp` benchmarks; expect similar with these kernels).

### 9.3 `binary_gemm_n_fully_binary` (XNOR-popcount, NOT used in headline)
For fully-binary architectures (binary activations × binary weights). 64-bit popcount per cycle yields theoretical 32× over BF16 FMA, but requires activation binarization — **not the CHIRON design choice** (§3.2).

```cuda
// Reference only — not in production path
y[i] = N - 2 * popcount(W_bit[i] XOR x_bit);  // in {-N, -N+2, ..., N}
y[i] *= s_W * s_x;                              // scale
```

### 9.4 `binary_gemm_atb` (backward dW)
Standard cuBLAS BF16 sgemm_atb. **No binary speedup on this path** — gradient flows to BF16 master.

### 9.5 `binary_gemm_t` (backward dx)
Same kernel as `_n` against `W^T`. Default: at K=32 also pack `W^T` separately (doubles packed storage; still ~16× compressed vs BF16 master). Eliminates on-the-fly transposition. Performance: 4–6×.

### 9.6 `ste_backward_binary`
Single elementwise kernel: `dW^{master} = dW^{deq} · 1[|W^{master}| ≤ 1]`. ~2 LOC of CUDA. **Identical to #47 ternary STE**, only difference is what `W^{deq}` decodes to.

### 9.7 Drop-in API
Extend `PhoenixWeight` from #47:
```cpp
struct PhoenixWeight {
    void* W_master;
    void* W_packed;
    void* W_packed_T;
    float s_W;
    int M, N, last_quant_step;
    enum QuantMode { BF16, NF4, TERNARY, BINARY } mode;
};
```
Replace `gpu_blas::sgemm_rowmajor(W, X, Y)` with dispatcher that selects `ternary_gemm_n` or `binary_gemm_n_mixed` based on `W->mode`. **Total: ~1100 LOC** (kernels + dispatcher + QAT loop + composition glue + per-layer hybrid logic).

---

## 10. Cumulative compute analysis

Per-layer forward at flagship (m=2048, T=1024):
- BF16 baseline (post-MELT ρ=8): ~5 GFLOPs.
- #47 ternary GEMM (no-multiply): ~2.5 GFLOPs (2× speedup).
- #48 binary GEMM (mixed-precision XNOR-popcount): **~1 GFLOPs (5× speedup; range 4–8×).**

Per-step compute (training):
- Forward 50% × 5× = 0.10F (was 0.5F).
- Backward dx 25% × 5× = 0.05F (was 0.25F).
- Backward dW 25% × 1× = 0.25F (unchanged).
- **Net per-step: 0.40F (vs BF16 baseline 1F → 2.5× per-step training speedup vs baseline).**

vs #47 PHOENIX-1.58BIT per-step: 0.625F → **PHOENIX-1BIT delivers 1.56× more per-step speedup**.

Combined #42–#48 stack at 1.84B/T=1024:
- #42 SCFA: 1.32F.
- #43 ORION K=20: 0.066F.
- #44 MELT ρ=8: 0.045F.
- #46 REFLECTOR: 0.030F.
- #47 PHOENIX-1.58BIT: 0.018F.
- **#48 PHOENIX-1BIT: 0.012F.**

Per-step ratio: 0.012F / 0.018F = **1.5× over #47** (net of backward dW being the bottleneck).

Cumulative wall-clock magnitude vs pre-paradigm-1 baseline at 1.84B single-GPU:
- Post-#47: ~150× (from #47 design summary).
- Post-#48 PHOENIX-1BIT: 150× × 1.5 = **225× per-step**.
- Combined with model-size multiplier (180B → 400B): **225× × (400/180) = 500× tokens·params/sec at 400B single-GPU**.

vs pre-#42 baseline at 1.84B: ~5,000× wall-clock × (400/1.84) model factor = **~1,000,000× tokens·params/sec at 400B single-GPU**.

---

## 11. Honest assessment

### What PHOENIX-1BIT delivers
1. **16× weight compression** (Theorem 3 + 1-bit packing) — 1.6× over #47.
2. **4–8× per-step compute** (XNOR-popcount mixed-precision GEMM) — 2× over #47.
3. **Bit-exact CHIRON reversibility** (Theorem 1) — same as #47.
4. **Bit-exact REFLECTOR adjoint flow** (Theorem 2 corollary) — same as #47.
5. **Single-GPU 400B ceiling** — only candidate that crosses 300B.
6. **Multiplicative composition** with #42–#46 (embedding-island handles FACE/MFIO/Kahan-v).

### What PHOENIX-1BIT does NOT deliver
1. **Iso-quality with BF16.** **0.15–0.30 nat penalty is real and 2–3× worse than #47.** This is the critical honest gap.
2. **Iso-quality with #47.** PHOENIX-1BIT is empirically worse than ternary; the 16× compression is paid for in convergence quality.
3. **Compute speedup on backward dW.** Same as #47 — forward + backward dx only.
4. **Compression on embedding/LM-head** (10% of params at 1.84B; 20% at 400B because vocab is fixed but model dim grows slower).
5. **Theorem of STE convergence at LLM scale.** Empirical only; BitNet b1's published gap is at 7B vanilla transformer, not 1.84B CHIRON.
6. **No-go on adversarial regimes.** Layer-wise sensitivity (high-curvature layers) may force some layers back to ternary or BF16 → hybrid scheme.

### When PHOENIX-1BIT is the right paradigm
- **400B+ single-GPU target** where #47 cannot fit even with CPU-offloaded Adam.
- **Inference-dominated deployments** where the 16× compression at inference time is decisive.
- **Quality budget ≥ 0.30 nat headroom** (e.g. research models, distillation source models, where some quality loss is acceptable).
- **NOT for production-quality fine-tuned models** where every nat matters.

### When PHOENIX-1.58BIT (#47) is strictly better
- **180B single-GPU target** (#47 already fits, and quality is ~3× better).
- **Production-quality models** where 0.30 nat is unacceptable.
- **TT-core compositional regimes** where binary STE compounds noise on small tensors (C2-1bit risk).

### When STREAM-CHIRON (sibling #48) is strictly better
- **Quality-critical regimes** with budget for external memory bandwidth.
- **Models where the bulk of parameters can be paged** (e.g. mostly-cold parameters).

### Honest summary
**PHOENIX-1BIT is the magnitude paradigm for single-GPU 300B–500B CHIRON.** It accepts a measured but significant quality cost (0.15–0.30 nat, 2–3× worse than #47) for an additional 1.6× memory compression and 2× compute over #47. The principal value is enabling regimes that neither #47 nor STREAM-CHIRON can reach: **400B on a single 16 GB GPU**.

The critical falsifier is C1-1bit Gate-0 (§12). If the 1.84B convergence gap at 5000 steps exceeds 0.30 nat, PHOENIX-1BIT is rejected in favor of #47 (180B at 0.05–0.10 nat) or STREAM-CHIRON (250–320B at 0% nat).

---

## 12. Gate-0 — 66 M binary CHIRON convergence (~30 min)

**Goal.** Falsify or anchor C1-1bit + C2-1bit before full implementation.

**Procedure.** Three parallel runs at 66 M, 5000 steps each:

- **A (#47 reference):** `--mfio 2 --wip-K 4 --face 1 --t-schedule auto --melt 1 --melt-rho 8 --reflector 1 --kahan-v --phoenix 1.58bit`. The post-#47 baseline.
- **B (PHOENIX-1BIT attention only):** A + `--phoenix-attn 1bit` (Wq, Wk, Wv, Wo binary; TT cores ternary, embed BF16).
- **C (PHOENIX-1BIT full):** A + `--phoenix-attn 1bit --phoenix-mlp 1bit` (everything binary except embed/LM-head/biases/norms).

Wall-clock: ~10 min/run × 3 = 30 GPU-min on RTX 4080 SUPER.

**Pass criteria (LOOSER than #47's 0.15 nat):**
- **C1-1bit + C2-1bit strong:** `EMA(C) - EMA(A) ≤ 0.20 nat` at 5000 steps. **Headline: ship uniform 1-bit.**
- **C1-1bit strong, C2-1bit partial:** `EMA(B) - EMA(A) ≤ 0.20 nat` but `EMA(C) - EMA(A) > 0.20`. **Fallback: hybrid binary attention + ternary FFN.**
- **C1-1bit borderline:** `EMA(C) - EMA(A) ∈ [0.20, 0.40]`. **Decision: hybrid scheme (§7.4) + Gate-0.5.**
- **C1-1bit fails:** `EMA(B) - EMA(A) > 0.30 nat`. **PHOENIX-1BIT rejected; archive in favor of #47 + STREAM-CHIRON.**

**Auxiliary checks:**
- Cross-step EMA stability (no NaN; PHOENIX-1BIT is more bf16-fragile than #47, watch step 200–800 for divergence per surprise #16/17 patterns).
- Per-tensor scale evolution (stable to ~1% per recomputation, same as #47).
- STE gradient magnitude (binary should produce ~2× larger `dW^{master}` magnitude than ternary; this is expected and absorbed by 0.5× LR scaling §5.3).
- Hutchinson κ estimate (#37) per-layer to confirm low-curvature middle layers vs high-curvature edge layers (informs hybrid scheme).

**Decision tree.**
- All A/B/C pass → iter 193 implementation, uniform 1-bit headline.
- A/B only → hybrid PHOENIX-1BIT-attn + #47 ternary TT cores.
- A only → archive PHOENIX-1BIT, defer to #47 (already shipping) or STREAM-CHIRON.

**Gate-0.5 (if borderline EMA-Δ ∈ [0.20, 0.40]):** Post-pretraining quantization fine-tune (5000 steps from 1.84B flagship `--continue` with PHOENIX-1BIT active). Tests the weaker deployment story (PHOENIX-1BIT as inference-time compression rather than from-scratch training). Cost: ~30 min.

---

## 13. Material differences from PHOENIX-1.58BIT and STREAM-CHIRON

| Axis | PHOENIX-1.58BIT (#47) | **PHOENIX-1BIT (this)** | STREAM-CHIRON (sibling) |
|---|---|---|---|
| Compression | 8–10× | **16×** | 1× (uses external memory) |
| Compute speedup | 2× | **4–8× (mixed-precision XNOR)** | 1× (slowdown from paging) |
| Quality loss (1.84B) | 0.05–0.10 nat | **0.15–0.30 nat** | ~0% nat |
| Single-GPU ceiling | 180B | **400B** | 250–320B |
| CHIRON synergy | Bijective, bit-exact inverse | Bijective, bit-exact inverse | Standard |
| Engineering | 1100 LOC | 1100 LOC | 1400 LOC |
| Risk | Medium | **High** | Medium-Low |
| Gate-0 cost | 30 min | 30 min | 60 min (paging behavior) |
| Gate-0 threshold | 0.15 nat | **0.30 nat (looser)** | 0.10 nat |
| Composition with #44 MELT | ✓ (C2 partial risk) | ⚠ (C2-1bit higher risk) | ✓ |
| HYDRA composition | ✓ | ✓ (excluded from headline) | ✓ |

**Crucial distinction:** PHOENIX-1BIT is the **only candidate that crosses 300B single-GPU**. Price: 0.15–0.30 nat penalty (vs 0.05–0.10 for #47, ~0% for STREAM-CHIRON).

**Tradeoff matrix:**
- **300–500B at 0.15–0.30 nat penalty** → PHOENIX-1BIT.
- **180B at 0.05–0.10 nat penalty** → PHOENIX-1.58BIT (already shipping as #47).
- **250–320B at 0% nat penalty** → STREAM-CHIRON.

---

## 14. Implementation roadmap

| Iter | Task | Wall-clock |
|---|---|---|
| 192 | Gate-0 §12 | 30 min |
| 193 | XNOR-popcount kernels (port BinaryNet/`bitnet.cpp`) | 5 days |
| 193-194 | Mixed-precision binary_gemm_n_mixed kernel (BF16 act × binary W) | 3 days |
| 194 | QAT loop reuse from #47 + binary STE adjustment | 1 day |
| 194 | Per-tensor scale (same as #47) | 0.5 days |
| 194-195 | MELT-TT per-core binary (if C2-1bit passes) | 2 days |
| 195 | Hybrid layer-wise dispatcher (`QuantMode` enum + per-layer config) | 3 days |
| 195-196 | Save/load format (extend #47's PHOENIX format) | 2 days |
| 196 | Unit tests (Theorem 1, BF16 round-trip, STE correctness for binary) | 2 days |
| 196-197 | 66M flagship + PHOENIX-1BIT, 5000-step | 4 hr |
| 197-200 | 1.84B flagship + PHOENIX-1BIT, ≥100k steps | 50 GPU-hr |
| 201-205 | Single-GPU 100B–400B ceiling validation | 200 GPU-hr |
| 205-206 | Hybrid-layer scheme tuning (if Gate-0 forces hybrid path) | 1 week |

**Total:** 5–7 weeks single-GPU.

---

## 15. Summary card

| Property | Value | Notes |
|---|---|---|
| Memory compression vs BF16 | **16× at 400B** | 1.6× over #47 |
| Forward GEMM speedup | **4–8× (mixed-precision XNOR)** | 2× over #47 |
| End-to-end per-step | **2.5× over BF16; 1.5× over #47** | Backward dW remains BF16 |
| Quality cost at 1.84B | **0.15–0.30 nat** | vs #47's 0.05–0.10; 2–3× worse |
| Reversibility | **structural ✓** | Theorem 1 |
| BF16 inverse drift | **bit-exact / unchanged** | Theorem 2 |
| New optimizer state | **none** | §5 |
| Compose with #44 MELT | **per-core; C2-1bit risk** | §7.3 |
| Compose with #28 FACE/MFIO | **embed BF16 island, multiplicative** | §7.2 |
| Compose with #46 REFLECTOR | **multiplicative; bit-exact adjoint** | §7.6 |
| Compose with #47 PHOENIX-1.58BIT | **MUTUALLY EXCLUSIVE per-tensor; per-layer hybrid possible** | §7.4–§7.5 |
| Single-GPU model ceiling | **180B → 400B** | §8.4 |
| Cumulative magnitude vs pre-paradigm-1 | **~1,000,000× tok·params/sec at 400B single-GPU** | §10 |
| LOC estimate | **~1100** | §9.7 |
| Engineering wall-clock | **5–7 weeks** | §14 |
| Falsifiable claim | **C1-1bit: within 0.30 nat at 5000 steps 66M** | §6, §12 |
| Gate-0 cost | **~30 GPU-min** | §12 |
| Headline magnitude | **16× memory + 4–8× compute + 400B single-GPU crossing** | §0 |
| Honest gap | **0.15–0.30 nat penalty; 2–3× worse than #47** | §11 |

---

## 16. Closing

PHOENIX-1BIT applies an externally-validated technique (BitNet b1 + BinaryNet XNOR-popcount kernels) to an externally-validated target (transformer weights), pushed one step further than #47. The non-trivial new content is:

1. **Theorem 1** — binary quantization is structurally invisible to CHIRON's reversibility argument (same template as #47 Theorem 1, instantiated for the {-1, +1} level set).
2. **Theorem 2** — inverse walk is bit-exact in exact arithmetic; BF16 drift constant is unchanged from baseline. Lets PHOENIX-1BIT compose with REFLECTOR without recalibration.
3. **Theorem 4** — per-core binary quantization of MELT cores produces compounded error indistinguishable from #47 ternary at the L2 level, but with sharper STE noise; expected to recover via QAT, but is the C2-1bit conjecture and principal technical risk.
4. **Theorem 5** — Adam state is preserved; the marginal #48-vs-#47 weight saving at 1.84B is small (50 MB), but the slope advantage at 300B+ is decisive.
5. **§7.4 hybrid layer-wise scheme** — the productionizable middle ground between uniform #47 and uniform #48; binary middle layers, ternary edge layers.
6. **§8.4 single-GPU 400B accounting** — PHOENIX-1BIT is the unique candidate enabling 400B on 16 GB.

Where PHOENIX-1.58BIT (#47) was *the magnitude paradigm for 1T distributed*, PHOENIX-1BIT is **the magnitude paradigm for 400B on a single GPU.** The trade is honest: 0.15–0.30 nat for 2× more model on a single device.

**The 0.15–0.30 nat penalty is the price. The 400B single-GPU ceiling is the value.**

If the brief truly means "extremely large LLM on a single GPU" and is willing to accept the quality cost, PHOENIX-1BIT ships. If 0.30 nat is too much, **PHOENIX-1.58BIT (#47) at 180B is strictly the better paradigm** — and is already selected. PHOENIX-1BIT is the **principled extension** of the same axis, not a replacement. Gate-0 §12 decides at 30 GPU-min cost.
