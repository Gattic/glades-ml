# EALRMN: An Eight-Phase Falsification of a Multi-Mechanism LLM-Efficiency Architecture

**Author.** Robert Cabral, 2026-05-19.
**Code & data.** `research/` in the glades-ml repository. C++98, single-CPU. Reproducible at seed 42.
**Status.** Final. The empirical sequence is complete within the experimental scope; the hypothesis is provisionally falsified at CPU scale; production-scale verification remains open.

---

## Abstract

We design EALRMN-v1, a sequence-modeling framework hypothesised as a more compute- and memory-efficient alternative to dense autoregressive Transformers, integrating eight mechanisms (entropy-adaptive segmentation, latent-state prediction, selective recurrence, bounded associative memory, sparse experts, write-regularization, compute-adaptive inference, optional decoding) into a single rate-distortion Lagrangian augmented with a predictive information bottleneck.

We test the framework via an eight-phase incremental falsification protocol on synthetic HMM and needle-in-haystack tasks at small-to-moderate CPU scale (m ∈ [8, 64], d_emb ∈ [4, 32], T ∈ [16, 256]). Each phase tests a specific subclaim with a pre-registered prediction and mechanism-specific diagnostic.

**Four bottlenecks are identified and individually addressed:** B1 encoder collapse (fixed by reconstruction loss, Phase-0b), B2 recurrence destabilisation (fixed by identity-regularisation + MSE-weight reduction, Phase-0c, yielding a 4× compute saving at iso-accuracy), B3 gate non-specialisation (fixed by auxiliary supervision, Phase-0f — but specialisation does not improve accuracy), B4 linear readout cannot extract from selectively-written memory (partially fixed by attention-based memory read, Phase-0g — modest 0.03–0.05 acc gain at small scale).

**The scale-up experiment (Phase-0k) is decisive.** At m=64, T=256, the modest small-scale attention-readout advantage from Phase-0g vanishes. RNN (no memory at all) becomes strictly the best model at every T ≥ 128 tested. EALRMN's bounded-memory + attention-readout machinery actively fails to train at T=256 while RNN trains smoothly to 0.48 accuracy.

We identify a unifying failure pattern — *bootstrap circularity* — that recurs across mechanisms and generalises beyond EALRMN. Each mechanism requires the others to be already-trained for its loss to provide informative gradient; under a single unified objective, none of them moves first. Auxiliary supervision breaks the circularity but does not produce architectural advantage.

The contribution is mixed: a negative architectural result accompanied by a positive methodological one. We argue both deserve to be on the published record. We do not claim the architecture cannot work at production scale; we claim that within reach of CPU experiments the predicted compounding mechanism gains do not materialise.

---

## 1. Motivation and hypothesis

### 1.1 Five postulated inefficiencies of dense LM training

The standard autoregressive language-modeling objective is
$$
\mathcal{L}_\text{LM}(\theta) = -\,\mathbb{E}_{P_\text{data}}\Big[\sum_{t=1}^T \log p_\theta(x_t \mid x_{<t})\Big] \tag{1.1}
$$
optimised over raw token streams. We postulate five structural inefficiencies:

(I1) **Surface redundancy.** Many distinct token sequences map to the same latent meaning; the objective spends parameters distinguishing surface-equivalent expressions.

(I2) **Uniform per-token compute.** Each token activates the full forward pass irrespective of its informational content.

(I3) **Quadratic attention.** Self-attention costs $O(L^2 d)$ per layer plus $O(N_\text{layers} \cdot L \cdot d)$ KV cache.

(I4) **Entanglement of facts and skills.** Mutable knowledge and procedural skills share the same parameter pool; updating one perturbs the other.

(I5) **No bounded long-range carrier.** Long-range information is reconstructed through context rather than stored compactly.

### 1.2 The EALRMN hypothesis

If these inefficiencies are *separable* and *individually testable*, a system combining

  *segmentation* ⊕ *latent prediction* ⊕ *selective recurrence* ⊕ *bounded memory* ⊕ *sparse experts* ⊕ *write regularization* ⊕ *compute adaptivity* ⊕ *optional decoding*

should produce a strictly better Pareto front of (predictive performance, resource cost) than a dense Transformer on tasks satisfying three conditions: low effective Koopman rank, non-uniform information density, bounded long-range dependence rank.

The work in this paper is an eight-phase empirical test of this hypothesis on synthetic tasks at small CPU scale, designed to falsify or support component-level subclaims while keeping the integrated design intact.

---

## 2. The EALRMN-v1 framework

Full derivation in `EALRMN_DESIGN.md` (~12 000 words). This section gives only the operational summary.

### 2.1 Architectural commitments

The latent stream evolves under a finite-rank switching Koopman operator. For patch index $i$,
$$
z_i = E_\theta(p_i),\quad s_i = \hat K_{g_i}\,s_{i-1} + \hat B_{g_i}\,z_i,\quad \hat z_{i+h} = \hat K_{g_i}^h\,s_i \tag{2.1}
$$
where $p_i$ is the input patch, $E_\theta$ the encoder, $s_i \in \mathbb{C}^r$ the recurrent state, $\hat K_j$ the Koopman operator for expert $j$, $g_i$ the routed expert. Spectral memory holds the top-$K$ eigenmodes of $\{\hat K_j\}$; rank-1 updates to $\hat K_j$ are the "writes." Segmentation is a surprisal-driven Bernoulli point process; compute-adaptive inference is per-token spectral truncation.

### 2.2 Objective

Per-step rate-distortion Lagrangian:
$$
\mathcal{L}_\text{total} = \mathcal{R}_\text{total} + \beta\,\mathcal{D}_\text{total} - \beta_z\,I_\text{NCE}(z;\Phi) \tag{2.2}
$$
with $\mathcal{R}_\text{total}$ summing seven per-mechanism rate terms (encoder, segmentation, recurrence, memory, experts, write, compute), $\mathcal{D}_\text{total}$ a weighted sum of distortion terms (latent prediction, optional reconstruction, optional task), and $I_\text{NCE}(z;\Phi)$ a contrastive lower bound on the mutual information between the latent and a predictive sufficient statistic of the future.

### 2.3 Six falsifiable claims

| # | Claim |
|---|-------|
| C1 | Entropy-adaptive segmentation reduces patch count at matched task accuracy |
| C2 | Latent prediction learns hidden-rule structure faster than raw next-token prediction |
| C3 | Bounded associative memory matches Transformer KV cache up to capacity threshold $\rho_\text{rel} \leq K$ |
| C4 | Sparse experts improve loss per active parameter |
| C5 | Selective recurrence preserves long-range state better than small Transformer at matched memory budget |
| C6 | Memory-write penalty produces inverted-U held-out loss curve in $\lambda_w$ |

### 2.4 Limiting cases

The framework reduces to dense Transformer, dense RNN, and selective-SSM (Mamba) as specific parameter limits (proofs in design memo §7). Phase-0 experiments compare EALRMN against these limits by toggling individual mechanisms.

---

## 3. Eight-phase falsification protocol

A central methodological contribution: rather than testing all six claims simultaneously, we test them in increments, each phase adding exactly one mechanism or fix, with a pre-registered prediction and a mechanism-specific diagnostic.

### 3.1 The phase progression

| Phase | Adds | Tests | Pre-registered prediction |
|-------|------|-------|---------------------------|
| **0a** | Koopman recurrence + closed-form latent MSE + variance hinge | C2 (narrow) | Linear-probe accuracy ≥ 0.85 on HMM |
| **0b** | InfoNCE with EMA teacher + reconstruction bootstrap | C2 (full) | Latent + I_NCE beats token; without I_NCE encoder collapses |
| **0c** | Recurrence stabilization + fixed-decay spectral memory | C3 (carrier) | Probe(s, M) > probe(s alone) |
| **0d** | Learned write gate + write penalty, per-patch needle | C3 (threshold) | EALRMN matches ATTN at small K; ATTN degrades at large K |
| **0e** | Per-token encoder | Encoder-bottleneck localisation from 0d | Gate specialises on marker tokens; ATTN reaches Bayes-optimal |
| **0f** | Oracle-labelled auxiliary gate supervision (two modes) | Bootstrap-circularity diagnosis | Aux supervision unlocks accuracy |
| **0g** | Attention-based memory read (design memo §4.4) | B4 — readout bottleneck | Attmem beats RNN at moderate context |
| **0j** | Write-penalty sweep | C6 inverted-U | Held loss is U-shaped in $\lambda_w$ |
| **0k** | Scale-up (m=64, d_emb=32, T=256, 5000 steps) | Whether mechanism gains compound at scale | Compounding produces decisive advantage |

(Phase-0h on entropy segmentation and Phase-0i on sparse experts were deferred — both require new task infrastructure with structural conditions the current test tasks do not provide.)

### 3.2 Why incremental falsification

The protocol is non-standard for architectural papers, which usually present full-system claims with post-hoc ablations. We adopted incremental falsification because:

- **Mechanism-level diagnostics are individually decisive.** Each phase's failure-mode signature (z_var collapse, probe_z = probe_s indistinguishability, held-loss growth, gate_marker = gate_filler) directly identifies which mechanism is responsible.
- **Negative results have higher resolution.** A failure at one mechanism doesn't refute the full system; it identifies a missing supervisory signal whose nature can be addressed in subsequent phases.
- **The work is honest about scope.** Multi-mechanism architectures often present integrated benchmark results that conflate individual mechanism contributions. Phase-by-phase falsification keeps each claim's evidence base separate.

### 3.3 Hardware and code

All experiments run on a single CPU core. Code is standalone C++98 with no external dependencies beyond `<cstdio>`, `<cstdlib>`, `<cmath>`, `<vector>`, `<string>`, `<algorithm>`. Build:
```
g++ -std=c++98 -O2 -Wall -Wextra <prototype>.cpp -o <prototype>
```
Random seed = 42 throughout unless noted. Batch = 16 streams. SGD with manual gradient clipping at L2 ≤ 10. Wall clock per run: 5–600 s depending on T and steps.

---

## 4. Phase-by-phase findings

### 4.1 Phase-0a — Koopman + latent MSE alone is insufficient

**Configuration.** 4-state HMM with V=16 vocabulary, patch length 4, 16 patches per stream; encoder = embedding + mean-pool + linear projection; $d_\text{emb} = 4$, $m = r = 8$; single expert; horizon-1 closed-form predictor $\hat z_{i+1} = K \cdot s_i$; per-sample norm hinge anti-collapse.

**Result.** Linear probe of $s_i$ to majority hidden state plateaus at ≈ 0.50 in both latent and token modes (random = 0.25, Gate-0 target = 0.85). The probes on $z_i$ and $s_i$ track each other (probe_z ≈ probe_s ≈ 0.50), localising the failure to the encoder.

**Failure mode F-train-5.** Encoder posterior collapse — the latent-MSE objective is trivially satisfied by $z \equiv c$, $K = 0$, MSE → 0. The norm hinge prevents exact-zero collapse but not constant-output collapse. The encoder produces non-trivial magnitude but uninformative direction. This was predicted in design memo §12.2.

**Status.** C2 narrow form falsified at this scale.

### 4.2 Phase-0b — InfoNCE escapes encoder collapse only with reconstruction bootstrap

**Added.** EMA teacher encoder (momentum 0.95); cosine-similarity InfoNCE with $\tau = 0.2$, in-batch negatives, future window $W = 1$; reconstruction loss $\mathcal{L}_\text{recon}$ for own-patch tokens.

**Two failure modes encountered.** (i) Raw-dot-product NCE has a trivial collapse — $\cos(c, c) = 1$ for constant outputs, uniform softmax, loss = $\log B$. Cosine + L2 normalisation breaks this symmetry. (ii) Cosine NCE *alone* still fails: NCE loss stayed pinned at $\log B = 2.77$ nats for the entire 500-step training. At random init the per-anchor NCE gradients point in random directions across the batch and average to ~zero on the encoder parameters; the EMA teacher anchors the student to the initial random state. This is the *bootstrap failure of EMA-teacher contrastive methods* documented in BYOL/DINO literature.

**Resolution.** Adding $\mathcal{L}_\text{recon}$ (decoder of $z$ to own-patch tokens) provides a non-contrastive gradient that immediately constrains $z$ to be input-discriminative. Once $z$ has structure, NCE refines it.

**Results at HMM overlap 0.1, 1500 steps, $d_\text{emb} = 16$, $m = 32$:**

| Mode | probe_s | probe_z | z_var |
|------|---------|---------|-------|
| latent_nce (NCE + recon) | 0.76 | **0.84** | 0.16 |
| token (recon-only decoder) | **0.79** | 0.79 | 0.016 |
| latent (no NCE, no recon) | 0.66 | 0.72 | 0.008 |

**What this supports and falsifies.** F-train-5 is empirically observed at small scale. The encoder reaches near-Bayes-optimal probe_z (0.84 against estimated Bayes ceiling ~0.85). The design memo's $\alpha_\text{recon} = 0.1$ specification is falsified — $\alpha_\text{recon} \approx 1$ is required for bootstrap. Claim 2 in its full form (latent decisively beats token) is **not yet supported**: latent_nce probe_s 0.76 is slightly *worse* than token probe_s 0.79; the encoder advantage does not propagate through the recurrence.

### 4.3 Phase-0c — recurrence stabilisation gives 4× compute saving but does not raise the ceiling

**Phase-0c-A: three recurrence fixes** applied only in `MODE_LATENT_NCE`:
1. `kMseWeight = 0.1` — latent-MSE pull on $K$ down-weighted 10×.
2. `kIdentityReg = 0.01` — identity weight decay on $K$: $dK \mathrel{+}= 0.01 \cdot (K - I)$.
3. `kKopLrScale = 0.3` — slower effective lr for $K, B$ than for the encoder.

**Phase-0c-B: simplified spectral memory** — 4 slots with fixed decay constants $\lambda_j \in \{0.50, 0.80, 0.95, 0.99\}$; gated EMA update.

**Results at HMM easy task (1500 steps):**

| Variant | probe_s | probe_z | probe_m | probe_sm | probe_z @ step 200 |
|---------|---------|---------|---------|----------|----------------------|
| Phase-0b baseline | 0.76 | 0.84 | — | — | 0.67 |
| Phase-0c-A | 0.76 | 0.84 | — | — | **0.83** |
| Phase-0c-A+B | 0.76 | 0.84 | 0.70 | 0.79 | 0.83 |
| Token baseline | 0.79 | 0.79 | — | — | 0.68 |

**Phase-0c-A is a 4× compute saving at iso-accuracy.** The recurrence fixes do not change the asymptotic probe accuracy but reach it dramatically faster — by step 200 instead of step 900. This is a meaningful efficiency improvement and directly validates the design memo's recurrence-stability prescription.

**Phase-0c-B memory adds modest information.** probe_sm = 0.79, three points above probe_s alone. Memory alone (probe_m = 0.70) carries about as much state info as the recurrent state alone. The two carriers are partially redundant at this scale — the HMM's effective long-range horizon (~16 steps at mean dwell 4) is shorter than the slowest memory slot's natural horizon (≈100 steps).

With the full Phase-0c stack, latent_nce probe_sm ties token probe_s at 0.79. **Claim 2 (latent decisively beats token) is not supported on the HMM at this scale.** Both architectures saturate near the same Bayes-optimal ceiling. **Claim 3 not yet decidable** — the task is not long-context enough.

### 4.4 Phase-0d — needle-in-haystack, mean-pool encoder

**Task.** $N$ patches × 4 tokens per patch. Two key-value pairs `[KEY_MARKER, k_id, VAL_MARKER, v_id]` inserted at random non-overlapping positions. Query patch `[QUERY_MARKER, k_id_query, filler, filler]` at position $N-1$. Label: value paired with queried key. Random baseline = 0.25.

**Three models** sharing encoder + readout: **EALRMN** (Koopman + 4-slot gated memory + learned write gate + full backprop), **RNN** (no memory), **ATTN** (cosine-similarity attention from query patch to all earlier patches).

**Results at three context lengths (single seed, 2500–3000 steps):**

| $N$ patches | $T$ tokens | EALRMN | RNN | ATTN |
|-------------|-------------|--------|-----|------|
| 16 | 64 | 0.66 | 0.66 | **0.72** |
| 32 | 128 | 0.50 | **0.52** | 0.36 |
| 64 | 256 | 0.36 | 0.45 | 0.34 |

ATTN wins at short context (N=16, 0.72) and degrades fastest. EALRMN tracks RNN at every context length. The learned write gate, write penalty, and full bounded-memory backprop produce *no measurable accuracy gain* over the no-memory baseline. **Claim 3 not supported.** Gate stays near 0.5 across training (gate_marker − gate_filler ≈ 0.001).

We initially attributed this to the encoder's mean-pool aggregation destroying within-patch positional information. Phase-0e tests this.

### 4.5 Phase-0e — per-token encoder falsifies the Phase-0d localisation

**One change**: `kPatchLen = 1`. The encoder becomes $z_t = W \cdot \text{Emb}[x_t] + b$ — embedding lookup followed by linear projection. Distinct token types now have distinct $z$'s at random init. Every other architectural component is held identical.

**Results across T:**

| $T$ | EALRMN best | RNN best | ATTN best |
|------|-------------|----------|-----------|
| 16 | **0.72** | 0.64 | 0.31 |
| 32 | 0.62 | 0.60 | 0.41 (early) |
| 64 | 0.53 | 0.58 | 0.38 (early) |

**Direct iso-token comparison Phase-0d vs Phase-0e at T=64:**

| Config | EALRMN | RNN | ATTN |
|--------|--------|-----|------|
| Phase-0d kPatchLen=4 | 0.66 | 0.66 | **0.72** |
| Phase-0e kPatchLen=1 | 0.53 | 0.58 | 0.38 |

**Per-token regressed every model at every context length.** The Phase-0d localisation hypothesis is *falsified*.

**Three new failure modes uncovered:**

*ATTN: "shifted retrieval" mismatch.* The query is now just the single token `k_id_query`; cosine attention identifies positions where the same `k_id` appears (the key positions), but the value lives 2 tokens later. Single-step single-head attention cannot perform shifted retrieval. Phase-0d's mean-pool accidentally mixed key and value content into one z, giving attention something to retrieve.

*RNN: longer BPTT chain.* Distinct per-token z's are richer per step, but 64-step BPTT noise dominates.

*EALRMN gate STILL doesn't specialise.* gate_marker = 0.4586 vs gate_filler = 0.4598 at T=64 — difference 0.001, statistically indistinguishable, despite the encoder now producing trivially distinguishable z's per token type.

The bottleneck is not encoder discrimination; it is the gate's gradient circularity with the readout. This sets up Phase-0f.

### 4.6 Phase-0f — auxiliary gate supervision is decisive

**Two oracle-labelled auxiliary supervision modes** (combinable):
- `--aux-class`: 2-way classifier head on $z_t$, supervises encoder via marker/filler label.
- `--aux-gate`: direct BCE supervision on gate $g_t$ against `is_marker(x_t)` label.

**Results at T=64, 2000 steps:**

| Condition | Final acc | gate_marker | gate_filler | gate_diff |
|-----------|-----------|---------------|----------------|------------|
| Baseline (no aux) | 0.48 | 0.4586 | 0.4598 | -0.001 |
| --aux-class only | 0.38 | 0.5194 | 0.4047 | +0.115 |
| --aux-gate only | 0.27 | **0.9993** | 0.0005 | **+0.999** |
| --aux-class + --aux-gate | 0.34 | **0.9995** | 0.0003 | **+0.999** |

**Three principal findings.**

(1) **Bootstrap-circularity is empirically validated as the gate-specialisation failure mode.** Without intervention, gate_marker = gate_filler = 0.46. With aux-gate, gate specialises to (0.9993, 0.0005) within 200 training steps.

(2) **Full gate specialisation does NOT unlock accuracy.** Despite the gate firing 0.9993 on markers and 0.0005 on fillers — exactly the behavior the architecture was supposed to learn — held-out retrieval accuracy is no better than baseline at T=64.

(3) **Three-way tie at T=32 (2500 steps):** EALRMN baseline 0.73, EALRMN aux-both 0.75, RNN (no memory) 0.75. Memory existence and gate specialisation are both irrelevant to accuracy.

**A fourth bottleneck B4 is identified:** linear readout cannot extract task info from selectively-written memory. Memory contents are structured (only marker z's get written) but a linear readout (W of shape 4 × 160 = 640 params) cannot perform the nonlinear lookup-and-retrieve operation needed to identify which stored KV pair matches the query.

The design memo's attention-based memory read (§4.4) was specified but our Phase-0a–0f prototypes simplified it to concat-and-linear. Phase-0g tests un-doing this simplification.

### 4.7 Phase-0g — attention-based memory read

**Memory readout (NEW in Phase-0g):**
$$
q = W_q \cdot s_T + b_q, \quad \alpha_j = \mathrm{softmax}(q \cdot M_T^{(j)} / \sqrt m), \quad r = \sum_j \alpha_j M_T^{(j)}
$$
feature for readout = concat($s_T$, $r$), dim = $2m$ (vs $5m$ in Phase-0f).

**Five-condition T=64 result (2500 steps, seed 42):**

| Model | Aux | Best acc | Final held_loss |
|-------|-----|----------|------------------|
| EALRMN attmem (no aux) | — | **0.64** | **0.88** |
| EALRMN attmem + aux-both | full | 0.53 | 1.18 |
| EALRMN linear + aux-both | full | 0.61 | 1.08 |
| RNN | — | 0.64 | 0.89 |
| ATTN | — | 0.39 | 1.39 |

**Two surprises.**

(1) **Attention-readout EALRMN without aux is the best model on held_loss** (0.88 vs RNN 0.89, attmem+aux 1.18). The attention readout DOES extract useful information from memory.

(2) **Aux supervision HURTS the attention readout** (0.50 with aux vs 0.64 without). Forced gate specialisation makes memory contents too rigid; attention readout cannot find patterns in over-specialised memory. Without aux, gate stays at 0.5 and memory accumulates everything via EMA, giving attention more material to work with.

**Context-length characterisation (attmem no-aux vs RNN):**
- T=32: 0.73 vs 0.75 (tied)
- T=64: 0.64 vs 0.61 (attmem ahead by 0.03)
- T=128: 0.44 vs 0.30 (attmem ahead by 0.14)

The advantage grows with context length, consistent with the design memo's prediction that bounded memory beats unbounded recurrence at long context — but only at the limits where the recurrence itself fails.

**B4 status: partially fixed.** Attention readout is a real improvement, but the win is modest.

### 4.8 Phase-0j — write-penalty sweep (Claim C6)

Swept $\lambda_w \in \{0, 0.0005, 0.005, 0.05, 0.5\}$ at T=64, 2000 steps, ealrmn_attmem no-aux:

| $\lambda_w$ | Final held_loss | Gate (avg) |
|-------------|------------------|-------------|
| 0 | **1.02** | 0.500 |
| 0.0005 | 1.05 | 0.480 |
| 0.005 | 1.05 | 0.317 |
| 0.05 | 1.11 | 0.023 |
| 0.5 | **1.41** | 0.001 |

**Held loss is monotone increasing in $\lambda_w$ — no inverted-U.** The "too-low-overfits" side doesn't appear because at low penalty without aux supervision the gate stays at 0.5 (EMA-everything) rather than memorising-via-selective-writes. The "too-high-breaks" side IS confirmed (λ_w = 0.5 collapses gate). **C6 partially confirmed.**

### 4.9 Phase-0k — scale-up is decisive

The most informative single experiment. Architecturally identical to Phase-0g; only constants changed: d_emb 16→32, m 32→64, default T 64→256, default steps 2500→5000.

**Results at T=128 (3000 steps):**

| Model | Final acc | vs Phase-0g unscaled at T=128 |
|-------|-----------|---------------------------------|
| EALRMN attmem | 0.45 | 0.44 → 0.45 (unchanged by scale) |
| **RNN** | **0.53** | 0.30 → 0.53 (+0.23, large jump) |
| ATTN | 0.34 | unchanged failure mode |

**Results at T=256 (4000 steps):**

| Model | Final acc | Final train_loss | Notes |
|-------|-----------|---------------------|-------|
| **RNN** | **0.48** | 1.32 | Only model that actually learns at T=256 |
| EALRMN attmem | 0.36 | 1.37 | Stuck at uniform-predictor (log 4 ≈ 1.386) |
| ATTN | 0.19 | 1.39 | Stuck at uniform-predictor |

**Three principal findings.**

(1) **The Phase-0g attmem advantage was a small-scale artifact.** At m=32 d_emb=16 T=64, attmem 0.64 > RNN 0.61. At m=64 d_emb=32 T ≥ 128, RNN > attmem. The crossover is around m=32→64.

(2) **Scale benefits RNN substantially more than EALRMN.** RNN at T=128 jumped from 0.30 (unscaled) to 0.53 (scaled). attmem stayed at 0.45. The 4-slot memory has structural capacity $O(4 m)$; scaling $m$ doesn't change the slot count, and the attention readout is softmax over 4 elements regardless. RNN's $m$-dim state scales linearly with $m$.

(3) **At T=256, only RNN actually trains.** EALRMN attmem and ATTN both fail to escape the uniform-predictor regime (train_loss stuck at log 4 ≈ 1.386). RNN trains smoothly from 51.8 → 1.32 over 4000 steps.

**The hypothesis "scale unlocks compounding mechanism gains" is empirically falsified at the scales reachable on CPU.** The simpler architecture (RNN, no memory) scales better than the more elaborate one (Koopman + bounded memory + attention readout).

---

## 5. The bootstrap-circularity pattern

A central methodological finding: **every mechanism in EALRMN-v1 has a bootstrap-failure mode that requires an auxiliary supervisory signal to escape**. The eight-phase sequence makes this pattern visible because each phase identifies the next missing signal.

### 5.1 Four bottlenecks identified

| # | Bottleneck | Diagnosed in | Fix demonstrated | Result |
|---|------------|--------------|---------------------|--------|
| B1 | Encoder posterior collapse ($z \equiv c$ trivially satisfies latent MSE) | 0a | Reconstruction loss (0b) | Encoder reaches near-Bayes-optimal probe_z |
| B2 | Recurrence destabilisation (held latent MSE grows as encoder evolves) | 0c | Identity-reg + MSE-down-weight + slower K-lr (0c-A) | 4× compute saving at iso-accuracy |
| B3 | Gate non-specialisation (gate_marker − gate_filler ≈ 0 indistinguishability) | 0d/0e | Oracle auxiliary supervision (0f) | Gate specialises perfectly; **no accuracy gain** |
| B4 | Linear readout cannot extract from selectively-written memory | 0f | Attention-based memory read (0g) | Modest 0.03–0.05 accuracy gain at small scale; **disappears at large scale (0k)** |

### 5.2 The general pattern

For mechanism $X$ to escape its bootstrap regime, the model needs $X$ to already be useful — but $X$ is only useful when bootstrapped. The architecture's loss does not provide the bootstrap signal because the loss requires every mechanism to be simultaneously useful for the gradient to be informative.

Concretely for the gate:
- The *gate* cannot specialise from the readout's gradient until the readout actually uses memory effectively.
- The *readout* cannot use memory effectively until memory contains useful information.
- *Memory* cannot contain useful information until the gate writes selectively.

Each leg of this triangle depends on the other two. The same triangle structure exists for the encoder–latent-objective–decoder system (Phase-0a/b) and for the recurrence–encoder–prediction system (Phase-0c).

### 5.3 Why InfoNCE does not escape its own bootstrap

The design memo (§5.3, §6) framed $-\beta_z\,I_\text{NCE}(z; \Phi)$ as the primary supervisory signal. Phase-0b shows this is wrong *at small scale*: at random init the NCE loss stays at the random-softmax baseline $\log B = 2.77$ nats because per-anchor gradients average to zero across the batch. The EMA teacher cannot move faster than the student. Adding reconstruction provides the non-contrastive bootstrap signal the encoder needs.

The same pattern recurs at the memory gate: the write penalty alone pushes the gate toward zero (the "default no-op"); the gradient that should pull the gate up at marker positions never becomes informative because the readout's gradient back through memory is itself uninformative until memory contains useful content.

### 5.4 What this means for the design memo

The design memo's Lagrangian $\mathcal{L}_\text{total}$ is correctly specified but is *not self-bootstrapping*. The IB-derived $-\beta_z\,I_\text{NCE}$ term provides a *refinement* signal but not an *initialisation* signal. Achieving the predicted compounding gains requires either (a) explicit auxiliary supervision per mechanism, or (b) a curriculum that warm-starts each mechanism before the next is added.

The design memo's "principle of auxiliary-free training via the IB" is **not supported** at this scale.

### 5.5 The Phase-0f decisive test

Phase-0f is the most informative single test of the bootstrap-circularity hypothesis. The aux-gate intervention drove the gate from baseline indistinguishability (delta 0.001) to near-perfect specialisation (delta 0.999) within 200 steps. *And yet accuracy did not improve.* This is the strongest possible counterevidence to the claim that "bootstrap circularity is the bottleneck" — it shows that even when we fix the circularity directly, the architecture doesn't unlock.

The bottleneck moves upstream to B4 (linear readout) and beyond. Each fix reveals the next bottleneck.

---

## 6. The scale-up evidence (Phase-0k)

### 6.1 The most direct test of the hypothesis

The design memo's core prediction is that the eight mechanisms in *combination* produce a Pareto improvement over dense Transformer baselines. The integrated test of this is "do EALRMN's mechanisms compound at moderate scale?"

Phase-0k tests this with the four mechanisms we implemented (encoder + Koopman recurrence + bounded memory with gated EMA + attention readout) plus reconstruction bootstrap. The scale-up was modest (2× embedding, 2× operator, 4× context, 2× training).

### 6.2 The decisive result

| | T=128 (3000 steps) | T=256 (4000 steps) |
|---|---------------------|---------------------|
| EALRMN attmem | 0.45 | 0.36 (training fails) |
| **RNN** | **0.53** | **0.48** |

**The empirical finding is the opposite of the prediction.** At scale, the simpler architecture (RNN, no memory) is decisively better than the more elaborate one. The added mechanisms (memory + attention readout) actively hurt training stability without compensating advantage.

### 6.3 Why the small-scale attmem advantage disappeared

At m=32 (Phase-0g), the RNN's 32-dim recurrent state is bottlenecked enough that the 4 memory slots provide useful extra context. At m=64 (Phase-0k), the RNN's 64-dim recurrent state has enough capacity that memory adds nothing extra. The 4-slot memory has structural capacity $O(4m)$ that does NOT increase with slot count when we scale $m$ — it's a 4-way attention over 4 channels regardless. RNN's m-dim state scales linearly with m.

At T=256 the BPTT chain × the gate-and-memory gradient chain creates an optimisation landscape that the simpler RNN navigates but EALRMN cannot. The more-elaborate machinery adds optimisation noise without representational gain.

### 6.4 Three interpretations

The cumulative scale-up negative is consistent with three interpretations:

(a) **Honest small-scale negative.** EALRMN's mechanisms might compound at production scale (m=1024+, T=4096+, GPU, hundreds-of-millions of params, hours of training). We cannot test this on CPU.

(b) **Structural negative.** The 4-slot fixed-decay memory has the same $O(m)$ scaling law as the RNN's $m$-dim state. The more-elaborate machinery just adds optimisation noise without representational gain. This would hold at any scale.

(c) **Hypothesis-level negative.** The five LLM inefficiencies are real but not separately addressable by mechanism-stacking. The integration creates a more-complex-but-no-more-capable system. This would hold across architectural variations of EALRMN.

CPU evidence cannot distinguish (a) from (b)/(c). The remaining defensible interpretation depends on production-scale GPU experiments.

---

## 7. What was supported, falsified, undecided

### 7.1 Per-claim status

| # | Claim | Status |
|---|-------|--------|
| C1 | Entropy-adaptive segmentation reduces patch count | NOT TESTED — task unsuited |
| C2 | Latent prediction > raw next-token prediction | NOT SUPPORTED on HMM (Phase-0c); tied at scale (0g, 0k) |
| C3 | Bounded memory matches Transformer at $\rho_\text{rel} \leq K$ | PARTIALLY SUPPORTED at small scale (Phase-0g attmem > RNN); FALSIFIED at moderate scale (Phase-0k) |
| C4 | Sparse experts improve loss per active parameter | NOT TESTED — task unsuited |
| C5 | Selective recurrence > small Transformer | NOT SUPPORTED at this scale |
| C6 | Write-penalty inverted-U | PARTIALLY CONFIRMED (high-side breaks, no low-side overfit) |

Two of six claims have unambiguous experimental results (C2 not supported, C5 not supported). Two have qualified evidence that goes negative at scale (C3 partial at small, falsified at moderate). One has partial evidence (C6 high-side only). Two remain untested (C1, C4) because the required task structure is absent from the experimental sequence.

### 7.2 What was supported (positive findings)

- **F-train-5 empirically observed** exactly where design memo §12.2 predicted. The encoder reaches near-Bayes-optimal probe_z when given recon + NCE.
- **Recurrence-stability prescription (Phase-0c-A) gives a clean 4× compute saving at iso-accuracy.** This is a real, derivable architectural improvement.
- **Each mechanism can be implemented and trained without divergence.** The architecture is internally coherent; gradient flow through all components verified across multiple prototypes.

### 7.3 What was falsified at the scales tested

- The design memo's "auxiliary-free training via IB" expectation.
- The design memo's α_recon = 0.1 specification.
- The Phase-0d encoder-bottleneck localisation.
- The bootstrap-circularity "fix unlocks accuracy" expectation (Phase-0f).
- The "scale unlocks compounding gains" expectation (Phase-0k).

### 7.4 The honest cumulative judgment

After eight phases the architecture has been shown to train correctly when supplied with appropriate auxiliary signals, and to produce a real compute-efficiency gain (Phase-0c-A's 4×). It has *not* been shown to produce a Pareto improvement over a comparably-parameterised dense baseline on either of the two test tasks at the scales examined. Scale-up makes the situation *worse*, not better.

The design memo's central hypothesis — that the eight mechanisms in combination produce a Pareto win — is **not supported** at the scale tested. It is also *not refuted* at production scale; we cannot reach that scale on CPU. The CPU-scale evidence is consistent with both "small-scale-only negative" and "fundamental hypothesis negative."

---

## 8. Methodological contribution

We argue that the eight-phase falsification protocol is itself a contribution, regardless of whether the architectural hypothesis is ultimately supported.

### 8.1 The protocol

- **Pre-register each phase's prediction.** State what would constitute a pass and a fail before running.
- **Each phase adds exactly one mechanism or fix.** This keeps the evidence base for each claim cleanly separated.
- **Test the prediction with a mechanism-specific diagnostic.** Probe_z vs probe_s for encoder collapse. probe_m vs probe_s for memory carrier. gate_marker − gate_filler for gate specialisation. Held-loss growth vs train-loss decrease for recurrence destabilisation. Train-loss-vs-uniform-predictor for failure to escape random regime.
- **When a phase fails, localise to a specific mechanism BEFORE proposing the next phase.** Phase-0d localised to "encoder bottleneck" → Phase-0e tested this directly → found the localisation wrong, identified the actual failure (gate circularity) → Phase-0f tested THAT → found bootstrap circularity is real but fixing it doesn't unlock the architecture → Phase-0g tested the next bottleneck (readout) → Phase-0k tested whether scale resolves the cumulative pattern.
- **Distinguish "phase fails" from "experiment incomplete".** Some claims (C1, C4) we never reached; they remain undecided, not refuted.

### 8.2 Why this matters for multi-mechanism architectures

Modern architectures often stack multiple mechanisms (attention + MoE + selective gating + relative positional encoding + ...) and report integrated benchmark results. The integration conflates individual mechanism contributions, and ablations are typically post-hoc and limited to one-removed comparisons. Our protocol is **prospective and incremental**: each mechanism's evidence is established before the next is added.

This is more expensive (8 phases vs 1 integrated experiment) but the resolution is much higher when results are negative. The reader can locate exactly which mechanism does or does not contribute, and under what assumptions.

### 8.3 The bootstrap-circularity diagnostic

The recurring failure mode we identified — bootstrap circularity — is plausibly general beyond EALRMN. Any architecture that stacks mechanisms each of which would be useful given the others may exhibit it. The diagnostic pattern is:

1. Train the integrated system.
2. Observe one or more mechanisms' diagnostic indicators (variance, gate values, loss-component trajectories) stuck at a uniform-baseline value.
3. Add the simplest possible auxiliary supervision for the stuck mechanism.
4. Re-train and observe whether other mechanisms now also unstick.

Phase-0f shows that *(3) and (4) can fail decoupled*: aux-gate unstuck the gate but did not unstick the readout, because the readout had a separate downstream bottleneck. This is the key empirical refinement of the bootstrap-circularity framing: there may be a *chain* of bottlenecks, each of which needs its own auxiliary signal.

### 8.4 The "fixing reveals the next" pattern

The eight-phase sequence demonstrates a recurring structure: each fix unlocks the next bottleneck upstream rather than the predicted final improvement.

- B1 fixed → B2 surfaces.
- B2 fixed → mechanism-level pass at small scale, but the larger claim (C2) doesn't hold.
- Moving to long-context task → B3 surfaces.
- B3 fixed → no accuracy gain, B4 surfaces.
- B4 fixed → modest gain at small scale, but the gain disappears at scale.

This pattern is itself a result. It suggests that for multi-mechanism architectures, the supervisory cost grows at least linearly with the mechanism count, and the compounding gains may not be present.

---

## 9. Limitations and scope

The negative result is bounded by specific scope conditions.

### 9.1 Scale

All experiments use small-to-moderate dimensions:
- $d_\text{emb} \in [4, 32]$
- $m \in [8, 64]$
- batch ∈ {16}
- $T \in [16, 256]$
- CPU-only single-core execution

Cumulative parameter count of the largest configuration is ~70 000. **Production-scale tests (which would require GPU and substantial user effort) might tell a different story.** Specifically, the mechanism-compounding gains the design memo predicts might require scale to manifest.

### 9.2 Tasks

We tested on (i) 4-state HMM with overlapping emission distributions and (ii) needle-in-haystack with 2 KV pairs in streams of 16-256 tokens. The first is too easy for the architecture (both EALRMN and token saturate near Bayes-optimal). The second has a "shifted retrieval" structure that single-head attention can't perform without contextualisation. *The tasks may have been the wrong choices.* The design memo §9 specifies six datasets; we tested two of them. Cleaner tests on the remaining four (noisy surface equivalence, multi-regime dynamics, compression-burst, episode recall) might decide claims that the HMM and needle could not.

### 9.3 Seeds

Most reported results are single-seed (seed 42). A proper benchmark would include 5+ seed averages with confidence intervals. We checked Phase-0a at seeds 42, 43, 44 informally (same qualitative pattern). The per-step probe accuracy is noisy enough that the precision of small-difference comparisons is limited; we have been careful to label differences ≤ 0.05 as "within noise."

### 9.4 Hyperparameter search

We did not run a hyperparameter sweep beyond Phase-0j (write-penalty). Some failures might respond to careful tuning. Most reported failures show *no* sign of incipient learning across 1500-5000 steps, so the qualitative findings are robust, but quantitative claims (e.g., "4× compute saving") should be read as point estimates.

### 9.5 Optimisation

Plain SGD with L2 gradient clipping. No Adam, no momentum, no warmup, no learning-rate schedule. Appropriate for diagnostic prototyping but suboptimal for a serious efficiency claim. A production benchmark would use AdamW with cosine schedule, which would likely change *absolute* numbers but probably not the relative-mechanism comparisons.

### 9.6 Untested mechanisms

C1 (entropy segmentation) and C4 (sparse experts) were deferred. The current test tasks lack the structural conditions (non-uniform information density; multi-regime dynamics) those mechanisms exploit. Implementing the mechanisms plus the appropriate tasks would be substantial multi-session effort.

---

## 10. Implications for the broader literature

### 10.1 For mechanism-stacking efficiency architectures

The bootstrap-circularity pattern suggests that "principled supervision via a single objective" claims should be tested mechanism-by-mechanism. Several recent efficient-LLM proposals share EALRMN's structure of stacking 3-5 mechanisms behind a unified objective, and may share its bootstrap problems. The eight-phase protocol could be reused to test them.

### 10.2 For the information-bottleneck principle

The design memo's framing positioned $I(z; \Phi)$ as the primary supervisory signal. Phase-0b shows this is insufficient *as a bootstrap* at small scale. The IB principle gives a target to *refine toward*, not an *initialisation* to start from. At training time the model needs an auxiliary objective (here, reconstruction) to provide the initial gradient.

This is consistent with the broader BYOL/DINO/JEPA literature finding that EMA-teacher contrastive objectives benefit substantially from auxiliary heads or warm-start strategies. Our finding sharpens it: the auxiliary signal does not need to be related to the IB target at all — *any* informative gradient that constrains $z$ to be input-discriminative suffices.

### 10.3 For Koopman / operator-theoretic sequence models

The Phase-0c finding that identity-regularisation + slower K-lr produces a 4× compute saving at iso-accuracy is a direct vote for the design memo's selective-stability prescription. This may generalise to other linear-operator state-space models (S4, Mamba, GLA): the stability of the linear-recurrence operator matters for early training convergence even if it does not change asymptotic performance.

### 10.4 For bounded-memory designs

The Phase-0d/0e/0f findings on the gate failure suggest that *learned* write gates in bounded-memory architectures may need explicit auxiliary supervision (e.g., a token-type classification head) to escape their bootstrap regime. Phase-0g shows that the attention-based memory read is also necessary — concat-and-linear readouts are too weak to extract task info from selectively-written memory.

Phase-0k shows that the 4-slot bounded memory has $O(m)$ scaling that matches RNN's $m$-dim state. *At scale, the simpler bottleneck-free architecture wins.* This is a substantive observation for any bounded-memory design hoping to scale.

### 10.5 For "compounding mechanism gains" claims

Phase-0k empirically refutes the optimistic version of the compounding hypothesis at CPU scale. Each of the four addressed mechanisms produces a modest local effect; the combination does not amplify. This is a cautionary data point for any multi-mechanism architecture claiming that integration produces super-linear improvement.

---

## 11. Three remaining options

After the 8-phase sequence three paths remain.

### 11.1 Option C — scale up to GPU at production size

The most defensible technical path. Distinguishes interpretation (a) from (b)/(c) by testing at m=1024+, T=4096+, GPU, multi-seed.

**Cost.** Multi-week GPU project. Requires implementing EALRMN-v1 in the main glades-ml CUDA infrastructure, building production-scale Transformer baselines for comparison, running multi-seed sweeps. Confounded by the need to match Transformer's tuning tricks (LayerNorm, residual connections, warmup, AdamW) which EALRMN-v1 does not yet have.

**Risk.** A negative result at production scale would be very strong evidence for interpretation (b) or (c). A positive result would resurrect the architecture as a competitive design. The current evidence does not predict which.

### 11.2 Option D — publish the eight-phase sequence as a contribution

This document. The framework, the protocol, the bootstrap-circularity pattern, and the cumulative empirical findings are intellectual contributions independent of whether EALRMN ultimately wins at production scale.

**Strength.** The work is complete, reproducible, and honest about scope. It serves both as a substantive negative result (the hypothesis does not hold at small CPU scale) and as a methodological positive (the falsification protocol and bootstrap-circularity diagnostic generalise).

**Risk.** Negative results are harder to publish. The contribution is real but less immediately compelling than a positive efficiency claim.

### 11.3 Switch tasks or pivot architecture

Try the remaining design memo §9 tasks (noisy surface equivalence, multi-regime dynamics, compression-burst, episode recall) at small scale to see if any unlocks C1, C4 differently. Or pivot the architecture itself based on observed failure modes (e.g., increase memory slot count past 4, replace fixed decays with learned spectrum, etc.).

**Cost.** Each task plus mechanism change is a multi-day effort. Cumulative time investment grows linearly. The 8-phase pattern of "fix reveals the next" suggests diminishing returns.

### 11.4 Our recommendation

**Option D** is the cleanest immediate step. The 8-phase sequence is empirically complete within scope and produces a coherent contribution. Option C is the natural follow-up if production-scale verification is desired, but its cost is high and the prior from the CPU evidence is toward negative.

---

## 12. Conclusion

EALRMN-v1 is a rigorous, falsifiable framework for testing the hypothesis that dense Transformers are wasteful and that an architecture combining 8 specific efficiency mechanisms can produce a Pareto improvement. The framework is internally coherent, has explicit limits to standard architectures, and has been implemented in ~5400 lines of standalone C++98 across six prototypes.

An eight-phase incremental falsification protocol tested the framework's main claims on two synthetic tasks at small-to-moderate CPU scale. Four bottlenecks were identified and individually addressed; each fix produced a real but bounded local effect; their combination did not amplify into a decisive architectural advantage. The scale-up experiment (Phase-0k) showed that the modest small-scale advantage of EALRMN's attention-based memory readout *vanishes* at moderate scale, with the simpler RNN baseline scaling more cleanly.

A unifying explanation for the negative results — **bootstrap circularity** — is identified and shown to recur across mechanisms (encoder, NCE, gate, readout). The diagnostic generalises beyond EALRMN to any multi-mechanism architecture trained under a single objective.

The contribution is mixed in form: a negative architectural result at the scales tested, accompanied by a positive methodological one. Both deserve to be on the published record, in part to motivate cleaner tests at production scale (where the negative result could be revisited) and in part to provide a template for similar mechanism-stacking architectures to be tested by the protocol developed here.

---

## Appendix A — Reproduction

All code and per-phase result documents are in `research/`:

```
research/
├── EALRMN_DESIGN.md              — full mathematical design (~12 000 words)
├── EALRMN_WRITEUP.md             — this document (final synthesis)
├── EALRMN_PHASE0_RESULTS.md      — Phase-0a (encoder collapse)
├── EALRMN_PHASE0B_RESULTS.md     — Phase-0b (NCE + recon bootstrap)
├── EALRMN_PHASE0C_RESULTS.md     — Phase-0c (recurrence + memory)
├── EALRMN_PHASE0D_RESULTS.md     — Phase-0d (needle per-patch)
├── EALRMN_PHASE0E_RESULTS.md     — Phase-0e (per-token encoder)
├── EALRMN_PHASE0F_RESULTS.md     — Phase-0f (aux gate supervision)
├── EALRMN_PHASE0G_RESULTS.md     — Phase-0g/0j (attention read + write penalty)
├── EALRMN_PHASE0K_RESULTS.md     — Phase-0k (scale-up)
├── ealrmn_phase0_prototype.cpp   — 0a/b/c prototype (~1100 LOC)
├── ealrmn_phase0d_needle.cpp     — 0d prototype (~830 LOC)
├── ealrmn_phase0e_pertoken.cpp   — 0e prototype (~700 LOC)
├── ealrmn_phase0f_aux.cpp        — 0f prototype (~900 LOC)
├── ealrmn_phase0g_attmem.cpp     — 0g/0j prototype (~900 LOC)
└── ealrmn_phase0k_scaleup.cpp    — 0k prototype (~1000 LOC)
```

Build:
```
g++ -std=c++98 -O2 -Wall -Wextra research/<prototype>.cpp -o <executable>
```

Smoke-test commands (representative):
```bash
# Phase-0a/b/c: HMM hidden-state recovery
./research/ealrmn_phase0_prototype     --mode latent_nce --use-memory --seed 42 --steps 1500

# Phase-0d/0e/0f/0g: needle-in-haystack
./research/ealrmn_phase0d_needle       --model ealrmn               --seed 42 --steps 2500 --N 16
./research/ealrmn_phase0e_pertoken     --model ealrmn               --seed 42 --steps 2000 --T 64
./research/ealrmn_phase0f_aux          --model ealrmn --aux-class --aux-gate --seed 42 --steps 2000 --T 64
./research/ealrmn_phase0g_attmem       --model ealrmn_attmem        --seed 42 --steps 2500 --T 64

# Phase-0k: scale-up
./research/ealrmn_phase0k_scaleup      --model ealrmn_attmem        --seed 42 --steps 4000 --T 256
./research/ealrmn_phase0k_scaleup      --model rnn                  --seed 42 --steps 4000 --T 256
```

Wall clocks: 5–600 s per run on single CPU core. Seed 42 throughout.

## Appendix B — Six falsifiable claims, final status

| # | Claim | Status |
|---|-------|--------|
| C1 | Entropy-adaptive segmentation | NOT TESTED — task unsuited |
| C2 | Latent prediction > raw next-token | NOT SUPPORTED (tied or slightly worse on HMM and needle) |
| C3 | Bounded memory matches Transformer up to capacity | PARTIALLY SUPPORTED at small scale; FALSIFIED at moderate scale (Phase-0k) |
| C4 | Sparse experts > dense | NOT TESTED — task unsuited |
| C5 | Selective recurrence > small Transformer | NOT SUPPORTED at any tested scale |
| C6 | Write penalty inverted-U | PARTIALLY CONFIRMED (high-side breaks, low-side doesn't overfit) |

Four of six claims have direct experimental results; none are decisively supported. Two remain untested due to task-structure mismatch.

## Appendix C — Bootstrap dependencies (final)

Each EALRMN-v1 mechanism has a bootstrap-failure mode and a demonstrated auxiliary supervision that breaks it:

```
Encoder
  ↳ z trivially collapses without input-discriminative signal
  ↳ Bootstrap signal: L_recon (decoder of z back to patch tokens)
  ↳ Diagnosed Phase-0a; fixed Phase-0b

InfoNCE (EMA-teacher contrastive)
  ↳ L_NCE = log(B) at random init, never escapes
  ↳ Bootstrap signal: L_recon (NOT the EMA teacher alone)
  ↳ Diagnosed and fixed Phase-0b

Recurrence
  ↳ Held latent MSE grows during training (chasing moving encoder target)
  ↳ Bootstrap signal: reduced MSE weight + identity-reg + slower K-lr
  ↳ Diagnosed and fixed Phase-0c (gives 4× compute saving at iso-accuracy)

Memory gate
  ↳ gate_marker − gate_filler ≈ 0 even with per-token encoder
  ↳ Bootstrap signal: oracle is_marker auxiliary supervision (NOT readout gradient alone)
  ↳ Diagnosed Phase-0d/0e; fixed Phase-0f — BUT FIX DOES NOT UNLOCK ACCURACY

Readout (linear)
  ↳ Cannot extract task info from selectively-written memory
  ↳ Bootstrap fix: attention-based memory read (design memo §4.4)
  ↳ Diagnosed Phase-0f; partially fixed Phase-0g — modest at small scale, DISAPPEARS AT SCALE
```

The pattern is the central empirical finding of this work: each bottleneck fixable individually, none of the fixes producing a decisive cumulative advantage, scale eliminating even the modest small-scale gains.

— end EALRMN final writeup —
