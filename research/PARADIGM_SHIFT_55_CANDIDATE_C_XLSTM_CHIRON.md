# Paradigm Shift #55 Candidate C — XLSTM-CHIRON: Extended LSTM with Matrix Memory × CHIRON Symplectic Shear

**Status:** candidate-C design; one of three parallel proposals for paradigm shift #55. **Recommendation up front: REJECT for #55.** This doc exists so the team has a written close-out of the xLSTM direction and does not re-derive its trade-offs in iter-200+.
**Date:** 2026-05-08 (Ralph-loop iter 198/199, post-#54 NEXUS-SSM/JAMBA selection, under the iter-197+ brief: *"Continue inventing novel LLM architectures, algorithms, and training methods."*).
**Axis:** **architecture-class change inside the symplectic shear** — replace CHIRON's softmax-attention `Y(q)` with an **mLSTM time-mixing block** (Beck et al. 2024, "xLSTM: Extended Long Short-Term Memory"). mLSTM is an extended LSTM whose hidden state is a **matrix** rather than a vector: `C_t ∈ ℝ^{N×N}` accumulates outer products of value/key vectors with exponential gating. Reversibility is structural (still a shear), so CHIRON's bijectivity, O(1)-activation invertibility, and cotangent-lift gradient (#46 REFLECTOR) carry through unchanged.
**Tagline.** *xLSTM is the cheapest per-block recurrence on the menu — ~10× fewer time-mixing FLOPs than Mamba at typical N — but pays for it with N²-per-token state memory, less validation at LLM scale than Mamba, and a paradigm-#55 narrative that doesn't materially differ from #54 NEXUS-SSM. SOPHIA-CHIRON (training-method novelty) dominates on expected speedup × empirical maturity.*

**Materially distinct from competing #55 candidates.** Candidate A (SOPHIA-CHIRON / second-order optimizer) and candidate B (separate doc) both target the **training-method axis** that #54 left untouched. XLSTM-CHIRON re-enters the **architecture-class axis** that #54 already explored. The case for xLSTM at #55: *if we are going to do another architecture shift*, here is what a competing recurrent architecture looks like. The honest case against: **#54 already addressed the recurrent-attention-replacement story**; #55 should select a different axis.

**Materially distinct from #54-A NEXUS-SSM (Mamba selective-scan, state dim N≈16 vector-per-token, state O(N) bytes, FLOPs O(T·m·N)) and #54-B JAMBA-CHIRON (Mamba+SCFA hybrid).** XLSTM-CHIRON uses mLSTM matrix memory: state dim N≈64 **matrix**-per-token, state O(N²) = 4 KB at N=64 (vs Mamba's 32 B), per-block FLOPs O(T·N²) at the time-mixing layer — **~10× cheaper per-block than Mamba** in the inner loop — but `m × N` projections still cost O(T·m·N).

---

## 0. Executive summary

xLSTM (Beck et al. 2024) generalizes LSTM in two directions: **sLSTM** (scalar memory + exponential gating + log-domain stabilizer) and **mLSTM** (matrix-valued state `C_t ∈ ℝ^{N×N}` accumulating outer products). XLSTM-CHIRON adopts **mLSTM** — sLSTM is too close to vanilla LSTM to be worth a paradigm shift. The block sits in CHIRON's symplectic shear `(q, p) ↦ (q, p + Y(q))` with

$$
Y_{\text{xLSTM}}(q)[t] := o_t \odot (C_t \cdot q_t) / \max(|n_t^\top q_t|, 1), \qquad C_t = f_t \cdot C_{t-1} + i_t \cdot v_t \cdot k_t^{\top},
$$

where `q_t, k_t, v_t ∈ ℝ^N` come from `m × N` projections of `q_t^{in} ∈ ℝ^m`, and `f_t, i_t, o_t` are scalar gates per head with log-domain stabilization. **This is "linear attention with multiplicative gating"** — attention without softmax, plus explicit forget/input gates. Reversibility holds because `Y(q)` is continuous in `q` alone (Theorem 3 of #42 SCFA).

**Honest headline.** XLSTM-CHIRON's quality is **mLSTM-class**: Beck et al. 2024 report parity with transformers at 1.4B; **less validation above 7B than Mamba** (Mamba-2 at 7B+, Jamba at 12B/52B). Matrix memory is **128× heavier per token** than Mamba's vector state. The time-mixing inner loop is 10× cheaper than Mamba's, but `m × N` projection costs erase that advantage at flagship: end-to-end per-block FLOPs **~4× more than Mamba** at T=1024.

**Wall-clock at flagship 1.84B-active, T=1024**: XLSTM-CHIRON ~3.2 ms/step (slight regression vs pre-#54 ~3.0 ms); NEXUS-SSM ~2.6 ms (1.15× speedup); JAMBA-CHIRON ~1.7 ms (1.8× speedup). **XLSTM-CHIRON is the slowest of the three.**

**Engineering scope.** ~2200 LOC, 7–10 weeks. Risk: medium engineering/structural, **high empirical** (1.4B is the only public xLSTM run; no independent 7B+ reproduction).

**Recommendation: REJECT for #55.** Select SOPHIA-CHIRON (candidate A) — publicly validated ~2× wall-clock at GPT-2/3 scale, orthogonal to #54, attacks the training-method axis the brief explicitly invites. Reserve XLSTM-CHIRON as niche fallback if (i) a hardware target makes matrix-memory more efficient than vector-SSM, or (ii) xLSTM-7B+ closes the validation gap with Mamba.

---

## 1. xLSTM mathematics

Fix layer ℓ; batch suppressed. CHIRON's symplectic pairing is `(q, p) ∈ ℝ^{T×m} × ℝ^{T×m}`.

### 1.1 Primitive objects (mLSTM, Beck et al. 2024 §2.2)

| Symbol | Type | Definition |
|---|---|---|
| `T, m` | int | sequence length (1024–32768); embedding dim (2048 for 1.84B) |
| `N, n_H, d_h` | int | matrix-state dim (32–128); heads (4–16); per-head dim `N/n_H` |
| `W_q, W_k, W_v` | ℝ^{m × N} | query, key, value projections |
| `W_i, W_f, W_o` | ℝ^{m × 1} | scalar input/forget/output gate projections (per head) |
| `W_out` | ℝ^{N × m} | output projection back to embedding |
| `C_t` | ℝ^{N × N} | per-head matrix state, `C_0 = 0` |
| `n_t, m_t` | ℝ^{N}, ℝ | normalizer state and log-domain stabilizer |

### 1.2 mLSTM time-mixing recurrence

For input `q_t^{in} ∈ ℝ^m`:

```
# Projections
q_t = W_q · q_t^in;   k_t = W_k · q_t^in / √N;   v_t = W_v · q_t^in     # ∈ ℝ^N

# Scalar gates (pre-stabilization)
i_t_pre = W_i · q_t^in + b_i;   f_t_pre = W_f · q_t^in + b_f
o_t = σ(W_o · q_t^in + b_o)                                              # ∈ (0,1)

# Log-domain stabilizer (prevents exponential overflow)
m_t = max(f_t_pre + m_{t-1}, i_t_pre)
i_t = exp(i_t_pre - m_t);  f_t = exp(f_t_pre + m_{t-1} - m_t)            # both ∈ [0, 1]

# Matrix state and normalizer updates
C_t = f_t · C_{t-1} + i_t · v_t · k_t^T          # outer-product accumulation, ℝ^{N × N}
n_t = f_t · n_{t-1} + i_t · k_t                  # ℝ^N

# Normalized matrix-vector readout
h_tilde_t = (C_t · q_t) / max(|n_t^T · q_t|, 1)
y_t = o_t · h_tilde_t
Y(q)[t] = W_out · y_t                            # back to ℝ^m
```

The log-domain stabilizer (`m_t`) is critical: without it, `f_t · C_{t-1}` decays exponentially fast for any negative `f_t_pre`, and `i_t · v_t · k_t^T` grows exponentially fast for any positive `i_t_pre`, leading to BF16 dynamic-range catastrophe within hundreds of steps. With `m_t`, both `i_t` and `f_t · m_{t-1}/m_t` are bounded in [0, 1], and `C_t` evolves stably. Beck et al. 2024 §2.2 prove the stabilizer is mathematically equivalent to the unstabilized form (the `m_t` factors cancel in the readout when using the normalizer `n_t`). **No FP32 escalation needed** (unlike Mamba's `(A_log, Δ)` per #54-A §8.3) — a simplicity win vs Mamba.

### 1.3 Multi-head extension and chunked parallelism

Split `N` into `n_H` heads of dim `d_h := N/n_H`. Each head has its own `(C_t^{(h)}, n_t^{(h)}, m_t^{(h)})` and gates; output is concatenated and projected via `W_out`. Default: N=64, n_H=8, d_h=8.

mLSTM's recurrence is **chunkable but not associative** in Mamba's selective-scan sense. Within a chunk of length `L_chunk = 64`, per-chunk matrix state computes with parallel-friendly inner loops (analogous to FlashAttention-2's tile-and-accumulate but with `N × N` outer-product tiles); inter-chunk state propagates as `(C, n, m)` per head. CUDA kernel ~30% simpler than Mamba's (no input-dependent step size, scalar-per-head gating). XLSTM-CHIRON does **not** modify the FFN slot — MELT-TT on every layer, #53 MOSAIC-MOE alternating.

---

## 2. CHIRON-symplectic integration

### 2.1 Substitution and reversibility

CHIRON's per-block symplectic shear is `Φ_ℓ : (q, p) ↦ (q, p + Y_ℓ(q))`. XLSTM-CHIRON sets `Y_\ell^{\text{xLSTM}}(q) := \text{mLSTM}_\ell(q)` per §1.2. The output `Y_\ell(q) ∈ ℝ^{T × m}` slots into the standard CHIRON shear exactly as in NEXUS-SSM.

**Theorem 1 (xLSTM-CHIRON reversibility).** *For any continuous `Y_\ell^{\text{xLSTM}} : ℝ^{T \times m} \to ℝ^{T \times m}`, the shear `Φ_\ell : (q, p) \mapsto (q, p + Y_\ell^{\text{xLSTM}}(q))` is bijective with explicit inverse `(q', p') \mapsto (q', p' - Y_\ell^{\text{xLSTM}}(q'))`, preserving the symplectic form `ω = dq ∧ dp` and `det DΦ_\ell = 1`. The internal mLSTM recurrence is irrelevant to the reversibility argument.* □

This is Theorem 3 of #42 SCFA — the proof never enters `Y`'s class. **XLSTM-CHIRON is immediately reversible.**

### 2.2 Cotangent-lift gradient (#46 REFLECTOR)

REFLECTOR's exact-gradient lift through the shear (`δp ↦ δp; δq ↦ δq + ∂Y/∂q · δp`) requires `∂Y/∂q` per token. Backward pass implements:
- **Direct path** through `q_t, k_t, v_t`, gates at time t.
- **State-propagation path** through `C_t, n_t, m_t` from time s ≤ t (chunked reverse-mode through the recurrence — analogous to RNN BPTT but with chunk-parallel structure).

CUDA kernel ~30% simpler than Mamba's (no associative-scan reverse algebra); ~50% more complex than RWKV's (matrix state).

### 2.3 BF16 stability and composition

mLSTM's log-domain stabilizer keeps per-step BF16 dynamic range bounded without FP32 escalation. Cumulative inverse-walk error at L=53 is ~`L · ε_BF16 ≈ 1e-3`, same as pre-#54. **No new BF16 hazard.** #53 MOSAIC-MOE composes orthogonally (xLSTM in attention slot, MOE in FFN slot — no conflict). #46 REFLECTOR, #44 MELT, #28 FACE on embeddings, #38 SLC, #39 RLG, #51 ATLAS-COMPILE, #52 NIMBUS all carry through. **#42 SCFA and #54 NEXUS-SSM are mutually exclusive** (all replace the attention slot). **#50 HELIUM FA-3 is inapplicable** (no softmax).

---

## 3. Compute and memory analysis

### 3.1 Per-block FLOPs at T=1024, m=2048, N=64

| Component | FLOPs / token |
|---|---|
| `q_t, k_t, v_t = (W_q, W_k, W_v) · q_t^in` | 3 × 2 m N = 786 K |
| Gate projections (i, f, o) | 3 × 2 m = 12 K |
| `C_t` outer-product update | 2 N² = 8 K |
| `C_t · q_t` readout | 2 N² = 8 K |
| `W_out · y_t` | 2 N m = 262 K |
| **Per-token total** | **~1.07 MFLOPs** |
| **Per-block at T=1024** | **~1.10 GFLOPs** |

mLSTM **time-mixing only** (without `m × N` projections): T·N² = 4.2 MFLOPs/block. **This is the "10× cheaper than Mamba" headline claim.** Compare full per-block costs:

| Architecture | T=1024 | T=8192 | T=32768 | Per-token state |
|---|---|---|---|---|
| Pure attention | 8.6 GF | 549 GF | 8,786 GF | 4 KB (KV cache) |
| Mamba (NEXUS-SSM) | 0.27 GF | 2.16 GF | 8.64 GF | **32 B** |
| **mLSTM (XLSTM-CHIRON)** | **1.10 GF** | **8.78 GF** | **35.1 GF** | **4 KB (N² at N=64)** |
| RWKV | 17 GF | 136 GF | 549 GF | 2 KB |
| SCFA | 1.98 GF | 7.4 GF | 49.5 GF | 0.5 MB total scratch |

**Honest reading**: mLSTM is **4× more expensive per block than Mamba** at flagship `T=1024, m=2048` once `m × N` projections are included, because Mamba's `B, C` projections are similar dim but Mamba's parallel scan needs no matrix readout. The "10× cheaper" applies only to the time-mixing inner loop, not end-to-end.

### 3.2 Per-step wall-clock (Ada-realistic, post-#50 + #52)

| T | XLSTM-CHIRON | NEXUS-SSM | JAMBA-CHIRON | Pure-attention CHIRON |
|---|---|---|---|---|
| 1024 | **~3.2 ms** | ~2.6 ms | ~1.7 ms | ~3.0 ms |
| 8192 | **~22 ms** | ~15 ms | ~7 ms | ~37 ms |
| 32768 | **~75 ms** | ~65 ms | ~32 ms | infeasible |

**XLSTM-CHIRON is slower than NEXUS-SSM at every T** because Mamba's compact `N=16` vector state is more bandwidth-efficient than mLSTM's `N=64` matrix state on Ada (672 GB/s). Slower than JAMBA-CHIRON because JAMBA reuses post-#50 FA-3 kernels on surviving SCFA layers.

### 3.3 Memory: matrix-state burden

Per-token state in bytes: pure attention 4 KB (KV); Mamba 32 B; **mLSTM 4 KB (matches pure attention)**; RWKV 2 KB. At T=32768 with L=53, mLSTM matrix-state across the sequence consumes ~6.8 GB activation memory (batch=1) — **a significant disadvantage vs NEXUS-SSM's ~64 MB** for long-context training. Per-token memory ratio mLSTM:Mamba = 128× at typical (N=64, N_Mamba=16).

---

## 4. Material distinction from #54-A NEXUS-SSM and #54-B JAMBA-CHIRON

| Aspect | NEXUS-SSM | JAMBA-CHIRON | **XLSTM-CHIRON** |
|---|---|---|---|
| Mechanism | Selective state-space (Mamba) | Mamba + SCFA hybrid | **Matrix memory (mLSTM)** |
| Inner-loop FLOPs at T=1024 | ~33 MFLOPs | mixed | **~4.2 MFLOPs (10× cheaper)** |
| Total per-block at T=1024 (with projections) | 0.27 GF | 0.27–1.98 GF | **1.10 GF (4× more than Mamba)** |
| Per-token state | 32 B | mixed | **4 KB (128× more than Mamba)** |
| Quality at LLM scale | Mamba-class (parity 7B+) | Jamba-class (parity 12B+) | **mLSTM-class (1.4B; less above)** |
| Empirical maturity | More validated | Most validated | **Least validated** |
| Engineering scope | ~3000 LOC, 10–14 weeks | ~2470 LOC, 7–10 weeks | **~2200 LOC, 7–10 weeks** |
| KV-cache at flagship | Eliminated | Halved | **Eliminated** |
| Composes with #50 FA-3 | N/A | Full on SCFA | **N/A (no attention)** |
| Composes with #42 SCFA | Mutually exclusive | Built in | **Mutually exclusive** |
| Long-context unblock T≥32k | Excellent (uniform O(T·m·N)) | Mixed | **Good but worse than Mamba** (matrix state burns bandwidth) |
| Empirical risk | Medium | Lower | **Higher** |
| Per-paradigm narrative | Replace attention with linear SSM | Hybrid Mamba+attention | **Replace attention with matrix-memory linear attention** — close to NEXUS-SSM's narrative |

### 4.1 The decisive distinction

XLSTM-CHIRON is **dominated** on the iter-197+ brief's primary axes:
- **Wall-clock speedup**: NEXUS-SSM 1.15× at T=1024, JAMBA-CHIRON 1.8× at T=1024. **XLSTM-CHIRON 0.94×** (slight regression).
- **Empirical maturity**: Mamba > Jamba > xLSTM. xLSTM is the least validated.
- **Long-context**: NEXUS-SSM > XLSTM-CHIRON > JAMBA-CHIRON; XLSTM-CHIRON middle, with matrix-state memory cost limiting reach.
- **Differentiation from #54**: NEXUS-SSM and XLSTM-CHIRON are both "replace attention with a linear-recurrence shear". Selecting xLSTM at #55 after Mamba at #54 is **incremental, not paradigmatic**.

**The brief asks for novel paradigm shifts**; replacing one recurrence with another whose only differentiator is "10× cheaper inner loop but less validated and slower end-to-end" is a tuning, not a paradigm shift.

### 4.2 Where XLSTM-CHIRON would lead

If the brief shifted to "smallest per-block compute inner loop", "in-context associative recall", or "hardware target where matrix outer-products are tensor-core-efficient", XLSTM-CHIRON would close some gap. **None of these shifts are present in iter-198/199.**

---

## 5. Composition with shipped paradigms

| Pre-#55 paradigm | Composition | Multiplier |
|---|---|---|
| #1 CHIRON reversibility | Built in (Theorem 1) | Full |
| #28 FACE | Embedding full; xLSTM weights empirical | ≈ 0.7× |
| #38 SLC, #39 RLG | Independent | Full |
| #42 SCFA | **Mutually exclusive** | 0× |
| #44 MELT | FFN backbone unchanged | Full |
| #46 REFLECTOR | New cotangent-lift through xLSTM | Full |
| #47 PHOENIX-1.58BIT | `W_q, W_k, W_v, W_out` full; gate biases BF16 | ≈ 0.85× |
| #50 HELIUM FA-3 | Inapplicable (no softmax) | 0.5× FP8 only |
| #51 ATLAS-COMPILE | Static layer pattern | Full |
| #52 NIMBUS | Optimizer parameter-agnostic | Full |
| #53 MOSAIC-MOE | Orthogonal FFN slot | Full |
| #54 NEXUS-SSM | **Mutually exclusive** | 0× |

XLSTM-CHIRON would deliver ~`917× × 1.0 ≈ 917×` cumulative at T=1024 (zero new wall-clock from #55, because xLSTM is slower than #54's Mamba). The novelty is non-overlapping with NEXUS-SSM only because the mechanism is different; the speedup is regressive.

**The 0× multiplier on #42 and #54** means XLSTM-CHIRON cannot compose with either: all three replace the attention slot. Selecting XLSTM-CHIRON at #55 means **un-selecting #54**, which the iter-197+ pipeline does not contemplate (post-#54 selection is final). A realistic "XLSTM-CHIRON at #55" would have to be a hybrid (xLSTM blocks alternating with Mamba blocks) — adding engineering complexity without buying anything beyond what JAMBA's Mamba+attention already explored.

---

## 6. The empirical-validation gap (decisive)

### 6.1 What Beck et al. 2024 measured

xLSTM is published at **1.4B parameters** on SlimPajama-627B for ~300B tokens — comparable NLL to Llama-2-1.4B (HellaSwag, ARC, Winogrande). Better than Mamba at 350M on some long-range arena tasks; comparable at 1.4B. **No published 7B+ run.** **No independent reproduction at 1.4B as of early 2026.**

### 6.2 What Mamba/Jamba have

- Mamba-1 at 1.4B, 2.8B (Gu & Dao 2023); Mamba-2 at 7B (Dao & Gu 2024) with SSD duality.
- Falcon-Mamba 7B (independent, 2024) — open-weight.
- Jamba 12B/52B (Lieber et al. 2024) — Mamba+attention hybrid, parity with Mixtral-8×22B.

### 6.3 What this means at 1.84B-flagship

1.84B-active sits **just above** xLSTM's published 1.4B. Transferring Beck et al. 2024's parity claim to 1.84B-CHIRON-reversible-MOE-augmented has gaps: no CHIRON-reversibility validation; no MOE-FFN composition validation; no 7B+ scaling (Mamba-1.4B → Mamba-7B shows diminishing relative gains, xLSTM may follow); no independent reproduction. **NEXUS-SSM** inherits Mamba's 1.4B/7B/Jamba-12B base — strong. **JAMBA-CHIRON** inherits Jamba's 12B/52B — strongest. **XLSTM-CHIRON** inherits xLSTM's 1.4B only — weakest.

The empirical-validation gap is **decisive** when comparing #55 candidates against SOPHIA-CHIRON, whose ~2× wall-clock-to-fixed-NLL is publicly validated at GPT-2/3 scale (Liu et al. 2023) and reproduced independently.

---

## 7. Concrete primitives and Gate-0 protocol (if pursued)

New: `gpu_xlstm_block.{h,cu}` (~1350 LOC: chunked recurrence, outer-product accumulation, log-stabilizer); `xlstm_state.h` (~100 LOC); `xlstm_chiron_test.cpp` + `xlstm_chiron_smoke_test.cpp` (~450 LOC); `training_config.h` XLSTMConfig (~40 LOC); `run.sh --xlstm` flag (~30 LOC). Modifications: `sgd_transformer.cpp` + `transformer_infer.cpp` + `transformer_generate.cpp` (~260 LOC delta). Total: **~2200 LOC** new, **~290 LOC** modifications. Public API unchanged.

**Gate-0 (~6 GPU-hours on 66M):**
- **Probe A (3h)**: NLL parity at 66M after replacing all attention with mLSTM. **Pass**: within 0.10 nat at 5K steps.
- **Probe B (1h)**: Stabilizer dynamic range — `max_t,ℓ |m_t^{(ℓ)}|` over 2K steps. **Pass**: stays below 80.
- **Probe C (1h)**: Cumulative reversibility at L=53. **Pass**: ≤ 1e-3 in BF16.
- **Probe D (1h)**: Wall-clock vs NEXUS-SSM at flagship. **Expected outcome: FAIL** (XLSTM-CHIRON ~1.2–1.5× slower than NEXUS-SSM because of `m × N` projection cost) → reject.

---

## 8. Honest gaps

1. **Headline "10× cheaper per-block than Mamba" is technically true but practically misleading.** It applies only to the time-mixing inner loop. Once `m × N` projections are included (which dominate at flagship), XLSTM-CHIRON is **4× more expensive per-block than Mamba**.

2. **mLSTM's 1.4B published evidence does not transfer cleanly to 1.84B-CHIRON-reversible-MOE-augmented.** No reversibility validation, no MOE composition validation, no 7B+ scaling, no independent reproduction. Gate-0 Probe A measures the first; the others remain priors.

3. **Per-token matrix state is 128× larger than Mamba's vector state.** At T=32768 this consumes ~6.8 GB of activation memory (53 layers, batch=1) — a significant disadvantage vs NEXUS-SSM for long-context training.

4. **XLSTM-CHIRON duplicates NEXUS-SSM's narrative.** Both replace softmax attention with a linear recurrence. The differentiator (matrix vs vector state) is too narrow to constitute a paradigm shift on its own. **#55 should select a different axis from #54.**

5. **SOPHIA-CHIRON dominates on expected speedup × empirical maturity.** SOPHIA's ~2× wall-clock-to-fixed-NLL is publicly validated; XLSTM-CHIRON's per-block FLOP win is throttled by projection costs and unmeasured at flagship. The honest expected-value comparison favors SOPHIA on every axis the iter-197+ brief weights ("training methods" as first-class research output, validated speedup at LLM scale, orthogonal composition with #54).

6. **Independent reproduction of xLSTM at 1.4B is missing as of early 2026.** Mamba has multiple independent reproductions; xLSTM has only Beck et al.'s codebase. **Evidence base is thinner than NEXUS-SSM's and JAMBA-CHIRON's.**

7. **The "matrix state holds outer-product structure" expressivity claim is theoretical.** Beck et al. 2024 do not isolate this advantage at LLM scale. Associative-recall benchmarks (where matrix state should help) are not LLM-flagship benchmarks.

8. **#55-XLSTM selection un-selects #54-NEXUS-SSM.** Both replace the attention slot; cannot compose. The paradigm-shift sequence does not contemplate undoing a shipped shift.

9. **BF16 stabilizer tripwire (`m_t` saturation) is theoretically protected but unmeasured at deep stacks.** Beck et al. 2024 measure at L=12; CHIRON-1.84B is L=53. Compounding stabilizer drift across 53 layers is a new validation surface (Probe B addresses).

10. **The user has previously preferred bit-exact paradigms.** XLSTM-CHIRON is bit-exact at the symplectic-shear level (Theorem 1) but diverges from pre-#54 NLL (≤ 0.10 nat behind by Beck et al. 2024 transfer). Same posture as #54-A and #54-B — passes the iter-197+ brief, would fail the pre-iter-197 bit-exact preference.

---

## 9. Summary

XLSTM-CHIRON replaces softmax attention with mLSTM matrix-memory blocks (Beck et al. 2024). Reversibility holds structurally per Theorem 1 — the mLSTM recurrence is continuous in `q`, so the symplectic shear is bijective and `det DΦ = 1` regardless of internal mechanism. The log-domain stabilizer (`m_t`) keeps BF16 dynamic range bounded **without FP32 escalation** — a simplicity win vs Mamba's `(A_log, Δ)` FP32 requirement.

**Honest headline.** XLSTM-CHIRON delivers the **cheapest per-block recurrence inner-loop** on the menu (~10× fewer FLOPs than Mamba at the time-mixing layer) but **pays for it** with `m × N` projection costs that erase the advantage end-to-end (~4× more expensive than Mamba per block at flagship), with N²-per-token state memory (128× more than Mamba), and with the **weakest empirical-validation base** of the three #55 candidates (xLSTM at 1.4B only; Mamba at 7B+; Jamba at 12B/52B). Per-step wall-clock at T=1024 is ~3.2 ms (vs NEXUS-SSM ~2.6 ms and JAMBA-CHIRON ~1.7 ms) — a **regression** vs both #54 candidates.

**Decisive case against:**
1. **Architecture-class duplication of #54.** Both NEXUS-SSM and XLSTM-CHIRON replace softmax attention with a linear recurrence. Differentiator (matrix vs vector state) too narrow for a paradigm shift; #55 should select a different axis.
2. **Domination by SOPHIA-CHIRON on expected speedup × empirical maturity.** SOPHIA's ~2× wall-clock-to-fixed-NLL is publicly validated at GPT-2/3 scale (Liu et al. 2023) and reproduced independently. XLSTM-CHIRON's per-block win is throttled by projection costs and unmeasured at CHIRON-flagship.

**Recommendation: REJECT for paradigm shift #55.** Reserve as niche-deployment if (i) a hardware target makes matrix-memory more efficient than vector-SSM, (ii) xLSTM-7B+ closes the validation gap with Mamba, or (iii) the brief shifts to "smallest research-engineering footprint with empirical risk on per-token state". None are present in iter-198/199.

For #55, select **SOPHIA-CHIRON (candidate A)** — second-order optimizer with publicly validated ~2× wall-clock speedup, composes orthogonally with #54's architectural shift, and addresses the training-method axis the iter-197+ brief explicitly invites.
