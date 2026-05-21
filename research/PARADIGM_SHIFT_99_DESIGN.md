# Paradigm Shift #99 — NEURAL-CACHE-COMPRESSION-DISTILL: Learned KV Compression Beyond #76 MLA

**Status:** SELECTED. Seventh paradigm under iter-236 brief; **eighth iteration in alternation pattern** (4 operational + 4 novel-with-test). Pre-#100 milestone iteration.
**Date:** 2026-05-08 (Ralph-loop iter 243).
**Axis:** STATE-PER-TOKEN (extends #76 MLA). No new axis.
**Magnitude target:** **10× KV cache compression** (vs #76 MLA's 7×). Conditional on Gate-0 PASS.

---

## 0. Executive summary

Iter-243 alternation: novel-with-built-in-test paradigm following operational #98. Mechanism extends shipped #76 MLA's linear KV projection with neural-network-based learned compressor.

**Mechanism:** Replace #76 MLA's linear W_DKV (down-projection) with a small neural network (~1M params; 2-layer MLP) that learns nonlinear KV compression. Larger compression ratio (10× vs 7×) without NLL degradation if neural compressor learns task-conditional bottleneck.

**Bold testable claim:** Neural KV compressor achieves 10× KV cache compression vs MLA's 7× at NLL drift ≤ 0.05 nat (matching MLA's bound).

**Built-in 1-day Gate-0 (~6 GPU-hours):**
- 200M coordinator with #76 MLA replaced by neural compressor (d_c=256 vs MLA's d_c=384).
- 50k-step training at T=8K.
- **PASS criterion:** NLL drift ≤ 0.05 nat AND KV compression ratio ≥ 10× AND inverse-walk reconstruction error ≤ 1e-5.
- **Hard FAIL signals:**
  - Neural compressor training diverges (abort 2h).
  - NLL drift > 0.10 nat at step 25k (abort 4h).
  - KV compression < 8× (abort 5h; below MLA baseline).
  - Inverse-walk reconstruction error > 1e-3 (abort 3h; bijectivity broken).

**Honest framing:**
- Speculative; #76 MLA's linear projection is theoretically simple; neural nonlinearity may overfit.
- ~30-40% mechanism overlap with #76 MLA.
- Magnitude lift: 7× → 10× = ~1.4× incremental on KV-compression axis.
- Iter-200 microopt critique applies (1.4× borderline).
- Built-in 1-day Gate-0 enables fast falsification.

**Engineering:** ~700 LOC over 3 weeks.

---

## 1. Mechanism

### 1.1 Standard #76 MLA

```
c_t^KV = h_t · W_DKV    # linear projection ℝ^d_h → ℝ^{d_c=384}
K_t = c_t^KV · W_UK     # decompress to K
V_t = c_t^KV · W_UV     # decompress to V
```

### 1.2 #99 NEURAL-CACHE-COMPRESSION

```
c_t^KV = MLP_compress(h_t)    # 2-layer MLP; ~1M params
                                # nonlinear: ℝ^d_h → ℝ^{d_c=256}
K_t = MLP_decompress_K(c_t^KV)
V_t = MLP_decompress_V(c_t^KV)
```

Bottleneck d_c=256 < #76's 384 — 1.5× more aggressive compression.

### 1.3 Composition

| Paradigm | Composes? |
|---|---|
| **#76 MLA** | Replaced (#99 supersedes if Gate-0 PASS) |
| **#74 PHOENIX-1BIT** | ✓ Neural compressor stays BF16 (sensitive); trunk PHOENIX-quantized |
| **#78 ATTENTION-SINK** | ✓ Sink + window apply orthogonally |
| **#79 MoD** | ✓ |

---

## 2. Theoretical analysis

### 2.1 Compression-NLL trade-off

**Claim.** Neural compressor at d_c=256 achieves 10× KV cache compression at NLL drift ≤ 0.05 nat IF the compressor learns task-conditional bottleneck.

**Falsification.** If neural compressor overfits to training distribution, NLL drift at validation > 0.10 nat. Hard FAIL signal triggers.

### 2.2 Bijectivity preservation

Neural compressor output → decompressor → K, V is a deterministic function. Bijectivity preserved if the function is invertible (autoencoder structure with reconstruction loss).

**Auxiliary reconstruction loss:** `L_recon = ‖K_recon - K_orig‖² + ‖V_recon - V_orig‖²`. Add at λ_recon=0.05.

### 2.3 Joint Gate-0 PASS probability

```
Neural compressor training stability:                 ~75%
Compression ratio ≥ 10× achieved:                     ~80%
NLL drift ≤ 0.05 nat:                                 ~65%
Inverse-walk bijectivity error ≤ 1e-5:                 ~75%
LLM-scale empirical confirmation:                     ~50%

Joint Gate-0 PASS:                                    ~50-60%
LLM-scale conditional:                                ~50%
Production-viable:                                    ~25-30%
```

---

## 3. Built-in 1-day Gate-0

### 3.1 Probe spec (~6 GPU-hours)

- 200M coordinator + neural compressor (d_c=256).
- 50k-step training at T=8K.
- Pile-CC + LongBook subset (long-context evaluation).

### 3.2 Decision tree

| Outcome | Action |
|---|---|
| **Strong PASS** (10× compression + NLL drift ≤ 0.02) | Build at 1.84B; supersede #76 |
| **PASS** (10× + NLL ≤ 0.05) | Build at 1.84B conditional |
| **Hard FAIL: NLL drift > 0.10** | Close paradigm; #76 MLA remains canonical |
| **Hard FAIL: compression < 8×** | Close paradigm; below baseline |
| **Soft FAIL** (NLL in [0.05, 0.10]) | Hyperparameter sensitivity probe (4 hours additional) |

---

## 4. Updated cumulative stack

```
Iter 242 close (post-#98):
  All 27 axes ≈preserved
  Validation: top-20 paradigms scheduled

Iter 243 (NEURAL-CACHE-COMPRESSION-DISTILL):
  All 27 axes ≈preserved (no new axis)
  Conditional on Gate-0 PASS: 10× KV cache compression supersedes #76 MLA's 7×
```

---

## 5. Engineering scope

| Component | LOC | Weeks |
|---|---|---|
| Neural compressor MLP (1M params) | 250 | 1 |
| Auxiliary reconstruction loss | 100 | 0.5 |
| Bijectivity verification harness | 150 | 0.5 |
| Gate-0 probe runner | 100 | 0.5 |
| Evaluation + #76 comparison | 100 | 0.5 |
| **Total** | **~700** | **3** |

---

## 6. Memory advantage preservation

| Component | Memory |
|---|---|
| Neural compressor (1M params BF16) | +2 MB GPU |
| KV cache (10× compressed; d_c=256) | -3.4 GB at T=10K (vs MLA's d_c=384) |
| **Net** | **-3.4 GB headroom (FREES memory)** |

**Conditional on Gate-0 PASS, memory advantage strengthened by additional 3.4 GB at T=10K.**

---

## 7. Honest gaps

1. **Speculative.** Neural compressor's nonlinearity may overfit; theoretical case for 10× over MLA's 7× is conjectural.
2. **30-40% mechanism overlap with #76 MLA.**
3. **1.4× incremental on KV-compression axis** (7× → 10×); borderline iter-200 microopt.
4. **Production-viable ~25-30%** — moderate uncertainty.
5. **If #76 MLA Gate-0 (per #94) hasn't PASSed, #99 cannot supersede** — ordering dependency.

---

## 8. Bottom line

**NEURAL-CACHE-COMPRESSION-DISTILL is iter-243's novel mechanism.** Bold claim with built-in 1-day Gate-0; falsifiable in 4 hours via hard-FAIL signals.

**Cumulative single-GPU stack at iter-243 close:**
- All 27 prior axes ≈preserved
- Conditional on Gate-0 PASS: KV cache compression 7× → 10× (supersedes #76 MLA)

**Engineering:** ~700 LOC over 3 weeks. **Joint Gate-0 PASS ~50-60%; LLM-scale conditional ~50%; production-viable ~25-30%.**

After 59 paradigms, **27 axes unchanged**.

**Iter-236+ pattern (8 paradigms / 8 iterations):**
- Operational: #92, #94, #96, #98 (top-5/10/15/20 campaigns).
- Novel-with-built-in-test: #93, #95, #97, #99 (ASTRA-KAHAN, MULTI-TEACHER-ROUTING, DRAFT-VERIFIER-CO-LEARN, NEURAL-CACHE-COMPRESSION).

**Pre-#100 milestone:** iter-244's #100 deserves either a major synthesis paradigm or strategic milestone declaration.
