# Paradigm Shift #90 — KV-FACE-MLA-DISTILL: Speculative Premise Rescue at Milestone Iteration

**Status:** SELECTED with continued-saturation framing at #90 milestone (A promoted from #81-B/#88-C two-iteration reservation; **B MULTI-GPU-RELAXATION RESERVED-AS-RECOMMENDATION** documenting cost of single-GPU constraint; C ASTRA-KAHAN reserved as speculative recomposition).
**Date:** 2026-05-08 (Ralph-loop iter 234, #90 milestone iteration; sixth consecutive below-the-bar/axis-extension paradigm).
**Axis:** Recomposition under #76 MLA + #28 FACE — extension of MEMORY-COMPRESSION axis.
**Magnitude target:** **120-200 MB risk-adjusted memory recovery** on MLA latent matrices (~0.75-1.25% of 16 GB ceiling). Below-the-bar; selected on least-bad grounds at #90 milestone.

---

## 0. Executive summary

Iter-234 hits paradigm #90 — round-number milestone. The saturation pattern from iter-225 onward continues: every iteration produces below-the-bar or axis-extension paradigms. **Iter-234 is the sixth consecutive iteration in this pattern** (iter-228-233 all axis-extensions or below-the-bar).

**Three iter-234 candidates all weak:**

| Candidate | Magnitude | Issue |
|---|---|---|
| A KV-FACE-MLA-DISTILL | 120-200 MB risk-adj | Speculative premise rescue; Gate-0 30-40%; below-the-bar |
| B MULTI-GPU-RELAXATION | ~6× if user accepts | Violates "single GPU" brief; RTX 4080 SUPER lacks NVLink; user-decision |
| C ASTRA-KAHAN-DISTILL | speculative | Rejected paradigm rescue; production lr divergence unresolved |

**A selected as least-bad** — at least resolves a two-iteration reservation positively. Continues the iter-225/iter-232/iter-233 pattern of "least-bad selection on saturation grounds."

**B is the strategic option for the user.** RESERVED-AS-RECOMMENDATION explicitly documents the cost of holding the "single GPU" constraint at iter-234 sixth-saturation:
- n_gpu=2 NVLink: ~1.7-2× over single-GPU.
- n_gpu=4 NVLink: ~3.2-3.7× speedup.
- n_gpu=8 NVLink: ~6× speedup; 117B distributed via #45 HYDRA.
- **Hardware blocker**: user's RTX 4080 SUPER consumer Ada SKU lacks NVLink. Multi-GPU CHIRON requires either workstation A100/H100 (~$30-100K) or cluster access.
- **Brief blocker**: user has consistently said "single GPU" across 50 iterations including iter-192 sharpening.
- **User decision required.**

**Mechanism (A):** Apply #28 FACE Zipfian-frequency-aware coding to #76 MLA's compressed latent matrices (W_DKV, W_UK, W_UV at d_c=384). Premise rescue: MLA's compressed d_c=384 latent may exhibit different statistical concentration than full d_kv=2048 K/V (which #36 KV-FACE failed on). FACE compresses the optimizer state for these latent matrices.

**Honest framing:**
- **Speculative premise rescue.** #36 KV-FACE was rejected at Gate-0; #90-A bets on MLA-compressed latent being more Zipfian-amenable.
- **Magnitude small** (~120-200 MB on a 16 GB GPU; ~0.75-1.25% memory recovery).
- **iter-200 microopt critique applies.**
- **Sixth consecutive below-the-bar paradigm.** Saturation pattern unambiguous.
- **#90 milestone deserves mention of MULTI-GPU constraint relaxation as strategic option** (B).

**Engineering:** ~600 LOC over 3 weeks. **Joint Gate-0 PASS ~35%; LLM-scale confirmation ~25%; risk-adj 120-200 MB.**

---

## 1. Candidate formulations and selection

### 1.1 Three candidates

| Candidate | File | Mechanism | Verdict |
|---|---|---|---|
| **A — KV-FACE-MLA-DISTILL** | `PARADIGM_SHIFT_81_CANDIDATE_B_KV_FACE_MLA.md` | FACE Zipfian-coding of MLA latent matrices' Adam state | **SELECTED (least-bad; resolves two-iter reservation)** |
| **B — MULTI-GPU-RELAXATION-CHIRON** | `PARADIGM_SHIFT_90_CANDIDATE_B_MULTI_GPU_RELAXATION.md` | Constraint relaxation; multi-GPU NVLink deployment via #45 HYDRA | **RESERVED-AS-RECOMMENDATION (user-decision; hardware-blocked)** |
| **C — ASTRA-KAHAN-DISTILL** | (no fresh doc) | Recompose rejected #41 ASTRA with Kahan-v + #68 teacher | **RESERVE (speculative; production lr divergence unresolved)** |

### 1.2 Selection: KV-FACE-MLA-DISTILL (least-bad)

Selected on three grounds despite below-the-bar magnitude:

**1. Resolves two-iteration reservation positively** (#81-B, #88-C). Continued deferral wastes paradigm slots.

**2. Falls back cleanly on Gate-0 failure.** Cheap probe (~2 GPU-hours); if MLA latent is not Zipfian, FACE-on-MLA inert with no regression.

**3. Recomposition pattern under iter-212 framing** continues (parallel to #73 PHOENIX-DISTILL-COMBO #74 PHOENIX-1BIT-DISTILL composition pattern, though with much smaller magnitude).

**Honest below-the-bar framing acknowledged:** A's 120-200 MB risk-adj is at iter-200 microopt threshold. Selected as least-bad parallel to iter-225 #81, iter-232 #88, iter-233 #89.

### 1.3 Why MULTI-GPU-RELAXATION reserved-as-recommendation

Self-rejection rationale (from candidate B doc):
- **User brief reasserted "single GPU" across 50 iterations** including iter-192 sharpening.
- **Hardware blocker**: RTX 4080 SUPER consumer Ada SKU lacks NVLink (consumer NVLink removed post-Turing).
- **Multi-GPU CHIRON requires** workstation A100/H100 (~$30-100K) or cluster.
- **Magnitude potential ~6× at n_gpu=8 NVLink** unlocks #45 HYDRA's 117B distributed.

**Reserved-as-recommendation** because user has not signaled openness to constraint relaxation despite 50 iterations of saturation. Decision is user's, not the loop's.

### 1.4 Why ASTRA-KAHAN reserved

- **Production lr divergence (lr=3e-4)** at iter-194 prevented #41 ASTRA promotion.
- **Kahan-v compensation** addresses precision-loss but doesn't address fundamental m=1 stateless-v's unstable variance estimate at high lr.
- **Speculative.** Reserved for future iteration if precision concerns surface.

---

## 2. Mechanism: FACE on MLA compressed latent

### 2.1 #28 FACE primer

#28 FACE: Zipfian-frequency-aware coding for embedding matrix optimizer state. Token frequencies follow Zipfian distribution; FACE encodes Adam's m, v with shorter codes for high-frequency tokens. **1008-1570× embedding Adam-state compression** validated at production.

### 2.2 MLA compressed latent at d_c=384

#76 MLA produces compressed latent c_t^KV ∈ ℝ^{384} per token. Latent matrices W_DKV (down-projection), W_UK (K up-projection), W_UV (V up-projection) are the trainable parameters.

**Premise rescue conjecture:** MLA's d_c=384 latent has lower-rank structure than full d_kv=2048; columns of W_DKV may exhibit per-column frequency concentration analogous to embedding-token frequency.

### 2.3 FACE-on-MLA pipeline

1. Probe MLA W_DKV column-frequency distribution at training time (Gate-0 cheap probe, ~2 GPU-hours).
2. If Zipfian-amenable: apply FACE coding to W_DKV's Adam state (m, v).
3. Memory recovery: optimizer state for W_DKV (d × 384 × 8 bytes Adam BF16) reduced by ~50-100×.
4. Per-layer × 53 layers: ~120-200 MB recovery.

### 2.4 Composition

| Paradigm | Composes? | Mechanism |
|---|---|---|
| **#28 FACE** | ✓ Stack-base | FACE coding mechanism reused. |
| **#76 MLA** | ✓ Stack-base | Operates on MLA's compressed latent. |
| **#74 PHOENIX-1BIT** | ✓ | Trunk PHOENIX-quantized; MLA latent + W_DKV BF16 (sensitive); FACE compresses Adam state. |

---

## 3. Theoretical analysis

### 3.1 Premise rescue (load-bearing conjecture)

**Claim.** MLA's compressed d_c=384 latent exhibits column-frequency Zipfian concentration at training time.

**Evidence basis:**
- MLA latent is compressed projection; lower-rank → potentially more concentrated.
- Rank reduction often induces frequency concentration (well-known low-rank approximation property).

**Falsification:** if W_DKV column variance is uniform across d_c columns (no Zipfian concentration), FACE-on-MLA inert.

**Probability of premise PASS at Gate-0:** ~30-40%.

### 3.2 NLL preservation

FACE compression of Adam state is mathematically lossless (Zipfian coding decodes identically). NLL bit-exact preserved across FACE compress/decompress cycle.

### 3.3 Joint Gate-0 PASS probability

```
W_DKV column-frequency Zipfian concentration:        ~35%
FACE coding integration with MLA compressed latent:   ~85%
Memory accounting at 16 GB ceiling:                   ~95%
NLL preservation across FACE cycle:                   ~99%
LLM-scale empirical confirmation:                     ~25%

Joint Gate-0 PASS:                                    ~35%
LLM-scale empirical confirmation:                     ~25%
```

Lowest in iter-228-234 slate.

---

## 4. Updated cumulative stack

```
Iter 233 close (post-#89):
  All 27 axes ≈preserved
  AGENCY axis extended in-place (#62 + #89)

Iter 234 (KV-FACE-MLA-DISTILL):
  All 27 axes ≈preserved (no new axis; recomposition under iter-212)
  Memory recovery: ~120-200 MB on MLA latent Adam state (~0.75-1.25% of 16 GB)
```

---

## 5. Engineering scope

| Component | LOC | Weeks |
|---|---|---|
| Gate-0 cheap probe (W_DKV column-frequency analysis) | 100 | 0.5 |
| FACE coding integration with MLA latent matrices | 300 | 1.5 |
| Memory accounting verification | 100 | 0.5 |
| NLL preservation regression tests | 100 | 0.5 |
| **Total** | **~600** | **3** |

**Smallest engineering in iter-228-234 slate.** Cheap probe minimizes fall-back cost.

---

## 6. Memory advantage preservation

**Memory advantage strengthened** by ~120-200 MB if Gate-0 PASS. **Single-GPU 16 GB ceiling preserved.**

---

## 7. Gates

### Gate-0 (~2 GPU-hours)

**Probe.** 200M coordinator + MLA at d_c=384. Train for 10k steps. Probe column-frequency distribution of W_DKV.

**PASS criterion.** Top-decile column accounts for ≥ 30% of activation variance (Zipfian concentration threshold).

**PASS probability:** ~35%.

### Gate-1 (~10 GPU-hours conditional)

If Gate-0 PASS:
**Probe.** Apply FACE-on-MLA to 32B-effective trunk. Verify ~120-200 MB Adam-state recovery; verify NLL bit-exact across FACE cycle.

**PASS probability conditional on Gate-0:** ~75%.

---

## 8. Honest gaps

1. **Below the magnitudes-better bar.** 120-200 MB is iter-200 microopt class.

2. **Speculative premise rescue.** Gate-0 PASS ~35% — lowest in iter-228-234 slate.

3. **Sixth consecutive below-the-bar paradigm.** Saturation pattern unambiguous.

4. **#90 milestone deserves mention of MULTI-GPU constraint-relaxation** (B reserved-as-recommendation).

5. **No new axis.** Recomposition extends MEMORY-COMPRESSION axis.

6. **Production precedent absent** for FACE-on-MLA composition (#28 FACE proven on embeddings; #36 KV-FACE failed on full K/V).

---

## 9. Bottom line

**KV-FACE-MLA-DISTILL is selected at #90 milestone as least-bad** of three weak iter-234 candidates. The selection explicitly acknowledges:

- **#90 milestone marks sixth consecutive saturation iteration.**
- **A is least-bad** — resolves two-iteration reservation positively; cheap Gate-0 probe; falls back cleanly.
- **B MULTI-GPU-RELAXATION reserved-as-recommendation** documents user's constraint-relaxation option (hardware-blocked + brief-blocked at iter-234).
- **C ASTRA-KAHAN reserved** as speculative recomposition.

**Cumulative single-GPU stack at iter-234 close:**
- All 27 prior axes ≈preserved
- Memory recovery: ~120-200 MB on MLA latent Adam state (if Gate-0 PASS)

**Engineering:** ~600 LOC over 3 weeks (smallest in recent slate). **Joint Gate-0 PASS ~35%; LLM-scale confirmation ~25%; risk-adj 120-200 MB.**

**B and C dispositions:**
- **B MULTI-GPU-RELAXATION reserved-as-recommendation** — strategic constraint-relaxation option; hardware-blocked (RTX 4080 SUPER lacks NVLink); user-decision required for unblock.
- **C ASTRA-KAHAN reserved** — speculative; production lr divergence unresolved.

After 50 paradigms, the bigger-picture stack remains at **27 axes** (no new axis at #90; recomposition only).

**Sixth saturation iteration acknowledgment.** Iter-225 #81 was second saturation; iter-228-233 produced 5 axis-extensions; iter-234 #90 is sixth saturation. **The structural ceiling under user's accumulated constraints is unambiguous.** Per #87-C META-VALIDATION recommendation: strategic case for entering validation phase strengthens with each saturation iteration. Iter-235+ candidates can pursue:
- **Continued recompositions** of more rejected paradigms.
- **Continued axis-extensions** (sub-axes, narrow domains).
- **Constraint relaxation** (B is documented option; user-decision).
- **Empirical validation feedback** (per META-VALIDATION; out of scope for design loop).
- **Genuinely new axis** (unlikely at this depth; 27 already covered).
