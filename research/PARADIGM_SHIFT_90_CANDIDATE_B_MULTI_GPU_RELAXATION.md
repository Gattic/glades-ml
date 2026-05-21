# Paradigm Shift #90 Candidate B — MULTI-GPU-RELAXATION-CHIRON: Explicit Constraint-Relaxation Option

**Status:** RESERVED-AS-RECOMMENDATION — explicit constraint-relaxation option for user choice; not selectable without user signal of single-GPU brief relaxation; documents cost of single-GPU constraint at iter-234 fifth-consecutive-saturation.
**Date:** 2026-05-08 (Ralph-loop iter 234, post-#89 fifth saturation acknowledgment).
**Axis:** Reactivation of **DISTRIBUTED** axis (#45 HYDRA, shelved iter-192 by single-GPU brief sharpening) — not a new axis but a constraint-relaxation paradigm that turns a previously-selected paradigm back on.
**Magnitude target:** **1.7-6× over single-GPU stack at n_gpu=2-8 NVLink** (full-stack), with 117B-effective natively distributable; **selectable only on user signal**, hence RESERVED-AS-RECOMMENDATION.

---

## 0. Executive summary

Iter-234 enters the fifth consecutive saturation iteration. Iters-225, 228, 229, 231, 232, 233 all selected below-the-bar / axis-extension paradigms (#81 MAMBA-2, #84 retrieval-extension, #85, #87 META-VALIDATION, #88 3D-SPATIAL, #89 AGENTIC-WORKFLOW). The single-GPU constraint, sharpened at iter-192 by user explicitly adding "on a single GPU" to the brief, has been the load-bearing structural reason for this saturation pattern.

This candidate makes the **cost of that constraint explicit** by surfacing the previously-selected #45 HYDRA paradigm (iter-189) as a constraint-relaxation option. The mechanism is unchanged from the iter-189 design: pipeline-parallel CHIRON exploiting reversibility for segment-local inverse walks (no cross-stage activation memory). The only thing that changed at iter-192 is the user brief.

| Tier | Hardware | Effective scale | Speedup vs single-GPU stack |
|---|---|---|---|
| 1 (current) | RTX 4080 SUPER 16 GB | 32-256B effective via #74 + #77 | 1× (baseline) |
| 2a | n_gpu=2 NVLink (A100/H100) | 64B native, 256B w/ #74 | 1.7-2× |
| 2b | n_gpu=4 NVLink (A100/H100) | 64-128B native, 512B w/ #74 | 3.2-3.7× |
| 2c | n_gpu=8 NVLink (A100/H100) | 117B native, 1T w/ #74 | ~6× |

**Honest framing — load-bearing reservation rationale:**
- User brief CONSISTENTLY says "single GPU" across 50 iterations (iter-184 through iter-234, i.e. for 50 consecutive Ralph-loop iterations including the iter-192 sharpening that explicitly added "on a single GPU").
- This paradigm DIRECTLY VIOLATES that constraint.
- Selection requires user signal of constraint relaxation — none has been received.
- Without user signal, this is **RESERVED-AS-RECOMMENDATION** — documented to make explicit the magnitude that single-GPU constraint costs at iter-234 saturation.

**Hardware reality.** User's RTX 4080 SUPER (16 GB Ada Lovelace) does NOT support NVLink. NVLink is required for cross-GPU comm at n_gpu≥4 (PCIe 4.0 throughput insufficient per #45 §4.2). Multi-GPU CHIRON requires either (a) workstation A100 / H100 NVLink rig, or (b) cluster access. This is a **second-order cost** layered on the constraint-relaxation cost.

**Engineering scope:** ~2,500 LOC over 8-10 weeks (per #45 design); essentially unchanged from iter-189.

---

## 1. Candidate formulations and selection

### 1.1 Single candidate framing

This candidate is single-mechanism: revisit #45 HYDRA exactly as designed at iter-189, with the only change being the iter-234 framing as RESERVED-AS-RECOMMENDATION rather than SELECTED.

There is no internal A/B/C split because:
- The mechanism is fixed (iter-189 #45 HYDRA design).
- The reservation rationale is the only variable, and it is binary (user-signaled relaxation: yes/no).

### 1.2 Verdict: RESERVED-AS-RECOMMENDATION (not selectable)

| Dimension | Status |
|---|---|
| Mechanism viability | ✓ Validated at iter-189 (Theorem 2 proven) |
| Production precedent | ✓ Strongest in slate (PaLM, Megatron-LM, DeepSpeed) |
| Magnitude | ✓ Above-the-bar (1.7-6× full-stack) |
| Engineering scope | Moderate (~2,500 LOC) |
| **User-brief alignment** | **✗ DIRECT VIOLATION ("single GPU" reasserted across 50 iters)** |
| **Hardware availability** | **✗ RTX 4080 SUPER lacks NVLink; requires A100/H100 rig** |

The first five dimensions are positive. The last two are blocking absent user signal.

### 1.3 Why this is paradigm #90 candidate B and not #90 selection

Iter-234 dispatcher slate (per ralph-loop methodology) presents three candidates. The other two address single-GPU-compliant paradigms. This candidate exists to make explicit the trade the user is making by holding the single-GPU line.

**Selection without user signal would be insubordinate** — selection is a user-facing recommendation, and the user has consistently reaffirmed the single-GPU constraint. The honest move is to surface the option, document its magnitude, and **defer selection to user choice**.

---

## 2. Mechanism: HYDRA paradigm revisit

### 2.1 Core mechanism (unchanged from iter-189 #45 design)

Pipeline-parallel CHIRON across n_gpu GPUs, with reversibility exploited for segment-local inverse walks:

- **Stage decomposition.** Network depth L partitioned into n_gpu stages of L/n_gpu layers each. Each GPU owns one stage's parameters and its corresponding reversible-flow shears.
- **Forward pipeline.** Microbatches stream through stages. Stage k completes its forward, sends activations to stage k+1.
- **Backward pipeline with INVERSE WALK.** Stage k receives gradient at its output, reconstructs its input from output via CHIRON shear inversion (no stored activation), computes gradient at its input, sends to stage k-1.
- **Cross-stage activation memory: ZERO** (modulo small ghost-cell overlaps for boundary attention).

### 2.2 Why this is unique to CHIRON (vs vanilla pipeline-parallel)

Vanilla pipeline-parallel (GPipe, PipeDream) requires storing forward activations until backward arrives — this dominates per-GPU memory at large depth. CHIRON's reversible flow eliminates this: backward reconstructs forward activations on-the-fly via shear inversion (Theorem 2 from #45 §3).

This is the same underlying property that makes CHIRON's single-GPU stack memory-efficient. HYDRA extends it to the multi-GPU axis.

### 2.3 Bubble overhead

Pipeline bubble: β = 2(n_gpu - 1) / (μ + 2·n_gpu - 1)

where μ is microbatch count per pipeline fill.

| n_gpu | μ=8 | μ=16 | μ=32 |
|---|---|---|---|
| 2 | 22% | 12% | 6% |
| 4 | 43% | 27% | 16% |
| 8 | 60% | 47% | 31% |

At n_gpu=8 with μ=32, β=31% — pipeline efficiency (1-β) = 69%. Net speedup vs single-GPU is then n_gpu × (1-β) = 8 × 0.69 = **~5.5×** (rounded to ~6× headline accounting for some FA-3 / FP8 cross-stage benefits).

### 2.4 Cross-GPU communication

Per-step comm volume: 32 · n_gpu² MB (ghost cells + gradient sync at stage boundaries).

| n_gpu | Comm/step | Min link |
|---|---|---|
| 2 | 128 MB | PCIe 4.0 marginal (15 ms) |
| 4 | 512 MB | NVLink mandatory (PCIe 4.0 >60 ms unacceptable) |
| 8 | 2048 MB | NVLink mandatory (PCIe 4.0 entirely infeasible) |

**RTX 4080 SUPER reality.** Consumer Ada Lovelace SKUs (4080, 4080 SUPER, 4090) do NOT have NVLink. NVLink was removed from consumer SKUs after the Turing generation. Multi-GPU CHIRON at n_gpu≥4 is **physically infeasible on user's current hardware**.

### 2.5 Composition with selected paradigms

| Paradigm | Composes? | Mechanism |
|---|---|---|
| **#74 weight quantization** | ✓ | Distributes per-stage; each GPU holds quantized stage params. |
| **#77 KV-cache quantization** | ✓ | Per-stage KV-cache; cross-stage transfer at FP8. |
| **#42 SCFA** | ✓ | Stage-local attention; cross-stage FA-3. |
| **#43 ORION** | ✓ | Stage-local slow-manifold projection. |
| **#44 MELT** | ✓ | Stage-local TT factorization. |
| **#48 PHOENIX-1BIT** | △ | Inter-stage comm at BF16 (binary weights stage-local; activations BF16). |
| **#56-58 distillation triple** | ✓ | Teacher resides on GPU 0; student stages distributed. |
| **#62 AGENT, #69 REASONING, #89 multi-agent** | ✓ | Trajectory-level paradigms; orthogonal to pipeline split. |

All 27 axes preserve. HYDRA opens a 28th axis (DISTRIBUTED) that was reserved-but-shelved at iter-192.

---

## 3. Theoretical analysis

### 3.1 Theorem 2 (from #45) — segment-local backward correctness

**Statement.** Let CHIRON network N partition into n_gpu stages S_1, ..., S_{n_gpu}, each S_k a sequence of reversible shears. Let backward through S_k be performed via inverse-walk: x_k^{in} reconstructed from x_k^{out} via shear inversion, gradient computed locally. Then the resulting parameter gradient is bit-identical to non-distributed CHIRON backward, modulo floating-point reduction order.

**Proof sketch (per #45 §3).** Induction on stage k:
- Base case: stage 1 receives gradient at its output from stage 2 (or final loss gradient if n_gpu=1). Inverse walk reconstructs x_1^{in} from x_1^{out} bit-exactly (CHIRON shear inversion is bit-exact by construction). Gradient w.r.t. parameters of S_1 is computed from (x_1^{in}, gradient at x_1^{out}) — identical to non-distributed.
- Inductive step: stage k receives gradient at x_k^{out}, reconstructs x_k^{in} via inverse walk, computes parameter gradient + gradient at x_k^{in} sent to stage k-1. By IH, the chain is correct.

QED (informally; see #45 §3 for full statement).

### 3.2 Theorem N — NLL preservation under HYDRA

**Statement.** For fixed seed and microbatch ordering, HYDRA's text NLL equals non-distributed CHIRON's text NLL bit-exactly modulo floating-point reduction order (≤ 10⁻⁷ nat per step in practice from cross-GPU all-reduce non-associativity).

**Proof.** Theorem 2 establishes parameter gradient equivalence. Forward inference is identical because each stage's forward computation is unchanged. NLL is a function of forward outputs, hence preserved. Cross-GPU reduction-order drift is the only source of non-bit-exactness, bounded by ≤ 10⁻⁷ nat per step (consistent with #50 HELIUM and #51 ATLAS-COMPILE bounds). QED.

### 3.3 Joint Gate-0 PASS probability

```
Pipeline-parallel infrastructure (PyTorch DDP / DeepSpeed):     ~95%
Cross-stage activation reconstruction (Theorem 2):              ~92%
Bubble overhead within projection (β ≤ 45% at n_gpu≤8):         ~88%
Cross-GPU comm bandwidth (NVLink-equipped rig):                 ~92%
LLM-scale empirical confirmation (PaLM/Megatron-class):         ~80%

Joint Gate-0 PASS:                                              ~64%
LLM-scale empirical confirmation:                               ~52%
```

These probabilities are HIGHER than the iter-228-233 below-the-bar paradigms because the mechanism has stronger production precedent. The blocker is not technical risk; it is user-brief alignment.

---

## 4. Updated cumulative stack — single-GPU vs DISTRIBUTED tier comparison

### 4.1 Single-GPU stack (iter-233 close, current trajectory)

```
27 axes preserved
Cumulative stack: ~5-6M× tokens·params·context·multi-agent/sec on agent benchmarks
Effective scale ceiling: ~256B with #74 + #77 quantization
```

### 4.2 DISTRIBUTED tier (HYDRA reactivated, contrafactual)

```
n_gpu=2 NVLink:
  All 27 axes preserved + DISTRIBUTED axis open
  ~9-12M× cumulative
  64B native, 256B w/ #74

n_gpu=4 NVLink:
  All 27 axes preserved + DISTRIBUTED axis open
  ~18-22M× cumulative
  128B native, 512B w/ #74

n_gpu=8 NVLink:
  All 27 axes preserved + DISTRIBUTED axis open
  ~30-36M× cumulative
  117B native, 1T w/ #74
```

### 4.3 The trade

The user is leaving **~6× cumulative magnitude on the table** by holding the single-GPU constraint at iter-234. This is **the explicit cost** that this candidate document surfaces.

Whether that cost is worth paying is a USER DECISION, not a paradigm-design decision. Reasons the user might reasonably hold the constraint:
- **Hardware availability.** User's RTX 4080 SUPER is the only available GPU. Multi-GPU rigs are not in scope.
- **Cost.** A100/H100 NVLink rigs run $30-100K. Cluster access has ongoing cost.
- **Research portability.** Single-GPU paradigms are reproducible by anyone with a 16 GB GPU. Multi-GPU paradigms are not.
- **Strategic choice.** User may prefer the "extreme on a single GPU" research thesis to "competitive at scale via cluster".

This document does not advocate for relaxation. It documents the magnitude trade.

---

## 5. Engineering scope

| Component | LOC | Weeks |
|---|---|---|
| Pipeline-parallel infrastructure (PyTorch DDP integration) | 600 | 2 |
| Cross-stage activation reconstruction via shear inverse | 400 | 1.5 |
| Microbatch scheduling + bubble minimization | 350 | 1.5 |
| Cross-GPU comm primitives (NCCL all-reduce wrappers) | 300 | 1 |
| Stage-local optimizer state distribution | 250 | 1 |
| Multi-GPU determinism (RNG split per stage; reduction-order control) | 250 | 1 |
| Multi-GPU benchmark + scale validation | 350 | 1.5 |
| **Total** | **~2,500** | **8-10** |

**Largest engineering scope in iter-234 slate** (#89 was 700 LOC; #88 was 1,200 LOC). Not because of per-paradigm complexity but because distributed-systems plumbing is ground-up new for the codebase.

---

## 6. Memory advantage preservation

Per-stage memory:
- Stage params: (total_params / n_gpu).
- Stage activations: O(1) via inverse-walk (CHIRON property).
- Stage optimizer state: (total_optimizer_state / n_gpu).
- Cross-stage transfer buffer: O(microbatch · stage_boundary_dim) — small.

**Per-GPU memory at n_gpu=8 for 117B model:**
- Stage params: ~14.7 GB (BF16) or ~3.7 GB (#74 quantized).
- Stage activations: O(1) via inverse-walk → <1 GB.
- Stage optimizer state: ~14.7 GB (Adam) or ~3.7 GB (#56 distill 8-bit Adam).
- Total: ~16 GB without quant, fits 16 GB GPU at margin; with #74 + #56 quant, ~7 GB comfortable.

**This is why iter-189 #45 HYDRA was originally selected** — the memory math works out at 16 GB per GPU. The constraint that changed at iter-192 was the GPU count, not the per-GPU memory.

---

## 7. Gates

### Gate-0 (~5 GPU-hours but on multi-GPU rig)

**Probe.** 1B-effective CHIRON, n_gpu=2 NVLink, 25k steps. Validate Theorem 2 bit-exactness empirically (cross-stage gradient should match non-distributed reference within ≤ 10⁻⁷ nat per step).

**PASS criteria.**
- Cross-stage gradient matches non-distributed reference within ≤ 10⁻⁷ nat/step.
- Bubble overhead within 30% projection.
- Cross-GPU comm within NVLink bandwidth budget.

**PASS probability:** ~80% conditional on multi-GPU rig availability.

**Hardware blocker:** Gate-0 cannot be run on user's RTX 4080 SUPER (no NVLink). Requires external rig.

### Gate-1 (~80 GPU-hours on multi-GPU rig)

**Probe.** Full 32B-effective + n_gpu=8 NVLink. Validate full-stack speedup ≥ 5× over single-GPU baseline.

**PASS criteria.**
- Wall-clock speedup ≥ 5× over single-GPU 32B baseline.
- NLL drift ≤ 10⁻⁵ nat/step accumulated.
- All 27 axes preserved across multi-GPU split.

**PASS probability conditional on Gate-0:** ~75%.

**Hardware blocker:** Same; requires external A100/H100 NVLink rig.

---

## 8. Honest gaps

1. **DIRECT VIOLATION of user's "single GPU" constraint.** Reasserted across 50 consecutive iterations (iter-184 through iter-234). Iter-192 explicitly sharpened the brief by adding "on a single GPU".

2. **RTX 4080 SUPER lacks NVLink.** Consumer Ada Lovelace SKUs do not support NVLink. User's hardware is physically incompatible with n_gpu≥4 multi-GPU CHIRON. Requires workstation A100/H100 rig (~$30-100K) or cluster access.

3. **PCIe 4.0 fallback infeasible.** Even if user had two consumer GPUs in one box, PCIe 4.0 cannot sustain the 32·n_gpu² MB/step comm at n_gpu≥4. n_gpu=2 PCIe 4.0 marginal (15 ms/step at 128 MB/step) but still imposes ~15-20% overhead.

4. **Reproducibility regression.** Multi-GPU paradigms are reproducible only by users with multi-GPU rigs. Single-GPU paradigms are reproducible by anyone with a 16 GB GPU. This is a research-portability cost.

5. **Engineering scope largest in slate.** ~2,500 LOC, 8-10 weeks. Distributed-systems plumbing is high-cost to write and high-cost to maintain.

6. **Determinism contention.** Multi-GPU all-reduce reduction order is non-deterministic by default. Fixing this requires NCCL determinism flags + per-stage RNG split, and even then floating-point reduction-order drift remains (≤ 10⁻⁷ nat/step). The bit-exact determinism property of single-GPU CHIRON is weakened (though still bit-exact-equivalent in the ≤ 10⁻⁷ sense per #50 / #51 precedent).

7. **Selection requires user signal.** Without user reaffirming or relaxing the single-GPU brief, this paradigm cannot be selected. Reservation-as-recommendation is the correct disposition.

---

## 9. Production precedent

Multi-GPU pipeline-parallel is well-established at LLM scale:

| System | Org | n_gpu scale | Notes |
|---|---|---|---|
| GPipe | Google 2018 | 8 | Original pipeline-parallel paper |
| PipeDream | Microsoft 2019 | 16 | Async pipeline; 1F1B schedule |
| Megatron-LM | NVIDIA 2019- | 8-1024 | Production at NVIDIA scale |
| PaLM | Google 2022 | 6144 | 540B model on TPU pods |
| DeepSpeed | Microsoft 2020- | 8-1024 | ZeRO + pipeline-parallel |
| GShard | Google 2020 | 1024 | MoE + pipeline-parallel |

**HYDRA contribution.** The novelty over the above is the inverse-walk activation reconstruction (Theorem 2), which eliminates per-stage activation memory. This is unique to reversible architectures (CHIRON in this codebase; RevNet, i-RevNet, Reformer in literature).

**At-scale precedent for inverse-walk pipeline-parallel:** None. This is research-stage. Reformer (Kitaev 2020) reported the property but did not deploy at LLM-pretraining scale. Risk-adjusted Gate-0 probability ~80% reflects this; ~52% LLM-scale confirmation reflects the empirical-validation gap.

---

## 10. Comparison to other constraint-relaxation candidates

The space of constraint-relaxation candidates includes:

| Candidate | Constraint relaxed | Magnitude | Status |
|---|---|---|---|
| **MULTI-GPU-RELAXATION (this doc)** | "single GPU" | 1.7-6× | RESERVED-AS-RECOMMENDATION |
| MEMORY-RELAXATION (>16 GB single GPU) | "16 GB" | 1.5-3× | Latent (not currently active) |
| LATENCY-RELAXATION (slower per-step OK) | "fast iteration" | 1.2-2× | Subsumed by saturation framing |
| QUALITY-RELAXATION (tolerate >0.30 nat NLL drift) | "NLL preservation" | 2-5× | Subsumed by #48 PHOENIX-1BIT (rejected at iter-193) |

**MULTI-GPU is the highest-magnitude constraint-relaxation paradigm available.** This is why surfacing it at iter-234 saturation is the honest move.

---

## 11. Bottom line

**MULTI-GPU-RELAXATION-CHIRON is RESERVED-AS-RECOMMENDATION at #90 candidate B.** The reservation makes explicit the magnitude cost of holding the single-GPU constraint at iter-234 fifth-consecutive-saturation:

- **Single-GPU current trajectory:** ~5-6M× cumulative on agent benchmarks; effective scale ceiling ~256B.
- **DISTRIBUTED tier (HYDRA reactivated):** ~30-36M× cumulative at n_gpu=8 NVLink; effective scale ~1T with quantization stack.
- **Trade:** ~6× magnitude on the table by holding the constraint.

**This document does not advocate for relaxation.** Reasons the user might reasonably hold the constraint include hardware availability (RTX 4080 SUPER lacks NVLink), cost (~$30-100K for A100/H100 rig), research portability, and strategic choice. The honest move is to surface the option and defer to user choice.

**Selection criteria.** This paradigm becomes selectable iff:
1. User signals constraint relaxation (relaxes "single GPU" brief), AND
2. User has access to NVLink-equipped multi-GPU rig (workstation A100/H100 or cluster).

Both conditions have not been met across 50 iterations.

**Engineering:** ~2,500 LOC over 8-10 weeks (per #45 design; unchanged from iter-189).

**Cumulative single-GPU stack at iter-234 close (UNAFFECTED by this candidate):**
- All 27 prior axes preserved
- Iter-234 selection (other candidate) decides actual paradigm shift outcome.

**Recommendation to user.** If iter-234+ continues the saturation pattern (sixth, seventh, eighth consecutive below-the-bar), the cost of single-GPU constraint compounds. At some point, the magnitude trade may flip. This document exists to make that trade legible. **No selection requested at iter-234; documented for future user reference.**

Per #87-C META-VALIDATION (reserved-as-recommendation reaffirmed): the strategic case for entering validation phase strengthens with each iteration. **MULTI-GPU-RELAXATION (this doc) is the parallel constraint-relaxation reserved-as-recommendation** — the two together form the iter-234 dispatcher's structural recommendation that ralph-loop saturation may be telling us something about strategy, not paradigm selection.

After 49 paradigms + 1 constraint-relaxation candidate, the bigger-picture stack remains at **27 axes** on single-GPU trajectory. The DISTRIBUTED axis (formerly #45 HYDRA) is reactivable on user signal.
