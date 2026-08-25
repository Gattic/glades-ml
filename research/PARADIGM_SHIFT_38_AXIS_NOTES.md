# Paradigm Shift #38 — Axis Analysis

**Date:** 2026-04-23 (Ralph-loop iter 128)
**Context:** After KV-FACE (#36) rejected + HUTCH-DIAG (#37) marginal +
SPAREC (#35) found inapplicable to CHIRON's architecture.
**Constraint:** Any new paradigm must target CHIRON's actual compute
distribution: ~75% attention + embedding, 0% FFN.

---

## 1. CHIRON compute breakdown (empirical)

Per-step compute at 1.84B (m=2048, L=53, T=1024):

| Surface | FLOPs per step | % of step |
|---------|----------------|-----------|
| Attention forward (QK^T, softmax, PV) | ~7.2 TFLOPs | ~35% |
| Attention backward | ~6.8 TFLOPs | ~33% |
| Projection GEMMs (Wq/Wk/Wv/Wo forward+backward) | ~2.5 TFLOPs | ~12% |
| Embedding gather + unembed (logits + softmax CE) | ~2.0 TFLOPs | ~10% |
| LayerNorm forward + backward | ~0.5 TFLOPs | ~2% |
| Optimizer update (Adam + FACE + MFIO) | ~0.3 TFLOPs | ~1% |
| Other (misc kernels, launches) | ~1.0 TFLOPs | ~5% |
| **Total** | **~20 TFLOPs** | **100%** |

**Attention compute is the dominant bottleneck at T=1024.** Any paradigm
that halves attention compute halves step time by ~30%.

## 2. Candidate axes for paradigm #38

### A. Sequence-length curriculum (SLC)

**Mechanism:** T is the dominant cost factor in attention (T² scaling).
Schedule a ramp: T=256 (first 30% of steps) → T=512 (next 30%) → T=1024
(last 40%). Attention FLOPs average drops:
- Baseline T=1024: T² = 1,048,576 per head per query
- SLC average: 0.3·256² + 0.3·512² + 0.4·1024² = 497,772 per head per query
- **~2.1× attention FLOP reduction** averaged over training
- End-to-end speedup: ~1.4× (attention is ~68% of step)

**Precedent:** Shortformer (Press+ 2021), DeepSeek-V2 uses similar schedules.
Not novel as a technique but a NEW stack composition with FACE + MFIO
on CHIRON has not been tested.

**Gate-0 probe:** Run T=256 for 2000 steps at 66M, compare wall-clock
and final loss to T=1024 for 2000 steps. Accept if T=256 reaches
equivalent loss in < 70% wall-clock.

**Implementation:** Moderate — needs --t-schedule flag, forward-pass
T threading through attention kernels. ~200 lines.

### B. Attention-probability top-K sparsity (Q-TOP-K)

**Mechanism:** After computing attention scores S = QK^T/√d, only keep
top-k keys per query (k ≤ 128 for T=1024). softmax over top-k, skip
the remaining T-k during PV computation.

**FLOP savings:**
- Attention forward: O(T·k·d_head) instead of O(T²·d_head).
- For T=1024, k=128: 8× savings on attention forward.

**Precedent:** Reformer (LSH), BigBird, Longformer (all k > 200). Novel
angle: learn per-query k_q adaptively from attention entropy.

**Gate-0 probe:** Measure per-query attention entropy at trained state.
If H_q < α·log(T) for some α < 0.5 on average, top-k is viable.
Prior finding (iter 127): per-row popularity Gini ≈ 0.5 → AVG entropy
~log(T)-0.5 at causal uniform → rows are relatively spread. **Top-K
may not be effective at small scale.**

**Implementation:** Substantial — new attention kernel with top-k
selection. ~500 lines of CUDA.

### C. Attention-softmax Taylor cache (ATC-ATTN)

**Mechanism:** Per paradigm #26 ATC-Δ but applied only to attention.
Cache softmax(QK^T) from step t-1; at step t, recompute only if gradient
on Q, K is large enough to shift entries significantly.

**Memory cost:** Caching full P at T=1024, 1.84B params (L=53, nH=16):
53 × 16 × 1024² floats = 3.6 GB of cache. **Blows memory budget.**

Reduction: cache only ROW-MAX and ROW-SUM per head per layer.
- 53 × 16 × 1024 × 2 = 1.7 M floats = 6.8 MB. Tractable.
- But recomputing P still needs full S = QK^T. No compute savings.

**Verdict:** Memory OR compute savings but not both.

### D. Sparse-gradient embedding (SG-EMB)

**Mechanism:** During embedding backward, only tokens that appeared in
the current batch update their embedding rows. Existing FACE mechanism
already handles this — the row-frequency-weighted EMA effectively zeros
out rare tokens' updates.

**Verdict:** Already implicit in FACE. Not a new axis.

## 3. Selection: SLC (Option A)

**Rationale:**
1. Largest expected speedup on CHIRON's bottleneck axis (~1.4× end-to-end)
2. No expensive custom CUDA (uses existing attention kernels with smaller T)
3. Known mechanism (Shortformer) — de-risked premise
4. Composable with FACE + MFIO + bf16 + 1.84B ceiling
5. Cheap Gate-0 probe (single 2000-step run at 66M)

**De-prioritize:**
- Q-TOP-K: per-row entropy measurement from iter 127 suggests attention
  rows are NOT strongly concentrated. Mechanism premise weak.
- ATC-ATTN: memory/compute tradeoff doesn't clear "magnitudes less
  memory AND faster" bar.

## 4. SLC paradigm #38 design sketch

**Trainer flag:** `--t-schedule T1@step1,T2@step2,T3@step3,...`
   Each entry specifies the T active from step `step_i` until the next
   entry. Fallback: --seq-len (= T_max = last schedule entry's T).

**Scratch allocation:** All buffers sized for T_max. Only first T_current
elements are used per step.

**Forward pass:** Thread T_current through as an alias of T where
appropriate. Attention kernels use T_current in their grid dimensions.

**Backward pass:** Also uses T_current. Gradient tensors sized for T_max
but only first T_current rows are populated.

**Optimizer update:** Unchanged (parameter gradients are still over full
weight tensors).

**Composition:**
- FACE operates on embedding Adam state — unchanged
- MFIO operates on attention weight preconditioner — unchanged  
- bf16 stack — unchanged

**Expected effect at 1.84B × 2500:**
- Baseline: 1578s wall (iter 126)
- SLC schedule `256@0,512@1000,1024@1500`:
  - Steps 0-999 at T=256: attention 16× cheaper
  - Steps 1000-1499 at T=512: attention 4× cheaper
  - Steps 1500-2499 at T=1024: baseline
  - Weighted avg attention cost: 0.4·1/16 + 0.2·1/4 + 0.4·1 = 0.475
  - Attention is ~68% of step → end-to-end: 0.68·0.475 + 0.32 = 0.643
  - Expected wall: 1578 × 0.643 = **1015s (37% faster)**

**Risk:** Convergence may degrade if training dynamics depend on full T
context. Mitigation: long T=1024 tail ensures final optimization happens
at full sequence length.

## 5. Gate-0 protocol

**Test:** Run 66M × 2500 at T=512 vs T=1024, fixed seed, compare:
- Wall-clock to equivalent EMA loss threshold
- Final EMA at step 2500
- Throughput in tok/s

**Accept:** T=512 reaches the same EMA as T=1024 in ≤ 50% wall-clock,
OR T=512's final EMA is within 0.2 nat of T=1024's final EMA at equal
step count.

**Reject:** T=512 fails to converge OR final EMA is >0.5 nat worse,
indicating T-curriculum breaks training dynamics.

## 6. Next iteration action

Iteration 129: Implement the Gate-0 probe for SLC — two 66M × 2500
runs (T=512, T=1024) for direct wall-clock and loss comparison.
Total compute: ~30 minutes. If Gate-0 passes, design + implement
the full --t-schedule flag in iter 130+.
