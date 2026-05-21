## Iter 112 — Engineering-ceiling affirmation META (post-iter-111 FAIL)

**Date**: 2026-05-21
**Iter**: 112 (META, no new code)
**Branch**: vesta5
**Verdict**: **META** — second META in two iters affirms the single-iter engineering ceiling.  iter 111 FAIL added a second mechanism-class boundary marker; the remaining un-attacked targets are either sub-noise (per iter 104) or face the iter 108/111 FAIL boundary.  Production-ready combined stack stays at iter 109 (+8.35% n=3 multi-seed strict PASS, 1.80× cumulative).

---

## Why a second META iter?

iter 110 declared the ceiling. iter 111 attempted the next-largest single-iter target (line-9201 memcpy elimination via in-place reln_backward) and FAILed at +0.117 nat NLL drift. iter 112's attempt would be Plan A (alternation with explicit dq_out_buf_override parameter for scfa_attention_backward), but:

1. **Multi-axis complexity**: requires modifying scfa_attention_backward (~30 LOC across 4 call sites), the bwd loop body (~10 LOC), AND every per-iter memcpy site (3-4 sites). High risk of missing one.

2. **iter 108 + iter 111 FAIL pattern**: both attempted to push past the established mechanism-class boundaries (BF16-dst beta=0 and in-place multi-kernel function). Both ended in math break. iter 112 Plan A is also at the boundary of "modifying a multi-call cuBLAS accumulator function" — high probability of FAIL.

3. **Diminishing returns**: even if Plan A PASSes, predicted +1-1.8% wall (per iter 103/106/109 scaling at 3.2 GB/step). On top of current +8.35%, would yield ~+9.3%. Cumulative 1.81× → 1.82×. Marginal addition vs the risk.

4. **Strategic priority**: the iter 109 stack's +8.35% strict-bar PASS is the strongest production retrain candidate. User authorization for the 30k Phase 2 retrain (per iter 94 ship pattern) would deliver the actual production deployment. Further single-iter scope work is diminishing returns at this point.

## Mechanism-class boundary marker, full picture

After iter 95-111 exploration:

**SAFE patterns** (validated PASS):
| pattern | iter | scale |
|---|---:|---|
| smem-load arith (fold producer → consumer at smem) | 97 | per-element |
| dual-output writes (fold consumer → producer with multi-write) | 99 | per-element |
| dual-output side-write (fold intermediate as side output) | 101 | per-element |
| pure memcpy skip (pure buffer rename, no math change) | 103 | 3.2 GB/step |
| pure memset skip (pre-zero where downstream single-assigns) | 106, 109 | 3.2 GB/step |
| cuBLAS beta=0 + FP32-dst single-write caller pre-zero skip | 107 | 1.15 GB/step |

**UNSAFE patterns** (validated FAIL):
| pattern | iter | failure mode |
|---|---:|---|
| cuBLAS beta=0 + BF16-dst multi-write accumulator | 108 | NLL +0.5 nat drift |
| In-place dout/dx for multi-kernel function (layernorm_backward) | 111 | NLL +0.12 nat, ||g|| explosion |

**SUB-NOISE** (validated NULL):
| pattern | iter | observed |
|---|---:|---|
| Memcpy/memset <200 MB/step | 104 | +0.03% wall (sub-noise floor) |
| Softmax + BF16 cast fuse | 102 | +0.08% sub-noise |
| Tile w=4 fwd | 96 | bit-id NULL |
| Tile w=4 bwd | 95 | bit-id NULL |

## Remaining single-iter targets (all sub-3%)

After iter 95-111 exploration, the un-attacked single-iter candidates and why each is not worth attempting:

| candidate | wall potential | obstacle |
|---|---:|---|
| Bwd softmax + softmax_backward_attn fusion | +0.5-1.5% | Requires custom kernel + cuBLAS reorder + ~150 LOC; complexity high |
| dq_buf → dq memcpy via Plan A alternation | +1-1.8% | Multi-axis refactor (~50 LOC); iter 111 FAIL nearby boundary |
| Adam batching (96 → 1 launch) | +0.05-0.1% | Sub-noise |
| LN bwd dgamma_dbeta partial buffer opt | <0.5% | Has reductions; structural limit |
| s.dp/s.dE pre-zero skip + beta=0 | <0.1% | Sub-noise, below iter 104 threshold |
| Reln_inverse + reln_backward fusion | <0.5% | Custom kernel with cross-reduction state; high complexity |

None individually clears the +3% iter 60 bar at single-iter scope. The iter 108/111 FAIL boundaries constrain the safer-looking ones.

## Path forward

**Single-iter exhausted.** The combined opt-in stack `iter97/99/101/103/106/109` (6 trainer flags) + `iter107` (unconditional library change) = **+8.35% n=3 multi-seed strict-bar PASS** is the production-ready ceiling at this scope.

**Multi-iter scope candidates** (per iter 82 strategic memo, would each take 2-5 iters of focused work):
- **FlashAttention-fused SCFA inner**: predicted +5-10% headroom (per iter 82 wall-ceiling analysis at ratio=32). High EV. Requires custom CUDA kernel that fuses inner Q/K/V/softmax/attention/P/V into one tiled kernel. iter 41 NLL-divergence risk at compression ratios > 16 — would need careful per-precision-tier engineering.
- **CUDA Graph capture of per-layer**: predicted +2-3% from launch-overhead elimination (~14-19 ms / step launch overhead at production scale). Multi-day work to refactor stream/event coordination.
- **FP8 precision tier**: blocked on CUDA 12.0 toolkit (cuBLAS LT FP8 readout requires 12.3+). Hardware (Ada SUPER) supports FP8 tensor cores; software dispatch gap.
- **MoE / conditional computation**: ralph.txt §3 priority 3. Architectural change, scope > single iter.

## Strategic recommendation for user (reiterate)

Production retrain arc per iter 94 Phase 2 ship pattern:

1. **Compute cost**: ~10.5 hours total (5.4h baseline + 5.0h treatment, both 30k steps)
2. **Expected outcome**: ~27,302 tok/s new flagship (+8.76% over iter 94 ship 25,103)
3. **Risk**: very low (math at single-element FP32 bit-identical, sub-ULP cuBLAS-scheduling drift well-characterized at 30k scale per iter 91/94 evidence)

If user authorizes retrain, the path is:
1. Run baseline 30k @ existing iter 94 stack, seed=1337
2. Run treatment 30k with all 6 flags + iter 107 lib, seed=1337
3. Compare final NLL @ step 30k; if within ±0.02 nat, PASS
4. Flip defaults to ON in chiron_main.cpp; update flagship docs

If user defers retrain, iter 113+ should target multi-iter scope work. Single-iter ralph-loop iterations at this point return diminishing-noise increments and risk parity FAILs.

## Files

- This document.
- All iter 95-111 writeups in `research/ITER<N>_*.md`.
- No new code in iter 112.
