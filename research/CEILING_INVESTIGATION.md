# Scale Ceiling Investigation — 1.84B is the Practical Limit

**Date:** 2026-04-24 (Ralph-loop iter 146)
**Purpose:** Investigate paths past the 1.84B VRAM ceiling on 16 GB GPU.

---

## 1. Current ceiling

At 1.84B (m=2048, L=53, T=1024) with full bf16 stack + MFIO + FACE:
- Weights (bf16):     ~7 GB
- Adam m, v (bf16):   ~7 GB
- scratch_P at T=1024: ~3.5 GB
- Other:              ~0.5 GB
- **Total:** 15.53 / 15.56 GB (0.2% free)

## 2. Paths investigated

### 2.1 CPU-offload Adam (`--cpu-adam`)

**Theoretical savings:** ~7 GB VRAM (Adam state → host memory).
**Theoretical max scale:** could fit 2.5-3B params.

**Empirical test:** 500M × 3 steps with --cpu-adam + MFIO + FACE:
- VRAM: 5.00 GB (vs 5.75 GB with GPU bf16-adam)
- Throughput: **791 tok/s** (vs 4360 tok/s bf16-adam)
- **5.5× SLOWER** per step due to PCIe transfer overhead

**Constraint:** CPU-Adam is incompatible with `--bf16-weights` and
`--bf16-grads`. Can only stack with FACE + MFIO (no bf16 stack).
This compounds the memory problem since weights stay fp32.

**Verdict:** NOT VIABLE for practical research. A 1.5-hour 1.84B
baseline would become 8+ hours. Iteration-cycle time is the limiting
research resource, not VRAM.

### 2.2 Flash attention at full L=53

Could save 3.5 GB (scratch_P) via flash-attn. Tested briefly iter 136.
At L=55+ with flash-attn: still OOMs because other tensors grow
faster than scratch_P savings (at L=55 the fixed-overhead allocations
add up to > 3.5 GB).

**Verdict:** flash-attn doesn't unlock meaningful additional scale
beyond 1.84B on 16 GB at m=2048.

### 2.3 Gradient checkpointing (not implemented)

Theoretical savings: recomputing activations during backward costs
compute but saves activation memory. But CHIRON is already FULLY
reversible — no activations stored. So gradient checkpointing gives
no additional savings.

**Verdict:** CHIRON's reversibility already provides this benefit.

## 3. Conclusion

**1.84B is the practical scale ceiling on 16 GB consumer GPU**
given:
- Iteration-cycle time is the scarce research resource
- CPU-Adam overhead dominates the VRAM savings
- flash-attn doesn't significantly unlock beyond L=53 at m=2048
- CHIRON already eliminates activation memory

To push past 1.84B would require:
- Larger VRAM (24GB+ GPU) — not available on consumer hardware
- Multi-GPU training — not in scope
- Fundamentally different architecture (MoE, compressed weights beyond bf16)

## 4. Research-program boundary recognition

The Ralph-loop program has reached the practical memory ceiling for
the given hardware. Further memory-axis research has diminishing
returns. The three validated disrupting paradigms (FACE + SLC + RLG)
successfully maximize utilization of the hardware:

- **Memory:** 4000× compression → 1.84B on 16 GB
- **Throughput:** ~2× wall-clock via curriculum paradigms
- **Convergence:** FACE's Zipfian preconditioner

Future iterations should focus on:
1. Paradigm #40 on genuinely new axes (if any remain)
2. Longer-horizon validation of current stack
3. Cross-architecture validation (port paradigms to non-CHIRON models)
4. Productionization (stable defaults, recipe templates)

Rather than continuing to push the scale ceiling, which has been
empirically shown to be constrained by hardware VRAM on 16 GB GPUs.
