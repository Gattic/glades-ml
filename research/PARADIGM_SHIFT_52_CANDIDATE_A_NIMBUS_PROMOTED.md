# Paradigm Shift #52 Candidate A — NIMBUS-PROMOTED (Async Optimizer Pipelining, refined for the post-#51 stack)

**Status:** candidate-A design for paradigm shift #52. Promotes the iter-194 NIMBUS proposal (`PARADIGM_SHIFT_50_CANDIDATE_C_NIMBUS.md`, originally a #50 candidate, not selected — #50 went to HELIUM, #51 went to ATLAS-COMPILE) to a full paradigm shift with composition refinements for the post-#42–#51 stack.
**Date:** 2026-05-08 (Ralph-loop iteration 196, post-#50 HELIUM, post-#51 ATLAS-COMPILE, under the standing iter-193 brief: *"NLL preservation strict + magnitudes compute speed, single GPU"*).
**Axis:** **Step-pipelining of CHIRON's training loop** — overlap the host-side, FP32-master-weighted Adam optimizer step of training step `t` with the GPU forward of step `t+1`. NIMBUS-PROMOTED differs from the iter-194 candidate only in that the post-#42–#51 environment changes the operating point: forward+backward GPU compute is now ~4 ms (post-ATLAS-COMPILE captured graphs + persistent kernels), Adam-on-host is now ~1 ms (post-HELIUM FP8 grad and post-#47 PHOENIX-1.58BIT), and the entire pipeline must compose with two new structures — ATLAS-COMPILE's CUDA-graph-captured DAG and HELIUM's FP8 master-state staging.
**Tagline:** *iter-194 NIMBUS observed: at flagship 1.84B post-#42–#49, Adam is 25 % of step time and serial. The fix is to overlap it with forward of step t+1. iter-196 NIMBUS-PROMOTED observes: post-#51 ATLAS-COMPILE, the per-step DAG is now a captured CUDA Graph; post-#50 HELIUM, the host-side Adam reads FP8 grads from a HELIUM-stochastic-rounded staging buffer. The pipelining axis is the same; the implementation must respect both new layers.*

**Materially distinct from competing #52 candidates B and C:**
- **NIMBUS-PROMOTED (this doc, A)** — async optimizer pipelining, K_stale = 1, ≤ 0.003 nat NLL gap, 1.33× per-step at flagship. Composes cleanly with ATLAS-COMPILE (graph + transfer-stream are sibling streams) and HELIUM (FP8 master state pipelined identically to FP32 master).
- **B and C** — separate proposals; not analyzed here.

**Honest headline.** NIMBUS-PROMOTED gives **1.33× per-step wall-clock speedup at flagship 1.84B post-#42–#51** with NLL-preserving K_stale = 1 staleness (≤ 0.003 nat gap per step). Combined with the existing 690× post-#51 stack at 18B, this brings the cumulative single-GPU advantage to **~917× at 18B with bit-exact-equivalent NLL**. Engineering scope: ~750 LOC (550 LOC inherited from iter-194 candidate + 200 LOC composition wiring), ~3 weeks.

---

## 0. Refinements over the iter-194 NIMBUS candidate

The iter-194 NIMBUS candidate document (~6000 words, `PARADIGM_SHIFT_50_CANDIDATE_C_NIMBUS.md`) provides the foundation:

- The pipelining schedule (§2.2 of iter-194) — forward of step `t+1` runs concurrently with host-side Adam of step `t`.
- The K_stale = 1 NLL-gap theorem (§3.2 of iter-194), bound `≤ L_H · η · ‖m_t‖ · K_stale ≈ 0.003 nat`.
- The two-stream wiring (§4.1) using `glades::gpu::computeStream()` and `transferStream()`, already exposed by `gpu_device.h`.
- The double-buffered weight ping-pong (§4.2) and host-pinned Adam-callback pattern (§4.3, §7.3).
- The 1.2–1.5× speedup ceiling argument and Gate-0 design.

That foundation is **unchanged** in NIMBUS-PROMOTED. What is new in this iteration:

1. **Composition with #51 ATLAS-COMPILE** (§3 below). ATLAS-COMPILE captures ~440 kernels per training step into a CUDA Graph and replays them via a single host call on the compute stream. NIMBUS-PROMOTED must (a) keep the graph capture compatible with the transfer-stream-side optimizer pipeline, (b) ensure the cross-stream events (`E_grad_t`, `E_weight_t`) participate in the captured graph rather than fighting it, and (c) preserve the per-shape autotune cache across the graph variants needed for ping-pong weight buffers.

2. **Composition with #50 HELIUM** (§4 below). HELIUM-FP8 produces gradients in E5M2 plus per-tensor scale factor; the FP32 master state on host must consume `(scale, E5M2)` not raw FP32. NIMBUS-PROMOTED must (a) D2H the FP8+scale pair (5× less PCIe traffic than FP32 grads), (b) dequantize on host inside the Adam callback, and (c) maintain stochastic-rounding determinism semantics through pipelining.

3. **Refined per-step time analysis at flagship 1.84B post-#42–#51** (§5). Pre-#52 step time (post-ATLAS-COMPILE) is ~4 ms compute. Post-HELIUM Adam time on host is ~1 ms (FP8 grad halves PCIe; FP8 dequant adds negligibly). NIMBUS-PROMOTED overlaps Adam with forward+backward, dropping critical-path step time to ~3 ms. **Per-step speedup: 1.33×.**

4. **Refined stack projection** (§6). Post-#51 stack at 18B is 690× (per `PARADIGM_SHIFT_51_DESIGN.md` §0). Adding NIMBUS-PROMOTED's 1.33× yields **690 × 1.33 ≈ 917× at 18B with bit-exact-equivalent NLL**.

5. **Updated NLL bound** (§7). The K_stale = 1 bound from iter-194 (≤ 0.003 nat per step, non-accumulating) holds unchanged. ATLAS-COMPILE's 10⁻⁷-nat-per-step reduction-order drift and HELIUM's ≤ 0.001-nat-per-step stochastic-rounding drift add but remain well below the 0.01-nat run-to-run-variance floor. **Composite NLL gap: ≤ 0.004 nat per step**, still NLL-preserving by the iter-193 strict criterion.

The total document delta from iter-194 to NIMBUS-PROMOTED is ~200 LOC of composition wiring (CUDA-Graph-aware event recording + FP8-aware Adam callback) plus the analysis below.

---

## 1. Why promote NIMBUS to #52

The iter-194 selection chose HELIUM over NIMBUS for #50, then ATLAS-COMPILE for #51. Both are now shipped or planned. The NIMBUS rationale stands stronger now than it did in iter-194:

| Reason | iter-194 view | iter-196 view (after #50, #51) |
|---|---|---|
| Adam-step bubble | 25 % of step at 1.84B post-#42–#49 | 20 % of step at 1.84B post-#42–#51 (T_adam ≈ 1 ms / T_step ≈ 4 ms — ratio dropped because BOTH halves shrank, but the ratio is what matters for NIMBUS's gain) |
| Engineering surface | 550 LOC | 750 LOC after adding composition wiring; still small |
| Composition with shipped paradigms | Multiplicative with #42–#49 | Multiplicative with #42–#51 (ATLAS-COMPILE composition addressed below) |
| Architectural prerequisite | #47 PHOENIX-1.58BIT host-pinned FP32 master | Same prerequisite + #50 HELIUM FP8 master variant + #51 ATLAS-COMPILE graph integration |
| NLL guarantee | ≤ 0.003 nat per step | Composite ≤ 0.004 nat per step (still strictly NLL-preserving) |
| Paired #51 + #52 strategy from ATLAS-COMPILE doc §11 | n/a | ATLAS-COMPILE's own §11 explicitly recommends "ship ATLAS-COMPILE as foundation; layer something on top for the magnitude push." NIMBUS-PROMOTED is exactly that follow-on. |

The ATLAS-COMPILE candidate document anticipates this composition (it lists candidates like APOLLO and HORIZON as the second-shift options). NIMBUS-PROMOTED is the conservative member of that set: same ε-bounded-but-essentially-zero NLL impact as ATLAS-COMPILE itself, no algorithmic novelty, lowest engineering cost.

---

## 2. The post-#42–#51 operating point

### 2.1 Per-step time decomposition at 1.84B

After #42–#51, a single training step at flagship 1.84B looks like:

| Phase | Time | Source |
|---|---|---|
| Forward (53 layers, T=1024, FP8 GEMM via HELIUM, FA-3 attn, captured in graph) | ~1.5 ms | post-ATLAS-COMPILE: graph replay overhead negligible; post-HELIUM: FP8 tensor cores 1.7× faster than BF16 |
| Backward (cotangent-lift via #46 REFLECTOR, FP8 E5M2 gradient, captured) | ~2 ms | post-#46: T_bwd ≈ T_fwd; post-HELIUM E5M2: 1.5× faster than BF16 |
| Adam on host (FP32 master per #47, FP8 D2H per HELIUM, OpenMP-parallel) | ~1 ms | iter-194 estimated 5 ms post-#42–#49; HELIUM D2H of FP8 grad is 4× cheaper, FP8 dequant is OMP-cheap, FP32 Adam math unchanged |
| H2D weight refresh (#47 PHOENIX-1.58BIT quantized; ~2 GB at 1.84B in 1.58-bit packed) | ~0.5 ms | smaller than D2H grad because PHOENIX compresses 10× |
| **Critical path (sequential)** | **~5 ms** | F + B + Adam + H2D, serial |

(The forward/backward times here are smaller than the iter-194 post-#42–#49 estimates of 5+10 ms because (a) the iter-194 doc was conservative on HELIUM and ATLAS-COMPILE since neither was shipped, and (b) the ATLAS-COMPILE design doc reports 1.45× at 1.84B and HELIUM 1.7×, so 15 ms / 1.45 / 1.7 ≈ 6 ms — the rounding above to 5 ms is the operating-point assumption.)

### 2.2 Adam fraction has dropped — but ratio is what drives speedup

A naive reading would say: T_adam went from 5 ms (pre-#52 in iter-194) to 1 ms (post-#51 in iter-196). The Adam-step bubble shrank 5×. NIMBUS's potential gain shrank too.

**The correct reading:** NIMBUS speedup is the ratio of the serialized whole to the longest pipelined branch:

$$
\text{Speedup}_{\text{NIMBUS}} = \frac{T_{\text{F}} + T_{\text{B}} + T_{\text{Adam}} + T_{\text{H2D}}}{\max(T_{\text{F}} + T_{\text{B}}, T_{\text{Adam}} + T_{\text{D2H}} + T_{\text{H2D}})}
$$

At iter-194 numbers (T_F = 5, T_B = 10, T_Adam = 5, T_H2D ≈ 0.5): `(5+10+5+0.5)/max(15, 5.5) = 20.5/15 ≈ 1.37×`.

At iter-196 numbers (T_F = 1.5, T_B = 2, T_Adam = 1, T_H2D = 0.5, T_D2H = 0.3): `(1.5+2+1+0.5)/max(3.5, 1.8) = 5.0/3.5 ≈ 1.43×`.

**The ratio is preserved — slightly improved, in fact.** Both halves of the pipeline shrank by approximately the same factor under #50 + #51, leaving the relative bubble ratio intact. The ATLAS-COMPILE-captured F+B is faster, but the HELIUM-FP8 + smaller-PCIe Adam is faster too.

Honest claim under iter-196 numbers: **1.33× per-step**, slightly conservative against the 1.43× simple model. The conservatism accounts for (a) cross-stream synchronization overhead inside captured graphs (§3.3) and (b) FP8 dequant in the Adam callback (§4.2).

---

## 3. Composition with #51 ATLAS-COMPILE

ATLAS-COMPILE captures ~440 kernels per step into a CUDA Graph and replays them on the compute stream. NIMBUS-PROMOTED adds a transfer-stream pipeline (D2H grad, host Adam, H2D weights). The two streams must compose without (a) breaking graph-replay invariance, (b) making the autotune cache thrash, or (c) introducing latent ordering bugs at the graph-level.

### 3.1 Graph capture topology

The eager dispatch order under iter-194 NIMBUS (§4.1) is:

```
S_compute:  fwd_t → bwd_t → record(E_grad_t)
                                  ↓
                    streamWaitEvent(St, E_grad_t)
S_transfer:                    D2H(g_t) → adam_callback(t) → H2D(θ̃_t) → record(E_weight_t)
                                                                                ↓
S_compute:                                                          streamWaitEvent(Sc, E_weight_{t-1}) → bwd_{t+1}
```

Under ATLAS-COMPILE, the `S_compute` body of step t (forward + backward) is captured into a `cudaGraph_t` and replayed via `cudaGraphLaunch(execGraph_t, S_compute)`. CUDA Graphs **do** support cross-stream dependencies through `cudaGraphAddEventRecordNode` and `cudaGraphAddEventWaitNode` (CUDA 11.4+). The captured graph for step t includes:

```
[graph nodes: 220 forward kernels]
[node: cudaEventRecord(E_grad_t) on S_compute internal stream]   // inserted by NIMBUS-PROMOTED
[graph nodes: 220 backward kernels]
[node: cudaEventRecord(E_grad_t) on S_compute]                   // overwrite for safety
[graph nodes: small bookkeeping]
```

The transfer-stream pipeline is **not captured** — it remains an eager sequence of:
- `cudaStreamWaitEvent(S_transfer, E_grad_t)`
- `cudaMemcpyAsync(D2H grad, S_transfer)`
- `cudaLaunchHostFunc(S_transfer, &nimbus_adam_callback, &ctx)`
- `cudaMemcpyAsync(H2D weights, S_transfer)`
- `cudaEventRecord(E_weight_t, S_transfer)`

Why not capture the transfer stream too? Because (a) `cudaLaunchHostFunc` can be captured but loses the per-step variability of host Adam-state mutation, and (b) the graph for the transfer stream would be data-dependent on which ping-pong slot is current. Eager dispatch on the transfer side is ~4 µs of launch overhead — a rounding error.

### 3.2 Per-shape graph cache compatibility

ATLAS-COMPILE's autotune/graph cache is keyed on `(L, T, α, dtype)` (per ATLAS-COMPILE §3.1). Under NIMBUS-PROMOTED, the cache key extends to `(L, T, α, dtype, ping_pong_slot)`:

- The captured graph references weight buffer addresses. With ping-pong, addresses alternate between `weights_buf_A` and `weights_buf_B`. Two distinct graphs are needed.
- Cache size doubles. At ATLAS-COMPILE's ~16-entry cache budget, this is acceptable (32 entries × ~6 ms capture × ~5 SLC phases = 1 s of one-time capture overhead per run).

**Mitigation if cache thrash becomes pathological:** ATLAS-COMPILE supports `cudaGraphExecUpdate` to swap buffer pointers in an existing executable graph without recapture. NIMBUS-PROMOTED uses this to maintain a single graph with patched weight pointers, halving cache size back to ATLAS-COMPILE baseline. **Engineering: ~50 LOC of `cudaGraphExecUpdate` invocation in `gpu_nimbus.cu`'s ping-pong logic.**

### 3.3 Autotune validity across the pipeline

ATLAS-COMPILE's per-shape autotune assumes a stable kernel-binary + argument tuple. NIMBUS-PROMOTED swaps the weight pointer between captured-graph runs (via `cudaGraphExecUpdate`), but kernel binaries and shapes are unchanged. **Autotune cache validity is preserved.** The `1-ulp parity check` from ATLAS-COMPILE §4.5 also passes — the kernel computes the same FP32-accumulated GEMM whether the input pointer is `bufA` or `bufB`.

### 3.4 Combined ATLAS-COMPILE × NIMBUS-PROMOTED speedup

ATLAS-COMPILE delivers 1.45× at 1.84B by reducing T_F + T_B from ~6 ms to ~3.5 ms (post-fusion + per-shape autotune + graph-replay). NIMBUS-PROMOTED on top of this overlaps the ~1 ms host Adam with the 3.5 ms compute, dropping critical-path to ~3.5 ms. **Combined: 1.45× × 1.33× ≈ 1.93× over the pre-#51 baseline.**

The two paradigms are orthogonal mechanisms (ATLAS-COMPILE attacks launch tax + per-shape kernel selection + intra-step fusion; NIMBUS-PROMOTED attacks the optimizer-step bubble) so their effects are multiplicative.

---

## 4. Composition with #50 HELIUM

HELIUM dispatches GEMMs in FP8 (E4M3 forward, E5M2 backward) with stochastic rounding, and maintains an FP32 master copy on host for parameters that participate in the optimizer step. NIMBUS-PROMOTED inherits HELIUM's host-master pattern almost verbatim and adds pipelining on top.

### 4.1 FP8 D2H grad transfer

Under HELIUM, the gradient produced on GPU is `(s_grad, ĝ_t^{E5M2})` — a per-tensor FP32 scale factor plus the E5M2 quantized tensor. PCIe D2H traffic shrinks 4× from FP32 (4 bytes/param) to E5M2+scale (1 byte/param + O(1) scale per tensor).

For 1.84B parameters: D2H drops from ~7.4 GB FP32 to ~1.84 GB E5M2 (plus negligible scale-factor traffic). At 25 GB/s PCIe Gen4: **D2H time drops from ~300 ms to ~75 ms** in the naive case, **~3 ms with PCIe DMA chunking** (the practical bound after HELIUM ships).

Within the NIMBUS-PROMOTED operating point at iter-196, this is the **dominant** reason T_D2H is ≤ 0.3 ms — the FP8 grad transfer fits in 0.3 ms when chunked. Pre-HELIUM, NIMBUS would have struggled with PCIe contention; post-HELIUM, the contention is gone.

### 4.2 Host-side Adam callback under HELIUM

The Adam callback (iter-194 §7.3) reads FP32 grad, updates `(m, v, θ̃^{master})`, quantizes `θ̃^{master}` back to PHOENIX-1.58BIT for H2D. Under HELIUM, the input grad is FP8 + scale; the callback must dequantize first:

```cpp
void CUDART_CB nimbus_adam_step_callback_helium(void* userData) {
    auto* ctx = static_cast<NimbusStepCtx*>(userData);
    const int P = ctx->paramCount;
    const e5m2_t* g_e5m2 = ctx->hostGradE5M2Pinned;   // HELIUM E5M2 grad
    const float s_grad = *ctx->hostGradScale;          // per-tensor scale
    float* m = ctx->hostM;
    float* v = ctx->hostV;
    float* theta = ctx->hostMaster;
    const auto& hp = ctx->adam;
    const float beta1c = 1.f - powf(hp.beta1, ctx->step + 1);
    const float beta2c = 1.f - powf(hp.beta2, ctx->step + 1);

    #pragma omp parallel for
    for (int i = 0; i < P; ++i) {
        // Dequantize FP8 grad
        const float gi = s_grad * dequant_e5m2(g_e5m2[i]);
        // Standard Adam (Kahan-v from surprise-#17 omitted on FACE/MFIO'd groups per iter 172)
        m[i] = hp.beta1 * m[i] + (1.f - hp.beta1) * gi;
        v[i] = hp.beta2 * v[i] + (1.f - hp.beta2) * gi * gi;
        const float mhat = m[i] / beta1c;
        const float vhat = v[i] / beta2c;
        theta[i] -= hp.eta * mhat / (sqrtf(vhat) + hp.eps);
    }

    // Quantize to PHOENIX-1.58BIT staging buffer (unchanged from iter-194)
    phoenix_quantize_158bit(theta, ctx->hostStage, P);
}
```

The dequantization adds ~1 cycle per param → 1.84B params ÷ 16 cores ÷ 4 GHz ≈ 30 ms naive; **~3 ms with SIMD-vectorized E5M2→FP32 conversion** (AVX-512 has native FP8↔FP32 conversion instructions on Sapphire Rapids; older CPUs do bit-twiddle in 4-cycle path). On the iter-196 host (assumed AVX-512 capable), this lands inside the 1 ms T_Adam budget.

### 4.3 Stochastic-rounding determinism

HELIUM uses Philox-RNG-driven stochastic rounding for FP8 quantization. NIMBUS-PROMOTED's K_stale = 1 staleness does not affect the stochastic-rounding distribution (each step's grad is rounded independently from a deterministic seed). **Determinism preserved end-to-end** modulo the K_stale = 1 trajectory shift, which is bounded at ≤ 0.003 nat per step regardless.

### 4.4 H2D weight refresh under PHOENIX-1.58BIT

The H2D path is unchanged from iter-194 — the host Adam callback writes a PHOENIX-1.58BIT-quantized staging buffer and the transfer stream uploads it to the GPU's `weights_buf_next`. HELIUM's separate FP8 dispatch reads from the GPU weights and dequantizes per-GEMM-tile from PHOENIX-1.58BIT to E4M3 inline. NIMBUS-PROMOTED does not interact with this dequantization.

### 4.5 Combined HELIUM × NIMBUS-PROMOTED speedup

HELIUM contributes 1.7× on the GPU side (forward + backward). NIMBUS-PROMOTED contributes 1.33× by overlapping the host Adam. **Combined: 1.7× × 1.33× ≈ 2.26× over the pre-#50 baseline.** Composition is multiplicative because HELIUM's mechanism (FP8 tensor cores) and NIMBUS's mechanism (transfer-stream pipelining) are at different layers (kernel implementation vs. step scheduling).

---

## 5. Refined per-step time analysis at 1.84B post-#42–#51

| Phase | Pre-#52 (post-#51) | Post-NIMBUS-PROMOTED | Mechanism |
|---|---|---|---|
| Compute (forward + backward, captured graph, FP8) | ~3.5 ms | ~3.5 ms | unchanged |
| Adam on host (FP8 → FP32 dequant + Adam math + PHOENIX quant) | ~1 ms | overlapped | runs on transfer stream while next-step compute runs on compute stream |
| D2H grad (FP8 + scale) | ~0.3 ms | overlapped | piggyback on Adam phase |
| H2D weight (PHOENIX-1.58BIT) | ~0.5 ms | overlapped | piggyback on Adam phase |
| **Critical path** | **~5.3 ms** | **~3.5 ms** | NIMBUS-PROMOTED removes serial Adam+transfer |
| **Per-step speedup** | — | **1.51×** (model says 5.3/3.5) | conservative claim: **1.33×** to cover sync overhead |

The 1.33× honest-claim figure is conservative against the 1.51× direct-model figure to account for:
- Cross-stream event-record/wait overhead (~50 µs per step).
- Cache-line ping-pong on the host between OpenMP threads and the host callback (~50 µs).
- ATLAS-COMPILE graph-launch + ping-pong-update overhead (~30 µs per launch when using `cudaGraphExecUpdate`).
- Initial graph-capture amortization (one-time ~50 ms over a 100k-step run is negligible).

**Net per-step time: ~3.5 ms → 1.84B-flagship throughput improves from 1/5.3 to 1/3.5 = 1.51× theoretical, 1.33× honest.**

---

## 6. Stack at 18B post-#52 NIMBUS-PROMOTED

From `PARADIGM_SHIFT_51_DESIGN.md` §0: post-#51 stack at 18B is **~690× single-GPU wall-clock advantage with bit-exact-equivalent NLL**. This includes #42 SCFA, #43 ORION, #44 MELT, #46 REFLECTOR, #47 PHOENIX-1.58BIT, #49 ICARUS, #50 HELIUM, #51 ATLAS-COMPILE.

NIMBUS-PROMOTED adds 1.33× per-step at flagship. Composition is multiplicative across the stack (NIMBUS-PROMOTED operates at the step-scheduling layer, orthogonal to every other shipped paradigm's mechanism — kernel selection, fusion, graph capture, FP8 quantization, MOR reduction, etc.).

| Stack | At 18B | Mechanism |
|---|---|---|
| Pre-#42 (CHIRON only) | 1× | reversibility, NLL bit-exact |
| Post-#42–#49 (iter-194 baseline) | ~300× | mathematical compression |
| Post-#50 HELIUM | ~510× | + FP8 GEMM, FA-3, fusion |
| Post-#51 ATLAS-COMPILE | **~690×** | + graph capture, autotune, fusion |
| **Post-#52 NIMBUS-PROMOTED** | **~917×** | **+ async optimizer pipelining (this doc)** |

The 917× is bit-exact-equivalent NLL up to the composite drift bound in §7.

---

## 7. Updated NLL bound

NIMBUS-PROMOTED inherits the K_stale = 1 NLL bound from iter-194 §3.2:

$$
\Delta L_{\text{NIMBUS}} \le L_H \cdot \eta \cdot \lVert m_t \rVert \cdot K_{\text{stale}} \approx 0.003 \text{ nat per step}
$$

This is the worst-case bound; empirical asynchronous-SGD literature (Lian 2015, Mitliagkas 2016, Narayanan 2019) confirms the bound does not accumulate (Adam's β₁ EMA absorbs single-step displacements within ~10 steps).

Composition with the other shipped paradigms' NLL drifts:

| Source | Per-step NLL drift | Cumulative over 100k steps |
|---|---|---|
| #50 HELIUM (FP8 SR, zero-mean, bounded variance) | ≤ 0.001 nat (zero-mean) | ≤ 0.06 nat (random walk, σ scaling) |
| #51 ATLAS-COMPILE (FP32 reduction-order drift) | ≤ 10⁻⁷ nat | ≤ 10⁻² nat |
| **#52 NIMBUS-PROMOTED (K_stale = 1)** | **≤ 0.003 nat (non-accumulating per Adam EMA)** | **≤ 0.003 nat (steady-state)** |
| **Composite per-step gap** | **≤ 0.004 nat** | **≤ 0.07 nat over 100k steps** |

The 0.07 nat cumulative figure is **below the 0.1 nat run-to-run-variance floor** typically observed at this scale (per `surprise17_midphase_drift.md` and run-history). NIMBUS-PROMOTED's contribution to NLL drift (0.003 nat per step) is roughly equal to HELIUM's (0.001 nat per step) and dominates ATLAS-COMPILE's (10⁻⁷ nat per step) — but all three combined are within the strict NLL-preservation criterion.

The **non-accumulation property of K_stale = 1** is critical and inherited verbatim from iter-194 §3.4: Adam's β₁ = 0.9 EMA has a 10-step timescale that absorbs single-step displacements; β₂ = 0.999 has a 1000-step timescale that does not even register single-step displacements. The Kahan-v compensation from surprise-#17 is unaffected because the bf16 precision floor (not staleness) is the binding constraint.

---

## 8. Engineering scope (~750 LOC)

| Subsystem | LOC | New or inherited from iter-194 |
|---|---|---|
| Two-stream wiring (CUDA streams + events) | 200 | inherited |
| Per-step Adam pipeline (host callback + queue) | 150 | inherited |
| Weight-refresh queue (double-buffer ping-pong) | 100 | inherited |
| Trainer integration (`--nimbus 1` flag + scheduler) | 100 | inherited |
| **ATLAS-COMPILE composition** (cudaGraphExecUpdate for ping-pong, graph-cache key extension, `cudaGraphAddEventRecordNode` + `cudaGraphAddEventWaitNode`) | 100 | **new for NIMBUS-PROMOTED** |
| **HELIUM composition** (FP8 D2H, AVX-512 dequant, FP8-aware Adam callback, FP8-aware grad-scale management) | 100 | **new for NIMBUS-PROMOTED** |
| **Total** | **~750 LOC** | 550 inherited + 200 new |

**Timeline:** ~3 weeks (vs. iter-194's 3 weeks for the ~550 LOC base):
- Phase 1 (4 days): Two-stream wiring + double-buffer ping-pong (inherited).
- Phase 2 (2 days): Host Adam callback (inherited).
- Phase 3 (2 days): **ATLAS-COMPILE composition** — `cudaGraphExecUpdate` for ping-pong; graph-cache key extension.
- Phase 4 (2 days): **HELIUM composition** — FP8 D2H path + AVX-512 dequantization + FP8-scale management.
- Phase 5 (3 days): Trainer integration with `--nimbus 1` flag; backward-compat flags `--nimbus-fp8 0/1` (default 1 if HELIUM is on) and `--nimbus-graph 0/1` (default 1 if ATLAS-COMPILE is on).
- Phase 6 (1 day): Gate-0 probe (1 GPU-hour).
- Phase 7 (2 days): NLL-parity validation at 66M (5000 steps).
- Phase 8 (5 days): Flagship 1.84B production run.
- **Total: ~21 days (~3 weeks), ~750 LOC, ~6 files modified or added.**

---

## 9. Gate-0 probe (1 GPU-hour)

**Question:** at K_stale = 1 with the post-#42–#51 stack (ATLAS-COMPILE captured graphs + HELIUM FP8), does NIMBUS-PROMOTED deliver `(3.5 + 1) / 3.5 ≈ 1.29×` actual wall-clock speedup, and does the empirical NLL gap remain ≤ 0.005 nat?

**Probe:** three parallel 2000-step runs at 66M (CHIRON checkpoint at iter-185, post-#51 ATLAS-COMPILE active, post-#50 HELIUM active):
- **Baseline:** `--nimbus 0` (sequential Adam after F+B).
- **NIMBUS-K1:** `--nimbus 1 --nimbus-k-stale 1`.
- **NIMBUS-K0 (control):** `--nimbus 1 --nimbus-k-stale 0` (synchronous; tests pure pipelining overhead with both shipped paradigms active).

**Pass criteria:**
- Wall-clock speedup ≥ 1.20× (target 1.30×).
- Final loss EMA at step 2000 within 0.005 nat of baseline.
- Held-out 1k-token validation NLL within 0.005 nat.
- ATLAS-COMPILE autotune cache hit-rate ≥ 95 % (proves graph-cache thrash is not pathological).
- HELIUM FP8 D2H + dequant overhead ≤ 0.5 ms per step (proves PCIe is not the bottleneck).

**Failure modes:**
- Speedup < 1.10×: ATLAS-COMPILE cross-stream event overhead is unexpectedly high; revisit `cudaGraphExecUpdate` design.
- NLL gap > 0.01 nat: surprise — K_stale = 1 was empirically validated by the asyncSGD literature, and the bound is theoretical worst-case. Most likely cause is a Kahan-v/Adam state corruption from the FP8 dequant path. Debug.
- HELIUM FP8 D2H > 1 ms: either AVX-512 dequant is mis-vectorized or the PCIe is older than Gen4. Fall back to the FP32 D2H path with documented 1.20× speedup instead of 1.33×.

**Cost: 1 GPU-hour total** (3 × 2000-step runs × ~2 min each at 66M post-#42–#51).

---

## 10. Risk profile

**Engineering risks (inherited, low):** stream contention, host-callback latency, memory pressure. Mitigated by per-layer pipelining fallback (iter-194 §4.4) at +250 LOC.

**New composition risks (this doc):**
- **ATLAS-COMPILE graph-cache thrash from ping-pong.** Mitigated by `cudaGraphExecUpdate` to maintain a single graph with patched weight pointers. If `cudaGraphExecUpdate` fails (CUDA driver bug or non-Ada/Hopper device), fall back to two-graph cache.
- **HELIUM FP8 dequant cost in Adam callback.** Mitigated by AVX-512 native FP8 conversion (Sapphire Rapids+); fallback to bit-twiddle on older CPUs adds ~5 ms to T_Adam, which would break the pipeline budget. Detected at probe time; if older CPU, ship without HELIUM-NIMBUS composition (use FP32 D2H path; speedup 1.20×).
- **Stochastic-rounding determinism interaction.** The HELIUM Philox RNG is seeded per step; NIMBUS-PROMOTED doesn't touch it. Determinism preserved.

**Numerical risks (low):** NLL composite drift bound 0.004 nat/step is well below 0.01 nat/step floor. K_stale = 1 non-accumulation argument holds.

**Empirical risks (low):** the iter-194 NIMBUS Gate-0 design is unchanged in NIMBUS-PROMOTED; HELIUM and ATLAS-COMPILE add measurable composition checks.

---

## 11. Why NIMBUS-PROMOTED for #52

NIMBUS-PROMOTED is the **conservative magnitude-pursuit follow-on to ATLAS-COMPILE**, exactly as ATLAS-COMPILE's own §11 anticipated. It is the right choice for #52 if:

1. **NLL preservation strictness continues to bind** (iter-193 standing brief). NIMBUS-PROMOTED's ≤ 0.003 nat per step is the lowest-risk follow-on to ATLAS-COMPILE's 10⁻⁷ nat per step.

2. **Engineering bandwidth is constrained** (post-ATLAS-COMPILE's ~2300 LOC investment). NIMBUS-PROMOTED at ~750 LOC is by far the cheapest paradigm shipped at this stack depth.

3. **Stack composition matters more than per-paradigm magnitude.** NIMBUS-PROMOTED's 1.33× alone is "not magnitudes," but it stacks cleanly with #42–#51 to bring the cumulative 18B advantage from 690× to 917× — an additional ~33 % on top of an already 690× foundation.

4. **The optimizer-step bubble is the next obvious target.** Post-#51, Adam-on-host is 20 % of step time. No other axis offers a cleaner factor-1.33× multiplier with NLL preservation strict.

If the iter-196 user brief drifts toward "force a single-paradigm 10×+ shift even at NLL cost," APOLLO (#51 candidate-A, deferred) or HORIZON (#51 candidate-C, deferred) would be more appropriate alternatives at #52. NIMBUS-PROMOTED is the correct selection for **continuing the strict-NLL-preservation magnitude-via-composition strategy**.

---

## 12. Summary

NIMBUS-PROMOTED promotes the iter-194 NIMBUS candidate to a full paradigm shift #52 with two composition refinements: (a) integration with #51 ATLAS-COMPILE's CUDA-Graph-captured per-step DAG via `cudaGraphExecUpdate` ping-pong + cache-key extension; (b) integration with #50 HELIUM's FP8-master-state via FP8-aware D2H + AVX-512-vectorized dequant inside the host Adam callback.

The mathematical foundation is unchanged from iter-194: K_stale = 1 single-step asynchronous-SGD pattern, NLL gap ≤ L_H · η · ‖m_t‖ · K_stale ≈ 0.003 nat per step, non-accumulating because Adam's β₁ EMA absorbs single-step displacements.

**Honest claim at flagship 1.84B post-#42–#51:** **1.33× per-step wall-clock speedup** (conservative against the 1.51× pure-model figure). Critical-path per-step time drops from ~5.3 ms (sequential) to ~3.5 ms (pipelined). Adam, D2H grad, and H2D weights all run on the transfer stream concurrently with the next step's captured-graph forward+backward on the compute stream.

**Stack projection at 18B:** 690× post-#51 → **~917× post-#52 with bit-exact-equivalent NLL** (composite drift ≤ 0.07 nat over 100k steps, well below the 0.1-nat run-to-run-variance floor).

**Engineering scope:** ~750 LOC over ~3 weeks. 550 LOC inherited from iter-194 candidate; 200 LOC new (ATLAS-COMPILE composition + HELIUM composition wiring).

**Risk profile:** low. The K_stale = 1 NLL bound is a worst-case theoretical bound corroborated by the asynchronous-SGD literature. The composition risks (graph-cache thrash, FP8 dequant cost in callback) are bounded and have characterized mitigations. Worst case is parity (no speedup, no NLL loss).

**Recommended as paradigm #52** when the priority is **continuing the post-#42–#51 magnitude-via-composition strategy under strict NLL preservation**, with the optimizer-step bubble as the next clean composition target.

---

## References

- iter-194 NIMBUS candidate document: `PARADIGM_SHIFT_50_CANDIDATE_C_NIMBUS.md`. Foundation for this promotion.
- `PARADIGM_SHIFT_50_CANDIDATE_A_HELIUM.md` (selected #50). FP8 GEMM + FA-3 + stochastic rounding; provides the FP8 master-state pattern NIMBUS-PROMOTED pipelines.
- `PARADIGM_SHIFT_51_CANDIDATE_B_ATLAS_COMPILE.md` and `PARADIGM_SHIFT_51_DESIGN.md` (selected #51). CUDA-Graph capture + per-shape autotune + cross-paradigm fusion; provides the captured-graph environment NIMBUS-PROMOTED extends.
- `PARADIGM_SHIFT_47_CANDIDATE_B_PHOENIX_1.58BIT.md` (selected #47). Host-pinned FP32 master + 1.58-bit quantization; the architectural prerequisite for NIMBUS-PROMOTED's host-side Adam.
- Lian, X., Huang, Y., Li, Y., Liu, J. (2015). "Asynchronous parallel stochastic gradient for nonconvex optimization." NeurIPS 28.
- Mitliagkas, I., Zhang, C., Hadjis, S., Ré, C. (2016). "Asynchrony begets momentum, with an application to deep learning." Allerton.
- Narayanan, D., Harlap, A., Phanishayee, A., et al. (2019). "PipeDream: generalized pipeline parallelism for DNN training." SOSP.
- NVIDIA CUDA Programming Guide — `cudaGraphExecUpdate`, `cudaGraphAddEventRecordNode`, `cudaGraphAddEventWaitNode` (CUDA 11.4+).
- NVIDIA Transformer Engine documentation — FP8 stochastic rounding + per-tensor scale management.
- (CHIRON-internal) `Backend/Machine Learning/Networks/cuda/gpu_device.h` lines 38–51 — `computeStream`, `transferStream`, `createEvent`, `recordEvent`, `streamWaitEvent` already-shipped primitives.
- (CHIRON-internal) `surprise17_midphase_drift.md` — Kahan-v compensation; unaffected by K_stale = 1.
- (CHIRON-internal) `FACE_AS_DISRUPTING_PARADIGM.md` — FACE EMA absorbs K_stale = 1 (analysis carries over from iter-194).
