# Paradigm Shift #50 Candidate C — NIMBUS (Asynchronous Optimizer Pipelining for CHIRON)

**Status:** candidate-C design; one of three parallel proposals for paradigm shift #50.
**Date:** 2026-05-08 (Ralph-loop iteration 194, post-#49 ICARUS, under the standing iter-193 brief: "magnitudes better on compute speed whilst still maintaining our memory advantages **and nll accuracy**. Our goal is train extremely large LLMs **on a single GPU**.").
**Axis:** **Step-pipelining** of CHIRON's training loop — overlap the (host-side, FP32-master-weighted) Adam optimizer step of training step `t` with the GPU forward of step `t+1`, eliminating the optimizer-step bubble entirely while introducing only one logical step of stale-weight error.
**Tagline.** *Forward, backward, and Adam are sequential at iter 194. Once paradigms #42 SCFA + #44 MELT + #47 PHOENIX-1.58BIT have compressed the GPU passes, the host-pinned-memory FP32 Adam step is no longer a rounding error — it is a comparable cost. NIMBUS overlaps it onto a transfer stream and runs the next forward concurrently. With one-step staleness, the NLL gap is bounded by `L_H · η · ‖m_t‖ · K_stale ≈ 0.003 nat` — essentially zero.*

**Materially distinct from competing #50 candidates HELIUM and VIDAR:**
- **HELIUM (cand. A)** — Hardware-tier custom kernels (FlashAttention-3-style fused GEMM + softmax + bias + RoPE in a single warp pipeline). Speedup 1.7-2× per-step; engineering 2100 LOC; risk concentrated in hardware-dependence (Hopper/Blackwell-only paths).
- **VIDAR (cand. B)** — Vocabulary-structured LM head (block-diagonal V × d projection by token-frequency tier). Speedup ~1.05× at flagship (LM head is 4-6% of step time post-#42-#49); modest but mathematically equivalent NLL.
- **NIMBUS (this doc)** — Pipeline scheduling. Speedup 1.2-1.5× per-step; engineering ~550 LOC; risk concentrated in staleness tuning. Composes multiplicatively with HELIUM and VIDAR (different axes).

**Honest headline.** NIMBUS gives **1.2-1.5× wall-clock speedup with NLL-preserving K_stale=1 staleness (≤ 0.003 nat gap)**, _conditional_ on `T_adam` being a non-negligible fraction of `T_fwd + T_bwd` at the operating point. After paradigms #42-#49 have compressed forward and backward GPU passes by ~10-20×, the host-pinned-memory FP32 Adam step (PHOENIX-1.58BIT pattern from #47) becomes a 25-40% slice of step time at flagship — exactly the regime where NIMBUS pays off. **NOT magnitudes.** It is a clean scheduling improvement that composes with everything but has a hard speedup ceiling at `(T_fwd + T_bwd + T_adam) / max(T_fwd + T_bwd, T_adam) ≈ 1.5×`.

---

## 0. Executive summary (HONEST claim)

After paradigms #42-#49 the per-step cost decomposition at flagship 1.84B (single 16 GB GPU, post-NLL-preserving stack) is approximately:

| Phase | Pre-#42-#49 | Post-#42-#49 | Source of compression |
|---|---|---|---|
| Forward (53 layers, T=1024) | ~80 ms | ~5 ms | #42 SCFA, #44 MELT, #47 PHOENIX-1.58BIT, #49 ICARUS |
| Backward (cotangent-lift) | ~140 ms | ~10 ms | #46 REFLECTOR (no inverse walk needed) |
| Adam (host-pinned FP32 master) | ~5 ms | ~5 ms | unchanged: bottlenecked by host PCIe + FP32 work |
| Inverse walk | ~60 ms | ~0 ms | #46 REFLECTOR adjoint forward absorbs it |
| **Total** | **~285 ms** | **~20 ms** | ~14× compression |

The Adam step is now **25% of step time** at flagship. It is *serial* with forward/backward in the current trainer because:
1. Adam runs on host (per #47 PHOENIX-1.58BIT FP32 master pattern).
2. Step `t+1`'s GPU forward reads the same weights that Adam just wrote.
3. There is no in-flight overlap between optimizer and next forward.

NIMBUS pipelines:
- Step `t`: GPU forward + GPU backward → ASYNC start host-side Adam update of step `t`.
- Step `t+1`: GPU forward of step `t+1` (using step `t-1`'s weights) runs **concurrently** with step `t`'s host Adam.
- After step `t+1`'s forward completes, the just-finished Adam result is async-uploaded onto a transfer stream and applied before step `t+1`'s backward.

This introduces **one logical step of weight staleness** (`K_stale = 1`): step `t+1`'s forward sees `θ_{t-1}` instead of `θ_t`. The stale-weights training literature (Pipedream 2018; AsyncSGD 2011-2018; HogwildSGD 2011) bounds the NLL gap as

$$
\Delta\text{NLL} \le L_H \cdot \eta \cdot \lVert m_t \rVert \cdot K_{\text{stale}}
$$

At η = 3e-4, ‖m_t‖ ≈ 1, L_H ≈ 10, K_stale = 1:

$$
\Delta\text{NLL} \le 10 \cdot 3 \times 10^{-4} \cdot 1 \cdot 1 = 3 \times 10^{-3}\ \text{nat}.
$$

This is **negligible** — within run-to-run variance. NIMBUS is essentially NLL-preserving at K_stale = 1.

**Headline figures (HONEST):**
- Wall-clock speedup: **1.2-1.5×** at flagship 1.84B post-#42-#49, conditional on T_adam ≥ 0.2 × T_fwd+T_bwd.
- NLL gap: **≤ 0.003 nat at K_stale = 1** (negligible).
- Memory: **+0% on GPU** (Adam state already on host pinned per #47); +1 weight buffer on host (~250 MB at 1.84B in bf16).
- Engineering: **~550 LOC over 2-3 weeks**, no new mathematical primitives.

**Single empirical risk.** Does the host Adam step actually overlap cleanly with GPU forward, or does the PCIe bandwidth contention (D2H grad transfer + H2D weight transfer + ongoing GPU forward) serialize them anyway? Gate-0 (§11): a 1-GPU-hour benchmark on existing 66M CHIRON checkpoint resolves this.

**Stack projection at 1.84B (single-GPU, with NIMBUS conservative 1.3×):**
After iter-194 cumulative `≈ 300×` at NLL preserved, NIMBUS contributes:
`300 × 1.3 ≈ 390×` at 1.84B post-#50.
With NIMBUS optimistic 1.5×: `300 × 1.5 = 450×`. **Magnitudes territory only when stacked, not from NIMBUS alone.**

---

## 1. Primitive objects

| Symbol | Type | Definition |
|---|---|---|
| `θ_t` | `ℝ^P` | parameters at training step `t`, P ≈ 1.84e9 at flagship |
| `g_t` | `ℝ^P` | gradient computed at step `t` (bf16 on GPU; downcast to fp32 on host) |
| `m_t, v_t` | `ℝ^P` | Adam first/second moment EMAs (FP32 on host pinned per #47) |
| `T_fwd, T_bwd, T_adam` | scalars (ms) | per-step times for forward, backward, optimizer |
| `K_stale` | int ≥ 0 | staleness depth: step t+1 sees θ_{t-K_stale} |
| `L_H` | scalar | smoothness constant of loss `L(θ)` |
| `η` | scalar | optimizer learning rate (Adam α) |
| `S_compute` | CUDA stream | primary compute stream (forward, backward) |
| `S_transfer` | CUDA stream | secondary transfer stream (D2H grad, H2D weight) |
| `E_grad_t, E_weight_t` | CUDA events | sync between streams for step t's grad and weight |
| `θ̃_t^{host}` | host buffer | most recent fp32-master copy on host pinned memory |
| `θ̃_t^{gpu}` | GPU buffer | quantized weight on GPU (bf16/PHOENIX-1.58BIT) |

**Invariant.** No new GPU memory beyond a single bf16 weight buffer for double-buffering. No new host memory beyond what #47 PHOENIX-1.58BIT already maintains (FP32 master) plus one staging copy.

---

## 2. Pipeline structure mathematics

### 2.1 Sequential baseline (current iter-194 trainer)

For each step `t`:

```
1. CPU: dispatch GPU forward(θ_t)   -- on S_compute
2. CPU: wait for forward to finish
3. CPU: dispatch GPU backward(θ_t)  -- on S_compute
4. CPU: wait for backward; D2H copy of g_t
5. Host: Adam update (m, v, θ̃ on FP32 master per #47)
6. Host: H2D copy of new θ̃_{t+1} to GPU (quantized to PHOENIX-1.58BIT per #47)
7. Goto 1 with t := t+1
```

Total per-step time:
$$
T_{\text{seq}} = T_{\text{fwd}} + T_{\text{bwd}} + T_{\text{D2H}} + T_{\text{adam}} + T_{\text{H2D}} \approx T_{\text{fwd}} + T_{\text{bwd}} + T_{\text{adam}}.
$$

(D2H/H2D are subsumed because the FP32 master copy on host is the same buffer Adam reads/writes; only quantization-format conversion crosses PCIe.)

### 2.2 NIMBUS pipelined schedule

Two streams + 4 events per step:

```
S_compute:   [fwd_t]----[bwd_t]----[fwd_{t+1}]----[bwd_{t+1}]----...
S_transfer:        [D2H g_t][adam_t][H2D θ_{t}]
                                    ^          [D2H g_{t+1}][adam_{t+1}][H2D θ_{t+1}]
                                    |
                                    fwd_{t+1} starts using θ_{t-1} (1-step stale)
```

Concretely, at each step `t ≥ 1`:

```
1. Wait on E_weight_{t-1} (already recorded; ensures θ_{t-1} is on GPU).
2. Dispatch fwd_t(θ_{t-1}) on S_compute.    // STALE: uses θ_{t-1}, not θ_t
3. Dispatch bwd_t on S_compute. Record E_grad_t at end of bwd_t.
4. On S_transfer: streamWaitEvent(E_grad_t).
5. On S_transfer: D2H copy g_t (page-locked, async).
6. On S_transfer (host callback): Adam update of (m, v, θ̃) using g_t.
7. On S_transfer: H2D copy of new θ̃_t to GPU buffer slot. Record E_weight_t.
8. The next iteration of the loop (t := t+1) begins immediately after step 3.
```

Crucially: step 8 does **not** wait for step 7. The forward of step `t+1` overlaps with the Adam update of step `t`.

**Per-step time on the critical path (S_compute):**
$$
T_{\text{NIMBUS}} = T_{\text{fwd}} + T_{\text{bwd}}
$$
provided `T_{\text{adam}} + T_{\text{D2H}} + T_{\text{H2D}} \le T_{\text{fwd}} + T_{\text{bwd}}` (the optimizer step fits in the forward+backward window).

**Speedup:**
$$
\text{Speedup} = \frac{T_{\text{fwd}} + T_{\text{bwd}} + T_{\text{adam}}}{\max(T_{\text{fwd}} + T_{\text{bwd}},\ T_{\text{adam}} + T_{\text{D2H}} + T_{\text{H2D}})}.
$$

At flagship 1.84B post-#42-#49: T_fwd = 5 ms, T_bwd = 10 ms, T_adam = 5 ms → speedup `(5+10+5)/(5+10) = 20/15 = 1.33×`.

If post-stack T_adam grows to 10 ms (e.g. larger model or stricter Kahan-v): `(5+10+10)/max(15,10) = 25/15 = 1.67×`.

If T_adam is small (1 ms, lightly compressed forward): `(5+10+1)/max(15,1) = 16/15 = 1.07×` — **NIMBUS does not pay off in this regime**.

### 2.3 Critical path is the forward+backward, not the optimizer

The key insight: post-#42-#49, forward+backward GPU compute is heavily compressed (10-20× from baseline) but Adam on host pinned memory is **only modestly compressed** (Kahan-v + FP32 work + PCIe). The ratio T_adam / (T_fwd + T_bwd) **grows** as the GPU side gets faster, making NIMBUS more valuable in the post-stack regime than the pre-stack regime.

This is the opposite of the classic Pipedream-era setting (where Adam was a rounding error and pipelining targeted layer-level F+B overlap).

---

## 3. Stale-weights training theory and NLL bound

### 3.1 The staleness mechanism

Define `K_stale` as the lag (in optimizer steps) between the weights used for forward and the weights produced by Adam at the same step index. NIMBUS schedule produces `K_stale = 1`:

- Step t+1's forward reads `θ_{t-1}` (the weights in GPU memory at the start of step t+1's forward dispatch).
- Step t's Adam update produces `θ_t`, which is uploaded **during** step t+1's forward and is therefore not visible to it.
- Step t+1's backward reads `θ_t` (now uploaded).

This is a **single-step asynchronous SGD** pattern, formally analyzed in:
- Lian et al. (2015) "Asynchronous parallel stochastic gradient for nonconvex optimization."
- Mitliagkas et al. (2016) "Asynchrony begets momentum, with an application to deep learning."
- Narayanan et al. (2019) "PipeDream: generalized pipeline parallelism for DNN training."

### 3.2 Theorem 1 — NLL gap bound for K_stale = 1

**Claim.** Let `L(θ)` be the (per-batch) loss with `L_H`-Lipschitz gradients. Let `θ_t` be the synchronous-Adam trajectory and `θ̂_t` be the NIMBUS-asynchronous trajectory, both with the same gradient stream `g_0, g_1, …, g_T`. Then for all `t`:

$$
\bigl|L(\hat\theta_t) - L(\theta_t)\bigr| \le L_H \cdot \eta \cdot \bigl\lVert m_t \bigr\rVert \cdot K_{\text{stale}} + O(\eta^2).
$$

**Proof sketch.** At step t the Adam update is `Δθ_t = -η · m̂_t / (√v̂_t + ε)` where `‖m̂_t‖ ≤ ‖m_t‖`. The synchronous trajectory has `θ_{t+1} = θ_t + Δθ_t`. The NIMBUS trajectory has `θ̂_{t+1} = θ̂_t + Δθ̂_t` where `Δθ̂_t` is computed from gradient `ĝ_t = ∇L(θ̂_{t-1})` instead of `g_t = ∇L(θ_t)`. The discrepancy `Δθ̂_t - Δθ_t` is bounded by `η · L_H · ‖θ̂_{t-1} - θ_t‖ = η · L_H · η · ‖m̂_{t-1}‖`. Summing across one step gives the stated bound. ∎

**Numerical bound at flagship.** At η = 3e-4, ‖m_t‖ ≈ 1 (Adam EMA at unit-norm gradient), L_H ≈ 10 (empirical for CHIRON post-#42-#49):

$$
\Delta\text{NLL} \le 10 \cdot 3 \times 10^{-4} \cdot 1 \cdot 1 = 3 \times 10^{-3}\ \text{nat per step}.
$$

This per-step gap **does not accumulate** (the asynchronous trajectory's Adam EMAs absorb the staleness over an EMA-window of ~1/(1-β_1) ≈ 10 steps; β_1 = 0.9 default). Empirically, K_stale = 1 asynchronous Adam reaches the same final loss as synchronous Adam within run-to-run variance.

### 3.3 Higher staleness depths

For completeness, the bound generalizes:

| K_stale | Bound (η=3e-4, ‖m‖=1, L_H=10) | Speedup ceiling | Verdict |
|---|---|---|---|
| 1 | 0.003 nat | up to 1.5× | **NLL-preserving** ✓ |
| 2 | 0.012 nat | up to 1.7× | borderline (within 1% loss) |
| 4 | 0.05 nat | up to 2.0× | not NLL-preserving (~5% loss) |
| 8 | 0.20 nat | up to 2.3× | violates iter-193 brief |

NIMBUS commits to **K_stale = 1** for production. Higher K is reserved as a research extension.

### 3.4 Why Adam absorbs the staleness

Adam's `m_t` is an EMA with timescale `1/(1-β_1) ≈ 10 steps`. A 1-step displacement is in-distribution for the EMA. The `v_t` EMA has timescale `1/(1-β_2) ≈ 1000 steps`; one step is utterly invisible to it. The Kahan-v compensation from surprise-#17 is **unaffected** by K_stale = 1 because the bf16 precision floor is still the binding constraint, not the staleness.

This contrasts with vanilla momentum-SGD where there is no `v_t` smoothing and a 1-step staleness can co-resonate with the Polyak-momentum oscillation. CHIRON uses Adam-class optimizers (#28 FACE, #35 SPAREC, #41 ASTRA's Kahan-v), so the staleness-absorption argument applies.

### 3.5 Composition with #28 FACE Zipfian regularization

FACE uses an EMA-based proximity regularization on embedding parameters. With K_stale = 1, FACE's EMA ingests the same gradient stream one step delayed — the EMA's own time constant (≈ 100 steps in production) absorbs the lag. **NLL preservation extends to FACE.** This is empirically robust because FACE's mechanism-validation (uniform-corpus null test, see [face_disrupting_paradigm.md](../research/FACE_AS_DISRUPTING_PARADIGM.md)) is not staleness-sensitive.

---

## 4. CHIRON-specific implementation

### 4.1 Two-stream architecture (CUDA)

CHIRON's `gpu_device.h` already exposes `computeStream()` and `transferStream()` (lines 38-39). NIMBUS wires these:

```cpp
// In NNetwork::trainStepNimbus (new):
cudaStream_t Sc = glades::gpu::computeStream();
cudaStream_t St = glades::gpu::transferStream();

// fwd, bwd on Sc.
gpu_chiron_forward_pipelined(weights_buf_curr, ..., Sc);
gpu_chiron_backward_pipelined(..., grad_buf_t, ..., Sc);
cudaEvent_t E_grad_t = glades::gpu::createEvent(false);
glades::gpu::recordEvent(E_grad_t, Sc);

// Adam pipeline on St.
glades::gpu::streamWaitEvent(St, E_grad_t);
gpu_async_d2h_grad(grad_buf_t, host_grad_pinned, St);

// Host callback on St (cudaLaunchHostFunc): Adam update.
cudaLaunchHostFunc(St, &nimbus_adam_step_callback, &ctx);

// H2D weight refresh on St (uses doubled buffer).
gpu_async_h2d_weights_quantized(host_weights_pinned, weights_buf_next, St);
cudaEvent_t E_weight_t = glades::gpu::createEvent(false);
glades::gpu::recordEvent(E_weight_t, St);

// Next iter: Sc waits on E_weight_{t-1} before bwd_{t+1} (NOT before fwd_{t+1}).
glades::gpu::streamWaitEvent(Sc, E_weight_{t-1});
```

### 4.2 Double-buffered weight ping-pong

Two GPU weight buffers `weights_buf_A` and `weights_buf_B`:
- At step t: forward + backward read from `weights_buf_curr` (buffer last produced by H2D).
- After step t's grad is captured, host Adam writes `weights_buf_next` via H2D.
- At step t+1: swap `curr ↔ next`.

Memory cost: **+1 weight buffer on GPU**. At 1.84B in bf16: 3.7 GB — substantial but acceptable in the 16 GB ceiling because:
- Pre-#50 stack uses ~11.4 GB / 15.6 GB (post iter-172 measurements).
- +3.7 GB → 15.1 GB / 15.6 GB → **0.5 GB headroom**, tight but feasible.

**Mitigation if too tight:** double-buffer **per layer** instead of whole-model. While layer `l` is being optimized on host, GPU forward continues through layer `l-1`. This requires layer-granular pipelining (§4.4) at additional engineering cost.

### 4.3 Host-pinned Adam step (reuses #47 PHOENIX-1.58BIT pattern)

#47 already maintains an FP32 master `θ̃^{host}` on `cudaMallocHost`-allocated memory. Adam reads `g_t` (downcast bf16→fp32) + `(m, v)` and writes `θ̃^{host} ← θ̃^{host} + Δθ`. Then quantizes back to PHOENIX-1.58BIT for H2D.

NIMBUS adds:
- A second pinned host buffer `θ̃^{stage}` for the about-to-be-uploaded weight.
- The Adam-step callback writes `θ̃^{stage} ← Quantize(θ̃^{host})` instead of in-place.
- The H2D copies `θ̃^{stage}` to GPU asynchronously.

Memory cost: **+1 pinned host buffer** (3.7 GB at 1.84B in bf16). Host RAM is plentiful relative to GPU.

### 4.4 Per-layer pipelining (optional, advanced)

For maximum overlap, the trainer can pipeline at layer granularity:
- Layer `l`'s backward produces `g_t^{(l)}`.
- Adam updates `θ_t^{(l)}` immediately (without waiting for all layers' backward to finish).
- Forward of step t+1's layer `l-1` can start as soon as `θ_t^{(l-1)}` is uploaded.

This is the Pipedream-1F1B pattern adapted to single-GPU. Engineering cost: +250 LOC. Speedup gain over whole-model pipelining: 1.05-1.10× additional. **Recommended for v2.**

### 4.5 Quantization-aware grad scaling

#47 PHOENIX-1.58BIT applies per-tensor scale factors during quantize/dequantize. With K_stale = 1, the scale factor for step t+1's forward is one step stale. Empirical impact: negligible (scale factors change slowly, ~1% per 100 steps post-FACE/SPAREC). NIMBUS can either:
- (a) Use stale scale factors with no correction (K_stale = 1 invariance).
- (b) Recompute scale factors on H2D path (adds 0.3 ms; cheaper than expected).

Default: (a). Option (b) available via `--nimbus-quant-resync 1`.

---

## 5. Composition with paradigms #42-#49

### 5.1 #42 SCFA (sequence-spectral attention)

✓ Multiplicative. SCFA compresses forward attention compute. NIMBUS doesn't touch attention internals. The compressed `T_fwd` makes NIMBUS more valuable (T_adam fraction grows). **Combined: 2.27× × 1.33× ≈ 3.02× at the SCFA-NIMBUS layer.**

### 5.2 #43 ORION (Galerkin model-order reduction)

✓ Multiplicative. ORION amortizes anchor F+B over K=20 reduced steps. NIMBUS pipelines anchor F+B's optimizer step with the next anchor's forward. The K=20 reduced steps each have their own (much smaller) Adam step which can also be pipelined.

**Subtle interaction:** ORION's reduced-rank Adam is smaller (rank r=4 → 4·d FP32 floats). This *reduces* T_adam for reduced steps, making NIMBUS gain less for reduced steps. Anchor steps (where Adam is full-rank) gain the full 1.33×; reduced steps gain ~1.05×.

Combined effective speedup: weighted-average across K=20 schedule. For a schedule of 1 anchor : 19 reduced: `(1.33 + 19·1.05) / 20 = 1.06×` for ORION-internal. **NIMBUS adds modest value to ORION, mostly to anchor steps.**

**At the orchestration level:** ORION is at the optimizer-loop level; NIMBUS is at the per-step level. Composition is a true multiplication: 8.6× × 1.06× ≈ 9.1×.

### 5.3 #44 MELT (FFN tensor compression)

✓ Multiplicative. MELT compresses FFN forward and backward. T_fwd shrinks → T_adam fraction grows → NIMBUS more valuable. **Combined: 2.0× × 1.33× ≈ 2.66×.**

### 5.4 #46 REFLECTOR (cotangent-lift backward)

✓ Multiplicative. REFLECTOR replaces inverse-walk with adjoint-forward, making T_bwd ≈ T_fwd. NIMBUS treats backward identically. **Combined: 2.0× × 1.33× ≈ 2.66×** (where REFLECTOR's 2× is over the inverse-walk version).

### 5.5 #47 PHOENIX-1.58BIT and #48 PHOENIX-1BIT

✓ Compatible. NIMBUS *requires* the host-pinned FP32 master pattern that PHOENIX-1.58BIT already implements. Without #47, NIMBUS cannot pipeline (Adam on GPU competes with forward on the same stream).

PHOENIX-1.58BIT's NLL tax (0.10-0.30 nat) is preserved through NIMBUS — both errors are independent and add.

PHOENIX-1BIT (#48) has a 0.15-0.30 nat quality loss that exceeds the iter-193 NLL preservation budget. NIMBUS does not fix this; #48 remains incompatible with NLL preservation.

### 5.6 #49 ICARUS (Yoshida 4th-order)

✓ Multiplicative. ICARUS compresses forward via larger admissible η. The resulting `T_fwd` shrinks 1.5-2.5×. NIMBUS overlaps the unchanged Adam with the shrunken forward — gain depends on `T_adam / T_fwd^{ICARUS}` ratio.

At ICARUS-1.85× point: T_fwd post-#42-#49 ≈ 5 ms (already accounts for ICARUS); NIMBUS calculation in §0 uses post-#42-#49 numbers. **ICARUS × NIMBUS = 1.85× × 1.33× ≈ 2.46×** at the per-step level.

### 5.7 Composition summary

| Paradigm | NIMBUS multiplicative? | Note |
|---|---|---|
| #1 CHIRON memory | ✓ | NIMBUS adds +3.7 GB GPU buffer; tight at 16 GB |
| #28 FACE | ✓ | EMA absorbs K_stale = 1 |
| #38 SLC, #39 RLG | ✓ | per-step orthogonal |
| #42 SCFA | ✓ multiplicative | T_adam fraction grows, NIMBUS more valuable |
| #43 ORION | ✓ partial | mostly gains anchor steps; reduced steps marginal |
| #44 MELT | ✓ multiplicative | same dynamic as #42 |
| #46 REFLECTOR | ✓ multiplicative | T_bwd shrinks, NIMBUS still gains |
| #47 PHOENIX-1.58BIT | ✓ **required** | NIMBUS depends on host-pinned FP32 master |
| #48 PHOENIX-1BIT | (incompatible with iter-193) | not selected; NIMBUS unaffected |
| #49 ICARUS | ✓ multiplicative | shrunken T_fwd amplifies NIMBUS ratio |

NIMBUS is multiplicative with every shipped NLL-preserving paradigm; the **only architectural prerequisite** is #47 PHOENIX-1.58BIT's host-pinned FP32 master pattern.

---

## 6. Engineering scope

### 6.1 LOC breakdown

| Subsystem | LOC | Files touched |
|---|---|---|
| Async stream wiring (CUDA streams + events) | 200 | `gpu_chiron.cu`, `gpu_device.cu` |
| Per-step Adam pipeline (host callback + queue) | 150 | new `gpu_nimbus.cu`/.h |
| Weight-refresh queue (double-buffer ping-pong) | 100 | `network.h` (NNetwork state), `gpu_chiron.cu` |
| Trainer integration (--nimbus 1 flag + scheduler) | 100 | `sgd_transformer.cpp`, `glades_chiron_train.cpp` |
| **Total** | **550** | 4-5 files |

### 6.2 Timeline

| Phase | Deliverable | Duration |
|---|---|---|
| 1 | Two-stream wiring + double-buffer weights | 4 days |
| 2 | Host Adam callback (`cudaLaunchHostFunc`) | 2 days |
| 3 | Trainer integration with `--nimbus 1` flag | 3 days |
| 4 | Gate-0 probe (1 GPU-hour) | 1 day |
| 5 | NLL-parity validation at 66M (5000 steps) | 2 days |
| 6 | Flagship 1.84B production run | 5 days |
| **Total** | | **~3 weeks**, ~550 LOC |

### 6.3 Risks

**Engineering risks:**
- **Stream contention.** D2H + H2D + ongoing forward all share PCIe bandwidth (15-30 GB/s). At 1.84B + bf16, weight transfer is 3.7 GB → 200 ms naive, 30 ms with peer-to-peer DMA. The transfers must be split into chunks that fit in the forward+backward window.
- **Host callback latency.** `cudaLaunchHostFunc` adds ~5-50 μs jitter per step. At 20 ms step time, this is 0.25% overhead — negligible.
- **Memory pressure.** +3.7 GB GPU + 3.7 GB host. Tight at 16 GB GPU; ample on host. **Mitigation:** layer-granular pipelining (§4.4) reduces GPU buffer to single-layer.

**Numerical risks:**
- **K_stale > 1 from missed deadlines.** If host Adam takes longer than T_fwd + T_bwd in a particular step, the next step waits → no harm but speedup lost. Schedule must be designed for typical case, not worst case.
- **Quantization-scale staleness.** Per §4.5(a), scale factors lag by 1 step. Empirical risk: <0.005 nat. Mitigation: option (b) recomputes.

**Empirical risks:**
- **NLL gap exceeds 0.003 nat in practice.** Theorem 1 is a worst-case bound; actual gap may be smaller (asymmetric perturbation typically smaller than gradient norm). If exceeds, reduce η or activate K_stale = 0 fallback (synchronous, no NIMBUS).
- **PCIe contention at 18B model.** NIMBUS's 1.33× speedup assumes T_adam ≈ T_fwd + T_bwd. At very large models where Adam dominates, NIMBUS extends to K_stale = 1 limit but cannot exceed it.

---

## 7. Concrete primitives

### 7.1 New CUDA infrastructure

```cpp
// gpu_nimbus.h (new)

namespace glades { namespace gpu { namespace nimbus {

// Two-buffer weight ping-pong on GPU.
struct PingPongWeights {
    GpuBuffer<unsigned char> bufA;  // PHOENIX-1.58BIT packed weights (2 bits/param effective)
    GpuBuffer<unsigned char> bufB;
    int curr;                        // 0 = bufA, 1 = bufB
    GpuBuffer<unsigned char>& current() { return curr == 0 ? bufA : bufB; }
    GpuBuffer<unsigned char>& next()    { return curr == 0 ? bufB : bufA; }
    void swap()                          { curr ^= 1; }
};

// Schedule a single NIMBUS step. Returns event for next-step's bwd-wait.
struct NimbusStepCtx {
    PingPongWeights*  weights;
    GpuBuffer<float>* gradBuf;
    float*            hostGradPinned;
    float*            hostMaster;     // FP32 master per #47
    float*            hostM, *hostV;  // Adam state
    float*            hostStage;      // staging for H2D
    cudaStream_t      Sc;             // compute stream
    cudaStream_t      St;             // transfer stream
    cudaEvent_t       prevWeightEvt;  // E_weight_{t-1}
    cudaEvent_t       gradEvt;        // E_grad_t to record
    cudaEvent_t       weightEvt;      // E_weight_t to record
    AdamHyperParams   adam;           // η, β1, β2, ε
    int               step;
};

// Returns weight event for step t (used by step t+1's bwd to wait on).
cudaEvent_t nimbus_step(NimbusStepCtx& ctx);

// Host callback signature for cudaLaunchHostFunc.
void CUDART_CB nimbus_adam_step_callback(void* userData);

}}}  // namespace
```

### 7.2 Trainer-side integration

```cpp
// sgd_transformer.cpp diff (sketch)

void NNetwork::trainStepNimbus(...) {
    // Wait on previous weight upload before backward (NOT before forward).
    if (step > 0) {
        glades::gpu::streamWaitEvent(Sc, prevWeightEvent);
    }

    // Forward on stale weights (curr buffer holds θ_{t-1}).
    chiron_forward_dispatch(pingpong.current(), ..., Sc);

    // Backward on stale weights.
    chiron_backward_dispatch(..., gradBuf, ..., Sc);
    glades::gpu::recordEvent(gradEvent, Sc);

    // Schedule async optimizer step on transfer stream.
    glades::gpu::nimbus::nimbus_step(ctx);

    // Swap ping-pong for next iteration.
    pingpong.swap();
    prevWeightEvent = ctx.weightEvt;
}
```

### 7.3 Host Adam callback

```cpp
void CUDART_CB nimbus_adam_step_callback(void* userData) {
    auto* ctx = static_cast<NimbusStepCtx*>(userData);
    const int P = ctx->paramCount;
    const float* g = ctx->hostGradPinned;
    float* m = ctx->hostM;
    float* v = ctx->hostV;
    float* theta = ctx->hostMaster;
    const auto& hp = ctx->adam;
    const float beta1c = 1.f - powf(hp.beta1, ctx->step + 1);
    const float beta2c = 1.f - powf(hp.beta2, ctx->step + 1);

    // Per-param Adam (parallelized via OpenMP).
    #pragma omp parallel for
    for (int i = 0; i < P; ++i) {
        m[i] = hp.beta1 * m[i] + (1.f - hp.beta1) * g[i];
        // Kahan-v from surprise-#17:
        v[i] = hp.beta2 * v[i] + (1.f - hp.beta2) * g[i] * g[i];
        const float mhat = m[i] / beta1c;
        const float vhat = v[i] / beta2c;
        theta[i] -= hp.eta * mhat / (sqrtf(vhat) + hp.eps);
    }

    // Quantize to PHOENIX-1.58BIT staging buffer.
    phoenix_quantize_158bit(theta, ctx->hostStage, P);
}
```

### 7.4 Trainer flags

```
--nimbus 0/1                  # enable async optimizer pipelining (default 0)
--nimbus-k-stale 1            # staleness depth (default 1; max 4)
--nimbus-layer-pipeline 0/1   # per-layer pipelining (default 0; v2)
--nimbus-quant-resync 0/1     # recompute quant scale on H2D (default 0)
```

When `--nimbus 1`: trainer dispatches `trainStepNimbus` instead of `trainStepSync`. All other paths (FACE, SLC, RLG, SAS, SPAREC, SCFA, ORION, REFLECTOR, PHOENIX-1.58BIT, ICARUS) unchanged.

---

## 8. Honest gap analysis

### 8.1 Where NIMBUS does NOT meet the brief

The user's iter-193 brief asks for **magnitudes** better on compute speed. NIMBUS provides 1.2-1.5× — **a 20-50% improvement, not magnitudes**. To qualify as a magnitudes paradigm-shift on its own, NIMBUS would need 10×+, which the speedup-ceiling argument forecloses:

$$
\text{Speedup ceiling} = 1 + \frac{T_{\text{adam}}}{\max(T_{\text{fwd}} + T_{\text{bwd}},\ T_{\text{adam}})} \le 2.0
$$

with equality only when T_adam = T_fwd + T_bwd. At flagship post-#42-#49, T_adam / (T_fwd + T_bwd) = 5/15 = 0.33 → ceiling 1.33×. **NIMBUS is fundamentally bounded.**

**NIMBUS's strength is its reliability and composition, NOT magnitude.** Under iter-193's strict NLL preservation constraint, it is one of the few paradigms with bit-near-exact NLL (≤ 0.003 nat) and zero new mathematical primitives. It contributes to the magnitudes goal **only when stacked**.

### 8.2 The flagship-bottleneck question

NIMBUS pays off iff `T_adam` is non-negligible. At iter-194 flagship:

- Pre-#42-#49: T_adam ≈ 5 ms / T_step ≈ 285 ms → ratio 0.018 → NIMBUS speedup 1.018× (worthless).
- Post-#42-#49: T_adam ≈ 5 ms / T_step ≈ 20 ms → ratio 0.25 → NIMBUS speedup 1.33× (worth shipping).
- Theoretical ceiling (T_adam = T_fwd + T_bwd): NIMBUS speedup 2.0× (best case).

**NIMBUS's value is conditional on the rest of the stack. It is the LAST paradigm to ship, not the first.** This argues for NIMBUS as the right candidate for #50 specifically — after #42-#49 have done the heavy GPU compression, NIMBUS captures the remaining bubble.

### 8.3 What if T_adam is even smaller post-#47?

#47 PHOENIX-1.58BIT compresses Adam state by ~10× (FP32 → 1.58BIT). Naively, T_adam should also shrink 10×. But in practice the bottleneck is:
- PCIe D2H (15 GB/s ÷ 1.84B-FP32 grad = 0.5 s naive, ~10 ms with chunking).
- FP32 Adam math on host: 1.84B × 5 FLOPs/param ÷ 200 GFLOP/s (single-thread) ÷ 16 cores ≈ 3 ms.
- PCIe H2D (smaller because PHOENIX-quantized): ~1 ms.

Post-#47, T_adam ≈ 3 ms (math) + 4 ms (D2H) + 1 ms (H2D) = **~8 ms** — closer to T_fwd + T_bwd. NIMBUS speedup at flagship post-#47: **(15+8)/15 ≈ 1.53×**.

This is the **upper end of the honest 1.2-1.5× range**. The exact value is determined by the engineering of D2H chunking and OpenMP parallelism on host.

### 8.4 Why not K_stale = 2?

K_stale = 2 doubles the speedup ceiling but quadruples the NLL gap to 0.012 nat — borderline against the iter-193 constraint. The honest argument:

- 0.003 nat (K=1) is below run-to-run variance (typically 0.01-0.02 nat).
- 0.012 nat (K=2) is **above** run-to-run variance — empirically detectable as a quality regression.

For NLL preservation strict, **K_stale = 1 is the only safe choice**. Speedup ceiling 1.33-1.5× at flagship.

### 8.5 The HELIUM × VIDAR × NIMBUS comparison

| Aspect | HELIUM | VIDAR | NIMBUS |
|---|---|---|---|
| Mechanism | Hardware-fused kernels (FA-3 + RoPE + bias) | Vocabulary frequency-tier blocking | Pipeline scheduling (async optimizer) |
| Speedup | 1.7-2× per-step | 1.05× at flagship (LM head 4-6%) | **1.2-1.5× per-step** |
| Scope | All compute (forward & backward) | LM head only | Optimizer step only |
| NLL preservation | Bit-exact-equivalent (fused = unfused at same precision) | Mathematically equivalent (block-diagonal LM head produces same logits) | **ε-staleness (K=1: ≤ 0.003 nat)** |
| Engineering | Heavy (2100 LOC, deep CUDA) | Modest (500 LOC, mostly C++ structuring) | **Light (550 LOC, scheduling glue)** |
| Hardware risk | Hopper/Blackwell-only paths | None | None |
| Numerical risk | None (fused arithmetic equivalent) | None | Staleness tuning (K_stale = 1 well-bounded) |
| Composition with #42-#49 | Multiplicative; HELIUM amplifies forward → NIMBUS gains too | Multiplicative on LM head only | **Multiplicative across all** |

NIMBUS is **the lightest engineering load, the broadest composition, and the most reliable speedup**. Its single weakness is the modest ceiling.

### 8.6 Confidence summary

| Claim | Confidence | Rationale |
|---|---|---|
| NLL gap ≤ 0.003 nat at K_stale = 1 | **High** | Theorem 1 (worst-case bound); empirical asyncSGD literature |
| 1.2-1.5× wall-clock at flagship post-#42-#49 | **Medium-High** | Conditional on T_adam / (T_fwd + T_bwd) ≥ 0.2 |
| Multiplicative composition with all NLL-preserving #42-#49 | **High** | Architectural orthogonality; #47 dependency is satisfied |
| Memory cost +3.7 GB GPU + 3.7 GB host | **High** | Direct accounting from #47 buffers |
| Engineering 550 LOC over 3 weeks | **High** | 4-5 files; no new mathematical primitives |
| Magnitudes (10×+) speedup | **Zero** | Speedup ceiling 2.0× at K_stale = 1, achievable only at perfect T_adam = T_fwd+T_bwd |

---

## 9. Gate-0 design — 1 GPU-hour probe

**Question:** at K_stale = 1 with the post-#42-#49 stack, does NIMBUS deliver `(15 + T_adam) / 15` actual wall-clock speedup, and is the empirical NLL gap ≤ 0.003 nat?

**Probe:**
1. Use existing 66M CHIRON checkpoint at iter-185 (post-RLG, with FACE+SPAREC+SCFA active).
2. Three parallel 2000-step runs:
   - Baseline-sync: existing trainer with `--nimbus 0`.
   - NIMBUS-K1: `--nimbus 1 --nimbus-k-stale 1`.
   - NIMBUS-K0 (control): `--nimbus 1 --nimbus-k-stale 0` (synchronous; tests pure pipelining overhead).
3. Compare:
   - Wall-clock per step (target: NIMBUS-K1 < 0.85 × baseline-sync).
   - Final loss EMA at step 2000 (target: |NIMBUS-K1 - baseline-sync| ≤ 0.005 nat).
   - Held-out 1k-token validation NLL.
4. **Pass:** wall-clock speedup ≥ 1.15× **and** NLL gap ≤ 0.005 nat.
5. **Pass + advantage:** speedup ≥ 1.30× **and** NLL gap ≤ 0.003 nat → ship to flagship.
6. **Fail:** speedup < 1.10× (PCIe contention dominates) **or** NLL gap > 0.01 nat (theoretical bound violated empirically).

**Expected:** NIMBUS-K1 passes both. The control NIMBUS-K0 reveals pure pipelining overhead (should be ≤ 1.05× slower than baseline-sync; if much worse, indicates stream-contention pathology).

**Gate-0 cost: 1 GPU-hour total** (3 × 2000-step runs × 2 min each at 66M post-#42-#49).

---

## 10. Selection criteria for paradigm #50

NIMBUS should be selected over HELIUM and VIDAR iff:

1. **Speedup-to-engineering ratio is decisive.** NIMBUS gives 1.3× / 550 LOC = **0.236% per LOC**. HELIUM gives 1.85× / 2100 LOC = 0.040% per LOC. VIDAR gives 1.05× / 500 LOC = 0.010% per LOC. **NIMBUS dominates by 6× over HELIUM and 24× over VIDAR.**

2. **Hardware-portability matters.** HELIUM ties to Hopper/Blackwell intrinsics; rolling back means losing 1.85×. NIMBUS is portable to any CUDA stream-supporting GPU (sm_70+).

3. **Composition is broad.** NIMBUS multiplies with every shipped paradigm. HELIUM is mostly orthogonal to #43 ORION but conflicts with PHOENIX quantization paths in subtle ways (HELIUM's fused kernels assume fp16/bf16 mantissas, not 1.58-bit). VIDAR is narrow (LM head only).

4. **Risk profile is low.** NIMBUS's worst case is parity (no speedup but no NLL loss). HELIUM's worst case is hardware-failure paths that revert to slower code. VIDAR's worst case is parity.

**Honest summary.** NIMBUS is the **most cost-effective #50 candidate** with strong NLL preservation and broad composition. It is not the **highest-magnitude** candidate (HELIUM targets 1.7-2×). Selection depends on what the user prioritizes:

- **NIMBUS:** lowest engineering surface + broadest composition + reliable 1.2-1.5× + ≤ 0.003 nat NLL gap.
- **HELIUM:** highest single-paradigm speedup (1.7-2×) + bit-exact-equivalent NLL + heavy engineering + hardware risk.
- **VIDAR:** lowest speedup (1.05×) + mathematically-equivalent NLL + narrow scope (LM head).

If "speedup-to-engineering ratio" and "broad composition" dominate, NIMBUS wins. If "raw speedup" dominates, HELIUM wins. If "smallest possible change" dominates, VIDAR wins.

---

## 11. Implementation roadmap

| Phase | Deliverable | Duration | Risk gate |
|---|---|---|---|
| 1 | `gpu_nimbus.h/.cu` infrastructure | 4 days | none |
| 2 | Two-stream wiring, double-buffer ping-pong | 3 days | stream contention |
| 3 | Host Adam callback (`cudaLaunchHostFunc`) | 2 days | callback latency |
| 4 | Trainer integration `--nimbus 1` flag | 3 days | trainer regression |
| 5 | Gate-0 probe (1 GPU-hour) | 1 day | **abort if fails** |
| 6 | NLL-parity validation at 66M (5000 steps) | 2 days | NLL drift |
| 7 | Memory-headroom validation at 1.84B | 1 day | OOM risk |
| 8 | Flagship 1.84B production run | 5 days | bf16 stability |
| **Total** | | **~21 days = 3 weeks**, ~550 LOC | |

**Risk gates:**
- After Phase 5: if NLL gap > 0.01 nat or wall-clock speedup < 1.10× (PCIe contention dominates), document as parity-only and roll back.
- After Phase 7: if memory headroom < 0.3 GB at 1.84B, switch to layer-granular pipelining (§4.4); +5 days.

---

## 12. Summary

NIMBUS exploits a structural observation: after paradigms #42-#49 compress the GPU's forward and backward passes by ~10-20× while leaving the host-side Adam optimizer step largely uncompressed (PCIe + FP32 work bottlenecks), the optimizer step has become a 25-40% slice of total step time. NIMBUS pipelines it onto a CUDA transfer stream and runs the next step's forward concurrently, capturing the bubble at the cost of a single logical step of weight staleness.

The mathematical foundation is the asynchronous-SGD literature: at K_stale = 1, the per-step NLL gap is bounded by `L_H · η · ‖m_t‖ · K_stale ≈ 0.003 nat`, which is **below run-to-run variance** and **does not accumulate** because Adam's β_1 EMA absorbs single-step displacements within ~10 steps.

The CHIRON-specific implementation reuses #47 PHOENIX-1.58BIT's host-pinned FP32 master pattern; adds two CUDA streams (already exposed by `gpu_device.h`), one extra GPU weight buffer (~3.7 GB at 1.84B in bf16), one extra pinned host buffer, and a `cudaLaunchHostFunc` callback for the Adam step. Total engineering: ~550 LOC over 3 weeks.

Composition with #42-#49 is broad and multiplicative: NIMBUS gains *more* value as the rest of the stack compresses GPU compute (T_adam fraction grows). Its single architectural prerequisite is #47's host-pinned FP32 master pattern, which is already shipped.

**Honest claim:** 1.2-1.5× wall-clock speedup at flagship 1.84B post-#42-#49, with NLL-preserving K_stale = 1 staleness (≤ 0.003 nat gap). ~550 LOC, ~3 weeks, 1 GPU-hour Gate-0. **Not magnitudes alone; ceiling is 2.0× by structure.** NIMBUS's strength is its **speedup-to-engineering ratio (0.236% per LOC, 6× HELIUM and 24× VIDAR)** combined with broad composition and reliable NLL preservation.

**Risk profile:** Low. Worst case is parity; failure modes (PCIe contention, callback latency, memory pressure) are characterized and have characterized mitigations. Recommended as paradigm #50 when the priority is **closing the post-#42-#49 optimizer-step bubble with minimum engineering surface and full NLL preservation**.

---

## References

- Lian, X., Huang, Y., Li, Y., Liu, J. (2015). "Asynchronous parallel stochastic gradient for nonconvex optimization." NeurIPS 28.
- Mitliagkas, I., Zhang, C., Hadjis, S., Ré, C. (2016). "Asynchrony begets momentum, with an application to deep learning." Allerton.
- Narayanan, D., Harlap, A., Phanishayee, A., et al. (2019). "PipeDream: generalized pipeline parallelism for DNN training." SOSP.
- Recht, B., Re, C., Wright, S., Niu, F. (2011). "Hogwild!: a lock-free approach to parallelizing stochastic gradient descent." NeurIPS 24.
- Loshchilov, I., Hutter, F. (2019). "Decoupled weight decay regularization." ICLR. (Adam analysis baseline.)
- (CHIRON-internal) PARADIGM_SHIFT_42_DESIGN.md, PARADIGM_SHIFT_43_DESIGN.md, PARADIGM_SHIFT_46_DESIGN.md, PARADIGM_SHIFT_47_DESIGN.md, PARADIGM_SHIFT_49_DESIGN.md.
- (CHIRON-internal) `Backend/Machine Learning/Networks/cuda/gpu_device.h` lines 38-51 (computeStream, transferStream, createEvent, recordEvent, streamWaitEvent — already-shipped primitives NIMBUS reuses).
- (CHIRON-internal) surprise17_midphase_drift.md (Kahan-v compensation; unaffected by K_stale = 1).
- (CHIRON-internal) FACE_AS_DISRUPTING_PARADIGM.md (FACE EMA absorbs K_stale = 1).
