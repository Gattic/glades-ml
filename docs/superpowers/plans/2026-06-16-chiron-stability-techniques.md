# CHIRON Stability/Optimization Techniques Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement five techniques to cure (not just contain) the q-side
reverse-amplification instability that gates CHIRON data-scaling: Adaptive
Gradient Clipping (AGC), Gradient Centralization (GC), bounded ReLN backward,
spectral/weight normalization, and SAM.

**Architecture:** Each is an independent, default-off, parity-gated phase
sharing one validation harness (the fresh accum=4 danger-zone run). Phases are
ordered by expected-value/cost: cheapest and closest-to-validated first (AGC),
most invasive last (SAM). The current `clamp_vector_l2norm` per-group clamp
(mitigation 1, validated 2026-06-15) is the baseline each phase A/B's against.

**Tech Stack:** C++98 + CUDA (glades-ml `libglades.so`), glades-trainer
`chiron_main.cpp`. Library kernels in `gpu_kernels.cu`/`gpu_chiron.cu` with
header decls + no-CUDA stubs in the matching `.h`; trainer flags forward-
declared in `chiron_main.cpp` (snapshot lags); `run.sh` passthrough.

---

## Shared context (read once)

- **Failure mode** (research/QSIDE_INSTABILITY_INVESTIGATION_2026_06_14.md):
  past ~1.3B tokens the model enters a fragile/sharp minimum; dgamma/dbeta
  spike broadly (up to 43/48 vectors), summing past the 1e20 global guard.
  accum=4/lr3e-4 diverges @1.49B; accum=1/lr7.5e-5 degrades @1.6B.
- **Validated baseline**: `--grad-group-clamp 1.0` (fixed per-group L2 clamp on
  dgamma/dbeta via `clamp_vector_l2norm`) held a fresh run through the danger
  zone (0 skips to 1.64B). Kernel: `gpu_kernels.cu`; trainer site: just before
  `compute_grad_norm_sq` in the training loop (search `cfg.gradGroupClamp`).
- **Integration anchors** (verified):
  - ReLN backward: `gpu_chiron.cu:968 chiron_reln_backward` → delegates to
    `layernorm_backward(dq_out,q_in,gamma,mean,invStd,T,m,dq_in,dgamma,dbeta)`.
  - dgamma/dbeta buffers: `W.dgamma[l]`, `W.dbeta[l]` = `GpuBuffer<float>` size m.
  - Weights: `W.gamma[l]`,`W.beta[l]` (size m); `W.Wq/Wk/Wv/Wo[l]` (FP32) or
    `W.Wq_bf[l]` etc (BF16) when `cfg.bf16Weights`; grads `W.dWq[l]`/`W.dWq_bf[l]`.
  - Adam: trainer `adam_step(cfg,W,step,gradScale,lrScale)` wraps the
    `glades::gpu::adam_update_int8_state_bf16w_bf16g_*` family.
  - Training-loop fwd/bwd/adam region: search `if (skipAdamUpdate)` and the
    `chiron_backward`/forward calls in `transformerGpuTrainEpoch`-equivalent.
- **Shared validation harness** (every phase): fresh accum=4 / lr 3e-4 run to
  step 25000 (through the 22737 danger zone) with the phase's flag, full
  recipe. PASS = 0 grad-skips + val trajectory on-trend (matches the
  gg-clamp baseline ~3.36 @ step 24000). Resume is INVALID (data pos not
  restored — recovery-A lesson); always fresh. ~16h GPU/phase.
- **Parity for default-off**: with the flag off, every phase must be
  bit-identical to current (300-step A/B/C vs rerun-noise control, the
  established methodology).

---

## Phase 1 — Adaptive Gradient Clipping (AGC)

**Rationale:** the principled generalization of the validated fixed per-group
clamp. Instead of fixed maxNorm=1.0 (fires on most steps past 1.3B), clip each
group's grad to `λ·max(‖param‖, ε)` — auto-scaled to the parameter's own
magnitude (NFNets, Brock 2021). Should fire selectively (only when a grad is
large *relative to its weights*), removing the calibration guess.

**Files:**
- Modify: `Backend/Machine Learning/Networks/cuda/gpu_kernels.cu` (new kernel + wrapper)
- Modify: `Backend/Machine Learning/Networks/cuda/gpu_kernels.h` (decl + stub)
- Test: `unit-tests/Backend/Machine Learning/chiron-test.cpp` (+selector)
- Modify: `glades-trainer/trainer/chiron_main.cpp` (flag + site)
- Modify: `glades-trainer/run.sh` (passthrough)

- [ ] **Step 1.1: Write the failing unit test** in chiron-test.cpp,
  `CHIRONAgcClampTest`: param ‖w‖=2.0 (n=2048 const), grad ‖g‖=10.0, λ=0.01 →
  expect grad scaled to ‖g‖==λ·‖w‖==0.02; grad ‖g‖=0.01 (< λ‖w‖) → untouched
  (bit-identical); param ‖w‖≈0 (use ε floor) → grad clamped to λ·ε; count
  exact. Register `chiron-agc`/`agc` selector in `unit-tests/main.cpp`.

- [ ] **Step 1.2: Run, verify FAIL** (`agc_clamp_vector` undefined).
  `cd unit-tests && ./build/glades-unit-tests chiron-agc`.

- [ ] **Step 1.3: Implement kernel** in gpu_kernels.cu (anon namespace, after
  `clamp_vector_l2norm_kernel`):

```cpp
__global__ void agc_clamp_vector_kernel(float* __restrict__ g,
                                        const float* __restrict__ w,
                                        int n, float lambda, float eps,
                                        int* __restrict__ clampedCount)
{
	__shared__ double s_gg[256]; __shared__ double s_ww[256]; __shared__ float s_scale;
	double gg = 0.0, ww = 0.0;
	for (int i = threadIdx.x; i < n; i += blockDim.x) {
		const float gv = g[i], wv = w[i];
		gg += (double)gv * gv; ww += (double)wv * wv;
	}
	s_gg[threadIdx.x] = gg; s_ww[threadIdx.x] = ww; __syncthreads();
	for (int s = blockDim.x/2; s > 0; s >>= 1) {
		if (threadIdx.x < (unsigned)s) { s_gg[threadIdx.x]+=s_gg[threadIdx.x+s]; s_ww[threadIdx.x]+=s_ww[threadIdx.x+s]; }
		__syncthreads();
	}
	if (threadIdx.x == 0) {
		const double gnorm = sqrt(s_gg[0]);
		double wnorm = sqrt(s_ww[0]); if (wnorm < (double)eps) wnorm = (double)eps;
		const double maxg = (double)lambda * wnorm;
		float scale = 1.0f;
		if (gnorm > maxg && gnorm > 0.0) { scale = (float)(maxg / gnorm); if (clampedCount) atomicAdd(clampedCount, 1); }
		s_scale = scale;
	}
	__syncthreads();
	const float sc = s_scale; if (sc == 1.0f) return;
	for (int i = threadIdx.x; i < n; i += blockDim.x) g[i] *= sc;
}
```

  Wrapper (public, after `clamp_vector_l2norm`):

```cpp
bool agc_clamp_vector(float* g, const float* w, int n, float lambda, float eps, int* d_count) {
	if (!g || !w || n <= 0) return false;
	if (!(lambda > 0.0f)) return false;
	agc_clamp_vector_kernel<<<1, 256, 0, computeStream()>>>(g, w, n, lambda, eps, d_count);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}
```

- [ ] **Step 1.4: Add decl + no-CUDA stub** in gpu_kernels.h next to
  `clamp_vector_l2norm`:
  `bool agc_clamp_vector(float* g, const float* w, int n, float lambda, float eps, int* d_count);`
  and stub `inline bool agc_clamp_vector(float*,const float*,int,float,float,int*){return false;}`

- [ ] **Step 1.5: Build + run test, verify PASS.**
  `cmake --build build -j4 && cmake --build unit-tests/build -j4 && cd unit-tests && ./build/glades-unit-tests chiron-agc` → 0 failures. Then `cd build && make install`.

- [ ] **Step 1.6: Commit** (glades-ml):
  `git add -A && git commit -m "Add agc_clamp_vector kernel + test (AGC, q-side mitigation)"` (+Co-Authored-By trailer).

- [ ] **Step 1.7: Wire trainer flag.** In chiron_main.cpp: forward-declare
  `agc_clamp_vector` (next to the `clamp_vector_l2norm` decl); add
  `float agcLambda;` config (init 0.0f = off) + `float agcEps;` (init 1e-3f);
  parse `--agc-lambda F`/`--agc-eps F`. At the SAME site as the per-group clamp
  (before `compute_grad_norm_sq`), add an alternative branch: when
  `cfg.agcLambda > 0`, for each l call
  `agc_clamp_vector(W.dgamma[l]->data(), W.gamma[l]->data(), cfg.m, cfg.agcLambda, cfg.agcEps, count)`
  and same for `dbeta[l]`/`W.beta[l]`. Reuse `gradGroupClampCount` buffer +
  the `[grad-group-clamp]`-style log. Add to the CUDA-graphs auto-disable list.
  AGC and `--grad-group-clamp` are mutually exclusive (assert or document).

- [ ] **Step 1.8: run.sh passthrough** for `--agc-lambda`/`--agc-eps`
  (append to REGSTACK_ARGS, all 3 arg paths, per the established pattern).

- [ ] **Step 1.9: Build trainer + smoke** (30 steps, accum=4, `--agc-lambda 0.01`):
  confirm no crash, inert on healthy early steps (count 0 first steps), and
  off-vs-on losses match the rerun-noise band. Commit trainer.

- [ ] **Step 1.10: Validation gate** (harness above) with `--agc-lambda 0.01`
  (NFNet default; sweep 0.01/0.02/0.04 if it over- or under-clamps). PASS =
  0 skips through danger zone + on-trend val + FEWER clamp fires than the
  fixed-norm baseline (the selectivity win). Document in the investigation doc.

**Cost:** ~1 day dev + 16h validation. **Risk:** LOW (close to validated
mechanism). **Default off.**

---

## Phase 2 — Gradient Centralization (GC)

**Rationale:** subtract the per-output-row mean of each weight gradient before
the optimizer (Yong et al. 2020) — cheap implicit regularization that flattens
the loss landscape and improves stability/generalization. Complements clipping
(addresses the landscape, not just the spike).

**Files:** same set as Phase 1 (kernel in gpu_kernels.cu; flag in chiron_main.cpp).

- [ ] **Step 2.1: Failing test** `CHIRONGradCentralizeTest`: a [rows=4,cols=8]
  grad with known per-row means → expect each row's mean == 0 after, and the
  centered values == original − row_mean (rtol 1e-6). Register `chiron-gc`.

- [ ] **Step 2.2: Run, verify FAIL.**

- [ ] **Step 2.3: Implement kernel** (one block per row, reduce row sum,
  subtract mean):

```cpp
__global__ void gradient_centralize_kernel(float* __restrict__ g, int rows, int cols) {
	const int r = blockIdx.x; if (r >= rows) return;
	float* gr = g + (size_t)r * cols;
	__shared__ double s[256];
	double sum = 0.0;
	for (int i = threadIdx.x; i < cols; i += blockDim.x) sum += (double)gr[i];
	s[threadIdx.x] = sum; __syncthreads();
	for (int st = blockDim.x/2; st > 0; st >>= 1) { if (threadIdx.x < (unsigned)st) s[threadIdx.x]+=s[threadIdx.x+st]; __syncthreads(); }
	__shared__ float mean; if (threadIdx.x == 0) mean = (float)(s[0] / (double)cols); __syncthreads();
	const float mu = mean;
	for (int i = threadIdx.x; i < cols; i += blockDim.x) gr[i] -= mu;
}
```
  Wrapper `bool gradient_centralize(float* g, int rows, int cols)` (grid=rows,
  256 threads); guard rows/cols>0.

- [ ] **Step 2.4: Decl + stub** in gpu_kernels.h.

- [ ] **Step 2.5: Build + test PASS + install.**

- [ ] **Step 2.6: Commit** glades-ml.

- [ ] **Step 2.7: Wire trainer.** `bool gradCentralize;` config (off); flag
  `--grad-centralize`. Apply to the weight-grad matrices (dWq/k/v/o[l],
  treated as [m, dModel]) AFTER backward, BEFORE adam. NOTE: under
  `cfg.bf16Grads` the weight grads are BF16 (`W.dWq_bf[l]`) — GC needs FP32;
  either (a) restrict GC to dgamma/dbeta + FP32-grad path for the prototype,
  or (b) add a BF16-in/BF16-out GC variant. Prototype = FP32 path + dgamma/
  dbeta only; document the bf16 extension as follow-up. Graphs auto-disable
  not needed (no host download). 

- [ ] **Step 2.8: run.sh passthrough** `--grad-centralize`.

- [ ] **Step 2.9: Build + smoke + commit trainer** (off = bit-identical; on =
  300-step A/B/C within rerun noise).

- [ ] **Step 2.10: Validation gate** + document.

**Cost:** ~0.5 day + 16h. **Risk:** LOW. **Default off.** Stacks with AGC.

---

## Phase 3 — Bounded ReLN backward (at the source)

**Rationale:** the dgamma/dbeta overflow ORIGINATES in `chiron_reln_backward` →
`layernorm_backward`. Bounding it post-hoc (clamps) is containment; bounding
the per-token contribution INSIDE the reduction cures it at the source and
removes the need for downstream clamps. Two sub-options — prototype 3a first.

**3a — clamp reconstructed q_in row-norm before the dgamma reduction.** The
dgamma = Σ_t dq_out[t]·q̂[t]; huge q̂ (BF16-drift in the reversible recompute)
inflates it. Re-normalize q̂ per row to its ReLN-implied unit scale before the
reduction.

**3b — clamp the per-token dq_out·q̂ product** inside a fused
`layernorm_backward_bounded` variant before summation.

**Files:** `gpu_chiron.cu` (`chiron_reln_backward` variant), the
`layernorm_backward` kernel (likely `gpu_kernels.cu`), headers, test, trainer.

- [ ] **Step 3.1: Locate `layernorm_backward`** kernel def (`grep -n
  "layernorm_backward" gpu_kernels.cu gpu_chiron.cu`); read the dgamma/dbeta
  reduction. Confirm whether q̂ (q_in) is re-derived or passed.

- [ ] **Step 3.2: Failing test** `CHIRONRelnBackwardBoundedTest`: construct
  dq_out/q_in/stats where one token's q̂ row is huge (1e10) → bounded variant's
  dgamma stays finite and ≈ the variant with that row renormalized; healthy
  inputs → bit-identical to `chiron_reln_backward`. Register `chiron-relnbound`.

- [ ] **Step 3.3: Run, verify FAIL.**

- [ ] **Step 3.4: Implement `chiron_reln_backward_bounded(... , float qRowMax)`**
  — same signature as `chiron_reln_backward` + a `qRowMax` param. Add a pre-pass
  (reuse `row_rms_clamp` on a copy of q_in, OR fold a per-row clamp into the
  layernorm_backward kernel's q̂ load). Prototype 3a: clamp q_in rows to qRowMax
  RMS via the existing `row_rms_clamp` before `layernorm_backward`. (This is the
  minimal, reuses a validated kernel; 3b fused variant is a follow-up if 3a's
  extra pass costs too much.) When qRowMax<=0, delegate to the plain
  `chiron_reln_backward` (bit-identical).

- [ ] **Step 3.5: Decl + stub.** Build + test PASS + install. Commit glades-ml.

- [ ] **Step 3.6: Wire trainer.** `float relnBackwardQClamp;` (off); flag
  `--reln-bwd-qclamp F`. In the backward layer loop, replace
  `chiron_reln_backward(...)` with `chiron_reln_backward_bounded(..., cfg.relnBackwardQClamp)`
  when on. (This site is the same loop as the existing `--dq-layer-clamp`;
  ensure they compose — reln-bwd-qclamp bounds q̂, dq-layer-clamp bounds dq_out.)

- [ ] **Step 3.7: run.sh passthrough.** Build + smoke + commit trainer.

- [ ] **Step 3.8: Validation gate.** Critically, test WITHOUT the downstream
  gg-clamp (`--grad-group-clamp 0`) to see if the source-bound ALONE prevents
  the overflow — that's the "cure vs contain" question. Document the verdict.

**Cost:** ~1.5 days + 16h. **Risk:** MEDIUM (kernel-internal, FMA-order parity
care per the iter-70/73 drift-class history). **Default off. Highest "cure"
value.**

---

## Phase 4 — Spectral / weight normalization

**Rationale:** constrain the spectral norm (largest singular value σ_max) of
the attention projections Wq/Wk/Wv/Wo so the symplectic forward/backward cannot
amplify — attacking the amplification's source in the weights. Power iteration
(1–2 iters/step) estimates σ_max cheaply with a persistent left/right singular
vector. Cheaper variant: **spectral-aware init** (scale W at init so σ_max≈1),
zero per-step cost — implement this first as the floor.

**Files:** `gpu_kernels.cu` (power-iteration + scale kernels), headers, test,
trainer (`ChironParams` persistent u-buffers, post-adam hook).

- [ ] **Step 4.1: Spectral-init sub-step (cheap floor).** Add `--spectral-init F`
  (target σ_max): after weight init, power-iterate σ_max of each Wq/Wk/Wv/Wo and
  scale W *= F/σ_max once. Kernel: `spectral_norm_estimate(W, rows, cols, u, v, iters)`
  → returns σ_max; then `scale_array`. Test `CHIRONSpectralNormTest`: a matrix
  with known σ_max (diagonal) → estimate within 1% after 10 iters.

- [ ] **Step 4.2: Failing test, run FAIL, implement power-iteration kernel**
  (v = Wᵀu/‖Wᵀu‖; u = Wv/‖Wv‖; σ = uᵀWv; repeat). Build + test PASS + install + commit.

- [ ] **Step 4.3: Per-step spectral normalization (main variant).** Add
  persistent `u` buffers per weight in ChironParams. Flag `--spectral-norm F`:
  after adam each step, power-iterate 1 step (warm-started from persistent u),
  if σ_max > F scale W down to F. Apply to Wq/Wk/Wv/Wo (FP32 or via BF16
  mirror). Test: repeated application keeps σ_max ≤ F.

- [ ] **Step 4.4: Wire trainer** (flags, u-buffer alloc, post-adam hook),
  run.sh passthrough, graphs auto-disable (host σ read). Build + smoke + commit.

- [ ] **Step 4.5: Validation gate** — both `--spectral-init` alone (free) and
  `--spectral-norm` (per-step). Check σ_max trajectory + stability + val.

**Cost:** ~2 days + 16h. **Risk:** MEDIUM (per-step cost; quality impact of
constraining weights — validate NLL doesn't regress). **Default off.**

---

## Phase 5 — SAM (Sharpness-Aware Minimization)

**Rationale:** directly targets the diagnosed fragile/sharp minimum (H2) by
minimizing worst-case loss in a ρ-ball: it seeks FLAT minima, which by
construction don't have explosive gradients. The most principled cure, but ~2×
fwd/bwd cost and an invasive training-loop change — hence last.

**Files:** `glades-trainer/trainer/chiron_main.cpp` (training-loop
restructure), a weight-perturb/restore kernel in gpu_kernels.cu, headers, test.

- [ ] **Step 5.1: Perturb/restore kernel + test.** `sam_perturb(W, g, rho,
  gNormInv)`: W += rho·g·gNormInv (ascent to the ρ-ball boundary). Test:
  W'=W+rho·g/‖g‖ element-wise (rtol 1e-6). Restore = subtract the same.
  Register `chiron-sam`. Build + test PASS + commit.

- [ ] **Step 5.2: Global grad-norm for SAM.** Reuse `compute_grad_norm_sq` to
  get ‖g‖ across all params for the normalization.

- [ ] **Step 5.3: Training-loop restructure.** Flag `--sam-rho F` (off). Per
  optimizer step when on: (1) fwd+bwd → grads g, ‖g‖; (2) perturb ALL weights
  W += rho·g/‖g‖ (save nothing — restore by exact inverse); (3) fwd+bwd at
  perturbed point → SAM grads g'; (4) restore W -= rho·g/‖g‖; (5) adam with g'.
  Memory: no weight copy needed (perturbation is invertible); compute: 2×
  fwd/bwd. Interacts with grad-accum: apply SAM per micro-step or per accum
  window — prototype = per accum window (perturb once on accumulated g).
  CAUTION: the symplectic forward + inverse-recompute backward must be re-run
  cleanly at the perturbed weights; verify state buffers are reset between the
  two passes (this is the main implementation risk).

- [ ] **Step 5.4: run.sh passthrough** `--sam-rho`. Build + smoke (5 steps;
  confirm 2× backward calls, finite losses, restore exact via off-vs-on-rho-0).

- [ ] **Step 5.5: Validation gate** with `--sam-rho 0.05` (typical) — and
  measure the wall cost (expect ~+80–100%). PASS criteria weigh stability +
  NLL against the doubled cost; SAM ships only if it cures where cheaper phases
  don't. Document.

**Cost:** ~3 days + validation. **Risk:** HIGH (invasive loop change, symplectic
double-pass correctness, 2× cost). **Default off. Evaluate last.**

---

## Sequencing & decision logic

1. **Phase 1 (AGC)** first — closest to validated, cheapest, likely the
   immediate win (selective clipping vs the chronic-firing fixed clamp).
2. **Phase 2 (GC)** — cheap landscape-flattening, stacks with AGC.
3. **Phase 3 (ReLN-bound)** — first real "cure" attempt; test it standalone
   (gg-clamp off) to see if source-bounding removes the need for clipping.
4. **Phase 4 (spectral)** — init-only variant is free; per-step if needed.
5. **Phase 5 (SAM)** — only if 1–4 leave residual instability; its 2× cost
   must be justified by curing what the others can't.

After each phase's gate, update
`research/QSIDE_INSTABILITY_INVESTIGATION_2026_06_14.md` with the verdict.
The goal state: a recipe that reaches ≥5B tokens with zero skips and NO
chronic clamp firing (i.e., the instability is cured, not just bounded).

All phases default-off; the current production flagship and the in-flight 5B
gg-clamp run are unaffected. GPU validation for every phase is queued behind
the in-flight run (resume-invalid → fresh runs only).
