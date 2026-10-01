# CHIRON PACT — E0–E2 Implementation Plan (kernels, trainer flags, calibration)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement PACT (deterministic gated anti-cancellation penalty, design
`docs/superpowers/specs/2026-07-04-chiron-pact-anti-cancellation-design.md`) as default-off
CUDA kernels + CPU refs + unit suite in glades-ml and `--pact-coef` plumbing in glades-trainer,
pass E0 (bit-parity off) and E1 (units), then run E2 (λ calibration at production shape).

**Architecture:** Training-only. Forward: the PIED masked commit kernel gains a fused variant
that additionally accumulates the damped-sum/mass fields A, M (two [T×m] FP32 buffers, zeroed
per µstep). Backward: the PIED dual dy hand-off gains a fused variant that adds the clamped,
gated, A-preserving PACT field g to the increment-branch adjoint. A per-step damping table D
[L×m] comes from rot_phi; a per-channel detached scale σ̂ [m] is EMA'd from column-means of M.
No checkpoint delta, no serving change, no new GEMMs.

**Tech Stack:** C++98, CUDA (gpu_chiron.cu/h style, CPU refs in transformer_chiron_ops.h),
unit-tests framework (`ASSERT`, chiron-test.cpp, `test.sh chiron-pact`). Repos:
`~/dev/glades-ml` and `~/dev/glades-trainer`, both branch `chiron4`.

## Global Constraints

- Design formulas (spec §5.2–5.3), per (t,i), profile Y ∈ ℝ^L, D_l > 0:
  `A = Σ_l D_l·u_l`, `M = Σ_l D_l·|u_l|`, `Dsq_i = Σ_l D²_{l,i}`,
  `res_l = u_l − A·D_l/Dsq`, `r = res_l/σ̂_i`, `rc = clamp(r, ±κ)`,
  `χ = clamp01((M²−A²)/(M²+ε_M·Dsq_i·σ̂²_i))` (detached; `--pact-gate 0` ⇒ χ≡1),
  field `g_l = coef·χ·rc/(Dsq_i·σ̂_i)` with `coef = 2λ/(T·m)`,
  penalty value `= (λ/(T·m))·Σ χ·h_κ(r)/Dsq`, `h_κ(r) = r²` if |r|≤κ else `κ(2|r|−κ)`.
- **Two documented deviations from spec §5.1/§5.4** (record in the E-phase research note):
  (1) σ̂_i is the EMA (rate = `cfg.whiscEma`, default 0.05) of `mean_t(M_{t,i}) / D1_i`,
  `D1_i = Σ_l D_{l,i}` — a mean-|u| scale from the already-materialized M buffer instead of a
  Y²-RMS EMA (same detached, scale-free role per F3; avoids per-element atomics; the E2
  field-RMS rule absorbs the constant). σ̂ floored at `PACT_EPS0 = 1e-6`.
  (2) A, M are FP32 (2×128 MiB, fits the ~0.86 GB headroom) — not BF16 — for exactness;
  BF16 is the perf fallback only if E2 VRAM fails.
- **First-touch rule:** σ̂ is initialized (β=1) at the END of the first PACT forward (M final,
  before any backward reads it), then frozen within each µstep and EMA-updated at µstep end.
  Guards the σ̂-floor hazard (tiny σ̂ ⇒ large clamped field).
- Exact identity at `--pact-coef 0` / flag absent: no new kernel dispatched, no buffers
  allocated (E0).
- Key invariants the unit suite MUST assert: `Σ_l D_l·g_l = 0` (unclamped, FP tolerance);
  clamped field = exact gradient of the Huberized penalty (FD check); χ = 0 on sign-coherent
  profiles (incl. one-hot); ρ-independence (synthetic ρ = 45 changes nothing in the field
  given the same increments); p/p_bf16 outputs of the fused commit variant bit-identical to
  `chiron_scfa_axpy2_masked_dual_p`; dy output at λ=0 bit-identical to
  `chiron_incdrop_scale_copy_dual`.
- The PIED mask: the pact field g carries NO η factor; the task branch keeps `alpha·η·src`
  exactly as today (same key/thr/lo/hi).
- C++98 host code. Kernels in `Backend/Machine Learning/Networks/cuda/gpu_chiron.cu`,
  declarations in `gpu_chiron.h` (namespace `glades::gpu`); CPU refs in
  `Backend/Machine Learning/Networks/transformer_chiron_ops.h` (namespace `glades::chiron`,
  header-only inline).
- Build/install discipline: after glades-ml kernel changes `cd ~/dev/glades-ml/build && make
  install`, then REBUILD the trainer (`cd ~/dev/glades-trainer && bash build.sh`) — static link.
- Commits end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- GPU runs: never overwrite `chiron_1B_pied_e4.final`; E2 runs save to fresh scratch dirs under
  `database/checkpoints/pact_e2_*` (pre-create the dir — silent save failure otherwise).

---

### Task 1: CPU reference math + pure-CPU unit tests

**Files:**
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/transformer_chiron_ops.h` (append
  a PACT section after the PIED helpers ~line 404)
- Modify: `~/dev/glades-ml/unit-tests/Backend/Machine Learning/chiron-test.h` (declare test fns)
- Modify: `~/dev/glades-ml/unit-tests/Backend/Machine Learning/chiron-test.cpp` (implement)

**Interfaces (produced, used by Tasks 2–4):** in `namespace glades { namespace chiron {`:

```cpp
// PACT (2026-07-04): gated anti-cancellation penalty — CPU references.
// Design: docs/superpowers/specs/2026-07-04-chiron-pact-anti-cancellation-design.md §5.
static const float PACT_EPS0 = 1e-6f;   // sigma floor
static const float PACT_EPSM = 1.0f;    // gate floor multiplier

// Damping from per-layer angles: D[l*m+i] = prod_{l'>=l} cos(thetaMax*tanh(phi[l'][i])).
inline void chiron_pact_damp_ref(const float* const* phi /*L ptrs, each [m]*/,
                                 int L, int m, float thetaMax,
                                 float* D /*[L*m]*/, float* Dsq /*[m]*/, float* D1 /*[m]*/);

// Gate chi for one (t,i): A,M are damped sums; sigma the detached scale.
inline float chiron_pact_chi(float A, float M, float Dsq_i, float sigma_i);

// Huber value/derivative on r with clamp kappa.
inline float chiron_pact_huber(float r, float kappa);      // r^2 or kappa*(2|r|-kappa)
inline float chiron_pact_huber_d(float r, float kappa);    // 2r clamped to ±2kappa ... see note

// The field for one element (layer l, channel i) given profile summaries:
//   g = coef * chi * clamp(res/sigma, ±kappa) / (Dsq * sigma)
inline float chiron_pact_field(float u, float A, float M,
                               float Dl, float Dsq_i, float sigma_i,
                               float coef, float kappa, int gateOn);

// Whole-profile helpers for tests: penalty value for one (t,i) profile
// (chi * sum_l h_kappa(res_l/sigma) / Dsq, caller scales by lambda/(T*m)),
// and the field vector.
inline double chiron_pact_profile_value(const float* u /*[L]*/, const float* Dcol /*[L]*/,
                                        float Dsq_i, float sigma_i, float kappa, int gateOn);
inline void   chiron_pact_profile_field(const float* u, const float* Dcol,
                                        float Dsq_i, float sigma_i,
                                        float coef, float kappa, int gateOn,
                                        float* g /*[L]*/);
```

Note on `chiron_pact_huber_d`: define `h_κ(r) = r²` for |r|≤κ, `κ(2|r|−κ)` beyond; then
`dh/dr = 2·clamp(r,±κ)` — the field uses `clamp(r,±κ)` (the ½ is absorbed: field g is the
gradient of ½·coef·χ·σ̂²·ŵ·h(r) summed appropriately; the FD test below pins the exact
correspondence: `g_l = ∂/∂u_l [ (coef/2)·χ_detached·Σ_l h_κ(res_l/σ̂)/Dsq ]`).

- [ ] **Step 1: Implement the refs** exactly per the formulas (χ detached: `profile_field`
  computes χ from the passed A,M then treats it as a constant; `res_l = u_l − A·D_l/Dsq` with
  A recomputed inside from u and Dcol — A IS a function of u in the derivative, and the exact
  cancellation of the second term (spec §5.3) is what makes the simple form correct; the FD
  test verifies this).
- [ ] **Step 2: Write the CPU tests** in chiron-test.cpp (register in Task 5):

```cpp
void CHIRONPactRefMathTest()
{
    // Toy gate goldens (M0-probe hand values): u={+1,-1},D=1,L=2: chi=2/3 at sigma=1.
    // one-hot u={c,0}: chi==0 exactly. pure pair u={+1,-1}: A=0.
    // dead-zone: u={1e-9,-1e-9}, sigma=1: chi ~ 0 (< 1e-12).
    // Orthogonality: random profiles (L=24, seeded LCG), random D in [0.945,1]:
    //   sum_l D_l*g_l == 0 to 1e-5 relative (UNCLAMPED: kappa=1e9).
    // Huber/FD: for 20 random profiles and kappa=4, perturb each u_l by ±1e-3:
    //   numeric d(value)/du_l vs 2/coef... — assert |g_fd - g_ref| <= 1e-4*max(|g|).
    //   (value uses chi FROZEN at the unperturbed profile — chi is detached.)
    // rho-independence: scale a synthetic "p magnitude" by 45 — no input of the
    //   field changes (structural: assert field depends only on (u,D,sigma) inputs
    //   by recomputing with an unrelated rho variable — documentation assert).
    // Scale-freedom: u->c*u with sigma->c*sigma leaves chi identical and g -> g/c... 
    //   assert g(c*u, c*sigma) == g(u,sigma)/1 * (1/c) to 1e-5 rel, c=32.
}
```

  (Write real loops + `ASSERT(msg, pred)` per the house framework — the comment block above is
  the required coverage list, not pseudo-placeholders; every listed assert must exist.)
- [ ] **Step 3: Build + run** the existing chiron suite compiles: `cd unit-tests/build && sh
  .configure.sh cuda` then `cd unit-tests && bash test.sh chiron` (existing tests still green;
  new test registered in Task 5 — for now call `CHIRONPactRefMathTest()` temporarily from the
  `chiron` selector or build-check only).
- [ ] **Step 4: Commit** (`pact: CPU reference math + ref-level unit tests`).

---

### Task 2: D-table + σ̂-update kernels

**Files:**
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/cuda/gpu_chiron.h` (+3 decls,
  after the PIED section ~line 395)
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/cuda/gpu_chiron.cu` (+3 kernels)
- Modify: chiron-test.{h,cpp} (GPU parity tests)

**Interfaces (produced):**

```cpp
// PACT damping table: cos row for one layer (theta = thetaMax * tanh(phi[i])).
bool chiron_pact_cos_row(const float* phi, float thetaMax, float* cosRow,
                         int m, cudaStream_t stream = 0);
// Suffix products over layers: D[l*m+i] = prod_{l'>=l} cosTable[l'*m+i];
// Dsq[i] = sum_l D^2; D1[i] = sum_l D.  One thread per channel (L<=64 loop).
bool chiron_pact_damp_finalize(const float* cosTable, int L, int m,
                               float* D, float* Dsq, float* D1,
                               cudaStream_t stream = 0);
// sigma EMA from the mass accumulator: colmean = (1/T) sum_t M[t*m+i]; then
// sigma[i] = firstTouch ? colmean/D1[i] : (1-beta)*sigma[i] + beta*colmean/D1[i],
// floored at eps0.  One thread per channel, sequential T-loop (deterministic).
bool chiron_pact_sigma_update(const float* Macc, const float* D1, int T, int m,
                              float beta, float eps0, int firstTouch,
                              float* sigma, cudaStream_t stream = 0);
```

- [ ] Step 1: implement kernels (grid-stride over m; damp_finalize/sigma_update: one thread per
  channel, fixed-order loops — deterministic).
- [ ] Step 2: GPU-vs-CPU parity test `CHIRONPactDampSigmaParityTest()` — random phi (L=24,
  m=64), thetaMax=0.07: D/Dsq/D1 match `chiron_pact_damp_ref` to 1e-6 rel; sigma_update
  matches a CPU loop (both firstTouch=1 and an EMA step) to 1e-6.
- [ ] Step 3: build lib + tests green (temporary selector as in Task 1).
- [ ] Step 4: Commit (`pact: damp-table + sigma kernels + parity tests`).

---

### Task 3: Fused forward commit variant (A/M accumulation)

**Files:** gpu_chiron.h / gpu_chiron.cu / chiron-test.{h,cpp}

**Interfaces (produced):**

```cpp
// PACT variant of the masked dual-p commit: identical p_fp32/p_bf16 writes
// (SAME eta and SR sequence — bit-identical outputs), plus per-element
// A[idx] += Drow[i]*u, M[idx] += Drow[i]*|u| with u = a[idx]+b[idx] (CLEAN,
// pre-mask), i = idx % m.  Forward (alpha=+1) only; the inverse walk keeps
// calling the non-pact kernel.
bool chiron_scfa_axpy2_masked_dual_p_pact(float* p_fp32, unsigned short* p_bf16,
                                          float alpha,
                                          const float* a, const float* b,
                                          int n, int m,
                                          unsigned int key, unsigned int thr,
                                          float lo, float hi,
                                          unsigned int srBaseSeed,
                                          unsigned int srStepIdx,
                                          const float* Drow,
                                          float* Aacc, float* Macc,
                                          cudaStream_t stream = 0);
```

- [ ] Step 1: implement by COPYING the existing `chiron_scfa_axpy2_masked_dual_p` kernel body
  verbatim (p-path statements untouched, same launch config) and appending the three
  accumulation lines. Read the existing kernel first; do not restructure it.
- [ ] Step 2: tests `CHIRONPactCommitParityTest()`: (i) p_fp32 and p_bf16 outputs bit-identical
  (memcmp) to the existing kernel at identical (key,thr,lo,hi,seed,ctr) on random inputs
  (n=8192, m=64); (ii) A/M match the CPU accumulation (using `chiron_pied_eta`-independent
  clean u) to 1e-6; (iii) zeroed-A/M + two sequential layer calls accumulate correctly.
- [ ] Step 3: build + green; Step 4: Commit (`pact: fused masked commit variant + parity`).

---

### Task 4: Fused backward field variant (dy = η⊙dp + g)

**Files:** gpu_chiron.h / gpu_chiron.cu / chiron-test.{h,cpp}

**Interfaces (produced):**

```cpp
// PACT variant of the dual dy hand-off: dst = alpha*eta*src + g, g the clamped
// gated field (Global Constraints formulas); dst_bf = BF16-RN of dst.  stats4
// (monitor-only, float atomics): {sum chi*h_kappa(r)/Dsq, clamp count,
// sum g^2, sum (alpha*eta*src)^2}.  At coef==0 the kernel must not be called
// (trainer dispatch guard); a defensive early-identical path is still required:
// if coef==0, dst/dst_bf outputs must be bit-identical to
// chiron_incdrop_scale_copy_dual.
bool chiron_incdrop_scale_copy_dual_pact(float* dst, unsigned short* dst_bf,
                                         float alpha, const float* src,
                                         const float* a, const float* b,
                                         const float* Aacc, const float* Macc,
                                         const float* Drow, const float* Dsq,
                                         const float* sigma,
                                         float coef, float kappa,
                                         float epsM, float eps0, int gateOn,
                                         int n, int m,
                                         unsigned int key, unsigned int thr,
                                         float lo, float hi,
                                         float* stats4,
                                         cudaStream_t stream = 0);
```

Per element (i = idx % m): `eta` via the same device hash as the existing kernel; task branch
`dyt = alpha*eta*src[idx]` textually identical; `u = a[idx]+b[idx]`; `res = u −
Aacc[idx]*Drow[i]/Dsq[i]`; `sg = max(sigma[i], eps0)`; `r = res/sg`; `rc = clamp(r,±kappa)`;
`chi` per formula (or 1.0f if !gateOn); `g = coef*chi*rc/(Dsq[i]*sg)`; `out = dyt + g`;
BF16-RN mirror as in the existing kernel. stats4 accumulated via per-block shared-memory
reduction then one atomicAdd per block per stat (monitor-only; document nondeterministic
rounding).

- [ ] Step 1: implement (copy existing `chiron_incdrop_scale_copy_dual` body for the task
  branch + mirror; append field math + stats).
- [ ] Step 2: tests `CHIRONPactFieldParityTest()`:
  (i) coef=0 ⇒ dst/dst_bf bit-identical to the existing dual kernel;
  (ii) GPU field vs `chiron_pact_profile_field` CPU ref across a synthetic (L=24, T=32, m=64)
  problem — build A/M on CPU from random u per layer, call the GPU kernel per layer, compare
  summed |g_gpu − g_cpu| ≤ 1e-5 rel;
  (iii) orthogonality in vivo: Σ_l Drow_l·g_l per (t,i) ≤ 1e-4·‖g‖ (kappa=1e9);
  (iv) clamp path: kappa=0.5 forces clamps; field values match CPU ref; stats4 clamp count
  exact; (v) gate-off (`gateOn=0`) matches χ≡1 CPU ref;
  (vi) ρ=45: scale a dummy p-buffer by 45 — not an input; assert field bytes unchanged
  (structural documentation assert: recompute with identical inputs).
- [ ] Step 3: build + green; Step 4: Commit (`pact: fused dy field variant + parity`).

---

### Task 5: Suite registration + full green + install

**Files:** `unit-tests/main.cpp` (selector `chiron-pact` / `pact` after the chiron-pied block
~line 293), `unit-tests/test.sh` (add to the selector list), `CLAUDE.md` (test-name list:
append `chiron-pact`), chiron-test.h declarations.

- [ ] Step 1: register all four test fns under `chiron-pact`; remove any temporary hooks.
- [ ] Step 2: `cd unit-tests && bash test.sh chiron-pact` — all green; then `bash test.sh
  chiron-pied` and `bash test.sh chiron-whisc` — regression green.
- [ ] Step 3: `cd ~/dev/glades-ml/build && make install`.
- [ ] Step 4: Commit (`pact: chiron-pact unit suite registered; suite green`).

---

### Task 6: Trainer plumbing + E0

**Files:** `~/dev/glades-trainer/trainer/chiron_main.cpp`

Config fields + defaults (`float pactCoef; int pactGate; float pactClamp;` = 0.0f/1/4.0f);
flags `--pact-coef`, `--pact-gate`, `--pact-clamp` in the parse chain (~line 2177 area, next to
--inc-dropout); validation: `pactCoef > 0` hard-errors unless
(`scfa && incDropout>0 && bf16ResidualP && iter70FusedAxpy2DualP && layerDropPMax==0 &&
sasAlpha out of gating range && sfa inactive`) — mirror the existing PIED/interlock error style.

Scratch (allocate iff pactCoef>0): `pact_A`, `pact_M` GpuBuffer<float> [T·m]; `pact_cos`,
`pact_D` [L·m]; `pact_Dsq`, `pact_D1`, `pact_sigma` [m]; `pact_stats` [4];
host flags `pactSigmaTouched`, step-cadence log state.

Wiring (all guarded `cfg.pactCoef > 0 && isTraining`):
1. Per optimizer step (µstep 0), before the layer loop: build D
   (`chiron_pact_cos_row` per layer from `W.rot_phi[l]` + `chiron_pact_damp_finalize`).
2. At layer-loop start (l==0): `s.pact_A.zero(); s.pact_M.zero();`.
3. Commit routing in `scfa_attention_forward` (the `s.piedActive` fused branch, ~line 9316):
   when `cfg.pactCoef>0 && !invert && s.piedActive && fused-path` → call
   `chiron_scfa_axpy2_masked_dual_p_pact(..., s.pact_D.data()+(size_t)l*m, s.pact_A.data(),
   s.pact_M.data())`; the `invert` call sites stay on the non-pact kernel.
4. End of forward (after the layer loop, at the pact-m0 finalize site): if
   `!pactSigmaTouched` → `chiron_pact_sigma_update(..., firstTouch=1)`,
   set `pactSigmaTouched=true`.
5. Backward dy hand-off (in `scfa_attention_backward`, the dual copy site ~line 10090 region):
   route to `chiron_incdrop_scale_copy_dual_pact` with `coef = 2.0f*cfg.pactCoef/((float)T*m)`,
   `kappa=cfg.pactClamp`, `epsM=1.0f`, `eps0=1e-6f`, `gateOn=cfg.pactGate`, Drow for layer l,
   stats into `s.pact_stats`. VERIFY at the call site that `W.scfa_ypar/scfa_yperp` still hold
   layer-l clean increments (they are recomputed by the inverse walk before this seam — same
   guarantee the inverse-commit uses).
6. µstep end (after backward returns in the accum loop): `chiron_pact_sigma_update(...,
   firstTouch=0, beta=cfg.whiscEma)`.
7. `[pact]` monitor at the normalLogStep cadence (next to `[sorc]`, ~line 18649 region):
   download stats4 → log `value=λ/(Tm)·s0`, `field/dy=sqrt(s2/s3)`,
   `clampRate=s1/(L·Tm·accum)`; zero stats at each log. fflush.

- [ ] Step 1: implement all of the above; build (`make install` already done in Task 5; then
  `bash build.sh`).
- [ ] Step 2 (E0 gate): rerun the M0 eval protocol WITHOUT any pact flag (same command as the
  M0 full8 run, probe env vars UNSET) — the three val lines must reproduce exactly
  `nll=1.0891 / 1.1174 / 1.1909` (8-batch, from `logs/pact_m0_full8_20260704_0435.log`; the
  eval forward is bit-reproducible — proven by the M0 determinism check). Any drift ⇒ STOP:
  flag-off path touched.
- [ ] Step 3 (smoke-on): 30-step tiny-shape or short flagship-recipe run (`--steps 30020` NOT
  from the flagship dir — use a fresh scratch save; or small-shape `run.sh` mode) with
  `--pact-coef 3e-3` — assert `[pact]` lines appear, value finite and > 0, no crash, no skip.
- [ ] Step 4: Commit (`pact: trainer flags + plumbing + [pact] monitor; E0 parity verified`).

---

### Task 7: E2 calibration runs + records

- [ ] Step 1 (control): fresh flagship-recipe run, seed 1337, `--steps 1000`, save to
  `database/checkpoints/pact_e2_ctl/` (pre-create), flag-off. Record: tok/s, val@{100..1000},
  ‖g‖ census, 0 skips.
- [ ] Step 2 (treatment λ=3e-3): identical + `--pact-coef 3e-3`, save `pact_e2_a/`. Read
  `[pact]`: field/dy ratio trajectory, penalty value trajectory, clampRate.
- [ ] Step 3 (decide λ): target ratio 0.03–0.05 at step ~1k. Field ∝ λ ⇒ scale linearly; if a
  second point is needed run 300 steps at the adjusted λ. Pre-registered grid {3e-3, 1e-2};
  pick λ*.
- [ ] Step 4 (E2 gates): wall ≤ +2% vs control (tok/s); penalty value decreasing after ~200
  steps; clampRate < 1%; val@≤1000 gap vs control ≥ −0.00/+0.01 (no immediate tax beyond
  jitter); 0 skips; ‖g‖ max ≤ 1.1× control. **Kill:** ratio uncontrollable within the grid,
  wall > 2%, or val gap > +0.02 at all grid points.
- [ ] Step 5: write `research/CHIRON_PACT_E0_E2_2026_07_04.md` (house gate-note style): E0
  parity evidence, E1 suite summary (assert counts), E2 tables (ratio/penalty/clamp/wall/val),
  λ* decision, the two documented deviations (σ̂ from M; FP32 accumulators), and the E3
  protocol reminder (matched 2500-step pair, fresh same-binary baseline, ~7 GPU-hr — next
  gate). Update spec §13 (E0/E1/E2 bullets with results). Update memory
  (`pact_anti_cancellation_design.md` + MEMORY.md hook).
- [ ] Step 6: Commit both repos.

---

## Self-review

- Spec coverage: §13 prototype build (kernels, flags, no ckpt delta) → Tasks 1–6; E0 → Task 6
  Step 2; E1 (units incl. FD, orthogonality, Huber-consistency, gate goldens, determinism,
  coef-0 parity) → Tasks 1–4 tests; E2 (λ calibration, field-RMS rule, tax pricing, perf) →
  Task 7. E3/E4 intentionally excluded (next gate; E4 owner-gated).
- Deviations documented: σ̂ definition (from M), FP32 accumulators, penalty value not added to
  the printed CE loss (logged as `[pact]` — spec §5.2 "logged separately").
- Type consistency: buffer names `pact_A/pact_M/pact_D/pact_Dsq/pact_D1/pact_sigma/pact_stats`
  and kernel names `chiron_pact_cos_row / chiron_pact_damp_finalize / chiron_pact_sigma_update
  / chiron_scfa_axpy2_masked_dual_p_pact / chiron_incdrop_scale_copy_dual_pact` are used
  identically across Tasks 2–6.
