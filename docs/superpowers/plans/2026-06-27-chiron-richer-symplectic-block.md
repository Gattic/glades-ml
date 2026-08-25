# CHIRON Richer Symplectic Block (OBSD) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-layer, reversible, ReZero-gated, mass-preconditioned nonlinear drift
`q += scale·a_l ⊙ tanh(M_l⁻¹ ⊙ N(p) + b_l)` to the CHIRON block so attention composes across
depth through `q`, while preserving exact reversibility / O(1) activation memory and matching the
flagship bit-for-bit at init.

**Architecture:** Diagnose/prototype-first. OBSD is the dormant `--fuse-attn-per-layer` code path
with three kernels swapped (linear reln → tanh drift; scalar α → per-channel `a_l` init 0;
reanchor on the drift backward). The backward reuses the shipped `chiron_reln_backward_reanchor`
(feed it `du = dq_out⊙a⊙φ'(u)`; it returns `dp`, `dγ_p`, `dβ_p` in one call). Build kernels +
CPU references + GPU-parity/finite-difference unit tests first (glades-ml), then wire into the
trainer behind `--per-layer-drift` (default off), gate at small shape (E0/E1/E2), then run the
production sweep + quality gate (E3/E4).

**Tech Stack:** C++98 + CUDA. glades-ml library (`Backend/Machine Learning/Networks/cuda/gpu_chiron.{h,cu}`,
`transformer_chiron_ops.h`), glades-ml unit tests (`unit-tests/Backend/Machine Learning/chiron-test.{h,cpp}`),
glades-trainer standalone trainer (`trainer/chiron_main.cpp`, `run.sh`, `runner.sh`).

**Spec:** `docs/superpowers/specs/2026-06-27-chiron-richer-symplectic-block-design.md`

---

## Repository layout & build

- **Library code:** `/home/robert/dev/glades-ml/Backend/Machine Learning/Networks/`.
  - Build: `cd /home/robert/dev/glades-ml && sh .configure.sh cuda`.
  - **Build gotcha (from the reanchor arc):** the trainer links the **installed** glades library.
    After ANY glades-ml library change, run `cd /home/robert/dev/glades-ml/build && make install`
    before rebuilding the trainer.
- **Unit tests:** `/home/robert/dev/glades-ml/unit-tests/`.
  - Build: `cd unit-tests/build && sh .configure.sh cuda`.
  - Run chiron suite: `cd unit-tests && bash test.sh nnall` (the chiron cases run inside it).
- **Trainer:** `/home/robert/dev/glades-trainer/`.
  - Build: `cd /home/robert/dev/glades-trainer && cmake --build build --target glades_chiron_train -j"$(nproc)"`
    (or `sh run.sh ...` which re-rsyncs headers and builds — preferred; see the header-sync gotcha).
  - **Header-sync gotcha:** `glades-trainer/include/Backend/Machine Learning/` holds LOCAL COPIES
    of glades-ml headers. `run.sh` re-rsyncs them automatically; a raw `cmake --build` may not.
    After changing `gpu_chiron.h`, run `sh run.sh` once (or rsync) so the trainer sees the new decl.

## Reference (verified) facts the plan relies on

- Dormant forward fuse kernel: `chiron_reln_axpy_into_q(p,q,stats,gamma,beta,alpha,T,m,eps)` →
  `q[i] += alpha·(gamma[i]·(p[i]−μ)/σ + beta[i])` (`gpu_chiron.cu:881`, kernel `…_rows` at 822).
- Reanchor backward: `chiron_reln_backward_reanchor(dq_out,q_in,gamma,T,m,eps,dq_in,dgamma,dbeta,scratch)`
  re-derives `(mean,invStd)` from `q_in` itself; `scratch` is `2*T` floats (`gpu_chiron.h:234`).
  For the affine `out = gamma⊙x̂ + beta`: `dq_in` = grad to `q_in`, `dgamma = Σ_t dq_out·x̂`,
  `dbeta = Σ_t dq_out`.
- CPU references live in namespace `glades::chiron` in `transformer_chiron_ops.h`
  (`reln_forward_row` at :115, `reln_inverse_row` at :158).
- Trainer forward fuse sites (mirror these): SCFA branch `chiron_main.cpp:12708–12726`, bf16w branch
  `12848–12854`, standard branch `12955+`; the flag is `cfg.fuseAttnPerLayer`. Backward/inverse
  sites are near `4296`, `7103`, `14374` (search `cfg.fuseAttnPerLayer`).
- Flagship recipe runs `--no-fuse-attn --fuse-attn-reln` (`run.sh:499`) ⇒ `fuseAttnPerLayer=false`.
  OBSD adds an independent flag `--per-layer-drift` so it composes with the production recipe.
- Per-layer params `gamma_p[l], beta_p[l] ∈ ℝ^m` (init ones/zeros) already exist with Adam state
  and a checkpoint bit. OBSD reuses them as `M_l⁻¹` and `b_l`.

## File structure

| File | Responsibility | New/Modify |
|---|---|---|
| `transformer_chiron_ops.h` | CPU reference: `drift_into_q_row/_inverse`, `drift_backward_row` | Modify |
| `cuda/gpu_chiron.h` | Decls: `chiron_drift_into_q`, `chiron_drift_backward` (+ CPU-stub fallbacks) | Modify |
| `cuda/gpu_chiron.cu` | Kernels: `chiron_drift_into_q_rows`, `chiron_drift_pre_backward_rows`, `chiron_col_reduce_rows` + wrappers | Modify |
| `unit-tests/.../chiron-test.h` | Declare new test fns | Modify |
| `unit-tests/.../chiron-test.cpp` | `CHIRONDriftCpuGpuParityTest`, `CHIRONDriftReversibilityTest`, `CHIRONDriftGradCheckTest` | Modify |
| `unit-tests/main.cpp` | Register the 3 new tests | Modify |
| trainer `chiron_main.cpp` | `a_drift[l]` param (alloc/Adam/checkpoint), `--per-layer-drift`/`--drift-warmup`, fwd/inv/bwd wiring, budget log | Modify |
| trainer `run.sh` | Pass-through for the new flags (if it whitelists flags) | Modify |
| glades-ml `research/CHIRON_OBSD_RESULT_2026_06_27.md` | E0–E4 findings/runbook | New |

---

## Task 1: CPU reference for the drift (forward / inverse / backward)

**Files:**
- Modify: `Backend/Machine Learning/Networks/transformer_chiron_ops.h` (add after `reln_inverse` ~line 193)
- Test: `unit-tests/Backend/Machine Learning/chiron-test.cpp` (+ decl in `chiron-test.h`, register in `main.cpp`)

- [ ] **Step 1: Write the failing finite-difference grad-check test**

In `chiron-test.cpp`, add (and declare `void CHIRONDriftGradCheckTest();` in `chiron-test.h`,
register it in `unit-tests/main.cpp` next to the other `CHIRON*Test()` calls):

```cpp
// Finite-difference check of the CPU drift backward against the CPU forward.
// Objective J = sum_{t,i} dq_out[t,i] * q_out[t,i], q_out = q_in + drift(p).
void CHIRONDriftGradCheckTest()
{
	const unsigned int T = 4, m = 8;
	const float eps = 1e-4f, scale = 1.0f;
	std::vector<float> p(T*m), a(m), gp(m), bp(m), dq(T*m);
	for (unsigned i=0;i<T*m;++i){ p[i]=0.3f*sinf(0.7f*i+1.f); dq[i]=0.2f*cosf(0.3f*i); }
	for (unsigned i=0;i<m;++i){ a[i]=0.5f+0.1f*i; gp[i]=1.0f+0.05f*i; bp[i]=0.02f*i; }

	// Analytic grads.
	std::vector<float> da(m,0.f), dgp(m,0.f), dbp(m,0.f), dp(T*m,0.f);
	glades::chiron::drift_backward(&dq[0], &p[0], &a[0], &gp[0], &bp[0], scale,
	                               T, m, eps, &dp[0], &da[0], &dgp[0], &dbp[0]);

	// FD helper: J(theta).
	const float h = 1e-3f;
	#define DRIFT_J(PP) ({ std::vector<float> qo((PP).size(),0.f); \
		glades::chiron::drift_into_q(&(PP)[0], &qo[0], &a[0], &gp[0], &bp[0], +1.f, scale, T, m, eps); \
		double J=0.0; for(unsigned k=0;k<T*m;++k) J += (double)dq[k]*qo[k]; J; })

	float maxRelErr = 0.f;
	for (unsigned j=0;j<m;++j){            // check da[j]
		float save=a[j]; a[j]=save+h; double Jp=DRIFT_J(p); a[j]=save-h; double Jm=DRIFT_J(p); a[j]=save;
		float fd=(float)((Jp-Jm)/(2.0*h)); float e=fabsf(fd-da[j])/(1e-3f+fabsf(fd));
		if(e>maxRelErr)maxRelErr=e;
	}
	for (unsigned j=0;j<m;++j){            // check dgp[j]
		float save=gp[j]; gp[j]=save+h; double Jp=DRIFT_J(p); gp[j]=save-h; double Jm=DRIFT_J(p); gp[j]=save;
		float fd=(float)((Jp-Jm)/(2.0*h)); float e=fabsf(fd-dgp[j])/(1e-3f+fabsf(fd));
		if(e>maxRelErr)maxRelErr=e;
	}
	for (unsigned k=0;k<T*m;++k){          // check dp[k]
		float save=p[k]; p[k]=save+h; double Jp=DRIFT_J(p); p[k]=save-h; double Jm=DRIFT_J(p); p[k]=save;
		float fd=(float)((Jp-Jm)/(2.0*h)); float e=fabsf(fd-dp[k])/(1e-3f+fabsf(fd));
		if(e>maxRelErr)maxRelErr=e;
	}
	#undef DRIFT_J
	char msg[128]; snprintf(msg,sizeof(msg),"CHIRON drift backward FD grad-check (maxRelErr=%.4f)",maxRelErr);
	ASSERT(msg, maxRelErr < 2e-2f);
}
```

- [ ] **Step 2: Run it to verify it fails (compile error — functions undefined)**

```bash
cd /home/robert/dev/glades-ml/unit-tests/build && sh .configure.sh cuda 2>&1 | tail -5
```
Expected: compile failure — `glades::chiron::drift_into_q` / `drift_backward` undeclared.

- [ ] **Step 3: Implement the CPU forward + inverse drift**

In `transformer_chiron_ops.h`, inside `namespace glades { namespace chiron {`, after `reln_inverse`:

```cpp
// OBSD per-layer drift (CPU reference).  Forward (sign=+1): q += scale·a ⊙ tanh(gamma·x̂ + beta),
// x̂ = (p−μ)/σ, μ,σ the per-row mean/std of p.  Inverse (sign=−1): subtract the same term
// (recompute x̂ from p, which is untouched by the drift).  Parameter-free normalize: μ,σ carry
// no learnable affine; gamma=M⁻¹ and beta act AFTER the normalize, INSIDE the tanh.
inline void drift_into_q_row(const float* p, float* q, const float* a,
                             const float* gamma, const float* beta,
                             float sign, float scale, unsigned int m, float eps)
{
	double sum=0.0; for (unsigned i=0;i<m;++i) sum += p[i];
	const float mu = (float)(sum/(double)m);
	double vs=0.0; for (unsigned i=0;i<m;++i){ float d=p[i]-mu; vs += (double)d*d; }
	const float sigma = sqrtf((float)(vs/(double)m) + eps);
	const float inv = 1.0f/sigma;
	for (unsigned i=0;i<m;++i){
		float xhat = (p[i]-mu)*inv;
		float u = gamma[i]*xhat + beta[i];
		q[i] += sign*scale*a[i]*tanhf(u);
	}
}

inline void drift_into_q(const float* p, float* q, const float* a,
                         const float* gamma, const float* beta,
                         float sign, float scale, unsigned int T, unsigned int m, float eps)
{
	for (unsigned t=0;t<T;++t)
		drift_into_q_row(p + t*m, q + t*m, a, gamma, beta, sign, scale, m, eps);
}
```

- [ ] **Step 4: Implement the CPU backward drift**

Append in the same namespace. `dp/da/dgamma/dbeta` are ACCUMULATED (pre-zero by caller).

```cpp
// OBSD drift backward (CPU reference).  Given dq_out and p (μ,σ re-derived from p — reanchor),
// accumulate da, dgamma(=dM⁻¹), dbeta, and dp.  Chain:
//   u = gamma·x̂ + beta ; s = tanh(u) ; q_out = q_in + scale·a·s
//   da    += Σ_t scale·s·dq_out
//   du     = scale·a·(1−s²)·dq_out ;  dgamma += Σ_t du·x̂ ;  dbeta += Σ_t du
//   dp     = normalize_backward(du as dout, p, gamma=1)   [parameter-free; reanchored stats]
inline void drift_backward(const float* dq_out, const float* p, const float* a,
                           const float* gamma, const float* beta, float scale,
                           unsigned int T, unsigned int m, float eps,
                           float* dp, float* da, float* dgamma, float* dbeta)
{
	for (unsigned t=0;t<T;++t)
	{
		const float* pr = p + t*m; const float* dr = dq_out + t*m; float* dpr = dp + t*m;
		double sum=0.0; for (unsigned i=0;i<m;++i) sum+=pr[i];
		const float mu=(float)(sum/(double)m);
		double vs=0.0; for (unsigned i=0;i<m;++i){ float d=pr[i]-mu; vs+=(double)d*d; }
		const float sigma=sqrtf((float)(vs/(double)m)+eps); const float inv=1.0f/sigma;
		// Per-row du and the layernorm-backward reductions for dp (gamma=1 normalize).
		std::vector<float> xhat(m), du(m);
		double sum_du=0.0, sum_du_xh=0.0;
		for (unsigned i=0;i<m;++i){
			float xh=(pr[i]-mu)*inv; xhat[i]=xh;
			float u=gamma[i]*xh+beta[i]; float s=tanhf(u); float sp=1.0f-s*s;
			da[i]     += scale*s*dr[i];
			float dui = scale*a[i]*sp*dr[i]; du[i]=dui;
			dgamma[i] += dui*xh;            // dM⁻¹
			dbeta[i]  += dui;
			sum_du    += dui; sum_du_xh += (double)dui*xh;
		}
		const float mean_du=(float)(sum_du/(double)m), mean_du_xh=(float)(sum_du_xh/(double)m);
		// Standard LN-backward for a parameter-free normalize (gamma=1):
		//   dp_i = (1/σ)·( du_i − mean(du) − x̂_i·mean(du·x̂) )
		for (unsigned i=0;i<m;++i)
			dpr[i] += inv*( du[i] - mean_du - xhat[i]*mean_du_xh );
	}
}
```
Ensure `#include <vector>` and `<cmath>` are available in `transformer_chiron_ops.h` (add if missing).

- [ ] **Step 5: Run the grad-check test, verify PASS**

```bash
cd /home/robert/dev/glades-ml/unit-tests/build && sh .configure.sh cuda 2>&1 | tail -3
cd /home/robert/dev/glades-ml/unit-tests && bash test.sh nnall 2>&1 | grep -i "drift\|FAIL\|PASS" | head
```
Expected: `CHIRON drift backward FD grad-check` PASSES (maxRelErr < 2e-2).

- [ ] **Step 6: Commit**

```bash
cd /home/robert/dev/glades-ml
git add "Backend/Machine Learning/Networks/transformer_chiron_ops.h" \
        "unit-tests/Backend/Machine Learning/chiron-test.cpp" \
        "unit-tests/Backend/Machine Learning/chiron-test.h" unit-tests/main.cpp
git commit -m "CHIRON OBSD: CPU reference drift fwd/inv/backward + FD grad-check test"
```

---

## Task 2: GPU forward/inverse drift kernel + CPU-GPU parity & reversibility tests

**Files:**
- Modify: `cuda/gpu_chiron.h` (decl + CPU-stub), `cuda/gpu_chiron.cu` (kernel + wrapper)
- Test: `chiron-test.cpp` (`CHIRONDriftCpuGpuParityTest`, `CHIRONDriftReversibilityTest`)

- [ ] **Step 1: Write the failing parity + reversibility tests**

Declare `void CHIRONDriftCpuGpuParityTest(); void CHIRONDriftReversibilityTest();` in
`chiron-test.h`, register in `main.cpp`, and add to `chiron-test.cpp` (model the GpuBuffer usage on
the existing `CHIRONGpuParityTest` at `chiron-test.cpp:1529`):

```cpp
void CHIRONDriftCpuGpuParityTest()
{
	const int T=5, m=16; const float eps=1e-4f, scale=0.9f;
	std::vector<float> p(T*m), q0(T*m), a(m), gp(m), bp(m);
	for (int i=0;i<T*m;++i){ p[i]=0.4f*sinf(0.5f*i); q0[i]=0.1f*i; }
	for (int i=0;i<m;++i){ a[i]=0.3f+0.02f*i; gp[i]=1.0f; bp[i]=0.0f; }
	// CPU.
	std::vector<float> qc=q0;
	glades::chiron::drift_into_q(&p[0], &qc[0], &a[0], &gp[0], &bp[0], +1.f, scale, T, m, eps);
	// GPU.
	glades::gpu::GpuBuffer<float> dP, dQ, dA, dG, dB;
	dP.upload(&p[0], T*m); dQ.upload(&q0[0], T*m); dA.upload(&a[0], m); dG.upload(&gp[0], m); dB.upload(&bp[0], m);
	glades::gpu::chiron_drift_into_q(dP.data(), dQ.data(), dA.data(), dG.data(), dB.data(), +1.f, scale, T, m, eps);
	std::vector<float> qg(T*m); dQ.download(&qg[0], T*m);
	float maxErr=0.f; for(int i=0;i<T*m;++i){ float e=fabsf(qg[i]-qc[i]); if(e>maxErr)maxErr=e; }
	char msg[128]; snprintf(msg,sizeof(msg),"CHIRON drift fwd CPU/GPU parity (maxErr=%.2e)",maxErr);
	ASSERT(msg, maxErr < 1e-4f);
}

void CHIRONDriftReversibilityTest()
{
	const int T=5, m=16; const float eps=1e-4f, scale=1.0f;
	std::vector<float> p(T*m), q0(T*m), a(m), gp(m), bp(m);
	for (int i=0;i<T*m;++i){ p[i]=0.4f*sinf(0.5f*i+0.3f); q0[i]=0.7f*cosf(0.2f*i); }
	for (int i=0;i<m;++i){ a[i]=0.5f; gp[i]=1.1f; bp[i]=0.05f; }
	glades::gpu::GpuBuffer<float> dP, dQ, dA, dG, dB;
	dP.upload(&p[0],T*m); dQ.upload(&q0[0],T*m); dA.upload(&a[0],m); dG.upload(&gp[0],m); dB.upload(&bp[0],m);
	glades::gpu::chiron_drift_into_q(dP.data(),dQ.data(),dA.data(),dG.data(),dB.data(),+1.f,scale,T,m,eps); // forward
	glades::gpu::chiron_drift_into_q(dP.data(),dQ.data(),dA.data(),dG.data(),dB.data(),-1.f,scale,T,m,eps); // inverse
	std::vector<float> qb(T*m); dQ.download(&qb[0],T*m);
	float maxErr=0.f; for(int i=0;i<T*m;++i){ float e=fabsf(qb[i]-q0[i]); if(e>maxErr)maxErr=e; }
	char msg[128]; snprintf(msg,sizeof(msg),"CHIRON drift fwd∘inv reconstructs q (maxErr=%.2e)",maxErr);
	ASSERT(msg, maxErr < 1e-5f);
}
```

- [ ] **Step 2: Run to verify failure (undeclared `chiron_drift_into_q`)**

```bash
cd /home/robert/dev/glades-ml/unit-tests/build && sh .configure.sh cuda 2>&1 | tail -5
```
Expected: compile error — `chiron_drift_into_q` not a member of `glades::gpu`.

- [ ] **Step 3: Declare the wrapper in `gpu_chiron.h`**

Add near `chiron_reln_axpy_into_q` (~line 186) in the CUDA-enabled section:

```cpp
// OBSD per-layer drift.  q += sign·scale·a ⊙ tanh(gamma·(p−μ)/σ + beta), μ,σ per-row of p.
// sign=+1 forward, sign=−1 inverse (recomputes from p, which the drift never modifies).
// gamma = M⁻¹ (init 1), beta = b (init 0), a = ReZero gate (init 0).  No stats output
// (recomputed on inverse/backward — reanchor).
bool chiron_drift_into_q(const float* p, float* q, const float* a,
                         const float* gamma, const float* beta,
                         float sign, float scale, int T, int m, float eps);
```
And add the no-CUDA stub in the `#else` block (~line 684, mirroring the `chiron_reln_axpy_into_q`
stub): `inline bool chiron_drift_into_q(const float*, float*, const float*, const float*, const float*, float, float, int, int, float){ return false; }`.

- [ ] **Step 4: Implement the kernel + wrapper in `gpu_chiron.cu`**

In the anonymous namespace (next to `chiron_reln_axpy_into_q_rows`, ~line 822):

```cpp
__global__ void chiron_drift_into_q_rows(const float* __restrict__ p,
                                         const float* __restrict__ a,
                                         const float* __restrict__ gamma,
                                         const float* __restrict__ beta,
                                         float sign, float scale, float eps, int cols,
                                         float* __restrict__ q)
{
	int row = blockIdx.x;
	const float* xRow = p + (size_t)row * cols;
	float*       qRow = q + (size_t)row * cols;
	extern __shared__ float smem[];
	float* sSumA = smem;
	float* sSumB = smem + (blockDim.x / 32 + 1);
	__shared__ float sMean, sSigma;

	float s = 0.0f;
	for (int i = threadIdx.x; i < cols; i += blockDim.x) s += xRow[i];
	s = blockReduceSum(s, sSumA);
	if (threadIdx.x == 0) sMean = s / (float)cols;
	__syncthreads();
	const float mu = sMean;

	float v = 0.0f;
	for (int i = threadIdx.x; i < cols; i += blockDim.x) { float d = xRow[i]-mu; v += d*d; }
	v = blockReduceSum(v, sSumB);
	if (threadIdx.x == 0) { float var = v/(float)cols + eps; sSigma = sqrtf(var); }
	__syncthreads();
	const float inv_sigma = 1.0f / sSigma;

	for (int i = threadIdx.x; i < cols; i += blockDim.x) {
		float xhat = (xRow[i] - mu) * inv_sigma;
		float u = gamma[i] * xhat + beta[i];
		qRow[i] += sign * scale * a[i] * tanhf(u);
	}
}
```

Outside the namespace (next to the `chiron_reln_axpy_into_q` wrapper, ~line 881):

```cpp
bool chiron_drift_into_q(const float* p, float* q, const float* a,
                         const float* gamma, const float* beta,
                         float sign, float scale, int T, int m, float eps)
{
	if (T <= 0 || m <= 0) return true;
	int block = rowBlockSize(m);
	int smemBytes = (block / 32 + 2) * 2 * sizeof(float);
	chiron_drift_into_q_rows<<<T, block, smemBytes, computeStream()>>>(
	    p, a, gamma, beta, sign, scale, eps, m, q);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}
```

- [ ] **Step 5: Build the library, install, rebuild tests, run**

```bash
cd /home/robert/dev/glades-ml && sh .configure.sh cuda 2>&1 | tail -3
cd /home/robert/dev/glades-ml/build && make install 2>&1 | tail -2
cd /home/robert/dev/glades-ml/unit-tests/build && sh .configure.sh cuda 2>&1 | tail -3
cd /home/robert/dev/glades-ml/unit-tests && bash test.sh nnall 2>&1 | grep -iE "drift|FAIL" | head
```
Expected: both `CHIRON drift fwd CPU/GPU parity` and `CHIRON drift fwd∘inv reconstructs q` PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/robert/dev/glades-ml
git add "Backend/Machine Learning/Networks/cuda/gpu_chiron.cu" \
        "Backend/Machine Learning/Networks/cuda/gpu_chiron.h" \
        "unit-tests/Backend/Machine Learning/chiron-test.cpp" \
        "unit-tests/Backend/Machine Learning/chiron-test.h" unit-tests/main.cpp
git commit -m "CHIRON OBSD: GPU forward/inverse drift kernel + parity & reversibility tests"
```

---

## Task 3: GPU backward drift (reuse reanchor) + GPU-vs-CPU backward parity

**Files:**
- Modify: `cuda/gpu_chiron.h` (decl + stub), `cuda/gpu_chiron.cu` (pre-backward kernel + da column-reduce + wrapper)
- Test: `chiron-test.cpp` (`CHIRONDriftBackwardParityTest`)

The GPU backward = (1) a pre-backward kernel writing `du[T,m]` and `sdq[T,m]=scale·s·dq_out`, plus
recomputing nothing the reanchor needs; (2) a deterministic column-reduce of `sdq` → `da`;
(3) one call to `chiron_reln_backward_reanchor(du, p, gamma_p, …)` → `dp, dgamma_p, dbeta_p`.

- [ ] **Step 1: Write the failing backward-parity test**

Declare `void CHIRONDriftBackwardParityTest();`, register, and add:

```cpp
void CHIRONDriftBackwardParityTest()
{
	const int T=6, m=16; const float eps=1e-4f, scale=0.8f;
	std::vector<float> p(T*m), dq(T*m), a(m), gp(m), bp(m);
	for (int i=0;i<T*m;++i){ p[i]=0.4f*sinf(0.5f*i+0.2f); dq[i]=0.15f*cosf(0.3f*i); }
	for (int i=0;i<m;++i){ a[i]=0.4f+0.01f*i; gp[i]=1.0f+0.03f*i; bp[i]=0.02f*i; }
	// CPU reference.
	std::vector<float> dpc(T*m,0.f), dac(m,0.f), dgc(m,0.f), dbc(m,0.f);
	glades::chiron::drift_backward(&dq[0],&p[0],&a[0],&gp[0],&bp[0],scale,T,m,eps,&dpc[0],&dac[0],&dgc[0],&dbc[0]);
	// GPU.
	glades::gpu::GpuBuffer<float> dP,dDQ,dA,dG,dB,dDP,dDA,dDG,dDB,dScratch;
	dP.upload(&p[0],T*m); dDQ.upload(&dq[0],T*m); dA.upload(&a[0],m); dG.upload(&gp[0],m); dB.upload(&bp[0],m);
	dDP.zero(T*m); dDA.zero(m); dDG.zero(m); dDB.zero(m); dScratch.zero(2*T);
	glades::gpu::chiron_drift_backward(dDQ.data(),dP.data(),dA.data(),dG.data(),dB.data(),scale,T,m,eps,
	                                   dDP.data(),dDA.data(),dDG.data(),dDB.data(),dScratch.data());
	std::vector<float> dpg(T*m),dag(m),dgg(m),dbg(m);
	dDP.download(&dpg[0],T*m); dDA.download(&dag[0],m); dDG.download(&dgg[0],m); dDB.download(&dbg[0],m);
	float me=0.f; for(int i=0;i<T*m;++i) me=fmaxf(me,fabsf(dpg[i]-dpc[i]));
	for(int i=0;i<m;++i){ me=fmaxf(me,fabsf(dag[i]-dac[i])); me=fmaxf(me,fabsf(dgg[i]-dgc[i])); me=fmaxf(me,fabsf(dbg[i]-dbc[i])); }
	char msg[128]; snprintf(msg,sizeof(msg),"CHIRON drift backward CPU/GPU parity (maxErr=%.2e)",me);
	ASSERT(msg, me < 2e-4f);
}
```

- [ ] **Step 2: Run to verify failure (undeclared `chiron_drift_backward`)**

```bash
cd /home/robert/dev/glades-ml/unit-tests/build && sh .configure.sh cuda 2>&1 | tail -5
```
Expected: compile error — `chiron_drift_backward` not in `glades::gpu`.

- [ ] **Step 3: Declare the wrapper in `gpu_chiron.h`**

```cpp
// OBSD drift backward.  Accumulates dp, da, dgamma(=dM⁻¹), dbeta from dq_out and p.
// Internally: pre-backward kernel forms du=scale·a·(1−tanh²(u))·dq_out and sdq=scale·tanh(u)·dq_out
// (μ,σ re-derived from p — reanchor); da=colsum(sdq); then chiron_reln_backward_reanchor(du,p,gamma)
// yields dp, dgamma, dbeta.  scratch: 2*T floats (shared with the reanchor split) + the wrapper
// allocates du/sdq internally via the device allocator.  All grads ACCUMULATE (caller pre-zeros).
bool chiron_drift_backward(const float* dq_out, const float* p, const float* a,
                           const float* gamma, const float* beta, float scale,
                           int T, int m, float eps,
                           float* dp, float* da, float* dgamma, float* dbeta,
                           float* scratch_stats_split);
```
Add the matching no-CUDA stub returning `false` in the `#else` block.

- [ ] **Step 4: Implement the pre-backward kernel + column-reduce + wrapper in `gpu_chiron.cu`**

In the anonymous namespace:

```cpp
// Pre-backward: per row recompute μ,σ,x̂,u; write du and sdq; (da reduced separately).
__global__ void chiron_drift_pre_backward_rows(const float* __restrict__ p,
                                               const float* __restrict__ dq,
                                               const float* __restrict__ a,
                                               const float* __restrict__ gamma,
                                               const float* __restrict__ beta,
                                               float scale, float eps, int cols,
                                               float* __restrict__ du,
                                               float* __restrict__ sdq)
{
	int row = blockIdx.x;
	const float* pr = p + (size_t)row*cols; const float* dr = dq + (size_t)row*cols;
	float* duR = du + (size_t)row*cols; float* sdqR = sdq + (size_t)row*cols;
	extern __shared__ float smem[];
	float* sA = smem; float* sB = smem + (blockDim.x/32 + 1);
	__shared__ float sMean, sSigma;
	float s=0.f; for (int i=threadIdx.x;i<cols;i+=blockDim.x) s+=pr[i];
	s=blockReduceSum(s,sA); if(threadIdx.x==0) sMean=s/(float)cols; __syncthreads();
	const float mu=sMean;
	float v=0.f; for (int i=threadIdx.x;i<cols;i+=blockDim.x){ float d=pr[i]-mu; v+=d*d; }
	v=blockReduceSum(v,sB); if(threadIdx.x==0){ float var=v/(float)cols+eps; sSigma=sqrtf(var);} __syncthreads();
	const float inv=1.0f/sSigma;
	for (int i=threadIdx.x;i<cols;i+=blockDim.x){
		float xhat=(pr[i]-mu)*inv; float u=gamma[i]*xhat+beta[i]; float sa=tanhf(u); float sp=1.0f-sa*sa;
		duR[i]  = scale*a[i]*sp*dr[i];
		sdqR[i] = scale*sa*dr[i];
	}
}

// Deterministic column sum: one block per channel column, loop over rows. out[j] += Σ_t in[t*cols+j].
__global__ void chiron_col_accumulate(const float* __restrict__ in, int rows, int cols,
                                      float* __restrict__ out)
{
	int j = blockIdx.x; if (j>=cols) return;
	float acc=0.f; for (int t=threadIdx.x; t<rows; t+=blockDim.x) acc += in[(size_t)t*cols + j];
	extern __shared__ float red[];
	acc = blockReduceSum(acc, red);
	if (threadIdx.x==0) out[j] += acc;
}
```

Outside the namespace:

```cpp
bool chiron_drift_backward(const float* dq_out, const float* p, const float* a,
                           const float* gamma, const float* beta, float scale,
                           int T, int m, float eps,
                           float* dp, float* da, float* dgamma, float* dbeta,
                           float* scratch_stats_split)
{
	if (T <= 0 || m <= 0) return true;
	// Scratch for du and sdq (T*m each).  Use the existing device scratch allocator.
	glades::gpu::GpuBuffer<float> du, sdq;
	du.zero((size_t)T*m); sdq.zero((size_t)T*m);
	int block = rowBlockSize(m);
	int smemBytes = (block/32 + 2) * 2 * sizeof(float);
	chiron_drift_pre_backward_rows<<<T, block, smemBytes, computeStream()>>>(
	    p, dq_out, a, gamma, beta, scale, eps, m, du.data(), sdq.data());
	GLADES_CUDA_CHECK(cudaGetLastError());
	// da += colsum(sdq).
	int rblock = 256; int rsmem = (rblock/32 + 1) * sizeof(float);
	chiron_col_accumulate<<<m, rblock, rsmem, computeStream()>>>(sdq.data(), T, m, da);
	GLADES_CUDA_CHECK(cudaGetLastError());
	// dp, dgamma, dbeta from one reanchor backward on the affine gamma·x̂+beta (dout = du).
	return chiron_reln_backward_reanchor(du.data(), p, gamma, T, m, eps,
	                                     dp, dgamma, dbeta, scratch_stats_split);
}
```
Note: confirm `GpuBuffer<float>::zero(size_t)` exists (used in tests via `dDP.zero(...)`); if the
API is `zero()` with a prior `upload`/sizing, allocate via the same idiom the other wrappers use
for internal scratch (search `GpuBuffer<float>` scratch usage in `gpu_chiron.cu`). `chiron_drift_backward`
calls `chiron_reln_backward_reanchor`, defined earlier in the same TU — no header change needed.

- [ ] **Step 5: Build, install, rebuild tests, run all chiron tests**

```bash
cd /home/robert/dev/glades-ml && sh .configure.sh cuda 2>&1 | tail -3
cd /home/robert/dev/glades-ml/build && make install 2>&1 | tail -2
cd /home/robert/dev/glades-ml/unit-tests/build && sh .configure.sh cuda 2>&1 | tail -3
cd /home/robert/dev/glades-ml/unit-tests && bash test.sh nnall 2>&1 | grep -iE "drift|FAIL|CHIRON" | tail -20
```
Expected: `CHIRON drift backward CPU/GPU parity` PASSES; all prior CHIRON tests still pass.

- [ ] **Step 6: Commit**

```bash
cd /home/robert/dev/glades-ml
git add "Backend/Machine Learning/Networks/cuda/gpu_chiron.cu" \
        "Backend/Machine Learning/Networks/cuda/gpu_chiron.h" \
        "unit-tests/Backend/Machine Learning/chiron-test.cpp" \
        "unit-tests/Backend/Machine Learning/chiron-test.h" unit-tests/main.cpp
git commit -m "CHIRON OBSD: GPU drift backward (reuses reanchor for dp/dgamma/dbeta) + parity test"
```

---

## Task 4: Trainer — `a_drift[l]` parameter (alloc, init 0, Adam, checkpoint)

**Files:**
- Modify: `/home/robert/dev/glades-trainer/trainer/chiron_main.cpp`

Mirror the existing `gamma_p[l]`/`beta_p[l]` lifecycle exactly (search `gamma_p` to find every site:
declaration in the weights struct, allocation, optimizer-state allocation, checkpoint save, load).

- [ ] **Step 1: Add the parameter + Adam state**

In the weights struct where `gamma_p`/`beta_p` are declared, add `std::vector<GpuBuffer<float>*> a_drift;`
(and its Adam moment buffers, mirroring `gamma_p`'s `m_/v_` vectors). In the allocation routine,
allocate `a_drift[l]` as `[m]`, **initialize to 0.0f** (NOT ones — this is the ReZero gate), and
allocate its Adam state zeroed. Match the `beta_p` init-to-zero call exactly, just for `a_drift`.

- [ ] **Step 2: Persist in the checkpoint**

In `save_full` (where `gamma_p`/`beta_p` are written under the per-layer-fuse checkpoint bit),
write `a_drift[l]` too. Reserve a NEW checkpoint flag bit (next free bit) `has_a_drift` so older
checkpoints load without it. In `load_full`, read `a_drift[l]` when `has_a_drift` is set; when
absent, leave `a_drift[l]` at its zero-init (so loading a pre-OBSD checkpoint gives the flagship).

- [ ] **Step 3: Build the trainer**

```bash
cd /home/robert/dev/glades-trainer && sh run.sh 2>&1 | tail -5   # builds + re-rsyncs headers
```
Expected: `glades_chiron_train` builds clean.

- [ ] **Step 4: Commit**

```bash
cd /home/robert/dev/glades-trainer
git add trainer/chiron_main.cpp
git commit -m "CHIRON OBSD: add per-layer a_drift gate param (init 0, Adam, checkpoint bit)"
```

---

## Task 5: Trainer — `--per-layer-drift` flag + forward wiring + E0 bit-identity gate

**Files:**
- Modify: `/home/robert/dev/glades-trainer/trainer/chiron_main.cpp`, `run.sh`

- [ ] **Step 1: Add the flags**

Next to `--fuse-attn-per-layer` parsing (`chiron_main.cpp:2102`): add
`else if (streq(a, "--per-layer-drift")) cfg.perLayerDrift = true;` and
`else if (streq(a, "--drift-warmup")) cfg.driftWarmup = atoi(argv[++i]);`. Declare
`bool perLayerDrift; int driftWarmup;` in the config struct, default `false` / `0` in the
constructor (next to `fuseAttnPerLayer(true)` at :1486).

- [ ] **Step 2: Wire the forward drift at every layer**

In EACH forward attention branch (SCFA `~12708`, bf16w `~12845`, standard `~12955`), immediately
AFTER `chiron_attention_shear*` (so `p` already holds `p += Y_l(q)`) and BEFORE the q-side
`chiron_reln_forward`, add (computing the warmup scale; `cfg.driftWarmup<=0` ⇒ scale 1):

```cpp
if (cfg.perLayerDrift)
{
	float scale = 1.0f;
	if (cfg.driftWarmup > 0) { scale = (float)s.step / (float)cfg.driftWarmup; if (scale > 1.0f) scale = 1.0f; }
	if (!glades::gpu::chiron_drift_into_q(
	        s.p.data(), s.q.data(), W.a_drift[l]->data(),
	        W.gamma_p[l]->data(), W.beta_p[l]->data(),
	        /*sign=*/+1.0f, scale, T, m, cfg.eps_reln)) return false;
}
```
Use the step counter the trainer already threads (the same one used by `--sira-warmup`; search
`siraWarmup` for the exact field, e.g. `s.step` / `stepForDrift`). Apply at all layers `l`
(including `L−1`, on top of the existing `--fuse-attn-reln` fold). With `a_drift=0` it is exactly
the identity.

- [ ] **Step 3: Whitelist the flags in `run.sh` if needed**

If `run.sh` forwards only known flags, add `--per-layer-drift` and `--drift-warmup` to its
pass-through. (If it forwards `"$@"` verbatim, skip.)

- [ ] **Step 4: Build + run the E0 bit-identity gate**

```bash
cd /home/robert/dev/glades-trainer && sh run.sh 2>&1 | tail -3
# Tiny deterministic run WITHOUT the flag:
sh run.sh flagship --seq-len 256 --m 256 --layers 4 --heads 4 --dhead 64 --steps 3 --seed 1337 \
   --no-fuse-attn --fuse-attn-reln 2>&1 | grep -E "step|loss" > /tmp/obsd_base.txt
# Same WITH --per-layer-drift (a_drift=0 ⇒ must be bit-identical):
sh run.sh flagship --seq-len 256 --m 256 --layers 4 --heads 4 --dhead 64 --steps 3 --seed 1337 \
   --no-fuse-attn --fuse-attn-reln --per-layer-drift 2>&1 | grep -E "step|loss" > /tmp/obsd_drift0.txt
diff /tmp/obsd_base.txt /tmp/obsd_drift0.txt && echo "E0 PASS: bit-identical at a=0"
```
Expected: **identical loss every step** (E0). If they differ, the forward drift is not a true
no-op at `a=0` — STOP and fix before proceeding (likely a stale/garbage `a_drift` init).

- [ ] **Step 5: Commit**

```bash
cd /home/robert/dev/glades-trainer
git add trainer/chiron_main.cpp run.sh
git commit -m "CHIRON OBSD: --per-layer-drift forward wiring; E0 bit-identity at a=0 confirmed"
```

---

## Task 6: Trainer — inverse-walk + backward wiring + E2 reconstruction

**Files:**
- Modify: `/home/robert/dev/glades-trainer/trainer/chiron_main.cpp`

Find the backward/inverse-walk fuse sites (`cfg.fuseAttnPerLayer` near `4296`, `7103`, `14374`).
The OBSD inverse undoes the drift AFTER the q-reln inverse and BEFORE the attention-shear inverse;
the OBSD backward runs where the per-layer-fuse backward runs.

- [ ] **Step 1: Wire the inverse-walk drift (reconstruct q)**

In the inverse walk, after `chiron_reln_inverse` recovers the post-shear `q` and before
`chiron_attention_shear(..., invert=true)`, add the drift inverse (same `scale` as the forward of
this step):

```cpp
if (cfg.perLayerDrift)
{
	float scale = 1.0f;
	if (cfg.driftWarmup > 0) { scale = (float)s.step / (float)cfg.driftWarmup; if (scale > 1.0f) scale = 1.0f; }
	if (!glades::gpu::chiron_drift_into_q(
	        s.p.data(), s.q.data(), W.a_drift[l]->data(),
	        W.gamma_p[l]->data(), W.beta_p[l]->data(),
	        /*sign=*/-1.0f, scale, T, m, cfg.eps_reln)) return false;
}
```

- [ ] **Step 2: Wire the backward (accumulate grads)**

At the per-layer-fuse backward site, after the q-reln backward produces `dq` flowing into the
drift output and BEFORE the attention-shear backward, add:

```cpp
if (cfg.perLayerDrift)
{
	float scale = 1.0f;
	if (cfg.driftWarmup > 0) { scale = (float)s.step / (float)cfg.driftWarmup; if (scale > 1.0f) scale = 1.0f; }
	if (!glades::gpu::chiron_drift_backward(
	        s.dq.data(), s.p.data(), W.a_drift[l]->data(),
	        W.gamma_p[l]->data(), W.beta_p[l]->data(), scale,
	        T, m, cfg.eps_reln,
	        s.dp.data(), W.a_drift_grad[l]->data(),
	        W.gamma_p_grad[l]->data(), W.beta_p_grad[l]->data(),
	        s.reanchor_scratch.data())) return false;
}
```
Use the trainer's actual gradient-buffer names (search `gamma_p_grad` / the dgamma accumulation for
`gamma_p`; `a_drift_grad` is the gradient buffer added in Task 4). `s.dq`/`s.dp` are the adjoint
buffers used by the existing fuse backward; `s.reanchor_scratch` is the `2*T` scratch the reanchor
path already allocates (search `scratch_stats_split` / the `--reln-reanchor` wiring). Ensure
`a_drift` is included in the Adam update loop (mirror `gamma_p`'s update call).

- [ ] **Step 3: Build + run the E2 reconstruction test**

```bash
cd /home/robert/dev/glades-trainer && sh run.sh 2>&1 | tail -3
# 20-step run with a NONZERO drift (force a_drift to learn by seeding small, or set --drift-warmup
# low) and the reconstruction/inverse-walk self-check the trainer already runs (search for the
# recon error print, e.g. "recon" / "inverse"); confirm it stays within BF16 ULP bound.
sh run.sh flagship --seq-len 512 --m 256 --layers 8 --heads 4 --dhead 64 --steps 20 --seed 1337 \
   --no-fuse-attn --fuse-attn-reln --per-layer-drift --drift-warmup 5 2>&1 | grep -iE "recon|nan|inf|skip" | tail
```
Expected: 0 NaN/Inf, 0 grad-skips, reconstruction error within the trainer's existing tolerance.
If the trainer lacks a recon print, add a one-off `||q_reconstructed − q_saved||_inf` check at small
shape (compare a forward-saved `q` against the inverse-walk result for layer 0).

- [ ] **Step 4: Commit**

```bash
cd /home/robert/dev/glades-trainer
git add trainer/chiron_main.cpp
git commit -m "CHIRON OBSD: inverse-walk + backward wiring (reanchored); E2 reconstruction within ULP"
```

---

## Task 7: Trainer — budget logging `B = Σ_l √(‖G_l‖‖A_l‖)` (monitoring)

**Files:**
- Modify: `/home/robert/dev/glades-trainer/trainer/chiron_main.cpp`

A cheap proxy is enough for monitoring (full power-iteration is deferred): use
`‖G_l‖ ≈ scale·‖a_l‖∞·‖M_l⁻¹‖∞` (closed form, §8.4 of the spec) and `‖A_l‖ ≈ ‖dp_l‖/‖dq_l‖` from
the adjoint norms already computed for grad-norm logging.

- [ ] **Step 1: Add the budget log behind `--per-layer-drift`**

Where the trainer logs the global grad norm, accumulate
`B += sqrtf( (scale*max|a_drift[l]|*max|gamma_p[l]|) * attnJacEst_l )` over layers and print
`[obsd] step=… B=… max|a|=…` every `--log-every`. `attnJacEst_l` = the per-layer `‖dp‖/‖dq‖`
ratio if available, else 1.0 (still lets you watch `B` grow with `‖a‖`).

- [ ] **Step 2: Build + smoke-test the log**

```bash
cd /home/robert/dev/glades-trainer && sh run.sh 2>&1 | tail -3
sh run.sh flagship --seq-len 512 --m 256 --layers 8 --heads 4 --dhead 64 --steps 20 --seed 1337 \
   --no-fuse-attn --fuse-attn-reln --per-layer-drift --drift-warmup 10 2>&1 | grep -E "\[obsd\]" | tail
```
Expected: `[obsd]` lines printing a finite, slowly-growing `B` as `max|a|` grows from 0.

- [ ] **Step 3: Commit**

```bash
cd /home/robert/dev/glades-trainer
git add trainer/chiron_main.cpp
git commit -m "CHIRON OBSD: budget B = sum sqrt(||G||·||A||) monitoring log"
```

---

## Task 8: Small-shape validation E0/E1/E2 + findings doc

**Files:**
- Create: `/home/robert/dev/glades-ml/research/CHIRON_OBSD_RESULT_2026_06_27.md`

- [ ] **Step 1: E1 — stability A/B in the exploding regime**

Run the historically-exploding config two ways and capture grad norm + skips. The legacy unstable
path is `--fuse-attn-per-layer` (scalar α=1/√L); OBSD is `--per-layer-drift`:

```bash
cd /home/robert/dev/glades-trainer
echo "=== (a) legacy fuse-per-layer (expect explosion) ==="
sh run.sh flagship --seq-len 512 --m 1024 --layers 24 --heads 8 --dhead 128 --steps 5 --seed 1337 \
   --no-fuse-attn --fuse-attn-per-layer 2>&1 | grep -iE "grad|skip|nan|inf|loss" | head
echo "=== (b) OBSD per-layer-drift (expect bounded, 0 skips) ==="
sh run.sh flagship --seq-len 512 --m 1024 --layers 24 --heads 8 --dhead 128 --steps 50 --seed 1337 \
   --no-fuse-attn --fuse-attn-reln --per-layer-drift --drift-warmup 25 2>&1 | grep -iE "grad|skip|nan|inf|\[obsd\]|loss" | tail -20
```
**PASS:** (a) shows ‖g‖ blowing up / grad-skips; (b) holds bounded ‖g‖ with **0 grad-skips**, and
the `[obsd] B` value tracks `log‖g‖` (record both). Ablation: rerun (b) routing the drift backward
through the NON-reanchor path (temporary `chiron_reln_backward` instead of `_reanchor`) — expect
the κ_N spike to reappear (confirms reanchor is load-bearing).

- [ ] **Step 2: Record E0/E1/E2 in the findings doc**

Create `research/CHIRON_OBSD_RESULT_2026_06_27.md` with: the E0 diff result (bit-identical),
the E1 A/B numbers (‖g‖, skips, B, and the reanchor-off ablation), the E2 reconstruction error,
the exact commands, and the verdict (proceed to production sweep / or stop with diagnosis).

- [ ] **Step 3: Commit**

```bash
cd /home/robert/dev/glades-ml
git add research/CHIRON_OBSD_RESULT_2026_06_27.md
git commit -m "CHIRON OBSD: E0/E1/E2 small-shape validation (bit-identity, stability A/B, reconstruction)"
```

---

## Task 9: Production E3/E4 (B-sweep, quality gate, capped multi-seed) + document/close

**Files:**
- Modify: `research/CHIRON_OBSD_RESULT_2026_06_27.md`, `CLAUDE.md` (only if it ships),
  `/home/robert/.claude/projects/-home-robert-dev-glades-ml/memory/MEMORY.md`

- [ ] **Step 1: E3 — budget sweep at production shape (short runs)**

Sweep init scale / warmup to locate a `B*` that improves val while staying sub-overflow. Use the
production shape but short (≤5k) runs (pre-create `--save`):

```bash
cd /home/robert/dev/glades-trainer
for w in 250 1000; do
  sh run.sh flagship --accum 4 --lr 3e-4 --warmup 750 --sira-warmup 250 --zloss-coef 1e-4 \
     --qk-norm --sira-coef 1e-2 --sira-energy-weight 1.0 --sira-balance-weight 0.25 \
     --sira-action-weight 0.0 --grad-clip 0.5 --dq-layer-clamp 1.0 --dq-embed-clamp 1.0 \
     --reln-reanchor --no-fuse-attn --fuse-attn-reln --per-layer-drift --drift-warmup $w \
     --steps 5000 --seed 1337 --save-every 5000 --save /tmp/obsd_w$w 2>&1 | grep -iE "val|\[obsd\]|skip" | tail
done
```
**Record:** val NLL vs the flagship's 5k baseline, `B`, 0-skip status, for each warmup. Pick the
best stable warmup for E4.

- [ ] **Step 2: E4 — production quality gate (single-seed primary)**

Run the full recipe (CLAUDE.md flagship reproduce command) PLUS `--per-layer-drift --drift-warmup <best>`,
single seed 1337, to the flagship's step budget. Compare val NLL to 1.92 (use the same wide-val and
`chiron_infer` TF check the flagship used).

```bash
cd /home/robert/dev/glades-trainer
sh run.sh flagship --accum 4 --lr 3e-4 --warmup 750 --sira-warmup 250 --zloss-coef 1e-4 --qk-norm \
   --sira-coef 1e-2 --sira-energy-weight 1.0 --sira-balance-weight 0.25 --sira-action-weight 0.0 \
   --grad-clip 0.5 --dq-layer-clamp 1.0 --dq-embed-clamp 1.0 --reln-reanchor \
   --no-fuse-attn --fuse-attn-reln --per-layer-drift --drift-warmup <best> \
   --steps 60000 --seed 1337 --save-every 6000 --save <dir>
```
**Ship gate:** val NLL < 1.92, 0 grad-skips through the ~1.6B-token regime, wall ≤ +10% vs the
reanchor flagship, `chiron_infer --tf-check` reproduces (within the serving path). If it does NOT
beat 1.92, STOP and record the negative result (this is FM-1 — the budget that helps may exceed the
BF16-safe budget); the design's verdict is "stable but at flagship perplexity."

- [ ] **Step 3: Multi-seed confirmation (capped at 5k–15k per seed)**

Per the owner cost-control decision: seeds `{2024, 4242}` (1337 already done), each **5k or 15k
steps at most** (NOT 30k+). Confirm the val-NLL **sign** (improving vs the matched-step flagship
baseline) and **0 grad-skips** cross-seed. This is a sign + stability gate, not a full-trajectory
multi-seed ship.

```bash
for sd in 2024 4242; do
  sh run.sh flagship --accum 4 --lr 3e-4 --warmup 750 --sira-warmup 250 --zloss-coef 1e-4 --qk-norm \
     --sira-coef 1e-2 --sira-energy-weight 1.0 --sira-balance-weight 0.25 --sira-action-weight 0.0 \
     --grad-clip 0.5 --dq-layer-clamp 1.0 --dq-embed-clamp 1.0 --reln-reanchor \
     --no-fuse-attn --fuse-attn-reln --per-layer-drift --drift-warmup <best> \
     --steps 15000 --seed $sd --save-every 15000 --save /tmp/obsd_seed$sd 2>&1 | grep -iE "val|skip" | tail
done
```

- [ ] **Step 4: Document & close**

Finalize `research/CHIRON_OBSD_RESULT_2026_06_27.md` (E3 sweep table, E4 ship/no-ship verdict +
val numbers, multi-seed sign/stability). **If it ships:** add an OBSD section to `CLAUDE.md`
(new production flagship recipe = reanchor recipe + `--per-layer-drift --drift-warmup <best>`),
update `runner.sh` preference if a new checkpoint is promoted, and add a one-line pointer to
`MEMORY.md` linking `[[reln_reanchor_cure_arc]]`. **If it does not ship:** record the negative
result + FM-1 diagnosis in the research doc and `MEMORY.md`; leave `--per-layer-drift` as a
default-off, validated, documented flag.

- [ ] **Step 5: Commit**

```bash
cd /home/robert/dev/glades-ml
git add research/CHIRON_OBSD_RESULT_2026_06_27.md CLAUDE.md
git commit -m "CHIRON OBSD: production E3/E4 result + ship/no-ship verdict (multi-seed capped 5k-15k)"
```

---

## Self-Review

**Spec coverage:**
- §5 mechanism (fwd/inv/bwd, mass inside φ, gate outside, parameter-free normalize) → Tasks 1–3 (CPU ref + GPU kernels) + Task 5/6 (wiring) ✓
- §6 flagship recovery at init → Task 5 Step 4 E0 bit-identity gate ✓
- §7 schedule (zero-init, warmup ramp, optional budget) → Task 4 (init 0), Task 5/6 (`--drift-warmup` scale), Task 7 (budget log); budget *projection* intentionally deferred (spec marks it optional) ✓
- §8.3 mass non-absorbability → enforced by design: φ=tanh (nonlinear) + parameter-free normalize (Tasks 1–2) ✓
- §8.5 operator budget + explosion diagnosis → Task 7 (B log) + Task 8 E1 (A/B vs legacy) ✓
- §8.7 reanchor as conditioning → Task 3 (backward reuses `chiron_reln_backward_reanchor`) + Task 8 reanchor-off ablation ✓
- §8.1/8.2 exact reversibility / BF16 ULP → Task 2 reversibility test + Task 6 E2 reconstruction ✓
- §8.6 single-kick (no new GEMM) → no second attention pass anywhere; drift is elementwise ✓
- §12 experiment ladder E0–E4 → Tasks 5,8,9 ✓
- §14 ship gate + capped multi-seed → Task 9 Steps 2–3 ✓
- §9 cost / code mapping (reuse gamma_p/beta_p, add a_drift, generalize the fuse path) → Tasks 4–6 ✓

**Placeholder scan:** code steps contain complete kernel/CPU/test code. Trainer wiring (Tasks 4–6)
references existing sites by symbol (`gamma_p`, `cfg.fuseAttnPerLayer`, `scratch_stats_split`)
because exact line numbers drift; each instruction names the symbol to search and the code to add.
`<best>`/`<dir>` in Task 9 are run-time choices from E3, not unfilled plan blanks.

**Type/name consistency:** `chiron_drift_into_q(p,q,a,gamma,beta,sign,scale,T,m,eps)` and
`chiron_drift_backward(dq_out,p,a,gamma,beta,scale,T,m,eps,dp,da,dgamma,dbeta,scratch)` are used
identically in decls (Task 2/3 Step 3), kernels/wrappers (Step 4), and tests (Step 1). CPU refs
`glades::chiron::drift_into_q` / `drift_backward` match across Task 1 (impl) and Tasks 1–3 (tests).
Param `a_drift[l]` (+ `a_drift_grad[l]`) consistent across Tasks 4–6. Flags `--per-layer-drift` /
`--drift-warmup` consistent across Tasks 5–9.
