# CHIRON WhiSC-D Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement WhiSC-D — a per-channel *whitened* symplectic coupling for CHIRON — by wrapping the existing SORC rotation in a detached, per-channel scale-balancing frame, so the cross-depth coupling's backward gradient is bounded independent of the `‖p‖/‖q‖` asymmetry (the requirement SORC failed), then run the pre-registered 2500-step E3 divergence gate.

**Architecture:** WhiSC-D = `unwhiten ∘ rot ∘ whiten` per channel, where `whiten` is the detached symplectic scaling `q̃=q/a, p̃=a·p`, `a=(E[q²]/E[p²])^{1/4}` (EMA, per channel), and `rot` is the *unmodified* SORC rotation kernel. Because the rotation runs in the balanced (whitened) frame, its backward `da/dc/dphi` are naturally O(1) — no `ρ`-amplification. We reuse the SORC parameter `rot_phi`, its Adam state, checkpoint bit 1024, and the `[sorc]` monitor; the only new code is two GPU kernels (`chiron_whisc_scale`, `chiron_whisc_update_stats`), their CPU refs + unit tests, and the trainer wiring that inserts the whiten/unwhiten and the stats update.

**Tech Stack:** C++98 library kernels + CUDA (`gpu_chiron.cu`); the C++ trainer (`glades-trainer/trainer/chiron_main.cpp`); custom `ASSERT` unit-test framework.

## Global Constraints

- **Design source:** `docs/superpowers/specs/2026-06-30-chiron-whitened-frame-coupling-design.md` (WhiSC-D = §5.2, §6, §7, §13).
- **Library install gotcha:** the trainer links the **installed** glades (`find_package(glades REQUIRED)` → `~/.local`). After any change under `Backend/Machine Learning/`, you MUST `cd build && make install` before rebuilding the trainer. Unit tests link the **in-tree** build (no install needed for them).
- **Library build:** `cd ~/dev/glades-ml && sh .configure.sh cuda` (CUDA build). Install: `cd ~/dev/glades-ml/build && make install`.
- **Unit-test build/run:** `cd ~/dev/glades-ml/unit-tests/build && sh .configure.sh cuda`, then `cd ~/dev/glades-ml/unit-tests && bash test.sh chiron-whisc`.
- **Trainer build:** `cd ~/dev/glades-trainer && bash build.sh` (NOT `sh` — it uses `set -o pipefail`). Binary: `build/glades_chiron_train`.
- **Default-off + E0 parity:** all new behavior is gated behind `--whisc-coupling` (default off). With the gate off the forward/backward must be **bit-identical** to the current baseline. The coupling angle param `rot_phi` is init 0 ⇒ `θ=0 ⇒ rot=identity` ⇒ the block is identity even with `--whisc-coupling` on, at step 0.
- **Frozen-per-step statistic:** the whitening `a` is computed **once per step in the forward** (updating the EMA) and **cached**; the backward reuses the cached `a` (it must NOT re-update the EMA), or reversibility breaks.
- **Determinism:** the EMA stat is `stop_gradient` (detached); no autodiff edge runs through `a`, `Pbar`, or `Qbar`.
- **Math reference (per channel `i`, token `t`):** `whiten`: `q̃=q/a_i, p̃=a_i·p`; `a_i = clamp((Qbar_i/(Pbar_i+ε))^{1/4}, 1/C, C)` with `Qbar=E[q²], Pbar=E[p²]` (post-shear), `ε=1e-12`, clamp `C` (default 8). This gives balanced `E[q̃²]=E[p̃²]=√(Qbar·Pbar)` and `det(whiten)=1` (symplectic).

---

## File structure

**Library (`~/dev/glades-ml`):**
- `Backend/Machine Learning/Networks/transformer_chiron_ops.h` — add CPU refs `whisc_scale`, `whisc_update_stats` (immediately after `rot_backward`, line 323).
- `Backend/Machine Learning/Networks/cuda/gpu_chiron.h` — declare `chiron_whisc_scale`, `chiron_whisc_update_stats` (after the rot decls, ~line 300) + their CUDA-absent inline stubs (after line 776).
- `Backend/Machine Learning/Networks/cuda/gpu_chiron.cu` — implement both kernels + host wrappers (after the `chiron_rot_*` definitions).
- `unit-tests/Backend/Machine Learning/chiron-test.h` — declare `WhiSCScaleCpuTest`, `WhiSCStatsCpuTest`, `WhiSCGpuParityTest`, `WhiSCBackwardParityTest` (after line 119).
- `unit-tests/Backend/Machine Learning/chiron-test.cpp` — implement the four tests.
- `unit-tests/main.cpp` — register a `chiron-whisc` arg (mirror the `chiron-rot` block at line 269).

**Trainer (`~/dev/glades-trainer`):**
- `trainer/chiron_main.cpp` — Config fields + defaults, CLI parse, mutual-exclusion guard, Scratch EMA buffers + allocation, forward wiring (3 branches), backward wiring, monitor extension. (All exact line anchors given per task.)

---

## Task 1: CPU reference `whisc_scale` + reversibility/symplecticity test

**Files:**
- Modify: `Backend/Machine Learning/Networks/transformer_chiron_ops.h:323` (after `rot_backward`)
- Modify: `unit-tests/Backend/Machine Learning/chiron-test.h:119` (declare)
- Test: `unit-tests/Backend/Machine Learning/chiron-test.cpp` (add `WhiSCScaleCpuTest`)
- Modify: `unit-tests/main.cpp:269` (register `chiron-whisc`)

**Interfaces:**
- Produces: `void glades::chiron::whisc_scale(float* q, float* p, const float* a, float sign, unsigned int T, unsigned int m)` — `sign=+1` whitens (`q/=a, p*=a`), `sign=-1` unwhitens (`q*=a, p/=a`), per channel `a[i]`, broadcast over `T`.
- Consumes: existing `glades::chiron::rot_coeffs`, `rot_forward`, `rot_inverse` (transformer_chiron_ops.h:275-301).

- [ ] **Step 1: Declare the test** in `unit-tests/Backend/Machine Learning/chiron-test.h` after line 119 (`void CHIRONRotBackwardParityTest();`):

```cpp
void WhiSCScaleCpuTest();
void WhiSCStatsCpuTest();
void WhiSCGpuParityTest();
void WhiSCBackwardParityTest();
```

- [ ] **Step 2: Register `chiron-whisc`** in `unit-tests/main.cpp`. Find the block at line 269 (`else if (strcmp(argv[1], "chiron-rot") == 0 ...`) and add immediately after its closing brace:

```cpp
	    else if (strcmp(argv[1], "chiron-whisc") == 0 || strcmp(argv[1], "whisc") == 0)
	    {
		WhiSCScaleCpuTest();
		WhiSCStatsCpuTest();
		WhiSCGpuParityTest();
		WhiSCBackwardParityTest();
	    }
```

- [ ] **Step 3: Write the failing test** `WhiSCScaleCpuTest` at the end of `unit-tests/Backend/Machine Learning/chiron-test.cpp`. It asserts (a) whiten then unwhiten is identity, and (b) the composite `unwhiten∘rot∘whiten` is reversible via `unwhiten∘rot_inverse∘whiten`, and (c) det=1 (norm of `(q,p)` in the whitened metric is preserved by rot).

```cpp
void WhiSCScaleCpuTest()
{
	const unsigned int T = 5, m = 4;
	std::vector<float> a(m);
	for (unsigned i = 0; i < m; ++i) a[i] = 0.2f + 0.5f * (float)i; // distinct, >0
	std::vector<float> q(T*m), p(T*m), q0, p0;
	for (unsigned k = 0; k < T*m; ++k) { q[k] = 0.3f*(float)k - 1.0f; p[k] = 17.0f*(0.1f*(float)k + 0.5f); } // p ~ 17x q (the asymmetry)
	q0 = q; p0 = p;

	// (a) whiten then unwhiten == identity
	glades::chiron::whisc_scale(&q[0], &p[0], &a[0], +1.0f, T, m);
	glades::chiron::whisc_scale(&q[0], &p[0], &a[0], -1.0f, T, m);
	float maxerr = 0.0f;
	for (unsigned k = 0; k < T*m; ++k) { maxerr = std::max(maxerr, std::fabs(q[k]-q0[k])); maxerr = std::max(maxerr, std::fabs(p[k]-p0[k])); }
	ASSERT("WhiSC scale whiten/unwhiten not identity", maxerr < 1e-5f);

	// (b) composite reversibility: unwhiten . rot . whiten , inverted by unwhiten . rot_inverse . whiten
	std::vector<float> rc_a(m), rc_c(m); float th;
	for (unsigned i = 0; i < m; ++i) glades::chiron::rot_coeffs(0.4f*(float)i - 0.5f, 0.1f, 1.0f, rc_a[i], rc_c[i], th);
	q = q0; p = p0;
	glades::chiron::whisc_scale(&q[0], &p[0], &a[0], +1.0f, T, m);
	glades::chiron::rot_forward(&q[0], &p[0], &rc_a[0], &rc_c[0], T, m);
	glades::chiron::whisc_scale(&q[0], &p[0], &a[0], -1.0f, T, m);
	// inverse:
	glades::chiron::whisc_scale(&q[0], &p[0], &a[0], +1.0f, T, m);
	glades::chiron::rot_inverse(&q[0], &p[0], &rc_a[0], &rc_c[0], T, m);
	glades::chiron::whisc_scale(&q[0], &p[0], &a[0], -1.0f, T, m);
	maxerr = 0.0f;
	for (unsigned k = 0; k < T*m; ++k) { maxerr = std::max(maxerr, std::fabs(q[k]-q0[k])); maxerr = std::max(maxerr, std::fabs(p[k]-p0[k])); }
	ASSERT("WhiSC composite not reversible", maxerr < 1e-4f);
}
```

- [ ] **Step 4: Run it to verify it fails** (build first; `whisc_scale` does not exist yet):

```
cd ~/dev/glades-ml && sh .configure.sh cuda
cd ~/dev/glades-ml/unit-tests/build && sh .configure.sh cuda
```
Expected: COMPILE ERROR — `'whisc_scale' is not a member of 'glades::chiron'`.

- [ ] **Step 5: Implement `whisc_scale`** in `transformer_chiron_ops.h` immediately after `rot_backward` (after line 323):

```cpp
// WhiSC per-channel symplectic whitening scale. sign=+1 whitens (q/=a, p*=a);
// sign=-1 unwhitens (q*=a, p/=a). a is per-channel [m], broadcast over T. det=1.
inline void whisc_scale(float* q, float* p, const float* a, float sign, unsigned int T, unsigned int m) {
	for (unsigned t=0;t<T;++t) for (unsigned i=0;i<m;++i) {
		unsigned long k=(unsigned long)t*m+i; float ai=a[i];
		if (sign > 0.0f) { q[k] = q[k]/ai; p[k] = p[k]*ai; }
		else             { q[k] = q[k]*ai; p[k] = p[k]/ai; }
	}
}
```

- [ ] **Step 6: Build and run the test to verify it passes:**

```
cd ~/dev/glades-ml && sh .configure.sh cuda
cd ~/dev/glades-ml/unit-tests/build && sh .configure.sh cuda
cd ~/dev/glades-ml/unit-tests && bash test.sh chiron-whisc
```
Expected: PASS (no `ASSERT` failures for `WhiSCScaleCpuTest`; the other three are not implemented yet — they will print nothing or you may temporarily comment their calls in main.cpp; re-enable in later tasks).

- [ ] **Step 7: Commit:**

```bash
cd ~/dev/glades-ml
git add "Backend/Machine Learning/Networks/transformer_chiron_ops.h" "unit-tests/Backend/Machine Learning/chiron-test.h" "unit-tests/Backend/Machine Learning/chiron-test.cpp" unit-tests/main.cpp
git commit -m "WhiSC Task 1: whisc_scale CPU ref + reversibility/symplecticity test"
```

---

## Task 2: CPU reference `whisc_update_stats` + balance test

**Files:**
- Modify: `Backend/Machine Learning/Networks/transformer_chiron_ops.h` (after `whisc_scale`)
- Test: `unit-tests/Backend/Machine Learning/chiron-test.cpp` (add `WhiSCStatsCpuTest`)

**Interfaces:**
- Produces: `void glades::chiron::whisc_update_stats(const float* q, const float* p, unsigned int T, unsigned int m, float ema, float eps, float clamp, float* Pbar, float* Qbar, float* a)` — updates EMA second moments `Qbar=E[q²], Pbar=E[p²]` per channel and writes `a[i] = clamp((Qbar_i/(Pbar_i+eps))^{1/4}, 1/clamp, clamp)`.

- [ ] **Step 1: Write the failing test** `WhiSCStatsCpuTest` (append to `chiron-test.cpp`). Verifies that after the stats update, whitening with the produced `a` *balances* the two subspaces (`E[q̃²] ≈ E[p̃²]`):

```cpp
void WhiSCStatsCpuTest()
{
	const unsigned int T = 256, m = 3;
	std::vector<float> q(T*m), p(T*m);
	// channel 0: p ~ 17x q ; channel 1: p ~ 5x q ; channel 2: balanced
	float pscale[3] = {17.0f, 5.0f, 1.0f};
	for (unsigned t=0;t<T;++t) for (unsigned i=0;i<m;++i) {
		float u = std::sin(0.123f*(float)(t*m+i)); // deterministic pseudo-noise
		q[t*m+i] = u; p[t*m+i] = pscale[i]*std::cos(0.077f*(float)(t*m+i));
	}
	std::vector<float> Pbar(m,1.0f), Qbar(m,1.0f), a(m,1.0f);
	// one-shot EMA=1.0 => Qbar,Pbar become the batch means exactly
	glades::chiron::whisc_update_stats(&q[0], &p[0], T, m, /*ema=*/1.0f, /*eps=*/1e-12f, /*clamp=*/8.0f, &Pbar[0], &Qbar[0], &a[0]);
	// whiten and check balance per channel
	std::vector<float> qt=q, pt=p;
	glades::chiron::whisc_scale(&qt[0], &pt[0], &a[0], +1.0f, T, m);
	for (unsigned i=0;i<m;++i) {
		double sq=0, sp=0; for (unsigned t=0;t<T;++t){ sq+=qt[t*m+i]*qt[t*m+i]; sp+=pt[t*m+i]*pt[t*m+i]; }
		double eq=sq/T, ep=sp/T;
		ASSERT("WhiSC stats: whitened subspaces not balanced", std::fabs(eq-ep) < 0.05*(eq+ep)+1e-6);
	}
}
```

- [ ] **Step 2: Run to verify it fails:**

```
cd ~/dev/glades-ml && sh .configure.sh cuda && cd unit-tests/build && sh .configure.sh cuda
```
Expected: COMPILE ERROR — `'whisc_update_stats' is not a member of 'glades::chiron'`.

- [ ] **Step 3: Implement `whisc_update_stats`** in `transformer_chiron_ops.h` after `whisc_scale`:

```cpp
// WhiSC per-channel EMA second-moment stats + whitening scale derivation.
// Qbar=E[q^2], Pbar=E[p^2] over T tokens, EMA-blended; a=clamp((Qbar/(Pbar+eps))^0.25, 1/clamp, clamp).
inline void whisc_update_stats(const float* q, const float* p, unsigned int T, unsigned int m,
                               float ema, float eps, float clamp, float* Pbar, float* Qbar, float* a) {
	for (unsigned i=0;i<m;++i) {
		double sq=0.0, sp=0.0;
		for (unsigned t=0;t<T;++t) { float qv=q[(unsigned long)t*m+i], pv=p[(unsigned long)t*m+i]; sq+=(double)qv*qv; sp+=(double)pv*pv; }
		float mq=(float)(sq/(double)T), mp=(float)(sp/(double)T);
		Qbar[i]=(1.0f-ema)*Qbar[i]+ema*mq;
		Pbar[i]=(1.0f-ema)*Pbar[i]+ema*mp;
		float ai=powf(Qbar[i]/(Pbar[i]+eps), 0.25f);
		float lo=1.0f/clamp, hi=clamp;
		a[i] = (ai<lo)?lo:((ai>hi)?hi:ai);
	}
}
```

- [ ] **Step 4: Build and run to verify it passes:**

```
cd ~/dev/glades-ml && sh .configure.sh cuda && cd unit-tests/build && sh .configure.sh cuda
cd ~/dev/glades-ml/unit-tests && bash test.sh chiron-whisc
```
Expected: PASS for `WhiSCScaleCpuTest` and `WhiSCStatsCpuTest`.

- [ ] **Step 5: Commit:**

```bash
cd ~/dev/glades-ml && git add "Backend/Machine Learning/Networks/transformer_chiron_ops.h" "unit-tests/Backend/Machine Learning/chiron-test.cpp"
git commit -m "WhiSC Task 2: whisc_update_stats CPU ref + balance test"
```

---

## Task 3: GPU kernels `chiron_whisc_scale` + `chiron_whisc_update_stats` + GPU/CPU parity

**Files:**
- Modify: `Backend/Machine Learning/Networks/cuda/gpu_chiron.h` (declare after line 300; stubs after line 776)
- Modify: `Backend/Machine Learning/Networks/cuda/gpu_chiron.cu` (implement after the `chiron_rot_*` host wrappers)
- Test: `unit-tests/Backend/Machine Learning/chiron-test.cpp` (add `WhiSCGpuParityTest`)

**Interfaces:**
- Produces: `bool glades::gpu::chiron_whisc_scale(float* q, float* p, const float* a, float sign, int T, int m)` and `bool glades::gpu::chiron_whisc_update_stats(const float* q, const float* p, int T, int m, float ema, float eps, float clamp, float* Pbar, float* Qbar, float* a)` (device pointers; semantics identical to the CPU refs).

- [ ] **Step 1: Declare the wrappers** in `gpu_chiron.h` after the `chiron_rot_backward` declaration (line 300):

```cpp
// WhiSC per-channel whitening scale (sign=+1 whiten q/=a,p*=a; sign=-1 unwhiten).
bool chiron_whisc_scale(float* q, float* p, const float* a, float sign, int T, int m);
// WhiSC EMA second-moment stats: updates Pbar=E[p^2], Qbar=E[q^2] per channel, writes a=clamp((Qbar/(Pbar+eps))^0.25,1/clamp,clamp).
bool chiron_whisc_update_stats(const float* q, const float* p, int T, int m,
                               float ema, float eps, float clamp, float* Pbar, float* Qbar, float* a);
```

- [ ] **Step 2: Add the CUDA-absent inline stubs** in `gpu_chiron.h` after line 776 (alongside the other `inline bool chiron_rot_*` stubs in the `#else` branch):

```cpp
inline bool chiron_whisc_scale(float*, float*, const float*, float, int, int) { return false; }
inline bool chiron_whisc_update_stats(const float*, const float*, int, int, float, float, float, float*, float*, float*) { return false; }
```

- [ ] **Step 3: Write the failing parity test** `WhiSCGpuParityTest` (append to `chiron-test.cpp`). Mirrors `CHIRONRotGpuParityTest` (line 18639) for buffer setup:

```cpp
void WhiSCGpuParityTest()
{
	const int T = 64, m = 8;
	std::vector<float> q(T*m), p(T*m), a(m);
	for (int k=0;k<T*m;++k){ q[k]=0.2f*std::sin(0.3f*k); p[k]=17.0f*std::cos(0.21f*k); }
	for (int i=0;i<m;++i) a[i]=0.15f+0.1f*i;
	// CPU scale
	std::vector<float> qc=q, pc=p; glades::chiron::whisc_scale(&qc[0],&pc[0],&a[0],+1.0f,(unsigned)T,(unsigned)m);
	// GPU scale
	glades::gpu::GpuBuffer<float> dQ,dP,dA; dQ.allocate(T*m); dP.allocate(T*m); dA.allocate(m);
	dQ.upload(&q[0],T*m); dP.upload(&p[0],T*m); dA.upload(&a[0],m);
	ASSERT("whisc_scale gpu failed", glades::gpu::chiron_whisc_scale(dQ.data(),dP.data(),dA.data(),+1.0f,T,m));
	std::vector<float> qg(T*m), pg(T*m); dQ.download(&qg[0],T*m); dP.download(&pg[0],T*m);
	float e=0; for (int k=0;k<T*m;++k){ e=std::max(e,std::fabs(qg[k]-qc[k])); e=std::max(e,std::fabs(pg[k]-pc[k])); }
	ASSERT("whisc_scale gpu/cpu mismatch", e < 1e-4f);

	// stats parity (ema=1 -> batch means)
	std::vector<float> Pc(m,1.0f),Qc(m,1.0f),Ac(m,1.0f);
	glades::chiron::whisc_update_stats(&q[0],&p[0],(unsigned)T,(unsigned)m,1.0f,1e-12f,8.0f,&Pc[0],&Qc[0],&Ac[0]);
	glades::gpu::GpuBuffer<float> dPb,dQb,dAo; dPb.allocate(m); dQb.allocate(m); dAo.allocate(m);
	std::vector<float> ones(m,1.0f); dPb.upload(&ones[0],m); dQb.upload(&ones[0],m);
	dQ.upload(&q[0],T*m); dP.upload(&p[0],T*m); // pristine q,p (dQ,dP were whitened by the scale test above)
	ASSERT("whisc_stats gpu failed",
	       glades::gpu::chiron_whisc_update_stats(dQ.data(),dP.data(),T,m,1.0f,1e-12f,8.0f,dPb.data(),dQb.data(),dAo.data()));
	std::vector<float> Ag(m); dAo.download(&Ag[0],m);
	float ea=0; for (int i=0;i<m;++i) ea=std::max(ea,std::fabs(Ag[i]-Ac[i]));
	ASSERT("whisc_stats gpu/cpu a mismatch", ea < 1e-4f);
}
```

- [ ] **Step 4: Run to verify it fails:**

```
cd ~/dev/glades-ml && sh .configure.sh cuda
```
Expected: LINK ERROR — undefined reference to `chiron_whisc_scale` / `chiron_whisc_update_stats`.

- [ ] **Step 5: Implement the kernels** in `gpu_chiron.cu` after the `chiron_rot_backward` host wrapper (search for `bool chiron_rot_backward`). Add:

```cpp
__global__ void whisc_scale_kernel(float* __restrict__ q, float* __restrict__ p,
                                   const float* __restrict__ a, float sign, int T, int m) {
	long k = (long)blockIdx.x * blockDim.x + threadIdx.x;
	long n = (long)T * m;
	if (k >= n) return;
	int i = (int)(k % m);
	float ai = a[i];
	if (sign > 0.0f) { q[k] = q[k] / ai; p[k] = p[k] * ai; }
	else             { q[k] = q[k] * ai; p[k] = p[k] / ai; }
}

bool chiron_whisc_scale(float* q, float* p, const float* a, float sign, int T, int m) {
	if (!q || !p || !a || T <= 0 || m <= 0) return false;
	long n = (long)T * m; int blk = 256; int grid = (int)((n + blk - 1) / blk);
	whisc_scale_kernel<<<grid, blk, 0, computeStream()>>>(q, p, a, sign, T, m);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}

// one block per channel; block reduces sum of squares over T tokens (column i, stride m).
__global__ void whisc_stats_kernel(const float* __restrict__ q, const float* __restrict__ p,
                                   int T, int m, float ema, float eps, float clamp,
                                   float* __restrict__ Pbar, float* __restrict__ Qbar, float* __restrict__ a) {
	int i = blockIdx.x; if (i >= m) return;
	__shared__ float sq[256]; __shared__ float sp[256];
	float lq = 0.0f, lp = 0.0f;
	for (int t = threadIdx.x; t < T; t += blockDim.x) {
		float qv = q[(long)t*m + i], pv = p[(long)t*m + i];
		lq += qv*qv; lp += pv*pv;
	}
	sq[threadIdx.x] = lq; sp[threadIdx.x] = lp; __syncthreads();
	for (int s = blockDim.x/2; s > 0; s >>= 1) { if (threadIdx.x < s) { sq[threadIdx.x]+=sq[threadIdx.x+s]; sp[threadIdx.x]+=sp[threadIdx.x+s]; } __syncthreads(); }
	if (threadIdx.x == 0) {
		float mq = sq[0]/(float)T, mp = sp[0]/(float)T;
		float qb = (1.0f-ema)*Qbar[i] + ema*mq;
		float pb = (1.0f-ema)*Pbar[i] + ema*mp;
		Qbar[i]=qb; Pbar[i]=pb;
		float ai = powf(qb/(pb+eps), 0.25f);
		float lo = 1.0f/clamp, hi = clamp;
		a[i] = (ai<lo)?lo:((ai>hi)?hi:ai);
	}
}

bool chiron_whisc_update_stats(const float* q, const float* p, int T, int m,
                               float ema, float eps, float clamp, float* Pbar, float* Qbar, float* a) {
	if (!q || !p || !Pbar || !Qbar || !a || T <= 0 || m <= 0) return false;
	whisc_stats_kernel<<<m, 256, 0, computeStream()>>>(q, p, T, m, ema, eps, clamp, Pbar, Qbar, a);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}
```

- [ ] **Step 6: Build and run to verify it passes:**

```
cd ~/dev/glades-ml && sh .configure.sh cuda && cd unit-tests/build && sh .configure.sh cuda
cd ~/dev/glades-ml/unit-tests && bash test.sh chiron-whisc
```
Expected: PASS for the three implemented tests.

- [ ] **Step 7: Commit:**

```bash
cd ~/dev/glades-ml && git add "Backend/Machine Learning/Networks/cuda/gpu_chiron.h" "Backend/Machine Learning/Networks/cuda/gpu_chiron.cu" "unit-tests/Backend/Machine Learning/chiron-test.cpp"
git commit -m "WhiSC Task 3: GPU whisc_scale + whisc_update_stats kernels + parity test"
```

---

## Task 4: Composite WhiSC backward — finite-difference dphi parity at ρ=45 (the R1 guard)

This is the decisive correctness test: it proves the whitened-frame coupling's `dphi` is correct **and precise** even when `p` is 45× larger than `q` (the production regime that exploded SORC). It validates the whole `whiten ∘ rot ∘ unwhiten` backward composition end-to-end on CPU before any trainer wiring.

**Files:**
- Test: `unit-tests/Backend/Machine Learning/chiron-test.cpp` (add `WhiSCBackwardParityTest`)

**Interfaces:**
- Consumes: `whisc_scale`, `rot_coeffs`, `rot_forward`, `rot_inverse`, `rot_backward` (CPU refs).

- [ ] **Step 1: Write the failing test** `WhiSCBackwardParityTest` (append to `chiron-test.cpp`). It (1) builds inputs with `p ≈ 45·q`, (2) computes `dphi` analytically through the composite backward, (3) compares against central finite differences of a random linear loss. The composite backward mirrors exactly what the trainer will do (Task 7):

```cpp
void WhiSCBackwardParityTest()
{
	const unsigned int T = 8, m = 3;
	const float theta_max = 0.07f, sw = 1.0f, RHO = 45.0f;
	std::vector<float> phi(m), q0(T*m), p0(T*m), a(m), cot_q(T*m), cot_p(T*m);
	for (unsigned i=0;i<m;++i){ phi[i]=0.2f*(float)i-0.3f; a[i]=powf(1.0f/(RHO*RHO),0.25f); } // a=(E[q2]/E[p2])^.25 with E[p2]=RHO^2 E[q2]
	for (unsigned k=0;k<T*m;++k){ q0[k]=std::sin(0.4f*(float)k); p0[k]=RHO*std::cos(0.27f*(float)k); cot_q[k]=std::sin(0.13f*k+1.0f); cot_p[k]=std::cos(0.31f*k); }

	// forward F(q,p)=unwhiten . rot . whiten ; loss L=<cot,(q_out,p_out)>
	// analytic dphi via composite backward:
	std::vector<float> rc_a(m), rc_c(m); float th;
	for (unsigned i=0;i<m;++i) glades::chiron::rot_coeffs(phi[i],theta_max,sw,rc_a[i],rc_c[i],th);
	// d_out = cot ; dR_out = unwhiten-adjoint(cot) = whisc_scale(cot, sign=-1)
	std::vector<float> dq=cot_q, dp=cot_p;
	glades::chiron::whisc_scale(&dq[0], &dp[0], &a[0], -1.0f, T, m);
	// whitened rot-input = whiten(q0,p0)
	std::vector<float> qt=q0, pt=p0;
	glades::chiron::whisc_scale(&qt[0], &pt[0], &a[0], +1.0f, T, m);
	// rot_backward -> da,dc (state adjoints dq,dp overwritten but unused here)
	std::vector<float> dqi(T*m), dpi(T*m), da(m,0.0f), dc(m,0.0f);
	glades::chiron::rot_backward(&dq[0],&dp[0],&qt[0],&pt[0],&rc_a[0],&rc_c[0],T,m,&dqi[0],&dpi[0],&da[0],&dc[0]);
	// coeff chain: a=-tan(th/2), c=sin th, th=sw*tmax*tanh(phi)
	std::vector<float> dphi(m,0.0f);
	for (unsigned i=0;i<m;++i){
		float thi = sw*theta_max*tanhf(phi[i]);
		float dadth = -0.5f/ (cosf(0.5f*thi)*cosf(0.5f*thi)); // d(-tan(th/2))/dth = -1/2 sec^2(th/2)
		float dcdth = cosf(thi);
		float dthdphi = sw*theta_max*(1.0f - tanhf(phi[i])*tanhf(phi[i]));
		dphi[i] = (da[i]*dadth + dc[i]*dcdth) * dthdphi;
	}

	// central finite differences of L wrt phi[i]
	const float h = 1e-3f;
	for (unsigned i=0;i<m;++i){
		float save=phi[i];
		double Lp, Lm;
		for (int s=0; s<2; ++s){
			phi[i] = save + (s==0? h : -h);
			std::vector<float> rca(m), rcc(m); for (unsigned j=0;j<m;++j) glades::chiron::rot_coeffs(phi[j],theta_max,sw,rca[j],rcc[j],th);
			std::vector<float> q=q0,p=p0;
			glades::chiron::whisc_scale(&q[0],&p[0],&a[0],+1.0f,T,m);
			glades::chiron::rot_forward(&q[0],&p[0],&rca[0],&rcc[0],T,m);
			glades::chiron::whisc_scale(&q[0],&p[0],&a[0],-1.0f,T,m);
			double L=0; for (unsigned k=0;k<T*m;++k) L += (double)cot_q[k]*q[k] + (double)cot_p[k]*p[k];
			if (s==0) Lp=L; else Lm=L;
		}
		phi[i]=save;
		float fd = (float)((Lp-Lm)/(2.0*h));
		float rel = std::fabs(dphi[i]-fd) / (std::fabs(fd)+1e-4f);
		ASSERT("WhiSC dphi != finite-difference at rho=45", rel < 2e-2f);
	}
}
```

- [ ] **Step 2: Build and run:**

```
cd ~/dev/glades-ml && sh .configure.sh cuda && cd unit-tests/build && sh .configure.sh cuda
cd ~/dev/glades-ml/unit-tests && bash test.sh chiron-whisc
```
Expected: PASS — `dphi` matches finite differences within 2% even at ρ=45. (If this FAILS with a large relative error, the precision concern from the spec is real for this code path; switch the `dphi` accumulation to the per-element whitened symplectic-area form `Σ_t (q̃' p̄̃' − p̃' q̄̃')` — but the composite-on-balanced-frame path above is expected to pass because `rot_backward` runs on the balanced whitened state, keeping `da/dc` O(1).)

- [ ] **Step 3: Commit:**

```bash
cd ~/dev/glades-ml && git add "unit-tests/Backend/Machine Learning/chiron-test.cpp"
git commit -m "WhiSC Task 4: composite backward dphi vs finite-difference parity at rho=45 (R1 guard)"
```

- [ ] **Step 4: Install the library** so the trainer (Tasks 5-8) picks up the new kernels:

```bash
cd ~/dev/glades-ml/build && make install
```
Expected: installs `libglades.so` + headers to `~/.local`.

---

## Task 5: Trainer config, CLI, mutual-exclusion, scratch buffers

Adds the `--whisc-coupling` flag (which implies the rot machinery + whitening), the whitening hyperparameters, and the per-layer EMA scratch buffers. No behavior change yet beyond flag parsing + allocation; verified by a clean build and a `--help`-style smoke parse.

**Files:**
- Modify: `~/dev/glades-trainer/trainer/chiron_main.cpp` — Config (after line 1074), defaults (after line 1511), CLI (after line 2145), mutual-exclusion (lines 2588-2597), Scratch (after line 7079), allocation (lines 7404-7410).

**Interfaces:**
- Produces: `cfg.rotWhiten` (bool), `cfg.whiscEma` (float), `cfg.whiscClamp` (float); Scratch buffers `s.whisc_Pbar`, `s.whisc_Qbar`, `s.whisc_a` (each `std::vector<glades::gpu::GpuBuffer<float> >` of length `L`, `[m]` each).
- Consumes: existing `cfg.rotCoupling`, `cfg.rotThetaMax`, `cfg.rotWarmup`, `W.rot_phi`.

- [ ] **Step 1: Add Config fields.** In `struct Config`, after line 1074 (`int rotWarmup;`):

```cpp
	bool  rotWhiten;     // WhiSC-D: whiten (q,p) per-channel before the rotation coupling
	float whiscEma;      // EMA decay for the per-channel second-moment stats
	float whiscClamp;    // a clamp: a in [1/whiscClamp, whiscClamp]
```

- [ ] **Step 2: Add constructor defaults.** After line 1511 (`rotCoupling(false), rotThetaMax(...), rotWarmup(0),`):

```cpp
	      rotWhiten(false), whiscEma(0.05f), whiscClamp(8.0f),
```

- [ ] **Step 3: Add CLI parsing.** After line 2145 (the `--rot-warmup` parse):

```cpp
		else if (streq(a, "--whisc-coupling")) { cfg.rotCoupling = true; cfg.rotWhiten = true; }
		else if (streq(a, "--whisc-ema") && i + 1 < argc) cfg.whiscEma = parse_f32(argv[++i], cfg.whiscEma);
		else if (streq(a, "--whisc-clamp") && i + 1 < argc) cfg.whiscClamp = parse_f32(argv[++i], cfg.whiscClamp);
```

(θmax and warmup reuse `--rot-theta-max` / `--rot-warmup`. Recommended run uses `--whisc-coupling --rot-theta-max 0.07`.)

- [ ] **Step 4: Validate hyperparameters.** Add to the config-validation block (near where other ranges are checked, e.g. after line 2471's `siraDebug` validation — search for a nearby `if (cfg.` range check in the validate function):

```cpp
	if (cfg.rotWhiten && (cfg.whiscEma <= 0.0f || cfg.whiscEma > 1.0f)) { log_error("chiron","--whisc-ema must be in (0,1]\n"); return false; }
	if (cfg.rotWhiten && cfg.whiscClamp < 1.0f) { log_error("chiron","--whisc-clamp must be >= 1\n"); return false; }
```

- [ ] **Step 5: Declare scratch buffers.** In `struct Scratch`, after line 7079 (`GpuBuffer<float> rot_scratch_dc;`):

```cpp
	std::vector<glades::gpu::GpuBuffer<float> > whisc_Pbar;  // [L] of [m] : EMA E[p^2]
	std::vector<glades::gpu::GpuBuffer<float> > whisc_Qbar;  // [L] of [m] : EMA E[q^2]
	std::vector<glades::gpu::GpuBuffer<float> > whisc_a;     // [L] of [m] : cached per-step whitening scale
```

- [ ] **Step 6: Allocate + init the buffers.** In the scratch-allocation block, after line 7410 (the `rot_scratch_dc.allocate` inside the `if (cfg.rotCoupling)` guard, before its closing brace):

```cpp
		if (cfg.rotWhiten)
		{
			whisc_Pbar.resize((size_t)L); whisc_Qbar.resize((size_t)L); whisc_a.resize((size_t)L);
			std::vector<float> ones((size_t)m, 1.0f);
			for (int l = 0; l < L; ++l)
			{
				if (!whisc_Pbar[(size_t)l].allocate((size_t)m)) return false;
				if (!whisc_Qbar[(size_t)l].allocate((size_t)m)) return false;
				if (!whisc_a[(size_t)l].allocate((size_t)m)) return false;
				whisc_Pbar[(size_t)l].upload(&ones[0], (size_t)m);  // init E[p^2]=1
				whisc_Qbar[(size_t)l].upload(&ones[0], (size_t)m);  // init E[q^2]=1 -> a=1 (irrelevant at phi=0)
				whisc_a[(size_t)l].upload(&ones[0], (size_t)m);
			}
		}
```

(`L` and `m` are in scope in this allocation function as they are for the `rot_*` allocations above.)

- [ ] **Step 7: Add the mutual-exclusion note.** At the guard block lines 2588-2597 (rot vs drift/fuse), no change is needed since `--whisc-coupling` sets `rotCoupling=true` and inherits that guard. Confirm by reading lines 2588-2597 that the guard keys on `cfg.rotCoupling` (it does). No code change; this step is a verification read.

- [ ] **Step 8: Build and smoke-test the flag parse:**

```bash
cd ~/dev/glades-trainer && bash build.sh 2>&1 | tail -3
./build/glades_chiron_train --help 2>&1 | head -1 || true
```
Expected: build succeeds (`Built target glades_chiron_train`). (The binary may not have `--help`; the goal of this step is a clean compile of the new fields/flags.)

- [ ] **Step 9: Commit:**

```bash
cd ~/dev/glades-trainer && git add trainer/chiron_main.cpp
git commit -m "WhiSC Task 5: trainer config/CLI/scratch for --whisc-coupling (no behavior yet)"
```

---

## Task 6: Trainer forward wiring (stats + whiten/rot/unwhiten) + E0 bit-parity

Inserts, in each of the three forward branches, the per-step stats update and the `whiten → rot → unwhiten` wrap around the existing rot forward. Gated on `cfg.rotWhiten`; when off, the path is the unchanged rot path (and with `--whisc-coupling` off entirely, bit-identical baseline).

**Files:**
- Modify: `~/dev/glades-trainer/trainer/chiron_main.cpp` — the three forward rot blocks (SCFA 13056-13063, bf16w 13208-13215, standard 13344-13351).

**Interfaces:**
- Consumes: `s.q`, `s.p` (post-shear state), `s.whisc_Pbar/Qbar/a`, `cfg.whiscEma/whiscClamp/rotThetaMax/rotWarmup`, `W.rot_phi`, `s.rot_a`, `s.rot_c`.
- Produces (for Task 7): the cached `s.whisc_a[l]` written this step (reused unchanged in backward).

- [ ] **Step 1: Define a forward helper** to avoid triplicating the body. Add a file-scope static function just above the forward function (search for `static bool forward(` near line 12727; insert before it):

```cpp
// WhiSC-D forward: stats update (once/step) + whiten . rot . unwhiten, in place on s.q,s.p.
// Caches s.whisc_a[l] for the backward. Falls back to plain rot when !rotWhiten.
static bool whisc_or_rot_forward(const Config& cfg, ChironParams& W, Scratch& s, int l, int T, int m, int stepForDrift, bool isTraining)
{
	float sw = 1.0f;
	if (cfg.rotWarmup > 0) { sw = (float)stepForDrift / (float)cfg.rotWarmup; if (sw > 1.0f) sw = 1.0f; }
	if (!isTraining) sw = 1.0f;
	if (!glades::gpu::chiron_rot_coeffs(W.rot_phi[l]->data(), cfg.rotThetaMax, sw, m, s.rot_a.data(), s.rot_c.data())) return false;
	if (cfg.rotWhiten)
	{
		// update EMA stats only during training (and only on the real forward, not eval),
		// so the cached a matches what the backward reuses.
		if (isTraining)
		{
			if (!glades::gpu::chiron_whisc_update_stats(s.q.data(), s.p.data(), T, m,
			        cfg.whiscEma, 1e-12f, cfg.whiscClamp,
			        s.whisc_Pbar[(size_t)l].data(), s.whisc_Qbar[(size_t)l].data(), s.whisc_a[(size_t)l].data())) return false;
		}
		const float* aw = s.whisc_a[(size_t)l].data();
		if (!glades::gpu::chiron_whisc_scale(s.q.data(), s.p.data(), aw, +1.0f, T, m)) return false; // whiten
		if (!glades::gpu::chiron_rot_forward(s.q.data(), s.p.data(), s.rot_a.data(), s.rot_c.data(), +1.0f, T, m)) return false;
		if (!glades::gpu::chiron_whisc_scale(s.q.data(), s.p.data(), aw, -1.0f, T, m)) return false; // unwhiten
		return true;
	}
	return glades::gpu::chiron_rot_forward(s.q.data(), s.p.data(), s.rot_a.data(), s.rot_c.data(), +1.0f, T, m);
}
```

- [ ] **Step 2: Replace the SCFA forward block** (lines 13056-13063). Replace the body inside `if (cfg.rotCoupling) { ... }` with a single call:

```cpp
			if (cfg.rotCoupling)
			{
				if (!whisc_or_rot_forward(cfg, W, s, l, T, m, stepForDrift, isTraining)) return false;
			}
```

(`T` and `m` are the local `int T, int m` from the forward function header at line 12731 — pass them directly.)

- [ ] **Step 3: Replace the bf16w forward block** (lines 13208-13215) identically:

```cpp
			if (cfg.rotCoupling)
			{
				if (!whisc_or_rot_forward(cfg, W, s, l, T, m, stepForDrift, isTraining)) return false;
			}
```

- [ ] **Step 4: Replace the standard forward block** (lines 13344-13351) identically:

```cpp
			if (cfg.rotCoupling)
			{
				if (!whisc_or_rot_forward(cfg, W, s, l, T, m, stepForDrift, isTraining)) return false;
			}
```

- [ ] **Step 5: Build, install-free (trainer-only change), and run the E0 bit-parity check.** With `--whisc-coupling` OFF, training must be bit-identical to the baseline. Run 1 step each, compare the step-1 loss exactly:

```bash
cd ~/dev/glades-trainer && bash build.sh 2>&1 | tail -2
# baseline (no whisc): reanchor recipe, 1 step
sh run.sh flagship --steps 1 --seed 1337 --accum 1 --qk-norm --reln-reanchor --sira-coef 1e-2 --sira-warmup 250 2>&1 | grep -E "\[step      1\]" > /tmp/whisc_base.txt
# whisc OFF flag present but not enabled: identical
cat /tmp/whisc_base.txt
```
Expected: a `[step 1] loss=...` line. Record the loss value. (E0 "off" parity is structural here: with `cfg.rotCoupling=false` the new code is never entered.)

- [ ] **Step 6: Run the E0 "on-but-phi=0" identity check.** With `--whisc-coupling` ON but at step 1 (`rot_phi=0 ⇒ θ=0 ⇒ rot=identity ⇒ whiten∘I∘unwhiten=identity`), the loss must equal the baseline:

```bash
cd ~/dev/glades-trainer
sh run.sh flagship --steps 1 --seed 1337 --accum 1 --qk-norm --reln-reanchor --sira-coef 1e-2 --sira-warmup 250 --whisc-coupling --rot-theta-max 0.07 2>&1 | grep -E "\[step      1\]"
```
Expected: the `[step 1] loss=...` value is **identical** (to ~5 sig figs) to `/tmp/whisc_base.txt`. (At step 1 the coupling is identity regardless of `a`, because `θ=0`.) If it differs, the whiten/unwhiten are not exact inverses on-device — debug `chiron_whisc_scale` before proceeding.

- [ ] **Step 7: Commit:**

```bash
cd ~/dev/glades-trainer && git add trainer/chiron_main.cpp
git commit -m "WhiSC Task 6: forward wiring (stats + whiten/rot/unwhiten) + E0 identity check"
```

---

## Task 7: Trainer backward wiring + monitor extension

Mirrors the forward: the backward replaces the rot inverse-walk + `chiron_rot_backward` with the whitened composite (scale the adjoints by `W^{-T}`, whiten the recovered rot-input, call `chiron_rot_backward` in the balanced frame, restore, scale the adjoints by `W^{T}`). Reuses the cached `s.whisc_a[l]` from the forward (no EMA re-update). Extends the `[sorc]` monitor to also log the `a` range.

**Files:**
- Modify: `~/dev/glades-trainer/trainer/chiron_main.cpp` — the backward rot block (lines 14823-14835); the monitor (lines 18493-18513).

**Interfaces:**
- Consumes: `s.whisc_a[l]` (cached in Task 6 forward), `dq_out_ptr`, `s.dp`, `s.q`, `s.p`, `W.rot_phi`, `W.drot_phi`, `s.rot_a/rot_c`, `s.rot_scratch_da/dc`.

- [ ] **Step 1: Add a backward helper** just above the `forward` helper from Task 6 (or directly above `static bool forward(`):

```cpp
// WhiSC-D backward: inverse-walk (recover pre-coupling q,p into s.q,s.p) + adjoint of W^{-1} R W.
// dq,dp are in/out adjoints (in place). Reuses the cached s.whisc_a[l] (no EMA update here).
static bool whisc_or_rot_backward(const Config& cfg, ChironParams& W, Scratch& s, int l, int T, int m,
                                  int stepForDebug, float* dq, float* dp)
{
	float sw = 1.0f;
	if (cfg.rotWarmup > 0) { sw = (float)stepForDebug / (float)cfg.rotWarmup; if (sw > 1.0f) sw = 1.0f; }
	if (!glades::gpu::chiron_rot_coeffs(W.rot_phi[l]->data(), cfg.rotThetaMax, sw, m, s.rot_a.data(), s.rot_c.data())) return false;
	if (cfg.rotWhiten)
	{
		const float* aw = s.whisc_a[(size_t)l].data();
		// inverse-walk: F^{-1} = whiten . rot_inverse . unwhiten  (recovers pre-coupling state in s.q,s.p)
		if (!glades::gpu::chiron_whisc_scale(s.q.data(), s.p.data(), aw, +1.0f, T, m)) return false;
		if (!glades::gpu::chiron_rot_forward(s.q.data(), s.p.data(), s.rot_a.data(), s.rot_c.data(), -1.0f, T, m)) return false;
		if (!glades::gpu::chiron_whisc_scale(s.q.data(), s.p.data(), aw, -1.0f, T, m)) return false;
		// adjoint: dR_out = W^{-T} d_out  (= scale dq*=a, dp/=a, i.e. sign=-1)
		if (!glades::gpu::chiron_whisc_scale(dq, dp, aw, -1.0f, T, m)) return false;
		// whitened rot-input (in place on s.q,s.p)
		if (!glades::gpu::chiron_whisc_scale(s.q.data(), s.p.data(), aw, +1.0f, T, m)) return false;
		if (!glades::gpu::chiron_rot_backward(dq, dp, s.q.data(), s.p.data(),
		        s.rot_a.data(), s.rot_c.data(), W.rot_phi[l]->data(), cfg.rotThetaMax, sw, T, m,
		        dq, dp, W.drot_phi[l]->data(), s.rot_scratch_da.data(), s.rot_scratch_dc.data())) return false;
		// restore pre-coupling state and complete d_input = W^{T} dR_in (= scale dq/=a, dp*=a, sign=+1)
		if (!glades::gpu::chiron_whisc_scale(s.q.data(), s.p.data(), aw, -1.0f, T, m)) return false;
		if (!glades::gpu::chiron_whisc_scale(dq, dp, aw, +1.0f, T, m)) return false;
		return true;
	}
	// plain SORC backward
	if (!glades::gpu::chiron_rot_forward(s.q.data(), s.p.data(), s.rot_a.data(), s.rot_c.data(), -1.0f, T, m)) return false;
	return glades::gpu::chiron_rot_backward(dq, dp, s.q.data(), s.p.data(),
	        s.rot_a.data(), s.rot_c.data(), W.rot_phi[l]->data(), cfg.rotThetaMax, sw, T, m,
	        dq, dp, W.drot_phi[l]->data(), s.rot_scratch_da.data(), s.rot_scratch_dc.data());
}
```

- [ ] **Step 2: Replace the backward rot block** (lines 14823-14835). The surrounding code uses `dq_out_ptr` for the q-adjoint and `s.dp` for the p-adjoint (confirm by reading lines 14823-14835). Replace the `if (cfg.rotCoupling) { ... }` body with:

```cpp
		if (cfg.rotCoupling)
		{
			if (!whisc_or_rot_backward(cfg, W, s, l, T, m, stepForDebug, dq_out_ptr, s.dp.data())) return false;
		}
```

- [ ] **Step 3: Extend the `[sorc]` monitor** (lines 18493-18513) to also report the whitening range. After the existing `log_info("chiron","[sorc] step=%d maxTheta=%.4f thetaMax=%.4f\n", ...)` at line 18510-18511, add (inside the same `if (normalLogStep && cfg.rotCoupling && !W.rot_phi.empty())` block, after the maxTheta log):

```cpp
		if (cfg.rotWhiten && !s.whisc_a.empty())
		{
			std::vector<float> ah((size_t)m);
			float amin = 1e30f, amax = -1e30f;
			for (int l = 0; l < (int)s.whisc_a.size(); ++l)
			{
				s.whisc_a[(size_t)l].download(&ah[0], (size_t)m);
				for (int i = 0; i < m; ++i) { if (ah[(size_t)i] < amin) amin = ah[(size_t)i]; if (ah[(size_t)i] > amax) amax = ah[(size_t)i]; }
			}
			log_info("chiron","[whisc] step=%d a_min=%.4g a_max=%.4g (rho_eff~[%.3g,%.3g])\n",
			         step, (double)amin, (double)amax, 1.0/((double)amax*(double)amax*(double)amax*(double)amax), 1.0/((double)amin*(double)amin*(double)amin*(double)amin));
		}
```

(`a = ρ^{-1/4}` ⇒ `ρ = a^{-4}`, so `a_max → ρ_min` and `a_min → ρ_max`.)

- [ ] **Step 4: Build the trainer and run a 3-step gradient sanity check** with `--whisc-coupling` and a non-zero angle forced (set `rot_phi≠0` is not exposed via CLI, so instead verify `‖g‖` stays bounded and `drot_phi` is finite over a few steps where the optimizer moves `rot_phi` off 0):

```bash
cd ~/dev/glades-trainer && bash build.sh 2>&1 | tail -2
sh run.sh flagship --steps 5 --seed 1337 --accum 1 --qk-norm --reln-reanchor --sira-coef 1e-2 --sira-warmup 250 --whisc-coupling --rot-theta-max 0.07 2>&1 | grep -E "\[step|\[whisc|\[sorc"
```
Expected: 5 `[step N] loss=... ||g||=...` lines with **finite, O(1) ‖g‖** (no NaN, no explosion); a `[whisc]` line showing `a_min/a_max` (a_max≈1 early, a_min dropping as p grows). This is the smoke test that the backward is wired correctly; the real verdict is Task 8.

- [ ] **Step 5: Commit:**

```bash
cd ~/dev/glades-trainer && git add trainer/chiron_main.cpp
git commit -m "WhiSC Task 7: backward wiring (whitened composite) + [whisc] monitor"
```

---

## Task 8: The E3 divergence gate (the pre-registered falsifier)

Run the matched single-seed 2500-step T=16384 production gate that SORC failed. PASS = no divergence.

**Files:** none (run + evaluate).

- [ ] **Step 1: Launch the WhiSC-D gate run** (background; ~3.4 GPU-hr). Use the production reanchor recipe + `--whisc-coupling`:

```bash
cd ~/dev/glades-trainer
nohup env CHIRON_PQ_PROBE= sh run.sh flagship \
  --steps 2500 --seed 1337 --accum 4 --lr 3e-4 --warmup 750 \
  --sira-warmup 250 --zloss-coef 1e-4 --qk-norm \
  --sira-coef 1e-2 --sira-energy-weight 1.0 --sira-balance-weight 0.25 --sira-action-weight 0.0 \
  --grad-clip 0.5 --dq-layer-clamp 1.0 --dq-embed-clamp 1.0 --reln-reanchor \
  --whisc-coupling --rot-theta-max 0.07 \
  > /tmp/whisc_e3.log 2>&1 &
echo "pid=$!"
```

- [ ] **Step 2: Monitor for divergence or completion** (re-arm as needed):

```bash
grep -E "\[step  2500\]|\[val|\[whisc|\[sorc" /tmp/whisc_e3.log | tail -8
grep -ciE "nan|loss-scale|grad-skip|skip" /tmp/whisc_e3.log
```

- [ ] **Step 3: Evaluate against the pre-registered PASS criteria** (spec §13):
  - **val@2500 ≈ baseline 3.58** (NOT ~14.45) — the divergence metric.
  - **‖g‖ ~ O(1)** throughout (NOT 4.7e10), 0–few grad-skips.
  - **`[whisc]`/`[sorc]` monitors**: `maxTheta` bounded (≤ θmax), `drot_phi` grad bounded across depth (no 2.5×/layer cascade — compare `[grad-trace]` per-layer drot_phi if enabled).

  Record the verdict:
  - **PASS** → the whitening fix works; proceed to E4 (30k perplexity gate, separate plan) and/or WhiSC-M if perplexity is flat.
  - **DIVERGE** → the whitening hypothesis is falsified (the asymmetry is not the operative cause); write the NO-GO record and close the cross-depth-coupling direction.

- [ ] **Step 4: Write the result record** to `research/CHIRON_WHISC_D_GATE_2026_MM_DD.md` (date of the run): the trajectory, the `[whisc]` a-range, the val@2500, ‖g‖ history, and the PASS/DIVERGE verdict vs SORC's 14.45/4.7e10. Update `MEMORY.md` and the WhiSC spec's status line.

- [ ] **Step 5: Commit the result record:**

```bash
cd ~/dev/glades-ml && git add research/CHIRON_WHISC_D_GATE_*.md
git commit -m "WhiSC-D E3 gate result: <PASS|DIVERGE>"
```

---

## Self-review notes

**Spec coverage (§ → task):** §5.2 WhiSC-D map → Tasks 1,3,6,7; §6 R1 (whitened isometry, dθ=O(1), the "accumulate in whitened frame" note) → Task 4 (FD parity at ρ=45, on the balanced frame) + the helper structure in Tasks 6/7 (rot_backward runs on whitened state); §7 algorithm (EMA stats, folded/composite forward, frozen-per-step a, inverse-walk) → Tasks 2,5,6,7; §9 init p→0 (clamp) → Task 2 (`whiscClamp`) + Task 5 init `Pbar=Qbar=1`; §13 prototype (flags, checkpoint reuse, E0, E1 dθ test, E3 gate) → Tasks 5 (flags), 1 (E1 reversibility), 4 (E1 dθ), 6 (E0), 8 (E3). Checkpoint persistence is **reused** (rot_phi + bit 1024 unchanged — Tasks deliberately do not touch save/load; a WhiSC run persists rot_phi exactly as SORC does). The `[sorc]` monitor is extended (Task 7) rather than duplicated.

**Deliberate scope limits (YAGNI):** WhiSC-M (2×2 Mahalanobis) and WhiSC-T (thermostat) are NOT in this plan — they are the upgrade path behind the E3 result (spec §14). The EMA stats are not checkpointed (re-warm on resume; irrelevant for a from-scratch 2500-step gate, and `θ=0` at step 0 makes early-`a` immaterial).

**Type consistency check:** `whisc_scale(... float sign ...)`, `whisc_update_stats(... float* Pbar, float* Qbar, float* a)`, `whisc_or_rot_forward(cfg,W,s,l,T,m,stepForDrift,isTraining)`, `whisc_or_rot_backward(cfg,W,s,l,T,m,stepForDebug,dq,dp)` — names/signatures match across Tasks 1-7. The backward adjoint scaling uses `chiron_whisc_scale(dq,dp,a,sign)` with `sign=-1` for `W^{-T}` and `sign=+1` for `W^{T}` (correct because `W=diag(1/a,a)` is symmetric, so `W^{T}=W` scales `q` by `1/a` (sign=+1) and `W^{-T}=W^{-1}` scales `q` by `a` (sign=-1)).

**One open implementation risk (flagged, with the fallback in Task 4 Step 2):** if the composite-frame `dphi` somehow fails the ρ=45 FD parity, switch the `dphi` accumulation to the explicit per-element whitened symplectic-area form. The chosen path (run `chiron_rot_backward` on the already-whitened balanced state) is expected to pass precisely because that keeps `da/dc` O(1).
