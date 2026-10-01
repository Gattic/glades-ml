# CHIRON SORC (Symplectic Orthogonal Rotation Coupling) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-channel, exactly-reversible, **norm-preserving symplectic rotation** of `(q,p)` at every CHIRON layer — realized as 3 shears `R(θ)=S_u(−tan θ/2)·S_l(sin θ)·S_u(−tan θ/2)` with a bounded angle `θ=θ_max·tanh(φ)` — so attention composes across depth with a coupling that **cannot run away** (boundedness by conservation: `‖R‖₂=1`), unlike the OBSD drift it succeeds.

**Architecture:** Replace OBSD's *additive* drift (which injected unbounded magnitude) with a *conservative exchange*. Block order **kick → rotate → reln**. The rotation is computed as a single fused per-element kernel (3 fused shears) that maps `(q,p)→(q',p')`; it is exactly invertible (`R⁻¹=R(−θ)`, the 3 negated shears reversed) so O(1)-in-depth memory holds. Build kernels + CPU refs + unit tests in glades-ml first (mirroring the committed OBSD drift work), then wire into the trainer behind `--rot-coupling` (default off), gate at small shape (E0/E1/E2), then run the production sweep + decisive 30k gate (E3/E4).

**Tech Stack:** C++98 + CUDA. glades-ml (`Backend/Machine Learning/Networks/cuda/gpu_chiron.{h,cu}`, `transformer_chiron_ops.h`, `unit-tests/.../chiron-test.{h,cpp}`), glades-trainer (`trainer/chiron_main.cpp`, `run.sh`).

**Spec:** `docs/superpowers/specs/2026-06-30-chiron-sorc-symplectic-rotation-design.md`

## Global Constraints

- **Exact reversibility:** every q/p update must read only the opposite branch (unit-triangular); inverse reconstructs activations with nothing stored (O(1)-in-depth memory). Verify within BF16 ULP.
- **Boundedness:** the coupling is a rotation (`R(θ)∈SO(2)`, `‖R‖₂=1`) with `θ=θ_max·tanh(φ)`, `θ_max<π/2`. NO clamp/cap on a free parameter, NO unbounded gate.
- **Flagship recovery:** `φ=0 ⇒ θ=0 ⇒ R=I` ⇒ block bit-identical to flagship. Default `--rot-coupling` off; with it on and `φ=0`, identical to flagship.
- **Cheap:** elementwise only; NO new GEMM/attention pass. Target ≤ +10% wall.
- **Determinism preserved** (reductions deterministic, like the OBSD `chiron_col_accumulate`).
- **Reference implementation to mirror:** the committed OBSD drift work on branch `chiron3` (glades-ml) / `reln-reanchor` (glades-trainer): kernels `chiron_drift_into_q`/`chiron_drift_backward` (`gpu_chiron.cu`), CPU refs `drift_into_q`/`drift_backward` (`transformer_chiron_ops.h`), tests `CHIRONDrift*Test` (`chiron-test.cpp`), trainer `a_drift` param + `--per-layer-drift` wiring. SORC's param/flag/checkpoint/insertion-point lifecycle is **identical** to those — read them as the template.

---

## Repository layout & build

- **glades-ml library:** `cd /home/robert/dev/glades-ml && sh .configure.sh cuda`; then `cd build && make install` (trainer links the installed lib).
- **glades-ml unit tests:** `cd /home/robert/dev/glades-ml/unit-tests && sh .configure.sh cuda`; run `bash test.sh <target>` (the drift target name is registered in `unit-tests/main.cpp` — add a `sorc`/`rot` target similarly).
- **trainer:** `cd /home/robert/dev/glades-trainer && sh run.sh` (re-rsyncs headers + builds `glades_chiron_train`). **After any glades-ml change, `make install` first.**

## Key math (verified)

- `R(θ) = [[cos θ, −sin θ],[sin θ, cos θ]] = S_u(a)·S_l(c)·S_u(a)`, `a=−tan(θ/2)`, `c=sin θ`, where `S_u(a):q+=a·p`, `S_l(c):p+=c·q`.
- Forward (per element): `q1=q0+a·p0; p1=p0+c·q1; q2=q1+a·p1`; output `(q2,p1)`.
- Inverse from `(q2,p1)`: `q1=q2−a·p1; p0=p1−c·q1; q0=q1−a·p0`.
- Angle: `θ_eff = s_warm·θ_max·tanh(φ)` (warmup ramps the **angle**); `∂θ/∂φ = s_warm·θ_max·(1−tanh²φ)`, `∂a/∂θ = −½/cos²(θ/2)`, `∂c/∂θ = cos θ`.

## File structure

| File | Responsibility | New/Modify |
|---|---|---|
| `transformer_chiron_ops.h` | CPU ref: `rot_coeffs`, `rot_forward`, `rot_inverse`, `rot_backward` | Modify |
| `cuda/gpu_chiron.h` | Decls: `chiron_rot_coeffs`, `chiron_rot_forward`, `chiron_rot_backward` (+CPU stubs) | Modify |
| `cuda/gpu_chiron.cu` | Kernels: `chiron_rot_coeffs_kernel`, `chiron_rot_forward_rows`, `chiron_rot_backward_rows` + col-reduce reuse + wrappers | Modify |
| `unit-tests/.../chiron-test.{h,cpp}`, `main.cpp` | `CHIRONRot{Compose,Reversibility,CpuGpuParity,NormConserve,BackwardParity,GradCheck}Test` | Modify |
| trainer `chiron_main.cpp` | `rot_phi[l]` param (alloc/Adam/checkpoint), `--rot-coupling`/`--rot-theta-max`/`--rot-warmup`, fwd/inv/bwd wiring, mutual-exclusion guard | Modify |
| trainer `run.sh` | flag pass-through | Modify |
| glades-ml `research/CHIRON_SORC_RESULT_2026_06_30.md` | E0–E4 findings | New |

---

## Task 1: CPU reference (coeffs + rotation fwd/inv/backward) + tests

**Files:** Modify `transformer_chiron_ops.h` (namespace `glades::chiron`, after the OBSD `drift_*` funcs); Test `unit-tests/Backend/Machine Learning/chiron-test.cpp` (+`.h` decl, register in `main.cpp` under a new `rot`/`sorc` target).

**Interfaces — Produces:**
- `void rot_coeffs(float phi, float theta_max, float s_warm, float& a, float& c, float& theta_eff)`
- `void rot_forward(float* q, float* p, const float* a, const float* c, unsigned T, unsigned m)`
- `void rot_inverse(float* q, float* p, const float* a, const float* c, unsigned T, unsigned m)`
- `void rot_backward(const float* dq_out, const float* dp_out, const float* q_in, const float* p_in, const float* a, const float* c, unsigned T, unsigned m, float* dq_in, float* dp_in, float* da, float* dc)` (da,dc ACCUMULATE)

- [ ] **Step 1: Write the failing tests** (declare in `chiron-test.h`, register in `main.cpp`):

```cpp
// (1) the 3 shears compose to R(theta); (2) fwd then inverse reconstructs; (3) norm conserved; (4) FD grad-check.
void CHIRONRotCpuTest()
{
	const unsigned T=4, m=6; const float theta_max=1.0471975512f /*60deg*/, sw=1.0f;
	std::vector<float> phi(m), a(m), c(m); std::vector<float> q(T*m), p(T*m), q0, p0;
	for (unsigned i=0;i<m;++i){ phi[i]=0.3f*(float)i-0.6f; float th; glades::chiron::rot_coeffs(phi[i],theta_max,sw,a[i],c[i],th); }
	for (unsigned k=0;k<T*m;++k){ q[k]=0.5f*sinf(0.7f*k+1.f); p[k]=0.4f*cosf(0.3f*k); }
	q0=q; p0=p;
	// (1)+(3): rotate, check it equals R(theta) per element AND conserves q^2+p^2.
	glades::chiron::rot_forward(&q[0],&p[0],&a[0],&c[0],T,m);
	float maxComposeErr=0.f, maxNormErr=0.f;
	for (unsigned t=0;t<T;++t) for (unsigned i=0;i<m;++i){
		float th=theta_max*tanhf(phi[i]); size_t k=(size_t)t*m+i;
		float qr=cosf(th)*q0[k]-sinf(th)*p0[k], pr=sinf(th)*q0[k]+cosf(th)*p0[k];
		maxComposeErr=fmaxf(maxComposeErr, fmaxf(fabsf(q[k]-qr),fabsf(p[k]-pr)));
		float n0=q0[k]*q0[k]+p0[k]*p0[k], n1=q[k]*q[k]+p[k]*p[k];
		maxNormErr=fmaxf(maxNormErr, fabsf(n1-n0));
	}
	ASSERT("SORC 3-shear composes to R(theta)", maxComposeErr<1e-4f);
	ASSERT("SORC rotation conserves q^2+p^2", maxNormErr<1e-4f);
	// (2): inverse reconstructs.
	glades::chiron::rot_inverse(&q[0],&p[0],&a[0],&c[0],T,m);
	float maxRecon=0.f; for (unsigned k=0;k<T*m;++k) maxRecon=fmaxf(maxRecon, fmaxf(fabsf(q[k]-q0[k]),fabsf(p[k]-p0[k])));
	ASSERT("SORC fwd then inverse reconstructs (q,p)", maxRecon<1e-5f);
	// (4): FD grad-check of dphi (objective J = sum dq_out*q2 + dp_out*p1).
	std::vector<float> dqo(T*m), dpo(T*m); for (unsigned k=0;k<T*m;++k){ dqo[k]=0.2f*cosf(0.5f*k); dpo[k]=0.15f*sinf(0.4f*k); }
	std::vector<float> dqi(T*m,0.f), dpi(T*m,0.f), da(m,0.f), dc(m,0.f);
	glades::chiron::rot_backward(&dqo[0],&dpo[0],&q0[0],&p0[0],&a[0],&c[0],T,m,&dqi[0],&dpi[0],&da[0],&dc[0]);
	// map da,dc -> dphi analytically:
	std::vector<float> dphi(m);
	for (unsigned i=0;i<m;++i){ float th=theta_max*tanhf(phi[i]); float dadth=-0.5f/(cosf(0.5f*th)*cosf(0.5f*th)); float dcdth=cosf(th); float dthdphi=theta_max*(1.f-tanhf(phi[i])*tanhf(phi[i])); dphi[i]=(da[i]*dadth+dc[i]*dcdth)*dthdphi; }
	const float h=1e-3f; float maxRel=0.f;
	for (unsigned j=0;j<m;++j){
		float save=phi[j]; float aj,cj,th;
		#define ROTJ(PH) ({ std::vector<float> qq=q0, pp=p0; std::vector<float> av=a, cv=c; glades::chiron::rot_coeffs((PH),theta_max,sw,av[j],cv[j],th); glades::chiron::rot_forward(&qq[0],&pp[0],&av[0],&cv[0],T,m); double J=0.0; for(unsigned k=0;k<T*m;++k) J+=(double)dqo[k]*qq[k]+(double)dpo[k]*pp[k]; J; })
		double Jp=ROTJ(save+h), Jm=ROTJ(save-h); phi[j]=save; (void)aj;(void)cj;
		float fd=(float)((Jp-Jm)/(2.0*h)); float e=fabsf(fd-dphi[j])/(1e-3f+fabsf(fd)); if(e>maxRel)maxRel=e;
		#undef ROTJ
	}
	char msg[128]; std::snprintf(msg,sizeof(msg),"SORC rot backward FD grad-check (maxRel=%.4f)",maxRel);
	ASSERT(msg, maxRel<2e-2f);
}
```
(If the GCC statement-expression `ROTJ` doesn't compile, hoist it to a `static double rotJ(...)` helper — the OBSD Task-1 implementer used the helper form; match it.)

- [ ] **Step 2: Run to verify it fails** (undefined `rot_*`): `cd /home/robert/dev/glades-ml/unit-tests/build && sh .configure.sh cuda 2>&1 | tail -5`. Expected: undefined-reference / undeclared.

- [ ] **Step 3: Implement the CPU refs** in `transformer_chiron_ops.h` (namespace `glades::chiron`, after the `drift_*` funcs; ensure `<cmath>`/`<vector>` available):

```cpp
// SORC: per-channel symplectic rotation coupling (CPU reference).
inline void rot_coeffs(float phi, float theta_max, float s_warm, float& a, float& c, float& theta_eff) {
	theta_eff = s_warm * theta_max * tanhf(phi);
	a = -tanf(0.5f * theta_eff);
	c = sinf(theta_eff);
}
// Forward: (q,p) <- R(theta)(q,p) per element, via 3 shears. a,c are per-channel [m], broadcast over t.
inline void rot_forward(float* q, float* p, const float* a, const float* c, unsigned int T, unsigned int m) {
	for (unsigned t=0;t<T;++t) for (unsigned i=0;i<m;++i) {
		unsigned long k=(unsigned long)t*m+i; float qv=q[k], pv=p[k], ai=a[i], ci=c[i];
		qv = qv + ai*pv;   // shear1: q += a*p
		pv = pv + ci*qv;   // shear2: p += c*q
		qv = qv + ai*pv;   // shear3: q += a*p
		q[k]=qv; p[k]=pv;
	}
}
// Inverse from (q2,p1): recover (q0,p0).
inline void rot_inverse(float* q, float* p, const float* a, const float* c, unsigned int T, unsigned int m) {
	for (unsigned t=0;t<T;++t) for (unsigned i=0;i<m;++i) {
		unsigned long k=(unsigned long)t*m+i; float qv=q[k], pv=p[k], ai=a[i], ci=c[i];
		qv = qv - ai*pv;   // q1 = q2 - a*p1
		pv = pv - ci*qv;   // p0 = p1 - c*q1
		qv = qv - ai*pv;   // q0 = q1 - a*p0
		q[k]=qv; p[k]=pv;
	}
}
// Backward: given output adjoints (dq_out=dL/dq2, dp_out=dL/dp1) and the rotation INPUT (q_in=q0,p_in=p0),
// produce input adjoints (dq_in,dp_in) and accumulate da,dc (per channel). Recomputes q1,p1 from inputs.
inline void rot_backward(const float* dq_out, const float* dp_out, const float* q_in, const float* p_in,
                         const float* a, const float* c, unsigned int T, unsigned int m,
                         float* dq_in, float* dp_in, float* da, float* dc) {
	for (unsigned t=0;t<T;++t) for (unsigned i=0;i<m;++i) {
		unsigned long k=(unsigned long)t*m+i; float ai=a[i], ci=c[i];
		float q0=q_in[k], p0=p_in[k];
		float q1=q0+ai*p0;          // shear1
		float p1=p0+ci*q1;          // shear2  (q2 not needed; dq_out is adjoint of q2)
		float dq2=dq_out[k], dp1=dp_out[k];
		// shear3 bwd: q2=q1+a*p1
		float dq1=dq2; float dp1_acc=dp1 + ai*dq2; float da_el=p1*dq2;
		// shear2 bwd: p1=p0+c*q1
		float dp0=dp1_acc; dq1 += ci*dp1_acc; float dc_el=q1*dp1_acc;
		// shear1 bwd: q1=q0+a*p0
		float dq0=dq1; dp0 += ai*dq1; da_el += p0*dq1;
		dq_in[k]=dq0; dp_in[k]=dp0;
		da[i]+=da_el; dc[i]+=dc_el;
	}
}
```

- [ ] **Step 4: Run the tests, verify PASS:** `cd /home/robert/dev/glades-ml/unit-tests/build && sh .configure.sh cuda 2>&1 | tail -3 && cd .. && bash test.sh rot 2>&1 | grep -iE "SORC|Success|FAIL"`. Expected: all 4 SORC CPU asserts pass (compose<1e-4, norm<1e-4, recon<1e-5, gradcheck<2e-2).

- [ ] **Step 5: Commit:**
```bash
cd /home/robert/dev/glades-ml
git add "Backend/Machine Learning/Networks/transformer_chiron_ops.h" "unit-tests/Backend/Machine Learning/chiron-test.cpp" "unit-tests/Backend/Machine Learning/chiron-test.h" unit-tests/main.cpp
git commit -m "CHIRON SORC: CPU reference rotation coeffs/fwd/inv/backward + compose/norm/recon/gradcheck tests"
```

---

## Task 2: GPU forward/inverse rotation kernel + coeffs + tests

**Files:** Modify `cuda/gpu_chiron.h` (decls + CPU stubs), `cuda/gpu_chiron.cu` (kernels + wrappers); Test `chiron-test.cpp`.

**Interfaces — Produces (in `namespace glades::gpu`):**
- `bool chiron_rot_coeffs(const float* phi, float theta_max, float s_warm, int m, float* a, float* c)`
- `bool chiron_rot_forward(float* q, float* p, const float* a, const float* c, float sign, int T, int m)` (`sign=+1` forward, `−1` inverse)

- [ ] **Step 1: Write the failing GPU parity + reversibility test** (model GpuBuffer usage on the OBSD `CHIRONDriftCpuGpuParityTest`; remember `GpuBuffer<float>` needs `.allocate(n)` before `.upload`):
```cpp
void CHIRONRotGpuParityTest()
{
	const int T=5,m=8; const float theta_max=1.0471975512f, sw=0.9f;
	std::vector<float> phi(m), a(m), c(m), q(T*m), p(T*m);
	for(int i=0;i<m;++i){ phi[i]=0.25f*i-0.7f; float th; glades::chiron::rot_coeffs(phi[i],theta_max,sw,a[i],c[i],th); }
	for(int k=0;k<T*m;++k){ q[k]=0.4f*sinf(0.5f*k); p[k]=0.3f*cosf(0.2f*k); }
	std::vector<float> qc=q, pc=p; glades::chiron::rot_forward(&qc[0],&pc[0],&a[0],&c[0],T,m); // CPU
	glades::gpu::GpuBuffer<float> dPhi,dA,dC,dQ,dP;
	dPhi.allocate(m); dPhi.upload(&phi[0],m); dA.allocate(m); dC.allocate(m); dQ.allocate(T*m); dQ.upload(&q[0],T*m); dP.allocate(T*m); dP.upload(&p[0],T*m);
	glades::gpu::chiron_rot_coeffs(dPhi.data(),theta_max,sw,m,dA.data(),dC.data());
	glades::gpu::chiron_rot_forward(dQ.data(),dP.data(),dA.data(),dC.data(),+1.f,T,m);
	std::vector<float> qg(T*m),pg(T*m); dQ.download(&qg[0],T*m); dP.download(&pg[0],T*m);
	float me=0.f; for(int k=0;k<T*m;++k) me=fmaxf(me,fmaxf(fabsf(qg[k]-qc[k]),fabsf(pg[k]-pc[k])));
	char msg[128]; std::snprintf(msg,sizeof(msg),"SORC rot fwd CPU/GPU parity (maxErr=%.2e)",me); ASSERT(msg, me<1e-4f);
	// reversibility: forward then inverse on GPU reconstructs.
	glades::gpu::chiron_rot_forward(dQ.data(),dP.data(),dA.data(),dC.data(),-1.f,T,m); // inverse (sign=-1)
	dQ.download(&qg[0],T*m); dP.download(&pg[0],T*m);
	float mr=0.f; for(int k=0;k<T*m;++k) mr=fmaxf(mr,fmaxf(fabsf(qg[k]-q[k]),fabsf(pg[k]-p[k])));
	std::snprintf(msg,sizeof(msg),"SORC rot fwd∘inv reconstructs (maxErr=%.2e)",mr); ASSERT(msg, mr<1e-5f);
}
```

- [ ] **Step 2: Run to verify failure** (undeclared `chiron_rot_*`).

- [ ] **Step 3: Declare wrappers in `gpu_chiron.h`** (CUDA section + no-CUDA stubs returning false):
```cpp
bool chiron_rot_coeffs(const float* phi, float theta_max, float s_warm, int m, float* a, float* c);
bool chiron_rot_forward(float* q, float* p, const float* a, const float* c, float sign, int T, int m);
```

- [ ] **Step 4: Implement kernels + wrappers in `gpu_chiron.cu`** (anonymous namespace + outside):
```cpp
__global__ void chiron_rot_coeffs_kernel(const float* __restrict__ phi, float theta_max, float s_warm, int m,
                                         float* __restrict__ a, float* __restrict__ c) {
	int i = blockIdx.x*blockDim.x + threadIdx.x; if (i>=m) return;
	float th = s_warm*theta_max*tanhf(phi[i]);
	a[i] = -tanf(0.5f*th); c[i] = sinf(th);
}
// sign=+1 forward (3 shears), sign=-1 inverse (3 negated shears reversed). One block-stride over T*m.
__global__ void chiron_rot_forward_rows(float* __restrict__ q, float* __restrict__ p,
                                        const float* __restrict__ a, const float* __restrict__ c,
                                        float sign, int T, int m) {
	long n=(long)T*m;
	for (long k=blockIdx.x*(long)blockDim.x+threadIdx.x; k<n; k+=(long)gridDim.x*blockDim.x) {
		int i=k%m; float qv=q[k], pv=p[k], ai=a[i], ci=c[i];
		if (sign>0.f) { qv+=ai*pv; pv+=ci*qv; qv+=ai*pv; }       // forward
		else          { qv-=ai*pv; pv-=ci*qv; qv-=ai*pv; }       // inverse
		q[k]=qv; p[k]=pv;
	}
}
```
```cpp
bool chiron_rot_coeffs(const float* phi, float theta_max, float s_warm, int m, float* a, float* c) {
	if (m<=0) return true; int blk=256, grd=(m+blk-1)/blk;
	chiron_rot_coeffs_kernel<<<grd,blk,0,computeStream()>>>(phi,theta_max,s_warm,m,a,c);
	GLADES_CUDA_CHECK(cudaGetLastError()); return true;
}
bool chiron_rot_forward(float* q, float* p, const float* a, const float* c, float sign, int T, int m) {
	if (T<=0||m<=0) return true; long n=(long)T*m; int blk=256; int grd=(int)((n+blk-1)/blk); if(grd>65535)grd=65535;
	chiron_rot_forward_rows<<<grd,blk,0,computeStream()>>>(q,p,a,c,sign,T,m);
	GLADES_CUDA_CHECK(cudaGetLastError()); return true;
}
```

- [ ] **Step 5: Build lib + install + tests + run:**
```bash
cd /home/robert/dev/glades-ml && sh .configure.sh cuda 2>&1 | tail -3 && cd build && make install 2>&1 | tail -2
cd /home/robert/dev/glades-ml/unit-tests && sh .configure.sh cuda 2>&1 | tail -3 && bash test.sh rot 2>&1 | grep -iE "SORC|FAIL|Success"
```
Expected: parity (<1e-4) and reversibility (<1e-5) pass; Task-1 CPU tests still pass.

- [ ] **Step 6: Commit:**
```bash
cd /home/robert/dev/glades-ml
git add "Backend/Machine Learning/Networks/cuda/gpu_chiron.cu" "Backend/Machine Learning/Networks/cuda/gpu_chiron.h" "unit-tests/Backend/Machine Learning/chiron-test.cpp" "unit-tests/Backend/Machine Learning/chiron-test.h" unit-tests/main.cpp
git commit -m "CHIRON SORC: GPU rotation coeffs + fused fwd/inv rotation kernel + parity/reversibility tests"
```

---

## Task 3: GPU rotation backward + coeff-chain (dphi) + parity

**Files:** Modify `cuda/gpu_chiron.h`, `cuda/gpu_chiron.cu`; Test `chiron-test.cpp`.

**Interfaces — Produces:**
- `bool chiron_rot_backward(const float* dq_out, const float* dp_out, const float* q_in, const float* p_in, const float* a, const float* c, const float* phi, float theta_max, float s_warm, int T, int m, float* dq_in, float* dp_in, float* dphi, float* scratch_da, float* scratch_dc)` — `dphi` ACCUMULATES; `scratch_da/dc` are `m`-float scratch.

Internally: (1) a per-element kernel writes `dq_in,dp_in` and per-element `da_el,dc_el` into two `T×m` scratch buffers; (2) deterministic column-reduce (`chiron_col_accumulate`) → `da[m],dc[m]`; (3) a small kernel maps `(da,dc,phi)→dphi` via the chain.

- [ ] **Step 1: Write the failing backward-parity test** (CPU `rot_backward`+chain as oracle):
```cpp
void CHIRONRotBackwardParityTest()
{
	const int T=6,m=8; const float theta_max=1.0471975512f, sw=0.8f;
	std::vector<float> phi(m),a(m),c(m),q(T*m),p(T*m),dqo(T*m),dpo(T*m);
	for(int i=0;i<m;++i){ phi[i]=0.2f*i-0.5f; float th; glades::chiron::rot_coeffs(phi[i],theta_max,sw,a[i],c[i],th); }
	for(int k=0;k<T*m;++k){ q[k]=0.4f*sinf(0.5f*k+0.2f); p[k]=0.3f*cosf(0.3f*k); dqo[k]=0.15f*cosf(0.4f*k); dpo[k]=0.12f*sinf(0.35f*k); }
	// CPU oracle: rot_backward -> da,dc -> dphi.
	std::vector<float> dqi(T*m,0.f),dpi(T*m,0.f),da(m,0.f),dc(m,0.f),dphic(m,0.f);
	glades::chiron::rot_backward(&dqo[0],&dpo[0],&q[0],&p[0],&a[0],&c[0],T,m,&dqi[0],&dpi[0],&da[0],&dc[0]);
	for(int i=0;i<m;++i){ float th=theta_max*tanhf(phi[i]); float dadth=-0.5f/(cosf(0.5f*th)*cosf(0.5f*th)); float dcdth=cosf(th); float dthdphi=sw*theta_max*(1.f-tanhf(phi[i])*tanhf(phi[i])); dphic[i]=(da[i]*dadth+dc[i]*dcdth)*dthdphi; }
	// GPU.
	glades::gpu::GpuBuffer<float> dPhi,dA,dC,dQ,dP,dDQO,dDPO,dDQI,dDPI,dDPHI,dSda,dSdc;
	dPhi.allocate(m);dPhi.upload(&phi[0],m); dA.allocate(m);dC.allocate(m);
	glades::gpu::chiron_rot_coeffs(dPhi.data(),theta_max,sw,m,dA.data(),dC.data());
	dQ.allocate(T*m);dQ.upload(&q[0],T*m); dP.allocate(T*m);dP.upload(&p[0],T*m);
	dDQO.allocate(T*m);dDQO.upload(&dqo[0],T*m); dDPO.allocate(T*m);dDPO.upload(&dpo[0],T*m);
	dDQI.allocate(T*m);dDPI.allocate(T*m); dDPHI.allocate(m);dDPHI.zero(m); dSda.allocate(m);dSdc.allocate(m);
	glades::gpu::chiron_rot_backward(dDQO.data(),dDPO.data(),dQ.data(),dP.data(),dA.data(),dC.data(),dPhi.data(),theta_max,sw,T,m,dDQI.data(),dDPI.data(),dDPHI.data(),dSda.data(),dSdc.data());
	std::vector<float> dqig(T*m),dpig(T*m),dphig(m); dDQI.download(&dqig[0],T*m); dDPI.download(&dpig[0],T*m); dDPHI.download(&dphig[0],m);
	float me=0.f; for(int k=0;k<T*m;++k) me=fmaxf(me,fmaxf(fabsf(dqig[k]-dqi[k]),fabsf(dpig[k]-dpi[k])));
	for(int i=0;i<m;++i) me=fmaxf(me,fabsf(dphig[i]-dphic[i]));
	char msg[128]; std::snprintf(msg,sizeof(msg),"SORC rot backward CPU/GPU parity (maxErr=%.2e)",me); ASSERT(msg, me<2e-4f);
}
```

- [ ] **Step 2: Run to verify failure** (undeclared `chiron_rot_backward`).

- [ ] **Step 3: Declare in `gpu_chiron.h`** (+ stub).

- [ ] **Step 4: Implement in `gpu_chiron.cu`** — pre-backward kernel (writes dq_in,dp_in + da_el,dc_el scratch), reuse `chiron_col_accumulate` for da,dc, then the chain kernel:
```cpp
__global__ void chiron_rot_pre_backward_rows(const float* __restrict__ dqo, const float* __restrict__ dpo,
        const float* __restrict__ q_in, const float* __restrict__ p_in,
        const float* __restrict__ a, const float* __restrict__ c, int T, int m,
        float* __restrict__ dqi, float* __restrict__ dpi, float* __restrict__ da_el, float* __restrict__ dc_el) {
	long n=(long)T*m;
	for (long k=blockIdx.x*(long)blockDim.x+threadIdx.x; k<n; k+=(long)gridDim.x*blockDim.x) {
		int i=k%m; float ai=a[i], ci=c[i]; float q0=q_in[k], p0=p_in[k];
		float q1=q0+ai*p0; float p1=p0+ci*q1;
		float dq2=dqo[k], dp1=dpo[k];
		float dq1=dq2; float dp1a=dp1+ai*dq2; float dael=p1*dq2;       // shear3 bwd
		float dp0=dp1a; dq1+=ci*dp1a; float dcel=q1*dp1a;             // shear2 bwd
		float dq0=dq1; dp0+=ai*dq1; dael+=p0*dq1;                     // shear1 bwd
		dqi[k]=dq0; dpi[k]=dp0; da_el[k]=dael; dc_el[k]=dcel;
	}
}
__global__ void chiron_rot_chain_kernel(const float* __restrict__ da, const float* __restrict__ dc,
        const float* __restrict__ phi, float theta_max, float s_warm, int m, float* __restrict__ dphi) {
	int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=m) return;
	float th=s_warm*theta_max*tanhf(phi[i]);
	float dadth=-0.5f/(cosf(0.5f*th)*cosf(0.5f*th)); float dcdth=cosf(th);
	float dthdphi=s_warm*theta_max*(1.f-tanhf(phi[i])*tanhf(phi[i]));
	dphi[i] += (da[i]*dadth + dc[i]*dcdth)*dthdphi;   // ACCUMULATE
}
```
Wrapper `chiron_rot_backward`: allocate two `T×m` scratch (`da_el,dc_el`) via the device allocator (as OBSD's drift backward does for `du/sdq`), launch pre-backward, then `chiron_col_accumulate(da_el,T,m,scratch_da)` and `chiron_col_accumulate(dc_el,T,m,scratch_dc)` (zero `scratch_da/dc` first), then `chiron_rot_chain_kernel(scratch_da,scratch_dc,phi,theta_max,s_warm,m,dphi)`. (Confirm `chiron_col_accumulate` does `out[j]+=`; if so, zero `scratch_da/dc` before; mirror OBSD's usage at `gpu_chiron.cu`.)

- [ ] **Step 5: Build/install/test:** as Task 2 Step 5. Expected: backward parity <2e-4; all prior SORC tests pass.

- [ ] **Step 6: Commit:**
```bash
cd /home/robert/dev/glades-ml
git add "Backend/Machine Learning/Networks/cuda/gpu_chiron.cu" "Backend/Machine Learning/Networks/cuda/gpu_chiron.h" "unit-tests/Backend/Machine Learning/chiron-test.cpp" "unit-tests/Backend/Machine Learning/chiron-test.h" unit-tests/main.cpp
git commit -m "CHIRON SORC: GPU rotation backward (per-element + col-reduce + coeff-chain to dphi) + parity test"
```

---

## Task 4: Trainer `rot_phi[l]` param (alloc/init0/Adam/checkpoint) + flags

**Files:** Modify `/home/robert/dev/glades-trainer/trainer/chiron_main.cpp`, `run.sh`. **Mirror the `a_drift` lifecycle exactly** (search `a_drift`): struct decl + Adam moments, `rot_phi_grad`, allocation init **0.0**, per-step grad-zero, Adam update, checkpoint save/load under a NEW bit (next free after 512, e.g. **1024** `has_rot_phi`), load-absent→zero.

- [ ] **Step 1:** Add `rot_phi[l] ∈ ℝ^m` (init 0) + `rot_phi_grad` + Adam state, mirroring `a_drift`. Allocate under `(cfg.rotCoupling || …)` — but for Task 4, allocate wherever `a_drift` is allocated; Task 5 broadens the gate to `cfg.rotCoupling`. Add flags: `--rot-coupling` (`cfg.rotCoupling=true`), `--rot-theta-max F` (`cfg.rotThetaMax=parse_f32(...)`, default `1.0471975512f`≈π/3), `--rot-warmup N` (`cfg.rotWarmup`). Declare `bool rotCoupling; float rotThetaMax; int rotWarmup;` defaulted `false/π·...·/0`.
- [ ] **Step 2:** Checkpoint: persist `rot_phi[l]` under bit 1024; load when present else leave zero-init (→ flagship).
- [ ] **Step 3:** Build: `cd /home/robert/dev/glades-trainer && sh run.sh 2>&1 | tail -5`. Expected: clean build. Smoke a tiny run to confirm alloc doesn't crash.
- [ ] **Step 4:** Commit (`trainer/chiron_main.cpp`): `CHIRON SORC: add per-layer rot_phi param (init 0, Adam, checkpoint bit 1024) + flags`.

---

## Task 5: Trainer forward wiring + mutual-exclusion + E0 bit-identity

**Files:** Modify `chiron_main.cpp`, `run.sh`.

- [ ] **Step 1:** Broaden the `gamma_p/beta_p/a_drift` allocation + grad-zero + Adam guards to also fire on `cfg.rotCoupling` (so `rot_phi` allocates on the flagship path). Add a mutual-exclusion guard (next to the OBSD `perLayerDrift && fuseAttnPerLayer` guard): error if `cfg.rotCoupling && (cfg.perLayerDrift || cfg.fuseAttnPerLayer)` (they share the q→p insertion slot).
- [ ] **Step 2:** Wire the forward rotation at every layer, at the SAME insertion point as the OBSD drift (after the attention shear / before the q-side reln) in all attention branches, behind `if (cfg.rotCoupling)`. Compute coeffs then rotate:
```cpp
if (cfg.rotCoupling)
{
	float sw = 1.0f;
	if (cfg.rotWarmup > 0) { sw = (float)stepForDrift / (float)cfg.rotWarmup; if (sw > 1.0f) sw = 1.0f; }
	if (!isTraining) sw = 1.0f;   // full strength in val (mirror OBSD's val-mode fix)
	if (!glades::gpu::chiron_rot_coeffs(W.rot_phi[l]->data(), cfg.rotThetaMax, sw, m,
	        s.rot_a.data(), s.rot_c.data())) return false;            // s.rot_a, s.rot_c: per-run m-float scratch (alloc under rotCoupling)
	if (!glades::gpu::chiron_rot_forward(s.q.data(), s.p.data(), s.rot_a.data(), s.rot_c.data(),
	        /*sign=*/+1.0f, T, m)) return false;
}
```
Allocate `s.rot_a, s.rot_c` (`m` floats each) under `cfg.rotCoupling`.
- [ ] **Step 3:** `run.sh` flag pass-through (mirror `--per-layer-drift`/`--drift-warmup`: add `--rot-coupling`, `--rot-theta-max`, `--rot-warmup`).
- [ ] **Step 4:** Build + **E0 bit-identity gate** (φ=0 ⇒ rotation = identity ⇒ loss bit-identical with/without `--rot-coupling`), on the non-SCFA path at a valid small shape (nH·dH=2·m), e.g.:
```bash
cd /home/robert/dev/glades-trainer && sh run.sh 2>&1 | tail -3
COMMON="--pretokenized --data-dir pretok-data/ --vocab 32000 --seq-len 256 --m 256 --layers 4 --heads 4 --dhead 128 --max-steps 3 --seed 1337 --no-fuse-attn --fuse-attn-reln"
./build/glades_chiron_train $COMMON 2>&1 | grep -E "step " | sed 's/wall=.*//' > /tmp/sorc_base.txt
./build/glades_chiron_train $COMMON --rot-coupling 2>&1 | grep -E "step " | sed 's/wall=.*//' > /tmp/sorc_rot0.txt
diff /tmp/sorc_base.txt /tmp/sorc_rot0.txt && echo "E0 PASS: bit-identical at phi=0"
```
Note: with the backward unwired (Task 6) and φ=0, dphi stays 0 so Adam leaves φ at 0 → identity holds (same logic as OBSD E0). Expected: bit-identical losses.
- [ ] **Step 5:** Commit: `CHIRON SORC: --rot-coupling forward wiring + mutual-exclusion guard; E0 bit-identity at phi=0`.

---

## Task 6: Trainer inverse-walk + backward wiring + E2 reconstruction

**Files:** Modify `chiron_main.cpp`. Mirror the OBSD inverse/backward insertion sites (search `perLayerDrift` near the inverse walk and backward).

- [ ] **Step 1:** Inverse walk: after reln-inverse recovers q and before the attention-shear inverse, recompute coeffs (same `sw`) and call `chiron_rot_forward(s.q, s.p, rot_a, rot_c, /*sign=*/-1.0f, T, m)` to undo the rotation (recovers pre-rotation (q,p)).
- [ ] **Step 2:** Backward: at the OBSD-drift backward site, after the q-reln backward yields the adjoints `dq,dp` (w.r.t. the rotation outputs) and using the reconstructed (q,p), call:
```cpp
if (cfg.rotCoupling)
{
	float sw = 1.0f; if (cfg.rotWarmup>0){ sw=(float)stepForDebug/(float)cfg.rotWarmup; if(sw>1.f)sw=1.f; }
	if (!glades::gpu::chiron_rot_backward(s.dq.data(), s.dp.data(), s.q.data(), s.p.data(),
	        s.rot_a.data(), s.rot_c.data(), W.rot_phi[l]->data(), cfg.rotThetaMax, sw, T, m,
	        s.dq.data(), s.dp.data(), W.rot_phi_grad[l]->data(), s.rot_scratch_da.data(), s.rot_scratch_dc.data())) return false;
}
```
Note `chiron_rot_backward` writes `dq_in,dp_in` — pass `s.dq,s.dp` as BOTH in and out (the rotation backward transforms the adjoints in place: `(dq,dp) ← R(θ)ᵀ(dq,dp)`). Recompute `rot_a,rot_c` first (or reuse if still valid for this layer). Allocate `s.rot_scratch_da/dc` (m floats). Ensure `rot_phi_grad` is zeroed per accum-window (Task 4) and `rot_phi` is in the Adam update + grad-norm/clip.
- [ ] **Step 3:** Build + **E2 reconstruction** (rotation active): force φ to learn (`--rot-warmup 5`, ~30 steps), confirm 0 grad-skips, finite loss, reconstruction within tolerance, and φ moves from 0:
```bash
cd /home/robert/dev/glades-trainer && sh run.sh 2>&1 | tail -3
./build/glades_chiron_train --pretokenized --data-dir pretok-data/ --vocab 32000 --seq-len 512 --m 256 --layers 8 --heads 4 --dhead 128 --max-steps 30 --seed 1337 --no-fuse-attn --fuse-attn-reln --rot-coupling --rot-warmup 5 2>&1 | grep -iE "step|nan|inf|skip|recon" | tail
```
Expected: finite decreasing loss, 0 grad-skips, no NaN/Inf.
- [ ] **Step 4:** Commit: `CHIRON SORC: inverse-walk + backward wiring (R^T adjoint), val full-strength; E2 reconstruction`.

---

## Task 7: Small-shape validation (E0/E1/E2) + budget log + findings doc

**Files:** Modify `chiron_main.cpp` (optional `[sorc]` log: max|θ| per layer = `θ_max·max|tanh φ|`); Create `research/CHIRON_SORC_RESULT_2026_06_30.md`.

- [ ] **Step 1:** Add a `[sorc]` log (under `cfg.rotCoupling`, log-every cadence): `max_l max_i |θ_{l,i}|` (download `rot_phi`, compute `θ_max·|tanh φ|`) — the plateau monitor (must stay ≤ θ_max, vs OBSD's maxA→1.8).
- [ ] **Step 2:** SCFA-path validation at a working shape (full bf16 stack, `--bf16-weights`): confirm SORC runs through the production SCFA path (the OBSD-era `--scfa requires --bf16-weights` guard is already in place). E0-SCFA: loss bit-identical at φ=0; short run with `--rot-coupling`: 0 grad-skips, `[sorc]` max|θ| ≤ θ_max.
- [ ] **Step 3:** Write `research/CHIRON_SORC_RESULT_2026_06_30.md`: E0/E1/E2 results (bit-identity, reversibility, norm-conservation, reconstruction, max|θ| plateau), the commands, verdict (proceed to production / stop).
- [ ] **Step 4:** Commit (glades-ml): `CHIRON SORC: [sorc] max|theta| plateau log + E0/E1/E2 findings`.

---

## Task 8: Production E3/E4 — the decisive gate (NEEDS USER GO-AHEAD for long runs)

**Files:** Modify `research/CHIRON_SORC_RESULT_2026_06_30.md`; `CLAUDE.md` + `MEMORY.md` (on outcome).

- [ ] **Step 1: E3 θ_max sweep** (production T=16384, full flagship recipe + `--rot-coupling`, ≤5k steps each, θ_max ∈ {π/6, π/3, 1.4}): record val vs the matched baseline trajectory, `[sorc]` max|θ| (must plateau), 0-skip. Pick θ_max*.
- [ ] **Step 2: E4 decisive 30k** (single-seed 1337, matched baseline, θ_max*): **Ship/validate iff** val NLL < baseline with a **non-shrinking gap through 24–30k**, max|θ| plateaued, 0 grad-skips, wall ≤ +10%, reconstruction within BF16 ULP. **The decisive test:** if the gap reverses *despite* max|θ| plateauing, that falsifies "gate magnitude was the OBSD problem" and implicates cross-depth composition itself — record either way.
- [ ] **Step 3: Multi-seed** `{2024,4242}` capped 5–15k (sign + cross-seed plateau).
- [ ] **Step 4: Document & close:** finalize the research doc; update `CLAUDE.md` (ship section if positive, or NO-GO note if negative) + `MEMORY.md` pointer linking `[[obsd_per_layer_coupling_nogo]]`.
- [ ] **Step 5: Commit.**

---

## Self-Review

**Spec coverage:**
- §5.2 3-shear `R(θ)` → Task 1/2 (compose test verifies it) ✓
- §5.3 forward kick→rotate→reln, angle-ramped warmup → Task 5 ✓
- §5.4 inverse (negated shears) → Task 2 (sign=−1) + Task 6 ✓
- §5.5 backward (adjoint shears + coeff chain) → Task 3 ✓
- §6 flagship recovery at φ=0 → Task 5 E0 ✓
- §7.1 conservation (‖R‖=1) → Task 1/2 norm-conservation test ✓
- §7.2 bounded angle θ=θ_max·tanh(φ) → coeffs (Task 1/2), plateau monitor (Task 7) ✓
- §7.4 reversibility/O(1) → Task 2 reversibility + Task 6 E2 ✓
- §11 kernel/param reuse → Tasks 1–6 mirror OBSD ✓
- §12 E0–E4 ladder + decisive test → Tasks 5,7,8 ✓
- §13 ship gate → Task 8 ✓

**Placeholder scan:** kernels/CPU-refs/tests have complete code. Trainer wiring (Tasks 4–6) references the committed OBSD `a_drift`/`--per-layer-drift` implementation as the concrete template and names the exact symbols/insertion points to mirror. `θ_max*` in Task 8 is a runtime choice from E3, not an unfilled blank.

**Type/name consistency:** `chiron_rot_coeffs(phi,theta_max,s_warm,m,a,c)`, `chiron_rot_forward(q,p,a,c,sign,T,m)`, `chiron_rot_backward(dq_out,dp_out,q_in,p_in,a,c,phi,theta_max,s_warm,T,m,dq_in,dp_in,dphi,scratch_da,scratch_dc)` consistent across decls (Task 2/3 Step 3), kernels (Step 4), and tests (Step 1). CPU refs `glades::chiron::rot_coeffs/rot_forward/rot_inverse/rot_backward` consistent Tasks 1–3. Param `rot_phi[l]`/`rot_phi_grad[l]`, flags `--rot-coupling`/`--rot-theta-max`/`--rot-warmup` consistent Tasks 4–8.
