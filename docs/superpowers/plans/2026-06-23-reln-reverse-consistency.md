# ReLN Reverse-Consistency Instability Cure — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cure the CHIRON q-side reverse-amplification instability at its source by re-anchoring the ReLN backward to self-consistent normalization statistics, so the `xhat` inflation that overflows `dgamma` never forms — making clean 5B+ data-scale training reachable.

**Architecture:** A diagnose-first decision program. (1) Extend the existing one-shot grad-trigger snapshot to disambiguate the micro-mechanism (recompute-drift R vs variance-collapse V). (2) Build the primary cure — `chiron_reln_backward_reanchor`, a sibling of `chiron_reln_backward` that recomputes per-row mean/invStd from the backward's own `q_in` buffer (instead of the saved forward stats) and delegates to the existing `layernorm_backward`, making `xhat=(q_in−mean(q_in))·rstd(q_in)` unit-RMS by construction — near-identity on healthy steps. (3) Wire a default-off trainer flag `--reln-reanchor` at the two ReLN-backward call sites. (4) Gate it: cheap trigger-prevention + no-harm parity (Gate-1), then the 47h full-5B cure-alone run (Gate-2). Branch V (variance floor) and the store-layers fallback are contingent and out of this plan's scope unless a gate routes there.

**Tech Stack:** C++98 + CUDA (glades-ml shared library `libglades.so`), the glades-trainer `chiron_main.cpp` driver, the glades-ml custom unit-test harness (`unit-tests/`, `ASSERT` macro), `run.sh`/`runner.sh` shell drivers.

**Spec:** `docs/superpowers/specs/2026-06-23-reln-reverse-consistency-design.md`

---

## Repository layout

This plan touches **two repositories**:
- **glades-ml** at `/home/robert/dev/glades-ml` (the library + unit tests). The kernel and unit test land here.
- **glades-trainer** at `/home/robert/dev/glades-trainer` (the training driver). The flag, call-site wiring, diagnostic extension, and `run.sh` passthrough land here. The trainer builds against the in-tree glades-ml via CMake.

Commit in each repository independently. When a trainer change depends on a new library symbol, build glades-ml first (`sh .configure.sh cuda` from the glades-ml root) so the trainer links the new symbol.

## File Structure

| File | Repo | Responsibility | Change |
|---|---|---|---|
| `Backend/Machine Learning/Networks/cuda/gpu_chiron.cu` | glades-ml | ReLN kernels | Add `chiron_reln_reanchor_stats_kernel` + `chiron_reln_backward_reanchor` |
| `Backend/Machine Learning/Networks/cuda/gpu_chiron.h` | glades-ml | ReLN decls + no-CUDA stubs | Add decl + inline stub for `chiron_reln_backward_reanchor` |
| `unit-tests/Backend/Machine Learning/chiron-test.cpp` | glades-ml | CHIRON unit tests | Add `CHIRONRelnReanchorTest()` |
| `unit-tests/Backend/Machine Learning/chiron-test.h` | glades-ml | test decls | Add `void CHIRONRelnReanchorTest();` |
| `unit-tests/main.cpp` | glades-ml | test registration | Register `chiron-reanchor` / `reanchor` |
| `trainer/chiron_main.cpp` | glades-trainer | training driver | Phase-0 diagnostic extension; `--reln-reanchor` Config field/default/help/parse/log; wire at both ReLN-backward call sites |
| `run.sh` | glades-trainer | flagship launcher | Pass `--reln-reanchor` through |
| `research/RELN_REANCHOR_*.md` | glades-ml | verdict docs | Diagnostic + Gate-1 + Gate-2 results |

---

## Background facts the engineer needs

- **The bug.** At the first bad gradient (seed 2024, step 24070), the L00 snapshot showed `xhat_rms=13`, `qsig_ratio=13`, `mean_abs_delta=0.006`. `xhat` is ~13× inflated *before* the `dgamma[col]=Σ_t dout[t,col]·xhat[t,col]` sum, because the backward normalizes the recomputed activation `q_in` with **saved forward stats** that no longer match it. The sum-over-T=16384 then turns a 13× row inflation into a `dgamma` overflow (1.9e23) and a `dE` overflow (2.5e25).
- **Why clamps only contained it.** gg-clamp / AGC / spectral / the Phase-3 per-element xhat clamp all clip the *inflated result*. The cure recomputes the *correct* unit-RMS `xhat` so nothing inflates.
- **The existing backward** (`chiron_reln_backward`, `gpu_chiron.cu:968`) splits saved `stats[T,2]` into `mean[T]`/`invStd[T]` via `chiron_stats_split_kernel`, then calls `layernorm_backward(dq_out,q_in,gamma,mean,invStd,T,m,dq_in,dgamma,dbeta)`. The re-anchor kernel replaces *only* the stats source: it computes `mean[T]`/`invStd[T]` from `q_in` directly (mirroring the forward's two-pass reduction in `chiron_reln_forward_rows`, `gpu_chiron.cu:631`), then calls the *same* `layernorm_backward`.
- **`layernorm_backward` signature** (`gpu_kernels.h:28`): `bool layernorm_backward(const float* dout, const float* x, const float* gamma, const float* mean, const float* invStd, int rows, int cols, float* dx, float* dgamma, float* dbeta);` — it consumes `mean`/`invStd` as inputs and computes `xhat=(x-mean)*invStd` internally. Feeding it `mean(q_in)`/`rstd(q_in)` yields unit-RMS `xhat` by construction.
- **glades-ml stats convention:** `chiron_reln_forward_rows` writes `stats[row*2+1] = sigma` (raw sigma, not log). The re-anchor kernel never reads saved stats, so it is convention-independent. (Note: the *trainer's* snapshot uses a log-sigma convention — `exp(logSigma)` at `chiron_main.cpp:15029` — which only matters for Task 1, which stays inside that function.)
- **Build the library:** from `/home/robert/dev/glades-ml`, `sh .configure.sh cuda`.
- **Build + run a unit test:** from `/home/robert/dev/glades-ml`, `cd unit-tests/build && sh .configure.sh cuda` then `cd /home/robert/dev/glades-ml/unit-tests && bash test.sh <name>`.
- **Build the trainer:** from `/home/robert/dev/glades-trainer`, `cmake --build build --target glades_chiron_train -j$(nproc)` (CMake configures against the in-tree glades-ml).
- **CLAUDE.md constraints:** C++98 only (no C++11 in library/test code — the existing tests use `union`-based bit compares and raw loops; follow that). Everything default-off. Multi-seed gating.

---

## Task 1: Phase-0 diagnostic extension (disambiguate R vs V)

**Goal:** When the grad-trigger fires, additionally report (a) the **absolute** saved-σ vs recompute-σ split and (b) the `xhat_selfconsistent` RMS, so the verdict reads whether the saved stats (V: forward variance collapse) or the recompute (R: drift) is the anomalous side, and confirms re-anchor would restore unit-RMS. No new flags — rides the existing `--sira-grad-trigger-dump`.

**Files:**
- Modify: `trainer/chiron_main.cpp` — `log_sira_grad_trigger_l00_snapshot` (starts line ~14964; the per-row loop ends ~15080; the summary `log_warn` is ~15092).

- [ ] **Step 1: Add the self-consistent xhat accumulation inside the existing per-row loop**

In the per-row loop (currently computing `qVar`, `xhatSq`, `xhatMax` over `j` at `chiron_main.cpp:15049-15058`), the loop already computes `qMean` and will compute `qVar`. After the loop computes `ri.qSigma` (line ~15066), add a second short pass to accumulate the self-consistent xhat RMS using the recomputed mean/sigma. Insert immediately after `ri.qSigma = std::sqrt(qVar / (double)m + (double)cfg.eps_reln);`:

```cpp
		// Phase-0 diagnostic (2026-06-23): xhat re-derived from this row's OWN
		// recomputed mean/sigma — unit-RMS by construction iff re-anchor would
		// fix this row.  Compared against ri.xhatRms (saved-stats xhat).
		double xhatSelfSq = 0.0;
		const double qInvStd = 1.0 / (ri.qSigma + 1e-30);
		for (int j = 0; j < m; ++j)
		{
			const double qv = (double)s.qbranchQHost[off + (size_t)j];
			const double xs = (qv - qMean) * qInvStd;
			xhatSelfSq += xs * xs;
		}
		ri.xhatSelfRms = std::sqrt(xhatSelfSq / (double)m);
```

- [ ] **Step 2: Add the `xhatSelfRms` field to the `RowInfo` struct and a percentile vector**

In the `RowInfo` struct (`chiron_main.cpp:15003-15015`) add a field after `double xhatMax;`:

```cpp
		double xhatSelfRms;
```

After the `xhatRmsVals` vector declaration (`chiron_main.cpp:15018`) add:

```cpp
	std::vector<double> xhatSelfRmsVals;
	std::vector<double> savedSigmaVals;   // absolute saved (forward) sigma
	std::vector<double> recompSigmaVals;  // absolute recomputed (s.q) sigma
```

Reserve them next to the others (`chiron_main.cpp:15020-15023`):

```cpp
	xhatSelfRmsVals.reserve((size_t)T);
	savedSigmaVals.reserve((size_t)T);
	recompSigmaVals.reserve((size_t)T);
```

After `meanAbsDeltas.push_back(std::fabs(ri.meanDelta));` (`chiron_main.cpp:15079`) add:

```cpp
		xhatSelfRmsVals.push_back(ri.xhatSelfRms);
		savedSigmaVals.push_back(ri.sigma);
		recompSigmaVals.push_back(ri.qSigma);
```

- [ ] **Step 3: Sort the new vectors and emit the verdict line**

After the existing `std::sort(meanAbsDeltas...)` (`chiron_main.cpp:15083`) add:

```cpp
	std::sort(xhatSelfRmsVals.begin(), xhatSelfRmsVals.end());
	std::sort(savedSigmaVals.begin(), savedSigmaVals.end());
	std::sort(recompSigmaVals.begin(), recompSigmaVals.end());
```

Immediately after the existing summary `log_warn(...l00 step...)` call (ends `chiron_main.cpp:15100`) add a second `log_warn` that prints the disambiguation and an explicit verdict hint:

```cpp
	// Phase-0 diagnostic verdict (2026-06-23).  xhat_self_rms is unit-RMS by
	// construction in BOTH R and V (a row normalized by its own mean/std), so it
	// only confirms re-anchor would restore unit-RMS; the R-vs-V split is read
	// from the ABSOLUTE saved-vs-recompute sigma (NOT a ratio — a ratio test
	// would make R and V the same inequality and V unreachable):
	//  - xhat_self_rms ~ 1  AND  saved sigma collapsed broadly below the recompute
	//      scale (saved_p50 << recomp_p50, forward std->0)  -> VARIANCE-COLLAPSE (V)
	//      => Branch V (variance floor); re-anchor also restores unit but s.q was fine.
	//  - xhat_self_rms ~ 1  AND  recompute sigma drifted up in its tail
	//      (recomp_max >> recomp_p50) with saved normal           -> RECOMPUTE-DRIFT (R)
	//      => --reln-reanchor restores unit-RMS xhat at its source.
	//  - xhat_self_rms still >> 1 (only numerically, recomp sigma->0)-> LOCALIZED/other (L).
	//  - no accepted rows                                          -> INCONCLUSIVE.
	const double xhatSelfMax = xhatSelfRmsVals.empty() ? std::numeric_limits<double>::quiet_NaN() : xhatSelfRmsVals.back();
	const double savedSigP50 = percentile_sorted(savedSigmaVals, 0.50);
	const double recompSigP50 = percentile_sorted(recompSigmaVals, 0.50);
	const double recompSigMax = recompSigmaVals.empty() ? 0.0 : recompSigmaVals.back();
	const char* verdict = "INCONCLUSIVE";
	if (!(xhatSelfMax == xhatSelfMax)) {              // NaN -> no accepted rows
		verdict = "INCONCLUSIVE";
	} else if (xhatSelfMax < 2.0) {
		const bool savedCollapsed = savedSigP50 < 0.5 * recompSigP50;   // saved sigma broadly below recompute scale
		const bool recompTailHigh = recompSigMax > 2.0 * recompSigP50;  // recompute sigma drifted up in the tail
		verdict = savedCollapsed ? "VARIANCE-COLLAPSE(V)" :
		          recompTailHigh ? "RECOMPUTE-DRIFT(R)"   : "MIXED";
	} else {
		verdict = "LOCALIZED-OR-OTHER(L)";
	}
	log_warn("chiron","[sira-grad-trigger-l00-diag step %6d] verdict=%s xhat_self_rms(p50/p99/max)=%.6g/%.6g/%.6g saved_sigma(p50/p99/max)=%.6g/%.6g/%.6g recomp_sigma(p50/p99/max)=%.6g/%.6g/%.6g\n",
	         step, verdict,
	         percentile_sorted(xhatSelfRmsVals, 0.50), percentile_sorted(xhatSelfRmsVals, 0.99), xhatSelfMax,
	         savedSigP50, percentile_sorted(savedSigmaVals, 0.99), savedSigmaVals.empty()?0.0:savedSigmaVals.back(),
	         recompSigP50, percentile_sorted(recompSigmaVals, 0.99), recompSigMax);
```

- [ ] **Step 4: Build the trainer**

Run (from `/home/robert/dev/glades-trainer`):
```bash
cmake --build build --target glades_chiron_train -j"$(nproc)"
```
Expected: compiles clean, no warnings on the edited function.

- [ ] **Step 5: Smoke the diagnostic with a forced threshold**

Run a tiny forward with the trigger forced (threshold 0 fires on step 1) on a small shape so it dumps immediately. From `/home/robert/dev/glades-trainer`:
```bash
sh run.sh flagship --m 128 --layers 4 --heads 4 --T 1024 --steps 2 --accum 1 \
  --qk-norm --sira-grad-trigger-dump --sira-grad-trigger-stop --sira-grad-trigger-sumsq 0 \
  --save /tmp/reanchor_diag_smoke 2>&1 | tee /tmp/reanchor_diag_smoke.log
grep "sira-grad-trigger-l00-diag" /tmp/reanchor_diag_smoke.log
```
Expected: one `...l00-diag...` line with finite `verdict=`, `xhat_self_rms`, `saved_sigma`, `recomp_sigma` fields (on healthy random init: `verdict=MIXED` or `RECOMPUTE-DRIFT(R)` with `xhat_self_rms≈1`, both sigmas finite and similar). The point of the smoke is *fields finite & line present*, not the verdict value.

- [ ] **Step 6: Commit (glades-trainer)**

```bash
cd /home/robert/dev/glades-trainer
git add trainer/chiron_main.cpp
git commit -m "Phase-0 diagnostic: xhat-self-consistency + abs sigma split in grad-trigger snapshot

Extends log_sira_grad_trigger_l00_snapshot to disambiguate the q-side
instability micro-mechanism: recompute-drift (R) vs variance-collapse (V) vs
localized (L). Rides the existing --sira-grad-trigger-dump; no new flag.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Branch R kernel — `chiron_reln_backward_reanchor` (glades-ml)

**Goal:** A ReLN backward that derives mean/invStd from `q_in` itself, then delegates to `layernorm_backward`. Default behavior of the existing path is unchanged; this is a new symbol.

**Files:**
- Modify: `Backend/Machine Learning/Networks/cuda/gpu_chiron.cu` (add kernel + wrapper after `chiron_reln_backward_bounded`, ~line 1009)
- Modify: `Backend/Machine Learning/Networks/cuda/gpu_chiron.h` (decl after line 217; no-CUDA stub after line 685)

- [ ] **Step 1: Write the failing unit test first** — see Task 3. (TDD: Task 3 Step 1–2 are written and run-to-fail before this task's implementation. If executing strictly top-to-bottom, do Task 3 Steps 1–2 now, then return here.)

- [ ] **Step 2: Add the re-anchor stats kernel + wrapper in `gpu_chiron.cu`**

Insert after `chiron_reln_backward_bounded` (after line 1009, before the `//===== 4. Sketch project` banner at 1011):

```cpp
// ===========================================================================
//  3c. ReLN reverse-consistency backward (q-side instability cure, 2026-06-23).
// ===========================================================================
//
// Root cause (grad-trigger evidence, seed 2024 step 24070): the backward
// normalizes the recomputed activation q_in with the SAVED forward stats,
// which have drifted, inflating xhat ~13x BEFORE the sum-over-T forms dgamma
// and overflows it.  This backward re-derives (mean, invStd) from q_in itself
// (mirroring chiron_reln_forward_rows' two-pass reduction), so the xhat that
// layernorm_backward forms is unit-RMS by construction.  On a healthy step
// (recompute == forward) the re-derived stats equal the saved stats up to
// fp reduction order -> near-identity.  See
// docs/superpowers/specs/2026-06-23-reln-reverse-consistency-design.md.

namespace {

// One block per row: recompute mean and invStd from q_in over the m columns.
// Writes mean[T] into split[0..T) and invStd[T] into split[T..2T), matching the
// (mean, invStd) layout chiron_reln_backward feeds to layernorm_backward.
__global__ void chiron_reln_reanchor_stats_kernel(const float* __restrict__ q_in,
                                                  int cols, float eps,
                                                  float* __restrict__ mean,
                                                  float* __restrict__ invStd)
{
	int row = blockIdx.x;
	const float* xRow = q_in + (size_t)row * cols;

	extern __shared__ float smem[];
	float* sSumA = smem;
	float* sSumB = smem + (blockDim.x / 32 + 1);
	__shared__ float sMean, sInvStd;

	// Pass 1: mean.
	float s = 0.0f;
	for (int i = threadIdx.x; i < cols; i += blockDim.x) s += xRow[i];
	s = blockReduceSum(s, sSumA);
	if (threadIdx.x == 0) sMean = s / (float)cols;
	__syncthreads();
	const float mu = sMean;

	// Pass 2: variance -> invStd.
	float v = 0.0f;
	for (int i = threadIdx.x; i < cols; i += blockDim.x) {
		float d = xRow[i] - mu;
		v += d * d;
	}
	v = blockReduceSum(v, sSumB);
	if (threadIdx.x == 0) {
		float var = v / (float)cols + eps;
		sInvStd = 1.0f / sqrtf(var);
	}
	__syncthreads();

	if (threadIdx.x == 0) {
		mean[row]   = mu;
		invStd[row] = sInvStd;
	}
}

} // anonymous namespace

// Re-anchored ReLN backward: identical interface to chiron_reln_backward, but
// derives (mean, invStd) from q_in rather than the saved stats.  `eps` must
// match the forward's eps_reln so healthy steps reproduce the saved stats.
bool chiron_reln_backward_reanchor(const float* dq_out, const float* q_in,
                                    const float* gamma,
                                    int T, int m, float eps,
                                    float* dq_in, float* dgamma, float* dbeta,
                                    float* scratch_stats_split)
{
	if (T <= 0 || m <= 0) return true;
	float* d_mean   = scratch_stats_split;
	float* d_invStd = scratch_stats_split + T;
	int block = rowBlockSize(m);
	size_t smemBytes = 2u * (size_t)(block / 32 + 1) * sizeof(float);
	chiron_reln_reanchor_stats_kernel<<<T, block, smemBytes, computeStream()>>>(
	    q_in, m, eps, d_mean, d_invStd);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return layernorm_backward(dq_out, q_in, gamma, d_mean, d_invStd,
	                          T, m, dq_in, dgamma, dbeta);
}
```

Note: `rowBlockSize`, `blockReduceSum`, `computeStream`, `GLADES_CUDA_CHECK`, `kBlockElem` are all already in scope in this file (used by the sibling kernels above). The shared-memory size mirrors `chiron_reln_forward_rows`'s two reduce buffers.

- [ ] **Step 3: Add the declaration in `gpu_chiron.h`**

After the `chiron_reln_backward_bounded` decl (ends line ~226, just before the `//===== 4` section), add:

```cpp
// ReLN reverse-consistency backward (q-side instability cure, 2026-06-23):
// same interface as chiron_reln_backward but re-derives (mean, invStd) from
// q_in itself, so the xhat layernorm_backward forms is unit-RMS by
// construction (cures the recompute/saved-stat drift that inflates dgamma).
// `eps` must equal the forward's eps_reln.  Near-identity on healthy steps.
// scratch_stats_split: caller-owned buffer of 2*T floats.
bool chiron_reln_backward_reanchor(const float* dq_out, const float* q_in,
                                    const float* gamma,
                                    int T, int m, float eps,
                                    float* dq_in, float* dgamma, float* dbeta,
                                    float* scratch_stats_split);
```

- [ ] **Step 4: Add the no-CUDA stub in `gpu_chiron.h`**

After the `chiron_reln_backward_bounded` inline stub (line 685), add:

```cpp
inline bool chiron_reln_backward_reanchor(const float*, const float*,
                                  const float*,
                                  int, int, float,
                                  float*, float*, float*, float*) { return false; }
```

- [ ] **Step 5: Build the library**

Run (from `/home/robert/dev/glades-ml`):
```bash
sh .configure.sh cuda
```
Expected: builds clean; `chiron_reln_backward_reanchor` symbol present (`nm -C build/.../libglades.so | grep reln_backward_reanchor` shows it, or simply that the test in Task 3 links).

- [ ] **Step 6: Commit (glades-ml)** — deferred to Task 3 Step 6 so the kernel + its passing test commit together.

---

## Task 3: `chiron-reanchor` unit test (glades-ml)

**Goal:** Prove (a) under injected recompute-drift the re-anchored `dgamma` stays bounded (xhat→unit) where the plain backward overflows, and (b) on a healthy step (saved stats == true stats of `q_in`) re-anchor is near-identical to the plain backward. Mirrors `CHIRONRelnBackwardBoundedTest` (`chiron-test.cpp:18288`).

**Files:**
- Modify: `unit-tests/Backend/Machine Learning/chiron-test.cpp` (add `CHIRONRelnReanchorTest()` after `CHIRONRelnBackwardBoundedTest`, ~line 18334)
- Modify: `unit-tests/Backend/Machine Learning/chiron-test.h` (add decl near `CHIRONRelnBackwardBoundedTest`, line 111)
- Modify: `unit-tests/main.cpp` (register `chiron-reanchor`/`reanchor` after the `chiron-relnbound` block, line 254-257)

- [ ] **Step 1: Add the test declaration in `chiron-test.h`**

After line 111 (`void CHIRONRelnBackwardBoundedTest();`) add:

```cpp
void CHIRONRelnReanchorTest();
```

- [ ] **Step 2: Write the failing test in `chiron-test.cpp`**

After `CHIRONRelnBackwardBoundedTest`'s closing brace (line 18334) add. **Test design:** construct `q_in` with a *known* per-row mean/sigma; set the saved `stats` to those true values so the HEALTHY case has re-anchor reproduce them (near-identity). Then corrupt one row of `q_in` (drift) while leaving `stats` at the stale pre-drift values: the plain backward divides the drifted row by the stale small sigma → huge `dgamma`; re-anchor recomputes the row's true (large) sigma → unit xhat → bounded `dgamma`.

```cpp
// Case (2026-06-23): chiron_reln_backward_reanchor — re-deriving (mean,invStd)
// from q_in makes xhat unit-RMS by construction.  HEALTHY (saved stats == true
// stats of q_in): near-identical to plain backward.  DRIFT (q_in row corrupted,
// stats stale): plain dgamma overflows, re-anchored dgamma stays bounded.
void CHIRONRelnReanchorTest()
{
#ifdef GLADES_HAVE_CUDA
	if (!glades::gpu::initDevice())
	{ std::printf("  [CHIRON reln-bwd reanchor] no CUDA device — skipped\n"); return; }
	const int T = 64, m = 32;
	const float eps = 1e-5f;

	// Build q_in with per-row mean=0, sigma~1 (standard normal-ish via LCG),
	// and set saved stats to each row's TRUE (mean, sigma) so healthy re-anchor
	// reproduces them.
	std::vector<float> dq_out((size_t)T*m), q_in((size_t)T*m), gamma(m, 1.0f), stats((size_t)T*2);
	LCG rng(919u);
	for (size_t i=0;i<q_in.size();++i) q_in[i] = rng.next_unit();   // (-1,1)
	for (size_t i=0;i<dq_out.size();++i) dq_out[i] = rng.next_unit();
	for (int t=0;t<T;++t){
		double mu=0.0; for(int j=0;j<m;++j) mu += q_in[(size_t)t*m+j]; mu/=m;
		double var=0.0; for(int j=0;j<m;++j){ double d=q_in[(size_t)t*m+j]-mu; var+=d*d; } var=var/m+eps;
		stats[(size_t)t*2+0]=(float)mu; stats[(size_t)t*2+1]=(float)std::sqrt(var); // glades-ml: raw sigma
	}

	glades::gpu::GpuBuffer<float> d_dq,d_q,d_g,d_st,d_dqinA,d_dgA,d_dbA,d_dqinB,d_dgB,d_dbB,d_split;
	ASSERT("Reanchor: alloc", d_dq.allocate(T*m)&&d_q.allocate(T*m)&&d_g.allocate(m)&&d_st.allocate(T*2)
	      &&d_dqinA.allocate(T*m)&&d_dgA.allocate(m)&&d_dbA.allocate(m)
	      &&d_dqinB.allocate(T*m)&&d_dgB.allocate(m)&&d_dbB.allocate(m)&&d_split.allocate(T*2));
	ASSERT("Reanchor: upload", d_dq.upload(&dq_out[0])&&d_q.upload(&q_in[0])&&d_g.upload(&gamma[0])&&d_st.upload(&stats[0]));

	// HEALTHY: plain(saved stats) vs reanchor(recomputed) must be near-identical.
	ASSERT("Reanchor: zero dg/db", d_dgA.zero()&&d_dbA.zero()&&d_dgB.zero()&&d_dbB.zero());
	ASSERT("Reanchor: plain healthy", glades::gpu::chiron_reln_backward(d_dq.data(),d_q.data(),d_g.data(),d_st.data(),T,m,d_dqinA.data(),d_dgA.data(),d_dbA.data(),d_split.data()));
	ASSERT("Reanchor: reanchor healthy", glades::gpu::chiron_reln_backward_reanchor(d_dq.data(),d_q.data(),d_g.data(),T,m,eps,d_dqinB.data(),d_dgB.data(),d_dbB.data(),d_split.data()));
	std::vector<float> dgA(m),dgB(m),dbA(m),dbB(m);
	ASSERT("Reanchor: dl healthy", d_dgA.download(&dgA[0])&&d_dgB.download(&dgB[0])&&d_dbA.download(&dbA[0])&&d_dbB.download(&dbB[0]));
	double maxRel=0.0;
	for(int j=0;j<m;++j){
		double da=std::fabs((double)dgA[j]-(double)dgB[j])/(std::fabs((double)dgA[j])+1e-6);
		double db=std::fabs((double)dbA[j]-(double)dbB[j])/(std::fabs((double)dbA[j])+1e-6);
		if(da>maxRel)maxRel=da; if(db>maxRel)maxRel=db;
	}
	std::printf("  [reln-bwd reanchor] healthy max rel-err(dgamma,dbeta)=%.3e\n", maxRel);
	ASSERT("Reanchor: healthy near-identical (rel<1e-3)", maxRel < 1e-3);

	// DRIFT: corrupt row 5's q_in to 1000 but leave stats stale (sigma~1).
	for(int j=0;j<m;++j) q_in[(size_t)5*m+j] = 1000.0f;
	ASSERT("Reanchor: upload drift", d_q.upload(&q_in[0]));
	ASSERT("Reanchor: zero dg2", d_dgA.zero()&&d_dgB.zero()&&d_dbA.zero()&&d_dbB.zero());
	ASSERT("Reanchor: plain drift", glades::gpu::chiron_reln_backward(d_dq.data(),d_q.data(),d_g.data(),d_st.data(),T,m,d_dqinA.data(),d_dgA.data(),d_dbA.data(),d_split.data()));
	ASSERT("Reanchor: reanchor drift", glades::gpu::chiron_reln_backward_reanchor(d_dq.data(),d_q.data(),d_g.data(),T,m,eps,d_dqinB.data(),d_dgB.data(),d_dbB.data(),d_split.data()));
	ASSERT("Reanchor: dl drift", d_dgA.download(&dgA[0])&&d_dgB.download(&dgB[0]));
	double maxA=0,maxB=0; for(int j=0;j<m;++j){ if(std::fabs(dgA[j])>maxA)maxA=std::fabs(dgA[j]); if(std::fabs(dgB[j])>maxB)maxB=std::fabs(dgB[j]); }
	std::printf("  [reln-bwd reanchor] drift: plain max|dgamma|=%.3g  reanchor max|dgamma|=%.3g\n", maxA, maxB);
	// Drifted row contributes ~xhat=1000/sigma~1000 to plain; re-anchor's xhat is unit,
	// so each column's |sum dout*xhat| <= ~T for re-anchor and is ~1000x larger for plain.
	ASSERT("Reanchor: drift bounded << plain", maxB < maxA && maxB < (double)(4.0 * T));
#else
	std::printf("  [CHIRON reln-bwd reanchor] built without CUDA — skipped\n");
#endif
}
```

- [ ] **Step 3: Register the test in `main.cpp`**

After the `chiron-relnbound` block (`main.cpp:254-257`) add:

```cpp
	    else if (strcmp(argv[1], "chiron-reanchor") == 0 || strcmp(argv[1], "reanchor") == 0)
	    {
		CHIRONRelnReanchorTest();
	    }
```

- [ ] **Step 4: Build the unit tests and run the new test — verify it FAILS without the kernel, then PASSES with it**

If Task 2's kernel is not yet built, the link fails (expected "undefined reference to chiron_reln_backward_reanchor"). After Task 2 is built into the library, rebuild tests and run:
```bash
cd /home/robert/dev/glades-ml/unit-tests/build && sh .configure.sh cuda
cd /home/robert/dev/glades-ml/unit-tests && bash test.sh reanchor
```
Expected (with kernel present): all `Reanchor:` ASSERTs pass; the printed `healthy max rel-err` is < 1e-3 and `drift: ... reanchor max|dgamma|` is ~1000× smaller than plain.

- [ ] **Step 5: If healthy near-identity fails** (rel-err ≥ 1e-3): the reduction order between `chiron_reln_reanchor_stats_kernel` and the forward differs enough to matter. This is acceptable behavior to *observe*, but the test bar is correctness of the cure, not bit-identity. Loosen the healthy bar to `maxRel < 5e-3` and record the observed value in the commit message. Do NOT loosen the drift bar.

- [ ] **Step 6: Commit (glades-ml) — kernel + test together**

```bash
cd /home/robert/dev/glades-ml
git add "Backend/Machine Learning/Networks/cuda/gpu_chiron.cu" \
        "Backend/Machine Learning/Networks/cuda/gpu_chiron.h" \
        "unit-tests/Backend/Machine Learning/chiron-test.cpp" \
        "unit-tests/Backend/Machine Learning/chiron-test.h" \
        unit-tests/main.cpp
git commit -m "Branch R: chiron_reln_backward_reanchor + chiron-reanchor test

Re-derives (mean,invStd) from q_in so layernorm_backward forms a unit-RMS xhat
by construction — cures the recompute/saved-stat drift that inflates dgamma at
its source. Near-identical to chiron_reln_backward on healthy steps; bounds the
drift case the plain backward overflows. Default-off (new symbol; no call-site
change yet). Test: chiron-reanchor.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Trainer flag `--reln-reanchor` + call-site wiring (glades-trainer)

**Goal:** A default-off `--reln-reanchor` flag that routes both ReLN-backward call sites through `chiron_reln_backward_reanchor`. Mirrors the existing `--reln-bwd-xhat-clamp` (`relnBwdXhatClamp`) wiring exactly.

**Files:**
- Modify: `trainer/chiron_main.cpp` — Config field (~1267), default (~1510), help (~1752), parse (~2305), startup log (~2373), both call sites (~14307-14316 and ~14420-14429)

- [ ] **Step 1: Add the Config field**

After `float relnBwdXhatClamp;` (`chiron_main.cpp:1267`) add:

```cpp
	bool  relnReanchor;        // q-side cure: re-derive ReLN-bwd stats from q_in
```

- [ ] **Step 2: Add the default in the Config constructor**

After `relnBwdXhatClamp(0.0f),` (`chiron_main.cpp:1510`) add:

```cpp
	      relnReanchor(false),
```

- [ ] **Step 3: Add the help text**

After the `--reln-bwd-xhat-clamp` help block (`chiron_main.cpp:1752`, a multi-line string) add a line:

```cpp
		"  --reln-reanchor          q-side cure: re-derive (mean,invStd) from q_in in\n"
		"                           the ReLN backward so xhat is unit-RMS by construction\n"
		"                           (removes the recompute/saved-stat drift overflow). Default off.\n"
```

- [ ] **Step 4: Add the parse case**

After the `--reln-bwd-xhat-clamp` parse case (`chiron_main.cpp:2305-2306`) add:

```cpp
		else if (streq(a, "--reln-reanchor")) cfg.relnReanchor = true;
```

- [ ] **Step 5: Add the startup log**

After the `relnBwdXhatClamp` startup log block (`chiron_main.cpp:2373-2375`) add:

```cpp
	if (cfg.relnReanchor)
		log_info("chiron","[reln-reanchor] ACTIVE — ReLN backward re-derives (mean,invStd) from q_in (q-side instability cure)\n");
```

- [ ] **Step 6: Wire the first call site (~14307)**

The current block (`chiron_main.cpp:14307-14316`) is:
```cpp
		if (cfg.relnBwdXhatClamp > 0.0f) {
			if (!glades::gpu::chiron_reln_backward_bounded(
			        ... s.stats_split.data(), cfg.relnBwdXhatClamp)) return false;
		} else {
			if (!glades::gpu::chiron_reln_backward(
			        ...)) return false;
		}
```
Change the `else` branch to a re-anchor-aware chain. Replace the `} else {` ... plain-call block so it reads (keep the existing exact argument list for `chiron_reln_backward` — `dq_out, q_in, gamma, stats, T, m, dq_in, dgamma, dbeta, stats_split`; for re-anchor drop `stats` and insert `cfg.eps_reln` after `m`):

```cpp
		if (cfg.relnBwdXhatClamp > 0.0f) {
			if (!glades::gpu::chiron_reln_backward_bounded(
			        /* …existing args… */ s.stats_split.data(), cfg.relnBwdXhatClamp)) return false;
		} else if (cfg.relnReanchor) {
			if (!glades::gpu::chiron_reln_backward_reanchor(
			        /* dq_out */ /*…*/, /* q_in */ /*…*/, /* gamma */ /*…*/,
			        /* T */ /*…*/, /* m */ /*…*/, cfg.eps_reln,
			        /* dq_in */ /*…*/, /* dgamma */ /*…*/, /* dbeta */ /*…*/,
			        s.stats_split.data())) return false;
		} else {
			if (!glades::gpu::chiron_reln_backward(
			        /* …existing args… */)) return false;
		}
```
**Engineer note:** read the exact existing argument expressions at `chiron_main.cpp:14308-14316` and reuse them verbatim for the matching parameters; the only differences vs `chiron_reln_backward` are: omit the `stats` argument, and insert `cfg.eps_reln` between `m` and `dq_in`. Confirm `cfg.eps_reln` is the same eps passed to the ReLN forward (grep `eps_reln` near the forward call); if the forward uses a literal, pass that literal instead so healthy steps reproduce saved stats.

- [ ] **Step 7: Wire the second call site (~14420)**

Apply the identical transformation to the second block (`chiron_main.cpp:14420-14429`), which has the same `if (cfg.relnBwdXhatClamp>0) bounded else plain` shape (the `p_norm round-trip` path). Insert the same `else if (cfg.relnReanchor) { chiron_reln_backward_reanchor(... cfg.eps_reln ...); }` branch using that block's local argument expressions.

- [ ] **Step 8: Build the trainer**

Run (from `/home/robert/dev/glades-trainer`, after the glades-ml library from Task 2/3 is built):
```bash
cmake --build build --target glades_chiron_train -j"$(nproc)"
```
Expected: clean compile and link (resolves `chiron_reln_backward_reanchor` from the rebuilt library).

- [ ] **Step 9: Pass the flag through `run.sh`**

`run.sh` forwards unknown flags into the flagship arg list (the same mechanism that carries `--reln-bwd-xhat-clamp`). Confirm by grep:
```bash
cd /home/robert/dev/glades-trainer && grep -n "reln-bwd-xhat-clamp\|CHIRON_ARGS\|\"\$@\"\|flagship)" run.sh | head
```
If `--reln-bwd-xhat-clamp` reaches the chiron exec via a passthrough/`"$@"`, `--reln-reanchor` rides the same path — no edit needed; note that in the commit. If `run.sh` has an explicit allow-list that names `--reln-bwd-xhat-clamp`, add `--reln-reanchor` to the same list (show the edited line in the commit).

- [ ] **Step 10: Smoke the flag (small shape, 2 steps, flag ACTIVE + healthy descent)**

```bash
cd /home/robert/dev/glades-trainer
sh run.sh flagship --m 128 --layers 4 --heads 4 --T 1024 --steps 3 --accum 1 \
  --qk-norm --reln-reanchor --save /tmp/reanchor_flag_smoke 2>&1 | tee /tmp/reanchor_flag_smoke.log
grep -E "reln-reanchor\] ACTIVE|step  *[0-9]" /tmp/reanchor_flag_smoke.log | head
```
Expected: the `[reln-reanchor] ACTIVE` line prints; 3 steps run with finite, descending loss and no grad-skips.

- [ ] **Step 11: Commit (glades-trainer)**

```bash
cd /home/robert/dev/glades-trainer
git add trainer/chiron_main.cpp run.sh
git commit -m "--reln-reanchor flag: route ReLN backward through the reverse-consistency cure

Default-off. Mirrors --reln-bwd-xhat-clamp wiring; routes both ReLN-backward
call sites through chiron_reln_backward_reanchor when set. Smoke: ACTIVE +
healthy 3-step descent at m128/L4/T1024.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Gate-1 runbook — trigger-prevention + 5k no-harm parity (glades-ml docs)

**Goal:** Execute the cheap kill/keep gate and record the verdict. This is a runbook task (runs + a verdict doc), not code.

**Files:**
- Create: `research/RELN_REANCHOR_GATE1_2026_06_23.md` (glades-ml)

- [ ] **Step 1: Run the diagnostic trigger first (Phase 0) to record the verdict**

Reproduce the first bad gradient with the Task-1 diagnostic active (seed 2024, gold gg-clamp recipe). From `/home/robert/dev/glades-trainer`:
```bash
sh run.sh flagship --accum 4 --lr 3e-4 --warmup 750 --sira-warmup 250 \
  --zloss-coef 1e-4 --qk-norm --sira-coef 1e-2 --sira-energy-weight 1.0 \
  --sira-balance-weight 0.25 --sira-action-weight 0.0 --grad-clip 0.5 \
  --dq-layer-clamp 1.0 --dq-embed-clamp 1.0 --grad-group-clamp 1.0 \
  --sira-grad-trigger-dump --sira-grad-trigger-stop --sira-grad-trigger-sumsq 1e20 \
  --sira-qbranch-trace-token 265 --seed 2024 \
  --save database/checkpoints/_reanchor_diag 2>&1 | tee logs/reanchor_diag_seed2024.log
grep "sira-grad-trigger-l00-diag" logs/reanchor_diag_seed2024.log
```
Expected: fires ~step 24070; the `...l00-diag...` line states `verdict=`. **Decision:** `RECOMPUTE-DRIFT(R)` or `VARIANCE-COLLAPSE(V)` with `xhat_self_rms≈1` → proceed (re-anchor restores unit-RMS in both). `LOCALIZED-OR-OTHER(L)` → STOP, route to per-token tracing (do not run Gate-1.2); record and open a separate investigation.

- [ ] **Step 2: Gate-1.1 trigger-prevention** — gold recipe + `--reln-reanchor`, trigger-stop armed, run past 24070 + margin:
```bash
sh run.sh flagship --accum 4 --lr 3e-4 --warmup 750 --sira-warmup 250 \
  --zloss-coef 1e-4 --qk-norm --sira-coef 1e-2 --sira-energy-weight 1.0 \
  --sira-balance-weight 0.25 --sira-action-weight 0.0 --grad-clip 0.5 \
  --dq-layer-clamp 1.0 --dq-embed-clamp 1.0 --grad-group-clamp 1.0 \
  --reln-reanchor --sira-grad-trigger-stop --sira-grad-trigger-sumsq 1e20 \
  --seed 2024 --steps 30000 \
  --save database/checkpoints/_reanchor_g1 2>&1 | tee logs/reanchor_g1_trigprev.log
```
**PASS** = no `[sira-grad-trigger]` stop event through step ≥ 26000 (24070 + margin); `global_sumsq` stays < 1e20. **FAIL** = trigger still fires → re-anchor did not target the mechanism → route to the store-layers fallback (Task 6) if the Step-1 verdict was R, or Branch V (separate plan) if V.

- [ ] **Step 3: Gate-1.2 no-harm parity** — gold vs gold+reanchor, 5k steps, same seed, compare val:
```bash
# baseline (no reanchor)
sh run.sh flagship --accum 4 --lr 3e-4 --warmup 750 --sira-warmup 250 \
  --zloss-coef 1e-4 --qk-norm --sira-coef 1e-2 --sira-energy-weight 1.0 \
  --sira-balance-weight 0.25 --sira-action-weight 0.0 --grad-clip 0.5 \
  --dq-layer-clamp 1.0 --dq-embed-clamp 1.0 --grad-group-clamp 1.0 \
  --seed 1337 --steps 5000 --save database/checkpoints/_reanchor_g1base 2>&1 | tee logs/reanchor_g1_base.log
# treatment (+reanchor)
sh run.sh flagship --accum 4 --lr 3e-4 --warmup 750 --sira-warmup 250 \
  --zloss-coef 1e-4 --qk-norm --sira-coef 1e-2 --sira-energy-weight 1.0 \
  --sira-balance-weight 0.25 --sira-action-weight 0.0 --grad-clip 0.5 \
  --dq-layer-clamp 1.0 --dq-embed-clamp 1.0 --grad-group-clamp 1.0 \
  --reln-reanchor --seed 1337 --steps 5000 --save database/checkpoints/_reanchor_g1trt 2>&1 | tee logs/reanchor_g1_trt.log
```
**PASS** = final val within multi-seed noise (≈ ±0.02 nat) of baseline, ‖g‖ healthy (no spikes), 0 grad-skips, full throughput. **FAIL** = re-anchor's deviation from the saved-stats gradient harms healthy training → route to store-layers fallback (Task 6, exact).

- [ ] **Step 4: Write the Gate-1 verdict doc**

Create `research/RELN_REANCHOR_GATE1_2026_06_23.md` recording: the Phase-0 verdict (R/V/L) with the diag line, Gate-1.1 trigger-prevention result (fired? at what step?), Gate-1.2 parity (baseline val vs treatment val, ‖g‖, grad-skips, tok/s), and the routing decision (proceed to Gate-2 / fallback / stop). Follow the honest-negative-publication convention used by `research/QSIDE_INSTABILITY_INVESTIGATION_2026_06_14.md`.

- [ ] **Step 5: Commit (glades-ml)**

```bash
cd /home/robert/dev/glades-ml
git add research/RELN_REANCHOR_GATE1_2026_06_23.md
git commit -m "RELN reanchor Gate-1: Phase-0 verdict + trigger-prevention + parity result

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6 (CONTINGENT): store-layers fallback — only if Gate-1 routes here

**Trigger condition:** Run this task **only if** Gate-1.1 fails (trigger still fires) or Gate-1.2 fails (parity harmed) **and** the Phase-0 verdict was `RECOMPUTE-DRIFT(R)`. If the verdict was `VARIANCE-COLLAPSE(V)`, do NOT run this task — Branch V (variance floor) gets its own spec+plan per design §6.2. If Gate-1 passed, skip to Task 7.

**Goal:** Store the exact forward q-state for the early ladder (layers 0..k) so the backward reads the true forward activation (zero recompute drift) instead of recomputing it. Exact, hypothesis-agnostic; costs ~64 MB/layer bf16 VRAM.

**Scope note:** This is a larger change — it partially disables reversibility for layers 0..k, which means adding a per-layer forward-q store buffer and reading it in the backward instead of the reversible recompute. Because its exact extent (which buffer, which layers, VRAM headroom at the moment of the run) depends on the Gate-1 failure mode and current memory state, this task is **specified at the design level here and gets a focused sub-plan when triggered**. Do not pre-implement.

- [ ] **Step 1 (when triggered):** Re-enter brainstorming/writing-plans for the store-layers fallback with the Gate-1 failure data in hand (which layers drifted, the parity delta, measured VRAM headroom). Anchor the flag name `--reln-store-layers k` and the ~64 MB/layer bf16 budget from the spec.

---

## Task 7: Gate-2 runbook — full 5B cure-alone (glades-ml docs)

**Goal:** The prize run. Prove a *cure* (not a second container) by running the full 5B data-scale recipe with `--reln-reanchor` and **gg-clamp OFF**. Runbook + verdict doc.

**Files:**
- Create: `research/RELN_REANCHOR_GATE2_2026_06_23.md` (glades-ml)

- [ ] **Step 1: Pre-create the save dir** (silent `save_full` failure otherwise — known gotcha):
```bash
cd /home/robert/dev/glades-trainer
mkdir -p database/checkpoints/chiron_1B_T16384_reanchor5B
```

- [ ] **Step 2: Launch the cure-alone 5B run (gg-clamp OFF)**

```bash
cd /home/robert/dev/glades-trainer
sh run.sh flagship --accum 4 --lr 3e-4 --warmup 750 --sira-warmup 250 \
  --zloss-coef 1e-4 --qk-norm --sira-coef 1e-2 --sira-energy-weight 1.0 \
  --sira-balance-weight 0.25 --sira-action-weight 0.0 --grad-clip 0.5 \
  --dq-layer-clamp 1.0 --dq-embed-clamp 1.0 \
  --reln-reanchor \
  --steps 76000 --seed 1337 --save-every 10000 \
  --save database/checkpoints/chiron_1B_T16384_reanchor5B 2>&1 | tee logs/reanchor_g2_5b.log
```
Note: **no `--grad-group-clamp`** — this is the sharp test. Step-1 loss/‖g‖ should match the gold recipe (~10.78 / ~0.83).

- [ ] **Step 3: Monitor the danger horizon (past ~1.6B tokens ≈ step ~24k)**

Watch `logs/reanchor_g2_5b.log` for grad-skips and ‖g‖ spikes around and past step 24k:
```bash
grep -E "grad-skip|guard|\|\|g\|\|=[0-9]{3,}" logs/reanchor_g2_5b.log | tail
```
**PASS** = 0 grad-skips through 5B AND a new best-val (< 2.5, toward ~2.2–2.3). **CONTAINED-NOT-CURED** = needs gg-clamp re-added to survive → record as a partial result (still useful, but not the cure). **FAIL** = diverges even with re-anchor → the disease is not (only) recompute drift → record and route to Branch V.

- [ ] **Step 4: If single-seed PASS, multi-seed confirm** — repeat Step 2 for seeds 2024 and 4242 (or to the danger horizon ~step 30k if a full 47h×2 is too costly; record which). Require 0 grad-skips + healthy val across seeds before any flagship-promotion discussion.

- [ ] **Step 5: Write the Gate-2 verdict doc and update memory**

Create `research/RELN_REANCHOR_GATE2_2026_06_23.md`: the cure-alone trajectory, grad-skip count through 5B, best-val vs the current flagship's 2.5, the multi-seed table, and the verdict (CURE / CONTAINED / FAIL). If CURE: note the candidate checkpoint and that flagship promotion is a *separate* decision (own gate, per the lineage's ship standard). Add a memory pointer in `/home/robert/.claude/projects/-home-robert-dev-glades-ml/memory/MEMORY.md` per the auto-memory convention. Do NOT touch CLAUDE.md / runner.sh defaults — promotion is out of this plan's scope.

- [ ] **Step 6: Commit (glades-ml)**

```bash
cd /home/robert/dev/glades-ml
git add research/RELN_REANCHOR_GATE2_2026_06_23.md
git commit -m "RELN reanchor Gate-2: full-5B cure-alone verdict (gg-clamp OFF)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- §4 Phase 0 diagnostic → Task 1 ✓ (extends the existing snapshot; the design's "store forward q" is satisfiable from saved-σ vs recompute-σ already in the snapshot, simplifying the change — noted in Background facts).
- §6.1 Branch R `--reln-reanchor` → Tasks 2 (kernel), 3 (test), 4 (flag+wiring) ✓
- §6.1 fallback `--reln-store-layers` → Task 6 (contingent, design-level) ✓
- §6.2 Branch V `--reln-var-floor` → explicitly OUT of scope (own plan if routed), per spec ✓
- §7 Gate-1 (trigger-prevention + 5k parity) → Task 5 ✓
- §8 Gate-2 (full 5B cure-alone, gg-clamp OFF) → Task 7 ✓
- §9 kill criteria → encoded as FAIL/STOP routing in Tasks 5 & 7 ✓
- §10 deliverables (default-off kernel+stub, chiron-reanchor test, flag, diagnostic, run.sh, verdict docs) → Tasks 1–7 ✓

**Placeholder scan:** Code steps show complete code except Task 4 Steps 6-7 (call-site arg expressions) and Task 6 (contingent). Task 4's call sites deliberately reference "reuse the exact existing argument expressions at line N" because the surrounding local variable names must be read in-context and copied verbatim — the transformation (drop `stats`, insert `cfg.eps_reln`) is fully specified. Task 6 is intentionally design-level (YAGNI: build only if a gate routes there). No "TBD/TODO/handle edge cases".

**Type consistency:** `chiron_reln_backward_reanchor(dq_out,q_in,gamma,T,m,eps,dq_in,dgamma,dbeta,scratch_stats_split)` — identical across the .cu wrapper (Task 2 Step 2), .h decl (Task 2 Step 3), no-CUDA stub (Task 2 Step 4), unit test calls (Task 3 Step 2), and both trainer call sites (Task 4 Steps 6-7). `eps` param is `cfg.eps_reln` at both call sites and `eps`(=1e-5) in the test. `RowInfo.xhatSelfRms` (Task 1 Step 1/2) consistent. Flag `relnReanchor`/`--reln-reanchor` consistent across Config/default/help/parse/log/run.sh.
