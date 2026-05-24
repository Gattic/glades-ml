# CHIRON 1B CUDA Graphs Production-Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `--cuda-graphs` (paradigm #51) production-ready under the regstack Phase 2 flagship recipe. Bit-identical NLL ≤ 3.5734 @ 30k, throughput ≥ +2% over ship's 28,072 tok/s.

**Architecture:** Three concrete blockers identified by `research/ITER55_PARADIGM_FLAGS_NULL.md` + `research/CUDA_GRAPHS_FIX4_NOTES.md`:
1. **Blocker A:** `cublasGet/SetMathMode` toggle inside `sgemm_rowmajor` hot path → refactor to two-handle dispatch (one TF32, one strict-FP32; math mode set once at init).
2. **Blocker B:** `scfaFuseStreams` event-record sequence flagged as per-step branching by iter 55 NULL doc, but production flagship doesn't use `--scfa-parallel-branches` (the gate that actually fires cross-stream events). Empirical verification: G1 capture may succeed at flagship recipe with just Blocker A fixed. If not, force `scfaFuseStreams=false` when `--cuda-graphs` is on.
3. **Blocker C:** Explicit mutual exclusion for `--cuda-graphs` vs `--fp8-attn` / `--medal-train` / `--ul2-enabled` at CLI parse (the latter two already added by prior arcs).

After fixes, drop `cfg.scfa` from `chiron_main.cpp:12644` auto-disable list. Math is unchanged — NLL bit-identical (or sub-ULP from FMA reordering).

**Tech Stack:** C++98 + CUDA 13.2. Library `gpu_blas.cu` refactor (~50-100 LOC). Trainer mutual exclusion + auto-disable cleanup (~30 LOC).

**Repos involved:**
- `~/dev/glades-ml/` — `gpu_blas.cu` math-mode refactor.
- `~/dev/glades-trainer/` — CLI mutex + auto-disable cleanup.

**Source-of-truth spec:** `docs/superpowers/specs/2026-05-23-chiron-1b-cuda-graphs-design.md`.

**Existing infrastructure:**
- `--cuda-graphs` flag wired (commit `d2351cc`).
- Capture/replay/invalidation/fallback machinery at `chiron_main.cpp:12894-13007`.
- `cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed)` at `gpu_graph.cu:34` (Relaxed mode — captures only the main stream, NOT side streams).

**Key insight:** the flagship recipe does NOT use `--scfa-parallel-branches` (per iter 71 NULL on Ada). So the cross-stream event records at chiron_main.cpp:6965/6995/7792/7822/10169 do NOT fire. The flagship's actual kernel sequence may already be graph-compatible after Blocker A fixes. G1 empirical test will confirm.

---

## Phase 0: Setup

### Task 0.1: Baseline build sanity check

**Files:** No file changes; verification only.

- [ ] **Step 1: Confirm starting state**

```bash
cd ~/dev/glades-ml
git status
git log --oneline -3
```

Expected: clean working tree on `chiron2`, top commits are the UL2 FAIL doc + CUDA-graphs spec.

```bash
cd ~/dev/glades-trainer
git status
git log --oneline -3
```

Expected: clean working tree; top commit is `67e8af7` (UL2 MASK init revert) or later.

- [ ] **Step 2: Build library + unit tests + run chiron suite**

```bash
cd ~/dev/glades-ml
sh .configure.sh cuda 2>&1 | tail -3
cd ~/dev/glades-ml/unit-tests && sh .configure.sh cuda 2>&1 | tail -3
cd ~/dev/glades-ml/unit-tests && bash test.sh chiron 2>&1 | tail -10
```

Expected: clean builds; all tests pass except pre-existing `CHIRONOvfgStiefelAdamDescentTest` at line 5973 (baseline failure).

- [ ] **Step 3: Build trainer**

```bash
cd ~/dev/glades-trainer && bash build.sh 2>&1 | tail -5
```

Expected: clean build.

- [ ] **Step 4: Verify --cuda-graphs is recognized but auto-disabled on flagship**

```bash
cd ~/dev/glades-trainer
sh run.sh flagship --zloss-coef 1e-4 --qk-norm --cuda-graphs \
    --steps 1 --seed 1337 \
    --save /tmp/cg_smoke_pre 2>&1 | grep -E "cuda-graphs"
```

Expected: log line `[cuda-graphs] disabled — incompatible with --scfa (per-step CPU branching varies the kernel sequence)`. This is the auto-disable check at line 12644 firing. Confirms the baseline state we're about to fix.

---

## Phase 1: Blocker A — math-mode refactor

### Task 1.1: Add g_handleTf32 second cuBLAS handle in blasInit

**Files:**
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/cuda/gpu_blas.cu`

- [ ] **Step 1: Read the current g_handle initialization**

```bash
sed -n '263,295p' "/home/robert/dev/glades-ml/Backend/Machine Learning/Networks/cuda/gpu_blas.cu"
```

Confirm:
- Line 263-275 area: `blasInit()` creates `g_handle`, calls `cublasSetStream(g_handle, computeStream())`.
- Line 278: `cublasSetMathMode(g_handle, CUBLAS_TF32_TENSOR_OP_MATH)` is the init-time TF32 enable.

- [ ] **Step 2: Add g_handleTf32 + g_handleStrict global declarations**

In `gpu_blas.cu`, find the existing `g_handle` declaration (around line 50-80). Add immediately after:

```cpp
// Two-handle dispatch design for CUDA Graphs compatibility:
// g_handleStrict is always set to CUBLAS_DEFAULT_MATH (strict FP32).
// g_handleTf32 is always set to CUBLAS_TF32_TENSOR_OP_MATH (when CC >= 8.0).
// Per-call mathMode picks the appropriate handle WITHOUT toggling state.
// This eliminates cublasSet/GetMathMode calls inside the hot path, which
// are not capture-compatible (paradigm #51 ATLAS-COMPILE).
static cublasHandle_t g_handleStrict = nullptr;
static cublasHandle_t g_handleTf32   = nullptr;
```

If the file uses `NULL` instead of `nullptr` (C++98 codebase per CLAUDE.md), use `NULL`.

- [ ] **Step 3: Update blasInit to create both handles**

In `blasInit()`, find where `g_handle` is created and configured. Replace that block with:

```cpp
bool blasInit()
{
	if (g_initialized)
		return true;

	// Create the strict-FP32 handle.
	cublasStatus_t st = cublasCreate(&g_handleStrict);
	if (st != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] cublasCreate (strict) failed: %d\n", static_cast<int>(st));
		return false;
	}
	cublasSetStream(g_handleStrict, computeStream());
	cublasSetMathMode(g_handleStrict, CUBLAS_DEFAULT_MATH);

	// Create the TF32 handle.
	st = cublasCreate(&g_handleTf32);
	if (st != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] cublasCreate (tf32) failed: %d\n", static_cast<int>(st));
		cublasDestroy(g_handleStrict);
		g_handleStrict = NULL;
		return false;
	}
	cublasSetStream(g_handleTf32, computeStream());
	if (computeCapabilityMajor() >= 8)
	{
		cublasSetMathMode(g_handleTf32, CUBLAS_TF32_TENSOR_OP_MATH);
	}
	else
	{
		// Pre-Ampere: no TF32; both handles use CUBLAS_DEFAULT_MATH.
		cublasSetMathMode(g_handleTf32, CUBLAS_DEFAULT_MATH);
	}

	// Maintain g_handle as an alias to g_handleTf32 for backward compatibility
	// with code paths that haven't been updated yet.  Will be removed in
	// Task 1.4 once all callers route through the two-handle dispatch.
	g_handle = g_handleTf32;

	g_initialized = true;
	return true;
}
```

> **Note:** the existing `blasInit()` may have additional setup (cublasGetVersion logs, computeCapabilityMajor check, etc.). PRESERVE all of that — only replace the handle-creation block.

- [ ] **Step 4: Add a helper to dispatch handle by mathMode**

Above `sgemm_rowmajor_impl`, add:

```cpp
// Pick the appropriate handle for the requested mathMode.
// At init, g_handleTf32 is always TF32 (CC>=8) or DEFAULT (CC<8);
// g_handleStrict is always DEFAULT.  This dispatcher eliminates
// per-call cublasSet/GetMathMode, making the call sequence
// graph-captureable.
static inline cublasHandle_t pick_handle(cublasMath_t mathMode)
{
	if (mathMode == CUBLAS_TF32_TENSOR_OP_MATH)
		return g_handleTf32;
	return g_handleStrict;
}
```

- [ ] **Step 5: Build library; verify compile**

```bash
cd ~/dev/glades-ml && sh .configure.sh cuda 2>&1 | tail -5
```

Expected: clean build. No new warnings.

- [ ] **Step 6: Commit**

```bash
cd ~/dev/glades-ml
git add Backend/Machine\ Learning/Networks/cuda/gpu_blas.cu
git commit -m "$(cat <<'EOF'
Add g_handleStrict + g_handleTf32 two-handle dispatch in blasInit

Scaffolds the two-handle design needed to remove cublasGet/SetMathMode
toggling from the hot path (paradigm #51 ATLAS-COMPILE / cuda-graphs
compatibility).  g_handleStrict is always CUBLAS_DEFAULT_MATH;
g_handleTf32 is always CUBLAS_TF32_TENSOR_OP_MATH (CC>=8).  pick_handle
dispatcher selects per call without state changes.

g_handle is aliased to g_handleTf32 for backward compatibility with
code paths not yet updated; alias removed in subsequent task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 1.2: Refactor sgemm_rowmajor_impl to use two-handle dispatch

**Files:**
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/cuda/gpu_blas.cu` lines 121-179

- [ ] **Step 1: Read current `sgemm_rowmajor_impl`**

```bash
sed -n '121,180p' "/home/robert/dev/glades-ml/Backend/Machine Learning/Networks/cuda/gpu_blas.cu"
```

- [ ] **Step 2: Replace with two-handle version**

Replace the entire `sgemm_rowmajor_impl` function (lines 121-179) with:

```cpp
static bool sgemm_rowmajor_impl(cublasMath_t mathMode,
                                cublasOperation_t transa,
                                cublasOperation_t transb,
                                int M, int N, int K,
                                float alpha,
                                const float* A, int lda,
                                const float* B, int ldb,
                                float beta,
                                float* C, int ldc,
                                const char* label)
{
	if (!g_initialized && !blasInit())
		return false;

	// Two-handle dispatch: pick the handle whose math mode matches the
	// caller's request.  No state toggling, no cublasSet/GetMathMode in
	// the hot path — capture-compatible.
	cublasHandle_t h = pick_handle(mathMode);

	cublasStatus_t st = cublasSgemm(h,
	                                transa, transb,
	                                N, M, K,
	                                &alpha,
	                                B, ldb,
	                                A, lda,
	                                &beta,
	                                C, ldc);

	if (st != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] %s failed: %d (M=%d N=%d K=%d)\n",
		        label, static_cast<int>(st), M, N, K);
		return false;
	}
	return true;
}
```

This removes 30+ lines of get/set/restore-mode logic and replaces them with one `pick_handle` call.

- [ ] **Step 3: Build + run chiron suite**

```bash
cd ~/dev/glades-ml && sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests && sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests && bash test.sh chiron 2>&1 | tail -10
```

Expected: clean build, all tests preserve baseline (only pre-existing OVFG+Stiefel at line 5973 fails). **CRITICAL:** if any other test fails after this refactor, there's a handle/state mismatch — debug before proceeding.

- [ ] **Step 4: Commit**

```bash
cd ~/dev/glades-ml
git add Backend/Machine\ Learning/Networks/cuda/gpu_blas.cu
git commit -m "$(cat <<'EOF'
Refactor sgemm_rowmajor_impl to two-handle dispatch (no hot-path mode toggle)

Replaces the cublasGetMathMode + cublasSetMathMode + restore dance per
call with a single pick_handle() lookup that selects between
g_handleStrict (CUBLAS_DEFAULT_MATH) and g_handleTf32
(CUBLAS_TF32_TENSOR_OP_MATH).  The math mode is set once at init
(per Task 1.1) and never toggled in the hot path.  This is the
critical change for cuda-graphs (paradigm #51) compatibility.

Saves ~30 LOC + 2 cuBLAS API calls per GEMM.  Math is bit-identical:
the per-call cublasSgemm receives the same handle state it would have
received under the previous get/set/restore sequence.

Chiron unit-test suite preserves baseline (only pre-existing
OVFG+Stiefel failure at line 5973).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 1.3: Refactor sgemm_batched_pointer_impl similarly

**Files:**
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/cuda/gpu_blas.cu` lines 181-243

- [ ] **Step 1: Read current `sgemm_batched_pointer_impl`**

```bash
sed -n '181,244p' "/home/robert/dev/glades-ml/Backend/Machine Learning/Networks/cuda/gpu_blas.cu"
```

- [ ] **Step 2: Replace with two-handle version**

Replace the entire `sgemm_batched_pointer_impl` function with:

```cpp
static bool sgemm_batched_pointer_impl(cublasMath_t mathMode,
                                       cublasOperation_t transa,
                                       cublasOperation_t transb,
                                       int M, int N, int K,
                                       float alpha,
                                       float** Aarray, int lda,
                                       float** Barray, int ldb,
                                       float beta,
                                       float** Carray, int ldc,
                                       int batchCount,
                                       const char* label)
{
	if (!g_initialized && !blasInit())
		return false;
	if (!Aarray || !Barray || !Carray || batchCount <= 0)
		return true;

	cublasHandle_t h = pick_handle(mathMode);

	cublasStatus_t st = cublasSgemmBatched(h,
	                                       transa, transb,
	                                       N, M, K,
	                                       &alpha,
	                                       reinterpret_cast<const float* const*>(Barray), ldb,
	                                       reinterpret_cast<const float* const*>(Aarray), lda,
	                                       &beta,
	                                       Carray, ldc,
	                                       batchCount);

	if (st != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "[glades-cuda] %s failed: %d (M=%d N=%d K=%d batch=%d)\n",
		        label, static_cast<int>(st), M, N, K, batchCount);
		return false;
	}
	return true;
}
```

- [ ] **Step 3: Build + run chiron suite**

```bash
cd ~/dev/glades-ml && sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests && bash test.sh chiron 2>&1 | tail -10
```

Expected: baseline preserved.

- [ ] **Step 4: Commit**

```bash
cd ~/dev/glades-ml
git add Backend/Machine\ Learning/Networks/cuda/gpu_blas.cu
git commit -m "$(cat <<'EOF'
Refactor sgemm_batched_pointer_impl to two-handle dispatch

Same pattern as Task 1.2: pick_handle() instead of cublasGet/SetMathMode
toggle.  Removes hot-path mode toggling from batched-GEMM path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 1.4: Update set_tf32_enabled to flag-only

**Files:**
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/cuda/gpu_blas.cu` lines 249-258

- [ ] **Step 1: Replace `set_tf32_enabled`**

Find the existing `set_tf32_enabled` function (around line 250):

```cpp
void set_tf32_enabled(bool enabled)
{
	g_tf32_enabled = enabled;
	if (g_initialized && g_handle)
	{
		cublasSetMathMode(g_handle, enabled ? CUBLAS_TF32_TENSOR_OP_MATH
		                                     : CUBLAS_DEFAULT_MATH);
	}
}
```

Replace with:

```cpp
void set_tf32_enabled(bool enabled)
{
	// Flag-only API: just record the user's preference.  The two-handle
	// dispatch means callers explicitly request TF32 or strict via the
	// mathMode arg per call; this flag is read by callers that want to
	// HONOR the user's global toggle.
	//
	// IMPORTANT for cuda-graphs compatibility: NO cublasSetMathMode
	// happens here in the hot path.  If a runtime toggle is needed
	// during training (rare), it changes the future per-call dispatch
	// but never the existing handle state.
	g_tf32_enabled = enabled;
}
```

- [ ] **Step 2: Audit `get_tf32_enabled` and any consumers**

```bash
grep -nE "get_tf32_enabled|g_tf32_enabled" "/home/robert/dev/glades-ml/Backend/Machine Learning/Networks/cuda/gpu_blas.cu" | head -10
```

The flag is read by code that decides the mathMode for each GEMM. After this refactor, the flag's semantics change from "current global cuBLAS state" to "user preference for default mathMode". Callers that read `g_tf32_enabled` to PICK between TF32 and strict for a GEMM call still work correctly — they just now use the flag as an input to the two-handle dispatch instead of as a global state.

If any caller does `cublasSetMathMode(g_handle, ...)` directly outside of `set_tf32_enabled`, that's a problem — search for it:

```bash
grep -rn "cublasSetMathMode" "/home/robert/dev/glades-ml/Backend/Machine Learning/Networks/cuda/" | head -10
```

Expected: only the 3 sites already touched (Task 1.1's init, Task 1.4's flag-only set_tf32_enabled). If any other site exists, refactor it to either (a) use the appropriate handle directly, or (b) remove the call entirely.

- [ ] **Step 3: Build + run chiron suite**

```bash
cd ~/dev/glades-ml && sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests && bash test.sh chiron 2>&1 | tail -10
```

Expected: baseline preserved.

- [ ] **Step 4: Commit**

```bash
cd ~/dev/glades-ml
git add Backend/Machine\ Learning/Networks/cuda/gpu_blas.cu
git commit -m "$(cat <<'EOF'
Make set_tf32_enabled flag-only (no hot-path cublasSetMathMode)

Last step of Blocker A: remove the runtime cublasSetMathMode that
set_tf32_enabled() was making.  The two-handle design (Task 1.1)
means callers pick handle by mathMode arg, NOT by global state, so
the runtime toggle becomes a flag that informs future per-call
dispatch decisions.

This completes the gpu_blas.cu refactor for cuda-graphs compatibility.
No more cublasSet/GetMathMode calls in the hot path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 1.5: Audit g_handleSide (the side-stream handle in sgemm_rowmajor_fast16bf_side)

**Files:**
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/cuda/gpu_blas.cu`

The side-stream cuBLAS handle `g_handleSide` (created around lines 100-115) is used by `--scfa-parallel-branches` which is NOT in the flagship recipe. But it still uses `cublasSetMathMode` once at init. Verify this is OK for capture (init-time set is fine; only hot-path set is the problem).

- [ ] **Step 1: Read the g_handleSide init**

```bash
sed -n '85,120p' "/home/robert/dev/glades-ml/Backend/Machine Learning/Networks/cuda/gpu_blas.cu"
```

- [ ] **Step 2: Verify `g_handleSide` does NOT have a hot-path set/restore dance**

```bash
grep -nE "sgemm_rowmajor_fast16bf_side|g_handleSide" "/home/robert/dev/glades-ml/Backend/Machine Learning/Networks/cuda/gpu_blas.cu" | head -20
```

If the side-stream functions also have hot-path `cublasSet/GetMathMode`, they need the same refactor. Apply the same pattern as Tasks 1.2/1.3 (two-handle dispatch for the side handle: `g_handleSideStrict` + `g_handleSideTf32`).

If `g_handleSide` only has init-time `cublasSetMathMode` (line 114-115), no change needed — init is captured-safe.

- [ ] **Step 3: If refactor needed, apply + build + test. If not, skip.**

For the flagship recipe specifically: `--scfa-parallel-branches` is OFF (per iter 71 NULL on Ada), so the side-stream path is not exercised. We can DEFER side-handle refactor to a follow-up if needed; flag this as a known limitation in the result doc.

- [ ] **Step 4: Commit (skip if no changes)**

```bash
cd ~/dev/glades-ml
git status -s "Backend/Machine Learning/Networks/cuda/gpu_blas.cu"
# If clean (no further changes from Step 3), skip the commit.
# Otherwise:
# git add Backend/Machine\ Learning/Networks/cuda/gpu_blas.cu
# git commit -m "..."
```

---

## Phase 2: Blocker C — explicit mutual exclusion + auto-disable cleanup

### Task 2.1: Add --cuda-graphs ⨯ --fp8-attn mutual exclusion

**Files:**
- Modify: `~/dev/glades-trainer/trainer/chiron_main.cpp` (config-validation block, near the existing `--ul2-enabled` ⨯ `--medal-train` mutex)

- [ ] **Step 1: Find the existing UL2 mutual-exclusion block**

```bash
grep -nE "ul2-enabled and --medal-train are mutually exclusive" /home/robert/dev/glades-trainer/trainer/chiron_main.cpp
```

This locates the validation block from the UL2 arc (around line 12633-12637).

- [ ] **Step 2: Add cuda-graphs ⨯ fp8-attn mutex**

Immediately AFTER the existing `if (cfg.ul2Enabled) { ... }` validation block (around line 12638), add:

```cpp
	// --cuda-graphs ⨯ --fp8-attn mutual exclusion.
	// fp8Attn uses a per-step `static bool s_fp8_runtime_disabled` CPU
	// branch (chiron_main.cpp:9085-9110) that varies the kernel sequence
	// step-to-step.  CUDA graphs capture would record the first step's
	// branch and replay it unconditionally, masking later fallbacks.
	// fp8Attn is not in the flagship recipe (NULL per iter 55 and on
	// CUDA 13.2); enforce the exclusion at CLI parse rather than runtime.
	if (cfg.cudaGraphs && cfg.fp8Attn)
	{
		log_error("chiron", "--cuda-graphs is mutually exclusive with --fp8-attn "
		          "(paradigm #50 runtime fallback uses CPU branching; not "
		          "capture-compatible).  Disable one.\n");
		return 1;
	}
```

If `log_error` isn't the API name, use whatever matches the existing pattern (`log_warn` followed by `return 1` is also fine).

- [ ] **Step 3: Build + verify the error fires**

```bash
cd ~/dev/glades-trainer && bash build.sh 2>&1 | tail -5
./build/glades_chiron_train --cuda-graphs --fp8-attn 2>&1 | head -5
```

Expected: error message + exit code 1.

- [ ] **Step 4: Commit**

```bash
cd ~/dev/glades-trainer
git add trainer/chiron_main.cpp
git commit -m "$(cat <<'EOF'
Add --cuda-graphs ⨯ --fp8-attn mutual exclusion

fp8Attn's runtime fallback (static bool s_fp8_runtime_disabled at
chiron_main.cpp:9085-9110) introduces a per-step CPU branch that
varies the kernel sequence.  Cuda-graphs capture would record the
first step's path and replay incorrectly if fp8 ever fails mid-run.

The flag is not in the flagship recipe (NULL per iter 55 and confirmed
on CUDA 13.2).  Enforcing the exclusion at CLI parse avoids subtle
correctness bugs.

Mutual exclusions for --cuda-graphs ⨯ --medal-train and
⨯ --ul2-enabled were added by earlier arcs (LayerDrop ef9d540 / UL2
ac68907 added these to the CUDA-graphs auto-disable list).  This
commit hardens fp8Attn to the same explicit-error semantics.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 2.2: Drop cfg.scfa from cuda-graphs auto-disable list

**Files:**
- Modify: `~/dev/glades-trainer/trainer/chiron_main.cpp` line 12644

- [ ] **Step 1: Locate the auto-disable check**

```bash
grep -nE "cfg\.scfa.*incompat|incompat.*scfa" /home/robert/dev/glades-trainer/trainer/chiron_main.cpp | head -5
```

Should match around line 12644:
```cpp
if (cfg.scfa)             { incompat = true; incompatReason = "--scfa"; }
```

- [ ] **Step 2: Remove the cfg.scfa check**

Replace:
```cpp
if (cfg.scfa)             { incompat = true; incompatReason = "--scfa"; }
```

with a comment explaining the change:
```cpp
// 2026-05-23: cfg.scfa removed from auto-disable list.  After the
// gpu_blas.cu math-mode refactor (cuda-graphs arc), SCFA's flagship-
// recipe kernel sequence is graph-compatible.  --scfa-parallel-branches
// (the only SCFA sub-flag with cross-stream events) is gated separately
// below; it's NOT in the flagship recipe (NULL per iter 71 on Ada).
// --scfa-fuse-streams (default off in flagship-non-parallel paths) is
// also graph-compatible — its kernels run on the main compute stream
// with no cross-stream events.
```

Then add the more-specific gates for the still-incompat scenarios:

```cpp
		// SCFA sub-flags that ARE still graph-incompat (the cross-stream
		// event paths inside --scfa-parallel-branches).  These are not in
		// the flagship recipe but guard against user mis-configuration.
		else if (cfg.scfaParallelBranches) { incompat = true; incompatReason = "--scfa-parallel-branches (cross-stream events not in Relaxed capture mode)"; }
```

- [ ] **Step 3: Build + verify auto-disable behavior**

```bash
cd ~/dev/glades-trainer && bash build.sh 2>&1 | tail -5
# Flagship recipe with --cuda-graphs should NO LONGER auto-disable:
sh run.sh flagship --zloss-coef 1e-4 --qk-norm --cuda-graphs \
    --steps 1 --seed 1337 \
    --save /tmp/cg_smoke_post_dropdisable 2>&1 | grep -E "cuda-graphs" | head -5
```

Expected: no `disabled — incompatible with --scfa` line. May still see the EXPERIMENTAL warning at line 12663 (about cublas math-mode — which should now be FIXED by Phase 1, so re-verify the warning is no longer triggered after Phase 1+2 land).

If `--cuda-graphs` is still auto-disabled, look at the incompatReason in the log line and investigate.

- [ ] **Step 4: Verify --scfa-parallel-branches still auto-disables (as expected)**

```bash
# This config SHOULD still auto-disable cuda-graphs:
sh run.sh flagship --zloss-coef 1e-4 --qk-norm --cuda-graphs --scfa-parallel-branches \
    --steps 1 --seed 1337 \
    --save /tmp/cg_smoke_parallelb 2>&1 | grep -E "cuda-graphs" | head -3
```

Expected: `disabled — incompatible with --scfa-parallel-branches`.

- [ ] **Step 5: Commit**

```bash
cd ~/dev/glades-trainer
git add trainer/chiron_main.cpp
git commit -m "$(cat <<'EOF'
Drop cfg.scfa from --cuda-graphs auto-disable list

After Phase 1's gpu_blas.cu math-mode refactor lands, SCFA's flagship-
recipe kernel sequence is graph-compatible.  The original iter 55 NULL
result that put --scfa on the auto-disable list cited two issues:
(a) cublasGet/SetMathMode hot-path toggle, now fixed in gpu_blas.cu;
(b) --scfa-parallel-branches cross-stream events, which remain
incompatible but are NOT in the flagship recipe (NULL per iter 71 on
Ada).

The check is narrowed to gate only the still-incompat sub-flag
(--scfa-parallel-branches).  Flagship recipe + --cuda-graphs no longer
auto-disables.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 2.3: Remove the EXPERIMENTAL warning (Phase 1 fixed the underlying issue)

**Files:**
- Modify: `~/dev/glades-trainer/trainer/chiron_main.cpp` lines 12656-12663

- [ ] **Step 1: Read the existing warning block**

```bash
sed -n '12655,12665p' /home/robert/dev/glades-trainer/trainer/chiron_main.cpp
```

The block has a comment block explaining the cublas-math-mode issue + a `log_warn` that emits the EXPERIMENTAL warning. After Phase 1 (Blocker A), the underlying issue is fixed.

- [ ] **Step 2: Replace the warning with a positive confirmation**

Replace lines ~12656-12663 with:

```cpp
		} else {
			// 2026-05-23: gpu_blas.cu math-mode refactor (cuda-graphs arc)
			// removed cublasGet/SetMathMode from the hot path via two-handle
			// dispatch.  Capture should succeed at step 0 with the flagship
			// recipe.  If capture fails, log_warn paths below trigger the
			// transparent fallback.
			log_info("chiron","[cuda-graphs] capture ENABLED — two-handle dispatch in gpu_blas.cu is graph-compatible.\n");
		}
```

- [ ] **Step 3: Build + verify the log line changes**

```bash
cd ~/dev/glades-trainer && bash build.sh 2>&1 | tail -3
sh run.sh flagship --zloss-coef 1e-4 --qk-norm --cuda-graphs \
    --steps 1 --seed 1337 \
    --save /tmp/cg_smoke_post_warning 2>&1 | grep -E "cuda-graphs" | head -5
```

Expected: `[cuda-graphs] capture ENABLED — two-handle dispatch in gpu_blas.cu is graph-compatible.` (or whatever phrasing — matches the comment update).

- [ ] **Step 4: Commit**

```bash
cd ~/dev/glades-trainer
git add trainer/chiron_main.cpp
git commit -m "$(cat <<'EOF'
Remove cuda-graphs EXPERIMENTAL warning (Phase 1 fixed the issue)

The EXPERIMENTAL warning at chiron_main.cpp:12656-12663 was added when
cublasGet/SetMathMode toggling in sgemm_rowmajor was a known capture
blocker (research/CUDA_GRAPHS_FIX4_NOTES.md).  Phase 1 of the cuda-
graphs arc refactored gpu_blas.cu to two-handle dispatch (no hot-path
mode toggling), so the warning is stale.

Replaced with a positive confirmation that the math-mode refactor
landed and capture is expected to succeed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3: G1 pilot — empirically verify capture succeeds + measure wall

### Task 3.1: G0 500-step baseline (no graphs)

**Files:** No code changes; verification only.

- [ ] **Step 1: Launch G0**

```bash
cd ~/dev/glades-trainer
sh run.sh flagship --zloss-coef 1e-4 --qk-norm \
    --steps 500 --seed 1337 \
    --save database/checkpoints/chiron_1B_T16384_g0_500step 2>&1 \
    | tee logs/g0_500step_$(date +%Y%m%d_%H%M).log
```

Expected: ~5 min wall.

- [ ] **Step 2: Extract G0 metrics**

```bash
grep -E "val val|tok/s.*wall|chiron-train\] done" logs/g0_500step_*.log | tail -10
```

Record:
- `G0_NLL_500` (final val NLL at step 500).
- `G0_TOKS` (steady-state tok/s, typically the final-step value).
- `G0_WALL` (total wall time).

### Task 3.2: G1 500-step with --cuda-graphs (the main empirical test)

- [ ] **Step 1: Launch G1**

```bash
cd ~/dev/glades-trainer
sh run.sh flagship --zloss-coef 1e-4 --qk-norm --cuda-graphs \
    --steps 500 --seed 1337 \
    --save database/checkpoints/chiron_1B_T16384_g1_graphs_500step 2>&1 \
    | tee logs/g1_500step_$(date +%Y%m%d_%H%M).log
```

Expected: ~4.5-4.7 min wall (5-7% faster than G0 if graphs work).

- [ ] **Step 2: Verify capture succeeded**

```bash
grep -E "cuda-graphs" logs/g1_500step_*.log | head -10
```

EXPECTED (success case):
```
[cuda-graphs] capture ENABLED — two-handle dispatch in gpu_blas.cu is graph-compatible.
[cuda-graphs] graph captured successfully at step 0
... (no further [cuda-graphs] lines except possible RLG invalidations)
```

FAILURE CASES + actions:
- `[cuda-graphs] beginCapture failed` → some non-capture-compatible API call exists. Read the surrounding line for context; identify the kernel; debug.
- `[cuda-graphs] launch failed at step N` → captured graph fails to replay. Investigate the step-N kernel sequence vs step-0.
- `[cuda-graphs] endCaptureAndInstantiate failed` → graph too large or has issues at instantiate time. Inspect the cuda error code.
- `[cuda-graphs] forward failed during capture at step N` → forward path uses a kernel that's not capture-compatible at the time of capture. Identify + fix.

If capture FAILS at this stage, this task becomes ITERATIVE: identify the offending kernel, fix, re-run G1, repeat until capture succeeds. Each iter is ~5 min wall.

- [ ] **Step 3: Extract G1 metrics**

```bash
grep -E "val val|tok/s.*wall|chiron-train\] done" logs/g1_500step_*.log | tail -10
```

Record:
- `G1_NLL_500` (final val NLL at step 500).
- `G1_TOKS` (steady-state tok/s).
- `G1_WALL` (total wall time).

- [ ] **Step 4: Apply G1 pilot gate**

Compute:
```
ΔNLL = |G1_NLL_500 - G0_NLL_500|
Δtoks_pct = (G1_TOKS - G0_TOKS) / G0_TOKS * 100
```

```
if ΔNLL > 0.0001:
    BUG → math diverged.  Debug the gpu_blas.cu refactor.
elif Δtoks_pct < 1.0:
    WALL CEILING → kernel-launch overhead is smaller than expected at this scale.
                   Document as known finding; arc result is "infrastructure
                   fixed but no measurable wall benefit".  Library + trainer
                   code stays; the flag is now available for future scales.
elif Δtoks_pct >= 1.0:
    PASS → proceed to G2 30k Phase-2 retrain.
```

- [ ] **Step 5: Record pilot decision**

Create or append to `~/dev/glades-ml/research/CUDA_GRAPHS_DRAFT.md`:

```markdown
# CUDA-Graphs Arc — G1 Pilot Results Draft

## G0 (regstack Phase 2 ship baseline, 500-step single-seed seed=1337)
- Val NLL @ 500: <G0_NLL_500>
- tok/s: <G0_TOKS>
- Wall: <G0_WALL>

## G1 (regstack + --cuda-graphs, 500-step single-seed seed=1337)
- Val NLL @ 500: <G1_NLL_500>
- tok/s: <G1_TOKS>
- Wall: <G1_WALL>
- Capture: <SUCCESS / FAILED>
- Log highlights: <relevant [cuda-graphs] log lines>

## Deltas
- ΔNLL = |<G1_NLL_500>| - <G0_NLL_500>| = <delta_nll>
- Δtok/s % = ((G1_TOKS - G0_TOKS) / G0_TOKS) * 100 = <delta_toks_pct>%

## Pilot decision
- Gate: ΔNLL ≤ 0.0001 nat AND Δtok/s ≥ +1.0% → PASS
- Decision: <PASS / WALL_CEILING / BUG>
```

### Task 3.3 (conditional): Iterative debugging if G1 capture fails

**Run only if Task 3.2 Step 2 showed capture failure.**

For each failure type, follow the specific debugging path:

- [ ] **Step 1: If `beginCapture failed`:**

```bash
# Look at the recent CUDA error log lines BEFORE the beginCapture log:
grep -B 5 "beginCapture failed" logs/g1_500step_*.log
```

Common cause: a kernel was launched BEFORE the capture started. Trace back to find the offending kernel.

- [ ] **Step 2: If `launch failed at step N` (N > 0):**

This means capture succeeded at step 0 but replay fails at step N. Most likely cause: a per-step CPU branch that wasn't accounted for. Re-examine the auto-disable list audits.

- [ ] **Step 3: If `endCaptureAndInstantiate failed`:**

Likely a cuda error during instantiation. Get the error code from the log; check NVIDIA docs.

- [ ] **Step 4: After identifying the offending kernel/branch, fix it + re-run G1**

The fix likely is one of:
- Remove a hidden per-step CPU branch.
- Force a code path to a single branch when `cfg.cudaGraphs` is true.
- Add to mutual-exclusion list if the conflict is fundamental.

Repeat Task 3.2 until G1 PASSes.

---

## Phase 4: G2 30k Phase-2 retrain (gated on G1 PASS)

### Task 4.1: Launch G2 30k Phase-2

**Files:** No code changes.

- [ ] **Step 1: Launch G2**

```bash
cd ~/dev/glades-trainer
sh run.sh flagship --zloss-coef 1e-4 --qk-norm --cuda-graphs \
    --steps 30000 --seed 1337 \
    --save database/checkpoints/chiron_1B_T16384_regstack_graphs_phase2 2>&1 \
    | tee logs/g2_30k_$(date +%Y%m%d_%H%M).log
```

Expected: ~4.5-5 h wall (with the +2-5% wall improvement, faster than ship's ~4.7 h).

- [ ] **Step 2: Extract G2 metrics**

```bash
grep -E "step 30000|chiron-train\] final-val|tok/s.*wall|peak VRAM" logs/g2_30k_*.log | tail -10
```

Record:
- `G2_NLL_30k` (final val NLL at step 30000).
- `G2_TOKS_final` (steady-state final tok/s).
- `G2_VRAM_peak` (peak VRAM throughout the run).
- Full val NLL trajectory (each `[val val]` line in the log).

- [ ] **Step 3: Apply G2 30k Phase-2 gate**

Compute:
```
ΔNLL_30k = |G2_NLL_30k - 3.5734|
Δtok/s_pct = (G2_TOKS_final - 28072) / 28072 * 100
```

```
if ΔNLL_30k > 0.001:
    BUG_OR_DRIFT → debug or document as wider FP32 noise floor than expected.
elif Δtok/s_pct >= 2.0 AND ΔNLL_30k <= 0.001 AND VRAM <= 15.72 GB:
    SHIP → archive prior ship, publish new ship, update CLAUDE.md.
elif Δtok/s_pct < 2.0 AND ΔNLL_30k <= 0.001:
    SHIP_WALL_BELOW_TARGET → still bit-identical NLL, but wall improvement
        below the +2% gate.  Document; library/trainer code stays as opt-in.
        Decision to ship at the slightly lower wall improvement is a user call.
else:
    FAIL → publish honest negative; close arc.
```

---

## Phase 5: Documentation + ship

### Task 5.1: Write result doc

**Files:**
- Create: `~/dev/glades-ml/research/CUDA_GRAPHS_<RESULT>_2026_MM_DD.md`

- [ ] **Step 1: Choose result-doc filename**

Based on Task 4.1 Step 3 outcome:
- `CUDA_GRAPHS_PASS_2026_MM_DD.md` — Δtok/s ≥ +2%, NLL bit-identical.
- `CUDA_GRAPHS_PASS_WALL_BELOW_TARGET_2026_MM_DD.md` — Δtok/s < +2%, NLL bit-identical.
- `CUDA_GRAPHS_FAIL_2026_MM_DD.md` — NLL drift > 0.001 nat OR capture never succeeded.

- [ ] **Step 2: Populate using prior doc as template**

Use `~/dev/glades-ml/research/LAYERDROP_5K_FAIL_2026_05_23.md` or `UL2_5K_FAIL_2026_05_23.md` as the template. For CUDA Graphs specifically, the doc should include:
- G0 / G1 / G2 throughput numbers.
- G0 → G2 NLL trajectory (full table — bit-identicality demonstration).
- The three Blocker fixes (A, B, C) with commit references.
- Capture log highlights.
- Position-stratified eval (forced-S at step 30000, identical to ship's).
- Final disposition.

- [ ] **Step 3: Commit the result doc**

```bash
cd ~/dev/glades-ml
git add research/CUDA_GRAPHS_*.md
git commit -m "$(cat <<'EOF'
Document CUDA Graphs arc result (PASS|WALL_BELOW_TARGET|FAIL) — 2026-MM-DD

G0 baseline / G1 pilot / G2 30k Phase-2 results.  [Insert one-sentence
headline result + headline tok/s number.]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 5.2: Update CLAUDE.md flagship pointer (only if G2 PASSes)

**Files:**
- Modify: `~/dev/glades-ml/CLAUDE.md`

- [ ] **Step 1: Update "Current Production Flagship" block**

If G2 PASSes:
- Checkpoint: `chiron_1B_T16384_regstack_graphs_phase2.final`.
- Stack: regstack Phase 2 (Z-loss + QK-Norm) **PLUS** CUDA Graphs production-readiness.
- Perf: `<G2_TOKS_final>` tok/s @ T=16384 (was 28,072 at regstack Phase 2 ship, +`<Δtok/s_pct>`%); same VRAM ~14.97 GB peak.
- Final val NLL: `<G2_NLL_30k>` @ step 30000 (bit-identical to regstack ship 3.5734).
- Reproduce: `sh run.sh flagship --zloss-coef 1e-4 --qk-norm --cuda-graphs`.
- Full spec / evidence: spec + result doc.

Move prior "regstack Phase 2 ship" block to a "kept for context" subsection.

- [ ] **Step 2: Commit CLAUDE.md update**

```bash
cd ~/dev/glades-ml
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
Ship CUDA Graphs CHIRON 1B as new production flagship

New ship: chiron_1B_T16384_regstack_graphs_phase2.final
Stack: regstack Phase 2 (Z-loss + QK-Norm) + CUDA Graphs production-
readiness (paradigm #51 ATLAS-COMPILE).

Final val NLL <N> @ 30k (bit-identical to regstack ship 3.5734).
Throughput <T> tok/s (+<Δ%> vs regstack 28,072).
Peak VRAM <V> GB.

Reproduce: sh run.sh flagship --zloss-coef 1e-4 --qk-norm --cuda-graphs

Math is unchanged from regstack Phase 2 ship; the win is from
removing per-kernel launch overhead via graph capture/replay.  The
new ship's checkpoint VALUES are bit-identical to the regstack ship's
(modulo sub-ULP FMA reordering); both checkpoints are equivalent at
inference.  The "ship swap" is a recipe-only change.

Spec: docs/superpowers/specs/2026-05-23-chiron-1b-cuda-graphs-design.md
Evidence: research/CUDA_GRAPHS_PASS_2026_MM_DD.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 5.3: Memory entry

- [ ] **Step 1: Add memory entry**

Create `/home/robert/.claude/projects/-home-robert-dev-glades-ml/memory/cuda_graphs_ship.md`:

```markdown
---
name: cuda_graphs_ship
description: CUDA Graphs arc PASS — new CHIRON 1B flagship adds --cuda-graphs to ship recipe. Throughput <T> tok/s (+<Δ%> vs regstack 28,072), bit-identical NLL @ 30k. Pure-infra arc; math unchanged.
metadata:
  type: project
---

[Fill in: PASS headline, the three blocker fixes (commits in
glades-ml + glades-trainer), G0→G1→G2 wall measurements, pointer to
result doc + spec.]
```

- [ ] **Step 2: Update MEMORY.md index**

Add ONE line at top of `## Topics` in MEMORY.md:

```markdown
- [CUDA GRAPHS SHIP 2026-MM-DD](cuda_graphs_ship.md) — New CHIRON 1B flagship: regstack Phase 2 + --cuda-graphs (paradigm #51). Throughput <T> tok/s (+<Δ%> vs 28,072 regstack), bit-identical NLL @ 30k. Math unchanged; win from kernel-launch overhead elimination.
```

Keep under ~200 chars per index convention.

---

## Self-review

- **Spec §1 motivation:** Phase 0 + Phases 1-2 land the infrastructure. ✓
- **Spec §2.1 Blocker A (math-mode refactor):** Phase 1 Tasks 1.1-1.5. ✓
- **Spec §2.1 Blocker B (scfaFuseStreams):** Phase 2 + Phase 3 (empirical test). ✓
- **Spec §2.1 Blocker C (fp8-attn mutex):** Phase 2 Task 2.1. ✓
- **Spec §2.2 audit checklist (regstack mechanisms):** verified empirically at Task 3.2 (capture succeeds). ✓
- **Spec §2.3 drop --scfa from auto-disable list:** Phase 2 Task 2.2. ✓
- **Spec §2.6 bit-identicality at NLL:** verified at Task 3.2 Step 4 (G1 ΔNLL ≤ 0.0001) + Task 4.1 Step 3 (G2 ΔNLL ≤ 0.001). ✓
- **Spec §3.1 run plan (G0/G1/G2):** Phases 3-4. ✓
- **Spec §3.2 gate criteria:** Tasks 3.2 Step 4 + 4.1 Step 3. ✓
- **Spec §3.3 PASS handling:** Phase 5. ✓
- **Spec §3.4 FAIL handling:** Task 4.1 Step 3 + Task 5.1 conditional filenames. ✓
- **Spec §4 risks (R-Graphs-1 through R-Graphs-10):** addressed across tasks. R-Graphs-1 (math-mode mismatch) by build+test at Tasks 1.2-1.5. R-Graphs-3 (more per-step branches) by Task 3.3 iterative debug. R-Graphs-5 (small wall savings) by Task 3.2 Step 4 WALL_CEILING decision. R-Graphs-7 (refactor breaks non-flagship paths) by chiron unit-test suite verification at each refactor step. R-Graphs-10 (NaN at GEMM mode-mismatch) caught by unit tests at Tasks 1.2-1.5. ✓
- **Spec §5 out of scope:** explicit in spec; plan doesn't venture there. ✓
- **Spec §6 deliverables:** matches plan tasks. ✓

This plan implements the CUDA-graphs arc spec at `docs/superpowers/specs/2026-05-23-chiron-1b-cuda-graphs-design.md`.
