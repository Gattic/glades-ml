# CHIRON 1B Regularization Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Z-loss, QK-Norm, and simple MTP (multi-token prediction) regularizers to the CHIRON 1B production flagship, gated by a 5-run × 5k pilot arc and a 30k Phase-2 retrain.

**Architecture:** Each mechanism lives behind a config flag in `TransformerRunConfig` (defaulting to a no-op value that preserves strict bit-identicality). Z-loss adds an auxiliary term to the readout CE backward; QK-Norm L2-normalizes Q/K per head and substitutes a learnable per-head scalar γ for `1/√d_h` at all 8 attention-scale sites; MTP shares the tied readout via one extra linear projection to predict the t+2 token.

**Tech Stack:** C++98 + CUDA (CUDA 13.2 toolchain), shared with the `libglades.so` library. Trainer is a separate repo (`~/dev/glades-trainer`) that wraps the library and exposes CLI flags. Tests use the project's `ASSERT(failmsg, predicate)` macro from `unit-tests/unit-test.h`.

**Repos involved:**
- `~/dev/glades-ml/` — library (most changes here)
- `~/dev/glades-trainer/` — trainer binary + CLI + run.sh recipe

**Execution recommendation:** Run this plan in an isolated git worktree (per `superpowers:using-git-worktrees`). All training runs use the existing `~/dev/glades-trainer` install — no need to worktree that repo.

**Source-of-truth spec:** `docs/superpowers/specs/2026-05-22-chiron-1b-regularization-stack-design.md`. Refer to it for motivation, gates, and risks. This plan implements that spec.

---

## Phase 0: Setup + config scaffolding

### Task 0.1: Baseline build sanity check

**Files:** No file changes; verification only.

- [ ] **Step 1: Confirm starting branch state**

```bash
cd ~/dev/glades-ml
git status
git log --oneline -3
```

Expected: clean working tree on branch `chiron2`, top commit is `807871f46 Pre-register CHIRON 1B regularization stack design`.

- [ ] **Step 2: Build library with CUDA (clean baseline)**

```bash
cd ~/dev/glades-ml
sh .configure.sh cuda
```

Expected: build completes without errors. `~/.local/lib/libglades.so` updated.

- [ ] **Step 3: Build unit tests**

```bash
cd ~/dev/glades-ml/unit-tests/build && sh .configure.sh cuda
```

Expected: build completes.

- [ ] **Step 4: Run existing chiron tests as a sanity check**

```bash
cd ~/dev/glades-ml/unit-tests && bash test.sh chiron 2>&1 | tail -30
```

Expected: all ASSERTs pass. Note any pre-existing test failures so they aren't blamed on this work.

### Task 0.2: Add config fields (no-op defaults)

**Files:**
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/training_config.h`

- [ ] **Step 1: Add Z-loss, QK-Norm, MTP field declarations**

Open `Backend/Machine Learning/Networks/training_config.h`. After the existing `rlgInitialLayers` field declaration in `TransformerRunConfig` (around line 223, just before the constructor), insert:

```cpp
	// Z-loss auxiliary objective (paradigm shift: PaLM/T5/Gemini-style logit
	// regularization). When > 0, adds zlossCoef * mean_t(log²(Z_t)) to the
	// readout loss, where Z_t = sum_v exp(logit_{t,v}). Default 0.0f =
	// disabled, math bit-identical to baseline. Recommended: 1e-4 (PaLM
	// default). Improves training stability under FP8 readout by bounding
	// logit magnitudes.
	float zlossCoef;

	// QK-Norm (paradigm shift: DeepSeek-V3 / modern Llama). When true, Q and
	// K are L2-normalized per-head before the attention dot product, and the
	// constant `1/sqrt(dHead)` scale is replaced by a learnable per-head
	// scalar γ. Default false = disabled, math bit-identical to baseline.
	bool qkNormEnabled;

	// Initial value for the QK-Norm γ scalar (per-head, shared across all
	// blocks at init). When <= 0 (default), initialized to log2(T) at first
	// forward pass per DeepSeek-V3 init. Set explicitly to override.
	float qkNormGammaInit;

	// Multi-token prediction depth (paradigm shift: DeepSeek-V3). When > 0,
	// adds N auxiliary heads each predicting the token at offset +k for
	// k in {2, ..., N+1}. The current implementation supports depth = 1 (a
	// single +2-offset head). Default 0 = disabled, math bit-identical.
	int mtpDepth;

	// MTP auxiliary loss coefficient. Each MTP head's CE is weighted by
	// (mtpCoef / mtpDepth) and added to the main CE loss. DeepSeek-V3 uses
	// 0.1 after a brief warmup at 0.3. Default 0.1f.
	float mtpCoef;
```

- [ ] **Step 2: Add field initializers in the constructor**

In the same file, the `TransformerRunConfig()` constructor initializer list ends at `rlgInitialLayers(0)` followed by the body `{ }` (around lines 257-259). Modify the initializer list so that `rlgInitialLayers(0)` has a trailing comma and append:

```cpp
	      rlgInitialLayers(0),
	      zlossCoef(0.0f),
	      qkNormEnabled(false),
	      qkNormGammaInit(0.0f),
	      mtpDepth(0),
	      mtpCoef(0.1f)
```

(Leave the `{ }` body unchanged.)

- [ ] **Step 3: Rebuild library, expect no errors**

```bash
cd ~/dev/glades-ml
sh .configure.sh cuda 2>&1 | tail -10
```

Expected: build completes, `libglades.so` updated. No new warnings about uninitialized members.

- [ ] **Step 4: Rebuild unit tests + run chiron suite**

```bash
cd ~/dev/glades-ml/unit-tests/build && sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests && bash test.sh chiron 2>&1 | tail -10
```

Expected: same pass/fail status as Task 0.1 step 4. The new fields are unused so nothing changes.

- [ ] **Step 5: Commit**

```bash
cd ~/dev/glades-ml
git add Backend/Machine\ Learning/Networks/training_config.h
git commit -m "$(cat <<'EOF'
Add zlossCoef, qkNormEnabled, qkNormGammaInit, mtpDepth, mtpCoef config fields

All default to no-op values (0.0 / false / 0). No behavior change.
Scaffolding for the regularization-stack spec; subsequent commits wire
them through the forward/backward passes guarded by these flags.
EOF
)"
```

---

## Phase 1: Z-loss

### Task 1.1: Strict-bar parity unit test for Z-loss disabled-default

**Files:**
- Modify: `~/dev/glades-ml/unit-tests/Backend/Machine Learning/chiron-test.cpp`
- Modify: `~/dev/glades-ml/unit-tests/Backend/Machine Learning/chiron-test.h`
- Modify: `~/dev/glades-ml/unit-tests/main.cpp` (registers the new test)

- [ ] **Step 1: Read the chiron-test.h header to find the registration pattern**

```bash
head -60 ~/dev/glades-ml/unit-tests/Backend/Machine\ Learning/chiron-test.h
```

Note the existing function declarations (e.g., `CHIRONLocalAttentionFullWindowParityTest()`). You'll mirror this pattern.

- [ ] **Step 2: Declare the new test function in chiron-test.h**

Open `unit-tests/Backend/Machine Learning/chiron-test.h`. Find the block of existing test function declarations and append:

```cpp
void CHIRONZlossDisabledParityTest();
void CHIRONZlossEnabledMathTest();
void CHIRONQkNormDisabledParityTest();
void CHIRONQkNormEnabledMathTest();
void CHIRONMtpDisabledParityTest();
void CHIRONMtpTargetShiftTest();
```

(All six are declared up-front so we don't have to edit the header six more times.)

- [ ] **Step 3: Find where chiron tests are invoked in main.cpp**

```bash
grep -n "CHIRON" ~/dev/glades-ml/unit-tests/main.cpp | head -20
```

Note the registration pattern. Add new lines registering the six functions above next to the existing CHIRON test calls (e.g., right after `CHIRONLocalAttentionFullWindowParityTest()`).

- [ ] **Step 4: Implement the disabled-parity test**

In `unit-tests/Backend/Machine Learning/chiron-test.cpp`, at the bottom of the file (after the last existing test function), add:

```cpp
// === REGSTACK PARITY TESTS (2026-05-22 spec) ===

// Verify that at zlossCoef == 0.0f, the readout CE forward/backward path
// produces bit-identical loss and gradients vs the same path before the
// Z-loss code was added. This is a strict-bar smoke test using a tiny
// transformer config; the real production parity test is a 100-step
// trainer smoke (run after implementation completes).
void CHIRONZlossDisabledParityTest()
{
	// Reference values: handcomputed for a 4-class softmax with target=0 and
	// logits = [1.0, 0.5, -0.5, 0.0]. No randomness — these are exact.
	const float logits[4] = {1.0f, 0.5f, -0.5f, 0.0f};
	const int target = 0;

	// Compute reference CE = -log(softmax[target]).
	float lse = 0.0f;
	{
		float maxLogit = logits[0];
		for (int i = 1; i < 4; ++i) if (logits[i] > maxLogit) maxLogit = logits[i];
		float sumExp = 0.0f;
		for (int i = 0; i < 4; ++i) sumExp += expf(logits[i] - maxLogit);
		lse = maxLogit + logf(sumExp);
	}
	const float refCE = lse - logits[target];

	// Call the new helper that computes (CE, zloss) given coef. At coef=0,
	// zloss must be 0 and CE must equal refCE EXACTLY (bit-identical).
	float ce = 0.0f;
	float zloss = 0.0f;
	glades::transformer_kernels::softmax_ce_with_zloss(
		logits, 4, target, /*zlossCoef=*/0.0f, &ce, &zloss);

	ASSERT("zloss_disabled: CE must match reference exactly at coef=0",
	       ce == refCE);
	ASSERT("zloss_disabled: zloss term must be exactly 0 at coef=0",
	       zloss == 0.0f);

	// Also verify the gradient. Reference grad: (softmax - one_hot) / 1.
	float probs[4];
	{
		float maxLogit = logits[0];
		for (int i = 1; i < 4; ++i) if (logits[i] > maxLogit) maxLogit = logits[i];
		float sumExp = 0.0f;
		for (int i = 0; i < 4; ++i) sumExp += expf(logits[i] - maxLogit);
		for (int i = 0; i < 4; ++i) probs[i] = expf(logits[i] - maxLogit) / sumExp;
	}
	float refGrad[4];
	for (int i = 0; i < 4; ++i) refGrad[i] = probs[i] - (i == target ? 1.0f : 0.0f);

	float grad[4];
	glades::transformer_kernels::softmax_ce_with_zloss_grad(
		probs, 4, target, lse, /*zlossCoef=*/0.0f, grad);

	for (int i = 0; i < 4; ++i)
	{
		ASSERT("zloss_disabled: gradient must match reference exactly at coef=0",
		       grad[i] == refGrad[i]);
	}
}
```

The test references two new helpers (`softmax_ce_with_zloss` and `softmax_ce_with_zloss_grad`) that don't exist yet. That's intentional — the test should fail to compile until Task 1.2.

- [ ] **Step 5: Try to build, expect compile failure**

```bash
cd ~/dev/glades-ml/unit-tests/build && sh .configure.sh cuda 2>&1 | tail -20
```

Expected: link or compile error referencing `softmax_ce_with_zloss` or `softmax_ce_with_zloss_grad`. This is the "failing test" state.

### Task 1.2: Implement Z-loss CPU forward + backward helpers

**Files:**
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/transformer_kernels.h`
- Create or modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/transformer_kernels.cpp` (if .cpp exists; otherwise add inline to .h)

- [ ] **Step 1: Locate where transformer_kernels lives and check for existing softmax helpers**

```bash
grep -n "softmax_stable_into\|softmax_into" ~/dev/glades-ml/Backend/Machine\ Learning/Networks/transformer_kernels.h
grep -rn "softmax_stable_into" ~/dev/glades-ml/Backend/Machine\ Learning/Networks/ | head -5
```

Note which file declares vs defines `softmax_stable_into`. The new Z-loss helpers go in the same file alongside it.

- [ ] **Step 2: Add Z-loss helper declarations to transformer_kernels.h**

In `Backend/Machine Learning/Networks/transformer_kernels.h`, in the `glades::transformer_kernels` namespace (near `softmax_stable_into`), add:

```cpp
	// Fused softmax-CE forward with Z-loss auxiliary term.
	// Inputs:
	//   logits[cols]   — pre-softmax FP32 logits for a single position.
	//   cols           — vocab size.
	//   target         — ground-truth token id (in [0, cols)).
	//   zlossCoef      — auxiliary loss coefficient λ_z. At 0.0f, *zlossOut
	//                    is exactly 0 and *ceOut is bit-identical to the
	//                    standalone softmax-CE.
	// Outputs:
	//   *ceOut         — -log(softmax(logits)[target]).
	//   *zlossOut      — zlossCoef * (logsumexp(logits))^2.
	// Note: total loss is ceOut + zlossOut. Caller sums across positions.
	void softmax_ce_with_zloss(const float* logits, int cols, int target,
	                           float zlossCoef, float* ceOut, float* zlossOut);

	// Fused softmax-CE backward with Z-loss gradient contribution.
	// Inputs:
	//   probs[cols]    — softmax(logits) (computed earlier).
	//   cols, target   — as above.
	//   lse            — logsumexp(logits) (precomputed; reuses the value
	//                    from softmax_ce_with_zloss for free).
	//   zlossCoef      — auxiliary loss coefficient λ_z.
	// Output:
	//   gradOut[cols]  — d(L_main + L_zloss)/d(logit_i)
	//                  = (probs[i] - (i==target)) + 2*λ_z*lse*probs[i].
	// At zlossCoef = 0.0f, gradOut is bit-identical to the standalone
	// softmax-CE gradient.
	void softmax_ce_with_zloss_grad(const float* probs, int cols, int target,
	                                float lse, float zlossCoef,
	                                float* gradOut);
```

- [ ] **Step 3: Implement the helpers**

Find where `softmax_stable_into` is *defined* (likely `transformer_kernels.cpp` or `transformer_ops.h`). Append the two implementations alongside it:

```cpp
void softmax_ce_with_zloss(const float* logits, int cols, int target,
                           float zlossCoef, float* ceOut, float* zlossOut)
{
	// Compute logsumexp in a numerically stable way.
	float maxLogit = logits[0];
	for (int i = 1; i < cols; ++i) if (logits[i] > maxLogit) maxLogit = logits[i];
	float sumExp = 0.0f;
	for (int i = 0; i < cols; ++i) sumExp += expf(logits[i] - maxLogit);
	const float lse = maxLogit + logf(sumExp);

	// Main CE: -log(softmax(logits)[target]) = lse - logits[target].
	*ceOut = lse - logits[target];

	// Z-loss: λ_z * log²(Z) = λ_z * lse².
	// At λ_z == 0 this returns exactly 0.0f (no FP rounding from multiply
	// since the result IS the constant 0).
	if (zlossCoef == 0.0f)
		*zlossOut = 0.0f;
	else
		*zlossOut = zlossCoef * (lse * lse);
}

void softmax_ce_with_zloss_grad(const float* probs, int cols, int target,
                                float lse, float zlossCoef,
                                float* gradOut)
{
	// Main CE grad: probs - one_hot(target).
	// Z-loss grad: 2 * λ_z * lse * probs.
	// At λ_z == 0 the Z-loss contribution is skipped entirely → bit-identical
	// to the standalone CE grad.
	const float zlossScale = (zlossCoef == 0.0f) ? 0.0f
	                                              : (2.0f * zlossCoef * lse);
	for (int i = 0; i < cols; ++i)
	{
		float g = probs[i] - (i == target ? 1.0f : 0.0f);
		if (zlossScale != 0.0f) g += zlossScale * probs[i];
		gradOut[i] = g;
	}
}
```

- [ ] **Step 4: Rebuild library and tests, expect success**

```bash
cd ~/dev/glades-ml
sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests/build && sh .configure.sh cuda 2>&1 | tail -5
```

Expected: clean build.

- [ ] **Step 5: Run the disabled-parity test, expect PASS**

```bash
cd ~/dev/glades-ml/unit-tests && bash test.sh chiron 2>&1 | grep -E "zloss_disabled|FAILED|ASSERT|PASS" | head -20
```

Expected: `CHIRONZlossDisabledParityTest` passes (all 6 ASSERTs hold).

- [ ] **Step 6: Commit**

```bash
cd ~/dev/glades-ml
git add Backend/Machine\ Learning/Networks/transformer_kernels.h \
        Backend/Machine\ Learning/Networks/transformer_kernels.cpp \
        unit-tests/Backend/Machine\ Learning/chiron-test.cpp \
        unit-tests/Backend/Machine\ Learning/chiron-test.h \
        unit-tests/main.cpp
git commit -m "$(cat <<'EOF'
Add softmax_ce_with_zloss + softmax_ce_with_zloss_grad helpers (CPU)

At zlossCoef == 0.0f the helpers are bit-identical to the standalone
softmax-CE forward/backward. Strict-bar parity test pinned in
CHIRONZlossDisabledParityTest. Z-loss is not yet wired into the trainer
hot path; that lands in the next commit.
EOF
)"
```

### Task 1.3: Wire Z-loss into the readout forward (CPU path)

**Files:**
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/sgd_transformer.cpp`

- [ ] **Step 1: Locate the readout softmax + loss accumulation in transformerCpuForwardPass**

```bash
grep -n "softmax_stable_into\|tied_embedding_logits_forward_rows" \
    ~/dev/glades-ml/Backend/Machine\ Learning/Networks/sgd_transformer.cpp | head -10
```

Find the block (around line 7990-8020) where logits are converted to probs via `softmax_stable_into` and the per-position CE is summed into the training loss.

- [ ] **Step 2: Read the surrounding 60-line context to understand the loss accumulation**

```bash
sed -n '7980,8040p' ~/dev/glades-ml/Backend/Machine\ Learning/Networks/sgd_transformer.cpp
```

You're looking for: (a) where `lse` / `logsumexp` is computed (or computable as a side effect of softmax), and (b) where the per-position CE is added to the run total.

- [ ] **Step 3: Refactor the per-position softmax+CE call to use `softmax_ce_with_zloss`**

The exact diff depends on the existing structure you see in Step 2. The pattern: instead of computing softmax + CE in two passes, call the fused helper once per position and accumulate both the CE and the Z-loss into the running loss.

Example (adjust to match the actual surrounding code structure):

```cpp
// BEFORE (illustrative — match the actual code you see):
glades::transformer_kernels::softmax_stable_into(rowLogits, vocabSize, rowProbs);
float ceT = -logf(rowProbs[targetIds[t]] + 1e-30f);
runLoss += ceT;

// AFTER:
float ceT = 0.0f;
float zlossT = 0.0f;
glades::transformer_kernels::softmax_ce_with_zloss(
    rowLogits, static_cast<int>(vocabSize), targetIds[t],
    transformerRunConfig.zlossCoef, &ceT, &zlossT);
// Recompute probs explicitly for the backward (softmax_ce_with_zloss
// only returns the loss). Use the existing softmax_stable_into for probs.
glades::transformer_kernels::softmax_stable_into(rowLogits, vocabSize, rowProbs);
runLoss += ceT + zlossT;
// Save lse for backward — store it in a per-position scratch buffer.
transformerScratch.logZ[t] = ceT + rowLogits[targetIds[t]];  // lse = ce + logit[target]
```

The `logZ[t]` storage is needed by the backward to scale the Z-loss gradient. Add a `std::vector<float> logZ` member to `TransformerScratch` (look in the corresponding scratch header — usually `transformer_state.h` or similar). Allocate `logZ.resize(T)` at the same site where other per-position scratch buffers are sized.

- [ ] **Step 4: Build library + tests; run chiron suite; expect parity test still PASS**

```bash
cd ~/dev/glades-ml
sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests/build && sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests && bash test.sh chiron 2>&1 | grep -E "zloss_disabled|FAILED" | head -10
```

Expected: `CHIRONZlossDisabledParityTest` still passes (because the trainer hot path now uses the new helper but with `zlossCoef=0.0f` by default — bit-identical).

- [ ] **Step 5: Commit**

```bash
cd ~/dev/glades-ml
git add Backend/Machine\ Learning/Networks/sgd_transformer.cpp \
        Backend/Machine\ Learning/Networks/transformer_state.h
git commit -m "$(cat <<'EOF'
Wire Z-loss into CPU readout forward via softmax_ce_with_zloss helper

logsumexp is cached per position into transformerScratch.logZ for the
backward pass. At zlossCoef==0 the math is bit-identical to baseline
(verified by CHIRONZlossDisabledParityTest).
EOF
)"
```

### Task 1.4: Wire Z-loss into the GPU CE backward kernel

**Files:**
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/cuda/gpu_kernels.cu`
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/cuda/gpu_kernels.h`
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/sgd_transformer.cpp` (GPU dispatch site, ~line 10953)

- [ ] **Step 1: Read the existing softmax_cross_entropy_backward kernel**

```bash
sed -n '650,690p' ~/dev/glades-ml/Backend/Machine\ Learning/Networks/cuda/gpu_kernels.cu
```

Confirm the signature: `(const float* probs, const int* targets, int cols, float* dlogits)`. The host-side wrapper is `softmax_cross_entropy_bwd` (rows, cols).

- [ ] **Step 2: Add a Z-loss-aware variant of the kernel**

In `gpu_kernels.cu`, immediately after `softmax_cross_entropy_backward` and its `softmax_cross_entropy_bwd` wrapper, add:

```cpp
// Z-loss-aware variant of softmax_cross_entropy_backward.
// Adds (2 * zlossCoef * logZ[row]) * probs[row, i] to the standard
// softmax-CE gradient per element. At zlossCoef == 0.0f the kernel
// produces bit-identical output to softmax_cross_entropy_backward
// (the multiply is short-circuited).
__global__ void softmax_cross_entropy_backward_zloss(const float* __restrict__ probs,
                                                     const int* __restrict__ targets,
                                                     const float* __restrict__ logZ,
                                                     float zlossCoef,
                                                     int cols,
                                                     float* __restrict__ dlogits)
{
	int row = blockIdx.x;
	int target = targets[row];
	const float* pRow = probs   + (size_t)row * cols;
	float* dRow       = dlogits + (size_t)row * cols;
	const float zScale = (zlossCoef == 0.0f)
	                       ? 0.0f
	                       : (2.0f * zlossCoef * logZ[row]);

	for (int i = threadIdx.x; i < cols; i += blockDim.x)
	{
		float p = pRow[i];
		float g = (i == target) ? (p - 1.0f) : p;
		if (zScale != 0.0f) g += zScale * p;
		dRow[i] = g;
	}
}

bool softmax_cross_entropy_bwd_zloss(const float* probs, const int* targets,
                                     const float* logZ, float zlossCoef,
                                     int rows, int cols, float* dlogits)
{
	if (rows <= 0 || cols <= 0) return true;
	int block = rowBlockSize(cols);
	softmax_cross_entropy_backward_zloss<<<rows, block, 0, computeStream()>>>(
	    probs, targets, logZ, zlossCoef, cols, dlogits);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}
```

- [ ] **Step 3: Declare the wrapper in gpu_kernels.h**

In `Backend/Machine Learning/Networks/cuda/gpu_kernels.h`, alongside the existing `softmax_cross_entropy_bwd` declaration, add:

```cpp
bool softmax_cross_entropy_bwd_zloss(const float* probs, const int* targets,
                                     const float* logZ, float zlossCoef,
                                     int rows, int cols, float* dlogits);
```

- [ ] **Step 4: Add a logZ GPU buffer + upload after forward**

In `gpu_transformer_state.cu` (or whichever header defines `gpuTransformerScratch`), add a `gpu_buffer<float> logZ;` member alongside `probs`/`dLogits`. Allocate to size T at the same site the other per-position GPU scratch buffers are allocated.

After the CPU forward computes `transformerScratch.logZ[t]` (Task 1.3), copy it to GPU before the GPU backward:

```cpp
// Just before the existing gpu::softmax_cross_entropy_bwd call:
GLADES_CUDA_CHECK(cudaMemcpyAsync(
    gpuTransformerScratch->logZ.data(),
    &transformerScratch.logZ[0],
    static_cast<size_t>(T) * sizeof(float),
    cudaMemcpyHostToDevice,
    computeStream()));
```

(If the forward pass already runs on GPU and computes probs there, you can compute `logZ[t]` on GPU instead — see Task 1.5 for the GPU-resident version. For now, the CPU-fallback host→device copy is fine because forward CE summation still passes through the CPU path.)

- [ ] **Step 5: Swap the bwd dispatch to the Z-loss variant**

In `sgd_transformer.cpp` around the line that calls `gpu::softmax_cross_entropy_bwd(...)` (~line 10953), replace it with:

```cpp
glades::gpu::softmax_cross_entropy_bwd_zloss(
    gpuTransformerScratch->probs.data(),
    gpuTransformerScratch->gpuTargetsT.data(),
    gpuTransformerScratch->logZ.data(),
    transformerRunConfig.zlossCoef,
    static_cast<int>(T), static_cast<int>(vocabSize),
    gpuTransformerScratch->dLogits.data());
```

- [ ] **Step 6: Build, run chiron test, expect PASS**

```bash
cd ~/dev/glades-ml
sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests/build && sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests && bash test.sh chiron 2>&1 | grep -E "zloss|FAILED" | head -10
```

Expected: `CHIRONZlossDisabledParityTest` still passes. Because we only changed the GPU kernel and at `zlossCoef == 0.0f` the new kernel is mathematically identical to the old one.

- [ ] **Step 7: Integration smoke — 100-step trainer parity vs baseline**

```bash
cd ~/dev/glades-trainer
./build/glades_chiron_train flagship --fp8-readout-fwd --steps 100 \
    --seed 1337 --val-every 10 --zloss-coef 0.0 \
    --save-prefix /tmp/zloss_off_smoke 2>&1 | tee /tmp/zloss_off_smoke.log
```

(If `--zloss-coef` flag isn't yet wired in the trainer — Task 4.1 hasn't run yet — skip the `--zloss-coef 0.0` arg; it defaults to 0.)

Check the loss trajectory:

```bash
grep -E "step\s+(0|10|50|100)\s" /tmp/zloss_off_smoke.log
```

Expected: loss values bit-identical to a baseline run pre-Z-loss-wiring (compare to a baseline `/tmp/baseline_smoke.log` from Task 0.1 if you saved one). Acceptable variance: 0.0 at single seed (we're running with zlossCoef=0 → math is identical).

- [ ] **Step 8: Commit**

```bash
cd ~/dev/glades-ml
git add Backend/Machine\ Learning/Networks/cuda/gpu_kernels.cu \
        Backend/Machine\ Learning/Networks/cuda/gpu_kernels.h \
        Backend/Machine\ Learning/Networks/cuda/gpu_transformer_state.cu \
        Backend/Machine\ Learning/Networks/sgd_transformer.cpp
git commit -m "$(cat <<'EOF'
Wire Z-loss into GPU readout backward via softmax_ce_bwd_zloss

New kernel softmax_cross_entropy_backward_zloss adds the Z-loss
gradient contribution (2·λ_z·lse·softmax) on top of the standard
CE gradient. At λ_z == 0 the kernel short-circuits the multiply and
output is bit-identical to softmax_cross_entropy_backward.
logZ is computed CPU-side in Task 1.3 and uploaded H2D per step.

100-step smoke at λ_z=0 produces bit-identical loss to baseline.
EOF
)"
```

### Task 1.5: Z-loss enabled math test

**Files:**
- Modify: `~/dev/glades-ml/unit-tests/Backend/Machine Learning/chiron-test.cpp`

- [ ] **Step 1: Add the enabled-math test**

Append to `chiron-test.cpp` after `CHIRONZlossDisabledParityTest`:

```cpp
// Verify that at zlossCoef > 0, the loss and gradient include the
// expected Z-loss contribution.
void CHIRONZlossEnabledMathTest()
{
	const float logits[4] = {2.0f, 0.0f, -1.0f, 0.5f};
	const int target = 2;
	const float lambdaZ = 0.1f;  // larger than production to make signal obvious

	// Reference: compute by hand.
	float maxLogit = logits[0];
	for (int i = 1; i < 4; ++i) if (logits[i] > maxLogit) maxLogit = logits[i];
	float sumExp = 0.0f;
	for (int i = 0; i < 4; ++i) sumExp += expf(logits[i] - maxLogit);
	const float lse = maxLogit + logf(sumExp);
	const float refCE = lse - logits[target];
	const float refZloss = lambdaZ * lse * lse;

	float probs[4];
	for (int i = 0; i < 4; ++i) probs[i] = expf(logits[i] - maxLogit) / sumExp;

	float refGrad[4];
	const float zScale = 2.0f * lambdaZ * lse;
	for (int i = 0; i < 4; ++i)
	{
		refGrad[i] = probs[i] - (i == target ? 1.0f : 0.0f);
		refGrad[i] += zScale * probs[i];
	}

	// Call our helpers.
	float ce = 0.0f, zloss = 0.0f;
	glades::transformer_kernels::softmax_ce_with_zloss(
		logits, 4, target, lambdaZ, &ce, &zloss);
	float grad[4];
	glades::transformer_kernels::softmax_ce_with_zloss_grad(
		probs, 4, target, lse, lambdaZ, grad);

	// Math tolerance: 1e-6 relative (FP32 round-off).
	const float ceErr = fabsf(ce - refCE);
	const float zErr = fabsf(zloss - refZloss);
	ASSERT("zloss_enabled: CE matches reference", ceErr < 1e-5f);
	ASSERT("zloss_enabled: zloss term matches reference", zErr < 1e-5f);
	for (int i = 0; i < 4; ++i)
	{
		const float gErr = fabsf(grad[i] - refGrad[i]);
		ASSERT("zloss_enabled: gradient matches reference", gErr < 1e-5f);
	}
}
```

- [ ] **Step 2: Build and run, expect PASS**

```bash
cd ~/dev/glades-ml/unit-tests/build && sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests && bash test.sh chiron 2>&1 | grep -E "zloss_enabled|FAILED" | head -10
```

Expected: all 4 ASSERTs pass.

- [ ] **Step 3: Commit**

```bash
cd ~/dev/glades-ml
git add unit-tests/Backend/Machine\ Learning/chiron-test.cpp
git commit -m "Add CHIRONZlossEnabledMathTest verifying λ_z=0.1 reference math"
```

---

## Phase 2: QK-Norm

### Task 2.1: Add per-head γ to Block + Adam state

**Files:**
- Modify: the file that declares `TensorTransformerState::Block` (likely `~/dev/glades-ml/Backend/Machine Learning/Networks/transformer_state.h` or similar)
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/sgd_transformer.cpp`

- [ ] **Step 1: Locate the Block struct**

```bash
grep -rn "ln1Gamma" ~/dev/glades-ml/Backend/Machine\ Learning/Networks/*.h | head -5
```

Find the struct definition (look for `std::vector<float> ln1Gamma;` and the adjacent `gLn1Gamma`, `mLn1Gamma`, `v2Ln1Gamma`).

- [ ] **Step 2: Add per-head γ + Adam state to Block**

In the same struct, immediately after the LayerNorm gamma fields (or another logical place — match the codebase's existing grouping), insert:

```cpp
	// QK-Norm: per-head learnable scalar replacing 1/sqrt(dHead).
	// Size = nHeads when qkNormEnabled; otherwise empty.
	std::vector<float> qknormGamma;
	std::vector<float> gQknormGamma;
	std::vector<float> mQknormGamma;
	std::vector<float> v2QknormGamma;
```

- [ ] **Step 3: Initialize γ at allocation site**

Find the existing block-init site (search for `ln1Gamma.assign(`). At the same site, after the LayerNorm gamma init, add:

```cpp
// QK-Norm γ init: log2(T) per DeepSeek-V3, or qkNormGammaInit override.
if (transformerRunConfig.qkNormEnabled)
{
	const float gammaInit = (transformerRunConfig.qkNormGammaInit > 0.0f)
	                          ? transformerRunConfig.qkNormGammaInit
	                          : log2f(static_cast<float>(T));
	b.qknormGamma.assign(nHeads, gammaInit);
	b.gQknormGamma.assign(nHeads, 0.0f);
	b.mQknormGamma.assign(nHeads, 0.0f);
	b.v2QknormGamma.assign(nHeads, 0.0f);
}
// When qkNormEnabled is false, vectors stay empty — sentinel for "not in use".
```

- [ ] **Step 4: Add to gradient zeroing**

Find the existing `std::fill(b.gLn1Gamma.begin(), b.gLn1Gamma.end(), 0.0f);` line (around line 1163). Right after it, add:

```cpp
if (!b.gQknormGamma.empty())
	std::fill(b.gQknormGamma.begin(), b.gQknormGamma.end(), 0.0f);
```

- [ ] **Step 5: Add to all Adam update sites**

There are ~10 sites where `Adam::update_param(b.ln1Gamma, ...)` is called (lines 5153, 5208, 5258, 5295, 5332, 5369, 5406, 5455, 5492, 5511, 6331). At each of those sites, after the `ln1Gamma` update, add a guarded update for `qknormGamma`:

```cpp
if (!b.qknormGamma.empty())
{
	Adam::update_param(b.qknormGamma, b.mQknormGamma, b.v2QknormGamma,
	                   b.gQknormGamma, lr, beta1, beta2, inv1mB1t, inv1mB2t,
	                   eps, invBatch, gradScale);
}
```

(Use the same `lr` / `beta1` / `beta2` / etc. that the surrounding ln1Gamma update uses; copy the call args from the matching ln1Gamma call.)

For the atlas update sites (lines 5550, 5980, 6141), similarly add:

```cpp
if (!b.qknormGamma.empty())
{
	if (!atlas::updateBias(&b.qknormGamma[0], &b.gQknormGamma[0],
	                       static_cast<unsigned int>(b.qknormGamma.size()),
	                       invBatch, lr, gradScale))
	{
		// fallback or error path — match the surrounding error handling
	}
}
```

- [ ] **Step 6: Build, run chiron suite, expect baseline tests still PASS**

```bash
cd ~/dev/glades-ml
sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests/build && sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests && bash test.sh chiron 2>&1 | tail -20
```

Expected: all existing tests still pass. The new vectors are empty when `qkNormEnabled=false`, so no behavior change.

- [ ] **Step 7: Commit**

```bash
cd ~/dev/glades-ml
git add Backend/Machine\ Learning/Networks/transformer_state.h \
        Backend/Machine\ Learning/Networks/sgd_transformer.cpp
git commit -m "$(cat <<'EOF'
Add per-head QK-Norm γ parameter + Adam state to TensorTransformerState::Block

When qkNormEnabled=false the vectors stay empty and all gradient zero /
optimizer update sites short-circuit on .empty(). When enabled, γ is
initialized to log2(T) per DeepSeek-V3 and updated via the same
optimizer path as the existing LayerNorm gammas.
EOF
)"
```

### Task 2.2: QK-Norm forward (CPU reference)

**Files:**
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/transformer_kernels.h`
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/transformer_kernels.cpp` (or wherever the existing CPU forward kernels live)

- [ ] **Step 1: Add helper declarations**

In `transformer_kernels.h`:

```cpp
	// QK-Norm forward: L2-normalize each row (per-head, per-token) in place.
	// Input/output: x has shape (T, nHeads * dHead). Normalization is over
	// dHead within each head, per token. Adds eps inside the norm for
	// numerical safety.
	void qknorm_forward(float* x, int T, int nHeads, int dHead, float eps);

	// QK-Norm backward: given x_orig (pre-norm), normInv = 1/||x|| per row,
	// and dxNorm (gradient at normalized output), compute gradient at x_orig:
	//   dx = normInv * (dxNorm - (x_norm · dxNorm) * x_norm)
	// Operates per-token, per-head.
	void qknorm_backward(const float* xNorm, const float* normInv,
	                     const float* dxNorm, int T, int nHeads, int dHead,
	                     float* dxOrig);
```

- [ ] **Step 2: Implement the helpers**

```cpp
void qknorm_forward(float* x, int T, int nHeads, int dHead, float eps)
{
	for (int t = 0; t < T; ++t)
	{
		for (int h = 0; h < nHeads; ++h)
		{
			float* row = x + (size_t)t * nHeads * dHead + (size_t)h * dHead;
			float ss = 0.0f;
			for (int i = 0; i < dHead; ++i) ss += row[i] * row[i];
			const float invNorm = 1.0f / sqrtf(ss + eps);
			for (int i = 0; i < dHead; ++i) row[i] *= invNorm;
		}
	}
}

void qknorm_backward(const float* xNorm, const float* normInv,
                     const float* dxNorm, int T, int nHeads, int dHead,
                     float* dxOrig)
{
	for (int t = 0; t < T; ++t)
	{
		for (int h = 0; h < nHeads; ++h)
		{
			const size_t off = (size_t)t * nHeads * dHead + (size_t)h * dHead;
			const float* xn = xNorm + off;
			const float* dn = dxNorm + off;
			const float ni = normInv[(size_t)t * nHeads + h];

			// (x_norm · dxNorm)
			float xnDotDn = 0.0f;
			for (int i = 0; i < dHead; ++i) xnDotDn += xn[i] * dn[i];

			float* dox = dxOrig + off;
			for (int i = 0; i < dHead; ++i)
				dox[i] = ni * (dn[i] - xnDotDn * xn[i]);
		}
	}
}
```

- [ ] **Step 3: Write the enabled-math unit test**

Append to `chiron-test.cpp`:

```cpp
void CHIRONQkNormEnabledMathTest()
{
	// Tiny shape: T=2, nHeads=2, dHead=4.
	const int T = 2;
	const int nHeads = 2;
	const int dHead = 4;
	std::vector<float> x(T * nHeads * dHead);
	for (int i = 0; i < (int)x.size(); ++i) x[i] = (float)(i + 1);

	// Reference: per (t, h) row, normalize by L2 norm.
	std::vector<float> ref(x);
	for (int t = 0; t < T; ++t)
	{
		for (int h = 0; h < nHeads; ++h)
		{
			float* row = &ref[(t * nHeads + h) * dHead];
			float ss = 0.0f;
			for (int i = 0; i < dHead; ++i) ss += row[i] * row[i];
			const float invN = 1.0f / sqrtf(ss + 1e-6f);
			for (int i = 0; i < dHead; ++i) row[i] *= invN;
		}
	}

	std::vector<float> got(x);
	glades::transformer_kernels::qknorm_forward(&got[0], T, nHeads, dHead, 1e-6f);

	float worst = max_abs_diff(got, ref);
	ASSERT("qknorm_enabled: forward matches reference within 1e-6", worst < 1e-6f);
}
```

(Also declare `CHIRONQkNormEnabledMathTest` in `main.cpp` invocation list if not already.)

- [ ] **Step 4: Build, run, expect PASS**

```bash
cd ~/dev/glades-ml && sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests/build && sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests && bash test.sh chiron 2>&1 | grep -E "qknorm_enabled|FAILED" | head -10
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd ~/dev/glades-ml
git add Backend/Machine\ Learning/Networks/transformer_kernels.h \
        Backend/Machine\ Learning/Networks/transformer_kernels.cpp \
        unit-tests/Backend/Machine\ Learning/chiron-test.cpp
git commit -m "Add qknorm_forward + qknorm_backward CPU helpers with math test"
```

### Task 2.3: GPU QK-Norm forward + backward kernels

**Files:**
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/cuda/gpu_kernels.cu`
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/cuda/gpu_kernels.h`

- [ ] **Step 1: Add GPU forward kernel**

In `gpu_kernels.cu`, after an existing per-row L2-norm style kernel (search for `rsqrtf` to find similar reduction patterns — line 162 has the LayerNorm pattern), add:

```cpp
// QK-Norm forward: per-token, per-head L2 normalize.
// x shape: (T, nHeads, dHead) row-major. eps for numerical stability.
// Writes back into x in place, and writes invNorm[t,h] = 1/||x_orig|| out for backward.
__global__ void qknorm_forward_kernel(float* __restrict__ x,
                                      float* __restrict__ invNorm,
                                      int nHeads, int dHead, float eps)
{
	const int t = blockIdx.x;
	const int h = blockIdx.y;
	const int tid = threadIdx.x;
	const int block = blockDim.x;
	float* row = x + ((size_t)t * nHeads + h) * dHead;

	// Reduction: sum of squares.
	float local = 0.0f;
	for (int i = tid; i < dHead; i += block) local += row[i] * row[i];

	// Warp-shuffle reduce (deterministic within warp).
	for (int offset = warpSize / 2; offset > 0; offset >>= 1)
		local += __shfl_down_sync(0xffffffff, local, offset);

	// Block reduce via shared memory (only first lane of each warp contributes).
	__shared__ float warpSums[32];  // up to 32 warps per block
	int laneId = tid & (warpSize - 1);
	int warpId = tid / warpSize;
	if (laneId == 0) warpSums[warpId] = local;
	__syncthreads();

	if (warpId == 0)
	{
		const int numWarps = (block + warpSize - 1) / warpSize;
		float s = (laneId < numWarps) ? warpSums[laneId] : 0.0f;
		for (int offset = warpSize / 2; offset > 0; offset >>= 1)
			s += __shfl_down_sync(0xffffffff, s, offset);
		if (laneId == 0)
			warpSums[0] = s;
	}
	__syncthreads();

	const float ss = warpSums[0];
	const float invN = rsqrtf(ss + eps);

	if (tid == 0) invNorm[(size_t)t * nHeads + h] = invN;

	for (int i = tid; i < dHead; i += block) row[i] *= invN;
}

bool qknorm_forward_gpu(float* x, float* invNorm, int T, int nHeads, int dHead, float eps)
{
	if (T <= 0 || nHeads <= 0 || dHead <= 0) return true;
	const int block = (dHead < 256) ? dHead : 256;
	dim3 grid(T, nHeads);
	qknorm_forward_kernel<<<grid, block, 0, computeStream()>>>(
	    x, invNorm, nHeads, dHead, eps);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}
```

- [ ] **Step 2: Add GPU backward kernel**

Immediately after:

```cpp
// QK-Norm backward.
// xNorm, dxNorm, dxOrig: (T, nHeads, dHead).
// invNorm: (T, nHeads).
// dx = invN * (dxNorm - (xNorm · dxNorm) * xNorm)
__global__ void qknorm_backward_kernel(const float* __restrict__ xNorm,
                                       const float* __restrict__ invNorm,
                                       const float* __restrict__ dxNorm,
                                       int nHeads, int dHead,
                                       float* __restrict__ dxOrig)
{
	const int t = blockIdx.x;
	const int h = blockIdx.y;
	const int tid = threadIdx.x;
	const int block = blockDim.x;
	const size_t off = ((size_t)t * nHeads + h) * dHead;
	const float* xn = xNorm + off;
	const float* dn = dxNorm + off;
	float* dox = dxOrig + off;

	// Reduction: xn · dn.
	float local = 0.0f;
	for (int i = tid; i < dHead; i += block) local += xn[i] * dn[i];
	for (int offset = warpSize / 2; offset > 0; offset >>= 1)
		local += __shfl_down_sync(0xffffffff, local, offset);

	__shared__ float warpSums[32];
	int laneId = tid & (warpSize - 1);
	int warpId = tid / warpSize;
	if (laneId == 0) warpSums[warpId] = local;
	__syncthreads();

	if (warpId == 0)
	{
		const int numWarps = (block + warpSize - 1) / warpSize;
		float s = (laneId < numWarps) ? warpSums[laneId] : 0.0f;
		for (int offset = warpSize / 2; offset > 0; offset >>= 1)
			s += __shfl_down_sync(0xffffffff, s, offset);
		if (laneId == 0) warpSums[0] = s;
	}
	__syncthreads();

	const float xnDotDn = warpSums[0];
	const float ni = invNorm[(size_t)t * nHeads + h];

	for (int i = tid; i < dHead; i += block)
		dox[i] = ni * (dn[i] - xnDotDn * xn[i]);
}

bool qknorm_backward_gpu(const float* xNorm, const float* invNorm,
                         const float* dxNorm, int T, int nHeads, int dHead,
                         float* dxOrig)
{
	if (T <= 0 || nHeads <= 0 || dHead <= 0) return true;
	const int block = (dHead < 256) ? dHead : 256;
	dim3 grid(T, nHeads);
	qknorm_backward_kernel<<<grid, block, 0, computeStream()>>>(
	    xNorm, invNorm, dxNorm, nHeads, dHead, dxOrig);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}
```

- [ ] **Step 3: Add declarations to gpu_kernels.h**

```cpp
bool qknorm_forward_gpu(float* x, float* invNorm,
                        int T, int nHeads, int dHead, float eps);

bool qknorm_backward_gpu(const float* xNorm, const float* invNorm,
                         const float* dxNorm, int T, int nHeads, int dHead,
                         float* dxOrig);
```

- [ ] **Step 4: Build, expect success (no test runs needed yet)**

```bash
cd ~/dev/glades-ml && sh .configure.sh cuda 2>&1 | tail -5
```

Expected: clean build. Kernels are defined but not yet called.

- [ ] **Step 5: Commit**

```bash
cd ~/dev/glades-ml
git add Backend/Machine\ Learning/Networks/cuda/gpu_kernels.cu \
        Backend/Machine\ Learning/Networks/cuda/gpu_kernels.h
git commit -m "Add qknorm_forward_gpu + qknorm_backward_gpu CUDA kernels"
```

### Task 2.4: Wire QK-Norm into attention forward (CPU + GPU dispatch)

**Files:**
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/sgd_transformer.cpp`

- [ ] **Step 1: Locate Q/K projection sites + attention dispatch**

```bash
sed -n '7635,7740p' ~/dev/glades-ml/Backend/Machine\ Learning/Networks/sgd_transformer.cpp
```

You'll see Q, K, V projection (lines ~7641-7650) followed by the per-head attention loop (~7695-7735). The QK-Norm insertion point is **between Q/K projection and the attention call**.

- [ ] **Step 2: Insert QK-Norm forward after Q/K projection**

Right after the V projection, before the attention loop, add:

```cpp
// QK-Norm forward (when enabled): L2-normalize Q and K per head per token.
// V is not normalized.
std::vector<float> qInvNorm, kInvNorm;
if (!b.qknormGamma.empty())
{
	qInvNorm.assign(static_cast<size_t>(T) * nHeads, 0.0f);
	kInvNorm.assign(static_cast<size_t>(T) * nKVHeads, 0.0f);
	// For now, CPU reference path (GPU equivalent added in Task 2.5):
	glades::transformer_kernels::qknorm_forward(Q, T, nHeads, dHead, 1e-6f);
	glades::transformer_kernels::qknorm_forward(K, T, nKVHeads, dHead, 1e-6f);
	// invNorm vectors are kept for backward (filled at GPU time in 2.5;
	// CPU path: just recompute on demand).
}
```

(The `qInvNorm`/`kInvNorm` vectors are placeholders. The CPU path is not used in production for the flagship — GPU dispatch in Task 2.5 will handle the real path. The CPU path is for parity-test convergence.)

- [ ] **Step 3: Replace per-head `1/sqrt(dHead)` scaling with γ_h substitution**

When QK-Norm is on, the attention call expects the dot product `(Q'·K')` to be multiplied by `γ_h` instead of `1/sqrt(dHead)`. The cleanest approach: pre-multiply Q by `γ_h` (since `γ_h * Q' · K' = (γ_h * Q') · K'`), keeping the attention kernel's `1/sqrt(dHead)` literal intact for the disabled-default path.

Wait — that would double-scale. Better: when `qkNormEnabled`, leave the attention kernel's `1/sqrt(dHead)` scaling in place but pre-multiply Q' by `(γ_h * sqrt(dHead))` so the net scale becomes `γ_h`. This avoids touching the attention kernel.

```cpp
if (!b.qknormGamma.empty())
{
	const float sqrtDh = sqrtf(static_cast<float>(dHead));
	for (int t = 0; t < (int)T; ++t)
	{
		for (int h = 0; h < (int)nHeads; ++h)
		{
			float* qRow = Q + ((size_t)t * nHeads + h) * dHead;
			const float scale = b.qknormGamma[h] * sqrtDh;
			for (int i = 0; i < dHead; ++i) qRow[i] *= scale;
		}
	}
}
```

This converts the attention into: `softmax((γ_h * sqrt(dHead) * Q') · K' / sqrt(dHead)) = softmax(γ_h * (Q' · K'))`, which is the desired QK-Norm formula.

- [ ] **Step 4: Build + run chiron suite, expect baseline tests still PASS**

```bash
cd ~/dev/glades-ml && sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests/build && sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests && bash test.sh chiron 2>&1 | tail -10
```

Expected: PASS. When `qkNormEnabled=false`, `b.qknormGamma` is empty, the new block is skipped, and Q/K are unmodified — bit-identical baseline.

- [ ] **Step 5: Commit**

```bash
cd ~/dev/glades-ml
git add Backend/Machine\ Learning/Networks/sgd_transformer.cpp
git commit -m "$(cat <<'EOF'
Wire QK-Norm forward into CPU attention path (guarded by qknormGamma.empty)

When QK-Norm is on, normalize Q and K per-head then pre-multiply Q by
γ_h * sqrt(dHead) so the attention kernel's existing 1/sqrt(dHead)
scale becomes γ_h * (Q' · K'). When off, the block is skipped and
Q/K are bit-identical to baseline.

GPU dispatch + backward land in subsequent commits.
EOF
)"
```

### Task 2.5: QK-Norm GPU forward dispatch

**Files:**
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/sgd_transformer.cpp` (the GPU forward block, ~line 7700+)
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/cuda/gpu_transformer_state.cu` (add invNorm GPU scratch)

- [ ] **Step 1: Add invNorm GPU buffers to gpu_transformer_state**

In `gpu_transformer_state.cu` / its header, add:

```cpp
gpu_buffer<float> qInvNorm;  // (T, nHeads)
gpu_buffer<float> kInvNorm;  // (T, nKVHeads)
gpu_buffer<float> qPreNorm;  // (T, nHeads, dHead) — copy of Q before normalize, needed for bwd
gpu_buffer<float> kPreNorm;  // (T, nKVHeads, dHead)
```

At the existing allocate-on-first-use site (search for `probs.alloc(`), allocate `qInvNorm.alloc(T * nHeads)` etc. when `qkNormEnabled`.

- [ ] **Step 2: Locate the GPU forward attention block in sgd_transformer.cpp**

```bash
grep -n "gpu_flash_attention_fwd\|gpu_scfa_fwd\|cublasGemmEx.*Q\b" \
    ~/dev/glades-ml/Backend/Machine\ Learning/Networks/sgd_transformer.cpp | head -10
```

Find the per-layer GPU forward attention dispatch. This is structured similarly to the CPU path but with GPU buffer calls.

- [ ] **Step 3: Insert GPU QK-Norm forward dispatch**

Right after the GPU Q/K projection completes, before the GPU attention call, add:

```cpp
if (!b.qknormGamma.empty())
{
	// Save pre-norm Q and K for backward.
	GLADES_CUDA_CHECK(cudaMemcpyAsync(
	    gpuTransformerScratch->qPreNorm.data() + ((size_t)li * T * nHeads * dHead),
	    Q_gpu, (size_t)T * nHeads * dHead * sizeof(float),
	    cudaMemcpyDeviceToDevice, computeStream()));
	GLADES_CUDA_CHECK(cudaMemcpyAsync(
	    gpuTransformerScratch->kPreNorm.data() + ((size_t)li * T * nKVHeads * dHead),
	    K_gpu, (size_t)T * nKVHeads * dHead * sizeof(float),
	    cudaMemcpyDeviceToDevice, computeStream()));

	// Normalize Q and K in place, save invNorm.
	glades::gpu::qknorm_forward_gpu(Q_gpu,
	    gpuTransformerScratch->qInvNorm.data() + ((size_t)li * T * nHeads),
	    T, nHeads, dHead, 1e-6f);
	glades::gpu::qknorm_forward_gpu(K_gpu,
	    gpuTransformerScratch->kInvNorm.data() + ((size_t)li * T * nKVHeads),
	    T, nKVHeads, dHead, 1e-6f);

	// Pre-multiply Q by γ_h * sqrt(dHead).
	// Cheap launch — reuse an existing per-head scale kernel or add a tiny
	// new one. For now, use a small inline kernel:
	const float sqrtDh = sqrtf(static_cast<float>(dHead));
	std::vector<float> hostGammaScale(nHeads);
	for (int h = 0; h < nHeads; ++h)
		hostGammaScale[h] = b.qknormGamma[h] * sqrtDh;
	// Upload to a per-head scratch and apply via gpu::scale_q_per_head.
	GLADES_CUDA_CHECK(cudaMemcpyAsync(
	    gpuTransformerScratch->qknormGammaScale.data(),
	    &hostGammaScale[0], nHeads * sizeof(float),
	    cudaMemcpyHostToDevice, computeStream()));
	glades::gpu::scale_q_per_head(Q_gpu,
	    gpuTransformerScratch->qknormGammaScale.data(),
	    T, nHeads, dHead);
}
```

The helper `gpu::scale_q_per_head` is new — implement it as a small kernel in `gpu_kernels.cu`:

```cpp
__global__ void scale_q_per_head_kernel(float* __restrict__ Q,
                                        const float* __restrict__ gammaScale,
                                        int nHeads, int dHead)
{
	const int t = blockIdx.x;
	const int h = blockIdx.y;
	const int tid = threadIdx.x;
	const int block = blockDim.x;
	float* row = Q + ((size_t)t * nHeads + h) * dHead;
	const float s = gammaScale[h];
	for (int i = tid; i < dHead; i += block) row[i] *= s;
}

bool scale_q_per_head(float* Q, const float* gammaScale,
                      int T, int nHeads, int dHead)
{
	if (T <= 0 || nHeads <= 0 || dHead <= 0) return true;
	const int block = (dHead < 256) ? dHead : 256;
	dim3 grid(T, nHeads);
	scale_q_per_head_kernel<<<grid, block, 0, computeStream()>>>(
	    Q, gammaScale, nHeads, dHead);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}
```

- [ ] **Step 4: Add qknormGammaScale to gpu_transformer_state**

```cpp
gpu_buffer<float> qknormGammaScale;  // (nHeads), scratch for per-step γ upload
```

Alloc to `nHeads` when `qkNormEnabled`.

- [ ] **Step 5: Build + run chiron suite + 100-step smoke**

```bash
cd ~/dev/glades-ml && sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests/build && sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests && bash test.sh chiron 2>&1 | tail -10
cd ~/dev/glades-trainer
./build/glades_chiron_train flagship --fp8-readout-fwd --steps 100 \
    --seed 1337 --val-every 10 \
    --save-prefix /tmp/qknorm_off_smoke 2>&1 | tail -20
```

Expected: bit-identical loss trajectory to baseline (qkNormEnabled=false default).

- [ ] **Step 6: Commit**

```bash
cd ~/dev/glades-ml
git add Backend/Machine\ Learning/Networks/sgd_transformer.cpp \
        Backend/Machine\ Learning/Networks/cuda/gpu_transformer_state.cu \
        Backend/Machine\ Learning/Networks/cuda/gpu_kernels.cu \
        Backend/Machine\ Learning/Networks/cuda/gpu_kernels.h
git commit -m "$(cat <<'EOF'
Wire QK-Norm GPU forward dispatch + scale_q_per_head helper

When qknormGamma is non-empty per layer, GPU forward: save Q/K
pre-norm copies (needed by bwd), normalize Q/K in place, pre-multiply
Q by γ_h * sqrt(dHead) so attention kernel's 1/sqrt(dHead) recovers
γ_h * (Q'·K'). At default-off (qknormGamma empty), the entire block
is skipped and Q/K are bit-identical to baseline.

100-step smoke at qknorm-off produces bit-identical loss.
EOF
)"
```

### Task 2.6: QK-Norm GPU backward + γ_h gradient

**Files:**
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/sgd_transformer.cpp` (GPU backward block, ~line 11000+)
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/cuda/gpu_kernels.cu`

- [ ] **Step 1: Locate the GPU attention backward block**

```bash
grep -n "gpu_flash_attention_bwd\|flash_attention_bwd_multihead\|gpu_attention_bwd" \
    ~/dev/glades-ml/Backend/Machine\ Learning/Networks/sgd_transformer.cpp | head -10
```

Find where the attention backward writes `dQ`, `dK`, `dV` per layer.

- [ ] **Step 2: After attention bwd produces dQ/dK, reverse the QK-Norm + γ scaling**

Right after the GPU attention backward call (so we have `dQ_gpu`, `dK_gpu` at the post-normalize/post-γ-scaled gradients), add:

```cpp
if (!b.qknormGamma.empty())
{
	// Step A: dQ at this point is grad w.r.t. (γ_h * sqrt(dHead) * Q_norm).
	// First, accumulate γ_h gradient: dγ_h = sum_{t,i} (dQ_post-scale[t,h,i] * sqrt(dHead) * Q_norm[t,h,i]).
	// Use Q_post-norm (currently in Q_gpu, but has been overwritten by attention?
	// — we saved qPreNorm; need Q_norm specifically. If overwritten,
	// recompute Q_norm = qPreNorm[i] * qInvNorm[h] OR save Q_norm explicitly.
	// Simplest: save Q_norm (post-norm, pre-γ-scale) in another scratch buffer
	// in Task 2.5 (add gpu_buffer<float> qNorm and copy after norm).

	// γ_h gradient kernel:
	glades::gpu::qknorm_gamma_grad(
	    dQ_gpu,  // input: grad at γ*sqrt(dHead)*Q_norm
	    gpuTransformerScratch->qNorm.data() + ((size_t)li * T * nHeads * dHead),
	    /*sqrtDh*/ sqrtf((float)dHead),
	    T, nHeads, dHead,
	    /*dGamma*/ gpu_tied_d_qknormGamma_per_layer_buffer);

	// Step B: undo the γ * sqrt(dHead) pre-scale on dQ to get grad at Q_norm.
	// Reuse scale_q_per_head with the inverse scale.
	std::vector<float> hostGammaInv(nHeads);
	for (int h = 0; h < nHeads; ++h)
		hostGammaInv[h] = b.qknormGamma[h] * sqrtf((float)dHead);
	// Actually we want to multiply dQ by γ*sqrt(dHead) (chain rule for the
	// pre-multiply on Q forward): dQ_norm = (γ*sqrt(dHead)) * dQ_post-scale.
	// Just call scale_q_per_head(dQ, gammaScale) — same scale as forward.
	glades::gpu::scale_q_per_head(dQ_gpu,
	    gpuTransformerScratch->qknormGammaScale.data(),
	    T, nHeads, dHead);

	// Step C: undo Q normalize. dQ_orig = qknorm_backward_gpu(Q_norm, qInvNorm, dQ_norm).
	glades::gpu::qknorm_backward_gpu(
	    gpuTransformerScratch->qNorm.data() + ((size_t)li * T * nHeads * dHead),
	    gpuTransformerScratch->qInvNorm.data() + ((size_t)li * T * nHeads),
	    dQ_gpu,
	    T, nHeads, dHead,
	    dQ_gpu);  // in-place

	// Step D: same for K. K was not pre-multiplied by γ, only normalized.
	glades::gpu::qknorm_backward_gpu(
	    gpuTransformerScratch->kNorm.data() + ((size_t)li * T * nKVHeads * dHead),
	    gpuTransformerScratch->kInvNorm.data() + ((size_t)li * T * nKVHeads),
	    dK_gpu,
	    T, nKVHeads, dHead,
	    dK_gpu);  // in-place

	// Step E: accumulate γ_h gradient into b.gQknormGamma (host-side).
	std::vector<float> hostDgamma(nHeads, 0.0f);
	GLADES_CUDA_CHECK(cudaMemcpyAsync(
	    &hostDgamma[0], gpu_tied_d_qknormGamma_per_layer_buffer,
	    nHeads * sizeof(float),
	    cudaMemcpyDeviceToHost, computeStream()));
	GLADES_CUDA_CHECK(cudaStreamSynchronize(computeStream()));
	for (int h = 0; h < nHeads; ++h) b.gQknormGamma[h] += hostDgamma[h];
}
```

- [ ] **Step 3: Implement qknorm_gamma_grad GPU helper**

In `gpu_kernels.cu`:

```cpp
// γ_h gradient: dγ_h = (1 / sqrtDh) * sum_{t,i} (dQ_post-scale[t,h,i] * Q_norm[t,h,i] * sqrtDh)
//                    = sum_{t,i} dQ_post-scale[t,h,i] * Q_norm[t,h,i]
// Wait — we pre-multiplied Q by γ * sqrtDh. So forward: Q_scaled = γ * sqrtDh * Q_norm.
// dγ = d(loss)/d(γ) = sum_{t,i} (∂loss/∂Q_scaled[t,h,i]) * sqrtDh * Q_norm[t,h,i]
__global__ void qknorm_gamma_grad_kernel(const float* __restrict__ dQScaled,
                                         const float* __restrict__ qNorm,
                                         float sqrtDh,
                                         int nHeads, int dHead,
                                         int T,
                                         float* __restrict__ dGamma)
{
	const int h = blockIdx.x;
	const int tid = threadIdx.x;
	const int block = blockDim.x;
	float local = 0.0f;
	for (int t = 0; t < T; ++t)
	{
		const size_t off = ((size_t)t * nHeads + h) * dHead;
		for (int i = tid; i < dHead; i += block)
			local += dQScaled[off + i] * qNorm[off + i];
	}
	local *= sqrtDh;
	// Warp + block reduce.
	for (int offset = warpSize / 2; offset > 0; offset >>= 1)
		local += __shfl_down_sync(0xffffffff, local, offset);
	__shared__ float warpSums[32];
	int laneId = tid & (warpSize - 1);
	int warpId = tid / warpSize;
	if (laneId == 0) warpSums[warpId] = local;
	__syncthreads();
	if (warpId == 0)
	{
		const int numWarps = (block + warpSize - 1) / warpSize;
		float s = (laneId < numWarps) ? warpSums[laneId] : 0.0f;
		for (int offset = warpSize / 2; offset > 0; offset >>= 1)
			s += __shfl_down_sync(0xffffffff, s, offset);
		if (laneId == 0) dGamma[h] = s;
	}
}

bool qknorm_gamma_grad(const float* dQScaled, const float* qNorm,
                       float sqrtDh, int T, int nHeads, int dHead,
                       float* dGamma)
{
	if (T <= 0 || nHeads <= 0 || dHead <= 0) return true;
	const int block = 256;
	qknorm_gamma_grad_kernel<<<nHeads, block, 0, computeStream()>>>(
	    dQScaled, qNorm, sqrtDh, nHeads, dHead, T, dGamma);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}
```

Also declare `qknorm_gamma_grad` in `gpu_kernels.h`.

- [ ] **Step 4: Save Q_norm and K_norm in Task 2.5 (retroactive)**

Re-open the QK-Norm GPU forward block from Task 2.5. After `qknorm_forward_gpu(Q_gpu, ...)` (but BEFORE the γ pre-multiply), add:

```cpp
GLADES_CUDA_CHECK(cudaMemcpyAsync(
    gpuTransformerScratch->qNorm.data() + ((size_t)li * T * nHeads * dHead),
    Q_gpu, (size_t)T * nHeads * dHead * sizeof(float),
    cudaMemcpyDeviceToDevice, computeStream()));
```

Same for K:

```cpp
GLADES_CUDA_CHECK(cudaMemcpyAsync(
    gpuTransformerScratch->kNorm.data() + ((size_t)li * T * nKVHeads * dHead),
    K_gpu, (size_t)T * nKVHeads * dHead * sizeof(float),
    cudaMemcpyDeviceToDevice, computeStream()));
```

Add `gpu_buffer<float> qNorm, kNorm;` to gpu_transformer_state and alloc.

- [ ] **Step 5: Build, run smoke, expect bit-identical baseline**

```bash
cd ~/dev/glades-ml && sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-trainer
./build/glades_chiron_train flagship --fp8-readout-fwd --steps 100 \
    --seed 1337 --val-every 10 \
    --save-prefix /tmp/qknorm_bwd_off_smoke 2>&1 | tail -20
```

Expected: bit-identical loss trajectory to baseline (qkNormEnabled=false).

- [ ] **Step 6: Commit**

```bash
cd ~/dev/glades-ml
git add Backend/Machine\ Learning/Networks/sgd_transformer.cpp \
        Backend/Machine\ Learning/Networks/cuda/gpu_kernels.cu \
        Backend/Machine\ Learning/Networks/cuda/gpu_kernels.h \
        Backend/Machine\ Learning/Networks/cuda/gpu_transformer_state.cu
git commit -m "$(cat <<'EOF'
Wire QK-Norm GPU backward + γ_h gradient accumulation

Backward chain:
  1. dQ_post-attention has shape of post-scale Q (γ·sqrt(dHead)·Q_norm).
  2. Accumulate γ_h grad: dγ_h = sqrt(dHead) · sum_{t,i} dQ_post · Q_norm.
  3. Undo γ·sqrt(dHead) scale (multiply dQ by same scale for chain rule).
  4. Undo Q L2 normalize via qknorm_backward_gpu.
  5. Same for K (no γ scale path).
  6. host-side accumulate dγ into b.gQknormGamma for Adam.

At qknormGamma.empty() the entire block is skipped — bit-identical
to baseline. 100-step smoke at qknorm-off confirms.
EOF
)"
```

### Task 2.7: QK-Norm enabled smoke train

**Files:** No code changes; verification.

- [ ] **Step 1: Run 500-step trainer smoke with QK-Norm enabled**

```bash
cd ~/dev/glades-trainer
./build/glades_chiron_train flagship --fp8-readout-fwd --qk-norm \
    --steps 500 --seed 1337 --val-every 50 \
    --save-prefix /tmp/qknorm_on_500_smoke 2>&1 | tee /tmp/qknorm_on_500_smoke.log
```

(If `--qk-norm` flag isn't wired in the trainer yet — Task 4.1 — invoke `glades_chiron_train` with a temporary code modification that sets `cfg.transformer.qkNormEnabled = true` and rebuild.)

- [ ] **Step 2: Check for NaN/inf and reasonable loss curve**

```bash
grep -E "nan|inf|step\s+(0|100|250|500)\s" /tmp/qknorm_on_500_smoke.log | head -20
```

Expected:
- No `nan` or `inf` in the log
- Loss decreases from ~10.6 at step 0 to ~4.8 at step 500 (similar to baseline trajectory)
- ||g|| (grad norm if logged) stays bounded (<10)

If NaN/inf appears, debug:
- Check γ_init: `log2(16384) = 14`. If γ is being decayed by Adam to negative/zero values, attention scores collapse. Try fixed γ for the first 100 steps.
- Check qInvNorm: should be ~1/16 for d_h=256 with random Q.

- [ ] **Step 3: Commit (no code; verification only)**

```bash
# No git commit needed — this is a verification run, not a code change.
echo "qknorm-on 500-step smoke verified."
```

---

## Phase 3: MTP (Multi-Token Prediction)

### Task 3.1: Add MTP weight + targets to Block / scratch

**Files:**
- Modify: Block declaration (transformer_state.h equivalent)
- Modify: `sgd_transformer.cpp` (allocation site)

- [ ] **Step 1: Add Wmtp + targetsMtp + Adam state to Block**

In the Block struct, add:

```cpp
	// MTP (Multi-Token Prediction) — single linear projection for +2-offset head.
	// Wmtp: (dModel, dModel). Maps post-final-LN hidden to MTP hidden, which
	// then goes through the tied readout. Empty when mtpDepth == 0.
	std::vector<float> Wmtp;
	std::vector<float> gWmtp;
	std::vector<float> mWmtp;
	std::vector<float> v2Wmtp;
```

(Single Wmtp per *model*, not per layer. So actually this goes on the model-level `TensorTransformerState` / `tt` object, not per-Block. Find where `tokE` lives and put it next to it.)

- [ ] **Step 2: Allocate + Glorot-init Wmtp when mtpDepth > 0**

At the model-level init site (where `tokE` is sized to `vocabSize * dModel`), add:

```cpp
if (transformerRunConfig.mtpDepth > 0)
{
	const size_t mtpSize = (size_t)dModel * dModel;
	tt.Wmtp.assign(mtpSize, 0.0f);
	// Glorot-uniform: bound = sqrt(6 / (fan_in + fan_out)) = sqrt(6 / (2*dModel))
	const float bound = sqrtf(6.0f / (2.0f * static_cast<float>(dModel)));
	for (size_t i = 0; i < mtpSize; ++i)
		tt.Wmtp[i] = (rng::nextUniform() * 2.0f - 1.0f) * bound;
	tt.gWmtp.assign(mtpSize, 0.0f);
	tt.mWmtp.assign(mtpSize, 0.0f);
	tt.v2Wmtp.assign(mtpSize, 0.0f);
}
```

(Use the project's existing `glades::rng::*` API for the RNG; check `Backend/Machine Learning/DETERMINISM_AND_CONCURRENCY.md` for the exact call.)

- [ ] **Step 3: Add MTP target buffer + zero / Adam update**

Add `std::vector<int> targetsMtp;` next to `targetIds` in the scratch. Add Adam update for `Wmtp` at the same sites as `tokE` updates.

- [ ] **Step 4: Build, run chiron suite, expect baseline still PASS**

```bash
cd ~/dev/glades-ml && sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests/build && sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests && bash test.sh chiron 2>&1 | tail -10
```

Expected: PASS (Wmtp empty when mtpDepth=0).

- [ ] **Step 5: Commit**

```bash
cd ~/dev/glades-ml
git add Backend/Machine\ Learning/Networks/transformer_state.h \
        Backend/Machine\ Learning/Networks/sgd_transformer.cpp
git commit -m "Add Wmtp + Adam state + targetsMtp scaffolding (empty when mtpDepth=0)"
```

### Task 3.2: MTP target-shift unit test

**Files:**
- Modify: `~/dev/glades-ml/unit-tests/Backend/Machine Learning/chiron-test.cpp`

- [ ] **Step 1: Add the target-shift test**

```cpp
void CHIRONMtpTargetShiftTest()
{
	// Verify: targetsMtp[t] = targetIds[t+1] for t in [0, T-2),
	// and targetsMtp[T-1] = paddingMarker (-1).
	const int T = 8;
	std::vector<int> targetIds(T);
	for (int t = 0; t < T; ++t) targetIds[t] = 100 + t;  // synthetic IDs

	std::vector<int> targetsMtp(T, -999);
	// The implementation in sgd_transformer.cpp will be:
	//   for (t = 0; t < T - 1; ++t) targetsMtp[t] = targetIds[t + 1];
	//   targetsMtp[T - 1] = -1;  // ignore label
	// For the test, replicate that logic and compare to a reference.
	// Reference helper (will exist after Task 3.3):
	glades::transformer_kernels::compute_mtp_targets(
	    &targetIds[0], T, /*ignoreLabel=*/-1, &targetsMtp[0]);

	for (int t = 0; t < T - 1; ++t)
	{
		ASSERT("mtp_target: shift +1 matches", targetsMtp[t] == 100 + t + 1);
	}
	ASSERT("mtp_target: last position is ignore label", targetsMtp[T - 1] == -1);
}
```

- [ ] **Step 2: Build — expect failure (compute_mtp_targets undefined)**

```bash
cd ~/dev/glades-ml/unit-tests/build && sh .configure.sh cuda 2>&1 | tail -10
```

Expected: link error referencing `compute_mtp_targets`.

### Task 3.3: Implement compute_mtp_targets helper

**Files:**
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/transformer_kernels.h` + .cpp

- [ ] **Step 1: Declare**

```cpp
	// Compute MTP +2-offset target IDs from a sequence of +1-offset targets.
	// targetIds[t] is the token at position t+1 (standard next-token target).
	// targetsMtp[t] = targetIds[t+1] for t < T-1; targetsMtp[T-1] = ignoreLabel.
	void compute_mtp_targets(const int* targetIds, int T, int ignoreLabel,
	                         int* targetsMtp);
```

- [ ] **Step 2: Implement**

```cpp
void compute_mtp_targets(const int* targetIds, int T, int ignoreLabel,
                         int* targetsMtp)
{
	for (int t = 0; t < T - 1; ++t) targetsMtp[t] = targetIds[t + 1];
	targetsMtp[T - 1] = ignoreLabel;
}
```

- [ ] **Step 3: Build + run unit test, expect PASS**

```bash
cd ~/dev/glades-ml && sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests/build && sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests && bash test.sh chiron 2>&1 | grep -E "mtp_target|FAILED" | head -10
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
cd ~/dev/glades-ml
git add Backend/Machine\ Learning/Networks/transformer_kernels.h \
        Backend/Machine\ Learning/Networks/transformer_kernels.cpp \
        unit-tests/Backend/Machine\ Learning/chiron-test.cpp
git commit -m "Add compute_mtp_targets helper + CHIRONMtpTargetShiftTest"
```

### Task 3.4: Wire MTP forward into the readout block

**Files:**
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/sgd_transformer.cpp`

- [ ] **Step 1: After the main CE loss accumulation, add MTP forward**

In the readout block (the area around `tied_embedding_logits_forward_rows`, ~line 7897-8020), after the main CE is accumulated, add:

```cpp
if (transformerRunConfig.mtpDepth > 0 && !tt.Wmtp.empty())
{
	// Compute MTP +2 targets.
	glades::transformer_kernels::compute_mtp_targets(
	    &targetIds[0], static_cast<int>(T),
	    /*ignoreLabel=*/transformerRunConfig.padTokenId >= 0
	        ? transformerRunConfig.padTokenId : -1,
	    &transformerScratch.targetsMtp[0]);

	// h_mtp[t] = Wmtp @ hPostFinalLN[t].
	// Reuse linear_forward_maybe_lowp for the (m,m) projection.
	glades::transformer_kernels::linear_forward_rows(
	    hPostFinalLN, T, dModel,
	    &tt.Wmtp[0], /*bias=*/NULL, dModel,
	    &transformerScratch.hMtp[0]);

	// logits_mtp[t, v] = tokE[v, :] · hMtp[t, :] + lmBias[v].
	glades::transformer_kernels::tied_embedding_logits_forward_rows(
	    &transformerScratch.hMtp[0], T, dModel,
	    tt.tokE, tt.lmBias, vocabSize,
	    &transformerScratch.logitsMtp[0]);

	// MTP loss accumulation.
	float mtpLossSum = 0.0f;
	int mtpCount = 0;
	for (unsigned int t = 0; t < T; ++t)
	{
		const int tgt = transformerScratch.targetsMtp[t];
		if (tgt < 0) continue;  // ignore label
		const float* row = &transformerScratch.logitsMtp[(size_t)t * vocabSize];
		float ce = 0.0f, zlossDummy = 0.0f;
		glades::transformer_kernels::softmax_ce_with_zloss(
		    row, static_cast<int>(vocabSize), tgt,
		    /*zlossCoef=*/0.0f, &ce, &zlossDummy);
		mtpLossSum += ce;
		++mtpCount;
	}
	const float mtpLossMean = (mtpCount > 0)
	    ? (mtpLossSum / static_cast<float>(mtpCount))
	    : 0.0f;
	runLoss += transformerRunConfig.mtpCoef * mtpLossMean;
}
```

Add `std::vector<float> hMtp;`, `std::vector<float> logitsMtp;`, `std::vector<int> targetsMtp;` to TransformerScratch + resize them when `mtpDepth > 0`.

- [ ] **Step 2: Build, run chiron suite, baseline parity check**

```bash
cd ~/dev/glades-ml && sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests && bash test.sh chiron 2>&1 | tail -10
cd ~/dev/glades-trainer
./build/glades_chiron_train flagship --fp8-readout-fwd --steps 100 \
    --seed 1337 --val-every 10 \
    --save-prefix /tmp/mtp_off_smoke 2>&1 | tail -20
```

Expected: bit-identical loss trajectory to baseline (mtpDepth=0 default → block skipped).

- [ ] **Step 3: Commit**

```bash
cd ~/dev/glades-ml
git add Backend/Machine\ Learning/Networks/sgd_transformer.cpp \
        Backend/Machine\ Learning/Networks/transformer_state.h
git commit -m "$(cat <<'EOF'
Wire MTP forward into readout block (CPU path)

When mtpDepth > 0, compute +2-offset targets, project h_post_LN through
Wmtp, route through tied readout, accumulate CE into runLoss with
mtpCoef weight. At mtpDepth == 0 the block is skipped — bit-identical
to baseline.
EOF
)"
```

### Task 3.5: MTP backward (GPU dispatch)

**Files:**
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/sgd_transformer.cpp` (GPU bwd block, ~line 10950)

- [ ] **Step 1: Add MTP backward right after the main readout backward**

In the GPU backward block, immediately after the existing main readout backward (the `gpu_gemm_atb_mp` for `gTokE`), add:

```cpp
if (transformerRunConfig.mtpDepth > 0 && !gpuTransformerWeights->Wmtp.data())
{
	// dLogitsMtp = softmax_ce_bwd(probsMtp, targetsMtp) * (mtpCoef / mtpCount).
	// Note: mtpCount could be T-1 or less if padding. For simplicity, use
	// the same per-row pattern as the main CE — the softmax_cross_entropy_bwd
	// kernel ignores rows where target < 0 by NOT writing them (or writing 0).
	// Add an "ignore label" variant if needed.

	glades::gpu::softmax_cross_entropy_bwd(
	    gpuTransformerScratch->probsMtp.data(),
	    gpuTransformerScratch->gpuTargetsMtpT.data(),
	    static_cast<int>(T), static_cast<int>(vocabSize),
	    gpuTransformerScratch->dLogitsMtp.data());

	// Scale dLogitsMtp by mtpCoef / T (since loss was sum/T·mtpCoef).
	const float mtpScale = transformerRunConfig.mtpCoef
	                     / static_cast<float>(T);
	glades::gpu::scale_inplace(
	    gpuTransformerScratch->dLogitsMtp.data(),
	    static_cast<size_t>(T) * vocabSize,
	    mtpScale);

	// dHmtp = dLogitsMtp @ tokE  (T, dModel)
	gpu_gemm_mp(bf16Head,
	    static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(vocabSize), 1.0f,
	    gpuTransformerScratch->dLogitsMtp.data(),
	    gpuTransformerWeights->tokE.data(),
	    0.0f, gpuTransformerScratch->dHmtp.data());

	// gTokE += dLogitsMtp^T @ hMtp  (vocabSize, dModel) — accumulate
	gpu_gemm_atb_mp(bf16Head,
	    static_cast<int>(vocabSize), static_cast<int>(dModel), static_cast<int>(T), 1.0f,
	    gpuTransformerScratch->dLogitsMtp.data(),
	    gpuTransformerScratch->hMtp.data(),
	    1.0f, gpuTransformerWeights->gTokE.data());

	// dWmtp += dHmtp^T @ hPostFinalLN  (dModel, dModel)
	gpu_gemm_atb_mp(bf16Head,
	    static_cast<int>(dModel), static_cast<int>(dModel), static_cast<int>(T), 1.0f,
	    gpuTransformerScratch->dHmtp.data(),
	    gpuTransformerScratch->hPostFinalLN.data(),
	    1.0f, gpuTransformerWeights->gWmtp.data());

	// dHPostFinalLN += dHmtp @ Wmtp^T  (T, dModel) — accumulate into existing dH
	gpu_gemm_abt_mp(bf16Head,
	    static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(dModel), 1.0f,
	    gpuTransformerScratch->dHmtp.data(),
	    gpuTransformerWeights->Wmtp.data(),
	    1.0f, gpuTransformerScratch->dH.data());
}
```

Add `gpuTransformerScratch->probsMtp`, `dLogitsMtp`, `dHmtp`, `hMtp`, `gpuTargetsMtpT` GPU buffers (allocate to T×vocabSize / T×dModel / T sizes).

Also add `gpu::scale_inplace(...)` if not already present:

```cpp
__global__ void scale_inplace_kernel(float* x, size_t n, float s)
{
	const size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
	if (idx < n) x[idx] *= s;
}
bool scale_inplace(float* x, size_t n, float s)
{
	const int block = 256;
	const size_t grid = (n + block - 1) / block;
	scale_inplace_kernel<<<grid, block, 0, computeStream()>>>(x, n, s);
	GLADES_CUDA_CHECK(cudaGetLastError());
	return true;
}
```

- [ ] **Step 2: Build, 100-step smoke at mtp-off**

```bash
cd ~/dev/glades-ml && sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-trainer
./build/glades_chiron_train flagship --fp8-readout-fwd --steps 100 \
    --seed 1337 --val-every 10 \
    --save-prefix /tmp/mtp_bwd_off_smoke 2>&1 | tail -20
```

Expected: bit-identical to baseline.

- [ ] **Step 3: Smoke at mtp-on (depth=1, coef=0.1)**

(Temporarily edit `sgd_transformer.cpp` or trainer to set `cfg.transformer.mtpDepth = 1; cfg.transformer.mtpCoef = 0.1f;` if `--mtp-depth` flag isn't wired yet.)

```bash
./build/glades_chiron_train flagship --fp8-readout-fwd --mtp-depth 1 \
    --mtp-coef 0.1 --steps 500 --seed 1337 --val-every 50 \
    --save-prefix /tmp/mtp_on_500_smoke 2>&1 | tail -20
```

Expected: loss decreases reasonably (no NaN/inf), with the *reported* training loss now including both main CE and MTP CE (so it's higher than the off variant by ~0.1× main_loss).

- [ ] **Step 4: Commit**

```bash
cd ~/dev/glades-ml
git add Backend/Machine\ Learning/Networks/sgd_transformer.cpp \
        Backend/Machine\ Learning/Networks/cuda/gpu_kernels.cu \
        Backend/Machine\ Learning/Networks/cuda/gpu_kernels.h \
        Backend/Machine\ Learning/Networks/cuda/gpu_transformer_state.cu
git commit -m "$(cat <<'EOF'
Wire MTP GPU backward: dLogitsMtp → dHmtp → dWmtp + tokE accumulation

dHmtp accumulates into dH so the residual stream gradient sees MTP
contribution. gTokE accumulates from BOTH main + MTP softmax-CE
backward passes (tied readout). gWmtp is new per-step accumulator.

500-step smoke at --mtp-depth 1 --mtp-coef 0.1 trains without NaN/inf;
loss higher than baseline by ~mtpCoef·mainCE as expected (MTP CE
counted in reported loss).
EOF
)"
```

---

## Phase 4: Trainer CLI + run.sh

### Task 4.1: Add CLI flags to trainer

**Files:**
- Modify: `~/dev/glades-trainer/trainer/main.cpp`

- [ ] **Step 1: Locate the existing flag-parsing block**

```bash
grep -n "tokenLmSampledNegatives\|attnSinkCount\|localAttnWindow" \
    ~/dev/glades-trainer/trainer/main.cpp | head -10
```

Find the cluster (~lines 1000-1090) where transformer-related config fields are populated from CLI args.

- [ ] **Step 2: Add argument variables + flag parsing**

In the variable declarations section (early in `main()`), add:

```cpp
float zlossCoef = 0.0f;
bool qkNormEnabled = false;
float qkNormGammaInit = 0.0f;
int mtpDepth = 0;
float mtpCoef = 0.1f;
```

In the argv-parsing loop (where flags like `--fp8-readout-fwd` are recognized), add cases:

```cpp
} else if (strcmp(argv[i], "--zloss-coef") == 0) {
    if (i + 1 >= argc) { /* error */ return 1; }
    zlossCoef = static_cast<float>(atof(argv[++i]));
} else if (strcmp(argv[i], "--qk-norm") == 0) {
    qkNormEnabled = true;
} else if (strcmp(argv[i], "--qk-norm-gamma-init") == 0) {
    if (i + 1 >= argc) return 1;
    qkNormGammaInit = static_cast<float>(atof(argv[++i]));
} else if (strcmp(argv[i], "--mtp-depth") == 0) {
    if (i + 1 >= argc) return 1;
    mtpDepth = atoi(argv[++i]);
} else if (strcmp(argv[i], "--mtp-coef") == 0) {
    if (i + 1 >= argc) return 1;
    mtpCoef = static_cast<float>(atof(argv[++i]));
}
```

- [ ] **Step 3: Populate `cfg.transformer` from the parsed args**

Near the existing population (e.g., next to `cfg.transformer.localAttnWindow = localAttnWindow;`):

```cpp
cfg.transformer.zlossCoef = zlossCoef;
cfg.transformer.qkNormEnabled = qkNormEnabled;
cfg.transformer.qkNormGammaInit = qkNormGammaInit;
cfg.transformer.mtpDepth = mtpDepth;
cfg.transformer.mtpCoef = mtpCoef;
```

- [ ] **Step 4: Build trainer**

```bash
cd ~/dev/glades-trainer && sh build.sh 2>&1 | tail -5
```

Expected: builds with no warnings.

- [ ] **Step 5: Verify flags accepted**

```bash
./build/glades_chiron_train flagship --fp8-readout-fwd --zloss-coef 1e-4 \
    --qk-norm --mtp-depth 1 --mtp-coef 0.1 --steps 10 --seed 1337 \
    --save-prefix /tmp/flags_test 2>&1 | head -30
```

Expected: trainer starts without "unknown flag" errors; runs 10 steps.

- [ ] **Step 6: Commit (in glades-trainer repo)**

```bash
cd ~/dev/glades-trainer
git add trainer/main.cpp
git commit -m "Add --zloss-coef, --qk-norm, --qk-norm-gamma-init, --mtp-depth, --mtp-coef flags"
```

### Task 4.2: Wire flags into run.sh flagship recipe

**Files:**
- Modify: `~/dev/glades-trainer/run.sh`

- [ ] **Step 1: Read the flagship block to find the arg passthrough**

```bash
grep -n "flagship\|fp8-readout" ~/dev/glades-trainer/run.sh | head -20
```

Find the shell case-statement or function that builds the command-line for the `flagship` recipe. New flags should be passed through if set as shell vars.

- [ ] **Step 2: Add passthrough handling**

In the flagship-args section, add:

```bash
# Regularization stack flags (regstack spec 2026-05-22).
if [ -n "${ZLOSS_COEF:-}" ]; then
    FLAGSHIP_ARGS="$FLAGSHIP_ARGS --zloss-coef $ZLOSS_COEF"
fi
if [ "${QK_NORM:-0}" = "1" ]; then
    FLAGSHIP_ARGS="$FLAGSHIP_ARGS --qk-norm"
fi
if [ -n "${MTP_DEPTH:-}" ] && [ "$MTP_DEPTH" -gt 0 ]; then
    FLAGSHIP_ARGS="$FLAGSHIP_ARGS --mtp-depth $MTP_DEPTH --mtp-coef ${MTP_COEF:-0.1}"
fi
```

Alternative (simpler): just pass `"$@"` through so the user can put the flags directly on the run.sh command line, no shell-var setup needed. Check what the existing pattern does.

- [ ] **Step 3: Verify with --zloss-coef on the run.sh command line**

```bash
cd ~/dev/glades-trainer
sh run.sh flagship --fp8-readout-fwd --zloss-coef 1e-4 --steps 10 \
    --seed 1337 --save-prefix /tmp/runsh_test 2>&1 | head -20
```

Expected: runs 10 steps.

- [ ] **Step 4: Commit**

```bash
cd ~/dev/glades-trainer
git add run.sh
git commit -m "Plumb regstack flags (--zloss-coef, --qk-norm, --mtp-depth, --mtp-coef) through flagship recipe"
```

---

## Phase 5: Validation arc (per spec §3)

### Task 5.1: Baseline B0 5k single-seed

**Files:** None; data run.

- [ ] **Step 1: Confirm VRAM available + GPU clocks**

```bash
nvidia-smi --query-gpu=memory.used,memory.total,clocks.current.graphics --format=csv
```

Expected: <2 GB used (no other workload), full 15.56 GB available.

- [ ] **Step 2: Launch B0**

```bash
cd ~/dev/glades-trainer
sh run.sh flagship --fp8-readout-fwd --steps 5000 --seed 1337 \
    --val-every 500 \
    --save-prefix database/checkpoints/chiron_1B_T16384_b0_5k \
    2>&1 | tee logs/regstack_b0_5k.log
```

Expected: ~47 minutes wall, final val NLL at step 5000 reported.

- [ ] **Step 3: Extract result**

```bash
grep -E "step\s+5000\s|val.*NLL" ~/dev/glades-trainer/logs/regstack_b0_5k.log | tail -10
```

Record:
- `B0_val_NLL_5k` = ___ (main-head CE only)
- `B0_throughput_tok_s` = ___
- `B0_peak_vram_GB` = ___

### Task 5.2: Z-loss B1 5k

- [ ] **Step 1: Launch B1**

```bash
cd ~/dev/glades-trainer
sh run.sh flagship --fp8-readout-fwd --zloss-coef 1e-4 \
    --steps 5000 --seed 1337 --val-every 500 \
    --save-prefix database/checkpoints/chiron_1B_T16384_b1_zloss_5k \
    2>&1 | tee logs/regstack_b1_zloss_5k.log
```

- [ ] **Step 2: Extract result + gate check**

```bash
grep -E "step\s+5000\s|val.*NLL" ~/dev/glades-trainer/logs/regstack_b1_zloss_5k.log | tail -10
```

Record `B1_val_NLL_5k` (main-head CE only, excluding Z-loss aux term — confirm trainer logs both).

Gate: `B1_val_NLL_5k <= B0_val_NLL_5k + 0.02` (no >0.02 nat regression).

### Task 5.3: QK-Norm B2 5k

- [ ] **Step 1: Launch B2**

```bash
cd ~/dev/glades-trainer
sh run.sh flagship --fp8-readout-fwd --qk-norm \
    --steps 5000 --seed 1337 --val-every 500 \
    --save-prefix database/checkpoints/chiron_1B_T16384_b2_qknorm_5k \
    2>&1 | tee logs/regstack_b2_qknorm_5k.log
```

- [ ] **Step 2: Extract + gate**

```bash
grep -E "step\s+5000\s|val.*NLL" ~/dev/glades-trainer/logs/regstack_b2_qknorm_5k.log | tail -10
```

Record `B2_val_NLL_5k`. Gate as in Task 5.2.

If FAIL (regression > 0.02 nat), check R-RegStack-2 (SCFA inner attention QK-Norm interaction): try `--qk-norm` on outer attention only as a fallback. See spec §4.

### Task 5.4: MTP B3 5k

- [ ] **Step 1: Launch B3**

```bash
cd ~/dev/glades-trainer
sh run.sh flagship --fp8-readout-fwd --mtp-depth 1 --mtp-coef 0.1 \
    --steps 5000 --seed 1337 --val-every 500 \
    --save-prefix database/checkpoints/chiron_1B_T16384_b3_mtp_5k \
    2>&1 | tee logs/regstack_b3_mtp_5k.log
```

- [ ] **Step 2: Extract + gate**

```bash
grep -E "step\s+5000\s|val.*NLL" ~/dev/glades-trainer/logs/regstack_b3_mtp_5k.log | tail -10
```

Record `B3_val_NLL_5k` (main-head CE only — strip out MTP aux term). Gate as in Task 5.2.

### Task 5.5: Stacked B4 5k

- [ ] **Step 1: Launch B4**

```bash
cd ~/dev/glades-trainer
sh run.sh flagship --fp8-readout-fwd --zloss-coef 1e-4 --qk-norm \
    --mtp-depth 1 --mtp-coef 0.1 \
    --steps 5000 --seed 1337 --val-every 500 \
    --save-prefix database/checkpoints/chiron_1B_T16384_b4_stacked_5k \
    2>&1 | tee logs/regstack_b4_stacked_5k.log
```

- [ ] **Step 2: Extract + apply decision tree**

```bash
grep -E "step\s+5000\s|val.*NLL" ~/dev/glades-trainer/logs/regstack_b4_stacked_5k.log | tail -10
```

Record `B4_val_NLL_5k`.

Compute `Δ_i = B0_val_NLL_5k - B_i_val_NLL_5k` for i in 1..4.

Per spec §3.3:
- If `Δ4 ≥ max(Δ1, Δ2, Δ3)` → B5 := B4 (all three stacked)
- Else if `Δ4 ≥ 0.02` → B5 := B4 (sub-additive but still clears bar)
- Else if `max(Δ1, Δ2, Δ3) ≥ 0.02` → B5 := single best Bi
- Else → no B5; publish negative pilot result

Document the choice in `~/dev/glades-ml/research/REGSTACK_PILOT_2026_MM_DD.md`.

### Task 5.6: Production retrain B5 30k

- [ ] **Step 1: Launch B5 (config depends on decision tree from 5.5)**

Example for B4-stacked-wins case:

```bash
cd ~/dev/glades-trainer
sh run.sh flagship --fp8-readout-fwd --zloss-coef 1e-4 --qk-norm \
    --mtp-depth 1 --mtp-coef 0.1 \
    --steps 30000 --seed 1337 --val-every 1500 \
    --save-prefix database/checkpoints/chiron_1B_T16384_v5_fp8_regstack_phase2 \
    2>&1 | tee logs/regstack_b5_phase2.log
```

Expected: ~4.7 hours wall. Run in foreground with `tee`; safe to leave overnight.

- [ ] **Step 2: Extract gate-by-gate results**

```bash
grep -E "step\s+(15000|30000)\s|val.*NLL|tok/s|peak" ~/dev/glades-trainer/logs/regstack_b5_phase2.log | tail -20
```

Record:
- `B5_val_NLL_30k` (main-head CE)
- `B5_throughput_tok_s`
- `B5_peak_vram_GB`

Gate (per spec §3.2):
- `B5_val_NLL_30k <= 4.1517` (≥ 0.02 nat improvement vs v5+FP8 ship 4.1717)
- `B5_throughput_tok_s >= 27,500` (≤ 5% wall regression)
- `B5_peak_vram_GB <= 15.72` (≤ 1% above the 15.56 GB ceiling)

---

## Phase 6: Documentation + ship

### Task 6.1: Write result doc

**Files:**
- Create: `~/dev/glades-ml/research/REGSTACK_PHASE2_<RESULT>_2026_MM_DD.md` (replace `<RESULT>` with `PASS` or `FAIL`, fill in actual date)

- [ ] **Step 1: Write the doc**

Sections to include (mirror `V5_FP8_30K_PHASE2_PASS_2026_05_22.md` structure):
1. **Headline**: regstack mechanism, +Δ NLL, +Δ wall, +Δ VRAM.
2. **Configuration**: full B5 config + reproduce command.
3. **Trajectory table**: val NLL at step {1500, 3000, ..., 30000} for v5+FP8 baseline vs regstack ship.
4. **Per-mechanism attribution from B1/B2/B3/B4 pilots**: a table mapping each mechanism to its individual + stacked Δ.
5. **Gate-by-gate evidence**: each gate from spec §3.2 with PASS/FAIL marking.
6. **Risks observed vs pre-registered**: cross-reference to spec §4.

- [ ] **Step 2: Commit**

```bash
cd ~/dev/glades-ml
git add research/REGSTACK_PHASE2_*.md
git commit -m "Document regstack Phase 2 result (PASS/FAIL details inline)"
```

### Task 6.2: Update CLAUDE.md flagship pointer (only if B5 PASSes)

**Files:**
- Modify: `~/dev/glades-ml/CLAUDE.md`

- [ ] **Step 1: Update Current Production Flagship block**

Replace the `chiron_1B_T16384_v5_fp8_phase2.final` checkpoint reference with `chiron_1B_T16384_v5_fp8_regstack_phase2.final`. Update:
- Perf line (new tok/s, NLL)
- Reproduce training command (add new flags)
- Spec doc reference (add `research/REGSTACK_PHASE2_PASS_*.md`)

Also: move the existing v5+FP8 ship to the "Prior iter X ship" section.

- [ ] **Step 2: Commit**

```bash
cd ~/dev/glades-ml
git add CLAUDE.md
git commit -m "Update flagship pointer to chiron_1B_T16384_v5_fp8_regstack_phase2"
```

### Task 6.3: Archive prior ship + memory entry

- [ ] **Step 1: Archive prior ship checkpoint**

```bash
cd ~/dev/glades-trainer
mv database/checkpoints/chiron_1B_T16384/chiron_1B_T16384_v5_fp8_phase2.final \
   database/checkpoints/chiron_1B_T16384_v5_fp8_phase2/
```

(Or follow whatever the existing archival pattern is — check the iter 94 / iter 116 archives.)

- [ ] **Step 2: Write memory entry**

Save a memory entry summarizing the result. Use the pattern of existing entries from `~/.claude/projects/-home-robert-dev-glades-ml/memory/` (e.g., `v5_fp8_30k_phase2_pass.md`).

Key fields for the memory entry:
- name: `regstack_phase2_<result>`
- description: one-line headline (e.g., "Z-loss+QK-Norm+MTP stack: +X% wall, -Y nat NLL, ships at chiron_1B_T16384_v5_fp8_regstack_phase2.final")
- body: per-mechanism deltas, gate evidence

Then add the entry to `MEMORY.md` index.

---

## Self-Review

**Spec coverage check** (mapping spec sections to plan tasks):

- Spec §2.1 (Z-loss architecture) → Tasks 1.1-1.5 + 4.1
- Spec §2.2 (QK-Norm architecture) → Tasks 2.1-2.7 + 4.1
- Spec §2.3 (MTP architecture) → Tasks 3.1-3.5 + 4.1
- Spec §3.1 (run plan) → Tasks 5.1-5.6
- Spec §3.2 (gate criteria) → embedded in Tasks 5.1-5.6
- Spec §3.3 (decision tree) → Task 5.5 Step 2
- Spec §3.4 (production retrain handoff) → Tasks 6.1-6.3
- Spec §4 (risks) → cross-referenced from gates; mitigation steps embedded
- Spec §5 (out of scope) → enforced by NOT including those mechanisms
- Spec §7 (repro commands) → matched in Tasks 5.1-5.6

All spec sections covered.

**Open risks the engineer should be aware of:**

1. **Block struct location**: the exact filename for the `Block` struct was not verified during plan-writing. Task 2.1 Step 1's `grep` finds the right file; if it's split across multiple headers, the engineer needs to put the new fields in whichever header declares `ln1Gamma` and update the matching `transformer_chiron_ops.h` view structs.

2. **GPU forward attention variants**: there are 8 sites with `rsqrtf(dHead)` in `gpu_kernels.cu` (lines 4148, 4230, 4382, 4482, 4689, 4779, 4879, 5008) covering forward/backward × FP32/BF16 × different attention variants. The QK-Norm approach in Task 2.4 (pre-multiply Q by `γ·sqrt(dHead)`) is designed to LEAVE these sites alone — the kernel's `1/sqrt(dHead)` survives, and the pre-multiply on Q recovers the desired effective scale. Engineer should verify this is consistent across all 8 variants (FP32 / BF16 / FlashAttn / SCFA inner / SCFA outer); any variant that applies the scale differently (e.g., on K instead of Q, or post-dot-product instead of pre-multiply) needs special handling.

3. **SCFA inner attention QK-Norm interaction (R-RegStack-2)**: the SCFA inner attention uses compressed Q/K via depthwise causal conv (`scfa_depthwise_causal_conv_fwd`, gpu_kernels.cu:8142+). Applying L2 norm on the compressed Q/K representations is mathematically different from L2 norm on the full Q/K. The pilot run B2 (Task 5.3) is the integration test for this — if it fails, fallback per the spec.

4. **MTP increases reported loss**: B3/B4 logs will show higher training loss than B0/B1/B2 because MTP CE is included in the reported number. The gate compares MAIN-HEAD CE only (Task 5.4 Step 2). Verify the trainer logs both separately, or extract them from val-eval output.

5. **The trainer is in a separate repo**: changes in `~/dev/glades-trainer/` (Tasks 4.1, 4.2) must be committed in THAT repo's git, not glades-ml's. The library and trainer must be rebuilt/installed in sequence after every library change.

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-22-chiron-1b-regularization-stack.md`.**
