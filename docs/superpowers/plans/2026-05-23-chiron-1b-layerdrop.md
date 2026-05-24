# CHIRON 1B LayerDrop (Stochastic Depth) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add LayerDrop (stochastic depth) — per-layer Bernoulli skip with linear-rising `p_l = (l/(L-1))·p_max` and inverted-dropout rescaling — to the CHIRON 1B regstack Phase 2 production flagship. Gated by a 5k single-seed pilot and a 30k Phase-2 retrain.

**Architecture:** A new config field `layerDropPMax` (default `0.0f` = no-op, bit-identical to regstack ship) drives a per-step per-layer Bernoulli mask. When the mask is `1` (keep), both sub-residuals of the block (attn-residual and FFN-residual) are scaled by `1/(1-p_l)` and added to the residual stream. When the mask is `0` (drop), the entire block — both attention (incl. SCFA inner + outer, Q/K/V projections, QK-Norm, Wo) and FFN (W1, activation, W2) — is skipped on both forward and backward, saving compute. The mask is drawn from the per-network deterministic `rngEngine` via the existing `glades::transformer_kernels::generate_dropout_mask` helper called with `n=1` per layer.

**Tech Stack:** C++98 + CUDA (CUDA 13.2 toolchain), shared with the `libglades.so` library. Trainer is a separate repo (`~/dev/glades-trainer`) that wraps the library and exposes CLI flags. Tests use the project's `ASSERT(failmsg, predicate)` macro from `unit-tests/unit-test.h`.

**Repos involved:**
- `~/dev/glades-ml/` — library (most changes here)
- `~/dev/glades-trainer/` — trainer binary + CLI + `run.sh` recipe

**Execution recommendation:** Run this plan in an isolated git worktree (per `superpowers:using-git-worktrees`). All training runs use the existing `~/dev/glades-trainer` install — no need to worktree that repo.

**Source-of-truth spec:** `docs/superpowers/specs/2026-05-23-chiron-1b-layerdrop-design.md`. Refer to it for motivation, gates, and risks. This plan implements that spec.

**Convention used in this plan (timm-style with shared mask):** "Full-block drop" means a single Bernoulli mask `mask_l` per (step, layer). When `mask_l = 1`, BOTH the attn sub-residual and the FFN sub-residual are scaled by `s = 1/(1-p_l)`:
```
hAfterAttn = hIn        + s · attnOut       (sub-residual 1)
final      = hAfterAttn + s · ffOut         (sub-residual 2)
```
When `mask_l = 0`, neither sub-residual is computed:
```
final = hIn
```
The inner LN2 sees `hAfterAttn = hIn + s · attnOut` (the scaled residual stream), which is the standard timm/torchvision convention.

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

Expected: clean working tree on branch `chiron2`, top commit is `d63a759cc Pre-register CHIRON 1B LayerDrop arc design`.

- [ ] **Step 2: Build library with CUDA (clean baseline)**

```bash
cd ~/dev/glades-ml
sh .configure.sh cuda 2>&1 | tail -10
```

Expected: build completes without errors. `~/.local/lib/libglades.so` updated.

- [ ] **Step 3: Build unit tests**

```bash
cd ~/dev/glades-ml/unit-tests/build && sh .configure.sh cuda 2>&1 | tail -5
```

Expected: build completes.

- [ ] **Step 4: Run existing chiron tests as sanity check**

```bash
cd ~/dev/glades-ml/unit-tests && bash test.sh chiron 2>&1 | tail -30
```

Expected: all ASSERTs pass. Note any pre-existing test failures so they aren't blamed on this work. The regstack tests (`CHIRONZlossDisabledParityTest`, `CHIRONQkNormEnabledMathTest`, etc.) should all pass.

### Task 0.2: Add config fields (no-op defaults)

**Files:**
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/training_config.h`
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/transformer_config.cpp`

- [ ] **Step 1: Add LayerDrop field declarations to TransformerRunConfig**

Open `Backend/Machine Learning/Networks/training_config.h`. After the `mtpCoef` field declaration (around line 253, just before the `TransformerRunConfig()` constructor at line 255), insert:

```cpp
	// LayerDrop / stochastic depth (paradigm shift: Fan 2019 / Huang 2016 /
	// timm). When > 0, each transformer block l ∈ {0..L-1} is dropped with
	// probability p_l. Linear-rising schedule: p_l = (l/(L-1)) · layerDropPMax,
	// so layer 0 never drops and the deepest layer drops with probability
	// layerDropPMax. Kept layers' sub-residuals are scaled by 1/(1-p_l)
	// (inverted-dropout convention). Default 0.0f = disabled, math
	// bit-identical to baseline. Recommended for CHIRON 1B: 0.1.
	float layerDropPMax;

	// LayerDrop schedule type. true = linear-rising (p_l = (l/(L-1)) · pMax,
	// timm convention). false = constant (p_l = pMax for all l). Default
	// true. The constant variant is reserved for a possible future arc; this
	// arc uses linear-rising exclusively.
	bool layerDropLinearSchedule;
```

- [ ] **Step 2: Add field initializers in the constructor**

In the same file, find the `TransformerRunConfig()` constructor initializer list. It currently ends at `mtpCoef(0.1f)` followed by `{ }`. Modify the initializer list so that `mtpCoef(0.1f)` has a trailing comma and append:

```cpp
	      mtpCoef(0.1f),
	      layerDropPMax(0.0f),
	      layerDropLinearSchedule(true)
```

(Leave the `{ }` body unchanged.)

- [ ] **Step 3: Add validation in transformer_config.cpp**

Open `Backend/Machine Learning/Networks/transformer_config.cpp`. Find the existing `embeddingDropoutRate` validation block (around lines 47-50). After the `residualDropoutRate` validation, add:

```cpp
	if (runtimeCfg.layerDropPMax < 0.0f || runtimeCfg.layerDropPMax >= 1.0f)
		return invalid_argument(where, "layerDropPMax must be in [0,1)");
```

- [ ] **Step 4: Rebuild library**

```bash
cd ~/dev/glades-ml
sh .configure.sh cuda 2>&1 | tail -10
```

Expected: build completes, `libglades.so` updated. No new warnings about uninitialized members.

- [ ] **Step 5: Rebuild unit tests + run chiron suite**

```bash
cd ~/dev/glades-ml/unit-tests/build && sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests && bash test.sh chiron 2>&1 | tail -10
```

Expected: same pass/fail status as Task 0.1 step 4. The new fields are unused so nothing changes.

- [ ] **Step 6: Commit**

```bash
cd ~/dev/glades-ml
git add Backend/Machine\ Learning/Networks/training_config.h \
        Backend/Machine\ Learning/Networks/transformer_config.cpp
git commit -m "$(cat <<'EOF'
Add layerDropPMax, layerDropLinearSchedule config fields

Both default to no-op values (0.0f / true). No behavior change.
Scaffolding for the LayerDrop spec; subsequent commits wire them
through the CPU + GPU forward/backward passes guarded by the
layerDropPMax == 0.0f early-return.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 1: LayerDrop math helpers + unit tests

### Task 1.1: Add layer-schedule helper + math unit test

**Files:**
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/transformer_kernels.h`
- Modify: `~/dev/glades-ml/unit-tests/Backend/Machine Learning/chiron-test.cpp`
- Modify: `~/dev/glades-ml/unit-tests/Backend/Machine Learning/chiron-test.h`
- Modify: `~/dev/glades-ml/unit-tests/main.cpp`

- [ ] **Step 1: Add the linear-rising p_l helper to transformer_kernels.h**

Open `Backend/Machine Learning/Networks/transformer_kernels.h`. Find the existing `generate_dropout_mask` template (around line 1483) and immediately AFTER its closing `}`, insert:

```cpp
// === LayerDrop schedule helper ===
//
// Linear-rising stochastic-depth schedule:
//   p_l = (l / (L - 1)) * pMax,    l ∈ {0, 1, ..., L-1}
// So p_0 = 0 (layer 0 never drops; protects embedding-adjacent state),
// p_{L-1} = pMax (deepest layer drops with probability pMax).
// When L == 1, returns 0 (the only layer is never dropped).
inline float layer_drop_p_l(unsigned int l, unsigned int L, float pMax, bool linearSchedule)
{
	if (pMax <= 0.0f || L == 0u)
		return 0.0f;
	if (!linearSchedule)
		return pMax;
	if (L == 1u)
		return 0.0f;
	return (static_cast<float>(l) / static_cast<float>(L - 1u)) * pMax;
}

// Single-Bernoulli draw helper (returns true = "keep this layer", false = "drop").
// Reuses generate_dropout_mask with n=1 to share the determinism path.
template <typename EngineT>
inline bool layer_drop_keep(EngineT& eng, float p_l)
{
	if (p_l <= 0.0f)
		return true;
	if (p_l >= 1.0f)
		return false;
	unsigned char mask = 1;
	generate_dropout_mask(eng, &mask, 1u, p_l);
	return mask != 0u;
}
```

- [ ] **Step 2: Declare unit test function in chiron-test.h**

Open `unit-tests/Backend/Machine Learning/chiron-test.h`. Find the block of existing test function declarations (e.g., where `CHIRONZlossDisabledParityTest` is declared) and append:

```cpp
void CHIRONLayerDropScheduleMathTest();
void CHIRONLayerDropDisabledParityTest();
void CHIRONLayerDropDeterministicMasksTest();
```

(All three declared up-front so we don't re-edit the header.)

- [ ] **Step 3: Implement the schedule math test in chiron-test.cpp**

Open `unit-tests/Backend/Machine Learning/chiron-test.cpp`. At the bottom (after the last existing test function), add:

```cpp
// === LAYERDROP TESTS (2026-05-23 spec) ===

// Verify that layer_drop_p_l(l, L=24, pMax=0.1, linear=true) gives
// p_0 = 0, p_{23} = 0.1, p_{12} ≈ 0.0522..., and the sum over l matches
// the expected (pMax * L / 2) = 1.2.
void CHIRONLayerDropScheduleMathTest()
{
	using glades::transformer_kernels::layer_drop_p_l;

	const unsigned int L = 24;
	const float pMax = 0.1f;

	// p_0 = 0
	{
		const float p0 = layer_drop_p_l(0u, L, pMax, true);
		ASSERT("CHIRONLayerDropSchedule: p_0 must be 0", p0 == 0.0f);
	}

	// p_{L-1} = pMax
	{
		const float pLast = layer_drop_p_l(L - 1u, L, pMax, true);
		ASSERT("CHIRONLayerDropSchedule: p_{L-1} must equal pMax",
		       fabsf(pLast - pMax) < 1e-7f);
	}

	// p_{12} = (12/23) * 0.1 ≈ 0.0521739
	{
		const float pMid = layer_drop_p_l(12u, L, pMax, true);
		const float expected = (12.0f / 23.0f) * 0.1f;
		ASSERT("CHIRONLayerDropSchedule: p_{12} must match (12/23) * pMax",
		       fabsf(pMid - expected) < 1e-6f);
	}

	// Sum over l ∈ {0..L-1} = pMax * sum(0..L-1) / (L-1) = pMax * (L*(L-1)/2) / (L-1) = pMax * L / 2 = 1.2
	{
		float sum = 0.0f;
		for (unsigned int l = 0u; l < L; ++l)
			sum += layer_drop_p_l(l, L, pMax, true);
		const float expected = pMax * static_cast<float>(L) / 2.0f;  // 1.2
		ASSERT("CHIRONLayerDropSchedule: sum p_l must equal pMax*L/2 = 1.2",
		       fabsf(sum - expected) < 1e-5f);
	}

	// Constant schedule: all layers get pMax.
	{
		for (unsigned int l = 0u; l < L; ++l)
		{
			const float p = layer_drop_p_l(l, L, pMax, false);
			ASSERT("CHIRONLayerDropSchedule: constant schedule gives pMax",
			       fabsf(p - pMax) < 1e-7f);
		}
	}

	// L == 1: p_0 = 0 regardless of pMax.
	{
		const float p = layer_drop_p_l(0u, 1u, pMax, true);
		ASSERT("CHIRONLayerDropSchedule: L=1 must give p_0 = 0", p == 0.0f);
	}

	// pMax = 0: all p_l = 0.
	{
		for (unsigned int l = 0u; l < L; ++l)
		{
			const float p = layer_drop_p_l(l, L, 0.0f, true);
			ASSERT("CHIRONLayerDropSchedule: pMax=0 must give all p_l = 0",
			       p == 0.0f);
		}
	}
}
```

- [ ] **Step 4: Register the test in main.cpp**

Open `unit-tests/main.cpp`. Find where the existing regstack tests are invoked (e.g., `CHIRONZlossDisabledParityTest();`). Below them, add:

```cpp
	CHIRONLayerDropScheduleMathTest();
	CHIRONLayerDropDisabledParityTest();
	CHIRONLayerDropDeterministicMasksTest();
```

The latter two are defined in subsequent tasks; declaring them now keeps the registration block tidy.

- [ ] **Step 5: Add forward-declaration stubs for tasks 1.3 and 2.3**

In `unit-tests/Backend/Machine Learning/chiron-test.cpp`, immediately AFTER `CHIRONLayerDropScheduleMathTest()` (the function you just added), add empty placeholder bodies so the build links:

```cpp
void CHIRONLayerDropDisabledParityTest()
{
	// Implemented in Task 1.3 below.
}

void CHIRONLayerDropDeterministicMasksTest()
{
	// Implemented in Task 2.3 below.
}
```

- [ ] **Step 6: Rebuild + run the schedule test only**

```bash
cd ~/dev/glades-ml
sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests/build && sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests && bash test.sh chiron 2>&1 | tail -20
```

Expected: build succeeds; `CHIRONLayerDropScheduleMathTest` passes; the two stub tests pass trivially (empty bodies).

- [ ] **Step 7: Commit**

```bash
cd ~/dev/glades-ml
git add Backend/Machine\ Learning/Networks/transformer_kernels.h \
        unit-tests/Backend/Machine\ Learning/chiron-test.cpp \
        unit-tests/Backend/Machine\ Learning/chiron-test.h \
        unit-tests/main.cpp
git commit -m "$(cat <<'EOF'
Add layer_drop_p_l + layer_drop_keep helpers with math unit test

layer_drop_p_l(l, L, pMax, linear) returns the per-layer drop
probability under either linear-rising (p_l = (l/(L-1))*pMax, timm
convention) or constant schedule (p_l = pMax). layer_drop_keep(eng, p_l)
draws a single Bernoulli for the layer using the existing
generate_dropout_mask determinism path.

CHIRONLayerDropScheduleMathTest verifies the schedule at L=24, pMax=0.1
gives p_0=0, p_{23}=pMax, sum over l = pMax*L/2 = 1.2, and constant /
edge cases.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2: LayerDrop CPU forward + backward wiring

### Task 2.1: Add per-step mask buffer to TransformerScratch

**Files:**
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/sgd_transformer.cpp` (search for "TransformerScratch" struct definition and the buffer-allocation site)

- [ ] **Step 1: Find the TransformerScratch struct + allocation site**

```bash
grep -n "struct TransformerScratch\|dropoutMaskResAttn\|dropoutMaskResFF" "/home/robert/dev/glades-ml/Backend/Machine Learning/Networks/sgd_transformer.cpp" | head -20
```

Note the line numbers of:
- The struct definition (likely near the top, with member declarations like `std::vector<unsigned char> dropoutMaskResAttn;`).
- The `.resize(...)` call sites where those buffers are sized at training-step setup.

- [ ] **Step 2: Add `layerDropKept` mask buffer to the struct**

Inside the `TransformerScratch` struct, alongside `dropoutMaskResAttn` and `dropoutMaskResFF`, add:

```cpp
	// LayerDrop per-step per-layer keep mask. Indexed by `li` (layer index).
	// Value 1 = block kept (executed); 0 = block dropped (skipped fwd+bwd).
	// Sized to nLayers when layerDropPMax > 0, empty otherwise.
	std::vector<unsigned char> layerDropKept;
```

- [ ] **Step 3: Size the buffer at the existing scratch-allocation site**

Find the site where `dropoutMaskResAttn.resize(...)` is called (near the start of the training step / scratch allocation for the layer loop). Alongside it, add:

```cpp
	if (trainingConfig.transformer.layerDropPMax > 0.0f)
		transformerScratch.layerDropKept.resize(nLayers, 1u);  // default kept
	else if (!transformerScratch.layerDropKept.empty())
		transformerScratch.layerDropKept.clear();
```

(The default-1 init means `if (layerDropPMax == 0)` behavior is bit-identical: kept buffer stays empty, fwd/bwd take the "no LayerDrop" path.)

- [ ] **Step 4: Rebuild + run chiron tests as smoke check**

```bash
cd ~/dev/glades-ml
sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests/build && sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests && bash test.sh chiron 2>&1 | tail -5
```

Expected: all pass; new buffer is unused so behavior unchanged.

- [ ] **Step 5: Commit**

```bash
cd ~/dev/glades-ml
git add Backend/Machine\ Learning/Networks/sgd_transformer.cpp
git commit -m "$(cat <<'EOF'
Add layerDropKept mask buffer to TransformerScratch

Per-step per-layer Bernoulli mask buffer for the LayerDrop wiring in
subsequent commits. Sized to nLayers when layerDropPMax > 0, empty
otherwise (bit-identical no-op when LayerDrop is disabled).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 2.2: Wire LayerDrop into CPU forward pass

**Files:**
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/sgd_transformer.cpp` (around lines 7812-8095, the CPU per-layer forward loop)

- [ ] **Step 1: Read the CPU per-layer forward loop**

```bash
sed -n '7810,7820p;7980,8005p;8065,8100p' "/home/robert/dev/glades-ml/Backend/Machine Learning/Networks/sgd_transformer.cpp"
```

Confirm:
- Line ~7813: `for (unsigned int li = 0; li < nLayers; ++li)`
- Line ~7983-7996: attention residual dropout block (immediately before the attn residual add)
- Line ~8000-8001: attention residual add (`hAfterAttn[i] = hIn[i] + attnOut[i];`)
- Line ~8072-8081: FFN residual dropout block (immediately before the FFN residual add)
- Line ~8093-8101 (approx): FFN residual add (the final `hAfterFinalAdd[i] = hAfterAttn[i] + ffOut[i];` — exact line varies; locate by `for (size_t i = ...; i++) hAfter... = hAfterAttn[i] + ffOut[i];`).

- [ ] **Step 2: Add LayerDrop mask draw + skip-fwd shortcut at the top of the layer loop**

At the very start of the `for (unsigned int li = 0; li < nLayers; ++li)` body (immediately AFTER the `hIn` pointer is set up at line 7816-7817, but BEFORE LN1 starts), insert:

```cpp
		// === LayerDrop mask draw (paradigm shift: Fan 2019 / Huang 2016) ===
		// Decide per-step whether to drop this entire block. When dropped,
		// final residual stream = hIn (no attn/FFN contribution). Mask is
		// persisted to transformerScratch.layerDropKept[li] for the bwd
		// pass to read.
		float layerDropPL = 0.0f;
		float layerDropScale = 1.0f;
		bool layerDropKeepThisLayer = true;
		if (trainingConfig.transformer.layerDropPMax > 0.0f &&
		    !transformerScratch.layerDropKept.empty())
		{
			layerDropPL = glades::transformer_kernels::layer_drop_p_l(
			    li, nLayers, trainingConfig.transformer.layerDropPMax,
			    trainingConfig.transformer.layerDropLinearSchedule);
			layerDropKeepThisLayer = glades::transformer_kernels::layer_drop_keep(
			    rngEngine, layerDropPL);
			layerDropScale = (layerDropPL > 0.0f && layerDropPL < 1.0f)
			    ? (1.0f / (1.0f - layerDropPL))
			    : 1.0f;
			transformerScratch.layerDropKept[li] =
			    layerDropKeepThisLayer ? 1u : 0u;
		}

		if (!layerDropKeepThisLayer)
		{
			// Block dropped. Final residual stream for this layer equals hIn.
			// Copy hIn into this layer's hAfterFF slot so downstream layers
			// and the bwd pass see a populated buffer (= hIn).
			// Do NOT touch ffOut / attnOut / Q/K/V scratch — those stay
			// uninitialized; the bwd pass also takes the skip branch and
			// never reads them.
			float* hAfterFF_li = transformerScratch.hAfterFF.data()
			    + (static_cast<size_t>(li) * static_cast<size_t>(T)
			       * static_cast<size_t>(dModel));
			std::memcpy(hAfterFF_li, hIn,
			    sizeof(float) * static_cast<size_t>(T)
			    * static_cast<size_t>(dModel));
			continue;
		}
```

> **Note:** `transformerScratch.hAfterFF[li * T * dModel + ...]` is the per-layer output buffer used at line 8089-8090 of the existing loop:
> ```cpp
> hAfterFF[i] = hAfterAttn[i] + ffOut[i];
> ```
> The skip path memcpy makes `hAfterFF_li = hIn`, which is the identity-residual result (equivalent to `hAfterFF[li][i] = hIn[i] + 0 + 0`). The next layer iteration reads `hIn = transformerScratch.hAfterFF.data() + (li-1) * T * dModel` at line 7817 — which now correctly contains the pass-through value.

- [ ] **Step 3: Scale the attn residual add when layer is kept**

Find the attn residual add line (around 8000-8001 in the layer loop):

```cpp
		for (size_t i = 0; i < static_cast<size_t>(T) * static_cast<size_t>(dModel); ++i)
			hAfterAttn[i] = hIn[i] + attnOut[i];
```

Replace with:

```cpp
		for (size_t i = 0; i < static_cast<size_t>(T) * static_cast<size_t>(dModel); ++i)
			hAfterAttn[i] = hIn[i] + layerDropScale * attnOut[i];
```

When LayerDrop is disabled, `layerDropScale == 1.0f` → bit-identical.

- [ ] **Step 4: Scale the FFN residual add when layer is kept**

Find the FFN residual add at line ~8089-8090 (search for `hAfterFF[i] = hAfterAttn[i] + ffOut[i]`). The existing code:

```cpp
		float* hAfterFF = transformerScratch.hAfterFF.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
		for (size_t i = 0; i < static_cast<size_t>(T) * static_cast<size_t>(dModel); ++i)
			hAfterFF[i] = hAfterAttn[i] + ffOut[i];
```

Change the loop body to:

```cpp
		for (size_t i = 0; i < static_cast<size_t>(T) * static_cast<size_t>(dModel); ++i)
			hAfterFF[i] = hAfterAttn[i] + layerDropScale * ffOut[i];
```

When `layerDropScale == 1.0f` (default at `layerDropPMax == 0`) → bit-identical to the prior code.

- [ ] **Step 5: Rebuild library + unit tests**

```bash
cd ~/dev/glades-ml
sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests/build && sh .configure.sh cuda 2>&1 | tail -5
```

Expected: clean build. Any compile errors here usually mean the `hLayerOut` identifier was wrong — re-grep for the exact final-residual-add target name in the CPU loop.

- [ ] **Step 6: Run chiron tests to confirm bit-identicality at p_max=0**

```bash
cd ~/dev/glades-ml/unit-tests && bash test.sh chiron 2>&1 | tail -10
```

Expected: all pass. With `layerDropPMax == 0.0f` (default), the skip-branch is never entered (`layerDropKeepThisLayer == true`), `layerDropScale == 1.0f`, and the scaled residual adds are bit-identical to the prior code.

- [ ] **Step 7: Commit**

```bash
cd ~/dev/glades-ml
git add Backend/Machine\ Learning/Networks/sgd_transformer.cpp
git commit -m "$(cat <<'EOF'
Wire LayerDrop into CPU forward pass

Per-step per-layer Bernoulli mask drawn at the top of the layer loop
(uses the rngEngine determinism path). When mask=0, block is skipped
and the layer-output buffer is memcpy'd from hIn (downstream layer
sees identity residual update). When mask=1, both sub-residuals
(attn-residual and FFN-residual) are scaled by 1/(1-p_l) at the
respective residual-add steps.

Bit-identical at layerDropPMax == 0.0f (default): layerDropScale stays
1.0f, no skip branch taken, residual-add math reduces to baseline.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 2.3: Wire LayerDrop into CPU backward pass

**Files:**
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/sgd_transformer.cpp` (the CPU per-layer backward loop — search for `for (unsigned int li = ...; --li)` and lines ~9005 / ~9095 area)

- [ ] **Step 1: Confirm bwd-loop variable names**

The CPU bwd layer loop starts at line ~8987:
```cpp
for (int li = static_cast<int>(nLayers) - 1; li >= 0; --li)
```

The relevant local variables in the loop body (confirmed at lines 8990-9105):
- `dH` — running residual-stream gradient (a `std::vector<float, AlignedAllocator>`). At loop entry for iter `li`, dH = ∂L/∂hAfterFF[li]. After the iter completes, dH = ∂L/∂hIn[li] (= ∂L/∂hAfterFF[li-1] for the next iter).
- `dHAfterAttn` (= `transformerScratch.dH2`) — gradient w.r.t. `hAfterAttn[li]`. Copied from dH at line 9001 (identity path through FFN-residual).
- `dHAfterAttnFromLN` — LN2-bwd output gradient (added to dHAfterAttn at line 9088-9089).
- FFN bwd is invoked at line 9025: `linear_backward_accum_maybe_lowp(ff1Act, dH.data(), ..., b.gW2, b.gB2, ..., dFF1Act.data())` — consumes `dH` as ∂L/∂ffOut.
- Attn bwd (the corresponding ∂L/∂attnOut consumer) is invoked LATER in the loop body, after the resDropoutBwd at line 9094-9108. It consumes `dHAfterAttn` (post-resDropout) as ∂L/∂attnOut.

- [ ] **Step 2: Add bwd skip branch at the top of the bwd layer loop**

Insert at the start of the bwd loop body (immediately after the `hIn`-pointer setup at lines 8990-8991, before the `dHAfterAttn` setup at line 8999):

```cpp
		// === LayerDrop bwd skip branch ===
		// If this layer was dropped on fwd (transformerScratch.layerDropKept[li]
		// == 0), the block's bwd contributes zero to weight grads and the
		// residual-stream gradient `dH` flows directly to the previous layer.
		float layerDropPL_bwd = 0.0f;
		float layerDropScale_bwd = 1.0f;
		bool layerDropKeptThisLayer = true;
		if (trainingConfig.transformer.layerDropPMax > 0.0f &&
		    !transformerScratch.layerDropKept.empty())
		{
			layerDropKeptThisLayer =
			    transformerScratch.layerDropKept[static_cast<size_t>(li)] != 0u;
			layerDropPL_bwd = glades::transformer_kernels::layer_drop_p_l(
			    static_cast<unsigned int>(li), nLayers,
			    trainingConfig.transformer.layerDropPMax,
			    trainingConfig.transformer.layerDropLinearSchedule);
			layerDropScale_bwd =
			    (layerDropPL_bwd > 0.0f && layerDropPL_bwd < 1.0f)
			        ? (1.0f / (1.0f - layerDropPL_bwd))
			        : 1.0f;
		}

		if (!layerDropKeptThisLayer)
		{
			// Block was dropped on fwd. dH is already ∂L/∂hAfterFF[li] =
			// ∂L/∂hIn[li] (identity path). No weight-grad accumulation, no
			// LN1/LN2 bwd, no FFN bwd, no attn bwd. Skip to next iter.
			continue;
		}
```

- [ ] **Step 3: Scale `dH` by `layerDropScale_bwd` just before the FFN bwd at line 9025**

Math: with LayerDrop fwd `hAfterFF = hAfterAttn + s · ffOut`, the bwd is
`∂L/∂ffOut = s · ∂L/∂hAfterFF = s · dH`. So FFN bwd must receive `s · dH`, not `dH`.

The cleanest place is right before line 9025. After line 9001 (the `std::copy(dH.begin(), dH.end(), dHAfterAttn.begin())` that captures the identity-path gradient) and BEFORE line 9025 (the FFN bwd call), insert:

```cpp
		// LayerDrop bwd: scale dH by layerDropScale_bwd before the FFN bwd
		// consumes it as ∂L/∂ffOut. dHAfterAttn already captured the
		// unscaled identity-path gradient at line 9001, so this does not
		// double-scale.
		if (layerDropScale_bwd != 1.0f)
		{
			for (size_t i = 0; i < dH.size(); ++i)
				dH[i] *= layerDropScale_bwd;
		}
```

When `layerDropScale_bwd == 1.0f`, the `if` skips → bit-identical to baseline.

- [ ] **Step 4: Scale `dHAfterAttn` by `layerDropScale_bwd` just before the attn bwd consumer**

Locate where `dHAfterAttn` (or `dH` after line 9091's `std::copy(dHAfterAttn.begin(), dHAfterAttn.end(), dH.begin())`) is consumed by the attn-bwd call. Search:

```bash
grep -nE "linear_backward.*dHAfterAttn\|attn.*backward\|attention_backward\|.*Wo.*backward" "/home/robert/dev/glades-ml/Backend/Machine Learning/Networks/sgd_transformer.cpp" | awk -F: '$2 > 9108 && $2 < 9300' | head -10
```

Find the line where the gradient enters the attn block bwd (typically a `linear_backward_accum_maybe_lowp(..., dHAfterAttn.data(), ...)` for the Wo backward, or a `gpu::...` dispatch on the GPU path). Immediately before that call, insert:

```cpp
		// LayerDrop bwd: scale dHAfterAttn by layerDropScale_bwd before the
		// attn bwd consumes it as ∂L/∂attnOut. (dH at this point already
		// equals dHAfterAttn via the std::copy at line ~9091; scale dHAfterAttn
		// here so the resDropoutBwd at line ~9094-9108 sees the unscaled
		// value first, then this scale, then the attn-bwd consumer.)
		if (layerDropScale_bwd != 1.0f)
		{
			for (size_t i = 0; i < dHAfterAttn.size(); ++i)
				dHAfterAttn[i] *= layerDropScale_bwd;
		}
```

> **Note on ordering vs the attn-residual-dropout bwd at line 9094-9108:** the residual-dropout bwd modifies `dHAfterAttn` in place (multiplies by `mask · scale`). If LayerDrop is composed with residual dropout (not the production recipe — residualDropoutRate is 0 in the regstack ship), the two scalings compose as `dHAfterAttn *= layerDropScale_bwd · resDropoutScale · mask`. For the LayerDrop arc, residualDropoutRate stays 0, so this composition is trivially correct.

When `layerDropScale_bwd == 1.0f` → bit-identical.

- [ ] **Step 5: Rebuild + run chiron tests**

```bash
cd ~/dev/glades-ml
sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests/build && sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests && bash test.sh chiron 2>&1 | tail -10
```

Expected: all pass. Bit-identicality at `layerDropPMax == 0` confirmed by existing tests (no regression). True bit-identical behavior under LayerDrop active is tested next.

- [ ] **Step 6: Commit**

```bash
cd ~/dev/glades-ml
git add Backend/Machine\ Learning/Networks/sgd_transformer.cpp
git commit -m "$(cat <<'EOF'
Wire LayerDrop into CPU backward pass

Bwd reads transformerScratch.layerDropKept[li] (persisted by fwd) and
short-circuits the block's bwd when mask=0 (no weight-grad
accumulation, residual-stream grad flows through unchanged).
When mask=1, both gradAttnOut and gradFFOut are pre-multiplied by
1/(1-p_l) before the attn-bwd and FFN-bwd calls, propagating the fwd
scaling correctly via chain rule.

Bit-identical at layerDropPMax == 0.0f (default): no skip taken,
layerDropScale_bwd stays 1.0f, the pre-multiply reduces to a copy.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 2.4: Implement CHIRONLayerDropDisabledParityTest (helper-level no-op guard)

**Files:**
- Modify: `~/dev/glades-ml/unit-tests/Backend/Machine Learning/chiron-test.cpp`

The regstack `CHIRONZlossDisabledParityTest` is a math-helper-level test (verifies `softmax_ce_with_zloss(..., coef=0)` returns bit-identical results to the reference computation, NOT a full-trainer test). We mirror that pattern: verify `layer_drop_keep(eng, 0.0f)` always returns true and does NOT advance the RNG state (so a `layerDropPMax = 0` run is byte-identical to a "no LayerDrop code" run at the RNG-trace level).

System-level bit-identicality at `layerDropPMax = 0` is guarded implicitly by the existing CHIRON tests at default config (they all use `layerDropPMax = 0.0f` and exercise the full CPU + GPU paths; any LayerDrop-induced regression would surface there).

- [ ] **Step 1: Replace the stub body with the helper-level parity test**

Locate the `CHIRONLayerDropDisabledParityTest()` stub added in Task 1.1 Step 5. Replace its body with:

```cpp
void CHIRONLayerDropDisabledParityTest()
{
	using glades::transformer_kernels::layer_drop_keep;
	using glades::transformer_kernels::layer_drop_p_l;

	// At p_l = 0, layer_drop_keep MUST return true unconditionally and MUST
	// NOT advance the RNG state (the helper takes the early-return path).
	std::mt19937 eng_a(7777u);
	std::mt19937 eng_b(7777u);

	// On engine A: 24 calls to layer_drop_keep at p_l = 0 (mimicking a full
	// L=24 layer pass at p_max = 0).
	for (unsigned int li = 0; li < 24u; ++li)
	{
		const float p_l = layer_drop_p_l(li, 24u, 0.0f, true);
		ASSERT("CHIRONLayerDropDisabledParity: p_l must be 0 at p_max=0",
		       p_l == 0.0f);
		const bool keep = layer_drop_keep(eng_a, p_l);
		ASSERT("CHIRONLayerDropDisabledParity: layer_drop_keep must return true at p_l=0",
		       keep == true);
	}

	// Engine A and engine B started at the same seed and engine B has NOT
	// been called yet. After A's 24 no-op calls, both engines must produce
	// the same next 32-bit draw — i.e., the no-op calls did not perturb the
	// RNG state.
	const std::mt19937::result_type next_a = eng_a();
	const std::mt19937::result_type next_b = eng_b();
	ASSERT("CHIRONLayerDropDisabledParity: layer_drop_keep at p_l=0 must NOT advance RNG state",
	       next_a == next_b);

	// Also: at p_l = 1.0 (drop always), layer_drop_keep must return false
	// and MUST NOT consume RNG state (early-return path).
	std::mt19937 eng_c(8888u);
	std::mt19937 eng_d(8888u);
	for (int i = 0; i < 10; ++i)
	{
		const bool keep = layer_drop_keep(eng_c, 1.0f);
		ASSERT("CHIRONLayerDropDisabledParity: layer_drop_keep must return false at p_l=1.0",
		       keep == false);
	}
	const std::mt19937::result_type next_c = eng_c();
	const std::mt19937::result_type next_d = eng_d();
	ASSERT("CHIRONLayerDropDisabledParity: layer_drop_keep at p_l=1.0 must NOT advance RNG state",
	       next_c == next_d);
}
```

> **Note on RNG engine type:** the test above uses `std::mt19937` for the unit-level check, because `layer_drop_keep` is templated on `EngineT` and works with any `<random>` engine. The project's per-network `rngEngine` in `sgd_transformer.cpp` may or may not be exactly `std::mt19937` — but that doesn't affect this helper's no-state-advance behavior, which is what we're testing. If the project's engine type cannot be default-constructed with a seed (e.g., it's a custom wrapper), substitute the unit-test invocation with the project's engine type accordingly. Find the type with:
> ```bash
> grep -n "rngEngine\|RngEngine\|RNG.*engine" "/home/robert/dev/glades-ml/Backend/Machine Learning/Networks/sgd_transformer.cpp" | head -5
> ```

- [ ] **Step 2: Add `<random>` include if not already present**

Open `unit-tests/Backend/Machine Learning/chiron-test.cpp` and ensure `#include <random>` is at the top. If absent, add it among the existing standard-library includes.

- [ ] **Step 3: Rebuild + run the parity test**

```bash
cd ~/dev/glades-ml/unit-tests/build && sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests && bash test.sh chiron 2>&1 | tail -10
```

Expected: `CHIRONLayerDropDisabledParityTest` passes; full chiron suite still passes (the existing tests indirectly verify system-level bit-identicality at the default `layerDropPMax = 0`).

- [ ] **Step 4: Commit**

```bash
cd ~/dev/glades-ml
git add unit-tests/Backend/Machine\ Learning/chiron-test.cpp
git commit -m "$(cat <<'EOF'
Add CHIRONLayerDropDisabledParityTest (helper-level)

Verifies that layer_drop_keep takes the early-return path at p_l=0
(returns true, no RNG state advance) and at p_l=1.0 (returns false, no
RNG state advance). This guards the bit-identicality at layerDropPMax=0
at the helper level. System-level bit-identicality at layerDropPMax=0
is implicitly tested by the rest of the chiron suite running at default
config.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3: LayerDrop GPU forward + backward wiring

### Task 3.1: Wire LayerDrop into GPU forward dispatch (primary + variant paths)

**Files:**
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/sgd_transformer.cpp` (THREE separate GPU fwd loops — primary, activation-checkpoint re-fwd, and a third variant)

- [ ] **Step 1: Enumerate all GPU residual-add (gpu::add_two) call sites**

```bash
grep -n "gpu::add_two" "/home/robert/dev/glades-ml/Backend/Machine Learning/Networks/sgd_transformer.cpp"
```

Expected sites (3 fwd loops × 2 residual-adds each = 6 sites):
- Lines 9894 (attn-res) + 10023 (FFN-res): PRIMARY GPU fwd loop (training fwd, line 9639 start).
- Lines 10291 (attn-res) + 10405 (FFN-res): activation-checkpoint re-fwd loop (re-runs fwd to recompute saved activations for bwd).
- Lines 10981 (attn-res) + 11038 (FFN-res): a third fwd variant (probably mixed-precision or another fwd path).

Find the start of each loop by scanning upward for `for (unsigned int li`:
```bash
grep -nE "for \(unsigned int li = 0u?; li < nLayers" "/home/robert/dev/glades-ml/Backend/Machine Learning/Networks/sgd_transformer.cpp" | head -10
```

Confirm there are at least 3 GPU fwd loops to be wired. ALL must be wired or the activation-checkpoint re-fwd will produce a different layer skip pattern than the primary fwd, breaking the bwd.

- [ ] **Step 2: Wire the PRIMARY GPU fwd loop (line 9639)**

At the start of the primary fwd loop body (immediately after the `layerIn` pointer setup at lines 9651-9654, before the LN1 fwd at line 9661), insert:

```cpp
		// === LayerDrop mask draw (GPU primary fwd path) ===
		// Draw mask, persist to transformerScratch.layerDropKept[li]. The
		// activation-checkpoint re-fwd loop and any other fwd variants
		// MUST read this persisted mask (do NOT redraw); see Task 3.1
		// Step 3.
		float layerDropPL = 0.0f;
		float layerDropScale = 1.0f;
		bool layerDropKeepThisLayer = true;
		if (trainingConfig.transformer.layerDropPMax > 0.0f &&
		    !transformerScratch.layerDropKept.empty())
		{
			layerDropPL = glades::transformer_kernels::layer_drop_p_l(
			    li, nLayers, trainingConfig.transformer.layerDropPMax,
			    trainingConfig.transformer.layerDropLinearSchedule);
			layerDropKeepThisLayer = glades::transformer_kernels::layer_drop_keep(
			    rngEngine, layerDropPL);
			layerDropScale = (layerDropPL > 0.0f && layerDropPL < 1.0f)
			    ? (1.0f / (1.0f - layerDropPL))
			    : 1.0f;
			transformerScratch.layerDropKept[li] =
			    layerDropKeepThisLayer ? 1u : 0u;
		}

		if (!layerDropKeepThisLayer)
		{
			// Block dropped. Copy device-side layerIn -> hAfterFF_l (the
			// per-layer output slot) and skip the rest of this layer.
			float* hAfterFF_l_skip = gpuTransformerScratch->hAfterFF.data()
			    + slot * static_cast<size_t>(T) * static_cast<size_t>(dModel);
			glades::gpu::device_memcpy_d2d(hAfterFF_l_skip, layerIn,
			    static_cast<size_t>(T) * static_cast<size_t>(dModel)
			        * sizeof(float));
			continue;
		}
```

Then scale the two `gpu::add_two` calls. The existing primary-loop sites at line 9894 and 10023:

```cpp
		// Existing line 9894:
		gpu::add_two(hAfterAttn_l, layerIn, attnOut_l,
		             static_cast<int>(T * dModel));
```

`gpu::add_two(out, a, b, n)` computes `out = a + b` (no alpha scaling). LayerDrop needs `out = a + scale * b`. Two options:
(a) Add a new `gpu::add_two_scaled(out, a, b, beta, n)` kernel that computes `out = a + beta * b`.
(b) Replace `gpu::add_two` with `gpu::axpy_kernel` (if available — search `gpu_kernels.cu` for `axpy`) which already takes an alpha.
(c) Compute in-place: `gpu::scale_kernel(layerDropScale, b, b, n)` then `gpu::add_two(out, a, b, n)`.

Recommended: option (a) — add a `gpu::add_two_scaled` kernel (a 5-line CUDA edit). The existing `gpu::add_two` becomes a wrapper that calls `add_two_scaled(out, a, b, 1.0f, n)`. Bit-identical at `beta == 1.0f`.

After adding `add_two_scaled`, replace the two primary-loop residual-add lines:

```cpp
		// Line 9894 (attn-residual): hAfterAttn = layerIn + layerDropScale * attnOut
		glades::gpu::add_two_scaled(hAfterAttn_l, layerIn, attnOut_l,
		    layerDropScale, static_cast<int>(T * dModel));

		// Line 10023 (FFN-residual): hAfterFF = hAfterAttn + layerDropScale * ffOut
		glades::gpu::add_two_scaled(hAfterFF_l, hAfterAttn_l, ffOut_l,
		    layerDropScale, static_cast<int>(T * dModel));
```

Add the `add_two_scaled` kernel to `Backend/Machine Learning/Networks/cuda/gpu_kernels.cu` (or wherever `add_two` lives — grep for it):

```cpp
// add_two_scaled: out[i] = a[i] + beta * b[i]
__global__ void add_two_scaled_kernel(float* __restrict__ out,
                                      const float* __restrict__ a,
                                      const float* __restrict__ b,
                                      float beta, int n)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n)
		out[idx] = a[idx] + beta * b[idx];
}

namespace glades { namespace gpu {
void add_two_scaled(float* out, const float* a, const float* b,
                    float beta, int n)
{
	const int block = 256;
	const int grid = (n + block - 1) / block;
	add_two_scaled_kernel<<<grid, block, 0, computeStream()>>>(
	    out, a, b, beta, n);
}
}}
```

(Declare `void add_two_scaled(...)` in the corresponding header file, `gpu_chiron.h` or `gpu_kernels.h` — wherever `add_two` is declared.)

- [ ] **Step 3: Wire the activation-checkpoint re-fwd loop (~line 10086 / sites 10291 + 10405)**

This loop re-runs the per-layer fwd body to recompute activations for the bwd. It MUST use the same mask values as the primary fwd, NOT redraw. At the start of the re-fwd loop body, mirror Step 2 but READ the persisted mask:

```cpp
		// === LayerDrop mask read (GPU re-fwd path — use persisted, do NOT redraw) ===
		float layerDropPL = 0.0f;
		float layerDropScale = 1.0f;
		bool layerDropKeepThisLayer = true;
		if (trainingConfig.transformer.layerDropPMax > 0.0f &&
		    !transformerScratch.layerDropKept.empty())
		{
			layerDropKeepThisLayer =
			    transformerScratch.layerDropKept[li] != 0u;
			layerDropPL = glades::transformer_kernels::layer_drop_p_l(
			    li, nLayers, trainingConfig.transformer.layerDropPMax,
			    trainingConfig.transformer.layerDropLinearSchedule);
			layerDropScale = (layerDropPL > 0.0f && layerDropPL < 1.0f)
			    ? (1.0f / (1.0f - layerDropPL))
			    : 1.0f;
		}

		if (!layerDropKeepThisLayer)
		{
			float* hAfterFF_l_skip = gpuTransformerScratch->hAfterFF.data()
			    + slot * static_cast<size_t>(T) * static_cast<size_t>(dModel);
			glades::gpu::device_memcpy_d2d(hAfterFF_l_skip, layerIn,
			    static_cast<size_t>(T) * static_cast<size_t>(dModel)
			        * sizeof(float));
			continue;
		}
```

Then replace the two `gpu::add_two` calls in this loop (lines 10291 and 10405) with `gpu::add_two_scaled(..., layerDropScale, ...)` exactly as in Step 2.

- [ ] **Step 4: Wire the third GPU fwd variant (sites 10981 + 11038)**

Locate the third fwd loop by scanning upward from line 10981 for `for (unsigned int li`. This is likely a BF16-mixed-precision or alternative fwd path. Apply the SAME pattern as Step 3 (read persisted mask, skip-path memcpy, `add_two_scaled` at the two residual-add sites). DO NOT redraw the mask here either — only the primary fwd loop draws.

- [ ] **Step 5: Rebuild + run chiron tests (GPU build)**

```bash
cd ~/dev/glades-ml
sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests/build && sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests && bash test.sh chiron 2>&1 | tail -10
```

Expected: all pass. At `layerDropPMax == 0`, `layerDropScale == 1.0f` → `add_two_scaled(..., 1.0f, ...)` is bit-identical to `add_two`; no skip branch taken; nothing changes.

- [ ] **Step 6: Commit**

```bash
cd ~/dev/glades-ml
git add Backend/Machine\ Learning/Networks/sgd_transformer.cpp \
        Backend/Machine\ Learning/Networks/cuda/gpu_kernels.cu \
        Backend/Machine\ Learning/Networks/cuda/gpu_chiron.h
# (Add whichever header actually declares add_two_scaled — gpu_chiron.h or gpu_kernels.h.)
git commit -m "$(cat <<'EOF'
Wire LayerDrop into GPU forward dispatch (3 fwd paths)

Adds gpu::add_two_scaled (out = a + beta * b) kernel and wires it into
all three GPU fwd loops in sgd_transformer.cpp:
  - Primary fwd (line ~9639): draws mask, persists to layerDropKept.
  - Activation-checkpoint re-fwd (~10086): reads persisted mask
    (must match primary or bwd activations desync).
  - Third fwd variant (~10979): reads persisted mask.

Each loop's two residual-adds (attn + FFN) become add_two_scaled with
beta=layerDropScale. Skip-path d2d-memcpy when mask=0.

Bit-identical at layerDropPMax == 0.0f: add_two_scaled(..., 1.0f, ...)
matches add_two; no skip branch; residual-add math identical to
baseline. The existing add_two API stays as a wrapper for callers
outside this loop that don't need scaling.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 3.2: Wire LayerDrop into GPU backward dispatch

**Files:**
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/sgd_transformer.cpp` (the GPU per-layer backward loop)

- [ ] **Step 1: Locate the GPU bwd layer loop**

```bash
grep -nE "for.*li.*=.*nLayers.*--|for.*int li.*nLayers" "/home/robert/dev/glades-ml/Backend/Machine Learning/Networks/sgd_transformer.cpp" | tail -10
```

The bwd loop pattern is `for (int li = nLayers - 1; li >= 0; --li)` or similar. Confirm the line number. The GPU bwd grad-residual-add sites mirror the GPU fwd add sites (Task 3.1 Steps 3-4).

- [ ] **Step 2: Add bwd skip branch at top of GPU bwd loop**

Insert at the start of the GPU bwd loop body, mirroring Task 2.3 Step 2:

```cpp
		// === LayerDrop bwd skip branch (GPU bwd path) ===
		float layerDropPL_bwd = 0.0f;
		float layerDropScale_bwd = 1.0f;
		bool layerDropKeptThisLayer = true;
		if (trainingConfig.transformer.layerDropPMax > 0.0f &&
		    !transformerScratch.layerDropKept.empty())
		{
			layerDropKeptThisLayer =
			    transformerScratch.layerDropKept[li] != 0u;
			layerDropPL_bwd = glades::transformer_kernels::layer_drop_p_l(
			    li, nLayers, trainingConfig.transformer.layerDropPMax,
			    trainingConfig.transformer.layerDropLinearSchedule);
			layerDropScale_bwd =
			    (layerDropPL_bwd > 0.0f && layerDropPL_bwd < 1.0f)
			        ? (1.0f / (1.0f - layerDropPL_bwd))
			        : 1.0f;
		}

		if (!layerDropKeptThisLayer)
		{
			// Block was dropped on GPU fwd. The bwd's incoming gradient
			// (`gradH_d` or equivalent device buffer) flows directly to the
			// previous layer's bwd input without modification. Skip all
			// attn/FFN bwd kernels.
			continue;
		}
```

- [ ] **Step 3: Scale `gradAttnOut_d` and `gradFFOut_d` by `layerDropScale_bwd` on GPU**

Find the two GPU bwd grad-input scaling sites (the device-side equivalent of Task 2.3 Steps 3-4). They are typically `gpu::copy_d2d` or `gpu::scale_kernel` calls right before the attn-bwd / ffn-bwd kernels. Change the scale argument from implicit-1 to `layerDropScale_bwd`:

```cpp
		// Before (attn-bwd input):
		gpu::device_memcpy_d2d(gradAttnOut_d, gradHAfterAttn_d, T*dModel*sizeof(float));

		// After:
		gpu::scale_kernel(layerDropScale_bwd, gradHAfterAttn_d, gradAttnOut_d, T*dModel, stream);
```

(If `scale_kernel` doesn't exist by that name, look for `axpby_kernel` with beta=0 — that's the same operation. Or add a new tiny `gpu::scale_kernel` helper alongside the existing element-wise kernels in `gpu_kernels.cu` if needed.)

> **Audit note:** if `layerDropScale_bwd == 1.0f`, the operation reduces to a d2d copy → bit-identical to the prior path. So at `layerDropPMax == 0`, no behavior change.

Mirror the same change for the FFN-bwd grad-input scaling site.

- [ ] **Step 4: Rebuild + run chiron tests**

```bash
cd ~/dev/glades-ml
sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests/build && sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests && bash test.sh chiron 2>&1 | tail -10
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
cd ~/dev/glades-ml
git add Backend/Machine\ Learning/Networks/sgd_transformer.cpp
git commit -m "$(cat <<'EOF'
Wire LayerDrop into GPU backward dispatch

Mirrors the CPU bwd wiring: per-layer mask read from
transformerScratch.layerDropKept[li] (persisted by fwd), short-circuit
when mask=0 (skip all GPU attn/FFN bwd kernels, residual-stream grad
flows through unchanged), and pre-scale gradAttnOut_d / gradFFOut_d
by 1/(1-p_l) on the GPU when mask=1.

Bit-identical at layerDropPMax == 0.0f (default): layerDropScale_bwd
stays 1.0f, no skip branch, pre-scale reduces to d2d copy.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 3.3: Implement CHIRONLayerDropDeterministicMasksTest

**Files:**
- Modify: `~/dev/glades-ml/unit-tests/Backend/Machine Learning/chiron-test.cpp`

- [ ] **Step 1: Replace the stub body with the determinism test**

Locate the `CHIRONLayerDropDeterministicMasksTest()` stub added in Task 1.1 Step 5. Replace its body:

```cpp
void CHIRONLayerDropDeterministicMasksTest()
{
	// Two engine instantiations with the same seed must produce the same
	// 100-step × 24-layer mask trace at layerDropPMax = 0.1, linear schedule.
	const unsigned int L = 24;
	const unsigned int steps = 100;
	const float pMax = 0.1f;
	const unsigned int seed = 1337u;

	std::vector<unsigned char> trace_a(steps * L, 0u);
	std::vector<unsigned char> trace_b(steps * L, 0u);

	for (int run = 0; run < 2; ++run)
	{
		std::vector<unsigned char>& trace = (run == 0) ? trace_a : trace_b;

		// Construct a fresh RNG engine at the same seed. Use whichever
		// engine type the project's per-network RNG uses
		// (per DETERMINISM_AND_CONCURRENCY.md).
		//
		// IMPLEMENTATION NOTE: find the engine type by searching for the
		// rngEngine declaration in sgd_transformer.cpp. Mirror its
		// construction here.
		std::mt19937 eng(seed);   // replace if project uses a different engine

		for (unsigned int s = 0; s < steps; ++s)
		{
			for (unsigned int li = 0; li < L; ++li)
			{
				const float p_l = glades::transformer_kernels::layer_drop_p_l(
				    li, L, pMax, true);
				const bool keep = glades::transformer_kernels::layer_drop_keep(
				    eng, p_l);
				trace[s * L + li] = keep ? 1u : 0u;
			}
		}
	}

	// Traces must be byte-identical.
	bool allEqual = true;
	for (size_t i = 0; i < trace_a.size(); ++i)
	{
		if (trace_a[i] != trace_b[i])
		{
			allEqual = false;
			break;
		}
	}
	ASSERT("CHIRONLayerDropDeterministicMasks: same seed must give identical 100-step trace",
	       allEqual);

	// Also verify the mean keep-rate matches the expected (sum p_l / L) = 0.05 → mean keep = 0.95.
	size_t keepCount = 0;
	for (size_t i = 0; i < trace_a.size(); ++i)
		if (trace_a[i] != 0u)
			++keepCount;
	const float keepRate = static_cast<float>(keepCount) / static_cast<float>(trace_a.size());
	// At pMax=0.1, mean p_l = 0.05, expected keep rate = 0.95. 100*24 = 2400 draws,
	// standard deviation ≈ sqrt(0.95*0.05/2400) ≈ 0.0045. Allow ±0.02 tolerance (~4σ).
	ASSERT("CHIRONLayerDropDeterministicMasks: keep rate within tolerance of 0.95",
	       fabsf(keepRate - 0.95f) < 0.02f);
}
```

> **Implementation note:** find and substitute the project's RNG engine type. If the project's `rngEngine` is `std::mt19937`, the test stands. If it's a custom engine type, mirror its constructor. Search:
> ```bash
> grep -n "rngEngine\|RngEngine\|engine.*seed" "/home/robert/dev/glades-ml/Backend/Machine Learning/Networks/sgd_transformer.cpp" | head -10
> ```

- [ ] **Step 2: Rebuild + run the test**

```bash
cd ~/dev/glades-ml/unit-tests/build && sh .configure.sh cuda 2>&1 | tail -5
cd ~/dev/glades-ml/unit-tests && bash test.sh chiron 2>&1 | tail -10
```

Expected: `CHIRONLayerDropDeterministicMasksTest` passes.

- [ ] **Step 3: Commit**

```bash
cd ~/dev/glades-ml
git add unit-tests/Backend/Machine\ Learning/chiron-test.cpp
git commit -m "$(cat <<'EOF'
Add CHIRONLayerDropDeterministicMasksTest

Two RNG-engine instantiations at seed=1337 must produce identical
100-step × 24-layer mask traces at layerDropPMax=0.1 linear schedule.
Also verifies the empirical keep rate (0.95 ± 0.02 over 2400 draws)
matches the expected value derived from the linear schedule.

Guards R-LD-5 (determinism break in mask draws).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 4: Trainer CLI + run.sh wiring

### Task 4.1: Add `--layer-drop-pmax` flag in glades-trainer

**Files:**
- Modify: `~/dev/glades-trainer/chiron_main.cpp` (or equivalent — search for `--zloss-coef` to locate the flag-parsing block)
- Modify: `~/dev/glades-trainer/run.sh`

- [ ] **Step 1: Find the existing `--zloss-coef` flag-parsing block**

```bash
grep -rn "zloss-coef\|qk-norm\|mtp-depth\|TransformerRunConfig" ~/dev/glades-trainer/ 2>/dev/null | head -20
```

This locates the file(s) where the regstack flags are parsed and the `TransformerRunConfig` is populated. Likely `chiron_main.cpp`.

- [ ] **Step 2: Add `--layer-drop-pmax` flag parser**

In the chiron_main.cpp flag-parsing block (next to `--zloss-coef` and `--qk-norm`), add:

```cpp
		else if (strcmp(argv[i], "--layer-drop-pmax") == 0 && i + 1 < argc)
		{
			runtimeCfg.layerDropPMax = atof(argv[++i]);
		}
```

- [ ] **Step 3: Add `--layer-drop-pmax` to the usage block / help**

In `run.sh`, find the existing regstack flags help block (`Regularization stack flags (regstack, 2026-05-22 — all default to disabled):`) and add a new line:

```bash
    echo "  --layer-drop-pmax F    LayerDrop p_max (stochastic depth, Fan 2019). 0=off. Rec: 0.1."
```

If the trainer's CLI uses `--layer-drop-pmax F` as a pass-through, also ensure it is forwarded to the binary (search `run.sh` for how `--zloss-coef` is forwarded — typically it's just appended to the binary invocation).

- [ ] **Step 4: Rebuild the trainer**

```bash
cd ~/dev/glades-trainer && sh build.sh 2>&1 | tail -10
```

Expected: clean build.

- [ ] **Step 5: Smoke-test the flag (1-step dry run)**

```bash
cd ~/dev/glades-trainer
sh run.sh flagship --zloss-coef 1e-4 --qk-norm --layer-drop-pmax 0.1 \
    --steps 1 --seed 1337 --val-every 1 \
    --save /tmp/chiron_1B_layerdrop_smoke 2>&1 | tail -20
```

Expected: the trainer accepts the flag, runs one step, logs `layerDropPMax=0.1` (or similar config-dump line), exits cleanly. Loss value is logged; do not regress against a corresponding `--layer-drop-pmax 0.0` smoke run (both should be valid 1-step results).

- [ ] **Step 6: Commit**

```bash
cd ~/dev/glades-trainer
git add chiron_main.cpp run.sh   # adjust to actual files modified
git commit -m "$(cat <<'EOF'
Add --layer-drop-pmax flag to chiron_main + run.sh help

Wires the new TransformerRunConfig.layerDropPMax library field into
the trainer CLI alongside --zloss-coef and --qk-norm. Default 0.0f
(no-op, bit-identical to regstack Phase 2 ship). Recommended 0.1
per the LayerDrop spec.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 5: Validation arc (per spec §3)

### Task 5.1: Pilot baseline L0 (regstack ship, 5k single-seed)

**Files:** No code changes; training run + result logging only.

- [ ] **Step 1: Pre-flight check on existing regstack ship**

```bash
ls -la ~/dev/glades-trainer/database/checkpoints/chiron_1B_T16384_regstack_phase2/ 2>/dev/null | head -3
```

Expected: `chiron_1B_T16384_regstack_phase2.final` exists (the current ship). If not, the regstack Phase 2 ship has been moved — locate it and update the runner.sh path or the spec's baseline reference before continuing.

- [ ] **Step 2: Launch L0 5k pilot (regstack ship recipe, 5k steps, seed=1337)**

```bash
cd ~/dev/glades-trainer
sh run.sh flagship --zloss-coef 1e-4 --qk-norm \
    --steps 5000 --seed 1337 --val-every 500 \
    --save database/checkpoints/chiron_1B_T16384_l0_5k 2>&1 \
    | tee logs/l0_5k_$(date +%Y%m%d_%H%M).log
```

Expected: ~47 min wall. The final-step log line prints val NLL @ step 5000 and tok/s.

- [ ] **Step 3: Extract L0 5k metrics**

```bash
grep -E "step 5000|val NLL|tok/s|peak VRAM" logs/l0_5k_*.log | tail -20
```

Record L0 numbers:
- `L0_NLL_5k` (val NLL @ step 5000)
- `L0_TOKS` (final tok/s)
- `L0_VRAM` (peak VRAM)

- [ ] **Step 4: Sanity-check vs published regstack Phase 2 5k trajectory**

The regstack Phase 2 ship doc (`research/REGSTACK_PHASE2_2026_05_22.md`) records the 5k trajectory. Confirm `L0_NLL_5k` is within ±0.05 nat of the published value at step 5000.

If the deviation is >0.05 nat, F-LD-1 fires per spec §3.2. Stop. Investigate:
- Has any other code change landed since the regstack Phase 2 ship that affects the regstack stack? (Check `git log` for `chiron2`-branch commits since `c59ed13be Ship CHIRON 1B regstack Phase 2`.)
- Is the trainer build current? Re-run `sh build.sh` in `~/dev/glades-trainer/`.

Do not proceed to L1 until L0 reproduces.

### Task 5.2: Pilot L1 (LayerDrop p_max=0.1, 5k single-seed)

- [ ] **Step 1: Launch L1 5k pilot**

```bash
cd ~/dev/glades-trainer
sh run.sh flagship --zloss-coef 1e-4 --qk-norm --layer-drop-pmax 0.1 \
    --steps 5000 --seed 1337 --val-every 500 \
    --save database/checkpoints/chiron_1B_T16384_l1_layerdrop_5k 2>&1 \
    | tee logs/l1_5k_$(date +%Y%m%d_%H%M).log
```

Expected: ~44 min wall (expected -6% improvement vs L0). The final-step log line prints val NLL, tok/s, peak VRAM.

- [ ] **Step 2: Extract L1 5k metrics**

```bash
grep -E "step 5000|val NLL|tok/s|peak VRAM" logs/l1_5k_*.log | tail -20
```

Record:
- `L1_NLL_5k`
- `L1_TOKS`
- `L1_VRAM`

- [ ] **Step 3: Apply pilot gate per spec §3.2**

Compute `ΔNLL = L1_NLL_5k - L0_NLL_5k` and `Δtoks_pct = (L1_TOKS - L0_TOKS) / L0_TOKS * 100`.

```
if ΔNLL > 0.05:
    OUTRIGHT FAIL → publish honest negative result; close arc; do NOT run L2.
elif 0.02 < ΔNLL ≤ 0.05:
    BORDERLINE → run seed=1338 pilot before deciding (Task 5.3).
elif Δtoks_pct < -5.0:
    WALL REGRESSION → flag for investigation (LayerDrop should improve wall);
    proceed to L2 only if NLL gate is solidly clean (ΔNLL ≤ -0.03 nat).
elif ΔNLL ≤ 0.02 AND Δtoks_pct ≥ -5.0:
    PASS → proceed to L2 (Task 5.4).
```

- [ ] **Step 4: Record pilot decision**

Append a short note to `research/REGSTACK_LAYERDROP_PHASE2_DRAFT.md` (create the file) with:
- `L0_NLL_5k`, `L1_NLL_5k`, ΔNLL, decision.
- `L0_TOKS`, `L1_TOKS`, Δtoks_pct.
- Peak VRAM both runs.
- The exact run commands (for reproducibility) and timestamps.

### Task 5.3: (Optional) seed=1338 borderline-disambiguation pilot

**Run only if Task 5.2 Step 3 marked the result BORDERLINE.**

- [ ] **Step 1: Launch L1 5k pilot at seed=1338**

```bash
cd ~/dev/glades-trainer
sh run.sh flagship --zloss-coef 1e-4 --qk-norm --layer-drop-pmax 0.1 \
    --steps 5000 --seed 1338 --val-every 500 \
    --save database/checkpoints/chiron_1B_T16384_l1_layerdrop_5k_s1338 2>&1 \
    | tee logs/l1_5k_s1338_$(date +%Y%m%d_%H%M).log
```

- [ ] **Step 2: Also run L0 at seed=1338 (apples-to-apples baseline rerun)**

```bash
cd ~/dev/glades-trainer
sh run.sh flagship --zloss-coef 1e-4 --qk-norm \
    --steps 5000 --seed 1338 --val-every 500 \
    --save database/checkpoints/chiron_1B_T16384_l0_5k_s1338 2>&1 \
    | tee logs/l0_5k_s1338_$(date +%Y%m%d_%H%M).log
```

- [ ] **Step 3: Compute n=2 mean ΔNLL**

```
ΔNLL_mean = ((L1_NLL_s1337 - L0_NLL_s1337) + (L1_NLL_s1338 - L0_NLL_s1338)) / 2
```

Decision:
- If `ΔNLL_mean ≤ 0.02 nat` AND `|ΔNLL_s1337 - ΔNLL_s1338| ≤ 0.05 nat` (low spread) → PASS → proceed to L2.
- If `ΔNLL_mean > 0.02 nat` → FAIL → publish honest result; close arc.
- If high spread (`> 0.05 nat between seeds`) → FAIL (instability); close arc.

### Task 5.4: 30k Phase-2 retrain L2 (gated on pilot PASS)

- [ ] **Step 1: Launch L2 30k Phase-2 retrain**

```bash
cd ~/dev/glades-trainer
sh run.sh flagship --zloss-coef 1e-4 --qk-norm --layer-drop-pmax 0.1 \
    --steps 30000 --seed 1337 --val-every 1500 \
    --save database/checkpoints/chiron_1B_T16384_regstack_layerdrop_phase2 2>&1 \
    | tee logs/l2_30k_$(date +%Y%m%d_%H%M).log
```

Expected: ~4.5 hours wall.

- [ ] **Step 2: Extract L2 30k metrics**

```bash
grep -E "step 30000|val NLL|tok/s|peak VRAM" logs/l2_30k_*.log | tail -30
```

Record:
- `L2_NLL_30k`
- `L2_TOKS_final`
- `L2_VRAM_peak`
- Full trajectory: val NLL at each val checkpoint (steps 1500, 3000, ..., 30000) → 20 data points.

- [ ] **Step 3: Apply 30k Phase-2 gate per spec §3.2**

Recall: regstack Phase 2 ship final NLL is 3.5734, throughput 28,072 tok/s. Gate thresholds:
- NLL: `L2_NLL_30k ≤ 3.5534` (+0.02 nat improvement bar; equivalently `ΔNLL_30k ≤ -0.02 nat`).
- Throughput: `L2_TOKS ≥ 26,668` (≤5% wall regression; expected ~29,500 = +5-6%).
- VRAM: `L2_VRAM ≤ 15.72 GB`.

```
if NLL gate passes AND throughput gate passes AND VRAM gate passes:
    PASS → proceed to position-stratified eval (Task 5.5).
elif NLL gate fails AND throughput improves:
    HALF-FAIL (wall-only) → publish honest result; do NOT swap ship default.
    Keep --layer-drop-pmax 0.1 as opt-in flag.
else:
    FAIL → publish honest negative; close arc.
```

### Task 5.5: Position-stratified val NLL eval at L2 step 30000

**Files:** Helper script in `~/dev/glades-trainer/scripts/` (create if absent).

- [ ] **Step 1: Find or create the position-stratified eval helper**

```bash
ls ~/dev/glades-trainer/scripts/ 2>/dev/null | grep -i "pos\|stratif\|bucket"
```

If a helper exists from the regstack arc, reuse it. Otherwise create
`~/dev/glades-trainer/scripts/eval_position_stratified.sh` mirroring the
regstack Phase 2 doc's position-bucketed eval. The buckets are
`[0, 4k)`, `[4k, 8k)`, `[8k, 12k)`, `[12k, 16k]`.

- [ ] **Step 2: Run position-stratified eval on the L2 checkpoint**

```bash
cd ~/dev/glades-trainer
sh scripts/eval_position_stratified.sh \
    database/checkpoints/chiron_1B_T16384_regstack_layerdrop_phase2.final \
    --buckets 0,4096,8192,12288,16384 2>&1 | tee logs/l2_posstrat_$(date +%Y%m%d).log
```

- [ ] **Step 3: Compare against regstack Phase 2 ship per-bucket NLL**

The regstack Phase 2 ship doc records per-bucket NLL at step 30000 (or recompute it on the ship checkpoint using the same script).

Check the deep-position bucket `[12k, 16k]`:
- If `L2_NLL[12k,16k] ≤ regstack_NLL[12k,16k] + 0.05 nat` → PASS (deep positions did not regress beyond noise).
- If `L2_NLL[12k,16k] > regstack_NLL[12k,16k] + 0.05 nat` → R-LD-6 fires per spec §3.2. Flag for follow-up before ship swap. Possible mitigation: retry with `p_max = 0.05` or a non-linear schedule that protects deep layers more.

---

## Phase 6: Documentation + ship

### Task 6.1: Write final result doc

**Files:**
- Create: `~/dev/glades-ml/research/REGSTACK_LAYERDROP_PHASE2_<RESULT>_2026_MM_DD.md` (substitute PASS or FAIL_NLL or FAIL_WALL in place of `<RESULT>` and today's date in place of `MM_DD`).

- [ ] **Step 1: Choose the result doc filename**

Based on Task 5.4 / 5.5 outcomes:
- `REGSTACK_LAYERDROP_PHASE2_PASS_2026_MM_DD.md` if both gates passed.
- `REGSTACK_LAYERDROP_PHASE2_FAIL_NLL_2026_MM_DD.md` if NLL gate failed.
- `REGSTACK_LAYERDROP_PHASE2_FAIL_WALL_2026_MM_DD.md` if wall regressed unexpectedly.

- [ ] **Step 2: Populate the doc using the regstack Phase 2 doc as a template**

```bash
ls ~/dev/glades-ml/research/REGSTACK_PHASE2_2026_05_22.md
```

Mirror its structure: motivation, pilot results table (L0 / L1), 30k Phase-2 results table (L2 trajectory at val checkpoints), position-stratified table, decision, comparison to regstack ship. Include:
- Pilot ΔNLL and Δtoks_pct.
- Full 30k trajectory val NLL (20 data points).
- Position-stratified NLL at step 30000 across 4 buckets.
- Peak VRAM at L2.
- Mask-trace summary: mean keep rate per layer over the 30k run (e.g., layer 0 = 1.00, layer 23 = 0.90, mean = 0.95).
- γ_h evolution comparison vs regstack ship at steps 1000/2500/5000/15000/30000 (if logged).

- [ ] **Step 3: Commit the result doc**

```bash
cd ~/dev/glades-ml
git add research/REGSTACK_LAYERDROP_PHASE2_*.md
git commit -m "$(cat <<'EOF'
Document LayerDrop arc result (PASS|FAIL_NLL|FAIL_WALL) — 2026-MM-DD

Pilot L0/L1 5k, 30k Phase-2 L2, position-stratified eval at step 30000.
[Insert one-sentence headline result.]

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 6.2: Update CLAUDE.md flagship pointer (only if L2 PASSes)

**Files:**
- Modify: `~/dev/glades-ml/CLAUDE.md`

- [ ] **Step 1: Replace the "Current Production Flagship" block**

Open `CLAUDE.md` and find the `## Current Production Flagship — CHIRON 1B @ T=16384 (regstack Phase 2 ship 2026-05-22)` section. Replace its checkpoint name, NLL number, throughput, VRAM, reproduce command, and "Full spec" pointer with the LayerDrop arc values:

- Checkpoint: `chiron_1B_T16384_regstack_layerdrop_phase2.final`
- Stack: regstack Phase 2 (Z-loss + QK-Norm) **PLUS** LayerDrop p_max=0.1 linear-rising.
- Perf: <L2_TOKS> tok/s @ T=16384 (was 28,072 at regstack Phase 2 ship, +X.XX%); same VRAM as regstack (~14.97 GB peak).
- Final val NLL <L2_NLL_30k> @ step 30000 (vs regstack Phase 2 ship 3.5734; Δ <L2_NLL_30k - 3.5734>).
- Reproduce training: `sh run.sh flagship --zloss-coef 1e-4 --qk-norm --layer-drop-pmax 0.1`.
- Run inference: `sh runner.sh --flagship` (verify path).
- Full spec / evidence: `docs/superpowers/specs/2026-05-23-chiron-1b-layerdrop-design.md` and `research/REGSTACK_LAYERDROP_PHASE2_PASS_2026_MM_DD.md`.

Move the previous "Current Production Flagship — CHIRON 1B regstack Phase 2 ship" block to a new "## Prior regstack Phase 2 flagship (kept for context)" subsection, immediately below.

- [ ] **Step 2: Commit CLAUDE.md update**

```bash
cd ~/dev/glades-ml
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
Ship LayerDrop CHIRON 1B as new production flagship

New ship: chiron_1B_T16384_regstack_layerdrop_phase2.final
Stack: regstack Phase 2 (Z-loss + QK-Norm) + LayerDrop p_max=0.1
linear-rising stochastic depth (Fan 2019).

Final val NLL <N> @ 30k (Δ <Δ> vs regstack ship 3.5734).
Throughput <T> tok/s (Δ <Δ%> vs regstack 28,072).
Peak VRAM <V> GB.

Reproduce: sh run.sh flagship --zloss-coef 1e-4 --qk-norm --layer-drop-pmax 0.1
Spec: docs/superpowers/specs/2026-05-23-chiron-1b-layerdrop-design.md
Evidence: research/REGSTACK_LAYERDROP_PHASE2_PASS_2026_MM_DD.md

Prior regstack Phase 2 ship demoted to "kept for context" subsection;
remains loadable (math bit-identical when --layer-drop-pmax 0.0).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 6.3: Archive prior ship + memory entry

- [ ] **Step 1: Archive the regstack Phase 2 ship checkpoint**

If the prior ship has not already been archived to a stable location:

```bash
cd ~/dev/glades-trainer
ls -la database/checkpoints/chiron_1B_T16384_regstack_phase2/
```

Confirm the prior ship is at the canonical archive path. No move is necessary — both ships coexist in the database/checkpoints/ tree. The runner.sh `--flagship` pointer (and CLAUDE.md) is the only "live" pointer.

- [ ] **Step 2: Add a memory entry**

Add a file at `/home/robert/.claude/projects/-home-robert-dev-glades-ml/memory/layerdrop_phase2_ship.md` (project memory) with:

```markdown
---
name: layerdrop_phase2_ship
description: LayerDrop arc PASS — new CHIRON 1B flagship layerDropPMax=0.1 linear-rising on top of regstack Phase 2. Val NLL <N> @ 30k vs regstack 3.5734 (Δ <Δ> nat). Wall <T> tok/s (Δ <Δ%> vs 28,072).
metadata:
  type: project
---

[fill in: PASS headline, 30k trajectory highlights, position-stratified
result on [12k,16k] bucket, mask-trace summary, γ_h evolution
comparison, plus pointer to research/REGSTACK_LAYERDROP_PHASE2_PASS_...md
and the new spec/plan docs.]
```

Update `MEMORY.md` (the index) by adding ONE new line at the top of `## Topics`:

```markdown
- [LAYERDROP PHASE 2 SHIP 2026-MM-DD](layerdrop_phase2_ship.md) — New CHIRON 1B flagship: regstack Phase 2 + `--layer-drop-pmax 0.1` linear. Val NLL <N> @ 30k vs regstack 3.5734 (Δ <Δ>). Wall <T> tok/s (Δ <Δ%>). Mean keep rate 0.95.
```

(Keep the line under ~200 chars per MEMORY.md's existing convention; the index is a one-liner-per-topic file.)

---

## Self-review checklist (run before declaring plan complete)

This is the checklist YOU (the planner) ran while drafting; the implementer can refer to it as a sanity check at the end:

- **Spec §1 motivation:** Phase 0 + Phase 1 + Phase 2 + Phase 3 wire the mechanism. ✓
- **Spec §2.1 mechanism math:** Phase 1 helpers (schedule + Bernoulli) + Phase 2 CPU + Phase 3 GPU. ✓
- **Spec §2.1 bit-identicality at p_max=0:** Phase 2.4 test (`CHIRONLayerDropDisabledParityTest`). ✓
- **Spec §2.1 implementation sites:** training_config.h, transformer_config.cpp, sgd_transformer.cpp CPU + GPU, transformer_kernels.h, run.sh, chiron_main.cpp. ✓ All covered.
- **Spec §2.1 RNG/determinism:** Phase 3 Task 3.3 (`CHIRONLayerDropDeterministicMasksTest`). ✓
- **Spec §2.1 SCFA interaction:** the audit notes in Tasks 2.2/3.1 cover R-LD-4. ✓
- **Spec §3.1 run plan:** Phase 5 Tasks 5.1, 5.2, 5.3, 5.4. ✓
- **Spec §3.2 gate criteria:** Phase 5 Task 5.2 Step 3 (pilot gate) + Task 5.4 Step 3 (Phase-2 gate). ✓
- **Spec §3.2 borderline-null seed=1338 disambiguation:** Phase 5 Task 5.3. ✓
- **Spec §3.2 position-stratified deep-bucket:** Phase 5 Task 5.5. ✓
- **Spec §3.3 production retrain handoff:** Phase 6 Tasks 6.1, 6.2, 6.3. ✓
- **Spec §4 risks (R-LD-1 to R-LD-7):** all referenced in the relevant tasks (Task 5.3 for R-LD-1, audit notes for R-LD-2/R-LD-4, Task 3.3 for R-LD-5, Task 5.5 for R-LD-6). R-LD-3 (Z-loss interaction) and R-LD-7 (eval-mode zeroing) — see notes below.
- **R-LD-3 Z-loss interaction:** Mention in Task 5.2 Step 4 to check `Z-loss term magnitude` in L1 logs vs L0.
- **R-LD-7 eval-mode zeroing:** The current code path uses `layerDropPMax > 0.0f` as the gate. The eval call site sets `layerDropPMax = 0.0f` (or `trainingConfig.transformer.layerDropPMax` is read from a struct that is configured for training vs eval). Confirm the eval-mode call into the network sets `layerDropPMax = 0` — typically done by passing a different `TransformerRunConfig` to the eval path. (Add a small audit step at Task 4.1 Step 5: smoke-run eval to confirm.)

---

This plan implements the LayerDrop arc spec at `docs/superpowers/specs/2026-05-23-chiron-1b-layerdrop-design.md`. The next arc (UL2 mixture-of-denoisers) is out of scope and will be brainstormed fresh in its own session per the brainstorming-skill's scope decomposition.
