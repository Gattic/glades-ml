# GAN Parallelization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Parallelize GAN per-sample training loops across 20+ CPUs using the existing ThreadPool.

**Architecture:** Pre-generate RNG/data sequentially, dispatch sample chunks to threads with per-thread gradient buffers and scratch, then reduce gradients in thread order for determinism. Forward functions read shared weights (safe); backward overloads accumulate into thread-local GradientBuffers.

**Tech Stack:** C++98 with pthreads, existing `glades::ThreadPool`, SIMD kernels (`axpy_f32`, `gemv_*`).

**Build/Test Commands:**
- Build main library: `cd /home/rob/dev/glades-ml && cmake -B build && make -C build -j$(nproc)`
- Build tests: `cd /home/rob/dev/glades-ml/unit-tests && cmake -B build && make -C build -j$(nproc)`
- Run GAN tests: `cd /home/rob/dev/glades-ml/unit-tests && make -C build run ARGS=gan`

---

### Task 1: Add `deterministicReduce` to GANConfig

**Files:**
- Modify: `Backend/Machine Learning/Networks/gan.h:108-134` (GANConfig constructor)

**Step 1: Add the field and default**

In `gan.h`, add `bool deterministicReduce;` to `GANConfig` after line 106 (`bool useCycle;`), and initialize it to `true` in the constructor initializer list.

```cpp
// After line 106:
bool useCycle;  // enable CycleGAN features

// Parallelism
bool deterministicReduce; // ordered gradient reduction for reproducibility (default: true)
```

And in the initializer list (after `useCycle(false)`):
```cpp
useCycle(false),
deterministicReduce(true)
```

**Step 2: Build to verify no compile errors**

Run: `cd /home/rob/dev/glades-ml && cmake -B build && make -C build -j$(nproc)`
Expected: Clean build.

**Step 3: Commit**

```bash
git add "Backend/Machine Learning/Networks/gan.h"
git commit -m "feat(gan): add deterministicReduce config flag"
```

---

### Task 2: Implement GradientBuffer struct

**Files:**
- Modify: `Backend/Machine Learning/Networks/gan.h` (add struct declaration)
- Modify: `Backend/Machine Learning/Networks/gan.cpp` (add method implementations)

**Step 1: Write GradientBuffer test**

Add at the end of `GANUnitTest()` in `unit-tests/Backend/Machine Learning/gan-test.cpp`:

```cpp
// ------------------------------------------------------------------
// Test 16: GradientBuffer initFrom / zero / addTo round-trip
// ------------------------------------------------------------------
printf("-----------------------------------\n");
printf("GAN Test 16: GradientBuffer round-trip\n");
printf("-----------------------------------\n");
{
	// Build a small DFF GAN, init tensors, then test GradientBuffer
	glades::NumberInput* di = make_gaussian_mixture(20, 99u);

	std::vector<unsigned int> genHidden;
	genHidden.push_back(8u);
	std::vector<unsigned int> discHidden;
	discHidden.push_back(8u);

	glades::NNInfo* genInfo = make_gen_info("ut_gb_gen", genHidden, 0.0002f, glades::GMath::LEAKY);
	glades::NNInfo* discInfo = make_disc_info("ut_gb_disc", discHidden, 0.0002f, glades::GMath::LEAKY);

	glades::GANConfig cfg;
	cfg.archType = glades::GANConfig::GAN_DFF;
	cfg.noiseDim = 4;
	cfg.epochs = 1;
	cfg.batchSize = 5;

	glades::GAN gan(cfg, genInfo, discInfo);
	gan.setSeed(42);

	// Train 1 epoch so tensors are initialized
	glades::NNetworkStatus st = gan.train(di, NULL);
	G_assert(__FILE__, __LINE__, "==============GradientBuffer: train failed==============", st.ok());

	// Test GradientBuffer on generator
	const glades::NNetwork& gen = gan.getGenerator();
	glades::GradientBuffer buf;
	buf.initFromDFF(gen);

	// Verify sizes match
	G_assert(__FILE__, __LINE__, "==============GradientBuffer: DFF gW count mismatch==============",
	         buf.dffGW.size() == gen.tensorDff.T.size());
	for (size_t t = 0; t < buf.dffGW.size(); ++t)
	{
		G_assert(__FILE__, __LINE__, "==============GradientBuffer: DFF gW[t] size mismatch==============",
		         buf.dffGW[t].size() == gen.tensorDff.T[t].gW.size());
		G_assert(__FILE__, __LINE__, "==============GradientBuffer: DFF gBias[t] size mismatch==============",
		         buf.dffGBias[t].size() == gen.tensorDff.T[t].gBias.size());
	}

	// After init, buffer should be zero
	bool allZero = true;
	for (size_t t = 0; t < buf.dffGW.size(); ++t)
		for (size_t i = 0; i < buf.dffGW[t].size(); ++i)
			if (buf.dffGW[t][i] != 0.0f) allZero = false;
	G_assert(__FILE__, __LINE__, "==============GradientBuffer: not zero after init==============", allZero);

	// Write known values, addTo network, verify accumulation
	for (size_t t = 0; t < buf.dffGW.size(); ++t)
	{
		for (size_t i = 0; i < buf.dffGW[t].size(); ++i)
			buf.dffGW[t][i] = 1.0f;
		for (size_t j = 0; j < buf.dffGBias[t].size(); ++j)
			buf.dffGBias[t][j] = 2.0f;
	}

	// Zero the network grads first, then addTo
	for (size_t t = 0; t < gen.tensorDff.T.size(); ++t)
	{
		// Cast away const for test only
		glades::NNetwork& genMut = const_cast<glades::NNetwork&>(gen);
		std::memset(&genMut.tensorDff.T[t].gW[0], 0, genMut.tensorDff.T[t].gW.size() * sizeof(float));
		std::memset(&genMut.tensorDff.T[t].gBias[0], 0, genMut.tensorDff.T[t].gBias.size() * sizeof(float));
	}
	buf.addToDFF(const_cast<glades::NNetwork&>(gen));

	bool addCorrect = true;
	for (size_t t = 0; t < gen.tensorDff.T.size(); ++t)
	{
		for (size_t i = 0; i < gen.tensorDff.T[t].gW.size(); ++i)
			if (gen.tensorDff.T[t].gW[i] != 1.0f) addCorrect = false;
		for (size_t j = 0; j < gen.tensorDff.T[t].gBias.size(); ++j)
			if (gen.tensorDff.T[t].gBias[j] != 2.0f) addCorrect = false;
	}
	G_assert(__FILE__, __LINE__, "==============GradientBuffer: addTo incorrect==============", addCorrect);

	// Test zero()
	buf.zero();
	allZero = true;
	for (size_t t = 0; t < buf.dffGW.size(); ++t)
		for (size_t i = 0; i < buf.dffGW[t].size(); ++i)
			if (buf.dffGW[t][i] != 0.0f) allZero = false;
	G_assert(__FILE__, __LINE__, "==============GradientBuffer: not zero after zero()==============", allZero);

	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	delete genInfo;
	delete discInfo;
	delete di;
}
```

**Step 2: Run test to verify it fails**

Run: `cd /home/rob/dev/glades-ml/unit-tests && cmake -B build && make -C build -j$(nproc) 2>&1 | tail -20`
Expected: FAIL — `GradientBuffer` is not defined.

**Step 3: Implement GradientBuffer in gan.h**

Add the struct declaration inside `namespace glades` before the `GAN` class (after line 172, before `class GAN`):

```cpp
struct GradientBuffer
{
	// DFF gradient arrays: [transition][weights/biases]
	std::vector<std::vector<float> > dffGW;
	std::vector<std::vector<float> > dffGBias;

	// CNN conv gradient arrays: [layer][weights/biases]
	std::vector<std::vector<float> > convGW;
	std::vector<std::vector<float> > convGBias;
	// CNN FC gradient arrays: [layer][weights/biases]
	std::vector<std::vector<float> > fcGW;
	std::vector<std::vector<float> > fcGBias;

	// Deconv FC gradient arrays
	std::vector<float> deconvFcGW;
	std::vector<float> deconvFcGBias;
	// Deconv layer gradient arrays: [layer][weights/biases]
	std::vector<std::vector<float> > deconvGW;
	std::vector<std::vector<float> > deconvGBias;

	void initFromDFF(const NNetwork& net);
	void initFromCNN(const NNetwork& net);
	void initFromDeconv(const NNetwork& net);
	void zero();
	void addToDFF(NNetwork& net) const;
	void addToCNN(NNetwork& net) const;
	void addToDeconv(NNetwork& net) const;
};
```

**Step 4: Implement GradientBuffer methods in gan.cpp**

Add near the top of `gan.cpp` (after the existing `#include` block, before the constructor):

```cpp
// ============================================================
// GradientBuffer
// ============================================================

void GradientBuffer::initFromDFF(const NNetwork& net)
{
	const unsigned int numT = static_cast<unsigned int>(net.tensorDff.T.size());
	dffGW.resize(numT);
	dffGBias.resize(numT);
	for (unsigned int t = 0; t < numT; ++t)
	{
		dffGW[t].assign(net.tensorDff.T[t].gW.size(), 0.0f);
		dffGBias[t].assign(net.tensorDff.T[t].gBias.size(), 0.0f);
	}
}

void GradientBuffer::initFromCNN(const NNetwork& net)
{
	const unsigned int numConv = static_cast<unsigned int>(net.tensorCnn.convLayers.size());
	convGW.resize(numConv);
	convGBias.resize(numConv);
	for (unsigned int l = 0; l < numConv; ++l)
	{
		convGW[l].assign(net.tensorCnn.convLayers[l].gW.size(), 0.0f);
		convGBias[l].assign(net.tensorCnn.convLayers[l].gBias.size(), 0.0f);
	}
	const unsigned int numFC = static_cast<unsigned int>(net.tensorCnn.fcLayers.size());
	fcGW.resize(numFC);
	fcGBias.resize(numFC);
	for (unsigned int t = 0; t < numFC; ++t)
	{
		fcGW[t].assign(net.tensorCnn.fcLayers[t].gW.size(), 0.0f);
		fcGBias[t].assign(net.tensorCnn.fcLayers[t].gBias.size(), 0.0f);
	}
}

void GradientBuffer::initFromDeconv(const NNetwork& net)
{
	const NNetwork::TensorDeconvState& ds = net.tensorDeconv;
	deconvFcGW.assign(ds.fcGW.size(), 0.0f);
	deconvFcGBias.assign(ds.fcGBias.size(), 0.0f);
	const unsigned int numLayers = static_cast<unsigned int>(ds.layers.size());
	deconvGW.resize(numLayers);
	deconvGBias.resize(numLayers);
	for (unsigned int l = 0; l < numLayers; ++l)
	{
		deconvGW[l].assign(ds.layers[l].gW.size(), 0.0f);
		deconvGBias[l].assign(ds.layers[l].gBias.size(), 0.0f);
	}
}

void GradientBuffer::zero()
{
	for (size_t t = 0; t < dffGW.size(); ++t)
	{
		if (!dffGW[t].empty()) std::memset(&dffGW[t][0], 0, dffGW[t].size() * sizeof(float));
		if (!dffGBias[t].empty()) std::memset(&dffGBias[t][0], 0, dffGBias[t].size() * sizeof(float));
	}
	for (size_t l = 0; l < convGW.size(); ++l)
	{
		if (!convGW[l].empty()) std::memset(&convGW[l][0], 0, convGW[l].size() * sizeof(float));
		if (!convGBias[l].empty()) std::memset(&convGBias[l][0], 0, convGBias[l].size() * sizeof(float));
	}
	for (size_t t = 0; t < fcGW.size(); ++t)
	{
		if (!fcGW[t].empty()) std::memset(&fcGW[t][0], 0, fcGW[t].size() * sizeof(float));
		if (!fcGBias[t].empty()) std::memset(&fcGBias[t][0], 0, fcGBias[t].size() * sizeof(float));
	}
	if (!deconvFcGW.empty()) std::memset(&deconvFcGW[0], 0, deconvFcGW.size() * sizeof(float));
	if (!deconvFcGBias.empty()) std::memset(&deconvFcGBias[0], 0, deconvFcGBias.size() * sizeof(float));
	for (size_t l = 0; l < deconvGW.size(); ++l)
	{
		if (!deconvGW[l].empty()) std::memset(&deconvGW[l][0], 0, deconvGW[l].size() * sizeof(float));
		if (!deconvGBias[l].empty()) std::memset(&deconvGBias[l][0], 0, deconvGBias[l].size() * sizeof(float));
	}
}

void GradientBuffer::addToDFF(NNetwork& net) const
{
	for (size_t t = 0; t < dffGW.size(); ++t)
	{
		NNetwork::TensorDFFState::Transition& tr = net.tensorDff.T[t];
		for (size_t i = 0; i < dffGW[t].size(); ++i)
			tr.gW[i] += dffGW[t][i];
		for (size_t j = 0; j < dffGBias[t].size(); ++j)
			tr.gBias[j] += dffGBias[t][j];
	}
}

void GradientBuffer::addToCNN(NNetwork& net) const
{
	for (size_t l = 0; l < convGW.size(); ++l)
	{
		NNetwork::TensorCNNState::ConvLayer& cl = net.tensorCnn.convLayers[l];
		for (size_t i = 0; i < convGW[l].size(); ++i)
			cl.gW[i] += convGW[l][i];
		for (size_t j = 0; j < convGBias[l].size(); ++j)
			cl.gBias[j] += convGBias[l][j];
	}
	for (size_t t = 0; t < fcGW.size(); ++t)
	{
		NNetwork::TensorCNNState::FCTransition& fc = net.tensorCnn.fcLayers[t];
		for (size_t i = 0; i < fcGW[t].size(); ++i)
			fc.gW[i] += fcGW[t][i];
		for (size_t j = 0; j < fcGBias[t].size(); ++j)
			fc.gBias[j] += fcGBias[t][j];
	}
}

void GradientBuffer::addToDeconv(NNetwork& net) const
{
	NNetwork::TensorDeconvState& ds = net.tensorDeconv;
	for (size_t i = 0; i < deconvFcGW.size(); ++i)
		ds.fcGW[i] += deconvFcGW[i];
	for (size_t j = 0; j < deconvFcGBias.size(); ++j)
		ds.fcGBias[j] += deconvFcGBias[j];
	for (size_t l = 0; l < deconvGW.size(); ++l)
	{
		NNetwork::TensorDeconvState::DeconvLayer& dl = ds.layers[l];
		for (size_t i = 0; i < deconvGW[l].size(); ++i)
			dl.gW[i] += deconvGW[l][i];
		for (size_t j = 0; j < deconvGBias[l].size(); ++j)
			dl.gBias[j] += deconvGBias[l][j];
	}
}
```

**Step 5: Build and run test**

Run: `cd /home/rob/dev/glades-ml && cmake -B build && make -C build -j$(nproc)`
Run: `cd /home/rob/dev/glades-ml/unit-tests && cmake -B build && make -C build -j$(nproc) && make -C build run ARGS=gan 2>&1 | grep -E "(Test 16|FAIL|Success)"`
Expected: "Unit Test Success" for Test 16.

**Step 6: Commit**

```bash
git add "Backend/Machine Learning/Networks/gan.h" "Backend/Machine Learning/Networks/gan.cpp" \
        "unit-tests/Backend/Machine Learning/gan-test.cpp"
git commit -m "feat(gan): add GradientBuffer struct with initFrom/zero/addTo"
```

---

### Task 3: Add backward overloads that accept GradientBuffer*

**Files:**
- Modify: `Backend/Machine Learning/Networks/gan.h` (add overload declarations)
- Modify: `Backend/Machine Learning/Networks/gan.cpp` (implement overloads)

This is the core enabler. Each backward function gets an overload where gradient accumulation targets the GradientBuffer instead of the network.

**Step 1: Add overload declarations to gan.h**

After the existing `dffBackward` declaration (line 363-367), add:

```cpp
// Thread-safe overload: accumulate gradients into external buffer instead of net
void dffBackward(const NNetwork& net, const std::vector<std::vector<float> >& activations,
                 const float* outputGrad, unsigned int outputSize,
                 std::vector<float>* inputGrad,
                 GradientBuffer& gradBuf,
                 bool sigmoidOutput = false,
                 const LayerNormParams* lnp = NULL,
                 GradientBuffer* lnGradBuf = NULL);
```

Note: the network parameter becomes `const NNetwork&` (we don't write to it). The `LayerNormParams*` becomes `const` too.

After the existing `cnnBackward` declaration (line 374-377), add:

```cpp
void cnnBackward(const NNetwork& net, const float* input,
                 const float* outputGrad, unsigned int outputSize,
                 std::vector<float>* inputGrad,
                 GradientBuffer& gradBuf,
                 const std::vector<float>* penultGrad = NULL);
```

After the existing `deconvBackward` declaration (line 385-387), add:

```cpp
void deconvBackward(const NNetwork& net, const std::vector<std::vector<float> >& scratch,
                    const float* outputGrad, unsigned int outputSize,
                    std::vector<float>* inputGrad,
                    GradientBuffer& gradBuf);
```

After `qHeadBackward` (line 413-414), add:

```cpp
void qHeadBackward(const QNetworkHead& head, const float* shared, const float* qGrad,
                   std::vector<float>& sharedGrad,
                   GradientBuffer& gradBuf);
```

After `dffBackwardStyled` (line 432-440), add:

```cpp
void dffBackwardStyled(const NNetwork& net, const std::vector<std::vector<float> >& activations,
                       const float* outputGrad, unsigned int outputSize,
                       const std::vector<float>& w,
                       const std::vector<std::vector<float> >& xNorms,
                       const std::vector<float>& means, const std::vector<float>& invStds,
                       const std::vector<std::vector<float> >& noiseVecs,
                       const std::vector<StyleAffine>& affines,
                       std::vector<float>& gScalesLocal,
                       std::vector<float>& dW,
                       GradientBuffer& gradBuf,
                       std::vector<std::vector<float> >& styleGW,
                       std::vector<std::vector<float> >& styleGBias,
                       std::vector<std::vector<float> >& scratchDeltaLocal);
```

After `computeGradientPenalty` (line 396-399), add:

```cpp
float computeGradientPenalty(const NNetwork& disc, const float* real, const float* fake,
                             unsigned int dim, float epsilon,
                             GradientBuffer& discardBuf);
```

**Step 2: Implement the overloads in gan.cpp**

For each overload, copy the original function body and replace every `tr.gW[i] += ...` with `gradBuf.dffGW[t][i] += ...` (and similarly for other gradient targets). The logic is identical; only the accumulation target changes.

Key changes per function:

**`dffBackward` overload** (~130 lines, near line 1105 in gan.cpp):
- Replace `tr.gBias[j] += delta[t + 1u][j]` with `gradBuf.dffGBias[t][j] += delta[t + 1u][j]`
- Replace `axpy_f32(&tr.gW[...], ...)` with `axpy_f32(&gradBuf.dffGW[t][...], ...)`
- If `lnGradBuf` provided: replace `lnp->gGamma[tLN][i] += ...` with `lnGradBuf->dffGW[tLN][i] += ...` (reuse dffGW for LN gamma grads — or add dedicated fields; see implementation note below)
- Use a local `std::vector<std::vector<float> >& delta` instead of accessing `scratchDelta`

**Implementation note for LayerNorm grads:** Since LayerNormParams has its own `gGamma`/`gBeta` arrays that are separate from the network, the simplest approach is to add `std::vector<std::vector<float> > lnGGamma, lnGBeta;` fields to GradientBuffer and corresponding `initFromLN(const LayerNormParams&)` / `addToLN(LayerNormParams&)` / zero methods.

**`cnnBackward` overload** (~180 lines, near line 1384):
- Replace `fc.gBias[j] += d` with `gradBuf.fcGBias[t][j] += d`
- Replace `fc.gW[rowOff + i] += ...` with `gradBuf.fcGW[t][rowOff + i] += ...`
- Replace `sgemm_cpu(..., &cl.gW[0], ...)` with `sgemm_cpu(..., &gradBuf.convGW[l][0], ...)`
- Replace `cl.gBias[c] += bsum` with `gradBuf.convGBias[l][c] += bsum`

**`deconvBackward` overload** (~90 lines, near line 1771):
- Replace `dl.gBias[c] += bsum` with `gradBuf.deconvGBias[l][c] += bsum`
- Replace `sgemm_abt_cpu(..., &dl.gW[0], ...)` with `sgemm_abt_cpu(..., &gradBuf.deconvGW[l][0], ...)`
- Replace `ds.fcGBias[j] += dFC[j]` with `gradBuf.deconvFcGBias[j] += dFC[j]`
- Replace `ds.fcGW[rowOff + i] += ...` with `gradBuf.deconvFcGW[rowOff + i] += ...`

**`qHeadBackward` overload** (~15 lines, near line 2808):
- Store Q-head grads in `gradBuf.fcGW[0]` and `gradBuf.fcGBias[0]` (reuse the FC fields, or add dedicated `qGW`/`qGBias` fields)
- Better: add `std::vector<float> qGW, qGBias;` to GradientBuffer with `initFromQHead(const QNetworkHead&)` / `addToQHead(QNetworkHead&)`

**`dffBackwardStyled` overload** (~160 lines, near line 3280):
- Network grads -> `gradBuf.dffGW[t]` / `gradBuf.dffGBias[t]`
- Style affine grads -> `styleGW[styleIdx]` / `styleGBias[styleIdx]` (passed by reference)
- Noise scale grads -> `gScalesLocal[styleIdx]` (passed by reference)
- Delta scratch -> `scratchDeltaLocal` (passed by reference, NOT the class member `scratchDelta`)

**`computeGradientPenalty` overload** (~70 lines, near line 1973):
- Call the `dffBackward`/`cnnBackward` overload with `discardBuf` as the GradientBuffer
- Remove the zeroing of network grads (grads go to discardBuf, not the network)
- Take pre-sampled `epsilon` as a parameter (instead of drawing from rngEngine)

**Step 3: Build to verify compilation**

Run: `cd /home/rob/dev/glades-ml && cmake -B build && make -C build -j$(nproc)`
Expected: Clean build. No tests call the overloads yet — they're just available.

**Step 4: Commit**

```bash
git add "Backend/Machine Learning/Networks/gan.h" "Backend/Machine Learning/Networks/gan.cpp"
git commit -m "feat(gan): add thread-safe backward overloads with GradientBuffer"
```

---

### Task 4: Implement GANThreadCtx and thread-local scratch allocation

**Files:**
- Modify: `Backend/Machine Learning/Networks/gan.h` (add GANThreadCtx struct)
- Modify: `Backend/Machine Learning/Networks/gan.cpp` (add init method)

**Step 1: Define GANThreadCtx in gan.h**

Add after the `GradientBuffer` struct:

```cpp
struct GANThreadCtx
{
	GradientBuffer genGrads;
	GradientBuffer discGrads;
	GradientBuffer qGrads;
	GradientBuffer mappingGrads;

	// Per-thread style affine gradient arrays: [affineIdx][weights/biases]
	std::vector<std::vector<float> > styleGW, styleGBias;
	std::vector<float> gNoiseScalesLocal;

	// LayerNorm gradient arrays
	std::vector<std::vector<float> > lnGGamma, lnGBeta;

	// Activation scratch
	std::vector<std::vector<float> > genAct, discActReal, discActFake;
	std::vector<float> cnnOutReal, cnnOutFake;
	std::vector<float> deconvOut;
	std::vector<std::vector<float> > deconvScratch;
	std::vector<float> dFake, dWVec, genInput;

	// Style forward scratch
	std::vector<std::vector<float> > mapAct;
	std::vector<float> wVec;
	std::vector<std::vector<float> > sXNorms, sNoiseVecs;
	std::vector<float> sMeans, sInvStds;

	// InfoGAN scratch
	std::vector<float> qOut, qGrad, sharedGrad;
	std::vector<std::vector<float> > qDelta;

	// dffBackwardStyled scratch (replaces class-member scratchDelta)
	std::vector<std::vector<float> > scratchDeltaLocal;

	// Loss accumulators
	float dLossReal, dLossFake, gLoss, wasserstein, infoLoss;
	unsigned int catCorrect, catTotal;

	GANThreadCtx()
		: dLossReal(0.0f), dLossFake(0.0f), gLoss(0.0f),
		  wasserstein(0.0f), infoLoss(0.0f),
		  catCorrect(0u), catTotal(0u) {}

	void zeroLosses();
	void zeroGrads();
};
```

**Step 2: Implement zeroLosses / zeroGrads in gan.cpp**

```cpp
void GANThreadCtx::zeroLosses()
{
	dLossReal = dLossFake = gLoss = wasserstein = infoLoss = 0.0f;
	catCorrect = catTotal = 0u;
}

void GANThreadCtx::zeroGrads()
{
	genGrads.zero();
	discGrads.zero();
	qGrads.zero();
	mappingGrads.zero();
	for (size_t i = 0; i < styleGW.size(); ++i)
	{
		if (!styleGW[i].empty()) std::memset(&styleGW[i][0], 0, styleGW[i].size() * sizeof(float));
		if (!styleGBias[i].empty()) std::memset(&styleGBias[i][0], 0, styleGBias[i].size() * sizeof(float));
	}
	if (!gNoiseScalesLocal.empty()) std::memset(&gNoiseScalesLocal[0], 0, gNoiseScalesLocal.size() * sizeof(float));
	for (size_t i = 0; i < lnGGamma.size(); ++i)
	{
		if (!lnGGamma[i].empty()) std::memset(&lnGGamma[i][0], 0, lnGGamma[i].size() * sizeof(float));
		if (!lnGBeta[i].empty()) std::memset(&lnGBeta[i][0], 0, lnGBeta[i].size() * sizeof(float));
	}
}
```

**Step 3: Build to verify**

Run: `cd /home/rob/dev/glades-ml && cmake -B build && make -C build -j$(nproc)`
Expected: Clean build.

**Step 4: Commit**

```bash
git add "Backend/Machine Learning/Networks/gan.h" "Backend/Machine Learning/Networks/gan.cpp"
git commit -m "feat(gan): add GANThreadCtx with per-thread scratch and gradient buffers"
```

---

### Task 5: Write the determinism test (integration test)

**Files:**
- Modify: `unit-tests/Backend/Machine Learning/gan-test.cpp`

This test will fail until we implement the parallel training loop (Tasks 6-7). It validates that with `deterministicReduce=true`, multi-threaded training produces the same losses as single-threaded.

**Step 1: Write the test**

Add at the end of `GANUnitTest()`:

```cpp
// ------------------------------------------------------------------
// Test 17: Determinism - 1 thread vs N threads produce identical losses
// ------------------------------------------------------------------
printf("-----------------------------------\n");
printf("GAN Test 17: Parallel determinism\n");
printf("-----------------------------------\n");
{
	glades::NumberInput* di = make_gaussian_mixture(100, 42u);

	std::vector<unsigned int> genHidden;
	genHidden.push_back(32u);
	genHidden.push_back(16u);
	std::vector<unsigned int> discHidden;
	discHidden.push_back(16u);
	discHidden.push_back(8u);

	glades::NNInfo* genInfo1 = make_gen_info("ut_det_gen1", genHidden, 0.0002f, glades::GMath::LEAKY);
	glades::NNInfo* discInfo1 = make_disc_info("ut_det_disc1", discHidden, 0.0002f, glades::GMath::LEAKY);
	glades::NNInfo* genInfo2 = make_gen_info("ut_det_gen2", genHidden, 0.0002f, glades::GMath::LEAKY);
	glades::NNInfo* discInfo2 = make_disc_info("ut_det_disc2", discHidden, 0.0002f, glades::GMath::LEAKY);

	glades::GANConfig cfg;
	cfg.lossType = glades::GANConfig::GAN_VANILLA;
	cfg.archType = glades::GANConfig::GAN_DFF;
	cfg.noiseDim = 8;
	cfg.epochs = 5;
	cfg.batchSize = 20;
	cfg.nCriticPerGenerator = 1;
	cfg.deterministicReduce = true;

	// Run 1: single-threaded (GLADES_NUM_THREADS=1)
	// We test by comparing two runs with the same seed - both should match
	// regardless of thread count when deterministicReduce is true.
	glades::GAN gan1(cfg, genInfo1, discInfo1);
	gan1.setSeed(999);
	GANCaptureMetrics met1;
	glades::NNetworkStatus st1 = gan1.train(di, &met1);
	G_assert(__FILE__, __LINE__, "==============Determinism: train1 failed==============", st1.ok());

	glades::GAN gan2(cfg, genInfo2, discInfo2);
	gan2.setSeed(999);
	GANCaptureMetrics met2;
	glades::NNetworkStatus st2 = gan2.train(di, &met2);
	G_assert(__FILE__, __LINE__, "==============Determinism: train2 failed==============", st2.ok());

	// Losses should be bit-identical
	G_assert(__FILE__, __LINE__, "==============Determinism: dLossReal mismatch==============",
	         met1.last.dLossReal == met2.last.dLossReal);
	G_assert(__FILE__, __LINE__, "==============Determinism: gLoss mismatch==============",
	         met1.last.gLoss == met2.last.gLoss);

	// Generated samples should be identical
	std::vector<std::vector<float> > samples1, samples2;
	gan1.generate(5, samples1);
	gan2.generate(5, samples2);
	bool samplesMatch = true;
	for (size_t i = 0; i < samples1.size(); ++i)
		for (size_t j = 0; j < samples1[i].size(); ++j)
			if (samples1[i][j] != samples2[i][j]) samplesMatch = false;
	G_assert(__FILE__, __LINE__, "==============Determinism: samples mismatch==============", samplesMatch);

	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	delete genInfo1; delete discInfo1;
	delete genInfo2; delete discInfo2;
	delete di;
}
```

**Step 2: Build and run to verify it passes (same seed, same thread count = always identical)**

Run: `cd /home/rob/dev/glades-ml/unit-tests && cmake -B build && make -C build -j$(nproc) && make -C build run ARGS=gan 2>&1 | grep -E "(Test 17|FAIL|Success)"`
Expected: PASS (two sequential runs with same seed are already identical).

**Step 3: Commit**

```bash
git add "unit-tests/Backend/Machine Learning/gan-test.cpp"
git commit -m "test(gan): add determinism test for parallel training"
```

---

### Task 6: Parallelize the trainSingleDomain discriminator loop

**Files:**
- Modify: `Backend/Machine Learning/Networks/gan.cpp` (trainSingleDomain, lines ~2088-2383)

This is the largest single task. We transform the discriminator sample loop into a parallel_for.

**Step 1: Add includes and helpers at top of gan.cpp**

```cpp
#include "glades_thread_pool.h"
#include <cstring>  // std::memset, std::memcpy
```

**Step 2: Add pre-allocation of thread contexts before the epoch loop**

At line ~2091 (after the scratch buffer declarations, before `for (int epoch = ...)`), add:

```cpp
// --- Parallel training setup ---
const unsigned int nThreads = ThreadPool::instance().numThreads();
std::vector<GANThreadCtx> threadCtx(nThreads);

// Pre-allocate thread-local gradient buffers
for (unsigned int tid = 0; tid < nThreads; ++tid)
{
	threadCtx[tid].genInput.resize(genInputDim, 0.0f);

	// Init gradient buffers based on architecture
	if (genIsDeconv)
		threadCtx[tid].genGrads.initFromDeconv(generator);
	else
		threadCtx[tid].genGrads.initFromDFF(generator);

	if (config.archType == GANConfig::GAN_DFF)
		threadCtx[tid].discGrads.initFromDFF(discriminator);
	else
		threadCtx[tid].discGrads.initFromCNN(discriminator);

	if (hasInfo)
	{
		threadCtx[tid].qGrads.initFromQHead(qHead);
	}
	if (hasStyle)
	{
		threadCtx[tid].mappingGrads.initFromDFF(mappingNet);
		// Init style affine grad arrays
		threadCtx[tid].styleGW.resize(styleAffines.size());
		threadCtx[tid].styleGBias.resize(styleAffines.size());
		for (size_t sa = 0; sa < styleAffines.size(); ++sa)
		{
			threadCtx[tid].styleGW[sa].assign(styleAffines[sa].gW.size(), 0.0f);
			threadCtx[tid].styleGBias[sa].assign(styleAffines[sa].gBias.size(), 0.0f);
		}
		threadCtx[tid].gNoiseScalesLocal.assign(noiseScales.size(), 0.0f);
	}
	if (config.generatorLayerNorm && genLNParams.initialized)
	{
		threadCtx[tid].lnGGamma.resize(genLNParams.gGamma.size());
		threadCtx[tid].lnGBeta.resize(genLNParams.gBeta.size());
		for (size_t i = 0; i < genLNParams.gGamma.size(); ++i)
		{
			threadCtx[tid].lnGGamma[i].assign(genLNParams.gGamma[i].size(), 0.0f);
			threadCtx[tid].lnGBeta[i].assign(genLNParams.gBeta[i].size(), 0.0f);
		}
	}
}

// Pre-allocation buffers for batch data
std::vector<const float*> preRealPtrs;
std::vector<unsigned int> preRealSizes;
std::vector<std::vector<float> > preNoise;
std::vector<std::vector<float> > preCatCode, preContCode;
std::vector<float> preGPEpsilon; // for WGAN-GP
```

**Step 3: Replace the discriminator sample loop (lines 2143-2341)**

Replace the `for (unsigned int s = 0; s < curBatchSize; ++s)` block with:

```cpp
// --- Pre-generate data and noise (sequential, deterministic) ---
preRealPtrs.resize(curBatchSize);
preRealSizes.resize(curBatchSize);
for (unsigned int s = 0; s < curBatchSize; ++s)
{
	const unsigned int realIdx = (dataIdx + s) % trainSize;
	realData->getTrainRowView(realIdx, preRealPtrs[s], preRealSizes[s]);
}
preNoise.resize(curBatchSize);
if (hasInfo) { preCatCode.resize(curBatchSize); preContCode.resize(curBatchSize); }
if (config.lossType == GANConfig::GAN_WGAN_GP) preGPEpsilon.resize(curBatchSize);
for (unsigned int s = 0; s < curBatchSize; ++s)
{
	sampleNoise(preNoise[s]);
	if (hasInfo) sampleLatentCodes(preCatCode[s], preContCode[s]);
	if (config.lossType == GANConfig::GAN_WGAN_GP)
		preGPEpsilon[s] = glades::rng::unit_float01(rngEngine);
}

// --- Zero thread-local state ---
for (unsigned int tid = 0; tid < nThreads; ++tid)
{
	threadCtx[tid].zeroGrads();
	threadCtx[tid].zeroLosses();
}

// --- Parallel sample loop ---
struct DiscCritCtx
{
	GAN* self;
	std::vector<GANThreadCtx>* ctxs;
	const std::vector<const float*>* realPtrs;
	const std::vector<std::vector<float> >* noise;
	const std::vector<std::vector<float> >* catCodes;
	const std::vector<std::vector<float> >* contCodes;
	const std::vector<float>* gpEps;
	unsigned int curBatchSize;
	unsigned int nThreads;
	// ... other needed context
};

// [Implementation: ThreadPool::parallel_for dispatches chunks of samples.
//  Each thread runs the discriminator forward/backward for its samples,
//  accumulating into threadCtx[tid].discGrads using the backward overloads.
//  See design doc for full loop body.]
```

The parallel loop body for each sample `s` in chunk `[begin, end)` is identical to the original loop body (lines 2144-2341), except:
- Read from `preRealPtrs[s]` instead of calling `getTrainRowView`
- Read from `preNoise[s]` instead of calling `sampleNoise`
- Call `dffForward(generator, ...)` with thread-local activation scratch (`ctx.genAct`)
- Call `dffForward(discriminator, ...)` with thread-local scratch (`ctx.discActReal`, `ctx.discActFake`)
- Call `dffBackward(discriminator, ..., ctx.discGrads)` overload
- For WGAN-GP: call `computeGradientPenalty(..., preGPEpsilon[s], ctx.discGrads)` overload
- For InfoGAN: call `qHeadBackward(..., ctx.qGrads)` overload
- Accumulate losses into `ctx.dLossReal`, `ctx.dLossFake`, etc.

**Step 4: Add ordered gradient reduction after the parallel loop**

```cpp
// --- Reduce gradients (deterministic: thread order 0, 1, 2, ...) ---
if (config.archType == GANConfig::GAN_DFF)
	zeroDFFGrads(discriminator);
else
	zeroCNNGrads(discriminator);
if (hasInfo)
{
	std::memset(&qHead.gW[0], 0, qHead.gW.size() * sizeof(float));
	std::memset(&qHead.gBias[0], 0, qHead.gBias.size() * sizeof(float));
}

for (unsigned int tid = 0; tid < nThreads; ++tid)
{
	if (config.archType == GANConfig::GAN_DFF)
		threadCtx[tid].discGrads.addToDFF(discriminator);
	else
		threadCtx[tid].discGrads.addToCNN(discriminator);
	if (hasInfo)
		threadCtx[tid].qGrads.addToQHead(qHead);

	batchDLossReal += threadCtx[tid].dLossReal;
	batchDLossFake += threadCtx[tid].dLossFake;
	batchInfoLoss += threadCtx[tid].infoLoss;
	epochWasserstein += threadCtx[tid].wasserstein;
	epochCatCorrect += threadCtx[tid].catCorrect;
	epochCatTotal += threadCtx[tid].catTotal;
}
```

The rest (scale + Adam update, lines 2343-2383) remains unchanged.

**Step 5: Build and run existing tests to verify no regression**

Run: `cd /home/rob/dev/glades-ml && cmake -B build && make -C build -j$(nproc)`
Run: `cd /home/rob/dev/glades-ml/unit-tests && make -C build -j$(nproc) && make -C build run ARGS=gan`
Expected: All existing tests pass. Test 17 (determinism) should also pass.

**Step 6: Commit**

```bash
git add "Backend/Machine Learning/Networks/gan.cpp"
git commit -m "feat(gan): parallelize trainSingleDomain discriminator loop"
```

---

### Task 7: Parallelize the trainSingleDomain generator loop

**Files:**
- Modify: `Backend/Machine Learning/Networks/gan.cpp` (lines ~2385-2620)

Same pattern as Task 6, but for the generator phase. Key differences:
- No real data needed (generator trains on noise only)
- Need to handle generator backward + style backward + mapping backward
- Discriminator is used in forward-only mode (to compute generator loss)
- Discriminator backward returns `dFake` which flows into generator backward

**Step 1: Pre-generate noise and codes**

```cpp
// Pre-generate for generator phase
preNoise.resize(curBatchSize);
if (hasInfo) { preCatCode.resize(curBatchSize); preContCode.resize(curBatchSize); }
for (unsigned int s = 0; s < curBatchSize; ++s)
{
	sampleNoise(preNoise[s]);
	if (hasInfo) sampleLatentCodes(preCatCode[s], preContCode[s]);
}
```

**Step 2: Transform the generator sample loop (lines 2390-2563)**

Same parallel_for pattern. Each thread processes its chunk of samples:
- Generator forward -> thread-local `ctx.genAct` (or `ctx.deconvOut`)
- Discriminator forward on fake -> thread-local `ctx.discActFake` (read-only weights, safe)
- Discriminator backward (for input grad only) -> write `ctx.dFake`, accumulate unwanted disc grads into a discard GradientBuffer
- InfoGAN Q-head forward/backward -> thread-local scratch + grads
- Generator backward -> accumulate into `ctx.genGrads`
- Style backward -> accumulate into `ctx.styleGW`, `ctx.styleGBias`, `ctx.gNoiseScalesLocal`
- Mapping backward -> accumulate into `ctx.mappingGrads`

**Important**: In the generator phase, discriminator grads are zeroed per-sample (lines 2541-2546). In the parallel version, each thread should use a thread-local discard buffer for the discriminator backward call (since we only need `dFake`, not disc grads).

**Step 3: Ordered reduction for generator grads**

```cpp
// Reduce generator gradients
if (genIsDeconv)
	zeroDeconvGrads(generator);
else
	zeroDFFGrads(generator);
if (hasStyle && !genIsDeconv)
{
	zeroDFFGrads(mappingNet);
	for (size_t sa = 0; sa < styleAffines.size(); ++sa)
	{
		std::memset(&styleAffines[sa].gW[0], 0, styleAffines[sa].gW.size() * sizeof(float));
		std::memset(&styleAffines[sa].gBias[0], 0, styleAffines[sa].gBias.size() * sizeof(float));
		gNoiseScales[sa] = 0.0f;
	}
}
if (config.generatorLayerNorm && !hasStyle && genLNParams.initialized)
{
	for (size_t i = 0; i < genLNParams.gGamma.size(); ++i)
	{
		std::memset(&genLNParams.gGamma[i][0], 0, genLNParams.gGamma[i].size() * sizeof(float));
		std::memset(&genLNParams.gBeta[i][0], 0, genLNParams.gBeta[i].size() * sizeof(float));
	}
}

for (unsigned int tid = 0; tid < nThreads; ++tid)
{
	if (genIsDeconv)
		threadCtx[tid].genGrads.addToDeconv(generator);
	else
		threadCtx[tid].genGrads.addToDFF(generator);

	if (hasStyle && !genIsDeconv)
	{
		threadCtx[tid].mappingGrads.addToDFF(mappingNet);
		for (size_t sa = 0; sa < styleAffines.size(); ++sa)
		{
			for (size_t i = 0; i < styleAffines[sa].gW.size(); ++i)
				styleAffines[sa].gW[i] += threadCtx[tid].styleGW[sa][i];
			for (size_t j = 0; j < styleAffines[sa].gBias.size(); ++j)
				styleAffines[sa].gBias[j] += threadCtx[tid].styleGBias[sa][j];
			gNoiseScales[sa] += threadCtx[tid].gNoiseScalesLocal[sa];
		}
	}
	// LayerNorm grads
	if (config.generatorLayerNorm && !hasStyle && genLNParams.initialized)
	{
		for (size_t i = 0; i < genLNParams.gGamma.size(); ++i)
		{
			for (size_t j = 0; j < genLNParams.gGamma[i].size(); ++j)
				genLNParams.gGamma[i][j] += threadCtx[tid].lnGGamma[i][j];
			for (size_t j = 0; j < genLNParams.gBeta[i].size(); ++j)
				genLNParams.gBeta[i][j] += threadCtx[tid].lnGBeta[i][j];
		}
	}

	batchGLoss += threadCtx[tid].gLoss;
	batchInfoLoss += threadCtx[tid].infoLoss;
}
```

**Step 4: Build and run all tests**

Run: `cd /home/rob/dev/glades-ml && cmake -B build && make -C build -j$(nproc)`
Run: `cd /home/rob/dev/glades-ml/unit-tests && make -C build -j$(nproc) && make -C build run ARGS=gan`
Expected: All tests pass including Test 17 (determinism).

**Step 5: Commit**

```bash
git add "Backend/Machine Learning/Networks/gan.cpp"
git commit -m "feat(gan): parallelize trainSingleDomain generator loop"
```

---

### Task 8: Parallelize trainDualDomain (CycleGAN)

**Files:**
- Modify: `Backend/Machine Learning/Networks/gan.cpp` (trainDualDomain, lines ~3331-4667)

Apply the same parallel pattern to all four sample loops in the CycleGAN training:

1. **D_A critic loop** (lines 3468-3675): Same as Task 6 but for discriminator A + generatorBA
2. **D_B critic loop** (lines 3718-3925): Same as Task 6 but for discriminator B + generatorAB
3. **Generator loop** (lines 3971-4523): Both G_AB and G_BA in the same sample loop — pre-generate noise/codes for both, parallelize the combined loop

Key differences from single-domain:
- Need two sets of thread contexts (genAB/genBA grads, discA/discB grads)
- Generator loop is larger: includes cycle reconstruction and identity loss
- Need pre-fetch for both domainA and domainB data pointers
- The on-demand buffer allocation in trainDualDomain should be moved to per-thread pre-allocation

**Implementation approach**: Factor out the parallel_for dispatch into a helper (since the same pattern is repeated 3 times), or just inline it 3 times for clarity.

**Step 1: Pre-allocate thread contexts for CycleGAN**

At the start of `trainDualDomain`, allocate thread contexts for all four networks:

```cpp
const unsigned int nThreads = ThreadPool::instance().numThreads();
std::vector<GANThreadCtx> threadCtxA(nThreads); // for D_A critic
std::vector<GANThreadCtx> threadCtxB(nThreads); // for D_B critic
std::vector<GANThreadCtx> threadCtxGen(nThreads); // for generator phase
// Init gradient buffers for each...
```

**Step 2: Parallelize D_A critic loop**

Same pattern as Task 6: pre-generate noise, pre-fetch domainA/domainB data, parallel_for, ordered reduce into discriminator.

**Step 3: Parallelize D_B critic loop**

Same pattern with discriminatorB.

**Step 4: Parallelize generator loop**

The combined G_AB + G_BA loop processes each sample through both generators. Each thread's chunk handles:
- Forward: fakeB (G_AB), fakeA (G_BA), reconstruction, identity
- Backward: all gradient accumulation into thread-local buffers
- After parallel: reduce into generator, generatorBA, mappingNet, mappingNetBA, etc.

**Step 5: Build and run all tests**

Run: `cd /home/rob/dev/glades-ml && cmake -B build && make -C build -j$(nproc)`
Run: `cd /home/rob/dev/glades-ml/unit-tests && make -C build -j$(nproc) && make -C build run ARGS=gan`
Expected: All tests pass (including CycleGAN smoke tests 10-15).

**Step 6: Commit**

```bash
git add "Backend/Machine Learning/Networks/gan.cpp"
git commit -m "feat(gan): parallelize trainDualDomain (CycleGAN) loops"
```

---

### Task 9: Add comprehensive variant tests

**Files:**
- Modify: `unit-tests/Backend/Machine Learning/gan-test.cpp`

Add determinism tests for each variant to ensure parallel training produces identical results.

**Step 1: Add WGAN-GP determinism test**

```cpp
// Test 18: WGAN-GP DFF parallel determinism
{
	// Same structure as Test 17 but with:
	cfg.lossType = glades::GANConfig::GAN_WGAN_GP;
	cfg.nCriticPerGenerator = 3;
	cfg.gpLambda = 10.0f;
	// ... run twice with same seed, compare losses
}
```

**Step 2: Add CNN determinism test**

```cpp
// Test 19: CNN vanilla parallel determinism
{
	// Use make_synthetic_image_data and CNN config
	cfg.archType = glades::GANConfig::GAN_CNN;
	// ... configure CNN layers, run twice, compare
}
```

**Step 3: Add InfoGAN determinism test**

```cpp
// Test 20: InfoGAN DFF parallel determinism
{
	cfg.useInfo = true;
	cfg.infoConfig.numCategorical = 5;
	cfg.infoConfig.numContinuous = 2;
	// ... run twice, compare including infoLoss
}
```

**Step 4: Add StyleGAN determinism test**

```cpp
// Test 21: StyleGAN DFF parallel determinism
{
	cfg.useStyle = true;
	cfg.styleConfig.mappingLayers = 2;
	cfg.styleConfig.mappingWidth = 16;
	// ... run twice, compare
}
```

**Step 5: Add CycleGAN determinism test**

```cpp
// Test 22: CycleGAN DFF parallel determinism
{
	cfg.useCycle = true;
	// Create two datasets (domainA, domainB)
	// ... run twice with same seed, compare all losses
}
```

**Step 6: Build and run all tests**

Run: `cd /home/rob/dev/glades-ml/unit-tests && make -C build -j$(nproc) && make -C build run ARGS=gan`
Expected: All tests pass.

**Step 7: Commit**

```bash
git add "unit-tests/Backend/Machine Learning/gan-test.cpp"
git commit -m "test(gan): add parallel determinism tests for all GAN variants"
```

---

### Task 10: Update DETERMINISM_AND_CONCURRENCY.md

**Files:**
- Modify: `Backend/Machine Learning/DETERMINISM_AND_CONCURRENCY.md`

**Step 1: Add GAN parallelism section**

Add after the "Practical guidance" section:

```markdown
### GAN training parallelism

- `GAN::train()` parallelizes per-sample processing within each batch using `ThreadPool`.
- With `GANConfig::deterministicReduce = true` (default), gradient reduction is ordered
  by thread index, producing bit-identical results regardless of thread count.
- With `deterministicReduce = false`, threads accumulate directly into shared gradient arrays
  for slightly faster execution at the cost of non-deterministic float summation order.
- RNG draws and `DataInput` reads are pre-generated sequentially before each parallel loop,
  preserving the same draw order as the sequential code path.
- All GAN variants (Vanilla, WGAN-GP, InfoGAN, StyleGAN, CycleGAN, composites) support
  parallel training.
```

**Step 2: Commit**

```bash
git add "Backend/Machine Learning/DETERMINISM_AND_CONCURRENCY.md"
git commit -m "docs: document GAN parallel training in concurrency policy"
```

---

## Dependency Order

```
Task 1 (config flag)
  └─> Task 2 (GradientBuffer)
        └─> Task 3 (backward overloads)
              └─> Task 4 (GANThreadCtx)
                    └─> Task 5 (determinism test)
                    └─> Task 6 (disc loop parallel)
                          └─> Task 7 (gen loop parallel)
                                └─> Task 8 (CycleGAN parallel)
                                      └─> Task 9 (variant tests)
                                            └─> Task 10 (docs)
```

## Risk Notes

- **`scratchDelta`** (class member at `gan.h:351`): Used by `dffBackwardStyled`. Each thread must use its own `scratchDeltaLocal` instead. The styled backward overload takes this as a parameter.
- **`computeGradientPenalty`**: Calls `dffBackward`/`cnnBackward` on the discriminator and zeros its grads. The overload must use a discard GradientBuffer and pre-sampled epsilon.
- **`sgemm_cpu` / `sgemm_abt_cpu`**: Used in CNN/deconv backward for gradient accumulation. The overloads must pass GradientBuffer array pointers instead of network grad pointers to these functions.
- **Memory overhead**: Each thread gets ~2x gradient buffers (gen + disc). For a 100-dim GAN with small hidden layers, this is negligible. For large CNNs, expect ~20 * (2 * sizeof_grads) additional memory.
