# ATLAS Extension Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend ATLAS optimizer to RNN/GRU/LSTM networks, add checkpoint persistence for all ATLAS state, and create an ATLAS-vs-SGD comparison benchmark.

**Architecture:** Add `atlas::WeightState` fields to `TensorRNNState` and `TensorGatedState` structs, add `if (useAtlas)` branches in each recurrent SGD file mirroring the DFF pattern, serialize/deserialize ATLAS state alongside existing optimizer state in checkpoint persistence, and create a benchmark test comparing ATLAS vs momentum SGD.

**Tech Stack:** C++98, CMake, custom test framework (`ASSERT` macro), `gettimeofday()` timing.

---

### Task 1: Add ATLAS state fields to RNN/GRU/LSTM state structs in network.h

**Files:**
- Modify: `Backend/Machine Learning/Networks/network.h:166-263`

**Step 1: Add atlas includes and state fields to TensorRNNState**

At line 180, add `atlasWxh` and `atlasWhh` to `Hidden`, and `atlasWhy` to `Out`:

```cpp
// In TensorRNNState::Hidden (after gBias line 179):
			atlas::WeightState atlasWxh;
			atlas::WeightState atlasWhh;

// In TensorRNNState::Out (after gBias line 192):
			atlas::WeightState atlasWhy;
```

**Step 2: Add atlas state fields to TensorGatedState**

```cpp
// In TensorGatedState::Hidden (after gBias line 233):
			atlas::WeightState atlasW;
			atlas::WeightState atlasU;

// In TensorGatedState::Out (after gBias line 245):
			atlas::WeightState atlasWhy;
```

**Step 3: Build to verify no compilation errors**

Run: `cd /home/rob/dev/glades-ml/build && cmake .. && make -j$(nproc)`
Expected: Clean build, no errors.

**Step 4: Commit**

```bash
git add "Backend/Machine Learning/Networks/network.h"
git commit -m "Add ATLAS state fields to RNN/GRU/LSTM state structs"
```

---

### Task 2: Add ATLAS optimizer branch to sgd_rnn.cpp

**Files:**
- Modify: `Backend/Machine Learning/Networks/sgd_rnn.cpp:197-425`

**Step 1: Add rngEngine reference to ApplyBatch struct**

In the `ApplyBatch` struct (lines 197-428), add `glades::rng::Engine& rngEngine;` as a member and capture it in the constructor from `net.rngEngine`.

At line 206, after `unsigned int outSize;`, add:
```cpp
		glades::rng::Engine& rngEngine;
```

At line 219, change `outSize(os)` to:
```cpp
		      outSize(os),
		      rngEngine(net.rngEngine)
```

**Step 2: Add ATLAS branch in the weight update section**

After the grad clipping section (line 323), before the existing output weight update loop (line 324), add a check for ATLAS and wrap existing code in `else`:

```cpp
			const bool useAtlas = (trainingConfig.optimizer.type == OptimizerConfig::ATLAS);
			if (useAtlas)
			{
				const ATLASConfig& ac = trainingConfig.atlas;
				const float lrOut = skeleton->getLearningRate(static_cast<unsigned int>(H)) * lrScheduleMultiplier;
				const float wd1Out = skeleton->getWeightDecay1(static_cast<unsigned int>(H));
				const float wd2Out = skeleton->getWeightDecay2(static_cast<unsigned int>(H));

				// Output weights
				if (!tensorRnn.O.atlasWhy.initialized && tensorRnn.O.out > 0 && tensorRnn.O.in > 0)
					atlas::initWeightState(tensorRnn.O.atlasWhy, tensorRnn.O.out, tensorRnn.O.in, ac.rank, ac.muMin, rngEngine);
				if (tensorRnn.O.atlasWhy.initialized)
					atlas::applyStep(tensorRnn.O.atlasWhy, &tensorRnn.O.Why[0], &tensorRnn.O.gWhy[0],
						tensorRnn.O.out, tensorRnn.O.in, invBatch, lrOut, wd1Out, wd2Out, gradScale,
						ac.beta, ac.muMin, ac.muMax, ac.eps, ac.tSub, ac.powerIters, rngEngine);

				// Output bias: standard SGD
				for (unsigned int k = 0; k < outSize; ++k)
				{
					float gB = tensorRnn.O.gBias[k] * invBatch;
					gB *= gradScale;
					tensorRnn.O.bias[k] -= (lrOut * gB);
					tensorRnn.O.gBias[k] = 0.0f;
					if (!is_finite(tensorRnn.O.bias[k]))
					{
						lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_RNN: non-finite output bias after ATLAS update (NaN/Inf)");
						running = false;
						return false;
					}
				}

				// Hidden layers
				for (int l = 0; l < H; ++l)
				{
					const unsigned int li = static_cast<unsigned int>(l);
					TensorRNNState::Hidden& hl = tensorRnn.H[static_cast<size_t>(l)];
					const float lr = skeleton->getLearningRate(li) * lrScheduleMultiplier;
					const float wd1 = skeleton->getWeightDecay1(li);
					const float wd2 = skeleton->getWeightDecay2(li);

					// Wxh
					if (!hl.atlasWxh.initialized && hl.h > 0 && hl.in > 0)
						atlas::initWeightState(hl.atlasWxh, hl.h, hl.in, ac.rank, ac.muMin, rngEngine);
					if (hl.atlasWxh.initialized)
						atlas::applyStep(hl.atlasWxh, &hl.Wxh[0], &hl.gWxh[0],
							hl.h, hl.in, invBatch, lr, wd1, wd2, gradScale,
							ac.beta, ac.muMin, ac.muMax, ac.eps, ac.tSub, ac.powerIters, rngEngine);

					// Whh
					if (!hl.atlasWhh.initialized && hl.h > 0)
						atlas::initWeightState(hl.atlasWhh, hl.h, hl.h, ac.rank, ac.muMin, rngEngine);
					if (hl.atlasWhh.initialized)
						atlas::applyStep(hl.atlasWhh, &hl.Whh[0], &hl.gWhh[0],
							hl.h, hl.h, invBatch, lr, wd1, wd2, gradScale,
							ac.beta, ac.muMin, ac.muMax, ac.eps, ac.tSub, ac.powerIters, rngEngine);

					// Bias: standard SGD
					for (unsigned int i = 0; i < hl.h; ++i)
					{
						float gB = hl.gBias[i] * invBatch;
						gB *= gradScale;
						hl.bias[i] -= (lr * gB);
						hl.gBias[i] = 0.0f;
						if (!is_finite(hl.bias[i]))
						{
							lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_RNN: non-finite hidden bias after ATLAS update (NaN/Inf)");
							running = false;
							return false;
						}
					}
				}

				return true;
			}
			// else: existing momentum SGD code follows unchanged
```

Then wrap the existing output/hidden update code (lines 324-425) in an `else` block (just the opening `else {` before line 324 and closing `}` after line 425).

**Step 3: Build to verify**

Run: `cd /home/rob/dev/glades-ml/build && cmake .. && make -j$(nproc)`
Expected: Clean build.

**Step 4: Commit**

```bash
git add "Backend/Machine Learning/Networks/sgd_rnn.cpp"
git commit -m "Add ATLAS optimizer branch to RNN SGD"
```

---

### Task 3: Add ATLAS optimizer branch to sgd_gru.cpp

**Files:**
- Modify: `Backend/Machine Learning/Networks/sgd_gru.cpp:197-430`

**Step 1: Add rngEngine to ApplyBatch**

Same pattern as Task 2: add `glades::rng::Engine& rngEngine;` member and capture from `net.rngEngine`.

**Step 2: Add ATLAS branch**

After grad clipping (line 323), before existing output weight update (line 324):

```cpp
			const bool useAtlas = (trainingConfig.optimizer.type == OptimizerConfig::ATLAS);
			if (useAtlas)
			{
				const ATLASConfig& ac = trainingConfig.atlas;
				const float lrOut = skeleton->getLearningRate(static_cast<unsigned int>(H)) * lrScheduleMultiplier;
				const float wd1Out = skeleton->getWeightDecay1(static_cast<unsigned int>(H));
				const float wd2Out = skeleton->getWeightDecay2(static_cast<unsigned int>(H));

				// Output: Why
				if (!tensorGru.O.atlasWhy.initialized && tensorGru.O.out > 0 && tensorGru.O.in > 0)
					atlas::initWeightState(tensorGru.O.atlasWhy, tensorGru.O.out, tensorGru.O.in, ac.rank, ac.muMin, rngEngine);
				if (tensorGru.O.atlasWhy.initialized)
					atlas::applyStep(tensorGru.O.atlasWhy, &tensorGru.O.Why[0], &tensorGru.O.gWhy[0],
						tensorGru.O.out, tensorGru.O.in, invBatch, lrOut, wd1Out, wd2Out, gradScale,
						ac.beta, ac.muMin, ac.muMax, ac.eps, ac.tSub, ac.powerIters, rngEngine);

				// Output bias: standard SGD
				for (unsigned int k = 0; k < outSize; ++k)
				{
					float gB = tensorGru.O.gBias[k] * invBatch;
					gB *= gradScale;
					tensorGru.O.bias[k] -= (lrOut * gB);
					tensorGru.O.gBias[k] = 0.0f;
				}

				// Hidden layers: whole-matrix ATLAS over packed gates
				for (int l = 0; l < H; ++l)
				{
					const unsigned int li = static_cast<unsigned int>(l);
					TensorGatedState::Hidden& hl = tensorGru.H[static_cast<size_t>(l)];
					const float lr = skeleton->getLearningRate(li) * lrScheduleMultiplier;
					const float wd1 = skeleton->getWeightDecay1(li);
					const float wd2 = skeleton->getWeightDecay2(li);

					// W: [gateCount*h, in] as single matrix
					const unsigned int wRows = tensorGru.gateCount * hl.h;
					if (!hl.atlasW.initialized && wRows > 0 && hl.in > 0)
						atlas::initWeightState(hl.atlasW, wRows, hl.in, ac.rank, ac.muMin, rngEngine);
					if (hl.atlasW.initialized)
						atlas::applyStep(hl.atlasW, &hl.W[0], &hl.gW[0],
							wRows, hl.in, invBatch, lr, wd1, wd2, gradScale,
							ac.beta, ac.muMin, ac.muMax, ac.eps, ac.tSub, ac.powerIters, rngEngine);

					// U: [gateCount*h, h] as single matrix
					const unsigned int uRows = tensorGru.gateCount * hl.h;
					if (!hl.atlasU.initialized && uRows > 0 && hl.h > 0)
						atlas::initWeightState(hl.atlasU, uRows, hl.h, ac.rank, ac.muMin, rngEngine);
					if (hl.atlasU.initialized)
						atlas::applyStep(hl.atlasU, &hl.U[0], &hl.gU[0],
							uRows, hl.h, invBatch, lr, wd1, wd2, gradScale,
							ac.beta, ac.muMin, ac.muMax, ac.eps, ac.tSub, ac.powerIters, rngEngine);

					// Bias: standard SGD
					for (size_t bi = 0; bi < hl.bias.size(); ++bi)
					{
						float gB = hl.gBias[bi] * invBatch;
						gB *= gradScale;
						hl.bias[bi] -= (lr * gB);
						hl.gBias[bi] = 0.0f;
					}
				}

				return true;
			}
```

Then wrap existing output/hidden code (lines 324-425) in `else { ... }`.

**Step 3: Build and commit**

Run: `cd /home/rob/dev/glades-ml/build && cmake .. && make -j$(nproc)`

```bash
git add "Backend/Machine Learning/Networks/sgd_gru.cpp"
git commit -m "Add ATLAS optimizer branch to GRU SGD"
```

---

### Task 4: Add ATLAS optimizer branch to sgd_lstm.cpp

**Files:**
- Modify: `Backend/Machine Learning/Networks/sgd_lstm.cpp:197-434`

**Step 1: Add rngEngine to ApplyBatch and add ATLAS branch**

Identical pattern to Task 3 but using `tensorLstm` instead of `tensorGru`, and error messages say "LSTM" instead of "GRU". The gate count is 4 instead of 3 but this is handled automatically via `tensorLstm.gateCount`.

**Step 2: Build and commit**

Run: `cd /home/rob/dev/glades-ml/build && cmake .. && make -j$(nproc)`

```bash
git add "Backend/Machine Learning/Networks/sgd_lstm.cpp"
git commit -m "Add ATLAS optimizer branch to LSTM SGD"
```

---

### Task 5: Add RNN/GRU/LSTM ATLAS unit tests

**Files:**
- Modify: `unit-tests/Backend/Machine Learning/atlas-test.cpp`

**Step 1: Add RNN ATLAS test (Test 4)**

After the existing Test 3 block (line 253), add:

```cpp
	// ---------------------------------------------------------------
	// Test 4: ATLAS RNN convergence
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 4: RNN convergence\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* di = new glades::NumberInput();
		di->trainMatrix = shmea::GMatrix(4, shmea::GVector<float>(2, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(4, shmea::GVector<float>(1, 0.0f));

		di->trainMatrix[0][0] = 0.1f; di->trainMatrix[0][1] = 0.2f;
		di->trainExpectedMatrix[0][0] = 0.3f;
		di->trainMatrix[1][0] = 0.4f; di->trainMatrix[1][1] = 0.5f;
		di->trainExpectedMatrix[1][0] = 0.9f;
		di->trainMatrix[2][0] = 0.2f; di->trainMatrix[2][1] = 0.3f;
		di->trainExpectedMatrix[2][0] = 0.5f;
		di->trainMatrix[3][0] = 0.6f; di->trainMatrix[3][1] = 0.7f;
		di->trainExpectedMatrix[3][0] = 1.3f;

		di->testMatrix = di->trainMatrix;
		di->testExpectedMatrix = di->trainExpectedMatrix;

		glades::InputLayerInfo* in = new glades::InputLayerInfo(
		    4, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::SIGMOID, 1.0f);

		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(
		    8, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::SIGMOID, 1.0f));

		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
		glades::NNInfo* info = new glades::NNInfo("ut_atlas_rnn", in, hidden, out);

		glades::NNetwork net(info, glades::NNetwork::TYPE_RNN);
		net.setSeed(42u);
		net.getTerminatorMutable().setEpoch(300);
		net.getTerminatorMutable().setAccuracy(0);

		{
			glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
			cfg.optimizer.type = glades::OptimizerConfig::ATLAS;
			cfg.atlas.rank = 4;
			cfg.atlas.tSub = 50;
		}

		const glades::NNetworkStatus st = net.train(di);
		ASSERT("==============ATLAS::RNN TrainStatus() Failed==============", st.ok());
		printf("[UT] ATLAS RNN: training completed successfully\n");

		delete di;
		delete info;
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);
```

**Step 2: Add GRU ATLAS test (Test 5) and LSTM ATLAS test (Test 6)**

Same pattern with `TYPE_GRU` and `TYPE_LSTM`, names "ut_atlas_gru" and "ut_atlas_lstm".

**Step 3: Build and run tests**

Run: `cd /home/rob/dev/glades-ml/unit-tests/build && cmake .. && make -j$(nproc) && cd .. && bash test.sh atlas`
Expected: All 6 tests pass.

**Step 4: Commit**

```bash
git add "unit-tests/Backend/Machine Learning/atlas-test.cpp"
git commit -m "Add RNN/GRU/LSTM ATLAS unit tests"
```

---

### Task 6: Add ATLAS state to checkpoint save path

**Files:**
- Modify: `Backend/Machine Learning/Networks/checkpoint_persistence.cpp:1150-1760`

**Step 1: Add helper to enqueue ATLAS state tensors**

Add a static helper function near the top of the file (in the anonymous namespace) that takes a prefix string and a `const atlas::WeightState&` and pushes 4 tensor entries (U, fisherDiag, prevGz, meta) into a `tensorsToWrite` vector:

```cpp
static void enqueueAtlasWrite(std::vector<TensorWriteRef>& out,
                               const std::string& prefix,
                               const atlas::WeightState& st,
                               const std::string& dt)
{
	if (!st.initialized)
		return;
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(st.m));
		sh.push_back(static_cast<uint64_t>(st.r));
		out.push_back(TensorWriteRef(prefix + ".atlas.U", &st.U, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(st.r));
		out.push_back(TensorWriteRef(prefix + ".atlas.fisher", &st.fisherDiag, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(st.r));
		sh.push_back(static_cast<uint64_t>(st.n));
		out.push_back(TensorWriteRef(prefix + ".atlas.prevGz", &st.prevGz, dt, sh));
	}
	// Pack mu and step as 2 floats
	// NOTE: We need mutable storage for the meta vector.
	// The caller must keep atlasMeta alive until write completes.
}
```

Actually, the `meta` approach is problematic because `TensorWriteRef` takes `const std::vector<float>*`. Instead, store `mu` and `step` in the manifest as key-value pairs alongside the tensors. This is simpler and matches how `transformer.optimizerStep` is already stored.

**Revised approach:** For each ATLAS WeightState, save 3 tensors (U, fisherDiag, prevGz). Save `mu` and `step` as manifest key-value entries keyed by the tensor prefix.

Add ATLAS tensor entries after existing `includeOpt` blocks for each net type:
- DFF (after line 1177): For each transition `t`, call `enqueueAtlasWrite` with prefix `dff.t{t}`
- RNN (after line 1240): For each hidden `l`, call for `rnn.h{l}.Wxh`, `rnn.h{l}.Whh`, and `rnn.o`
- GRU/LSTM (after line 1309): For each hidden `l`, call for `{prefix}.h{l}.W`, `{prefix}.h{l}.U`, and `{prefix}.o`
- Transformer (after line 1427): For `tr.tokE`, `tr.WIn`, `tr.WOut`, and per-block `tr.b{l}.Wq/Wk/Wv/Wo/W1/W2`
- CNN: For each conv/FC layer

Also add ATLAS mu/step to the manifest key-value section.

**Step 2: Build to verify**

Run: `cd /home/rob/dev/glades-ml/build && cmake .. && make -j$(nproc)`

**Step 3: Commit**

```bash
git add "Backend/Machine Learning/Networks/checkpoint_persistence.cpp"
git commit -m "Add ATLAS state to checkpoint save path"
```

---

### Task 7: Add ATLAS state to checkpoint load path

**Files:**
- Modify: `Backend/Machine Learning/Networks/checkpoint_persistence.cpp:1940-2500`

**Step 1: Add helper to enqueue ATLAS state reads**

Mirror `enqueueAtlasWrite` with an `enqueueAtlasRead` that pushes `TensorReadRef` entries into `expected`, pointing at the mutable `WeightState` fields. After loading, set `initialized = true`.

```cpp
static void enqueueAtlasRead(std::vector<TensorReadRef>& out,
                              const std::string& prefix,
                              atlas::WeightState& st,
                              unsigned int m, unsigned int n, unsigned int r,
                              const std::string& dt)
{
	st.m = m;
	st.n = n;
	st.r = r;
	st.U.resize(static_cast<size_t>(m) * r);
	st.fisherDiag.resize(r);
	st.prevGz.resize(static_cast<size_t>(r) * n);
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(m));
		sh.push_back(static_cast<uint64_t>(r));
		out.push_back(TensorReadRef(prefix + ".atlas.U", &st.U, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(r));
		out.push_back(TensorReadRef(prefix + ".atlas.fisher", &st.fisherDiag, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(r));
		sh.push_back(static_cast<uint64_t>(n));
		out.push_back(TensorReadRef(prefix + ".atlas.prevGz", &st.prevGz, dt, sh));
	}
}
```

After all tensors load, restore `mu` and `step` from manifest KV, and set `initialized = true`.

**Step 2: Add ATLAS read entries for each net type**

Mirror the save path: after existing `includeOpt` reads for DFF, RNN, GRU/LSTM, Transformer, CNN, add `enqueueAtlasRead` calls with the same prefixes. Only add these entries when `includeOpt` is true AND `trainingConfig.optimizer.type == OptimizerConfig::ATLAS`.

**Step 3: Build and commit**

Run: `cd /home/rob/dev/glades-ml/build && cmake .. && make -j$(nproc)`

```bash
git add "Backend/Machine Learning/Networks/checkpoint_persistence.cpp"
git commit -m "Add ATLAS state to checkpoint load path"
```

---

### Task 8: Add ATLAS checkpoint save/load unit test

**Files:**
- Modify: `unit-tests/Backend/Machine Learning/atlas-test.cpp`

**Step 1: Add checkpoint round-trip test**

After the existing tests, add a test that:
1. Creates a DFF network with ATLAS optimizer
2. Trains for 50 epochs
3. Saves a checkpoint
4. Creates a new network, loads the checkpoint
5. Trains for 50 more epochs
6. Asserts training status is OK
7. Cleans up checkpoint files

**Step 2: Build and run**

Run: `cd /home/rob/dev/glades-ml/unit-tests/build && cmake .. && make -j$(nproc) && cd .. && bash test.sh atlas`
Expected: All tests pass.

**Step 3: Commit**

```bash
git add "unit-tests/Backend/Machine Learning/atlas-test.cpp"
git commit -m "Add ATLAS checkpoint save/load round-trip test"
```

---

### Task 9: Create ATLAS-vs-SGD benchmark test

**Files:**
- Create: `unit-tests/Backend/Machine Learning/atlas-bench.h`
- Create: `unit-tests/Backend/Machine Learning/atlas-bench.cpp`
- Modify: `unit-tests/main.cpp`
- Modify: `unit-tests/Backend/Machine Learning/CMakeLists.txt`
- Modify: `CLAUDE.md` (add `atlas-bench` to test list)

**Step 1: Create atlas-bench.h**

```cpp
#pragma once
void ATLASBenchmark(int argc, char* argv[]);
```

**Step 2: Create atlas-bench.cpp**

Follow the `nn-benchmarks.cpp` pattern:
- `now_ms()` timing helper (gettimeofday)
- `CaptureMetricsCallbacks` class
- Run each network type (DFF, RNN) with both SGD and ATLAS
- Use `datasets/rnn.csv` as default dataset
- Command-line args: `--epochs` (default 200), `--hidden` (default 8), `--rank` (default 4), `--repeats` (default 3)
- TSV output: Type, Optimizer, Train(ms), FinalLoss, R², Status

**Step 3: Register in main.cpp and CMakeLists**

Add `#include "Backend/Machine Learning/atlas-bench.h"` and dispatch `"atlas-bench"` to `ATLASBenchmark(argc, argv)`.

Add `atlas-bench.cpp` to `PCATests_src_files` in `unit-tests/Backend/Machine Learning/CMakeLists.txt`.

Add `atlas-bench` to the test names list in `CLAUDE.md`.

**Step 4: Build and run**

Run: `cd /home/rob/dev/glades-ml/unit-tests/build && cmake .. && make -j$(nproc) && cd .. && bash test.sh atlas-bench`
Expected: Benchmark runs, prints TSV results.

**Step 5: Commit**

```bash
git add "unit-tests/Backend/Machine Learning/atlas-bench.h" \
        "unit-tests/Backend/Machine Learning/atlas-bench.cpp" \
        "unit-tests/main.cpp" \
        "unit-tests/Backend/Machine Learning/CMakeLists.txt" \
        CLAUDE.md
git commit -m "Add ATLAS-vs-SGD comparison benchmark"
```

---

### Task 10: Run full test suite and verify no regressions

**Step 1: Run existing ATLAS tests**

Run: `cd /home/rob/dev/glades-ml/unit-tests && bash test.sh atlas`
Expected: All tests pass (6 network tests + checkpoint test).

**Step 2: Run existing save-load tests**

Run: `bash test.sh save-load`
Expected: All pass (no regressions in checkpoint code).

**Step 3: Run nn-recurrent tests**

Run: `bash test.sh nn-recurrent`
Expected: All pass (no regressions in RNN/GRU/LSTM SGD).

**Step 4: Run benchmark**

Run: `bash test.sh atlas-bench`
Expected: Benchmark completes with valid results.

**Step 5: Commit if any final fixes needed**
