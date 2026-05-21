# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current Production Flagship — CHIRON 1B @ T=16384 (iter 94 triple-stack ship 2026-05-20)

The current production LLM flagship is **CHIRON 1B triple-stack**
(checkpoint `chiron_1B_T16384_triple_phase2.final`):

- **Shape**: m=2048, L=24, nH=16, dH=256, V=32000 BPE, T=16384 context.
- **Params**: 870.94M (~"1B").
- **Stack**: SCFA (spectral-compressed flash attention, ratio=16, k=1024,
  **conv-w=4** ← was 8 prior to iter 94) +
  BF16 weights/grads/attn/logits-storage + int8-Adam + fuse-attn-reln +
  **iter70-fused-axpy2-dual-p** (fused SCFA shear + BF16-p mirror cast) +
  **iter73-dwconv-fwd-tiled** (shared-mem tiled depthwise backward) +
  **scfa-checkpoint-inner-bf16** (BF16 7-tensor cache, 1.69 GB). The
  exact training flags are in `glades-trainer/research/run_postfix_experiments.sh`;
  these four are now code defaults in `chiron_main.cpp`.
- **Perf**: **25,103 tok/s** @ T=16384 (was 20,108 pre-iter94, **+24.8%**),
  13.22/15.56 GB VRAM (RTX 4080 SUPER 16 GB),
  **final val NLL 4.20 @ step 30000** (apples-to-apples baseline at same
  recipe gives 4.22; best EMA train loss ~3.80, comparable to prior 3.77).
- **Reproduce training**: `cd ~/dev/glades-trainer && sh run.sh flagship`
  (no extra flags needed — triple-stack is the default).
- **Run inference**: `cd ~/dev/glades-trainer && sh runner.sh --flagship`.
- **Full spec**: `research/FLAGSHIP_T16384_2026_05_14.md` (see "iter 94 ship"
  section at bottom for triple-stack details).
- **Phase 2 evidence**: `research/ITER94_30K_PHASE2_PASS.md` (apples-to-apples
  30k retrain at production recipe; ship-clean PASS).
- **Phase-3 program** (improving CHIRON 1B via novel research, no external
  libs / no external baselines): pre-registration in
  `research/PHASE3_GATE3A_PREREG.md`. Any new architecture work should
  anchor on the triple-stack flagship as the baseline.

**Prior w=8 flagship** (`chiron_1B_T16384.step30000`, 20,108 tok/s, best EMA
3.77) is documented in the same FLAGSHIP doc and archived as the Phase 2
baseline at `database/checkpoints/chiron_1B_T16384_baseline_phase2/`. It is
NOT loadable into the new flagship (filter dimensions differ at conv_w=4 vs
prior 8 — math change requires retrain, which Phase 2 IS).

The repository also contains a separate **CHIRON-stack research line** under
`run.sh chiron --scale {66M..1.84B}` using FACE + MFIO + WIP + SAS + RLG +
SLC. That is NOT the production flagship — it's the FACE-optimizer +
curriculum-scaling research program. When in doubt, default to the
`flagship` recipe above for new training runs.

## Build Commands

**Build the library** (from project root):
```bash
sh .configure.sh
```

**Build with CUDA**:
```bash
sh .configure.sh cuda
```

**Build and run unit tests**:
```bash
cd unit-tests/build && sh .configure.sh
cd unit-tests/build && sh .configure.sh cuda # compile with cuda
cd unit-tests && bash test.sh nnall    # run all tests
```

**Available single test names**: `nn`, `nn-recurrent`, `nn-transformer`, `transformer-serving` (or `serving`), `nn-bench`, `pca`, `kmeans`, `bayes`, `bayes-optimizer`, `bayes-optimizer-nd`, `ohe`, `mapped`, `cv`, `save-load`, `nn-mixed-precision` (or `nn-mp`), `prop-fuzz`, `parallel`, `ddp`, `transformer-improvements` (or `ti`), `gpu-training`, `cnn`, `cnn-mnist`, `garch`, `egarch`, `gan`, `search-space`, `hp-tuner`, `bayes-lr`, `hp-tuner-full`, `atlas`, `atlas-bench`

**Install**: `cd build && make install` (installs to `~/.local`)

## Project Overview

C++98 machine learning library built as a shared library (`libglades.so`). Depends on the `shmea` library (database/networking, installed separately). Uses CMake 3.10+, Release mode with `-O3 -march=native`.

The unit tests have their own CMake project under `unit-tests/` with a separate `build/` directory. Tests link against the in-tree build of glades (not the installed copy). The CMakeCache in each build dir is sticky — delete it when changing build type or flags.

## Architecture

### Library Targets (built by `Backend/Machine Learning/CMakeLists.txt`)
- **ML** — top-level ML module, links all sub-targets below
- **Networks** — all neural network implementations (forward/backward, training, inference, persistence)
- **MLStructure** — network architecture definitions (`NNInfo`, `LayerInfo` variants)
- **MLState** — training state management (`Terminator`)
- **DataObjects** — data input abstractions (`NumberInput`, `ImageInput`, `TokenInput`, `MappedMatrix`)
- **GMath** — math utilities (`PCA`, `KMeans`, `GARCH`, `OHE`, `CMatrix`)

### Network Types
Defined in `Backend/Machine Learning/Networks/network.h` as `TYPE_DFF`, `TYPE_RNN`, `TYPE_GRU`, `TYPE_LSTM`, `TYPE_TRANSFORMER_ENCODER`, `TYPE_TRANSFORMER_DECODER`, `TYPE_CNN`. GANs are a separate class wrapping these.

Each network type has a dedicated SGD implementation file: `sgd_dff.cpp`, `sgd_rnn.cpp`, `sgd_gru.cpp`, `sgd_lstm.cpp`, `sgd_cnn.cpp`, `sgd_transformer.cpp`.

### Transformer Stack
- `sgd_transformer.cpp` — training (backward passes, gradient accumulation)
- `transformer_infer.cpp` — forward-only inference
- `transformer_generate.cpp` — token generation and sampling
- `transformer_ops.h` — attention kernels, softmax, activation functions (GELU, SiLU, ReLU)
- `transformer_kernels.h` — SIMD-optimized math (`dot_f32`, `axpy_f32`) with scalar fallbacks
- `transformer_ops.h` includes `transformer_kernels.h` for SIMD helpers
- `training_config.h` — transformer configuration (RoPE/sinusoidal, LayerNorm/RMSNorm, MLP/SwiGLU, KV-cache dtypes)
- `transformer_public_api.h` — stable C++98 wrapper for generation APIs

### Public API Entry Point
`Backend/Machine Learning/main.h` defines the `glades` namespace with `train()`, `test()`, `trainOwned()`, `testOwned()` overloads. `glades::init()` must be called first.

### Determinism
The engine is deterministic by default. All randomness goes through `glades::rng::*` with per-network RNG engines controlled by `NNetwork::setSeed()`. See `Backend/Machine Learning/DETERMINISM_AND_CONCURRENCY.md` for the full policy.

### Test Framework
Custom assertion macro `ASSERT(failmsg, predicate)` defined in `unit-tests/unit-test.h`. Each test suite is a standalone function (e.g., `NNUnitTest()`, `PCAUnitTest()`) registered in `unit-tests/main.cpp`. Test source files live under `unit-tests/Backend/Machine Learning/`.
