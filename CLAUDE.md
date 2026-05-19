# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current Production Flagship — CHIRON 1B @ T=16384

The current production LLM flagship is **CHIRON 1B** (checkpoint
`chiron_1B_T16384.step30000`):

- **Shape**: m=2048, L=24, nH=16, dH=256, V=32000 BPE, T=16384 context.
- **Params**: 870.94M (~"1B").
- **Stack**: SCFA (spectral-compressed flash attention, ratio=16, k=1024) +
  BF16 weights/grads/attn/logits-storage + int8-Adam + fuse-attn-reln. The
  exact training flags are in `glades-trainer/research/run_postfix_experiments.sh`.
- **Perf**: 20,108 tok/s @ T=16384, 13.22/15.56 GB VRAM (RTX 4080 SUPER 16 GB),
  best val NLL 3.77 @ step 29341 / 30k steps / 491.5 M tokens trained.
- **Reproduce training**: `cd ~/dev/glades-trainer && sh run.sh flagship`.
- **Run inference**: `cd ~/dev/glades-trainer && sh runner.sh --flagship`.
- **Full spec**: `research/FLAGSHIP_T16384_2026_05_14.md`.
- **Phase-3 program** (improving CHIRON 1B via novel research, no external
  libs / no external baselines): pre-registration in
  `research/PHASE3_GATE3A_PREREG.md`. Any new architecture work should
  anchor on this flagship as the baseline.

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
