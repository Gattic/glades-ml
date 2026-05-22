# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current Production Flagship — CHIRON 1B @ T=16384 (v5+FP8 ship 2026-05-22)

The current production LLM flagship is **CHIRON 1B v5+FP8 stack**
(checkpoint `chiron_1B_T16384_v5_fp8_phase2.final`):

- **Shape**: m=2048, L=24, nH=16, dH=256, V=32000 BPE, T=16384 context.
- **Params**: 870.94M (~"1B").
- **Stack**: iter 116 ship 10-mechanism stack (see "Prior iter 116 flagship"
  below for the full mechanism list) **PLUS two new layers landed 2026-05-21
  / 2026-05-22**:
  - **CUDA 13.2 toolchain** (was CUDA 12.0): cuBLAS 13.x BF16 GEMM dispatch.
  - **v5 BF16-cast fix** in `Backend/Machine Learning/Networks/cuda/gpu_blas.cu`:
    `sgemm_rowmajor_fast16bf_impl` pre-casts FP32 inputs to BF16 scratch +
    dispatches `cublasGemmEx` with `CUDA_R_16BF` inputs. Forces cuBLAS 13's
    `ampere_s1688gemm_bf16_*` fast path on Ada (without this, cuBLAS 13's
    heuristic for FP32-input FAST_16BF dispatches a non-BF16-specialized
    kernel ~1.82× slower per call on iter 116 ship shapes — a -15.18%
    regression vs CUDA 12.0 if unfixed).
  - **`scfa_B` constant cache**: trainer pre-casts the DCT-II basis to BF16
    once at SCFA init and registers it via the new `register_fast16bf_constant`
    API. The 216 SCFA outer GEMMs/step skip the per-call A-side cast.
  - **`--fp8-readout-fwd` enabled**: 3 readout GEMMs/step route through
    cuBLASLt FP8 (E4M3) at shape (T=16384, V=32000, m=2048) ≈ 1 TFLOP each.
    Adds +1.59% wall on top of v5 fix at strict NLL parity. Unblocked by
    CUDA 13.2 cuBLASLt FP8 algo coverage on Ada (was blocked on CUDA 12.0
    per `ITER62_FP8_READOUT_NULL.md`).
- **Perf**: **28,887 tok/s** @ T=16384 (was 28,257 at iter 116 ship,
  **+2.23%**; 14.97/15.56 GB VRAM, same as iter 116 ship), **final val NLL
  4.1717 @ step 30000** (iter 94 baseline at same recipe gives 4.1983; Δ
  **−0.0266 nat BETTER**, far within strict ±0.02). Mean trajectory drift
  vs iter 116 ship across 10 val checkpoints: +0.00007 nat (essentially
  zero). Cumulative since pre-ralph-loop 15,200 tok/s: **1.90×**.
- **Reproduce training**: `cd ~/dev/glades-trainer && sh run.sh flagship`
  (10 iter 116 mechanisms default; **append `--fp8-readout-fwd`** until the
  flag is added to the flagship recipe in run.sh — or invoke
  `./build/glades_chiron_train` directly per the Reproduction block in
  `research/V5_FP8_30K_PHASE2_PASS_2026_05_22.md`).
- **Run inference**: `cd ~/dev/glades-trainer && sh runner.sh --flagship`.
- **Full spec**: `research/V5_FP8_30K_PHASE2_PASS_2026_05_22.md` (Phase 2
  ship doc with full trajectory table) and `research/CUDA13_BF16_REGRESSION_FIX_2026_05_21.md`
  (diagnosis + fix details). Also see `research/FLAGSHIP_T16384_2026_05_14.md`
  for the iter 116 ship 10-mechanism details that v5+FP8 builds on.
- **Phase 2 evidence**: `research/V5_FP8_30K_PHASE2_PASS_2026_05_22.md`
  (apples-to-apples 30k retrain vs iter 116 ship & iter 94 triple-stack
  baselines; ship-clean PASS at +2.23% wall + −0.0322 nat NLL improvement
  vs iter 116 ship).
- **FP8 path evidence**: `research/FP8_READOUT_CUDA13_REVAL_2026_05_21.md`
  (n=3 multi-seed at 200 steps showing +1.38% standalone wall on CUDA 13.2,
  strict NLL parity).
- **Per-mechanism evidence**: `research/ITER<N>_*.md` for N ∈ {97, 99, 100,
  101, 103, 106, 107, 109, 113, 115, 116} (the iter 116 ship's 10 mechanisms).
- **Phase-3 program** (improving CHIRON 1B via novel research, no external
  libs / no external baselines): pre-registration in
  `research/PHASE3_GATE3A_PREREG.md`. Any new architecture work should
  anchor on the v5+FP8 flagship as the baseline.

**Prior iter 116 ship** (`chiron_1B_T16384_iter116_treatment_phase2.final`,
28,257 tok/s, NLL 4.2039 @ 30k, on CUDA 12.0) is archived at
`database/checkpoints/chiron_1B_T16384_iter116_treatment_phase2/` and remains
loadable into the v5+FP8 flagship (math is bit-identical at single-element
FP32 between iter 116 ship and v5; v5+FP8 adds a bounded FP8 readout drift
of ~0.005-0.007 nat per iter 62 design). The iter 116 ship was the CUDA 12.0
production flagship; v5+FP8 ships the CUDA 13.2 equivalent + FP8 readout.

**Prior iter 94 flagship** (`chiron_1B_T16384_triple_phase2.final`, 25,103
tok/s, NLL 4.1983) is archived at `database/checkpoints/chiron_1B_T16384_triple_phase2/`
and remains loadable. The iter 94 baseline
`chiron_1B_T16384_baseline_phase2.final` (24,246 tok/s, NLL 4.2228) is
the pre-w=4 archive at `database/checkpoints/chiron_1B_T16384_baseline_phase2/`.

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
