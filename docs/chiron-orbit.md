# CHIRON ORBIT optimizer

ORBIT is a trainer-only, opt-in replacement for the per-matrix second moment of
CHIRON's flagship AdamW path. It keeps the validated global L2 clipping contract
(`--grad-clip 0.5`) and stock Adam for small vectors while using int8 momentum
and CHIRON-derived factored metrics for `E`, `Wq`, `Wk`, `Wv`, and `Wo`.

The normative design and experiment bars are in
[`superpowers/specs/2026-07-09-chiron-native-optimizers-design.md`](superpowers/specs/2026-07-09-chiron-native-optimizers-design.md).

## Metric ownership

| Matrix | row factor | column factor |
|---|---|---|
| `Wq/Wk/Wv [m,dModel]` | gradient row square mean | gradient column square mean |
| `Wo [dModel,m]` | strided inner activation square mean | shared CHIRON adjoint energy |
| tied `E [V,m]` | readout occupancy plus input frequency × `dq` energy | full accumulated `dE` column square mean |

Every factor is an FP32 EMA. Factors are normalized to mean one before use, and
the tensor scalar `S` supplies the overall second-moment scale. A deterministic
quantile floor prevents near-zero factor entries. The first `orbitFreeze` steps
use identity shape factors while retaining the scalar metric.

Backward hooks write only transient samples. Persistent factors advance in the
accepted optimizer-step branch, after the finite-gradient guard, so skipped
steps do not age the metric. Global clipping is applied before factor updates,
momentum updates, and parameter updates.

## CUDA and CPU interfaces

- CPU references/accounting: `Backend/Machine Learning/Networks/chiron_orbit.{h,cpp}`
- CUDA kernels: `Backend/Machine Learning/Networks/cuda/gpu_orbit.{h,cu}`
- Unit tests: `unit-tests/Backend/Machine Learning/chiron-orbit-test.cpp`

Run focused checks:

```bash
cd unit-tests/build
cmake .. -DGLADES_ENABLE_CUDA=ON
cmake --build . -j2
./glades-unit-tests chiron-orbit
./glades-unit-tests chiron-orbit-bench
```

The tests cover CPU/CUDA factor parity, FP32 and BF16 gradients, all five factor
families, accepted-step EMA semantics, factored update math, delayed cap math,
deterministic BF16 stochastic writeback, correlation, and production-shape
state accounting.

## Trainer flags

ORBIT defaults off. The sibling `glades-trainer/run.sh flagship` wrapper exposes:

```text
--orbit
--orbit-beta-f F       (default 0.98)
--orbit-freeze N       (default 200)
--orbit-damp-q F       (default 0.1)
--orbit-delta F        (default 0, delayed cap disabled)
--[no-]orbit-e-fisher  (default enabled)
--orbit-kappa F        (default 1.0)
--orbit-act-stride N   (default 16)
--orbit-wd-metric
--orbit-config-smoke
--orbit-memory-preview
```

The production interlock requires int8 Adam, BF16 weights and matrix gradients,
BF16 logits storage, SCFA, final-only `fuse-attn-reln`, fixed depth/length, and
enabled global clipping. ORBIT deliberately rejects incompatible combinations.

## Checkpoints and diagnostics

Full CHRF checkpoints set `CKPT_BIT_ORBIT_STATE` (`8192`). They persist matrix
momentum/code scales, factor EMAs and sums, accepted-step/clip counters, delayed
cap state, and exact ORBIT configuration. Resume fails if runtime ORBIT enablement or configuration differs
from the checkpoint. The ORBIT section precedes trainer model tails so the
existing EOF serving-tail contract remains intact.

At `--log-every`, the trainer emits per-class (`E`, `QKV`, `Wo`) update RMS,
weight RMS, normalized factor range, correlation proxy, delayed metric length,
and global clip rate. Allocation emits baseline/ORBIT byte counts and savings.

Trainer checkpoint verification:

```bash
cd ../glades-trainer
./build/glades_chiron_train --checkpoint-self-test
./run.sh flagship --orbit --orbit-config-smoke
```

## Gate record (2026-07-16)

- E1 correctness passed, including exact ORBIT checkpoint roundtrip and strict
  configuration-mismatch rejection.
- Production state accounting is `1.648 GiB -> 0.827 GiB` (`0.5019x`). A matched
  20-step T=16384 probe measured `45.8 s` AdamW versus `45.7 s` ORBIT.
- The pre-registered E2 `L=4,m=256,T=1024,500` comparison did **not** pass:
  final validation was `9.2299` AdamW versus `9.3126` ORBIT; final correlation
  proxies were `0.5653` for Wo (target `>0.7`) and `0.2110` for QKV (target
  `>0.5`). There were zero gradient skips.
- Consequently E3/E4 were not launched. ORBIT remains implemented and default
  off, but the experiment is a no-go unless a new, separately pre-registered
  hypothesis justifies reopening it.
