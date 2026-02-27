# GAN Per-Sample Data Parallelism Design

## Goal

Parallelize GAN training across 20+ CPUs to reduce wall-clock time on large datasets.
All GAN variants (Vanilla, WGAN-GP, InfoGAN, StyleGAN, CycleGAN, and composites) are in scope.

## Approach: Per-Sample Data Parallelism

Parallelize the inner sample loops in `trainSingleDomain` and `trainDualDomain`.
Each thread processes a chunk of samples from the batch, accumulating gradients into
thread-local buffers. After the loop, gradients are reduced in thread order for
deterministic results.

## Architecture

```
Pre-generate (sequential)      Parallel compute             Ordered reduce
 - sample noise[0..B]    -->   Thread 0: samples [0..k)     --> sum grads t0,t1,...
 - sample codes[0..B]          Thread 1: samples [k..2k)    --> sum losses
 - fetch data ptrs[0..B]       Thread N: samples [...B)      --> scale + Adam update
```

Forward functions read shared weights (thread-safe).
Backward functions accumulate into per-thread GradientBuffers (no races).
Existing ThreadPool (`glades_thread_pool.h`) is reused.

## Key Data Structures

### GradientBuffer

Mirrors the shape of gradient arrays in NNetwork. Supports DFF, CNN, and Deconv layouts.

```cpp
struct GradientBuffer {
    // DFF: gW[transition], gBias[transition]
    std::vector<std::vector<float>> gW, gBias;
    // CNN conv: convGW[layer], convGBias[layer]
    std::vector<std::vector<float>> convGW, convGBias;
    // CNN FC: fcGW[layer], fcGBias[layer]
    std::vector<std::vector<float>> fcGW, fcGBias;
    // Deconv
    std::vector<float> deconvFcGW, deconvFcGBias;
    std::vector<std::vector<float>> deconvGW, deconvGBias;

    void initFrom(const NNetwork& net);  // allocate to match network shapes
    void zero();                          // memset all arrays to 0
    void addTo(NNetwork& net);            // net.gW[i] += this->gW[i]
};
```

### GANThreadCtx

Per-thread context holding scratch buffers and gradient accumulators.

```cpp
struct GANThreadCtx {
    GradientBuffer genGrads, discGrads;
    GradientBuffer qGrads;          // InfoGAN Q-head
    GradientBuffer mappingGrads;    // StyleGAN mapping net
    // StyleAffine + LayerNorm gradient vectors

    // Scratch (activation buffers for forward/backward)
    std::vector<std::vector<float>> genAct, discActReal, discActFake;
    std::vector<float> cnnOutReal, cnnOutFake;
    std::vector<float> deconvOut;
    std::vector<std::vector<float>> deconvScratch;
    std::vector<float> dFake, genInput;
    // Style/Info scratch as needed

    // Loss accumulators
    float dLossReal, dLossFake, gLoss, wasserstein, infoLoss;
    unsigned int catCorrect, catTotal;
};
```

## RNG & Data Pre-Loading

Before each parallel sample loop (sequential, deterministic):

1. **Noise**: Pre-sample `preNoise[curBatchSize]` from `rngEngine` in sample order.
2. **Latent codes**: Pre-sample `preCatCode[curBatchSize]`, `preContCode[curBatchSize]` (InfoGAN).
3. **Data pointers**: Pre-fetch `preReal[curBatchSize]` via `getTrainRowView()`.

This preserves RNG determinism (same draw order as sequential code) and solves
ImageInput's thread-unsafe LRU cache by doing all cache mutations before parallelism.

## Backward Function Changes

Add overloads to `dffBackward`, `cnnBackward`, `deconvBackward` that accept a
`GradientBuffer*` parameter. When non-NULL, gradients accumulate into the buffer
instead of the network's own `gW`/`gBias`. Existing overloads (NULL default)
remain unchanged for single-threaded callers.

## Parallel Loop (Discriminator Phase)

```
1. Pre-generate noise, codes, data pointers (sequential)
2. Zero per-thread gradient buffers
3. parallel_for(curBatchSize):
     tid = derive from chunk assignment
     for each sample in chunk:
       build genInput from preNoise[s] + preCodes[s]
       generator forward (shared weights, thread-local activations)
       discriminator forward on preReal[s] and fake
       compute loss -> thread-local accumulators
       backward through discriminator -> thread-local discGrads
       InfoGAN Q-head backward -> thread-local qGrads (if enabled)
4. Zero master network gradients
5. Ordered reduction: for t=0..N-1: threadCtx[t].discGrads.addTo(discriminator)
6. Scale gradients by 1/curBatchSize + Adam update (unchanged)
```

Generator phase follows the same pattern. CycleGAN's `trainDualDomain` applies
the identical pattern to all four sample loops (D_A, D_B, G_AB, G_BA).

## Determinism

- Default: `deterministicReduce = true` in GANConfig.
- Ordered reduction (thread 0, 1, 2...) produces identical floating-point results
  regardless of thread count, given the same seed.
- When `false`: threads accumulate directly into master grads (faster, non-deterministic
  due to float addition order).

## Files Changed

| File | Change |
|------|--------|
| `gan.h` | Add `GradientBuffer`, `GANThreadCtx`; add `deterministicReduce` to GANConfig; backward overloads |
| `gan.cpp` | Parallelize sample loops in `trainSingleDomain`/`trainDualDomain`; implement GradientBuffer; backward overloads |
| `network.h` | (Optional) helper to iterate gradient arrays generically |
| `gan-test.cpp` | Determinism tests, correctness tests, scaling benchmarks |

No changes to ThreadPool, DataInput, ImageInput, or other existing code.

## Testing

1. **Determinism**: Same seed, 1 thread vs N threads, `deterministicReduce=true`.
   Assert losses match at every epoch.
2. **Correctness**: Compare generated samples between single/multi-threaded training.
   Should be bit-identical in deterministic mode.
3. **All variants**: Vanilla DFF, WGAN-GP CNN, InfoGAN, StyleGAN, CycleGAN, composites.
4. **Scaling**: Benchmark 1/4/8/16/20 threads, verify near-linear speedup.
