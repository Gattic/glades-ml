## Determinism and concurrency policy (ML engine)

This document defines what “deterministic” means for this codebase, and what is (and is not)
safe to run concurrently.

### Determinism policy

The engine is **deterministic by default**.

- **RNG**
  - All ML randomness (weight init, dropout, etc.) must go through `glades::rng::*`.
  - `Trainer::run()` installs the network’s RNG engine (`NNetwork::rngEngine`) as the thread-local
    “current” RNG via `glades::rng::ScopedEngine`.
  - `NNetwork::setSeed(seed)` fully controls randomness for that network’s subsequent runs.
  - If you want different runs to be different, you must explicitly supply different seeds.

- **Data order**
  - Training and evaluation currently iterate data in a fixed order (no implicit shuffling).
  - If you later add shuffling, it must be driven by the per-network RNG and be seed-controlled.

- **Evaluation**
  - Evaluation (`RUN_TEST` / `RUN_VALIDATE`) must be side-effect free with respect to:
    - model parameters
    - training epoch counters
    - learning-rate schedule state
    - dropout masks (dropout is disabled in eval paths)

### Concurrency policy

- **Per-network non-reentrancy**
  - A single `glades::NNetwork` instance is **NOT re-entrant** and must not be used concurrently
    (e.g., calling `train()` and `test()` at the same time from different threads).
  - Enforced at runtime by a run-lock acquired in `Trainer::run()` (returns `INVALID_STATE` if violated).

- **Multiple networks**
  - Different `NNetwork` instances **may** be run concurrently on different threads.
  - RNG cross-talk is avoided because each thread uses its own thread-local “current RNG engine”.

- **DataInput thread-safety**
  - `DataInput` implementations are not guaranteed to be thread-safe.
    - Example: `ImageInput` maintains a mutable LRU cache inside `getTrainRow()` / `getTestRow()`.
  - If you run multiple networks in parallel, either:
    - give each thread its own `DataInput`, or
    - ensure the shared `DataInput` is externally synchronized, or
    - use `*RowView()` APIs that are explicitly documented as thread-safe for a specific implementation.

### Practical guidance

- **Reproducible experiment**: set explicit seeds and save them in your run metadata.
- **Parallel training**: create one `NNetwork` + one `DataInput` per thread.
- **Do not** rely on “random by default” behavior; in this engine you must opt in by choosing seeds.

