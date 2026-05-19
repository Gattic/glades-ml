# Subsystem Execution Program: Transformer Token-LM Implementation

## Target Resolution
- Requested target: `LLM Implementation`
- Resolved target: `Transformer token-LM implementation` across training, inference, one-shot generation, serving, token-id data intake, tokenizer artifacts, and checkpoint/model packaging
- Why this match: the repo uses transformer/token-LM terminology rather than `LLM`, and the implementation locus is concentrated in `Backend/Machine Learning/Networks`, `Backend/Machine Learning/DataObjects/TokenInput.*`, and the transformer-specific unit-test aliases in `unit-tests/main.cpp`
- Planning boundary / what was excluded: excluded DFF/RNN/GRU/LSTM/CNN/GAN/ATLAS feature work, generic `GMath`, datasets/fixtures outside transformer-facing tests, and docs except where they provide build/test evidence

## Planning Strategy
- This subsystem is too large for one critique-style plan: core behavior is split across oversized hotspot files such as `network.h`, `network.cpp`, `transformer_infer.cpp`, `transformer_generate.cpp`, `sgd_transformer.cpp`, `checkpoint_persistence.cpp`, and `unit-tests/Backend/Machine Learning/nn-test.cpp`
- The decomposition seams are: token-LM contract and token data, inference/session + one-shot generation, continuous batching + serving wrapper, training/numerics + integration, and persistence/package/checkpoint workflows
- The intended batch order is foundational contracts first, runtime next, serving after runtime, training/integration after contracts stabilize, and persistence last so on-disk contracts are not moving while core behavior is still being re-sliced
- The main sequencing risk is that token-id semantics, run-lock policy, and transformer config validation underpin both runtime and training; changing those late would invalidate later batches and inflate merge risk in `network.h` and `network.cpp`

## Batch Overview
- Batch 1 - Contract and Data Boundary: stabilize token-LM config, public shim, token-id data, and trainer preflight seams. Entry: only the current repo state and verified build/test commands. Exit: token/data/config contracts are explicit enough that runtime and training batches can move without re-deciding token-LM invariants.
- Batch 2 - Inference and Generation Core: extract stable KV-session and one-shot generation seams without touching the serving wrapper. Entry: Batch 1 invariants hold for token-id inputs and config validation. Exit: single-request and batched session logic have explicit boundaries and parity gates.
- Batch 3 - Serving Runtime: isolate the persistent batcher and synchronized serving wrapper on top of the Batch 2 runtime core. Entry: Batch 2 session/generation behavior is stable. Exit: serving callback, snapshot, and slot-lifecycle code can evolve without reopening infer/generate internals.
- Batch 4 - Training and Integration: split transformer training, numerics helpers, and parallel/CUDA integration into reviewable seams. Entry: Batch 1 contract work is complete and runtime behavior is stable enough to keep training changes focused. Exit: `sgd_transformer.cpp` and GPU handoff paths are no longer monolithic blockers.
- Batch 5 - Persistence and Packaging: isolate model package, tokenizer artifact, checkpoint, and serializer responsibilities. Entry: earlier batches have stabilized runtime/training contracts that persistence serializes. Exit: package/checkpoint work is independently maintainable and validated by transformer-specific save/load tests.

## Shared Context
- Build command: `sh .configure.sh && cd unit-tests && sh .configure.sh`; CUDA variant when needed: `sh .configure.sh cuda && cd unit-tests && sh .configure.sh cuda`
- Test command: `cd unit-tests && bash test.sh nn-transformer && bash test.sh transformer-serving && bash test.sh transformer-improvements && bash test.sh save-load && bash test.sh parallel && bash test.sh transformer-grad && bash test.sh sampling && bash test.sh transformer-ops && bash test.sh transformer-kernels && bash test.sh attention-bwd`; add `bash test.sh ddp` and `bash test.sh gpu-training` only when the batch touches those paths and both trees were rebuilt with CUDA
- Global invariants to preserve: token IDs remain first-class ints via `TokenInput`; token-LM config stays fail-fast and non-re-entrant; decoder full-forward logits stay parity-safe with KV-cache paths across RoPE/GQA/typed-KV modes; serving callbacks keep mutex-release-before-callback and reject re-entrant `step()`; save/load/checkpoint flows keep tokenizer and transformer metadata consistent
- Cross-batch hotspots: `Backend/Machine Learning/Networks/network.h`, `Backend/Machine Learning/Networks/network.cpp`, `Backend/Machine Learning/Networks/transformer_infer.cpp`, `Backend/Machine Learning/Networks/transformer_generate.cpp`, `Backend/Machine Learning/Networks/sgd_transformer.cpp`, `Backend/Machine Learning/Networks/checkpoint_persistence.cpp`, `unit-tests/Backend/Machine Learning/nn-test.cpp`
- Global adjacent work that should not go through `/execute-plan`: `/test-design transformer verification matrix` - cover `gradientCheckpointing`, `tokenLmAllowHugeFullSoftmax`, fuzz-harness, and CUDA/DDP gaps; `/perf transformer infer/generate hot loops` - profile before kernel-oriented rewrites; `/observability transformer serving and package publish failures` - keep telemetry separate from structural refactors

## Batch 1 - Contract and Data Boundary
- Goal: stabilize token-LM config validation, public transformer-facing shims, token-id ingestion, and trainer preflight so later batches can assume one contract.
- Why now: `TokenInput`, `TransformerRunConfig`, and `Trainer::run` preflight are shared dependencies for both inference and training, and they are the least risky extraction seams.
- Main risks: `network.h` is a chronic merge hotspot; stale build docs can mislead verification; token-id split/pad semantics can silently break perplexity, save/load gating, or `setTrainingConfig` rejection behavior.
- Merge/conflict notes: allow `transformer_config.*`, `transformer_public_api.h`, `transformer_types.h`, `TokenInput.*`, `trainer.cpp`, and only targeted `network.h`/`network.cpp` touch points; do not mix this batch with `transformer_infer.cpp`, `transformer_generate.cpp`, or persistence bodies.
- Stop condition before next batch: `nn-transformer`, `transformer-improvements`, and `save-load` pass for token-input/config/package contract paths, and the unit-test build path is still `cd unit-tests && sh .configure.sh`.

## Action Plan

### Build commands
build: `sh .configure.sh && cd unit-tests && sh .configure.sh`
test: `cd unit-tests && bash test.sh nn-transformer && bash test.sh transformer-improvements && bash test.sh save-load`

### Phase 1 — Foundation
1. `/refactor transformer config snapshot boundary in Backend/Machine Learning/Networks/transformer_config.{h,cpp}`
   - Addresses: `LLM1-1` — unify token-LM config validation and snapshot building used by init and runtime mutation
   - Rationale: `buildTransformerModelConfigSnapshot` already gates both parameter init and `setTrainingConfig`, so it is the safest shared seam to stabilize first.
   - Key files: `Backend/Machine Learning/Networks/transformer_config.h`, `Backend/Machine Learning/Networks/transformer_config.cpp`
   - Scope: Medium
   - Callers: `Backend/Machine Learning/Networks/network.cpp:1363`, `Backend/Machine Learning/Networks/network.cpp:2850`
   - Tests: `unit-tests/Backend/Machine Learning/nn-test.cpp` config rejection near `3338` and `4355`; `unit-tests/Backend/Machine Learning/transformer-improvements-test.cpp` `745` save/load follow-on
   - Conventions: C++98, explicit `NNetworkStatus`, fail-fast validation, preserve current snapshot defaults
2. `/refactor transformer public API/type alias boundary in Backend/Machine Learning/Networks/transformer_public_api.h, transformer_types.h, network.h, and network.cpp`
   - Addresses: `LLM1-2` — keep freestanding transformer types/public wrappers separate from `NNetwork` aliases and pass-through shims
   - Rationale: the wrappers are already thin and can be cleaned up without reopening runtime math.
   - Key files: `Backend/Machine Learning/Networks/transformer_public_api.h`, `Backend/Machine Learning/Networks/transformer_types.h`
   - Scope: Medium
   - Callers: `Backend/Machine Learning/Networks/network.cpp:202`, `Backend/Machine Learning/Networks/network.cpp:257`
   - Tests: `unit-tests/Backend/Machine Learning/nn-test.cpp` facade-forward parity near `3990`; gap: no facade-only alias target
   - Conventions: keep `NNetwork` compatibility, const-correct wrappers, no unexpected ABI drift, avoid pulling heavy headers into light callers
3. `/refactor TokenInput path-import vs table-import split in Backend/Machine Learning/DataObjects/TokenInput.{h,cpp}`
   - Addresses: `LLM1-3` — make token-id ingestion, split semantics, and pad-token rules explicit
   - Rationale: token LM training and later package/checkpoint work depend on `TokenInput` remaining the authoritative integer-token contract.
   - Key files: `Backend/Machine Learning/DataObjects/TokenInput.h`, `Backend/Machine Learning/DataObjects/TokenInput.cpp`
   - Scope: Medium
   - Callers: `Backend/Machine Learning/Networks/trainer.cpp:101`, `Backend/Machine Learning/Networks/network.cpp:2854`
   - Tests: `unit-tests/Backend/Machine Learning/nn-test.cpp` `3635`; `unit-tests/Backend/Machine Learning/transformer-improvements-test.cpp` `842`
   - Conventions: token IDs are first-class ints, preserve sequence spans, no float-row materialization, explicit train/test split semantics

### Phase 2 — Structural improvement
1. `/refactor transformer run preflight and token-id gating in Backend/Machine Learning/Networks/trainer.cpp`
   - Addresses: `LLM1-4` — isolate token-LM/sequence/data-shape preflight from the rest of `Trainer::run`
   - Rationale: every later batch relies on stable `hasTokenIdInput`, sequence validity, and row-shape contracts before any model code runs.
   - Key files: `Backend/Machine Learning/Networks/trainer.cpp`, `Backend/Machine Learning/Networks/trainer.h`
   - Scope: Medium
   - Callers: `Backend/Machine Learning/Networks/network.h:1330`, `Backend/Machine Learning/Networks/network.h:1332`
   - Tests: `unit-tests/Backend/Machine Learning/nn-test.cpp` `3282`, `3373`, `3621`; gap: no standalone trainer-preflight alias
   - Conventions: run-lock first, explicit `NNetworkStatus`, no stdout side effects, no retained `DataInput` ownership

### Deferred
- Runtime session extraction and sampling semantics move to Batch 2.
- Serving wrapper and persistent batcher work move to Batch 3.
- Package/checkpoint format cleanup moves to Batch 5.

### Adjacent work (not refactor)
- `/test-design transformer config contract` - add explicit coverage for `tokenLmAllowHugeFullSoftmax` and `gradientCheckpointing`
- `/observability trainer preflight failures` - separate status/logging improvements from structural cleanup

## Batch 2 - Inference and Generation Core
- Goal: extract stable KV-session, token-step, full-forward, and one-shot generation seams without touching the serving adapter.
- Why now: both persistent batching and serving wrappers depend on `transformer_infer.cpp` and `transformer_generate.cpp`, so their core session/generation boundaries must settle first.
- Main risks: `transformer_infer.cpp` owns typed-KV limits, RoPE/GQA parity, and full-forward debug paths in one hotspot; sampling semantics are easy to regress if mixed with serving work.
- Merge/conflict notes: keep `transformer_serving_layer.*`, persistence, and `sgd_transformer.cpp` out of this batch; allow targeted `nn-test.cpp`, `parallel-test.cpp`, and `fuzz_transformer_infer.cpp` fixture edits only for runtime verification.
- Stop condition before next batch: `nn-transformer`, `sampling`, and `parallel` pass for KV parity, API guard, and deterministic generation cases; any optional fuzz harness still compiles manually.

## Action Plan

### Build commands
build: `sh .configure.sh && cd unit-tests && sh .configure.sh`
test: `cd unit-tests && bash test.sh nn-transformer && bash test.sh sampling && bash test.sh parallel`

### Phase 1 — Foundation
1. `/refactor TransformerLmSession and TransformerLmBatchSession contracts in Backend/Machine Learning/Networks/network.h and transformer_infer.cpp`
   - Addresses: `LLM2-1` — make session reset/init sizing and state invariants explicit across single and batched KV caches
   - Rationale: generation and serving both reuse these session objects, so they must stabilize before higher-level schedulers.
   - Key files: `Backend/Machine Learning/Networks/network.h`, `Backend/Machine Learning/Networks/transformer_infer.cpp`
   - Scope: Large
   - Callers: `Backend/Machine Learning/Networks/transformer_generate.cpp:474`, `Backend/Machine Learning/Networks/transformer_generate.cpp:653`
   - Tests: `unit-tests/Backend/Machine Learning/nn-test.cpp` `2392`, `2640`, `3394`; `unit-tests/Backend/Machine Learning/parallel-test.cpp:916`
   - Conventions: const inference state, fail-fast sizing, preserve typed-KV caps, no shared RNG mutation
2. `/refactor shared token-step core for transformerLmSessionAppend and transformerLmBatchSessionAppendSelective in Backend/Machine Learning/Networks/transformer_infer.cpp`
   - Addresses: `LLM2-2` — deduplicate per-token append logic across single and batched decode paths
   - Rationale: parity bugs and cap-handling regressions are easiest to control when single and batched append share one internal shape.
   - Key files: `Backend/Machine Learning/Networks/transformer_infer.cpp`, `Backend/Machine Learning/Networks/transformer_common_utils.h`
   - Scope: Large
   - Callers: `unit-tests/Backend/Machine Learning/nn-test.cpp:2406`, `unit-tests/Backend/Machine Learning/nn-test.cpp:2681`
   - Tests: `unit-tests/Backend/Machine Learning/nn-test.cpp` `2317`, `2594`, `2724`, `2805`; `unit-tests/Backend/Machine Learning/transformer-improvements-test.cpp:580`
   - Conventions: preserve full-forward vs KV parity, keep RoPE/GQA behavior, no hidden allocations in hot loops

### Phase 2 — Structural improvement
1. `/refactor transformerLmForwardLastLogits and one-shot generate path in Backend/Machine Learning/Networks/transformer_infer.cpp and transformer_generate.cpp`
   - Addresses: `LLM2-3` — align debug/full-forward inference with extracted runtime cores and keep single-request generation reviewable
   - Rationale: `forwardLastLogits` is the parity oracle for later serving and persistence work.
   - Key files: `Backend/Machine Learning/Networks/transformer_infer.cpp`, `Backend/Machine Learning/Networks/transformer_generate.cpp`
   - Scope: Large
   - Callers: `Backend/Machine Learning/Networks/transformer_public_api.h:40`, `unit-tests/Backend/Machine Learning/nn-test.cpp:3573`
   - Tests: `unit-tests/Backend/Machine Learning/nn-test.cpp` `2061`, `3503`, `4175`; `unit-tests/Backend/Machine Learning/fuzz_transformer_infer.cpp:144` opt-in gap
   - Conventions: deterministic seed derivation, explicit stop-token handling, no re-entrant runs, preserve public config semantics
2. `/refactor sampling helper surface in Backend/Machine Learning/Networks/sampling_utils.h and transformer_generate.cpp`
   - Addresses: `LLM2-4` — separate deterministic sampling policy from generate control flow
   - Rationale: greedy/top-k/top-p equivalence and cap behavior have independent tests and should stay independently reviewable.
   - Key files: `Backend/Machine Learning/Networks/sampling_utils.h`, `Backend/Machine Learning/Networks/transformer_generate.cpp`
   - Scope: Medium
   - Callers: `unit-tests/Backend/Machine Learning/nn-test.cpp:4209`, `unit-tests/Backend/Machine Learning/nn-test.cpp:4240`
   - Tests: `unit-tests/Backend/Machine Learning/sampling-test.cpp:44`; `unit-tests/Backend/Machine Learning/nn-test.cpp` `4213`, `4244`, `4275`, `4303`
   - Conventions: deterministic per-seed behavior, explicit full-vocab vs capped-top-p semantics, no global RNG coupling

### Deferred
- Persistent batcher and serving wrapper work move to Batch 3.
- Training/numerics and GPU integration move to Batch 4.
- Persistence/package coupling stays out until Batch 5.

### Adjacent work (not refactor)
- `/perf transformer infer hot loops` - profile token-step changes before any kernel-oriented tuning
- `/test-design fuzz_transformer_infer build path` - make opt-in fuzzing easier without bloating this batch

## Batch 3 - Serving Runtime
- Goal: isolate the persistent batcher and synchronized serving wrapper after one-shot runtime behavior is stable.
- Why now: `TransformerServingLayer` is layered on top of `TransformerServeBatcher`, so serving work is safer once Batch 2 session and generation semantics stop moving.
- Main risks: callback lifetime, snapshot lifecycle, request-slot cleanup, and explicit rejection of re-entrant `step()`.
- Merge/conflict notes: allow `network.h`, `transformer_generate.cpp`, `transformer_serving_layer.{h,cpp}`, `nn-test.cpp`, and `transformer-serving-layer-test.cpp`; keep `sgd_transformer.cpp`, `checkpoint_persistence.cpp`, and `model_persistence.cpp` out.
- Stop condition before next batch: `transformer-serving`, serving cases inside `nn-transformer`, and `parallel` run-lock checks pass with no callback/snapshot regressions.

## Action Plan

### Build commands
build: `sh .configure.sh && cd unit-tests && sh .configure.sh`
test: `cd unit-tests && bash test.sh transformer-serving && bash test.sh nn-transformer && bash test.sh parallel`

### Phase 1 — Foundation
1. `/refactor TransformerServeBatcher state machine in Backend/Machine Learning/Networks/network.h and transformer_generate.cpp`
   - Addresses: `LLM3-1` — make persistent slot, prompt, generated-token, and request-result transitions explicit
   - Rationale: continuous batching complexity should be separated from one-shot generation and from the serving wrapper.
   - Key files: `Backend/Machine Learning/Networks/network.h`, `Backend/Machine Learning/Networks/transformer_generate.cpp`
   - Scope: Large
   - Callers: `Backend/Machine Learning/Networks/network.cpp:257`, `Backend/Machine Learning/Networks/transformer_serving_layer.cpp:169`
   - Tests: `unit-tests/Backend/Machine Learning/nn-test.cpp:3506`, `unit-tests/Backend/Machine Learning/parallel-test.cpp:1596`
   - Conventions: no per-step hot-path allocations after reset, explicit slot lifecycle, deterministic RNG override isolation, fail-fast capacity checks
2. `/refactor transformer serving layer scheduler and callback adapter in Backend/Machine Learning/Networks/transformer_serving_layer.{h,cpp}`
   - Addresses: `LLM3-2` — keep mutex, queue, snapshot, and callback sequencing clear on top of the batcher
   - Rationale: the serving wrapper has its own concurrency rules and should be reviewable without re-reading core decode math.
   - Key files: `Backend/Machine Learning/Networks/transformer_serving_layer.h`, `Backend/Machine Learning/Networks/transformer_serving_layer.cpp`
   - Scope: Medium
   - Callers: `unit-tests/Backend/Machine Learning/transformer-serving-layer-test.cpp:424`, `unit-tests/Backend/Machine Learning/transformer-serving-layer-test.cpp:974`
   - Tests: `unit-tests/Backend/Machine Learning/transformer-serving-layer-test.cpp` `523`, `768`, `1005`; gap: no external server/event-loop integration test
   - Conventions: release mutex before user callbacks, reject re-entrant `step()`, preserve callback ownership, no hidden thread creation

### Phase 2 — Structural improvement
1. `/refactor serving-focused LLM fixtures in unit-tests/Backend/Machine Learning/nn-test.cpp and transformer-serving-layer-test.cpp`
   - Addresses: `LLM3-3` — reduce fixture duplication across serve-batch parity and serving-layer lifecycle tests
   - Rationale: serving refactors are safer when test fixtures expose request-vs-slot semantics explicitly.
   - Key files: `unit-tests/Backend/Machine Learning/nn-test.cpp`, `unit-tests/Backend/Machine Learning/transformer-serving-layer-test.cpp`
   - Scope: Medium
   - Callers: `unit-tests/main.cpp:88`, `unit-tests/main.cpp:86`
   - Tests: `unit-tests/Backend/Machine Learning/nn-test.cpp` `3844`, `4069`; `unit-tests/Backend/Machine Learning/transformer-serving-layer-test.cpp:424`
   - Conventions: deterministic tiny-model fixtures, explicit request IDs vs slot IDs, callback-stop coverage, no hidden global state

### Deferred
- Training/numerics work stays in Batch 4.
- Package/checkpoint format work stays in Batch 5.

### Adjacent work (not refactor)
- `/observability transformer serving metrics` - keep logs/telemetry separate from control-flow refactors
- `/perf transformer continuous batcher` - profile before changing slot scheduling policy

## Batch 4 - Training and Integration
- Goal: split transformer training, numerics helpers, and parallel/CUDA integration into reviewable seams without mixing them into serving or persistence changes.
- Why now: after contracts and runtime are stable, the largest remaining operational risk is `sgd_transformer.cpp` plus its helper and GPU handoff coupling.
- Main risks: backward correctness, numerics drift, `gradientCheckpointing` and full-softmax config gaps, run-lock regressions during training, and CUDA ownership mismatches between root and unit-test builds.
- Merge/conflict notes: allow `sgd_transformer.cpp`, `transformer_train_detail.{h,cpp}`, `transformer_ops.h`, `transformer_kernels.h`, `trainer.cpp`, `parallel-test.cpp`, and CUDA transformer-state files; keep persistence files and serving wrapper out.
- Stop condition before next batch: `transformer-grad`, `transformer-ops`, `attention-bwd`, `transformer-kernels`, and `parallel` pass; `ddp` and `gpu-training` pass only when this batch actually touches those integration paths and CUDA was enabled in both builds.

## Action Plan

### Build commands
build: `sh .configure.sh && cd unit-tests && sh .configure.sh`
test: `cd unit-tests && bash test.sh transformer-grad && bash test.sh transformer-ops && bash test.sh attention-bwd && bash test.sh transformer-kernels && bash test.sh parallel`

### Phase 1 — Foundation
1. `/refactor transformer_train_detail helper ABI and header dependencies in Backend/Machine Learning/Networks/transformer_train_detail.{h,cpp}`
   - Addresses: `LLM4-1` — decouple low-level helper contexts from `network.h` and make helper inputs explicit
   - Rationale: this is the cleanest seam inside the training core and reduces header drag before touching SGD bodies.
   - Key files: `Backend/Machine Learning/Networks/transformer_train_detail.h`, `Backend/Machine Learning/Networks/transformer_train_detail.cpp`
   - Scope: Medium
   - Callers: `Backend/Machine Learning/Networks/sgd_transformer.cpp:1484`, `Backend/Machine Learning/Networks/sgd_transformer.cpp:1934`
   - Tests: `unit-tests/Backend/Machine Learning/transformer-ops-test.cpp:745`, `unit-tests/Backend/Machine Learning/transformer-kernels-test.cpp:34`
   - Conventions: plain-data helper contexts, reduce `network.h` coupling, no behavior change first
2. `/refactor CPU transformer SGD orchestration in Backend/Machine Learning/Networks/sgd_transformer.cpp`
   - Addresses: `LLM4-2` — split epoch orchestration, forward loop, and backward/update flow into reviewable subroutines
   - Rationale: `sgd_transformer.cpp` is the single largest hotspot and needs internal seams before any performance or feature work.
   - Key files: `Backend/Machine Learning/Networks/sgd_transformer.cpp`, `Backend/Machine Learning/Networks/network.h`
   - Scope: Large
   - Callers: `Backend/Machine Learning/Networks/network.h:1152`, `Backend/Machine Learning/Networks/trainer.cpp:317`
   - Tests: `unit-tests/Backend/Machine Learning/transformer-gradient-test.cpp:483`, `unit-tests/Backend/Machine Learning/transformer-improvements-test.cpp:244`
   - Conventions: preserve `TransformerEpochCfg`, deterministic seed behavior, explicit status propagation, no serving/persistence edits in this item
3. `/refactor transformer numerics utility surface in Backend/Machine Learning/Networks/transformer_ops.h, transformer_kernels.h, and sampling_utils.h`
   - Addresses: `LLM4-3` — isolate finite-difference-tested math helpers from orchestration code
   - Rationale: numerics helpers have their own verification surface and should not be implicitly rewritten inside SGD patches.
   - Key files: `Backend/Machine Learning/Networks/transformer_ops.h`, `Backend/Machine Learning/Networks/transformer_kernels.h`
   - Scope: Medium
   - Callers: `Backend/Machine Learning/Networks/transformer_train_detail.cpp:235`, `Backend/Machine Learning/Networks/transformer_train_detail.cpp:447`
   - Tests: `unit-tests/Backend/Machine Learning/attention-backward-test.cpp:554`, `unit-tests/Backend/Machine Learning/transformer-kernels-test.cpp:286`
   - Conventions: preserve scalar/SIMD parity, keep RoPE/norm/tied-embedding parity, numerics-first changes

### Phase 2 — Structural improvement
1. `/refactor parallel and DDP transformer run-state integration in Backend/Machine Learning/Networks/trainer.cpp and unit-tests/Backend/Machine Learning/parallel-test.cpp`
   - Addresses: `LLM4-4` — keep run-lock, callback, and worker-wrapper behavior explicit when training is active
   - Rationale: concurrency regressions are hard to bisect after GPU or persistence changes land.
   - Key files: `Backend/Machine Learning/Networks/trainer.cpp`, `unit-tests/Backend/Machine Learning/parallel-test.cpp`
   - Scope: Medium
   - Callers: `unit-tests/Backend/Machine Learning/parallel-test.cpp:1529`, `unit-tests/Backend/Machine Learning/ddp-test.cpp:768`
   - Tests: `unit-tests/Backend/Machine Learning/parallel-test.cpp` `827`, `1458`, `1529`; gap: `ddp` focuses more on wrappers/warmup than full multi-worker gradient sync
   - Conventions: non-re-entrant `NNetwork`, explicit callback ordering, no hidden shared state, preserve current `DDPDataInputWrapper` assumptions
2. `/refactor GPU transformer state and training handoff in Backend/Machine Learning/Networks/network.h, cuda/gpu_transformer_state.{h,cu}, and sgd_transformer.cpp`
   - Addresses: `LLM4-5` — separate GPU ownership/allocation/upload semantics from the GPU train-epoch launch path
   - Rationale: GPU enablement should be an integration seam, not interleaved with CPU training cleanup.
   - Key files: `Backend/Machine Learning/Networks/cuda/gpu_transformer_state.h`, `Backend/Machine Learning/Networks/cuda/gpu_transformer_state.cu`
   - Scope: Large
   - Callers: `Backend/Machine Learning/Networks/network.h:930`, `Backend/Machine Learning/Networks/sgd_transformer.cpp:2968`
   - Tests: `unit-tests/Backend/Machine Learning/gpu-training-test.cpp:30` CUDA-only; `unit-tests/CMakeLists.txt:12` build-mode coupling gap
   - Conventions: CPU/CUDA build parity, explicit ownership, no silent fallback, root/unit-test CUDA flags stay aligned

### Deferred
- Package/checkpoint serializer and tokenizer artifacts move to Batch 5.
- Any metrics/telemetry work for training stays adjacent, not inside the refactor batch.

### Adjacent work (not refactor)
- `/test-design transformer training config matrix` - cover `gradientCheckpointing` and large-full-softmax gaps explicitly
- `/perf sgd_transformer hot regions` - profile before any kernel-motivated rewrite

## Batch 5 - Persistence and Packaging
- Goal: isolate model package, tokenizer artifact, checkpoint, and serializer responsibilities after runtime and training contracts are stable.
- Why now: persistence duplicates config and tensor-catalog logic and is easiest to clean up once earlier batches stop moving the underlying contracts.
- Main risks: transformer vs non-transformer save/load divergence, tokenizer artifact completeness, checkpoint metadata drift, and serializer ordering regressions.
- Merge/conflict notes: allow `network.h`, `network.cpp`, `model_persistence.cpp`, `checkpoint_persistence.cpp`, `tokenizer_artifacts.cpp`, `nn-save-load-test.cpp`, and `transformer-improvements-test.cpp`; keep `transformer_generate.cpp`, `transformer_serving_layer.cpp`, and `sgd_transformer.cpp` out.
- Stop condition before next batch: `save-load` and `transformer-improvements` pass for transformer packages/checkpoints, and tokenizer artifacts round-trip unchanged.

## Action Plan

### Build commands
build: `sh .configure.sh && cd unit-tests && sh .configure.sh`
test: `cd unit-tests && bash test.sh save-load && bash test.sh transformer-improvements && bash test.sh nn-transformer`

### Phase 1 — Foundation
1. `/refactor TokenizerArtifacts validation and mutation contract in Backend/Machine Learning/Networks/network.h and tokenizer_artifacts.cpp`
   - Addresses: `LLM5-1` — make tokenizer metadata rules explicit before touching package IO
   - Rationale: tokenizer artifacts are optional in storage but semantically required for deployment-grade token LMs.
   - Key files: `Backend/Machine Learning/Networks/network.h`, `Backend/Machine Learning/Networks/tokenizer_artifacts.cpp`
   - Scope: Small
   - Callers: `Backend/Machine Learning/Networks/model_persistence.cpp:998`, `unit-tests/Backend/Machine Learning/nn-save-load-test.cpp:1468`
   - Tests: `unit-tests/Backend/Machine Learning/nn-save-load-test.cpp` `1468`, `1513`
   - Conventions: explicit validation, preserve backward compatibility, no filesystem side effects in pure mutation helpers
2. `/refactor transformer model package save/load seams in Backend/Machine Learning/Networks/model_persistence.cpp`
   - Addresses: `LLM5-2` — separate manifest, tokenizer directory, and tensor package responsibilities
   - Rationale: `saveModel/loadModel` is already a large filesystem hotspot and needs clearer sub-seams before any format changes.
   - Key files: `Backend/Machine Learning/Networks/model_persistence.cpp`, `Backend/Machine Learning/Networks/network.h`
   - Scope: Large
   - Callers: `Backend/Machine Learning/Networks/network.h:1279`, `Backend/Machine Learning/main.cpp:56`
   - Tests: `unit-tests/Backend/Machine Learning/nn-save-load-test.cpp:953`, `unit-tests/Backend/Machine Learning/transformer-improvements-test.cpp:758`
   - Conventions: atomic publish/rename behavior, explicit manifest versioning, no silent tokenizer omission, preserve non-transformer paths

### Phase 2 — Structural improvement
1. `/refactor transformer checkpoint tensor-catalog and config-restore seams in Backend/Machine Learning/Networks/checkpoint_persistence.cpp`
   - Addresses: `LLM5-3` — isolate checkpoint manifest/config parsing from tensor shard IO
   - Rationale: checkpoint load/save duplicates config handling and has stricter correctness requirements than model packages.
   - Key files: `Backend/Machine Learning/Networks/checkpoint_persistence.cpp`, `Backend/Machine Learning/Networks/network.h`
   - Scope: Large
   - Callers: `Backend/Machine Learning/Networks/network.h:1320`, `unit-tests/Backend/Machine Learning/nn-save-load-test.cpp:1782`
   - Tests: `unit-tests/Backend/Machine Learning/nn-save-load-test.cpp` `1782`, `1848`, `1947`
   - Conventions: strict tensor-name/element-count validation, explicit `TrainingConfig` restore, sharded IO stays current-format compatible
2. `/refactor transformer tensor weight serializer/load order in Backend/Machine Learning/Networks/network.cpp`
   - Addresses: `LLM5-4` — make transformer tensor save/load ordering auditable and separate from other net types
   - Rationale: the serializer is the hidden contract behind both model packages and tests that patch `weights.bin`.
   - Key files: `Backend/Machine Learning/Networks/network.cpp`, `unit-tests/Backend/Machine Learning/nn-test.cpp`
   - Scope: Medium
   - Callers: `Backend/Machine Learning/Networks/network.cpp:1917`, `Backend/Machine Learning/Networks/network.cpp:2181`
   - Tests: `unit-tests/Backend/Machine Learning/nn-test.cpp` `400`, `3102`, `3210`, `3814`, `3909`
   - Conventions: preserve on-disk order, keep non-transformer branches untouched, explicit version checks, no format drift without test updates
3. `/refactor persistence-focused LLM fixtures in unit-tests/Backend/Machine Learning/nn-save-load-test.cpp and transformer-improvements-test.cpp`
   - Addresses: `LLM5-5` — isolate transformer package/checkpoint/tokenizer assertions from unrelated save/load coverage
   - Rationale: persistence changes are safer when transformer-specific fixtures are explicit and compact.
   - Key files: `unit-tests/Backend/Machine Learning/nn-save-load-test.cpp`, `unit-tests/Backend/Machine Learning/transformer-improvements-test.cpp`
   - Scope: Medium
   - Callers: `unit-tests/main.cpp:112`, `unit-tests/main.cpp:122`
   - Tests: `unit-tests/Backend/Machine Learning/nn-save-load-test.cpp:953`, `1468`, `1782`; `unit-tests/Backend/Machine Learning/transformer-improvements-test.cpp:745`
   - Conventions: deterministic package names, isolated filesystem cleanup, keep transformer/non-transformer assertions distinct

### Deferred
- Fuzz-harness integration and docs cleanup stay outside this subsystem program.

### Adjacent work (not refactor)
- `/test-design persistence + checkpoint matrix` - cover full-softmax, CUDA, and `gradientCheckpointing` permutations explicitly
- `/observability model/checkpoint publish failures` - keep logging/telemetry separate from format refactors

## Ordering Notes
- Execute Batch 1 first. It fixes the token-id/config/public-boundary assumptions that every later batch depends on.
- The critical dependency chain is `Batch 1 -> Batch 2 -> Batch 3`; do not start serving work before session and one-shot generation seams from Batch 2 are stable.
- Batch 4 depends on Batch 1 and benefits from Batch 2, but it can be re-planned after Batch 1 if contract cleanup exposes a better split inside `sgd_transformer.cpp`.
- Batch 5 should remain last because package/checkpoint code serializes contracts touched by the earlier batches; re-plan it if Batch 1 or Batch 4 materially changes config or tensor ownership shape.
- Do not parallelize batches that touch `network.h`, `network.cpp`, `transformer_infer.cpp`, `transformer_generate.cpp`, `sgd_transformer.cpp`, `checkpoint_persistence.cpp`, or `unit-tests/Backend/Machine Learning/nn-test.cpp`; only adjacent non-refactor work can safely run in parallel.
