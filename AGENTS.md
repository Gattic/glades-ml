# AGENTS.md

Guidance for coding agents working in `glades-ml`. Last reviewed: 2026-08-05.

## Scope and working rules

- This repository is the Glades machine-learning library. The sibling
  `~/dev/glades-trainer` repository owns the standalone training/inference CLIs and
  research harnesses that consume this library.
- The core project is C++98. Do not introduce newer C++ features into library or unit-test
  code unless the build standard is deliberately changed project-wide. CUDA code follows
  the same public API and ownership constraints.
- Treat current source, tests, and final/result research records as authoritative. Drafts
  and older ship reports are historical evidence, not present-tense operating policy.
- Preserve unrelated work and retained experiment evidence. Do not edit generated files,
  build trees, logs, checkpoints, datasets, or research outputs unless the task explicitly
  targets them.
- Optional CHIRON mechanisms are default-off unless current code and a reviewed protocol
  say otherwise. Implementation does not imply scientific approval or production status.

## Current CHIRON status

The current implementation is **causal-block SCFA**. `CKPT_BIT_CAUSAL_SCFA` (bit 4096)
identifies that operator. Current training and serving code reject SCFA checkpoints that
lack the marker because the historical global-DCT operator leaked future tokens.

Consequences:

- The July 3 PIED 30k checkpoint remains the best historical **pre-causal perplexity
  record**, but it is not a serving-eligible current model and cannot seed a current causal
  resume. It lives in the trainer repository, not this one.
- There is currently no promoted causal production checkpoint. In
  `glades-trainer`, `run.sh flagship` names the causal 1B recipe; it does not prove that a
  serving flagship exists. Long training requires explicit approval.
- `glades-trainer/runner.sh --flagship` still discovers the historical PIED checkpoint.
  The current library is expected to reject it at the causal interlock; do not describe
  that path as working production serving.
- The practical-v1 program passed causal serving/cache engineering P0-P3 but ended at
  **P4 NO-GO** on eight-document memorization. See
  `docs/superpowers/plans/2026-07-30-chiron-practical-trainability.md`.
- Contextual Rank Margin (CRM) is an isolated default-off path whose Q0 engineering
  qualification has passed. It is not a shipped recipe; no E0 or scientific optimizer step
  has run. Scientific execution remains governed by the exact authorization and gate
  contract in the trainer's
  `research/generation-aware/CHIRON_CONTEXTUAL_RANK_MARGIN_INTERVENTION_DRAFT_2026_07_31.md`.
- SIRA, ECHO, ORBIT, PACT, PIED, FFN/GQA experiments, ARREST, and other research features
  remain opt-in. Follow their latest final/result record; do not revive closed NO-GO arms
  or rerun settled gates without a new approved protocol.

Historical ship details remain in `research/CHIRON_PIED_E4_GATE_2026_07_03.md` and the
other dated `research/` records. Keep that chronology out of this operational file.

## Cross-repository build boundary

`glades-trainer` uses installed Glades headers/libraries and statically links the Glades
CUDA kernels. After any library/header/kernel change, rebuild in this order:

```sh
cd ~/dev/glades-ml/build
make install
cd ~/dev/glades-trainer
bash build.sh
```

`make install` alone does **not** update the trainer binary. The trainer no longer has a
vendored Glades include tree. On a fresh machine, Shmea headers may need the one-time copy
shown in `~/dev/glades-trainer/AGENTS.md`.

## Build and test commands

Configure/build the library from the repository root:

```sh
sh .configure.sh          # CPU
sh .configure.sh cuda     # CUDA
```

Configure/build unit tests from `unit-tests/` (not `unit-tests/build/`):

```sh
cd unit-tests
sh .configure.sh cuda
```

For an existing configured tree, prefer incremental builds:

```sh
cmake --build build -j "$(nproc)"
cmake --build unit-tests/build -j "$(nproc)"
```

The CMake caches are sticky. Remove/reconfigure a build tree only when changing toolchain,
build type, or incompatible flags; never clean as a routine first response to a failure.

Run focused tests through the wrapper:

```sh
bash unit-tests/test.sh chiron-model
bash unit-tests/test.sh chiron-generate-cpu
bash unit-tests/test.sh chiron-generate
bash unit-tests/test.sh chiron
```

Useful focused selectors include `chiron-token-count`, `chiron-sira`, `chiron-phs`,
`chiron-ptoc`, `chiron-qclamp`, `chiron-reanchor`, `chiron-rot`, `chiron-whisc`,
`chiron-pied`, `chiron-gqa-ffn`, `chiron-vitals`, `chiron-orbit`, `chiron-pact`,
`chiron-echo`, `chiron-echo-cpu`, `chiron-crm`, `chiron-crm-cpu`, `chiron-model`,
`chiron-generate-cpu`, and `chiron-generate`. Use `test_project(filter=...)` when the
harness is available.

### Known verification issue

Debug ledger thread `intermittent-chiron-decode-reset-reproducibility` is open. One chained
`chiron` then `chiron-model` execution intermittently failed the
`decode reset reproducible` assertion, while unchanged focused runs before and after
passed. Do not weaken the assertion or retry the already-recorded no-change attempt.
Consult the ledger first and record only new hypotheses/attempts.

## CHIRON ownership and invariants

### Checkpoint format

- `Backend/Machine Learning/Networks/chiron_checkpoint.{h,cpp}` is the single source of
  truth for CHRN/CHRF model sections, flag bits, version rules, and serving reads.
- The trainer owns optimizer sections around those shared codecs; do not duplicate model
  section parsing in the trainer or serving tools.
- Preserve canonical section order. WhiSC/optimizer/architecture state precedes the two
  EOF tails; `a_drift` is immediately before `rot_phi`, and `rot_phi` must remain last.
- Unknown bits hard-error intentionally. Causal SCFA resumes require both their SCFA state
  and bit 4096; WhiSC causal resumes also require persisted bit-2048 calibration state.
- `CKPT_BIT_FP32_EMBEDDING` (bit 65536) carries the resume-exact embedding override and
  selects CHRF v5. Update round-trip and interlock tests for any format change.

### Serving, generation, and decode

- `chiron_serving.{h,cpp}` owns serving interlocks, configuration resolution,
  `ChironEvalScratch`, and `chiron_eval_forward`.
- `chiron_generate.{h,cpp}` owns sampling, observed generation, cached generation,
  teacher-forcing evaluation, and degeneration metrics. `chiron_tf_eval` is the only
  source of truth for CHIRON TF-NLL.
- `chiron_decode_cache.{h,cpp}` owns exact-geometry causal prefill/decode,
  reset/clone/snapshot state, and cached branches. The v1 cache does **not** slide and must
  reject positions at or beyond native `T` without mutation.
- Observers run after sampling but before append/commit. Observer failure must not append a
  token or advance externally visible output.
- `ChironMt19937` is pinned to the libstdc++-13.3.0 canonical-double stream. Do not change
  it without deliberately re-pinning CPU/GPU generation goldens.

### Current module map

- `Backend/Machine Learning/Networks/sgd_transformer.cpp` — standard transformer training.
- `transformer_{infer,generate}.cpp`, `transformer_ops.h`, `transformer_kernels.h` —
  transformer inference/generation/math.
- `transformer_chiron_ops.h` — CHIRON host/reference helpers, including default-off
  research objectives such as CRM.
- `Backend/Machine Learning/Networks/cuda/gpu_kernels.{h,cu}` and related CUDA files — GPU
  kernels used by both library tests and the statically linked trainer.
- `chiron_vitals.{h,cpp}` and `chiron_orbit.{h,cpp}` — opt-in telemetry/optimizer support.
- `training_config.h` / `transformer_config.cpp` — shared configuration defaults and
  validation.
- `transformer_public_api.h` — stable serving/generation-facing API.

## General library architecture

- `Backend/Machine Learning/CMakeLists.txt` composes `ML`, `Networks`, `MLStructure`,
  `MLState`, `DataObjects`, and `GMath`.
- Network types are declared in `Backend/Machine Learning/Networks/network.h`; each major
  family has a dedicated SGD implementation.
- Public entry points live in `Backend/Machine Learning/main.h`; call `glades::init()`
  before public training/testing APIs.
- Randomness must flow through the project RNG policy. See
  `Backend/Machine Learning/DETERMINISM_AND_CONCURRENCY.md`.
- Unit tests use `ASSERT(failmsg, predicate)` from `unit-tests/unit-test.h` and are
  registered in `unit-tests/main.cpp`.

## Verification expectations

- Documentation-only changes: inspect the final diff, run `git diff --check`, and verify
  referenced paths/commands.
- C++/CUDA changes: build the affected tree and run the narrowest relevant selector first;
  expand to `chiron`, `chiron-model`, or generation suites according to impact.
- Checkpoint, serving, generation, cache, RNG, or cross-repository API changes require the
  corresponding round-trip/interlock/parity tests and a rebuilt trainer.
- Before claiming commit or publish readiness, inspect both repositories when the change
  crosses the static-link boundary.
