# CHIRON Serving Unification into glades-ml — Design

**Date:** 2026-07-03
**Status:** Approved (owner sign-off in session)
**Repos:** glades-ml (lib) + glades-trainer (consumer)

## 1. Problem

CHIRON exists in glades-ml only as per-op CUDA kernels (`Networks/cuda/gpu_chiron.h/.cu`)
plus CPU reference ops (`Networks/transformer_chiron_ops.h`, test-only). Everything
above the kernels — CHRN/CHRF checkpoint format knowledge, the model-level forward
(embedding → L×[SCFA shear, QK-Norm, WhiSC rot, fuse, reln] → readout), and serving
feature resolution — is implemented **twice in glades-trainer**:

- `trainer/chiron_main.cpp` (~19k lines): training forward `forward()` (~13007),
  SCFA forward (~8851), checkpoint writer `save_full_checkpoint` (~6046) and readers
  (`load_full_checkpoint` ~6353, `load_weights` ~5899).
- `tools/chiron_infer.cpp` (~1.4k lines): independent checkpoint reader
  (`loadCheckpoint` 87–355), eval forward (`forwardInfer` 580–740,
  `scfa_shear_infer` 484–578), serving interlocks/auto-enables (1040–1234).

The duplication has a documented silent-garbage bug history: QK-Norm omission
(degenerate generation), missing fuse-attn-reln (TF-NLL ~63 for every production
checkpoint until the 2026-06-27 serving fix), WhiSC dead-gamma_p path (TF-NLL ~20).
Two dormant divergences exist today: (a) infer's fuse-attn-per-layer path omits the
trainer's SCFA k/T damping; (b) bit 512 (OBSD `a_drift`) is recognized but never
loaded/applied — a drift-trained checkpoint would serve silently wrong. Three
"mirrors chiron_main.cpp:NNNN" comments point at stale line numbers.

A second source-of-truth gap: glades-ml's `make install` installs **no Networks
headers** (GMath only), so the trainer compiles against a hand-synced vendored copy
of 109 lib headers (`glades-trainer/include/Backend/...`) while linking the
separately-built `~/.local/lib` archives — a structural skew vector (currently in
sync, byte-identical).

## 2. Decisions (owner-approved)

1. **Scope**: move *inference + checkpoint format* into the lib. The trainer's
   training forward (perf-critical research surface: PIED, bf16 overlays,
   backward, optimizer) stays in the trainer.
2. **Headers**: glades-ml installs its `Backend/Machine Learning` public header
   tree; the trainer's vendored `include/` copy is deleted.
3. **Parity bar**: bit-identical logits pre/post refactor + exact TF-metric
   reproduction on three era checkpoints + new lib unit tests (§6).
4. **Val forward**: `chiron_main`'s validation **keeps the trainer forward**
   (`isTraining=false`) — switching it to the lib eval forward would shift every
   future val number relative to historical gate records (the two paths round
   differently by design: trainer val uses bf16-residual-p mirrors / fused reln;
   the eval forward is plain FP32; the ship records report them as two separate
   verification methods, e.g. PIED 1.0259 trainer-val vs 1.0897 chiron_infer TF).
   Cross-implementation drift is instead caught by a parity tool (§5).

## 3. New lib module — `glades::chiron` model layer

Two file pairs in `Backend/Machine Learning/Networks/` (Networks CMake target,
C++98, `GLADES_HAVE_CUDA`-gated with inline no-op stubs, same pattern as
`gpu_chiron.h`).

### 3.1 `chiron_checkpoint.h/.cpp` — CHRN/CHRF format single source of truth

- **`ChironCkptBits`** constant registry: FACE=1, BF16_ADAM=2, KAHAN=4, GAMMA_P=8,
  INT8_ADAM=16, FP32_ADAM=32, BF16_DISK=64, SCFA=128, QKNORM_GAMMA=256,
  A_DRIFT=512, ROT_PHI=1024; plus `CHIRON_CKPT_MAX_KNOWN_BIT` and the CHRN-v3
  legacy bits (0x1 bf16 weights, 0x2 hasGammaP). Both binaries compile against
  this one registry.
- **`ChironModelDims`** (T, m, L, nH, dH, V, dModel).
- **`ChironModelWeights`**: GpuBuffers for E, per-layer Wq/Wk/Wv/Wo/γ/β, optional
  γ_p/β_p; SCFA state (k, w, per-layer D; B recomputed from (T,k) as today);
  host-side qknorm γ (L·nH) and rot_phi (L·m); presence flags.
- **Block codecs** moved from `chiron_main.cpp`: fp32/bf16 weight block
  (`save/load_weight_block`), bf16-group-with-count, int8-Adam group, fp32-Adam
  group. Exposed so the trainer writes its optimizer sections through them.
- **Serving reader** `chiron_load_model(path, dims, weights, err)` — ported from
  `chiron_infer::loadCheckpoint` with behavior preserved exactly:
  - Accepts CHRN v1–3 and CHRF v2–4.
  - Sequential read of header → weights blob (fp32 or bf16-on-disk) → SCFA (bit
    128) → qknorm γ (bit 256); `rot_phi` (bit 1024) read from the EOF tail
    (depends on the writer emitting rot_phi last — a contract the module now
    owns on both sides).
  - CHRN-v3 gamma_p file-size fallback preserved.
  - Hard-error on any flag bit > 1024 (unknown newer format would corrupt the
    EOF-tail read). Reports via error codes/messages, not `exit()` — the CLI
    maps them to its existing exit codes.
- **Model-section writer helpers**: header+flags word, weights blob, SCFA, qknorm
  γ, a_drift, rot_phi sections. The trainer's `save_full_checkpoint` keeps its
  orchestration and optimizer sections (Adam/Kahan/FACE) but emits every model
  section and the flags word via these helpers.
- **Rejected alternative**: porting the entire writer including optimizer state
  into the lib. Those sections have exactly one producer and one consumer (both
  the trainer), so moving them buys no de-duplication and couples optimizer
  research churn to the lib.

### 3.2 `chiron_serving.h/.cpp` — feature resolution + eval forward

- **`chiron_resolve_serving(dims, weights, overrides) → ChironServingConfig | error`**
  — the decision table moved from `chiron_infer` main (1040–1234):
  - QK-Norm auto-enable when bit-256 γ present; γ=log₂(T) fallback otherwise.
  - fuse-attn-reln auto-enable for SCFA checkpoints without gamma_p
    (the 2026-06-27 serving fix).
  - WhiSC interlock: checkpoint-has-rot_phi XOR `--whisc-coupling` → fatal both
    directions (exit-7 class).
  - WhiSC dead-gamma_p guard (switch to reln-fuse).
  - SCFA auto-enable when D loaded.
  - **New — bit-512 refusal**: `a_drift` present → hard-error. The eval forward
    does not apply OBSD drift, so serving such a checkpoint would be silently
    wrong (same failure class as the historic QK-Norm/fuse omissions). OBSD is a
    closed NO-GO; no production checkpoint carries the bit. Refuse loudly.
- **`chiron_eval_forward(dims, weights, cfg, tokens, useLen, scratch, logits)`** +
  **`ChironEvalScratch`** — ported from `forwardInfer`/`scfa_shear_infer`,
  **kernel-call-sequence-preserving** (the property that makes the bit-parity
  gate achievable). Embedding → per layer: SCFA shear (QK-Norm decomposed path
  when enabled) or dense shear → fuse (reln or per-layer) → WhiSC rot (per-batch
  stats, ema=1.0, theta_max from config) → q-reln → readout `logits = q_L·Eᵀ`.
- **Intentional divergence fix**: the fuse-attn-per-layer path adopts the
  trainer's SCFA k/T damping (`α` damped by k/T on the SCFA branch;
  chiron_main ~13304 vs chiron_infer's plain 1/√L at 617). Off the production
  path (flagship serves fuse-attn-reln), so the bit-parity gate is unaffected;
  recorded as an intentional behavior change.

## 4. glades-ml build/install + unit tests

- Install the `Backend/Machine Learning` public header tree to
  `${prefix}/include/glades/Backend/Machine Learning/...` and add
  `${prefix}/include/glades` to the exported target's
  `INTERFACE_INCLUDE_DIRECTORIES`, so `find_package(glades)` delivers headers.
  In-tree unit tests are unaffected (they include from the source tree).
- New unit suite **`chiron-model`** (registered in `unit-tests/main.cpp`, run via
  `test.sh chiron-model`):
  - Checkpoint write→read roundtrips at tiny shape across flag combinations,
    incl. rot_phi EOF-tail, legacy CHRN v1–3 / CHRF v2–4, bf16-on-disk, and
    unknown-bit rejection.
  - `chiron_resolve_serving` decision-table tests — one per interlock/auto-enable,
    including the new bit-512 refusal.
  - Eval-forward parity vs a CPU-reference composition built from
    `transformer_chiron_ops.h` at small shape.
  - All GpuBuffer downloads asserted (known gotcha: silent download failure).

## 5. glades-trainer changes

- **`tools/chiron_infer.cpp`** shrinks to a thin CLI: arg parsing, BPE, sampling
  (`sampleToken`), generation loop, TF eval (`--tf-check`), degeneration metrics,
  `--dump-logits` (§6). Model loading/resolution/forward via the lib. BPE and
  sampling stay in the trainer (the lib has no tokenizer; sampling is CLI
  policy). Existing exit-code contract preserved (2–7).
- **`trainer/chiron_main.cpp`**: `save_full_checkpoint` / `load_full_checkpoint` /
  `load_weights` switch to lib bit constants + block codecs + model-section
  helpers; optimizer-section orchestration stays local. **Training/val forward
  untouched.**
- **New `tools/chiron_parity.cpp`**: loads a checkpoint, runs the trainer forward
  (`isTraining=false`) and `chiron_eval_forward` on the same token window,
  asserts NLL agreement within tolerance (~1e-3, calibrated to the known
  FP32-vs-bf16-path gap). Replaces the "mirrors chiron_main.cpp:NNNN" comment
  convention as the drift tripwire.
- Delete the vendored `include/` tree; `COMMON_INCLUDES` switches to the
  installed glades include dir from `find_package(glades)`.

## 6. Parity gates (pre-registered, in order)

1. **Golden capture (before any refactor)**: add `--dump-logits FILE` to the
   *current* `chiron_infer` (additive-only), capture raw FP32 logits for a fixed
   token window on the three era checkpoints: `chiron_1B_pied_e4.final`,
   `chiron_1B_T16384_whisc30k.final`, `chiron_1B_T16384_reanchor5B_finish.final`.
2. Lib unit tests green: new `chiron-model` + existing `chiron`/`chiron-rot`/
   `chiron-whisc`/`chiron-pied` suites unregressed.
3. **Bit-identity**: refactored `chiron_infer` logits byte-identical to the
   goldens on all three checkpoints (forward-only is run-to-run deterministic;
   the port preserves the kernel call sequence). Fallback to gate 4 alone only
   with a documented root cause.
4. **TF metrics** reproduce at printed precision: PIED 1.0897/0.7176, whisc30k
   1.4655/0.626, reanchor 1.7798.
5. **Interlock regression**: scripted decision-table check (WhiSC ckpt without
   `--whisc-coupling` → exit 7 and vice versa; dead-gamma_p guard; QK-Norm/fuse
   auto-enables; unknown-bit and bit-512 refusals) — same exit codes as today.
6. **Trainer side**: rebuilt trainer (static-link gotcha: `make install` alone
   does not update it — run `bash build.sh`) passes `run_checkpoint_self_test`,
   a short smoke train + save/load, and resumes an existing flagship checkpoint;
   `runner.sh --flagship` end-to-end unchanged.
7. `chiron_parity` passes on the flagship checkpoint.

## 7. Risks & constraints

- **C++98 in the lib**: ported code sheds C++11-isms (e.g. the `readBlock`
  lambda becomes a helper struct); mechanical.
- **Legacy checkpoints must keep loading**: reader port is behavior-preserving
  by construction; roundtrip tests cover the legacy paths.
- **Build-order coupling**: unchanged in kind (trainer already requires rebuild
  after lib changes); header changes now also flow through `make install`.
  Documented in both repos' CLAUDE.md.
- **Out of scope**: KV-cache/incremental decoding, generation quality, any
  checkpoint format change, the training forward, multi-batch inference.

## 8. References

- Ship records: `research/CHIRON_PIED_E4_GATE_2026_07_03.md`,
  `research/CHIRON_WHISC_D_GATE_2026_06_30.md`,
  `research/RELN_REANCHOR_GATE2_2026_06_27.md`.
- Serving-bug history: chiron_infer.cpp comments at 903–905 (QK-Norm), 1070–1084
  (2026-06-27 fuse fix), 1040–1063 (WhiSC interlock), 1086–1122 (dead-gamma_p).
