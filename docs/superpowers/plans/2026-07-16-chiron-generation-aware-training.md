# CHIRON ARREST Generation-Aware Training — Implementation Plan

> **Execution boundary:** this is a plan, not authorization to run it. Do not pause, modify, or share
> outputs with the active causal FineWeb 275k run. GPU gates begin only after an explicit owner decision
> and a suitable causal checkpoint exists.

**Goal:** determine whether free-running repetition collapse transfers to fully trained prefix-causal
CHIRON, then—only if it does—implement a default-off rollout-conditioned hazard-mass objective without
harming causal next-token quality.

**Architecture:** ARREST alternates offline deterministic rollout collection with sparse auxiliary
aggregate-unlikelihood updates on collected generated-prefix states. Clean PIED CE remains the anchor.
Auxiliary replay uses serving-equivalent eval mode (PIED/training-only fields off). Rollout buffers and
an atomic run sidecar are versioned training inputs; there is no new model, serving, or checkpoint-format
state.

**Spec:** `docs/superpowers/specs/2026-07-16-chiron-generation-aware-training-design.md`

**Evidence:** `research/CHIRON_GENERATION_LOOP_FIX_2026_06_27.md`,
`research/CHIRON_FINEWEB_STAGE1_RECOVERY_PLAN_2026_07_14.md`, and
`docs/decisions/2026-07-16-echo-regularizer-no-go.md`.

---

## 1. Hard constraints

1. **Calibration, transfer, and replay feasibility first.** Do not implement the loss until Task 3
   passes G0a/G0b/G1 on a fully trained causal-SCFA checkpoint (`flags=0x1fd8` or a documented causal
   successor).
2. **Legacy evidence is not the baseline.** The global-DCT checkpoint can test detector sensitivity but
   cannot authorize treatment or serve as the ship comparator.
3. **ECHO remains closed.** Do not alter/re-tune `--echo-*`; clean-text copy mass is not ARREST input.
4. **No online rollout in the trainer.** A separate managed collector writes pinned local buffers.
5. **Coefficient-zero exactness.** At `--arrest-coef 0`, no buffer opens, scratch allocates, or branch
   runs; one-step checkpoints and telemetry are bit-identical.
6. **No serving/checkpoint-format delta.** ARREST is parameter-free and training-only; provenance and
   crash-resume state live in an atomic sidecar.
7. **Immutable three-way split.** Calibration/evaluation document IDs and offsets never enter rollout
   training buffers.
8. **No diversity-only pass.** Every recovery gate includes NLL, stability, and realism/coherence.
9. **Active-run isolation.** CPU/docs/build work can proceed separately; no GPU command competes with the
   275k job unless the owner explicitly schedules it.

---

## 2. Repository map

### glades-ml

- `Backend/Machine Learning/Networks/chiron_generate.{h,cpp}` — metrics, prefix-online detector,
  compatible step observer, and row-only generation
- existing `cuda/gpu_kernels.h` and `cuda/gpu_device.h` copy/event/stream primitives — row D2H; no
  generic `GpuBuffer` API change
- `Backend/Machine Learning/Networks/cuda/gpu_kernels.{h,cu}` — BF16 hazard loss/backward
- `Backend/Machine Learning/Networks/transformer_chiron_ops.h` — CPU reference
- `unit-tests/Backend/Machine Learning/chiron-generate-test.{h,cpp}` — detector and token-stream parity
- new `unit-tests/Backend/Machine Learning/chiron-arrest-test.{h,cpp}` — objective/kernel tests
- unit-test `CMakeLists.txt` and `main.cpp` registration

### glades-trainer

- new `tools/chiron_rollout_collect.cpp`
- new `trainer/chiron_arrest_buffer.{h,cpp}`
- modify `trainer/chiron_main.cpp`, `CMakeLists.txt`, and `run.sh`
- new `scripts/chiron_arrest_smoke.sh` and `scripts/chiron_arrest_eval.py`

After glades changes, install glades before rebuilding trainer. Prefer harness build/test tools and managed
long jobs over manual background commands.

---

## 3. Task 0 — freeze evidence and manifests

**Files:**

- new `research/generation-aware/detector_calibration_v1.tsv`
- new `research/generation-aware/eval_manifest_v1.tsv`
- new `research/generation-aware/train_manifest_v1.tsv`
- new `research/generation-aware/README.md`
- new trainer `scripts/build_chiron_generation_manifests.py`

- [ ] Define rows as
  `split,stratum,domain,corpus_sha256,shard,document_id,token_offset,active_T,active_k,prompt_tokens,rollout_tokens`.
- [ ] Freeze a disjoint detector set with at least 128 blind positive-loop labels and 64 blind
  real-continuation negatives per domain from synthetic loops, legacy generations, and real text.
- [ ] Freeze 128 evaluation contexts: 64 prose, 24 code, 16 math/markup, 24 mixed; stratify 96 reduced
  no-slide and 32 full no-slide, with a 16-context sliding diagnostic. Build at least 512 training
  contexts from the training split only.
- [ ] Deterministically regenerate sorted manifests; fail on any calibration/train/eval document overlap,
  shard crossing, short context, invalid SCFA geometry, or hash mismatch.
- [ ] Record corpus/vocab/script/manifest SHA-256 and causal-checkpoint requirements in the README.
- [ ] Implement `scripts/chiron_arrest_eval.py` with context-clustered paired bootstrap, Holm correction
  for raw decoder endpoints/domains, seed-level tables, and explicit early-stop/missing handling; freeze
  its hash before G0b.
- [ ] Add a CPU test that regenerates in `/tmp`, compares bytes, and asserts disjointness.

**Manifest gate M0:** deterministic regeneration and zero overlap across calibration, train, and
evaluation.

---

## 4. Task 1 — robust detector API

**Files:** `chiron_generate.{h,cpp}`, `chiron-generate-test.{h,cpp}`.

Keep `chiron_degeneration_metrics` as a compatibility wrapper. Add C++98 structures for:

- `ChironRepetitionConfig`: maximum period 64, period support `max(32,2p)`, threshold .80, n-gram
  window 128, hazard cap 16;
- `ChironRepetitionMetrics`: distinct-1/2/4, repeat fraction, maximum cycle score/period, repeated-span
  coverage, longest suffix copy, max run, collapse onset, collapsed flag;
- `ChironHazardRow`: 16 deduplicated token IDs/confidences, count, overflow flag.

Add pure functions `chiron_repetition_metrics(...)` and `chiron_repetition_hazards(...)`.

- [ ] RED fixtures: constant token; period 2/3/8/16/32/64; high-distinct-4 template loop; long repeated
  span; varied token soup; real continuation; code delimiters; empty/short input; duplicate hazards;
  cap overflow.
- [ ] Implement deterministic distinct-n, repeat fraction, cycle score/onset, and collapse event exactly
  as the spec.
- [ ] Add append-invariance tests: a hazard row for state `y_<t` is unchanged when arbitrary future
  tokens are appended; detector/hazard construction cannot look past `t`.
- [ ] Construct only run continuation, confident period continuation, and repeated n-gram closure
  (`n=2..8`, trailing 128); do not blacklist all prior tokens.
- [ ] Use fixed-size rows in hot loops; deduplicate by highest confidence and expose overflow.
- [ ] Prove the old metric wrapper returns unchanged distinct-4/max-run outputs.

**Gate G0a-engineering:** fixtures pass, golden outputs are compiler-order independent, old API
unchanged, and frozen blind calibration meets sensitivity/FPR bars before causal evaluation.

---

## 5. Task 2 — row-only logits and versioned rollout buffers

### 2A. Row-only logits

Current generation downloads all `[T,V]` logits per token. ARREST must transfer one contiguous `V` row.

- [ ] Add a backward-compatible `ChironGenerationStepObserver` entry point that synchronously exposes
  raw row logits, state row, sampled token, and step; keep the current `chiron_generate` signature as a
  null-observer wrapper.
- [ ] RED tests: observed row equals a full download; first/last row; bounds; CUDA-absent behavior; old
  API token/sink parity.
- [ ] Use existing `device_memcpy_d2h` plus an explicit compute-to-transfer event dependency and transfer
  synchronization on `s.logits.data()+row*V`; do not add a gather kernel or generic `GpuBuffer` method.
- [ ] Remove the host `[T,V]` allocation from generation and retain only `V` logits.
- [ ] Prove generated token IDs and RNG draw counts remain bit-identical on golden seeds.
- [ ] Add reduced/full SCFA parity tests using existing `ChironServingOverrides::seqLen` and
  `scfaKOverride`. Require exact divisibility and preserve block width (`T_r/k_r=T_0/k_0`), loaded QK
  gamma, WhiSC state, and `k/T` fuse scale. For the current geometry, start with `2048/128` versus
  `16384/1024`; changing `T` alone is prohibited.
- [ ] Benchmark per-token wall, D2H bytes, and projected per-round GPU-hours. This optimization is not
  ARREST efficacy evidence.

### 2B. CHAB v1 codec

Create fixed little-endian `CHAB` records in `trainer/chiron_arrest_buffer.{h,cpp}`:

```text
Header: magic/version, source_T/source_k, active_T/active_k, V,H,K_A,record_count;
        checkpoint/vocab/corpus/manifest/detector hashes; checkpoint flags;
        decoder-table and block-geometry hashes.
Record: manifest row, split/stratum/domain, corpus offset, decoder id/seed;
        uint16_t token_sequence[active_T], generated start/count;
        each row's hazard count, hazard_ids[K_A], weight, raw behavior hazard mass,
        raw sampled-action log-probability; summary metrics, collapse onset, CRC32.
```

A record uses `prompt_len=active_T-H` followed by `H` generated tokens. Reduced records are padded and
replayed under full training geometry only after G1 proves the scored rows equivalent. At rollout step
`t`, construct the hazard row before sampling `y_t` and attach it to the exact logits row that produced
`y_t`; a sentinel test catches any one-token shift. Sliding trajectories cannot be represented by one
teacher-forced record and are diagnostic-only.

- [ ] Round-trip, corruption, truncation, whole-file SHA, wrong-T/k/V, block-ratio, noncausal-flags,
  decoder-table, and three-way split-leakage tests; require `V<=65536` for uint16 token storage.
- [ ] Collector/loader require causal flags, exact persisted QK gamma, and WhiSC state where the recipe
  requires them.
- [ ] Implement `chiron_rollout_collect`: load checkpoint once, resolve serving, read manifest, generate
  pinned raw/production arms, compute hazards, and write CHAB plus JSON/TSV summary.
- [ ] Record raw pre-decoding entropy/top1-top2 margin plus realized length and EOS/pad/control rates;
  sampler penalties cannot contaminate raw observables.
- [ ] Add `--dry-run` that validates paths/hashes and estimates records/bytes/forwards without GPU use.

**Gate G1-collector:** token/row/replay parity, codec/provenance validation, no leakage, and measured
collection cost `<=10%` target / `<=25%` hard kill.

---

## 6. Task 3 — G0a/G0b/G1 preregistration and branch point

**Hard branch point:** do not proceed to Task 4 unless all applicable gates pass. Open an
`experiment_ledger` entry before any generation, use `run_deck`, and never disturb the source job.

**Inputs:** fully trained causal checkpoint; frozen calibration/eval manifests; `H=256`; raw greedy, raw
nucleus `.8/40/.95`, and production decoding; 96 reduced no-slide, 32 full no-slide, and 16 sliding
sentinel contexts. Stochastic reduced arms use three fixed decoder seeds and full arms one fixed seed.

- [ ] **G0a detector:** blind-label only the disjoint calibration set (at least 128 positives and 64
  real negatives/domain); require one-sided 95% sensitivity lower bound `>=.95`, overall FPR upper bound
  `<=.05`, and per-domain FPR upper bound `<=.08`; then freeze detector/config/data hashes and paired
  realism noninferiority margins.
- [ ] Confirm causal flags, exact QK gamma, WhiSC state, checkpoint EOF, and zero validation/training
  steps.
- [ ] **G1 geometry/replay:** using isolated/reloaded model instances, prove reduced/full same-row raw
  probability/hazard-mass parity (`<=2e-3` absolute) and golden generated-token parity. Then replay CHAB
  through trainer `isTraining=false`/PIED-off and require p95 per-row hazard-mass difference
  `<=max(2e-3,.05*q_serving)`. If reduced parity fails, benchmark full geometry only.
- [ ] Benchmark row-only collection and preregister records/context mix per round. Target collection
  `<=10%` of fine-tune GPU-hours; `>25%` is a hard NO-GO.
- [ ] **G0b transfer:** collect all frozen eval contexts. Pair treatment-independent decoder seeds,
  cluster bootstrap by context, and use simultaneous intervals across raw decoder endpoints.
- [ ] Adjudicate:
  - **GO:** reduced no-slide incidence `>=20%` in a raw mode, adjusted 95% lower bound `>10%`, confirmed
    loops;
  - **NO-GO:** reduced/full no-slide incidence `<=5%` with adjusted upper bounds `<10%`;
  - **LONG-ONLY:** only full no-slide collapses—continue only if full-buffer cost passes G1;
  - **SLIDE-ONLY:** only sliding collapses—stop and scope context-shift work;
  - **INCONCLUSIVE:** expand once with new held-out contexts and unchanged thresholds.
- [ ] Write `research/CHIRON_ARREST_G0_CAUSAL_TRANSFER_2026_MM_DD.md` and record every PASS/FAIL/NULL.

## 7. Task 4 — CPU aggregate hazard loss

**Files:** `transformer_chiron_ops.h`, new `chiron-arrest-test.{h,cpp}`.

Add FP32 reference functions for row loss and dlogits with inputs `(probs,V,hazard_ids,count,weight,
coef,eps)`. Use

```text
q = sum_{j in A} p_j
L = -weight log(1-q)
dL/dz_j = weight p_j (1{j in A}-q)/(1-q).
```

- [ ] RED finite differences for one/many hazards, deduplication, empty row, `q` near zero/one,
  weight/coefficient linearity, and logit shift.
- [ ] Reject invalid token IDs; deduplicate before computing `q`.
- [ ] Check row gradient sum `<=1e-6` in FP32.
- [ ] Normalize by total active weight `Z=sum_t w_t`; skip exactly when `Z=0`, and test invariance to
  inactive padding, active-row count, accumulation, and buffer composition.
- [ ] Check one sufficiently small gradient step decreases fixed-state `q`.
- [ ] Treat detector age/onset weights and behavior-policy fields as detached inputs.

**Gate G2-math:** finite-difference relative/absolute error `<=2e-4` in FP32 and all invariants pass.

---

## 8. Task 5 — CUDA BF16 loss/backward and attribution

**Files:** `gpu_kernels.{h,cu}`, `chiron-arrest-test.cpp`, test dispatcher/CMake.

Use compact `[T,K_A]` IDs, row counts, and weights. Implement:

1. hazard stats: FP32-accumulated, deduplicated `q`, total active weight `Z`, weighted loss,
   behavior/current ratio, overflow/clamp/error counters;
2. dense BF16-dlogit base `-weight*p_j*q/(1-q)` over `[T,V]`;
3. sparse owner correction `+weight*p_j/(1-q)` at each hazard ID;
4. direct component-norm/inner-product telemetry before adding the normalized field to total dlogits.

Reuse ECHO's validated dense-plus-sparse **pattern**, not its teacher-forced target construction.

- [ ] CPU/GPU parity on random and adversarial rows; tolerance `<=2e-3` for BF16 probabilities.
- [ ] Empty rows and `coef=0` write exact zero/no-op.
- [ ] Duplicates never double-count; BF16 probability-sum drift is covered; `q>=1-eps` is clamped only
  for arithmetic and counted, with `.1%` active-row clamp frequency as the preregistered kill bar.
- [ ] Guard `T,V,K_A`, null pointers, and CUDA errors in the public wrapper.
- [ ] Measure incremental allocation (`<=8 MiB` target, `50 MiB` kill) and active kernel wall (`<=1%`
  of a normal micro-step).
- [ ] Add a component replay: save pre-ARREST dlogits, inject only ARREST, verify measured norm/dot and
  one-step hazard-mass direction.

**Gate G2-kernel:** all math/parity/invariant/memory/timing bars pass; no existing ECHO/CE/Z test changes.

---

## 9. Task 6 — default-off trainer integration

**Files:** trainer `chiron_main.cpp`, `chiron_arrest_buffer.{h,cpp}`, `run.sh`, `CMakeLists.txt`.

### Flags

```text
--arrest-coef F             default 0
--arrest-buffer PATH        required iff coef>0
--arrest-run-state PATH     atomic provenance/resume sidecar iff coef>0
--arrest-every N            default 25 optimizer steps
--arrest-warmup N           default 0
--arrest-max-age N          default 2 rounds (collector metadata)
--arrest-max-stale-frac F   default .10
--arrest-eps F              numerical denominator guard
--arrest-grad-probe         telemetry only
--arrest-config-smoke       parse/validate/exit before CUDA
```

`run.sh` mirrors them as `ARREST_COEF`, `ARREST_BUFFER`, `ARREST_RUN_STATE`, `ARREST_EVERY`, and
`ARREST_WARMUP`; none are set in existing production recipes.

### Integration sequence

- [ ] Parse and print flags only after strict validation. Reject `coef>0` without valid CHAB/sidecar,
  nonpositive cadence, incompatible T/k/V, block-ratio mismatch, calibration/eval split, noncausal
  flags, decoder/detector mismatch, or optimizer-recipe mismatch.
- [ ] At `coef=0`, do not construct the loader or allocate hazard device buffers.
- [ ] After four ordinary clean accumulation micro-steps and before the same Adam update/grad-norm
  boundary, execute one extra rollout forward/backward only when `optimizer_step%arrest_every==0`.
- [ ] Reuse model scratch sequentially; pad a reduced CHAB record to full `T`, call the trainer forward
  in serving-equivalent eval mode (`isTraining=false`, PIED/PACT/other training-only fields off), run
  softmax and ARREST dlogits, then existing backward accumulation. Do not add CE/Z/other auxiliary
  fields on generated targets.
- [ ] Before the first update, compare collector raw probabilities/hazard mass to trainer replay at
  sentinel rows using the G1 tolerance; abort on mismatch.
- [ ] Scale by `arrest-coef/Z` so the field is invariant to ordinary `accum`, active-row count, and
  record mix; document exact normalization in telemetry.
- [ ] Add `Scratch::arrestReplay` (or equivalent pass-kind) only on the active branch. Existing clean
  SCFA stochastic-rounding counter calls remain textually/behaviorally unchanged; replay calls use a
  separate deterministic hash of `(optimizer_step,record_index,layer,op)` and never increment clean
  static counters. PIED remains off. Add clean-counter/checkpoint parity and resume tests.
- [ ] Compare current versus raw behavior hazard mass using
  `abs(log((q+1e-6)/(q_behavior+1e-6)))>log(2)`. If this holds on more than
  `--arrest-max-stale-frac` active rows, stop before Adam and request a refreshed buffer.
- [ ] Emit loss, active weight/rows, mean/p95/max current and behavior `q`, ratio histogram, onset,
  overflow/clamps, component grad norm/dot/cosine, loader age, and amortized wall.
- [ ] Do not serialize ARREST state in model checkpoints. Atomically update the run sidecar with base
  checkpoint SHA, CHAB SHA, detector/decoder/geometry hashes, round start, current optimizer step, and
  output checkpoint chain.

### Exact-parity tests

- [ ] One optimizer step baseline versus explicit `--arrest-coef 0`: loss lines and checkpoint bytes
  identical.
- [ ] Existing command with no ARREST flags: parse and behavior identical.
- [ ] Resume at coefficient zero: checkpoint EOF/sections unchanged.

**Gate G2-integration:** coefficient-zero bit parity, unchanged serving/checkpoint contract, bounded RNG.

---

## 10. Task 7 — smoke and verification ladder

**Create:** `scripts/chiron_arrest_smoke.sh`.

The script must use tiny dimensions and synthetic CHAB records; it is not an efficacy experiment.

- [ ] **Config smoke:** valid flags accept; missing/wrong hash, T/k/V, flags, calibration/eval split,
  corrupt record/sidecar, bad block ratio, and bad cadence reject before CUDA.
- [ ] **One-step no-op:** omitted flags and coefficient zero are byte-identical.
- [ ] **One-step active:** exactly one auxiliary pass; finite loss/grad; hazard mass drops on fixture.
- [ ] **Resume:** fresh round requires CHAB source SHA to match the loaded base checkpoint; mid-round
  resume validates the loaded checkpoint against the sidecar chain and reproduces record order.
- [ ] **Replay parity:** iterative serving and trainer eval-mode replay match scored logits/hazard mass.
- [ ] **PIED/SR interaction:** auxiliary replay is PIED-off and uses the isolated SR key domain;
  ordinary PIED and clean static-counter stochastic-rounding streams remain pinned.
- [ ] **Memory:** peak delta within G2 bar.

Run focused to broad:

```bash
cd /home/robert/dev/glades-ml
bash unit-tests/test.sh chiron-generate
bash unit-tests/test.sh chiron-arrest
bash unit-tests/test.sh chiron
bash unit-tests/test.sh checkpoint

cd /home/robert/dev/glades-trainer
bash build.sh
bash scripts/chiron_arrest_smoke.sh
```

Then use `test_runner infer`, `test_runner run`, and `gitverify` before any completion claim. Run
Semgrep/CodeGraph supplement only if changed-code scope or security review warrants it; their absence is
not a blocker.

---

## 11. Task 8 — gated GPU program

Every arc is preregistered in `experiment_ledger` before launch. Attach commands/logs to `run_deck`,
record every PASS/FAIL/NULL, and bank exact checkpoint/buffer hashes. Never tune on the fixed eval set.

### E0 — bit parity and collection benchmark

- baseline versus explicit coefficient zero, same binary/seed/data;
- exact one-step loss/checkpoint parity;
- row-only observer/token parity and reduced/full/trainer replay parity;
- atomic sidecar fresh/resume validation;
- rollout collection `<=10%` target and `<=25%` hard kill of proposed fine-tune GPU-hours.

### E1 — induced-loop mechanism fixture

Use a tiny/small model or controlled logit fixture for mechanism correctness; no flagship efficacy
claim. After it passes, use the frozen causal checkpoint for a no-update gradient calibration only.

- ARREST-only fixture step reduces fixed-state hazard mass;
- no-hazard rows are exact zero;
- finite gradients and clamp rate `<.1%` outside adversarial tests;
- active kernel `<=1%` micro-step wall;
- on actual rollout-training records, a no-update probe over `{.003,.01,.03,.1}` selects the smallest
  `arrest-coef` with ARREST parameter-gradient norm 2–5% of matched clean CE; fail if none qualifies,
  otherwise freeze before E2.

### E2 — 500-step single-seed pilot

Fresh same-binary control/treatment from one causal checkpoint; seed 1337; identical restored optimizer
state, continuation LR schedule, clean data order, and PIED stream. The treatment uses five 100-step
mini-rounds or one preregistered equivalent refresh schedule.

**PASS only if:**

- newly generated hazard mass falls `>=20%` versus control;
- collapse incidence falls `>=50%` relatively on the reduced eval stratum and is non-worse on the
  full-length stratum;
- validation NLL delta `<=+0.02`; top-1 loss `<=0.5` point;
- zero skips/nonfinites; p99 grad ratio `<=1.15`;
- aggregate JS-1/JS-2, punctuation/control/EOS rates, realized length, and blind gibberish meet frozen
  paired noninferiority margins.

A mechanism miss closes the minimal objective. Do not compensate with a coefficient sweep.

### E3 — 2,500-step three-seed recovery gate

Seeds `1337`, `2024`, `4242`, each with a fresh same-binary control and matched optimizer/LR/data/PIED
state. Five 500-step outer rounds are the default; coefficient and cadence are preregistered from E1,
confirmed—not tuned—by E2, and fixed across seeds.

**PASS only if all bars hold:**

- mean validation NLL delta `<=0.00`, paired 95% upper bound `<=+0.01`, no seed worse than `+0.02`;
- reduced-stratum collapse incidence reduced `>=80%` relatively and absolute incidence `<=10%` under
  raw greedy/nucleus; any full stratum with baseline incidence `>=20%` meets the same relative bar, and
  all other full/production strata are non-worse;
- p95 `max_run<=4`, p95 `cycle_max<=.35`, repeated-span coverage non-worse;
- clean-continuation NLL delta `<=+0.02`, top-1 loss `<=0.5` point;
- no domain stratum regresses incidence by over 5 points;
- aggregate JS/entropy/margin/punctuation/control/EOS/length metrics meet frozen paired
  noninferiority margins;
- zero skips/nonfinites, p99 grad ratio `<=1.15`;
- amortized training wall `<=1.5%`, collection `<=10%` target / `<=25%` hard kill of GPU-hours;
- component attribution shows direct ARREST hazard-mass reduction.

### E4 — generalization/promotion

Evaluate one selected E3 checkpoint on 256 **new** disjoint contexts and at least 64 blinded pairs.

- context-clustered, multiplicity-adjusted 95% upper bound on collapse incidence `<10%` in both raw
  modes, including full-length contexts;
- at least two blinded raters, randomized order, ties, and reported agreement;
- treatment-minus-control coherence preference `>=30` points with paired-bootstrap 95% lower bound
  `>0`; gibberish meets the frozen noninferiority margin;
- wide causal NLL/top-1 and serving/checkpoint parity retain E3 bars.

A passing artifact is labeled a **generation checkpoint**. It does not silently replace the perplexity
flagship.

---

## 12. Task 9 — closeout and decisions

- [ ] Write a result report with exact commits, commands, manifests, checkpoint/buffer hashes,
  GPU-hours, all seed-level metrics, confidence intervals, and failed gates.
- [ ] If G0a/G0b/G1 is NO-GO, write the applicable calibration, transfer, replay, or cost no-go and
  stop before objective code.
- [ ] If E2/E3 fails, write `docs/decisions/YYYY-MM-DD-arrest-no-go.md`; do not rebrand as ECHO or tune
  detector thresholds post hoc.
- [ ] If loops fall but coherence fails, classify PARTIAL and propose PAIR as a **new** preregistered
  program; do not bundle it into ARREST.
- [ ] If E4 passes, update the design status and add a separate owner-reviewed promotion record.
- [ ] Run `git diff --check`, inspect the final diff, then use the repository verification workflow.

---

## 13. Expected commit groups (when explicitly requested)

1. `research: freeze CHIRON generation-recovery manifests`
2. `chiron-generate: robust repetition and cycle detector`
3. `chiron-generate: row-only logits and rollout collector support`
4. `chiron: ARREST hazard-loss reference and CUDA kernels` *(only after G0a/G0b/G1 PASS)*
5. `trainer: default-off ARREST rollout-buffer integration`
6. `test: ARREST parity, resume, and smoke coverage`
7. `research: record ARREST gate verdict`

Keep glades source/tests together and trainer source/smoke together. Do not commit generated rollouts,
checkpoints, raw logs, or unrelated working-tree changes.
