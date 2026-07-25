# CHIRON ARREST — Phased Implementation Plan

**Date:** 2026-07-16

**Status:** post-285k E1b diagnosis closed; ARREST execution plan; no ARREST GPU run or objective implementation is authorized by this document

**Normative sources:**

- `docs/superpowers/specs/2026-07-16-chiron-generation-aware-training-design.md`
- `docs/superpowers/plans/2026-07-16-chiron-generation-aware-training.md`
- `docs/decisions/2026-07-16-echo-regularizer-no-go.md`

This document converts the reviewed design into dependency-ordered engineering phases. If it conflicts
with the normative sources, the stricter stop rule or acceptance bar wins.

> **Operational boundary.** Phases 0 and 1 are CPU/documentation work. GPU-dependent tests and Phase 2
> collection wait for owner scheduling. The prior causal FineWeb run has ended, but E1b completion does
> not authorize an ARREST run. Phase 4 loss code does not begin until a fully trained causal checkpoint
> passes G0a, G0b, and G1.

---

## 0. Completed prerequisite — post-285k E1b diagnosis (2026-07-25)

The preregistered held-out experiment `exp-chiron-e1b-held-out-factual` is closed. Its scope was limited
to deciding whether the earlier small-suite signal was a synchronized checkpoint effect (**H-S**) or
large enough legacy `chiron_generate` synchronization instability (**H-R**). It did not test ARREST and
did not authorize training, checkpoint mutation, source changes, model attribution, or E2/E3 work.

### E1b integrity and budget

| Gate | Result | Evidence |
|---|---|---|
| Frozen manifest E0 | **PASS** | 96 disjoint prompts: 32 factual, 32 multilingual, 32 nonoverlapping FineWeb windows; 24 frozen race sentinels; no EOS in selected FineWeb windows |
| Synchronized packet integrity | **PASS** | 288/288 packets; zero downloaded-row argmax mismatches; zero nonfinite logits |
| Legacy packet/command integrity | **PASS** | 360/360 packets; five identical command hashes per checkpoint; zero nonfinite logits |
| Checkpoint immutability | **PASS** | all three post-run checkpoint SHA-256 values equal their frozen manifest values |
| Frozen compute budget | **PASS** | exactly 24,192 generated token-forwards; 8,181 wall seconds; 2.2725 GPU-hours, below the 2.5 GPU-hour and 3 wall-hour caps |

The pi harness was restarted during the final legacy arm. The original process tree remained healthy,
was reattached rather than duplicated, and exited zero. No arm was rerun.

### Preregistered conclusions

For H-S, each confirmatory subgroup had to pass all four core bars after the fixed Bonferroni correction.
Positive distinct-4 deltas and negative max-run deltas denote improvement, not regression.

| 305k−285k subgroup | Distinct-4 delta (bar `<=−0.03`) | One-sided 97.5% bootstrap upper bound (bar `<0`) | Max-run delta (bar `>=+1.5`) | Worsened fraction (bar `>=60%`) | Verdict |
|---|---:|---:|---:|---:|---|
| Factual (`n=32`) | `+0.009221` | `+0.035348` | `−0.1875` | `37.5%` | **H-S FAIL** |
| Multilingual (`n=32`) | `+0.005635` | `+0.016906` | `−2.09375` | `15.625%` | **H-S FAIL** |

The three specificity controls all passed: in-domain distinct-4 delta `+0.037398 > −0.01`, in-domain
max-run delta `−9.28125 < +0.5`, and reference NLL delta `−0.010846 < +0.05` nat. The required 300k
report also opposed regression: factual distinct-4/max-run `+0.014344/−0.03125`, multilingual
`+0.002561/−1.78125`, and reference NLL `−0.009628` nat. Keyword accuracy remained descriptively at
floor (`0/32` for each subgroup at each checkpoint) and did not enter the decision.

H-R **PASS**: all `24/24` sentinels showed cross-repeat disagreement or disagreement with synchronized
raw argmax at each checkpoint, with 92 token-level legacy/posthoc-row argmax mismatches overall.
Legacy subgroup-mean distinct-4 repeat ranges exceeded the preregistered `0.03` sufficiency bar
(facts: `0.01282/0.03205/0.01923`; multilingual: `0.03846/0.03846/0.05128` at 285k/300k/305k).
The multilingual matched-repeat 305k−285k deltas reproduced the adverse direction in all five repeats
(distinct-4 `−0.03205` to `−0.09615`; max-run `+2.5` to `+3.0`). Max-run repeat range alone did not
reach its alternative `1.0` bar.

The frozen decision matrix therefore resolves **H-S FAIL / H-R sufficient PASS → evaluation artifact**.
The held-out synchronized data do not support a post-285k factual or multilingual model regression;
the legacy unsynchronized generation path is sufficiently unstable to explain the earlier small-suite
direction. ARREST efficacy gates must not consume legacy unsynchronized generations. P2.1's explicit
compute-to-transfer ordering, transfer-stream synchronization, and row-equality regression test are now
preconditions for collector evidence. This diagnosis prioritizes that correctness work but does not
advance any ARREST phase or authorize a follow-up run.

### Evidence and immutable hashes

Artifact root: `/home/robert/dev/glades-trainer/logs/chiron_e1b_20260725/`

| Artifact | SHA-256 |
|---|---|
| `manifest.json` | `a8ca3d3f5dd117d3be898532e0151504bda4e05deaa22186415cf39f71622b6f` |
| `analysis.json` | `31152eb4f35f1e57a226295e5400c8be6c02cf3613e239c3ef8a69e34e874d7d` |
| `RESULTS.log` | `8dc65a5b09279959c8d60d013fde715aabb1b4009d1dcb72889a6ccd81d00696` |
| `checkpoint-sha256-after.txt` | `0c7cb7afb4cfe1104eb293eeeba0372dd21145a97e865aec8f08a8d7e9633118` |
| `SHA256SUMS` (864 evidence files; excludes itself) | `267ab372d9dd422111989583fa2614a7115726e0f584ec28594f9e9b2518d20d` |

Frozen checkpoint SHA-256 values, reverified after E1b:

- 285k: `921178c25bfabcd0b808bf3bec0a502e0031626293ca36b72b53dbaaea0e4bfc`
- 300k: `2a70db1b64061f810274346eb6b72ce73d5e0a696ade6f1be07dff890ef42c66`
- 305k: `01513097b09564aa476b27ea03dce558af4ebca732cf61a423aa61bd579d5c31`

The complete `SHA256SUMS` index was reverified with `sha256sum -c` after terminal analysis; all 864
entries passed. Raw packets, decoded generations, checkpoints, and logs remain local and are not
committed.

---

## 1. End state and invariants

The implemented system consists of:

1. a deterministic, prefix-online repetition detector and evaluator;
2. a row-only observed generation API;
3. an offline CHAB rollout collector and validated local buffer format;
4. CPU/CUDA aggregate hazard-mass loss primitives;
5. a default-off trainer replay path using serving-equivalent PIED-off states;
6. no model parameters, serving fields, or checkpoint-format sections;
7. an atomic sidecar for buffer/round/resume provenance; and
8. a gated experiment ladder ending in a separately labeled generation checkpoint.

The following remain immutable throughout implementation:

- `--arrest-coef 0` opens no buffer, allocates no ARREST scratch, and is checkpoint/loss bit-identical;
- rollout training uses training-split contexts only; calibration and evaluation never enter CHAB;
- ECHO stays closed and cannot be composed with ARREST;
- raw model probability, not post-decoder probability, defines hazard mass;
- sliding-window trajectories are diagnostic-only because one teacher-forced record cannot replay them;
- reduced serving geometry is allowed only after `T/k`-preserving parity;
- the auxiliary replay is PIED-off and cannot advance clean PIED or SCFA stochastic-rounding streams;
- no efficacy gate is passed by distinct-n, hazard mass, or one seed alone.

---

## 2. Dependency graph and phase summary

```text
P0 contract/manifests/analysis
  -> P1 detector + G0a
  -> P2 observed generation + CHAB + collector
  -> P3 fully-trained causal G0b/G1 branch point
       FAIL/SLIDE-ONLY/COST FAIL -> STOP and decision record
       PASS
         -> P4 CPU/CUDA objective
         -> P5 trainer integration + G2
         -> P6 E0/E1 + 500-step E2 pilot
              FAIL -> STOP and no-go
              PASS
                -> P7 2,500-step three-seed E3
                     FAIL/PARTIAL -> decision record
                     PASS
                       -> P8 held-out E4 promotion decision
```

| Phase | Purpose | GPU? | Hard output/gate |
|---|---|---:|---|
| P0 | Freeze schemas, splits, statistics | No | M0 |
| P1 | Detector and hazard construction | CPU first | G0a |
| P2 | Observed generation, CHAB, collector | Tests/benchmark | G1 engineering evidence |
| P3 | Causal transfer and collection feasibility | Yes, scheduled | G0b + G1 PASS |
| P4 | Aggregate hazard loss | CUDA tests | G2 math/kernel |
| P5 | Trainer replay and default-off integration | CUDA tests | G2 integration |
| P6 | Parity, dose calibration, 500-step pilot | Yes | E0/E1/E2 |
| P7 | Three matched 2,500-step pairs | Yes | E3/G4 |
| P8 | New-context generalization/promotion | Yes | E4/G5 |

No phase may consume artifacts from a later phase. Every generated artifact records source commit,
checkpoint SHA-256, manifest hash, detector hash, decoder table, and command.

### 2.1 Two parity-gated performance passes

Correctness baselines land first. Optimizations are retained only when bit/tolerance parity holds and a
measured resource bar improves; neither pass may weaken G1/G2.

1. **PERF-A — compact sparse hazard layout (after P4 correctness).** Replace `[T,16]` hazard metadata
   with active-row CSR plus one `rowSlot[T]` map. Target metadata falls from about 1.3 MiB to `<128 KiB`
   at `T=16384,H=256`, while CPU/GPU loss and dlogits remain within G2 tolerances.
2. **PERF-B — asynchronous CHAB prefetch (after P5 correctness).** Double-buffer the compact metadata,
   read/decode the next record on CPU, and upload it on the transfer stream during clean accumulation.
   The auxiliary pass waits on one event. Keep only if staging wall drops `>=50%`, total auxiliary wall
   does not regress, clean RNG/checkpoint parity remains exact, and amortized overhead stays `<=1.5%`.

---

## 3. Phase P0 — freeze contracts, manifests, and analysis

- **Dependencies:** reviewed docs at `1a1df09f8` or a documented successor.
- **GPU:** forbidden.
- **Exit:** M0 PASS.
- **Execution status (2026-07-25):** **M0 PASS**. The real 384-row calibration index,
  512-row training manifest, and 128-row evaluation manifest reproduced byte-for-byte. The logical
  corpus digest is `62b69479629b6683bc62e05fa71053cf26f9f0eca09025ac782184b38c49e10c`;
  local verification packet SHA-256 is
  `0636f4391764af3bddee0bab016eee0c1d2ddf9baf7d02b9b143fb2bce1c5bcd`. Evidence and privacy
  boundaries are recorded in the companion trainer's `research/generation-aware/README.md` at
  `1e139e48cc419916b5d9c302faa0a20641370135`. P1/G0a
  remains pending; M0 does not authorize model inference, training, P2 collection, or any E1b model
  claim.

### P0.1 Manifest generator and schemas

**Files**

- Create `/home/robert/dev/glades-trainer/scripts/build_chiron_generation_manifests.py`
- Create `research/generation-aware/README.md`
- Generate `research/generation-aware/detector_calibration_v1.tsv`
- Generate local, gitignored `artifacts/chiron_arrest/detector_calibration_v1.tok.bin`
- Generate `research/generation-aware/train_manifest_v1.tsv`
- Generate `research/generation-aware/eval_manifest_v1.tsv`

**CLI/API**

```text
build_chiron_generation_manifests.py
  --corpus PATH --vocab PATH --seed N
  --inventory-cache PATH [--train-shards CSV]
  --calibration-out PATH --calibration-payload PATH
  --train-out PATH --eval-out PATH [--self-test]
```

Train/evaluation row contract:

```text
split, stratum, domain, corpus_sha256, shard, document_id, token_offset,
active_T, active_k, prompt_tokens, rollout_tokens
```

Calibration index contract:

```text
sample_id, domain, source_kind, blind_label, payload_sha256, token_offset, token_count
```

The corpus input is the existing pretokenized layout (`train/*.tok.bin`, `val.tok.bin`, `vocab.bpe`).
The generator samples EOS/BOS-delimited documents with bounded mmap searches, decodes only selected
windows with the GLADES BPE merge table, and assigns deterministic auditable domain heuristics. It never
loads a shard into RAM. A reusable inventory caches full shard SHA-256 by path/size/mtime/inode; a changed
identity forces rehash, while unchanged multi-gigabyte shards are not re-read.

`eval_manifest_v1.tsv` contains 128 contexts: 64 prose, 24 code, 16 math/markup, 24 mixed;
96 reduced no-slide, 32 full no-slide, and 16 of the full rows tagged for sliding diagnostics.
`train_manifest_v1.tsv` contains at least 512 training-split contexts (at least 384 reduced and 128
full, domain-stratified). Calibration contains at least
128 positive loops and 64 real negatives per domain. The committed calibration TSV contains labels and
hashes only; token payloads/model outputs stay in the local gitignored artifact.

**Tests**

- `python3 scripts/build_chiron_generation_manifests.py --self-test`
- regenerate into a temporary directory and compare byte-for-byte;
- fail on duplicate document IDs across splits, overlapping offsets, shard/document crossing, missing
  BOS/EOS bounds, short contexts, wrong corpus/vocab hash, stale inventory identity, non-divisible
  `T/k`, or unexpected row counts;
- validate reduced geometry `2048/128` and full geometry `16384/1024` without claiming parity.

**Acceptance — M0**

- deterministic bytes and stable hashes;
- zero calibration/train/eval document overlap;
- every train row names the training split and every eval row the held-out split;
- README pins script/corpus-inventory/vocab/manifest hashes, domain-heuristic version, and privacy rules;
- peak Python RSS is `<128 MiB` on synthetic large sparse TOKB and generation performs no whole-shard
  token materialization.

### P0.2 Analysis tool and frozen statistical contract

**Files**

- Create `/home/robert/dev/glades-trainer/scripts/chiron_arrest_eval.py`

**CLI/API**

```text
chiron_arrest_eval.py
  --mode detector --predictions PATH --out-prefix PATH
or
  --mode generation --manifest PATH --samples PATH
  --control-label NAME --treatment-label NAME --decoder-table PATH
  --bootstrap-seed N --out-prefix PATH
  [--self-test]
```

Detector mode owns exact one-sided sensitivity/FPR confidence bounds overall and by domain. Generation
mode owns context-paired effects, context-clustered bootstrap, Holm correction for the two raw endpoints
and domain secondaries, seed-level tables, early-stop/missing handling, and JSON+Markdown output. It
never tunes detector thresholds.

**Tests**

- detector confusion-matrix fixture with known one-sided sensitivity/FPR bounds;
- synthetic paired generation fixture with known 20-point effect;
- duplicate samples from one context do not increase effective `n`;
- arm-order permutation negates the effect;
- Holm-adjusted intervals are no narrower than unadjusted intervals;
- early-stop/EOS rows use the preregistered length convention;
- `--self-test` is CPU-only and deterministic.

**Acceptance**

- script hash frozen before G0b;
- all interval/multiplicity tests pass;
- output includes point estimates, confidence bounds, context count, decoder seeds, and missing count.

### P0.3 Phase verification

```bash
cd /home/robert/dev/glades-trainer
python3 scripts/build_chiron_generation_manifests.py --self-test
python3 scripts/chiron_arrest_eval.py --self-test
```

Do not commit decoded corpus excerpts, model outputs, checkpoints, or raw logs.

---

## 4. Phase P1 — detector, metrics, and G0a calibration

- **Dependencies:** P0/M0.
- **GPU:** CPU implementation/tests may run while GPU is occupied.
- **Exit:** G0a PASS and frozen detector hash.

### P1.1 Public detector API

**Files**

- Modify `Backend/Machine Learning/Networks/chiron_generate.h`
- Modify `Backend/Machine Learning/Networks/chiron_generate.cpp`

**Proposed C++98 API**

```cpp
struct ChironRepetitionConfig {
    int maxPeriod;          // 64
    int minCycleSupport;    // 32; effective support=max(32,2p)
    float cycleThreshold;   // .80
    int ngramWindow;        // 128
    int maxHazards;         // 16
    ChironRepetitionConfig();
};

struct ChironRepetitionMetrics {
    double distinct1, distinct2, distinct4;
    double repeatFraction, cycleMax, repeatedSpanCoverage;
    int cyclePeriod, longestSuffixCopy, maxRun, collapseOnset;
    bool collapsed;
};

struct ChironHazardRow {
    int tokenIds[16];
    float confidence[16];
    int count;
    bool overflow;
};

void chiron_repetition_metrics(const std::vector<int>& generated,
                               const ChironRepetitionConfig& config,
                               ChironRepetitionMetrics& out);
void chiron_repetition_hazards(const std::vector<int>& generated,
                               const ChironRepetitionConfig& config,
                               std::vector<ChironHazardRow>& rows,
                               std::vector<float>& rowWeights);
```

Keep `chiron_degeneration_metrics(...)` unchanged as a wrapper returning the legacy distinct-4/max-run
values.

**Implementation rules**

- metrics search periods 1–64 with support `max(32,2p)`;
- hazards use only `generated[0:t]`: run continuation, period continuation, n-gram closure 2–8;
- deduplicate by token ID, retain maximum confidence, sort deterministically, expose overflow;
- no future tokens, decoded strings, post-decoder logits, or corpus labels enter hazard construction.

### P1.2 Detector tests and selectors

**Files**

- Modify `unit-tests/Backend/Machine Learning/chiron-generate-test.h`
- Modify `unit-tests/Backend/Machine Learning/chiron-generate-test.cpp`
- Modify `unit-tests/main.cpp`

Add individual tests for constant, periods 2/3/8/16/32/64, high-distinct-4 template, repeated span,
varied soup, real continuation, code delimiters, short/empty input, duplicate hazards, overflow, and
append invariance. Add `CHIRONGenerateCpuUnitTest()` and selector `chiron-generate-cpu`; retain
`CHIRONGenerateUnitTest()` and selector `chiron-generate` for the existing GPU-bearing aggregate.

**Tests**

```bash
cd /home/robert/dev/glades-ml
bash unit-tests/test.sh chiron-generate-cpu
bash unit-tests/test.sh chiron-generate   # only when GPU is scheduled
```

### P1.3 Calibration scorer

**Files in glades-trainer**

- Create `tools/chiron_arrest_detect.cpp`
- Modify `CMakeLists.txt` to add CPU target `chiron_arrest_detect` linked with the installed glades
  library and existing common libraries

**CLI/API**

```text
chiron_arrest_detect
  --calibration-index PATH --calibration-payload PATH
  --detector-config PATH --out PATH [--self-test]
```

The tool verifies payload/config hashes, reads each indexed token span, calls
`chiron_repetition_metrics(...)` and `chiron_repetition_hazards(...)`, and writes sample ID, blind label,
prediction, onset, period, run/span metrics, overflow, and detector hash. It never reads evaluation or
training manifests. `--self-test` builds a tiny local payload with one period loop and one real-like
negative and compares deterministic output bytes.

### P1.4 G0a calibration

Run `chiron_arrest_detect` over the disjoint calibration set and pass its output to the frozen P0
analysis script. Blind labels must provide at least 128 positive loops and 64 real negatives per domain.

**Acceptance — G0a**

- one-sided 95% sensitivity lower bound `>=.95`;
- overall FPR upper bound `<=.05`;
- each domain FPR upper bound `<=.08`;
- append invariance and deterministic goldens pass;
- detector source/config/data hashes frozen before G0b.

If G0a fails, correct detector logic only against calibration data, version the config, and repeat G0a.
Evaluation rows remain sealed.

```bash
cd /home/robert/dev/glades-trainer
bash build.sh
./build/chiron_arrest_detect --self-test
```

---

## 5. Phase P2 — observed generation, CHAB, and collector

- **Dependencies:** P1 detector API; M0 manifests.
- **GPU:** implementation can proceed; parity/benchmark waits for scheduling.
- **Exit:** collector engineering complete; final G1 closes in P3.

### P2.1 Backward-compatible observed generation

**Files**

- Modify `Backend/Machine Learning/Networks/chiron_generate.h`
- Modify `Backend/Machine Learning/Networks/chiron_generate.cpp`
- Modify `unit-tests/Backend/Machine Learning/chiron-generate-test.{h,cpp}`

**Proposed API**

```cpp
struct ChironGenerationStep {
    int step;
    int logitsRow;
    int sampledToken;
    const float* rawLogits;                    // valid during callback only
    int vocabSize;
    const std::vector<int>* contextBeforeSample;
};
typedef bool (*ChironGenerationStepObserver)(void* ctx,
                                              const ChironGenerationStep& step);

bool chiron_generate_observed(/* existing arguments */,
                              ChironGenerationStepObserver observer,
                              void* observerCtx,
                              std::vector<int>* outTokens);
```

The existing `chiron_generate(...)` remains source-compatible and calls the observed entry point with a
null observer.

**Row transfer implementation**

- remove host `logitsAll[T*V]` from `chiron_generate.cpp`;
- after `chiron_eval_forward`, record an event on `gpu::computeStream()`;
- make `gpu::transferStream()` wait on that event;
- call existing `gpu::device_memcpy_d2h(logitsRow, s.logits.data()+row*V, V*sizeof(float))`;
- require `gpu::synchronizeTransferStream()` success before sampling;
- destroy/reuse the event without adding `GpuBuffer::downloadRange` or a gather kernel.

**Tests**

- row-only logits equal the same row of a full download;
- observer context excludes the sampled token; observer token equals emitted token;
- observer failure propagates without an extra RNG draw;
- old/new API token streams, sink calls, and MT19937 draw counts are identical;
- first/last row, invalid prompt, and CUDA-absent stubs retain existing behavior.

### P2.2 CHAB v1 codec and atomic run state

**Files in glades-trainer**

- Create `trainer/chiron_arrest_buffer.h`
- Create `trainer/chiron_arrest_buffer.cpp`
- Create `tools/chiron_arrest_buffer_test.cpp`
- Modify `CMakeLists.txt`

Create a shared `chiron_arrest_io` static library and link it into `glades_chiron_train`,
`chiron_rollout_collect`, and `chiron_arrest_buffer_test`.

**Proposed API**

```cpp
struct ChironArrestHeader;
struct ChironArrestRecord;
struct ChironArrestRunState;

class ChironArrestWriter {
public:
    bool open(const std::string& path, const ChironArrestHeader& header,
              std::string& error);
    bool append(const ChironArrestRecord& record, std::string& error);
    bool finalize(std::string& sha256, std::string& error);
};

class ChironArrestReader {
public:
    bool open(const std::string& path, std::string& error);
    bool validate(const ChironArrestHeader& expected, std::string& error) const;
    bool read(size_t index, ChironArrestRecord& record, std::string& error) const;
    size_t size() const;
};

bool chiron_arrest_sha256_file(const std::string&, std::string& hexDigest,
                               std::string& error);
bool chiron_arrest_load_run_state(const std::string&, ChironArrestRunState&,
                                  std::string& error);
bool chiron_arrest_write_run_state_atomic(const std::string&,
                                          const ChironArrestRunState&,
                                          std::string& error);
```

CHAB v1 is single-geometry: header `active_T,active_k` applies to every record in the file. Reduced and
full records therefore use separate collector invocations/files; the first prototype never mixes them
behind one `--arrest-buffer`. A mixed-geometry buffer requires a versioned format/design amendment.
CHAB uses explicit fixed-width little-endian integer/IEEE-754 encoders; never serialize C++ struct
padding. Records follow the reviewed schema: source/active `T,k`; checkpoint/vocab/corpus/manifest/detector/
decoder/geometry hashes; flags; token IDs; hazard IDs/counts/weights; deterministic FP64 `weightSum`;
raw behavior `q`; raw sampled-token log-probability; metrics; CRC32. The sidecar pins base checkpoint, CHAB SHA, round/step interval, current
checkpoint chain, and record order. CHAB writer finalization and sidecar writes both use unique
same-directory temporary files, `fflush`/`fsync`, rename, and parent-directory `fsync`; readers never
open a `.tmp` file. Enforce `V<=65536` for `uint16_t` tokens.
Implement small SHA-256 and CRC32 helpers locally (no new OpenSSL/zlib API dependency) and pin them
with NIST empty/`abc` and CRC32 `123456789` vectors.

**Tests — `chiron_arrest_buffer_test`**

- byte-exact round trip and random-access order;
- NIST SHA-256 empty/`abc` vectors and CRC32 `123456789 == 0xcbf43926`;
- truncation, CRC, whole-file SHA, wrong version/endian, T/k/V, flags, decoder table, geometry, and split;
- duplicate hazard canonicalization, FP64 weight-sum validation, zero-weight training-record rejection,
  and one-token row-alignment sentinel;
- fresh base-checkpoint match versus valid/invalid mid-round sidecar chain;
- interrupted CHAB/sidecar replacement leaves either old or new complete bytes, never a readable
  partial file.

### P2.3 Rollout collector

**Files in glades-trainer**

- Create `tools/chiron_rollout_collect.cpp`
- Modify `CMakeLists.txt`: build `chiron_rollout_collect` from the tool plus `trainer/bpe.cpp`, link
  `chiron_arrest_io` and the existing common glades/shmea/zstd/thread libraries

**Current APIs reused**

- `chiron_tools::read_token_window(...)` and `TokenWindowInfo` from the existing
  `tools/chiron_token_window.h` for raw/TOKB 64-bit-offset prompt reads;
- `chiron_load_model(...)` from `chiron_checkpoint.h`;
- `chiron_resolve_serving(...)`, `ChironServingOverrides::seqLen`, and `scfaKOverride`;
- `ChironEvalScratch::allocate(...)`;
- `chiron_generate_observed(...)` from P2.1;
- `ChironMt19937` and detector APIs from `chiron_generate.h`.

**Collector CLI**

```text
chiron_rollout_collect
  --checkpoint PATH --manifest PATH --vocab PATH --out PATH
  --decoder-table PATH --detector-config PATH --stratum NAME
  --active-t N --active-k N --horizon N
  [--summary PATH] [--dry-run] [--parity-check]
```

`--dry-run` validates every path/hash/flag/geometry and uses `read_token_window` to validate prompt
bounds without CUDA allocation; it estimates forwards, bytes, and GPU-hours. The collector rejects
noncausal flags, missing exact QK gamma/WhiSC state, non-training rows for CHAB, TOKB vocab/header
mismatch, invalid decoder IDs, mixed active geometry in one CHAB, and geometry that changes block width.
Records with zero/nonfinite `weightSum` stay in JSON/TSV evaluation summaries but are omitted from
training CHAB. Add collector
fixtures for raw uint16 and TOKB inputs, including an offset beyond 2^31, reusing the existing
token-window test pattern.

### P2.4 Diagnostic-only trainer replay check

G1 requires trainer/serving parity before ARREST loss code exists. Add an exit-before-training mode rather
than creating a circular dependency on P5.

**Files**

- Modify `/home/robert/dev/glades-trainer/trainer/chiron_main.cpp` (`Config` diagnostic fields,
  `Scratch::arrestReplay`, early dispatch, and forward SR pass-kind branch)
- Link `chiron_arrest_io` into `glades_chiron_train` in `CMakeLists.txt`

**CLI/API**

```text
--arrest-replay-check CHAB_PATH
--arrest-replay-record N
--arrest-replay-summary PATH
```

After normal full-checkpoint initialization but before validation/training, the mode validates CHAB,
pads the selected record to full T, sets a diagnostic `Scratch::arrestReplay` pass kind, calls existing
`forward(...,isTraining=false)`, downloads only the V-wide BF16 probability rows scored by the record,
computes raw q on host, writes a parity summary, and exits with zero optimizer steps. Add the separate
forward SR key domain here so G1 measures the same replay arithmetic P5 will use; every ordinary path
keeps the existing static counters. The mode does not allocate ARREST dlogits, call backward, enter the
training loop, or alter checkpoint format.

**Tests**

- wrong source SHA/split/geometry rejects before forward;
- one-token hazard-row alignment sentinel;
- reported q equals a host sum from the same BF16 row;
- mode exits before Adam/checkpoint write;
- ordinary command without the replay flag is unchanged.

### P2.5 Reduced/full parity and benchmark

Use isolated/reloaded model instances because `chiron_resolve_serving` mutates `dims.T` and `w.scfa.k`.
Compare `2048/128` against `16384/1024` on scored prefix rows, then compare collector q with the
P2.4 trainer replay summary.

**Acceptance — engineering portion of G1**

- row-only/full-row equality and golden token parity;
- reduced/full raw probability and hazard-mass absolute error `<=2e-3`;
- golden sampled tokens identical;
- all CHAB and sidecar tests pass;
- projected collection cost reported, not assumed.

If reduced parity fails, mark reduced geometry unavailable; P3 benchmarks full geometry only.

## 6. Phase P3 — fully trained causal transfer and feasibility branch

- **Dependencies:** P0–P2; fully trained causal checkpoint; explicit GPU scheduling.
- **GPU:** yes.
- **Exit:** G0b and G1 PASS, or a terminal no-go/redirect.
- **Hard rule:** no ARREST loss/reference/kernel or active training-path code before this phase passes;
  P2's exit-before-training replay diagnostic is the only trainer-side prerequisite.

### P3.1 Preregister the run

Open an `experiment_ledger` record before generation. Pin:

- causal checkpoint SHA and required flags (`0x1fd8` or documented successor);
- detector/config/data/analysis hashes;
- 256-token horizon;
- raw greedy, raw nucleus `.8/40/.95`, and production decoder table;
- three stochastic seeds for each reduced context and one for each full context;
- G0b/G1 thresholds and collection-cost formula.

Register collector work with `run_deck`; never run it beside the active 275k trainer.

### P3.2 Zero-step and replay prerequisites

- verify checkpoint EOF, causal SCFA, exact QK gamma, WhiSC state, and zero optimizer steps;
- collect reduced/full parity sentinels with isolated model instances;
- pad reduced CHAB records to full `T` and replay through trainer `forward(...,isTraining=false)` without
  backward; compare raw hazard mass;
- measure row-only generation wall and derive the actual records-per-round budget.

**G1 acceptance**

- reduced/full raw `q` absolute error `<=2e-3` and golden token identity, or reduced mode disabled;
- trainer replay p95 `|q_trainer-q_serving| <= max(2e-3,.05*q_serving)`;
- source/split/decoder/geometry/sidecar validation passes;
- collection cost `<=10%` target and `<=25%` hard kill of proposed fine-tune GPU-hours;
- one training geometry is preregistered per round: reduced after ordinary GO, full after LONG-ONLY;
  no mixed-geometry CHAB is accepted.

### P3.3 Causal transfer evaluation

Collect the frozen 96 reduced, 32 full, and 16 sliding-sentinel contexts. Analyze only with the frozen
P0 tool.

**G0b verdicts**

- **GO:** reduced no-slide incidence `>=20%` in a raw mode and adjusted 95% lower bound `>10%`;
- **NO-GO:** reduced/full no-slide incidence `<=5%` with adjusted upper bounds `<10%`;
- **LONG-ONLY:** continue only if full-geometry CHAB cost passes G1;
- **SLIDE-ONLY:** stop ARREST and open a context-shift diagnosis;
- **INCONCLUSIVE:** expand once with new held-out contexts and unchanged thresholds.

Write `research/CHIRON_ARREST_G0_CAUSAL_TRANSFER_2026_MM_DD.md` and record PASS/FAIL/NULL. Only
G0a+G0b+G1 PASS unlocks P4.

---

## 7. Phase P4 — aggregate hazard objective, CPU then CUDA

- **Dependencies:** P3 PASS.
- **GPU:** CPU reference first; CUDA parity second.
- **Exit:** G2 math/kernel PASS.

### P4.1 CPU reference

**Files**

- Modify `Backend/Machine Learning/Networks/transformer_chiron_ops.h`
- Create `unit-tests/Backend/Machine Learning/chiron-arrest-test.h`
- Create `unit-tests/Backend/Machine Learning/chiron-arrest-test.cpp`
- Modify `unit-tests/Backend/Machine Learning/CMakeLists.txt`
- Modify `unit-tests/main.cpp`

**Proposed API**

```cpp
float chiron_arrest_loss_row_cpu(const float* probs, int V,
                                 const int* hazardIds, int hazardCount,
                                 float weight, float eps);
void chiron_arrest_backward_row_cpu(const float* probs, int V,
                                    const int* hazardIds, int hazardCount,
                                    float weight, float scale, float eps,
                                    float* dlogits);
```

The caller computes `scale=lambda/Z`; behavior-policy fields are detached. CPU code deduplicates IDs,
rejects out-of-range IDs, uses FP64/FP32 reference accumulation as appropriate, and returns exact zero
for empty rows.

**Tests/selector**

Add `CHIRONArrestCpuUnitTest()` and `chiron-arrest-cpu`:

- central finite differences (`<=2e-4` relative/absolute);
- row sum `<=1e-6`;
- one/many/duplicate/empty hazards;
- `q` near zero/one and invalid IDs;
- weight/coefficient linearity and logit-shift invariance;
- one small gradient step reduces fixed-state `q`;
- invariance to inactive padding, active-row replication, accumulation, and buffer mix after `Z`
  normalization.

### P4.2 CUDA stats and backward

**Files**

- Modify `Backend/Machine Learning/Networks/cuda/gpu_kernels.h`
- Modify `Backend/Machine Learning/Networks/cuda/gpu_kernels.cu`
- Extend `chiron-arrest-test.cpp`

**Proposed API**

```cpp
enum ArrestSummaryIndex {
    ARREST_SUM_LOSS = 0, ARREST_SUM_WEIGHT, ARREST_SUM_Q,
    ARREST_ACTIVE_ROWS, ARREST_STALE_ROWS, ARREST_CLAMP_ROWS,
    ARREST_OVERFLOW_ROWS, ARREST_COMPONENT_SQ, ARREST_CLEAN_DLOGITS_SQ,
    ARREST_COMPONENT_DOT_CLEAN, ARREST_SUMMARY_SIZE
};

bool arrest_hazard_stats_bf16(const uint16_t* probs,
                              const int* hazardIds, const uint8_t* counts,
                              const float* weights, const float* behaviorQ,
                              int T, int V, int K, float eps,
                              float* currentQ, float* summary);
bool arrest_hazard_dense_bwd_bf16(const uint16_t* probs,
                                  const float* currentQ, const float* weights,
                                  float scale, int T, int V, float eps,
                                  uint16_t* dlogits);
bool arrest_hazard_scatter_bf16(const uint16_t* probs,
                                const int* hazardIds, const uint8_t* counts,
                                const float* currentQ, const float* weights,
                                float scale, int T, int V, int K, float eps,
                                uint16_t* dlogits);
bool arrest_component_stats_bf16(const uint16_t* cleanDlogits,
                                  const uint16_t* probs,
                                  const int* hazardIds, const uint8_t* counts,
                                  const float* currentQ, const float* weights,
                                  float scale, int T, int V, int K, float eps,
                                  float* summary);
```

`stats` sums BF16 probabilities into FP32, ignores any hazard slot whose ID appeared earlier in the same
at-most-16 row, and computes `Z`, loss, current `q`, clamp/overflow counts, factor-two staleness, and a telemetry-only GPU `Z` cross-check. The
load-bearing scale uses the reader-validated host FP64 `weightSum`, never an atomic floating reduction.
Dense backward writes
`-scale*w*p*q/(1-q)`; sparse scatter adds `scale*w*p/(1-q)` once per unique hazard. The optional
component probe runs before dlogits are overwritten and reports ARREST dlogit norm, clean-last-microstep
dlogit norm, dot, and cosine; it is not part of normal cadence.

Add non-CUDA stubs beside the existing ECHO stubs in `gpu_kernels.h`.

**Tests/selector**

Add `CHIRONArrestGpuParityTest()` and `chiron-arrest`:

- CPU/GPU random and adversarial parity `<=2e-3` for BF16 probabilities;
- duplicate slots do not double count or race;
- coefficient zero and empty rows are exact zero;
- BF16 row-sum drift and `q>=1-eps` clamp accounting;
- stale-row classification around `log(2)` boundary;
- component norm/dot/cosine equals a CPU dense-plus-sparse reference;
- dense-plus-sparse result equals CPU reference;
- active-kernel timing and allocation accounting.

### P4.3 PERF-A — compact sparse hazard layout

After the `[T,16]` correctness reference passes, change the production wrapper to active-row CSR:

```text
activeRows[A], rowOffsets[A+1], hazardIds[N], counts[A],
weights[A], behaviorQ[A], currentQ[A], rowSlot[T]
```

`A<=H=256`, `N<=16A`, and `rowSlot[t]` is `-1` for inactive rows. Stats launches over `A`, scatter over
`N`, and the unavoidable dense dlogits pass performs one `rowSlot[t]` lookup. Keep the original layout
only inside tests as a reference.

**PERF-A tests/gate**

- compact versus reference summary/dlogits parity within the existing CPU/GPU tolerance;
- duplicate, empty, max-`A`, max-`N`, and inactive-row fixtures;
- measured metadata `<128 KiB` at `T=16384,H=256,K=16` and at least 75% below the reference layout;
- no active-kernel wall regression over 1%; otherwise retain the correctness layout and record PERF-A
  as NULL rather than weakening G2.

**G2 math/kernel acceptance**

- all CPU/GPU tests pass;
- clamp frequency `<.1%` outside adversarial fixtures;
- incremental device state `<=8 MiB` target, `<=50 MiB` hard kill; PERF-A additionally targets
  `<128 KiB` metadata;
- active kernels `<=1%` of a normal micro-step;
- no ECHO test or API changes except shared-pattern regression coverage.

Verification:

```bash
cd /home/robert/dev/glades-ml
bash unit-tests/test.sh chiron-arrest-cpu
bash unit-tests/test.sh chiron-arrest
bash unit-tests/test.sh chiron
```

---

## 8. Phase P5 — default-off trainer replay integration

- **Dependencies:** P4/G2 math/kernel; CHAB reader from P2.
- **GPU:** tiny smoke/parity only.
- **Exit:** complete G2 integration PASS.

### P5.1 Config, wrapper, and compatibility interlocks

**Files**

- Modify `/home/robert/dev/glades-trainer/trainer/chiron_main.cpp`
- Modify `/home/robert/dev/glades-trainer/run.sh`
- Modify `/home/robert/dev/glades-trainer/CMakeLists.txt`

Add reviewed flags to `Config`, help, parser, validation, startup logging,
`arrest_should_apply_cfg(const Config&,long long)`, and `run_arrest_config_smoke`:

```text
--arrest-coef, --arrest-buffer, --arrest-run-state, --arrest-every,
--arrest-warmup, --arrest-max-age, --arrest-max-stale-frac,
--arrest-eps, --arrest-grad-probe, --arrest-config-smoke
```

`run.sh` mirrors coefficient, buffer, run state, cadence, and warmup through the existing flagship
argument plumbing used by ECHO. `arrest_should_apply_cfg` is exactly
`coef>0 && step>=warmup && step%every==0` for one-indexed optimizer steps.

For the minimal prototype, `arrest-coef>0` requires causal SCFA, exact QK/WhiSC state,
`--bf16-logits-storage`, fixed T/L, and restored optimizer state. Reject composition with ECHO, SIRA,
PACT, MTP, SAM, ORBIT, ORION, UL2, MEDAL, distillation, CUDA graphs, T/L schedules, or any alternate
objective/optimizer absent from the pinned causal starting recipe. VITALS is also rejected
in the first implementation because auxiliary replay would overwrite its clean-step scratch; relax only
under a separately tested compatibility task. Ordinary gradient clipping/AGC may remain if matched.

At coefficient zero, ignore ARREST paths entirely: no file open, sidecar, allocation, interlock, logging
branch, or altered counter sequence.

### P5.2 Scratch and loader

**Files/symbols**

- `Scratch` in `trainer/chiron_main.cpp`
- `trainer/chiron_arrest_buffer.{h,cpp}`

Allocate only when `arrestCoef>0`. The correctness fallback uses `[T,16]` IDs/counts/weights/current-q.
When PERF-A passes, production instead allocates compact `activeRows`, `rowOffsets`, `hazardIds`,
`weights`, `behaviorQ`, `currentQ`, and `rowSlot[T]`, plus one
`arrest_summary[ARREST_SUMMARY_SIZE]`. Host record/padded token storage and `Scratch::arrestReplay` are
shared.

Assert/log measured bytes: about 1.3 MiB for the fallback and `<128 KiB` target for PERF-A. The loader
rejects training records with zero/nonfinite FP64 `weightSum`, converts the validated sum once to the
kernel scale, and uses `auxOrdinal = count of cadence-eligible steps since sidecar.roundStart` and
`recordIndex = auxOrdinal % recordCount`; no STL hash or mutable RNG is allowed. It validates age and
all header hashes, pads reduced records to full T, and uploads hazards only to scored rows.

### P5.3 Custom backward mode

**Files/symbols in `chiron_main.cpp`**

- Extend `BackwardLossMode` with `BACKWARD_LOSS_ARREST`
- Extend `backward_loss_mode_name(...)`
- Add the BF16 branch in `backward(...)`
- Add `run_arrest_auxiliary(...)`

```cpp
enum ArrestAuxResult { ARREST_AUX_OK, ARREST_AUX_REFRESH_REQUIRED, ARREST_AUX_ERROR };
```

`BACKWARD_LOSS_ARREST`:

1. requires BF16 logits storage;
2. calls ARREST stats and downloads the small summary;
3. returns `ARREST_AUX_REFRESH_REQUIRED` before backward/Adam if stale fraction exceeds `.10`; numerical
   violations return `ARREST_AUX_ERROR`;
4. writes ARREST-only dlogits with dense+scatter kernels;
5. calls the existing reversible model backward with `accumulate=true`;
6. passes `lossNorm=1.0f` because kernels already apply `arrestCoef/Z_host` from the validated record;
7. does not run CE, Z-loss, ECHO, SIRA, MTP, or generated-token targets.

### P5.4 Exact training-loop insertion

Use the existing accumulation boundary around `microStep`, `accumN`, and the final clean `backward(...)`:

```text
clean forward/backward for this micro-step
if ((microStep+1)%accumN == 0 && arrest_should_apply(step+1)):
    run_arrest_auxiliary(..., accumulate=true, optimizer_step=step+1)
++microStep
if incomplete accumulation: continue
++step
existing SAM/global-norm/clipping/Adam/logging
```

The active prototype rejects SAM, so the ARREST field cannot be discarded by SAM's clean-only replay.
Do not increment `microStep`, `tokens_total`, clean loss counters, or clean PIED index for the auxiliary
pass. ARREST loss is separate telemetry and never contaminates validation/running CE. A refresh result
logs `[arrest-refresh-required]`, exits with a dedicated nonzero `ARREST_EXIT_REFRESH` code, and writes no
Adam update or checkpoint; orchestration then collects a new buffer and resumes from the last committed
checkpoint/sidecar.

### P5.5 PIED and stochastic-rounding isolation

P2.4 already adds `Scratch::arrestReplay` and the forward replay SR domain.
`forward(...,isTraining=false)` keeps PIED and other training-only fields inactive. Complete the same
pass-kind treatment in `scfa_attention_backward(...)`:

- leave every existing clean forward/backward counter expression and seed unchanged;
- for replay only, derive the SR key from
  `(bf16WeightsSeed,optimizer_step,record_index,layer,operation)` in a separate domain;
- never increment clean `sr_axpy*` or `sr_pied*` static counters;
- force `whiscCalibrating=false`, verify WhiSC Pbar/Qbar/a are byte-unchanged, and restore all transient
  scratch/pass flags after replay, even on failure.

Coefficient-zero checkpoint parity and treatment/control clean-stream parity are hard acceptance gates.

### P5.6 Sidecar, resume, telemetry, and smoke

**Files**

- Create `/home/robert/dev/glades-trainer/scripts/chiron_arrest_smoke.sh`
- Extend `trainer/chiron_arrest_buffer.{h,cpp}` run-state APIs

Fresh starts require CHAB source SHA to equal the loaded base checkpoint. Mid-round resumes require the
loaded checkpoint in the atomic sidecar chain, compatible optimizer step range, and unchanged CHAB/
detector/decoder/geometry hashes. The sidecar updates only after a successful checkpoint write.

Normal telemetry includes active weight/rows, current/behavior q mean/p95/max, staleness histogram,
onset, overflow/clamp counts, record age, auxiliary wall, and amortized wall. `--arrest-grad-probe`
additionally runs the P4 dlogit component norm/dot/cosine reducer before overwrite; a first-active-step
ARREST-only parameter-gradient replay may reuse the exact-snapshot pattern of
`run_echo_grad_component_probe`, must restore clean gradients at `<=1e-5` relative sumsq, and is never
part of production cadence.

**Smoke cases**

- valid/invalid config before CUDA;
- omitted flags versus explicit coefficient zero checkpoint bytes;
- one active auxiliary pass and fixed-state q reduction;
- stale buffer stops before Adam;
- fresh and mid-round resume;
- iterative serving versus trainer replay;
- clean PIED/SR stream parity and byte-unchanged WhiSC Pbar/Qbar/a across replay;
- memory and cadence accounting.

### P5.7 PERF-B — asynchronous compact-record prefetch

After synchronous P5 correctness passes, add two compact metadata slots. While ordinary accumulation
runs, a CPU worker reads/validates the next CHAB record and the transfer stream uploads the inactive
slot; an event makes the auxiliary pass wait only at consumption. Model tokens and the large `[T,V]`
probability/dlogit buffers are never duplicated.

**PERF-B tests/gate**

- synchronous versus prefetched record bytes, record order, summary, dlogits, checkpoint, and RNG parity;
- cancellation/error/refresh paths join the worker and never consume a partial slot;
- peak incremental VRAM remains `<256 KiB` with two compact slots;
- median staging wall falls `>=50%` over at least 100 synthetic records, total auxiliary wall does not
  regress, and amortized training overhead remains `<=1.5%`;
- if any bar misses, keep synchronous compact staging and record PERF-B as NULL.

**G2 integration acceptance**

- coefficient-zero loss/checkpoint bit parity;
- no serving or checkpoint-format change;
- replay p95 q tolerance from G1 retained;
- clean PIED/SR sequence unchanged;
- no nonfinite/clamp/stale contract violation;
- active-kernel `<=1%`, amortized trainer overhead `<=1.5%`; PERF-B is optional unless all its parity
  and measured-benefit bars pass;
- existing checkpoint, CHIRON, ECHO, and generation tests remain green.

Verification:

```bash
cd /home/robert/dev/glades-ml
bash unit-tests/test.sh chiron-generate
bash unit-tests/test.sh chiron-arrest
bash unit-tests/test.sh chiron
bash unit-tests/test.sh checkpoint

cd /home/robert/dev/glades-trainer
bash build.sh
./build/chiron_arrest_buffer_test
bash scripts/chiron_arrest_smoke.sh
```

---

## 9. Phase P6 — E0/E1 and 500-step matched pilot

- **Dependencies:** P5/G2; explicit owner approval and scheduled GPU.
- **Exit:** E2/G3 PASS or terminal no-go.

### P6.1 Preregister and launch

Open an experiment ledger before any run. Pin commits, checkpoint, CHAB hashes, clean-data order,
optimizer/LR state, PIED stream, decoder seeds, analysis hash, and all bars. Register every process with
`run_deck`.

### P6.2 E0 parity/cost

- same-binary omitted flags versus explicit coefficient zero;
- exact one-step checkpoint/loss parity;
- observed/full/reduced/trainer replay parity;
- fresh/resume sidecar validation;
- collection `<=10%` target and `<=25%` hard kill.

### P6.3 E1 mechanism and coefficient calibration

- induced-loop fixture lowers fixed-state q;
- no-hazard rows exact zero; clamp `<.1%` outside adversarial fixtures;
- no-update parameter-gradient probe on real training-rollout records for
  `{.003,.01,.03,.1}`;
- select the smallest coefficient with ARREST norm 2–5% of matched clean CE; if none qualifies, stop;
- freeze coefficient/cadence before E2.

### P6.4 E2 500-step pair

Run control/treatment from the same causal checkpoint and restored optimizer state, seed 1337, identical
clean data and continuation LR schedule. Use five 100-step refresh rounds or the G1-preregistered
cost-equivalent schedule.

**PASS**

- fresh hazard mass `>=20%` lower than control;
- reduced-stratum collapse incidence `>=50%` lower and full stratum non-worse;
- validation NLL delta `<=+.02`, top-1 loss `<=.5` point;
- zero skips/nonfinites, p99 grad ratio `<=1.15`;
- frozen JS/control/EOS/length/gibberish noninferiority margins pass.

Failure closes the minimal objective; no coefficient or detector sweep follows. Write
`research/CHIRON_ARREST_E2_PILOT_YYYY_MM_DD.md` with exact commits, hashes, commands, GPU-hours, and
paired results.

---

## 10. Phase P7 — three-seed recovery

- **Dependencies:** P6 PASS.
- **Exit:** E3/G4 PASS, FAIL, or PARTIAL.

Run seeds 1337, 2024, and 4242, each with a fresh same-binary control and matched optimizer/LR/data/PIED
state. Use five 500-step rounds and the coefficient/cadence frozen in P6.

**Acceptance — all required**

- mean validation NLL delta `<=0`, paired 95% upper bound `<=+.01`, no seed `>+.02`;
- reduced raw-greedy/nucleus collapse `>=80%` lower and absolute `<=10%`;
- full stratum meets the same relative bar when baseline incidence `>=20%`, otherwise non-worse;
- p95 max-run `<=4`, cycle-max `<=.35`, repeated-span noninferiority;
- clean-continuation NLL `<=+.02`, top-1 loss `<=.5` point;
- no domain incidence regression over 5 points;
- frozen realism/entropy/EOS/length margins pass;
- zero skips/nonfinites, p99 grad ratio `<=1.15`;
- amortized training `<=1.5%`, collection `<=10%` target / `<=25%` hard kill;
- component attribution shows direct ARREST q reduction.

Write `research/CHIRON_ARREST_E3_RECOVERY_YYYY_MM_DD.md` with seed-level gates and record every verdict.
Any miss is FAIL/PARTIAL, not a rewritten pass.

---

## 11. Phase P8 — held-out generalization and promotion

- **Dependencies:** P7 PASS.
- **Exit:** E4/G5 promotion decision.

Evaluate one selected P7 checkpoint on 256 new disjoint contexts and at least 64 blinded pairs with two
independent raters, randomized order, ties, and agreement reporting.

**Acceptance**

- context-clustered, multiplicity-adjusted upper 95% collapse bound `<10%` in both raw modes, including
  full-length contexts;
- treatment-control coherence preference `>=30` points and paired-bootstrap lower bound `>0`;
- gibberish noninferiority passes;
- wide causal NLL/top-1 retains P7 bars;
- absent-ARREST serving/checkpoint parity remains unchanged.

Write `research/CHIRON_ARREST_E4_GENERALIZATION_YYYY_MM_DD.md`. A passing artifact is labeled a
**generation checkpoint**. Promotion requires a separate owner-reviewed
`docs/decisions/YYYY-MM-DD-chiron-arrest-promotion.md`; it does not silently replace the perplexity
flagship.

---

## 12. Cross-phase verification and commit boundaries

### Verification order after each code phase

1. CPU-only selector/self-test;
2. focused GPU selector when scheduled;
3. affected CHIRON/checkpoint tests;
4. glades install and trainer rebuild when public library code changed;
5. trainer smoke/parity;
6. final diff review, `test_runner infer/run`, and `gitverify`.

Use `build_project`/`test_project` or repository scripts rather than ad hoc binary paths. Use managed
background jobs for builds, collectors, and experiments.

### Logical commits when explicitly requested

1. `research: freeze CHIRON ARREST manifests and analysis contract` — P0 data/docs/scripts;
2. `chiron-generate: add ARREST repetition detector` — P1 source/tests;
3. `chiron-generate: expose observed row-only generation` — P2.1 source/tests;
4. `trainer: add CHAB codec and rollout collector` — P2.2/P2.3 source/tests;
5. `research: record ARREST causal transfer gate` — P3 report only;
6. `chiron: add ARREST hazard-loss kernels` — P4 source/tests, only after P3 PASS;
7. `trainer: integrate default-off ARREST replay` — P5 source/smoke;
8. `research: record ARREST pilot/recovery verdict` — P6–P8 reports.

Never commit CHAB buffers, checkpoints, decoded corpus text, raw model outputs, or logs.

---

## 13. Definition of done

Implementation is complete only when:

- M0, G0a, G0b, G1, and all G2 subgates pass;
- coefficient-zero is bit-identical and the tree's existing tests remain green;
- E0/E1, E2, and the three-seed E3/G4 gate pass without post-hoc tuning;
- E4/G5 passes on new contexts and blind judgments;
- final reports contain exact commits, commands, hashes, GPU-hours, seed-level outcomes, confidence
  intervals, and failed/partial bars;
- production promotion is separately approved.

Until then ARREST remains default-off research infrastructure, not a shipped training recipe.
