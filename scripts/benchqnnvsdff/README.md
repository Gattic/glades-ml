# QNN (`TYPE_QNN`) vs. DFNN (`TYPE_DFF`) Isolated Benchmark Protocol

This harness runs a controlled, repeatable head-to-head comparison between `glades::NNetwork::TYPE_QNN` (Quaternion Neural Network) and `glades::NNetwork::TYPE_DFF` (traditional real-valued Dense Feed-Forward Neural Network) across the XOR gate and two Iris classification regimes.

For architectural trade-offs and guidance on when to pick `TYPE_QNN` over `TYPE_DFF`, see [`docs/qnn-vs-dff.md`](../../docs/qnn-vs-dff.md).

## Files

- `scripts/benchqnnvsdff.sh` — Entry-point driver that builds `glades` and the standalone worker binary, spawns one fresh OS process per `(task, model)`, writes `results.tsv`, and renders Markdown summary tables (`report.md`).
- `scripts/benchqnnvsdff/bench_qnn_vs_dff.cpp` — C++98-compatible isolated worker executable implementing the datasets, model matrix, training/evaluation loops, and timing/memory probes.
- `scripts/benchqnnvsdff/CMakeLists.txt` — Minimal CMake project linking against the in-tree `build/` `glades` target (`find_package(glades REQUIRED)`), automatically matching `GLADES_ENABLE_CUDA` from `build/CMakeCache.txt`.

## Quickstart

Run the full isolated suite (27 fresh worker processes: 9 XOR + 9 Iris 80/20 + 9 Iris 40/60):

```sh
bash scripts/benchqnnvsdff.sh
```

Useful options:

```sh
# Print tables to stdout without saving under artifacts/
bash scripts/benchqnnvsdff.sh --no-artifacts

# Run a subset of tasks or models
bash scripts/benchqnnvsdff.sh --tasks xor --no-artifacts
bash scripts/benchqnnvsdff.sh --tasks iris80,iris40 --models qnn_1_6_3_real_sgdm,dff_4_15_3_sgdm

# Skip CMake rebuilds when build/ and build/benchqnnvsdff/ are already up to date
bash scripts/benchqnnvsdff.sh --skip-build --no-artifacts

# List all supported (task, model_id, label) rows
bash scripts/benchqnnvsdff.sh --list
```

By default, artifacts are written to `artifacts/bench_qnn_vs_dff_<YYYYMMDD-HHMMSS>/` (`results.tsv` and `report.md`), which is gitignored. Temporary model checkpoints created solely to measure `weights.bin` byte size live inside an ephemeral `mktemp -d` working directory that is removed on exit; the repository's `database/` tree is never touched.

## Isolation & Noise Control

1. **Process-per-configuration isolation**: Each `(task, model)` pair executes in its own fresh child process so `getrusage(RUSAGE_SELF).ru_maxrss` and glibc heap high-water marks cannot leak across models or tasks.
2. **Logger suppression**: Every `NNetwork` attaches a quiet `shmea::GLogger(LOG_FATAL)` with `setPrintToConsole(false)`, ensuring wall-clock timings and heap deltas measure model compute and parameter state rather than console I/O or log buffer growth.
3. **Shared preprocessing**: Both `TYPE_QNN` and `TYPE_DFF` consume the exact same `glades::QuaternionInput` (which inherits from `glades::NumberInput`) per split seed, guaranteeing identical stratified train/test splits and train-only Z-score statistics.

## Datasets & Splits

| Task ID | Dataset | Split & Preprocessing | Runs per Model |
|---|---|---|---|
| `xor` | `unit-tests/datasets/xorgate.csv` | 4 samples, full-batch (`batch=4`), no standardization, `PACK_GROUPS_OF_4` (`(x, y, 0, 0)`) | 20 training seeds (`100..119`) |
| `iris80` | `unit-tests/datasets/iris.data` | Stratified 80/20 (`120 train / 30 test`), train-only `GMath::ZSCORE`, `batch=16`, `PACK_GROUPS_OF_4` (`(SL, SW, PL, PW)`) | 5 split seeds × 5 training seeds = **25 runs** (`{42, 1337, 2024, 7777, 9001}`) |
| `iris40` | `unit-tests/datasets/iris.data` | Stratified 40/60 low-data (`60 train / 90 test`), train-only `GMath::ZSCORE`, `batch=16`, `PACK_GROUPS_OF_4` | 5 split seeds × 5 training seeds = **25 runs** (`{42, 1337, 2024, 7777, 9001}`) |

## Model Matrix & Fairness Controls

Because `TYPE_DFF` implements `SGD_MOMENTUM` (and `ATLAS`) rather than `ADAMW`, the primary head-to-head compares `TYPE_QNN` and `TYPE_DFF` under **identical `SGD_MOMENTUM` hyperparameters, activations (`TANH` hidden), seeds, and splits**, matching each QNN against DFNN baselines of equal real hidden width ($4 H_q$), equal active parameter count, and equal nominal hidden node count ($H_q$). Native `QNN (AdamW)` and a slower-LR `DFF (lr=0.03)` reference are included alongside.

### XOR (`TANH -> SIGMOID`, `batch=4`)

- **Standard width (800 epochs)**:
  - `qnn_1_4_1_sgdm` — QNN `1->4->1` (49 active / 52 stored), `SGD+M lr=0.2, m=0.9`
  - `dff_2_16_1_sgdm` — DFF `2->16->1` [width-matched] (65 params), `SGD+M lr=0.2, m=0.9`
  - `dff_2_12_1_sgdm` — DFF `2->12->1` [param-matched] (49 params), `SGD+M lr=0.2, m=0.9`
  - `qnn_1_4_1_adamw` — QNN `1->4->1` (49/52), `AdamW lr=0.05`
- **Minimal capacity (1500 epochs)**:
  - `qnn_1_1_1_sgdm` — QNN `1->1->1` (13 active / 16 stored), `SGD+M lr=0.2, m=0.9`
  - `dff_2_4_1_sgdm` — DFF `2->4->1` [width-matched] (17 params), `SGD+M lr=0.2, m=0.9`
  - `dff_2_3_1_sgdm` — DFF `2->3->1` [param-matched] (13 params), `SGD+M lr=0.2, m=0.9`
  - `dff_2_2_1_sgdm` — DFF `2->2->1` [minimal 2-node] (9 params), `SGD+M lr=0.2, m=0.9`
  - `qnn_1_1_1_adamw` — QNN `1->1->1` (13/16), `AdamW lr=0.05`

### Iris (`TANH -> LINEAR` Softmax CE, `batch=16`, 300 epochs, `wd2=1e-4`)

Shared across both `iris80` and `iris40`:

- `qnn_1_6_3_real_sgdm` — QNN `1->6->3` `READOUT_REAL_PART` (123 active / 132 stored), `SGD+M lr=0.1, m=0.9`
- `qnn_1_6_1_split_sgdm` — QNN `1->6->1` `READOUT_SPLIT_COMPONENTS` (75 active / 76 stored), `SGD+M lr=0.1, m=0.9`
- `dff_4_24_3_sgdm` — DFF `4->24->3` [width-matched] (195 params), `SGD+M lr=0.1, m=0.9`
- `dff_4_15_3_sgdm` — DFF `4->15->3` [param-matched to Real] (123 params), `SGD+M lr=0.1, m=0.9`
- `dff_4_9_3_sgdm` — DFF `4->9->3` [param-matched to Split] (75 params), `SGD+M lr=0.1, m=0.9`
- `dff_4_6_3_sgdm` — DFF `4->6->3` [node-matched] (51 params), `SGD+M lr=0.1, m=0.9`
- `qnn_1_6_3_real_adamw` — QNN `1->6->3` `READOUT_REAL_PART` (123/132), `AdamW lr=0.02`
- `qnn_1_6_1_split_adamw` — QNN `1->6->1` `READOUT_SPLIT_COMPONENTS` (75/76), `AdamW lr=0.02`
- `dff_4_15_3_sgdm_lr003` — DFF `4->15->3` (123 params), `SGD+M lr=0.03, m=0.9`

## Metric Definitions

- **Params (Act / Stored)**: Active scalar parameters contributing to the readout vs. total stored floats in the weight/bias tensors. For `TYPE_QNN`, unused imaginary/padding bias lanes on the output layer are stored in quaternion structs (`3 * Q_out` under `READOUT_REAL_PART`; `4 * Q_out - D_out` under `READOUT_SPLIT_COMPONENTS`) but never read.
- **Accuracy & Convergence**:
  - Exact sample-level Top-1 accuracy from per-row evaluation (`net.test` on single-row views).
  - `Canon 42 (%)`: Mean held-out Top-1 accuracy across the 5 training seeds on the canonical split (`splitSeed=42`).
  - XOR `Ep->BCE<=0.17`: First epoch with training BCE $\le 0.17$ (since $\frac{1}{4}\ln 2 = 0.1733$, this guarantees 4/4 classification across both architectures).
  - Iris `Ep->95% / 98%`: First epoch reaching 95% / 98% training Top-1 accuracy (recovered exactly from Glades' one-vs-rest accuracy via $\text{Top1} = 100 - \frac{C}{2}(100 - \text{OvR})$).
- **Loss & Calibration**: Exact per-sample binary cross-entropy (XOR) or multiclass negative log-likelihood (Iris), plus mean squared probability error (Brier score, lower is better).
- **Per-Class F1 & Macro-F1**: Computed from canonical-label confusion matrices (`Class 0`/`Class 1` on XOR; `Iris-setosa`/`Iris-versicolor`/`Iris-virginica` resolved by label name from `QuaternionInput::getClassLabels()` on Iris).
- **Wall-Clock Time**:
  - `Train (ms)`: `CLOCK_MONOTONIC` elapsed milliseconds per `net.train()` call (mean ± std across runs).
  - `Predict us/split (us/smp)`: Full-split `net.test()` wall-clock latency averaged over `--predict-repeats-xor` (default 100) or `--predict-repeats-iris` (default 50) repeats.
- **Memory**:
  - `Peak Heap Delta (KiB)`: Peak live glibc `mallinfo2().uordblks` minus the post-dataset baseline during `NNetwork` lifecycle (`init` + `train` + `test`).
  - `Core / w.bin (B)`: Analytical byte footprint of `TensorQNNState` / `TensorDFFState` core parameter/optimizer/gradient/scratch buffers (excluding `TensorDFFState::helm` and `aster` buffers, which are captured in `Peak Heap Delta`) alongside actual serialized `weights.bin` file size.
  - `Peak RSS (KiB)`: Isolated worker `getrusage(RUSAGE_SELF).ru_maxrss`.

## Output Schema (`results.tsv`)

Tab-separated, one row per `(task, model_id)`:

`task`, `model_id`, `label`, `active_params`, `stored_params`, `budget_epochs`, `runs`, `converged_runs`, `train_acc_mean`, `train_acc_std`, `test_acc_mean`, `test_acc_std`, `canon_test_acc`, `ep_metric1_mean`, `ep_metric1_med`, `ep_metric2_mean`, `train_loss_mean`, `test_loss_mean`, `test_loss_std`, `test_brier_mean`, `f1_c0`, `f1_c1`, `f1_c2`, `macro_f1_mean`, `macro_f1_std`, `train_ms_mean`, `train_ms_std`, `predict_split_us`, `predict_sample_us`, `peak_heap_delta_kib`, `core_state_bytes`, `weights_bin_bytes`, `peak_rss_kib`.
