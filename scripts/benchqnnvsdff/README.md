# QNN (`TYPE_QNN`) vs. DFNN (`TYPE_DFF`) Isolated Benchmark Protocol

This harness runs a controlled, repeatable head-to-head comparison between `glades::NNetwork::TYPE_QNN` (standard and exact $\mathrm{SO}(3)$-equivariant Quaternion Neural Networks) and `glades::NNetwork::TYPE_DFF` (traditional real-valued Dense Feed-Forward Neural Network) across the baseline suite (`xor`, `iris80`, `iris40`), the Tier 1 geometric, parity, financial, and tabular suite (`--tasks tier1`), and the Tier 2 $\mathrm{SO}(3)$-symmetric molecular & orbital suite (`--tasks tier2`).

For architectural trade-offs and guidance on when to pick `TYPE_QNN` (or `LAYER_SO3_EQUIVARIANT` + `ACT_POLAR_TANH`) over `TYPE_DFF`, see [`docs/qnn-vs-dff.md`](../../docs/qnn-vs-dff.md).

## Files

- `scripts/benchqnnvsdff.sh` — Entry-point driver that builds `glades` and the standalone worker binary (forwarding `CMAKE_CUDA_COMPILER` / `CMAKE_CUDA_ARCHITECTURES` from `build/CMakeCache.txt`), spawns one fresh OS process per `(task, model)`, writes `results.tsv`, and renders Markdown summary tables (`report.md`). With `--json` it instead writes one JSON object per run into `results.json`.
- `scripts/benchqnnvsdff/bench_qnn_vs_dff.cpp` — C++98-compatible isolated worker executable implementing all datasets, model matrices, training/evaluation loops, equivariance/invariance probes, and timing/memory probes.
- `scripts/benchqnnvsdff/CMakeLists.txt` — Minimal CMake project linking against the in-tree `build/` `glades` target (`find_package(glades REQUIRED)`), automatically matching `GLADES_ENABLE_CUDA` from `build/CMakeCache.txt`.

## Quickstart

```sh
# Baseline suite (27 fresh worker processes: xor, iris80, iris40)
bash scripts/benchqnnvsdff.sh

# Full Tier 1 suite (97 fresh worker processes: SO(3), parity, Dow OHLC, Wine)
bash scripts/benchqnnvsdff.sh --tasks tier1 --no-artifacts

# Full Tier 2 suite (56 fresh worker processes: H2O PES+Force, Chirality, Slater-Koster, Kepler)
bash scripts/benchqnnvsdff.sh --tasks tier2 --no-artifacts

# Everything (180 fresh worker processes)
bash scripts/benchqnnvsdff.sh --tasks all --no-artifacts

# Targeted task or alias subsets
bash scripts/benchqnnvsdff.sh --tasks so3_rot_curve,so3_rot_ood,so3_comp --no-artifacts
bash scripts/benchqnnvsdff.sh --tasks mol_pes,mol_chiral,orbital_slater,orbital_kepler --no-artifacts
bash scripts/benchqnnvsdff.sh --tasks parity3,parity4,dow_ohlc,wine --no-artifacts
bash scripts/benchqnnvsdff.sh --tasks iris80,iris40 --models qnn_1_6_3_real_sgdm,dff_4_15_3_sgdm

# Skip CMake rebuilds when build/ and build/benchqnnvsdff/ are already up to date
bash scripts/benchqnnvsdff.sh --skip-build --tasks tier2 --no-artifacts

# List all supported (task, model_id, label) rows
bash scripts/benchqnnvsdff.sh --list

# Emit machine-readable JSON (one object per run) for the Tier 2 suite
bash scripts/benchqnnvsdff.sh --tasks tier2 --json --out-dir /tmp/qnn_tier2
```

The Tier 2 tables in [`docs/qnn-vs-dff.md`](../../docs/qnn-vs-dff.md) are
verified against that JSON by
`bash unit-tests/checkdocstier2.sh` (committed snapshot
`unit-tests/data/qnn_tier2_benchmark.json`; `--generate` re-runs the suite,
`--json PATH` checks a specific file).

Supported task aliases:
- `tier1` $\to$ `so3_rot_curve,so3_rot_ood,so3_comp_id,so3_comp_ood,xorparity3,parity3_odd,parity4,dow_ohlc,wine`
- `tier2` $\to$ `mol_pes_id,mol_pes_ood,mol_chiral_id,mol_chiral_ood,orbital_slater_id,orbital_slater_ood,orbital_kepler_id,orbital_kepler_ood`
- `all` $\to$ `xor,iris80,iris40` + `tier1` + `tier2`
- `so3_comp` $\to$ `so3_comp_id,so3_comp_ood`
- `parity3` $\to$ `xorparity3,parity3_odd`
- `mol_pes` $\to$ `mol_pes_id,mol_pes_ood`
- `mol_chiral` $\to$ `mol_chiral_id,mol_chiral_ood`
- `orbital_slater` $\to$ `orbital_slater_id,orbital_slater_ood`
- `orbital_kepler` $\to$ `orbital_kepler_id,orbital_kepler_ood`

By default, artifacts are written to `artifacts/bench_qnn_vs_dff_<YYYYMMDD-HHMMSS>/` (`results.tsv` and `report.md`), which is gitignored. Temporary model checkpoints created solely to measure `weights.bin` byte size live inside an ephemeral `mktemp -d` working directory that is removed on exit; the repository's `database/` tree is never touched.

## Isolation & Noise Control

1. **Process-per-configuration isolation**: Each `(task, model)` pair executes in its own fresh child process so `getrusage(RUSAGE_SELF).ru_maxrss` and glibc heap high-water marks cannot leak across models or tasks.
2. **Logger suppression**: Every `NNetwork` attaches a quiet `shmea::GLogger(LOG_FATAL)` with `setPrintToConsole(false)`, ensuring wall-clock timings and heap deltas measure model compute and parameter state rather than console I/O or log buffer growth.
3. **Shared preprocessing**: Both `TYPE_QNN` and `TYPE_DFF` consume the exact same `glades::QuaternionInput` (which inherits from `glades::NumberInput`) per split/seed, guaranteeing identical splits, standardization, and synthetic samples.

## Datasets & Splits

| Task ID | Dataset | Split & Preprocessing | Runs per Model |
|---|---|---|---|
| `xor` | `unit-tests/datasets/xorgate.csv` | 4 samples, full-batch (`batch=4`), `PACK_GROUPS_OF_4` | 20 seeds (`100..119`) |
| `iris80` | `unit-tests/datasets/iris.data` | Stratified 80/20 (`120 / 30`), train-only Z-score, `batch=16` | 5 splits × 5 seeds = **25 runs** |
| `iris40` | `unit-tests/datasets/iris.data` | Stratified 40/60 (`60 / 90`), train-only Z-score, `batch=16` | 5 splits × 5 seeds = **25 runs** |
| `so3_rot_curve` | Synthetic `loadSO3Rotation` | $v' = q_{\text{rot}} v q_{\text{rot}}^*$, $N_{\text{train}} \in \{32,64,128,256,512\}$, $N_{\text{test}}=256$, `batch=16`, 400 ep; input `(0,v), q`, target `(0,v')` (`READOUT_SPLIT_COMPONENTS`) | 5 seeds (`42, 1337, 2024, 7777, 9001`) per $N$ |
| `so3_rot_ood` | Synthetic `loadSO3Rotation(..., ood=true)` | Train $v_z > 0.25, \theta \in [0,\pi/2]$ ($N=128$); Test $v_z < -0.25, \theta \in [\pi/2,\pi]$ ($N=256$), 400 ep | 5 seeds |
| `so3_comp_id` | Synthetic `loadSO3Composition(..., ood=false)` | $q_1 \otimes q_2$, $N_{\text{train}}=128, N_{\text{test}}=256$, `batch=16`, 400 ep | 5 seeds |
| `so3_comp_ood` | Synthetic `loadSO3Composition(..., ood=true)` | Train $q_1.k, q_2.k > 0.15$; Test $q_1.k, q_2.k < -0.15$ ($N=128 / 256$), 400 ep | 5 seeds |
| `xorparity3` | `datasets/xorparity3.csv` | 8 rows, exact-one-bit (`sum==1`; row `1,1,1 -> 0`), `batch=8` | 20 seeds (`100..119`) |
| `parity3_odd` | Synthetic `loadBinaryParity(3)` | 8 rows, true 3-bit odd parity, `batch=8` | 20 seeds (`100..119`) |
| `parity4` | Synthetic `loadBinaryParity(4)` | 16 rows, true 4-bit odd parity, `batch=16` | 20 seeds (`100..119`) |
| `dow_ohlc` | `datasets/dow_jones_index_{train,test}.csv` | 360 Q1 train $\to$ 390 Q2 test; train-only Z-scored `(open, high, low, close)` packed into 1 quat predicting standardized `next_weeks_close`, `batch=32`, 200 ep | 5 seeds |
| `wine` | `datasets/wine.data` | 178 rows, 13 features, 3 classes; stratified 80/20, train-only Z-score, `batch=16`; compares `PACK_GROUPS_OF_4` vs `ONE_QUAT_PER_FEATURE`, `wd2=1e-4` vs `1e-2`, and 60-epoch early-stop budget | 5 splits × 5 seeds = **25 runs** |
| `mol_pes_id` / `mol_pes_ood` | Synthetic `loadMolecularPES` | Triatomic $\text{H}_2\text{O}$ Morse stretch + harmonic cosine bend + exact analytic Oxygen force $\mathbf{F}_O = -\nabla_{\mathbf{x}_O} E$; input `(0,r1), (0,r2)` (2 quats), target `(E, Fx, Fy, Fz)` (`READOUT_SPLIT_COMPONENTS`); $N_{\text{train}}=192, N_{\text{test}}=256$, `batch=16`, 400 ep. OOD: train rotated within `[0, 30°]`, test within `[60°, 180°]` | 5 seeds |
| `mol_chiral_id` / `mol_chiral_ood` | Synthetic `loadMolecularChirality` | Balanced 3-class chiral center $\text{C-L}_1\text{L}_2\text{L}_3$ (`Planar`, `R-Enantiomer`, `S-Enantiomer`); input `(0.25,r1), (0.50,r2), (0.75,r3)` (3 quats), target 3 one-hot classes (`READOUT_REAL_PART`, strictly $\mathrm{SO}(3)$-invariant & parity-breaking); $N_{\text{train}}=180, N_{\text{test}}=240$, `batch=16`, 300 ep. OOD: train `[0, 35°]`, test `[60°, 180°]` | 5 seeds |
| `orbital_slater_id` / `orbital_slater_ood` | Synthetic `loadSlaterKosterOrbital` | Tight-binding Slater–Koster $p$–$p$ molecular orbital overlap $S_{pp}(\mathbf{r},\mathbf{p}_1,\mathbf{p}_2)$; input `(0,r), (0,p1), (0,p2)` (3 quats), target 1 scalar (`READOUT_REAL_PART`, strictly $\mathrm{SO}(3)$-invariant); $N_{\text{train}}=192, N_{\text{test}}=256$, `batch=16`, 400 ep. OOD: train upper polar cone $\hat{r}_z > 0.65$, test lower hemisphere $\hat{r}_z < -0.35$ | 5 seeds |
| `orbital_kepler_id` / `orbital_kepler_ood` | Synthetic `loadKeplerOrbit` | 3D Keplerian two-body orbits ($\mu=1$); input `(0,r), (0,v)` (2 quats), target `(\varepsilon, e_x, e_y, e_z)` (invariant specific energy + equivariant Laplace–Runge–Lenz eccentricity vector, `READOUT_SPLIT_COMPONENTS`); $N_{\text{train}}=192, N_{\text{test}}=256$, `batch=16`, 400 ep. OOD: train near-equatorial prograde ($i \in [0^\circ, 25^\circ]$), test inclined/polar/retrograde ($i \in [65^\circ, 180^\circ]$) | 5 seeds |

## Metric Definitions

- **Params (Act / Stored)**: Active scalar parameters contributing to the forward pass vs. total stored floats in weight/bias tensors. For `LAYER_SO3_EQUIVARIANT`, each bias stores 3 active scalars `(bL, bR, bS)` in a 4-float `Quaternion` (`k` unused).
- **Classification (`xor`, `xorparity3`, `parity3_odd`, `parity4`, `iris80`, `iris40`, `wine`, `mol_chiral_*`)**: Exact per-row Top-1 accuracy, BCE/CE loss, Brier score, canonical per-class F1, Macro-F1, and epoch-to-threshold (`BCE <= 0.17` for truth tables; `95% / 98%` train Top-1 for tabular/chiral). `mol_chiral_*` additionally reports `Invar Err (max)` over 32 random 3D rotations of 3-quaternion inputs.
- **$\mathrm{SO}(3)$ Regression (`so3_rot_*`, `so3_comp_*`)**:
  - `3D RMSE` / `4D RMSE`: Per-component RMSE over the 3D pure-vector lanes `(i, j, k)` (rotations) or all 4 quaternion lanes `(r, i, j, k)` (composition).
  - `Ang Err (deg)` / `Geod Err (deg)`: Mean 3D vector angle error $\arccos(\hat{\mathbf{v}}\cdot\mathbf{v}' / \|\hat{\mathbf{v}}\|\|\mathbf{v}'\|)$ or $\mathrm{SO}(3)$ geodesic error $2\arccos(|\langle \hat{q}/\|\hat{q}\|, q\rangle|)$.
  - `Test R2`: Explained variance over the evaluated vector/quaternion lanes.
  - `Equiv Err (max)`: Worst-case $\|f(u q_0 u^*, u q_1 u^*) - u f(q_0, q_1) u^*\|_2$ over 32 deterministic random `(q0, q1, u)` triples (`glades::rng::Engine`).
- **Tier 2 Molecular & Orbital Regression (`mol_pes_*`, `orbital_kepler_*`, `orbital_slater_*`)**:
  - Joint scalar + 3D vector tasks (`mol_pes_*`, `orbital_kepler_*`): `4D RMSE` over `(scalar, vx, vy, vz)`, `Energy RMSE` on lane 0, `Force Ang (deg)` / `LRL Ang (deg)` on lanes `(1, 2, 3)`, `Test R2`, and `Equiv Err (max)`.
  - Scalar invariant task (`orbital_slater_*`): `Test RMSE`, `Test MAE`, `Test R2`, and `Invar Err (max)` ($\max |f(u \mathbf{q} u^*) - f(\mathbf{q})|$ over 32 random rotations of 3-quaternion inputs).
- **Dow Jones OHLC (`dow_ohlc`)**: Test RMSE in $\sigma$ and USD, test MAE ($\sigma$), test $R^2$, and directional accuracy (% of Q2 weeks where `sign(pred - close) == sign(actual_next_close - close)`).
- **Wall-Clock & Memory**: `Train (ms)` (`CLOCK_MONOTONIC`), `Predict us/split (us/smp)` averaged over `--predict-repeats-*`, `Peak Heap Delta (KiB)` (`mallinfo2().uordblks`), analytical `Core` state bytes vs. serialized `weights.bin` bytes, and isolated `Peak RSS (KiB)` (`getrusage`).

## Output Schema (`results.json`, `--json`)

`--json` emits JSON Lines: one object per `(task, model)` run with the same
fields as the TSV schema below, using full `%.10g` precision. Field names:
`task`, `model_id`, `label`, `active_params`, `stored_params`, `budget_epochs`,
`runs`, `converged_runs`, `train_acc_mean`, `train_acc_std`, `test_acc_mean`,
`test_acc_std`, `canon_test_acc`, `ep_metric1_mean`, `ep_metric1_med`,
`ep_metric2_mean`, `train_loss_mean`, `test_loss_mean`, `test_loss_std`,
`test_brier_mean`, `f1_c0`, `f1_c1`, `f1_c2`, `macro_f1_mean`, `macro_f1_std`,
`train_ms_mean`, `train_ms_std`, `predict_split_us`, `predict_sample_us`,
`peak_heap_delta_kib`, `core_state_bytes`, `weights_bin_bytes`, `peak_rss_kib`,
`reg_rmse_mean`, `reg_rmse_std`, `reg_aux_mean`, `reg_aux_std`, `reg_r2_mean`,
`equiv_err_mean`.

## Output Schema (`results.tsv`)

Tab-separated, one row per `(task, model_id)`:

`task`, `model_id`, `label`, `active_params`, `stored_params`, `budget_epochs`, `runs`, `converged_runs`, `train_acc_mean`, `train_acc_std`, `test_acc_mean`, `test_acc_std`, `canon_test_acc`, `ep_metric1_mean`, `ep_metric1_med`, `ep_metric2_mean`, `train_loss_mean`, `test_loss_mean`, `test_loss_std`, `test_brier_mean`, `f1_c0`, `f1_c1`, `f1_c2`, `macro_f1_mean`, `macro_f1_std`, `train_ms_mean`, `train_ms_std`, `predict_split_us`, `predict_sample_us`, `peak_heap_delta_kib`, `core_state_bytes`, `weights_bin_bytes`, `peak_rss_kib`, `reg_rmse_mean`, `reg_rmse_std`, `reg_aux_mean`, `reg_aux_std`, `reg_r2_mean`, `equiv_err_mean`.
