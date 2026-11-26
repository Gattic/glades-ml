# Choosing Between `TYPE_QNN` and `TYPE_DFF`

`glades-ml` provides two dense feed-forward architectures through `glades::NNetwork`:

- **`TYPE_DFF`** (`Backend/Machine Learning/Networks/sgd_dff.cpp`) — Traditional real-valued Dense Feed-Forward network supporting CPU/CUDA execution, `SGD_MOMENTUM`, and `ATLAS`.
- **`TYPE_QNN`** (`Backend/Machine Learning/Networks/sgd_qnn.cpp`) — Quaternion-valued Neural Network representing nodes, weights, biases, and gradients in $\mathbb{H}$ with Hamilton-product dense transitions, component-wise split activations, and `SGD_MOMENTUM` / `ADAMW` optimizers (currently CPU-only).

The isolated benchmark harness and protocol live in [`scripts/benchqnnvsdff/README.md`](../scripts/benchqnnvsdff/README.md) (`scripts/benchqnnvsdff.sh`).

## Decision Matrix

| Criterion | Prefer `TYPE_QNN` | Prefer `TYPE_DFF` |
|---|---|---|
| **Feature geometry** | Natively coupled 3D/4D channels (quaternions, 3D pose/IMU/rotations, RGB+D or color triples, polarimetric or 4-channel signals) | Independent or heterogeneous scalar tabular features |
| **Compactness per real width** | Need a wide real hidden representation with fewer parameters and smaller checkpoints (up to 4× fewer weights per real layer width) | Optimizing strictly for minimum FLOPs per parameter |
| **Tiny-capacity non-linear tasks** | Very small networks where Hamilton coupling generates orthogonal hyperplane pairs from a single hidden node (e.g., 1-hidden-node XOR) | Standard-width tabular MLPs |
| **Small-sample tabular generalization** | Only with early stopping, shorter epoch budgets, or stronger L2 regularization | Default choice for tabular accuracy and calibrated probabilities |
| **Optimizer & hardware support** | Want built-in `ADAMW` on CPU dense layers | Need CUDA acceleration, `ATLAS`, or maximum throughput at large layer widths |

## Benchmark Evidence (`scripts/benchqnnvsdff.sh`)

Controlled head-to-head comparisons match `TYPE_QNN` and `TYPE_DFF` under identical `SGD_MOMENTUM` hyperparameters, `TANH` hidden activations, deterministic seeds, and shared `glades::QuaternionInput` preprocessing across XOR (20 seeds) and Iris 80/20 & 40/60 splits (5 stratified splits × 5 seeds = 25 runs each), with each `(task, model)` executed in an isolated child process.

### Where `TYPE_QNN` Improves on `TYPE_DFF`

1. **Tiny-capacity non-linear expressivity (XOR):**
   - `QNN 1->4->1` (49 active / 52 stored params, `SGD+M`) reaches 4/4 convergence (`BCE <= 0.17`) **14% faster** than width-matched `DFF 2->16->1` (`43.0` vs. `49.9` epochs) with **28% lower final BCE** (`0.00066` vs. `0.00092`), **25% fewer active parameters**, and **12% faster training** (`6.24` vs. `7.10 ms`).
   - At minimal capacity (`13` active params), `QNN 1->1->1 (SGD+M)` solves **20/20 seeds (100%)** (`BCE 0.00075`), outperforming parameter-matched `DFF 2->3->1` (`19/20`, `97.5%`, `BCE 0.01856`) and minimal `DFF 2->2->1` (`9/20`, `72.5%`).
2. **Epoch convergence, checkpoint compression, and peak live heap vs. width-matched `TYPE_DFF` (Iris):**
   - At equal 24-unit real hidden width (`QNN 1->6->*` vs. `DFF 4->24->3`, `SGD+M`), `TYPE_QNN` reaches 98% training accuracy **2.7–4.4× faster** (`9.8` vs. `43.2` epochs on 80/20; `27.5` vs. `78.1` epochs on 40/60).
   - `TYPE_QNN` reduces active parameters by **37–62%** (`123` or `75` vs. `195`), shrinks `weights.bin` by **27–52%** (`644` / `420 B` vs. `880 B`), runs **7–12% faster** in training and prediction, and uses **9–21% less peak live heap** (`16.20–18.55 KiB` vs. `19.20–20.73 KiB`, because `TensorDFFState` allocates `helm`/`aster` buffers unconditionally).
   - On Iris 80/20, `QNN 1->6->1 (READOUT_SPLIT_COMPONENTS, AdamW)` matches parameter-matched `DFF 4->15->3` in held-out accuracy (`98.53%`) with `75` active parameters and the lowest test Brier score (`0.0250`).

### Where `TYPE_DFF` Remains Better

1. **Tabular generalization and calibration on small datasets:**
   - On low-data Iris 40/60 (`60` train / `90` test, 300 epochs), every `TYPE_DFF` baseline (`96.40–96.98%` test accuracy, `0.088–0.138` test CE, `0.049–0.061` Brier) outperforms every `TYPE_QNN` variant (`95.29–96.13%` test accuracy, `0.170–0.194` test CE, `0.068–0.075` Brier).
   - On Iris 80/20 under identical `SGD+M`, parameter-matched `DFF 4->15->3` edges `QNN 1->6->3` (`98.53%` vs. `98.13%` test accuracy; `0.0714` vs. `0.0760` test CE), and slower-LR `DFF 4->15->3 (lr=0.03)` leads overall (`98.67%`, CE `0.0597`).
   - **Cause:** `TYPE_QNN` fits small training sets ~5× deeper (`0.003` vs. `0.015` train CE on Iris), so fixed-epoch schedules without early stopping or stronger L2 weight decay overfit borderline tabular points when features lack intrinsic quaternion symmetry.
2. **Compute cost per parameter and large-width scaling:**
   - Each Hamilton product executes 16 multiplications and 12 additions across 4 shared weight floats (~3–4× more FLOPs per parameter than a real multiply-add). Against parameter-matched `TYPE_DFF`, `TYPE_QNN` is **7–15% slower** to train and **4–16% slower** to predict at Iris/XOR scale, and real BLAS/CUDA GEMMs scale faster at larger layer widths.

## Practical Configuration Guidance for `TYPE_QNN`

- **Input packing (`TrainingConfig::qnn.inputPacking`):** Default to `QuaternionConfig::PACK_GROUPS_OF_4` (`QUAT_PACK_GROUPS_OF_4`) when inputs naturally group into 3- or 4-tuples (e.g., `(SL, SW, PL, PW)` or `(x, y, z, w)`). Use `QuaternionConfig::ONE_QUAT_PER_FEATURE` (`QUAT_ONE_PER_FEATURE`) when unrelated scalar features should remain separated at layer 0.
- **Output readout (`TrainingConfig::qnn.outputReadout`):** Use `QuaternionConfig::READOUT_REAL_PART` (`QUAT_READOUT_REAL_PART`, default) for one quaternion per output logit, or `QuaternionConfig::READOUT_SPLIT_COMPONENTS` (`QUAT_READOUT_SPLIT_COMPONENTS`) when minimizing output parameters and checkpoint size (e.g., packing 3 Iris classes into 1 output quaternion).
- **Regularization:** Because `TYPE_QNN` drives training loss down faster than `TYPE_DFF`, pair it with early stopping, shorter epoch budgets, or stronger L2 weight decay (`NNInfo::setWeightDecay2`) on small tabular datasets.
- **Reproducing the comparison:** Run `bash scripts/benchqnnvsdff.sh --no-artifacts` (full protocol in [`scripts/benchqnnvsdff/README.md`](../scripts/benchqnnvsdff/README.md)).
