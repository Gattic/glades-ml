# Choosing Between `TYPE_QNN` and `TYPE_DFF`

`glades-ml` provides two dense feed-forward architectures through `glades::NNetwork`:

- **`TYPE_DFF`** (`Backend/Machine Learning/Networks/sgd_dff.cpp`) — Traditional real-valued Dense Feed-Forward network supporting CPU/CUDA execution, `SGD_MOMENTUM`, and `ATLAS`.
- **`TYPE_QNN`** (`Backend/Machine Learning/Networks/sgd_qnn.cpp`) — Quaternion-valued Neural Network representing nodes, weights, biases, and gradients in $\mathbb{H}$ with `SGD_MOMENTUM` / `ADAMW` optimizers (currently CPU-only) and two composable architectural options in `TrainingConfig::qnn`:
  - **`layerMode`**:
    - `QuaternionConfig::LAYER_STANDARD` (default) — Dense Hamilton-product transitions $z_m = b_m + \sum_n W_{m,n} \otimes a_n$.
    - `QuaternionConfig::LAYER_SO3_EQUIVARIANT` — Exact $\mathrm{SO}(3)$ conjugation-equivariant bilinear Hamilton transitions $z_m = (b_{S,m}, \mathbf{0}) + L_m \otimes R_m$ where $L_m, R_m$ are equivariant scalar/vector projections of input quaternions. Because conjugation $q \mapsto u q u^*$ ($u \in \mathbb{S}^3$) is a Hamilton automorphism, every layer preserves exact 3D rotational equivariance to float32 machine precision ($\sim 10^{-7}$).
  - **`activationMode`**:
    - `QuaternionConfig::ACT_SPLIT` (default) — Component-wise activation $(f(r), f(i), f(j), f(k))$ (best for axis-aligned binary logic and tabular classification).
    - `QuaternionConfig::ACT_POLAR_TANH` — Singularity-free $C^\infty$ norm-equivariant activation $\frac{\tanh(\|z\|)}{\|z\|} z$ (with 4th-order Taylor expansion near $\|z\| = 0$), preserving exact $\mathrm{SO}(3)$ equivariance when paired with `LAYER_SO3_EQUIVARIANT`.

The isolated benchmark harness and protocol live in [`scripts/benchqnnvsdff/README.md`](../scripts/benchqnnvsdff/README.md) (`scripts/benchqnnvsdff.sh`).

## Decision Matrix

| Criterion | Prefer `TYPE_QNN` | Prefer `TYPE_DFF` |
|---|---|---|
| **3D geometry, molecular physics & orbital mechanics ($\mathrm{SO}(3)$)** | **`LAYER_SO3_EQUIVARIANT` + `ACT_POLAR_TANH`**: 3D rotations, quaternion composition, molecular PES + forces, chiral enantiomers, Slater–Koster overlaps, and Keplerian orbits (15–176× lower RMSE, `100%` OOD chirality vs. `47–54%`, exact $\mathrm{SO}(3)$ equivariance/invariance to $\sim 10^{-7}$–$10^{-6}$) | Not competitive without massive data augmentation |
| **Coupled 4-channel continuous signals** | `LAYER_STANDARD` (`ACT_SPLIT` / `ACT_POLAR_TANH`) with `PACK_GROUPS_OF_4` (e.g., weekly OHLC 4-tuples: matches 2×-wider DFF at half the parameters) | When CUDA / large-width BLAS scaling is primary |
| **Compactness per real width** | Need a wide real hidden representation with fewer parameters and smaller checkpoints (up to 4× fewer weights per real layer width) | Optimizing strictly for minimum FLOPs per parameter |
| **Tiny-capacity non-linear logic** | `LAYER_STANDARD` + `ACT_SPLIT` on tight parameter budgets (1-hidden-node XOR, 3-hidden-node 4-bit parity: 20/20 vs. DFF 9/20–15/20) | Standard-width tabular MLPs |
| **Heterogeneous tabular classification** | Competitive with `ONE_QUAT_PER_FEATURE` + stronger L2 (`wd2=1e-2`) (matches DFF on Wine with lower test CE); `PACK_GROUPS_OF_4` trades slight CE for 2× compression | Default choice for unregularized small-sample tabular accuracy and calibration (e.g., low-data Iris 40/60) |
| **Optimizer & hardware support** | Want built-in `ADAMW` on CPU dense layers | Need CUDA acceleration, `ATLAS`, or maximum throughput at large layer widths |

## Benchmark Evidence (`scripts/benchqnnvsdff.sh`)

Controlled head-to-head comparisons execute each `(task, model)` in an isolated OS process with suppressed logging, deterministic seeds, and shared `glades::QuaternionInput` preprocessing across the baseline suite (`xor`, `iris80`, `iris40`), the Tier 1 suite (`--tasks tier1`), and the Tier 2 $\mathrm{SO}(3)$-symmetric molecular & orbital suite (`--tasks tier2`).

### 1. Tier 1: $\mathrm{SO}(3)$ 3D Rotation & Quaternion Composition (Decisive `TYPE_QNN` Win)

Entries in both tables report `Test RMSE (Mean Angular Error, R²)` across 5 seeds (`400` epochs, $N_{\text{test}}=256$).

#### $\mathrm{SO}(3)$ 3D Vector Rotation — Sample-Efficiency Curve ($v' = q_{\text{rot}} v q_{\text{rot}}^*$)

| $N_{\text{train}}$ | `QNN-SO3-Equiv` (`2->8->1`, 123 act, `SGD+M`) | `QNN-SO3-Equiv` (`2->8->1`, 123 act, `AdamW`) | `QNN-Std Polar` (`2->8->1`, 132 act, `AdamW`) | `QNN-Std Split` (`2->8->1`, 132 act, `AdamW`) | `DFF` `[width=]` (`8->32->4`, 420p, `SGD+M`) | `DFF` `[param=]` (`8->10->4`, 134p, `SGD+M`) |
|---:|---|---|---|---|---|---|
| **32** | `0.0114` (`0.54°`, $R^2=0.9996$) | **`0.0105`** (`0.59°`, $R^2=0.9996$) | `0.4927` (`44.3°`, $R^2=0.305$) | `0.7948` (`73.7°`, $R^2=-0.782$) | `0.7611` (`74.3°`, $R^2=-0.634$) | `0.9174` (`77.5°`, $R^2=-1.378$) |
| **64** | **`0.0081`** (`0.39°`, $R^2=0.9998$) | `0.0110` (`0.73°`, $R^2=0.9997$) | `0.3401` (`28.0°`, $R^2=0.675$) | `0.6475` (`62.8°`, $R^2=-0.167$) | `0.6709` (`60.4°`, $R^2=-0.257$) | `0.8210` (`70.3°`, $R^2=-0.889$) |
| **128** | **`0.0054`** (`0.29°`, $R^2=0.9999$) | `0.0073` (`0.51°`, $R^2=0.9998$) | `0.2833` (`22.0°`, $R^2=0.776$) | `0.4839` (`46.1°`, $R^2=0.347$) | `0.5358` (`47.0°`, $R^2=0.198$) | `0.6265` (`63.9°`, $R^2=-0.096$) |
| **256** | **`0.0038`** (`0.20°`, $R^2=1.0000$) | `0.0110` (`0.80°`, $R^2=0.9995$) | `0.2950` (`23.0°`, $R^2=0.759$) | `0.4900` (`49.8°`, $R^2=0.335$) | `0.6007` (`53.4°`, $R^2=-0.005$) | `0.6076` (`66.2°`, $R^2=-0.019$) |
| **512** | **`0.0034`** (`0.21°`, $R^2=1.0000$) | `0.0096` (`0.66°`, $R^2=0.9997$) | `0.2992` (`24.0°`, $R^2=0.751$) | `0.5482` (`66.8°`, $R^2=0.173$) | `0.5782` (`61.5°`, $R^2=0.082$) | `0.5672` (`62.5°`, $R^2=0.115$) |

#### $\mathrm{SO}(3)$ OOD 3D Rotation & Quaternion Composition ($N_{\text{train}}=128, N_{\text{test}}=256$, 5 seeds)

| Task | `QNN-SO3-Equiv` (`AdamW`) | `QNN-SO3-Equiv` (`SGD+M`) | `QNN-Std Polar` (`AdamW`) | `QNN-Std Split` (`SGD+M`) | `DFF` `[width=]` (`SGD+M`) | `DFF` `[param=]` (`SGD+M`) |
|---|---|---|---|---|---|---|
| **3D Rot OOD** ($v_z < -0.25$, $\theta \in [\pi/2,\pi]$) | **`0.0929`** (`6.16°`, $R^2=0.9747$) | `0.1137` (`7.75°`, $R^2=0.9631$) | `0.7622` (`85.8°`, $R^2=-0.614$) | `0.8894` (`91.0°`, $R^2=-1.210$) | `0.9578` (`91.9°`, $R^2=-1.561$) | `1.0928` (`96.4°`, $R^2=-2.350$) |
| **Quat Comp ID** ($q_1 \otimes q_2$) | **`0.0044`** (`0.75°`, $R^2=0.9999$) | `0.0050` (`0.77°`, $R^2=0.9999$) | `0.2661` (`52.5°`, $R^2=0.704$) | `0.3799` (`82.7°`, $R^2=0.381$) | `0.2442` (`45.5°`, $R^2=0.745$) | `0.3462` (`74.1°`, $R^2=0.497$) |
| **Quat Comp OOD** (opposite frame) | `0.0048` (`0.93°`, $R^2=0.9998$) | **`0.0045`** (`0.74°`, $R^2=0.9999$) | `0.6752` (`115.1°`, $R^2=-1.226$) | `0.6591` (`131.3°`, $R^2=-1.102$) | `0.7602` (`126.3°`, $R^2=-1.795$) | `0.7941` (`138.0°`, $R^2=-2.044$) |

#### Top Findings (Tier 1 $\mathrm{SO}(3)$-Equivariant vs. `TYPE_DFF`)

1. **Extreme sample efficiency (>16× data advantage, up to 167× lower RMSE):** With only **$N=32$** training samples and `123` active parameters, `LAYER_SO3_EQUIVARIANT` + `ACT_POLAR_TANH` achieves **`0.54°`** angular error ($R^2=0.9996$), outperforming `TYPE_DFF` trained on **16× more data ($N=512$)** by **50× in RMSE** (`0.0114` vs. `0.5672`). At $N=512$, `QNN-SO3-Equiv (SGD+M)` reaches **`0.21°`** (`RMSE 0.0034`, **167× lower** than param-matched DFF).
2. **Exact float32 $\mathrm{SO}(3)$ equivariance and zero OOD frame degradation:** Max conjugation-equivariance error stays at **`4.48e-7`–`7.90e-7`** (vs. `0.32–4.68` for `QNN-Std` and `DFF`). On opposite-frame OOD quaternion composition, `QNN-SO3-Equiv` (`93` active params) exhibits **zero OOD degradation** (`0.74°` geodesic error, $R^2=0.9999$, **176× lower RMSE** than param-matched DFF); on combined unseen-hemisphere + large-angle 3D rotation OOD, it achieves **`6.16°`** (`10.3–11.8×` lower RMSE, $R^2=+0.9747$) where every non-equivariant baseline collapses to negative $R^2$ ($\ge 85°$ error).
3. **`ACT_POLAR_TANH` helps even without equivariant layers — but is domain-specific:** On standard `LAYER_STANDARD` QNNs, switching from `ACT_SPLIT` to `ACT_POLAR_TANH` cuts 3D-rotation RMSE nearly in half (`0.283` vs. `0.484` at $N=128$). Conversely, on discrete binary parity (e.g., 4-bit parity), `ACT_SPLIT` wins decisively (`20/20` at minimal `37` active params vs. DFF `15/20`, while `ACT_POLAR_TANH` scores `0/20`).
4. **Complementary Tier 1 non-geometric takeaways:** On coupled 4-channel Dow Jones OHLC, `QNN 1->4->1 Split (AdamW)` matches 2×-wider DFF (`$2.08` RMSE, $R^2=0.9962$) at half the parameters (`49` vs. `97`) while beating param-matched DFF by `1.76–1.91×`; on heterogeneous tabular Wine (`13` features), pairing `ONE_QUAT_PER_FEATURE` with stronger L2 (`wd2=1e-2`) eliminates QNN's calibration gap and yields the lowest test CE overall (`0.0565`, `97.78%` accuracy).

### 2. Tier 2: $\mathrm{SO}(3)$-Symmetric Molecular & Orbital Physics (Decisive `TYPE_QNN` Win)

Four physically grounded 3D benchmarks (`--tasks tier2`, 5 seeds each, ID and OOD orientation splits) test **joint scalar+vector equivariance**, **chiral ($\mathrm{SO}(3)$-invariant, $\mathrm{O}(3)$/parity-breaking) classification**, and **multi-vector $\mathrm{SO}(3)$ invariance**. All four tasks use exact parameter-matched and width-matched `TYPE_DFF` baselines (`2->8->6->1` = `325` act / `340` stored vs. `DFF [param=]` `339p` and `[width=]` `1,180p`; `3->8->3` = `225` act vs. `227p`/`515p`; `3->8->6->1` = `357` act vs. `359p`/`1,233p`).

#### Joint Scalar + 3D Vector & Scalar Invariant Regression ($N_{\text{train}}=192, N_{\text{test}}=256$, 400 epochs, 5 seeds)

| Task | `QNN-SO3-Equiv` (`AdamW`) | `QNN-SO3-Equiv` (`SGD+M`) | `QNN-Std Polar` (`AdamW`) | `QNN-Std Split` (`AdamW`) | `DFF` `[width=]` (`SGD+M`) | `DFF` `[param=]` (`SGD+M`) |
|---|---|---|---|---|---|---|
| **H₂O PES + Force (ID)** `(E, F_O)` | `0.0280` (`E=0.0257`, `2.55°`, $R^2=0.9975$) | **`0.0254`** (`E=0.0172`, `2.27°`, $R^2=0.9981$) | `0.1998` (`16.87°`, $R^2=0.8807$) | `0.6849` (`65.12°`, $R^2=-0.392$) | `0.6570` (`65.03°`, $R^2=-0.279$) | `0.7216` (`69.81°`, $R^2=-0.545$) |
| **H₂O PES + Force (OOD)** `[0,30°] -> [60,180°]` | **`0.0232`** (`E=0.0163`, `1.95°`, $R^2=0.9984$) | `0.0289` (`E=0.0240`, `2.47°`, $R^2=0.9976$) | `1.0232` (`73.95°`, $R^2=-2.009$) | `1.3575` (`82.20°`, $R^2=-4.333$) | `1.7495` (`85.80°`, $R^2=-7.788$) | `1.6208` (`85.98°`, $R^2=-6.546$) |
| **Slater–Koster $p$–$p$ (ID)** `S_pp` | **`0.0286`** (`MAE=0.0204`, $R^2=0.9938$) | `0.0288` (`MAE=0.0196`, $R^2=0.9942$) | `0.4224` (`MAE=0.3252`, $R^2=-0.211$) | `0.5117` (`MAE=0.4007`, $R^2=-0.767$) | `0.6224` (`MAE=0.4956`, $R^2=-1.601$) | `0.6821` (`MAE=0.5383`, $R^2=-2.135$) |
| **Slater–Koster $p$–$p$ (OOD)** `z>0.65 -> z<-0.35` | `0.0286` (`MAE=0.0189`, $R^2=0.9939$) | **`0.0276`** (`MAE=0.0196`, $R^2=0.9944$) | `0.5867` (`MAE=0.4701`, $R^2=-1.467$) | `0.5803` (`MAE=0.4661`, $R^2=-1.418$) | `0.7655` (`MAE=0.6279`, $R^2=-3.228$) | `0.9160` (`MAE=0.7525`, $R^2=-5.304$) |
| **Kepler Energy + LRL (ID)** `(\varepsilon, e)` | **`0.0072`** (`E=0.0079`, `1.62°`, $R^2=0.9983$) | `0.0200` (`E=0.0281`, `3.15°`, $R^2=0.9870$) | `0.0831` (`23.16°`, $R^2=0.7820$) | `0.1741` (`54.24°`, $R^2=0.0448$) | `0.1047` (`30.82°`, $R^2=0.6530$) | `0.1432` (`46.50°`, $R^2=0.3522$) |
| **Kepler Energy + LRL (OOD)** `i in [0,25°] -> [65,180°]` | **`0.0065`** (`E=0.0079`, `1.31°`, $R^2=0.9985$) | `0.0193` (`E=0.0266`, `3.14°`, $R^2=0.9882$) | `0.4501` (`64.68°`, $R^2=-5.376$) | `0.3351` (`82.74°`, $R^2=-2.489$) | `0.7098` (`88.03°`, $R^2=-14.529$) | `0.6681` (`88.98°`, $R^2=-12.755$) |

#### Molecular Chirality 3-Class (`Planar` / `R` / `S` Enantiomer, $N_{\text{train}}=180, N_{\text{test}}=240$, 300 epochs, 5 seeds)

| Split | `QNN-SO3-Equiv` (`AdamW`, 225p) | `QNN-SO3-Equiv` (`SGD+M`, 225p) | `QNN-Std Polar` (`AdamW`, 227p) | `QNN-Std Split` (`SGD+M`, 227p) | `DFF` `[width=]` (`515p`, `SGD+M`) | `DFF` `[param=]` (`227p`, `SGD+M`) |
|---|---|---|---|---|---|---|
| **Chiral ID** (full $\mathrm{SO}(3)$) | **`100.00%`** (`CE=0.0000`, `Inv=6.61e-7`) | **`100.00%`** (`CE=0.0003`, `Inv=5.59e-7`) | `97.83%` (`CE=0.0659`) | `80.67%` (`CE=0.6873`) | `83.33%` (`CE=0.5152`) | `75.58%` (`CE=0.9772`) |
| **Chiral OOD** `[0,35°] -> [60,180°]` | **`100.00%`** (`CE=0.0000`, `Inv=6.60e-7`) | **`100.00%`** (`CE=0.0002`, `Inv=7.07e-7`) | `51.08%` (`CE=4.7292`) | `53.58%` (`CE=4.4119`) | `46.92%` (`CE=4.8027`) | `47.83%` (`CE=4.6401`) |

#### Top Findings (Tier 2 Molecular & Orbital Physics)

1. **Zero OOD orientation degradation across all four physical domains (28–109× lower OOD RMSE):**
   - **H₂O PES + Force (`mol_pes_ood`)**: `QNN-SO3-Equiv (AdamW)` achieves **`0.0232` 4D RMSE (`E=0.0163`, `1.95°` force angle, $R^2=0.9984$, `Equiv=1.92e-6`)**, beating param-matched DFF by **69.9×** and 3.6×-larger width-matched DFF by **75.4×**.
   - **Keplerian Energy + LRL (`orbital_kepler_ood`)**: Trained only on near-equatorial prograde orbits ($i \in [0^\circ, 25^\circ]$) and tested on polar/retrograde orbits ($i \in [65^\circ, 180^\circ]$), `QNN-SO3-Equiv (AdamW)` reaches **`0.0065` 4D RMSE (`1.31°` LRL angle, $R^2=0.9985$, `Equiv=5.97e-7`)** — **102.8× lower RMSE** than param-matched DFF (`0.6681`, `88.98°`, $R^2=-12.75$) and **109.2× lower** than width-matched DFF (`0.7098`, `88.03°`, $R^2=-14.53$).
   - **Quantum Slater–Koster $p$–$p$ Overlap (`orbital_slater_ood`)**: Using `READOUT_REAL_PART` for exact $\mathrm{SO}(3)$ scalar invariance (`Invar=2.92e-7–5.72e-7`), `QNN-SO3-Equiv (SGD+M)` reaches **`0.0276` RMSE (`MAE 0.0196`, $R^2=0.9944$)** — **33.2× lower RMSE** than param-matched DFF (`0.9160`, $R^2=-5.30$) and **27.7× lower** than width-matched DFF (`0.7655`, $R^2=-3.23$).
2. **Exact $\mathrm{SO}(3)$ invariance that preserves chirality ($\mathrm{O}(3)$ parity breaking):** Distance-and-angle-only descriptors cannot distinguish mirror-image enantiomers (`R` vs. `S`), whereas Hamilton products natively compute oriented cross products ($\mathbf{r}_1 \times \mathbf{r}_2$) and signed triple products ($\mathbf{r}_1 \cdot (\mathbf{r}_2 \times \mathbf{r}_3)$) in the real lane. `QNN-SO3-Equiv (3->8->3, READOUT_REAL_PART)` reaches **`100.00% ± 0.00%` accuracy (`CE 0.0000`, `Invar 6.60e-7`) in <3 epochs** on both ID and OOD chirality, while every non-equivariant baseline drops to **`46.9–53.6%`** OOD (`F1 R/S = 35–41%`).
3. **Faster and smaller than width-matched `TYPE_DFF` in 2-hidden-layer architectures:** On `mol_pes` and `orbital_kepler`, `QNN-SO3-Equiv (2->8->6->1)` (`325` active params, `1,508 B` `weights.bin`, `16.3 KiB` peak heap) trains **~31% faster** (`261–270 ms` vs. `383–407 ms`) and predicts **~1.8× faster** (`1.68 µs/smp` vs. `3.01–3.12 µs/smp`) than width-matched `DFF 8->32->24->4` (`1,180` params, `4,844 B` `weights.bin`, `35.1 KiB` peak heap).

### 3. Binary Parity & Tiny-Capacity Non-Linear Expressivity (20 Seeds Each)

- **XOR (`xorgate.csv`)**: At minimal capacity (`13` active params), `QNN 1->1->1 (SGD+M)` solves **20/20 seeds (100%, BCE `0.00075`)** vs. param-matched `DFF 2->3->1` (`19/20`, BCE `0.01856`) and minimal `DFF 2->2->1` (`9/20`, `72.5%`). At standard width, `QNN 1->4->1` converges 14% faster than width-matched `DFF 2->16->1` (`43.0` vs. `49.9` ep) with 25% fewer params.
- **In-repo `datasets/xorparity3.csv` (exact-1-bit `sum==1`) & synthetic 3-bit odd parity**: `QNN 1->4->1 Split (SGD+M)` solves 20/20 faster and to lower BCE than both width-matched and param-matched DFF (`71.4` vs. `76.3`/`83.0` ep on 3-bit odd parity; minimal `QNN 1->2->1` beats `DFF 3->5->1` `68.0` vs. `92.0` ep).
- **Synthetic 4-bit odd parity (16 rows)**: `QNN 1->6->1 Split (SGD+M)` (`73` params) solves 20/20 in `178.0` ep (`35.35 ms`) vs. width-matched `DFF 4->24->1` (`145` params, `239.2` ep, `40.89 ms`) and param-matched `DFF 4->12->1` (`216.8` ep). At minimal capacity (`37` active params), **`QNN 1->3->1 Split (SGD+M)` solves 20/20 (100%, BCE `0.00058`)** while param-matched `DFF 4->6->1` solves only **15/20 (`98.44%`, BCE `0.03059`)**.
- **Activation trade-off**: `ACT_POLAR_TANH` underperforms `ACT_SPLIT` on binary parity (`0/20` on 4-bit parity) because radial norm coupling resists sharp axis-aligned decision cuts. Use `ACT_SPLIT` for binary/tabular logic and `ACT_POLAR_TANH` for geometric/continuous quaternion signals.

### 4. Coupled 4-Channel Financial Regression (`dow_jones_index`, Q1 $\to$ Q2 OHLC, 5 Seeds)

- Packing `(open, high, low, close)` into 1 quaternion (`QNN 1->4->1`, 49 active params):
  - `QNN 1->4->1 Split (AdamW)` achieves **Q2 RMSE `$2.08 ± $0.07` (`0.0656 sigma`, $R^2=0.9962$, MAE `0.0449`)** — matching 2×-larger width-matched `DFF 4->16->1 (SGD+M lr=0.01)` (`97` params, `$2.08 ± $0.05`, $R^2=0.9962$) while beating param-matched `DFF 4->8->1` at both `lr=0.05` (`$3.98 ± $1.28`, `1.91×` worse) and `lr=0.01` (`$3.66 ± $0.43`, `1.76×` worse).
  - Under matched `SGD+M (lr=0.05)`, `QNN 1->4->1 Polar` achieves **`$2.16 ± $0.14` ($R^2=0.9959$)** whereas `DFF 4->16->1` becomes unstable (`$8.39 ± $12.42`).
  - Note: For `LAYER_SO3_EQUIVARIANT` on unbounded Z-scored inputs, use `AdamW` or a smaller `SGD+M` step (`lr=0.01`) since bilinear products scale quadratically with input norm.

### 5. Tabular Classification: Iris & Wine (5 Stratified Splits × 5 Seeds = 25 Runs)

- **Wine (`datasets/wine.data`, 13 features, 3 classes, 80/20)**:
  - Confirms the regularization & packing rules: raising L2 weight decay from `1e-4` to `1e-2` cuts QNN test CE by **21–35%** and Brier by **20–29%**.
  - `QNN 13->6->3 (ONE_QUAT_PER_FEATURE, SGD+M, wd=1e-2)` matches DFF's **97.78% ± 1.11% test accuracy** and achieves the **lowest test CE overall (`0.0565 ± 0.0172`)**, edging `DFF 13->12->3 (wd=1e-2)` (`0.0586`) and `DFF 13->24->3` (`0.0668`).
  - `QNN 4->6->3 (PACK_GROUPS_OF_4, wd=1e-2)` delivers `97.67%` accuracy (`CE 0.0650`) with **53% fewer parameters** (`195` vs. `411`).
- **Iris (`iris80` & `iris40`)**:
  - Against width-matched `DFF 4->24->3` (`195` params), `TYPE_QNN` reaches 98% training accuracy 2.7–4.4× faster, cuts active params by 37–62%, shrinks `weights.bin` by 27–52%, and uses 9–21% less peak live heap.
  - On low-data `iris40` (`60` train / `90` test) with weak L2 (`1e-4`, 300 epochs), `TYPE_DFF` (`96.40–96.98%`, CE `0.088–0.138`) still outperforms unregularized `TYPE_QNN` (`95.29–96.13%`, CE `0.170–0.194`) because `TYPE_QNN` fits small training sets ~5× deeper.

### 6. Compute Cost & Scaling Caveat

- Each Hamilton product performs 16 multiplications and 12 additions across 4 shared weight floats (~3–4× more FLOPs per parameter than a real multiply-add). Against parameter-matched `TYPE_DFF` on non-geometric tasks, `TYPE_QNN` is ~7–30% slower on CPU, whereas against width-matched `TYPE_DFF` (especially in 2-hidden-layer Tier 2 architectures), `TYPE_QNN` is ~30–45% faster on CPU with 3.5× fewer parameters. `TYPE_DFF` retains CUDA acceleration and real BLAS throughput at large layer widths.

## Practical Configuration Guidance for `TYPE_QNN`

- **Layer & activation mode (`TrainingConfig::qnn.{layerMode, activationMode}`)**:
  - For **joint scalar + 3D vector $\mathrm{SO}(3)$-equivariant regression** (rotations, quaternion composition, molecular PES + force, Keplerian energy + LRL vector): set `LAYER_SO3_EQUIVARIANT` + `ACT_POLAR_TANH` + `PACK_GROUPS_OF_4` + `READOUT_SPLIT_COMPONENTS`, encoding 3D vectors as `(s, vx, vy, vz)` (with invariant scalar `s`, e.g., `0` or species/channel tag) so the output scalar `r` lane is strictly $\mathrm{SO}(3)$-invariant and the `(i, j, k)` vector lanes are strictly $\mathrm{SO}(3)$-equivariant.
  - For **strictly $\mathrm{SO}(3)$-invariant scalar regression or chiral classification** (Slater–Koster orbital overlap, `R`/`S` enantiomers): set `LAYER_SO3_EQUIVARIANT` + `ACT_POLAR_TANH` + `PACK_GROUPS_OF_4` + `READOUT_REAL_PART`, reading only the conjugation-invariant scalar lane of each output quaternion.
  - For **general non-equivariant classification / binary logic**: keep `LAYER_STANDARD` + `ACT_SPLIT`.
  - For **coupled 4-channel continuous signals** (e.g., OHLC): try `LAYER_STANDARD` with either `ACT_SPLIT` (`AdamW`) or `ACT_POLAR_TANH` (`SGD+M`).
- **Input packing (`TrainingConfig::qnn.inputPacking`)**: Use `PACK_GROUPS_OF_4` when inputs naturally form 3- or 4-tuples or when compact checkpoints are preferred; use `ONE_QUAT_PER_FEATURE` for heterogeneous scalar tabular features (e.g., Wine's 13 measurements) when prioritizing test CE/accuracy over compression.
- **Output readout (`TrainingConfig::qnn.outputReadout`)**: Use `READOUT_REAL_PART` (default) for one quaternion per scalar target/logit (also gives exact $\mathrm{SO}(3)$ invariance under `LAYER_SO3_EQUIVARIANT`), or `READOUT_SPLIT_COMPONENTS` for 4-lane quaternion outputs (`(scalar, vx, vy, vz)`) or minimal output checkpoint size.
- **Regularization**: On small tabular datasets, pair `TYPE_QNN` with stronger L2 weight decay (`wd2 = 1e-2`) or early stopping.
- **Reproducing**: Run `bash scripts/benchqnnvsdff.sh --tasks all --no-artifacts`, `--tasks tier1 --no-artifacts`, or `--tasks tier2 --no-artifacts` (full protocol in [`scripts/benchqnnvsdff/README.md`](../scripts/benchqnnvsdff/README.md)).
