# Cellular Sheaf Attention — Gate-0 Detailed Implementation Plan

**Status:** operational plan (Ralph-loop iter 10, 2026-05-14). Bridges design (iters 1-9) to validation.
**Date:** 2026-05-14.
**Branch:** vesta5.
**Purpose:** Concrete instructions to run Gate-0 probes A-N. Specifies file locations in `glades-ml`, C++ pseudocode for each measurement, expected outputs with tolerances, and decision criteria. After running this plan (≤8.5 GPU-hours total), the CSA program has empirical evidence on 11 conjectures.

---

## 0. Overview

Gate-0 has **three stages** per the iter-9 critical reflection priority ordering:

| Stage | Cost | Probes | Conjectures tested |
|---|---|---|---|
| 1: Cheap-decisive | 1 GPU-hour | B', I, ρ_k measurement, H, P=2 vs P=1 | 1', 4, 6, 7, 9 |
| 2: Confirmatory | 2.5 GPU-hours | B, C, D, E, F, G, J | 1, 2, BF16-stability, Lipschitz, SRA structural, PSA training |
| 3: Capability | 5 GPU-hours | K, L, M, N | SLR + CSR specific |

**Total**: 8.5 GPU-hours on RTX 4080 SUPER + flagship checkpoint `chiron_1B_T16384.step30000`.

Decision after Stage 1: ≥4 of 5 pass = continue; ≤3 = halt and reassess.

---

## 1. Repository layout for Gate-0

All Gate-0 code lives in:

```
glades-ml/
├── unit-tests/
│   └── Backend/
│       └── Machine Learning/
│           ├── sfa-gate0-probes.cpp                  ← NEW: Stage 1 probes B', I, ρ_k, H, P-comparison
│           ├── sfa-gate0-probes.h                    ← NEW: shared declarations
│           ├── sfa-gate0-stage2.cpp                  ← NEW: Stage 2 probes B, C, D, E, F, G, J
│           ├── sfa-gate0-stage2.h                    ← NEW
│           ├── sfa-gate0-stage3.cpp                  ← NEW: Stage 3 probes K, L, M, N
│           └── sfa-gate0-stage3.h                    ← NEW
├── Backend/
│   └── Machine Learning/
│       └── Networks/
│           ├── transformer_sfa_ops.h                 ← NEW: SFA primitives (CPU prototype for Gate-0)
│           ├── transformer_sfa_ops.cpp               ← NEW: implementation
│           ├── transformer_sfa_chebyshev.h           ← NEW: Chebyshev recurrence + Jacobi precond
│           └── transformer_sfa_chebyshev.cpp         ← NEW
└── research/
    └── CSA_GATE0_IMPLEMENTATION_PLAN.md             ← this document
```

Plus modifications to:
- `unit-tests/test.sh` — add `sfa-gate0` as a test name.
- `unit-tests/main.cpp` — register `SFAGate0Probes()`.
- `unit-tests/Backend/Machine Learning/CMakeLists.txt` — add new source files.

---

## 2. SFA primitives needed for Gate-0 (CPU prototype)

Gate-0 does NOT require the GPU-optimised SFA kernels (those are Phase 2 work). It only needs a **CPU prototype** of the sheaf Laplacian matvec + Chebyshev recurrence sufficient to swap one layer of the flagship model.

### 2.1 Header sketch — `transformer_sfa_ops.h`

```cpp
#ifndef _TRANSFORMER_SFA_OPS_H
#define _TRANSFORMER_SFA_OPS_H

#include <vector>
#include <Backend/Machine Learning/Networks/transformer_types.h>

namespace glades {

// Per-layer SFA parameters (CPU prototype storage).
struct SFAParams {
    int T;        // sequence length
    int d_s;      // stalk dimension
    int r;        // stalk-frame rank
    int W;        // sliding-window half-width
    int n_sinks;  // number of sink vertices

    // Stalk frames: U[T][d_s][r], stored row-major as flat array.
    std::vector<float> U;

    // Edge modulators: Sigma[|E|][r], diagonal entries per edge.
    std::vector<float> Sigma;

    // Edge list: (src, tgt) pairs.
    std::vector<int> edge_src;
    std::vector<int> edge_tgt;

    // Stalk injection/readout: P_q, P_v, P_o all [d_s][d_h].
    std::vector<float> P_q, P_v, P_o;

    // Tikhonov regulariser (softplus-parameterised).
    float lambda;

    // Value injection scale.
    float gamma;
};

// Build edge list for causal-sliding-window-plus-sinks graph.
// E = { (i,j) : i <= j AND (|i-j| <= W OR i in S_sink) }.
void sfa_build_edge_set(int T, int W, int n_sinks,
                        std::vector<int>& edge_src,
                        std::vector<int>& edge_tgt);

// Apply L_F to a section: out = L_F * s.
// L_F encoded by U + Sigma + edge_list per §4.3 of PARADIGM_SHIFT_250_DESIGN.md.
// s and out are [T][d_s] row-major.
void sfa_laplacian_matvec(const SFAParams& params,
                          const float* s,
                          float* out);

// Run M-step Chebyshev on (L_F + lambda*I) for source b.
// Returns approximate (L_F + lambda*I)^{-1} * b in `result`.
// mu_max: pre-estimated upper bound on L_F's spectrum (via power iteration).
void sfa_chebyshev_solve(const SFAParams& params,
                         const float* b,
                         float mu_max,
                         int M,            // Chebyshev degree
                         float* result);

// Source assembly: b_i = U_i U_i^T P_q W_Q x_i + gamma * P_v W_V x_i.
// Inputs: x [T][m], W_Q, W_V [m][d_h]. Outputs: b [T][d_s].
void sfa_source_assembly(const SFAParams& params,
                          const float* x, int T, int m,
                          const float* W_Q,
                          const float* W_V,
                          int d_h,
                          float* b);

// Full SFA forward: y = SFA-forward(x).
void sfa_forward(const SFAParams& params,
                  const float* x, int T, int m,
                  const float* W_Q, const float* W_V, int d_h,
                  float* y);

// Power iteration estimate of mu_max = ||L_F||_op.
float sfa_estimate_mu_max(const SFAParams& params, int n_power_iters);

// Diagonal-Jacobi preconditioner application.
void sfa_jacobi_precondition(const SFAParams& params, float* v_in_out);

}  // namespace glades

#endif
```

### 2.2 Implementation outline — `transformer_sfa_ops.cpp`

Key routines (~500 lines total in CPU prototype):

**sfa_build_edge_set** (~30 lines): generate edge list per eq. 2 of design doc. Causal: `i <= j`. Sliding window: `|i - j| <= W`. Sinks: `i in S_sink` connects to all `j >= i`. Default `S_sink = {1, 2, ..., n_sinks}`.

**sfa_laplacian_matvec** (~80 lines): compute out = L_F * s where L_F has block structure per §3.3 of design.
- For each edge (i, j) ∈ E: compute `R_{j ← i} s_i = U_j (Sigma(i,j) * (U_i^T s_i))` per eq. 16. Accumulate into out_j (off-diagonal) and into out_i (diagonal degree contribution).
- ~3 nested loops, vectorisable with SIMD.

**sfa_chebyshev_solve** (~100 lines): M-step Chebyshev iteration. Standard recurrence T_{n+1}(x) = 2x T_n(x) − T_{n−1}(x). Pre-scale L_F by 2/mu_max so eigenvalues are in [−1, 1].
- Apply Jacobi preconditioner first: D_F^{-1} L_F D_F^{-1/2}.
- Run M steps, accumulating Chebyshev coefficients.
- Final: result = Σ_n c_n T_n(L_F) b.

**sfa_estimate_mu_max** (~30 lines): power iteration, default n_power_iters = 2.

**sfa_jacobi_precondition** (~40 lines): apply D_F^{-1} per block — where D_F is the block-diagonal of L_F.

**sfa_source_assembly** (~60 lines): standard matrix products to compute b_i.

**sfa_forward** (~150 lines): orchestrates source assembly + Chebyshev solve + readout.

**Cost estimate**: CPU prototype runs ~10× slower than future GPU implementation, but T=1024 fits in seconds on CPU. Adequate for Gate-0 probes.

---

## 3. Stage 1 probes (1 GPU-hour total)

### 3.1 Probe B' — Φ-rich vs Φ-poor breakdown

**Test name**: `--sfa-gate0-probe-bprime`.

**File**: `unit-tests/Backend/Machine Learning/sfa-gate0-probes.cpp`.

**Goal**: Verify that SFA's NLL improvement over SCFA is **concentrated** on Φ-rich sequences (those containing anaphora, agreement chains, embedded discourse) rather than uniform across data.

**Procedure**:

```cpp
void ProbeBprime() {
    // 1. Load flagship checkpoint.
    NNetwork* model = loadFlagshipCheckpoint("research/runs/2026-05-14-production-T16384-50k/chiron_1B_T16384.step30000");

    // 2. Hot-swap layer 12 from SCFA to SFA (d_s=64, r=4, W=128, n_sinks=8, M=8, lambda=1e-2).
    SFAParams sfa_params = initSFAFromSCFA(model, layer_idx=12, d_s=64, r=4, W=128, n_sinks=8);
    swapLayerToSFA(model, 12, sfa_params);

    // 3. Load Φ-tagged val set. (Phi-rich = sequences with linguistic complexity tags;
    //    Phi-poor = simple factual statements.)
    std::vector<TokenSequence> phi_rich_val = loadPhiTaggedValSet("data/val_phi_rich.tok.bin");
    std::vector<TokenSequence> phi_poor_val = loadPhiTaggedValSet("data/val_phi_poor.tok.bin");

    // 4. Compute NLL on both subsets.
    float nll_rich_sfa = computeNLL(model, phi_rich_val);
    float nll_poor_sfa = computeNLL(model, phi_poor_val);

    // 5. Restore SCFA at layer 12, recompute.
    swapLayerToSCFA(model, 12);
    float nll_rich_scfa = computeNLL(model, phi_rich_val);
    float nll_poor_scfa = computeNLL(model, phi_poor_val);

    // 6. Compute ratio.
    float delta_rich = nll_rich_scfa - nll_rich_sfa;
    float delta_poor = nll_poor_scfa - nll_poor_sfa;
    float ratio = delta_rich / std::max(delta_poor, 1e-6f);

    printf("Probe B': delta_rich=%.4f, delta_poor=%.4f, ratio=%.2f\n",
           delta_rich, delta_poor, ratio);

    // Decision.
    if (ratio >= 4.0f) {
        printf("PASS: cocycle mechanism validated.\n");
    } else {
        printf("FAIL: ratio %.2f < 4.0; cocycle mechanism unsupported.\n", ratio);
    }
}
```

**Φ-tagging**: requires a one-time pre-processing of the val set. Heuristic taggers (based on Stanza or spaCy parses) suffice for Gate-0. Tags include:
- has_anaphora: presence of pronouns with 3+ token-distance to antecedent.
- has_agreement: subject-verb agreement with intervening clauses.
- has_embedded_discourse: nested quotation or relative clauses.
- Φ-rich: any of the above; Φ-poor: none.

**Cost**: 5 minutes (1 forward pass on 1024-sample val × 4 = ~30 sec each, including swap overhead).

**Pass criterion**: `ratio ≥ 4.0`.

**Fail action**: cocycle mechanism is suspect; the magnitude claim's *mechanism* (not just magnitude) is undermined. Re-evaluate program assumptions.

### 3.2 Probe I — 30% layer pruning

**Test name**: `--sfa-gate0-probe-i`.

**Goal**: Verify that pruning the 30% lowest-`birth_ℓ + death_ℓ` layers preserves NLL within 0.02 nat.

**Procedure**:

```cpp
void ProbeI() {
    // 1. Load flagship checkpoint with full SFA (all 24 layers SFA, d_s=64, r=4).
    //    NOTE: requires prior Phase 1 SFA implementation — for Gate-0,
    //    we approximate by computing PSA on SCFA flagship instead.
    NNetwork* model = loadFlagshipCheckpoint(/* path */);

    // 2. Compute PSA persistence diagram on the SCFA model.
    //    Per §3 of PARADIGM_SHIFT_252_DESIGN.md.
    PSAPersistenceDiagram pd0 = computePSAPersistenceDiagram(model, k=0);

    // 3. Compute per-layer birth + death counts.
    std::vector<int> birth_death(model->numLayers());
    for (int ell = 0; ell < model->numLayers(); ell++) {
        birth_death[ell] = pd0.birthCount(ell) + pd0.deathCount(ell);
    }

    // 4. Sort layers by birth + death; identify bottom 30%.
    std::vector<int> sorted_layers = argsort_ascending(birth_death);
    std::vector<int> to_prune(sorted_layers.begin(),
                              sorted_layers.begin() + model->numLayers() / 3);

    // 5. Create pruned model.
    NNetwork* pruned = pruneLayers(model, to_prune);

    // 6. Fine-tune pruned model for 500 steps to re-stabilise.
    fineTune(pruned, 500_steps, lr=1e-4);

    // 7. Compare NLL on val set.
    float nll_orig = computeNLL(model, val_set);
    float nll_pruned = computeNLL(pruned, val_set);
    float delta = nll_pruned - nll_orig;

    printf("Probe I: delta NLL = %.4f, layers pruned = %d/%d\n",
           delta, to_prune.size(), model->numLayers());

    if (delta <= 0.02f) {
        printf("PASS: pruning preserves NLL.\n");
    } else {
        printf("FAIL: delta %.4f > 0.02; pruning damages quality.\n", delta);
    }
}
```

**Cost**: 30 minutes (500 fine-tune steps on the 1B model with 30% fewer layers, ~70% of original step cost = ~30 sec each).

**Pass criterion**: `delta NLL ≤ 0.02 nat`.

**Fail action**: PSA's pruning mechanism doesn't work as advertised. PSA's value reduces to interpretability (the diagram is still meaningful) but no compute win from pruning.

### 3.3 Conjecture 9 measurement — ρ_k spectrum

**Test name**: `--sfa-gate0-rho-k`.

**Goal**: Measure `ρ_k = λ_1(C_k) / σ_max(C_k)` for k = 1, 2, 4, 8 on the flagship.

**Procedure**:

```cpp
void RhoKMeasurement() {
    // 1. Load flagship.
    NNetwork* model = loadFlagshipCheckpoint(/* path */);

    // 2. Extract per-layer restriction-map approximations from SCFA's basis B.
    //    For SCFA at layer ell, the "restriction map" R_{j←i}^{(ell)} is
    //    approximated by B_ell^T B_ell (the spectral projector).
    std::vector<RestrictionMaps> per_layer_R(model->numLayers());
    for (int ell = 0; ell < model->numLayers(); ell++) {
        per_layer_R[ell] = approximateRestrictionMapsFromSCFA(model, ell);
    }

    // 3. For k = 1, 2, 4, 8:
    //    Compute C_k via stochastic Lanczos.
    std::array<float, 4> rho_k;
    for (int k_idx = 0; k_idx < 4; k_idx++) {
        int k = std::pow(2, k_idx);
        rho_k[k_idx] = computeRhoK(per_layer_R, k, n_lanczos=32);
    }

    printf("ρ_k spectrum: ρ_1=%.3f, ρ_2=%.3f, ρ_4=%.3f, ρ_8=%.3f\n",
           rho_k[0], rho_k[1], rho_k[2], rho_k[3]);

    // Decision: ρ_4 > 0.5 (some multi-step reasoning structure present)
    if (rho_k[2] > 0.5f) {
        printf("PASS: multi-step reasoning structure detected.\n");
    } else {
        printf("FAIL: ρ_4 = %.3f < 0.5; reasoning structure absent or weak.\n", rho_k[2]);
    }
}
```

**Cost**: 5 minutes (Lanczos on 4 k values × ~30 sec each on RTX 4080 SUPER).

**Pass criterion**: `ρ_4 > 0.5`.

**Fail action**: CSR's reasoning-ceiling metric doesn't detect structure on the current flagship. This could mean either (a) the SCFA model itself doesn't have multi-step reasoning structure to detect, or (b) the ρ_k definition needs refinement. Pre-implementation, treat as **inconclusive** rather than fail.

### 3.4 Probe H — persistence diagram health

**Test name**: `--sfa-gate0-probe-h`.

**Goal**: Verify that the persistence diagram on the flagship has ≥30% of bars with length ≥ L/2 = 12.

**Procedure**:

```cpp
void ProbeH() {
    // 1. Load flagship.
    NNetwork* model = loadFlagshipCheckpoint(/* path */);

    // 2. Compute persistence diagram (as in Probe I, but report bar statistics).
    PSAPersistenceDiagram pd0 = computePSAPersistenceDiagram(model, k=0);

    // 3. Bar-length statistics.
    int total_bars = pd0.totalBars();
    int long_bars = pd0.barsWithLengthAtLeast(model->numLayers() / 2);
    float long_fraction = (float)long_bars / total_bars;

    printf("Probe H: total_bars=%d, long_bars (>=L/2)=%d, fraction=%.2f\n",
           total_bars, long_bars, long_fraction);

    if (long_fraction >= 0.30f) {
        printf("PASS: persistence diagram has rich long-bar structure.\n");
    } else {
        printf("FAIL: fraction %.2f < 0.30; few long bars — degenerate diagram.\n", long_fraction);
    }
}
```

**Cost**: 5 minutes (reuses Probe I's PSA computation).

**Pass criterion**: `long_fraction ≥ 0.30`.

**Fail action**: PSA's mechanism is questionable. The model's geometry doesn't show the expected hierarchy.

### 3.5 P=2 vs P=1 expressivity gain

**Test name**: `--sfa-gate0-multipole`.

**Goal**: Verify that multi-pole SRA (P=2) gives a measurable NLL improvement over single-pole (P=1).

**Procedure**:

```cpp
void MultiPoleProbe() {
    // 1. Set up two variant models: P=1 and P=2 at layer 12.
    NNetwork* model_p1 = loadFlagshipCheckpoint(/* path */);
    NNetwork* model_p2 = cloneAndSwapToSRA(model_p1, layer=12, P=2);
    NNetwork* model_p1_swapped = cloneAndSwapToSRA(model_p1, layer=12, P=1);

    // 2. Fine-tune both for 200 steps.
    fineTune(model_p1_swapped, 200_steps, lr=1e-4);
    fineTune(model_p2, 200_steps, lr=1e-4);

    // 3. Compute val NLL.
    float nll_p1 = computeNLL(model_p1_swapped, val_set);
    float nll_p2 = computeNLL(model_p2, val_set);
    float delta = nll_p1 - nll_p2;

    printf("Multi-pole: nll_p1=%.4f, nll_p2=%.4f, delta=%.4f\n", nll_p1, nll_p2, delta);

    if (delta >= 0.005f) {
        printf("PASS: P=2 gives measurable improvement.\n");
    } else {
        printf("FAIL: delta %.4f < 0.005; multi-pole expressivity claim unsupported.\n", delta);
    }
}
```

**Cost**: 15 minutes (200 steps × 2 models).

**Pass criterion**: `delta NLL ≥ 0.005 nat`.

**Fail action**: SRA's multi-pole feature doesn't help. Reduce to single-pole SRA only.

---

## 4. Stage 2 probes (2.5 GPU-hours total)

### 4.1 Probe B — Full cocycle expressivity (30 min)

Same as Probe B' but on the **full val set** (not Φ-tagged subsets). Pass: `Δ NLL_step_500 ≤ −0.015 nat` (refined from iter 2 §5.3).

### 4.2 Probe C — Sparse vs causal-complete edge set (60 min)

Run Probe B twice: once with sparse edge set (W=128, sinks=8), once with causal-complete (chunked at T=512 to fit in memory). Pass: NLL diff ≤ 0.01 nat.

### 4.3 Probes D, E, F, G, J

Smaller probes (5-30 min each). Implementations follow the pattern: load model + modify config + measure + compare. See individual paradigm design docs for specifications.

---

## 5. Stage 3 probes (5 GPU-hours total)

### 5.1 Probe K — SLR role-matched configuration (30 min on 66M)

Train two 66M models for 5000 steps: uniform SFA/SRA vs SLR with role-matched configs (after PSA-derived role assignment at step 1000). Compare final NLL and wall-clock.

### 5.2 Probes L, M, N

L: ρ_k measurement on the post-Stage-2 model.
M: train two 66M models with/without `L_reason` regulariser.
N: train three 66M models with different depth allocations.

Implementations follow the same load-modify-measure pattern.

---

## 6. Build and run instructions

### 6.1 Adding files to the build

Edit `unit-tests/Backend/Machine Learning/CMakeLists.txt`:

```cmake
# Append to existing source list:
list(APPEND TEST_SOURCES
    sfa-gate0-probes.cpp
    sfa-gate0-stage2.cpp
    sfa-gate0-stage3.cpp
)

# Append to library sources in Backend/Machine Learning/CMakeLists.txt:
list(APPEND ML_NETWORK_SOURCES
    Networks/transformer_sfa_ops.cpp
    Networks/transformer_sfa_chebyshev.cpp
)
```

Register the test in `unit-tests/main.cpp`:

```cpp
extern int SFAGate0Stage1Tests();
extern int SFAGate0Stage2Tests();
extern int SFAGate0Stage3Tests();

// In main():
if (testName == "sfa-gate0-stage1") {
    return SFAGate0Stage1Tests();
}
if (testName == "sfa-gate0-stage2") {
    return SFAGate0Stage2Tests();
}
if (testName == "sfa-gate0-stage3") {
    return SFAGate0Stage3Tests();
}
```

Add the test name to `unit-tests/test.sh`:

```bash
# Append to existing test list:
"sfa-gate0-stage1")
    cd $TEST_DIR && ./glades_unit_tests sfa-gate0-stage1
    ;;
"sfa-gate0-stage2")
    cd $TEST_DIR && ./glades_unit_tests sfa-gate0-stage2
    ;;
"sfa-gate0-stage3")
    cd $TEST_DIR && ./glades_unit_tests sfa-gate0-stage3
    ;;
```

### 6.2 Build commands

```bash
# Build with CUDA:
cd /home/robert/dev/glades-ml/unit-tests/build && sh .configure.sh cuda

# Run Gate-0 Stage 1 (~1 GPU-hour):
cd /home/robert/dev/glades-ml/unit-tests && bash test.sh sfa-gate0-stage1

# Inspect output for pass/fail of each probe.
# If Stage 1 ≥4 pass, proceed:
cd /home/robert/dev/glades-ml/unit-tests && bash test.sh sfa-gate0-stage2

# If Stage 2 all pass, proceed:
cd /home/robert/dev/glades-ml/unit-tests && bash test.sh sfa-gate0-stage3
```

### 6.3 Expected output format

Each probe prints to stdout:

```
[Probe X] STAGE 1 of 3
  Input: chiron_1B_T16384.step30000
  Configuration: d_s=64, r=4, W=128, sinks=8, M=8
  Measurements:
    metric_1: 0.0123 nat
    metric_2: 4.5
    ...
  Decision: PASS (or FAIL with reason)
  Wall-clock: 0:05:12 (5 min 12 sec)
```

A summary line at the end of each stage:

```
[Stage 1] 5 probes run. PASS: 4. FAIL: 1 (Probe B' ratio 2.3 < 4.0).
Decision: CONTINUE (≥4 pass threshold met). Proceed to Stage 2.
```

### 6.4 Aggregating results

A driver script `unit-tests/sfa-gate0-summary.sh`:

```bash
#!/bin/bash
# Run all Gate-0 stages sequentially and print summary.

set -e

cd /home/robert/dev/glades-ml/unit-tests

bash test.sh sfa-gate0-stage1 | tee stage1.log
S1_PASSED=$(grep -c "PASS" stage1.log)
S1_FAILED=$(grep -c "FAIL" stage1.log)
echo "Stage 1: PASS=$S1_PASSED FAIL=$S1_FAILED"

if [ $S1_PASSED -lt 4 ]; then
    echo "ABORT: Stage 1 had fewer than 4 passes. Halting."
    exit 1
fi

bash test.sh sfa-gate0-stage2 | tee stage2.log
S2_PASSED=$(grep -c "PASS" stage2.log)
S2_FAILED=$(grep -c "FAIL" stage2.log)
echo "Stage 2: PASS=$S2_PASSED FAIL=$S2_FAILED"

bash test.sh sfa-gate0-stage3 | tee stage3.log
S3_PASSED=$(grep -c "PASS" stage3.log)
S3_FAILED=$(grep -c "FAIL" stage3.log)
echo "Stage 3: PASS=$S3_PASSED FAIL=$S3_FAILED"

echo "===== SUMMARY ====="
echo "Total: PASS=$((S1_PASSED + S2_PASSED + S3_PASSED)) FAIL=$((S1_FAILED + S2_FAILED + S3_FAILED))"
echo "Wall-clock: see logs"
```

---

## 7. Decision matrix

After Gate-0 completes, decide implementation path based on result counts:

| Stage 1 pass | Stage 2 pass | Stage 3 pass | Action |
|---|---|---|---|
| ≥4 of 5 | ≥6 of 7 | ≥3 of 4 | **Full implementation** of #250-#254. ~50 iterations to production. |
| ≥4 of 5 | ≥6 of 7 | <3 of 4 | **Limited implementation** of #250-#252 only (skip #253/#254). ~30 iterations. |
| ≥4 of 5 | <6 of 7 | n/a | **Reduced implementation** of #250 only (skip #251+). ~20 iterations. |
| <4 of 5 | n/a | n/a | **Halt and reassess**. Possibly revert to SCFA-only improvements. |

The decision matrix encodes the priority hierarchy: paradigms with passing Gate-0 evidence proceed; those with failing evidence are skipped.

---

## 8. What success looks like

If the full implementation completes (all stages pass + Phase 1-5 implementation):

| Metric | Baseline (SCFA flagship) | Target |
|---|---|---|
| Wall-clock per training step (T=16384, 1B) | ~50 ms | **~5-10 ms** |
| Steps to target NLL 4.0 | ~50K | **~25-35K** |
| Tokens per dollar (cloud cost) | 1× | **5-10×** |
| NLL on val.tok.bin at convergence | 4.29 EMA / 3.7670 best | **0.05-0.10 nat better** (cocycle expressivity) |
| Reasoning task (synthetic multi-hop QA) | baseline | **+0.15-0.30 nat improvement** |
| Pruning ratio (PSA) | n/a | **30% layers pruned, NLL stable** |

These targets define what "success" means in a way that is **operationally checkable** post-implementation.

---

## 9. Estimated implementation timeline

After Gate-0 (≤8.5 GPU-hours):

| Phase | Iterations | Wall-clock |
|---|---|---|
| Phase 1 — CPU prototype | 3-5 | 1 week |
| Phase 2 — GPU primitives | 5-8 | 2 weeks |
| Phase 3 — Trainer wire-in | 3-5 | 1 week |
| Phase 4 — Validation (66M → 1B) | 3-5 | 1.5 weeks |
| Phase 5 — Production rollout | 1-2 | 0.5 weeks |
| **Total #250 (SFA)** | **15-25 iterations** | **~6 weeks** |
| Then add #251 SRA | 8-12 iterations | 3 weeks |
| Then add #252 PSA | 7-10 iterations | 2 weeks |
| Then add #253 SLR (optional) | 6-8 iterations | 1.5 weeks |
| Then add #254 CSR (optional) | 8-10 iterations | 2 weeks |
| **Total full stack** | **~44-65 iterations** | **~14 weeks** |

At Ralph-loop pace of ~1 iter/day, this is **~3.5 months to full production**. At faster paces (parallelisable phases), could be 6-8 weeks.

---

## 10. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| SFA primitives have bugs in CPU prototype | Medium | Run Probe A (SCFA-recovery at d_s=1) as parity test. If A passes, primitives are correct. |
| Chebyshev convergence is too slow in practice | Medium-low | Increase M to 16; switch to Lanczos with selective re-orthog if needed. |
| BF16 conditioning fails despite Jacobi preconditioning | Low-medium | Fall back to FP32 in the inner Chebyshev loop. Add diagnostic for condition number. |
| Φ-tagging is too noisy for Probe B' | Medium | Use multiple tagger heuristics; aggregate by majority vote. Or run on a hand-tagged 100-sample set. |
| Persistence-diagram computation is too slow | Low | Already pre-computed in iter 4 to be ~1 sec on 24-layer 1B model. |
| Layer-pruning fine-tune doesn't converge in 500 steps | Medium | Increase to 1000 steps if needed (×2 cost). |

---

## 11. Summary

This document is the operational bridge from CSA's design (iters 1-9) to its empirical validation. It specifies:

- 11 new C++ source files (5 in `unit-tests/`, 4 in `Backend/Machine Learning/Networks/`).
- 1 driver script `sfa-gate0-summary.sh`.
- CMake modifications, test registration, build commands.
- Probe-by-probe pseudocode, expected output format, decision criteria.
- 3-stage Gate-0 with 8.5 GPU-hour budget.

After running this plan:
- If ≥4 of Stage 1 passes (1 GPU-hour): proceed to Stage 2.
- If Stage 2 also passes: proceed to Stage 3 OR proceed to implementation.
- If failures: halt and reassess per decision matrix.

The total CSA program is now ten iterations:
1-8: design (paradigms #250-#254 + universal approximation theorem + critical reflection)
9: meta-assessment (priority ordering, expected value)
**10: operational plan (this doc) — implementation-ready**

What remains is **execution**: implement the CPU prototype, run Stage 1 of Gate-0, decide based on the priority-1 probes. The math is ready; the engineering is documented; the experiments are budgeted.
