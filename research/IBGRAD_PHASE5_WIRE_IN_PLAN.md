# IBGRAD Phase 5 — Trainer Wire-In Plan

**Status:** design complete; handoff ready.
**Date:** 2026-04-22 (Ralph-loop iteration 29).
**Prerequisites (all complete):**
- gpu_ibgrad.{h,cu} with 6 primitives, all parity tests passing
- QR bug fixed: now works at N=1M+ attention-matrix scale (commit `026e0471a`)
- CLI scaffolding: `--ibgrad-rank R`, `--ibgrad-audit-every K`,
  `--ibgrad-oja-eta F` on glades_chiron_train (commit `2802e18`)
- E2E convergence validated at N=128: 241× loss ratio with Phase-4 audit

---

## 1. Goal

Wire IBGRAD into `adam_step()` in `glades-trainer/trainer/chiron_main.cpp`.
When `cfg.ibgradRank > 0`, each attention-weight matrix (Wq, Wk, Wv, Wo)
per layer uses the subspace Adam path:

1. Dense backward produces the full N-dim gradient `g` (unchanged).
2. Project: `y = P^T · g` (r-dim).
3. Subspace Adam: `(m_sub, v_sub)` Adam update → `update_sub`.
4. Unproject: `update_full = P · update_sub`.
5. Apply: `θ -= update_full`.
6. Oja streaming PCA: `P += η_oja · g · y^T`.
7. Every K_audit steps: check captured fraction; optionally refresh + QR.

---

## 2. Additions to ChironParams

New state fields (per attention-weight matrix per layer):

```cpp
struct ChironParams
{
    // ... existing fields ...

    // Paradigm shift #19 — IBGRAD per-matrix subspace Adam.
    // Only allocated when cfg.ibgradRank > 0.
    int ibgradRank;  // 0 = disabled; typical 32 for r/N ~ 0.03 at d=1024
    std::vector<glades::gpu::GpuBuffer<float>*> Wq_P, Wk_P, Wv_P, Wo_P;
    std::vector<glades::gpu::GpuBuffer<float>*> Wq_m_sub, Wq_v_sub;
    std::vector<glades::gpu::GpuBuffer<float>*> Wk_m_sub, Wk_v_sub;
    std::vector<glades::gpu::GpuBuffer<float>*> Wv_m_sub, Wv_v_sub;
    std::vector<glades::gpu::GpuBuffer<float>*> Wo_m_sub, Wo_v_sub;
    // Shared per-step scratches (reused across all 4·L matrices in one step).
    glades::gpu::GpuBuffer<float> ibgrad_y;           // [r]
    glades::gpu::GpuBuffer<float> ibgrad_update_sub;  // [r]
    glades::gpu::GpuBuffer<float> ibgrad_update_full; // [N_attn]
    float ibgradOjaEta;
    int   ibgradAuditEvery;
    // Host-side audit state (captured-fraction tracking across matrices).
    int   ibgradStepCount;  // increments each adam_step call
};
```

Size calculation at pile_large (d_model=1024, L=24):
- N_attn per matrix = 1024² = 1M
- r = 32 (target r/N = 0.03)
- P per matrix: 1M × 32 × 4B = 128 MB → total 4·24·128 MB = 12.3 GB
- m_sub + v_sub per matrix: 32 × 4B × 2 = 256 B (negligible)

**Memory overhead**: 12.3 GB for the P matrices (competing with the 16 GB VRAM budget).
This is concerning — IBGRAD only pays if the backward-GEMM savings and
Adam-state savings exceed the P overhead.

**Alternative:** use BF16 storage for P (128 MB → 64 MB per matrix) →
6.1 GB total.  Still substantial.

**Better alternative:** share P across matrices in the same layer
(one P per layer, used for all 4 attention weights).  This reduces to
24 × 128 MB = 3 GB.  But changes the per-matrix-gradient semantics.

**Conclusion:** **IBGRAD at r/N = 0.03 on all 4·L matrices is infeasible
for 16 GB VRAM at pile_large.**  Must gate IBGRAD to the LARGEST matrix
group only (e.g., just Wo = L · 128 MB = 3 GB), accept partial
coverage.  Alternatively scale down r (r = 8 → 32 MB per matrix → 3 GB
total).

## 3. Init code additions (in ChironParams::allocate())

```cpp
if (ibgradRank > 0) {
    const size_t N_attn = (size_t)dModel * dModel;
    const unsigned int r = (unsigned int)ibgradRank;

    // Allocate scratches once.
    ibgrad_y.allocate(r);
    ibgrad_update_sub.allocate(r);
    ibgrad_update_full.allocate(N_attn);

    // Per-layer, per-matrix allocation.
    for (int l = 0; l < L; ++l) {
        // Wq, Wk, Wv, Wo: each gets its own P + Adam state.
        for (auto& vec : {&Wq_P, &Wk_P, &Wv_P, &Wo_P}) {
            auto* P = new glades::gpu::GpuBuffer<float>();
            P->allocate(N_attn * r);
            glades::gpu::ibgrad_init_projection(P->data(), N_attn, r,
                /* seed = */ (uint64_t)(seed + l * 100 + vec - &Wq_P));
            glades::gpu::ibgrad_qr_reorthogonalize(P->data(), N_attn, r);
            vec->push_back(P);
        }
        for (auto& vec : {&Wq_m_sub, &Wq_v_sub, &Wk_m_sub, &Wk_v_sub,
                          &Wv_m_sub, &Wv_v_sub, &Wo_m_sub, &Wo_v_sub}) {
            auto* buf = new glades::gpu::GpuBuffer<float>();
            buf->allocate(r);
            std::vector<float> z(r, 0.0f);
            buf->upload(&z[0], r);
            vec->push_back(buf);
        }
    }
}
```

## 4. adam_step modifications

Add an IBGRAD branch inside the per-layer loop.  Pseudocode:

```cpp
static bool adam_step(const Config& cfg, ChironParams& W, int step,
                      float gradScale, float lrScale)
{
    // ... existing CPU-Adam path, int8-Adam path, etc ...

    if (W.ibgradRank > 0) {
        const unsigned int r = (unsigned int)W.ibgradRank;
        const size_t N_attn = (size_t)cfg.m * 2 * cfg.m * 2;  // dModel²

        // For each layer's attention-weight matrix:
        for (int l = 0; l < cfg.L; ++l) {
            for (int k = 0; k < 4; ++k) {  // Wq, Wk, Wv, Wo
                float* theta      = (k==0? W.Wq[l] : k==1? W.Wk[l] : k==2? W.Wv[l] : W.Wo[l])->data();
                float* g          = (k==0? W.dWq[l] : k==1? W.dWk[l] : k==2? W.dWv[l] : W.dWo[l])->data();
                float* P          = (k==0? W.Wq_P[l] : k==1? W.Wk_P[l] : k==2? W.Wv_P[l] : W.Wo_P[l])->data();
                float* m_sub      = (k==0? W.Wq_m_sub[l] : k==1? W.Wk_m_sub[l] : k==2? W.Wv_m_sub[l] : W.Wo_m_sub[l])->data();
                float* v_sub      = (k==0? W.Wq_v_sub[l] : k==1? W.Wk_v_sub[l] : k==2? W.Wv_v_sub[l] : W.Wo_v_sub[l])->data();

                // 1. Project
                glades::gpu::ibgrad_project(P, g, N_attn, r, W.ibgrad_y.data());

                // 2. Subspace Adam
                //    m_sub ← b1·m_sub + (1-b1)·y
                //    v_sub ← b2·v_sub + (1-b2)·y²
                //    update_sub ← lr · m_hat / (sqrt(v_hat) + eps)
                glades::gpu::adam_update(W.ibgrad_update_sub.data(),
                                         W.ibgrad_y.data(),
                                         m_sub, v_sub,
                                         cfg.lr, cfg.beta1, cfg.beta2,
                                         cfg.eps_adam, 0.0f, gradScale, step,
                                         (int)r);
                //    Note: adam_update is in-place update of params.
                //    We need a variant that writes the update to a buffer
                //    without mutating params.  TODO: add adam_compute_update.

                // 3. Unproject
                glades::gpu::ibgrad_unproject(P, W.ibgrad_update_sub.data(),
                                              N_attn, r,
                                              W.ibgrad_update_full.data());

                // 4. Apply update to θ
                //    θ -= update_full.  TODO: need a simple axpy-style kernel.

                // 5. Oja streaming PCA
                glades::gpu::ibgrad_oja_rank1_update(P, g, W.ibgrad_y.data(),
                                                    N_attn, r, W.ibgradOjaEta);

                // 6. Phase-4 audit every K_audit steps
                if (step % W.ibgradAuditEvery == 0) {
                    // Measure captured_frac; decide if refresh needed.
                    // This needs a GPU primitive ibgrad_captured_fraction
                    // that returns a float on device.  TODO.
                    glades::gpu::ibgrad_refresh_first_column(P, g, N_attn, r);
                    glades::gpu::ibgrad_qr_reorthogonalize(P, N_attn, r);
                }
            }
        }
        return true;
    }

    // ... fall through to existing dense Adam path ...
}
```

## 5. Missing primitives (to add in Phase 5 Phase A)

1. **`ibgrad_compute_update`**: like `adam_update` but writes the update to
   a separate buffer (doesn't mutate params).  Needed so we can unproject
   before applying.

2. **`ibgrad_apply_update`**: in-place `θ -= update_full`.  Can be a
   trivial axpy kernel.

3. **`ibgrad_captured_fraction`** (optional, for conditional refresh):
   returns `‖P^T g‖² / ‖g‖²` as a device scalar.  Can be composed from
   existing `ibgrad_project` + `ibgrad_norm_sq` (already in
   refresh_first_column).

## 6. Phased rollout

Phase 5A: add missing primitives.  Parity tests.
Phase 5B: allocate IBGRAD state in ChironParams (guarded by ibgradRank > 0).
Phase 5C: wire the subspace Adam path in adam_step (guarded).
Phase 5D: smoke test at small config (L=2, m=128) verifying it produces
  sensible loss descent.
Phase 5E: scale to pile_large config; measure throughput vs dense; measure
  Adam-state VRAM savings.

## 7. Memory constraints & decision tree

At r=32, the P memory is 128 MB × (4·L) = 12 GB at pile_large.  With 16 GB
VRAM total, this does not fit alongside the other state.  Options:

- **Option A**: limit IBGRAD to Wo only (3 GB) — covers 25% of attention
  params with full subspace benefit.
- **Option B**: reduce r to 8 (32 MB × 4L = 3 GB) — covers all 4 matrices
  but at 4× smaller subspace (may hurt convergence).
- **Option C**: BF16 storage for P (64 MB × 4L = 6 GB) — half the memory
  with mild precision loss in the Oja updates.
- **Option D**: share P across attention matrices within a layer
  (32 MB × L = 0.8 GB) — most compact but requires per-layer gradient
  fusion.

Recommendation: **start with Option A** (Wo only, full r=32) for Phase
5D smoke test.  Evaluate convergence.  Iterate to A→C→D as needed.

## 8. Test plan

- **Smoke test:** chiron_train at `--layers 2 --m 128 --ibgrad-rank 16
  --ibgrad-audit-every 10` with 100 steps.  Assert: loss < initial
  within 5% and no NaN.
- **Scale test:** chiron_train at `--layers 24 --m 512 --ibgrad-rank 32
  --ibgrad-audit-every 50` with 500 steps.  Measure tokens/sec; compare
  to dense Adam baseline.  Target: IBGRAD within 1.5× of dense
  wall-clock at 20× Adam state reduction for Wo.
- **Parity test:** a 4-block transformer trained with and without
  IBGRAD-on-Wo; assert final loss within 20% of dense.

## 9. Open issues

- Adam variant that writes update without mutating params: needs a new
  GPU primitive.
- Pool-shared P (Option D) would require a new fused backward + project
  kernel.
- BF16 P (Option C) requires casting in every Oja + QR + gather.
- Currently ibgrad_refresh_first_column uses thread-local static device
  pointer for ‖g‖² scalar — fine for single-threaded trainer use but
  not safe for parallel trainers.  Wrap in ChironParams state for
  thread safety at trainer-level.

---

**This plan is the handoff document for Phase 5.**  Estimated effort:
~1500 LOC total across gpu_ibgrad.cu (new primitives) and chiron_main.cpp
(state allocation + adam_step wire-in).  ~3 iterations of Ralph-loop
work depending on scope.
