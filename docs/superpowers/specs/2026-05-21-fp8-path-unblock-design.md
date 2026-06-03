# FP8 Path Unblock — Design Spec

**Date**: 2026-05-21
**Branch**: pants (glades-ml) + main (glades-trainer)
**Status**: design proposal, approved for execution
**Author**: Claude Code session 2026-05-21
**Baseline**: CHIRON 1B iter 116 ship flagship (28,257 tok/s @ T=16384, val NLL 4.2039 @ 30k)

---

## 1. TL;DR

Two FP8 paths exist in glades-ml/glades-trainer as **dead code with runtime kill-switches**:

1. `--fp8-readout-fwd` (iter 62 / Arc 1) — three readout GEMMs via E4M3
2. `--fp8-attn` (iter 55 / HELIUM) — Q/K/V/O projections via E4M3

Both were falsified at iter 55 / iter 62 because **CUDA 12.0 cuBLASLt has incomplete FP8 algo coverage on Ada (sm_8.9)**. cuBLASLt returns `CUBLAS_STATUS_NOT_SUPPORTED` (status 15) at the actual GEMM shapes; the runtime kill-switch then falls back to BF16. The iter 62 doc explicitly notes that **upgrading to CUDA 12.3+ would auto-unblock both paths with no further code work**.

The system now has **CUDA 13.2** installed at `/usr/local/cuda-13.2` (well past the 12.3+ threshold). Default `nvcc` (`/usr/bin/nvcc`) still resolves to CUDA 12.0. This spec covers the **toolchain switch + revalidation** workflow.

This is **not a new design** — the FP8 kernel design lives in `PARADIGM_FP8_READOUT_DESIGN.md` and `PARADIGM_SHIFT_50_CANDIDATE_A_HELIUM.md`. This spec captures the operational unblock plan.

---

## 2. Goals

| Goal | Metric |
|---|---|
| Switch build to CUDA 13.2 cleanly | Compile + link OK on glades-ml + glades-trainer with no source rewrites beyond minimal API drift |
| FP8 readout actually executes | No "FP8 readout disabled" log; nsys shows FP8 GEMM kernel(s) on the readout shape |
| FP8 attention actually executes | No "FP8 attn disabled" log; nsys shows FP8 GEMM kernel(s) on attention shapes |
| Gate-0 PASS per existing design docs | tok/s ≥ +3% over iter 116 baseline at val NLL ≤ +0.02 nat strict @ step 200 |
| Multi-seed PASS (if Gate-0 passes) | n=3 seed mean ≤ +0.02 nat NLL drift, std ≤ 0.10, mean wall ≥ +3% |
| Zero flagship risk | All work behind opt-in flags; iter 116 ship default-on behavior unchanged |

Non-goals:

- Implementing FA-3 (HELIUM's larger scope).
- Adding FP8 storage on FFN matrices (deferred).
- Production retrain — only proposed AFTER smoke + multi-seed PASS, with explicit user approval.

---

## 3. Background

### 3.1 Why FP8 was blocked

`ITER62_FP8_READOUT_NULL.md`:

> cuBLAS status 15 = `CUBLAS_STATUS_NOT_SUPPORTED`. Three priors converge:
> 1. iter 55 HELIUM hit the same status at attention shapes (smaller).
> 2. Shape (M=8192, N=V=32000, K=2048): V=32000 is non-power-of-2; cuBLASLt FP8 algo tables on Ada may require finer alignment.
> 3. CUDA 12.0 cuBLASLt is known to have incomplete FP8 algo coverage on sm_8.9. CUDA 12.3+ improves this materially.

> **iter 62 fail = paradigm invalidated** per the original design plan. Pivot to BF16-residual-p or MoE FFN.
>
> **However**: "If the cuBLAS toolkit upgrades to 12.3+ in a future system-software refresh, the FP8 wrapper is in tree and the flag `--fp8-readout-fwd` would light up automatically — Gate-0 retesting is then a single bench command. No further code work needed."

### 3.2 What is in tree

**glades-ml**:

- `Backend/Machine Learning/Networks/cuda/gpu_blas_fp8.cu` + `.h` — cuBLASLt FP8 GEMM wrappers
  - `sgemm_rowmajor_fp8_e4m3_bf16` (NN, FP32 out — used by HELIUM)
  - `sgemm_rowmajor_abt_fp8_e4m3_bf16_bf16out` (ABT, BF16 out — used by FP8 readout)
- `Backend/Machine Learning/Networks/cuda/gpu_chiron.cu` — `chiron_attention_shear_fp8w_tiled` (HELIUM attention shear)

**glades-trainer**:

- `trainer/chiron_main.cpp` — `--fp8-attn`, `--fp8-readout-fwd` flag wiring + scale buffers + runtime kill-switches.

### 3.3 Current build state

```
/usr/bin/nvcc            → CUDA 12.0.140 (default; blocked)
/usr/local/cuda-13.2/bin/nvcc → CUDA 13.2.78 (target)
/usr/local/cuda          → /etc/alternatives/cuda (symlink)
libcublasLt.so.13        → /usr/local/cuda/targets/x86_64-linux/lib/libcublasLt.so.13
libcublasLt.so.12        → /lib/x86_64-linux-gnu/libcublasLt.so.12 (legacy)
```

Build CMake declares: `CMAKE_CUDA_ARCHITECTURES = "60;70;75;80;86;89;90"`. **CUDA 13 dropped sm_60 and sm_70**; will fail to compile without trimming.

---

## 4. Plan

### Phase 0 — Build toolchain switch (Tasks 1 + 2)

1. Identify all build entry points that invoke nvcc/CMake:
   - `/home/robert/dev/glades-ml/.configure.sh`
   - `/home/robert/dev/glades-ml/unit-tests/build/.configure.sh`
   - `/home/robert/dev/glades-trainer/.configure.sh` (or equivalent)
2. Export `CUDACXX=/usr/local/cuda-13.2/bin/nvcc` + `PATH=/usr/local/cuda-13.2/bin:$PATH` + `LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64:$LD_LIBRARY_PATH` for build + run.
3. Trim `CMAKE_CUDA_ARCHITECTURES` in CMakeLists.txt to `75;80;86;89;90` (drop 60, 70).
4. Clean rebuild glades-ml; install to `~/.local`.
5. Clean rebuild glades-trainer linking against the freshly-installed glades-ml.
6. Diff any cuBLAS 13 API breakage. Known risk surface in `gpu_blas_fp8.cu`:
   - `cublasLtMatmulDescSetAttribute` enum names
   - `CUBLAS_COMPUTE_*` compute-type semantics
   - cuBLASLt FP8 scaling-mode attributes (some refactored in 12.4+)
   - `cudaDataType_t` deprecations
7. Patch in place. Acceptance: clean `make install` exit 0 + `ldd libglades.so | grep cublasLt` shows `.so.13`. **Patch budget**: ≤50 LOC across ≤3 source files. If exceeded, pause and escalate (likely indicates a broader cuBLAS 13 API rewrite spec is needed).

### Phase A — FP8 readout smoke (Task 3)

Single-seed bench against iter 116 ship baseline:

```bash
cd ~/dev/glades-trainer
sh run.sh flagship --fp8-readout-fwd --max-steps 200 --seed 1337 --no-checkpoint
```

Acceptance:

- No log line `[W] [fp8-readout-fwd] cuBLASLt FP8 readout GEMM rejected`.
- nsys profile shows at least one FP8 kernel (`*_e4m3_*` or cublasLt FP8 algo) on the readout shape (M=T, N=V=32000, K=2048).
- tok/s recorded for comparison; val NLL @ step 200 recorded.

Failure modes:

- Kill-switch trips → cuBLASLt 13.2 still rejects on Ada at this shape. Genuine algo gap, not toolkit gap. Document in `research/ITER_FP8_READOUT_REVAL_NULL.md`. Stop.
- Compile/link OK but runtime crash → file-level patch to `gpu_blas_fp8.cu` (API drift). Iterate.

### Phase B — FP8 attention (HELIUM) smoke (Task 4)

Same workflow with `--fp8-attn`. Same acceptance criteria, attention shapes per iter 55 doc (~0.07 TFLOP per QKVO GEMM at L=24).

### Phase C — Gate-0 + multi-seed validation (Task 5)

If Phase A passes single-seed:

1. Apples-to-apples 200-step bench: baseline (iter 116 ship, no FP8 flag) vs `--fp8-readout-fwd`, same seed=1337. Compare wall + NLL.
2. Per iter 62 Gate-0:
   - `C0a` tok/s ≥ +3% (relaxed bar per iter 60)
   - `C0b` val NLL @ step 200 ≤ +0.02 nat
   - `C0c` val NLL @ step 1000 ≤ +0.02 nat (slow-drift trap; conditional on C0a + C0b PASS)
3. If single-seed Gate-0 PASS, n=3 seeds (1337/1338/1339) for strict NLL parity claim per iter 91 methodology.

If Phase B passes single-seed: same protocol, HELIUM-specific gates.

### Phase D — Production retrain (BLOCKED on explicit user approval)

If multi-seed PASS:

- 30k-step apples-to-apples Phase 2 per iter 94 protocol.
- Estimated cost: ~10h on RTX 4080 SUPER if +3-6% wall realized.
- Output: new flagship checkpoint candidate.

**Phase D is not executed without separate user authorization.** This spec only authorizes Phases 0/A/B/C.

---

## 5. Risks & mitigations

| Risk | Trigger | Mitigation |
|---|---|---|
| CUDA 13 API drift breaks FP8 wrappers | Compile error in gpu_blas_fp8.cu | Patch in place; small expected surface |
| cuBLAS 13 ABI break breaks rest of build | Compile or link error in unrelated cuBLAS calls | Same; iterate file by file |
| FP8 algo gap on Ada persists at CUDA 13.2 | Kill-switch trips at runtime | Document as genuine Ada limitation; HELIUM/readout stay null. Truly stop. |
| FP8 NLL drift > +0.05 nat @ step 200 | iter 62 R1/R2 (early softmax / E5M2 underflow) | Revert flag; stays kill-switched |
| Multi-seed instability (iter 80 pattern) | n=3 std > 4× baseline std | Mark as parity-clean-below-bar; do not ship default-on |
| Unrelated code regression in BF16 path | iter 116 ship baseline NLL changes after rebuild | Revert toolchain; investigate |
| CUDA 13 deprecation surface beyond FP8 wrappers (e.g., cuRAND, NCCL, cuDNN) | Compile/link errors outside gpu_blas_fp8.cu | Per-file patch; if total patch >50 LOC or touches >3 distinct API surfaces, escalate to user for scope decision |

---

## 6. Rollback

Any failure during Phases 0/A/B:

```bash
unset CUDACXX PATH LD_LIBRARY_PATH  # restore default
# revert CMAKE_CUDA_ARCHITECTURES edit if compile fails
cd ~/dev/glades-ml && sh .configure.sh cuda  # rebuild with CUDA 12.0 default
cd ~/dev/glades-trainer && sh .configure.sh cuda  # likewise
```

`--fp8-readout-fwd` and `--fp8-attn` remain default-off; iter 116 ship behavior is bit-identical to pre-spec state.

---

## 7. Deliverables

| File | Purpose |
|---|---|
| `docs/superpowers/specs/2026-05-21-fp8-path-unblock-design.md` | This spec |
| `research/ITER_FP8_REVALIDATION_<verdict>.md` | Per-phase bench results (one per phase that runs) |
| `CLAUDE.md` update (conditional on Phase C PASS) | Note CUDA 13.2 + FP8 flags as production-ready opt-in |
| `~/.claude/projects/.../memory/iter_fp8_revalidation.md` | Auto-memory entry mirroring research doc |

---

## 8. Success criteria (overall)

**Spec succeeds if at least one of:**

1. Phase A passes single-seed AND multi-seed Gate-0 → `--fp8-readout-fwd` is production-ready opt-in; user can authorize Phase D.
2. Phase B passes equivalently → `--fp8-attn` likewise.
3. Both phases trip kill-switch on CUDA 13.2 → documented as definitive Ada FP8 algo gap; FP8 paths confirmed null on this hardware regardless of toolkit. (Negative result is still informative; closes the question.)

**Spec fails if:**

- Phase 0 doesn't complete (build broken on CUDA 13.2 with non-trivial source rewrites required). Falls back to user direction (possibly: stay on CUDA 12.0, or budget code rewrite as a separate spec).

---

## 9. Out of scope

- FA-3 implementation (HELIUM Phase 1-2, ~1100 LOC). Separate spec if pursued.
- FP8 FFN MLP (HELIUM Phase 5+). Separate spec.
- Hopper FP8 path with TMA / warp-specialization. Not present on RTX 4080 SUPER.
- New trainer flags. Reuses existing `--fp8-attn` and `--fp8-readout-fwd`.
- Inference-side FP8 (`transformer_infer.cpp` stays BF16).
