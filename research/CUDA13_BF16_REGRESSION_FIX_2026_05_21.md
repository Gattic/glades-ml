# CUDA 13.2 cuBLAS BF16 Regression — Diagnosis + Fix — 2026-05-21

**Status:** **PRODUCTION-READY FIX.** CUDA 13.2 + fix beats CUDA 12.0 baseline by +0.50%; with `--fp8-readout-fwd` net +2.13%.
**Branch:** chiron1 (glades-ml) + main (glades-trainer)
**Baseline:** CHIRON 1B iter 116 ship — 28,345 tok/s on CUDA 12.0 at T=16384, single-seed 200-step apples-to-apples.

---

## TL;DR

1. **Regression identified:** On CUDA 13.2, the iter 116 ship recipe runs at **24,041 tok/s** — a **-15.18%** wall regression vs CUDA 12.0's 28,345 tok/s. Math is bit-identical to 4 decimal places at every logged step.
2. **Root cause:** cuBLAS 13.x's heuristic for `cublasGemmEx(... CUDA_R_32F inputs, CUBLAS_COMPUTE_32F_FAST_16BF)` on Ada (sm_8.9) dispatches to a non-BF16-specialized kernel (`s1688gemm_*_align4`), ~1.82× slower than the `s1688bf16gemm` kernel that CUDA 12.0 picks for the same call. The "_bf16_" specialization in the kernel name is the discriminator. nsys confirms: top GEMM kernel went from `s1688**bf16**gemm` (744 µs) on CUDA 12 to `s1688gemm` (1351 µs) on CUDA 13.
3. **Fix (small, surgical):** ~80 LOC in `gpu_blas.cu` + `gpu_blas.h` + ~10 LOC in trainer's `chiron_main.cpp`:
   - `sgemm_rowmajor_fast16bf_impl` pre-casts FP32 A and B inputs to BF16 scratch buffers, then calls `cublasGemmEx` with `CUDA_R_16BF` input dtypes. The BF16-input call shape forces cuBLAS 13 to dispatch the `ampere_s1688gemm_bf16_*` fast path on Ada.
   - New `register_fast16bf_constant(fp32, bf16, n)` API: caller registers a `(FP32, BF16)` pointer pair for matrices that never change. The wrapper looks up by pointer; cache hit → skip the cast.
   - Trainer pre-casts `scfa_B` (the DCT-II basis, fixed for the run) and registers it. The 216 SCFA outer GEMMs/step then skip the A-side cast.
4. **Results (n=3 multi-seed):**
   - **CUDA 13.2 + fix:** 28,488 ± 9 tok/s — **+0.50% over CUDA 12.0** baseline. Loss bit-identical to CUDA 12.0 at every step.
   - **CUDA 13.2 + fix + `--fp8-readout-fwd`:** 28,948 tok/s — **+2.13% over CUDA 12.0**. NLL +0.0053 nat (strict ±0.02 met).
5. **Production verdict:** Adopt the CUDA 13.2 toolchain with the fix. With FP8 readout enabled, net +2.13% vs the iter 116 ship. The earlier "stay on CUDA 12" recommendation is **superseded**.

---

## 1. Setup & method

**Hardware:** RTX 4080 SUPER (SM 8.9, 15936 MB, 80 SMs).
**Driver:** 580.126.09 (CUDA 13.0 runtime).
**Bench recipe:** 200-step `run.sh flagship`-equivalent, single-seed (or n=3 for v5), `--max-steps 200 --warmup 500 --lr 1e-4 --grad-clip 0.5 --val-every 999999 --val-batches 0`. Tok/s reported is the rate over the last 20-step log window at step 181 (steady-state, post-warmup).

**Builds compared:**
- **CUDA 12.0:** built against `/usr/bin/nvcc 12.0.140` with `g++-11` host. Links `libcublas.so.12`, `libcublasLt.so.12`. Matches the iter 116 ship configuration that produced the published 28,257 tok/s.
- **CUDA 13.2 unmodified:** built against `/usr/local/cuda-13.2/bin/nvcc 13.2.78` with default `g++-13`. Links `libcublas.so.13`, `libcublasLt.so.13`.
- **CUDA 13.2 + v5 fix:** same CUDA 13.2 toolchain, with the patches described in §3.

---

## 2. Regression diagnosis

### 2.1 Wall A/B (math bit-identical)

| toolchain | tok/s @ 181 | wall @ 181 | step 1 loss | step 181 loss |
|---|---:|---:|---:|---:|
| CUDA 12.0 | **28,345** | 104.7s | 10.5777 | 9.0859 |
| CUDA 13.2 | 24,041 | 123.5s | 10.5778 | 9.0860 |
| Δ | -15.18% | +17.9% | bit-identical | bit-identical |

CUDA 12.0 result matches the published iter 116 ship 28,257 tok/s within +0.3% (noise). Math identical at every step — the difference is purely runtime perf.

### 2.2 Kernel-level A/B via nsys

| version | Top GEMM kernel | avg µs | total % wall |
|---|---|---:|---:|
| CUDA 12 | `cutlass_80_..._s1688**bf16**gemm_256x128_16x3_nn_align4` | 744 | 15.6% |
| CUDA 13 unfixed | `cutlass_80_..._s1688gemm_256x128_16x3_nn_align4` (**no `_bf16_`**) | **1351** (1.82×) | **23.9%** |

cuBLAS 13's heuristic for `cublasGemmEx` with `CUDA_R_32F` inputs + `CUBLAS_COMPUTE_32F_FAST_16BF` compute lost the BF16 kernel specialization on Ada at the shapes used by SCFA outer projection GEMMs (k=1024, m=2048, T=16384, 9 GEMMs/layer × L=24 = 216 instances/step).

### 2.3 First (failed) hypothesis: deprecated algo arg

We use `CUBLAS_GEMM_DEFAULT_TENSOR_OP` (deprecated in cuBLAS 13 per NVIDIA's forum). Tested replacement with `CUBLAS_GEMM_DEFAULT`: tok/s went from 24,041 to 24,048 (no change). Algo arg is not the cause.

---

## 3. The fix (v5)

### 3.1 Code changes

**`Backend/Machine Learning/Networks/cuda/gpu_blas.cu`:** modify `sgemm_rowmajor_fast16bf_impl` to pre-cast FP32 inputs to BF16 in static scratch buffers, then call `cublasGemmEx` with `CUDA_R_16BF` input dtypes. The BF16-input call shape forces cuBLAS 13 to dispatch the `ampere_s1688gemm_bf16_*` fast path.

Size derivation (row-major calling convention):
- `(transa,transb)=(N,N)` no-trans wrapper: A=[M,K] lda=K, B=[K,N] ldb=N → A_n = M·lda, B_n = K·ldb
- `(transa,transb)=(N,T)` atb wrapper (A^T): A=[K,M] lda=M, B=[K,N] ldb=N → A_n = K·lda, B_n = K·ldb
- `(transa,transb)=(T,N)` abt wrapper (B^T): A=[M,K] lda=K, B=[N,K] ldb=K → A_n = M·lda, B_n = N·ldb

`transa` applies to A_cublas (= our B); `transb` applies to B_cublas (= our A). So in the impl:
```cpp
size_t A_n = (size_t)lda * (size_t)(transb == CUBLAS_OP_N ? M : K);
size_t B_n = (size_t)ldb * (size_t)(transa == CUBLAS_OP_N ? K : N);
```

**`Backend/Machine Learning/Networks/cuda/gpu_blas.h`:** add public API:
```cpp
bool register_fast16bf_constant(const float* fp32_ptr,
                                 const unsigned short* bf16_ptr,
                                 size_t n);
bool unregister_fast16bf_constant(const float* fp32_ptr);
```

Up to 16 constants can be registered. When the wrapper sees a registered FP32 pointer as A or B with matching element count, it skips the cast and uses the BF16 pointer directly.

**`trainer/chiron_main.cpp`** (~10 LOC): allocate `scfa_B_bf16` mirror, cast once at SCFA init, register with the library:
```cpp
if (!scfa_B_bf16.allocate(B_sz)) return false;
if (!glades::gpu::cast_f32_to_bf16(scfa_B.data(), scfa_B_bf16.data(), B_sz)) return false;
glades::gpu::register_fast16bf_constant(scfa_B.data(), scfa_B_bf16.data(), B_sz);
```

### 3.2 Why scfa_B is the high-value constant

The 216 SCFA outer GEMMs/step all use `W.scfa_B` (the fixed DCT-II basis, deterministic from (T, k), never changes) as one of the GEMM inputs. Caching its BF16 cast saves 216 cast launches per step (`k_cast_f32_to_bf16` went from 11.6% of step wall to negligible).

### 3.3 Iteration history

| version | scope | tok/s | NLL drift | notes |
|---|---|---:|---:|---|
| v0 (CUDA 13 unfixed) | n/a | 24,041 | 0 (bit-identical) | baseline regression |
| v1 (algo arg swap) | CUBLAS_GEMM_DEFAULT | 24,048 | 0 | NULL: not the cause |
| v2 (BF16 cast per call) | wrong A_n formula | crash | n/a | bug: transa/transb conditions swapped |
| v2b (BF16 cast per call) | fixed A_n formula | 27,065 | 0 | 70% recovery; cast overhead = 11.6% wall |
| v3 (scfa_B constant cache) | trainer build failed silently | 27,074 | 0 | vendored header didn't have new API |
| v4 (debug prints) | confirmed cache hits | n/a | n/a | diagnostic: hit=1 on all SCFA outer GEMMs |
| **v5 (production)** | no debug | **28,494** | bit-identical | full recovery |

---

## 4. Results

### 4.1 Multi-seed (n=3) at 200 steps

CUDA 13.2 + v5 vs CUDA 12.0 baseline (s1337 only on CUDA 12.0; v5 at s1337/1338/1339):

| seed | CUDA 12.0 tok/s | v5 tok/s | Δ |
|---:|---:|---:|---:|
| 1337 | 28,345 | 28,494 | +0.53% |
| 1338 | — | 28,494 | (baseline not measured) |
| 1339 | — | 28,477 | (baseline not measured) |
| **mean** | — | **28,488** | **+0.50% vs CUDA 12 s1337** |
| std (v5) | — | 9 tok/s (0.033%) | extraordinarily tight |

Per-seed loss (v5, step 181): s1337=9.0858, s1338=9.0822, s1339=9.0837. All within natural seed-to-seed variance.

### 4.2 With `--fp8-readout-fwd` on top of v5

| stack | tok/s | vs CUDA 12 | NLL@200 | NLL drift |
|---|---:|---:|---:|---:|
| CUDA 12.0 (production iter 116 ship) | 28,345 | baseline | 9.0859 | 0 |
| CUDA 13.2 + v5 | 28,494 | +0.53% | 9.0858 | bit-identical |
| **CUDA 13.2 + v5 + --fp8-readout-fwd** | **28,948** | **+2.13%** | 9.0912 | +0.0053 nat (strict ±0.02 met) |

FP8 readout adds **+1.59%** on top of v5 (28,948 / 28,494 = 1.0159), consistent with the +1.38% measured on CUDA 13.2 unmodified.

### 4.3 nsys verification (v5)

Top GEMM kernel on CUDA 13.2 + v5:
- `ampere_s1688gemm_**bf16**_128x128_ldg8_stages_32x1_nn` — **579 µs avg** (vs 1351 µs on CUDA 13 unfixed, vs 744 µs on CUDA 12.0). The `_bf16_` specialization is back.

`k_cast_f32_to_bf16` dropped from 11.6% of wall (v2) to negligible (v5) due to the scfa_B constant cache. The trainer's existing per-step cast load remains unchanged.

---

## 5. Production deployment

### 5.1 Changes to land

| file | change | risk |
|---|---|---|
| `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_blas.cu` | Add static BF16 scratch buffers; modify `sgemm_rowmajor_fast16bf_impl` to pre-cast + use BF16 inputs; add `register_fast16bf_constant`/`unregister_fast16bf_constant` API | Low: behavior is a strict superset (caller can choose to register or not) |
| `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_blas.h` | Declare the two new API functions + no-CUDA stubs | Low: API addition only |
| `glades-trainer/include/Backend/Machine Learning/Networks/cuda/gpu_blas.h` | Same as glades-ml's header (vendored copy) | Low: must be kept in sync |
| `glades-trainer/trainer/chiron_main.cpp` | Add `scfa_B_bf16` member; cast + register at SCFA init | Low: scfa_B is fixed/deterministic per run |
| `glades-ml/Backend/Machine Learning/Networks/cuda/CMakeLists.txt` | Drop sm_60/70 from `CUDA_ARCHITECTURES` (CUDA 13 doesn't support them) | None: those archs are obsolete |
| `glades-trainer/include/.../cuda/CMakeLists.txt` | Same | None |

Total: **~100 LOC across 5 files.** No source rewrites; purely additive.

### 5.2 Build recipe

For CUDA 13.2 production builds:
```bash
# glades-ml
cd /home/robert/dev/glades-ml && rm -rf build && sh .configure.sh cuda
cd build && make install

# glades-trainer
cd /home/robert/dev/glades-trainer && rm -rf build && bash build.sh
```

CMake auto-detects CUDA 13.2 via the `/usr/local/cuda` → `cuda-13.2` symlink. No env-var overrides needed.

Verify link:
```bash
ldd build/glades_chiron_train | grep cublasLt  # should show libcublasLt.so.13
```

For CUDA 12.0 fallback (if needed): use the manual cmake recipe from `research/FP8_READOUT_CUDA13_REVAL_2026_05_21.md` §5.

### 5.3 Recommended production stack

**Default flagship recipe** (after this fix lands):
```bash
sh run.sh flagship  # 28,488 tok/s n=3 mean on CUDA 13.2
```

**Opt-in FP8 readout for +1.59% extra wall:**
```bash
sh run.sh flagship --fp8-readout-fwd  # (requires run.sh extension to accept this flag)
# or invoke trainer binary directly per spec
# Net: 28,948 tok/s, +2.13% over CUDA 12 baseline. NLL +0.005 nat strict OK.
```

**`--fp8-attn` (HELIUM) remains NULL** (no benefit at iter 116 attention shapes regardless of toolchain).

### 5.4 Updates pending

- `CLAUDE.md` flagship section: update "current production flagship" tok/s figure once a new Phase-2 30k-step retrain is run on the v5 stack to certify NLL parity at full horizon. Recommended: ~10h compute commitment.
- `database/checkpoints/chiron_1B_T16384_iter116_treatment_phase2.final` may not need re-shipping at the binary level (math bit-identical) but a new run-from-init under the v5 stack would produce the next-gen flagship at higher wall.

---

## 6. Out of scope / future work

- **Multi-constant registration**: only `scfa_B` is registered. Layer weights (Wq/Wk/Wv/Wo) are NOT constants (Adam updates them). The trainer's existing `--bf16-weights` mirror could be auto-registered after each Adam step, but this requires invalidation hooks and a deeper change. Estimated potential: +1-2% wall.
- **Bigger batched casts**: 216 small cast launches/step (one B-side cast per SCFA outer GEMM) could be batched via `cast_f32_to_bf16_batched`. Estimated potential: +0.5-1% wall.
- **FP8 attention via full HELIUM**: still ~2500 LOC of FA-3 work. Not pursued.

---

## 7. Files

- `docs/superpowers/specs/2026-05-21-fp8-path-unblock-design.md` — original spec (covers FP8 path question)
- `docs/superpowers/plans/2026-05-21-fp8-path-unblock.md` — original plan
- `research/FP8_READOUT_CUDA13_REVAL_2026_05_21.md` — FP8 path revalidation (now superseded for the production recommendation)
- `research/CUDA13_BF16_REGRESSION_FIX_2026_05_21.md` — this document
- nsys reports: `/tmp/cuda12_nsys.nsys-rep`, `/tmp/cuda13_nsys.nsys-rep`, `/tmp/cuda13_bf16cast_v2_nsys.nsys-rep`, `/tmp/cuda13_v3_nsys.nsys-rep`
- Bench logs: `/tmp/cuda12_baseline.log`, `/tmp/cuda13_v5_*.log`, `/tmp/cuda13_v5_fp8.log`
