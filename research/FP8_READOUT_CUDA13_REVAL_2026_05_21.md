# FP8 Readout Revalidation on CUDA 13.2 — 2026-05-21

**Status:** PASS-below-bar (silent-accrual candidate) on iter 116 ship flagship.
**Date:** 2026-05-21
**Branch:** pants (glades-ml) + main (glades-trainer)
**Baseline:** CHIRON 1B iter 116 ship (chiron_1B_T16384_iter116, 28,257 tok/s @ T=16384 on CUDA 12.0).

---

## TL;DR

The `--fp8-readout-fwd` flag (iter 62 implementation) was previously **paradigm-falsified** at iter 62 because CUDA 12.0 cuBLASLt returned `CUBLAS_STATUS_NOT_SUPPORTED` at the readout shape (T=8192, V=32000, K=2048) on Ada sm_8.9 — no FP8 algo coverage. The iter 62 NULL doc explicitly noted: *"If the cuBLAS toolkit upgrades to 12.3+ in a future system-software refresh, the FP8 wrapper is in tree and the flag `--fp8-readout-fwd` would light up automatically — Gate-0 retesting is then a single bench command."*

This session installed CUDA 13.2 (`/usr/local/cuda-13.2`, `nvcc V13.2.78`) and:

1. **Re-built glades-ml + glades-trainer under CUDA 13.2** — clean compile, no API drift in `gpu_blas_fp8.cu`. Trimmed `CMAKE_CUDA_ARCHITECTURES` to `75;80;86;89;90` (CUDA 13 dropped Pascal/Volta).
2. **Validated `--fp8-readout-fwd`** at n=3 multi-seed (1337/1338/1339) against the iter 116 ship baseline at T=16384.

**Headline:**
- **No runtime kill-switch trip.** cuBLASLt 13.2 accepts the FP8 E4M3 readout GEMM at shape (T=16384, V=32000, K=2048) on Ada sm_8.9. The iter 62 toolkit gap is resolved.
- **+1.38% mean wall** (std 0.015%) — **below the +3% iter 60 strong-bar** but well above +1% noise.
- **+0.0058 nat mean ΔNLL** (std 0.0010, max 0.0069) — **within strict ±0.02 nat per iter 91 multi-seed methodology**.
- **Class**: silent-accrual mechanism (similar to iter 97 +1.54%, iter 99 +1.40%).

The companion `--fp8-attn` (HELIUM iter 55) path was also smoke-tested on CUDA 13.2 — **NULL at +0.0% wall, NLL bit-identical**. Confirms iter 55 finding holds: attention shapes (~0.07 TFLOP per QKV GEMM) too small for FP8 to amortize amax-cast overhead, regardless of toolkit. Recorded in §6.

---

## 1. Setup

**Hardware:** RTX 4080 SUPER (SM 8.9, 16 GB VRAM, 80 SMs)
**Driver:** 580.126.09 (CUDA 13.0 runtime)
**Toolkit:** CUDA 13.2.78 (`/usr/local/cuda-13.2/bin/nvcc`) — replaces CUDA 12.0.140 (`/usr/bin/nvcc`, still installed)
**Library links:** `libcublasLt.so.13`, `libcublas.so.13`, `libcudart.so.13` (verified via `ldd`)

**Build changes:**
- `glades-ml/Backend/Machine Learning/Networks/cuda/CMakeLists.txt:94` — `CUDA_ARCHITECTURES "60;70;75;80;86;89;90"` → `"75;80;86;89;90"`
- `glades-trainer/include/Backend/Machine Learning/Networks/cuda/CMakeLists.txt:65` — same
- Required because CUDA 13 dropped sm_60 (Pascal) and sm_70 (Volta) support.
- **Zero source-file edits needed** — `gpu_blas_fp8.cu` and `gpu_chiron.cu` compile clean under cuBLASLt 13. No API drift.

**Baseline stack** (matches `run.sh flagship`):
```
--int8-adam --bf16-grads --bf16-weights --bf16-attn
--no-fuse-attn --fuse-attn-reln
--scfa --scfa-bf16-inner --scfa-bf16-outer --scfa-fuse-streams --scfa-reln-opt
--bf16-logits --bf16-logits-storage
--max-steps 200 --warmup 500 --lr 1e-4 --grad-clip 0.5
--log-every 20 --val-every 100 --val-batches 4
```

**Treatment**: baseline + `--fp8-readout-fwd`.

**Multi-seed**: seeds 1337, 1338, 1339 per iter 91 methodology.

---

## 2. Multi-seed results — `--fp8-readout-fwd`

### 2.1 Per-seed table

| seed | baseline NLL@200 | treatment NLL@200 | ΔNLL | baseline tok/s | treatment tok/s | Δtok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1337 | 8.9582 | 8.9635 | +0.0053 | 24,042 | 24,379 | +1.40% |
| 1338 | 8.9603 | 8.9672 | +0.0069 | 24,047 | 24,377 | +1.37% |
| 1339 | 8.9616 | 8.9667 | +0.0051 | 24,047 | 24,378 | +1.38% |

### 2.2 Aggregated stats (n=3)

| stat | value |
|---|---:|
| mean Δtok/s | **+1.38%** |
| std Δtok/s | 0.015% |
| mean ΔNLL @ step 200 | **+0.0058 nat** |
| std ΔNLL @ step 200 | 0.0010 nat |
| max per-seed \|ΔNLL\| | 0.0069 (s1338) |

### 2.3 iter 62 design Gate-0 evaluation

Per `PARADIGM_FP8_READOUT_DESIGN.md` §5:

| criterion | bar | result | verdict |
|---|---|---:|---|
| C0a tok/s win | +3% to +6% strong; <+2% revert | +1.38% | below strong, above revert |
| C0b val NLL @ step 200 | ≤ +0.020 nat strict | +0.0058 | **PASS strict** |
| C0c val NLL @ step 1000 | ≤ +0.020 nat (slow-drift trap) | n/a (200-step bench) | deferred |
| C0d VRAM | ≤ baseline | unchanged (FP8 writes BF16 logits directly) | **PASS** |
| C0e per-row \|Σp − 1\| | < 1e-5 | not measured | n/a (probs stay BF16) |
| C0f argmax top-1 | within ±0.5% of baseline | 0.0680/0.0681 = within 0.001 abs | **PASS** |

### 2.4 iter 91 multi-seed strict-bar evaluation

Per `iter91_1k_multiseed_strict_pass.md`:

| criterion | bar | result | verdict |
|---|---|---:|---|
| mean ΔNLL within strict | ≤ ±0.020 nat | +0.0058 | **PASS** |
| mean Δtok/s ≥ +3% | iter 60 ship bar | +1.38% | below strong bar |
| std ΔNLL | reasonable spread | 0.0010 (4.3% of strict bound) | **PASS** |
| per-seed \|ΔNLL\| within multi-seed | ≤ ±0.050 nat | max 0.0069 | **PASS** |
| kill-switch trip on any seed | 0 trips required | 0 trips | **PASS** |

### 2.5 Verdict

**PASS-below-bar (silent-accrual candidate)**.

This is the iter 97 / iter 99 / iter 103 mechanism class: stable, parity-clean wall gain below the +3% strong-bar but above the +1% noise floor. The mechanism class has 11 prior PASS realizations in the iter 116 ship (see CLAUDE.md). FP8 readout adds a 12th candidate sub-mechanism: "dtype reduction on the largest readout GEMM via cuBLASLt FP8".

Math is **not bit-identical** (E4M3 quantization is lossy by design), but mean NLL drift is sub-strict. The +0.0058 mean is small enough to be safely absorbed by the production training trajectory.

---

## 3. FP8 activation confirmation

At every seed, the trainer init log includes:

```
[fp8-readout-fwd] iter 62: forward F1 readout GEMM routes through cuBLASLt FP8 (E4M3)
at shape (T=16384, V=32000, m=2048) ≈ 1 TFLOP. Backward + softmax stay BF16
(FP8 path writes BF16 logits directly). Per-tensor amax recomputed each step from
q_L_bf + E_bf_cache.
```

And **zero** later instances of the kill-switch warning:
```
[fp8-readout-fwd] cuBLASLt FP8 readout GEMM rejected at shape... Disabling FP8 readout
```

(Verified by `grep -E "fp8.*disabl|cuBLASLt FP8.*reject" /tmp/fp8_readout_s*.log` → empty.)

This confirms the cuBLASLt 13.2 FP8 algo table on Ada sm_8.9 now covers the (T=16384, V=32000, m=2048) readout shape — the precise gap that falsified iter 62 on CUDA 12.0.

---

## 4. Why the gain is modest

The iter 62 design predicted +3-6% wall at iso-NLL based on the 2× FP8-vs-BF16 theoretical peak on Ada. Actual is +1.38%. Three plausible reasons (not falsified here, recorded for future investigation):

1. **Readout is ~18% of step time at T=16384 m=2048.** A 2× speedup on that bucket caps total gain at ~9%. The realized +1.38% is ~15% of that envelope — consistent with the FP8 GEMM itself running at maybe 1.15× BF16 speed at this shape, not 2×.
2. **amax recompute per step** for `q_L_bf` (T·m = 33.5M entries) and `E_bf_cache` (V·m = 65.5M entries). Even at peak memory bandwidth, that's ~0.1 ms/step overhead, eating ~0.3-0.5% of the gain.
3. **cuBLASLt 13.2 FP8 algo for this shape** may not be peak-tuned. The shape (M=16384, N=32000, K=2048) is the iter 62 readout shape; cuBLASLt accepts it (no kill-switch) but the dispatched algo may not deliver peak FP8 throughput. nsys profiling deferred.

iter 62 risk R6 ("Slice fragmentation caps win") materialized: realized gain at the modest end of the design envelope.

---

## 5. CUDA 13.2 cuBLAS BF16 regression — CONFIRMED via toolchain A/B

**Definitive finding** (added 2026-05-21 post initial draft). Same source, same flags, same hardware, same data — only the toolkit differs:

| toolchain | tok/s @ step 181 | wall @ step 181 | step 1 loss | step 181 loss |
|---|---:|---:|---:|---:|
| CUDA 12.0 (system /usr install) | **28,345** | 104.7s | 10.5777 | 9.0859 |
| CUDA 13.2 (/usr/local/cuda-13.2) | 24,041 | 123.5s | 10.5778 | 9.0860 |
| Δ | **-15.18%** (4,304 tok/s) | +17.9% wall | bit-identical | bit-identical |

Loss values are **bit-identical to 4 decimal places at every logged step**, confirming the regression is purely runtime perf, not numerics. The CUDA 12.0 result (28,345) matches the published iter 116 ship 28,257 within +0.3% (noise) — the production flagship reproduces exactly on CUDA 12.0.

**Repro of the A/B (verified clean):**

```bash
# CUDA 12.0 baseline
rm -rf /home/robert/dev/glades-ml/build && mkdir -p /home/robert/dev/glades-ml/build
cd /home/robert/dev/glades-ml/build && LD_LIBRARY_PATH= PATH=/usr/bin:$PATH \
  CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 \
  cmake .. -DGLADES_ENABLE_CUDA=ON \
    -DCMAKE_CUDA_COMPILER=/usr/bin/nvcc \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11 \
    -DCMAKE_C_COMPILER=/usr/bin/gcc-11 -DCMAKE_CXX_COMPILER=/usr/bin/g++-11
LD_LIBRARY_PATH= PATH=/usr/bin:$PATH make -j$(nproc) && make install
# (similar manual cmake for glades-trainer/build)
# Result: libcublas.so.12 linked, 28,345 tok/s.

# CUDA 13.2 (the FP8-capable toolchain)
rm -rf build && sh .configure.sh cuda  # picks up /usr/local/cuda → cuda-13.2 by default
# Result: libcublas.so.13 linked, 24,041 tok/s. Same source, same flags.
```

**Root cause** (high confidence): cuBLAS 13.x BF16 GEMM algo dispatch on Ada (sm_8.9) at the production shapes (T=16384, m=2048, k=1024, V=32000). Loss is bit-identical so no numerical change is involved; only wall regresses. cuBLASLt's heuristic algo selection appears to pick a worse algo for at least one of the high-frequency GEMMs in the SCFA + readout chain on CUDA 13.

This is **not the FP8 question** — both baseline and treatment in §2 ran under CUDA 13.2, so the +1.38% delta is apples-to-apples valid. But the absolute regression dominates the FP8 benefit in production, see §5.1.

### 5.1 Net production tok/s analysis (the bottom line)

| stack | toolchain | net tok/s | vs CUDA 12.0 prod baseline |
|---|---|---:|---:|
| iter 116 ship (no FP8) | **CUDA 12.0** | **28,345** | (production baseline) |
| iter 116 ship (no FP8) | CUDA 13.2 | 24,041 | **-15.18%** (loss to switch) |
| iter 116 ship + `--fp8-readout-fwd` | CUDA 13.2 | 24,379 | **-14.00%** (loss to switch + add FP8) |
| iter 116 ship + `--fp8-attn` (HELIUM) | CUDA 13.2 | 24,042 | -15.18% (no FP8 attn benefit) |

**FP8 readout adds +1.38% on CUDA 13.2 but the toolchain switch from CUDA 12.0 costs -15.18%. Net: -14.00% to deploy FP8 readout in production today.** The FP8 path is "technically unblocked" but **not a net production win** under the current cuBLAS 13.x algo state.

### 5.2 Mitigation paths

1. **Stay on CUDA 12.0** for production training (current default `/usr/bin/nvcc` already CUDA 12.0.140). FP8 paths stay falsified per iter 62. Production tok/s unchanged at 28,345.
2. **Try intermediate CUDA versions (12.3-12.8)** — these may have FP8 algo coverage (per iter 62 prediction) without the cuBLAS 13.x BF16 regression. Test cost: ~20 min per version.
3. **Override cuBLASLt algo selection** on CUDA 13.2 for the regressed GEMM shapes via `cublasLtMatmulPreferenceSetAttribute` algo tuning. Requires nsys profiling to identify which kernels regressed and which algo to force. Substantial engineering work.
4. **Wait for NVIDIA cuBLAS 13.x update** that restores Ada BF16 algo perf. No timeline.

Recommendation: **option 1 (stay on CUDA 12.0)** unless FP8 is critical. Note option 2 as a possible path forward if FP8 becomes priority.

---

## 6. Phase B — `--fp8-attn` (HELIUM iter 55) NULL on CUDA 13.2

Sanity-checked the HELIUM `--fp8-attn` path on CUDA 13.2 with the same flagship stack minus `--bf16-attn` (per `chiron_main.cpp:11284` incompatibility), single seed 1337, 200 steps.

| metric | baseline (no --bf16-attn, no FP8) | treatment (--fp8-attn) | delta |
|---|---:|---:|---:|
| val NLL @ step 200 | 8.9582 | 8.9582 | 0.0000 (bit-identical) |
| tok/s @ step 181 | 24,046 | 24,042 | -0.02% (noise) |

**HELIUM ACTIVE init log** confirmed (no kill-switch trip on CUDA 13.2). FP8 dispatch is live for Q/K/V/O projection GEMMs but contributes zero wall gain at the iter 116 shape. This **re-confirms** the iter 55 finding: attention shape per-call cost (~0.07 TFLOP) is below the FP8-amortization threshold on Ada. Toolkit upgrade does not change this.

Per the iter 62 design doc's HELIUM precedent: *"HELIUM (--fp8-attn, iter 55) returned ~0% because attention shapes (k=512, m=2048, ~0.07 TFLOP) are too small for FP8 to amortize amax-cast overhead."* This holds at T=16384 / m=2048 / dH=256 too.

`--fp8-attn` stays in tree as latent code; no production utility on CHIRON 1B. **The full HELIUM paradigm** (FA-3 + FP8 + fusion) per `PARADIGM_SHIFT_50_CANDIDATE_A_HELIUM.md` is **NOT realized** by the in-tree `--fp8-attn` flag — that's only the partial FP8 dispatch slice. Full HELIUM remains ~2500 LOC of unimplemented work (FA-3 forward + backward, stochastic-rounding casts, fused QKV+attn+output kernels). Out of scope for this session.

---

## 7. What stays in tree

Unchanged from iter 62 closure:

- `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_blas_fp8.cu` + `.h` — cuBLASLt FP8 wrappers. Now ACTIVELY EXECUTING at production scale via `--fp8-readout-fwd`.
- `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_chiron.cu` — `chiron_attention_shear_fp8w_tiled`. Active via `--fp8-attn` but NULL contribution.
- `glades-trainer/trainer/chiron_main.cpp` — `--fp8-readout-fwd`, `--fp8-attn` flag wiring + runtime kill-switches (kill-switches latent, never trip on CUDA 13.2).
- iter 116 ship default behavior **unchanged**: neither flag is default-on; only opt-in.

---

## 8. What changes (this session)

- **Build toolchain**: CUDA 13.2 used for re-build (default `/usr/bin/nvcc` is still CUDA 12.0; symlink `/usr/local/cuda` → `cuda-13.2`).
- **CMake**: `CUDA_ARCHITECTURES` trimmed to `75;80;86;89;90` (2 files).
- **New evidence file**: this document.
- **Spec + plan files** under `docs/superpowers/specs/2026-05-21-fp8-path-unblock-design.md` and `docs/superpowers/plans/2026-05-21-fp8-path-unblock.md`.

No production-default flags flipped. No flagship recipe changes.

---

## 9. Recommendations (UPDATED after §5 CUDA 13.2 cuBLAS BF16 regression confirmed)

1. **Stay on CUDA 12.0 toolchain for production CHIRON 1B training.** The iter 116 ship 28,257 tok/s reproduces exactly under CUDA 12.0 (28,345 tok/s, +0.3% noise). CUDA 13.2 costs -15.18% wall on the same code. **Do not switch the default toolchain.**
2. **Do NOT deploy `--fp8-readout-fwd` to production today.** Even though it adds +1.38% wall on CUDA 13.2, the toolchain switch required to enable it costs -15.18%. **Net: -14.00% tok/s vs the CUDA 12.0 + no-FP8 production baseline.** Strict loss.
3. **Investigation path for future FP8 production deployment** (if FP8 becomes priority):
   - (a) Try CUDA 12.3 / 12.6 / 12.8 — may have FP8 coverage (per iter 62 prediction) without the 13.x BF16 algo regression. ~20 min/version to test.
   - (b) cuBLASLt algo override for the regressed BF16 GEMM(s) on CUDA 13.2 via `cublasLtMatmulPreferenceSetAttribute`. Requires nsys profiling. Substantial work.
   - (c) Wait for NVIDIA cuBLAS 13.x update restoring Ada BF16 algo perf.
4. **`--fp8-readout-fwd` remains a valid silent-accrual MECHANISM** at the +1.38% n=3 multi-seed strict NLL parity level, ready to deploy if (and only if) the toolchain regression is resolved. Mechanism code is correct.
5. **Do NOT pursue HELIUM full paradigm** without a separate spec; partial `--fp8-attn` is NULL even on CUDA 13.2 and full HELIUM (FA-3 etc.) is out of scope for CHIRON 1B.
6. **Build system left on CUDA 12.0** as of session end (2026-05-21). To re-enable CUDA 13.2 for FP8 work: see §5 repro block.

---

## 10. Files / commands

- Spec: `docs/superpowers/specs/2026-05-21-fp8-path-unblock-design.md`
- Plan: `docs/superpowers/plans/2026-05-21-fp8-path-unblock.md`
- Per-seed bench logs: `/tmp/fp8_baseline_smoke.log`, `/tmp/fp8_readout_smoke.log`, `/tmp/fp8_baseline_s133[89].log`, `/tmp/fp8_readout_s133[89].log`, `/tmp/fp8_attn_baseline_smoke.log`, `/tmp/fp8_attn_smoke.log`
- Reproduce treatment (single-seed):
  ```bash
  cd ~/dev/glades-trainer && LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64:$LD_LIBRARY_PATH \
    ./build/glades_chiron_train --pretokenized --data-dir pretok-data/ \
    --seq-len 16384 --m 2048 --layers 24 --heads 16 --dhead 256 --vocab 32000 \
    --int8-adam --bf16-grads --bf16-weights --bf16-attn \
    --no-fuse-attn --fuse-attn-reln \
    --scfa --scfa-bf16-inner --scfa-bf16-outer --scfa-fuse-streams --scfa-reln-opt \
    --bf16-logits --bf16-logits-storage \
    --fp8-readout-fwd \
    --max-steps 200 --warmup 500 --lr 1e-4 --grad-clip 0.5 --seed 1337 \
    --log-every 20 --val-every 100 --val-batches 4 --save /tmp/fp8_readout_ckpt
  ```
