# FP8 Path Unblock Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Switch glades-ml/glades-trainer build to CUDA 13.2 and revalidate the two existing FP8 paths (`--fp8-readout-fwd`, `--fp8-attn`) that were falsified on CUDA 12.0 due to cuBLASLt FP8 algo coverage gaps on Ada (sm_8.9).

**Architecture:** No new FP8 design — the kernels and wrappers exist as dead code with runtime kill-switches in glades-ml's `gpu_blas_fp8.cu`, `gpu_chiron.cu`, and glades-trainer's `chiron_main.cpp`. CUDA 13.2 is well past the 12.3+ unblock threshold cited in `research/ITER62_FP8_READOUT_NULL.md`. This plan executes: (1) toolchain switch + minimal API-drift patches, (2) per-path smoke + Gate-0 + multi-seed validation following the existing design-doc gates verbatim.

**Tech Stack:** CMake 3.18+, CUDA 13.2 / nvcc, cuBLASLt FP8 (E4M3), Ada sm_8.9 (RTX 4080 SUPER), C++98 glades-ml + glades-trainer.

**Spec:** `docs/superpowers/specs/2026-05-21-fp8-path-unblock-design.md`

---

## File Structure

| Path | Type | Responsibility |
|---|---|---|
| `glades-ml/Backend/Machine Learning/Networks/cuda/CMakeLists.txt` | Modify L94 | Drop sm_60/70 from `CUDA_ARCHITECTURES` |
| `glades-trainer/include/Backend/Machine Learning/Networks/cuda/CMakeLists.txt` | Modify L65 | Same |
| `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_blas_fp8.cu` | Patch if needed | API drift fixes for cuBLAS 13 FP8 calls |
| `glades-ml/Backend/Machine Learning/Networks/cuda/gpu_chiron.cu` | Patch if needed | API drift for HELIUM attention shear |
| `glades-ml/research/ITER_FP8_README_REVAL_<verdict>.md` | Create | Per-phase verdict doc |
| `glades-ml/research/ITER_FP8_ATTN_REVAL_<verdict>.md` | Create | Per-phase verdict doc |
| (env) `CUDACXX`, `PATH`, `LD_LIBRARY_PATH` | Export | Direct CMake to CUDA 13.2 toolchain |

No new source files. No flag additions — `--fp8-readout-fwd` and `--fp8-attn` already exist in `chiron_main.cpp` (verified at lines 1364, 1503).

---

## Phase 0 — Build toolchain switch (Tasks 1 + 2)

### Task 1: Update CMAKE_CUDA_ARCHITECTURES (drop sm_60, sm_70)

CUDA 13 dropped support for compute capabilities < 7.5 (Pascal/Volta). The two CMakeLists that hardcode arch list must be updated.

**Files:**
- Modify: `/home/robert/dev/glades-ml/Backend/Machine Learning/Networks/cuda/CMakeLists.txt:94`
- Modify: `/home/robert/dev/glades-trainer/include/Backend/Machine Learning/Networks/cuda/CMakeLists.txt:65`

- [ ] **Step 1: Inspect current arch list in glades-ml**

Run: `grep -n "CUDA_ARCHITECTURES" "/home/robert/dev/glades-ml/Backend/Machine Learning/Networks/cuda/CMakeLists.txt"`
Expected: line 94 contains `set_target_properties(GladesCUDA PROPERTIES CUDA_ARCHITECTURES "60;70;75;80;86;89;90")`

- [ ] **Step 2: Edit glades-ml CMakeLists.txt to drop sm_60/70**

Replace the line:
```cmake
set_target_properties(GladesCUDA PROPERTIES CUDA_ARCHITECTURES "60;70;75;80;86;89;90")
```
with:
```cmake
set_target_properties(GladesCUDA PROPERTIES CUDA_ARCHITECTURES "75;80;86;89;90")
```

- [ ] **Step 3: Edit glades-trainer CMakeLists.txt identically**

Replace the line in `/home/robert/dev/glades-trainer/include/Backend/Machine Learning/Networks/cuda/CMakeLists.txt:65`:
```cmake
set_target_properties(GladesCUDA PROPERTIES CUDA_ARCHITECTURES "60;70;75;80;86;89;90")
```
with:
```cmake
set_target_properties(GladesCUDA PROPERTIES CUDA_ARCHITECTURES "75;80;86;89;90")
```

- [ ] **Step 4: Verify both edits**

Run: `grep -n "CUDA_ARCHITECTURES" "/home/robert/dev/glades-ml/Backend/Machine Learning/Networks/cuda/CMakeLists.txt" "/home/robert/dev/glades-trainer/include/Backend/Machine Learning/Networks/cuda/CMakeLists.txt"`
Expected: both lines now show `"75;80;86;89;90"` and neither contains `60;70`.

### Task 2: Clean stale build dirs and CMakeCache

The existing `build/` directories have CMakeCache.txt entries pinned to CUDA 12.0. CMakeCache is sticky across reconfigures; must be deleted.

**Files:** delete `glades-ml/build/`, `glades-ml/unit-tests/build/`, `glades-trainer/build/` if present.

- [ ] **Step 1: Check what build dirs exist**

Run: `ls -d /home/robert/dev/glades-ml/build /home/robert/dev/glades-ml/unit-tests/build /home/robert/dev/glades-trainer/build 2>&1`
Expected: list of which exist.

- [ ] **Step 2: Remove glades-ml build dir**

Run: `rm -rf /home/robert/dev/glades-ml/build`
Expected: silent success.

- [ ] **Step 3: Remove glades-ml unit-tests build dir**

Run: `rm -rf /home/robert/dev/glades-ml/unit-tests/build`
Expected: silent success.

- [ ] **Step 4: Remove glades-trainer build dir**

Run: `rm -rf /home/robert/dev/glades-trainer/build`
Expected: silent success.

---

## Phase 0 — Build (Tasks 3 + 4)

### Task 3: Rebuild glades-ml with CUDA 13.2

`.configure.sh` calls `mkdir build; cd build; cmake .. $CMAKE_ARGS; make -j$(nproc)`. We override the CUDA toolchain via `CUDACXX` + `PATH`. We do NOT modify `.configure.sh` — the env vars propagate cleanly.

**Files:** none modified; runs build only.

- [ ] **Step 1: Verify CUDA 13.2 nvcc works**

Run: `/usr/local/cuda-13.2/bin/nvcc --version | head -5`
Expected: includes `release 13.2, V13.2.78` (build cuda_13.2.r13.2).

- [ ] **Step 2: Configure + build glades-ml**

Run (single command, all from project root):
```bash
cd /home/robert/dev/glades-ml && \
  CUDACXX=/usr/local/cuda-13.2/bin/nvcc \
  PATH=/usr/local/cuda-13.2/bin:$PATH \
  LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-} \
  sh .configure.sh cuda 2>&1 | tee /tmp/glades_ml_build.log | tail -60
```
Expected: tail shows `make[1]: Leaving directory` or similar success indicator with no `error:` lines. If compile errors appear, see Task 5.

- [ ] **Step 3: Verify libglades.so links libcublasLt.so.13**

Run: `ldd /home/robert/dev/glades-ml/build/libglades.so 2>&1 | grep cublasLt`
Expected: contains `libcublasLt.so.13` (NOT `.so.12`).

- [ ] **Step 4: Install glades-ml**

Run: `cd /home/robert/dev/glades-ml/build && make install 2>&1 | tail -10`
Expected: `Install configuration:` + installed files in `~/.local`, exit 0.

- [ ] **Step 5: Verify installed library**

Run: `ldd ~/.local/lib/libglades.so 2>&1 | grep cublasLt`
Expected: `libcublasLt.so.13`.

### Task 4: Rebuild glades-trainer with CUDA 13.2

**Files:** none modified; runs build only.

- [ ] **Step 1: Build glades-trainer**

Run:
```bash
cd /home/robert/dev/glades-trainer && \
  CUDACXX=/usr/local/cuda-13.2/bin/nvcc \
  PATH=/usr/local/cuda-13.2/bin:$PATH \
  LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-} \
  sh build.sh 2>&1 | tee /tmp/glades_trainer_build.log | tail -60
```
Expected: tail shows `Built: build/glades_pile_train`. No `error:` lines. If compile errors, see Task 5.

- [ ] **Step 2: Verify trainer binary links libcublasLt.so.13**

Run: `ldd /home/robert/dev/glades-trainer/build/glades_pile_train 2>&1 | grep cublasLt`
Expected: `libcublasLt.so.13`.

- [ ] **Step 3: Verify trainer recognizes FP8 flags**

Run: `/home/robert/dev/glades-trainer/build/glades_pile_train --help 2>&1 | grep -E "fp8-(attn|readout)"`
Expected: two lines, one each for `--fp8-attn` and `--fp8-readout-fwd`.

### Task 5: Patch cuBLAS 13 API drift (CONDITIONAL — only if Task 3 or 4 fails)

If Task 3 or 4 fails with errors in `gpu_blas_fp8.cu` or related cuBLASLt code, this task patches them in place. Skip if both builds succeed.

**Files:**
- Patch as needed: `/home/robert/dev/glades-ml/Backend/Machine Learning/Networks/cuda/gpu_blas_fp8.cu`
- Patch as needed: `/home/robert/dev/glades-ml/Backend/Machine Learning/Networks/cuda/gpu_chiron.cu`
- Patch budget: ≤50 LOC total across ≤3 files. If exceeded, STOP and escalate.

- [ ] **Step 1: Extract compile errors**

Run: `grep -E "error:|undefined reference|undeclared" /tmp/glades_ml_build.log /tmp/glades_trainer_build.log 2>/dev/null | head -30`
Expected: list of distinct errors (may be empty if Task 3+4 succeeded — skip rest of this task).

- [ ] **Step 2: Identify root cause**

Common CUDA 12→13 break surfaces (categorize errors):
- (a) Removed/renamed `cublasLt*` enums: e.g., `CUBLASLT_MATMUL_DESC_SCALE_TYPE`, `CUBLASLT_MATMUL_DESC_FAST_ACCUM`.
- (b) Removed/renamed `cudaDataType_t` values: `CUDA_R_8F_E4M3`, `CUDA_R_8F_E5M2` are now `CUDA_R_8F_E4M3` (verify in `<cuda_fp8.h>`).
- (c) `cuda_fp8.h` constructor changes for `__nv_fp8_e4m3`.
- (d) `CUBLAS_COMPUTE_32F` semantics changes (now sometimes requires explicit precision attribute).

Record category for each error.

- [ ] **Step 3: Apply minimal fix per category**

For each error, edit the relevant file (gpu_blas_fp8.cu or gpu_chiron.cu) to use the CUDA 13 equivalent. **Do NOT** rewrite the kernel logic — only update the API call signatures/enums. Show actual diff for each edit (no placeholders).

- [ ] **Step 4: Re-run build**

Run: Task 3 Step 2 again.
Expected: clean build.

- [ ] **Step 5: Commit the patch**

Run:
```bash
cd /home/robert/dev/glades-ml && git add -- "Backend/Machine Learning/Networks/cuda/gpu_blas_fp8.cu" "Backend/Machine Learning/Networks/cuda/gpu_chiron.cu" "Backend/Machine Learning/Networks/cuda/CMakeLists.txt"
cd /home/robert/dev/glades-ml && git commit -m "$(cat <<'EOF'
build: switch CUDA toolchain to 13.2 + drop sm_60/70 from arch list

CUDA 13 dropped Pascal/Volta. Trim CMAKE_CUDA_ARCHITECTURES to
75;80;86;89;90. Patch cuBLASLt 13 API drift in gpu_blas_fp8.cu and
gpu_chiron.cu (see commit body for per-symbol changes).

Unblocks --fp8-readout-fwd and --fp8-attn paths per
research/ITER62_FP8_READOUT_NULL.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```
Expected: commit succeeds.

**Note**: per session policy, do not commit unless the user has authorized. The user said "Proceed" — interpret as authorizing build-stage commits if needed; ask before pushing.

---

## Phase A — FP8 readout smoke (Task 6)

### Task 6: Run --fp8-readout-fwd 200-step bench

**Files:** none modified; runs bench only. Output: `/tmp/fp8_readout_smoke.log`.

- [ ] **Step 1: Capture baseline iter 116 ship metrics for the same 200 steps**

Run:
```bash
cd /home/robert/dev/glades-trainer && \
  CUDACXX=/usr/local/cuda-13.2/bin/nvcc \
  PATH=/usr/local/cuda-13.2/bin:$PATH \
  LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-} \
  sh run.sh flagship --max-steps 200 --seed 1337 --save /tmp/fp8_baseline 2>&1 | tee /tmp/fp8_baseline_smoke.log | tail -40
```
Expected: completes 200 steps; emits "step 200" line with tok/s and a "val_NLL" line.

- [ ] **Step 2: Extract baseline metrics**

Run:
```bash
grep -E "step\s+200\s|val_NLL|val_nll|tok/s" /tmp/fp8_baseline_smoke.log | tail -10
```
Expected: tok/s value + val NLL value (record both for comparison).

- [ ] **Step 3: Run --fp8-readout-fwd treatment**

Run:
```bash
cd /home/robert/dev/glades-trainer && \
  CUDACXX=/usr/local/cuda-13.2/bin/nvcc \
  PATH=/usr/local/cuda-13.2/bin:$PATH \
  LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-} \
  sh run.sh flagship --max-steps 200 --seed 1337 --fp8-readout-fwd --save /tmp/fp8_readout 2>&1 | tee /tmp/fp8_readout_smoke.log | tail -60
```
Expected: completes 200 steps.

- [ ] **Step 4: Check for kill-switch trip**

Run:
```bash
grep -E "fp8-readout-fwd.*disabl|cuBLASLt FP8 readout GEMM rejected" /tmp/fp8_readout_smoke.log
```
Expected: NO output (zero matches) — meaning the kill-switch did NOT fire.

If output is non-empty (kill-switch tripped): genuine Ada FP8 algo gap persists at CUDA 13.2 for the readout shape. Document in Step 6 and STOP this phase. Phase B may still be worth attempting (different shape).

- [ ] **Step 5: Verify FP8 GEMM actually executed**

Run:
```bash
grep -E "HELIUM ACTIVE|fp8-readout-fwd.*iter 62|FP8.*readout|E4M3" /tmp/fp8_readout_smoke.log | head -10
```
Expected: contains the trainer's init log line `[fp8-readout-fwd] iter 62: forward F1 readout GEMM routes through cuBLASLt FP8` AND no later disable warning.

- [ ] **Step 6: Compare to baseline + record verdict**

Compute deltas:
- `Δtok/s = (treatment_tok/s - baseline_tok/s) / baseline_tok/s × 100%`
- `ΔNLL = treatment_val_NLL - baseline_val_NLL`

Verdict matrix (per iter 62 design doc Gate-0):
- PASS-strong: `Δtok/s ≥ +3%` AND `ΔNLL ≤ +0.02 nat` → proceed to Task 8 (multi-seed)
- PASS-below-bar: `Δtok/s ∈ [+1%, +3%)` AND `ΔNLL ≤ +0.02 nat` → still proceed to Task 8 (silent-accrual candidate per iter 60)
- NULL: `|Δtok/s| < +1%` → write NULL doc; do not proceed to multi-seed; move to Task 7 (Phase B)
- FAIL-NLL: `ΔNLL > +0.05 nat` → revert; document; do not proceed
- FAIL-kill-switch (from Step 4): document; do not proceed for readout

Write verdict to `/home/robert/dev/glades-ml/research/ITER_FP8_README_REVAL_<verdict>.md` where `<verdict>` is `PASS` / `NULL` / `FAIL` per above. Body should include: tok/s baseline + treatment + delta, val NLL baseline + treatment + delta, kill-switch status, nsys (if collected) FP8 kernel names + percent wall.

---

## Phase B — FP8 attention smoke (Task 7)

### Task 7: Run --fp8-attn 200-step bench

`--fp8-attn` requires dropping `--bf16-attn` from the stack (incompatible per `chiron_main.cpp:11284`). The baseline for this phase is the flagship stack with `--no-bf16-attn`, NOT the default flagship — apples-to-apples.

**Files:** none modified; runs bench only.

- [ ] **Step 1: Determine the override mechanism for --bf16-attn**

The flagship `STACK` (line 237 of `run.sh`) hardcodes `--bf16-attn`. Check whether trainer accepts a CLI `--no-bf16-attn` override after the stack args. Run:
```bash
/home/robert/dev/glades-trainer/build/glades_pile_train --help 2>&1 | grep -E "no-bf16-attn|bf16-attn"
```
Expected: shows if `--no-bf16-attn` exists. If yes, append to extra args. If no, this task must edit run.sh STACK locally (a temporary edit, reverted after Phase B).

- [ ] **Step 2 (case: --no-bf16-attn exists): Run baseline (flagship + --no-bf16-attn, no FP8)**

Run:
```bash
cd /home/robert/dev/glades-trainer && \
  CUDACXX=/usr/local/cuda-13.2/bin/nvcc \
  PATH=/usr/local/cuda-13.2/bin:$PATH \
  LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-} \
  sh run.sh flagship --max-steps 200 --seed 1337 --no-bf16-attn --save /tmp/fp8_attn_baseline 2>&1 | tee /tmp/fp8_attn_baseline_smoke.log | tail -40
```
Expected: 200 steps complete.

- [ ] **Step 2 (alternative case: must edit run.sh)**: If `--no-bf16-attn` is not a CLI flag, edit `/home/robert/dev/glades-trainer/run.sh` line 237 to remove ` --bf16-attn `, run baseline + treatment, then revert the edit.

- [ ] **Step 3: Run --fp8-attn treatment**

Run:
```bash
cd /home/robert/dev/glades-trainer && \
  CUDACXX=/usr/local/cuda-13.2/bin/nvcc \
  PATH=/usr/local/cuda-13.2/bin:$PATH \
  LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-} \
  sh run.sh flagship --max-steps 200 --seed 1337 --no-bf16-attn --fp8-attn --save /tmp/fp8_attn 2>&1 | tee /tmp/fp8_attn_smoke.log | tail -60
```

- [ ] **Step 4: Check for kill-switch trip**

Run:
```bash
grep -E "fp8-attn.*Disabl|cuBLASLt FP8 matmul rejected" /tmp/fp8_attn_smoke.log
```
Expected: NO output.

- [ ] **Step 5: Verify FP8 attention activated**

Run:
```bash
grep -E "HELIUM ACTIVE" /tmp/fp8_attn_smoke.log
```
Expected: `[fp8-attn] HELIUM ACTIVE — Q/K/V/O projection GEMMs route through cuBLASLt FP8 (E4M3, per-tensor amax-derived scales recomputed each call).`

- [ ] **Step 6: Compute deltas and verdict**

Same procedure as Task 6 Step 6, but reference HELIUM iter 55 gates. Note: HELIUM design doc anticipated low gain on attention shapes (~0.07 TFLOP per GEMM) — the bar here is lower than readout's. PASS at +1% would still be a meaningful result.

Write verdict to `/home/robert/dev/glades-ml/research/ITER_FP8_ATTN_REVAL_<verdict>.md`.

---

## Phase C — Multi-seed validation (Task 8)

### Task 8: Multi-seed n=3 strict-bar bench

ONLY runs if Task 6 OR Task 7 returned a PASS verdict (strong or below-bar). Per iter 91 lesson: single-seed PASS at +3% must be backed by multi-seed strict (±0.02 nat) before any production retrain commitment.

**Files:** none modified; runs bench only.

- [ ] **Step 1: Determine which path passed**

If Task 6 PASS: run multi-seed for readout. If Task 7 PASS: run multi-seed for attn. If both PASS: run multi-seed for both (sequentially, attn first since smaller per-iter signal is more fragile).

- [ ] **Step 2: Run treatment at seed 1338 (200 steps)**

For readout path (example; substitute `--fp8-attn` + `--no-bf16-attn` for attention):
```bash
cd /home/robert/dev/glades-trainer && \
  CUDACXX=/usr/local/cuda-13.2/bin/nvcc \
  PATH=/usr/local/cuda-13.2/bin:$PATH \
  LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-} \
  sh run.sh flagship --max-steps 200 --seed 1338 --fp8-readout-fwd --save /tmp/fp8_readout_s1338 2>&1 | tee /tmp/fp8_readout_s1338.log | tail -10
```

- [ ] **Step 3: Run baseline at seed 1338**

```bash
cd /home/robert/dev/glades-trainer && \
  CUDACXX=/usr/local/cuda-13.2/bin/nvcc \
  PATH=/usr/local/cuda-13.2/bin:$PATH \
  LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64:${LD_LIBRARY_PATH:-} \
  sh run.sh flagship --max-steps 200 --seed 1338 --save /tmp/fp8_baseline_s1338 2>&1 | tee /tmp/fp8_baseline_s1338.log | tail -10
```

- [ ] **Step 4: Run treatment + baseline at seed 1339**

Same as Steps 2 + 3 with `--seed 1339`.

- [ ] **Step 5: Compute multi-seed mean and std**

For each of (tok/s, val NLL), compute the per-seed delta then mean and std across seeds 1337/1338/1339.

Strict PASS gates per iter 91:
- `mean(Δtok/s) ≥ +3%`
- `mean(ΔNLL) ≤ +0.02 nat` (strict single-seed bound)
- `std(ΔNLL) ≤ 0.10 nat` (n=3 stability)
- Per-seed: each `|ΔNLL| ≤ 0.05` (multi-seed bound)

- [ ] **Step 6: Write multi-seed verdict doc**

Write to `/home/robert/dev/glades-ml/research/ITER_FP8_<path>_MULTISEED_<verdict>.md` where `<path>` is `READOUT` or `ATTN` and `<verdict>` is `PASS`/`FAIL`. Include the per-seed table, mean, std, deltas.

If PASS: note that Phase D (production retrain) is now eligible pending user authorization.

---

## Phase D — Production retrain (DEFERRED)

Production retrain (30k-step Phase 2 per iter 94 protocol) is NOT in scope of this plan. After Task 8 PASS, surface to the user with cost estimate and bench data; await explicit go/no-go.

---

## Final cleanup tasks

### Task 9: Update auto-memory + CLAUDE.md (conditional on any PASS)

**Files:** Create/update memory; conditionally update `/home/robert/dev/glades-ml/CLAUDE.md`.

- [ ] **Step 1: Write auto-memory entry**

Create `/home/robert/.claude/projects/-home-robert-dev-glades-ml/memory/fp8_revalidation_2026_05_21.md` with topic frontmatter and short body summarizing the result.

- [ ] **Step 2: Index the memory in MEMORY.md**

Add a one-line entry to `/home/robert/.claude/projects/-home-robert-dev-glades-ml/memory/MEMORY.md` (under ~150 chars).

- [ ] **Step 3 (PASS only): Note in CLAUDE.md flagship section**

If multi-seed PASS, add a short note to CLAUDE.md "Current Production Flagship" section noting that `--fp8-readout-fwd` and/or `--fp8-attn` are now production-ready opt-in flags under CUDA 13.2 toolchain. Do NOT change the default flagship recipe until Phase D ships a new checkpoint.

### Task 10: Commit cleanup

- [ ] **Step 1: Stage research docs**

Run:
```bash
cd /home/robert/dev/glades-ml && git status --short
```
Expected: lists the new research/ITER_FP8_* docs, possibly CLAUDE.md changes, plus the plan + spec under docs/superpowers/.

- [ ] **Step 2: Ask user before committing**

Per session policy, ask the user whether to commit the research + spec/plan docs together, or leave them uncommitted. Do NOT auto-commit.

---

## Risks & blockers

| Risk | Trigger | Action |
|---|---|---|
| Compile errors > 50 LOC patch budget | Task 5 grows beyond budget | STOP; surface to user; this becomes a separate cuBLAS 13 migration spec |
| Kill-switch trips at CUDA 13.2 too | Task 6 Step 4 or Task 7 Step 4 non-empty | Document as terminal Ada FP8 algo gap; FP8 paths confirmed null. Negative result is still informative. |
| NLL drift > +0.05 nat | Task 6 Step 6 or Task 7 Step 6 | Revert flag; FAIL doc; do not proceed |
| BF16 baseline NLL changes after CUDA 13 rebuild | Comparing /tmp/fp8_baseline_smoke.log val NLL to iter 116 published 200-step expectation | Investigate (could be cuBLAS BF16 algo drift). May invalidate apples-to-apples. |

---

## Self-review notes

- **Spec coverage**: Phases 0/A/B/C from spec map to Tasks 1-8; Phase D is correctly deferred.
- **Placeholders**: None — every step has command + expected output.
- **Type consistency**: Flag names verified against `chiron_main.cpp:1364,1503` (`--fp8-readout-fwd`, `--fp8-attn`); arch list `75;80;86;89;90` consistent across Tasks 1 + 5.
- **Conditional tasks**: Task 5 (API drift) and Task 8 (multi-seed) are explicitly conditional with skip criteria stated.
