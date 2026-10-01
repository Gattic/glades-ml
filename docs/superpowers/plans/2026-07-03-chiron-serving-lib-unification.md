# CHIRON Serving Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move CHIRON checkpoint-format knowledge, serving feature-resolution, and the model-level eval forward from glades-trainer into glades-ml as the single source of truth, shrink `chiron_infer` to a thin CLI, and prove no regression via bit-identical logits + TF-metric reproduction.

**Architecture:** Two new lib modules under `Backend/Machine Learning/Networks/` — `chiron_checkpoint.{h,cpp}` (flag-bit registry, block codecs, serving reader, model-section writer helpers) and `chiron_serving.{h,cpp}` (resolve decision table + kernel-sequence-preserving eval forward). glades-ml installs its ML header tree; the trainer drops its vendored header copy, and both trainer binaries consume the lib modules.

**Tech Stack:** C++98 (lib), C++11 (trainer), CUDA via existing `glades::gpu` primitives, CMake.

**Spec:** `docs/superpowers/specs/2026-07-03-chiron-serving-lib-unification-design.md`

## Global Constraints

- Lib code (glades-ml) is **C++98**: no `auto`, lambdas, range-for, `nullptr` (use `0`), no `<random>`. Trainer code is C++11.
- **Kernel call sequence in `chiron_eval_forward` must be preserved exactly** from `forwardInfer`/`scfa_shear_infer` (same primitives, same order, same arguments) — this is what makes the bit-parity gate achievable.
- **No checkpoint format changes.** Byte layout, section order, version rules, and the rot_phi EOF-tail convention are ported verbatim.
- **Default behavior preserved**: same stdout messages (they are operator documentation), same exit codes (2=initDevice, 3=vocab, 4=load, 5=scratch/alloc, 6=SCFA-init, 7=interlock).
- One intentional behavior change only (spec §3.2): fuse-attn-per-layer SCFA damping adopts the trainer's α formula; and one new interlock: bit-512 (`a_drift`) present → hard-error (exit-7 class).
- **Build gotcha (both repos):** the trainer links glades statically. After ANY glades-ml change: `cd ~/dev/glades-ml/build && make install`, THEN `cd ~/dev/glades-trainer && bash build.sh`. `make install` alone never updates the trainer.
- glades-ml builds: `sh .configure.sh cuda` (root); unit tests: `cd unit-tests/build && sh .configure.sh cuda`, then `cd unit-tests && bash test.sh <name>`.
- Commit after every task, in the repo the task touches (some tasks touch both — two commits).
- Production checkpoints for gates (all under `~/dev/glades-trainer/database/checkpoints/`):
  - PIED (flagship): `chiron_1B_pied_e4/chiron_1B_pied_e4.final` — serve with `--whisc-coupling --rot-theta-max 0.07`; recorded TF nll 1.0897 / top1 0.7176.
  - whisc30k: `chiron_1B_T16384_whisc30k/chiron_1B_T16384_whisc30k.final` — same flags; recorded TF 1.4655 / 0.626.
  - reanchor: `chiron_1B_T16384_reanchor5B_finish.final` — NO whisc flags; recorded TF nll 1.7798.

---

## Phase 0 — Golden capture (trainer repo, BEFORE any refactor)

### Task 1: Add `--dump-logits` to the current chiron_infer and capture goldens

**Files:**
- Modify: `~/dev/glades-trainer/tools/chiron_infer.cpp` (CLI parse ~line 943–1005; tf-check block ~line 1369–1402)
- Output: `~/dev/glades-trainer/logs/chiron-unify-goldens/` (git-ignored artifacts + a README)

**Interfaces:**
- Produces: golden files `{pied,whisc30k,reanchor}.logits.bin` and `{...}.tf.txt` that Task 10 compares against, and the dump format: `int32 header[3] = {useLen, V, stride=128}` followed by full V-float fp32 rows for positions `0, 128, 256, …` (< useLen) plus position `useLen-1`.

- [ ] **Step 1: Add the flag and dump code**

In the variable block near line 913 (`bool tfCheck = false;`), add:

```cpp
	std::string dumpLogitsPath;  // 2026-07-03 parity harness: raw logit rows (with --tf-check)
```

In the CLI parse loop (after the `--tf-check` case at line 968), add:

```cpp
		else if (streq(argv[i], "--dump-logits") && i + 1 < argc) dumpLogitsPath = argv[++i];
```

In the `tfCheck` block, immediately AFTER `s.logits.download(&logitsAll[0], logitsAll.size());` (line 1382) and before the accuracy loop, add:

```cpp
			if (!dumpLogitsPath.empty())
			{
				std::FILE* df = std::fopen(dumpLogitsPath.c_str(), "wb");
				if (!df) { std::fprintf(stderr, "chiron_infer: cannot open %s\n", dumpLogitsPath.c_str()); return 1; }
				const int32_t dhdr[3] = { (int32_t)useLen, (int32_t)dims.V, 128 };
				std::fwrite(dhdr, sizeof(int32_t), 3, df);
				for (int i2 = 0; i2 < useLen; i2 += 128)
					std::fwrite(&logitsAll[(size_t)i2 * dims.V], sizeof(float), dims.V, df);
				if ((useLen - 1) % 128 != 0)
					std::fwrite(&logitsAll[(size_t)(useLen - 1) * dims.V], sizeof(float), dims.V, df);
				std::fclose(df);
				std::printf("[dump-logits] wrote %s (useLen=%d V=%d stride=128)\n",
				            dumpLogitsPath.c_str(), useLen, dims.V);
			}
```

- [ ] **Step 2: Rebuild the trainer**

Run: `cd ~/dev/glades-trainer && bash build.sh`
Expected: clean build, `build/chiron_infer` updated.

- [ ] **Step 3: Capture goldens on all three checkpoints**

```bash
cd ~/dev/glades-trainer
mkdir -p logs/chiron-unify-goldens
G=logs/chiron-unify-goldens
./build/chiron_infer --model database/checkpoints/chiron_1B_pied_e4/chiron_1B_pied_e4.final \
  --vocab-file pretok-data/vocab.bpe --whisc-coupling --rot-theta-max 0.07 \
  --tokens-file pretok-data/val.tok.bin --tf-check --dump-logits $G/pied.logits.bin \
  | tee $G/pied.tf.txt
./build/chiron_infer --model database/checkpoints/chiron_1B_T16384_whisc30k/chiron_1B_T16384_whisc30k.final \
  --vocab-file pretok-data/vocab.bpe --whisc-coupling --rot-theta-max 0.07 \
  --tokens-file pretok-data/val.tok.bin --tf-check --dump-logits $G/whisc30k.logits.bin \
  | tee $G/whisc30k.tf.txt
./build/chiron_infer --model database/checkpoints/chiron_1B_T16384_reanchor5B_finish.final \
  --vocab-file pretok-data/vocab.bpe \
  --tokens-file pretok-data/val.tok.bin --tf-check --dump-logits $G/reanchor.logits.bin \
  | tee $G/reanchor.tf.txt
```

Expected: each prints a `[tf-check] positions=… top1_acc=… mean_nll=…` line. Sanity-check against the recorded ship numbers (PIED 1.0897/0.7176, whisc30k 1.4655/0.626, reanchor nll 1.7798). If a number differs from the record, the tokens window differs from the one used at ship time — that is OK: **the captured `*.tf.txt` become the authoritative parity baseline**; note the discrepancy in `$G/README.md`.

- [ ] **Step 4: Verify run-to-run determinism (validates the bit-parity bar)**

```bash
./build/chiron_infer --model database/checkpoints/chiron_1B_pied_e4/chiron_1B_pied_e4.final \
  --vocab-file pretok-data/vocab.bpe --whisc-coupling --rot-theta-max 0.07 \
  --tokens-file pretok-data/val.tok.bin --tf-check --dump-logits $G/pied.logits.rerun.bin >/dev/null
cmp $G/pied.logits.bin $G/pied.logits.rerun.bin && echo DETERMINISTIC
```

Expected: `DETERMINISTIC`. If it fails, STOP — record which kernel is nondeterministic in `$G/README.md`; the bit-parity gate (Task 10) then falls back to exact `tf.txt` reproduction per the spec's documented-fallback clause.

- [ ] **Step 5: Write `$G/README.md`** recording: date, git SHA of the trainer, the three exact commands, the three TF lines, determinism result.

- [ ] **Step 6: Commit (trainer repo)**

```bash
cd ~/dev/glades-trainer && git add tools/chiron_infer.cpp && git commit -m "chiron_infer: --dump-logits parity harness (pre-refactor golden capture)"
```

---

## Phase 1 — glades-ml: chiron_checkpoint module

### Task 2: `chiron_checkpoint.h` — bit registry, dims, weights structs, API declarations

**Files:**
- Create: `~/dev/glades-ml/Backend/Machine Learning/Networks/chiron_checkpoint.h`
- Create: `~/dev/glades-ml/Backend/Machine Learning/Networks/chiron_checkpoint.cpp` (stubs returning false, so it links)
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/CMakeLists.txt` (add `chiron_checkpoint.cpp` to `Networks_src_files`)

**Interfaces:**
- Produces (used by Tasks 3–6, 9, 12): everything declared in the header below. Signatures are frozen here — later tasks use them verbatim.

- [ ] **Step 1: Write the header**

```cpp
// chiron_checkpoint.h — CHRN/CHRF checkpoint format: single source of truth.
//
// Owns: the flag-bit registry, version rules, block codecs, the serving
// reader (model sections only), and model-section writer helpers.  The
// trainer (glades-trainer) composes its optimizer sections (Adam/Kahan/FACE)
// around these helpers; chiron_infer consumes the reader wholesale.
//
// Canonical CHRF section order (writer contract — reader depends on it):
//   magic 'CHRF' | version u32 | hdr i32[6]={T,m,L,nH,dH,V}
//   | step i32 | slcLastTransitionStep i32 | runtimeT i32 | runtimeL i32
//   | runtimeSasAlpha f32 | flags u32 | faceStepCount i32
//   | weights blob (E, then per layer Wq,Wk,Wv,Wo,gamma,beta[,gamma_p,beta_p])
//   | [bit 128] SCFA: k i32, w i32, per-layer D[m*(w+1)] f32
//   | [bit 256] QK-Norm gamma: L*nH f32
//   | [bits 2|16|32] Adam state | [bit 4] Kahan | [bit 1] FACE   (trainer-owned)
//   | [bit 512] a_drift: L*m f32
//   | [bit 1024] rot_phi: L*m f32       <-- MUST STAY LAST (EOF-tail read)
// Version rules: v4 iff bit 64 (bf16-on-disk); else v3 iff bit 8 (gamma_p);
// else v2.  CHRN (legacy) v1..3; v3 prepends a u32 flags word (bits below).
//
// C++98.

#ifndef _GLADES_CHIRON_CHECKPOINT_H_
#define _GLADES_CHIRON_CHECKPOINT_H_

#include <cstdio>
#include <stdint.h>
#include <string>
#include <vector>
#include "cuda/gpu_buffer.h"

namespace glades {
namespace chiron {

// CHRF flags word bits (both binaries compile against this one registry).
enum ChironCkptBits
{
	CKPT_BIT_FACE         = 1,
	CKPT_BIT_BF16_ADAM    = 2,
	CKPT_BIT_KAHAN        = 4,
	CKPT_BIT_GAMMA_P      = 8,
	CKPT_BIT_INT8_ADAM    = 16,
	CKPT_BIT_FP32_ADAM    = 32,
	CKPT_BIT_BF16_DISK    = 64,
	CKPT_BIT_SCFA         = 128,
	CKPT_BIT_QKNORM_GAMMA = 256,
	CKPT_BIT_A_DRIFT      = 512,
	CKPT_BIT_ROT_PHI      = 1024
};
// All bits the current format defines (reader hard-errors on anything above:
// an unknown section would corrupt the rot_phi EOF-tail read).
static const uint32_t CKPT_KNOWN_BITS_MASK = 0x7FFu;  // == 2047 == bits 1..1024

// CHRN v3 legacy flags word bits.
enum ChironChrnBits
{
	CHRN_BIT_BF16_WEIGHTS = 1,
	CHRN_BIT_HAS_GAMMA_P  = 2
};

struct ChironModelDims
{
	int T, m, L, nH, dH, V, dModel;
	ChironModelDims() : T(0), m(0), L(0), nH(0), dH(0), V(0), dModel(0) {}
};

struct ChironScfaState
{
	int k;
	int w;
	bool present;   // SCFA forward active (resolve-time)
	bool dLoaded;   // D came from the checkpoint
	glades::gpu::GpuBuffer<float> B;                 // [T,k] DCT-II basis (recomputed, not stored)
	std::vector<glades::gpu::GpuBuffer<float>*> D;   // [L][m*(w+1)]
	ChironScfaState();
	~ChironScfaState();   // deletes D[i]
private:
	ChironScfaState(const ChironScfaState&);
	ChironScfaState& operator=(const ChironScfaState&);
};

struct ChironModelWeights
{
	glades::gpu::GpuBuffer<float> E;                       // [V,m]
	std::vector<glades::gpu::GpuBuffer<float>*> Wq, Wk, Wv, Wo;  // [m,dModel]/[dModel,m]
	std::vector<glades::gpu::GpuBuffer<float>*> gamma, beta;     // [m]
	std::vector<glades::gpu::GpuBuffer<float>*> gamma_p, beta_p; // [m]; empty if absent
	ChironScfaState scfa;
	std::vector<float> qknormGamma;  // L*nH iff bit 256, else empty
	std::vector<float> rotPhi;       // L*m  iff bit 1024, else empty
	bool hasADrift;                  // bit 512 seen (payload NOT parsed — serve-refusal signal)
	ChironModelWeights();
	~ChironModelWeights();  // deletes all per-layer buffers
private:
	ChironModelWeights(const ChironModelWeights&);
	ChironModelWeights& operator=(const ChironModelWeights&);
};

// ---- Block codecs (shared framing helpers) ----
// Read n weight elements (fp32, or bf16 widened to fp32 when bf16OnDisk).
bool chiron_read_block(std::FILE* fp, std::vector<float>& fp32buf, size_t n, bool bf16OnDisk);
// Write n elements from fp32buf (as fp32, or truncation-rounded bf16).
bool chiron_write_block(std::FILE* fp, const std::vector<float>& fp32buf, size_t n, bool bf16OnDisk);
// Optimizer-group codecs (uint16 bf16 pairs / int8 quads / fp32 pairs), ported
// verbatim from the trainer.  Exact signatures are fixed in Task 3 to match
// the trainer originals (chiron_main.cpp:5584-5740) minus `static`.

// ---- Header ----
struct ChironCkptHeader
{
	ChironModelDims dims;
	int32_t step;
	int32_t slcLastTransitionStep;
	int32_t runtimeT, runtimeL;
	float runtimeSasAlpha;
	uint32_t flags;
	int32_t faceStepCount;
	ChironCkptHeader() : step(0), slcLastTransitionStep(-1), runtimeT(0),
	                     runtimeL(0), runtimeSasAlpha(0.0f), flags(0), faceStepCount(0) {}
};
// Writes magic+version+hdr+meta+flags.  version derived from flags (see top).
bool chiron_write_header(std::FILE* fp, const ChironCkptHeader& h);

// ---- Model-section writers (byte layout owned here; data sourcing is caller's) ----
bool chiron_write_scfa_section(std::FILE* fp, int k, int w,
                               const std::vector<const float*>& D_host, size_t D_sz);
bool chiron_write_qknorm_section(std::FILE* fp, const float* gammaHost, size_t n); // n = L*nH
bool chiron_write_f32_tail_section(std::FILE* fp, const float* vals, size_t n);    // a_drift / rot_phi

// ---- Serving reader ----
// Loads model sections of a CHRN/CHRF checkpoint straight to GPU buffers.
// Returns true on success.  On failure: err gets a printable message and
// errCode gets 4 (load/parse failure) — callers map to their exit codes.
// Behavior is verbatim chiron_infer::loadCheckpoint (2026-07-03):
// sequential header/weights/SCFA/qknorm reads, rot_phi from the EOF tail,
// CHRN-v3 gamma_p file-size fallback, hard-error on flags & ~CKPT_KNOWN_BITS_MASK.
bool chiron_load_model(const std::string& path, ChironModelDims& dims,
                       ChironModelWeights& w, std::string& err, int& errCode);

} // namespace chiron
} // namespace glades

#endif
```

- [ ] **Step 2: Write the stub .cpp** — every declared function returns `false` (and sets `err="unimplemented"; errCode=4;` for the reader); struct ctors/dtors implemented for real (dtors delete vector members; `ChironModelWeights::ChironModelWeights() : hasADrift(false) {}`).

- [ ] **Step 3: Register in the build**

In `Backend/Machine Learning/Networks/CMakeLists.txt`, add `chiron_checkpoint.cpp` to `Networks_src_files` (after `checkpoint_persistence.cpp`, line ~12).

- [ ] **Step 4: Build**

Run: `cd ~/dev/glades-ml && sh .configure.sh cuda`
Expected: clean build (stubs compile+link).

- [ ] **Step 5: Commit (lib repo)**

```bash
cd ~/dev/glades-ml && git add "Backend/Machine Learning/Networks/chiron_checkpoint.h" "Backend/Machine Learning/Networks/chiron_checkpoint.cpp" "Backend/Machine Learning/Networks/CMakeLists.txt" && git commit -m "chiron_checkpoint: CHRN/CHRF format module skeleton (bit registry + API)"
```

### Task 3: Block codecs (port) + first unit tests

**Files:**
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/chiron_checkpoint.cpp`
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/chiron_checkpoint.h` (add the four optimizer-group codec declarations with the exact trainer signatures)
- Create: `~/dev/glades-ml/unit-tests/Backend/Machine Learning/chiron-model-test.h` and `chiron-model-test.cpp`
- Modify: `~/dev/glades-ml/unit-tests/main.cpp` (+ the unit-tests CMake source list, wherever `chiron-test.cpp` is registered — grep for it)

**Interfaces:**
- Consumes: Task 2 declarations.
- Produces: working `chiron_read_block`/`chiron_write_block` + `chiron_{save,load}_bf16_group_with_count`, `chiron_{save,load}_int8_adam_group`, `chiron_{save,load}_fp32_adam_group` (exact signatures copied from `chiron_main.cpp:5584-5740`, `static` dropped, `chiron_` prefix added); test entry `CHIRONModelUnitTest()` dispatched by `test.sh chiron-model`.

- [ ] **Step 1: Write failing tests first.** In `chiron-model-test.cpp`, following the existing `chiron-test.cpp` conventions (`ASSERT(msg, pred)` from `unit-test.h`), write:

```cpp
// Test 1: fp32/bf16 block roundtrip via tmpfile().
void CHIRONCkptBlockCodecTest()
{
	std::vector<float> src(37);
	for (size_t i = 0; i < src.size(); ++i) src[i] = 0.5f * (float)i - 3.0f;
	// fp32: exact roundtrip
	std::FILE* fp = std::tmpfile();
	ASSERT("tmpfile", fp != 0);
	ASSERT("write fp32", glades::chiron::chiron_write_block(fp, src, src.size(), false));
	std::rewind(fp);
	std::vector<float> got;
	ASSERT("read fp32", glades::chiron::chiron_read_block(fp, got, src.size(), false));
	for (size_t i = 0; i < src.size(); ++i)
		ASSERT("fp32 exact", got[i] == src[i]);
	std::fclose(fp);
	// bf16: roundtrip matches explicit truncation of the source
	fp = std::tmpfile();
	ASSERT("write bf16", glades::chiron::chiron_write_block(fp, src, src.size(), true));
	std::rewind(fp);
	ASSERT("read bf16", glades::chiron::chiron_read_block(fp, got, src.size(), true));
	for (size_t i = 0; i < src.size(); ++i)
	{
		union { float f; uint32_t u; } v; v.f = src[i];
		v.u &= 0xFFFF0000u;   // must match the trainer's bf16 encode (verify at port time!)
		ASSERT("bf16 roundtrip", got[i] == v.f);
	}
	std::fclose(fp);
}
```

Add `CHIRONModelUnitTest()` in the same file calling the codec test; declare both in `chiron-model-test.h`; register the source file in the unit-tests CMake list and add to `unit-tests/main.cpp` (next to the `chiron-pied` dispatch at ~line 284):

```cpp
	    else if (strcmp(argv[1], "chiron-model") == 0)
	        CHIRONModelUnitTest();
```

- [ ] **Step 2: Run to verify failure**

Run: `cd ~/dev/glades-ml/unit-tests/build && sh .configure.sh cuda && cd .. && bash test.sh chiron-model`
Expected: FAIL (stubs return false) — the "write fp32" assert fires.

- [ ] **Step 3: Port the codecs.** Open `~/dev/glades-trainer/trainer/chiron_main.cpp` and port, into `chiron_checkpoint.cpp`:
  - `load_weight_block` (5887) → `chiron_read_block`; `save_weight_block` (5741) → `chiron_write_block` (note: the trainer version takes a mutable scratch vector — check whether it round-trips through bf16 encode in-place; the lib version takes `const&` and uses a local scratch. **The bf16 encode must be bit-identical to the trainer's** — copy the exact truncation/rounding expression, and fix the Step-1 test to match if the trainer rounds-to-nearest instead of truncating).
  - `save/load_bf16_group_with_count` (5584/5596), `save/load_int8_adam_group` (5630/5657), `save/load_fp32_adam_group` (5692/5707) → same names with `chiron_` prefix, `static` dropped, signatures otherwise verbatim (they take `std::FILE*` + GpuBuffer pointers + counts). Add their exact declarations to the header.
  - C++98 check: these are already C++98-compatible C-stdio code.

- [ ] **Step 4: Run tests to verify pass**

Run: `cd ~/dev/glades-ml/unit-tests && bash test.sh chiron-model`
Expected: PASS.

- [ ] **Step 5: Commit (lib repo)** — `git add` the four files, message `"chiron_checkpoint: block codecs ported from trainer + chiron-model test suite"`.

### Task 4: Serving reader + model-section writers + roundtrip tests

**Files:**
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/chiron_checkpoint.cpp`
- Modify: `~/dev/glades-ml/unit-tests/Backend/Machine Learning/chiron-model-test.cpp`

**Interfaces:**
- Consumes: Task 2 API, Task 3 codecs.
- Produces: working `chiron_load_model`, `chiron_write_header`, `chiron_write_scfa_section`, `chiron_write_qknorm_section`, `chiron_write_f32_tail_section`.

- [ ] **Step 1: Write failing roundtrip tests.** In `chiron-model-test.cpp` add `CHIRONCkptRoundtripTest()` (and call it from `CHIRONModelUnitTest`). Tiny shape: `T=8, m=4, L=2, nH=1, dH=4, V=16` (`dModel=4`). Helper writes a synthetic CHRF to a temp file **using the lib writers**, with deterministic weight values (`w[i] = sin(0.1*i + section)` pattern via a fill function), then `chiron_load_model`s it back, downloads every GpuBuffer (ASSERT the download return — known gotcha) and compares element-exact. Cases:
  1. Minimal CHRF v2 (flags=0): weights only.
  2. CHRF v3 with gamma_p (bit 8).
  3. CHRF v4 bf16-on-disk (bit 64): compare against bf16-truncated source.
  4. Bits 128+256+1024 (SCFA + qknorm + rot_phi), **with a dummy 100-float fp32-Adam-like payload written between the qknorm section and the rot_phi tail with bit 32 set** — proves the EOF-tail read skips unparsed optimizer sections exactly like production files.
  5. Unknown-bit rejection: write flags with bit 2048 set → `chiron_load_model` must return false with errCode 4 and an err mentioning "unknown".
  6. Bit-512 signal: write flags with bit 512 + a dummy L*m payload before rot_phi → load succeeds and `w.hasADrift == true` (payload untouched).
  7. Legacy CHRN v1 (weights only, no flags) and CHRN v3 with `CHRN_BIT_BF16_WEIGHTS|CHRN_BIT_HAS_GAMMA_P` — write with a small local writer in the test (CHRN writer is not lib API; layout: magic+version+[flags]+hdr6+blob).

- [ ] **Step 2: Run to verify failure** — `bash test.sh chiron-model` → FAIL (reader is a stub).

- [ ] **Step 3: Implement.**
  - `chiron_write_header`: exact field order from `save_full_checkpoint` (chiron_main.cpp:6103–6112); version rule: `(flags & CKPT_BIT_BF16_DISK) ? 4 : ((flags & CKPT_BIT_GAMMA_P) ? 3 : 2)`.
  - Section writers: SCFA = `k i32, w i32`, then L×`D[m*(w+1)]` fp32 rows from `D_host`; qknorm = `n` fp32; tail = `n` fp32.
  - `chiron_load_model`: port `loadCheckpoint` from `~/dev/glades-trainer/tools/chiron_infer.cpp:87–355` **verbatim in behavior**, with these mechanical transformations:
    - The `readBlock` lambda (226–239) becomes a call to `chiron_read_block` (Task 3 — identical logic).
    - Every `std::fprintf(stderr, …); return false;` becomes: format the same text into `err` (use `char buf[512]; std::snprintf` — note C++98: `snprintf` is fine via `<cstdio>` on this toolchain, matching existing lib usage), set `errCode=4`, `std::fclose(fp)`, `return false`.
    - The informational `std::printf("[chiron-infer] …")` lines stay, but with the tag changed to `[chiron-ckpt]` — **exception**: keep the text after the tag identical.
    - Bit tests use the `ChironCkptBits` constants; the unknown-bit mask check uses `CKPT_KNOWN_BITS_MASK`.
    - Bit-512: after the qknorm section, set `w.hasADrift = (chrfFlags & CKPT_BIT_A_DRIFT) != 0;` (new — the old reader only "knew" the bit in a comment).
    - Output params map onto `ChironModelWeights` members (same names).
  - Guard the GpuBuffer-touching body with `#ifdef GLADES_HAVE_CUDA` (non-CUDA: `err="CUDA required"; errCode=4; return false;`), same pattern as `gpu_chiron.h`.

- [ ] **Step 4: Run tests to verify pass** — `bash test.sh chiron-model` → PASS. Also run `bash test.sh chiron` to confirm no regression in the existing suite.

- [ ] **Step 5: Commit (lib repo)** — message `"chiron_checkpoint: serving reader + model-section writers + roundtrip tests"`.

---

## Phase 2 — glades-ml: chiron_serving module

### Task 5: `chiron_serving.h` + resolve decision table + tests

**Files:**
- Create: `~/dev/glades-ml/Backend/Machine Learning/Networks/chiron_serving.h`
- Create: `~/dev/glades-ml/Backend/Machine Learning/Networks/chiron_serving.cpp`
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/CMakeLists.txt` (add `chiron_serving.cpp`)
- Modify: `~/dev/glades-ml/unit-tests/Backend/Machine Learning/chiron-model-test.cpp`

**Interfaces:**
- Consumes: `chiron_checkpoint.h` types.
- Produces:

```cpp
// chiron_serving.h (namespace glades::chiron; C++98; GLADES_HAVE_CUDA-gated)
struct ChironServingOverrides
{
	int   fuseAttnPerLayer;   // -1 unset (default-on heuristic applies), 0 --no-fuse-attn, 1 --fuse-attn-per-layer
	bool  fuseAttnReln;       // --fuse-attn-reln
	bool  qkNorm;             // --qk-norm
	float qkNormGamma;        // --qk-norm-gamma; <=0 => auto log2(T)
	int   scfaForceMode;      // 0 auto, 1 --scfa, -1 --no-scfa
	int   scfaKOverride;      // 0 = checkpoint's k
	int   scfaWOverride;      // -1 = checkpoint's w
	bool  whiscCoupling;      // --whisc-coupling
	float rotThetaMax;        // default 0.07f (flagship recipe)
	float whiscClamp;         // default 8.0f
	int   seqLen;             // 0 = checkpoint T
	ChironServingOverrides(); // defaults: {-1,false,false,0.0f,0,0,-1,false,0.07f,8.0f,0}
};
struct ChironServingConfig
{
	bool  fuseAttnPerLayer, fuseAttnReln, qkNorm, useScfa, whiscCoupling;
	float rotThetaMax, whiscClamp;
	float epsReln;                        // 1e-4f
	std::vector<float> qknormGammaScale;  // L*nH of gamma*sqrt(dH), ready for upload; empty if !qkNorm
	ChironServingConfig();
};
// Applies the serving decision table.  May mutate dims.T (seqLen) and
// w.scfa (k/w resolution, B build, D=0 fallback).  Returns 0 = ok,
// 6 = SCFA init failure, 7 = interlock violation (message in err).
int chiron_resolve_serving(ChironModelDims& dims, ChironModelWeights& w,
                           const ChironServingOverrides& o,
                           ChironServingConfig& cfg, std::string& err);
```

- [ ] **Step 1: Write failing decision-table tests** in `chiron-model-test.cpp` (`CHIRONResolveServingTest()`, called from `CHIRONModelUnitTest`). Build tiny in-memory `ChironModelWeights` states (no file IO needed — populate fields directly; allocate tiny GpuBuffers where the rule inspects them) and assert, one sub-case per rule, matching `chiron_infer.cpp` main (lines noted):
  1. rotPhi present + `!o.whiscCoupling` → returns 7 (interlock, 1047–1055).
  2. rotPhi empty + `o.whiscCoupling` → returns 7 (1056–1063).
  3. `w.hasADrift` → returns 7 (**new** bit-512 refusal; err mentions `a_drift`).
  4. gamma_p empty + fuse unset → `cfg.fuseAttnPerLayer == false` (auto-disable, 1064–1069).
  5. gamma_p empty + `scfa.dLoaded` + `!o.fuseAttnReln` → `cfg.fuseAttnReln == true` (2026-06-27 auto-enable, 1079–1084).
  6. WhiSC + gamma_p at dead init (upload gamma_p[0]=1.0…, beta_p[0]=0.0) + scfa.dLoaded + fuse unset → `fuseAttnPerLayer=false, fuseAttnReln=true` (dead-gamma_p guard, 1097–1116).
  7. Same but gamma_p[0][0]=1.5 (trained) → per-layer fuse kept (warn path, 1117–1121).
  8. qknormGamma sized L*nH + `!o.qkNorm` → `cfg.qkNorm == true` and `cfg.qknormGammaScale[i] == qknormGamma[i]*sqrt(dH)` (auto-enable + exact-γ prefill, 1202–1229).
  9. qknormGamma empty + `o.qkNorm` → approx γ = log2(T) used in the scale (1224–1225).
  10. `scfa.dLoaded` + auto mode → `cfg.useScfa == true`; `o.scfaForceMode==-1` → false (1135–1138).
  11. `o.seqLen=4` (< T) → `dims.T == 4` (1129).
  12. SCFA on with no loaded D → `w.scfa.D` allocated+zeroed, `w.scfa.B` allocated, k/w defaulted to `T/16` (min 4) and `8` (1140–1185); returns 0.

- [ ] **Step 2: Run to verify failure** — `bash test.sh chiron-model` → FAIL (`chiron_resolve_serving` undefined/stub).

- [ ] **Step 3: Implement `chiron_resolve_serving`** by porting `chiron_infer.cpp:1040–1234` in order: WhiSC interlocks → **new bit-512 refusal** (insert after the WhiSC interlocks; message: `"FATAL — checkpoint carries OBSD a_drift (CHRF flag bit 512) but the eval forward does not apply per-layer drift. Serving would be silently wrong. (OBSD is a closed NO-GO arc; retrain without --per-layer-drift or extend chiron_serving.)"`) → gamma_p auto-disable → fuse-attn-reln auto-enable → dead-gamma_p guard → seqLen override → SCFA mode/k/w/B/D resolution → QK-Norm auto-enable + γ·√dH prefill (into `cfg.qknormGammaScale`, upload happens in scratch alloc, Task 6). Keep every `printf` message verbatim (tag → `[chiron-serving]`). Fatal messages go to `err` + return code, not stderr.

- [ ] **Step 4: Run tests to verify pass** — `bash test.sh chiron-model` → PASS.

- [ ] **Step 5: Commit (lib repo)** — `"chiron_serving: resolve decision table (interlocks + auto-enables + new a_drift refusal)"`.

### Task 6: Eval scratch + eval forward + orchestration parity test

**Files:**
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/chiron_serving.h` / `.cpp`
- Modify: `~/dev/glades-ml/unit-tests/Backend/Machine Learning/chiron-model-test.cpp`

**Interfaces:**
- Consumes: Tasks 2–5 types.
- Produces:

```cpp
struct ChironEvalScratch
{
	glades::gpu::GpuBuffer<int>   d_tokens;   // [T] caller uploads
	glades::gpu::GpuBuffer<float> logits;     // [T,V] caller downloads
	// …all other members ported 1:1 from chiron_infer.cpp Scratch (359–458):
	// q, p, q_tmp, stats, sQ/sK/sV/sO, scratch_P, p_norm, stats_p,
	// scfa_* (12 buffers), qknorm_invNorm, qknorm_gamma_scale,
	// rot_a, rot_c, whisc_Pbar, whisc_Qbar, whisc_a,
	// plus rotPhiGpu (std::vector<GpuBuffer<float>*>, owned, from w.rotPhi).
	bool allocate(const ChironModelDims& d, const ChironModelWeights& w,
	              const ChironServingConfig& cfg);   // also uploads qknormGammaScale + rotPhi
	~ChironEvalScratch();
};
bool chiron_eval_forward(const ChironModelDims& d, const ChironModelWeights& w,
                         const ChironServingConfig& cfg, ChironEvalScratch& s);
```

- [ ] **Step 1: Write the failing orchestration-parity test** (`CHIRONEvalForwardParityTest()`). Rationale: kernel *math* is already covered by the existing chiron parity suites; the risk class here is orchestration (wrong order/argument/buffer), so the reference is a **direct composition of the same gpu primitives in the test**, transcribed independently from `forwardInfer` (580–740). Config: dense path, `T=8, m=4, L=2, nH=1, dH=4, V=16`, deterministic weights, `fuseAttnReln=true`. Reference: `embedding_gather` → per layer `chiron_attention_shear_tiled` (dense) → at `l==L-1` `axpy(1.0,p,q)` → `chiron_reln_forward` + d2d copy → `sgemm_rowmajor_abt` readout. Assert `chiron_eval_forward` logits are **bit-identical** to the reference (same primitives, same order → same bits). Add a second case with `whiscCoupling=true` (tiny rotPhi values, the 4-call WhiSC block in the reference per 692–706) and a third with the SCFA path + qkNorm at `T=8, k=4, w=1` (reference transcribed from `scfa_shear_infer` 484–578, QK-Norm branch 526–548).

- [ ] **Step 2: Run to verify failure** — `bash test.sh chiron-model` → FAIL.

- [ ] **Step 3: Implement.** Port, preserving the kernel call sequence exactly:
  - `ChironEvalScratch::allocate` from `Scratch::allocate` (403–457) + the qknormGammaScale upload (1229) + the rotPhi per-layer upload loop (1238–1258, printing the same `rot_phi |.|max` line, tag `[chiron-serving]`).
  - `scfa_shear_eval` (file-local static) from `scfa_shear_infer` (484–578) — keep `dbg_mag`/`CHIRON_DBG` hooks.
  - `chiron_eval_forward` from `forwardInfer` (580–740) — keep the `CHIRON_DBG` blocks. **One intentional change** (spec §3.2): the `applyFuse` α on the SCFA path adopts the trainer's damped formula — read `chiron_main.cpp:13289–13325`, replicate its SCFA-branch α exactly (k/T-damped), keep plain `1/sqrt(L)` on the dense path; comment the divergence-fix with the date.
  - C++98: no lambdas/auto in the ported code (the source has none outside `readBlock`, already handled).

- [ ] **Step 4: Run tests to verify pass** — `bash test.sh chiron-model` → PASS; run `bash test.sh chiron` + `bash test.sh chiron-whisc` for regression.

- [ ] **Step 5: Commit (lib repo)** — `"chiron_serving: eval scratch + kernel-sequence-preserving eval forward + orchestration parity tests"`.

---

## Phase 3 — glades-ml: header install

### Task 7: Install the ML header tree + export the include dir

**Files:**
- Modify: `~/dev/glades-ml/CMakeLists.txt` (install rules ~line 213–230)

- [ ] **Step 1: Add the install rule** (keep the existing GMath rule for compatibility). After the GMath `install(FILES …)` block:

```cmake
# Install the full ML public header tree (namespaced under include/glades/) so
# downstream consumers (glades-trainer) compile against the installed headers
# instead of vendored copies.  Layout preserves the "Backend/Machine Learning/…"
# include convention.
install(DIRECTORY "Backend/Machine Learning/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/glades/Backend/Machine Learning"
    FILES_MATCHING PATTERN "*.h")
target_include_directories(glades INTERFACE
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/glades>)
```

- [ ] **Step 2: Build + install**

Run: `cd ~/dev/glades-ml && sh .configure.sh cuda && cd build && make install`
Expected: headers appear; verify:

```bash
test -f ~/.local/include/glades/"Backend/Machine Learning/Networks/chiron_serving.h" && \
test -f ~/.local/include/glades/"Backend/Machine Learning/Networks/cuda/gpu_chiron.h" && echo OK
grep -n "include/glades" ~/.local/share/glades/cmake/gladesTargets.cmake
```
Expected: `OK` and the interface include dir present in the exported targets file.

- [ ] **Step 3: Commit (lib repo)** — `"install: export ML header tree under include/glades (kills trainer header vendoring)"`.

---

## Phase 4 — glades-trainer switchover

### Task 8: Trainer consumes installed headers; delete the vendored tree

**Files:**
- Modify: `~/dev/glades-trainer/CMakeLists.txt` (line 30: `COMMON_INCLUDES`)
- Delete: `~/dev/glades-trainer/include/` (entire vendored tree)

- [ ] **Step 1: Check for vendored-only header usage.** 7 vendored headers don't exist upstream (`State/layer.h`, `State/LayerBuilder.h`, `State/NetworkState.h`, `State/edge.h`, `State/node.h`, `Networks/training_core.h`, `Networks/RNN.h`). Run:

```bash
cd ~/dev/glades-trainer && grep -rn 'State/layer.h\|LayerBuilder.h\|NetworkState.h\|State/edge.h\|State/node.h\|training_core.h\|Networks/RNN.h' trainer/ tools/ --include='*.cpp' --include='*.h'
```
Expected: no hits. If any hit: that header must be restored upstream in glades-ml first — stop and report.

- [ ] **Step 2: Switch the include path.** In `CMakeLists.txt` line 30 change:

```cmake
set(COMMON_INCLUDES ${ZSTD_INCLUDE_DIR})
```
(The glades include dir now flows from `find_package(glades)` via the target's interface.)

- [ ] **Step 3: Delete the vendored tree and rebuild**

```bash
cd ~/dev/glades-trainer && git rm -r include/ -q && rm -rf build/CMakeCache.txt build/CMakeFiles && bash build.sh
```
Expected: clean build of all targets (`glades_pile_train`, `glades_chiron_train`, `chiron_infer`, …). CMakeCache is sticky — the cache delete is required after include-path changes.

- [ ] **Step 4: Smoke check** — `./build/chiron_infer --help` prints usage; re-run the Task-1 PIED capture command with `--dump-logits /tmp/claude-1000/-home-robert-dev-glades-ml/d588a29b-1de0-4805-b992-4cebf0ba59af/scratchpad/t8.bin` and `cmp` against the golden — must be identical (nothing but include paths changed).

- [ ] **Step 5: Commit (trainer repo)** — `"build: consume installed glades headers (include/glades); delete vendored header tree"`.

### Task 9: Thin out chiron_infer to the lib API

**Files:**
- Modify: `~/dev/glades-trainer/tools/chiron_infer.cpp`

**Interfaces:**
- Consumes: `glades::chiron::{chiron_load_model, chiron_resolve_serving, chiron_eval_forward, ChironModelDims, ChironModelWeights, ChironServingOverrides, ChironServingConfig, ChironEvalScratch}`.

- [ ] **Step 1: Rewrite.** Keep (verbatim): the self-test block, `gen_degeneration_metrics`, `sampleToken`, the CLI parse loop, the `generate` lambda, tokens-file / tf-check / dump-logits / seed-tail / interactive dispatch. Delete: `ModelDims`, `ScfaState`, `loadCheckpoint` (56–355), `Scratch` (359–458), `dbg_mag`, `scfa_shear_infer`, `forwardInfer` (473–740), **and the manual buffer-cleanup loop at 1427–1432** (`ChironModelWeights`/`ChironEvalScratch` destructors own that now). New main flow:

```cpp
#include "Backend/Machine Learning/Networks/chiron_checkpoint.h"
#include "Backend/Machine Learning/Networks/chiron_serving.h"
using glades::chiron::ChironModelDims;   // etc.

	// after CLI parse + initDevice(2) + vocab(3):
	ChironModelDims dims;
	glades::chiron::ChironModelWeights w;
	std::string err; int errCode = 0;
	if (!glades::chiron::chiron_load_model(modelPath, dims, w, err, errCode))
	{ std::fprintf(stderr, "chiron_infer: %s\n", err.c_str()); return 4; }

	glades::chiron::ChironServingOverrides ov;
	ov.fuseAttnPerLayer = fusePerLayerExplicit ? (fuseAttnPerLayer ? 1 : 0) : -1;
	ov.fuseAttnReln = fuseAttnReln;   ov.qkNorm = qkNorm;  ov.qkNormGamma = qkNormGamma;
	ov.scfaForceMode = scfaForceMode; ov.scfaKOverride = scfaKOverride; ov.scfaWOverride = scfaWOverride;
	ov.whiscCoupling = whiscCoupling; ov.rotThetaMax = rotThetaMax; ov.whiscClamp = whiscClamp;
	ov.seqLen = seqLen;

	glades::chiron::ChironServingConfig cfg;
	int rc = glades::chiron::chiron_resolve_serving(dims, w, ov, cfg, err);
	if (rc != 0) { std::fprintf(stderr, "chiron_infer: %s\n", err.c_str()); return rc; }  // 6 or 7

	glades::chiron::ChironEvalScratch s;
	if (!s.allocate(dims, w, cfg))
	{ std::fprintf(stderr, "chiron_infer: scratch allocation failed — likely OOM at T=%d V=%d%s\n",
	               dims.T, dims.V, cfg.useScfa ? "" : " (try --scfa or smaller --seq-len)"); return 5; }
```

Every former `forwardInfer(dims, s, Wq, …)` call site becomes `chiron_eval_forward(dims, w, cfg, s)` (token upload/logits download via `s.d_tokens`/`s.logits` unchanged).

- [ ] **Step 2: Rebuild** — `cd ~/dev/glades-ml/build && make install && cd ~/dev/glades-trainer && bash build.sh` (lib unchanged here, but keep the habit). Expected: clean build; `chiron_infer.cpp` now ~600 lines.

- [ ] **Step 3: Quick functional check** — run the PIED TF command from Task 1 (without dump); the `[chiron-ckpt]`/`[chiron-serving]` messages appear and the TF line prints.

- [ ] **Step 4: Commit (trainer repo)** — `"chiron_infer: thin CLI over glades::chiron lib modules (load/resolve/eval-forward)"`.

### Task 10: GATE — bit-identity + TF metrics on the three era checkpoints

**Files:** none (verification only; results appended to `logs/chiron-unify-goldens/README.md`)

- [ ] **Step 1: Re-run all three Task-1 capture commands** with output to `$G/{name}.post.logits.bin` / `$G/{name}.post.tf.txt`.

- [ ] **Step 2: Bit-identity**

```bash
cd ~/dev/glades-trainer/logs/chiron-unify-goldens
for n in pied whisc30k reanchor; do cmp $n.logits.bin $n.post.logits.bin && echo "$n BIT-IDENTICAL"; done
```
Expected: three `BIT-IDENTICAL` lines. On any mismatch: STOP, diagnose (the eval-forward port changed a kernel call), fix, re-run. Only with a documented root cause may this gate fall back to Step 3 alone (record it in README.md).

- [ ] **Step 3: TF metrics** — `grep '\[tf-check\]' *.post.tf.txt` must equal the baseline `tf.txt` lines at printed precision (and match the recorded ship numbers per Task-1 findings).

- [ ] **Step 4: Append results to README.md; commit (trainer repo)** — `"parity gate: post-refactor chiron_infer bit-identical + TF metrics reproduced on PIED/whisc30k/reanchor"` (README only; goldens stay untracked).

### Task 11: GATE — interlock regression script

**Files:**
- Create: `~/dev/glades-trainer/scripts/chiron_serving_interlocks.sh`

- [ ] **Step 1: Write the script** (checks exit codes + key messages; uses the real checkpoints):

```bash
#!/bin/sh
# Serving-interlock regression for chiron_infer (2026-07-03 unification).
set -u
cd "$(dirname "$0")/.."
BIN=build/chiron_infer
PIED=database/checkpoints/chiron_1B_pied_e4/chiron_1B_pied_e4.final
REAN=database/checkpoints/chiron_1B_T16384_reanchor5B_finish.final
fail=0
expect() { # name expected_code cmd...
  name=$1; want=$2; shift 2
  "$@" >/dev/null 2>&1; got=$?
  [ "$got" = "$want" ] && echo "PASS $name (exit $got)" || { echo "FAIL $name: exit $got want $want"; fail=1; }
}
# 1. WhiSC ckpt without --whisc-coupling -> 7
expect whisc-missing-flag 7 $BIN --model $PIED --vocab-file pretok-data/vocab.bpe --tf-check --tokens-file pretok-data/val.tok.bin --tokens-file-n 64
# 2. --whisc-coupling on non-WhiSC ckpt -> 7
expect whisc-wrong-flag 7 $BIN --model $REAN --vocab-file pretok-data/vocab.bpe --whisc-coupling --tf-check --tokens-file pretok-data/val.tok.bin --tokens-file-n 64
# 3. Happy path (auto-enables: QK-Norm, fuse-reln, SCFA; WhiSC flags) -> 0
expect flagship-happy 0 $BIN --model $PIED --vocab-file pretok-data/vocab.bpe --whisc-coupling --rot-theta-max 0.07 --tf-check --tokens-file pretok-data/val.tok.bin --tokens-file-n 4096
# 4. Missing model -> 4 ; bad vocab -> 3
expect load-fail 4 $BIN --model /nonexistent.ckpt --vocab-file pretok-data/vocab.bpe
expect vocab-fail 3 $BIN --model $PIED --vocab-file /nonexistent.bpe
exit $fail
```

(Unknown-bit and bit-512 refusals are covered by the lib unit tests — no production checkpoint carries them.)

- [ ] **Step 2: Run** — `sh scripts/chiron_serving_interlocks.sh`; Expected: all PASS, exit 0.

- [ ] **Step 3: Commit (trainer repo)** — `"scripts: chiron serving-interlock regression"`.

### Task 12: chiron_main switches to lib format constants/codecs/writers

**Files:**
- Modify: `~/dev/glades-trainer/trainer/chiron_main.cpp` (codecs 5584–5798; `save_weights`/`load_weights` 5802–6031; `save_full_checkpoint` 6046–6351; `load_full_checkpoint` 6353–6688)

- [ ] **Step 1: Replace the file-local codecs** with `#include "Backend/Machine Learning/Networks/chiron_checkpoint.h"` + `using` the `glades::chiron::chiron_*` codec functions; delete the local `save/load_weight_block`, `save/load_bf16_group_with_count`, `save/load_int8_adam_group`, `save/load_fp32_adam_group` definitions and fix call sites (same signatures — mechanical rename).

- [ ] **Step 2: Switch `save_full_checkpoint`**: build the flags word from `ChironCkptBits` constants (replacing the literal `1u/2u/…/1024u` at 6089–6099); write the preamble via `chiron_write_header` (populate `ChironCkptHeader` from the same locals); write SCFA/qknorm/a_drift/rot_phi sections via `chiron_write_scfa_section`/`chiron_write_qknorm_section`/`chiron_write_f32_tail_section` (the download-to-host loops stay local); Adam/Kahan/FACE orchestration unchanged (now calling lib codecs). **rot_phi stays the last write before fclose** — the reader contract.

- [ ] **Step 3: Switch the readers** (`load_weights`, `load_full_checkpoint`): lib codecs + bit constants; orchestration (incl. optimizer-state parsing) stays local.

- [ ] **Step 4: Byte-compat gate.** Rebuild (`bash build.sh`), then:
  1. `run_checkpoint_self_test` — run the trainer's self-test path (grep chiron_main.cpp:6905 for its trigger flag, e.g. `--ckpt-self-test`; run it). Expected: PASS.
  2. Smoke train + cross-reader check: train a tiny model for ~20 steps with `--scfa --qk-norm --whisc-coupling` at small scale (use the smallest `run.sh chiron --scale 66M`-class config that exercises bits 128|256|1024), save, then load the saved file with the NEW `chiron_infer` (lib reader) — TF-check must run without error (proves new-writer ↔ lib-reader).
  3. Resume gate: resume `chiron_1B_pied_e4.final` with `--steps <current+2>` (2 steps) — loads full optimizer-bearing CHRF cleanly, no errors.

- [ ] **Step 5: Commit (trainer repo)** — `"chiron_main: checkpoint format via glades::chiron (bit registry + codecs + section writers)"`.

### Task 13: chiron_parity cross-forward tool

**Files:**
- Create: `~/dev/glades-trainer/tools/chiron_parity.cpp`
- Modify: `~/dev/glades-trainer/CMakeLists.txt` (new executable, same pattern as `chiron_infer`, plus `trainer/chiron_main.cpp`-shared sources are NOT needed — see below)

- [ ] **Step 1: Scope decision (already made in spec):** the tool compares **lib eval forward** vs **trainer forward**. The trainer forward lives inside `chiron_main.cpp` (not a library). Rather than extract it (out of scope), the tool shells the comparison: `chiron_parity` runs the lib forward on a token window and writes per-position NLL; the trainer side reuses the *existing* val path. Concretely `chiron_parity.cpp`: load checkpoint via lib, resolve, forward over `--tokens-file` window, print `mean_nll` at full precision (`%.6f`) — i.e. a headless tf-check. The cross-check compares it against the trainer's `run_validation` NLL on the same window (`glades_chiron_train` resumed with `--steps 0`-equivalent val-only invocation — find the existing val-only flag; `run.sh` exposes the lr=0 resume trick used for wide-val). Tolerance: `|Δnll| < 2e-2` (the known FP32-vs-bf16-path gap is ~0.06 at flagship scale between differently-batched windows; on an IDENTICAL window the paths differ only by bf16-residual-p rounding — calibrate on first run, then pin the observed gap + 50% margin in the script).

- [ ] **Step 2: Write the tool** (≈120 lines: arg parse `--model --tokens-file --tokens-file-n --whisc-coupling --rot-theta-max`, then load→resolve→forward→NLL exactly like chiron_infer's tf-check minus vocab/sampling; exit 0 with `[parity] nll=…` line).

- [ ] **Step 3: Run both sides on the flagship + same window**; record both numbers and the pinned tolerance in `logs/chiron-unify-goldens/README.md`. Expected: within tolerance.

- [ ] **Step 4: Commit (trainer repo)** — `"tools: chiron_parity — lib eval forward vs trainer val forward tripwire"`.

### Task 14: Docs, memory, final verification sweep

**Files:**
- Modify: `~/dev/glades-ml/CLAUDE.md` (Build Commands section: note the header install; new `chiron-model` test name in the test list)
- Modify: `~/dev/glades-trainer/CLAUDE.md` (build gotcha now includes headers via `make install`; vendored include/ is GONE; chiron_infer is a thin CLI over `glades::chiron`)
- Modify: memory `MEMORY.md` + new memory file per the memory instructions (project-type: serving unification landed; the golden/parity workflow)

- [ ] **Step 1: Update both CLAUDE.mds** (facts only: new modules, install layout, `test.sh chiron-model`, parity harness locations, the bit-512 refusal).

- [ ] **Step 2: Full regression sweep**

```bash
cd ~/dev/glades-ml/unit-tests && bash test.sh chiron-model && bash test.sh chiron && bash test.sh chiron-rot && bash test.sh chiron-whisc && bash test.sh chiron-pied
cd ~/dev/glades-trainer && sh scripts/chiron_serving_interlocks.sh && sh runner.sh --flagship --tf-check --tokens-file pretok-data/val.tok.bin | grep '\[tf-check\]'
```
Expected: all suites pass; runner.sh flagship TF line matches the golden.

- [ ] **Step 3: Final commits both repos; summarize** the shipped state (modules, gates passed, behavior changes: bit-512 refusal + SCFA fuse-α fix) to the user.

---

## Self-review notes (spec coverage)

- Spec §3.1 (checkpoint module) → Tasks 2–4. §3.2 (serving module incl. bit-512 refusal + k/T damping) → Tasks 5–6. §4 (install + unit tests) → Tasks 3–7. §5 (trainer changes) → Tasks 8–9, 12–13. §6 gates 1–7 → Tasks 1, 3–6 (unit), 10 (bit+TF), 11 (interlocks), 12 (trainer smoke/resume), 13 (parity tool), 14 (runner.sh sweep). §7 constraints → Global Constraints.
- Known open detail resolved at port time (flagged in-task): exact bf16 encode rounding (Task 3 Step 3), trainer SCFA fuse-α formula (Task 6 Step 3), trainer val-only invocation flag (Task 13 Step 1).
