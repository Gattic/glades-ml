# CHIRON Full Inference Move Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move CHIRON generation (sampler + loop), teacher-forcing eval, and degeneration metrics into glades-ml as `chiron_generate.{h,cpp}`, with a C++98 MT19937 port that keeps seeded stochastic generation bit-identical to the current binary.

**Architecture:** Fourth CHIRON lib module alongside checkpoint/serving. Golden-first parity: capture RNG streams, sampler picks, and generation token sequences with the CURRENT binary before refactoring; the lib port must reproduce all of them exactly. The forward is already parity-locked from the previous arc, so any sequence mismatch localizes to RNG/sampler/loop.

**Tech Stack:** C++98 (lib), C++11 (trainer), existing `glades::chiron` serving API, libstdc++ `<random>` as the bit-behavior reference.

**Spec:** `docs/superpowers/specs/2026-07-03-chiron-full-inference-lib-design.md`
**Predecessor arc:** serving unification (ledger `.superpowers/sdd/progress.md`; goldens dir `~/dev/glades-trainer/logs/chiron-unify-goldens/`).

## Global Constraints

- Lib code is **C++98**: no `<random>`, no lambdas/auto/nullptr/range-for. The two sort-comparator lambdas in the ported sampler become functor structs.
- **Verbatim-port rule**: the sampler's penalty application order, candidate selection, `std::partial_sort`/`std::sort` calls (same algorithms — do NOT "upgrade" to stable_sort; tie permutations must match the current libstdc++ introsort behavior), softmax/CDF math, and the generation loop's window fill/slide semantics are preserved exactly. Any cleanup that changes observable behavior breaks gates G2–G4.
- **Bit-parity chain**: G1 (RNG streams) → G2 (sampler picks) → G3 (`--top-k 1` sequences) → G4 (same-seed stochastic sequences) → G5 (TF lines). Goldens captured BEFORE any refactor (Task 1).
- `ChironMt19937::next_canonical_double()` must replicate this machine's libstdc++ (`/usr/include/c++/*/bits/random.h`, `random.tcc`): `std::uniform_real_distribution<double>(0,1)` over `std::mt19937` — including `generate_canonical<double,53>`'s exact draw count, combine arithmetic, and any ≥1.0 retry guard present in the installed version. Read the installed headers; do not implement from memory.
- Build/run rules (unchanged from last arc): after ANY glades-ml change, `cd ~/dev/glades-ml/build && make install` THEN `cd ~/dev/glades-trainer && bash build.sh`. GPU runs sequential, Bash timeouts up to 600000 ms. NEVER write into `database/checkpoints/`.
- Production checkpoints and their serving flags: PIED `database/checkpoints/chiron_1B_pied_e4/chiron_1B_pied_e4.final` (`--whisc-coupling --rot-theta-max 0.07`); reanchor `database/checkpoints/chiron_1B_T16384_reanchor5B_finish.final` (no whisc flags). TF references: PIED 1.0897/0.7176, reanchor 1.7798.
- Commit after every task, in the repo touched.

---

## Task 1: Golden capture with the CURRENT binary (trainer)

**Files:**
- Modify: `~/dev/glades-trainer/tools/chiron_infer.cpp` (additive only: `--dump-tokens` flag + `CHIRON_SAMPLER_GOLDEN` selftest block)
- Output: `~/dev/glades-trainer/logs/chiron-unify-goldens/gen/` (untracked artifacts + README section)

**Interfaces:**
- Produces: golden files consumed by Tasks 3, 4, 9 — `rng_streams.txt` (seeds 1337/2024/4242 × 64 canonical doubles as `%a` hex), `sampler_matrix.txt` (config-matrix picks), `{pied,reanchor}.{stoch,topk1}.tokens.bin` + `.txt` + `.metrics` (64-token sequences), and the dump-tokens format: `int32 count` then `count` int32 token IDs.

- [ ] **Step 1: Add `--dump-tokens`** (generated-token IDs, written after the loop). In the variable block after `dumpLogitsPath` (line ~226):

```cpp
	std::string dumpTokensPath; // 2026-07-03 gen-parity harness: generated token IDs (int32)
```

CLI parse (after `--dump-logits`, line ~282):

```cpp
		else if (streq(argv[i], "--dump-tokens") && i + 1 < argc) dumpTokensPath = argv[++i];
```

In the `generate` lambda, generated-token tracking + dump. After `std::printf("\n\n");` (line ~434), insert (note: `tokens` currently holds prompt+generated; the generated slice starts at the pre-loop length — capture it before the loop):

At the top of the lambda body, right after the prompt/preTokens branch sets up `tokens` (after line ~380):
```cpp
		const size_t genStart = tokens.size();
```
After `std::printf("\n\n");`:
```cpp
		if (!dumpTokensPath.empty())
		{
			std::FILE* df = std::fopen(dumpTokensPath.c_str(), "wb");
			if (!df) { std::fprintf(stderr, "chiron_infer: cannot open %s\n", dumpTokensPath.c_str()); }
			else
			{
				const int32_t n = (int32_t)(tokens.size() - genStart);
				std::fwrite(&n, sizeof(int32_t), 1, df);
				for (size_t di = genStart; di < tokens.size(); ++di)
				{ int32_t t32 = (int32_t)tokens[di]; std::fwrite(&t32, sizeof(int32_t), 1, df); }
				std::fclose(df);
				std::printf("[dump-tokens] wrote %s (%d generated tokens)\n", dumpTokensPath.c_str(), (int)n);
			}
		}
```

- [ ] **Step 2: Add the `CHIRON_SAMPLER_GOLDEN` selftest block** right after the existing `CHIRON_INFER_SELFTEST` block (line ~50). It must call the REAL `sampleToken` — so it lives in this file, compiled standalone. Note: `sampleToken` and `gen_degeneration_metrics` are inside the anonymous namespace below; the golden main goes AFTER the namespace close (line ~186), guarded so the normal `main` is excluded:

```cpp
#ifdef CHIRON_SAMPLER_GOLDEN
// Golden capture for the mt19937 lib port (2026-07-03 full-inference arc).
// Build: g++ -std=c++11 -O2 -DCHIRON_SAMPLER_GOLDEN -I. tools/chiron_infer_golden_stub.cpp  — see Step 3.
int main()
{
	// (a) RNG streams: first 64 canonical doubles per seed, printed as %a hex
	//     (exact bit pattern; %.17g would round-trip too but %a is unambiguous).
	const unsigned int seeds[3] = { 1337u, 2024u, 4242u };
	for (int s = 0; s < 3; ++s)
	{
		std::mt19937 rng(seeds[s]);
		std::uniform_real_distribution<double> U(0.0, 1.0);
		std::printf("seed %u\n", seeds[s]);
		for (int i = 0; i < 64; ++i) std::printf("%a\n", U(rng));
	}
	// (b) Sampler matrix: V=64 synthetic logits, logit[i] = sin(0.37*i)*4.
	//     History = {3,7,3,9,3,7} (triggers penalties + the {3,7}->3 ngram ban at n=3).
	std::vector<float> lg(64);
	for (int i = 0; i < 64; ++i) lg[i] = std::sin(0.37 * i) * 4.0f;
	std::vector<int> hist; hist.push_back(3); hist.push_back(7); hist.push_back(3);
	hist.push_back(9); hist.push_back(3); hist.push_back(7);
	struct Cfg { float t; int k; float p; int rw; float rp, fp, pp; int nn; const char* name; };
	const Cfg cfgs[6] = {
		{0.8f, 40, 0.95f, 256, 1.0f, 1.2f, 0.4f, 3, "defaults"},
		{0.8f, 40, 0.95f,   0, 1.0f, 0.0f, 0.0f, 0, "no-penalties"},
		{0.8f,  1, 0.95f, 256, 1.0f, 1.2f, 0.4f, 3, "topk1"},
		{0.8f,  0, 0.50f, 256, 1.0f, 1.2f, 0.4f, 3, "topp-only"},
		{1.5f,  8, 1.00f, 256, 1.5f, 0.7f, 0.2f, 2, "hot-ngram2"},
		{0.2f, 64, 0.99f,  4,  1.0f, 2.0f, 1.0f, 0, "cold-heavy-pen"},
	};
	for (int c = 0; c < 6; ++c)
	{
		std::mt19937 rng(1337u);
		std::printf("cfg %s:", cfgs[c].name);
		std::vector<int> h(hist);
		for (int i = 0; i < 16; ++i)
		{
			int pick = sampleToken(lg, cfgs[c].t, cfgs[c].k, cfgs[c].p, rng, h,
			                       cfgs[c].rw, cfgs[c].rp, cfgs[c].fp, cfgs[c].pp, cfgs[c].nn);
			std::printf(" %d", pick);
			h.push_back(pick);
		}
		std::printf("\n");
	}
	return 0;
}
#endif
```

Also move `sampleToken` and `gen_degeneration_metrics` OUT of the anonymous namespace (make them file-static as they already are — `static` suffices; the anonymous namespace wrapper can keep `streq` only) so the golden main can call them. Behavior-neutral.

- [ ] **Step 3: Build the golden tool + normal binary; capture.** The golden main conflicts with the file's normal main; the normal main is NOT compiled when `CHIRON_SAMPLER_GOLDEN` is defined — wrap the normal `int main(int argc, char** argv)` in `#ifndef CHIRON_SAMPLER_GOLDEN` / matching `#endif` at end of file (and the `CHIRON_INFER_SELFTEST` main already coexists — keep exclusions consistent: golden build also defines nothing else). Compile standalone (no CUDA needed for the sampler — but the file includes chiron headers; the stubs make it link-light. If linking is heavy, extract via `-DCHIRON_SAMPLER_GOLDEN` + linking `libglades` like the normal binary — simplest: just build the full chiron_infer with the define through a one-off g++ line mirroring the CMake flags, or temporarily via CMake. Choose the least invasive route and document it):

```bash
cd ~/dev/glades-trainer && bash build.sh    # normal binary with --dump-tokens
mkdir -p logs/chiron-unify-goldens/gen && G=logs/chiron-unify-goldens/gen
# golden tool (document the exact command you used in the README):
g++ -std=c++11 -O2 -DCHIRON_SAMPLER_GOLDEN -DGLADES_HAVE_CUDA=0 tools/chiron_infer.cpp trainer/bpe.cpp \
    $(pkg-config --cflags-only-I 2>/dev/null; echo -I"$HOME/.local/include/glades") \
    -L"$HOME/.local/lib" -lglades ... -o /tmp/claude-1000/-home-robert-dev-glades-ml/d588a29b-1de0-4805-b992-4cebf0ba59af/scratchpad/sampler_golden || true
```
If the standalone link fights you for more than ~15 minutes, STOP fighting and instead add a hidden `--sampler-golden` CLI flag to the normal binary (running the same golden main body then exiting 0) — equally valid capture, remove-later noted in README. Then:

```bash
/tmp/claude-1000/-home-robert-dev-glades-ml/d588a29b-1de0-4805-b992-4cebf0ba59af/scratchpad/sampler_golden > $G/rng_and_sampler.txt
# split for clarity (optional): head = rng_streams, tail = sampler_matrix
```

- [ ] **Step 4: End-to-end sequence goldens** (4 runs, each a multi-minute GPU generation of 64 tokens = 64 full-T forwards — up to ~10 min each; be patient):

```bash
cd ~/dev/glades-trainer && G=logs/chiron-unify-goldens/gen
B="./build/chiron_infer --vocab-file pretok-data/vocab.bpe --tokens-file pretok-data/val.tok.bin --tokens-file-n 2048 --max-tokens 64 --seed 1337 --gen-metrics"
P="--model database/checkpoints/chiron_1B_pied_e4/chiron_1B_pied_e4.final --whisc-coupling --rot-theta-max 0.07"
R="--model database/checkpoints/chiron_1B_T16384_reanchor5B_finish.final"
$B $P --dump-tokens $G/pied.stoch.tokens.bin  | tee $G/pied.stoch.txt
$B $P --top-k 1 --dump-tokens $G/pied.topk1.tokens.bin | tee $G/pied.topk1.txt
$B $R --dump-tokens $G/reanchor.stoch.tokens.bin | tee $G/reanchor.stoch.txt
$B $R --top-k 1 --dump-tokens $G/reanchor.topk1.tokens.bin | tee $G/reanchor.topk1.txt
```
Determinism check: re-run the first command with `--dump-tokens $G/pied.stoch.rerun.bin`; `cmp` must match (same seed → same stream → same tokens; forward is deterministic per last arc). If it does NOT match, STOP and report — the stochastic gate design is invalid.

- [ ] **Step 5: README section** in `logs/chiron-unify-goldens/README.md`: date, SHAs, the golden-tool build command used, all capture commands, determinism result.

- [ ] **Step 6: Commit (trainer)** — `chiron_infer: --dump-tokens + sampler-golden harness (pre-refactor gen goldens)`.

---

## Task 2: `chiron_generate.h` frozen API + stubs (lib)

**Files:**
- Create: `~/dev/glades-ml/Backend/Machine Learning/Networks/chiron_generate.h`, `chiron_generate.cpp`
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/CMakeLists.txt` (add `chiron_generate.cpp` after `chiron_serving.cpp`)

**Interfaces:**
- Consumes: `chiron_serving.h` types (ChironModelDims/Weights/ServingConfig/EvalScratch, chiron_eval_forward).
- Produces (frozen — later tasks use verbatim):

```cpp
// chiron_generate.h — CHIRON token generation, sampling, TF eval.
// Single source of truth for CHIRON inference-side generation (2026-07-03 arc).
// C++98.
#ifndef _GLADES_CHIRON_GENERATE_H_
#define _GLADES_CHIRON_GENERATE_H_

#include <stdint.h>
#include <cstddef>
#include <vector>
#include "chiron_serving.h"

namespace glades {
namespace chiron {

// C++98 MT19937 bit-compatible with std::mt19937, plus a canonical-double
// generator bit-compatible with this toolchain's
// std::uniform_real_distribution<double>(0,1) over std::mt19937
// (libstdc++ generate_canonical<double,53>: TOOLCHAIN-COUPLED — pinned by the
// rng-stream goldens in the chiron-generate unit test; a libstdc++ behavior
// change would shift stochastic streams, not correctness).
struct ChironMt19937
{
	uint32_t mt[624];
	int mti;
	explicit ChironMt19937(uint32_t seed);
	uint32_t next_u32();
	double next_canonical_double();   // one variate == one U(0,1) draw of the old sampler
};

struct ChironGenParams
{
	int maxTokens;        // 100
	float temperature;    // 0.8f
	int topK;             // 40
	float topP;           // 0.95f
	int repWindow;        // 256
	float repPenalty;     // 1.0f
	float freqPenalty;    // 1.2f
	float presPenalty;    // 0.4f
	int noRepeatN;        // 3
	uint32_t seed;        // 1337
	ChironGenParams();    // sets exactly the defaults above (today's CLI defaults)
};

// Streaming sink: called once per generated token; return false to stop early.
typedef bool (*ChironTokenSink)(void* ctx, int token);

// Verbatim port of chiron_infer::sampleToken (penalties -> ngram ban ->
// temperature/softmax -> top-k -> top-p -> single CDF draw).
int chiron_sample_token(const std::vector<float>& logitsIn,
                        const ChironGenParams& gp,
                        const std::vector<int>& context,
                        ChironMt19937& rng);

// Generation loop (pad-to-T window, keep-last-T slide, one eval forward per
// token, sample at position useLen-1, append, emit to sink).  promptTokens are
// clamped to [0,V) as today.  Returns false on forward/scratch failure.
// outTokens (optional) receives ONLY the generated tokens.
bool chiron_generate(const ChironModelDims& dims, const ChironModelWeights& w,
                     const ChironServingConfig& cfg, ChironEvalScratch& s,
                     const std::vector<int>& promptTokens,
                     const ChironGenParams& gp,
                     ChironTokenSink sink, void* sinkCtx,
                     std::vector<int>* outTokens);

struct ChironTfResult
{
	long positions;
	double top1Acc;
	double meanNll;
	ChironTfResult() : positions(0), top1Acc(0.0), meanNll(0.0) {}
};

// Teacher-forcing eval: one forward over the (padded) window, per-position
// argmax + double-precision log-sum-exp NLL vs tokens[i+1].
// logitsAllOut (optional): receives the full [T,V] host logits from the
// forward (for --dump-logits) so callers need not run a second forward.
bool chiron_tf_eval(const ChironModelDims& dims, const ChironModelWeights& w,
                    const ChironServingConfig& cfg, ChironEvalScratch& s,
                    const std::vector<int>& tokens,
                    ChironTfResult& out,
                    std::vector<float>* logitsAllOut);

// distinct4 = unique 4-grams / total; maxRun = longest identical-token run.
void chiron_degeneration_metrics(const std::vector<int>& gen,
                                 double& distinct4, int& maxRun);

} // namespace chiron
} // namespace glades

#endif
```

- [ ] **Step 1:** Write the header verbatim; stub .cpp (functions return false/0/no-op; `ChironGenParams()` ctor sets the REAL defaults; `ChironMt19937` ctor zero-fills and sets mti=625 — stub); register in CMake.
- [ ] **Step 2:** Build: `cd ~/dev/glades-ml && sh .configure.sh cuda` — clean.
- [ ] **Step 3:** Commit (lib) — `chiron_generate: module skeleton (frozen API: mt19937, sampler, generate, tf-eval)`.

---

## Task 3: ChironMt19937 (TDD, gates G1)

**Files:**
- Modify: `~/dev/glades-ml/Backend/Machine Learning/Networks/chiron_generate.cpp`
- Create: `~/dev/glades-ml/unit-tests/Backend/Machine Learning/chiron-generate-test.h`, `chiron-generate-test.cpp`
- Modify: `unit-tests/main.cpp` (dispatch `chiron-generate`) + the unit-tests CMake source list

**Interfaces:**
- Consumes: Task 1's `rng_and_sampler.txt` golden values (transcribed as hard-coded constants) at `~/dev/glades-trainer/logs/chiron-unify-goldens/gen/`.
- Produces: working `ChironMt19937`; test entry `CHIRONGenerateUnitTest()` via `bash test.sh chiron-generate`.

- [ ] **Step 1: Failing test.** In `chiron-generate-test.cpp`: `CHIRONMt19937GoldenTest()` — for each of the three seeds, construct `ChironMt19937`, draw 64 `next_canonical_double()`, and ASSERT bit-equality against the Task-1 goldens. Transcribe the `%a` hex values from `rng_and_sampler.txt` into a static table (parse hex-float literals with `strtod` — C99/available — or store as uint64 bit patterns via a small converter you run once; document which). Also assert the first three raw `next_u32()` values for the default MT reference seed 5489 against the published reference (3499211612u, 581869302u, 3890346734u) to catch core-generator bugs independently of the combine. Register the suite (dispatch next to `chiron-model` in `unit-tests/main.cpp`; add the source file to the unit-tests CMake list).
- [ ] **Step 2:** Run `cd unit-tests/build && sh .configure.sh cuda && cd .. && bash test.sh chiron-generate` → FAIL (stub).
- [ ] **Step 3: Implement.** Read this machine's `/usr/include/c++/*/bits/random.h` + `random.tcc`: (1) mt19937 seeding (`_M_x[0]=seed; _M_x[i] = 1812433253u*(_M_x[i-1]^(_M_x[i-1]>>30))+i`), twist, and tempering — replicate; (2) `uniform_real_distribution<double>::operator()` for (0,1) — it routes through `__detail::_Adaptor`/`generate_canonical<double, digits, mt19937>`; replicate the EXACT draw count (2 for 53 bits over a 32-bit engine), combine loop, divisor, and the installed version's ≥1.0 handling (some versions clamp/retry — copy what is actually there). Cite the header path + libstdc++ version in a comment.
- [ ] **Step 4:** `bash test.sh chiron-generate` → PASS. (G1 banked.)
- [ ] **Step 5:** Commit (lib) — `chiron_generate: C++98 MT19937 + libstdc++-compatible canonical double (G1 golden-exact)`.

## Task 4: chiron_sample_token (TDD, gates G2)

**Files:**
- Modify: `chiron_generate.cpp`; `unit-tests/Backend/Machine Learning/chiron-generate-test.cpp`

**Interfaces:**
- Consumes: `ChironMt19937` (Task 3); Task-1 sampler-matrix goldens.
- Produces: working `chiron_sample_token` (frozen signature, Task 2).

- [ ] **Step 1: Failing test.** `CHIRONSamplerGoldenTest()` replicating Task 1's Step-2 matrix EXACTLY in lib terms: same V=64 `sin(0.37*i)*4` logits, same history `{3,7,3,9,3,7}`, same 6 configs, `ChironMt19937 rng(1337)` per config, 16 sequential picks appended to history — ASSERT each pick equals the golden (transcribed table from `rng_and_sampler.txt`). Map the config tuples onto `ChironGenParams` fields (t→temperature, k→topK, p→topP, rw→repWindow, rp/fp/pp→penalties, nn→noRepeatN).
- [ ] **Step 2:** RED.
- [ ] **Step 3: Port** `sampleToken` from `~/dev/glades-trainer/tools/chiron_infer.cpp:80–182` verbatim: penalties block (92–110, order: freqPenalty per occurrence → presence once → repPenalty divide/multiply), ngram ban (115–126, `-1e30f`), temperature+softmax (129–137), top-k via `std::partial_sort` (140–151), top-p via `std::sort` (154–170), single `next_canonical_double()` + CDF walk (173–181). The two comparator lambdas become functor structs holding `const std::vector<double>& p`. `(temperature > 0.0f) ? 1.0f/temperature : 1.0f` preserved exactly.
- [ ] **Step 4:** GREEN (G2 banked). Also run `bash test.sh chiron-model` (regression).
- [ ] **Step 5:** Commit (lib) — `chiron_generate: sampler port (penalties/ngram/topk/topp, G2 golden-exact)`.

## Task 5: chiron_tf_eval + degeneration metrics (TDD)

**Files:**
- Modify: `chiron_generate.cpp`; `chiron-generate-test.cpp`

**Interfaces:**
- Consumes: `chiron_eval_forward` (existing); tiny-shape test fixtures already established in `chiron-model-test.cpp` (reuse the same weight-fill pattern — copy the helper into this test file if sharing is awkward; do NOT refactor chiron-model-test).
- Produces: working `chiron_tf_eval` (with `logitsAllOut`) and `chiron_degeneration_metrics`.

- [ ] **Step 1: Failing tests.** (a) `CHIRONDegenMetricsTest()` — port the two cases from the `CHIRON_INFER_SELFTEST` block (40×token-2 → d4<0.1, run≥30; varied `(i*7+3)%50` → d4>0.8, run≤2) plus empty and <4-length edge cases (d4==1.0). (b) `CHIRONTfEvalTest()` — tiny shape (T=8, m=4, L=2, nH=1, dH=4, V=16, dense+fuseReln, same deterministic fills as the chiron-model eval-parity test): run `chiron_tf_eval` on a fixed 8-token window; independently compute expected NLL/top1 in the test by downloading logits via a direct `chiron_eval_forward` call and running a hand-written double log-sum-exp loop (`target = tokens[i+1]`, positions = useLen-1); ASSERT meanNll/top1Acc bit-equal (same math, same order) and `logitsAllOut` equals the direct download element-exact.
- [ ] **Step 2:** RED.
- [ ] **Step 3: Implement.** `chiron_degeneration_metrics`: port `chiron_infer.cpp:57–77` verbatim. `chiron_tf_eval`: port the tf-check math from `chiron_infer.cpp:476–511` — pad window (`useLen = min(tokens.size, T)`, zeros elsewhere), upload `s.d_tokens`, `chiron_eval_forward`, download `[T,V]` host logits (into `*logitsAllOut` if given, else a local), then the exact loop: per position `i < useLen-1`, argmax scan, `mx` scan, `Z` sum of `exp(row[v]-mx)`, `nllSum += -(row[tgt]-mx-log(Z))` with `tgt = tokens[i+1]`. Fill `ChironTfResult`.
- [ ] **Step 4:** GREEN; `bash test.sh chiron-model` regression.
- [ ] **Step 5:** Commit (lib) — `chiron_generate: tf-eval (with logits out-param) + degeneration metrics`.

## Task 6: chiron_generate loop (TDD)

**Files:**
- Modify: `chiron_generate.cpp`; `chiron-generate-test.cpp`

**Interfaces:**
- Consumes: everything above.
- Produces: working `chiron_generate` (frozen signature).

- [ ] **Step 1: Failing tests** at the Task-5 tiny shape: (a) `topK=1` determinism — run `chiron_generate` (maxTokens=6, prompt = 3 fixed tokens) and compare against a reference loop composed IN THE TEST from the same primitives (window fill → `chiron_eval_forward` → row at useLen-1 → `chiron_sample_token` → append): token vectors identical; (b) window-slide — prompt longer than T=8 (e.g. 12 tokens): assert the reference and the API agree (exercises `offset = size-useLen`); (c) sink early-stop — sink returns false after 2 tokens: `outTokens->size()==2` wait: current loop has no early-stop (sink is NEW behavior at the API boundary; the CLI sink always returns true) — assert generation stops after the sink refuses, and outTokens contains exactly the tokens emitted BEFORE the refusal plus the refused one or not — DEFINE: the refused token IS generated (appended to outTokens) but no further iterations run. Assert that. (d) token-id clamp: prompt containing `-5` and `V+3` → both fed as 0 into the window (reference agrees by construction).
- [ ] **Step 2:** RED.
- [ ] **Step 3: Implement** — port the generate-lambda core from `chiron_infer.cpp:382–433` minus CLI concerns: window fill/pad/slide (`input[i]=0` reset each step, clamp, `offset`), upload, forward (return false on failure after emitting nothing further), download `[T,V]`, copy row `useLen-1`, `chiron_sample_token` with the FULL accumulated `tokens` (prompt+generated — exactly what the old code passed as `context`), append, sink emit, honor sink=false stop. Seed the engine once at entry: `ChironMt19937 rng(gp.seed)`. Keep the `CHIRON_DBG` top-5 debug block (env-gated, harmless in lib).
- [ ] **Step 4:** GREEN; commit (lib) — `chiron_generate: generation loop (window slide, sink streaming)`. Then `cd build && make install`.

---

## Task 7: chiron_infer thin-out #2 (trainer)

**Files:**
- Modify: `~/dev/glades-trainer/tools/chiron_infer.cpp`

**Interfaces:**
- Consumes: the full `chiron_generate.h` API.

- [ ] **Step 1: Rewrite.** DELETE: `sampleToken` (80–182), `gen_degeneration_metrics` (57–77) + its `CHIRON_INFER_SELFTEST` block, the `CHIRON_SAMPLER_GOLDEN` block (goldens now live in lib tests — note removal in the commit body), `std::mt19937 rng(seed)` (365), `<random>`/`<set>` includes if now unused. KEEP: all CLI flags incl. `--dump-tokens`/`--dump-logits`, BPE, tokens-file reading, seed-tail slicing, REPL. REPLACE:
  - The `generate` lambda's loop body with:
```cpp
		glades::chiron::ChironGenParams gp;
		gp.maxTokens = maxTokens; gp.temperature = temperature; gp.topK = topK; gp.topP = topP;
		gp.repWindow = repWindow; gp.repPenalty = repPenalty; gp.freqPenalty = freqPenalty;
		gp.presPenalty = presPenalty; gp.noRepeatN = noRepeatN; gp.seed = seed;
		std::vector<int> genTokens;
		if (!glades::chiron::chiron_generate(dims, w, cfg, s, tokens, gp,
		                                     &streamSink, (void*)&tok, &genTokens))
			std::fprintf(stderr, "\n[chiron-infer] generation failed\n");
```
    with a file-static sink that decodes+streams (matching today's per-token printf+fflush):
```cpp
static bool streamSink(void* ctx, int token)
{
	trainer::BPETokenizer* tk = (trainer::BPETokenizer*)ctx;
	std::vector<int> one(1, token); std::string piece;
	tk->decode(one, piece);
	std::printf("%s", piece.c_str()); std::fflush(stdout);
	return true;
}
```
    `[gen-metrics]` and `--dump-tokens` now consume `genTokens` (which is exactly the generated slice — the old `genStart` bookkeeping goes away). The preamble printfs (`prompt=… generating…`) stay.
  - The tf-check block: `chiron_tf_eval(dims, w, cfg, s, seedTokens, tf, dumpLogitsPath.empty() ? 0 : &logitsAll)` then the existing dump-logits writer consumes `logitsAll` (`useLen` recomputed as `min(seedTokens.size, T)`), and the `[tf-check]` printf formats from `tf.positions/top1Acc/meanNll` with the IDENTICAL format string (`positions=%ld  top1_acc=%.4f  mean_nll=%.4f …`) and the VERDICT line unchanged.
- [ ] **Step 2:** `make install` already done (Task 6); `bash build.sh`; expect ~300-line file.
- [ ] **Step 3:** Quick check: PIED `--tf-check` run → `[tf-check]` line matches golden (1.0897/0.7176).
- [ ] **Step 4:** Commit (trainer) — `chiron_infer: generation/sampling/tf-eval via glades::chiron (thin-out #2)`.

## Task 8: chiron_parity switches to chiron_tf_eval (trainer)

**Files:**
- Modify: `~/dev/glades-trainer/tools/chiron_parity.cpp`

- [ ] **Step 1:** Replace its private forward+NLL loop with `chiron_tf_eval(..., tf, 0)`; keep the `[parity] positions=%ld top1_acc=%.6f mean_nll=%.6f` format verbatim.
- [ ] **Step 2:** Build; run `bash scripts/chiron_parity_check.sh` → PASS with `mean_nll=1.089714` (identical printed value — same math relocated).
- [ ] **Step 3:** Commit (trainer) — `chiron_parity: use lib chiron_tf_eval (kills the duplicated NLL loop)`.

## Task 9: GATES G3/G4/G5/G6 (verification only)

**Files:** none (results appended to the goldens README)

- [ ] **Step 1 (G3+G4):** Re-run all four Task-1 Step-4 capture commands with the NEW binary, dumping to `$G/{name}.post.tokens.bin` / `.post.txt`; then:
```bash
cd ~/dev/glades-trainer/logs/chiron-unify-goldens/gen
for n in pied.stoch pied.topk1 reanchor.stoch reanchor.topk1; do cmp $n.tokens.bin $n.post.tokens.bin && echo "$n TOKENS-IDENTICAL"; done
diff <(grep gen-metrics pied.stoch.txt) <(grep gen-metrics pied.stoch.post.txt)
```
Expected: four `TOKENS-IDENTICAL`, metrics lines equal. **G4 (same-seed stochastic identity) is the arc's headline gate.** Any mismatch: STOP, report BLOCKED with the first diverging token index (`cmp -l`), and note that Tasks 3–4's golden tests passing while G4 fails implicates the loop's RNG consumption pattern (e.g. an extra draw) — do not fix without diagnosis.
- [ ] **Step 2 (G5):** `[tf-check]` (PIED + reanchor) and `[parity]` lines equal to their baselines; `--dump-logits` cmp vs the Task-10 (previous arc) goldens still byte-identical (forward untouched — cheap insurance).
- [ ] **Step 3 (G6):** `bash test.sh chiron-generate && bash test.sh chiron-model && bash test.sh chiron` (lib); `sh scripts/chiron_serving_interlocks.sh`; `sh runner.sh --flagship --tf-check --tokens-file pretok-data/val.tok.bin | grep tf-check`.
- [ ] **Step 4:** Append results + SHAs to the README. No commit needed if the README stays untracked (confirm as before).

## Task 10: Docs + final sweep

**Files:**
- Modify: `~/dev/glades-ml/CLAUDE.md` (test-names list += `chiron-generate`; the CHIRON serving-modules subsection gains one short paragraph: chiron_generate module, mt19937 toolchain-coupling note, TF-eval single source)
- Modify: `~/dev/glades-trainer/CLAUDE.md` (chiron_infer is now CLI-only — BPE/tokens-file/flags; generation/sampling/TF-eval live in the lib; goldens/gen harness location)

- [ ] **Step 1:** Doc edits (~8 lines each, match style).
- [ ] **Step 2:** Re-run the two suites most coupled to the docs claims (`test.sh chiron-generate`, interlock script) as a final spot check.
- [ ] **Step 3:** Commits both repos — `docs: chiron_generate module (full-inference move)` / `docs: chiron_infer is CLI-only; gen goldens harness`. Controller handles ledger/memory.

---

## Self-review notes (spec coverage)

- Spec §3 API → Task 2 (frozen header, incl. the `logitsAllOut` addition surfaced during planning — the dump-logits path needs raw logits without a second forward). §2.2/§7 RNG port + toolchain pinning → Tasks 1, 3. Sampler → Task 4. TF-eval consolidation → Tasks 5, 7, 8. Generation loop → Tasks 6, 7. §6 G0→Task 1, G1→Task 3, G2→Task 4, G3/G4→Task 9, G5→Tasks 7/8/9, G6→Task 9. §5 exclusions honored (no KV/BPE/CLI-move/batching tasks).
- Sink early-stop semantics (spec silent): defined in Task 6 Step 1(c) — refused token still appended, loop stops. CLI sink always returns true, so CLI behavior is unaffected.
- Type consistency: `ChironGenParams` fields used in Tasks 4/6/7 match Task 2; `chiron_tf_eval(..., ChironTfResult&, std::vector<float>*)` consistent across Tasks 2/5/7/8.
