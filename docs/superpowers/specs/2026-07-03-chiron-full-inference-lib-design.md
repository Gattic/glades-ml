# CHIRON Full Inference Move to glades-ml — Design

**Date:** 2026-07-03
**Status:** Approved (owner sign-off in session)
**Repos:** glades-ml (lib) + glades-trainer (consumer)
**Predecessor:** `2026-07-03-chiron-serving-lib-unification-design.md` (shipped: lib owns
checkpoint format, serving resolution, eval forward; bit-identity gate PASS ×3 checkpoints)

## 1. Problem

After the serving unification, the remaining CHIRON *inference* code still lives in the
trainer: the sampler (`sampleToken` — temperature/top-k/top-p plus the repetition controls
added 2026-06-20: rep-window/rep-penalty/freq-penalty/presence-penalty/no-repeat-ngram),
the generation loop (window fill/slide, per-token full-T forward, streaming decode), the
teacher-forcing eval (now duplicated between `chiron_infer` --tf-check and
`tools/chiron_parity.cpp` — duplication created by the previous arc), and the degeneration
metrics. The lib's existing `sampling_utils.h` has temperature/top-k/top-p over
`glades::rng` but none of the repetition controls, and no generation orchestration exists
for CHIRON.

## 2. Decisions (owner-approved)

1. **Scope**: move generation loop, sampler, TF-eval, and degeneration metrics into the
   lib as a token-in/token-out API. **BPE stays in the trainer** (shared with the training
   data pipeline via `bpe_train`/`pretokenize`; the lib's `tokenizer_artifacts.cpp`
   explicitly scopes the lib to vocab *metadata*, not tokenization *algorithms*).
   **The CLI binary stays in the trainer.**
2. **RNG**: port **MT19937 to C++98 inside the lib** so seeded stochastic generation is
   **bit-identical** to the current binary (owner choice over the glades::rng
   alternative). The current sampler consumes exactly one
   `std::uniform_real_distribution<double>(0,1)` draw per token
   (chiron_infer.cpp:173–174), so bit-identity requires MT19937 (standard-specified) plus
   a replication of libstdc++'s `generate_canonical<double,53>` combine (2×32-bit draws
   per variate, implementation-defined — pinned by golden streams, see §6).
3. **Approach**: new module `chiron_generate.{h,cpp}` (approach A) rather than growing
   `chiron_serving.cpp` (B) or wiring into `TransformerPublicAPI`/NNetwork dispatch (C —
   rejected as YAGNI; CHIRON deliberately bypasses NNetwork).

## 3. New lib module — `Backend/Machine Learning/Networks/chiron_generate.{h,cpp}`

`namespace glades::chiron`, C++98, Networks CMake target, GLADES_HAVE_CUDA-gated where it
touches the forward (same conventions as chiron_serving).

- **`ChironGenParams`**: `maxTokens, temperature, topK, topP, repWindow, repPenalty,
  freqPenalty, presPenalty, noRepeatN, seed` — constructor defaults equal today's CLI
  defaults (100, 0.8f, 40, 0.95f, 256, 1.0f, 1.2f, 0.4f, 3, 1337).
- **`ChironMt19937`**: C++98 MT19937 (624-word state, standard init-by-seed and
  tempering) + `next_canonical_double()` replicating libstdc++'s
  `generate_canonical<double, 53, mt19937>` exactly. Header comment documents the
  toolchain coupling (libstdc++ combine) and the golden-stream test that pins it.
- **`chiron_sample_token(const std::vector<float>& logitsRow, const ChironGenParams&,
  const std::vector<int>& history, ChironMt19937&)`** — verbatim port of
  `chiron_infer::sampleToken`: penalty application order, candidate selection, sort and
  tie behavior preserved exactly (lambdas → functors; `<random>` → ChironMt19937).
- **`ChironTokenSink`**: `typedef bool (*ChironTokenSink)(void* ctx, int token);`
  (return false = stop generation early). C-style callback per the lib's C++98 public-API
  conventions.
- **`chiron_generate(dims, w, cfg, scratch, const std::vector<int>& promptTokens,
  const ChironGenParams&, ChironTokenSink sink, void* sinkCtx,
  std::vector<int>* outTokens)`** — the generation loop ported verbatim from
  `chiron_infer`'s `generate` lambda: pad-to-T window, keep-last-T slide, one
  `chiron_eval_forward` per token, logits row at `useLen-1`, sample, append, emit to
  sink. Token-id clamping (`<0 || >=V → 0`) preserved. Prompt slicing policies
  (`--seed-tail`) remain caller-side.
- **`ChironTfResult`** { `long positions; double top1Acc; double meanNll;` } and
  **`chiron_tf_eval(dims, w, cfg, scratch, const std::vector<int>& tokens,
  ChironTfResult&)`** — the TF NLL/top1 computation (double-precision log-sum-exp,
  target `i+1`), consolidating the `chiron_infer` and `chiron_parity` copies.
- **`chiron_degeneration_metrics(const std::vector<int>& gen, double& distinct4,
  int& maxRun)`** — moved as-is.

## 4. Trainer changes

- **`tools/chiron_infer.cpp`** shrinks (~543 → ~300 lines): keeps CLI parsing, BPE
  encode/decode, tokens-file (.tok.bin) reading, `--seed-tail` slicing, interactive REPL,
  `--dump-logits`, and output formatting. The `generate` lambda body becomes a
  `chiron_generate` call with a printf/fflush streaming sink; `sampleToken`,
  `gen_degeneration_metrics`, and the tf-check math are deleted in favor of the lib
  calls. Printed `[tf-check]` and `[gen-metrics]` values must remain identical.
- **`tools/chiron_parity.cpp`**: TF-NLL loop replaced by `chiron_tf_eval`; the printed
  `[parity]` line values must remain identical.
- No build-system changes beyond the lib rebuild/install flow already in place.

## 5. Explicitly out of scope

- **KV-cache / incremental decoding** — per-token O(T²·L) full-window recompute stays;
  SCFA's full-window DCT basis makes incremental decode a research arc, not a refactor.
- **BPE tokenizer relocation** (decision §2.1) and **the CLI binary's location**.
- **Generation quality** — the repetition attractor is a training-side problem
  (`research/CHIRON_GENERATION_LOOP_FIX_2026_06_27.md`); this arc relocates code only.
- **Batch generation / serving-runtime integration** (TransformerPublicAPI-style
  ServingRuntime) — revisit only if a consumer materializes.

## 6. Parity gates (pre-registered, golden-first)

**G0 — capture with the CURRENT binary, before any refactor** (extends
`logs/chiron-unify-goldens/`):
  a. RNG-stream goldens: first 64 canonical doubles for seeds 1337/2024/4242, captured
     via a small temporary selftest built with the current toolchain's `<random>`
     (hard-coded into the lib unit test afterwards).
  b. Sampler goldens: fixed synthetic logits vectors × a config matrix (defaults;
     penalties-off; top-k 1; top-p only; no-repeat-ngram trigger case) → chosen tokens.
  c. End-to-end sequence goldens on PIED and reanchor: `--seed 1337 --max-tokens 64`
     stochastic default-params token sequences, and `--top-k 1` deterministic sequences
     (tokens-file-seeded, fixed offset).
**G1** — lib `ChironMt19937` reproduces (a) bit-exact.
**G2** — lib `chiron_sample_token` reproduces (b) exactly.
**G3** — refactored binary reproduces the `--top-k 1` sequences (c) exactly.
**G4** — **headline**: refactored binary reproduces the same-seed stochastic sequences
(c) exactly — token-for-token.
**G5** — `[tf-check]` and `[parity]` lines remain golden-exact (PIED 1.0897/0.7176 etc.)
after the TF-eval consolidation; `chiron_parity_check.sh` still PASSes.
**G6** — full regression: `test.sh chiron-model` (+ new generate tests) and the existing
chiron suites; `scripts/chiron_serving_interlocks.sh`; `runner.sh --flagship --tf-check`.

## 7. Risks & constraints

- **libstdc++ coupling**: `next_canonical_double` replicates this toolchain's combine;
  a future toolchain change would shift stochastic streams (not correctness). Documented
  in the header; G1 goldens detect it.
- **Tie-breaking**: top-k/top-p candidate ordering must match the current sort behavior;
  enforced by verbatim port + G2/G4.
- **C++98**: no `<random>`, no lambdas in ported code.
- **The forward is already parity-locked**: `chiron_eval_forward` is untouched by this
  arc; any G3/G4 failure localizes to RNG/sampler/loop by construction.

## 8. References

- Predecessor spec + ledger: `docs/superpowers/specs/2026-07-03-chiron-serving-lib-unification-design.md`,
  `.superpowers/sdd/progress.md`.
- Current code: `glades-trainer/tools/chiron_infer.cpp` (sampler ~81–190, generate loop
  ~365–445, tf-check ~1056+ per current layout), `tools/chiron_parity.cpp`;
  lib `sampling_utils.h` (pattern reference), `transformer_public_api.h` (API
  conventions), `rng.h` (why glades::rng was NOT chosen: owner decision §2.2).
