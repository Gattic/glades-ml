# CHIRON Generation-Loop Fix (Local Coherence) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make CHIRON generate locally-coherent continuations from a real-text context (no digit-loops / subword-soup) via an inference-side fix, without perturbing the validated forward.

**Architecture:** Diagnose-first. The whole failure is the teacher-forced-vs-autoregressive gap (TF top1 0.53, but the generate loop collapses). Phase 0 localizes it; Phase 1 fixes the identified inference cause; Phase 2 validates with a degeneration metric + a TF-unchanged regression. **Refinement discovered during grounding:** the in-distribution generation seeds a *full-T window that slides every step* (no trailing pad) — so the in-scope cause is the **window slide** against SCFA's absolute-position DCT basis, and the likely fix is to **grow the window instead of sliding** (honor the dead comment at `chiron_infer.cpp:1077-1078` that intended to seed `T−maxTokens`). Whether the (shrinking) trailing pad during growth is harmless is gated on **SCFA causality** (Phase-0 probe D2).

**Tech Stack:** C++98 + CUDA, glades-trainer `tools/chiron_infer.cpp` (the serving generate loop + `forwardInfer`), the in-tree glades-ml library (SCFA kernels, only if the causal-guard fallback is needed), `runner.sh`. Findings/runbook in glades-ml `research/`.

**Spec:** `docs/superpowers/specs/2026-06-27-chiron-generation-loop-fix-design.md`

---

## Repository layout & build

- Code change: **glades-trainer** `/home/robert/dev/glades-trainer/tools/chiron_infer.cpp`.
- Build: `cd /home/robert/dev/glades-trainer && cmake --build build --target chiron_infer -j"$(nproc)"`.
- Flagship checkpoint (the model under test): `database/checkpoints/chiron_1B_T16384_reanchor5B_finish.final`.
- Val tokens (uint16): `pretok-data/val.tok.bin`. Vocab: `pretok-data/vocab.bpe`.
- Reference (verified) numbers: `--tf-check` mean_nll **1.7798**, top1 **0.5346** (must stay unchanged — it exercises the forward, not the loop).
- Build gotcha (from the reanchor arc): the trainer links the **installed** glades library; if any glades-ml library file is touched (only in the contingent causal-guard task), run `cd /home/robert/dev/glades-ml/build && make install` before rebuilding chiron_infer.

## Key code facts (read these first)

- `generate()` lambda — `tools/chiron_infer.cpp:1004-1073`. Window fill at **1019-1034** (`useLen = min(tokens.size(), T)`, real tokens left-aligned `[0..useLen)`, rest zero-padded; `offset = tokens.size()-useLen` slides once `tokens.size() > T`). Logits read at `lastPos = useLen-1` (**1044-1048**). Existing `CHIRON_DBG` top-5 dump at **1050-1059**. Sample+append at **1061-1063**.
- tokensFile seeding — `1060-1076` builds `seedTokens` (all `want = min(avail, T)` tokens). Dispatch at **1126**: `generate("", &seedTokens)` passes the **full** seed (the dead comment at 1077-1078 says "use the last T−maxTokens" but the code does **not** slice). With a full-T seed, the loop slides from gen step 1.
- `--tf-check` branch — `1077-1124`, single forward, no pad, the validated path.
- `sampleToken` — `624+` (temperature/top-k/top-p + the existing repetition controls). Greedy = `--temperature 0`.
- `forwardInfer` — `499-620`; SCFA via `scfa_shear_infer` (line **553**); position structure is carried by SCFA's DCT basis (no separate RoPE/sinusoid in chiron_infer).

---

## Task 1: Diagnostic + degeneration-metric instrumentation

**Goal:** add the tools Phase 0 needs and the quantitative gate Phase 2 uses — a `--gen-metrics` readout, a `--seed-tail` option (grow-don't-slide; also the candidate fix path), and a unit-testable metric function. No behavior change to the default generate path yet.

**Files:**
- Modify: `tools/chiron_infer.cpp` — add a free function `gen_degeneration_metrics(...)` near `sampleToken` (~before line 624); add `--gen-metrics` and `--seed-tail` flag parsing (near the other flag parses ~794-809); call the metric at the end of `generate()` (~1072) when enabled; apply `--seed-tail` to the tokensFile seed (~1076/1126).
- Test: a standalone C++98 assertion in a `#ifdef CHIRON_INFER_SELFTEST` block + a one-shot compile/run (chiron_infer has no unit-test harness; use a tiny self-test main guard).

- [ ] **Step 1: Write the failing metric self-test**

Add near the top of `tools/chiron_infer.cpp` after the includes, a self-test that exercises the metric on a coherent vs degenerate token list. Put it behind `#ifdef CHIRON_INFER_SELFTEST` with its own `main`:

```cpp
#ifdef CHIRON_INFER_SELFTEST
// distinct-4-gram ratio + longest single-token run over a token list.
void gen_degeneration_metrics(const std::vector<int>& gen, double& distinct4, int& maxRun);
int main()
{
    // Degenerate: a digit loop -> low distinct4, long run.
    std::vector<int> deg; for (int i=0;i<40;++i) deg.push_back(2);
    double d4; int run; gen_degeneration_metrics(deg, d4, run);
    if (!(d4 < 0.1 && run >= 30)) { std::printf("FAIL deg: d4=%.3f run=%d\n", d4, run); return 1; }
    // Coherent-ish: varied tokens -> high distinct4, short run.
    std::vector<int> coh; for (int i=0;i<40;++i) coh.push_back((i*7+3)%50);
    gen_degeneration_metrics(coh, d4, run);
    if (!(d4 > 0.8 && run <= 2)) { std::printf("FAIL coh: d4=%.3f run=%d\n", d4, run); return 1; }
    std::printf("OK metrics self-test\n"); return 0;
}
#endif
```

- [ ] **Step 2: Run it to verify it fails (link error — function undefined)**

```bash
cd /home/robert/dev/glades-trainer
g++ -std=c++98 -DCHIRON_INFER_SELFTEST -x c++ tools/chiron_infer.cpp -o /tmp/ci_selftest 2>&1 | tail -3
```
Expected: FAIL — `undefined reference to gen_degeneration_metrics` (or many CUDA undefineds; if CUDA symbols dominate, instead compile *just* the metric by extracting it — see Step 3 note). The point: the assertion is written before the implementation.

- [ ] **Step 3: Implement the metric**

Add this free function near `sampleToken` (before line 624) in `tools/chiron_infer.cpp`:

```cpp
// Degeneration metrics over a generated token list (no model forward needed):
//   distinct4 = unique 4-grams / total 4-grams  (low => repetitive/collapsed)
//   maxRun    = longest run of identical consecutive tokens (high => stuck)
void gen_degeneration_metrics(const std::vector<int>& gen, double& distinct4, int& maxRun)
{
	maxRun = gen.empty() ? 0 : 1;
	int run = 1;
	for (size_t i = 1; i < gen.size(); ++i)
	{
		if (gen[i] == gen[i-1]) { ++run; if (run > maxRun) maxRun = run; }
		else run = 1;
	}
	if (gen.size() < 4) { distinct4 = 1.0; return; }
	std::set<std::string> seen;
	long total = 0;
	char buf[64];
	for (size_t i = 0; i + 3 < gen.size(); ++i)
	{
		std::snprintf(buf, sizeof(buf), "%d,%d,%d,%d", gen[i], gen[i+1], gen[i+2], gen[i+3]);
		seen.insert(std::string(buf));
		++total;
	}
	distinct4 = total ? (double)seen.size() / (double)total : 1.0;
}
```
Ensure `#include <set>` and `<string>` are present (add if missing).

- [ ] **Step 4: Run the self-test, verify PASS**

If Step 2's full-file compile is dominated by CUDA undefineds, compile the function in isolation: copy the function + the self-test `main` into `/tmp/metric_test.cpp` with `#include <vector><set><string><cstdio>` and run `g++ -std=c++98 /tmp/metric_test.cpp -o /tmp/mt && /tmp/mt`. Expected: `OK metrics self-test`. (This isolates the pure-C++ metric from the CUDA TU.)

- [ ] **Step 5: Wire `--gen-metrics` and `--seed-tail`**

Add flag parsing next to the existing flags (~794-809):
```cpp
		else if (streq(argv[i], "--gen-metrics")) genMetrics = true;
		else if (streq(argv[i], "--seed-tail")) seedTail = true;
```
Declare `bool genMetrics = false, seedTail = false;` with the other flag defaults (~772-790). At the end of `generate()` (after the gen loop, ~1071) add:
```cpp
		if (genMetrics)
		{
			const int seedLen = preTokens ? (int)preTokens->size() : 0;
			std::vector<int> gen(tokens.begin() + std::min((size_t)seedLen, tokens.size()), tokens.end());
			double d4; int run; gen_degeneration_metrics(gen, d4, run);
			std::printf("\n[gen-metrics] generated=%zu distinct4=%.3f max_token_run=%d\n", gen.size(), d4, run);
		}
```
Apply `--seed-tail` to the tokensFile seed: at line ~1126 (the `generate("", &seedTokens)` call), when `seedTail` is set, pass only the last `T - maxTokens` tokens so the window GROWS instead of sliding:
```cpp
		else if (seedTail) {
			int keep = dims.T - maxTokens; if (keep < 1) keep = 1;
			int off = (int)seedTokens.size() - keep; if (off < 0) off = 0;
			std::vector<int> tail(seedTokens.begin() + off, seedTokens.end());
			generate("", &tail);
		}
		else
			generate("", &seedTokens);
```

- [ ] **Step 6: Build chiron_infer (no behavior change with flags off)**

```bash
cd /home/robert/dev/glades-trainer && cmake --build build --target chiron_infer -j"$(nproc)" 2>&1 | tail -1
```
Expected: `Built target chiron_infer`. Sanity: `--tf-check` still prints mean_nll 1.78 (flags off → default path unchanged).

- [ ] **Step 7: Commit**

```bash
cd /home/robert/dev/glades-trainer
git add tools/chiron_infer.cpp
git commit -m "chiron_infer: --gen-metrics (distinct4/max-run) + --seed-tail (grow-don't-slide) for generation diagnosis"
```

---

## Task 2: Phase-0 diagnostics — localize the cause

**Goal:** run four probes and write a verdict (slide / SCFA-non-causal / sampling / exposure-bias) that selects the Phase-1 fix. Runbook + findings doc.

**Files:**
- Create: `research/CHIRON_GENERATION_LOOP_FIX_2026_06_27.md` (glades-ml) — record each probe's output + the verdict.

- [ ] **Step 1: D1/G1 — slide vs grow, greedy.** Two greedy runs from the same val context, with `--gen-metrics`:
```bash
cd /home/robert/dev/glades-trainer
# (a) full-T seed -> slides every step (the degenerate baseline)
./build/chiron_infer --model database/checkpoints/chiron_1B_T16384_reanchor5B_finish.final \
  --vocab-file pretok-data/vocab.bpe --tokens-file pretok-data/val.tok.bin \
  --max-tokens 120 --temperature 0 --gen-metrics 2>&1 | grep -E "gen-metrics" 
# (b) seed-tail -> window GROWS, no slide
./build/chiron_infer --model database/checkpoints/chiron_1B_T16384_reanchor5B_finish.final \
  --vocab-file pretok-data/vocab.bpe --tokens-file pretok-data/val.tok.bin \
  --max-tokens 120 --temperature 0 --seed-tail --gen-metrics 2>&1 | grep -E "gen-metrics"
```
**Interpretation:** (b) distinct4 high (>0.5) + short run while (a) is low/long → **the SLIDE is the bug** (grow-don't-slide is the fix). Both degenerate → not the slide; go to D2/G3/exposure.

- [ ] **Step 2: D2 — SCFA causality.** Does the trailing pad (future positions) affect the read-position logits? Run the grow case (`--seed-tail`, which has a trailing pad) with `CHIRON_DBG=1` and compare the gen step-0 top-5 logits when the pad is token-0 vs a copied last-real token. Add a tiny env-gated variant if needed, or reason from D1: **if (b) grow-with-pad is coherent, SCFA is effectively causal** (the pad didn't corrupt) → the simple grow-don't-slide fix is valid. If (b) ALSO degenerates despite no slide, suspect non-causal pad leakage → causal-guard fallback (Task 3 contingent).

- [ ] **Step 3: G3 — greedy vs sampled.** Re-run (a) and (b) at `--temperature 0.8 --top-k 40 --top-p 0.95 --gen-metrics`. If greedy (Step 1) already degenerates, it is not a sampling artifact. If greedy is clean but sampled degenerates → sampling/temperature issue (Task 3 sampling branch).

- [ ] **Step 4: D4 — confirm position-under-slide is the mechanism.** If D1 shows slide-degenerates/grow-coherent, that *is* the position-under-slide confirmation (sliding re-indexes SCFA's absolute DCT positions). Note it; no extra run needed.

- [ ] **Step 5: Write the verdict** into `research/CHIRON_GENERATION_LOOP_FIX_2026_06_27.md`: the four probe outputs (distinct4/max-run numbers + sample text), and the selected Phase-1 branch: **SLIDE** (→ grow-don't-slide), **NON-CAUSAL** (→ causal guard), **SAMPLING** (→ sampleToken), or **EXPOSURE-BIAS** (→ document + STOP, defer training).

- [ ] **Step 6: Commit the findings doc (glades-ml)**

```bash
cd /home/robert/dev/glades-ml
git add research/CHIRON_GENERATION_LOOP_FIX_2026_06_27.md
git commit -m "Phase-0 verdict: CHIRON generation-loop degeneration cause"
```

---

## Task 3: Phase-1 fix — the diagnosed cause

**Goal:** apply the fix the verdict selected. The primary (most-likely) branch is fully specified; the others are contingent.

**Files:**
- Modify: `tools/chiron_infer.cpp` — `generate()` seeding/loop.
- (CONTINGENT, only if NON-CAUSAL) Modify: glades-ml SCFA kernel — out of this file; gets its own focused sub-plan.

### Branch SLIDE (primary): make grow-don't-slide the default for in-distribution generation

- [ ] **Step 1: Promote `--seed-tail` behavior to the default tokensFile path** so `runner.sh --flagship --tokens-file ...` generates coherently without a special flag. At line ~1126, change the dispatch so the in-distribution generate always seeds the tail (grow-don't-slide) when `maxTokens < T`:
```cpp
		else {
			int keep = dims.T - maxTokens; if (keep < 1) keep = 1;
			int off = (int)seedTokens.size() - keep; if (off < 0) off = 0;
			std::vector<int> tail(seedTokens.begin() + off, seedTokens.end());
			generate("", &tail);   // grow the window; never slide for N<=T-seed
		}
```
Keep `--seed-tail` as a no-op alias (or remove it) — the behavior is now default. This honors the dead comment at 1077-1078.

- [ ] **Step 2: Guard against the slide regime explicitly.** Inside `generate()` (~1026), when the window would slide (`tokens.size() > T`) emit a one-time warning so future long-generation callers know they've left the validated regime:
```cpp
			if ((int)tokens.size() > dims.T && gen == 0)
				std::fprintf(stderr, "[chiron-infer] note: window now slides (gen beyond T-seed); "
				                     "local-coherence regime is N<=T-seed.\n");
```

- [ ] **Step 3: Build + quick check** (the grow path is now default):
```bash
cd /home/robert/dev/glades-trainer && cmake --build build --target chiron_infer -j"$(nproc)" 2>&1 | tail -1
./build/chiron_infer --model database/checkpoints/chiron_1B_T16384_reanchor5B_finish.final \
  --vocab-file pretok-data/vocab.bpe --tokens-file pretok-data/val.tok.bin \
  --max-tokens 120 --temperature 0 --gen-metrics 2>&1 | grep -E "gen-metrics"
```
Expected: distinct4 now high (matching Task-2 Step-1 case (b)).

- [ ] **Step 4: Commit**
```bash
cd /home/robert/dev/glades-trainer
git add tools/chiron_infer.cpp
git commit -m "chiron_infer: grow-don't-slide default for in-distribution generation (fixes slide-induced degeneration)"
```

### Branch NON-CAUSAL (contingent — only if Task-2 D2 shows pad leakage)

- [ ] **Step 1 (when triggered):** Re-enter scoping for an SCFA inference causal guard (zero the spectral-B and conv contributions of positions beyond the read position) in the glades-ml SCFA kernel. This is a library change with its own spec+plan; do NOT implement inline. Record the D2 evidence (logits move under pad change) in the findings doc and stop this branch.

### Branch SAMPLING (contingent — only if Task-2 G3 shows greedy-clean/sampled-degenerate)

- [ ] **Step 1 (when triggered):** the forward + grow fix are fine; the degeneration is decoding. Tighten the default decoding for `runner.sh --flagship` (e.g. lower default temperature, enable the existing `--freq-penalty`/`--no-repeat-ngram` by default) in `runner.sh` or chiron_infer defaults, and re-validate with `--gen-metrics`. Specify exact values from the G3 sweep.

### Branch EXPOSURE-BIAS (contingent — Task-2 says forward consistent, gradual drift)

- [ ] **Step 1 (when triggered):** STOP. Document in the findings doc that inference cannot fix it; scope generation-aware training as a separate plan (per the owner "defer training" decision). This plan delivers the diagnosis + any decoding improvement only.

---

## Task 4: Phase-2 validation

**Goal:** prove local coherence improved quantitatively, across contexts, with the forward unchanged.

**Files:**
- Modify: `research/CHIRON_GENERATION_LOOP_FIX_2026_06_27.md` — before/after table + samples.

- [ ] **Step 1: Degeneration metric, before vs after, multiple contexts.** Run the default (fixed) path greedy + sampled on ≥3 distinct val offsets (use `--tokens-file-n` to vary the window content), capture `--gen-metrics`:
```bash
cd /home/robert/dev/glades-trainer
for tn in 16384 100000 200000; do
  echo "=== context tokens-file-n=$tn (greedy) ==="
  ./build/chiron_infer --model database/checkpoints/chiron_1B_T16384_reanchor5B_finish.final \
    --vocab-file pretok-data/vocab.bpe --tokens-file pretok-data/val.tok.bin \
    --tokens-file-n $tn --max-tokens 120 --temperature 0 --gen-metrics 2>&1 | grep -E "gen-metrics"
done
```
**PASS:** distinct4 materially higher than the Task-2 Step-1(a) baseline (target distinct4 > 0.5, max_token_run < 10) and the streamed text is grammatical, non-degenerate (no digit-soup). Paste 1-2 samples into the doc.

- [ ] **Step 2: Regression — forward unchanged.** The fix is generation-only; `--tf-check` must be identical to the pre-fix value:
```bash
./build/chiron_infer --model database/checkpoints/chiron_1B_T16384_reanchor5B_finish.final \
  --vocab-file pretok-data/vocab.bpe --tf-check --tokens-file pretok-data/val.tok.bin 2>&1 | grep "tf-check\]"
```
**PASS:** `mean_nll=1.7798 top1_acc=0.5346` (unchanged). FAIL = the change leaked into the forward — revert and investigate.

- [ ] **Step 3: End-to-end via runner.sh.** Confirm the user-facing path benefits:
```bash
cd /home/robert/dev/glades-trainer
sh runner.sh --flagship --tokens-file pretok-data/val.tok.bin --max-tokens 120 --temperature 0 --gen-metrics 2>&1 | grep -E "gen-metrics|using checkpoint"
```
**PASS:** resolves to the reanchor flagship and prints improved metrics.

- [ ] **Step 4: Commit the before/after results**
```bash
cd /home/robert/dev/glades-ml
git add research/CHIRON_GENERATION_LOOP_FIX_2026_06_27.md
git commit -m "Gen-loop fix validation: distinct4 before/after + TF-unchanged regression"
```

---

## Task 5: Document & close

- [ ] **Step 1: Finalize the findings doc** — verdict, the fix (or the exposure-bias deferral), before/after metrics + samples, and a one-line note in CLAUDE.md's flagship "Caveats" if generation is now locally coherent (update "free generation is repetition-limited" to reflect the in-distribution-coherent / short-prompt-still-OOD nuance). If exposure-bias branch: note generation remains model-limited and point to the deferred training plan.

- [ ] **Step 2: Memory pointer** — add/update a one-line entry in `/home/robert/.claude/projects/-home-robert-dev-glades-ml/memory/MEMORY.md` (and a small memory file if warranted) recording the generation-loop verdict + fix, linking `[[reln_reanchor_cure_arc]]`.

- [ ] **Step 3: Commit**
```bash
cd /home/robert/dev/glades-ml
git add CLAUDE.md research/CHIRON_GENERATION_LOOP_FIX_2026_06_27.md
git commit -m "Document CHIRON generation-loop fix outcome"
```

---

## Self-Review

**Spec coverage:**
- §4 diagnose-first architecture → Tasks 2 (diagnose) + 3 (fix) + 4 (validate) ✓
- §5 Phase-0 probes (D1/D2/D3/D4) → Task 2 Steps 1-4, reframed for the in-distribution full-window-slide reality (D1 = slide-vs-grow; D2 = SCFA causality via pad-variation; D3 = greedy-vs-sampled; D4 = position-under-slide confirmed by D1) ✓
- §6 Phase-1 fix (loop fix primary, causal guard fallback) → Task 3 Branch SLIDE (primary, full) + Branch NON-CAUSAL (contingent) ✓; spec's "trailing-pad" framing corrected to "window-slide / grow-don't-slide" for the in-scope case (noted in Architecture + flagged to owner).
- §7 validation (coherence metric + TF-unchanged) → Task 4 ✓
- §8 deliverables (instrumentation, fix, findings doc, runbook) → Tasks 1,3,2/4/5 ✓
- §9 exit criteria (fixed / exposure-bias-defer) → Task 3 Branch EXPOSURE-BIAS + Task 5 ✓

**Placeholder scan:** code steps show complete code; contingent branches (NON-CAUSAL/SAMPLING/EXPOSURE-BIAS) are intentionally design-level (built only if the verdict routes there, per diagnose-first). No "TBD/handle-edge-cases".

**Type/name consistency:** `gen_degeneration_metrics(const std::vector<int>&, double&, int&)` consistent across self-test (Task1 S1), implementation (S3), and call site (S5). Flags `genMetrics`/`seedTail` consistent. `--gen-metrics`/`--seed-tail`/`--tf-check`/`--tokens-file-n` consistent across tasks. Reference numbers (TF 1.7798/0.5346) consistent (Task1 S6, Task4 S2).

**Spec-vs-plan note (surfaced to owner):** the plan corrects the spec's prime-suspect emphasis from trailing-pad (the short-prompt, out-of-scope case) to window-slide (the in-scope in-distribution case); diagnose-first structure and SCFA-causality probe are unchanged and load-bearing.
