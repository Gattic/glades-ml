# CHIRON Generation-Loop Diagnosis & Fix — Local Coherence (Design)

**Date:** 2026-06-27
**Status:** Design — pending implementation plan
**Scope:** inference-side only; success bar = local coherence (continue real text)
**Repo of change:** glades-trainer (`tools/chiron_infer.cpp`); spec/research in glades-ml.

---

## 1. Problem

The reanchor-cure flagship (`chiron_1B_T16384_reanchor5B_finish.final`) has
excellent perplexity (val 1.92) and a now-correct inference forward (teacher-forced
top1 0.53, nll 1.78 via `--tf-check` — the fuse-attn-reln serving fix, `fffe70d`).
But **free generation collapses into low-entropy attractors**:

- short prompt (≈3 real tokens) → subword soup (`...the ation of e, and tion that
  ation...`);
- in-distribution (≈16k real tokens primed) → digit loop (`35356 353560... 2, 2,
  2 ... 8, 8, 8`);
- decoding-side repetition control (freq/presence penalty, no-repeat-ngram) breaks
  exact loops but the output stays degenerate.

## 2. Root-cause framing (the design pivot)

**The entire failure is the teacher-forced-vs-autoregressive gap.** The model is
near-perfect teacher-forced (top1 0.53) yet the autoregressive loop collapses
*immediately*. A model that good at next-token prediction should produce at least
locally-coherent greedy continuations. So the suspicion is that the generation
**loop** presents the model an input the TF path does not — i.e. the degeneration
is an **inference-loop bug**, not fundamental exposure bias. Precedent: the *last*
"generation incoherence" was a serving bug (missing fuse-attn-reln), not a limit.

**Concrete bug candidate — the trailing pad.** Training used full T=16384 windows
with **zero padding** (every position a real token). The generate loop
(`chiron_infer.cpp` ~1019–1048) **left-aligns** the context and **pads the trailing
window positions with token-0**, then reads logits at the last real position
`useLen−1`:

- short prompt: `useLen`≈3 → ~16381 trailing pad tokens (massively OOD);
- in-distribution: `useLen`=T−maxTokens → up to `maxTokens` (≤60) trailing pad
  until the window fills.

For a strictly causal model the trailing pad (future positions) cannot affect
position `useLen−1`. But **SCFA** (spectral compressed flow attention) applies a
**global DCT-II basis B[T,k] over all T positions**; if its spectral component is
not strictly causal, the trailing token-0 pad leaks into the representation at the
read position → corrupted logits exactly where generation samples. The `--tf-check`
has **no** trailing pad (full real window), which is why it is clean. Whether SCFA
is non-causal is the central unknown Phase 0 resolves.

## 3. Goal & non-goals

**Goal.** Make CHIRON produce **locally coherent continuations** — given a real-text
context, generate grammatical, non-degenerate English for N tokens (no
digit-loops / subword-soup) — via an **inference-side** fix, without degrading the
perplexity flagship (the validated forward must stay bit-for-bit on the TF path).

**Non-goals (explicit, separate follow-ons).**
- Short-prompt / near-empty-context coherence (the padding-OOD regime).
- Instruction-following / assistant behaviour (needs SFT/RLHF + a data pipeline).
- Generation-aware *training* (scheduled sampling / unlikelihood / RL). If Phase 0
  finds genuine exposure bias, this plan documents it and STOPS — training is a
  separate plan (owner decision 2026-06-27: cap at inference-side, defer training).

## 4. Architecture — diagnose-first, inference-only

```
Phase 0 (diagnose) ─┬─ loop bug (trailing-pad / causality / position / sampling) ─► Phase 1: fix generate loop
                    └─ genuine exposure bias (correct forward, gradual drift)     ─► document + STOP (defer training)
Phase 2: validate local coherence + degeneration metric + perplexity-unchanged regression
```

## 5. Phase 0 — Diagnostics (cheap; a handful of generation runs on real val context)

All via `chiron_infer` on `database/checkpoints/chiron_1B_T16384_reanchor5B_finish.final`,
seeded from `pretok-data/val.tok.bin` (uint16). Add `CHIRON_DBG` instrumentation as
needed.

- **D1 — full-window vs padded (most decisive).** Generate seeded with **exactly T
  real tokens** (window full from step 0, no trailing pad) vs the current T−60 seed
  (60-token trailing pad). **Full-window coherent + padded degenerate → trailing-pad
  is the bug.**
- **D2 — SCFA causality probe.** At gen step 0, with a fixed real prefix, vary the
  trailing-pad tokens (token-0 vs copy-of-last-real vs random in-vocab) and check
  whether the logits at `useLen−1` change. **Logits move → SCFA non-causal**
  (future pad leaks); unchanged → SCFA causal, the pad is not the cause.
- **D3 — greedy onset.** Greedy (temp→0) from a full real context: (a) does
  generated token 1 equal the TF-check argmax at that position (it must if the
  forwards match)? (b) does it stay coherent for K tokens then drift (exposure
  bias) or collapse on token 1 (bug)?
- **D4 — position-under-slide.** Once the window slides (drop oldest, newest token
  enters the window), confirm the position encoding the model sees matches training
  (RoPE relative is slide-invariant; absolute sinusoidal is not — verify which
  CHIRON uses and that the slide preserves it).

**Decision rule.**
- D1 full-window coherent, or D2 logits move → **trailing-pad / SCFA-causality bug**
  → Phase 1 (loop fix and/or causal guard).
- D3 immediate collapse with matching forwards + D2 no pad effect → **position or
  sampling bug** → Phase 1 (fix the identified site).
- D3 gradual drift, forwards consistent, no pad/position/sampling defect → **genuine
  exposure bias** → document in the findings doc, STOP, defer to a training plan.

## 6. Phase 1 — Fix (likely: eliminate the trailing-pad OOD)

The probable fix is small and inference-only: **always read next-token logits from a
full, pad-free window.** For local coherence the context is real text, so:

- Seed with ≥T real tokens and generate with a sliding **full** window — maintain
  the latest T tokens (positions [0..T)), read logits at **position T−1**, sample,
  append, slide (drop oldest). The loop never presents trailing pad; `useLen` is
  always T. (Current loop leaves a trailing pad until `tokens.size() ≥ T`.)
- **Fallback (if D2 shows SCFA non-causality is the deeper cause):** add a
  strict-causal guard on the inference SCFA path so positions > the read position
  contribute zero — mask/zero the spectral (B) and conv contributions beyond
  `useLen−1`. This also makes short-prompt generation well-posed (a later
  follow-on), but is only added here if D2 demands it.

**Files:** `tools/chiron_infer.cpp` — `generate()` (loop); `forwardInfer` only if the
causal guard is required. No change to the forward math exercised by `--tf-check`.

## 7. Phase 2 — Validate

- **Local coherence (qualitative):** greedy + sampled continuations from several
  real contexts (prose + code slices of val.tok.bin). PASS = grammatical,
  non-degenerate English; no digit-loops / subword-soup.
- **Degeneration metric (quantitative, before vs after):** repetition rate
  (fraction of repeated tokens in a window), distinct-4-gram ratio (unique/total),
  and the model's own mean NLL on its *generated* continuation (a coherent
  continuation has moderate NLL; a collapsed one has near-zero NLL on its repeats).
- **Regression (must hold):** `--tf-check` nll stays **1.78** (unchanged) — the fix
  is generation-only and must not perturb the validated forward.

## 8. Deliverables (glades-trainer inference-side; docs in glades-ml)

- `tools/chiron_infer.cpp`: Phase-0 `CHIRON_DBG` diagnostic probes (D1–D4) + the
  generate-loop fix (full-window read; causal guard only if D2 requires).
- A short validation runbook + a degeneration-metric snippet (repetition /
  distinct-n / generated-NLL).
- Findings doc `research/CHIRON_GENERATION_LOOP_FIX_2026_06_27.md` (glades-ml):
  the Phase-0 verdict, the fix (or the exposure-bias finding + training follow-on
  scope), and before/after generation samples + metrics.

## 9. Exit criteria / kill conditions

- **Fixed:** local-coherence PASS + degeneration metric materially improved +
  `--tf-check` unchanged → done; the flagship now generates coherent continuations
  via `runner.sh --flagship`.
- **Exposure bias (not inference-fixable):** Phase 0 shows a consistent forward with
  gradual drift and no loop defect → document, ship the diagnosis + any decoding
  improvement, and scope generation-aware training as a separate plan. Do **not**
  add training here.

## 10. Risks / open questions

- **SCFA causality is the load-bearing unknown.** D2 settles it; the fix branches on
  it (cheap loop fix vs a causal guard on the spectral path).
- **The fix could be a no-op for short prompts.** That is acceptable — short-prompt
  coherence is an explicit non-goal; this milestone is local continuation.
- **If the degeneration is partly genuine drift even with a clean window**, the
  inference fix improves but may not fully resolve it; per scope, that residue is
  documented and deferred, not trained away here.
