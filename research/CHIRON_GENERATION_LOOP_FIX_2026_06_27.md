# CHIRON Generation-Loop Diagnosis — Result (2026-06-27)

**Verdict: EXPOSURE BIAS / intrinsic repetition attractor — NOT inference-fixable.**
The diagnose-first investigation refuted every inference-side hypothesis (window
slide, trailing pad, decoding/sampling, forward bug). The reanchor-cure flagship
generates incoherently because, when free-running on its own outputs, it collapses
into a repetition loop whose next-token distribution is so peaked that even nucleus
sampling cannot escape. This is generation-side and **training-fixable, not
inference-fixable**. Per the owner decision (cap at inference-side; defer training),
this plan delivers the diagnosis and stops; generation-aware training is scoped as
a separate follow-on.

Spec: `docs/superpowers/specs/2026-06-27-chiron-generation-loop-fix-design.md`.
Plan: `docs/superpowers/plans/2026-06-27-chiron-generation-loop-fix.md` (Branch
EXPOSURE-BIAS). Model: `chiron_1B_T16384_reanchor5B_finish.final` (val 1.92, TF
top1 0.5346 / nll 1.7798).

## What the design got wrong, and how diagnose-first caught it

The spec's prime suspect was the **trailing pad** (short-prompt OOD). Grounding
corrected that to the **window slide** (the in-distribution case seeds a full-T
window that slides every step against SCFA's absolute-position DCT). **Both were
wrong.** Phase 0 tested the slide hypothesis directly and refuted it — had we
implemented the planned "grow-don't-slide" fix without diagnosing, it would not
have worked. This is the value of diagnose-first.

## Evidence chain

1. **Forward is correct.** `--tf-check` top1 0.5346, nll 1.7798 — the model
   predicts real next tokens well. Generation's per-step forward re-embeds the
   full window with no cross-step state, so the first generated token's logits ARE
   a clean forward's logits → degeneration is not a forward/loop bug.
2. **Slide vs grow (D1), greedy + sampled (G3) — all degenerate:**
   - SLIDE (full-T seed, slides): greedy → `6060… s will be s will be`; sampled →
     `35356… 2,2,2… 8,8,8`.
   - GROW (`--seed-tail`, no slide): greedy → `. ![. ![. Therefore, the…`; sampled
     → `]{} & ]{} & ----------- where $…`.
   These early seeds (first 16 384 val tokens) were **numeric/markdown/LaTeX**
   regions — a confound: the model was partly continuing structured data.
3. **Prose-seed test (added `--tokens-file-offset` + seed-context decode):** seeded
   from clean prose regions and decoded what the model continues:
   - offset 200000 — context `"…decreased methotrexate"` →
     greedy `decreased decreased decreased a decreased lower…`;
     sampled (t0.8, p0.95) `decreased decreased decreased later…` (**same loop**).
   - offset 1000000 — context `"…booting off the curtained disk"` →
     greedy `from a from a from a from the…`;
     sampled `from a from a from a change from a…` (**same loop**).
   - offset 200000 + repetition control (freq/presence penalty, no-repeat-ngram):
     `methmethmeth… ) { . The ) { . A ) {…` (breaks exact loops → subword/markdown
     soup).
4. **Conclusions from (3):**
   - **Not the slide** — grow degenerates too.
   - **Not the pad** — the SLIDE prose cases have no pad and degenerate.
   - **Not decoding** — nucleus sampling gives the *same* repetition loop; the
     distribution collapses onto the repeated token so hard that top-p 0.95 keeps
     selecting it. A healthy model's greedy repetition is escaped by nucleus
     sampling; this one is not.
   - **Not a forward bug** — forward is the validated TF forward.
   → The model has an **intrinsic repetition attractor**: free-running on its own
   repeats is OOD (training text doesn't repeat), and the model becomes
   increasingly confident in the repeat (cf. the documented logit-climb 8.6→38).

## Why a great-perplexity model still does this

Perplexity (teacher-forced) and free-generation coherence are orthogonal. The
model is excellent at "predict the next token given REAL context" (top1 0.53), but
has never been trained on its OWN outputs, so once it emits a repeat it enters a
self-reinforcing OOD state. The very sharp QK-Norm attention (γ≈log₂T≈14) likely
amplifies the latch onto a recent salient token. None of this is reachable from
the inference loop or the decoder.

## Outcome / what landed

- **No fix shipped** (correctly — the inference-side hypotheses were all refuted).
- **Diagnostic tooling landed in `tools/chiron_infer.cpp`** (default-off; useful
  for the deferred training work): `--gen-metrics` (distinct-4-gram + max
  single-token run), `--seed-tail` (grow-don't-slide), `--tokens-file-offset`
  (seed a deeper region), `CHIRON_DBG` seed-context decode.
- **Metric limitation noted:** distinct-4-gram is INADEQUATE here — templated
  repetition with slot variation (`decreased … decreased lower … decreased
  higher`) keeps distinct-4 ≈ 1.0 while the text is degenerate. A type-token ratio
  (distinct-1) or repeated-token-fraction metric is needed for the follow-on.

## Follow-on (deferred, separate plan): generation-aware training

Make CHIRON a coherent generator by training against its own free-running behavior
— candidates: **unlikelihood training** (penalize repeated tokens/n-grams in the
loss), **scheduled sampling** (mix model tokens into the teacher-forced context),
or **DPO/RL** on a coherence/repetition reward. Resume from the flagship; gate
default-off; guard the perplexity flagship (must not regress val 1.92). Out of
scope here by owner decision; this doc is the entry point.

## Standing caveat confirmed

CHIRON flagships are **perplexity flagships, not generators** (true of the whole
lineage). The reanchor-cure flagship's −0.62 nat perplexity win does NOT confer
coherent free generation; the repetition attractor is a separate, training-shaped
problem. CLAUDE.md's generation caveat stands, now with a precise mechanism.
