# Data-Scale Flagship Candidate + Serving Fix — Session Findings (2026-06-20)

This doc consolidates a multi-day session covering four threads:
1. **Stability techniques** (GC / spectral / SAM) — implemented; spectral NEGATIVE.
2. **The 5B data-scale run** — a **−0.9 nat** flagship candidate (val ~2.5 vs the
   production flagship's 3.5062).
3. **A run.sh `--lr-decay` flagship bug** — found + fixed (it gated the win).
4. **The serving forward** — chiron_infer was missing QK-Norm since 2026-05-22;
   found + fixed + validated; γ now persisted; generation repetition mitigated.

Anchor baselines: production flagship `chiron_1B_T16384_sira_clamp_phase2.final`
(val **3.5062**). Prior data-scale context: accum=1 run banked val 2.797 @ 1.3B
before degrading; gg-clamp validated 0 skips to 1.64B (see
`QSIDE_INSTABILITY_INVESTIGATION_2026_06_14.md`).

---

## 1. Stability techniques (GC / spectral / SAM) — implemented; spectral NEGATIVE

Per owner request, the three remaining techniques from
`docs/superpowers/plans/2026-06-16-chiron-stability-techniques.md` were built
end-to-end (kernel + unit test + trainer flag + run.sh + smoke), all default-off.

| Phase | Flag | Status |
|---|---|---|
| 2 Gradient Centralization | `--grad-centralize` | implemented (`gradient_centralize_bf16`, glades-ml `9e3a15c5e`; trainer `00d11fb`) |
| 4 Spectral norm | `--spectral-init/-norm/-iters` | implemented (`dbb0dd605`,`8b2729f18`; trainer `55888d0`) → **NEGATIVE** |
| 5 SAM | `--sam-rho` | implemented (`25467b0b5`; trainer `c51fcfc`); ε-scratch ≈1 grad-set VRAM (OOMs at flagship T) |

Also closed earlier: **Phase 3 bounded-ReLN "source cure" FAILED** (structural):
a per-element xhat clamp can't bound a sum-over-T=16384 overflow; only the
aggregate gg-clamp does (`957dffe00`).

**Spectral-norm verdict (NEGATIVE, `eb7bfcf2c`,`57a45568b`,`89182e236`):**
the flagship's natural σ_max(Wq/k/v/o) ≈ 2.17. Per-step capping below that
rescales weights down each step while Adam pushes back → an **effective-LR
explosion**. Both F=1.5 and F=2.0 reached an *implausibly* low val (2.63 @ 0.33B
tokens — better than our best-ever at 4× less data) coupled with instability;
F=2.0 *separated* the modes (fixed the instability but kept the degenerate val),
proving the degeneration is intrinsic to per-step rescaling, not cap strength.
One-shot `--spectral-init` is **benign** (val 3.6614 ≈ gold), confirming the
per-step rescaling is the culprit. **Per-step spectral-norm closed; gg-clamp
remains the containment fix.** GC/SAM remain banked default-off (untested at
scale; SAM has a flagship-shape VRAM constraint).

---

## 2. The 5B data-scale run — a −0.9 nat flagship candidate

**Recipe:** full flagship + accum=4/lr3e-4 + `--grad-group-clamp 1.0`, seed 1337,
76000 steps (target 5B tokens). Log: `glades-trainer/logs/datascale5B_20260618_221547/`.

**Trajectory (val NLL):**

| step | tokens | val | note |
|---|---|---:|---|
| 15000 | 1.0B | 3.4117 | tracks gg-clamp gold |
| 20000 | 1.3B | 3.4854 | unstable-regime bump |
| 25000 | 1.6B | 3.3569 | instability re-emerged here (q-side) but **contained** |
| 30000 | 2.0B | 3.3252 | |
| 40000 | 2.6B | 2.9919 | data-scale curve kicks in |
| 45000 | 2.95B | 2.8285 | |
| 55000 | 3.6B | **2.7619** | plateau (LR-bound, see §3) |

**Instability:** the q-side instability re-emerged at ~1.6B (the unvalidated
horizon past gg-clamp's 1.64B), but the gg-clamp **contained** it — ‖g‖ bursts to
1e3–1e6 with loss-scale collapsing to 0, gg-clamp firing ~every step (36k+ fires),
yet **0 grad-skips** the whole run and val kept improving. This validates the
investigation's thesis: the instability is the binding constraint, gg-clamp
contains-not-cures, and data scale delivers the win once past ~2B tokens.

**Operational note:** the initial `--save` path dir didn't exist → silent
save_full failures for the first ~16h. Fixed mid-run (`mkdir`); always pre-create
the save dir. Banked checkpoints: `chiron_1B_T16384_datascale5B/{step30000,40000,
50000,60000}` + `..._BEST_val2p771.step50000` (val 2.7709).

---

## 3. The run.sh `--lr-decay` flagship bug (found + fixed)

The 5B run **plateaued at ~2.76** because run.sh **silently dropped `--lr-decay`
in flagship mode**: it set `CHIRON_LR_DECAY=1` but only appended `--lr-decay` to
`CHIRON_ARGS` (the *chiron-mode* exec); the flagship exec (line 570:
`SHAPE STACK SCHED IO_ARGS REGSTACK_ARGS`) never received it. So the run trained
at constant lr=3e-4 start→finish and never annealed (lr logged 3e-4 at step 60k).

**Fix (`9f45803`):** append `--lr-decay`/`--lr-decay-min` to `SCHED` when
`CHIRON_LR_DECAY=1` + add the `--lr-decay-min` parse arm.

**LR-anneal finish (the payoff):** resumed step60000 full-state via a direct
binary call at flat lr 3e-5 (10× lower), 5000 steps → **val broke the plateau
2.76 → ~2.5**. Checkpoint `chiron_1B_T16384_datascale5B_finish.final`. The plateau
was LR-bound, confirmed. (Caveat: a resume-warmup re-triggered, so this was
"drop-and-hold-low," not a clean cosine-to-zero; a clean retrain via the fixed
run.sh would do it properly.)

**Result: held-out val ~2.5–2.65** (32-batch windows; window variance 2.51–2.79,
TF nll 2.3757) = **−0.85 to −1.0 nat vs the flagship's 3.5062**. Validated
legitimate (not the spectral-style degeneracy): acc1 ~0.35, flat per-position NLL,
on the proven accum=1 data-scale curve.

**To ship as production flagship:** one clean retrain through the fixed run.sh
(proper cosine anneal over the full horizon, no resume artifact, γ persisted per
§4) — reproduces ~2.5 in one run, in serveable format. ~47h.

---

## 4. The serving forward — QK-Norm was missing (found + fixed + validated)

**Discovery:** chiron_infer generated degenerate garbage from **every** SCFA
checkpoint, *including the production flagship* — so the flagship was never
actually generation-serveable. Root-caused via a new teacher-forcing oracle
(`--tf-check`: one forward, argmax-vs-next-token at every position).

**Two bugs + one harness trap:**
1. **chiron_infer never implemented QK-Norm.** The regstack/SIRA/data-scale models
   train with `--qk-norm` (per-head L2-normalize Q/K, scale by learned per-head
   γ≈14 instead of 1/√d), added 2026-05-22; inference used plain 1/√d → wrong
   attention. **Fixed (`8c8b5a3`):** decomposed QK-Norm SCFA inner path
   (project → `qknorm_forward_gpu` → γ·√dH scale → flash-attn → Wo), behind
   `--qk-norm`.
2. **The CHRF save never persisted the trained γ** → serving had to approximate
   γ=log₂(T)≈14. **Fixed (`8b307b3`):** new CHRF flag bit 256 + L·nH γ blob;
   trainer save+load and chiron_infer load wired; per-layer γ used when present,
   γ≈14 fallback otherwise. Validated round-trip (train→save→resume→infer "EXACT").
3. **Harness trap:** my `--tokens-file` diagnostic read **int32** but pretok
   `.tok.bin` is **uint16** → fed all-token-0 → a false "forwardInfer is broken"
   conclusion. **Fixed (`561f8f3`).** Lesson: verify the token dtype before trusting
   an inference diff.

**Validation (with correct uint16 tokens):** inference now reproduces each model's
training val perplexity exactly —

| model | QK-Norm | inference TF nll | training val |
|---|---|---:|---:|
| data-scale | on | **2.38** | ~2.5 |
| data-scale | off | 21.7 | — |
| flagship | on | **3.48** | **3.5062** |

So **the flagship and the data-scale model are now correctly served for
scoring/perplexity** — which was never true before this session.

**Generation repetition (mitigated, model-limited):** free generation hits a
runaway repetition attractor (a repeated token's logit climbs 8.6→38). Added
repetition control to chiron_infer (`624d007`): subtractive frequency penalty
(the effective lever), presence penalty, no-repeat-ngram, recent-window. Breaks
the catastrophic single-token loops, but the model *games* token-level blocks via
BPE-alternation → free-gen coherence is **model-limited** (a perplexity-LM
property), not a forward bug. Coherent generation would need generation-aware
fine-tuning.

---

## Commit & checkpoint reference

**glades-trainer:** `bb15257` (Phase3 flag), `00d11fb` (GC), `55888d0` (spectral),
`c51fcfc` (SAM), `9f45803` (run.sh lr-decay fix), `8c8b5a3` (QK-Norm infer),
`7649e1d` (tf-check/dbg), `561f8f3` (uint16 fix), `8b307b3` (γ persistence),
`624d007` (repetition control).
**glades-ml:** `6b53b9f86`,`957dffe00` (Phase3), `9e3a15c5e` (GC), `dbb0dd605`,
`8b2729f18` (spectral), `25467b0b5` (SAM), `eb7bfcf2c`,`57a45568b`,`89182e236`
(spectral verdicts), and this doc.

**Checkpoints:** `chiron_1B_T16384_datascale5B_finish.final` (anneal result, val
~2.5 — flagship candidate); `chiron_1B_T16384_datascale5B_BEST_val2p771.step50000`
(pre-anneal best, 2.7709); `chiron_1B_T16384_datascale5B/step{30,40,50,60}000`.

## Open items / recommendations

1. **Clean 5B+anneal retrain** via the fixed run.sh (proper cosine, γ persisted)
   to lock in the ~2.5 flagship candidate reproducibly + serveably. ~47h.
2. **Don't pursue per-step spectral-norm** (NEGATIVE). gg-clamp stays the
   containment fix; GC/SAM are optional hardening (untested at scale).
3. **Generation:** needs generation-aware fine-tuning, not decoding, for coherence.
4. The data-scale run shows clean 5B isn't reachable with the current recipe —
   the instability is contained but chronic past ~1.6B; a true *cure* remains open.
