# ReLN Reverse-Consistency — Architecture-Native Instability Cure (Design)

**Date:** 2026-06-23
**Status:** Design — pending implementation plan
**Owner arc:** q-side reverse-amplification instability (CHIRON 1B @ T=16384)
**Predecessor records:** `research/QSIDE_INSTABILITY_INVESTIGATION_2026_06_14.md`,
`research/SIRA_TERMINAL_30K_RESULT_2026_05_27.md` (grad-trigger tooling),
`research/DATASCALE_FLAGSHIP_AND_SERVING_2026_06_20.md` (the prize this unblocks).

---

## 1. Problem

The q-side reverse-amplification instability is the binding constraint on data
scale. It re-emerges at ~1.6B tokens (accum=1) and forces the production
flagship to ship at ~2B tokens / val ~2.5 instead of pushing into the 5B+
regime the data-scale curve still projects (~2.2–2.3). The per-group grad clamp
(`--grad-group-clamp 1.0`, `clamp_vector_l2norm`) **contains** it (0 grad-skips
to 2B) but fires chronically past ~1.6B and does not cure it; clean 5B is
unreachable with the current recipe.

Every prior fix attempt was a **clamp** (bounds the symptom) and every one
either contained-without-curing or failed structurally:

| Attempt | Class | Result |
|---|---|---|
| gg-clamp (`clamp_vector_l2norm`) | aggregate clamp | CONTAINS, no cure |
| AGC (adaptive gradient clipping) | per-element/relative | structurally can't bound a sum |
| per-step spectral-norm (F=1.5/2.0) | weight rescale | NEGATIVE (effective-LR explosion) |
| spectral-init (one-shot) | weight rescale | BENIGN no-op |
| Phase-3 per-element xhat clamp | per-element source clamp | FAIL (can't bound a sum-over-T) |

## 2. Root-cause reframe (the design pivot)

The instability was modeled as *"a correct unit-RMS `xhat` summed over T=16384
grows too big"* — i.e. `dgamma[col] = Σ_t dout[t,col]·xhat[t,col]` overflowing
because T is large. **The grad-trigger snapshot refutes this.**

First bad gradient (seed 2024, step 24070, `global_sumsq=2.497e25`):
- Overflowing groups: `dE` (2.478e25, BAD) and `L00.dgamma` (1.913e23, BAD),
  then the early-layer `dgamma/dbeta` ladder.
- L00 q-state snapshot (`s.q` vs **saved** L00 ReLN stats): `qsig_ratio_max=13.03`,
  `xhat_rms_max=13.02`, `mean_abs_delta_max=0.00608`.
- Worst rows: `token=50 pos=9` (`qsig=13.03`); `token=265` (count 209,
  `pos=1761/2978/1317`, `qsig≈9.57`).

`xhat_rms=13` (healthy ≈ 1) is the disease: `xhat` is **~13× inflated before
the sum-over-T ever runs**. A unit-RMS `xhat` summed over T does not overflow at
this magnitude; an `xhat` that is already 13× too large does. So:

> **The source is a ReLN normalization inconsistency — the `xhat` used in the
> backward is formed against a state/statistics pair that has diverged —
> magnified by the sum-over-T. Clamps clip the inflated result instead of
> fixing the inflation.**

This is **architecture-native**: it is specific to CHIRON's reversible
(q,p)+ReLN structure, where ReLN **saves** forward normalization statistics and
the reversible block **recomputes** the activation in the backward pass. When
the recompute (`s.q`) and the saved stats diverge, the backward normalizes one
state with another state's statistics.

### 2.1 Why the micro-mechanism is still ambiguous (and must be diagnosed)

`qsig_ratio=13` / `xhat_rms=13` say the spread blew up, but
`mean_abs_delta_max=0.006` says the q-state barely moved *on average*. Three
mutually-exclusive readings remain, each implying a **different** cure:

- **(R) q/stat pairing mismatch (recompute-drift):** a sparse set of recomputed
  elements diverge from forward (consistent with tiny mean-delta but huge
  per-row sigma), so saved-forward stats no longer normalize `s.q` to unit RMS.
- **(V) variance-collapse:** the *saved* `rstd` is itself pathological (forward
  std → 0 on the worst rows), so any state normalized by it explodes.
- **(L) localized outlier-token phenomenon:** concentrated entirely on a few
  tokens (50, 265) — neither a global recompute nor a global variance effect.

Phase 3 failed precisely by committing to a cure before resolving this.
**Therefore the design leads with a diagnostic.**

## 3. Goal & non-goals

**Goal.** A default-off, multi-seed-gated mechanism that removes the `xhat`
inflation **at its source** in the ReLN backward, so the trigger event does not
fire and clean 5B+ becomes reachable (new best-val, 0 grad-skips), ideally
*replacing* gg-clamp rather than layering a second container on top.

**Non-goals.** No ported mechanism (MTP/UL2/LayerDrop-class — established to fail
on the symplectic update). No new aggregate or per-element clamp (the whole
failed class). No change to the production default until a clean 5B Gate-2 pass.

## 4. Architecture — a 4-phase decision program

```
Phase 0 (diagnostic) ─┬─► recompute-drift (R) ─► Branch R: --reln-reanchor   (fallback: --reln-store-layers)
                      ├─► variance-collapse (V) ─► Branch V: --reln-var-floor
                      └─► localized (L)        ─► per-token trace; NO cure built on a guess
Phase 1: build the selected branch (default-off kernel + flag + unit test)
Phase 2: Gate-1 (cheap)  — trigger-prevention + 5k no-harm parity   [kill/keep]
Phase 3: Gate-2 (47h)    — full 5B, cure-alone (gg-clamp OFF)        [ship/stop]
```

Each phase is a hard gate: a failure routes to a named fallback or a stop, never
an open-ended iterate.

## 5. Phase 0 — Diagnostic

**Purpose.** Resolve R vs V vs L with one decisive observable, at the cost of a
single trigger run that stops at the first bad gradient (~step 24k, `rc=0`,
~28k tok/s warm — already demonstrated by the existing hook).

**Instrumentation (extends the existing `--sira-grad-trigger-*` dump).** At the
trigger step, for the top-K L00 rows ranked by `|dout·xhat|`, record:
- saved forward `mean[row]`, `rstd[row]` (from the ReLN forward save),
- **forward `q[row]`** (stored for this diagnostic run only),
- **recompute `q[row]`** (`s.q`),
- `xhat_saved = (s.q − mean_saved)·rstd_saved` (what the backward currently uses),
- `xhat_selfconsistent = (s.q − mean(s.q))·rstd(s.q)` (the counterfactual cure-R),
- `dout[row]`.

Extend to the early ladder (L00–L03) since the `dgamma/dbeta` ladder lit up,
not only L00.

**Decision rule.** Note `xhat_selfconsistent` is unit-RMS *by construction* in
**both** R and V (any finite row normalized by its own mean/std → RMS≈1) — so it
confirms only that **re-anchor would restore unit-RMS**, not which mechanism. The
R-vs-V split is read from the **absolute** saved-σ vs recompute-σ magnitudes:
- `xhat_selfconsistent` RMS ≈ 1 **and** the **recompute** σ is the anomalous side
  (`σ_recompute` has a large tail above its own typical value while `σ_saved`
  stays normal) → **(R) recompute-drift** → Branch R (re-anchor cures it; it was
  the drifted recompute, so re-anchor bounds a wrong q).
- `xhat_selfconsistent` RMS ≈ 1 **and** the **saved** σ is the anomalous side
  (`σ_saved` collapsed broadly below the recompute scale, i.e. forward std→0) →
  **(V) variance-collapse** → Branch V (re-anchor *also* restores unit here, but
  `s.q` was fine — a variance floor is the targeted fix).
- `xhat_selfconsistent` RMS still ≫ 1 (re-anchor would *not* restore unit — only
  possible numerically when `σ_recompute`→0), or the effect is confined to a
  handful of tokens/positions with both global σ distributions healthy →
  **(L) localized** → per-token trace (tokens 50, 265 first); **do not build a cure**.

**Verification.** Smoke with a forced threshold (`--sira-grad-trigger-sumsq 0`)
must dump all fields finite on a tiny shape before the real run.

## 6. Phase 1 — Cure branches

### 6.1 Branch R — `--reln-reanchor` (primary)

In the ReLN backward, form `xhat` from statistics **re-derived from the same
recompute buffer used in the backward**, not the saved forward stats:

```
mean' = rowmean(q_recompute)
rstd' = 1 / sqrt(rowvar(q_recompute) + eps)
xhat' = (q_recompute - mean') * rstd'         # unit-RMS by construction
```

- **Source correction, not a clamp:** it computes the mathematically-correct
  unit-RMS `xhat` for the state the backward actually holds, rather than clipping
  an inflated one. Healthy steps (recompute ≈ forward) → `mean'≈mean_saved`,
  `rstd'≈rstd_saved` → **near-identity** (bf16-level delta).
- **Soundness:** reversible nets *assume* recompute = forward; re-anchoring
  enforces that assumption's consequence (unit-RMS `xhat`) when the recompute has
  drifted. It is the self-consistent ReLN gradient for the recomputed activation.
- **Kernel:** `reln_backward_reanchor` in `Backend/Machine Learning/Networks/cuda/gpu_chiron.cu`
  (+ no-CUDA stub in `gpu_chiron.h`), reusing the forward's existing per-row
  mean/var reduction. One extra reduction per row over the m dimension; negligible
  wall.
- **Flag:** `--reln-reanchor` (glades-trainer), default-off.

**Fallback (if re-anchor harms parity > 0.05 nat at Gate-1.2):**
`--reln-store-layers k` — store the exact forward q-state for layers 0..k
(~64 MB/layer bf16 at flagship shape; the early ladder L00–L03 ⇒ ~256 MB,
inside the ~0.86 GB headroom), so the backward reads the **exact forward
activation** with no recompute at the origin layers. Hypothesis-agnostic and
exact for any recompute-drift flavor; costs VRAM and forgoes reversibility's
memory win on those layers only.

### 6.2 Branch V — `--reln-var-floor` (only if Phase 0 = V)

Floor/condition the ReLN variance so neither saved nor recomputed `rstd` can
explode: a larger ε-floor or a variance-stabilizing reparam on the L00–L03 ReLN.
Built only if Phase 0 selects V; design detail deferred to that branch's plan
(the diagnostic returns the actual `rstd`/std magnitudes that set the floor).

## 7. Phase 2 — Gate-1 (cheap, kill/keep)

1. **Trigger-prevention.** Gold recipe + cure +
   `--sira-grad-trigger-stop --sira-grad-trigger-sumsq 1e20`, run to ~step 30k.
   **PASS** = trigger never fires past where gold tripped (24070) + margin;
   `global_sumsq` stays < 1e20.
2. **No-harm parity.** Gold vs gold+cure, 5k steps, same seed.
   **PASS** = val within multi-seed noise (≈ ±0.02), ‖g‖ healthy (no spikes),
   0 grad-skips, full throughput.

**Both** must pass to authorize Gate-2. Trigger-prevention FAIL → the cure
didn't target the mechanism → route to the Phase-0-indicated fallback. Parity
FAIL → re-anchor's deviation from the saved-stats gradient matters → fall back
to `--reln-store-layers` (exact).

## 8. Phase 3 — Gate-2 (the prize, ~47h)

Full flagship + `--accum 4 --lr 3e-4` + cure, 76k steps / ~5B tokens, seed 1337,
`--save-every 10000` keep-last-3 (pre-create the save dir — silent `save_full`
failure otherwise).

- **Sharp test — run cure-alone with `--grad-group-clamp` OFF first.** If the
  cure removes the inflation at the source, gg-clamp is unnecessary; this is the
  only way to prove a *cure* rather than a second containment layer. If it needs
  gg-clamp as a safety net, that is a contained-not-cured verdict (still useful,
  but not the goal).
- **PASS** = 0 grad-skips through 5B **AND** new best-val (< 2.5, toward
  ~2.2–2.3). Multi-seed (2024/4242) confirmation if single-seed passes.
- **Window-noise caveat:** val is 4-batch noisy; use the same multi-batch window
  for any cross-run comparison; same-seed runs are not bit-reproducible at
  production shape (atomic-ordering noise).

## 9. Kill criteria (stop, do not iterate)

- Phase 0 inconclusive (R/V both unclean) → it is localized (L): per-token trace,
  no cure built.
- Re-anchor fails trigger-prevention → not a pairing mismatch → store-early, else
  Branch V.
- Store-early **also** fails trigger-prevention → the disease is in the forward
  stats, not the recompute → Branch V is the only remaining path.
- Branch V fails → the instability is intrinsic to the data-scale regime;
  gg-clamp containment is the ceiling. **Publish the negative and stop.**

## 10. Deliverables (all default-off, per scope discipline)

**glades-ml:**
- `reln_backward_reanchor` kernel + no-CUDA stub (`gpu_chiron.{cu,h}`).
- Unit test `chiron-reanchor`: (a) `xhat`→unit-RMS under injected recompute
  drift; (b) near-identity (bf16 tol) when recompute == forward.
- Branch-V kernel + test only if Phase 0 selects V.

**glades-trainer:**
- `--reln-reanchor`, `--reln-store-layers k` flags; Phase-0 diagnostic dump
  extension to the `--sira-grad-trigger-*` hook; `run.sh` passthrough; smoke.

**Research/memory:**
- A verdict doc per phase (diagnostic result, Gate-1, Gate-2) with honest-negative
  publication per the Gate-0 methodology; memory pointer on any ship/closure.

## 11. Risks & open questions

- **Re-anchor changes the gradient when drift is present.** It is near-identity
  on healthy steps, but on a drifted step it deliberately uses a different
  (self-consistent) gradient than the saved-stats one. The parity gate bounds the
  healthy-step cost; the trigger-prevention gate bounds the drifted-step benefit.
- **The drift may be *correct* signal, not numerical error.** If `s.q` genuinely
  carries a large-spread activation the model needs, forcing unit-RMS `xhat`
  could suppress a real gradient. Gate-2 best-val is the arbiter: a cure that
  helps perplexity is signal-preserving; one that flattens val is over-damping.
- **VRAM for store-early at full L0–L3.** ~256 MB inside ~0.86 GB headroom at
  flagship shape; if a deeper ladder is needed it may not fit and would require a
  streamed/checkpointed variant (follow-up).
- **Diagnostic cost.** One trigger run to ~step 24k (~hours), not free; but far
  cheaper than a 47h structural miss.

## 12. Reproduction anchors

- Trigger run: `--sira-grad-trigger-dump --sira-grad-trigger-stop
  --sira-grad-trigger-sumsq 1e20 --sira-qbranch-trace-token 265` on the gold
  gg-clamp recipe, seed 2024 (fires step 24070).
- Gold recipe: `sh run.sh flagship --accum 4 --lr 3e-4 --warmup 750
  --sira-warmup 250 --zloss-coef 1e-4 --qk-norm --sira-coef 1e-2
  --sira-energy-weight 1.0 --sira-balance-weight 0.25 --sira-action-weight 0.0
  --grad-clip 0.5 --dq-layer-clamp 1.0 --dq-embed-clamp 1.0
  --grad-group-clamp 1.0 --seed 2024`.
