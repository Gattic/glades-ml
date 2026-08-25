# CHIRON CHARGE — Phase-0 Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a minimal, default-off, optimizer-only **gradient reservoir** (CHARGE) in the
trainer and run the matched 2500-step diagnostic that tests the hypothesis underlying the whole
SIPHON program — *"the energy discarded by the chronic global grad-clip contains recoverable
perplexity value"* — with zero forward-map risk. Issue the pre-registered GO/KILL decision.

**Architecture:** A single global scalar reservoir `ρ ≥ 0` in the trainer's runtime state. At the
existing global grad-clip site (`chiron_main.cpp:18797–18839`), when the clip *fires*
(`gradNorm > τ`) charge the discarded magnitude `ρ += (gradNorm − τ)` (capped); on *calm* steps
(`gradNorm ≤ τ`) discharge `δ = γ·min(ρ, τ − gradNorm)` by **boosting** `gradScale` so the effective
update norm is `gradNorm + δ ≤ τ` (still inside the baseline trust region), and `ρ -= δ`. This is
the budget-safe scalar reduction of the CHARGE candidate (spec App. A): it recaptures the clip's
discarded *magnitude* by letting calm steps spend the saved budget. Provably `‖update‖ ≤ τ` always,
so it can never be less stable than baseline. No glades-ml (lib) changes; no new kernels; no
forward/backward change; optimizer-side only ⇒ reversibility + determinism untouched.

**Tech Stack:** C++98, `glades-trainer/trainer/chiron_main.cpp` (single-file trainer),
`getenv`-free CLI flags (parse in the `--flag` block ~line 2308), `log_info("chiron",...)`
(`[sorc]`/`[sira]` monitor precedent). Repo `~/dev/glades-trainer`, branch `chiron4`.
GPU RTX 4080 SUPER 16 GB.

**Scope:** Phase-0 ONLY (spec §10). SIPHON (the forward-pass thermostat) and CHARGE's per-group /
`m̂`-projected refinement are OUT of scope and contingent on this gate. This plan produces a
self-contained deliverable: the flags, the reservoir hook, the `[charge]` monitor, the matched
2500-step pair, and the decision record.

## Execution note (2026-07-05) — pivoted to a τ-sweep

Tasks 1–2 were implemented and validated (E0 bit-clean; charge-side + `ρ_max` cap + `effNorm≤τ`
invariant all confirmed in a 30-step smoke). **The smoke exposed that the discharge mechanism is
inert in the real regime:** the global clip binds *every* step (‖g‖ chronically 0.8–4 vs τ=0.5), so
the `gradNorm ≤ τ` discharge trigger essentially never fires — the reservoir pins at `ρ_max` and does
nothing (the "never-calm regime" is the actual regime). Rather than run a 3.8 GPU-hr pair that would
report a false "no effect," Task 3 was **pivoted (owner call) to a grad-clip sweep** that tests the
*underlying* premise directly: matched 2500-step arms at **τ ∈ {0.5, 1.0, 2.0}**, seed 1337, ONLY
`--grad-clip` varying (`scripts/tau_sweep.sh`). If a looser clip lowers val@2500 → the chronic clip
is taxing perplexity and recapture has real headroom; if it raises val or adds grad-skips → the clip
is load-bearing and recapture must be gentle (informs SIPHON's self-calibrating damping). The CHARGE
scaffolding (Tasks 1–2) is held **uncommitted/dormant** pending the sweep verdict, at which point it
is either finalized (bounded-ceiling or direction-biasing variant) or reverted. Tasks 1–2 below stand
as the record of what was built.

## Global Constraints

- **Design source of truth:** `~/dev/glades-ml/docs/superpowers/specs/2026-07-05-chiron-siphon-thermostat-energy-recapture-design.md` §3 (selection), §10 (Phase-0), App. A (CHARGE).
- **Clip site (verified):** `chiron_main.cpp:18797` — `if (cfg.gradClip > 0.0f)`; `gradNorm` at
  18807; `badGradNorm→gradScale=0,skipAdamUpdate=true` at 18827–18833; `gradNorm>clip→
  gradScale=clip/gradNorm` at 18835–18838. `gradScale` (init `1.0f`) multiplies grads before Adam.
  τ = `cfg.gradClip` (production 0.5).
- **Reservoir update rules** (global scalar, per optimizer step, only when `cfg.chargeEnable`):
  - `u = min(gradNorm, τ)` (effective clipped norm this step); `overflow = (gradNorm − τ)_+`.
  - **Charge:** `ρ ← min(ρ + overflow, ρ_max)`  (only when `!badGradNorm`).
  - **Discharge (calm, `gradNorm ≤ τ`, `!badGradNorm`):** `δ = γ · min(ρ, τ − gradNorm)`;
    `ρ ← ρ − δ`; set `gradScale = (gradNorm + δ) / gradNorm` (boost > 1, bounded so `‖a‖ = gradNorm
    + δ ≤ τ`).
  - **Clip step (`gradNorm > τ`):** `gradScale = τ/gradNorm` UNCHANGED (charge only).
  - **Bad step:** no charge, no discharge, `ρ` persists; `gradScale=0` unchanged.
- **Stability invariant (must hold in code):** effective update norm `= gradScale·gradNorm ≤ τ`
  on every non-bad step. Assert in the monitor.
- **Defaults:** `--charge` off; `--charge-gamma 0.25`; `--charge-rho-max` = `10·τ`.
- **E0 requirement:** with `--charge` absent, the code path is bit-identical to baseline (the
  reservoir block is a no-op guarded by `cfg.chargeEnable`). Verify by a 20-step same-seed diff.
- **Determinism/reversibility:** CHARGE touches ONLY `gradScale` (a pre-Adam scalar). No RNG, no
  forward/backward/inverse-walk change. Do not add any counter-hash or stochastic element.
- **Verify `gradScale` flows unclamped:** confirm no downstream code clamps `gradScale ≤ 1`
  (grep the Adam-apply path); a boost > 1 must reach the update. If clamped, lift the clamp to `τ`
  headroom only under `cfg.chargeEnable`.
- **No checkpoint persistence needed** (2500-step non-resumed diagnostic); `ρ` is a runtime float
  init 0. Persistence is a follow-up, out of scope.

---

### Task 1: Config flags + runtime reservoir state (default-off, E0 bit-parity)

**Files:**
- Modify: `trainer/chiron_main.cpp` — config struct (near `gradClip` decl ~line 1130), defaults
  (~line 1549), CLI parse (~line 2308), usage string (~line 1675).

**Interfaces:**
- Produces: `cfg.chargeEnable (bool)`, `cfg.chargeGamma (float)`, `cfg.chargeRhoMax (float)`; a
  runtime `double chargeReservoir = 0.0;` in the training-loop scope.

- [ ] **Step 1: Add config fields.** In the config struct near `float gradClip;`:
```cpp
bool  chargeEnable;   // CHARGE Phase-0: recapture clipped grad magnitude (default off)
float chargeGamma;    // discharge fraction per calm step (default 0.25)
float chargeRhoMax;   // reservoir cap; <=0 => 10*gradClip (default)
```

- [ ] **Step 2: Defaults.** In the initializer list (~1549) add `chargeEnable(false),
  chargeGamma(0.25f), chargeRhoMax(0.0f),`.

- [ ] **Step 3: CLI parse.** In the `--flag` block (~2308):
```cpp
else if (streq(a, "--charge")) cfg.chargeEnable = true;
else if (streq(a, "--charge-gamma") && i + 1 < argc) cfg.chargeGamma = parse_f32(argv[++i], cfg.chargeGamma);
else if (streq(a, "--charge-rho-max") && i + 1 < argc) cfg.chargeRhoMax = parse_f32(argv[++i], cfg.chargeRhoMax);
```

- [ ] **Step 4: Usage line** (~1675, near `--grad-clip`):
```cpp
"  --charge            CHARGE Phase-0: recapture clipped grad magnitude into a reservoir,\n"
"                      spent on calm steps within the clip budget (default off).\n"
"  --charge-gamma F    Discharge fraction per calm step (default 0.25).\n"
"  --charge-rho-max F  Reservoir cap (default 10*grad-clip).\n"
```

- [ ] **Step 5: Runtime reservoir.** Just before the training loop (same scope as `siraGradTriggerFired`), add `double chargeReservoir = 0.0;` and resolve the cap once: `const double chargeRhoMax = (cfg.chargeRhoMax > 0.0f) ? cfg.chargeRhoMax : 10.0 * cfg.gradClip;`.

- [ ] **Step 6: Build.** `cd ~/dev/glades-trainer && bash build.sh` — expect clean build.

- [ ] **Step 7: E0 bit-parity check.** Run 20 steps WITHOUT `--charge` (flagship recipe, seed 1337, `--val-every 999999 --max-steps 20`) and confirm step-1..20 `loss`/`‖g‖` are byte-identical to a pre-change run (or to the current `logs/`), i.e. the flag being absent changes nothing. Expected: identical loss trajectory.

- [ ] **Step 8: Commit.**
```bash
git add trainer/chiron_main.cpp
git commit -m "chiron_main: CHARGE Phase-0 flags + runtime reservoir (default-off, no-op)"
```

---

### Task 2: Charge/discharge hook at the global-clip site + `[charge]` monitor

**Files:**
- Modify: `trainer/chiron_main.cpp:18797–18839` (inside `if (cfg.gradClip > 0.0f)`, after
  `gradNorm` is known and after the `badGradNorm` / `gradNorm>clip` branches set `gradScale`).

**Interfaces:**
- Consumes: `gradNorm`, `badGradNorm`, `gradScale`, `cfg.gradClip (τ)`, `cfg.chargeGamma`,
  `chargeReservoir`, `chargeRhoMax`, `step`, `cfg.logEvery`.

- [ ] **Step 1: Insert the reservoir logic** immediately AFTER the `else if (gradNorm > cfg.gradClip)` block (after line 18838), still inside `if (cfg.gradClip > 0.0f)`:
```cpp
// CHARGE Phase-0 (spec App. A): recapture the clipped magnitude, spend on calm steps.
double chargeDischarge = 0.0, chargeOverflow = 0.0;
if (cfg.chargeEnable && !badGradNorm)
{
    const double tau = (double)cfg.gradClip;
    if ((double)gradNorm > tau)                       // clip fired: charge only
    {
        chargeOverflow = (double)gradNorm - tau;
        chargeReservoir = std::min(chargeReservoir + chargeOverflow, chargeRhoMax);
        // gradScale already = tau/gradNorm from the clip branch — unchanged.
    }
    else if ((double)gradNorm > 0.0)                  // calm: discharge into headroom
    {
        const double headroom = tau - (double)gradNorm;
        chargeDischarge = (double)cfg.chargeGamma * std::min(chargeReservoir, headroom);
        chargeReservoir -= chargeDischarge;
        gradScale = (float)(((double)gradNorm + chargeDischarge) / (double)gradNorm); // boost >1, ||a||<=tau
    }
}
```

- [ ] **Step 2: `[charge]` monitor** at log cadence (after the block, mirroring `[sorc]`/`[sira]` logging; place near where those log, guarded by the existing log-step predicate):
```cpp
if (cfg.chargeEnable && cfg.logEvery > 0 && (step == 1 || step % cfg.logEvery == 0))
{
    const double effNorm = (double)gradScale * (double)gradNorm;   // must be <= tau
    log_info("chiron","[charge] step=%6d gradNorm=%.4g clipFire=%d overflow=%.4g discharge=%.4g reservoir=%.4g effNorm=%.4g (bar<=%.3g)\n",
             step, (double)gradNorm, (int)(gradNorm > cfg.gradClip), chargeOverflow, chargeDischarge,
             chargeReservoir, effNorm, (double)cfg.gradClip);
}
```

- [ ] **Step 3: Stability assertion (debug guard).** In the monitor, if `effNorm > cfg.gradClip * 1.0001` on a non-bad step, `log_warn` — this must never fire (the discharge is headroom-bounded). It is the in-vivo proof of the `‖update‖≤τ` invariant.

- [ ] **Step 4: Build.** `bash build.sh` — clean.

- [ ] **Step 5: Smoke (30 steps, `--charge`).** Run 30 steps WITH `--charge` on the flagship recipe (seed 1337). Expect: step-1 `loss`/`‖g‖` identical to baseline (reservoir empty ⇒ first step can only charge, gradScale unchanged); `[charge]` lines show `reservoir` rising on clip-fire steps and `discharge>0` / `effNorm ≤ τ` on any calm step; the stability assert never fires.

- [ ] **Step 6: Commit.**
```bash
git add trainer/chiron_main.cpp
git commit -m "chiron_main: CHARGE charge/discharge hook at global-clip site + [charge] monitor"
```

---

### Task 3: Matched 2500-step diagnostic pair + GO/KILL decision

**Files:**
- Create: `scripts/charge_phase0_pair.sh` (driver, mirrors `scripts/pact_perf_probe.sh` recipe).
- Create: `research/CHIRON_CHARGE_PHASE0_2026_07_05.md` (decision record).

**Pre-registered bars (spec §10 Phase-0):**
- **Stability (hard):** arm B (CHARGE on) must have **0 grad-skips** — the `‖update‖≤τ` theorem says
  it cannot be worse than A. Any grad-skip in B but not A ⇒ implementation bug.
- **Recapture signature (the GO condition):** arm B `Σ‖a_t‖ > Σ‖u_t‖` (mass delivered), AND
  `val@2500(B) ≤ val@2500(A) − 0.02` (beyond the ~0.02 atomic-noise floor; both arms 4-batch same
  windows), with the gap **not concentrated pre-spike** (ideally concentrated at/after the largest
  `‖g‖` excursions).
- **KILL:** `val@2500(B) ≥ val@2500(A)` (within noise) AND reservoir sawtooths normally ⇒ recapture
  is inert ⇒ the SIPHON premise is unsupported; **do not build SIPHON** (or first try `γ` sweep +
  the per-group/`m̂` refinement before final kill).

- [ ] **Step 1: Driver script.** Write `scripts/charge_phase0_pair.sh` running the flagship recipe (from `scripts/pact_perf_probe.sh`, MINUS `--pact-coef`, PLUS `--inc-dropout 0.1 --whisc-coupling --rot-theta-max 0.07`), `--max-steps 2500 --val-every 250 --val-batches 4 --seed 1337`, arg `$1 ∈ {baseline,charge}` selecting the save dir and appending `--charge` for the treatment arm. Both arms bit-identical at step 1 (reservoir empty).

- [ ] **Step 2: Run arm A (baseline).** `sh scripts/charge_phase0_pair.sh baseline > logs/charge_p0_baseline.log 2>&1` (~1.9 GPU-hr). Record val@250..2500, grad-skips, `‖g‖` spike steps.

- [ ] **Step 3: Run arm B (CHARGE).** `sh scripts/charge_phase0_pair.sh charge > logs/charge_p0_charge.log 2>&1` (~1.9 GPU-hr). Record the same + the `[charge]` reservoir trajectory (sawtooth expected).

- [ ] **Step 4: Analyze.** Compare val trajectories (matched 4-batch windows), grad-skip counts, and whether B's advantage (if any) concentrates at/after the `‖g‖` excursion steps. Confirm the `effNorm ≤ τ` invariant held every step (no stability-assert warnings).

- [ ] **Step 5: Decision record.** Write `research/CHIRON_CHARGE_PHASE0_2026_07_05.md`: the two trajectories, the reservoir sawtooth, grad-skip counts, the GO/KILL call against the bars above, and — if GO — the hand-off to SIPHON Phase-1; if KILL, whether to try the `γ`-sweep / per-group refinement or close the direction.

- [ ] **Step 6: Clean up scratch.** `rm -rf database/checkpoints/charge_p0_*` (per the keep-only-flagship+fixture policy); retain the two logs + the decision record.

- [ ] **Step 7: Commit.**
```bash
git add scripts/charge_phase0_pair.sh
git commit -m "chiron: CHARGE Phase-0 matched-pair driver"
cd ~/dev/glades-ml && git add research/CHIRON_CHARGE_PHASE0_2026_07_05.md
git commit -m "docs: CHARGE Phase-0 diagnostic result + GO/KILL decision"
```

---

## Self-review notes
- **Spec coverage:** implements spec §10 Phase-0 + App. A scalar prototype. The per-group / `m̂`-
  projected version and SIPHON are explicitly deferred (contingent on this gate).
- **The one honest simplification:** the global-scalar reservoir discharges by boosting `gradScale`
  along the *current* calm-step gradient, not the `m̂`-projected persistent direction. This tests
  the *core* "recapture-the-magnitude" hypothesis at minimal risk; if the signal is marginal, the
  `m̂`-projection (recapture only the persistent component, filtering calm-step noise) is the first
  refinement before a KILL. Stated so the reviewer/experimenter is not misled.
- **Stability is proven, not hoped:** the `effNorm ≤ τ` assertion (Task 2 Step 3) is the in-vivo
  check of the `‖a_t‖≤τ` theorem; 0 grad-skips in arm B is a hard bar.
