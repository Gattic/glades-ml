# CHIRON PIED — Implementation Record (E0/E1 PASS, E3-ready)

**Date:** 2026-07-01
**Status:** Minimal prototype implemented, default-off, E0 + E1 + production-shape smoke PASS.
E2/E3 pre-registered and ready to run (commands below). NOT yet a training result — no claim
about perplexity is made here.
**Design:** `docs/superpowers/specs/2026-07-01-chiron-pied-increment-dropout-design.md`
(3-candidate research-framework-design synthesis: PIED selected over WhiSK [symplectic kicks]
and SFD [spectral flow dropout]; see design §2–3).

## What PIED is

Mean-one two-point Bernoulli mask `η ∈ {0, 1/(1−π)}` on the SCFA attention increment at the
shear commit, `p += sign·η⊙(y_par+y_perp)`, i.i.d. per (layer, µstep, token, channel),
regenerated from a stateless counter hash — the CHIRON-native dropout (gate contributions on
the linear p-accumulator, never states; exact ensemble semantics; exactly invertible in the
inverse walk). Training passes only; val/inference never mask. Parameter-free: no checkpoint
delta, no serving change.

## What landed

**glades-ml (library):**
- `gpu_chiron.cu`: `chiron_scfa_axpy2_masked` (masked commit; sign=−1 is the exact inverse) +
  `chiron_incdrop_scale_copy` (masked dy hand-off) + the `chiron_pied_mix32/eta` device hash
  (lowbias32-mix; keyed `mix32(key ^ i·0x9E3779B9)`).
- `gpu_chiron.h`: declarations + no-CUDA inline fallbacks.
- `transformer_chiron_ops.h`: CPU references (`chiron_pied_mix32/eta`,
  `chiron_scfa_axpy2_masked_cpu`, `chiron_incdrop_scale_copy_cpu`) — bit-identical hash.
- Unit tests `test.sh chiron-pied` (chiron-test.cpp): mask determinism; E[η]=1 within 4σ over
  2²⁰ draws (Bernoulli + symmetric arms); keep-rate within 4σ; zero-rate bit-identity;
  commit+inverse reconstruction at synthetic ρ=45; commit/dy η-field bit-agreement;
  CPU/GPU η bit-parity; GPU value parity; GPU commit+inverse reconstruction.

**glades-trainer:**
- Flags `--inc-dropout <π>` (default 0.0 = bit-exact off), `--inc-dropout-symmetric`
  (ablation arm η∈{1−a,1+a}, matched variance, no deletion); help text; run.sh passthrough.
- Guards (fail-fast): π∈[0,0.95); requires `--scfa`; incompatible with `--cuda-graphs`
  (replay would freeze the mask key) and `--sfa-swap-layer`. `--bf16-residual-p` (default-on)
  IS supported: the commit falls back to the unfused masked-axpy2 + SR-cast pair (dual_p is
  not mask-aware) — one extra [T×m] pass per commit only while PIED is active.
- Wiring: `forward()` arms `s.piedActive/piedStepKey` on EVERY pass (val passes
  isTraining=false → inactive — the LayerDrop val-mode lesson is structurally handled); the
  training loop sets `s.piedMicroIdx = microStep % accumN`; key = pure function of
  (seed, stepForDrift, µidx, layer) — no RNG state, resume-reproducible.
- Insertion points (all three consistent by the shared key):
  1. forward commit in `scfa_attention_forward` (masked axpy2, sign=+1);
  2. inverse-commit in `scfa_attention_backward` (same kernel, sign flipped — bit-identical
     subtracted increment by IEEE sign-flip);
  3. dy hand-off (masked scale-copy replacing the scaled_copy/memcpy; the iter116 dy≡dp alias
     and the castElimDy dp-mirror registration are disabled while PIED is active since
     dy = η⊙dp ≠ dp; the through-going s.dp trunk adjoint is never masked).

## Evidence

**E1 (unit tests, all PASS):**
- η field CPU/GPU **bit-exact** (8192 elems).
- Masked commit CPU/GPU maxErr 3.81e-06 (FMA-contraction class; bar 1e-4).
- GPU commit+inverse reconstruction @ρ=45: maxRelErr **1.09e-07** (bar 1e-5) — the unmasked
  shear's tolerance class, as designed.
- Mean-one/keep-rate/symmetric-support/zero-rate-identity all pass. Existing `chiron-whisc`
  (28 asserts) and `chiron-rot` (8) suites unchanged — no regression.

**E0 (production shape, flagship WhiSC recipe, seed 1337, 2 steps + per-step 4-batch val):**
- Baseline vs `--inc-dropout 0.0`: **train steps bit-identical** (step1 loss 10.7816
  ‖g‖ 0.831; step2 10.7374 ‖g‖ 1.079; sorc/whisc monitor lines identical). Val nll differs in
  the 4th decimal — and an identical-command baseline rerun (a-vs-a2 control) shows the SAME
  jitter class, i.e. the documented atomic-ordering val noise, not a flag effect. **E0 PASS.**

**Smoke (mechanism active, same recipe):**
- `--inc-dropout 0.1`: step1 loss 10.7706 (differs from baseline as designed), **‖g‖ 0.833 vs
  baseline 0.831** — no gradient inflation (the R1 ≤ 1/(1−π) bound in vivo); 0 skips, 0 NaN;
  val sane and noise-free; `[whisc]` monitor nominal (a_max 1.063→1.12 matches baseline).
- `--inc-dropout 0.1 --inc-dropout-symmetric`: clean, distinct third trajectory (10.7714),
  ‖g‖ 0.832.
- tok/s at step-2: 19293 vs baseline 19872 (≈−3% on a 2-step window incl. warmup/val — NOT a
  perf claim; the bf16-residual-p fallback adds one [T×m] pass per commit; measure properly at
  E3 against the ≤2% budget, and consider a masked dual_p kernel if it matters).

## Next (pre-registered in the design doc §13)

- **E2** small-shape: E_η[logits] unbiasedness vs the O(θ_max²) leak bound; measured Δ_tax ≈
  R_PIED numerical check.
- **E3 (~3.4 GPU-hr, the gate before any 30k spend):**
  ```bash
  cd ~/dev/glades-trainer && sh run.sh flagship --steps 2500 --accum 4 --lr 3e-4 --warmup 750 \
    --sira-warmup 250 --zloss-coef 1e-4 --qk-norm --sira-coef 1e-2 --sira-energy-weight 1.0 \
    --sira-balance-weight 0.25 --sira-action-weight 0.0 --grad-clip 0.5 --dq-layer-clamp 1.0 \
    --dq-embed-clamp 1.0 --reln-reanchor --whisc-coupling --rot-theta-max 0.07 --seed 1337 \
    --inc-dropout 0.1 --save <ckpt-dir>/chiron_1B_pied_e3
  ```
  vs the matched no-pied baseline (same command minus `--inc-dropout`). PASS bars: no
  divergence, 0 skips, ‖g‖ ≤ 1.1× baseline max, val@2500 ≤ baseline+0.10 with the gap
  SHRINKING across {1k,1.5k,2k,2.5k}, measured Δ_tax ∈ [0.01,0.05] (else one retry at π=0.05),
  wall ≤ 2%.
- **E4** matched 30k vs whisc30k — ship bar Δ ≤ −0.02 nat; kill ≥ +0.02; neutral zone keeps
  default-off and reads the mechanism observables (Herfindahl concentration, ablation
  robustness, gen-metrics).

## Gotchas recorded

- The trainer **vendors** the glades headers under `glades-trainer/include/` (plain copies,
  not symlinks) — after changing `gpu_chiron.h` / `transformer_chiron_ops.h` in glades-ml,
  copy them over, then `make install` (libGladesCUDA.a) + `bash build.sh` (static link).
- `--bf16-residual-p` is **default-ON** in the trainer config (iter 68 ship) even though it is
  not in the run.sh STACK line — any new commit-site mechanism must handle it (PIED's first
  guard attempt wrongly blocked the production path; fixed by the unfused fallback).
- SR-cast counters are file-static and increment per call — a mechanism must NOT rely on them
  for regenerable randomness (PIED uses pure counter keys instead; the SR mirror refresh uses
  its own fresh statics, which only seed rounding noise).
