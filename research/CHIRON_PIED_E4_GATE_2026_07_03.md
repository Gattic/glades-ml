# CHIRON PIED — E4 Gate Result: PASS, deep ship zone (2026-07-03)

**Status:** E4 **PASS** — the pre-registered decisive perplexity gate. PIED at matched 30k
beats the whisc30k production flagship by **Δ ≈ −0.35 nat (4-batch final-val) and
−0.46/−0.52 nat on the wide-32 ship metric at matched windows**, with top-1 up ~+9–11
points, 0 grad-skips, and the run's only ‖g‖ event being the flagship's own step-9001
lineage spike at less than half the magnitude. Margins are 3–5× beyond the identified
era-drift allowance. **Ship promotion is an owner decision** (single-seed precedent applies;
serving needs zero changes — the checkpoint is a standard WhiSC-format checkpoint).
**Ladder:** design `docs/superpowers/specs/2026-07-01-chiron-pied-increment-dropout-design.md` (§13);
E0/E1 `research/CHIRON_PIED_IMPLEMENTATION_2026_07_01.md`; E3 `research/CHIRON_PIED_E3_GATE_2026_07_02.md`.

## Setup

Treatment = the exact whisc30k flagship recipe + `--inc-dropout 0.1` (seed 1337, accum 4,
lr 3e-4 constant, 30k steps, T=16384, save-every 6000):

```
sh run.sh flagship --steps 30000 --accum 4 --lr 3e-4 --warmup 750 --sira-warmup 250 \
  --zloss-coef 1e-4 --qk-norm --sira-coef 1e-2 --sira-energy-weight 1.0 \
  --sira-balance-weight 0.25 --sira-action-weight 0.0 --grad-clip 0.5 \
  --dq-layer-clamp 1.0 --dq-embed-clamp 1.0 --reln-reanchor \
  --whisc-coupling --rot-theta-max 0.07 --save-every 6000 --seed 1337 --inc-dropout 0.1
```

Checkpoint: `database/checkpoints/chiron_1B_pied_e4/chiron_1B_pied_e4.final`.
Log: `logs/pied_e4_treatment_20260702_1054.log`. Wall 78,344 s (21.76 h), ~25,105 tok/s.
Baseline = the whisc30k flagship run itself (per the §13 pre-registration), val trajectory
step-confirmed from its hourly logs; wide-32 windows from its wideval log (2026-07-01-H23).

## Primary result — 4-batch final-val (matched val batches, same seed/data order)

| metric | whisc30k (flagship) | PIED π=0.1 | Δ |
|---|---:|---:|---:|
| val NLL @30000 | 1.3753 | **1.0259** | **−0.349** |
| acc@1 | 0.633 | **0.7224** | **+0.089** |

Ship bar was Δ ≤ −0.02 (val ≤ 1.3553): cleared by **17×** the bar.

## Wide-32 confirm — the ship metric (two 33.5M-token windows, matched window identity)

Run via the whisc30k wideval protocol (lr=0 + `--whisc-ema 1.0` resume trick, val-batches 32;
log `logs/pied_e4_wideval_20260703_0840.log`; window A = val@30001, window B = the next window):

| window | whisc30k | PIED | Δ |
|---|---:|---:|---:|
| A (batches 1–32) | 1.6390 / acc1 0.5813 | **1.1788 / 0.6870** | **−0.461 / +0.106** |
| B (batches 33–64) | 1.8191 / acc1 0.5594 | **1.3019 / 0.6679** | **−0.517 / +0.109** |
| C (batches 65–96) | — | 1.5730 / 0.6087 | (no counterpart; windows get harder deeper into the val set) |

For lineage context: the reanchor flagship's wide val was 1.92 at 66k+finish; whisc30k's
1.639/1.819 at 30k; **PIED 1.179/1.302 at the same 30k** — each generation at a fraction of
the predecessor's total effort.

## Full matched val trajectory

| step | whisc30k | PIED | Δ |
|---:|---:|---:|---:|
| 3000 | 2.8244 | 2.6370 | −0.187 |
| 6000 | 2.1690 | 2.0091 | −0.160 |
| 9000 | 1.9342 | 1.7454 | −0.189 |
| 12000 | 2.0979 | 1.7676 | −0.330 |
| 15000 | 1.6751 | 1.3213 | −0.354 |
| 18000 | 1.7662 | 1.3616 | −0.405 |
| 21000 | 1.8376 | 1.3974 | −0.440 |
| 24000 | 1.5778 | 1.2029 | −0.375 |
| 27000 | 1.5824 | 1.1701 | −0.412 |
| 30000 | 1.3753 | 1.0259 | −0.349 |

The gap **widens with training** (−0.16 early → −0.35..−0.44 late) — the E3 concern that the
mid-run advantage would wash out at long horizons is refuted; the opposite occurred. PIED
crossed whisc30k's *final* quality at step ~15k (half the budget).

## Stability census

- **0 grad-skips, 0 NaN** over 30k steps / 1.97B tokens.
- Max ‖g‖ = 4.895 at **step 9001** — the identical step where the whisc30k flagship logged its
  only anomaly (recovered spike, ‖g‖ 10.8). Same seed/data ⇒ same hard batch; **PIED damped
  the lineage event to less than half the magnitude** — direct in-vivo evidence for the
  design's conditioning claim (Fisher-weighted increment-energy penalty), consistent with E3
  (treatment max ‖g‖ 2.958 < baseline 3.211).
- `[whisc]` monitor nominal throughout (a-clamp at 0.125, ρ_eff ~4.1e3 — flagship-identical).

## Honest caveats

1. **Era drift.** The baseline is the 2026-07-01 whisc30k run, not a same-binary paired arm.
   The measured era-drift scale (fresh E3 baseline vs whisc30k at matched effort) is
   ~0.10–0.15 nat early-run — the E4 margins (−0.35 final, −0.46/−0.52 wide) are 3–5× beyond
   it, so the direction and ship-zone verdict are drift-robust. The *precise attribution* of
   the total to PIED alone is bounded below by the E3 paired measurement and above by these
   numbers.
2. **Single seed (1337).** Same as the whisc30k and reanchor ship precedents; multi-seed
   (≥3) not run. The magnitude (largest matched-30k jump in the lineage after WhiSC-D's own
   −1.27) dwarfs documented seed variance (~0.02), but cross-seed reproduction is not
   established.
3. **Wall.** Same-binary A/B: PIED costs −1.87% (perf-pass measurement, inside the ≤2%
   budget). Cross-era: this run's wall (78,344 s) is +2.6% vs the whisc30k run's — the
   residual ~0.7% is binary-era difference, not PIED.
4. **Perplexity result, not a generation result.** The lineage's repetition-attractor caveat
   is untouched by this gate; PIED's anti-repetition conjecture (secondary endpoint) has not
   been evaluated here.

## Disposition — SHIPPED 2026-07-03 (owner decision)

- **E4 PASS → owner shipped PIED as the production flagship** (single-seed, wide-32 banked,
  matched recipe — the lineage's own standard).
- **TF parity check (pre-ship, PASS):** chiron_infer teacher-forcing on
  `chiron_1B_pied_e4.final` via `runner.sh --flagship`: **nll 1.0897 / top1 0.7176**
  (vs trainer final-val 1.0259/0.7224; prior whisc30k ship TF was 1.4655/0.626).
  QK-Norm exact-γ + WhiSC rot_phi auto-loaded; forward verdict CORRECT.
- **Serving promotion:** `runner.sh --flagship` now prefers
  `chiron_1B_pied_e4/chiron_1B_pied_e4.final` (whisc30k = first fallback), and the WhiSC
  flag-injection case was extended to `*whisc*|*pied*` (the PIED checkpoint is WhiSC
  bit-1024 format; without the extension chiron_infer would exit-7). End-to-end smoke via
  `runner.sh --flagship --tf-check` verified.
- **CLAUDE.md promoted** (PIED section at top; WhiSC-D demoted to prior-flagship context).
- Open follow-ups (from the design §14): mechanism-separation arm (`--inc-dropout-symmetric`),
  per-head (B2) / spectral (SFD-G1) variants, π sweep, multi-seed, gen-metrics read on the
  E4 checkpoint.
