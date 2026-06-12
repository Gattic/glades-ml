# SIRA+clamp Flagship Ship Record (2026-06-12)

**New production flagship:** `chiron_1B_T16384_sira_clamp_phase2.final`
(canonical copy of `chiron_sira_layerclamp_tau1_seed2024_20260611_081639.final`,
the seed-2024 per-layer-clamp PASS run).

**Recipe** (all flags default-off; omitting them reproduces the regstack
Phase 2 ship):

```bash
cd ~/dev/glades-trainer && sh run.sh flagship \
  --zloss-coef 1e-4 --qk-norm \
  --sira-coef 1e-2 --sira-energy-weight 1.0 --sira-balance-weight 0.25 \
  --sira-action-weight 0.0 --sira-warmup 1000 \
  --lr 7.5e-5 --grad-clip 0.5 \
  --dq-layer-clamp 1.0 --dq-embed-clamp 1.0
```

**Headline numbers** (vs regstack Phase 2 ship, val NLL 3.5734 @ 30k,
28,072 tok/s):

| metric | SIRA+clamp ship | Δ |
|---|---:|---:|
| Final val NLL (seed 2024) | **3.5062** | **−0.0672** |
| 3-seed mean (2024/4242/1337) | 3.5223 ± 0.0146 | −0.0511 |
| Warm tok/s | ~27,200 | −3.1% |
| VRAM | ~14.7/15.56 GB | ≈ |
| Grad-skips across 3×30k | **0** | (was 5,809 at seed 2024) |

Attribution (matched no-SIRA baseline, same recipe/binary/seed): LR
7.5e-5/clip 0.5 recipe ≈ −0.036 nat; SIRA terminal loss ≈ −0.031 nat.

**Mechanisms shipped in the recipe:**
- Terminal-only CHIRON-native SIRA (energy 1.0 / balance 0.25; opt-in since
  2026-05-27, promoted after the full six-criterion gate).
- `--dq-layer-clamp 1.0`: per-row RMS clamp (`row_rms_clamp`, glades-ml) on
  the incoming dq at every backward layer boundary — bounds per-layer
  dgamma/dbeta and breaks the frozen-weight skip-wave loop.
- `--dq-embed-clamp 1.0`: same kernel on dq_0 before `embedding_scatter_add`.
- Both clamps fired on exactly one of three gate seeds (2024's data-driven
  burst, 23.8–24.3k window, self-terminated) and were inert elsewhere —
  zero firings on 4242/1337, so those runs are math-identical to unclamped.

**Full evidence chain:** `research/SIRA_TERMINAL_30K_RESULT_2026_05_27.md`
(candidate runs, seed-2024 failure, overflow-path tracing, dE-clamp FAIL,
per-layer PASS, criterion-2 attribution, multi-seed gate).
**Mechanism spec:** `docs/superpowers/specs/2026-06-11-dq-embed-clamp-design.md`.
**Known caveats:** same-seed full-recipe runs are not bit-reproducible at
production shape (atomic-ordering noise — use rerun controls for parity
claims); the 21k val bump + bucket-3 spike is a data-window feature on all
seeds (recovers by 24k); prior flagships remain loadable (clamps/SIRA are
trainer-side and default-off).
