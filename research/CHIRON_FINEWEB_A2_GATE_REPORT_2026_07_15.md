# CHIRON FineWeb Causal A2 — Gate Report

**Date:** 2026-07-15  
**Run:** `run-chiron-fineweb-causal-a2-step1000-460e`  
**Operational decision:** **ADVANCE to the approval gate only**; this is not authorization for the 275k run.  
**Strict preregistered verdict:** **PARTIAL**, because one numeric E0 bar missed; all continuation, numerical, checkpoint, and protection bars passed.

## Pinned causal implementation

- `glades-ml` causal source/test commit: `142d8802e430e208d38cabad08807b00203ac0e4`.
- `glades-trainer` causal resume-contract commit: `7b339636e8e09888a502778cfed9255004445b26`.
- These commits are the implementation lineage for any A2 validation or proposed continuation.
- The 275k continuation remains stopped until the owner explicitly approves the separate approval packet.

## Result

- Resume source: A1 `pilot.final`, CHRF v4 step 150, flags `0x1fd8`.
- Completion: exit 0, global step 1000, 850 new steps, 55,705,600 additional tokens.
- Main loss: `8.8953` at step 151 → `8.5471` at last logged step 976; best `8.4010` at step 952.
- Gradient norm: maximum `4.889` at step 951; zero true `[grad-skip]` events.
- Non-finite main/SIRA/WhiSC/validation metrics: zero.
- Mean steady throughput: `29,081 tok/s`.
- Validation NLL: `8.9067 → 8.8843 → 8.8265 → 8.8336 → 8.8220 → 8.8004 → 8.7572 → 8.5592 → 8.5726 → 8.5043`.
- Final validation delta versus A1 `8.8922`: `-0.3879`; no two-gate regression above the `9.3922` stop threshold.
- Retained checkpoints at steps 500, 750, 1000, plus final: flags `0x1fd8`, size 3,513,345,100 bytes, finite WhiSC/tails.
- Protected Stage 1 final and backup remain SHA-256 `26f95190e475233266229e5ea402d36ac49563d1d4f719e8a6a7beb1f2cf4f68` with unchanged metadata.

## Zero-step resume adjudication

The zero-step process restored persisted `Pbar/Qbar/a`, ran no optimizer step (`total_tokens=0`), and did not invoke legacy calibration.

- Aggregate NLL: `8.9067` versus preregistered A1 final `8.8922`, delta `+0.0145`.
- Preregistered bar: delta at most `0.01`.
- Result: **bar miss by 0.0045**. This bar cannot honestly be marked pass.
- Deterministic replacement evidence: A1 source and zero-step re-save are byte-identical from CHRF byte 60 through EOF. Only `slcLast` header metadata changed from `-1` to `150` to re-arm the intended resume warmup. Thus weights, optimizer state, SCFA D, QK gamma, WhiSC state, `a_drift`, and `rot_phi` round-tripped exactly.

The aggregate NLL miss reflects different validation-stream batches after process restart, not checkpoint drift. It remains recorded as `PARTIAL` rather than rewriting the preregistered result.

## Preregistered bars

| Bar | Verdict |
|---|---|
| A1 source v4/step150/`0x1fd8`; no legacy calibration | PASS |
| Zero-step persisted WhiSC restore; no optimizer step | PASS |
| Zero-step aggregate NLL delta ≤ `0.01` | **MISS (`+0.0145`)** |
| Deterministic CHRF body parity | PASS (byte-exact) |
| Isolated A2 outputs/logs | PASS |
| No NaN/Inf; zero true gradient skips | PASS |
| Logged global gradient norm `<10` | PASS (`4.889` max) |
| Reach step 1000, exit 0 | PASS |
| Final validation below A1 | PASS (`8.5043`) |
| No two-gate validation regression | PASS |
| Every retained checkpoint `0x1fd8`, finite state | PASS |
| Protected hashes unchanged | PASS |

## Evidence

- Raw run log: `/home/robert/dev/glades-trainer/logs/fineweb_causal_v1_a2.log`  
  SHA-256 `738791d50dda009104b47ba894042e47b81edfe0f1579d7dcf24e4d1a2207817`
- Raw zero-step log: `/home/robert/dev/glades-trainer/logs/fineweb_causal_v1_a2_preflight.log`  
  SHA-256 `18219673e151d1d17834cec5af25fafeacc15ad363c83c1253452d0b7614ae35`
- Final checkpoint: `/media/robert/AI1/chiron_fineweb_20B_causal_v1_a2/run/a2.final`  
  SHA-256 `a91266941eecc8d66c0a9209d228bb12188b320a268e4846eda22056dd0fa4ee`
- Zero-step checkpoint: `/media/robert/AI1/chiron_fineweb_20B_causal_v1_a2/preflight/a2_preflight.final`  
  SHA-256 `6ea992653bdb3b54a7b29b2bfafb29968c84af773f502884d8df214d3304f145`
- Parsed evidence packet: `/tmp/track-a-a2-artifact-verification.json`
- Evidence hashes: `/tmp/track-a-a2-evidence-sha256.txt`
- Protected post-run check: `/tmp/track-a-a2-protected-after.txt`

`run_deck`'s displayed `skips=1` and auxiliary loss were parser false positives from the informational phrase “backward skips step 1” and SIRA loss. Direct raw-log parsing found zero true `[grad-skip]` records and used only main `[step ...]` lines for loss/gradient metrics.
