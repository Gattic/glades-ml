# CHIRON FineWeb Causal 275k Continuation — Approval Packet

**Prepared:** 2026-07-15  
**Authorization state:** **NOT APPROVED / NOT RUNNING**  
**Proposed lineage:** verified causal A2 checkpoint at global step 1000  
**Historical gate verdict:** **PARTIAL** — the preregistered zero-step aggregate NLL delta missed by `0.0045`; this packet does not rewrite that result.

## Immutable implementation and checkpoint pins

- `glades-ml`: `142d8802e430e208d38cabad08807b00203ac0e4` (`chiron: make SCFA and WhiSC prefix-causal`).
- `glades-trainer`: `7b339636e8e09888a502778cfed9255004445b26` (`chiron: enforce causal SCFA resume state`).
- Resume checkpoint: `/media/robert/AI1/chiron_fineweb_20B_causal_v1_a2/run/a2.final`.
- Resume checkpoint SHA-256: `a91266941eecc8d66c0a9209d228bb12188b320a268e4846eda22056dd0fa4ee`.
- Resume header: CHRF v4, step `1000`, flags `0x1fd8`; persisted SCFA, causal marker, and WhiSC `Pbar/Qbar/a` are present.
- Protected legacy artifacts SHA-256: `26f95190e475233266229e5ea402d36ac49563d1d4f719e8a6a7beb1f2cf4f68`.

Only the two code commits and A2 checkpoint above define the proposed causal continuation. Legacy flags `0x7d8` are prohibited as a resume source.

## Evidence carried into the approval decision

- A2 reached global step 1000 with zero true gradient skips or non-finite metrics.
- Last logged loss `8.5471`; best loss `8.4010`; maximum logged global gradient `4.889`.
- Final validation NLL `8.5043`, improving on A1 `8.8922`.
- All A2 checkpoints have flags `0x1fd8` and finite WhiSC/tail state.
- Full-window versus prefix-only logits are bit-identical at tested positions.
- A2 checkpoint body round-trip is byte-identical from byte 60 through EOF.
- Strict E0 aggregate NLL delta was `+0.0145` against `≤0.01`: **MISS by `0.0045`**, retained as `PARTIAL`.

See `CHIRON_FINEWEB_A2_GATE_REPORT_2026_07_15.md` for the complete adjudication.

## Proposed continuation

- Start: global step `1000`.
- Stop target: global step `275000`.
- New optimizer steps: `274000`.
- Remaining tokens: `17,956,864,000` at `65,536` tokens/step.
- Planning throughput: A2 mean `29,081 tok/s`.
- Estimated uninterrupted runtime: `617,478 s`, approximately **7.15 days**. Allow **6.5–8 days** for validation, checkpoint I/O, and throughput variation.
- Checkpoint size: `3,513,345,100` bytes (`3.27 GiB`). Default keep-last 3 plus final is about `13.1 GiB`; reserve at least `20 GiB` for write overlap and logs. AI1 had approximately `1.1 TiB` free when this packet was prepared.
- Isolated output root: `/media/robert/AI1/chiron_fineweb_20B_causal_v1_run1`.

## Exact proposed command — approval guard intentionally blocks execution

Do not remove the approval guard until the owner explicitly authorizes the run. When approved, launch this command through `run_deck` rather than an unmanaged `nohup` process.

```bash
set -euo pipefail
: "${CHIRON_275K_APPROVED:?owner approval required; do not launch}"

export DATA=/media/robert/AI1/fineweb-pretok-sample350BT
export RESUME=/media/robert/AI1/chiron_fineweb_20B_causal_v1_a2/run/a2.final
export OUT=/media/robert/AI1/chiron_fineweb_20B_causal_v1_run1
export EXPECTED_A2_SHA=a91266941eecc8d66c0a9209d228bb12188b320a268e4846eda22056dd0fa4ee
export EXPECTED_PROTECTED_SHA=26f95190e475233266229e5ea402d36ac49563d1d4f719e8a6a7beb1f2cf4f68

test "$(sha256sum "$RESUME" | awk '{print $1}')" = "$EXPECTED_A2_SHA"
test "$(sha256sum /media/robert/AI1/chiron_fineweb_20B_stage1_final_backup.ckpt | awk '{print $1}')" = "$EXPECTED_PROTECTED_SHA"
! pgrep -f '[g]lades_chiron_train'
mkdir -p "$OUT" /home/robert/dev/glades-trainer/logs

cd /home/robert/dev/glades-trainer
PRETOK_DIR="$DATA" \
sh run.sh flagship \
  --load "$RESUME" \
  --steps 275000 --accum 4 --lr 3e-4 --warmup 2000 \
  --sira-warmup 250 --zloss-coef 1e-4 --qk-norm \
  --sira-coef 1e-2 --sira-energy-weight 1.0 --sira-balance-weight 0.25 --sira-action-weight 0.0 \
  --grad-clip 0.5 --dq-layer-clamp 1.0 --dq-embed-clamp 1.0 --reln-reanchor \
  --whisc-coupling --rot-theta-max 0.07 --inc-dropout 0.1 --seed 1337 \
  --val-every 5000 --val-batches 8 \
  --save "$OUT/chiron_fineweb_20B_causal_v1" \
  --save-every 25000 \
  > /home/robert/dev/glades-trainer/logs/fineweb_20B_causal_v1_run1.log 2>&1
```

The built-in resume mini-warmup remains enabled because the pretokenized stream position is not serialized.

## Required pre-launch gates

1. Both repositories are clean and resolve to the pinned commits above.
2. Installed `glades-ml` and the trainer binary were rebuilt in the required order from those commits.
3. Full CHIRON tests, all checkpoint self-test modes, and legacy `0x7d8` refusal pass.
4. A production-sized zero-step A2 load restores causal SCFA and WhiSC state with no optimizer step.
5. A2 and protected hashes match this packet before opening the trainer.
6. No trainer process is alive and the GPU is idle.
7. Output root is new and has at least `20 GiB` available.
8. The owner explicitly approves this exact continuation despite the recorded strict `PARTIAL` A2 verdict.

## Early stop gates

Stop and retain all evidence if any condition occurs:

- loader does not report causal SCFA plus persisted WhiSC state;
- any non-finite loss, gradient, WhiSC statistic, or validation metric;
- any true `[grad-skip]` event;
- logged global gradient norm reaches `10` or higher;
- first post-resume loss is grossly inconsistent with the A2 neighborhood (`8.4–8.9`);
- validation exceeds `9.3922` for two consecutive gates;
- any retained checkpoint lacks flags `0x1fd8`, finite state, or exact EOF alignment;
- either protected legacy hash changes;
- output is directed anywhere under a protected checkpoint prefix.

## Decision field

- [ ] **APPROVE** the exact causal A2 → step-275000 continuation above.
- [ ] **NO-GO**; retain A2 as the terminal causal artifact.
- [x] **PENDING**; no 275k process may be started.
