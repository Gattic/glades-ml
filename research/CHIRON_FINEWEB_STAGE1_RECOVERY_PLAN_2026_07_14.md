# CHIRON FineWeb Stage 1 — Non-destructive Recovery Plan

> **Status (2026-07-15):** causal A1/A2 gates completed; A2 is operationally eligible for an approval decision but retains a strict `PARTIAL` verdict. No 275k run or legacy migration is authorized by this document.
> The protected Stage 1 checkpoint must never be used as a save target.
>
> **Pinned causal code:** `glades-ml` `142d8802e430e208d38cabad08807b00203ac0e4`; `glades-trainer` `7b339636e8e09888a502778cfed9255004445b26`.

## Decision summary

There are two distinct recovery objectives, and they must not be mixed:

1. **Recommended production recovery: train a valid causal model.** Start a new causal Stage 1 from random initialization under a new save prefix. The protected `0x7d8` checkpoint cannot seed this run because its SCFA weights were trained with global future leakage.
2. **Optional legacy forensic salvage: continue the historical teacher-forced run.** Use isolated, pinned pre-causal binaries to calibrate the missing WhiSC state into a new checkpoint, then resume only from that new checkpoint. This remains global-DCT/noncausal and is not eligible for autoregressive serving or promotion.

There is **no supported current-binary resume command for the protected checkpoint**. The current causal loader must continue to reject it before validation or optimization.

## Immutable inputs and new output roots

```bash
export STAGE1=/media/robert/AI1/chiron_fineweb_20B/chiron_fineweb_20B.final
export PROTECTED=/media/robert/AI1/chiron_fineweb_20B_stage1_final_backup.ckpt
export EXPECTED_SHA=26f95190e475233266229e5ea402d36ac49563d1d4f719e8a6a7beb1f2cf4f68
export DATA=/media/robert/AI1/fineweb-pretok-sample350BT
export LEGACY_OUT=/media/robert/AI1/chiron_fineweb_20B_legacy_recovery
export CAUSAL_OUT=/media/robert/AI1/chiron_fineweb_20B_causal_v1
```

Hard rules:

- Never pass `$STAGE1`, `$PROTECTED`, or their parent prefix to `--save`.
- Do not rename, truncate, chmod, `chattr`, or copy over either protected artifact.
- All migration, pilot, and finish outputs go to a newly created directory.
- Keep Stage 2 stopped until the selected track passes every gate.
- Re-check the protected hash before and after every operation that opens a trainer.

Read-only protection gate:

```bash
set -euo pipefail
for f in "$STAGE1" "$PROTECTED"; do
  test -f "$f"
  test "$(sha256sum "$f" | awk '{print $1}')" = "$EXPECTED_SHA"
done
stat -c '%n size=%s mtime=%y inode=%i' "$STAGE1" "$PROTECTED"
pgrep -af '[g]lades_chiron_train' && { echo 'trainer already running' >&2; exit 1; } || true
```

Expected metadata for both inputs:

```text
CHRF v4; step=275000; T=16384 m=2048 L=24 nH=16 dH=256 V=32000
flags=0x7d8; size=3512755276
```

`0x7d8` has SCFA and `rot_phi`, but neither WhiSC state (`0x800`) nor causal SCFA (`0x1000`).

## Implemented and verified causal safeguards

The pinned causal source contains:

- `CKPT_BIT_WHISC_STATE = 2048`.
- CHRF serialization of physical-depth `Pbar`, `Qbar`, and next-forward `a` before `a_drift`/`rot_phi`.
- Restore into both next-forward and current-forward WhiSC buffers.
- A one-forward, no-gradient, `ema=1` calibration path for legacy checkpoints.
- `CKPT_BIT_CAUSAL_SCFA = 4096` and refusal of markerless global-DCT SCFA checkpoints.
- A prefix-invariance unit regression and checkpoint round-trip self-test.

The legacy calibration path diagnoses/migrates historical state only. It must not bypass the causal-SCFA refusal in the current production binary.

Verification commands:

```bash
# Current causal implementation.
cd /home/robert/dev/glades-ml
# Agent/harness gate: build_project, then test_project(filter="chiron")

# Required install/build order.
cd /home/robert/dev/glades-ml/build
make install
cd /home/robert/dev/glades-trainer
bash build.sh

# WhiSC Pbar/Qbar/a round-trip, all checkpoint modes.
./build/glades_chiron_train --checkpoint-self-test \
  > /tmp/chiron-recovery-checkpoint-selftest.log 2>&1
grep -q 'ALL TESTS PASSED' /tmp/chiron-recovery-checkpoint-selftest.log

# Protected legacy SCFA must be rejected before an optimizer step.
bash scripts/chiron_whisc_resume_regression.sh "$PROTECTED"
```

Required results:

- Full CHIRON suite passes.
- Checkpoint self-test reports all tests passed.
- Legacy refusal script reports `flags=0x7d8 rejected; zero optimizer steps`.
- Prefix-only versus full-window logits are bit-identical at the same positions.

Evidence already banked on 2026-07-14:

- Full CHIRON suite passed in 304 seconds.
- WhiSC checkpoint self-test passed for FP32, Kahan, and BF16-on-disk modes.
- Before the causal interlock, isolated legacy calibration recovered trainer NLL `0.362100` versus serving NLL `0.359665` with zero optimizer steps.
- Current legacy-refusal regression passed with zero optimizer steps.
- Causal full-window versus prefix-only logits had `max_abs=0` at real positions 0 and 128 and in the unit regression.

## Track A — recommended causal recovery

### A1. Preflight pilot from random initialization

Do not use `--load`. Write only to the new causal root:

```bash
set -euo pipefail
mkdir -p "$CAUSAL_OUT" /home/robert/dev/glades-trainer/logs
cd /home/robert/dev/glades-trainer

PRETOK_DIR="$DATA" \
sh run.sh flagship \
  --steps 150 --accum 4 --lr 3e-4 --warmup 100 \
  --sira-warmup 250 --zloss-coef 1e-4 --qk-norm \
  --sira-coef 1e-2 --sira-energy-weight 1.0 --sira-balance-weight 0.25 --sira-action-weight 0.0 \
  --grad-clip 0.5 --dq-layer-clamp 1.0 --dq-embed-clamp 1.0 --reln-reanchor \
  --whisc-coupling --rot-theta-max 0.07 --inc-dropout 0.1 --seed 1337 \
  --val-every 50 --val-batches 4 \
  --save "$CAUSAL_OUT/pilot" --save-every 50 \
  > logs/fineweb_causal_v1_pilot.log 2>&1
```

Pilot gates:

- Loss descends from random-init scale; no NaN/Inf or gradient skip.
- Full-vs-prefix logits from `pilot.final` agree exactly at selected interior positions.
- Header contains both WhiSC and causal markers. For the same full recipe, expected flags are `0x1fd8` (`0x7d8 | 0x800 | 0x1000`).
- Generation's first token agrees with prefix-forward argmax.

### A2. No-gradient save/resume gate — completed with strict `PARTIAL`

The causal checkpoint loaded with its saved step as `max_steps`, performed zero optimizer steps, and restored persisted WhiSC state without legacy calibration.

Recorded results:

- `total_tokens=0` and no `[step ...]` line: pass.
- Persisted `Pbar/Qbar/a` restored; no legacy calibration: pass.
- Same-position logits and CHRF body bytes were deterministic: pass.
- Aggregate validation NLL delta was `+0.0145` against a preregistered `≤0.01` bar: miss by `0.0045` because restart selected different validation batches.
- Re-saved checkpoint remained `0x1fd8`, finite, and byte-identical from byte 60 through EOF: pass.

The miss remains part of the historical verdict and must not be rewritten. See `CHIRON_FINEWEB_A2_GATE_REPORT_2026_07_15.md`.

### A3. Historical fresh-run command template

A1/A2 and both code commits are complete, but the 275k run remains unauthorized. The current proposal continues only from the verified A2 checkpoint and is specified in `CHIRON_FINEWEB_CAUSAL_275K_APPROVAL_PACKET_2026_07_15.md`. The fresh-run template below is retained for historical reference and must not be launched without a separate explicit decision:

```bash
set -euo pipefail
mkdir -p "$CAUSAL_OUT"
cd /home/robert/dev/glades-trainer

PRETOK_DIR="$DATA" \
nohup sh run.sh flagship \
  --steps 275000 --accum 4 --lr 3e-4 --warmup 2000 \
  --sira-warmup 250 --zloss-coef 1e-4 --qk-norm \
  --sira-coef 1e-2 --sira-energy-weight 1.0 --sira-balance-weight 0.25 --sira-action-weight 0.0 \
  --grad-clip 0.5 --dq-layer-clamp 1.0 --dq-embed-clamp 1.0 --reln-reanchor \
  --whisc-coupling --rot-theta-max 0.07 --inc-dropout 0.1 --seed 1337 \
  --val-every 5000 --val-batches 8 \
  --save "$CAUSAL_OUT/chiron_fineweb_20B_causal_v1" \
  --save-every 25000 \
  > logs/fineweb_20B_causal_v1_base.log 2>&1 &
echo $! > logs/fineweb_20B_causal_v1_base.pid
```

### A4. Exact crash-resume command for a new causal checkpoint

Set `RESUME` to the newest verified `0x1fd8` periodic checkpoint. Deliberately omit `--no-resume-warmup`: the pretokenized stream offset is not persisted, so the trainer's 5,000-step resume mini-warmup is the safer policy.

```bash
export RESUME="$CAUSAL_OUT/chiron_fineweb_20B_causal_v1.stepNNNNNN"
test -f "$RESUME"

cd /home/robert/dev/glades-trainer
PRETOK_DIR="$DATA" \
nohup sh run.sh flagship \
  --load "$RESUME" \
  --steps 275000 --accum 4 --lr 3e-4 --warmup 2000 \
  --sira-warmup 250 --zloss-coef 1e-4 --qk-norm \
  --sira-coef 1e-2 --sira-energy-weight 1.0 --sira-balance-weight 0.25 --sira-action-weight 0.0 \
  --grad-clip 0.5 --dq-layer-clamp 1.0 --dq-embed-clamp 1.0 --reln-reanchor \
  --whisc-coupling --rot-theta-max 0.07 --inc-dropout 0.1 --seed 1337 \
  --val-every 5000 --val-batches 8 \
  --save "$CAUSAL_OUT/chiron_fineweb_20B_causal_v1" \
  --save-every 25000 \
  > logs/fineweb_20B_causal_v1_resume.log 2>&1 &
echo $! > logs/fineweb_20B_causal_v1_resume.pid
```

Resume gates before leaving it unattended:

- Loader reports causal SCFA and persisted WhiSC restore.
- First loss is finite and consistent with the pre-resume EMA.
- Global gradient norm `< 10`; stop immediately if `> 10` or non-finite.
- First validation is within `0.05` NLL of the pre-resume checkpoint.
- Protected hashes still equal `$EXPECTED_SHA`.

## Track B — optional isolated legacy salvage (not production)

This track exists only to recover the historical noncausal teacher-forced experiment. It must not modify the current working trees or globally installed causal library.

Pinned pre-causal/WhiSC-fix revisions:

```text
glades-ml      9cf93910a  chiron: add WhiSC checkpoint state format support
glades-trainer ba32d62    chiron: persist WhiSC calibration across resumes
```

### B1. Build isolated pinned binaries

```bash
set -euo pipefail
export LEGACY_ROOT=/home/robert/dev/chiron-stage1-legacy-recovery
export LEGACY_PREFIX="$LEGACY_ROOT/prefix"
mkdir -p "$LEGACY_ROOT"

git -C /home/robert/dev/glades-ml worktree add --detach \
  "$LEGACY_ROOT/glades-ml" 9cf93910a
git -C /home/robert/dev/glades-trainer worktree add --detach \
  "$LEGACY_ROOT/glades-trainer" ba32d62

cmake -S "$LEGACY_ROOT/glades-ml" -B "$LEGACY_ROOT/glades-ml/build" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$LEGACY_PREFIX"
cmake --build "$LEGACY_ROOT/glades-ml/build" -j "$(nproc)"
cmake --install "$LEGACY_ROOT/glades-ml/build"

cd "$LEGACY_ROOT/glades-trainer"
CMAKE_PREFIX_PATH="$LEGACY_PREFIX" BUILD_DIR=build-recovery bash build.sh
rg -n "$LEGACY_PREFIX" build-recovery/CMakeCache.txt
```

Do not run `make install` into the global causal prefix for this legacy track.

### B2. Zero-step WhiSC migration

This reconstructs an approximate calibration with one independent no-gradient `ema=1` forward, then saves it to a new checkpoint. The exact step-275000 running state is unrecoverable because it was never serialized.

```bash
set -euo pipefail
mkdir -p "$LEGACY_OUT/migrated" "$LEGACY_OUT/logs"
cd "$LEGACY_ROOT/glades-trainer"

./build-recovery/glades_chiron_train \
  --pretokenized --data-dir "$DATA" \
  --seq-len 16384 --m 2048 --layers 24 --heads 16 --dhead 256 --vocab 32000 \
  --int8-adam --bf16-grads --bf16-weights --bf16-attn \
  --no-fuse-attn --fuse-attn-reln \
  --scfa --scfa-bf16-inner --scfa-bf16-outer --scfa-fuse-streams --scfa-reln-opt \
  --bf16-logits --bf16-logits-storage --fp8-readout-fwd \
  --max-steps 275000 --warmup 0 --lr 3e-5 --grad-clip 0.5 --seed 1337 \
  --load "$PROTECTED" \
  --zloss-coef 1e-4 --qk-norm \
  --sira-coef 1e-2 --sira-energy-weight 1.0 --sira-balance-weight 0.25 \
  --sira-action-weight 0.0 --sira-warmup 250 \
  --dq-embed-clamp 1.0 --dq-layer-clamp 1.0 --reln-reanchor \
  --whisc-coupling --rot-theta-max 0.07 --inc-dropout 0.1 \
  --accum 4 --val-every 5000 --val-batches 8 \
  --save "$LEGACY_OUT/migrated/stage1_whisc_migrated" \
  > "$LEGACY_OUT/logs/migrate-zero-step.log" 2>&1
```

Migration gates:

- No `[step ...]` line and `total_tokens=0`.
- Log reports legacy WhiSC calibration before eval/update.
- Validation NLL is at most `0.37`; expected observed value is about `0.3621`.
- New checkpoint has bit `0x800`, expected flags `0xfd8`, and no causal marker.
- New file has finite `Pbar/Qbar/a`, exact EOF alignment, and `rot_phi` remains last.
- Both protected hashes remain `$EXPECTED_SHA`.

### B3. One-step guarded pilot

Before a long legacy finish, resume the migrated checkpoint for exactly one optimizer step in the foreground, writing to another new prefix. Use the default 5,000-step resume mini-warmup; do not pass `--no-resume-warmup`.

Use the same command as B2 with these substitutions:

```text
--load $LEGACY_OUT/migrated/stage1_whisc_migrated.final
--max-steps 275001
--save $LEGACY_OUT/pilot/stage1_legacy_pilot
--log-every 1 --val-every 1 --val-batches 8
```

Pilot pass bars:

- First loss `< 0.6`.
- Global gradient norm `< 10` and finite.
- Validation NLL `< 0.5`.
- WhiSC range remains finite and no optimizer group is missing.
- On any miss: stop, retain logs, and do not launch B4.

### B4. Exact legacy finish command

Only with explicit owner approval acknowledging that the result is noncausal and generation-invalid:

```bash
set -euo pipefail
mkdir -p "$LEGACY_OUT/finish"
cd "$LEGACY_ROOT/glades-trainer"

nohup ./build-recovery/glades_chiron_train \
  --pretokenized --data-dir "$DATA" \
  --seq-len 16384 --m 2048 --layers 24 --heads 16 --dhead 256 --vocab 32000 \
  --int8-adam --bf16-grads --bf16-weights --bf16-attn \
  --no-fuse-attn --fuse-attn-reln \
  --scfa --scfa-bf16-inner --scfa-bf16-outer --scfa-fuse-streams --scfa-reln-opt \
  --bf16-logits --bf16-logits-storage --fp8-readout-fwd \
  --load "$LEGACY_OUT/pilot/stage1_legacy_pilot.final" \
  --max-steps 305000 --accum 4 --lr 3e-5 --warmup 0 \
  --sira-warmup 250 --zloss-coef 1e-4 --qk-norm \
  --sira-coef 1e-2 --sira-energy-weight 1.0 --sira-balance-weight 0.25 --sira-action-weight 0.0 \
  --grad-clip 0.5 --dq-layer-clamp 1.0 --dq-embed-clamp 1.0 --reln-reanchor \
  --whisc-coupling --rot-theta-max 0.07 --inc-dropout 0.1 --seed 1337 \
  --val-every 1000 --val-batches 8 \
  --save "$LEGACY_OUT/finish/chiron_fineweb_20B_legacy_finish" \
  --save-every 5000 \
  > "$LEGACY_OUT/logs/legacy-finish.log" 2>&1 &
echo $! > "$LEGACY_OUT/logs/legacy-finish.pid"
```

`--no-resume-warmup` is intentionally omitted. The stream position is not checkpointed, so the safer recovery uses the built-in 5,000-step LR mini-warmup.

## Final go/no-go matrix

| Gate | Causal Track A | Legacy Track B |
|---|---|---|
| Protected hash unchanged | Required | Required |
| Current full CHIRON suite | Required | Required for production code |
| Prefix invariance | Exact | Expected to fail; therefore non-production |
| WhiSC state round-trip | Required | Required after migration |
| Zero-step resume parity | Required | NLL ≤ 0.37 after calibration |
| One-step gradient | `< 10`, finite | `< 10`, finite |
| Autoregressive promotion | Eligible after later eval | Prohibited |
| Long-run approval | Explicit owner approval | Separate explicit forensic approval |

## Recommended decision

Choose **Track A** for any model intended for generation. Track B can preserve research value from the eight-day Stage 1 run, but it cannot repair the checkpoint's future-token leakage and must remain isolated from causal checkpoints and serving binaries.
