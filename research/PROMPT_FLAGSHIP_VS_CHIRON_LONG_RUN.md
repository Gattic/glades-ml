# Claude Code prompt: Flagship-vs-CHIRON long-run comparison

Paste the block below into a fresh Claude Code session at `/home/robert/dev/glades-ml`.

---

```
I want to compare the latest flagship glades_pile_train (post-MLA-permanent-fix
stacked-paradigm config) against our most recent CHIRON run on a similarly-sized
model. Run both for ~30–45 min each, capture speed/NLL/memory, and report a
side-by-side comparison.

Repos:
- glades-ml at /home/robert/dev/glades-ml (branch: vesta5)
- glades-trainer at /home/robert/dev/glades-trainer (binaries in build/)

Step 1 — pick the comparison scale
- Find the most recent CHIRON run config from git/memory. Look at:
  - git log on glades-ml for commits matching "Chiron runs", "1.84B", "iter 1[5-8][0-9]"
  - /home/robert/.claude/projects/-home-robert-dev-glades-ml/memory/chiron_architecture.md
  - run.sh chiron --help (preset scales: 66M, 100M, 200M, 500M, 1.84B)
- Default target: --scale 1.84B (the documented ceiling on 16 GB).
- For glades_pile_train, choose dmodel/layers/dff that fit similar VRAM. The
  validated stacked config dmodel=768 layers=8 dff=2048 T=4096 already runs in
  ~3 GB. If you can fit a larger config in 16 GB with the stack, use it
  (try dmodel=1024 layers=12 dff=3072 first; fall back if OOM).

Step 2 — clean prior state
- Delete any /tmp/flagship_*.{out,err} and /tmp/chiron_*.{out,err}.
- Delete database/runs/flagship_long, database/checkpoints/flagship_long_*,
  database/models/flagship_long under /home/robert/dev/glades-trainer.

Step 3 — launch flagship (glades_pile_train, post-MLA-permanent-fix)
Run in background from /home/robert/dev/glades-trainer:

  nohup ./build/glades_pile_train \
    --pretok-dir ./pretok-uniform \
    --vocab-file ./pretok-uniform/vocab.bpe \
    --gpu --mp \
    --dmodel <D> --layers <L> --heads <H> --dff <FF> \
    --seq-len 4096 --tbptt 4096 \
    --max-tokens 10000000 \
    --attn-sinks 4 --local-attn 256 \
    --mla-dc 128 --binary-ffn \
    --lr 0.001 --weight-decay 0.0 \
    --model-name flagship_long \
    --no-auto-resume \
    > /tmp/flagship_long.out 2> /tmp/flagship_long.err &

Pick D/L/H/FF that fit. For ~100M class: 768/8/12/2048. For ~250M class:
1024/12/16/3072. Throughput target ~150K targets/sec at T=4096.

Step 4 — launch CHIRON (parallel, separate GPU stream is not possible on
RTX 4080 SUPER, so run sequentially after flagship completes OR use a smaller
CHIRON scale to fit alongside)

Sequential (recommended): wait for flagship to finish, then:

  cd /home/robert/dev/glades-trainer
  bash run.sh chiron --scale 1.84B --steps 5000 \
    --sas-schedule "0.3@0,0.5@2300,0.7@3700" \
    --save /tmp/chiron_long_ckpt \
    > /tmp/chiron_long.out 2> /tmp/chiron_long.err

(5000 steps at 1.84B is ~30 min by iter-166's 4.16-min/2500-steps benchmark.)

Step 5 — Use the Monitor tool to stream progress on each run. Filter on:
  "seq_done=([0-9]*[05])00[^0-9]|step=([0-9]*[05])000|epoch\] |chunk\] |Killed|FAILED|illegal|OOM|saved database/models|elapsed_steps="

Step 6 — Capture metrics for each run:
  - Wall-clock seconds (start to model save)
  - Targets/sec or tokens/sec sustained
  - Initial NLL → final NLL (Δ in nat)
  - Peak GPU memory (parse `nvidia-smi --query-gpu=memory.used --format=csv`
    every 30s during the run, or grep stderr for any memory log)
  - Total tokens trained
  - Effective parameters (sum of weight tensor sizes)

Step 7 — Build a comparison table:
  | Metric          | Flagship (pile_train + #74/#76/#78) | CHIRON 1.84B |
  | --------------- | ----------------------------------- | ------------ |
  | Params (M)      |                                     |              |
  | Wall-clock      |                                     |              |
  | Throughput      |                                     |              |
  | NLL initial     |                                     |              |
  | NLL final       |                                     |              |
  | NLL Δ / token   |                                     |              |
  | Peak GPU mem    |                                     |              |
  | Tokens trained  |                                     |              |
  | Status          |                                     |              |

Notes for fairness:
- Flagship and CHIRON are different architectures (CHIRON is reversible-flow
  with O(1) activation memory; flagship is standard transformer with MLA +
  attention-sink + binary FFN). A direct loss comparison is not equivalent —
  flagship trains the full model while CHIRON's compute model is different.
- Compare instead on:
  (a) Throughput per parameter (tokens·params/sec)
  (b) NLL drop per million tokens trained
  (c) Peak GPU memory at the same wall-clock budget
- The flagship has the JUST-FIXED MLA backward (commit 7dc1fe83c). Confirm
  via `git log -1 --format=%H 7dc1fe83c` is in HEAD's history before launch.

Step 8 — Write findings to research/FLAGSHIP_VS_CHIRON_LONG_RUN.md and
commit on branch vesta5 with message
  "research: flagship-vs-CHIRON 30-min comparison at <scale>"

Constraints:
- Don't block — use Bash run_in_background and Monitor.
- Don't burn cache: between runs, use one large sleep window (1500s+) rather
  than many short polls.
- If either run OOMs, fall back to a smaller scale and re-launch with a note.
- Auto-resume is OFF for both (--no-auto-resume / --fresh).
- Stop and ask only if both runs OOM at the smallest reasonable scale.
```

---

## What this prompt does (for reference)

| Phase | Action |
|---|---|
| 1 | Inspect git + memory to find the most recent CHIRON run scale |
| 2 | Pick matching dims for `glades_pile_train` |
| 3 | Launch flagship (post-MLA-fix stacked) for ~30–45 min |
| 4 | Launch CHIRON 1.84B sequentially |
| 5 | Monitor both via filtered streams |
| 6 | Capture wall-clock, throughput, NLL Δ, peak VRAM, tokens |
| 7 | Build side-by-side table (flagship vs CHIRON) |
| 8 | Commit findings doc |

## Why two binaries

| Binary | Architecture | Optimizations |
|---|---|---|
| `glades_pile_train` (flagship) | Standard transformer | `#74` PHOENIX-1BIT (binary FFN) + `#76` MLA (low-rank latent KV, just fixed) + `#78` ATTENTION-SINK + `#6` LOCAL-ATTN |
| `glades_chiron_train` (CHIRON) | Reversible symplectic-flow | FACE+MFIO+SLC+RLG+SAS, O(1) activation memory |

CHIRON's reversibility unlocks 1.84B on 16 GB. Flagship's stack at the same VRAM ceiling fits roughly 100–250M params.
