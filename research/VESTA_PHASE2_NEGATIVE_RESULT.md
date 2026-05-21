# VESTA Phase-2 — Honest Negative Result

**Date:** 2026-05-19
**Branch:** vesta5
**Status:** Program-level falsification complete. F-Phase2-2 confirmed via B5 baseline at 4 seeds.

---

## TL;DR

We pre-registered a 6-gate program to test whether GRP-RNN (Group-Rotation Product RNN, the Phase-1 candidate) could approach LLM-replacement quality on real benchmarks. Phase 1 demonstrated GRP-RNN solves the A_5 word-recognition task at sequence length T=8 with val_acc 1.00 while a diagonal-SSM control (LRU-equivalent) sits at random, providing the first clean empirical instance of the Merrill, Petty, Sabharwal (2024) bound that diagonal SSMs cannot perform non-solvable-group state tracking in O(1) depth.

Phase 2 transferred this expressivity primitive to a multi-layer LM stack on a real BPE-pretokenized corpus (V=32000, 219M tokens). After completing Gate-2A (per-token readout infrastructure, gradcheck-clean), we ran Gate-2B (multi-layer stacking) and discovered:

1. **Depth doesn't help GRP** on natural-language data (val_loss monotone *increases* with L at fixed width).
2. **A same-architecture control with the rotation mechanism disabled** (`--linear-recurrence=1`: `s_t = decay · s_{t−1} + W_in · x_t`, no Givens rotation) **beats GRP at every depth** and **does** show depth gain.

**Concrete numbers** (m=128, T=128, batch=32, 1500 steps, seed=42, real BPE corpus):

| Config              | val_loss @ 1500 |
|---------------------|-----------------|
| L=1 m=128 GRP       | 8.23            |
| L=1 m=128 LinearRNN | **8.00** (−0.23 nat) |
| L=2 m=128 GRP       | 8.30            |
| L=2 m=128 LinearRNN | **7.95** (−0.35 nat) |

The Phase-1 expressivity advantage does not transfer to natural language; in fact, the rotation mechanism is *actively harmful* (linear-RNN does strictly better with the same parameter budget and the same residual+MLP wrapping).

---

## Scientific Contribution (honest reading)

**The Merrill et al. (2024) diagonal-SSM expressivity bound is real and tight on synthetic state-tracking, but it is not the bottleneck for natural-language LM quality at the scales we tested.** Whatever the dominant capacity constraint is for natural-language LM, it is captured by what the simplest in-class baseline (scalar-decay linear RNN with residual+LN+MLP) can express, and is not relieved by adding non-diagonal recurrence.

This is a contribution to the literature on "formal-language expressivity bounds vs. practical LM quality" — a sub-field where prior work (Merrill, Sabharwal, Hahn, Strobl) characterized state-tracking limits of attention vs. SSMs vs. recurrent networks abstractly, but where the empirical question of whether those bounds matter for actual perplexity-on-text remained open.

Our result: the bound holds; the bound matters on its native task (A_5); but the bound does not predict natural-language performance.

---

## Pre-Registered Program (May 18 → May 19 revision)

### Phase 1 (May 18) — "find architecturally novel expressivity gap"

- 3 candidates: GRP-RNN (Givens-product), MuRe (multi-resolution scan), PULSAR (pulse-coupled oscillators).
- GRP-RNN selected over MuRe/PULSAR based on tractability + clearest mechanism-to-bound mapping.
- Pre-committed claims: N1 (A_5 expressivity gap), N2/N3 (mechanism-specific signatures), F-N1 falsification (the gap doesn't survive optimization beyond T=16).

### Phase 1 result (May 19)

- N1 PASSED at T∈{8, 16}: GRP-RNN val_acc 1.00 / 0.58 vs LRU-equivalent 0.10 / 0.04, gap +0.90 / +0.54 across 5 seeds.
- F-N1-a triggered at T≥32 (optimization failure, not expressivity).
- F2 mitigation (LayerNorm + multi-phase curriculum) extended the win to T=128. LN alone solves T=32 at acc 1.00 / 5 seeds; LRU+LN at T=32 stays at random — the gap is mechanism-level, not optimizer-induced.

This is the publishable Phase-1 result: a clean empirical demonstration of the Merrill bound on a real task.

### Phase 2 (May 19 revision) — "test LLM-replacement potential"

The May 19 brief retargeted the program from "find architecturally novel expressivity" (achieved) to "match or beat Mamba-2 at iso-FLOPs on real LM evals". Gates 2A through 2F sequential. Phase 1 reframed as a passed gate, not as the goal.

Mandatory baseline matrix:
- B1: GRP-RNN+LN (proposal)
- B2: LRU+LN (diagonal control)
- B3: Mamba-2 (when available)
- B4: causal Transformer (gold standard)
- B5: linear-RNN+orthogonal (Phase-1 floor)

Pre-registered failure modes for Phase 2:
- F-Phase2-1: depth doesn't compound. (Pre-Phase-2 probability: medium.)
- **F-Phase2-2**: matches but doesn't beat Mamba-2 on NL perplexity. (Pre-Phase-2 median expectation.)
- F-Phase2-3: scales but compute-inefficient.
- F-Phase2-4: stable but fails on long-range tasks.
- F-Phase2-5: works but engineering doesn't scale.

---

## Gate-2A — PASSED (2026-05-19 evening)

Implemented per-token LM readout in `model_grp_rnn.cuh`:
- New `cache.logits_all` (T, B, V), `cache.probs_all`, `cache.losses_all`, `cache.labels_lm`.
- Single batched `gemm_nn` for all logits, single `softmax_ce` over (T−1)·B rows.
- `cublasSaxpy` injection of `d_s_lm_all[t]` into the recurrence's `d_s` at each iter t < T−1.
- New kernel `k_build_lm_labels` for device-side label construction.

Two bugs fixed at this gate:
1. **LM head used task `n_classes` instead of `V_vocab`.** For tasks where V > n_classes (e.g., needle: V=88, n_classes=8), labels_lm values ≥ n_classes caused out-of-bounds reads in softmax_ce, producing nondeterministic gradcheck failures. Fix: pass `head_classes = lm_mode ? tw.V : tw.n_classes`.
2. **Pre-existing `--grp-fixed-angles` backward** missed the `phi_max · (1 − tanh²(b_a))` derivative on `b_a`, off by factor ~1.57 at init.

Gradcheck coverage: 126/128 configs PASS (4 tasks × LM={0,1} × LN={0,1} × {no-ablation, --grp-tanh-state, --grp-fixed-angles, --grp-disjoint-planes} × seed={42,43}). The 2 failures are numerical noise on small-magnitude gradient entries (rel ≤ 0.08 with absolute differences ~1e−3, at the FP32 cancellation floor).

Gate-2A is the infrastructure prerequisite for everything downstream and is robust.

---

## Gate-2B — FAILED + F-Phase2-2 CONFIRMED

### Built

`research/ealrmn_gpu/model_grp_stack.cuh` (~650 LOC). Pre-norm multi-layer stack:

```
x_0 = E[ids]
for ℓ = 0..L−1:
    h     = LN1_ℓ(x_ℓ)
    h     = GRP_recurrence_ℓ(h)        # per-step over t, K Givens, K=m
    x_ℓ'  = x_ℓ + h                     # residual #1
    h     = LN2_ℓ(x_ℓ')
    h     = MLP_ℓ(h)                    # m → 4m → m, ReLU between
    x_{ℓ+1} = x_ℓ' + h                  # residual #2
x_final = LN_final(x_L)
logits[t] = x_final[t] @ W_out + b_out
loss = (1 / ((T−1)·B)) · sum softmax_ce(logits[t], ids[t+1])
```

The `--linear-recurrence=1` flag disables the rotation: theta = 0 always, so `R_t = I` and the recurrence reduces to `s_t = decay · s_{t−1} + W_in · x_t`. Everything else (LN1, LN2, MLP, residuals, embedding, readout) is identical.

`research/ealrmn_gpu/tasks.cuh` added `CorpusTask`: reads BPE `.tok.bin` files (24-byte header + uint16 tokens), random-window sampling. Validated on `/home/robert/dev/glades-trainer/pretok-data/val.tok.bin` (219M tokens, V=32000 BPE).

Gradcheck: 91/96 (L=1) and 157/162 (L=2). Remaining ~5% fails verified to be FP32 precision-floor noise on small-magnitude entries (hand-summed `dy · xh` over rows matches analytic exactly; numeric noise is `O(√(ε_fp · L_loss) / 2·eps)`).

### Result 1: depth doesn't help GRP

At m=128 fixed:

| L | val_loss @ 1500 | Total params |
|:-:|:---------------:|:------------:|
| 1 | **8.26**        | 8.39M        |
| 2 | 8.34            | 8.55M        |
| 4 | 8.41            | 8.88M        |

Strictly monotone increase with L. LR-scaling-by-1/√L (standard transformer recipe) made things worse, not better. Zero-init W_mlp2 (standard "each-block-starts-as-identity" mitigation) made every config slightly worse. None of the standard depth-unlocking tricks worked.

### Result 2: the B5 baseline (rotation disabled) wins decisively

Same architecture, `--linear-recurrence=1`:

| Config              | val_loss @ 1500 | Δ vs GRP        |
|---------------------|-----------------|-----------------|
| L=1 m=128 GRP       | 8.23            | —               |
| L=1 m=128 LinearRNN | **8.00**        | **−0.23 nat**   |
| L=2 m=128 GRP       | 8.30            | —               |
| L=2 m=128 LinearRNN | **7.95**        | **−0.35 nat**   |

And critically: linear-RNN at L=2 (7.95) is better than linear-RNN at L=1 (8.00) — depth helps without the rotation. With the rotation, depth hurts.

The result has two implications:

**(a) The Phase-1 mechanism does not transfer to NL.** The non-diagonal Givens rotation that lets GRP-RNN solve A_5 expresses a capability that natural-language tokens don't need. Replacing the rotation with the identity (which is what `--linear-recurrence` does) preserves the residual + LN + MLP stack's full natural-language capacity.

**(b) The rotation is actively harmful, not just useless.** Two hypotheses (we don't disambiguate here):
- *Gradient instability*: the reverse-time Givens recursion compounds rotation noise through depth.
- *Wasted capacity*: K=m extra input-dependent angle parameters per layer act as noise that distracts optimization from the actually-useful W_in / MLP path.

Either way, the practical takeaway: don't use Givens-product recurrence for natural-language LM.

### What this means for Gate-2C

The brief's Gate-2C asks "match Mamba-2 at iso-params". Given the B5 control already beats GRP by 0.23–0.35 nat, Mamba-2 (a tuned modern SSM with selective scan, structured eigenvalues, and substantial engineering) will beat GRP by much more. Building a Mamba-2 baseline at this point would confirm but not change the verdict.

The honest read of the program is that **the falsification is complete at Gate-2B**. Mid-likelihood failure mode F-Phase2-2 is confirmed in its stronger form: GRP doesn't merely fail to beat a tuned baseline, it loses to the simplest in-class control.

---

## Reproduction

```bash
cd research/ealrmn_gpu && ./build.sh

# Single seed comparison (GRP vs LinearRNN, L=2 m=128, 1500 steps):
./ealrmn_gpu --mode=train --model=grp_stack --task=corpus \
    --corpus=/path/to/val.tok.bin \
    --n-layers=2 --m=128 --T=128 --batch=32 --eval-batch=64 \
    --steps=1500 --eval-every=500 --lr=3e-3 --warmup=100 --seed=42

# Same with rotation disabled (B5 baseline):
./ealrmn_gpu --mode=train --model=grp_stack --task=corpus \
    --corpus=/path/to/val.tok.bin \
    --n-layers=2 --m=128 --T=128 --batch=32 --eval-batch=64 \
    --steps=1500 --eval-every=500 --lr=3e-3 --warmup=100 --seed=42 \
    --linear-recurrence=1
```

Same corpus (`/home/robert/dev/glades-trainer/pretok-data/val.tok.bin`, 219M tokens, V=32000 BPE), same hyperparameters, same architecture — only the recurrence mechanism differs.

### Multi-seed validation (brief P5: ≥3 seeds required)

| Seed | GRP L=2 | LinearRNN L=2 | Δ (GRP − Linear) |
|:----:|:-------:|:-------------:|:----------------:|
| 42   | 8.30    | 7.95          | +0.35            |
| 43   | 8.34    | 7.93          | +0.42            |
| 44   | 8.14    | 7.63          | +0.51            |
| 45   | 8.27    | 7.90          | +0.37            |
| **Mean** | **8.26** | **7.85**  | **+0.41 nat**   |
| stdev    | 0.09     | 0.14        | 0.07             |

Linear-RNN beats GRP across **every seed**, by 0.35–0.51 nat. The 4-seed mean gap is 0.41 nat with std 0.07. The result is robust to seed variation.

---

## Open questions for follow-up

1. **What IS the bottleneck for natural-language LM at small scale?** The B5 result suggests the dominant capacity is in `W_in @ x_t` (input projection per step) and the MLP, not in the recurrence's eigenstructure.

2. **Does the GRP harm grow or shrink with scale?** All experiments here are at m=128 (8.4M total params); at 50M+ scale the rotation's distraction effect might wash out, or it might compound.

3. **Are there natural-language sub-tasks where GRP would help?** The Phase-1 win is on non-solvable-group state tracking. Are there genuine NL sub-tasks (deep nested syntactic dependencies, parity tracking in code, etc.) where this expressivity is the actual bottleneck? A targeted eval on such sub-tasks would be informative.

4. **What does Mamba-2's selective scan add over the B5 linear-RNN floor?** Even without GRP in the picture, the gap between B5 and Mamba-2 measures the value of selective gating, structured eigenvalues, and the SSM training recipe. This is the natural Phase-3 question if the program continues.

---

## Files in this commit

- `newmodel.txt` — Phase-2 brief (May 19 revision)
- `research/VESTA_REPORT.md` — Phase-1 report (treat as "passed gate" going forward)
- `research/VESTA_GATE2B_RESULT.md` — Gate-2B full experimental log
- `research/VESTA_PHASE2_NEGATIVE_RESULT.md` — this synthesis document
- `research/ealrmn_gpu/model_grp_stack.cuh` — multi-layer stack model (~650 LOC)
- `research/ealrmn_gpu/model_grp_rnn.cuh` — single-block GRP-RNN with LM-mode
- `research/ealrmn_gpu/tasks.cuh` — CorpusTask BPE loader
- `research/ealrmn_gpu/main.cu` — train_grp_stack, gradcheck integration, CLI

---

## Honest closing statement

The Phase-2 program was designed by a brief revision (May 19) that retargeted from "find architecturally novel expressivity" (Phase-1 achievement) to "build an LLM replacement" (Phase-2 attempt). The retargeting was deliberate, and we committed to publishing a clear pass/fail per the pre-committed gate criteria.

Gate-2B clears the bar for honest reporting: the B5 baseline (a control we pre-registered as the "Phase-1 floor") beats the proposal on natural language. This falsifies the program's central architectural hypothesis at the smallest scale tested. There is no honest way to claim partial success.

The Phase-1 mechanism remains valid for what it was designed for: it demonstrates the Merrill 2024 diagonal-SSM bound is empirically tight on A_5 state-tracking, and provides a clean controlled experiment within the formal-language expressivity literature. Phase-2's negative result is informative in its own right — it tells us that the expressivity bound is not the natural-language LM bottleneck at scales tested, narrowing the search for what is.
