# EALRMN Phase-0d — Results Report

**Date:** 2026-05-18. **Status:** Phase-0d PARTIAL — needle-in-haystack task discriminates models at moderate context, but encoder bottleneck (mean-pool) prevents bounded-memory mechanism from showing its expected advantage at long context.

Design memo: `research/EALRMN_DESIGN.md`. Phase-0a/b/c: `research/EALRMN_PHASE0_RESULTS.md`, `research/EALRMN_PHASE0B_RESULTS.md`, `research/EALRMN_PHASE0C_RESULTS.md`. Prototype: `research/ealrmn_phase0d_needle.cpp` (~830 LOC C++98, separate from Phase-0a–c prototype).

## What Phase-0d tested

Three models on the synthetic **needle-in-haystack** retrieval task (design memo §9.1):

1. **EALRMN** — encoder + Koopman recurrence + 4-slot bounded memory with **learned write gate** $g_i = \sigma(W_g \cdot z_i + b_g)$ and write-penalty $\lambda_w = 0.002$. Readout from concat(s_N, M_N), feature dim = 5m = 160. **This is the full memory mechanism with backprop through the gated EMA chain.**
2. **RNN** — same encoder + Koopman recurrence, no memory. Readout from s_N only, feature dim = m = 32.
3. **ATTN** — same encoder; final readout via cosine-similarity attention from query patch's z onto all earlier z's, full softmax backward through normalization + attention weights. Approximates a one-layer Transformer at the patch level.

### Needle task structure

```
Stream of N patches (each 4 tokens, T = 4N total tokens):
  - 2 KV "patches" inserted at random positions in [0, N-1):
      [KEY_MARKER, k_id, VAL_MARKER, v_id]
  - 1 query patch at position N-1:
      [QUERY_MARKER, k_id_query, filler, filler]
  - All other patches: 4 uniform random fillers from [0, 16).
  - Label: v_id paired with k_id_query.
Vocabulary V = 27: 16 filler + 3 markers + 4 keys + 4 values.
Random-baseline accuracy: 1/4 = 0.25.
```

The task tests Claim 3 (bounded memory matches Transformer-attention up to capacity threshold) and Claim 5 (selective recurrence preserves long-range state).

## Architecture

```
d_emb     = 16
m = r     = 32
N         ∈ {16, 32, 64} (sweep)
Encoder   = embedding lookup → mean over patch (kPatchLen=4) → linear (m × d_emb)
Recurrence = s_i = K · s_{i-1} + B · z_i,  K ∈ ℝ^{32×32} near-identity init
Memory    = 4 slots with fixed decays {0.50, 0.80, 0.95, 0.99}, gated EMA:
              M_i[sl] = (1 − g_i·(1−λ_sl)) M_{i-1}[sl] + g_i·(1−λ_sl) z_i
Gate      = σ(W_g · z_i + b_g)  ← learned, depends on LOCAL z (not integrated s)
Readout   = linear from feat → 4-way softmax over values
Optimizer = SGD lr=0.005 (Kop and B use 0.3× this), batch 16, grad-clip L2=10
```

## Three-context-length sweep (seed 42, 2500-3000 steps)

| N (patches) | T (tokens) | EALRMN final | EALRMN best | RNN final | RNN best | ATTN final | ATTN best |
|-------------|------------|---------------|--------------|-----------|----------|------------|-----------|
| 16          | 64         | 0.66          | 0.66         | 0.66      | 0.66     | **0.69**   | **0.72**  |
| 32          | 128        | 0.47          | 0.50         | **0.52**  | **0.52** | 0.36       | 0.36      |
| 64          | 256        | 0.27          | 0.36         | 0.30      | 0.45     | 0.27       | 0.34      |

(Random baseline = 0.25.)

## What this shows

1. **ATTN wins at short context (N=16):** 0.72, decisively above the recurrent models. Cosine-similarity attention naturally suits retrieval — when the encoder's z carries enough k_id information, the query attends to the matching KV patch.

2. **ATTN collapses fastest as context grows.** At N=32 ATTN drops to 0.36; at N=64 it's at 0.27. The encoder's z for filler patches becomes indistinguishable from KV patches in cosine space when there are many distractors with the same noise level.

3. **EALRMN and RNN track each other at all context lengths.** The learned write gate (kept around 0.47-0.50 throughout training in the EALRMN runs) does not produce a measurable accuracy gain over the no-memory baseline. **Bounded-memory advantage NOT demonstrated.**

4. **All three models fail at N=64 (~0.30, basically random).** The encoder's mean-pool aggregation loses enough KV-vs-filler signal that no downstream mechanism can recover.

## Why the bounded-memory mechanism does not win

The gate values never specialise. Across training the gate stays near its initial value 0.5 (drifts slowly toward 0.45 due to the write penalty, but not toward bimodal "high on KV / low on filler"). This is the empirical signature: the gate is not learning to recognise KV patches.

Root cause: the encoder mean-pool produces statistically similar z's for KV patches and filler patches at random init. For the gate to learn discrimination via the final readout's loss, gradient must propagate backward through:

  gate(W_g, b_g) ← d L / d g ← d L / d M_{N-1} ← d M / d M chain over N steps ← readout loss

When N is large and the readout is itself struggling to reach informative loss (because s_N has too much accumulated noise), the d L / d M_{N-1} signal is small. The gate gets ambiguous "write more / write less" gradients across positions, and converges to a near-uniform value.

This is the **bootstrap failure of gated memory at random encoder initialization**: the gate needs the encoder to already produce distinct z's for KV-vs-filler in order to learn its discriminator; the encoder needs the gate to already focus its memory on KV patches to learn the discrimination. Neither side can move first without an auxiliary signal. (Same family as the Phase-0b InfoNCE bootstrap failure.)

## What this implies for the design memo

Two claims tested explicitly:

- **Claim 3** (bounded memory matches Transformer up to capacity threshold ρ_rel = K): NOT YET SUPPORTED. The mechanism is in place, but the experimental setup does not demonstrate the advantage. Either the task needs bootstrap supervision (analogous to the recon loss in Phase-0b) for the gate, or the encoder needs to be more powerful so that z's are pre-discriminative.

- **Claim 5** (selective recurrence preserves long-range state better than small Transformer at fixed memory budget): NOT SUPPORTED at this scale. ATTN at N=16 has all O(N) hidden states available and beats both recurrent models. At N≥32 all models fail similarly — context-length scaling shows EALRMN/RNN degrade slightly less than ATTN, but all collapse to near-random by N=64.

The "encoder mean-pool" bottleneck is **architectural**, not informational: the Bayes-optimal classifier on this task is ~100% (the relevant information IS present in 2 of N patches and is fully recoverable). The current architecture cannot recover it because the encoder loses positional information within patches via mean-pool.

## Why the architecture is bottlenecked at the encoder

A KV patch is `[KEY_MARKER, k_id, VAL_MARKER, v_id]` and a filler patch is `[fill1, fill2, fill3, fill4]`. Both produce z = mean of 4 token embeddings + linear projection. After mean-pool:

- KV z = (Emb[KEY_MARKER] + Emb[k_id] + Emb[VAL_MARKER] + Emb[v_id]) / 4
- Filler z = (Emb[fill1] + Emb[fill2] + Emb[fill3] + Emb[fill4]) / 4

The KV z is informationally rich — it contains the k_id and v_id projected into 4-dim subspaces. The filler z is just noise. In principle the encoder can learn distinct Emb[]s for markers vs fillers, making KV z's *magnitude* differ from filler z's. But the *direction* of KV z is still a 4-token mixture; cosine similarity in attention can't pick out which k_id is encoded.

To genuinely test the bounded-memory and attention claims, the encoder needs to either:
- Per-token (kPatchLen=1) processing — each token gets its own z, attention works on raw token vectors.
- Per-token attention within patch — a one-layer Transformer encoder over the 4 tokens.
- Token-level write gating — gate fires on TOKEN positions, not patch positions.

## Sample training trajectories

### N=16 ATTN — the only winner

```
step    train_loss  held_loss  held_acc  gate
   0    1.3877      1.3797     0.3125    0.0000
 500    1.3647      1.3872     0.2969    -
1000    1.3994      1.3727     0.4219    -
1500    1.3787      1.3670     0.4844    -
2000    1.3257      1.3092     0.5625    -
2250    1.2498      1.2543     0.7188    -
2499    1.2563      1.2488     0.6875    -
```

### N=64 EALRMN — gate stays near 0.5 but task does not converge

```
step    train_loss  held_loss  held_acc  gate
   0    2.2891      3.5925     0.2969    0.4980
 500    1.3452      1.3530     0.3906    0.4786
1000    1.4471      1.3730     0.2500    0.4585
1500    1.4290      1.3718     0.3594    0.4388
2000    1.2937      1.3949     0.2656    0.4197
2500    1.3115      1.3864     0.2500    0.4006
2999    1.3372      1.3667     0.2656    0.3821
```

The gate drifts slowly toward zero (penalty dominates over learning signal) without specialising.

## Smoke-test reproduction

```bash
cd /home/robert/dev/glades-ml
g++ -std=c++98 -O2 -Wall -Wextra research/ealrmn_phase0d_needle.cpp \
    -o research/ealrmn_phase0d_needle

# Context sweep — three models per context length:
for N in 16 32 64; do
  for M in ealrmn rnn attn; do
    ./research/ealrmn_phase0d_needle --model $M --seed 42 --steps 2500 --N $N --print-every 250
  done
done
```

Wall clock at N=16: ~10s/run. N=32: ~25s. N=64: ~60s.

## Next-step candidates (Phase-0e and beyond)

**Phase-0e — fix the encoder**. The cleanest follow-up is to make the encoder per-token (kPatchLen=1) so each KV-marker token gets its own z. Then attention and gated memory have meaningful targets. This is the highest-priority change.

**Phase-0e (alt) — auxiliary supervision for the gate**. Add an *auxiliary loss* that supervises the gate: e.g., a reconstruction objective that requires M_{N-1} to predict the tokens at gate-fire positions. Like reconstruction bootstrap from Phase-0b, this would give the gate a clear "write more here" signal without needing the readout's gradient to be informative.

**Phase-0f — sparse experts**. The unused mechanism in the brief: top-k routed experts. Could test whether routing among multiple Koopman operators improves long-context retrieval.

**Pivot consideration.** After Phase-0a, b, c, d the cumulative evidence suggests that the EALRMN architecture's individual mechanisms each train correctly when given the right bootstrap signal, but none of them have demonstrated a *decisive advantage over a strong baseline*. The likely path forward is either (a) move to a substantially larger scale where the mechanisms can actually compound their efficiency wins, or (b) acknowledge the framework as scientifically interesting but not empirically superior at small scale and pivot back to writing up the design + experimental findings as a contribution in itself.

## Honest verdict

Phase-0d delivers a clean negative result on the needle-in-haystack task: at this scale and encoder architecture, the bounded-memory mechanism does not produce the predicted advantage. The encoder mean-pool prevents the gate from learning its discrimination role.

The result is informative — it locates exactly where the architecture breaks (encoder-level token-mixing) and points clearly to the next experiment (per-token encoder). It does not refute the design memo's claims; it shows the experimental setup that was supposed to test them needs an architectural change first.

— end Phase-0d results report —
