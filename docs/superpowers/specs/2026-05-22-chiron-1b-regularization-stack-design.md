# CHIRON 1B Regularization Stack — Design

**Date:** 2026-05-22
**Status:** Pre-registered design (no code yet)
**Program scope:** Phase-3 of CHIRON 1B production line. Adds the first
loss-side and attention-side generalization mechanisms to a stack that
has so far been dominated by optimizer-side and wall-time work.
**Baseline:** `chiron_1B_T16384_v5_fp8_phase2.final` (v5+FP8 ship
2026-05-22). Final val NLL 4.1717 @ step 30000, throughput 28,887 tok/s,
VRAM 14.97 / 15.56 GB on RTX-4080-SUPER.
**Output target:** A 30k Phase-2 production retrain at v5+FP8 + the new
regularization stack, gated on a 5k Phase-1 pilot arc.

---

## 1. Motivation

The current CHIRON 1B flagship has had heavy investment in the
**optimizer side** (FACE Adafactor for embeddings; ATLAS, ECHO, BiMAP,
MUON-lite, ARGOS, MATRA, RAMPART, MERIT and dozens more in
`training_config.h`) and the **wall-time / memory side** (iter 116 ship
+ v5 BF16 cast fix + FP8 readout, 1.90× cumulative throughput since
pre-ralph-loop). The **loss-side and architecture-side regularization
mechanisms** that are standard on modern LLMs are nearly absent:

- `embeddingDropoutRate` and `residualDropoutRate` are wired into the
  trainer but defaulted to 0.0f and not in the ship recipe.
- No Z-loss on the readout partition function.
- No QK-Norm on attention (standard `1/√d_h` scaling).
- No multi-token prediction (MTP) auxiliary objective.

At the v5+FP8 ship, val top-1 = 11.6% / top-5 = 45.5% (from the iter 116
ship measurement at step 32500 published in
`PHASE3_GATE3A_PREREG.md`§2). These are reasonable for an undertrained
1B token-LM, but indicate meaningful generalization headroom that
loss-side and attention-side regularizers can attack.

The three mechanisms in this spec are picked because:

1. Each is well-validated at LLM scale (PaLM/T5/Gemini for Z-loss;
   DeepSeek-V3 / modern Llama for QK-Norm; DeepSeek-V3 for MTP).
2. Each is implementable in ~50-300 LOC inside the existing codebase
   (no external libraries — complies with Phase-3 program rule).
3. Each is independent of the others at the gradient-flow level —
   stacking is well-defined and pre-committed gate-able.
4. Each composes cleanly with the existing v5+FP8 stack — none
   conflicts with FACE, SCFA, BF16 master-weights, FP8 readout, or any
   iter 116 wall-time mechanism.

---

## 2. Architecture

### 2.1 Z-loss (mechanism A)

Auxiliary loss on the softmax partition function:

```
L_total = L_CE + λ_z · mean_t( log²(Z_t) )
```

where `Z_t = sum_v exp(logit_{t,v})` is the partition function at
token position t, and `log(Z_t) = logsumexp(logits_t)` is computed
identically to the main CE pass.

**Why this fits CHIRON 1B specifically:**
- Bounds logit magnitudes, directly stabilizing the
  `--fp8-readout-fwd` path (FP8 E4M3 dynamic range ±448 — large
  logits saturate or quantize destructively). The iter 62 design doc
  for FP8 already discusses logit-magnitude sensitivity.
- Typical reported gain: −0.05 to −0.15 nat on val NLL (PaLM §5.1;
  T5 paper §3.1.1). At λ=0 the math is bit-identical to baseline,
  giving a clean strict-bar safety test.
- Backward shares the softmax with the main CE backward — no extra
  exponentials, no extra reductions. Wall-time cost: <0.1%.

**Forward (added to readout block):**
```c++
// After computing log_softmax_lse[T] = logsumexp(logits, dim=-1)
// during the standard CE forward:
float zloss_sum = 0.0f;
for (int t = 0; t < T; ++t) {
    float lse = log_softmax_lse[t];
    zloss_sum += lse * lse;
}
loss_zloss = (lambda_z * zloss_sum) / T;
loss_total = loss_main + loss_zloss;
```

**Backward (added to readout grad):**
```
∂L_zloss/∂logit_{t,v} = (2 · λ_z / T) · log(Z_t) · softmax(logits_t)[v]
```

This is added pointwise to the existing CE gradient buffer before the
FP8 readout backward. No new kernels — the term is `2·λ_z·log_Z/T`
scalar times the existing softmax vector that the CE backward already
computes.

**Implementation sites:**
- `Backend/Machine Learning/Networks/sgd_transformer.cpp` — readout
  CE forward/backward, scalar `lambda_z` parameter wiring.
- `Backend/Machine Learning/Networks/training_config.h` — add
  `float zlossCoef` to `TransformerRunConfig` (default 0.0f).
- `Backend/Machine Learning/Networks/cuda/gpu_kernels.cu` — fuse the
  z-loss gradient contribution into the existing softmax-CE backward
  kernel (or add a tiny separate kernel — TBD by implementation, the
  fused option is preferred for fewer launches).
- `glades-trainer` `chiron_main`: add `--zloss-coef` flag, default
  0.0, recommended 1e-4 (PaLM value).
- FP8 path interaction: Z-loss is computed in **FP32 from the FP32
  logits before the FP8 readout backward casts**. The FP8 path
  affects only the weight-gradient GEMM, not the loss term.

**Strict bit-identicality at λ=0:** the readout codepath when
`zlossCoef == 0.0f` must produce **bit-identical** loss and gradients
vs the current ship. Guarded by an early return.

---

### 2.2 QK-Norm (mechanism B)

Per-head L2-normalize Q and K before the attention dot product;
replace the constant `1/√d_h` with a learnable per-head scalar `γ_h`.

**Standard attention:**
```
A = softmax( (Q K^T) / √d_h )
```

**QK-Norm:**
```
Q'_{t,h} = Q_{t,h} / ||Q_{t,h}||_2 + ε
K'_{t,h} = K_{t,h} / ||K_{t,h}||_2 + ε
A = softmax( γ_h · (Q' K'^T) )
```

with `γ_h` initialized to `log₂(T) = log₂(16384) = 14.0` per the
DeepSeek-V3 init.

**Why this fits CHIRON 1B specifically:**
- T=16384 is exactly the regime where standard `1/√d_h` scaling
  begins to fail — the variance of `Q K^T` grows with context length
  and `1/√d_h` no longer normalizes it correctly. QK-Norm fixes this
  by making the scale invariant to Q/K magnitude, then giving the
  model a learnable scalar to recover any beneficial sharpness.
- Modern long-context LLMs (DeepSeek-V3, Chameleon, Gemma-2) ship
  QK-Norm by default.
- Improves both training stability and generalization. Reported gain
  on val NLL at long context: −0.03 to −0.08 nat.
- The learnable `γ_h` (per-head, 16 scalars per layer × 24 layers =
  384 params total) is negligible parameter overhead.

**Interaction with SCFA:** CHIRON 1B uses SCFA (Subspace-Compressed
FlashAttention) for inner attention plus a wider outer attention. Both
paths need QK-Norm:

- **Inner attention** (compressed, depthwise-conv based per
  `sgd_transformer.cpp` and the chiron_scfa_* kernels): apply
  QK-Norm to the compressed Q/K representations. The compression is
  rank-preserving for the column directions we care about, so
  per-head L2 norm on the compressed vector is well-defined.
- **Outer attention** (full cuBLAS GEMM path): apply QK-Norm to the
  full-rank Q/K before the standard FlashAttention-style block.

**Forward cost:** 2 elementwise norms per Q/K block per layer. For
T=16384, m=2048, nH=16, dH=256, this is 2 × 16 × 16384 × 256 ≈ 134 M
ops per layer ≈ 3.2 G ops total — about 0.1% of attention compute.

**Backward cost:** Norm derivative is
```
∂Q'_h / ∂Q_h = (1/||Q_h||) · (I - Q'_h ⊗ Q'_h)
```
which is one extra outer-product-like operation per Q/K per layer.
Similar 0.1-0.2% overhead.

**Implementation sites:**
- `Backend/Machine Learning/Networks/sgd_transformer.cpp` — wrapper
  around the attention call: normalize Q and K, then call attention
  with `γ_h` substituted for `1/√d_h`. Backward unrolls the norm grad.
- `Backend/Machine Learning/Networks/cuda/gpu_kernels.cu` — new
  `qknorm_forward_kernel` and `qknorm_backward_kernel` (small,
  per-head per-token elementwise + reduction). Possibly fused into
  the existing Q/K projection epilogue.
- The v5 BF16 cast wrappers (`sgemm_rowmajor_fast16bf_impl`) are NOT
  affected — QK-Norm operates on the post-projection Q/K tensors,
  not on the GEMM internals.
- `training_config.h` — add `bool qkNormEnabled` (default false) and
  `float qkNormGammaInit` (default 0.0f → derive log₂(T) at init).
- `chiron_main` — `--qk-norm` flag (bool) and `--qk-norm-gamma-init`
  (float, optional override).

**Determinism:** the per-head L2 norm reduction must use the existing
deterministic reduction policy (per
`Backend/Machine Learning/DETERMINISM_AND_CONCURRENCY.md`). On GPU,
this is a within-warp reduction over d_h=256 elements per head per
token, with deterministic ordering.

**Strict bit-identicality test:** at `qkNormEnabled = false`, the
attention path must be bit-identical to the current ship.

---

### 2.3 Multi-Token Prediction (mechanism C)

Auxiliary loss predicting the token at position `t + 2` from each
position `t` (in addition to the standard t+1 prediction).

**Simple variant (this spec, ~50 LOC):**
- Single MTP head: `h_t → W_mtp · h_t → tied_readout`
- Loss: `L_total = L_CE + λ_mtp · L_MTP`
- `λ_mtp = 0.1` (DeepSeek-V3 default; they ramp from 0.3 → 0.1)
- Target at position t for MTP head: token at t+2 (shifted by 1
  beyond the main head)
- Last token in each sequence has no MTP target → masked out.

**DeferreD: DeepSeek-style "deep" MTP** (N=4 with extra transformer
layers per MTP depth) is intentionally NOT in this spec. Each extra
MTP transformer block at L=24 costs ~4% wall — stacking 4 of them
adds ~17% wall and meaningfully changes the training budget. The
simple version captures most of the reported gain (DeepSeek-V3 §4.2
ablation shows ~60% of MTP improvement comes from depth-1) at near-
zero wall cost. If the simple variant PASSes, a follow-up spec can
test deep MTP.

**Why this fits CHIRON 1B specifically:**
- 1B at 30k × 16k tokens ≈ 500B tokens is severely undertrained
  (Chinchilla optimal ~20B params × 20 tokens/param = 20B tokens, so
  the model is at 25× Chinchilla data — undertrained relative to a
  20B-token model, but data-rich relative to the param count). MTP
  primarily helps data efficiency, which is the lever we have.
- DeepSeek-V3 reports −0.10 nat val NLL from MTP at scale; simple
  MTP captures ~60% per their ablation.
- Drop-at-inference: zero inference cost. No KV-cache impact.

**Forward (added to readout block, after main CE):**
```c++
// After main CE forward at position t producing h_t (residual stream
// output before main readout):
for (int t = 0; t < T - 1; ++t) {  // last token has no t+2 target
    float h_mtp[m];
    mat_vec_mul(W_mtp, h_t, h_mtp);  // (m, m) projection
    float logits_mtp[V];
    mat_vec_mul(tokE_tied, h_mtp, logits_mtp);  // shared readout
    int target_mtp = tokens[t + 2];
    float loss_mtp_t = softmax_ce(logits_mtp, target_mtp);
    loss_mtp_sum += loss_mtp_t;
}
loss_mtp = lambda_mtp * loss_mtp_sum / (T - 1);
loss_total = loss_main + loss_mtp;
```

**Parameter overhead:** `W_mtp` is one `(m, m)` matrix = 2048 × 2048
× 2 bytes (BF16) = 8.4 MB master weights + Adam state. Negligible vs
the 870.94M total params.

**Backward:** standard softmax CE backward through the tied readout
and the `W_mtp` projection. Adds one extra readout GEMM per training
step (T × V × m at FP8 if `--fp8-readout-fwd` is enabled, which it is
for the ship recipe). Wall cost: ~3 GEMMs are added (one fwd, two
bwd) for MTP head; at T=16384, V=32000, m=2048, that's ~3 TFLOPs added
on top of the existing 3 readout GEMMs = 100% relative readout cost
increase, or ~2% wall (readout is ~2% of step time per
`ITER105_POST_STACK_NSYS_META.md`).

**Implementation sites:**
- `Backend/Machine Learning/Networks/sgd_transformer.cpp` — MTP
  forward, backward, target index computation.
- `Backend/Machine Learning/Networks/cuda/gpu_blas.cu` — reuse the
  existing FP8 readout GEMM path for the MTP head.
- `training_config.h` — add `int mtpDepth` (default 0 = disabled,
  recommended 1) and `float mtpCoef` (default 0.1f).
- `chiron_main` — `--mtp-depth N` and `--mtp-coef λ` flags.

**Tied embedding interaction:** the MTP head shares `tokE` with the
main readout (per `tieEmbeddings=true` in the ship config). This means
MTP gradients flow back into `tokE` via the FACE Adafactor optimizer,
just like the main readout gradients. No additional FACE accounting
needed — gradients accumulate before FACE consumes them.

**Strict bit-identicality test:** at `mtpDepth == 0`, the codepath
must be bit-identical to the current ship. Guarded by an early return
of the MTP block.

---

## 3. Validation Arc

### 3.1 Run plan (5 pilots + 1 production retrain)

| ID | Steps | Seed | Config | Compute |
|---|---:|---:|---|---:|
| B0 | 5000 | 1337 | v5+FP8 ship (current) | ~47 min |
| B1 | 5000 | 1337 | B0 + Z-loss (λ_z=1e-4) | ~47 min |
| B2 | 5000 | 1337 | B0 + QK-Norm (γ_init=log₂(T)) | ~47 min |
| B3 | 5000 | 1337 | B0 + MTP (depth=1, λ_mtp=0.1) | ~48 min |
| B4 | 5000 | 1337 | B0 + Z-loss + QK-Norm + MTP | ~48 min |
| **B5** | **30000** | **1337** | **Winner of B4 (or best subset)** | **~4.7 h** |

Total compute: ~4 hours for the 5k pilot arc + ~4.7 hours for the
30k Phase-2 = ~9 hours wall. Single-seed (seed=1337) per the iter-93
/ iter-94 Phase-1 / Phase-2 precedent. Multi-seed (n=3) deferred to a
separate gate if B5 PASSes.

### 3.2 Gate criteria (pre-committed, no silent re-targeting)

**B0 sanity:** the re-run baseline B0 must reproduce the published
v5+FP8 5k trajectory to within ±0.05 nat at step 5000 (same-day
variance calibration). If not, F-RegStack-1 fires — investigate
non-determinism before continuing.

**Per-mechanism 5k gates (B1, B2, B3):**
- Each individual mechanism must NOT regress **main-head val NLL** (CE
  only, auxiliary losses excluded) by more than 0.02 nat vs B0 at step
  5000. Improvement is preferred but not required at 5k single-seed
  (noise floor is ~±0.05 nat).
- If any individual mechanism shows >0.05 nat regression at 5k,
  mark it FAIL and exclude from the stack.

**Stacked 5k gate (B4):**
- B4 val NLL improvement vs B0 at step 5000 must be ≥ max of
  individual mechanism improvements (additive or super-additive).
- If B4 is worse than the best individual, flag which pair
  interacts negatively before B5. Possible fallback: B5 uses the
  best-performing subset of the three mechanisms rather than all
  three.

**30k Phase-2 gate (B5):**
- B5 val NLL at step 30000 must be ≤ 4.1517 (= 4.1717 − 0.02 nat
  improvement vs v5+FP8 ship).
- B5 throughput must be ≥ 27,500 tok/s (≤ 5% wall regression vs
  v5+FP8 ship at 28,887). Z-loss and QK-Norm should be near-zero
  wall; MTP adds ~2%. Combined budget: 5%.
- B5 VRAM must be ≤ 15.72 GB (≤ 1% above the 15.56 GB ceiling).
- If B5 fails either NLL or wall, F-RegStack-2 fires — publish the
  honest result (per Phase-3 P6 rule), then decide whether to ship a
  subset or abandon the whole stack.

### 3.3 Decision tree for B5

All val NLL comparisons use **main-head CE NLL only** (not including
the Z-loss auxiliary term or the MTP auxiliary head's loss) — otherwise
the comparison is apples-to-oranges across configs with different
total-loss compositions. Z-loss and MTP auxiliary terms are reported
separately for visibility but don't enter the gate arithmetic.

Let `Δi` = main-head val NLL improvement of run `Bi` vs `B0` (so
positive Δi means lower NLL than baseline).

```
if Δ4 ≥ max(Δ1, Δ2, Δ3):
    B5 := B4 (all three stacked; additive or super-additive)
elif Δ4 ≥ 0.02:
    B5 := B4 (sub-additive but still clears B5 gate threshold)
elif max(Δ1, Δ2, Δ3) ≥ 0.02:
    B5 := single best Bi (drop interactions, ship the winner)
else:
    no B5; publish negative pilot result; close the spec
```

Note: pairwise configs (e.g., Z-loss + QK-Norm only) are not tested in
the pilot arc. If post-hoc analysis of B4 suggests one mechanism is
actively harmful in the stack, the implementation team may propose a
follow-up 5k pilot at the pair config before committing to B5 — but
that's an arc extension, not the default path.

### 3.4 Production retrain handoff

If B5 PASSes:
- Archive current v5+FP8 ship at
  `database/checkpoints/chiron_1B_T16384_v5_fp8_phase2/`.
- New ship checkpoint: `chiron_1B_T16384_v5_fp8_regstack_phase2.final`.
- Update `CLAUDE.md` "Current Production Flagship" block to point to
  the new ship, with NLL/throughput/VRAM numbers and the reproduce
  command (`sh run.sh flagship --zloss-coef 1e-4 --qk-norm
  --mtp-depth 1 --mtp-coef 0.1`).
- Write `research/REGSTACK_PHASE2_PASS_2026_MM_DD.md` with the
  trajectory table, gate-by-gate evidence, and per-mechanism
  attribution from the 5k pilots.
- Memory entry: regstack pilot + Phase 2 result, both the headline
  and per-mechanism deltas.

---

## 4. Risks (pre-registered)

| Risk | Probability | Mitigation |
|---|---|---|
| **R-RegStack-1:** 5k single-seed noise floor (~±0.05 nat) hides per-mechanism signals below ±0.02 nat | medium | Run B0 twice for same-day variance calibration; if either mechanism is borderline, extend its pilot to 10k steps before stacking |
| **R-RegStack-2:** QK-Norm interacts badly with SCFA's compressed inner-attention representation (the L2 norm on a compressed vector is not the same as L2 norm on the full Q/K) | medium | First B2 run is the integration test; if val NLL diverges or training is unstable, fall back to QK-Norm on outer attention only |
| **R-RegStack-3:** Z-loss interacts with FP8 readout — the log(Z) term may amplify FP8 quantization noise | low | Compute Z-loss in FP32 from the FP32 logits BEFORE FP8 quantization in the readout backward; if still unstable, gate Z-loss to FP32 readout only |
| **R-RegStack-4:** MTP increases readout compute ~2% but the FP8 path may have higher-than-budgeted overhead because the extra MTP GEMM has a different shape | low | Profile B3 readout block specifically; if wall > 5% over B0, drop MTP from B4/B5 |
| **R-RegStack-5:** B4 stacked is sub-additive because Z-loss and MTP both regularize the readout (overlapping mechanism) | medium | If B4 < B1 + B3 by more than 30% of the sum, ship the better single + QK-Norm instead of all three |
| **R-RegStack-6:** The 30k Phase-2 PASSes pilot bar at +0.02 nat but doesn't hold up at downstream eval — the val NLL improvement doesn't translate to top-k accuracy | low | Report top-1/top-5/top-10 and position-stratified NLL at B5 step 30000 alongside aggregate NLL; if PPL improves but top-k regresses, flag for further investigation |
| **R-RegStack-7:** Determinism break — QK-Norm reduction or MTP loss aggregation introduces non-deterministic ordering | low | Use the existing within-warp deterministic reduction policy from `DETERMINISM_AND_CONCURRENCY.md`; test bit-identical reruns at the same seed before running B1-B5 |

---

## 5. Out of scope for this spec

- **Deep MTP** (DeepSeek-V3 style with extra transformer layers per
  MTP depth) — separate spec if simple MTP PASSes.
- **Dropout sweep** — `embeddingDropoutRate` and `residualDropoutRate`
  exist but are not in this spec. Generally weak at 1B scale per the
  literature; can be a follow-on cheap A/B if the user wants.
- **R-Drop / SAM** — 2× compute, bad ROI right after a wall-time
  optimization arc. Deferred indefinitely.
- **Label smoothing** — commonly hurts at LLM scale (degrades
  calibration / log-prob). Not pursued.
- **EMA of weights** — small VRAM cost, useful for eval but not for
  the optimization arc itself. Separate spec if desired.
- **Multi-seed validation** of the new stack at any step count —
  deferred to a separate gate after B5 PASSes (per the iter-93/94
  precedent: single-seed Phase 1 + Phase 2, multi-seed in a
  subsequent N=3 confirmation arc).
- **Downstream eval (LAMBADA, MQAR-Pile, etc.)** — per Phase-3 P10
  (no external datasets); position-stratified val NLL on
  `pretok-data/` validation split is the eval methodology.

---

## 6. Deliverables checklist

- [ ] Pre-register this spec in memory (key:
      `chiron_1b_regularization_stack_prereg`).
- [ ] Implement Z-loss (mechanism A) — config field, forward, backward,
      flag wiring, strict-bar test at λ=0.
- [ ] Implement QK-Norm (mechanism B) — config fields, forward (inner
      + outer attention), backward, flag wiring, strict-bar test at
      `qkNormEnabled=false`.
- [ ] Implement simple MTP (mechanism C) — config fields, forward,
      backward, target shifting, flag wiring, strict-bar test at
      `mtpDepth=0`.
- [ ] Run B0 baseline 5k.
- [ ] Run B1 (Z-loss alone) 5k; compare to B0.
- [ ] Run B2 (QK-Norm alone) 5k; compare to B0.
- [ ] Run B3 (MTP alone) 5k; compare to B0.
- [ ] Run B4 (all three) 5k; check additivity vs B1/B2/B3.
- [ ] Apply decision tree (§3.3) to select B5 config.
- [ ] Run B5 30k Phase-2 retrain at selected config.
- [ ] Write `research/REGSTACK_PHASE2_<RESULT>_2026_MM_DD.md` with
      gate-by-gate evidence and per-mechanism attribution.
- [ ] If PASS: archive old ship, publish new ship, update CLAUDE.md,
      memory entries.
- [ ] If FAIL: honest negative-result publication per Phase-3 P6,
      no silent re-targeting.

---

## 7. Repro commands

**5k pilot arc (Phase 1):**
```bash
cd ~/dev/glades-trainer

# B0: baseline (current v5+FP8 ship recipe, 5k steps)
sh run.sh flagship --fp8-readout-fwd --steps 5000 --seed 1337 \
    --val-every 500 \
    --save-prefix chiron_1B_T16384_b0_5k

# B1: Z-loss alone
sh run.sh flagship --fp8-readout-fwd --zloss-coef 1e-4 \
    --steps 5000 --seed 1337 --val-every 500 \
    --save-prefix chiron_1B_T16384_b1_zloss_5k

# B2: QK-Norm alone
sh run.sh flagship --fp8-readout-fwd --qk-norm \
    --steps 5000 --seed 1337 --val-every 500 \
    --save-prefix chiron_1B_T16384_b2_qknorm_5k

# B3: MTP alone
sh run.sh flagship --fp8-readout-fwd --mtp-depth 1 --mtp-coef 0.1 \
    --steps 5000 --seed 1337 --val-every 500 \
    --save-prefix chiron_1B_T16384_b3_mtp_5k

# B4: stacked
sh run.sh flagship --fp8-readout-fwd --zloss-coef 1e-4 --qk-norm \
    --mtp-depth 1 --mtp-coef 0.1 \
    --steps 5000 --seed 1337 --val-every 500 \
    --save-prefix chiron_1B_T16384_b4_stacked_5k
```

**30k Phase 2 (B5; flags depend on §3.3 decision tree outcome — example
shows full stack):**
```bash
cd ~/dev/glades-trainer
sh run.sh flagship --fp8-readout-fwd --zloss-coef 1e-4 --qk-norm \
    --mtp-depth 1 --mtp-coef 0.1 \
    --steps 30000 --seed 1337 --val-every 1500 \
    --save-prefix chiron_1B_T16384_v5_fp8_regstack_phase2
```

If the flag names above don't exist in `run.sh` after implementation,
invoke `./build/glades_chiron_train` directly with the equivalent flag
list (the run.sh recipe is a convenience wrapper).

---

This pre-registration is committed before any implementation code
lands. Per the Phase-3 program P6 prohibition (no silent claim-dropping),
if any gate above fails, the result is published honestly and the
program adapts — not silently re-targeted.
