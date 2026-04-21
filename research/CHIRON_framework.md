# CHIRON: Canonical Hamiltonian Involutive Reversible Operator Network

## A Reversible-Flow Transformer Paradigm for O(1)-in-Depth Activation Memory

---

## 0. Status and selection rationale (2026-04-21)

This document is the **chosen** paradigm-shift framework after considering three
materially different candidates. Full candidate texts are preserved in:

- `research/candidate_A_reversible.md` — CHIRON (this framework, selected)
- `research/candidate_B_sketch.md` — SPECTRA (sketch-native activation propagation)
- `research/candidate_C_local.md` — CASCADE (local-objective decoupled training)

### 0.1 Candidate summary

| Axis | CHIRON (A) | SPECTRA (B) | CASCADE (C) |
|---|---|---|---|
| Core mechanism | Bijective symplectic forward; activations reconstructed by explicit inverse | Activations propagated as structured random sketches (SRHT + CountSketch); unbiased JL gradient estimator | Per-block variational info-bottleneck loss; cross-block gradient cut |
| Activation memory reduction | ~95× (O(1) in depth) pure, ~8× with k=8 anchoring | ~8–32× from `d→k`, `T→m` sketching | L-fold if pipeline-parallel, ~L/k otherwise |
| Compute impact | Backward = 3× fwd (equal to memory-comparable full-ckpt baseline); forward unchanged | Sketched GEMMs `d/k`× cheaper; unsketching patch adds O(T) overhead per softmax | Each block independent; wall-clock gated by slowest block |
| Theoretical guarantee | Exact inverse in exact arithmetic; unbiased sketch correction var O(1/r) under BF16 | Matrix-Bernstein / JL unbiasedness with variance `d/k` control-variate corrected | Telescoping Donsker–Varadhan: local descent ⇒ global descent up to staleness |
| Determinism / DDP | Seeded sketches, deterministic cuBLAS — fits existing policy | Seeded RNG for Ω per layer/step — fits policy | Per-block async state — needs new DDP semantics |
| Single-GPU benefit | **Immediate** — O(1) activation pathway works on 1 GPU | Immediate — sketch reduces per-layer memory on 1 GPU | Marginal single-GPU (layer-parallelism unused); best ≥ 8-GPU cluster |
| Attention compatibility | Reuses existing flash-attn forward kernel verbatim for forward *and* inverse | Requires new sketched-attention path; CountSketch signed-hash trick for exact scores | Standard attention within each block; no kernel changes |
| Composability with ATLAS/VESTA/HELIOS/BF16 | Orthogonal — all optimizers unaffected | Orthogonal at optimizer level; block-recursive sketching changes what optimizer sees per-step | Orthogonal; each block is independently optimized |
| Risk | BF16 inverse drift (addressed by sketch + anchoring) | Nonlinearity handling (softmax, SiLU) has thorny "unsketching gap" edge cases | Local → global proof depends on β,γ,K hyperparameters being well-chosen |
| Architectural invasiveness | Transformer block replaced; LayerNorm replaced; training loop grows inverse pass | Transformer block replaced; attention and MLP rewritten with sketch ops | Training loop replaced with per-block schedule; heads added per block |

### 0.2 Selection

**CHIRON is selected** for this codebase because:

1. **Magnitudes-level memory on our hardware.** On a single GPU with depth
   L≥48, CHIRON attacks activation memory (the largest remaining axis after
   BF16/ATLAS/VESTA/HELIOS) with a ~95× asymptote and 8–10× in the hybrid
   `k=8` anchor regime. SPECTRA offers 8–32×; CASCADE's L-fold win requires a
   pipeline-parallel cluster we do not currently have.

2. **Cleanest math.** Bijectivity is a structural invariant, not a stochastic
   guarantee. The inverse is an algebraic identity; the only randomness is
   the sketch residual correction, where it is controlled. SPECTRA gives
   unbiased gradients but large variance `d/k`; CASCADE's local→global
   theorem depends on tight bounds being realized at all `L` blocks.

3. **Reuses existing infrastructure.** The flash-attention forward kernel
   (already BF16-input, already shipped) is the exact building block we need
   for both forward *and* inverse. No new attention kernel required for the
   forward leg. SPECTRA requires new sketched-attention kernels; CASCADE
   requires new per-block loss and training loop scaffolding.

4. **Composes with every existing optimizer and DDP policy.** The shipped
   stack (ATLAS / VESTA / HELIOS / AdamW BF16-state, DDP with optional
   compressed all-reduce) operates on parameter gradients and needs no
   change. CHIRON only changes how gradients are *obtained*, not what is
   done with them.

5. **Minimal falsifying experiment is cheap.** A 4-layer d=64 T=16 CPU test
   can validate reversibility in one unit test; a BF16+sketch test in a
   second; a GPU parity test in a third. All of these are runnable inside
   the existing `unit-tests/` framework with no new external deps.

### 0.3 Why we do not discard SPECTRA and CASCADE

Both remain viable research directions layered **on top of** CHIRON:

- SPECTRA's sketched-activation idea is already how CHIRON controls BF16
  drift. At larger L we may combine a sketched residual-stream with CHIRON's
  bijective blocks, giving multiplicative savings.
- CASCADE's variational info-bottleneck gives a principled per-block
  regularizer. A future "CHIRON + CASCADE" hybrid could train each CHIRON
  block with a local predictive readout in addition to the global LM loss.

Both hybrids are out of scope for the minimal CHIRON rollout but belong on
the research roadmap (§12).

---

## 1. Executive summary

CHIRON replaces the standard transformer block with a composition of four
symplectic diffeomorphisms on a paired hidden-state space `(q, p) ∈ R^{T×m} ×
R^{T×m}` (with `m = d/2`). Each block is provably bijective with a unit
Jacobian determinant, so activations at every intermediate layer can be
exactly recovered from the final output by running the block inverse. The
backward pass consumes **O(1) activation storage in depth L** — the current
`(q, p)` pair only — rather than the ~10×L×T×d floats that the existing
transformer stores in `GpuTransformerScratch`.

The symplectic attention shear is a structural novelty: Q, K, V are all
derived from `q`; the attention output is added to `p`; the forward map
`(q, p) ↦ (q, p + Y(q))` is unit lower-triangular in block form and thus
symplectic with an explicit inverse `(q', p') ↦ (q', p' − Y(q'))`
computed with *one* flash-attention forward call. Two symplectic MLP
shears and a **reversible LayerNorm** (which stores `(μ, log σ)` into
reserved coordinates of `p` rather than discarding them) complete the
block.

To control BF16 round-off drift across the inverse reconstruction, each
forward pass stores a rank-`r` Gaussian sketch `z_ℓ = S_ℓ · vec(x_ℓ)` in
FP32 (~10⁵ bytes total for L=96). The backward pass uses the sketch to
produce an unbiased correction with variance O(1/r). A hybrid mode stores
full activations at every k-th block ("anchor") and sketches in between,
giving the practical sweet spot of 8× activation memory reduction with no
backward-compute penalty relative to a memory-comparable checkpointed
baseline.

**Memory**: activations reduced by 8–95× depending on `k`. Optimizer,
weights, gradients unchanged.
**Compute**: forward identical to baseline; backward = 3× forward FLOPs
(1 recomputation + standard backward), identical to full-checkpointing
baseline. No wall-clock penalty at equal memory.
**DDP**: no new all-reduce collectives. Sketch seeds broadcast once.
**BF16**: compatible by construction via sketch correction.
**Novelty vs RevNet/Reformer**: attention is the shear itself (not a
post-hoc additive coupling around a non-invertible core), reversible LN is
new, BF16-stable reconstruction is new.

---

## 2–10. Full framework

See `research/candidate_A_reversible.md` for the full development of
sections 2–10:

- §2 Primitive objects and state space
- §3 Forward evolution law (symplectic attention shear, Störmer–Verlet MLP,
  reversible LayerNorm)
- §4 Backward reconstruction algorithm (exact-arithmetic and BF16-stabilized)
- §5 Objective and training dynamics
- §6 Theoretical properties (well-posedness, Lipschitz, stability,
  continuous-time limit)
- §7 Expressivity (universal for volume-preserving seq-to-seq maps)
- §8 Memory and compute complexity table
- §9 Failure modes (reconstruction blow-up, reserved-coord interference,
  attention sharpening, DDP numeric asymmetry)
- §10 Minimal prototype implementation path

---

## 11. Implementation roadmap

### Phase 1 — minimal CPU prototype (iteration 1–3)

Goal: prove bijectivity on a 4-layer d=64 T=16 model in C++98.

- [ ] Add `TYPE_TRANSFORMER_CHIRON` to `Backend/Machine Learning/Networks/network.h`.
- [ ] Add `useChiron`, `chironSketchRank`, `chironAnchorPeriod` to
  `training_config.h` and its JSON serializer.
- [ ] Implement `reln_forward` / `reln_inverse` (reversible LayerNorm with
  reserved-coordinate stats) as header-only functions in a new
  `transformer_chiron_ops.h`.
- [ ] Implement the symplectic attention shear on CPU: reuse
  `scaled_dot_product_attention_forward` from `transformer_ops.h` to compute
  `Y(q)`, then add to `p`.
- [ ] Implement the two MLP shears and `Φ_ℓ^{-1}` on CPU.
- [ ] Unit test `chiron-reversibility`: 4-layer d=64 T=16 FP32 model.
  Assert `‖x̂_0 − x_0‖_∞ < 1e-5` after forward + inverse in FP32.

### Phase 2 — BF16 + sketch correction (iteration 3–5)

Goal: show BF16 inverse converges to FP32 ground truth within sketch tolerance.

- [ ] Add `sketchSeeds[L]` and `sketchZ[L][r]` to `TensorTransformerState` +
  `GpuTransformerWeights`.
- [ ] Implement Gaussian sketch projection (seed-derived matrix, never
  stored) as a CPU then GPU kernel.
- [ ] Wire the sketch correction into the CPU inverse.
- [ ] Extend unit test `chiron-reversibility` with BF16 + r=128,
  assert `‖x̂_0 − x_0‖_∞ < 1e-2`.
- [ ] Add negative-control test: BF16 without sketch correction fails
  (bound > 1e-1).

### Phase 3 — GPU path (iteration 5–8)

Goal: run CHIRON on GPU, parity with CPU.

- [ ] Create `Backend/Machine Learning/Networks/cuda/gpu_chiron.{h,cu}`.
- [ ] GPU kernels: `reln_forward`, `reln_inverse`, `sketch_project`,
  `sketch_lift`.
- [ ] Hook CHIRON into `transformerGpuTrainEpoch` behind `useChiron`
  flag. Reuse existing flash-attention BF16 kernels for the attention
  shear.
- [ ] Unit test `chiron-gpu-parity`: gradient parity with CPU ground truth
  at element-wise `|Δgrad| / |grad| < 5e-3`.
- [ ] Memory test: HBM usage < 20% of full-activation baseline at d=512, L=24.

### Phase 4 — glades-trainer integration (iteration 8–10)

Goal: train largest LLM we can on pile data.

- [ ] Add `--chiron` flag to `glades-trainer/run.sh`.
- [ ] Wire through to `trainer/main.cpp` → `TrainingConfig`.
- [ ] Run comparative benchmarks: baseline AdamW+BF16 vs CHIRON at
  matched parameter count, log memory and step time.
- [ ] Target: push dModel from 1024 to 4096+ at fixed GPU VRAM.

### Phase 5 — optimizations and composability (iteration 10+)

- [ ] Fuse the sketch projection into the forward pass for HBM efficiency.
- [ ] Explore CHIRON + VESTA (spectral-homeostasis + symplectic).
- [ ] Explore CHIRON + CASCADE (local objectives per block).
- [ ] Benchmark attention-shear gradient stability at depth L ≥ 96.
- [ ] Write `research/CHIRON_PROGRESS.md` journal tracking empirical findings.

---

## 12. Falsification criteria

From §10.4 of `candidate_A_reversible.md`:

- If CHIRON LM loss on WikiText-103 fails to close the gap to a
  standard-transformer baseline by ≥ 95% within 3× the baseline training
  budget → **retire** the framework.
- If gradient variance (batch-split estimator) exceeds 10× standard
  transformer at r=1024 → **retire**.
- If BF16 reconstruction drift > 2⁻⁴ after k=8 anchoring → **retire** (and
  try SPECTRA's sketched-activation approach instead).

If any of the three retirement criteria fire, fall back to candidate B
(SPECTRA) as the second-best direction for this codebase.
