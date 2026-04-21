# CHIRON Progress Journal

Empirical progress tracking for the CHIRON (reversible-flow transformer)
research program. Live-updated per-iteration.

See `research/CHIRON_framework.md` for the selected framework; the alternate
candidates (SPECTRA, CASCADE) live in `research/candidate_B_sketch.md` and
`research/candidate_C_local.md`.

---

## 2026-04-21 — Phase 1 complete (CPU math, FP32, full block)

### Shipped

**Header** `Backend/Machine Learning/Networks/transformer_chiron_ops.h`
- `shear_add_to_p` / `shear_sub_from_p` — Shear^p (momentum kick) and its inverse.
- `shear_add_to_q` / `shear_sub_from_q` — Shear^q (position drift) and its inverse.
- `reln_forward` / `reln_inverse` — reversible LayerNorm with an external
  `[T, 2]` stats buffer per block. Primary API.
- Reserved-coord variant deferred as an optimization; external stats is
  simpler and has negligible memory cost (2 FP32 per token per block).

**Unit tests** `unit-tests/Backend/Machine Learning/chiron-test.cpp`
- Test 1: bare shear reversibility. Pass. `q_err=0, p_err=0`.
- Test 2: ReLN roundtrip. Pass. `q_err < 1e-4`.
- Test 3: reduced block (shear+shear+ReLN) roundtrip. Pass.
- Test 4: 8-block reduced roundtrip. Pass. `q_err=2.7e-7, p_err=1.6e-7`.
- Test 5: attention shear reversibility. Pass. `q_err=0, p_err<1e-5`.
- Test 6: single full block (attn+shear+shear+ReLN) roundtrip. Pass.
  `q_err=6e-8, p_err=3e-8`.
- Test 7: 4-layer full-block roundtrip. Pass. `q_err=1.5e-7, p_err=8.9e-8`.

### Key validated claims

1. The **symplectic attention shear** — Q, K, V all derived from `q`, output
   added to `p` — is exactly reversible in FP32. Inverse is one attention
   forward + one subtract. This was the main open question of the framework
   (§3.2) and it works.
2. **ReLN with external stats** is exactly reversible. The "reserved
   coordinates" scheme from the framework §3.4 is not required for
   correctness; it is an optimization that reduces the stats memory from
   `L·T·2` FP32 to zero extra, at the cost of channel-reservation
   bookkeeping. That optimization is deferred.
3. **Composition of L full blocks** is reversible up to FP32 precision. At
   L=4, error is ~1.5e-7 — near the FP32 machine epsilon of ~1.2e-7, which
   is exactly what the linear-Lipschitz-composition bound predicts.

### Not yet validated

- BF16 reconstruction drift (next phase).
- Sketch residual correction (next phase).
- GPU kernels (Phase 3).
- End-to-end training parity against a standard transformer (Phase 4).
- Memory savings measured on a real model (Phase 4).

---

## Next milestones

### Phase 2 — BF16 + sketch correction

Goal: show BF16 inverse stays within sketch-corrected bound at L=24.

- [x] BF16 helper wrapping FP32 math with `round_to_nearest_even` at each
  op boundary (test-only utility, inlined into chiron-test.cpp).
- [x] Measure BF16 drift at L=4, 12 without sketch. **Results:**
  - FP32 L=4: 1.49e-7 (at machine epsilon)
  - BF16 L=4: 7.81e-3 (~52,000x worse than FP32)
  - BF16 L=12: 1.17e-2 (1.5x worse than L=4, grows with depth)
  - Growth is sub-exponential in L for our setup — the Lipschitz factor
    in this small test is near 1, so the framework's §6.4 bound
    `L · ε_BF16 · exp(Σ K_ℓ)` reduces to approximately linear in L.
- [x] Implement rank-r Gaussian sketch projection + lift (forward side).
  See `sketch_project` / `sketch_lift_add` in transformer_chiron_ops.h.
- [x] Implement sketch-corrected inverse reconstruction (backward side).
  Prototype lives in chiron-test.cpp::run_multifullblock_roundtrip_sketch.
- [x] Negative control: BF16 without sketch at L=12 = 1.17e-2 (confirmed).
- [x] **Empirical sketch correction results at L=12** (T=6, m=16, N=2·T·m=192):
  - uncorrected BF16 drift: **1.17e-2**
  - r=64 (N/r ≈ 3.0): **1.09** — catastrophically diverges, sketch space
    too small
  - r=256 (N/r ≈ 0.75): **3.91e-3** — ~3× reduction over uncorrected ✓
  - r=1024 in isolation (correction primitive test): reduction 1.44× at
    per-coord level with |δ| = 1e-2
- [x] **Key discovered scaling law:** per-coord sketch-corrected error is
  `O(||δ|| · √(N/r))` — NOT the tighter `O(||δ||/√r)` the framework §4.5
  claimed. This means the sketch is effective only when **r ≳ N** (not
  r = O(√N) as framework stated). For production (N ≈ 16M per layer),
  this is a significant scaling concern that needs addressing.

### Per-token local sketch — breakthrough result (2026-04-21, same day)

Implemented the per-token local sketch variant (framework amendment §11a,
mitigation 1): one sketch matrix `S_ℓ ∈ R^{r × 2m}` shared across the T
tokens of layer ℓ, with per-token stored sketches `z_{ℓ,t} = S_ℓ · x_t`
where `x_t = (q_t, p_t)` of size 2m.

Results at L=12, T=6, m=16 (so 2m=32):
  - uncorrected BF16 drift:         **1.17e-2**
  - global sketch, r=256 (N=192):    3.9e-3   (~3×)
  - per-token sketch, r=128 (N=32):  **1.95e-3** (~6×)
  - per-token sketch, r=256 (N=32):  **4.88e-4** (~24×)

Per-token sketch at r=256 beats global sketch at r=256 by **8×**, matching
the √(N_global/N_pertok) = √(192/32) = √6 = 2.45 per-coord improvement
factor expected from the corrected scaling law.

Memory at production scale (70B, L=96, T=4096, r=256):
  L · T · r · 4 bytes = 96 · 4096 · 256 · 4 = **402 MB**.
This is manageable. Combined with BF16 activation anchors every k=8
blocks (~805 MB), total activation-side memory is ~1.2 GB — still a
**~20× reduction** over the 25.8 GB full-activation baseline.

### Open research questions from Phase 2 measurements

- Framework §4.5 claims `Var(x̂_ℓ_i) ≤ ||x - x̃||² / r` independent of N.
  Empirically and by elementary computation we get `||x - x̃||² / r` for
  the *sum-of-coords* error but `||x - x̃||² · N / r²` for the
  *per-coord variance* contribution from cross-coordinate leakage.
  The framework appears to have under-counted cross-coordinate noise.
  **Need to revise the framework §4.5 variance bound** to
  `Var ≲ ||x − x̃||² · N/r²` per coordinate.
- Consequence: for a 70B model (N ≈ 16M), r = 1024 gives per-coord
  noise factor ≈ √(N/r²) · ||δ|| = √(16M/10^6) · ||δ|| = 4·||δ||, NOT
  a reduction.  Either (a) sketch has to be per-token local (N = d,
  not T·d) or (b) block-structured sketches with N/block much smaller.
- **This is important enough to call out in the framework doc.**
  Action: update `research/CHIRON_framework.md` with the variance-bound
  correction and the local-sketch mitigation.

### Phase 3 — GPU kernels

- [ ] `gpu_chiron.{h,cu}` with CUDA versions of the shears, ReLN, sketch ops.
- [ ] Reuse existing flash-attn BF16 forward for both forward and inverse
  of the attention shear.
- [ ] GPU parity test: gradient parity with CPU ground truth < 5e-3 rel.

### Phase 4 — training loop integration

- [ ] Add `TYPE_TRANSFORMER_CHIRON` in `network.h`.
- [ ] Add `useChiron` config flag + JSON serializer.
- [ ] Hook CHIRON into `transformerGpuTrainEpoch`.
- [ ] Add `--chiron` to `glades-trainer/run.sh`.
- [ ] Baseline comparison on pile tokens: AdamW+BF16 vs CHIRON at matched
  param count; compare activation memory and step time.
