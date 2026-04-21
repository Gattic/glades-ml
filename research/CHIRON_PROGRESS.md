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

- [ ] BF16 helper wrapping FP32 math with `round_to_nearest_even` at each
  op boundary (new test-only utility, not production).
- [ ] Measure BF16 drift at L=4, 8, 24 without sketch. Expected: grows
  exponentially in L per §6.4 of the candidate doc.
- [ ] Implement rank-r Gaussian sketch projection + lift (forward side).
- [ ] Implement sketch-corrected inverse reconstruction (backward side).
- [ ] Assertion: with r=128, residual bound < 1e-2 at L=8.
- [ ] Negative control: BF16 without sketch at L=24 should fail.

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
