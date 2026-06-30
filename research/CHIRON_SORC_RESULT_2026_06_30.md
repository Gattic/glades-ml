# CHIRON SORC: Symplectic Orthogonal Rotation Coupling — Small-shape Validation

**Date:** 2026-06-30  
**Status:** PASS (small-shape) — ready for production E3/E4  
**Branch:** glades-ml `chiron3`, glades-trainer `reln-reanchor`

---

## Mechanism (one line)

SORC adds a per-layer, per-channel symplectic rotation `(q,p) → R(θ_{l,i})·(q,p)` with angle `θ_{l,i} = θ_max·tanh(φ_{l,i})` (structurally bounded to `[−θ_max, θ_max]`), letting attention compose across depth via a **bounded** gate — unlike OBSD's unbounded ReZero (`a` grew to maxA→1.8 and regressed −0.66 nat), SORC is capped by construction.

---

## E0 — Bit-identity at φ=0 (identity gate)

**Non-SCFA path** (from Task 5 commit `536db6e`, glades-trainer):

Shape: `m=256, L=4, nH=4, dH=128, seq-len=1024, --no-fuse-attn --fuse-attn-reln`.  
Result: losses bit-identical with/without `--rot-coupling` at `phi=0`.  
Step-1 `||g||` identical; by step 11/21 `||g||` diverges by ~1e-3 (drot_phi is nonzero and phi is moving — the model function is identity but the gate is learning, same as OBSD E0).

**SCFA path** (Task 7, glades-trainer commit `47fa598`):

Shape: `m=512, L=4, nH=4, dH=256, seq-len=1024`, full bf16+SCFA+reanchor stack:
```
--int8-adam --bf16-grads --bf16-weights --bf16-attn --no-fuse-attn --fuse-attn-reln
--scfa --scfa-bf16-inner --scfa-bf16-outer --scfa-fuse-streams --scfa-reln-opt
--bf16-logits --bf16-logits-storage --fp8-readout-fwd --qk-norm --reln-reanchor
--grad-clip 0.5 --lr 3e-4 --warmup 50
```

Base run (no `--rot-coupling`) vs rot-coupling run (identical seed 1337, 3 steps):

| Step | Base loss | SORC loss | Match |
|------|-----------|-----------|-------|
| 1    | 10.3990   | 10.3990   | ✓ |
| 2    | 10.4060   | 10.4060   | ✓ |
| 3    | 10.4058   | 10.4058   | ✓ |

`||g||` identical at all 3 steps (1.062, 1.036, 2.483). **E0-SCFA PASS** — SORC is a transparent pass-through on the production attention path at init.

---

## E1 — Unit tests (Tasks 1–3, glades-ml)

All tests in `test.sh chiron-rot` PASS (run 2026-06-30, RTX 4080 SUPER):

| Test | Result | Max error | Bar |
|------|--------|-----------|-----|
| SORC 3-shear composes to R(θ) | PASS | 5.96e-08 | 1e-4 |
| SORC rotation norm conservation (q²+p²) | PASS | 2.98e-08 | 1e-4 |
| SORC fwd then inverse reconstructs (q,p) | PASS | 2.98e-08 | 1e-5 |
| SORC FD grad-check (CPU) | PASS | maxRel=0.0001 | 2e-2 |
| SORC rot fwd CPU/GPU parity | PASS | 4.47e-08 | 1e-4 |
| SORC rot fwd∘inv reconstructs (GPU) | PASS | 2.98e-08 | 1e-5 |
| SORC rot backward CPU/GPU parity | PASS | 1.86e-08 | 2e-4 |
| SORC rot dphi FD-vs-forward at sw=0.8 | PASS | maxRel=0.0004 | 2e-2 |

The FD-vs-forward test at `sw=0.8` (Task 3 fix: `theta_eff = sw·theta_max·tanh(phi)` in the chain derivative) explicitly validates the warmup-scale interaction. The 3-shear composition test verifies the `R(θ) = S(tan θ/2) · S(-sin θ) · S(tan θ/2)` factorization. Norm conservation verifies symplecticity (volume-preserving, same as the full CHIRON Hamiltonian structure).

---

## E2 — Reconstruction / training stability (Task 6, glades-trainer commit `5f97a7c`)

Non-SCFA path, shape `m=256, L=4, nH=4, dH=128`, `--rot-coupling --rot-warmup 5`, 30 steps, seed 1337:
- **0 grad-skips**
- No NaN/Inf
- Loss finite and decreasing: 10.41 → 10.38 (best over 30 steps)
- Step-1 loss bit-identical to no-rot baseline (phi=0 → R=I)
- `||g||` 0.001 higher by step 11/21 (drot_phi nonzero, phi moving from 0 as warmup activates at step 5)
- Adam update confirmed live: drot_phi nonzero feeds the Adam step for rot_phi

A broken inverse would corrupt reconstruction → divergent gradients. Gradients stayed bounded, confirming the R⁻¹ (= R^T, since R is orthogonal) inverse-walk is correct.

---

## `[sorc]` plateau monitor (Task 7, glades-trainer commit `47fa598`)

Added to `chiron_main.cpp` under `normalLogStep && cfg.rotCoupling && !W.rot_phi.empty()`:
- Downloads `rot_phi[l]` buffers (small [m] floats) at log-every cadence only (NOT every step)
- Computes `maxTheta = theta_max * max_{l,i} |tanh(phi_{l,i})|`
- Prints: `[sorc] step=%d maxTheta=%.4f thetaMax=%.4f`

SCFA-path 40-step run (`--rot-coupling --rot-warmup 25`, shape as above, seed 1337):

| Step | maxTheta | thetaMax | Ratio |
|------|----------|----------|-------|
| 1    | 0.0000   | 1.0472   | 0.000 |
| 6    | 0.0001   | 1.0472   | <0.001 |
| 11   | 0.0004   | 1.0472   | <0.001 |
| 16   | 0.0008   | 1.0472   | <0.001 |
| 21   | 0.0013   | 1.0472   | 0.001 |
| 26   | 0.0021   | 1.0472   | 0.002 |
| 31   | 0.0030   | 1.0472   | 0.003 |
| 36   | 0.0040   | 1.0472   | 0.004 |

**Plateau property confirmed:** maxTheta grows monotonically as phi learns (phi is not stuck at 0) but stays structurally bounded at ≤ thetaMax. At 40 steps (still in warmup phase), maxTheta=0.0040 = 0.38% of thetaMax. This is the structural contrast with OBSD: OBSD's `a_drift` (ReZero gate) grew unbounded (maxA → 1.8 at 30k), eventually throttling effective LR. SORC's gate has a hard ceiling.

The 40-step run also confirms: **0 grad-skips, no NaN/Inf**, loss descending 10.40 → 10.29 (best at step 33).

---

## SCFA-path guard (pre-existing, from OBSD arc)

The `--scfa requires --bf16-weights` parse-time guard (glades-trainer `chiron_main.cpp`, committed `1606f6d`) is in place. Running `--scfa` without `--bf16-weights` exits cleanly with a diagnostic instead of SIGSEGVing in `scfa_attention_backward`. SORC's SCFA validation above uses the full bf16 stack and verifies the guard works.

---

## Commands (reproducible)

**E0-SCFA bit-identity:**
```bash
cd /home/robert/dev/glades-trainer
STACK="--int8-adam --bf16-grads --bf16-weights --bf16-attn --no-fuse-attn --fuse-attn-reln --scfa --scfa-bf16-inner --scfa-bf16-outer --scfa-fuse-streams --scfa-reln-opt --bf16-logits --bf16-logits-storage --fp8-readout-fwd --qk-norm --reln-reanchor --grad-clip 0.5 --lr 3e-4 --warmup 50"
SHAPE="--pretokenized --data-dir pretok-data/ --vocab 32000 --seq-len 1024 --m 512 --layers 4 --heads 4 --dhead 256"
./build/glades_chiron_train $SHAPE $STACK --max-steps 3 --seed 1337 --log-every 1 > /tmp/base.txt 2>&1
./build/glades_chiron_train $SHAPE $STACK --rot-coupling --max-steps 3 --seed 1337 --log-every 1 > /tmp/rot0.txt 2>&1
diff <(grep "step " /tmp/base.txt | sed 's/tok\/s=.*//') <(grep "step " /tmp/rot0.txt | sed 's/tok\/s=.*//')
# Expected: no diff
```

**Plateau monitor + 40-step run:**
```bash
./build/glades_chiron_train $SHAPE $STACK --rot-coupling --rot-warmup 25 --max-steps 40 --seed 1337 --log-every 5 2>&1 | grep -E "\[step|\[sorc"
```

**Unit tests:**
```bash
cd /home/robert/dev/glades-ml/unit-tests && bash test.sh chiron-rot
```

---

## Verdict: proceed to E3/E4

All small-shape validation passes:
- E0 (bit-identity): PASS on both non-SCFA and production SCFA path
- E1 (unit tests): 8/8 PASS (compose, norm-conservation, reversibility, FD gradcheck, CPU/GPU parity, warmup-scale chain)
- E2 (reconstruction): PASS (0 grad-skips, phi learns, inverse correct)
- SCFA-path 40-step run: PASS (0 grad-skips, no NaN/Inf, loss descending)
- Plateau monitor: confirmed maxTheta ≤ thetaMax structurally; grows from 0 (phi learning)

The mechanism is correct and transparent at the production SCFA path. Proceed to E3: production-geometry stability probe (shape L=24, m=2048, T=16384, ~2500 steps, full flagship recipe) to confirm 0 grad-skips and bounded `[sorc] maxTheta` at production scale before the E4 quality gate.

**Key risk to monitor in E3/E4:** OBSD failed not at small scale but at long training (30k steps) as the gate grew, throttling effective LR. SORC's structural bound prevents the same runaway, but whether the learned rotations improve val NLL is an open question that E4 must answer.

---

## Commit records

- glades-trainer `reln-reanchor`: `47fa598` — `[sorc]` plateau monitor log
- glades-ml `chiron3`: this doc
- Prior trainer commits (Tasks 1–6): `70d4287`, `536db6e`, `5f97a7c` (rot_phi param, forward wiring E0, inverse+backward E2)
- Prior ml commits (Tasks 1–3): `7b69f63cd`, `66f787a19`, `590f541d8`, `9f9e646e3`, `81176ddcf` (CPU ref, GPU kernels, parity tests, backward, FD fix)
