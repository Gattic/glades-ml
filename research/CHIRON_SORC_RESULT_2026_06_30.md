# CHIRON SORC: Symplectic Orthogonal Rotation Coupling — RESULT: NO-GO

**Date:** 2026-06-30  
**Status:** **NO-GO** — small-shape validation PASSED (E0/E1/E2), but **E3 (production scale) DIVERGED**. Closed as NO-GO; all code default-off.  
**Branch:** glades-ml `chiron3`, glades-trainer `reln-reanchor`

> **TL;DR:** SORC is implemented correctly (verified math + reviewer-confirmed wiring) and the bounded-angle property holds exactly. But at production scale the treatment **diverges at step ~831** (val@2500 **14.45** vs baseline **3.58**, 1619 grad-skips, ‖g‖→4.7e10). Root cause is **fundamental, not a bug**: the symplectic block has `‖p‖ ≈ 17×‖q‖` (reln normalizes q each layer; p is un-normalized and accumulates attention across depth), so the rotation's `q += a·p` is a *p-scaled* perturbation to the much smaller q. The `‖R‖₂=1` conservation preserves the **joint (q,p) norm — which is p-dominated** — and therefore does **not** protect the q subspace. Same root-cause family as OBSD ([[research/CHIRON_OBSD_RESULT_2026_06_27.md]]); different symptom (OBSD stable-but-regresses, SORC unstable/diverges). See **E3 — Production RESULT** below.

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

## E3 — Production RESULT: FAILED (treatment diverges) — DECISIVE

Production geometry (T=16384, L=24, m=2048, nH=16, dH=256), full reanchor-flagship recipe
(`--accum 4 --lr 3e-4 --warmup 750 --sira-warmup 250 --zloss-coef 1e-4 --qk-norm
--sira-coef 1e-2 --sira-energy-weight 1.0 --sira-balance-weight 0.25 --sira-action-weight 0.0
--grad-clip 0.5 --dq-layer-clamp 1.0 --dq-embed-clamp 1.0 --reln-reanchor`), seed 1337,
2500 steps, wide 8-batch val. Baseline = identical minus `--rot-coupling`. Treatment adds
`--rot-coupling --rot-theta-max 1.0472 (π/3) --rot-warmup 250`. Single GPU → serial.

| metric @ 2500 | baseline | SORC (θ_max=π/3) |
|---|---|---|
| val NLL | **3.5803** | **14.4541** (diverged) |
| grad-skips | 0 | **1619** |
| ‖g‖ (last) | 0.340 | **4.7e10** |
| loss scale | 1.000 | **0.000** (collapsed) |
| E0 step-1 loss | 10.7955 | 10.7955 (bit-identical ✓) |

**Divergence trajectory.** SORC tracked the baseline closely through step ~748
(loss 10.80→4.07, best 4.0095@727), then **exploded at step ~831**: loss 4.07→21.0,
‖g‖→6e10, loss-scale→0, and it never recovered (val 14.45 @ 2500). The onset coincides
with **lr finishing its warmup to full 3e-4** AND the learned angle reaching **θ≈0.08**
(maxTheta 0.0806 @ 748). The baseline is perfectly stable at that same lr (‖g‖ 0.34).

**The exploding gradient is exclusively `drot_phi`, in a clean geometric cascade across depth**
(grad-detail step 820):

```
L15.drot_phi  norm = 150          (late layer — shallow in backward)
L14           = 2,963             ×~20
L13           = 51,083            ×~17
...           (≈ ×2.5 per layer)
L03           = 1.02e10
L02           = 2.54e10
L01           = 7.13e10           (early layer — deep in backward)
```

### Why this is fundamental, not a bug

A correct **orthogonal** rotation backward (Rᵀ) preserves the adjoint norm exactly and
*cannot* amplify. We verified the implementation is correct at all three layers:
- **CPU-ref backward** = the exact orthogonal Rᵀ — all three reversed-transpose adjoint
  shears traced by hand (`rot_backward`, `transformer_chiron_ops.h`).
- **Trainer wiring** correct (coeffs recomputed from φ with matching `sw`; inverse-walk
  recovers pre-rotation (q,p); both adjoints Rᵀ-transformed in place) — Task-6 reviewer-confirmed.
- **E0** step-1 bit-identical to baseline (φ=0 ⇒ R=I). The blow-up emerges only as φ grows.

**Root cause — the p/q scale asymmetry.** The SIRA diagnostic shows the block runs with
`mean(p²)/mean(q²) ≈ 300`, i.e. **‖p‖ ≈ 17×‖q‖**, because **reln normalizes q every layer
while p is un-normalized and accumulates attention outputs across depth**. SORC's
`q += a·p` therefore injects a **p-scaled** perturbation into the ~17× smaller q. The
conservation theorem (`‖R‖₂=1`) preserves the **joint** (q,p) norm — but that joint norm is
**p-dominated** (`p²/q²≈300`), so conserving it does **nothing** to protect the q subspace.
This creates positive feedback: rotation perturbs the dynamics → **p runs away**
(`mean(p²)` jumps **373 → 79527, ×213**, exactly at the step-831 explosion, while q² stays
~1.23) → `q += a·p` dumps an enormous p into q → backward adjoints and `drot_phi` explode.

**The conservation argument was mathematically correct but conserves the wrong quantity.**
The forward stayed finite (loss 21, not inf — the isometry held); the *backward* exploded.
Norm-preservation of the forward map does **not** bound the backward gradient when the
phase space is scale-asymmetric.

### Relationship to OBSD

Same root-cause family — both per-layer symplectic-coupling mechanisms founder on the
block's p/q structure ([[research/CHIRON_OBSD_RESULT_2026_06_27.md]]):
- **OBSD** (`q += a⊙tanh(M⁻¹⊙N(p)+b)`, ReZero gate): *stable but regresses* −0.66 nat at
  30k (gate grew unbounded → throttled effective LR).
- **SORC** (orthogonal rotation, hard-bounded angle): *unstable, diverges* at 2500 steps
  (bounded angle held, but the p-dominated conservation didn't protect q).

Fixing OBSD's unbounded gate with a hard angle bound removed the *runaway-gate* failure but
exposed a deeper one: the coupling's perturbation to q is intrinsically p-scaled.

---

## Verdict: NO-GO (closed 2026-06-30)

- **E0** (bit-identity, incl. production SCFA path): PASS
- **E1** (unit tests, 8/8): PASS
- **E2** (small-shape reconstruction): PASS
- **E3** (production-scale, 2500 steps): **FAILED — diverges (val 14.45 vs 3.58, 1619 skips)**
- **E4** (30k decisive gate): **NOT RUN** — the E3 gate failed; running E4 was not warranted.
  The cheap E3 gate caught the divergence in ~3.4 GPU-hr and saved the ~44 hr E4 pair.

Per-layer symplectic **rotation** coupling is ruled out as a perplexity lever for CHIRON:
the bounded-angle design is *implemented correctly and the bound holds*, but the mechanism
**diverges at production scale** because the symplectic block's `‖p‖ ≫ ‖q‖` asymmetry makes
any joint-norm-conserving q↔p coupling a large perturbation to the small q coordinate.

**Engineering is sound, reusable, and committed default-off**: kernels (`chiron_rot_coeffs`,
`chiron_rot_forward` fwd/inv, `chiron_rot_backward` + dphi chain), CPU refs, 8 unit tests
(`test.sh chiron-rot`), the `rot_phi[l]` param (checkpoint bit 1024), the `[sorc]` plateau
monitor, and flags `--rot-coupling` / `--rot-theta-max` / `--rot-warmup`. None affect the
default path (φ=0 ⇒ R=I, E0-verified).

**If anyone revisits cross-depth coupling for CHIRON:** the lever must act in a
**scale-normalized / whitened (q,p) frame** so the perturbation to q is bounded by `‖q‖`,
not `‖p‖`. A joint-norm conservation law is the *wrong* invariant here; the relevant one is
per-subspace (q vs p) scale control. Until that is solved, both additive (OBSD) and
rotational (SORC) per-layer symplectic coupling are NO-GO.

---

## Commit records

- glades-trainer `reln-reanchor`: `47fa598` — `[sorc]` plateau monitor log
- glades-ml `chiron3`: this doc
- Prior trainer commits (Tasks 1–6): `70d4287`, `536db6e`, `5f97a7c` (rot_phi param, forward wiring E0, inverse+backward E2)
- Prior ml commits (Tasks 1–3): `7b69f63cd`, `66f787a19`, `590f541d8`, `9f9e646e3`, `81176ddcf` (CPU ref, GPU kernels, parity tests, backward, FD fix)
