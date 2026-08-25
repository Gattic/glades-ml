# CHIRON Reversible FFN + GQA — Design & Implementation Plan (2026-07-07)

**Status (2026-07-16): CLOSED NO-GO for the combined SCFA + full-token FFN proposal.**
The compact GQA implementation and its launch optimizations are retained; reversible FFN support
remains opt-in and disabled by default. See §8 for the measured gate record and disposition.

**Goal:** Add the missing feed-forward capacity to the CHIRON flagship as a **reversible FFN
shear**, and pay for it — parameters *and* VRAM — by shrinking the attention K/V projections with
**Grouped-Query Attention (GQA)**, so the upgrade is **memory-neutral** under the 16 GB ceiling
(flagship runs at 14.71/15.56 GB, ~0.85 GB free).

**Why:** The architecture audit found the flagship block has **zero FFN** (0 MLP/SwiGLU matches in
`gpu_chiron.cu`) — every layer's only per-position transform is ReLN + the SCFA softmax; there is no
gated nonlinear channel mixer, which interpretability work (FFN = key-value memories) associates with
much of a transformer's capacity. Separately, **GQA is not plumbed** in the SCFA path
(`gpu_chiron.cu:3049` hardcodes `nHeads == nKVHeads`), so K/V projections carry full 16-head width
that GQA shows is largely redundant. This plan converts that redundancy into the missing FFN.

**Non-goals / deferred (sequenced backlog):** over-training past Chinchilla (memory-free, separate),
learned-low-rank/hybrid-local attention, generation-aware fine-tune, larger vocab (costs VRAM —
rejected under the constraint), the matched standard-transformer external baseline (companion).

---

## 1. Mechanism

### 1.1 The reversible FFN shear

Per layer, insert **one** new operation immediately after the existing attention shear, mirroring it
exactly (read the representation `q`, write the momentum bus `p`, leave `q` untouched):

```
p += SCFA(q)            # existing attention shear  (gpu_chiron.cu chiron_attention_shear_*)
p += FFN(q)             # NEW reversible FFN shear
[WhiSC rotation on (q,p); ReLN on q]      # unchanged
```

`FFN` is a **SwiGLU bottleneck**:
```
h_gate = SiLU(q · W_gate)         # [T,m]·[m,H] → [T,H]
h_up   =        q · W_up          # [T,m]·[m,H] → [T,H]
FFN(q) = (h_gate ⊙ h_up) · W_down # [T,H]·[H,m] → [T,m]
```
with per-layer weights `W_gate,W_up ∈ ℝ^{m×H}`, `W_down ∈ ℝ^{H×m}`, and **H = m = 2048** (the width
that makes it exactly memory-neutral vs nKV=4 GQA — see §2).

### 1.2 Why `p += FFN(q)` and not `q += FFN(p)`

- It is **structurally identical to the proven attention shear** `p += SCFA(q)`: both read the
  incoming normalized `q` and add to `p`. Lowest-risk insertion; the FFN reads the actual
  representation (a normal FFN input), not the momentum bus.
- Reversibility is trivially clean: the shear does not modify `q`, so `FFN(q)` is exactly recomputable
  in the inverse walk (§1.4).

### 1.3 E0 identity via zero-init `W_down` (NOT a ReZero gate)

`W_down` is **initialized to zero** ⇒ `FFN(q) ≡ 0` at init ⇒ the augmented model is **bit-identical**
to the current flagship at step 0 (the E0 gate). The FFN then grows organically as `W_down` learns.

**This deliberately avoids a ReZero scalar gate.** The OBSD arc ([[obsd_per_layer_coupling_nogo]])
died precisely because its ReZero gate grew **unbounded** → ‖g‖ inflation → grad-clip throttled the
effective LR → the mechanism missed late-training acceleration. Zero-init-`W_down` (Fixup / zero-init
residual) gives the same identity-at-init property with **no unbounded scalar to run away**.

### 1.4 Reversibility (O(1) activation memory preserved)

The block map stays unit-triangular / exactly invertible. Inverse walk per layer (reverse order):
```
undo ReLN(q)                       # existing (uses stored μ,σ)
undo WhiSC rotation on (q,p)        # existing
p -= FFN(q)                        # NEW: q is now recovered & was untouched by the shear ⇒ exact
p -= SCFA(q)                       # existing
```
Because `q` is identical before/after both shears, `FFN(q)` (and `SCFA(q)`) are recomputed exactly
during the backward inverse walk — **no FFN activations are stored** (recomputed like everything else,
per the `chiron_rot_backward_invwalk` pattern, `gpu_chiron.cu:2072-2110`). Backward:
`dp_in = dp_out`; `dq += J_FFN(q)^T · dp_out`; `dW_{gate,up,down}` via standard SwiGLU backward on the
recomputed hidden.

---

## 2. Memory-neutrality (the whole point)

Marginal cost per parameter in the flagship config (`--bf16-weights --bf16-grads --int8-adam`):
**BF16 weight (2 B) + BF16 grad (2 B) + int8 Adam m+v (2 B) = 6 B/param.** (Corrects the audit's
FP32-grad line: the flagship log shows `bf16_grads=1`.)

| | Params (per layer × L=24) | ×6 B |
|---|---|---|
| **GQA nKV=4 frees** (Wk,Wv: `m×dModel`→`m×dModelKV`, dModelKV=nKV·dH=1024) | `2·m·(4096−1024)·24` = **302.0M** | **−1.81 GB** |
| **SwiGLU FFN H=2048 costs** (`3·m·H·L`) | `3·2048·2048·24` = **302.0M** | **+1.81 GB** |
| **Net parameter memory Δ** | **0** | **0.00 GB** |

Activations: GQA shrinks `sK,sV` from `[T×4096]`→`[T×1024]` (**~−400 MB**); the FFN adds one reusable
`[T×H]` scratch (**~+134–268 MB**). **Net activation Δ ≈ neutral-to-negative.** The upgrade fits inside
the existing 14.71 GB with margin to spare.

**Knobs if we want a wider FFN:** nKV=2 frees 352M → H≈2389; nKV=1 (MQA) frees 378M → H≈2560. The
0.85 GB true headroom independently funds ~+141M params (≈ +H·0.5×). Default plan uses **nKV=4, H=2048**
(exact match, most conservative on attention quality).

---

## 3. Compute / wall-clock expectation (be honest)

The FFN adds 3 GEMMs/layer (~206 GFLOP fwd/layer at H=2048), partly offset by GQA's smaller Wk,Wv.
Expect **−20 % to −35 % throughput pre-optimization**, then a perf pass (BF16-TC GEMMs via the
existing `sgemm_rowmajor_bf16` wrappers `gpu_blas.cu`, fused SiLU⊙up, `register_fast16bf_constant`
weight mirrors) targeting **<10 % net wall** — the same arc every prior mechanism ran (WhiSC −9.7 %,
PIED −1.87 %, PACT −6.8 %). Perf is an E2 concern, gated after correctness.

---

## 4. Phased gate plan (cheap-gate discipline — nothing ships without passing each)

Each mechanism is **default-off behind a flag** (`--gqa-kv-heads N`, `--ffn-shear`), and the two halves
are **validated independently** so a GQA regression vs an FFN regression is isolable.

- **Phase A — GQA-enable SCFA** (funding side, ~quality-neutral, frees memory)
  - A/E0: `--gqa-kv-heads 16` bit-identical to today (no-op path).
  - A/E1: units — grouped K/V projection + head-expansion in the tiled core; fwd/bwd/inverse parity vs CPU ref; grad finite-diff.
  - A/E2: numerics + VRAM — confirm ~1.8 GB freed, ~400 MB activation freed; perf delta.
  - A/E3: matched 2500-step gate (seed 1337, FineWeb) `nKV=4` vs `nKV=16` — expect ≤ +0.02 nat (near-neutral) and the memory drop. **Kill if GQA alone regresses > +0.05 nat.**

- **Phase B — reversible FFN shear** (capacity side, spends the freed budget)
  - B/E0: `--ffn-shear` with `W_down=0` bit-identical to today.
  - B/E1: units — CPU ref `chiron_ffn_shear_cpu` in `transformer_chiron_ops.h`; GPU fwd/inverse roundtrip < 1e-4; backward finite-diff on W_gate/W_up/W_down (pattern: `chiron-test.cpp:1610-1956`).
  - B/E2: numerics + perf pass (target <10 % net wall vs Phase-A binary).
  - B/E3: matched 2500-step gate — FFN-on (funded by nKV=4 GQA) vs the Phase-A GQA-only binary. **GO if ≤ −0.02 nat with 0 grad-skips; KILL if regresses or destabilizes.**

- **Phase C — combined E4 (30k) ship gate**
  - Full recipe (PIED + GQA nKV=4 + FFN H=2048) vs the current flagship, matched steps/seed/data,
    **memory-neutral confirmed live** (VRAM ≤ flagship). Ship iff wide-32 val beats flagship beyond
    era-drift (~0.1 nat) at ≤ flagship VRAM. Multi-seed remains an owner call (lineage precedent).

---

## 5. Implementation tasks (bite-sized, with audit anchors)

> Repos: kernels/tests in `glades-ml` (`~/dev/glades-ml`); wiring in `glades-trainer`
> (`~/dev/glades-trainer/trainer/chiron_main.cpp`). Rebuild order: `make install` in glades-ml **then**
> `bash build.sh` in glades-trainer (trainer links glades CUDA **statically**).

### Phase A — GQA in SCFA

- **A1. Thread `nKVHeads` through the SCFA kernels.** Add an `nKVHeads` param to
  `chiron_attention_shear_*_tiled()` (`gpu_chiron.cu:3049+`, currently asserts `nHeads==nKVHeads`);
  size Wk/Wv projections to `dModelKV = nKVHeads·dH`; expand K/V across query-head groups in the
  `[k×k]` flash core. TDD: extend `chiron-test.cpp` with a grouped-vs-full parity case (nKV=nH must
  reproduce the current path bit-for-bit).
- **A2. Wire `--gqa-kv-heads N`** in `chiron_main.cpp` (parse → `TrainingConfig.nKVHeadsOverride`,
  currently unread in the chiron path — audit §2); pass `nKVHeads` to every
  `chiron_attention_shear_*` call site (`chiron_main.cpp:13727, 15671, …`); shrink Wk/Wv allocation +
  `paramCount()` (`chiron_main.cpp:3734`).
- **A3. Checkpoint compat.** GQA changes Wk/Wv shapes; bump the reader to accept `dModelKV` (store
  `nKVHeads` in the header/flags; no new EOF-tail section needed — Wk/Wv live in the per-layer weights
  blob, `chiron_checkpoint` §4). Verify a fresh nKV=4 checkpoint round-trips.
- **A4. Gates A/E0–E3** (§4).

### Phase B — reversible FFN shear

- **B1. Kernels** (`gpu_chiron.cu` + `.h`): `chiron_ffn_shear_forward` (`p += FFN(q)`),
  `chiron_ffn_shear_backward_invwalk` (recompute `FFN(q)`, accumulate dW_*, feed dq/dp) — mirror
  `chiron_rot_backward_invwalk` (`gpu_chiron.cu:2072-2110`); reuse `sgemm_rowmajor_bf16` (`gpu_blas.cu`)
  for the 3 GEMMs and the SiLU kernels in `gpu_atcd.cu`. CPU ref in `transformer_chiron_ops.h`.
- **B2. Per-layer weights** in `chiron_main.cpp` — mirror the `rot_phi` plumbing
  (`chiron_main.cpp:4539-4557` alloc; `addAdam` `:3740-3810`; grad-norm `:15950`): declare
  `FFN_gate/up/down[L]` + grads + Adam state; **`W_down` zero-init, `W_gate/W_up` scaled-normal**;
  insert the forward shear right after the attention shear in the layer loop; accumulate dW in backward.
- **B3. Checkpoint section** (`chiron_checkpoint.{h,cpp}`): add `CKPT_BIT_FFN = 2048`, extend
  `CKPT_KNOWN_BITS_MASK`, write the FFN weights **before the a_drift/rot_phi EOF tail** (audit §4 —
  rot_phi MUST stay last for the `SEEK_END` read); mirror in the reader.
- **B4. Flag** `--ffn-shear` (+ optional `--ffn-hidden N`, default m).
- **B5. Perf pass** (E2) + **Gates B/E0–E3** (§4).

### Phase C — 30k E4 ship gate (§4).

---

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| GQA on the *spectral* SCFA path behaves unlike standard GQA | Phase A isolates it; A/E3 kill-switch at +0.05 nat |
| FFN adds instability (new nonlinear channel) | zero-init `W_down` (identity start), grad-clip 0.5, reln-reanchor already in recipe; B/E1 finite-diff + B/E3 skip-count gate |
| Wall-clock regression | explicit E2 perf pass, <10 % net target (lineage precedent) |
| GQA hurts more than FFN helps (net wash) | the two-phase gates measure each half separately before the combined E4 |
| Checkpoint format break (rot_phi EOF tail) | FFN section inserted strictly before a_drift; round-trip test in A3/B3 |
| GQA kernel surgery in the tiled flash core is the deepest change | land + parity-test A1 in isolation before any training |

## 7. Open questions

- H=2048 (1× SwiGLU) is narrow; is nKV=2/H≈2389 worth the extra attention-quality risk? (decide after A/E3 shows GQA's true cost).
- Should the FFN read `q` pre- or post-WhiSC-rotation? (plan: pre-rotation, same incoming `q` as attention — simplest reversibility).
- Multi-seed for the ship (lineage ships single-seed; owner call at Phase C).

---

## 8. Engineering gate record and disposition (2026-07-16)

The pre-registered engineering experiment used a matched SCFA/int8-Adam shape
`T=2048, m=128, L=2, nH=16, dH=16`; the treatment used `nKV=4, H=m=128`.
All measurements compare steady-state medians from the same executable, GPU, seed, data, and
configuration.

| Gate | Result | Evidence |
|---|---|---|
| E0 legacy compatibility | **PASS** | `nKV=nH` uses the historical tiled path and is byte-identical; zero-`W_down` CPU and CUDA FFN shears are byte-identical to FFN-off. |
| E1 correctness | **PASS** | Compact GQA matched expanded-KV forward and backward references (`dK` max error `2.328e-10`, `dV` `1.490e-08`); FFN inverse, finite differences, and production BF16 parity passed. |
| E2 persistent/live memory | **PASS** | Baseline and combined arms have equal changed-matrix parameter counts (`8m²` per layer). Post-allocation VRAM was `1.10 GiB` baseline versus `1.09 GiB` combined; a 10 ms process sampler observed `326 MiB` versus `320 MiB` live peaks. |
| E2 SCFA wall time | **FAIL** | Baseline `2,081,114 tok/s`; GQA-only `2,075,935 tok/s` (`0.998×`); combined `1,694,511 tok/s` (`0.814×`). Combined wall was `1.228×` baseline, missing the `≤1.10×` bar. |
| Baseline/compatibility | **PASS** | Finite loss and gradients, zero reported skips, CHRF v4 save/resume, serving inference, focused tests, and the full CHIRON suite passed. |

The GQA performance defect was fixed: replacing serial per-head cuBLAS calls with grouped
pointer-array GEMMs and a compact deterministic `dK/dV` reduction improved GQA-only throughput from
about `0.60×` to `0.998×` baseline. The remaining miss belongs to the mechanism, not compact GQA:
SCFA projects attention at compressed length `k=T/16`, while the proposed SwiGLU executes three
matrix families over all `T` tokens during forward/inverse/backward. Compact K/V therefore funds the
FFN's persistent memory but does not fund its full-token compute.

A dense-attention control reached `110,972 tok/s` combined versus `108,872 tok/s` baseline
(`1.019×` throughput), confirming that the failed bar is specific to pairing a full-token FFN with
compressed SCFA. The experiment was not reclassified using that control because the registered
hypothesis explicitly targeted SCFA.

**Disposition:**

- Retain compact GQA across CUDA, SCFA, checkpoints, trainer state, and serving.
- Retain grouped GQA GEMM dispatch and compact gradient reduction.
- Keep reversible FFN/checkpoint compatibility behind `--ffn-hidden`; default `0` remains FFN-off.
- Do not enable the combined SCFA + `H=m` FFN in the flagship recipe and do not launch E3/E4.
- Reopen only under a separately pre-registered hypothesis, such as compressed-domain or less-frequent
  FFN placement, a separately quality-gated fused/lower-precision kernel, or an explicitly revised
  wall-time bar.
