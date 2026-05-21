# Unified Flagship + CHIRON Architecture (HELIX)

**Date:** 2026-05-09
**Status:** design (not yet implemented)
**Author:** generated alongside flagship-vs-CHIRON 30-min comparison run
**Premise:** "ignore engineering time and complexity — combine flagship and
CHIRON to maximize memory, throughput, and NLL on a single 16 GB GPU."

This document defines **HELIX**, a single-architecture single-trainer design
that subsumes:

* the flagship `glades_pile_train` paradigm stack (#6 LOCAL-ATTN, #74
  PHOENIX-1BIT FFN, #76 MLA, #78 ATTENTION-SINK, MP/BF16 weights);
* the CHIRON paradigm stack (#11 MFIO, #28 FACE, #38 SLC, #39 RLG, #40 SAS,
  reversible symplectic-flow shear + reversible LN, BF16 + int8 Adam, Kahan-v);
* the optimizer-side compressions (#41 ASTRA-style stateless variants,
  reserved as ablation knobs).

The point is not "stack flags," it is to find the *one* architecture that
drops every paradigm into its highest-leverage slot once compatibility is
designed in from the start, rather than retrofit.

---

## 0. Design constraints

| Constraint | Source | Why it dominates |
|---|---|---|
| Single 16 GB GPU (RTX 4080 SUPER) | user brief 2026-05-09 | "we really just care about the GPU version" |
| Bit-exact NLL preservation where possible | iter-193 paradigm constraint | quality is the metric, not raw throughput |
| Deterministic per-seed start | `glades::rng::Engine` | reproducibility for Gate-0 probes |
| BPE-32k corpus, T ≤ 4096 default | flagship trainer + CHIRON 1.84B preset | dataloading already wired |
| Must run inside existing `glades-trainer` build | iteration speed | avoid greenfield |

The "ignore engineering time" license unlocks the structural changes that
are blocked today by ABI fragility (header-mirror sync) and library
boundaries between glades-ml and glades-trainer; HELIX assumes both repos
are co-built from a single CMake project.

---

## 1. The macro architecture

HELIX is a **block-typed** transformer. Every layer is one of three block
types, all sharing the same residual stream `(q, p) ∈ ℝ^{T × m} × ℝ^{T × m}`
(paired state, CHIRON's symplectic convention). The block type is chosen
per-layer at architecture-spec time, not learned.

```
                Embedding (FACE-managed)
                       │
                       ▼
         L₁  HELIX-block (q,p) → (q',p')
                       │
                       ▼
         L₂  HELIX-block (q,p) → (q',p')
                       ⋮
                       ▼
         L_L HELIX-block (q,p) → (q',p')
                       │
                       ▼
                 Final RMSNorm
                       │
                       ▼
              Output projection
                       │
                       ▼
              Vocab logits → CE loss
```

### 1.1 Block types

| Type | Compute pattern | Use case | Backward |
|---|---|---|---|
| **R** (Reversible) | CHIRON shear: `p += Y(q; Wq, Wk, Wv, Wo)` ; q → reln(q; γ, β) | majority of the stack (depth-saturated) | inverse walk: O(1) activation memory |
| **S** (Standard) | pre-LN block: `q' = q + Attn(LN(q))`; `p' = q' + FFN(LN(q'))` | first/last 2 layers (numerical edge guard) | full saved activations: O(L_S) memory |
| **M** (MoE) | shared backbone + per-expert LoRA adapter, deterministic routing on q | mid-stack (paradigm #53 MOSAIC-MOE inheritance) | inverse walk over backbone, materialize active experts |

A 53-layer HELIX-1.84B uses pattern `S R R R … R R S` with 51 R-blocks plus
2 S-blocks at the boundaries. M-blocks are inserted only when the
configuration explicitly requests MoE (`--experts E --topk k`); a vanilla
HELIX has no M-blocks.

### 1.2 Why pair-state for both R and S blocks

Standard transformer blocks naturally compute `q' = q + δ(q)` — single
state. CHIRON's reversibility *requires* paired (q, p). Forcing all blocks
to share the (q, p) interface lets us:

1. Mix R-blocks and S-blocks without conversion glue.
2. Unify the optimizer: every block's input/output is the same shape.
3. Make the "S-blocks at edges" choice a tuning knob, not a major
   refactor.

For an S-block, p is initialized at block entry as a copy of q, and the
output `p'` becomes the new q (the block is otherwise a standard
transformer layer — p was just a memory buffer). The cost is a single extra
T×m tensor allocation, paid only at S-block boundaries (constant in L).

---

## 2. Attention inside R/M-blocks

The shear `Y(q; Wq, Wk, Wv, Wo)` is computed by **MLA-shear** — MLA's
low-rank K/V projection applied inside CHIRON's symplectic shear:

```
c   = q · W_DKV               # [T, d_c], d_c = 128
K   = c · W_UK                # [T, d_KV]
V   = c · W_UV                # [T, d_KV]
Q   = q · W_q                 # [T, d_model]
S   = softmax(Q · Kᵀ / √d_h + sinks + local-mask)
out = S · V · W_o             # [T, d_model]
p  += out
```

The local-attention mask + 4 attention sinks (paradigms #6 + #78) is
applied *inside the shear's softmax* — same kernel as the flagship today,
just dispatched through CHIRON's per-block scratch layout. This is the
single biggest design synergy: MLA's low-rank K/V cache is exactly what
makes the shear's K/V reconstruction cheap on the inverse walk.

**Reversibility theorem (informal).** The shear's bijectivity proof in the
CHIRON architecture doc requires only that `Y(q; ·)` be a function of q
(not p). The MLA factoring `K = q·W_DKV·W_UK` keeps Y a function of q
alone, so MLA composes with reversibility *for free* — no proof to extend.

**Inverse-walk cost.** Recovering K, V on backward needs `c = q · W_DKV`
(d_model × 128 GEMM) instead of K, V directly (d_model × d_KV GEMM, with
d_KV typically equal to d_model). Asymptotic backward attention compute
drops by factor `d_KV / d_c = 8×` at d_c=128, d_KV=1024 — *additional*
to the regular forward win MLA already provides.

---

## 3. FFN inside R/S/M-blocks

The FFN is **PHOENIX-1.58BIT (ternary)** for R-blocks at the deep middle,
**PHOENIX-1BIT (binary)** for the deep middle's deepest sub-stack only
when `--phoenix 1bit` is set, and **dense BF16** for S-blocks.

| Block range | FFN precision | Rationale |
|---|---|---|
| S-blocks (edges) | BF16 dense (today's `--mp` path) | edge layers carry most of the embedding-channel signal; quantization noise here costs disproportionate NLL |
| R-blocks (mid) | ternary {-1, 0, +1} (paradigm #47 PHOENIX-1.58BIT) | 5× memory + 2× compute vs BF16 with 1-2% NLL loss budget |
| R-blocks (deep mid only) | binary {-1, +1} (paradigm #48 PHOENIX-1BIT) | optional `--phoenix-1bit-deep-half` for the inner half-depth; another 2× memory at additional 0.10-0.15 nat NLL cost |
| M-blocks shared backbone | BF16 dense | LoRA adapters are tiny; backbone matters most |
| M-blocks expert adapters | LoRA rank-r=4 in BF16 | adapter parameters are ~0.5% of dense FFN, no quantization needed |

Crucially, `--binary-ffn` (today's flagship flag) maps to "PHOENIX-1BIT on
all R-blocks." The current flagship's straight-through estimator backward
is preserved because R-blocks reconstruct the FFN forward before backward,
and the STE acts on the float master copy that's kept on the host
(downloaded only on save).

---

## 4. Optimizer state (the biggest single VRAM win)

The optimizer is **AdamW + paradigm stack**, with state compressions chosen
per-tensor based on its training-stability profile:

| Tensor class | m, v storage | Paradigm | Per-param state bytes |
|---|---|---|---|
| Embedding `tokE` | FACE Adafactor (`zn̄`, `dn̄`, `q̂`, `gF̄`) | #28 FACE | ~0.06 (508× compression vs dense Adam at vocab=32k, d=1024) |
| LM head `WOut` | FACE-symmetric (output side) | #28 FACE | ~0.06 |
| Attention `Wq, Wo` | int8 m + uint8 v + Kahan-c | #11 MFIO + #41-A Kahan-v | 1.5 + 0.5 = 2.0 |
| Attention `W_DKV, W_UK, W_UV` (MLA) | int8 m + uint8 v | #11 MFIO | 1.5 |
| FFN `W1, W2` (R-block, ternary/binary) | int8 m + uint8 v | #11 MFIO | 1.5 |
| FFN `W1, W2` (S-block, BF16 dense) | bf16 m + bf16 v + Kahan-c | #41-A Kahan-v | 4 + 2 = 6 |
| LN gamma/beta | FP32 m + v | (no compression — tiny) | 8 |
| MoE LoRA adapters | bf16 m + bf16 v | (per-expert tiny) | 4 |

For HELIX-1.84B (1840M params, with embedding ≈ 65M and S-block FFN ≈ 200M
and R-block FFN ≈ 1300M and attention ≈ 275M):

| Component | Old (FP32 Adam) | HELIX | Saving |
|---|---|---|---|
| Embedding state | 65M × 8 = 520 MB | 65M × 0.06 = 4 MB | 130× |
| S-block FFN state | 200M × 8 = 1.60 GB | 200M × 6 = 1.20 GB | 1.33× |
| R-block FFN state | 1300M × 8 = 10.4 GB | 1300M × 1.5 = 1.95 GB | 5.3× |
| Attention state | 275M × 8 = 2.20 GB | 275M × 2.0 = 0.55 GB | 4× |
| **Total Adam state** | **14.7 GB** | **3.71 GB** | **4×** |

That alone fits 1.84B-equivalent Adam on a 16 GB GPU. Add weights (BF16
≈ 3.6 GB) plus activations (CHIRON O(1) ≈ 0.5 GB at T=4096 with paired
state) plus working buffers (~1 GB). **Total 8.8 GB / 16 GB.** Headroom
remains for SCFA/ORION reserved paradigms or longer T.

---

## 5. The training loop

```
for stage in COSMIC-stages [Foundation, Reasoning, Refinement]:
  for step in stage.steps:
    1. SLC: pick T_step from stage's sequence-length curriculum
    2. SAS: sample α_step from stage's attention-skip schedule
    3. RLG: insert/grow R-block layer if step matches RLG schedule
    4. Forward pass:
         - tokens → embedding (FACE)
         - per-layer: dispatch to R/S/M block kernel
         - final LN + output projection → CE loss + (#56 distill loss)
    5. Backward pass:
         - inverse walk over R/M-blocks (no stored activations)
         - full backward over S-blocks (saved activations)
         - SPAREC sparsity threshold on FFN backward (paradigm #35)
    6. Optimizer step:
         - per-tensor: dispatch to int8/bf16/Adafactor/FACE/MFIO update
         - GLOBAL-NORM grad clip ‖g‖ ≤ 1.0 (Kahan-compensated reduction)
         - ICARUS 4th-order symplectic update for R-block weights
           (paradigm #49, optional via --icarus)
    7. NIMBUS: H2D async to overlap next-step forward with this step's
       Adam update on host CPU (paradigm #52, optional via --nimbus)
    8. ATLAS-COMPILE: per-shape autotuned kernel cache + CUDA-Graph replay
       for the steady-state (paradigm #51, optional via --cuda-graphs)
```

### 5.1 Curricula (per-stage)

| Curriculum | Stage 1 (Foundation) | Stage 2 (Reasoning) | Stage 3 (Refinement) |
|---|---|---|---|
| SLC sequence length | 256 → 1024 over 60% steps | 1024 fixed | 1024 → 4096 over 20% steps |
| RLG layer count | 16 → 32 over 50% steps | 32 → 53 at step 25% | 53 fixed |
| SAS α | 0.3 → 0.5 over stage | 0.5 → 0.7 | 0.7 → 0.9 |
| MoE k (active experts) | k=2 fixed | k=2 fixed | k=2 → k=1 at step 80% |
| FFN precision | BF16 | ternary | binary inner-half |
| LR | warmup 1000 steps → 1e-3 cosine to 5e-4 | 5e-4 cosine to 2e-4 | 2e-4 cosine to 5e-5 |

Stage-1-to-2 and 2-to-3 transitions warm-restart the optimizer state
(the FACE/MFIO/Kahan compressed state is RLE-zero-prefix on freshly
allocated tensors, no precision loss).

### 5.2 Regularization and sampling

* **Embedding dropout**: 0.0 (off — FACE-managed embedding already
  per-token rate-controlled).
* **Residual dropout**: 0.0 on R-blocks (would break inverse walk),
  0.05 on S-blocks.
* **Stochastic depth**: SAS provides this on attention; FFN gets nothing
  (binary/ternary already injects substantial noise).
* **Sampled-softmax**: 64 negatives per target during pretraining
  (existing flagship default); switch to full softmax for stage 3
  refinement if VRAM headroom allows.

---

## 6. Memory budget at 1.84B on 16 GB

| Bucket | Size at 1.84B + T=4096 | Notes |
|---|---|---|
| BF16 weights (master) | 3.68 GB | tied embeddings save another 130 MB |
| Adam state (compressed per §4) | 3.71 GB | dominated by R-block FFN |
| Activations (R-blocks O(1)) | 0.50 GB | paired (q, p) per step |
| Activations (S-blocks ~4 layers) | 0.40 GB | full saved |
| Attention scratch | 0.30 GB | local W=256 + 4 sinks bounded |
| FFN scratch | 0.40 GB | binary/ternary GEMM workspaces |
| FACE embedding state | 0.004 GB | rounding error on this scale |
| Token cache (data loader) | 0.50 GB | not VRAM, host RAM |
| Misc working / cuBLAS / curand | 0.30 GB | |
| **Total VRAM** | **~9.3 GB** | **6.7 GB headroom for SCFA / ORION / longer T** |

The headroom is the design margin for adding paradigms #42 SCFA, #43
ORION, and #44 MELT later without redesigning the budget.

---

## 7. Training-side paradigms layered on top

These run *outside* the architecture but on the same checkpoint stream:

| Paradigm | Where it lives | Marginal cost |
|---|---|---|
| #56 DISTILL-FORWARD | teacher pass at start of each step (KL term in loss) | +2% step cost |
| #57 SCROLL active learning | importance sampling in dataloader | +0.5F (teacher already amortized) |
| #58 METAGEN synthetic data | offline corpus generation, then standard pretraining | one-shot 4 weeks |
| #59 PRM-CHIRON process reward | auxiliary head + 0.1 weight on PRM loss | +1% step cost |
| #60 TOOL-LLM | special tokens in vocab + tool-use loss masking | data-side |
| #61 COSMIC stages | curriculum schedule above (§5.1) | scheduling, no runtime cost |
| #62 AGENT-CHIRON multi-step | rollout-time data generation + REINFORCE on success | offline rollouts |
| #63 META-LEARN-CHIRON | class-conditional EMA on grads | +5% step cost |
| #64 MEMORY-CHIRON | retrieval bank + RETRO cross-attention | +0.5F retrieval lookup |
| #65 WORLD-MODEL retrieval | bank-row schema extension | bank-side |

These are independent additions: HELIX architecture doesn't constrain
which subset is enabled.

---

## 8. What HELIX gives up vs flagship-only or CHIRON-only

**Loss vs flagship (not net wins everywhere):**

* S-blocks at the edges still pay O(L_S) activation memory — typically
  4 layers × T × m × 4 bytes ≈ 0.4 GB at L=53, T=4096, m=2048. Pure
  flagship has this everywhere; HELIX confines it to 4 layers.
* Standard transformer's debug-friendly intermediate activations are
  available only at S-block boundaries. R-block activations exist only
  during inverse walks (transient).
* MLA-shear forward is slightly more expensive than CHIRON's vanilla
  shear (extra projection layer through W_DKV).

**Loss vs CHIRON (also not free):**

* HELIX adds binary/ternary FFN to R-blocks → extra 0.10-0.30 nat NLL
  compared to CHIRON's BF16 dense FFN; recovered partly by the wider
  reachable param count.
* Local-attention + sinks on R-block shears means full attention is no
  longer recoverable from the inverse walk (you'd need to re-run the
  attention with the same mask). Adds ~0.05 nat NLL bias if the corpus
  has long-range structure that the W=256 + 4-sinks scheme misses.
* MoE M-blocks introduce routing that must match the inverse walk —
  the routing is forced deterministic on q (not learned) to preserve
  bijectivity. This is weaker than learned MoE routing.

**What no flag combination gets you today:**

The flagship and CHIRON paths today are *mutually exclusive* — they live
in different binaries, with different state layouts and different
optimizers. You cannot use FACE on flagship's embeddings, you cannot use
MLA on CHIRON's attention shear, and you cannot mix int8 Adam with
flagship's FP32 Adam path. Building HELIX is the act of teaching the
system that these are different *kernel choices for the same network*,
not different networks.

---

## 9. Path-of-least-resistance subset (if engineering time matters)

If the user does want to spend less than infinite engineering time, the
ranked subset that captures most of the value is:

1. **GPU-side init (already implemented in this session)** — eliminates
   the 10-min CPU init wall for any size.
2. **int8 Adam in flagship** — port `adam_update_int8_state` from
   CHIRON's gpu_kernels.cu to flagship's optimizer dispatch. Lifts
   flagship's GPU ceiling from ~213M to ~700M-1B in one go. Validated
   kernel exists (per `MEMORY.md` chiron_architecture entry, line 26).
3. **FACE on flagship embedding** — paradigm #28's Adafactor path applies
   verbatim to flagship's `tokE` table. Saves ~125 MB Adam state at
   d=1024 vocab=32k, scales linearly with vocab × d.
4. **SLC curriculum on flagship** — pure training-loop change, 1.5×
   wall-clock speedup demonstrated. ~200 LOC.
5. **Reversible R-blocks for the deep middle** — the largest structural
   change, 4-6 weeks. Gets you past 1.5B without needing FP32 Adam
   state on the deep stack.
6. **Everything else** in §7 is data-side or training-side and orthogonal
   to the architecture.

Items 1-4 alone close most of the param-gap to CHIRON without committing
to the full HELIX architecture.

---

## 10. Open questions and Gate-0 candidates

1. **MLA-shear bijectivity (Theorem 1)** — formally extend CHIRON's
   shear-bijectivity proof to the MLA factoring (likely trivial:
   composition of linear maps preserves bijectivity of the outer
   transformation in q). Gate-0: numerical bit-exact inverse-walk test
   on a small 4-layer MLA-shear stack.
2. **Local-attention recovery on inverse walk** — the W=256 + 4-sinks
   mask is deterministic given the token positions, so inverse walk
   reproduces exactly the same attention pattern. Gate-0: equality test
   between forward attention output and inverse-walk reconstruction.
3. **Binary FFN STE inside reversible block** — STE drops gradient
   information that the inverse walk wants to reconstruct. The float
   master is on host; pulling back into device for backward each step
   adds PCIe traffic. Gate-0: end-to-end NLL parity vs CHIRON-style
   dense FFN at 100M scale across 1000 steps.
4. **Mixed-precision Adam state on R-block FFN backward** — int8 Adam
   on a binary-FFN R-block produces gradient that's already ternary in
   the relevant subspace. The MFIO compression may be lossless here.
   Gate-0: gradient-norm parity vs FP32 Adam at the same step.

Answers to all four are likely positive (composition of well-studied
mechanisms). The point of staging Gate-0 is: build kernels first, then
validate at minimum scale before committing to architecture-wide changes.

---

## 11. Comparison summary

| Metric | flagship today (213M actual) | CHIRON today (1.84B) | HELIX target (1.84B) |
|---|---|---|---|
| GPU memory | ~13 GB | ~15.3 GB | ~9.3 GB (6.7 GB headroom) |
| Throughput (tok/s) | ~1859 | ~1665 | projected 2200-2800 (S-blocks at edges + R-blocks for the rest, with ATLAS-COMPILE + NIMBUS) |
| Tokens·params/s | 4.0×10¹¹ | 3.1×10¹² | projected 5.0×10¹² (same as CHIRON × 1.6 throughput) |
| Adam state bytes/param | 8 (FP32) | ~1.5 (int8) | ~2.0 (mixed per-tensor) |
| Activation memory | O(L) | O(1) | O(L_S=4) ≈ O(1) for L≥16 |
| NLL preservation vs CE-only | exact | exact | within 0.10-0.30 nat (binary FFN cost) |
| Single-binary | yes | yes | yes (replaces both) |

HELIX is the design that lives in the upper-right of the
"throughput vs ceiling" frontier — neither pure flagship nor pure CHIRON
gets there alone.
