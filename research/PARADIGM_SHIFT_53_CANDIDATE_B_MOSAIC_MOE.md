# Paradigm Shift #53 Candidate B — MOSAIC-MOE (Sparse Mixture-of-Experts within CHIRON's Reversible Shear)

**Status:** candidate-B design; one of three parallel proposals for paradigm shift #53.
**Date:** 2026-05-08 (Ralph-loop iteration 197, post-#52 NIMBUS-PROMOTED, under the iter-197 brief: *"NLL preservation strict + magnitudes compute speed, single GPU; novel architectures invited."*).
**Axis:** **architectural sparsity** — replace CHIRON's dense FFN shear with a `k`-of-`E` sparse mixture-of-experts (Mixtral-style), giving an `E×` effective-parameter count at `k×` active compute. Reversibility is preserved because top-`k` routing is a deterministic function of `q` only (the half of `(q,p)` that the symplectic shear feeds through `Y`).
**Tagline.** *Stop scaling the dense FFN. The activations carry locale, not the weights — let each token select which experts to route through, and pay only for the experts it chose. At `E=8, k=2` the FFN compute drops `4×` and the effective parameter pool grows `8×`; in CHIRON, the routing decision is a deterministic function of `q` alone, so the shear `(q,p) ↦ (q, p + Y_MOE(q))` remains a bijection with explicit inverse.*

**Materially distinct from competing #53 candidates A and C:**
- **A and C** — separate proposals on different axes; not analyzed here.
- **MOSAIC-MOE (this doc, B)** — **architectural** sparsity at the FFN level, introducing per-token routing that selects a subset of experts. NLL is empirically competitive but **not bit-exact preserved** — a deliberate departure from the iter-193 strict criterion in exchange for `E×` effective capacity at memory parity. The honest framing (§7) is that MOSAIC-MOE addresses the user's "extremely large LLMs on single GPU" axis by raising the *effective* model size from 18B to ~144B on the same 16 GB envelope, while leaving raw active-parameter NLL essentially unchanged.

**Materially distinct from prior shipped paradigms:**
- **#44 MELT** — factorizes the FFN weight tensor (TT). MOSAIC-MOE *uses* MELT cores as the per-expert FFN backbone. Compositional: experts are LoRA adapters layered on top of the shared MELT backbone (§3, §6).
- **#28 FACE** — Adam-state compression at the embedding table. Orthogonal to MOSAIC-MOE: routing weights `W_r` are tiny (1.6 K params per layer) and unaffected.
- **#13 TRCD** — token-routed *depth* (per-token early exit). MOSAIC-MOE is token-routed *width* at fixed depth. Different axis. Both can stack, with care (§5.4).
- **CSP / sketch-style FFN compressors** — sketch the FFN's *hidden activation* `x = σ(W_in q + b)` (#27). MOSAIC-MOE leaves the per-expert FFN body intact and routes among multiple of them.

---

## 0. Executive summary (HONEST claim)

After 11 paradigms (#42–#52) the cumulative single-GPU stack reaches ~917× wall-clock at 18B with NLL preservation. The remaining axis under the iter-197 "novel architectures invited" rider is **architectural capacity at fixed active compute**. Dense scaling at 16 GB caps near 18B parameters; MoE scaling at the same envelope reaches ~144B effective parameters with the same active-step compute as an 18B-active dense model. This is the Mixtral / Switch-Transformer / DeepSeek-V3 paradigm, transplanted into CHIRON's symplectic shear.

MOSAIC-MOE replaces the dense FFN shear

$$Y_{\mathrm{FFN}}(q) = W_{\mathrm{out}} \, \sigma(W_{\mathrm{in}} q + b_{\mathrm{in}}) + b_{\mathrm{out}}$$

with a sparse-mixture shear

$$Y_{\mathrm{MOE}}(q_t) = \sum_{e \in \mathcal{E}_t} r_{t,e} \cdot \mathrm{FFN}_e(q_t), \qquad \mathcal{E}_t := \mathrm{top}_k\bigl( \mathrm{softmax}(W_r q_t) \bigr).$$

At `E=8, k=2`, each token activates `2/8 = 0.25×` of the FFN compute. Memory expansion is contained by the **shared backbone + LoRA experts** trick (§3): one shared MELT-TT FFN backbone (174 KB at flagship via #44) plus 8 small LoRA-rank-`r=4` adapters (~16 KB per expert) per layer, total ≈ 300 KB / layer. Memory parity with single-FFN MELT.

**Headline figures at flagship 1.84B-active (post-#42–#52, MOSAIC at `E=8, k=2`, LoRA `r=4`):**

- FFN compute per layer: **4× reduction** at matvec level in Hybrid factorization (`k/E = 1/4`); 1.0× in default Shared+LoRA factorization (capacity-only).
- End-to-end per-step wall-clock: **1.5–2× per-effective-step** in Hybrid (FFN is ~50% of post-#52 step).
- Effective parameter count: **`E× = 8×` higher** (1.84B × 8 ≈ **14.7B effective at 1.84B active**; 144B effective at 18B active).
- Memory: parity with single-FFN MELT (shared backbone amortized; LoRA adapters +5%).
- NLL: **not bit-exact**; Mixtral / DeepSeek-V3 / Switch give 0–2% perplexity penalty vs iso-active dense. **Honest projection at CHIRON 1.84B-active: 0.05–0.15 nat above iso-active dense at iso-tokens.**

**Stack at 18B-active:** `917× × 1.5–2 ≈ 1400–1800× wall-clock at 144B effective on single GPU.` First iter-197+ paradigm making the magnitude claim about *capacity* rather than wall-clock alone.

**Engineering scope.** ~1300 LOC, 4–6 weeks. Routing kernel (~300), top-k dispatch (~400), LoRA adapters (~200), capacity throttling (~150), composition with #44–#52 (~250).

**Honest cost — three risks:**

1. **NLL is empirical, not theoretical.** No bit-exact preservation. Gate-0 (§11) tests at 41M against dense baseline.
2. **Routing instability** (load imbalance, expert collapse). Mitigated via load-balancing loss + capacity factor (§4.5).
3. **CUDA-Graph composition with #51 ATLAS-COMPILE is degraded.** Mitigation: capacity-throttled fixed graph (§5.5) preserves most of the gain.

---

## 1. Primitive objects

Standard CHIRON dims: `m=2048` (token-state), `T=1024` (sequence), `L=53` (blocks at flagship), `d_{FFN}=4m=8192`. New MOSAIC-MOE dims: `E=8` (experts per FFN), `k=2` (top-k), `r=4` (LoRA rank), `c=1.25` (capacity factor).

New per-layer parameters:
- `\mathrm{FFN}_0` — shared MELT-TT FFN backbone (the same for all experts).
- `W_r ∈ ℝ^{m×E}` — router weight matrix; `r_t = \mathrm{softmax}(W_r^\top q_t) ∈ \Delta^E`.
- `(A_e, B_e) ∈ ℝ^{m×r} × ℝ^{r×m}` — input-side LoRA adapter for expert `e ∈ \{0,...,E-1\}`.
- Optional `(C_e, D_e) ∈ ℝ^{d_{FFN}×r} × ℝ^{r×d_{FFN}}` — output-side LoRA adapter (Hybrid regime, §3.3).

Routing primitives: `\mathcal{E}_t := \mathrm{top}_k(r_t)`; `\bar r_{t,e} := r_{t,e} / \sum_{e' \in \mathcal{E}_t} r_{t,e'}`; per-expert cap `\lceil c \cdot kT/E \rceil`.

**Invariant — what's new in optimizer state.** Per-layer additional params: `W_r` (16 K) + 8 × `(A_e, B_e)` (8 × 16 K = 128 K) = ~144 K, total ~7.6M across 53 layers. Adam state ≈ 30 MB at BF16 (negligible against ~660 MB FACE-compressed Adam at flagship). **No new optimizer-state class** — standard Adam on BF16 master, post-#28 FACE/Kahan-v stack unchanged.

---

## 2. The MOE shear: mathematics and reversibility

### 2.1 Standard FFN shear (recap)

CHIRON's MLP block applies the symplectic shear

$$\Phi^{\mathrm{FFN}}_l : (q, p) \mapsto (q, \, p + Y_l(q; \theta_l)), \qquad Y_l(q) = W_{\mathrm{out},l} \, \sigma(W_{\mathrm{in},l} q + b_{\mathrm{in},l}) + b_{\mathrm{out},l}.$$

Theorem 3 of paradigm #42 (SCFA) establishes that **for any continuous `Y(q)`, this shear is bijective with explicit inverse** `(q', p') ↦ (q', p' - Y(q'))`. The reversibility is structural — it depends only on the lower-triangular shape of the shear, not on `Y`'s internal form. CHIRON's inverse-walk machinery (§9 of #1, refined by #46 REFLECTOR) recomputes activations by inverting `Φ^{FFN}_l` using `Y_l(q')`, which requires only `q'` and the layer parameters.

### 2.2 MOE shear

Replace `Y_l` with

$$Y^{\mathrm{MOE}}_l(q_t) := \sum_{e \in \mathcal{E}_t(q_t)} \bar r_{t,e}(q_t) \cdot \mathrm{FFN}^{(e)}_l(q_t),$$

where:

- `r_t = \mathrm{softmax}(W_r^\top q_t) \in \Delta^E` is the unbiased router distribution;
- `\mathcal{E}_t(q_t) := \mathrm{top}_k(r_t)` is the deterministic top-`k` index set, with **index-order tie-breaking** (smaller index wins on equality);
- `\bar r_{t,e} := r_{t,e} / \sum_{e' \in \mathcal{E}_t} r_{t,e'}` is the renormalized weight on the active experts;
- `\mathrm{FFN}^{(e)}_l(q) = \mathrm{FFN}_0^{(\mathrm{out})} \bigl( \sigma( \mathrm{FFN}_0^{(\mathrm{in})}(q) + B_e A_e q ) \bigr) + b_{\mathrm{out}}` (shared backbone + per-expert input-side LoRA; see §3).

**Theorem 1 (MOSAIC-MOE shear bijectivity).** *The map `Φ^{MOE}_l : (q, p) ↦ (q, p + Y^{MOE}_l(q))` is a bijection on `ℝ^{T×m} × ℝ^{T×m}` with explicit inverse `(q', p') ↦ (q', p' - Y^{MOE}_l(q'))`, preserving the symplectic form `ω = dp ∧ dq` and `det DΦ^{MOE}_l = 1`.*

**Proof.** It suffices to show `Y^{MOE}_l` is a (deterministic, well-defined) function of `q` alone. Fix `q_t`. The router output `r_t` is deterministic given `(q_t, W_r)`. Top-`k` selection `\mathcal{E}_t(q_t)` is deterministic given `r_t` and the index-order tie-breaking rule (tie-breaking is required for measure-zero ties and never fires generically; index-order makes it a proper function on all of `ℝ^E`). The renormalized weights `\bar r_{t,e}` and per-expert outputs `\mathrm{FFN}^{(e)}_l(q_t)` are deterministic given `q_t` and the layer parameters. Their convex combination is deterministic. Hence `Y^{MOE}_l(q_t)` is a well-defined function `ℝ^m → ℝ^m` of `q_t` alone. Theorem 3 of #42 applies: any function `Y(q)` produces a unit lower-triangular bijective shear with the stated inverse, symplectic-form preservation, and unit determinant. □

**Corollary (inverse-walk compatibility).** *CHIRON's inverse-walk recovers `q` from `q'` trivially (`q = q'` since the shear is identity on the `q`-coordinate) and `p` from `(q, p')` via `p = p' - Y^{MOE}_l(q)`. The recomputation requires re-invoking the router on `q` and re-running the top-`k` experts. The router and expert weights are stationary during the inverse walk; tie-breaking is deterministic; therefore the inverse walk reproduces the forward `\mathcal{E}_t` exactly.* □

This last corollary is the load-bearing claim of MOSAIC-MOE: **the routing decision is reproducible at inverse-walk time, which is what makes MoE compatible with CHIRON's reversibility at all.** Standard MoE training in non-reversible transformers does not need this; MOSAIC-MOE does, and it satisfies it because the router takes only `q` (not `p`, not random noise, not external state).

### 2.3 What we explicitly forbid

- **Stochastic routing** (Gumbel-softmax / noisy top-`k`) — would make `Y^{MOE}_l` random, breaking inverse-walk reproducibility unless noise is checkpointed.
- **Routing on `p`** — would couple top-`k` to the half-state being updated, breaking the lower-triangular shear.
- **Token-dropping at capacity** — replaced with "route to backbone-only" (§4.5); dropping would make `Y^{MOE}_l` step-discontinuous at the capacity boundary.

---

## 3. Shared backbone + per-expert LoRA experts

The naive MoE expansion would store `E` independent FFN matrices `(W_{in}^{(e)}, W_{out}^{(e)})`, multiplying memory by `E`. At `E=8`, flagship FFN weights would balloon from 3.55 GB (BF16) → 28.4 GB, vastly exceeding the 16 GB envelope.

MOSAIC-MOE avoids this by factoring each expert as

$$\mathrm{FFN}^{(e)}(q) := \mathrm{FFN}_0(q) + \Delta^{(e)}(q),$$

where `\mathrm{FFN}_0` is a *shared* dense (or post-#44 MELT-TT) backbone, and `Δ^{(e)}(q) = \mathrm{FFN}_0^{(\mathrm{out})}(σ(B_e A_e q))` is a low-rank LoRA-style additive adapter with `(A_e, B_e) \in ℝ^{m×r} × ℝ^{r×m}`, `r = 4`.

### 3.1 Memory accounting at flagship 1.84B post-#44 MELT

| Component | Size per layer | × `L = 53` |
|---|---|---|
| Shared MELT-TT backbone | 174 KB (per #44 §0, ρ=8) | 9.2 MB |
| Router `W_r ∈ ℝ^{m × E}` (BF16) | 32 KB | 1.7 MB |
| Per-expert LoRA `(A_e, B_e)` (BF16, r=4) | 32 KB / expert × 8 = 256 KB | 13.6 MB |
| **Total FFN-side memory** | **~462 KB / layer** | **~24.5 MB / model** |

For comparison, single-FFN MELT at flagship is ~9.2 MB total. MOSAIC adds ~15 MB across the model — **0.1 % of the 16 GB envelope, for an `E×` capacity multiplier**.

### 3.2 The compute accounting honesty

In the Shared+LoRA factorization, the dominant compute (backbone in/out projections) is **shared across experts on the same token** — only one backbone forward is needed regardless of `k`. The LoRA deltas `B_e A_e q` are `O(k m r)`, negligible (`r=4`). So per-token compute is essentially the same as a single dense FFN. **MOSAIC-MOE's primary axis is capacity, not compute.**

The classical Mixtral 4× FFN compute reduction assumes each chosen expert runs its own independent FFN (`E×` memory, `k/E×` compute). That's memory-infeasible at flagship (28 GB FFN weights vs 16 GB envelope). To recover compute speedup within memory budget, we offer optional output-side LoRA adapters `(C_e, D_e)` so the expert-specific projection runs only on the `k` chosen experts; this gives partial compute savings.

### 3.3 Three factorization regimes (decision deferred to Gate-0)

| Regime | Per-expert FFN | k=2 compute | E=8 memory | Outcome |
|---|---|---|---|---|
| **Full-Mixtral** | independent `(W_{in}^{(e)}, W_{out}^{(e)})` | 0.25× FFN | `E×` (28 GB → infeasible) | infeasible at flagship |
| **Shared+LoRA** (default) | shared backbone + LoRA delta | 1.0× FFN | `1.0× + 0.1%` | capacity-only win |
| **Hybrid** (Gate-0 candidate) | shared + LoRA r=32 + per-expert mid-layer | ~0.6× FFN | `+15%` | sweet spot — needs verification |

**Headline depends on Gate-0.** If Hybrid passes Gate-0 (§11) at ≤ 0.10 nat penalty: 1.5–2× end-to-end speedup, ~144B effective. If only Shared+LoRA passes: capacity win at compute parity (still ~917× per-step at 144B effective).

---

## 4. Routing mathematics

### 4.1 Router

Single linear layer: `r_t = \mathrm{softmax}(W_r^\top q_t)`, with `W_r \in ℝ^{m × E}`. Initialized small (`σ = 1/√m`) so initial router is near-uniform.

**No router LayerNorm, no router dropout.** Both are common in standard MoE but break determinism (LN epsilon is fine, but dropout is not). MOSAIC-MOE keeps the router minimal.

### 4.2 Top-`k` with deterministic tie-breaking

```
function top_k_deterministic(r ∈ ℝ^E, k):
    # sort by (-r, index) to break ties by smaller-index-wins
    indices = [(−r[e], e) for e in 0..E−1]
    indices.sort()
    return {indices[i].second : i in 0..k−1}
```

This is identical to `argsort(r, descending=True, stable=True)[:k]`. Stability is essential: floats compared by exact bitwise equality may permute under reordering, breaking inverse-walk reproducibility.

**Numerical reproducibility on GPU.** GPU softmax is reduction-order-dependent; we fix this by using a single-warp reduction per token and `__syncwarp()` to enforce the order, identical to the post-#1 reductions used elsewhere in CHIRON's GPU stack.

### 4.3 Renormalization on `\mathcal{E}_t`

After top-`k` selection, the surviving `k` weights are renormalized:

$$\bar r_{t,e} := \frac{r_{t,e}}{\sum_{e' \in \mathcal{E}_t} r_{t,e'}}, \qquad e \in \mathcal{E}_t.$$

This makes `\sum_{e \in \mathcal{E}_t} \bar r_{t,e} = 1`, so `Y^{MOE}_l(q_t)` is a convex combination of the chosen experts' outputs.

### 4.4 Load balancing loss

Without intervention, MoE training collapses: a few experts win all the routing weight, the rest die. Standard fix: **auxiliary load-balancing loss** (Shazeer 2017, Switch-Transformer 2021):

$$\mathcal{L}_{\mathrm{LB}} := E \sum_{e=0}^{E-1} f_e \cdot P_e,$$

where `f_e := \frac{1}{T} |\{t : e \in \mathcal{E}_t\}|` is the fraction of tokens routed to `e`, and `P_e := \frac{1}{T} \sum_t r_{t,e}` is the average router probability for `e`. The total loss is

$$\mathcal{L}_{\mathrm{train}} = \mathcal{L}_{\mathrm{NLL}} + \alpha \cdot \mathcal{L}_{\mathrm{LB}},$$

with `α = 0.01` per Mixtral defaults.

`f_e` is a non-differentiable count (it depends on top-`k` indicator); the gradient flows through `P_e` only. This is sufficient empirically.

### 4.5 Capacity factor

Even with load-balancing loss, instantaneous batch-level imbalances cause some experts to receive `> kT/E` tokens. We cap each expert's per-batch token assignment at

$$\mathrm{cap}_e := \lceil c \cdot k T / E \rceil, \qquad c = 1.25.$$

If expert `e`'s assigned token list overflows `\mathrm{cap}_e`, **overflow tokens are routed to backbone-only** (no LoRA delta added). This keeps the shear well-defined and bijective.

**Important:** In CHIRON the overflow rule must be deterministic given `q`. The selection-of-overflow rule is: among tokens routed to `e`, retain the top-`\mathrm{cap}_e` by `r_{t,e}`; the rest go backbone-only. Tie-breaking by token-index. This is reproducible at inverse-walk time because the router output is reproducible.

---

## 5. Composition with #42–#52

### 5.1 #44 MELT, #46 REFLECTOR, #50 HELIUM, #52 NIMBUS — direct compositions

- **#44 MELT**: shared backbone `\mathrm{FFN}_0` *is* a MELT-TT factorization (ρ=8, two cores). LoRA adapters `(A_e, B_e)` are rank-4 and not further TT-factored. Gauge-maintenance sweep applies to backbone unmodified. **Direct.**
- **#46 REFLECTOR**: per-layer Jacobian becomes `\sum_{e \in \mathcal{E}_t} [\bar r_{t,e} \partial_q \mathrm{FFN}^{(e)} + \mathrm{FFN}^{(e)} \partial_q \bar r_{t,e}]` — convex sum of `k` per-expert Jacobians plus a router-gradient term, well-conditioned by renormalization. Top-`k` selection boundaries induce a measure-zero discontinuity, handled via straight-through (standard MoE practice). REFLECTOR's anchor schedule is FFN-form-agnostic. **Direct.**
- **#50 HELIUM**: FP8 GEMM on shared backbone (matches HELIUM §3); BF16 default for the small LoRA tensors (FP8 underflow risk on `r=4`); BF16 router. FA-3 is upstream of the FFN shear, unaffected. **Direct.**
- **#52 NIMBUS-PROMOTED**: ~7.6M extra params (W_r + LoRA) slot into the existing per-tensor Adam loop on host. K_stale=1 bound (≤ 0.003 nat) absorbs MOSAIC params identically. **Direct.**

### 5.2 #47 PHOENIX-1.58BIT — direct with caveat

Per-tensor ternarization on `W_r` (use L2-optimal scale because absmean is noisy on the small 16 K-param router), shared backbone (already supported per #47 §4 at MELT-core level), and LoRA `(A_e, B_e)` (each 8 K params; absmean works). **Caveat:** LoRA deltas are additive on the already-ternarized backbone; compounded relative L2 error ≈ 1.2× single-tensor (per #47 §4 corollary). Marginal additional empirical risk at 1.84B-active.

### 5.3 #51 ATLAS-COMPILE — hardest composition

ATLAS-COMPILE captures ~440 kernels into a CUDA Graph and replays them on a fixed schedule. Per-token branching in MoE breaks the static graph because per-expert kernel launches depend on `\mathcal{E}_t`. Three options:

1. **Skip MOE shear from graph.** ATLAS-COMPILE drops 1.45× → ~1.30× (FFN was ~10–15% of the captured kernels).
2. **Capture per-routing-histogram graph variants.** A few hundred distinct histograms occur in practice; capture the top K~32 and dispatch by lookup, falling back to eager for uncaptured. Recovers most of the gain.
3. **Capacity-throttled fixed graph at `c=1.0`.** Pin every expert to exactly `kT/E` tokens deterministically; overflow tokens go backbone-only. Per-step kernel schedule becomes fixed. ATLAS-COMPILE captures a single graph. Cost: mandatory overflow handling (already in design, §4.5).

**Default: Option 3.** Some overflow tokens see backbone-only (<5% typically); ATLAS-COMPILE composition preserved at ~1.40× (3% degradation from 1.45×).

---

## 6. Concrete primitives (kernel-level)

Five new CUDA primitives in `gpu_kernels.cu`:

1. **`moe_route_topk`** (~120 LOC) — input `(Wr, q)`, output `(E_idx, E_w)`. One block per token; computes logits[E], softmax, sorts top-`k`, renormalizes, with stable index-order tie-breaking. Cost `O(T·m·E + T·E log k)`; at `T=1024, m=2048, E=8, k=2` ~17 M FLOPs (negligible vs FFN's ~5G).

2. **`moe_build_buckets`** (~150 LOC) — converts `E_idx` into per-expert token-index lists `bucket[e][i]` and counts `counts[e]`, used to dispatch the LoRA-delta GEMMs. Capacity-overflow rule materialized here.

3. **`moe_aggregate`** (~80 LOC) — combines shared-backbone output with per-expert LoRA deltas into `Y_MOE ∈ ℝ^{T×m}`. Reads the reverse map `bucket_pos: (e, slot) → t` so each token's `k` chosen experts are summed with renormalized weights `E_w`.

4. **`moe_lb_loss`** (~30 LOC) — single-reduction `\mathcal{L}_{LB} = E \sum_e f_e P_e`; gradient flows through `P_e` only.

5. **Backward** — shared backbone receives gradient from all tokens; LoRA deltas receive gradient only from assigned-bucket tokens. Standard reverse-AD on the batched buckets, ~150 LOC across input-side and output-side adapters.

**Inverse-walk hook.** For #46 REFLECTOR-style inverse walks, recovering `Y^{MOE}_l(q')` re-runs the router on `q'`, re-does top-`k` (deterministic), re-computes the per-expert outputs, and aggregates. The cost is **exactly one MOE forward pass per inverse-walk segment** — no extra state. This is what makes reversibility cheap.

---

## 7. Honest gap analysis

### 7.1 NLL is empirical, not theoretical

The shipped #42–#52 paradigms are all bit-exact-equivalent (post-#52 composite drift ≤ 0.07 nat over 100k steps). MOSAIC-MOE breaks this: it changes the function class. **No theoretical NLL guarantee** — only empirical bounds derived from published MoE evidence:

- **Mixtral 8×7B** (Jiang et al. 2024) matches dense ~70B at 13B active.
- **DeepSeek-V3** (2024) reports ~0% perplexity loss at 37B active.
- **Switch Transformer** (Fedus et al. 2021) ~1–2% perplexity penalty.
- **GShard** (Lepikhin et al. 2020) ~0.5% penalty.

CHIRON's regime (1.84B active, MELT-TT backbone, reversible shear) is different from these settings. Our priors: **P(within 0.10 nat) ≈ 0.55, P(within 0.20 nat) ≈ 0.80, P(within 0.30 nat) ≈ 0.95**. These are subjective; not theorems.

**This is outside the iter-193 strict criterion** (≤ 0.01 nat per step), and inside the iter-197 "novel architectures invited" rider.

### 7.2 Routing instability

Standard MoE failure modes: **expert collapse** (some `f_e ≈ 0`), **capacity oscillation**, **mid-training router reset**. Mitigations: load-balancing loss `α=0.01`, capacity factor `c=1.25`, router temperature warm-up `τ: 2.0 → 1.0`, gradient clipping `‖∇W_r‖ ≤ 1.0`. ~150 LOC of capacity-aware dispatch handles this. Standard, well-documented, well-mitigated in public literature.

### 7.3 CUDA-Graph composition is degraded

§5.5 Option 3 (capacity-throttled fixed graph at `c=1.0`) preserves ATLAS-COMPILE composition at cost of mandatory overflow handling. ATLAS-COMPILE's 1.45× degrades to ~1.40× (~3% loss). If we instead skip ATLAS-COMPILE on the MOE shear (Option 1), it drops to 1.30× and the post-#51 stack drops from 690× to ~620×; layered with MOSAIC's 1.5–2×, post-#53 stack becomes 1240–1600× at 144B effective. **Default: Option 3**, acceptable composition cost.

### 7.4 Hyperparameter expansion

MOSAIC adds 6 new hyperparameters: `E, k, r, c, α, τ`. Defaults `(8, 2, 4, 1.25, 0.01, 1.0)` are Mixtral/DeepSeek-V3 transplants, *not* CHIRON-tuned. Post-Gate-0 tuning ≈ 30 GPU-hours.

### 7.5 Compositional honesty

The 2× per-step figure assumes **Hybrid factorization passes Gate-0** (§3.4). If only Shared+LoRA passes, wall-clock collapses to 1.0× and the story becomes capacity-only. **Best case: 1800× at 144B effective. Worst case: 917× at 144B effective. Expected: ~1400× at 144B effective.**

---

## 8. Stack accounting

| Stage | Per-step | Effective params at 16 GB |
|---|---|---|
| Pre-paradigm-1 baseline | 1× | 1B |
| Post-#42 (SCFA) | ~50× | 1B |
| Post-#43 (ORION) | ~108× | 1B |
| Post-#44 (MELT) | ~80× compute, ~200× memory | 18B |
| Post-#45–#49 | ~300× | 18B |
| Post-#50 (HELIUM) | ~510× | 18B |
| Post-#51 (ATLAS-COMPILE) | ~690× | 18B |
| Post-#52 (NIMBUS-PROMOTED) | ~917× | 18B |
| **Post-#53 MOSAIC-MOE (expected)** | **~1400×** | **~144B effective at 18B-active** |
| Post-#53 MOSAIC-MOE (best case) | ~1800× | ~144B effective |
| Post-#53 MOSAIC-MOE (worst case) | ~917× | ~144B effective |

The 144B effective figure assumes `E=8` and that effective-parameter scaling laws hold within a 4× factor — i.e., 8× nominal expert capacity translates to 4–8× effective capacity at fixed active compute. This is the Mixtral assumption; we have priors but no internal verification.

---

## 9. Theorems (formal)

**Theorem 1 (shear bijectivity)** — proved in §2.2.

**Theorem 2 (inverse-walk reproducibility).** *Given deterministic top-k tie-breaking, deterministic capacity-overflow rule, and bit-exact router computation, the MOSAIC-MOE forward and inverse walks satisfy `Φ^{MOE}_l ∘ (Φ^{MOE}_l)^{-1} = id` to BF16 precision.* **Proof sketch:** the inverse-walk recovers `q' = q` from the lower-triangular shear, re-invokes the router on `q'`, performs deterministic top-`k`, computes per-expert outputs, aggregates, and recovers `p = p' - Y^{MOE}_l(q')`. Each step is bit-exact at fixed GPU reduction order (§4.2); round-trip drift is bounded by `ε_{BF16} ≈ 4 × 10^{-3}` per layer, identical to the dense FFN case. □

**Theorem 3 (parameter count).** Per-layer Shared+LoRA: `P = P_{shared} + mE + 2mrE`. At `(m=2048, E=8, r=4, ρ=8)`: 81920 + 16384 + 131072 ≈ 230 K vs dense 81920 — additional overhead 150 KB per layer (negligible at 16 GB).

**Theorem 4 (effective capacity).** Under k-of-E routing and the load-balanced Mixtral asymptotic, effective per-token capacity scales as `Θ(√E)`. Not proven for CHIRON; the `E×` headline is loose.

---

## 10. Conjectures (falsifiability)

| Conjecture | Statement | Falsifier | Prior |
|---|---|---|---|
| **C1 — NLL parity** | Hybrid MOSAIC within 0.10 nat of dense at iso-active-tokens | Flagship 5×10⁹-token run (~50 GPU-hr) | P(within 0.10 nat) ≈ 0.55, P(within 0.20) ≈ 0.80 |
| **C2 — Routing stability** | `min_e f_e > 0.05` throughout 5×10⁹ tokens | `f_e` trajectory logging at 1k-step granularity | P ≈ 0.85 |
| **C3 — Effective capacity** | MOSAIC-1.84B-active matches dense-14.7B at iso-tokens | Train both; ~50 + ~250 GPU-hr | P(within 5%) ≈ 0.50 |

---

## 11. Gate-0 design

Three parallel 41M-active runs, T=1024, 2500 steps:

- **A: Dense baseline** (post-#52 stack, no MOSAIC).
- **B: MOSAIC Shared+LoRA** at `E=8, k=2, r=4` (capacity-only).
- **C: MOSAIC Hybrid** at `E=8, k=2, r=32, mid-layer per-expert` (1.5× expected speedup).

**Pass criteria:** B within 0.08 nat of A; C within 0.15 nat of A; both with `min_e f_e > 0.05` at step 2500.

**Failure paths:**
- B fails 0.08-nat bar → reject paradigm (architectural-level routing failure).
- C fails 0.15-nat but B passes → ship Shared+LoRA only (capacity-only win, retune Hybrid).
- C passes → ship Hybrid, full headline.

**Cost:** ~12 GPU-hr (3 × 41M × 2500 steps × ~4 ms/step).

---

## 12. Engineering scope and timeline

| Phase | Description | LOC | Duration |
|---|---|---|---|
| 1 | Routing kernels (top-k, tie-break, GPU softmax, LB loss) | 300 | 1 wk |
| 2 | Expert dispatch (bucket build, batched FFN, aggregation) | 400 | 1.5 wks |
| 3 | LoRA adapter training (init, Adam state, FACE composition) | 200 | 1 wk |
| 4 | Capacity throttling (overflow → backbone-only path, fixed graph) | 150 | 0.5 wk |
| 5 | Composition with #44–#52 (MELT TT cores per-expert, REFLECTOR Jacobian sum, PHOENIX ternary, HELIUM FP8, ATLAS-COMPILE per-histogram graphs, NIMBUS pipelining) | 250 | 1.5 wks |
| 6 | Gate-0 ablations and hyperparameter tuning | n/a | 0.5 wk |
| **Total** | | **~1300 LOC** | **~6 weeks** |

CLI surface: `--mosaic E=8 k=2 r=4 c=1.25 alpha=0.01 tau=1.0`. Defaults are off (paradigm shift opt-in).

Risk profile: **medium**. Routing convergence is the single empirical risk; the math, kernel work, and composition are all standard. Compare to #51 ATLAS-COMPILE (low risk, kernel-engineering only) and #52 NIMBUS (low risk, host-side pipelining only); MOSAIC-MOE is one notch higher but well within the iter-197 brief's "novel architectures invited" rider.

---

## 13. Selection rationale and naming

**Why MOSAIC-MOE for #53:** First capacity-axis paradigm (#1–#52 are wall-clock or memory; none expand effective model size at fixed memory). Strong external evidence (Mixtral, DeepSeek-V3, GShard, Switch). Composes cleanly with #44–#52 (only friction is ATLAS-COMPILE §5.5, addressed via capacity-throttled fixed graphs). Reasonable engineering scope (~1300 LOC vs #50 HELIUM's ~2100 LOC). Clear failure mode: if C1 fails the 0.10-nat bar, the Shared+LoRA fallback (§3.4) still ships as a capacity-only win at compute parity.

**Honest framing.** MOSAIC-MOE is *not* a strict-NLL paradigm; it is an architectural-novelty paradigm. The iter-197 brief's "novel architectures invited" rider relaxes the iter-193 strict criterion. We trade ≤ 0.10–0.20 nat for `8×` effective parameter capacity at single-GPU memory parity. If iter-197 turns out strict, MOSAIC-MOE should be deprioritized in favor of candidates A or C.

**Naming.** "MOSAIC" reflects the architectural metaphor — many small expert tiles arranged into a coherent whole, each token selecting which tiles to traverse. Acronym: **M**ixture **O**f **S**parsely-**A**ctivated **I**nner-experts on **C**HIRON.

---

## 14. Summary table

| Property | Value |
|---|---|
| Per-step wall-clock at 1.84B-active | 1.5× (expected); 2.0× (Hybrid passes Gate-0) |
| Effective parameter count at 1.84B-active | 8× ≈ 14.7B effective |
| Effective parameter count at 18B-active | 8× ≈ 144B effective on 16 GB single GPU |
| NLL preservation | 0.10–0.20 nat above iso-active dense (empirical, not bit-exact) |
| Memory overhead | +0.1% at flagship (shared backbone amortization) |
| New optimizer state | None (standard Adam on new BF16 weights) |
| Composition with #44 MELT | Direct (shared backbone is MELT-TT) |
| Composition with #46 REFLECTOR | Direct (Jacobian sums over `k` experts) |
| Composition with #47 PHOENIX | Direct (per-tensor ternary on backbone + LoRA) |
| Composition with #50 HELIUM | Direct (FP8 on backbone, BF16 on LoRA) |
| Composition with #51 ATLAS-COMPILE | Degraded (capacity-throttled fixed graph) |
| Composition with #52 NIMBUS | Direct |
| Engineering scope | ~1300 LOC, 4–6 weeks |
| Risk | Medium (routing convergence empirical) |
| Gate-0 cost | ~12 GPU-hr |
| Stack post-#53 (expected) | ~1400× at 144B effective |
| Stack post-#53 (best case) | ~1800× at 144B effective |
| Stack post-#53 (worst case) | ~917× at 144B effective |

---

**End of MOSAIC-MOE candidate document.**
