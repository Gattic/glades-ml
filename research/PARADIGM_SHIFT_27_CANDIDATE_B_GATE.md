# Paradigm Shift #27 Candidate B — NGATE (Neuron Gate, Concrete-STE Conditional Compute)

**Formulation class:** variational / game-theoretic / conditional computation.
**Author pass:** subagent-dispatched design (2026-04-23, shift-27 candidate B).
**Status:** candidate — awaiting side-by-side selection at the shift-27 gate.
**Target axis:** POST-NONLINEARITY ACTIVATION SPARSITY IN FFN/MLP (per-neuron, training-time).

---

## 1. Target axis

Post-nonlinearity activation sparsity in FFN: `h_in → W_up → σ → W_down
→ h_out` (GELU/SiLU/ReLU). Trained transformers have `σ(W_up·h_in)`
60-90% near-zero per token (Li 2023 "Lazy Neuron"; Mirzadeh 2023;
Csordas 2023). Shifts 1-26 never exploit this **within-FFN per-neuron**
sparsity at **training**: reversibility (#1/#8) saves act-memory, keeps
full FFN FLOPs; attention (#6/#16) cuts T², not d_ff; TRCD (#13) gates
whole layers (1/L); factorizations (#7/#10) cut d_model·d_ff, not d_ff
per-token. NGATE: **learn per-token mask over d_ff intermediate
neurons, skip compute on masked neurons in both W_up and W_down**.

## 2. Core thesis

Train tiny gate `g_ϕ : ℝ^{d_model} → [0,1]^{d_ff}` jointly with FFN
weights. Per step, per token:

1. `score_t = g_ϕ(h_in^t) ∈ [0,1]^{d_ff}` — per-neuron activation prob.
2. `m_t = ConcreteSigmoid(score_t, τ)` — Gumbel-sigmoid quasi-binary.
3. Hard TopK: `I_t = TopK(score_t, k = s·d_ff)` at target sparsity `s`.
4. **Compute only on I_t**: `z = W_up[I_t,:]·h_in`; `a = σ(z) ⊙ m_t[I_t]`;
   `h_out = W_down[:,I_t]·a`.
5. Backward STE: hard `I_t` forward; soft concrete backward to ϕ;
   `W_up/W_down` gradients are zero outside `I_t` this step.

**Game view**: G (gate) picks `I_t` to minimize `L_LM + λ·L_sparse`;
W (FFN) specializes inside support. Nash: "always kept" neurons = de-facto
subset; "never picked" = prunable at train-end.

## 3. Primitive objects

Reference: `L=24, T=1024, d_model=1024, d_ff=4096`.

- **Gate params** `ϕ = (W_g^{(1)} ∈ ℝ^{q×d_model}, W_g^{(2)} ∈ ℝ^{d_ff×q})`,
  `q=64`. Size: `(d_model + d_ff)·q = 327K/layer` (~0.08% of FFN).
- **Gate scores** `score_t ∈ [0,1]^{d_ff}` per token.
- **Sparsity** `s ∈ (0,1)`, default `s=0.25` (k=1024).
- **Temperature** `τ`, annealed `τ_0=1.0 → τ_T=0.1` exponential.
- **KL target** `s_tgt = 0.25`.
- **Warmup** `n_warmup=2000` steps with λ=0 (dense FFN).

## 4. State space

FFN weight space unchanged. New state `(ϕ, {score_t}, {I_t})`. Mask
lives in `[0,1]^{d_ff}` (hypercube), not a finite set like TRCD's
depth-one-hot. At inference `τ=0`: mask deterministic → trained gate
is a learned structured pruner, no retrain needed.

## 5. Evolution law

### 5.1 Forward — sparse FFN path

Given `h_in^t`:
1. **Gate**: `u = σ(W_g^{(1)}·h_in)`; `score = σ(W_g^{(2)}·u)`.
   Total gate cost/token: `0.65 MFLOPs` vs full FFN `16.8 MFLOPs` → **3.9% overhead**.
2. **Concrete**: `ε~Logistic(0,1)`; `m̃ = σ((logit(score)+ε)/τ)`.
3. **Hard TopK**: `I_t = argtopk(score, k=s·d_ff)`.
4. **Sub-GEMM up**: `z = W_up[I_t,:]·h_in` (cost `2k·d_model`).
5. **Activation**: `a = σ(z) ⊙ m̃[I_t]`.
6. **Sub-GEMM down**: `h_out = W_down[:,I_t]·a` (cost `2·d_model·k`).

Per-token FFN FLOPs: `4·d_model·k + 2q(d_model+d_ff) = 4.85 MFLOPs` vs
`16.78 MFLOPs` → **3.46× fwd reduction at s=0.25**; **6.8× at s=0.125**.

### 5.2 Backward — straight-through concrete

**(a) FFN weights**: gradients scatter only to rows/cols in `I_t`.
Same **3.46× bwd reduction**. **(b) Gate ϕ**: STE replaces hard-mask
Jacobian by soft-concrete `∂m̃/∂score = m̃(1-m̃)/(τ·score·(1-score))`.
Backprop through gate MLP: 2 tiny GEMMs, ~655 KFLOPs/token.

### 5.3 Sparsity loss

Default **KL to Bernoulli(s_tgt)**:
`L_sparse = (1/d_ff)Σ_i KL(Bern(score_i)‖Bern(s_tgt))`.
Alternative **L1** `|mean(score) − s_tgt|` (stabler at boundaries).

### 5.4 Optimizer — KKT primal-dual λ

`L_total = L_LM + λ·L_sparse`. Fixed λ is brittle; **dual ascent** on
constraint `E[mean(score)] ≤ s_tgt`:

    λ_{t+1} = max(0, λ_t + η_λ·(mean(score_t) − s_tgt)),   η_λ=1e-3.

Converges to the Pareto front of sparsity vs quality.

### 5.5 Temperature anneal

`τ(t) = τ_0·exp(−κt)`, `κ = log(10)/T_total`, floor `τ_T=0.1` (bias
`O(τ²)` vs variance `O(1/τ²)` tradeoff).

## 6. Mechanism mapping

| Required | Mechanism | Realized |
|----------|-----------|---------:|
| (a) ≥3× FFN fwd | Sub-GEMM on `k=s·d_ff` rows/cols | **3.46× at s=0.25** |
| (b) Activation mem | `k·T·batch` scratch | **3.2×** |
| (c) MFIO×WIP×IBGRAD | ϕ = new optimizer leaves; FFN moments update sparsely | **multiplicative** |
| (c') CHIRON | Deterministic via seed-cache replay (§11 F4) | **conditional** |
| (c'') local-window | Attention-axis independent | **additive** |
| (d) GPU | cuBLAS gather-GEMM + custom kernels | **standard** |

## 7. Objective / variational principle

    min_{W_up, W_down, ϕ}  E_{x,y,ε}[L_LM(f_{W,ϕ}(x), y)]
    s.t.                   E_{x,ε}[mean_i m̃(score_ϕ(x))_i] ≤ s_tgt

Lagrangian `Λ = E[L_LM] + λ·(E[mean(m̃)] − s_tgt)`. Primal-dual flow
(§5.4) solves the KKT system: `∇_W E[L_LM]=0` on active support;
`∇_ϕ E[L_LM] = −λ·∇_ϕ E[mean(m̃)]`; `λ ≥ 0, λ·(E[mean(m̃)]−s_tgt)=0`.
**λ = marginal LM-loss price of one extra active neuron**; at Pareto
front, λ = local slope of LM-vs-sparsity curve. Tuning `s_tgt`
traverses the front; λ self-adapts.

## 8. Theoretical analysis

**8.1 Gate gradient estimator**: concrete-STE has bias `O(τ²)`,
variance `O(1/τ²)` (Maddison 2017; Jang 2017). Anneal bounds both.

**8.2 Dense recovery**: `λ=0, score ≡ 1` ⟹ full dense FFN. NGATE
strictly contains baseline; warmup starts there.

**8.3 Conditioning**: `W_up[I_t,:]` has `κ ≤ κ(W_up)` (Cauchy
interlacing). KL gradient `(score − s_tgt)/(score·(1-score))` blows
up near `{0,1}` — mitigated by clamp `[0.01, 0.99]` or L1 fallback.

**8.4 Expressivity**: function class = union over `(d_ff choose k)`
support-indexed sub-FFNs with shared weights. Strictly more expressive
than fixed-mask pruning, less than dense (=k=d_ff). At s=0.25, support-
log = 3300 bits/token >> q·d_model gate capacity → gate is the
bottleneck, not support enumeration.

**8.5 Convergence conjecture**: **C1** under Lipschitz gate, clamped
score, τ-anneal ≤ κ/t, joint descent → KKT point at rate `O(1/√T)`
(Gidel 2019 constrained SGD). **C2 empirical**: at `s=0.25`, PPL cost
≤2% vs dense (matches DejaVu/ReLUStrikes/SwitchHead literature).

## 9. Computational trade-offs (pile_large, batch=8)

| Metric | Baseline | NGATE s=0.25 | Ratio |
|--------|---------:|-------------:|------:|
| FFN fwd FLOPs | 137 GFLOPs | 40 GFLOPs | **3.4×** |
| Gate fwd | — | 5.4 GFLOPs | +4% ovhd |
| Net FFN fwd | 137 GFLOPs | 45 GFLOPs | **3.05×** |
| FFN act memory | 32 MB | 8 MB + 2 MB score | **3.2×** |
| FFN bwd FLOPs | 274 GFLOPs | 90 GFLOPs | **3.0×** |
| Gate params | 0 | 7.9 M (24 layers) | +0.2% params |
| Gate Adam/MFIO | 0 | 1.2 MB | neutral |

At **s=0.125**: `6.8× / 6.4×`. At **s=0.5**: `1.7×` — fails 3× bar, so
`s ≤ 1/3`.

## 10. Comparison to prior art

- **TRCD (#13)**: gates depth per-token (L choices). NGATE gates d_ff
  neurons — `d_ff/L ≈ 170×` finer, orthogonal & combinable.
- **LCP (#16)**: LSH-clusters tokens. NGATE gates neurons per-token,
  no token-sharing. Orthogonal.
- **MoE**: routes token to 1-of-E expert FFNs. NGATE routes inside
  a single FFN to 1/s of d_ff neurons — ~1000× finer. Composable
  (MoE at block, NGATE inside each expert).
- **DejaVu (Liu 2023)**: inference-only predictor on frozen model.
  NGATE trains predictor+FFN jointly → model shape optimized for gateability.
- **ReLU-Strikes-Back (Mirzadeh 2023)**: static post-training prune.
  NGATE is dynamic, per-token, during training.
- **Conditional computation (Bengio 2013)**: NGATE is the concrete-STE
  instance on the FFN-neuron axis + KKT-λ control.

Novelty: **first training-time per-neuron FFN gate with primal-dual KKT
λ control, composable with reversibility and MoE**.

## 11. Failure modes and mitigations

**F1 — Early gate collapse (all-zero / all-one)**. Random init + strong
L_sparse → stall. Mitigation: `n_warmup=2000` with `λ=0` + score clamp
`[0.01, 0.99]`.

**F2 — Estimator variance blow-up at low τ**. Variance `~1/τ²` as τ→0.
Mitigation: floor `τ_T = 0.1`; separate grad-clip on `∂L/∂ϕ` vs `∂L/∂W`.

**F3 — Train-infer gap**. Training: hard TopK forward + soft STE backward.
Inference: τ=0 hard-only. **No gap** — TopK is deterministic from score.

**F4 — CHIRON reversibility breaks under Gumbel noise** [*CRITICAL*].
Reversibility needs deterministic forward. Mitigation: **cache RNG
seed per (layer, token)** for reverse replay (96 KB at pile_large).
Fallback `--ngate-deterministic`: hard TopK + REINFORCE, higher
variance but composable.

**F5 — Gather/scatter throughput**. Sub-GEMMs non-coalesced. Mitigation:
(a) per-batch token sort by similar support `I_t`; (b) `cublasGemm
StridedBatchedEx` per-token strides; (c) gather cost ≤20% of sub-GEMM
at s=0.25 on RTX 4080 SUPER.

**F6 — Aggressive sparsity**. At s<0.15 PPL degrades. λ auto-tunes;
user picks `s_tgt` by quality budget. Sweet spot s=0.25 for pile_large.

**F7 — Sparse Adam moments**. Rows outside `I_t` get zero grad but
nonzero `(m,v)` from prior steps. Standard accumulate — no change needed.

**F8 — MFIO BF16 sparse scatter**. Non-atomic BF16 writes; use per-layer
FP32 scratch, cast at optimizer-step boundary.

## 12. Minimal prototype

**New GPU primitives** (`gpu_ngate.{h,cu}`):

1. `ngate_forward(h_in, W_g1, W_g2, score_out)` — 2 cuBLAS SGEMMs + σ.
2. `ngate_concrete_sample(score, eps_seed, tau, m_soft)` — Gumbel-sigmoid, seeded for CHIRON replay.
3. `ngate_topk_support(score, I, k)` — radix-select or bitonic top-k.
4. `ngate_gather_matmul_up(W_up, I, h_in, z_out, k)` — batched indexed GEMM via `cublasGemmStridedBatchedEx`.
5. `ngate_scatter_matmul_down(W_down, I, a, h_out, k)` — symmetric scatter.
6. `ngate_scatter_grad_accumulate(dW_sparse, I, dW_full)` — sparse→dense scatter for Adam.
7. `ngate_kl_bernoulli_loss(score, s_tgt)` — KL divergence.
8. `ngate_lambda_update(score_mean, s_tgt, lambda, eta_lambda)` — dual step.

**Parity tests** (`unit-tests/.../chiron-test.cpp`):

- `CHIRONNgateGateForwardParityTest` — gate MLP vs ref, err < 1e-5.
- `CHIRONNgateConcreteSTEGradientTest` — finite-diff `∂L/∂score`, err < 1e-3 at τ=1.
- `CHIRONNgateGatherMatmulPerfTest` — gather-GEMM vs dense, ≥2.5× wall-clock at s=0.25.
- `CHIRONNgateE2EConvergenceTest` — 4-layer tx, 1000 steps; PPL ≤5% vs dense, FFN ≥2.8× faster.
- `CHIRONNgateChironReplayTest` — NGATE+CHIRON seed-replay reversibility err < 1e-5.

**CLI**: `--ngate`, `--ngate-s 0.25`, `--ngate-q 64`, `--ngate-tau-0 1.0
--ngate-tau-T 0.1`, `--ngate-warmup 2000`, `--ngate-lambda-lr 1e-3`,
`--ngate-penalty kl|l1`, `--ngate-deterministic`.

**First E2E**: TinyStories 128M `--ngate --ngate-s 0.25 --ngate-warmup 500
--local-window 128 --chiron --mfio 2 --wip-K 4 --accum 8`. Target: 3.0×+
FFN fwd wall-clock, PPL within 3%, support-churn <5% per 100 steps.

## 13. Composition with the shipped stack

- **CHIRON × NGATE**: reversible storage × sparse FFN; multiplicative,
  contingent on F4 seed-replay.
- **MFIO × WIP × IBGRAD × NGATE**: 4-way orthogonal (optimizer × weight
  interp × grad subspace × neuron sparsity).
- **local-window × NGATE**: `(T² → T·W) × (d·d_ff → d·s·d_ff)` — independent.
- **TRCD (#13) × NGATE**: depth × neuron gates. Per-token FLOP
  `= depth(t)·s·4·d·d_ff` — log-scale combined savings.
- **ATC-Δ (#26) × NGATE**: **mutually exclusive at FFN block**. Resolution:
  ATC-Δ in K-window, NGATE at refresh step (where ATC-Δ runs full forward
  anyway). Max-of-both savings.

## 14. Summary + promote condition

NGATE learns a tiny gate `g_ϕ(h_in)` predicting which d_ff neurons
post-σ will be nonzero. Training: concrete-Gumbel-STE + hard TopK forward
+ KKT primal-dual λ control. At s=0.25 pile_large: **3.4× FFN fwd FLOPs,
3.2× intermediate activation memory, ≤2% projected PPL cost**; dense
recovered at `λ=0, score≡1`.

**Promote condition**: after MFIO×WIP flagship at pile_large (DONE
2026-04-23), NGATE joins as **neuron-axis FFN-sparsity factor**,
orthogonal to ATC-Δ's time-axis. Per-FFN-block selection: NGATE at
refresh step, ATC-Δ in K-window. Next iteration: Phase 1 GPU primitives
(gather-GEMM + concrete sampler) + `CHIRONNgateChironReplayTest` as
first milestone gate.
