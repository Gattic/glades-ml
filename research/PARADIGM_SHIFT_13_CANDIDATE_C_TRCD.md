# Paradigm shift #13 — Candidate C: Token-Routed Conditional Depth (TRCD)

**Formulation class:** game-theoretic / bilevel-control.
**Author pass:** subagent-dispatched design (2026-04-22).
**Status:** candidate design.  Implementation deferred to selection.

---

## 1. Short name and core thesis

**TRCD** — **T**oken-**R**outed **C**onditional **D**epth.  Working
codename: *λ-threshold depth market*.

**Thesis.** Pre-training wastes compute by forcing every token through
every layer.  A token predicting the suffix of a closed-class word,
"[PAD]" continuations, or the closing bracket of a dict literal is
near zero cross-entropy at layer 2-3; another 21 layers add nothing.
Conversely, a token at the end of a genuine ambiguity needs every
layer plus attention against the full context to reach its best NLL.
TRCD makes **depth a per-token decision** governed by a KKT condition
on a bilevel program: each token is a selfish agent that buys layers
one at a time at a price λ set by the outer loss; the shared
early-exit head prices the token's current state in NLL units.  The
outer loop tunes λ to drive the observed average depth to a target
d̄.  At equilibrium, every advancing token has equal marginal utility
per FLOP, and every exiter has saturated its predictive content.

Against Mixture-of-Depth (MoD) this differs in that MoD routes at a
*fixed* top-k per layer (heuristic capacity schedule); against
Schuster et al. early-exit it differs in using a FLOP price rather
than a confidence threshold; against Universal Transformer it
differs in routing through depth rather than halting at a per-token
count of the same layer.  **λ is not a hyperparameter** — it is the
Lagrange multiplier the outer training loop enforces.

---

## 2. Primitive objects and state space

Let the backbone have L = 24 transformer blocks of width d = 1536,
vocabulary V, sequence length T = 1024.

| symbol          | shape                       | meaning                                        |
|-----------------|-----------------------------|------------------------------------------------|
| h_{l,t}         | ℝ^d                         | residual-stream state of token t after block l |
| π_l             | ℝ^d → [0,1]                 | per-layer routing policy (continue probability) |
| a_l             | ℝ^d, b_l ∈ ℝ                | π_l(h) = σ(a_l^⊤ h + b_l); d parameters/layer  |
| W_exit          | ℝ^{d × V}                   | **shared** early-exit head, tied to W_emb^⊤     |
| s_{l,t}         | {0,1}                       | Gumbel-softmax / ST binary: continue vs. exit  |
| α_{l,t}         | [0,1]                       | soft continue probability used during training |
| d(t)            | {1,…,L}                     | sampled/STE depth of token t this step         |
| λ               | ℝ_+                         | current Lagrange price of one layer (FLOPs)    |
| d̄_target        | ℝ                           | operator-set target average depth (e.g., 8)    |
| u_{l,t}         | ℝ                           | local utility estimate ΔNLL_l(h) at layer l    |

**Shared exit head.**  W_exit ∈ ℝ^{d×V} is a single matrix.  It is
**not duplicated** per layer; per-layer classification heads would
cost L·d·V = 24·1536·65 536 ≈ 2.4 B parameters and dominate memory.
Instead every layer consumes the same head.  Input normalization
prior to W_exit is a per-layer RMSNorm γ_l ∈ ℝ^d (trivial storage
L·d ≈ 37 K parameters).  This keeps per-layer exit cost at d
parameters + d RMSNorm weights — essentially free.

**Routing policy.**  π_l is deliberately linear-in-h (a·h+b) not a
deep MLP.  A deep router at every layer would re-introduce the depth
cost we are removing.  The routing parameters {a_l, b_l} are trained
jointly by the same Gumbel-softmax relaxation that the rest of the
network sees.

**Training state.**  Per optimizer step we track (a_l, b_l, γ_l) for
the routing and exit-head normalization at each layer, plus a
scalar-valued EMA of average depth d̄_obs used by the λ-controller
(Section 6).

---

## 3. Evolution — forward

At layer l, for each token t currently alive (i.e., s_{l',t}=1 for
all l' < l):

1. Compute residual-stream update as usual: h_{l,t} ← block_l(h_{l-1,t}).
2. Compute continue logit: z_{l,t} = a_l^⊤ h_{l,t} + b_l.
3. Sample Gumbel noise g_{l,t} ~ Gumbel(0,1); straight-through
   (ST) discrete decision
   s_{l,t} = 𝟙[z_{l,t} + g_{l,t} > 0],
   soft value α_{l,t} = σ((z_{l,t} + g_{l,t}) / τ)
   with temperature τ annealed 1 → 0.3 over training.
4. If s_{l,t} = 0 (exit): token t's final logits are
   z^{out}_t = W_exit · RMSNorm(h_{l,t}; γ_l); its depth d(t) = l.
5. If s_{l,t} = 1 (continue): token t advances to layer l+1.
6. At l = L, any remaining tokens auto-exit.

**Differentiability.**  The forward computation at layer l multiplies
the update to h by the continue-indicator α in the residual stream:

    h_{l,t} = h_{l-1,t} + α_{l,t} · Δblock_l(h_{l-1,t})

so that if α_{l,t} = 0 the block does nothing and h_{l,t} = h_{l-1,t}.
This is the Mixture-of-Depth-style gating.  During training the
forward uses α_{l,t} ∈ [0,1] (soft); during inference we use the hard
s_{l,t} ∈ {0,1} and physically skip the block on exited tokens.

**Loss materialization.**  Every token contributes *exactly one*
cross-entropy term — the one at its exit layer d(t).  A token that
exits at layer l contributes

    ℓ_t = − log p_{W_exit}(y_t ∣ h_{l,t}).

Tokens that exit at different depths produce gradients that
accumulate into the **same** W_exit and the per-layer γ_l, plus
gradients into the per-layer block parameters restricted to l ≤ d(t).

---

## 4. The bilevel program and the KKT-derived policy

**Outer objective.**

    L_total(θ, π) = E_t [ NLL_{d(t)}(h_{d(t),t}) + λ · d(t) ]
                        s.t. E_t[d(t)] ≤ d̄_target.

λ is the KKT multiplier on the budget constraint.  The outer loop
performs gradient descent on (θ, π) while a separate PI controller
(Section 6) adjusts λ so that the constraint binds.

**Inner problem (per-token best response).**  Fix all θ and λ.  Token
t, having just produced h_{l,t}, chooses whether to pay one more
layer (FLOP cost λ) in exchange for a reduced NLL at exit.  Define
the per-layer predicted utility

    u_{l,t} = E[NLL_l(h_{l,t}) − NLL_{l+1}(h_{l+1,t}) ∣ h_{l,t}].

This is the expected NLL drop from advancing one more layer —
computable in the forward pass by comparing the exit-head loss at
layer l against a 1-layer-ahead estimate (we use the soft-advance
activation α_{l,t}·h_{l+1,t} with detached block parameters to
estimate u_{l,t} at no extra backward cost).

**KKT condition.**  The token's rational rule at layer l is

    continue if u_{l,t} > λ,   exit if u_{l,t} ≤ λ.

Proof sketch.  The per-token utility of a depth-l exit is
U(l,t) = −NLL_l(h_{l,t}) − λl.  Moving from l to l+1 changes U by
u_{l,t} − λ.  The agent advances iff this is positive.  Because
NLL(·) is monotone-decreasing along the optimal backbone (trained
so), u_{l,t} is weakly decreasing in l, so the first l at which
u_{l,t} ≤ λ is the agent's unique best response.  QED.

**Training enforces this.**  The routing logit z_{l,t} = a_l^⊤
h_{l,t} + b_l is *trained* to approximate u_{l,t} − λ: we add an
auxiliary regression loss

    L_route = Σ_{l,t}  (z_{l,t} − (u_{l,t} − λ))².

This grounds the policy in the KKT condition.  Without L_route the
router is a free Gumbel gate and can degenerate (Section 9).  With
L_route the router is forced to learn the *gradient of the value
function* — the canonical shadow price structure.

---

## 5. Shared-head gradient flow

W_exit receives gradients from tokens at every exit layer.  Because
training uses gated residuals (not hard branching), each layer
contributes a weighted loss

    ℓ_l(t) = (1 − α_{l,t}) · (Π_{l'<l} α_{l',t}) · CE(W_exit · RMSNorm(h_{l,t}; γ_l), y_t).

The first factor is "exit at l" probability; the second is
"survived to l".  Total loss is Σ_l ℓ_l(t) with α_{L,t} ≡ 0
(layer L forces exit).  Differentiable in α and upstream θ.

**Shared-head stability.**  A naive shared head would output
meaningful logits only at depth L and random logits at depth 1 —
the *"deep-layer degrades shallow"* pathology.  TRCD defuses it
three ways:

1. *Layerwise RMSNorm* γ_l: the head sees RMSNorm-normalized
   activations that can have different scales/directions per layer.
2. *Routing-weighted aggregation*: tokens with high exit-probability
   at layer l contribute disproportionately to the head's gradient
   at layer l's activation geometry.  Low-entropy tokens early →
   head learns their class of inputs; high-entropy tokens late →
   head learns those.  The two do not fight over the same
   representation because the routing makes them *disjoint*.
3. *Anti-collapse regulariser*: a target-matching term on the head
   output distribution per layer,
   D_KL(p^{out}(·|l, t) ‖ p̄(·|l)) where p̄ is the running empirical
   marginal at layer l.  Prevents the head from degenerating to the
   marginal unconditionally.

---

## 6. The λ-controller

λ is adapted by a simple proportional-integral law:

    e_k = d̄_obs,k − d̄_target
    λ_{k+1} = max(0, λ_k + η_P · e_k + η_I · Σ_{j≤k} e_j)

with η_P = 1e-4, η_I = 1e-6 in units of nats/layer.  If observed
depth is too small, λ decreases (layers get cheaper → more tokens
advance); if too large, λ increases.

d̄_target is set by the FLOP budget.  For pile_large with L = 24 and
a 3× forward-compute reduction we want d̄_target ≈ 8.

**Why PI and not fixed λ.**  Fixed λ at the beginning of training
drives all tokens to exit at layer 1 (NLL is terrible everywhere,
λ·d dominates).  A decaying schedule on λ is brittle and trades a
tuning knob (the schedule) for the one we were trying to eliminate.
The PI controller uses the empirical budget as the exogenous signal
and is self-tuning.

**Stability of the control loop.**  The depth response to λ is
monotone-decreasing (higher λ → fewer advances) and locally linear;
standard anti-windup on the integral term and a dead-band of ±0.2
layers on e prevent oscillation.

---

## 7. Per-token GPU scheduling

The efficiency claim *requires* that skipped layers are actually not
computed.  During training with the gated-residual formulation this
is not trivially true — α·Δblock still evaluates Δblock.  To
realize the FLOP savings we use dynamic batching per layer.

**Per-layer kernel flow.**

1. After layer l−1 produces continue-probabilities α_{l-1,·},
   compute a bitmask m_l = (α_{l-1,·} > 0) ∈ {0,1}^{B·T}.
2. Prefix-sum on m_l yields a compact index list idx_l ∈ ℕ^{N_l}
   where N_l is the count of active tokens at layer l.
3. Gather h_{l-1,idx_l} into a contiguous buffer of size N_l·d.
4. Run block_l on the compact buffer, producing Δh of size N_l·d.
5. Scatter back: h_{l,·} = h_{l-1,·} + α_{l-1,·} · scatter(Δh,idx_l).

Steps 2–5 are standard MoD scheduling; our gather/scatter kernels
(`indexed_gather_rowmajor`, `indexed_scatter_add_rowmajor`) already
exist in `Backend/Machine Learning/Networks/cuda/gpu_kernels.cu`
from paradigm shift #6.

For attention specifically (which couples tokens), we use a
per-layer *active-mask* that zeros out rows and columns in the
attention score matrix corresponding to exited tokens.  This is
compatible with the existing local-window kernel because the
local-window kernel already consumes a mask.  The K/V cache is only
advanced for active tokens.

**Inference mode.**  At inference we use the hard s_{l,t} and
*physically* skip the block.  Softmax-temperature Gumbel noise is
disabled; routing is the deterministic σ threshold at 0.5.  This
addresses failure mode (4) below.

---

## 8. Expected FLOP savings

Total backbone FLOPs per step are L·T·f where f is per-layer
per-token forward cost.  With TRCD, FLOPs are T·d̄_obs·f.  At
d̄_target=8, L=24, this is a 3× reduction in forward and — because
backprop only traverses forward-traversed layers — 3× in backward.
Overall step time reduced by [2.5, 3.0]×; the residual is routing,
gather/scatter, and exit-head costs which scale O(L·d) not O(L·d²).

**Memory.**  Per-layer activations stored only for active tokens;
activation memory reduced by d̄/L = 1/3.  Additive with CHIRON (#1)
— CHIRON removes memory for traversed layers, TRCD removes the
traversed layers themselves.

---

## 9. Failure modes and mitigations

**(F1) Trivial equilibria — all-d=1 or all-d=L.**  Low λ advances
all tokens to L (no savings); high λ exits all at 1 (catastrophic
NLL).  The PI controller on λ cannot stay at either extreme
because the constraint binds only at d̄_target.  We also enforce
minimum depth d_min=2, clamp α to [0.02, 0.98] preventing Gumbel
saturation, and use L_route to pin routing to the KKT condition —
blocking λ-insensitive collapses.

**(F2) Routing-π_l gradient instability.**  Binary Gumbel ST has
high variance.  Mitigations: (i) τ annealed 1.0→0.3 (Jang et al.);
(ii) soft α in the training forward, hard gate inference-only;
(iii) L_route supervises a_l·h+b via a path independent of Gumbel;
(iv) per-layer gradient clipping on (a_l, b_l) at 1.0; (v) no
backprop through d(t), only through smooth α_{l,t}.

**(F3) Shared head competing objectives across depth.**  Early
training the head receives mostly shallow activations (backbone is
still junk), biasing it.  Mitigations: (i) per-depth RMSNorm γ_l
absorbs depth-specific scale; (ii) anti-collapse KL prevents the
head from memorising the marginal; (iii) 2000-step warm-up forces
d(t)=L (λ starts at 0, d̄_target ramps L→8 linearly) — head learns
the full task before depth adaptivity begins; (iv) EMA of W_exit
with half-life 500 steps decouples from fast perturbations.

**(F4) Inference non-determinism from Gumbel.**  Two runs on the
same input must produce identical outputs.  Inference sets g≡0 and
thresholds σ(z_{l,t})>0.5; the glades `generate()` API propagates a
`deterministic=true` flag enforced at every routing decision;
unit tests assert exact-equal logits across two inference calls.

**(F5, anticipated) Shared-head depth interference.**  Gradients
from exit-at-3 and exit-at-24 tokens co-train W_exit.  Per-layer
RMSNorm decorrelates the inputs (MoD-style fix); we monitor the
condition number of W_exit and alert if it exceeds 1e4.

---

## 10. Relation to prior work

- **MoD (Raposo et al., 2024).**  Fixed per-layer capacity ratio
  (top-k tokens pass).  TRCD replaces top-k with a per-token
  λ-threshold derived from a bilevel.  Result: token depth is
  content-determined, not rank-determined.
- **Schuster et al. early-exit.**  Confidence-threshold at each
  layer, heuristic threshold tuning.  TRCD gives the threshold a
  KKT interpretation and couples it to a FLOP constraint via λ.
- **Universal Transformer (Dehghani et al.).**  Halting on a
  per-token count of the *same* layer applied repeatedly; no
  depth specialization.  TRCD has a proper L-layer backbone and
  routes *through* depth, not around it.
- **SkipDecode, LayerSkip, CALM.**  All use per-layer classifier
  heads (L · d · V parameters) — TRCD's single shared W_exit is
  essential for the 16 GB ceiling.

**Novelty claim.**  The λ-controlled bilevel is the piece that no
prior work has.  It turns depth from a heuristic into an
equilibrium.

---

## 11. Interaction with previous paradigm shifts

- **#1 CHIRON** — composes cleanly; skipped reversible blocks cost
  zero activation memory, traversed ones use CHIRON recomputation.
- **#2/#6 attention kernels** — compact gather produces a smaller
  N_l; the local-window kernel already accepts a variable token
  count, tile parameters need minor adjustment.
- **#7 Stiefel × Σ** — orthogonal weights are orthogonal at any
  depth; preserved structurally.
- **#9 OVFG** — per-layer rank-T gradient factorization preserved;
  only the set of layers receiving gradients changes.
- **#11 MFIO** — no TRCD-specific concern.
- **#12 DFA** — feedback paths must mask by token-depth; small
  change to the DFA dispatcher but compatible.

Net: additive with all twelve prior shifts.

---

## 12. Implementation sketch and evaluation plan

Three engineering deliverables: (1) routing gate in
`transformer_routing.h/.cu` — linear projection + Gumbel + ST gate,
~200 LOC; (2) dynamic batcher in `dynamic_batch.cu` reusing
paradigm-#6 indexed_gather/scatter, ~300 LOC; (3) PI λ-controller
in `lambda_controller.h/.cpp`, ~100 LOC.  The shared exit head
reuses the tied W_emb^⊤ output projection.  Estimated effort: 2
engineering weeks; medium risk, mostly in PI gain tuning and
anti-collapse KL weighting.

Evaluation: (a) microbench on pile_large (L=24, d=1536, 16k steps)
— verify d̄_obs tracks d̄_target=8 within ±0.3 layers, step-time
2.5-3.0× faster, final-validation NLL within 0.05 nats of dense
baseline; (b) plot d(t) histogram by y_t entropy bucket, expect
monotonicity; (c) ablations (no L_route, fixed λ, no anti-collapse
KL) should fail predictably per Section 9; (d) end-to-end on a
2.23 B model, target ≥2.5× speedup at Δperplexity ≤ 0.2.

---

*End of candidate C.*
