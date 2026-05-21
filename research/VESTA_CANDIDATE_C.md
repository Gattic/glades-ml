# VESTA Candidate C — **PULSAR**: Point-process Unified Locus of Sparse Asynchronous Recurrence

**Date:** 2026-05-19
**Author scope:** one of three parallel VESTA candidate-framework designs (A = non-diagonal structured recurrence, B = causal information-bottleneck multi-particle, **C = event-driven hierarchical compute, R1+R6+R7 unified — this document**).
**References:** `research/VESTA_AUDIT.md` (R1=MambaByte, R6=no baseline, R7=Mixture-of-Depths), `newmodel.txt`, `research/EALRMN_PHASE0F_RESULTS.md`, `research/EALRMN_PHASE0G_RESULTS.md`.

---

## 0. Executive abstract

Three reference axes of the VESTA brief — **R1 (entropy-adaptive chunking)**, **R6 (memory-write regularisation)**, and **R7 (compute-adaptive inference)** — are each fundamentally a question of *when to spend resources on which token*. Existing literature treats them as three independent mechanisms, with three independent gates, three independent regularisers, and (in the cases of R6 and R8) no clean baseline at all (`VESTA_AUDIT.md` §2).

**PULSAR proposes that all three axes are governed by a single object: a learned marked point process N on the sequence, whose atoms ("events") simultaneously trigger (i) a chunk boundary, (ii) a memory write, and (iii) a depth-expansion compute path.** The default token path is cheap (single shallow block + diagonal linear recurrence, *no* memory write, *no* extra compute); the event path is expensive (deep block stack + scatter write to a bounded slot memory + attention-based read from memory).

The architectural commitment is that **fewer mechanisms = fewer bootstrap-circularity surfaces**. The prior EALRMN program's central failure was bootstrap circularity at three independent gates (encoder, memory write, attention read; Phase-0a–0g, `EALRMN_PHASE0F_RESULTS.md` §5). PULSAR collapses these to one event detector whose training is anchored by an **endogenous surprisal residual** — the detector is supervised by the running gap between the cheap path's prediction and an EMA of the full prediction. This is *self*-supervised in the sense that the supervision is internal to the model, not task-specific (contrast EALRMN Phase-0f's `--aux-gate` which required oracle marker labels).

The framework reduces, by ablation, to:
- **All events off**: a plain LRU-style linear-recurrence baseline → must reproduce VESTA Claim B0 (≥100× over tanh on needle T=2048).
- **All events on**: a fixed-depth dense transformer with full slot memory.
- **Fixed-stride events**: a stride-S Mixture-of-Depths variant on the deep path, MambaByte-style fixed-stride patching on the chunk path.

The adversarial baselines are:
1. **MambaByte** (R1) at iso-FLOPs on the *compression-burst* task (T6 of `newmodel.txt`).
2. **Mixture-of-Depths** (R7) at iso-FLOPs on the *heterogeneous-compute* task (T7).
3. **LRU + write-every-stride** (R6 null) on a *bandwidth-constrained slot memory* task (T2 multi-needle).

Claim N1 requires PULSAR to beat MoD by ≥0.04 nat at iso-FLOPs on T7. Claim N2 requires the event rate to stay in [0.01, 0.5] for ≥80% of training without auxiliary supervision. Claim N3 requires the all-events-off ablation to reproduce VESTA B0.

C++ cost: ~3,500 LOC of new code in `glades-ml`, dominated by sparse gather/scatter kernels for the event-conditional depth-expansion. Estimated single-developer wall-clock: 3 weeks to prototype-of-prototype (CPU C++98 in `research/`), ~6 weeks to CUDA-backed `glades-ml` module.

Most-worried failure mode: **event-rate collapse**. The endogenous surprisal residual decays as the model fits the data, so the supervision signal weakens late in training and the event rate collapses to ~0 (no events → degenerate LRU) or saturates to ~1 (every token an event → degenerate dense transformer). Mitigations: (i) explicit Lagrangian dual variable on event rate (Section 6.4), (ii) hard floor via clipping the rate-loss before backprop, (iii) Gumbel-Top-K with annealed temperature.

---

## 1. Short name and core idea

**PULSAR** — *Point-process Unified Locus of Sparse Asynchronous Recurrence.*

**Core idea (one sentence):** A learned **marked point process** on the sequence produces sparse event times {τ_k}; the model maintains two coupled streams — a *default* diagonal-linear-recurrent state advanced every step, and a *high-resolution* deep state advanced only at events — and a single set of regularisers governs all three of (chunking, memory writes, depth) by acting on the event process.

The name is a deliberate astrophysical pun: a pulsar emits coherent, sparse, periodic bursts from a high-energy core surrounded by a quiet magnetosphere — the geometry of *cheap continuous default + expensive episodic burst* is the model's architectural commitment.

---

## 2. Primitive objects

### 2.1 The sequence

Let `x = (x_1, …, x_T) ∈ V^T` be a sequence of token-or-byte ids over vocabulary `V`. (PULSAR is byte-friendly by construction; the cheap path's per-step cost is small enough that byte-level operation does not blow up the wall-clock — this is the basis for the R1 comparison against MambaByte.)

### 2.2 The point process N

Let `N` be a marked point process on `[1, T]` with atoms `0 ≤ τ_1 < τ_2 < … < τ_K ≤ T`, K random. Each atom carries a mark `m_k ∈ {0, 1}^d_mark` (a real-valued vector that the event path can use to differentiate event types). N is parameterised by an *intensity* function `λ(t | F_{t-1})` where `F_{t-1}` is the σ-algebra of everything known at step t-1. The conditional intensity is realised in discrete time as a per-step *probability* `p_t = σ(η_t)` where `η_t` is the **logit-intensity** computed from the model's own state (Section 4).

`p_t` is the model's instantaneous probability of declaring step `t` an event. Sampled events are written as `e_t ∈ {0, 1}` with `e_t ~ Bern(p_t)`; the event count is `K = Σ_t e_t`.

### 2.3 The two streams

PULSAR maintains **two state streams**:

(a) The **default state** `s_t ∈ ℝ^d_s`, advanced every step `t` by a cheap linear recurrence:
$$s_t = A · s_{t-1} + B · u_t$$
with `A` diagonal complex (LRU-style; `VESTA_AUDIT.md` row LRU). `u_t = E[x_t]` is the embedding of the current token. `A` is initialised with magnitudes uniformly in `[0.9, 0.999]` and phases uniformly in `[0, 2π]`, per LRU recipe (Orvieto et al. 2023, arXiv 2303.06349). The default state is a *general-purpose* compressor of the sequence — fixed-state-size, no write-gate, no nonlinearity.

(b) The **high-resolution event state** `h_e ∈ ℝ^d_h` (`d_h ≫ d_s`, e.g. `d_h = 4·d_s`). `h_e` is a sequence indexed by event number `e = 1, …, K`, *not* by token index t. The event state advances only at event times:
$$h_e = G_θ(h_{e-1},\, s_{τ_e},\, m_e,\, R(M_{e-1},\, s_{τ_e}))$$
where `G_θ` is a *deep* block — a 2–4 layer transformer block, OR a Mamba block, OR a Hawk-style RG-LRU+local-attention sandwich. It receives:
- the previous event state `h_{e-1}`,
- the *current* default state `s_{τ_e}` (this is the read-from-default channel),
- the mark `m_e` (a learnable function of `s_{τ_e}` and a position embedding for `e`),
- the read from a bounded slot memory, `R(M_{e-1}, s_{τ_e})`.

### 2.4 The slot memory M

The slot memory `M ∈ ℝ^{S × d_m}` has `S` slots (S = 8, 16, or 32). It is **updated only at events**:
$$M_e = W(M_{e-1},\, h_e,\, s_{τ_e})$$
with `W` a write operator (Section 3.4). Between events the memory is frozen. The memory is read by both the default path (rarely — only for the final readout) and the event path (every event, via attention).

### 2.5 The coupling map

The full per-step state is `(s_t, h_{e(t)}, M_{e(t)})` where `e(t) := max{e : τ_e ≤ t}` is the number of events up to and including step t. Step-by-step the operations are:

```
For t = 1, …, T:
  1. s_t = A · s_{t-1} + B · E[x_t]                      # default path (cheap, dense)
  2. η_t = f_φ(s_t, h_{e(t-1)}, M_{e(t-1)})              # event-detector logit
  3. e_t ~ Bern(σ(η_t))                                  # sample event (Gumbel-sigmoid at train)
  4. if e_t = 1:
     a. e := e(t-1) + 1, τ_e := t
     b. r_e = R(M_{e-1}, s_t)                            # attention read from memory
     c. m_e = M_mark(s_t)                                # mark function
     d. h_e = G_θ(h_{e-1}, s_t, m_e, r_e)                # deep block (expensive)
     e. M_e = W(M_{e-1}, h_e, s_t)                       # slot write
  5. logits_t = U · s_t + V · h_{e(t)}                   # final readout (both streams)
```

The default stream `s_t` is the *cheap path*. The event stream `h_e` is the *expensive path*. The event-detector `f_φ` decides when to take the expensive path. The slot memory `M` is *only* written/read at events — its update rate is bounded by `K/T = average event rate`.

### 2.6 Default-cheap, event-expensive cost decomposition

Per-step cost decomposes as:
$$\text{cost}(\text{PULSAR}) = T · c_{\text{default}} + K · c_{\text{event}}$$
where `c_default ≪ c_event`. If the event rate `K/T = ρ`, the total cost is `T · (c_default + ρ · c_event)`. Compared to a dense transformer of equivalent expressivity, which pays `T · c_dense ≈ T · c_event`, PULSAR's relative cost is `(c_default / c_event) + ρ`. For typical sizes (`d_s = 256`, `d_h = 1024`, 2-layer deep block) this is roughly `0.05 + ρ`. **At ρ = 0.1 PULSAR runs at ~15% of dense cost.** This is the headline R7 claim.

---

## 3. Evolution law (formal)

### 3.0 Notation and conventions

Throughout: lowercase letters for scalars (`t`, `T`), bold-lowercase for vectors (we use plain in prose), uppercase for matrices (`A`, `B`), calligraphic for losses (`\mathcal{L}_{\text{LM}}`). Time index `t ∈ {1, …, T}`. Event index `e ∈ {1, …, K}`, K random. The map `e(t) = max{e : τ_e ≤ t}` is the running event count. We write `\mathbb{1}[·]` for the indicator. All expectations are over training-data distribution unless stated.

The point process N is realised as the *thinning* of the deterministic intensity `λ_t = σ(η_t)`. The likelihood of observing `(e_1, …, e_T) ∈ {0,1}^T` factorises as
$$p(e_{1:T} \mid F_{1:T}) = \prod_{t=1}^T \sigma(η_t)^{e_t} \cdot (1 - σ(η_t))^{1-e_t}.$$
This is the standard discrete-time Bernoulli point-process likelihood (Daley & Vere-Jones 2003, ch. 7). The marks `m_e` are deterministic functions of `s_{τ_e}` so the marked process is fully specified by `λ_t` and the deterministic mark map.

### 3.1 Default-stream recurrence

The default state is a **diagonal complex linear RNN** (LRU). In real-valued representation:
$$\begin{pmatrix} s_t^{\Re} \\ s_t^{\Im} \end{pmatrix}
= \begin{pmatrix} \text{diag}(\nu \cos\theta) & -\text{diag}(\nu \sin\theta) \\ \text{diag}(\nu \sin\theta) & \text{diag}(\nu \cos\theta) \end{pmatrix} \begin{pmatrix} s_{t-1}^{\Re} \\ s_{t-1}^{\Im} \end{pmatrix}
+ \begin{pmatrix} B^{\Re} \\ B^{\Im} \end{pmatrix} u_t$$
with `ν ∈ ℝ^{d_s/2}`, `θ ∈ ℝ^{d_s/2}` parameterised as `ν_i = exp(-exp(ν^{raw}_i))` (LRU stability parameterisation) and `θ_i = exp(θ^{raw}_i)`. The output of the default block is `y_t = Re(C · s_t)` with `C ∈ ℂ^{d_s/2 × d_y}` (real-valued readout dim `d_y`).

**Computational cost per step:** `O(d_s)` (diagonal matvec) + `O(d_s · d_y)` (readout). Linear in d, no quadratic attention cost.

**Memory cost:** persistent state `s_t` of size `O(d_s)`. No KV-cache.

The default stream is *the* baseline that VESTA Claim B0 requires us to reproduce. Section 8.3 derives PULSAR-with-all-events-off = LRU exactly.

### 3.2 Event-stream advance

The event state advances by the rule
$$h_e = G_θ(h_{e-1},\, s_{τ_e},\, m_e,\, r_e)$$
where `G_θ` is a *deep transformer block* implementing:
- Pre-LN of the inputs.
- Cross-attention from `h_{e-1}` to `[s_{τ_e}; m_e; r_e]`.
- Self-attention on `h_{e-1}` only (1-token sequence, but with multi-head structure on the `d_h`-dim representation).
- MLP / SwiGLU.
- Residual connections.

`G_θ` is typically 2 to 4 transformer blocks deep. Per-event cost: `O(d_h²)` for the MLP, `O(d_h · S)` for the memory read, `O(d_h · d_s)` for the default-stream cross. With `S = 16`, `d_h = 1024`, `d_s = 256`, this is dominated by the `d_h²` MLP, costing ~`10⁶` ops per event.

**Memory cost:** persistent `h_e` of size `O(d_h)`; *no* per-token KV cache because event indices are sparse (K events total, not T).

### 3.3 Slot-memory write

The write operator at event `e` is parameterised in two variants:

(W1) **Soft-write with EMA decay** (LRU-of-slots):
$$M_e^{(j)} = α_j · M_{e-1}^{(j)} + (1 - α_j) · v_e$$
where `v_e = W_v · [h_e; s_{τ_e}]` is the value to write and `α_j ∈ [0, 1]` is a per-slot decay (learnable; initialised to `α_j = 1 - 2^{-j-1}` for `j = 0, …, S-1` to span decay timescales geometrically).

(W2) **Hard-write with content-addressable replacement** (LRU-cache style):
$$j_e^* = \arg\min_j \|M_{e-1}^{(j)} - v_e\|^2 + \mathrm{age}(j) · η_{\text{age}}$$
$$M_e^{(j)} = \mathbb{1}[j = j_e^*] · v_e + \mathbb{1}[j \neq j_e^*] · M_{e-1}^{(j)}$$

W1 is differentiable end-to-end; W2 uses straight-through with the argmin replaced by Gumbel-softmax at train time. The default is W1 because (i) it's simpler, (ii) hard replacement risks information loss with no recovery path, (iii) the EMA decay structure subsumes both "always write the same slot" and "rotate through slots" as limiting cases.

Memory writes are **only** at event times. Between events, `M_e = M_{e-1}` — *frozen*.

### 3.4 Slot-memory read

The read is **attention from `s_{τ_e}` over the slots**:
$$α_e^{(j)} = \mathrm{softmax}_j\!\left(\frac{(W_q s_{τ_e}) · (W_k M_{e-1}^{(j)})}{\sqrt{d_k}}\right)$$
$$r_e = \sum_j α_e^{(j)} · (W_v M_{e-1}^{(j)})$$

with parameters `W_q ∈ ℝ^{d_k × d_s}`, `W_k, W_v ∈ ℝ^{d_k × d_m}`. The read cost is `O(S · d_k)`. With `S = 16` and `d_k = 64`, this is `~10³` ops per event, negligible compared to the deep-block forward.

The read of `M_{e-1}` (one event *before* the current event) is intentional — it preserves causality and lets the system "look at what it just wrote" without a circular dependency.

### 3.4.1 Why memory read at event e uses M_{e-1}, not M_e

The read happens *before* the write at the same event:
```
For event e:
  r_e = R(M_{e-1}, s_{τ_e})        # read from previous memory
  h_e = G_θ(h_{e-1}, s_{τ_e}, m_e, r_e)
  M_e = W(M_{e-1}, h_e, s_{τ_e})   # write happens after
```
This ordering ensures the event state `h_e` cannot read what it just wrote, preventing trivial copy behavior. It also makes the architecture exactly causal: the read at time τ_e depends only on data up to time τ_{e-1} (via M_{e-1}) plus the current default state. The write at time τ_e *adds* information from the current event. This matches the standard "read-modify-write" structure of memory networks (Sukhbaatar et al. 2015) and neural Turing machines (Graves et al. 2014).

### 3.5 Continuous-time view

PULSAR has a natural **piecewise-deterministic Markov process (PDMP)** interpretation. The default state `s_t` is continuous deterministic flow:
$$\frac{d s}{d t} = \log(A) · s + B · u(t),\qquad t \notin \{τ_k\}$$
At each event time `τ_k` (an atom of N), `(h, M)` undergoes a discrete jump:
$$\begin{aligned}
h \to & G_θ(h, s(τ_k), m_k, R(M, s(τ_k))) \\
M \to & W(M, h^{\text{new}}, s(τ_k))
\end{aligned}$$
This is exactly the structure of a PDMP (Davis 1984) with deterministic flow + jump kernel + jump-rate function `λ(t)`. The point-process / PDMP literature provides clean handles on the likelihood (Section 6) and stability (Section 7).

The PDMP framing also clarifies the relationship to existing models:
- **Mamba**: a PDMP with `λ ≡ ∞` (every step is an event) and the deep block degenerated to a selective scan.
- **LRU**: a PDMP with `λ ≡ 0` (never an event) and the deep block + memory absent.
- **MoD**: an *unstructured* sparse-compute model with `λ` per *layer*, not per *token-time*.

PULSAR's contribution is to put `λ` at the *token-time* level and unify the chunk / memory / compute decisions under it.

---

## 4. The event detector (the failure-prone component)

### 4.1 What the prior program failed at

The EALRMN Phase-0a–0g sequence (`EALRMN_PHASE0F_RESULTS.md` §3) established a pattern: *every gate in the architecture failed to specialise from the loss gradient alone.* Encoder gate (Phase-0a), recurrence (Phase-0c), memory-write gate (Phase-0d/e/f), attention readout (Phase-0g) — each required either an auxiliary loss with oracle labels (`--aux-class`, `--aux-gate`) or an architectural change to bootstrap out of the random-init basin.

The pattern is structurally a **bootstrap circularity**:
> X is useful given Y is trained; Y is useful given X is trained; under a unified end-to-end loss, neither moves first, so both stay at random init.

A naive PULSAR event detector that takes the same form — "supervise the event probability by the downstream task loss flowing back through the event branch" — would be a textbook case of this failure mode. The deep block produces useful information for the readout only if events fire on informative tokens; events fire on informative tokens only if the detector has learned what "informative" means; the detector has no signal until the deep block produces useful information. Identical to Phase-0d/e/f.

### 4.2 The endogenous-surprisal residual

PULSAR's event detector is supervised by a **self-generated** signal that does *not* depend on the deep block's quality:

For each step `t`, compute two cheap predictions:
- `p_t^{\text{def}} = \text{softmax}(W_d · s_t)` — the default stream's next-token distribution.
- `p_t^{\text{ema}}` — an EMA, over training steps, of the *full* model's next-token distribution at step `t` of the same position.

Define the **surprisal residual**:
$$ζ_t := \mathrm{KL}(p_t^{\text{ema}} \| p_t^{\text{def}}) - \bar{ζ}$$
where `\bar{ζ}` is a running mean. `ζ_t` is large at positions where the default stream is much *worse* than the (slower-moving) full model — i.e. positions where the deep block + memory is doing real work.

The event detector logit is:
$$η_t = w_ζ · ζ_t + g_φ(s_t, h_{e(t-1)}, M_{e(t-1)})$$
where `g_φ` is a learnable MLP. The crucial term is `w_ζ · ζ_t` — a hand-set positive weight (`w_ζ = 1` typically) that gives the detector a *non-zero initial signal* without any oracle labels. At init the default stream is random, so `p_t^{\text{def}}` is uniform, `ζ_t ≈ const · I_{x_t = \text{argmax}\, p_t^{\text{ema}}}`. Even at init this is *not* zero and *is* discriminative.

**Why this isn't another bootstrap circularity.** The signal `ζ_t` is computed from the *default stream's* prediction quality, not the event stream's. The default stream trains on a *standard cross-entropy* loss (see Section 5) — its training is *not* gated by the event detector. So the supervision signal for the event detector is *exogenous to the event detector itself*. This breaks the circular dependency.

More precisely: at training step `n`, the default stream's parameters move toward better `p_t^{\text{def}}`. The EMA `p_t^{\text{ema}}` moves toward the full model's `p_t^{\text{full}}`. Both are functions of the *current* parameters; both have well-defined non-degenerate gradients independent of the event branch. The KL between them is a well-defined scalar for every `t`.

### 4.3 Two regimes of `ζ_t`

(R-α) **Default-stream-undertrained.** Early in training, `p_t^{\text{def}}` is poorly fit, so `ζ_t` is large *everywhere*. The detector fires on most tokens. The deep block trains under high event rate. This is fine — at this stage the deep block is the only path that's getting meaningful gradient anyway.

(R-β) **Default-stream-well-trained.** Late in training, `p_t^{\text{def}}` fits well on easy tokens and poorly on hard ones. `ζ_t` becomes *sparse* — large only at tokens where the default stream genuinely can't predict. The detector specialises to fire on those tokens. **This is the regime where PULSAR earns its R7 compute-adaptive claim**: events fire where compute is needed.

The transition (R-α) → (R-β) is the actual training trajectory. The event rate `ρ = K/T` is *predicted to start high (0.7–1.0) and decay to a steady-state value (0.05–0.2)*. This is a *testable* prediction (Section 7, Claim N2).

### 4.4 Discretisation and gradient

`e_t ~ Bern(σ(η_t))` is non-differentiable. PULSAR uses **Gumbel-sigmoid with annealed temperature**:
$$e_t = \sigma\!\left(\frac{η_t + g_t - g_t'}{\tau}\right),\quad g_t, g_t' \sim \mathrm{Gumbel}(0,1)$$
where `τ` is annealed from `τ = 1.0` at start to `τ = 0.1` by end of training. `e_t ∈ (0, 1)` at training time; for the event-conditional operations (deep block, memory write) we use straight-through hard rounding `e_t^{\text{hard}} = \mathbb{1}[e_t > 0.5]` in the forward pass and `e_t` in the backward.

The Gumbel noise injects exploration; the annealed temperature lets the gate sharpen over training. This is the standard recipe from Jang et al. 2017, Maddison et al. 2017, and Raposo et al. 2024 (MoD uses Gumbel-Top-K).

### 4.5 What PULSAR's detector is *not*

It is *not*:
- **Surprisal of the model's own output** (the byte-latent transformer / Pagnoni et al. 2024 patching uses this). PULSAR uses surprisal of the *default stream specifically*, not the full model. This isolates the signal from the event branch.
- **Entropy of the prediction** (the SpaceByte / dynamic-patching family uses this). Entropy is monotone in uncertainty; surprisal-vs-ema captures *uncertainty relative to what could be predicted*. The difference matters when the model is on a long low-entropy stretch (entropy is small, but if the EMA full model is even smaller, `ζ_t` is small too — no event).
- **A separate predictor head trained on event labels** (this is what EALRMN Phase-0f's `--aux-class` does with oracle labels). PULSAR's detector has no oracle labels.

### 4.6 Anti-collapse regulariser

To prevent `ρ → 0` or `ρ → 1` collapses, PULSAR adds a Lagrangian-dual regulariser on the event rate:
$$\mathcal{L}_{\text{rate}} = \lambda · (\bar{p} - ρ^*)^2$$
where `\bar{p} = (1/T) Σ_t σ(η_t)` is the empirical fire rate, `ρ^*` is a target rate (a hyperparameter, e.g. 0.1), and `λ` is a *learnable dual variable* that adapts to keep `\bar{p}` near `ρ^*`. The dual update is:
$$\lambda \leftarrow \mathrm{ReLU}(\lambda + η_λ · (\bar{p} - ρ^*))$$
which pushes `λ` up when `\bar{p} > ρ^*` (suppressing events) and down when `\bar{p} < ρ^*` (encouraging events). This is the constrained-optimisation version of the rate penalty used in MoD (Raposo et al. 2024 §3.2).

A weaker version that is simpler and almost as effective: a *fixed* `λ` set so that `\bar{p}` enters [0.05, 0.5] within 1000 steps. This is the version we will test first; the dual is a robustness addition.

### 4.6.1 Connection to existing literature on adaptive computation

The endogenous-surprisal-residual approach has precedents but no exact duplicate in the LM literature. Closest cited work:

- **Byte-Latent Transformer / SpaceByte** (Pagnoni et al. 2024): uses *entropy of the predictive distribution* as a chunking signal. PULSAR's `ζ_t` is more discriminative because it normalises against an EMA of the *full model's* output, so a uniformly-uncertain stretch (e.g., random tokens) doesn't trigger events — events fire only when the *default stream specifically* is worse than the model average.
- **PonderNet** (Banino et al. 2021): a halting mechanism with learned per-token confidence. The halting probability is supervised by a balanced commitment loss; PULSAR's rate regulariser plays the analogous role. PULSAR differs in maintaining two parallel streams rather than a single "ponder until confident" path.
- **Adaptive Computation Time** (Graves 2016): a sigmoid halting probability summed across steps. Same family as PonderNet. PULSAR replaces "halt" with "expand", and uses the surprisal residual rather than a halting head.
- **Confidence-gated MoE** (Du et al. 2022, GLaM): per-token expert routing. PULSAR's event detector is *across-time* routing rather than across-expert routing. The two are orthogonal — a future "PULSAR + MoE" stack would have per-event expert routing on top of event-time gating.

The novelty in PULSAR's detector design is not the per-token gating *per se* (well-explored) but the *self-distillation supervision via an EMA of the full model's output*. We have not found this specific recipe in the cited literature; if it exists, the framework absorbs it as a special case and the novelty downshifts to "PULSAR is a system that combines [name] with point-process structure".

### 4.7 Why this story is more robust than EALRMN's

EALRMN had three gates (encoder write, memory write, attention read), each with its own bootstrap problem. The Phase-0g result (`EALRMN_PHASE0G_RESULTS.md` §1.3) showed that even *fixing* the gate (via `--aux-gate`) did not unlock the architecture because the *readout* was also bootstrap-broken.

PULSAR has *one* gate (the event detector). Once it works, the deep block + memory get correctly sparse supervision and train as normal supervised modules. There is no second-order bootstrap chain. The risk is concentrated at one mechanism — but that mechanism has an explicit non-circular supervision signal (`ζ_t`) and an explicit anti-collapse regulariser.

**This is the load-bearing design decision of the framework.** If the endogenous surprisal residual doesn't bootstrap the detector, the framework fails and we report a clean negative result. We will *not* add `--aux-class`-style oracle supervision — doing so would replicate EALRMN's "fix one bottleneck, reveal next one upstream" pattern (Phase-0f §2).

---

### 4.8 Theorem (informal): non-stalling of the detector under endogenous supervision

**Setup.** Let `(θ^{(n)}, φ^{(n)})` be the parameters at training step `n`. Let the default-stream parameters `(A, B, W_d, C)` train under `\mathcal{L}_{\text{def}}` only (β > 0). Let the event-detector parameters `(g_φ, w_ζ)` train under `\mathcal{L}_{\text{LM}} + γ \mathcal{L}_{\text{rate}}` only.

**Claim.** If at init `\nabla_{W_d} \mathcal{L}_{\text{def}} \neq 0` (which holds whenever the data has any predictive structure), then `\nabla_{η_t} (\mathcal{L}_{\text{LM}} + γ \mathcal{L}_{\text{rate}}) \neq 0` at every step `n ≥ 1`, with `ζ_t` providing a non-zero baseline of the detector's decision variable.

**Sketch of argument.** Under the auxiliary loss `\mathcal{L}_{\text{def}}`, the default stream trains, producing a non-uniform `p_t^{\text{def}}`. The EMA `p_t^{\text{ema}}` is a slow average of the *full* model's output, which (after at least 100 EMA steps) is non-uniform too. The KL divergence `\mathrm{KL}(p_t^{\text{ema}} \| p_t^{\text{def}})` is therefore data-dependent and non-zero at every t. This adds a non-degenerate bias `w_ζ · ζ_t` to `η_t`, ensuring `σ(η_t) ≠ 0.5` at random init even before the gradient through `g_φ` flows.

**Caveat.** This is not a proof that the detector *converges* — it's a proof that it doesn't *stall* at random init. The convergence to a useful policy depends on the LM loss gradient through the deep block. The non-stalling property is the property EALRMN lacked at each of its gates (Phase-0e §B.3); PULSAR has it by construction.

---

## 5. Reduction to existing methods

### 5.1 All events off → LRU

Set `w_ζ = 0`, `g_φ ≡ -∞` (or simply force `η_t = -10^6`). Then `σ(η_t) = 0` for all t, no events fire (K = 0), the event branch is dead. Default stream alone produces logits:
$$\text{logits}_t = U · s_t + V · h_{e(t)} = U · s_t + V · h_0 = U · s_t + \text{const}$$
This is exactly LRU with one extra constant added per readout (a bias). The constant has no effect on the cross-entropy minimiser. **PULSAR-with-no-events = LRU.**

VESTA Claim B0 (`newmodel.txt` §FALSIFIABLE CLAIMS): "linear recurrence + orthogonal init beats tanh RNN by ≥ 100× val_loss on the needle task at T=2048 m=1024." With the all-events-off reduction, PULSAR satisfies this iff LRU does, which (per `VESTA_AUDIT.md`) it does on LRA tasks at similar scales. Reproducing Claim B0 with PULSAR-no-events is therefore a *direct* test of infrastructure correctness and an indirect test of the recipe's stability.

### 5.2 All events on → dense deep transformer

Force `η_t = +10^6` so `σ(η_t) = 1` for all t. Then every step is an event:
$$h_t = G_θ(h_{t-1}, s_t, m_t, r_t),\quad t = 1, …, T$$
This is a transformer that *also* maintains a separate LRU stream. With the LRU stream removed (set B = 0, A = 0), we get pure transformer + slot memory + cross-attention. The slot memory updates every step → essentially a 16-slot constant-size memory cache. The compute is `O(T · d_h²)` — dense.

**PULSAR-with-all-events = a memory-augmented dense transformer.** Not identical to vanilla transformer (the slot memory and cross-attention to LRU are extra) but in the same complexity class.

This is the "expressivity upper bound" — if the framework is well-designed, the all-on configuration should never lose to a vanilla dense transformer. We verify this in the ablation (Section 7).

### 5.3 Fixed-stride events → MambaByte-like patching

Set `η_t = +10^6` if `t mod S = 0` else `-10^6`. Events fire at fixed stride S. The model effectively chunks the sequence at fixed boundaries, processes each chunk's last token with the deep block. With the deep block degenerated to identity, this is MambaByte-style fixed-stride patching (with byte-level x).

In ablation flag terms: `--events-fixed-stride S` is the "MambaByte" comparator. Claim N1 (Section 7) requires *learned* events to beat fixed-stride events at iso-FLOPs.

### 5.4 Events per layer → Mixture-of-Depths

If we have multiple deep blocks `G_θ^{(1)}, …, G_θ^{(L)}` stacked, and we let each have its *own* event detector `η_t^{(ℓ)}`, then per-layer, per-token, the model selects a subset of tokens to pass through the deep block, exactly like MoD's top-k routing. PULSAR with separate per-layer detectors → MoD.

The PULSAR commitment is to have **one** detector that fires at the same `t` for all deep layers — chunk boundaries are shared across depth. This is more restrictive than MoD but more architecturally honest (the "event" is a property of the *sequence position*, not of *layer × position*). The R7 question (Section 7, Claim N1) is whether this restriction loses or gains over MoD.

### 5.5 No memory → MoD + LRU stream

Set `S = 0` (no slot memory). Then the read/write paths are dead. PULSAR collapses to: LRU default + per-event deep block, with the deep block reading from the LRU state but no persistent slot memory. This is structurally similar to Hawk (`VESTA_AUDIT.md` row Griffin/Hawk) except that the deep-block activation is sparse in time.

### 5.6 Diagram

| `w_ζ` | `g_φ` | `S` | resulting model |
|------|------|-----|-----------------|
| 0    | -∞   | 0   | **LRU** (Claim B0 baseline) |
| 0    | +∞   | 0   | LRU + dense deep transformer (memory-augmented stack) |
| 0    | fixed-stride | 0 | **MambaByte-style** fixed-stride chunking |
| trainable | trainable | 0 | LRU + event-triggered deep block (Hawk-like) |
| trainable | trainable | 16  | **Full PULSAR** |
| per-layer | per-layer | 0 | **MoD** |

All six configurations are reachable by setting ablation flags. The strong baselines in the VESTA audit are special cases of PULSAR under specific flag choices. This is the property the framework needs in order to *isolate* the contribution of the learned point process specifically.

---

## 6. Objective

### 6.1 Components

PULSAR's training objective is:
$$\mathcal{L}_{\text{PULSAR}}(θ, φ) = \alpha \mathcal{L}_{\text{LM}} + \beta \mathcal{L}_{\text{def}} + \gamma \mathcal{L}_{\text{rate}} + \delta \mathcal{L}_{\text{entropy}}$$

- `\mathcal{L}_{\text{LM}}` — the **primary cross-entropy loss** on the *full* readout `logits_t = U·s_t + V·h_{e(t)}`. This is the autoregressive next-token loss. `α = 1` (set by convention).
- `\mathcal{L}_{\text{def}}` — an **auxiliary cross-entropy** on the default-stream-only readout `p_t^{\text{def}}`. This makes the default stream by itself a useful next-token predictor, which (a) gives the `ζ_t` signal a meaningful baseline, (b) provides a fall-back for inference if events are turned off, (c) prevents the model from offloading all prediction to the event path.
- `\mathcal{L}_{\text{rate}}` — the **anti-collapse regulariser** from Section 4.6. Encourages `ρ` to stay near `ρ^*`. Set `\rho^* = 0.1` as default; can sweep `\rho^* ∈ {0.05, 0.1, 0.2}`.
- `\mathcal{L}_{\text{entropy}}` — a **gate-entropy regulariser** that encourages the gate distribution `σ(η_t)` to become *bimodal* late in training (near 0 or near 1, rare in middle). Defined as:
$$\mathcal{L}_{\text{entropy}} = -\frac{1}{T}\sum_t \left[\sigma(η_t) \log σ(η_t) + (1-σ(η_t)) \log(1-σ(η_t))\right]$$
This is the *negative* entropy summed; minimising this *minimises* gate entropy, encouraging bimodality. Annealed from 0 to a small positive value over training. (Inspired by the bimodal-gate target of `VESTA_AUDIT.md`'s implicit success criterion.)

### 6.2 Default coefficients

- `α = 1`
- `β = 0.5` (default-stream auxiliary CE)
- `γ = 0.1` (rate regulariser; the dual variable absorbs further adjustment)
- `δ = 0.01 · \mathrm{sched}(t)` with `\mathrm{sched}(t) = \max(0, (t - t_0) / (T_{\text{end}} - t_0))` ramping from 0 at `t_0 = 5000` steps to 1 at `T_{\text{end}}` (avoid forcing bimodality before the gate has explored).

### 6.3 Ablation flags

Each loss component has a flag:
- `--no-aux-def` → `β = 0` (no default-stream CE; expect the default stream to degenerate)
- `--no-rate-reg` → `γ = 0` (no rate regulariser; expect either ρ→0 or ρ→1)
- `--no-bimodal` → `δ = 0` (no entropy regulariser; expect gate to stay around 0.5 throughout)

Plus architectural flags:
- `--events-off` → `w_ζ = 0, g_φ ≡ -∞` (reduces to LRU; tests Claim B0)
- `--events-on` → `η_t = +∞` (reduces to dense; tests expressivity upper bound)
- `--events-fixed-stride S` → fixed-stride MambaByte mode
- `--slots S` → memory size (set 0 to disable memory)
- `--deep-block-depth d` → number of transformer layers in `G_θ` (1 to 4)
- `--moid` → use per-layer event detectors (reduces to MoD-like routing)

Each flag is independently ablatable. The B0 ablation (events off) tests infrastructure. The other ablations test which component of PULSAR is doing the work.

### 6.4 The dual-Lagrangian for rate

The rate regulariser implements a soft constraint `\bar{p} = ρ^*`. The dual form is:
$$\mathcal{L}_{\text{rate}} = \lambda (ρ^* - \bar{p})$$
with `\lambda ≥ 0` updated by ascent:
$$\lambda \leftarrow \max(0, \lambda + \eta_\lambda (\bar{p} - ρ^*))$$
This gives stronger control than the squared penalty (it tracks the *direction* of the constraint violation linearly rather than quadratically). The dual `λ` is a single scalar so the additional state cost is negligible.

The dual variant is the *default* for PULSAR because it provides hard-ish constraint enforcement without requiring the gradient of the rate term to be exactly zero at `ρ^*`. If `\bar{p}` drifts above `ρ^*`, `λ` increases, suppressing events.

### 6.5 Gradient sources

The event detector `η_t` receives gradient from:
1. The LM loss via the deep block (sparse — only at events).
2. The rate regulariser (always present).
3. The bimodal-entropy regulariser (always present).
4. Implicitly via `ζ_t`, but `ζ_t` itself is a constant w.r.t. `η_t` (only depends on default stream parameters and EMA). So `ζ_t` does not provide *gradient* — it provides a non-zero baseline to `η_t`.

This is the resolved version of the bootstrap problem: `ζ_t` is an *additive bias* on `η_t` that's already non-zero at init. The detector's gradient lives in `g_φ` and is bounded by the regularisers. The detector cannot stall at random init because its decision variable is *not* at random init — it's at `w_ζ · ζ_t` which is data-dependent and non-zero.

---

### 6.6 Loss decomposition table for ablation reporting

The framework's objective decomposes as follows for reporting per-mechanism contributions:

| Symbol | Component | Default coef | Ablation flag | Role |
|--------|-----------|--------------|----------------|------|
| `\mathcal{L}_{\text{LM}}` | Full-model cross-entropy | α = 1 | (always on) | Primary task loss |
| `\mathcal{L}_{\text{def}}` | Default-stream-only CE | β = 0.5 | `--no-aux-def` (β=0) | Bootstrapping the default stream → enables `ζ_t` signal |
| `\mathcal{L}_{\text{rate}}` | Lagrangian rate constraint | γ = 0.1 | `--no-rate-reg` (γ=0) | Prevents `\bar{p}` collapse/saturation |
| `\mathcal{L}_{\text{entropy}}` | Gate-distribution bimodal | δ = 0.01 (annealed) | `--no-bimodal` (δ=0) | Encourages bimodality late in training |
| (architectural) | event-detector active | n/a | `--events-off` | Reduces to LRU |
| (architectural) | slot memory size | S = 16 | `--slots N` (any N) | Reduces to no-memory (S=0) variant |

Per-loss-term measurement protocol (per `newmodel.txt` §MATHEMATICAL FORMULATION):
- Train with the full loss, record best val_loss `L_{\text{full}}`.
- For each term k, retrain with that term's coef set to 0, record `L_{-k}`.
- Report `Δ_k := L_{-k} - L_{\text{full}}` as the *measured* contribution of term k.

This is the iso-everything ablation. The prior EALRMN program (`newmodel.txt` E1) was criticised for *not* doing this — specifying many mechanisms as a single object without per-term measurement. PULSAR commits to per-term measurement up front.

---

## 7. Pre-committed claims (three, falsifiable)

### Claim N1 — R7 compute-adaptive on heterogeneous-compute task

**Statement.** On task T7 (heterogeneous-compute: some tokens trivial, some hard) at iso-FLOPs, PULSAR's event-triggered deep-block routing beats Mixture-of-Depths (Raposo et al. 2024) by ≥ 0.04 nat val_loss at T = 4096, m_active = 256.

**Task design.** T7 = a "mixed-task stream" where each contiguous block of `[B_low, B_high]` (e.g., [16, 64]) tokens is drawn from one of two regimes:
- **Easy regime**: deterministic continuation of a simple pattern (modular increment, copy of a fixed string). The default stream alone should fit this with low loss.
- **Hard regime**: a multi-step compositional task — e.g., "given a sequence of (key, value) pairs in the block prefix, answer a query token". The default stream cannot fit this without memory; the deep block + memory should.

The mixed stream contains ~70% easy regime by token count and ~30% hard. The token-level *compute-required* signal is therefore non-uniform.

**Iso-FLOPs.** Both PULSAR and MoD are constrained to compute exactly the same FLOPs per forward pass on average. For PULSAR this means setting `ρ^* = ρ_{\text{MoD}}^{\text{equiv}}` so that `T · (c_{\text{def}} + ρ^* · c_{\text{event}}) = T · c_{\text{MoD}}`. The deep-block params + memory params are matched at iso-active-parameter count.

**Baselines.**
- MoD (Raposo et al. 2024) at top-k = 1, capacity-factor matched.
- LRU (PULSAR with events off; this is the "default-only" lower bound).
- Dense transformer at the same total params (an upper bound that pays full FLOPs).

**Metric.** Val cross-entropy at iso-step (so MoD and PULSAR train for the same step budget); also reported at iso-wall-clock and iso-FLOPs-spent.

**Threshold.**

| Observation | Interpretation |
|-------------|---------------|
| PULSAR beats MoD by ≥ 0.10 nat | **Strong support** for N1. Event-time routing > layer-time routing on heterogeneous compute. |
| PULSAR beats MoD by 0.04–0.10 nat | **Support.** Margin matches threshold. Conclusion: event-time routing is non-trivially better. |
| PULSAR within ±0.04 nat of MoD | **Inconclusive**. The two routing granularities are equivalent on this task. Falls back on "PULSAR's R6 advantage if any". |
| PULSAR loses to MoD by 0.04+ nat | **N1 falsified.** Event-time routing is the wrong granularity. MoD-style per-layer routing is preferred. PULSAR's R7 claim is dead. |
| PULSAR's event rate doesn't converge to a stable regime | **N1 untestable**; debug N2 first. |

**Confounds.**
- (a) Active-parameter mismatch: PULSAR's deep block + slot memory may have more total params than MoD's expert. Resolution: enforce iso-active-param at d_h and report.
- (b) Optimisation fragility: prior literature (`VESTA_AUDIT.md` §3.6, Zoology) reports linear-RNN family models with narrow LR windows. Resolution: sweep LR over 4 decades; report best.
- (c) Task too easy: if the default stream alone reaches loss < 0.5 on the hard regime, the routing is irrelevant. Resolution: tune hard-regime difficulty until LRU-baseline val_loss is at least 0.5 nat above the dense-baseline val_loss.
- (d) Seed variance: ≥ 5 seeds; report mean ± sd.
- (e) Event rate hits the rate-regulariser ceiling: if `ρ̄` hits `ρ^*` and stops, the routing learned is dominated by the regulariser, not the data. Resolution: also report the run with `γ = 0` and check N1.

### Claim N2 — Event detector trains without circularity

**Statement.** With *no* auxiliary supervision (no oracle event labels, no `--aux-class`, no `--aux-gate`), the event detector trained by `\mathcal{L}_{\text{LM}} + \mathcal{L}_{\text{def}} + \mathcal{L}_{\text{rate}}` reaches a steady state where (i) the event rate `\bar{p}` is in [0.01, 0.5] for ≥ 80% of training steps after step 2000, and (ii) the gate-distribution becomes bimodal (entropy ≤ 0.3 nats per gate) by end of training.

This is the **falsification of bootstrap-circularity** for PULSAR's specific construction.

**Task.** A standard byte-level next-token loss on a *natural* stream (e.g., Pile-CC bytes) — *not* a synthetic task — for 50k steps at m = 256.

**Diagnostic measurements.**
- `\bar{p}(t)` = running event rate at training step `t`.
- `H(σ(η_t))` = average per-gate entropy at training step `t`.
- `\mathrm{corr}_t(ζ_t, σ(η_t))` = correlation across token positions between the surprisal residual and the gate firing prob — should grow from 0 at init to ≥ 0.5 by mid-training.

**Threshold.**

| Observation | Interpretation |
|-------------|---------------|
| `\bar{p} ∈ [0.01, 0.5]` for ≥ 80% of steps, gate-entropy < 0.3 by end, corr(ζ, σ(η)) ≥ 0.5 | **Strong support** for N2. The detector trains without circularity. Bootstrap-circularity is a model-specific failure mode of EALRMN, not a general one. |
| `\bar{p} ∈ [0.01, 0.5]`, gate-entropy in [0.3, 0.5], corr < 0.5 | **Partial support.** Detector trains but doesn't fully specialise. The architecture works but the supervision signal could be sharper. |
| `\bar{p}` collapses to <0.01 OR saturates to >0.5 (sustained > 1000 steps) | **N2 falsified, mode "collapse"**. The endogenous surprisal residual does *not* provide enough signal to keep the detector in the live regime. Either the rate regulariser is needed (test with stronger `γ`) or oracle supervision is required (restores EALRMN's pattern). |
| Gate-entropy stays > 0.6 throughout (gate stuck at 0.5) | **N2 falsified, mode "stall"**. Same failure as EALRMN Phase-0e. The framework's central design decision was wrong. Report and pivot. |
| `\bar{p}` oscillates between regimes without convergence | **Inconclusive.** Optimisation issue (LR too high or dual variable too aggressive). Sweep and re-test. |

**Confounds.**
- (a) `\mathcal{L}_{\text{def}}` is providing the actual supervision, not `ζ_t`: test with `β = 0`. If the gate still trains, N2 was supported by the wrong mechanism.
- (b) The rate regulariser is forcing `\bar{p}` into range regardless of signal: test with `γ = 0`. If the rate goes to 0 or 1, the regulariser was load-bearing — N2 is still supported but the signal isn't from `ζ_t` alone.
- (c) The bimodal regulariser is forcing entropy down: test with `δ = 0`. If entropy stays high, N2 (gate specialisation) is supported only with the regulariser.

These three tests *jointly* discriminate: does the model train when `\mathcal{L}_{\text{rate}} = 0` AND `\mathcal{L}_{\text{def}} = 0` AND `\mathcal{L}_{\text{entropy}} = 0`? If yes, the endogenous signal is sufficient. If no, the framework is *additively* using rate / aux / entropy supervision, which is a more honest claim than "supervision-free" but is also weaker.

### Claim N3 — B0 replication with events off

**Statement.** PULSAR with `--events-off` reproduces VESTA Claim B0: linear recurrence + orthogonal init beats tanh RNN by ≥ 100× val_loss on the needle task at T = 2048, m = 1024.

**Task.** Needle-in-haystack from `EALRMN_PHASE0D_RESULTS.md` (§Task), but at the longer T = 2048, larger m = 1024 specification. Two KV pairs inserted at random positions in a filler stream; query at end.

**Baselines.**
- Tanh RNN at same total parameter count.
- LRU (a separate, well-tuned LRU implementation).
- PULSAR with `--events-off`.

**Threshold.**

| Observation | Interpretation |
|-------------|---------------|
| PULSAR-events-off beats tanh RNN by ≥ 100× val_loss | **Strong support** for N3. Infrastructure works; LRU recipe is correctly implemented. |
| Beats by 10–100× | **Support.** Recipe is essentially right; tweaks to init / LR may matter. Continue but flag. |
| Beats by 1–10× | **Weak support / inconclusive.** Something is wrong with the infrastructure. Debug before proceeding. |
| Beats by < 1× or loses | **N3 falsified — DEBUG INFRASTRUCTURE.** Per `newmodel.txt`, this is non-optional. |
| Tied with separate LRU implementation | Healthy sign — confirms PULSAR-no-events ≡ LRU as derived in §5.1. |
| PULSAR-no-events significantly worse than separate LRU | Indicates the PULSAR readout (`U · s_t + V · h_0`) is interfering. Bug or design problem. Fix before proceeding. |

**Confounds.**
- (a) Different tokeniser / different needle task between PULSAR and the LRU reference: enforce identical task implementation.
- (b) Tanh RNN initialisation: per `EALRMN_PHASE1_GPU_RESULTS.md`, tanh RNN with orthogonal init does better than tanh with Xavier; we use tanh + Xavier as the strawman baseline (per VESTA brief's strawman definition).
- (c) Seed selection: ≥ 10 seeds; report median and worst-case.

---

### 7.4 Pre-committed measurement protocol summary

Each claim has the structure:
1. Pre-registered task design (specified above).
2. Pre-registered baseline (named model + spec).
3. Pre-registered metric (val cross-entropy).
4. Pre-registered threshold (specific Δ in nats).
5. Pre-registered interpretation table (Table in claim section, ≥5 rows).
6. Pre-registered confound enumeration (named confounds with mitigations).
7. Pre-registered action on each outcome (passed / partial / failed).

This is the methodology refinement explicitly demanded by `newmodel.txt` §FALSIFIABLE CLAIMS. We commit to publishing the pre-registration *before* running the experiments; any deviation from pre-registration is reported as a deviation in the results writeup. (Following `newmodel.txt` §ITERATION DISCIPLINE.)

### 7.5 What we do NOT pre-commit (open questions)

These are deliberately *not* claims, because (a) we don't know the answer and (b) failing to predict them shouldn't falsify the framework:

- The optimal value of `ρ^*` (event rate target). We expect 0.05–0.2 but treat this as a hyperparameter sweep.
- The optimal value of `S` (slot count). We expect 8–32 but sweep.
- The deep-block depth `d_G` (number of transformer layers in G_θ). We expect 2–4 but sweep.
- The right basis for the EMA cache when V is large (top-K? sketched? per-position centroids?). Defer to Phase 4 if scaling.
- Whether W1 (soft) or W2 (hard) write is preferred. Treat as a config flag and run both.

These are the *hyperparameters* of the framework, not its *commitments*.

---

## 8. Implementation sketch

### 8.1 File layout in `glades-ml`

```
Backend/Machine Learning/
├── Networks/
│   ├── pulsar.h                        # PULSAR network class (extends NNetwork)
│   ├── pulsar.cpp                      # constructor, forward dispatcher
│   ├── sgd_pulsar.cpp                  # PULSAR training (default + event branches)
│   ├── pulsar_infer.cpp                # PULSAR inference (event-conditional)
│   ├── pulsar_event_detector.h/.cpp    # logit_intensity, gumbel-sigmoid, ema
│   ├── pulsar_default_stream.h/.cpp    # LRU diagonal complex recurrence
│   ├── pulsar_event_stream.h/.cpp      # deep block; uses transformer_ops.h
│   ├── pulsar_slot_memory.h/.cpp       # M_e write/read, both W1 and W2 variants
│   ├── pulsar_kernels.h                # gather/scatter for sparse events
│   ├── pulsar_kernels_cpu.cpp          # CPU reference impl of all kernels
│   ├── pulsar_kernels_cuda.cu          # CUDA kernels (deferred until CPU-prototype works)
│   └── pulsar_public_api.h             # C++98-stable wrapper for inference
├── MLStructure/
│   └── pulsar_info.h                   # PulsarInfo (config + architecture)
└── DataObjects/                        # (no new data types; reuses TokenInput)

unit-tests/Backend/Machine Learning/
├── pulsar_unit_test.cpp                # event-detector parity, full-pass sanity
├── pulsar_events_off_test.cpp          # verifies --events-off = LRU exactly
└── pulsar_smoke_test.cpp               # end-to-end smoke run

research/
├── VESTA_CANDIDATE_C.md                # this document
├── pulsar_phase0_cpu.cpp               # CPU-only prototype (~1500 LOC C++98)
├── pulsar_phase1_n3_b0.cpp             # B0-replication smoke test
├── pulsar_phase2_n2_natural.cpp        # natural-data event-rate test
└── pulsar_phase3_n1_t7.cpp             # heterogeneous-compute T7 test
```

### 8.2 LOC budget

| Component | LOC | Notes |
|-----------|-----|-------|
| `pulsar_default_stream` (LRU) | 200 | Diagonal complex linear RNN, FFT-free per-step form |
| `pulsar_event_detector` (incl. Gumbel-sigmoid, EMA, rate dual) | 250 | The novel logic |
| `pulsar_event_stream` (deep block) | 100 | Wrapper around existing transformer block |
| `pulsar_slot_memory` (W1 + W2) | 300 | Both write variants |
| `pulsar_kernels_cpu` (gather/scatter, sparse routing) | 400 | Per-token mask logic |
| `pulsar_kernels_cuda` (CUDA versions) | 600 | Sparse scatter on GPU |
| `sgd_pulsar` (backward, gradient accumulation) | 500 | Most failure-prone part |
| `pulsar_phase0_cpu` (prototype + task gen + main) | 1500 | Mirrors `ealrmn_phase0f_aux.cpp` structure |
| unit tests | 400 | parity, smoke |
| | **~4250** | |

This is comparable to the CASCADE candidate's LOC (`candidate_C_local.md` ~6000 LOC) and the EALRMN GPU prototype (`research/ealrmn_gpu` ~3000 LOC per memory). Realistic for a 6-week implementation budget.

### 8.3 Kernel inventory

**CPU kernels (always required):**
- `pulsar_lru_step`: complex-diagonal linear-RNN step. Costs O(d_s).
- `pulsar_event_detect`: compute η_t, sample Gumbel-sigmoid, return e_t. Costs O(d_s · d_η).
- `pulsar_deep_block_forward`: invoke G_θ on event indices only. Variable-shape — uses gather/scatter.
- `pulsar_slot_write_w1`: EMA decay slot update. Costs O(S · d_m).
- `pulsar_slot_read_attn`: attention from `s_{τ_e}` over slots. Costs O(S · d_k).
- `pulsar_event_index_build`: compute event indices from `e_t` mask. Costs O(T) prefix-sum.

**CUDA-specific (deferred for prototype):**
- Sparse gather: collect `s_{τ_e}` into a contiguous batch for the deep block.
- Sparse scatter: write `h_e` back into a sequence-indexed cache.
- Persistent-kernel event loop (advanced; only if performance demands).

The CUDA implementation is **deferred until the CPU prototype validates Claim N3 and Claim N2**. If either fails, no CUDA work is undertaken. This is the safe path informed by `EALRMN_PHASE0G_RESULTS.md` §Final verdict ("Further experimental phases would either need substantially larger scale or task changes to potentially flip the verdict"); we will *not* build GPU infrastructure for a falsified design.

### 8.4 Numerical considerations

- **Complex LRU state.** Stored as interleaved `(s^R, s^I)` in a single `Vector<float>`. The diagonal multiplication is implemented as two real diagonal mults + crosses, avoiding any complex-arithmetic library dependency.
- **EMA of prediction distribution.** `p_t^{\text{ema}}` is a 32-bit float vector of size V. EMA update is `p_t^{\text{ema}} ← 0.99 · p_t^{\text{ema}} + 0.01 · p_t^{\text{full}}`. Stored *per position t* for the sequence under training — this is `T · V` floats. At T = 4096 and V = 256 this is 4 MB; manageable. (For larger V the EMA would need to be approximated by a smaller representation — e.g., top-k logits or a centroidal sketch.)
- **Gumbel-sigmoid stability.** At temperature `τ → 0` the gate can saturate gradients. Mitigation: clip the gradient through the gate to [-10, 10] (standard MoE trick).
- **Determinism.** The Gumbel noise is drawn from `glades::rng::*` per the project policy (`CLAUDE.md` §Determinism). The event indices are a function of the seed; reproducibility is preserved.

### 8.5 Variable-shape handling

Sparse events produce variable per-batch event counts `K_b` for batch element `b`. Two implementation approaches:

(A) **Padded mask** — allocate a worst-case K_max events per batch, pad the rest with masked-out positions. Costs O(T) memory but uniform shape. Simpler to implement.
(B) **Gather/scatter** — compact the event positions into a `(\sum_b K_b)`-length tensor for the deep block. More efficient but requires per-batch prefix-sum and careful index bookkeeping.

The CPU prototype uses (A). The CUDA implementation uses (B). The deep-block forward/backward is the same in both cases; only the input/output marshalling differs.

### 8.6 Backward pass

Most novel part of the implementation:

1. The default stream's gradient flows through `s_t` for every t. Standard LRU backward.
2. The event stream's gradient flows through `h_e` and the slot memory; this is the standard transformer + slot-memory backward, but only at event indices.
3. The event detector's gradient comes from three sources:
   - **`\mathcal{L}_{\text{LM}}`**: via the readout `V · h_{e(t)}` → `∂\mathcal{L}_{\text{LM}}/∂ e_t` requires differentiating through the discrete event indicator. Use the Gumbel-sigmoid pathway (`∂σ((η+g-g')/τ)/∂η`).
   - **`\mathcal{L}_{\text{rate}}`**: `∂\mathcal{L}_{\text{rate}}/∂η_t = γ · (dual `λ`) · σ'(η_t) / T`.
   - **`\mathcal{L}_{\text{entropy}}`**: `∂\mathcal{L}_{\text{entropy}}/∂η_t = δ · (\log\frac{σ(η_t)}{1-σ(η_t)}) · σ'(η_t) / T`.
4. The `ζ_t` term contributes a *constant* w.r.t. `η_t` only via `g_φ` (the learnable part); the `w_ζ · ζ_t` bias is a hard-coded scalar.

A correct backward through Gumbel-sigmoid with straight-through is the implementation hot spot; standard reference is Jang et al. 2017 §3.2 (categorical reparam).

---

### 8.7 Test plan (unit + smoke)

Following the `glades-ml` test framework conventions (`CLAUDE.md` §Test Framework):

```cpp
// unit-tests/Backend/Machine Learning/pulsar_unit_test.cpp
void PULSARUnitTest() {
  // 1. Default-stream LRU step parity vs reference.
  ASSERT("LRU step matches reference within 1e-5",
         test_lru_step_parity(seed=42, d_s=64));

  // 2. Event detector with all w_ζ=0, g_φ=zero, η=0 → σ(0) = 0.5 exactly.
  ASSERT("event detector at zero init produces 0.5 fire rate",
         test_event_rate_zero_init() == 0.5);

  // 3. With --events-off: forward output bit-matches a pure LRU model.
  ASSERT("events-off == LRU bit-exact",
         test_events_off_eq_lru());

  // 4. With --events-on: per-event computation is invoked T times.
  ASSERT("events-on calls deep block T times",
         count_deep_block_invocations() == T);

  // 5. Slot memory W1: EMA decay matches closed-form.
  ASSERT("W1 EMA matches alpha^n * M_0 + (1-alpha^n)/(1-alpha) * v",
         test_w1_ema_closed_form());

  // 6. Backward through Gumbel-sigmoid: gradients sum to zero w.r.t. fixed loss.
  ASSERT("Gumbel-sigmoid backward gradient parity",
         test_gumbel_sigmoid_grad_parity());
}
```

The test cases mirror the `nn-recurrent` and `nn-transformer` tests already in the unit-tests suite. The most important test is #3 — `--events-off` exactly recovers LRU — because this is the load-bearing reduction for Claim N3 infrastructure.

---

## 9. Failure modes (three most likely)

### 9.1 Event-rate collapse / saturation

**Symptom.** During training, `\bar{p}` either drops to <0.01 (no events fire, model degenerates to LRU + dead branch) or saturates to >0.5 (every token an event, model is a dense transformer paying full cost). The intermediate "sparse but live" regime [0.01, 0.5] is never sustained.

**Why this is the worst risk.** Both Claim N1 and Claim N2 require sustained sparsity. If the rate collapses or saturates, the entire framework loses its R7 advantage and the comparison to MoD becomes trivial (PULSAR runs at dense cost; obviously loses on FLOPs).

**Diagnostic signature.**
- `\bar{p}(t)` plot: not monotone decay to 0 or rise to 1, but *drifts unstably*.
- Gate-entropy `H(σ(η_t))` plot: gate distribution monomodal at 0 or 1.
- Loss curve: looks like LRU baseline (collapse) or dense (saturation), not intermediate.

**Mitigations (in order of escalation):**

1. **Stronger rate regulariser.** Increase `γ` to 1.0 or 10.0. Lagrangian dual variable will compensate.
2. **Symmetric dual constraint.** Add a *lower-bound* dual variable so the dual penalises `\bar{p} < ρ^*` symmetrically. This prevents collapse.
3. **Hard clamping.** If `\bar{p}` drops below 0.01, set `η_t ← η_t + Δ` with `Δ > 0` for the next 100 steps to revive event firing. (Brittle, last resort.)
4. **Anneal `w_ζ`.** Start with high `w_ζ` (5–10) to force strong endogenous signal at init; decay to 1 over training so the learnable `g_φ` takes over later.
5. **Hard target rate.** Replace the soft Lagrangian with a *Gumbel-Top-K with K = ρ^* · T*. This guarantees exactly `ρ^*` events per sequence by construction. (Less expressive — can't have variable event rates per sequence — but eliminates collapse.)

The **Gumbel-Top-K with K = ρ^* T** fallback is the *safe-default ablation*: if the soft rate regulariser fails, we drop to Top-K hard routing (identical in spirit to MoD's top-k mechanism). This guarantees PULSAR can match MoD on routing but with event-time (instead of layer-time) granularity. The cost is losing the "learned event rate" novelty — PULSAR with Top-K is just "MoD at sequence level" instead of "MoD at layer level".

**Pre-committed action on this failure.** If we cannot sustain `\bar{p} ∈ [0.01, 0.5]` with the soft regulariser by training step 5k, switch to Gumbel-Top-K with K fixed. Document the switch as a falsification of "auxiliary-free rate control" but maintain the architectural test of "event-time vs layer-time routing" (Claim N1) on the Top-K variant.

### 9.2 Bootstrap circularity (the EALRMN pattern repeats)

**Symptom.** The event detector never fires meaningfully — the gate stays at 0.5 across the entire sequence regardless of `ζ_t`. The endogenous surprisal residual is not providing enough signal because the default stream is *also* at random init and its surprisal residual is noise.

**Why this is the second-worst risk.** Claim N2 fails. The framework reduces to the EALRMN pattern: we add auxiliary supervision (oracle labels) to bootstrap the gate, which restores the "fix one bottleneck, reveal next" trajectory and the framework loses its honesty.

**Diagnostic signature.**
- `corr(ζ_t, σ(η_t))` stays at zero across training.
- Gate-entropy stays at log(2) ≈ 0.693 throughout.
- N2's confound test (training with `β = 0`) shows the detector training comes entirely from `\mathcal{L}_{\text{def}}`, not from `ζ_t`.

**Pre-committed action.** If the detector does *not* train without `\mathcal{L}_{\text{def}}` (β = 0), then the endogenous surprisal residual is *not* doing the work, and N2 is falsified in the strong form. We accept the weaker form: PULSAR's detector trains with `\mathcal{L}_{\text{def}}` auxiliary supervision. This is *not* oracle supervision (no task-specific labels) so it's still an architectural improvement over EALRMN — but the framework's central novelty claim is downgraded.

**Mitigation paths if even `\mathcal{L}_{\text{def}}` doesn't unlock training:**

1. **Schedule the default stream first.** Train the default stream alone (`α = 0`, `β = 1`, no event branch) for 5k steps until the EMA `p_t^{\text{ema}}` is meaningful. Then unfreeze the event branch and train end-to-end.
2. **Use a more informative endogenous signal.** Replace KL(ema || def) with KL(target || def) where target is the actual one-hot true distribution. This is just per-token CE — not a circular signal. But it tells us where the default stream is doing poorly, which is what we wanted.
3. **(Last resort)** Allow a single round of *non-task-specific* auxiliary supervision: e.g., a small head that predicts the local entropy of the next token, supervised against the entropy of `p_t^{\text{ema}}`. This is *closely related* to the surprisal-based byte-latent-transformer (Pagnoni et al. 2024) approach. It is *not* oracle supervision but it is an auxiliary task. Document as a partial defeat.

### 9.3 Memory write redundancy / slot collapse

**Symptom.** The S slots in the memory all converge to similar contents — the model writes the same `v_e` everywhere, or only one slot is ever read with high attention weight, or the slot decay rates `α_j` all converge to the same value.

**Why this matters.** Claim R6 (memory-write regularisation, the no-baseline axis from `VESTA_AUDIT.md` §2) is implicit in PULSAR's slot-write design — the write happens only at events, which is a structural form of "write regularisation". If the slots collapse, PULSAR is functionally a single-slot memory, which is a single accumulator, which is *less* expressive than the LRU state. The R6 axis is dead.

**Diagnostic signature.**
- `\mathrm{rank}(M_e)` ≪ S (slots are linearly dependent).
- Attention weights `α_e^{(j)}` peak at a single `j*` regardless of query.
- `\sigma(\alpha_j^{\text{decay}})` all converge to the same value.

**Mitigations:**

1. **Orthogonality regulariser on `M`.** Add `\|M_e^T M_e - I\|_F^2` to the objective. Forces slot diversity.
2. **Disjoint slot initialisation.** Initialise `α_j` at geometric spacing (already in §3.3) to bias slots toward different timescales.
3. **Slot dropout.** Randomly mask out slots during training so the model can't rely on a single slot.
4. **Pre-committed action.** If the slots collapse, report it as a partial falsification of R6's "more slots are useful" implicit claim. PULSAR's R6 advantage disappears; the R7 advantage may still stand. Report cleanly.

---

## 10. What PULSAR is *not* claiming

To avoid the "quietly drop the claim" failure mode (`newmodel.txt` E5), here is the explicit non-claim list:

- PULSAR is **not claiming** to beat Mamba-2 or Hawk on standard long-context LM at iso-param.
- PULSAR is **not claiming** state-tracking expressivity beyond LRU (the deep block helps but doesn't structurally fix the non-solvable-group limitation; see `VESTA_AUDIT.md` §3.5).
- PULSAR is **not claiming** to solve multi-query associative recall (MQAR). Slot memory has only S slots; MQAR at scale needs O(KV-pairs) memory. Based (Arora et al. 2024) is the right comparator there; PULSAR is orthogonal.
- PULSAR is **not claiming** to be auxiliary-free in the strict sense. `\mathcal{L}_{\text{def}}` is auxiliary; the rate regulariser is auxiliary. The honest claim is "no *oracle-label* auxiliary supervision is required", which is strictly weaker than EALRMN's failed claim.
- PULSAR is **not claiming** a generic LLM efficiency gain — only an iso-FLOPs *win on tasks with heterogeneous compute density*. If the data stream is uniform-difficulty (e.g., random binary noise), there is no win.

---

## 11. Pre-committed phase plan

**Phase 0 — CPU prototype (1-2 weeks).** Build `research/pulsar_phase0_cpu.cpp` modelled after `ealrmn_phase0f_aux.cpp`. Implement: LRU default stream, point-process detector with endogenous `ζ_t`, deep block (use a 2-layer transformer from `glades-ml`'s existing transformer stack), slot memory W1 + attention read. Smoke-test on a tiny needle task.

**Phase 1 — Claim N3 (B0 replication) (3-5 days).** Run PULSAR with `--events-off` on needle T=2048 m=1024 against tanh RNN. Verify ≥100× win. *Gate: if fail, debug LRU implementation before proceeding.*

**Phase 2 — Claim N2 (event detector trains) (1 week).** Run PULSAR on a *natural* byte stream (Pile-CC subset or similar) for 50k steps. Track `\bar{p}`, `H(σ(η))`, `corr(ζ, σ(η))`. *Gate: if `\bar{p}` collapses or saturates, run the mitigation cascade in §9.1. If N2 fails in strong form, report and either pivot or accept weak form.*

**Phase 3 — Claim N1 (R7 win on T7) (1-2 weeks).** Design T7 task as in §7-N1. Run PULSAR, MoD, LRU, dense. Compare at iso-FLOPs and iso-step. *Gate: if PULSAR within ±0.04 nat of MoD, N1 is inconclusive — report as such. If PULSAR loses, report as honest negative and document that event-time vs layer-time routing is not architecturally significant.*

**Phase 4 (optional, if N1–N3 all support) — Scale to m=1024.** Re-run all three claims at larger m on GPU. This is where the C++ implementation goes from `research/` prototype to `glades-ml` proper.

The phase plan **explicitly authorises early termination** if a claim is falsified. The prior EALRMN program ran 8 phases past its first major falsification (`EALRMN_PHASE0_RESULTS.md` Phase-0a/0b); PULSAR commits to terminating after Phase 2 if N2 is falsified, and after Phase 3 if N1 is falsified.

---

## 12. Comparison to Candidates A and B

(For triangulation; from outside the parallel-design constraint these are independently-developed siblings.)

Candidate A (non-diagonal structured recurrence, expressivity-first) attacks R3 by replacing the LRU's diagonal `A` with a structured non-diagonal matrix. PULSAR is *complementary* — PULSAR's default stream is a vanilla LRU; if Candidate A wins, the PULSAR default stream can adopt Candidate A's recurrence as a drop-in replacement.

Candidate B (causal information-bottleneck with multi-particle state, R2-first) attacks R2 by replacing the single recurrent state with a particle filter over states. PULSAR is *also complementary* — PULSAR's default stream is a single state, but the design is agnostic to that choice. A future "PULSAR + B" hybrid would have a particle-filter default stream and an event-triggered deep block. The R2 + R7 combination is unexplored.

Candidate C (this document, R1 + R6 + R7) attacks the *resource-allocation* axis. The three candidates together cover the four axes (R1, R2, R3, R6, R7) with no overlap; if all three support their respective claims, a stacked model is conceivable for VESTA Phase 2.

The three candidates are *not* competing — they are testing orthogonal hypotheses. PULSAR's specific contribution to the joint design space is **the point-process resource allocator**.

---

## 13. Mathematical summary table

| Quantity | Definition | Cost |
|----------|-----------|------|
| `s_t ∈ ℂ^{d_s/2}` | LRU state | `O(d_s)` per step |
| `η_t ∈ ℝ` | event-detector logit | `O(d_s · d_η)` per step |
| `e_t ∈ {0,1}` | Gumbel-sigmoid sample | `O(1)` per step |
| `h_e ∈ ℝ^{d_h}` | event state | `O(d_h^2)` per event |
| `M_e ∈ ℝ^{S × d_m}` | slot memory | `O(S · d_m)` write/read per event |
| `r_e ∈ ℝ^{d_m}` | attention read | `O(S · d_k)` per event |
| `ζ_t ∈ ℝ` | endogenous surprisal residual | `O(V)` per step (KL between two distributions) |
| `\bar{p} ∈ [0,1]` | empirical event rate | `O(T)` per sequence |
| Total FWD cost | | `T · (c_{def} + ρ · c_{event}) ≈ T · (d_s + ρ · d_h^2)` |
| Total memory | | `O(T · d_s + K · d_h + S · d_m + T · V)` (the last is the EMA cache) |

The EMA cache `O(T · V)` is the largest persistent training-time memory after parameters; at byte-level V = 256, T = 4096 this is 4 MB — negligible. At V = 32000 (subword) and T = 16384 it's ~2 GB; mitigate by storing top-K logits or a sketched form. For the prototype phases (Phase 0-3) we use byte-level so this is not a concern.

---

## 14. Conclusion

PULSAR is a deliberate consolidation of three of VESTA's eight reference axes (R1, R6, R7) into a single mechanism — a learned marked point process. The unification is motivated by the EALRMN program's evidence that multiple independent gates cause cascading bootstrap-circularity failures (`EALRMN_PHASE0F_RESULTS.md` §3 four-bottleneck taxonomy). One gate, with one explicit non-circular supervision signal (the endogenous surprisal residual `ζ_t`), is the architectural commitment.

The framework reduces by ablation to LRU (events off — Claim N3, the VESTA infrastructure check), MambaByte (fixed-stride events — the R1 baseline), and MoD (per-layer events — the R7 baseline). Each baseline is reachable by a single flag, ensuring isolation experiments can be run cleanly.

Three claims are pre-committed:
- **N1** (R7 win on heterogeneous-compute T7 task vs MoD) — the main *novel* claim.
- **N2** (event detector trains without oracle-label supervision) — the *falsifiability* claim.
- **N3** (events-off recovers LRU baseline B0) — the *infrastructure* claim.

Three failure modes are pre-committed with mitigation paths: event-rate collapse, bootstrap-circularity repeat, and slot-memory collapse. The worst-feared failure mode is the first; the safe-default mitigation is to fall back to a Gumbel-Top-K hard router, which preserves the architectural test of "event-time vs layer-time routing" even if learned-rate fails.

C++ implementation cost is ~4250 LOC over 3-6 weeks single-developer wall, with the CPU prototype phase explicitly gating CUDA work on Claim N3 success. The framework will not deploy GPU resources to validate a falsified design.

If all three claims clear, PULSAR earns its R1/R6/R7 contribution and is suitable for VESTA Phase 2 scaling. If N2 fails in strong form, PULSAR is downgraded to a "supervised-but-not-oracle-supervised" framework and remains comparable to MoD. If N1 fails, PULSAR's R7 claim is dead and we report cleanly.

— end VESTA Candidate C —
