# Paradigm Shift #59 — Candidate C: MDL-PRETRAIN

**Status:** REJECTED
**Date:** 2026-05-08
**Iteration context:** post-iter-200, "bigger picture" candidate sweep
**Verdict:** Mathematically interesting reframing; practically equivalent to existing cross-entropy + weight decay. Does not meet the bar for paradigm-shift status.

---

## 1. Executive Summary

MDL-PRETRAIN proposes to replace the standard cross-entropy (CE) pretraining loss with a Minimum Description Length (MDL) objective, framing language model training as joint compression of the data **and** the model parameters. The candidate is grounded in solid information-theoretic foundations (Rissanen 1978; Grünwald 2007) and has aesthetic appeal: the "right" way to train a model from a coding-theory perspective is to minimize total bits-to-describe.

After analysis, however, MDL-PRETRAIN is **not** a candidate that warrants paradigm-shift treatment for #59. The honest assessment is:

- The practical realization of MDL on dense LLM weights collapses, up to additive constants and scalar reweighting, to **cross-entropy + weight decay** (or, with different priors, cross-entropy + L1 or hierarchical Gaussian regularization).
- Modern pretraining recipes already use weight decay (typically AdamW-style decoupled L2). The vocabulary is different ("MDL" vs. "regularization"), but the optimization problem is the same up to a constant scaling.
- More principled MDL realizations (NML, Bayesian mixture codes, two-part codes with adaptive precision) require either intractable normalizers, posterior approximations that re-introduce existing methods (variational inference, Laplace), or per-step overhead that swamps the theoretical benefit at LLM scale.
- The places where MDL **does** offer a non-trivial bit on top of CE+WD (e.g., model-selection across architectures, automatic complexity control, principled stopping) are not where transformer pretraining bottlenecks live in 2026.

For these reasons, **REJECT MDL-PRETRAIN as paradigm shift #59**. The expected practical speedup or quality improvement over the current `--mfio 2 --wip-K 4 --face 1 --slc --rlg --kahan-v --sparec` flagship stack is at best within noise, and almost certainly within the band that a single weight-decay coefficient sweep would already cover.

The remainder of this document documents why, partly to record the reasoning and partly so a future iteration that revisits MDL (e.g., for architecture search or for principled width/depth selection) can pick up where this analysis stops.

---

## 2. MDL Theory

### 2.1 The Rissanen formulation

Rissanen's original Minimum Description Length principle (1978) states that the best model from a family is the one that minimizes the **total description length** of the data:

$$
\mathrm{MDL}(D) \;=\; L(D \mid \theta) \;+\; L(\theta)
$$

where $L(D \mid \theta)$ is the number of bits needed to encode the data $D$ given the model $\theta$, and $L(\theta)$ is the number of bits needed to encode the model itself. This is the **two-part code** form. The information-theoretic identity $L(D \mid \theta) = -\log_2 P_\theta(D)$ makes the connection to log-likelihood explicit:

$$
\mathrm{MDL}(D) \;=\; -\log_2 P_\theta(D) \;+\; L(\theta).
$$

For autoregressive language models with sequence $y_{1:T}$ given context $x_{<t}$, the per-token form is:

$$
\mathrm{MDL}_{\text{per-token}} \;=\; -\sum_{t} \log_2 P_\theta(y_t \mid x_{<t}) \;+\; \frac{L(\theta)}{T}.
$$

Up to a $\log_2 e$ scaling, this is exactly the natural-base cross-entropy loss plus a complexity term divided by token count.

### 2.2 The Grünwald refinement

Grünwald (2007) and the later "modern MDL" literature distinguish three flavors:

1. **Crude two-part codes** (Rissanen 1978): pick a discretization for $\theta$ and pay $L(\theta)$ bits to specify it. Equivalent to MAP under a discrete prior.
2. **Mixture codes** (Bayesian MDL): $-\log \int P_\theta(D) \pi(\theta) d\theta$. Equivalent to Bayesian model evidence.
3. **Normalized Maximum Likelihood (NML)**: $-\log P_{\hat\theta(D)}(D) + \log \sum_{D'} P_{\hat\theta(D')}(D')$. Optimal in the worst-case regret sense but typically intractable.

Variants (1) and (2) are the only ones with any chance of being computed at LLM scale. Variant (3)'s normalizer requires summing over all possible token sequences and is hopeless for $V \approx 50\text{k}$ and $T = 1024$.

### 2.3 What MDL gives you that CE alone doesn't

The advertised benefits of MDL over plain cross-entropy:

- **Automatic complexity control:** the $L(\theta)$ term penalizes unused capacity, so MDL "self-regularizes" model size.
- **Model selection without held-out data:** MDL is consistent (under regularity conditions) for selecting the true model class as $T \to \infty$ even on the training set.
- **Generalization guarantees:** PAC-Bayes-style bounds connect MDL to generalization gap.
- **Compression as a quality metric:** a model that achieves shorter MDL is provably better at compressing the data.

These are real properties. The question is whether any of them translate into a practical training-time advantage for transformer pretraining at the 1.84 B-parameter scale we currently operate at.

---

## 3. Practical Implementation: What MDL-PRETRAIN Reduces To

### 3.1 The reduction to CE + weight decay

To turn the abstract MDL functional into something a CUDA kernel can compute, we need a concrete encoding for $L(\theta)$. The standard choices:

**Choice A: Gaussian prior, fixed precision.**
Encode each weight $\theta_i$ relative to a Gaussian $\mathcal{N}(0, \sigma^2)$. Then
$$
L(\theta) \;=\; \tfrac{1}{2\sigma^2}\sum_i \theta_i^2 \;+\; \text{const.}
$$
Plug this into the MDL objective, divide by token count $T$, multiply by $\ln 2$ to get nats:
$$
\mathcal{L}_{\mathrm{MDL}} \;=\; \mathcal{L}_{\mathrm{CE}} \;+\; \frac{\lambda}{2}\,\|\theta\|_2^2,\qquad \lambda \;=\; \frac{\ln 2}{T \sigma^2}.
$$

This is **L2-regularized cross-entropy with weight decay coefficient $\lambda$**. Identical to AdamW's decoupled weight decay term up to a per-coordinate learning-rate factor, identical to plain SGD-with-WD up to the AdamW vs. coupled-WD distinction.

**Choice B: Laplace prior.**
Encode each weight relative to $\mathrm{Laplace}(0, b)$:
$$
L(\theta) \;=\; \tfrac{1}{b}\sum_i |\theta_i| \;+\; \text{const}.
$$
Yields **L1-regularized cross-entropy**. Standard since Tibshirani 1996.

**Choice C: Mixture-of-Gaussians or "spike-and-slab" prior.**
Encode each weight as drawn from a mixture, where one component is concentrated at zero. Yields a sparsity-inducing regularizer that has been studied extensively (sparse coding, Bayesian neural nets, variational dropout). Closer to non-trivial, but the practical realizations (e.g., Molchanov et al. 2017's variational dropout) are an existing, well-studied technique — not a new paradigm.

**Choice D: Universal codes for floats.**
Use a universal integer code (Elias gamma, omega) on the bit-level representation of bf16/fp32 weights. This is the "true" minimum-bits encoding. In practice it gives a per-weight cost of a few bits regardless of magnitude and does not differentiate cleanly, so it cannot be used as a training loss term — it only scores models post-hoc.

**Choice E: Bayesian mixture / variational MDL.**
Replace the point estimate $\theta$ with a posterior $q(\theta)$, minimize $-\mathbb{E}_{q}[\log P_\theta(D)] + \mathrm{KL}(q \| \pi)$. This is the **ELBO** of variational inference. Existing technique (Hinton & van Camp 1993; Blundell et al. 2015; the entire variational-Bayes-NN literature). Practically too expensive at 1.84 B params unless we restrict $q$ to factorized Gaussians, in which case we recover variational dropout / Bayes-by-Backprop with their well-known scaling difficulties.

### 3.2 What "MDL-PRETRAIN" would actually look like in glades

A faithful glades implementation of MDL-PRETRAIN (Choice A, the practical default) would be:

1. Add a `--mdl-pretrain` flag.
2. In the loss reduction, after `cross_entropy_nll_loss`, accumulate `0.5 * lambda * sum(theta^2)` across all weight tensors.
3. In the backward, add `lambda * theta` to the gradient before Adam's `m, v` update.
4. Expose `--mdl-lambda` (defaulting to a value calibrated against the current implicit weight decay).

Step 3 is **literally what AdamW already does** as decoupled weight decay (`theta -= lr * wd * theta` after the Adam step, mathematically equivalent to gradient-time L2 with a factor that absorbs lr). The glades trainer already passes a weight-decay coefficient through the Adam update kernel.

So MDL-PRETRAIN as Choice A is `--weight-decay` with a different name and a different theoretical justification.

### 3.3 What about the per-token term divided by $T$?

The MDL objective divides $L(\theta)$ by $T$. As $T$ grows, the relative weight of the complexity term decreases, which is the MDL prescription: with more data, less regularization. In a fixed-$T$ pretraining run this is just a constant absorbed into $\lambda$. In a multi-epoch / variable-$T$ training schedule, MDL would prescribe **annealing weight decay down as more tokens are seen** — an interesting prescription that, again, is already a known empirical technique (cosine-decay weight decay, e.g., in Llama recipes) and not a paradigm shift.

The genuinely novel piece would be: anneal $\lambda(t)$ on a schedule **derived from MDL theory** rather than hand-tuned. The expected gain over a well-tuned cosine WD schedule is small and would require an A/B at 1.84 B to measure.

### 3.4 Beyond Choice A: NML and the elephant in the room

The information-theoretically "correct" MDL is NML (Choice 3 above). For language models, NML's normalizer is

$$
\log \sum_{y_{1:T} \in \mathcal{V}^T} P_{\hat\theta(y_{1:T})}(y_{1:T})
$$

which requires fitting a separate model on every possible token sequence and summing. The Stochastic Complexity / approximate-NML literature gives upper bounds (BIC, $\frac{k}{2}\log T$ as the leading term), but these collapse to a constant times "number of parameters times log token count" — again something already well-understood and roughly captured by existing scaling-law machinery.

There is no tractable NML-style MDL-PRETRAIN at LLM scale.

---

## 4. Comparison to Existing CE + Regularization Training

### 4.1 What current glades / typical 2026 pretraining already does

The current glades flagship (`--mfio 2 --wip-K 4 --face 1 --slc --rlg --kahan-v --sparec`) uses:

- Cross-entropy loss with NLL reduction (`cross_entropy_nll_loss` GPU kernel).
- AdamW-style update with weight decay (the `adam_update_*` kernels accept a wd term).
- LR schedule (cosine decay, warmup; recently `--lr-decay` for continuation).
- FACE Zipfian regularization on embeddings (a structured, frequency-aware regularizer; arguably a stronger MDL-flavored term than uniform L2).
- SLC / RLC curriculum scheduling.

The FACE shift is interesting in this context: FACE's Zipfian-regularization mechanism is **already** a frequency-prior MDL realization for the embedding table. The mechanism EMPIRICALLY VALIDATES the MDL idea for one specific weight group (embeddings), where the prior structure (Zipfian token frequencies) is genuinely informative. FACE works because the prior is non-trivial and matches data structure.

**Plain MDL-PRETRAIN with a Gaussian prior on dense FFN/attention weights does not have an analogous structural prior to exploit.** The Gaussian prior is uninformative; the resulting regularizer is L2 with a coefficient.

### 4.2 The honest comparison table

| Property | Standard CE + AdamW-WD | MDL-PRETRAIN (Choice A) | FACE (paradigm #28) |
|---|---|---|---|
| Per-step compute overhead | baseline | identical (WD is already there) | small Adam-state save |
| New optimizer state | none | none | $K$ FACE buckets (1008–1570× compression) |
| Regularizer structure | uniform L2 | uniform L2 | Zipfian, frequency-aware |
| Theoretical grounding | empirical | MDL / Bayesian | empirical + Zipfian structure |
| Empirical validation at 1.84 B | yes (current flagship) | not run; expected within-noise | yes; ~1.13 nat peak, 0.4–0.8 sustained |
| Ships a new compute kernel | no | no | yes (FACE bucket update) |
| Paradigm-shift status | n/a (baseline) | **rejected here** | accepted (#28) |

The only column where MDL-PRETRAIN differs from the baseline is "Theoretical grounding" — and that is not a column on which paradigms ship.

### 4.3 Could MDL-PRETRAIN be tuned to win?

A weight-decay sweep at the 1.84 B scale, holding everything else constant, can swing eval loss by something on the order of 0.05–0.2 nats depending on how badly tuned the baseline was. If the current glades $\lambda$ is poorly chosen, an "MDL-PRETRAIN" rebrand that re-derives $\lambda$ from theory could appear to win. But the win is a weight-decay sweep, not a paradigm shift. The honest move is to sweep $\lambda$ and report it as a hyperparameter tune.

---

## 5. Honest Recommendation: REJECT for #59

### 5.1 Where MDL-PRETRAIN fails the paradigm-shift bar

A glades paradigm shift (per the FACE / SLC / RLG / SAS pattern in `MEMORY.md`) clears one or more of:

- Independently-validated 1.5×+ wall-clock speedup (SLC, RLG).
- Independently-validated quality improvement of $\geq 0.4$ nats sustained (FACE).
- Eliminates a category of optimizer/state cost (FACE embeddings; iter-172's MFIO Wq/Wk/Wv skip).
- Enables a previously-infeasible regime (RLG mid-training depth growth; SLC long-context refinement).

MDL-PRETRAIN clears **none** of these. Its mechanism collapses to weight decay; weight decay is already in the trainer; no new compute kernel ships; no new optimizer-state class is eliminated.

### 5.2 Comparison to other rejected paradigms

The pattern of rejections in `MEMORY.md` informs the decision:

- **#36 KV-FACE:** rejected post Gate-0 because the Zipfian-concentration premise didn't hold for attention Q/K weights. MDL-PRETRAIN is weaker — its premise (that L2 regularization is a paradigm) is already conceded as established practice.
- **#41 ASTRA:** rejected post Gate-0 (catastrophic divergence) and after VRAM accounting showed the new accumulator cost equaled the eliminated state. MDL-PRETRAIN does not even reach the VRAM-accounting stage because there is nothing new to accumulate.

In both cases, the rejection bar was a Gate-0 probe. MDL-PRETRAIN does not warrant a Gate-0 probe because the analytic reduction to AdamW already settles the question.

### 5.3 What would change the recommendation

MDL-PRETRAIN could become interesting if any of the following held:

1. A **structurally informative prior** beyond Gaussian/Laplace was identified for dense transformer weights (analogous to what FACE does for embeddings using Zipfian frequencies). Candidates: low-rank priors (already #7 Stiefel × Σ); spectral priors (already studied in spectral-norm regularization); sparsity priors (already studied in pruning / lottery ticket).
2. A **tractable approximation to NML** at LLM scale was discovered. None currently exists; the research literature has been working on this for two decades without a clean answer.
3. The objective was **architecture-search over models**, not weight-update during training. MDL is genuinely useful for choosing $L$, $d_{model}$, $d_{ff}$, vocab size — but that is a model-selection use case, not a pretraining-loss use case. If glades adds an automated arch-search component, MDL-as-selection-criterion may earn its own paradigm slot.
4. A **per-layer or per-tensor adaptive $\lambda$** was derived from MDL theory and showed gains over uniform AdamW WD in an ablation. This is the most plausible salvage; it would be paradigm shift "Adaptive-WD-from-MDL", not "MDL-PRETRAIN".

None of these conditions hold today. The candidate as proposed should be rejected.

### 5.4 What to do with the salvageable pieces

The two pieces of MDL-PRETRAIN that are worth preserving for future iterations:

- **MDL-as-WD-schedule-derivation:** the prescription that $\lambda \propto 1/T$ (decaying with token count) is principled and matches existing cosine WD schedules. Worth holding as a small "principled WD schedule" idea in `FUTURE_PARADIGM_CANDIDATES.md`, not as #59.
- **MDL-as-model-selection:** if/when glades adds NAS or architecture sweeping (e.g., for choosing CHIRON depth or $d_{model}$ at a given VRAM budget), MDL provides a clean, training-set-only selection criterion that does not require held-out data. Worth a paragraph in `BEYOND_CHIRON.md` or the equivalent forward-looking document.

Neither of these warrants paradigm-shift #59 status.

---

## 6. Final Verdict

**REJECT MDL-PRETRAIN as paradigm shift #59.**

Reasoning, in one sentence: MDL-PRETRAIN's practical realization at LLM scale is mathematically equivalent to AdamW with weight decay, which is already in the trainer; the theoretical reframing does not unlock any new compute kernel, optimizer-state elimination, or quality regime that meets the paradigm-shift bar.

The candidate is preserved here as a documented rejection so a future iteration that revisits MDL — most plausibly for adaptive per-tensor weight-decay scheduling, or for architecture-search selection criteria — can build on this analysis instead of rediscovering the same reduction.

---

## References

- Rissanen, J. (1978). *Modeling by shortest data description.* Automatica, 14(5), 465–471.
- Grünwald, P. D. (2007). *The Minimum Description Length Principle.* MIT Press.
- Hinton, G. E., & van Camp, D. (1993). *Keeping neural networks simple by minimizing the description length of the weights.* COLT.
- Tibshirani, R. (1996). *Regression shrinkage and selection via the lasso.* J. Royal Stat. Soc. B.
- Blundell, C. et al. (2015). *Weight uncertainty in neural networks.* ICML.
- Molchanov, D., Ashukha, A., & Vetrov, D. (2017). *Variational dropout sparsifies deep neural networks.* ICML.
- Loshchilov, I., & Hutter, F. (2019). *Decoupled weight decay regularization.* ICLR. (AdamW.)
- Internal: `MEMORY.md` (FACE, SLC, RLG, ASTRA, KV-FACE entries).
- Internal: `research/FACE_AS_DISRUPTING_PARADIGM.md`.
- Internal: `research/PARADIGM_41_ASTRA_DESIGN.md` (rejection precedent).
