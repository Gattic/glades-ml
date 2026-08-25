# KELLY-LM: Kelly-Style Allocation for LLM Training, Testing, and Inference

Kelly does not tell an LLM how to predict the next token. It tells an LLM system how aggressively to allocate scarce budget when every training run, evaluation campaign, and rollout can compound gains or compound regressions.

In this setting, the correct interpretation is not "maximize profit." It is:

- maximize validated model improvement per unit budget,
- avoid overcommitting compute or traffic to noisy ideas,
- make testing an explicit part of the uncertainty model,
- preserve long-run geometric growth of system utility rather than chasing short-run benchmark spikes.

So Kelly is not a direct optimizer for perplexity, accuracy, or latency. It is a growth-optimal control law for **validated LLM utility under multiplicative deployment risk**.

## Core Mechanism

Let \(U_t > 0\) denote validated system utility capital after control interval \(t\). This is not raw training loss. It is a strictly positive scalar built from the metrics the stack actually cares about, for example:

\[
S_t
=
\alpha_q Q_t
-\alpha_c C_t
-\alpha_l L_t
-\alpha_s H_t
-\alpha_r G_t
\quad\text{and}\quad
U_t = e^{S_t}
\]

where:

- \(Q_t\) is held-out quality or task utility,
- \(C_t\) is compute or serving cost,
- \(L_t\) is latency or throughput pressure,
- \(H_t\) is safety or policy-harm mass,
- \(G_t\) is regression debt, rollback pressure, or reliability loss.

At interval \(t\), allocate fractions \(w_t \in \mathbb{R}^n\) of the available budget across candidate actions \(a_1,\dots,a_n\):

- training actions: optimizer changes, data-mixture changes, architecture flags, checkpoint promotions,
- testing actions: eval expansion, canaries, adversarial suites, calibration passes,
- inference actions: model routing, decode policy, retrieval/tool policy, context-window policy.

Let \(R_{t+1} \in \mathbb{R}^n\) be the random validated return per normalized unit of budget over the next interval. Then:

\[
U_{t+1} = U_t \left( 1 + w_t^\top R_{t+1} \right)
\]

with pathwise drawdown:

\[
D_t = 1 - \frac{U_t}{\max_{s \le t} U_s}
\]

The Kelly objective is:

\[
\max_{w_t} \; \mathbb{E}_t \left[\log\left(1 + w_t^\top R_{t+1}\right)\right]
\]

If \(1 + w_t^\top R_{t+1} \le 0\), the rollout has invalidated the operating point. In production terms, that means a hard regression, SLA break, or safety failure severe enough that the step must be treated as absorbing failure, not as a normal noisy sample.

For small returns:

\[
g(w) \approx w^\top \mu - \frac{1}{2} w^\top K w
\]

where:

\[
\mu = \mathbb{E}[R],
\qquad
K = \Sigma + \Lambda + \Gamma
\]

and:

- \(\Sigma\) is aleatoric covariance from run-to-run or traffic noise,
- \(\Lambda\) is epistemic uncertainty from weak testing, low sample counts, or distribution shift,
- \(\Gamma\) is an operational penalty matrix for tail risk: OOMs, p99 latency, safety incidents, callback failures, or checkpoint corruption.

The unconstrained optimizer is:

\[
w^* = K^{-1}\mu
\]

In real systems the nonnegative, capped version is more appropriate:

\[
\max_{w \ge 0,\; \mathbf{1}^\top w \le 1}
\;
w^\top \mu - \frac{1}{2} w^\top K w
\]

That is the LLM-compatible Kelly rule: allocate budget where validated edge is high, but charge every action for noise, uncertainty, and operational fragility.

## LLM Interpretation of the Terms

### Training

A training action should be scored on **validated improvement**, not raw optimization progress. A reasonable return proxy is:

\[
R_i^{\text{train}}
=
\frac{
\Delta S_i^{\text{val}}
- \lambda_{\text{rollback}} \,\Delta B_i
}{
\text{budget}_i
}
\]

where \(\Delta S_i^{\text{val}}\) is held-out improvement in the composite utility score and \(\Delta B_i\) is regression or rollback burden created by the run.

This immediately rules out common overbetting mistakes:

- allocating by training loss instead of validation utility,
- treating one lucky seed as stable edge,
- scaling a new optimizer or architecture before covariance with existing experiments is understood,
- spending the full GPU week on a promising but weakly tested configuration.

### Testing

Testing is the main LLM-specific extension beyond the finance version. Evals do not usually create product utility directly. They reduce uncertainty in \(\mu\) and increase confidence in \(K\).

A practical diagonal model is:

\[
\Lambda_{ii}
=
\frac{c_1}{N^{\text{eval}}_i}
+ c_2 \,\text{coverageGap}_i
+ c_3 \,\text{shiftScore}_i
\]

where:

- \(N^{\text{eval}}_i\) is effective sample count,
- \(\text{coverageGap}_i\) measures missing scenario coverage,
- \(\text{shiftScore}_i\) measures distance from the data or traffic regime where the action was last validated.

So testing budget is valuable because it shrinks \(\Lambda\). Under-tested actions should not receive large allocations even if their recent mean looks strong.

### Inference

Inference returns are not "model quality only." They are net user-facing utility after latency, cost, and failure penalties:

\[
R_i^{\text{infer}}
=
\frac{
\lambda_q \Delta Q_i
- \lambda_l \Delta L_i
- \lambda_c \Delta C_i
- \lambda_s \Delta H_i
}{
\text{budget}_i
}
\]

This covers:

- traffic routing between small and large models,
- decode aggressiveness such as temperature or top-p,
- speculative decoding and cache policy,
- retrieval or tool-use policy,
- long-context activation and fallback behavior.

The rule says: do not route live traffic based on average benchmark gain alone. Route based on expected validated utility net of covariance with known failure modes.

## Why Overbetting Causes LLM Drawdown

The same geometric effect from finance appears here. If allocation exceeds Kelly, arithmetic improvement can still look positive while compounded validated utility degrades.

Under the scalar small-return approximation:

\[
g(f) \approx f\mu - \frac{1}{2}f^2 k
\qquad\Rightarrow\qquad
f^* = \frac{\mu}{k}
\]

where \(k\) is the scalar analogue of total risk penalty.

At roughly \(2f^*\),

\[
g(2f^*) \approx 0
\]

So doubling the growth-optimal rollout share already pushes long-run geometric improvement close to flat.

A concrete inference example:

- a new routing policy has estimated \(\mu = 0.01\) validated utility gain per unit traffic share,
- total risk penalty is \(k = 0.04\),
- Kelly traffic share is \(f^* = 0.25\).

Then:

\[
g(0.25) \approx 0.25(0.01) - \frac{1}{2}(0.25)^2(0.04) = 0.00125
\]

but:

\[
g(0.50) \approx 0.50(0.01) - \frac{1}{2}(0.50)^2(0.04) = 0
\]

Routing 50% of traffic looks only "2x more aggressive," but it has already spent the entire expected geometric gain on variance, shift risk, and operational fragility.

The same failure mode shows up in training:

- too much cluster time on one unproven recipe,
- too much trust in a single benchmark family,
- too much rollout share for a new serving path,
- too little eval coverage relative to the claimed improvement.

## Why the Vector Form Matters More in LLM Systems

The correct object is almost never scalar. It is:

\[
w^* = K^{-1}\mu
\]

because LLM actions are strongly correlated.

Examples:

- two "different" training recipes may share the same data-mixture blind spot,
- two checkpoints may fail on the same long-context or tool-use regime,
- two inference policies may both amplify the same tail-latency path,
- multiple benchmark wins may come from the same latent capability factor.

The vector rule helps because it:

- penalizes duplicate bets on the same underlying failure mode,
- converts shared blind spots into an explicit capital charge,
- de-levers actions whose apparent edge comes from the same narrow eval slice,
- avoids spending both GPU budget and live traffic on what is really one thesis.

This is especially important for transformer systems because training, testing, and serving are coupled. A weak eval protocol in testing can make both training allocation and inference rollout look better than they really are.

## What KELLY-LM Does Not Do

KELLY-LM does not directly optimize:

- next-token loss,
- benchmark score in isolation,
- maximum drawdown,
- tail latency in isolation,
- safety guarantees,
- architecture search by itself.

It is also not a substitute for:

- hard rollout caps,
- canary gates,
- adversarial evaluation,
- rollback machinery,
- kill switches,
- budget ceilings,
- human review where safety policy requires it.

If the literal objective is "minimize regressions at all cost," the optimum is again trivial: allocate less and eventually allocate nothing. Kelly remains an anti-overcommitment rule, not a zero-risk rule.

## Why Fractional Kelly Is Mandatory

Full Kelly is too aggressive for modern LLM work because the main error is not market noise. It is estimation error:

- benchmark leakage,
- narrow eval coverage,
- seed sensitivity,
- traffic nonstationarity,
- regime shifts after data or prompt changes,
- coupled infra failures,
- delayed regressions that appear only after deployment scale.

So production should use:

\[
w_t = \eta \,\hat{K}_t^{-1}\hat{\mu}_t
\qquad\text{with}\qquad
0 < \eta < 1
\]

plus shrinkage of \(\hat{\mu}\), floors on \(\hat{K}\), and hard caps on any single action.

Under the scalar approximation:

\[
g(\eta f^*) \approx \left(\eta - \frac{\eta^2}{2}\right)\frac{\mu^2}{k}
\]

So half-Kelly keeps about 75% of asymptotic log-growth while materially reducing rollout regret, checkpoint waste, and live-traffic drawdown.

That is usually the right starting point for LLM systems.

## Production Interpretation

KELLY-LM should sit here:

1. Define a composite validated utility score \(S_t\) that combines quality, cost, latency, safety, and reliability.
2. Enumerate candidate training and inference actions for the next control window.
3. Measure returns on held-out evals, canaries, and live slices, never on raw training loss alone.
4. Estimate \(\hat{\mu}\) from recent evidence with shrinkage toward a safe baseline.
5. Estimate \(\hat{\Sigma}\) from seed variance, rollout variance, and cross-action correlation.
6. Estimate \(\hat{\Lambda}\) from test coverage gaps, sample counts, and shift detectors.
7. Add \(\Gamma\) for operational hazards and non-negotiable tail-risk penalties.
8. Solve a constrained fractional-Kelly allocation, then apply hard rollout caps and kill switches.
9. Increase test budget when \(\mathrm{tr}(\Lambda)\) rises, when traffic shifts, or when a new model family enters the stack.
10. Recompute allocations continuously; do not treat last week's edge as stationary.

So the clean statement is:

KELLY-LM improves LLM training, testing, and inference by treating compute, evaluation coverage, and rollout share as allocatable capital, then sizing each commitment so that expected validated utility is not overwhelmed by noise, blind spots, covariance concentration, and deployment tail risk. It is best understood as an anti-overcommitment framework for LLM operations, not as a direct replacement for gradient descent, evaluation design, or serving policy.
