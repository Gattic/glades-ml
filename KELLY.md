# Kelly Criterion and Drawdown in Trading Algos

Kelly helps with drawdown in one specific way: it keeps sizing out of the overbetting regime where positive expectancy is destroyed by multiplicative compounding. It is not a drawdown minimizer. It is a growth-optimal leverage rule. The distinction matters.

## Core Mechanism

For a strategy with fractional allocation \(f_t\) and next-period return \(R_{t+1}\),

\[
W_{t+1}=W_t(1+f_tR_{t+1})
\]

Drawdown is path-dependent:

\[
D_t = 1 - \frac{W_t}{\max_{s\le t} W_s}
\]

Kelly chooses \(f_t\) to maximize expected log wealth increment:

\[
\max_{f_t} \; \mathbb{E}_t[\log(1+f_tR_{t+1})]
\]

Why that helps:

- Log utility is the correct objective for multiplicative wealth dynamics.
- It explicitly penalizes downside convexity.
- It internalizes volatility drag or variance drain, which linear expectancy ignores.
- It makes bankruptcy an absorbing boundary, because if \(1+fR \le 0\), log wealth is undefined and the process is effectively dead.

For small returns, the log-growth expansion is:

\[
g(f) \approx f\mu - \frac{1}{2}f^2\sigma^2
\]

where \(\mu=\mathbb{E}[R]\), \(\sigma^2=\mathrm{Var}(R)\).
The optimizer is:

\[
f^* = \frac{\mu}{\sigma^2}
\]

Interpretation:

- \(f\mu\) is edge capture.
- \(\frac{1}{2}f^2\sigma^2\) is the variance tax.
- Kelly is the point where marginal alpha equals marginal variance penalty.

That is the exact drawdown connection: Kelly prevents levering past the point where additional size increases path volatility faster than it increases geometric growth.

## Why Overbetting Causes Drawdown

If sizing exceeds Kelly, arithmetic expectancy may still look fine, but geometric growth deteriorates. Under the same small-return approximation:

\[
g(2f^*) \approx 0
\]

So at roughly 2x Kelly, long-run compound growth is already near zero. Beyond that, expected log growth turns negative. In practice this means:

- deeper peak-to-trough excursions,
- longer underwater periods,
- higher first-passage probability to a drawdown barrier,
- much larger recovery burden.

A concrete binary example:

- Even-money strategy, \(p=0.55\), \(q=0.45\)
- Kelly fraction:

\[
f^* = p-q = 0.10
\]

If the bet is 10% of capital per trade, 10 consecutive losses leaves:

\[
0.9^{10} \approx 0.349
\]

That is a 65.1% drawdown.

If the bet is 20% per trade, 10 consecutive losses leaves:

\[
0.8^{10} \approx 0.107
\]

That is an 89.3% drawdown.

So Kelly does not remove drawdown. It prevents making drawdown terminal by over-sizing.

## Why It Matters More in Algos Than in Discretionary Trading

In a multi-algo portfolio, the correct Kelly object is vector-valued:

\[
w^* \approx \Sigma^{-1}\mu
\]

under the small-return approximation, where:

- \(\mu\) is the vector of expected strategy returns,
- \(\Sigma\) is the covariance matrix of strategy returns.

This reduces drawdown through several advanced mechanisms:

- It penalizes highly correlated alpha streams, so the system does not unknowingly double-count the same latent factor exposure.
- It de-levers low-information-ratio algos.
- It converts covariance into an explicit capital charge.
- It reduces concentration in crowded regimes where multiple independent algos are really one trade in disguise.

This is a major drawdown-control benefit in crypto and systematic trading, because many strategy variants are just different parameterizations of the same beta, momentum, carry, or microstructure exposure.

## What Kelly Does Not Do

Kelly does not directly optimize:

- maximum drawdown,
- CVaR,
- short-horizon ruin probability,
- pathwise barrier constraints.

If the objective is literally to minimize drawdown, the mathematical optimum is trivial: use less size, asymptotically all the way to zero.

So the precise statement is:

- Kelly minimizes avoidable drawdown caused by structural overbetting.
- It does not guarantee low drawdown in finite samples.
- Full Kelly can still produce brutal drawdowns, especially under fat tails, serial correlation, regime shifts, or estimation error.

## Why Fractional Kelly Is Usually Better in Production

Real systems estimate \(\mu\) and \(\Sigma\) with noise. Full Kelly is extremely sensitive to estimation error, especially in \(\mu\). That causes systematic overbetting.

Use:

\[
w_t = \eta \, \hat{\Sigma}_t^{-1}\hat{\mu}_t
\quad\text{with}\quad 0<\eta<1
\]

where \(\eta\) is a fractional-Kelly multiplier.

A useful approximation:

\[
g(\eta f^*) \approx \left(\eta - \frac{\eta^2}{2}\right)\frac{\mu^2}{\sigma^2}
\]

At half-Kelly, \(\eta=0.5\), the system retains about 75% of asymptotic log-growth while materially reducing drawdown depth, barrier-hitting probability, and estimator fragility.

That is why serious systematic shops rarely run full Kelly on raw estimates.

## Production Interpretation

In an algo stack, Kelly should sit here:

1. Estimate conditional edge \(\mu_t\).
2. Estimate conditional covariance or tail risk \(\Sigma_t\).
3. Apply shrinkage or Bayesian regularization to both.
4. Compute fractional Kelly sizing.
5. Apply hard risk constraints: exposure caps, drawdown breakers, kill switches, liquidity caps.
6. De-lever further under regime instability or left-tail stress.

So the clean answer is:

Kelly helps avoid drawdown by sizing exposure such that expected edge is not overwhelmed by multiplicative variance drag, covariance concentration, and proximity-to-ruin effects. It is best understood as an anti-overbetting framework, not a direct drawdown minimizer.
