// Copyright 2024 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
// associated documentation files (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge, publish, distribute,
// sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#include "egarch.h"
#include <cmath>
#include <cstdio>
#include <limits>
#include <algorithm>

using namespace glades;

static const double NLL_PENALTY = 1e12;
static const double LOG_2PI = 1.8378770664093453;
static const double E_ABS_Z = 0.7978845608028654; // sqrt(2/pi) for standard normal
static const double LOG_VAR_FLOOR = -50.0;         // exp(-50) ~ 1.9e-22
static const double LOG_VAR_CEIL  =  50.0;         // exp(50) ~ 5.2e+21

// ============================================================================
// Constructor
// ============================================================================

EGARCH::EGARCH(int p_, int q_, int maxIterations_, double tolerance_)
	: p(p_)
	, q(q_)
	, maxIterations(maxIterations_)
	, tolerance(tolerance_)
	, fitted(false)
	, omega(0.0)
{
}

// ============================================================================
// Parameter transformation
//
// EGARCH parameters are mostly unconstrained (a key advantage over GARCH).
// Only beta needs bounding for stationarity: |beta_j| < 1.
//
// Layout: theta = [omega, alpha_1..q, gamma_1..q, beta_1..p]
//         nParams = 1 + 2*q + p
//
// Transform:
//   omega, alpha_i, gamma_i: identity (unconstrained)
//   beta_j = tanh(phi_{1+2q+j})
// ============================================================================

std::vector<double> EGARCH::fromUnconstrained(const std::vector<double>& phi) const
{
	int nParams = 1 + 2 * q + p;
	std::vector<double> theta(nParams);

	// omega, alpha, gamma: identity
	int nIdentity = 1 + 2 * q;
	for (int k = 0; k < nIdentity; ++k)
		theta[k] = phi[k];

	// beta: tanh
	for (int j = 0; j < p; ++j)
		theta[nIdentity + j] = tanh(phi[nIdentity + j]);

	return theta;
}

std::vector<double> EGARCH::toUnconstrained() const
{
	int nParams = 1 + 2 * q + p;
	std::vector<double> phi(nParams);

	// omega
	phi[0] = omega;

	// alpha
	for (int i = 0; i < q; ++i)
		phi[1 + i] = alpha[i];

	// gamma
	for (int i = 0; i < q; ++i)
		phi[1 + q + i] = gamma[i];

	// beta: atanh
	for (int j = 0; j < p; ++j)
	{
		double b = beta[j];
		// Clamp to avoid atanh singularity
		if (b >= 0.9999) b = 0.9999;
		if (b <= -0.9999) b = -0.9999;
		phi[1 + 2 * q + j] = 0.5 * log((1.0 + b) / (1.0 - b)); // atanh
	}

	return phi;
}

void EGARCH::transformGradient(const std::vector<double>& phi,
	const std::vector<double>& gradTheta,
	std::vector<double>& gradPhi) const
{
	int nParams = 1 + 2 * q + p;
	gradPhi.resize(nParams);

	// omega, alpha, gamma: identity transform, gradient passes through
	int nIdentity = 1 + 2 * q;
	for (int k = 0; k < nIdentity; ++k)
		gradPhi[k] = gradTheta[k];

	// beta: d(tanh(phi))/d(phi) = 1 - tanh^2(phi) = 1 - beta^2
	for (int j = 0; j < p; ++j)
	{
		double b = tanh(phi[nIdentity + j]);
		gradPhi[nIdentity + j] = gradTheta[nIdentity + j] * (1.0 - b * b);
	}
}

// ============================================================================
// Negative log-likelihood with analytical gradient (single-pass)
//
// Log-variance recursion:
//   h_t = omega + sum_{i=1}^{q} [alpha_i*(|z_{t-i}|-E|z|) + gamma_i*z_{t-i}]
//                + sum_{j=1}^{p} beta_j * h_{t-j}
//
// where z_t = eps_t / sigma_t = eps_t * exp(-h_t/2), E|z| = sqrt(2/pi)
//
// NLL = 0.5 * sum_t [ log(2*pi) + h_t + eps_t^2 * exp(-h_t) ]
//
// Gradient: d(NLL)/d(theta) = 0.5 * sum_t (1 - z_t^2) * d(h_t)/d(theta)
//
// Conditional score approach: treat z_{t-i} as fixed w.r.t. parameters.
// The exact gradient has sign(z) discontinuities that cause BFGS
// oscillation. The conditional score is smooth and converges to the
// same MLE asymptotically (standard practice in EGARCH implementations).
// ============================================================================

double EGARCH::negLogLikelihood(const std::vector<double>& theta,
	std::vector<double>& gradient) const
{
	int T = (int)returns.size();
	int nParams = 1 + 2 * q + p;

	double omegaVal = theta[0];
	// theta layout: [omega, alpha_1..q, gamma_1..q, beta_1..p]

	// Compute sample variance for initialization
	double sampleVar = 0.0;
	for (int t = 0; t < T; ++t)
		sampleVar += returns[t] * returns[t];
	sampleVar /= T;
	if (sampleVar < 1e-20)
		sampleVar = 1e-20;
	double initH = log(sampleVar);

	// Initialize gradient
	for (int k = 0; k < nParams; ++k)
		gradient[k] = 0.0;

	double nll = 0.0;

	// Circular buffers for lagged h, z, and derivatives
	int maxLag = p > q ? p : q;
	if (maxLag < 1) maxLag = 1;

	std::vector<double> hBuf(maxLag, initH);
	std::vector<double> zBuf(maxLag, 0.0); // pre-sample z = 0

	// Derivative buffer: dh_buf[lag * nParams + k]
	std::vector<double> dhBuf(maxLag * nParams, 0.0);

	int bufIdx = 0;

	for (int t = 0; t < T; ++t)
	{
		// Compute h_t = log(sigma2_t)
		double h = omegaVal;

		for (int i = 0; i < q; ++i)
		{
			int lag = i + 1;
			double laggedZ = (t >= lag) ?
				zBuf[((bufIdx - lag) % maxLag + maxLag) % maxLag] : 0.0;
			double absZterm = fabs(laggedZ) - E_ABS_Z;
			h += theta[1 + i] * absZterm + theta[1 + q + i] * laggedZ;
		}

		for (int j = 0; j < p; ++j)
		{
			int lag = j + 1;
			double laggedH = (t >= lag) ?
				hBuf[((bufIdx - lag) % maxLag + maxLag) % maxLag] : initH;
			h += theta[1 + 2 * q + j] * laggedH;
		}

		// Clamp h to prevent overflow/underflow in exp
		if (h < LOG_VAR_FLOOR) h = LOG_VAR_FLOOR;
		if (h > LOG_VAR_CEIL) h = LOG_VAR_CEIL;

		double sigma2 = exp(h);
		double z = returns[t] / sqrt(sigma2);

		// Compute derivatives d(h_t)/d(theta_k)
		std::vector<double> dh(nParams, 0.0);

		// Direct term for omega
		dh[0] = 1.0;

		// Direct terms for alpha_i and gamma_i
		for (int i = 0; i < q; ++i)
		{
			int lag = i + 1;
			double laggedZ = (t >= lag) ?
				zBuf[((bufIdx - lag) % maxLag + maxLag) % maxLag] : 0.0;
			dh[1 + i] = fabs(laggedZ) - E_ABS_Z;       // d/d(alpha_i)
			dh[1 + q + i] = laggedZ;                     // d/d(gamma_i)
		}

		// Direct terms for beta_j
		for (int j = 0; j < p; ++j)
		{
			int lag = j + 1;
			dh[1 + 2 * q + j] = (t >= lag) ?
				hBuf[((bufIdx - lag) % maxLag + maxLag) % maxLag] : initH;
		}

		// Recursive terms via beta persistence only (conditional score)
		for (int j = 0; j < p; ++j)
		{
			int lag = j + 1;
			if (t >= lag)
			{
				int lagSlot = ((bufIdx - lag) % maxLag + maxLag) % maxLag;
				for (int k = 0; k < nParams; ++k)
					dh[k] += theta[1 + 2 * q + j] * dhBuf[lagSlot * nParams + k];
			}
		}

		// NLL contribution: 0.5 * [log(2*pi) + h + eps^2 * exp(-h)]
		double eps2 = returns[t] * returns[t];
		nll += 0.5 * (LOG_2PI + h + eps2 / sigma2);

		// Gradient contribution: 0.5 * (1 - z^2) * dh/dtheta
		double gradCoeff = 0.5 * (1.0 - z * z);
		for (int k = 0; k < nParams; ++k)
			gradient[k] += gradCoeff * dh[k];

		// Store in circular buffers
		hBuf[bufIdx % maxLag] = h;
		zBuf[bufIdx % maxLag] = z;
		for (int k = 0; k < nParams; ++k)
			dhBuf[(bufIdx % maxLag) * nParams + k] = dh[k];
		++bufIdx;
	}

	// Check for NaN
	if (nll != nll)
		return NLL_PENALTY;

	return nll;
}

// ============================================================================
// BFGS optimizer (same structure as GARCH)
// ============================================================================

double EGARCH::lineSearch(const std::vector<double>& phi,
	const std::vector<double>& direction,
	double fCurrent,
	const std::vector<double>& gradCurrent,
	double& stepSize) const
{
	int n = (int)phi.size();
	double c1 = 1e-4;
	double rho = 0.5;

	double dirDeriv = 0.0;
	for (int i = 0; i < n; ++i)
		dirDeriv += gradCurrent[i] * direction[i];

	if (dirDeriv >= 0.0)
	{
		stepSize = 0.0;
		return fCurrent;
	}

	stepSize = 1.0;
	int maxBacktrack = 30;

	for (int iter = 0; iter < maxBacktrack; ++iter)
	{
		std::vector<double> phiNew(n);
		for (int i = 0; i < n; ++i)
			phiNew[i] = phi[i] + stepSize * direction[i];

		std::vector<double> thetaNew = fromUnconstrained(phiNew);
		int nParams = 1 + 2 * q + p;
		std::vector<double> gradDummy(nParams);
		double fNew = negLogLikelihood(thetaNew, gradDummy);

		if (fNew <= fCurrent + c1 * stepSize * dirDeriv)
			return fNew;

		stepSize *= rho;
	}

	return fCurrent;
}

bool EGARCH::bfgsMinimize(std::vector<double>& phi, double& fVal, int& iters)
{
	int n = (int)phi.size();
	int nParams = 1 + 2 * q + p;

	std::vector<double> theta = fromUnconstrained(phi);
	std::vector<double> gradTheta(nParams);
	fVal = negLogLikelihood(theta, gradTheta);

	std::vector<double> gradPhi(n);
	transformGradient(phi, gradTheta, gradPhi);

	// Initialize inverse Hessian to identity
	std::vector<double> H(n * n, 0.0);
	for (int i = 0; i < n; ++i)
		H[i * n + i] = 1.0;

	int staleCount = 0;

	for (iters = 0; iters < maxIterations; ++iters)
	{
		double gradNorm = 0.0;
		for (int i = 0; i < n; ++i)
			gradNorm += gradPhi[i] * gradPhi[i];
		gradNorm = sqrt(gradNorm);

		if (gradNorm < tolerance)
			return true;

		// Search direction: d = -H * grad
		std::vector<double> direction(n, 0.0);
		for (int i = 0; i < n; ++i)
			for (int j = 0; j < n; ++j)
				direction[i] -= H[i * n + j] * gradPhi[j];

		double stepSize = 0.0;
		lineSearch(phi, direction, fVal, gradPhi, stepSize);

		if (stepSize < 1e-15)
			break;

		// Update phi
		std::vector<double> s(n);
		for (int i = 0; i < n; ++i)
		{
			s[i] = stepSize * direction[i];
			phi[i] += s[i];
		}

		// New gradient
		theta = fromUnconstrained(phi);
		std::vector<double> gradThetaNew(nParams);
		double fNew = negLogLikelihood(theta, gradThetaNew);

		std::vector<double> gradPhiNew(n);
		transformGradient(phi, gradThetaNew, gradPhiNew);

		if (fNew != fNew)
		{
			fVal = NLL_PENALTY;
			return false;
		}

		// Check for relative function value convergence
		double relChange = fabs(fNew - fVal) / (fabs(fVal) + 1e-10);
		if (relChange < tolerance)
		{
			++staleCount;
			if (staleCount >= 3)
			{
				fVal = fNew;
				gradPhi = gradPhiNew;
				return true;
			}
		}
		else
		{
			staleCount = 0;
		}

		// BFGS update
		std::vector<double> y(n);
		for (int i = 0; i < n; ++i)
			y[i] = gradPhiNew[i] - gradPhi[i];

		double sTy = 0.0;
		for (int i = 0; i < n; ++i)
			sTy += s[i] * y[i];

		if (sTy > 1e-10)
		{
			double rhoVal = 1.0 / sTy;

			std::vector<double> Hy(n, 0.0);
			for (int i = 0; i < n; ++i)
				for (int j = 0; j < n; ++j)
					Hy[i] += H[i * n + j] * y[j];

			double yTHy = 0.0;
			for (int i = 0; i < n; ++i)
				yTHy += y[i] * Hy[i];

			for (int i = 0; i < n; ++i)
				for (int j = 0; j < n; ++j)
					H[i * n + j] += rhoVal * ((1.0 + rhoVal * yTHy) * s[i] * s[j]
						- Hy[i] * s[j] - s[i] * Hy[j]);
		}
		else
		{
			for (int i = 0; i < n * n; ++i)
				H[i] = 0.0;
			for (int i = 0; i < n; ++i)
				H[i * n + i] = 1.0;
		}

		fVal = fNew;
		gradPhi = gradPhiNew;
	}

	return false;
}

// ============================================================================
// Initialization
// ============================================================================

void EGARCH::initializeParameters()
{
	double sampleVar = 0.0;
	int T = (int)returns.size();
	for (int t = 0; t < T; ++t)
		sampleVar += returns[t] * returns[t];
	sampleVar /= T;
	if (sampleVar < 1e-20)
		sampleVar = 1e-20;

	alpha.resize(q);
	gamma.resize(q);
	beta.resize(p);

	// Target: long-run h = log(sampleVar)
	// h = omega / (1 - sum(beta)), so omega = log(sampleVar) * (1 - sum(beta))
	double betaSum = 0.0;
	for (int j = 0; j < p; ++j)
	{
		beta[j] = 0.95 / p;
		betaSum += beta[j];
	}

	omega = log(sampleVar) * (1.0 - betaSum);

	for (int i = 0; i < q; ++i)
	{
		alpha[i] = 0.10 / q;
		gamma[i] = -0.05 / q; // slight negative leverage as starting guess
	}
}

// ============================================================================
// fit()
// ============================================================================

EgarchResult EGARCH::fit(const std::vector<double>& inputReturns)
{
	result = EgarchResult();
	fitted = false;

	int T = (int)inputReturns.size();
	int nParams = 1 + 2 * q + p;
	int minObs = 2 * nParams + 1;
	if (T < minObs)
	{
		printf("EGARCH: insufficient data (%d observations, need at least %d)\n", T, minObs);
		return result;
	}

	returns = inputReturns;

	initializeParameters();

	std::vector<double> phi = toUnconstrained();

	double fVal = 0.0;
	int iters = 0;
	bool converged = bfgsMinimize(phi, fVal, iters);

	// Extract fitted parameters
	std::vector<double> theta = fromUnconstrained(phi);
	omega = theta[0];
	for (int i = 0; i < q; ++i)
		alpha[i] = theta[1 + i];
	for (int i = 0; i < q; ++i)
		gamma[i] = theta[1 + q + i];
	for (int j = 0; j < p; ++j)
		beta[j] = theta[1 + 2 * q + j];

	// Compute conditional variances and residuals via forward recursion
	double sampleVar = 0.0;
	for (int t = 0; t < T; ++t)
		sampleVar += returns[t] * returns[t];
	sampleVar /= T;
	double initH = log(sampleVar);

	conditionalVariances.resize(T);
	residuals.resize(T);

	// Store log-variances and z values for the recursion
	std::vector<double> hVec(T);
	std::vector<double> zVec(T);

	for (int t = 0; t < T; ++t)
	{
		double h = omega;

		for (int i = 0; i < q; ++i)
		{
			int lag = i + 1;
			double laggedZ = (t >= lag) ? zVec[t - lag] : 0.0;
			h += alpha[i] * (fabs(laggedZ) - E_ABS_Z) + gamma[i] * laggedZ;
		}

		for (int j = 0; j < p; ++j)
		{
			int lag = j + 1;
			double laggedH = (t >= lag) ? hVec[t - lag] : initH;
			h += beta[j] * laggedH;
		}

		if (h < LOG_VAR_FLOOR) h = LOG_VAR_FLOOR;
		if (h > LOG_VAR_CEIL) h = LOG_VAR_CEIL;

		hVec[t] = h;
		double sigma2 = exp(h);
		conditionalVariances[t] = sigma2;
		double z = returns[t] / sqrt(sigma2);
		zVec[t] = z;
		residuals[t] = z;
	}

	// Fill result
	result.logLikelihood = -fVal;
	result.AIC = 2.0 * fVal + 2.0 * nParams;
	result.BIC = 2.0 * fVal + nParams * log((double)T);
	result.iterations = iters;
	result.converged = converged;

	fitted = true;
	return result;
}

// ============================================================================
// forecast()
//
// For future steps, E[z] = 0 and E[|z| - E|z|] = 0, so the news terms
// vanish. The log-variance recursion simplifies to:
//   h_{T+s} = omega + sum_j beta_j * h_{T+s-j}
//           + sum_i [alpha_i*(|z_{T+s-i}|-E|z|) + gamma_i*z_{T+s-i}]
//             (only for in-sample lags where T+s-i <= T)
//
// Forecasted variance: sigma2_{T+s} = exp(h_{T+s})
// ============================================================================

std::vector<double> EGARCH::forecast(int h) const
{
	std::vector<double> fcast;
	if (!fitted || h <= 0)
		return fcast;

	int T = (int)returns.size();
	int maxLag = p > q ? p : q;
	if (maxLag < 1) maxLag = 1;

	// Reconstruct trailing h and z values from fitted state
	// Need the last maxLag values
	std::vector<double> hHist(maxLag);
	std::vector<double> zHist(maxLag);

	// Recompute h and z for the tail of the sample
	{
		double sampleVar = 0.0;
		for (int t = 0; t < T; ++t)
			sampleVar += returns[t] * returns[t];
		sampleVar /= T;
		double initH = log(sampleVar);

		std::vector<double> hAll(T);
		std::vector<double> zAll(T);

		for (int t = 0; t < T; ++t)
		{
			double hVal = omega;
			for (int i = 0; i < q; ++i)
			{
				int lag = i + 1;
				double laggedZ = (t >= lag) ? zAll[t - lag] : 0.0;
				hVal += alpha[i] * (fabs(laggedZ) - E_ABS_Z) + gamma[i] * laggedZ;
			}
			for (int j = 0; j < p; ++j)
			{
				int lag = j + 1;
				double laggedH = (t >= lag) ? hAll[t - lag] : initH;
				hVal += beta[j] * laggedH;
			}
			if (hVal < LOG_VAR_FLOOR) hVal = LOG_VAR_FLOOR;
			if (hVal > LOG_VAR_CEIL) hVal = LOG_VAR_CEIL;
			hAll[t] = hVal;
			zAll[t] = returns[t] / sqrt(exp(hVal));
		}

		for (int i = 0; i < maxLag; ++i)
		{
			int t = T - maxLag + i;
			if (t >= 0)
			{
				hHist[i] = hAll[t];
				zHist[i] = zAll[t];
			}
			else
			{
				hHist[i] = initH;
				zHist[i] = 0.0;
			}
		}
	}

	fcast.resize(h);
	for (int s = 0; s < h; ++s)
	{
		double hVal = omega;

		// News impact terms (only for in-sample z values)
		for (int i = 0; i < q; ++i)
		{
			int lag = i + 1;
			int idx = (int)zHist.size() - lag;
			if (idx >= 0 && (s < lag)) // only use actual z values, not future
			{
				double laggedZ = zHist[idx];
				hVal += alpha[i] * (fabs(laggedZ) - E_ABS_Z) + gamma[i] * laggedZ;
			}
			// For future z: E[|z|-E|z|] = 0 and E[z] = 0, so no contribution
		}

		// Log-variance persistence
		for (int j = 0; j < p; ++j)
		{
			int lag = j + 1;
			int idx = (int)hHist.size() - lag;
			if (idx >= 0)
				hVal += beta[j] * hHist[idx];
		}

		if (hVal < LOG_VAR_FLOOR) hVal = LOG_VAR_FLOOR;
		if (hVal > LOG_VAR_CEIL) hVal = LOG_VAR_CEIL;

		fcast[s] = exp(hVal);

		// Push forecast h and z=0 for future steps
		hHist.push_back(hVal);
		zHist.push_back(0.0);
	}

	return fcast;
}

std::vector<double> EGARCH::forecastVolatility(int h) const
{
	std::vector<double> var = forecast(h);
	for (size_t i = 0; i < var.size(); ++i)
		var[i] = sqrt(var[i]);
	return var;
}

// ============================================================================
// Accessors
// ============================================================================

double EGARCH::getOmega() const { return omega; }
const std::vector<double>& EGARCH::getAlpha() const { return alpha; }
const std::vector<double>& EGARCH::getGamma() const { return gamma; }
const std::vector<double>& EGARCH::getBeta() const { return beta; }

double EGARCH::getPersistence() const
{
	double pers = 0.0;
	for (size_t j = 0; j < beta.size(); ++j)
		pers += fabs(beta[j]);
	return pers;
}

double EGARCH::getUnconditionalVariance() const
{
	double betaSum = 0.0;
	for (size_t j = 0; j < beta.size(); ++j)
		betaSum += beta[j];
	if (fabs(betaSum) >= 1.0)
		return -1.0;
	double unconditionalH = omega / (1.0 - betaSum);
	return exp(unconditionalH);
}

const std::vector<double>& EGARCH::getConditionalVariances() const { return conditionalVariances; }
const std::vector<double>& EGARCH::getResiduals() const { return residuals; }
const EgarchResult& EGARCH::getResult() const { return result; }
bool EGARCH::isFitted() const { return fitted; }

bool EGARCH::hasLeverageEffect() const
{
	// Leverage effect: negative gamma means negative returns increase volatility
	for (size_t i = 0; i < gamma.size(); ++i)
	{
		if (gamma[i] < 0.0)
			return true;
	}
	return false;
}

// ============================================================================
// Order selection via BIC
// ============================================================================

std::pair<int, int> EGARCH::selectOrder(const std::vector<double>& returns,
	int maxP, int maxQ)
{
	double bestBIC = std::numeric_limits<double>::max();
	int bestP = 1;
	int bestQ = 1;

	for (int pp = 1; pp <= maxP; ++pp)
	{
		for (int qq = 1; qq <= maxQ; ++qq)
		{
			EGARCH model(pp, qq);
			EgarchResult res = model.fit(returns);
			if (res.converged && res.BIC < bestBIC)
			{
				bestBIC = res.BIC;
				bestP = pp;
				bestQ = qq;
			}
		}
	}

	return std::pair<int, int>(bestP, bestQ);
}
