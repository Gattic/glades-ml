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
#include "garch.h"
#include <cmath>
#include <cstdio>
#include <limits>
#include <algorithm>

using namespace glades;

static const double VARIANCE_FLOOR = 1e-12;
static const double NLL_PENALTY = 1e12;
static const double LOG_2PI = 1.8378770664093453;

// ============================================================================
// Constructor
// ============================================================================

GARCH::GARCH(int p_, int q_, int maxIterations_, double tolerance_)
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
// Constrained parameters: omega > 0, alpha_i >= 0, beta_j >= 0,
//                         sum(alpha) + sum(beta) < 1
//
// Unconstrained phi[0] -> omega = exp(phi[0])
// Unconstrained phi[1..p+q] -> softmax partition for alpha/beta:
//   exp_i = exp(phi[1+i])
//   sumExp = 1 + sum(exp_i for i in 1..p+q)
//   alpha_i = exp(phi[1+i]) / sumExp        for i in 0..q-1
//   beta_j  = exp(phi[1+q+j]) / sumExp      for j in 0..p-1
//   The "1" in denominator reserves a share for (1 - sum(alpha) - sum(beta))
// ============================================================================

std::vector<double> GARCH::fromUnconstrained(const std::vector<double>& phi) const
{
	int n = 1 + q + p;
	std::vector<double> theta(n);

	// omega = exp(phi[0])
	theta[0] = exp(phi[0]);

	// Softmax partition for alpha/beta
	double sumExp = 1.0;
	for (int i = 1; i < n; ++i)
	{
		sumExp += exp(phi[i]);
	}

	for (int i = 1; i < n; ++i)
	{
		theta[i] = exp(phi[i]) / sumExp;
	}

	return theta;
}

std::vector<double> GARCH::toUnconstrained() const
{
	int n = 1 + q + p;
	std::vector<double> phi(n);

	// phi[0] = log(omega)
	phi[0] = log(omega);

	// Recover softmax inputs: we need the reserved share
	double persistence = 0.0;
	for (int i = 0; i < q; ++i)
		persistence += alpha[i];
	for (int j = 0; j < p; ++j)
		persistence += beta[j];

	double reserved = 1.0 - persistence;
	if (reserved < 1e-10)
		reserved = 1e-10;

	// softmax: theta_i = exp(phi_i) / (1 + sum(exp(phi)))
	// Set the reference (reserved) portion to exp(0) = 1, so phi_ref = 0
	// Then exp(phi_i) = theta_i / reserved
	for (int i = 0; i < q; ++i)
	{
		double ratio = alpha[i] / reserved;
		if (ratio < 1e-15)
			ratio = 1e-15;
		phi[1 + i] = log(ratio);
	}
	for (int j = 0; j < p; ++j)
	{
		double ratio = beta[j] / reserved;
		if (ratio < 1e-15)
			ratio = 1e-15;
		phi[1 + q + j] = log(ratio);
	}

	return phi;
}

void GARCH::transformGradient(const std::vector<double>& phi,
	const std::vector<double>& gradTheta,
	std::vector<double>& gradPhi) const
{
	int n = 1 + q + p;
	gradPhi.resize(n);

	// d(NLL)/d(phi[0]) = d(NLL)/d(omega) * d(omega)/d(phi[0])
	// omega = exp(phi[0]), so d(omega)/d(phi[0]) = omega = exp(phi[0])
	double omegaVal = exp(phi[0]);
	gradPhi[0] = gradTheta[0] * omegaVal;

	// For the softmax partition:
	// theta_i = exp(phi_i) / S, where S = 1 + sum(exp(phi_k)) for k=1..n-1
	// d(theta_i)/d(phi_j) = theta_i * (delta_ij - theta_j)
	// d(NLL)/d(phi_j) = sum_i gradTheta[i] * theta_i * (delta_ij - theta_j)

	// Compute softmax values
	double sumExp = 1.0;
	std::vector<double> softmax(n - 1);
	for (int i = 0; i < n - 1; ++i)
	{
		softmax[i] = exp(phi[1 + i]);
		sumExp += softmax[i];
	}
	for (int i = 0; i < n - 1; ++i)
	{
		softmax[i] /= sumExp;
	}

	// Compute dot = sum_i gradTheta[1+i] * theta_i (softmax values)
	double dot = 0.0;
	for (int i = 0; i < n - 1; ++i)
	{
		dot += gradTheta[1 + i] * softmax[i];
	}

	for (int j = 0; j < n - 1; ++j)
	{
		gradPhi[1 + j] = softmax[j] * (gradTheta[1 + j] - dot);
	}
}

// ============================================================================
// Negative log-likelihood with analytical gradient (single-pass)
//
// sigma2_t = omega + sum_{i=1}^{q} alpha_i * eps2_{t-i}
//                   + sum_{j=1}^{p} beta_j * sigma2_{t-j}
//
// NLL = 0.5 * sum_t [ log(2*pi) + log(sigma2_t) + eps2_t / sigma2_t ]
//
// Gradient computed simultaneously via recursive derivatives:
// d(sigma2_t)/d(omega) = 1 + sum_j beta_j * d(sigma2_{t-j})/d(omega)
// d(sigma2_t)/d(alpha_i) = eps2_{t-i} + sum_j beta_j * d(sigma2_{t-j})/d(alpha_i)
// d(sigma2_t)/d(beta_j) = sigma2_{t-j} + sum_k beta_k * d(sigma2_{t-k})/d(beta_j)
// ============================================================================

double GARCH::negLogLikelihood(const std::vector<double>& theta,
	std::vector<double>& gradient) const
{
	int T = (int)returns.size();
	int nParams = 1 + q + p;

	double omegaVal = theta[0];
	// theta[1..q] = alpha, theta[1+q..q+p] = beta

	// Compute sample variance for backcasting
	double sampleVar = 0.0;
	for (int t = 0; t < T; ++t)
		sampleVar += returns[t] * returns[t];
	sampleVar /= T;
	if (sampleVar < VARIANCE_FLOOR)
		sampleVar = VARIANCE_FLOOR;

	// Initialize gradient
	for (int k = 0; k < nParams; ++k)
		gradient[k] = 0.0;

	double nll = 0.0;

	// Circular buffers for lagged sigma2 and derivatives
	int maxLag = p > q ? p : q;
	if (maxLag < 1) maxLag = 1;

	std::vector<double> sigma2Buf(maxLag, sampleVar);
	std::vector<double> eps2Buf(maxLag, sampleVar);

	// Derivative buffer: dsigma2_buf[lag * nParams + k]
	std::vector<double> dsigma2Buf(maxLag * nParams, 0.0);

	int bufIdx = 0;

	for (int t = 0; t < T; ++t)
	{
		double eps2 = returns[t] * returns[t];

		// Compute sigma2_t
		double sigma2 = omegaVal;
		for (int i = 0; i < q; ++i)
		{
			int lag = i + 1;
			double laggedEps2 = (t >= lag) ?
				eps2Buf[((bufIdx - lag) % maxLag + maxLag) % maxLag] : sampleVar;
			sigma2 += theta[1 + i] * laggedEps2;
		}
		for (int j = 0; j < p; ++j)
		{
			int lag = j + 1;
			double laggedSigma2 = (t >= lag) ?
				sigma2Buf[((bufIdx - lag) % maxLag + maxLag) % maxLag] : sampleVar;
			sigma2 += theta[1 + q + j] * laggedSigma2;
		}

		if (sigma2 < VARIANCE_FLOOR)
			sigma2 = VARIANCE_FLOOR;

		// Compute derivatives d(sigma2_t)/d(theta_k)
		std::vector<double> dsigma2(nParams, 0.0);

		// d/d(omega) = 1
		dsigma2[0] = 1.0;

		// d/d(alpha_i) = eps2_{t-i}
		for (int i = 0; i < q; ++i)
		{
			int lag = i + 1;
			dsigma2[1 + i] = (t >= lag) ?
				eps2Buf[((bufIdx - lag) % maxLag + maxLag) % maxLag] : sampleVar;
		}

		// d/d(beta_j) = sigma2_{t-j}
		for (int j = 0; j < p; ++j)
		{
			int lag = j + 1;
			dsigma2[1 + q + j] = (t >= lag) ?
				sigma2Buf[((bufIdx - lag) % maxLag + maxLag) % maxLag] : sampleVar;
		}

		// Add recursive beta terms: sum_j beta_j * d(sigma2_{t-j})/d(theta_k)
		for (int j = 0; j < p; ++j)
		{
			int lag = j + 1;
			if (t >= lag)
			{
				int lagSlot = ((bufIdx - lag) % maxLag + maxLag) % maxLag;
				for (int k = 0; k < nParams; ++k)
				{
					dsigma2[k] += theta[1 + q + j] * dsigma2Buf[lagSlot * nParams + k];
				}
			}
		}

		// NLL contribution: 0.5 * [log(2*pi) + log(sigma2) + eps2/sigma2]
		nll += 0.5 * (LOG_2PI + log(sigma2) + eps2 / sigma2);

		// Gradient contribution: 0.5 * (1/sigma2 - eps2/sigma2^2) * dsigma2/dtheta
		double gradCoeff = 0.5 * (1.0 / sigma2 - eps2 / (sigma2 * sigma2));
		for (int k = 0; k < nParams; ++k)
		{
			gradient[k] += gradCoeff * dsigma2[k];
		}

		// Store in circular buffers
		sigma2Buf[bufIdx % maxLag] = sigma2;
		eps2Buf[bufIdx % maxLag] = eps2;
		for (int k = 0; k < nParams; ++k)
		{
			dsigma2Buf[(bufIdx % maxLag) * nParams + k] = dsigma2[k];
		}
		++bufIdx;
	}

	// Check for NaN
	if (nll != nll)
		return NLL_PENALTY;

	return nll;
}

// ============================================================================
// BFGS optimizer
// ============================================================================

double GARCH::lineSearch(const std::vector<double>& phi,
	const std::vector<double>& direction,
	double fCurrent,
	const std::vector<double>& gradCurrent,
	double& stepSize) const
{
	int n = (int)phi.size();
	double c1 = 1e-4; // Armijo condition parameter
	double rho = 0.5;  // Backtracking factor

	// Directional derivative
	double dirDeriv = 0.0;
	for (int i = 0; i < n; ++i)
		dirDeriv += gradCurrent[i] * direction[i];

	// If not a descent direction, return current value
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
		int nParams = 1 + q + p;
		std::vector<double> gradDummy(nParams);
		double fNew = negLogLikelihood(thetaNew, gradDummy);

		// Armijo condition
		if (fNew <= fCurrent + c1 * stepSize * dirDeriv)
			return fNew;

		stepSize *= rho;
	}

	return fCurrent;
}

bool GARCH::bfgsMinimize(std::vector<double>& phi, double& fVal, int& iters)
{
	int n = (int)phi.size();
	int nParams = 1 + q + p;

	// Evaluate initial function and gradient
	std::vector<double> theta = fromUnconstrained(phi);
	std::vector<double> gradTheta(nParams);
	fVal = negLogLikelihood(theta, gradTheta);

	std::vector<double> gradPhi(n);
	transformGradient(phi, gradTheta, gradPhi);

	// Initialize inverse Hessian to identity (flat n*n)
	std::vector<double> H(n * n, 0.0);
	for (int i = 0; i < n; ++i)
		H[i * n + i] = 1.0;

	for (iters = 0; iters < maxIterations; ++iters)
	{
		// Check gradient norm for convergence
		double gradNorm = 0.0;
		for (int i = 0; i < n; ++i)
			gradNorm += gradPhi[i] * gradPhi[i];
		gradNorm = sqrt(gradNorm);

		if (gradNorm < tolerance)
			return true;

		// Compute search direction: d = -H * gradPhi
		std::vector<double> direction(n, 0.0);
		for (int i = 0; i < n; ++i)
		{
			for (int j = 0; j < n; ++j)
			{
				direction[i] -= H[i * n + j] * gradPhi[j];
			}
		}

		// Line search
		double stepSize = 0.0;
		double fNew = lineSearch(phi, direction, fVal, gradPhi, stepSize);

		if (stepSize < 1e-15)
			break;

		// Update phi
		std::vector<double> s(n);
		for (int i = 0; i < n; ++i)
		{
			s[i] = stepSize * direction[i];
			phi[i] += s[i];
		}

		// Evaluate new gradient
		theta = fromUnconstrained(phi);
		std::vector<double> gradThetaNew(nParams);
		fNew = negLogLikelihood(theta, gradThetaNew);

		std::vector<double> gradPhiNew(n);
		transformGradient(phi, gradThetaNew, gradPhiNew);

		// Check for NaN
		if (fNew != fNew)
		{
			fVal = NLL_PENALTY;
			return false;
		}

		// Compute y = gradNew - gradOld
		std::vector<double> y(n);
		for (int i = 0; i < n; ++i)
			y[i] = gradPhiNew[i] - gradPhi[i];

		// Curvature condition: sTy > 0
		double sTy = 0.0;
		for (int i = 0; i < n; ++i)
			sTy += s[i] * y[i];

		if (sTy > 1e-10)
		{
			// BFGS update of inverse Hessian
			// H = (I - rho*s*yT) * H * (I - rho*y*sT) + rho*s*sT
			double rhoVal = 1.0 / sTy;

			// Compute H*y
			std::vector<double> Hy(n, 0.0);
			for (int i = 0; i < n; ++i)
				for (int j = 0; j < n; ++j)
					Hy[i] += H[i * n + j] * y[j];

			double yTHy = 0.0;
			for (int i = 0; i < n; ++i)
				yTHy += y[i] * Hy[i];

			for (int i = 0; i < n; ++i)
			{
				for (int j = 0; j < n; ++j)
				{
					H[i * n + j] += rhoVal * ((1.0 + rhoVal * yTHy) * s[i] * s[j]
						- Hy[i] * s[j] - s[i] * Hy[j]);
				}
			}
		}
		else
		{
			// Reset inverse Hessian to identity
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
// Initialization via variance targeting
// ============================================================================

void GARCH::initializeParameters()
{
	double sampleVar = 0.0;
	int T = (int)returns.size();
	for (int t = 0; t < T; ++t)
		sampleVar += returns[t] * returns[t];
	sampleVar /= T;
	if (sampleVar < VARIANCE_FLOOR)
		sampleVar = VARIANCE_FLOOR;

	alpha.resize(q);
	beta.resize(p);

	for (int i = 0; i < q; ++i)
		alpha[i] = 0.05 / q;

	for (int j = 0; j < p; ++j)
		beta[j] = 0.90 / p;

	// omega = sampleVar * (1 - sum(alpha) - sum(beta))
	double persistence = 0.0;
	for (int i = 0; i < q; ++i)
		persistence += alpha[i];
	for (int j = 0; j < p; ++j)
		persistence += beta[j];
	omega = sampleVar * (1.0 - persistence);
}

// ============================================================================
// fit()
// ============================================================================

GarchResult GARCH::fit(const std::vector<double>& inputReturns)
{
	result = GarchResult();
	fitted = false;

	int T = (int)inputReturns.size();
	int minObs = 2 * (p + q) + 1;
	if (T < minObs)
	{
		printf("GARCH: insufficient data (%d observations, need at least %d)\n", T, minObs);
		return result;
	}

	returns = inputReturns;

	// Initialize parameters
	initializeParameters();

	// Transform to unconstrained space
	std::vector<double> phi = toUnconstrained();

	// Run BFGS
	double fVal = 0.0;
	int iters = 0;
	bool converged = bfgsMinimize(phi, fVal, iters);

	// Extract fitted parameters
	std::vector<double> theta = fromUnconstrained(phi);
	omega = theta[0];
	for (int i = 0; i < q; ++i)
		alpha[i] = theta[1 + i];
	for (int j = 0; j < p; ++j)
		beta[j] = theta[1 + q + j];

	// Compute conditional variances and residuals
	double sampleVar = 0.0;
	for (int t = 0; t < T; ++t)
		sampleVar += returns[t] * returns[t];
	sampleVar /= T;

	conditionalVariances.resize(T);
	residuals.resize(T);

	for (int t = 0; t < T; ++t)
	{
		double sigma2 = omega;
		for (int i = 0; i < q; ++i)
		{
			int lag = i + 1;
			double laggedEps2 = (t >= lag) ?
				returns[t - lag] * returns[t - lag] : sampleVar;
			sigma2 += alpha[i] * laggedEps2;
		}
		for (int j = 0; j < p; ++j)
		{
			int lag = j + 1;
			double laggedSigma2 = (t >= lag) ?
				conditionalVariances[t - lag] : sampleVar;
			sigma2 += beta[j] * laggedSigma2;
		}
		if (sigma2 < VARIANCE_FLOOR)
			sigma2 = VARIANCE_FLOOR;

		conditionalVariances[t] = sigma2;
		residuals[t] = returns[t] / sqrt(sigma2);
	}

	// Fill result
	int nParams = 1 + q + p;
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
// ============================================================================

std::vector<double> GARCH::forecast(int h) const
{
	std::vector<double> fcast;
	if (!fitted || h <= 0)
		return fcast;

	int T = (int)returns.size();
	int maxLag = p > q ? p : q;
	if (maxLag < 1) maxLag = 1;

	// Build history of sigma2 and eps2 up to end of sample
	std::vector<double> sigma2Hist(maxLag);
	std::vector<double> eps2Hist(maxLag);

	for (int i = 0; i < maxLag; ++i)
	{
		int t = T - maxLag + i;
		if (t >= 0)
		{
			sigma2Hist[i] = conditionalVariances[t];
			eps2Hist[i] = returns[t] * returns[t];
		}
		else
		{
			double sampleVar = 0.0;
			for (int k = 0; k < T; ++k)
				sampleVar += returns[k] * returns[k];
			sampleVar /= T;
			sigma2Hist[i] = sampleVar;
			eps2Hist[i] = sampleVar;
		}
	}

	fcast.resize(h);
	for (int s = 0; s < h; ++s)
	{
		double sigma2 = omega;

		for (int i = 0; i < q; ++i)
		{
			int lag = i + 1;
			int idx = (int)eps2Hist.size() - lag;
			if (idx >= 0)
				sigma2 += alpha[i] * eps2Hist[idx];
		}

		for (int j = 0; j < p; ++j)
		{
			int lag = j + 1;
			int idx = (int)sigma2Hist.size() - lag;
			if (idx >= 0)
				sigma2 += beta[j] * sigma2Hist[idx];
		}

		fcast[s] = sigma2;

		// For future steps, E[eps2_t] = sigma2_t
		eps2Hist.push_back(sigma2);
		sigma2Hist.push_back(sigma2);
	}

	return fcast;
}

std::vector<double> GARCH::forecastVolatility(int h) const
{
	std::vector<double> var = forecast(h);
	for (size_t i = 0; i < var.size(); ++i)
		var[i] = sqrt(var[i]);
	return var;
}

// ============================================================================
// Accessors
// ============================================================================

double GARCH::getOmega() const { return omega; }
const std::vector<double>& GARCH::getAlpha() const { return alpha; }
const std::vector<double>& GARCH::getBeta() const { return beta; }

double GARCH::getPersistence() const
{
	double pers = 0.0;
	for (size_t i = 0; i < alpha.size(); ++i)
		pers += alpha[i];
	for (size_t j = 0; j < beta.size(); ++j)
		pers += beta[j];
	return pers;
}

double GARCH::getUnconditionalVariance() const
{
	double pers = getPersistence();
	if (pers >= 1.0)
		return -1.0;
	return omega / (1.0 - pers);
}

const std::vector<double>& GARCH::getConditionalVariances() const { return conditionalVariances; }
const std::vector<double>& GARCH::getResiduals() const { return residuals; }
const GarchResult& GARCH::getResult() const { return result; }
bool GARCH::isFitted() const { return fitted; }

// ============================================================================
// Order selection via BIC
// ============================================================================

std::pair<int, int> GARCH::selectOrder(const std::vector<double>& returns,
	int maxP, int maxQ)
{
	double bestBIC = std::numeric_limits<double>::max();
	int bestP = 1;
	int bestQ = 1;

	for (int pp = 1; pp <= maxP; ++pp)
	{
		for (int qq = 1; qq <= maxQ; ++qq)
		{
			GARCH model(pp, qq);
			GarchResult res = model.fit(returns);
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
