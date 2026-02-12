// Copyright 2026 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
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
#include "gmath.h"
#include "Backend/Database/GType.h"

using namespace glades;

namespace {
// Epsilon choices:
// - For probabilities going into log()/division: keep away from {0,1}.
// - For divisors: avoid division-by-zero while preserving sign where relevant.
static inline float clampf(float x, float lo, float hi)
{
	if (x < lo)
		return lo;
	if (x > hi)
		return hi;
	return x;
}

static inline float clamp_prob01(float p)
{
	// Small enough to avoid biasing typical training, large enough to prevent inf/NaN.
	const float eps = 1e-7f;
	return clampf(p, eps, 1.0f - eps);
}

static inline float safe_div(float num, float den)
{
	const float eps = 1e-12f;
	if (den > -eps && den < eps)
		return num / (den < 0.0f ? -eps : eps);
	return num / den;
}
} // namespace

// Define static class constants
const float glades::GMath::INLIER = 0.954f;
const float glades::GMath::OUTLIER = 0.046f;

float glades::GMath::squash(float netInput, int activationFx, float fxParam)
{
	float netOutput = 0.0f;

	switch (activationFx)
	{
	case TANH:
	{
		netOutput = tanhf(netInput);

		break;
	}
	case TANHP:
	{
		if (netInput > 1.0f - fxParam)
			netOutput = 1.0f;
		else if (netInput < fxParam - 1.0f)
			netOutput = -1.0f;
		else
			netOutput = tanhf(netInput);

		break;
	}
	case SIGMOID:
	{
		// Numerically-stable sigmoid.
		// Avoids overflow in expf() for large-magnitude inputs.
		if (netInput >= 0.0f)
		{
			const float z = expf(-netInput);
			netOutput = 1.0f / (1.0f + z);
		}
		else
		{
			const float z = expf(netInput);
			netOutput = z / (1.0f + z);
		}

		break;
	}
	case SIGMOIDP:
	{
		if (netInput < fxParam)
			netOutput = 0.01f;
		else if (netInput > 1.0f - fxParam)
			netOutput = 0.99f;
		else
		{
			// Same stable sigmoid core as SIGMOID.
			if (netInput >= 0.0f)
			{
				const float z = expf(-netInput);
				netOutput = 1.0f / (1.0f + z);
			}
			else
			{
				const float z = expf(netInput);
				netOutput = z / (1.0f + z);
			}
		}

		break;
	}
	case RELU:
	{
		// Standard ReLU threshold is 0.0 (not OUTLIER).
		if (netInput <= 0.0f) // including negative vals
			netOutput = 0.0f;
		else
			netOutput = netInput;

		break;
	}
	case LEAKY:
	{
		if (fxParam > 0.1f)
			printf("[MATH] WARNING: Passed activation param too large for Leaky ReLU\n");

		// Standard Leaky ReLU threshold is 0.0 (not OUTLIER).
		if (netInput <= 0.0f)
			netOutput = fxParam * netInput; // fxParam should be small positive (e.g. 0.01)
		else
			netOutput = netInput;

		break;
	}
	case LINEAR:
	{
		netOutput = fxParam * netInput;

		break;
	}
	case STEP:
	{
		if (netInput < fxParam)
			netOutput = 0;
		else
			netOutput = 1;

		break;
	}
	}

	return netOutput;
}

float glades::GMath::unsquash(float netInput, int activationFx, float fxParam)
{
	float netOutput = 0.0f;

	switch (activationFx)
	{
	case TANH:
	{
		// Inverse tanh; clamp away from {-1, 1} to avoid inf.
		netOutput = atanhf(clampf(netInput, -1.0f + 1e-7f, 1.0f - 1e-7f));

		break;
	}
	case TANHP:
	{
		// "TANHP" is a clipped tanh in squash(). Here we do best-effort inverse for values
		// in (-1, 1), and saturate outside.
		if (netInput <= -1.0f)
			netOutput = -INLIER; // large negative; avoid returning a tiny value
		else if (netInput >= 1.0f)
			netOutput = INLIER;
		else
			netOutput = atanhf(clampf(netInput, -1.0f + 1e-7f, 1.0f - 1e-7f));

		break;
	}
	case SIGMOID:
	{
		// Inverse sigmoid (logit). Previous implementation was incorrect.
		const float p = clamp_prob01(netInput);
		netOutput = logf(safe_div(p, (1.0f - p)));

		break;
	}
	case SIGMOIDP:
	{
		// Best-effort inverse for "SIGMOIDP" outputs.
		// squash(SIGMOIDP) clamps outputs into [0.01, 0.99] in some ranges; clamp here too.
		const float p = clamp_prob01(netInput);
		netOutput = logf(safe_div(p, (1.0f - p)));

		break;
	}
	case RELU:
	{
		if (netInput <= 0.0f)
			netOutput = 0.0f;
		else
			netOutput = netInput;

		break;
	}
	case LEAKY:
	{
		if (fxParam > 0.1f)
			printf("[MATH] WARNING: Passed activation param too large for Leaky ReLU\n");

		if (netInput <= 0.0f)
			netOutput = safe_div(netInput, fxParam); // fxParam should be small positive
		else
			netOutput = netInput;

		break;
	}
	case LINEAR:
	{
		if (fxParam == 0.0f)
		{
			printf("[MATH] WARNING: Linear unsquash with fxParam=0\n");
			netOutput = 0.0f;
		}
		else
			netOutput = netInput / fxParam;

		break;
	}
	case STEP:
	{
		printf("[MATH] WARNING: Step Function is non-differentiable and can't be unsquashed\n");
		netOutput = 0.0f;

		break;
	}
	}

	return netOutput;
}

float glades::GMath::activationErrDer(float netInput, int activationFx, float fxParam)
{
	float netErrDer = 1.0f;

	switch (activationFx)
	{
	case TANH:
	case TANHP:
	{
		// Tanh der: 1-tanh(x)^2
		netErrDer = 1.0f - (netInput * netInput);

		break;
	}
	case SIGMOID:
	case SIGMOIDP:
	{
		//  Sigmoid der: sigm(x) * (1 - sigm(x))
		netErrDer = netInput * (1.0f - netInput);

		break;
	}
	case RELU:
	{
		// ReLU der: 1 if x > 0; 0 otherwise
		if (netInput > 0.0f)
			netErrDer = 1.0f;
		else
			netErrDer = 0.0f;

		break;
	}
	case LEAKY:
	{
		// Standard Leaky ReLU derivative: 1 if x > 0; fxParam otherwise.
		// fxParam should be small positive (e.g. 0.01).
		netErrDer = (netInput > 0.0f) ? 1.0f : fxParam;

		break;
	}
	case LINEAR:
	{
		// Linear der: always fxParam
		netErrDer = fxParam;

		break;
	}
	case STEP:
	{
		// Step der: "almost surely" 0
		netErrDer = 0.0f;

		break;
	}
	}

	return netErrDer;
}

float glades::GMath::error(float expectation, float prediction)
{
	return (expectation - prediction);
}

float glades::GMath::PercentError(float prediction, float expectation, float meanSqErr)
{
	float percentError = 0.0f;

	// Historically this behaved like a clipped absolute error in [0,1].
	// Make it a defensible "percent-like" error while keeping the old behavior
	// for expectation ~= 0 to avoid massive spikes.
	(void)meanSqErr; // kept for API compatibility
	const float absDiff = fabs(prediction - expectation);
	const float denom = fabs(expectation);
	if (denom < 1e-7f)
		percentError = absDiff;
	else
		percentError = absDiff / denom;

	if (percentError > 1.0f)
		percentError = 1.0f;

	return percentError;
}

float glades::GMath::MeanSquaredError(float expectation, float prediction)
{
	float calculatedError = error(expectation, prediction);
	return (calculatedError * calculatedError);
}

float glades::GMath::CrossEntropyCost(float expectation, float prediction)
{
	// Binary cross-entropy; clamp prediction to avoid log(0).
	const float y = clampf(expectation, 0.0f, 1.0f);
	const float p = clamp_prob01(prediction);
	return -((y * logf(p)) + ((1.0f - y) * logf(1.0f - p)));
}

float glades::GMath::KLDivergence(float expectation, float prediction)
{
	// Element-wise KL contribution: p * log(p / q).
	// If p == 0, contribution is 0 by continuity.
	if (expectation <= 0.0f)
		return 0.0f;

	const float p = clamp_prob01(expectation);
	const float q = clamp_prob01(prediction);
	return p * logf(safe_div(p, q));
}

float glades::GMath::outputNodeCost(float expectation, float prediction, float dataSize, int costFx)
{
	float netCost = 0.0f;
	if (dataSize <= 0.0f)
		dataSize = 1.0f;

	switch (costFx)
	{
	case REGRESSION:
	{
		// Regression uses MSE
		netCost = MeanSquaredError(expectation, prediction);
		netCost /= dataSize;

		break;
	}
	case CLASSIFICATION:
	{
		// Classification uses Cross Entropy
		netCost = CrossEntropyCost(expectation, prediction);
		netCost /= dataSize;

		break;
	}
	case KL:
	{
		// Classification uses Cross Entropy
		netCost = KLDivergence(expectation, prediction);
		netCost /= dataSize;

		break;
	}
	}

	return netCost;
}

float glades::GMath::costErrDer(float expectation, float prediction, int costFx)
{
	float netErrDer = 1.0f;

	switch (costFx)
	{
	case REGRESSION:
	{
		// regression uses MSE cost
		netErrDer = 2.0f * (prediction - expectation); // DIFFERENT THAN error()!!!

		break;
	}
	case CLASSIFICATION:
	{
		// classification uses XENT cost
		// d/dp BCE(y,p) = (p - y) / (p(1-p)), but clamp p for stability.
		const float y = clampf(expectation, 0.0f, 1.0f);
		const float p = clamp_prob01(prediction);
		netErrDer = safe_div((p - y), (p * (1.0f - p)));

		break;
	}
	case KL:
	{
		// Kullback–Leibler divergence cost
		if (expectation <= 0.0f)
			netErrDer = 0.0f;
		else
			netErrDer = -safe_div(expectation, clamp_prob01(prediction));

		break;
	}
	}

	return netErrDer;
}

float glades::GMath::norm_inv_CDF(
	float x) // source = https://stackedboxes.org/2017/05/01/acklams-normal-quantile-function/
{
	double a1 = -39.69683028665376;
	double a2 = 220.9460984245205;
	double a3 = -275.9285104469687;
	double a4 = 138.3577518672690;
	double a5 = -30.66479806614716;
	double a6 = 2.506628277459239;

	double b1 = -54.47609879822406;
	double b2 = 161.5858368580409;
	double b3 = -155.6989798598866;
	double b4 = 66.80131188771972;
	double b5 = -13.280681552885721;

	double c1 = -0.007784894002430293;
	double c2 = -0.3223964580411365;
	double c3 = -2.400758277161838;
	double c4 = -2.549732539343734;
	double c5 = 4.374664141464968;
	double c6 = 2.938163982698783;

	double d1 = 0.007784695709041462;
	double d2 = 0.3224671290700398;
	double d3 = 2.445134137142996;
	double d4 = 3.754408661907416;

	float p_low = 0.02452;
	float p_high = 1 - p_low;

	if (x < p_low)
	{
		double q = sqrt(-2 * log(x));
		return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
			   ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
	}
	else if (x < p_high)
	{
		double q = x - 0.5;
		double r = q * q;
		return ((((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q) /
			   (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1);
	}
	else
	{
		double q = sqrt(-2 * log(1 - x));
		return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
			   ((((d1 * q + d2) * q + d3) * q + d4) * q + 1);
	}
}

float glades::GMath::normal_pdf(float z)
{
	return (1.0f / sqrtf(6.2831853f)) * expf(-(z * z) / 2.0f);
}

int glades::GMath::argmax(const float* data, unsigned int count)
{
	if (!data || count == 0u)
		return 0;
	int best = 0;
	float bestVal = data[0];
	for (unsigned int i = 1; i < count; ++i)
	{
		if (data[i] > bestVal)
		{
			bestVal = data[i];
			best = static_cast<int>(i);
		}
	}
	return best;
}

std::vector<int> glades::GMath::naiveVectorDecomp(const std::vector<float>& needle)
{
	std::vector<int> retVector(needle.size(), 0);

	float max = 0.0f;
	int counter = 0, index = -1;
	std::vector<float>::const_iterator itr = needle.begin();
	for (; itr != needle.end(); ++itr)
	{
		if ((*itr) > max)
		{
			max = (*itr);
			index = counter;
		}
		++counter;
	}

	if (index >= 0)
		retVector[index] = 1;

	return retVector;
}

shmea::GList glades::GMath::naiveVectorDecomp(const shmea::GList& needle)
{
	shmea::GList retList(needle.size(), shmea::GType(0));

	float max = 0.0f;
	int counter = 0, index = -1;
	for (unsigned int i = 0; i < needle.size(); ++i)
	{
		if (needle.getFloat(i) > max)
		{
			max = needle.getFloat(i);
			index = counter;
		}
		++counter;
	}

	if (index >= 0)
		retList.setGType(index, 1);

	return retList;
}
