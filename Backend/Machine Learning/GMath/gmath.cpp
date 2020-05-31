// Copyright 2020 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
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

float glades::GMath::squash(float netInput, int activationFx, float fxParam)
{
	float netOutput = 0.0f;

	switch (activationFx)
	{
	case TANH:
	{
		netOutput = tanh(netInput);

		break;
	}
	case TANHP:
	{
		if (netInput > 1.0f - fxParam)
			netOutput = 1.0f;
		else if (netInput < fxParam - 1.0f)
			netOutput = -1.0f;
		else
			netOutput = tanh(netInput);

		break;
	}
	case SIGMOID:
	{
		netOutput = 1.0f / (1.0f + exp(-netInput));

		break;
	}
	case SIGMOIDP:
	{
		if (netInput < fxParam)
			netOutput = 0.01f;
		else if (netInput > 1.0f - fxParam)
			netOutput = 0.99f;
		else
			netOutput = 1.0f / (1.0f + exp(-netInput));

		break;
	}
	case RELU:
	{
		if (netInput < OUTLIER) // including negative vals
			netOutput = 0.0f;
		else
			netOutput = netInput;

		break;
	}
	case LEAKY:
	{
		if (fxParam > 0.1f)
			printf("[MATH] WARNING: Passed activation param too large for Leaky ReLU\n");

		if (netInput < OUTLIER)
			netOutput = fxParam * netInput; // fxParam should be small
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
		netOutput = atanh(netInput);

		break;
	}
	case TANHP:
	{
		if (netInput <= -1.0f)
			netOutput = -fxParam;
		else if (netInput >= 1.0f)
			netOutput = fxParam;
		else
			netOutput = atanh(netInput);

		break;
	}
	case SIGMOID:
	{
		netOutput = (1.0f + exp(-netInput));

		break;
	}
	case SIGMOIDP:
	{
		if (netInput <= 0.0f)
			netOutput = 1 - fxParam;
		else if (netInput >= 1.0f)
			netOutput = fxParam;
		else
			netOutput = (1.0f + exp(-netInput));

		break;
	}
	case RELU:
	{
		if (netInput <= 0.0f)
			netOutput = OUTLIER;
		else
			netOutput = netInput;

		break;
	}
	case LEAKY:
	{
		if (fxParam > 0.1f)
			printf("[MATH] WARNING: Passed activation param too large for Leaky ReLU\n");

		if (netInput <= OUTLIER * fxParam)
			netOutput = netInput / fxParam; // fxParam should be small
		else
			netOutput = netInput;

		break;
	}
	case LINEAR:
	{
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
		// Leaky ReLU der: 1 if x > 0; fxParam otherwise
		if (netInput > 0.0f)
			netErrDer = 1.0f;
		else
			// Leaky part of deriv should be negative
			if (fxParam < 0)
			netErrDer = fxParam;
		else
			netErrDer = -fxParam;

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

	// Cannot divide by zero
	// if (expectation == 0.0f)
	// 	percentError = (prediction - expectation);
	// else
	// 	percentError = ((prediction - expectation) / expectation);
	percentError = (prediction - expectation);

	if (percentError < 0.0f)
		percentError = -percentError;

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
	return -((expectation * log(prediction)) + ((1 - expectation) * log(1 - prediction)));
}

float glades::GMath::KLDivergence(float expectation, float prediction)
{
	return log(expectation / prediction);
}

float glades::GMath::outputNodeCost(float expectation, float prediction, float dataSize, int costFx)
{
	float netCost = 0.0f;

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
		netErrDer = (prediction - expectation) / ((1 - prediction) * prediction);

		break;
	}
	case KL:
	{
		// Kullback–Leibler divergence cost
		netErrDer = -(expectation / prediction);

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
	return (1 / sqrt(6.2831853)) * exp(-(z * z) / 2);
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
