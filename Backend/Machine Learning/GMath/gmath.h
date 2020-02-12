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
#ifndef _GMATH
#define _GMATH

#include "../../../include/Backend/Database/GList.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

namespace glades {

class GMath
{
public:
	static const float INLIER = 0.954f;
	static const float OUTLIER = 0.046f;

	// cost function flags
	static const int REGRESSION = 0;
	static const int CLASSIFICATION = 1;
	static const int KL = 2;

	// activation function flags
	static const int TANH = 0;
	static const int TANHP = 1;
	static const int SIGMOID = 2;
	static const int SIGMOIDP = 3;
	static const int LINEAR = 4;
	static const int RELU = 5;
	static const int LEAKY = 6;
	static const int STEP = 7;

	// standardization flags
	static const int MINMAX = 0;
	static const int ZSCORE = 1;

	static float squash(float, int, float = 0.1f);
	static float unsquash(float, int, float = 0.1f);
	static float activationErrDer(float, int, float = 0.1f);
	static float error(float, float);
	static float PercentError(float, float, float);
	static float MeanSquaredError(float, float);
	static float CrossEntropyCost(float, float);
	static float KLDivergence(float, float);
	static float costErrDer(float, float, int);
	static float outputNodeCost(float, float, float, int);
	static float norm_inv_CDF(float); // inverse CDF of normal distribution
	static float normal_pdf(float);
	static std::vector<int> naiveVectorDecomp(const std::vector<float>&);
	static shmea::GList naiveVectorDecomp(const shmea::GList&);
};
};

#endif
