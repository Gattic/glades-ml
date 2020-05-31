// Copyright 2020 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan, Kevin Ko
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
#ifndef _GANALYSIS
#define _GANALYSIS
#include "GTable.h"
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

namespace shmea {

class GAnalysis
{
private:
	GTable inputData;

	std::vector<float> SimpleMovingAverage(const std::vector<float>&, unsigned int);
	std::vector<float> RateofChange(const std::vector<float>&, unsigned int);
	std::vector<float> ExpMovingAverage(const std::vector<float>&, unsigned int);
	std::pair<std::vector<float>, std::vector<float> > Bollinger(const std::vector<float>&,
																 unsigned int, unsigned int);
	std::vector<float> GainLoss(const std::vector<float>&);
	std::vector<float> RSI(const std::vector<float>&, unsigned int);
	std::vector<float> OBV(const std::vector<float>&, const std::vector<float>&);
	std::pair<std::vector<float>, std::vector<float> > Vortex(const std::vector<float>&,
															  const std::vector<float>&,
															  const std::vector<float>&,
															  unsigned int);
	std::vector<float> MFI(const std::vector<float>&, const std::vector<float>&,
						   const std::vector<float>&, const std::vector<float>&, unsigned int);
	std::vector<float> Stochastic(const std::vector<float>&, const std::vector<float>&,
								  const std::vector<float>&, unsigned int);
	std::vector<float> VWAP(const std::vector<float>&, const std::vector<float>&,
							const std::vector<float>&, const std::vector<float>&, unsigned int);
	std::vector<float> WilliamsR(const std::vector<float>&, const std::vector<float>&,
								 const std::vector<float>&, unsigned int);

public:
	static void test();

	GAnalysis(const GTable&);
	void Apply(unsigned int, unsigned int);
	void Apply(unsigned int);
	GTable getResult();
};
};

#endif
