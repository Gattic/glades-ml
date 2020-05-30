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
#ifndef _GQL_HIDDENLAYERINFO
#define _GQL_HIDDENLAYERINFO

#include "Backend/Database/GList.h"
#include "layerinfo.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

namespace shmea {
class GList;
};

namespace glades {

class HiddenLayerInfo : public LayerInfo
{
private:
	float learningRate;
	float momentumFactor;
	float weightDecay;
	float pHidden;
	int activationType;
	float activationParam;

public:
	HiddenLayerInfo(int, float, float, float, float, int, float);
	~HiddenLayerInfo();

	void copyParamsFrom(const HiddenLayerInfo*);

	// gets
	float getLearningRate() const;
	float getMomentumFactor() const;
	float getWeightDecay() const;
	float getPHidden() const;
	int getActivationType() const;
	float getActivationParam() const;
	shmea::GList getGTableRow() const;

	// sets
	void setLearningRate(float);
	void setMomentumFactor(float);
	void setWeightDecay(float);
	void setPHidden(float);
	void setActivationType(int);
	void setActivationParam(float);

	int getLayerType() const;
};
};

#endif
