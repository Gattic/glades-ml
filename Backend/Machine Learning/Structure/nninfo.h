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
#ifndef _GQL_NNINFO
#define _GQL_NNINFO

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include "Backend/Database/GString.h"

namespace shmea {
class GTable;
};

namespace glades {

class LayerInfo;
class InputLayerInfo;
class HiddenLayerInfo;
class OutputLayerInfo;
class NNetwork;
class RNN;

class NNInfo
{
private:
	friend NNetwork;
	friend RNN;

	shmea::GString name;
	int inputType; // DataInput enum: 0 = csv, 1 = image, 2 = text
	InputLayerInfo* inputLayer;
	OutputLayerInfo* outputLayer;
	std::vector<HiddenLayerInfo*> layers;
	int hiddenLayerCount;
	int batchSize;

	//
	shmea::GTable toGTable() const;
	bool fromGTable(const shmea::GString&, const shmea::GTable&);

	// Database
	bool load(const shmea::GString&);
	void save() const;

public:
	static const int BATCH_FULL = 0;
	static const int BATCH_STOCHASTIC = 1;

	// structure: size, pInput, batchSize, learningRate, momentumFactor, weightDecay1, weightDecay2, pDropout,
	// activationType,
	// activationParam, outputType
	static const int COL_SIZE = 0;
	static const int COL_BATCH_SIZE = 1;
	static const int COL_LEARNING_RATE = 2;
	static const int COL_MOMENTUM_FACTOR = 3;
	static const int COL_WEIGHT_DECAY1 = 4;
	static const int COL_WEIGHT_DECAY2 = 5;
	static const int COL_PDROPOUT = 6;
	static const int COL_ACTIVATION_TYPE = 7;
	static const int COL_ACTIVATION_PARAM = 8;
	static const int COL_OUTPUT_TYPE = 9;

	NNInfo(const shmea::GString&);
	NNInfo(const shmea::GString&, const shmea::GTable&);
	NNInfo(const shmea::GString&, InputLayerInfo*, const std::vector<HiddenLayerInfo*>&,
		   OutputLayerInfo*);
	~NNInfo();

	// gets
	shmea::GString getName() const;
	int getInputType() const;
	int getOutputType() const;
	float getPInput() const;
	int getBatchSize() const;
	InputLayerInfo* getInputLayer() const;
	std::vector<HiddenLayerInfo*> getLayers() const;
	int numHiddenLayers() const;
	int getInputLayerSize() const;
	int getHiddenLayerSize(unsigned int) const;
	unsigned int getOutputLayerSize() const;
	float getLearningRate(unsigned int) const;
	float getMomentumFactor(unsigned int) const;
	float getWeightDecay1(unsigned int) const;
	float getWeightDecay2(unsigned int) const;
	float getPDropout(unsigned int) const;
	int getActivationType(unsigned int) const;
	float getActivationParam(unsigned int) const;
	void print() const;

	// sets
	void setName(shmea::GString);
	void setInputType(int);
	void setOutputType(int);
	void setOutputSize(int);
	void setPInput(float);
	void setBatchSize(int);
	void setLayers(const std::vector<HiddenLayerInfo*>&);
	void setLearningRate(unsigned int, float);
	void setMomentumFactor(unsigned int, float);
	void setWeightDecay1(unsigned int, float);
	void setWeightDecay2(unsigned int, float);
	void setPDropout(unsigned int, float);
	void setActivationType(unsigned int, int);
	void setActivationParam(unsigned int, float);
	void addHiddenLayer(HiddenLayerInfo*);
	void copyHiddenLayer(unsigned int, unsigned int);
	void resizeHiddenLayers(unsigned int);
	void removeHiddenLayer(unsigned int);
};
};

#endif
