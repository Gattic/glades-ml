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
#ifndef _LAYER
#define _LAYER

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

namespace glades {

class Node;

class Layer
{
private:
	std::vector<Node*> children;
	std::vector<bool> dropoutFlag;
	int64_t id;
	float biasWeight;
	int type;
	
	// Batch normalization parameters
	bool useBatchNorm;
	std::vector<float> batchNormGamma;  // Scale parameter
	std::vector<float> batchNormBeta;   // Shift parameter
	std::vector<float> batchNormMean;   // Running mean
	std::vector<float> batchNormVar;    // Running variance
	std::vector<float> batchNormXNorm;  // Normalized input cache
	std::vector<float> batchNormXCentered; // Centered input cache
	float batchNormMomentum;            // Momentum for running statistics
	float batchNormEpsilon;             // Small constant for numerical stability

public:
	static const int INPUT_TYPE = 0;
	static const int HIDDEN_TYPE = 1;
	static const int OUTPUT_TYPE = 2;

	// constructors and destructors
	Layer(int64_t, int, float = 0.0f);
	Layer(int);
	~Layer();

	// gets
	int64_t getID() const;
	float getBiasWeight() const;
	int getType() const;
	unsigned int size() const;
	bool possiblePath(unsigned int) const;
	unsigned int firstValidPath() const;
	unsigned int lastValidPath() const;

	// sets
	void setID(int64_t);
	void setBiasWeight(float);
	void setType(int);

	// children
	const std::vector<glades::Node*>& getChildren() const;
	Node* getNode(unsigned int);
	void setupDropout();
	void generateDropout(float);
	void clearDropout();
	void addNode(Node*);
	void initWeights(int, unsigned int, int, int);
	std::vector<Node*>::iterator removeNode(Node*);
	void clean();
	void print() const;

	Node* operator[](unsigned int);

	void setupContext();
	
	// Batch normalization methods
	void setupBatchNorm(float momentum = 0.9f, float epsilon = 1e-5f);
	bool isBatchNormEnabled() const;
	void enableBatchNorm(bool enable);
	float getBatchNormGamma(unsigned int index) const;
	float getBatchNormBeta(unsigned int index) const;
	float getBatchNormMean(unsigned int index) const;
	float getBatchNormVar(unsigned int index) const;
	void setBatchNormGamma(unsigned int index, float value);
	void setBatchNormBeta(unsigned int index, float value);
	void setBatchNormMean(unsigned int index, float value);
	void setBatchNormVar(unsigned int index, float value);
	void updateBatchNormStats(unsigned int index, float mean, float var);
	float applyBatchNorm(unsigned int index, float input, bool training = true);
	float getBatchNormGradient(unsigned int index, float gradient);
	void resetBatchNormCache();
	
	// Batch normalization parameter updates
	void updateBatchNormGamma(unsigned int index, float gradient, float learningRate);
	void updateBatchNormBeta(unsigned int index, float gradient, float learningRate);
	float getBatchNormGammaGradient(unsigned int index, float inputGradient, float normalized);
	float getBatchNormBetaGradient(unsigned int index, float inputGradient);
};
};

#endif
