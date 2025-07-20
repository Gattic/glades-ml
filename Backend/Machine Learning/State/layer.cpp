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
#include "layer.h"
#include "../GMath/gmath.h"
#include "node.h"

using namespace glades;

glades::Layer::Layer(int64_t newID, int newType, float newBias)
{
	id = newID;
	biasWeight = newBias;
	type = newType;
	useBatchNorm = false;
	batchNormMomentum = 0.9f;
	batchNormEpsilon = 1e-5f;
}

glades::Layer::Layer(int newType)
{
	id = -1;
	biasWeight = 0.0f;
	type = newType;
	useBatchNorm = false;
	batchNormMomentum = 0.9f;
	batchNormEpsilon = 1e-5f;
}

glades::Layer::~Layer()
{
	id = -1;
	biasWeight = 0.0f;
	type = 0;
	children.clear();
	dropoutFlag.clear();
	
	// Clean up batch normalization variables
	batchNormGamma.clear();
	batchNormBeta.clear();
	batchNormMean.clear();
	batchNormVar.clear();
	batchNormXNorm.clear();
	batchNormXCentered.clear();
}

int64_t glades::Layer::getID() const
{
	return id;
}

float glades::Layer::getBiasWeight() const
{
	return biasWeight;
}

int glades::Layer::getType() const
{
	return type;
}

unsigned int glades::Layer::size() const
{
	return children.size();
}

void glades::Layer::setID(int64_t newID)
{
	id = newID;
}
void glades::Layer::setBiasWeight(float newBiasWeight)
{
	biasWeight = newBiasWeight;
}

void glades::Layer::setType(int newType)
{
	type = newType;
}

const std::vector<glades::Node*>& glades::Layer::getChildren() const
{
	return children;
}

glades::Node* glades::Layer::getNode(unsigned int index)
{
	if (index >= children.size())
		return NULL;

	return children[index];
}

void glades::Layer::setupDropout()
{
	// Populate the dropout vector
	while (dropoutFlag.size() < size())
	{
		dropoutFlag.push_back(false);
	}
}

void glades::Layer::generateDropout(float p)
{
	if (dropoutFlag.size() < size())
	    setupDropout();
	
	int cDropoutRate = p * 100.0f;
	if (cDropoutRate < 0)
		return;

	if (cDropoutRate >= 100)
		return;

	bool fullLayerDropped = true;
	do
	{
		fullLayerDropped = true;

		// generate the dropout probabilities
		for (unsigned int i = 0; i < size(); ++i)
		{
			bool cDropped = false;
			int dart = (rand() % 100) + 1; // 1-100 (100 possibilities)
			if (dart <= cDropoutRate)
				cDropped = true;
			else
				fullLayerDropped = false;

			// Populate the dropout vector
			dropoutFlag[i] = cDropped;
		}

	} while (fullLayerDropped);
}

bool glades::Layer::possiblePath(unsigned int index) const
{
	// cannot perform dropout on this layer
	if (type == OUTPUT_TYPE)
		return true;

	if (dropoutFlag.size() == 0)
		return true;

	if (index >= dropoutFlag.size())
		return true;

	// return the element
	return !dropoutFlag[index];
}

unsigned int glades::Layer::firstValidPath() const
{
	// cannot perform dropout on this layer
	if (type == OUTPUT_TYPE)
		return 0;

	for (unsigned int i = 0; i < dropoutFlag.size(); ++i)
	{
		bool possible = (!dropoutFlag[i]);
		if (possible)
			return i;
	}

	return (unsigned int)-1;
}

unsigned int glades::Layer::lastValidPath() const
{
	// cannot perform dropout on this layer
	// output layer dropoutFlag.size() == 0
	if (type == OUTPUT_TYPE)
		return children.size() - 1;

	for (int i = dropoutFlag.size() - 1; i >= 0; --i)
	{
		bool possible = (!dropoutFlag[i]);
		if (possible)
			return i;
	}

	return (unsigned int)-1;
}

void glades::Layer::clearDropout()
{
	dropoutFlag.clear();
}

void glades::Layer::addNode(Node* child)
{
	if (child)
		children.push_back(child);
}

void glades::Layer::initWeights(int prevLayerSize, unsigned int cLayerSize, int initType,
								int activationType)
{
	if (initType == Node::INIT_XAVIER || initType == Node::INIT_POSXAVIER)
	{
		unsigned int num_layers = 2048;
		float zigg_layers[num_layers + 1]; // array of pre-calculated x values
		float normal_CDF = 1.0;
		for (unsigned int i = 1; i < num_layers; i++) // assigning x values to layers
		{
			normal_CDF = normal_CDF - (1 / ((float)num_layers * 2));
			zigg_layers[i] = GMath::norm_inv_CDF(normal_CDF);
		}
		zigg_layers[0] = zigg_layers[1];
		zigg_layers[num_layers] = 0;
		while (size() < cLayerSize)
		{
			Node* newNode = new Node();
			newNode->initWeights(prevLayerSize, zigg_layers, num_layers, initType, activationType);
			addNode(newNode);
		}
	}
	else
	{
		while (size() < cLayerSize)
		{
			Node* newNode = new Node();
			newNode->initWeights(prevLayerSize, initType);
			addNode(newNode);
		}
	}
}

std::vector<Node*>::iterator glades::Layer::removeNode(Node* child)
{
	if (child)
	{
		std::vector<Node*>::iterator itr = children.begin();
		for (; itr != children.end(); ++itr)
		{
			if ((*itr) == child)
				return children.erase(itr);
		}
	}
	return children.end();
}

void glades::Layer::clean()
{
	std::vector<Node*>::iterator itr = children.begin();
	for (; itr != children.end(); ++itr)
	{
		if (((*itr)->getWeight() < 0.1f) && ((*itr)->getWeight() > -0.1f))
		{
			itr = removeNode(*itr);
			--itr;
		}
	}
}

void glades::Layer::print() const
{
	printf("[");
	std::vector<Node*>::const_iterator itr = children.begin();
	while (itr != children.end())
	{
		printf("%f(%d)", (*itr)->getWeight(), (*itr)->numEdges());

		// print the weights
		if ((type == HIDDEN_TYPE) || (type == OUTPUT_TYPE))
		{
			(*itr)->print();
		}

		++itr;
		if (itr != children.end())
			printf(", ");
	}
	printf("]");

	// print bias
	if (type == HIDDEN_TYPE)
		printf(" + [B=%f]", getBiasWeight());
	printf("\n");
}

Node* glades::Layer::operator[](unsigned int index)
{
	if (index >= size())
		return NULL;

	return children[index];
}

void Layer::setupContext()
{
	for(unsigned int i=0;i<children.size();++i)
	{
		printf("SETUP CONTEXT: %u\n", i);
		shmea::GPointer<Node> newNode(new Node());
		newNode->initWeights(1, Node::INIT_POSRAND);
		children[i]->setContextNode(newNode);
	}
}

void Layer::setupBatchNorm(float momentum, float epsilon)
{
	useBatchNorm = true;
	batchNormMomentum = momentum;
	batchNormEpsilon = epsilon;
	
	// Initialize batch norm parameters for each node in the layer
	batchNormGamma.resize(children.size(), 1.0f);
	batchNormBeta.resize(children.size(), 0.0f);
	batchNormMean.resize(children.size(), 0.0f);
	batchNormVar.resize(children.size(), 1.0f);
	batchNormXNorm.resize(children.size(), 0.0f);
	batchNormXCentered.resize(children.size(), 0.0f);
}

bool Layer::isBatchNormEnabled() const
{
	return useBatchNorm;
}

void Layer::enableBatchNorm(bool enable)
{
	useBatchNorm = enable;
}

float Layer::getBatchNormGamma(unsigned int index) const
{
	if (index < batchNormGamma.size())
		return batchNormGamma[index];
	return 1.0f;
}

float Layer::getBatchNormBeta(unsigned int index) const
{
	if (index < batchNormBeta.size())
		return batchNormBeta[index];
	return 0.0f;
}

float Layer::getBatchNormMean(unsigned int index) const
{
	if (index < batchNormMean.size())
		return batchNormMean[index];
	return 0.0f;
}

float Layer::getBatchNormVar(unsigned int index) const
{
	if (index < batchNormVar.size())
		return batchNormVar[index];
	return 1.0f;
}

void Layer::setBatchNormGamma(unsigned int index, float value)
{
	if (index < batchNormGamma.size())
		batchNormGamma[index] = value;
}

void Layer::setBatchNormBeta(unsigned int index, float value)
{
	if (index < batchNormBeta.size())
		batchNormBeta[index] = value;
}

void Layer::setBatchNormMean(unsigned int index, float value)
{
	if (index < batchNormMean.size())
		batchNormMean[index] = value;
}

void Layer::setBatchNormVar(unsigned int index, float value)
{
	if (index < batchNormVar.size())
		batchNormVar[index] = value;
}

void Layer::updateBatchNormStats(unsigned int index, float mean, float var)
{
	if (index < batchNormMean.size() && index < batchNormVar.size())
	{
		batchNormMean[index] = batchNormMomentum * batchNormMean[index] + (1.0f - batchNormMomentum) * mean;
		batchNormVar[index] = batchNormMomentum * batchNormVar[index] + (1.0f - batchNormMomentum) * var;
	}
}

float Layer::applyBatchNorm(unsigned int index, float input, bool training)
{
	if (!useBatchNorm || index >= children.size())
		return input;
	
	if (training)
	{
		// During training, use batch statistics
		// For a more realistic implementation, we would accumulate statistics across the batch
		// Here we use a simplified approach that updates running statistics with current input
		float currentMean = input;
		float currentVar = 0.0f; // Simplified - in practice, compute variance across batch
		
		// Update running statistics using exponential moving average
		updateBatchNormStats(index, currentMean, currentVar);
		
		// For training, use current input statistics (simplified batch statistics)
		float mean = currentMean;
		float var = currentVar + 1e-8f; // Add small constant for numerical stability
		
		// Cache for backpropagation
		batchNormXCentered[index] = input - mean;
		batchNormXNorm[index] = batchNormXCentered[index] / sqrt(var + batchNormEpsilon);
	}
	else
	{
		// During inference, use running statistics
		batchNormXCentered[index] = input - batchNormMean[index];
		batchNormXNorm[index] = batchNormXCentered[index] / sqrt(batchNormVar[index] + batchNormEpsilon);
	}
	
	// Apply scale and shift: y = γ * x_norm + β
	return batchNormGamma[index] * batchNormXNorm[index] + batchNormBeta[index];
}

float Layer::getBatchNormGradient(unsigned int index, float gradient)
{
	if (!useBatchNorm || index >= children.size())
		return gradient;
	
	// Use the complete gradient computation from GMath
	float gammaGrad, betaGrad, inputGradOut;
	float normalized = batchNormXNorm[index];
	float gamma = batchNormGamma[index];
	float beta = batchNormBeta[index];
	float mean = batchNormMean[index];
	float variance = batchNormVar[index];
	
	// Compute all gradients using the complete implementation
	GMath::batchNormGradients(gradient, normalized, gamma, beta, mean, variance, 
							  batchNormEpsilon, 1, gammaGrad, betaGrad, inputGradOut);
	
	// Update the batch normalization parameters
	updateBatchNormGamma(index, gammaGrad, 0.01f); // Use a small learning rate for batch norm params
	updateBatchNormBeta(index, betaGrad, 0.01f);
	
	// Return the gradient with respect to the input
	return inputGradOut;
}

void Layer::resetBatchNormCache()
{
	for (unsigned int i = 0; i < batchNormXNorm.size(); ++i)
	{
		batchNormXNorm[i] = 0.0f;
		batchNormXCentered[i] = 0.0f;
	}
}

void glades::Layer::updateBatchNormGamma(unsigned int index, float gradient, float learningRate)
{
	if (!useBatchNorm || index >= batchNormGamma.size())
		return;
	
	// Update gamma parameter: γ = γ - learningRate * ∂L/∂γ
	batchNormGamma[index] -= learningRate * gradient;
}

void glades::Layer::updateBatchNormBeta(unsigned int index, float gradient, float learningRate)
{
	if (!useBatchNorm || index >= batchNormBeta.size())
		return;
	
	// Update beta parameter: β = β - learningRate * ∂L/∂β
	batchNormBeta[index] -= learningRate * gradient;
}

float glades::Layer::getBatchNormGammaGradient(unsigned int index, float inputGradient, float normalized)
{
	if (!useBatchNorm || index >= batchNormGamma.size())
		return 0.0f;
	
	// ∂L/∂γ = ∂L/∂y * x_norm
	return inputGradient * normalized;
}

float glades::Layer::getBatchNormBetaGradient(unsigned int index, float inputGradient)
{
	if (!useBatchNorm || index >= batchNormBeta.size())
		return 0.0f;
	
	// ∂L/∂β = ∂L/∂y
	return inputGradient;
}
