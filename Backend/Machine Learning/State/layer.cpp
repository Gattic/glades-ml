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

Layer::Layer(int64_t newID, int newType, float newBias)
{
	id = newID;
	biasWeight = newBias;
	type = newType;
}

Layer::Layer(int newType)
{
	id = -1;
	biasWeight = 0.0f;
	type = newType;
}

Layer::~Layer()
{
	id = -1;
	biasWeight = 0.0f;
	type = 0;
	children.clear();
	dropoutFlag.clear();
}

int64_t Layer::getID() const
{
	return id;
}

float Layer::getBiasWeight() const
{
	return biasWeight;
}

int Layer::getType() const
{
	return type;
}

unsigned int Layer::size() const
{
	return children.size();
}

void Layer::setID(int64_t newID)
{
	id = newID;
}
void Layer::setBiasWeight(float newBiasWeight)
{
	biasWeight = newBiasWeight;
}

void Layer::setType(int newType)
{
	type = newType;
}

const std::vector<shmea::GPointer<Node> >& Layer::getChildren() const
{
	return children;
}

shmea::GPointer<Node> Layer::getNode(unsigned int index)
{
	if (index >= children.size())
		return shmea::GPointer<Node>();

	return children[index];
}

void Layer::generateDropout(float p)
{
	int cDropoutRate = p * 100.0f;
	if (cDropoutRate < 0)
		return;

	if (cDropoutRate >= 100)
		return;

	bool fullLayerDropped = true;
	do
	{
		clearDropout();
		fullLayerDropped = true;

		// generate the dropout probabilities
		while (dropoutFlag.size() < size())
		{
			bool cDropped = false;
			int dart = (rand() % 100) + 1; // 1-100 (100 possibilities)
			if (dart <= cDropoutRate)
				cDropped = true;
			else
				fullLayerDropped = false;

			// Populate the dropout vector
			dropoutFlag.push_back(cDropped);
		}

	} while (fullLayerDropped);
}

bool Layer::possiblePath(unsigned int index) const
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

unsigned int Layer::firstValidPath() const
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

unsigned int Layer::lastValidPath() const
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

void Layer::clearDropout()
{
	dropoutFlag.clear();
}

void Layer::addNode(const shmea::GPointer<Node>& child)
{
	if (child)
		children.push_back(child);
}

void Layer::initWeights(int prevLayerSize, unsigned int cLayerSize, int initType,
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
			shmea::GPointer<Node> newNode(new Node());
			newNode->initWeights(prevLayerSize, zigg_layers, num_layers, initType, activationType);
			addNode(newNode);
		}
	}
	else
	{
		while (size() < cLayerSize)
		{
			shmea::GPointer<Node> newNode(new Node());
			newNode->initWeights(prevLayerSize, initType);
			addNode(newNode);
		}
	}
}

void Layer::clean()
{
	std::vector<shmea::GPointer<Node> >::iterator itr = children.begin();
	for (; itr != children.end(); ++itr)
	{
		if (((*itr)->getWeight() < 0.1f) && ((*itr)->getWeight() > -0.1f))
		{
			itr = children.erase(itr);
			--itr;
		}
	}
}

void Layer::print() const
{
	printf("[");
	std::vector<shmea::GPointer<Node> >::const_iterator itr = children.begin();
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

shmea::GPointer<Node> Layer::operator[](unsigned int index)
{
	if (index >= size())
		return shmea::GPointer<Node>();

	return children[index];
}
