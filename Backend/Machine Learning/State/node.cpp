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
#include "node.h"
#include "../GMath/gmath.h"
#include "edge.h"

using namespace glades;

glades::Node::Node()
{
	clean();
	activationMutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
	pthread_mutex_init(activationMutex, NULL);
}

glades::Node::Node(const Node& node2)
{
	copy(node2);
}

glades::Node::~Node()
{
	clean();
	pthread_mutex_destroy(activationMutex);
}

void glades::Node::copy(const Node& node2)
{
	id = node2.id;
	weight = node2.weight;
	edges = node2.edges;
	errorDer = node2.errorDer;
	activationMutex = node2.activationMutex;
}

int64_t glades::Node::getID() const
{
	return id;
}

float glades::Node::getWeight() const
{
	return weight;
}

float glades::Node::getEdgeWeight(unsigned int index) const
{
	if (index >= edges.size())
		return 0.0f;

	return edges[index]->getWeight();
}

/*!
 * @brief get an edge id
 * @details get the id for edge at the requested index
 * @param index the edge location in the edges vector
 * @return the edge id at the requested index
 */
int64_t glades::Node::getEdgeID(unsigned int index) const
{
	if (index >= edges.size())
		return 0L;

	return edges[index]->getID();
}

float glades::Node::getActivation() const
{
	float fullActivation = 0.0f;
	for (unsigned int i = 0; i < edges.size(); ++i)
	{
		if (!edges[i]->getActivated())
			continue;

		fullActivation += (edges[i]->getActivation());
	}
	return fullActivation; // * activationScalar;
}

float glades::Node::getActivationScalar() const
{
	return activationScalar;
}

float glades::Node::getErrDer() const
{
	return errorDer;
}

unsigned int glades::Node::numEdges() const
{
	return edges.size();
}

std::vector<float> glades::Node::getPrevDeltas(unsigned int index) const
{
	std::vector<float> empty;
	if (index >= edges.size())
		return empty;

	return edges[index]->getPrevDeltas();
}

float glades::Node::getLastPrevDelta(unsigned int index) const
{
	if (index >= edges.size())
		return 0.0f;

	return edges[index]->getPrevDelta(edges[index]->numPrevDeltas() - 1);
}

void glades::Node::setID(int64_t newID)
{
	id = newID;
}

void glades::Node::setWeight(float newWeight)
{
	weight = newWeight;
}

void glades::Node::setEdges(const std::vector<glades::Edge*>& newEdges)
{
	edges = newEdges;
}

void glades::Node::setEdgeWeight(unsigned int wIndex, float newWeight)
{
	if (wIndex >= edges.size())
		return;

	edges[wIndex]->setWeight(newWeight);
}

void glades::Node::setActivation(unsigned int aIndex, float newActivation)
{
	if (aIndex >= edges.size())
		return;

	pthread_mutex_lock(activationMutex);
	edges[aIndex]->setActivation(newActivation);
	pthread_mutex_unlock(activationMutex);
}

void glades::Node::setActivationScalar(float newActivationScalar)
{
	activationScalar = newActivationScalar;
}

void glades::Node::clearActivation()
{
	pthread_mutex_lock(activationMutex);
	for (unsigned int i = 0; i < edges.size(); ++i)
		edges[i]->Deactivate();
	pthread_mutex_unlock(activationMutex);
}

void glades::Node::adjustErrDer(float newErrorDer)
{
	errorDer += newErrorDer;
}

void glades::Node::clearErrDer()
{
	errorDer = 0;
}

void glades::Node::addPrevDelta(unsigned int index, float newPrevDelta)
{
	if (index >= edges.size())
		return;

	edges[index]->addPrevDelta(newPrevDelta);
}

void glades::Node::clearPrevDeltas(unsigned int index)
{
	if (index >= edges.size())
		return;

	edges[index]->clearPrevDeltas();
}

void glades::Node::clean()
{
	id = -1;
	weight = 0.0f;
	errorDer = 0.0f;
}

void glades::Node::print() const
{
	printf(" [");
	for (unsigned int i = 0; i < edges.size(); ++i)
	{
		printf("%f", getEdgeWeight(i));

		if (i < edges.size() - 1)
			printf(", ");
	}

	printf("]");
}

void glades::Node::initWeights(unsigned int newNumEdges, int initType)
{
	while (numEdges() < newNumEdges)
	{
		if (initType == INIT_RANDOM)
		{
			int randomNum = rand() % 100 + 1; //+1 so we dont divide by zero
			float randomFloat = ((float)(randomNum)) / (((float)100));
			edges.push_back(new glades::Edge(numEdges(), randomFloat));
		}
		else if (initType == INIT_POSRAND)
		{
			int randomNum = rand() % 100 + 1; //+1 so we dont divide by zero
			float randomFloat = ((float)(randomNum)) / (((float)100));
			edges.push_back(new glades::Edge(numEdges(), randomFloat));
		}
		else if (initType == INIT_EMPTY)
			edges.push_back(new glades::Edge(numEdges(), 0.0f));
	}
}

void glades::Node::initWeights(unsigned int newNumEdges, float zigg_layers[],
							   unsigned int num_layers, int initType, int activationType)
{
	float std_dev;
	if (activationType != GMath::RELU)
		std_dev = sqrt(1 / (float)newNumEdges);
	else
		std_dev = sqrt(2 / (float)newNumEdges);
	while (numEdges() < newNumEdges)
	{
		bool accepted = false;
		float candidate_x;
		while (!accepted)
		{
			int layer = rand() % 2048;
			candidate_x = rand() / (RAND_MAX / zigg_layers[layer]);
			if (layer == 0) // tail; possibly not worth calculating values past 3 std devs, maybe
							// revisit another day
				continue;
			else if (candidate_x < zigg_layers[layer + 1])
				accepted = true;
			else if (candidate_x > zigg_layers[layer + 1])
			{
				float randU = rand() / RAND_MAX;
				float candidate_y =
					GMath::normal_pdf(zigg_layers[layer]) +
					randU * (GMath::normal_pdf(zigg_layers[layer - 1] -
											   GMath::normal_pdf(zigg_layers[layer])));
				if (candidate_y < GMath::normal_pdf(candidate_x))
					accepted = true;
			}
		}
		candidate_x = candidate_x * std_dev;
		edges.push_back(new glades::Edge(numEdges(), candidate_x));
	}
}

void glades::Node::getDelta(unsigned int index, float baseError, float cInputNodeWeight,
							float learningRate, float momentumFactor, float weightDecay1, float weightDecay2)
{
	if (index >= edges.size())
		return;

	// calculate the optimized delta rule
	float deltaW =
		((baseError * cInputNodeWeight) + (momentumFactor * getLastPrevDelta(index)) // momentum
		 + (weightDecay1 * learningRate * abs(cInputNodeWeight)) // weight decay L1
		 + (weightDecay2 * learningRate * cInputNodeWeight)); // weight decay L2

	// Add the new PrevDelta
	addPrevDelta(index, deltaW);
}

void glades::Node::applyDeltas(unsigned int index, int minibatchSize)
{
	if (index >= edges.size())
		return;

	float deltaW = 0.0f;
	for (int i = 0; i < minibatchSize; ++i)
		deltaW += edges[index]->getPrevDelta(i);

	deltaW /= minibatchSize;

	// Set the new weight
	setEdgeWeight(index, getEdgeWeight(index) - deltaW);
}
