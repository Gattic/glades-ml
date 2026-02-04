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
#include "node.h"
#include "../GMath/gmath.h"
#include "edge.h"
#include "../rng.h"

using namespace glades;

glades::Node::Node()
{
	clean();
}

glades::Node::Node(const Node& node2)
{
	copy(node2);
}

glades::Node::~Node()
{
	clean();
}

glades::Node& glades::Node::operator=(const Node& node2)
{
	if (this != &node2)
		copy(node2);
	return *this;
}

void glades::Node::copy(const Node& node2)
{
	weight = node2.weight;
	edges = node2.edges;
	errorDer = node2.errorDer;
	activationScalar = node2.activationScalar;
	cellState = node2.cellState;
	contextNode = node2.contextNode;
}

float glades::Node::getWeight() const
{
	return weight;
}

float glades::Node::getCellState() const
{
	return cellState;
}

float glades::Node::getEdgeWeight(unsigned int index) const
{
	if (index >= edges.size())
		return 0.0f;

	return edges[index]->getWeight();
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

	// Deprecated: we no longer store per-sample delta vectors on edges.
	return empty;
}

float glades::Node::getLastPrevDelta(unsigned int index) const
{
	if (index >= edges.size())
		return 0.0f;

	// Momentum uses the edge's velocity (last update step).
	return edges[index]->getVelocity();
}

void glades::Node::setWeight(float newWeight)
{
	weight = newWeight;
}

void glades::Node::setCellState(float newCellState)
{
	cellState = newCellState;
}

void glades::Node::setEdges(const std::vector<shmea::GPointer<glades::Edge> >& newEdges)
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

	edges[aIndex]->setActivation(newActivation);
}

void glades::Node::setActivationScalar(float newActivationScalar)
{
	activationScalar = newActivationScalar;
}

void glades::Node::clearActivation()
{
	for (unsigned int i = 0; i < edges.size(); ++i)
		edges[i]->Deactivate();
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
	weight = 0.0f;
	errorDer = 0.0f;
	activationScalar = 0.0f;
	cellState = 0.0f;
	edges.clear();
	contextNode.reset();
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
			const int randomNum = glades::rng::uniform_int(1, 100); // 1..100
			const float randomFloat = (static_cast<float>(randomNum)) / 100.0f;
			edges.push_back(shmea::GPointer<glades::Edge>(new glades::Edge(numEdges(), randomFloat)));
		}
		else if (initType == INIT_POSRAND)
		{
			const int randomNum = glades::rng::uniform_int(1, 100); // 1..100
			const float randomFloat = (static_cast<float>(randomNum)) / 100.0f;
			edges.push_back(shmea::GPointer<glades::Edge>(new glades::Edge(numEdges(), randomFloat)));
		}
		else if (initType == INIT_EMPTY)
			edges.push_back(shmea::GPointer<glades::Edge>(new glades::Edge(numEdges(), 0.0f)));
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
			const int layer = glades::rng::uniform_int(0, 2047);
			// candidate_x in [0, zigg_layers[layer])
			candidate_x = static_cast<float>(glades::rng::uniform_double(0.0, static_cast<double>(zigg_layers[layer])));
			if (layer == 0) // tail; possibly not worth calculating values past 3 std devs, maybe
							// revisit another day
				continue;
			else if (candidate_x < zigg_layers[layer + 1])
				accepted = true;
			else if (candidate_x > zigg_layers[layer + 1])
			{
				const float randU = static_cast<float>(glades::rng::uniform_double(0.0, 1.0));
				float candidate_y =
					GMath::normal_pdf(zigg_layers[layer]) +
					randU * (GMath::normal_pdf(zigg_layers[layer - 1] -
											   GMath::normal_pdf(zigg_layers[layer])));
				if (candidate_y < GMath::normal_pdf(candidate_x))
					accepted = true;
			}
		}
		candidate_x = candidate_x * std_dev;
		edges.push_back(shmea::GPointer<glades::Edge>(new glades::Edge(numEdges(), candidate_x)));
	}
}

void glades::Node::getDelta(unsigned int index, float baseError, float cInputNodeWeight,
							float learningRate, float momentumFactor, float weightDecay1, float weightDecay2)
{
	if (index >= edges.size())
		return;

	// Calculate the optimized delta rule.
	//
	// IMPORTANT: weightDecay1/2 are regularization terms on the *weight*, not on the input
	// activation. The historical implementation incorrectly used cInputNodeWeight, which
	// makes the "decay" depend on data scale rather than parameter magnitude.
	//
	// L1: lambda1 * sign(w)
	// L2: lambda2 * w
	const float w = getEdgeWeight(index);
	const float wSign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);

	const float deltaW =
		((baseError * cInputNodeWeight) +
		 (momentumFactor * getLastPrevDelta(index)) +
		 (weightDecay1 * learningRate * wSign) +
		 (weightDecay2 * learningRate * w));

	// Add the new PrevDelta
	addPrevDelta(index, deltaW);
}

void glades::Node::applyDeltas(unsigned int index, int minibatchSize)
{
	if (index >= edges.size())
		return;

	if (minibatchSize <= 0)
		return;

	// We accumulate update steps for this edge during the minibatch window.
	// Historically, missing deltas were treated as 0 (via getPrevDelta() bounds checks),
	// so we intentionally divide by the caller-supplied minibatchSize rather than by
	// the number of accumulated steps.
	const float deltaW = edges[index]->getDeltaAccum() / static_cast<float>(minibatchSize);

	// Set the new weight
	setEdgeWeight(index, getEdgeWeight(index) - deltaW);
}

void glades::Node::setContextNode(const shmea::GPointer<Node>& newContextNode)
{
	contextNode = newContextNode;
}

Node* glades::Node::getContextNode()
{
	return contextNode;
}
