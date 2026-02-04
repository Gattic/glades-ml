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
#include "layer.h"
#include "../GMath/gmath.h"
#include "node.h"
#include "../rng.h"
#include <algorithm>

using namespace glades;

glades::Layer::Layer(int64_t newID, int newType, float newBias)
{
	type = newType;
	// Legacy behavior: if a non-zero layer bias is passed, it will be applied later
	// once nodes exist (via setBiasWeight). Constructors do not create nodes.
	(void)newID;
	(void)newBias;
}

glades::Layer::Layer(int newType)
{
	type = newType;
}

glades::Layer::~Layer()
{
	type = 0;
	children.clear();
	dropoutFlag.clear();
}

float glades::Layer::getBiasWeight() const
{
	// Legacy API: historically a single scalar bias was stored per-layer.
	// The engine now represents per-neuron biases as the final edge on each node.
	// For backwards compatibility and for file headers/debug, we return the average
	// per-neuron bias if available.
	//
	// For gated recurrent layers (GRU/LSTM), each neuron has multiple bias edges
	// (one per gate). In that case we return the average bias across all gates.
	if (type == INPUT_TYPE)
		return 0.0f;

	double sum = 0.0;
	unsigned int n = 0;
	for (unsigned int j = 0; j < children.size(); ++j)
	{
		const Node* node = children[j].get();
		if (!node)
			continue;
		const unsigned int eTotal = node->numEdges();
		if (eTotal == 0)
			continue;

		// Infer gateCount from context-node layout if present:
		// ctxEdges == gateCount * hiddenSize (hiddenSize == children.size()).
		unsigned int gateCount = 1;
		{
			Node* ctx = const_cast<Node*>(node)->getContextNode();
			const unsigned int hiddenSize = static_cast<unsigned int>(children.size());
			if (ctx && hiddenSize > 0)
			{
				const unsigned int ctxEdges = ctx->numEdges();
				if (ctxEdges >= hiddenSize && (ctxEdges % hiddenSize) == 0)
					gateCount = ctxEdges / hiddenSize;
			}
		}
		if (gateCount == 0)
			gateCount = 1;

		const unsigned int block = (gateCount > 0 && (eTotal % gateCount) == 0) ? (eTotal / gateCount) : eTotal;
		for (unsigned int g = 0; g < gateCount; ++g)
		{
			if (block == 0)
				continue;
			const unsigned int biasIdx = ((g + 1) * block) - 1;
			if (biasIdx >= eTotal)
				continue;
			sum += static_cast<double>(node->getEdgeWeight(biasIdx));
			++n;
		}
	}

	if (n == 0)
		return 0.0f;

	return static_cast<float>(sum / static_cast<double>(n));
}

int glades::Layer::getType() const
{
	return type;
}

unsigned int glades::Layer::size() const
{
	return children.size();
}

void glades::Layer::setBiasWeight(float newBiasWeight)
{
	// "Sane" bias semantics: each neuron has its own bias.
	// We represent per-neuron biases as an extra edge weight on each node:
	//   edges[prevLayerSize] * 1.0
	// This keeps training/update logic local to Node/Edge (and is backward-compatible
	// with old models that stored a single scalar bias in the layer header).
	//
	// For gated recurrent layers (GRU/LSTM), each neuron has multiple bias edges
	// (one per gate). We update all gate bias edges.
	if (type == INPUT_TYPE)
		return;

	for (unsigned int j = 0; j < children.size(); ++j)
	{
		Node* node = children[j].get();
		if (!node)
			continue;
		const unsigned int eTotal = node->numEdges();
		if (eTotal == 0)
			continue;

		unsigned int gateCount = 1;
		{
			Node* ctx = node->getContextNode();
			const unsigned int hiddenSize = static_cast<unsigned int>(children.size());
			if (ctx && hiddenSize > 0)
			{
				const unsigned int ctxEdges = ctx->numEdges();
				if (ctxEdges >= hiddenSize && (ctxEdges % hiddenSize) == 0)
					gateCount = ctxEdges / hiddenSize;
			}
		}
		if (gateCount == 0)
			gateCount = 1;

		const unsigned int block = (gateCount > 0 && (eTotal % gateCount) == 0) ? (eTotal / gateCount) : eTotal;
		for (unsigned int g = 0; g < gateCount; ++g)
		{
			if (block == 0)
				continue;
			const unsigned int biasIdx = ((g + 1) * block) - 1;
			if (biasIdx >= eTotal)
				continue;
			node->setEdgeWeight(biasIdx, newBiasWeight);
		}
	}
}

void glades::Layer::setType(int newType)
{
	type = newType;
}

const std::vector<shmea::GPointer<glades::Node> >& glades::Layer::getChildren() const
{
	return children;
}

glades::Node* glades::Layer::getNode(unsigned int index)
{
	if (index >= children.size())
		return NULL;

	return children[index].get();
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
			const int dart = glades::rng::uniform_int(1, 100); // 1..100
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

void glades::Layer::addNode(const shmea::GPointer<Node>& child)
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
			shmea::GPointer<Node> newNode(new Node());
			newNode->initWeights(prevLayerSize, zigg_layers, num_layers, initType, activationType);
			// Add a dedicated per-neuron bias edge (weight initialized to 0, then set to layer bias).
			newNode->initWeights(prevLayerSize + 1, Node::INIT_EMPTY);
			newNode->setEdgeWeight(prevLayerSize, 0.0f);
			addNode(newNode);
		}
	}
	else
	{
		while (size() < cLayerSize)
		{
			shmea::GPointer<Node> newNode(new Node());
			newNode->initWeights(prevLayerSize, initType);
			// Add a dedicated per-neuron bias edge (weight initialized to 0, then set to layer bias).
			newNode->initWeights(prevLayerSize + 1, Node::INIT_EMPTY);
			newNode->setEdgeWeight(prevLayerSize, 0.0f);
			addNode(newNode);
		}
	}
}

void glades::Layer::initGatedWeights(int prevLayerSize, unsigned int cLayerSize, int initType, int activationType, unsigned int gateCount)
{
	if (gateCount == 0)
		return;

	// We reuse the same Xavier/Ziggurat machinery as initWeights(), but we allocate
	// per-gate weight chunks so network.cpp can index gates cheaply.
	const bool useXavier = (initType == Node::INIT_XAVIER || initType == Node::INIT_POSXAVIER);

	unsigned int num_layers = 0;
	std::vector<float> zigg; // length num_layers+1
	if (useXavier)
	{
		num_layers = 2048;
		zigg.assign(num_layers + 1, 0.0f);
		float normal_CDF = 1.0f;
		for (unsigned int i = 1; i < num_layers; i++)
		{
			normal_CDF = normal_CDF - (1 / ((float)num_layers * 2));
			zigg[i] = GMath::norm_inv_CDF(normal_CDF);
		}
		zigg[0] = zigg[1];
		zigg[num_layers] = 0.0f;
	}

	while (size() < cLayerSize)
	{
		shmea::GPointer<Node> node(new Node());
		if (!node)
			break;

		// Per gate: [prevLayerSize weights] + [bias]
		for (unsigned int g = 0; g < gateCount; ++g)
		{
			const unsigned int start = node->numEdges();
			const unsigned int wantW = start + static_cast<unsigned int>(prevLayerSize);
			const unsigned int wantWB = wantW + 1;

			// Append W weights for this gate
			if (useXavier)
				node->initWeights(wantW, (zigg.empty() ? NULL : &zigg[0]), num_layers, initType, activationType);
			else
				node->initWeights(wantW, initType);

			// Append bias edge for this gate
			node->initWeights(wantWB, Node::INIT_EMPTY);
			node->setEdgeWeight(wantW, 0.0f);
		}

		addNode(node);
	}
}

std::vector<shmea::GPointer<Node> >::iterator glades::Layer::removeNode(Node* child)
{
	if (child)
	{
		std::vector<shmea::GPointer<Node> >::iterator itr = children.begin();
		for (; itr != children.end(); ++itr)
		{
			if ((*itr).get() == child)
				return children.erase(itr);
		}
	}
	return children.end();
}

void glades::Layer::clean()
{
	std::vector<shmea::GPointer<Node> >::iterator itr = children.begin();
	for (; itr != children.end(); ++itr)
	{
		if (((*itr)->getWeight() < 0.1f) && ((*itr)->getWeight() > -0.1f))
		{
			itr = removeNode((*itr).get());
			--itr;
		}
	}
}

void glades::Layer::print() const
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

	// Print legacy layer bias for debug (per-neuron bias edge lives on nodes).
	if (type == HIDDEN_TYPE || type == OUTPUT_TYPE)
		printf(" + [B(layer)=%f]", getBiasWeight());
	printf("\n");
}

Node* glades::Layer::operator[](unsigned int index)
{
	if (index >= size())
		return NULL;

	return children[index].get();
}

void Layer::setupContext(unsigned int gateCount)
{
	// RNN context nodes:
	// Each hidden node owns a "context node" that stores the recurrent weights (Wh row).
	//
	// Historically this code initialized context nodes with a single edge weight, which
	// implements only a *diagonal* recurrence (h_i depends only on h_i(t-1)).
	//
	// For a proper Elman-style RNN, each hidden unit i should have a weight for every
	// previous hidden unit j, i.e. Wh is (hiddenSize x hiddenSize).
	const unsigned int hiddenSize = static_cast<unsigned int>(children.size());
	if (hiddenSize == 0)
		return;

	if (gateCount == 0)
		gateCount = 1;

	const unsigned int totalEdges = hiddenSize * gateCount;
	for (unsigned int i = 0; i < children.size(); ++i)
	{
		shmea::GPointer<Node> ctx(new Node());
		if (!ctx)
			continue;

		// Initialize recurrent weights to small random values in [-0.1, 0.1].
		// We intentionally do not use INIT_POSRAND (positive-only) because recurrent
		// matrices need sign flexibility for stable dynamics.
		ctx->initWeights(totalEdges, Node::INIT_EMPTY);
		for (unsigned int j = 0; j < totalEdges; ++j)
		{
			const float u = static_cast<float>(glades::rng::uniform_int(-1000, 1000)) / 1000.0f; // [-1, 1]
			ctx->setEdgeWeight(j, 0.1f * u);
		}

		children[i]->setContextNode(ctx);
	}
}
