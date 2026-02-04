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
#include "LayerBuilder.h"
#include "../../../main.h"
#include "../GMath/gmath.h"
#include "../Structure/nninfo.h"
#include "../Networks/network.h"
#include "../DataObjects/DataInput.h"
#include "Backend/Database/GList.h"
#include "Backend/Database/GTable.h"
#include "Backend/Database/GType.h"
#include "Backend/Database/GLogger.h"
#include "Backend/Database/SaveFolder.h"
#include "Backend/Database/SaveTable.h"
#include "Backend/Database/maxid.h"
#include "NetworkState.h"
#include "edge.h"
#include "layer.h"
#include "node.h"
// #include <ctime>    // For clock()
// #include <cstdio>   // For printf
// #include <vector>   // For vector

using namespace glades;

namespace {
static inline shmea::GLogger& ml_logger()
{
	// Default logger for ML subsystem code that is not attached to a server instance.
	// This keeps LayerBuilder free of hard dependencies on networking/server lifetime.
	static shmea::GLogger logger(shmea::GLogger::LOG_INFO);
	return logger;
}
} // namespace

glades::LayerBuilder::LayerBuilder()
{
	netType = NNetwork::TYPE_DFF;
	inputLayer.reset();
	inputRowCount = 0;
	inputFeatureCount = 0;
	dataInput = NULL;
}

glades::LayerBuilder::LayerBuilder(int newNetType)
{
	netType = newNetType;
	inputLayer.reset();
	inputRowCount = 0;
	inputFeatureCount = 0;
	dataInput = NULL;
}

glades::LayerBuilder::~LayerBuilder()
{
    //
}

bool glades::LayerBuilder::build(const NNInfo* skeleton, const DataInput* newInput, int newNetType, bool standardizeWeightsFlag)
{
	netType = newNetType;
	return build(skeleton, newInput, standardizeWeightsFlag);
}

bool glades::LayerBuilder::build(const NNInfo* skeleton, const DataInput* newInput, bool standardizeWeightsFlag)
{
	lastError.clear();
	if (!skeleton)
	{
		setError("LayerBuilder::build: skeleton is NULL");
		ml_logger().error("LayerBuilder", lastError.c_str());
		return false;
	}

	// Construct the input layers
	ml_logger().info("LayerBuilder", "Building input layer");
	buildInputLayers(skeleton, newInput);
	if (!inputLayer || inputRowCount == 0 || inputFeatureCount == 0)
	{
		setError("LayerBuilder::build: invalid input data (empty train size or feature count is zero)");
		ml_logger().error("LayerBuilder", lastError.c_str());
		return false;
	}

	// Build the hidden layer
	buildHiddenLayers(skeleton);

	if (netType == NNetwork::TYPE_RNN)
	{
		// Build the time state vec
		for (int i = 0; i < skeleton->numHiddenLayers(); ++i)
		{
			int hSize = skeleton->getHiddenLayerSize(i);
			std::vector<float> newVecEdges(hSize, 1.0f);
			std::vector<std::vector<float> > newVecNodes(skeleton->getHiddenLayerSize(i),
														 newVecEdges);
			timeState.push_back(newVecNodes);
		}
	}

	// Build the output layer
	buildOutputLayer(skeleton);

	if (layers.size() <= 0)
	{
		setError("LayerBuilder::build: no layers were constructed (invalid skeleton?)");
		ml_logger().error("LayerBuilder", lastError.c_str());
		return false;
	}

	// standardize the weights between the neurons
	if (standardizeWeightsFlag)
		standardizeWeights(skeleton);

	return true;
}

void glades::LayerBuilder::rebuildInputLayers(const NNInfo* skeleton, const DataInput* newInput)
{
    buildInputLayers(skeleton, newInput);
}

void glades::LayerBuilder::buildInputLayers(const NNInfo* skeleton, const DataInput* di)
{
	// Keep a non-owning pointer to the current dataset for on-demand row materialization.
	dataInput = di;
	// Historically the engine assumed "training rows" were the only rows.
	// For production evaluation/inference, DataInput may contain only a test split.
	// Keep the training count as the default, but fall back to test size when training is empty.
	inputRowCount = (di ? (di->getTrainSize() > 0u ? di->getTrainSize() : di->getTestSize()) : 0u);
	inputFeatureCount = (di ? di->getFeatureCount() : 0);

	if (inputRowCount == 0 || inputFeatureCount == 0)
	{
		setError("LayerBuilder::buildInputLayers: inputRowCount==0 or inputFeatureCount==0");
		return;
	}

	// Build a single reusable input layer of size = featureCount.
	// We'll overwrite node weights from the requested row in getInputLayer().
	inputLayer = shmea::GPointer<Layer>(new Layer(Layer::INPUT_TYPE));
	if (!inputLayer)
	{
		setError("LayerBuilder::buildInputLayers: failed to allocate input layer");
		return;
	}

	for (unsigned int c = 0; c < inputFeatureCount; ++c)
	{
		shmea::GPointer<Node> node(new Node());
		if (!node)
			continue;
		node->setWeight(0.0f);
		inputLayer->addNode(node);
	}
}

void glades::LayerBuilder::buildHiddenLayers(const NNInfo* skeleton)
{
	const int inputLayerSize = (inputLayer ? static_cast<int>(inputLayer->size()) : 0);
	int outputLayerSize = skeleton->getOutputLayerSize();
	int prevLayerSize = inputLayerSize;
	int outputType = skeleton->getOutputType();
	// Weight init policy:
	// Historically this function would start building with Xavier, then if it encountered any
	// "positive-only" activation (sigmoid/relu/leaky) *or* a classification output, it would
	// throw away everything built so far, switch to POSXAVIER, and rebuild all hidden layers.
	//
	// That control-flow hack (rewinding the loop and clearing `layers`) is fragile and obscures
	// intent. We compute the policy up-front instead.
	bool usePosXavier = (outputType == GMath::CLASSIFICATION);
	for (int i = 0; i < skeleton->numHiddenLayers(); ++i)
	{
		const int act = skeleton->getActivationType(i);
		if ((act == GMath::SIGMOID) || (act == GMath::RELU) || (act == GMath::LEAKY))
		{
			usePosXavier = true;
			break;
		}
	}

	int activationType;
	unsigned int gateCount = 1;
	if (netType == NNetwork::TYPE_GRU)
		gateCount = 3;
	else if (netType == NNetwork::TYPE_LSTM)
		gateCount = 4;

	if (inputLayerSize <= 0)
	{
		setError("LayerBuilder::buildHiddenLayers: input layer size is zero");
		return;
	}

	// Create each hidden layer
	for (int i = 0; i < skeleton->numHiddenLayers(); ++i)
	{
		activationType = skeleton->getActivationType(i);
		// Get the current layer size
		int cLayerSize = skeleton->getHiddenLayerSize(i);
		shmea::GPointer<Layer> cLayer(new Layer(Layer::HIDDEN_TYPE));

		const int init = usePosXavier ? Node::INIT_POSXAVIER : Node::INIT_XAVIER;
		if (gateCount > 1)
			cLayer->initGatedWeights(prevLayerSize, cLayerSize, init, activationType, gateCount);
		else
			cLayer->initWeights(prevLayerSize, cLayerSize, init, activationType);

		cLayer->setupContext(gateCount);
		layers.push_back(cLayer);
		prevLayerSize = cLayerSize;
	}
}

void glades::LayerBuilder::buildOutputLayer(const NNInfo* skeleton)
{
	const int inputLayerSize = (inputLayer ? static_cast<int>(inputLayer->size()) : 0);
	int outputLayerSize = skeleton->getOutputLayerSize();
	int prevLayerSize = inputLayerSize;
	int outputType = skeleton->getOutputType();
	bool isPositive = false;
	// Activation types in this engine are indexed by the *input-side* layer counter
	// (i.e. by transition). The output layer uses index == numHiddenLayers().
	const int outputActivationType = skeleton->getActivationType(skeleton->numHiddenLayers());

	if (inputLayerSize <= 0)
	{
		setError("LayerBuilder::buildOutputLayer: input layer size is zero");
		return;
	}

	// Create each hidden layer
	for (int i = 0; i < skeleton->numHiddenLayers(); ++i)
	{
		const int activationType = skeleton->getActivationType(i);
		// Get the current layer size
		int cLayerSize = skeleton->getHiddenLayerSize(i);

		// Create the hidden layer
		if ((activationType == GMath::SIGMOID) || (activationType == GMath::RELU) ||
			(activationType == GMath::LEAKY) || (outputType == GMath::CLASSIFICATION))
			isPositive = true;

		prevLayerSize = cLayerSize;
	}

	// Output-layer activation also affects weight init heuristics.
	if ((outputActivationType == GMath::SIGMOID) || (outputActivationType == GMath::RELU) ||
		(outputActivationType == GMath::LEAKY) || (outputType == GMath::CLASSIFICATION))
		isPositive = true;

	// Create the output layer
	shmea::GPointer<Layer> cLayer(new Layer(Layer::OUTPUT_TYPE));
	if (isPositive)
		cLayer->initWeights(prevLayerSize, outputLayerSize, Node::INIT_POSXAVIER, outputActivationType);
	else
		cLayer->initWeights(prevLayerSize, outputLayerSize, Node::INIT_XAVIER, outputActivationType);
	layers.push_back(cLayer);
}

glades::Layer* glades::LayerBuilder::getInputLayer(unsigned int inputRowCounter, unsigned int cInputLayerCounter)
{
	// Backward-compatible default: historically this always materialized from the training split.
	return getInputLayer(inputRowCounter, cInputLayerCounter, SPLIT_TRAIN);
}

glades::Layer* glades::LayerBuilder::getInputLayer(unsigned int inputRowCounter,
                                                   unsigned int cInputLayerCounter,
                                                   glades::LayerBuilder::InputSplit split)
{
	if (cInputLayerCounter >= layers.size())
		return NULL;

	// Current Input Layer
	Layer* cInputLayer = NULL;
	if (cInputLayerCounter == 0)
	{
		// Materialize the requested row into the reusable input layer.
		if (!inputLayer || !dataInput)
			return NULL;

		// Select the correct split. This is critical for evaluation/inference runs.
		// NOTE: We do *not* use inputRowCount for bounds checks here; inputRowCount is a build-time
		// convenience and may refer to either split depending on dataset population.
		shmea::GVector<float> row;
		if (split == SPLIT_TEST)
		{
			if (inputRowCounter >= dataInput->getTestSize())
				return NULL;
			row = dataInput->getTestRow(inputRowCounter);
		}
		else
		{
			if (inputRowCounter >= dataInput->getTrainSize())
				return NULL;
			row = dataInput->getTrainRow(inputRowCounter);
		}
		const unsigned int layerSize = static_cast<unsigned int>(inputLayer->size());
		const unsigned int rowSize = static_cast<unsigned int>(row.size());
		const unsigned int n = std::min(rowSize, layerSize);

		// Copy known features.
		for (unsigned int c = 0; c < n; ++c)
		{
			Node* node = inputLayer->getNode(c);
			if (node)
				node->setWeight(row[c]);
		}

		// IMPORTANT invariant:
		// If a row is shorter than the expected feature count, remaining input nodes must be
		// cleared to 0.0f. Otherwise those nodes retain stale weights from the previous row,
		// corrupting both forward-pass outputs and training gradients.
		for (unsigned int c = n; c < layerSize; ++c)
		{
			Node* node = inputLayer->getNode(c);
			if (node)
				node->setWeight(0.0f);
		}

		cInputLayer = inputLayer.get();
	}
	else
		cInputLayer = layers[cInputLayerCounter-1].get();
	if (!cInputLayer)
		return NULL;
	
	return cInputLayer;
}

glades::Layer* glades::LayerBuilder::getOutputLayer(unsigned int cOutputLayerCounter)
{
	if (cOutputLayerCounter > layers.size())
		return NULL;

	// Current Output Layer
	Layer* cOutputLayer = layers[cOutputLayerCounter-1].get();
	if (!cOutputLayer)
		return NULL;

	// Why would this happen??
	if (cOutputLayer->getType() == Layer::INPUT_TYPE)
		return NULL;
	
	return cOutputLayer;
}

glades::Node* glades::LayerBuilder::getInputNode(Layer* cInputLayer, unsigned int cInputNodeCounter)
{
	// Current Input Node Error Check
	if (cInputNodeCounter >= cInputLayer->size())
		return NULL;

	// Current Input Node
	Node* cInputNode = cInputLayer->getNode(cInputNodeCounter);
	if (!cInputNode)
		return NULL;

	return cInputNode;
}

glades::Node* glades::LayerBuilder::getOutputNode(Layer* cOutputLayer, unsigned int cOutputNodeCounter)
{
	// Current Output Node Error Check
	if (cOutputNodeCounter >= cOutputLayer->size())
		return NULL;

	// Current Output Node
	Node* cOutputNode = (*cOutputLayer)[cOutputNodeCounter];
	if (!cOutputNode)
		return NULL;

	return cOutputNode;
}

void glades::LayerBuilder::setTimeState(unsigned int cLayerCounter, unsigned int cNodeCounter,
										unsigned int cEdgeCounter, float newTimeState)
{
	if (cLayerCounter >= timeState.size())
		return;

	if (cNodeCounter >= timeState[cLayerCounter].size())
		return;

	if (cEdgeCounter >= timeState[cLayerCounter][cNodeCounter].size())
		return;

	timeState[cLayerCounter][cNodeCounter][cEdgeCounter] = newTimeState;
}

unsigned int glades::LayerBuilder::getInputLayersSize() const
{
	return inputRowCount;
}

unsigned int glades::LayerBuilder::getLayersSize() const
{
	return layers.size();
}

unsigned int glades::LayerBuilder::getLayerSize(unsigned int index) const
{
    if(index > layers.size()+1)
	return 0;

    if(index == 0)
	return inputLayer ? inputLayer->size() : 0;
	
    return layers[index-1]->size();
}

float glades::LayerBuilder::getTimeState(unsigned int cLayerCounter, unsigned int cNodeCounter,
										 unsigned int cEdgeCounter) const
{
	// Return 1.0f on error to retain old state
	if (cLayerCounter >= timeState.size())
		return 1.0f;

	if (cNodeCounter >= timeState[cLayerCounter].size())
		return 1.0f;

	if (cEdgeCounter >= timeState[cLayerCounter][cNodeCounter].size())
		return 1.0f;

	return timeState[cLayerCounter][cNodeCounter][cEdgeCounter];
}

shmea::GList glades::LayerBuilder::getWeights() const
{
   shmea::GList weights; 
   //We start with 1 because the first layer (input layer) doesn't have the data of the weights
    for(unsigned int i = 0; i < getLayersSize(); ++i)
    {
	// Bias unification:
	// Biases are stored as per-neuron bias *edge weights* (the final edge in each (fanIn+1) block,
	// per gate for GRU/LSTM). For GUI/export we do not include those bias edges here to avoid
	// duplicating bias information: callers that need bias visualization should use addBiasWeights().
	//
	// Determine fan-in for this layer so we can identify bias edges by index.
	const unsigned int prevSize =
	    (i == 0) ? (inputLayer ? static_cast<unsigned int>(inputLayer->size()) : 0u)
	             : static_cast<unsigned int>(layers[i - 1] ? layers[i - 1]->size() : 0u);
	const unsigned int stride = prevSize + 1u;

	const std::vector<shmea::GPointer<Node> >& cChildren = layers[i]->getChildren();
	for(unsigned int j = 0; j < cChildren.size(); ++j)
	{
	   for(unsigned int k = 0; k < cChildren[j]->numEdges(); ++k)
	   {
		// Skip per-neuron (per-gate) bias edges.
		if (stride > 0u && (k % stride) == prevSize)
			continue;
		float cWeight = cChildren[j]->getEdgeWeight(k);
		weights.addFloat(cWeight);
	   }
	   weights.addString(',');
	}
	weights.addString(';');
    }

    return weights;
}

void glades::LayerBuilder::addBiasWeights(shmea::GList& weights) const
{
    weights.addString('B');
    for(unsigned int i = 0; i < getLayersSize(); ++i)
    {
		weights.addFloat(layers[i]->getBiasWeight());
    }
}

void glades::LayerBuilder::resetContextState(float value)
{
	for (unsigned int i = 0; i < layers.size(); ++i)
	{
		Layer* layer = layers[i].get();
		if (!layer)
			continue;
		if (layer->getType() != Layer::HIDDEN_TYPE)
			continue;

		const std::vector<shmea::GPointer<Node> >& nodes = layer->getChildren();
		for (unsigned int j = 0; j < nodes.size(); ++j)
		{
			Node* node = nodes[j].get();
			if (!node)
				continue;
			Node* ctx = node->getContextNode();
			if (!ctx)
				continue;
			ctx->setWeight(value);
		}
	}
}

void glades::LayerBuilder::updateContextFromHiddenActivations()
{
	for (unsigned int i = 0; i < layers.size(); ++i)
	{
		Layer* layer = layers[i].get();
		if (!layer)
			continue;
		if (layer->getType() != Layer::HIDDEN_TYPE)
			continue;

		const std::vector<shmea::GPointer<Node> >& nodes = layer->getChildren();
		for (unsigned int j = 0; j < nodes.size(); ++j)
		{
			Node* node = nodes[j].get();
			if (!node)
				continue;
			Node* ctx = node->getContextNode();
			if (!ctx)
				continue;

			// Store the hidden node's *output activation* as next timestep's context.
			ctx->setWeight(node->getWeight());
		}
	}
}

void glades::LayerBuilder::standardizeWeights(const NNInfo* skeleton)
{
	// Structure required!
	if (!skeleton)
		return;

	// Standardize the initialization of the weights
	if (getLayersSize() <= 0)
		return;

	// Set the min and max of the weights
	xMin = 0.0f;
	xMax = 0.0f;

	// Check net vars
	int outputType = skeleton->getOutputType();
	bool isPositive = false;

	// iterate through the layers
	for (unsigned int i = 0; i < getLayersSize(); ++i)
	{
		// Check layer vars
		int activationType = skeleton->getActivationType(i);

		// Check if positive
		if ((activationType == GMath::SIGMOID) || (activationType == GMath::RELU) ||
			(activationType == GMath::LEAKY) || (outputType == GMath::CLASSIFICATION))
			isPositive = true;

		// Determine fan-in for this layer's nodes so we can skip per-neuron bias edges.
		const unsigned int prevSize =
		    (i == 0) ? (inputLayer ? static_cast<unsigned int>(inputLayer->size()) : 0u)
		             : static_cast<unsigned int>(layers[i - 1] ? layers[i - 1]->size() : 0);
		const unsigned int stride = prevSize + 1u;

		// iterate through the nodes
		const std::vector<shmea::GPointer<Node> >& cChildren = layers[i]->getChildren();
		for (unsigned int j = 0; j < cChildren.size(); ++j)
		{
			// iterate through the node weights
			for (unsigned int k = 0; k < cChildren[j]->numEdges(); ++k)
			{
				// Bias edges are at index == prevSize for dense layers, and at each
				// gate block's final index for gated recurrent layers.
				// Canonical rule: (k % (prevSize+1)) == prevSize.
				if (stride > 0u && (k % stride) == prevSize)
					continue;
				float cWeight = cChildren[j]->getEdgeWeight(k);
				if ((i == 0) && (j == 0) && (k == 0))
				{
					xMin = cWeight;
					xMax = cWeight;
				}

				// Check the mins and maxes
				if (cWeight < xMin)
					xMin = cWeight;
				if (cWeight > xMax)
					xMax = cWeight;
			}
		}
	}

	// standardize the weights
	xRange = xMax - xMin;
	if (xRange <= 0.0f)
		return;

	// iterate through the layers
	for (unsigned int i = 0; i < getLayersSize(); ++i)
	{
		const unsigned int prevSize =
		    (i == 0) ? (inputLayer ? static_cast<unsigned int>(inputLayer->size()) : 0u)
		             : static_cast<unsigned int>(layers[i - 1] ? layers[i - 1]->size() : 0);
		const unsigned int stride = prevSize + 1u;

		// iterate through the nodes
		const std::vector<shmea::GPointer<Node> >& cChildren = layers[i]->getChildren();
		for (unsigned int j = 0; j < cChildren.size(); ++j)
		{
			// iterate through the node weights
			for (unsigned int k = 0; k < cChildren[j]->numEdges(); ++k)
			{
				// Do not standardize per-neuron (per-gate) bias edges.
				if (stride > 0u && (k % stride) == prevSize)
					continue;
				float cWeight = cChildren[j]->getEdgeWeight(k);

				// Adjust the children
				if (isPositive)
					cChildren[j]->setEdgeWeight(k, ((cWeight - xMin) / (xRange)));
				else
					cChildren[j]->setEdgeWeight(k, ((cWeight - xMin) / (xRange)) - 0.5f);
			}
		}
	}
}

float glades::LayerBuilder::unstandardize(float value)
{
	return ((value + 0.5f) * xRange) + xMin;
}

void glades::LayerBuilder::scrambleDropout(unsigned int inputRowCounter, float pInput,
										   const std::vector<float>& pHidden)
{
	// Invalid arg
	if (layers.size() - 1 != pHidden.size())
		return;

	if (inputRowCounter >= inputRowCount)
		return;

	Layer* cInputLayer = inputLayer.get();
	if (!cInputLayer)
		return;

	// Input layer
	cInputLayer->generateDropout(pInput);

	// Hidden layers dropout
	for (unsigned int i = 0; i < layers.size(); ++i)
	{
		if (layers[i]->getType() == Layer::OUTPUT_TYPE)
			continue;

		// Hidden layer 'i'
		layers[i]->generateDropout(pHidden[i]);
	}
}

void glades::LayerBuilder::clearDropout()
{
	for (unsigned int i = 0; i < layers.size(); ++i)
		layers[i]->clearDropout();
}

void glades::LayerBuilder::print(const NNInfo* skeleton, bool override) const
{
	if (!inputLayer || inputRowCount == 0)
		return;

	if (layers.size() == 0)
		return;

	// print input layer info
	printf("[GQL] Input(r,c): (%u,%d)\n", inputRowCount, (int)inputLayer->size());
	if (override)
	{
		// We no longer store per-row input layers. Print the current reusable input layer.
		printf("Input [type=%d]: ", inputLayer->getType());
		inputLayer->print();
		printf("\n");
	}

	// print layer info
	if (override)
		skeleton->print();
	else
	{
		printf("[GQL] Hidden Layers (%ld)\n", layers.size() - 1); // minus the output layer
		printf("[GQL] Output Layer Size (%d)\n", layers[layers.size() - 1]->size());
	}

	// Network
	if (override)
	{
		printf("[GQL] Network\n");
		for (unsigned int i = 0; i < layers.size(); ++i)
		{
			printf("%d [%d]: ", i + 1, layers[i]->getType());
			layers[i]->print();
		}
		printf("\n");
	}
}

void glades::LayerBuilder::clean()
{
	inputLayer.reset();
	inputRowCount = 0;
	inputFeatureCount = 0;
	dataInput = NULL;
	layers.clear();
	timeState.clear();
	xMin = 0.0f;
	xMax = 0.0f;
	xRange = 0.0f;
}

// Database
/*!
 * @brief load network
 * @details load a NNetwork object from a location where the network is stored
 * @param networkData the table of neural network architecture information
 * @return whether or not the load was successful
 */
/*bool glades::LayerBuilder::load(const shmea::GTable& networkData)
{
	// We dont need to load the old instance (for now?)
	if (!networkData)
		return true;

	// loading just the network info is okay
	for (int networkRow = 0; networkRow < networkData->numberOfRows(); ++networkRow)
	{
		int64_t layerId = networkData.getCell(networkRow, 0)->getLong();
		int layerType = networkData.getCell(networkRow, 1)->getInt();
		float layerBiasWeight = networkData.getCell(networkRow, 2)->getFloat();

		Layer* layer = new Layer(layerId, layerType, layerBiasWeight);

		SaveTable* layerFile = new SaveTable("layers");
		layerFile->load_id(layerId);
		shmea::GTable* layerData = layerFile->getTable();
		if (!layerData)
			return false;

		for (int layerRow = 0; layerRow < layerData->numberOfRows(); ++layerRow)
		{
			int64_t nodeId = layerData.getCell(layerRow, 0)->getLong();

			Node* node = new Node();
			node->setID(nodeId);

			SaveTable* nodeFile = new SaveTable("nodes");
			nodeFile->load_id(nodeId);
			shmea::GTable* nodeData = nodeFile->getTable();
			if (!nodeData)
				return false;

			std::vector<Edge*> edges;
			for (int nodeRow = 0; nodeRow < nodeData->numberOfRows(); ++nodeRow)
			{
				float edgeWeight = nodeData.getCell(nodeRow, 0)->getFloat();
				float edgePrevDelta = nodeData.getCell(nodeRow, 1)->getFloat();

				Edge* edge = new Edge(nodeRow, edgeWeight);
				edge->setPrevDelta(edgePrevDelta);
				edges.push_back(edge);
				// SaveTable* edgeFile = new SaveTable("edges");
				// edgeFile->load_id(edgeId);
				// shmea::GTable* edgeData = edgeFile->getTable();
				// if (!edgeData)
				// 	return NULL;

				// for (int edgeRow = 0; edgeRow < edgeData->numberOfRows(); ++edgeRow)
				// {
				// 	float edgeWeight = edgeData.getCell(edgeRow, 0)->getFloat();
				// 	float edgePrevDelta = nodeData.getCell(edgeRow, 1)->getFloat();
				// 	Edge* edge = new Edge(edgeId, edgeWeight);
				// 	edge->setPrevDelta(edgePrevDelta);
				// 	edges.push_back(edge);
				// }
			}

			node->setEdges(edges);
			layer->addNode(node);
		}

		layers.push_back(layer);
	}

	return false;
}*/

bool glades::LayerBuilder::load(const std::string& netName)
{
	return false;
}

/*!
 * @brief save NNetwork
 * @details save all the information in the NNetwork to GTables
 * @return whether or not the save went through
 */
bool glades::LayerBuilder::save(const std::string& netName) const
{
	shmea::SaveFolder nnList(netName.c_str());

	shmea::GVector<shmea::GString> layerHeaders, edgeHeaders;
	layerHeaders.push_back("BiasWeight");
	edgeHeaders.push_back("layerID");
	edgeHeaders.push_back("nodeID");

	// Save all the layers in a
	shmea::GTable layerTable(',', layerHeaders);
	shmea::GTable edgeTable(',', edgeHeaders);
	for (unsigned int layerIdx = 0; layerIdx < getLayersSize(); ++layerIdx)
	{
		Layer* layer = layers[layerIdx].get();
		if (!layer)
			continue;

		// Add to the layer table
		shmea::GList layerRow;
		layerRow.addFloat(layer->getBiasWeight());
		layerTable.addRow(layerRow);

		// Save each edge in the edgeTable
		const std::vector<shmea::GPointer<Node> >& nodes = layer->getChildren();
		for (unsigned int nodeIdx = 0; nodeIdx < nodes.size(); ++nodeIdx)
		{
			// Add each edge to the edge file
			shmea::GList edgeRow;
			edgeRow.addInt(layerIdx);
			edgeRow.addInt(nodeIdx);
			edgeTable.addRow(edgeRow);
		}
	}

	// Save the layer information and edges
	nnList.newItem("layers", layerTable);
	nnList.newItem("edges", edgeTable);

	return true;
}

/*!
 * @brief saves bias and weights of a single layer to a file 
 * @details writes the layer’s bias, size, number of edges
 *          and edge weights directly to the file
 * @param layer the layer to be saved to the file
 * @param out the output file stream to write to
 * @return whether or not the layer save went through
*/
bool glades::LayerBuilder::saveLayer(glades::Layer* layer, std::ofstream& out) const
{
    if (!layer)
        return false;

    const std::vector<shmea::GPointer<Node> >& nodes = layer->getChildren();

    out << layer->getBiasWeight() << " ";
    out << nodes.size() << " ";
    if (nodes.size() > 0 && nodes[0]->numEdges() > 0)
        out << nodes[0]->numEdges() << "\n";
    else
        out << "0" << "\n";
//    out << layer->getType() << " ";
//    out << nodes.size() << "\n";

    for (unsigned int j = 0; j < nodes.size(); ++j) {
        Node* node = nodes[j].get();
        if (!node)
            continue;

//        out << node->getWeight() << " ";
//        out << node->numEdges() << "\n";

        for (unsigned int k = 0; k < node->numEdges(); ++k) {
            out << node->getEdgeWeight(k) << " ";
 /*           std::vector<float> prevDeltas = edge->getPrevDeltas();

            out << edge->getActivation() << " ";
            if (edge->getActivated())
                out << 1 << " ";
            else
                out << 0 << " ";
            out << prevDeltas.size() << " ";

            for (unsigned int l = 0; l < prevDeltas.size(); ++l) {
                out << prevDeltas[l] << " ";
            }
*/
        }
    }
    out << "\n";
    return true;
}

/*!
 * @brief saves biases and weights to a file 
 * @details writes directly to the file to avoid copying all 
 *          values into a new GTable object, which would be inefficient 
 *          for large networks with billions of parameters
 * @param fileName the name of the file to save to
 * @return whether or not the save went through
*/
bool glades::LayerBuilder::saveState(const char* fileName) const
{
	if (!fileName)
		return false;

	shmea::SaveFolder folderToSave("nn-state");
	if (!folderToSave.checkFolder())
		return false;

	return saveStateToFile(std::string((folderToSave.getPath() + fileName).c_str()));
}

/*!
 * @brief loads bias and weights of a single layer from a file 
 * @details reads the layer’s bias and edge weights directly from the file;
 *          verifies that the layer size and the number of edges in the input 
 *          file match to the expected skeleton values
 * @param layer the layer to be loaded from the file
 * @param nodesCount expected layer size
 * @param edgeCount expected number of edges
 * @param in the input file stream to read from
 * @return whether or not the layer load went through
*/
bool glades::LayerBuilder::loadLayer(Layer* layer, unsigned int nodesCount, unsigned int edgeCount, std::ifstream& in)
{
    if (!layer) {
        printf("Something wrong with the layer");
        return false;
    }

    float legacyLayerBias = 0.0f;
    if (!(in >> legacyLayerBias)) {
        printf("Error during reading the layer bias from file");
        return false;
    }

    unsigned int fileLayerSize;
    if (!(in >> fileLayerSize)) {
        printf("Error during reading the layer size from file");
        return false;
    }

    unsigned int fileEdgeCount;
    if (!(in >> fileEdgeCount)) {
        printf("Error during reading the edge count from file");
        return false;
    }

    if (fileLayerSize != nodesCount) {
        printf("Inconvenience between the layer size in file and the skeleton or input data layer size");
        return false;
    }

    // Backward compatibility:
    // - Old saved models stored *no per-neuron bias edge* (edgeCount == prevLayerSize).
    // - New models store an additional final bias edge per node (edgeCount == prevLayerSize + 1).
    //
    // We accept either:
    //   fileEdgeCount == edgeCount          (new format)
    //   fileEdgeCount == edgeCount - 1      (old format; bias edge will be initialized from layer bias)
    if (!(fileEdgeCount == edgeCount || (edgeCount > 0 && fileEdgeCount == edgeCount - 1))) {
        printf("Inconvenience between the edge count in file and the expected edge count");
        return false;
    }

    unsigned int layerSize = layer->size();

    const std::vector<shmea::GPointer<Node> >& nodes = layer->getChildren();
    for (unsigned int i = 0; i < nodesCount; ++i) {
        bool it_is_a_new_node = false;
        Node* node = NULL;
        shmea::GPointer<Node> ownedNode;
        if (i > layerSize - 1) {
            ownedNode = shmea::GPointer<Node>(new Node());
            node = ownedNode.get();
            it_is_a_new_node = true;
        }
        else
            node = nodes[i].get();

        if (!node)
            return false;

/*        float nodeWeight;
        in >> nodeWeight;
        node->setWeight(nodeWeight);

        unsigned int edgeCount;
        in >> edgeCount;
*/
        bool number_of_edges_is_less_then_skeleton_edgeCount = node->numEdges() < edgeCount;

        std::vector<shmea::GPointer<glades::Edge> > edges;

        // Read as many edges as exist in the file, then fill any missing final bias edge.
        for (unsigned int k = 0; k < fileEdgeCount; ++k) {
            float edgeWeight;
            if (!(in >> edgeWeight)) {
                printf("Error during reading the edge weight from file");
                return false;
            }

            if (number_of_edges_is_less_then_skeleton_edgeCount) {
                edges.push_back(shmea::GPointer<glades::Edge>(new Edge(-1, edgeWeight)));
            } else {
                node->setEdgeWeight(k, edgeWeight);
            }

/*
            Edge* edge = new Edge(-1, edgeWeight);

            float edgeActivation;
            in >> edgeActivation;
            edge->setActivation(edgeActivation);

            int edgeActivated;
            in >> edgeActivated;
            edge->setActivated(bool(edgeActivation));

            unsigned int prevDeltasCount;
            in >> prevDeltasCount;
            for (unsigned int l = 0; l < prevDeltasCount; ++l) {
                float prevDelta;
                in >> prevDelta;
                edge->addPrevDelta(prevDelta);
            }
            edges.push_back(edge);
*/
        }
        // If we're loading an old-format layer (missing the final bias edge), initialize it
        // from the layer's legacy bias scalar.
        if (fileEdgeCount + 1 == edgeCount)
        {
            const float b = legacyLayerBias;
            if (number_of_edges_is_less_then_skeleton_edgeCount)
                edges.push_back(shmea::GPointer<glades::Edge>(new Edge(-1, b)));
            else
                node->setEdgeWeight(edgeCount - 1, b);
        }
        if (number_of_edges_is_less_then_skeleton_edgeCount) {
            node->setEdges(edges);
        }
        if (it_is_a_new_node) 
            layer->addNode(ownedNode);
    }
    return true;
}

/*!
 * @brief loads biases and weights from a file 
 * @details reads directly from the file to avoid first loading values into a GTable object
 *          and then copying them into the LayerBuilder, which would be inefficient
 *          for large networks with billions of parameters
 * @param skeleton the expected network structure; the file’s contents must match
 *        this skeleton (number of layers, layer sizes, number of edges)
 * @param fileName the name of the file to load from
 * @return whether or not the load went through
*/
bool glades::LayerBuilder::loadState(const NNInfo* skeleton, const char* fileName)
{
	if (!fileName)
		return false;

	return loadStateFromFile(skeleton, std::string("database/nn-state/") + fileName);
}

bool glades::LayerBuilder::saveStateToFile(const std::string& filePath) const
{
	if (filePath.empty())
		return false;

	std::ofstream out(filePath.c_str());
	if (!out)
		return false;

	// === LayerBuilder state file format v2 ===
	//
	// This fixes historical persistence bugs for recurrent/gated nets:
	// - v1 saved only per-node (feedforward) edges and did NOT persist context-node edges
	//   (RNN Whh, GRU/LSTM U matrices). Loading such a file for recurrent nets produces a
	//   silently-wrong model.
	//
	// v2 persists:
	// - inputSize (so load does not depend on a pre-built input layer)
	// - netType (so we can compute gateCount and expected edge counts)
	// - per-layer: node-edge weights AND context-node edge weights where applicable
	//
	// Backwards compatibility:
	// - loadStateFromFile() still accepts v1 files (no magic header).
	static const char* kMagic = "GLADES_LAYER_STATE";
	static const int kVersion = 2;

	const unsigned int inputSize = (inputLayer ? static_cast<unsigned int>(inputLayer->size()) : 0u);
	const unsigned int layerCount = getLayersSize();

	out << kMagic << "\n";
	out << "version " << kVersion << "\n";
	out << "netType " << netType << "\n";
	out << "inputSize " << inputSize << "\n";
	out << "layers " << layerCount << "\n";

	// gateCount for GRU/LSTM context layouts
	unsigned int gateCount = 1u;
	if (netType == NNetwork::TYPE_GRU)
		gateCount = 3u;
	else if (netType == NNetwork::TYPE_LSTM)
		gateCount = 4u;

	for (unsigned int i = 0; i < layerCount; ++i)
	{
		Layer* layer = layers[i].get();
		if (!layer)
			return false;

		const unsigned int nodeCount = static_cast<unsigned int>(layer->size());
		// prevSize is needed to define the expected node-edge count.
		const unsigned int prevSize =
		    (i == 0) ? inputSize : static_cast<unsigned int>(layers[i - 1] ? layers[i - 1]->size() : 0u);
		const unsigned int curSize = nodeCount;
		const bool isOutput = (layer->getType() == Layer::OUTPUT_TYPE);

		unsigned int nodeEdges = 0u;
		unsigned int ctxEdges = 0u;

		if (isOutput)
		{
			// Output layer is always dense: [prevSize weights] + [1 bias]
			nodeEdges = prevSize + 1u;
			ctxEdges = 0u;
		}
		else
		{
			// Hidden layers
			if (netType == NNetwork::TYPE_GRU || netType == NNetwork::TYPE_LSTM)
			{
				nodeEdges = gateCount * (prevSize + 1u);
				ctxEdges = gateCount * curSize;
			}
			else if (netType == NNetwork::TYPE_RNN)
			{
				nodeEdges = prevSize + 1u;
				ctxEdges = curSize; // Wh row per hidden unit
			}
			else
			{
				// DFF
				nodeEdges = prevSize + 1u;
				ctxEdges = 0u;
			}
		}

		// Write layer header. biasAvg is included for legacy/debug only; biases live in edge weights.
		out << "layer " << i
		    << " type " << layer->getType()
		    << " biasAvg " << layer->getBiasWeight()
		    << " nodes " << nodeCount
		    << " nodeEdges " << nodeEdges
		    << " ctxEdges " << ctxEdges
		    << "\n";

		for (unsigned int j = 0; j < nodeCount; ++j)
		{
			Node* node = layer->getNode(j);
			if (!node)
				return false;

			out << "n " << j;
			for (unsigned int k = 0; k < nodeEdges; ++k)
				out << " " << node->getEdgeWeight(k);

			if (ctxEdges > 0u)
			{
				Node* ctx = node->getContextNode();
				for (unsigned int k = 0; k < ctxEdges; ++k)
					out << " " << (ctx ? ctx->getEdgeWeight(k) : 0.0f);
			}
			out << "\n";
		}
	}

	return static_cast<bool>(out);
}

bool glades::LayerBuilder::loadStateFromFile(const NNInfo* skeleton, const std::string& filePath)
{
	if (!skeleton)
		return false;
	if (filePath.empty())
		return false;

	std::ifstream in(filePath.c_str());
	if (!in)
		return false;

	// Detect v2 by magic header on the first line.
	std::string firstLine;
	if (!std::getline(in, firstLine))
		return false;
	// Tolerate Windows line endings when state files are moved across platforms.
	if (!firstLine.empty() && firstLine[firstLine.size() - 1] == '\r')
		firstLine.erase(firstLine.size() - 1);

	static const char* kMagic = "GLADES_LAYER_STATE";
	const bool isV2 = (firstLine == kMagic);

	if (!isV2)
	{
		// Legacy v1: rewind and use the historical loader (node-edge-only).
		in.clear();
		in.seekg(0, std::ios::beg);

		const unsigned int layerCount = static_cast<unsigned int>(skeleton->numHiddenLayers()) + 1;
		if (layers.size() < layerCount)
			layers.resize(layerCount);

		for (unsigned int i = 0; i < layerCount; ++i)
		{
			if (!layers[i])
			{
				if (i == layerCount - 1)
					layers[i] = shmea::GPointer<Layer>(new Layer(Layer::OUTPUT_TYPE));
				else
					layers[i] = shmea::GPointer<Layer>(new Layer(Layer::HIDDEN_TYPE));
			}

			unsigned int curLayerSize = 0;
			if (i == layerCount - 1)
				curLayerSize = skeleton->getOutputLayerSize();
			else
				curLayerSize = static_cast<unsigned int>(skeleton->getHiddenLayerSize(i));

			unsigned int prLayerSize = 0;
			if (i > 0)
				prLayerSize = static_cast<unsigned int>(skeleton->getHiddenLayerSize(i - 1));
			else
				prLayerSize = getLayerSize(0);

			// Legacy v1 stores only per-node edges. For GRU/LSTM hidden layers those edges
			// include all gates, so we must compute the correct per-node edge count.
			//
			// IMPORTANT: v1 does NOT persist context-node edges (RNN Whh / GRU/LSTM U), so
			// recurrent models loaded from v1 are incomplete by construction.
			unsigned int nodeEdgeCount = prLayerSize + 1u;
			if (i != layerCount - 1)
			{
				if (netType == NNetwork::TYPE_GRU)
					nodeEdgeCount = 3u * (prLayerSize + 1u);
				else if (netType == NNetwork::TYPE_LSTM)
					nodeEdgeCount = 4u * (prLayerSize + 1u);
			}

			if (!loadLayer(layers[i].get(), curLayerSize, nodeEdgeCount, in))
				return false;
		}
		return true;
	}

	// === v2 loader ===
	// Expected header:
	//   version <int>
	//   netType <int>
	//   inputSize <uint>
	//   layers <uint>
	std::string tag;
	int version = 0;
	int fileNetType = NNetwork::TYPE_DFF;
	unsigned int inputSize = 0u;
	unsigned int fileLayerCount = 0u;

	if (!(in >> tag) || tag != "version" || !(in >> version))
		return false;
	if (version != 2)
		return false;
	if (!(in >> tag) || tag != "netType" || !(in >> fileNetType))
		return false;
	if (!(in >> tag) || tag != "inputSize" || !(in >> inputSize))
		return false;
	if (!(in >> tag) || tag != "layers" || !(in >> fileLayerCount))
		return false;

	// Install file netType so future saves preserve the loaded layout.
	netType = fileNetType;

	// Ensure an input layer exists so getLayerSize(0) is well-defined for any legacy callers.
	if (!inputLayer || static_cast<unsigned int>(inputLayer->size()) != inputSize)
	{
		inputLayer = shmea::GPointer<Layer>(new Layer(Layer::INPUT_TYPE));
		for (unsigned int i = 0; i < inputSize; ++i)
		{
			shmea::GPointer<Node> n(new Node());
			if (n)
			{
				n->setWeight(0.0f);
				inputLayer->addNode(n);
			}
		}
		inputRowCount = 0u;
		inputFeatureCount = inputSize;
	}

	const unsigned int expectedLayerCount = static_cast<unsigned int>(skeleton->numHiddenLayers()) + 1u;
	if (fileLayerCount != expectedLayerCount)
		return false;

	// gateCount derived from netType
	unsigned int gateCount = 1u;
	if (netType == NNetwork::TYPE_GRU)
		gateCount = 3u;
	else if (netType == NNetwork::TYPE_LSTM)
		gateCount = 4u;

	if (layers.size() < expectedLayerCount)
		layers.resize(expectedLayerCount);

	// Helper for allocating edge vectors
	struct EdgeFactory
	{
		static std::vector<shmea::GPointer<glades::Edge> > make(unsigned int n, float fill = 0.0f)
		{
			std::vector<shmea::GPointer<glades::Edge> > v;
			v.reserve(n);
			for (unsigned int i = 0; i < n; ++i)
				v.push_back(shmea::GPointer<glades::Edge>(new glades::Edge(-1, fill)));
			return v;
		}
	};

	for (unsigned int li = 0; li < expectedLayerCount; ++li)
	{
		// Parse layer header
		unsigned int layerIdx = 0u;
		int layerType = 0;
		float biasAvg = 0.0f;
		unsigned int nodeCount = 0u;
		unsigned int nodeEdges = 0u;
		unsigned int ctxEdges = 0u;

		if (!(in >> tag) || tag != "layer" || !(in >> layerIdx) || layerIdx != li)
			return false;
		if (!(in >> tag) || tag != "type" || !(in >> layerType))
			return false;
		if (!(in >> tag) || tag != "biasAvg" || !(in >> biasAvg))
			return false;
		if (!(in >> tag) || tag != "nodes" || !(in >> nodeCount))
			return false;
		if (!(in >> tag) || tag != "nodeEdges" || !(in >> nodeEdges))
			return false;
		if (!(in >> tag) || tag != "ctxEdges" || !(in >> ctxEdges))
			return false;

		// Compute expected shape from skeleton + file netType.
		const bool isOutput = (li == expectedLayerCount - 1u);
		const unsigned int curSize = isOutput ? static_cast<unsigned int>(skeleton->getOutputLayerSize())
		                                      : static_cast<unsigned int>(skeleton->getHiddenLayerSize(li));
		const unsigned int prevSize =
		    isOutput
		        ? (expectedLayerCount == 1u ? inputSize : static_cast<unsigned int>(skeleton->getHiddenLayerSize(expectedLayerCount - 2u)))
		        : (li == 0u ? inputSize : static_cast<unsigned int>(skeleton->getHiddenLayerSize(li - 1u)));

		unsigned int expectNodeEdges = 0u;
		unsigned int expectCtxEdges = 0u;
		if (isOutput)
		{
			expectNodeEdges = prevSize + 1u;
			expectCtxEdges = 0u;
		}
		else
		{
			if (netType == NNetwork::TYPE_GRU || netType == NNetwork::TYPE_LSTM)
			{
				expectNodeEdges = gateCount * (prevSize + 1u);
				expectCtxEdges = gateCount * curSize;
			}
			else if (netType == NNetwork::TYPE_RNN)
			{
				expectNodeEdges = prevSize + 1u;
				expectCtxEdges = curSize;
			}
			else
			{
				expectNodeEdges = prevSize + 1u;
				expectCtxEdges = 0u;
			}
		}

		if (nodeCount != curSize)
			return false;
		if (nodeEdges != expectNodeEdges)
			return false;
		if (ctxEdges != expectCtxEdges)
			return false;

		// Ensure layer object exists and has the correct type.
		if (!layers[li])
		{
			layers[li] = shmea::GPointer<Layer>(new Layer(isOutput ? Layer::OUTPUT_TYPE : Layer::HIDDEN_TYPE));
		}
		Layer* layer = layers[li].get();
		if (!layer)
			return false;
		layer->setType(isOutput ? Layer::OUTPUT_TYPE : Layer::HIDDEN_TYPE);

		// Ensure correct number of nodes.
		if (layer->size() > nodeCount)
			return false;
		while (layer->size() < nodeCount)
		{
			shmea::GPointer<Node> n(new Node());
			if (!n)
				return false;
			n->setEdges(EdgeFactory::make(nodeEdges, 0.0f));
			if (ctxEdges > 0u)
			{
				shmea::GPointer<Node> ctx(new Node());
				if (!ctx)
					return false;
				ctx->setEdges(EdgeFactory::make(ctxEdges, 0.0f));
				n->setContextNode(ctx);
			}
			layer->addNode(n);
		}

		// Load per-node weights (and context-node weights).
		for (unsigned int ni = 0; ni < nodeCount; ++ni)
		{
			unsigned int nIdx = 0u;
			if (!(in >> tag) || tag != "n" || !(in >> nIdx) || nIdx != ni)
				return false;

			Node* node = layer->getNode(ni);
			if (!node)
				return false;

			// Ensure edge vector shape matches.
			if (node->numEdges() != nodeEdges)
				node->setEdges(EdgeFactory::make(nodeEdges, 0.0f));

			for (unsigned int k = 0; k < nodeEdges; ++k)
			{
				float w = 0.0f;
				if (!(in >> w))
					return false;
				node->setEdgeWeight(k, w);
			}

			if (ctxEdges > 0u)
			{
				Node* ctx = node->getContextNode();
				if (!ctx)
				{
					shmea::GPointer<Node> ownedCtx(new Node());
					if (!ownedCtx)
						return false;
					ownedCtx->setEdges(EdgeFactory::make(ctxEdges, 0.0f));
					node->setContextNode(ownedCtx);
					ctx = node->getContextNode();
				}
				if (ctx->numEdges() != ctxEdges)
					ctx->setEdges(EdgeFactory::make(ctxEdges, 0.0f));

				for (unsigned int k = 0; k < ctxEdges; ++k)
				{
					float w = 0.0f;
					if (!(in >> w))
						return false;
					ctx->setEdgeWeight(k, w);
				}
			}
			else
			{
				// Ensure we do not retain stale context nodes when loading a non-recurrent model.
				shmea::GPointer<Node> empty;
				node->setContextNode(empty);
			}
		}
	}

	return true;
}

