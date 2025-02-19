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
#include "LayerBuilder.h"
#include "../../../main.h"
#include "../GMath/gmath.h"
#include "../Structure/nninfo.h"
#include "../Networks/network.h"
#include "../DataObjects/DataInput.h"
#include "Backend/Database/GList.h"
#include "Backend/Database/GTable.h"
#include "Backend/Database/GType.h"
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

glades::LayerBuilder::LayerBuilder()
{
	netType = NNetwork::TYPE_DFF;
}

glades::LayerBuilder::LayerBuilder(int newNetType)
{
	netType = newNetType;
}

glades::LayerBuilder::~LayerBuilder()
{
    //
}

bool glades::LayerBuilder::build(const NNInfo* skeleton, const DataInput* newInput, bool standardizeWeightsFlag)
{
	if (!skeleton)
		return false;

	// Construct the input layers
	printf("[GQL] Building input layers\n");
	buildInputLayers(skeleton, newInput);
	if (inputLayers.size() <= 0)
	{
		printf("[GQL] Invalid data format[0] %s\n", skeleton->getName().c_str());
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
		printf("[GQL] Invalid data format[1] %s\n", skeleton->getName().c_str());
		return false;
	}

	// standardize the weights between the neurons
	if (standardizeWeightsFlag)
		standardizeWeights(skeleton);

	return true;
}

void glades::LayerBuilder::buildInputLayers(const NNInfo* skeleton, const DataInput* di)
{
	clock_t startFunction = clock();  // Overall function start time

	unsigned int featureCount = di->getFeatureCount(); 
	unsigned int trainsize = di->getTrainSize();
	unsigned int inputLayerssize = inputLayers.size();
	// Needs something to train/test on
	if (trainsize == 0)
		return;

	// Input and Output columns
	if (featureCount < 1)
		return;

	// TODO: SPEED THIS UP FOR IMAGES!!!!!!!!!!!
	inputLayers.clear();
	// unsigned int t = di->getTrainSize();
	printf("Train size: %d\n", trainsize);

	clock_t startOuterLoop = clock();  // Start time for the outer loop
	for (unsigned int r = 0; r < trainsize; ++r)
	{
		clock_t startInnerLoop = clock();  // Start time for the inner loop

		printf("[GQL] Building input layer %d\n", r);
		Layer* cLayer = new Layer(Layer::INPUT_TYPE, false);

		// for (unsigned int c = 0; c < featureCount; ++c)
		// {
		// 	// We can probably get rid of most of these conditions becuase Gtype auto types
		// 	float newWeight = 0.0f;
		// 	shmea::GType cCell = di->getTrainRow(r)[c];
		// }

		for (unsigned int c = 0; c < featureCount; ++c)
		{
			// We can probably get rid of most of these conditions becuase Gtype auto types
			float newWeight = 0.0f;
			shmea::GType cCell = di->getTrainRow(r)[c];

			shmea::GType::Type cellType = cCell.getType(); 
			// printf("Cell type %s",cellType);

			switch (cellType)
			{
				case shmea::GType::STRING_TYPE:
					if (!di->featureIsCategorical[c])
					{
						inputLayers.clear();
						return;
					}
					break;
				case shmea::GType::CHAR_TYPE:   newWeight = cCell.getChar();   break;
				case shmea::GType::SHORT_TYPE:  newWeight = cCell.getShort();  break;
				case shmea::GType::INT_TYPE:    newWeight = cCell.getInt();    break;
				case shmea::GType::LONG_TYPE:   newWeight = cCell.getLong();   break;
				case shmea::GType::FLOAT_TYPE:  newWeight = cCell.getFloat();  break;
				case shmea::GType::DOUBLE_TYPE: newWeight = cCell.getDouble(); break;
				case shmea::GType::BOOLEAN_TYPE:
					newWeight = cCell.getBoolean() ? 1.0f : 0.0f;
					break;
				default:
					break;  // Skip unsupported types
			}
			
			// Error
			Node* node = new Node();
			if (!node)
				continue;

			// set the node weight
			node->setWeight(newWeight);

			// add the new input node to the layer
			cLayer->addNode(node);

			// add the input layer to the dataset
			// bool lastCol = (c == featureCount - 1);
			if (c == featureCount - 1)
			{
			    printf("Adding input layer[%u:%u]: %d\n", r, trainsize, inputLayers.size());
				inputLayers.push_back(cLayer);
			}
		}
		
		clock_t endInnerLoop = clock();  // End time for the inner loop
        printf("Time for inner loop (row %u): %.3f ms\n", r,
               1000.0 * (endInnerLoop - startInnerLoop) / CLOCKS_PER_SEC);
	}
}

void glades::LayerBuilder::buildHiddenLayers(const NNInfo* skeleton)
{
	int inputLayerSize = inputLayers[0]->size();
	int outputLayerSize = skeleton->getOutputLayerSize();
	int prevLayerSize = inputLayerSize;
	int outputType = skeleton->getOutputType();
	bool isPositive = false;
	int activationType;

	// Create each hidden layer
	for (int i = 0; i < skeleton->numHiddenLayers(); ++i)
	{
		activationType = skeleton->getActivationType(i);
		// Get the current layer size
		int cLayerSize = skeleton->getHiddenLayerSize(i);
		Layer* cLayer = new Layer(Layer::HIDDEN_TYPE);

		if (isPositive)
		{
			// Create the hidden layer
			cLayer->initWeights(prevLayerSize, cLayerSize, Node::INIT_POSXAVIER, activationType);
		}
		else
		{
			// Create the hidden layer
			if ((activationType == GMath::SIGMOID) || (activationType == GMath::RELU) ||
				(activationType == GMath::LEAKY) || (outputType == GMath::CLASSIFICATION))
			{
				isPositive = true;
				i = -1;
				for (unsigned int j = 0; j < layers.size(); ++j)
					delete layers[j];
				layers.clear();
				continue;
			}

			cLayer->initWeights(prevLayerSize, cLayerSize, Node::INIT_XAVIER, activationType);
		}

		layers.push_back(cLayer);
		prevLayerSize = cLayerSize;
	}
}

void glades::LayerBuilder::buildOutputLayer(const NNInfo* skeleton)
{
	int inputLayerSize = inputLayers[0]->size();
	int outputLayerSize = skeleton->getOutputLayerSize();
	int prevLayerSize = inputLayerSize;
	int outputType = skeleton->getOutputType();
	bool isPositive = false;
	int activationType;

	// Create each hidden layer
	for (int i = 0; i < skeleton->numHiddenLayers(); ++i)
	{
		activationType = skeleton->getActivationType(i);
		// Get the current layer size
		int cLayerSize = skeleton->getHiddenLayerSize(i);

		// Create the hidden layer
		if ((activationType == GMath::SIGMOID) || (activationType == GMath::RELU) ||
			(activationType == GMath::LEAKY) || (outputType == GMath::CLASSIFICATION))
			isPositive = true;

		prevLayerSize = cLayerSize;
	}

	// Create the output layer
	Layer* cLayer = new Layer(Layer::OUTPUT_TYPE);
	if (isPositive)
		cLayer->initWeights(prevLayerSize, outputLayerSize, Node::INIT_POSXAVIER, activationType);
	else
		cLayer->initWeights(prevLayerSize, outputLayerSize, Node::INIT_XAVIER, activationType);
	layers.push_back(cLayer);
}

glades::NetworkState* glades::LayerBuilder::getNetworkStateFromLoc(unsigned int inputRowCounter, unsigned int cInputLayerCounter,
	unsigned int cOutputLayerCounter, unsigned int cInputNodeCounter, unsigned int cOutputNodeCounter)
{
	// Base case and Error case
	if (cOutputLayerCounter > layers.size())
		return NULL;

	if (cInputLayerCounter >= layers.size())
		return NULL;

	// Current Input Layer
	Layer* cInputLayer = NULL;
	if (cInputLayerCounter == 0)
		cInputLayer = inputLayers[inputRowCounter];
	else
		cInputLayer = layers[cInputLayerCounter-1];
	if (!cInputLayer)
		return NULL;

	// Current Input Node Error Check
	if (cInputNodeCounter >= cInputLayer->size())
		return NULL;

	// Current Input Node
	Node* cInputNode = cInputLayer->getNode(cInputNodeCounter);
	if (!cInputNode)
		return NULL;

	// Current Output Layer
	Layer* cOutputLayer = layers[cOutputLayerCounter-1];
	if (!cOutputLayer)
		return NULL;

	// Why would this happen??
	if (cOutputLayer->getType() == Layer::INPUT_TYPE)
		return NULL;

	// Current Output Node Error Check
	if (cOutputNodeCounter >= cOutputLayer->size())
		return NULL;

	// Current Output Node
	Node* cOutputNode = (*cOutputLayer)[cOutputNodeCounter];
	if (!cOutputNode)
		return NULL;

	// Input Dropout Check
	bool validInputNode = cInputLayer->possiblePath(cInputNodeCounter);

	// Output Dropout Check
	bool validOutputNode = cOutputLayer->possiblePath(cOutputNodeCounter);

	// Input helper vars
	bool firstValidInputNode = (cInputNodeCounter == cInputLayer->firstValidPath());
	bool lastValidInputNode = (cInputNodeCounter == cInputLayer->lastValidPath());

	// Output helper vars
	bool firstValidOutputNode = (cOutputNodeCounter == cOutputLayer->firstValidPath());
	bool lastValidOutputNode = (cOutputNodeCounter == cOutputLayer->lastValidPath());

	// Create the return structure
	glades::NetworkState* newLoc = new glades::NetworkState(
		cInputLayerCounter, cOutputLayerCounter, cInputNodeCounter, cOutputNodeCounter, cInputLayer,
		cOutputLayer, cInputNode, cOutputNode, firstValidInputNode, lastValidInputNode,
		firstValidOutputNode, lastValidOutputNode, validInputNode, validOutputNode);

	return newLoc;
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
	return inputLayers.size();
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
	return inputLayers[0]->size();
	
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

shmea::GList glades::LayerBuilder::getWeights()
{
   shmea::GList weights; 
   //We start with 1 because the first layer (input layer) doesn't have the data of the weights
    for(unsigned int i = 0; i < getLayersSize(); ++i)
    {
	std::vector<Node*> cChildren = layers[i]->getChildren();
	for(unsigned int j = 0; j < cChildren.size(); ++j)
	{
	   for(unsigned int k = 0; k < cChildren[j]->numEdges(); ++k)
	   {
		float cWeight = cChildren[j]->getEdgeWeight(k);
		weights.addFloat(cWeight);
	   }
	   weights.addString(',');
	}
	weights.addString(';');
    }

    return weights;
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

		// iterate through the nodes
		std::vector<Node*> cChildren = layers[i]->getChildren();
		for (unsigned int j = 0; j < cChildren.size(); ++j)
		{
			// iterate through the node weights
			for (unsigned int k = 0; k < cChildren[j]->numEdges(); ++k)
			{
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

		// iterate through the nodes
		std::vector<Node*> cChildren = layers[i]->getChildren();
		for (unsigned int j = 0; j < cChildren.size(); ++j)
		{
			// iterate through the node weights
			for (unsigned int k = 0; k < cChildren[j]->numEdges(); ++k)
			{
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

	if (inputRowCounter >= inputLayers.size())
		return;

	Layer* cInputLayer = inputLayers[inputRowCounter];
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
	if (inputLayers.size() == 0)
		return;

	if (layers.size() == 0)
		return;

	// print input layer info
	printf("[GQL] Input(r,c): (%ld,%d)\n", inputLayers.size(), inputLayers[0]->size());
	if (override)
	{
		for (unsigned int i = 0; i < inputLayers.size(); ++i)
		{
			printf("%d [%d]: ", i + 1, inputLayers[i]->getType());
			inputLayers[i]->print();
		}
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
	inputLayers.clear();
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
	shmea::SaveFolder* nnList = new shmea::SaveFolder(netName.c_str());

	std::vector<shmea::GString> layerHeaders, edgeHeaders;
	layerHeaders.push_back("BiasWeight");
	edgeHeaders.push_back("layerID");
	edgeHeaders.push_back("nodeID");

	// Save all the layers in a
	shmea::GTable layerTable(',', layerHeaders);
	shmea::GTable edgeTable(',', edgeHeaders);
	for (unsigned int layerIdx = 0; layerIdx < getLayersSize(); ++layerIdx)
	{
		Layer* layer = layers[layerIdx];
		if (!layer)
			continue;

		// Add to the layer table
		shmea::GList layerRow;
		layerRow.addFloat(layer->getBiasWeight());
		layerTable.addRow(layerRow);

		// Save each edge in the edgeTable
		std::vector<Node*> nodes = layer->getChildren();
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
	nnList->newItem("layers", layerTable);
	nnList->newItem("edges", edgeTable);

	return true;
}
