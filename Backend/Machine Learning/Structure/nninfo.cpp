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
#include "nninfo.h"
#include "../../../main.h"
#include "Backend/Database/GTable.h"
#include "Backend/Database/GType.h"
#include "Backend/Database/SaveFolder.h"
#include "Backend/Database/SaveTable.h"
#include "Backend/Networking/main.h"
#include "hiddenlayerinfo.h"
#include "inputlayerinfo.h"
#include "layerinfo.h"
#include "outputlayerinfo.h"

using namespace glades;

/*!
 * @brief NNInfo constructor
 * @details create a new NNInfo object
 * @param newName the NNInfo object's desired name
 */
glades::NNInfo::NNInfo(const shmea::GString& newName)
{
	name = newName;
	inputType = 0;
	hiddenLayerCount = 0;
}

/*!
 * @brief NNInfo constructor
 * @details create a new NNInfo object
 * @param newName the NNInfo object's desired name
 * @param newInputLayer the desired input layer
 * @param hidden the desired hidden layers
 * @param newOutputLayer the desired output layer
 */
glades::NNInfo::NNInfo(const shmea::GString& newName, InputLayerInfo* newInputLayer,
					   const std::vector<HiddenLayerInfo*>& hidden, OutputLayerInfo* newOutputLayer)
{
	name = newName;
	inputType = 0;
	hiddenLayerCount = hidden.size();
	inputLayer = newInputLayer;
	outputLayer = newOutputLayer;

	// populate the hidden layers
	for (int i = 0; i < hiddenLayerCount; ++i)
		layers.push_back(hidden[i]);
}

/*!
 * @brief NNInfo constructor from GTable
 * @details create a new NNInfo object from a GTable
 * @param  newTable the table storing the NNInfo data
 */
glades::NNInfo::NNInfo(const shmea::GString& newName, const shmea::GTable& newTable)
{
	int rows = newTable.numberOfRows(), cols = newTable.numberOfCols(), r = 0;
	if (cols < 10)
		printf("[NNINFO] Bad table size: (%d,%d)\n", rows, cols);
	if (newTable.getHeader(0) != "Size")
		printf("[NNINFO] Bad GTable schema!\n");

	hiddenLayerCount = rows - 2;
	layers.reserve(hiddenLayerCount);
	name = newName;
	fromGTable(newName, newTable);
}

/*!
 * @brief NNInfo destructor
 * @details destroy a NNInfo object
 */
glades::NNInfo::~NNInfo()
{
	name = "";
	inputType = 0;
	hiddenLayerCount = 0;
	for (unsigned int i = 0; i < layers.size(); ++i)
		delete layers[i];
	layers.clear();

	if (inputLayer)
		delete inputLayer;
	inputLayer = NULL;

	if (outputLayer)
		delete outputLayer;
	outputLayer = NULL;
}

/*!
 * @brief get name
 * @details get NNInfo's name
 * @return the NNInfo's name
 */
shmea::GString glades::NNInfo::getName() const
{
	return name;
}

int glades::NNInfo::getInputType() const
{
	return inputType;
}

/*!
 * @brief get output layer type
 * @details get NNInfo's output layer output type, Regression or Clasification
 * @return the NNInfo's output layer output type
 */
int glades::NNInfo::getOutputType() const
{
	if (!outputLayer)
		return OutputLayerInfo::REGRESSION;

	return outputLayer->getOutputType();
}

/*!
 * @brief get input layer dropout rate
 * @details get NNInfo's input layer dropout rate, the probability that training will drop input
 * neuron N
 * @return the NNInfo's input layer dropout rate
 */
float glades::NNInfo::getPInput() const
{
	if (!inputLayer)
		return 0.0f;

	return inputLayer->getPDropout();
}

/*!
 * @brief get input layer batch size
 * @details get NNInfo's batch size, look at the flags in the header file
 * neuron N
 * @return the NNInfo's batch size
 */
int glades::NNInfo::getBatchSize() const
{
	if (!inputLayer)
		return 1;

	return inputLayer->getBatchSize();
}

/*!
 * @brief get input layer
 * @details get NNInfo's input layer
 * @return the NNInfo's input layer
 */
InputLayerInfo* glades::NNInfo::getInputLayer() const
{
	return inputLayer;
}

/*!
 * @brief get layers
 * @details get NNInfo's layers
 * @return the NNInfo's layers, a vector where index `i` is the number of neurons in the `i`th layer
 */
std::vector<HiddenLayerInfo*> glades::NNInfo::getLayers() const
{
	return layers;
}

/*!
 * @brief get the number of layers
 * @details get NNInfo's number of layers
 * @return the NNInfo's number of layers
 */
int glades::NNInfo::numHiddenLayers() const
{
	return layers.size();
}

/*!
 * @brief get the size of a layer
 * @details get the size of a layer
 * @return get the size of layer 'index'
 */
int glades::NNInfo::getInputLayerSize() const
{
	if (!inputLayer)
		return 0;

	return inputLayer->size();
}

/*!
 * @brief get the size of a layer
 * @details get the size of a layer
 * @return get the size of layer 'index'
 */
int glades::NNInfo::getHiddenLayerSize(unsigned int index) const
{
	if (index >= layers.size())
		return 0;

	if (!layers[index])
		return 0;

	return layers[index]->size();
}

/*!
 * @brief get the size of a layer
 * @details get the size of a layer
 * @return get the size of layer 'index'
 */
unsigned int glades::NNInfo::getOutputLayerSize() const
{
	if (!outputLayer)
		return 0;

	return outputLayer->size();
}

/*!
 * @brief get learning rate
 * @details get NNInfo's learning rate
 * @param index the layer index
 * @return the NNInfo's learning rate
 */
float glades::NNInfo::getLearningRate(unsigned int index) const
{
	if (index > layers.size())
		return 0.01f;

	if(index == 0)
	    return inputLayer->getLearningRate();

	if (!layers[index-1])
		return 0.01f;

	return layers[index-1]->getLearningRate();
}

/*!
 * @brief get momentum factor
 * @details get NNInfo's momentum factor
 * @param index the layer index
 * @return the NNInfo's momentum factor
 */
float glades::NNInfo::getMomentumFactor(unsigned int index) const
{
	if (index > layers.size())
		return 0.0f;

	if(index == 0)
	    return inputLayer->getMomentumFactor();

	if (!layers[index-1])
		return 0.0f;

	return layers[index-1]->getMomentumFactor();
}

/*!
 * @brief get weight decay
 * @details get NNInfo's weight decay
 * @param index the layer index
 * @return the NNInfo's weight decay
 */
float glades::NNInfo::getWeightDecay(unsigned int index) const
{
	if (index > layers.size())
		return 0.0f;

	if(index == 0)
	    return inputLayer->getWeightDecay();

	if (!layers[index-1])
		return 0.0f;

	return layers[index-1]->getWeightDecay();
}

/*!
 * @brief get hidden layer dropout rate
 * @details get NNInfo's hidden layer dropout rate, the probability that training will drop hidden
 * neuron N
 * @param index the layer index
 * @return the NNInfo's hidden layer dropout rate
 */
float glades::NNInfo::getPDropout(unsigned int index) const
{
	if (index > layers.size())
		return 1.0f;

	if(index == 0)
	    return inputLayer->getPDropout();

	if (!layers[index-1])
		return 1.0f;

	return layers[index-1]->getPDropout();
}

/*!
 * @brief get hidden layer activation type
 * @details get NNInfo's hidden layer activation type
 * @param index the layer index
 * @return the NNInfo's hidden layer activation type
 */
int glades::NNInfo::getActivationType(unsigned int index) const
{
	if (index > layers.size())
		return 0;

	if(index == 0)
	    return inputLayer->getActivationType();

	if (!layers[index-1])
		return 0;

	return layers[index-1]->getActivationType();
}

/*!
 * @brief get hidden layer activation parameter
 * @details get NNInfo's hidden layer activation parameter, for use with certain activation types
 * @param index the layer index
 * @return the NNInfo's hidden layer activation parameter
 */
float glades::NNInfo::getActivationParam(unsigned int index) const
{
	if (index > layers.size())
		return 0;

	if(index == 0)
	    return inputLayer->getActivationParam();

	if (!layers[index-1])
		return 0;

	return layers[index-1]->getActivationParam();
}

/*!
 * @brief Prints NNInfo
 * @details Prints the values contained in the inputLayer, layers, and outputLayer
 */
void glades::NNInfo::print() const
{
	std::vector<shmea::GString> headers;
	headers.push_back("Layer");
	headers.push_back("Size");
	headers.push_back("batchSize");
	headers.push_back("lrnRate");
	headers.push_back("mFactor");
	headers.push_back("wDecay");
	headers.push_back("pDropout");
	headers.push_back("actvtn");
	headers.push_back("actParam");
	headers.push_back("cost");

	// put everything in a GTable
	shmea::GTable printTable(',', headers);
	printTable.addRow(inputLayer->getGTableRow());
	for (unsigned int i = 0; i < layers.size(); ++i)
		printTable.addRow(layers[i]->getGTableRow());
	printTable.addRow(outputLayer->getGTableRow());

	// print headers
	printf("[NNINFO] Current NNInfo Values\n");
	printf("[NNINFO] ");
	for (unsigned int i = 0; i < headers.size(); ++i)
		printf("%s\t", headers[i].c_str());
	printf("\n");

	// print layer info
	for (unsigned int row = 0; row < printTable.numberOfRows(); ++row)
	{
		bool firstRow = (row == 0);
		bool lastRow = (row == (printTable.numberOfRows() - 1));

		printf("[NNINFO] ");
		if (firstRow)
			printf("Input\t");
		else if (!lastRow)
			printf("H[%d]\t", row);
		else
			printf("Output\t");

		for (unsigned int col = 0; col < printTable.numberOfCols(); ++col)
		{
			shmea::GType cell = printTable.getCell(row, col);
			shmea::GString word = cell;
			if (atoi(word.c_str()) == -1)
				printf("\t");
			else
				printf("%s\t", word.c_str());
		}
		printf("\n");
	}
}

/*!
 * @brief set name
 * @details set NNInfo's name
 * @param newName the desired name for this NNInfo object
 */
void glades::NNInfo::setName(shmea::GString newName)
{
	name = newName;
}

void glades::NNInfo::setInputType(int newInputype)
{
	inputType = newInputype;
}

/*!
 * @brief set the p input
 * @details set the dropout rate for the input layer
 * @param newPInput the desired input dropout rate
 */
void glades::NNInfo::setPInput(float newPInput)
{
	if (!inputLayer)
		return;

	inputLayer->setPDropout(newPInput);
}

/*!
 * @brief set the batch size
 * @details set the batch size; check the flags in the header file
 * @param newBatchSize the desired batch size
 */
void glades::NNInfo::setBatchSize(int newBatchSize)
{
	if (!inputLayer)
		return;

	inputLayer->setBatchSize(newBatchSize);
}

/*!
 * @brief set the output layer type
 * @details set the output layer type
 * @param newOutputType the desired output layer type
 */
void glades::NNInfo::setOutputType(int newOutputType)
{
	if (!outputLayer)
		return;

	outputLayer->setOutputType(newOutputType);
}

/*!
 * @brief set the output layer size
 * @details set the output layer size
 * @param newOutputSize the desired output layer size
 */
void glades::NNInfo::setOutputSize(int newOutputSize)
{
	if (!outputLayer)
		return;

	outputLayer->setSize(newOutputSize);
}

/*!
 * @brief set layers
 * @details set NNInfo's layers
 * @param newLayers the desired layers, a vector where each index `i` corresponds to the number of
 * nodes in the `i`th layer
 */
void glades::NNInfo::setLayers(const std::vector<HiddenLayerInfo*>& newLayers)
{
	layers = newLayers;
}

/*!
 * @brief set learning rate
 * @details set NNInfo's learning rate
 * @param index the layer index
 * @param newLearningRate the desired learning rate for this NNInfo object
 */
void glades::NNInfo::setLearningRate(unsigned int index, float newLearningRate)
{
	if (index > layers.size())
		return;
	
	if(index == 0)
	    inputLayer->setLearningRate(newLearningRate);
	else
	{
	    if (!layers[index-1])
		    return;

	    layers[index-1]->setLearningRate(newLearningRate);
	}
}

/*!
 * @brief set momentum factor
 * @details set NNInfo's momentum factor
 * @param index the layer index
 * @param newMomentumFactor the desired momentum factor for this NNInfo object
 */
void glades::NNInfo::setMomentumFactor(unsigned int index, float newMomentumFactor)
{
	if (index > layers.size())
		return;

	if(index == 0)
	    inputLayer->setMomentumFactor(newMomentumFactor);
	else
	{
	    if (!layers[index-1])
		    return;

	    layers[index-1]->setMomentumFactor(newMomentumFactor);
	}
}

/*!
 * @brief set weight decay
 * @details set NNInfo's weight decay
 * @param index the layer index
 * @param newWeightDecay the desired weight decay for this NNInfo object
 */
void glades::NNInfo::setWeightDecay(unsigned int index, float newWeightDecay)
{
	if (index > layers.size())
		return;

	if(index == 0)
	    inputLayer->setWeightDecay(newWeightDecay);
	else
	{
	    if (!layers[index-1])
		    return;

	    layers[index-1]->setWeightDecay(newWeightDecay);
	}
}

/*!
 * @brief set the p hidden
 * @details set the dropout rate for hidden layer index
 * @param index the layer index
 * @param newPDropout the desired input dropout rate
 */
void glades::NNInfo::setPDropout(unsigned int index, float newPDropout)
{
	if (index > layers.size())
		return;

	if(index == 0)
	    inputLayer->setPDropout(newPDropout);
	else
	{
	    if (!layers[index-1])
		    return;

	    layers[index-1]->setPDropout(newPDropout);
	}
}

/*!
 * @brief set the activation type
 * @details set the activation type
 * @param index the layer index
 * @param newActivationType the desired activation type
 */
void glades::NNInfo::setActivationType(unsigned int index, int newActivationType)
{
	if (index > layers.size())
		return;

	if(index == 0)
	    inputLayer->setActivationType(newActivationType);
	else
	{
	    if (!layers[index-1])
		    return;

	    layers[index-1]->setActivationType(newActivationType);
	}
}

/*!
 * @brief set the activation parameter
 * @details set the activation parameter to be used with certain activation types
 * @param index the layer index
 * @param newActivationParam the desired activation parameter
 */
void glades::NNInfo::setActivationParam(unsigned int index, float newActivationParam)
{
	if (index > layers.size())
		return;

	if(index == 0)
	    inputLayer->setActivationParam(newActivationParam);
	else
	{
	    if (!layers[index-1])
		    return;

	    layers[index-1]->setActivationParam(newActivationParam);
	}
}

void glades::NNInfo::addHiddenLayer(HiddenLayerInfo* newLayer)
{
	if (!newLayer)
		return;

	layers.push_back(newLayer);
}

void glades::NNInfo::copyHiddenLayer(unsigned int dst, unsigned int src)
{
	if (dst >= layers.size())
		return;

	if (!layers[dst])
		return;

	if (src >= layers.size())
		return;

	if (!layers[src])
		return;

	layers[dst]->copyParamsFrom(layers[src]);
}

void glades::NNInfo::resizeHiddenLayers(unsigned int newCount)
{
	layers.resize(newCount, new HiddenLayerInfo(2, 0.01, 0.0, 0.0, 0.0, 0, 0.0));
}

void glades::NNInfo::removeHiddenLayer(unsigned int index)
{
	if (index >= layers.size())
		return;

	if (!layers[index])
		return;

	layers.erase(layers.begin() + index);
}

/*!
 * @brief NNInfo to GTable
 * @details convert the layers in an NNInfo object to a GTable object
 */
shmea::GTable glades::NNInfo::toGTable() const
{
	std::vector<shmea::GString> headers;
	headers.push_back("Size");
	headers.push_back("batchSize");
	headers.push_back("learningRate");
	headers.push_back("momentumFactor");
	headers.push_back("weightDecay");
	headers.push_back("pDropout");
	headers.push_back("activationType");
	headers.push_back("activationParam");
	headers.push_back("outputType");

	shmea::GTable newTable(',', headers);
	newTable.addRow(inputLayer->getGTableRow());
	for (unsigned int i = 0; i < layers.size(); ++i)
		newTable.addRow(layers[i]->getGTableRow());
	newTable.addRow(outputLayer->getGTableRow());

	return newTable;
}

/*!
 * @brief NNInfo to GTable
 * @details convert the layers in an NNInfo object to a GTable object
 */
bool glades::NNInfo::fromGTable(const shmea::GString& netName, const shmea::GTable& newTable)
{
	if (newTable.numberOfRows() < 2)
		return false;

	//
	name = netName;
	for (unsigned int i = 0; i < newTable.numberOfRows(); ++i)
	{
		if (i == 0)
		{
			// Input Layer
			inputLayer = new InputLayerInfo(newTable.getCell(i, COL_BATCH_SIZE).getLong(),//batch size not layer size
									newTable.getCell(i, COL_LEARNING_RATE).getFloat(),
									newTable.getCell(i, COL_MOMENTUM_FACTOR).getFloat(),
									newTable.getCell(i, COL_WEIGHT_DECAY).getFloat(),
									newTable.getCell(i, COL_PDROPOUT).getFloat(),
									newTable.getCell(i, COL_ACTIVATION_TYPE).getInt(),
									newTable.getCell(i, COL_ACTIVATION_PARAM).getFloat());
		}
		else if (i == newTable.numberOfRows() - 1)
		{
			// Output Layer
			outputLayer = new OutputLayerInfo(newTable.getCell(i, COL_SIZE).getInt(),
											  newTable.getCell(i, COL_OUTPUT_TYPE).getFloat());
		}
		else
		{
			// Hidden layer
			HiddenLayerInfo* newLayer =
				new HiddenLayerInfo(newTable.getCell(i, COL_SIZE).getInt(),
									newTable.getCell(i, COL_LEARNING_RATE).getFloat(),
									newTable.getCell(i, COL_MOMENTUM_FACTOR).getFloat(),
									newTable.getCell(i, COL_WEIGHT_DECAY).getFloat(),
									newTable.getCell(i, COL_PDROPOUT).getFloat(),
									newTable.getCell(i, COL_ACTIVATION_TYPE).getInt(),
									newTable.getCell(i, COL_ACTIVATION_PARAM).getFloat());
			layers.push_back(newLayer);
		}
	}

	//
	int hiddenLayerCount = layers.size();

	return true;
}

bool glades::NNInfo::load(const shmea::GString& netName)
{
	name = netName;
	shmea::SaveFolder* slItem = new shmea::SaveFolder("neuralnetworks");
	return fromGTable(name.c_str(), slItem->loadItem(name.c_str())->getTable());
}

void glades::NNInfo::save() const
{
	shmea::SaveFolder* slItem = new shmea::SaveFolder("neuralnetworks");
	slItem->newItem(name.c_str(), toGTable());
}
