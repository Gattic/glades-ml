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
#include "inputlayerinfo.h"
#include "Backend/Database/GList.h"
#include "Backend/Database/GType.h"

using namespace glades;

/*!
 * @brief InputLayerInfo constructor
 * @details create a new InputLayerInfo object
 * @param newSize the InputLayerInfo object's desired size
 * @param newDropout the dropout rate for the input layer
 */
glades::InputLayerInfo::InputLayerInfo(float newDropout, int newBatchSize) : LayerInfo(0)
{
	dropoutRate = newDropout;
	batchSize = newBatchSize;
}

/*!
 * @brief InputLayerInfo destructor
 * @details destroy a InputLayerInfo object
 */
glades::InputLayerInfo::~InputLayerInfo()
{
	dropoutRate = 0.0f;
}

/*!
 * @brief get input layer dropout rate
 * @details get InputLayerInfo's dropout rate, the probability that training will use input node
 * N
 * @return the InputLayerInfo's dropout rate
 */
float glades::InputLayerInfo::getDropoutRate() const
{
	return dropoutRate;
}

/*!
 * @brief get input layer batch size
 * @details get NN/InputLayerInfo's batch size
 * N
 * @return the NN/InputLayerInfo's batch size
 */
int glades::InputLayerInfo::getBatchSize() const
{
	return batchSize;
}

/*!
 * @brief get GTable row
 * @details get the GTable row from this layer
 * @return the InputLayer's GTable row representation
 */
shmea::GList glades::InputLayerInfo::getGTableRow() const
{
	shmea::GList row;
	// structure: size, pInput, batchSize, learningRate, momentumFactor, weightDecay, pHidden,
	// activationType,
	// activationParam, outputType
	// -1 = "blank"/placeholder
	row.addLong(-1);
	row.addFloat(getDropoutRate());
	row.addLong(getBatchSize());
	for (int i = 0; i < 4; ++i)
		row.addFloat(-1.0f);
	row.addLong(-1);
	row.addFloat(-1.0f);
	row.addLong(-1);

	return row;
}

/*!
 * @brief set the dropout rate
 * @details set the dropout rate for this InputLayerInfo object
 * @param newDropout the desired dropout rate
 */
void glades::InputLayerInfo::setDropoutRate(float newDropout)
{
	dropoutRate = newDropout;
}

/*!
 * @brief set the NN batch size
 * @details set the mini batch size for the nnetwork; Look at flags in NNInfo.h
 * @param newBatchSize the desired batch size
 */
void glades::InputLayerInfo::setBatchSize(int newBatchSize)
{
	batchSize = newBatchSize;
}

/*!
 * @brief get type
 * @details return the layer type, a constant stored in the LayerInfo class
 * @return the layer's type
 */
int glades::InputLayerInfo::getLayerType() const
{
	return LayerInfo::INPUT;
}
