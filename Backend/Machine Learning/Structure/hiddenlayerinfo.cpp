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
#include "hiddenlayerinfo.h"
#include "Backend/Database/GList.h"

using namespace glades;

/*!
 * @brief HiddenLayerInfo constructor
 * @details create a new HiddenLayerInfo object
 * @param newLearningRate the HiddenLayerInfo object's desired learning rate
 * @param newMomentumFactor the HiddenLayerInfo object's desired momentum factor
 * @param newWeightDecay the HiddenLayerInfo object's desired weight decay
 * @param newPDropout the dropout rate for the hidden layer
 * @param newActivationType the activation type for the hidden layer
 */
glades::HiddenLayerInfo::HiddenLayerInfo(int newSize, float newLearningRate,
    float newMomentumFactor, float newWeightDecay1, float newWeightDecay2,
    float newPDropout, int newActivationType,
    float newActivationParam)
	: LayerInfo(newSize)
{
	learningRate = newLearningRate;
	momentumFactor = newMomentumFactor;
	weightDecay1 = newWeightDecay1;
	weightDecay2 = newWeightDecay2;
	pDropout = newPDropout;
	activationType = newActivationType;
	activationParam = newActivationParam;
}

/*!
 * @brief HiddenLayerInfo destructor
 * @details destroy a HiddenLayerInfo object
 */
glades::HiddenLayerInfo::~HiddenLayerInfo()
{
	learningRate = 0.0f;
	momentumFactor = 0.0f;
	weightDecay1 = 0.0f;
	weightDecay2 = 0.0f;
	pDropout = 0.0f;
	activationType = 0;
	activationParam = 0;
}

/*!
 * @brief get GTable row
 * @details gets the GTable row equivalent of this HiddenLayerInfo object
 * @return the GTable row version of a HiddenLayerInfo object
 */
shmea::GList glades::HiddenLayerInfo::getGTableRow() const
{
	shmea::GList row;
	// structure: size, batchSize, learningRate, momentumFactor, weightDecay1, weightDecay2, pDropout, activationType,
	// activationParam, outputType
	// -1 = "blank"/placeholder
	row.addLong(size());
	row.addLong(1);
	row.addFloat(getLearningRate());
	row.addFloat(getMomentumFactor());
	row.addFloat(getWeightDecay1());
	row.addFloat(getWeightDecay2());
	row.addFloat(getPDropout());
	row.addLong(getActivationType());
	row.addFloat(getActivationParam());
	row.addLong(-1);
	return row;
}

/*!
 * @brief get the layer type
 * @details get the HiddenLayerInfo's layer type
 * @return the HiddenLayerInfo layer type
 */
int glades::HiddenLayerInfo::getLayerType() const
{
	return LayerInfo::HIDDEN;
}
