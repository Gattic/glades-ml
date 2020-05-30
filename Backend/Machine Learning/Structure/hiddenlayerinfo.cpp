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
 * @param newPHidden the dropout rate for the hidden layer
 * @param newActivationType the activation type for the hidden layer
 */
glades::HiddenLayerInfo::HiddenLayerInfo(int newSize, float newLearningRate,
										 float newMomentumFactor, float newWeightDecay,
										 float newPHidden, int newActivationType,
										 float newActivationParam)
	: LayerInfo(newSize)
{
	learningRate = newLearningRate;
	momentumFactor = newMomentumFactor;
	weightDecay = newWeightDecay;
	pHidden = newPHidden;
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
	weightDecay = 0.0f;
	pHidden = 0.0f;
	activationType = 0;
	activationParam = 0;
}

void glades::HiddenLayerInfo::copyParamsFrom(const HiddenLayerInfo* src)
{
	setSize(src->size());
	learningRate = src->getLearningRate();
	momentumFactor = src->getMomentumFactor();
	weightDecay = src->getWeightDecay();
	pHidden = src->getPHidden();
	activationType = src->getActivationType();
}
/*!
 * @brief get learning rate
 * @details get HiddenLayerInfo's learning rate
 * @return the HiddenLayerInfo's learning rate
 */
float glades::HiddenLayerInfo::getLearningRate() const
{
	return learningRate;
}

/*!
 * @brief get GTable row
 * @details gets the GTable row equivalent of this HiddenLayerInfo object
 * @return the GTable row version of a HiddenLayerInfo object
 */
shmea::GList glades::HiddenLayerInfo::getGTableRow() const
{
	shmea::GList row;
	// structure: size, pInput, learningRate, momentumFactor, weightDecay, pHidden, activationType,
	// activationParam, outputType
	// -1 = "blank"/placeholder
	row.addLong(size());
	row.addFloat(-1.0f);
	row.addLong(1);
	row.addFloat(getLearningRate());
	row.addFloat(getMomentumFactor());
	row.addFloat(getWeightDecay());
	row.addFloat(getPHidden());
	row.addLong(getActivationType());
	row.addFloat(getActivationParam());
	row.addLong(-1);
	return row;
}

/*!
 * @brief get momentum factor
 * @details get HiddenLayerInfo's momentum factor
 * @return the HiddenLayerInfo's momentum factor
 */
float glades::HiddenLayerInfo::getMomentumFactor() const
{
	return momentumFactor;
}

/*!
 * @brief get weight decay
 * @details get HiddenLayerInfo's weight decay
 * @return the HiddenLayerInfo's weight decay
 */
float glades::HiddenLayerInfo::getWeightDecay() const
{
	return weightDecay;
}

/*!
 * @brief get hidden layer dropout rate
 * @details get HiddenLayerInfo's hidden layer dropout rate, the probability that training will drop
 * hidden neuron N
 * @return the HiddenLayerInfo's hidden layer dropout rate
 */
float glades::HiddenLayerInfo::getPHidden() const
{
	return pHidden;
}

/*!
 * @brief get hidden layer activation type
 * @details get HiddenLayerInfo's hidden layer activation type
 * @return the HiddenLayerInfo's hidden layer activation type
 */
int glades::HiddenLayerInfo::getActivationType() const
{
	return activationType;
}

/*!
 * @brief get hidden layer activation parameter
 * @details get HiddenLayerInfo's hidden layer activation parameter
 * @return the HiddenLayerInfo's hidden layer activation parameter
 */
float glades::HiddenLayerInfo::getActivationParam() const
{
	return activationParam;
}

/*!
 * @brief set learning rate
 * @details set HiddenLayerInfo's learning rate
 * @param newLearningRate the desired learning rate for this HiddenLayerInfo object
 */
void glades::HiddenLayerInfo::setLearningRate(float newLearningRate)
{
	learningRate = newLearningRate;
}

/*!
 * @brief set momentum factor
 * @details set HiddenLayerInfo's momentum factor
 * @param newMomentumFactor the desired momentum factor for this HiddenLayerInfo object
 */
void glades::HiddenLayerInfo::setMomentumFactor(float newMomentumFactor)
{
	momentumFactor = newMomentumFactor;
}

/*!
 * @brief set weight decay
 * @details set HiddenLayerInfo's weight decay
 * @param newWeightDecay the desired weight decay for this HiddenLayerInfo object
 */
void glades::HiddenLayerInfo::setWeightDecay(float newWeightDecay)
{
	weightDecay = newWeightDecay;
}

/*!
 * @brief set the p hidden
 * @details set the dropout rate for hidden layer index
 * @param newPHidden the desired input dropout rate
 */
void HiddenLayerInfo::setPHidden(float newPHidden)
{
	pHidden = newPHidden;
}

/*!
 * @brief set the activation type
 * @details set the activation type
 * @param newActivationType the desired activation type
 */
void glades::HiddenLayerInfo::setActivationType(int newActivationType)
{
	activationType = newActivationType;
}

/*!
 * @brief set the activation parameter
 * @details set the activation parameter to be used with certain activation types
 * @param newActivationParam the desired activation parameter
 */
void glades::HiddenLayerInfo::setActivationParam(float newActivationParam)
{
	activationParam = newActivationParam;
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
