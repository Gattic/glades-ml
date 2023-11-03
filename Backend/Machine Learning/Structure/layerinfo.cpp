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
#include "layerinfo.h"

using namespace glades;

/*!
 * @brief LayerInfo constructor
 * @details creates a LayerInfo object
 * @param newSize desired LayerInfo size
 */
glades::LayerInfo::LayerInfo(int newSize)
{
	lSize = newSize;
	learningRate = 0.0f;
	momentumFactor = 0.0f;
	weightDecay = 0.0f;
	pDropout = 0.0f;
	activationType = 0;
	activationParam = 0;
}

/*!
 * @brief LayerInfo destructor
 * @details destroys a LayerInfo object
 * @param newSize desired LayerInfo size
 */
glades::LayerInfo::~LayerInfo()
{
	lSize = 0;
}
/*!
 * @brief get size
 * @details get LayerInfo's size
 * @return the LayerInfo's number of layers
 */
unsigned int glades::LayerInfo::size() const
{
	return lSize;
}

void glades::LayerInfo::copyParamsFrom(const LayerInfo* src)
{
	setSize(src->size());
	learningRate = src->getLearningRate();
	momentumFactor = src->getMomentumFactor();
	weightDecay = src->getWeightDecay();
	pDropout = src->getPDropout();
	activationType = src->getActivationType();
}
/*!
 * @brief get learning rate
 * @details get LayerInfo's learning rate
 * @return the LayerInfo's learning rate
 */
float glades::LayerInfo::getLearningRate() const
{
	return learningRate;
}

/*!
 * @brief get momentum factor
 * @details get LayerInfo's momentum factor
 * @return the LayerInfo's momentum factor
 */
float glades::LayerInfo::getMomentumFactor() const
{
	return momentumFactor;
}

/*!
 * @brief get weight decay
 * @details get LayerInfo's weight decay
 * @return the LayerInfo's weight decay
 */
float glades::LayerInfo::getWeightDecay() const
{
	return weightDecay;
}

/*!
 * @brief get layer dropout rate
 * @details get LayerInfo's layer dropout rate, the probability that training will drop
 * neuron N
 * @return the LayerInfo's layer dropout rate
 */
float glades::LayerInfo::getPDropout() const
{
	return pDropout;
}

/*!
 * @brief get layer activation type
 * @details get LayerInfo's layer activation type
 * @return the LayerInfo's layer activation type
 */
int glades::LayerInfo::getActivationType() const
{
	return activationType;
}

/*!
 * @brief get layer activation parameter
 * @details get LayerInfo's layer activation parameter
 * @return the LayerInfo's layer activation parameter
 */
float glades::LayerInfo::getActivationParam() const
{
	return activationParam;
}

/*!
 * @brief set size
 * @details set LayerInfo's size
 * @param newLearningRate the desired size for this LayerInfo object
 */
void glades::LayerInfo::setSize(int newSize)
{
	lSize = newSize;
}

/*!
 * @brief set learning rate
 * @details set LayerInfo's learning rate
 * @param newLearningRate the desired learning rate for this LayerInfo object
 */
void glades::LayerInfo::setLearningRate(float newLearningRate)
{
	learningRate = newLearningRate;
}

/*!
 * @brief set momentum factor
 * @details set LayerInfo's momentum factor
 * @param newMomentumFactor the desired momentum factor for this LayerInfo object
 */
void glades::LayerInfo::setMomentumFactor(float newMomentumFactor)
{
	momentumFactor = newMomentumFactor;
}

/*!
 * @brief set weight decay
 * @details set LayerInfo's weight decay
 * @param newWeightDecay the desired weight decay for this LayerInfo object
 */
void glades::LayerInfo::setWeightDecay(float newWeightDecay)
{
	weightDecay = newWeightDecay;
}

/*!
 * @brief set the p dropout
 * @details set the dropout rate for layer index
 * @param newPDropout the desired input dropout rate
 */
void LayerInfo::setPDropout(float newPDropout)
{
	pDropout = newPDropout;
}

/*!
 * @brief set the activation type
 * @details set the activation type
 * @param newActivationType the desired activation type
 */
void glades::LayerInfo::setActivationType(int newActivationType)
{
	activationType = newActivationType;
}

/*!
 * @brief set the activation parameter
 * @details set the activation parameter to be used with certain activation types
 * @param newActivationParam the desired activation parameter
 */
void glades::LayerInfo::setActivationParam(float newActivationParam)
{
	activationParam = newActivationParam;
}
