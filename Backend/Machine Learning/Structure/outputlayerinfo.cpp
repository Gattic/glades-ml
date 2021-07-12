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
#include "outputlayerinfo.h"
#include "Backend/Database/GList.h"
#include "Backend/Database/GType.h"

using namespace glades;

/*!
 * @brief OutputLayerInfo constructor
 * @details create a new OutputLayerInfo object
 * @param newSize the OutputLayerInfo object's desired size
 * @param newOutputType the output type
 */
glades::OutputLayerInfo::OutputLayerInfo(int newSize, int newOutputType)
	: glades::LayerInfo(newSize)
{
	outputType = newOutputType;
}

/*!
 * @brief OutputLayerInfo destructor
 * @details destroy a OutputLayerInfo object
 */
glades::OutputLayerInfo::~OutputLayerInfo()
{
	outputType = REGRESSION;
}

/*!
 * @brief get outpt layer output type
 * @details get OutputLayerInfo's output type (for now, regression or classification)
 * @return the OutputLayerInfo's output type
 */
int glades::OutputLayerInfo::getOutputType() const
{
	return outputType;
}

/*!
 * @brief get GTable row
 * @details get the GTable row from this layer
 * @return the HiddenLayer's GTable row representation
 */
shmea::GList glades::OutputLayerInfo::getGTableRow() const
{
	shmea::GList row;
	// structure: size, pInput, batchSize, learningRate, momentumFactor, weightDecay, pHidden,
	// activationType,
	// activationParam, outputType
	// -1 = "blank"/placeholder
	row.addLong(size());
	row.addFloat(-1.0f);
	row.addLong(1);
	for (int i = 0; i < 4; ++i)
		row.addFloat(-1.0f);
	row.addLong(-1);
	row.addFloat(-1.0f);
	row.addLong(getOutputType());

	return row;
}
/*!
 * @brief set the output layer output type
 * @details set the output type
 * @param newOutputType the desired type
 */
void glades::OutputLayerInfo::setOutputType(int newOutputType)
{
	outputType = newOutputType;
}

/*!
 * @brief set outpt layer's type
 * @details get OutputLayerInfo's type
 * @return the OutputLayerInfo's type
 */
int glades::OutputLayerInfo::getLayerType() const
{
	return LayerInfo::OUTPUT;
}
