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

/*!
 * @brief set size
 * @details set LayerInfo's size
 * @param newLearningRate the desired size for this LayerInfo object
 */
void glades::LayerInfo::setSize(int newSize)
{
	lSize = newSize;
}
