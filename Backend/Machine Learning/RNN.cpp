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
#include "RNN.h"
#include "State/LayerBuilder.h"
#include "State/NetworkState.h"
#include "State/node.h"
#include "Structure/nninfo.h"
#include "network.h"
#include "GMath/cmatrix.h"

/*!
 * @brief RNN constructor
 * @details Contruct a new recurrent model
 */
glades::RNN::RNN()
{
	running = false;
	skeleton = NULL;
	meat = NULL;
	confusionMatrix = NULL;
	serverInstance = NULL;
	cConnection = NULL;
	clean();
	meat = new LayerBuilder(TYPE_RNN);
	confusionMatrix = new glades::CMatrix();
	netType = TYPE_DFF;
	minibatchSize = NNInfo::BATCH_STOCHASTIC;
	netType = TYPE_RNN;
	printf("RNN Conststructor %d\n", netType);
}

/*!
 * @brief RNN destructor
 * @details destroys the RNN object
 */
glades::RNN::~RNN()
{
	//
}

void glades::RNN::beforeFwdEdge(const NetworkState* netState)
{
	//
}

void glades::RNN::beforeFwdNode(const NetworkState* netState)
{
	//netState->cOutputNode->setActivationScalar(meat->getTimeState(
		//netState->cOutputLayerCounter, netState->cOutputNodeCounter, netState->cInputNodeCounter));
}

void glades::RNN::beforeFwdLayer(const NetworkState* netState)
{
	//
}

void glades::RNN::beforeFwd()
{
	//
}

void glades::RNN::beforeBackEdge(const NetworkState* netState)
{
	//
}

void glades::RNN::beforeBackNode(const NetworkState* netState)
{
	//
}

void glades::RNN::beforeBackLayer(const NetworkState* netState)
{
	//
}

void glades::RNN::beforeBack()
{
	//
}

void glades::RNN::afterFwdEdge(const NetworkState* netState)
{
	//
}

void glades::RNN::afterFwdNode(const NetworkState* netState, float cOutputNodeActivation)
{
	/*meat->setTimeState(netState->cOutputLayerCounter, netState->cOutputNodeCounter,
					   netState->cOutputNode->getActivation());*/
}

void glades::RNN::afterFwdLayer(const NetworkState* netState, float cOutputLayerActivation)
{
	//
}

void glades::RNN::afterFwd()
{
	//
}

void glades::RNN::afterBackEdge(const NetworkState* netState)
{
	//
}

void glades::RNN::afterBackNode(const NetworkState* netState)
{
	//
}

void glades::RNN::afterBackLayer(const NetworkState* netState)
{
	//
}

void glades::RNN::afterBack()
{
	//
}
