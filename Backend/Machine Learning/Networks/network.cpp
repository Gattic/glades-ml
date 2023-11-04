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
#include "network.h"
#include "Backend/Database/GList.h"
#include "Backend/Database/GTable.h"
#include "Backend/Database/GType.h"
#include "Backend/Database/ServiceData.h"
#include "Backend/Networking/main.h"
#include "../GMath/OHE.h"
#include "../GMath/cmatrix.h"
#include "../GMath/gmath.h"
#include "../State/LayerBuilder.h"
#include "../State/NetworkState.h"
#include "../State/Terminator.h"
#include "../State/edge.h"
#include "../State/layer.h"
#include "../State/node.h"
#include "../Structure/nninfo.h"
#include "../DataObjects/NumberInput.h"
#include "../DataObjects/ImageInput.h"

using namespace glades;

// for stopping ml  training instances

/*!
 * @brief NNetwork constructor
 * @details creates an empty nnetwork
 */
glades::NNetwork::NNetwork()
{
	running = false;
	di = NULL;
	skeleton = NULL;
	meat = NULL;
	confusionMatrix = NULL;
	serverInstance = NULL;
	cConnection = NULL;
	clean();
	meat = new LayerBuilder();
	confusionMatrix = new glades::CMatrix();
	netType = TYPE_DFF;
	minibatchSize = NNInfo::BATCH_STOCHASTIC;
}

/*!
 * @brief NNetwork destructor
 * @details destroys the NNetwork object
 */
glades::NNetwork::NNetwork(NNInfo* newNNInfo)
{
	if (!newNNInfo)
		return;

	running = false;
	di = NULL;
	skeleton = NULL;
	meat = NULL;
	confusionMatrix = NULL;
	serverInstance = NULL;
	cConnection = NULL;
	clean();
	meat = new LayerBuilder();
	confusionMatrix = new glades::CMatrix();
	skeleton = newNNInfo;
	netType = TYPE_DFF;
	minibatchSize = skeleton->getBatchSize();
}

glades::NNetwork::~NNetwork()
{
	clean();
	resetGraphs();
}

bool glades::NNetwork::getRunning() const
{
	return running;
}

int glades::NNetwork::getEpochs() const
{
	return epochs;
}

void glades::NNetwork::stop()
{
	running = false;
}

void glades::NNetwork::run(DataInput* newDataInput, const Terminator* Arnold, int runType)
{
	if (!skeleton)
		return;

	if(!newDataInput)
		return;

	di = newDataInput;
	if ((di->getTrainSize() <= 0) || (di->getFeatureCount() <= 0))
		return;

	// Get the input, expected, and layers/nodes/edges
	if(epochs == 0)
		meat->build(skeleton, di, netType);

	// inputTable.print();
	// expected.print();

	// Set the mini batch size
	minibatchSize = skeleton->getBatchSize();
	// Valid layers?
	if ((meat->getInputLayersSize() <= 0) || (meat->getLayersSize() <= 0))
		return;

	if ((di->getTrainSize() <= 0) || (di->getFeatureCount() <= 0))
		return;

	// Build empty confusion matrix
	if ((skeleton->getOutputType() == GMath::CLASSIFICATION) ||
		(skeleton->getOutputType() == GMath::KL))
		confusionMatrix->build(skeleton->getOutputLayerSize());

	// if (DEBUG_ADVANCED)
	//	meat->print(skeleton);

	if (runType == RUN_TRAIN)
		printf("[NN] Training...\n");
	else if (runType == RUN_TEST)
		printf("[NN] Testing...\n");

	// debugging topbar
	if (runType == RUN_TRAIN)
	{
		if (skeleton->getOutputType() == GMath::REGRESSION)
			printf("[NN] Epochs\tAccuracy\n");

		if ((skeleton->getOutputType() == GMath::CLASSIFICATION) ||
			(skeleton->getOutputType() == GMath::KL))
			printf("[NN] Epochs\tAccuracy\tPrecision\tRecall\t\tSpecificity\tF1 Score\n");
	}

	// Reset the different graphcs e.g. learning curve
	resetGraphs();

	// arbitrary independent var (time dimension)
	running = true;
	while (running)
	{
		/*if (DEBUG_ADVANCED)
			printf("[NN] Expected\tPredicted\tError\t\tError^2\t\tAccuracy\tPrecise\n");*/

		// Global network statistics
		overallTotalError = 0.0f;
		overallTotalAccuracy = 0.0f;

		// Reset confusion matrix
		if ((skeleton->getOutputType() == GMath::CLASSIFICATION) ||
			(skeleton->getOutputType() == GMath::KL))
			confusionMatrix->reset();

		// Recursive FwdPass/BackProp
		for (unsigned int r = 0; r < meat->getInputLayersSize(); ++r)
			SGDHelper(r, runType);

		// Update the network vars
		++epochs;
		overallTotalAccuracy /=
			((float)meat->getInputLayersSize()) * ((float)skeleton->getOutputLayerSize());

		if (skeleton->getOutputType() == GMath::REGRESSION)
		{
			// Display and debugging
			if (runType == RUN_TRAIN)
			{
				// if (DEBUG_SIMPLE)
				{
					printf("\33[2K[NN] %d\t%f%%\r", epochs, overallTotalAccuracy);
					fflush(stdout);
				}
				/*else if (DEBUG_ADVANCED)
				{
					printf("\n[NN] Epochs\tAccuracy\n");
					printf("%d\t%f%%\n", epochs, overallTotalAccuracy);
					printf("===========================================================\n\n");
				}*/
			}
		}
		else if ((skeleton->getOutputType() == GMath::CLASSIFICATION) ||
				 (skeleton->getOutputType() == GMath::KL))
		{
			confusionMatrix->updateResultParams();

			overallClassAccuracy = (confusionMatrix->getOverallAccuracy() * 100.0f);
			overallClassPrecision = (confusionMatrix->getOverallPrecision() * 100.0f);
			overallClassRecall = (confusionMatrix->getOverallRecall() * 100.0f);
			overallClassSpecificity = (confusionMatrix->getOverallSpecificity() * 100.0f);
			overallClassF1 = confusionMatrix->getOverallF1Score() * 100.0f;

			// Display and debugging
			if (runType == RUN_TRAIN)
			{
				// if (DEBUG_SIMPLE)
				{
					if (epochs < 100)
					{
						printf("\33[2K[NN] %d\t\t%f%%\t%f%%\t%f%%\t%f%%\t%f%%\r", epochs,
							   overallClassAccuracy, overallClassPrecision, overallClassRecall,
							   overallClassSpecificity, overallClassF1);
						fflush(stdout);
					}
					else
					{
						printf("\33[2K[NN] %d\t%f%%\t%f%%\t%f%%\t%f%%\t%f%%\r", epochs,
							   overallClassAccuracy, overallClassPrecision, overallClassRecall,
							   overallClassSpecificity, overallClassF1);
						fflush(stdout);
					}
				}
				/*else if (DEBUG_ADVANCED)
				{
					printf("\n[NN] Epochs\tAccuracy\tPrecision\tRecall\t\tSpecificity\tF1 Score\n");
					printf("[NN] \t%d\t%f%%\t%f%%\t%f%%\t%f%%\t%f%%\n", epochs,
						   overallClassAccuracy, overallClassPrecision, overallClassRecall,
						   overallClassSpecificity, overallClassF1);
					confusionMatrix->print();
					printf("===========================================================\n\n");
				}*/
			}
		}

		if (runType == RUN_TRAIN)
		{
			// Update the GUI with the metrics
			//if(guiEnabled)
			if(serverInstance && cConnection)
			{
				// First epoch is random
				if (epochs > 0)
				{
					// Update the learning curve
					shmea::GList wData;
					wData.addInt(epochs-1); // -1 because we dont plot the first point
					wData.addFloat(overallTotalError);

					shmea::GList argData;
					argData.addString("PROGRESSIVE");

					shmea::ServiceData* cData = new shmea::ServiceData(cConnection, "GUI_Callback");
					cData->set(wData);
					cData->setArgList(argData);
					serverInstance->send(cData);
				}

				if (skeleton->getOutputType() == GMath::REGRESSION)
				{
					shmea::GList argData;
					argData.addString("ACC");

					shmea::GList wData;
					wData.addInt(epochs);
					wData.addFloat(overallTotalAccuracy);

					// Update the Accuracy label
					shmea::ServiceData* cData = new shmea::ServiceData(cConnection, "GUI_Callback");
					cData->set(wData);
					cData->setArgList(argData);
					serverInstance->send(cData);
				}
				else if ((skeleton->getOutputType() == GMath::CLASSIFICATION) ||
					(skeleton->getOutputType() == GMath::KL))
				{
					// Update the ROC Curve and Conf Matrix
					shmea::GList argData;
					argData.addString("CONF");
					argData.addFloat(confusionMatrix->getOverallFalseAlarm());
					argData.addFloat(confusionMatrix->getOverallRecall());

					shmea::ServiceData* cData = new shmea::ServiceData(cConnection, "GUI_Callback");
					cData->set(confusionMatrix->getMatrix());
					cData->setArgList(argData);
					serverInstance->send(cData);
				}
			}
		}

		// Update the scatter plot graph
		/*if ((Frontend::simulationPanel) && (Frontend::simulationPanel->isFocused()))
			Frontend::simulationPanel->PlotScatter(getResults());
		if ((Frontend::nnCreatorPanel) && (Frontend::nnCreatorPanel->isFocused()))
			Frontend::nnCreatorPanel->PlotScatter(getResults());*/

		// Shut it down?
		if ((Arnold) && (Arnold->triggered(time(NULL), epochs, overallTotalAccuracy)))
			break;

		// Shut it down?
		if (runType == RUN_TEST)
			break;
	}

	// For the carriage controlled print
	if (runType == RUN_TRAIN)
		printf("\n");
	else if (runType == RUN_TEST)
	{
		// Update the network vars and print
		printf("[NN] %s Accuracy: %f%%\n", skeleton->getName().c_str(), overallTotalAccuracy);
	}

	// Print the results
	meat->print(skeleton);

	// Clean confusion matrix
	if ((skeleton->getOutputType() == GMath::CLASSIFICATION) ||
		(skeleton->getOutputType() == GMath::KL))
		confusionMatrix->clean();

	printf("\n");

	// So the network doesnt immediately quit next time and we can prematurely start our net
	running = false;
}

void glades::NNetwork::SGDHelper(unsigned int inputRowCounter, int runType)
{
	if ((!skeleton) || (!meat))
		return;

	if (meat->getLayersSize() <= 0)
		return;

	// New Dropout probabilities
	std::vector<float> pHiddenVec;
	for (int i = 0; i < skeleton->numHiddenLayers(); ++i)
		pHiddenVec.push_back(skeleton->getPDropout(i));
	meat->scrambleDropout(inputRowCounter, skeleton->getPInput(), pHiddenVec);

	// Reset the results
	results.clear();
	nbRecord.clear();

	// Forward Pass and trigger events
	ForwardPass(inputRowCounter, 0, 1, 0, 0);

	// Add current results to cmatrix for accuracy vars
	if ((skeleton->getOutputType() == GMath::CLASSIFICATION) ||
		(skeleton->getOutputType() == GMath::KL))
		confusionMatrix->addResult(results);

	//printf("-------------------------------\n");

	// Back Propagation and trigger events
	if (runType == RUN_TRAIN)
	{
		// Start with the last output layer
		int cOutputLayerCounter = meat->getLayersSize();
		BackPropagation(inputRowCounter, cOutputLayerCounter - 1, cOutputLayerCounter, 0, 0);

		// Save the autotuning data
		float learningRate = skeleton->getLearningRate(cOutputLayerCounter - 1);
		shmea::GList nbRow;
		nbRow.addFloat(overallTotalAccuracy);
		nbRow.addFloat(learningRate);
		nbRecord.addRow(nbRow);

		// Create a bayes net
		/*shmea::GTable bTable = bModel.import(nbRecord);
		//bTable.print();
		bModel.train(bTable);

		// predict with a new learning rate
		shmea::GList testList;
		testList.addFloat(0.0f); // we want 0 error
		float newLearningRate = bModel.predict(testList);*/
		//printf("Predicted learning rate %f\n", newLearningRate);
		//skeleton->setLearningRate(cOutputLayerCounter - 1, newLearningRate);

		// Clear past dropout state
		meat->clearDropout();
	}

	//printf("=========================================\n");
}

void glades::NNetwork::ForwardPass(unsigned int inputRowCounter,
		int cInputLayerCounter, int cOutputLayerCounter,
		unsigned int cInputNodeCounter, unsigned int cOutputNodeCounter)
{
	NetworkState* netState =
		meat->getNetworkStateFromLoc(inputRowCounter, cInputLayerCounter, cOutputLayerCounter,
									 cInputNodeCounter, cOutputNodeCounter);
	if (!netState)
		return;

	//printf("ForwardPass: %d %d %d %d %d\n", inputRowCounter, cInputLayerCounter, cOutputLayerCounter,
		   //cInputNodeCounter, cOutputNodeCounter);

	// Does Dropout occur?
	bool dropout = (!((netState->validInputNode) && (netState->validOutputNode)));

	// Add the input node activation to the output node
	if (!dropout)
	{
		float cEdgeActivation = netState->cOutputNode->getEdgeWeight(cInputNodeCounter) *
								netState->cInputNode->getWeight();
		netState->cOutputNode->setActivation(cInputNodeCounter, cEdgeActivation);
	}

	// Last Input Node for the Output Node
	float cOutputLayerActivation = 0.0f;
	if (netState->lastValidInputNode)
	{
		// Get the current output node activation
		float cOutputNodeActivation = netState->cOutputNode->getActivation();

		// Clean the output node activation for next run (cleanup)
		netState->cOutputNode->clearActivation();

		// Add the bias if we are in a hidden layer or output layer
		// Input Layer fundamentally cannot have a bias
		if (netState->cInputLayer->getType() != Layer::INPUT_TYPE)
			cOutputNodeActivation += netState->cInputLayer->getBiasWeight();

		// Set Our prediction based on the cOutputNode activation
		int cActivationFx = skeleton->getActivationType(cInputLayerCounter);
		float cActivationParam = skeleton->getActivationParam(cInputLayerCounter);
		cOutputLayerActivation =
			GMath::squash(cOutputNodeActivation, cActivationFx, cActivationParam);
		netState->cOutputNode->setWeight(cOutputLayerActivation);

		// Output layer calculations
		if (netState->cOutputLayer->getType() == Layer::OUTPUT_TYPE)
		{
			// Get the prediction and expected vars
			float prediction = netState->cOutputNode->getWeight();
			shmea::GType expectedCell =
				di->getTrainExpectedRow(inputRowCounter)[cOutputNodeCounter];
			float expectation = expectedCell;

			// Add the expected and predicted to the result row
			results.addFloat(expectation);
			results.addFloat(prediction);

			// Cost function calculations
			float dataSize = (float)(di->getTrainSize() * netState->cOutputLayer->size());
			int costFx = skeleton->getOutputType();
			float cOutputCost = GMath::outputNodeCost(expectation, prediction, dataSize, costFx);

			// Error across every input instance
			overallTotalError += cOutputCost;

			// Accuracy vars
			float percentError = GMath::PercentError(prediction, expectation, cOutputCost);
			float calculatedError = GMath::error(expectation, prediction);
			bool isCorrect = percentError < GMath::OUTLIER;
			float accuracy = (1.0f - percentError) * 100.0f;
			if (accuracy < 0.0f)
				accuracy = 0.0f;
			overallTotalAccuracy += accuracy;

			// Advanced Debugging
			/*if (DEBUG_ADVANCED)
			{
				printf("%f\t%f\t%f\t%f\t%f%%\t(%s)\n", expectation, prediction, calculatedError,
					   cOutputCost, (1.0f - percentError) * 100.0f, isCorrect ? "True" : "False");

				// Multiple output nodes
				if (netState->cOutputLayer->size() > 1)
					printf("-----------------------------------------------------------\n");
			}*/
		}
	}

	// Recursive Calls
	if ((cInputNodeCounter == netState->cInputLayer->size() - 1) &&
		(cOutputNodeCounter == netState->cOutputLayer->size() - 1))
	{
		// Next Output Layer
		ForwardPass(inputRowCounter, cOutputLayerCounter, cOutputLayerCounter + 1, 0,
					0);
	}
	else if (cInputNodeCounter == netState->cInputLayer->size() - 1)
	{
		// Next Output Node
		ForwardPass(inputRowCounter, cInputLayerCounter, cOutputLayerCounter, 0,
					cOutputNodeCounter + 1);
	}
	else
	{
		// Next Input Node
		ForwardPass(inputRowCounter, cInputLayerCounter, cOutputLayerCounter,
					cInputNodeCounter + 1, cOutputNodeCounter);
	}

	delete netState;
}

void glades::NNetwork::BackPropagation(unsigned int inputRowCounter, int cInputLayerCounter,
									   int cOutputLayerCounter, unsigned int cInputNodeCounter,
									   unsigned int cOutputNodeCounter)
{
	NetworkState* netState =
		meat->getNetworkStateFromLoc(inputRowCounter, cInputLayerCounter, cOutputLayerCounter,
									 cInputNodeCounter, cOutputNodeCounter);
	if (!netState)
		return;

	//printf("BackPropagation: %d %d %d %d %d\n", inputRowCounter, cInputLayerCounter,
		   //cOutputLayerCounter, cInputNodeCounter, cOutputNodeCounter);

	// Output Layer Error Derivative Calculation
	float cOutputDer = 1.0f; // Output der is linear so its 1
	if (netState->cOutputLayer->getType() == Layer::OUTPUT_TYPE)
	{
		// Cost function error derivative for output layer(s)
		float prediction = netState->cOutputNode->getWeight();
		float expectation = 0.0f;
		shmea::GType expectedCell =
			di->getTrainExpectedRow(inputRowCounter)[cOutputNodeCounter];
		expectation = expectedCell.getFloat();

		int costFx = skeleton->getOutputType();
		netState->cOutputNode->setErrDer(0, GMath::costErrDer(expectation, prediction, costFx));
	}
	else if (netState->cOutputLayer->getType() == Layer::HIDDEN_TYPE)
	{
		// Activation error derivative
		int cActivationFx = skeleton->getActivationType(cInputLayerCounter);
		cOutputDer =
			GMath::activationErrDer(netState->cOutputNode->getWeight(), cActivationFx, 0.01f);
	}

	// Does Dropout occur?
	bool dropout = (!((netState->validInputNode) && (netState->validOutputNode)));

	// Node error derivative
	float cOutNetErrDer = netState->cOutputNode->getErrDer();
	if (!dropout)
	{
		cOutNetErrDer *= cOutputDer; // current error partial

		// Clean the output node err der (cleanup)
		if (cInputNodeCounter == netState->cInputLayer->size() - 1)
			netState->cOutputNode->clearErrDer();

		// MSE applied through gradient descent
		float learningRate = skeleton->getLearningRate(cInputLayerCounter);
		float momentumFactor = skeleton->getMomentumFactor(cInputLayerCounter);
		float weightDecay = skeleton->getWeightDecay(cInputLayerCounter);
		float baseError = learningRate * cOutNetErrDer;

		// Add the weight delta
		netState->cOutputNode->getDelta(cInputNodeCounter, baseError,
										netState->cInputNode->getWeight(), learningRate,
										momentumFactor, weightDecay);

		// Apply all deltas if we've hit the minibatch size
		if ((inputRowCounter % minibatchSize) == 0)
		{
			netState->cOutputNode->applyDeltas(cInputNodeCounter, minibatchSize);
			netState->cOutputNode->clearPrevDeltas(cInputNodeCounter);
		}

		// Update the bias (inputs fundamentally cannot have a bias)
		if (netState->cInputLayer->getType() != Layer::INPUT_TYPE)
			netState->cInputLayer->setBiasWeight(netState->cInputLayer->getBiasWeight() -
												 baseError);
	}

	// Update the error partials for the next recursive calls
	float cInNetErrDer = netState->cInputNode->getErrDer() +
						 (cOutNetErrDer * netState->cOutputNode->getEdgeWeight(cInputNodeCounter));
	netState->cInputNode->setErrDer(cOutputNodeCounter, cInNetErrDer);

	// Recursive Calls
	/*if ((cInputNodeCounter == netState->cInputLayer->size() - 1) &&
		(cOutputNodeCounter == netState->cOutputLayer->size() - 1))
	{
		// Next Input Layer
		if ((cInputLayerCounter == 0) && (cOutputLayerCounter == 1)) // Last recursive layer case
			BackPropagation(inputRowCounter, 0, 0, 0, 0);
		else
			BackPropagation(inputRowCounter, cInputLayerCounter - 1, cInputLayerCounter, 0, 0);
	}
	else if (cOutputNodeCounter == netState->cOutputLayer->size() - 1)
	{
		// Next Input Node
		BackPropagation(inputRowCounter, cInputLayerCounter, cOutputLayerCounter,
						cInputNodeCounter + 1, 0);
	}
	else
	{
		// Next Output Node
		BackPropagation(inputRowCounter, cInputLayerCounter, cOutputLayerCounter, cInputNodeCounter,
						cOutputNodeCounter + 1);
	}*/

	if (cOutputNodeCounter < netState->cOutputLayer->size() - 1)
	{
		// Next Output Node
		BackPropagation(inputRowCounter, cInputLayerCounter, cOutputLayerCounter, cInputNodeCounter,
						cOutputNodeCounter + 1);
	}
	else if (cInputNodeCounter < netState->cInputLayer->size() - 1)
	{
		// Next Input Node
		BackPropagation(inputRowCounter, cInputLayerCounter, cOutputLayerCounter,
						cInputNodeCounter + 1, 0);
	}
	else if(cOutputLayerCounter > 1)
	{
		// Next Input Layer
		BackPropagation(inputRowCounter, cInputLayerCounter - 1, cInputLayerCounter, 0, 0);
	}

	delete netState;
}

/*!
 * @brief get ID
 * @details get the network's unique ID
 * @return the network's ID
 */
int64_t glades::NNetwork::getID() const
{
	return id;
}

shmea::GString glades::NNetwork::getName() const
{
	if (!skeleton)
		return "";

	return skeleton->getName();
}

NNInfo* glades::NNetwork::getNNInfo()
{
	if (!skeleton)
		return NULL;

	return skeleton;
}

float glades::NNetwork::getAccuracy() const
{
	return overallTotalAccuracy;
}

bool glades::NNetwork::load(const shmea::GString& netName)
{
	if (!meat)
		return false;

	if (!skeleton)
		skeleton = new NNInfo(netName);

	// Load the nn state information
	if (!skeleton->load(netName))
		return false;

	//return meat->load(netName);
	return true;
}

bool glades::NNetwork::save() const
{
	if (!meat)
		return false;

	if (!skeleton)
		return false;

	skeleton->save();
	return true;
	/// return meat->save(skeleton->getName());
}

void glades::NNetwork::setServer(GNet::GServer* newServer, GNet::Connection* newConnection)
{
	serverInstance = newServer;
	cConnection = newConnection;
}

shmea::GList glades::NNetwork::getLearningCurve() const
{
	return learningCurve;
}

// const std::vector<Point2*>& NNetwork::getROCCurve() const
// {
// 	return rocCurve;
// }

shmea::GList glades::NNetwork::getResults() const
{
	return results;
}

void glades::NNetwork::clean()
{
	id = -1;
	skeleton = NULL;
	meat = NULL;
	confusionMatrix = NULL;
	serverInstance = NULL;
	cConnection = NULL;
	results.clear();
	nbRecord.clear();
	epochs = 0;
	overallTotalError = 0.0f;
	overallTotalAccuracy = 0.0f;
	overallClassAccuracy = 0.0f;
	overallClassPrecision = 0.0f;
	overallClassRecall = 0.0f;
	overallClassF1 = 0.0f;
	minibatchSize = NNInfo::BATCH_STOCHASTIC;
}

void glades::NNetwork::resetGraphs()
{
	learningCurve.clear();

	/*for (unsigned int i = 0; i < rocCurve.size(); ++i)
		delete rocCurve[i];*/
	rocCurve.clear();
	rocCurve.reserve(10000); // arbitrary number

	// create the results again
	results.clear();
}
