// Copyright 2026 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
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
#include "main.h"
#include "Backend/Database/GTable.h"
#include "Backend/Database/GType.h"
#include "Backend/Networking/main.h"
#include "GMath/gmath.h"
#include "Structure/nninfo.h"
#include "Networks/metanetwork.h"
#include "Networks/network.h"
#include "Networks/training_callbacks.h"
#include "DataObjects/ImageInput.h"

using namespace glades;

bool glades::doesDatabaseExist()
{
	struct stat buffer;
	return stat("database", &buffer) == 0 && S_ISDIR(buffer.st_mode);
}

void glades::createDatabase()
{
#if defined(_WIN32)
	_mkdir("database");
#else
	mkdir("database", 0777);
#endif
}

/*!
 * @brief initialize glades
 * @details initialize the neural network mutex
 */
void glades::init()
{
	// initialize the mutex
	if (!doesDatabaseExist())
		createDatabase();
}

bool glades::saveNeuralNetwork(glades::NNetwork* newNet)
{
	if (!newNet)
		return false;

	// Save the neural network
	const glades::NNetworkStatus st = newNet->saveModel(std::string(newNet->getName().c_str()));
	if (!st.ok())
	{
		char buffer[256];
		sprintf(buffer, "Unable to save \"%s\"", newNet->getName().c_str());
		puts(buffer);
		return false;
	}

	return true;
}

/*!
 * @brief train network
 * @details train a neural network
 * @param networkInfo the incoming or desired neural net info
 * @param newDataInput the data to use in training
 * @return the trained MetaNetwork object
 */
glades::MetaNetwork* glades::train(NNInfo* networkInfo, DataInput* newDataInput, GNet::GServer* serverInstance, GNet::Connection* cConnection)
{
	if (!networkInfo)
		return NULL;

	// metanetwork for aggregation
	glades::MetaNetwork* cMetaNetwork = new glades::MetaNetwork(networkInfo->getName());

	// Add the Neural Network
	cMetaNetwork->addSubnet(networkInfo);

	// Train the Neural Network
	std::vector<glades::NNetwork*> subnets = cMetaNetwork->getSubnets();
	for (unsigned int i = 0; i < subnets.size(); ++i)
	{
		if (serverInstance && cConnection)
			subnets[i]->setServer(serverInstance, cConnection);
		const glades::NNetworkStatus st = subnets[i]->train(newDataInput);
		if (!st.ok())
		{
			printf("[NN] Train failed: %s\n", st.message.c_str());
			delete cMetaNetwork;
			return NULL;
		}
	}

	return cMetaNetwork;
}

/*!
 * @brief train network
 * @details train a neural network
 * @param networkInfo the incoming or desired neural net info
 * @param newDataInput the data to use in training
 * @return the trained MetaNetwork object
 */
glades::MetaNetwork* glades::train(glades::NNetwork* cNetwork, DataInput* newDataInput,
    GNet::GServer* serverInstance, GNet::Connection* cConnection)
{
	if (!cNetwork)
		return NULL;

	// metanetwork for aggregation
	glades::MetaNetwork* cMetaNetwork = new glades::MetaNetwork(cNetwork->getName());

	// Add the Neural Network
	cMetaNetwork->addSubnet(cNetwork);

	// Train the Neural Network
	std::vector<glades::NNetwork*> subnets = cMetaNetwork->getSubnets();
	for (unsigned int i = 0; i < subnets.size(); ++i)
	{
		if (serverInstance && cConnection)
			subnets[i]->setServer(serverInstance, cConnection);
		const glades::NNetworkStatus st = subnets[i]->train(newDataInput);
		if (!st.ok())
		{
			printf("[NN] Train failed: %s\n", st.message.c_str());
			delete cMetaNetwork;
			return NULL;
		}
	}

	return cMetaNetwork;
}

/*!
 * @brief train a metanetwork
 * @details train a set of neural networks
 * @param networkInfo the incoming or desired neural net info
 * @param newDataInput the data to use in training
 * @return the trained MetaNetwork object
 */
glades::MetaNetwork* glades::train(glades::MetaNetwork* cMetaNetwork,
    DataInput* newDataInput, GNet::GServer* serverInstance, GNet::Connection* cConnection)
{
	if (!cMetaNetwork)
		return NULL;

	// Train the Neural Network
	std::vector<glades::NNetwork*> subnets = cMetaNetwork->getSubnets();
	for (unsigned int i = 0; i < subnets.size(); ++i)
	{
		if (serverInstance && cConnection)
			subnets[i]->setServer(serverInstance, cConnection);
		const glades::NNetworkStatus st = subnets[i]->train(newDataInput);
		if (!st.ok())
		{
			printf("[NN] Train failed: %s\n", st.message.c_str());
			return NULL;
		}
	}

	return cMetaNetwork;
}

/*!
 * @brief test network
 * @details test a network
 * @param networkInfo the incoming network's relevant information
 * @param newDataInput the data to use in testing
 * @return the tested MetaNetwork object
 */
glades::MetaNetwork* glades::test(NNInfo* networkInfo, DataInput* newDataInput, GNet::GServer* serverInstance, GNet::Connection* cConnection)
{
	if (!networkInfo)
		return NULL;

	// metanetwork for aggregation
	glades::MetaNetwork* cMetaNetwork = new glades::MetaNetwork(networkInfo->getName());

	// Add the Neural Network
	cMetaNetwork->addSubnet(networkInfo);

	// Test the Neural Network
	std::vector<glades::NNetwork*> subnets = cMetaNetwork->getSubnets();
	for (unsigned int i = 0; i < subnets.size(); ++i)
	{
		if (serverInstance && cConnection)
			subnets[i]->setServer(serverInstance, cConnection);
		const glades::NNetworkStatus st = subnets[i]->test(newDataInput);
		if (!st.ok())
		{
			printf("[NN] Test failed: %s\n", st.message.c_str());
			delete cMetaNetwork;
			return NULL;
		}
	}

	return cMetaNetwork;
}

/*!
 * @brief test network
 * @details test a network
 * @param networkInfo the incoming network's relevant information
 * @param newDataInput the data to use in testing
 * @return the tested MetaNetwork object
 */
glades::MetaNetwork* glades::test(glades::NNetwork* networkInfo, DataInput* newDataInput, GNet::GServer* serverInstance, GNet::Connection* cConnection)
{
	if (!networkInfo)
		return NULL;

	// metanetwork for aggregation
	glades::MetaNetwork* cMetaNetwork = new glades::MetaNetwork(networkInfo->getName());

	// Add the Neural Network
	cMetaNetwork->addSubnet(networkInfo);

	// Test the Neural Network
	std::vector<glades::NNetwork*> subnets = cMetaNetwork->getSubnets();
	for (unsigned int i = 0; i < subnets.size(); ++i)
	{
		if (serverInstance && cConnection)
			subnets[i]->setServer(serverInstance, cConnection);
		const glades::NNetworkStatus st = subnets[i]->test(newDataInput);
		if (!st.ok())
		{
			printf("[NN] Test failed: %s\n", st.message.c_str());
			delete cMetaNetwork;
			return NULL;
		}
	}

	return cMetaNetwork;
}

/*!
 * @brief test a metanetwork
 * @details test a set of neural networks
 * @param networkInfo the incoming or desired neural net info
 * @param newDataInput the data to use in testing
 * @return the tested MetaNetwork object
 */
glades::MetaNetwork* glades::test(glades::MetaNetwork* cMetaNetwork, DataInput* newDataInput, GNet::GServer* serverInstance, GNet::Connection* cConnection)
{
	if (!cMetaNetwork)
		return NULL;

	// Test the Neural Network
	std::vector<glades::NNetwork*> subnets = cMetaNetwork->getSubnets();
	for (unsigned int i = 0; i < subnets.size(); ++i)
	{
		if (serverInstance && cConnection)
			subnets[i]->setServer(serverInstance, cConnection);
		const glades::NNetworkStatus st = subnets[i]->test(newDataInput);
		if (!st.ok())
		{
			printf("[NN] Test failed: %s\n", st.message.c_str());
			return NULL;
		}
	}

	return cMetaNetwork;
}

// ===== Callback-aware wrappers =====
//
// These pass an explicit ITrainingCallbacks* to the network, bypassing the built-in
// default callbacks. Useful when the caller provides its own GUI/logging adapter.

glades::MetaNetwork* glades::train(glades::NNetwork* cNetwork, DataInput* newDataInput,
    ITrainingCallbacks* callbacks, GNet::GServer* serverInstance, GNet::Connection* cConnection)
{
	if (!cNetwork)
		return NULL;

	glades::MetaNetwork* cMetaNetwork = new glades::MetaNetwork(cNetwork->getName());
	cMetaNetwork->addSubnet(cNetwork);

	std::vector<glades::NNetwork*> subnets = cMetaNetwork->getSubnets();
	for (unsigned int i = 0; i < subnets.size(); ++i)
	{
		if (serverInstance && cConnection)
			subnets[i]->setServer(serverInstance, cConnection);
		const glades::NNetworkStatus st = subnets[i]->train(newDataInput, callbacks);
		if (!st.ok())
		{
			printf("[NN] Train failed: %s\n", st.message.c_str());
			delete cMetaNetwork;
			return NULL;
		}
	}

	return cMetaNetwork;
}

glades::MetaNetwork* glades::test(glades::NNetwork* cNetwork, DataInput* newDataInput,
    ITrainingCallbacks* callbacks, GNet::GServer* serverInstance, GNet::Connection* cConnection)
{
	if (!cNetwork)
		return NULL;

	glades::MetaNetwork* cMetaNetwork = new glades::MetaNetwork(cNetwork->getName());
	cMetaNetwork->addSubnet(cNetwork);

	std::vector<glades::NNetwork*> subnets = cMetaNetwork->getSubnets();
	for (unsigned int i = 0; i < subnets.size(); ++i)
	{
		if (serverInstance && cConnection)
			subnets[i]->setServer(serverInstance, cConnection);
		const glades::NNetworkStatus st = subnets[i]->test(newDataInput, callbacks);
		if (!st.ok())
		{
			printf("[NN] Test failed: %s\n", st.message.c_str());
			delete cMetaNetwork;
			return NULL;
		}
	}

	return cMetaNetwork;
}

// ===== Safer ownership wrappers (RAII) =====
//
// These functions wrap the legacy raw-pointer API and return a ref-counted `GPointer`.
// This avoids forcing callers to remember to `delete` the returned MetaNetwork.

shmea::GPointer<glades::MetaNetwork> glades::trainOwned(glades::NNInfo* networkInfo,
                                                       glades::DataInput* newDataInput,
                                                       GNet::GServer* serverInstance,
                                                       GNet::Connection* cConnection)
{
	return shmea::GPointer<glades::MetaNetwork>(glades::train(networkInfo, newDataInput, serverInstance, cConnection));
}

shmea::GPointer<glades::MetaNetwork> glades::trainOwned(glades::NNetwork* cNetwork,
                                                       glades::DataInput* newDataInput,
                                                       GNet::GServer* serverInstance,
                                                       GNet::Connection* cConnection)
{
	return shmea::GPointer<glades::MetaNetwork>(glades::train(cNetwork, newDataInput, serverInstance, cConnection));
}

shmea::GPointer<glades::MetaNetwork> glades::trainOwned(glades::MetaNetwork* cMetaNetwork,
                                                       glades::DataInput* newDataInput,
                                                       GNet::GServer* serverInstance,
                                                       GNet::Connection* cConnection)
{
	return shmea::GPointer<glades::MetaNetwork>(glades::train(cMetaNetwork, newDataInput, serverInstance, cConnection));
}

shmea::GPointer<glades::MetaNetwork> glades::testOwned(glades::NNInfo* networkInfo,
                                                      glades::DataInput* newDataInput,
                                                      GNet::GServer* serverInstance,
                                                      GNet::Connection* cConnection)
{
	return shmea::GPointer<glades::MetaNetwork>(glades::test(networkInfo, newDataInput, serverInstance, cConnection));
}

shmea::GPointer<glades::MetaNetwork> glades::testOwned(glades::NNetwork* networkInfo,
                                                      glades::DataInput* newDataInput,
                                                      GNet::GServer* serverInstance,
                                                      GNet::Connection* cConnection)
{
	return shmea::GPointer<glades::MetaNetwork>(glades::test(networkInfo, newDataInput, serverInstance, cConnection));
}

shmea::GPointer<glades::MetaNetwork> glades::testOwned(glades::MetaNetwork* cMetaNetwork,
                                                      glades::DataInput* newDataInput,
                                                      GNet::GServer* serverInstance,
                                                      GNet::Connection* cConnection)
{
	return shmea::GPointer<glades::MetaNetwork>(glades::test(cMetaNetwork, newDataInput, serverInstance, cConnection));
}

