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
#include "glades.h"
#include "../../include/Backend/Database/gtable.h"
#include "../../include/Backend/Database/gtype.h"
#include "../../main.h"
#include "GMath/gmath.h"
#include "RNN.h"
#include "State/layer.h"
#include "State/node.h"
#include "Structure/nninfo.h"
#include "metanetwork.h"
#include "network.h"

using namespace glades;

std::map<std::string, glades::MetaNetwork*>* glades::metaNets =
	new std::map<std::string, glades::MetaNetwork*>();
std::map<std::string, glades::NNetwork*>* glades::neuralNets =
	new std::map<std::string, glades::NNetwork*>();
pthread_mutex_t* glades::nnetworkMutex = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));

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
	pthread_mutex_init(getNNetworkMutex(), NULL);
	if (!doesDatabaseExist())
		createDatabase();
}

/*!
 * @brief cleanup glades
 * @details destroy the old metaNets
 */
void glades::cleanup()
{
	pthread_mutex_lock(getNNetworkMutex());
	std::map<std::string, glades::MetaNetwork*>::iterator itr = metaNets->begin();
	for (; itr != metaNets->end(); ++itr)
		delete itr->second;
	metaNets->clear();
	pthread_mutex_unlock(getNNetworkMutex());

	// destroy the neural network mutex
	pthread_mutex_destroy(nnetworkMutex);
	free(nnetworkMutex);
}

/*!
 * @brief get neural net mutex
 * @details retrieves glades' neural network mutex
 * @return the neural network mutex
 */
pthread_mutex_t* glades::getNNetworkMutex()
{
	return nnetworkMutex;
}

glades::NNetwork* glades::getNeuralNetwork(const std::string& netName)
{
	glades::NNetwork* cNetwork = new glades::NNetwork();
	if (!cNetwork->load(netName))
	{
		char buffer[256];
		sprintf(buffer, "Unable to load \"%s\"", netName.c_str());
		puts(buffer);
		return NULL;
	}

	return cNetwork;
}

glades::RNN* glades::getRNN(const std::string& netName)
{
	glades::RNN* cNetwork = new glades::RNN();
	if (!cNetwork->load(netName))
	{
		char buffer[256];
		sprintf(buffer, "Unable to load \"%s\"", netName.c_str());
		puts(buffer);
		return NULL;
	}

	return cNetwork;
}

bool glades::saveNeuralNetwork(glades::NNetwork* newNet)
{
	if (!newNet)
		return false;

	// Save the neural network
	if (!newNet->save())
	{
		char buffer[256];
		sprintf(buffer, "Unable to save \"%s\"", newNet->getName().c_str());
		puts(buffer);
		return false;
	}

	// Add the Neural net in container
	(*neuralNets)[newNet->getName()] = newNet;
	char buffer[256];
	sprintf(buffer, "Saved \"%s\"", newNet->getName().c_str());
	return true;
}

/*!
 * @brief add a MetaNetwork
 * @details adds a MetaNetwork to glades' existing metaNets
 * @param newNet the MetaNetwork to add
 */
void glades::addMetaNetwork(glades::MetaNetwork* newNet)
{
	if (newNet)
	{
		std::string netName = newNet->getName();
		if (metaNets->find(netName) == metaNets->end())
		{
			pthread_mutex_lock(getNNetworkMutex());
			metaNets->insert(std::pair<std::string, glades::MetaNetwork*>(netName, newNet));
			pthread_mutex_unlock(getNNetworkMutex());
		}
	}
}

/*!
 * @brief remove a MetaNetwork
 * @details remove a MetaNetwork by name from glades' existing metaNets, if it exists
 * @param netName the name of the MetaNetwork to remove
 */
void glades::removeMetaNetwork(const std::string& netName)
{
	if (metaNets->find(netName) != metaNets->end())
	{
		// delete it from the data structure
		pthread_mutex_lock(getNNetworkMutex());
		metaNets->erase(metaNets->find(netName));
		pthread_mutex_unlock(getNNetworkMutex());
	}
}

/*!
 * @brief train network
 * @details train a neural network
 * @param networkInfo the incoming or desired neural net info
 * @param dataTable the data to use in training
 * @return the trained MetaNetwork object
 */
glades::MetaNetwork* glades::train(NNInfo* networkInfo, const shmea::GTable& dataTable,
								   Terminator* Arnold)
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
		subnets[i]->run(dataTable, Arnold, glades::NNetwork::RUN_TRAIN);

	// Add the new metanet to the collection
	if (cMetaNetwork)
		glades::addMetaNetwork(cMetaNetwork);

	return cMetaNetwork;
}

/*!
 * @brief train network
 * @details train a neural network
 * @param networkInfo the incoming or desired neural net info
 * @param dataTable the data to use in training
 * @return the trained MetaNetwork object
 */
glades::MetaNetwork* glades::train(glades::NNetwork* cNetwork, const shmea::GTable& dataTable,
								   Terminator* Arnold)
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
		subnets[i]->run(dataTable, Arnold, glades::NNetwork::RUN_TRAIN);

	// Add the new metanet to the collection
	if (cMetaNetwork)
		glades::addMetaNetwork(cMetaNetwork);

	return cMetaNetwork;
}

/*!
 * @brief train a metanetwork
 * @details train a set of neural networks
 * @param networkInfo the incoming or desired neural net info
 * @param dataTable the data to use in training
 * @return the trained MetaNetwork object
 */
glades::MetaNetwork* glades::train(glades::MetaNetwork* cMetaNetwork,
								   const shmea::GTable& dataTable, Terminator* Arnold)
{
	if (!cMetaNetwork)
		return NULL;

	// Train the Neural Network
	std::vector<glades::NNetwork*> subnets = cMetaNetwork->getSubnets();
	for (unsigned int i = 0; i < subnets.size(); ++i)
		subnets[i]->run(dataTable, Arnold, glades::NNetwork::RUN_TRAIN);

	// Add the new metanet to the collection
	glades::addMetaNetwork(cMetaNetwork);

	return cMetaNetwork;
}

/*!
 * @brief test network
 * @details test a network
 * @param networkInfo the incoming network's relevant information
 * @param dataTable the data to use in testing
 * @return the tested MetaNetwork object
 */
glades::MetaNetwork* glades::test(NNInfo* networkInfo, const shmea::GTable& dataTable)
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
		subnets[i]->run(dataTable, NULL, glades::NNetwork::RUN_TEST);

	// Add the new metanet to the collection
	if (cMetaNetwork)
		glades::addMetaNetwork(cMetaNetwork);

	return cMetaNetwork;
}

/*!
 * @brief test network
 * @details test a network
 * @param networkInfo the incoming network's relevant information
 * @param dataTable the data to use in testing
 * @return the tested MetaNetwork object
 */
glades::MetaNetwork* glades::test(glades::NNetwork* networkInfo, const shmea::GTable& dataTable)
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
		subnets[i]->run(dataTable, NULL, glades::NNetwork::RUN_TEST);

	// Add the new metanet to the collection
	if (cMetaNetwork)
		glades::addMetaNetwork(cMetaNetwork);

	return cMetaNetwork;
}

/*!
 * @brief test a metanetwork
 * @details test a set of neural networks
 * @param networkInfo the incoming or desired neural net info
 * @param dataTable the data to use in testing
 * @return the tested MetaNetwork object
 */
glades::MetaNetwork* glades::test(glades::MetaNetwork* cMetaNetwork, const shmea::GTable& dataTable)
{
	if (!cMetaNetwork)
		return NULL;

	// Test the Neural Network
	std::vector<glades::NNetwork*> subnets = cMetaNetwork->getSubnets();
	for (unsigned int i = 0; i < subnets.size(); ++i)
		subnets[i]->run(dataTable, NULL, glades::NNetwork::RUN_TEST);

	// Add the new metanet to the collection
	glades::addMetaNetwork(cMetaNetwork);

	return cMetaNetwork;
}

/*!
 * @brief network cross-validation
 * @details cross-validate a neural net
 * @param networkInfo the incoming network's relevant information
 * @param fname the location of the data to use in cross-validation
 * @param saveInstance whether or not to save the results
 * @param importFlag flag designating where/how the cross validation data is stored (e.g. at a URL,
 * in a file)
 * @return the cross-validated MetaNetwork object
 */
glades::MetaNetwork* glades::crossValidate(NNInfo* networkInfo, std::string fname,
										   bool saveInstance, int importFlag)
{
	if (!networkInfo)
		return NULL;

	// 1.shuffle the data set randomly
	// 2. split it into k size chunks
	// 3. a
	// lets do it...
	/*printf("[glades] Cross Validating %s:%s...\n", networkInfo->getName().c_str(), fname.c_str());

	//the input data to cross validate
	GAnalysis* study=new GAnalysis(fname, ',', importFlag, false);
	shmea::GTable* inputFile=study->otherData;
	if(!inputFile)
		return NULL;

	//stratify the data for cross validation
	int k=4;
	float cvAccuracy=0.0f;
	std::vector<shmea::GTable*> stratifiedInputFiles=shmea::GTable::stratify(inputFile, k);

	// standardize it now that it's stratified
	for(int file; file < stratifiedInputFiles.size(); ++file)
		stratifiedInputFiles[file]->standardize();

	// create cMetaNetwork, a metaNetwork of size k
	glades::MetaNetwork* cMetaNetwork = new glades::MetaNetwork(networkInfo, false, k,
	stratifiedInputFiles[0]->numberOfCols());
	for(int i=0;i<stratifiedInputFiles.size();++i)//stratifiedInputFiles.size() == k
	{
		//The training set
		shmea::GTable* trainInputFile=new shmea::GTable(',');//size == k-1
		for(int j=0;j<stratifiedInputFiles.size();++j)
		{
			//skip the testing set
			if(i == j)
				continue;

			//Compile the training set
			for(int rowCounter=0;rowCounter<stratifiedInputFiles[j]->numberOfRows();++rowCounter)
				trainInputFile->addRow(stratifiedInputFiles[j]->getRow(rowCounter));
		}

		// select the i-th subnet of cMetaNetwork and train it
		glades::NNetwork*
	cNetwork=cMetaNetwork->getSubnetByName(networkInfo->getName()+GType::intTOstring(i));
		if(cNetwork)
		{
			cNetwork->run(glades::NNetwork::RUN_TRAIN, trainInputFile);

			//test the network
			shmea::GTable* testInputFile=stratifiedInputFiles[i];
			if(testInputFile->numberOfCols() > 0)
			{
				//Test our validation set
				cNetwork->run(glades::NNetwork::RUN_TEST, testInputFile);
				cvAccuracy+=cNetwork->getAccuracy();
			}
			else
				printf("[glades] Invalid data format[2]: %s\n", fname.c_str());
		}

		//done with this version
		if(trainInputFile)
			delete trainInputFile;
	}

	cvAccuracy/=k;
	printf("\n===========================================================\n");
	printf("[glades] CV Accuracy: %f%%\n", cvAccuracy);

	//requires further testing
	//if(cvAccuracy/100.0f < GMath::INLIER)

	return cMetaNetwork;*/
	return NULL;
}

/*!
 * @brief Train, test, cross-validate new net
 * @details Create a new network, then train, test, and cross-validate it with given data
 * @param netName the desired name for the new neural net
 * @param fname a list of locations of the data to use in training/testing/cross-validation
 * @oaram learningRate the desired learning rate for the new neural net
 * @param saveInstance whether or not to save the results
 * @param importFlag flag designating where/how the input data is stored (e.g. at URLs, in files)
 * @return the trained, tested, cross-validated MetaNetwork object
 */
glades::MetaNetwork* glades::crossValidate(std::string netName, std::vector<std::string> fname,
										   float learningRate, bool saveInstance, int importFlag)
{
	// lets do it...
	/*printf("[glades] Cross Validating %s:%ld files...\n", netName.c_str(), fname.size());

	//the input data to cross validate
	std::vector<shmea::GTable*> inputFiles;
	for(int i = 0; i < fname.size(); ++i)
	{
		shmea::GTable* inputFile=new shmea::GTable(fname[i], ',', importFlag);
		std::vector<shmea::GTable*> inputDatasets=PreProcessUnstandardized(fname[i], ',',
	importFlag);
		if(inputDatasets.size() == 1)
			inputFile=inputDatasets[0];
		else
			return NULL;
		inputFiles.push_back(inputFile);
	}

	//stratify the data for cross validation
	int k=4;
	float cvAccuracy=0.0f;
	std::vector<shmea::GTable*> stratifiedInputFiles=shmea::GTable::stratify(inputFiles, k);

	// standardize it now that it's stratified
	for(int file; file < stratifiedInputFiles.size(); ++file)
		stratifiedInputFiles[file]->standardize();

	// create cMetaNetwork, a metaNetwork of size k
	MetaNetwork* cMetaNetwork = new MetaNetwork(netName, k, stratifiedInputFiles[0]->numberOfCols(),
	learningRate, saveInstance);
	for(int i=0;i<stratifiedInputFiles.size();++i)//stratifiedInputFiles.size() == k
	{
		//The training set
		shmea::GTable* trainInputFile=new shmea::GTable(',');//size == k-1
		for(int j=0;j<stratifiedInputFiles.size();++j)
		{
			//skip the testing set
			if(i == j)
				continue;

			//Compile the training set
			for(int rowCounter=0;rowCounter<stratifiedInputFiles[j]->numberOfRows();++rowCounter)
				trainInputFile->addRow(stratifiedInputFiles[j]->getRow(rowCounter));
		}

		// select the i-th subnet of cMetaNetwork and train it
		glades::NNetwork* cNetwork = cMetaNetwork->subnets[i];
		cNetwork->run(glades::NNetwork::RUN_TRAIN, trainInputFile);

		//test the network
		shmea::GTable* testInputFile=stratifiedInputFiles[i];
		if(testInputFile->numberOfCols() > 0)
		{
			//Test our validation set
			cNetwork->run(glades::NNetwork::RUN_TEST, testInputFile);
			cvAccuracy+=cNetwork->getAccuracy();
		}
		else
			printf("[glades] Invalid data format[3]\n");// %s\n", fname.c_str());

		//done with this version
		if(trainInputFile)
			delete trainInputFile;
	}

	cvAccuracy/=k;
	printf("\n===========================================================\n");
	printf("[glades] CV Accuracy: %f%%\n", cvAccuracy);
*/
	// requires further testing
	// if(cvAccuracy/100.0f < GMath::INLIER)

	return NULL;
}
