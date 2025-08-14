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
#include "metanetwork.h"
#include "network.h"
#include "Backend/Database/GTable.h"
#include "../Structure/nninfo.h"
#include "../DataObjects/NumberInput.h"
#include "../DataObjects/ImageInput.h"

using namespace glades;

/*!
 * @brief MetaNetwork constructor
 * @param newName the lookup name of the metanetwork
 * @details user probably wants to manually call addSubnet
 */
glades::MetaNetwork::MetaNetwork(shmea::GString newName)
{
	name = newName;
	collectionType = ONE_TO_ONE;
}

/*!
 * @brief MetaNetwork constructor
 * @details initializes a MetaNetwork based on user-input data
 * @param input the input data for the neural net
 * @param networkInfo neural net information, including name
 * @param subnetCount the number of subnets in this neural net
 */
glades::MetaNetwork::MetaNetwork(NNInfo* networkInfo, int subnetCount)
{
	name = networkInfo->getName();

	if (subnetCount > 1)
		collectionType = ONE_TO_MANY;
	else
		collectionType = ONE_TO_ONE;

	// if making identical subnets (for k-folds CV or otherwise):
	// loop to fill subnets with the correct number of subnets
	for (int i = 0; i < subnetCount; ++i)
		addSubnet(networkInfo);
}

glades::MetaNetwork::MetaNetwork(shmea::GString metaNetName, shmea::GString nname, int k)
{
	name = metaNetName;
	if (k > 1)
		collectionType = ONE_TO_MANY;
	else
		collectionType = ONE_TO_ONE;
	for (int i = 0; i < k; ++i)
	{
		addSubnet(nname);
	}
}

/*!
 * @brief MetaNetwork destructor
 * @details destroys the MetaNetwork object
 */
glades::MetaNetwork::~MetaNetwork()
{
	name = "";
	clearSubnets();
}

/*!
 * @brief set MetaNetwork name
 * @details sets the name of a MetaNetwork
 * @param newName the new name for this MetaNetwork
 */
void glades::MetaNetwork::setName(shmea::GString newName)
{
	name = newName;
}

/*!
 * @brief add new subnet to the MetaNetwork
 * @details adds a new sub-network to a MetaNetwork's subnets list
 * @param networkInfo the desired network information for the subnet
 * @param inputSize the number of inputs to the subnet
 */
void glades::MetaNetwork::addSubnet(NNInfo* networkInfo)
{
	if (!networkInfo)
		return;

	glades::NNetwork* cNetwork = new glades::NNetwork(networkInfo);
	subnets.push_back(cNetwork);
}

void glades::MetaNetwork::addSubnet(const shmea::GString nNetName)
{
	if (nNetName.length() <= 0)
		return;
	glades::NNetwork* cNetwork = new glades::NNetwork();
	if (cNetwork->load(nNetName))
	{
		subnets.push_back(cNetwork);
	}
}

/*!
 * @brief add new subnet to the MetaNetwork
 * @details adds a new sub-network to a MetaNetwork's subnets list
 * @param networkInfo the desired network information for the subnet
 * @param inputSize the number of inputs to the subnet
 */
void glades::MetaNetwork::addSubnet(glades::NNetwork* cNetwork)
{
	if (!cNetwork)
		return;

	subnets.push_back(cNetwork);
}

/*!
 * @brief clear MetaNetwork's subnets
 * @details removes all the subnets from a MetaNetwork
 */
void glades::MetaNetwork::clearSubnets()
{
	subnets.clear();
}

/*!
 * @brief get MetaNetwork's name
 * @details retrieves a MetaNetwork's name
 * @return the MetaNetwork's name
 */
shmea::GString glades::MetaNetwork::getName() const
{
	return name;
}

/*!
 * @brief get MetaNetwork size
 * @details retrieves the number of subnets in a MetaNetwork
 * @return the number of subnets in the MetaNetwork
 */
int glades::MetaNetwork::size() const
{
	return subnets.size();
}

/*!
 * @brief get MetaNetwork subnets
 * @details retrieves the subnets in a MetaNetwork
 * @return the MetaNetwork's subnets, a vector of NNetwork objects
 */
std::vector<glades::NNetwork*> glades::MetaNetwork::getSubnets() const
{
	return subnets;
}

/*!
 * @brief get subnet
 * @details get the subnet at a particular index
 * @param index the index of the desired subnet in the MetaNetwork
 * @return the subnet at the requested index, or NULL if the index is invalid
 */
glades::NNetwork* glades::MetaNetwork::getSubnet(unsigned int index)
{
	if (index >= subnets.size())
		return NULL;

	return subnets[index];
}

/*!
 * @brief get subnet name
 * @details get the name of a subnet at a particular index
 * @param index the index of the desired subnet in the MetaNetwork
 * @return the name of the subnet at the requested index, or empty string if the index is invalid
 */
shmea::GString glades::MetaNetwork::getSubnetName(unsigned int index) const
{
	if (index >= subnets.size())
		return "";

	return subnets[index]->getName();
}

/*!
 * @brief get subnet by name
 * @details retrieves the subnet of a given name in the MetaNetwork
 * @param newName the name of the subnet to retrieve
 * @return the subnet with the requested name, or NULL if a subnet of the requested name is not
 * present
 */
glades::NNetwork* glades::MetaNetwork::getSubnetByName(shmea::GString newName) const
{
	for (unsigned int i = 0; i < subnets.size(); ++i)
	{
		if (subnets[i]->getName() == newName)
			return subnets[i];
	}

	// if it doesn't exist
	return NULL;
}

void glades::MetaNetwork::crossValidate(shmea::GString fNames, DataInput* newDataInput)
{
	// populate the input data
	shmea::GTable cinputFile(fNames.c_str(), ',', shmea::GTable::TYPE_FILE);

	float cvAccuracy = 0.0f;
	shmea::GVector<shmea::GTable*> stratifiedInputFiles =
		shmea::GTable::stratify(cinputFile, subnets.size()); // split it up into k segments

	for (unsigned int i = 0; i < stratifiedInputFiles.size(); ++i)
	{
		shmea::GTable trainInputFile; // size == k-1
		for (unsigned int j = 0; j < stratifiedInputFiles.size(); ++j)
		{
			// skip the testing set
			if (i == j)
				continue;

			// Compile the training set
			trainInputFile.append(stratifiedInputFiles[j]);
		}

		//
		subnets[i]->train(newDataInput);

		shmea::GTable testInputFile = *stratifiedInputFiles[i];
		if (testInputFile.numberOfCols() > 0)
		{
			// Test our validation set
			subnets[i]->test(newDataInput);
			cvAccuracy += subnets[i]->getAccuracy();
		}
	}

	printf("cvAccuracy: %f \t\n", cvAccuracy / subnets.size());
}

void glades::MetaNetwork::crossValidate(const shmea::GTable& inputGTable, 
                                        const int inputType,
                                        std::vector<glades::NNetwork*>& networks, 
                                        std::vector<float>& averageAccuracies, 
                                        unsigned int foldsNum,
                                        bool timingSeries)
{
    shmea::GVector<shmea::GTable*> stratTbls = shmea::GTable::stratify(inputGTable, foldsNum, timingSeries);
    averageAccuracies.clear();
    for (unsigned int netNum = 0; netNum < networks.size(); ++netNum) {
        addSubnet(networks[netNum]);
        averageAccuracies.push_back(0.0f);
    }

    unsigned int i = 0;
    if (timingSeries) {
        i = 1;
    }

    for (; i < stratTbls.size(); ++i) {
	    printf("===================================\n");
	    printf("===================================\n");
	    printf("Fold %d\n", i+1);
	    printf("-----------------------------------\n");
        
        glades::DataInput* diTrain = NULL;
        glades::DataInput* diTest = NULL;
        if (inputType == glades::DataInput::CSV) {
            diTrain = new glades::NumberInput();
            diTest = new glades::NumberInput();
        }
        else if (inputType == glades::DataInput::IMAGE) {
            diTrain = new glades::ImageInput();
            diTest = new glades::ImageInput();
        }
        else if (inputType == glades::DataInput::TEXT) {
        // TODO
            return;
        }
        else
            return;
        
        if (!diTrain || !diTest)
            return;

        // Load the input data
        shmea::GTable* trainGTable = shmea::GTable::unionFolds(stratTbls, i, timingSeries);
        
        diTrain->import(*trainGTable);
        diTest->import(*stratTbls[i]);

        for (int netInd = 0; netInd < size(); ++netInd) {
            printf("-----------------------------------\n");
            printf("Network %d\n", netInd+1);
            printf("-----------------------------------\n");
            glades::NNetwork* cNetwork = getSubnet(netInd);
            
            // Run the training and retrieve a metanetwork
            printf("Train\n");
            cNetwork->train(diTrain);

            // Run the test and retrieve a metanetwork
            printf("Test\n");
            cNetwork->test(diTest);

            averageAccuracies[netInd] += cNetwork->getAccuracy();
        }
    }

    for (int netInd = 0; netInd < size(); ++netInd) {
        averageAccuracies[netInd] /= (float)stratTbls.size();
    }
}

void glades::MetaNetwork::crossValidate(const shmea::GTable& inputGTable, 
                                        const int inputType,
                                        std::vector<glades::NNetwork*>& networks, 
                                        std::vector<float>& averageAccuracies, 
                                        std::vector<float>& validationAccuracies, 
                                        unsigned int validationPercent,
                                        unsigned int testPercent)
{
    averageAccuracies.clear();
    validationAccuracies.clear();
    unsigned int rowNum = inputGTable.numberOfRows();
    unsigned int validationRowNum = rowNum * validationPercent / 100;
    if (validationRowNum == 0) {
        ++validationRowNum;
    }
    unsigned int testRowNum = rowNum * testPercent / 100;
    if (testRowNum == 0) {
        ++testRowNum;
    }
    if (rowNum <= testRowNum + validationRowNum) {
        return;
    }
    unsigned int foldsNum = (rowNum - validationRowNum + testRowNum - 1) / testRowNum; //ceiling
    if (foldsNum == 0) {
        foldsNum = 1;
    }

    shmea::GTable* inputGTableTT = shmea::GTable::firstNRows(inputGTable, rowNum - validationRowNum);
    shmea::GTable* inputGTableV = shmea::GTable::lastNRows(inputGTable, validationRowNum);

    crossValidate(*inputGTableTT, inputType, networks, averageAccuracies, foldsNum);

    glades::DataInput* diValidation = NULL;
    if (inputType == glades::DataInput::CSV) {
        diValidation = new glades::NumberInput();
    }
    else if (inputType == glades::DataInput::IMAGE) {
        diValidation = new glades::ImageInput();
    }
    else if (inputType == glades::DataInput::TEXT) {
    // TODO
        return;
    }
    else
        return;
        
    diValidation->import(inputGTableV);

    validationAccuracies.clear();
    for (unsigned int netNum = 0; netNum < networks.size(); ++netNum) {
        validationAccuracies.push_back(0.0f);
    }

    for (int netInd = 0; netInd < size(); ++netInd) {
        glades::NNetwork* cNetwork = getSubnet(netInd);
        
        // Run the validation
        cNetwork->test(diValidation);
        validationAccuracies[netInd] = cNetwork->getAccuracy();
    }
}

