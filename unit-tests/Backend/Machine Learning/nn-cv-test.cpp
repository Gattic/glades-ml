// Confidential, unpublished property of Robert Carneiro

// The access and distribution of this material is limited solely to
// authorized personnel.  The use, disclosure, reproduction,
// modification, transfer, or transmittal of this work for any purpose
// in any form or by any means without the written permission of
// Robert Carneiro is strictly prohibited.
#include "nn-cv-test.h"
#include "../../unit-test.h"
#include "Backend/Database/GList.h"
#include "../../../Backend/Machine Learning/main.h"
#include "../../../Backend/Machine Learning/Networks/network.h"
#include "../../../Backend/Machine Learning/DataObjects/ImageInput.h"
#include "../../../Backend/Machine Learning/DataObjects/NumberInput.h"
#include "../../../Backend/Machine Learning/State/Terminator.h"
#include "../../../Backend/Machine Learning/Structure/nninfo.h"

// === This is the primary unit testing function:
// void G_assert(const char* fileName, int lineNo, const char* failureMsg, bool expr)

void NNCVUnitTestValidation()
{
///////////////////////Test Iris/////////////////////////
    shmea::GString inputFName = "iris.data";
    int inputType = glades::DataInput::CSV;
    //int inputType = glades::DataInput::IMAGE;
    //int inputType = glades::DataInput::TEXT;
    
    if (inputType == glades::DataInput::CSV) {
        inputFName = "datasets/" + inputFName;
    }
    else if (inputType == glades::DataInput::IMAGE) {
        // inputFName = "datasets/images/" + inputFName + "/";
    }
    else if (inputType == glades::DataInput::TEXT) {
	// TODO
        return;
    }
    else
        return;
	    
    // Load the input data to GTable
    shmea::GTable inputIrisTable = shmea::GTable(inputFName, ',', shmea::GTable::TYPE_FILE);
    shmea::GTable* shuffled_inputIrisTable = shmea::GTable::shuffleRows(inputIrisTable);
    shuffled_inputIrisTable->standardize();

    shmea::GString netName1 = "iris";
    glades::NNetwork irisNetwork1;
    shmea::GString netName2 = "iris";
    glades::NNetwork irisNetwork2;

    if ((irisNetwork1.getEpochs() == 0) && (!irisNetwork1.load(netName1))) {
        printf("[NN] Unable to load \"%s\"", netName1.c_str());
        return;
    }

    if ((irisNetwork2.getEpochs() == 0) && (!irisNetwork2.load(netName1))) {
        printf("[NN] Unable to load \"%s\"", netName2.c_str());
        return;
    }

    irisNetwork1.terminator.setEpoch(10000);
    irisNetwork1.terminator.setAccuracy(95);

    irisNetwork2.terminator.setEpoch(10000);
    irisNetwork2.terminator.setAccuracy(95);

    irisNetwork1.skeleton->setLearningRate(0, 0.0003f);
    irisNetwork1.skeleton->setWeightDecay1(0, 0.001f);
    irisNetwork1.skeleton->setWeightDecay2(0, 0.001f);

    irisNetwork1.skeleton->setLearningRate(1, 0.0003f);
    irisNetwork1.skeleton->setWeightDecay1(1, 0.001f);
    irisNetwork1.skeleton->setWeightDecay2(1, 0.001f);

    irisNetwork2.skeleton->setLearningRate(0, 0.0005f);
    irisNetwork2.skeleton->setLearningRate(1, 0.0005f);

    std::vector<glades::NNetwork*> irisNetworks;
    irisNetworks.push_back(&irisNetwork1);
    irisNetworks.push_back(&irisNetwork2);

    std::vector<float> averageAccuracies; 
    std::vector<float> validationAccuracies; 
    unsigned int validationPercent = 10;
    unsigned int testPercent = 20;
    glades::MetaNetwork* cMetaNetwork = glades::crossValidate(irisNetworks , *shuffled_inputIrisTable, glades::DataInput::CSV, 
                                         averageAccuracies, validationAccuracies, validationPercent, testPercent);

    for (unsigned int i = 0; i < averageAccuracies.size(); ++i) {
        printf("-----------------------------------\n");
        printf("Network %d\n", i+1);
        printf("Test Average accuracy %f\n", averageAccuracies[i]);
        printf("-----------------------------------\n");
    }

    for (unsigned int i = 0; i < averageAccuracies.size(); ++i) {
        printf("-----------------------------------\n");
        printf("Network %d\n", i+1);
        printf("Validateion Average accuracy %f\n", validationAccuracies[i]);
        printf("-----------------------------------\n");
    }

///////////////Timing series////////////
    inputFName = "tscv.csv";
    inputType = glades::DataInput::CSV;
    //int inputType = glades::DataInput::IMAGE;
    //int inputType = glades::DataInput::TEXT;
    
    if (inputType == glades::DataInput::CSV) {
        inputFName = "datasets/" + inputFName;
    }
    else if (inputType == glades::DataInput::IMAGE) {
        // inputFName = "datasets/images/" + inputFName + "/";
    }
    else if (inputType == glades::DataInput::TEXT) {
	// TODO
        return;
    }
    else
        return;
	    
    // Load the input data to GTable
    shmea::GTable inputTable = shmea::GTable(inputFName, ',', shmea::GTable::TYPE_FILE);
    inputTable.standardize();

    netName1 = "tscv";
    glades::NNetwork rnnNetwork1;
    netName2 = "tscv";
    glades::NNetwork rnnNetwork2;

    if ((rnnNetwork1.getEpochs() == 0) && (!rnnNetwork1.load(netName1))) {
        printf("[NN] Unable to load \"%s\"", netName1.c_str());
        return;
    }

    if ((rnnNetwork2.getEpochs() == 0) && (!rnnNetwork2.load(netName1))) {
        printf("[NN] Unable to load \"%s\"", netName2.c_str());
        return;
    }

    // Termination Conditions
    rnnNetwork1.terminator.setEpoch(50000);
    rnnNetwork1.terminator.setAccuracy(95);

    rnnNetwork2.terminator.setEpoch(50000);
    rnnNetwork2.terminator.setAccuracy(95);

    std::vector<glades::NNetwork*> networks;
    networks.push_back(&rnnNetwork1);
    networks.push_back(&rnnNetwork2);

    unsigned int foldsNum = 5;
    bool timingSeries = true;
    cMetaNetwork = glades::crossValidate(networks, inputTable, glades::DataInput::CSV, averageAccuracies, foldsNum, timingSeries);

    for (unsigned int i = 0; i < averageAccuracies.size(); ++i) {
        printf("-----------------------------------\n");
        printf("Network %d\n", i+1);
        printf("Average accuracy %f\n", averageAccuracies[i]);
        printf("-----------------------------------\n");
    }

}

