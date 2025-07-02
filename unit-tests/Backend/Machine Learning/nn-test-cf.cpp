// Confidential, unpublished property of Robert Carneiro

// The access and distribution of this material is limited solely to
// authorized personnel.  The use, disclosure, reproduction,
// modification, transfer, or transmittal of this work for any purpose
// in any form or by any means without the written permission of
// Robert Carneiro is strictly prohibited.
#include "nn-test-cf.h"
#include "../../unit-test.h"
#include "Backend/Database/GList.h"
#include "../../../Backend/Machine Learning/main.h"
#include "../../../Backend/Machine Learning/Networks/network.h"
#include "../../../Backend/Machine Learning/DataObjects/ImageInput.h"
#include "../../../Backend/Machine Learning/DataObjects/NumberInput.h"
#include "../../../Backend/Machine Learning/State/Terminator.h"

// === This is the primary unit testing function:
// void G_assert(const char* fileName, int lineNo, const char* failureMsg, bool expr)

void NNCFUnitTest()
{
    printf("============================================================\n");
    printf("-----------------------------------\n");
    printf("NN Test CF 1\n");
    printf("-----------------------------------\n");

    glades::NNetwork cNetwork;
    shmea::GString netName = "xornet";
    shmea::GString inputFName = "xorgate.csv";
    int inputType = glades::DataInput::CSV;
    //int inputType = glades::DataInput::IMAGE;
    //int inputType = glades::DataInput::TEXT;
    
    if (inputType == glades::DataInput::CSV)
    {
        inputFName = "datasets/" + inputFName;
    }
    else if (inputType == glades::DataInput::IMAGE)
    {
        // inputFName = "datasets/images/" + inputFName + "/";
    }
    else if (inputType == glades::DataInput::TEXT)
    {
	// TODO
        return;
    }
    else
        return;
	    
    // Load the input data
    shmea::GTable rawTable = shmea::GTable(inputFName, ',', shmea::GTable::TYPE_FILE);
    rawTable.standardize();
    unsigned int kFolds = 2;
    shmea::GVector<shmea::GTable*> stratTbls = shmea::GTable::stratify(rawTable, kFolds);

    float averageAccuracy = 0.0f;

    for(unsigned int i = 0; i < stratTbls.size(); ++i)
    {
	    printf("-----------------------------------\n");
	    printf("Iteration %d\n", i+1);
	    printf("-----------------------------------\n");
	    // Modify the paths to properly load the data later
	    glades::DataInput* diTrain = NULL;
	    glades::DataInput* diTest = NULL;
	    if (inputType == glades::DataInput::CSV)
	    {
            diTrain = new glades::NumberInput();
            diTest = new glades::NumberInput();
	    }
	    else if (inputType == glades::DataInput::IMAGE)
	    {
            diTrain = new glades::ImageInput();
            diTest = new glades::ImageInput();
	    }
	    else if (inputType == glades::DataInput::TEXT)
	    {
		// TODO
            return;
	    }
	    else
            return;
	    
	    if (!diTrain || !diTest)
            return;

	    // Load the input data
        shmea::GTable* trainGTable = shmea::GTable::unionFolds(stratTbls, i);
	    diTrain->import(*trainGTable);
	    diTest->import(*stratTbls[i]);
	    
	    // Load the neural network
	    if ((cNetwork.getEpochs() == 0) && (!cNetwork.load(netName)))
	    {
            printf("[NN] Unable to load \"%s\"", netName.c_str());
            return;
	    }
	    
	    // Termination Conditions
	    //cNetwork.setTimestamp(maxTimeStamp);
	    cNetwork.terminator.setEpoch(100000);
	    cNetwork.terminator.setAccuracy(95);
	    
	    // Run the training and retrieve a metanetwork
	    glades::MetaNetwork* newTrainNet =
            glades::train(&cNetwork, diTrain);

	    // Run the test and retrieve a metanetwork
	    glades::MetaNetwork* newTestNet =
            glades::test(&cNetwork, diTest);

        averageAccuracy += cNetwork.getAccuracy();

//	    G_assert (__FILE__, __LINE__, "==============NN1-test::Accuracy() Failed==============", cNetwork.getAccuracy() >= 95.0f);

    }
    averageAccuracy /= (float)stratTbls.size();

    printf("-----------------------------------\n");
    printf("Average accuracy %f\n", averageAccuracy);
    printf("-----------------------------------\n");


    
    
    /*
    printf("-----------------------------------\n");
    printf("NN Test 2\n");
    printf("-----------------------------------\n");

    glades::NNetwork cNetwork2;
    netName = "iris";
    inputFName = "iris.data";
    inputType = glades::DataInput::CSV;
    //int inputType = glades::DataInput::IMAGE;
    //int inputType = glades::DataInput::TEXT;
    
    // Modify the paths to properly load the data later
    glades::DataInput* di2 = NULL;
    if (inputType == glades::DataInput::CSV)
    {
    	inputFName = "datasets/" + inputFName;
    	di2 = new glades::NumberInput();
    }
    else if (inputType == glades::DataInput::IMAGE)
    {
    	// inputFName = "datasets/images/" + inputFName + "/";
    	di2 = new glades::ImageInput();
    }
    else if (inputType == glades::DataInput::TEXT)
    {
    	// TODO
    	return;
    }
    else
    	return;
    
    if (!di2)
    	return;
    
    // Load the input data
    di2->import(inputFName);
    
    // Load the neural network
    if ((cNetwork2.getEpochs() == 0) && (!cNetwork2.load(netName)))
    {
    	printf("[NN] Unable to load \"%s\"", netName.c_str());
    	return;
    }
    
    // Termination Conditions
    //cNetwork2.setTimestamp(maxTimeStamp);
    cNetwork2.terminator.setEpoch(100000);
    cNetwork2.terminator.setAccuracy(95);
    
    // Run the training and retrieve a metanetwork
    glades::MetaNetwork* newTrainNet2 =
    	glades::train(&cNetwork2, di2);
    
    G_assert (__FILE__, __LINE__, "==============NN2-test::Accuracy() Failed==============", cNetwork2.getAccuracy() >= 95.0f);


    printf("-----------------------------------\n");
    printf("NN Test 3\n");
    printf("-----------------------------------\n");

    glades::NNetwork cNetwork3;
    netName = "xorgateText";
    inputFName = "xorgateText.csv";
    inputType = glades::DataInput::CSV;
    //int inputType = glades::DataInput::IMAGE;
    //int inputType = glades::DataInput::TEXT;
    
    // Modify the paths to properly load the data later
    glades::DataInput* di3 = NULL;
    if (inputType == glades::DataInput::CSV)
    {
    	inputFName = "datasets/" + inputFName;
    	di3 = new glades::NumberInput();
    }
    else if (inputType == glades::DataInput::IMAGE)
    {
    	// inputFName = "datasets/images/" + inputFName + "/";
    	di3 = new glades::ImageInput();
    }
    else if (inputType == glades::DataInput::TEXT)
    {
    	// TODO
    	return;
    }
    else
    	return;
    
    if (!di3)
    	return;
    
    // Load the input data
    di3->import(inputFName);
    
    // Load the neural network
    if ((cNetwork3.getEpochs() == 0) && (!cNetwork3.load(netName)))
    {
    	printf("[NN] Unable to load \"%s\"", netName.c_str());
    	return;
    }
    
    // Termination Conditions
    //cNetwork3.setTimestamp(maxTimeStamp);
    cNetwork3.terminator.setEpoch(100000);
    cNetwork3.terminator.setAccuracy(95);
    
    // Run the training and retrieve a metanetwork
    glades::MetaNetwork* newTrainNet3 =
    	glades::train(&cNetwork3, di3);

    G_assert (__FILE__, __LINE__, "==============NN3-test::Accuracy() Failed==============", cNetwork3.getAccuracy() >= 95.0f);

    printf("\n============================================================\n");
    */
}
