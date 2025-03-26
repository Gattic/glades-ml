// Confidential, unpublished property of Robert Carneiro

// The access and distribution of this material is limited solely to
// authorized personnel.  The use, disclosure, reproduction,
// modification, transfer, or transmittal of this work for any purpose
// in any form or by any means without the written permission of
// Robert Carneiro is strictly prohibited.
#include "nnmodelsaveload-test.h"
#include "../../unit-test.h"
#include "Backend/Database/GList.h"
#include "Backend/Database/GTable.h"
#include "Backend/Database/GType.h"
#include "../../../Backend/Machine Learning/main.h"
#include "../../../Backend/Machine Learning/Networks/network.h"
#include "../../../Backend/Machine Learning/DataObjects/ImageInput.h"
#include "../../../Backend/Machine Learning/DataObjects/NumberInput.h"
#include "../../../Backend/Machine Learning/State/Terminator.h"
#include "Backend/Database/GTable.h"
#include "Backend/Database/GType.h"
#include "Backend/Database/SaveFolder.h"
#include "Backend/Database/SaveTable.h"
#include "Backend/Networking/main.h"


// === This is the primary unit testing function:
// void G_assert(const char* fileName, int lineNo, const char* failureMsg, bool expr)

void NNModelSaveLoadUnitTest()
{
    printf("============================================================\n");
    printf("-----------------------------------\n");
    printf("NN Model Save/Load Test 1\n");
    printf("-----------------------------------\n");

    glades::NNetwork cNetwork;
    // glades::NNetwork cNetwork2;
    shmea::GString netName = "xornet";
    shmea::GString inputFName = "xorgate.csv";
    int inputType = glades::DataInput::CSV;
    
    // Modify the paths to properly load the data later
    glades::DataInput* di = NULL;
    if (inputType == glades::DataInput::CSV)
    {
    	inputFName = "datasets/" + inputFName;
    	di = new glades::NumberInput();
    }
    else if (inputType == glades::DataInput::IMAGE)
    {
    	// inputFName = "datasets/images/" + inputFName + "/";
    	di = new glades::ImageInput();
    }
    else if (inputType == glades::DataInput::TEXT)
    {
    	// TODO
    	return;
    }
    else
    	return;
    
    if (!di)
    	return;
    
    // Load the input data
    di->import(inputFName);
    
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
    	glades::train(&cNetwork, di);

    // G_assert (__FILE__, __LINE__, "==============NN1-test::Accuracy() Failed==============", cNetwork.getAccuracy() >= 95.0f);

    // Save the Weight for Traiened neural network
    if (!cNetwork.saveNNmodel(netName.c_str()))
    {
    	printf("[NN] Unable to save \"%s\"", netName.c_str());
    	return;
    }
    
    printf("\n============================================================\n");

    printf("\n Test Load weigths from saved NN Model\n");

    // Load the weights from the saved file
    shmea::GString trainingFname = "database/";
    shmea::GString fileName = trainingFname +shmea::GString(netName.c_str()) + "_weights/weights";
    printf("File name: %s\n", fileName.c_str());
    shmea::GTable weightTable(fileName, ',', shmea::GTable::TYPE_FILE);
    weightTable.print();

    printf("No of rows :%u\n", weightTable.numberOfRows());

    if (weightTable.numberOfRows() == 0) {
        printf("Failed to load weights\n");
        return; // Loading failed
    }


    printf("===================TEST LOAD and Set saved WEIGHTS=========================================\n");
    
    glades::NNetwork cNetwork2;

    //  Load the neural network
    if ((cNetwork2.getEpochs() == 0) && (!cNetwork2.load(netName)))
    {
    	printf("[NN] Unable to load \"%s\"", netName.c_str());
    	return;
    }

    //  Set weights
    cNetwork2.build(di);
    if (!cNetwork2.setNewWeights(weightTable[0]))
    {
    	printf("[NN] Unable to save \"%s\"", netName.c_str());
    	return;
    }


    glades::MetaNetwork* newTrainNet2 =
    	glades::test(&cNetwork2, di);

      
    printf("Accuracy: %f\n", cNetwork2.getAccuracy());    

    G_assert (__FILE__, __LINE__, "==============NNModelSaveLoadUnitTest-test::Accuracy() Failed==============", cNetwork2.getAccuracy() >= 95.0f);


}
