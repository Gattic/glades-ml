// Confidential, unpublished property of Robert Carneiro

// The access and distribution of this material is limited solely to
// authorized personnel.  The use, disclosure, reproduction,
// modification, transfer, or transmittal of this work for any purpose
// in any form or by any means without the written permission of
// Robert Carneiro is strictly prohibited.
#include "nn-test.h"
#include "../../unit-test.h"
#include "Backend/Database/GList.h"
#include "../../../Backend/Machine Learning/main.h"
#include "../../../Backend/Machine Learning/Networks/network.h"
#include "../../../Backend/Machine Learning/DataObjects/ImageInput.h"
#include "../../../Backend/Machine Learning/DataObjects/NumberInput.h"
#include "../../../Backend/Machine Learning/State/Terminator.h"

// === This is the primary unit testing function:
// void G_assert(const char* fileName, int lineNo, const char* failureMsg, bool expr)

void ModelSaveLoadTest()
{
    printf("============================================================\n");
    printf("-----------------------------------\n");
    printf("Model Save and Load Test 1\n");
    printf("-----------------------------------\n");

    glades::NNetwork cNetwork;
    shmea::GString netName = "xorgateText";
    shmea::GString inputFName = "xorgateText.csv";
    std::string savePath = "xorgate_trained.model";
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
    cNetwork.terminator.setAccuracy(97);
    G_assert (__FILE__, __LINE__, "==============test::Accuracy() Failed==============", cNetwork.getAccuracy() < 97.0f);
    
    // Train the model
    cNetwork.train(di);

    // Save the model layer and weight
    if (!cNetwork.saveModel(savePath)) {
        printf("Error: Failed to save trained network to: %s\n", savePath.c_str());
        // delete dataInput;
        return;
    } else {
        printf("Trained network saved to: %s\n", savePath.c_str());
    }


    printf("\n============================================================\n");
}
