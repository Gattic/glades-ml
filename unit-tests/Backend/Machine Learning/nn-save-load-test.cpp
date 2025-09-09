// Confidential, unpublished property of Robert Carneiro

// The access and distribution of this material is limited solely to
// authorized personnel.  The use, disclosure, reproduction,
// modification, transfer, or transmittal of this work for any purpose
// in any form or by any means without the written permission of
// Robert Carneiro is strictly prohibited.
#include "nn-save-load-test.h"
#include "../../unit-test.h"
#include "Backend/Database/GList.h"
#include "../../../Backend/Machine Learning/main.h"
#include "../../../Backend/Machine Learning/Networks/network.h"
#include "../../../Backend/Machine Learning/DataObjects/ImageInput.h"
#include "../../../Backend/Machine Learning/DataObjects/NumberInput.h"
#include "../../../Backend/Machine Learning/State/Terminator.h"

void NNSaveLoadUnitTest()
{
    printf("============================================================\n");
    printf("-----------------------------------\n");
    printf("NN Save Load Test\n");
    printf("-----------------------------------\n");

    glades::NNetwork cNetwork;
    shmea::GString netName = "iris";
    shmea::GString inputFName = "iris.data";
    int inputType = glades::DataInput::CSV;
    //int inputType = glades::DataInput::IMAGE;
    //int inputType = glades::DataInput::TEXT;
    
    // Modify the paths to properly load the data later
    glades::DataInput* di = NULL;
    if (inputType == glades::DataInput::CSV) {
    	inputFName = "datasets/" + inputFName;
    	di = new glades::NumberInput();
    }
    else if (inputType == glades::DataInput::IMAGE) {
    	// inputFName = "datasets/images/" + inputFName + "/";
    	di = new glades::ImageInput();
    }
    else if (inputType == glades::DataInput::TEXT) {
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
    if ((cNetwork.getEpochs() == 0) && (!cNetwork.load(netName))) {
    	printf("[NN Save Load] Unable to load the skeleton");
    	return;
    }
    
    // Termination Conditions
    //cNetwork.setTimestamp(maxTimeStamp);
    cNetwork.terminator.setEpoch(100000);
    cNetwork.terminator.setAccuracy(95);
    
    // Run the training and retrieve a metanetwork
    glades::MetaNetwork* newTrainNet = glades::train(&cNetwork, di);

    if (!cNetwork.meat.saveState(netName.c_str())) {
    	printf("[NN Save Load] Unable to save the network biases and weights");
    	return;
    }

    glades::NNetwork cNetwork1;
    
    // Load the neural network skeleton
    if ((cNetwork1.getEpochs() == 0) && (!cNetwork1.load(netName))) {
    	printf("[NN Save Load] Unable to load the skeleton");
    	return;
    }

    //Create layers
    cNetwork1.setMustdBuildMeat(false);
    cNetwork1.meat.build(cNetwork.skeleton, di, false);

    //Load the biases and weights from file
    if (!cNetwork1.meat.loadState(cNetwork.skeleton, netName.c_str())) {
    	printf("[NN Save Load] Unable to load the network biases and weights");
    	return;
    }
    
    // Termination Conditions
    //cNetwork.setTimestamp(maxTimeStamp);
    cNetwork1.terminator.setEpoch(100000);
    cNetwork1.terminator.setAccuracy(95);
    
    // Run the training and retrieve a metanetwork
    glades::MetaNetwork* newTrainNet1 = glades::train(&cNetwork1, di);

    printf("\n============================================================\n");
}
