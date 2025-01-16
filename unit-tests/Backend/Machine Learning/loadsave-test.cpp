// #include "nn-test.h"
#include "../../unit-test.h"
// #include "Backend/Database/GList.h"
#include "../../../Backend/Machine Learning/main.h"
#include "../../../Backend/Machine Learning/Networks/network.h"
#include "../../../Backend/Machine Learning/DataObjects/ImageInput.h"
#include "../../../Backend/Machine Learning/DataObjects/NumberInput.h"
#include "../../../Backend/Machine Learning/DataObjects/DataInput.h"
#include "../../../Backend/Machine Learning/State/Terminator.h"
#include "Backend/Database/GList.h"
#include "Backend/Database/GTable.h"
#include "Backend/Database/GType.h"
#include "Backend/Database/SaveFolder.h"
#include "Backend/Database/SaveTable.h"


// Function to load and train a neural network
void RunNNTest(const std::string& netName, const std::string& inputFileName, int inputType, const std::string& savePath) {
    printf("\n-----------------------------------\n");
    printf("Running Test for Network: %s\n", netName.c_str());
    printf("-----------------------------------\n");

    // Initialize the network
    glades::NNetwork network;
    glades::DataInput* dataInput = NULL;
    // int inputType = glades::DataInput::CSV;
    // Determine input type and initialize appropriate DataInput
    std::string fullInputPath = "datasets/" + inputFileName; // Build file path

    if (inputType == glades::DataInput::CSV) {
        dataInput = new glades::NumberInput();
    } else if (inputType == glades::DataInput::IMAGE) {
        fullInputPath = "datasets/images/" + inputFileName;
        dataInput = new glades::ImageInput();
    } else {
        printf("Unsupported input type for network: %s\n", netName.c_str());
        return;
    }


    // Ensure the DataInput object was created
    if (!dataInput) {
        printf("Failed to initialize data input for network: %s\n", netName.c_str());
        return;
    }

    // Import data
    try {
        dataInput->import(fullInputPath.c_str()); // Use standard C-string for import
    } catch (...) {
        printf("Failed to import data from %s\n", fullInputPath.c_str());
        // delete dataInput;
        return;
    }

    // Load the network
    if ((network.getEpochs() == 0) && (!network.load(netName.c_str()))) {
        printf("[NN] Unable to load network: %s\n", netName.c_str());
    
        return;
    }

    printf("start %s",netName.c_str());
    network.load(netName.c_str());
    printf("end");
    // skeleton = new NNInfo(netName);
    // Set termination conditions
    network.terminator.setEpoch(10);
    shmea::GString val ;
    val = network.getEpochs();
    printf("===== %f",val);
    network.terminator.setAccuracy(95);

    // Train the network
    network.train(dataInput);

    shmea::GList weightlist;

    weightlist=network.getWeightsNew();
    
    shmea::GTable saveMe;
    saveMe.addHeader(0,"Weights");
	saveMe.addRow(weightlist);

    shmea::SaveFolder* nnList = new shmea::SaveFolder("xornet_model");
    // shmea::SaveFolder* nnList = new shmea::SaveFolder(savePath);
	// This will save the weights of the model with name "XORNET_Weights"
    nnList->newItem("Weights",saveMe);
    
    // Assert accuracy (initially low, expecting improvement after training)
    G_assert(__FILE__, __LINE__, "Test::Accuracy Failed", network.getAccuracy() < 95.0f);

    
    
    // Save BiasWeights
    network.saveBiasWeight("xornet_model");



    // Save the trained network directly using std::string
    if (!network.save()) {
        printf("Failed to save trained network to: %s\n", savePath.c_str());
    } else {
        printf("Trained network saved to: %s\n", savePath.c_str());
    }


    // Clean up
    delete dataInput;
    printf("Test for network %s completed.\n", netName.c_str());
}

// Main function to run all tests
void loadsave() {
    printf("============================================================\n");
    printf("Starting Neural Network Unit Tests\n");
    printf("============================================================\n");

    // Generate XOR dataset
    // GenerateXORData("xorgate.csv");

    // Test 1: XOR Network
    RunNNTest("xornet", "xorgate.csv", glades::DataInput::CSV, "models/xornet_trained.model");

    // Test 2: Iris Dataset
    // RunNNTest("iris", "iris.data", glades::DataInput::CSV, "models/iris_trained.model");

    // Test 3: XOR Network with Text (future extension example)
    // RunNNTest("xorgateText", "xorgateText.csv", glades::DataInput::CSV, "models/xorgateText_trained.model");

    printf("\n============================================================\n");
    printf("All Neural Network Unit Tests Completed\n");
    printf("============================================================\n");
}

