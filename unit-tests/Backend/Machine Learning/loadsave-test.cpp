
#include "../../unit-test.h"

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

    std::string fullInputPath = "datasets/" + inputFileName; // Build file path

    if (inputType == glades::DataInput::CSV) 
    {
        dataInput = new glades::NumberInput();
    } 
    else if (inputType == glades::DataInput::IMAGE) 
    {
        fullInputPath = "datasets/images/" + inputFileName;
        dataInput = new glades::ImageInput();
    } 
    else 
    {
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
        if (fullInputPath.empty()) {
            printf("Error: Input file path is empty.\n");
            // delete dataInput;
            return;
        }
        dataInput->import(fullInputPath.c_str());
    } catch (...) {
        printf("Failed to import data from %s\n", fullInputPath.c_str());
        // delete dataInput;  // Prevent memory leak
        return;
    }

    // Load the network
    printf("Starting to load network: %s\n", netName.c_str());
    if (!network.load(netName.c_str())) {
        printf("Error: Failed to load network: %s\n", netName.c_str());
        // delete dataInput;
        return;
    }
    printf("Network loaded successfully.\n");

    // Set network parameters
    network.terminator.setEpoch(1);
    shmea::GString val = network.getEpochs();
    printf("Current Epoch: %s\n", val.c_str());
    network.terminator.setAccuracy(95);

    // Train the network
    try {
        network.train(dataInput);
        printf("Training completed successfully.\n");
    } catch (...) {
        printf("Error: Training failed for network: %s\n", netName.c_str());
        // delete dataInput;
        return;
    }

    // Save the trained network
    if (!network.saveModel(savePath)) {
        printf("Error: Failed to save trained network to: %s\n", savePath.c_str());
        // delete dataInput;
        return;
    } else {
        printf("Trained network saved to: %s\n", savePath.c_str());
    }

    // Test loading the saved model
    printf("Testing loading the files into GTable...\n");
    shmea::SaveFolder* nnList2 = new shmea::SaveFolder(savePath.c_str());
    if (!nnList2) {
        printf("Error: Failed to allocate memory for SaveFolder.\n");
        // delete dataInput;
        return;
    }

    nnList2->loadItem(savePath.c_str());

    printf("Test for network %s completed successfully.\n", netName.c_str());
}

void loadsave() {
    printf("============================================================\n");
    printf("Starting Neural Network Unit Tests\n");
    printf("============================================================\n");

    // Test 1: XOR Network
    RunNNTest("xornet", "xorgate.csv", glades::DataInput::CSV, "xornet_trained.model");

    printf("\n============================================================\n");
    printf("All Neural Network Unit Tests Completed\n");
    printf("============================================================\n");
}