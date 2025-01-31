
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
    try 
    {
        dataInput->import(fullInputPath.c_str()); // Use standard C-string for import
    } catch (...) 
    {
        printf("Failed to import data from %s\n", fullInputPath.c_str());
       
        return;
    }

    printf("start %s",netName.c_str());
    network.load(netName.c_str());
    printf("end");

    network.terminator.setEpoch(1);
    shmea::GString val ;
    val = network.getEpochs();
    printf("===== %f",val);
    network.terminator.setAccuracy(95);

    // Train the network
    network.train(dataInput);

    
    shmea::GTable saveMe;
    shmea::SaveFolder* nnList = new shmea::SaveFolder(savePath.c_str());
    

    // Save the trained network directly using std::string
    if (!network.saveModel(savePath)) 
    {
        printf("Failed to save trained network to: %s\n", savePath.c_str());
    } 
    else 
    {
        printf("Trained network saved to: %s\n", savePath.c_str());
    }

    // TEST LOADING THE MODEL
    printf("TEST LOADING THE FILES TO GTABLE");
	shmea::SaveFolder* nnList2 = new shmea::SaveFolder(savePath.c_str());
	// nnList2->load();
 
	// std::vector<shmea::SaveTable*> saveTables = nnList2->getItems();
	// for (unsigned int i = 0; i < saveTables.size(); ++i)
	// {
	// 	shmea::SaveTable* cItem = saveTables[i];
	// 	if (!cItem)
	// 		continue;

	// }



    // Clean up
    delete dataInput;
    printf("Test for network %s completed.\n", netName.c_str());
}

// Main function to run all tests
void loadsave() 
{
    printf("============================================================\n");
    printf("Starting Neural Network Unit Tests\n");
    printf("============================================================\n");

    // Test 1: XOR Network
    RunNNTest("xornet", "xorgate.csv", glades::DataInput::CSV, "xornet_trained.model");

    printf("\n============================================================\n");
    printf("All Neural Network Unit Tests Completed\n");
    printf("============================================================\n");
}

