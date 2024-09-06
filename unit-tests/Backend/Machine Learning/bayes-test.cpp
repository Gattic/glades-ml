// Confidential, unpublished property of Robert Carneiro

// The access and distribution of this material is limited solely to
// authorized personnel.  The use, disclosure, reproduction,
// modification, transfer, or transmittal of this work for any purpose
// in any form or by any means without the written permission of
// Robert Carneiro is strictly prohibited.
#include "bayes-test.h"
#include "../../unit-test.h"
#include "Backend/Database/GList.h"
#include "../../../Backend/Machine Learning/Networks/bayes.h"

// === This is the primary unit testing function:
// void G_assert(const char* fileName, int lineNo, const char* failureMsg, bool expr)

void BayesUnitTest()
{
    printf("============================================================\n");
    printf("-----------------------------------\n");
    printf("Bayes Test\n");
    printf("-----------------------------------\n");
    shmea::GString trainingFname = "datasets/classifier.csv";
    shmea::GTable trainingTable(trainingFname, ',', shmea::GTable::TYPE_FILE);

    // Classifier 1
    // Create a bayes net
    glades::NaiveBayes bModel;
    shmea::GTable bTable = bModel.import(trainingTable);
    //bTable.print();
    bModel.train(bTable);

    // predict with a new learning rate
    shmea::GList testList;
    testList.addString("Ralph");
    int prediction = bModel.predict(testList);
    printf("-----------------------------------\n");
    bModel.print();
    printf("\n-----------------------------------\n");
    printf("Bayes Prediction: %d \"%s\"\n", prediction, bModel.getClassName(prediction).c_str());
    printf("-----------------------------------\n");
    G_assert (__FILE__, __LINE__, "==============Bayes-test::Accuracy() Failed==============", bModel.getClassName(prediction) == "Happy");
    printf("\n===================================\n");

    // Classifier 2
    trainingFname = "datasets/classifier2.csv";
    shmea::GTable trainingTable2(trainingFname, ',', shmea::GTable::TYPE_FILE);

    // Create a bayes net
    glades::NaiveBayes bModel2;
    shmea::GTable bTable2 = bModel2.import(trainingTable2);
    //bTable2.print();
    bModel2.train(bTable2);

    // predict with a new learning rate
    shmea::GList testList2;
    testList2.addString("Olga");
    prediction = bModel2.predict(testList2);
    printf("-----------------------------------\n");
    bModel2.print();
    printf("\n-----------------------------------\n");
    printf("Bayes Prediction: %d \"%s\"\n", prediction, bModel2.getClassName(prediction).c_str());
    printf("-----------------------------------\n");
    
    G_assert (__FILE__, __LINE__, "==============Bayes-test2::Accuracy() Failed==============", bModel2.getClassName(prediction) == "Angry");
    printf("\n===================================\n");

    // Classifier 3
    trainingFname = "datasets/wordlist.txt";
    shmea::GList trainingList3;
    trainingList3.loadWords(trainingFname);

    // Create a bayes net
    glades::NaiveBayes bModel3;
    shmea::GTable bTable3 = bModel3.import(trainingList3);
    //bTable3.print();
    bModel3.train(bTable3);

    // predict with a new learning rate
    shmea::GList testList3;
    testList3.addString("hello");
    prediction = bModel3.predict(testList3);
    printf("-----------------------------------\n");
    bModel3.print();
    printf("\n-----------------------------------\n");
    printf("Bayes Prediction: %d \"%s\"\n", prediction, bModel3.getClassName(prediction).c_str());
    printf("-----------------------------------\n");
    
    G_assert (__FILE__, __LINE__, "==============Bayes-test3::Accuracy() Failed==============", bModel3.getClassName(prediction) == "world");
    printf("\n===================================\n");

    // Classifier 4
    trainingFname = "datasets/wordlist2.txt";
    shmea::GList trainingList4;
    trainingList4.loadWords(trainingFname);

    // Create a bayes net
    glades::NaiveBayes bModel4;
    shmea::GTable bTable4 = bModel4.import(trainingList4);
    //bTable4.print();
    bModel4.train(bTable4);

    // predict with a new learning rate
    shmea::GList testList4;
    testList4.addString("the");
    prediction = bModel4.predict(testList4);
    printf("-----------------------------------\n");
    //bModel4.print();
    printf("\n-----------------------------------\n");
    printf("Bayes Prediction: %d \"%s\"\n", prediction, bModel4.getClassName(prediction).c_str());
    printf("-----------------------------------\n");
    
    G_assert (__FILE__, __LINE__, "==============Bayes-test4::Accuracy() Failed==============", bModel4.getClassName(prediction) == "forest");
    printf("\n===================================\n");

    printf("\n============================================================\n");
}
