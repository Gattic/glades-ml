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
    shmea::GString needle = "hello";
    shmea::GList testList3;
    testList3.addString(needle);
    prediction = bModel3.predict(testList3);
    printf("-----------------------------------\n");
    bModel3.print();
    printf("\n-----------------------------------\n");
    printf("Bayes Prediction: %d \"%s\" -> \"%s\"\n", prediction, needle.c_str(), bModel3.getClassName(prediction).c_str());
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
    needle = "the";
    shmea::GList testList4;
    testList4.addString(needle);
    prediction = bModel4.predict(testList4);
    printf("-----------------------------------\n");
    //bModel4.print();
    printf("\n-----------------------------------\n");
    printf("Bayes Prediction: %d \"%s\" -> \"%s\"\n", prediction, needle.c_str(), bModel4.getClassName(prediction).c_str());
    printf("-----------------------------------\n");
    
    G_assert (__FILE__, __LINE__, "==============Bayes-test4::Accuracy() Failed==============", bModel4.getClassName(prediction) == "forest");
    printf("\n===================================\n");

    printf("\n===================================\n");
    printf("N-Tuple (Multi-word context) Test\n");
    printf("===================================\n");

    // Test with bigrams (n=2): predict based on 2 previous words
    // These sample phrases demonstrate how words can follow different patterns
    // based on context
    shmea::GList trainingList5;
    trainingList5.addString("the");
    trainingList5.addString("quick");
    trainingList5.addString("brown");
    trainingList5.addString("fox");
    trainingList5.addString("jumps");
    trainingList5.addString("over");
    trainingList5.addString("a");
    trainingList5.addString("lazy");
    trainingList5.addString("dog");
    trainingList5.addString("the");
    trainingList5.addString("lazy");
    trainingList5.addString("cat");
    trainingList5.addString("sleeps");
    trainingList5.addString("all");
    trainingList5.addString("day");
    trainingList5.addString("the");
    trainingList5.addString("brown");
    trainingList5.addString("bear");
    trainingList5.addString("eats");
    trainingList5.addString("honey");

    // Create a bayes net with n-tuple (n=2)
    glades::NaiveBayes bModel5;
    shmea::GTable bTable5 = bModel5.importNTuple(trainingList5, 2);
    bModel5.train(bTable5);

    // Test prediction with two words of context: "the lazy"
    shmea::GList testContext1;
    testContext1.addString("the");
    testContext1.addString("lazy");
    int prediction5 = bModel5.predictWithContext(testContext1);
    printf("-----------------------------------\n");
    printf("Context: \"the lazy\"\n");
    printf("Bayes Prediction: %d \"%s\"\n", prediction5, bModel5.getClassName(prediction5).c_str());
    printf("-----------------------------------\n");
    G_assert(__FILE__, __LINE__, "==============Bayes-nTuple-test1::Accuracy() Failed==============", bModel5.getClassName(prediction5) == "cat");

    // Test with another context: "a lazy"
    shmea::GList testContext2;
    testContext2.addString("a");
    testContext2.addString("lazy");
    int prediction6 = bModel5.predictWithContext(testContext2);
    printf("-----------------------------------\n");
    printf("Context: \"a lazy\"\n");
    printf("Bayes Prediction: %d \"%s\"\n", prediction6, bModel5.getClassName(prediction6).c_str());
    printf("-----------------------------------\n");
    G_assert(__FILE__, __LINE__, "==============Bayes-nTuple-test2::Accuracy() Failed==============", bModel5.getClassName(prediction6) == "dog");

    printf("===================================\n");

    // Test with bigrams (n=3): predict based on 3 previous words
    // These sample phrases demonstrate how words can follow different patterns
    // based on context
    shmea::GList trainingList6;
    trainingList6.addString("the");
    trainingList6.addString("fat");
    trainingList6.addString("lazy");
    trainingList6.addString("cat");
    trainingList6.addString("jumps");
    trainingList6.addString("over");
    trainingList6.addString("the");
    trainingList6.addString("slow");
    trainingList6.addString("lazy");
    trainingList6.addString("dog");

    // Create a bayes net with n-tuple (n=3)
    glades::NaiveBayes bModel6;
    shmea::GTable bTable6 = bModel6.importNTuple(trainingList6, 3);
    bModel6.train(bTable6);

    // Test prediction with two words of context: "the lazy"
    shmea::GList testContext3;
    testContext3.addString("the");
    testContext3.addString("fat");
    testContext3.addString("lazy");
    int prediction7 = bModel6.predictWithContext(testContext3);
    printf("-----------------------------------\n");
    printf("Context: \"the fat lazy\"\n");
    printf("Bayes Prediction: %d \"%s\"\n", prediction7, bModel6.getClassName(prediction7).c_str());
    printf("-----------------------------------\n");
    G_assert(__FILE__, __LINE__, "==============Bayes-nTuple-test1::Accuracy() Failed==============", bModel6.getClassName(prediction7) == "cat");

    // Test prediction with two words of context: "the lazy"
    shmea::GList testContext4;
    testContext4.addString("the");
    testContext4.addString("slow");
    testContext4.addString("lazy");
    int prediction8 = bModel6.predictWithContext(testContext4);
    printf("-----------------------------------\n");
    printf("Context: \"the slow lazy\"\n");
    printf("Bayes Prediction: %d \"%s\"\n", prediction8, bModel6.getClassName(prediction8).c_str());
    printf("-----------------------------------\n");
    G_assert(__FILE__, __LINE__, "==============Bayes-nTuple-test1::Accuracy() Failed==============", bModel6.getClassName(prediction8) == "dog");

    printf("\n============================================================\n");
}
