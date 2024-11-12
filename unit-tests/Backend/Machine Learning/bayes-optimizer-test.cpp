#include "bayes-optimizer-test.h"
#include "../../unit-test.h"
#include "Backend/Database/GList.h"
#include "../../../Backend/Machine Learning/Networks/bayes-optimizer.h"

void BayesOptimizerUnitTest()
{
    shmea::GString trainingFname = "datasets/bayesOptTestLR";
    shmea::GTable trainingTable(trainingFname, ',', shmea::GTable::TYPE_FILE);
    //trainingTable.print();

    std::vector<std::pair<float, float> > trainingData;
    for (unsigned int r = 0; r < trainingTable.numberOfRows(); r++)
    {
	const shmea::GList& row = trainingTable[r];
	std::pair<float, float> dataPoint(row.getFloat(0), row.getFloat(1));
	trainingData.push_back(dataPoint);
    }

    glades::BayesianOptimizer optimizer;
    float bestGuess = optimizer.optimize(trainingData);
    printf("Best guess: %f\n", bestGuess);
    G_assert (__FILE__, __LINE__, "==============BayesOpt::Best Guess Failed==============", bestGuess == 0.001f);
    //optimizer.print();
    printf("===================================\n");
}

