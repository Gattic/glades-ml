#include "bayes-optimizer-test.h"
#include "../../unit-test.h"
#include "Backend/Database/GList.h"
#include "../../../Backend/Machine Learning/Networks/bayes-optimizer.h"

void BayesOptimizerUnitTest()
{
    glades::BayesianOptimizer optimizer(0.0f, 10.0f, 20);
    optimizer.optimize();
}

