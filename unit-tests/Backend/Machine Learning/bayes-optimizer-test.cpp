#include "bayes-optimizer-test.h"
#include "../../unit-test.h"
#include "Backend/Database/GList.h"
#include "../../../Backend/Machine Learning/Networks/bayes-optimizer.h"
#include <cmath>
#include <vector>

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

void BayesOptimizerMultiDimTest()
{
    printf("=== BayesOptimizer Multi-Dim Tests ===\n");

    // ---- Sub-test 1: 2D GP with known function f(x1,x2) = -(x1^2 + x2^2) ----
    {
        printf("Sub-test 1: 2D GP predict on f(x1,x2) = -(x1^2 + x2^2)\n");

        std::vector<float> ls(2, 1.0f);
        glades::GaussianProcess gp(ls, 1.0f, 1e-5f);

        // Add training samples on a grid
        for (int i = -2; i <= 2; ++i)
        {
            for (int j = -2; j <= 2; ++j)
            {
                float x1 = (float)i;
                float x2 = (float)j;
                float y = -(x1 * x1 + x2 * x2);
                std::vector<float> xv(2);
                xv[0] = x1;
                xv[1] = x2;
                gp.addSample(xv, y);
            }
        }
        gp.fit();

        // Predict at the origin: f(0,0) = 0
        std::vector<float> origin(2, 0.0f);
        std::pair<float, float> pred = gp.predict(origin);
        float mu = pred.first;
        float sigma2 = pred.second;
        printf("  predict(0,0): mu=%f, sigma2=%f\n", mu, sigma2);
        G_assert(__FILE__, __LINE__,
                 "==============BayesOpt-ND::2D GP origin mean==============",
                 fabs(mu - 0.0f) < 0.2f);
        G_assert(__FILE__, __LINE__,
                 "==============BayesOpt-ND::2D GP origin low variance==============",
                 sigma2 < 0.1f);

        // Predict at (1,1): f(1,1) = -2
        std::vector<float> p11(2, 1.0f);
        std::pair<float, float> pred11 = gp.predict(p11);
        printf("  predict(1,1): mu=%f, sigma2=%f\n", pred11.first, pred11.second);
        G_assert(__FILE__, __LINE__,
                 "==============BayesOpt-ND::2D GP (1,1) mean==============",
                 fabs(pred11.first - (-2.0f)) < 0.3f);

        // Predict at a far point (5,5): should have higher variance
        std::vector<float> p55(2);
        p55[0] = 5.0f;
        p55[1] = 5.0f;
        std::pair<float, float> pred55 = gp.predict(p55);
        printf("  predict(5,5): mu=%f, sigma2=%f\n", pred55.first, pred55.second);
        G_assert(__FILE__, __LINE__,
                 "==============BayesOpt-ND::2D GP (5,5) higher variance==============",
                 pred55.second > sigma2);

        printf("  Sub-test 1 PASSED\n");
    }

    // ---- Sub-test 2: Cholesky consistency (fit + predict at training point) ----
    {
        printf("Sub-test 2: Cholesky consistency - predict at training points\n");

        std::vector<float> ls(3, 0.5f);
        glades::GaussianProcess gp(ls, 1.0f, 1e-6f);

        // Add 5 training samples in 3D
        float xdata[][3] = {
            {0.1f, 0.2f, 0.3f},
            {0.5f, 0.5f, 0.5f},
            {0.8f, 0.1f, 0.9f},
            {0.3f, 0.7f, 0.2f},
            {0.6f, 0.4f, 0.6f}
        };
        float ydata[] = {1.0f, -0.5f, 2.0f, 0.3f, -1.0f};

        for (int i = 0; i < 5; ++i)
        {
            std::vector<float> xv(3);
            xv[0] = xdata[i][0];
            xv[1] = xdata[i][1];
            xv[2] = xdata[i][2];
            gp.addSample(xv, ydata[i]);
        }
        gp.fit();

        // Predict at each training point: mean should match y closely
        for (int i = 0; i < 5; ++i)
        {
            std::vector<float> xv(3);
            xv[0] = xdata[i][0];
            xv[1] = xdata[i][1];
            xv[2] = xdata[i][2];
            std::pair<float, float> pred = gp.predict(xv);
            printf("  predict at train[%d]: mu=%f (expected %f), sigma2=%f\n",
                   i, pred.first, ydata[i], pred.second);
            G_assert(__FILE__, __LINE__,
                     "==============BayesOpt-ND::Cholesky train-point mean==============",
                     fabs(pred.first - ydata[i]) < 0.05f);
            G_assert(__FILE__, __LINE__,
                     "==============BayesOpt-ND::Cholesky train-point variance==============",
                     pred.second < 0.01f);
        }

        printf("  Sub-test 2 PASSED\n");
    }

    // ---- Sub-test 3: ARD kernel sensitivity test ----
    {
        printf("Sub-test 3: ARD kernel sensitivity (different length scales)\n");

        // Create GP with long length scale in dim 0, short in dim 1
        // This means: dim 0 varies slowly (similar values far apart),
        //             dim 1 varies quickly (distinct values close together)
        std::vector<float> ls(2);
        ls[0] = 10.0f;  // Long length scale: dim 0 is "smooth"
        ls[1] = 0.1f;   // Short length scale: dim 1 is "wiggly"

        glades::GaussianProcess gp(ls, 1.0f, 1e-6f);

        // Train on a single point
        std::vector<float> trainX(2, 0.0f);
        gp.addSample(trainX, 1.0f);
        gp.fit();

        // Predict at point shifted only in dim 0 (long length scale -> still correlated)
        std::vector<float> shiftD0(2, 0.0f);
        shiftD0[0] = 1.0f;
        std::pair<float, float> predD0 = gp.predict(shiftD0);

        // Predict at point shifted only in dim 1 (short length scale -> less correlated)
        std::vector<float> shiftD1(2, 0.0f);
        shiftD1[1] = 1.0f;
        std::pair<float, float> predD1 = gp.predict(shiftD1);

        printf("  shift dim0 by 1: mu=%f, sigma2=%f\n", predD0.first, predD0.second);
        printf("  shift dim1 by 1: mu=%f, sigma2=%f\n", predD1.first, predD1.second);

        // Shifting in dim 0 (long ls=10) should have LOWER variance than shifting in dim 1 (short ls=0.1)
        // because dim 0 shift is "close" in kernel space while dim 1 shift is "far"
        G_assert(__FILE__, __LINE__,
                 "==============BayesOpt-ND::ARD dim0 lower variance==============",
                 predD0.second < predD1.second);

        // Mean for dim 0 shift should be closer to training value (1.0)
        G_assert(__FILE__, __LINE__,
                 "==============BayesOpt-ND::ARD dim0 mean closer to train==============",
                 fabs(predD0.first - 1.0f) < fabs(predD1.first - 1.0f));

        printf("  Sub-test 3 PASSED\n");
    }

    printf("=== All BayesOptimizer Multi-Dim Tests PASSED ===\n");
}
