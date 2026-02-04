// Copyright 2026 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
// associated documentation files (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge, publish, distribute,
// sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#include "main.h"
#include "Backend/Machine Learning/nn-test.h"
#include "Backend/Machine Learning/nn-cv-test.h"
#include "Backend/Machine Learning/nn-save-load-test.h"
#include "Backend/Machine Learning/nn-benchmarks.h"
#include "Backend/Machine Learning/pca-test.h"
#include "Backend/Machine Learning/kmeans-test.h"
#include "Backend/Machine Learning/bayes-test.h"
#include "Backend/Machine Learning/bayes-optimizer-test.h"
#include "Backend/Machine Learning/ohe-test.h"

int main(int argc, char* argv[])
{
	// For random numbers
	srand(time(NULL));

	if (argc == 1)
	{
	    NNUnitTest();
	    PCAUnitTest();
	    KMeansUnitTest();
	    BayesUnitTest();
	    BayesOptimizerUnitTest();
	    OHEUnitTest();
	}
	else if (argc > 1)
	{
	    if (strcmp(argv[1], "nn") == 0)
		NNUnitTest();
	    else if (strcmp(argv[1], "nn-recurrent") == 0)
		NNRecurrentUnitTest();
	    else if (strcmp(argv[1], "nn-bench") == 0)
		NNBenchmarks(argc, argv);
	    else if (strcmp(argv[1], "pca") == 0)
		PCAUnitTest();
	    else if (strcmp(argv[1], "kmeans") == 0)
		KMeansUnitTest();
	    else if (strcmp(argv[1], "bayes") == 0)
		BayesUnitTest();
	    else if (strcmp(argv[1], "bayes-optimizer") == 0)
		BayesOptimizerUnitTest();
	    else if (strcmp(argv[1], "ohe") == 0)
		OHEUnitTest();
	    else if (strcmp(argv[1], "cv") == 0)
		NNCVUnitTestValidation();
	    else if (strcmp(argv[1], "save-load") == 0)
		NNSaveLoadUnitTest();
	    else
		printf("Invalid test: %s\n", argv[1]);
	}

	printf("========================\n");
	printf("| Unit Tests Completed |\n");
	printf("========================\n");

	pthread_exit(EXIT_SUCCESS);
}
