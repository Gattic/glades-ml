// Confidential, unpublished property of Robert Carneiro

// The access and distribution of this material is limited solely to
// authorized personnel.  The use, disclosure, reproduction,
// modification, transfer, or transmittal of this work for any purpose
// in any form or by any means without the written permission of
// Robert Carneiro is strictly prohibited.
#include "main.h"
#include "Backend/Machine Learning/nn-test.h"
#include "Backend/Machine Learning/pca-test.h"
#include "Backend/Machine Learning/bayes-test.h"
#include "Backend/Machine Learning/bayes-optimizer-test.h"
#include "Backend/Machine Learning/ohe-test.h"
#include "Backend/Machine Learning/nnmodelsaveload-test.h"

int main(int argc, char* argv[])
{
	// For random numbers
	srand(time(NULL));

	if (argc == 1)
	{
	    NNUnitTest();
	    PCAUnitTest();
	    BayesUnitTest();
	    BayesOptimizerUnitTest();
	    OHEUnitTest();
		NNModelSaveLoadUnitTest();
	}
	else if (argc > 1)
	{
	    if (strcmp(argv[1], "nn") == 0)
		NNUnitTest();
	    else if (strcmp(argv[1], "pca") == 0)
		PCAUnitTest();
	    else if (strcmp(argv[1], "bayes") == 0)
		BayesUnitTest();
	    else if (strcmp(argv[1], "bayes-optimizer") == 0)
		BayesOptimizerUnitTest();
	    else if (strcmp(argv[1], "ohe") == 0)
		OHEUnitTest();
	    else if (strcmp(argv[1], "nn-model-save-load") == 0)
		NNModelSaveLoadUnitTest();
	    else
		printf("Invalid test: %s\n", argv[1]);
	}

	printf("========================\n");
	printf("| Unit Tests Completed |\n");
	printf("========================\n");

	pthread_exit(EXIT_SUCCESS);
}
