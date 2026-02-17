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
#include "Backend/Machine Learning/nn-mixed-precision-test.h"
#include "Backend/Machine Learning/nn-benchmarks.h"
#include "Backend/Machine Learning/pca-test.h"
#include "Backend/Machine Learning/kmeans-test.h"
#include "Backend/Machine Learning/bayes-test.h"
#include "Backend/Machine Learning/bayes-optimizer-test.h"
#include "Backend/Machine Learning/ohe-test.h"
#include "Backend/Machine Learning/mapped-dataset-test.h"
#include "Backend/Machine Learning/transformer-serving-layer-test.h"
#include "Backend/Machine Learning/prop-fuzz-test.h"
#include "Backend/Machine Learning/parallel-test.h"
#include "Backend/Machine Learning/ddp-test.h"
#include "Backend/Machine Learning/transformer-improvements-test.h"
#include "Backend/Machine Learning/gpu-training-test.h"
#include <vector>

int main(int argc, char* argv[])
{
	// For random numbers
	srand(time(NULL));

	if (argc == 1)
	{
	    NNUnitTest();
	    NNRecurrentUnitTest();
	    NNTransformerUnitTest();
	    TransformerServingLayerUnitTest();
	    PCAUnitTest();
	    KMeansUnitTest();
	    BayesUnitTest();
	    BayesOptimizerUnitTest();
	    OHEUnitTest();
	    MappedDatasetUnitTest();
	    NNMixedPrecisionUnitTest();
	    DDPUnitTest();
	}
	else if (argc > 1)
	{
	    if (strcmp(argv[1], "nn") == 0)
		NNUnitTest();
	    else if (strcmp(argv[1], "nn-recurrent") == 0)
		NNRecurrentUnitTest();
	    else if (strcmp(argv[1], "nn-transformer") == 0)
		NNTransformerUnitTest();
	    else if (strcmp(argv[1], "transformer-serving") == 0 || strcmp(argv[1], "serving") == 0)
		TransformerServingLayerUnitTest();
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
	    else if (strcmp(argv[1], "mapped") == 0)
		MappedDatasetUnitTest();
	    else if (strcmp(argv[1], "cv") == 0)
		NNCVUnitTestValidation();
	    else if (strcmp(argv[1], "save-load") == 0)
		NNSaveLoadUnitTest();
	    else if (strcmp(argv[1], "nn-mixed-precision") == 0 || strcmp(argv[1], "nn-mp") == 0)
		NNMixedPrecisionUnitTest();
	    else if (strcmp(argv[1], "prop-fuzz") == 0)
		PropFuzzUnitTest();
	    else if (strcmp(argv[1], "parallel") == 0)
		ParallelUnitTest();
	    else if (strcmp(argv[1], "ddp") == 0)
		DDPUnitTest();
	    else if (strcmp(argv[1], "transformer-improvements") == 0 || strcmp(argv[1], "ti") == 0)
		TransformerImprovementsUnitTest();
	    else if (strcmp(argv[1], "gpu-training") == 0)
		GpuTrainingUnitTest();
	    else if (strcmp(argv[1], "nnall") == 0)
        {
	        OHEUnitTest();
		    MappedDatasetUnitTest();
		    NNUnitTest();
		    NNRecurrentUnitTest();
		    NNTransformerUnitTest();
		    TransformerServingLayerUnitTest();

		    // Run a fixed benchmark configuration when invoked via `nnall`.
		    // (Avoid forwarding `argc/argv` from the unit-test runner.)
		    const char* bench_args[] = {
			"nn-bench",
			"--dataset", "datasets/rnn.csv",
			"--epochs", "200",
			"--hidden", "8",
			"--repeats", "3",
			"--lr", "0.05",
			"--lr-schedule", "step",
			"--step-size", "50",
			"--gamma", "0.5",
			"--clip-norm", "5",
		    };
		    const int bench_argc = (int)(sizeof(bench_args) / sizeof(bench_args[0]));

		    std::vector<char*> bench_argv(bench_argc, (char*)0);
		    for (int i = 0; i < bench_argc; ++i)
			    bench_argv[i] = strdup(bench_args[i]);

		    NNBenchmarks(bench_argc, &bench_argv[0]);

		    for (int i = 0; i < bench_argc; ++i)
			    free(bench_argv[i]);

		    NNSaveLoadUnitTest();
		    NNMixedPrecisionUnitTest();
		    PropFuzzUnitTest();
		    ParallelUnitTest();
		    DDPUnitTest();
		    TransformerImprovementsUnitTest();
        }
	    else
		printf("Invalid test: %s\n", argv[1]);
	}

	printf("========================\n");
	printf("| Unit Tests Completed |\n");
	printf("========================\n");

	return EXIT_SUCCESS;
}
