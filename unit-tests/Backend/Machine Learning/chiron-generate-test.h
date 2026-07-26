// chiron-generate-test.h — CHIRON generation-side unit tests.
// See Backend/Machine Learning/Networks/chiron_generate.h for the module.
//
// C++98.
#ifndef _UT_CHIRON_GENERATE
#define _UT_CHIRON_GENERATE

// Individual cases.
void CHIRONMt19937RawTest();        // core generator: seed 5489 -> published u32s
void CHIRONMt19937GoldenTest();     // canonical doubles bit-exact vs Task-1 goldens
void CHIRONSamplerGoldenTest();     // chiron_sample_token picks bit-exact vs G2 goldens
void CHIRONDegenMetricsTest();      // chiron_degeneration_metrics: 4 cases
void CHIRONRepetitionMetricsTest(); // ARREST detector-v1 metrics and collapse fixtures
void CHIRONRepetitionHazardsTest(); // strict-prefix hazards, confidence, cap/overflow
void CHIRONRepetitionAppendInvariantTest(); // future append cannot alter prior rows
void CHIRONTfEvalTest();            // chiron_tf_eval: NLL/acc bit-equal + logitsAllOut exact
void CHIRONGenerateStochasticDrawParityTest(); // draw-count parity: full-softmax, draw-dependent picks

// GPU-free aggregate plus the existing full aggregate.
void CHIRONGenerateCpuUnitTest();
void CHIRONGenerateUnitTest();

#endif
