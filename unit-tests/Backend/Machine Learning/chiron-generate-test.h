// chiron-generate-test.h — CHIRON generation-side unit tests.
// See Backend/Machine Learning/Networks/chiron_generate.h for the module.
//
// C++98.
#ifndef _UT_CHIRON_GENERATE
#define _UT_CHIRON_GENERATE

// Individual cases.
void CHIRONMt19937RawTest();     // core generator: seed 5489 -> published u32s
void CHIRONMt19937GoldenTest();  // canonical doubles bit-exact vs Task-1 goldens
void CHIRONSamplerGoldenTest();  // chiron_sample_token picks bit-exact vs G2 goldens

// Aggregate entry point, wired into unit-tests/main.cpp as "chiron-generate".
void CHIRONGenerateUnitTest();

#endif
