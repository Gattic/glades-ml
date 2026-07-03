// chiron-model-test.h — CHIRON model checkpoint unit tests.
// See chiron_checkpoint.h for the module being tested.
//
// C++98.
#ifndef _UT_CHIRON_MODEL
#define _UT_CHIRON_MODEL

// Individual cases.
void CHIRONCkptBlockCodecTest();
void CHIRONResolveServingTest();
void CHIRONEvalForwardParityTest();

// Aggregate entry point, wired into unit-tests/main.cpp as "chiron-model".
void CHIRONModelUnitTest();

#endif
