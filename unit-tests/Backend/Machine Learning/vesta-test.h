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
#ifndef _UT_VESTA
#define _UT_VESTA

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <unistd.h>

void VESTAGramSchmidtTest();
void VESTAThinQRTest();
void VESTASketchedSVDTest();
void VESTAInitStateTest();
void VESTALogScaleUpdateTest();
void VESTATrustRegionClampTest();
void VESTAStepDescentTest();
void VESTAOrthogonalInvarianceTest();
void VESTAGpuParityTest();
void VESTAGpuParityMomentumTest();
void VESTAGpuRefreshDeviceTest();
void VESTAGpuSingleRefreshTest();
void VESTAGpuRefreshBenchmark();
void VESTAGpuStepBenchmark();
void VESTAComplementMomentumTest();
void VESTATrackedEmaTest();
void VESTAGradientBasisTest();
void VESTATransformerIntegrationTest();
void VESTATransformerGpuIntegrationTest();
void VESTASweepBenchmark();
void VESTASweepV2Benchmark();
void VESTASweepLambdaPerpExtended();
void VESTASweepMomentumCompare();
void VESTASweepAblationCompare();
void VESTASweepScaleLadder();
void VESTASweepScalePush();
void VESTASweepRawMomentumLongHorizon();
void VESTASweepPlainRawAtScale();
void VESTASweepScaleGpu();
void VESTASweepLpAtScale();
void VESTASweepLongHorizonSchedule();
void VESTASweepSameMemory();
void VESTASweepRankAtScale();
void VESTASweepScaleUltra();
void VESTASweepScaleMega();
void VESTAProfileBench();
void VESTAProfileBenchLong();
void VESTAUnitTest();

#endif
