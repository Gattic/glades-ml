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
#ifndef _UT_CHIRON
#define _UT_CHIRON

// CHIRON (reversible-flow transformer) unit tests.
// See research/CHIRON_framework.md.

// Individual cases (callable separately for debugging).
void CHIRONShearReversibilityTest();
void CHIRONReLNRoundtripTest();
void CHIRONBlockRoundtripTest();
void CHIRONMultiBlockRoundtripTest();
void CHIRONAttentionShearReversibilityTest();
void CHIRONFullBlockRoundtripTest();
void CHIRONMultiFullBlockRoundtripTest();
void CHIRONBf16DriftTest();
void CHIRONSketchProjectLiftTest();
void CHIRONSketchCorrectedBf16Test();
void CHIRONPerTokenSketchBf16Test();
void CHIRONGpuParityTest();
void CHIRONBatchedSketchParityTest();
void CHIRONGpuEndToEndTest();
void CHIRONGpuAttentionShearParityTest();
void CHIRONGpuFullBlockEndToEndTest();
void CHIRONGpuReLNBackwardTest();
void CHIRONGpuAttentionShearBackwardTest();
void CHIRONGpuFullBlockBackwardTest();
void CHIRONGpuMultiBlockBackwardTest();
void CHIRONMicroTrainingDemoTest();
void CHIRONCublasTiledAttentionParityTest();
void CHIRONCublasTiledAttentionBf16ParityTest();
void CHIRONProductionScaleMemoryTest();
void CHIRONCublasTiledAttentionBackwardParityTest();
void CHIRONStochasticBf16RoundingTest();
void CHIRONFlashShearVsTiledBf16ParityTest();
void CHIRONFlashShearBackwardBf16ParityTest();
void CHIRONBf16WeightProjectionParityTest();
void CHIRONBf16WeightBackwardParityTest();
void CHIRONLocalAttentionFullWindowParityTest();
void CHIRONTrcdRouteLogitsParityTest();
void CHIRONTrcdRouteLogitsBackwardParityTest();
void CHIRONTrcdGumbelGateEvalTest();
void CHIRONTrcdApplyGateParityTest();
void CHIRONTrcdLambdaPiControllerTest();
void CHIRONTrcdApplyGateConvexParityTest();
void CHIRONLcpGatherScatterRoundtripTest();
void CHIRONLcpDeltaParityTest();
void CHIRONLcpEndToEndDetailCorrectionTest();
void CHIRONLcpRoutingThroughputBenchmark();
void CHIRONIbgradProjectUnprojectParityTest();
void CHIRONIbgradQrReorthogonalizeTest();
void CHIRONIbgradApplyUpdateAndCapturedFracTest();
void CHIRONIbgradThroughputBenchmark();
void CHIRONIbgradEndToEndConvergenceTest();
void CHIRONWipIbgradMathParityTest();
void CHIRONWipIbgradE2ETest();
void CHIRONLcpIbgradCompositionTest();
void CHIRONEdtEnergyDistilledTest();
void CHIRONTrcdRoutingThroughputBenchmark();
void CHIRONTrcdEndToEndConvergenceTest();
void CHIRONZlossDisabledParityTest();
void CHIRONZlossEnabledMathTest();
void CHIRONQkNormDisabledParityTest();
void CHIRONQkNormEnabledMathTest();
void CHIRONMtpDisabledParityTest();
void CHIRONMtpTargetShiftTest();
void CHIRONLayerDropScheduleMathTest();
void CHIRONLayerDropDisabledParityTest();
void CHIRONLayerDropDeterministicMasksTest();
void CHIRONUL2SpanSamplerMeanSpanTest();
void CHIRONUL2SpanSamplerRateTest();
void CHIRONUL2DisabledParityTest();
void CHIRONSiraConfigDefaultsTest();
void CHIRONSiraDisabledParityTest();
void CHIRONSiraDiagnosticsTest();
void CHIRONSiraEnabledMathTest();
void CHIRONSiraTrainingLossTest();
void CHIRONPhsConfigDefaultsTest();
void CHIRONPhsDisabledParityTest();
void CHIRONPhsDiagnosticsMathTest();
void CHIRONPhsEmaTest();
void CHIRONPtocConfigDefaultsTest();
void CHIRONPtocDisabledParityTest();
void CHIRONPtocDiagnosticsMathTest();
void CHIRONQClampMathTest();
void CHIRONQClampEdgeTest();
void CHIRONRelnDualMirrorTest();
void CHIRONDwconvDualMirrorTest();
void CHIRONInnerVOTest();
void CHIRONGradGroupClampTest();
void CHIRONAgcClampTest();
void CHIRONGradCentralizeTest();
void CHIRONGradCentralizeBf16Test();
void CHIRONSpectralNormTest();
void CHIRONSamTest();
void CHIRONRelnBackwardBoundedTest();
void CHIRONRelnReanchorTest();
void CHIRONDriftGradCheckTest();
void CHIRONDriftCpuGpuParityTest();
void CHIRONDriftReversibilityTest();
void CHIRONDriftBackwardParityTest();
void CHIRONRotCpuTest();
void CHIRONRotGpuParityTest();
void CHIRONRotBackwardParityTest();

void WhiSCScaleCpuTest();
void WhiSCStatsCpuTest();
void WhiSCGpuParityTest();
void WhiSCBackwardParityTest();
void WhiSCFoldBackwardParityTest();
void WhiSCInvWalkBackwardParityTest();

// PIED increment dropout (2026-07-01,
// docs/superpowers/specs/2026-07-01-chiron-pied-increment-dropout-design.md).
void CHIRONPiedMaskCpuTest();
void CHIRONPiedCommitInverseCpuTest();
void CHIRONPiedGpuParityTest();
void CHIRONPiedDualPParityTest();
void CHIRONPiedDyDualParityTest();

// PACT — Profile Anti-Cancellation Tax (2026-07-04,
// docs/superpowers/specs/2026-07-04-chiron-pact-anti-cancellation-design.md).
void CHIRONPactRefMathTest();
void CHIRONPactDampSigmaParityTest();
void CHIRONPactCommitParityTest();
void CHIRONPactFieldParityTest();

// Aggregate entry point, wired into unit-tests/main.cpp.
void CHIRONUnitTest();

// Performance benchmark entry point (separate from the correctness test
// because it runs at larger sizes and reports wall-clock numbers). Wired
// into unit-tests/main.cpp as `chiron-bench`.
void CHIRONBenchmark();

#endif
