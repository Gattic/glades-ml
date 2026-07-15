// Deterministic validation for CHIRON VITALS host estimators and CUDA taps.
#include "chiron-vitals-test.h"
#include "../../unit-test.h"
#include "../../../Backend/Machine Learning/Networks/chiron_vitals.h"

#ifdef GLADES_HAVE_CUDA
#include "../../../Backend/Machine Learning/Networks/cuda/gpu_buffer.h"
#include "../../../Backend/Machine Learning/Networks/cuda/gpu_chiron.h"
#include "../../../Backend/Machine Learning/Networks/cuda/gpu_device.h"
#include "../../../Backend/Machine Learning/Networks/cuda/gpu_kernels.h"
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

static bool near(double a, double b, double tol)
{
	return a == a && b == b && std::fabs(a - b) <= tol;
}

static glades::chiron::vitals::StepSample make_sample(int step)
{
	using namespace glades::chiron::vitals;
	StepSample s;
	s.step = step;
	s.tokensSeen = (unsigned long long)step * 1000000ULL;
	s.sequenceLength = 64;
	s.vocabularySize = 16;
	s.accumSteps = 4;
	s.loss = 4.0 - 0.02 * step;
	s.previousLoss = s.loss + 0.02;
	s.learningRate = 3e-4;
	s.clipScale = 0.5;
	// G^2=3, S=6 => ||gamma_1||^2=9, ||gbar_4||^2=4.5.
	s.microOneGradNormSq = 9.0;
	s.fullGradNormSq = 4.5;
	for (int g = 0; g < 6; ++g)
	{
		s.microOneGroupNormSq.push_back(9.0 + g);
		s.fullGroupNormSq.push_back(4.5 + 0.25 * g);
	}
	s.dqMeanSq.push_back(4.0); s.dqMeanSq.push_back(1.0); s.dqMeanSq.push_back(0.25);
	s.dpMeanSq.push_back(1.0); s.dpMeanSq.push_back(1.0); s.dpMeanSq.push_back(1.0);
	s.clampFraction.assign(3, 0.0);
	s.nonfiniteFraction.assign(3, 0.0);
	for (int i = 0; i < 128; ++i) s.observerResidual.push_back(i == 127 ? 0.2f : 0.001f);

	LayerEnergyTap e0, e1;
	e0.pBefore = 1.0; e0.pAfter = 1.3; e0.incrementEnergy = 0.45;
	e0.committedEnergy = 0.5; e0.alignment = -0.10;
	e1.pBefore = 1.3; e1.pAfter = 1.8; e1.incrementEnergy = 0.36;
	e1.committedEnergy = 0.4; e1.alignment = 0.05;
	s.energy.push_back(e0); s.energy.push_back(e1);
	s.fisherByLayer.push_back(2.0); s.fisherByLayer.push_back(1.0);

	AdamTap adam;
	adam.group = "all";
	adam.values[ADAM_COUNT] = 1000.0;
	adam.values[ADAM_M_SATURATED] = 1.0;
	adam.values[ADAM_V_SATURATED] = 2.0;
	adam.values[ADAM_DEAD_UPDATE] = 10.0;
	adam.values[ADAM_SIGN_AGREE] = 750.0;
	adam.values[ADAM_SIGNAL_POWER] = 30.0;
	adam.values[ADAM_NOISE_POWER] = 20.0;
	adam.values[ADAM_EFFICIENCY_SUM] = 600.0;
	adam.values[ADAM_NOISE_FIT] = 0.001;
	adam.values[ADAM_UPDATE_SQ] = 1.0;
	adam.values[ADAM_WEIGHT_SQ] = 100.0;
	adam.values[ADAM_REL_UPDATE_SUM] = 0.1;
	adam.values[ADAM_REL_UPDATE_MAX] = 0.02;
	s.optimizer.push_back(adam);

	s.frozenUnigram.resize(16);
	s.coverageMass.resize(16);
	for (int v = 0; v < 16; ++v)
	{
		s.frozenUnigram[(size_t)v] = (unsigned long long)(10000 / (v + 1));
		s.coverageMass[(size_t)v] = (double)(16 - v);
	}
	for (int t = 0; t < 64; ++t)
	{
		// Repeated octets exercise both bigram and 8-gram copy labels.
		s.tokens.push_back(t % 8);
		s.nll.push_back((float)(2.5 - 0.01 * step + 0.02 * (t % 7)));
		s.logZ.push_back((float)(3.0 + 0.001 * t));
		s.top1.push_back((unsigned char)((t % 3) == 0));
	}

	s.probe.valid = true;
	s.probe.canaryLoss = 2.0;
	s.probe.freshLoss = 2.2;
	s.probe.baselineGap = 0.05;
	s.probe.oldCanaryLoss = 2.05;
	s.probe.oldFreshLoss = 2.15;
	s.probe.oldBaselineGap = 0.02;
	s.probe.samBaseLoss = 2.0;
	s.probe.samPerturbedLoss = 2.1;
	s.probe.replaySpacing = 10.0;
	s.probe.atomicsVarianceFloor = 0.0;
	const float d1[] = {-0.12f,-0.10f,-0.08f,-0.11f,-0.09f};
	const float d2[] = {-0.23f,-0.20f,-0.17f,-0.22f,-0.18f};
	s.probe.deltaDt.assign(d1, d1 + 5);
	s.probe.delta2Dt.assign(d2, d2 + 5);
	const float masks[] = {1.9f,2.1f,2.0f,2.2f};
	s.probe.maskLosses.assign(masks, masks + 4);
	for (int i = 0; i < 8; ++i)
	{
		s.probe.repeatLogpCopy.push_back(-3.0f + 0.01f * i);
		s.probe.repeatLogpTruth.push_back(-2.0f);
	}
	return s;
}

} // anonymous namespace

void CHIRONVitalsCpuTest()
{
	using namespace glades::chiron::vitals;
	std::printf("  [CHIRON VITALS CPU]\n");
	std::vector<double> q; q.push_back(4); q.push_back(1); q.push_back(3); q.push_back(2);
	ASSERT("VITALS quantile median", near(quantile(q, 0.5), 2.5, 1e-12));

	double G = 0.0, S = 0.0;
	gradient_noise_pair(9.0, 4.5, G, S);
	ASSERT("VITALS GNS signal", near(G, 3.0, 1e-12));
	ASSERT("VITALS GNS noise", near(S, 6.0, 1e-12));

	std::vector<int> toks;
	for (int i = 0; i < 24; ++i) toks.push_back(i % 8);
	std::vector<unsigned char> labels = copy_labels(toks, 8);
	ASSERT("VITALS copy label first unseen", labels[7] == 0u);
	ASSERT("VITALS copy label repeat", labels[15] == 1u);

	Config cfg;
	cfg.enabled = true;
	cfg.warmupSteps = 0;
	cfg.piedDropout = 0.1f;
	Tracker tracker(cfg);
	Report r;
	for (int step = 1; step <= 10; ++step)
		r = tracker.observe(make_sample(step));

	ASSERT("V1 depth gain count", r.adjointGain.size() == 2u);
	ASSERT("V1 telescoping Lambda", near(r.depthLogGain, std::log(4.0), 1e-12));
	ASSERT("V1 dp flatness identity", r.dpFlatnessPass);
	ASSERT("V2 q999 finite", r.observerQ999 == r.observerQ999 && r.observerMax >= 0.2 - 1e-6);
	ASSERT("V3 ledger closure", r.ledgerClosurePass && r.ledgerClosureRelative < 1e-12);
	ASSERT("V3 mask calibration", r.maskCalibrationPass && near(r.maskCalibration, 0.9 / 0.81, 1e-12));
	ASSERT("V4 clip ledger finite", r.signalClippedFraction > 0.0 && r.signalClippedFraction < 1.0);
	ASSERT("V5 Adam stats", near(r.adamSignAgreement, 0.75, 1e-12));
	ASSERT("V6 noise scale", near(r.noiseScale, 2.0, 1e-12));
	ASSERT("V6 per-group noise", r.groupNoiseScale.size() == 6u && near(r.groupNoiseScale[0], 2.0, 1e-12));
	ASSERT("V7 efficiency", near(r.optimizerEfficiency, 0.6, 1e-12));
	ASSERT("V7 ledger accumulates", r.memorizationLedger > 0.009);
	ASSERT("V8 susceptibility", near(r.piedSusceptibility, 1.0 / 3.0, 1e-6));
	ASSERT("V8 MC variance", r.piedMcVariance > 0.0);
	ASSERT("V9 SAM sharpness", near(r.samSharpness, 0.1, 1e-12));
	ASSERT("V10 strata", r.strata.size() == (size_t)kStrata);
	ASSERT("V10 mixture identity", r.mixtureReconstructionPass);
	ASSERT("V11 utility", r.tokenUtilityPerBillion > 0.0);
	ASSERT("V11 slope identity", r.slopeReconstructionPass);
	ASSERT("V12 copy gap", r.copyGap == r.copyGap);
	ASSERT("V13 canary gap", near(r.memorizationGap, 0.15, 1e-12));
	ASSERT("V14 Peclet", r.peclet == r.peclet && r.transportDrift < 0.0);
	ASSERT("V15 coverage PR", r.coverageParticipation > 0.0 && r.coverageParticipation <= 1.0);
	ASSERT("V16 tail/logZ/z", r.hillTailIndex > 0.0 && r.logZQ99 > r.logZMean && r.batchZ == r.batchZ);
	ASSERT("V17 copy gain", r.copyGainShare == r.copyGainShare && r.repeatPreference < 0.0);
	ASSERT("V18 contextual gain", r.contextualGain == r.contextualGain);
}

void CHIRONVitalsGpuTapTest()
{
#ifdef GLADES_HAVE_CUDA
	if (!glades::gpu::initDevice() || !glades::gpu::isAvailable())
	{
		std::printf("  [CHIRON VITALS GPU] no CUDA device — skipped\n");
		return;
	}
	std::printf("  [CHIRON VITALS GPU]\n");
	const int n = 512;
	std::vector<float> p(n), a(n), b(n), dp(n);
	for (int i = 0; i < n; ++i)
	{
		p[i] = 0.01f * (float)(i % 17 - 8);
		a[i] = 0.001f * (float)(i % 11 - 5);
		b[i] = 0.002f * (float)(i % 7 - 3);
		dp[i] = 0.003f * (float)(i % 13 - 6);
	}
	glades::gpu::GpuBuffer<float> dP0,dP1,dA,dB,dDp,dStats,dFisher;
	glades::gpu::GpuBuffer<unsigned short> dPbf0,dPbf1;
	dP0.allocate(n); dP1.allocate(n); dA.allocate(n); dB.allocate(n); dDp.allocate(n);
	dPbf0.allocate(n); dPbf1.allocate(n); dStats.allocate(6); dFisher.allocate(1);
	dP0.upload(&p[0],n); dP1.upload(&p[0],n); dA.upload(&a[0],n); dB.upload(&b[0],n); dDp.upload(&dp[0],n);
	dStats.zero(); dFisher.zero();
	const unsigned int key=0x12345678u, thr=0x19999999u;
	ASSERT("VITALS baseline commit", glades::gpu::chiron_scfa_axpy2_masked_dual_p(
	    dP0.data(),dPbf0.data(),1.0f,dA.data(),dB.data(),n,key,thr,0.0f,1.1111111f,7u,9u));
	ASSERT("VITALS fused commit", glades::gpu::chiron_scfa_axpy2_vitals(
	    dP1.data(),dPbf1.data(),1.0f,dA.data(),dB.data(),n,true,key,thr,0.0f,1.1111111f,
	    7u,9u,dStats.data(),NULL,NULL));
	std::vector<float> hp0(n),hp1(n); std::vector<unsigned short> hb0(n),hb1(n);
	dP0.download(&hp0[0],n); dP1.download(&hp1[0],n); dPbf0.download(&hb0[0],n); dPbf1.download(&hb1[0],n);
	ASSERT("VITALS commit FP32 bit parity", std::memcmp(&hp0[0],&hp1[0],n*sizeof(float))==0);
	ASSERT("VITALS commit BF16 bit parity", std::memcmp(&hb0[0],&hb1[0],n*sizeof(unsigned short))==0);
	float est[6]={0}; dStats.download(est,6);
	ASSERT("VITALS commit count", near(est[0],n,0.5));
	// Production integration observes the canonical commit after it runs;
	// prove the read-only tap cannot perturb p or its BF16 mirror.
	dStats.zero();
	ASSERT("VITALS read-only commit tap", glades::gpu::chiron_scfa_axpy2_vitals(
	    dP1.data(),NULL,1.0f,dA.data(),dB.data(),n,true,key,thr,0.0f,1.1111111f,
	    0u,0u,dStats.data(),NULL,NULL,true));
	std::vector<float> hpReadOnly(n); dP1.download(&hpReadOnly[0],n);
	ASSERT("VITALS read-only p parity", std::memcmp(&hp1[0],&hpReadOnly[0],n*sizeof(float))==0);

	// Inverse telemetry produces Fisher while returning to the original p.
	ASSERT("VITALS inverse fisher", glades::gpu::chiron_scfa_axpy2_vitals(
	    dP1.data(),NULL,-1.0f,dA.data(),dB.data(),n,true,key,thr,0.0f,1.1111111f,
	    0u,0u,NULL,dDp.data(),dFisher.data()));
	float hf=0.0f; dFisher.download(&hf,1);
	double fref=0.0; for(int i=0;i<n;++i){double y=a[i]+b[i]; fref+=y*y*dp[i]*dp[i];}
	ASSERT("VITALS Fisher parity", near(hf,fref,1e-6));

	// V2 residual.
	const int T=8;
	std::vector<float> saved(2*T), split(2*T);
	for(int t=0;t<T;++t){saved[2*t]=0.1f*t;saved[2*t+1]=2.0f;split[t]=0.1f*t;split[T+t]=0.5f;}
	split[3]+=0.2f; split[4]+=0.4f;
	glades::gpu::GpuBuffer<float> dSaved,dSplit,dRes; dSaved.allocate(2*T);dSplit.allocate(2*T);dRes.allocate(T);
	dSaved.upload(&saved[0],2*T);dSplit.upload(&split[0],2*T);
	ASSERT("VITALS residual launch",glades::gpu::chiron_vitals_reanchor_residual(dSaved.data(),dSplit.data(),T,dRes.data()));
	std::vector<float> hr(T);dRes.download(&hr[0],T);
	ASSERT("VITALS residual identity",near(hr[0],0.0,1e-7));
	ASSERT("VITALS residual drift",near(hr[3],0.1,1e-5));
	dRes.zero();
	ASSERT("VITALS sampled residual launch",glades::gpu::chiron_vitals_reanchor_residual(
	    dSaved.data(),dSplit.data(),T,dRes.data(),2));
	std::vector<float> hrs(4);dRes.download(&hrs[0],4);
	ASSERT("VITALS sampled residual indexing",near(hrs[2],0.2,1e-5));

	// V1 clamp tap and V15 coverage.
	glades::gpu::GpuBuffer<float> dRows,dSq,dCoverage;
	glades::gpu::GpuBuffer<int> dCounts,dIds;
	std::vector<float> rows(32,1.0f); rows[0]=4.0f; std::vector<int> ids(4); for(int i=0;i<4;++i)ids[i]=i;
	dRows.allocate(32);dRows.upload(&rows[0],32);dSq.allocate(1);dSq.zero();dCounts.allocate(2);dCounts.zero();
	ASSERT("VITALS row clamp",glades::gpu::row_rms_clamp_vitals(dRows.data(),4,8,2.0f,dCounts.data(),dCounts.data()+1,dSq.data()));
	float hsq=0;dSq.download(&hsq,1);ASSERT("VITALS raw row sumsq",near(hsq,47.0,1e-5));
	dIds.allocate(4);dIds.upload(&ids[0],4);dCoverage.allocate(8);dCoverage.zero();
	ASSERT("VITALS coverage launch",glades::gpu::embedding_coverage_accumulate(dIds.data(),dRows.data(),4,8,8,dCoverage.data()));
	std::vector<float> hcov(8);dCoverage.download(&hcov[0],8);ASSERT("VITALS coverage positive",hcov[0]>0&&hcov[4]==0);

	// V10/V16 vectors and scalar counts share the shipped scans.
	const int TV=4,V=8; std::vector<float> probs(TV*V,0.01f);std::vector<int> targets(TV);
	for(int t=0;t<TV;++t){targets[t]=t;probs[t*V+t]=0.8f;}
	glades::gpu::GpuBuffer<float> dProbs,dNll,dLoss;
	glades::gpu::GpuBuffer<unsigned short> dProbsBf;
	glades::gpu::GpuBuffer<unsigned char> dTop;
	glades::gpu::GpuBuffer<int> dTargets,dLossCount,dCorrect,dValid;
	dProbs.allocate(TV*V);dProbs.upload(&probs[0],TV*V);dProbsBf.allocate(TV*V);
	glades::gpu::cast_f32_to_bf16(dProbs.data(),dProbsBf.data(),TV*V);
	dTargets.allocate(TV);dTargets.upload(&targets[0],TV);dNll.allocate(TV);dTop.allocate(TV);
	dLoss.allocate(1);dLossCount.allocate(1);dCorrect.allocate(1);dValid.allocate(1);
	ASSERT("VITALS output vectors",glades::gpu::chiron_vitals_output_vectors_bf16(
	    dProbsBf.data(),dTargets.data(),TV,V,-1,dLoss.data(),dLossCount.data(),dCorrect.data(),dValid.data(),dNll.data(),dTop.data()));
	int hlc=0,hc=0,hv=0;dLossCount.download(&hlc,1);dCorrect.download(&hc,1);dValid.download(&hv,1);
	ASSERT("VITALS output counts",hlc==TV&&hc==TV&&hv==TV);

	// V5/V7 int8 Adam telemetry is mathematically side-effect free.
	const int na=512, ns=glades::gpu::adam_int8_scale_count(na);
	std::vector<float> par(na,0.5f),grad(na);for(int i=0;i<na;++i)grad[i]=0.01f*(float)(i%9-4);
	glades::gpu::GpuBuffer<float> dParA,dParB,dGrad,dMSA,dVSA,dMSB,dVSB,dAdamStats;
	glades::gpu::GpuBuffer<signed char> dMIA,dMIB; glades::gpu::GpuBuffer<unsigned char> dVIA,dVIB;
	dParA.allocate(na);dParB.allocate(na);dGrad.allocate(na);dMSA.allocate(ns);dVSA.allocate(ns);dMSB.allocate(ns);dVSB.allocate(ns);
	dMIA.allocate(na);dMIB.allocate(na);dVIA.allocate(na);dVIB.allocate(na);dAdamStats.allocate(16);
	dParA.upload(&par[0],na);dParB.upload(&par[0],na);dGrad.upload(&grad[0],na);
	dMSA.zero();dVSA.zero();dMSB.zero();dVSB.zero();dMIA.zero();dMIB.zero();dVIA.zero();dVIB.zero();dAdamStats.zero();
	ASSERT("VITALS Adam base",glades::gpu::adam_update_int8_state(dParA.data(),dGrad.data(),
	    (int8_t*)dMIA.data(),(uint8_t*)dVIA.data(),dMSA.data(),dVSA.data(),3e-4f,0.9f,0.95f,1e-8f,0.01f,1.0f,1,na));
	ASSERT("VITALS Adam tapped",glades::gpu::adam_update_int8_state(dParB.data(),dGrad.data(),
	    (int8_t*)dMIB.data(),(uint8_t*)dVIB.data(),dMSB.data(),dVSB.data(),3e-4f,0.9f,0.95f,1e-8f,0.01f,1.0f,1,na,dAdamStats.data()));
	std::vector<float> pa(na),pb(na),ast(16);dParA.download(&pa[0],na);dParB.download(&pb[0],na);dAdamStats.download(&ast[0],16);
	ASSERT("VITALS Adam param bit parity",std::memcmp(&pa[0],&pb[0],na*sizeof(float))==0);
	ASSERT("VITALS Adam count",near(ast[0],na,0.5));
#else
	std::printf("  [CHIRON VITALS GPU] built without CUDA — skipped\n");
#endif
}

void CHIRONVitalsUnitTest()
{
	CHIRONVitalsCpuTest();
	CHIRONVitalsGpuTapTest();
}
