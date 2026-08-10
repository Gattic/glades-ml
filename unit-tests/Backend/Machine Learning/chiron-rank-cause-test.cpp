#include "chiron-rank-cause-test.h"
#include "../../unit-test.h"

#ifdef GLADES_HAVE_CUDA
#include "../../../Backend/Machine Learning/Networks/cuda/gpu_buffer.h"
#include "../../../Backend/Machine Learning/Networks/cuda/gpu_device.h"
#include "../../../Backend/Machine Learning/Networks/cuda/gpu_kernels.h"
#endif

#include <cmath>
#include <cstdio>
#include <vector>

void ChironRankCauseUnitTest()
{
#ifndef GLADES_HAVE_CUDA
	std::printf("  [rank-cause GPU] CUDA disabled -- skipped\n");
#else
	if (!glades::gpu::initDevice())
	{
		std::printf("  [rank-cause GPU] no device -- skipped\n");
		return;
	}
	const int T=2,V=3,d=3,chunk=2;
	const float xHost[T*d]={1,0,1, 0,1,1};
	const float wHost[V*d]={1,0,0, 0,1,0, 0,0,0};
	const int targetHost[T]={0,0};
	glades::gpu::GpuBuffer<float> x,w,dw,scratch,loss;
	glades::gpu::GpuBuffer<int> targets,competitor,valid;
	ASSERT("rank-cause X allocate",x.allocate(T*d));
	ASSERT("rank-cause W allocate",w.allocate(V*d));
	ASSERT("rank-cause dW allocate",dw.allocate(V*d));
	ASSERT("rank-cause scratch allocate",scratch.allocate(T*chunk+3*T));
	ASSERT("rank-cause scalar allocate",loss.allocate(1)&&valid.allocate(1));
	ASSERT("rank-cause int allocate",targets.allocate(T)&&competitor.allocate(T));
	ASSERT("rank-cause upload",x.upload(xHost,T*d)&&w.upload(wHost,V*d)&&targets.upload(targetHost,T));
	ASSERT("rank-cause margin forward",glades::gpu::chunked_squared_hinge_loss(
	    x.data(),w.data(),targets.data(),T,V,d,chunk,1.0f,
	    loss.data(),valid.data(),competitor.data(),scratch.data()));
	float lossHost=0.0f;int validHost=0,competitorHost[T]={-1,-1};
	ASSERT("rank-cause forward download",loss.download(&lossHost,1)&&valid.download(&validHost,1)&&competitor.download(competitorHost,T));
	ASSERT("rank-cause margin loss",std::fabs(lossHost-4.0f)<1e-6f&&validHost==2);
	ASSERT("rank-cause competitor tie",competitorHost[0]==1&&competitorHost[1]==1);
	const float* hinge=scratch.data()+T*chunk+2*T;
	ASSERT("rank-cause margin backward",glades::gpu::chunked_squared_hinge_backward(
	    x.data(),targets.data(),competitor.data(),hinge,T,V,d,chunk,validHost,
	    false,dw.data(),scratch.data()));
	float dwHost[V*d];ASSERT("rank-cause dW download",dw.download(dwHost,V*d));
	const float expected[V*d]={0,-2,-2, 0,2,2, 0,0,0};
	for(int i=0;i<V*d;++i)ASSERT("rank-cause dW exact",std::fabs(dwHost[i]-expected[i])<1e-6f);

	const float dotHost[4]={1,2,3,4};glades::gpu::GpuBuffer<float> a,partials;
	ASSERT("rank-cause dot allocate",a.allocate(4)&&partials.allocate(4));
	ASSERT("rank-cause dot upload",a.upload(dotHost,4));
	ASSERT("rank-cause dot kernel",glades::gpu::deterministic_dot_partials(a.data(),a.data(),4,partials.data(),4));
	float partialHost[4]={0,0,0,0};ASSERT("rank-cause dot download",partials.download(partialHost,4));
	ASSERT("rank-cause dot value",std::fabs(partialHost[0]-30.0f)<1e-6f);

	const float gaugeHost[9]={0,0,1,0,0,2,0,0,3};glades::gpu::GpuBuffer<float> gauge;
	ASSERT("rank-cause gauge allocate",gauge.allocate(9)&&gauge.upload(gaugeHost,9));
	ASSERT("rank-cause gauge kernel",glades::gpu::project_augmented_bias_gauge(gauge.data(),3,3,partials.data(),partials.size()));
	float gaugeOut[9];ASSERT("rank-cause gauge download",gauge.download(gaugeOut,9));
	ASSERT("rank-cause gauge values",std::fabs(gaugeOut[2]+1.0f)<1e-6f&&std::fabs(gaugeOut[5])<1e-6f&&std::fabs(gaugeOut[8]-1.0f)<1e-6f);
	std::printf("  [rank-cause GPU] deterministic margin/vector helpers passed\n");
#endif
}
