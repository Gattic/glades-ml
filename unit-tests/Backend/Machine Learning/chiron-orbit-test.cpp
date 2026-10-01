// ORBIT optimizer correctness and performance tests.
#include "chiron-orbit-test.h"
#include "../../unit-test.h"

#include "../../../Backend/Machine Learning/Networks/chiron_orbit.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#ifdef GLADES_HAVE_CUDA
#include "../../../Backend/Machine Learning/rng.h"
#include "../../../Backend/Machine Learning/Networks/cuda/gpu_orbit.h"
#include "../../../Backend/Machine Learning/Networks/cuda/gpu_kernels.h"
#include "../../../Backend/Machine Learning/Networks/cuda/gpu_device.h"
#include "../../../Backend/Machine Learning/Networks/cuda/gpu_buffer.h"
#include "../../../Backend/Machine Learning/Networks/transformer_kernels.h"
#include <cuda_runtime.h>
#endif

namespace {

struct OrbitRng
{
	unsigned int x;
	explicit OrbitRng(unsigned int seed) : x(seed) {}
	float next() { x = 1664525u*x+1013904223u; return (float)((x>>8)&0xffffffu)/(float)0x1000000u; }
};

void orbit_cpu_reference_tests()
{
	const int R=5,C=7;
	std::vector<float> g((size_t)R*C), row(R), col(C); OrbitRng rng(9u);
	for(size_t i=0;i<g.size();++i) g[i]=2.0f*rng.next()-1.0f;
	float S=0.0f;
	glades::chiron::chiron_orbit_row_col_sqsum_cpu(&g[0],R,C,0.75f,&row[0],&col[0],&S);
	double sr=0.0,sc=0.0;
	for(int i=0;i<R;++i) sr+=row[i];
	for(int j=0;j<C;++j) sc+=col[j];
	ASSERT("ORBIT CPU row scale pin",fabs(sr/R-S)<1e-6);
	ASSERT("ORBIT CPU col scale pin",fabs(sc/C-S)<1e-6);

	std::vector<float> dp((size_t)R*C), adj(C), hs(C), occ(C);
	for(size_t i=0;i<dp.size();++i) dp[i]=rng.next();
	glades::chiron::chiron_orbit_adjoint_rows_cpu(&dp[0],R,C,&adj[0]);
	glades::chiron::chiron_orbit_h_colsq_strided_cpu(&dp[0],R,C,2,&hs[0]);
	for(int j=0;j<C;++j) { ASSERT("ORBIT CPU adj finite",adj[j]>=0.0f); ASSERT("ORBIT CPU h finite",hs[j]>=0.0f); }
	std::vector<float> rank1((size_t)R*C),rankRecon((size_t)R*C);
	for(int i=0;i<R;++i)for(int j=0;j<C;++j)rank1[(size_t)i*C+j]=(0.2f+i)*(0.1f+0.3f*j);
	glades::chiron::chiron_orbit_row_col_sqsum_cpu(&rank1[0],R,C,1.0f,&row[0],&col[0],&S);
	for(int i=0;i<R;++i)for(int j=0;j<C;++j)rankRecon[(size_t)i*C+j]=row[i]*col[j]/S;
	std::vector<float> rankSq(rank1.size());for(size_t i=0;i<rank1.size();++i)rankSq[i]=rank1[i]*rank1[i];
	ASSERT("ORBIT rank-one factor correlation",glades::chiron::chiron_orbit_correlation_cpu(&rankRecon[0],&rankSq[0],rankSq.size())>0.99999);

	std::vector<float> p((size_t)R*C);
	for(int i=0;i<R;++i) { float z=0.0f; for(int j=0;j<C;++j){p[(size_t)i*C+j]=0.1f+rng.next();z+=p[(size_t)i*C+j];} for(int j=0;j<C;++j)p[(size_t)i*C+j]/=z; }
	glades::chiron::chiron_orbit_occupancy_cpu(&p[0],R,C,&occ[0]);
	for(int j=0;j<C;++j) ASSERT("ORBIT occupancy range",occ[j]>=0.0f&&occ[j]<=0.25f);
	int ids[R]={1,2,1,6,2}; std::vector<float> f(C); float dqS=0.0f;
	glades::chiron::chiron_orbit_embedding_input_cpu(ids,&dp[0],R,C,C,&f[0],&dqS);
	ASSERT("ORBIT frequency count",f[1]==2.0f&&f[2]==2.0f&&f[6]==1.0f);
	ASSERT("ORBIT dq mean square",dqS>0.0f);

	const glades::chiron::OrbitMemoryEstimate mem=
	    glades::chiron::chiron_orbit_memory_estimate(32000,2048,4096,24,150000);
	const double ratio=(double)mem.orbitBytes/(double)mem.baselineBytes;
	std::printf("  [ORBIT memory] baseline=%.3f GiB orbit=%.3f GiB ratio=%.4f saved=%.3f GiB\n",
	    (double)mem.baselineBytes/(1ull<<30),(double)mem.orbitBytes/(1ull<<30),ratio,
	    (double)(mem.baselineBytes-mem.orbitBytes)/(1ull<<30));
	ASSERT("ORBIT memory reduction below 25% bar",ratio<0.75);
}

#ifdef GLADES_HAVE_CUDA
void orbit_gpu_tests()
{
	if(!glades::gpu::initDevice()) { std::printf("  [ORBIT GPU] no device -- skipped\n"); return; }
	const int R=9,C=13; const size_t N=(size_t)R*C; OrbitRng rng(17u);
	std::vector<float> g(N); for(size_t i=0;i<N;++i)g[i]=2.0f*rng.next()-1.0f;
	std::vector<unsigned short> gb(N);
	for(size_t i=0;i<N;++i)gb[i]=glades::transformer_kernels::float_to_bf16_rn(g[i]);
	std::vector<float> gb32(N); for(size_t i=0;i<N;++i)gb32[i]=glades::transformer_kernels::bf16_to_float(gb[i]);
	glades::gpu::GpuBuffer<float> dg,re,ce,se,rs,cs,rp,cp,scalar;
	glades::gpu::GpuBuffer<unsigned short> dgb;
	dg.allocate(N);dg.upload(&g[0],N);dgb.allocate(N);dgb.upload(&gb[0],N);
	re.allocate(R);ce.allocate(C);se.allocate(1);rs.allocate(1);cs.allocate(1);
	rp.allocate(glades::gpu::orbit_row_partial_count(R,C));
	cp.allocate(glades::gpu::orbit_col_partial_count(R,C));scalar.allocate(1);
	re.zero();ce.zero();se.zero();
	ASSERT("ORBIT rowcol f32 dispatch",glades::gpu::orbit_row_col_sqsum(dg.data(),R,C,0.7f,0.0f,re.data(),ce.data(),se.data(),rs.data(),cs.data(),rp.data(),cp.data(),scalar.data()));
	std::vector<float> rg(R),cg(C),rr(R),cr(C);float sg=0,sr0=0;
	re.download(&rg[0],R);ce.download(&cg[0],C);se.download(&sg,1);
	glades::chiron::chiron_orbit_row_col_sqsum_cpu(&g[0],R,C,0.7f,&rr[0],&cr[0],&sr0);
	for(int i=0;i<R;++i)ASSERT("ORBIT row CPU/GPU",fabsf(rg[i]-rr[i])<2e-6f);
	for(int j=0;j<C;++j)ASSERT("ORBIT col CPU/GPU",fabsf(cg[j]-cr[j])<2e-6f);
	ASSERT("ORBIT scale CPU/GPU",fabsf(sg-sr0)<2e-6f);

	re.zero();ce.zero();se.zero();
	ASSERT("ORBIT rowcol bf16 dispatch",glades::gpu::orbit_row_col_sqsum_bf16(dgb.data(),R,C,1.0f,0.0f,re.data(),ce.data(),se.data(),rs.data(),cs.data(),rp.data(),cp.data(),scalar.data()));
	re.download(&rg[0],R);ce.download(&cg[0],C);se.download(&sg,1);
	glades::chiron::chiron_orbit_row_col_sqsum_cpu(&gb32[0],R,C,1.0f,&rr[0],&cr[0],&sr0);
	for(int i=0;i<R;++i)ASSERT("ORBIT bf16 row CPU/GPU",fabsf(rg[i]-rr[i])<2e-6f);
	for(int j=0;j<C;++j)ASSERT("ORBIT bf16 col CPU/GPU",fabsf(cg[j]-cr[j])<2e-6f);

	// Embedding-specialized dE column factor and tensor scale.
	ce.zero();se.zero();
	ASSERT("ORBIT embedding col dispatch",glades::gpu::orbit_col_sqsum_ema(dg.data(),R,C,1.0f,0.0f,ce.data(),se.data(),cs.data(),cp.data(),scalar.data()));
	ce.download(&cg[0],C);se.download(&sg,1);
	glades::chiron::chiron_orbit_row_col_sqsum_cpu(&g[0],R,C,1.0f,&rr[0],&cr[0],&sr0);
	for(int j=0;j<C;++j)ASSERT("ORBIT embedding col CPU/GPU",fabsf(cg[j]-cr[j])<2e-6f);
	ASSERT("ORBIT embedding scale CPU/GPU",fabsf(sg-sr0)<2e-6f);

	// Quantile damping and floor normalization are deterministic.
	std::vector<float> qv(C),qsorted(C),qfloor(C);for(int j=0;j<C;++j)qv[j]=(float)((j*7)%C)+0.25f;qsorted=qv;std::sort(qsorted.begin(),qsorted.end());
	glades::gpu::GpuBuffer<float> qbuf;qbuf.allocate(C);qbuf.upload(&qv[0],C);float qout=0;
	ASSERT("ORBIT quantile dispatch",glades::gpu::orbit_quantile(qbuf.data(),C,0.5f,scalar.data()));scalar.download(&qout,1);
	ASSERT("ORBIT quantile exact",qout==qsorted[(C-1)/2]);
	ASSERT("ORBIT floor dispatch",glades::gpu::orbit_floor_and_sum(qbuf.data(),C,scalar.data(),ce.data(),cs.data()));ce.download(&qfloor[0],C);float floorSum=0;cs.download(&floorSum,1);float floorRef=0;for(int j=0;j<C;++j){float x=std::max(qv[j],qout);floorRef+=x;ASSERT("ORBIT floor exact",qfloor[j]==x);}ASSERT("ORBIT floor sum",fabsf(floorSum-floorRef)<2e-6f);

	// Backward sample collectors and accepted-step gating.
	glades::gpu::GpuBuffer<float> dsample,dema,dsum; dsample.allocate(C);dema.allocate(C);dsum.allocate(1);dema.zero();
	ASSERT("ORBIT adjoint dispatch",glades::gpu::orbit_adjoint_rows(dg.data(),R,C,1.0f/(float)R,true,dsample.data()));
	std::vector<float> before(C),sample(C),after(C),adjRef(C);dema.download(&before[0],C);dsample.download(&sample[0],C);
	glades::chiron::chiron_orbit_adjoint_rows_cpu(&g[0],R,C,&adjRef[0]);
	for(int j=0;j<C;++j){ASSERT("ORBIT sample collector mutated EMA",before[j]==0.0f);ASSERT("ORBIT adjoint CPU/GPU",fabsf(sample[j]-adjRef[j])<2e-6f);}
	ASSERT("ORBIT accepted EMA",glades::gpu::orbit_vector_ema(dsample.data(),dema.data(),C,0.5f,dsum.data()));
	dema.download(&after[0],C);for(int j=0;j<C;++j)ASSERT("ORBIT EMA math",fabsf(after[j]-0.5f*sample[j])<1e-7f);
	ASSERT("ORBIT h collector",glades::gpu::orbit_h_colsq_strided(dg.data(),R,C,2,dsample.data()));
	std::vector<float> hRef(C);dsample.download(&sample[0],C);glades::chiron::chiron_orbit_h_colsq_strided_cpu(&g[0],R,C,2,&hRef[0]);
	for(int j=0;j<C;++j)ASSERT("ORBIT Wo activation CPU/GPU",fabsf(sample[j]-hRef[j])<2e-6f);

	std::vector<int> ids(R);for(int i=0;i<R;++i)ids[i]=(i*3)%C;
	glades::gpu::GpuBuffer<int> dids;glades::gpu::GpuBuffer<float> dfreq,dmean;dids.allocate(R);dids.upload(&ids[0],R);dfreq.allocate(C);dmean.allocate(1);
	ASSERT("ORBIT embedding input dispatch",glades::gpu::orbit_embedding_input_stats(dids.data(),dg.data(),R,C,C,dfreq.data(),dmean.data()));
	std::vector<float> freqGpu(C),freqRef(C);float meanGpu=0,meanRef=0;dfreq.download(&freqGpu[0],C);dmean.download(&meanGpu,1);
	glades::chiron::chiron_orbit_embedding_input_cpu(&ids[0],&g[0],R,C,C,&freqRef[0],&meanRef);
	for(int j=0;j<C;++j)ASSERT("ORBIT embedding frequency CPU/GPU",freqGpu[j]==freqRef[j]);
	ASSERT("ORBIT embedding dq CPU/GPU",fabsf(meanGpu-meanRef)<2e-6f);

	// Occupancy parity on bf16 probabilities.
	std::vector<unsigned short> pb(N);std::vector<float> pf(N);for(int i=0;i<R;++i){float z=0;for(int j=0;j<C;++j){pf[(size_t)i*C+j]=0.1f+rng.next();z+=pf[(size_t)i*C+j];}for(int j=0;j<C;++j){pf[(size_t)i*C+j]/=z;pb[(size_t)i*C+j]=glades::transformer_kernels::float_to_bf16_rn(pf[(size_t)i*C+j]);pf[(size_t)i*C+j]=glades::transformer_kernels::bf16_to_float(pb[(size_t)i*C+j]);}}
	glades::gpu::GpuBuffer<unsigned short> dpb;dpb.allocate(N);dpb.upload(&pb[0],N);
	ASSERT("ORBIT occupancy dispatch",glades::gpu::orbit_occupancy_bwd_bf16(dpb.data(),R,C,dsample.data()));
	std::vector<float> og(C),oref(C);dsample.download(&og[0],C);glades::chiron::chiron_orbit_occupancy_cpu(&pf[0],R,C,&oref[0]);
	for(int j=0;j<C;++j)ASSERT("ORBIT occupancy parity",fabsf(og[j]-oref[j])<2e-6f);

	// Factored step vs dense CPU reference at step 1.
	const int SR=4,SC=8;const size_t SN=(size_t)SR*SC;
	std::vector<float> p0(SN),gg(SN),mom(SN,0),rf(SR),cf(SC);for(size_t i=0;i<SN;++i){p0[i]=0.1f*(2*rng.next()-1);gg[i]=0.2f*(2*rng.next()-1);}for(int i=0;i<SR;++i)rf[i]=0.5f+i;for(int j=0;j<SC;++j)cf[j]=0.25f+0.2f*j;
	float rsum=0,csum=0;for(int i=0;i<SR;++i)rsum+=rf[i];for(int j=0;j<SC;++j)csum+=cf[j];
	const float betaF=0.98f,Strue=0.03f,Sem=Strue*(1-betaF),lr=1e-3f;
	std::vector<float> pref=p0;double metric=0;
	glades::chiron::chiron_orbit_factored_step_cpu(&pref[0],&gg[0],&mom[0],SR,SC,&rf[0],&cf[0],Sem,1-betaF,lr,0.9f,1e-8f,0.01f,1.0f,1,0,1.0f,&metric);
	glades::gpu::GpuBuffer<float> dpa,dga,drf,dcf,drs,dcs,dse,dprev,dcur;glades::gpu::GpuBuffer<int8_t> dmi;glades::gpu::GpuBuffer<float> dms;
	dpa.allocate(SN);dpa.upload(&p0[0],SN);dga.allocate(SN);dga.upload(&gg[0],SN);dmi.allocate(SN);dmi.zero();dms.allocate(1);dms.zero();drf.allocate(SR);drf.upload(&rf[0],SR);dcf.allocate(SC);dcf.upload(&cf[0],SC);drs.allocate(1);drs.upload(&rsum,1);dcs.allocate(1);dcs.upload(&csum,1);dse.allocate(1);dse.upload(&Sem,1);dprev.allocate(1);dprev.zero();dcur.allocate(1);dcur.zero();
	ASSERT("ORBIT factored step dispatch",glades::gpu::orbit_factored_step(dpa.data(),dga.data(),dmi.data(),dms.data(),drf.data(),drs.data(),dcf.data(),dcs.data(),dse.data(),SR,SC,lr,0.9f,betaF,1e-8f,0.01f,false,1.0f,1,0,0.0f,dprev.data(),dcur.data(),NULL));
	std::vector<float> pg(SN);dpa.download(&pg[0],SN);float worst=0;for(size_t i=0;i<SN;++i)worst=std::max(worst,fabsf(pg[i]-pref[i]));
	std::printf("  [ORBIT factored step CPU/GPU] worst_abs=%.3e\n",worst);ASSERT("ORBIT factored step CPU/GPU",worst<2e-6f);

	// Delayed function-space cap: prev length=2*delta must halve this step.
	std::vector<float> capRef=p0,capMom(SN,0);double capMetric=0;
	glades::chiron::chiron_orbit_factored_step_cpu(&capRef[0],&gg[0],&capMom[0],SR,SC,&rf[0],&cf[0],Sem,1-betaF,lr,0.9f,1e-8f,0.0f,1.0f,1,0,0.5f,&capMetric);
	glades::gpu::GpuBuffer<float> dcap;dcap.allocate(SN);dcap.upload(&p0[0],SN);dmi.zero();dms.zero();float prevMetric=4e-4f;dprev.upload(&prevMetric,1);dcur.zero();
	ASSERT("ORBIT delayed cap dispatch",glades::gpu::orbit_factored_step(dcap.data(),dga.data(),dmi.data(),dms.data(),drf.data(),drs.data(),dcf.data(),dcs.data(),dse.data(),SR,SC,lr,0.9f,betaF,1e-8f,0.0f,false,1.0f,1,0,0.01f,dprev.data(),dcur.data(),NULL));
	dcap.download(&pg[0],SN);worst=0;for(size_t i=0;i<SN;++i)worst=std::max(worst,fabsf(pg[i]-capRef[i]));ASSERT("ORBIT delayed cap CPU/GPU",worst<2e-6f);

	// Deterministic BF16 stochastic writeback.
	std::vector<unsigned short> pbf(SN),gbf(SN);for(size_t i=0;i<SN;++i){pbf[i]=glades::transformer_kernels::float_to_bf16_rn(p0[i]);gbf[i]=glades::transformer_kernels::float_to_bf16_rn(gg[i]);}
	glades::gpu::GpuBuffer<unsigned short> p1,p2,g1;p1.allocate(SN);p2.allocate(SN);g1.allocate(SN);p1.upload(&pbf[0],SN);p2.upload(&pbf[0],SN);g1.upload(&gbf[0],SN);glades::gpu::GpuBuffer<int8_t> m1,m2;glades::gpu::GpuBuffer<float> ms1,ms2;m1.allocate(SN);m2.allocate(SN);m1.zero();m2.zero();ms1.allocate(1);ms2.allocate(1);ms1.zero();ms2.zero();
	ASSERT("ORBIT bf16 step 1",glades::gpu::orbit_factored_step_bf16(p1.data(),g1.data(),m1.data(),ms1.data(),drf.data(),drs.data(),dcf.data(),dcs.data(),dse.data(),SR,SC,lr,0.9f,betaF,1e-8f,0.01f,false,1.0f,1,0,0,dprev.data(),dcur.data(),123u,7u,NULL));
	ASSERT("ORBIT bf16 step 2",glades::gpu::orbit_factored_step_bf16(p2.data(),g1.data(),m2.data(),ms2.data(),drf.data(),drs.data(),dcf.data(),dcs.data(),dse.data(),SR,SC,lr,0.9f,betaF,1e-8f,0.01f,false,1.0f,1,0,0,dprev.data(),dcur.data(),123u,7u,NULL));
	std::vector<unsigned short> h1(SN),h2(SN);p1.download(&h1[0],SN);p2.download(&h2[0],SN);ASSERT("ORBIT deterministic BF16 step",h1==h2);
}
#endif

} // namespace

void CHIRONOrbitUnitTest()
{
	std::printf("CHIRON ORBIT optimizer tests...\n");
	orbit_cpu_reference_tests();
#ifdef GLADES_HAVE_CUDA
	orbit_gpu_tests();
#else
	std::printf("  [ORBIT GPU] GLADES_HAVE_CUDA not defined -- skipped\n");
#endif
}

void CHIRONOrbitBenchmark()
{
#ifdef GLADES_HAVE_CUDA
	if(!glades::gpu::initDevice()){std::printf("[ORBIT bench] no CUDA device\n");return;}
	const int rows=512,cols=1024,n=rows*cols,iters=20;OrbitRng rng(99u);
	std::vector<float> h(n);for(int i=0;i<n;++i)h[i]=0.01f*(2*rng.next()-1);
	glades::gpu::GpuBuffer<float> p,g,re,ce,se,rs,cs,rp,cp,scalar;glades::gpu::GpuBuffer<int8_t> mi;glades::gpu::GpuBuffer<uint8_t> vi;glades::gpu::GpuBuffer<float> ms,vs;
	p.allocate(n);g.allocate(n);p.upload(&h[0],n);g.upload(&h[0],n);mi.allocate(n);vi.allocate(n);mi.zero();vi.zero();const int ns=glades::gpu::adam_int8_scale_count(n);ms.allocate(ns);vs.allocate(ns);ms.zero();vs.zero();re.allocate(rows);ce.allocate(cols);se.allocate(1);rs.allocate(1);cs.allocate(1);rp.allocate(glades::gpu::orbit_row_partial_count(rows,cols));cp.allocate(glades::gpu::orbit_col_partial_count(rows,cols));scalar.allocate(1);re.zero();ce.zero();se.zero();
	cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a);
	for(int t=1;t<=iters;++t)glades::gpu::adam_update_int8_state(p.data(),g.data(),mi.data(),vi.data(),ms.data(),vs.data(),1e-3f,0.9f,0.95f,1e-8f,0.01f,1.0f,t,n);
	cudaEventRecord(b);cudaEventSynchronize(b);float adamMs=0;cudaEventElapsedTime(&adamMs,a,b);
	mi.zero();ms.zero();p.upload(&h[0],n);cudaEventRecord(a);
	for(int t=1;t<=iters;++t){glades::gpu::orbit_row_col_sqsum(g.data(),rows,cols,1.0f,0.98f,re.data(),ce.data(),se.data(),rs.data(),cs.data(),rp.data(),cp.data(),scalar.data());glades::gpu::orbit_factored_step(p.data(),g.data(),mi.data(),ms.data(),re.data(),rs.data(),ce.data(),cs.data(),se.data(),rows,cols,1e-3f,0.9f,0.98f,1e-8f,0.01f,false,1.0f,t,0,0,NULL,NULL,NULL);}
	cudaEventRecord(b);cudaEventSynchronize(b);float orbitMs=0;cudaEventElapsedTime(&orbitMs,a,b);cudaEventDestroy(a);cudaEventDestroy(b);
	std::printf("[ORBIT bench] shape=%dx%d iters=%d Adam=%.3fms ORBIT(stats+step)=%.3fms optimizer_ratio=%.3f state_ratio=%.3f\n",rows,cols,iters,adamMs,orbitMs,orbitMs/adamMs,(double)(n+ns*4+4*(rows+cols+1))/(double)(2*n+2*ns*4));
#else
	std::printf("[ORBIT bench] GLADES_HAVE_CUDA not defined\n");
#endif
}
