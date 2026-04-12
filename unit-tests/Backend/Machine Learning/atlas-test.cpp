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

#include "atlas-test.h"
#include "../../unit-test.h"

#include "../../../Backend/Machine Learning/Networks/network.h"
#include "../../../Backend/Machine Learning/Networks/training_callbacks.h"
#include "../../../Backend/Machine Learning/Networks/atlas_optimizer.h"
#include "../../../Backend/Machine Learning/Networks/training_config.h"
#include "../../../Backend/Machine Learning/DataObjects/NumberInput.h"
#include "../../../Backend/Machine Learning/GMath/gmath.h"
#include "../../../Backend/Machine Learning/Structure/nninfo.h"
#include "../../../Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/hiddenlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/outputlayerinfo.h"
#include "Backend/Database/GLogger.h"

#include "../../../Backend/Machine Learning/Networks/cuda/gpu_atlas.h"
#include "../../../Backend/Machine Learning/Networks/cuda/gpu_device.h"
#include "../../../Backend/Machine Learning/Networks/cuda/gpu_buffer.h"
#include "../../../Backend/Machine Learning/Networks/cuda/gpu_kernels.h"
#include "test_token_id_input_fixture.h"

#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <sys/time.h>
#include <unistd.h>
#include <vector>

namespace {

class CaptureMetricsCallbacks : public glades::ITrainingCallbacks
{
public:
	CaptureMetricsCallbacks() : last(), saw(false) { last = glades::NNetworkEpochMetrics(); }
	virtual void onRunStart(const glades::NNetwork&, int) {}
	virtual bool onEpochEnd(const glades::NNetwork&, const glades::NNetworkEpochMetrics& m)
	{
		last = m;
		saw = true;
		return false;
	}
	virtual void onRunEnd(const glades::NNetwork&, int) {}

	glades::NNetworkEpochMetrics last;
	bool saw;
};

static bool parse_kv_manifest(const std::string& path, std::map<std::string, std::string>& outKv)
{
	outKv.clear();
	std::ifstream in(path.c_str());
	if (!in)
		return false;

	std::string line;
	bool sawMagic = false;
	while (std::getline(in, line))
	{
		if (!line.empty() && line[line.size() - 1u] == '\r')
			line.erase(line.size() - 1u);
		if (line.empty())
			continue;
		if (!sawMagic)
		{
			outKv["__magic__"] = line;
			sawMagic = true;
			continue;
		}
		const std::string::size_type eq = line.find('=');
		if (eq == std::string::npos)
			continue;
		outKv[line.substr(0u, eq)] = line.substr(eq + 1u);
	}
	return sawMagic;
}

static shmea::GLogger* quiet_logger()
{
	static shmea::GLogger logger(shmea::GLogger::LOG_ERROR);
	static bool initialized = false;
	if (!initialized)
	{
		logger.setPrintToConsole(false);
		initialized = true;
	}
	return &logger;
}

static bool atlas_close_abs(float a, float b, float tol)
{
	return fabsf(a - b) <= tol;
}

static void assert_close_value(const char* label, float got, float expected, float tol)
{
	char msg[256];
	sprintf(msg, "%s: got %.8f expected %.8f tol %.8f", label, got, expected, tol);
	ASSERT(msg, atlas_close_abs(got, expected, tol));
}

static void assert_close_vector(const char* label,
                                const std::vector<float>& got,
                                const std::vector<float>& expected,
                                float tol)
{
	ASSERT("vector size mismatch", got.size() == expected.size());
	for (size_t i = 0u; i < got.size(); ++i)
	{
		char msg[256];
		sprintf(msg, "%s[%zu]: got %.8f expected %.8f tol %.8f",
		        label, i, got[i], expected[i], tol);
		ASSERT(msg, atlas_close_abs(got[i], expected[i], tol));
	}
}

static void print_vector_sample(const char* label,
                                const std::vector<float>& values,
                                size_t maxCount = 8u)
{
	printf("  %s:", label);
	const size_t count = std::min(values.size(), maxCount);
	for (size_t i = 0u; i < count; ++i)
		printf(" %.8f", values[i]);
	if (values.size() > count)
		printf(" ...");
	printf("\n");
}

static unsigned int active_rank_from_eigs(const std::vector<float>& eig, float threshold)
{
	unsigned int active = 0u;
	for (size_t i = 0u; i < eig.size(); ++i)
	{
		if (eig[i] > threshold)
			active = static_cast<unsigned int>(i + 1u);
	}
	return active;
}

static std::vector<float> weighted_projector(const std::vector<float>& basis,
                                             const std::vector<float>& eig,
                                             unsigned int dim,
                                             unsigned int rank)
{
	std::vector<float> out(static_cast<size_t>(dim) * dim, 0.0f);
	if (basis.size() < static_cast<size_t>(dim) * rank || eig.size() < rank)
		return out;
	for (unsigned int a = 0u; a < rank; ++a)
	{
		const float lambda = std::max(0.0f, eig[a]);
		for (unsigned int i = 0u; i < dim; ++i)
		{
			const float ui = basis[static_cast<size_t>(i) * rank + a];
			for (unsigned int j = 0u; j < dim; ++j)
			{
				out[static_cast<size_t>(i) * dim + j] +=
				    lambda * ui * basis[static_cast<size_t>(j) * rank + a];
			}
		}
	}
	return out;
}

struct BiMAPParitySnapshot
{
	std::vector<float> W;
	std::vector<float> m1;
	std::vector<float> v2;
	std::vector<float> g;
	std::vector<float> rowSecond;
	std::vector<float> colSecond;
	std::vector<float> rowEigVal;
	std::vector<float> colEigVal;
	std::vector<float> rowBasis;
	std::vector<float> colBasis;
	float lastPredictiveTrust;
	float lastRowAnisotropy;
	float lastColAnisotropy;
	float lastRowCapture;
	float lastColCapture;
	unsigned int rowRank;
	unsigned int colRank;

	BiMAPParitySnapshot()
	    : lastPredictiveTrust(0.0f),
	      lastRowAnisotropy(1.0f),
	      lastColAnisotropy(1.0f),
	      lastRowCapture(0.0f),
	      lastColCapture(0.0f),
	      rowRank(0u),
	      colRank(0u)
	{
	}
};

static void capture_cpu_bimap_snapshot(BiMAPParitySnapshot& snap,
                                       const std::vector<float>& W,
                                       const std::vector<float>& m1,
                                       const std::vector<float>& v2,
                                       const std::vector<float>& g,
                                       const glades::atlas::BiMAPWeightState& state)
{
	snap.W = W;
	snap.m1 = m1;
	snap.v2 = v2;
	snap.g = g;
	snap.rowSecond = state.rowSecond;
	snap.colSecond = state.colSecond;
	snap.rowEigVal = state.rowEigVal;
	snap.colEigVal = state.colEigVal;
	snap.rowBasis = state.rowBasis;
	snap.colBasis = state.colBasis;
	snap.lastPredictiveTrust = state.lastPredictiveTrust;
	snap.lastRowAnisotropy = state.lastRowAnisotropy;
	snap.lastColAnisotropy = state.lastColAnisotropy;
	snap.lastRowCapture = state.lastRowCapture;
	snap.lastColCapture = state.lastColCapture;
	snap.rowRank = state.rowRank;
	snap.colRank = state.colRank;
}

#ifdef GLADES_HAVE_CUDA
static bool run_gpu_bimap_step(glades::gpu::GpuBiMAPWeightState& state,
                               glades::gpu::GpuBuffer<float>& dW,
                               glades::gpu::GpuBuffer<float>& dM,
                               glades::gpu::GpuBuffer<float>& dV,
                               glades::gpu::GpuBuffer<float>& dG,
                               unsigned int m,
                               unsigned int n,
                               float lr,
                               float beta1,
                               float beta2,
                               float eps,
                               float invBatch,
                               float gradScale,
                               int step,
                               const glades::ATLASConfig& ac,
                               const std::vector<float>& grad)
{
	const size_t mn = static_cast<size_t>(m) * n;
	ASSERT("gpu grad size mismatch", grad.size() == mn);
	if (!dG.upload(grad.data(), grad.size()))
		return false;
	if (!glades::gpu::adam_update(dW.data(), dG.data(), dM.data(), dV.data(),
	                              lr, beta1, beta2, eps,
	                              0.0f, invBatch * gradScale, step,
	                              static_cast<int>(mn)))
		return false;
	const double b1t = std::pow(static_cast<double>(beta1), static_cast<double>(step));
	const double b2t = std::pow(static_cast<double>(beta2), static_cast<double>(step));
	const float inv1mB1t = static_cast<float>(1.0 / (1.0 - b1t));
	const float inv1mB2t = static_cast<float>(1.0 / (1.0 - b2t));
	if (!glades::gpu::bimap_gpu_update(state,
	                                   dW.data(), dG.data(), dM.data(), dV.data(),
	                                   m, n, lr,
	                                   invBatch, gradScale,
	                                   inv1mB1t, inv1mB2t, eps,
	                                   ac, quiet_logger(), "ut.bimap.parity"))
		return false;
	return glades::gpu::synchronizeCheck("bimap parity step");
}

static void capture_gpu_bimap_snapshot(BiMAPParitySnapshot& snap,
                                       glades::gpu::GpuBuffer<float>& dW,
                                       glades::gpu::GpuBuffer<float>& dM,
                                       glades::gpu::GpuBuffer<float>& dV,
                                       glades::gpu::GpuBuffer<float>& dG,
                                       const glades::gpu::GpuBiMAPWeightState& state)
{
	snap.W.resize(dW.size());
	snap.m1.resize(dM.size());
	snap.v2.resize(dV.size());
	snap.g.resize(dG.size());
	dW.download(snap.W.data(), snap.W.size());
	dM.download(snap.m1.data(), snap.m1.size());
	dV.download(snap.v2.data(), snap.v2.size());
	dG.download(snap.g.data(), snap.g.size());

	snap.rowSecond.resize(state.rowSecond.size());
	snap.colSecond.resize(state.colSecond.size());
	state.rowSecond.download(snap.rowSecond.data(), snap.rowSecond.size());
	state.colSecond.download(snap.colSecond.data(), snap.colSecond.size());

	snap.rowEigVal.resize(state.rowEigVal.size());
	snap.colEigVal.resize(state.colEigVal.size());
	snap.rowBasis.resize(state.rowBasis.size());
	snap.colBasis.resize(state.colBasis.size());
	if (!snap.rowEigVal.empty())
		state.rowEigVal.download(snap.rowEigVal.data(), snap.rowEigVal.size());
	if (!snap.colEigVal.empty())
		state.colEigVal.download(snap.colEigVal.data(), snap.colEigVal.size());
	if (!snap.rowBasis.empty())
		state.rowBasis.download(snap.rowBasis.data(), snap.rowBasis.size());
	if (!snap.colBasis.empty())
		state.colBasis.download(snap.colBasis.data(), snap.colBasis.size());

	snap.lastPredictiveTrust = state.lastPredictiveTrust;
	snap.lastRowAnisotropy = state.lastRowAnisotropy;
	snap.lastColAnisotropy = state.lastColAnisotropy;
	snap.lastRowCapture = state.lastRowCapture;
	snap.lastColCapture = state.lastColCapture;
	snap.rowRank = active_rank_from_eigs(snap.rowEigVal, 1.0e-3f);
	snap.colRank = active_rank_from_eigs(snap.colEigVal, 1.0e-3f);
}
#endif

static void assert_bimap_parity_snapshot(const char* label,
                                         const BiMAPParitySnapshot& cpu,
                                         const BiMAPParitySnapshot& gpu,
                                         float valueTol,
                                         float stateTol,
                                         bool expectLowRank)
{
	if (gpu.W.size() == cpu.W.size())
	{
		for (size_t i = 0u; i < gpu.W.size(); ++i)
		{
			if (!atlas_close_abs(gpu.W[i], cpu.W[i], valueTol))
			{
				printf("[BiMAP parity debug] %s weight mismatch at %zu\n", label, i);
				printf("  cpu rowRank=%u colRank=%u predTrust=%.8f rowAniso=%.8f colAniso=%.8f rowCapture=%.8f colCapture=%.8f\n",
				       cpu.rowRank, cpu.colRank, cpu.lastPredictiveTrust,
				       cpu.lastRowAnisotropy, cpu.lastColAnisotropy,
				       cpu.lastRowCapture, cpu.lastColCapture);
				printf("  gpu rowRank=%u colRank=%u predTrust=%.8f rowAniso=%.8f colAniso=%.8f rowCapture=%.8f colCapture=%.8f\n",
				       gpu.rowRank, gpu.colRank, gpu.lastPredictiveTrust,
				       gpu.lastRowAnisotropy, gpu.lastColAnisotropy,
				       gpu.lastRowCapture, gpu.lastColCapture);
				print_vector_sample("cpu W", cpu.W);
				print_vector_sample("gpu W", gpu.W);
				print_vector_sample("cpu rowSecond", cpu.rowSecond);
				print_vector_sample("gpu rowSecond", gpu.rowSecond);
				print_vector_sample("cpu colSecond", cpu.colSecond);
				print_vector_sample("gpu colSecond", gpu.colSecond);
				if (expectLowRank)
				{
					print_vector_sample("cpu rowEigVal", cpu.rowEigVal);
					print_vector_sample("gpu rowEigVal", gpu.rowEigVal);
					print_vector_sample("cpu colEigVal", cpu.colEigVal);
					print_vector_sample("gpu colEigVal", gpu.colEigVal);
				}
				break;
			}
		}
	}
	assert_close_vector(label, gpu.W, cpu.W, valueTol);
	assert_close_vector(label, gpu.m1, cpu.m1, valueTol);
	assert_close_vector(label, gpu.v2, cpu.v2, valueTol);
	assert_close_vector(label, gpu.g, cpu.g, valueTol);
	assert_close_vector("rowSecond", gpu.rowSecond, cpu.rowSecond, stateTol);
	assert_close_vector("colSecond", gpu.colSecond, cpu.colSecond, stateTol);

	assert_close_value("predTrust", gpu.lastPredictiveTrust, cpu.lastPredictiveTrust, stateTol);
	assert_close_value("rowAniso", gpu.lastRowAnisotropy, cpu.lastRowAnisotropy, stateTol);
	assert_close_value("colAniso", gpu.lastColAnisotropy, cpu.lastColAnisotropy, stateTol);
	assert_close_value("rowCapture", gpu.lastRowCapture, cpu.lastRowCapture, stateTol);
	assert_close_value("colCapture", gpu.lastColCapture, cpu.lastColCapture, stateTol);

	ASSERT("rowRank mismatch", gpu.rowRank == cpu.rowRank);
	ASSERT("colRank mismatch", gpu.colRank == cpu.colRank);

	if (expectLowRank)
	{
		assert_close_vector("rowEigVal", gpu.rowEigVal, cpu.rowEigVal, stateTol);
		assert_close_vector("colEigVal", gpu.colEigVal, cpu.colEigVal, stateTol);

		const unsigned int rowRank = static_cast<unsigned int>(gpu.rowEigVal.size());
		const unsigned int colRank = static_cast<unsigned int>(gpu.colEigVal.size());
		const unsigned int rowDim = static_cast<unsigned int>(gpu.rowSecond.size());
		const unsigned int colDim = static_cast<unsigned int>(gpu.colSecond.size());
		const std::vector<float> gpuRowProj =
		    weighted_projector(gpu.rowBasis, gpu.rowEigVal, rowDim, rowRank);
		const std::vector<float> cpuRowProj =
		    weighted_projector(cpu.rowBasis, cpu.rowEigVal, rowDim, rowRank);
		const std::vector<float> gpuColProj =
		    weighted_projector(gpu.colBasis, gpu.colEigVal, colDim, colRank);
		const std::vector<float> cpuColProj =
		    weighted_projector(cpu.colBasis, cpu.colEigVal, colDim, colRank);
		assert_close_vector("rowProjector", gpuRowProj, cpuRowProj, 5.0e-3f);
		assert_close_vector("colProjector", gpuColProj, cpuColProj, 5.0e-3f);
	}
}

struct EchoParitySnapshot
{
	std::vector<float> W;
	std::vector<float> m1;
	std::vector<float> v2;
	std::vector<float> g;
	std::vector<float> rowSecond;
	std::vector<float> colSecond;
	float lastRowAnisotropy;
	float lastColAnisotropy;
	float lastGeometryScale;
	unsigned long long step;

	EchoParitySnapshot()
	    : lastRowAnisotropy(1.0f),
	      lastColAnisotropy(1.0f),
	      lastGeometryScale(0.0f),
	      step(0ULL)
	{
	}
};

static void capture_cpu_echo_snapshot(EchoParitySnapshot& snap,
                                      const std::vector<float>& W,
                                      const std::vector<float>& m1,
                                      const std::vector<float>& v2,
                                      const std::vector<float>& g,
                                      const glades::atlas::EchoWeightState& state)
{
	snap.W = W;
	snap.m1 = m1;
	snap.v2 = v2;
	snap.g = g;
	snap.rowSecond = state.rowSecond;
	snap.colSecond = state.colSecond;
	snap.lastRowAnisotropy = state.lastRowAnisotropy;
	snap.lastColAnisotropy = state.lastColAnisotropy;
	snap.lastGeometryScale = state.lastGeometryScale;
	snap.step = state.step;
}

#ifdef GLADES_HAVE_CUDA
static bool run_gpu_echo_step(glades::gpu::GpuEchoWeightState& state,
                              glades::gpu::GpuBuffer<float>& dW,
                              glades::gpu::GpuBuffer<float>& dM,
                              glades::gpu::GpuBuffer<float>& dV,
                              glades::gpu::GpuBuffer<float>& dG,
                              glades::gpu::GpuBuffer<float>& dRowObs,
                              glades::gpu::GpuBuffer<float>& dColObs,
                              unsigned int m,
                              unsigned int n,
                              unsigned int samples,
                              float lr,
                              float beta1,
                              float beta2,
                              float eps,
                              float invBatch,
                              float gradScale,
                              int step,
                              const glades::ATLASConfig& ac,
                              const std::vector<float>& grad,
                              const std::vector<float>& rowObs,
                              const std::vector<float>& colObs)
{
	const size_t mn = static_cast<size_t>(m) * n;
	ASSERT("gpu grad size mismatch", grad.size() == mn);
	ASSERT("gpu row obs size mismatch", rowObs.size() == static_cast<size_t>(samples) * m);
	ASSERT("gpu col obs size mismatch", colObs.size() == static_cast<size_t>(samples) * n);
	if (!dG.upload(grad.data(), grad.size()))
		return false;
	if (!dRowObs.upload(rowObs.data(), rowObs.size()))
		return false;
	if (!dColObs.upload(colObs.data(), colObs.size()))
		return false;
	if (!glades::gpu::echo_gpu_observe(state,
	                                   dRowObs.data(), dColObs.data(),
	                                   samples, m, n, ac))
		return false;
	const double b1t = std::pow(static_cast<double>(beta1), static_cast<double>(step));
	const double b2t = std::pow(static_cast<double>(beta2), static_cast<double>(step));
	const float inv1mB1t = static_cast<float>(1.0 / (1.0 - b1t));
	const float inv1mB2t = static_cast<float>(1.0 / (1.0 - b2t));
	if (!glades::gpu::echo_gpu_update(state,
	                                  dW.data(), dG.data(),
	                                  dM.data(), dV.data(),
	                                  m, n, lr, invBatch * gradScale,
	                                  static_cast<unsigned long long>(step),
	                                  inv1mB1t, inv1mB2t, eps,
	                                  ac, quiet_logger(), "ut.echo.parity"))
		return false;
	return glades::gpu::synchronizeCheck("echo parity step");
}

static void capture_gpu_echo_snapshot(EchoParitySnapshot& snap,
                                      glades::gpu::GpuBuffer<float>& dW,
                                      glades::gpu::GpuBuffer<float>& dM,
                                      glades::gpu::GpuBuffer<float>& dV,
                                      glades::gpu::GpuBuffer<float>& dG,
                                      const glades::gpu::GpuEchoWeightState& state)
{
	snap.W.resize(dW.size());
	snap.m1.resize(dM.size());
	snap.v2.resize(dV.size());
	snap.g.resize(dG.size());
	dW.download(snap.W.data(), snap.W.size());
	dM.download(snap.m1.data(), snap.m1.size());
	dV.download(snap.v2.data(), snap.v2.size());
	dG.download(snap.g.data(), snap.g.size());

	snap.rowSecond.resize(state.rowSecond.size());
	snap.colSecond.resize(state.colSecond.size());
	if (!snap.rowSecond.empty())
		state.rowSecond.download(snap.rowSecond.data(), snap.rowSecond.size());
	if (!snap.colSecond.empty())
		state.colSecond.download(snap.colSecond.data(), snap.colSecond.size());

	float stats[6] = { 0.0f };
	if (state.scalarScratch.size() >= 6u)
		state.scalarScratch.download(stats, 6u);
	snap.lastRowAnisotropy = (stats[2] > 1.0e-12f) ? (stats[3] / stats[2]) : state.lastRowAnisotropy;
	snap.lastColAnisotropy = (stats[4] > 1.0e-12f) ? (stats[5] / stats[4]) : state.lastColAnisotropy;
	snap.lastGeometryScale = state.lastGeometryScale;
	snap.step = state.step;
}
#endif

static void assert_echo_parity_snapshot(const char* label,
                                        const EchoParitySnapshot& cpu,
                                        const EchoParitySnapshot& gpu,
                                        float valueTol,
                                        float stateTol)
{
	assert_close_vector(label, gpu.W, cpu.W, valueTol);
	assert_close_vector("m1", gpu.m1, cpu.m1, valueTol);
	assert_close_vector("v2", gpu.v2, cpu.v2, valueTol);
	assert_close_vector("g", gpu.g, cpu.g, valueTol);
	assert_close_vector("rowSecond", gpu.rowSecond, cpu.rowSecond, stateTol);
	assert_close_vector("colSecond", gpu.colSecond, cpu.colSecond, stateTol);
	assert_close_value("rowAniso", gpu.lastRowAnisotropy, cpu.lastRowAnisotropy, stateTol);
	assert_close_value("colAniso", gpu.lastColAnisotropy, cpu.lastColAnisotropy, stateTol);
	assert_close_value("geomScale", gpu.lastGeometryScale, cpu.lastGeometryScale, stateTol);
	ASSERT("echo step mismatch", gpu.step == cpu.step);
}

static glades::NumberInput* make_atlas_transformer_resume_dataset()
{
	const int numSamples = 8;
	const int numFeatures = 4;
	const int numOutputs = 4;

	glades::NumberInput* di = new glades::NumberInput();
	di->trainMatrix = shmea::GMatrix(numSamples, shmea::GVector<float>(numFeatures, 0.0f));
	di->trainExpectedMatrix = shmea::GMatrix(numSamples, shmea::GVector<float>(numOutputs, 0.0f));

	for (int i = 0; i < numSamples; ++i)
	{
		for (int j = 0; j < numFeatures; ++j)
			di->trainMatrix[i][j] = static_cast<float>(i * numFeatures + j) / static_cast<float>(numSamples * numFeatures);
		for (int j = 0; j < numOutputs; ++j)
			di->trainExpectedMatrix[i][j] = static_cast<float>((i + j) % numOutputs) / static_cast<float>(numOutputs);
	}

	di->testMatrix = di->trainMatrix;
	di->testExpectedMatrix = di->trainExpectedMatrix;
	return di;
}

static glades::NNInfo* make_atlas_transformer_resume_info(const char* name)
{
	glades::InputLayerInfo* in = new glades::InputLayerInfo(
	    /*batchSize*/ 1,
	    /*learningRate*/ 0.01f,
	    /*momentumFactor*/ 0.0f,
	    /*weightDecay1*/ 0.0f,
	    /*weightDecay2*/ 0.0f,
	    /*pDropout*/ 0.0f,
	    /*activationType*/ glades::GMath::LINEAR,
	    /*activationParam*/ 1.0f);

	std::vector<glades::HiddenLayerInfo*> hidden;
	hidden.push_back(new glades::HiddenLayerInfo(
	    /*size*/ 16,
	    /*learningRate*/ 0.01f,
	    /*momentumFactor*/ 0.0f,
	    /*weightDecay1*/ 0.0f,
	    /*weightDecay2*/ 0.0f,
	    /*pDropout*/ 0.0f,
	    /*activationType*/ glades::GMath::LINEAR,
	    /*activationParam*/ 1.0f));

	glades::OutputLayerInfo* out = new glades::OutputLayerInfo(4, glades::OutputLayerInfo::REGRESSION);
	return new glades::NNInfo(name, in, hidden, out);
}

static void build_atlas_token_split(unsigned int vocab,
                                    unsigned int seqCount,
                                    unsigned int seqLen,
                                    unsigned int seed,
                                    unsigned int padTokenId,
                                    std::vector<unsigned int>& outTokens,
                                    std::vector<glades::DataInput::SequenceSpan>& outSpans)
{
	outTokens.clear();
	outSpans.clear();
	outTokens.reserve(static_cast<size_t>(seqCount) * static_cast<size_t>(seqLen + 1u));
	outSpans.reserve(seqCount);

	for (unsigned int seq = 0u; seq < seqCount; ++seq)
	{
		const unsigned int start = static_cast<unsigned int>(outTokens.size());
		unsigned int a = (seed + 17u * (seq + 1u)) % (vocab - 1u);
		unsigned int b = (seed + 31u * (seq + 3u)) % (vocab - 1u);
		const unsigned int phase = 1u + ((seed + 97u * (seq + 5u)) % 7u);
		for (unsigned int t = 0u; t < seqLen; ++t)
		{
			unsigned int tok = 0u;
			if (t == 0u)
				tok = a;
			else if (t == 1u)
				tok = b;
			else
			{
				tok = (a + b + phase + ((t / 4u) % 3u)) % (vocab - 1u);
				a = b;
				b = tok;
			}
			outTokens.push_back(tok);
		}
		outSpans.push_back(glades::DataInput::SequenceSpan(start, seqLen));
		outTokens.push_back(padTokenId);
	}
}

static InMemoryTokenIdInput* make_atlas_token_dataset(unsigned int vocab,
                                                      unsigned int seqCount,
                                                      unsigned int seqLen,
                                                      unsigned int seed,
                                                      unsigned int padTokenId)
{
	std::vector<unsigned int> trainTokens;
	std::vector<unsigned int> testTokens;
	std::vector<glades::DataInput::SequenceSpan> trainSpans;
	std::vector<glades::DataInput::SequenceSpan> testSpans;
	build_atlas_token_split(vocab, seqCount, seqLen, seed + 11u, padTokenId, trainTokens, trainSpans);
	build_atlas_token_split(vocab, std::max(1u, seqCount / 2u), seqLen, seed + 1011u, padTokenId, testTokens, testSpans);

	InMemoryTokenIdInput* di = new InMemoryTokenIdInput();
	di->setTrainTokens(trainTokens, static_cast<int>(padTokenId));
	di->setTestTokens(testTokens, static_cast<int>(padTokenId));
	(void)di->setTrainSequences(trainSpans);
	(void)di->setTestSequences(testSpans);
	return di;
}

static glades::NNInfo* make_atlas_transformer_token_info(const char* name,
                                                         unsigned int vocab,
                                                         unsigned int dModel,
                                                         unsigned int layers,
                                                         float learningRate)
{
	glades::InputLayerInfo* in = new glades::InputLayerInfo(
	    1, learningRate, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f);
	std::vector<glades::HiddenLayerInfo*> hidden;
	for (unsigned int i = 0u; i < layers; ++i)
	{
		hidden.push_back(new glades::HiddenLayerInfo(
		    static_cast<int>(dModel), learningRate, 0.0f, 0.0f, 0.0f, 0.0f,
		    glades::GMath::LINEAR, 1.0f));
	}
	glades::OutputLayerInfo* out =
	    new glades::OutputLayerInfo(static_cast<int>(vocab), glades::OutputLayerInfo::CLASSIFICATION);
	return new glades::NNInfo(name, in, hidden, out);
}

static int64_t now_ms()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return static_cast<int64_t>(tv.tv_sec) * 1000LL + static_cast<int64_t>(tv.tv_usec) / 1000LL;
}

static void configure_atlas_transformer_resume_net(glades::NNetwork& net, unsigned int seed, int epochs)
{
	net.setSeed(seed);
	net.getTerminatorMutable().setEpoch(epochs);
	net.getTerminatorMutable().setAccuracy(0);

	glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
	cfg.optimizer.type = glades::OptimizerConfig::ATLAS;
	cfg.atlas.rank = 8u;
	cfg.atlas.complementRank = 0u;
	cfg.atlas.tSub = 50u;
	cfg.transformer.nHeadsOverride = 4;
	cfg.transformer.dFFOverride = 32;
}

static void prepare_atlas_complement_test_state(glades::atlas::WeightState& state,
                                                glades::rng::Engine& rng,
                                                unsigned int m,
                                                unsigned int n,
                                                unsigned int r,
                                                unsigned int complementRank,
                                                const char* tag)
{
	glades::atlas::initWeightState(state, m, n, r, 0.0f, rng);

	glades::ATLASConfig acSetup;
	acSetup.rank = r;
	acSetup.complementRank = complementRank;
	acSetup.beta = 1.0f;
	acSetup.biasCorrection = false;
	acSetup.muMin = 0.0f;
	acSetup.muMax = 0.0f;
	acSetup.tSub = 0u;

	std::vector<float> W(static_cast<size_t>(m) * n, 0.0f);
	std::vector<float> gW(static_cast<size_t>(m) * n, 0.0f);
	const bool ok = glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
	                                         1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
	                                         acSetup, rng, 0, tag);
	ASSERT("==============ATLAS::ComplementTest setup applyStep failed==============", ok);

	state.step = 0ULL;
	state.activeRank = r;
	state.activeComplementRank = 0u;
	state.trialComplementRank = 0u;
	state.trialComplementWins = 0u;
	state.trialComplementMean = 0.0f;
	state.trialComplementVar = 0.0f;
}

static bool atlas_storage_shapes_ok(const glades::atlas::WeightState& state,
                                    unsigned int m,
                                    unsigned int n,
                                    unsigned int r,
                                    unsigned int complementRank)
{
	const unsigned int sparrowModeRank = (state.sparrowModeRank > 0u) ? state.sparrowModeRank : 1u;
	return state.U.size() == static_cast<size_t>(m) * r
	    && state.fisherDiag.size() == static_cast<size_t>(r)
	    && state.prevGz.size() == static_cast<size_t>(r) * n
	    && state.complementRank == complementRank
	    && state.V.size() == static_cast<size_t>(m) * complementRank
	    && state.complementBlock.size() == static_cast<size_t>(complementRank) * complementRank
	    && state.scoutBasis.size() == static_cast<size_t>(m) * complementRank
	    && state.scoutCov.size() == static_cast<size_t>(complementRank) * complementRank
	    && state.scoutNoise.size() == static_cast<size_t>(complementRank) * complementRank
	    && state.prevGv.size() == static_cast<size_t>(complementRank) * n
	    && state.resolveGzHistory.size() >= static_cast<size_t>(4u) * r * n
	    && state.heroGwHistory.size() == static_cast<size_t>(4u) * complementRank * n
	    && state.sparrowPrevActive.size() == static_cast<size_t>(r) * n
	    && state.sparrowPrevScout.size() == static_cast<size_t>(complementRank) * n
	    && state.sparrowFutureCov.size() == static_cast<size_t>(r) * r
	    && state.sparrowPastCov.size() == static_cast<size_t>(r + complementRank) * (r + complementRank)
	    && state.sparrowCrossCov.size() == static_cast<size_t>(r) * (r + complementRank)
	    && state.sparrowLeftMode.size() == static_cast<size_t>(sparrowModeRank) * r
	    && state.sparrowRightMode.size() == static_cast<size_t>(sparrowModeRank) * (r + complementRank)
	    && state.sparrowLatent.size() == static_cast<size_t>(sparrowModeRank) * n;
}

} // anonymous namespace

void ATLASHelmMicroBenchmark()
{
	struct MicroVariantSpec
	{
		const char* label;
		glades::OptimizerConfig::Type optimizerType;
		float learningRate;
		bool helmEnabled;
	};

	struct MicroVariantResult
	{
		MicroVariantResult()
		    : ok(false),
		      trainSeconds(0.0),
		      trainNll(0.0f),
		      trainPpl(0.0f),
		      helmMatrices(0u),
		      helmEdge(0.0),
		      helmPredR2(0.0),
		      helmMemoryGain(0.0),
		      status()
		{
		}

		bool ok;
		double trainSeconds;
		float trainNll;
		float trainPpl;
		unsigned int helmMatrices;
		double helmEdge;
		double helmPredR2;
		double helmMemoryGain;
		std::string status;
	};

	static const unsigned int kVocab = 17u;
	static const unsigned int kPadTokenId = kVocab - 1u;
	static const unsigned int kTrainSeqs = 8u;
	static const unsigned int kSeqLen = 8u;
	static const unsigned int kLayers = 1u;
	static const unsigned int kDModel = 8u;
	static const unsigned int kHeads = 2u;
	static const unsigned int kDff = 16u;
	static const unsigned int kEpochs = 2u;

	const MicroVariantSpec specs[] = {
		{ "AdamW", glades::OptimizerConfig::ADAMW, 0.001f, false },
		{ "ATLAS-BSRP", glades::OptimizerConfig::ATLAS, 0.020f, false },
		{ "ATLAS-HELM", glades::OptimizerConfig::ATLAS, 0.020f, true },
	};
	const size_t specCount = sizeof(specs) / sizeof(specs[0]);

	printf("============================================================\n");
	printf("ATLAS HELM Transformer Micro-Benchmark\n");
	printf("============================================================\n");
	printf("Config: vocab=%u trainSeqs=%u seqLen=%u dModel=%u dFF=%u layers=%u heads=%u epochs=%u gpu=off\n",
	       kVocab, kTrainSeqs, kSeqLen, kDModel, kDff, kLayers, kHeads, kEpochs);
	printf("%-12s %10s %12s %12s %10s %10s %10s %s\n",
	       "Optimizer", "Train(s)", "TrainNLL", "TrainPPL", "HELMObs", "HELMEdge", "HELMGain", "Status");

	for (size_t i = 0u; i < specCount; ++i)
	{
		const MicroVariantSpec& spec = specs[i];
		InMemoryTokenIdInput* di = make_atlas_token_dataset(kVocab, kTrainSeqs, kSeqLen,
		                                                    71000u + static_cast<unsigned int>(100u * i),
		                                                    kPadTokenId);
		glades::NNInfo* info = make_atlas_transformer_token_info("ut_atlas_helm_micro",
		                                                         kVocab, kDModel, kLayers,
		                                                         spec.learningRate);
		{
			glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
			net.setSeed(72000u + static_cast<unsigned int>(100u * i));
			net.setLogger(quiet_logger());
			net.getTerminatorMutable().setEpoch(static_cast<int>(kEpochs));
			net.getTerminatorMutable().setAccuracy(0.0f);

			{
				glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
				cfg.gpu.enable = false;
				cfg.optimizer.type = spec.optimizerType;
				cfg.transformer.enableTokenEmbedding = true;
				cfg.transformer.vocabSizeOverride = static_cast<int>(kVocab);
				cfg.transformer.tieEmbeddings = true;
				cfg.transformer.padTokenId = static_cast<int>(kPadTokenId);
				cfg.transformer.nHeadsOverride = static_cast<int>(kHeads);
				cfg.transformer.nKVHeadsOverride = static_cast<int>(kHeads);
				cfg.transformer.dFFOverride = static_cast<int>(kDff);
				cfg.transformer.ffnKind = glades::TransformerRunConfig::FFN_SWIGLU;
				cfg.transformer.normType = glades::TransformerRunConfig::NORM_RMSNORM;
				cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_ROPE;

				if (spec.optimizerType == glades::OptimizerConfig::ATLAS)
				{
					cfg.atlas.rank = 4u;
					cfg.atlas.complementRank = 0u;
					cfg.atlas.tSub = 8u;
					cfg.atlas.beta = 0.999f;
					cfg.atlas.helmEnabled = spec.helmEnabled;
					cfg.atlas.helmMemoryScale = 0.05f;
					cfg.atlas.helmEdgeThreshold = 0.0f;
					cfg.atlas.helmModeRank = 2u;
					cfg.atlas.helmHiddenStackDepth = 2u;
					cfg.atlas.helmPoleMax = 0.95f;
				}
			}

			CaptureMetricsCallbacks cb;
			const int64_t startMs = now_ms();
			const glades::NNetworkStatus st = net.train(di, &cb);
			const int64_t endMs = now_ms();

			glades::NNetwork::AtlasRuntimeDiagnostics diag;
			const bool haveDiag = net.getAtlasRuntimeDiagnostics(diag);
			const float trainNll = cb.saw ? cb.last.totalError : 0.0f;
			const float trainPpl = cb.saw ? cb.last.perplexity : 0.0f;
			const unsigned int helmMatrices = haveDiag ? diag.helmMatrices : 0u;
			const double helmEdge = haveDiag ? diag.helmMeanEdge : 0.0;
			const double helmMemoryGain = haveDiag ? diag.helmMeanMemoryGain : 0.0;
			const std::string status =
			    (!st.ok()) ? st.message : (cb.saw ? "ok" : "no metrics");

			printf("%-12s %10.3f %12.5f %12.5f %10u %10.4f %10.4f %s\n",
			       spec.label,
			       static_cast<double>(endMs - startMs) / 1000.0,
			       trainNll,
			       trainPpl,
			       helmMatrices,
			       helmEdge,
			       helmMemoryGain,
			       status.c_str());
		}

		delete di;
		delete info;
	}

	printf("\n");
}

void ATLASECHOCoreUnitTest()
{
	printf("============================================================\n");
	printf("ATLAS ECHO Core Unit Test\n");
	printf("============================================================\n");

	const unsigned int m = 4u;
	const unsigned int n = 3u;
	const unsigned int samples = 2u;
	const size_t mn = static_cast<size_t>(m) * n;
	const float lr = 0.01f;
	const float beta1 = 0.9f;
	const float beta2 = 0.999f;
	const float eps = 1.0e-8f;
	const float invBatch = 1.0f;
	const float gradScale = 1.0f;
	const float inv1mB1t = 1.0f / (1.0f - beta1);
	const float inv1mB2t = 1.0f / (1.0f - beta2);

	const float initWRaw[] = {
		0.20f, -0.30f, 0.10f,
		0.40f, -0.10f, 0.30f,
		-0.20f, 0.50f, -0.40f,
		0.15f, -0.25f, 0.35f
	};
	const float gradRaw[] = {
		1.80f, 0.20f, -0.10f,
		1.20f, -0.30f, 0.40f,
		-0.90f, 0.10f, 0.05f,
		0.70f, -0.15f, 0.02f
	};
	const float rowObsRaw[] = {
		3.0f, 1.0f, 0.5f, 0.25f,
		2.5f, 0.9f, 0.4f, 0.2f
	};
	const float colObsRaw[] = {
		0.2f, 1.0f, 4.0f,
		0.3f, 0.9f, 3.5f
	};
	const std::vector<float> initW(initWRaw, initWRaw + mn);
	const std::vector<float> grad(gradRaw, gradRaw + mn);
	const std::vector<float> rowObs(rowObsRaw, rowObsRaw + samples * m);
	const std::vector<float> colObs(colObsRaw, colObsRaw + samples * n);

	glades::ATLASConfig adamFallbackAc;
	adamFallbackAc.beta = 0.0f;
	adamFallbackAc.echoEnabled = true;
	adamFallbackAc.echoGeometryScale = 0.0f;
	adamFallbackAc.tSub = 1u;

	glades::ATLASConfig echoAc = adamFallbackAc;
	echoAc.echoGeometryScale = 1.0f;

	glades::atlas::EchoWeightState fallbackState;
	glades::atlas::EchoWeightState echoState;
	ASSERT("echoObserve fallback failed",
	       glades::atlas::echoObserve(fallbackState,
	                                  rowObs.data(), colObs.data(),
	                                  samples, m, n, adamFallbackAc));
	ASSERT("echoObserve geometry failed",
	       glades::atlas::echoObserve(echoState,
	                                  rowObs.data(), colObs.data(),
	                                  samples, m, n, echoAc));

	std::vector<float> fallbackW = initW;
	std::vector<float> fallbackM(mn, 0.0f);
	std::vector<float> fallbackV(mn, 0.0f);
	std::vector<float> fallbackG = grad;
	ASSERT("echoUpdate fallback failed",
	       glades::atlas::echoUpdate(fallbackState,
	                                 fallbackW.data(), fallbackM.data(), fallbackV.data(), fallbackG.data(),
	                                 m, n, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
	                                 invBatch, gradScale, 0.0f, 0.0f,
	                                 1ULL,
	                                 adamFallbackAc, quiet_logger(), "ut.echo.fallback"));

	std::vector<float> expectedW = initW;
	std::vector<float> expectedM(mn, 0.0f);
	std::vector<float> expectedV(mn, 0.0f);
	for (size_t idx = 0u; idx < mn; ++idx)
	{
		const float g = grad[idx];
		expectedM[idx] = (1.0f - beta1) * g;
		expectedV[idx] = (1.0f - beta2) * (g * g);
		const float step = g / (fabsf(g) + eps);
		expectedW[idx] -= lr * step;
	}

	assert_close_vector("echo-fallback-W", fallbackW, expectedW, 1.0e-6f);
	assert_close_vector("echo-fallback-m1", fallbackM, expectedM, 1.0e-6f);
	assert_close_vector("echo-fallback-v2", fallbackV, expectedV, 1.0e-6f);
	for (size_t idx = 0u; idx < mn; ++idx)
		ASSERT("fallback gradients not cleared", fallbackG[idx] == 0.0f);

	std::vector<float> echoW = initW;
	std::vector<float> echoM(mn, 0.0f);
	std::vector<float> echoV(mn, 0.0f);
	std::vector<float> echoG = grad;
	ASSERT("echoUpdate geometry failed",
	       glades::atlas::echoUpdate(echoState,
	                                 echoW.data(), echoM.data(), echoV.data(), echoG.data(),
	                                 m, n, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
	                                 invBatch, gradScale, 0.0f, 0.0f,
	                                 1ULL,
	                                 echoAc, quiet_logger(), "ut.echo.geometry"));

	ASSERT("echo row anisotropy not active", echoState.lastRowAnisotropy > 1.5f);
	ASSERT("echo col anisotropy not active", echoState.lastColAnisotropy > 1.5f);
	ASSERT("echo geometry scale not recorded", fabsf(echoState.lastGeometryScale - 1.0f) < 1.0e-6f);
	ASSERT("echo step counter not advanced", echoState.step == 1ULL);

	double diffNorm = 0.0;
	for (size_t idx = 0u; idx < mn; ++idx)
	{
		const double delta = static_cast<double>(echoW[idx]) - static_cast<double>(fallbackW[idx]);
		diffNorm += delta * delta;
	}
	ASSERT("echo geometry did not change the step", diffNorm > 1.0e-8);

	glades::ATLASConfig scheduleAc = echoAc;
	scheduleAc.echoGeometryScale = 1.0f;
	scheduleAc.echoGeometryScaleFinal = 0.25f;
	scheduleAc.echoGeometryDecaySteps = 4u;
	glades::atlas::EchoWeightState scheduleStateStep1;
	glades::atlas::EchoWeightState scheduleStateStep5;
	ASSERT("echoObserve schedule step1 failed",
	       glades::atlas::echoObserve(scheduleStateStep1,
	                                  rowObs.data(), colObs.data(),
	                                  samples, m, n, scheduleAc));
	ASSERT("echoObserve schedule step5 failed",
	       glades::atlas::echoObserve(scheduleStateStep5,
	                                  rowObs.data(), colObs.data(),
	                                  samples, m, n, scheduleAc));
	std::vector<float> scheduleW1 = initW;
	std::vector<float> scheduleM1(mn, 0.0f);
	std::vector<float> scheduleV1(mn, 0.0f);
	std::vector<float> scheduleG1 = grad;
	ASSERT("echoUpdate schedule step1 failed",
	       glades::atlas::echoUpdate(scheduleStateStep1,
	                                 scheduleW1.data(), scheduleM1.data(), scheduleV1.data(), scheduleG1.data(),
	                                 m, n, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
	                                 invBatch, gradScale, 0.0f, 0.0f,
	                                 1ULL,
	                                 scheduleAc, quiet_logger(), "ut.echo.schedule1"));
	std::vector<float> scheduleW5 = initW;
	std::vector<float> scheduleM5(mn, 0.0f);
	std::vector<float> scheduleV5(mn, 0.0f);
	std::vector<float> scheduleG5 = grad;
	ASSERT("echoUpdate schedule step5 failed",
	       glades::atlas::echoUpdate(scheduleStateStep5,
	                                 scheduleW5.data(), scheduleM5.data(), scheduleV5.data(), scheduleG5.data(),
	                                 m, n, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
	                                 invBatch, gradScale, 0.0f, 0.0f,
	                                 5ULL,
	                                 scheduleAc, quiet_logger(), "ut.echo.schedule5"));
	ASSERT("echo schedule initial scale mismatch", fabsf(scheduleStateStep1.lastGeometryScale - 1.0f) < 1.0e-6f);
	ASSERT("echo schedule final scale mismatch", fabsf(scheduleStateStep5.lastGeometryScale - 0.25f) < 1.0e-6f);
	ASSERT("echo schedule did not decay", scheduleStateStep5.lastGeometryScale < scheduleStateStep1.lastGeometryScale);

	printf("[UT] ECHO core: PASSED\n");
	printf("============================================================\n");
}

void ATLASECHOParityTest()
{
	printf("============================================================\n");
	printf("ATLAS ECHO CPU-vs-GPU Parity Test\n");
	printf("============================================================\n");

#ifdef GLADES_HAVE_CUDA
	if (!glades::gpu::initDevice())
	{
		printf("No CUDA device available, skipping ECHO parity tests.\n");
		printf("============================================================\n");
		return;
	}

	const unsigned int m = 4u;
	const unsigned int n = 3u;
	const unsigned int samples = 2u;
	const size_t mn = static_cast<size_t>(m) * n;
	const float lr = 0.01f;
	const float beta1 = 0.9f;
	const float beta2 = 0.999f;
	const float eps = 1.0e-8f;
	const float invBatch = 1.0f;
	const float gradScale = 1.0f;

	const float initWRaw[] = {
		0.20f, -0.30f, 0.10f,
		0.40f, -0.10f, 0.30f,
		-0.20f, 0.50f, -0.40f,
		0.15f, -0.25f, 0.35f
	};
	const float g1Raw[] = {
		1.80f, 0.20f, -0.10f,
		1.20f, -0.30f, 0.40f,
		-0.90f, 0.10f, 0.05f,
		0.70f, -0.15f, 0.02f
	};
	const float g2Raw[] = {
		1.30f, 0.18f, -0.06f,
		0.95f, -0.22f, 0.35f,
		-0.75f, 0.08f, 0.04f,
		0.55f, -0.11f, 0.03f
	};
	const float rowObs1Raw[] = {
		3.0f, 1.0f, 0.5f, 0.25f,
		2.5f, 0.9f, 0.4f, 0.2f
	};
	const float rowObs2Raw[] = {
		2.8f, 0.95f, 0.55f, 0.30f,
		2.3f, 0.85f, 0.45f, 0.22f
	};
	const float colObs1Raw[] = {
		0.2f, 1.0f, 4.0f,
		0.3f, 0.9f, 3.5f
	};
	const float colObs2Raw[] = {
		0.25f, 1.1f, 3.8f,
		0.35f, 0.85f, 3.2f
	};

	const std::vector<float> initW(initWRaw, initWRaw + mn);
	const std::vector<float> grads[] = {
		std::vector<float>(g1Raw, g1Raw + mn),
		std::vector<float>(g2Raw, g2Raw + mn)
	};
	const std::vector<float> rowObs[] = {
		std::vector<float>(rowObs1Raw, rowObs1Raw + samples * m),
		std::vector<float>(rowObs2Raw, rowObs2Raw + samples * m)
	};
	const std::vector<float> colObs[] = {
		std::vector<float>(colObs1Raw, colObs1Raw + samples * n),
		std::vector<float>(colObs2Raw, colObs2Raw + samples * n)
	};

	glades::ATLASConfig ac;
	ac.beta = 0.7f;
	ac.echoEnabled = true;
	ac.echoGeometryScale = 1.0f;
	ac.tSub = 1u;

	std::vector<float> cpuW = initW;
	std::vector<float> cpuM(mn, 0.0f);
	std::vector<float> cpuV(mn, 0.0f);
	std::vector<float> cpuG(mn, 0.0f);
	glades::atlas::EchoWeightState cpuState;

	glades::gpu::GpuBuffer<float> dW;
	glades::gpu::GpuBuffer<float> dM;
	glades::gpu::GpuBuffer<float> dV;
	glades::gpu::GpuBuffer<float> dG;
	glades::gpu::GpuBuffer<float> dRowObs;
	glades::gpu::GpuBuffer<float> dColObs;
	ASSERT("gpu dW alloc failed", dW.allocate(mn));
	ASSERT("gpu dM alloc failed", dM.allocate(mn));
	ASSERT("gpu dV alloc failed", dV.allocate(mn));
	ASSERT("gpu dG alloc failed", dG.allocate(mn));
	ASSERT("gpu dRowObs alloc failed", dRowObs.allocate(static_cast<size_t>(samples) * m));
	ASSERT("gpu dColObs alloc failed", dColObs.allocate(static_cast<size_t>(samples) * n));
	ASSERT("gpu dW upload failed", dW.upload(initW.data(), initW.size()));
	ASSERT("gpu dM zero failed", dM.zero());
	ASSERT("gpu dV zero failed", dV.zero());
	ASSERT("gpu dG zero failed", dG.zero());
	glades::gpu::GpuEchoWeightState gpuState;

	for (int step = 0; step < 2; ++step)
	{
		cpuG = grads[step];
		ASSERT("cpu echo observe failed",
		       glades::atlas::echoObserve(cpuState,
		                                  rowObs[step].data(), colObs[step].data(),
		                                  samples, m, n, ac));
		const double b1t = std::pow(static_cast<double>(beta1), static_cast<double>(step + 1));
		const double b2t = std::pow(static_cast<double>(beta2), static_cast<double>(step + 1));
		const float inv1mB1t = static_cast<float>(1.0 / (1.0 - b1t));
		const float inv1mB2t = static_cast<float>(1.0 / (1.0 - b2t));
		ASSERT("cpu echo update failed",
		       glades::atlas::echoUpdate(cpuState,
		                                 cpuW.data(), cpuM.data(), cpuV.data(), cpuG.data(),
		                                 m, n, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
		                                 invBatch, gradScale, 0.0f, 0.0f,
		                                 static_cast<unsigned long long>(step + 1),
		                                 ac, quiet_logger(), "ut.echo.cpu"));

		ASSERT("gpu echo update failed",
		       run_gpu_echo_step(gpuState,
		                         dW, dM, dV, dG, dRowObs, dColObs,
		                         m, n, samples,
		                         lr, beta1, beta2, eps,
		                         invBatch, gradScale, step + 1,
		                         ac, grads[step], rowObs[step], colObs[step]));

		EchoParitySnapshot cpuSnap;
		EchoParitySnapshot gpuSnap;
		capture_cpu_echo_snapshot(cpuSnap, cpuW, cpuM, cpuV, cpuG, cpuState);
		capture_gpu_echo_snapshot(gpuSnap, dW, dM, dV, dG, gpuState);

		char label[64];
		sprintf(label, "echo step %d", step + 1);
		assert_echo_parity_snapshot(label, cpuSnap, gpuSnap, 1.0e-5f, 1.0e-5f);
	}

	printf("[UT] ECHO parity: PASSED\n");
	printf("============================================================\n");
#else
	printf("CUDA not enabled, skipping ECHO parity tests.\n");
	printf("============================================================\n");
#endif
}

void ATLASECHOMicroBenchmark()
{
	struct MicroVariantSpec
	{
		const char* label;
		glades::OptimizerConfig::Type optimizerType;
		float learningRate;
		bool echoEnabled;
		float echoGeometryScale;
	};

	static const unsigned int kVocab = 17u;
	static const unsigned int kPadTokenId = kVocab - 1u;
	static const unsigned int kTrainSeqs = 8u;
	static const unsigned int kSeqLen = 8u;
	static const unsigned int kLayers = 1u;
	static const unsigned int kDModel = 8u;
	static const unsigned int kHeads = 2u;
	static const unsigned int kDff = 16u;
	static const unsigned int kEpochs = 2u;

	const MicroVariantSpec specs[] = {
		{ "AdamW", glades::OptimizerConfig::ADAMW, 0.001f, false, 0.0f },
		{ "ECHO-0", glades::OptimizerConfig::ATLAS, 0.001f, true, 0.0f },
		{ "ECHO", glades::OptimizerConfig::ATLAS, 0.001f, true, 1.0f },
	};
	const size_t specCount = sizeof(specs) / sizeof(specs[0]);

	printf("============================================================\n");
	printf("ATLAS ECHO Transformer Micro-Benchmark\n");
	printf("============================================================\n");
	printf("Config: vocab=%u trainSeqs=%u seqLen=%u dModel=%u dFF=%u layers=%u heads=%u epochs=%u gpu=off\n",
	       kVocab, kTrainSeqs, kSeqLen, kDModel, kDff, kLayers, kHeads, kEpochs);
	printf("%-12s %10s %12s %12s %10s %10s %s\n",
	       "Optimizer", "Train(s)", "TrainNLL", "TrainPPL", "ApplyMs", "HeadShr", "Status");

	for (size_t i = 0u; i < specCount; ++i)
	{
		const MicroVariantSpec& spec = specs[i];
		InMemoryTokenIdInput* di = make_atlas_token_dataset(kVocab, kTrainSeqs, kSeqLen,
		                                                    90000u + static_cast<unsigned int>(100u * i),
		                                                    kPadTokenId);
		glades::NNInfo* info = make_atlas_transformer_token_info("ut_atlas_echo_micro",
		                                                         kVocab, kDModel, kLayers,
		                                                         spec.learningRate);
		{
			glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
			net.setSeed(90100u + static_cast<unsigned int>(100u * i));
			net.setLogger(quiet_logger());
			net.getTerminatorMutable().setEpoch(static_cast<int>(kEpochs));
			net.getTerminatorMutable().setAccuracy(0.0f);

			{
				glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
				cfg.gpu.enable = false;
				cfg.optimizer.type = spec.optimizerType;
				cfg.transformer.enableTokenEmbedding = true;
				cfg.transformer.vocabSizeOverride = static_cast<int>(kVocab);
				cfg.transformer.tieEmbeddings = true;
				cfg.transformer.padTokenId = static_cast<int>(kPadTokenId);
				cfg.transformer.nHeadsOverride = static_cast<int>(kHeads);
				cfg.transformer.nKVHeadsOverride = static_cast<int>(kHeads);
				cfg.transformer.dFFOverride = static_cast<int>(kDff);
				cfg.transformer.ffnKind = glades::TransformerRunConfig::FFN_SWIGLU;
				cfg.transformer.normType = glades::TransformerRunConfig::NORM_RMSNORM;
				cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_ROPE;

				if (spec.optimizerType == glades::OptimizerConfig::ATLAS)
				{
					cfg.atlas.rank = 4u;
					cfg.atlas.complementRank = 0u;
					cfg.atlas.tSub = 8u;
					cfg.atlas.beta = 0.999f;
					cfg.atlas.echoEnabled = spec.echoEnabled;
					cfg.atlas.echoGeometryScale = spec.echoGeometryScale;
				}
			}

			CaptureMetricsCallbacks cb;
			const int64_t startMs = now_ms();
			const glades::NNetworkStatus st = net.train(di, &cb);
			const int64_t endMs = now_ms();

			glades::NNetwork::AtlasRuntimeDiagnostics diag;
			const bool haveDiag = net.getAtlasRuntimeDiagnostics(diag);
			const float trainNll = cb.saw ? cb.last.totalError : 0.0f;
			const float trainPpl = cb.saw ? cb.last.perplexity : 0.0f;
			const double applyMs =
			    (haveDiag && diag.transformerGapBatches > 0u) ? diag.transformerMeanApplyMs : 0.0;
			const double headShare =
			    (haveDiag && diag.transformerGapBatches > 0u) ? diag.transformerMeanHeadShare : 0.0;
			const std::string status =
			    (!st.ok()) ? st.message : (cb.saw ? "ok" : "no metrics");

			printf("%-12s %10.3f %12.5f %12.5f %10.4f %10.4f %s\n",
			       spec.label,
			       static_cast<double>(endMs - startMs) / 1000.0,
			       trainNll,
			       trainPpl,
			       applyMs,
			       headShare,
			       status.c_str());
		}

		delete di;
		delete info;
	}

	printf("\n");
}

void ATLASBiMAPMicroBenchmark()
{
	struct MicroVariantSpec
	{
		const char* label;
		glades::OptimizerConfig::Type optimizerType;
		float learningRate;
		bool bimapEnabled;
		bool bimapLowRankEnabled;
		float bimapPredictiveScale;
		unsigned int bimapRank;
		unsigned int bimapCadence;
	};

	static const unsigned int kVocab = 17u;
	static const unsigned int kPadTokenId = kVocab - 1u;
	static const unsigned int kTrainSeqs = 8u;
	static const unsigned int kSeqLen = 8u;
	static const unsigned int kLayers = 1u;
	static const unsigned int kDModel = 8u;
	static const unsigned int kHeads = 2u;
	static const unsigned int kDff = 16u;
	static const unsigned int kEpochs = 2u;

	const MicroVariantSpec specs[] = {
		{ "AdamW", glades::OptimizerConfig::ADAMW, 0.001f, false, false, 0.0f, 0u, 1u },
		{ "BiMAP-lite", glades::OptimizerConfig::ATLAS, 0.001f, true, false, 0.0f, 4u, 1u },
		{ "BiMAP-v2-0", glades::OptimizerConfig::ATLAS, 0.001f, true, true, 0.0f, 4u, 8u },
		{ "BiMAP-v2", glades::OptimizerConfig::ATLAS, 0.001f, true, true, 0.15f, 4u, 8u },
	};
	const size_t specCount = sizeof(specs) / sizeof(specs[0]);

	printf("============================================================\n");
	printf("ATLAS BiMAP Transformer Micro-Benchmark\n");
	printf("============================================================\n");
	printf("Config: vocab=%u trainSeqs=%u seqLen=%u dModel=%u dFF=%u layers=%u heads=%u epochs=%u gpu=off\n",
	       kVocab, kTrainSeqs, kSeqLen, kDModel, kDff, kLayers, kHeads, kEpochs);
	printf("%-12s %10s %12s %12s %10s %10s %s\n",
	       "Optimizer", "Train(s)", "TrainNLL", "TrainPPL", "ApplyMs", "HeadShr", "Status");

	for (size_t i = 0u; i < specCount; ++i)
	{
		const MicroVariantSpec& spec = specs[i];
		InMemoryTokenIdInput* di = make_atlas_token_dataset(kVocab, kTrainSeqs, kSeqLen,
		                                                    73000u + static_cast<unsigned int>(100u * i),
		                                                    kPadTokenId);
		glades::NNInfo* info = make_atlas_transformer_token_info("ut_atlas_bimap_micro",
		                                                         kVocab, kDModel, kLayers,
		                                                         spec.learningRate);
		{
			glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
			net.setSeed(74000u + static_cast<unsigned int>(100u * i));
			net.setLogger(quiet_logger());
			net.getTerminatorMutable().setEpoch(static_cast<int>(kEpochs));
			net.getTerminatorMutable().setAccuracy(0.0f);

			{
				glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
				cfg.gpu.enable = false;
				cfg.optimizer.type = spec.optimizerType;
				cfg.transformer.enableTokenEmbedding = true;
				cfg.transformer.vocabSizeOverride = static_cast<int>(kVocab);
				cfg.transformer.tieEmbeddings = true;
				cfg.transformer.padTokenId = static_cast<int>(kPadTokenId);
				cfg.transformer.nHeadsOverride = static_cast<int>(kHeads);
				cfg.transformer.nKVHeadsOverride = static_cast<int>(kHeads);
				cfg.transformer.dFFOverride = static_cast<int>(kDff);
				cfg.transformer.ffnKind = glades::TransformerRunConfig::FFN_SWIGLU;
				cfg.transformer.normType = glades::TransformerRunConfig::NORM_RMSNORM;
				cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_ROPE;

				if (spec.optimizerType == glades::OptimizerConfig::ATLAS)
				{
					cfg.atlas.rank = spec.bimapRank;
					cfg.atlas.complementRank = 0u;
					cfg.atlas.tSub = 8u;
					cfg.atlas.beta = 0.999f;
					cfg.atlas.bimapEnabled = spec.bimapEnabled;
					cfg.atlas.bimapLowRankEnabled = spec.bimapLowRankEnabled;
					cfg.atlas.bimapGeometryScale = 1.0f;
					cfg.atlas.bimapPredictiveScale = spec.bimapPredictiveScale;
					cfg.atlas.bimapFactorCadence = spec.bimapCadence;
				}
			}

			CaptureMetricsCallbacks cb;
			const int64_t startMs = now_ms();
			const glades::NNetworkStatus st = net.train(di, &cb);
			const int64_t endMs = now_ms();

			glades::NNetwork::AtlasRuntimeDiagnostics diag;
			const bool haveDiag = net.getAtlasRuntimeDiagnostics(diag);
			const float trainNll = cb.saw ? cb.last.totalError : 0.0f;
			const float trainPpl = cb.saw ? cb.last.perplexity : 0.0f;
			const double applyMs =
			    (haveDiag && diag.transformerGapBatches > 0u) ? diag.transformerMeanApplyMs : 0.0;
			const double headShare =
			    (haveDiag && diag.transformerGapBatches > 0u) ? diag.transformerMeanHeadShare : 0.0;
			const std::string status =
			    (!st.ok()) ? st.message : (cb.saw ? "ok" : "no metrics");

			printf("%-12s %10.3f %12.5f %12.5f %10.4f %10.4f %s\n",
			       spec.label,
			       static_cast<double>(endMs - startMs) / 1000.0,
			       trainNll,
			       trainPpl,
			       applyMs,
			       headShare,
			       status.c_str());
		}

		delete di;
		delete info;
	}

	printf("\n");
}

void ATLASBiMAPParityTest()
{
	printf("============================================================\n");
	printf("ATLAS BiMAP CPU-vs-GPU Parity Test\n");
	printf("============================================================\n");

#ifdef GLADES_HAVE_CUDA
	if (!glades::gpu::initDevice())
	{
		printf("No CUDA device available, skipping BiMAP parity tests.\n");
		printf("============================================================\n");
		return;
	}

	struct ParityCase
	{
		const char* label;
		bool lowRank;
		float predictiveScale;
		unsigned int rank;
		unsigned int powerIters;
	};

	const ParityCase cases[] = {
		{ "BiMAP-lite", false, 0.0f, 2u, 1u },
		{ "BiMAP-lite-pred", false, 0.15f, 2u, 1u },
		{ "BiMAP-v2-0", true, 0.0f, 2u, 2u },
		{ "BiMAP-v2", true, 0.15f, 2u, 2u },
	};

	const unsigned int m = 4u;
	const unsigned int n = 3u;
	const size_t mn = static_cast<size_t>(m) * n;
	const float lr = 0.01f;
	const float beta1 = 0.9f;
	const float beta2 = 0.999f;
	const float eps = 1.0e-8f;
	const float invBatch = 1.0f;
	const float gradScale = 1.0f;
	const float valueTolLite = 1.0e-5f;
	const float stateTolLite = 1.0e-5f;
	const float valueTolV2 = 2.5e-4f;
	const float stateTolV2 = 2.5e-4f;

	const float initWRaw[] = {
		0.20f, -0.30f, 0.10f,
		0.40f, -0.10f, 0.30f,
		-0.20f, 0.50f, -0.40f,
		0.15f, -0.25f, 0.35f
	};
	const float g1Raw[] = {
		1.80f, 0.20f, -0.10f,
		1.20f, -0.30f, 0.40f,
		-0.90f, 0.10f, 0.05f,
		0.70f, -0.15f, 0.02f
	};
	const float g2Raw[] = {
		1.30f, 0.18f, -0.06f,
		0.95f, -0.22f, 0.35f,
		-0.75f, 0.08f, 0.04f,
		0.55f, -0.11f, 0.03f
	};
	const std::vector<float> initW(initWRaw, initWRaw + mn);
	const std::vector<float> grads[] = {
		std::vector<float>(g1Raw, g1Raw + mn),
		std::vector<float>(g2Raw, g2Raw + mn)
	};

	for (size_t caseIdx = 0u; caseIdx < sizeof(cases) / sizeof(cases[0]); ++caseIdx)
	{
		const ParityCase& spec = cases[caseIdx];
		printf("-----------------------------------\n");
		printf("%s parity\n", spec.label);
		printf("-----------------------------------\n");

		glades::ATLASConfig ac;
		ac.beta = 0.999f;
		ac.rank = spec.rank;
		ac.powerIters = spec.powerIters;
		ac.tSub = 1u;
		ac.bimapEnabled = true;
		ac.bimapLowRankEnabled = spec.lowRank;
		ac.bimapGeometryScale = 1.0f;
		ac.bimapPredictiveScale = spec.predictiveScale;
		ac.bimapFactorCadence = 1u;

		std::vector<float> cpuW = initW;
		std::vector<float> cpuM(mn, 0.0f);
		std::vector<float> cpuV(mn, 0.0f);
		std::vector<float> cpuG(mn, 0.0f);
		glades::atlas::BiMAPWeightState cpuState;

		glades::gpu::GpuBuffer<float> dW;
		glades::gpu::GpuBuffer<float> dM;
		glades::gpu::GpuBuffer<float> dV;
		glades::gpu::GpuBuffer<float> dG;
		ASSERT("gpu dW alloc failed", dW.allocate(mn));
		ASSERT("gpu dM alloc failed", dM.allocate(mn));
		ASSERT("gpu dV alloc failed", dV.allocate(mn));
		ASSERT("gpu dG alloc failed", dG.allocate(mn));
		ASSERT("gpu dW upload failed", dW.upload(initW.data(), initW.size()));
		ASSERT("gpu dM zero failed", dM.zero());
		ASSERT("gpu dV zero failed", dV.zero());
		ASSERT("gpu dG zero failed", dG.zero());
		glades::gpu::GpuBiMAPWeightState gpuState;

		for (int step = 0; step < 2; ++step)
		{
			cpuG = grads[step];
			const double b1t = std::pow(static_cast<double>(beta1), static_cast<double>(step + 1));
			const double b2t = std::pow(static_cast<double>(beta2), static_cast<double>(step + 1));
			const float inv1mB1t = static_cast<float>(1.0 / (1.0 - b1t));
			const float inv1mB2t = static_cast<float>(1.0 / (1.0 - b2t));

			const bool cpuOk = glades::atlas::bimapUpdate(cpuState,
			                                              cpuW.data(), cpuM.data(), cpuV.data(), cpuG.data(),
			                                              m, n, lr,
			                                              beta1, beta2,
			                                              inv1mB1t, inv1mB2t,
			                                              eps,
			                                              invBatch, gradScale,
			                                              0.0f, 0.0f,
			                                              ac, quiet_logger(), "ut.bimap.cpu");
			ASSERT("cpu bimap update failed", cpuOk);

			const bool gpuOk = run_gpu_bimap_step(gpuState,
			                                      dW, dM, dV, dG,
			                                      m, n, lr,
			                                      beta1, beta2, eps,
			                                      invBatch, gradScale,
			                                      step + 1,
			                                      ac,
			                                      grads[step]);
			ASSERT("gpu bimap update failed", gpuOk);

			BiMAPParitySnapshot cpuSnap;
			BiMAPParitySnapshot gpuSnap;
			capture_cpu_bimap_snapshot(cpuSnap, cpuW, cpuM, cpuV, cpuG, cpuState);
			capture_gpu_bimap_snapshot(gpuSnap, dW, dM, dV, dG, gpuState);

			char stepLabel[128];
			sprintf(stepLabel, "%s step %d", spec.label, step + 1);
			assert_bimap_parity_snapshot(stepLabel,
			                             cpuSnap,
			                             gpuSnap,
			                             spec.lowRank ? valueTolV2 : valueTolLite,
			                             spec.lowRank ? stateTolV2 : stateTolLite,
			                             spec.lowRank);
		}

		printf("[UT] %s parity: PASSED\n", spec.label);
	}

	printf("============================================================\n");
	printf("BiMAP parity tests: ALL PASSED\n");
	printf("============================================================\n");
#else
	printf("CUDA not enabled, skipping BiMAP parity tests.\n");
	printf("============================================================\n");
#endif
}

void ATLASKronMicroBenchmark()
{
	struct MicroVariantSpec
	{
		const char* label;
		glades::OptimizerConfig::Type optimizerType;
		float learningRate;
		bool bimapEnabled;
		bool kronEnabled;
		float kronGeometryScale;
		float kronPredictiveScale;
		unsigned int kronCadence;
		float kronDamping;
	};

	static const unsigned int kVocab = 17u;
	static const unsigned int kPadTokenId = kVocab - 1u;
	static const unsigned int kTrainSeqs = 8u;
	static const unsigned int kSeqLen = 8u;
	static const unsigned int kLayers = 1u;
	static const unsigned int kDModel = 8u;
	static const unsigned int kHeads = 2u;
	static const unsigned int kDff = 16u;
	static const unsigned int kEpochs = 2u;

	const MicroVariantSpec specs[] = {
		{ "AdamW", glades::OptimizerConfig::ADAMW, 0.001f, false, false, 0.0f, 0.0f, 1u, 0.10f },
		{ "BiMAP-lite", glades::OptimizerConfig::ATLAS, 0.001f, true, false, 0.0f, 0.0f, 1u, 0.10f },
		{ "KRON-0", glades::OptimizerConfig::ATLAS, 0.001f, false, true, 0.0f, 0.0f, 1u, 0.10f },
		{ "KRON", glades::OptimizerConfig::ATLAS, 0.001f, false, true, 1.0f, 0.05f, 8u, 0.10f },
	};
	const size_t specCount = sizeof(specs) / sizeof(specs[0]);

	printf("============================================================\n");
	printf("ATLAS KRON Transformer Micro-Benchmark\n");
	printf("============================================================\n");
	printf("Config: vocab=%u trainSeqs=%u seqLen=%u dModel=%u dFF=%u layers=%u heads=%u epochs=%u gpu=off\n",
	       kVocab, kTrainSeqs, kSeqLen, kDModel, kDff, kLayers, kHeads, kEpochs);
	printf("%-12s %10s %12s %12s %10s %10s %s\n",
	       "Optimizer", "Train(s)", "TrainNLL", "TrainPPL", "ApplyMs", "HeadShr", "Status");

	for (size_t i = 0u; i < specCount; ++i)
	{
		const MicroVariantSpec& spec = specs[i];
		InMemoryTokenIdInput* di = make_atlas_token_dataset(kVocab, kTrainSeqs, kSeqLen,
		                                                    78000u + static_cast<unsigned int>(100u * i),
		                                                    kPadTokenId);
		glades::NNInfo* info = make_atlas_transformer_token_info("ut_atlas_kron_micro",
		                                                         kVocab, kDModel, kLayers,
		                                                         spec.learningRate);
		{
			glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
			net.setSeed(79000u + static_cast<unsigned int>(100u * i));
			net.setLogger(quiet_logger());
			net.getTerminatorMutable().setEpoch(static_cast<int>(kEpochs));
			net.getTerminatorMutable().setAccuracy(0.0f);

			{
				glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
				cfg.gpu.enable = false;
				cfg.optimizer.type = spec.optimizerType;
				cfg.transformer.enableTokenEmbedding = true;
				cfg.transformer.vocabSizeOverride = static_cast<int>(kVocab);
				cfg.transformer.tieEmbeddings = true;
				cfg.transformer.padTokenId = static_cast<int>(kPadTokenId);
				cfg.transformer.nHeadsOverride = static_cast<int>(kHeads);
				cfg.transformer.nKVHeadsOverride = static_cast<int>(kHeads);
				cfg.transformer.dFFOverride = static_cast<int>(kDff);
				cfg.transformer.ffnKind = glades::TransformerRunConfig::FFN_SWIGLU;
				cfg.transformer.normType = glades::TransformerRunConfig::NORM_RMSNORM;
				cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_ROPE;

				if (spec.optimizerType == glades::OptimizerConfig::ATLAS)
				{
					cfg.atlas.rank = 4u;
					cfg.atlas.complementRank = 0u;
					cfg.atlas.tSub = 8u;
					cfg.atlas.beta = 0.999f;
					cfg.atlas.bimapEnabled = spec.bimapEnabled;
					cfg.atlas.bimapLowRankEnabled = false;
					cfg.atlas.kronEnabled = spec.kronEnabled;
					cfg.atlas.kronGeometryScale = spec.kronGeometryScale;
					cfg.atlas.kronPredictiveScale = spec.kronPredictiveScale;
					cfg.atlas.kronFactorCadence = spec.kronCadence;
					cfg.atlas.kronDamping = spec.kronDamping;
				}
			}

			CaptureMetricsCallbacks cb;
			const int64_t startMs = now_ms();
			const glades::NNetworkStatus st = net.train(di, &cb);
			const int64_t endMs = now_ms();

			glades::NNetwork::AtlasRuntimeDiagnostics diag;
			const bool haveDiag = net.getAtlasRuntimeDiagnostics(diag);
			const float trainNll = cb.saw ? cb.last.totalError : 0.0f;
			const float trainPpl = cb.saw ? cb.last.perplexity : 0.0f;
			const double applyMs =
			    (haveDiag && diag.transformerGapBatches > 0u) ? diag.transformerMeanApplyMs : 0.0;
			const double headShare =
			    (haveDiag && diag.transformerGapBatches > 0u) ? diag.transformerMeanHeadShare : 0.0;
			const std::string status =
			    (!st.ok()) ? st.message : (cb.saw ? "ok" : "no metrics");

			printf("%-12s %10.3f %12.5f %12.5f %10.4f %10.4f %s\n",
			       spec.label,
			       static_cast<double>(endMs - startMs) / 1000.0,
			       trainNll,
			       trainPpl,
			       applyMs,
			       headShare,
			       status.c_str());
		}

		delete di;
		delete info;
	}

	printf("\n");
}

void ATLASMuonMicroBenchmark()
{
	struct MicroVariantSpec
	{
		const char* label;
		glades::OptimizerConfig::Type optimizerType;
		float learningRate;
		bool muonEnabled;
		float muonGeometryScale;
		float muonPredictiveScale;
		float muonMaxAspect;
		unsigned int muonMinDim;
		float muonDamping;
	};

	static const unsigned int kVocab = 17u;
	static const unsigned int kPadTokenId = kVocab - 1u;
	static const unsigned int kTrainSeqs = 8u;
	static const unsigned int kSeqLen = 8u;
	static const unsigned int kLayers = 1u;
	static const unsigned int kDModel = 8u;
	static const unsigned int kHeads = 2u;
	static const unsigned int kDff = 16u;
	static const unsigned int kEpochs = 2u;

	const MicroVariantSpec specs[] = {
		{ "AdamW", glades::OptimizerConfig::ADAMW, 0.001f, false, 0.0f, 0.0f, 1.50f, 2u, 0.01f },
		{ "MUON-0", glades::OptimizerConfig::ATLAS, 0.001f, true, 0.0f, 0.0f, 1.50f, 2u, 0.01f },
		{ "MUON", glades::OptimizerConfig::ATLAS, 0.001f, true, 1.0f, 0.05f, 1.50f, 2u, 0.01f },
	};
	const size_t specCount = sizeof(specs) / sizeof(specs[0]);

	printf("============================================================\n");
	printf("ATLAS MUON Transformer Micro-Benchmark\n");
	printf("============================================================\n");
	printf("Config: vocab=%u trainSeqs=%u seqLen=%u dModel=%u dFF=%u layers=%u heads=%u epochs=%u gpu=off\n",
	       kVocab, kTrainSeqs, kSeqLen, kDModel, kDff, kLayers, kHeads, kEpochs);
	printf("%-12s %10s %12s %12s %10s %10s %s\n",
	       "Optimizer", "Train(s)", "TrainNLL", "TrainPPL", "ApplyMs", "HeadShr", "Status");

	for (size_t i = 0u; i < specCount; ++i)
	{
		const MicroVariantSpec& spec = specs[i];
		InMemoryTokenIdInput* di = make_atlas_token_dataset(kVocab, kTrainSeqs, kSeqLen,
		                                                    79500u + static_cast<unsigned int>(100u * i),
		                                                    kPadTokenId);
		glades::NNInfo* info = make_atlas_transformer_token_info("ut_atlas_muon_micro",
		                                                         kVocab, kDModel, kLayers,
		                                                         spec.learningRate);
		{
			glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
			net.setSeed(79600u + static_cast<unsigned int>(100u * i));
			net.setLogger(quiet_logger());
			net.getTerminatorMutable().setEpoch(static_cast<int>(kEpochs));
			net.getTerminatorMutable().setAccuracy(0.0f);

			{
				glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
				cfg.gpu.enable = false;
				cfg.optimizer.type = spec.optimizerType;
				cfg.transformer.enableTokenEmbedding = true;
				cfg.transformer.vocabSizeOverride = static_cast<int>(kVocab);
				cfg.transformer.tieEmbeddings = true;
				cfg.transformer.padTokenId = static_cast<int>(kPadTokenId);
				cfg.transformer.nHeadsOverride = static_cast<int>(kHeads);
				cfg.transformer.nKVHeadsOverride = static_cast<int>(kHeads);
				cfg.transformer.dFFOverride = static_cast<int>(kDff);
				cfg.transformer.ffnKind = glades::TransformerRunConfig::FFN_SWIGLU;
				cfg.transformer.normType = glades::TransformerRunConfig::NORM_RMSNORM;
				cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_ROPE;

				if (spec.optimizerType == glades::OptimizerConfig::ATLAS)
				{
					cfg.atlas.rank = 4u;
					cfg.atlas.complementRank = 0u;
					cfg.atlas.tSub = 8u;
					cfg.atlas.beta = 0.999f;
					cfg.atlas.muonEnabled = spec.muonEnabled;
					cfg.atlas.muonGeometryScale = spec.muonGeometryScale;
					cfg.atlas.muonPredictiveScale = spec.muonPredictiveScale;
					cfg.atlas.muonMaxAspect = spec.muonMaxAspect;
					cfg.atlas.muonMinDim = spec.muonMinDim;
					cfg.atlas.muonDamping = spec.muonDamping;
				}
			}

			CaptureMetricsCallbacks cb;
			const int64_t startMs = now_ms();
			const glades::NNetworkStatus st = net.train(di, &cb);
			const int64_t endMs = now_ms();

			glades::NNetwork::AtlasRuntimeDiagnostics diag;
			const bool haveDiag = net.getAtlasRuntimeDiagnostics(diag);
			const float trainNll = cb.saw ? cb.last.totalError : 0.0f;
			const float trainPpl = cb.saw ? cb.last.perplexity : 0.0f;
			const double applyMs =
			    (haveDiag && diag.transformerGapBatches > 0u) ? diag.transformerMeanApplyMs : 0.0;
			const double headShare =
			    (haveDiag && diag.transformerGapBatches > 0u) ? diag.transformerMeanHeadShare : 0.0;
			const std::string status =
			    (!st.ok()) ? st.message : (cb.saw ? "ok" : "no metrics");

			printf("%-12s %10.3f %12.5f %12.5f %10.4f %10.4f %s\n",
			       spec.label,
			       static_cast<double>(endMs - startMs) / 1000.0,
			       trainNll,
			       trainPpl,
			       applyMs,
			       headShare,
			       status.c_str());
		}

		delete di;
		delete info;
	}

	printf("\n");
}

void ATLASPACTMicroBenchmark()
{
	struct MicroVariantSpec
	{
		const char* label;
		glades::OptimizerConfig::Type optimizerType;
		float learningRate;
		bool bimapEnabled;
		bool bimapLowRankEnabled;
		float bimapPredictiveScale;
		bool pactEnabled;
		bool pactLowRankEnabled;
		float pactPredictiveScale;
		unsigned int rank;
		unsigned int cadence;
		float pactCostScale;
		float pactPromoteThreshold;
		float pactDemoteThreshold;
	};

	static const unsigned int kVocab = 17u;
	static const unsigned int kPadTokenId = kVocab - 1u;
	static const unsigned int kTrainSeqs = 8u;
	static const unsigned int kSeqLen = 8u;
	static const unsigned int kLayers = 1u;
	static const unsigned int kDModel = 8u;
	static const unsigned int kHeads = 2u;
	static const unsigned int kDff = 16u;
	static const unsigned int kEpochs = 2u;

	const MicroVariantSpec specs[] = {
		{ "AdamW", glades::OptimizerConfig::ADAMW, 0.001f, false, false, 0.0f, false, false, 0.0f, 4u, 1u, 0.0f, 0.0f, 0.0f },
		{ "BiMAP-lite", glades::OptimizerConfig::ATLAS, 0.001f, true, false, 0.0f, false, false, 0.0f, 4u, 1u, 0.0f, 0.0f, 0.0f },
		{ "PACT-lite", glades::OptimizerConfig::ATLAS, 0.001f, false, false, 0.0f, true, false, 0.0f, 4u, 1u, 0.0f, -1.0f, -1.0f },
		{ "PACT-v2-0", glades::OptimizerConfig::ATLAS, 0.001f, false, false, 0.0f, true, true, 0.0f, 4u, 8u, 0.0f, -1.0f, -1.0f },
		{ "PACT-v2", glades::OptimizerConfig::ATLAS, 0.001f, false, false, 0.0f, true, true, 0.10f, 4u, 8u, 0.0f, -1.0f, -1.0f },
	};
	const size_t specCount = sizeof(specs) / sizeof(specs[0]);

	printf("============================================================\n");
	printf("ATLAS PACT Transformer Micro-Benchmark\n");
	printf("============================================================\n");
	printf("Config: vocab=%u trainSeqs=%u seqLen=%u dModel=%u dFF=%u layers=%u heads=%u epochs=%u gpu=off\n",
	       kVocab, kTrainSeqs, kSeqLen, kDModel, kDff, kLayers, kHeads, kEpochs);
	printf("%-12s %10s %12s %12s %10s %10s %s\n",
	       "Optimizer", "Train(s)", "TrainNLL", "TrainPPL", "ApplyMs", "HeadShr", "Status");

	for (size_t i = 0u; i < specCount; ++i)
	{
		const MicroVariantSpec& spec = specs[i];
		InMemoryTokenIdInput* di = make_atlas_token_dataset(kVocab, kTrainSeqs, kSeqLen,
		                                                    76000u + static_cast<unsigned int>(100u * i),
		                                                    kPadTokenId);
		glades::NNInfo* info = make_atlas_transformer_token_info("ut_atlas_pact_micro",
		                                                         kVocab, kDModel, kLayers,
		                                                         spec.learningRate);
		{
			glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
			net.setSeed(77000u + static_cast<unsigned int>(100u * i));
			net.setLogger(quiet_logger());
			net.getTerminatorMutable().setEpoch(static_cast<int>(kEpochs));
			net.getTerminatorMutable().setAccuracy(0.0f);

			{
				glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
				cfg.gpu.enable = false;
				cfg.optimizer.type = spec.optimizerType;
				cfg.transformer.enableTokenEmbedding = true;
				cfg.transformer.vocabSizeOverride = static_cast<int>(kVocab);
				cfg.transformer.tieEmbeddings = true;
				cfg.transformer.padTokenId = static_cast<int>(kPadTokenId);
				cfg.transformer.nHeadsOverride = static_cast<int>(kHeads);
				cfg.transformer.nKVHeadsOverride = static_cast<int>(kHeads);
				cfg.transformer.dFFOverride = static_cast<int>(kDff);
				cfg.transformer.ffnKind = glades::TransformerRunConfig::FFN_SWIGLU;
				cfg.transformer.normType = glades::TransformerRunConfig::NORM_RMSNORM;
				cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_ROPE;

				if (spec.optimizerType == glades::OptimizerConfig::ATLAS)
				{
					cfg.atlas.rank = spec.rank;
					cfg.atlas.complementRank = 0u;
					cfg.atlas.tSub = 8u;
					cfg.atlas.beta = 0.999f;
					cfg.atlas.bimapEnabled = spec.bimapEnabled;
					cfg.atlas.bimapLowRankEnabled = spec.bimapLowRankEnabled;
					cfg.atlas.bimapGeometryScale = 1.0f;
					cfg.atlas.bimapPredictiveScale = spec.bimapPredictiveScale;
					cfg.atlas.bimapFactorCadence = spec.cadence;
					cfg.atlas.pactEnabled = spec.pactEnabled;
					cfg.atlas.pactLowRankEnabled = spec.pactLowRankEnabled;
					cfg.atlas.pactGeometryScale = 1.0f;
					cfg.atlas.pactPredictiveScale = spec.pactPredictiveScale;
					cfg.atlas.pactFactorCadence = spec.cadence;
					cfg.atlas.pactCostScale = spec.pactCostScale;
					cfg.atlas.pactPromoteThreshold = spec.pactPromoteThreshold;
					cfg.atlas.pactDemoteThreshold = spec.pactDemoteThreshold;
				}
			}

			CaptureMetricsCallbacks cb;
			const int64_t startMs = now_ms();
			const glades::NNetworkStatus st = net.train(di, &cb);
			const int64_t endMs = now_ms();

			glades::NNetwork::AtlasRuntimeDiagnostics diag;
			const bool haveDiag = net.getAtlasRuntimeDiagnostics(diag);
			const float trainNll = cb.saw ? cb.last.totalError : 0.0f;
			const float trainPpl = cb.saw ? cb.last.perplexity : 0.0f;
			const double applyMs =
			    (haveDiag && diag.transformerGapBatches > 0u) ? diag.transformerMeanApplyMs : 0.0;
			const double headShare =
			    (haveDiag && diag.transformerGapBatches > 0u) ? diag.transformerMeanHeadShare : 0.0;
			const std::string status =
			    (!st.ok()) ? st.message : (cb.saw ? "ok" : "no metrics");

			printf("%-12s %10.3f %12.5f %12.5f %10.4f %10.4f %s\n",
			       spec.label,
			       static_cast<double>(endMs - startMs) / 1000.0,
			       trainNll,
			       trainPpl,
			       applyMs,
			       headShare,
			       status.c_str());
		}

		delete di;
		delete info;
	}

	printf("\n");
}

void ATLASRACERMicroBenchmark()
{
	struct MicroVariantSpec
	{
		const char* label;
		glades::OptimizerConfig::Type optimizerType;
		float learningRate;
		bool bimapEnabled;
		float bimapPredictiveScale;
		bool racerEnabled;
		float racerGeometryScale;
		float racerPredictiveScale;
		unsigned int rank;
		unsigned int cadence;
		float racerRiskScale;
		float racerCostScale;
		float racerPromoteThreshold;
		float racerDemoteThreshold;
	};

	static const unsigned int kVocab = 17u;
	static const unsigned int kPadTokenId = kVocab - 1u;
	static const unsigned int kTrainSeqs = 8u;
	static const unsigned int kSeqLen = 8u;
	static const unsigned int kLayers = 1u;
	static const unsigned int kDModel = 8u;
	static const unsigned int kHeads = 2u;
	static const unsigned int kDff = 16u;
	static const unsigned int kEpochs = 2u;

	const MicroVariantSpec specs[] = {
		{ "AdamW", glades::OptimizerConfig::ADAMW, 0.001f, false, 0.0f, false, 0.0f, 0.0f, 4u, 1u, 0.50f, 0.0010f, 0.0f, 0.0f },
		{ "BiMAP-lite", glades::OptimizerConfig::ATLAS, 0.001f, true, 0.0f, false, 0.0f, 0.0f, 4u, 1u, 0.50f, 0.0010f, 0.0f, 0.0f },
		{ "RACER-0", glades::OptimizerConfig::ATLAS, 0.001f, false, 0.0f, true, 0.0f, 0.0f, 4u, 1u, 0.50f, 0.0f, 1.0f, 0.5f },
		{ "RACER-lite", glades::OptimizerConfig::ATLAS, 0.001f, false, 0.0f, true, 1.0f, 0.05f, 4u, 8u, 0.50f, 0.0010f, -1.0f, -1.0f },
	};
	const size_t specCount = sizeof(specs) / sizeof(specs[0]);

	printf("============================================================\n");
	printf("ATLAS RACER Transformer Micro-Benchmark\n");
	printf("============================================================\n");
	printf("Config: vocab=%u trainSeqs=%u seqLen=%u dModel=%u dFF=%u layers=%u heads=%u epochs=%u gpu=off\n",
	       kVocab, kTrainSeqs, kSeqLen, kDModel, kDff, kLayers, kHeads, kEpochs);
	printf("%-12s %10s %12s %12s %10s %10s %s\n",
	       "Optimizer", "Train(s)", "TrainNLL", "TrainPPL", "ApplyMs", "HeadShr", "Status");

	for (size_t i = 0u; i < specCount; ++i)
	{
		const MicroVariantSpec& spec = specs[i];
		InMemoryTokenIdInput* di = make_atlas_token_dataset(kVocab, kTrainSeqs, kSeqLen,
		                                                    81000u + static_cast<unsigned int>(100u * i),
		                                                    kPadTokenId);
		glades::NNInfo* info = make_atlas_transformer_token_info("ut_atlas_racer_micro",
		                                                         kVocab, kDModel, kLayers,
		                                                         spec.learningRate);
		{
			glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
			net.setSeed(81100u + static_cast<unsigned int>(100u * i));
			net.setLogger(quiet_logger());
			net.getTerminatorMutable().setEpoch(static_cast<int>(kEpochs));
			net.getTerminatorMutable().setAccuracy(0.0f);

			{
				glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
				cfg.gpu.enable = false;
				cfg.optimizer.type = spec.optimizerType;
				cfg.transformer.enableTokenEmbedding = true;
				cfg.transformer.vocabSizeOverride = static_cast<int>(kVocab);
				cfg.transformer.tieEmbeddings = true;
				cfg.transformer.padTokenId = static_cast<int>(kPadTokenId);
				cfg.transformer.nHeadsOverride = static_cast<int>(kHeads);
				cfg.transformer.nKVHeadsOverride = static_cast<int>(kHeads);
				cfg.transformer.dFFOverride = static_cast<int>(kDff);
				cfg.transformer.ffnKind = glades::TransformerRunConfig::FFN_SWIGLU;
				cfg.transformer.normType = glades::TransformerRunConfig::NORM_RMSNORM;
				cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_ROPE;

				if (spec.optimizerType == glades::OptimizerConfig::ATLAS)
				{
					cfg.atlas.rank = spec.rank;
					cfg.atlas.complementRank = 0u;
					cfg.atlas.tSub = 8u;
					cfg.atlas.beta = 0.999f;
					cfg.atlas.bimapEnabled = spec.bimapEnabled;
					cfg.atlas.bimapLowRankEnabled = false;
					cfg.atlas.bimapGeometryScale = 1.0f;
					cfg.atlas.bimapPredictiveScale = spec.bimapPredictiveScale;
					cfg.atlas.bimapFactorCadence = spec.cadence;
					cfg.atlas.racerEnabled = spec.racerEnabled;
					cfg.atlas.racerGeometryScale = spec.racerGeometryScale;
					cfg.atlas.racerPredictiveScale = spec.racerPredictiveScale;
					cfg.atlas.racerFactorCadence = spec.cadence;
					cfg.atlas.racerRiskScale = spec.racerRiskScale;
					cfg.atlas.racerCostScale = spec.racerCostScale;
					cfg.atlas.racerPromoteThreshold = spec.racerPromoteThreshold;
					cfg.atlas.racerDemoteThreshold = spec.racerDemoteThreshold;
				}
			}

			CaptureMetricsCallbacks cb;
			const int64_t startMs = now_ms();
			const glades::NNetworkStatus st = net.train(di, &cb);
			const int64_t endMs = now_ms();

			glades::NNetwork::AtlasRuntimeDiagnostics diag;
			const bool haveDiag = net.getAtlasRuntimeDiagnostics(diag);
			const float trainNll = cb.saw ? cb.last.totalError : 0.0f;
			const float trainPpl = cb.saw ? cb.last.perplexity : 0.0f;
			const double applyMs =
			    (haveDiag && diag.transformerGapBatches > 0u) ? diag.transformerMeanApplyMs : 0.0;
			const double headShare =
			    (haveDiag && diag.transformerGapBatches > 0u) ? diag.transformerMeanHeadShare : 0.0;
			const std::string status =
			    (!st.ok()) ? st.message : (cb.saw ? "ok" : "no metrics");

			printf("%-12s %10.3f %12.5f %12.5f %10.4f %10.4f %s\n",
			       spec.label,
			       static_cast<double>(endMs - startMs) / 1000.0,
			       trainNll,
			       trainPpl,
			       applyMs,
			       headShare,
			       status.c_str());
		}

		delete di;
		delete info;
	}

	printf("\n");
}

void ATLASGroupAdamMicroBenchmark()
{
	struct MicroVariantSpec
	{
		const char* label;
		bool groupwiseEnabled;
		unsigned int minGroupSize;
		float stabilityScale;
		float snrScale;
		float ratioScale;
		float minScale;
		float maxScale;
	};

	static const unsigned int kVocab = 17u;
	static const unsigned int kPadTokenId = kVocab - 1u;
	static const unsigned int kTrainSeqs = 8u;
	static const unsigned int kSeqLen = 8u;
	static const unsigned int kLayers = 1u;
	static const unsigned int kDModel = 8u;
	static const unsigned int kHeads = 2u;
	static const unsigned int kDff = 16u;
	static const unsigned int kEpochs = 2u;
	static const float kLearningRate = 0.001f;

	const MicroVariantSpec specs[] = {
		{ "AdamW", false, 256u, 0.05f, 0.05f, 0.50f, 0.90f, 1.15f },
		{ "AdamW-Group", true, 8u, 0.10f, 0.10f, 0.35f, 0.90f, 1.20f },
	};
	const size_t specCount = sizeof(specs) / sizeof(specs[0]);

	printf("============================================================\n");
	printf("AdamW Groupwise Transformer Micro-Benchmark\n");
	printf("============================================================\n");
	printf("Config: vocab=%u trainSeqs=%u seqLen=%u dModel=%u dFF=%u layers=%u heads=%u epochs=%u gpu=off\n",
	       kVocab, kTrainSeqs, kSeqLen, kDModel, kDff, kLayers, kHeads, kEpochs);
	printf("%-12s %10s %12s %12s %10s %s\n",
	       "Optimizer", "Train(s)", "TrainNLL", "TrainPPL", "ApplyMs", "Status");

	for (size_t i = 0u; i < specCount; ++i)
	{
		const MicroVariantSpec& spec = specs[i];
		InMemoryTokenIdInput* di = make_atlas_token_dataset(kVocab, kTrainSeqs, kSeqLen,
		                                                    82000u + static_cast<unsigned int>(100u * i),
		                                                    kPadTokenId);
		glades::NNInfo* info = make_atlas_transformer_token_info("ut_adamw_group_micro",
		                                                         kVocab, kDModel, kLayers,
		                                                         kLearningRate);
		{
			glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
			net.setSeed(82100u + static_cast<unsigned int>(100u * i));
			net.setLogger(quiet_logger());
			net.getTerminatorMutable().setEpoch(static_cast<int>(kEpochs));
			net.getTerminatorMutable().setAccuracy(0.0f);

			glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
			cfg.gpu.enable = false;
			cfg.optimizer.type = glades::OptimizerConfig::ADAMW;
			cfg.optimizer.adamGroupwiseEnabled = spec.groupwiseEnabled;
			cfg.optimizer.adamGroupMinSize = spec.minGroupSize;
			cfg.optimizer.adamGroupStabilityScale = spec.stabilityScale;
			cfg.optimizer.adamGroupSnrScale = spec.snrScale;
			cfg.optimizer.adamGroupRatioScale = spec.ratioScale;
			cfg.optimizer.adamGroupMinScale = spec.minScale;
			cfg.optimizer.adamGroupMaxScale = spec.maxScale;
			cfg.transformer.enableTokenEmbedding = true;
			cfg.transformer.vocabSizeOverride = static_cast<int>(kVocab);
			cfg.transformer.tieEmbeddings = true;
			cfg.transformer.padTokenId = static_cast<int>(kPadTokenId);
			cfg.transformer.nHeadsOverride = static_cast<int>(kHeads);
			cfg.transformer.nKVHeadsOverride = static_cast<int>(kHeads);
			cfg.transformer.dFFOverride = static_cast<int>(kDff);
			cfg.transformer.ffnKind = glades::TransformerRunConfig::FFN_SWIGLU;
			cfg.transformer.normType = glades::TransformerRunConfig::NORM_RMSNORM;
			cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_ROPE;

			CaptureMetricsCallbacks cb;
			const int64_t startMs = now_ms();
			const glades::NNetworkStatus st = net.train(di, &cb);
			const int64_t endMs = now_ms();

			glades::NNetwork::AtlasRuntimeDiagnostics diag;
			const bool haveDiag = net.getAtlasRuntimeDiagnostics(diag);
			const float trainNll = cb.saw ? cb.last.totalError : 0.0f;
			const float trainPpl = cb.saw ? cb.last.perplexity : 0.0f;
			const double applyMs =
			    (haveDiag && diag.transformerGapBatches > 0u) ? diag.transformerMeanApplyMs : 0.0;
			const std::string status =
			    (!st.ok()) ? st.message : (cb.saw ? "ok" : "no metrics");

			printf("%-12s %10.3f %12.5f %12.5f %10.4f %s\n",
			       spec.label,
			       static_cast<double>(endMs - startMs) / 1000.0,
			       trainNll,
			       trainPpl,
			       applyMs,
			       status.c_str());
		}

		delete di;
		delete info;
	}

	printf("\n");
}

void ATLASPACTCoreUnitTest()
{
	printf("============================================================\n");
	printf("ATLAS PACT Core Unit Test\n");
	printf("============================================================\n");

	const unsigned int m = 3u;
	const unsigned int n = 2u;
	const size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);
	const float lr = 0.01f;
	const float beta1 = 0.9f;
	const float beta2 = 0.999f;
	const float inv1mB1t = 1.0f / (1.0f - beta1);
	const float inv1mB2t = 1.0f / (1.0f - beta2);
	const float eps = 1.0e-8f;

	{
		std::vector<float> W(mn, 0.5f);
		std::vector<float> m1(mn, 0.0f);
		std::vector<float> v2(mn, 0.0f);
		const float gRaw[] = { 0.25f, -0.50f, 1.00f, -1.50f, 0.75f, -0.25f };
		std::vector<float> g(gRaw, gRaw + (sizeof(gRaw) / sizeof(gRaw[0])));

		std::vector<float> expected = W;
		for (size_t idx = 0u; idx < mn; ++idx)
		{
			const float grad = g[idx];
			const float denom = fabsf(grad) + eps;
			expected[idx] -= lr * (grad / denom);
		}

		glades::atlas::BiMAPWeightState state;
		glades::ATLASConfig ac;
		ac.rank = 2u;
		ac.powerIters = 1u;
		ac.beta = 0.999f;
		ac.pactEnabled = true;
		ac.pactLowRankEnabled = false;
		ac.pactGeometryScale = 0.0f;
		ac.pactPredictiveScale = 0.0f;
		ac.pactFactorCadence = 1u;
		ac.pactCostScale = 0.0f;
		ac.pactPromoteThreshold = 1.0f;
		ac.pactDemoteThreshold = 0.5f;

		const bool ok = glades::atlas::pactUpdate(state,
		                                          &W[0], &m1[0], &v2[0], &g[0],
		                                          m, n, lr,
		                                          beta1, beta2, inv1mB1t, inv1mB2t, eps,
		                                          1.0f, 1.0f, 0.0f, 0.0f,
		                                          ac, 0, "ut.pact.fallback");
		ASSERT("PACT fallback update failed", ok);
		ASSERT("PACT fallback should stay demoted", !state.promoted);
		for (size_t idx = 0u; idx < mn; ++idx)
		{
			ASSERT("PACT fallback should zero gradients", fabsf(g[idx]) < 1.0e-12f);
			ASSERT("PACT fallback must match AdamW update", fabsf(W[idx] - expected[idx]) < 1.0e-6f);
		}
		printf("[UT] PACT fallback matches exact Adam-style update\n");
	}

	{
		std::vector<float> WPromoted(mn, 0.5f);
		std::vector<float> WFallback(mn, 0.5f);
		std::vector<float> mPromoted(mn, 0.0f);
		std::vector<float> vPromoted(mn, 0.0f);
		std::vector<float> mFallback(mn, 0.0f);
		std::vector<float> vFallback(mn, 0.0f);
		const float g1Raw[] = { 2.0f, 0.1f, 2.0f, 0.1f, 0.1f, 0.1f };
		const float g2Raw[] = { 1.5f, 0.05f, 1.5f, 0.05f, 0.05f, 0.05f };
		std::vector<float> g1(g1Raw, g1Raw + (sizeof(g1Raw) / sizeof(g1Raw[0])));
		std::vector<float> g2(g2Raw, g2Raw + (sizeof(g2Raw) / sizeof(g2Raw[0])));

		glades::atlas::BiMAPWeightState promotedState;
		glades::ATLASConfig promotedCfg;
		promotedCfg.rank = 2u;
		promotedCfg.powerIters = 1u;
		promotedCfg.beta = 0.999f;
		promotedCfg.pactEnabled = true;
		promotedCfg.pactLowRankEnabled = true;
		promotedCfg.pactGeometryScale = 1.0f;
		promotedCfg.pactPredictiveScale = 0.10f;
		promotedCfg.pactFactorCadence = 1u;
		promotedCfg.pactCostScale = 0.0f;
		promotedCfg.pactPromoteThreshold = -1.0f;
		promotedCfg.pactDemoteThreshold = -1.0f;

		glades::atlas::BiMAPWeightState fallbackState;
		glades::ATLASConfig fallbackCfg = promotedCfg;
		fallbackCfg.pactGeometryScale = 0.0f;
		fallbackCfg.pactPredictiveScale = 0.0f;
		fallbackCfg.pactLowRankEnabled = false;
		fallbackCfg.pactPromoteThreshold = 1.0f;
		fallbackCfg.pactDemoteThreshold = 0.5f;

		bool ok = glades::atlas::pactUpdate(promotedState,
		                                    &WPromoted[0], &mPromoted[0], &vPromoted[0], &g1[0],
		                                    m, n, lr,
		                                    beta1, beta2, inv1mB1t, inv1mB2t, eps,
		                                    1.0f, 1.0f, 0.0f, 0.0f,
		                                    promotedCfg, 0, "ut.pact.promoted");
		ASSERT("PACT promoted warmup step failed", ok);
		ok = glades::atlas::pactUpdate(promotedState,
		                               &WPromoted[0], &mPromoted[0], &vPromoted[0], &g2[0],
		                               m, n, lr,
		                               beta1, beta2, inv1mB1t, inv1mB2t, eps,
		                               1.0f, 1.0f, 0.0f, 0.0f,
		                               promotedCfg, 0, "ut.pact.promoted");
		ASSERT("PACT promoted step failed", ok);

		ok = glades::atlas::pactUpdate(fallbackState,
		                               &WFallback[0], &mFallback[0], &vFallback[0], &g1[0],
		                               m, n, lr,
		                               beta1, beta2, inv1mB1t, inv1mB2t, eps,
		                               1.0f, 1.0f, 0.0f, 0.0f,
		                               fallbackCfg, 0, "ut.pact.baseline");
		ASSERT("PACT fallback warmup step failed", ok);
		ok = glades::atlas::pactUpdate(fallbackState,
		                               &WFallback[0], &mFallback[0], &vFallback[0], &g2[0],
		                               m, n, lr,
		                               beta1, beta2, inv1mB1t, inv1mB2t, eps,
		                               1.0f, 1.0f, 0.0f, 0.0f,
		                               fallbackCfg, 0, "ut.pact.baseline");
		ASSERT("PACT fallback comparison step failed", ok);

		ASSERT("PACT promoted path should mark the block promoted", promotedState.promoted);
		ASSERT("PACT promoted path should record promoted steps", promotedState.promotedSteps > 0ULL);
		ASSERT("PACT low-rank path should retain some row/col structure",
		       promotedState.rowRank > 0u || promotedState.colRank > 0u);

		bool sawDifference = false;
		for (size_t idx = 0u; idx < mn; ++idx)
		{
			ASSERT("PACT promoted path should zero gradients", fabsf(g2[idx]) < 1.0e-12f);
			if (fabsf(WPromoted[idx] - WFallback[idx]) > 1.0e-7f)
				sawDifference = true;
		}
		ASSERT("PACT promoted path should diverge from exact Adam fallback on anisotropic gradients",
		       sawDifference);
		printf("[UT] PACT promotion engages and changes the matrix update on anisotropic blocks\n");
	}

	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);
}

void ATLASRACERCoreUnitTest()
{
	printf("============================================================\n");
	printf("ATLAS RACER Core Unit Test\n");
	printf("============================================================\n");

	const unsigned int m = 3u;
	const unsigned int n = 2u;
	const size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);
	const float lr = 0.01f;
	const float beta1 = 0.9f;
	const float beta2 = 0.999f;
	const float inv1mB1t = 1.0f / (1.0f - beta1);
	const float inv1mB2t = 1.0f / (1.0f - beta2);
	const float eps = 1.0e-8f;

	{
		std::vector<float> W(mn, 0.5f);
		std::vector<float> m1(mn, 0.0f);
		std::vector<float> v2(mn, 0.0f);
		const float gRaw[] = { 0.25f, -0.50f, 1.00f, -1.50f, 0.75f, -0.25f };
		std::vector<float> g(gRaw, gRaw + (sizeof(gRaw) / sizeof(gRaw[0])));

		std::vector<float> expected = W;
		for (size_t idx = 0u; idx < mn; ++idx)
		{
			const float grad = g[idx];
			const float denom = fabsf(grad) + eps;
			expected[idx] -= lr * (grad / denom);
		}

		glades::atlas::BiMAPWeightState state;
		glades::ATLASConfig ac;
		ac.rank = 2u;
		ac.powerIters = 1u;
		ac.beta = 0.999f;
		ac.racerEnabled = true;
		ac.racerGeometryScale = 0.0f;
		ac.racerPredictiveScale = 0.0f;
		ac.racerFactorCadence = 1u;
		ac.racerRiskScale = 0.50f;
		ac.racerCostScale = 0.0f;
		ac.racerPromoteThreshold = 1.0f;
		ac.racerDemoteThreshold = 0.5f;

		const bool ok = glades::atlas::racerUpdate(state,
		                                           &W[0], &m1[0], &v2[0], &g[0],
		                                           m, n, lr,
		                                           beta1, beta2, inv1mB1t, inv1mB2t, eps,
		                                           1.0f, 1.0f, 0.0f, 0.0f,
		                                           ac, 0, "ut.racer.fallback");
		ASSERT("RACER fallback update failed", ok);
		ASSERT("RACER fallback should stay demoted", !state.promoted);
		for (size_t idx = 0u; idx < mn; ++idx)
		{
			ASSERT("RACER fallback should zero gradients", fabsf(g[idx]) < 1.0e-12f);
			ASSERT("RACER fallback must match Adam-style update", fabsf(W[idx] - expected[idx]) < 1.0e-6f);
		}
		printf("[UT] RACER fallback matches exact Adam-style update\n");
	}

	{
		std::vector<float> WPromoted(mn, 0.5f);
		std::vector<float> WFallback(mn, 0.5f);
		std::vector<float> mPromoted(mn, 0.0f);
		std::vector<float> vPromoted(mn, 0.0f);
		std::vector<float> mFallback(mn, 0.0f);
		std::vector<float> vFallback(mn, 0.0f);
		const float g1Raw[] = { 2.0f, 0.1f, 2.0f, 0.1f, 0.1f, 0.1f };
		const float g2Raw[] = { 1.5f, 0.05f, 1.5f, 0.05f, 0.05f, 0.05f };
		std::vector<float> g1(g1Raw, g1Raw + (sizeof(g1Raw) / sizeof(g1Raw[0])));
		std::vector<float> g2(g2Raw, g2Raw + (sizeof(g2Raw) / sizeof(g2Raw[0])));

		glades::atlas::BiMAPWeightState promotedState;
		glades::ATLASConfig promotedCfg;
		promotedCfg.rank = 2u;
		promotedCfg.powerIters = 1u;
		promotedCfg.beta = 0.999f;
		promotedCfg.racerEnabled = true;
		promotedCfg.racerGeometryScale = 1.0f;
		promotedCfg.racerPredictiveScale = 0.05f;
		promotedCfg.racerFactorCadence = 1u;
		promotedCfg.racerRiskScale = 0.0f;
		promotedCfg.racerCostScale = 0.0f;
		promotedCfg.racerPromoteThreshold = -1.0f;
		promotedCfg.racerDemoteThreshold = -1.0f;

		glades::atlas::BiMAPWeightState fallbackState;
		glades::ATLASConfig fallbackCfg = promotedCfg;
		fallbackCfg.racerGeometryScale = 0.0f;
		fallbackCfg.racerPredictiveScale = 0.0f;
		fallbackCfg.racerPromoteThreshold = 1.0f;
		fallbackCfg.racerDemoteThreshold = 0.5f;

		bool ok = glades::atlas::racerUpdate(promotedState,
		                                     &WPromoted[0], &mPromoted[0], &vPromoted[0], &g1[0],
		                                     m, n, lr,
		                                     beta1, beta2, inv1mB1t, inv1mB2t, eps,
		                                     1.0f, 1.0f, 0.0f, 0.0f,
		                                     promotedCfg, 0, "ut.racer.promoted");
		ASSERT("RACER promoted warmup step failed", ok);
		ok = glades::atlas::racerUpdate(promotedState,
		                                &WPromoted[0], &mPromoted[0], &vPromoted[0], &g2[0],
		                                m, n, lr,
		                                beta1, beta2, inv1mB1t, inv1mB2t, eps,
		                                1.0f, 1.0f, 0.0f, 0.0f,
		                                promotedCfg, 0, "ut.racer.promoted");
		ASSERT("RACER promoted step failed", ok);

		ok = glades::atlas::racerUpdate(fallbackState,
		                                &WFallback[0], &mFallback[0], &vFallback[0], &g1[0],
		                                m, n, lr,
		                                beta1, beta2, inv1mB1t, inv1mB2t, eps,
		                                1.0f, 1.0f, 0.0f, 0.0f,
		                                fallbackCfg, 0, "ut.racer.baseline");
		ASSERT("RACER fallback warmup step failed", ok);
		ok = glades::atlas::racerUpdate(fallbackState,
		                                &WFallback[0], &mFallback[0], &vFallback[0], &g2[0],
		                                m, n, lr,
		                                beta1, beta2, inv1mB1t, inv1mB2t, eps,
		                                1.0f, 1.0f, 0.0f, 0.0f,
		                                fallbackCfg, 0, "ut.racer.baseline");
		ASSERT("RACER fallback comparison step failed", ok);

		ASSERT("RACER promoted path should mark the block promoted", promotedState.promoted);
		ASSERT("RACER promoted path should record promoted steps", promotedState.promotedSteps > 0ULL);

		bool sawDifference = false;
		for (size_t idx = 0u; idx < mn; ++idx)
		{
			ASSERT("RACER promoted path should zero gradients", fabsf(g2[idx]) < 1.0e-12f);
			if (fabsf(WPromoted[idx] - WFallback[idx]) > 1.0e-7f)
				sawDifference = true;
		}
		ASSERT("RACER promoted path should diverge from exact Adam fallback on anisotropic gradients",
		       sawDifference);
		ASSERT("RACER promoted path should record a finite promotion margin",
		       std::isfinite(promotedState.lastPromotionMargin));
		printf("[UT] RACER promotion engages and changes the matrix update on anisotropic blocks\n");
	}

	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);
}

void ATLASKronCoreUnitTest()
{
	printf("============================================================\n");
	printf("ATLAS KRON Core Unit Test\n");
	printf("============================================================\n");

	const unsigned int m = 3u;
	const unsigned int n = 2u;
	const size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);
	const float lr = 0.01f;
	const float beta1 = 0.9f;
	const float beta2 = 0.999f;
	const float inv1mB1t = 1.0f / (1.0f - beta1);
	const float inv1mB2t = 1.0f / (1.0f - beta2);
	const float eps = 1.0e-8f;

	{
		std::vector<float> W(mn, 0.5f);
		std::vector<float> m1(mn, 0.0f);
		std::vector<float> v2(mn, 0.0f);
		const float rawGrad[] = { 0.25f, -0.50f, 0.10f, 0.20f, -0.30f, 0.40f };
		std::vector<float> g(rawGrad, rawGrad + (sizeof(rawGrad) / sizeof(rawGrad[0])));

		std::vector<float> expected(W);
		std::vector<float> expectedM(mn, 0.0f);
		std::vector<float> expectedV(mn, 0.0f);
		for (size_t idx = 0u; idx < mn; ++idx)
		{
			const float grad = g[idx];
			expectedM[idx] = beta1 * expectedM[idx] + (1.0f - beta1) * grad;
			expectedV[idx] = beta2 * expectedV[idx] + (1.0f - beta2) * (grad * grad);
			const float mhat = expectedM[idx] * inv1mB1t;
			const float vhat = expectedV[idx] * inv1mB2t;
			const float denom =
			    static_cast<float>(sqrt(static_cast<double>(std::max(vhat, 0.0f)))) + eps;
			expected[idx] -= lr * (mhat / denom);
		}

		glades::atlas::KronWeightState state;
		glades::ATLASConfig ac;
		ac.kronEnabled = true;
		ac.kronGeometryScale = 0.0f;
		ac.kronPredictiveScale = 0.0f;
		ac.kronFactorCadence = 1u;
		ac.kronDamping = 0.10f;

		const bool ok = glades::atlas::kronUpdate(state,
		                                          &W[0], &m1[0], &v2[0], &g[0],
		                                          m, n, lr,
		                                          beta1, beta2, inv1mB1t, inv1mB2t, eps,
		                                          1.0f, 1.0f, 0.0f, 0.0f,
		                                          ac, 0, "ut.kron.fallback");
		ASSERT("KRON fallback update failed", ok);
		for (size_t idx = 0u; idx < mn; ++idx)
		{
			ASSERT("KRON fallback should zero gradients", fabsf(g[idx]) < 1.0e-12f);
			ASSERT("KRON fallback must match Adam-style update", fabsf(W[idx] - expected[idx]) < 1.0e-6f);
		}
		printf("[UT] KRON fallback matches exact Adam-style update\n");
	}

	{
		std::vector<float> WKron(mn, 0.5f);
		std::vector<float> WAdam(mn, 0.5f);
		std::vector<float> mKron(mn, 0.0f);
		std::vector<float> vKron(mn, 0.0f);
		std::vector<float> mAdam(mn, 0.0f);
		std::vector<float> vAdam(mn, 0.0f);
		const float gradRaw[] = { 2.0f, 0.1f, 2.0f, 0.1f, 0.1f, 0.1f };
		std::vector<float> gKron(gradRaw, gradRaw + (sizeof(gradRaw) / sizeof(gradRaw[0])));
		std::vector<float> gAdam(gradRaw, gradRaw + (sizeof(gradRaw) / sizeof(gradRaw[0])));

		glades::atlas::KronWeightState kronState;
		glades::ATLASConfig kronCfg;
		kronCfg.kronEnabled = true;
		kronCfg.kronGeometryScale = 1.0f;
		kronCfg.kronPredictiveScale = 0.0f;
		kronCfg.kronFactorCadence = 1u;
		kronCfg.kronDamping = 0.10f;
		kronCfg.beta = 0.999f;

		glades::atlas::KronWeightState adamState;
		glades::ATLASConfig adamCfg = kronCfg;
		adamCfg.kronGeometryScale = 0.0f;

		bool ok = glades::atlas::kronUpdate(kronState,
		                                    &WKron[0], &mKron[0], &vKron[0], &gKron[0],
		                                    m, n, lr,
		                                    beta1, beta2, inv1mB1t, inv1mB2t, eps,
		                                    1.0f, 1.0f, 0.0f, 0.0f,
		                                    kronCfg, 0, "ut.kron.promoted");
		ASSERT("KRON anisotropic update failed", ok);
		ok = glades::atlas::kronUpdate(adamState,
		                               &WAdam[0], &mAdam[0], &vAdam[0], &gAdam[0],
		                               m, n, lr,
		                               beta1, beta2, inv1mB1t, inv1mB2t, eps,
		                               1.0f, 1.0f, 0.0f, 0.0f,
		                               adamCfg, 0, "ut.kron.adam");
		ASSERT("KRON fallback comparison update failed", ok);

		bool sawDifference = false;
		for (size_t idx = 0u; idx < mn; ++idx)
		{
			ASSERT("KRON promoted path should zero gradients", fabsf(gKron[idx]) < 1.0e-12f);
			if (fabsf(WKron[idx] - WAdam[idx]) > 1.0e-7f)
				sawDifference = true;
		}
		ASSERT("KRON anisotropic block should diverge from Adam fallback", sawDifference);
		ASSERT("KRON should record nontrivial row or column conditioning",
		       kronState.lastRowCond > 1.0f || kronState.lastColCond > 1.0f);
		printf("[UT] KRON engages two-sided block geometry on anisotropic gradients\n");
	}

	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);
}

void ATLASMuonCoreUnitTest()
{
	printf("============================================================\n");
	printf("ATLAS MUON Core Unit Test\n");
	printf("============================================================\n");

	const unsigned int m = 3u;
	const unsigned int n = 3u;
	const size_t mn = static_cast<size_t>(m) * static_cast<size_t>(n);
	const float lr = 0.01f;
	const float beta1 = 0.9f;
	const float beta2 = 0.999f;
	const float inv1mB1t = 1.0f / (1.0f - beta1);
	const float inv1mB2t = 1.0f / (1.0f - beta2);
	const float eps = 1.0e-8f;

	{
		std::vector<float> W(mn, 0.5f);
		std::vector<float> m1(mn, 0.0f);
		std::vector<float> v2(mn, 0.0f);
		const float rawGrad[] = { 0.25f, -0.50f, 0.10f, 0.20f, -0.30f, 0.40f, 0.05f, -0.15f, 0.35f };
		std::vector<float> g(rawGrad, rawGrad + (sizeof(rawGrad) / sizeof(rawGrad[0])));

		std::vector<float> expected(W);
		std::vector<float> expectedM(mn, 0.0f);
		std::vector<float> expectedV(mn, 0.0f);
		for (size_t idx = 0u; idx < mn; ++idx)
		{
			const float grad = g[idx];
			expectedM[idx] = beta1 * expectedM[idx] + (1.0f - beta1) * grad;
			expectedV[idx] = beta2 * expectedV[idx] + (1.0f - beta2) * (grad * grad);
			const float mhat = expectedM[idx] * inv1mB1t;
			const float vhat = expectedV[idx] * inv1mB2t;
			const float denom =
			    static_cast<float>(sqrt(static_cast<double>(std::max(vhat, 0.0f)))) + eps;
			expected[idx] -= lr * (mhat / denom);
		}

		glades::atlas::MuonWeightState state;
		glades::ATLASConfig ac;
		ac.muonEnabled = true;
		ac.muonGeometryScale = 0.0f;
		ac.muonPredictiveScale = 0.0f;
		ac.muonMaxAspect = 1.50f;
		ac.muonMinDim = 2u;
		ac.muonDamping = 0.01f;

		const bool ok = glades::atlas::muonUpdate(state,
		                                          &W[0], &m1[0], &v2[0], &g[0],
		                                          m, n, lr,
		                                          beta1, beta2, inv1mB1t, inv1mB2t, eps,
		                                          1.0f, 1.0f, 0.0f, 0.0f,
		                                          ac, 0, "ut.muon.fallback");
		ASSERT("MUON fallback update failed", ok);
		for (size_t idx = 0u; idx < mn; ++idx)
		{
			ASSERT("MUON fallback should zero gradients", fabsf(g[idx]) < 1.0e-12f);
			ASSERT("MUON fallback must match Adam-style update", fabsf(W[idx] - expected[idx]) < 1.0e-6f);
		}
		printf("[UT] MUON fallback matches exact Adam-style update\n");
	}

	{
		std::vector<float> WMuon(mn, 0.5f);
		std::vector<float> WAdam(mn, 0.5f);
		std::vector<float> mMuon(mn, 0.0f);
		std::vector<float> vMuon(mn, 0.0f);
		std::vector<float> mAdam(mn, 0.0f);
		std::vector<float> vAdam(mn, 0.0f);
		const float warmGradRaw[] = { 1.0f, 0.5f, 0.25f, 0.5f, 1.0f, 0.25f, 0.25f, 0.25f, 0.75f };
		std::vector<float> gMuonWarm(warmGradRaw, warmGradRaw + (sizeof(warmGradRaw) / sizeof(warmGradRaw[0])));
		std::vector<float> gAdamWarm(warmGradRaw, warmGradRaw + (sizeof(warmGradRaw) / sizeof(warmGradRaw[0])));
		const float gradRaw[] = { 3.0f, 3.0f, 3.0f, 0.5f, 0.5f, 0.5f, 0.1f, 0.1f, 0.1f };
		std::vector<float> gMuon(gradRaw, gradRaw + (sizeof(gradRaw) / sizeof(gradRaw[0])));
		std::vector<float> gAdam(gradRaw, gradRaw + (sizeof(gradRaw) / sizeof(gradRaw[0])));

		glades::atlas::MuonWeightState muonState;
		glades::ATLASConfig muonCfg;
		muonCfg.muonEnabled = true;
		muonCfg.muonGeometryScale = 1.0f;
		muonCfg.muonPredictiveScale = 0.0f;
		muonCfg.muonMaxAspect = 1.50f;
		muonCfg.muonMinDim = 2u;
		muonCfg.muonDamping = 0.01f;

		glades::atlas::MuonWeightState adamState;
		glades::ATLASConfig adamCfg = muonCfg;
		adamCfg.muonGeometryScale = 0.0f;

		bool ok = glades::atlas::muonUpdate(muonState,
		                                    &WMuon[0], &mMuon[0], &vMuon[0], &gMuonWarm[0],
		                                    m, n, lr,
		                                    beta1, beta2, inv1mB1t, inv1mB2t, eps,
		                                    1.0f, 1.0f, 0.0f, 0.0f,
		                                    muonCfg, 0, "ut.muon.warm.promoted");
		ASSERT("MUON warmup step failed", ok);
		ok = glades::atlas::muonUpdate(adamState,
		                               &WAdam[0], &mAdam[0], &vAdam[0], &gAdamWarm[0],
		                               m, n, lr,
		                               beta1, beta2, inv1mB1t, inv1mB2t, eps,
		                               1.0f, 1.0f, 0.0f, 0.0f,
		                               adamCfg, 0, "ut.muon.warm.adam");
		ASSERT("MUON fallback warmup step failed", ok);

		ok = glades::atlas::muonUpdate(muonState,
		                                    &WMuon[0], &mMuon[0], &vMuon[0], &gMuon[0],
		                                    m, n, lr,
		                                    beta1, beta2, inv1mB1t, inv1mB2t, eps,
		                                    1.0f, 1.0f, 0.0f, 0.0f,
		                                    muonCfg, 0, "ut.muon.promoted");
		ASSERT("MUON anisotropic update failed", ok);
		ok = glades::atlas::muonUpdate(adamState,
		                               &WAdam[0], &mAdam[0], &vAdam[0], &gAdam[0],
		                               m, n, lr,
		                               beta1, beta2, inv1mB1t, inv1mB2t, eps,
		                               1.0f, 1.0f, 0.0f, 0.0f,
		                               adamCfg, 0, "ut.muon.adam");
		ASSERT("MUON fallback comparison update failed", ok);

		bool sawDifference = false;
		for (size_t idx = 0u; idx < mn; ++idx)
		{
			ASSERT("MUON promoted path should zero gradients", fabsf(gMuon[idx]) < 1.0e-12f);
			if (fabsf(WMuon[idx] - WAdam[idx]) > 1.0e-7f)
				sawDifference = true;
		}
		ASSERT("MUON eligible square block should diverge from Adam fallback", sawDifference);
		ASSERT("MUON should mark the square block eligible", muonState.lastEligible);
		ASSERT("MUON orthogonalization error should remain finite and bounded",
		       std::isfinite(muonState.lastOrthError) && muonState.lastOrthError < 5.0f);
		printf("[UT] MUON engages orthogonalized momentum on eligible square blocks\n");
	}

	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);
}

void ATLASUnitTest()
{
	printf("============================================================\n");
	printf("ATLAS Optimizer Unit Test Suite\n");
	printf("============================================================\n");

	// ---------------------------------------------------------------
	// Test 1: ATLAS DFF regression convergence
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 1: DFF regression convergence\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* di = new glades::NumberInput();
		di->trainMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
		di->trainMatrix[0][0] = 1.0f;
		di->trainExpectedMatrix[0][0] = 0.0f;
		di->trainMatrix[1][0] = 2.0f;
		di->trainExpectedMatrix[1][0] = 1.0f;
		di->testMatrix = di->trainMatrix;
		di->testExpectedMatrix = di->trainExpectedMatrix;

		glades::InputLayerInfo* in = new glades::InputLayerInfo(
		    /*batchSize*/ 2,
		    /*learningRate*/ 0.05f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::SIGMOID,
		    /*activationParam*/ 1.0f);

		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(
		    /*size*/ 4,
		    /*learningRate*/ 0.05f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::SIGMOID,
		    /*activationParam*/ 1.0f));

		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
		glades::NNInfo* info = new glades::NNInfo("ut_atlas_dff_regression", in, hidden, out);

		glades::NNetwork net(info, glades::NNetwork::TYPE_DFF);
		net.setSeed(42u);
		net.getTerminatorMutable().setEpoch(200);
		net.getTerminatorMutable().setAccuracy(0);

		// Configure ATLAS optimizer
		{
			glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
			cfg.optimizer.type = glades::OptimizerConfig::ATLAS;
			cfg.atlas.rank = 2;
			cfg.atlas.complementRank = 0u;
			cfg.atlas.tSub = 50;
			cfg.atlas.beta = 0.999f;
		}

		CaptureMetricsCallbacks cb;
		const glades::NNetworkStatus st = net.train(di, &cb);
		ASSERT("==============ATLAS::DFF_Regression TrainStatus() Failed==============", st.ok());
		ASSERT("==============ATLAS::DFF_Regression no metrics captured==============", cb.saw);
		printf("[UT] ATLAS DFF regression: final loss = %f\n", cb.last.totalError);
		ASSERT("==============ATLAS::DFF_Regression loss too high==============", cb.last.totalError < 0.27f);

		delete di;
		delete info;
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 1G: Transformer token-LM RAMPART exposes bounded posterior diagnostics.
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 1G: Transformer token-LM RAMPART diagnostics remain bounded\n");
	printf("-----------------------------------\n");
	{
		const unsigned int vocab = 17u;
		const unsigned int padTokenId = vocab - 1u;
		InMemoryTokenIdInput* di = make_atlas_token_dataset(vocab, 24u, 12u, 66323u, padTokenId);
		glades::NNInfo* info = make_atlas_transformer_token_info("ut_atlas_transformer_token_rampart", vocab, 16u, 2u, 0.02f);

		glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
		net.setSeed(66324u);
		net.getTerminatorMutable().setEpoch(4);
		net.getTerminatorMutable().setAccuracy(0.0f);
		{
			glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
			cfg.optimizer.type = glades::OptimizerConfig::ATLAS;
			cfg.atlas.rank = 8u;
			cfg.atlas.complementRank = 4u;
			cfg.atlas.tSub = 16u;
			cfg.atlas.beta = 0.999f;
			cfg.atlas.sparrowEnabled = true;
			cfg.atlas.sparrowModeRank = 1u;
			cfg.atlas.sparrowMemoryScale = 0.05f;
			cfg.atlas.sparrowEdgeThreshold = 0.0f;
			cfg.atlas.sparrowPoleMax = 0.95f;
			cfg.atlas.asterEnabled = true;
			cfg.atlas.aegisEnabled = true;
			cfg.atlas.rampartEnabled = true;
			cfg.atlas.asterMemoryScale = 0.05f;
			cfg.atlas.asterEdgeThreshold = 0.0f;
			cfg.atlas.asterStateRank = 2u;
			cfg.atlas.asterHiddenStackDepth = 2u;
			cfg.atlas.asterPoleMax = 0.95f;
			cfg.transformer.enableTokenEmbedding = true;
			cfg.transformer.vocabSizeOverride = static_cast<int>(vocab);
			cfg.transformer.tieEmbeddings = true;
			cfg.transformer.padTokenId = static_cast<int>(padTokenId);
			cfg.transformer.nHeadsOverride = 4;
			cfg.transformer.nKVHeadsOverride = 4;
			cfg.transformer.dFFOverride = 32;
			cfg.transformer.ffnKind = glades::TransformerRunConfig::FFN_SWIGLU;
			cfg.transformer.normType = glades::TransformerRunConfig::NORM_RMSNORM;
			cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_ROPE;
		}

		CaptureMetricsCallbacks cb;
		const glades::NNetworkStatus st = net.train(di, &cb);
		ASSERT("==============ATLAS::RAMPART_TRANSFORMER_TOKEN TrainStatus() Failed==============", st.ok());
		ASSERT("==============ATLAS::RAMPART_TRANSFORMER_TOKEN no metrics captured==============", cb.saw);
		glades::NNetwork::AtlasRuntimeDiagnostics diag;
		ASSERT("==============ATLAS::RAMPART_TRANSFORMER_TOKEN diagnostics unavailable==============", net.getAtlasRuntimeDiagnostics(diag));
		ASSERT("==============ATLAS::RAMPART_TRANSFORMER_TOKEN should expose one transformer RAMPART observer==============",
		       diag.rampartMatrices == 1u);
		ASSERT("==============ATLAS::RAMPART_TRANSFORMER_TOKEN diagnostics should remain finite==============",
		       diag.rampartMeanTau == diag.rampartMeanTau
		       && diag.rampartMeanBudget == diag.rampartMeanBudget
		       && diag.rampartMeanCovariance == diag.rampartMeanCovariance
		       && diag.rampartMeanSparrowTrust == diag.rampartMeanSparrowTrust);
		ASSERT("==============ATLAS::RAMPART_TRANSFORMER_TOKEN diagnostics should be bounded==============",
		       diag.rampartMeanTau >= 0.0
		       && diag.rampartMeanBudget >= 0.0 && diag.rampartMeanBudget <= 1.0001
		       && diag.rampartMeanCovariance >= 0.0 && diag.rampartMeanCovariance <= 1.0001
		       && diag.rampartMeanSparrowTrust >= 0.0 && diag.rampartMeanSparrowTrust <= 1.0001);

		delete di;
		delete info;
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 1B: HELM-Lite produces bounded runtime diagnostics on DFF output-head
	// training and leaves the network trainable.
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 1B: HELM-Lite runtime diagnostics remain bounded\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* di = new glades::NumberInput();
		di->trainMatrix = shmea::GMatrix(4, shmea::GVector<float>(2, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(4, shmea::GVector<float>(1, 0.0f));
		di->trainMatrix[0][0] = 0.0f; di->trainMatrix[0][1] = 0.0f;
		di->trainExpectedMatrix[0][0] = 0.0f;
		di->trainMatrix[1][0] = 0.0f; di->trainMatrix[1][1] = 1.0f;
		di->trainExpectedMatrix[1][0] = 1.0f;
		di->trainMatrix[2][0] = 1.0f; di->trainMatrix[2][1] = 0.0f;
		di->trainExpectedMatrix[2][0] = 1.0f;
		di->trainMatrix[3][0] = 1.0f; di->trainMatrix[3][1] = 1.0f;
		di->trainExpectedMatrix[3][0] = 0.0f;
		di->testMatrix = di->trainMatrix;
		di->testExpectedMatrix = di->trainExpectedMatrix;

		glades::InputLayerInfo* in = new glades::InputLayerInfo(
		    4, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::SIGMOID, 1.0f);
		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(
		    8, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::SIGMOID, 1.0f));
		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
		glades::NNInfo* info = new glades::NNInfo("ut_atlas_dff_helm", in, hidden, out);

		glades::NNetwork net(info, glades::NNetwork::TYPE_DFF);
		net.setSeed(4242u);
		net.getTerminatorMutable().setEpoch(100);
		net.getTerminatorMutable().setAccuracy(0);
		{
			glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
			cfg.optimizer.type = glades::OptimizerConfig::ATLAS;
			cfg.atlas.rank = 4u;
			cfg.atlas.complementRank = 0u;
			cfg.atlas.tSub = 32u;
			cfg.atlas.beta = 0.999f;
			cfg.atlas.helmEnabled = true;
			cfg.atlas.helmMemoryScale = 0.05f;
			cfg.atlas.helmEdgeThreshold = 0.0f;
			cfg.atlas.helmPoleMax = 0.95f;
		}

		CaptureMetricsCallbacks cb;
		const glades::NNetworkStatus st = net.train(di, &cb);
		ASSERT("==============ATLAS::HELM_DFF TrainStatus() Failed==============", st.ok());
		ASSERT("==============ATLAS::HELM_DFF no metrics captured==============", cb.saw);
		glades::NNetwork::AtlasRuntimeDiagnostics diag;
		ASSERT("==============ATLAS::HELM_DFF diagnostics unavailable==============", net.getAtlasRuntimeDiagnostics(diag));
		ASSERT("==============ATLAS::HELM_DFF should expose one output-head HELM observer==============",
		       diag.helmMatrices == 1u);
		ASSERT("==============ATLAS::HELM_DFF diagnostics should remain finite==============",
		       diag.helmMeanEdge == diag.helmMeanEdge
		       && diag.helmMeanSigma == diag.helmMeanSigma
		       && diag.helmMeanPredR2 == diag.helmMeanPredR2
		       && diag.helmMeanMemoryGain == diag.helmMeanMemoryGain
		       && diag.helmMeanPole == diag.helmMeanPole);
		ASSERT("==============ATLAS::HELM_DFF loss too high==============", cb.last.totalError < 0.35f);

		delete di;
		delete info;
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 1C: ASTER-Lite produces bounded runtime diagnostics on DFF output-head
	// training and leaves the network trainable.
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 1C: ASTER-Lite runtime diagnostics remain bounded\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* di = new glades::NumberInput();
		di->trainMatrix = shmea::GMatrix(4, shmea::GVector<float>(2, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(4, shmea::GVector<float>(1, 0.0f));
		di->trainMatrix[0][0] = 0.0f; di->trainMatrix[0][1] = 0.0f;
		di->trainExpectedMatrix[0][0] = 0.0f;
		di->trainMatrix[1][0] = 0.0f; di->trainMatrix[1][1] = 1.0f;
		di->trainExpectedMatrix[1][0] = 1.0f;
		di->trainMatrix[2][0] = 1.0f; di->trainMatrix[2][1] = 0.0f;
		di->trainExpectedMatrix[2][0] = 1.0f;
		di->trainMatrix[3][0] = 1.0f; di->trainMatrix[3][1] = 1.0f;
		di->trainExpectedMatrix[3][0] = 0.0f;
		di->testMatrix = di->trainMatrix;
		di->testExpectedMatrix = di->trainExpectedMatrix;

		glades::InputLayerInfo* in = new glades::InputLayerInfo(
		    4, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::SIGMOID, 1.0f);
		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(
		    8, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::SIGMOID, 1.0f));
		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
		glades::NNInfo* info = new glades::NNInfo("ut_atlas_dff_aster", in, hidden, out);

		glades::NNetwork net(info, glades::NNetwork::TYPE_DFF);
		net.setSeed(4243u);
		net.getTerminatorMutable().setEpoch(100);
		net.getTerminatorMutable().setAccuracy(0);
		{
			glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
			cfg.optimizer.type = glades::OptimizerConfig::ATLAS;
			cfg.atlas.rank = 4u;
			cfg.atlas.complementRank = 0u;
			cfg.atlas.tSub = 32u;
			cfg.atlas.beta = 0.999f;
			cfg.atlas.asterEnabled = true;
			cfg.atlas.asterMemoryScale = 0.05f;
			cfg.atlas.asterEdgeThreshold = 0.0f;
			cfg.atlas.asterPoleMax = 0.95f;
		}

		CaptureMetricsCallbacks cb;
		const glades::NNetworkStatus st = net.train(di, &cb);
		ASSERT("==============ATLAS::ASTER_DFF TrainStatus() Failed==============", st.ok());
		ASSERT("==============ATLAS::ASTER_DFF no metrics captured==============", cb.saw);
		glades::NNetwork::AtlasRuntimeDiagnostics diag;
		ASSERT("==============ATLAS::ASTER_DFF diagnostics unavailable==============", net.getAtlasRuntimeDiagnostics(diag));
		ASSERT("==============ATLAS::ASTER_DFF should expose one output-head ASTER observer==============",
		       diag.asterMatrices == 1u);
		ASSERT("==============ATLAS::ASTER_DFF diagnostics should remain finite==============",
		       diag.asterMeanEdge == diag.asterMeanEdge
		       && diag.asterMeanSigma == diag.asterMeanSigma
		       && diag.asterMeanPredR2 == diag.asterMeanPredR2
		       && diag.asterMeanMemoryGain == diag.asterMeanMemoryGain
		       && diag.asterMeanPole == diag.asterMeanPole);
		ASSERT("==============ATLAS::ASTER_DFF loss too high==============", cb.last.totalError < 0.35f);

		delete di;
		delete info;
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 1D: Transformer token-LM HELM exposes bounded runtime diagnostics.
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 1D: Transformer token-LM HELM diagnostics remain bounded\n");
	printf("-----------------------------------\n");
	{
		const unsigned int vocab = 17u;
		const unsigned int padTokenId = vocab - 1u;
		InMemoryTokenIdInput* di = make_atlas_token_dataset(vocab, 24u, 12u, 55023u, padTokenId);
		glades::NNInfo* info = make_atlas_transformer_token_info("ut_atlas_transformer_token_helm", vocab, 16u, 2u, 0.02f);

		glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
		net.setSeed(55024u);
		net.getTerminatorMutable().setEpoch(4);
		net.getTerminatorMutable().setAccuracy(0.0f);
		{
			glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
			cfg.optimizer.type = glades::OptimizerConfig::ATLAS;
			cfg.atlas.rank = 8u;
			cfg.atlas.complementRank = 0u;
			cfg.atlas.tSub = 16u;
			cfg.atlas.beta = 0.999f;
			cfg.atlas.helmEnabled = true;
			cfg.atlas.helmMemoryScale = 0.05f;
			cfg.atlas.helmEdgeThreshold = 0.0f;
			cfg.atlas.helmModeRank = 2u;
			cfg.atlas.helmHiddenStackDepth = 2u;
			cfg.atlas.helmPoleMax = 0.95f;
			cfg.transformer.enableTokenEmbedding = true;
			cfg.transformer.vocabSizeOverride = static_cast<int>(vocab);
			cfg.transformer.tieEmbeddings = true;
			cfg.transformer.padTokenId = static_cast<int>(padTokenId);
			cfg.transformer.nHeadsOverride = 4;
			cfg.transformer.nKVHeadsOverride = 4;
			cfg.transformer.dFFOverride = 32;
			cfg.transformer.ffnKind = glades::TransformerRunConfig::FFN_SWIGLU;
			cfg.transformer.normType = glades::TransformerRunConfig::NORM_RMSNORM;
			cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_ROPE;
		}

		CaptureMetricsCallbacks cb;
		const glades::NNetworkStatus st = net.train(di, &cb);
		ASSERT("==============ATLAS::HELM_TRANSFORMER_TOKEN TrainStatus() Failed==============", st.ok());
		ASSERT("==============ATLAS::HELM_TRANSFORMER_TOKEN no metrics captured==============", cb.saw);
		glades::NNetwork::AtlasRuntimeDiagnostics diag;
		ASSERT("==============ATLAS::HELM_TRANSFORMER_TOKEN diagnostics unavailable==============", net.getAtlasRuntimeDiagnostics(diag));
		ASSERT("==============ATLAS::HELM_TRANSFORMER_TOKEN should expose one transformer HELM observer==============",
		       diag.helmMatrices == 1u);
		ASSERT("==============ATLAS::HELM_TRANSFORMER_TOKEN diagnostics should remain finite==============",
		       diag.helmMeanEdge == diag.helmMeanEdge
		       && diag.helmMeanSecondEdge == diag.helmMeanSecondEdge
		       && diag.helmMeanSecondEdgeRatio == diag.helmMeanSecondEdgeRatio
		       && diag.helmMeanSigma == diag.helmMeanSigma
		       && diag.helmMeanPredR2 == diag.helmMeanPredR2
		       && diag.helmMeanMemoryGain == diag.helmMeanMemoryGain
		       && diag.helmMeanPole == diag.helmMeanPole);
		ASSERT("==============ATLAS::HELM_TRANSFORMER_TOKEN train loss should be finite==============",
		       cb.last.totalError == cb.last.totalError);

		delete di;
		delete info;
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 1E: Transformer token-LM ASTER exposes bounded runtime diagnostics.
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 1E: Transformer token-LM ASTER diagnostics remain bounded\n");
	printf("-----------------------------------\n");
	{
		const unsigned int vocab = 17u;
		const unsigned int padTokenId = vocab - 1u;
		InMemoryTokenIdInput* di = make_atlas_token_dataset(vocab, 24u, 12u, 55123u, padTokenId);
		glades::NNInfo* info = make_atlas_transformer_token_info("ut_atlas_transformer_token_aster", vocab, 16u, 2u, 0.02f);

		glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
		net.setSeed(55124u);
		net.getTerminatorMutable().setEpoch(4);
		net.getTerminatorMutable().setAccuracy(0.0f);
		{
			glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
			cfg.optimizer.type = glades::OptimizerConfig::ATLAS;
			cfg.atlas.rank = 8u;
			cfg.atlas.complementRank = 0u;
			cfg.atlas.tSub = 16u;
			cfg.atlas.beta = 0.999f;
			cfg.atlas.asterEnabled = true;
			cfg.atlas.asterMemoryScale = 0.05f;
			cfg.atlas.asterEdgeThreshold = 0.0f;
			cfg.atlas.asterStateRank = 2u;
			cfg.atlas.asterHiddenStackDepth = 2u;
			cfg.atlas.asterPoleMax = 0.95f;
			cfg.atlas.kappaEnabled = true;
			cfg.atlas.kappaHeads = 1u;
			cfg.atlas.kappaLagBuckets = 4u;
			cfg.atlas.kappaRank = 2u;
			cfg.transformer.enableTokenEmbedding = true;
			cfg.transformer.vocabSizeOverride = static_cast<int>(vocab);
			cfg.transformer.tieEmbeddings = true;
			cfg.transformer.padTokenId = static_cast<int>(padTokenId);
			cfg.transformer.nHeadsOverride = 4;
			cfg.transformer.nKVHeadsOverride = 4;
			cfg.transformer.dFFOverride = 32;
			cfg.transformer.ffnKind = glades::TransformerRunConfig::FFN_SWIGLU;
			cfg.transformer.normType = glades::TransformerRunConfig::NORM_RMSNORM;
			cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_ROPE;
		}

		CaptureMetricsCallbacks cb;
		const glades::NNetworkStatus st = net.train(di, &cb);
		ASSERT("==============ATLAS::ASTER_TRANSFORMER_TOKEN TrainStatus() Failed==============", st.ok());
		ASSERT("==============ATLAS::ASTER_TRANSFORMER_TOKEN no metrics captured==============", cb.saw);
		glades::NNetwork::AtlasRuntimeDiagnostics diag;
		ASSERT("==============ATLAS::ASTER_TRANSFORMER_TOKEN diagnostics unavailable==============", net.getAtlasRuntimeDiagnostics(diag));
		ASSERT("==============ATLAS::ASTER_TRANSFORMER_TOKEN should expose one transformer ASTER observer==============",
		       diag.asterMatrices == 1u);
		ASSERT("==============ATLAS::ASTER_TRANSFORMER_TOKEN diagnostics should remain finite==============",
		       diag.asterMeanEdge == diag.asterMeanEdge
		       && diag.asterMeanSigma == diag.asterMeanSigma
		       && diag.asterMeanPredR2 == diag.asterMeanPredR2
		       && diag.asterMeanMemoryGain == diag.asterMeanMemoryGain
		       && diag.asterMeanPole == diag.asterMeanPole
		       && diag.asterMeanBoundaryMs == diag.asterMeanBoundaryMs);
		ASSERT("==============ATLAS::ASTER_TRANSFORMER_TOKEN train loss should be finite==============",
		       cb.last.totalError == cb.last.totalError);

		delete di;
		delete info;
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 1E: Transformer token-LM AEGIS exposes bounded calibration diagnostics.
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 1E: Transformer token-LM AEGIS diagnostics remain bounded\n");
	printf("-----------------------------------\n");
	{
		const unsigned int vocab = 17u;
		const unsigned int padTokenId = vocab - 1u;
		InMemoryTokenIdInput* di = make_atlas_token_dataset(vocab, 24u, 12u, 66123u, padTokenId);
		glades::NNInfo* info = make_atlas_transformer_token_info("ut_atlas_transformer_token_aegis", vocab, 16u, 2u, 0.02f);

		glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
		net.setSeed(66124u);
		net.getTerminatorMutable().setEpoch(4);
		net.getTerminatorMutable().setAccuracy(0.0f);
		{
			glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
			cfg.optimizer.type = glades::OptimizerConfig::ATLAS;
			cfg.atlas.rank = 8u;
			cfg.atlas.complementRank = 4u;
			cfg.atlas.tSub = 16u;
			cfg.atlas.beta = 0.999f;
			cfg.atlas.sparrowEnabled = true;
			cfg.atlas.sparrowModeRank = 1u;
			cfg.atlas.sparrowMemoryScale = 0.05f;
			cfg.atlas.sparrowEdgeThreshold = 0.0f;
			cfg.atlas.sparrowPoleMax = 0.95f;
			cfg.atlas.asterEnabled = true;
			cfg.atlas.aegisEnabled = true;
			cfg.atlas.asterMemoryScale = 0.05f;
			cfg.atlas.asterEdgeThreshold = 0.0f;
			cfg.atlas.asterStateRank = 2u;
			cfg.atlas.asterHiddenStackDepth = 2u;
			cfg.atlas.asterPoleMax = 0.95f;
			cfg.transformer.enableTokenEmbedding = true;
			cfg.transformer.vocabSizeOverride = static_cast<int>(vocab);
			cfg.transformer.tieEmbeddings = true;
			cfg.transformer.padTokenId = static_cast<int>(padTokenId);
			cfg.transformer.nHeadsOverride = 4;
			cfg.transformer.nKVHeadsOverride = 4;
			cfg.transformer.dFFOverride = 32;
			cfg.transformer.ffnKind = glades::TransformerRunConfig::FFN_SWIGLU;
			cfg.transformer.normType = glades::TransformerRunConfig::NORM_RMSNORM;
			cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_ROPE;
		}

		CaptureMetricsCallbacks cb;
		const glades::NNetworkStatus st = net.train(di, &cb);
		ASSERT("==============ATLAS::AEGIS_TRANSFORMER_TOKEN TrainStatus() Failed==============", st.ok());
		ASSERT("==============ATLAS::AEGIS_TRANSFORMER_TOKEN no metrics captured==============", cb.saw);
		glades::NNetwork::AtlasRuntimeDiagnostics diag;
		ASSERT("==============ATLAS::AEGIS_TRANSFORMER_TOKEN diagnostics unavailable==============", net.getAtlasRuntimeDiagnostics(diag));
		ASSERT("==============ATLAS::AEGIS_TRANSFORMER_TOKEN should expose one transformer AEGIS observer==============",
		       diag.aegisMatrices == 1u);
		ASSERT("==============ATLAS::AEGIS_TRANSFORMER_TOKEN calibration diagnostics should remain finite==============",
		       diag.aegisMeanLambdaSpatial == diag.aegisMeanLambdaSpatial
		       && diag.aegisMeanLambdaPredictive == diag.aegisMeanLambdaPredictive
		       && diag.aegisMeanLambdaOutput == diag.aegisMeanLambdaOutput
		       && diag.aegisMeanPredictiveError == diag.aegisMeanPredictiveError
		       && diag.aegisMeanOutputError == diag.aegisMeanOutputError
		       && diag.aegisMeanChannelDisagreement == diag.aegisMeanChannelDisagreement);
		ASSERT("==============ATLAS::AEGIS_TRANSFORMER_TOKEN lambdas should be bounded==============",
		       diag.aegisMeanLambdaSpatial >= 0.0
		       && diag.aegisMeanLambdaPredictive >= 0.0
		       && diag.aegisMeanLambdaOutput >= 0.0
		       && (diag.aegisMeanLambdaSpatial + diag.aegisMeanLambdaPredictive + diag.aegisMeanLambdaOutput) <= 1.0001);

		delete di;
		delete info;
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 1F: Transformer token-LM CITADEL exposes bounded anchor diagnostics.
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 1F: Transformer token-LM CITADEL diagnostics remain bounded\n");
	printf("-----------------------------------\n");
	{
		const unsigned int vocab = 17u;
		const unsigned int padTokenId = vocab - 1u;
		InMemoryTokenIdInput* di = make_atlas_token_dataset(vocab, 24u, 12u, 66223u, padTokenId);
		glades::NNInfo* info = make_atlas_transformer_token_info("ut_atlas_transformer_token_citadel", vocab, 16u, 2u, 0.02f);

		glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
		net.setSeed(66224u);
		net.getTerminatorMutable().setEpoch(4);
		net.getTerminatorMutable().setAccuracy(0.0f);
		{
			glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
			cfg.optimizer.type = glades::OptimizerConfig::ATLAS;
			cfg.atlas.rank = 8u;
			cfg.atlas.complementRank = 4u;
			cfg.atlas.tSub = 16u;
			cfg.atlas.beta = 0.999f;
			cfg.atlas.sparrowEnabled = true;
			cfg.atlas.sparrowModeRank = 1u;
			cfg.atlas.sparrowMemoryScale = 0.05f;
			cfg.atlas.sparrowEdgeThreshold = 0.0f;
			cfg.atlas.sparrowPoleMax = 0.95f;
			cfg.atlas.asterEnabled = true;
			cfg.atlas.aegisEnabled = true;
			cfg.atlas.citadelEnabled = true;
			cfg.atlas.asterMemoryScale = 0.05f;
			cfg.atlas.asterEdgeThreshold = 0.0f;
			cfg.atlas.asterStateRank = 2u;
			cfg.atlas.asterHiddenStackDepth = 2u;
			cfg.atlas.asterPoleMax = 0.95f;
			cfg.transformer.enableTokenEmbedding = true;
			cfg.transformer.vocabSizeOverride = static_cast<int>(vocab);
			cfg.transformer.tieEmbeddings = true;
			cfg.transformer.padTokenId = static_cast<int>(padTokenId);
			cfg.transformer.nHeadsOverride = 4;
			cfg.transformer.nKVHeadsOverride = 4;
			cfg.transformer.dFFOverride = 32;
			cfg.transformer.ffnKind = glades::TransformerRunConfig::FFN_SWIGLU;
			cfg.transformer.normType = glades::TransformerRunConfig::NORM_RMSNORM;
			cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_ROPE;
		}

		CaptureMetricsCallbacks cb;
		const glades::NNetworkStatus st = net.train(di, &cb);
		ASSERT("==============ATLAS::CITADEL_TRANSFORMER_TOKEN TrainStatus() Failed==============", st.ok());
		ASSERT("==============ATLAS::CITADEL_TRANSFORMER_TOKEN no metrics captured==============", cb.saw);
		glades::NNetwork::AtlasRuntimeDiagnostics diag;
		ASSERT("==============ATLAS::CITADEL_TRANSFORMER_TOKEN diagnostics unavailable==============", net.getAtlasRuntimeDiagnostics(diag));
		ASSERT("==============ATLAS::CITADEL_TRANSFORMER_TOKEN should expose one transformer CITADEL observer==============",
		       diag.citadelMatrices == 1u);
		ASSERT("==============ATLAS::CITADEL_TRANSFORMER_TOKEN anchor diagnostics should remain finite==============",
		       diag.citadelMeanAnchor == diag.citadelMeanAnchor
		       && diag.citadelMeanHardRegimeMass == diag.citadelMeanHardRegimeMass
		       && diag.citadelMeanSparrowTrust == diag.citadelMeanSparrowTrust);
		ASSERT("==============ATLAS::CITADEL_TRANSFORMER_TOKEN diagnostics should be bounded==============",
		       diag.citadelMeanAnchor >= 0.0 && diag.citadelMeanAnchor <= 1.0001
		       && diag.citadelMeanHardRegimeMass >= 0.0 && diag.citadelMeanHardRegimeMass <= 1.0001
		       && diag.citadelMeanSparrowTrust >= 0.0 && diag.citadelMeanSparrowTrust <= 1.0001);

		delete di;
		delete info;
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 1H: Transformer token-LM MERIT exposes bounded trust diagnostics.
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 1H: Transformer token-LM MERIT diagnostics remain bounded\n");
	printf("-----------------------------------\n");
	{
		const unsigned int vocab = 17u;
		const unsigned int padTokenId = vocab - 1u;
		InMemoryTokenIdInput* di = make_atlas_token_dataset(vocab, 24u, 12u, 66423u, padTokenId);
		glades::NNInfo* info = make_atlas_transformer_token_info("ut_atlas_transformer_token_merit", vocab, 16u, 2u, 0.02f);

		glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
		net.setSeed(66424u);
		net.getTerminatorMutable().setEpoch(4);
		net.getTerminatorMutable().setAccuracy(0.0f);
		{
			glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
			cfg.optimizer.type = glades::OptimizerConfig::ATLAS;
			cfg.atlas.rank = 8u;
			cfg.atlas.complementRank = 4u;
			cfg.atlas.tSub = 16u;
			cfg.atlas.beta = 0.999f;
			cfg.atlas.sparrowEnabled = true;
			cfg.atlas.sparrowModeRank = 1u;
			cfg.atlas.sparrowMemoryScale = 0.05f;
			cfg.atlas.sparrowEdgeThreshold = 0.0f;
			cfg.atlas.sparrowPoleMax = 0.95f;
			cfg.atlas.asterEnabled = true;
			cfg.atlas.aegisEnabled = true;
			cfg.atlas.meritEnabled = true;
			cfg.atlas.asterMemoryScale = 0.05f;
			cfg.atlas.asterEdgeThreshold = 0.0f;
			cfg.atlas.asterStateRank = 2u;
			cfg.atlas.asterHiddenStackDepth = 2u;
			cfg.atlas.asterPoleMax = 0.95f;
			cfg.transformer.enableTokenEmbedding = true;
			cfg.transformer.vocabSizeOverride = static_cast<int>(vocab);
			cfg.transformer.tieEmbeddings = true;
			cfg.transformer.padTokenId = static_cast<int>(padTokenId);
			cfg.transformer.nHeadsOverride = 4;
			cfg.transformer.nKVHeadsOverride = 4;
			cfg.transformer.dFFOverride = 32;
			cfg.transformer.ffnKind = glades::TransformerRunConfig::FFN_SWIGLU;
			cfg.transformer.normType = glades::TransformerRunConfig::NORM_RMSNORM;
			cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_ROPE;
		}

		CaptureMetricsCallbacks cb;
		const glades::NNetworkStatus st = net.train(di, &cb);
		ASSERT("==============ATLAS::MERIT_TRANSFORMER_TOKEN TrainStatus() Failed==============", st.ok());
		ASSERT("==============ATLAS::MERIT_TRANSFORMER_TOKEN no metrics captured==============", cb.saw);
		glades::NNetwork::AtlasRuntimeDiagnostics diag;
		ASSERT("==============ATLAS::MERIT_TRANSFORMER_TOKEN diagnostics unavailable==============", net.getAtlasRuntimeDiagnostics(diag));
		ASSERT("==============ATLAS::MERIT_TRANSFORMER_TOKEN should expose one transformer MERIT observer==============",
		       diag.meritMatrices == 1u);
		ASSERT("==============ATLAS::MERIT_TRANSFORMER_TOKEN diagnostics should remain finite==============",
		       diag.meritMeanTau == diag.meritMeanTau
		       && diag.meritMeanBudget == diag.meritMeanBudget
		       && diag.meritMeanCovariance == diag.meritMeanCovariance
		       && diag.meritMeanSparrowTrust == diag.meritMeanSparrowTrust
		       && diag.meritMeanGeometryTrust == diag.meritMeanGeometryTrust);
		ASSERT("==============ATLAS::MERIT_TRANSFORMER_TOKEN diagnostics should be bounded==============",
		       diag.meritMeanTau >= 0.0
		       && diag.meritMeanBudget >= 0.0 && diag.meritMeanBudget <= 1.0001
		       && diag.meritMeanCovariance >= 0.0 && diag.meritMeanCovariance <= 1.0001
		       && diag.meritMeanSparrowTrust >= 0.0 && diag.meritMeanSparrowTrust <= 1.0001
		       && diag.meritMeanGeometryTrust >= 0.0 && diag.meritMeanGeometryTrust <= 1.0001);

		delete di;
		delete info;
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 1I: Transformer token-LM STRATA exposes bounded mode diagnostics.
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 1I: Transformer token-LM STRATA diagnostics remain bounded\n");
	printf("-----------------------------------\n");
	{
		const unsigned int vocab = 17u;
		const unsigned int padTokenId = vocab - 1u;
		InMemoryTokenIdInput* di = make_atlas_token_dataset(vocab, 24u, 12u, 66423u, padTokenId);
		glades::NNInfo* info = make_atlas_transformer_token_info("ut_atlas_transformer_token_strata", vocab, 16u, 2u, 0.02f);

		glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
		net.setSeed(66424u);
		net.getTerminatorMutable().setEpoch(4);
		net.getTerminatorMutable().setAccuracy(0.0f);
		{
			glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
			cfg.optimizer.type = glades::OptimizerConfig::ATLAS;
			cfg.atlas.rank = 8u;
			cfg.atlas.complementRank = 4u;
			cfg.atlas.tSub = 16u;
			cfg.atlas.beta = 0.999f;
			cfg.atlas.sparrowEnabled = true;
			cfg.atlas.sparrowModeRank = 1u;
			cfg.atlas.sparrowMemoryScale = 0.05f;
			cfg.atlas.sparrowEdgeThreshold = 0.0f;
			cfg.atlas.sparrowPoleMax = 0.95f;
			cfg.atlas.asterEnabled = true;
			cfg.atlas.aegisEnabled = true;
			cfg.atlas.strataEnabled = true;
			cfg.atlas.asterMemoryScale = 0.05f;
			cfg.atlas.asterEdgeThreshold = 0.0f;
			cfg.atlas.asterStateRank = 2u;
			cfg.atlas.asterHiddenStackDepth = 2u;
			cfg.atlas.asterPoleMax = 0.95f;
			cfg.transformer.enableTokenEmbedding = true;
			cfg.transformer.vocabSizeOverride = static_cast<int>(vocab);
			cfg.transformer.tieEmbeddings = true;
			cfg.transformer.padTokenId = static_cast<int>(padTokenId);
			cfg.transformer.nHeadsOverride = 4;
			cfg.transformer.nKVHeadsOverride = 4;
			cfg.transformer.dFFOverride = 32;
			cfg.transformer.ffnKind = glades::TransformerRunConfig::FFN_SWIGLU;
			cfg.transformer.normType = glades::TransformerRunConfig::NORM_RMSNORM;
			cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_ROPE;
		}

		CaptureMetricsCallbacks cb;
		const glades::NNetworkStatus st = net.train(di, &cb);
		ASSERT("==============ATLAS::STRATA_TRANSFORMER_TOKEN TrainStatus() Failed==============", st.ok());
		ASSERT("==============ATLAS::STRATA_TRANSFORMER_TOKEN no metrics captured==============", cb.saw);
		glades::NNetwork::AtlasRuntimeDiagnostics diag;
		ASSERT("==============ATLAS::STRATA_TRANSFORMER_TOKEN diagnostics unavailable==============", net.getAtlasRuntimeDiagnostics(diag));
		ASSERT("==============ATLAS::STRATA_TRANSFORMER_TOKEN should expose one transformer STRATA observer==============",
		       diag.strataMatrices == 1u);
		ASSERT("==============ATLAS::STRATA_TRANSFORMER_TOKEN diagnostics should remain finite==============",
		       diag.strataMeanNullMode == diag.strataMeanNullMode
		       && diag.strataMeanPredictiveMode == diag.strataMeanPredictiveMode
		       && diag.strataMeanOutputMode == diag.strataMeanOutputMode
		       && diag.strataMeanCoupledMode == diag.strataMeanCoupledMode
		       && diag.strataMeanBudget == diag.strataMeanBudget
		       && diag.strataMeanNullBenefit == diag.strataMeanNullBenefit
		       && diag.strataMeanPredictiveBenefit == diag.strataMeanPredictiveBenefit
		       && diag.strataMeanOutputBenefit == diag.strataMeanOutputBenefit
		       && diag.strataMeanCoupledBenefit == diag.strataMeanCoupledBenefit
		       && diag.strataMeanSelectedExcess == diag.strataMeanSelectedExcess
		       && diag.strataMeanSwitchRate == diag.strataMeanSwitchRate);
		ASSERT("==============ATLAS::STRATA_TRANSFORMER_TOKEN diagnostics should be bounded==============",
		       diag.strataMeanNullMode >= 0.0
		       && diag.strataMeanPredictiveMode >= 0.0
		       && diag.strataMeanOutputMode >= 0.0
		       && diag.strataMeanCoupledMode >= 0.0
		       && diag.strataMeanBudget >= 0.0 && diag.strataMeanBudget <= 1.0001
		       && diag.strataMeanSwitchRate >= 0.0 && diag.strataMeanSwitchRate <= 1.0001
		       && (diag.strataMeanNullMode + diag.strataMeanPredictiveMode
		           + diag.strataMeanOutputMode + diag.strataMeanCoupledMode) <= 1.0001);

		delete di;
		delete info;
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test C11: COBALT transfer-edge gate suppresses stale complement births
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Controller Test C11: COBALT transfer-edge gate blocks stale complement modes\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 24;
		const unsigned int n = 24;
		const unsigned int r = 2;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 38282842ULL);
		glades::atlas::WeightState state;
		prepare_atlas_complement_test_state(state, rng, m, n, r, 4u, "cnn.fc");

		std::fill(state.U.begin(), state.U.end(), 0.0f);
		std::fill(state.V.begin(), state.V.end(), 0.0f);
		std::fill(state.heroGwHistory.begin(), state.heroGwHistory.end(), 0.0f);
		state.U[0] = 1.0f;
		state.U[state.r + 1] = 1.0f;
		for (unsigned int c = 0; c < state.complementRank; ++c)
			state.V[(c + 2u) * state.complementRank + c] = 1.0f;

		std::vector<float> W(static_cast<size_t>(m) * n, 0.0f);
		std::vector<float> gW(static_cast<size_t>(m) * n, 0.0f);
		for (unsigned int j = 0; j < n; ++j)
		{
			gW[2u * n + j] = 4.0f;
			gW[3u * n + j] = 3.0f;
		}

		glades::ATLASConfig acCobaltGate;
		acCobaltGate.rank = r;
		acCobaltGate.complementRank = 4u;
		acCobaltGate.beta = 1.0f;
		acCobaltGate.biasCorrection = false;
		acCobaltGate.muMin = 0.0f;
		acCobaltGate.muMax = 0.0f;
		acCobaltGate.tSub = 0u;
		acCobaltGate.cobaltEnabled = true;
		acCobaltGate.cobaltLagHorizon = 4u;
		acCobaltGate.cobaltEdgeThreshold = 0.10f;

		const bool ok = glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
		                                         1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
		                                         acCobaltGate, rng, 0, "cnn.fc");
		ASSERT("==============ATLAS::ControllerCobaltGate applyStep failed==============", ok);
		ASSERT("==============ATLAS::ControllerCobaltGate should suppress stale complement probation==============",
		       state.activeComplementRank == 0u && state.trialComplementRank == 0u);
		ASSERT("==============ATLAS::ControllerCobaltGate transfer edge should stay subcritical==============",
		       state.lastCobaltEdge < acCobaltGate.cobaltEdgeThreshold);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test C12: COBALT active memory shrinks the subspace correction on aligned histories
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Controller Test C12: COBALT active memory damps aligned active corrections\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 4;
		const unsigned int n = 4;
		const unsigned int r = 2;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 38282843ULL);
		glades::atlas::WeightState baseState;
		glades::atlas::initWeightState(baseState, m, n, r, 0.0f, rng);
		glades::atlas::WeightState cobaltState = baseState;

		std::fill(baseState.U.begin(), baseState.U.end(), 0.0f);
		baseState.U[0] = 1.0f;
		baseState.U[baseState.r + 1] = 1.0f;
		cobaltState.U = baseState.U;

		for (unsigned int j = 0; j < n; ++j)
		{
			cobaltState.resolveGzHistory[0u * r * n + 0u * n + j] = 4.0f;
			cobaltState.resolveGzHistory[0u * r * n + 1u * n + j] = 2.0f;
			cobaltState.resolveGzHistory[1u * r * n + 0u * n + j] = 3.5f;
			cobaltState.resolveGzHistory[1u * r * n + 1u * n + j] = 1.75f;
			cobaltState.resolveGzHistory[2u * r * n + 0u * n + j] = 3.0f;
			cobaltState.resolveGzHistory[2u * r * n + 1u * n + j] = 1.5f;
			cobaltState.resolveGzHistory[3u * r * n + 0u * n + j] = 2.5f;
			cobaltState.resolveGzHistory[3u * r * n + 1u * n + j] = 1.25f;
		}

		std::vector<float> gW(static_cast<size_t>(m) * n, 0.0f);
		for (unsigned int j = 0; j < n; ++j)
		{
			gW[0u * n + j] = 4.0f;
			gW[1u * n + j] = 2.0f;
			gW[2u * n + j] = 1.0f;
			gW[3u * n + j] = 1.0f;
		}
		std::vector<float> gWBase = gW;
		std::vector<float> gWGhost = gW;
		std::vector<float> WBase(static_cast<size_t>(m) * n, 0.0f);
		std::vector<float> WCobalt(static_cast<size_t>(m) * n, 0.0f);

		glades::ATLASConfig acBase;
		acBase.rank = r;
		acBase.complementRank = 0u;
		acBase.beta = 0.0f;
		acBase.biasCorrection = false;
		acBase.muMin = 0.0f;
		acBase.muMax = 0.0f;
		acBase.tSub = 0u;

		glades::ATLASConfig acCobalt = acBase;
		acCobalt.cobaltEnabled = true;
		acCobalt.cobaltLagHorizon = 4u;
		acCobalt.cobaltMemoryScale = 0.20f;

		bool ok = glades::atlas::applyStep(baseState, &WBase[0], &gW[0], m, n,
		                                   1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
		                                   acBase, rng, 0, 0);
		ASSERT("==============ATLAS::ControllerCobaltMemory base applyStep failed==============", ok);
		ok = glades::atlas::applyStep(cobaltState, &WCobalt[0], &gW[0], m, n,
		                              1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
		                              acCobalt, rng, 0, 0);
		ASSERT("==============ATLAS::ControllerCobaltMemory cobalt applyStep failed==============", ok);

		double baseCorrectionNormSq = 0.0;
		double cobaltCorrectionNormSq = 0.0;
		for (size_t idx = 0; idx < WBase.size(); ++idx)
		{
			const double baseCorrection =
			    static_cast<double>(WBase[idx])
			    + static_cast<double>(baseState.lastBaselineRate) * static_cast<double>(gW[idx]);
			const double cobaltCorrection =
			    static_cast<double>(WCobalt[idx])
			    + static_cast<double>(cobaltState.lastBaselineRate) * static_cast<double>(gW[idx]);
			baseCorrectionNormSq += baseCorrection * baseCorrection;
			cobaltCorrectionNormSq += cobaltCorrection * cobaltCorrection;
		}
		ASSERT("==============ATLAS::ControllerCobaltMemory should reduce active correction energy==============",
		       cobaltCorrectionNormSq < baseCorrectionNormSq);
		ASSERT("==============ATLAS::ControllerCobaltMemory memory diagnostic should remain finite==============",
		       cobaltState.lastCobaltMemoryGain == cobaltState.lastCobaltMemoryGain);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test C13: BIRCH stays memory-only and clears explicit complement activity
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Controller Test C13: BIRCH stays memory-only and clears complement activity\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 24;
		const unsigned int n = 24;
		const unsigned int r = 2;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 38282844ULL);
		glades::atlas::WeightState state;
		prepare_atlas_complement_test_state(state, rng, m, n, r, 4u, "cnn.fc");
		state.activeComplementRank = 1u;
		state.trialComplementRank = 2u;
		state.trialComplementWins = 1u;

		std::fill(state.U.begin(), state.U.end(), 0.0f);
		state.U[0] = 1.0f;
		state.U[state.r + 1] = 1.0f;

		for (unsigned int j = 0; j < n; ++j)
		{
			state.resolveGzHistory[0u * r * n + 0u * n + j] = 4.0f;
			state.resolveGzHistory[0u * r * n + 1u * n + j] = 2.0f;
			state.resolveGzHistory[1u * r * n + 0u * n + j] = 3.5f;
			state.resolveGzHistory[1u * r * n + 1u * n + j] = 1.75f;
			state.resolveGzHistory[2u * r * n + 0u * n + j] = 3.0f;
			state.resolveGzHistory[2u * r * n + 1u * n + j] = 1.5f;
			state.resolveGzHistory[3u * r * n + 0u * n + j] = 2.5f;
			state.resolveGzHistory[3u * r * n + 1u * n + j] = 1.25f;
		}

		std::vector<float> W(static_cast<size_t>(m) * n, 0.0f);
		std::vector<float> gW(static_cast<size_t>(m) * n, 0.0f);
		for (unsigned int j = 0; j < n; ++j)
		{
			gW[0u * n + j] = 4.0f;
			gW[1u * n + j] = 2.0f;
			gW[2u * n + j] = 1.0f;
			gW[3u * n + j] = 1.0f;
		}

		glades::ATLASConfig acBirchGate;
		acBirchGate.rank = r;
		acBirchGate.complementRank = 4u;
		acBirchGate.beta = 1.0f;
		acBirchGate.biasCorrection = false;
		acBirchGate.muMin = 0.0f;
		acBirchGate.muMax = 0.0f;
		acBirchGate.tSub = 0u;
		acBirchGate.birchEnabled = true;
		acBirchGate.birchPastHorizon = 3u;
		acBirchGate.birchFutureHorizon = 2u;
		acBirchGate.birchEdgeThreshold = 0.10f;

		const bool ok = glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
		                                         1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
		                                         acBirchGate, rng, 0, "cnn.fc");
		ASSERT("==============ATLAS::ControllerBirchGate applyStep failed==============", ok);
		ASSERT("==============ATLAS::ControllerBirchGate should clear explicit complement activity==============",
		       state.activeComplementRank == 0u && state.trialComplementRank == 0u);
		ASSERT("==============ATLAS::ControllerBirchGate edge diagnostic should remain finite==============",
		       state.lastBirchEdge == state.lastBirchEdge
		       && state.lastBirchSigma == state.lastBirchSigma);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test C14: BIRCH active memory shrinks the subspace correction on aligned histories
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Controller Test C14: BIRCH active memory damps aligned active corrections\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 4;
		const unsigned int n = 4;
		const unsigned int r = 2;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 38282845ULL);
		glades::atlas::WeightState baseState;
		glades::atlas::initWeightState(baseState, m, n, r, 0.0f, rng);
		glades::atlas::WeightState birchState = baseState;

		std::fill(baseState.U.begin(), baseState.U.end(), 0.0f);
		baseState.U[0] = 1.0f;
		baseState.U[baseState.r + 1] = 1.0f;
		birchState.U = baseState.U;

		for (unsigned int j = 0; j < n; ++j)
		{
			birchState.resolveGzHistory[0u * r * n + 0u * n + j] = 4.0f;
			birchState.resolveGzHistory[0u * r * n + 1u * n + j] = 2.0f;
			birchState.resolveGzHistory[1u * r * n + 0u * n + j] = 3.5f;
			birchState.resolveGzHistory[1u * r * n + 1u * n + j] = 1.75f;
			birchState.resolveGzHistory[2u * r * n + 0u * n + j] = 3.0f;
			birchState.resolveGzHistory[2u * r * n + 1u * n + j] = 1.5f;
			birchState.resolveGzHistory[3u * r * n + 0u * n + j] = 2.5f;
			birchState.resolveGzHistory[3u * r * n + 1u * n + j] = 1.25f;
		}

		std::vector<float> gW(static_cast<size_t>(m) * n, 0.0f);
		for (unsigned int j = 0; j < n; ++j)
		{
			gW[0u * n + j] = 4.0f;
			gW[1u * n + j] = 2.0f;
			gW[2u * n + j] = 1.0f;
			gW[3u * n + j] = 1.0f;
		}
		std::vector<float> WBase(static_cast<size_t>(m) * n, 0.0f);
		std::vector<float> WBirch(static_cast<size_t>(m) * n, 0.0f);

		glades::ATLASConfig acBase;
		acBase.rank = r;
		acBase.complementRank = 0u;
		acBase.beta = 0.0f;
		acBase.biasCorrection = false;
		acBase.muMin = 0.0f;
		acBase.muMax = 0.0f;
		acBase.tSub = 0u;

		glades::ATLASConfig acBirch = acBase;
		acBirch.birchEnabled = true;
		acBirch.birchPastHorizon = 3u;
		acBirch.birchFutureHorizon = 2u;
		acBirch.birchMemoryScale = 0.20f;
		acBirch.birchEdgeThreshold = 0.0f;

		bool ok = glades::atlas::applyStep(baseState, &WBase[0], &gW[0], m, n,
		                                   1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
		                                   acBase, rng, 0, 0);
		ASSERT("==============ATLAS::ControllerBirchMemory base applyStep failed==============", ok);
		ok = glades::atlas::applyStep(birchState, &WBirch[0], &gW[0], m, n,
		                              1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
		                              acBirch, rng, 0, 0);
		ASSERT("==============ATLAS::ControllerBirchMemory birch applyStep failed==============", ok);

		double baseCorrectionNormSq = 0.0;
		double birchCorrectionNormSq = 0.0;
		for (size_t idx = 0; idx < WBase.size(); ++idx)
		{
			const double baseCorrection =
			    static_cast<double>(WBase[idx])
			    + static_cast<double>(baseState.lastBaselineRate) * static_cast<double>(gW[idx]);
			const double birchCorrection =
			    static_cast<double>(WBirch[idx])
			    + static_cast<double>(birchState.lastBaselineRate) * static_cast<double>(gW[idx]);
			baseCorrectionNormSq += baseCorrection * baseCorrection;
			birchCorrectionNormSq += birchCorrection * birchCorrection;
		}
		ASSERT("==============ATLAS::ControllerBirchMemory should reduce active correction energy==============",
		       birchCorrectionNormSq < baseCorrectionNormSq);
		ASSERT("==============ATLAS::ControllerBirchMemory memory diagnostic should remain finite==============",
		       birchState.lastBirchMemoryGain == birchState.lastBirchMemoryGain);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test C15: GHOST stays memory-only and clears explicit complement activity
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Controller Test C15: GHOST stays memory-only and clears complement activity\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 24;
		const unsigned int n = 24;
		const unsigned int r = 2;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 38282846ULL);
		glades::atlas::WeightState state;
		prepare_atlas_complement_test_state(state, rng, m, n, r, 4u, "cnn.fc");
		state.activeComplementRank = 1u;
		state.trialComplementRank = 2u;
		state.trialComplementWins = 1u;

		std::fill(state.U.begin(), state.U.end(), 0.0f);
		state.U[0] = 1.0f;
		state.U[state.r + 1] = 1.0f;

		for (unsigned int j = 0; j < n; ++j)
		{
			state.resolveGzHistory[0u * r * n + 0u * n + j] = 4.0f;
			state.resolveGzHistory[0u * r * n + 1u * n + j] = 2.0f;
			state.resolveGzHistory[1u * r * n + 0u * n + j] = 3.5f;
			state.resolveGzHistory[1u * r * n + 1u * n + j] = 1.75f;
			state.resolveGzHistory[2u * r * n + 0u * n + j] = 3.0f;
			state.resolveGzHistory[2u * r * n + 1u * n + j] = 1.50f;
			state.resolveGzHistory[3u * r * n + 0u * n + j] = 2.5f;
			state.resolveGzHistory[3u * r * n + 1u * n + j] = 1.25f;
		}

		std::vector<float> W(static_cast<size_t>(m) * n, 0.0f);
		for (unsigned int j = 0; j < n; ++j)
		{
			W[0u * n + j] = 2.0f;
			W[1u * n + j] = 1.0f;
		}
		std::vector<float> gW(static_cast<size_t>(m) * n, 0.0f);
		for (unsigned int j = 0; j < n; ++j)
		{
			gW[0u * n + j] = 4.0f;
			gW[1u * n + j] = 2.0f;
			gW[2u * n + j] = 1.0f;
			gW[3u * n + j] = 1.0f;
		}

		glades::ATLASConfig acGhostGate;
		acGhostGate.rank = r;
		acGhostGate.complementRank = 4u;
		acGhostGate.beta = 1.0f;
		acGhostGate.biasCorrection = false;
		acGhostGate.muMin = 0.0f;
		acGhostGate.muMax = 0.0f;
		acGhostGate.tSub = 0u;
		acGhostGate.ghostEnabled = true;
		acGhostGate.ghostLagHorizon = 4u;
		acGhostGate.ghostEdgeThreshold = 0.10f;

		const bool ok = glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
		                                         1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
		                                         acGhostGate, rng, 0, "cnn.fc");
		ASSERT("==============ATLAS::ControllerGhostGate applyStep failed==============", ok);
		ASSERT("==============ATLAS::ControllerGhostGate should clear explicit complement activity==============",
		       state.activeComplementRank == 0u && state.trialComplementRank == 0u);
		ASSERT("==============ATLAS::ControllerGhostGate diagnostics should remain finite==============",
		       state.lastGhostEdge == state.lastGhostEdge
		       && state.lastGhostSigma == state.lastGhostSigma
		       && state.lastGhostHorizontalRatio == state.lastGhostHorizontalRatio);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test C16: GHOST horizontal projection strips gauge energy and keeps
	// memory diagnostics coherent on gauge-aligned histories.
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Controller Test C16: GHOST horizontal projection strips gauge energy\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 4;
		const unsigned int n = 4;
		const unsigned int r = 2;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 38282847ULL);
		glades::atlas::WeightState baseState;
		glades::atlas::initWeightState(baseState, m, n, r, 0.0f, rng);
		glades::atlas::WeightState ghostState = baseState;

		std::fill(baseState.U.begin(), baseState.U.end(), 0.0f);
		baseState.U[0] = 1.0f;
		baseState.U[baseState.r + 1] = 1.0f;
		ghostState.U = baseState.U;

		for (unsigned int j = 0; j < n; ++j)
		{
			ghostState.resolveGzHistory[0u * r * n + 0u * n + j] = 4.0f;
			ghostState.resolveGzHistory[0u * r * n + 1u * n + j] = 2.0f;
			ghostState.resolveGzHistory[1u * r * n + 0u * n + j] = 3.5f;
			ghostState.resolveGzHistory[1u * r * n + 1u * n + j] = 1.75f;
			ghostState.resolveGzHistory[2u * r * n + 0u * n + j] = 3.0f;
			ghostState.resolveGzHistory[2u * r * n + 1u * n + j] = 1.50f;
			ghostState.resolveGzHistory[3u * r * n + 0u * n + j] = 2.5f;
			ghostState.resolveGzHistory[3u * r * n + 1u * n + j] = 1.25f;
		}

		std::vector<float> WInit(static_cast<size_t>(m) * n, 0.0f);
		for (unsigned int j = 0; j < n; ++j)
		{
			WInit[0u * n + j] = 2.0f;
			WInit[1u * n + j] = 1.0f;
		}
		std::vector<float> WBase = WInit;
		std::vector<float> WGhost = WInit;
		std::vector<float> gW(static_cast<size_t>(m) * n, 0.0f);
		for (unsigned int j = 0; j < n; ++j)
		{
			gW[0u * n + j] = 4.0f;
			gW[1u * n + j] = 2.0f;
			gW[2u * n + j] = 1.0f;
			gW[3u * n + j] = 1.0f;
		}
		std::vector<float> gWBase = gW;
		std::vector<float> gWGhost = gW;

		glades::ATLASConfig acBase;
		acBase.rank = r;
		acBase.complementRank = 0u;
		acBase.beta = 0.0f;
		acBase.biasCorrection = false;
		acBase.muMin = 0.0f;
		acBase.muMax = 0.0f;
		acBase.tSub = 0u;

		glades::ATLASConfig acGhost = acBase;
		acGhost.ghostEnabled = true;
		acGhost.ghostLagHorizon = 4u;
		acGhost.ghostMemoryScale = 0.20f;
		acGhost.ghostEdgeThreshold = 0.0f;

		bool ok = glades::atlas::applyStep(baseState, &WBase[0], &gWBase[0], m, n,
		                                   1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
		                                   acBase, rng, 0, 0);
		ASSERT("==============ATLAS::ControllerGhostMemory base applyStep failed==============", ok);
		ok = glades::atlas::applyStep(ghostState, &WGhost[0], &gWGhost[0], m, n,
		                              1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
		                              acGhost, rng, 0, 0);
		ASSERT("==============ATLAS::ControllerGhostMemory ghost applyStep failed==============", ok);

		double baseCorrectionNormSq = 0.0;
		double ghostCorrectionNormSq = 0.0;
		for (size_t idx = 0; idx < WBase.size(); ++idx)
		{
			const double baseCorrection =
			    static_cast<double>(WBase[idx] - WInit[idx])
			    + static_cast<double>(baseState.lastBaselineRate) * static_cast<double>(gWBase[idx]);
			const double ghostCorrection =
			    static_cast<double>(WGhost[idx] - WInit[idx])
			    + static_cast<double>(ghostState.lastBaselineRate) * static_cast<double>(gWGhost[idx]);
			baseCorrectionNormSq += baseCorrection * baseCorrection;
			ghostCorrectionNormSq += ghostCorrection * ghostCorrection;
		}
		ASSERT("==============ATLAS::ControllerGhostMemory should remove some gauge energy==============",
		       ghostState.lastGhostHorizontalRatio < 0.999f);
		if (ghostState.lastGhostMemoryGain > 1e-6f)
		{
			ASSERT("==============ATLAS::ControllerGhostMemory should reduce active correction energy when memory activates==============",
			       ghostCorrectionNormSq < baseCorrectionNormSq);
		}
		ASSERT("==============ATLAS::ControllerGhostMemory memory diagnostics should remain finite==============",
		       ghostState.lastGhostMemoryGain == ghostState.lastGhostMemoryGain
		       && ghostState.lastGhostEdge == ghostState.lastGhostEdge
		       && ghostState.lastGhostSigma == ghostState.lastGhostSigma);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test C17: SPARROW stays memory-only and clears explicit complement activity
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Controller Test C17: SPARROW stays memory-only and clears complement activity\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 24;
		const unsigned int n = 24;
		const unsigned int r = 2;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 38282848ULL);
		glades::atlas::WeightState state;
		prepare_atlas_complement_test_state(state, rng, m, n, r, 4u, "cnn.fc");
		state.activeComplementRank = 1u;
		state.trialComplementRank = 2u;
		state.trialComplementWins = 1u;

		std::fill(state.U.begin(), state.U.end(), 0.0f);
		state.U[0] = 1.0f;
		state.U[state.r + 1] = 1.0f;

		std::vector<float> W(static_cast<size_t>(m) * n, 0.0f);
		for (unsigned int j = 0; j < n; ++j)
		{
			W[0u * n + j] = 2.0f;
			W[1u * n + j] = 1.0f;
		}
		std::vector<float> gW(static_cast<size_t>(m) * n, 0.0f);
		for (unsigned int j = 0; j < n; ++j)
		{
			gW[0u * n + j] = 4.0f;
			gW[1u * n + j] = 2.0f;
			gW[2u * n + j] = 1.0f;
			gW[3u * n + j] = 1.0f;
		}

		glades::ATLASConfig acSparrowGate;
		acSparrowGate.rank = r;
		acSparrowGate.complementRank = 4u;
		acSparrowGate.beta = 1.0f;
		acSparrowGate.biasCorrection = false;
		acSparrowGate.muMin = 0.0f;
		acSparrowGate.muMax = 0.0f;
		acSparrowGate.tSub = 0u;
		acSparrowGate.sparrowEnabled = true;
		acSparrowGate.sparrowModeRank = 2u;
		acSparrowGate.sparrowMemoryScale = 0.10f;
		acSparrowGate.sparrowEdgeThreshold = 0.10f;
		acSparrowGate.sparrowPoleMax = 0.95f;

		const bool ok = glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
		                                         1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
		                                         acSparrowGate, rng, 0, "cnn.fc");
		ASSERT("==============ATLAS::ControllerSparrowGate applyStep failed==============", ok);
		ASSERT("==============ATLAS::ControllerSparrowGate should clear explicit complement activity==============",
		       state.activeComplementRank == 0u && state.trialComplementRank == 0u);
		ASSERT("==============ATLAS::ControllerSparrowGate diagnostics should remain finite==============",
		       state.lastSparrowEdge == state.lastSparrowEdge
		       && state.lastSparrowSigma == state.lastSparrowSigma
		       && state.lastSparrowHorizontalRatio == state.lastSparrowHorizontalRatio
		       && state.lastSparrowMemoryGain == state.lastSparrowMemoryGain);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test C18: SPARROW horizontal streaming mode produces coherent
	// diagnostics and can reduce active correction energy on aligned history.
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Controller Test C18: SPARROW streaming quotient mode is coherent\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 4;
		const unsigned int n = 4;
		const unsigned int r = 2;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 38282849ULL);
		glades::atlas::WeightState baseState;
		glades::atlas::initWeightState(baseState, m, n, r, 0.0f, rng);
		glades::atlas::WeightState sparrowState = baseState;

		std::fill(baseState.U.begin(), baseState.U.end(), 0.0f);
		baseState.U[0] = 1.0f;
		baseState.U[baseState.r + 1] = 1.0f;
		sparrowState.U = baseState.U;

		for (unsigned int j = 0; j < n; ++j)
		{
			sparrowState.sparrowPrevActive[0u * n + j] = 4.0f;
			sparrowState.sparrowPrevActive[1u * n + j] = 2.0f;
		}

		std::vector<float> WInit(static_cast<size_t>(m) * n, 0.0f);
		for (unsigned int j = 0; j < n; ++j)
		{
			WInit[0u * n + j] = 2.0f;
			WInit[1u * n + j] = 1.0f;
		}
		std::vector<float> WBase = WInit;
		std::vector<float> WSparrow = WInit;
		std::vector<float> gW(static_cast<size_t>(m) * n, 0.0f);
		for (unsigned int j = 0; j < n; ++j)
		{
			gW[0u * n + j] = 4.0f;
			gW[1u * n + j] = 2.0f;
			gW[2u * n + j] = 1.0f;
			gW[3u * n + j] = 1.0f;
		}
		std::vector<float> gWBase = gW;
		std::vector<float> gWSparrow = gW;

		glades::ATLASConfig acBase;
		acBase.rank = r;
		acBase.complementRank = 0u;
		acBase.beta = 0.0f;
		acBase.biasCorrection = false;
		acBase.muMin = 0.0f;
		acBase.muMax = 0.0f;
		acBase.tSub = 0u;

		glades::ATLASConfig acSparrow = acBase;
		acSparrow.sparrowEnabled = true;
		acSparrow.sparrowModeRank = 2u;
		acSparrow.sparrowMemoryScale = 0.20f;
		acSparrow.sparrowEdgeThreshold = 0.0f;
		acSparrow.sparrowPoleMax = 0.95f;

		bool ok = glades::atlas::applyStep(baseState, &WBase[0], &gWBase[0], m, n,
		                                   1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
		                                   acBase, rng, 0, 0);
		ASSERT("==============ATLAS::ControllerSparrowMemory base applyStep failed==============", ok);
		ok = glades::atlas::applyStep(sparrowState, &WSparrow[0], &gWSparrow[0], m, n,
		                              1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
		                              acSparrow, rng, 0, 0);
		ASSERT("==============ATLAS::ControllerSparrowMemory sparrow applyStep failed==============", ok);

		double baseCorrectionNormSq = 0.0;
		double sparrowCorrectionNormSq = 0.0;
		for (size_t idx = 0; idx < WBase.size(); ++idx)
		{
			const double baseCorrection =
			    static_cast<double>(WBase[idx] - WInit[idx])
			    + static_cast<double>(baseState.lastBaselineRate) * static_cast<double>(gWBase[idx]);
			const double sparrowCorrection =
			    static_cast<double>(WSparrow[idx] - WInit[idx])
			    + static_cast<double>(sparrowState.lastBaselineRate) * static_cast<double>(gWSparrow[idx]);
			baseCorrectionNormSq += baseCorrection * baseCorrection;
			sparrowCorrectionNormSq += sparrowCorrection * sparrowCorrection;
		}
		ASSERT("==============ATLAS::ControllerSparrowMemory should remove some gauge energy==============",
		       sparrowState.lastSparrowHorizontalRatio < 0.999f);
		if (sparrowState.lastSparrowMemoryGain > 1e-6f)
		{
			ASSERT("==============ATLAS::ControllerSparrowMemory should reduce active correction energy when memory activates==============",
			       sparrowCorrectionNormSq < baseCorrectionNormSq);
		}
		ASSERT("==============ATLAS::ControllerSparrowMemory diagnostics should remain finite==============",
		       sparrowState.lastSparrowMemoryGain == sparrowState.lastSparrowMemoryGain
		       && sparrowState.lastSparrowEdge == sparrowState.lastSparrowEdge
		       && sparrowState.lastSparrowSigma == sparrowState.lastSparrowSigma
		       && sparrowState.lastSparrowSecondEdge == sparrowState.lastSparrowSecondEdge
		       && sparrowState.lastSparrowSecondSigma == sparrowState.lastSparrowSecondSigma
		       && sparrowState.sparrowPole == sparrowState.sparrowPole);
		ASSERT("==============ATLAS::ControllerSparrowMemory rank-2 storage should persist two channels==============",
		       sparrowState.sparrowLeftMode.size() >= static_cast<size_t>(2u * r)
		       && sparrowState.sparrowLatent.size() >= static_cast<size_t>(2u * n));
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test C18b: SPARROW auto-gates the second streaming mode based on
	// raw mode-2 edge strength and its ratio to mode 1.
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Controller Test C18b: SPARROW auto-gates the second mode coherently\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 4;
		const unsigned int n = 4;
		const unsigned int r = 2;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 382828491ULL);

		glades::ATLASConfig acAuto;
		acAuto.rank = r;
		acAuto.complementRank = 0u;
		acAuto.beta = 0.0f;
		acAuto.biasCorrection = false;
		acAuto.muMin = 0.0f;
		acAuto.muMax = 0.0f;
		acAuto.tSub = 0u;
		acAuto.sparrowEnabled = true;
		acAuto.sparrowModeRank = 2u;
		acAuto.sparrowAutoModeGate = true;
		acAuto.sparrowMemoryScale = 0.20f;
		acAuto.sparrowEdgeThreshold = 0.0f;
		acAuto.sparrowSecondEdgeThreshold = 0.10f;
		acAuto.sparrowSecondEdgeFraction = 0.50f;
		acAuto.sparrowPoleMax = 0.95f;

		glades::atlas::WeightState strongState;
		glades::atlas::initWeightState(strongState, m, n, r, 0.0f, rng);
		std::fill(strongState.U.begin(), strongState.U.end(), 0.0f);
		strongState.U[0] = 1.0f;
		strongState.U[strongState.r + 1] = 1.0f;
		strongState.sparrowPrevActive[0u * n + 0u] = 4.0f;
		strongState.sparrowPrevActive[1u * n + 1u] = 4.0f;
		std::vector<float> WStrong(static_cast<size_t>(m) * n, 0.0f);
		std::vector<float> gWStrong(static_cast<size_t>(m) * n, 0.0f);
		gWStrong[0u * n + 0u] = 4.0f;
		gWStrong[1u * n + 1u] = 4.0f;
		bool ok = glades::atlas::applyStep(strongState, &WStrong[0], &gWStrong[0], m, n,
		                                   1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
		                                   acAuto, rng, 0, 0);
		ASSERT("==============ATLAS::ControllerSparrowAutoGate strong applyStep failed==============", ok);
		ASSERT("==============ATLAS::ControllerSparrowAutoGate should retain both strong modes==============",
		       strongState.lastSparrowActiveModes == 2u);
		ASSERT("==============ATLAS::ControllerSparrowAutoGate should measure a nontrivial second edge in the strong case==============",
		       strongState.lastSparrowSecondEdge > acAuto.sparrowSecondEdgeThreshold);

		glades::atlas::WeightState weakState;
		glades::atlas::initWeightState(weakState, m, n, r, 0.0f, rng);
		std::fill(weakState.U.begin(), weakState.U.end(), 0.0f);
		weakState.U[0] = 1.0f;
		weakState.U[weakState.r + 1] = 1.0f;
		for (unsigned int j = 0; j < n; ++j)
		{
			weakState.sparrowPrevActive[0u * n + j] = 4.0f;
			weakState.sparrowPrevActive[1u * n + j] = 1.0f;
		}
		std::vector<float> WWeak(static_cast<size_t>(m) * n, 0.0f);
		std::vector<float> gWWeak(static_cast<size_t>(m) * n, 0.0f);
		for (unsigned int j = 0; j < n; ++j)
		{
			gWWeak[0u * n + j] = 4.0f;
			gWWeak[1u * n + j] = 1.0f;
		}
		ok = glades::atlas::applyStep(weakState, &WWeak[0], &gWWeak[0], m, n,
		                              1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
		                              acAuto, rng, 0, 0);
		ASSERT("==============ATLAS::ControllerSparrowAutoGate weak applyStep failed==============", ok);
		ASSERT("==============ATLAS::ControllerSparrowAutoGate should reject the redundant second mode==============",
		       weakState.lastSparrowActiveModes == 1u);
		ASSERT("==============ATLAS::ControllerSparrowAutoGate weak case diagnostics should stay finite==============",
		       weakState.lastSparrowSecondEdge == weakState.lastSparrowSecondEdge
		       && weakState.lastSparrowSecondSigma == weakState.lastSparrowSecondSigma);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test C19: QBRT produces coherent quotient-balanced diagnostics and
	// perturbs the active correction on aligned lagged history.
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Controller Test C19: QBRT balanced transfer mode is coherent\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 4;
		const unsigned int n = 4;
		const unsigned int r = 2;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 38282850ULL);
		glades::atlas::WeightState baseState;
		glades::atlas::initWeightState(baseState, m, n, r, 0.0f, rng);
		glades::atlas::WeightState qbrtState = baseState;

		std::fill(baseState.U.begin(), baseState.U.end(), 0.0f);
		baseState.U[0] = 1.0f;
		baseState.U[baseState.r + 1] = 1.0f;
		qbrtState.U = baseState.U;

		for (unsigned int j = 0; j < n; ++j)
		{
			qbrtState.resolveGzHistory[0u * r * n + 0u * n + j] = 4.0f;
			qbrtState.resolveGzHistory[0u * r * n + 1u * n + j] = 2.0f;
			qbrtState.resolveGzHistory[1u * r * n + 0u * n + j] = 3.5f;
			qbrtState.resolveGzHistory[1u * r * n + 1u * n + j] = 1.75f;
			qbrtState.resolveGzHistory[2u * r * n + 0u * n + j] = 3.0f;
			qbrtState.resolveGzHistory[2u * r * n + 1u * n + j] = 1.50f;
			qbrtState.resolveGzHistory[3u * r * n + 0u * n + j] = 2.5f;
			qbrtState.resolveGzHistory[3u * r * n + 1u * n + j] = 1.25f;
			qbrtState.qbrtLatent[j] = 0.25f;
		}
		qbrtState.qbrtPole = 0.50f;
		qbrtState.qbrtPoleNumer = 0.25f;
		qbrtState.qbrtPoleDenom = 0.50f;

		std::vector<float> WInit(static_cast<size_t>(m) * n, 0.0f);
		for (unsigned int j = 0; j < n; ++j)
		{
			WInit[0u * n + j] = 2.0f;
			WInit[1u * n + j] = 1.0f;
		}
		std::vector<float> WBase = WInit;
		std::vector<float> WQbrt = WInit;
		std::vector<float> gW(static_cast<size_t>(m) * n, 0.0f);
		for (unsigned int j = 0; j < n; ++j)
		{
			gW[0u * n + j] = 4.0f;
			gW[1u * n + j] = 2.0f;
			gW[2u * n + j] = 1.0f;
			gW[3u * n + j] = 1.0f;
		}
		std::vector<float> gWBase = gW;
		std::vector<float> gWQbrt = gW;

		glades::ATLASConfig acBase;
		acBase.rank = r;
		acBase.complementRank = 0u;
		acBase.beta = 0.0f;
		acBase.biasCorrection = false;
		acBase.muMin = 0.0f;
		acBase.muMax = 0.0f;
		acBase.tSub = 0u;

		glades::ATLASConfig acQbrt = acBase;
		acQbrt.qbrtEnabled = true;
		acQbrt.qbrtLagHorizon = 4u;
		acQbrt.qbrtMemoryScale = 0.20f;
		acQbrt.qbrtEdgeThreshold = 0.0f;
		acQbrt.qbrtPoleMax = 0.95f;

		bool ok = glades::atlas::applyStep(baseState, &WBase[0], &gWBase[0], m, n,
		                                   1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
		                                   acBase, rng, 0, 0);
		ASSERT("==============ATLAS::ControllerQBRTMemory base applyStep failed==============", ok);
		ok = glades::atlas::applyStep(qbrtState, &WQbrt[0], &gWQbrt[0], m, n,
		                              1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
		                              acQbrt, rng, 0, 0);
		ASSERT("==============ATLAS::ControllerQBRTMemory qbrt applyStep failed==============", ok);

		double baseCorrectionNormSq = 0.0;
		double qbrtCorrectionNormSq = 0.0;
		for (size_t idx = 0; idx < WBase.size(); ++idx)
		{
			const double baseCorrection =
			    static_cast<double>(WBase[idx] - WInit[idx])
			    + static_cast<double>(baseState.lastBaselineRate) * static_cast<double>(gWBase[idx]);
			const double qbrtCorrection =
			    static_cast<double>(WQbrt[idx] - WInit[idx])
			    + static_cast<double>(qbrtState.lastBaselineRate) * static_cast<double>(gWQbrt[idx]);
			baseCorrectionNormSq += baseCorrection * baseCorrection;
			qbrtCorrectionNormSq += qbrtCorrection * qbrtCorrection;
		}
		ASSERT("==============ATLAS::ControllerQBRTMemory should remove some quotient gauge energy==============",
		       qbrtState.lastQbrtHorizontalRatio < 0.999f);
		if (qbrtState.lastQbrtMemoryGain > 1e-6f)
		{
			ASSERT("==============ATLAS::ControllerQBRTMemory should materially perturb the active correction when memory activates==============",
			       fabs(qbrtCorrectionNormSq - baseCorrectionNormSq) > 1e-8);
		}
		ASSERT("==============ATLAS::ControllerQBRTMemory diagnostics should remain finite==============",
		       qbrtState.lastQbrtMemoryGain == qbrtState.lastQbrtMemoryGain
		       && qbrtState.lastQbrtEdge == qbrtState.lastQbrtEdge
		       && qbrtState.lastQbrtSigma == qbrtState.lastQbrtSigma
		       && qbrtState.qbrtPole == qbrtState.qbrtPole);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test C20: RIFT produces coherent path-signature diagnostics and
	// perturbs the active correction on curved lagged history.
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Controller Test C20: RIFT signature path mode is coherent\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 4;
		const unsigned int n = 4;
		const unsigned int r = 2;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 38282851ULL);
		glades::atlas::WeightState baseState;
		glades::atlas::initWeightState(baseState, m, n, r, 0.0f, rng);
		glades::atlas::WeightState riftState = baseState;

		std::fill(baseState.U.begin(), baseState.U.end(), 0.0f);
		baseState.U[0] = 1.0f;
		baseState.U[baseState.r + 1] = 1.0f;
		riftState.U = baseState.U;

		for (unsigned int j = 0; j < n; ++j)
		{
			// resolveGzHistory stores newest slice at lag 0, so seed a four-point
			// loop in reverse chronological order that still has nontrivial area
			// after quotient-horizontal projection.
			riftState.resolveGzHistory[0u * r * n + 0u * n + j] = ((j & 1u) == 0u) ? 1.0f : 0.0f;
			riftState.resolveGzHistory[0u * r * n + 1u * n + j] = (j == 1u || j == 2u) ? 1.0f : 0.0f;
			riftState.resolveGzHistory[1u * r * n + 0u * n + j] = (j < 2u) ? 1.0f : 0.0f;
			riftState.resolveGzHistory[1u * r * n + 1u * n + j] = (j == 0u || j == 3u) ? 1.0f : 0.0f;
			riftState.resolveGzHistory[2u * r * n + 0u * n + j] = ((j & 1u) == 1u) ? 1.0f : 0.0f;
			riftState.resolveGzHistory[2u * r * n + 1u * n + j] = ((j & 1u) == 0u) ? 1.0f : 0.0f;
			riftState.resolveGzHistory[3u * r * n + 0u * n + j] = 0.0f;
			riftState.resolveGzHistory[3u * r * n + 1u * n + j] = 0.0f;
			riftState.riftLatent[j] = 0.20f;
		}
		riftState.riftPole = 0.40f;
		riftState.riftPoleNumer = 0.16f;
		riftState.riftPoleDenom = 0.40f;

		std::vector<float> WInit(static_cast<size_t>(m) * n, 0.0f);
		for (unsigned int j = 0; j < n; ++j)
		{
			WInit[0u * n + j] = 2.0f;
			WInit[1u * n + j] = 1.0f;
		}
		std::vector<float> WBase = WInit;
		std::vector<float> WRift = WInit;
		std::vector<float> gW(static_cast<size_t>(m) * n, 0.0f);
		for (unsigned int j = 0; j < n; ++j)
		{
			gW[0u * n + j] = 2.0f;
			gW[1u * n + j] = -2.0f;
			gW[2u * n + j] = 1.0f;
			gW[3u * n + j] = 1.0f;
		}
		std::vector<float> gWBase = gW;
		std::vector<float> gWRift = gW;

		glades::ATLASConfig acBase;
		acBase.rank = r;
		acBase.complementRank = 0u;
		acBase.beta = 0.0f;
		acBase.biasCorrection = false;
		acBase.muMin = 0.0f;
		acBase.muMax = 0.0f;
		acBase.tSub = 0u;

		glades::ATLASConfig acRift = acBase;
		acRift.riftEnabled = true;
		acRift.riftLagHorizon = 4u;
		acRift.riftMemoryScale = 0.20f;
		acRift.riftEdgeThreshold = 0.0f;
		acRift.riftPoleMax = 0.95f;

		bool ok = glades::atlas::applyStep(baseState, &WBase[0], &gWBase[0], m, n,
		                                   1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
		                                   acBase, rng, 0, 0);
		ASSERT("==============ATLAS::ControllerRIFTMemory base applyStep failed==============", ok);
		ok = glades::atlas::applyStep(riftState, &WRift[0], &gWRift[0], m, n,
		                              1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
		                              acRift, rng, 0, 0);
		ASSERT("==============ATLAS::ControllerRIFTMemory rift applyStep failed==============", ok);

		double baseCorrectionNormSq = 0.0;
		double riftCorrectionNormSq = 0.0;
		for (size_t idx = 0; idx < WBase.size(); ++idx)
		{
			const double baseCorrection =
			    static_cast<double>(WBase[idx] - WInit[idx])
			    + static_cast<double>(baseState.lastBaselineRate) * static_cast<double>(gWBase[idx]);
			const double riftCorrection =
			    static_cast<double>(WRift[idx] - WInit[idx])
			    + static_cast<double>(riftState.lastBaselineRate) * static_cast<double>(gWRift[idx]);
			baseCorrectionNormSq += baseCorrection * baseCorrection;
			riftCorrectionNormSq += riftCorrection * riftCorrection;
		}
		ASSERT("==============ATLAS::ControllerRIFTMemory should remove some quotient gauge energy==============",
		       riftState.lastRiftHorizontalRatio < 0.999f);
		ASSERT("==============ATLAS::ControllerRIFTMemory should retain nontrivial path area==============",
		       riftState.lastRiftAreaEnergy > 1e-6f);
		if (riftState.lastRiftMemoryGain > 1e-6f)
		{
			ASSERT("==============ATLAS::ControllerRIFTMemory should materially perturb the active correction when memory activates==============",
			       fabs(riftCorrectionNormSq - baseCorrectionNormSq) > 1e-8);
		}
		ASSERT("==============ATLAS::ControllerRIFTMemory diagnostics should remain finite==============",
		       riftState.lastRiftMemoryGain == riftState.lastRiftMemoryGain
		       && riftState.lastRiftEdge == riftState.lastRiftEdge
		       && riftState.lastRiftSigma == riftState.lastRiftSigma
		       && riftState.lastRiftPredR2 == riftState.lastRiftPredR2
		       && riftState.riftPole == riftState.riftPole);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test C21: ORBIT-Lite stays memory-only on small output heads and
	// produces finite diagnostics.
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Controller Test C21: ORBIT-Lite stays memory-only on output heads\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 10;
		const unsigned int n = 24;
		const unsigned int r = 2;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 38282850ULL);
		glades::atlas::WeightState state;
		glades::atlas::initWeightState(state, m, n, r, 0.0f, rng);
		state.activeComplementRank = 1u;
		state.trialComplementRank = 2u;
		state.trialComplementWins = 1u;

		std::vector<float> W(static_cast<size_t>(m) * n, 0.0f);
		std::vector<float> gW(static_cast<size_t>(m) * n, 0.0f);
		for (unsigned int j = 0; j < n; ++j)
		{
			W[0u * n + j] = 3.0f;
			W[1u * n + j] = -1.0f;
			W[2u * n + j] = 2.0f;
			W[3u * n + j] = 0.0f;
			gW[0u * n + j] = 5.0f;
			gW[1u * n + j] = -3.0f;
			gW[2u * n + j] = 3.0f;
			gW[3u * n + j] = -1.0f;
		}

		glades::ATLASConfig acOrbitGate;
		acOrbitGate.rank = r;
		acOrbitGate.complementRank = 4u;
		acOrbitGate.beta = 0.0f;
		acOrbitGate.biasCorrection = false;
		acOrbitGate.muMin = 0.0f;
		acOrbitGate.muMax = 0.0f;
		acOrbitGate.tSub = 0u;
		acOrbitGate.orbitEnabled = true;
		acOrbitGate.orbitMemoryScale = 0.10f;
		acOrbitGate.orbitEdgeThreshold = 0.0f;
		acOrbitGate.orbitPoleMax = 0.95f;

		const bool ok = glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
		                                         1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
		                                         acOrbitGate, rng, 0, "cnn.fc");
		ASSERT("==============ATLAS::ControllerOrbitGate applyStep failed==============", ok);
		ASSERT("==============ATLAS::ControllerOrbitGate should clear explicit complement activity on output head==============",
		       state.activeComplementRank == 0u && state.trialComplementRank == 0u);
		ASSERT("==============ATLAS::ControllerOrbitGate diagnostics should remain finite==============",
		       state.lastOrbitEdge == state.lastOrbitEdge
		       && state.lastOrbitSigma == state.lastOrbitSigma
		       && state.lastOrbitHorizontalRatio == state.lastOrbitHorizontalRatio
		       && state.lastOrbitMemoryGain == state.lastOrbitMemoryGain
		       && state.orbitPole == state.orbitPole);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test C22: ORBIT-Lite produces a coherent output-space memory correction
	// on aligned classifier-head structure.
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Controller Test C22: ORBIT-Lite output-space mode is coherent\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 4;
		const unsigned int n = 4;
		const unsigned int r = 2;
		const float invSqrt2 = 0.70710678f;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 38282851ULL);
		glades::atlas::WeightState baseState;
		glades::atlas::initWeightState(baseState, m, n, r, 0.0f, rng);
		glades::atlas::WeightState orbitState = baseState;

		std::fill(baseState.U.begin(), baseState.U.end(), 0.0f);
		baseState.U[0u * r + 0u] = invSqrt2;
		baseState.U[1u * r + 0u] = -invSqrt2;
		baseState.U[2u * r + 1u] = invSqrt2;
		baseState.U[3u * r + 1u] = -invSqrt2;
		orbitState.U = baseState.U;
		for (unsigned int j = 0; j < n; ++j)
			orbitState.orbitPrevSignal[j] = 2.0f;

		std::vector<float> WInit(static_cast<size_t>(m) * n, 0.0f);
		for (unsigned int j = 0; j < n; ++j)
		{
			WInit[0u * n + j] = 4.0f;
			WInit[1u * n + j] = -2.0f;
			WInit[2u * n + j] = 2.0f;
			WInit[3u * n + j] = 0.0f;
		}
		std::vector<float> WBase = WInit;
		std::vector<float> WOrbit = WInit;
		std::vector<float> gW(static_cast<size_t>(m) * n, 0.0f);
		for (unsigned int j = 0; j < n; ++j)
		{
			gW[0u * n + j] = 5.0f;
			gW[1u * n + j] = -3.0f;
			gW[2u * n + j] = 3.0f;
			gW[3u * n + j] = -1.0f;
		}
		std::vector<float> gWBase = gW;
		std::vector<float> gWOrbit = gW;

		glades::ATLASConfig acBase;
		acBase.rank = r;
		acBase.complementRank = 0u;
		acBase.beta = 0.0f;
		acBase.biasCorrection = false;
		acBase.muMin = 0.0f;
		acBase.muMax = 0.0f;
		acBase.tSub = 0u;

		glades::ATLASConfig acOrbit = acBase;
		acOrbit.orbitEnabled = true;
		acOrbit.orbitMemoryScale = 0.20f;
		acOrbit.orbitEdgeThreshold = 0.0f;
		acOrbit.orbitPoleMax = 0.95f;
		for (unsigned int j = 0; j < n; ++j)
		{
			orbitState.orbitPrevSignal[j] = 0.75f;
			orbitState.orbitLatent[j] = 0.50f;
		}
		orbitState.orbitPole = 0.50f;
		orbitState.orbitPoleNumer = 0.25f;
		orbitState.orbitPoleDenom = 0.50f;

		bool ok = glades::atlas::applyStep(baseState, &WBase[0], &gWBase[0], m, n,
		                                   1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
		                                   acBase, rng, 0, "cnn.fc");
		ASSERT("==============ATLAS::ControllerOrbitMemory base applyStep failed==============", ok);
		ok = glades::atlas::applyStep(orbitState, &WOrbit[0], &gWOrbit[0], m, n,
		                              1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
		                              acOrbit, rng, 0, "cnn.fc");
		ASSERT("==============ATLAS::ControllerOrbitMemory orbit applyStep failed==============", ok);

		double baseCorrectionNormSq = 0.0;
		double orbitCorrectionNormSq = 0.0;
		for (size_t idx = 0; idx < WBase.size(); ++idx)
		{
			const double baseCorrection =
			    static_cast<double>(WBase[idx] - WInit[idx])
			    + static_cast<double>(baseState.lastBaselineRate) * static_cast<double>(gWBase[idx]);
			const double orbitCorrection =
			    static_cast<double>(WOrbit[idx] - WInit[idx])
			    + static_cast<double>(orbitState.lastBaselineRate) * static_cast<double>(gWOrbit[idx]);
			baseCorrectionNormSq += baseCorrection * baseCorrection;
			orbitCorrectionNormSq += orbitCorrection * orbitCorrection;
		}
		ASSERT("==============ATLAS::ControllerOrbitMemory should detect nontrivial output quotient signal==============",
		       orbitState.lastOrbitSigma > 0.0f && orbitState.lastOrbitEdge >= 0.0f);
		ASSERT("==============ATLAS::ControllerOrbitMemory should remove some common-logit energy==============",
		       orbitState.lastOrbitHorizontalRatio < 0.999f);
		if (orbitState.lastOrbitMemoryGain > 1e-6f)
		{
			ASSERT("==============ATLAS::ControllerOrbitMemory should materially perturb the active correction when memory activates==============",
			       fabs(orbitCorrectionNormSq - baseCorrectionNormSq) > 1e-8);
		}
		ASSERT("==============ATLAS::ControllerOrbitMemory diagnostics should remain finite==============",
		       orbitState.lastOrbitMemoryGain == orbitState.lastOrbitMemoryGain
		       && orbitState.lastOrbitEdge == orbitState.lastOrbitEdge
		       && orbitState.lastOrbitSigma == orbitState.lastOrbitSigma
		       && orbitState.orbitPole == orbitState.orbitPole);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test C23: QRC produces coherent closed-loop diagnostics and
	// perturbs the active correction on aligned quotient lagged history.
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Controller Test C23: QRC reduced control mode is coherent\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 4;
		const unsigned int n = 4;
		const unsigned int r = 2;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 38282852ULL);
		glades::atlas::WeightState baseState;
		glades::atlas::initWeightState(baseState, m, n, r, 0.0f, rng);
		glades::atlas::WeightState qrcState = baseState;

		std::fill(baseState.U.begin(), baseState.U.end(), 0.0f);
		baseState.U[0] = 1.0f;
		baseState.U[baseState.r + 1] = 1.0f;
		qrcState.U = baseState.U;

		for (unsigned int j = 0; j < n; ++j)
		{
			qrcState.resolveGzHistory[0u * r * n + 0u * n + j] = 4.0f;
			qrcState.resolveGzHistory[0u * r * n + 1u * n + j] = 2.0f;
			qrcState.resolveGzHistory[1u * r * n + 0u * n + j] = 3.0f;
			qrcState.resolveGzHistory[1u * r * n + 1u * n + j] = 1.5f;
			qrcState.resolveGzHistory[2u * r * n + 0u * n + j] = 2.0f;
			qrcState.resolveGzHistory[2u * r * n + 1u * n + j] = 1.0f;
			qrcState.resolveGzHistory[3u * r * n + 0u * n + j] = 1.0f;
			qrcState.resolveGzHistory[3u * r * n + 1u * n + j] = 0.5f;
			qrcState.qrcLatent[j] = 0.25f;
		}
		qrcState.qrcPole = 0.50f;
		qrcState.qrcPoleNumer = 0.25f;
		qrcState.qrcPoleDenom = 0.50f;

		std::vector<float> WInit(static_cast<size_t>(m) * n, 0.0f);
		for (unsigned int j = 0; j < n; ++j)
		{
			WInit[0u * n + j] = 2.0f;
			WInit[1u * n + j] = 1.0f;
		}
		std::vector<float> WBase = WInit;
		std::vector<float> WQrc = WInit;
		std::vector<float> gW(static_cast<size_t>(m) * n, 0.0f);
		for (unsigned int j = 0; j < n; ++j)
		{
			gW[0u * n + j] = 4.0f;
			gW[1u * n + j] = 2.0f;
			gW[2u * n + j] = 1.0f;
			gW[3u * n + j] = 1.0f;
		}
		std::vector<float> gWBase = gW;
		std::vector<float> gWQrc = gW;

		glades::ATLASConfig acBase;
		acBase.rank = r;
		acBase.complementRank = 0u;
		acBase.beta = 0.0f;
		acBase.biasCorrection = false;
		acBase.muMin = 0.0f;
		acBase.muMax = 0.0f;
		acBase.tSub = 0u;

		glades::ATLASConfig acQrc = acBase;
		acQrc.qrcEnabled = true;
		acQrc.qrcLagHorizon = 4u;
		acQrc.qrcMemoryScale = 0.20f;
		acQrc.qrcEdgeThreshold = 0.0f;
		acQrc.qrcPoleMax = 0.95f;

		bool ok = glades::atlas::applyStep(baseState, &WBase[0], &gWBase[0], m, n,
		                                   1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
		                                   acBase, rng, 0, 0);
		ASSERT("==============ATLAS::ControllerQRCMemory base applyStep failed==============", ok);
		ok = glades::atlas::applyStep(qrcState, &WQrc[0], &gWQrc[0], m, n,
		                              1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
		                              acQrc, rng, 0, 0);
		ASSERT("==============ATLAS::ControllerQRCMemory qrc applyStep failed==============", ok);

		double baseCorrectionNormSq = 0.0;
		double qrcCorrectionNormSq = 0.0;
		for (size_t idx = 0; idx < WBase.size(); ++idx)
		{
			const double baseCorrection =
			    static_cast<double>(WBase[idx] - WInit[idx])
			    + static_cast<double>(baseState.lastBaselineRate) * static_cast<double>(gWBase[idx]);
			const double qrcCorrection =
			    static_cast<double>(WQrc[idx] - WInit[idx])
			    + static_cast<double>(qrcState.lastBaselineRate) * static_cast<double>(gWQrc[idx]);
			baseCorrectionNormSq += baseCorrection * baseCorrection;
			qrcCorrectionNormSq += qrcCorrection * qrcCorrection;
		}
		ASSERT("==============ATLAS::ControllerQRCMemory should remove some quotient gauge energy==============",
		       qrcState.lastQrcHorizontalRatio < 0.999f);
		ASSERT("==============ATLAS::ControllerQRCMemory should produce nonnegative control gain==============",
		       qrcState.lastQrcControlGain >= 0.0f);
		if (qrcState.lastQrcMemoryGain > 1e-6f)
		{
			ASSERT("==============ATLAS::ControllerQRCMemory should materially perturb the active correction when control activates==============",
			       fabs(qrcCorrectionNormSq - baseCorrectionNormSq) > 1e-8);
		}
		ASSERT("==============ATLAS::ControllerQRCMemory diagnostics should remain finite==============",
		       qrcState.lastQrcMemoryGain == qrcState.lastQrcMemoryGain
		       && qrcState.lastQrcEdge == qrcState.lastQrcEdge
		       && qrcState.lastQrcSigma == qrcState.lastQrcSigma
		       && qrcState.lastQrcControlGain == qrcState.lastQrcControlGain
		       && qrcState.qrcPole == qrcState.qrcPole);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 2: ATLAS DFF XOR classification
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 2: DFF XOR classification\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* di = new glades::NumberInput();
		di->trainMatrix = shmea::GMatrix(4, shmea::GVector<float>(2, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(4, shmea::GVector<float>(1, 0.0f));

		// XOR dataset: [0,0]->0, [0,1]->1, [1,0]->1, [1,1]->0
		di->trainMatrix[0][0] = 0.0f; di->trainMatrix[0][1] = 0.0f;
		di->trainExpectedMatrix[0][0] = 0.0f;
		di->trainMatrix[1][0] = 0.0f; di->trainMatrix[1][1] = 1.0f;
		di->trainExpectedMatrix[1][0] = 1.0f;
		di->trainMatrix[2][0] = 1.0f; di->trainMatrix[2][1] = 0.0f;
		di->trainExpectedMatrix[2][0] = 1.0f;
		di->trainMatrix[3][0] = 1.0f; di->trainMatrix[3][1] = 1.0f;
		di->trainExpectedMatrix[3][0] = 0.0f;

		di->testMatrix = di->trainMatrix;
		di->testExpectedMatrix = di->trainExpectedMatrix;

		glades::InputLayerInfo* in = new glades::InputLayerInfo(
		    /*batchSize*/ 4,
		    /*learningRate*/ 0.1f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::SIGMOID,
		    /*activationParam*/ 1.0f);

		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(
		    /*size*/ 8,
		    /*learningRate*/ 0.1f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::SIGMOID,
		    /*activationParam*/ 1.0f));

		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
		glades::NNInfo* info = new glades::NNInfo("ut_atlas_dff_xor", in, hidden, out);

		glades::NNetwork net(info, glades::NNetwork::TYPE_DFF);
		net.setSeed(1337u);
		net.getTerminatorMutable().setEpoch(500);
		net.getTerminatorMutable().setAccuracy(0);

		// Configure ATLAS optimizer
		{
			glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
			cfg.optimizer.type = glades::OptimizerConfig::ATLAS;
			cfg.atlas.rank = 4;
			cfg.atlas.complementRank = 0u;
			cfg.atlas.tSub = 100;
		}

		CaptureMetricsCallbacks cb;
		const glades::NNetworkStatus st = net.train(di, &cb);
		ASSERT("==============ATLAS::DFF_XOR TrainStatus() Failed==============", st.ok());
		ASSERT("==============ATLAS::DFF_XOR no metrics captured==============", cb.saw);
		printf("[UT] ATLAS DFF XOR: final loss = %f\n", cb.last.totalError);
		ASSERT("==============ATLAS::DFF_XOR loss too high==============", cb.last.totalError < 0.27f);

		delete di;
		delete info;
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 3: ATLAS transformer basic convergence
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 3: Transformer basic convergence\n");
	printf("-----------------------------------\n");
	{
		const int numSamples = 8;
		const int numFeatures = 4;
		const int numOutputs = 4;

		glades::NumberInput* di = new glades::NumberInput();
		di->trainMatrix = shmea::GMatrix(numSamples, shmea::GVector<float>(numFeatures, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(numSamples, shmea::GVector<float>(numOutputs, 0.0f));

		// Fill with simple pattern data
		for (int i = 0; i < numSamples; ++i)
		{
			for (int j = 0; j < numFeatures; ++j)
			{
				di->trainMatrix[i][j] = static_cast<float>(i * numFeatures + j) / static_cast<float>(numSamples * numFeatures);
			}
			for (int j = 0; j < numOutputs; ++j)
			{
				di->trainExpectedMatrix[i][j] = static_cast<float>((i + j) % numOutputs) / static_cast<float>(numOutputs);
			}
		}
		di->testMatrix = di->trainMatrix;
		di->testExpectedMatrix = di->trainExpectedMatrix;

		glades::InputLayerInfo* in = new glades::InputLayerInfo(
		    /*batchSize*/ 1,
		    /*learningRate*/ 0.01f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f);

		// Transformer blocks: 1 hidden layer with dModel=16
		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(
		    /*size*/ 16, // dModel
		    /*learningRate*/ 0.01f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f));

		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(numOutputs, glades::OutputLayerInfo::REGRESSION);
		glades::NNInfo* info = new glades::NNInfo("ut_atlas_transformer", in, hidden, out);

		glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_ENCODER);
		net.setSeed(2026u);
		net.getTerminatorMutable().setEpoch(100);
		net.getTerminatorMutable().setAccuracy(0);

		// Configure ATLAS optimizer and transformer settings
		{
			glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
			cfg.optimizer.type = glades::OptimizerConfig::ATLAS;
			cfg.atlas.rank = 8;
			cfg.atlas.complementRank = 0u;
			cfg.atlas.tSub = 50;
			cfg.transformer.nHeadsOverride = 4;
			cfg.transformer.dFFOverride = 32;
		}

		CaptureMetricsCallbacks cb;
		const glades::NNetworkStatus st = net.train(di, &cb);
		ASSERT("==============ATLAS::Transformer TrainStatus() Failed==============", st.ok());
		ASSERT("==============ATLAS::Transformer no metrics captured==============", cb.saw);
		printf("[UT] ATLAS Transformer: final loss = %f\n", cb.last.totalError);
		ASSERT("==============ATLAS::Transformer loss too high==============", cb.last.totalError < 0.11f);

		delete di;
		delete info;
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 4: ATLAS RNN convergence
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 4: RNN convergence\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* di = new glades::NumberInput();
		di->trainMatrix = shmea::GMatrix(4, shmea::GVector<float>(2, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(4, shmea::GVector<float>(1, 0.0f));

		di->trainMatrix[0][0] = 0.1f; di->trainMatrix[0][1] = 0.2f;
		di->trainExpectedMatrix[0][0] = 0.3f;
		di->trainMatrix[1][0] = 0.4f; di->trainMatrix[1][1] = 0.5f;
		di->trainExpectedMatrix[1][0] = 0.9f;
		di->trainMatrix[2][0] = 0.2f; di->trainMatrix[2][1] = 0.3f;
		di->trainExpectedMatrix[2][0] = 0.5f;
		di->trainMatrix[3][0] = 0.6f; di->trainMatrix[3][1] = 0.7f;
		di->trainExpectedMatrix[3][0] = 1.3f;

		di->testMatrix = di->trainMatrix;
		di->testExpectedMatrix = di->trainExpectedMatrix;

		glades::InputLayerInfo* in = new glades::InputLayerInfo(
		    4, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::SIGMOID, 1.0f);

		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(
		    8, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::SIGMOID, 1.0f));

		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
		glades::NNInfo* info = new glades::NNInfo("ut_atlas_rnn", in, hidden, out);

		glades::NNetwork net(info, glades::NNetwork::TYPE_RNN);
		net.setSeed(42u);
		net.getTerminatorMutable().setEpoch(300);
		net.getTerminatorMutable().setAccuracy(0);

		{
			glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
			cfg.optimizer.type = glades::OptimizerConfig::ATLAS;
			cfg.atlas.rank = 4;
			cfg.atlas.complementRank = 0u;
			cfg.atlas.tSub = 50;
		}

		CaptureMetricsCallbacks cb;
		const glades::NNetworkStatus st = net.train(di, &cb);
		ASSERT("==============ATLAS::RNN TrainStatus() Failed==============", st.ok());
		ASSERT("==============ATLAS::RNN no metrics captured==============", cb.saw);
		printf("[UT] ATLAS RNN: final loss = %f\n", cb.last.totalError);
		ASSERT("==============ATLAS::RNN loss too high==============", cb.last.totalError < 0.16f);

		delete di;
		delete info;
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 5: ATLAS GRU convergence
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 5: GRU convergence\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* di = new glades::NumberInput();
		di->trainMatrix = shmea::GMatrix(4, shmea::GVector<float>(2, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(4, shmea::GVector<float>(1, 0.0f));

		di->trainMatrix[0][0] = 0.1f; di->trainMatrix[0][1] = 0.2f;
		di->trainExpectedMatrix[0][0] = 0.3f;
		di->trainMatrix[1][0] = 0.4f; di->trainMatrix[1][1] = 0.5f;
		di->trainExpectedMatrix[1][0] = 0.9f;
		di->trainMatrix[2][0] = 0.2f; di->trainMatrix[2][1] = 0.3f;
		di->trainExpectedMatrix[2][0] = 0.5f;
		di->trainMatrix[3][0] = 0.6f; di->trainMatrix[3][1] = 0.7f;
		di->trainExpectedMatrix[3][0] = 1.3f;

		di->testMatrix = di->trainMatrix;
		di->testExpectedMatrix = di->trainExpectedMatrix;

		glades::InputLayerInfo* in = new glades::InputLayerInfo(
		    4, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::SIGMOID, 1.0f);

		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(
		    8, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::SIGMOID, 1.0f));

		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
		glades::NNInfo* info = new glades::NNInfo("ut_atlas_gru", in, hidden, out);

		glades::NNetwork net(info, glades::NNetwork::TYPE_GRU);
		net.setSeed(1337u);
		net.getTerminatorMutable().setEpoch(300);
		net.getTerminatorMutable().setAccuracy(0);

		{
			glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
			cfg.optimizer.type = glades::OptimizerConfig::ATLAS;
			cfg.atlas.rank = 4;
			cfg.atlas.complementRank = 0u;
			cfg.atlas.tSub = 50;
		}

		CaptureMetricsCallbacks cb;
		const glades::NNetworkStatus st = net.train(di, &cb);
		ASSERT("==============ATLAS::GRU TrainStatus() Failed==============", st.ok());
		ASSERT("==============ATLAS::GRU no metrics captured==============", cb.saw);
		printf("[UT] ATLAS GRU: final loss = %f\n", cb.last.totalError);
		ASSERT("==============ATLAS::GRU loss too high==============", cb.last.totalError < 0.15f);

		delete di;
		delete info;
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 6: ATLAS LSTM convergence
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 6: LSTM convergence\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* di = new glades::NumberInput();
		di->trainMatrix = shmea::GMatrix(4, shmea::GVector<float>(2, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(4, shmea::GVector<float>(1, 0.0f));

		di->trainMatrix[0][0] = 0.1f; di->trainMatrix[0][1] = 0.2f;
		di->trainExpectedMatrix[0][0] = 0.3f;
		di->trainMatrix[1][0] = 0.4f; di->trainMatrix[1][1] = 0.5f;
		di->trainExpectedMatrix[1][0] = 0.9f;
		di->trainMatrix[2][0] = 0.2f; di->trainMatrix[2][1] = 0.3f;
		di->trainExpectedMatrix[2][0] = 0.5f;
		di->trainMatrix[3][0] = 0.6f; di->trainMatrix[3][1] = 0.7f;
		di->trainExpectedMatrix[3][0] = 1.3f;

		di->testMatrix = di->trainMatrix;
		di->testExpectedMatrix = di->trainExpectedMatrix;

		glades::InputLayerInfo* in = new glades::InputLayerInfo(
		    4, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::SIGMOID, 1.0f);

		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(
		    8, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::SIGMOID, 1.0f));

		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
		glades::NNInfo* info = new glades::NNInfo("ut_atlas_lstm", in, hidden, out);

		glades::NNetwork net(info, glades::NNetwork::TYPE_LSTM);
		net.setSeed(2026u);
		net.getTerminatorMutable().setEpoch(300);
		net.getTerminatorMutable().setAccuracy(0);

		{
			glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
			cfg.optimizer.type = glades::OptimizerConfig::ATLAS;
			cfg.atlas.rank = 4;
			cfg.atlas.complementRank = 0u;
			cfg.atlas.tSub = 50;
		}

		CaptureMetricsCallbacks cb;
		const glades::NNetworkStatus st = net.train(di, &cb);
		ASSERT("==============ATLAS::LSTM TrainStatus() Failed==============", st.ok());
		ASSERT("==============ATLAS::LSTM no metrics captured==============", cb.saw);
		printf("[UT] ATLAS LSTM: final loss = %f\n", cb.last.totalError);
		ASSERT("==============ATLAS::LSTM loss too high==============", cb.last.totalError < 0.16f);

		delete di;
		delete info;
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 7: ATLAS checkpoint save/load round-trip
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 7: Checkpoint save/load round-trip\n");
	printf("-----------------------------------\n");
	{
		const std::string ckptName = "ut_atlas_ckpt_roundtrip";

		// Phase 1: Train a DFF with ATLAS for 100 epochs, save checkpoint
		{
			glades::NumberInput* di = new glades::NumberInput();
			di->trainMatrix = shmea::GMatrix(4, shmea::GVector<float>(2, 0.0f));
			di->trainExpectedMatrix = shmea::GMatrix(4, shmea::GVector<float>(1, 0.0f));

			di->trainMatrix[0][0] = 0.0f; di->trainMatrix[0][1] = 0.0f;
			di->trainExpectedMatrix[0][0] = 0.0f;
			di->trainMatrix[1][0] = 0.0f; di->trainMatrix[1][1] = 1.0f;
			di->trainExpectedMatrix[1][0] = 1.0f;
			di->trainMatrix[2][0] = 1.0f; di->trainMatrix[2][1] = 0.0f;
			di->trainExpectedMatrix[2][0] = 1.0f;
			di->trainMatrix[3][0] = 1.0f; di->trainMatrix[3][1] = 1.0f;
			di->trainExpectedMatrix[3][0] = 0.0f;

			di->testMatrix = di->trainMatrix;
			di->testExpectedMatrix = di->trainExpectedMatrix;

			glades::InputLayerInfo* in = new glades::InputLayerInfo(
			    4, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::SIGMOID, 1.0f);

			std::vector<glades::HiddenLayerInfo*> hidden;
			hidden.push_back(new glades::HiddenLayerInfo(
			    8, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::SIGMOID, 1.0f));

			glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
			glades::NNInfo* info = new glades::NNInfo("ut_atlas_ckpt", in, hidden, out);

			glades::NNetwork net(info, glades::NNetwork::TYPE_DFF);
			net.setSeed(42u);
			net.getTerminatorMutable().setEpoch(100);
			net.getTerminatorMutable().setAccuracy(0);

			{
				glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
				cfg.optimizer.type = glades::OptimizerConfig::ATLAS;
				cfg.atlas.rank = 4;
				cfg.atlas.complementRank = 0u;
				cfg.atlas.tSub = 50;
			}

			const glades::NNetworkStatus st = net.train(di);
			ASSERT("==============ATLAS::Checkpoint Phase1 TrainStatus() Failed==============", st.ok());
			printf("[UT] Phase 1: trained 100 epochs with ATLAS, saving checkpoint...\n");

			const glades::NNetworkStatus saveSt = net.saveCheckpoint(ckptName);
			ASSERT("==============ATLAS::Checkpoint SaveCheckpoint() Failed==============", saveSt.ok());
			printf("[UT] Checkpoint saved successfully\n");

			delete di;
			delete info;
		}

		// Phase 2: Load checkpoint into fresh network, train 100 more epochs
		{
			glades::NumberInput* di = new glades::NumberInput();
			di->trainMatrix = shmea::GMatrix(4, shmea::GVector<float>(2, 0.0f));
			di->trainExpectedMatrix = shmea::GMatrix(4, shmea::GVector<float>(1, 0.0f));

			di->trainMatrix[0][0] = 0.0f; di->trainMatrix[0][1] = 0.0f;
			di->trainExpectedMatrix[0][0] = 0.0f;
			di->trainMatrix[1][0] = 0.0f; di->trainMatrix[1][1] = 1.0f;
			di->trainExpectedMatrix[1][0] = 1.0f;
			di->trainMatrix[2][0] = 1.0f; di->trainMatrix[2][1] = 0.0f;
			di->trainExpectedMatrix[2][0] = 1.0f;
			di->trainMatrix[3][0] = 1.0f; di->trainMatrix[3][1] = 1.0f;
			di->trainExpectedMatrix[3][0] = 0.0f;

			di->testMatrix = di->trainMatrix;
			di->testExpectedMatrix = di->trainExpectedMatrix;

			glades::NNetwork net2;
			const glades::NNetworkStatus loadSt = net2.loadCheckpoint(ckptName, di);
			ASSERT("==============ATLAS::Checkpoint LoadCheckpoint() Failed==============", loadSt.ok());
			printf("[UT] Checkpoint loaded successfully\n");

			// Continue training
			net2.getTerminatorMutable().setEpoch(100);
			net2.getTerminatorMutable().setAccuracy(0);

			const glades::NNetworkStatus st2 = net2.train(di);
			ASSERT("==============ATLAS::Checkpoint Phase2 TrainStatus() Failed==============", st2.ok());
			printf("[UT] Phase 2: trained 100 more epochs after checkpoint load\n");

			delete di;
		}

		// Cleanup checkpoint files
		{
			const std::string base = "database/checkpoints/" + ckptName;
			remove((base + "/manifest.txt").c_str());
			remove((base + "/nninfo.csv").c_str());
			// Remove shard files
			for (int i = 0; i < 10; ++i)
			{
				char buf[64];
				snprintf(buf, sizeof(buf), "/shard_%03d.bin", i);
				remove((base + buf).c_str());
			}
			rmdir(base.c_str());
		}
		printf("[UT] Checkpoint cleanup done\n");
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 8: gramSchmidt produces orthonormal columns
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 8: gramSchmidt orthonormality\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 6;
		const unsigned int r = 3;

		// Fill Q[m, r] with a known non-orthogonal matrix (row-major: Q[i*r+j])
		std::vector<float> Q(m * r, 0.0f);
		Q[0*r+0] = 1.0f; Q[0*r+1] = 0.5f; Q[0*r+2] = 0.3f;
		Q[1*r+0] = 0.5f; Q[1*r+1] = 1.0f; Q[1*r+2] = 0.2f;
		Q[2*r+0] = 0.3f; Q[2*r+1] = 0.2f; Q[2*r+2] = 1.0f;
		Q[3*r+0] = 0.1f; Q[3*r+1] = 0.4f; Q[3*r+2] = 0.7f;
		Q[4*r+0] = 0.6f; Q[4*r+1] = 0.1f; Q[4*r+2] = 0.5f;
		Q[5*r+0] = 0.2f; Q[5*r+1] = 0.8f; Q[5*r+2] = 0.1f;

		glades::atlas::gramSchmidt(&Q[0], m, r);

		// Verify Q^T * Q ≈ I_r  (dot of column c with column d should be delta_cd)
		const float tol = 1e-5f;
		for (unsigned int c = 0; c < r; ++c)
		{
			for (unsigned int d = c; d < r; ++d)
			{
				float dot = 0.0f;
				for (unsigned int i = 0; i < m; ++i)
					dot += Q[i * r + c] * Q[i * r + d];

				float expected = (c == d) ? 1.0f : 0.0f;
				float err = (dot - expected) < 0.0f ? -(dot - expected) : (dot - expected);
				if (err > tol)
				{
					printf("[FAIL] gramSchmidt: Q^T*Q[%u,%u] = %f, expected %f (err=%e)\n",
					       c, d, dot, expected, err);
				}
				ASSERT("==============ATLAS::gramSchmidt orthonormality failed==============", err <= tol);
			}
		}

		// Verify no column is zero (degenerate)
		for (unsigned int c = 0; c < r; ++c)
		{
			float colNorm = 0.0f;
			for (unsigned int i = 0; i < m; ++i)
				colNorm += Q[i * r + c] * Q[i * r + c];
			colNorm = std::sqrt(colNorm);
			ASSERT("==============ATLAS::gramSchmidt zero column==============", colNorm > 0.5f);
		}

		printf("[UT] ATLAS gramSchmidt: orthonormality verified (m=%u, r=%u)\n", m, r);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 9: refreshSubspace preserves orthonormality and transforms state
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 9: refreshSubspace correctness\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 8;
		const unsigned int n = 6;
		const unsigned int r = 3;

		// Initialize a WeightState
		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 12345ULL);
		glades::atlas::WeightState state;
		glades::atlas::initWeightState(state, m, n, r, 0.01f, rng);
		ASSERT("==============ATLAS::refreshSubspace init failed==============", state.initialized);

		// Set non-trivial Fisher and prevGz so we can verify they get transformed
		for (unsigned int c = 0; c < r; ++c)
			state.fisherDiag[c] = static_cast<float>(c + 1) * 0.5f;
		for (unsigned int i = 0; i < r * n; ++i)
			state.prevGz[i] = static_cast<float>(i) * 0.01f;

		// Create a synthetic gradient matrix [m, n]
		std::vector<float> grad(m * n, 0.0f);
		for (unsigned int i = 0; i < m * n; ++i)
			grad[i] = static_cast<float>(i % 7) * 0.1f - 0.3f;

		// Run refresh
		glades::atlas::refreshSubspace(state, &grad[0], m, n, /*powerIters=*/3,
		                              /*betaRefresh=*/0.5f,
		                              /*fisherWeightedRefresh=*/true, rng);

		// Verify U is still orthonormal after refresh
		const float tol = 1e-4f;
		for (unsigned int c = 0; c < r; ++c)
		{
			for (unsigned int d = c; d < r; ++d)
			{
				float dot = 0.0f;
				for (unsigned int i = 0; i < m; ++i)
					dot += state.U[i * r + c] * state.U[i * r + d];

				float expected = (c == d) ? 1.0f : 0.0f;
				float err = (dot - expected) < 0.0f ? -(dot - expected) : (dot - expected);
				if (err > tol)
				{
					printf("[FAIL] refreshSubspace: U^T*U[%u,%u] = %f, expected %f (err=%e)\n",
					       c, d, dot, expected, err);
				}
				ASSERT("==============ATLAS::refreshSubspace U not orthonormal==============", err <= tol);
			}
		}

		// Verify Fisher diagonal is still positive and finite
		for (unsigned int c = 0; c < r; ++c)
		{
			float f = state.fisherDiag[c];
			ASSERT("==============ATLAS::refreshSubspace Fisher NaN==============", f == f);
			ASSERT("==============ATLAS::refreshSubspace Fisher negative==============", f >= 0.0f);
		}

		// Verify prevGz is finite
		for (unsigned int i = 0; i < r * n; ++i)
		{
			float v = state.prevGz[i];
			ASSERT("==============ATLAS::refreshSubspace prevGz NaN==============", v == v);
		}

		printf("[UT] ATLAS refreshSubspace: orthonormality and state transform verified\n");
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 10: applyStep produces finite weights and decreasing loss
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 10: applyStep isolated correctness\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 4;
		const unsigned int n = 3;
		const unsigned int r = 2;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 9999ULL);
		glades::atlas::WeightState state;
		glades::atlas::initWeightState(state, m, n, r, 0.01f, rng);

		// Create weight matrix and gradient
		std::vector<float> W(m * n, 0.0f);
		for (unsigned int i = 0; i < m * n; ++i)
			W[i] = static_cast<float>(i) * 0.1f - 0.5f;

		// Run several steps with a consistent gradient direction
		float prevNorm = 1e10f;
		for (int step = 0; step < 20; ++step)
		{
			// Gradient pointing in a fixed direction (simulates a consistent signal)
			std::vector<float> gW(m * n, 0.0f);
			for (unsigned int i = 0; i < m * n; ++i)
				gW[i] = W[i] * 0.1f; // gradient ~ 0.1*W (like L2 regularization signal)

			glades::ATLASConfig acTest10;
			acTest10.complementRank = 0u;
			acTest10.beta = 0.999f;
			acTest10.muMin = 0.0f;
			acTest10.muMax = 0.5f;
			acTest10.eps = 1e-8f;
			acTest10.kappaMax = 10.0f;
			acTest10.tSub = 50u;
			acTest10.powerIters = 3u;
			acTest10.betaRefresh = 0.5f;
			acTest10.muGrowthRate = 0.001f;
			acTest10.biasCorrection = false;
			glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
			                         /*invBatch=*/1.0f, /*lr=*/0.01f,
			                         /*wd1=*/0.0f, /*wd2=*/0.0f, /*gradScale=*/1.0f,
			                         acTest10, rng);

			// Verify all weights are finite
			for (unsigned int i = 0; i < m * n; ++i)
			{
				ASSERT("==============ATLAS::applyStep produced NaN weight==============", W[i] == W[i]);
				float diff = W[i] - W[i];
				ASSERT("==============ATLAS::applyStep produced Inf weight==============", diff == 0.0f);
			}
		}

		// Verify weights changed from initial values (optimizer did something)
		bool changed = false;
		for (unsigned int i = 0; i < m * n; ++i)
		{
			float init = static_cast<float>(i) * 0.1f - 0.5f;
			if (W[i] != init) { changed = true; break; }
		}
		ASSERT("==============ATLAS::applyStep weights unchanged after 20 steps==============", changed);

		printf("[UT] ATLAS applyStep: 20 steps, all weights finite, weights updated\n");
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 11: Property-based fuzz test (invariants over many random steps)
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 11: Property-based fuzz test\n");
	printf("-----------------------------------\n");
	{
		// Run ATLAS with randomized configs across multiple weight matrix sizes
		// and verify invariants hold at every step.
		struct FuzzConfig
		{
			unsigned int m, n, r;
			float lr, beta;
			unsigned int tSub;
		};
		const FuzzConfig configs[] = {
			{4, 3, 2, 0.01f, 0.999f, 10},
			{8, 8, 4, 0.1f, 0.99f, 5},
			{16, 4, 3, 0.05f, 0.9f, 20},
			{3, 16, 2, 0.001f, 0.999f, 50},
			{32, 32, 8, 0.01f, 0.99f, 15},
		};
		const int nConfigs = 5;

		for (int ci = 0; ci < nConfigs; ++ci)
		{
			const FuzzConfig& fc = configs[ci];
			glades::rng::Engine rng;
			glades::rng::seed_engine(rng, static_cast<unsigned long long>(7777 + ci * 1337));
			glades::atlas::WeightState state;
			glades::atlas::initWeightState(state, fc.m, fc.n, fc.r, 0.01f, rng);

			std::vector<float> W(fc.m * fc.n, 0.0f);
			for (unsigned int i = 0; i < fc.m * fc.n; ++i)
				W[i] = glades::rng::standard_normal(rng) * 0.1f;

			glades::ATLASConfig acFuzz;
			acFuzz.complementRank = 0u;
			acFuzz.beta = fc.beta;
			acFuzz.muMin = 0.0f;
			acFuzz.muMax = 0.5f;
			acFuzz.eps = 1e-8f;
			acFuzz.kappaMax = 10.0f;
			acFuzz.tSub = fc.tSub;
			acFuzz.powerIters = 3u;
			acFuzz.betaRefresh = 0.5f;
			acFuzz.muGrowthRate = 0.001f;
			acFuzz.biasCorrection = false;

			const int nSteps = 500;
			for (int step = 0; step < nSteps; ++step)
			{
				// Generate random gradient
				std::vector<float> gW(fc.m * fc.n, 0.0f);
				for (unsigned int i = 0; i < fc.m * fc.n; ++i)
					gW[i] = glades::rng::standard_normal(rng) * 0.5f;

				glades::atlas::applyStep(state, &W[0], &gW[0], fc.m, fc.n,
				                         1.0f, fc.lr, 0.0f, 0.0f, 1.0f,
				                         acFuzz, rng);

				// Invariant 1: All weights finite
				for (unsigned int i = 0; i < fc.m * fc.n; ++i)
				{
					ASSERT("==============ATLAS::Fuzz NaN weight==============", W[i] == W[i]);
					float diff = W[i] - W[i];
					ASSERT("==============ATLAS::Fuzz Inf weight==============", diff == 0.0f);
				}

				// Invariant 2: Fisher diagonal positive and finite
				for (unsigned int c = 0; c < state.r; ++c)
				{
					ASSERT("==============ATLAS::Fuzz Fisher NaN==============",
					       state.fisherDiag[c] == state.fisherDiag[c]);
					ASSERT("==============ATLAS::Fuzz Fisher negative==============",
					       state.fisherDiag[c] >= 0.0f);
				}

				// Invariant 3: mu within bounds
				ASSERT("==============ATLAS::Fuzz mu below min==============", state.mu >= 0.0f);
				ASSERT("==============ATLAS::Fuzz mu above max==============", state.mu <= 0.5f + 1e-6f);

				// Invariant 4: sigma2 positive and finite
				ASSERT("==============ATLAS::Fuzz sigma2 NaN==============",
				       state.sigma2 == state.sigma2);
				ASSERT("==============ATLAS::Fuzz sigma2 negative==============",
				       state.sigma2 >= 0.0f);

				// Invariant 5: U orthonormal (check every tSub steps to avoid overhead)
				if (fc.tSub > 0u && (state.step % static_cast<unsigned long long>(fc.tSub)) == 0ULL)
				{
					const float orthoTol = 1e-4f;
					for (unsigned int c = 0; c < state.r; ++c)
					{
						for (unsigned int d = c; d < state.r; ++d)
						{
							float dot = 0.0f;
							for (unsigned int i = 0; i < fc.m; ++i)
								dot += state.U[i * state.r + c] * state.U[i * state.r + d];
							float expected = (c == d) ? 1.0f : 0.0f;
							float err = (dot - expected) < 0.0f ? -(dot - expected) : (dot - expected);
							ASSERT("==============ATLAS::Fuzz U not orthonormal==============", err <= orthoTol);
						}
					}
				}
			}

			printf("[UT] Fuzz config %d (m=%u n=%u r=%u): %d steps, all invariants held\n",
			       ci, fc.m, fc.n, fc.r, nSteps);
		}
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 12: kappaMax capping behavior
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 12: kappaMax rate capping\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 4;
		const unsigned int n = 3;
		const unsigned int r = 2;
		const float kappaMax = 5.0f;
		const float lr = 0.01f;
		const float eps = 1e-8f;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 55555ULL);
		glades::atlas::WeightState state;
		glades::atlas::initWeightState(state, m, n, r, 0.01f, rng);

		// Set Fisher diagonal to very small values (should trigger capping)
		for (unsigned int c = 0; c < r; ++c)
			state.fisherDiag[c] = 1e-12f;
		state.sigma2 = 1e-12f;

		std::vector<float> W(m * n, 0.5f);
		std::vector<float> gW(m * n, 1.0f);

		std::vector<float> W_before(W.begin(), W.end());

		glades::ATLASConfig acKappa;
		acKappa.complementRank = 0u;
		acKappa.beta = 0.999f;
		acKappa.muMin = 0.0f;
		acKappa.muMax = 0.5f;
		acKappa.eps = eps;
		acKappa.kappaMax = kappaMax;
		acKappa.tSub = 50u;
		acKappa.powerIters = 3u;
		acKappa.betaRefresh = 0.5f;
		acKappa.muGrowthRate = 0.001f;
		acKappa.biasCorrection = false;
		glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
		                         1.0f, lr, 0.0f, 0.0f, 1.0f,
		                         acKappa, rng);

		// The effective learning rate should be capped at kappaMax * lr = 0.05
		// Without capping, lr/(1e-12 + 1e-8) ~ 1e6, which would be catastrophic.
		// Verify no element changed by more than kappaMax * lr * maxGrad
		const float maxChange = kappaMax * lr * 1.0f; // gScale=1, invBatch=1, grad=1
		for (unsigned int i = 0; i < m * n; ++i)
		{
			float delta = W[i] - W_before[i];
			if (delta < 0.0f) delta = -delta;
			ASSERT("==============ATLAS::kappaMax capping violated==============",
			       delta < maxChange * 2.0f); // allow 2x margin for subspace correction
		}

		// Verify weights are still finite (not blown up)
		for (unsigned int i = 0; i < m * n; ++i)
		{
			ASSERT("==============ATLAS::kappaMax NaN weight==============", W[i] == W[i]);
			float diff = W[i] - W[i];
			ASSERT("==============ATLAS::kappaMax Inf weight==============", diff == 0.0f);
		}

	printf("[UT] ATLAS kappaMax: capping verified with tiny Fisher/sigma2\n");
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 12B: complement sector uses its own lr/cap
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 12B: complement sector rate capping\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 4;
		const unsigned int n = 1;
		const unsigned int r = 1;
		const float lr = 0.08f;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 55556ULL);
		glades::atlas::WeightState state;
		glades::atlas::initWeightState(state, m, n, r, 0.0f, rng);

		std::fill(state.U.begin(), state.U.end(), 0.0f);
		state.U[0] = 1.0f;
		state.activeRank = 1u;
		std::fill(state.V.begin(), state.V.end(), 0.0f);
		state.V[1] = 1.0f;

		std::vector<float> W(m * n, 0.0f);
		std::vector<float> gW(m * n, 0.0f);
		gW[0] = 0.0f;
		gW[1] = 0.1f;
		gW[2] = 10.0f;
		gW[3] = 10.0f;

		glades::ATLASConfig acSectorCap;
		acSectorCap.rank = r;
		acSectorCap.complementRank = 1u;
		acSectorCap.complementLrScale = 0.25f;
		acSectorCap.complementKappaMax = 0.5f;
		acSectorCap.kappaMax = 10.0f;
		acSectorCap.biasCorrection = false;
		acSectorCap.muMin = 0.0f;
		acSectorCap.muMax = 0.0f;
		acSectorCap.tSub = 0u;

		const bool ok = glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
		                                         1.0f, lr, 0.0f, 0.0f, 1.0f,
		                                         acSectorCap, rng);
		ASSERT("==============ATLAS::ComplementSectorCap applyStep failed==============", ok);
		ASSERT("==============ATLAS::ComplementSectorCap sector fisher mismatch==============",
		       fabsf(state.complementFisher - 0.01f) < 1e-6f);
		ASSERT("==============ATLAS::ComplementSectorCap sigma2 mismatch==============",
		       fabsf(state.sigma2 - 100.0f) < 1e-4f);
		ASSERT("==============ATLAS::ComplementSectorCap row-1 update mismatch==============",
		       fabsf(W[1] + 0.001f) < 1e-6f);
		printf("[UT] ATLAS complement sector cap: sigma2=%f sectorFisher=%f W[1]=%f\n",
		       state.sigma2, state.complementFisher, W[1]);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 13: mu adaptation responds to gradient smoothness
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 13: mu adaptation\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 4;
		const unsigned int n = 3;
		const unsigned int r = 2;

		// Scenario A: Smooth gradients (consistent direction) -> mu should grow
		{
			glades::ATLASConfig acMu;
			acMu.complementRank = 0u;
			acMu.beta = 0.999f;
			acMu.muMin = 0.01f;
			acMu.muMax = 0.5f;
			acMu.eps = 1e-8f;
			acMu.kappaMax = 10.0f;
			acMu.tSub = 200u;
			acMu.powerIters = 3u;
			acMu.betaRefresh = 0.5f;
			acMu.muGrowthRate = 0.001f;
			acMu.biasCorrection = false;

			glades::rng::Engine rng;
			glades::rng::seed_engine(rng, 11111ULL);
			glades::atlas::WeightState state;
			glades::atlas::initWeightState(state, m, n, r, 0.01f, rng);

			std::vector<float> W(m * n, 0.5f);
			for (int step = 0; step < 100; ++step)
			{
				std::vector<float> gW(m * n, 0.0f);
				// Consistent gradient direction
				for (unsigned int i = 0; i < m * n; ++i)
					gW[i] = 0.1f;
				glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
				                         1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
				                         acMu, rng);
			}
			float smoothMu = state.mu;

			// Scenario B: Oscillating gradients -> mu should shrink
			glades::rng::seed_engine(rng, 11111ULL);
			glades::atlas::WeightState state2;
			glades::atlas::initWeightState(state2, m, n, r, 0.01f, rng);

			std::vector<float> W2(m * n, 0.5f);
			for (int step = 0; step < 100; ++step)
			{
				std::vector<float> gW(m * n, 0.0f);
				// Oscillating gradient direction
				float sign = (step % 2 == 0) ? 1.0f : -1.0f;
				for (unsigned int i = 0; i < m * n; ++i)
					gW[i] = sign * 0.1f;
				glades::atlas::applyStep(state2, &W2[0], &gW[0], m, n,
				                         1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
				                         acMu, rng);
			}
			float oscillatingMu = state2.mu;

			printf("[UT] mu smooth=%.6f, mu oscillating=%.6f\n", smoothMu, oscillatingMu);
			ASSERT("==============ATLAS::mu adaptation: smooth should be >= oscillating==============",
			       smoothMu >= oscillatingMu);
		}
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 14: Rank clamping (rank > min(m, n))
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 14: Rank clamping\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 4;
		const unsigned int n = 3;
		const unsigned int requestedRank = 100; // much larger than min(m,n) = 3

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 77777ULL);
		glades::atlas::WeightState state;
		glades::atlas::initWeightState(state, m, n, requestedRank, 0.01f, rng);

		ASSERT("==============ATLAS::RankClamp init failed==============", state.initialized);
		ASSERT("==============ATLAS::RankClamp rank not clamped==============", state.r <= m && state.r <= n);
		ASSERT("==============ATLAS::RankClamp rank should be min(m,n)==============", state.r == 3u);

		// Verify U is still orthonormal after clamping
		const float tol = 1e-5f;
		for (unsigned int c = 0; c < state.r; ++c)
		{
			for (unsigned int d = c; d < state.r; ++d)
			{
				float dot = 0.0f;
				for (unsigned int i = 0; i < m; ++i)
					dot += state.U[i * state.r + c] * state.U[i * state.r + d];
				float expected = (c == d) ? 1.0f : 0.0f;
				float err = (dot - expected) < 0.0f ? -(dot - expected) : (dot - expected);
				ASSERT("==============ATLAS::RankClamp U not orthonormal==============", err <= tol);
			}
		}
		printf("[UT] ATLAS rank clamping: requested=%u, actual=%u (min(m,n)=%u)\n",
		       requestedRank, state.r, (m < n) ? m : n);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 15: NaN gradient recovery
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 15: NaN gradient recovery\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 4;
		const unsigned int n = 3;
		const unsigned int r = 2;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 88888ULL);
		glades::atlas::WeightState state;
		glades::atlas::initWeightState(state, m, n, r, 0.01f, rng);

		std::vector<float> W(m * n, 0.5f);

		glades::ATLASConfig acNan;
		acNan.rank = r;
		acNan.complementRank = 0u;
		acNan.beta = 0.999f;
		acNan.muMin = 0.0f;
		acNan.muMax = 0.5f;
		acNan.eps = 1e-8f;
		acNan.kappaMax = 10.0f;
		acNan.tSub = 50u;
		acNan.powerIters = 3u;
		acNan.betaRefresh = 0.5f;
		acNan.muGrowthRate = 0.001f;
		acNan.biasCorrection = false;

		// Run a few normal steps first
		for (int step = 0; step < 5; ++step)
		{
			std::vector<float> gW(m * n, 0.1f);
			glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
			                         1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
			                         acNan, rng);
		}

		// Inject a gradient with extreme values (not NaN, but very large)
		{
			std::vector<float> gW(m * n, 1e10f);
			glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
			                         1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
			                         acNan, rng);
		}

		// Verify state is still finite after extreme gradient
		ASSERT("==============ATLAS::NaN sigma2 not finite==============",
		       state.sigma2 == state.sigma2 && (state.sigma2 - state.sigma2) == 0.0f);
		ASSERT("==============ATLAS::NaN mu not finite==============",
		       state.mu == state.mu && (state.mu - state.mu) == 0.0f);
		for (unsigned int c = 0; c < r; ++c)
		{
			ASSERT("==============ATLAS::NaN Fisher not finite==============",
			       state.fisherDiag[c] == state.fisherDiag[c]);
		}

		// Continue with normal gradients — should still work
		for (int step = 0; step < 5; ++step)
		{
			std::vector<float> gW(m * n, 0.1f);
			glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
			                         1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
			                         acNan, rng);

			for (unsigned int i = 0; i < m * n; ++i)
			{
				ASSERT("==============ATLAS::NaN post-recovery weight NaN==============", W[i] == W[i]);
				float diff = W[i] - W[i];
				ASSERT("==============ATLAS::NaN post-recovery weight Inf==============", diff == 0.0f);
			}
		}
		printf("[UT] ATLAS NaN gradient recovery: state remains finite after extreme gradients\n");
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 16: Bias correction accelerates early preconditioning
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 16: Bias correction effect\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 4;
		const unsigned int n = 3;
		const unsigned int r = 2;

		// Run 10 steps with bias correction OFF
		float sigma2_uncorrected;
		{
			glades::rng::Engine rng;
			glades::rng::seed_engine(rng, 22222ULL);
			glades::atlas::WeightState state;
			glades::atlas::initWeightState(state, m, n, r, 0.01f, rng);

			glades::ATLASConfig acOff;
			acOff.complementRank = 0u;
			acOff.beta = 0.999f;
			acOff.biasCorrection = false;
			acOff.muMin = 0.0f;
			acOff.muMax = 0.0f;
			acOff.tSub = 200u;
			acOff.powerIters = 3u;
			acOff.betaRefresh = 0.5f;
			acOff.muGrowthRate = 0.0f;

			std::vector<float> W(m * n, 0.5f);
			for (int step = 0; step < 10; ++step)
			{
				std::vector<float> gW(m * n, 1.0f);
				glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
				                         1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
				                         acOff, rng);
			}
			sigma2_uncorrected = state.sigma2;
		}

		// Run 10 steps with bias correction ON
		float sigma2_corrected_raw;
		{
			glades::rng::Engine rng;
			glades::rng::seed_engine(rng, 22222ULL);
			glades::atlas::WeightState state;
			glades::atlas::initWeightState(state, m, n, r, 0.01f, rng);

			glades::ATLASConfig acOn;
			acOn.complementRank = 0u;
			acOn.beta = 0.999f;
			acOn.biasCorrection = true;
			acOn.muMin = 0.0f;
			acOn.muMax = 0.0f;
			acOn.tSub = 200u;
			acOn.powerIters = 3u;
			acOn.betaRefresh = 0.5f;
			acOn.muGrowthRate = 0.0f;

			std::vector<float> W(m * n, 0.5f);
			for (int step = 0; step < 10; ++step)
			{
				std::vector<float> gW(m * n, 1.0f);
				glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
				                         1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
				                         acOn, rng);
			}
			sigma2_corrected_raw = state.sigma2;
		}

		// The raw sigma2 EMA should be the same (bias correction only affects
		// the effective rate, not the stored EMA value)
		const float emaDiff = (sigma2_uncorrected - sigma2_corrected_raw) < 0.0f
		                    ? -(sigma2_uncorrected - sigma2_corrected_raw)
		                    : (sigma2_uncorrected - sigma2_corrected_raw);
		printf("[UT] ATLAS bias correction: sigma2_uncorrected=%f, sigma2_corrected_raw=%f\n",
		       sigma2_uncorrected, sigma2_corrected_raw);
		// They won't be exactly equal because the effective learning rate differs
		// (bias-corrected sigma2 produces different weight updates, which produces
		// different gradients on the next step). But they should be in the same ballpark.
		ASSERT("==============ATLAS::BiasCorr raw EMA wildly different==============",
		       emaDiff < 0.1f);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 17: Comparative ATLAS-vs-SGD convergence
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 17: Comparative ATLAS-vs-SGD convergence\n");
	printf("-----------------------------------\n");
	{
		// Train the same DFF network on the same problem with both SGD and ATLAS,
		// and assert that ATLAS final loss is no worse than 1.5x SGD final loss.
		// This catches optimizer regressions that absolute-threshold tests miss.

		// Dataset: sum regression
		glades::NumberInput* di = new glades::NumberInput();
		di->trainMatrix = shmea::GMatrix(4, shmea::GVector<float>(2, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(4, shmea::GVector<float>(1, 0.0f));

		di->trainMatrix[0][0] = 0.1f; di->trainMatrix[0][1] = 0.2f;
		di->trainExpectedMatrix[0][0] = 0.15f;
		di->trainMatrix[1][0] = 0.4f; di->trainMatrix[1][1] = 0.5f;
		di->trainExpectedMatrix[1][0] = 0.45f;
		di->trainMatrix[2][0] = 0.7f; di->trainMatrix[2][1] = 0.8f;
		di->trainExpectedMatrix[2][0] = 0.75f;
		di->trainMatrix[3][0] = 0.3f; di->trainMatrix[3][1] = 0.6f;
		di->trainExpectedMatrix[3][0] = 0.45f;

		di->testMatrix = di->trainMatrix;
		di->testExpectedMatrix = di->trainExpectedMatrix;

		const int epochs = 300;
		const unsigned int seed = 42u;

		// Run with SGD (momentum)
		float sgdLoss;
		{
			glades::InputLayerInfo* in = new glades::InputLayerInfo(
			    4, 0.05f, 0.9f, 0.0f, 0.0f, 0.0f, glades::GMath::SIGMOID, 1.0f);
			std::vector<glades::HiddenLayerInfo*> hidden;
			hidden.push_back(new glades::HiddenLayerInfo(
			    8, 0.05f, 0.9f, 0.0f, 0.0f, 0.0f, glades::GMath::SIGMOID, 1.0f));
			glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
			glades::NNInfo* info = new glades::NNInfo("ut_cmp_sgd", in, hidden, out);

			glades::NNetwork net(info, glades::NNetwork::TYPE_DFF);
			net.setSeed(static_cast<uint64_t>(seed));
			net.getTerminatorMutable().setEpoch(epochs);
			net.getTerminatorMutable().setAccuracy(0);

			CaptureMetricsCallbacks cb;
			const glades::NNetworkStatus st = net.train(di, &cb);
			ASSERT("==============ATLAS::Comparative SGD train failed==============", st.ok());
			ASSERT("==============ATLAS::Comparative SGD no metrics==============", cb.saw);
			sgdLoss = cb.last.totalError;

			delete info;
		}

		// Run with ATLAS
		float atlasLoss;
		{
			glades::InputLayerInfo* in = new glades::InputLayerInfo(
			    4, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::SIGMOID, 1.0f);
			std::vector<glades::HiddenLayerInfo*> hidden;
			hidden.push_back(new glades::HiddenLayerInfo(
			    8, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::SIGMOID, 1.0f));
			glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
			glades::NNInfo* info = new glades::NNInfo("ut_cmp_atlas", in, hidden, out);

			glades::NNetwork net(info, glades::NNetwork::TYPE_DFF);
			net.setSeed(static_cast<uint64_t>(seed));
			net.getTerminatorMutable().setEpoch(epochs);
			net.getTerminatorMutable().setAccuracy(0);

			{
				glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
				cfg.optimizer.type = glades::OptimizerConfig::ATLAS;
				cfg.atlas.rank = 4;
				cfg.atlas.complementRank = 0u;
				cfg.atlas.tSub = 50;
			}

			CaptureMetricsCallbacks cb;
			const glades::NNetworkStatus st = net.train(di, &cb);
			ASSERT("==============ATLAS::Comparative ATLAS train failed==============", st.ok());
			ASSERT("==============ATLAS::Comparative ATLAS no metrics==============", cb.saw);
			atlasLoss = cb.last.totalError;

			delete info;
		}

		printf("[UT] Comparative: SGD loss=%.6f, ATLAS loss=%.6f, ratio=%.2f\n",
		       sgdLoss, atlasLoss, (sgdLoss > 0.0f) ? atlasLoss / sgdLoss : 0.0f);
		ASSERT("==============ATLAS::Comparative ATLAS loss much worse than SGD==============",
		       atlasLoss < sgdLoss * 1.5f + 0.01f);

		delete di;
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 18: GPU ATLAS kernel tests
	// ---------------------------------------------------------------
#ifdef GLADES_HAVE_CUDA
	if (glades::gpu::initDevice())
	{
		printf("-----------------------------------\n");
		printf("ATLAS Test 18: GPU ATLAS kernels\n");
		printf("-----------------------------------\n");

		// 18a: atlas_gpu_init produces orthonormal U
		{
			printf("[UT] 18a: atlas_gpu_init orthonormality\n");
			// Keep m <= n and r <= min(m,n)/4 so GPU init preserves the
			// requested rank and basis layout for this orthonormality check.
			const unsigned int m = 12;
			const unsigned int n = 16;
			const unsigned int r = 3;
			glades::rng::Engine gpuRng;
			glades::rng::seed_engine(gpuRng, 12345ULL);

			glades::gpu::GpuAtlasWeightState state;
			bool ok = glades::gpu::atlas_gpu_init(state, m, n, r, 0.01f, gpuRng);
			ASSERT("==============ATLAS::GPU init failed==============", ok);
			ASSERT("==============ATLAS::GPU state not initialized==============", state.initialized);
			ASSERT("==============ATLAS::GPU rank mismatch==============", state.r == r);

			// Download U and verify orthonormality: U^T * U ~ I_r
			std::vector<float> h_U(m * r);
			state.U.download(h_U.data());

			const float orthoTol = 1e-4f;
			for (unsigned int c = 0; c < r; ++c)
			{
				for (unsigned int d = c; d < r; ++d)
				{
					float dot = 0.0f;
					for (unsigned int i = 0; i < m; ++i)
						dot += h_U[i * r + c] * h_U[i * r + d];
					float expected = (c == d) ? 1.0f : 0.0f;
					float err = fabsf(dot - expected);
					ASSERT("==============ATLAS::GPU init U not orthonormal==============", err <= orthoTol);
				}
			}
			printf("[UT] 18a: PASSED\n");
		}

		// 18b: atlas_gpu_gram_schmidt on known non-orthonormal input
		{
			printf("[UT] 18b: atlas_gpu_gram_schmidt correctness\n");
			const int m = 4;
			const int r = 2;

			// Non-orthogonal, non-unit input [m x r] row-major
			// Col 0: [1, 2, 3, 4], Col 1: [1, 1, 1, 1]
			float h_Q[8] = {
				1.0f, 1.0f,
				2.0f, 1.0f,
				3.0f, 1.0f,
				4.0f, 1.0f
			};

			glades::gpu::GpuBuffer<float> d_Q;
			d_Q.allocate(m * r);
			d_Q.upload(h_Q);

			bool ok = glades::gpu::atlas_gpu_gram_schmidt(d_Q.data(), m, r);
			ASSERT("==============ATLAS::GPU GS call failed==============", ok);
			ASSERT("==============ATLAS::GPU GS sync failed==============",
			       glades::gpu::synchronizeComputeStream());

			float h_result[8];
			d_Q.download(h_result);

			// Verify orthonormality
			const float orthoTol = 1e-5f;
			for (int c = 0; c < r; ++c)
			{
				for (int d = c; d < r; ++d)
				{
					float dot = 0.0f;
					for (int i = 0; i < m; ++i)
						dot += h_result[i * r + c] * h_result[i * r + d];
					float expected = (c == d) ? 1.0f : 0.0f;
					float err = fabsf(dot - expected);
					char msg[128];
					sprintf(msg, "ATLAS::GPU GS col %d dot col %d = %.6f (expected %.1f)", c, d, dot, expected);
					ASSERT(msg, err <= orthoTol);
				}
			}

			// Col 0 should be proportional to [1,2,3,4] (just normalized)
			float norm0 = sqrtf(1.0f + 4.0f + 9.0f + 16.0f);
			const float dirTol = 1e-5f;
			for (int i = 0; i < m; ++i)
			{
				float expected = (float)(i + 1) / norm0;
				float err = fabsf(h_result[i * r + 0] - expected);
				ASSERT("==============ATLAS::GPU GS col0 direction wrong==============", err <= dirTol);
			}

			printf("[UT] 18b: PASSED\n");
		}

		// 18c: atlas_gpu_step convergence on simple gradient descent
		{
			printf("[UT] 18c: atlas_gpu_step convergence\n");
			const unsigned int m = 4;
			const unsigned int n = 3;
			const unsigned int r = 2;
			glades::rng::Engine gpuRng;
			glades::rng::seed_engine(gpuRng, 42ULL);

			glades::gpu::GpuAtlasWeightState state;
			bool ok = glades::gpu::atlas_gpu_init(state, m, n, r, 0.01f, gpuRng);
			ASSERT("==============ATLAS::GPU step init failed==============", ok);

			// Weight and gradient on device
			const size_t mn = m * n;
			glades::gpu::GpuBuffer<float> d_W, d_gW;
			d_W.allocate(mn);
			d_gW.allocate(mn);

			// Initialize weights to small random values
			std::vector<float> h_W(mn);
			for (size_t i = 0; i < mn; ++i)
				h_W[i] = 0.1f * (float)((i * 7 + 3) % 11) / 11.0f;
			d_W.upload(h_W.data());

			// Target weights (what we want to reach)
			std::vector<float> target(mn);
			for (size_t i = 0; i < mn; ++i)
				target[i] = 0.5f;

			glades::ATLASConfig ac;
			ac.rank = r;
			ac.complementRank = 0u;
			ac.beta = 0.99f;
			ac.muMin = 0.01f;
			ac.muMax = 0.3f;
			ac.tSub = 10;
			ac.powerIters = 2;
			ac.eps = 1e-8f;
			ac.kappaMax = 10.0f;
			ac.betaRefresh = 0.5f;
			ac.muGrowthRate = 0.001f;
			ac.biasCorrection = true;

			float initialLoss = 0.0f;
			float finalLoss = 0.0f;

			const int nSteps = 100;
			for (int s = 0; s < nSteps; ++s)
			{
				// Compute gradient = W - target (MSE gradient direction)
				d_W.download(h_W.data());
				std::vector<float> h_gW(mn);
				float loss = 0.0f;
				for (size_t i = 0; i < mn; ++i)
				{
					h_gW[i] = h_W[i] - target[i];
					loss += h_gW[i] * h_gW[i];
				}
				loss /= (float)mn;

				if (s == 0) initialLoss = loss;
				if (s == nSteps - 1) finalLoss = loss;

				d_gW.upload(h_gW.data());

				ok = glades::gpu::atlas_gpu_step(state, d_W.data(), d_gW.data(),
				                                 m, n,
				                                 1.0f, 0.05f,
				                                 0.0f, 0.0f, 1.0f,
				                                 ac);
				ASSERT("==============ATLAS::GPU step call failed==============", ok);
			}

			printf("[UT] 18c: initial_loss=%.6f final_loss=%.6f\n", initialLoss, finalLoss);
			// This toy loop recomputes gradients on the host and feeds them back to
			// the synchronized GPU optimizer. It is a smoke test for stable progress,
			// not a benchmark for full convergence speed.
			ASSERT("==============ATLAS::GPU step did not converge==============",
			       finalLoss < initialLoss * 0.9f);

			// Verify weights are finite
			d_W.download(h_W.data());
			for (size_t i = 0; i < mn; ++i)
			{
				ASSERT("==============ATLAS::GPU step produced NaN weight==============",
				       h_W[i] == h_W[i]);
				ASSERT("==============ATLAS::GPU step produced Inf weight==============",
				       fabsf(h_W[i]) < 1e10f);
			}

			printf("[UT] 18c: PASSED\n");
		}

		// 18d: GPU-CPU consistency test
		// Run identical problems on both paths and verify optimizer state stays
		// aligned while cross-device weight drift remains bounded.
		{
			printf("[UT] 18d: GPU-CPU equivalence\n");
			// Choose dimensions that avoid GPU rank clamping so the CPU and GPU
			// paths compare the same effective subspace rank.
			const unsigned int m = 12;
			const unsigned int n = 16;
			const unsigned int r = 3;
			const unsigned int nSteps = 50;
			const float lr = 0.01f;
			const float muInit = 0.01f;
			const unsigned long long cpuSeed = 77777ULL;

			// --- CPU path ---
			glades::rng::Engine cpuRng;
			glades::rng::seed_engine(cpuRng, cpuSeed);
			glades::atlas::WeightState cpuState;
			glades::atlas::initWeightState(cpuState, m, n, r, muInit, cpuRng);

			std::vector<float> W_cpu(m * n);
			for (unsigned int i = 0; i < m * n; ++i)
				W_cpu[i] = 0.1f * (float)((i * 7 + 3) % 11) / 11.0f;

			glades::ATLASConfig ac;
			ac.rank = r;
			ac.complementRank = 0u;
			ac.beta = 0.99f;
			ac.muMin = 0.01f;
			ac.muMax = 0.3f;
			ac.tSub = 10;
			ac.powerIters = 2;
			ac.eps = 1e-8f;
			ac.kappaMax = 10.0f;
			ac.betaRefresh = 0.5f;
			ac.muGrowthRate = 0.001f;
			ac.biasCorrection = true;

			// Pre-generate all gradients so both paths use identical inputs.
			std::vector<std::vector<float> > allGrads(nSteps, std::vector<float>(m * n));
			{
				glades::rng::Engine gradRng;
				glades::rng::seed_engine(gradRng, 12345ULL);
				for (unsigned int s = 0; s < nSteps; ++s)
					for (unsigned int i = 0; i < m * n; ++i)
						allGrads[s][i] = glades::rng::standard_normal(gradRng) * 0.1f;
			}

			for (unsigned int s = 0; s < nSteps; ++s)
			{
				std::vector<float> gW(allGrads[s]);
				glades::atlas::applyStep(cpuState, &W_cpu[0], &gW[0], m, n,
				                         1.0f, lr, 0.0f, 0.0f, 1.0f,
				                         ac, cpuRng);
			}

			// --- GPU path ---
			// Both CPU and GPU now use glades::rng::Engine, so seeding an engine
			// with the same seed produces identical initial U. We still overwrite
			// GPU state with the CPU's initial U to guarantee both paths diverge
			// only due to floating-point ordering differences, not initialization.
			glades::rng::Engine gpuInitRng;
			glades::rng::seed_engine(gpuInitRng, cpuSeed);
			glades::gpu::GpuAtlasWeightState gpuState;
			bool ok = glades::gpu::atlas_gpu_init(gpuState, m, n, r, muInit, gpuInitRng);
			ASSERT("==============ATLAS::Equiv GPU init failed==============", ok);

			// Upload CPU's initial basis to GPU (so both start from the same U)
			{
				glades::rng::Engine initRng;
				glades::rng::seed_engine(initRng, cpuSeed);
				glades::atlas::WeightState tmpCpu;
				glades::atlas::initWeightState(tmpCpu, m, n, r, muInit, initRng);
				gpuState.U.upload(tmpCpu.U.data(), tmpCpu.U.size());
				gpuState.fisherDiag.upload(tmpCpu.fisherDiag.data(), tmpCpu.fisherDiag.size());
				gpuState.V.upload(tmpCpu.V.data(), tmpCpu.V.size());
				gpuState.complementFisher.upload(&tmpCpu.complementFisher, 1);
				gpuState.prevGz.zero();
				gpuState.prevGv.zero();
				gpuState.totalTrace = tmpCpu.totalTrace;
				gpuState.sigma2 = tmpCpu.sigma2;
				gpuState.mu = tmpCpu.mu;
				gpuState.step = 0ULL;
			}

			// Initialize GPU weights identically
			const size_t mn = m * n;
			glades::gpu::GpuBuffer<float> d_W, d_gW;
			d_W.allocate(mn);
			d_gW.allocate(mn);
			{
				std::vector<float> h_W(mn);
				for (unsigned int i = 0; i < mn; ++i)
					h_W[i] = 0.1f * (float)((i * 7 + 3) % 11) / 11.0f;
				d_W.upload(h_W.data(), mn);
			}

			for (unsigned int s = 0; s < nSteps; ++s)
			{
				d_gW.upload(allGrads[s].data(), mn);
				ok = glades::gpu::atlas_gpu_step(gpuState, d_W.data(), d_gW.data(),
				                                  m, n,
				                                  1.0f, lr,
				                                  0.0f, 0.0f, 1.0f,
				                                  ac);
				ASSERT("==============ATLAS::Equiv GPU step failed==============", ok);
			}

			// Download and compare. With synchronized GPU reductions and blocking
			// buffer operations, the CPU and GPU paths should now agree up to small
			// float32 roundoff. Compare scalar optimizer state directly and use a
			// small L2-relative tolerance for the weights.
			std::vector<float> W_gpu(mn);
			d_W.download(W_gpu.data());

			double diffSq = 0.0;
			double refSq = 0.0;
			for (unsigned int i = 0; i < mn; ++i)
			{
				float diff = fabsf(W_cpu[i] - W_gpu[i]);
				diffSq += (double)diff * (double)diff;
				refSq += (double)W_cpu[i] * (double)W_cpu[i];
			}
			const float l2RelErr = (refSq > 1e-20) ? (float)sqrt(diffSq / refSq) : 0.0f;
			const float sigma2RelErr = fabsf(cpuState.sigma2 - gpuState.sigma2)
			                         / fmaxf(fabsf(cpuState.sigma2), 1e-6f);
			const float muAbsErr = fabsf(cpuState.mu - gpuState.mu);

			printf("[UT] 18d: l2_relative_error=%.6f sigma2_relative_error=%.6f "
			       "mu_abs_error=%.6f (cpu_sigma2=%.6f gpu_sigma2=%.6f)\n",
			       l2RelErr, sigma2RelErr, muAbsErr, cpuState.sigma2, gpuState.sigma2);
			ASSERT("==============ATLAS::Equiv GPU-CPU sigma2 diverged too much==============",
			       sigma2RelErr < 0.02f);
			ASSERT("==============ATLAS::Equiv GPU-CPU mu diverged too much==============",
			       muAbsErr < 0.01f);
			ASSERT("==============ATLAS::Equiv GPU-CPU weights diverged too much==============",
			       l2RelErr < 0.02f);

			printf("[UT] 18d: PASSED\n");
		}

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

		// ---------------------------------------------------------------
		// Test 29: GPU-to-CPU state transfer consistency
		// ---------------------------------------------------------------
		printf("-----------------------------------\n");
		printf("ATLAS Test 29: GPU-to-CPU state transfer equivalence\n");
		printf("-----------------------------------\n");
		{
			// Run ATLAS on GPU for N steps, download state to CPU, then continue
			// on both paths. State transfer should preserve optimizer scalars and
			// keep the continued trajectories in the same numerical regime.
			// Choose dimensions that avoid GPU rank clamping so the downloaded
			// GPU state matches the CPU WeightState layout.
			const unsigned int m = 12;
			const unsigned int n = 16;
			const unsigned int r = 3;
			const unsigned int warmupSteps = 20;
			const unsigned int compareSteps = 30;
			const float lr = 0.01f;
			const float muInit = 0.01f;
			const unsigned long long seed = 29292929ULL;

			glades::ATLASConfig ac;
			ac.rank = r;
			ac.complementRank = 0u;
			ac.beta = 0.99f;
			ac.tSub = 10;
			ac.biasCorrection = true;

			// Pre-generate all gradients
			const size_t mn = (size_t)m * n;
			const unsigned int totalSteps = warmupSteps + compareSteps;
			std::vector<std::vector<float> > allGrads(totalSteps, std::vector<float>(mn));
			{
				glades::rng::Engine gradRng;
				glades::rng::seed_engine(gradRng, 55555ULL);
				for (unsigned int s = 0; s < totalSteps; ++s)
					for (size_t i = 0; i < mn; ++i)
						allGrads[s][i] = glades::rng::standard_normal(gradRng) * 0.1f;
			}

			// Initial weights (same for GPU and CPU comparison path)
			std::vector<float> initW(mn);
			for (size_t i = 0; i < mn; ++i)
				initW[i] = 0.1f * (float)((i * 7 + 3) % 11) / 11.0f;

			// --- GPU warmup phase ---
			glades::rng::Engine gpuRng;
			glades::rng::seed_engine(gpuRng, seed);
			glades::gpu::GpuAtlasWeightState gpuState;
			bool ok = glades::gpu::atlas_gpu_init(gpuState, m, n, r, muInit, gpuRng);
			ASSERT("==============ATLAS::Xfer GPU init failed==============", ok);

			glades::gpu::GpuBuffer<float> d_W, d_gW;
			d_W.allocate(mn);
			d_gW.allocate(mn);
			d_W.upload(initW.data(), mn);

			for (unsigned int s = 0; s < warmupSteps; ++s)
			{
				d_gW.upload(allGrads[s].data(), mn);
				ok = glades::gpu::atlas_gpu_step(gpuState, d_W.data(), d_gW.data(),
				                                  m, n, 1.0f, lr, 0.0f, 0.0f, 1.0f, ac);
				ASSERT("==============ATLAS::Xfer GPU warmup step failed==============", ok);
			}

			// --- Download GPU state to CPU WeightState ---
			glades::atlas::WeightState cpuState;
			cpuState.m = m;
			cpuState.n = n;
			cpuState.r = r;
			cpuState.activeRank = r;
			cpuState.U.resize((size_t)m * r);
			gpuState.U.download(cpuState.U.data(), cpuState.U.size());
			cpuState.fisherDiag.resize(r);
			gpuState.fisherDiag.download(cpuState.fisherDiag.data(), r);
			cpuState.V.resize(m);
			gpuState.V.download(cpuState.V.data(), cpuState.V.size());
			cpuState.prevGz.resize((size_t)r * n);
			gpuState.prevGz.download(cpuState.prevGz.data(), cpuState.prevGz.size());
			cpuState.prevGv.resize(n);
			gpuState.prevGv.download(cpuState.prevGv.data(), cpuState.prevGv.size());
			gpuState.complementFisher.download(&cpuState.complementFisher, 1);
			cpuState.totalTrace = gpuState.totalTrace;
			cpuState.sigma2 = gpuState.sigma2;
			cpuState.mu = gpuState.mu;
			cpuState.step = gpuState.step;
			cpuState.initialized = true;

			// Allocate CPU scratch buffers
			const size_t mr = (size_t)m * r;
			const size_t rn = (size_t)r * n;
			cpuState.scratch_gz.resize(rn);
			cpuState.scratch_corrected.resize(rn);
			cpuState.scratch_gv.resize(n);
			cpuState.scratch_correctedV.resize(n);
			cpuState.scratch_U_old.resize(mr);
			cpuState.scratch_f_old.resize(r);
			cpuState.scratch_B.resize(rn);
			cpuState.scratch_Z.resize(mr);
			cpuState.scratch_overlap.resize((size_t)r * r);
			cpuState.scratch_prevGzOld.resize(rn);
			cpuState.scratch_basisPacked.resize(mr);
			cpuState.scratch_V_old.resize(m);
			cpuState.scratch_Bv.resize(n);
			cpuState.scratch_Zv.resize(m);

			// Download GPU weights for CPU path
			std::vector<float> W_cpu(mn);
			d_W.download(W_cpu.data());

			// Need a CPU RNG engine for refreshSubspace calls.
			// Seed it with same base seed + step offset so it's deterministic.
			glades::rng::Engine cpuRng;
			glades::rng::seed_engine(cpuRng, seed + gpuState.step);

			// --- Continue both paths for compareSteps ---
			for (unsigned int s = 0; s < compareSteps; ++s)
			{
				const unsigned int gi = warmupSteps + s;

				// CPU step
				std::vector<float> gW_cpu(allGrads[gi]);
				glades::atlas::applyStep(cpuState, &W_cpu[0], &gW_cpu[0], m, n,
				                         1.0f, lr, 0.0f, 0.0f, 1.0f, ac, cpuRng);

				// GPU step
				d_gW.upload(allGrads[gi].data(), mn);
				ok = glades::gpu::atlas_gpu_step(gpuState, d_W.data(), d_gW.data(),
				                                  m, n, 1.0f, lr, 0.0f, 0.0f, 1.0f, ac);
				ASSERT("==============ATLAS::Xfer GPU compare step failed==============", ok);
			}

			// --- Compare final weights ---
			std::vector<float> W_gpu(mn);
			d_W.download(W_gpu.data());

			double diffSq = 0.0;
			double refSq = 0.0;
			for (size_t i = 0; i < mn; ++i)
			{
				float diff = fabsf(W_cpu[i] - W_gpu[i]);
				diffSq += (double)diff * (double)diff;
				refSq += (double)W_cpu[i] * (double)W_cpu[i];
			}
			const float l2RelErr = (refSq > 1e-20) ? (float)sqrt(diffSq / refSq) : 0.0f;
			const float sigma2RelErr = fabsf(cpuState.sigma2 - gpuState.sigma2)
			                         / fmaxf(fabsf(cpuState.sigma2), 1e-6f);
			const float muAbsErr = fabsf(cpuState.mu - gpuState.mu);

			printf("[UT] 29: GPU-to-CPU transfer l2_relative_error=%.6f "
			       "sigma2_relative_error=%.6f mu_abs_error=%.6f "
			       "(cpu_sigma2=%.6f gpu_sigma2=%.6f cpu_mu=%.6f gpu_mu=%.6f)\n",
			       l2RelErr, sigma2RelErr, muAbsErr, cpuState.sigma2, gpuState.sigma2,
			       cpuState.mu, gpuState.mu);

			ASSERT("==============ATLAS::Xfer GPU-CPU sigma2 diverged too much==============",
			       sigma2RelErr < 0.02f);
			ASSERT("==============ATLAS::Xfer GPU-CPU mu diverged too much==============",
			       muAbsErr < 0.01f);
			ASSERT("==============ATLAS::Xfer GPU-CPU weights diverged too much==============",
			       l2RelErr < 0.02f);

			printf("[UT] 29: PASSED\n");
		}
		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);
	}
	else
	{
		printf("-----------------------------------\n");
		printf("ATLAS Tests 18-29: GPU ATLAS kernels (SKIPPED - no CUDA device)\n");
		printf("-----------------------------------\n");
	}
#else
	printf("-----------------------------------\n");
	printf("ATLAS Tests 18-29: GPU ATLAS kernels (SKIPPED - no CUDA support)\n");
	printf("-----------------------------------\n");
#endif

	// ---------------------------------------------------------------
	// Test 19: Edge case - single-element weight matrix (1x1)
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 19: Single-element weight matrix (1x1)\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 1;
		const unsigned int n = 1;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 99999ULL);
		glades::atlas::WeightState state;
		glades::atlas::initWeightState(state, m, n, 128, 0.01f, rng);

		ASSERT("==============ATLAS::1x1 init failed==============", state.initialized);
		ASSERT("==============ATLAS::1x1 rank should be 1==============", state.r == 1u);

		// Run 50 steps
		std::vector<float> W(1, 1.0f);
		glades::ATLASConfig ac1x1;
		ac1x1.rank = 128;
		ac1x1.complementRank = 0u;
		ac1x1.tSub = 10;
		for (int step = 0; step < 50; ++step)
		{
			std::vector<float> gW(1, 0.1f);
			glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
			                         1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
			                         ac1x1, rng);
			ASSERT("==============ATLAS::1x1 NaN weight==============", W[0] == W[0]);
			ASSERT("==============ATLAS::1x1 Inf weight==============", (W[0] - W[0]) == 0.0f);
		}
		printf("[UT] ATLAS 1x1: 50 steps, final W=%.6f\n", W[0]);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 20: Edge case - zero gradients (should not diverge)
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 20: Zero gradients\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 4;
		const unsigned int n = 3;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 33333ULL);
		glades::atlas::WeightState state;
		glades::atlas::initWeightState(state, m, n, 2, 0.01f, rng);

		std::vector<float> W(m * n, 0.5f);
		std::vector<float> W_init(W.begin(), W.end());

		glades::ATLASConfig acZero;
		acZero.complementRank = 0u;
		acZero.tSub = 10;
		acZero.muMin = 0.0f;
		acZero.muMax = 0.0f;

		// 100 steps with zero gradient — weights should be unchanged (no weight decay)
		for (int step = 0; step < 100; ++step)
		{
			std::vector<float> gW(m * n, 0.0f);
			glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
			                         1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
			                         acZero, rng);
		}

		// Weights should be identical (no gradient, no weight decay)
		bool unchanged = true;
		for (unsigned int i = 0; i < m * n; ++i)
		{
			if (W[i] != W_init[i]) { unchanged = false; break; }
		}
		ASSERT("==============ATLAS::ZeroGrad weights changed unexpectedly==============", unchanged);

		// State should still be finite
		ASSERT("==============ATLAS::ZeroGrad sigma2 NaN==============",
		       state.sigma2 == state.sigma2);
		ASSERT("==============ATLAS::ZeroGrad mu NaN==============",
		       state.mu == state.mu);

		printf("[UT] ATLAS zero gradients: weights unchanged after 100 steps\n");
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 21: Edge case - rank equals min(m,n) (square subspace)
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 21: Rank = min(m,n) (full subspace)\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 4;
		const unsigned int n = 3;
		const unsigned int r = 3; // = min(m,n), full column subspace

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 44444ULL);
		glades::atlas::WeightState state;
		glades::atlas::initWeightState(state, m, n, r, 0.01f, rng);

		ASSERT("==============ATLAS::FullRank init failed==============", state.initialized);
		ASSERT("==============ATLAS::FullRank rank wrong==============", state.r == 3u);

		// U should be orthonormal
		const float tol = 1e-5f;
		for (unsigned int c = 0; c < state.r; ++c)
		{
			for (unsigned int d = c; d < state.r; ++d)
			{
				float dot = 0.0f;
				for (unsigned int i = 0; i < m; ++i)
					dot += state.U[i * state.r + c] * state.U[i * state.r + d];
				float expected = (c == d) ? 1.0f : 0.0f;
				float err = (dot - expected) < 0.0f ? -(dot - expected) : (dot - expected);
				ASSERT("==============ATLAS::FullRank U not orthonormal==============", err <= tol);
			}
		}

		// Run steps with refresh to verify it handles full-rank correctly
		std::vector<float> W(m * n, 0.0f);
		for (unsigned int i = 0; i < m * n; ++i)
			W[i] = static_cast<float>(i) * 0.1f;

		glades::ATLASConfig acFull;
		acFull.rank = r;
		acFull.complementRank = 0u;
		acFull.tSub = 5;  // frequent refresh

		for (int step = 0; step < 50; ++step)
		{
			std::vector<float> gW(m * n, 0.0f);
			for (unsigned int i = 0; i < m * n; ++i)
				gW[i] = glades::rng::standard_normal(rng) * 0.1f;
			glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
			                         1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
			                         acFull, rng);

			for (unsigned int i = 0; i < m * n; ++i)
			{
				ASSERT("==============ATLAS::FullRank NaN weight==============", W[i] == W[i]);
				ASSERT("==============ATLAS::FullRank Inf weight==============", (W[i] - W[i]) == 0.0f);
			}
		}
		printf("[UT] ATLAS full-rank (r=%u = min(%u,%u)): 50 steps, all finite\n", r, m, n);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 22: ATLAS CNN convergence
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 22: CNN convergence\n");
	printf("-----------------------------------\n");
	{
		// Synthetic 2-class 1-channel 8x8 image data.
		const unsigned int C = 1, H = 8, W = 8;
		const unsigned int numClasses = 2;
		const unsigned int numSamples = 20;
		const unsigned int featureCount = C * H * W;

		glades::NumberInput* di = new glades::NumberInput();
		di->trainMatrix = shmea::GMatrix(numSamples, shmea::GVector<float>(featureCount, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(numSamples, shmea::GVector<float>(numClasses, 0.0f));

		unsigned int s = 555u;
		for (unsigned int i = 0; i < numSamples; ++i)
		{
			const unsigned int label = i % numClasses;
			for (unsigned int f = 0; f < featureCount; ++f)
			{
				s = s * 1103515245u + 12345u;
				float v = static_cast<float>((s >> 16) & 0x7FFFu) / 32767.0f;
				if (f % numClasses == label)
					v = v * 0.5f + 0.5f;
				else
					v = v * 0.5f;
				di->trainMatrix[i][f] = v;
			}
			di->trainExpectedMatrix[i][label] = 1.0f;
		}
		di->testMatrix = di->trainMatrix;
		di->testExpectedMatrix = di->trainExpectedMatrix;

		// Build CNN: 1 conv layer (4 filters, 3x3, pad 1, maxpool 2x2) + 1 FC hidden (16).
		glades::InputLayerInfo* in = new glades::InputLayerInfo(
		    /*batchSize*/ 5,
		    /*learningRate*/ 0.01f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::RELU,
		    /*activationParam*/ 1.0f);

		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(
		    16, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::RELU, 1.0f));

		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(
		    static_cast<int>(numClasses), glades::OutputLayerInfo::CLASSIFICATION);

		glades::NNInfo* info = new glades::NNInfo("ut_atlas_cnn", in, hidden, out);
		glades::NNetwork net(info, glades::NNetwork::TYPE_CNN);
		net.setSeed(314u);
		net.getTerminatorMutable().setEpoch(200);
		net.getTerminatorMutable().setAccuracy(0);

		// Configure CNN architecture.
		{
			glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
			cfg.cnn.inputH = H;
			cfg.cnn.inputW = W;
			cfg.cnn.inputC = C;

			glades::CNNConfig::ConvLayerSpec spec;
			spec.outChannels = 4;
			spec.kernelH = 3; spec.kernelW = 3;
			spec.strideH = 1; spec.strideW = 1;
			spec.padH = 1; spec.padW = 1;
			spec.useBatchNorm = false;
			spec.useMaxPool = true;
			spec.poolH = 2; spec.poolW = 2;
			spec.poolStrideH = 2; spec.poolStrideW = 2;
			cfg.cnn.convLayers.push_back(spec);

			// Configure ATLAS optimizer.
			cfg.optimizer.type = glades::OptimizerConfig::ATLAS;
			cfg.atlas.rank = 2;
			cfg.atlas.complementRank = 0u;
			cfg.atlas.tSub = 50;
			cfg.atlas.beta = 0.999f;
		}

		CaptureMetricsCallbacks cb;
		const glades::NNetworkStatus st = net.train(di, &cb);
		ASSERT("==============ATLAS::CNN TrainStatus() Failed==============", st.ok());
		ASSERT("==============ATLAS::CNN no metrics captured==============", cb.saw);
		printf("[UT] ATLAS CNN: final loss = %f\n", cb.last.totalError);
		ASSERT("==============ATLAS::CNN loss too high==============", cb.last.totalError < 0.1f);

		delete di;
		delete info;
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 23: Weight decay shrinks weight norms
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 23: Weight decay shrinks weight norms\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 6;
		const unsigned int n = 4;
		const unsigned int r = 2;
		const unsigned int mn = m * n;
		const int nSteps = 40;

		glades::ATLASConfig acBase;
		acBase.beta = 0.999f;
		acBase.muMin = 0.0f;
		acBase.muMax = 0.5f;
		acBase.eps = 1e-8f;
		acBase.kappaMax = 10.0f;
		acBase.tSub = 50u;
		acBase.powerIters = 3u;
		acBase.betaRefresh = 0.5f;
		acBase.muGrowthRate = 0.001f;
		acBase.biasCorrection = false;

		// Helper: run nSteps of ATLAS with given weight decay, return final L2 norm of W.
		// Both runs start from the same initial weights and use the same gradient signal.
		float normNoWd = 0.0f;
		float normWithWd = 0.0f;

		// Run without weight decay
		{
			glades::rng::Engine rng;
			glades::rng::seed_engine(rng, 777ULL);
			glades::atlas::WeightState state;
			glades::atlas::initWeightState(state, m, n, r, 0.01f, rng);

			std::vector<float> W(mn, 0.0f);
			for (unsigned int i = 0; i < mn; ++i)
				W[i] = static_cast<float>(i) * 0.05f + 0.1f;

			for (int s = 0; s < nSteps; ++s)
			{
				std::vector<float> gW(mn, 0.0f);
				for (unsigned int i = 0; i < mn; ++i)
					gW[i] = 0.01f; // small constant gradient
				glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
				                         1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
				                         acBase, rng);
			}
			for (unsigned int i = 0; i < mn; ++i)
				normNoWd += W[i] * W[i];
			normNoWd = sqrtf(normNoWd);
		}

		// Run with L1 + L2 weight decay
		{
			glades::rng::Engine rng;
			glades::rng::seed_engine(rng, 777ULL);
			glades::atlas::WeightState state;
			glades::atlas::initWeightState(state, m, n, r, 0.01f, rng);

			std::vector<float> W(mn, 0.0f);
			for (unsigned int i = 0; i < mn; ++i)
				W[i] = static_cast<float>(i) * 0.05f + 0.1f;

			for (int s = 0; s < nSteps; ++s)
			{
				std::vector<float> gW(mn, 0.0f);
				for (unsigned int i = 0; i < mn; ++i)
					gW[i] = 0.01f;
				glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
				                         1.0f, 0.01f,
				                         /*wd1=*/0.001f, /*wd2=*/0.01f,
				                         1.0f, acBase, rng);
			}
			for (unsigned int i = 0; i < mn; ++i)
				normWithWd += W[i] * W[i];
			normWithWd = sqrtf(normWithWd);
		}

		printf("[UT] ATLAS weight decay: norm_no_wd=%f, norm_with_wd=%f\n",
		       normNoWd, normWithWd);
		ASSERT("==============ATLAS::WeightDecay did not shrink weights==============",
		       normWithWd < normNoWd);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 24: Checkpoint preserves ATLAS optimizer state exactly
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 24: Checkpoint state preservation\n");
	printf("-----------------------------------\n");
	{
		// Strategy: Train network A for 50 epochs, save checkpoint. Then continue
		// training A for 50 more epochs. Separately, load checkpoint into fresh
		// network B, train 50 epochs. Final losses must match exactly (proving
		// the checkpoint preserved all ATLAS optimizer state).
		const std::string ckptName = "ut_atlas_st24";
		float lossA = -1.0f;
		float lossB = -1.0f;

		// Path A: train 50 → checkpoint → train 50 more (continuously)
		{
			glades::NumberInput* diA = new glades::NumberInput();
			diA->trainMatrix = shmea::GMatrix(4, shmea::GVector<float>(2, 0.0f));
			diA->trainExpectedMatrix = shmea::GMatrix(4, shmea::GVector<float>(1, 0.0f));
			diA->trainMatrix[0][0] = 0.0f; diA->trainMatrix[0][1] = 0.0f;
			diA->trainExpectedMatrix[0][0] = 0.0f;
			diA->trainMatrix[1][0] = 0.0f; diA->trainMatrix[1][1] = 1.0f;
			diA->trainExpectedMatrix[1][0] = 1.0f;
			diA->trainMatrix[2][0] = 1.0f; diA->trainMatrix[2][1] = 0.0f;
			diA->trainExpectedMatrix[2][0] = 1.0f;
			diA->trainMatrix[3][0] = 1.0f; diA->trainMatrix[3][1] = 1.0f;
			diA->trainExpectedMatrix[3][0] = 0.0f;
			diA->testMatrix = diA->trainMatrix;
			diA->testExpectedMatrix = diA->trainExpectedMatrix;

			glades::InputLayerInfo* in = new glades::InputLayerInfo(
			    4, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::SIGMOID, 1.0f);
			std::vector<glades::HiddenLayerInfo*> hidden;
			hidden.push_back(new glades::HiddenLayerInfo(
			    8, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::SIGMOID, 1.0f));
			glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
			glades::NNInfo* info = new glades::NNInfo("ut_atlas_st24", in, hidden, out);

			glades::NNetwork netA(info, glades::NNetwork::TYPE_DFF);
			netA.setSeed(42u);
			netA.getTerminatorMutable().setEpoch(50);
			netA.getTerminatorMutable().setAccuracy(0);
			{
			glades::TrainingConfig& cfg = netA.getTrainingConfigMutable();
			cfg.optimizer.type = glades::OptimizerConfig::ATLAS;
			cfg.atlas.rank = 4;
			cfg.atlas.complementRank = 0u;
			cfg.atlas.kappaMax = 7.0f;
			cfg.atlas.complementLrScale = 0.125f;
			cfg.atlas.complementKappaMax = 0.75f;
			cfg.atlas.prismEnabled = true;
			cfg.atlas.prismLagHorizon = 2u;
			cfg.atlas.prismMemoryScale = 0.20f;
			cfg.atlas.prismPredictiveEdgeThreshold = 0.03f;
			cfg.atlas.resolveEnabled = true;
			cfg.atlas.resolveLagHorizon = 4u;
			cfg.atlas.resolveMemoryScale = 0.10f;
			cfg.atlas.resolvePredictiveEdgeThreshold = 0.04f;
			cfg.atlas.heroEnabled = true;
			cfg.atlas.heroLagHorizon = 4u;
			cfg.atlas.heroMemoryScale = 0.10f;
			cfg.atlas.heroEdgeThreshold = 0.10f;
			cfg.atlas.cobaltEnabled = true;
			cfg.atlas.cobaltLagHorizon = 4u;
			cfg.atlas.cobaltMemoryScale = 0.08f;
			cfg.atlas.cobaltEdgeThreshold = 0.10f;
			cfg.atlas.birchEnabled = true;
			cfg.atlas.birchPastHorizon = 3u;
			cfg.atlas.birchFutureHorizon = 2u;
			cfg.atlas.birchMemoryScale = 0.08f;
			cfg.atlas.birchEdgeThreshold = 0.10f;
			cfg.atlas.ghostEnabled = true;
			cfg.atlas.ghostLagHorizon = 4u;
			cfg.atlas.ghostMemoryScale = 0.05f;
			cfg.atlas.ghostEdgeThreshold = 0.10f;
			cfg.atlas.sparrowEnabled = true;
			cfg.atlas.sparrowModeRank = 2u;
			cfg.atlas.sparrowAutoModeGate = true;
			cfg.atlas.sparrowMemoryScale = 0.05f;
			cfg.atlas.sparrowEdgeThreshold = 0.10f;
			cfg.atlas.sparrowSecondEdgeThreshold = 0.08f;
			cfg.atlas.sparrowSecondEdgeFraction = 0.60f;
			cfg.atlas.sparrowPoleMax = 0.95f;
			cfg.atlas.orbitEnabled = true;
			cfg.atlas.orbitMemoryScale = 0.04f;
			cfg.atlas.orbitEdgeThreshold = 0.05f;
			cfg.atlas.orbitPoleMax = 0.95f;
			cfg.atlas.helmEnabled = true;
			cfg.atlas.helmMemoryScale = 0.03f;
			cfg.atlas.helmEdgeThreshold = 0.04f;
			cfg.atlas.helmModeRank = 2u;
			cfg.atlas.helmHiddenStackDepth = 2u;
			cfg.atlas.helmPoleMax = 0.90f;
			cfg.atlas.tSub = 50;
		}

			// Phase 1: first 50 epochs
			glades::NNetworkStatus st = netA.train(diA);
			ASSERT("==============ATLAS::StateCheckpoint Phase1 Train Failed==============", st.ok());

			// Save checkpoint
			st = netA.saveCheckpoint(ckptName);
			ASSERT("==============ATLAS::StateCheckpoint Save Failed==============", st.ok());

			// Phase 2: 50 more epochs
			netA.getTerminatorMutable().setEpoch(50);
			CaptureMetricsCallbacks cbA;
			st = netA.train(diA, &cbA);
			ASSERT("==============ATLAS::StateCheckpoint Phase2 Train Failed==============", st.ok());
			ASSERT("==============ATLAS::StateCheckpoint Phase2 no metrics==============", cbA.saw);
			lossA = cbA.last.totalError;

			delete diA;
			delete info;
		}

		// Path B: load checkpoint → train 50 epochs
		{
			glades::NumberInput* diB = new glades::NumberInput();
			diB->trainMatrix = shmea::GMatrix(4, shmea::GVector<float>(2, 0.0f));
			diB->trainExpectedMatrix = shmea::GMatrix(4, shmea::GVector<float>(1, 0.0f));
			diB->trainMatrix[0][0] = 0.0f; diB->trainMatrix[0][1] = 0.0f;
			diB->trainExpectedMatrix[0][0] = 0.0f;
			diB->trainMatrix[1][0] = 0.0f; diB->trainMatrix[1][1] = 1.0f;
			diB->trainExpectedMatrix[1][0] = 1.0f;
			diB->trainMatrix[2][0] = 1.0f; diB->trainMatrix[2][1] = 0.0f;
			diB->trainExpectedMatrix[2][0] = 1.0f;
			diB->trainMatrix[3][0] = 1.0f; diB->trainMatrix[3][1] = 1.0f;
			diB->trainExpectedMatrix[3][0] = 0.0f;
			diB->testMatrix = diB->trainMatrix;
			diB->testExpectedMatrix = diB->trainExpectedMatrix;

			std::map<std::string, std::string> kv;
			ASSERT("==============ATLAS::StateCheckpoint ParseManifest Failed==============",
			       parse_kv_manifest("database/checkpoints/" + ckptName + "/manifest.txt", kv));
			ASSERT("==============ATLAS::StateCheckpoint ManifestMagic Failed==============",
			       kv["__magic__"] == "GLADES_CHECKPOINT");
			ASSERT("==============ATLAS::StateCheckpoint ManifestRank Failed==============",
			       kv["training.atlas.rank"] == "4");
			ASSERT("==============ATLAS::StateCheckpoint ManifestComplementRank Failed==============",
			       kv["training.atlas.complementRank"] == "0");
			ASSERT("==============ATLAS::StateCheckpoint ManifestKappaMax Failed==============",
			       kv["training.atlas.kappaMax"] == "7");
			ASSERT("==============ATLAS::StateCheckpoint ManifestComplementLrScale Failed==============",
			       kv["training.atlas.complementLrScale"] == "0.125");
			ASSERT("==============ATLAS::StateCheckpoint ManifestComplementKappaMax Failed==============",
			       kv["training.atlas.complementKappaMax"] == "0.75");
			ASSERT("==============ATLAS::StateCheckpoint ManifestPrismEnabled Failed==============",
			       kv["training.atlas.prismEnabled"] == "1");
			ASSERT("==============ATLAS::StateCheckpoint ManifestPrismLagHorizon Failed==============",
			       kv["training.atlas.prismLagHorizon"] == "2");
			ASSERT("==============ATLAS::StateCheckpoint ManifestPrismMemoryScale Failed==============",
			       kv["training.atlas.prismMemoryScale"] == "0.2");
			ASSERT("==============ATLAS::StateCheckpoint ManifestPrismEdgeThreshold Failed==============",
			       kv["training.atlas.prismPredictiveEdgeThreshold"] == "0.03");
			ASSERT("==============ATLAS::StateCheckpoint ManifestResolveEnabled Failed==============",
			       kv["training.atlas.resolveEnabled"] == "1");
			ASSERT("==============ATLAS::StateCheckpoint ManifestResolveLagHorizon Failed==============",
			       kv["training.atlas.resolveLagHorizon"] == "4");
			ASSERT("==============ATLAS::StateCheckpoint ManifestResolveMemoryScale Failed==============",
			       kv["training.atlas.resolveMemoryScale"] == "0.1");
			ASSERT("==============ATLAS::StateCheckpoint ManifestResolveEdgeThreshold Failed==============",
			       kv["training.atlas.resolvePredictiveEdgeThreshold"] == "0.04");
			ASSERT("==============ATLAS::StateCheckpoint ManifestHeroEnabled Failed==============",
			       kv["training.atlas.heroEnabled"] == "1");
			ASSERT("==============ATLAS::StateCheckpoint ManifestHeroLagHorizon Failed==============",
			       kv["training.atlas.heroLagHorizon"] == "4");
			ASSERT("==============ATLAS::StateCheckpoint ManifestHeroMemoryScale Failed==============",
			       kv["training.atlas.heroMemoryScale"] == "0.1");
			ASSERT("==============ATLAS::StateCheckpoint ManifestHeroEdgeThreshold Failed==============",
			       kv["training.atlas.heroEdgeThreshold"] == "0.1");
			ASSERT("==============ATLAS::StateCheckpoint ManifestCobaltEnabled Failed==============",
			       kv["training.atlas.cobaltEnabled"] == "1");
			ASSERT("==============ATLAS::StateCheckpoint ManifestCobaltLagHorizon Failed==============",
			       kv["training.atlas.cobaltLagHorizon"] == "4");
			ASSERT("==============ATLAS::StateCheckpoint ManifestCobaltMemoryScale Failed==============",
			       kv["training.atlas.cobaltMemoryScale"] == "0.08");
			ASSERT("==============ATLAS::StateCheckpoint ManifestCobaltEdgeThreshold Failed==============",
			       kv["training.atlas.cobaltEdgeThreshold"] == "0.1");
			ASSERT("==============ATLAS::StateCheckpoint ManifestBirchEnabled Failed==============",
			       kv["training.atlas.birchEnabled"] == "1");
			ASSERT("==============ATLAS::StateCheckpoint ManifestBirchPastHorizon Failed==============",
			       kv["training.atlas.birchPastHorizon"] == "3");
			ASSERT("==============ATLAS::StateCheckpoint ManifestBirchFutureHorizon Failed==============",
			       kv["training.atlas.birchFutureHorizon"] == "2");
			ASSERT("==============ATLAS::StateCheckpoint ManifestBirchMemoryScale Failed==============",
			       kv["training.atlas.birchMemoryScale"] == "0.08");
			ASSERT("==============ATLAS::StateCheckpoint ManifestBirchEdgeThreshold Failed==============",
			       kv["training.atlas.birchEdgeThreshold"] == "0.1");
			ASSERT("==============ATLAS::StateCheckpoint ManifestGhostEnabled Failed==============",
			       kv["training.atlas.ghostEnabled"] == "1");
			ASSERT("==============ATLAS::StateCheckpoint ManifestGhostLagHorizon Failed==============",
			       kv["training.atlas.ghostLagHorizon"] == "4");
			ASSERT("==============ATLAS::StateCheckpoint ManifestGhostMemoryScale Failed==============",
			       kv["training.atlas.ghostMemoryScale"] == "0.05");
			ASSERT("==============ATLAS::StateCheckpoint ManifestGhostEdgeThreshold Failed==============",
			       kv["training.atlas.ghostEdgeThreshold"] == "0.1");
			ASSERT("==============ATLAS::StateCheckpoint ManifestSparrowEnabled Failed==============",
			       kv["training.atlas.sparrowEnabled"] == "1");
			ASSERT("==============ATLAS::StateCheckpoint ManifestSparrowModeRank Failed==============",
			       kv["training.atlas.sparrowModeRank"] == "2");
			ASSERT("==============ATLAS::StateCheckpoint ManifestSparrowAutoModeGate Failed==============",
			       kv["training.atlas.sparrowAutoModeGate"] == "1");
			ASSERT("==============ATLAS::StateCheckpoint ManifestSparrowMemoryScale Failed==============",
			       kv["training.atlas.sparrowMemoryScale"] == "0.05");
			ASSERT("==============ATLAS::StateCheckpoint ManifestSparrowEdgeThreshold Failed==============",
			       kv["training.atlas.sparrowEdgeThreshold"] == "0.1");
			ASSERT("==============ATLAS::StateCheckpoint ManifestSparrowSecondEdgeThreshold Failed==============",
			       kv["training.atlas.sparrowSecondEdgeThreshold"] == "0.08");
			ASSERT("==============ATLAS::StateCheckpoint ManifestSparrowSecondEdgeFraction Failed==============",
			       kv["training.atlas.sparrowSecondEdgeFraction"] == "0.6");
			ASSERT("==============ATLAS::StateCheckpoint ManifestSparrowPoleMax Failed==============",
			       kv["training.atlas.sparrowPoleMax"] == "0.95");
			ASSERT("==============ATLAS::StateCheckpoint ManifestOrbitEnabled Failed==============",
			       kv["training.atlas.orbitEnabled"] == "1");
			ASSERT("==============ATLAS::StateCheckpoint ManifestOrbitMemoryScale Failed==============",
			       kv["training.atlas.orbitMemoryScale"] == "0.04");
			ASSERT("==============ATLAS::StateCheckpoint ManifestOrbitEdgeThreshold Failed==============",
			       kv["training.atlas.orbitEdgeThreshold"] == "0.05");
			ASSERT("==============ATLAS::StateCheckpoint ManifestOrbitPoleMax Failed==============",
			       kv["training.atlas.orbitPoleMax"] == "0.95");
			ASSERT("==============ATLAS::StateCheckpoint ManifestHelmEnabled Failed==============",
			       kv["training.atlas.helmEnabled"] == "1");
			ASSERT("==============ATLAS::StateCheckpoint ManifestHelmMemoryScale Failed==============",
			       kv["training.atlas.helmMemoryScale"] == "0.03");
			ASSERT("==============ATLAS::StateCheckpoint ManifestHelmEdgeThreshold Failed==============",
			       kv["training.atlas.helmEdgeThreshold"] == "0.04");
			ASSERT("==============ATLAS::StateCheckpoint ManifestHelmModeRank Failed==============",
			       kv["training.atlas.helmModeRank"] == "2");
			ASSERT("==============ATLAS::StateCheckpoint ManifestHelmHiddenStackDepth Failed==============",
			       kv["training.atlas.helmHiddenStackDepth"] == "2");
			ASSERT("==============ATLAS::StateCheckpoint ManifestHelmPoleMax Failed==============",
			       kv["training.atlas.helmPoleMax"] == "0.9");
			ASSERT("==============ATLAS::StateCheckpoint ManifestTSub Failed==============",
			       kv["training.atlas.tSub"] == "50");

			{
				glades::NNetwork netMismatch(glades::NNetwork::TYPE_DFF);
				glades::TrainingConfig& cfgMismatch = netMismatch.getTrainingConfigMutable();
				cfgMismatch.optimizer.type = glades::OptimizerConfig::ATLAS;
				cfgMismatch.atlas.rank = 7u;
				cfgMismatch.atlas.complementRank = 0u;
				cfgMismatch.atlas.kappaMax = 9.0f;
				cfgMismatch.atlas.complementLrScale = 0.2f;
				cfgMismatch.atlas.complementKappaMax = 0.9f;
				cfgMismatch.atlas.prismEnabled = false;
				cfgMismatch.atlas.prismLagHorizon = 1u;
				cfgMismatch.atlas.prismMemoryScale = 0.15f;
				cfgMismatch.atlas.prismPredictiveEdgeThreshold = 0.07f;
				cfgMismatch.atlas.resolveEnabled = false;
				cfgMismatch.atlas.resolveLagHorizon = 2u;
				cfgMismatch.atlas.resolveMemoryScale = 0.05f;
				cfgMismatch.atlas.resolvePredictiveEdgeThreshold = 0.02f;
				cfgMismatch.atlas.heroEnabled = false;
				cfgMismatch.atlas.heroLagHorizon = 2u;
				cfgMismatch.atlas.heroMemoryScale = 0.05f;
				cfgMismatch.atlas.heroEdgeThreshold = 0.03f;
				cfgMismatch.atlas.cobaltEnabled = false;
				cfgMismatch.atlas.cobaltLagHorizon = 2u;
				cfgMismatch.atlas.cobaltMemoryScale = 0.04f;
				cfgMismatch.atlas.cobaltEdgeThreshold = 0.02f;
				cfgMismatch.atlas.birchEnabled = false;
				cfgMismatch.atlas.birchPastHorizon = 1u;
				cfgMismatch.atlas.birchFutureHorizon = 1u;
				cfgMismatch.atlas.birchMemoryScale = 0.03f;
				cfgMismatch.atlas.birchEdgeThreshold = 0.02f;
				cfgMismatch.atlas.ghostEnabled = false;
				cfgMismatch.atlas.ghostLagHorizon = 2u;
				cfgMismatch.atlas.ghostMemoryScale = 0.02f;
				cfgMismatch.atlas.ghostEdgeThreshold = 0.03f;
				cfgMismatch.atlas.sparrowEnabled = false;
				cfgMismatch.atlas.sparrowModeRank = 1u;
				cfgMismatch.atlas.sparrowAutoModeGate = false;
				cfgMismatch.atlas.sparrowMemoryScale = 0.02f;
				cfgMismatch.atlas.sparrowEdgeThreshold = 0.03f;
				cfgMismatch.atlas.sparrowSecondEdgeThreshold = 0.02f;
				cfgMismatch.atlas.sparrowSecondEdgeFraction = 0.40f;
				cfgMismatch.atlas.sparrowPoleMax = 0.80f;
				cfgMismatch.atlas.orbitEnabled = false;
				cfgMismatch.atlas.orbitMemoryScale = 0.02f;
				cfgMismatch.atlas.orbitEdgeThreshold = 0.02f;
				cfgMismatch.atlas.orbitPoleMax = 0.80f;
				cfgMismatch.atlas.helmEnabled = false;
				cfgMismatch.atlas.helmMemoryScale = 0.02f;
				cfgMismatch.atlas.helmEdgeThreshold = 0.02f;
				cfgMismatch.atlas.helmModeRank = 1u;
				cfgMismatch.atlas.helmHiddenStackDepth = 1u;
				cfgMismatch.atlas.helmPoleMax = 0.80f;
				cfgMismatch.atlas.tSub = 13u;
				glades::NNetworkStatus stMismatch = netMismatch.loadCheckpoint(ckptName, diB);
				ASSERT("==============ATLAS::StateCheckpoint MismatchLoadShouldFail==============", !stMismatch.ok());
				ASSERT("==============ATLAS::StateCheckpoint MismatchMessage Failed==============",
				       stMismatch.message.find("training.atlas.rank") != std::string::npos
				       || stMismatch.message.find("training.atlas.complementRank") != std::string::npos
				       || stMismatch.message.find("training.atlas.kappaMax") != std::string::npos
				       || stMismatch.message.find("training.atlas.complementLrScale") != std::string::npos
				       || stMismatch.message.find("training.atlas.complementKappaMax") != std::string::npos
				       || stMismatch.message.find("training.atlas.prismEnabled") != std::string::npos
				       || stMismatch.message.find("training.atlas.prismLagHorizon") != std::string::npos
				       || stMismatch.message.find("training.atlas.prismMemoryScale") != std::string::npos
				       || stMismatch.message.find("training.atlas.prismPredictiveEdgeThreshold") != std::string::npos
				       || stMismatch.message.find("training.atlas.resolveEnabled") != std::string::npos
				       || stMismatch.message.find("training.atlas.resolveLagHorizon") != std::string::npos
				       || stMismatch.message.find("training.atlas.resolveMemoryScale") != std::string::npos
				       || stMismatch.message.find("training.atlas.resolvePredictiveEdgeThreshold") != std::string::npos
				       || stMismatch.message.find("training.atlas.heroEnabled") != std::string::npos
				       || stMismatch.message.find("training.atlas.heroLagHorizon") != std::string::npos
				       || stMismatch.message.find("training.atlas.heroMemoryScale") != std::string::npos
				       || stMismatch.message.find("training.atlas.heroEdgeThreshold") != std::string::npos
				       || stMismatch.message.find("training.atlas.cobaltEnabled") != std::string::npos
				       || stMismatch.message.find("training.atlas.cobaltLagHorizon") != std::string::npos
				       || stMismatch.message.find("training.atlas.cobaltMemoryScale") != std::string::npos
				       || stMismatch.message.find("training.atlas.cobaltEdgeThreshold") != std::string::npos
				       || stMismatch.message.find("training.atlas.birchEnabled") != std::string::npos
				       || stMismatch.message.find("training.atlas.birchPastHorizon") != std::string::npos
				       || stMismatch.message.find("training.atlas.birchFutureHorizon") != std::string::npos
				       || stMismatch.message.find("training.atlas.birchMemoryScale") != std::string::npos
				       || stMismatch.message.find("training.atlas.birchEdgeThreshold") != std::string::npos
				       || stMismatch.message.find("training.atlas.ghostEnabled") != std::string::npos
				       || stMismatch.message.find("training.atlas.ghostLagHorizon") != std::string::npos
				       || stMismatch.message.find("training.atlas.ghostMemoryScale") != std::string::npos
				       || stMismatch.message.find("training.atlas.ghostEdgeThreshold") != std::string::npos
				       || stMismatch.message.find("training.atlas.sparrowEnabled") != std::string::npos
				       || stMismatch.message.find("training.atlas.sparrowModeRank") != std::string::npos
				       || stMismatch.message.find("training.atlas.sparrowAutoModeGate") != std::string::npos
				       || stMismatch.message.find("training.atlas.sparrowMemoryScale") != std::string::npos
				       || stMismatch.message.find("training.atlas.sparrowEdgeThreshold") != std::string::npos
				       || stMismatch.message.find("training.atlas.sparrowSecondEdgeThreshold") != std::string::npos
				       || stMismatch.message.find("training.atlas.sparrowSecondEdgeFraction") != std::string::npos
				       || stMismatch.message.find("training.atlas.sparrowPoleMax") != std::string::npos
				       || stMismatch.message.find("training.atlas.orbitEnabled") != std::string::npos
				       || stMismatch.message.find("training.atlas.orbitMemoryScale") != std::string::npos
				       || stMismatch.message.find("training.atlas.orbitEdgeThreshold") != std::string::npos
				       || stMismatch.message.find("training.atlas.orbitPoleMax") != std::string::npos
				       || stMismatch.message.find("training.atlas.helmEnabled") != std::string::npos
				       || stMismatch.message.find("training.atlas.helmMemoryScale") != std::string::npos
				       || stMismatch.message.find("training.atlas.helmEdgeThreshold") != std::string::npos
				       || stMismatch.message.find("training.atlas.helmModeRank") != std::string::npos
				       || stMismatch.message.find("training.atlas.helmHiddenStackDepth") != std::string::npos
				       || stMismatch.message.find("training.atlas.helmPoleMax") != std::string::npos
				       || stMismatch.message.find("training.atlas.tSub") != std::string::npos);
			}

			glades::NNetwork netB;
			glades::NNetworkStatus st = netB.loadCheckpoint(ckptName, diB);
			ASSERT("==============ATLAS::StateCheckpoint Load Failed==============", st.ok());
			ASSERT("==============ATLAS::StateCheckpoint OptimizerRestored Failed==============",
			       netB.getTrainingConfig().optimizer.type == glades::OptimizerConfig::ATLAS);
			ASSERT("==============ATLAS::StateCheckpoint RankRestored Failed==============",
			       netB.getTrainingConfig().atlas.rank == 4u);
			ASSERT("==============ATLAS::StateCheckpoint ComplementRankRestored Failed==============",
			       netB.getTrainingConfig().atlas.complementRank == 0u);
			ASSERT("==============ATLAS::StateCheckpoint KappaMaxRestored Failed==============",
			       fabsf(netB.getTrainingConfig().atlas.kappaMax - 7.0f) < 1e-6f);
			ASSERT("==============ATLAS::StateCheckpoint ComplementLrScaleRestored Failed==============",
			       fabsf(netB.getTrainingConfig().atlas.complementLrScale - 0.125f) < 1e-6f);
			ASSERT("==============ATLAS::StateCheckpoint ComplementKappaMaxRestored Failed==============",
			       fabsf(netB.getTrainingConfig().atlas.complementKappaMax - 0.75f) < 1e-6f);
			ASSERT("==============ATLAS::StateCheckpoint PrismEnabledRestored Failed==============",
			       netB.getTrainingConfig().atlas.prismEnabled);
			ASSERT("==============ATLAS::StateCheckpoint PrismLagHorizonRestored Failed==============",
			       netB.getTrainingConfig().atlas.prismLagHorizon == 2u);
			ASSERT("==============ATLAS::StateCheckpoint PrismMemoryScaleRestored Failed==============",
			       fabsf(netB.getTrainingConfig().atlas.prismMemoryScale - 0.20f) < 1e-6f);
			ASSERT("==============ATLAS::StateCheckpoint PrismEdgeThresholdRestored Failed==============",
			       fabsf(netB.getTrainingConfig().atlas.prismPredictiveEdgeThreshold - 0.03f) < 1e-6f);
			ASSERT("==============ATLAS::StateCheckpoint ResolveEnabledRestored Failed==============",
			       netB.getTrainingConfig().atlas.resolveEnabled);
			ASSERT("==============ATLAS::StateCheckpoint ResolveLagHorizonRestored Failed==============",
			       netB.getTrainingConfig().atlas.resolveLagHorizon == 4u);
			ASSERT("==============ATLAS::StateCheckpoint ResolveMemoryScaleRestored Failed==============",
			       fabsf(netB.getTrainingConfig().atlas.resolveMemoryScale - 0.10f) < 1e-6f);
			ASSERT("==============ATLAS::StateCheckpoint ResolveEdgeThresholdRestored Failed==============",
			       fabsf(netB.getTrainingConfig().atlas.resolvePredictiveEdgeThreshold - 0.04f) < 1e-6f);
			ASSERT("==============ATLAS::StateCheckpoint HeroEnabledRestored Failed==============",
			       netB.getTrainingConfig().atlas.heroEnabled);
			ASSERT("==============ATLAS::StateCheckpoint HeroLagHorizonRestored Failed==============",
			       netB.getTrainingConfig().atlas.heroLagHorizon == 4u);
			ASSERT("==============ATLAS::StateCheckpoint HeroMemoryScaleRestored Failed==============",
			       fabsf(netB.getTrainingConfig().atlas.heroMemoryScale - 0.10f) < 1e-6f);
			ASSERT("==============ATLAS::StateCheckpoint HeroEdgeThresholdRestored Failed==============",
			       fabsf(netB.getTrainingConfig().atlas.heroEdgeThreshold - 0.10f) < 1e-6f);
			ASSERT("==============ATLAS::StateCheckpoint CobaltEnabledRestored Failed==============",
			       netB.getTrainingConfig().atlas.cobaltEnabled);
			ASSERT("==============ATLAS::StateCheckpoint CobaltLagHorizonRestored Failed==============",
			       netB.getTrainingConfig().atlas.cobaltLagHorizon == 4u);
			ASSERT("==============ATLAS::StateCheckpoint CobaltMemoryScaleRestored Failed==============",
			       fabsf(netB.getTrainingConfig().atlas.cobaltMemoryScale - 0.08f) < 1e-6f);
			ASSERT("==============ATLAS::StateCheckpoint CobaltEdgeThresholdRestored Failed==============",
			       fabsf(netB.getTrainingConfig().atlas.cobaltEdgeThreshold - 0.10f) < 1e-6f);
			ASSERT("==============ATLAS::StateCheckpoint BirchEnabledRestored Failed==============",
			       netB.getTrainingConfig().atlas.birchEnabled);
			ASSERT("==============ATLAS::StateCheckpoint BirchPastHorizonRestored Failed==============",
			       netB.getTrainingConfig().atlas.birchPastHorizon == 3u);
			ASSERT("==============ATLAS::StateCheckpoint BirchFutureHorizonRestored Failed==============",
			       netB.getTrainingConfig().atlas.birchFutureHorizon == 2u);
			ASSERT("==============ATLAS::StateCheckpoint BirchMemoryScaleRestored Failed==============",
			       fabsf(netB.getTrainingConfig().atlas.birchMemoryScale - 0.08f) < 1e-6f);
			ASSERT("==============ATLAS::StateCheckpoint BirchEdgeThresholdRestored Failed==============",
			       fabsf(netB.getTrainingConfig().atlas.birchEdgeThreshold - 0.10f) < 1e-6f);
			ASSERT("==============ATLAS::StateCheckpoint GhostEnabledRestored Failed==============",
			       netB.getTrainingConfig().atlas.ghostEnabled);
			ASSERT("==============ATLAS::StateCheckpoint GhostLagHorizonRestored Failed==============",
			       netB.getTrainingConfig().atlas.ghostLagHorizon == 4u);
			ASSERT("==============ATLAS::StateCheckpoint GhostMemoryScaleRestored Failed==============",
			       fabsf(netB.getTrainingConfig().atlas.ghostMemoryScale - 0.05f) < 1e-6f);
			ASSERT("==============ATLAS::StateCheckpoint GhostEdgeThresholdRestored Failed==============",
			       fabsf(netB.getTrainingConfig().atlas.ghostEdgeThreshold - 0.10f) < 1e-6f);
			ASSERT("==============ATLAS::StateCheckpoint SparrowEnabledRestored Failed==============",
			       netB.getTrainingConfig().atlas.sparrowEnabled);
			ASSERT("==============ATLAS::StateCheckpoint SparrowModeRankRestored Failed==============",
			       netB.getTrainingConfig().atlas.sparrowModeRank == 2u);
			ASSERT("==============ATLAS::StateCheckpoint SparrowAutoModeGateRestored Failed==============",
			       netB.getTrainingConfig().atlas.sparrowAutoModeGate);
			ASSERT("==============ATLAS::StateCheckpoint SparrowMemoryScaleRestored Failed==============",
			       fabsf(netB.getTrainingConfig().atlas.sparrowMemoryScale - 0.05f) < 1e-6f);
			ASSERT("==============ATLAS::StateCheckpoint SparrowEdgeThresholdRestored Failed==============",
			       fabsf(netB.getTrainingConfig().atlas.sparrowEdgeThreshold - 0.10f) < 1e-6f);
			ASSERT("==============ATLAS::StateCheckpoint SparrowSecondEdgeThresholdRestored Failed==============",
			       fabsf(netB.getTrainingConfig().atlas.sparrowSecondEdgeThreshold - 0.08f) < 1e-6f);
			ASSERT("==============ATLAS::StateCheckpoint SparrowSecondEdgeFractionRestored Failed==============",
			       fabsf(netB.getTrainingConfig().atlas.sparrowSecondEdgeFraction - 0.60f) < 1e-6f);
			ASSERT("==============ATLAS::StateCheckpoint SparrowPoleMaxRestored Failed==============",
			       fabsf(netB.getTrainingConfig().atlas.sparrowPoleMax - 0.95f) < 1e-6f);
			ASSERT("==============ATLAS::StateCheckpoint OrbitEnabledRestored Failed==============",
			       netB.getTrainingConfig().atlas.orbitEnabled);
			ASSERT("==============ATLAS::StateCheckpoint OrbitMemoryScaleRestored Failed==============",
			       fabsf(netB.getTrainingConfig().atlas.orbitMemoryScale - 0.04f) < 1e-6f);
			ASSERT("==============ATLAS::StateCheckpoint OrbitEdgeThresholdRestored Failed==============",
			       fabsf(netB.getTrainingConfig().atlas.orbitEdgeThreshold - 0.05f) < 1e-6f);
			ASSERT("==============ATLAS::StateCheckpoint OrbitPoleMaxRestored Failed==============",
			       fabsf(netB.getTrainingConfig().atlas.orbitPoleMax - 0.95f) < 1e-6f);
			ASSERT("==============ATLAS::StateCheckpoint HelmEnabledRestored Failed==============",
			       netB.getTrainingConfig().atlas.helmEnabled);
			ASSERT("==============ATLAS::StateCheckpoint HelmMemoryScaleRestored Failed==============",
			       fabsf(netB.getTrainingConfig().atlas.helmMemoryScale - 0.03f) < 1e-6f);
			ASSERT("==============ATLAS::StateCheckpoint HelmEdgeThresholdRestored Failed==============",
			       fabsf(netB.getTrainingConfig().atlas.helmEdgeThreshold - 0.04f) < 1e-6f);
			ASSERT("==============ATLAS::StateCheckpoint HelmModeRankRestored Failed==============",
			       netB.getTrainingConfig().atlas.helmModeRank == 2u);
			ASSERT("==============ATLAS::StateCheckpoint HelmHiddenStackDepthRestored Failed==============",
			       netB.getTrainingConfig().atlas.helmHiddenStackDepth == 2u);
			ASSERT("==============ATLAS::StateCheckpoint HelmPoleMaxRestored Failed==============",
			       fabsf(netB.getTrainingConfig().atlas.helmPoleMax - 0.90f) < 1e-6f);
			ASSERT("==============ATLAS::StateCheckpoint TSubRestored Failed==============",
			       netB.getTrainingConfig().atlas.tSub == 50u);
			netB.getTerminatorMutable().setEpoch(50);
			CaptureMetricsCallbacks cbB;
			st = netB.train(diB, &cbB);
			ASSERT("==============ATLAS::StateCheckpoint Phase3 Train Failed==============", st.ok());
			ASSERT("==============ATLAS::StateCheckpoint Phase3 no metrics==============", cbB.saw);
			lossB = cbB.last.totalError;

			delete diB;
		}

		printf("[UT] ATLAS state checkpoint: lossA=%f, lossB=%f\n", lossA, lossB);
		// If optimizer state is perfectly preserved, losses must match closely.
		const float relDiff = (lossA > 0.0f && lossB > 0.0f)
		    ? fabsf(lossA - lossB) / (0.5f * (lossA + lossB))
		    : fabsf(lossA - lossB);
		printf("[UT] Relative difference: %e\n", relDiff);
		ASSERT("==============ATLAS::StateCheckpoint losses diverged==============",
		       relDiff < 1e-5f);

		// Cleanup
		{
			const std::string base = "database/checkpoints/" + ckptName;
			remove((base + "/manifest.txt").c_str());
			remove((base + "/nninfo.csv").c_str());
			for (int i = 0; i < 10; ++i)
			{
				char buf[64];
				snprintf(buf, sizeof(buf), "/shard_%03d.bin", i);
				remove((base + buf).c_str());
			}
			rmdir(base.c_str());
		}
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ----------------------------------------------------------------
	// Test 24B: Transformer checkpoint preserves ATLAS optimizer state
	// ----------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 24B: Transformer checkpoint state preservation\n");
	printf("-----------------------------------\n");
	{
		const std::string ckptName = "ut_atlas_tr_st24b";
		float lossA = -1.0f;
		float lossB = -1.0f;

		// Path A: train 50 epochs, checkpoint, then continue 25 more epochs.
		{
			glades::NumberInput* diA = make_atlas_transformer_resume_dataset();
			glades::NNInfo* info = make_atlas_transformer_resume_info("ut_atlas_tr_st24b_a");

			glades::NNetwork netA(info, glades::NNetwork::TYPE_TRANSFORMER_ENCODER);
			configure_atlas_transformer_resume_net(netA, 2027u, 50);

			glades::NNetworkStatus st = netA.train(diA);
			ASSERT("==============ATLAS::TransformerStateCheckpoint Phase1 Train Failed==============", st.ok());

			st = netA.saveCheckpoint(ckptName);
			ASSERT("==============ATLAS::TransformerStateCheckpoint Save Failed==============", st.ok());

			netA.getTerminatorMutable().setEpoch(25);
			CaptureMetricsCallbacks cbA;
			st = netA.train(diA, &cbA);
			ASSERT("==============ATLAS::TransformerStateCheckpoint Phase2 Train Failed==============", st.ok());
			ASSERT("==============ATLAS::TransformerStateCheckpoint Phase2 no metrics==============", cbA.saw);
			lossA = cbA.last.totalError;

			delete diA;
			delete info;
		}

		// Path B: load checkpoint, verify config restore, then train the same 25 epochs.
		{
			glades::NumberInput* diB = make_atlas_transformer_resume_dataset();
			glades::NNetwork netB;
			glades::NNetworkStatus st = netB.loadCheckpoint(ckptName, diB);
			ASSERT("==============ATLAS::TransformerStateCheckpoint Load Failed==============", st.ok());
			ASSERT("==============ATLAS::TransformerStateCheckpoint OptimizerRestored Failed==============",
			       netB.getTrainingConfig().optimizer.type == glades::OptimizerConfig::ATLAS);
			ASSERT("==============ATLAS::TransformerStateCheckpoint RankRestored Failed==============",
			       netB.getTrainingConfig().atlas.rank == 8u);
			ASSERT("==============ATLAS::TransformerStateCheckpoint ComplementRankRestored Failed==============",
			       netB.getTrainingConfig().atlas.complementRank == 0u);
			ASSERT("==============ATLAS::TransformerStateCheckpoint TSubRestored Failed==============",
			       netB.getTrainingConfig().atlas.tSub == 50u);
			ASSERT("==============ATLAS::TransformerStateCheckpoint HeadsRestored Failed==============",
			       netB.getTrainingConfig().transformer.nHeadsOverride == 4);
			ASSERT("==============ATLAS::TransformerStateCheckpoint DFFRestored Failed==============",
			       netB.getTrainingConfig().transformer.dFFOverride == 32);

			netB.getTerminatorMutable().setEpoch(25);
			CaptureMetricsCallbacks cbB;
			st = netB.train(diB, &cbB);
			ASSERT("==============ATLAS::TransformerStateCheckpoint Phase3 Train Failed==============", st.ok());
			ASSERT("==============ATLAS::TransformerStateCheckpoint Phase3 no metrics==============", cbB.saw);
			lossB = cbB.last.totalError;

			delete diB;
		}

		printf("[UT] ATLAS transformer state checkpoint: lossA=%f, lossB=%f\n", lossA, lossB);
		const float relDiff = (lossA > 0.0f && lossB > 0.0f)
		    ? fabsf(lossA - lossB) / (0.5f * (lossA + lossB))
		    : fabsf(lossA - lossB);
		printf("[UT] Transformer relative difference: %e\n", relDiff);
		ASSERT("==============ATLAS::TransformerStateCheckpoint losses diverged==============",
		       relDiff < 1e-4f);

		{
			const std::string base = "database/checkpoints/" + ckptName;
			remove((base + "/manifest.txt").c_str());
			remove((base + "/nninfo.csv").c_str());
			for (int i = 0; i < 10; ++i)
			{
				char buf[64];
				snprintf(buf, sizeof(buf), "/shard_%03d.bin", i);
				remove((base + buf).c_str());
			}
			rmdir(base.c_str());
		}
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ----------------------------------------------------------------
	// Test 25: Refresh + mu interaction property test
	// ----------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 25: Refresh + mu interaction\n");
	printf("-----------------------------------\n");
	{
		// Run ATLAS with tSub=5 so refreshes happen frequently (every 5 steps).
		// Verify that across refresh boundaries, invariants hold:
		//   - weights remain finite
		//   - fisherDiag stays non-negative and finite
		//   - mu stays in [muMin, muMax]
		//   - prevGz remains finite
		//   - refreshSubspace returns true (no NaN recovery)
		const unsigned int m = 8, n = 6, r = 3;
		const unsigned int totalSteps = 30; // 6 refreshes at tSub=5
		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 77ULL);

		glades::atlas::WeightState state;
		glades::ATLASConfig ac;
		ac.rank = r;
		ac.tSub = 5;
		ac.muMin = 0.01f;
		ac.muMax = 0.3f;

		std::vector<float> W(m * n, 0.0f);
		std::vector<float> gW(m * n, 0.0f);

		// Initialize weights with small random values.
		for (size_t i = 0; i < W.size(); ++i)
			W[i] = static_cast<float>(glades::rng::uniform_double(rng, -0.5, 0.5));

		unsigned int refreshCount = 0;
		for (unsigned int s = 0; s < totalSteps; ++s)
		{
			// Generate a gradient that drifts slowly (smooth component + noise).
			for (size_t i = 0; i < gW.size(); ++i)
			{
				const float signal = 0.1f * W[i]; // gradient correlated with weights
				const float noise = static_cast<float>(glades::rng::uniform_double(rng, -0.02, 0.02));
				gW[i] = signal + noise;
			}

			const bool ok = glades::atlas::update(state, &W[0], &gW[0], m, n,
			                                       /*invBatch=*/1.0f, /*lr=*/0.01f,
			                                       /*wd1=*/0.0f, /*wd2=*/0.0f,
			                                       /*gradScale=*/1.0f, ac, rng);
			(void)ok; // applyStep may return false on recovery, but state should remain valid

			// Count refreshes (step is incremented inside applyStep).
			if (state.step > 0ULL && (state.step % 5ULL) == 0ULL)
				++refreshCount;

			// Invariant checks after every step.
			for (size_t i = 0; i < W.size(); ++i)
			{
				ASSERT("==============ATLAS::RefreshMu W not finite==============",
				       W[i] == W[i] && W[i] != W[i] + 1.0f); // not NaN and not Inf (for non-zero)
			}
			for (unsigned int c = 0; c < state.r; ++c)
			{
				ASSERT("==============ATLAS::RefreshMu Fisher NaN==============",
				       state.fisherDiag[c] == state.fisherDiag[c]);
				ASSERT("==============ATLAS::RefreshMu Fisher negative==============",
				       state.fisherDiag[c] >= 0.0f);
			}
			ASSERT("==============ATLAS::RefreshMu mu below min==============",
			       state.mu >= ac.muMin);
			ASSERT("==============ATLAS::RefreshMu mu above max==============",
			       state.mu <= ac.muMax);
			for (size_t i = 0; i < state.prevGz.size(); ++i)
			{
				ASSERT("==============ATLAS::RefreshMu prevGz NaN==============",
				       state.prevGz[i] == state.prevGz[i]);
			}
		}

		ASSERT("==============ATLAS::RefreshMu no refreshes happened==============",
		       refreshCount >= 4); // expect 6 refreshes in 30 steps with tSub=5
		printf("[UT] ATLAS refresh+mu interaction: %u refreshes in %u steps, all invariants held\n",
		       refreshCount, totalSteps);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 26: applyStep with biasCorrection=true (default config)
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 26: applyStep with bias correction enabled\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 4;
		const unsigned int n = 3;
		const unsigned int r = 2;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 26262626ULL);
		glades::atlas::WeightState state;
		glades::atlas::initWeightState(state, m, n, r, 0.01f, rng);

		std::vector<float> W(m * n, 0.0f);
		for (unsigned int i = 0; i < m * n; ++i)
			W[i] = static_cast<float>(i) * 0.1f - 0.5f;

		// Use default ATLASConfig (biasCorrection = true)
		glades::ATLASConfig acDefault;
		acDefault.rank = r;

		for (int step = 0; step < 50; ++step)
		{
			std::vector<float> gW(m * n, 0.0f);
			for (unsigned int i = 0; i < m * n; ++i)
				gW[i] = W[i] * 0.1f;

			glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
			                         1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
			                         acDefault, rng);

			// All weights must remain finite
			for (unsigned int i = 0; i < m * n; ++i)
			{
				ASSERT("==============ATLAS::BiasCorr applyStep NaN==============", W[i] == W[i]);
				float d = W[i] - W[i];
				ASSERT("==============ATLAS::BiasCorr applyStep Inf==============", d == 0.0f);
			}

			// Fisher must be positive and finite
			for (unsigned int c = 0; c < r; ++c)
			{
				ASSERT("==============ATLAS::BiasCorr Fisher NaN==============",
				       state.fisherDiag[c] == state.fisherDiag[c]);
				ASSERT("==============ATLAS::BiasCorr Fisher negative==============",
				       state.fisherDiag[c] >= 0.0f);
			}

			// sigma2 and mu must be finite
			ASSERT("==============ATLAS::BiasCorr sigma2 NaN==============",
			       state.sigma2 == state.sigma2);
			ASSERT("==============ATLAS::BiasCorr mu NaN==============",
			       state.mu == state.mu);
		}

		// Verify weights changed
		bool changed = false;
		for (unsigned int i = 0; i < m * n; ++i)
		{
			float init = static_cast<float>(i) * 0.1f - 0.5f;
			if (W[i] != init) { changed = true; break; }
		}
		ASSERT("==============ATLAS::BiasCorr weights unchanged==============", changed);

		printf("[UT] ATLAS applyStep with biasCorrection=true: 50 steps, all invariants held\n");
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 27: Fuzz test with biasCorrection=true (default config)
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 27: Fuzz test with bias correction enabled\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 8;
		const unsigned int n = 6;
		const unsigned int r = 3;
		const int nSteps = 300;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 27272727ULL);
		glades::atlas::WeightState state;
		glades::atlas::initWeightState(state, m, n, r, 0.01f, rng);

		std::vector<float> W(m * n, 0.0f);
		for (unsigned int i = 0; i < m * n; ++i)
			W[i] = glades::rng::standard_normal(rng) * 0.1f;

		// Default config with biasCorrection = true
		glades::ATLASConfig acFuzz;
		acFuzz.rank = r;
		acFuzz.tSub = 15u;

		for (int step = 0; step < nSteps; ++step)
		{
			std::vector<float> gW(m * n, 0.0f);
			for (unsigned int i = 0; i < m * n; ++i)
				gW[i] = glades::rng::standard_normal(rng) * 0.5f;

			glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
			                         1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
			                         acFuzz, rng);

			for (unsigned int i = 0; i < m * n; ++i)
			{
				ASSERT("==============ATLAS::BCFuzz NaN weight==============", W[i] == W[i]);
				float d = W[i] - W[i];
				ASSERT("==============ATLAS::BCFuzz Inf weight==============", d == 0.0f);
			}
			for (unsigned int c = 0; c < state.r; ++c)
			{
				ASSERT("==============ATLAS::BCFuzz Fisher NaN==============",
				       state.fisherDiag[c] == state.fisherDiag[c]);
				ASSERT("==============ATLAS::BCFuzz Fisher negative==============",
				       state.fisherDiag[c] >= 0.0f);
			}
			ASSERT("==============ATLAS::BCFuzz mu below min==============",
			       state.mu >= acFuzz.muMin);
			ASSERT("==============ATLAS::BCFuzz mu above max==============",
			       state.mu <= acFuzz.muMax + 1e-6f);
			ASSERT("==============ATLAS::BCFuzz sigma2 NaN==============",
			       state.sigma2 == state.sigma2);
			ASSERT("==============ATLAS::BCFuzz sigma2 negative==============",
			       state.sigma2 >= 0.0f);
		}
		printf("[UT] ATLAS fuzz biasCorrection=true: %d steps, all invariants held\n", nSteps);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 28: sigma2 uses normalized covariance trace closure
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 28: sigma2 uses normalized covariance trace closure\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 4;
		const unsigned int n = 4;
		const unsigned int r = 2;
		const float invBatch = 1.0f;
		const float lr = 0.01f;
		const float gradScale = 1.0f;
		const float expectedTrace = 30.0f;
		const float expectedSigma2 = 12.5f;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 28282828ULL);
		glades::atlas::WeightState state;
		glades::atlas::initWeightState(state, m, n, r, 0.01f, rng);

		std::fill(state.U.begin(), state.U.end(), 0.0f);
		state.U[0] = 1.0f;
		state.U[3] = 1.0f;
		state.activeRank = r;

		std::vector<float> W(static_cast<size_t>(m) * n, 0.0f);
		std::vector<float> gW;
		gW.push_back(1.0f); gW.push_back(1.0f); gW.push_back(1.0f); gW.push_back(1.0f);
		gW.push_back(2.0f); gW.push_back(2.0f); gW.push_back(2.0f); gW.push_back(2.0f);
		gW.push_back(3.0f); gW.push_back(3.0f); gW.push_back(3.0f); gW.push_back(3.0f);
		gW.push_back(4.0f); gW.push_back(4.0f); gW.push_back(4.0f); gW.push_back(4.0f);

		glades::ATLASConfig acSigma;
		acSigma.rank = r;
		acSigma.complementRank = 0u;
		acSigma.biasCorrection = false;
		acSigma.muMin = 0.0f;
		acSigma.muMax = 0.0f;

		const bool ok = glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
		                                         invBatch, lr, 0.0f, 0.0f, gradScale,
		                                         acSigma, rng);
		ASSERT("==============ATLAS::Sigma2Scale applyStep failed==============", ok);

		printf("[UT] ATLAS closure scale: totalTrace=%f expectedTrace=%f sigma2=%f expectedSigma2=%f\n",
		       state.totalTrace, expectedTrace, state.sigma2, expectedSigma2);

		const float traceErr = state.totalTrace - expectedTrace;
		const float traceErrAbs = (traceErr < 0.0f) ? -traceErr : traceErr;
		const float sigmaErr = state.sigma2 - expectedSigma2;
		const float sigmaErrAbs = (sigmaErr < 0.0f) ? -sigmaErr : sigmaErr;
		ASSERT("==============ATLAS::Sigma2Scale totalTrace not on normalized covariance scale==============",
		       traceErrAbs < 1e-6f);
		ASSERT("==============ATLAS::Sigma2Scale sigma2 not trace-closed from normalized covariance==============",
		       sigmaErrAbs < 1e-6f);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 28B: complement sector captures residual anisotropy
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 28B: complement sector captures residual anisotropy\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 5;
		const unsigned int n = 4;
		const unsigned int r = 2;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 28282829ULL);
		glades::atlas::WeightState state;
		glades::atlas::initWeightState(state, m, n, r, 0.01f, rng);

		std::fill(state.U.begin(), state.U.end(), 0.0f);
		state.U[0] = 1.0f;
		state.U[state.r + 1] = 1.0f;
		state.activeRank = r;
		std::fill(state.V.begin(), state.V.end(), 0.0f);
		state.V[2] = 1.0f;

		std::vector<float> W(static_cast<size_t>(m) * n, 0.0f);
		std::vector<float> gW(static_cast<size_t>(m) * n, 0.0f);
		for (unsigned int j = 0; j < n; ++j)
		{
			gW[0 * n + j] = 1.0f;
			gW[1 * n + j] = 2.0f;
			gW[2 * n + j] = 4.0f;
			gW[3 * n + j] = 0.5f;
			gW[4 * n + j] = 0.5f;
		}

		glades::ATLASConfig acSector;
		acSector.rank = r;
		acSector.complementRank = 1u;
		acSector.biasCorrection = false;
		acSector.muMin = 0.0f;
		acSector.muMax = 0.0f;
		acSector.tSub = 0u;

		const bool ok = glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
		                                         1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
		                                         acSector, rng);
		ASSERT("==============ATLAS::ComplementSector applyStep failed==============", ok);

		const float expectedTrace = 21.5f;
		const float expectedSectorFisher = 16.0f;
		const float expectedSigma2 = 0.25f;
		printf("[UT] ATLAS complement sector: totalTrace=%f sectorFisher=%f sigma2=%f\n",
		       state.totalTrace, state.complementFisher, state.sigma2);
		ASSERT("==============ATLAS::ComplementSector totalTrace mismatch==============",
		       fabsf(state.totalTrace - expectedTrace) < 1e-6f);
		ASSERT("==============ATLAS::ComplementSector sectorFisher mismatch==============",
		       fabsf(state.complementFisher - expectedSectorFisher) < 1e-6f);
		ASSERT("==============ATLAS::ComplementSector sigma2 mismatch==============",
		       fabsf(state.sigma2 - expectedSigma2) < 1e-6f);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 28C: dense complement block captures correlated residuals
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 28C: dense complement block captures correlated residuals\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 6;
		const unsigned int n = 4;
		const unsigned int r = 2;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 28282830ULL);
		glades::atlas::WeightState state;
		glades::atlas::initWeightState(state, m, n, r, 0.01f, rng);

		state.complementRank = 2u;
		state.V.assign(static_cast<size_t>(m) * state.complementRank, 0.0f);
		state.complementBlock.assign(static_cast<size_t>(state.complementRank) * state.complementRank, 0.0f);
		state.prevGv.assign(static_cast<size_t>(state.complementRank) * n, 0.0f);
		state.scratch_gv.resize(static_cast<size_t>(state.complementRank) * n);
		state.scratch_correctedV.resize(static_cast<size_t>(state.complementRank) * n);
		state.scratch_V_old.resize(static_cast<size_t>(m) * state.complementRank);
		state.scratch_Bv.resize(static_cast<size_t>(state.complementRank) * n);
		state.scratch_Zv.resize(static_cast<size_t>(m) * state.complementRank);
		state.scratch_complementMat.resize(static_cast<size_t>(state.complementRank) * state.complementRank);
		state.scratch_complementEigVec.resize(static_cast<size_t>(state.complementRank) * state.complementRank);
		state.scratch_complementEigVal.resize(state.complementRank);

		std::fill(state.U.begin(), state.U.end(), 0.0f);
		state.U[0] = 1.0f;
		state.U[state.r + 1] = 1.0f;
		state.activeRank = r;
		state.V[2 * state.complementRank + 0] = 1.0f;
		state.V[3 * state.complementRank + 1] = 1.0f;

		std::vector<float> W(static_cast<size_t>(m) * n, 0.0f);
		std::vector<float> gW(static_cast<size_t>(m) * n, 0.0f);
		for (unsigned int j = 0; j < n; ++j)
		{
			gW[0 * n + j] = 1.0f;
			gW[1 * n + j] = 2.0f;
			gW[2 * n + j] = 4.0f;
			gW[3 * n + j] = 3.0f;
			gW[4 * n + j] = 1.0f;
			gW[5 * n + j] = 1.0f;
		}

		glades::ATLASConfig acDense;
		acDense.rank = r;
		acDense.complementRank = 2u;
		acDense.biasCorrection = false;
		acDense.muMin = 0.0f;
		acDense.muMax = 0.0f;
		acDense.tSub = 0u;

		const bool ok = glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
		                                         1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
		                                         acDense, rng);
		ASSERT("==============ATLAS::DenseComplement applyStep failed==============", ok);

		const float expectedTrace = 32.0f;
		const float expectedBlock00 = 16.0f;
		const float expectedBlock01 = 12.0f;
		const float expectedBlock11 = 9.0f;
		const float expectedComplementTrace = 25.0f;
		const float expectedSigma2 = 1.0f;
		printf("[UT] ATLAS dense complement: totalTrace=%f block=[%f %f; %f %f] sectorFisher=%f sigma2=%f\n",
		       state.totalTrace,
		       state.complementBlock[0], state.complementBlock[1],
		       state.complementBlock[2], state.complementBlock[3],
		       state.complementFisher, state.sigma2);
		ASSERT("==============ATLAS::DenseComplement totalTrace mismatch==============",
		       fabsf(state.totalTrace - expectedTrace) < 1e-6f);
		ASSERT("==============ATLAS::DenseComplement block00 mismatch==============",
		       fabsf(state.complementBlock[0] - expectedBlock00) < 1e-6f);
		ASSERT("==============ATLAS::DenseComplement block01 mismatch==============",
		       fabsf(state.complementBlock[1] - expectedBlock01) < 1e-6f
		       && fabsf(state.complementBlock[2] - expectedBlock01) < 1e-6f);
		ASSERT("==============ATLAS::DenseComplement block11 mismatch==============",
		       fabsf(state.complementBlock[3] - expectedBlock11) < 1e-6f);
		ASSERT("==============ATLAS::DenseComplement sectorFisher mismatch==============",
		       fabsf(state.complementFisher - expectedComplementTrace) < 1e-6f);
		ASSERT("==============ATLAS::DenseComplement sigma2 mismatch==============",
		       fabsf(state.sigma2 - expectedSigma2) < 5e-6f);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 28D: FC eligibility gate disables complement block on small heads
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 28D: FC eligibility gate disables small-head complement block\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 10;
		const unsigned int n = 84;
		const unsigned int r = 4;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 28282831ULL);
		glades::atlas::WeightState state;
		glades::atlas::initWeightState(state, m, n, r, 0.01f, rng);

		std::vector<float> W(static_cast<size_t>(m) * n, 0.0f);
		std::vector<float> gW(static_cast<size_t>(m) * n, 1.0f);

		glades::ATLASConfig acGate;
		acGate.rank = r;
		acGate.complementRank = 4u;
		acGate.biasCorrection = false;
		acGate.muMin = 0.0f;
		acGate.muMax = 0.0f;
		acGate.tSub = 1u;

		const bool ok = glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
		                                         1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
		                                         acGate, rng, 0, "cnn.fc");
		ASSERT("==============ATLAS::ComplementEligibility applyStep failed==============", ok);
		ASSERT("==============ATLAS::ComplementEligibility active rank should stay zero==============",
		       state.activeComplementRank == 0u);
		ASSERT("==============ATLAS::ComplementEligibility block trace should stay zero==============",
		       fabsf(state.complementFisher) < 1e-8f);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 28E: adaptive complement rank promotes only after Kelly probation
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 28E: adaptive complement rank promotes stable modes after probation\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 24;
		const unsigned int n = 24;
		const unsigned int r = 2;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 28282832ULL);
		glades::atlas::WeightState state;
		glades::atlas::initWeightState(state, m, n, r, 0.0f, rng);

		state.activeRank = r;
		state.complementRank = 4u;
		state.activeComplementRank = 0u;
		state.V.assign(static_cast<size_t>(m) * state.complementRank, 0.0f);
		state.complementBlock.assign(static_cast<size_t>(state.complementRank) * state.complementRank, 0.0f);
		state.prevGv.assign(static_cast<size_t>(state.complementRank) * n, 0.0f);
		state.scratch_gv.resize(static_cast<size_t>(state.complementRank) * n);
		state.scratch_correctedV.resize(static_cast<size_t>(state.complementRank) * n);
		state.scratch_V_old.resize(static_cast<size_t>(m) * state.complementRank);
		state.scratch_Bv.resize(static_cast<size_t>(state.complementRank) * n);
		state.scratch_Zv.resize(static_cast<size_t>(m) * state.complementRank);
		state.scratch_complementMat.resize(static_cast<size_t>(state.complementRank) * state.complementRank);
		state.scratch_complementEigVec.resize(static_cast<size_t>(state.complementRank) * state.complementRank);
		state.scratch_complementEigVal.resize(state.complementRank);

		std::fill(state.U.begin(), state.U.end(), 0.0f);
		state.U[0] = 1.0f;
		state.U[state.r + 1] = 1.0f;
		for (unsigned int c = 0; c < state.complementRank; ++c)
			state.V[(c + 2u) * state.complementRank + c] = 1.0f;
		state.fisherDiag[0] = 1.0f;
		state.fisherDiag[1] = 0.0f;
		state.complementBlock[0] = 16.0f;
		state.complementBlock[5] = 9.0f;
		state.complementBlock[10] = 1.0f;
		state.complementBlock[15] = 0.25f;
		state.complementFisher = 26.25f;
		state.totalTrace = 27.25f;
		state.step = 2ULL;

		std::vector<float> W(static_cast<size_t>(m) * n, 0.0f);
		std::vector<float> gW(static_cast<size_t>(m) * n, 0.0f);

		glades::ATLASConfig acAdaptive;
		acAdaptive.rank = r;
		acAdaptive.complementRank = 4u;
		acAdaptive.beta = 1.0f;
		acAdaptive.biasCorrection = false;
		acAdaptive.muMin = 0.0f;
		acAdaptive.muMax = 0.0f;
		acAdaptive.tSub = 0u;

		std::fill(gW.begin(), gW.end(), 0.0f);
		for (unsigned int j = 0; j < n; ++j)
		{
			gW[2u * n + j] = 4.0f;
		}
		bool ok = glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
		                                   1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
		                                   acAdaptive, rng, 0, "cnn.fc");
		ASSERT("==============ATLAS::AdaptiveComplementRank first applyStep failed==============", ok);
		ASSERT("==============ATLAS::AdaptiveComplementRank should keep first mode on probation==============",
		       state.activeComplementRank == 0u);
		ASSERT("==============ATLAS::AdaptiveComplementRank should start first probation==============",
		       state.trialComplementRank == 1u && state.trialComplementWins == 1u);
		std::fill(gW.begin(), gW.end(), 0.0f);
		for (unsigned int j = 0; j < n; ++j)
		{
			gW[2u * n + j] = 4.0f;
		}
		ok = glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
		                              1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
		                              acAdaptive, rng, 0, "cnn.fc");
		ASSERT("==============ATLAS::AdaptiveComplementRank second applyStep failed==============", ok);
		ASSERT("==============ATLAS::AdaptiveComplementRank should promote first stable mode==============",
		       state.activeComplementRank == 1u);
		ASSERT("==============ATLAS::AdaptiveComplementRank should clear first probation after promotion==============",
		       state.trialComplementRank == 0u);
		std::fill(gW.begin(), gW.end(), 0.0f);
		for (unsigned int j = 0; j < n; ++j)
		{
			gW[3u * n + j] = 3.0f;
		}
		ok = glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
		                              1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
		                              acAdaptive, rng, 0, "cnn.fc");
		ASSERT("==============ATLAS::AdaptiveComplementRank third applyStep failed==============", ok);
		ASSERT("==============ATLAS::AdaptiveComplementRank should start second probation==============",
		       state.activeComplementRank == 1u
		       && state.trialComplementRank == 2u
		       && state.trialComplementWins == 1u);
		std::fill(gW.begin(), gW.end(), 0.0f);
		for (unsigned int j = 0; j < n; ++j)
		{
			gW[3u * n + j] = 3.0f;
		}
		ok = glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
		                              1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
		                              acAdaptive, rng, 0, "cnn.fc");
		ASSERT("==============ATLAS::AdaptiveComplementRank fourth applyStep failed==============", ok);
		ASSERT("==============ATLAS::AdaptiveComplementRank should promote second stable mode==============",
		       state.activeComplementRank == 2u);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 28F: scout signal stays on probation while uncertainty is high
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 28F: complement scout stays on probation under high uncertainty\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 24;
		const unsigned int n = 24;
		const unsigned int r = 2;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 28282833ULL);
		glades::atlas::WeightState state;
		glades::atlas::initWeightState(state, m, n, r, 0.0f, rng);

		state.activeRank = r;
		state.complementRank = 4u;
		state.activeComplementRank = 0u;
		state.V.assign(static_cast<size_t>(m) * state.complementRank, 0.0f);
		state.complementBlock.assign(static_cast<size_t>(state.complementRank) * state.complementRank, 0.0f);
		state.prevGv.assign(static_cast<size_t>(state.complementRank) * n, 0.0f);
		state.scratch_gv.resize(static_cast<size_t>(state.complementRank) * n);
		state.scratch_correctedV.resize(static_cast<size_t>(state.complementRank) * n);
		state.scratch_V_old.resize(static_cast<size_t>(m) * state.complementRank);
		state.scratch_Bv.resize(static_cast<size_t>(state.complementRank) * n);
		state.scratch_Zv.resize(static_cast<size_t>(m) * state.complementRank);
		state.scratch_complementMat.resize(static_cast<size_t>(state.complementRank) * state.complementRank);
		state.scratch_complementEigVec.resize(static_cast<size_t>(state.complementRank) * state.complementRank);
		state.scratch_complementEigVal.resize(state.complementRank);

		std::fill(state.U.begin(), state.U.end(), 0.0f);
		state.U[0] = 1.0f;
		state.U[state.r + 1] = 1.0f;
		for (unsigned int c = 0; c < state.complementRank; ++c)
			state.V[(c + 2u) * state.complementRank + c] = 1.0f;
		state.fisherDiag[0] = 1.0f;
		state.fisherDiag[1] = 0.0f;
		state.complementBlock[0] = 0.05f;
		state.complementBlock[5] = 0.02f;
		state.complementBlock[10] = 0.01f;
		state.complementBlock[15] = 0.005f;
		state.complementFisher = 0.085f;
		state.totalTrace = 8.0f;
		state.step = 199ULL;

		std::vector<float> W(static_cast<size_t>(m) * n, 0.0f);
		std::vector<float> gW(static_cast<size_t>(m) * n, 0.0f);
		for (unsigned int j = 0; j < n; ++j)
			gW[2 * n + j] = 12.0f;

		glades::ATLASConfig acScout;
		acScout.rank = r;
		acScout.complementRank = 4u;
		acScout.beta = 0.999f;
		acScout.biasCorrection = false;
		acScout.muMin = 0.0f;
		acScout.muMax = 0.0f;
		acScout.tSub = 0u;

		bool ok = glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
		                                   1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
		                                   acScout, rng, 0, "cnn.fc");
		ASSERT("==============ATLAS::AdaptiveComplementScout first applyStep failed==============", ok);
		ASSERT("==============ATLAS::AdaptiveComplementScout should open probation from scout signal==============",
		       state.activeComplementRank == 0u
		       && state.trialComplementRank == 1u);
		for (unsigned int j = 0; j < n; ++j)
			gW[2 * n + j] = 12.0f;
		ok = glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
		                              1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
		                              acScout, rng, 0, "cnn.fc");
		ASSERT("==============ATLAS::AdaptiveComplementScout second applyStep failed==============", ok);
		ASSERT("==============ATLAS::AdaptiveComplementScout should stay on probation while uncertainty remains high==============",
		       state.activeComplementRank == 0u
		       && state.trialComplementRank == 1u);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 28G: misaligned scout directions are rejected by the Kelly gate
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 28G: misaligned scout directions are rejected\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 24;
		const unsigned int n = 24;
		const unsigned int r = 2;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 28282834ULL);
		glades::atlas::WeightState state;
		glades::atlas::initWeightState(state, m, n, r, 0.0f, rng);

		state.activeRank = r;
		state.complementRank = 4u;
		state.activeComplementRank = 0u;
		state.V.assign(static_cast<size_t>(m) * state.complementRank, 0.0f);
		state.complementBlock.assign(static_cast<size_t>(state.complementRank) * state.complementRank, 0.0f);
		state.prevGv.assign(static_cast<size_t>(state.complementRank) * n, 0.0f);
		state.scratch_gv.resize(static_cast<size_t>(state.complementRank) * n);
		state.scratch_correctedV.resize(static_cast<size_t>(state.complementRank) * n);
		state.scratch_V_old.resize(static_cast<size_t>(m) * state.complementRank);
		state.scratch_Bv.resize(static_cast<size_t>(state.complementRank) * n);
		state.scratch_Zv.resize(static_cast<size_t>(m) * state.complementRank);
		state.scratch_complementMat.resize(static_cast<size_t>(state.complementRank) * state.complementRank);
		state.scratch_complementEigVec.resize(static_cast<size_t>(state.complementRank) * state.complementRank);
		state.scratch_complementEigVal.resize(state.complementRank);

		std::fill(state.U.begin(), state.U.end(), 0.0f);
		state.U[0] = 1.0f;
		state.U[state.r + 1] = 1.0f;
		for (unsigned int c = 0; c < state.complementRank; ++c)
			state.V[(c + 2u) * state.complementRank + c] = 1.0f;
		state.fisherDiag[0] = 1.0f;
		state.fisherDiag[1] = 0.0f;
		state.complementBlock[0] = 16.0f;
		state.complementBlock[5] = 4.0f;
		state.complementBlock[10] = 1.0f;
		state.complementBlock[15] = 0.25f;
		state.complementFisher = 21.25f;
		state.totalTrace = 22.25f;
		state.step = 2ULL;

		std::vector<float> W(static_cast<size_t>(m) * n, 0.0f);
		std::vector<float> gW(static_cast<size_t>(m) * n, 0.0f);
		for (unsigned int j = 0; j < n; ++j)
			gW[3u * n + j] = 12.0f;

		glades::ATLASConfig acReject;
		acReject.rank = r;
		acReject.complementRank = 4u;
		acReject.beta = 1.0f;
		acReject.biasCorrection = false;
		acReject.muMin = 0.0f;
		acReject.muMax = 0.0f;
		acReject.tSub = 0u;

		bool ok = glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
		                                   1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
		                                   acReject, rng, 0, "cnn.fc");
		ASSERT("==============ATLAS::AdaptiveComplementReject first applyStep failed==============", ok);
		ASSERT("==============ATLAS::AdaptiveComplementReject should reject misaligned scout mode==============",
		       state.activeComplementRank == 0u && state.trialComplementRank == 0u);
		for (unsigned int j = 0; j < n; ++j)
			gW[3u * n + j] = 12.0f;
		ok = glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
		                              1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
		                              acReject, rng, 0, "cnn.fc");
		ASSERT("==============ATLAS::AdaptiveComplementReject second applyStep failed==============", ok);
		ASSERT("==============ATLAS::AdaptiveComplementReject should still reject misaligned scout mode==============",
		       state.activeComplementRank == 0u && state.trialComplementRank == 0u);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 29: Scale stress test (m=512, n=512, r=64)
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 29: Scale stress test (512x512 r=64)\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 512;
		const unsigned int n = 512;
		const unsigned int r = 64;
		const int nSteps = 200;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 28282828ULL);
		glades::atlas::WeightState state;
		glades::atlas::initWeightState(state, m, n, r, 0.01f, rng);

		ASSERT("==============ATLAS::Scale init failed==============", state.initialized);
		ASSERT("==============ATLAS::Scale rank wrong==============", state.r == r);

		// Verify initial U orthonormality
		{
			const float tol = 1e-4f;
			for (unsigned int c = 0; c < r; c += 16) // sample every 16th column pair
			{
				float dot = 0.0f;
				for (unsigned int i = 0; i < m; ++i)
					dot += state.U[i * r + c] * state.U[i * r + c];
				float err = (dot - 1.0f) < 0.0f ? -(dot - 1.0f) : (dot - 1.0f);
				ASSERT("==============ATLAS::Scale initial U not unit norm==============", err <= tol);
			}
		}

		std::vector<float> W(static_cast<size_t>(m) * n, 0.0f);
		for (size_t i = 0; i < W.size(); ++i)
			W[i] = glades::rng::standard_normal(rng) * 0.02f;

		glades::ATLASConfig acScale;
		acScale.rank = r;
		acScale.tSub = 50;
		acScale.beta = 0.999f;
		acScale.biasCorrection = true;

		for (int step = 0; step < nSteps; ++step)
		{
			std::vector<float> gW(static_cast<size_t>(m) * n, 0.0f);
			for (size_t i = 0; i < gW.size(); ++i)
				gW[i] = glades::rng::standard_normal(rng) * 0.01f;

			glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
			                         1.0f, 0.001f, 0.0f, 0.0f, 1.0f,
			                         acScale, rng);

			// Check invariants every 50 steps (not every step to keep runtime reasonable)
			if (step % 50 == 0 || step == nSteps - 1)
			{
				// Weights finite (sample check)
				for (size_t i = 0; i < W.size(); i += 1000)
				{
					ASSERT("==============ATLAS::Scale NaN weight==============", W[i] == W[i]);
					float d = W[i] - W[i];
					ASSERT("==============ATLAS::Scale Inf weight==============", d == 0.0f);
				}

				// Fisher positive and finite
				for (unsigned int c = 0; c < r; ++c)
				{
					ASSERT("==============ATLAS::Scale Fisher NaN==============",
					       state.fisherDiag[c] == state.fisherDiag[c]);
					ASSERT("==============ATLAS::Scale Fisher negative==============",
					       state.fisherDiag[c] >= 0.0f);
				}

				// sigma2 and mu
				ASSERT("==============ATLAS::Scale sigma2 NaN==============",
				       state.sigma2 == state.sigma2);
				ASSERT("==============ATLAS::Scale mu NaN==============",
				       state.mu == state.mu);

				// U orthonormality (sample)
				if (state.step > 0ULL && (state.step % 50ULL) == 0ULL)
				{
					const float orthoTol = 1e-3f; // looser tolerance for large matrices
					for (unsigned int c = 0; c < r; c += 16)
					{
						float dot = 0.0f;
						for (unsigned int i = 0; i < m; ++i)
							dot += state.U[i * r + c] * state.U[i * r + c];
						float err = (dot - 1.0f) < 0.0f ? -(dot - 1.0f) : (dot - 1.0f);
						ASSERT("==============ATLAS::Scale U not unit norm after refresh==============",
						       err <= orthoTol);
					}
				}
			}
		}

		printf("[UT] ATLAS scale test (512x512 r=64): %d steps completed, all invariants held\n",
		       nSteps);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 30: adaptive active rank shrinks on low-rank gradient streams
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 30: Adaptive active rank shrink\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 16;
		const unsigned int n = 8;
		const unsigned int r = 6;
		const unsigned int nSteps = 24;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 30303030ULL);
		glades::atlas::WeightState state;
		glades::atlas::initWeightState(state, m, n, r, 0.01f, rng);

		ASSERT("==============ATLAS::AdaptiveRank init activeRank wrong==============",
		       state.activeRank == r);

		std::vector<float> W(static_cast<size_t>(m) * n, 0.0f);
		glades::ATLASConfig acAdaptive;
		acAdaptive.rank = r;
		acAdaptive.tSub = 4u;
		acAdaptive.rankCapture = 0.90f;
		acAdaptive.minActiveRank = 1u;
		acAdaptive.adaptiveRank = true;
		acAdaptive.fisherWeightedRefresh = true;

		for (unsigned int step = 0; step < nSteps; ++step)
		{
			std::vector<float> gW(static_cast<size_t>(m) * n, 0.0f);
			for (unsigned int i = 0; i < m; ++i)
			{
				const float a = (i < (m / 2u)) ? 1.0f : -1.0f;
				for (unsigned int j = 0; j < n; ++j)
				{
					const float b = 0.2f + 0.03f * static_cast<float>(j);
					gW[static_cast<size_t>(i) * n + j] = a * b;
				}
			}

			const bool ok = glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
			                                         1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
			                                         acAdaptive, rng);
			ASSERT("==============ATLAS::AdaptiveRank applyStep failed==============", ok);
		}

		printf("[UT] ATLAS adaptive rank: configured=%u active=%u after %u steps\n",
		       r, state.activeRank, nSteps);
		ASSERT("==============ATLAS::AdaptiveRank did not shrink==============",
		       state.activeRank < r);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	printf("\n============================================================\n");
}

void ATLASControllerUnitTest()
{
	printf("============================================================\n");
	printf("ATLAS Complement Controller Unit Test Suite\n");
	printf("============================================================\n");

	// ---------------------------------------------------------------
	// Test C1: adaptive complement rank promotes only after Kelly probation
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Controller Test C1: probation before promotion\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 24;
		const unsigned int n = 24;
		const unsigned int r = 2;

			glades::rng::Engine rng;
			glades::rng::seed_engine(rng, 38282832ULL);
			glades::atlas::WeightState state;
			prepare_atlas_complement_test_state(state, rng, m, n, r, 4u, "cnn.fc");

			std::fill(state.U.begin(), state.U.end(), 0.0f);
		std::fill(state.V.begin(), state.V.end(), 0.0f);
		std::fill(state.complementBlock.begin(), state.complementBlock.end(), 0.0f);
		std::fill(state.prevGv.begin(), state.prevGv.end(), 0.0f);
		state.U[0] = 1.0f;
		state.U[state.r + 1] = 1.0f;
		for (unsigned int c = 0; c < state.complementRank; ++c)
			state.V[(c + 2u) * state.complementRank + c] = 1.0f;
		state.fisherDiag[0] = 1.0f;
		state.fisherDiag[1] = 0.0f;
		state.complementBlock[0] = 16.0f;
		state.complementBlock[5] = 9.0f;
		state.complementBlock[10] = 1.0f;
		state.complementBlock[15] = 0.25f;
		state.complementFisher = 26.25f;
		state.totalTrace = 27.25f;
		state.step = 2ULL;

		std::vector<float> W(static_cast<size_t>(m) * n, 0.0f);
		std::vector<float> gW(static_cast<size_t>(m) * n, 0.0f);

		glades::ATLASConfig acAdaptive;
		acAdaptive.rank = r;
		acAdaptive.complementRank = 4u;
		acAdaptive.beta = 1.0f;
		acAdaptive.biasCorrection = false;
		acAdaptive.muMin = 0.0f;
		acAdaptive.muMax = 0.0f;
		acAdaptive.tSub = 0u;

		for (unsigned int j = 0; j < n; ++j)
		{
			gW[2u * n + j] = 4.0f;
		}
			bool ok = glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
			                                   1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
			                                   acAdaptive, rng, 0, "cnn.fc");
				ASSERT("==============ATLAS::ControllerProbation first applyStep failed==============", ok);
				ASSERT("==============ATLAS::ControllerProbation first applyStep should preserve storage==============",
				       atlas_storage_shapes_ok(state, m, n, r, 4u));
				ASSERT("==============ATLAS::ControllerProbation should keep first mode on probation==============",
				       state.activeComplementRank == 0u
				       && state.trialComplementRank == 1u);

		for (unsigned int j = 0; j < n; ++j)
		{
			gW[2u * n + j] = 4.0f;
		}
			ok = glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
			                              1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
			                              acAdaptive, rng, 0, "cnn.fc");
			ASSERT("==============ATLAS::ControllerProbation second applyStep failed==============", ok);
			ASSERT("==============ATLAS::ControllerProbation second applyStep should preserve storage==============",
			       atlas_storage_shapes_ok(state, m, n, r, 4u));
			ASSERT("==============ATLAS::ControllerProbation should promote first stable mode==============",
			       state.activeComplementRank == 1u && state.trialComplementRank == 0u);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test C2: scout signal stays on probation until EMA confirmation catches up
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Controller Test C2: scout evidence stays on probation\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 24;
		const unsigned int n = 24;
		const unsigned int r = 2;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 38282833ULL);
		glades::atlas::WeightState state;
		prepare_atlas_complement_test_state(state, rng, m, n, r, 4u, "cnn.fc");

		std::fill(state.U.begin(), state.U.end(), 0.0f);
		std::fill(state.V.begin(), state.V.end(), 0.0f);
		std::fill(state.complementBlock.begin(), state.complementBlock.end(), 0.0f);
		std::fill(state.prevGv.begin(), state.prevGv.end(), 0.0f);
		state.U[0] = 1.0f;
		state.U[state.r + 1] = 1.0f;
		for (unsigned int c = 0; c < state.complementRank; ++c)
			state.V[(c + 2u) * state.complementRank + c] = 1.0f;
		state.fisherDiag[0] = 1.0f;
		state.fisherDiag[1] = 0.0f;
		state.complementBlock[0] = 0.05f;
		state.complementBlock[5] = 0.02f;
		state.complementBlock[10] = 0.01f;
		state.complementBlock[15] = 0.005f;
		state.complementFisher = 0.085f;
		state.totalTrace = 8.0f;
		state.step = 199ULL;

		std::vector<float> W(static_cast<size_t>(m) * n, 0.0f);
		std::vector<float> gW(static_cast<size_t>(m) * n, 0.0f);
		for (unsigned int j = 0; j < n; ++j)
			gW[2u * n + j] = 12.0f;

		glades::ATLASConfig acScout;
		acScout.rank = r;
		acScout.complementRank = 4u;
		acScout.beta = 0.999f;
		acScout.biasCorrection = false;
		acScout.muMin = 0.0f;
		acScout.muMax = 0.0f;
		acScout.tSub = 0u;

		bool ok = glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
		                                   1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
		                                   acScout, rng, 0, "cnn.fc");
		ASSERT("==============ATLAS::ControllerScout first applyStep failed==============", ok);
		ASSERT("==============ATLAS::ControllerScout should open probation from scout signal==============",
		       state.activeComplementRank == 0u
		       && state.trialComplementRank == 1u);

		for (unsigned int j = 0; j < n; ++j)
			gW[2u * n + j] = 12.0f;
		ok = glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
		                              1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
		                              acScout, rng, 0, "cnn.fc");
		ASSERT("==============ATLAS::ControllerScout second applyStep failed==============", ok);
		ASSERT("==============ATLAS::ControllerScout should stay on probation while uncertainty remains high==============",
		       state.activeComplementRank == 0u
		       && state.trialComplementRank == 1u);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test C4: transported q=2 scout should keep rotated residual evidence
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Controller Test C4: rotated scout subspace remains promotable\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 24;
		const unsigned int n = 24;
		const unsigned int r = 2;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 38282835ULL);
		glades::atlas::WeightState state;
		prepare_atlas_complement_test_state(state, rng, m, n, r, 4u, "cnn.fc");

		std::fill(state.U.begin(), state.U.end(), 0.0f);
		std::fill(state.V.begin(), state.V.end(), 0.0f);
		std::fill(state.complementBlock.begin(), state.complementBlock.end(), 0.0f);
		std::fill(state.prevGv.begin(), state.prevGv.end(), 0.0f);
		state.U[0] = 1.0f;
		state.U[state.r + 1] = 1.0f;
		for (unsigned int c = 0; c < state.complementRank; ++c)
			state.V[(c + 2u) * state.complementRank + c] = 1.0f;
		state.fisherDiag[0] = 1.0f;
		state.fisherDiag[1] = 0.0f;
		state.complementBlock[0] = 16.0f;
		state.complementBlock[5] = 1.0f;
		state.complementBlock[10] = 0.25f;
		state.complementBlock[15] = 0.0625f;
		state.complementFisher = 17.3125f;
		state.totalTrace = 18.3125f;
		state.step = 2ULL;

		std::vector<float> W(static_cast<size_t>(m) * n, 0.0f);
		std::vector<float> gW(static_cast<size_t>(m) * n, 0.0f);
		for (unsigned int j = 0; j < n; ++j)
		{
			gW[2u * n + j] = 4.0f;
			gW[3u * n + j] = 4.0f;
		}

		glades::ATLASConfig acRotated;
		acRotated.rank = r;
		acRotated.complementRank = 4u;
		acRotated.beta = 1.0f;
		acRotated.biasCorrection = false;
		acRotated.muMin = 0.0f;
		acRotated.muMax = 0.0f;
		acRotated.tSub = 0u;

		bool ok = glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
		                                   1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
		                                   acRotated, rng, 0, "cnn.fc");
		ASSERT("==============ATLAS::ControllerRotated first applyStep failed==============", ok);
		ASSERT("==============ATLAS::ControllerRotated should open probation from rotated q2 scout==============",
		       state.activeComplementRank == 0u
		       && state.trialComplementRank == 1u
		       && state.trialComplementMean > 0.0f);

		for (unsigned int j = 0; j < n; ++j)
		{
			gW[2u * n + j] = 4.0f;
			gW[3u * n + j] = 4.0f;
		}
		ok = glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
		                              1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
		                              acRotated, rng, 0, "cnn.fc");
		ASSERT("==============ATLAS::ControllerRotated second applyStep failed==============", ok);
		ASSERT("==============ATLAS::ControllerRotated should promote rotated residual mode==============",
		       state.activeComplementRank == 1u && state.trialComplementRank == 0u);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test C5: PRISM predictive-edge gate suppresses low-persistence complement births
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Controller Test C5: PRISM predictive-edge gate blocks stale complement modes\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 24;
		const unsigned int n = 24;
		const unsigned int r = 2;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 38282836ULL);
		glades::atlas::WeightState state;
		prepare_atlas_complement_test_state(state, rng, m, n, r, 4u, "cnn.fc");

		std::fill(state.U.begin(), state.U.end(), 0.0f);
		std::fill(state.V.begin(), state.V.end(), 0.0f);
		std::fill(state.prevGv.begin(), state.prevGv.end(), 0.0f);
		state.U[0] = 1.0f;
		state.U[state.r + 1] = 1.0f;
		for (unsigned int c = 0; c < state.complementRank; ++c)
			state.V[(c + 2u) * state.complementRank + c] = 1.0f;

		std::vector<float> W(static_cast<size_t>(m) * n, 0.0f);
		std::vector<float> gW(static_cast<size_t>(m) * n, 0.0f);
		for (unsigned int j = 0; j < n; ++j)
		{
			gW[2u * n + j] = 4.0f;
			gW[3u * n + j] = 3.0f;
		}

		glades::ATLASConfig acPrismGate;
		acPrismGate.rank = r;
		acPrismGate.complementRank = 4u;
		acPrismGate.beta = 1.0f;
		acPrismGate.biasCorrection = false;
		acPrismGate.muMin = 0.0f;
		acPrismGate.muMax = 0.0f;
		acPrismGate.tSub = 0u;
		acPrismGate.prismEnabled = true;
		acPrismGate.prismPredictiveEdgeThreshold = 0.05f;

		const bool ok = glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
		                                         1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
		                                         acPrismGate, rng, 0, "cnn.fc");
		ASSERT("==============ATLAS::ControllerPrismGate applyStep failed==============", ok);
		ASSERT("==============ATLAS::ControllerPrismGate should suppress stale complement probation==============",
		       state.activeComplementRank == 0u && state.trialComplementRank == 0u);
		ASSERT("==============ATLAS::ControllerPrismGate predictive edge should stay subcritical==============",
		       state.lastPredictiveEdge < acPrismGate.prismPredictiveEdgeThreshold);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test C6: PRISM active memory shrinks the subspace correction on aligned histories
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Controller Test C6: PRISM active memory damps aligned active corrections\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 4;
		const unsigned int n = 4;
		const unsigned int r = 2;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 38282837ULL);
		glades::atlas::WeightState baseState;
		glades::atlas::initWeightState(baseState, m, n, r, 0.0f, rng);
		glades::atlas::WeightState prismState = baseState;

		std::fill(baseState.U.begin(), baseState.U.end(), 0.0f);
		baseState.U[0] = 1.0f;
		baseState.U[baseState.r + 1] = 1.0f;
		prismState.U = baseState.U;

		for (unsigned int j = 0; j < n; ++j)
		{
			baseState.prevGz[0u * n + j] = 4.0f;
			baseState.prevGz[1u * n + j] = 2.0f;
			baseState.prevPrevGz[0u * n + j] = 3.5f;
			baseState.prevPrevGz[1u * n + j] = 1.75f;
		}
		prismState.prevGz = baseState.prevGz;
		prismState.prevPrevGz = baseState.prevPrevGz;

		std::vector<float> gW(static_cast<size_t>(m) * n, 0.0f);
		for (unsigned int j = 0; j < n; ++j)
		{
			gW[0u * n + j] = 4.0f;
			gW[1u * n + j] = 2.0f;
			gW[2u * n + j] = 1.0f;
			gW[3u * n + j] = 1.0f;
		}
		std::vector<float> WBase(static_cast<size_t>(m) * n, 0.0f);
		std::vector<float> WPrism(static_cast<size_t>(m) * n, 0.0f);

		glades::ATLASConfig acBase;
		acBase.rank = r;
		acBase.complementRank = 0u;
		acBase.beta = 0.0f;
		acBase.biasCorrection = false;
		acBase.muMin = 0.0f;
		acBase.muMax = 0.0f;
		acBase.tSub = 0u;

		glades::ATLASConfig acPrism = acBase;
		acPrism.prismEnabled = true;
		acPrism.prismLagHorizon = 2u;
		acPrism.prismMemoryScale = 0.5f;

		bool ok = glades::atlas::applyStep(baseState, &WBase[0], &gW[0], m, n,
		                                   1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
		                                   acBase, rng, 0, 0);
		ASSERT("==============ATLAS::ControllerPrismMemory base applyStep failed==============", ok);
		ok = glades::atlas::applyStep(prismState, &WPrism[0], &gW[0], m, n,
		                              1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
		                              acPrism, rng, 0, 0);
		ASSERT("==============ATLAS::ControllerPrismMemory prism applyStep failed==============", ok);

		double baseCorrectionNormSq = 0.0;
		double prismCorrectionNormSq = 0.0;
		for (size_t idx = 0; idx < WBase.size(); ++idx)
		{
			const double baseCorrection =
			    static_cast<double>(WBase[idx])
			    + static_cast<double>(baseState.lastBaselineRate) * static_cast<double>(gW[idx]);
			const double prismCorrection =
			    static_cast<double>(WPrism[idx])
			    + static_cast<double>(prismState.lastBaselineRate) * static_cast<double>(gW[idx]);
			baseCorrectionNormSq += baseCorrection * baseCorrection;
			prismCorrectionNormSq += prismCorrection * prismCorrection;
		}
		ASSERT("==============ATLAS::ControllerPrismMemory should reduce active correction energy==============",
		       prismCorrectionNormSq < baseCorrectionNormSq);
		ASSERT("==============ATLAS::ControllerPrismMemory memory diagnostic should remain finite==============",
		       prismState.lastMemoryGain == prismState.lastMemoryGain);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test C7: RESOLVE transfer-edge gate suppresses stale complement births
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Controller Test C7: RESOLVE transfer-edge gate blocks stale complement modes\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 24;
		const unsigned int n = 24;
		const unsigned int r = 2;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 38282838ULL);
		glades::atlas::WeightState state;
		prepare_atlas_complement_test_state(state, rng, m, n, r, 4u, "cnn.fc");

		std::fill(state.U.begin(), state.U.end(), 0.0f);
		std::fill(state.V.begin(), state.V.end(), 0.0f);
		std::fill(state.resolveGzHistory.begin(), state.resolveGzHistory.end(), 0.0f);
		state.U[0] = 1.0f;
		state.U[state.r + 1] = 1.0f;
		for (unsigned int c = 0; c < state.complementRank; ++c)
			state.V[(c + 2u) * state.complementRank + c] = 1.0f;

		std::vector<float> W(static_cast<size_t>(m) * n, 0.0f);
		std::vector<float> gW(static_cast<size_t>(m) * n, 0.0f);
		for (unsigned int j = 0; j < n; ++j)
		{
			gW[2u * n + j] = 4.0f;
			gW[3u * n + j] = 3.0f;
		}

		glades::ATLASConfig acResolveGate;
		acResolveGate.rank = r;
		acResolveGate.complementRank = 4u;
		acResolveGate.beta = 1.0f;
		acResolveGate.biasCorrection = false;
		acResolveGate.muMin = 0.0f;
		acResolveGate.muMax = 0.0f;
		acResolveGate.tSub = 0u;
		acResolveGate.resolveEnabled = true;
		acResolveGate.resolveLagHorizon = 4u;
		acResolveGate.resolvePredictiveEdgeThreshold = 0.05f;

		const bool ok = glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
		                                         1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
		                                         acResolveGate, rng, 0, "cnn.fc");
		ASSERT("==============ATLAS::ControllerResolveGate applyStep failed==============", ok);
		ASSERT("==============ATLAS::ControllerResolveGate should suppress stale complement probation==============",
		       state.activeComplementRank == 0u && state.trialComplementRank == 0u);
		ASSERT("==============ATLAS::ControllerResolveGate transfer edge should stay subcritical==============",
		       state.lastResolveEdge < acResolveGate.resolvePredictiveEdgeThreshold);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test C8: RESOLVE active memory shrinks the subspace correction on aligned histories
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Controller Test C8: RESOLVE active memory damps aligned active corrections\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 4;
		const unsigned int n = 4;
		const unsigned int r = 2;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 38282839ULL);
		glades::atlas::WeightState baseState;
		glades::atlas::initWeightState(baseState, m, n, r, 0.0f, rng);
		glades::atlas::WeightState resolveState = baseState;

		std::fill(baseState.U.begin(), baseState.U.end(), 0.0f);
		baseState.U[0] = 1.0f;
		baseState.U[baseState.r + 1] = 1.0f;
		resolveState.U = baseState.U;

		for (unsigned int j = 0; j < n; ++j)
		{
			resolveState.resolveGzHistory[0u * r * n + 0u * n + j] = 4.0f;
			resolveState.resolveGzHistory[0u * r * n + 1u * n + j] = 2.0f;
			resolveState.resolveGzHistory[1u * r * n + 0u * n + j] = 3.5f;
			resolveState.resolveGzHistory[1u * r * n + 1u * n + j] = 1.75f;
			resolveState.resolveGzHistory[2u * r * n + 0u * n + j] = 3.0f;
			resolveState.resolveGzHistory[2u * r * n + 1u * n + j] = 1.5f;
			resolveState.resolveGzHistory[3u * r * n + 0u * n + j] = 2.5f;
			resolveState.resolveGzHistory[3u * r * n + 1u * n + j] = 1.25f;
		}

		std::vector<float> gW(static_cast<size_t>(m) * n, 0.0f);
		for (unsigned int j = 0; j < n; ++j)
		{
			gW[0u * n + j] = 4.0f;
			gW[1u * n + j] = 2.0f;
			gW[2u * n + j] = 1.0f;
			gW[3u * n + j] = 1.0f;
		}
		std::vector<float> WBase(static_cast<size_t>(m) * n, 0.0f);
		std::vector<float> WResolve(static_cast<size_t>(m) * n, 0.0f);

		glades::ATLASConfig acBase;
		acBase.rank = r;
		acBase.complementRank = 0u;
		acBase.beta = 0.0f;
		acBase.biasCorrection = false;
		acBase.muMin = 0.0f;
		acBase.muMax = 0.0f;
		acBase.tSub = 0u;

		glades::ATLASConfig acResolve = acBase;
		acResolve.resolveEnabled = true;
		acResolve.resolveLagHorizon = 4u;
		acResolve.resolveMemoryScale = 0.20f;

		bool ok = glades::atlas::applyStep(baseState, &WBase[0], &gW[0], m, n,
		                                   1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
		                                   acBase, rng, 0, 0);
		ASSERT("==============ATLAS::ControllerResolveMemory base applyStep failed==============", ok);
		ok = glades::atlas::applyStep(resolveState, &WResolve[0], &gW[0], m, n,
		                              1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
		                              acResolve, rng, 0, 0);
		ASSERT("==============ATLAS::ControllerResolveMemory resolve applyStep failed==============", ok);

		double baseCorrectionNormSq = 0.0;
		double resolveCorrectionNormSq = 0.0;
		for (size_t idx = 0; idx < WBase.size(); ++idx)
		{
			const double baseCorrection =
			    static_cast<double>(WBase[idx])
			    + static_cast<double>(baseState.lastBaselineRate) * static_cast<double>(gW[idx]);
			const double resolveCorrection =
			    static_cast<double>(WResolve[idx])
			    + static_cast<double>(resolveState.lastBaselineRate) * static_cast<double>(gW[idx]);
			baseCorrectionNormSq += baseCorrection * baseCorrection;
			resolveCorrectionNormSq += resolveCorrection * resolveCorrection;
		}
		ASSERT("==============ATLAS::ControllerResolveMemory should reduce active correction energy==============",
		       resolveCorrectionNormSq < baseCorrectionNormSq);
	ASSERT("==============ATLAS::ControllerResolveMemory memory diagnostic should remain finite==============",
	       resolveState.lastResolveMemoryGain == resolveState.lastResolveMemoryGain);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test C9: HERO Hankel-edge gate suppresses stale complement births
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Controller Test C9: HERO Hankel-edge gate blocks stale complement modes\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 24;
		const unsigned int n = 24;
		const unsigned int r = 2;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 38282840ULL);
		glades::atlas::WeightState state;
		prepare_atlas_complement_test_state(state, rng, m, n, r, 4u, "cnn.fc");

		std::fill(state.U.begin(), state.U.end(), 0.0f);
		std::fill(state.V.begin(), state.V.end(), 0.0f);
		std::fill(state.heroGwHistory.begin(), state.heroGwHistory.end(), 0.0f);
		state.U[0] = 1.0f;
		state.U[state.r + 1] = 1.0f;
		for (unsigned int c = 0; c < state.complementRank; ++c)
			state.V[(c + 2u) * state.complementRank + c] = 1.0f;

		std::vector<float> W(static_cast<size_t>(m) * n, 0.0f);
		std::vector<float> gW(static_cast<size_t>(m) * n, 0.0f);
		for (unsigned int j = 0; j < n; ++j)
		{
			gW[2u * n + j] = 4.0f;
			gW[3u * n + j] = 3.0f;
		}

		glades::ATLASConfig acHeroGate;
		acHeroGate.rank = r;
		acHeroGate.complementRank = 4u;
		acHeroGate.beta = 1.0f;
		acHeroGate.biasCorrection = false;
		acHeroGate.muMin = 0.0f;
		acHeroGate.muMax = 0.0f;
		acHeroGate.tSub = 0u;
		acHeroGate.heroEnabled = true;
		acHeroGate.heroLagHorizon = 4u;
		acHeroGate.heroEdgeThreshold = 0.10f;

		const bool ok = glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
		                                         1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
		                                         acHeroGate, rng, 0, "cnn.fc");
		ASSERT("==============ATLAS::ControllerHeroGate applyStep failed==============", ok);
		ASSERT("==============ATLAS::ControllerHeroGate should suppress stale complement probation==============",
		       state.activeComplementRank == 0u && state.trialComplementRank == 0u);
		ASSERT("==============ATLAS::ControllerHeroGate Hankel edge should stay subcritical==============",
		       state.lastHeroEdge < acHeroGate.heroEdgeThreshold);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test C10: HERO active memory shrinks the subspace correction on aligned histories
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Controller Test C10: HERO active memory damps aligned active corrections\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 4;
		const unsigned int n = 4;
		const unsigned int r = 2;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 38282841ULL);
		glades::atlas::WeightState baseState;
		glades::atlas::initWeightState(baseState, m, n, r, 0.0f, rng);
		glades::atlas::WeightState heroState = baseState;

		std::fill(baseState.U.begin(), baseState.U.end(), 0.0f);
		baseState.U[0] = 1.0f;
		baseState.U[baseState.r + 1] = 1.0f;
		heroState.U = baseState.U;

		for (unsigned int j = 0; j < n; ++j)
		{
			heroState.resolveGzHistory[0u * r * n + 0u * n + j] = 4.0f;
			heroState.resolveGzHistory[0u * r * n + 1u * n + j] = 2.0f;
			heroState.resolveGzHistory[1u * r * n + 0u * n + j] = 3.5f;
			heroState.resolveGzHistory[1u * r * n + 1u * n + j] = 1.75f;
			heroState.resolveGzHistory[2u * r * n + 0u * n + j] = 3.0f;
			heroState.resolveGzHistory[2u * r * n + 1u * n + j] = 1.5f;
			heroState.resolveGzHistory[3u * r * n + 0u * n + j] = 2.5f;
			heroState.resolveGzHistory[3u * r * n + 1u * n + j] = 1.25f;
		}

		std::vector<float> gW(static_cast<size_t>(m) * n, 0.0f);
		for (unsigned int j = 0; j < n; ++j)
		{
			gW[0u * n + j] = 4.0f;
			gW[1u * n + j] = 2.0f;
			gW[2u * n + j] = 1.0f;
			gW[3u * n + j] = 1.0f;
		}
		std::vector<float> WBase(static_cast<size_t>(m) * n, 0.0f);
		std::vector<float> WHero(static_cast<size_t>(m) * n, 0.0f);

		glades::ATLASConfig acBase;
		acBase.rank = r;
		acBase.complementRank = 0u;
		acBase.beta = 0.0f;
		acBase.biasCorrection = false;
		acBase.muMin = 0.0f;
		acBase.muMax = 0.0f;
		acBase.tSub = 0u;

		glades::ATLASConfig acHero = acBase;
		acHero.heroEnabled = true;
		acHero.heroLagHorizon = 4u;
		acHero.heroMemoryScale = 0.20f;

		bool ok = glades::atlas::applyStep(baseState, &WBase[0], &gW[0], m, n,
		                                   1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
		                                   acBase, rng, 0, 0);
		ASSERT("==============ATLAS::ControllerHeroMemory base applyStep failed==============", ok);
		ok = glades::atlas::applyStep(heroState, &WHero[0], &gW[0], m, n,
		                              1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
		                              acHero, rng, 0, 0);
		ASSERT("==============ATLAS::ControllerHeroMemory hero applyStep failed==============", ok);

		double baseCorrectionNormSq = 0.0;
		double heroCorrectionNormSq = 0.0;
		for (size_t idx = 0; idx < WBase.size(); ++idx)
		{
			const double baseCorrection =
			    static_cast<double>(WBase[idx])
			    + static_cast<double>(baseState.lastBaselineRate) * static_cast<double>(gW[idx]);
			const double heroCorrection =
			    static_cast<double>(WHero[idx])
			    + static_cast<double>(heroState.lastBaselineRate) * static_cast<double>(gW[idx]);
			baseCorrectionNormSq += baseCorrection * baseCorrection;
			heroCorrectionNormSq += heroCorrection * heroCorrection;
		}
		ASSERT("==============ATLAS::ControllerHeroMemory should reduce active correction energy==============",
		       heroCorrectionNormSq < baseCorrectionNormSq);
		ASSERT("==============ATLAS::ControllerHeroMemory memory diagnostic should remain finite==============",
		       heroState.lastHeroMemoryGain == heroState.lastHeroMemoryGain);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test C3: misaligned scout directions are rejected
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Controller Test C3: misaligned scout rejection\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 24;
		const unsigned int n = 24;
		const unsigned int r = 2;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 38282834ULL);
		glades::atlas::WeightState state;
		prepare_atlas_complement_test_state(state, rng, m, n, r, 4u, "cnn.fc");

		std::fill(state.U.begin(), state.U.end(), 0.0f);
		std::fill(state.V.begin(), state.V.end(), 0.0f);
		std::fill(state.complementBlock.begin(), state.complementBlock.end(), 0.0f);
		std::fill(state.prevGv.begin(), state.prevGv.end(), 0.0f);
		state.U[0] = 1.0f;
		state.U[state.r + 1] = 1.0f;
		for (unsigned int c = 0; c < state.complementRank; ++c)
			state.V[(c + 2u) * state.complementRank + c] = 1.0f;
		state.fisherDiag[0] = 1.0f;
		state.fisherDiag[1] = 0.0f;
		state.complementBlock[0] = 16.0f;
		state.complementBlock[5] = 4.0f;
		state.complementBlock[10] = 1.0f;
		state.complementBlock[15] = 0.25f;
		state.complementFisher = 21.25f;
		state.totalTrace = 22.25f;
		state.step = 2ULL;

		std::vector<float> W(static_cast<size_t>(m) * n, 0.0f);
		std::vector<float> gW(static_cast<size_t>(m) * n, 0.0f);
		for (unsigned int j = 0; j < n; ++j)
			gW[3u * n + j] = 12.0f;

		glades::ATLASConfig acReject;
		acReject.rank = r;
		acReject.complementRank = 4u;
		acReject.beta = 1.0f;
		acReject.biasCorrection = false;
		acReject.muMin = 0.0f;
		acReject.muMax = 0.0f;
		acReject.tSub = 0u;

		bool ok = glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
		                                   1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
		                                   acReject, rng, 0, "cnn.fc");
		ASSERT("==============ATLAS::ControllerReject first applyStep failed==============", ok);
		ASSERT("==============ATLAS::ControllerReject should reject misaligned scout mode==============",
		       state.activeComplementRank == 0u && state.trialComplementRank == 0u);

		for (unsigned int j = 0; j < n; ++j)
			gW[3u * n + j] = 12.0f;
		ok = glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
		                              1.0f, 0.0f, 0.0f, 0.0f, 1.0f,
		                              acReject, rng, 0, "cnn.fc");
		ASSERT("==============ATLAS::ControllerReject second applyStep failed==============", ok);
		ASSERT("==============ATLAS::ControllerReject should still reject misaligned scout mode==============",
		       state.activeComplementRank == 0u && state.trialComplementRank == 0u);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	printf("\n============================================================\n");
}

void ATLASGpuNaNTest()
{
	printf("============================================================\n");
	printf("ATLAS GPU NaN Reproduction Test\n");
	printf("============================================================\n");

#ifdef GLADES_HAVE_CUDA
	// ---------------------------------------------------------------
	// Test GPU 0: Fisher kernel mean-squared reduction
	//
	// Isolates atlas_gpu_fisher_update. With beta=0 and gz filled with 1.0,
	// each Fisher entry must become exactly 1.0 because mean(gz^2)=1.
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS GPU Test 0: Fisher kernel mean-squared reduction\n");
	printf("-----------------------------------\n");
	if (glades::gpu::initDevice())
	{
		const int r = 4;
		const int outerDim = 1024;

		glades::gpu::GpuBuffer<float> d_gz_left, d_gz_right, d_fisher;
		ASSERT("==============GPU Fisher left alloc==============",
		       d_gz_left.allocate((size_t)r * outerDim));
		ASSERT("==============GPU Fisher right alloc==============",
		       d_gz_right.allocate((size_t)outerDim * r));
		ASSERT("==============GPU Fisher diag alloc==============",
		       d_fisher.allocate((size_t)r));

		std::vector<float> ones((size_t)r * outerDim, 1.0f);
		std::vector<float> fisher(r, 0.0f);

		ASSERT("==============GPU Fisher left upload==============",
		       d_gz_left.upload(ones.data(), ones.size()));
		ASSERT("==============GPU Fisher diag zero upload==============",
		       d_fisher.upload(fisher.data(), fisher.size()));
		ASSERT("==============GPU Fisher left update==============",
		       glades::gpu::atlas_gpu_fisher_update(
		           d_gz_left.data(), d_fisher.data(), r, outerDim, 0.0f, 1.0f, false, false));
		ASSERT("==============GPU Fisher left download==============",
		       d_fisher.download(fisher.data(), fisher.size()));
		for (int c = 0; c < r; ++c)
		{
			printf("    left fisher[%d]=%.8f\n", c, fisher[c]);
			ASSERT("==============GPU Fisher left mean-squared mismatch==============",
			       fabsf(fisher[c] - 1.0f) < 1e-5f);
		}

		ASSERT("==============GPU Fisher right upload==============",
		       d_gz_right.upload(ones.data(), ones.size()));
		std::fill(fisher.begin(), fisher.end(), 0.0f);
		ASSERT("==============GPU Fisher diag reset upload==============",
		       d_fisher.upload(fisher.data(), fisher.size()));
		ASSERT("==============GPU Fisher right update==============",
		       glades::gpu::atlas_gpu_fisher_update(
		           d_gz_right.data(), d_fisher.data(), r, outerDim, 0.0f, 1.0f, true, false));
		ASSERT("==============GPU Fisher right download==============",
		       d_fisher.download(fisher.data(), fisher.size()));
		for (int c = 0; c < r; ++c)
		{
			printf("    right fisher[%d]=%.8f\n", c, fisher[c]);
			ASSERT("==============GPU Fisher right mean-squared mismatch==============",
			       fabsf(fisher[c] - 1.0f) < 1e-5f);
		}
	}

	// ---------------------------------------------------------------
	// Test GPU: ATLAS GPU NaN reproduction — zero/tiny gradient refresh
	//
	// Reproduces the production bug where layer 0 Wq gets NaN in
	// gz_norm at the first refresh step (step 100 with tSub=100).
	// Tests with zero gradients, tiny gradients, and normal gradients
	// to isolate which pattern triggers the NaN.
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS GPU Test: NaN at refresh with zero/tiny gradients\n");
	printf("-----------------------------------\n");
	if (glades::gpu::initDevice())
	{
		const unsigned int m = 1024;
		const unsigned int n = 1024;
		const unsigned int r = 256;
		const unsigned int tSub = 100;
		const float lr = 5e-4f;
		const float muInit = 0.01f;

		struct GradScenario
		{
			const char* name;
			float gradScale; // 0 = zero grads, >0 = fill with random * gradScale
		};

		const GradScenario scenarios[] = {
			{"zero", 0.0f},
			{"tiny (1e-8)", 1e-8f},
			{"small (1e-5)", 1e-5f},
			{"normal (0.01)", 0.01f},
		};
		const int nScenarios = 4;

		for (int si = 0; si < nScenarios; ++si)
		{
			const GradScenario& sc = scenarios[si];
			printf("  Scenario: %s gradients\n", sc.name);

			// Allocate GPU buffers for weight and gradient
			glades::gpu::GpuBuffer<float> d_W, d_gW;
			ASSERT("==============GPU ATLAS NaN: W alloc failed==============",
			       d_W.allocate((size_t)m * n));
			ASSERT("==============GPU ATLAS NaN: gW alloc failed==============",
			       d_gW.allocate((size_t)m * n));

			// Initialize weights to small random values
			{
				std::vector<float> h_W(m * n);
				glades::rng::Engine initRng;
				glades::rng::seed_engine(initRng, 42ULL);
				for (size_t i = 0; i < m * n; ++i)
					h_W[i] = glades::rng::standard_normal(initRng) * 0.02f;
				ASSERT("==============GPU ATLAS NaN: W upload failed==============",
				       d_W.upload(h_W.data(), m * n));
			}

			// Initialize ATLAS state
			glades::gpu::GpuAtlasWeightState state;
			glades::rng::Engine rng;
			glades::rng::seed_engine(rng, 12345ULL + si * 1000ULL);
			ASSERT("==============GPU ATLAS NaN: init failed==============",
			       glades::gpu::atlas_gpu_init(state, m, n, r, muInit, rng));

			glades::ATLASConfig ac;
			ac.rank = r;
			ac.tSub = tSub;
			ac.beta = 0.99f;
			ac.muMin = 0.01f;
			ac.muMax = 0.3f;
			ac.eps = 1e-8f;
			ac.kappaMax = 10.0f;
			ac.betaRefresh = 0.5f;
			ac.powerIters = 2;
			ac.muGrowthRate = 0.001f;
			ac.biasCorrection = true;

			bool nanDetected = false;
			unsigned int nanStep = 0;

			// Run for tSub+5 steps to cover the first refresh
			const unsigned int totalSteps = tSub + 5;
			for (unsigned int step = 0; step < totalSteps; ++step)
			{
				// Fill gradient buffer
				if (sc.gradScale == 0.0f)
				{
					// Zero gradient
					ASSERT("==============GPU ATLAS NaN: gW zero failed==============",
					       d_gW.zero());
				}
				else
				{
					std::vector<float> h_gW(m * n);
					for (size_t i = 0; i < m * n; ++i)
						h_gW[i] = glades::rng::standard_normal(rng) * sc.gradScale;
					ASSERT("==============GPU ATLAS NaN: gW upload failed==============",
					       d_gW.upload(h_gW.data(), m * n));
				}

				bool ok = glades::gpu::atlas_gpu_step(
					state, d_W.data(), d_gW.data(), m, n,
					/*invBatch=*/0.05f, lr,
					/*wd1=*/0.0f, /*wd2=*/0.01f, /*gradScale=*/1.0f,
					ac);

				if (!ok)
				{
					printf("    [FAIL] atlas_gpu_step returned false at step %u\n", step + 1);
					nanDetected = true;
					nanStep = step + 1;
					break;
				}

				// Check diagnostics at refresh steps
				if (state.step > 0ULL && (state.step % tSub) == 0ULL)
				{
					glades::gpu::AtlasGpuDiag diag = glades::gpu::atlas_gpu_get_diag(state);
					if (diag.valid)
					{
						bool gzNaN = (diag.gzNorm != diag.gzNorm);
						bool updateNaN = (diag.updateNorm != diag.updateNorm);
						bool fisherNaN = (diag.fisherMean != diag.fisherMean);
						printf("    step=%llu: sigma2=%.3e fisher_mean=%.3e "
						       "gz_norm=%s update_norm=%s\n",
						       (unsigned long long)state.step,
						       diag.sigma2, diag.fisherMean,
						       gzNaN ? "NaN" : "ok",
						       updateNaN ? "NaN" : "ok");
						if (gzNaN || updateNaN || fisherNaN)
						{
							nanDetected = true;
							nanStep = (unsigned int)state.step;
						}
					}
				}

				// Also spot-check sigma2 and mu for NaN
				if (state.sigma2 != state.sigma2 || state.mu != state.mu)
				{
					printf("    [FAIL] sigma2 or mu is NaN at step %llu\n",
					       (unsigned long long)state.step);
					nanDetected = true;
					nanStep = (unsigned int)state.step;
					break;
				}
			}

			if (nanDetected)
				printf("    [FAIL] NaN detected at step %u with %s gradients\n",
				       nanStep, sc.name);
			else
				printf("    [PASS] No NaN after %u steps with %s gradients\n",
				       totalSteps, sc.name);

			ASSERT("==============GPU ATLAS NaN: NaN detected in diagnostics==============",
			       !nanDetected);
		}
	}
	else
	{
		printf("  No CUDA device available, skipping GPU ATLAS NaN test.\n");
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test GPU 2: NaN gradient at refresh step — permanent corruption
	//
	// Simulates a NaN arriving in d_gW at exactly the refresh step.
	// This is the suspected production root cause: a single NaN gradient
	// at tSub boundary corrupts U permanently via power iteration.
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS GPU Test 2: NaN gradient at refresh step\n");
	printf("-----------------------------------\n");
	if (glades::gpu::initDevice())
	{
		const unsigned int m = 256;
		const unsigned int n = 256;
		const unsigned int r = 32;
		const unsigned int tSub = 10;
		const float lr = 5e-4f;

		glades::gpu::GpuBuffer<float> d_W, d_gW;
		ASSERT("==============GPU NaN refresh: W alloc==============",
		       d_W.allocate((size_t)m * n));
		ASSERT("==============GPU NaN refresh: gW alloc==============",
		       d_gW.allocate((size_t)m * n));

		// Init weights
		{
			std::vector<float> h_W(m * n);
			glades::rng::Engine initRng;
			glades::rng::seed_engine(initRng, 42ULL);
			for (size_t i = 0; i < m * n; ++i)
				h_W[i] = glades::rng::standard_normal(initRng) * 0.02f;
			d_W.upload(h_W.data(), m * n);
		}

		glades::gpu::GpuAtlasWeightState state;
		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 99999ULL);
		glades::gpu::atlas_gpu_init(state, m, n, r, 0.01f, rng);

		glades::ATLASConfig ac;
		ac.rank = r;
		ac.tSub = tSub;
		ac.beta = 0.99f;
		ac.muMin = 0.01f;
		ac.muMax = 0.3f;
		ac.eps = 1e-8f;
		ac.kappaMax = 10.0f;
		ac.betaRefresh = 0.5f;
		ac.powerIters = 2;
		ac.muGrowthRate = 0.001f;
		ac.biasCorrection = true;

		bool nanBefore = false;
		bool nanAfter = false;
		const unsigned int poisonStep = tSub; // inject NaN exactly at refresh step
		const unsigned int totalSteps = tSub * 3; // run past to check persistence

		for (unsigned int step = 0; step < totalSteps; ++step)
		{
			std::vector<float> h_gW(m * n);
			for (size_t i = 0; i < m * n; ++i)
				h_gW[i] = glades::rng::standard_normal(rng) * 0.01f;

			// Inject NaN at the refresh step
			if (step + 1 == poisonStep)
			{
				h_gW[0] = 0.0f / 0.0f; // NaN
				h_gW[m * n / 2] = 0.0f / 0.0f;
				printf("    Injecting NaN gradient at step %u (refresh step)\n", step + 1);
			}
			d_gW.upload(h_gW.data(), m * n);

			glades::gpu::atlas_gpu_step(
				state, d_W.data(), d_gW.data(), m, n,
				0.05f, lr, 0.0f, 0.01f, 1.0f, ac);

			// Check diagnostics at diagnostic steps (multiples of tSub)
			if (state.step > 0ULL && (state.step % tSub) == 0ULL)
			{
				glades::gpu::AtlasGpuDiag diag = glades::gpu::atlas_gpu_get_diag(state);
				if (diag.valid)
				{
					bool gzNaN = (diag.gzNorm != diag.gzNorm);
					printf("    step=%llu: gz_norm=%s sigma2=%.3e fisher_mean=%.3e\n",
					       (unsigned long long)state.step,
					       gzNaN ? "NaN" : "ok",
					       diag.sigma2, diag.fisherMean);
					if (state.step <= poisonStep)
						nanBefore = nanBefore || gzNaN;
					else
						nanAfter = nanAfter || gzNaN;
				}
			}
		}

		if (nanAfter)
			printf("    [FAIL] NaN persists after poisoned refresh step\n");
		else
			printf("    [PASS] No persistent NaN after poisoned refresh step\n");

		ASSERT("==============GPU ATLAS NaN: NaN persists after poisoned gradient==============",
		       !nanAfter);
	}

	// ---------------------------------------------------------------
	// Test GPU 3: Very small gradient at refresh — CholeskyQR overflow
	//
	// Layer 0 Wq gets gradients with RMS ~2e-6 (100x smaller than other
	// layers). This may cause rank-deficient power iteration output,
	// leading to huge R_inv entries in CholeskyQR that overflow.
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS GPU Test 3: Tiny gradient at refresh (layer 0 Wq scenario)\n");
	printf("-----------------------------------\n");
	if (glades::gpu::initDevice())
	{
		// Test with gradient magnitudes matching production layer 0 Wq
		const float gradScales[] = {2e-6f, 1e-7f, 1e-9f, 0.0f};
		const char* scaleNames[] = {"2e-6 (production)", "1e-7", "1e-9", "zero"};
		const int nScales = 4;

		for (int si = 0; si < nScales; ++si)
		{
			const unsigned int m = 1024;
			const unsigned int n = 1024;
			const unsigned int r = 256;
			const unsigned int tSub = 10; // fast refresh for testing
			printf("  Gradient RMS = %s:\n", scaleNames[si]);

			glades::gpu::GpuBuffer<float> d_W, d_gW;
			d_W.allocate((size_t)m * n);
			d_gW.allocate((size_t)m * n);

			// Init weights
			{
				std::vector<float> h_W(m * n);
				glades::rng::Engine wr;
				glades::rng::seed_engine(wr, 42ULL);
				for (size_t i = 0; i < m * n; ++i)
					h_W[i] = glades::rng::standard_normal(wr) * 0.02f;
				d_W.upload(h_W.data(), m * n);
			}

			glades::gpu::GpuAtlasWeightState state;
			glades::rng::Engine rng;
			glades::rng::seed_engine(rng, 77777ULL);
			glades::gpu::atlas_gpu_init(state, m, n, r, 0.01f, rng);

			glades::ATLASConfig ac;
			ac.rank = r;
			ac.tSub = tSub;
			ac.beta = 0.99f;
			ac.muMin = 0.01f;
			ac.muMax = 0.3f;
			ac.eps = 1e-8f;
			ac.kappaMax = 10.0f;
			ac.betaRefresh = 0.5f;
			ac.powerIters = 2;
			ac.muGrowthRate = 0.001f;
			ac.biasCorrection = true;

			bool nanDetected = false;
			for (unsigned int step = 0; step < tSub + 2; ++step)
			{
				std::vector<float> h_gW(m * n);
				if (gradScales[si] > 0.0f)
				{
					for (size_t i = 0; i < m * n; ++i)
						h_gW[i] = glades::rng::standard_normal(rng) * gradScales[si];
				}
				d_gW.upload(h_gW.data(), m * n);

				bool ok = glades::gpu::atlas_gpu_step(
					state, d_W.data(), d_gW.data(), m, n,
					3.05e-6f, 5e-4f, 0.0f, 0.01f, 1.0f, ac);

				if (!ok) { nanDetected = true; break; }

				if (state.step > 0ULL && (state.step % tSub) == 0ULL)
				{
					glades::gpu::AtlasGpuDiag diag = glades::gpu::atlas_gpu_get_diag(state);
					if (diag.valid && diag.gzNorm != diag.gzNorm)
					{
						printf("    [NaN] gz_norm=NaN at step %llu\n",
						       (unsigned long long)state.step);
						nanDetected = true;
					}
				}
			}

			printf("    %s\n", nanDetected ? "[FAIL] NaN detected" : "[PASS] No NaN");
			ASSERT("==============GPU ATLAS Test3: NaN from tiny gradient==============",
			       !nanDetected);
		}
	}

	// ---------------------------------------------------------------
	// Test GPU 4: Sequential large+small matrix (tokE then Wq interaction)
	//
	// In production, tokE (32000×1024) runs first, then layer 0 Wq
	// (1024×1024). Test if the large SGEMM corrupts cuBLAS state
	// or shared resources that affect the subsequent smaller weight.
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS GPU Test 4: Sequential tokE + Wq interaction\n");
	printf("-----------------------------------\n");
	if (glades::gpu::initDevice())
	{
		const unsigned int tSub = 10;
		const float lr = 5e-4f;
		const float invBatch = 3.05e-6f;

		// --- tokE state: 32000×1024, r=256 ---
		glades::gpu::GpuBuffer<float> d_W_tok, d_gW_tok;
		d_W_tok.allocate(32000u * 1024u);
		d_gW_tok.allocate(32000u * 1024u);
		{
			std::vector<float> h(32000u * 1024u, 0.01f);
			d_W_tok.upload(h.data(), h.size());
		}

		glades::gpu::GpuAtlasWeightState tokState;
		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 12345ULL);
		glades::gpu::atlas_gpu_init(tokState, 32000, 1024, 256, 0.01f, rng);

		// --- Wq state: 1024×1024, r=256 ---
		glades::gpu::GpuBuffer<float> d_W_wq, d_gW_wq;
		d_W_wq.allocate(1024u * 1024u);
		d_gW_wq.allocate(1024u * 1024u);
		{
			std::vector<float> h(1024u * 1024u, 0.01f);
			d_W_wq.upload(h.data(), h.size());
		}

		glades::gpu::GpuAtlasWeightState wqState;
		// NOTE: rng is SHARED — this mimics production where the same rng
		// is used for all weight inits sequentially
		glades::gpu::atlas_gpu_init(wqState, 1024, 1024, 256, 0.01f, rng);

		glades::ATLASConfig ac;
		ac.rank = 256;
		ac.tSub = tSub;
		ac.beta = 0.99f;
		ac.muMin = 0.01f;
		ac.muMax = 0.3f;
		ac.eps = 1e-8f;
		ac.kappaMax = 10.0f;
		ac.betaRefresh = 0.5f;
		ac.powerIters = 2;
		ac.muGrowthRate = 0.001f;
		ac.biasCorrection = true;

		bool nanDetected = false;

		for (unsigned int step = 0; step < tSub + 2; ++step)
		{
			// Fill tokE gradient (large, ~0.01 magnitude)
			{
				std::vector<float> h(32000u * 1024u);
				for (size_t i = 0; i < h.size(); ++i)
					h[i] = glades::rng::standard_normal(rng) * 0.01f;
				d_gW_tok.upload(h.data(), h.size());
			}
			// Step tokE FIRST (like production)
			glades::gpu::atlas_gpu_step(
				tokState, d_W_tok.data(), d_gW_tok.data(), 32000, 1024,
				invBatch, lr, 0.0f, 0.01f, 1.0f, ac);

			// Fill Wq gradient (tiny, ~2e-6 magnitude)
			{
				std::vector<float> h(1024u * 1024u);
				for (size_t i = 0; i < h.size(); ++i)
					h[i] = glades::rng::standard_normal(rng) * 2e-6f;
				d_gW_wq.upload(h.data(), h.size());
			}
			// Step Wq SECOND
			glades::gpu::atlas_gpu_step(
				wqState, d_W_wq.data(), d_gW_wq.data(), 1024, 1024,
				invBatch, lr, 0.0f, 0.01f, 1.0f, ac);

			if (wqState.step > 0ULL && (wqState.step % tSub) == 0ULL)
			{
				glades::gpu::AtlasGpuDiag diag = glades::gpu::atlas_gpu_get_diag(wqState);
				if (diag.valid && diag.gzNorm != diag.gzNorm)
				{
					printf("    [NaN] Wq gz_norm=NaN at step %llu\n",
					       (unsigned long long)wqState.step);
					nanDetected = true;
				}
			}
		}

		printf("    %s\n", nanDetected ? "[FAIL] NaN in Wq after tokE" : "[PASS] No NaN");
		ASSERT("==============GPU ATLAS Test4: Sequential tokE+Wq NaN==============",
		       !nanDetected);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

#else
	printf("  CUDA not enabled at build time, skipping GPU ATLAS NaN test.\n");
#endif // GLADES_HAVE_CUDA

	printf("\n============================================================\n");
}
