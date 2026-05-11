// Net-type-specific SGD: Transformer encoder/decoder-only path
#include "network.h"
#include "transformer_config.h"
#include "sgd_utils.h"
#include "transformer_common_utils.h"
#include "transformer_train_detail.h"
#include "transformer_kernels.h"
#include "gemm_helpers.h"
#include "glades_thread_pool.h"
#include "ddp_comm.h"
#include <cstdlib> // getenv for BF16 per-site debug
#include <cstring> // strcmp for GLADES_CHIRON_ATTN env gating

#ifdef GLADES_HAVE_CUDA
#include "cuda/gpu_dispatch.h"
#include "cuda/gpu_kernels.h"
#include "cuda/gpu_blas.h"
#include "cuda/gpu_atlas.h"
#include "cuda/gpu_chiron.h"
#include "cuda/gpu_transformer_state.h"
#endif

#include "Backend/Database/GLogger.h"

#include "../DataObjects/DataInput.h"
#include "../GMath/gmath.h"
#include "../rng.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <time.h>
#include <vector>

#include "logfmt_utils.h"

using namespace glades;
using namespace glades::logfmt;
using namespace glades::transformer_train_detail;

namespace {

static timespec monotonic_now()
{
	timespec ts;
	ts.tv_sec = 0;
	ts.tv_nsec = 0;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts;
}

static double monotonic_elapsed_ns(const timespec& start, const timespec& end)
{
	const double sec = static_cast<double>(end.tv_sec - start.tv_sec) * 1.0e9;
	const double nsec = static_cast<double>(end.tv_nsec - start.tv_nsec);
	return sec + nsec;
}

static bool bimap_scope_uses_head(const glades::ATLASConfig& ac)
{
	return ac.bimapScope == glades::ATLASConfig::BIMAP_SCOPE_ALL
	    || ac.bimapScope == glades::ATLASConfig::BIMAP_SCOPE_HEAD_ONLY
	    || ac.bimapScope == glades::ATLASConfig::BIMAP_SCOPE_LATE_HEAD;
}

static bool bimap_scope_uses_late_block(const glades::ATLASConfig& ac)
{
	return ac.bimapScope == glades::ATLASConfig::BIMAP_SCOPE_ALL
	    || ac.bimapScope == glades::ATLASConfig::BIMAP_SCOPE_LATE_ONLY
	    || ac.bimapScope == glades::ATLASConfig::BIMAP_SCOPE_LATE_HEAD;
}

static bool bimap_scope_uses_input_block(const glades::ATLASConfig& ac)
{
	return ac.bimapScope == glades::ATLASConfig::BIMAP_SCOPE_ALL;
}

static bool bimap_scope_uses_decoder_block(const glades::ATLASConfig& ac,
                                           unsigned int blockIndex,
                                           unsigned int layerCount)
{
	if (!bimap_scope_uses_late_block(ac))
		return false;
	if (ac.bimapScope == glades::ATLASConfig::BIMAP_SCOPE_ALL)
		return true;
	if (layerCount == 0u)
		return false;
	return blockIndex + 1u == layerCount;
}

static bool muon_matrix_eligible(const glades::ATLASConfig& ac,
                                 unsigned int rows,
                                 unsigned int cols)
{
	if (rows == 0u || cols == 0u)
		return false;
	const float geomScale = std::max(0.0f, ac.muonGeometryScale);
	if (geomScale <= 0.0f)
		return false;
	const unsigned int shortDim = std::min(rows, cols);
	if (shortDim < std::max(1u, ac.muonMinDim))
		return false;
	const unsigned int longDim = std::max(rows, cols);
	const float aspect =
	    static_cast<float>(longDim) / static_cast<float>(std::max(1u, shortDim));
	return aspect <= std::max(1.0f, ac.muonMaxAspect);
}

static bool argos_scope_uses_head(const glades::ATLASConfig& ac)
{
	return ac.argosScope == glades::ATLASConfig::ARGOS_SCOPE_ALL
	    || ac.argosScope == glades::ATLASConfig::ARGOS_SCOPE_HEAD_ONLY
	    || ac.argosScope == glades::ATLASConfig::ARGOS_SCOPE_LATE_HEAD;
}

static bool argos_scope_uses_late_block(const glades::ATLASConfig& ac)
{
	return ac.argosScope == glades::ATLASConfig::ARGOS_SCOPE_ALL
	    || ac.argosScope == glades::ATLASConfig::ARGOS_SCOPE_LATE_ONLY
	    || ac.argosScope == glades::ATLASConfig::ARGOS_SCOPE_LATE_HEAD;
}

static bool argos_scope_uses_input_block(const glades::ATLASConfig& ac)
{
	return ac.argosScope == glades::ATLASConfig::ARGOS_SCOPE_ALL;
}

static bool argos_scope_uses_decoder_block(const glades::ATLASConfig& ac,
                                           unsigned int blockIndex,
                                           unsigned int layerCount)
{
	if (!argos_scope_uses_late_block(ac))
		return false;
	if (ac.argosScope == glades::ATLASConfig::ARGOS_SCOPE_ALL)
		return true;
	if (layerCount == 0u)
		return false;
	return blockIndex + 1u == layerCount;
}

static bool matra_batch_eligible(const glades::ATLASConfig& ac,
                                 unsigned int rows,
                                 unsigned int cols)
{
	if (rows == 0u || cols == 0u)
		return false;
	const float geomScale = std::max(0.0f, ac.matraGeometryScale);
	const float orthScale = std::max(0.0f, ac.matraOrthogonalScale);
	if (geomScale <= 0.0f && orthScale <= 0.0f)
		return false;
	return true;
}

static bool echo_scope_large_matrix(unsigned int rows, unsigned int cols)
{
	return static_cast<unsigned long long>(rows) * static_cast<unsigned long long>(cols) >= 4096ULL;
}

static bool echo_scope_uses_head_matrix(const glades::ATLASConfig& ac,
                                        unsigned int rows,
                                        unsigned int cols)
{
	switch (ac.echoScope)
	{
	case glades::ATLASConfig::ECHO_SCOPE_ALL:
	case glades::ATLASConfig::ECHO_SCOPE_LATE_HEAD:
		return true;
	case glades::ATLASConfig::ECHO_SCOPE_LARGE_ONLY:
	case glades::ATLASConfig::ECHO_SCOPE_LATE_HEAD_LARGE:
		return echo_scope_large_matrix(rows, cols);
	default:
		return false;
	}
}

static bool echo_scope_uses_input_matrix(const glades::ATLASConfig& ac,
                                         unsigned int rows,
                                         unsigned int cols)
{
	switch (ac.echoScope)
	{
	case glades::ATLASConfig::ECHO_SCOPE_ALL:
		return true;
	case glades::ATLASConfig::ECHO_SCOPE_LARGE_ONLY:
		return echo_scope_large_matrix(rows, cols);
	default:
		return false;
	}
}

static bool echo_scope_uses_decoder_matrix(const glades::ATLASConfig& ac,
                                           unsigned int blockIndex,
                                           unsigned int layerCount,
                                           unsigned int rows,
                                           unsigned int cols)
{
	switch (ac.echoScope)
	{
	case glades::ATLASConfig::ECHO_SCOPE_ALL:
		return true;
	case glades::ATLASConfig::ECHO_SCOPE_LARGE_ONLY:
		return echo_scope_large_matrix(rows, cols);
	case glades::ATLASConfig::ECHO_SCOPE_LATE_HEAD:
		return (layerCount > 0u) && (blockIndex + 1u == layerCount);
	case glades::ATLASConfig::ECHO_SCOPE_LATE_HEAD_LARGE:
		return (layerCount > 0u) && (blockIndex + 1u == layerCount)
		    && echo_scope_large_matrix(rows, cols);
	default:
		return false;
	}
}

static bool observe_echo_cpu(glades::atlas::EchoWeightState& state,
                             const float* rowObs,
                             const float* colObs,
                             unsigned int samples,
                             unsigned int rows,
                             unsigned int cols,
                             const glades::ATLASConfig& ac,
                             bool enabled)
{
	if (!ac.echoEnabled || !enabled)
		return true;
	return glades::atlas::echoObserve(state, rowObs, colObs, samples, rows, cols, ac);
}

#ifdef GLADES_HAVE_CUDA
static bool observe_echo_gpu(glades::gpu::GpuEchoWeightState& state,
                             const float* d_rowObs,
                             const float* d_colObs,
                             unsigned int samples,
                             unsigned int rows,
                             unsigned int cols,
                             const glades::ATLASConfig& ac,
                             bool enabled)
{
	if (!ac.echoEnabled || !enabled)
		return true;
	return glades::gpu::echo_gpu_observe(state, d_rowObs, d_colObs, samples, rows, cols, ac);
}

struct MuonBatchGroup
{
	unsigned int rows;
	unsigned int cols;
	std::vector<glades::gpu::GpuMuonBatchItem> items;

	MuonBatchGroup()
	    : rows(0u), cols(0u), items()
	{
	}
};

struct MatraBatchGroup
{
	unsigned int rows;
	unsigned int cols;
	std::vector<glades::gpu::GpuMatraBatchItem> items;

	MatraBatchGroup()
	    : rows(0u), cols(0u), items()
	{
	}
};

static unsigned int argos_role_flags_for_head()
{
	return glades::atlas::ARGOS_ROLE_HEAD;
}

static unsigned int argos_role_flags_for_block(unsigned int blockIndex,
                                               unsigned int nLayers)
{
	unsigned int flags = glades::atlas::ARGOS_ROLE_NONE;
	if ((blockIndex + 1u) == nLayers)
		flags |= glades::atlas::ARGOS_ROLE_LATE;
	return flags;
}

static bool run_argos_gpu(glades::gpu::GpuArgosWeightState& state,
                          glades::gpu::GpuBuffer<float>& param,
                          glades::gpu::GpuBuffer<float>& grad,
                          glades::gpu::GpuBuffer<float>& m1,
                          glades::gpu::GpuBuffer<float>& v2,
                          unsigned int rows,
                          unsigned int cols,
                          float lr,
                          float invBatch,
                          float gradScale,
                          float inv1mB1t,
                          float inv1mB2t,
                          float adamEps,
                          const glades::ATLASConfig& ac,
                          unsigned int roleFlags,
                          shmea::GLogger* logger,
                          const char* tag)
{
	if (param.size() == 0u)
		return true;
	return glades::gpu::argos_gpu_update_with_role(state,
	                                               param.data(), grad.data(),
	                                               m1.data(), v2.data(),
	                                               rows, cols, lr,
	                                               invBatch, gradScale,
	                                               inv1mB1t, inv1mB2t, adamEps,
	                                               ac,
	                                               roleFlags,
	                                               logger, tag);
}

static bool queue_or_run_matra_gpu(std::vector<MatraBatchGroup>& groups,
                                   glades::gpu::GpuTransformerWeights* gpuTransformerWeights,
                                   glades::gpu::GpuMatraWeightState& state,
                                   glades::gpu::GpuBuffer<float>& param,
                                   glades::gpu::GpuBuffer<float>& grad,
                                   glades::gpu::GpuBuffer<float>& m1,
                                   glades::gpu::GpuBuffer<float>& v2,
                                   unsigned int rows,
                                   unsigned int cols,
                                   float baseLr,
                                   float lrScale,
                                   float invBatch,
                                   float gradScale,
                                   float inv1mB1t,
                                   float inv1mB2t,
                                   float adamEps,
                                   const glades::ATLASConfig& ac,
                                   shmea::GLogger* logger,
                                   const char* tag)
{
	if (param.size() == 0u)
		return true;

	const bool batchEligible =
	    matra_batch_eligible(ac, rows, cols)
	    && (std::min(rows, cols) <= 64u)
	    && (gpuTransformerWeights != NULL)
	    && (gpuTransformerWeights->d_matraBatchItems != NULL)
	    && (gpuTransformerWeights->d_matraCoreBatchPtrs != NULL)
	    && (gpuTransformerWeights->d_matraStepBatchPtrs != NULL)
	    && (gpuTransformerWeights->d_matraInfoBatch != NULL)
	    && (gpuTransformerWeights->matraCoreBatchCapacity > 0);
	if (!batchEligible)
	{
		return glades::gpu::matra_gpu_update(state, param.data(), grad.data(),
		                                     m1.data(), v2.data(),
		                                     rows, cols, baseLr * lrScale,
		                                     invBatch, gradScale,
		                                     inv1mB1t, inv1mB2t, adamEps,
		                                     ac, logger, tag);
	}

	glades::gpu::GpuMatraBatchItem item;
	item.state = &state;
	item.d_W = param.data();
	item.d_gW = grad.data();
	item.d_m = m1.data();
	item.d_v = v2.data();
	item.m = rows;
	item.n = cols;
	item.lr = baseLr;
	item.tag = tag;

	for (size_t gi = 0; gi < groups.size(); ++gi)
	{
		if (groups[gi].rows == rows && groups[gi].cols == cols)
		{
			groups[gi].items.push_back(item);
			return true;
		}
	}

	MatraBatchGroup group;
	group.rows = rows;
	group.cols = cols;
	group.items.push_back(item);
	groups.push_back(group);
	return true;
}

static bool queue_or_run_muon_gpu(std::vector<MuonBatchGroup>& groups,
                                  glades::gpu::GpuTransformerWeights* gpuTransformerWeights,
                                  glades::gpu::GpuMuonWeightState& state,
                                  glades::gpu::GpuBuffer<float>& param,
                                  glades::gpu::GpuBuffer<float>& grad,
                                  glades::gpu::GpuBuffer<float>& m1,
                                  glades::gpu::GpuBuffer<float>& v2,
                                  unsigned int rows,
                                  unsigned int cols,
                                  float lr,
                                  float inv1mB1t,
                                  float inv1mB2t,
                                  float adamEps,
                                  const glades::ATLASConfig& ac,
                                  shmea::GLogger* logger,
                                  const char* tag)
{
	if (param.size() == 0u)
		return true;
	if (!muon_matrix_eligible(ac, rows, cols))
		return true;

	const bool batchEligible =
	    (rows >= cols)
	    && (cols <= 64u)
	    && (gpuTransformerWeights != NULL)
	    && (gpuTransformerWeights->d_muonCoreBatchPtrs != NULL)
	    && (gpuTransformerWeights->muonCoreBatchCapacity > 0);
	if (!batchEligible)
	{
		return glades::gpu::muon_gpu_update_lite(state, param.data(), grad.data(),
		                                         m1.data(), v2.data(),
		                                         rows, cols, lr,
		                                         inv1mB1t, inv1mB2t, adamEps,
		                                         ac, logger, tag);
	}

	glades::gpu::GpuMuonBatchItem item;
	item.state = &state;
	item.d_W = param.data();
	item.d_gW = grad.data();
	item.d_m = m1.data();
	item.d_v = v2.data();
	item.m = rows;
	item.n = cols;
	item.lr = lr;
	item.tag = tag;

	for (size_t gi = 0; gi < groups.size(); ++gi)
	{
		if (groups[gi].rows == rows && groups[gi].cols == cols)
		{
			groups[gi].items.push_back(item);
			return true;
		}
	}

	MuonBatchGroup group;
	group.rows = rows;
	group.cols = cols;
	group.items.push_back(item);
	groups.push_back(group);
	return true;
}
#endif

static bool invert_small_dense_row_major(const std::vector<float>& matrix,
                                         unsigned int dim,
                                         float ridge,
                                         std::vector<float>& inverseOut)
{
	if (dim == 0u || matrix.size() < static_cast<size_t>(dim) * static_cast<size_t>(dim))
		return false;

	const unsigned int augCols = dim * 2u;
	std::vector<double> aug(static_cast<size_t>(dim) * static_cast<size_t>(augCols), 0.0);
	for (unsigned int r = 0; r < dim; ++r)
	{
		for (unsigned int c = 0; c < dim; ++c)
		{
			double v = static_cast<double>(matrix[static_cast<size_t>(r) * static_cast<size_t>(dim) + c]);
			if (r == c)
				v += static_cast<double>(ridge);
			aug[static_cast<size_t>(r) * static_cast<size_t>(augCols) + c] = v;
		}
		aug[static_cast<size_t>(r) * static_cast<size_t>(augCols) + (dim + r)] = 1.0;
	}

	for (unsigned int col = 0; col < dim; ++col)
	{
		unsigned int pivot = col;
		double pivotAbs = std::fabs(aug[static_cast<size_t>(pivot) * static_cast<size_t>(augCols) + col]);
		for (unsigned int row = col + 1u; row < dim; ++row)
		{
			const double candAbs = std::fabs(aug[static_cast<size_t>(row) * static_cast<size_t>(augCols) + col]);
			if (candAbs > pivotAbs)
			{
				pivot = row;
				pivotAbs = candAbs;
			}
		}
		if (!(pivotAbs > 1e-12))
			return false;
		if (pivot != col)
		{
			for (unsigned int c = 0; c < augCols; ++c)
				std::swap(aug[static_cast<size_t>(pivot) * static_cast<size_t>(augCols) + c],
				          aug[static_cast<size_t>(col) * static_cast<size_t>(augCols) + c]);
		}
		const double invPivot = 1.0 / aug[static_cast<size_t>(col) * static_cast<size_t>(augCols) + col];
		for (unsigned int c = 0; c < augCols; ++c)
			aug[static_cast<size_t>(col) * static_cast<size_t>(augCols) + c] *= invPivot;
		for (unsigned int row = 0; row < dim; ++row)
		{
			if (row == col)
				continue;
			const double scale = aug[static_cast<size_t>(row) * static_cast<size_t>(augCols) + col];
			if (std::fabs(scale) <= 1e-18)
				continue;
			for (unsigned int c = 0; c < augCols; ++c)
			{
				aug[static_cast<size_t>(row) * static_cast<size_t>(augCols) + c] -=
				    scale * aug[static_cast<size_t>(col) * static_cast<size_t>(augCols) + c];
			}
		}
	}

	inverseOut.assign(static_cast<size_t>(dim) * static_cast<size_t>(dim), 0.0f);
	for (unsigned int r = 0; r < dim; ++r)
	{
		for (unsigned int c = 0; c < dim; ++c)
		{
			inverseOut[static_cast<size_t>(r) * static_cast<size_t>(dim) + c] =
			    static_cast<float>(aug[static_cast<size_t>(r) * static_cast<size_t>(augCols) + (dim + c)]);
		}
	}
	return true;
}

static void extract_top_singular_modes_row_major(const std::vector<float>& matrix,
                                                 unsigned int rows,
                                                 unsigned int cols,
                                                 unsigned int rank,
                                                 std::vector<float>& sigmaOut,
                                                 std::vector<float>& leftOut,
                                                 std::vector<float>& rightOut)
{
	sigmaOut.assign(rank, 0.0f);
	leftOut.assign(static_cast<size_t>(rank) * static_cast<size_t>(rows), 0.0f);
	rightOut.assign(static_cast<size_t>(rank) * static_cast<size_t>(cols), 0.0f);
	if (rows == 0u || cols == 0u || rank == 0u
	    || matrix.size() < static_cast<size_t>(rows) * static_cast<size_t>(cols))
		return;

	std::vector<float> gram(static_cast<size_t>(rows) * static_cast<size_t>(rows), 0.0f);
	for (unsigned int r1 = 0; r1 < rows; ++r1)
	{
		for (unsigned int r2 = 0; r2 < rows; ++r2)
		{
			double accum = 0.0;
			const size_t off1 = static_cast<size_t>(r1) * static_cast<size_t>(cols);
			const size_t off2 = static_cast<size_t>(r2) * static_cast<size_t>(cols);
			for (unsigned int c = 0; c < cols; ++c)
				accum += static_cast<double>(matrix[off1 + c]) * static_cast<double>(matrix[off2 + c]);
			gram[static_cast<size_t>(r1) * static_cast<size_t>(rows) + r2] = static_cast<float>(accum);
		}
	}

	for (unsigned int m = 0; m < rank; ++m)
	{
		std::vector<float> leftVec(rows, 0.0f);
		leftVec[(m < rows) ? m : 0u] = 1.0f;
		bool valid = true;
		for (unsigned int iter = 0; iter < 6u && valid; ++iter)
		{
			std::vector<float> nextVec(rows, 0.0f);
			for (unsigned int r = 0; r < rows; ++r)
			{
				double accum = 0.0;
				for (unsigned int c = 0; c < rows; ++c)
					accum += static_cast<double>(gram[static_cast<size_t>(r) * static_cast<size_t>(rows) + c])
					      * static_cast<double>(leftVec[c]);
				nextVec[r] = static_cast<float>(accum);
			}
			for (unsigned int pm = 0; pm < m; ++pm)
			{
				const size_t prevOff = static_cast<size_t>(pm) * static_cast<size_t>(rows);
				double proj = 0.0;
				for (unsigned int r = 0; r < rows; ++r)
					proj += static_cast<double>(nextVec[r]) * static_cast<double>(leftOut[prevOff + r]);
				for (unsigned int r = 0; r < rows; ++r)
					nextVec[r] -= static_cast<float>(proj * static_cast<double>(leftOut[prevOff + r]));
			}
			double normSq = 0.0;
			for (unsigned int r = 0; r < rows; ++r)
				normSq += static_cast<double>(nextVec[r]) * static_cast<double>(nextVec[r]);
			if (!(normSq > 1e-18))
			{
				valid = false;
				break;
			}
			const float invNorm = 1.0f / static_cast<float>(sqrt(normSq));
			for (unsigned int r = 0; r < rows; ++r)
				leftVec[r] = nextVec[r] * invNorm;
		}
		if (!valid)
			continue;

		std::vector<float> gramLeft(rows, 0.0f);
		for (unsigned int r = 0; r < rows; ++r)
		{
			double accum = 0.0;
			for (unsigned int c = 0; c < rows; ++c)
				accum += static_cast<double>(gram[static_cast<size_t>(r) * static_cast<size_t>(rows) + c])
				      * static_cast<double>(leftVec[c]);
			gramLeft[r] = static_cast<float>(accum);
		}
		double lambda = 0.0;
		for (unsigned int r = 0; r < rows; ++r)
			lambda += static_cast<double>(leftVec[r]) * static_cast<double>(gramLeft[r]);
		const float sigma = static_cast<float>(sqrt(std::max(0.0, lambda)));
		if (!(sigma > 1e-12f))
			continue;

		const size_t leftOff = static_cast<size_t>(m) * static_cast<size_t>(rows);
		for (unsigned int r = 0; r < rows; ++r)
			leftOut[leftOff + r] = leftVec[r];
		sigmaOut[m] = sigma;

		std::vector<float> rightVec(cols, 0.0f);
		for (unsigned int c = 0; c < cols; ++c)
		{
			double accum = 0.0;
			for (unsigned int r = 0; r < rows; ++r)
				accum += static_cast<double>(matrix[static_cast<size_t>(r) * static_cast<size_t>(cols) + c])
				      * static_cast<double>(leftVec[r]);
			rightVec[c] = static_cast<float>(accum / sigma);
		}
		for (unsigned int pm = 0; pm < m; ++pm)
		{
			const size_t prevOff = static_cast<size_t>(pm) * static_cast<size_t>(cols);
			double proj = 0.0;
			for (unsigned int c = 0; c < cols; ++c)
				proj += static_cast<double>(rightVec[c]) * static_cast<double>(rightOut[prevOff + c]);
			for (unsigned int c = 0; c < cols; ++c)
				rightVec[c] -= static_cast<float>(proj * static_cast<double>(rightOut[prevOff + c]));
		}
		double rightNormSq = 0.0;
		for (unsigned int c = 0; c < cols; ++c)
			rightNormSq += static_cast<double>(rightVec[c]) * static_cast<double>(rightVec[c]);
		if (!(rightNormSq > 1e-18))
		{
			sigmaOut[m] = 0.0f;
			for (unsigned int r = 0; r < rows; ++r)
				leftOut[leftOff + r] = 0.0f;
			continue;
		}
		const float invNorm = 1.0f / static_cast<float>(sqrt(rightNormSq));
		const size_t rightOff = static_cast<size_t>(m) * static_cast<size_t>(cols);
		for (unsigned int c = 0; c < cols; ++c)
			rightOut[rightOff + c] = rightVec[c] * invNorm;
	}
}

static unsigned int aster_mix_u32(unsigned int x)
{
	x ^= x >> 16;
	x *= 0x7feb352dU;
	x ^= x >> 15;
	x *= 0x846ca68bU;
	x ^= x >> 16;
	return x;
}

static void aster_sketch_dense_row(const float* row,
                                   unsigned int rowDim,
                                   unsigned int sketchDim,
                                   unsigned int seed,
                                   std::vector<float>& accum,
                                   size_t accumOffset = 0u)
{
	if (!row || sketchDim == 0u)
		return;
	for (unsigned int i = 0; i < rowDim; ++i)
	{
		const float v = row[i];
		if (v == 0.0f)
			continue;
		const unsigned int h = aster_mix_u32(seed + i * 0x9e3779b9U);
		const unsigned int bucket = h % sketchDim;
		const float sign = ((h >> 31) != 0u) ? -1.0f : 1.0f;
		accum[accumOffset + bucket] += sign * v;
	}
}

static inline void aster_sketch_sparse_value(unsigned int index,
                                             float value,
                                             unsigned int sketchDim,
                                             unsigned int seed,
                                             std::vector<float>& accum,
                                             size_t accumOffset = 0u)
{
	if (sketchDim == 0u || value == 0.0f)
		return;
	const unsigned int h = aster_mix_u32(seed + index * 0x9e3779b9U);
	const unsigned int bucket = h % sketchDim;
	const float sign = ((h >> 31) != 0u) ? -1.0f : 1.0f;
	accum[accumOffset + bucket] += sign * value;
}

static void aster_insert_top_support(unsigned int vid,
                                     float residual,
                                     float logit,
                                     std::vector<unsigned int>& ids,
                                     std::vector<float>& residuals,
                                     std::vector<float>& logits)
{
	if (ids.empty() || residual <= 0.0f || logits.size() != ids.size())
		return;
	const size_t n = ids.size();
	size_t pos = n;
	for (size_t i = 0; i < n; ++i)
	{
		if (residual > residuals[i])
		{
			pos = i;
			break;
		}
	}
	if (pos == n)
		return;
	for (size_t i = n - 1u; i > pos; --i)
	{
		ids[i] = ids[i - 1u];
		residuals[i] = residuals[i - 1u];
		logits[i] = logits[i - 1u];
	}
	ids[pos] = vid;
	residuals[pos] = residual;
	logits[pos] = logit;
}

static void aster_accumulate_support_role(unsigned int vid,
                                          unsigned int regime,
                                          unsigned int regimeCount,
                                          unsigned int role,
                                          unsigned int supportDim,
                                          float roleValue,
                                          const float* hiddenRow,
                                          unsigned int dModel,
                                          std::vector<float>& batchSupportValueSum,
                                          std::vector<float>& batchSupportCount,
                                          std::vector<float>& batchSupportHiddenRawSum,
                                          std::vector<unsigned int>& batchSupportIds,
                                          std::vector<float>& batchSupportRoleCounts)
{
	if (role >= supportDim || regime >= regimeCount)
		return;
	const size_t roleIndex = static_cast<size_t>(regime) * static_cast<size_t>(supportDim) + role;
	const size_t hiddenRoleOff =
	    (static_cast<size_t>(regime) * static_cast<size_t>(supportDim) + role) * static_cast<size_t>(dModel);
	if (roleIndex >= batchSupportValueSum.size()
	    || roleIndex >= batchSupportCount.size()
	    || (hiddenRoleOff + static_cast<size_t>(dModel)) > batchSupportHiddenRawSum.size())
		return;

	batchSupportValueSum[roleIndex] += roleValue;
	batchSupportCount[roleIndex] += 1.0f;
	if (hiddenRow)
	{
		for (unsigned int i = 0; i < dModel; ++i)
			batchSupportHiddenRawSum[hiddenRoleOff + i] += hiddenRow[i];
	}

	size_t slot = batchSupportIds.size();
	for (size_t i = 0; i < batchSupportIds.size(); ++i)
	{
		if (batchSupportIds[i] == vid)
		{
			slot = i;
			break;
		}
	}
	if (slot == batchSupportIds.size())
	{
		batchSupportIds.push_back(vid);
		batchSupportRoleCounts.resize((slot + 1u) * static_cast<size_t>(regimeCount) * static_cast<size_t>(supportDim), 0.0f);
	}
	batchSupportRoleCounts[(slot * static_cast<size_t>(regimeCount) + regime) * static_cast<size_t>(supportDim) + role] += 1.0f;
}

static inline void aster_accumulate_support_value_only(unsigned int regime,
                                                       unsigned int role,
                                                       unsigned int supportDim,
                                                       float value,
                                                       std::vector<float>& batchSupportValueSum)
{
	if (role >= supportDim)
		return;
	const size_t roleIndex = static_cast<size_t>(regime) * static_cast<size_t>(supportDim) + role;
	if (roleIndex >= batchSupportValueSum.size())
		return;
	batchSupportValueSum[roleIndex] += value;
}

static inline float aster_segment_mean(const float* row,
                                       unsigned int dim,
                                       unsigned int rank,
                                       unsigned int slot)
{
	if (!row || dim == 0u || rank == 0u)
		return 0.0f;
	const unsigned int begin = (slot * dim) / rank;
	const unsigned int end = std::max(begin + 1u, ((slot + 1u) * dim) / rank);
	if (begin >= dim)
		return 0.0f;
	const unsigned int clampedEnd = std::min(end, dim);
	double accum = 0.0;
	for (unsigned int i = begin; i < clampedEnd; ++i)
		accum += static_cast<double>(row[i]);
	return static_cast<float>(accum / static_cast<double>(std::max(1u, clampedEnd - begin)));
}

static unsigned int aster_transformer_regime_from_logits(const float* probsRow,
                                                         const float* logitsRow,
                                                         unsigned int count,
                                                         unsigned int targetIndex,
                                                         int padIndex)
{
	if (!probsRow || !logitsRow || count == 0u || targetIndex >= count)
		return 0u;
	double entropy = 0.0;
	double maxEntropy = 0.0;
	float maxNegLogit = -std::numeric_limits<float>::max();
	for (unsigned int i = 0; i < count; ++i)
	{
		if (padIndex >= 0 && static_cast<int>(i) == padIndex)
			continue;
		const float p = probsRow[i];
		if (p > 1e-20f)
			entropy -= static_cast<double>(p) * std::log(static_cast<double>(p));
		maxEntropy += 1.0;
		if (i != targetIndex)
			maxNegLogit = std::max(maxNegLogit, logitsRow[i]);
	}
	if (maxEntropy > 1.0)
		maxEntropy = std::log(maxEntropy);
	else
		maxEntropy = 0.0;
	const bool highEntropy = (maxEntropy > 0.0) && (entropy > (0.75 * maxEntropy));
	const float margin = logitsRow[targetIndex] - maxNegLogit;
	const bool lowMargin = !(margin > 0.75f);
	return (highEntropy ? 2u : 0u) + (lowMargin ? 1u : 0u);
}

static glades::transformer_train_detail::LinearWeightView make_linear_weight_view(const std::vector<float>& weights,
                                                                                  const std::vector<uint16_t>& lowpWeights,
                                                                                  const std::vector<float>& bias,
                                                                                  bool useLowpWeights,
                                                                                  int lowpDType)
{
	glades::transformer_train_detail::LinearWeightView view;
	view.weights = weights.empty() ? NULL : &weights[0];
	view.weightCount = static_cast<unsigned int>(weights.size());
	view.lowpWeights = lowpWeights.empty() ? NULL : &lowpWeights[0];
	view.lowpWeightCount = static_cast<unsigned int>(lowpWeights.size());
	view.bias = bias.empty() ? NULL : &bias[0];
	view.biasCount = static_cast<unsigned int>(bias.size());
	view.useLowpWeights = useLowpWeights;
	view.lowpDType = lowpDType;
	return view;
}

static glades::transformer_train_detail::DoubleBufferView make_double_buffer_view(const std::vector<double>& values)
{
	glades::transformer_train_detail::DoubleBufferView view;
	view.data = values.empty() ? NULL : &values[0];
	view.size = static_cast<unsigned int>(values.size());
	return view;
}

#ifdef GLADES_HAVE_CUDA
static glades::gpu::HostFloatBufferView make_host_float_buffer_view(std::vector<float>& values)
{
	glades::gpu::HostFloatBufferView view;
	view.data = values.empty() ? NULL : &values[0];
	view.size = values.size();
	return view;
}

static glades::gpu::TransformerHostBlockWeightsView make_transformer_host_block_view(std::vector<float>& ln1Gamma,
                                                                                     std::vector<float>& ln1Beta,
                                                                                     std::vector<float>& Wq,
                                                                                     std::vector<float>& Wk,
                                                                                     std::vector<float>& Wv,
                                                                                     std::vector<float>& Wo,
                                                                                     std::vector<float>& bq,
                                                                                     std::vector<float>& bk,
                                                                                     std::vector<float>& bv,
                                                                                     std::vector<float>& bo,
                                                                                     std::vector<float>& ln2Gamma,
                                                                                     std::vector<float>& ln2Beta,
                                                                                     std::vector<float>& W1,
                                                                                     std::vector<float>& W2,
                                                                                     std::vector<float>& b1,
                                                                                     std::vector<float>& b2)
{
	glades::gpu::TransformerHostBlockWeightsView view;
	view.ln1Gamma = make_host_float_buffer_view(ln1Gamma);
	view.ln1Beta = make_host_float_buffer_view(ln1Beta);
	view.Wq = make_host_float_buffer_view(Wq);
	view.Wk = make_host_float_buffer_view(Wk);
	view.Wv = make_host_float_buffer_view(Wv);
	view.Wo = make_host_float_buffer_view(Wo);
	view.bq = make_host_float_buffer_view(bq);
	view.bk = make_host_float_buffer_view(bk);
	view.bv = make_host_float_buffer_view(bv);
	view.bo = make_host_float_buffer_view(bo);
	view.ln2Gamma = make_host_float_buffer_view(ln2Gamma);
	view.ln2Beta = make_host_float_buffer_view(ln2Beta);
	view.W1 = make_host_float_buffer_view(W1);
	view.W2 = make_host_float_buffer_view(W2);
	view.b1 = make_host_float_buffer_view(b1);
	view.b2 = make_host_float_buffer_view(b2);
	return view;
}
#endif

static void log_transformer_epoch_progress(shmea::GLogger* logger,
                                           int netType,
                                           bool isTrain,
                                           int epochIdx,
                                           unsigned int seqDone,
                                           unsigned int seqCount,
                                           int64_t nowMs,
                                           int64_t epochStartMs,
                                           unsigned long long tokensProcessed,
                                           unsigned long long targetsProcessed,
                                           bool tokenLM,
                                           glades::TransformerRunConfig::TokenLMLossKind tokenLmLossKind,
                                           double tokenLmNllSum,
                                           unsigned long long tokenLmTokenCount,
                                           unsigned long long clsCorrect,
                                           unsigned long long clsTotal,
                                           float lossSoFar,
                                           float lrScheduleMultiplier,
                                           float globalGradClipNorm,
                                           float lastGradNorm,
                                           float lastGradNormScale,
                                           bool mpUseLossScaling,
                                           float mpLossScale,
                                           unsigned long long optimizerStep)
{
	if (!logger)
		return;

	const double elapsedMs = static_cast<double>(nowMs - epochStartMs);
	const double tokPerSec = (elapsedMs > 0.0) ? (static_cast<double>(targetsProcessed) / (elapsedMs / 1000.0)) : 0.0;

	std::ostringstream oss;
	oss << "event=nn_epoch_progress";
	append_logfmt_kv(oss, "net_type", netType);
	append_logfmt_kv(oss, "run_type", std::string(isTrain ? "train" : "eval"));
	append_logfmt_kv(oss, "epoch", epochIdx);
	append_logfmt_kv(oss, "seq_done", seqDone);
	append_logfmt_kv(oss, "seq_total", seqCount);
	append_logfmt_kv(oss, "tokens_seen", tokensProcessed);
	append_logfmt_kv(oss, "targets_seen", targetsProcessed);
	append_logfmt_kv(oss, "targets_per_sec", tokPerSec);
	if (tokenLM)
	{
		const bool tokenLmFullSoftmax = (tokenLmLossKind == glades::TransformerRunConfig::TOKEN_LM_FULL_SOFTMAX);
		append_logfmt_kv(oss, "token_lm_loss_kind", std::string(tokenLmFullSoftmax ? "full_softmax" : "sampled_softmax"));
		const double meanNll = (tokenLmTokenCount > 0ULL) ? (tokenLmNllSum / static_cast<double>(tokenLmTokenCount)) : 0.0;
		append_logfmt_kv(oss, "nll", meanNll);
		if (tokenLmFullSoftmax)
		{
			double ppl = 0.0;
			if (tokenLmTokenCount > 0ULL)
			{
				double arg = meanNll;
				if (arg > 80.0) arg = 80.0;
				if (arg < -80.0) arg = -80.0;
				ppl = exp(arg);
			}
			append_logfmt_kv(oss, "perplexity", ppl);
			append_logfmt_kv(oss, "acc_top1", (clsTotal > 0ULL) ? (100.0 * static_cast<double>(clsCorrect) / static_cast<double>(clsTotal)) : 0.0);
		}
		else
		{
			append_logfmt_kv(oss, "perplexity", std::string("na"));
			append_logfmt_kv(oss, "acc_top1", std::string("na"));
		}
	}
	else
	{
		append_logfmt_kv(oss, "loss_so_far", lossSoFar);
	}
	append_logfmt_kv(oss, "lr_mult", lrScheduleMultiplier);
	if (globalGradClipNorm > 0.0f)
	{
		append_logfmt_kv(oss, "grad_norm", lastGradNorm);
		append_logfmt_kv(oss, "grad_norm_scale", lastGradNormScale);
	}
	else
	{
		append_logfmt_kv(oss, "grad_norm", std::string("na"));
		append_logfmt_kv(oss, "grad_norm_scale", std::string("na"));
	}
	append_logfmt_kv(oss, "optimizer_step", optimizerStep);
	if (mpUseLossScaling)
		append_logfmt_kv(oss, "loss_scale", mpLossScale);

	logger->info("NNetwork", shmea::GString(oss.str().c_str()));
}

} // namespace

void glades::NNetwork::SGDHelper_TRANSFORMER(unsigned int inputRowCounter, int runType)
{
	using namespace glades::sgd_detail;
	(void)inputRowCounter;

	const bool isTrain = (runType == RUN_TRAIN);
	const unsigned int dataSize = isTrain ? (di ? di->getTrainSize() : 0u) : (di ? di->getTestSize() : 0u);

	// Only run once per epoch (Trainer loops over steps=1 for sequence models).
	if (inputRowCounter != 0u)
		return;

	// Epoch-local progress logging (rate-limited by a fixed number of updates per epoch).
	// This keeps long-running LLM epochs from appearing "stuck" while avoiding log spam.
	shmea::GLogger* logger = getLogger();
	const int64_t epochStartMs = getCurrentTimeMilliseconds();
	const int epochIdx = epochs; // current epoch number for this run (Trainer increments after SGDHelper returns)

	// Transformers do not use the graph's dropout masks (no node-level dropout here).
	// (legacy graph dropout removed)

	const unsigned int seqCount = di ? (isTrain ? di->getTrainSequenceCount() : di->getTestSequenceCount()) : 0u;
	if (seqCount == 0u)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE,
		                            isTrain ? "SGDHelper_TRANSFORMER: no train sequences"
		                                    : "SGDHelper_TRANSFORMER: no test sequences");
		storeRunningFlag(false);
		return;
	}

	// Ensure transformer parameters exist.
	if (!ensureTensorParametersInitialized())
	{
		storeRunningFlag(false);
		return;
	}

	const TensorTransformerState& ttConst = tensorTransformer;
	const unsigned int inputSize = ttConst.inputSize;
	const unsigned int outSize = ttConst.outSize;
	const unsigned int dModel = ttConst.dModel;
	const unsigned int dFF = ttConst.dFF;
	const unsigned int nHeads = ttConst.nHeads;
	const unsigned int nLayers = ttConst.nLayers;
	const bool causal = ttConst.causal;
	const bool tokenLM = ttConst.tokenModel;
	const unsigned int vocabSize = ttConst.vocabSize;
	const int padTokenId = ttConst.padTokenId;
	const bool tieEmb = ttConst.tieEmbeddings;

	if (inputSize == 0u || outSize == 0u || dModel == 0u || dFF == 0u || nHeads == 0u || nLayers == 0u)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "SGDHelper_TRANSFORMER: invalid transformer state sizes");
		storeRunningFlag(false);
		return;
	}
	if (tokenLM)
	{
		if (!tieEmb)
		{
			lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "SGDHelper_TRANSFORMER: token LM mode requires tieEmbeddings=true");
			storeRunningFlag(false);
			return;
		}
		if (inputSize != 1u)
		{
			lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "SGDHelper_TRANSFORMER: token LM mode requires inputSize==1 (token id)");
			storeRunningFlag(false);
			return;
		}
		if (outSize != vocabSize || vocabSize == 0u)
		{
			lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "SGDHelper_TRANSFORMER: token LM mode requires outSize==vocabSize>0");
			storeRunningFlag(false);
			return;
		}
		if (tieEmb && (ttConst.tokE.size() != static_cast<size_t>(vocabSize) * static_cast<size_t>(dModel)))
		{
			lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "SGDHelper_TRANSFORMER: token LM embedding table is not initialized");
			storeRunningFlag(false);
			return;
		}
	}

	NNetworkStatus cfgStatus = validateTransformerTrainingConfig("SGDHelper_TRANSFORMER", trainingConfig);
	if (!cfgStatus.ok())
	{
		lastStatus = cfgStatus;
		storeRunningFlag(false);
		return;
	}
	TransformerRuntimeConfigSnapshot runtimeCfg;
	cfgStatus = buildTransformerRuntimeConfigSnapshot("SGDHelper_TRANSFORMER", trainingConfig.transformer, runtimeCfg);
	if (!cfgStatus.ok())
	{
		lastStatus = cfgStatus;
		storeRunningFlag(false);
		return;
	}

	const float gradClip = trainingConfig.perElementGradClip;
	const int costFx = skeleton->getOutputType();
	const float lnEps = runtimeCfg.layerNormEps;
	const int posEnc = static_cast<int>(runtimeCfg.positionalEncoding);
	const int normType = static_cast<int>(runtimeCfg.normType);
	const int ffnKind = static_cast<int>(runtimeCfg.ffnKind);
	const int ffnAct = static_cast<int>(runtimeCfg.ffnActivation);
	const float ropeTheta = runtimeCfg.ropeTheta;
	const int ropeDimOverride = runtimeCfg.ropeDimOverride;
	const glades::TransformerRunConfig::TokenLMLossKind tokenLmLossKind = trainingConfig.transformer.tokenLmLossKind;
	const int tokenLmNegK = trainingConfig.transformer.tokenLmSampledNegatives;
	const bool tokenLmAllowHuge = trainingConfig.transformer.tokenLmAllowHugeFullSoftmax;
	const bool ddpEnabled = trainingConfig.ddp.enable && (glades::ddp::worldSize() > 1);

	// Trainability triage:
	// - For real LLMs, AdamW or ATLAS are the supported optimizers in this backend.
	// - Full softmax is guarded to avoid silently allocating/computing O(T*vocab) buffers.
	if (tokenLM && isTrain && (trainingConfig.optimizer.type != glades::OptimizerConfig::ADAMW)
	                       && (trainingConfig.optimizer.type != glades::OptimizerConfig::ATLAS)
	                       && (trainingConfig.optimizer.type != glades::OptimizerConfig::VESTA)
	                       && (trainingConfig.optimizer.type != glades::OptimizerConfig::HELIOS)
	                       && (trainingConfig.optimizer.type != glades::OptimizerConfig::SOPHIA_G))
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "SGDHelper_TRANSFORMER: token LM training requires optimizer=ADAMW, ATLAS, VESTA, HELIOS, or SOPHIA_G for LLM-scale stability");
		storeRunningFlag(false);
		return;
	}
	// Reset per-epoch bookkeeping
	results.clear();
	if (isTrain)
		nbRecord.clear();

	// Minibatch: number of sequences to accumulate before applying an update.
	const unsigned int seqBatchMax = (minibatchSize > 0 ? static_cast<unsigned int>(minibatchSize) : 1u);
	const unsigned int optimizerStepsPerEpoch = (seqCount + seqBatchMax - 1u) / seqBatchMax;
	unsigned int seqInBatch = 0u;
	// Gradient averaging divisor for the minibatch:
	// - Non-tokenLM: total timesteps across sequences in the batch (as before).
	// - Token LM: total *valid target tokens* (non-pad) across sequences in the batch.
	//   This matches the loss normalization (mean NLL over non-pad targets).
	unsigned int timeStepsInBatch = 0u;

	// Helper: clear gradient accumulators when starting a new minibatch.
	struct ClearGrads
	{
		TensorTransformerState& tt;
		ClearGrads(TensorTransformerState& t) : tt(t) {}
		void operator()() const
		{
			std::fill(tt.gTokE.begin(), tt.gTokE.end(), 0.0f);
			std::fill(tt.gLmBias.begin(), tt.gLmBias.end(), 0.0f);
			std::fill(tt.gWIn.begin(), tt.gWIn.end(), 0.0f);
			std::fill(tt.gBIn.begin(), tt.gBIn.end(), 0.0f);
			std::fill(tt.gWOut.begin(), tt.gWOut.end(), 0.0f);
			std::fill(tt.gBOut.begin(), tt.gBOut.end(), 0.0f);
			std::fill(tt.gLnFinalGamma.begin(), tt.gLnFinalGamma.end(), 0.0f);
			std::fill(tt.gLnFinalBeta.begin(), tt.gLnFinalBeta.end(), 0.0f);
			for (size_t l = 0; l < tt.blocks.size(); ++l)
			{
				TensorTransformerState::Block& b = tt.blocks[l];
				std::fill(b.gLn1Gamma.begin(), b.gLn1Gamma.end(), 0.0f);
				std::fill(b.gLn1Beta.begin(), b.gLn1Beta.end(), 0.0f);
				std::fill(b.gWq.begin(), b.gWq.end(), 0.0f);
				std::fill(b.gWk.begin(), b.gWk.end(), 0.0f);
				std::fill(b.gWv.begin(), b.gWv.end(), 0.0f);
				std::fill(b.gWo.begin(), b.gWo.end(), 0.0f);
				std::fill(b.gBq.begin(), b.gBq.end(), 0.0f);
				std::fill(b.gBk.begin(), b.gBk.end(), 0.0f);
				std::fill(b.gBv.begin(), b.gBv.end(), 0.0f);
				std::fill(b.gBo.begin(), b.gBo.end(), 0.0f);
				std::fill(b.gLn2Gamma.begin(), b.gLn2Gamma.end(), 0.0f);
				std::fill(b.gLn2Beta.begin(), b.gLn2Beta.end(), 0.0f);
				std::fill(b.gW1.begin(), b.gW1.end(), 0.0f);
				std::fill(b.gW2.begin(), b.gW2.end(), 0.0f);
				std::fill(b.gB1.begin(), b.gB1.end(), 0.0f);
				std::fill(b.gB2.begin(), b.gB2.end(), 0.0f);
			}
			if (tt.helm.initialized)
			{
				std::fill(tt.helm.batchHiddenSum.begin(), tt.helm.batchHiddenSum.end(), 0.0f);
				std::fill(tt.helm.batchHiddenSqSum.begin(), tt.helm.batchHiddenSqSum.end(), 0.0f);
				std::fill(tt.helm.batchResidualSum.begin(), tt.helm.batchResidualSum.end(), 0.0f);
				std::fill(tt.helm.batchResidualSqSum.begin(), tt.helm.batchResidualSqSum.end(), 0.0f);
				tt.helm.batchTokenCount = 0u;
			}
			if (tt.aster.initialized)
			{
				std::fill(tt.aster.batchFinalHiddenRawSum.begin(), tt.aster.batchFinalHiddenRawSum.end(), 0.0f);
				std::fill(tt.aster.batchFinalHiddenSketchSum.begin(), tt.aster.batchFinalHiddenSketchSum.end(), 0.0f);
				std::fill(tt.aster.batchLayerHiddenRawSum.begin(), tt.aster.batchLayerHiddenRawSum.end(), 0.0f);
				std::fill(tt.aster.batchLayerHiddenSketchSum.begin(), tt.aster.batchLayerHiddenSketchSum.end(), 0.0f);
				std::fill(tt.aster.batchLayerAttnRawSum.begin(), tt.aster.batchLayerAttnRawSum.end(), 0.0f);
				std::fill(tt.aster.batchLayerAttnSketchSum.begin(), tt.aster.batchLayerAttnSketchSum.end(), 0.0f);
				std::fill(tt.aster.batchLayerPatternSum.begin(), tt.aster.batchLayerPatternSum.end(), 0.0f);
				std::fill(tt.aster.batchLayerKappaSum.begin(), tt.aster.batchLayerKappaSum.end(), 0.0f);
				std::fill(tt.aster.batchResidualSum.begin(), tt.aster.batchResidualSum.end(), 0.0f);
				std::fill(tt.aster.batchSupportLogitSum.begin(), tt.aster.batchSupportLogitSum.end(), 0.0f);
				std::fill(tt.aster.batchSupportResidualSum.begin(), tt.aster.batchSupportResidualSum.end(), 0.0f);
				std::fill(tt.aster.batchSupportCount.begin(), tt.aster.batchSupportCount.end(), 0.0f);
				std::fill(tt.aster.batchSupportHiddenRawSum.begin(), tt.aster.batchSupportHiddenRawSum.end(), 0.0f);
				std::fill(tt.aster.batchTargetMarginSum.begin(), tt.aster.batchTargetMarginSum.end(), 0.0f);
				std::fill(tt.aster.batchHardNegativeLogitSum.begin(), tt.aster.batchHardNegativeLogitSum.end(), 0.0f);
				std::fill(tt.aster.batchBaselineWorseSum.begin(), tt.aster.batchBaselineWorseSum.end(), 0.0f);
				std::fill(tt.aster.batchRegimeTokenCount.begin(), tt.aster.batchRegimeTokenCount.end(), 0.0f);
				tt.aster.batchTouchedIds.clear();
				tt.aster.batchSupportIds.clear();
				tt.aster.batchSupportRoleCounts.clear();
				tt.aster.batchTokenCount = 0u;
			}
		}
	};

	// Helper: apply accumulated gradients (averaged by timesteps) with momentum + optional global norm clip.
	struct ApplyBatch
	{
		NNetwork& net;
		TensorTransformerState& tt;
		unsigned int nLayers;
		unsigned int dModel;
		unsigned int outSize;

		ApplyBatch(NNetwork& n, TensorTransformerState& t, unsigned int nl, unsigned int dm, unsigned int os)
		    : net(n), tt(t), nLayers(nl), dModel(dm), outSize(os)
		{
		}

		bool operator()(unsigned int batchTimeSteps) const
		{
			using namespace glades::sgd_detail;
			if (batchTimeSteps == 0u)
				return true;

			const float invBatch = 1.0f / static_cast<float>(batchTimeSteps);
			const bool useAdamW = (net.trainingConfig.optimizer.type == glades::OptimizerConfig::ADAMW);
			const bool useAtlas = (net.trainingConfig.optimizer.type == glades::OptimizerConfig::ATLAS);
			const bool useVesta = (net.trainingConfig.optimizer.type == glades::OptimizerConfig::VESTA);
			const bool useHelios = (net.trainingConfig.optimizer.type == glades::OptimizerConfig::HELIOS);
			// Token LM mode uses a tied embedding head: logits = H * E^T + lmBias.
			// In this mode, the generic output projection (WOut/bOut) is UNUSED and must not:
			// - contribute to global grad-norm clipping (via weight decay terms), or
			// - be updated/decayed by the optimizer.
			//
			// Otherwise, the model will "silently" change unused parameters and can skew grad clipping
			// scale for the parameters that actually affect the forward pass.
			const bool tokenLMTiedHead = tt.tokenModel;

			// Warmup + DDP LR scaling multipliers.
			const float warmupMult = net.trainingConfig.warmup.multiplier(static_cast<int>(tt.optimizerStep));
			const float ddpLRScale = (net.trainingConfig.ddp.enable && net.trainingConfig.ddp.linearLRScaling)
			                       ? static_cast<float>(glades::ddp::worldSize()) : 1.0f;
			// Paradigm shift #38 SLC mini-LR-warmup (port from CHIRON iter-178).
			// At each T-schedule transition, the trainer marks tt.optimizerStep
			// via NNetwork::setSLCTransition(); the next slcMiniWarmupSteps
			// optimizer steps clamp lrMult to a 0→1 linear ramp.  Element-wise
			// min with warmupMult ensures the global warmup still applies
			// at run start, and the SLC ramp re-applies on each transition.
			float slcMult = 1.0f;
			if (net.trainingConfig.slcLastTransitionStep >= 0 &&
			    net.trainingConfig.slcMiniWarmupSteps > 0)
			{
				const long long since = (long long)tt.optimizerStep -
				                        net.trainingConfig.slcLastTransitionStep;
				if (since >= 0 &&
				    since < (long long)net.trainingConfig.slcMiniWarmupSteps)
				{
					slcMult = (float)since /
					          (float)net.trainingConfig.slcMiniWarmupSteps;
				}
			}
			const float lowestWarmup = (slcMult < warmupMult) ? slcMult : warmupMult;
			const float extraLRMult = lowestWarmup * ddpLRScale;

			// Optional global grad norm clip (same semantics as other tensor paths).
			float gradNorm = 0.0f;
			float gradScale = 1.0f;
			const float clipNorm = net.trainingConfig.globalGradClipNorm;
			if (clipNorm > 0.0f)
			{
				double sumsq = 0.0;

				if (useAdamW || useAtlas || useVesta)
				{
					// For AdamW/ATLAS/VESTA, clip is applied to raw gradients (weight decay is decoupled).
					if (tt.tokenModel)
					{
						for (size_t i = 0; i < tt.gTokE.size(); ++i)
						{
							const double gd = static_cast<double>(tt.gTokE[i] * invBatch);
							sumsq += gd * gd;
						}
						for (size_t i = 0; i < tt.gLmBias.size(); ++i)
						{
							const double gd = static_cast<double>(tt.gLmBias[i] * invBatch);
							sumsq += gd * gd;
						}
					}
					for (size_t i = 0; i < tt.gWIn.size(); ++i)
					{
						const double gd = static_cast<double>(tt.gWIn[i] * invBatch);
						sumsq += gd * gd;
					}
					for (size_t i = 0; i < tt.gBIn.size(); ++i)
					{
						const double gd = static_cast<double>(tt.gBIn[i] * invBatch);
						sumsq += gd * gd;
					}
					for (unsigned int li = 0; li < nLayers; ++li)
					{
						const TensorTransformerState::Block& b = tt.blocks[li];
						for (size_t j = 0; j < b.gWq.size(); ++j) { const double gd = static_cast<double>(b.gWq[j] * invBatch); sumsq += gd * gd; }
						for (size_t j = 0; j < b.gWk.size(); ++j) { const double gd = static_cast<double>(b.gWk[j] * invBatch); sumsq += gd * gd; }
						for (size_t j = 0; j < b.gWv.size(); ++j) { const double gd = static_cast<double>(b.gWv[j] * invBatch); sumsq += gd * gd; }
						for (size_t j = 0; j < b.gWo.size(); ++j) { const double gd = static_cast<double>(b.gWo[j] * invBatch); sumsq += gd * gd; }
						for (size_t j = 0; j < b.gW1.size(); ++j) { const double gd = static_cast<double>(b.gW1[j] * invBatch); sumsq += gd * gd; }
						for (size_t j = 0; j < b.gW2.size(); ++j) { const double gd = static_cast<double>(b.gW2[j] * invBatch); sumsq += gd * gd; }
						for (size_t j = 0; j < b.gBq.size(); ++j) { const double gd = static_cast<double>(b.gBq[j] * invBatch); sumsq += gd * gd; }
						for (size_t j = 0; j < b.gBk.size(); ++j) { const double gd = static_cast<double>(b.gBk[j] * invBatch); sumsq += gd * gd; }
						for (size_t j = 0; j < b.gBv.size(); ++j) { const double gd = static_cast<double>(b.gBv[j] * invBatch); sumsq += gd * gd; }
						for (size_t j = 0; j < b.gBo.size(); ++j) { const double gd = static_cast<double>(b.gBo[j] * invBatch); sumsq += gd * gd; }
						for (size_t j = 0; j < b.gB1.size(); ++j) { const double gd = static_cast<double>(b.gB1[j] * invBatch); sumsq += gd * gd; }
						for (size_t j = 0; j < b.gB2.size(); ++j) { const double gd = static_cast<double>(b.gB2[j] * invBatch); sumsq += gd * gd; }
						for (size_t j = 0; j < b.gLn1Gamma.size(); ++j) { const double gd = static_cast<double>(b.gLn1Gamma[j] * invBatch); sumsq += gd * gd; }
						for (size_t j = 0; j < b.gLn1Beta.size(); ++j) { const double gd = static_cast<double>(b.gLn1Beta[j] * invBatch); sumsq += gd * gd; }
						for (size_t j = 0; j < b.gLn2Gamma.size(); ++j) { const double gd = static_cast<double>(b.gLn2Gamma[j] * invBatch); sumsq += gd * gd; }
						for (size_t j = 0; j < b.gLn2Beta.size(); ++j) { const double gd = static_cast<double>(b.gLn2Beta[j] * invBatch); sumsq += gd * gd; }
					}
					// Final LayerNorm gradients
					for (size_t j = 0; j < tt.gLnFinalGamma.size(); ++j) { const double gd = static_cast<double>(tt.gLnFinalGamma[j] * invBatch); sumsq += gd * gd; }
					for (size_t j = 0; j < tt.gLnFinalBeta.size(); ++j) { const double gd = static_cast<double>(tt.gLnFinalBeta[j] * invBatch); sumsq += gd * gd; }
					// Output projection gradients exist only for non-tokenLM paths.
					if (!tokenLMTiedHead)
					{
						for (size_t i = 0; i < tt.gWOut.size(); ++i)
						{
							const double gd = static_cast<double>(tt.gWOut[i] * invBatch);
							sumsq += gd * gd;
						}
						for (size_t i = 0; i < tt.gBOut.size(); ++i)
						{
							const double gd = static_cast<double>(tt.gBOut[i] * invBatch);
							sumsq += gd * gd;
						}
					}
				}
				else
				{
					// Historical semantics: include L1/L2 weight decay in clip norm.
					// Token LM embedding + bias (index 0)
					if (tt.tokenModel)
					{
						const float wd1 = net.skeleton->getWeightDecay1(0u);
						const float wd2 = net.skeleton->getWeightDecay2(0u);
						for (size_t i = 0; i < tt.tokE.size(); ++i)
						{
							float g = tt.gTokE[i] * invBatch;
							if ((wd1 != 0.0f) || (wd2 != 0.0f))
							{
								const float w = tt.tokE[i];
								const float wSign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
								g += (wd1 * wSign) + (wd2 * w);
							}
							const double gd = static_cast<double>(g);
							sumsq += gd * gd;
						}
						for (size_t i = 0; i < tt.lmBias.size(); ++i)
						{
							const double gd = static_cast<double>(tt.gLmBias[i] * invBatch);
							sumsq += gd * gd;
						}
					}

					// Input projection (index 0)
					{
						const float wd1 = net.skeleton->getWeightDecay1(0u);
						const float wd2 = net.skeleton->getWeightDecay2(0u);
						for (size_t i = 0; i < tt.WIn.size(); ++i)
						{
							float g = tt.gWIn[i] * invBatch;
							if ((wd1 != 0.0f) || (wd2 != 0.0f))
							{
								const float w = tt.WIn[i];
								const float wSign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
								g += (wd1 * wSign) + (wd2 * w);
							}
							const double gd = static_cast<double>(g);
							sumsq += gd * gd;
						}
						for (size_t i = 0; i < tt.bIn.size(); ++i)
						{
							const double gd = static_cast<double>(tt.gBIn[i] * invBatch);
							sumsq += gd * gd;
						}
					}

					// Blocks (index 1..nLayers)
					for (unsigned int li = 0; li < nLayers; ++li)
					{
						const unsigned int idx = li + 1u;
						const float wd1 = net.skeleton->getWeightDecay1(idx);
						const float wd2 = net.skeleton->getWeightDecay2(idx);
						const TensorTransformerState::Block& b = tt.blocks[li];
						// Wq/Wk/Wv/Wo/W1/W2
						{
							const std::vector<float>* Wv[6] = {&b.Wq, &b.Wk, &b.Wv, &b.Wo, &b.W1, &b.W2};
							const std::vector<float>* Gv[6] = {&b.gWq, &b.gWk, &b.gWv, &b.gWo, &b.gW1, &b.gW2};
							for (unsigned int wi = 0; wi < 6u; ++wi)
							{
								const std::vector<float>& W = *Wv[wi];
								const std::vector<float>& gW = *Gv[wi];
								for (size_t j = 0; j < W.size(); ++j)
								{
									float g = gW[j] * invBatch;
									if ((wd1 != 0.0f) || (wd2 != 0.0f))
									{
										const float w = W[j];
										const float wSign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
										g += (wd1 * wSign) + (wd2 * w);
									}
									const double gd = static_cast<double>(g);
									sumsq += gd * gd;
								}
							}
						}
						// Include biases, LN params
						for (size_t j = 0; j < b.bq.size(); ++j) sumsq += static_cast<double>(b.gBq[j] * invBatch) * static_cast<double>(b.gBq[j] * invBatch);
						for (size_t j = 0; j < b.bk.size(); ++j) sumsq += static_cast<double>(b.gBk[j] * invBatch) * static_cast<double>(b.gBk[j] * invBatch);
						for (size_t j = 0; j < b.bv.size(); ++j) sumsq += static_cast<double>(b.gBv[j] * invBatch) * static_cast<double>(b.gBv[j] * invBatch);
						for (size_t j = 0; j < b.bo.size(); ++j) sumsq += static_cast<double>(b.gBo[j] * invBatch) * static_cast<double>(b.gBo[j] * invBatch);
						for (size_t j = 0; j < b.b1.size(); ++j) sumsq += static_cast<double>(b.gB1[j] * invBatch) * static_cast<double>(b.gB1[j] * invBatch);
						for (size_t j = 0; j < b.b2.size(); ++j) sumsq += static_cast<double>(b.gB2[j] * invBatch) * static_cast<double>(b.gB2[j] * invBatch);
						for (size_t j = 0; j < b.ln1Gamma.size(); ++j) sumsq += static_cast<double>(b.gLn1Gamma[j] * invBatch) * static_cast<double>(b.gLn1Gamma[j] * invBatch);
						for (size_t j = 0; j < b.ln1Beta.size(); ++j) sumsq += static_cast<double>(b.gLn1Beta[j] * invBatch) * static_cast<double>(b.gLn1Beta[j] * invBatch);
						for (size_t j = 0; j < b.ln2Gamma.size(); ++j) sumsq += static_cast<double>(b.gLn2Gamma[j] * invBatch) * static_cast<double>(b.gLn2Gamma[j] * invBatch);
						for (size_t j = 0; j < b.ln2Beta.size(); ++j) sumsq += static_cast<double>(b.gLn2Beta[j] * invBatch) * static_cast<double>(b.gLn2Beta[j] * invBatch);
					}

					// Final LayerNorm gradients
					for (size_t j = 0; j < tt.gLnFinalGamma.size(); ++j) sumsq += static_cast<double>(tt.gLnFinalGamma[j] * invBatch) * static_cast<double>(tt.gLnFinalGamma[j] * invBatch);
					for (size_t j = 0; j < tt.gLnFinalBeta.size(); ++j) sumsq += static_cast<double>(tt.gLnFinalBeta[j] * invBatch) * static_cast<double>(tt.gLnFinalBeta[j] * invBatch);

					// Output projection (index nLayers)
					if (!tokenLMTiedHead)
					{
						const unsigned int idx = nLayers;
						const float wd1 = net.skeleton->getWeightDecay1(idx);
						const float wd2 = net.skeleton->getWeightDecay2(idx);
						for (size_t i = 0; i < tt.WOut.size(); ++i)
						{
							float g = tt.gWOut[i] * invBatch;
							if ((wd1 != 0.0f) || (wd2 != 0.0f))
							{
								const float w = tt.WOut[i];
								const float wSign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
								g += (wd1 * wSign) + (wd2 * w);
							}
							const double gd = static_cast<double>(g);
							sumsq += gd * gd;
						}
						for (size_t i = 0; i < tt.bOut.size(); ++i)
						{
							const double gd = static_cast<double>(tt.gBOut[i] * invBatch);
							sumsq += gd * gd;
						}
					}
				}

				if (!is_finite_double(sumsq))
				{
					net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_TRANSFORMER: non-finite grad-norm accumulation detected (NaN/Inf)");
					net.storeRunningFlag(false);
					return false;
				}
				gradNorm = static_cast<float>(sqrt(sumsq));
				const float eps = 1e-12f;
				gradScale = (gradNorm > clipNorm) ? (clipNorm / (gradNorm + eps)) : 1.0f;
			}

			net.lastGradNorm = gradNorm;
			net.lastGradNormScale = gradScale;
			if (!is_finite(net.lastGradNorm) || !is_finite(net.lastGradNormScale) || net.lastGradNormScale <= 0.0f)
			{
				net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_TRANSFORMER: non-finite grad clipping metadata detected (NaN/Inf)");
				net.storeRunningFlag(false);
				return false;
			}

			const bool captureGapDiagnostics =
			    net.trainingConfig.transformer.captureOptimizerGapDiagnostics
			    && (tt.gapApplyCount == 0ULL);
			timespec optimizerApplyStart;
			optimizerApplyStart.tv_sec = 0;
			optimizerApplyStart.tv_nsec = 0;
			if (captureGapDiagnostics)
				optimizerApplyStart = monotonic_now();

			struct Adam
			{
				static float signf(float x) { return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f); }
				static float clampf(float x, float lo, float hi)
				{
					return (x < lo) ? lo : ((x > hi) ? hi : x);
				}

				static float compute_group_scale(const std::vector<float>& P,
				                                const std::vector<float>& m,
				                                const std::vector<float>& v2,
				                                const std::vector<float>& gP,
				                                float beta1,
				                                float beta2,
				                                float inv1mB1t,
				                                float inv1mB2t,
				                                float eps,
				                                float invBatch,
				                                float gradScale,
				                                float prevStepRms,
				                                const glades::OptimizerConfig& opt,
				                                float* outStepRms)
				{
					if (!opt.adamGroupwiseEnabled || P.empty() || P.size() < opt.adamGroupMinSize)
					{
						if (outStepRms)
							*outStepRms = prevStepRms;
						return 1.0f;
					}

					const float oneMinusB1 = 1.0f - beta1;
					const float oneMinusB2 = 1.0f - beta2;
					double stepSqSum = 0.0;
					double weightSqSum = 0.0;
					double mHatSqSum = 0.0;
					double vHatSum = 0.0;
					for (size_t i = 0; i < P.size(); ++i)
					{
						const float g = (gP[i] * invBatch) * gradScale;
						const float mi = (beta1 * m[i]) + (oneMinusB1 * g);
						const float vi = (beta2 * v2[i]) + (oneMinusB2 * (g * g));
						const float mhat = mi * inv1mB1t;
						const float vhat = vi * inv1mB2t;
						const float denom = static_cast<float>(sqrt(static_cast<double>(vhat))) + eps;
						const float step = mhat / denom;
						stepSqSum += static_cast<double>(step) * step;
						weightSqSum += static_cast<double>(P[i]) * P[i];
						mHatSqSum += static_cast<double>(mhat) * mhat;
						vHatSum += static_cast<double>(vhat);
					}

					const double invN = 1.0 / static_cast<double>(P.size());
					const float stepRms = static_cast<float>(sqrt(std::max(0.0, stepSqSum * invN)));
					if (outStepRms)
						*outStepRms = stepRms;
					const float weightRms = static_cast<float>(sqrt(std::max(0.0, weightSqSum * invN)));
					const float snr = static_cast<float>((mHatSqSum * invN) / ((vHatSum * invN) + eps));
					const float stability = (prevStepRms > 0.0f)
					    ? (std::min(stepRms, prevStepRms) / (std::max(stepRms, prevStepRms) + eps))
					    : 1.0f;
					const float updateRatio = stepRms / (weightRms + eps);
					float scale = 1.0f
					    + (opt.adamGroupStabilityScale * stability)
					    + (opt.adamGroupSnrScale * log1pf(std::max(snr, 0.0f)))
					    - (opt.adamGroupRatioScale * updateRatio);
					if (!is_finite(scale))
						scale = 1.0f;
					return clampf(scale, opt.adamGroupMinScale, opt.adamGroupMaxScale);
				}

				static void update_weight(std::vector<float>& W,
				                          std::vector<float>& m,
				                          std::vector<float>& v2,
				                          std::vector<float>& gW,
				                          float lr,
				                          float beta1,
				                          float beta2,
				                          float inv1mB1t,
				                          float inv1mB2t,
				                          float eps,
				                          float invBatch,
				                          float gradScale,
				                          float wd1,
				                          float wd2)
				{
					const float oneMinusB1 = 1.0f - beta1;
					const float oneMinusB2 = 1.0f - beta2;
					for (size_t i = 0; i < W.size(); ++i)
					{
						float g = gW[i] * invBatch;
						if (wd1 != 0.0f)
							g += wd1 * signf(W[i]);
						g *= gradScale;

						const float mi = (beta1 * m[i]) + (oneMinusB1 * g);
						const float vi = (beta2 * v2[i]) + (oneMinusB2 * (g * g));
						m[i] = mi;
						v2[i] = vi;

						const float mhat = mi * inv1mB1t;
						const float vhat = vi * inv1mB2t;
						const float denom = static_cast<float>(sqrt(static_cast<double>(vhat))) + eps;
						const float step = mhat / denom;

						if (wd2 != 0.0f)
							W[i] -= lr * wd2 * W[i];
						W[i] -= lr * step;
						gW[i] = 0.0f;
					}
				}

				static void update_param(std::vector<float>& P,
				                         std::vector<float>& m,
				                         std::vector<float>& v2,
				                         std::vector<float>& gP,
				                         float lr,
				                         float beta1,
				                         float beta2,
				                         float inv1mB1t,
				                         float inv1mB2t,
				                         float eps,
				                         float invBatch,
				                         float gradScale)
				{
					const float oneMinusB1 = 1.0f - beta1;
					const float oneMinusB2 = 1.0f - beta2;
					for (size_t i = 0; i < P.size(); ++i)
					{
						float g = (gP[i] * invBatch) * gradScale;
						const float mi = (beta1 * m[i]) + (oneMinusB1 * g);
						const float vi = (beta2 * v2[i]) + (oneMinusB2 * (g * g));
						m[i] = mi;
						v2[i] = vi;

						const float mhat = mi * inv1mB1t;
						const float vhat = vi * inv1mB2t;
						const float denom = static_cast<float>(sqrt(static_cast<double>(vhat))) + eps;
						const float step = mhat / denom;

						P[i] -= lr * step;
						gP[i] = 0.0f;
					}
				}
			};

			struct Geode
			{
				static bool invert_small(std::vector<float>& mat, unsigned int dim)
				{
					if (dim == 0u)
						return true;
					std::vector<float> inv(static_cast<size_t>(dim) * dim, 0.0f);
					for (unsigned int i = 0u; i < dim; ++i)
						inv[static_cast<size_t>(i) * dim + i] = 1.0f;
					for (unsigned int col = 0u; col < dim; ++col)
					{
						unsigned int pivotRow = col;
						float pivotAbs = fabsf(mat[static_cast<size_t>(col) * dim + col]);
						for (unsigned int row = col + 1u; row < dim; ++row)
						{
							const float candAbs = fabsf(mat[static_cast<size_t>(row) * dim + col]);
							if (candAbs > pivotAbs)
							{
								pivotAbs = candAbs;
								pivotRow = row;
							}
						}
						if (!(pivotAbs > 1.0e-8f) || !is_finite(pivotAbs))
							return false;
						if (pivotRow != col)
						{
							for (unsigned int j = 0u; j < dim; ++j)
							{
								std::swap(mat[static_cast<size_t>(col) * dim + j],
								          mat[static_cast<size_t>(pivotRow) * dim + j]);
								std::swap(inv[static_cast<size_t>(col) * dim + j],
								          inv[static_cast<size_t>(pivotRow) * dim + j]);
							}
						}
						const float pivot = mat[static_cast<size_t>(col) * dim + col];
						const float invPivot = 1.0f / pivot;
						for (unsigned int j = 0u; j < dim; ++j)
						{
							mat[static_cast<size_t>(col) * dim + j] *= invPivot;
							inv[static_cast<size_t>(col) * dim + j] *= invPivot;
						}
						for (unsigned int row = 0u; row < dim; ++row)
						{
							if (row == col)
								continue;
							const float factor = mat[static_cast<size_t>(row) * dim + col];
							if (fabsf(factor) <= 1.0e-12f)
								continue;
							for (unsigned int j = 0u; j < dim; ++j)
							{
								mat[static_cast<size_t>(row) * dim + j] -= factor * mat[static_cast<size_t>(col) * dim + j];
								inv[static_cast<size_t>(row) * dim + j] -= factor * inv[static_cast<size_t>(col) * dim + j];
							}
						}
					}
					mat.swap(inv);
					return true;
				}

				static bool update_weight(std::vector<float>& W,
				                          std::vector<float>& m,
				                          std::vector<float>& v2,
				                          std::vector<float>& gW,
				                          glades::atlas::WeightState& state,
				                          unsigned int rows,
				                          unsigned int cols,
				                          float lr,
				                          float beta1,
				                          float beta2,
				                          float inv1mB1t,
				                          float inv1mB2t,
				                          float eps,
				                          float invBatch,
				                          float gradScale,
				                          float wd1,
				                          float wd2,
				                          const glades::ATLASConfig& ac,
				                          glades::rng::Engine& rng,
				                          shmea::GLogger* logger,
				                          const char* tag)
				{
					if (W.empty())
						return true;
					if (W.size() != gW.size() || W.size() != m.size() || W.size() != v2.size())
						return false;

					const bool predictiveEnabled = (ac.geodePredictiveScale > 0.0f);
					if (!state.initialized && rows > 0u && cols > 0u)
						glades::atlas::initWeightState(state, rows, cols, ac.rank, ac.muMin, rng, logger);
					if (!state.initialized || state.m != rows || state.n != cols)
						return false;
					if ((state.step % std::max(1u, ac.tSub)) == 0u)
					{
						if (!glades::atlas::refreshSubspace(state, &gW[0], rows, cols,
						                                    3u, ac.betaRefresh, false, rng, logger))
							return false;
					}

					const unsigned int activeRank =
					    std::min<unsigned int>(
					        (state.activeRank > 0u) ? state.activeRank : state.r,
					        std::min<unsigned int>(state.r,
					                               static_cast<unsigned int>(state.fisherDiag.size())));
					const float oneMinusB1 = 1.0f - beta1;
					const float oneMinusB2 = 1.0f - beta2;
					const float betaGeom = std::min<float>(std::max<float>(ac.beta, 0.0f), 1.0f);
					for (size_t i = 0u; i < W.size(); ++i)
					{
						const float gScaledRaw = gW[i] * invBatch * gradScale;
						float g = gScaledRaw;
						if (wd1 != 0.0f)
							g += (wd1 * Adam::signf(W[i]) * gradScale);
						const float mi = (beta1 * m[i]) + (oneMinusB1 * g);
						const float vi = (beta2 * v2[i]) + (oneMinusB2 * (g * g));
						m[i] = mi;
						v2[i] = vi;
						gW[i] = gScaledRaw;
					}

					std::vector<float>& basisPacked = state.scratch_basisPacked;
					if (basisPacked.size() < static_cast<size_t>(rows) * activeRank)
						basisPacked.resize(static_cast<size_t>(rows) * activeRank);
					for (unsigned int i = 0u; i < rows; ++i)
					{
						const size_t uOff = static_cast<size_t>(i) * state.r;
						const size_t pOff = static_cast<size_t>(i) * activeRank;
						for (unsigned int c = 0u; c < activeRank; ++c)
							basisPacked[pOff + c] = state.U[uOff + c];
					}
					std::vector<float>& projectedGrad = state.scratch_gz;
					if (projectedGrad.size() < static_cast<size_t>(activeRank) * cols)
						projectedGrad.resize(static_cast<size_t>(activeRank) * cols);
					if (activeRank > 0u && !state.U.empty())
					{
						glades::gemm::atb(&projectedGrad[0], &basisPacked[0], &gW[0],
						                  activeRank, rows, cols, 1.0f);
						for (unsigned int c = 0u; c < activeRank; ++c)
						{
							double meanSq = 0.0;
							for (unsigned int j = 0u; j < cols; ++j)
							{
								const double v = static_cast<double>(projectedGrad[static_cast<size_t>(c) * cols + j]);
								meanSq += v * v;
							}
							meanSq /= static_cast<double>(std::max(1u, cols));
							state.fisherDiag[c] =
							    betaGeom * state.fisherDiag[c]
							    + (1.0f - betaGeom) * static_cast<float>(meanSq);
						}
					}

					if (!(activeRank > 0u) || !(ac.geodeGeometryScale > 0.0f) || state.U.empty())
					{
						for (size_t i = 0u; i < W.size(); ++i)
						{
							const float mhatVal = m[i] * inv1mB1t;
							const float vhat = v2[i] * inv1mB2t;
							const float denom = static_cast<float>(sqrt(static_cast<double>(vhat))) + eps;
							const float invDiag = 1.0f / std::max(denom, 1.0e-12f);
							if (wd2 != 0.0f)
								W[i] -= lr * wd2 * W[i];
							W[i] -= lr * invDiag * mhatVal;
							gW[i] = 0.0f;
						}
						return true;
					}

					std::vector<float> cInv(activeRank, 0.0f);
					for (unsigned int c = 0u; c < activeRank; ++c)
					{
						const float fisher = (c < state.fisherDiag.size()) ? std::max(0.0f, state.fisherDiag[c]) : 0.0f;
						const float geomDiag =
						    std::max(1.0e-6f,
						             std::max(0.0f, ac.geodeGeometryScale)
						                 * (static_cast<float>(sqrt(static_cast<double>(fisher + eps))) + eps));
						cInv[c] = 1.0f / geomDiag;
					}

					const float predictiveAlpha =
					    predictiveEnabled
					        ? std::max(0.0f,
					                   std::min(1.0f, ac.geodePredictiveScale))
					        : 0.0f;

					std::vector<float>& rhsCol = state.scratch_geodeRhsCol;
					std::vector<float>& invDiagCol = state.scratch_geodeInvDiagCol;
					std::vector<float>& activeCurrent = state.scratch_geodeActiveCurrent;
					std::vector<float>& activeDelta = state.scratch_geodeActiveDelta;
					std::vector<float>& systemMat = state.scratch_geodeSystemMat;
					std::vector<float>& rhs = state.scratch_geodeRhs;
					std::vector<float>& solution = state.scratch_geodeSolution;
					if (rhsCol.size() < rows)
						rhsCol.resize(rows);
					if (invDiagCol.size() < rows)
						invDiagCol.resize(rows);
					if (activeCurrent.size() < activeRank)
						activeCurrent.resize(activeRank);
					if (activeDelta.size() < activeRank)
						activeDelta.resize(activeRank);
					if (systemMat.size() < static_cast<size_t>(activeRank) * activeRank)
						systemMat.resize(static_cast<size_t>(activeRank) * activeRank);
					if (rhs.size() < activeRank)
						rhs.resize(activeRank);
					if (solution.size() < activeRank)
						solution.resize(activeRank);

					for (unsigned int j = 0u; j < cols; ++j)
					{
						std::fill(rhsCol.begin(), rhsCol.end(), 0.0f);
						std::fill(invDiagCol.begin(), invDiagCol.end(), 0.0f);
						std::fill(activeCurrent.begin(), activeCurrent.end(), 0.0f);
						std::fill(activeDelta.begin(), activeDelta.end(), 0.0f);
						for (unsigned int i = 0u; i < rows; ++i)
						{
							const size_t idx = static_cast<size_t>(i) * cols + j;
							const float mhatVal = m[idx] * inv1mB1t;
							const float vhat = v2[idx] * inv1mB2t;
							const float denom = static_cast<float>(sqrt(static_cast<double>(vhat))) + eps;
							const float invDiagVal = 1.0f / std::max(denom, 1.0e-12f);
							rhsCol[i] = mhatVal;
							invDiagCol[i] = invDiagVal;
							const size_t pOff = static_cast<size_t>(i) * activeRank;
							for (unsigned int c = 0u; c < activeRank; ++c)
								activeCurrent[c] += basisPacked[pOff + c] * rhsCol[i];
						}

						if (predictiveAlpha > 0.0f && !state.prevGz.empty())
						{
							bool anyDelta = false;
							for (unsigned int c = 0u; c < activeRank; ++c)
							{
								const size_t histIdx = static_cast<size_t>(c) * cols + j;
								const float prevActive =
								    (histIdx < state.prevGz.size()) ? state.prevGz[histIdx] : 0.0f;
								const float currentActive =
								    (histIdx < projectedGrad.size()) ? projectedGrad[histIdx] : activeCurrent[c];
								const float predicted =
								    activeCurrent[c] + predictiveAlpha * (currentActive - prevActive);
								activeDelta[c] = predicted - activeCurrent[c];
								anyDelta = anyDelta || (fabsf(activeDelta[c]) > 1.0e-12f);
							}
							if (anyDelta)
							{
								for (unsigned int i = 0u; i < rows; ++i)
								{
									const size_t pOff = static_cast<size_t>(i) * activeRank;
									float delta = 0.0f;
									for (unsigned int c = 0u; c < activeRank; ++c)
										delta += basisPacked[pOff + c] * activeDelta[c];
									rhsCol[i] += delta;
								}
							}
						}

						std::fill(systemMat.begin(), systemMat.end(), 0.0f);
						std::fill(rhs.begin(), rhs.end(), 0.0f);
						for (unsigned int c = 0u; c < activeRank; ++c)
							systemMat[static_cast<size_t>(c) * activeRank + c] = cInv[c];
						for (unsigned int i = 0u; i < rows; ++i)
						{
							const float invd = invDiagCol[i];
							const size_t pOff = static_cast<size_t>(i) * activeRank;
							for (unsigned int a = 0u; a < activeRank; ++a)
							{
								const float uia = basisPacked[pOff + a];
								rhs[a] += uia * invd * rhsCol[i];
								for (unsigned int b = 0u; b < activeRank; ++b)
									systemMat[static_cast<size_t>(a) * activeRank + b]
									    += uia * invd * basisPacked[pOff + b];
							}
						}

						const bool solved = invert_small(systemMat, activeRank);
						if (solved)
						{
							for (unsigned int a = 0u; a < activeRank; ++a)
							{
								double sum = 0.0;
								for (unsigned int b = 0u; b < activeRank; ++b)
									sum += static_cast<double>(systemMat[static_cast<size_t>(a) * activeRank + b])
									     * static_cast<double>(rhs[b]);
								solution[a] = static_cast<float>(sum);
							}
						}
						else
						{
							std::fill(solution.begin(), solution.end(), 0.0f);
						}

						for (unsigned int i = 0u; i < rows; ++i)
						{
							const size_t idx = static_cast<size_t>(i) * cols + j;
							float correction = 0.0f;
							const size_t pOff = static_cast<size_t>(i) * activeRank;
							for (unsigned int c = 0u; c < activeRank; ++c)
								correction += basisPacked[pOff + c] * solution[c];
							const float step = invDiagCol[i] * (rhsCol[i] - correction);
							if (wd2 != 0.0f)
								W[idx] -= lr * wd2 * W[idx];
							W[idx] -= lr * step;
							gW[idx] = 0.0f;
						}
					}

					if (activeRank > 0u && state.prevGz.size() >= projectedGrad.size())
					{
						for (size_t k = 0u; k < projectedGrad.size(); ++k)
							state.prevGz[k] = projectedGrad[k];
					}
					state.step += 1ULL;

					return true;
				}
			};

			// --- Apply updates ---
			if (!useAdamW && !useAtlas && !useVesta && !useHelios)
			{
				// SGD + momentum (historical behavior).
				// Token embedding (index 0 in LM mode)
				if (tt.tokenModel)
				{
					const float lr = net.skeleton->getLearningRate(0u) * net.lrScheduleMultiplier * extraLRMult;
					const float mf = net.skeleton->getMomentumFactor(0u);
					const float wd1 = net.skeleton->getWeightDecay1(0u);
					const float wd2 = net.skeleton->getWeightDecay2(0u);

					for (size_t i = 0; i < tt.tokE.size(); ++i)
					{
						float g = tt.gTokE[i] * invBatch;
						if ((wd1 != 0.0f) || (wd2 != 0.0f))
						{
							const float w = tt.tokE[i];
							const float wSign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
							g += (wd1 * wSign) + (wd2 * w);
						}
						g *= gradScale;
						const float v = (mf * tt.vTokE[i]) + (lr * g);
						tt.vTokE[i] = v;
						tt.tokE[i] -= v;
						tt.gTokE[i] = 0.0f;
					}
					for (size_t i = 0; i < tt.lmBias.size(); ++i)
					{
						const float gB = (tt.gLmBias[i] * invBatch) * gradScale;
						tt.lmBias[i] -= lr * gB;
						tt.gLmBias[i] = 0.0f;
					}
				}

				// Input projection (index 0)
				{
					const float lr = net.skeleton->getLearningRate(0u) * net.lrScheduleMultiplier * extraLRMult;
					const float mf = net.skeleton->getMomentumFactor(0u);
					const float wd1 = net.skeleton->getWeightDecay1(0u);
					const float wd2 = net.skeleton->getWeightDecay2(0u);
					for (size_t i = 0; i < tt.WIn.size(); ++i)
					{
						float g = tt.gWIn[i] * invBatch;
						if ((wd1 != 0.0f) || (wd2 != 0.0f))
						{
							const float w = tt.WIn[i];
							const float wSign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
							g += (wd1 * wSign) + (wd2 * w);
						}
						g *= gradScale;
						const float v = (mf * tt.vWIn[i]) + (lr * g);
						tt.vWIn[i] = v;
						tt.WIn[i] -= v;
						tt.gWIn[i] = 0.0f;
					}
					for (size_t i = 0; i < tt.bIn.size(); ++i)
					{
						const float gB = (tt.gBIn[i] * invBatch) * gradScale;
						tt.bIn[i] -= lr * gB;
						tt.gBIn[i] = 0.0f;
					}
				}

				// Blocks (index 1..nLayers)
				for (unsigned int li = 0; li < nLayers; ++li)
				{
					const unsigned int idx = li + 1u;
					const float lr = net.skeleton->getLearningRate(idx) * net.lrScheduleMultiplier * extraLRMult;
					const float mf = net.skeleton->getMomentumFactor(idx);
					const float wd1 = net.skeleton->getWeightDecay1(idx);
					const float wd2 = net.skeleton->getWeightDecay2(idx);
					TensorTransformerState::Block& b = tt.blocks[li];

					// Attention/FFN weights with momentum/decay
					struct Upd
					{
						static void run(std::vector<float>& W, std::vector<float>& vW, std::vector<float>& gW,
						                float lr, float mf, float wd1, float wd2, float invBatch, float gradScale)
						{
							for (size_t i = 0; i < W.size(); ++i)
							{
								float g = gW[i] * invBatch;
								if ((wd1 != 0.0f) || (wd2 != 0.0f))
								{
									const float w = W[i];
									const float wSign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
									g += (wd1 * wSign) + (wd2 * w);
								}
								g *= gradScale;
								const float v = (mf * vW[i]) + (lr * g);
								vW[i] = v;
								W[i] -= v;
								gW[i] = 0.0f;
							}
						}
					};

					Upd::run(b.Wq, b.vWq, b.gWq, lr, mf, wd1, wd2, invBatch, gradScale);
					Upd::run(b.Wk, b.vWk, b.gWk, lr, mf, wd1, wd2, invBatch, gradScale);
					Upd::run(b.Wv, b.vWv, b.gWv, lr, mf, wd1, wd2, invBatch, gradScale);
					Upd::run(b.Wo, b.vWo, b.gWo, lr, mf, wd1, wd2, invBatch, gradScale);
					Upd::run(b.W1, b.vW1, b.gW1, lr, mf, wd1, wd2, invBatch, gradScale);
					Upd::run(b.W2, b.vW2, b.gW2, lr, mf, wd1, wd2, invBatch, gradScale);

					// Biases and LN params (no momentum/decay)
					for (size_t i = 0; i < b.bq.size(); ++i) { b.bq[i] -= lr * (b.gBq[i] * invBatch) * gradScale; b.gBq[i] = 0.0f; }
					for (size_t i = 0; i < b.bk.size(); ++i) { b.bk[i] -= lr * (b.gBk[i] * invBatch) * gradScale; b.gBk[i] = 0.0f; }
					for (size_t i = 0; i < b.bv.size(); ++i) { b.bv[i] -= lr * (b.gBv[i] * invBatch) * gradScale; b.gBv[i] = 0.0f; }
					for (size_t i = 0; i < b.bo.size(); ++i) { b.bo[i] -= lr * (b.gBo[i] * invBatch) * gradScale; b.gBo[i] = 0.0f; }
					for (size_t i = 0; i < b.b1.size(); ++i) { b.b1[i] -= lr * (b.gB1[i] * invBatch) * gradScale; b.gB1[i] = 0.0f; }
					for (size_t i = 0; i < b.b2.size(); ++i) { b.b2[i] -= lr * (b.gB2[i] * invBatch) * gradScale; b.gB2[i] = 0.0f; }
					for (size_t i = 0; i < b.ln1Gamma.size(); ++i) { b.ln1Gamma[i] -= lr * (b.gLn1Gamma[i] * invBatch) * gradScale; b.gLn1Gamma[i] = 0.0f; }
					for (size_t i = 0; i < b.ln1Beta.size(); ++i) { b.ln1Beta[i] -= lr * (b.gLn1Beta[i] * invBatch) * gradScale; b.gLn1Beta[i] = 0.0f; }
					for (size_t i = 0; i < b.ln2Gamma.size(); ++i) { b.ln2Gamma[i] -= lr * (b.gLn2Gamma[i] * invBatch) * gradScale; b.gLn2Gamma[i] = 0.0f; }
					for (size_t i = 0; i < b.ln2Beta.size(); ++i) { b.ln2Beta[i] -= lr * (b.gLn2Beta[i] * invBatch) * gradScale; b.gLn2Beta[i] = 0.0f; }
				}

				// Final LayerNorm (SGD, use block 0 LR; no weight decay)
				{
					const float lr = net.skeleton->getLearningRate(1u) * net.lrScheduleMultiplier * extraLRMult;
					for (size_t i = 0; i < tt.lnFinalGamma.size(); ++i) { tt.lnFinalGamma[i] -= lr * (tt.gLnFinalGamma[i] * invBatch) * gradScale; tt.gLnFinalGamma[i] = 0.0f; }
					for (size_t i = 0; i < tt.lnFinalBeta.size(); ++i) { tt.lnFinalBeta[i] -= lr * (tt.gLnFinalBeta[i] * invBatch) * gradScale; tt.gLnFinalBeta[i] = 0.0f; }
				}

				// Output projection (index nLayers) is unused in token LM tied-head mode.
				if (!tokenLMTiedHead)
				{
					const unsigned int idx = nLayers;
					const float lr = net.skeleton->getLearningRate(idx) * net.lrScheduleMultiplier * extraLRMult;
					const float mf = net.skeleton->getMomentumFactor(idx);
					const float wd1 = net.skeleton->getWeightDecay1(idx);
					const float wd2 = net.skeleton->getWeightDecay2(idx);
					for (size_t i = 0; i < tt.WOut.size(); ++i)
					{
						float g = tt.gWOut[i] * invBatch;
						if ((wd1 != 0.0f) || (wd2 != 0.0f))
						{
							const float w = tt.WOut[i];
							const float wSign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
							g += (wd1 * wSign) + (wd2 * w);
						}
						g *= gradScale;
						const float v = (mf * tt.vWOut[i]) + (lr * g);
						tt.vWOut[i] = v;
						tt.WOut[i] -= v;
						tt.gWOut[i] = 0.0f;
					}
					for (size_t i = 0; i < tt.bOut.size(); ++i)
					{
						const float gB = (tt.gBOut[i] * invBatch) * gradScale;
						tt.bOut[i] -= lr * gB;
						tt.gBOut[i] = 0.0f;
					}
				}
				else
				{
					// Defensive: ensure gradients are cleared so stale values never leak into later non-tokenLM runs.
					std::fill(tt.gWOut.begin(), tt.gWOut.end(), 0.0f);
					std::fill(tt.gBOut.begin(), tt.gBOut.end(), 0.0f);
				}
			}
			else if (useAtlas)
			{
				// ATLAS optimizer: subspace-projected natural gradient with temporal prediction.
				const glades::ATLASConfig& ac = net.trainingConfig.atlas;

				tt.optimizerStep += 1ULL;

				// Compute weight matrix dimensions.
				const unsigned int dmTT = tt.dModel;
				const unsigned int dFFTT = tt.dFF;
				const unsigned int nHeadsTT = tt.nHeads;
				const unsigned int nKVHeadsTT = (tt.nKVHeads > 0u ? tt.nKVHeads : nHeadsTT);
				const unsigned int dHeadTT = dmTT / nHeadsTT;
				const unsigned int dModelKVTT = nKVHeadsTT * dHeadTT;
				const unsigned int ffnKindTT = tt.ffnKind;
				const unsigned int ff1WidthTT = (ffnKindTT == static_cast<unsigned int>(glades::TransformerRunConfig::FFN_SWIGLU)) ? (2u * dFFTT) : dFFTT;
				const bool geodeEnabled = ac.geodeEnabled;
				const bool echoEnabled = ac.echoEnabled;
				const bool bimapEnabled = ac.bimapEnabled;
				const bool pactEnabled = ac.pactEnabled;
				const bool racerEnabled = ac.racerEnabled;
				const bool kronEnabled = ac.kronEnabled;
				const bool matraEnabled = ac.matraEnabled;
				const bool argosEnabled = ac.argosEnabled;
				const bool muonEnabled = ac.muonEnabled;
				const bool auroraAdamwBackbone = ac.auroraEnabled && ac.auroraAdamwBackbone;
				const float beta1 = net.trainingConfig.optimizer.adamBeta1;
				const float beta2 = net.trainingConfig.optimizer.adamBeta2;
				const float eps = net.trainingConfig.optimizer.adamEps;
				const bool biasCorr = net.trainingConfig.optimizer.adamBiasCorrection;
				const double t = static_cast<double>(tt.optimizerStep);
				const double b1t = biasCorr ? pow(static_cast<double>(beta1), t) : 0.0;
				const double b2t = biasCorr ? pow(static_cast<double>(beta2), t) : 0.0;
				const float inv1mB1t = biasCorr ? static_cast<float>(1.0 / (1.0 - b1t)) : 1.0f;
				const float inv1mB2t = biasCorr ? static_cast<float>(1.0 / (1.0 - b2t)) : 1.0f;

				// Token embedding (index 0 in LM mode)
				if (tt.tokenModel)
				{
					const float lr = net.skeleton->getLearningRate(0u) * net.lrScheduleMultiplier * extraLRMult;
					const float wd1 = net.skeleton->getWeightDecay1(0u);
					const float wd2 = net.skeleton->getWeightDecay2(0u);
					TensorTransformerState::HelmState& helm = tt.helm;
					TensorTransformerState::AsterState& aster = tt.aster;
					if (ac.helmEnabled && helm.initialized
					    && helm.batchTokenCount > 0u
					    && !helm.batchHiddenSum.empty()
					    && !helm.batchResidualSum.empty())
					{
						const unsigned int hiddenDim = helm.hiddenDim;
						const unsigned int outputDim =
						    std::min<unsigned int>(helm.outputDim, dModel);
						const unsigned int pastDim = hiddenDim + outputDim;
						const unsigned int modeRank =
						    std::min<unsigned int>(std::max(1u, helm.modeRank),
						                           std::max(1u, outputDim));
						const float invTokenCount =
						    1.0f / static_cast<float>(std::max(1u, helm.batchTokenCount));
						const float betaHelm =
						    std::min<float>(std::max<float>(ac.beta, 0.0f), 1.0f);
						const float ridge = 1e-4f;
						const float varEps = 1e-6f;
						const float sigmaEps = 1e-12f;
						std::vector<float> hiddenMean(hiddenDim, 0.0f);
						std::vector<float> residualMean(outputDim, 0.0f);
						std::vector<float> hiddenStd(hiddenDim, 1.0f);
						std::vector<float> residualStd(outputDim, 1.0f);
						std::vector<float> pastSignal(pastDim, 0.0f);
						std::vector<float> currentResidualW(outputDim, 0.0f);
						std::vector<float> predictedLatent(modeRank, 0.0f);
						std::vector<float> predictedResidualW(outputDim, 0.0f);
						std::vector<float> nextSigma;
						std::vector<float> nextLeft;
						std::vector<float> nextRight;
						for (unsigned int i = 0; i < hiddenDim; ++i)
						{
							hiddenMean[i] = helm.batchHiddenSum[i] * invTokenCount;
							const float secondMoment = helm.batchHiddenSqSum[i] * invTokenCount;
							const float updatedVar =
							    (betaHelm * helm.hiddenVar[i]) + ((1.0f - betaHelm) * std::max(secondMoment, varEps));
							helm.hiddenVar[i] = std::max(updatedVar, varEps);
							hiddenStd[i] = sqrtf(helm.hiddenVar[i]);
							pastSignal[i] =
							    (hiddenStd[i] > sigmaEps) ? (helm.prevHiddenMean[i] / hiddenStd[i]) : 0.0f;
						}
						for (unsigned int j = 0; j < outputDim; ++j)
						{
							residualMean[j] = helm.batchResidualSum[j] * invTokenCount;
							const float secondMoment = helm.batchResidualSqSum[j] * invTokenCount;
							const float updatedVar =
							    (betaHelm * helm.residualVar[j]) + ((1.0f - betaHelm) * std::max(secondMoment, varEps));
							helm.residualVar[j] = std::max(updatedVar, varEps);
							residualStd[j] = sqrtf(helm.residualVar[j]);
							pastSignal[hiddenDim + j] =
							    (residualStd[j] > sigmaEps) ? (helm.prevResidualMean[j] / residualStd[j]) : 0.0f;
							currentResidualW[j] =
							    (residualStd[j] > sigmaEps) ? (residualMean[j] / residualStd[j]) : 0.0f;
						}

						double effectiveSigmaSq = 0.0;
						for (unsigned int m = 0; m < modeRank; ++m)
						{
							const float modeSigma =
							    (m < helm.sigma.size()) ? std::max(0.0f, helm.sigma[m]) : 0.0f;
							const float modePole =
							    (m < helm.pole.size()) ? helm.pole[m] : 0.0f;
							const float modeLatent =
							    (m < helm.latent.size()) ? helm.latent[m] : 0.0f;
							double latentInput = 0.0;
							const size_t rightOff =
							    static_cast<size_t>(m) * static_cast<size_t>(pastDim);
							for (unsigned int p = 0; p < pastDim && (rightOff + p) < helm.rightMode.size(); ++p)
								latentInput += static_cast<double>(helm.rightMode[rightOff + p])
								             * static_cast<double>(pastSignal[p]);
							predictedLatent[m] = static_cast<float>(
							    static_cast<double>(modePole) * static_cast<double>(modeLatent) + latentInput);
							effectiveSigmaSq += static_cast<double>(modeSigma) * static_cast<double>(modeSigma);
							const size_t leftOff =
							    static_cast<size_t>(m) * static_cast<size_t>(outputDim);
							for (unsigned int j = 0; j < outputDim && (leftOff + j) < helm.leftMode.size(); ++j)
								predictedResidualW[j] += modeSigma * helm.leftMode[leftOff + j] * predictedLatent[m];
						}
						const float effectiveSigma =
						    static_cast<float>(sqrt(std::max(0.0, effectiveSigmaSq)));

						double targetNormSq = 0.0;
						double errNormSq = 0.0;
						for (unsigned int j = 0; j < outputDim; ++j)
						{
							const double target = static_cast<double>(currentResidualW[j]);
							const double err = target - static_cast<double>(predictedResidualW[j]);
							targetNormSq += target * target;
							errNormSq += err * err;
						}
						float helmPredR2 = 0.0f;
						if (targetNormSq > 1e-12)
							helmPredR2 = static_cast<float>(
							    std::max<double>(0.0, 1.0 - (errNormSq / targetNormSq)));
						const float helmEdge = std::max(0.0f, effectiveSigma);
						float helmTransferScale = 0.0f;
						if (helmEdge > ac.helmEdgeThreshold)
							helmTransferScale =
							    (helmEdge - ac.helmEdgeThreshold) / (helmEdge + 1e-6f);
						float helmMemoryGain = ac.helmMemoryScale
						                     * std::max(0.0f, helmTransferScale)
						                     * std::max(0.0f, helmPredR2);
						if (!is_finite(helmMemoryGain))
							helmMemoryGain = 0.0f;
						if (helm.forwardCorrection.size() != outputDim)
							helm.forwardCorrection.assign(outputDim, 0.0f);
						for (unsigned int j = 0; j < outputDim; ++j)
						{
							const float predictedResidual = predictedResidualW[j] * residualStd[j];
							const float corr =
							    -helmMemoryGain
							    * clipf_maybe(predictedResidual, net.trainingConfig.perElementGradClip);
							helm.forwardCorrection[j] = is_finite(corr) ? corr : 0.0f;
						}

						for (unsigned int m = 0; m < modeRank; ++m)
						{
							if (m >= helm.poleNumer.size() || m >= helm.poleDenom.size()
							    || m >= helm.pole.size() || m >= helm.latent.size())
								continue;
							const float oldLatent = helm.latent[m];
							const float poleNumer = (betaHelm * helm.poleNumer[m])
							                      + ((1.0f - betaHelm) * predictedLatent[m] * oldLatent);
							const float poleDenom = (betaHelm * helm.poleDenom[m])
							                      + ((1.0f - betaHelm) * oldLatent * oldLatent);
							helm.poleNumer[m] = poleNumer;
							helm.poleDenom[m] = poleDenom;
							if (poleDenom > sigmaEps)
								helm.pole[m] = std::max(-ac.helmPoleMax,
								                        std::min(ac.helmPoleMax,
								                                 poleNumer / (poleDenom + sigmaEps)));
							helm.latent[m] = predictedLatent[m];
						}

						for (unsigned int j = 0; j < outputDim; ++j)
						{
							const size_t rowOff =
							    static_cast<size_t>(j) * static_cast<size_t>(pastDim);
							for (unsigned int p = 0; p < pastDim; ++p)
							{
								const float sample = currentResidualW[j] * pastSignal[p];
								helm.crossCov[rowOff + p] =
								    (betaHelm * helm.crossCov[rowOff + p]) + ((1.0f - betaHelm) * sample);
							}
						}
						extract_top_singular_modes_row_major(helm.crossCov, outputDim, pastDim,
						                                     modeRank, nextSigma, nextLeft, nextRight);
						unsigned int helmActiveModes = 0u;
						for (unsigned int m = 0; m < modeRank; ++m)
						{
							if (m < helm.sigma.size())
								helm.sigma[m] = nextSigma[m];
							if (nextSigma[m] > ac.helmEdgeThreshold)
								helmActiveModes += 1u;
						}
						helm.leftMode.swap(nextLeft);
						helm.rightMode.swap(nextRight);
						helm.lastActiveModes = helmActiveModes;
						helm.lastEdge = (modeRank > 0u) ? std::max(0.0f, helm.sigma[0]) : 0.0f;
						helm.lastSecondEdge =
						    (modeRank > 1u) ? std::max(0.0f, helm.sigma[1]) : 0.0f;
						helm.lastSecondEdgeRatio =
						    (helm.lastEdge > sigmaEps)
						        ? std::max(0.0f, helm.lastSecondEdge / (helm.lastEdge + sigmaEps))
						        : 0.0f;
						helm.lastSigma = effectiveSigma;
						helm.lastPredR2 = helmPredR2;
						helm.lastMemoryGain = helmMemoryGain;
						for (unsigned int i = 0; i < hiddenDim; ++i)
							helm.prevHiddenMean[i] = hiddenMean[i];
						for (unsigned int j = 0; j < outputDim; ++j)
							helm.prevResidualMean[j] = residualMean[j];
					}
					if (ac.asterEnabled && aster.initialized
					    && aster.batchTokenCount > 0u
					    && !aster.batchFinalHiddenRawSum.empty()
					    && !tt.gTokE.empty()
					    && !tt.gLmBias.empty())
					{
						const timespec asterBoundaryStart = monotonic_now();
						const unsigned int bundleDim = aster.sketchDim;
						const unsigned int supportDim = aster.supportDim;
						const unsigned int marginDim = (supportDim > 0u) ? (supportDim - 1u) : 0u;
						const unsigned int tokenCondDim = std::max(1u, aster.tokenCondDim);
						const unsigned int supportTokenCondDim = std::min<unsigned int>(tokenCondDim, 4u);
						const unsigned int explicitTokenCondOff = supportTokenCondDim;
						const unsigned int kappaObsDim = aster.kappaObsDim;
						const unsigned int obsDim = bundleDim + supportDim + marginDim + tokenCondDim + kappaObsDim;
						const unsigned int tokenCondOff = bundleDim + supportDim + marginDim;
						const unsigned int kappaOff = tokenCondOff + tokenCondDim;
						const unsigned int controlStreams = (kappaObsDim > 0u) ? 4u : 3u;
						const unsigned int controlStride = controlStreams * obsDim;
						const unsigned int controlDim = aster.controlDim;
						const unsigned int featureDim = (2u * obsDim) + (3u * controlDim);
						const unsigned int stateRank =
						    std::min<unsigned int>(std::max(1u, aster.stateRank), std::max(1u, obsDim));
						const unsigned int stateFeatureDim = (2u * stateRank) + (3u * controlDim);
						const unsigned int regimeCount = std::max(1u, aster.regimeCount);
						const unsigned int trackedLayers =
						    std::min<unsigned int>(aster.hiddenStackDepth,
						                           static_cast<unsigned int>(aster.trackedBlockIndices.size()));
						const bool sampledSoftmax =
						    (net.trainingConfig.transformer.tokenLmLossKind == glades::TransformerRunConfig::TOKEN_LM_SAMPLED_SOFTMAX);
						if (sampledSoftmax)
						{
							std::sort(aster.batchTouchedIds.begin(), aster.batchTouchedIds.end());
							aster.batchTouchedIds.erase(
							    std::unique(aster.batchTouchedIds.begin(), aster.batchTouchedIds.end()),
							    aster.batchTouchedIds.end());
						}
						const float betaAster =
						    std::min<float>(std::max<float>(ac.beta, 0.0f), 1.0f);
						const float marginBaselineBeta = 0.95f;
						const float ridge = 1e-4f;
						const float varEps = 1e-6f;
						const float sigmaEps = 1e-12f;
						double weightedActiveModes = 0.0;
						double weightedSecondModeFrac = 0.0;
						double weightedEdge = 0.0;
						double weightedSecondEdge = 0.0;
						double weightedSecondEdgeRatio = 0.0;
						double weightedSigma = 0.0;
						double weightedPredR2 = 0.0;
						double weightedMemoryGain = 0.0;
						double weightedPole = 0.0;
						double weightedAegisLambdaSpatial = 0.0;
						double weightedAegisLambdaPredictive = 0.0;
						double weightedAegisLambdaOutput = 0.0;
						double weightedAegisPredictivePredicted = 0.0;
						double weightedAegisPredictiveRealized = 0.0;
						double weightedAegisOutputPredicted = 0.0;
						double weightedAegisOutputRealized = 0.0;
						double weightedAegisPredictiveError = 0.0;
						double weightedAegisOutputError = 0.0;
						double weightedAegisDisagreement = 0.0;
						double weightedCitadelAnchor = 0.0;
						double weightedCitadelHardRegimeMass = 0.0;
						double weightedCitadelSparrowTrust = 0.0;
						double weightedRampartTau = 0.0;
						double weightedRampartBudget = 0.0;
						double weightedRampartCovariance = 0.0;
						double weightedRampartSparrowTrust = 0.0;
						double weightedMeritTau = 0.0;
						double weightedMeritBudget = 0.0;
						double weightedMeritCovariance = 0.0;
						double weightedMeritSparrowTrust = 0.0;
						double weightedMeritGeometryTrust = 0.0;
						double weightedStrataNullMode = 0.0;
						double weightedStrataPredictiveMode = 0.0;
						double weightedStrataOutputMode = 0.0;
						double weightedStrataCoupledMode = 0.0;
						double weightedStrataBudget = 0.0;
						double weightedStrataNullBenefit = 0.0;
						double weightedStrataPredictiveBenefit = 0.0;
						double weightedStrataOutputBenefit = 0.0;
						double weightedStrataCoupledBenefit = 0.0;
						double weightedStrataSelectedExcess = 0.0;
						double weightedStrataSwitchRate = 0.0;
						double weightSum = 0.0;
						unsigned int maxActiveModes = 0u;
						double asterSetupNs = 0.0;
						double asterTransportNs = 0.0;
						double asterTransferFitNs = 0.0;
						double asterStateFitNs = 0.0;
						double asterInnovationFitNs = 0.0;
						double asterApplyNs = 0.0;
						const float batchHardRegimeMass =
						    (aster.batchTokenCount > 0u && regimeCount > 3u
						         && 3u < aster.batchRegimeTokenCount.size())
						        ? std::max(0.0f, std::min(1.0f,
						                                   aster.batchRegimeTokenCount[3u]
						                                       / static_cast<float>(aster.batchTokenCount)))
						        : 0.0f;

						for (unsigned int regime = 0; regime < regimeCount; ++regime)
						{
							const float regimeMass =
							    (regime < aster.batchRegimeTokenCount.size()) ? aster.batchRegimeTokenCount[regime] : 0.0f;
							if (!(regimeMass > 0.0f))
								continue;
							const timespec regimeStart = monotonic_now();
							const unsigned int tokenCount = std::max(1u, static_cast<unsigned int>(regimeMass));
							const float invTokenCount = 1.0f / static_cast<float>(tokenCount);
							const size_t regimeFinalRawOff = static_cast<size_t>(regime) * static_cast<size_t>(dModel);
							const size_t regimeBundleOff = static_cast<size_t>(regime) * static_cast<size_t>(bundleDim);
							const size_t regimeSupportOff = static_cast<size_t>(regime) * static_cast<size_t>(supportDim);
							const size_t regimeSupportHiddenOff = regimeSupportOff * static_cast<size_t>(dModel);
							const size_t regimeLayerRawBase =
							    static_cast<size_t>(regime) * static_cast<size_t>(trackedLayers) * static_cast<size_t>(dModel);
							const size_t regimeLayerSketchBase =
							    static_cast<size_t>(regime) * static_cast<size_t>(trackedLayers) * static_cast<size_t>(bundleDim);
							const size_t regimeLayerAttnRawBase =
							    static_cast<size_t>(regime) * static_cast<size_t>(trackedLayers) * static_cast<size_t>(dModel);
							const size_t regimeLayerAttnSketchBase =
							    static_cast<size_t>(regime) * static_cast<size_t>(trackedLayers) * static_cast<size_t>(bundleDim);
							const size_t regimeLayerKappaBase =
							    static_cast<size_t>(regime) * static_cast<size_t>(trackedLayers) * static_cast<size_t>(kappaObsDim);
							const size_t regimeControlBase = static_cast<size_t>(regime) * static_cast<size_t>(controlDim);
							const size_t regimeObsBase = static_cast<size_t>(regime) * static_cast<size_t>(obsDim);
							const size_t regimeFeatureBase = static_cast<size_t>(regime) * static_cast<size_t>(featureDim) * static_cast<size_t>(featureDim);
							const size_t regimeCrossBase = static_cast<size_t>(regime) * static_cast<size_t>(obsDim) * static_cast<size_t>(featureDim);
							const size_t regimeStateFeatureBase = static_cast<size_t>(regime) * static_cast<size_t>(stateFeatureDim) * static_cast<size_t>(stateFeatureDim);
							const size_t regimeStateCrossBase = static_cast<size_t>(regime) * static_cast<size_t>(stateRank) * static_cast<size_t>(stateFeatureDim);
							const size_t regimeInnovationBase = static_cast<size_t>(regime) * static_cast<size_t>(obsDim) * static_cast<size_t>(obsDim);
							const size_t regimeInnovationCrossBase = static_cast<size_t>(regime) * static_cast<size_t>(stateRank) * static_cast<size_t>(obsDim);
							const size_t regimeLatentBase = static_cast<size_t>(regime) * static_cast<size_t>(stateRank);
							const size_t regimeLeftBase = static_cast<size_t>(regime) * static_cast<size_t>(stateRank) * static_cast<size_t>(obsDim);
							const size_t regimeRightBase = static_cast<size_t>(regime) * static_cast<size_t>(stateRank) * static_cast<size_t>(featureDim);
							std::vector<float> finalHiddenMeanRaw(dModel, 0.0f);
							std::vector<float> finalHiddenMeanSketch(bundleDim, 0.0f);
							std::vector<float> layerHiddenMeanRaw(static_cast<size_t>(trackedLayers) * dModel, 0.0f);
							std::vector<float> layerHiddenMeanSketch(static_cast<size_t>(trackedLayers) * bundleDim, 0.0f);
							std::vector<float> layerAttnMeanRaw(static_cast<size_t>(trackedLayers) * dModel, 0.0f);
							std::vector<float> layerAttnMeanSketch(static_cast<size_t>(trackedLayers) * bundleDim, 0.0f);
							std::vector<float> layerPatternMean(static_cast<size_t>(trackedLayers) * tokenCondDim, 0.0f);
							std::vector<float> layerKappaMean(static_cast<size_t>(trackedLayers) * kappaObsDim, 0.0f);
							std::vector<float> supportHiddenMeanRaw(static_cast<size_t>(supportDim) * dModel, 0.0f);
							std::vector<float> supportCount(supportDim, 0.0f);
							std::vector<float> supportLogitMean(supportDim, 0.0f);
							std::vector<float> supportResidualMean(supportDim, 0.0f);
							std::vector<float> controlMean(controlDim, 0.0f);
							std::vector<float> residualMean(obsDim, 0.0f);
							std::vector<float> controlStd(controlDim, 1.0f);
							std::vector<float> residualStd(obsDim, 1.0f);
							std::vector<float> feature(featureDim, 0.0f);
							std::vector<float> prevPrevControlW(controlDim, 0.0f);
							std::vector<float> prevControlW(controlDim, 0.0f);
							std::vector<float> currentControlW(controlDim, 0.0f);
							std::vector<float> prevPrevResidualW(obsDim, 0.0f);
							std::vector<float> currentResidualW(obsDim, 0.0f);
							std::vector<float> predictedResidualCurrentW(obsDim, 0.0f);
							std::vector<float> predictedResidualNextW(obsDim, 0.0f);
							std::vector<float> observedLatent(stateRank, 0.0f);
							std::vector<float> prevPrevLatent(stateRank, 0.0f);
							std::vector<float> prevLatentAligned(stateRank, 0.0f);
							std::vector<float> predictedState(stateRank, 0.0f);
							std::vector<float> filteredState(stateRank, 0.0f);
							std::vector<float> nextState(stateRank, 0.0f);
							std::vector<float> innovation(obsDim, 0.0f);
							std::vector<float> stateCorrection(stateRank, 0.0f);
							std::vector<float> stateFeature(stateFeatureDim, 0.0f);
							std::vector<float> invTransportCov;
							std::vector<float> invPastCov;
							std::vector<float> invStatePastCov;
							std::vector<float> invInnovationCov;
							std::vector<float> theta(static_cast<size_t>(obsDim) * featureDim, 0.0f);
							std::vector<float> stateModel(static_cast<size_t>(stateRank) * stateFeatureDim, 0.0f);
							std::vector<float> innovationGain(static_cast<size_t>(stateRank) * obsDim, 0.0f);
							std::vector<float> nextSigma;
							std::vector<float> nextLeft;
							std::vector<float> nextRight;
							std::vector<float> prevPrevControlMeanHist(controlDim, 0.0f);
							std::vector<float> prevControlMeanHist(controlDim, 0.0f);
							std::vector<float> prevPrevResidualMeanHist(obsDim, 0.0f);
							std::vector<float> prevResidualMeanHist(obsDim, 0.0f);
							std::vector<float> controlVarHist(controlDim, 1.0f);
							std::vector<float> residualVarHist(obsDim, 1.0f);
							std::vector<float> transportPastCov(static_cast<size_t>(trackedLayers) * bundleDim * bundleDim, 0.0f);
							std::vector<float> transportCrossCov(static_cast<size_t>(trackedLayers) * bundleDim * bundleDim, 0.0f);
							std::vector<float> pastCovHist(static_cast<size_t>(featureDim) * featureDim, 0.0f);
							std::vector<float> crossCovHist(static_cast<size_t>(obsDim) * featureDim, 0.0f);
							std::vector<float> thetaHist(static_cast<size_t>(obsDim) * featureDim, 0.0f);
							std::vector<float> statePastCovHist(static_cast<size_t>(stateFeatureDim) * stateFeatureDim, 0.0f);
							std::vector<float> stateCrossCovHist(static_cast<size_t>(stateRank) * stateFeatureDim, 0.0f);
							std::vector<float> innovationCovHist(static_cast<size_t>(obsDim) * obsDim, 0.0f);
							std::vector<float> innovationCrossHist(static_cast<size_t>(stateRank) * obsDim, 0.0f);
							std::vector<float> sigmaHist(stateRank, 0.0f);
							std::vector<float> leftModeHist(static_cast<size_t>(stateRank) * obsDim, 0.0f);
							std::vector<float> rightModeHist(static_cast<size_t>(stateRank) * featureDim, 0.0f);
							std::vector<float> prevPrevLatentHist(stateRank, 0.0f);
							std::vector<float> latentHist(stateRank, 0.0f);
							std::vector<float> poleNumerHist(stateRank, 0.0f);
							std::vector<float> poleDenomHist(stateRank, 0.0f);
							std::vector<float> poleHist(stateRank, 0.0f);

						for (unsigned int i = 0; i < controlDim && (regimeControlBase + i) < aster.prevPrevControlMean.size(); ++i)
							prevPrevControlMeanHist[i] = aster.prevPrevControlMean[regimeControlBase + i];
						for (unsigned int i = 0; i < controlDim && (regimeControlBase + i) < aster.prevControlMean.size(); ++i)
							prevControlMeanHist[i] = aster.prevControlMean[regimeControlBase + i];
						for (unsigned int i = 0; i < obsDim && (regimeObsBase + i) < aster.prevPrevResidualMean.size(); ++i)
							prevPrevResidualMeanHist[i] = aster.prevPrevResidualMean[regimeObsBase + i];
						for (unsigned int i = 0; i < obsDim && (regimeObsBase + i) < aster.prevResidualMean.size(); ++i)
							prevResidualMeanHist[i] = aster.prevResidualMean[regimeObsBase + i];
						for (unsigned int i = 0; i < controlDim && (regimeControlBase + i) < aster.controlVar.size(); ++i)
							controlVarHist[i] = aster.controlVar[regimeControlBase + i];
						for (unsigned int i = 0; i < obsDim && (regimeObsBase + i) < aster.residualVar.size(); ++i)
							residualVarHist[i] = aster.residualVar[regimeObsBase + i];
						for (unsigned int i = 0; i < transportPastCov.size(); ++i)
						{
							const size_t src = static_cast<size_t>(regime) * transportPastCov.size() + i;
							if (src < aster.transportPastCov.size())
								transportPastCov[i] = aster.transportPastCov[src];
							if (src < aster.transportCrossCov.size())
								transportCrossCov[i] = aster.transportCrossCov[src];
						}
						for (unsigned int i = 0; i < pastCovHist.size(); ++i)
						{
							const size_t src = regimeFeatureBase + i;
							if (src < aster.pastCov.size())
								pastCovHist[i] = aster.pastCov[src];
						}
						for (unsigned int i = 0; i < crossCovHist.size(); ++i)
						{
							const size_t src = regimeCrossBase + i;
							if (src < aster.crossCov.size())
								crossCovHist[i] = aster.crossCov[src];
							if (src < aster.theta.size())
								thetaHist[i] = aster.theta[src];
						}
						for (unsigned int i = 0; i < statePastCovHist.size(); ++i)
						{
							const size_t src = regimeStateFeatureBase + i;
							if (src < aster.statePastCov.size())
								statePastCovHist[i] = aster.statePastCov[src];
						}
						for (unsigned int i = 0; i < stateCrossCovHist.size(); ++i)
						{
							const size_t src = regimeStateCrossBase + i;
							if (src < aster.stateCrossCov.size())
								stateCrossCovHist[i] = aster.stateCrossCov[src];
						}
						for (unsigned int i = 0; i < innovationCovHist.size(); ++i)
						{
							const size_t src = regimeInnovationBase + i;
							if (src < aster.innovationCov.size())
								innovationCovHist[i] = aster.innovationCov[src];
						}
						for (unsigned int i = 0; i < innovationCrossHist.size(); ++i)
						{
							const size_t src = regimeInnovationCrossBase + i;
							if (src < aster.innovationCross.size())
								innovationCrossHist[i] = aster.innovationCross[src];
						}
						for (unsigned int i = 0; i < stateRank; ++i)
						{
							const size_t sigmaIdx = static_cast<size_t>(regime) * static_cast<size_t>(stateRank) + i;
							if (sigmaIdx < aster.sigma.size())
								sigmaHist[i] = aster.sigma[sigmaIdx];
							if ((regimeLatentBase + i) < aster.prevPrevLatent.size())
								prevPrevLatentHist[i] = aster.prevPrevLatent[regimeLatentBase + i];
							if ((regimeLatentBase + i) < aster.latent.size())
								latentHist[i] = aster.latent[regimeLatentBase + i];
							if ((regimeLatentBase + i) < aster.poleNumer.size())
								poleNumerHist[i] = aster.poleNumer[regimeLatentBase + i];
							if ((regimeLatentBase + i) < aster.poleDenom.size())
								poleDenomHist[i] = aster.poleDenom[regimeLatentBase + i];
							if ((regimeLatentBase + i) < aster.pole.size())
								poleHist[i] = aster.pole[regimeLatentBase + i];
						}
						for (unsigned int i = 0; i < leftModeHist.size(); ++i)
						{
							const size_t src = regimeLeftBase + i;
							if (src < aster.leftMode.size())
								leftModeHist[i] = aster.leftMode[src];
						}
						for (unsigned int i = 0; i < rightModeHist.size(); ++i)
						{
							const size_t src = regimeRightBase + i;
							if (src < aster.rightMode.size())
								rightModeHist[i] = aster.rightMode[src];
						}

						for (unsigned int i = 0; i < dModel && (regimeFinalRawOff + i) < aster.batchFinalHiddenRawSum.size(); ++i)
							finalHiddenMeanRaw[i] = aster.batchFinalHiddenRawSum[regimeFinalRawOff + i] * invTokenCount;
						for (unsigned int i = 0; i < bundleDim && (regimeBundleOff + i) < aster.batchFinalHiddenSketchSum.size(); ++i)
							finalHiddenMeanSketch[i] = aster.batchFinalHiddenSketchSum[regimeBundleOff + i] * invTokenCount;
						for (unsigned int i = 0; i < bundleDim && (regimeBundleOff + i) < aster.batchResidualSum.size(); ++i)
							residualMean[i] = aster.batchResidualSum[regimeBundleOff + i] * invTokenCount;
						for (unsigned int r = 0; r < supportDim; ++r)
						{
							const size_t roleIndex = regimeSupportOff + r;
							const float count =
							    (roleIndex < aster.batchSupportCount.size()) ? aster.batchSupportCount[roleIndex] : 0.0f;
							supportCount[r] = count;
							if (!(count > 0.0f))
								continue;
							if (roleIndex < aster.batchSupportLogitSum.size())
							{
								supportLogitMean[r] = aster.batchSupportLogitSum[roleIndex] / count;
								residualMean[bundleDim + r] = supportLogitMean[r];
							}
							if (roleIndex < aster.batchSupportResidualSum.size())
								supportResidualMean[r] = aster.batchSupportResidualSum[roleIndex] / count;
							const size_t roleOff = regimeSupportHiddenOff + static_cast<size_t>(r) * static_cast<size_t>(dModel);
							const size_t localRoleOff = static_cast<size_t>(r) * static_cast<size_t>(dModel);
							for (unsigned int i = 0; i < dModel
							                        && (roleOff + i) < aster.batchSupportHiddenRawSum.size(); ++i)
								supportHiddenMeanRaw[localRoleOff + i] =
								    aster.batchSupportHiddenRawSum[roleOff + i] / count;
						}
						for (unsigned int r = 0; r < marginDim; ++r)
						{
							if (!(supportCount[0] > 0.0f) || !(supportCount[r + 1u] > 0.0f))
								continue;
							residualMean[bundleDim + supportDim + r] = supportLogitMean[0] - supportLogitMean[r + 1u];
						}
						const float targetMarginMean =
						    (regime < aster.batchTargetMarginSum.size())
						        ? (aster.batchTargetMarginSum[regime] * invTokenCount)
						        : ((supportDim > 1u) ? (supportLogitMean[0] - supportLogitMean[1]) : 0.0f);
						const float hardNegativeLogitMean =
						    (regime < aster.batchHardNegativeLogitSum.size())
						        ? (aster.batchHardNegativeLogitSum[regime] * invTokenCount)
						        : ((supportDim > 1u) ? supportLogitMean[1] : supportLogitMean[0]);
						const float marginBaseline =
						    (regime < aster.targetMarginEma.size()) ? aster.targetMarginEma[regime] : 0.75f;
						const float hardNegativeBaseline =
						    (regime < aster.hardNegativeLogitEma.size()) ? aster.hardNegativeLogitEma[regime] : 0.0f;
						const float hardShortfallBaseline =
						    (regime < aster.hardMarginShortfallEma.size()) ? aster.hardMarginShortfallEma[regime] : 0.0f;
						const float marginShortfallMean = std::max(0.0f, marginBaseline - targetMarginMean);
						const float negativePressureMean = std::max(0.0f, hardNegativeLogitMean - hardNegativeBaseline);
						const float rawHardMarginShortfallMean = (regime == 3u) ? marginShortfallMean : 0.0f;
						const float hardMarginShortfallMean =
						    std::max(0.0f, rawHardMarginShortfallMean - hardShortfallBaseline);
						const float baselineWorseRate =
						    (regime < aster.batchBaselineWorseSum.size())
						        ? (aster.batchBaselineWorseSum[regime] * invTokenCount)
						        : ((marginShortfallMean > 0.0f) ? 1.0f : 0.0f);
						float targetRepeatHitMean = 0.0f;
						float targetRepeatClosenessMean = 0.0f;
						float hardNegRepeatHitMean = 0.0f;
						float hardNegRepeatClosenessMean = 0.0f;
						if (trackedLayers > 0u && tokenCondDim > 4u)
						{
							for (unsigned int l = 0; l < trackedLayers; ++l)
							{
								const size_t patternSrcOff =
								    (static_cast<size_t>(regime) * static_cast<size_t>(trackedLayers) + l) * static_cast<size_t>(tokenCondDim);
								if ((patternSrcOff + 4u) < aster.batchLayerPatternSum.size())
									targetRepeatHitMean += aster.batchLayerPatternSum[patternSrcOff + 4u] * invTokenCount;
								if ((patternSrcOff + 5u) < aster.batchLayerPatternSum.size())
									targetRepeatClosenessMean += aster.batchLayerPatternSum[patternSrcOff + 5u] * invTokenCount;
								if ((patternSrcOff + 6u) < aster.batchLayerPatternSum.size())
									hardNegRepeatHitMean += aster.batchLayerPatternSum[patternSrcOff + 6u] * invTokenCount;
								if ((patternSrcOff + 7u) < aster.batchLayerPatternSum.size())
									hardNegRepeatClosenessMean += aster.batchLayerPatternSum[patternSrcOff + 7u] * invTokenCount;
							}
							const float invTracked = 1.0f / static_cast<float>(trackedLayers);
							targetRepeatHitMean *= invTracked;
							targetRepeatClosenessMean *= invTracked;
							hardNegRepeatHitMean *= invTracked;
							hardNegRepeatClosenessMean *= invTracked;
						}
						for (size_t slot = 0; slot < aster.batchSupportIds.size(); ++slot)
						{
							const unsigned int vid = aster.batchSupportIds[slot];
							if (vid >= tt.vocabSize)
								continue;
							for (unsigned int r = 0; r < supportDim; ++r)
							{
								if (!(supportCount[r] > 0.0f))
									continue;
								const size_t countOff =
								    (slot * static_cast<size_t>(regimeCount) + regime) * static_cast<size_t>(supportDim) + r;
								if (countOff >= aster.batchSupportRoleCounts.size())
									break;
								const float roleMass = aster.batchSupportRoleCounts[countOff];
								if (!(roleMass > 0.0f))
									continue;
								const float roleWeight = roleMass / supportCount[r];
								if (supportTokenCondDim > 0u)
									aster_sketch_sparse_value(vid,
									                          roleWeight * supportResidualMean[r],
									                          supportTokenCondDim,
									                          0xA57E3001u,
									                          residualMean,
									                          tokenCondOff);
							}
						}
						if ((explicitTokenCondOff + 0u) < tokenCondDim)
							residualMean[tokenCondOff + explicitTokenCondOff + 0u] = marginShortfallMean;
						if ((explicitTokenCondOff + 1u) < tokenCondDim)
							residualMean[tokenCondOff + explicitTokenCondOff + 1u] = negativePressureMean;
						if ((explicitTokenCondOff + 2u) < tokenCondDim)
							residualMean[tokenCondOff + explicitTokenCondOff + 2u] = hardMarginShortfallMean;
						if ((explicitTokenCondOff + 3u) < tokenCondDim)
							residualMean[tokenCondOff + explicitTokenCondOff + 3u] = baselineWorseRate;
						if ((explicitTokenCondOff + 4u) < tokenCondDim)
							residualMean[tokenCondOff + explicitTokenCondOff + 4u] = targetRepeatHitMean;
						if ((explicitTokenCondOff + 5u) < tokenCondDim)
							residualMean[tokenCondOff + explicitTokenCondOff + 5u] = targetRepeatClosenessMean;
						if ((explicitTokenCondOff + 6u) < tokenCondDim)
							residualMean[tokenCondOff + explicitTokenCondOff + 6u] = hardNegRepeatHitMean;
						if ((explicitTokenCondOff + 7u) < tokenCondDim)
							residualMean[tokenCondOff + explicitTokenCondOff + 7u] = hardNegRepeatClosenessMean;
						if (kappaObsDim > 0u)
						{
							for (unsigned int i = 0; i < kappaObsDim; ++i)
							{
								double accum = 0.0;
								for (unsigned int l = 0; l < trackedLayers; ++l)
								{
									const size_t src = regimeLayerKappaBase
									                 + static_cast<size_t>(l) * static_cast<size_t>(kappaObsDim)
									                 + i;
									if (src < aster.batchLayerKappaSum.size())
										accum += static_cast<double>(aster.batchLayerKappaSum[src]) * static_cast<double>(invTokenCount);
								}
								residualMean[kappaOff + i] = static_cast<float>(accum / static_cast<double>(std::max(1u, trackedLayers)));
							}
						}
						for (unsigned int l = 0; l < trackedLayers; ++l)
						{
							const size_t rawOff = regimeLayerRawBase + static_cast<size_t>(l) * static_cast<size_t>(dModel);
							const size_t localRawOff = static_cast<size_t>(l) * static_cast<size_t>(dModel);
							for (unsigned int i = 0; i < dModel && (rawOff + i) < aster.batchLayerHiddenRawSum.size(); ++i)
								layerHiddenMeanRaw[localRawOff + i] =
								    aster.batchLayerHiddenRawSum[rawOff + i] * invTokenCount;
							const size_t attnRawOff = regimeLayerAttnRawBase + static_cast<size_t>(l) * static_cast<size_t>(dModel);
							for (unsigned int i = 0; i < dModel && (attnRawOff + i) < aster.batchLayerAttnRawSum.size(); ++i)
								layerAttnMeanRaw[localRawOff + i] =
								    aster.batchLayerAttnRawSum[attnRawOff + i] * invTokenCount;
							const size_t srcOff = regimeLayerSketchBase + static_cast<size_t>(l) * static_cast<size_t>(bundleDim);
							const size_t localSrcOff = static_cast<size_t>(l) * static_cast<size_t>(bundleDim);
							for (unsigned int i = 0; i < bundleDim; ++i)
								layerHiddenMeanSketch[localSrcOff + i] =
								    aster.batchLayerHiddenSketchSum[srcOff + i] * invTokenCount;
							const size_t attnSrcOff = regimeLayerAttnSketchBase + static_cast<size_t>(l) * static_cast<size_t>(bundleDim);
							for (unsigned int i = 0; i < bundleDim; ++i)
								layerAttnMeanSketch[localSrcOff + i] =
								    aster.batchLayerAttnSketchSum[attnSrcOff + i] * invTokenCount;
							const size_t patternSrcOff =
							    (static_cast<size_t>(regime) * static_cast<size_t>(trackedLayers) + l) * static_cast<size_t>(tokenCondDim);
							const size_t localPatternOff = static_cast<size_t>(l) * static_cast<size_t>(tokenCondDim);
							for (unsigned int i = 0; i < tokenCondDim && (patternSrcOff + i) < aster.batchLayerPatternSum.size(); ++i)
								layerPatternMean[localPatternOff + i] =
								    aster.batchLayerPatternSum[patternSrcOff + i] * invTokenCount;
							const size_t kappaSrcOff =
							    regimeLayerKappaBase + static_cast<size_t>(l) * static_cast<size_t>(kappaObsDim);
							const size_t localKappaOff = static_cast<size_t>(l) * static_cast<size_t>(kappaObsDim);
							for (unsigned int i = 0; i < kappaObsDim && (kappaSrcOff + i) < aster.batchLayerKappaSum.size(); ++i)
								layerKappaMean[localKappaOff + i] =
								    aster.batchLayerKappaSum[kappaSrcOff + i] * invTokenCount;
						}
						const double regimeSetupNs =
						    monotonic_elapsed_ns(regimeStart, monotonic_now());

						const timespec asterTransportStart = monotonic_now();
						for (unsigned int l = 0; l < trackedLayers && ((l + 1u) * controlStride) <= controlDim; ++l)
						{
							const size_t covOff = static_cast<size_t>(l) * static_cast<size_t>(bundleDim) * static_cast<size_t>(bundleDim);
							const size_t rawLayerOff = static_cast<size_t>(l) * static_cast<size_t>(dModel);
							const size_t layerOff = static_cast<size_t>(l) * static_cast<size_t>(bundleDim);
							const size_t patternLayerOff = static_cast<size_t>(l) * static_cast<size_t>(tokenCondDim);
							const size_t kappaLayerOff = static_cast<size_t>(l) * static_cast<size_t>(kappaObsDim);
							const unsigned int hiddenControlOff = l * controlStride;
							const unsigned int attnControlOff = hiddenControlOff + obsDim;
							const unsigned int patternControlOff = attnControlOff + obsDim;
							const unsigned int kappaControlOff = patternControlOff + obsDim;
							for (unsigned int r = 0; r < bundleDim; ++r)
							{
								const size_t rowOff = covOff + static_cast<size_t>(r) * static_cast<size_t>(bundleDim);
								for (unsigned int c = 0; c < bundleDim; ++c)
								{
									const float layerSample = layerHiddenMeanSketch[layerOff + r] * layerHiddenMeanSketch[layerOff + c];
									transportPastCov[rowOff + c] =
									    (betaAster * transportPastCov[rowOff + c]) + ((1.0f - betaAster) * layerSample);
									const float crossSample = finalHiddenMeanSketch[r] * layerHiddenMeanSketch[layerOff + c];
									transportCrossCov[rowOff + c] =
									    (betaAster * transportCrossCov[rowOff + c]) + ((1.0f - betaAster) * crossSample);
								}
							}
							const std::vector<float> pastBlock(
							    transportPastCov.begin() + static_cast<std::ptrdiff_t>(covOff),
							    transportPastCov.begin() + static_cast<std::ptrdiff_t>(covOff + static_cast<size_t>(bundleDim) * bundleDim));
							if (invert_small_dense_row_major(pastBlock, bundleDim, ridge, invTransportCov))
							{
								for (unsigned int r = 0; r < bundleDim; ++r)
								{
									double accum = 0.0;
									for (unsigned int c = 0; c < bundleDim; ++c)
									{
										double transportCoeff = 0.0;
										for (unsigned int k = 0; k < bundleDim; ++k)
										{
											const size_t rowOff = covOff + static_cast<size_t>(r) * static_cast<size_t>(bundleDim);
											transportCoeff += static_cast<double>(transportCrossCov[rowOff + k])
											               * static_cast<double>(invTransportCov[static_cast<size_t>(k) * static_cast<size_t>(bundleDim) + c]);
										}
										accum += transportCoeff * static_cast<double>(layerHiddenMeanSketch[layerOff + c]);
									}
									controlMean[hiddenControlOff + r] = static_cast<float>(accum);
								}
							}
							else
							{
								for (unsigned int i = 0; i < bundleDim; ++i)
									controlMean[hiddenControlOff + i] = layerHiddenMeanSketch[layerOff + i];
							}
							for (unsigned int i = 0; i < bundleDim; ++i)
								controlMean[attnControlOff + i] = layerAttnMeanSketch[layerOff + i];
							for (unsigned int r = 0; r < supportDim; ++r)
							{
								if (!(supportCount[r] > 0.0f))
									continue;
								double hiddenAccum = 0.0;
								double attnAccum = 0.0;
								for (size_t slot = 0; slot < aster.batchSupportIds.size(); ++slot)
								{
									const size_t countOff =
									    (slot * static_cast<size_t>(regimeCount) + regime) * static_cast<size_t>(supportDim) + r;
									if (countOff >= aster.batchSupportRoleCounts.size())
										break;
									const float roleMass = aster.batchSupportRoleCounts[countOff];
									if (!(roleMass > 0.0f))
										continue;
									const unsigned int vid = aster.batchSupportIds[slot];
									if (vid >= tt.vocabSize)
										continue;
									const size_t eOff = static_cast<size_t>(vid) * static_cast<size_t>(dModel);
									double hiddenDot = 0.0;
									double attnDot = 0.0;
									for (unsigned int i = 0; i < dModel; ++i)
									{
										const double eVal = static_cast<double>(tt.tokE[eOff + i]);
										hiddenDot += eVal * static_cast<double>(layerHiddenMeanRaw[rawLayerOff + i]);
										attnDot += eVal * static_cast<double>(layerAttnMeanRaw[rawLayerOff + i]);
									}
									const double roleWeight =
									    static_cast<double>(roleMass) / static_cast<double>(supportCount[r]);
									hiddenAccum += roleWeight * hiddenDot;
									attnAccum += roleWeight * attnDot;
								}
								controlMean[hiddenControlOff + bundleDim + r] = static_cast<float>(hiddenAccum);
								controlMean[attnControlOff + bundleDim + r] = static_cast<float>(attnAccum);
							}
							for (unsigned int r = 0; r < marginDim; ++r)
							{
								controlMean[hiddenControlOff + bundleDim + supportDim + r] =
								    controlMean[hiddenControlOff + bundleDim]
								    - controlMean[hiddenControlOff + bundleDim + r + 1u];
								controlMean[attnControlOff + bundleDim + supportDim + r] =
								    controlMean[attnControlOff + bundleDim]
								    - controlMean[attnControlOff + bundleDim + r + 1u];
							}
							for (size_t slot = 0; slot < aster.batchSupportIds.size(); ++slot)
							{
								const unsigned int vid = aster.batchSupportIds[slot];
								if (vid >= tt.vocabSize)
									continue;
								for (unsigned int r = 0; r < supportDim; ++r)
								{
									if (!(supportCount[r] > 0.0f))
										continue;
									const size_t countOff =
									    (slot * static_cast<size_t>(regimeCount) + regime) * static_cast<size_t>(supportDim) + r;
									if (countOff >= aster.batchSupportRoleCounts.size())
										break;
									const float roleMass = aster.batchSupportRoleCounts[countOff];
									if (!(roleMass > 0.0f))
										continue;
									const float roleWeight = roleMass / supportCount[r];
									if (supportTokenCondDim > 0u)
									{
										aster_sketch_sparse_value(vid,
										                          roleWeight * controlMean[hiddenControlOff + bundleDim + r],
										                          supportTokenCondDim,
										                          0xA57E3001u,
										                          controlMean,
										                          hiddenControlOff + tokenCondOff);
										aster_sketch_sparse_value(vid,
										                          roleWeight * controlMean[attnControlOff + bundleDim + r],
										                          supportTokenCondDim,
										                          0xA57E3001u,
										                          controlMean,
										                          attnControlOff + tokenCondOff);
									}
								}
							}
							for (unsigned int i = 0; i < supportTokenCondDim; ++i)
								controlMean[patternControlOff + tokenCondOff + i] = layerPatternMean[patternLayerOff + i];
							if ((explicitTokenCondOff + 0u) < tokenCondDim)
								controlMean[patternControlOff + tokenCondOff + explicitTokenCondOff + 0u] = marginShortfallMean;
							if ((explicitTokenCondOff + 1u) < tokenCondDim)
								controlMean[patternControlOff + tokenCondOff + explicitTokenCondOff + 1u] = negativePressureMean;
							if ((explicitTokenCondOff + 2u) < tokenCondDim)
								controlMean[patternControlOff + tokenCondOff + explicitTokenCondOff + 2u] = hardMarginShortfallMean;
							if ((explicitTokenCondOff + 3u) < tokenCondDim)
								controlMean[patternControlOff + tokenCondOff + explicitTokenCondOff + 3u] = baselineWorseRate;
							if ((explicitTokenCondOff + 4u) < tokenCondDim)
								controlMean[patternControlOff + tokenCondOff + explicitTokenCondOff + 4u] = targetRepeatHitMean;
							if ((explicitTokenCondOff + 5u) < tokenCondDim)
								controlMean[patternControlOff + tokenCondOff + explicitTokenCondOff + 5u] = targetRepeatClosenessMean;
							if ((explicitTokenCondOff + 6u) < tokenCondDim)
								controlMean[patternControlOff + tokenCondOff + explicitTokenCondOff + 6u] = hardNegRepeatHitMean;
							if ((explicitTokenCondOff + 7u) < tokenCondDim)
								controlMean[patternControlOff + tokenCondOff + explicitTokenCondOff + 7u] = hardNegRepeatClosenessMean;
							for (unsigned int i = 0; i < kappaObsDim; ++i)
								controlMean[kappaControlOff + kappaOff + i] = layerKappaMean[kappaLayerOff + i];
						}
						const double regimeTransportNs =
						    monotonic_elapsed_ns(asterTransportStart, monotonic_now());

						const timespec asterTransferFitStart = monotonic_now();
						for (unsigned int j = 0; j < obsDim; ++j)
						{
							const float secondMoment = residualMean[j] * residualMean[j];
							const float updatedVar =
							    (betaAster * residualVarHist[j]) + ((1.0f - betaAster) * std::max(secondMoment, varEps));
							residualVarHist[j] = std::max(updatedVar, varEps);
							residualStd[j] = sqrtf(residualVarHist[j]);
							prevPrevResidualW[j] =
							    (residualStd[j] > sigmaEps) ? (prevPrevResidualMeanHist[j] / residualStd[j]) : 0.0f;
							currentResidualW[j] = (residualStd[j] > sigmaEps) ? (residualMean[j] / residualStd[j]) : 0.0f;
							feature[j] = prevPrevResidualW[j];
							feature[obsDim + j] =
							    (residualStd[j] > sigmaEps) ? (prevResidualMeanHist[j] / residualStd[j]) : 0.0f;
						}
						for (unsigned int i = 0; i < controlDim; ++i)
						{
							const float secondMoment = controlMean[i] * controlMean[i];
							const float updatedVar =
							    (betaAster * controlVarHist[i]) + ((1.0f - betaAster) * std::max(secondMoment, varEps));
							controlVarHist[i] = std::max(updatedVar, varEps);
							controlStd[i] = sqrtf(controlVarHist[i]);
							prevPrevControlW[i] =
							    (controlStd[i] > sigmaEps) ? (prevPrevControlMeanHist[i] / controlStd[i]) : 0.0f;
							prevControlW[i] =
							    (controlStd[i] > sigmaEps) ? (prevControlMeanHist[i] / controlStd[i]) : 0.0f;
							currentControlW[i] =
							    (controlStd[i] > sigmaEps) ? (controlMean[i] / controlStd[i]) : 0.0f;
							feature[(2u * obsDim) + i] = prevPrevControlW[i];
							feature[(2u * obsDim) + controlDim + i] = prevControlW[i];
							feature[(2u * obsDim) + (2u * controlDim) + i] = currentControlW[i];
						}

						for (unsigned int r = 0; r < featureDim; ++r)
						{
							const size_t rowOff = static_cast<size_t>(r) * static_cast<size_t>(featureDim);
							for (unsigned int c = 0; c < featureDim; ++c)
							{
								const float sample = feature[r] * feature[c];
								pastCovHist[rowOff + c] =
								    (betaAster * pastCovHist[rowOff + c]) + ((1.0f - betaAster) * sample);
							}
						}
						for (unsigned int j = 0; j < obsDim; ++j)
						{
							const size_t rowOff = static_cast<size_t>(j) * static_cast<size_t>(featureDim);
							for (unsigned int c = 0; c < featureDim; ++c)
							{
								const float sample = currentResidualW[j] * feature[c];
								crossCovHist[rowOff + c] =
								    (betaAster * crossCovHist[rowOff + c]) + ((1.0f - betaAster) * sample);
							}
						}
						if (invert_small_dense_row_major(pastCovHist, featureDim, ridge, invPastCov))
						{
							for (unsigned int j = 0; j < obsDim; ++j)
							{
								const size_t rowOff = static_cast<size_t>(j) * static_cast<size_t>(featureDim);
								for (unsigned int c = 0; c < featureDim; ++c)
								{
									double accum = 0.0;
									for (unsigned int k = 0; k < featureDim; ++k)
									{
										accum += static_cast<double>(crossCovHist[rowOff + k])
										      * static_cast<double>(invPastCov[static_cast<size_t>(k) * static_cast<size_t>(featureDim) + c]);
									}
									theta[rowOff + c] = static_cast<float>(accum);
								}
							}
						}

						thetaHist = theta;
						extract_top_singular_modes_row_major(theta, obsDim, featureDim, stateRank,
						                                     nextSigma, nextLeft, nextRight);

						for (unsigned int m = 0; m < stateRank; ++m)
						{
							bool aligned = false;
							if ((m < latentHist.size())
							    && (leftModeHist.size() >= static_cast<size_t>(stateRank) * static_cast<size_t>(obsDim))
							    && (nextLeft.size() >= static_cast<size_t>(stateRank) * static_cast<size_t>(obsDim)))
							{
								double accum = 0.0;
								const size_t nextLeftOff = static_cast<size_t>(m) * static_cast<size_t>(obsDim);
								for (unsigned int pm = 0; pm < stateRank && pm < latentHist.size(); ++pm)
								{
									const size_t prevLeftOff = static_cast<size_t>(pm) * static_cast<size_t>(obsDim);
									double corr = 0.0;
									for (unsigned int j = 0; j < obsDim; ++j)
									{
										corr += static_cast<double>(nextLeft[nextLeftOff + j])
										      * static_cast<double>(leftModeHist[prevLeftOff + j]);
									}
									accum += corr * static_cast<double>(latentHist[pm]);
								}
								prevLatentAligned[m] = static_cast<float>(accum);
								aligned = true;
							}
							if (!aligned && m < latentHist.size())
								prevLatentAligned[m] = latentHist[m];
						}

						for (unsigned int m = 0; m < stateRank; ++m)
						{
							if (m < prevPrevLatentHist.size())
								prevPrevLatent[m] = prevPrevLatentHist[m];
						}
						for (unsigned int m = 0; m < stateRank; ++m)
							stateFeature[m] = prevPrevLatent[m];
						for (unsigned int m = 0; m < stateRank; ++m)
							stateFeature[stateRank + m] = prevLatentAligned[m];
						for (unsigned int i = 0; i < controlDim; ++i)
						{
							stateFeature[(2u * stateRank) + i] = prevPrevControlW[i];
							stateFeature[(2u * stateRank) + controlDim + i] = prevControlW[i];
							stateFeature[(2u * stateRank) + (2u * controlDim) + i] = currentControlW[i];
						}

						for (unsigned int m = 0; m < stateRank; ++m)
						{
							const size_t leftOff = static_cast<size_t>(m) * static_cast<size_t>(obsDim);
							for (unsigned int j = 0; j < obsDim && (leftOff + j) < nextLeft.size(); ++j)
								observedLatent[m] += nextLeft[leftOff + j] * currentResidualW[j];
						}
						const double regimeTransferFitNs =
						    monotonic_elapsed_ns(asterTransferFitStart, monotonic_now());

						const timespec asterStateFitStart = monotonic_now();
						if (statePastCovHist.size() >= static_cast<size_t>(stateFeatureDim) * static_cast<size_t>(stateFeatureDim)
						    && stateCrossCovHist.size() >= static_cast<size_t>(stateRank) * static_cast<size_t>(stateFeatureDim))
						{
							for (unsigned int r = 0; r < stateFeatureDim; ++r)
							{
								const size_t rowOff = static_cast<size_t>(r) * static_cast<size_t>(stateFeatureDim);
								for (unsigned int c = 0; c < stateFeatureDim; ++c)
								{
									const float sample = stateFeature[r] * stateFeature[c];
									statePastCovHist[rowOff + c] =
									    (betaAster * statePastCovHist[rowOff + c]) + ((1.0f - betaAster) * sample);
								}
							}
							for (unsigned int m = 0; m < stateRank; ++m)
							{
								const size_t rowOff = static_cast<size_t>(m) * static_cast<size_t>(stateFeatureDim);
								for (unsigned int c = 0; c < stateFeatureDim; ++c)
								{
									const float sample = observedLatent[m] * stateFeature[c];
									stateCrossCovHist[rowOff + c] =
									    (betaAster * stateCrossCovHist[rowOff + c]) + ((1.0f - betaAster) * sample);
								}
							}
							if (invert_small_dense_row_major(statePastCovHist, stateFeatureDim, ridge, invStatePastCov))
							{
								for (unsigned int m = 0; m < stateRank; ++m)
								{
									const size_t rowOff = static_cast<size_t>(m) * static_cast<size_t>(stateFeatureDim);
									for (unsigned int c = 0; c < stateFeatureDim; ++c)
									{
										double accum = 0.0;
										for (unsigned int k = 0; k < stateFeatureDim; ++k)
										{
											accum += static_cast<double>(stateCrossCovHist[rowOff + k])
											      * static_cast<double>(invStatePastCov[static_cast<size_t>(k) * static_cast<size_t>(stateFeatureDim) + c]);
										}
										stateModel[rowOff + c] = static_cast<float>(accum);
									}
								}
							}
						}

						float maxRowAbs = 0.0f;
						for (unsigned int m = 0; m < stateRank; ++m)
						{
							const size_t rowOff = static_cast<size_t>(m) * static_cast<size_t>(stateFeatureDim);
							float rowAbs = 0.0f;
							for (unsigned int s = 0; s < stateRank; ++s)
								rowAbs += std::fabs(stateModel[rowOff + s]);
							maxRowAbs = std::max(maxRowAbs, rowAbs);
						}
						if (maxRowAbs > ac.asterPoleMax && maxRowAbs > sigmaEps)
						{
							const float scale = ac.asterPoleMax / maxRowAbs;
							for (unsigned int m = 0; m < stateRank; ++m)
							{
								const size_t rowOff = static_cast<size_t>(m) * static_cast<size_t>(stateFeatureDim);
								for (unsigned int s = 0; s < stateRank; ++s)
									stateModel[rowOff + s] *= scale;
							}
						}

						for (unsigned int m = 0; m < stateRank; ++m)
						{
							const size_t rowOff = static_cast<size_t>(m) * static_cast<size_t>(stateFeatureDim);
							double accum = 0.0;
							for (unsigned int s = 0; s < stateRank; ++s)
								accum += static_cast<double>(stateModel[rowOff + s]) * static_cast<double>(prevPrevLatent[s]);
							for (unsigned int s = 0; s < stateRank; ++s)
								accum += static_cast<double>(stateModel[rowOff + stateRank + s]) * static_cast<double>(prevLatentAligned[s]);
							for (unsigned int i = 0; i < controlDim; ++i)
							{
								accum += static_cast<double>(stateModel[rowOff + (2u * stateRank) + i]) * static_cast<double>(prevPrevControlW[i]);
								accum += static_cast<double>(stateModel[rowOff + (2u * stateRank) + controlDim + i]) * static_cast<double>(prevControlW[i]);
								accum += static_cast<double>(stateModel[rowOff + (2u * stateRank) + (2u * controlDim) + i]) * static_cast<double>(currentControlW[i]);
							}
							predictedState[m] = static_cast<float>(accum);
						}
						for (unsigned int m = 0; m < stateRank; ++m)
						{
							const size_t leftOff = static_cast<size_t>(m) * static_cast<size_t>(obsDim);
							for (unsigned int j = 0; j < obsDim && (leftOff + j) < nextLeft.size(); ++j)
								predictedResidualCurrentW[j] += nextLeft[leftOff + j] * predictedState[m];
						}

						double targetNormSq = 0.0;
						double errNormSq = 0.0;
						for (unsigned int j = 0; j < obsDim; ++j)
						{
							const double target = static_cast<double>(currentResidualW[j]);
							const double err = target - static_cast<double>(predictedResidualCurrentW[j]);
							innovation[j] = static_cast<float>(err);
							targetNormSq += target * target;
							errNormSq += err * err;
						}
						float asterPredR2 = 0.0f;
						if (targetNormSq > 1e-12)
							asterPredR2 = static_cast<float>(std::max<double>(0.0, 1.0 - (errNormSq / targetNormSq)));
						const double regimeStateFitNs =
						    monotonic_elapsed_ns(asterStateFitStart, monotonic_now());

						for (unsigned int m = 0; m < stateRank; ++m)
							stateCorrection[m] = observedLatent[m] - predictedState[m];
						const timespec asterInnovationFitStart = monotonic_now();
						if (innovationCovHist.size() >= static_cast<size_t>(obsDim) * static_cast<size_t>(obsDim)
						    && innovationCrossHist.size() >= static_cast<size_t>(stateRank) * static_cast<size_t>(obsDim))
						{
							for (unsigned int r = 0; r < obsDim; ++r)
							{
								const size_t rowOff = static_cast<size_t>(r) * static_cast<size_t>(obsDim);
								for (unsigned int c = 0; c < obsDim; ++c)
								{
									const float sample = innovation[r] * innovation[c];
									innovationCovHist[rowOff + c] =
									    (betaAster * innovationCovHist[rowOff + c]) + ((1.0f - betaAster) * sample);
								}
							}
							for (unsigned int m = 0; m < stateRank; ++m)
							{
								const size_t rowOff = static_cast<size_t>(m) * static_cast<size_t>(obsDim);
								for (unsigned int j = 0; j < obsDim; ++j)
								{
									const float sample = stateCorrection[m] * innovation[j];
									innovationCrossHist[rowOff + j] =
									    (betaAster * innovationCrossHist[rowOff + j]) + ((1.0f - betaAster) * sample);
								}
							}
							if (invert_small_dense_row_major(innovationCovHist, obsDim, ridge, invInnovationCov))
							{
								for (unsigned int m = 0; m < stateRank; ++m)
								{
									const size_t rowOff = static_cast<size_t>(m) * static_cast<size_t>(obsDim);
									for (unsigned int j = 0; j < obsDim; ++j)
									{
										double accum = 0.0;
										for (unsigned int k = 0; k < obsDim; ++k)
										{
											accum += static_cast<double>(innovationCrossHist[rowOff + k])
											      * static_cast<double>(invInnovationCov[static_cast<size_t>(k) * static_cast<size_t>(obsDim) + j]);
										}
										innovationGain[rowOff + j] = static_cast<float>(accum);
									}
								}
							}
						}
						for (unsigned int m = 0; m < stateRank; ++m)
						{
							double accum = static_cast<double>(predictedState[m]);
							const size_t rowOff = static_cast<size_t>(m) * static_cast<size_t>(obsDim);
							for (unsigned int j = 0; j < obsDim; ++j)
								accum += static_cast<double>(innovationGain[rowOff + j]) * static_cast<double>(innovation[j]);
							filteredState[m] = static_cast<float>(accum);
						}
						for (unsigned int m = 0; m < stateRank; ++m)
						{
							const size_t rowOff = static_cast<size_t>(m) * static_cast<size_t>(stateFeatureDim);
							double accum = 0.0;
							for (unsigned int s = 0; s < stateRank; ++s)
								accum += static_cast<double>(stateModel[rowOff + s]) * static_cast<double>(prevLatentAligned[s]);
							for (unsigned int s = 0; s < stateRank; ++s)
								accum += static_cast<double>(stateModel[rowOff + stateRank + s]) * static_cast<double>(filteredState[s]);
							for (unsigned int i = 0; i < controlDim; ++i)
							{
								accum += static_cast<double>(stateModel[rowOff + (2u * stateRank) + i]) * static_cast<double>(prevControlW[i]);
								accum += static_cast<double>(stateModel[rowOff + (2u * stateRank) + controlDim + i]) * static_cast<double>(currentControlW[i]);
								accum += static_cast<double>(stateModel[rowOff + (2u * stateRank) + (2u * controlDim) + i]) * static_cast<double>(currentControlW[i]);
							}
							nextState[m] = static_cast<float>(accum);
						}
						for (unsigned int m = 0; m < stateRank; ++m)
						{
							const size_t leftOff = static_cast<size_t>(m) * static_cast<size_t>(obsDim);
							for (unsigned int j = 0; j < obsDim && (leftOff + j) < nextLeft.size(); ++j)
								predictedResidualNextW[j] += nextLeft[leftOff + j] * nextState[m];
						}
						const double regimeInnovationFitNs =
						    monotonic_elapsed_ns(asterInnovationFitStart, monotonic_now());

						const timespec asterApplyStart = monotonic_now();
						const float asterEdge = (!nextSigma.empty()) ? std::max(0.0f, nextSigma[0]) : 0.0f;
						float asterTransferScale = 0.0f;
						if (asterEdge > ac.asterEdgeThreshold)
							asterTransferScale = (asterEdge - ac.asterEdgeThreshold) / (asterEdge + 1e-6f);
						const atlas::WeightState& headAtlas =
						    tt.tieEmbeddings ? tt.atlasTokE : tt.atlasWOut;
						const float predictiveEvidence =
						    std::max(0.0f, headAtlas.lastSparrowEdge)
						    * std::max(0.0f, headAtlas.lastSparrowHorizontalRatio);
						float meritGeometryTrust = 0.0f;
						if (headAtlas.activeRank > 0u
						    && !headAtlas.fisherDiag.empty()
						    && headAtlas.totalTrace > 1.0e-6f)
						{
							const unsigned int captureRank =
							    std::min(headAtlas.activeRank,
							             static_cast<unsigned int>(headAtlas.fisherDiag.size()));
							double activeTrace = 0.0;
							for (unsigned int c = 0u; c < captureRank; ++c)
								activeTrace += static_cast<double>(headAtlas.fisherDiag[c]);
							const float capture =
							    static_cast<float>(activeTrace / (static_cast<double>(headAtlas.totalTrace) + 1.0e-6));
							meritGeometryTrust =
							    std::max(0.0f,
							             std::min(1.0f, ac.meritGeometryScale * std::max(0.0f, capture)));
						}
						const float outputEvidenceRaw =
						    std::max(0.0f, asterEdge) * std::max(0.0f, asterPredR2);
						float asterMemoryGain = ac.asterMemoryScale
						                      * std::max(0.0f, asterTransferScale)
						                      * std::max(0.0f, asterPredR2);
						if (!is_finite(asterMemoryGain))
							asterMemoryGain = 0.0f;
						float aegisLambdaSpatial = 0.0f;
						float aegisLambdaPredictive = 0.0f;
						float aegisLambdaOutput = 0.0f;
						float aegisPredictivePredicted = 0.0f;
						float aegisPredictiveRealized = predictiveEvidence;
						float aegisOutputPredicted = 0.0f;
						float aegisOutputRealized = outputEvidenceRaw;
						float citadelAnchor = 0.0f;
						float citadelSparrowTrust = 1.0f;
						float rampartTau = 0.0f;
						float rampartBudget = 0.0f;
						float rampartCovariance = 0.0f;
						float rampartSparrowTrust = 1.0f;
						float meritTau = 0.0f;
						float meritBudget = 0.0f;
						float meritCovariance = 0.0f;
						float meritSparrowTrust = 1.0f;
						float strataNullMode = 1.0f;
						float strataPredictiveMode = 0.0f;
						float strataOutputMode = 0.0f;
						float strataCoupledMode = 0.0f;
						float strataBudget = 0.0f;
						float strataNullBenefit = 0.0f;
						float strataPredictiveBenefit = 0.0f;
						float strataOutputBenefit = 0.0f;
						float strataCoupledBenefit = 0.0f;
						float strataSelectedExcess = 0.0f;
						float strataSwitchRate = 0.0f;
						float auroraPatternSignal = 0.0f;
						float auroraKappaSignal = 0.0f;
						float auroraRepeatSignal = 0.0f;
						float auroraRepeatConfusion = 0.0f;
						float auroraRecallPressure = 0.0f;
						float auroraObservationStrength = 0.0f;
						{
							const unsigned int patternObs = std::min<unsigned int>(4u, tokenCondDim);
							double patternAccum = 0.0;
							unsigned int patternCount = 0u;
							for (unsigned int l = 0; l < trackedLayers; ++l)
							{
								const size_t patternLayerOff = static_cast<size_t>(l) * static_cast<size_t>(tokenCondDim);
								for (unsigned int i = 0; i < patternObs; ++i)
								{
									patternAccum += fabs(static_cast<double>(layerPatternMean[patternLayerOff + i]));
									patternCount += 1u;
								}
							}
							if (patternCount > 0u)
							{
								const float meanAbsPattern =
								    static_cast<float>(patternAccum / static_cast<double>(patternCount));
								auroraPatternSignal =
								    std::max(0.0f, std::min(1.0f, meanAbsPattern / (1.0f + meanAbsPattern)));
							}
							if (kappaObsDim > 0u)
							{
								double kappaAccum = 0.0;
								unsigned int kappaCount = 0u;
								for (unsigned int l = 0; l < trackedLayers; ++l)
								{
									const size_t kappaLayerOff = static_cast<size_t>(l) * static_cast<size_t>(kappaObsDim);
									for (unsigned int i = 0; i < kappaObsDim; ++i)
									{
										kappaAccum += fabs(static_cast<double>(layerKappaMean[kappaLayerOff + i]));
										kappaCount += 1u;
									}
								}
								if (kappaCount > 0u)
								{
									const float meanAbsKappa =
									    static_cast<float>(kappaAccum / static_cast<double>(kappaCount));
									auroraKappaSignal =
									    std::max(0.0f, std::min(1.0f, meanAbsKappa / (1.0f + meanAbsKappa)));
								}
							}
							if (tokenCondDim > 4u)
							{
								double targetRepeatHitAccum = 0.0;
								double targetRepeatCloseAccum = 0.0;
								double hardNegRepeatHitAccum = 0.0;
								double hardNegRepeatCloseAccum = 0.0;
								unsigned int repeatCount = 0u;
								for (unsigned int l = 0; l < trackedLayers; ++l)
								{
									const size_t patternLayerOff = static_cast<size_t>(l) * static_cast<size_t>(tokenCondDim);
									if ((patternLayerOff + 7u) >= layerPatternMean.size())
										break;
									targetRepeatHitAccum += layerPatternMean[patternLayerOff + 4u];
									targetRepeatCloseAccum += layerPatternMean[patternLayerOff + 5u];
									hardNegRepeatHitAccum += layerPatternMean[patternLayerOff + 6u];
									hardNegRepeatCloseAccum += layerPatternMean[patternLayerOff + 7u];
									repeatCount += 1u;
								}
								if (repeatCount > 0u)
								{
									const float invRepeatCount = 1.0f / static_cast<float>(repeatCount);
									const float targetRepeatHitMean =
									    static_cast<float>(targetRepeatHitAccum) * invRepeatCount;
									const float targetRepeatCloseMean =
									    static_cast<float>(targetRepeatCloseAccum) * invRepeatCount;
									const float hardNegRepeatHitMean =
									    static_cast<float>(hardNegRepeatHitAccum) * invRepeatCount;
									const float hardNegRepeatCloseMean =
									    static_cast<float>(hardNegRepeatCloseAccum) * invRepeatCount;
									const float targetRepeatSupport =
									    std::max(0.0f,
									             std::min(1.0f,
									                      0.45f * targetRepeatHitMean
									                          + 0.55f * targetRepeatCloseMean));
									const float hardNegRepeatSupport =
									    std::max(0.0f,
									             std::min(1.0f,
									                      0.35f * hardNegRepeatHitMean
									                          + 0.65f * hardNegRepeatCloseMean));
									auroraRepeatSignal = targetRepeatSupport;
									auroraRepeatConfusion =
									    std::max(0.0f,
									             std::min(1.0f,
									                      hardNegRepeatSupport
									                          - 0.50f * targetRepeatSupport));
								}
							}
							const float marginShortfallNorm =
							    std::max(0.0f, std::min(1.0f, marginShortfallMean / (1.0f + marginShortfallMean)));
							const float negativePressureNorm =
							    std::max(0.0f, std::min(1.0f, negativePressureMean / (1.0f + negativePressureMean)));
							const float hardShortfallNorm =
							    std::max(0.0f, std::min(1.0f, hardMarginShortfallMean / (1.0f + hardMarginShortfallMean)));
							auroraRecallPressure =
							    std::max(0.0f,
							             std::min(1.0f,
							                      0.28f * marginShortfallNorm
							                          + 0.16f * negativePressureNorm
							                          + 0.16f * hardShortfallNorm
							                          + 0.18f * baselineWorseRate
							                          + 0.14f * auroraRepeatSignal
							                          + 0.08f * auroraRepeatConfusion));
							auroraObservationStrength =
							    std::max(0.0f,
							             std::min(1.0f,
							                      0.35f * auroraPatternSignal
							                          + 0.15f * auroraKappaSignal
							                          + 0.25f * auroraRecallPressure
							                          + 0.15f * auroraRepeatSignal
							                          + 0.10f * auroraRepeatConfusion));
						}
						if (ac.aegisEnabled || ac.auroraEnabled || ac.seamEnabled || ac.quasarEnabled)
						{
							const float aegisCalibBeta = 0.90f;
							const float aegisPrecisionFloor = 0.025f;
							const float aegisPrecisionMax = 32.0f;
							const float citadelTrustBeta = 0.90f;
							const float strataBenefitBeta = 0.90f;
							aegisPredictivePredicted = std::max(0.0f, aster.aegisPrevPredictiveScore);
							aegisOutputPredicted = std::max(0.0f, aster.aegisPrevOutputScore);
							if (aster.timingBoundaryCount > 0ULL)
							{
								const float predictiveErr =
								    fabsf(aegisPredictivePredicted - aegisPredictiveRealized);
								const float outputErr =
								    fabsf(aegisOutputPredicted - aegisOutputRealized);
								aster.aegisPredictiveErrorEma =
								    aegisCalibBeta * aster.aegisPredictiveErrorEma
								    + (1.0f - aegisCalibBeta) * predictiveErr;
								aster.aegisOutputErrorEma =
								    aegisCalibBeta * aster.aegisOutputErrorEma
								    + (1.0f - aegisCalibBeta) * outputErr;
							}
							const float rawPredictiveTrust =
							    std::min(aegisPrecisionMax,
							             ac.aegisPredictiveScale * predictiveEvidence
							                 / (aster.aegisPredictiveErrorEma + aegisPrecisionFloor));
							const float rawOutputTrust =
							    std::min(aegisPrecisionMax,
							             ac.aegisOutputScale * outputEvidenceRaw
							                 / (aster.aegisOutputErrorEma + aegisPrecisionFloor));
							if (aster.timingBoundaryCount > 0ULL)
							{
								aster.citadelPredictiveTrustEma =
								    citadelTrustBeta * aster.citadelPredictiveTrustEma
								    + (1.0f - citadelTrustBeta) * rawPredictiveTrust;
								aster.citadelOutputTrustEma =
								    citadelTrustBeta * aster.citadelOutputTrustEma
								    + (1.0f - citadelTrustBeta) * rawOutputTrust;
							}
							else
							{
								aster.citadelPredictiveTrustEma = rawPredictiveTrust;
								aster.citadelOutputTrustEma = rawOutputTrust;
							}
							const float disagreementNorm =
							    fabsf(aegisPredictiveRealized - aegisOutputRealized)
							    / (fabsf(aegisPredictiveRealized) + fabsf(aegisOutputRealized) + 1e-6f);
							float residualScale = 1.0f;
							float spatialPrecision = 1.0f;
							if (ac.auroraEnabled)
							{
								const float predictiveBoost =
								    1.0f + 0.30f * auroraPatternSignal + 0.15f * auroraObservationStrength;
								const float outputBoost =
								    1.0f + 0.55f * auroraObservationStrength
								    + 0.25f * auroraRecallPressure + 0.10f * auroraKappaSignal;
								const float adjustedPredictiveTrust =
								    std::max(0.0f, rawPredictiveTrust * predictiveBoost);
								const float adjustedOutputTrust =
								    std::max(0.0f,
								             rawOutputTrust * outputBoost
								                 + 0.50f * auroraObservationStrength
								                 + 0.25f * auroraRecallPressure);
								const float predictiveTrustNorm =
								    adjustedPredictiveTrust / (adjustedPredictiveTrust + 1.0f);
								const float outputTrustNorm =
								    adjustedOutputTrust / (adjustedOutputTrust + 1.0f);
								const float predictiveErrNorm =
								    std::max(0.0f,
								             std::min(1.0f,
								                      aster.aegisPredictiveErrorEma
								                          / (aegisPredictivePredicted
								                             + aegisPredictiveRealized + 1.0e-6f)));
								const float outputErrNorm =
								    std::max(0.0f,
								             std::min(1.0f,
								                      aster.aegisOutputErrorEma
								                          / (aegisOutputPredicted
								                             + aegisOutputRealized + 1.0e-6f)));
								const float uncertainty = 0.5f * (predictiveErrNorm + outputErrNorm);
								const float contextPenalty =
								    batchHardRegimeMass
								    * std::max(0.0f,
								               1.0f - outputTrustNorm
								                         * (0.50f + 0.50f * auroraObservationStrength));
								const float horizonPred =
								    ac.auroraHorizonBlend * aster.citadelPredictiveTrustEma
								    + (1.0f - ac.auroraHorizonBlend) * adjustedPredictiveTrust;
								const float horizonOut =
								    ac.auroraHorizonBlend * aster.citadelOutputTrustEma
								    + (1.0f - ac.auroraHorizonBlend) * adjustedOutputTrust;
								const float horizonPredNorm = horizonPred / (horizonPred + 1.0f);
								const float horizonOutNorm = horizonOut / (horizonOut + 1.0f);
								rampartTau =
								    0.35f + 1.65f
								                * std::max(0.0f,
								                           std::min(1.0f,
								                                    0.45f * uncertainty
								                                        + 0.25f * disagreementNorm
								                                        + 0.20f * contextPenalty
								                                        - 0.10f * auroraObservationStrength));
								rampartCovariance =
								    std::max(0.0f,
								             std::min(0.95f,
								                      0.45f * std::min(horizonPredNorm, horizonOutNorm)
								                          * std::max(0.0f, 1.0f - disagreementNorm)
								                          * std::max(0.0f, 1.0f - 0.5f * uncertainty)
								                          * (0.80f + 0.20f * auroraObservationStrength)
								                          * std::max(0.0f, 1.0f - contextPenalty)));
								const float coupling =
								    rampartCovariance * sqrtf(std::max(0.0f, horizonPred * horizonOut));
								const float h11 = std::max(1.0e-6f, rampartTau + horizonPred);
								const float h22 = std::max(1.0e-6f, rampartTau + horizonOut);
								const float det = std::max(1.0e-6f, h11 * h22 - coupling * coupling);
								float wPredictive =
								    (horizonPred * horizonPredNorm * h22
								     - coupling * horizonOut * horizonOutNorm)
								    / det;
								float wOutput =
								    (h11 * horizonOut * horizonOutNorm
								     - coupling * horizonPred * horizonPredNorm)
								    / det;
								wPredictive = std::max(0.0f, wPredictive);
								wOutput = std::max(0.0f, wOutput);
								const float weightSum = wPredictive + wOutput + 1.0e-6f;
								wPredictive /= weightSum;
								wOutput /= weightSum;
								const float budgetSignal =
								    std::max(0.0f,
								             std::min(1.0f,
								                      0.55f * (0.5f * (horizonPredNorm + horizonOutNorm))
								                          + 0.15f * auroraObservationStrength
								                          + 0.10f * auroraRecallPressure
								                          + 0.20f * std::max(0.0f, 1.0f - uncertainty)
								                          - 0.15f * disagreementNorm
								                          - 0.25f * contextPenalty));
								const float budgetTarget =
								    std::max(0.0f,
								             std::min(1.0f, ac.auroraBudgetMax * budgetSignal));
								rampartBudget =
								    (aster.timingBoundaryCount > 0ULL)
								        ? (citadelTrustBeta * aster.rampartLastBudget
								           + (1.0f - citadelTrustBeta) * budgetTarget)
								        : budgetTarget;
								const float residualNorm =
								    sqrtf(std::max(0.0f,
								                   wPredictive * wPredictive + wOutput * wOutput
								                       + 2.0f * rampartCovariance * wPredictive * wOutput));
								const float budgetScale =
								    (residualNorm > 1.0e-6f)
								        ? std::max(0.0f, std::min(1.0f, rampartBudget / residualNorm))
								        : 0.0f;
								aegisLambdaPredictive =
								    std::max(0.0f, std::min(1.0f, wPredictive * budgetScale));
								aegisLambdaOutput =
								    std::max(0.0f, std::min(1.0f, wOutput * budgetScale));
								aegisLambdaSpatial =
								    std::max(0.0f, std::min(1.0f, 1.0f - aegisLambdaPredictive - aegisLambdaOutput));
								rampartSparrowTrust = aegisLambdaPredictive;
								const float headPriority =
								    std::max(0.0f,
								             std::min(1.0f,
								                      0.30f * horizonOutNorm
								                          + 0.25f * auroraObservationStrength
								                          + 0.20f * auroraRecallPressure
								                          + 0.15f * contextPenalty
								                          + 0.10f * std::max(0.0f, 1.0f - disagreementNorm)));
								const float headGain =
								    1.0f + std::max(0.0f, ac.auroraHeadGain - 1.0f) * headPriority;
								asterMemoryGain *=
								    std::max(0.0f,
								             std::min(1.0f,
								                      aegisLambdaOutput
								                          * (0.85f + 0.30f * auroraObservationStrength)))
								    * headGain;
							}
							else if (ac.seamEnabled)
							{
								const float predictiveTrustNorm =
								    rawPredictiveTrust / (rawPredictiveTrust + 1.0f);
								const float outputTrustNorm =
								    rawOutputTrust / (rawOutputTrust + 1.0f);
								const float predictiveErrNorm =
								    std::max(0.0f,
								             std::min(1.0f,
								                      aster.aegisPredictiveErrorEma
								                          / (aegisPredictivePredicted
								                             + aegisPredictiveRealized + 1.0e-6f)));
								const float outputErrNorm =
								    std::max(0.0f,
								             std::min(1.0f,
								                      aster.aegisOutputErrorEma
								                          / (aegisOutputPredicted
								                             + aegisOutputRealized + 1.0e-6f)));
								const float uncertainty = 0.5f * (predictiveErrNorm + outputErrNorm);
								const float contextPenalty =
								    batchHardRegimeMass * std::max(0.0f, 1.0f - outputTrustNorm);
								const float spatialTarget =
								    meritGeometryTrust - 0.35f * (predictiveTrustNorm + outputTrustNorm)
								    + 0.10f * std::max(0.0f, contextPenalty - predictiveTrustNorm);
								const float predictiveTarget =
								    predictiveTrustNorm * std::max(0.0f, 1.0f - predictiveErrNorm)
								    - 0.20f * disagreementNorm - 0.20f * contextPenalty;
								const float outputTarget =
								    outputTrustNorm * std::max(0.0f, 1.0f - outputErrNorm)
								    + 0.10f * std::max(0.0f, 1.0f - meritGeometryTrust)
								    - 0.15f * disagreementNorm - 0.15f * contextPenalty;
								if (aster.timingBoundaryCount > 0ULL)
								{
									aster.strataNullBenefitEma =
									    0.85f * aster.strataNullBenefitEma
									    + ac.seamMirrorStep * spatialTarget;
									aster.strataPredictiveBenefitEma =
									    0.85f * aster.strataPredictiveBenefitEma
									    + ac.seamMirrorStep * predictiveTarget;
									aster.strataOutputBenefitEma =
									    0.85f * aster.strataOutputBenefitEma
									    + ac.seamMirrorStep * outputTarget;
								}
								else
								{
									aster.strataNullBenefitEma = ac.seamMirrorStep * spatialTarget;
									aster.strataPredictiveBenefitEma = ac.seamMirrorStep * predictiveTarget;
									aster.strataOutputBenefitEma = ac.seamMirrorStep * outputTarget;
								}
								const float maxLogit =
								    std::max(aster.strataNullBenefitEma,
								             std::max(aster.strataPredictiveBenefitEma,
								                      aster.strataOutputBenefitEma));
								const float expSpatial = expf(aster.strataNullBenefitEma - maxLogit);
								const float expPredictive = expf(aster.strataPredictiveBenefitEma - maxLogit);
								const float expOutput = expf(aster.strataOutputBenefitEma - maxLogit);
								const float coordSum = expSpatial + expPredictive + expOutput + 1.0e-6f;
								const float wSpatial = expSpatial / coordSum;
								const float wPredictive = expPredictive / coordSum;
								const float wOutput = expOutput / coordSum;
								const float residualMass =
								    std::max(0.0f, std::min(1.0f, 1.0f - wSpatial));
								meritBudget =
								    std::max(0.0f,
								             std::min(1.0f,
								                      ac.seamBudgetMax * residualMass
								                          * std::max(0.0f,
								                                     1.0f - 0.5f * uncertainty
								                                         - 0.20f * disagreementNorm
								                                         - 0.20f * contextPenalty)));
								const float residualShare = wPredictive + wOutput + 1.0e-6f;
								aegisLambdaPredictive =
								    std::max(0.0f, std::min(1.0f, meritBudget * (wPredictive / residualShare)));
								aegisLambdaOutput =
								    std::max(0.0f, std::min(1.0f, meritBudget * (wOutput / residualShare)));
								aegisLambdaSpatial =
								    std::max(0.0f, std::min(1.0f, 1.0f - aegisLambdaPredictive - aegisLambdaOutput));
								meritSparrowTrust = aegisLambdaPredictive;
								asterMemoryGain *= std::max(0.0f, std::min(1.0f, aegisLambdaOutput));
							}
							else if (ac.quasarEnabled)
							{
								const float predictiveTrustNorm =
								    rawPredictiveTrust / (rawPredictiveTrust + 1.0f);
								const float outputTrustNorm =
								    rawOutputTrust / (rawOutputTrust + 1.0f);
								const float predictiveErrNorm =
								    std::max(0.0f,
								             std::min(1.0f,
								                      aster.aegisPredictiveErrorEma
								                          / (aegisPredictivePredicted
								                             + aegisPredictiveRealized + 1.0e-6f)));
								const float outputErrNorm =
								    std::max(0.0f,
								             std::min(1.0f,
								                      aster.aegisOutputErrorEma
								                          / (aegisOutputPredicted
								                             + aegisOutputRealized + 1.0e-6f)));
								const float uncertainty = 0.5f * (predictiveErrNorm + outputErrNorm);
								const float contextPenalty =
								    batchHardRegimeMass * std::max(0.0f, 1.0f - outputTrustNorm);
								const float residualPressure =
								    std::max(predictiveTrustNorm, outputTrustNorm);
								const float nullSignal =
								    0.20f + 0.45f * uncertainty + 0.30f * disagreementNorm
								    + 0.55f * contextPenalty - 0.25f * residualPressure;
								const float predictiveSignal =
								    predictiveTrustNorm * std::max(0.0f, 1.0f - predictiveErrNorm)
								    + 0.20f * meritGeometryTrust
								    - 0.10f * uncertainty - 0.15f * disagreementNorm
								    - 0.20f * contextPenalty;
								const float outputSignal =
								    outputTrustNorm * std::max(0.0f, 1.0f - outputErrNorm)
								    + 0.10f * std::max(0.0f, 1.0f - meritGeometryTrust)
								    - 0.10f * uncertainty - 0.10f * disagreementNorm
								    - 0.15f * contextPenalty;
								const float coupledSignal =
								    0.45f * (predictiveTrustNorm + outputTrustNorm)
								    + 0.30f * std::min(predictiveTrustNorm, outputTrustNorm)
								    + 0.15f * meritGeometryTrust
								    - 0.15f * uncertainty - 0.25f * disagreementNorm
								    - 0.30f * contextPenalty;
								if (aster.timingBoundaryCount > 0ULL)
								{
									aster.strataNullBenefitEma =
									    strataBenefitBeta * aster.strataNullBenefitEma
									    + (1.0f - strataBenefitBeta) * nullSignal;
									aster.strataPredictiveBenefitEma =
									    strataBenefitBeta * aster.strataPredictiveBenefitEma
									    + (1.0f - strataBenefitBeta) * predictiveSignal;
									aster.strataOutputBenefitEma =
									    strataBenefitBeta * aster.strataOutputBenefitEma
									    + (1.0f - strataBenefitBeta) * outputSignal;
									aster.strataCoupledBenefitEma =
									    strataBenefitBeta * aster.strataCoupledBenefitEma
									    + (1.0f - strataBenefitBeta) * coupledSignal;
								}
								else
								{
									aster.strataNullBenefitEma = nullSignal;
									aster.strataPredictiveBenefitEma = predictiveSignal;
									aster.strataOutputBenefitEma = outputSignal;
									aster.strataCoupledBenefitEma = coupledSignal;
								}
								const float temperature = std::max(0.05f, ac.quasarTemperature);
								const float maxScore =
								    std::max(std::max(aster.strataNullBenefitEma, aster.strataPredictiveBenefitEma),
								             std::max(aster.strataOutputBenefitEma, aster.strataCoupledBenefitEma));
								const float qNull = expf((aster.strataNullBenefitEma - maxScore) / temperature);
								const float qPredictive = expf((aster.strataPredictiveBenefitEma - maxScore) / temperature);
								const float qOutput = expf((aster.strataOutputBenefitEma - maxScore) / temperature);
								const float qCoupled = expf((aster.strataCoupledBenefitEma - maxScore) / temperature);
								const float probSum = qNull + qPredictive + qOutput + qCoupled + 1.0e-6f;
								strataNullMode = qNull / probSum;
								strataPredictiveMode = qPredictive / probSum;
								strataOutputMode = qOutput / probSum;
								strataCoupledMode = qCoupled / probSum;
								strataBudget =
								    std::max(0.0f,
								             std::min(1.0f,
								                      ac.quasarBudgetMax
								                          * std::max(0.0f, 1.0f - strataNullMode)
								                          * std::max(0.0f,
								                                     1.0f - 0.5f * uncertainty
								                                         - 0.15f * disagreementNorm
								                                         - 0.25f * contextPenalty)));
								aegisLambdaPredictive =
								    std::max(0.0f,
								             std::min(1.0f,
								                      strataBudget
								                          * (strataPredictiveMode + 0.5f * strataCoupledMode)));
								aegisLambdaOutput =
								    std::max(0.0f,
								             std::min(1.0f,
								                      strataBudget
								                          * (strataOutputMode + 0.5f * strataCoupledMode)));
								aegisLambdaSpatial =
								    std::max(0.0f, std::min(1.0f, 1.0f - aegisLambdaPredictive - aegisLambdaOutput));
								asterMemoryGain *= std::max(0.0f, std::min(1.0f, aegisLambdaOutput));
								strataSelectedExcess = 1.0f - strataNullMode;
							}
							else if (ac.strataEnabled)
							{
								const float predictiveTrustNorm =
								    rawPredictiveTrust / (rawPredictiveTrust + 1.0f);
								const float outputTrustNorm =
								    rawOutputTrust / (rawOutputTrust + 1.0f);
								const float predictiveErrNorm =
								    std::max(0.0f,
								             std::min(1.0f,
								                      aster.aegisPredictiveErrorEma
								                          / (aegisPredictivePredicted
								                             + aegisPredictiveRealized + 1.0e-6f)));
								const float outputErrNorm =
								    std::max(0.0f,
								             std::min(1.0f,
								                      aster.aegisOutputErrorEma
								                          / (aegisOutputPredicted
								                             + aegisOutputRealized + 1.0e-6f)));
								const float uncertainty =
								    0.5f * (predictiveErrNorm + outputErrNorm);
								const float contextPenalty =
								    batchHardRegimeMass * std::max(0.0f, 1.0f - outputTrustNorm);
								const float predictiveGeom =
								    std::max(0.0f,
								             std::min(1.0f,
								                      meritGeometryTrust
								                          * std::max(0.0f, ac.strataPredictiveGeometryScale)));
								const float coupledGeom =
								    std::max(0.0f,
								             std::min(1.0f,
								                      meritGeometryTrust
								                          * std::max(0.0f, ac.strataCoupledGeometryScale)));
								const float residualPressure =
								    std::max(predictiveTrustNorm, outputTrustNorm);
								const float nullSignal =
								    std::max(0.0f,
								             ac.strataNullBias
								                 + 0.45f * uncertainty
								                 + 0.30f * disagreementNorm
								                 + 0.80f * contextPenalty
								                 - 0.25f * residualPressure);
								const float predictiveSignal =
								    std::max(0.0f,
								             predictiveTrustNorm * std::max(0.0f, 1.0f - predictiveErrNorm)
								                 + 0.25f * predictiveGeom
								                 - 0.10f * uncertainty
								                 - 0.15f * disagreementNorm
								                 - 0.20f * contextPenalty);
								const float outputSignal =
								    std::max(0.0f,
								             outputTrustNorm * std::max(0.0f, 1.0f - outputErrNorm)
								                 + 0.10f * std::max(0.0f, 1.0f - predictiveGeom)
								                 - 0.10f * uncertainty
								                 - 0.10f * disagreementNorm
								                 - 0.15f * contextPenalty);
								const float coupledSignal =
								    std::max(0.0f,
								             0.45f * (predictiveTrustNorm + outputTrustNorm)
								                 + 0.30f * std::min(predictiveTrustNorm, outputTrustNorm)
								                 + 0.20f * coupledGeom
								                 - 0.12f * uncertainty
								                 - 0.20f * disagreementNorm
								                 - 0.35f * contextPenalty);
								const float maxResidualSignal =
								    std::max(predictiveSignal, std::max(outputSignal, coupledSignal));
								const float realizedNullBenefit = nullSignal - maxResidualSignal;
								const float realizedPredictiveBenefit = predictiveSignal - nullSignal;
								const float realizedOutputBenefit = outputSignal - nullSignal;
								const float realizedCoupledBenefit = coupledSignal - nullSignal;
								const float priorNullBenefit = aster.strataNullBenefitEma;
								const float priorPredictiveBenefit = aster.strataPredictiveBenefitEma;
								const float priorOutputBenefit = aster.strataOutputBenefitEma;
								const float priorCoupledBenefit = aster.strataCoupledBenefitEma;
								int previousMode = 0;
								float previousModeWeight = aster.strataLastNullMode;
								if (aster.strataLastPredictiveMode > previousModeWeight)
								{
									previousMode = 1;
									previousModeWeight = aster.strataLastPredictiveMode;
								}
								if (aster.strataLastOutputMode > previousModeWeight)
								{
									previousMode = 2;
									previousModeWeight = aster.strataLastOutputMode;
								}
								if (aster.strataLastCoupledMode > previousModeWeight)
									previousMode = 3;
								float modeScores[4];
								modeScores[0] =
								    priorNullBenefit
								    + 0.25f * realizedNullBenefit
								    + 0.20f * ac.strataNullBias;
								modeScores[1] =
								    priorPredictiveBenefit
								    + 0.25f * realizedPredictiveBenefit
								    + 0.10f * predictiveGeom;
								modeScores[2] =
								    priorOutputBenefit
								    + 0.25f * realizedOutputBenefit;
								modeScores[3] =
								    priorCoupledBenefit
								    + 0.25f * realizedCoupledBenefit
								    + 0.05f * coupledGeom;
								for (int modeIndex = 0; modeIndex < 4; ++modeIndex)
								{
									if (modeIndex != previousMode)
										modeScores[modeIndex] -= std::max(0.0f, ac.strataDwellPenalty);
								}
								int selectedMode = 0;
								float selectedScore = modeScores[0];
								for (int modeIndex = 1; modeIndex < 4; ++modeIndex)
								{
									if (modeScores[modeIndex] > selectedScore)
									{
										selectedMode = modeIndex;
										selectedScore = modeScores[modeIndex];
									}
								}
								strataSelectedExcess =
								    (selectedMode == 1)
								        ? realizedPredictiveBenefit
								        : ((selectedMode == 2)
								               ? realizedOutputBenefit
								               : ((selectedMode == 3)
								                      ? realizedCoupledBenefit
								                      : realizedNullBenefit));
								if (selectedMode != 0 && strataSelectedExcess <= 0.0f)
								{
									selectedMode = 0;
									strataSelectedExcess = realizedNullBenefit;
								}
								const float budgetSignal =
								    std::max(0.0f, std::min(1.0f, strataSelectedExcess));
								const float budgetTarget =
								    std::max(0.0f, std::min(1.0f, ac.strataBudgetMax * budgetSignal));
								strataBudget =
								    (aster.timingBoundaryCount > 0ULL)
								        ? (citadelTrustBeta * aster.strataLastBudget
								           + (1.0f - citadelTrustBeta) * budgetTarget)
								        : budgetTarget;
								strataNullBenefit = realizedNullBenefit;
								strataPredictiveBenefit = realizedPredictiveBenefit;
								strataOutputBenefit = realizedOutputBenefit;
								strataCoupledBenefit = realizedCoupledBenefit;
								strataSwitchRate = (selectedMode == previousMode) ? 0.0f : 1.0f;
								aster.strataNullBenefitEma =
								    (aster.timingBoundaryCount > 0ULL)
								        ? (strataBenefitBeta * aster.strataNullBenefitEma
								           + (1.0f - strataBenefitBeta) * realizedNullBenefit)
								        : realizedNullBenefit;
								aster.strataPredictiveBenefitEma =
								    (aster.timingBoundaryCount > 0ULL)
								        ? (strataBenefitBeta * aster.strataPredictiveBenefitEma
								           + (1.0f - strataBenefitBeta) * realizedPredictiveBenefit)
								        : realizedPredictiveBenefit;
								aster.strataOutputBenefitEma =
								    (aster.timingBoundaryCount > 0ULL)
								        ? (strataBenefitBeta * aster.strataOutputBenefitEma
								           + (1.0f - strataBenefitBeta) * realizedOutputBenefit)
								        : realizedOutputBenefit;
								aster.strataCoupledBenefitEma =
								    (aster.timingBoundaryCount > 0ULL)
								        ? (strataBenefitBeta * aster.strataCoupledBenefitEma
								           + (1.0f - strataBenefitBeta) * realizedCoupledBenefit)
								        : realizedCoupledBenefit;
								if (selectedMode == 1)
								{
									strataNullMode = 0.0f;
									strataPredictiveMode = 1.0f;
									aegisLambdaPredictive = strataBudget;
									aegisLambdaOutput = 0.0f;
								}
								else if (selectedMode == 2)
								{
									strataNullMode = 0.0f;
									strataOutputMode = 1.0f;
									aegisLambdaPredictive = 0.0f;
									aegisLambdaOutput = strataBudget;
								}
								else if (selectedMode == 3)
								{
									strataNullMode = 0.0f;
									strataCoupledMode = 1.0f;
									const float predictiveShare =
									    predictiveTrustNorm / (predictiveTrustNorm + outputTrustNorm + 1.0e-6f);
									aegisLambdaPredictive =
									    std::max(0.0f, std::min(1.0f, strataBudget * predictiveShare));
									aegisLambdaOutput =
									    std::max(0.0f, std::min(1.0f, strataBudget - aegisLambdaPredictive));
								}
								else
								{
									aegisLambdaPredictive = 0.0f;
									aegisLambdaOutput = 0.0f;
								}
								aegisLambdaSpatial =
								    std::max(0.0f, std::min(1.0f, 1.0f - aegisLambdaPredictive - aegisLambdaOutput));
								meritSparrowTrust = aegisLambdaPredictive;
								asterMemoryGain *= std::max(0.0f, std::min(1.0f, aegisLambdaOutput));
							}
							else if (ac.meritEnabled)
							{
								const float predictiveTrustNorm =
								    rawPredictiveTrust / (rawPredictiveTrust + 1.0f);
								const float outputTrustNorm =
								    rawOutputTrust / (rawOutputTrust + 1.0f);
								const float predictiveErrNorm =
								    std::max(0.0f,
								             std::min(1.0f,
								                      aster.aegisPredictiveErrorEma
								                          / (aegisPredictivePredicted
								                             + aegisPredictiveRealized + 1.0e-6f)));
								const float outputErrNorm =
								    std::max(0.0f,
								             std::min(1.0f,
								                      aster.aegisOutputErrorEma
								                          / (aegisOutputPredicted
								                             + aegisOutputRealized + 1.0e-6f)));
								const float uncertainty =
								    0.5f * (predictiveErrNorm + outputErrNorm);
								const float contextPenalty =
								    batchHardRegimeMass * std::max(0.0f, 1.0f - outputTrustNorm);
								const float tauSignal =
								    std::max(0.0f,
								             std::min(1.0f,
								                      0.45f * uncertainty
								                          + 0.30f * disagreementNorm
								                          - 0.30f * meritGeometryTrust
								                          + 0.25f * contextPenalty));
								const float tauTarget =
								    ac.meritTauMin
								    + (ac.meritTauMax - ac.meritTauMin) * tauSignal;
								meritTau =
								    (aster.timingBoundaryCount > 0ULL)
								        ? (citadelTrustBeta * aster.meritLastTau
								           + (1.0f - citadelTrustBeta) * tauTarget)
								        : tauTarget;
								meritCovariance =
								    std::max(0.0f,
								             std::min(0.95f,
								                      ac.meritCovarianceMix
								                          * std::min(predictiveTrustNorm, outputTrustNorm)
								                          * std::max(0.0f, 1.0f - disagreementNorm)));
								const float coupling =
								    meritCovariance
								    * sqrtf(std::max(0.0f, rawPredictiveTrust * rawOutputTrust));
								const float h11 = std::max(1.0e-6f, meritTau + rawPredictiveTrust);
								const float h22 = std::max(1.0e-6f, meritTau + rawOutputTrust);
								const float det = std::max(1.0e-6f, h11 * h22 - coupling * coupling);
								float wPredictive =
								    (rawPredictiveTrust * predictiveTrustNorm * h22
								     - coupling * rawOutputTrust * outputTrustNorm)
								    / det;
								float wOutput =
								    (h11 * rawOutputTrust * outputTrustNorm
								     - coupling * rawPredictiveTrust * predictiveTrustNorm)
								    / det;
								wPredictive = std::max(0.0f, wPredictive);
								wOutput = std::max(0.0f, wOutput);
								const float weightSum = wPredictive + wOutput + 1.0e-6f;
								wPredictive /= weightSum;
								wOutput /= weightSum;
								const float budgetSignal =
								    std::max(0.0f,
								             std::min(1.0f,
								                      0.45f * (0.5f * (predictiveTrustNorm + outputTrustNorm))
								                          + 0.35f * meritGeometryTrust
								                          - 0.15f * disagreementNorm
								                          - 0.15f * uncertainty
								                          - 0.20f * contextPenalty));
								const float budgetTarget =
								    std::max(0.0f, std::min(1.0f, ac.meritBudgetMax * budgetSignal));
								meritBudget =
								    (aster.timingBoundaryCount > 0ULL)
								        ? (citadelTrustBeta * aster.meritLastBudget
								           + (1.0f - citadelTrustBeta) * budgetTarget)
								        : budgetTarget;
								const float residualNorm =
								    sqrtf(std::max(0.0f,
								                   wPredictive * wPredictive + wOutput * wOutput
								                       + 2.0f * meritCovariance * wPredictive * wOutput));
								const float budgetScale =
								    (residualNorm > 1.0e-6f)
								        ? std::max(0.0f, std::min(1.0f, meritBudget / residualNorm))
								        : 0.0f;
								aegisLambdaPredictive =
								    std::max(0.0f, std::min(1.0f, wPredictive * budgetScale));
								aegisLambdaOutput =
								    std::max(0.0f, std::min(1.0f, wOutput * budgetScale));
								aegisLambdaSpatial =
								    std::max(0.0f, std::min(1.0f, 1.0f - aegisLambdaPredictive - aegisLambdaOutput));
								meritSparrowTrust = aegisLambdaPredictive;
								asterMemoryGain *= std::max(0.0f, std::min(1.0f, aegisLambdaOutput));
							}
							else if (ac.rampartEnabled)
							{
								const float predictiveTrustNorm =
								    rawPredictiveTrust / (rawPredictiveTrust + 1.0f);
								const float outputTrustNorm =
								    rawOutputTrust / (rawOutputTrust + 1.0f);
								const float predictiveErrNorm =
								    std::max(0.0f,
								             std::min(1.0f,
								                      aster.aegisPredictiveErrorEma
								                          / (aegisPredictivePredicted
								                             + aegisPredictiveRealized + 1.0e-6f)));
								const float outputErrNorm =
								    std::max(0.0f,
								             std::min(1.0f,
								                      aster.aegisOutputErrorEma
								                          / (aegisOutputPredicted
								                             + aegisOutputRealized + 1.0e-6f)));
								const float uncertainty =
								    0.5f * (predictiveErrNorm + outputErrNorm);
								const float contextPenalty =
								    batchHardRegimeMass * std::max(0.0f, 1.0f - outputTrustNorm);
								const float tauSignal =
								    std::max(0.0f,
								             std::min(1.0f,
								                      0.4f * uncertainty + 0.3f * disagreementNorm
								                          + 0.3f * contextPenalty));
								const float tauTarget =
								    ac.rampartTauMin
								    + (ac.rampartTauMax - ac.rampartTauMin) * tauSignal;
								rampartTau =
								    (aster.timingBoundaryCount > 0ULL)
								        ? (citadelTrustBeta * aster.rampartLastTau
								           + (1.0f - citadelTrustBeta) * tauTarget)
								        : tauTarget;
								rampartCovariance =
								    std::max(0.0f,
								             std::min(0.95f,
								                      ac.rampartCovarianceMix
								                          * std::min(predictiveTrustNorm, outputTrustNorm)
								                          * std::max(0.0f, 1.0f - disagreementNorm)));
								const float coupling =
								    rampartCovariance
								    * sqrtf(std::max(0.0f, rawPredictiveTrust * rawOutputTrust));
								const float h11 = std::max(1.0e-6f, rampartTau + rawPredictiveTrust);
								const float h22 = std::max(1.0e-6f, rampartTau + rawOutputTrust);
								const float det = std::max(1.0e-6f, h11 * h22 - coupling * coupling);
								float wSpatial = (rampartTau + spatialPrecision)
								               / std::max(1.0e-6f, rampartTau + spatialPrecision);
								float wPredictive =
								    (rawPredictiveTrust * predictiveTrustNorm * h22
								     - coupling * rawOutputTrust * outputTrustNorm)
								    / det;
								float wOutput =
								    (h11 * rawOutputTrust * outputTrustNorm
								     - coupling * rawPredictiveTrust * predictiveTrustNorm)
								    / det;
								wSpatial = std::max(0.0f, wSpatial);
								wPredictive = std::max(0.0f, wPredictive);
								wOutput = std::max(0.0f, wOutput);
								const float weightSum = wSpatial + wPredictive + wOutput + 1.0e-6f;
								wSpatial /= weightSum;
								wPredictive /= weightSum;
								wOutput /= weightSum;
								const float budgetSignal =
								    std::max(0.0f,
								             std::min(1.0f,
								                      0.5f * (predictiveTrustNorm + outputTrustNorm)
								                          - 0.20f * disagreementNorm
								                          - 0.15f * uncertainty
								                          - 0.15f * contextPenalty));
								const float budgetTarget =
								    std::max(0.0f, std::min(1.0f, ac.rampartBudgetMax * budgetSignal));
								rampartBudget =
								    (aster.timingBoundaryCount > 0ULL)
								        ? (citadelTrustBeta * aster.rampartLastBudget
								           + (1.0f - citadelTrustBeta) * budgetTarget)
								        : budgetTarget;
								const float residualNorm =
								    sqrtf(std::max(0.0f,
								                   wPredictive * wPredictive + wOutput * wOutput
								                       + 2.0f * rampartCovariance * wPredictive * wOutput));
								const float budgetScale =
								    (residualNorm > 1.0e-6f)
								        ? std::max(0.0f, std::min(1.0f, rampartBudget / residualNorm))
								        : 0.0f;
								aegisLambdaPredictive = std::max(0.0f, std::min(1.0f, wPredictive * budgetScale));
								aegisLambdaOutput = std::max(0.0f, std::min(1.0f, wOutput * budgetScale));
								aegisLambdaSpatial =
								    std::max(0.0f, std::min(1.0f, 1.0f - aegisLambdaPredictive - aegisLambdaOutput));
								rampartSparrowTrust = aegisLambdaPredictive;
								asterMemoryGain *= std::max(0.0f, std::min(1.0f, aegisLambdaOutput));
							}
							else if (ac.citadelEnabled)
							{
								const float predictiveTrustNorm =
								    aster.citadelPredictiveTrustEma / (aster.citadelPredictiveTrustEma + 1.0f);
								const float outputTrustNorm =
								    aster.citadelOutputTrustEma / (aster.citadelOutputTrustEma + 1.0f);
								const float predictiveDominance =
								    std::max(0.0f, predictiveTrustNorm - outputTrustNorm);
								const float citadelConflict =
								    disagreementNorm * predictiveDominance;
								const float hardPressure =
								    batchHardRegimeMass * predictiveDominance * std::max(0.0f, 1.0f - outputTrustNorm);
								const float anchorTarget =
								    std::max(0.0f,
								             std::min(0.95f,
								                      ac.citadelAnchorBase
								                          + ac.citadelHardRegimeScale * hardPressure
								                          + ac.citadelDisagreementScale * citadelConflict));
								citadelAnchor =
								    (aster.timingBoundaryCount > 0ULL)
								        ? (citadelTrustBeta * aster.citadelLastAnchor
								           + (1.0f - citadelTrustBeta) * anchorTarget)
								        : anchorTarget;
								residualScale = std::max(0.0f, 1.0f - citadelAnchor);
								spatialPrecision +=
								    std::max(0.0f, ac.citadelSpatialScale) * citadelAnchor;
							}
							const float predictivePrecision =
							    residualScale
							    * rawPredictiveTrust;
							const float outputPrecision =
							    residualScale
							    * rawOutputTrust;
							if (!ac.rampartEnabled && !ac.meritEnabled && !ac.strataEnabled
							    && !ac.auroraEnabled && !ac.seamEnabled && !ac.quasarEnabled)
							{
								const float totalPrecision =
								    spatialPrecision + predictivePrecision + outputPrecision + 1e-6f;
								aegisLambdaSpatial = spatialPrecision / totalPrecision;
								aegisLambdaPredictive = predictivePrecision / totalPrecision;
								aegisLambdaOutput = outputPrecision / totalPrecision;
								if (ac.citadelEnabled)
								{
									const float predictiveTrustNorm =
									    aster.citadelPredictiveTrustEma / (aster.citadelPredictiveTrustEma + 1.0f);
									citadelSparrowTrust =
									    std::max(0.0f,
									             std::min(1.0f,
									                      predictiveTrustNorm
									                          * std::max(0.0f, 1.0f - citadelAnchor)));
								}
								asterMemoryGain *= std::max(0.0f, std::min(1.0f, aegisLambdaOutput));
							}
						}

						if (asterMemoryGain > 0.0f)
						{
							const float bundleBatchScale = static_cast<float>(tokenCount) * asterMemoryGain;
							const bool sampledSoftmax =
							    (net.trainingConfig.transformer.tokenLmLossKind == glades::TransformerRunConfig::TOKEN_LM_SAMPLED_SOFTMAX);
							if (sampledSoftmax)
							{
								std::sort(aster.batchTouchedIds.begin(), aster.batchTouchedIds.end());
								aster.batchTouchedIds.erase(
								    std::unique(aster.batchTouchedIds.begin(), aster.batchTouchedIds.end()),
								    aster.batchTouchedIds.end());
							}
							const unsigned int applyCount =
							    sampledSoftmax ? static_cast<unsigned int>(aster.batchTouchedIds.size()) : tt.vocabSize;
							for (unsigned int idx = 0; idx < applyCount; ++idx)
							{
								const unsigned int vid = sampledSoftmax ? aster.batchTouchedIds[idx] : idx;
								if (vid >= tt.vocabSize)
									continue;
								const unsigned int h = aster_mix_u32(0xA57E0001u + vid * 0x9e3779b9U);
								const unsigned int bucket = h % bundleDim;
								const float sign = ((h >> 31) != 0u) ? -1.0f : 1.0f;
								const float predictedResidual =
								    clip_maybe(sign * predictedResidualNextW[bucket] * residualStd[bucket],
								               net.trainingConfig.perElementGradClip);
								if (!is_finite(predictedResidual))
									continue;
								tt.gLmBias[static_cast<size_t>(vid)] += bundleBatchScale * predictedResidual;
								const size_t eOff = static_cast<size_t>(vid) * static_cast<size_t>(dModel);
								for (unsigned int i = 0; i < dModel; ++i)
									tt.gTokE[eOff + i] += bundleBatchScale * predictedResidual * finalHiddenMeanRaw[i];
							}
							for (size_t slot = 0; slot < aster.batchSupportIds.size(); ++slot)
							{
								const unsigned int vid = aster.batchSupportIds[slot];
								if (vid >= tt.vocabSize)
									continue;
								const size_t eOff = static_cast<size_t>(vid) * static_cast<size_t>(dModel);
								for (unsigned int r = 0; r < supportDim; ++r)
								{
									if (!(supportCount[r] > 0.0f))
										continue;
									const size_t roleCountOff =
									    (slot * static_cast<size_t>(regimeCount) + regime) * static_cast<size_t>(supportDim) + r;
									if (roleCountOff >= aster.batchSupportRoleCounts.size())
										break;
									const float roleMass = aster.batchSupportRoleCounts[roleCountOff];
									if (!(roleMass > 0.0f))
										continue;
									const float predictedSupportLogit =
									    clip_maybe(predictedResidualNextW[bundleDim + r] * residualStd[bundleDim + r],
									               net.trainingConfig.perElementGradClip);
									if (!is_finite(predictedSupportLogit))
										continue;
									float predictedRoleSignal = predictedSupportLogit;
									if (r == 0u && marginDim > 0u)
									{
										float marginAccum = 0.0f;
										unsigned int marginCount = 0u;
										for (unsigned int mr = 0; mr < marginDim; ++mr)
										{
											const float predictedMargin =
											    clip_maybe(predictedResidualNextW[bundleDim + supportDim + mr]
											                  * residualStd[bundleDim + supportDim + mr],
											               net.trainingConfig.perElementGradClip);
											if (!is_finite(predictedMargin))
												continue;
											marginAccum += predictedMargin;
											marginCount += 1u;
										}
										if (marginCount > 0u)
											predictedRoleSignal += marginAccum / static_cast<float>(marginCount);
									}
									else if (r > 0u && (r - 1u) < marginDim)
									{
										const float predictedMargin =
										    clip_maybe(predictedResidualNextW[bundleDim + supportDim + (r - 1u)]
										                  * residualStd[bundleDim + supportDim + (r - 1u)],
										               net.trainingConfig.perElementGradClip);
										if (is_finite(predictedMargin))
											predictedRoleSignal -= predictedMargin;
									}
									const float delta = asterMemoryGain * roleMass * predictedRoleSignal;
									tt.gLmBias[static_cast<size_t>(vid)] += delta;
									const size_t roleHiddenOff = static_cast<size_t>(r) * static_cast<size_t>(dModel);
									for (unsigned int i = 0; i < dModel; ++i)
										tt.gTokE[eOff + i] += delta * supportHiddenMeanRaw[roleHiddenOff + i];
								}
							}
						}

						const std::vector<float> oldLatent = latentHist;
						for (unsigned int m = 0; m < stateRank; ++m)
						{
							if (m >= poleNumerHist.size() || m >= poleDenomHist.size()
							    || m >= poleHist.size() || m >= latentHist.size())
								continue;
							const size_t rowOff = static_cast<size_t>(m) * static_cast<size_t>(stateFeatureDim);
							const float diagPole = (m < stateRank) ? stateModel[rowOff + m] : 0.0f;
							poleNumerHist[m] = diagPole;
							poleDenomHist[m] = 1.0f;
							poleHist[m] = std::max(-ac.asterPoleMax,
							                       std::min(ac.asterPoleMax, diagPole));
							latentHist[m] = filteredState[m];
						}

						unsigned int asterActiveModes = 0u;
						double nextEffectiveSigmaSq = 0.0;
						for (unsigned int m = 0; m < stateRank; ++m)
						{
							const float sigma = (m < nextSigma.size()) ? nextSigma[m] : 0.0f;
							if (m < sigmaHist.size())
								sigmaHist[m] = sigma;
							nextEffectiveSigmaSq += static_cast<double>(sigma) * static_cast<double>(sigma);
							if (sigma > ac.asterEdgeThreshold)
								asterActiveModes += 1u;
						}
						leftModeHist = nextLeft;
						rightModeHist = nextRight;
						const float regimeEdge =
						    (stateRank > 0u && !sigmaHist.empty()) ? std::max(0.0f, sigmaHist[0]) : 0.0f;
						const float regimeSecondEdge =
						    (stateRank > 1u && sigmaHist.size() > 1u) ? std::max(0.0f, sigmaHist[1]) : 0.0f;
						const float regimeSecondEdgeRatio =
						    (regimeEdge > sigmaEps)
						        ? std::max(0.0f, regimeSecondEdge / (regimeEdge + sigmaEps))
						        : 0.0f;
						const float regimeSigma = static_cast<float>(sqrt(nextEffectiveSigmaSq));
						const float regimePole = !poleHist.empty() ? poleHist[0] : 0.0f;
						for (unsigned int i = 0; i < controlDim; ++i)
							prevPrevControlMeanHist[i] = prevControlMeanHist[i];
						for (unsigned int i = 0; i < controlDim; ++i)
							prevControlMeanHist[i] = controlMean[i];
						for (unsigned int j = 0; j < obsDim; ++j)
							prevPrevResidualMeanHist[j] = prevResidualMeanHist[j];
						for (unsigned int j = 0; j < obsDim; ++j)
							prevResidualMeanHist[j] = residualMean[j];
						for (unsigned int m = 0; m < stateRank && m < prevPrevLatentHist.size() && m < oldLatent.size(); ++m)
							prevPrevLatentHist[m] = oldLatent[m];

						for (unsigned int i = 0; i < controlDim && (regimeControlBase + i) < aster.prevPrevControlMean.size(); ++i)
							aster.prevPrevControlMean[regimeControlBase + i] = prevPrevControlMeanHist[i];
						for (unsigned int i = 0; i < controlDim && (regimeControlBase + i) < aster.prevControlMean.size(); ++i)
							aster.prevControlMean[regimeControlBase + i] = prevControlMeanHist[i];
						for (unsigned int i = 0; i < obsDim && (regimeObsBase + i) < aster.prevPrevResidualMean.size(); ++i)
							aster.prevPrevResidualMean[regimeObsBase + i] = prevPrevResidualMeanHist[i];
						for (unsigned int i = 0; i < obsDim && (regimeObsBase + i) < aster.prevResidualMean.size(); ++i)
							aster.prevResidualMean[regimeObsBase + i] = prevResidualMeanHist[i];
						for (unsigned int i = 0; i < controlDim && (regimeControlBase + i) < aster.controlVar.size(); ++i)
							aster.controlVar[regimeControlBase + i] = controlVarHist[i];
						for (unsigned int i = 0; i < obsDim && (regimeObsBase + i) < aster.residualVar.size(); ++i)
							aster.residualVar[regimeObsBase + i] = residualVarHist[i];
						for (unsigned int i = 0; i < transportPastCov.size(); ++i)
						{
							const size_t dst = static_cast<size_t>(regime) * transportPastCov.size() + i;
							if (dst < aster.transportPastCov.size())
								aster.transportPastCov[dst] = transportPastCov[i];
							if (dst < aster.transportCrossCov.size())
								aster.transportCrossCov[dst] = transportCrossCov[i];
						}
						for (unsigned int i = 0; i < pastCovHist.size(); ++i)
						{
							const size_t dst = regimeFeatureBase + i;
							if (dst < aster.pastCov.size())
								aster.pastCov[dst] = pastCovHist[i];
						}
						for (unsigned int i = 0; i < crossCovHist.size(); ++i)
						{
							const size_t dst = regimeCrossBase + i;
							if (dst < aster.crossCov.size())
								aster.crossCov[dst] = crossCovHist[i];
							if (dst < aster.theta.size())
								aster.theta[dst] = thetaHist[i];
						}
						for (unsigned int i = 0; i < statePastCovHist.size(); ++i)
						{
							const size_t dst = regimeStateFeatureBase + i;
							if (dst < aster.statePastCov.size())
								aster.statePastCov[dst] = statePastCovHist[i];
						}
						for (unsigned int i = 0; i < stateCrossCovHist.size(); ++i)
						{
							const size_t dst = regimeStateCrossBase + i;
							if (dst < aster.stateCrossCov.size())
								aster.stateCrossCov[dst] = stateCrossCovHist[i];
						}
						for (unsigned int i = 0; i < innovationCovHist.size(); ++i)
						{
							const size_t dst = regimeInnovationBase + i;
							if (dst < aster.innovationCov.size())
								aster.innovationCov[dst] = innovationCovHist[i];
						}
						for (unsigned int i = 0; i < innovationCrossHist.size(); ++i)
						{
							const size_t dst = regimeInnovationCrossBase + i;
							if (dst < aster.innovationCross.size())
								aster.innovationCross[dst] = innovationCrossHist[i];
						}
						for (unsigned int i = 0; i < stateRank; ++i)
						{
							const size_t sigmaIdx = static_cast<size_t>(regime) * static_cast<size_t>(stateRank) + i;
							if (sigmaIdx < aster.sigma.size())
								aster.sigma[sigmaIdx] = sigmaHist[i];
							if ((regimeLatentBase + i) < aster.prevPrevLatent.size())
								aster.prevPrevLatent[regimeLatentBase + i] = prevPrevLatentHist[i];
							if ((regimeLatentBase + i) < aster.latent.size())
								aster.latent[regimeLatentBase + i] = latentHist[i];
							if ((regimeLatentBase + i) < aster.poleNumer.size())
								aster.poleNumer[regimeLatentBase + i] = poleNumerHist[i];
							if ((regimeLatentBase + i) < aster.poleDenom.size())
								aster.poleDenom[regimeLatentBase + i] = poleDenomHist[i];
							if ((regimeLatentBase + i) < aster.pole.size())
								aster.pole[regimeLatentBase + i] = poleHist[i];
						}
						for (unsigned int i = 0; i < leftModeHist.size(); ++i)
						{
							const size_t dst = regimeLeftBase + i;
							if (dst < aster.leftMode.size())
								aster.leftMode[dst] = leftModeHist[i];
						}
						for (unsigned int i = 0; i < rightModeHist.size(); ++i)
						{
							const size_t dst = regimeRightBase + i;
							if (dst < aster.rightMode.size())
								aster.rightMode[dst] = rightModeHist[i];
						}
						if (regime < aster.targetMarginEma.size())
							aster.targetMarginEma[regime] =
							    (marginBaselineBeta * aster.targetMarginEma[regime]) + ((1.0f - marginBaselineBeta) * targetMarginMean);
						if (regime < aster.hardNegativeLogitEma.size())
							aster.hardNegativeLogitEma[regime] =
							    (marginBaselineBeta * aster.hardNegativeLogitEma[regime]) + ((1.0f - marginBaselineBeta) * hardNegativeLogitMean);
						if (regime < aster.hardMarginShortfallEma.size())
							aster.hardMarginShortfallEma[regime] =
							    (marginBaselineBeta * aster.hardMarginShortfallEma[regime]) + ((1.0f - marginBaselineBeta) * rawHardMarginShortfallMean);

						const double regimeWeight =
						    static_cast<double>(tokenCount) / static_cast<double>(std::max(1u, aster.batchTokenCount));
						weightSum += regimeWeight;
						weightedActiveModes += regimeWeight * static_cast<double>(asterActiveModes);
						if (asterActiveModes >= 2u)
							weightedSecondModeFrac += regimeWeight;
						weightedEdge += regimeWeight * static_cast<double>(regimeEdge);
						weightedSecondEdge += regimeWeight * static_cast<double>(regimeSecondEdge);
						weightedSecondEdgeRatio += regimeWeight * static_cast<double>(regimeSecondEdgeRatio);
						weightedSigma += regimeWeight * static_cast<double>(regimeSigma);
						weightedPredR2 += regimeWeight * static_cast<double>(asterPredR2);
						weightedMemoryGain += regimeWeight * static_cast<double>(asterMemoryGain);
						weightedPole += regimeWeight * static_cast<double>(regimePole);
						weightedAegisLambdaSpatial += regimeWeight * static_cast<double>(aegisLambdaSpatial);
						weightedAegisLambdaPredictive += regimeWeight * static_cast<double>(aegisLambdaPredictive);
						weightedAegisLambdaOutput += regimeWeight * static_cast<double>(aegisLambdaOutput);
						weightedAegisPredictivePredicted += regimeWeight * static_cast<double>(aegisPredictivePredicted);
						weightedAegisPredictiveRealized += regimeWeight * static_cast<double>(aegisPredictiveRealized);
						weightedAegisOutputPredicted += regimeWeight * static_cast<double>(aegisOutputPredicted);
						weightedAegisOutputRealized += regimeWeight * static_cast<double>(aegisOutputRealized);
						weightedAegisPredictiveError += regimeWeight * static_cast<double>(aster.aegisPredictiveErrorEma);
						weightedAegisOutputError += regimeWeight * static_cast<double>(aster.aegisOutputErrorEma);
						weightedAegisDisagreement += regimeWeight * static_cast<double>(fabsf(aegisPredictiveRealized - aegisOutputRealized));
						weightedCitadelAnchor += regimeWeight * static_cast<double>(citadelAnchor);
						weightedCitadelHardRegimeMass += regimeWeight * static_cast<double>(batchHardRegimeMass);
						weightedCitadelSparrowTrust += regimeWeight * static_cast<double>(citadelSparrowTrust);
						weightedRampartTau += regimeWeight * static_cast<double>(rampartTau);
						weightedRampartBudget += regimeWeight * static_cast<double>(rampartBudget);
						weightedRampartCovariance += regimeWeight * static_cast<double>(rampartCovariance);
						weightedRampartSparrowTrust += regimeWeight * static_cast<double>(rampartSparrowTrust);
						weightedMeritTau += regimeWeight * static_cast<double>(meritTau);
						weightedMeritBudget += regimeWeight * static_cast<double>(meritBudget);
						weightedMeritCovariance += regimeWeight * static_cast<double>(meritCovariance);
						weightedMeritSparrowTrust += regimeWeight * static_cast<double>(meritSparrowTrust);
						weightedMeritGeometryTrust += regimeWeight * static_cast<double>(meritGeometryTrust);
						weightedStrataNullMode += regimeWeight * static_cast<double>(strataNullMode);
						weightedStrataPredictiveMode += regimeWeight * static_cast<double>(strataPredictiveMode);
						weightedStrataOutputMode += regimeWeight * static_cast<double>(strataOutputMode);
						weightedStrataCoupledMode += regimeWeight * static_cast<double>(strataCoupledMode);
						weightedStrataBudget += regimeWeight * static_cast<double>(strataBudget);
						weightedStrataNullBenefit += regimeWeight * static_cast<double>(strataNullBenefit);
						weightedStrataPredictiveBenefit += regimeWeight * static_cast<double>(strataPredictiveBenefit);
						weightedStrataOutputBenefit += regimeWeight * static_cast<double>(strataOutputBenefit);
						weightedStrataCoupledBenefit += regimeWeight * static_cast<double>(strataCoupledBenefit);
						weightedStrataSelectedExcess += regimeWeight * static_cast<double>(strataSelectedExcess);
						weightedStrataSwitchRate += regimeWeight * static_cast<double>(strataSwitchRate);
						maxActiveModes = std::max(maxActiveModes, asterActiveModes);
						asterSetupNs += regimeSetupNs;
						asterTransportNs += regimeTransportNs;
						asterTransferFitNs += regimeTransferFitNs;
						asterStateFitNs += regimeStateFitNs;
						asterInnovationFitNs += regimeInnovationFitNs;
						asterApplyNs += monotonic_elapsed_ns(asterApplyStart, monotonic_now());
						}

						if (weightSum > 0.0)
						{
							aster.lastActiveModes = maxActiveModes;
							aster.lastEdge = static_cast<float>(weightedEdge / weightSum);
							aster.lastSecondEdge = static_cast<float>(weightedSecondEdge / weightSum);
							aster.lastSecondEdgeRatio = static_cast<float>(weightedSecondEdgeRatio / weightSum);
							aster.lastSigma = static_cast<float>(weightedSigma / weightSum);
							aster.lastPredR2 = static_cast<float>(weightedPredR2 / weightSum);
							aster.lastMemoryGain = static_cast<float>(weightedMemoryGain / weightSum);
							aster.lastPoleSummary = static_cast<float>(weightedPole / weightSum);
							aster.aegisPrevPredictiveScore = static_cast<float>(weightedAegisPredictiveRealized / weightSum);
							aster.aegisPrevOutputScore = static_cast<float>(weightedAegisOutputRealized / weightSum);
							aster.aegisLastLambdaSpatial = static_cast<float>(weightedAegisLambdaSpatial / weightSum);
							aster.aegisLastLambdaPredictive = static_cast<float>(weightedAegisLambdaPredictive / weightSum);
							aster.aegisLastLambdaOutput = static_cast<float>(weightedAegisLambdaOutput / weightSum);
							aster.aegisLastPredictivePredicted = static_cast<float>(weightedAegisPredictivePredicted / weightSum);
							aster.aegisLastPredictiveRealized = static_cast<float>(weightedAegisPredictiveRealized / weightSum);
							aster.aegisLastOutputPredicted = static_cast<float>(weightedAegisOutputPredicted / weightSum);
							aster.aegisLastOutputRealized = static_cast<float>(weightedAegisOutputRealized / weightSum);
							aster.aegisLastChannelDisagreement = static_cast<float>(weightedAegisDisagreement / weightSum);
							aster.citadelLastAnchor = static_cast<float>(weightedCitadelAnchor / weightSum);
							aster.citadelLastHardRegimeMass = static_cast<float>(weightedCitadelHardRegimeMass / weightSum);
							aster.citadelLastSparrowTrust = static_cast<float>(weightedCitadelSparrowTrust / weightSum);
							aster.rampartLastTau = static_cast<float>(weightedRampartTau / weightSum);
							aster.rampartLastBudget = static_cast<float>(weightedRampartBudget / weightSum);
							aster.rampartLastCovariance = static_cast<float>(weightedRampartCovariance / weightSum);
							aster.rampartLastSparrowTrust = static_cast<float>(weightedRampartSparrowTrust / weightSum);
							aster.meritLastTau = static_cast<float>(weightedMeritTau / weightSum);
							aster.meritLastBudget = static_cast<float>(weightedMeritBudget / weightSum);
							aster.meritLastCovariance = static_cast<float>(weightedMeritCovariance / weightSum);
							aster.meritLastSparrowTrust = static_cast<float>(weightedMeritSparrowTrust / weightSum);
							aster.meritLastGeometryTrust = static_cast<float>(weightedMeritGeometryTrust / weightSum);
							aster.strataLastNullMode = static_cast<float>(weightedStrataNullMode / weightSum);
							aster.strataLastPredictiveMode = static_cast<float>(weightedStrataPredictiveMode / weightSum);
							aster.strataLastOutputMode = static_cast<float>(weightedStrataOutputMode / weightSum);
							aster.strataLastCoupledMode = static_cast<float>(weightedStrataCoupledMode / weightSum);
							aster.strataLastBudget = static_cast<float>(weightedStrataBudget / weightSum);
							aster.strataLastNullBenefit = static_cast<float>(weightedStrataNullBenefit / weightSum);
							aster.strataLastPredictiveBenefit = static_cast<float>(weightedStrataPredictiveBenefit / weightSum);
							aster.strataLastOutputBenefit = static_cast<float>(weightedStrataOutputBenefit / weightSum);
							aster.strataLastCoupledBenefit = static_cast<float>(weightedStrataCoupledBenefit / weightSum);
							aster.strataLastSelectedExcess = static_cast<float>(weightedStrataSelectedExcess / weightSum);
							aster.strataLastSwitchRate = static_cast<float>(weightedStrataSwitchRate / weightSum);
						}
						else
						{
							aster.lastActiveModes = 0u;
							aster.lastEdge = 0.0f;
							aster.lastSecondEdge = 0.0f;
							aster.lastSecondEdgeRatio = 0.0f;
							aster.lastSigma = 0.0f;
							aster.lastPredR2 = 0.0f;
							aster.lastMemoryGain = 0.0f;
							aster.lastPoleSummary = 0.0f;
							aster.aegisPrevPredictiveScore = 0.0f;
							aster.aegisPrevOutputScore = 0.0f;
							aster.aegisLastLambdaSpatial = 0.0f;
							aster.aegisLastLambdaPredictive = 0.0f;
							aster.aegisLastLambdaOutput = 0.0f;
							aster.aegisLastPredictivePredicted = 0.0f;
							aster.aegisLastPredictiveRealized = 0.0f;
							aster.aegisLastOutputPredicted = 0.0f;
							aster.aegisLastOutputRealized = 0.0f;
							aster.aegisLastChannelDisagreement = 0.0f;
							aster.citadelLastAnchor = 0.0f;
							aster.citadelLastHardRegimeMass = 0.0f;
							aster.citadelLastSparrowTrust = 1.0f;
							aster.rampartLastTau = 0.0f;
							aster.rampartLastBudget = 0.0f;
							aster.rampartLastCovariance = 0.0f;
							aster.rampartLastSparrowTrust = 1.0f;
							aster.meritLastTau = 0.0f;
							aster.meritLastBudget = 0.0f;
							aster.meritLastCovariance = 0.0f;
							aster.meritLastSparrowTrust = 1.0f;
							aster.meritLastGeometryTrust = 0.0f;
							aster.strataLastNullMode = 1.0f;
							aster.strataLastPredictiveMode = 0.0f;
							aster.strataLastOutputMode = 0.0f;
							aster.strataLastCoupledMode = 0.0f;
							aster.strataLastBudget = 0.0f;
							aster.strataLastNullBenefit = 0.0f;
							aster.strataLastPredictiveBenefit = 0.0f;
							aster.strataLastOutputBenefit = 0.0f;
							aster.strataLastCoupledBenefit = 0.0f;
							aster.strataLastSelectedExcess = 0.0f;
							aster.strataLastSwitchRate = 0.0f;
						}
						const timespec asterBoundaryEnd = monotonic_now();
						aster.timingBoundaryCount += 1ULL;
						aster.totalBoundaryNs += monotonic_elapsed_ns(asterBoundaryStart, asterBoundaryEnd);
						aster.totalSetupNs += asterSetupNs;
						aster.totalTransportNs += asterTransportNs;
						aster.totalTransferFitNs += asterTransferFitNs;
						aster.totalStateFitNs += asterStateFitNs;
						aster.totalInnovationFitNs += asterInnovationFitNs;
						aster.totalApplyNs += asterApplyNs;
					}

					if (geodeEnabled)
					{
						if (!Geode::update_weight(tt.tokE, tt.vTokE, tt.v2TokE, tt.gTokE,
						                          tt.atlasTokE, tt.vocabSize, dmTT, lr,
						                          beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                          invBatch, gradScale, wd1, wd2,
						                          ac, net.rngEngine, net.getLogger(), "tr.tokE"))
						{
							net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
								"SGDHelper_Transformer: GEODE tokE update entered NaN recovery");
							net.storeRunningFlag(false);
							return false;
						}
						Adam::update_param(tt.lmBias, tt.mLmBias, tt.v2LmBias, tt.gLmBias,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                   invBatch, gradScale);
					}
					else if (echoEnabled)
					{
						const bool useEchoTokE = echo_scope_uses_head_matrix(ac, tt.vocabSize, dmTT);
						if (useEchoTokE)
						{
							if (!atlas::echoUpdate(tt.echoTokE, &tt.tokE[0], &tt.vTokE[0], &tt.v2TokE[0], &tt.gTokE[0],
							                       tt.vocabSize, dmTT, lr,
							                       beta1, beta2, inv1mB1t, inv1mB2t, eps,
							                       invBatch, gradScale, wd1, wd2,
							                       tt.optimizerStep,
							                       ac, net.getLogger(), "tr.tokE"))
							{
								net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
									"SGDHelper_Transformer: ECHO tokE update entered NaN recovery");
								net.storeRunningFlag(false);
								return false;
							}
						}
						else
							Adam::update_weight(tt.tokE, tt.vTokE, tt.v2TokE, tt.gTokE,
							                    lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
							                    invBatch, gradScale, wd1, wd2);
						Adam::update_param(tt.lmBias, tt.mLmBias, tt.v2LmBias, tt.gLmBias,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                   invBatch, gradScale);
					}
					else if (bimapEnabled)
					{
						const bool useBiMAPTokE = bimap_scope_uses_head(ac);
						if (useBiMAPTokE)
						{
							if (!atlas::bimapUpdate(tt.bimapTokE, &tt.tokE[0], &tt.vTokE[0], &tt.v2TokE[0], &tt.gTokE[0],
							                        tt.vocabSize, dmTT, lr,
							                        beta1, beta2, inv1mB1t, inv1mB2t, eps,
							                        invBatch, gradScale, wd1, wd2,
							                        ac, net.getLogger(), "tr.tokE"))
							{
								net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
									"SGDHelper_Transformer: BiMAP tokE update entered NaN recovery");
								net.storeRunningFlag(false);
								return false;
							}
						}
						else
							Adam::update_weight(tt.tokE, tt.vTokE, tt.v2TokE, tt.gTokE,
							                    lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
							                    invBatch, gradScale, wd1, wd2);
						Adam::update_param(tt.lmBias, tt.mLmBias, tt.v2LmBias, tt.gLmBias,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                   invBatch, gradScale);
					}
					else if (pactEnabled)
					{
						if (!atlas::pactUpdate(tt.pactTokE, &tt.tokE[0], &tt.vTokE[0], &tt.v2TokE[0], &tt.gTokE[0],
						                       tt.vocabSize, dmTT, lr,
						                       beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                       invBatch, gradScale, wd1, wd2,
						                       ac, net.getLogger(), "tr.tokE"))
						{
							net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
								"SGDHelper_Transformer: PACT tokE update entered NaN recovery");
							net.storeRunningFlag(false);
							return false;
						}
						Adam::update_param(tt.lmBias, tt.mLmBias, tt.v2LmBias, tt.gLmBias,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                   invBatch, gradScale);
					}
					else if (racerEnabled)
					{
						if (!atlas::racerUpdate(tt.racerTokE, &tt.tokE[0], &tt.vTokE[0], &tt.v2TokE[0], &tt.gTokE[0],
						                        tt.vocabSize, dmTT, lr,
						                        beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                        invBatch, gradScale, wd1, wd2,
						                        ac, net.getLogger(), "tr.tokE"))
						{
							net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
								"SGDHelper_Transformer: RACER tokE update entered NaN recovery");
							net.storeRunningFlag(false);
							return false;
						}
						Adam::update_param(tt.lmBias, tt.mLmBias, tt.v2LmBias, tt.gLmBias,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                   invBatch, gradScale);
					}
					else if (kronEnabled)
					{
						if (!atlas::kronUpdate(tt.kronTokE, &tt.tokE[0], &tt.vTokE[0], &tt.v2TokE[0], &tt.gTokE[0],
						                       tt.vocabSize, dmTT, lr,
						                       beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                       invBatch, gradScale, wd1, wd2,
						                       ac, net.getLogger(), "tr.tokE"))
						{
							net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
								"SGDHelper_Transformer: KRON tokE update entered NaN recovery");
							net.storeRunningFlag(false);
							return false;
						}
						Adam::update_param(tt.lmBias, tt.mLmBias, tt.v2LmBias, tt.gLmBias,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                   invBatch, gradScale);
					}
					else if (matraEnabled)
					{
						if (!atlas::matraUpdate(tt.matraTokE, &tt.tokE[0], &tt.vTokE[0], &tt.v2TokE[0], &tt.gTokE[0],
						                        tt.vocabSize, dmTT, lr,
						                        beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                        invBatch, gradScale, wd1, wd2,
						                        ac, net.getLogger(), "tr.tokE"))
						{
							net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
								"SGDHelper_Transformer: MATRA tokE update entered NaN recovery");
							net.storeRunningFlag(false);
							return false;
						}
						Adam::update_param(tt.lmBias, tt.mLmBias, tt.v2LmBias, tt.gLmBias,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                   invBatch, gradScale);
					}
					else if (argosEnabled)
					{
						if (argos_scope_uses_head(ac))
						{
							if (!atlas::argosUpdateWithRole(tt.argosTokE, &tt.tokE[0], &tt.vTokE[0], &tt.v2TokE[0], &tt.gTokE[0],
							                                tt.vocabSize, dmTT, lr,
							                                beta1, beta2, inv1mB1t, inv1mB2t, eps,
							                                invBatch, gradScale, wd1, wd2,
							                                ac,
							                                argos_role_flags_for_head(),
							                                net.getLogger(), "tr.tokE"))
							{
								net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
									"SGDHelper_Transformer: ARGOS tokE update entered NaN recovery");
								net.storeRunningFlag(false);
								return false;
							}
						}
						else
						{
							Adam::update_weight(tt.tokE, tt.vTokE, tt.v2TokE, tt.gTokE,
							                    lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
							                    invBatch, gradScale, wd1, wd2);
						}
						Adam::update_param(tt.lmBias, tt.mLmBias, tt.v2LmBias, tt.gLmBias,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                   invBatch, gradScale);
					}
					else if (muonEnabled)
					{
						if (!atlas::muonUpdate(tt.muonTokE, &tt.tokE[0], &tt.vTokE[0], &tt.v2TokE[0], &tt.gTokE[0],
						                       tt.vocabSize, dmTT, lr,
						                       beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                       invBatch, gradScale, wd1, wd2,
						                       ac, net.getLogger(), "tr.tokE"))
						{
							net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
								"SGDHelper_Transformer: MUON tokE update entered NaN recovery");
							net.storeRunningFlag(false);
							return false;
						}
						Adam::update_param(tt.lmBias, tt.mLmBias, tt.v2LmBias, tt.gLmBias,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                   invBatch, gradScale);
					}
					else if (auroraAdamwBackbone)
					{
						Adam::update_weight(tt.tokE, tt.vTokE, tt.v2TokE, tt.gTokE,
						                    lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                    invBatch, gradScale, wd1, wd2);
						Adam::update_param(tt.lmBias, tt.mLmBias, tt.v2LmBias, tt.gLmBias,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                   invBatch, gradScale);
					}
					else
					{
						// tokE: [vocabSize, dModel]
						if (!atlas::update(tt.atlasTokE, &tt.tokE[0], &tt.gTokE[0],
						              tt.vocabSize, dmTT, invBatch, lr, wd1, wd2, gradScale,
						              ac, net.rngEngine, net.getLogger(), "tr.tokE"))
						{
							net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
								"SGDHelper_Transformer: ATLAS tokE update entered NaN recovery");
							net.storeRunningFlag(false);
							return false;
						}

						// lmBias: simple SGD (no subspace projection for 1D bias)
						if (!atlas::updateBias(&tt.lmBias[0], &tt.gLmBias[0],
						                       static_cast<unsigned int>(tt.lmBias.size()),
						                       invBatch, lr, gradScale))
						{
							net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
								"SGDHelper_Transformer: ATLAS lmBias update produced NaN/Inf");
							net.storeRunningFlag(false);
							return false;
						}
					}
				}
				const float transformerSparrowTrust =
				    ((ac.citadelEnabled || ac.rampartEnabled || ac.meritEnabled || ac.strataEnabled
				      || ac.auroraEnabled || ac.seamEnabled || ac.quasarEnabled) && tt.aster.initialized)
				        ? std::max(0.0f,
				                   std::min(1.0f,
				                            (ac.strataEnabled || ac.auroraEnabled || ac.seamEnabled || ac.quasarEnabled)
				                                ? tt.aster.aegisLastLambdaPredictive
				                                : (ac.meritEnabled
				                                ? tt.aster.meritLastSparrowTrust
				                                : (ac.rampartEnabled
				                                       ? tt.aster.rampartLastSparrowTrust
				                                       : tt.aster.citadelLastSparrowTrust))))
				        : 1.0f;
				const float transformerBodySparrowTrust =
				    ac.auroraEnabled
				        ? std::max(0.0f,
				                   std::min(1.0f, transformerSparrowTrust * ac.auroraBodyTrustScale))
				        : transformerSparrowTrust;
				tt.atlasTokE.externalSparrowTrust = transformerSparrowTrust;
				tt.atlasWIn.externalSparrowTrust = transformerBodySparrowTrust;
				tt.atlasWOut.externalSparrowTrust = transformerBodySparrowTrust;
				for (unsigned int li = 0; li < nLayers; ++li)
				{
					TensorTransformerState::Block& b = tt.blocks[li];
					b.atlasWq.externalSparrowTrust = transformerBodySparrowTrust;
					b.atlasWk.externalSparrowTrust = transformerBodySparrowTrust;
					b.atlasWv.externalSparrowTrust = transformerBodySparrowTrust;
					b.atlasWo.externalSparrowTrust = transformerBodySparrowTrust;
					b.atlasW1.externalSparrowTrust = transformerBodySparrowTrust;
					b.atlasW2.externalSparrowTrust = transformerBodySparrowTrust;
				}

				// Input projection (index 0)
				{
					const float lr = net.skeleton->getLearningRate(0u) * net.lrScheduleMultiplier * extraLRMult;
					const float wd1 = net.skeleton->getWeightDecay1(0u);
					const float wd2 = net.skeleton->getWeightDecay2(0u);
					if (geodeEnabled)
					{
						if (!Geode::update_weight(tt.WIn, tt.vWIn, tt.v2WIn, tt.gWIn,
						                          tt.atlasWIn, dmTT, tt.inputSize, lr,
						                          beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                          invBatch, gradScale, wd1, wd2,
						                          ac, net.rngEngine, net.getLogger(), "tr.WIn"))
						{
							net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
								"SGDHelper_Transformer: GEODE WIn update entered NaN recovery");
							net.storeRunningFlag(false);
							return false;
						}
						Adam::update_param(tt.bIn, tt.mBIn, tt.v2BIn, tt.gBIn,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                   invBatch, gradScale);
					}
					else if (echoEnabled)
					{
						const bool useEchoWIn = echo_scope_uses_input_matrix(ac, dmTT, tt.inputSize);
						if (useEchoWIn)
						{
							if (!atlas::echoUpdate(tt.echoWIn, &tt.WIn[0], &tt.vWIn[0], &tt.v2WIn[0], &tt.gWIn[0],
							                       dmTT, tt.inputSize, lr,
							                       beta1, beta2, inv1mB1t, inv1mB2t, eps,
							                       invBatch, gradScale, wd1, wd2,
							                       tt.optimizerStep,
							                       ac, net.getLogger(), "tr.WIn"))
							{
								net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
									"SGDHelper_Transformer: ECHO WIn update entered NaN recovery");
								net.storeRunningFlag(false);
								return false;
							}
						}
						else
							Adam::update_weight(tt.WIn, tt.vWIn, tt.v2WIn, tt.gWIn,
							                    lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
							                    invBatch, gradScale, wd1, wd2);
						Adam::update_param(tt.bIn, tt.mBIn, tt.v2BIn, tt.gBIn,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                   invBatch, gradScale);
					}
					else if (bimapEnabled)
					{
						const bool useBiMAPWIn = bimap_scope_uses_input_block(ac);
						if (useBiMAPWIn)
						{
							if (!atlas::bimapUpdate(tt.bimapWIn, &tt.WIn[0], &tt.vWIn[0], &tt.v2WIn[0], &tt.gWIn[0],
							                        dmTT, tt.inputSize, lr,
							                        beta1, beta2, inv1mB1t, inv1mB2t, eps,
							                        invBatch, gradScale, wd1, wd2,
							                        ac, net.getLogger(), "tr.WIn"))
							{
								net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
									"SGDHelper_Transformer: BiMAP WIn update entered NaN recovery");
								net.storeRunningFlag(false);
								return false;
							}
						}
						else
							Adam::update_weight(tt.WIn, tt.vWIn, tt.v2WIn, tt.gWIn,
							                    lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
							                    invBatch, gradScale, wd1, wd2);
						Adam::update_param(tt.bIn, tt.mBIn, tt.v2BIn, tt.gBIn,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                   invBatch, gradScale);
					}
					else if (pactEnabled)
					{
						if (!atlas::pactUpdate(tt.pactWIn, &tt.WIn[0], &tt.vWIn[0], &tt.v2WIn[0], &tt.gWIn[0],
						                       dmTT, tt.inputSize, lr,
						                       beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                       invBatch, gradScale, wd1, wd2,
						                       ac, net.getLogger(), "tr.WIn"))
						{
							net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
								"SGDHelper_Transformer: PACT WIn update entered NaN recovery");
							net.storeRunningFlag(false);
							return false;
						}
						Adam::update_param(tt.bIn, tt.mBIn, tt.v2BIn, tt.gBIn,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                   invBatch, gradScale);
					}
					else if (racerEnabled)
					{
						if (!atlas::racerUpdate(tt.racerWIn, &tt.WIn[0], &tt.vWIn[0], &tt.v2WIn[0], &tt.gWIn[0],
						                        dmTT, tt.inputSize, lr,
						                        beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                        invBatch, gradScale, wd1, wd2,
						                        ac, net.getLogger(), "tr.WIn"))
						{
							net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
								"SGDHelper_Transformer: RACER WIn update entered NaN recovery");
							net.storeRunningFlag(false);
							return false;
						}
						Adam::update_param(tt.bIn, tt.mBIn, tt.v2BIn, tt.gBIn,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                   invBatch, gradScale);
					}
					else if (kronEnabled)
					{
						if (!atlas::kronUpdate(tt.kronWIn, &tt.WIn[0], &tt.vWIn[0], &tt.v2WIn[0], &tt.gWIn[0],
						                       dmTT, tt.inputSize, lr,
						                       beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                       invBatch, gradScale, wd1, wd2,
						                       ac, net.getLogger(), "tr.WIn"))
						{
							net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
								"SGDHelper_Transformer: KRON WIn update entered NaN recovery");
							net.storeRunningFlag(false);
							return false;
						}
						Adam::update_param(tt.bIn, tt.mBIn, tt.v2BIn, tt.gBIn,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                   invBatch, gradScale);
					}
					else if (matraEnabled)
					{
						if (!atlas::matraUpdate(tt.matraWIn, &tt.WIn[0], &tt.vWIn[0], &tt.v2WIn[0], &tt.gWIn[0],
						                        dmTT, tt.inputSize, lr,
						                        beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                        invBatch, gradScale, wd1, wd2,
						                        ac, net.getLogger(), "tr.WIn"))
						{
							net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
								"SGDHelper_Transformer: MATRA WIn update entered NaN recovery");
							net.storeRunningFlag(false);
							return false;
						}
						Adam::update_param(tt.bIn, tt.mBIn, tt.v2BIn, tt.gBIn,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                   invBatch, gradScale);
					}
					else if (argosEnabled)
					{
						if (argos_scope_uses_input_block(ac))
						{
							if (!atlas::argosUpdateWithRole(tt.argosWIn, &tt.WIn[0], &tt.vWIn[0], &tt.v2WIn[0], &tt.gWIn[0],
							                                dmTT, tt.inputSize, lr,
							                                beta1, beta2, inv1mB1t, inv1mB2t, eps,
							                                invBatch, gradScale, wd1, wd2,
							                                ac,
							                                glades::atlas::ARGOS_ROLE_NONE,
							                                net.getLogger(), "tr.WIn"))
							{
								net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
									"SGDHelper_Transformer: ARGOS WIn update entered NaN recovery");
								net.storeRunningFlag(false);
								return false;
							}
						}
						else
						{
							Adam::update_weight(tt.WIn, tt.vWIn, tt.v2WIn, tt.gWIn,
							                    lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
							                    invBatch, gradScale, wd1, wd2);
						}
						Adam::update_param(tt.bIn, tt.mBIn, tt.v2BIn, tt.gBIn,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                   invBatch, gradScale);
					}
					else if (muonEnabled)
					{
						if (!atlas::muonUpdate(tt.muonWIn, &tt.WIn[0], &tt.vWIn[0], &tt.v2WIn[0], &tt.gWIn[0],
						                       dmTT, tt.inputSize, lr,
						                       beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                       invBatch, gradScale, wd1, wd2,
						                       ac, net.getLogger(), "tr.WIn"))
						{
							net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
								"SGDHelper_Transformer: MUON WIn update entered NaN recovery");
							net.storeRunningFlag(false);
							return false;
						}
						Adam::update_param(tt.bIn, tt.mBIn, tt.v2BIn, tt.gBIn,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                   invBatch, gradScale);
					}
					else if (auroraAdamwBackbone)
					{
						Adam::update_weight(tt.WIn, tt.vWIn, tt.v2WIn, tt.gWIn,
						                    lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                    invBatch, gradScale, wd1, wd2);
						Adam::update_param(tt.bIn, tt.mBIn, tt.v2BIn, tt.gBIn,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                   invBatch, gradScale);
					}
					else
					{
						// WIn: [dModel, inputSize]
						if (!atlas::update(tt.atlasWIn, &tt.WIn[0], &tt.gWIn[0],
						              dmTT, tt.inputSize, invBatch, lr, wd1, wd2, gradScale,
						              ac, net.rngEngine, net.getLogger(), "tr.WIn"))
						{
							net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
								"SGDHelper_Transformer: ATLAS WIn update entered NaN recovery");
							net.storeRunningFlag(false);
							return false;
						}

						// bIn: simple SGD
						if (!atlas::updateBias(&tt.bIn[0], &tt.gBIn[0],
						                       static_cast<unsigned int>(tt.bIn.size()),
						                       invBatch, lr, gradScale))
						{
							net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
								"SGDHelper_Transformer: ATLAS bIn update produced NaN/Inf");
							net.storeRunningFlag(false);
							return false;
						}
					}
				}

				// Blocks (index 1..nLayers)
				for (unsigned int li = 0; li < nLayers; ++li)
				{
					const unsigned int idx = li + 1u;
					const float lr = net.skeleton->getLearningRate(idx) * net.lrScheduleMultiplier * extraLRMult;
					const float wd1 = net.skeleton->getWeightDecay1(idx);
					const float wd2 = net.skeleton->getWeightDecay2(idx);
					TensorTransformerState::Block& b = tt.blocks[li];
					if (geodeEnabled)
					{
						if (!Geode::update_weight(b.Wq, b.vWq, b.v2Wq, b.gWq,
						                          b.atlasWq, dmTT, dmTT, lr,
						                          beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                          invBatch, gradScale, wd1, wd2,
						                          ac, net.rngEngine, net.getLogger(), "tr.Wq")
						    || !Geode::update_weight(b.Wk, b.vWk, b.v2Wk, b.gWk,
						                             b.atlasWk, dModelKVTT, dmTT, lr,
						                             beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                             invBatch, gradScale, wd1, wd2,
						                             ac, net.rngEngine, net.getLogger(), "tr.Wk")
						    || !Geode::update_weight(b.Wv, b.vWv, b.v2Wv, b.gWv,
						                             b.atlasWv, dModelKVTT, dmTT, lr,
						                             beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                             invBatch, gradScale, wd1, wd2,
						                             ac, net.rngEngine, net.getLogger(), "tr.Wv")
						    || !Geode::update_weight(b.Wo, b.vWo, b.v2Wo, b.gWo,
						                             b.atlasWo, dmTT, dmTT, lr,
						                             beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                             invBatch, gradScale, wd1, wd2,
						                             ac, net.rngEngine, net.getLogger(), "tr.Wo")
						    || !Geode::update_weight(b.W1, b.vW1, b.v2W1, b.gW1,
						                             b.atlasW1, ff1WidthTT, dmTT, lr,
						                             beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                             invBatch, gradScale, wd1, wd2,
						                             ac, net.rngEngine, net.getLogger(), "tr.W1")
						    || !Geode::update_weight(b.W2, b.vW2, b.v2W2, b.gW2,
						                             b.atlasW2, dmTT, dFFTT, lr,
						                             beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                             invBatch, gradScale, wd1, wd2,
						                             ac, net.rngEngine, net.getLogger(), "tr.W2"))
						{
							net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
								"SGDHelper_Transformer: GEODE block weight update entered NaN recovery");
							net.storeRunningFlag(false);
							return false;
						}
						Adam::update_param(b.bq, b.mBq, b.v2Bq, b.gBq, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.bk, b.mBk, b.v2Bk, b.gBk, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.bv, b.mBv, b.v2Bv, b.gBv, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.bo, b.mBo, b.v2Bo, b.gBo, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.b1, b.mB1, b.v2B1, b.gB1, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.b2, b.mB2, b.v2B2, b.gB2, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln1Gamma, b.mLn1Gamma, b.v2Ln1Gamma, b.gLn1Gamma, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln1Beta, b.mLn1Beta, b.v2Ln1Beta, b.gLn1Beta, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln2Gamma, b.mLn2Gamma, b.v2Ln2Gamma, b.gLn2Gamma, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln2Beta, b.mLn2Beta, b.v2Ln2Beta, b.gLn2Beta, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					}
					else if (echoEnabled)
					{
						if ((echo_scope_uses_decoder_matrix(ac, li, nLayers, dmTT, dmTT)
						      && !atlas::echoUpdate(b.echoWq, &b.Wq[0], &b.vWq[0], &b.v2Wq[0], &b.gWq[0],
						                            dmTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                            invBatch, gradScale, wd1, wd2, tt.optimizerStep, ac, net.getLogger(), "tr.Wq"))
						    || (echo_scope_uses_decoder_matrix(ac, li, nLayers, dModelKVTT, dmTT)
						        && !atlas::echoUpdate(b.echoWk, &b.Wk[0], &b.vWk[0], &b.v2Wk[0], &b.gWk[0],
						                              dModelKVTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                              invBatch, gradScale, wd1, wd2, tt.optimizerStep, ac, net.getLogger(), "tr.Wk"))
						    || (echo_scope_uses_decoder_matrix(ac, li, nLayers, dModelKVTT, dmTT)
						        && !atlas::echoUpdate(b.echoWv, &b.Wv[0], &b.vWv[0], &b.v2Wv[0], &b.gWv[0],
						                              dModelKVTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                              invBatch, gradScale, wd1, wd2, tt.optimizerStep, ac, net.getLogger(), "tr.Wv"))
						    || (echo_scope_uses_decoder_matrix(ac, li, nLayers, dmTT, dmTT)
						        && !atlas::echoUpdate(b.echoWo, &b.Wo[0], &b.vWo[0], &b.v2Wo[0], &b.gWo[0],
						                              dmTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                              invBatch, gradScale, wd1, wd2, tt.optimizerStep, ac, net.getLogger(), "tr.Wo"))
						    || (echo_scope_uses_decoder_matrix(ac, li, nLayers, ff1WidthTT, dmTT)
						        && !atlas::echoUpdate(b.echoW1, &b.W1[0], &b.vW1[0], &b.v2W1[0], &b.gW1[0],
						                              ff1WidthTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                              invBatch, gradScale, wd1, wd2, tt.optimizerStep, ac, net.getLogger(), "tr.W1"))
						    || (echo_scope_uses_decoder_matrix(ac, li, nLayers, dmTT, dFFTT)
						        && !atlas::echoUpdate(b.echoW2, &b.W2[0], &b.vW2[0], &b.v2W2[0], &b.gW2[0],
						                              dmTT, dFFTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                              invBatch, gradScale, wd1, wd2, tt.optimizerStep, ac, net.getLogger(), "tr.W2")))
						{
							net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
								"SGDHelper_Transformer: ECHO block weight update entered NaN recovery");
							net.storeRunningFlag(false);
							return false;
						}
						if (!echo_scope_uses_decoder_matrix(ac, li, nLayers, dmTT, dmTT))
							Adam::update_weight(b.Wq, b.vWq, b.v2Wq, b.gWq, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
						if (!echo_scope_uses_decoder_matrix(ac, li, nLayers, dModelKVTT, dmTT))
							Adam::update_weight(b.Wk, b.vWk, b.v2Wk, b.gWk, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
						if (!echo_scope_uses_decoder_matrix(ac, li, nLayers, dModelKVTT, dmTT))
							Adam::update_weight(b.Wv, b.vWv, b.v2Wv, b.gWv, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
						if (!echo_scope_uses_decoder_matrix(ac, li, nLayers, dmTT, dmTT))
							Adam::update_weight(b.Wo, b.vWo, b.v2Wo, b.gWo, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
						if (!echo_scope_uses_decoder_matrix(ac, li, nLayers, ff1WidthTT, dmTT))
							Adam::update_weight(b.W1, b.vW1, b.v2W1, b.gW1, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
						if (!echo_scope_uses_decoder_matrix(ac, li, nLayers, dmTT, dFFTT))
							Adam::update_weight(b.W2, b.vW2, b.v2W2, b.gW2, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
						Adam::update_param(b.bq, b.mBq, b.v2Bq, b.gBq, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.bk, b.mBk, b.v2Bk, b.gBk, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.bv, b.mBv, b.v2Bv, b.gBv, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.bo, b.mBo, b.v2Bo, b.gBo, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.b1, b.mB1, b.v2B1, b.gB1, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.b2, b.mB2, b.v2B2, b.gB2, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln1Gamma, b.mLn1Gamma, b.v2Ln1Gamma, b.gLn1Gamma, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln1Beta, b.mLn1Beta, b.v2Ln1Beta, b.gLn1Beta, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln2Gamma, b.mLn2Gamma, b.v2Ln2Gamma, b.gLn2Gamma, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln2Beta, b.mLn2Beta, b.v2Ln2Beta, b.gLn2Beta, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					}
					else if (bimapEnabled)
					{
						const bool useBiMAPBlock = bimap_scope_uses_decoder_block(ac, li, nLayers);
						if (useBiMAPBlock)
						{
							if (!atlas::bimapUpdate(b.bimapWq, &b.Wq[0], &b.vWq[0], &b.v2Wq[0], &b.gWq[0],
							                        dmTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
							                        invBatch, gradScale, wd1, wd2, ac, net.getLogger(), "tr.Wq")
							    || !atlas::bimapUpdate(b.bimapWk, &b.Wk[0], &b.vWk[0], &b.v2Wk[0], &b.gWk[0],
							                           dModelKVTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
							                           invBatch, gradScale, wd1, wd2, ac, net.getLogger(), "tr.Wk")
							    || !atlas::bimapUpdate(b.bimapWv, &b.Wv[0], &b.vWv[0], &b.v2Wv[0], &b.gWv[0],
							                           dModelKVTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
							                           invBatch, gradScale, wd1, wd2, ac, net.getLogger(), "tr.Wv")
							    || !atlas::bimapUpdate(b.bimapWo, &b.Wo[0], &b.vWo[0], &b.v2Wo[0], &b.gWo[0],
							                           dmTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
							                           invBatch, gradScale, wd1, wd2, ac, net.getLogger(), "tr.Wo")
							    || !atlas::bimapUpdate(b.bimapW1, &b.W1[0], &b.vW1[0], &b.v2W1[0], &b.gW1[0],
							                           ff1WidthTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
							                           invBatch, gradScale, wd1, wd2, ac, net.getLogger(), "tr.W1")
							    || !atlas::bimapUpdate(b.bimapW2, &b.W2[0], &b.vW2[0], &b.v2W2[0], &b.gW2[0],
							                           dmTT, dFFTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
							                           invBatch, gradScale, wd1, wd2, ac, net.getLogger(), "tr.W2"))
							{
								net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
									"SGDHelper_Transformer: BiMAP block weight update entered NaN recovery");
								net.storeRunningFlag(false);
								return false;
							}
						}
						else
						{
							Adam::update_weight(b.Wq, b.vWq, b.v2Wq, b.gWq, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
							Adam::update_weight(b.Wk, b.vWk, b.v2Wk, b.gWk, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
							Adam::update_weight(b.Wv, b.vWv, b.v2Wv, b.gWv, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
							Adam::update_weight(b.Wo, b.vWo, b.v2Wo, b.gWo, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
							Adam::update_weight(b.W1, b.vW1, b.v2W1, b.gW1, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
							Adam::update_weight(b.W2, b.vW2, b.v2W2, b.gW2, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
						}
						Adam::update_param(b.bq, b.mBq, b.v2Bq, b.gBq, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.bk, b.mBk, b.v2Bk, b.gBk, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.bv, b.mBv, b.v2Bv, b.gBv, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.bo, b.mBo, b.v2Bo, b.gBo, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.b1, b.mB1, b.v2B1, b.gB1, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.b2, b.mB2, b.v2B2, b.gB2, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln1Gamma, b.mLn1Gamma, b.v2Ln1Gamma, b.gLn1Gamma, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln1Beta, b.mLn1Beta, b.v2Ln1Beta, b.gLn1Beta, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln2Gamma, b.mLn2Gamma, b.v2Ln2Gamma, b.gLn2Gamma, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln2Beta, b.mLn2Beta, b.v2Ln2Beta, b.gLn2Beta, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					}
					else if (pactEnabled)
					{
						if (!atlas::pactUpdate(b.pactWq, &b.Wq[0], &b.vWq[0], &b.v2Wq[0], &b.gWq[0],
						                       dmTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                       invBatch, gradScale, wd1, wd2, ac, net.getLogger(), "tr.Wq")
						    || !atlas::pactUpdate(b.pactWk, &b.Wk[0], &b.vWk[0], &b.v2Wk[0], &b.gWk[0],
						                          dModelKVTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                          invBatch, gradScale, wd1, wd2, ac, net.getLogger(), "tr.Wk")
						    || !atlas::pactUpdate(b.pactWv, &b.Wv[0], &b.vWv[0], &b.v2Wv[0], &b.gWv[0],
						                          dModelKVTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                          invBatch, gradScale, wd1, wd2, ac, net.getLogger(), "tr.Wv")
						    || !atlas::pactUpdate(b.pactWo, &b.Wo[0], &b.vWo[0], &b.v2Wo[0], &b.gWo[0],
						                          dmTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                          invBatch, gradScale, wd1, wd2, ac, net.getLogger(), "tr.Wo")
						    || !atlas::pactUpdate(b.pactW1, &b.W1[0], &b.vW1[0], &b.v2W1[0], &b.gW1[0],
						                          ff1WidthTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                          invBatch, gradScale, wd1, wd2, ac, net.getLogger(), "tr.W1")
						    || !atlas::pactUpdate(b.pactW2, &b.W2[0], &b.vW2[0], &b.v2W2[0], &b.gW2[0],
						                          dmTT, dFFTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                          invBatch, gradScale, wd1, wd2, ac, net.getLogger(), "tr.W2"))
						{
							net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
								"SGDHelper_Transformer: PACT block weight update entered NaN recovery");
							net.storeRunningFlag(false);
							return false;
						}
						Adam::update_param(b.bq, b.mBq, b.v2Bq, b.gBq, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.bk, b.mBk, b.v2Bk, b.gBk, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.bv, b.mBv, b.v2Bv, b.gBv, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.bo, b.mBo, b.v2Bo, b.gBo, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.b1, b.mB1, b.v2B1, b.gB1, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.b2, b.mB2, b.v2B2, b.gB2, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln1Gamma, b.mLn1Gamma, b.v2Ln1Gamma, b.gLn1Gamma, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln1Beta, b.mLn1Beta, b.v2Ln1Beta, b.gLn1Beta, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln2Gamma, b.mLn2Gamma, b.v2Ln2Gamma, b.gLn2Gamma, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln2Beta, b.mLn2Beta, b.v2Ln2Beta, b.gLn2Beta, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					}
					else if (racerEnabled)
					{
						if (!atlas::racerUpdate(b.racerWq, &b.Wq[0], &b.vWq[0], &b.v2Wq[0], &b.gWq[0],
						                        dmTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                        invBatch, gradScale, wd1, wd2, ac, net.getLogger(), "tr.Wq")
						    || !atlas::racerUpdate(b.racerWk, &b.Wk[0], &b.vWk[0], &b.v2Wk[0], &b.gWk[0],
						                           dModelKVTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                           invBatch, gradScale, wd1, wd2, ac, net.getLogger(), "tr.Wk")
						    || !atlas::racerUpdate(b.racerWv, &b.Wv[0], &b.vWv[0], &b.v2Wv[0], &b.gWv[0],
						                           dModelKVTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                           invBatch, gradScale, wd1, wd2, ac, net.getLogger(), "tr.Wv")
						    || !atlas::racerUpdate(b.racerWo, &b.Wo[0], &b.vWo[0], &b.v2Wo[0], &b.gWo[0],
						                           dmTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                           invBatch, gradScale, wd1, wd2, ac, net.getLogger(), "tr.Wo")
						    || !atlas::racerUpdate(b.racerW1, &b.W1[0], &b.vW1[0], &b.v2W1[0], &b.gW1[0],
						                           ff1WidthTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                           invBatch, gradScale, wd1, wd2, ac, net.getLogger(), "tr.W1")
						    || !atlas::racerUpdate(b.racerW2, &b.W2[0], &b.vW2[0], &b.v2W2[0], &b.gW2[0],
						                           dmTT, dFFTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                           invBatch, gradScale, wd1, wd2, ac, net.getLogger(), "tr.W2"))
						{
							net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
								"SGDHelper_Transformer: RACER block weight update entered NaN recovery");
							net.storeRunningFlag(false);
							return false;
						}
						Adam::update_param(b.bq, b.mBq, b.v2Bq, b.gBq, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.bk, b.mBk, b.v2Bk, b.gBk, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.bv, b.mBv, b.v2Bv, b.gBv, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.bo, b.mBo, b.v2Bo, b.gBo, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.b1, b.mB1, b.v2B1, b.gB1, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.b2, b.mB2, b.v2B2, b.gB2, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln1Gamma, b.mLn1Gamma, b.v2Ln1Gamma, b.gLn1Gamma, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln1Beta, b.mLn1Beta, b.v2Ln1Beta, b.gLn1Beta, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln2Gamma, b.mLn2Gamma, b.v2Ln2Gamma, b.gLn2Gamma, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln2Beta, b.mLn2Beta, b.v2Ln2Beta, b.gLn2Beta, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					}
					else if (kronEnabled)
					{
						if (!atlas::kronUpdate(b.kronWq, &b.Wq[0], &b.vWq[0], &b.v2Wq[0], &b.gWq[0],
						                       dmTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                       invBatch, gradScale, wd1, wd2, ac, net.getLogger(), "tr.Wq")
						    || !atlas::kronUpdate(b.kronWk, &b.Wk[0], &b.vWk[0], &b.v2Wk[0], &b.gWk[0],
						                          dModelKVTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                          invBatch, gradScale, wd1, wd2, ac, net.getLogger(), "tr.Wk")
						    || !atlas::kronUpdate(b.kronWv, &b.Wv[0], &b.vWv[0], &b.v2Wv[0], &b.gWv[0],
						                          dModelKVTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                          invBatch, gradScale, wd1, wd2, ac, net.getLogger(), "tr.Wv")
						    || !atlas::kronUpdate(b.kronWo, &b.Wo[0], &b.vWo[0], &b.v2Wo[0], &b.gWo[0],
						                          dmTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                          invBatch, gradScale, wd1, wd2, ac, net.getLogger(), "tr.Wo")
						    || !atlas::kronUpdate(b.kronW1, &b.W1[0], &b.vW1[0], &b.v2W1[0], &b.gW1[0],
						                          ff1WidthTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                          invBatch, gradScale, wd1, wd2, ac, net.getLogger(), "tr.W1")
						    || !atlas::kronUpdate(b.kronW2, &b.W2[0], &b.vW2[0], &b.v2W2[0], &b.gW2[0],
						                          dmTT, dFFTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                          invBatch, gradScale, wd1, wd2, ac, net.getLogger(), "tr.W2"))
						{
							net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
								"SGDHelper_Transformer: KRON block weight update entered NaN recovery");
							net.storeRunningFlag(false);
							return false;
						}
						Adam::update_param(b.bq, b.mBq, b.v2Bq, b.gBq, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.bk, b.mBk, b.v2Bk, b.gBk, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.bv, b.mBv, b.v2Bv, b.gBv, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.bo, b.mBo, b.v2Bo, b.gBo, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.b1, b.mB1, b.v2B1, b.gB1, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.b2, b.mB2, b.v2B2, b.gB2, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln1Gamma, b.mLn1Gamma, b.v2Ln1Gamma, b.gLn1Gamma, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln1Beta, b.mLn1Beta, b.v2Ln1Beta, b.gLn1Beta, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln2Gamma, b.mLn2Gamma, b.v2Ln2Gamma, b.gLn2Gamma, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln2Beta, b.mLn2Beta, b.v2Ln2Beta, b.gLn2Beta, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					}
					else if (matraEnabled)
					{
						if (!atlas::matraUpdate(b.matraWq, &b.Wq[0], &b.vWq[0], &b.v2Wq[0], &b.gWq[0],
						                        dmTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                        invBatch, gradScale, wd1, wd2, ac, net.getLogger(), "tr.Wq")
						    || !atlas::matraUpdate(b.matraWk, &b.Wk[0], &b.vWk[0], &b.v2Wk[0], &b.gWk[0],
						                           dModelKVTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                           invBatch, gradScale, wd1, wd2, ac, net.getLogger(), "tr.Wk")
						    || !atlas::matraUpdate(b.matraWv, &b.Wv[0], &b.vWv[0], &b.v2Wv[0], &b.gWv[0],
						                           dModelKVTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                           invBatch, gradScale, wd1, wd2, ac, net.getLogger(), "tr.Wv")
						    || !atlas::matraUpdate(b.matraWo, &b.Wo[0], &b.vWo[0], &b.v2Wo[0], &b.gWo[0],
						                           dmTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                           invBatch, gradScale, wd1, wd2, ac, net.getLogger(), "tr.Wo")
						    || !atlas::matraUpdate(b.matraW1, &b.W1[0], &b.vW1[0], &b.v2W1[0], &b.gW1[0],
						                           ff1WidthTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                           invBatch, gradScale, wd1, wd2, ac, net.getLogger(), "tr.W1")
						    || !atlas::matraUpdate(b.matraW2, &b.W2[0], &b.vW2[0], &b.v2W2[0], &b.gW2[0],
						                           dmTT, dFFTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                           invBatch, gradScale, wd1, wd2, ac, net.getLogger(), "tr.W2"))
						{
							net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
								"SGDHelper_Transformer: MATRA block weight update entered NaN recovery");
							net.storeRunningFlag(false);
							return false;
						}
						Adam::update_param(b.bq, b.mBq, b.v2Bq, b.gBq, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.bk, b.mBk, b.v2Bk, b.gBk, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.bv, b.mBv, b.v2Bv, b.gBv, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.bo, b.mBo, b.v2Bo, b.gBo, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.b1, b.mB1, b.v2B1, b.gB1, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.b2, b.mB2, b.v2B2, b.gB2, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln1Gamma, b.mLn1Gamma, b.v2Ln1Gamma, b.gLn1Gamma, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln1Beta, b.mLn1Beta, b.v2Ln1Beta, b.gLn1Beta, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln2Gamma, b.mLn2Gamma, b.v2Ln2Gamma, b.gLn2Gamma, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln2Beta, b.mLn2Beta, b.v2Ln2Beta, b.gLn2Beta, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					}
					else if (argosEnabled)
					{
						const unsigned int argosRoleFlags = argos_role_flags_for_block(li, nLayers);
						const bool useArgosBlock = argos_scope_uses_decoder_block(ac, li, nLayers);
						if ((useArgosBlock
						     && (!atlas::argosUpdateWithRole(b.argosWq, &b.Wq[0], &b.vWq[0], &b.v2Wq[0], &b.gWq[0],
						                                   dmTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                                   invBatch, gradScale, wd1, wd2, ac, argosRoleFlags, net.getLogger(), "tr.Wq")
						         || !atlas::argosUpdateWithRole(b.argosWk, &b.Wk[0], &b.vWk[0], &b.v2Wk[0], &b.gWk[0],
						                                        dModelKVTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                                        invBatch, gradScale, wd1, wd2, ac, argosRoleFlags, net.getLogger(), "tr.Wk")
						         || !atlas::argosUpdateWithRole(b.argosWv, &b.Wv[0], &b.vWv[0], &b.v2Wv[0], &b.gWv[0],
						                                        dModelKVTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                                        invBatch, gradScale, wd1, wd2, ac, argosRoleFlags, net.getLogger(), "tr.Wv")
						         || !atlas::argosUpdateWithRole(b.argosWo, &b.Wo[0], &b.vWo[0], &b.v2Wo[0], &b.gWo[0],
						                                        dmTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                                        invBatch, gradScale, wd1, wd2, ac, argosRoleFlags, net.getLogger(), "tr.Wo")
						         || !atlas::argosUpdateWithRole(b.argosW1, &b.W1[0], &b.vW1[0], &b.v2W1[0], &b.gW1[0],
						                                        ff1WidthTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                                        invBatch, gradScale, wd1, wd2, ac, argosRoleFlags, net.getLogger(), "tr.W1")
						         || !atlas::argosUpdateWithRole(b.argosW2, &b.W2[0], &b.vW2[0], &b.v2W2[0], &b.gW2[0],
						                                        dmTT, dFFTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                                        invBatch, gradScale, wd1, wd2, ac, argosRoleFlags, net.getLogger(), "tr.W2"))))
						{
							net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
								"SGDHelper_Transformer: ARGOS block weight update entered NaN recovery");
							net.storeRunningFlag(false);
							return false;
						}
						if (!useArgosBlock)
						{
							Adam::update_weight(b.Wq, b.vWq, b.v2Wq, b.gWq, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
							Adam::update_weight(b.Wk, b.vWk, b.v2Wk, b.gWk, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
							Adam::update_weight(b.Wv, b.vWv, b.v2Wv, b.gWv, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
							Adam::update_weight(b.Wo, b.vWo, b.v2Wo, b.gWo, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
							Adam::update_weight(b.W1, b.vW1, b.v2W1, b.gW1, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
							Adam::update_weight(b.W2, b.vW2, b.v2W2, b.gW2, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
						}
						Adam::update_param(b.bq, b.mBq, b.v2Bq, b.gBq, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.bk, b.mBk, b.v2Bk, b.gBk, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.bv, b.mBv, b.v2Bv, b.gBv, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.bo, b.mBo, b.v2Bo, b.gBo, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.b1, b.mB1, b.v2B1, b.gB1, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.b2, b.mB2, b.v2B2, b.gB2, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln1Gamma, b.mLn1Gamma, b.v2Ln1Gamma, b.gLn1Gamma, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln1Beta, b.mLn1Beta, b.v2Ln1Beta, b.gLn1Beta, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln2Gamma, b.mLn2Gamma, b.v2Ln2Gamma, b.gLn2Gamma, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln2Beta, b.mLn2Beta, b.v2Ln2Beta, b.gLn2Beta, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					}
					else if (muonEnabled)
					{
						if (!atlas::muonUpdate(b.muonWq, &b.Wq[0], &b.vWq[0], &b.v2Wq[0], &b.gWq[0],
						                       dmTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                       invBatch, gradScale, wd1, wd2, ac, net.getLogger(), "tr.Wq")
						    || !atlas::muonUpdate(b.muonWk, &b.Wk[0], &b.vWk[0], &b.v2Wk[0], &b.gWk[0],
						                          dModelKVTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                          invBatch, gradScale, wd1, wd2, ac, net.getLogger(), "tr.Wk")
						    || !atlas::muonUpdate(b.muonWv, &b.Wv[0], &b.vWv[0], &b.v2Wv[0], &b.gWv[0],
						                          dModelKVTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                          invBatch, gradScale, wd1, wd2, ac, net.getLogger(), "tr.Wv")
						    || !atlas::muonUpdate(b.muonWo, &b.Wo[0], &b.vWo[0], &b.v2Wo[0], &b.gWo[0],
						                          dmTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                          invBatch, gradScale, wd1, wd2, ac, net.getLogger(), "tr.Wo")
						    || !atlas::muonUpdate(b.muonW1, &b.W1[0], &b.vW1[0], &b.v2W1[0], &b.gW1[0],
						                          ff1WidthTT, dmTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                          invBatch, gradScale, wd1, wd2, ac, net.getLogger(), "tr.W1")
						    || !atlas::muonUpdate(b.muonW2, &b.W2[0], &b.vW2[0], &b.v2W2[0], &b.gW2[0],
						                          dmTT, dFFTT, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                          invBatch, gradScale, wd1, wd2, ac, net.getLogger(), "tr.W2"))
						{
							net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
								"SGDHelper_Transformer: MUON block weight update entered NaN recovery");
							net.storeRunningFlag(false);
							return false;
						}
						Adam::update_param(b.bq, b.mBq, b.v2Bq, b.gBq, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.bk, b.mBk, b.v2Bk, b.gBk, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.bv, b.mBv, b.v2Bv, b.gBv, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.bo, b.mBo, b.v2Bo, b.gBo, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.b1, b.mB1, b.v2B1, b.gB1, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.b2, b.mB2, b.v2B2, b.gB2, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln1Gamma, b.mLn1Gamma, b.v2Ln1Gamma, b.gLn1Gamma, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln1Beta, b.mLn1Beta, b.v2Ln1Beta, b.gLn1Beta, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln2Gamma, b.mLn2Gamma, b.v2Ln2Gamma, b.gLn2Gamma, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln2Beta, b.mLn2Beta, b.v2Ln2Beta, b.gLn2Beta, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					}
					else if (auroraAdamwBackbone)
					{
						Adam::update_weight(b.Wq, b.vWq, b.v2Wq, b.gWq, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
						Adam::update_weight(b.Wk, b.vWk, b.v2Wk, b.gWk, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
						Adam::update_weight(b.Wv, b.vWv, b.v2Wv, b.gWv, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
						Adam::update_weight(b.Wo, b.vWo, b.v2Wo, b.gWo, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
						Adam::update_weight(b.W1, b.vW1, b.v2W1, b.gW1, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
						Adam::update_weight(b.W2, b.vW2, b.v2W2, b.gW2, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
						Adam::update_param(b.bq, b.mBq, b.v2Bq, b.gBq, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.bk, b.mBk, b.v2Bk, b.gBk, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.bv, b.mBv, b.v2Bv, b.gBv, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.bo, b.mBo, b.v2Bo, b.gBo, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.b1, b.mB1, b.v2B1, b.gB1, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.b2, b.mB2, b.v2B2, b.gB2, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln1Gamma, b.mLn1Gamma, b.v2Ln1Gamma, b.gLn1Gamma, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln1Beta, b.mLn1Beta, b.v2Ln1Beta, b.gLn1Beta, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln2Gamma, b.mLn2Gamma, b.v2Ln2Gamma, b.gLn2Gamma, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(b.ln2Beta, b.mLn2Beta, b.v2Ln2Beta, b.gLn2Beta, lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					}
					else
					{
						// Wq: [dModel, dModel]
						if (!atlas::update(b.atlasWq, &b.Wq[0], &b.gWq[0], dmTT, dmTT,
						              invBatch, lr, wd1, wd2, gradScale, ac, net.rngEngine, net.getLogger(), "tr.Wq")
						// Wk: [dModelKV, dModel]
						    || !atlas::update(b.atlasWk, &b.Wk[0], &b.gWk[0], dModelKVTT, dmTT,
						              invBatch, lr, wd1, wd2, gradScale, ac, net.rngEngine, net.getLogger(), "tr.Wk")
						// Wv: [dModelKV, dModel]
						    || !atlas::update(b.atlasWv, &b.Wv[0], &b.gWv[0], dModelKVTT, dmTT,
						              invBatch, lr, wd1, wd2, gradScale, ac, net.rngEngine, net.getLogger(), "tr.Wv")
						// Wo: [dModel, dModel]
						    || !atlas::update(b.atlasWo, &b.Wo[0], &b.gWo[0], dmTT, dmTT,
						              invBatch, lr, wd1, wd2, gradScale, ac, net.rngEngine, net.getLogger(), "tr.Wo")
						// W1: [ff1Width, dModel]
						    || !atlas::update(b.atlasW1, &b.W1[0], &b.gW1[0], ff1WidthTT, dmTT,
						              invBatch, lr, wd1, wd2, gradScale, ac, net.rngEngine, net.getLogger(), "tr.W1")
						// W2: [dModel, dFF]
						    || !atlas::update(b.atlasW2, &b.W2[0], &b.gW2[0], dmTT, dFFTT,
						              invBatch, lr, wd1, wd2, gradScale, ac, net.rngEngine, net.getLogger(), "tr.W2"))
						{
							net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
								"SGDHelper_Transformer: ATLAS block weight update entered NaN recovery");
							net.storeRunningFlag(false);
							return false;
						}

						// Biases and LN params: simple SGD (no subspace projection)
						if (!atlas::updateBias(&b.bq[0], &b.gBq[0], static_cast<unsigned int>(b.bq.size()), invBatch, lr, gradScale)
						    || !atlas::updateBias(&b.bk[0], &b.gBk[0], static_cast<unsigned int>(b.bk.size()), invBatch, lr, gradScale)
						    || !atlas::updateBias(&b.bv[0], &b.gBv[0], static_cast<unsigned int>(b.bv.size()), invBatch, lr, gradScale)
						    || !atlas::updateBias(&b.bo[0], &b.gBo[0], static_cast<unsigned int>(b.bo.size()), invBatch, lr, gradScale)
						    || !atlas::updateBias(&b.b1[0], &b.gB1[0], static_cast<unsigned int>(b.b1.size()), invBatch, lr, gradScale)
						    || !atlas::updateBias(&b.b2[0], &b.gB2[0], static_cast<unsigned int>(b.b2.size()), invBatch, lr, gradScale)
						    || !atlas::updateBias(&b.ln1Gamma[0], &b.gLn1Gamma[0], static_cast<unsigned int>(b.ln1Gamma.size()), invBatch, lr, gradScale)
						    || !atlas::updateBias(&b.ln1Beta[0], &b.gLn1Beta[0], static_cast<unsigned int>(b.ln1Beta.size()), invBatch, lr, gradScale)
						    || !atlas::updateBias(&b.ln2Gamma[0], &b.gLn2Gamma[0], static_cast<unsigned int>(b.ln2Gamma.size()), invBatch, lr, gradScale)
						    || !atlas::updateBias(&b.ln2Beta[0], &b.gLn2Beta[0], static_cast<unsigned int>(b.ln2Beta.size()), invBatch, lr, gradScale))
						{
							net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
								"SGDHelper_Transformer: ATLAS block bias/LN update produced NaN/Inf");
							net.storeRunningFlag(false);
							return false;
						}
					}
				}

				// Final LayerNorm (SGD, use block 0 LR; no weight decay)
				{
					const float lr = net.skeleton->getLearningRate(1u) * net.lrScheduleMultiplier * extraLRMult;
					if (geodeEnabled)
					{
						Adam::update_param(tt.lnFinalGamma, tt.mLnFinalGamma, tt.v2LnFinalGamma, tt.gLnFinalGamma,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(tt.lnFinalBeta, tt.mLnFinalBeta, tt.v2LnFinalBeta, tt.gLnFinalBeta,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					}
					else if (echoEnabled)
					{
						Adam::update_param(tt.lnFinalGamma, tt.mLnFinalGamma, tt.v2LnFinalGamma, tt.gLnFinalGamma,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(tt.lnFinalBeta, tt.mLnFinalBeta, tt.v2LnFinalBeta, tt.gLnFinalBeta,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					}
					else if (bimapEnabled)
					{
						Adam::update_param(tt.lnFinalGamma, tt.mLnFinalGamma, tt.v2LnFinalGamma, tt.gLnFinalGamma,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(tt.lnFinalBeta, tt.mLnFinalBeta, tt.v2LnFinalBeta, tt.gLnFinalBeta,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					}
					else if (pactEnabled)
					{
						Adam::update_param(tt.lnFinalGamma, tt.mLnFinalGamma, tt.v2LnFinalGamma, tt.gLnFinalGamma,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(tt.lnFinalBeta, tt.mLnFinalBeta, tt.v2LnFinalBeta, tt.gLnFinalBeta,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					}
					else if (racerEnabled)
					{
						Adam::update_param(tt.lnFinalGamma, tt.mLnFinalGamma, tt.v2LnFinalGamma, tt.gLnFinalGamma,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(tt.lnFinalBeta, tt.mLnFinalBeta, tt.v2LnFinalBeta, tt.gLnFinalBeta,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					}
					else if (kronEnabled)
					{
						Adam::update_param(tt.lnFinalGamma, tt.mLnFinalGamma, tt.v2LnFinalGamma, tt.gLnFinalGamma,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(tt.lnFinalBeta, tt.mLnFinalBeta, tt.v2LnFinalBeta, tt.gLnFinalBeta,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					}
					else if (matraEnabled)
					{
						Adam::update_param(tt.lnFinalGamma, tt.mLnFinalGamma, tt.v2LnFinalGamma, tt.gLnFinalGamma,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(tt.lnFinalBeta, tt.mLnFinalBeta, tt.v2LnFinalBeta, tt.gLnFinalBeta,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					}
					else if (argosEnabled)
					{
						Adam::update_param(tt.lnFinalGamma, tt.mLnFinalGamma, tt.v2LnFinalGamma, tt.gLnFinalGamma,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(tt.lnFinalBeta, tt.mLnFinalBeta, tt.v2LnFinalBeta, tt.gLnFinalBeta,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					}
					else if (muonEnabled)
					{
						Adam::update_param(tt.lnFinalGamma, tt.mLnFinalGamma, tt.v2LnFinalGamma, tt.gLnFinalGamma,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(tt.lnFinalBeta, tt.mLnFinalBeta, tt.v2LnFinalBeta, tt.gLnFinalBeta,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					}
					else if (auroraAdamwBackbone)
					{
						Adam::update_param(tt.lnFinalGamma, tt.mLnFinalGamma, tt.v2LnFinalGamma, tt.gLnFinalGamma,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
						Adam::update_param(tt.lnFinalBeta, tt.mLnFinalBeta, tt.v2LnFinalBeta, tt.gLnFinalBeta,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					}
					else if (!atlas::updateBias(&tt.lnFinalGamma[0], &tt.gLnFinalGamma[0],
					                            static_cast<unsigned int>(tt.lnFinalGamma.size()),
					                            invBatch, lr, gradScale)
					         || !atlas::updateBias(&tt.lnFinalBeta[0], &tt.gLnFinalBeta[0],
					                               static_cast<unsigned int>(tt.lnFinalBeta.size()),
					                               invBatch, lr, gradScale))
					{
						net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
							"SGDHelper_Transformer: ATLAS final LN update produced NaN/Inf");
						net.storeRunningFlag(false);
						return false;
					}
				}

				// Output projection (index nLayers) is unused in token LM tied-head mode.
				if (!tokenLMTiedHead)
				{
					const unsigned int idx = nLayers;
					const float lr = net.skeleton->getLearningRate(idx) * net.lrScheduleMultiplier * extraLRMult;
					const float wd1 = net.skeleton->getWeightDecay1(idx);
					const float wd2 = net.skeleton->getWeightDecay2(idx);
					if (geodeEnabled)
					{
						if (!Geode::update_weight(tt.WOut, tt.vWOut, tt.v2WOut, tt.gWOut,
						                          tt.atlasWOut, outSize, dmTT, lr,
						                          beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                          invBatch, gradScale, wd1, wd2,
						                          ac, net.rngEngine, net.getLogger(), "tr.WOut"))
						{
							net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
								"SGDHelper_Transformer: GEODE WOut update entered NaN recovery");
							net.storeRunningFlag(false);
							return false;
						}
						Adam::update_param(tt.bOut, tt.mBOut, tt.v2BOut, tt.gBOut,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                   invBatch, gradScale);
					}
					else if (echoEnabled)
					{
						const bool useEchoWOut = echo_scope_uses_head_matrix(ac, outSize, dmTT);
						if (useEchoWOut)
						{
							if (!atlas::echoUpdate(tt.echoWOut, &tt.WOut[0], &tt.vWOut[0], &tt.v2WOut[0], &tt.gWOut[0],
							                       outSize, dmTT, lr,
							                       beta1, beta2, inv1mB1t, inv1mB2t, eps,
							                       invBatch, gradScale, wd1, wd2,
							                       tt.optimizerStep,
							                       ac, net.getLogger(), "tr.WOut"))
							{
								net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
									"SGDHelper_Transformer: ECHO WOut update entered NaN recovery");
								net.storeRunningFlag(false);
								return false;
							}
						}
						else
							Adam::update_weight(tt.WOut, tt.vWOut, tt.v2WOut, tt.gWOut,
							                    lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
							                    invBatch, gradScale, wd1, wd2);
						Adam::update_param(tt.bOut, tt.mBOut, tt.v2BOut, tt.gBOut,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                   invBatch, gradScale);
					}
					else if (bimapEnabled)
					{
						const bool useBiMAPWOut = bimap_scope_uses_head(ac);
						if (useBiMAPWOut)
						{
							if (!atlas::bimapUpdate(tt.bimapWOut, &tt.WOut[0], &tt.vWOut[0], &tt.v2WOut[0], &tt.gWOut[0],
							                        outSize, dmTT, lr,
							                        beta1, beta2, inv1mB1t, inv1mB2t, eps,
							                        invBatch, gradScale, wd1, wd2,
							                        ac, net.getLogger(), "tr.WOut"))
							{
								net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
									"SGDHelper_Transformer: BiMAP WOut update entered NaN recovery");
								net.storeRunningFlag(false);
								return false;
							}
						}
						else
							Adam::update_weight(tt.WOut, tt.vWOut, tt.v2WOut, tt.gWOut,
							                    lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
							                    invBatch, gradScale, wd1, wd2);
						Adam::update_param(tt.bOut, tt.mBOut, tt.v2BOut, tt.gBOut,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                   invBatch, gradScale);
					}
					else if (pactEnabled)
					{
						if (!atlas::pactUpdate(tt.pactWOut, &tt.WOut[0], &tt.vWOut[0], &tt.v2WOut[0], &tt.gWOut[0],
						                       outSize, dmTT, lr,
						                       beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                       invBatch, gradScale, wd1, wd2,
						                       ac, net.getLogger(), "tr.WOut"))
						{
							net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
								"SGDHelper_Transformer: PACT WOut update entered NaN recovery");
							net.storeRunningFlag(false);
							return false;
						}
						Adam::update_param(tt.bOut, tt.mBOut, tt.v2BOut, tt.gBOut,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                   invBatch, gradScale);
					}
					else if (racerEnabled)
					{
						if (!atlas::racerUpdate(tt.racerWOut, &tt.WOut[0], &tt.vWOut[0], &tt.v2WOut[0], &tt.gWOut[0],
						                        outSize, dmTT, lr,
						                        beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                        invBatch, gradScale, wd1, wd2,
						                        ac, net.getLogger(), "tr.WOut"))
						{
							net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
								"SGDHelper_Transformer: RACER WOut update entered NaN recovery");
							net.storeRunningFlag(false);
							return false;
						}
						Adam::update_param(tt.bOut, tt.mBOut, tt.v2BOut, tt.gBOut,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                   invBatch, gradScale);
					}
					else if (kronEnabled)
					{
						if (!atlas::kronUpdate(tt.kronWOut, &tt.WOut[0], &tt.vWOut[0], &tt.v2WOut[0], &tt.gWOut[0],
						                       outSize, dmTT, lr,
						                       beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                       invBatch, gradScale, wd1, wd2,
						                       ac, net.getLogger(), "tr.WOut"))
						{
							net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
								"SGDHelper_Transformer: KRON WOut update entered NaN recovery");
							net.storeRunningFlag(false);
							return false;
						}
						Adam::update_param(tt.bOut, tt.mBOut, tt.v2BOut, tt.gBOut,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                   invBatch, gradScale);
					}
					else if (matraEnabled)
					{
						if (!atlas::matraUpdate(tt.matraWOut, &tt.WOut[0], &tt.vWOut[0], &tt.v2WOut[0], &tt.gWOut[0],
						                        outSize, dmTT, lr,
						                        beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                        invBatch, gradScale, wd1, wd2,
						                        ac, net.getLogger(), "tr.WOut"))
						{
							net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
								"SGDHelper_Transformer: MATRA WOut update entered NaN recovery");
							net.storeRunningFlag(false);
							return false;
						}
						Adam::update_param(tt.bOut, tt.mBOut, tt.v2BOut, tt.gBOut,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                   invBatch, gradScale);
					}
					else if (argosEnabled)
					{
						if (argos_scope_uses_head(ac))
						{
							if (!atlas::argosUpdateWithRole(tt.argosWOut, &tt.WOut[0], &tt.vWOut[0], &tt.v2WOut[0], &tt.gWOut[0],
							                                outSize, dmTT, lr,
							                                beta1, beta2, inv1mB1t, inv1mB2t, eps,
							                                invBatch, gradScale, wd1, wd2,
							                                ac,
							                                argos_role_flags_for_head(),
							                                net.getLogger(), "tr.WOut"))
							{
								net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
									"SGDHelper_Transformer: ARGOS WOut update entered NaN recovery");
								net.storeRunningFlag(false);
								return false;
							}
						}
						else
						{
							Adam::update_weight(tt.WOut, tt.vWOut, tt.v2WOut, tt.gWOut,
							                    lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
							                    invBatch, gradScale, wd1, wd2);
						}
						Adam::update_param(tt.bOut, tt.mBOut, tt.v2BOut, tt.gBOut,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                   invBatch, gradScale);
					}
					else if (muonEnabled)
					{
						if (!atlas::muonUpdate(tt.muonWOut, &tt.WOut[0], &tt.vWOut[0], &tt.v2WOut[0], &tt.gWOut[0],
						                       outSize, dmTT, lr,
						                       beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                       invBatch, gradScale, wd1, wd2,
						                       ac, net.getLogger(), "tr.WOut"))
						{
							net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
								"SGDHelper_Transformer: MUON WOut update entered NaN recovery");
							net.storeRunningFlag(false);
							return false;
						}
						Adam::update_param(tt.bOut, tt.mBOut, tt.v2BOut, tt.gBOut,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                   invBatch, gradScale);
					}
					else if (auroraAdamwBackbone)
					{
						Adam::update_weight(tt.WOut, tt.vWOut, tt.v2WOut, tt.gWOut,
						                    lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                    invBatch, gradScale, wd1, wd2);
						Adam::update_param(tt.bOut, tt.mBOut, tt.v2BOut, tt.gBOut,
						                   lr, beta1, beta2, inv1mB1t, inv1mB2t, eps,
						                   invBatch, gradScale);
					}
					else
					{
						// WOut: [outSize, dModel]
						if (!atlas::update(tt.atlasWOut, &tt.WOut[0], &tt.gWOut[0],
						              outSize, dmTT, invBatch, lr, wd1, wd2, gradScale,
						              ac, net.rngEngine, net.getLogger(), "tr.WOut"))
						{
							net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
								"SGDHelper_Transformer: ATLAS WOut update entered NaN recovery");
							net.storeRunningFlag(false);
							return false;
						}

						// bOut: simple SGD
						if (!atlas::updateBias(&tt.bOut[0], &tt.gBOut[0],
						                       static_cast<unsigned int>(tt.bOut.size()),
						                       invBatch, lr, gradScale))
						{
							net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
								"SGDHelper_Transformer: ATLAS bOut update produced NaN/Inf");
							net.storeRunningFlag(false);
							return false;
						}
					}
				}
				else
				{
					// Defensive: ensure gradients are cleared so stale values never leak into later non-tokenLM runs.
					std::fill(tt.gWOut.begin(), tt.gWOut.end(), 0.0f);
					std::fill(tt.gBOut.begin(), tt.gBOut.end(), 0.0f);
				}
			}
			else if (useVesta)
			{
				// VESTA: Bregman mirror descent under von Neumann spectral entropy.
				const glades::VestaConfig& vc = net.trainingConfig.vesta;

				tt.optimizerStep += 1ULL;

				const unsigned int dmTT = tt.dModel;
				const unsigned int dFFTT = tt.dFF;
				const unsigned int nHeadsTT = tt.nHeads;
				const unsigned int nKVHeadsTT = (tt.nKVHeads > 0u ? tt.nKVHeads : nHeadsTT);
				const unsigned int dHeadTT = dmTT / nHeadsTT;
				const unsigned int dModelKVTT = nKVHeadsTT * dHeadTT;
				const unsigned int ffnKindTT = tt.ffnKind;
				const unsigned int ff1WidthTT = (ffnKindTT == static_cast<unsigned int>(glades::TransformerRunConfig::FFN_SWIGLU)) ? (2u * dFFTT) : dFFTT;

				// Token embedding or input projection.
				if (tt.tokenModel)
				{
					const float lr = net.skeleton->getLearningRate(0u) * net.lrScheduleMultiplier * extraLRMult;
					const float wd1 = net.skeleton->getWeightDecay1(0u);
					const float wd2 = net.skeleton->getWeightDecay2(0u);
					if (!vesta::update(tt.vestaTokE, &tt.tokE[0], &tt.gTokE[0],
					                   tt.vocabSize, dmTT, invBatch, lr, wd1, wd2, gradScale,
					                   vc, net.rngEngine, net.getLogger(), "tr.tokE"))
					{
						net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
							"SGDHelper_Transformer: VESTA tokE update produced NaN/Inf");
						net.storeRunningFlag(false);
						return false;
					}
					if (!atlas::updateBias(&tt.lmBias[0], &tt.gLmBias[0],
					                       static_cast<unsigned int>(tt.lmBias.size()),
					                       invBatch, lr, gradScale))
					{
						net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
							"SGDHelper_Transformer: VESTA lmBias update produced NaN/Inf");
						net.storeRunningFlag(false);
						return false;
					}
				}

				// Input projection (index 0)
				{
					const float lr = net.skeleton->getLearningRate(0u) * net.lrScheduleMultiplier * extraLRMult;
					const float wd1 = net.skeleton->getWeightDecay1(0u);
					const float wd2 = net.skeleton->getWeightDecay2(0u);
					if (!vesta::update(tt.vestaWIn, &tt.WIn[0], &tt.gWIn[0],
					                   dmTT, tt.inputSize, invBatch, lr, wd1, wd2, gradScale,
					                   vc, net.rngEngine, net.getLogger(), "tr.WIn"))
					{
						net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
							"SGDHelper_Transformer: VESTA WIn update produced NaN/Inf");
						net.storeRunningFlag(false);
						return false;
					}
					if (!atlas::updateBias(&tt.bIn[0], &tt.gBIn[0],
					                       static_cast<unsigned int>(tt.bIn.size()),
					                       invBatch, lr, gradScale))
					{
						net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
							"SGDHelper_Transformer: VESTA bIn update produced NaN/Inf");
						net.storeRunningFlag(false);
						return false;
					}
				}

				// Blocks.
				for (unsigned int li = 0; li < nLayers; ++li)
				{
					const unsigned int idx = li + 1u;
					const float lr = net.skeleton->getLearningRate(idx) * net.lrScheduleMultiplier * extraLRMult;
					const float wd1 = net.skeleton->getWeightDecay1(idx);
					const float wd2 = net.skeleton->getWeightDecay2(idx);
					TensorTransformerState::Block& b = tt.blocks[li];

					if (!vesta::update(b.vestaWq, &b.Wq[0], &b.gWq[0], dmTT, dmTT,
					                   invBatch, lr, wd1, wd2, gradScale, vc, net.rngEngine, net.getLogger(), "tr.Wq")
					    || !vesta::update(b.vestaWk, &b.Wk[0], &b.gWk[0], dModelKVTT, dmTT,
					                      invBatch, lr, wd1, wd2, gradScale, vc, net.rngEngine, net.getLogger(), "tr.Wk")
					    || !vesta::update(b.vestaWv, &b.Wv[0], &b.gWv[0], dModelKVTT, dmTT,
					                      invBatch, lr, wd1, wd2, gradScale, vc, net.rngEngine, net.getLogger(), "tr.Wv")
					    || !vesta::update(b.vestaWo, &b.Wo[0], &b.gWo[0], dmTT, dmTT,
					                      invBatch, lr, wd1, wd2, gradScale, vc, net.rngEngine, net.getLogger(), "tr.Wo")
					    || !vesta::update(b.vestaW1, &b.W1[0], &b.gW1[0], ff1WidthTT, dmTT,
					                      invBatch, lr, wd1, wd2, gradScale, vc, net.rngEngine, net.getLogger(), "tr.W1")
					    || !vesta::update(b.vestaW2, &b.W2[0], &b.gW2[0], dmTT, dFFTT,
					                      invBatch, lr, wd1, wd2, gradScale, vc, net.rngEngine, net.getLogger(), "tr.W2"))
					{
						net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
							"SGDHelper_Transformer: VESTA block weight update produced NaN/Inf");
						net.storeRunningFlag(false);
						return false;
					}

					if (!atlas::updateBias(&b.bq[0], &b.gBq[0], static_cast<unsigned int>(b.bq.size()), invBatch, lr, gradScale)
					    || !atlas::updateBias(&b.bk[0], &b.gBk[0], static_cast<unsigned int>(b.bk.size()), invBatch, lr, gradScale)
					    || !atlas::updateBias(&b.bv[0], &b.gBv[0], static_cast<unsigned int>(b.bv.size()), invBatch, lr, gradScale)
					    || !atlas::updateBias(&b.bo[0], &b.gBo[0], static_cast<unsigned int>(b.bo.size()), invBatch, lr, gradScale)
					    || !atlas::updateBias(&b.b1[0], &b.gB1[0], static_cast<unsigned int>(b.b1.size()), invBatch, lr, gradScale)
					    || !atlas::updateBias(&b.b2[0], &b.gB2[0], static_cast<unsigned int>(b.b2.size()), invBatch, lr, gradScale)
					    || !atlas::updateBias(&b.ln1Gamma[0], &b.gLn1Gamma[0], static_cast<unsigned int>(b.ln1Gamma.size()), invBatch, lr, gradScale)
					    || !atlas::updateBias(&b.ln1Beta[0], &b.gLn1Beta[0], static_cast<unsigned int>(b.ln1Beta.size()), invBatch, lr, gradScale)
					    || !atlas::updateBias(&b.ln2Gamma[0], &b.gLn2Gamma[0], static_cast<unsigned int>(b.ln2Gamma.size()), invBatch, lr, gradScale)
					    || !atlas::updateBias(&b.ln2Beta[0], &b.gLn2Beta[0], static_cast<unsigned int>(b.ln2Beta.size()), invBatch, lr, gradScale))
					{
						net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
							"SGDHelper_Transformer: VESTA block bias/LN update produced NaN/Inf");
						net.storeRunningFlag(false);
						return false;
					}
				}

				// Final LayerNorm (block-0 LR, no weight decay)
				{
					const float lr = net.skeleton->getLearningRate(1u) * net.lrScheduleMultiplier * extraLRMult;
					if (!atlas::updateBias(&tt.lnFinalGamma[0], &tt.gLnFinalGamma[0],
					                       static_cast<unsigned int>(tt.lnFinalGamma.size()),
					                       invBatch, lr, gradScale)
					    || !atlas::updateBias(&tt.lnFinalBeta[0], &tt.gLnFinalBeta[0],
					                          static_cast<unsigned int>(tt.lnFinalBeta.size()),
					                          invBatch, lr, gradScale))
					{
						net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
							"SGDHelper_Transformer: VESTA lnFinal update produced NaN/Inf");
						net.storeRunningFlag(false);
						return false;
					}
				}

				// Output projection (skip if tokenLM + tied head; gradients should already be routed to tokE).
				if (!tokenLMTiedHead)
				{
					const unsigned int idx = nLayers;
					const float lr = net.skeleton->getLearningRate(idx) * net.lrScheduleMultiplier * extraLRMult;
					const float wd1 = net.skeleton->getWeightDecay1(idx);
					const float wd2 = net.skeleton->getWeightDecay2(idx);
					if (!vesta::update(tt.vestaWOut, &tt.WOut[0], &tt.gWOut[0],
					                   outSize, dmTT, invBatch, lr, wd1, wd2, gradScale,
					                   vc, net.rngEngine, net.getLogger(), "tr.WOut"))
					{
						net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
							"SGDHelper_Transformer: VESTA WOut update produced NaN/Inf");
						net.storeRunningFlag(false);
						return false;
					}
					if (!atlas::updateBias(&tt.bOut[0], &tt.gBOut[0],
					                       static_cast<unsigned int>(tt.bOut.size()),
					                       invBatch, lr, gradScale))
					{
						net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
							"SGDHelper_Transformer: VESTA bOut update produced NaN/Inf");
						net.storeRunningFlag(false);
						return false;
					}
				}
				else
				{
					std::fill(tt.gWOut.begin(), tt.gWOut.end(), 0.0f);
					std::fill(tt.gBOut.begin(), tt.gBOut.end(), 0.0f);
				}
			}
			else if (useHelios)
			{
				// HELIOS: Hamiltonian Ensemble Langevin Integrator with
				// Sharpness-adaptive Thermostat. BAOAB stochastic-symplectic
				// integration of the underdamped Langevin-Nose-Hoover SDE.
				const glades::HeliosConfig& hc = net.trainingConfig.helios;

				tt.optimizerStep += 1ULL;

				const unsigned int dmTT = tt.dModel;
				const unsigned int dFFTT = tt.dFF;
				const unsigned int nHeadsTT = tt.nHeads;
				const unsigned int nKVHeadsTT = (tt.nKVHeads > 0u ? tt.nKVHeads : nHeadsTT);
				const unsigned int dHeadTT = dmTT / nHeadsTT;
				const unsigned int dModelKVTT = nKVHeadsTT * dHeadTT;
				const unsigned int ffnKindTT = tt.ffnKind;
				const unsigned int ff1WidthTT = (ffnKindTT == static_cast<unsigned int>(glades::TransformerRunConfig::FFN_SWIGLU)) ? (2u * dFFTT) : dFFTT;

				if (tt.tokenModel)
				{
					const float lr = net.skeleton->getLearningRate(0u) * net.lrScheduleMultiplier * extraLRMult;
					const float wd1 = net.skeleton->getWeightDecay1(0u);
					const float wd2 = net.skeleton->getWeightDecay2(0u);
					if (!helios::update(tt.heliosTokE, &tt.tokE[0], &tt.gTokE[0],
					                    tt.vocabSize, dmTT, invBatch, lr, wd1, wd2, gradScale,
					                    hc, net.rngEngine, net.getLogger(), "tr.tokE"))
					{
						net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
							"SGDHelper_Transformer: HELIOS tokE update produced NaN/Inf");
						net.storeRunningFlag(false);
						return false;
					}
					if (!atlas::updateBias(&tt.lmBias[0], &tt.gLmBias[0],
					                       static_cast<unsigned int>(tt.lmBias.size()),
					                       invBatch, lr, gradScale))
					{
						net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
							"SGDHelper_Transformer: HELIOS lmBias update produced NaN/Inf");
						net.storeRunningFlag(false);
						return false;
					}
				}

				{
					const float lr = net.skeleton->getLearningRate(0u) * net.lrScheduleMultiplier * extraLRMult;
					const float wd1 = net.skeleton->getWeightDecay1(0u);
					const float wd2 = net.skeleton->getWeightDecay2(0u);
					if (!helios::update(tt.heliosWIn, &tt.WIn[0], &tt.gWIn[0],
					                    dmTT, tt.inputSize, invBatch, lr, wd1, wd2, gradScale,
					                    hc, net.rngEngine, net.getLogger(), "tr.WIn"))
					{
						net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
							"SGDHelper_Transformer: HELIOS WIn update produced NaN/Inf");
						net.storeRunningFlag(false);
						return false;
					}
					if (!atlas::updateBias(&tt.bIn[0], &tt.gBIn[0],
					                       static_cast<unsigned int>(tt.bIn.size()),
					                       invBatch, lr, gradScale))
					{
						net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
							"SGDHelper_Transformer: HELIOS bIn update produced NaN/Inf");
						net.storeRunningFlag(false);
						return false;
					}
				}

				for (unsigned int li = 0; li < nLayers; ++li)
				{
					const unsigned int idx = li + 1u;
					const float lr = net.skeleton->getLearningRate(idx) * net.lrScheduleMultiplier * extraLRMult;
					const float wd1 = net.skeleton->getWeightDecay1(idx);
					const float wd2 = net.skeleton->getWeightDecay2(idx);
					TensorTransformerState::Block& b = tt.blocks[li];

					if (!helios::update(b.heliosWq, &b.Wq[0], &b.gWq[0], dmTT, dmTT,
					                    invBatch, lr, wd1, wd2, gradScale, hc, net.rngEngine, net.getLogger(), "tr.Wq")
					    || !helios::update(b.heliosWk, &b.Wk[0], &b.gWk[0], dModelKVTT, dmTT,
					                       invBatch, lr, wd1, wd2, gradScale, hc, net.rngEngine, net.getLogger(), "tr.Wk")
					    || !helios::update(b.heliosWv, &b.Wv[0], &b.gWv[0], dModelKVTT, dmTT,
					                       invBatch, lr, wd1, wd2, gradScale, hc, net.rngEngine, net.getLogger(), "tr.Wv")
					    || !helios::update(b.heliosWo, &b.Wo[0], &b.gWo[0], dmTT, dmTT,
					                       invBatch, lr, wd1, wd2, gradScale, hc, net.rngEngine, net.getLogger(), "tr.Wo")
					    || !helios::update(b.heliosW1, &b.W1[0], &b.gW1[0], ff1WidthTT, dmTT,
					                       invBatch, lr, wd1, wd2, gradScale, hc, net.rngEngine, net.getLogger(), "tr.W1")
					    || !helios::update(b.heliosW2, &b.W2[0], &b.gW2[0], dmTT, dFFTT,
					                       invBatch, lr, wd1, wd2, gradScale, hc, net.rngEngine, net.getLogger(), "tr.W2"))
					{
						net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
							"SGDHelper_Transformer: HELIOS block weight update produced NaN/Inf");
						net.storeRunningFlag(false);
						return false;
					}

					if (!atlas::updateBias(&b.bq[0], &b.gBq[0], static_cast<unsigned int>(b.bq.size()), invBatch, lr, gradScale)
					    || !atlas::updateBias(&b.bk[0], &b.gBk[0], static_cast<unsigned int>(b.bk.size()), invBatch, lr, gradScale)
					    || !atlas::updateBias(&b.bv[0], &b.gBv[0], static_cast<unsigned int>(b.bv.size()), invBatch, lr, gradScale)
					    || !atlas::updateBias(&b.bo[0], &b.gBo[0], static_cast<unsigned int>(b.bo.size()), invBatch, lr, gradScale)
					    || !atlas::updateBias(&b.b1[0], &b.gB1[0], static_cast<unsigned int>(b.b1.size()), invBatch, lr, gradScale)
					    || !atlas::updateBias(&b.b2[0], &b.gB2[0], static_cast<unsigned int>(b.b2.size()), invBatch, lr, gradScale)
					    || !atlas::updateBias(&b.ln1Gamma[0], &b.gLn1Gamma[0], static_cast<unsigned int>(b.ln1Gamma.size()), invBatch, lr, gradScale)
					    || !atlas::updateBias(&b.ln1Beta[0], &b.gLn1Beta[0], static_cast<unsigned int>(b.ln1Beta.size()), invBatch, lr, gradScale)
					    || !atlas::updateBias(&b.ln2Gamma[0], &b.gLn2Gamma[0], static_cast<unsigned int>(b.ln2Gamma.size()), invBatch, lr, gradScale)
					    || !atlas::updateBias(&b.ln2Beta[0], &b.gLn2Beta[0], static_cast<unsigned int>(b.ln2Beta.size()), invBatch, lr, gradScale))
					{
						net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
							"SGDHelper_Transformer: HELIOS block bias/LN update produced NaN/Inf");
						net.storeRunningFlag(false);
						return false;
					}
				}

				{
					const float lr = net.skeleton->getLearningRate(1u) * net.lrScheduleMultiplier * extraLRMult;
					if (!atlas::updateBias(&tt.lnFinalGamma[0], &tt.gLnFinalGamma[0],
					                       static_cast<unsigned int>(tt.lnFinalGamma.size()),
					                       invBatch, lr, gradScale)
					    || !atlas::updateBias(&tt.lnFinalBeta[0], &tt.gLnFinalBeta[0],
					                          static_cast<unsigned int>(tt.lnFinalBeta.size()),
					                          invBatch, lr, gradScale))
					{
						net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
							"SGDHelper_Transformer: HELIOS lnFinal update produced NaN/Inf");
						net.storeRunningFlag(false);
						return false;
					}
				}

				if (!tokenLMTiedHead)
				{
					const unsigned int idx = nLayers;
					const float lr = net.skeleton->getLearningRate(idx) * net.lrScheduleMultiplier * extraLRMult;
					const float wd1 = net.skeleton->getWeightDecay1(idx);
					const float wd2 = net.skeleton->getWeightDecay2(idx);
					if (!helios::update(tt.heliosWOut, &tt.WOut[0], &tt.gWOut[0],
					                    outSize, dmTT, invBatch, lr, wd1, wd2, gradScale,
					                    hc, net.rngEngine, net.getLogger(), "tr.WOut"))
					{
						net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
							"SGDHelper_Transformer: HELIOS WOut update produced NaN/Inf");
						net.storeRunningFlag(false);
						return false;
					}
					if (!atlas::updateBias(&tt.bOut[0], &tt.gBOut[0],
					                       static_cast<unsigned int>(tt.bOut.size()),
					                       invBatch, lr, gradScale))
					{
						net.lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
							"SGDHelper_Transformer: HELIOS bOut update produced NaN/Inf");
						net.storeRunningFlag(false);
						return false;
					}
				}
				else
				{
					std::fill(tt.gWOut.begin(), tt.gWOut.end(), 0.0f);
					std::fill(tt.gBOut.begin(), tt.gBOut.end(), 0.0f);
				}
			}
			else
			{
				// AdamW (recommended for transformers).
				const float beta1 = net.trainingConfig.optimizer.adamBeta1;
				const float beta2 = net.trainingConfig.optimizer.adamBeta2;
				const float eps = net.trainingConfig.optimizer.adamEps;
				const bool biasCorr = net.trainingConfig.optimizer.adamBiasCorrection;
				const glades::OptimizerConfig& opt = net.trainingConfig.optimizer;

				tt.optimizerStep += 1ULL;
				const double t = static_cast<double>(tt.optimizerStep);
				const double b1t = biasCorr ? pow(static_cast<double>(beta1), t) : 0.0;
				const double b2t = biasCorr ? pow(static_cast<double>(beta2), t) : 0.0;
				const float inv1mB1t = biasCorr ? static_cast<float>(1.0 / (1.0 - b1t)) : 1.0f;
				const float inv1mB2t = biasCorr ? static_cast<float>(1.0 / (1.0 - b2t)) : 1.0f;
				struct AdamGroupScaleCursor
				{
					TensorTransformerState& tt;
					const glades::OptimizerConfig& opt;
					float beta1;
					float beta2;
					float inv1mB1t;
					float inv1mB2t;
					float eps;
					float invBatch;
					float gradScale;
					size_t index;

					AdamGroupScaleCursor(TensorTransformerState& state,
					                     const glades::OptimizerConfig& optimizer,
					                     float beta1_,
					                     float beta2_,
					                     float inv1mB1t_,
					                     float inv1mB2t_,
					                     float eps_,
					                     float invBatch_,
					                     float gradScale_)
					    : tt(state),
					      opt(optimizer),
					      beta1(beta1_),
					      beta2(beta2_),
					      inv1mB1t(inv1mB1t_),
					      inv1mB2t(inv1mB2t_),
					      eps(eps_),
					      invBatch(invBatch_),
					      gradScale(gradScale_),
					      index(0u)
					{
					}

					void ensure_slot(size_t idx)
					{
						if (tt.adamGroupPrevStepRms.size() <= idx)
						{
							tt.adamGroupPrevStepRms.resize(idx + 1u, 0.0f);
							tt.adamGroupLastScale.resize(idx + 1u, 1.0f);
						}
					}

					float next(const std::vector<float>& P,
					           const std::vector<float>& m,
					           const std::vector<float>& v2,
					           const std::vector<float>& gP)
					{
						const size_t idx = index++;
						ensure_slot(idx);
						float nextStepRms = tt.adamGroupPrevStepRms[idx];
						const float scale = Adam::compute_group_scale(
						    P, m, v2, gP,
						    beta1, beta2, inv1mB1t, inv1mB2t, eps,
						    invBatch, gradScale,
						    tt.adamGroupPrevStepRms[idx],
						    opt, &nextStepRms);
						tt.adamGroupPrevStepRms[idx] = nextStepRms;
						tt.adamGroupLastScale[idx] = scale;
						return scale;
					}
				};
				AdamGroupScaleCursor groupScaleCursor(
				    tt, opt, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);

				// Token embedding (index 0 in LM mode)
				if (tt.tokenModel)
				{
					const float lrBase = net.skeleton->getLearningRate(0u) * net.lrScheduleMultiplier * extraLRMult;
					const float wd1 = net.skeleton->getWeightDecay1(0u);
					const float wd2 = net.skeleton->getWeightDecay2(0u);
					const float tokScale = groupScaleCursor.next(tt.tokE, tt.vTokE, tt.v2TokE, tt.gTokE);
					const float biasScale = groupScaleCursor.next(tt.lmBias, tt.mLmBias, tt.v2LmBias, tt.gLmBias);
					Adam::update_weight(tt.tokE, tt.vTokE, tt.v2TokE, tt.gTokE,
					                    lrBase * tokScale, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
					Adam::update_param(tt.lmBias, tt.mLmBias, tt.v2LmBias, tt.gLmBias,
					                   lrBase * biasScale, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
				}

				// Input projection (index 0)
				{
					const float lrBase = net.skeleton->getLearningRate(0u) * net.lrScheduleMultiplier * extraLRMult;
					const float wd1 = net.skeleton->getWeightDecay1(0u);
					const float wd2 = net.skeleton->getWeightDecay2(0u);
					const float weightScale = groupScaleCursor.next(tt.WIn, tt.vWIn, tt.v2WIn, tt.gWIn);
					const float biasScale = groupScaleCursor.next(tt.bIn, tt.mBIn, tt.v2BIn, tt.gBIn);
					Adam::update_weight(tt.WIn, tt.vWIn, tt.v2WIn, tt.gWIn,
					                    lrBase * weightScale, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
					Adam::update_param(tt.bIn, tt.mBIn, tt.v2BIn, tt.gBIn,
					                   lrBase * biasScale, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
				}

				// Blocks (index 1..nLayers)
				for (unsigned int li = 0; li < nLayers; ++li)
				{
					const unsigned int idx = li + 1u;
					const float lrBase = net.skeleton->getLearningRate(idx) * net.lrScheduleMultiplier * extraLRMult;
					const float wd1 = net.skeleton->getWeightDecay1(idx);
					const float wd2 = net.skeleton->getWeightDecay2(idx);
					TensorTransformerState::Block& b = tt.blocks[li];

					Adam::update_weight(b.Wq, b.vWq, b.v2Wq, b.gWq, lrBase * groupScaleCursor.next(b.Wq, b.vWq, b.v2Wq, b.gWq), beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
					Adam::update_weight(b.Wk, b.vWk, b.v2Wk, b.gWk, lrBase * groupScaleCursor.next(b.Wk, b.vWk, b.v2Wk, b.gWk), beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
					Adam::update_weight(b.Wv, b.vWv, b.v2Wv, b.gWv, lrBase * groupScaleCursor.next(b.Wv, b.vWv, b.v2Wv, b.gWv), beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
					Adam::update_weight(b.Wo, b.vWo, b.v2Wo, b.gWo, lrBase * groupScaleCursor.next(b.Wo, b.vWo, b.v2Wo, b.gWo), beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
					Adam::update_weight(b.W1, b.vW1, b.v2W1, b.gW1, lrBase * groupScaleCursor.next(b.W1, b.vW1, b.v2W1, b.gW1), beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
					Adam::update_weight(b.W2, b.vW2, b.v2W2, b.gW2, lrBase * groupScaleCursor.next(b.W2, b.vW2, b.v2W2, b.gW2), beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);

					// Biases + LN params (no weight decay)
					Adam::update_param(b.bq, b.mBq, b.v2Bq, b.gBq, lrBase * groupScaleCursor.next(b.bq, b.mBq, b.v2Bq, b.gBq), beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					Adam::update_param(b.bk, b.mBk, b.v2Bk, b.gBk, lrBase * groupScaleCursor.next(b.bk, b.mBk, b.v2Bk, b.gBk), beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					Adam::update_param(b.bv, b.mBv, b.v2Bv, b.gBv, lrBase * groupScaleCursor.next(b.bv, b.mBv, b.v2Bv, b.gBv), beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					Adam::update_param(b.bo, b.mBo, b.v2Bo, b.gBo, lrBase * groupScaleCursor.next(b.bo, b.mBo, b.v2Bo, b.gBo), beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					Adam::update_param(b.b1, b.mB1, b.v2B1, b.gB1, lrBase * groupScaleCursor.next(b.b1, b.mB1, b.v2B1, b.gB1), beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					Adam::update_param(b.b2, b.mB2, b.v2B2, b.gB2, lrBase * groupScaleCursor.next(b.b2, b.mB2, b.v2B2, b.gB2), beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					Adam::update_param(b.ln1Gamma, b.mLn1Gamma, b.v2Ln1Gamma, b.gLn1Gamma, lrBase * groupScaleCursor.next(b.ln1Gamma, b.mLn1Gamma, b.v2Ln1Gamma, b.gLn1Gamma), beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					Adam::update_param(b.ln1Beta, b.mLn1Beta, b.v2Ln1Beta, b.gLn1Beta, lrBase * groupScaleCursor.next(b.ln1Beta, b.mLn1Beta, b.v2Ln1Beta, b.gLn1Beta), beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					Adam::update_param(b.ln2Gamma, b.mLn2Gamma, b.v2Ln2Gamma, b.gLn2Gamma, lrBase * groupScaleCursor.next(b.ln2Gamma, b.mLn2Gamma, b.v2Ln2Gamma, b.gLn2Gamma), beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					Adam::update_param(b.ln2Beta, b.mLn2Beta, b.v2Ln2Beta, b.gLn2Beta, lrBase * groupScaleCursor.next(b.ln2Beta, b.mLn2Beta, b.v2Ln2Beta, b.gLn2Beta), beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
				}

				// Final LayerNorm (use block 0 LR; no weight decay)
				{
					const float lrBase = net.skeleton->getLearningRate(1u) * net.lrScheduleMultiplier * extraLRMult;
					Adam::update_param(tt.lnFinalGamma, tt.mLnFinalGamma, tt.v2LnFinalGamma, tt.gLnFinalGamma, lrBase * groupScaleCursor.next(tt.lnFinalGamma, tt.mLnFinalGamma, tt.v2LnFinalGamma, tt.gLnFinalGamma), beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
					Adam::update_param(tt.lnFinalBeta, tt.mLnFinalBeta, tt.v2LnFinalBeta, tt.gLnFinalBeta, lrBase * groupScaleCursor.next(tt.lnFinalBeta, tt.mLnFinalBeta, tt.v2LnFinalBeta, tt.gLnFinalBeta), beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
				}

				// Output projection (index nLayers) is unused in token LM tied-head mode.
				if (!tokenLMTiedHead)
				{
					const unsigned int idx = nLayers;
					const float lrBase = net.skeleton->getLearningRate(idx) * net.lrScheduleMultiplier * extraLRMult;
					const float wd1 = net.skeleton->getWeightDecay1(idx);
					const float wd2 = net.skeleton->getWeightDecay2(idx);
					const float weightScale = groupScaleCursor.next(tt.WOut, tt.vWOut, tt.v2WOut, tt.gWOut);
					const float biasScale = groupScaleCursor.next(tt.bOut, tt.mBOut, tt.v2BOut, tt.gBOut);
					Adam::update_weight(tt.WOut, tt.vWOut, tt.v2WOut, tt.gWOut,
					                    lrBase * weightScale, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale, wd1, wd2);
					Adam::update_param(tt.bOut, tt.mBOut, tt.v2BOut, tt.gBOut,
					                   lrBase * biasScale, beta1, beta2, inv1mB1t, inv1mB2t, eps, invBatch, gradScale);
				}
				else
				{
					std::fill(tt.gWOut.begin(), tt.gWOut.end(), 0.0f);
					std::fill(tt.gBOut.begin(), tt.gBOut.end(), 0.0f);
				}
			}

			if (captureGapDiagnostics)
			{
				tt.gapApplyCount += 1ULL;
				const timespec optimizerApplyEnd = monotonic_now();
				tt.gapApplyNsSum += monotonic_elapsed_ns(optimizerApplyStart, optimizerApplyEnd);
			}

			return true;
		}
	};

	TensorTransformerState& tt = tensorTransformer;
	ClearGrads clearGrads(tt);
	ApplyBatch applyBatch(*this, tt, nLayers, dModel, outSize);

	// Helper: AllReduce all gradient vectors across DDP workers (SUM).
	struct DDPReduceGrads
	{
		TensorTransformerState& tt;
		unsigned int nLayers;
		DDPReduceGrads(TensorTransformerState& t, unsigned int nl) : tt(t), nLayers(nl) {}
		void operator()(unsigned int& timeStepsInBatch) const
		{
			if (glades::ddp::worldSize() <= 1) return;

			const int maxBufs = 8 + 16 * static_cast<int>(nLayers);
			std::vector<float*> bufs;
			std::vector<size_t> sizes;
			bufs.reserve(maxBufs);
			sizes.reserve(maxBufs);

			// Global tensors
			if (!tt.gTokE.empty())        { bufs.push_back(&tt.gTokE[0]);        sizes.push_back(tt.gTokE.size()); }
			if (!tt.gLmBias.empty())       { bufs.push_back(&tt.gLmBias[0]);      sizes.push_back(tt.gLmBias.size()); }
			if (!tt.gWIn.empty())          { bufs.push_back(&tt.gWIn[0]);         sizes.push_back(tt.gWIn.size()); }
			if (!tt.gBIn.empty())          { bufs.push_back(&tt.gBIn[0]);         sizes.push_back(tt.gBIn.size()); }
			if (!tt.gWOut.empty())         { bufs.push_back(&tt.gWOut[0]);        sizes.push_back(tt.gWOut.size()); }
			if (!tt.gBOut.empty())         { bufs.push_back(&tt.gBOut[0]);        sizes.push_back(tt.gBOut.size()); }
			if (!tt.gLnFinalGamma.empty()) { bufs.push_back(&tt.gLnFinalGamma[0]); sizes.push_back(tt.gLnFinalGamma.size()); }
			if (!tt.gLnFinalBeta.empty())  { bufs.push_back(&tt.gLnFinalBeta[0]);  sizes.push_back(tt.gLnFinalBeta.size()); }

			// Per-block tensors
			for (unsigned int l = 0; l < nLayers; ++l)
			{
				TensorTransformerState::Block& b = tt.blocks[l];
				if (!b.gWq.empty())      { bufs.push_back(&b.gWq[0]);      sizes.push_back(b.gWq.size()); }
				if (!b.gWk.empty())      { bufs.push_back(&b.gWk[0]);      sizes.push_back(b.gWk.size()); }
				if (!b.gWv.empty())      { bufs.push_back(&b.gWv[0]);      sizes.push_back(b.gWv.size()); }
				if (!b.gWo.empty())      { bufs.push_back(&b.gWo[0]);      sizes.push_back(b.gWo.size()); }
				if (!b.gBq.empty())      { bufs.push_back(&b.gBq[0]);      sizes.push_back(b.gBq.size()); }
				if (!b.gBk.empty())      { bufs.push_back(&b.gBk[0]);      sizes.push_back(b.gBk.size()); }
				if (!b.gBv.empty())      { bufs.push_back(&b.gBv[0]);      sizes.push_back(b.gBv.size()); }
				if (!b.gBo.empty())      { bufs.push_back(&b.gBo[0]);      sizes.push_back(b.gBo.size()); }
				if (!b.gLn1Gamma.empty()){ bufs.push_back(&b.gLn1Gamma[0]);sizes.push_back(b.gLn1Gamma.size()); }
				if (!b.gLn1Beta.empty()) { bufs.push_back(&b.gLn1Beta[0]); sizes.push_back(b.gLn1Beta.size()); }
				if (!b.gLn2Gamma.empty()){ bufs.push_back(&b.gLn2Gamma[0]);sizes.push_back(b.gLn2Gamma.size()); }
				if (!b.gLn2Beta.empty()) { bufs.push_back(&b.gLn2Beta[0]); sizes.push_back(b.gLn2Beta.size()); }
				if (!b.gW1.empty())      { bufs.push_back(&b.gW1[0]);      sizes.push_back(b.gW1.size()); }
				if (!b.gW2.empty())      { bufs.push_back(&b.gW2[0]);      sizes.push_back(b.gW2.size()); }
				if (!b.gB1.empty())      { bufs.push_back(&b.gB1[0]);      sizes.push_back(b.gB1.size()); }
				if (!b.gB2.empty())      { bufs.push_back(&b.gB2[0]);      sizes.push_back(b.gB2.size()); }
			}

			int numBufs = static_cast<int>(bufs.size());
			glades::ddp::allReduceSumInPlaceBucketed(
				numBufs > 0 ? &bufs[0] : NULL,
				numBufs > 0 ? &sizes[0] : NULL,
				numBufs,
				&timeStepsInBatch, 1);
		}
	};
	DDPReduceGrads ddpReduceGrads(tt, nLayers);

	// Mixed precision helper (must live inside this member function because TensorTransformerState is private).
	struct MixedPrecisionHelper
	{
		static void quantize_lowp(const std::vector<float>& src, std::vector<uint16_t>& dst, int lowpDType)
		{
			dst.resize(src.size());
			for (size_t i = 0; i < src.size(); ++i)
				dst[i] = glades::transformer_kernels::float_to_lowp(src[i], lowpDType);
		}

		static int lowp_dtype_from_cfg(const glades::TrainingConfig& cfg)
		{
			return (cfg.mixedPrecision.weightDType == glades::MixedPrecisionConfig::WEIGHT_BF16) ? glades::transformer_kernels::LOWP_BF16
			                                                                                     : glades::transformer_kernels::LOWP_F16;
		}

		static void ensure_transformer_lowp_weights(TensorTransformerState& tt, const glades::TrainingConfig& cfg)
		{
			const bool mpEnable = cfg.mixedPrecision.enable &&
			                      (cfg.mixedPrecision.weightDType != glades::MixedPrecisionConfig::WEIGHT_F32);
			if (!mpEnable)
			{
				tt.mpLowpReady = false;
				tt.mpLowpDType = 0;
				tt.tokELowp.clear();
				tt.WInLowp.clear();
				tt.WOutLowp.clear();
				for (size_t li = 0; li < tt.blocks.size(); ++li)
				{
					TensorTransformerState::Block& b = tt.blocks[li];
					b.WqLowp.clear();
					b.WkLowp.clear();
					b.WvLowp.clear();
					b.WoLowp.clear();
					b.W1Lowp.clear();
					b.W2Lowp.clear();
				}
				return;
			}

			const int lowpDType = lowp_dtype_from_cfg(cfg);
			tt.mpLowpDType = lowpDType;

			// Initialize loss scale on first use.
			if (cfg.mixedPrecision.useLossScaling)
			{
				if (!tt.mpLowpReady)
					tt.mpLossScale = cfg.mixedPrecision.lossScaleInit;
				if (tt.mpLossScale < cfg.mixedPrecision.lossScaleMin)
					tt.mpLossScale = cfg.mixedPrecision.lossScaleMin;
				if (tt.mpLossScale > cfg.mixedPrecision.lossScaleMax)
					tt.mpLossScale = cfg.mixedPrecision.lossScaleMax;
			}
			else
			{
				tt.mpLossScale = 1.0f;
			}

			// Always resync from master when called: correctness-first baseline.
			quantize_lowp(tt.tokE, tt.tokELowp, lowpDType);
			quantize_lowp(tt.WIn, tt.WInLowp, lowpDType);
			quantize_lowp(tt.WOut, tt.WOutLowp, lowpDType);
			for (size_t li = 0; li < tt.blocks.size(); ++li)
			{
				TensorTransformerState::Block& b = tt.blocks[li];
				quantize_lowp(b.Wq, b.WqLowp, lowpDType);
				quantize_lowp(b.Wk, b.WkLowp, lowpDType);
				quantize_lowp(b.Wv, b.WvLowp, lowpDType);
				quantize_lowp(b.Wo, b.WoLowp, lowpDType);
				quantize_lowp(b.W1, b.W1Lowp, lowpDType);
				quantize_lowp(b.W2, b.W2Lowp, lowpDType);
			}
			tt.mpLowpReady = true;
		}

		static bool grads_all_finite(const TensorTransformerState& tt)
		{
			for (size_t i = 0; i < tt.gTokE.size(); ++i)
				if (!glades::transformer_kernels::is_finite(tt.gTokE[i]))
					return false;
			for (size_t i = 0; i < tt.gLmBias.size(); ++i)
				if (!glades::transformer_kernels::is_finite(tt.gLmBias[i]))
					return false;
			for (size_t i = 0; i < tt.gWIn.size(); ++i)
				if (!glades::transformer_kernels::is_finite(tt.gWIn[i]))
					return false;
			for (size_t i = 0; i < tt.gBIn.size(); ++i)
				if (!glades::transformer_kernels::is_finite(tt.gBIn[i]))
					return false;
			for (size_t i = 0; i < tt.gWOut.size(); ++i)
				if (!glades::transformer_kernels::is_finite(tt.gWOut[i]))
					return false;
			for (size_t i = 0; i < tt.gBOut.size(); ++i)
				if (!glades::transformer_kernels::is_finite(tt.gBOut[i]))
					return false;
			for (size_t li = 0; li < tt.blocks.size(); ++li)
			{
				const TensorTransformerState::Block& b = tt.blocks[li];
				const std::vector<float>* gv[] = {&b.gLn1Gamma, &b.gLn1Beta, &b.gWq, &b.gWk, &b.gWv, &b.gWo, &b.gBq, &b.gBk, &b.gBv, &b.gBo,
				                                  &b.gLn2Gamma, &b.gLn2Beta, &b.gW1, &b.gW2, &b.gB1, &b.gB2};
				for (unsigned int gi = 0; gi < (sizeof(gv) / sizeof(gv[0])); ++gi)
				{
					const std::vector<float>& g = *gv[gi];
					for (size_t j = 0; j < g.size(); ++j)
						if (!glades::transformer_kernels::is_finite(g[j]))
							return false;
				}
			}
			return true;
		}

		static void scale_all_grads(TensorTransformerState& tt, float scale)
		{
			for (size_t i = 0; i < tt.gTokE.size(); ++i)
				tt.gTokE[i] *= scale;
			for (size_t i = 0; i < tt.gLmBias.size(); ++i)
				tt.gLmBias[i] *= scale;
			for (size_t i = 0; i < tt.gWIn.size(); ++i)
				tt.gWIn[i] *= scale;
			for (size_t i = 0; i < tt.gBIn.size(); ++i)
				tt.gBIn[i] *= scale;
			for (size_t i = 0; i < tt.gWOut.size(); ++i)
				tt.gWOut[i] *= scale;
			for (size_t i = 0; i < tt.gBOut.size(); ++i)
				tt.gBOut[i] *= scale;
			for (size_t li = 0; li < tt.blocks.size(); ++li)
			{
				TensorTransformerState::Block& b = tt.blocks[li];
				std::vector<float>* gv[] = {&b.gLn1Gamma, &b.gLn1Beta, &b.gWq, &b.gWk, &b.gWv, &b.gWo, &b.gBq, &b.gBk, &b.gBv, &b.gBo,
				                            &b.gLn2Gamma, &b.gLn2Beta, &b.gW1, &b.gW2, &b.gB1, &b.gB2};
				for (unsigned int gi = 0; gi < (sizeof(gv) / sizeof(gv[0])); ++gi)
				{
					std::vector<float>& g = *gv[gi];
					for (size_t j = 0; j < g.size(); ++j)
						g[j] *= scale;
				}
			}
		}
	};

	// Mixed precision (Transformer):
	// - FP32 master weights live in tt.* vectors (single source of truth).
	// - Optional low-precision copies are kept in tt.*Lowp and used for forward/backward matmuls.
	const bool mpEnable = trainingConfig.mixedPrecision.enable &&
	                      (trainingConfig.mixedPrecision.weightDType != glades::MixedPrecisionConfig::WEIGHT_F32);
	const bool mpUseLossScaling = mpEnable && trainingConfig.mixedPrecision.useLossScaling;
	const bool mpDynamicLossScaling = mpUseLossScaling && trainingConfig.mixedPrecision.dynamicLossScaling;
	if (mpEnable)
		MixedPrecisionHelper::ensure_transformer_lowp_weights(tt, trainingConfig);
	const bool useLowpWeights = mpEnable && tt.mpLowpReady;
	const int lowpDType = tt.mpLowpDType;

	// Attention scratch buffers (reused) to avoid per-head allocations.
	const unsigned int nKVHeads = (ttConst.nKVHeads > 0u ? ttConst.nKVHeads : nHeads);
	if (nHeads == 0u || (dModel % nHeads) != 0u)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "SGDHelper_TRANSFORMER: dModel is not divisible by nHeads");
		storeRunningFlag(false);
		return;
	}
	if ((nHeads % nKVHeads) != 0u)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "SGDHelper_TRANSFORMER: nHeads is not divisible by nKVHeads");
		storeRunningFlag(false);
		return;
	}
	const unsigned int dHead = dModel / nHeads;
	const unsigned int dModelKV = nKVHeads * dHead;
	const unsigned int ff1Width = (ffnKind == static_cast<int>(glades::TransformerRunConfig::FFN_SWIGLU)) ? (2u * dFF) : dFF;

	// Bundle config for extracted sub-functions.
	TransformerEpochCfg epochCfg;
	epochCfg.inputSize = inputSize;
	epochCfg.outSize = outSize;
	epochCfg.dModel = dModel;
	epochCfg.dFF = dFF;
	epochCfg.nHeads = nHeads;
	epochCfg.nKVHeads = nKVHeads;
	epochCfg.nLayers = nLayers;
	epochCfg.vocabSize = vocabSize;
	epochCfg.dHead = dHead;
	epochCfg.dModelKV = dModelKV;
	epochCfg.ff1Width = ff1Width;
	epochCfg.padTokenId = padTokenId;
	epochCfg.causal = causal;
	epochCfg.tokenLM = tokenLM;
	epochCfg.tieEmb = tieEmb;
	epochCfg.isTrain = isTrain;
	epochCfg.gradClip = gradClip;
	epochCfg.lnEps = lnEps;
	epochCfg.ropeTheta = ropeTheta;
	epochCfg.costFx = costFx;
	epochCfg.posEnc = posEnc;
	epochCfg.normType = normType;
	epochCfg.ffnKind = ffnKind;
	epochCfg.ffnAct = ffnAct;
	epochCfg.ropeDimOverride = ropeDimOverride;
	epochCfg.tokenLmNegK = tokenLmNegK;
	epochCfg.ddpEnabled = ddpEnabled;
	epochCfg.useLowpWeights = useLowpWeights;
	epochCfg.lowpDType = lowpDType;
	epochCfg.mpEnable = mpEnable;
	epochCfg.mpUseLossScaling = mpUseLossScaling;
	epochCfg.mpDynamicLossScaling = mpDynamicLossScaling;
	epochCfg.seqBatchMax = seqBatchMax;
	epochCfg.tokenLmLossKind = tokenLmLossKind;
	epochCfg.tokenLmAllowHuge = tokenLmAllowHuge;
	epochCfg.lrScheduleMultiplier = lrScheduleMultiplier;

	// Token LM metrics:
	// Accumulate mean NLL over non-pad tokens (natural log).
	double tokenLmNllSum = 0.0;
	unsigned long long tokenLmTokenCount = 0ULL;

	unsigned long long tokensProcessed = 0ULL;
	unsigned long long targetsProcessed = 0ULL; // token LM: non-pad targets; else: timesteps

	// Emit ~20 progress updates per epoch (plus final).
	unsigned int progressEverySeq = 1u;
	if (seqCount > 20u)
		progressEverySeq = seqCount / 20u;
	if (progressEverySeq == 0u)
		progressEverySeq = 1u;
	int64_t lastProgressMs = epochStartMs;
	static const int64_t kProgressIntervalMs = 5000; // 5s heartbeat

	struct MinibatchDriver
	{
		glades::TrainingConfig& trainingConfig;
		TensorTransformerState& tt;
		ClearGrads& clearGrads;
		ApplyBatch& applyBatch;
		DDPReduceGrads& ddpReduceGrads;
		shmea::GLogger* logger;
		bool ddpEnabled;
		bool mpEnable;
		bool mpUseLossScaling;
		bool mpDynamicLossScaling;
		unsigned int optimizerStepsPerEpoch;
		int epochIdx;
		int lrScheduleEpochOffset;
		float& lrScheduleMultiplier;
		int netType;

		MinibatchDriver(glades::TrainingConfig& cfg,
		                TensorTransformerState& state,
		                ClearGrads& clearFn,
		                ApplyBatch& applyFn,
		                DDPReduceGrads& ddpReduceFn,
		                shmea::GLogger* log,
		                bool ddpOn,
		                bool mpOn,
		                bool mpLossScaleOn,
		                bool mpDynamicOn,
		                unsigned int stepsPerEpoch,
		                int epoch,
		                int epochOffset,
		                float& lrMult,
		                int type)
		    : trainingConfig(cfg),
		      tt(state),
		      clearGrads(clearFn),
		      applyBatch(applyFn),
		      ddpReduceGrads(ddpReduceFn),
		      logger(log),
		      ddpEnabled(ddpOn),
		      mpEnable(mpOn),
		      mpUseLossScaling(mpLossScaleOn),
		      mpDynamicLossScaling(mpDynamicOn),
		      optimizerStepsPerEpoch(stepsPerEpoch),
		      epochIdx(epoch),
		      lrScheduleEpochOffset(epochOffset),
		      lrScheduleMultiplier(lrMult),
		      netType(type)
		{
		}

		void log_loss_scale_change(const char* eventName, float prevScale) const
		{
			if (!logger)
				return;
			std::ostringstream oss;
			oss << "event=" << eventName;
			append_logfmt_kv(oss, "net_type", netType);
			append_logfmt_kv(oss, "epoch", epochIdx);
			append_logfmt_kv(oss, "optimizer_step", static_cast<unsigned long long>(tt.optimizerStep));
			append_logfmt_kv(oss, "loss_scale_prev", prevScale);
			append_logfmt_kv(oss, "loss_scale_new", tt.mpLossScale);
			logger->info("NNetwork", shmea::GString(oss.str().c_str()));
		}

		void maybe_backoff_loss_scale()
		{
			if (!mpDynamicLossScaling)
				return;
			const float prev = tt.mpLossScale;
			tt.mpLossScale *= trainingConfig.mixedPrecision.backoffFactor;
			if (tt.mpLossScale < trainingConfig.mixedPrecision.lossScaleMin)
				tt.mpLossScale = trainingConfig.mixedPrecision.lossScaleMin;
			tt.mpLossScaleGoodSteps = 0;
			log_loss_scale_change("nn_loss_scale_backoff", prev);
		}

		void maybe_grow_loss_scale()
		{
			if (!mpDynamicLossScaling)
				return;
			tt.mpLossScaleGoodSteps += 1;
			if (tt.mpLossScaleGoodSteps < trainingConfig.mixedPrecision.growthInterval)
				return;
			const float prev = tt.mpLossScale;
			tt.mpLossScale *= trainingConfig.mixedPrecision.growthFactor;
			if (tt.mpLossScale > trainingConfig.mixedPrecision.lossScaleMax)
				tt.mpLossScale = trainingConfig.mixedPrecision.lossScaleMax;
			tt.mpLossScaleGoodSteps = 0;
			if (tt.mpLossScale != prev)
				log_loss_scale_change("nn_loss_scale_grow", prev);
		}

		bool apply_ready_batch(unsigned int& batchTimeSteps,
		                       unsigned int stepInEpoch)
		{
			if (batchTimeSteps == 0u)
				return true;

			if (mpUseLossScaling && !MixedPrecisionHelper::grads_all_finite(tt))
			{
				maybe_backoff_loss_scale();
				clearGrads();
				return true;
			}

			if (mpUseLossScaling && tt.mpLossScale != 1.0f)
				MixedPrecisionHelper::scale_all_grads(tt, 1.0f / tt.mpLossScale);

			if (ddpEnabled)
				ddpReduceGrads(batchTimeSteps);

			lrScheduleMultiplier = transformer_schedule_multiplier(
			    trainingConfig.lrSchedule,
			    epochIdx + lrScheduleEpochOffset,
			    stepInEpoch,
			    optimizerStepsPerEpoch);
			if (!applyBatch(batchTimeSteps))
				return false;

			maybe_grow_loss_scale();

			if (mpEnable)
				MixedPrecisionHelper::ensure_transformer_lowp_weights(tt, trainingConfig);
			return true;
		}
	};

	MinibatchDriver minibatchDriver(trainingConfig, tt,
	                                clearGrads, applyBatch, ddpReduceGrads,
	                                logger, ddpEnabled, mpEnable,
	                                mpUseLossScaling, mpDynamicLossScaling,
	                                optimizerStepsPerEpoch, epochIdx,
	                                lrScheduleEpochOffset, lrScheduleMultiplier,
	                                netType);

#ifdef GLADES_HAVE_CUDA
	// === GPU accelerated training path ===
	//
	// When GPU is enabled and available, offload the entire forward/backward/optimizer
	// loop to the GPU. Weights stay GPU-resident; only token inputs and loss/metrics
	// cross PCIe per sequence.
	//
	// Keep the launch path narrow: allocation/upload ownership live behind dedicated helpers.
	if (tryRunTransformerGpuEpoch(epochCfg, seqCount, epochIdx, epochStartMs,
	                              tokensProcessed, targetsProcessed,
	                              tokenLmNllSum, tokenLmTokenCount,
	                              clsCorrect, clsTotal, logger))
		return; // GPU path complete; skip CPU fallback.
#endif // GLADES_HAVE_CUDA

	if (trainingConfig.transformer.captureOptimizerGapDiagnostics)
	{
		tt.gapApplyCount = 0ULL;
		tt.gapInputUpdateNormSum = 0.0;
		tt.gapBlockUpdateNormSums.assign(nLayers, 0.0);
		tt.gapFinalNormUpdateNormSum = 0.0;
		tt.gapHeadUpdateNormSum = 0.0;
		tt.gapHeadShareSum = 0.0;
		tt.gapNonHeadShareSum = 0.0;
		tt.gapApplyNsSum = 0.0;
		tt.gapMarginSnapshotCount = 0ULL;
		tt.gapTargetMarginSum = 0.0;
		tt.gapHardNegativeLogitSum = 0.0;
	}

	// Build shuffled sequence order. When DDP is active, each rank uses a different
	// seed so workers process sequences in different orders (reducing correlation).
	std::vector<unsigned int> seqOrder(seqCount);
	for (unsigned int si = 0; si < seqCount; ++si)
		seqOrder[si] = si;
	if (isTrain && seqCount > 1u)
	{
		unsigned int seed = static_cast<unsigned int>(epochIdx * 31 + 7);
		if (ddpEnabled)
			seed += static_cast<unsigned int>(glades::ddp::rank()) * 1000003u;
		// Fisher-Yates shuffle with a simple LCG.
		for (unsigned int i = seqCount - 1; i > 0; --i)
		{
			seed = seed * 1664525u + 1013904223u;
			const unsigned int j = seed % (i + 1u);
			const unsigned int tmp = seqOrder[i];
			seqOrder[i] = seqOrder[j];
			seqOrder[j] = tmp;
		}
	}

	for (unsigned int si = 0; si < seqCount; ++si)
	{
		if (!loadRunningFlag())
			break;

		const unsigned int s = seqOrder[si];
		const unsigned int T = isTrain ? di->getTrainSequenceLength(s) : di->getTestSequenceLength(s);
		if (T == 0u)
			continue;
		tokensProcessed += static_cast<unsigned long long>(T);

		// For sampled-softmax token LM, we do NOT allocate [T,vocab] logits/probs/dLogits.
		// Instead, logits/probs are sized [T,(1+K)] for K negatives per token.
		unsigned int scratchOutSize = outSize;
		unsigned int sampleCount = 0u;
		if (tokenLM && (tokenLmLossKind == glades::TransformerRunConfig::TOKEN_LM_SAMPLED_SOFTMAX))
		{
			const unsigned int K = static_cast<unsigned int>(tokenLmNegK);
			sampleCount = 1u + K;
			scratchOutSize = sampleCount;
		}

		// Guardrail: full-softmax token LM allocates O(T*vocab) buffers (logits+probs+dLogits).
		if (tokenLM && (tokenLmLossKind == glades::TransformerRunConfig::TOKEN_LM_FULL_SOFTMAX) && !tokenLmAllowHuge)
		{
			const size_t Tv = static_cast<size_t>(T) * static_cast<size_t>(vocabSize);
			// logits + probs + dLogits ~= 3 buffers of float32
			const size_t bytes = Tv * 3u * sizeof(float);
			// Hard cap to stop pretending this backend trains real LLMs via full softmax.
			static const size_t kMaxSoftmaxScratchBytes = static_cast<size_t>(256ull * 1024ull * 1024ull);
			if (Tv > 0u && bytes > kMaxSoftmaxScratchBytes)
			{
				std::ostringstream oss;
				oss << "SGDHelper_TRANSFORMER: token LM full softmax would allocate ~" << (bytes / (1024ull * 1024ull))
				    << " MiB just for logits/probs/dLogits (T=" << T << ", vocab=" << vocabSize << "). "
				    << "Use sampled-softmax (tokenLmLossKind=SAMPLED) or set tokenLmAllowHugeFullSoftmax=1 to override.";
				lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, oss.str());
				storeRunningFlag(false);
				return;
			}
		}

		transformerScratch.ensure(T, inputSize, scratchOutSize, dModel, dFF, dModelKV, nHeads, nLayers, ff1Width,
		                         trainingConfig.transformer.embeddingDropoutRate,
		                         trainingConfig.transformer.residualDropoutRate,
		                         trainingConfig.gradientCheckpointing);

		// Load x[t] for this sequence into scratch.x (non-tokenLM).
		// In tokenLM mode, inputs are token ids (ints) and scratch.x is unused.
		std::vector<int> tokenIds;
		std::vector<int> targetIds;
		// Padding mask for attention in token LM mode:
		// keyAllowed[t] == 1 => timestep t participates as a key/value
		// keyAllowed[t] == 0 => timestep t is padding and must be masked out of attention
		std::vector<unsigned char> keyAllowed;
		if (tokenLM)
		{
			tokenIds.assign(T, 0);
			targetIds.assign(T, 0);
			// Ensure the (unused) float input view is deterministic/finite.
			std::fill(transformerScratch.x.begin(),
			          transformerScratch.x.begin() + (static_cast<size_t>(T) * static_cast<size_t>(inputSize)),
			          0.0f);
		}
		for (unsigned int t = 0; t < T; ++t)
		{
			if (tokenLM)
			{
				// Token LM requires first-class token-id accessors (no float casting fallback).
				int tid = 0;
				bool okTok = false;
				if (isTrain)
					okTok = di->getTrainSequenceTokenId(s, t, tid);
				else
					okTok = di->getTestSequenceTokenId(s, t, tid);
				if (!okTok)
				{
					lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
					                            "SGDHelper_TRANSFORMER: token LM requires integer token-id accessors on DataInput (get*SequenceTokenId)");
					storeRunningFlag(false);
					return;
				}
				if (tid < 0 || static_cast<unsigned int>(tid) >= vocabSize)
				{
					lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "SGDHelper_TRANSFORMER: token id out of range");
					storeRunningFlag(false);
					return;
				}
				tokenIds[t] = tid;

				// Expected token id (next token).
				int yid = padTokenId;
				bool okY = false;
				if (isTrain)
					okY = di->getTrainSequenceExpectedTokenId(s, t, yid);
				else
					okY = di->getTestSequenceExpectedTokenId(s, t, yid);
				if (!okY)
				{
					lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
					                            "SGDHelper_TRANSFORMER: token LM requires integer expected-token-id accessors on DataInput (get*SequenceExpectedTokenId)");
					storeRunningFlag(false);
					return;
				}
				targetIds[t] = yid;
			}
			else
			{
				const float* row = NULL;
				unsigned int rowSize = 0u;
				if (isTrain)
					di->getTrainSequenceRowView(s, t, row, rowSize);
				else
					di->getTestSequenceRowView(s, t, row, rowSize);
				const size_t off = static_cast<size_t>(t) * static_cast<size_t>(inputSize);
				for (unsigned int i = 0; i < inputSize; ++i)
				{
					const float v = (row && i < rowSize) ? row[i] : 0.0f;
					if (!is_finite(v))
					{
						lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "SGDHelper_TRANSFORMER: non-finite input sequence value detected (NaN/Inf)");
						storeRunningFlag(false);
						return;
					}
					transformerScratch.x[off + i] = v;
				}
			}
		}

		// Build key mask (token LM only). If padTokenId < 0, masking is disabled.
		keyAllowed.clear();
		if (tokenLM && padTokenId >= 0)
		{
			keyAllowed.assign(T, 1u);
			for (unsigned int t = 0; t < T; ++t)
				if (tokenIds[t] == padTokenId)
					keyAllowed[t] = 0u;
		}

		// Progress counters:
		// - tokenLM: count valid target tokens (non-pad) for loss normalization/throughput.
		// - non-tokenLM: count timesteps.
		if (tokenLM)
		{
			unsigned int valid = 0u;
			for (unsigned int t = 0; t < T; ++t)
			{
				const int yid = targetIds[t];
				if (padTokenId >= 0 && yid == padTokenId)
					continue;
				if (yid < 0 || static_cast<unsigned int>(yid) >= vocabSize)
					continue;
				++valid;
			}
			targetsProcessed += static_cast<unsigned long long>(valid);
		}
		else
		{
			targetsProcessed += static_cast<unsigned long long>(T);
		}

		// === Forward + output head (delegated to extracted method) ===
		transformerCpuForwardPass(epochCfg, T, s, tokenIds, targetIds, keyAllowed, scratchOutSize, sampleCount);


		// === Metrics ===
		if (tokenLM)
		{
			if (tokenLmLossKind == glades::TransformerRunConfig::TOKEN_LM_FULL_SOFTMAX)
			{
				// Next-token cross entropy + top-1 accuracy.
				for (unsigned int t = 0; t < T; ++t)
				{
					const int yid = targetIds[t];
					if (padTokenId >= 0 && yid == padTokenId)
						continue;
					if (yid < 0 || static_cast<unsigned int>(yid) >= vocabSize)
						continue;

					const size_t off = static_cast<size_t>(t) * static_cast<size_t>(vocabSize);
					// argmax
					unsigned int argm = 0u;
					float best = transformerScratch.probs[off + 0u];
					for (unsigned int v = 1u; v < vocabSize; ++v)
					{
						const float p = transformerScratch.probs[off + v];
						if (p > best)
						{
							best = p;
							argm = v;
						}
					}
					++clsTotal;
					if (argm == static_cast<unsigned int>(yid))
						++clsCorrect;

					const float py = clamp_prob01(transformerScratch.probs[off + static_cast<unsigned int>(yid)]);
					tokenLmNllSum += -log(static_cast<double>(py));
					++tokenLmTokenCount;
					if (trainingConfig.transformer.captureOptimizerGapDiagnostics)
					{
						const float targetLogit = transformerScratch.logits[off + static_cast<unsigned int>(yid)];
						float hardNegativeLogit = -std::numeric_limits<float>::infinity();
						for (unsigned int v = 0; v < vocabSize; ++v)
						{
							if (v == static_cast<unsigned int>(yid))
								continue;
							const float z = transformerScratch.logits[off + v];
							if (z > hardNegativeLogit)
								hardNegativeLogit = z;
						}
						if (!is_finite(hardNegativeLogit))
							hardNegativeLogit = targetLogit;
						tt.gapTargetMarginSum += static_cast<double>(targetLogit - hardNegativeLogit);
						tt.gapHardNegativeLogitSum += static_cast<double>(hardNegativeLogit);
						tt.gapMarginSnapshotCount += 1ULL;
					}

					// Results: store [expectedTokenId, predictedTokenId] for last timestep processed.
					results.clear();
					results.addFloat(static_cast<float>(yid));
					results.addFloat(static_cast<float>(argm));
				}
			}
			else
			{
				// Sampled-softmax: loss is NOT exact NLL; do not report as perplexity (handled in Trainer).
				const unsigned int S = scratchOutSize;
				for (unsigned int t = 0; t < T; ++t)
				{
					const int yid = targetIds[t];
					if (padTokenId >= 0 && yid == padTokenId)
						continue;
					if (yid < 0 || static_cast<unsigned int>(yid) >= vocabSize)
						continue;

					const size_t off = static_cast<size_t>(t) * static_cast<size_t>(S);
					const float py = clamp_prob01(transformerScratch.probs[off + 0u]); // col 0 is target
					tokenLmNllSum += -log(static_cast<double>(py));
					++tokenLmTokenCount;
					if (trainingConfig.transformer.captureOptimizerGapDiagnostics)
					{
						const float targetLogit = transformerScratch.logits[off + 0u];
						float hardNegativeLogit = targetLogit;
						for (unsigned int j = 1u; j < S; ++j)
						{
							const float z = transformerScratch.logits[off + j];
							if (j == 1u || z > hardNegativeLogit)
								hardNegativeLogit = z;
						}
						tt.gapTargetMarginSum += static_cast<double>(targetLogit - hardNegativeLogit);
						tt.gapHardNegativeLogitSum += static_cast<double>(hardNegativeLogit);
						tt.gapMarginSnapshotCount += 1ULL;
					}

					// Results: expected token, predicted token is unknown without full vocab.
					results.clear();
					results.addFloat(static_cast<float>(yid));
					results.addFloat(-1.0f);
				}
			}
		}
		else
		{
			// Accumulate loss and confusion per timestep like recurrent paths.
			for (unsigned int t = 0; t < T; ++t)
			{
				results.clear();
				const float* expRow = NULL;
				unsigned int expSize = 0u;
				if (isTrain)
					di->getTrainSequenceExpectedRowView(s, t, expRow, expSize);
				else
					di->getTestSequenceExpectedRowView(s, t, expRow, expSize);

				const unsigned int N = dataSize;
				const double denom = (N > 0 && outSize > 0) ? static_cast<double>(N) * static_cast<double>(outSize) : 1.0;
				const bool useSoftmax = ((costFx == GMath::CLASSIFICATION) || (costFx == GMath::KL)) && (outSize > 1u);
				double loss = 0.0;

				const size_t off = static_cast<size_t>(t) * static_cast<size_t>(outSize);
				for (unsigned int k = 0; k < outSize; ++k)
				{
					const float expv = (expRow && k < expSize) ? expRow[k] : 0.0f;
					const float pred = transformerScratch.probs[off + k];
					results.addFloat(expv);
					results.addFloat(pred);

					if (costFx == GMath::REGRESSION)
					{
						const double diff = static_cast<double>(pred) - static_cast<double>(expv);
						regSSE += diff * diff;
						regSAE += fabs(diff);
						regSumY += static_cast<double>(expv);
						regSumY2 += static_cast<double>(expv) * static_cast<double>(expv);
						++regCount;
					}
					else if (useSoftmax)
					{
						const float p = clamp_prob01(pred);
						if (costFx == GMath::CLASSIFICATION)
							loss += -static_cast<double>(expv) * log(static_cast<double>(p));
						else
							loss += static_cast<double>(GMath::KLDivergence(expv, pred));
					}
					else
					{
						overallTotalError += static_cast<float>(GMath::outputNodeCost(expv, pred, static_cast<float>(denom), costFx));
					}
				}

				if (useSoftmax && N > 0)
					overallTotalError += static_cast<float>(loss / static_cast<double>(N));
				if ((costFx == GMath::CLASSIFICATION) || (costFx == GMath::KL))
				{
					const int expIdx = GMath::argmax(expRow, expSize);
					const int predIdx = GMath::argmax(&transformerScratch.probs[off], outSize);
					confusionMatrix.addResultDirect(static_cast<unsigned int>(expIdx), static_cast<unsigned int>(predIdx));
				}
			}
		}

		// === Backward + grad accumulation (delegated to extracted method) ===
		if (isTrain)
		{
			if (seqInBatch == 0u)
			{
				clearGrads();
				timeStepsInBatch = 0u;
			}

			transformerCpuBackwardPass(epochCfg, T, s, tokenIds, targetIds, scratchOutSize, seqInBatch, timeStepsInBatch);


			++seqInBatch;
			if (seqInBatch >= seqBatchMax)
			{
				// Note: for token LM, timeStepsInBatch counts valid target tokens and may be 0
				// (e.g. all-pad targets). In that case applyBatch() is a no-op.
				if (timeStepsInBatch > 0u)
				{
					// === HELIOS sharpness probe (FD-HVP, framework Section 7 step 10) ===
					// Every K_hvp minibatches, perturb one weight matrix along its
					// momentum direction by +-eps, replay the last sequence's
					// forward+backward, and EMA the resulting directional curvature
					// v^T H v into that matrix's helios::WeightState::kappa. The
					// O-step friction gamma_0 + alpha*kappa then adapts per
					// framework Section 6. Probe is a no-op unless HELIOS is the
					// active optimizer and hc.alpha > 0 and hc.kHvp > 0. One probe
					// event costs 2 extra forward+backward passes on ONE sequence
					// (not the full minibatch); amortized ~2/K_hvp overhead per
					// minibatch.
					{
						const bool useHeliosProbe = (trainingConfig.optimizer.type
						    == glades::OptimizerConfig::HELIOS);
						const glades::HeliosConfig& hc = trainingConfig.helios;
						if (useHeliosProbe && hc.alpha > 0.0f && hc.kHvp > 0u
						    && tokenIds.size() == static_cast<size_t>(T))
						{
							tt.heliosHvpStepCounter++;
							if ((tt.heliosHvpStepCounter
							     % static_cast<unsigned long long>(hc.kHvp)) == 0ULL)
							{
								// Round-robin 6 matrix types * nLayers.
								const unsigned int totalTargets = 6u * nLayers;
								const unsigned int cycleIdx = static_cast<unsigned int>(
								    tt.heliosHvpCycleCounter
								    % static_cast<unsigned long long>(totalTargets));
								tt.heliosHvpCycleCounter++;
								const unsigned int layerIdx = cycleIdx / 6u;
								const unsigned int mtypeIdx = cycleIdx % 6u;

								TensorTransformerState::Block& bPrb = tt.blocks[layerIdx];
								std::vector<float>* W_p = 0;
								std::vector<float>* g_p = 0;
								glades::helios::WeightState* hst = 0;
								switch (mtypeIdx)
								{
									case 0: W_p = &bPrb.Wq; g_p = &bPrb.gWq;
									        hst = &bPrb.heliosWq; break;
									case 1: W_p = &bPrb.Wk; g_p = &bPrb.gWk;
									        hst = &bPrb.heliosWk; break;
									case 2: W_p = &bPrb.Wv; g_p = &bPrb.gWv;
									        hst = &bPrb.heliosWv; break;
									case 3: W_p = &bPrb.Wo; g_p = &bPrb.gWo;
									        hst = &bPrb.heliosWo; break;
									case 4: W_p = &bPrb.W1; g_p = &bPrb.gW1;
									        hst = &bPrb.heliosW1; break;
									case 5: W_p = &bPrb.W2; g_p = &bPrb.gW2;
									        hst = &bPrb.heliosW2; break;
								}

								if (W_p && g_p && hst && hst->initialized
								    && W_p->size() > 0
								    && hst->p.size() == W_p->size())
								{
									const size_t Np = W_p->size();
									// v = p / ||p||. Skip if momentum is ~0 (first
									// minibatch typically): there is no meaningful
									// direction yet.
									double pNorm2 = 0.0;
									for (size_t i = 0; i < Np; ++i)
									{
										const double pi =
										    static_cast<double>(hst->p[i]);
										pNorm2 += pi * pi;
									}
									if (pNorm2 > 1e-12)
									{
										const float invNorm = 1.0f /
										    static_cast<float>(std::sqrt(pNorm2));
										std::vector<float> v(Np);
										for (size_t i = 0; i < Np; ++i)
											v[i] = hst->p[i] * invNorm;

										// Snapshot target matrix weights.
										std::vector<float> W_save = *W_p;

										// Snapshot ALL grad buffers (probe will
										// overwrite via clearGrads+backward).
										std::vector<float> snapTokE = tt.gTokE;
										std::vector<float> snapLmBias = tt.gLmBias;
										std::vector<float> snapWIn = tt.gWIn;
										std::vector<float> snapBIn = tt.gBIn;
										std::vector<float> snapWOut = tt.gWOut;
										std::vector<float> snapBOut = tt.gBOut;
										std::vector<float> snapLnFG = tt.gLnFinalGamma;
										std::vector<float> snapLnFB = tt.gLnFinalBeta;
										std::vector<std::vector<float> > snapBlocks(
										    static_cast<size_t>(nLayers) * 16u);
										for (unsigned int li = 0; li < nLayers; ++li)
										{
											TensorTransformerState::Block& bl = tt.blocks[li];
											const size_t base = static_cast<size_t>(li) * 16u;
											snapBlocks[base+0]  = bl.gLn1Gamma;
											snapBlocks[base+1]  = bl.gLn1Beta;
											snapBlocks[base+2]  = bl.gWq;
											snapBlocks[base+3]  = bl.gWk;
											snapBlocks[base+4]  = bl.gWv;
											snapBlocks[base+5]  = bl.gWo;
											snapBlocks[base+6]  = bl.gBq;
											snapBlocks[base+7]  = bl.gBk;
											snapBlocks[base+8]  = bl.gBv;
											snapBlocks[base+9]  = bl.gBo;
											snapBlocks[base+10] = bl.gLn2Gamma;
											snapBlocks[base+11] = bl.gLn2Beta;
											snapBlocks[base+12] = bl.gW1;
											snapBlocks[base+13] = bl.gW2;
											snapBlocks[base+14] = bl.gB1;
											snapBlocks[base+15] = bl.gB2;
										}

										const float epsFd = 1e-3f;

										// +eps perturbation, replay on last sequence.
										for (size_t i = 0; i < Np; ++i)
											(*W_p)[i] = W_save[i] + epsFd * v[i];
										clearGrads();
										{
											unsigned int rsib = 0u, rtsb = 0u;
											transformerCpuForwardPass(epochCfg, T, s,
											    tokenIds, targetIds, keyAllowed,
											    scratchOutSize, sampleCount);
											transformerCpuBackwardPass(epochCfg, T, s,
											    tokenIds, targetIds, scratchOutSize,
											    rsib, rtsb);
										}
										std::vector<float> gPlus = *g_p;

										// -eps perturbation, replay again.
										for (size_t i = 0; i < Np; ++i)
											(*W_p)[i] = W_save[i] - epsFd * v[i];
										clearGrads();
										{
											unsigned int rsib = 0u, rtsb = 0u;
											transformerCpuForwardPass(epochCfg, T, s,
											    tokenIds, targetIds, keyAllowed,
											    scratchOutSize, sampleCount);
											transformerCpuBackwardPass(epochCfg, T, s,
											    tokenIds, targetIds, scratchOutSize,
											    rsib, rtsb);
										}

										// kappa = v . (gPlus - gMinus) / (2 eps).
										double kappaAcc = 0.0;
										for (size_t i = 0; i < Np; ++i)
										{
											kappaAcc += static_cast<double>(v[i]) *
											    (static_cast<double>(gPlus[i])
											     - static_cast<double>((*g_p)[i]));
										}
										const float kappa = static_cast<float>(
										    kappaAcc / (2.0 * static_cast<double>(epsFd)));

										// Restore target weights.
										*W_p = W_save;

										// Restore all grad buffers (swap is O(1)).
										tt.gTokE.swap(snapTokE);
										tt.gLmBias.swap(snapLmBias);
										tt.gWIn.swap(snapWIn);
										tt.gBIn.swap(snapBIn);
										tt.gWOut.swap(snapWOut);
										tt.gBOut.swap(snapBOut);
										tt.gLnFinalGamma.swap(snapLnFG);
										tt.gLnFinalBeta.swap(snapLnFB);
										for (unsigned int li = 0; li < nLayers; ++li)
										{
											TensorTransformerState::Block& bl = tt.blocks[li];
											const size_t base =
											    static_cast<size_t>(li) * 16u;
											bl.gLn1Gamma.swap(snapBlocks[base+0]);
											bl.gLn1Beta.swap(snapBlocks[base+1]);
											bl.gWq.swap(snapBlocks[base+2]);
											bl.gWk.swap(snapBlocks[base+3]);
											bl.gWv.swap(snapBlocks[base+4]);
											bl.gWo.swap(snapBlocks[base+5]);
											bl.gBq.swap(snapBlocks[base+6]);
											bl.gBk.swap(snapBlocks[base+7]);
											bl.gBv.swap(snapBlocks[base+8]);
											bl.gBo.swap(snapBlocks[base+9]);
											bl.gLn2Gamma.swap(snapBlocks[base+10]);
											bl.gLn2Beta.swap(snapBlocks[base+11]);
											bl.gW1.swap(snapBlocks[base+12]);
											bl.gW2.swap(snapBlocks[base+13]);
											bl.gB1.swap(snapBlocks[base+14]);
											bl.gB2.swap(snapBlocks[base+15]);
										}

											// EMA update kappa on the probed matrix's state.
										glades::helios::updateSharpness(*hst, kappa, hc);
									}
								}
							}
						}
					}
					// === End HELIOS sharpness probe ===

					const unsigned int stepInEpoch = (s + 1u) / seqBatchMax;
					if (!minibatchDriver.apply_ready_batch(timeStepsInBatch, stepInEpoch))
						return;
				}
				seqInBatch = 0u;
				timeStepsInBatch = 0u;
			}

			// Save autotuning record for parity (use last layer's effective LR).
			{
				const float learningRate = skeleton->getLearningRate(nLayers) * lrScheduleMultiplier;
				shmea::GList nbRow;
				nbRow.addFloat(overallTotalAccuracy);
				nbRow.addFloat(learningRate);
				nbRecord.addRow(nbRow);
			}
		}

		// Periodic progress logs within the epoch (exclude last; a final log is emitted below).
		if (logger && (s + 1u) < seqCount)
		{
			const int64_t nowMs = getCurrentTimeMilliseconds();
			const bool dueBySeq = (((s + 1u) % progressEverySeq) == 0u);
			const bool dueByTime = ((nowMs - lastProgressMs) >= kProgressIntervalMs);
			if (!(dueBySeq || dueByTime))
				continue;
			lastProgressMs = nowMs;
			log_transformer_epoch_progress(logger, netType, isTrain, epochIdx,
			                               s + 1u, seqCount, nowMs, epochStartMs,
			                               tokensProcessed, targetsProcessed,
			                               tokenLM, tokenLmLossKind,
			                               tokenLmNllSum, tokenLmTokenCount,
			                               clsCorrect, clsTotal,
			                               overallTotalError, lrScheduleMultiplier,
			                               trainingConfig.globalGradClipNorm,
			                               lastGradNorm, lastGradNormScale,
			                               mpUseLossScaling, tt.mpLossScale,
			                               static_cast<unsigned long long>(tt.optimizerStep));
		}
	} // sequences

	// Progress log (final) and periodic log points.
	// NOTE: emit at end of the epoch regardless of seqCount%progressEverySeq to provide a clear heartbeat.
	if (logger)
	{
		const int64_t nowMs = getCurrentTimeMilliseconds();
		log_transformer_epoch_progress(logger, netType, isTrain, epochIdx,
		                               seqCount, seqCount, nowMs, epochStartMs,
		                               tokensProcessed, targetsProcessed,
		                               tokenLM, tokenLmLossKind,
		                               tokenLmNllSum, tokenLmTokenCount,
		                               clsCorrect, clsTotal,
		                               overallTotalError, lrScheduleMultiplier,
		                               trainingConfig.globalGradClipNorm,
		                               lastGradNorm, lastGradNormScale,
		                               mpUseLossScaling, tt.mpLossScale,
		                               static_cast<unsigned long long>(tt.optimizerStep));
	}

	// DDP: aggregate metrics across all workers before finalization.
	if (ddpEnabled)
	{
		glades::ddp::allReduceSumInPlace(&tokenLmNllSum, 1);
		glades::ddp::allReduceSumInPlace(&tokenLmTokenCount, 1);
		glades::ddp::allReduceSumInPlace(&clsCorrect, 1);
		glades::ddp::allReduceSumInPlace(&clsTotal, 1);
		glades::ddp::allReduceSumInPlace(&regSSE, 1);
		glades::ddp::allReduceSumInPlace(&regSAE, 1);
		glades::ddp::allReduceSumInPlace(&regSumY, 1);
		glades::ddp::allReduceSumInPlace(&regSumY2, 1);
		glades::ddp::allReduceSumInPlace(&regCount, 1);
	}

	// Normalize token LM loss: mean NLL per non-pad token.
	// (Trainer expects overallTotalError to be an epoch-level mean-like quantity.)
	if (tokenLM)
	{
		if (tokenLmTokenCount > 0ULL)
			overallTotalError = static_cast<float>(tokenLmNllSum / static_cast<double>(tokenLmTokenCount));
		else
			overallTotalError = 0.0f;
	}

	// Flush partial minibatch
	if (isTrain && seqInBatch > 0u && timeStepsInBatch > 0u)
	{
		if (!minibatchDriver.apply_ready_batch(timeStepsInBatch, optimizerStepsPerEpoch))
			return;
		seqInBatch = 0u;
		timeStepsInBatch = 0u;
	}
}

// ---------------------------------------------------------------------------
// Extracted CPU forward pass (previously inlined in SGDHelper_TRANSFORMER).
// Computes forward activations through all transformer blocks and produces
// logits/probs in transformerScratch.  Called once per sequence.
// ---------------------------------------------------------------------------
void glades::NNetwork::transformerCpuForwardPass(const TransformerEpochCfg& cfg, unsigned int T, unsigned int s,
                                                 const std::vector<int>& tokenIds,
                                                 const std::vector<int>& targetIds,
                                                 const std::vector<unsigned char>& keyAllowed,
                                                 unsigned int scratchOutSize, unsigned int sampleCount)
{
	TensorTransformerState& tt = tensorTransformer;

	const unsigned int dModel = cfg.dModel;
	const unsigned int dFF = cfg.dFF;
	const unsigned int nHeads = cfg.nHeads;
	const unsigned int nKVHeads = cfg.nKVHeads;
	const unsigned int nLayers = cfg.nLayers;
	const unsigned int vocabSize = cfg.vocabSize;
	const unsigned int inputSize = cfg.inputSize;
	const unsigned int outSize = cfg.outSize;
	const unsigned int dHead = cfg.dHead;
	const unsigned int dModelKV = cfg.dModelKV;
	const unsigned int ff1Width = cfg.ff1Width;
	const bool tokenLM = cfg.tokenLM;
	const bool causal = cfg.causal;
	const int posEnc = cfg.posEnc;
	const int normType = cfg.normType;
	const int ffnKind = cfg.ffnKind;
	const int ffnAct = cfg.ffnAct;
	const float lnEps = cfg.lnEps;
	const float ropeTheta = cfg.ropeTheta;
	const int ropeDimOverride = cfg.ropeDimOverride;
	const bool useLowpWeights = cfg.useLowpWeights;
	const int lowpDType = cfg.lowpDType;
	const int costFx = cfg.costFx;
	const glades::TransformerRunConfig::TokenLMLossKind tokenLmLossKind = cfg.tokenLmLossKind;
	const int tokenLmNegK = cfg.tokenLmNegK;
	const int padTokenId = cfg.padTokenId;
	const bool helmEnabled =
	    tokenLM
	    && cfg.isTrain
	    && (trainingConfig.optimizer.type == glades::OptimizerConfig::ATLAS)
	    && trainingConfig.atlas.helmEnabled
	    && tt.helm.initialized;

	// === Forward ===
	// Input to h
	if (tokenLM)
	{
		// Embedding lookup.
		const bool haveLowpE = useLowpWeights && (tt.tokELowp.size() == tt.tokE.size()) && !tt.tokELowp.empty();
		for (unsigned int t = 0; t < T; ++t)
		{
			const int tid = tokenIds[t];
			const size_t eOff = static_cast<size_t>(tid) * static_cast<size_t>(dModel);
			const size_t hOff = static_cast<size_t>(t) * static_cast<size_t>(dModel);
			for (unsigned int i = 0; i < dModel; ++i)
			{
				transformerScratch.h[hOff + i] = haveLowpE ? glades::transformer_kernels::lowp_to_float(tt.tokELowp[eOff + i], lowpDType)
				                                          : tt.tokE[eOff + i];
			}
		}
	}
	else
	{
		const LinearWeightView inputProj = make_linear_weight_view(tt.WIn, tt.WInLowp, tt.bIn, useLowpWeights, lowpDType);
		linear_forward_maybe_lowp(transformerScratch.x.data(), T, inputSize, inputProj, dModel, transformerScratch.h.data());
	}
	if (posEnc == static_cast<int>(glades::TransformerRunConfig::POSENC_SINUSOIDAL))
	{
		transformerPosEncCache.ensureSinusoidal(dModel);
		add_positional_encoding(transformerScratch.h.empty() ? NULL : &transformerScratch.h[0], T, dModel,
		                        make_double_buffer_view(transformerPosEncCache.sinInvDenomPair));
	}

	// Embedding dropout
	{
		const float embDropRate = trainingConfig.transformer.embeddingDropoutRate;
		if (embDropRate > 0.0f && !transformerScratch.h.empty())
		{
			const size_t hLen = static_cast<size_t>(T) * static_cast<size_t>(dModel);
			unsigned char* mask = transformerScratch.dropoutMaskEmb.empty() ? NULL : &transformerScratch.dropoutMaskEmb[0];
			if (mask)
			{
				glades::transformer_kernels::generate_dropout_mask(rngEngine, mask, hLen, embDropRate);
				const float scale = 1.0f / (1.0f - embDropRate);
				glades::transformer_kernels::apply_dropout_mask_inplace(&transformerScratch.h[0], mask, scale, hLen);
			}
		}
	}

	// RoPE precompute
	const bool useRope = (posEnc == static_cast<int>(glades::TransformerRunConfig::POSENC_ROPE));
	unsigned int ropeDim = dHead;
	if (ropeDimOverride > 0)
	{
		const unsigned int rd = static_cast<unsigned int>(ropeDimOverride);
		ropeDim = (rd < ropeDim) ? rd : ropeDim;
	}
	if ((ropeDim % 2u) != 0u)
		ropeDim -= 1u;
	const std::vector<double>* ropeInvFreq = NULL;
	if (useRope && ropeDim >= 2u)
	{
		transformerPosEncCache.ensureRope(ropeDim, ropeTheta);
		ropeInvFreq = &transformerPosEncCache.ropeInvFreq;
	}
	const unsigned int groupSize = (nKVHeads > 0u) ? (nHeads / nKVHeads) : 0u;

	// Per-layer forward
	for (unsigned int li = 0; li < nLayers; ++li)
	{
		const TensorTransformerState::Block& b = tt.blocks[li];
		const float* hIn = (li == 0u) ? transformerScratch.h.data()
		                              : (transformerScratch.hAfterFF.data() + (static_cast<size_t>(li - 1u) * static_cast<size_t>(T) * static_cast<size_t>(dModel)));

		float* x1 = transformerScratch.x1.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
		float* ln1Mean = transformerScratch.ln1Mean.data() + (static_cast<size_t>(li) * static_cast<size_t>(T));
		float* ln1InvStd = transformerScratch.ln1InvStd.data() + (static_cast<size_t>(li) * static_cast<size_t>(T));
		{
			const bool normWorthParallel = (static_cast<unsigned long long>(T) * dModel >= 65536ULL);
			glades::ThreadPool& pool = glades::ThreadPool::instance();
			if (normWorthParallel && T > 1u && pool.numThreads() > 1u)
			{
				if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
					std::fill(ln1Mean, ln1Mean + T, 0.0f);
				NormFwdCtx nctx;
				nctx.X = hIn;
				nctx.D = dModel;
				nctx.gamma = b.ln1Gamma.empty() ? NULL : &b.ln1Gamma[0];
				nctx.beta = b.ln1Beta.empty() ? NULL : &b.ln1Beta[0];
				nctx.gammaSize = static_cast<unsigned int>(b.ln1Gamma.size());
				nctx.betaSize = static_cast<unsigned int>(b.ln1Beta.size());
				nctx.eps = lnEps;
				nctx.Y = x1;
				nctx.meanOut = (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM)) ? NULL : ln1Mean;
				nctx.invStdOut = ln1InvStd;
				nctx.isRmsNorm = (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM));
				pool.parallel_for(T, norm_fwd_body, &nctx);
			}
			else if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
			{
				std::fill(ln1Mean, ln1Mean + T, 0.0f);
				glades::transformer_kernels::rmsnorm_forward_rows(hIn, T, dModel, b.ln1Gamma, b.ln1Beta, lnEps, x1, ln1InvStd);
			}
			else
			{
				glades::transformer_kernels::layernorm_forward_rows(hIn, T, dModel, b.ln1Gamma, b.ln1Beta, lnEps, x1, ln1Mean, ln1InvStd);
			}
		}

		float* Q = transformerScratch.Q.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
		float* K = transformerScratch.K.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModelKV));
		float* V = transformerScratch.V.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModelKV));

		const LinearWeightView qProj = make_linear_weight_view(b.Wq, b.WqLowp, b.bq, useLowpWeights, lowpDType);
		const LinearWeightView kProj = make_linear_weight_view(b.Wk, b.WkLowp, b.bk, useLowpWeights, lowpDType);
		const LinearWeightView vProj = make_linear_weight_view(b.Wv, b.WvLowp, b.bv, useLowpWeights, lowpDType);
		linear_forward_maybe_lowp(x1, T, dModel, qProj, dModel, Q);
		linear_forward_maybe_lowp(x1, T, dModel, kProj, dModelKV, K);
		linear_forward_maybe_lowp(x1, T, dModel, vProj, dModelKV, V);

		// RoPE
		if (useRope && ropeInvFreq)
		{
			const bool ropeWorthParallel = (static_cast<unsigned long long>(T) * ropeDim >= 4096ULL);
			glades::ThreadPool& pool = glades::ThreadPool::instance();
			if (ropeWorthParallel && pool.numThreads() > 1u)
			{
				if (nHeads > 1u)
				{
					RopeFwdCtx rctx;
					rctx.buf = Q; rctx.T = T; rctx.rowStride = dModel; rctx.dHead = dHead;
					rctx.ropeDim = ropeDim; rctx.invFreq = make_double_buffer_view(*ropeInvFreq); rctx.inverse = false;
					pool.parallel_for(nHeads, rope_body, &rctx);
				}
				else
				{
					glades::transformer_kernels::rope_apply_inplace_strided(Q, T, dModel, dHead, ropeDim, *ropeInvFreq, false);
				}
				if (nKVHeads > 1u)
				{
					RopeFwdCtx rctx;
					rctx.buf = K; rctx.T = T; rctx.rowStride = dModelKV; rctx.dHead = dHead;
					rctx.ropeDim = ropeDim; rctx.invFreq = make_double_buffer_view(*ropeInvFreq); rctx.inverse = false;
					pool.parallel_for(nKVHeads, rope_body, &rctx);
				}
				else
				{
					for (unsigned int hk = 0; hk < nKVHeads; ++hk)
						glades::transformer_kernels::rope_apply_inplace_strided(
						    K + static_cast<size_t>(hk) * static_cast<size_t>(dHead), T, dModelKV, dHead, ropeDim, *ropeInvFreq, false);
				}
			}
			else
			{
				for (unsigned int h = 0; h < nHeads; ++h)
					glades::transformer_kernels::rope_apply_inplace_strided(
					    Q + static_cast<size_t>(h) * static_cast<size_t>(dHead), T, dModel, dHead, ropeDim, *ropeInvFreq, false);
				for (unsigned int hk = 0; hk < nKVHeads; ++hk)
					glades::transformer_kernels::rope_apply_inplace_strided(
					    K + static_cast<size_t>(hk) * static_cast<size_t>(dHead), T, dModelKV, dHead, ropeDim, *ropeInvFreq, false);
			}
		}

		// Multi-head attention forward
		float* attnConcat = transformerScratch.attnConcat.data() +
		                    (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
		std::fill(attnConcat, attnConcat + (static_cast<size_t>(T) * static_cast<size_t>(dModel)), 0.0f);
		{
			const bool attnWorthParallel = (static_cast<unsigned long long>(T) * T * dHead >= 32768ULL);
			glades::ThreadPool& pool = glades::ThreadPool::instance();
			if (attnWorthParallel && nHeads > 1u && pool.numThreads() > 1u)
			{
				AttnFwdCtx actx;
				actx.Q = Q; actx.K = K; actx.V = V; actx.O = attnConcat;
				actx.dModel = dModel; actx.dModelKV = dModelKV; actx.dHead = dHead;
				actx.nHeads = nHeads; actx.nKVHeads = nKVHeads; actx.T = T;
				actx.groupSize = groupSize; actx.causal = causal;
				actx.keyAllowed = keyAllowed.empty() ? NULL : &keyAllowed[0];
				actx.sinkCount = trainingConfig.transformer.attnSinkCount > 0
				    ? static_cast<unsigned int>(trainingConfig.transformer.attnSinkCount) : 0u;
				actx.windowSize = trainingConfig.transformer.localAttnWindow > 0
				    ? static_cast<unsigned int>(trainingConfig.transformer.localAttnWindow) : 0u;
				pool.parallel_for(nHeads, attn_fwd_body, &actx);
			}
			else
			{
				const unsigned int sinkCount_fb = trainingConfig.transformer.attnSinkCount > 0
				    ? static_cast<unsigned int>(trainingConfig.transformer.attnSinkCount) : 0u;
				const unsigned int windowSize_fb = trainingConfig.transformer.localAttnWindow > 0
				    ? static_cast<unsigned int>(trainingConfig.transformer.localAttnWindow) : 0u;
				for (unsigned int h = 0; h < nHeads; ++h)
				{
					const unsigned int kvHead = (nKVHeads == nHeads) ? h : (groupSize > 0u ? (h / groupSize) : 0u);
					glades::transformer_ops::scaled_dot_product_attention_forward_flash_strided_sw(
					    Q + static_cast<size_t>(h) * static_cast<size_t>(dHead), dModel,
					    K + static_cast<size_t>(kvHead) * static_cast<size_t>(dHead), dModelKV,
					    V + static_cast<size_t>(kvHead) * static_cast<size_t>(dHead), dModelKV,
					    T, dHead, dHead, causal,
					    sinkCount_fb, windowSize_fb,
					    attnConcat + static_cast<size_t>(h) * static_cast<size_t>(dHead), dModel,
					    keyAllowed.empty() ? NULL : &keyAllowed[0]);
				}
			}
		}

		// Wo projection
		float* attnOut = transformerScratch.attnOut.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
		const LinearWeightView oProj = make_linear_weight_view(b.Wo, b.WoLowp, b.bo, useLowpWeights, lowpDType);
		linear_forward_maybe_lowp(attnConcat, T, dModel, oProj, dModel, attnOut);

		// Residual attention dropout
		{
			const float resDropRate = trainingConfig.transformer.residualDropoutRate;
			if (resDropRate > 0.0f)
			{
				const size_t layerOff = static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel);
				const size_t n = static_cast<size_t>(T) * static_cast<size_t>(dModel);
				unsigned char* mask = transformerScratch.dropoutMaskResAttn.empty() ? NULL : &transformerScratch.dropoutMaskResAttn[layerOff];
				if (mask)
				{
					glades::transformer_kernels::generate_dropout_mask(rngEngine, mask, n, resDropRate);
					const float scale = 1.0f / (1.0f - resDropRate);
					glades::transformer_kernels::apply_dropout_mask_inplace(attnOut, mask, scale, n);
				}
			}
		}

		// Residual add
		float* hAfterAttn = transformerScratch.hAfterAttn.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
		for (size_t i = 0; i < static_cast<size_t>(T) * static_cast<size_t>(dModel); ++i)
			hAfterAttn[i] = hIn[i] + attnOut[i];

		// LN2 forward
		float* x2 = transformerScratch.x2.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
		float* ln2Mean = transformerScratch.ln2Mean.data() + (static_cast<size_t>(li) * static_cast<size_t>(T));
		float* ln2InvStd = transformerScratch.ln2InvStd.data() + (static_cast<size_t>(li) * static_cast<size_t>(T));
		{
			const bool normWorthParallel = (static_cast<unsigned long long>(T) * dModel >= 65536ULL);
			glades::ThreadPool& pool = glades::ThreadPool::instance();
			if (normWorthParallel && T > 1u && pool.numThreads() > 1u)
			{
				if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
					std::fill(ln2Mean, ln2Mean + T, 0.0f);
				NormFwdCtx nctx;
				nctx.X = hAfterAttn; nctx.D = dModel;
				nctx.gamma = b.ln2Gamma.empty() ? NULL : &b.ln2Gamma[0];
				nctx.beta = b.ln2Beta.empty() ? NULL : &b.ln2Beta[0];
				nctx.gammaSize = static_cast<unsigned int>(b.ln2Gamma.size());
				nctx.betaSize = static_cast<unsigned int>(b.ln2Beta.size());
				nctx.eps = lnEps; nctx.Y = x2;
				nctx.meanOut = (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM)) ? NULL : ln2Mean;
				nctx.invStdOut = ln2InvStd;
				nctx.isRmsNorm = (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM));
				pool.parallel_for(T, norm_fwd_body, &nctx);
			}
			else if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
			{
				std::fill(ln2Mean, ln2Mean + T, 0.0f);
				glades::transformer_kernels::rmsnorm_forward_rows(hAfterAttn, T, dModel, b.ln2Gamma, b.ln2Beta, lnEps, x2, ln2InvStd);
			}
			else
			{
				glades::transformer_kernels::layernorm_forward_rows(hAfterAttn, T, dModel, b.ln2Gamma, b.ln2Beta, lnEps, x2, ln2Mean, ln2InvStd);
			}
		}

		// FFN
		float* ff1 = transformerScratch.ff1.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(ff1Width));
		float* ff1Act = transformerScratch.ff1Act.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dFF));
		const LinearWeightView ff1Proj = make_linear_weight_view(b.W1, b.W1Lowp, b.b1, useLowpWeights, lowpDType);
		linear_forward_maybe_lowp(x2, T, dModel, ff1Proj, ff1Width, ff1);
		if (ffnKind == static_cast<int>(glades::TransformerRunConfig::FFN_SWIGLU))
		{
			for (unsigned int t = 0; t < T; ++t)
			{
				const size_t preOff = static_cast<size_t>(t) * static_cast<size_t>(ff1Width);
				const size_t outOff = static_cast<size_t>(t) * static_cast<size_t>(dFF);
				for (unsigned int i = 0; i < dFF; ++i)
				{
					const float gatePre = ff1[preOff + i];
					const float upPre = ff1[preOff + static_cast<size_t>(dFF) + i];
					ff1Act[outOff + i] = glades::transformer_ops::silu(gatePre) * upPre;
				}
			}
		}
		else
		{
			const size_t actLen = static_cast<size_t>(T) * static_cast<size_t>(dFF);
			if (ffnAct == static_cast<int>(glades::TransformerRunConfig::FFN_GELU))
				glades::transformer_kernels::gelu_forward_buf(ff1, ff1Act, actLen);
			else
				for (size_t i = 0; i < actLen; ++i)
					ff1Act[i] = glades::transformer_ops::relu(ff1[i]);
		}

		float* ffOut = transformerScratch.ffOut.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
		const LinearWeightView ff2Proj = make_linear_weight_view(b.W2, b.W2Lowp, b.b2, useLowpWeights, lowpDType);
		linear_forward_maybe_lowp(ff1Act, T, dFF, ff2Proj, dModel, ffOut);

		// Residual FFN dropout
		{
			const float resDropRate = trainingConfig.transformer.residualDropoutRate;
			if (resDropRate > 0.0f)
			{
				const size_t layerOff = static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel);
				const size_t n = static_cast<size_t>(T) * static_cast<size_t>(dModel);
				unsigned char* mask = transformerScratch.dropoutMaskResFF.empty() ? NULL : &transformerScratch.dropoutMaskResFF[layerOff];
				if (mask)
				{
					glades::transformer_kernels::generate_dropout_mask(rngEngine, mask, n, resDropRate);
					const float scale = 1.0f / (1.0f - resDropRate);
					glades::transformer_kernels::apply_dropout_mask_inplace(ffOut, mask, scale, n);
				}
			}
		}

		// Residual add
		float* hAfterFF = transformerScratch.hAfterFF.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
		for (size_t i = 0; i < static_cast<size_t>(T) * static_cast<size_t>(dModel); ++i)
			hAfterFF[i] = hAfterAttn[i] + ffOut[i];

		// Per-layer NaN detection: check hidden state after each layer and abort early.
		{
			const size_t layerElems = static_cast<size_t>(T) * static_cast<size_t>(dModel);
			if (!glades::transformer_kernels::all_finite_full(hAfterFF, layerElems))
			{
				std::ostringstream oss;
				oss << "transformerCpuForwardPass: non-finite hidden state detected at layer " << li;
				lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, oss.str());
				storeRunningFlag(false);
				return;
			}
		}
	}

	const float* hFinal = transformerScratch.hAfterFF.data() + (static_cast<size_t>(nLayers - 1u) * static_cast<size_t>(T) * static_cast<size_t>(dModel));

	// Final LayerNorm
	float* hPostFinalLN = transformerScratch.hPostFinalLN.data();
	if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
	{
		glades::transformer_kernels::rmsnorm_forward_rows(hFinal, T, dModel,
		    tt.lnFinalGamma, tt.lnFinalBeta, lnEps,
		    hPostFinalLN, transformerScratch.lnFinalInvStd.data());
	}
	else
	{
		glades::transformer_kernels::layernorm_forward_rows(hFinal, T, dModel,
		    tt.lnFinalGamma, tt.lnFinalBeta, lnEps,
		    hPostFinalLN, transformerScratch.lnFinalMean.data(), transformerScratch.lnFinalInvStd.data());
	}
	if (helmEnabled && !tt.helm.forwardCorrection.empty() && tt.helm.lastMemoryGain > 0.0f)
	{
		for (unsigned int t = 0; t < T; ++t)
		{
			const size_t hOff = static_cast<size_t>(t) * static_cast<size_t>(dModel);
			for (unsigned int i = 0; i < dModel && i < tt.helm.forwardCorrection.size(); ++i)
			{
				const float corr = tt.helm.forwardCorrection[i];
				if (std::isfinite(static_cast<double>(corr)))
					hPostFinalLN[hOff + i] += corr;
			}
		}
	}

	// Output head logits
	if (tokenLM)
	{
		if (tokenLmLossKind == glades::TransformerRunConfig::TOKEN_LM_FULL_SOFTMAX)
		{
			if (useLowpWeights && (tt.tokELowp.size() == tt.tokE.size()) && !tt.tokELowp.empty())
			{
				glades::transformer_kernels::tied_embedding_logits_forward_rows_lowp(hPostFinalLN, T, dModel, &tt.tokELowp[0], lowpDType, tt.lmBias,
				                                                                    vocabSize, transformerScratch.logits.data());
			}
			else
			{
				const bool embWorthParallel = (static_cast<unsigned long long>(T) * vocabSize * dModel >= 500000ULL);
				glades::ThreadPool& pool = glades::ThreadPool::instance();
				if (embWorthParallel && T > 1u && pool.numThreads() > 1u)
				{
					TiedEmbLogitsCtx ectx;
					ectx.H = hPostFinalLN; ectx.dModel = dModel;
					ectx.tokE = tt.tokE.empty() ? NULL : &tt.tokE[0];
					ectx.lmBias = tt.lmBias.empty() ? NULL : &tt.lmBias[0];
					ectx.lmBiasSize = static_cast<unsigned int>(tt.lmBias.size());
					ectx.vocab = vocabSize;
					ectx.logitsOut = transformerScratch.logits.data();
					pool.parallel_for(T, tied_emb_logits_body, &ectx);
				}
				else
				{
					glades::transformer_kernels::tied_embedding_logits_forward_rows(hPostFinalLN, T, dModel, tt.tokE, tt.lmBias, vocabSize,
					                                                               transformerScratch.logits.data());
				}
			}
		}
		else
		{
			// Sampled-softmax logits
			const unsigned int K = static_cast<unsigned int>(tokenLmNegK);
			const unsigned int S = sampleCount;
			for (unsigned int t = 0; t < T; ++t)
			{
				const int yid = targetIds[t];
				if ((padTokenId >= 0 && yid == padTokenId) || yid < 0 || static_cast<unsigned int>(yid) >= vocabSize)
				{
					const size_t off = static_cast<size_t>(t) * static_cast<size_t>(S);
					for (unsigned int j = 0u; j < S; ++j)
					{
						transformerScratch.logits[off + j] = 0.0f;
						transformerScratch.tokenLmSampleIds[off + j] = -1;
					}
					continue;
				}

				const size_t off = static_cast<size_t>(t) * static_cast<size_t>(S);
				transformerScratch.tokenLmSampleIds[off + 0u] = yid;
				for (unsigned int k = 0; k < K; ++k)
				{
					int neg = glades::rng::uniform_int(rngEngine, 0, static_cast<int>(vocabSize) - 2);
					if (neg >= yid) ++neg;
					transformerScratch.tokenLmSampleIds[off + 1u + k] = neg;
				}

				const size_t hOff = static_cast<size_t>(t) * static_cast<size_t>(dModel);
				const bool haveLowpE = useLowpWeights && (tt.tokELowp.size() == tt.tokE.size()) && !tt.tokELowp.empty();
				for (unsigned int j = 0u; j < S; ++j)
				{
					const int vid = transformerScratch.tokenLmSampleIds[off + j];
					if (vid < 0 || static_cast<unsigned int>(vid) >= vocabSize)
					{
						transformerScratch.logits[off + j] = 0.0f;
						continue;
					}
					float dot = 0.0f;
					const size_t eOff = static_cast<size_t>(vid) * static_cast<size_t>(dModel);
					if (haveLowpE)
					{
						for (unsigned int d = 0; d < dModel; ++d)
							dot += hPostFinalLN[hOff + d] * glades::transformer_kernels::lowp_to_float(tt.tokELowp[eOff + d], lowpDType);
					}
					else
					{
						dot = glades::transformer_kernels::dot_f32(&hPostFinalLN[hOff], &tt.tokE[eOff], dModel);
					}
					const float bias = (static_cast<size_t>(vid) < tt.lmBias.size()) ? tt.lmBias[static_cast<size_t>(vid)] : 0.0f;
					transformerScratch.logits[off + j] = dot + bias;
				}
			}
		}
	}
	else
	{
		const LinearWeightView outProj = make_linear_weight_view(tt.WOut, tt.WOutLowp, tt.bOut, useLowpWeights, lowpDType);
		linear_forward_maybe_lowp(hPostFinalLN, T, dModel, outProj, outSize, transformerScratch.logits.data());
	}

	// Softmax / sigmoid / identity
	if (tokenLM)
	{
		for (unsigned int t = 0; t < T; ++t)
		{
			const size_t off = static_cast<size_t>(t) * static_cast<size_t>(scratchOutSize);
			glades::transformer_kernels::softmax_stable_into(&transformerScratch.logits[off], static_cast<size_t>(scratchOutSize),
			                                                &transformerScratch.probs[off]);
		}
	}
	else if (((costFx == GMath::CLASSIFICATION) || (costFx == GMath::KL)) && (outSize > 1u))
	{
		for (unsigned int t = 0; t < T; ++t)
		{
			const size_t off = static_cast<size_t>(t) * static_cast<size_t>(outSize);
			glades::transformer_kernels::softmax_stable_into(&transformerScratch.logits[off], static_cast<size_t>(outSize),
			                                                &transformerScratch.probs[off]);
		}
	}
	else if ((costFx == GMath::CLASSIFICATION) && (outSize == 1u))
	{
		for (unsigned int t = 0; t < T; ++t)
		{
			const float z = transformerScratch.logits[static_cast<size_t>(t) * static_cast<size_t>(outSize)];
			transformerScratch.probs[static_cast<size_t>(t) * static_cast<size_t>(outSize)] = GMath::squash(z, GMath::SIGMOID, 0.0f);
		}
	}
	else
	{
		std::copy(transformerScratch.logits.begin(), transformerScratch.logits.end(), transformerScratch.probs.begin());
	}
}

// ---------------------------------------------------------------------------
// Extracted CPU backward pass + gradient accumulation.
// Assumes the forward pass has already been run.
// ---------------------------------------------------------------------------
void glades::NNetwork::transformerCpuBackwardPass(const TransformerEpochCfg& cfg, unsigned int T, unsigned int s,
                                                  const std::vector<int>& tokenIds,
                                                  const std::vector<int>& targetIds,
                                                  unsigned int scratchOutSize,
                                                  unsigned int& seqInBatch,
                                                  unsigned int& timeStepsInBatch)
{
	using namespace glades::sgd_detail;

	TensorTransformerState& tt = tensorTransformer;

	const unsigned int dModel = cfg.dModel;
	const unsigned int dFF = cfg.dFF;
	const unsigned int nHeads = cfg.nHeads;
	const unsigned int nKVHeads = cfg.nKVHeads;
	const unsigned int nLayers = cfg.nLayers;
	const unsigned int vocabSize = cfg.vocabSize;
	const unsigned int inputSize = cfg.inputSize;
	const unsigned int outSize = cfg.outSize;
	const unsigned int dHead = cfg.dHead;
	const unsigned int dModelKV = cfg.dModelKV;
	const unsigned int ff1Width = cfg.ff1Width;
	const bool tokenLM = cfg.tokenLM;
	const bool causal = cfg.causal;
	const int posEnc = cfg.posEnc;
	const int normType = cfg.normType;
	const int ffnKind = cfg.ffnKind;
	const int ffnAct = cfg.ffnAct;
	const float lnEps = cfg.lnEps;
	const float ropeTheta = cfg.ropeTheta;
	const int ropeDimOverride = cfg.ropeDimOverride;
	const bool useLowpWeights = cfg.useLowpWeights;
	const int lowpDType = cfg.lowpDType;
	const int costFx = cfg.costFx;
	const float gradClip = cfg.gradClip;
	const glades::TransformerRunConfig::TokenLMLossKind tokenLmLossKind = cfg.tokenLmLossKind;
	const int padTokenId = cfg.padTokenId;
	const bool mpUseLossScaling = cfg.mpUseLossScaling;
	const glades::ATLASConfig& ac = trainingConfig.atlas;
	const bool helmEnabled =
	    tokenLM
	    && (trainingConfig.optimizer.type == glades::OptimizerConfig::ATLAS)
	    && ac.helmEnabled
	    && tt.helm.initialized;
	const bool asterEnabled =
	    tokenLM
	    && (trainingConfig.optimizer.type == glades::OptimizerConfig::ATLAS)
	    && ac.asterEnabled
	    && tt.aster.initialized;

	const float lossScale = (mpUseLossScaling ? tt.mpLossScale : 1.0f);

#define GLADES_ECHO_CPU_OBSERVE(enabled_, state_, rowObs_, colObs_, samples_, rows_, cols_) do { \
	if (!observe_echo_cpu((state_), (rowObs_), (colObs_), (samples_), (rows_), (cols_), ac, (enabled_))) \
	{ \
		lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, \
		    "transformerCpuBackwardPass: ECHO operand observation failed"); \
		storeRunningFlag(false); \
		return; \
	} \
} while (0)

	std::vector<float, glades::AlignedAllocator<float, 64> >& dLogits = transformerScratch.dLogits;
	if (dLogits.size() != (static_cast<size_t>(T) * static_cast<size_t>(scratchOutSize)))
		dLogits.resize(static_cast<size_t>(T) * static_cast<size_t>(scratchOutSize));
	std::fill(dLogits.begin(), dLogits.end(), 0.0f);

	std::vector<float, glades::AlignedAllocator<float, 64> >& dH = transformerScratch.dH;
	if (dH.size() != (static_cast<size_t>(T) * static_cast<size_t>(dModel)))
		dH.resize(static_cast<size_t>(T) * static_cast<size_t>(dModel));

	const float* hFinal = transformerScratch.hAfterFF.data() + (static_cast<size_t>(nLayers - 1u) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
	float* hPostFinalLN = transformerScratch.hPostFinalLN.data();

	if (tokenLM)
	{
		TensorTransformerState::HelmState* helm = helmEnabled ? &tt.helm : NULL;
		TensorTransformerState::AsterState* aster = asterEnabled ? &tt.aster : NULL;
		const unsigned int asterRegimeCount = (aster && aster->regimeCount > 0u) ? aster->regimeCount : 1u;
		const unsigned int asterSketchSeed = 0xA57E0001u;
		const unsigned int hiddenSketchSeed = 0xA57E1001u;
		unsigned int validTargetsThisSeq = 0u;
		std::vector<int> tokenLastSeen(vocabSize, -1);
		std::fill(dH.begin(), dH.end(), 0.0f);
		if (tokenLmLossKind == glades::TransformerRunConfig::TOKEN_LM_FULL_SOFTMAX)
		{
			for (unsigned int t = 0; t < T; ++t)
			{
				const int yid = targetIds[t];
				if (padTokenId >= 0 && yid == padTokenId) continue;
				if (yid < 0 || static_cast<unsigned int>(yid) >= vocabSize) continue;
				++validTargetsThisSeq;
				if (t < tokenIds.size())
				{
					const int xid = tokenIds[t];
					if (xid >= 0
					    && static_cast<unsigned int>(xid) < vocabSize
					    && !(padTokenId >= 0 && xid == padTokenId))
						tokenLastSeen[static_cast<unsigned int>(xid)] = static_cast<int>(t);
				}
				const size_t off = static_cast<size_t>(t) * static_cast<size_t>(vocabSize);
				const unsigned int supportNegCount =
				    (aster && aster->supportDim > 0u) ? (aster->supportDim - 1u) : 0u;
				std::vector<unsigned int> topNegIds(supportNegCount, vocabSize);
				std::vector<float> topNegResiduals(supportNegCount, 0.0f);
				std::vector<float> topNegLogits(supportNegCount, 0.0f);
				for (unsigned int v = 0; v < vocabSize; ++v)
				{
					const float rawResidual = transformerScratch.probs[off + v]
					                        - ((v == static_cast<unsigned int>(yid)) ? 1.0f : 0.0f);
					dLogits[off + v] = rawResidual;
					if (supportNegCount > 0u
					    && v != static_cast<unsigned int>(yid)
					    && !(padTokenId >= 0 && v == static_cast<unsigned int>(padTokenId)))
					{
						aster_insert_top_support(v, std::max(0.0f, rawResidual),
						                         transformerScratch.logits[off + v],
						                         topNegIds, topNegResiduals, topNegLogits);
					}
				}
				const int targetLastSeen = tokenLastSeen[static_cast<unsigned int>(yid)];
				const float targetRepeatHit = (targetLastSeen >= 0) ? 1.0f : 0.0f;
				const float targetRepeatCloseness =
				    (targetLastSeen >= 0)
				        ? (1.0f / static_cast<float>((t + 1u) - static_cast<unsigned int>(targetLastSeen)))
				        : 0.0f;
				const unsigned int hardNegId =
				    (!topNegIds.empty() && topNegIds[0] < vocabSize) ? topNegIds[0] : vocabSize;
				const int hardNegLastSeen =
				    (hardNegId < vocabSize) ? tokenLastSeen[hardNegId] : -1;
				const float hardNegRepeatHit = (hardNegLastSeen >= 0) ? 1.0f : 0.0f;
				const float hardNegRepeatCloseness =
				    (hardNegLastSeen >= 0)
				        ? (1.0f / static_cast<float>((t + 1u) - static_cast<unsigned int>(hardNegLastSeen)))
				        : 0.0f;
				if (aster)
				{
					const unsigned int trackedLayers =
					    std::min<unsigned int>(aster->hiddenStackDepth,
					                           static_cast<unsigned int>(aster->trackedBlockIndices.size()));
					const unsigned int tokenCondDim = std::max(1u, aster->tokenCondDim);
					const unsigned int regime =
					    aster_transformer_regime_from_logits(&transformerScratch.probs[off],
					                                         &transformerScratch.logits[off],
					                                         vocabSize,
					                                         static_cast<unsigned int>(yid),
					                                         padTokenId);
					const float targetLogit =
					    transformerScratch.logits[off + static_cast<unsigned int>(yid)];
					const float hardNegativeLogit =
					    (supportNegCount > 0u && !topNegIds.empty() && topNegIds[0] < vocabSize)
					        ? topNegLogits[0]
					        : targetLogit;
					const float targetMargin = targetLogit - hardNegativeLogit;
					const float marginBaseline =
					    (regime < aster->targetMarginEma.size()) ? aster->targetMarginEma[regime] : 0.75f;
					if (regime < aster->batchTargetMarginSum.size())
						aster->batchTargetMarginSum[regime] += targetMargin;
					if (regime < aster->batchHardNegativeLogitSum.size())
						aster->batchHardNegativeLogitSum[regime] += hardNegativeLogit;
					if (regime < aster->batchBaselineWorseSum.size())
						aster->batchBaselineWorseSum[regime] += (targetMargin < marginBaseline) ? 1.0f : 0.0f;
					const size_t regimeBundleOff = static_cast<size_t>(regime) * static_cast<size_t>(aster->sketchDim);
					const size_t regimeFinalRawOff = static_cast<size_t>(regime) * static_cast<size_t>(dModel);
					const size_t regimeLayerRawBase =
					    static_cast<size_t>(regime) * static_cast<size_t>(trackedLayers) * static_cast<size_t>(dModel);
					const size_t regimeLayerSketchBase =
					    static_cast<size_t>(regime) * static_cast<size_t>(trackedLayers) * static_cast<size_t>(aster->sketchDim);
					for (unsigned int v = 0; v < vocabSize; ++v)
					{
						const float rawResidual = dLogits[off + v];
						aster_sketch_sparse_value(v, rawResidual, aster->sketchDim,
						                          asterSketchSeed, aster->batchResidualSum, regimeBundleOff);
					}
					const size_t hOff = static_cast<size_t>(t) * static_cast<size_t>(dModel);
					const float* finalHiddenRow = &hPostFinalLN[hOff];
					aster_accumulate_support_role(static_cast<unsigned int>(yid), regime, asterRegimeCount, 0u, aster->supportDim,
					                              transformerScratch.logits[off + static_cast<unsigned int>(yid)],
					                              finalHiddenRow, dModel,
					                              aster->batchSupportLogitSum,
					                              aster->batchSupportCount,
					                              aster->batchSupportHiddenRawSum,
					                              aster->batchSupportIds,
					                              aster->batchSupportRoleCounts);
					aster_accumulate_support_value_only(regime, 0u, aster->supportDim,
					                                    dLogits[off + static_cast<unsigned int>(yid)],
					                                    aster->batchSupportResidualSum);
					for (unsigned int r = 0; r < supportNegCount; ++r)
					{
						if (topNegIds[r] >= vocabSize)
							continue;
						aster_accumulate_support_role(topNegIds[r], regime, asterRegimeCount, r + 1u, aster->supportDim,
						                              topNegLogits[r], finalHiddenRow, dModel,
						                              aster->batchSupportLogitSum,
						                              aster->batchSupportCount,
						                              aster->batchSupportHiddenRawSum,
						                              aster->batchSupportIds,
						                              aster->batchSupportRoleCounts);
						aster_accumulate_support_value_only(regime, r + 1u, aster->supportDim,
						                                    topNegResiduals[r],
						                                    aster->batchSupportResidualSum);
					}
					for (unsigned int i = 0; i < dModel && (regimeFinalRawOff + i) < aster->batchFinalHiddenRawSum.size(); ++i)
						aster->batchFinalHiddenRawSum[regimeFinalRawOff + i] += finalHiddenRow[i];
					aster_sketch_dense_row(finalHiddenRow, dModel, aster->sketchDim,
					                       hiddenSketchSeed, aster->batchFinalHiddenSketchSum, regimeBundleOff);
					for (unsigned int l = 0; l < trackedLayers; ++l)
					{
						const unsigned int blockIdx = aster->trackedBlockIndices[l];
						if (blockIdx >= nLayers)
							continue;
						const float* layerHidden =
						    transformerScratch.hAfterFF.data()
						    + (static_cast<size_t>(blockIdx) * static_cast<size_t>(T) * static_cast<size_t>(dModel))
						    + hOff;
						const float* layerAttn =
						    transformerScratch.attnOut.data()
						    + (static_cast<size_t>(blockIdx) * static_cast<size_t>(T) * static_cast<size_t>(dModel))
						    + hOff;
						const float* layerQ =
						    transformerScratch.Q.data()
						    + (static_cast<size_t>(blockIdx) * static_cast<size_t>(T) * static_cast<size_t>(dModel))
						    + hOff;
						const size_t rawLayerOff = regimeLayerRawBase + static_cast<size_t>(l) * static_cast<size_t>(dModel);
						const size_t layerOff = regimeLayerSketchBase + static_cast<size_t>(l) * static_cast<size_t>(aster->sketchDim);
						const size_t patternOff =
						    (static_cast<size_t>(regime) * static_cast<size_t>(trackedLayers) + l) * static_cast<size_t>(tokenCondDim);
						for (unsigned int i = 0; i < dModel && (rawLayerOff + i) < aster->batchLayerHiddenRawSum.size(); ++i)
							aster->batchLayerHiddenRawSum[rawLayerOff + i] += layerHidden[i];
						for (unsigned int i = 0; i < dModel && (rawLayerOff + i) < aster->batchLayerAttnRawSum.size(); ++i)
							aster->batchLayerAttnRawSum[rawLayerOff + i] += layerAttn[i];
						for (unsigned int i = 0; i < dModel; ++i)
						{
							const unsigned int h = aster_mix_u32(hiddenSketchSeed
							                                     + (blockIdx + 1u) * 0x7f4a7c15U
							                                     + i * 0x9e3779b9U);
							const unsigned int bucket = h % aster->sketchDim;
							const float sign = ((h >> 31) != 0u) ? -1.0f : 1.0f;
							const float hiddenV = layerHidden[i];
							if (hiddenV != 0.0f)
								aster->batchLayerHiddenSketchSum[layerOff + bucket] += sign * hiddenV;
							const float attnV = layerAttn[i];
							if (attnV != 0.0f)
								aster->batchLayerAttnSketchSum[layerOff + bucket] += sign * attnV;
						}
						static const unsigned int kPatternLags[4] = { 1u, 2u, 4u, 8u };
						const unsigned int patternCount = std::min<unsigned int>(tokenCondDim, 4u);
						const float invScale = (dModelKV > 0u) ? (1.0f / sqrtf(static_cast<float>(dModelKV))) : 1.0f;
						for (unsigned int p = 0; p < patternCount && (patternOff + p) < aster->batchLayerPatternSum.size(); ++p)
						{
							float patternValue = 0.0f;
							const unsigned int lag = kPatternLags[p];
							if (t >= lag)
							{
								const size_t prevKOff =
								    (static_cast<size_t>(blockIdx) * static_cast<size_t>(T) + static_cast<size_t>(t - lag))
								    * static_cast<size_t>(dModelKV);
								const float* prevKey = transformerScratch.K.data() + prevKOff;
								double accum = 0.0;
								for (unsigned int i = 0; i < dModelKV; ++i)
									accum += static_cast<double>(layerQ[i]) * static_cast<double>(prevKey[i]);
								patternValue = static_cast<float>(accum) * invScale;
							}
							aster->batchLayerPatternSum[patternOff + p] += patternValue;
						}
						if ((4u + 0u) < tokenCondDim && (patternOff + 4u) < aster->batchLayerPatternSum.size())
							aster->batchLayerPatternSum[patternOff + 4u] += targetRepeatHit;
						if ((4u + 1u) < tokenCondDim && (patternOff + 5u) < aster->batchLayerPatternSum.size())
							aster->batchLayerPatternSum[patternOff + 5u] += targetRepeatCloseness;
						if ((4u + 2u) < tokenCondDim && (patternOff + 6u) < aster->batchLayerPatternSum.size())
							aster->batchLayerPatternSum[patternOff + 6u] += hardNegRepeatHit;
						if ((4u + 3u) < tokenCondDim && (patternOff + 7u) < aster->batchLayerPatternSum.size())
							aster->batchLayerPatternSum[patternOff + 7u] += hardNegRepeatCloseness;
						if (aster->kappaObsDim > 0u && dHead > 0u && nKVHeads > 0u)
						{
							const unsigned int kappaHeads = std::min<unsigned int>(aster->kappaHeads, nHeads);
							const unsigned int kappaLagBuckets = std::min<unsigned int>(aster->kappaLagBuckets, 4u);
							const unsigned int kappaRank = std::max(1u, aster->kappaRank);
							const float invHeadScale = 1.0f / sqrtf(static_cast<float>(dHead));
							const size_t kappaOff =
							    (static_cast<size_t>(regime) * static_cast<size_t>(trackedLayers) + l)
							    * static_cast<size_t>(aster->kappaObsDim);
							for (unsigned int hq = 0; hq < kappaHeads; ++hq)
							{
								const unsigned int kvHead =
								    std::min<unsigned int>((hq * nKVHeads) / std::max(1u, nHeads), nKVHeads - 1u);
								const float* qHead = layerQ + static_cast<size_t>(hq) * static_cast<size_t>(dHead);
								for (unsigned int p = 0; p < kappaLagBuckets; ++p)
								{
									const unsigned int lag = kPatternLags[p];
									if (t < lag)
										continue;
									const size_t prevKVOff =
									    (static_cast<size_t>(blockIdx) * static_cast<size_t>(T) + static_cast<size_t>(t - lag))
									    * static_cast<size_t>(dModelKV)
									    + static_cast<size_t>(kvHead) * static_cast<size_t>(dHead);
									const float* prevKey = transformerScratch.K.data() + prevKVOff;
									const float* prevValue = transformerScratch.V.data() + prevKVOff;
									double qk = 0.0;
									for (unsigned int i = 0; i < dHead; ++i)
										qk += static_cast<double>(qHead[i]) * static_cast<double>(prevKey[i]);
									const float attnScore = static_cast<float>(qk) * invHeadScale;
									const size_t featureBase =
									    kappaOff + (static_cast<size_t>(hq) * static_cast<size_t>(kappaLagBuckets) + p)
									             * static_cast<size_t>(kappaRank);
									for (unsigned int r = 0; r < kappaRank; ++r)
										aster->batchLayerKappaSum[featureBase + r] +=
										    attnScore * aster_segment_mean(prevValue, dHead, kappaRank, r);
								}
							}
						}
					}
					if (regime < aster->batchRegimeTokenCount.size())
						aster->batchRegimeTokenCount[regime] += 1.0f;
					aster->batchTokenCount += 1u;
				}
			}

			// Apply gradient clipping and loss scaling to dLogits in-place
			{
				const size_t dLogitsLen = static_cast<size_t>(T) * static_cast<size_t>(vocabSize);
				for (size_t idx = 0; idx < dLogitsLen; ++idx)
					dLogits[idx] = clip_maybe(dLogits[idx], gradClip) * lossScale;
			}

			// Parallelized tied-embedding backward:
			//   gTokE  += dLogits^T * hPostFinalLN   (weight gradient)
			//   gLmBias += sum_t dLogits[t,:]         (bias gradient)
			//   dH      += dLogits * tokE             (input gradient)
			const LinearWeightView tiedEmbHead = make_linear_weight_view(tt.tokE, tt.tokELowp, tt.lmBias, useLowpWeights, lowpDType);
			linear_backward_accum_maybe_lowp(hPostFinalLN, dLogits.data(), T, dModel, vocabSize,
			    tt.gTokE, tt.gLmBias, tiedEmbHead, dH.data());
			GLADES_ECHO_CPU_OBSERVE(echo_scope_uses_head_matrix(ac, vocabSize, dModel),
			                        tt.echoTokE, dLogits.data(), hPostFinalLN, T, vocabSize, dModel);
		}
		else
		{
			const unsigned int S = scratchOutSize;
			for (unsigned int t = 0; t < T; ++t)
			{
				const int yid = targetIds[t];
				if (padTokenId >= 0 && yid == padTokenId) continue;
				if (yid < 0 || static_cast<unsigned int>(yid) >= vocabSize) continue;
				++validTargetsThisSeq;
				if (t < tokenIds.size())
				{
					const int xid = tokenIds[t];
					if (xid >= 0
					    && static_cast<unsigned int>(xid) < vocabSize
					    && !(padTokenId >= 0 && xid == padTokenId))
						tokenLastSeen[static_cast<unsigned int>(xid)] = static_cast<int>(t);
				}

				const size_t off = static_cast<size_t>(t) * static_cast<size_t>(S);
				for (unsigned int j = 0u; j < S; ++j)
					dLogits[off + j] = transformerScratch.probs[off + j];
				dLogits[off + 0u] -= 1.0f;

				const unsigned int supportNegCount =
				    (aster && aster->supportDim > 0u) ? (aster->supportDim - 1u) : 0u;
				std::vector<unsigned int> topNegIds(supportNegCount, vocabSize);
				std::vector<float> topNegResiduals(supportNegCount, 0.0f);
				std::vector<float> topNegLogits(supportNegCount, 0.0f);

				const size_t hOff = static_cast<size_t>(t) * static_cast<size_t>(dModel);
				const bool haveLowpE = useLowpWeights && (tt.tokELowp.size() == tt.tokE.size()) && !tt.tokELowp.empty();
				unsigned int regime = 0u;
				if (aster)
				{
					regime = aster_transformer_regime_from_logits(&transformerScratch.probs[off],
					                                              &transformerScratch.logits[off],
					                                              S,
					                                              0u,
					                                              -1);
				}
				for (unsigned int j = 0u; j < S; ++j)
				{
					const int vid = transformerScratch.tokenLmSampleIds[off + j];
					if (vid >= 0 && static_cast<unsigned int>(vid) < vocabSize && aster)
					{
						const float rawResidual = dLogits[off + j];
						aster_sketch_sparse_value(static_cast<unsigned int>(vid), rawResidual, aster->sketchDim,
						                          asterSketchSeed, aster->batchResidualSum,
						                          static_cast<size_t>(regime) * static_cast<size_t>(aster->sketchDim));
						aster->batchTouchedIds.push_back(static_cast<unsigned int>(vid));
					}
					if (supportNegCount > 0u && j > 0u && vid >= 0 && static_cast<unsigned int>(vid) < vocabSize)
						aster_insert_top_support(static_cast<unsigned int>(vid), std::max(0.0f, dLogits[off + j]),
						                         transformerScratch.logits[off + j],
						                         topNegIds, topNegResiduals, topNegLogits);
					const float dz = clip_maybe(dLogits[off + j], gradClip) * lossScale;
					if (dz == 0.0f) continue;
					if (vid < 0 || static_cast<unsigned int>(vid) >= vocabSize) continue;
					tt.gLmBias[static_cast<size_t>(vid)] += dz;
					const size_t eOff = static_cast<size_t>(vid) * static_cast<size_t>(dModel);
					for (unsigned int i = 0; i < dModel; ++i)
					{
						tt.gTokE[eOff + i] += dz * hPostFinalLN[hOff + i];
						const float ev = haveLowpE ? glades::transformer_kernels::lowp_to_float(tt.tokELowp[eOff + i], lowpDType) : tt.tokE[eOff + i];
						dH[hOff + i] += dz * ev;
					}
				}
				const int targetLastSeen = tokenLastSeen[static_cast<unsigned int>(yid)];
				const float targetRepeatHit = (targetLastSeen >= 0) ? 1.0f : 0.0f;
				const float targetRepeatCloseness =
				    (targetLastSeen >= 0)
				        ? (1.0f / static_cast<float>((t + 1u) - static_cast<unsigned int>(targetLastSeen)))
				        : 0.0f;
				const unsigned int hardNegId =
				    (!topNegIds.empty() && topNegIds[0] < vocabSize) ? topNegIds[0] : vocabSize;
				const int hardNegLastSeen =
				    (hardNegId < vocabSize) ? tokenLastSeen[hardNegId] : -1;
				const float hardNegRepeatHit = (hardNegLastSeen >= 0) ? 1.0f : 0.0f;
				const float hardNegRepeatCloseness =
				    (hardNegLastSeen >= 0)
				        ? (1.0f / static_cast<float>((t + 1u) - static_cast<unsigned int>(hardNegLastSeen)))
				        : 0.0f;
				if (aster)
				{
					const unsigned int tokenCondDim = std::max(1u, aster->tokenCondDim);
					const float targetLogit = transformerScratch.logits[off + 0u];
					const float hardNegativeLogit =
					    (supportNegCount > 0u && !topNegIds.empty() && topNegIds[0] < vocabSize)
					        ? topNegLogits[0]
					        : targetLogit;
					const float targetMargin = targetLogit - hardNegativeLogit;
					const float marginBaseline =
					    (regime < aster->targetMarginEma.size()) ? aster->targetMarginEma[regime] : 0.75f;
					if (regime < aster->batchTargetMarginSum.size())
						aster->batchTargetMarginSum[regime] += targetMargin;
					if (regime < aster->batchHardNegativeLogitSum.size())
						aster->batchHardNegativeLogitSum[regime] += hardNegativeLogit;
					if (regime < aster->batchBaselineWorseSum.size())
						aster->batchBaselineWorseSum[regime] += (targetMargin < marginBaseline) ? 1.0f : 0.0f;
					const float* finalHiddenRow = &hPostFinalLN[hOff];
					const size_t regimeFinalRawOff = static_cast<size_t>(regime) * static_cast<size_t>(dModel);
					const size_t regimeLayerRawBase =
					    static_cast<size_t>(regime) * static_cast<size_t>(aster->hiddenStackDepth) * static_cast<size_t>(dModel);
					const size_t regimeLayerSketchBase =
					    static_cast<size_t>(regime) * static_cast<size_t>(aster->hiddenStackDepth) * static_cast<size_t>(aster->sketchDim);
					aster_accumulate_support_role(static_cast<unsigned int>(yid), regime, asterRegimeCount, 0u, aster->supportDim,
					                              transformerScratch.logits[off + 0u], finalHiddenRow, dModel,
					                              aster->batchSupportLogitSum,
					                              aster->batchSupportCount,
					                              aster->batchSupportHiddenRawSum,
					                              aster->batchSupportIds,
					                              aster->batchSupportRoleCounts);
					aster_accumulate_support_value_only(regime, 0u, aster->supportDim,
					                                    dLogits[off + 0u],
					                                    aster->batchSupportResidualSum);
					for (unsigned int r = 0; r < supportNegCount; ++r)
					{
						if (topNegIds[r] >= vocabSize)
							continue;
						aster_accumulate_support_role(topNegIds[r], regime, asterRegimeCount, r + 1u, aster->supportDim,
						                              topNegLogits[r], finalHiddenRow, dModel,
						                              aster->batchSupportLogitSum,
						                              aster->batchSupportCount,
						                              aster->batchSupportHiddenRawSum,
						                              aster->batchSupportIds,
						                              aster->batchSupportRoleCounts);
						aster_accumulate_support_value_only(regime, r + 1u, aster->supportDim,
						                                    topNegResiduals[r],
						                                    aster->batchSupportResidualSum);
					}
					for (unsigned int i = 0; i < dModel && (regimeFinalRawOff + i) < aster->batchFinalHiddenRawSum.size(); ++i)
						aster->batchFinalHiddenRawSum[regimeFinalRawOff + i] += finalHiddenRow[i];
					aster_sketch_dense_row(finalHiddenRow, dModel, aster->sketchDim,
					                       hiddenSketchSeed, aster->batchFinalHiddenSketchSum,
					                       static_cast<size_t>(regime) * static_cast<size_t>(aster->sketchDim));
					const unsigned int trackedLayers =
					    std::min<unsigned int>(aster->hiddenStackDepth,
					                           static_cast<unsigned int>(aster->trackedBlockIndices.size()));
					for (unsigned int l = 0; l < trackedLayers; ++l)
					{
						const unsigned int blockIdx = aster->trackedBlockIndices[l];
						if (blockIdx >= nLayers)
							continue;
						const float* layerHidden =
						    transformerScratch.hAfterFF.data()
						    + (static_cast<size_t>(blockIdx) * static_cast<size_t>(T) * static_cast<size_t>(dModel))
						    + hOff;
						const float* layerAttn =
						    transformerScratch.attnOut.data()
						    + (static_cast<size_t>(blockIdx) * static_cast<size_t>(T) * static_cast<size_t>(dModel))
						    + hOff;
						const float* layerQ =
						    transformerScratch.Q.data()
						    + (static_cast<size_t>(blockIdx) * static_cast<size_t>(T) * static_cast<size_t>(dModel))
						    + hOff;
						const size_t rawLayerOff = regimeLayerRawBase + static_cast<size_t>(l) * static_cast<size_t>(dModel);
						const size_t layerOff = regimeLayerSketchBase + static_cast<size_t>(l) * static_cast<size_t>(aster->sketchDim);
						const size_t patternOff =
						    (static_cast<size_t>(regime) * static_cast<size_t>(trackedLayers) + l) * static_cast<size_t>(tokenCondDim);
						for (unsigned int i = 0; i < dModel && (rawLayerOff + i) < aster->batchLayerHiddenRawSum.size(); ++i)
							aster->batchLayerHiddenRawSum[rawLayerOff + i] += layerHidden[i];
						for (unsigned int i = 0; i < dModel && (rawLayerOff + i) < aster->batchLayerAttnRawSum.size(); ++i)
							aster->batchLayerAttnRawSum[rawLayerOff + i] += layerAttn[i];
						for (unsigned int i = 0; i < dModel; ++i)
						{
							const unsigned int h = aster_mix_u32(hiddenSketchSeed
							                                     + (blockIdx + 1u) * 0x7f4a7c15U
							                                     + i * 0x9e3779b9U);
							const unsigned int bucket = h % aster->sketchDim;
							const float sign = ((h >> 31) != 0u) ? -1.0f : 1.0f;
							const float hiddenV = layerHidden[i];
							if (hiddenV != 0.0f)
								aster->batchLayerHiddenSketchSum[layerOff + bucket] += sign * hiddenV;
							const float attnV = layerAttn[i];
							if (attnV != 0.0f)
								aster->batchLayerAttnSketchSum[layerOff + bucket] += sign * attnV;
						}
						static const unsigned int kPatternLags[4] = { 1u, 2u, 4u, 8u };
						const unsigned int patternCount = std::min<unsigned int>(tokenCondDim, 4u);
						const float invScale = (dModelKV > 0u) ? (1.0f / sqrtf(static_cast<float>(dModelKV))) : 1.0f;
						for (unsigned int p = 0; p < patternCount && (patternOff + p) < aster->batchLayerPatternSum.size(); ++p)
						{
							float patternValue = 0.0f;
							const unsigned int lag = kPatternLags[p];
							if (t >= lag)
							{
								const size_t prevKOff =
								    (static_cast<size_t>(blockIdx) * static_cast<size_t>(T) + static_cast<size_t>(t - lag))
								    * static_cast<size_t>(dModelKV);
								const float* prevKey = transformerScratch.K.data() + prevKOff;
								double accum = 0.0;
								for (unsigned int i = 0; i < dModelKV; ++i)
									accum += static_cast<double>(layerQ[i]) * static_cast<double>(prevKey[i]);
								patternValue = static_cast<float>(accum) * invScale;
							}
							aster->batchLayerPatternSum[patternOff + p] += patternValue;
						}
						if ((4u + 0u) < tokenCondDim && (patternOff + 4u) < aster->batchLayerPatternSum.size())
							aster->batchLayerPatternSum[patternOff + 4u] += targetRepeatHit;
						if ((4u + 1u) < tokenCondDim && (patternOff + 5u) < aster->batchLayerPatternSum.size())
							aster->batchLayerPatternSum[patternOff + 5u] += targetRepeatCloseness;
						if ((4u + 2u) < tokenCondDim && (patternOff + 6u) < aster->batchLayerPatternSum.size())
							aster->batchLayerPatternSum[patternOff + 6u] += hardNegRepeatHit;
						if ((4u + 3u) < tokenCondDim && (patternOff + 7u) < aster->batchLayerPatternSum.size())
							aster->batchLayerPatternSum[patternOff + 7u] += hardNegRepeatCloseness;
						if (aster->kappaObsDim > 0u && dHead > 0u && nKVHeads > 0u)
						{
							const unsigned int kappaHeads = std::min<unsigned int>(aster->kappaHeads, nHeads);
							const unsigned int kappaLagBuckets = std::min<unsigned int>(aster->kappaLagBuckets, 4u);
							const unsigned int kappaRank = std::max(1u, aster->kappaRank);
							const float invHeadScale = 1.0f / sqrtf(static_cast<float>(dHead));
							const size_t kappaOff =
							    (static_cast<size_t>(regime) * static_cast<size_t>(trackedLayers) + l)
							    * static_cast<size_t>(aster->kappaObsDim);
							for (unsigned int hq = 0; hq < kappaHeads; ++hq)
							{
								const unsigned int kvHead =
								    std::min<unsigned int>((hq * nKVHeads) / std::max(1u, nHeads), nKVHeads - 1u);
								const float* qHead = layerQ + static_cast<size_t>(hq) * static_cast<size_t>(dHead);
								for (unsigned int p = 0; p < kappaLagBuckets; ++p)
								{
									const unsigned int lag = kPatternLags[p];
									if (t < lag)
										continue;
									const size_t prevKVOff =
									    (static_cast<size_t>(blockIdx) * static_cast<size_t>(T) + static_cast<size_t>(t - lag))
									    * static_cast<size_t>(dModelKV)
									    + static_cast<size_t>(kvHead) * static_cast<size_t>(dHead);
									const float* prevKey = transformerScratch.K.data() + prevKVOff;
									const float* prevValue = transformerScratch.V.data() + prevKVOff;
									double qk = 0.0;
									for (unsigned int i = 0; i < dHead; ++i)
										qk += static_cast<double>(qHead[i]) * static_cast<double>(prevKey[i]);
									const float attnScore = static_cast<float>(qk) * invHeadScale;
									const size_t featureBase =
									    kappaOff + (static_cast<size_t>(hq) * static_cast<size_t>(kappaLagBuckets) + p)
									             * static_cast<size_t>(kappaRank);
									for (unsigned int r = 0; r < kappaRank; ++r)
										aster->batchLayerKappaSum[featureBase + r] +=
										    attnScore * aster_segment_mean(prevValue, dHead, kappaRank, r);
								}
							}
						}
					}
					if (regime < aster->batchRegimeTokenCount.size())
						aster->batchRegimeTokenCount[regime] += 1.0f;
					aster->batchTokenCount += 1u;
				}
			}
		}
		if (helm)
		{
			const unsigned int trackedLayers =
			    std::min<unsigned int>(helm->hiddenStackDepth,
			                           static_cast<unsigned int>(helm->trackedBlockIndices.size()));
			for (unsigned int t = 0; t < T; ++t)
			{
				const int yid = targetIds[t];
				if (padTokenId >= 0 && yid == padTokenId) continue;
				if (yid < 0 || static_cast<unsigned int>(yid) >= vocabSize) continue;
				const size_t hOff = static_cast<size_t>(t) * static_cast<size_t>(dModel);
				for (unsigned int l = 0; l < trackedLayers; ++l)
				{
					const unsigned int blockIdx = helm->trackedBlockIndices[l];
					if (blockIdx >= nLayers)
						continue;
					const float* layerHidden =
					    transformerScratch.hAfterFF.data()
					    + (static_cast<size_t>(blockIdx) * static_cast<size_t>(T) * static_cast<size_t>(dModel))
					    + hOff;
					const size_t layerOff = static_cast<size_t>(l) * static_cast<size_t>(dModel);
					for (unsigned int i = 0; i < dModel && (layerOff + i) < helm->batchHiddenSum.size(); ++i)
					{
						const float h = layerHidden[i];
						helm->batchHiddenSum[layerOff + i] += h;
						helm->batchHiddenSqSum[layerOff + i] += h * h;
					}
				}
				for (unsigned int i = 0; i < dModel && i < helm->batchResidualSum.size(); ++i)
				{
					const float r = dH[hOff + i];
					helm->batchResidualSum[i] += r;
					helm->batchResidualSqSum[i] += r * r;
				}
				helm->batchTokenCount += 1u;
			}
		}
		timeStepsInBatch += validTargetsThisSeq;
	}
	else
	{
		for (unsigned int t = 0; t < T; ++t)
		{
			const float* expRow = NULL;
			unsigned int expSize = 0u;
			di->getTrainSequenceExpectedRowView(s, t, expRow, expSize);
			const size_t off = static_cast<size_t>(t) * static_cast<size_t>(outSize);
			for (unsigned int k = 0; k < outSize; ++k)
			{
				const float expv = (expRow && k < expSize) ? expRow[k] : 0.0f;
				const float pred = transformerScratch.probs[off + k];
				float d = 0.0f;
				const bool useSoftmax = ((costFx == GMath::CLASSIFICATION) || (costFx == GMath::KL)) && (outSize > 1u);
				if (useSoftmax)
					d = pred - expv;
				else if ((costFx == GMath::CLASSIFICATION) && (outSize == 1u))
					d = pred - expv;
				else
					d = GMath::costErrDer(expv, pred, costFx);
				dLogits[off + k] = clip_maybe(d, gradClip) * lossScale;
			}
		}

		const LinearWeightView outProj = make_linear_weight_view(tt.WOut, tt.WOutLowp, tt.bOut, useLowpWeights, lowpDType);
		linear_backward_accum_maybe_lowp(hPostFinalLN, dLogits.data(), T, dModel, outSize,
		                                 tt.gWOut, tt.gBOut, outProj, dH.data());
		GLADES_ECHO_CPU_OBSERVE(echo_scope_uses_head_matrix(ac, outSize, dModel),
		                        tt.echoWOut, dLogits.data(), hPostFinalLN, T, outSize, dModel);
		timeStepsInBatch += T;
	}

	// Backprop Final LayerNorm
	{
		std::vector<float, glades::AlignedAllocator<float, 64> >& dHPreFinalLN = transformerScratch.dH2;
		if (dHPreFinalLN.size() != dH.size()) dHPreFinalLN.resize(dH.size());
		std::fill(dHPreFinalLN.begin(), dHPreFinalLN.end(), 0.0f);
		if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
		{
			glades::transformer_kernels::rmsnorm_backward_rows_accum(hFinal, dH.data(), T, dModel, tt.lnFinalGamma,
			    transformerScratch.lnFinalInvStd.data(), dHPreFinalLN.data(), tt.gLnFinalGamma, tt.gLnFinalBeta);
		}
		else
		{
			glades::transformer_kernels::layernorm_backward_rows_accum(hFinal, dH.data(), T, dModel, tt.lnFinalGamma,
			    transformerScratch.lnFinalMean.data(), transformerScratch.lnFinalInvStd.data(),
			    dHPreFinalLN.data(), tt.gLnFinalGamma, tt.gLnFinalBeta);
		}
		std::copy(dHPreFinalLN.begin(), dHPreFinalLN.end(), dH.begin());
	}

	// RoPE precompute for backward
	const bool useRope = (posEnc == static_cast<int>(glades::TransformerRunConfig::POSENC_ROPE));
	unsigned int ropeDim = dHead;
	if (ropeDimOverride > 0)
	{
		const unsigned int rd = static_cast<unsigned int>(ropeDimOverride);
		ropeDim = (rd < ropeDim) ? rd : ropeDim;
	}
	if ((ropeDim % 2u) != 0u) ropeDim -= 1u;
	const std::vector<double>* ropeInvFreq = NULL;
	if (useRope && ropeDim >= 2u)
	{
		transformerPosEncCache.ensureRope(ropeDim, ropeTheta);
		ropeInvFreq = &transformerPosEncCache.ropeInvFreq;
	}
	const unsigned int groupSize = (nKVHeads > 0u) ? (nHeads / nKVHeads) : 0u;

	// Backprop through blocks (reverse)
	for (int li = static_cast<int>(nLayers) - 1; li >= 0; --li)
	{
		TensorTransformerState::Block& b = tt.blocks[static_cast<size_t>(li)];
		const float* hIn = (li == 0) ? transformerScratch.h.data()
		                             : (transformerScratch.hAfterFF.data() + (static_cast<size_t>(li - 1) * static_cast<size_t>(T) * static_cast<size_t>(dModel)));

		const float* hAfterAttn = transformerScratch.hAfterAttn.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
		const float* x1 = transformerScratch.x1.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
		const float* x2 = transformerScratch.x2.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
		const float* ff1 = transformerScratch.ff1.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(ff1Width));
		const float* ff1Act = transformerScratch.ff1Act.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dFF));

		std::vector<float, glades::AlignedAllocator<float, 64> >& dHAfterAttn = transformerScratch.dH2;
		if (dHAfterAttn.size() != dH.size()) dHAfterAttn.resize(dH.size());
		std::copy(dH.begin(), dH.end(), dHAfterAttn.begin());

		// FFN residual dropout backward
		{
			const float resDropRate = trainingConfig.transformer.residualDropoutRate;
			if (resDropRate > 0.0f)
			{
				const size_t layerOff = static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel);
				const size_t n = static_cast<size_t>(T) * static_cast<size_t>(dModel);
				const unsigned char* mask = transformerScratch.dropoutMaskResFF.empty() ? NULL : &transformerScratch.dropoutMaskResFF[layerOff];
				if (mask)
				{
					const float scale = 1.0f / (1.0f - resDropRate);
					for (size_t i = 0; i < n; ++i)
						dH[i] *= mask[i] ? scale : 0.0f;
				}
			}
		}

		// FFN backward
		std::vector<float, glades::AlignedAllocator<float, 64> >& dFF1Act = transformerScratch.dFF1Act;
		if (dFF1Act.size() != (static_cast<size_t>(T) * static_cast<size_t>(dFF)))
			dFF1Act.resize(static_cast<size_t>(T) * static_cast<size_t>(dFF));
		const LinearWeightView ff2Proj = make_linear_weight_view(b.W2, b.W2Lowp, b.b2, useLowpWeights, lowpDType);
		linear_backward_accum_maybe_lowp(ff1Act, dH.data(), T, dFF, dModel, b.gW2, b.gB2, ff2Proj, dFF1Act.data());
		GLADES_ECHO_CPU_OBSERVE(echo_scope_uses_decoder_matrix(ac, static_cast<unsigned int>(li), nLayers, dModel, dFF),
		                        b.echoW2, dH.data(), ff1Act, T, dModel, dFF);

		std::vector<float, glades::AlignedAllocator<float, 64> >& dX2 = transformerScratch.dX2;
		if (dX2.size() != (static_cast<size_t>(T) * static_cast<size_t>(dModel)))
			dX2.resize(static_cast<size_t>(T) * static_cast<size_t>(dModel));
		if (ffnKind == static_cast<int>(glades::TransformerRunConfig::FFN_SWIGLU))
		{
			std::vector<float, glades::AlignedAllocator<float, 64> >& dFF1Cat = transformerScratch.dFF1Cat;
			if (dFF1Cat.size() != (static_cast<size_t>(T) * static_cast<size_t>(ff1Width)))
				dFF1Cat.resize(static_cast<size_t>(T) * static_cast<size_t>(ff1Width));
			std::fill(dFF1Cat.begin(), dFF1Cat.end(), 0.0f);
			for (unsigned int t = 0; t < T; ++t)
			{
				const size_t preOff = static_cast<size_t>(t) * static_cast<size_t>(ff1Width);
				const size_t outOff = static_cast<size_t>(t) * static_cast<size_t>(dFF);
				for (unsigned int i = 0; i < dFF; ++i)
				{
					const float gatePre = ff1[preOff + i];
					const float upPre = ff1[preOff + static_cast<size_t>(dFF) + i];
					const float siluVal = glades::transformer_ops::silu(gatePre);
					const float dOut = dFF1Act[outOff + i];
					dFF1Cat[preOff + i] = dOut * upPre * glades::transformer_ops::silu_deriv(gatePre);
					dFF1Cat[preOff + static_cast<size_t>(dFF) + i] = dOut * siluVal;
				}
			}
			const LinearWeightView ff1Proj = make_linear_weight_view(b.W1, b.W1Lowp, b.b1, useLowpWeights, lowpDType);
			linear_backward_accum_maybe_lowp(x2, dFF1Cat.data(), T, dModel, ff1Width, b.gW1, b.gB1, ff1Proj, dX2.data());
			GLADES_ECHO_CPU_OBSERVE(echo_scope_uses_decoder_matrix(ac, static_cast<unsigned int>(li), nLayers, ff1Width, dModel),
			                        b.echoW1, dFF1Cat.data(), x2, T, ff1Width, dModel);
		}
		else
		{
			const size_t actLen = dFF1Act.size();
			if (ffnAct == static_cast<int>(glades::TransformerRunConfig::FFN_GELU))
				glades::transformer_kernels::gelu_backward_buf(ff1, dFF1Act.data(), actLen);
			else
				for (size_t i = 0; i < actLen; ++i)
					dFF1Act[i] *= glades::transformer_ops::relu_deriv_from_y(ff1Act[i]);
			const LinearWeightView ff1Proj = make_linear_weight_view(b.W1, b.W1Lowp, b.b1, useLowpWeights, lowpDType);
			linear_backward_accum_maybe_lowp(x2, dFF1Act.data(), T, dModel, dFF, b.gW1, b.gB1, ff1Proj, dX2.data());
			GLADES_ECHO_CPU_OBSERVE(echo_scope_uses_decoder_matrix(ac, static_cast<unsigned int>(li), nLayers, dFF, dModel),
			                        b.echoW1, dFF1Act.data(), x2, T, dFF, dModel);
		}

		// LN2 backward
		std::vector<float, glades::AlignedAllocator<float, 64> >& dHAfterAttnFromLN = transformerScratch.dHAfterAttnFromLN;
		if (dHAfterAttnFromLN.size() != (static_cast<size_t>(T) * static_cast<size_t>(dModel)))
			dHAfterAttnFromLN.resize(static_cast<size_t>(T) * static_cast<size_t>(dModel));
		std::fill(dHAfterAttnFromLN.begin(), dHAfterAttnFromLN.end(), 0.0f);
		const float* ln2Mean = transformerScratch.ln2Mean.data() + (static_cast<size_t>(li) * static_cast<size_t>(T));
		const float* ln2InvStd = transformerScratch.ln2InvStd.data() + (static_cast<size_t>(li) * static_cast<size_t>(T));
		if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
		{
			glades::transformer_kernels::rmsnorm_backward_rows_accum(hAfterAttn, dX2.data(), T, dModel, b.ln2Gamma, ln2InvStd,
			                                                        dHAfterAttnFromLN.data(), b.gLn2Gamma, b.gLn2Beta);
		}
		else
		{
			glades::transformer_kernels::layernorm_backward_rows_accum(hAfterAttn, dX2.data(), T, dModel, b.ln2Gamma, ln2Mean, ln2InvStd,
			                                                          dHAfterAttnFromLN.data(), b.gLn2Gamma, b.gLn2Beta);
		}
		for (size_t i = 0; i < dHAfterAttn.size(); ++i)
			dHAfterAttn[i] += dHAfterAttnFromLN[i];

		std::copy(dHAfterAttn.begin(), dHAfterAttn.end(), dH.begin());

		// Attention residual dropout backward
		{
			const float resDropRate = trainingConfig.transformer.residualDropoutRate;
			if (resDropRate > 0.0f)
			{
				const size_t layerOff = static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel);
				const size_t n = static_cast<size_t>(T) * static_cast<size_t>(dModel);
				const unsigned char* mask = transformerScratch.dropoutMaskResAttn.empty() ? NULL : &transformerScratch.dropoutMaskResAttn[layerOff];
				if (mask)
				{
					const float scale = 1.0f / (1.0f - resDropRate);
					for (size_t i = 0; i < n; ++i)
						dHAfterAttn[i] *= mask[i] ? scale : 0.0f;
				}
			}
		}

		const float* attnConcat = transformerScratch.attnConcat.data() +
		                          (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel));

		// Backprop Wo
		std::vector<float, glades::AlignedAllocator<float, 64> >& dAttnConcat = transformerScratch.dAttnConcat;
		if (dAttnConcat.size() != (static_cast<size_t>(T) * static_cast<size_t>(dModel)))
			dAttnConcat.resize(static_cast<size_t>(T) * static_cast<size_t>(dModel));
		const LinearWeightView oProj = make_linear_weight_view(b.Wo, b.WoLowp, b.bo, useLowpWeights, lowpDType);
		linear_backward_accum_maybe_lowp(attnConcat, dHAfterAttn.data(), T, dModel, dModel,
		                                 b.gWo, b.gBo, oProj, dAttnConcat.data());
		GLADES_ECHO_CPU_OBSERVE(echo_scope_uses_decoder_matrix(ac, static_cast<unsigned int>(li), nLayers, dModel, dModel),
		                        b.echoWo, dHAfterAttn.data(), attnConcat, T, dModel, dModel);

		const float* Vfull = transformerScratch.V.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModelKV));
		const float* Qfull = transformerScratch.Q.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModel));
		const float* Kfull = transformerScratch.K.data() + (static_cast<size_t>(li) * static_cast<size_t>(T) * static_cast<size_t>(dModelKV));
		std::vector<float, glades::AlignedAllocator<float, 64> >& dQfull = transformerScratch.dQfull;
		std::vector<float, glades::AlignedAllocator<float, 64> >& dKfull = transformerScratch.dKfull;
		std::vector<float, glades::AlignedAllocator<float, 64> >& dVfull = transformerScratch.dVfull;
		if (dQfull.size() != (static_cast<size_t>(T) * static_cast<size_t>(dModel)))
			dQfull.resize(static_cast<size_t>(T) * static_cast<size_t>(dModel));
		if (dKfull.size() != (static_cast<size_t>(T) * static_cast<size_t>(dModelKV)))
			dKfull.resize(static_cast<size_t>(T) * static_cast<size_t>(dModelKV));
		if (dVfull.size() != (static_cast<size_t>(T) * static_cast<size_t>(dModelKV)))
			dVfull.resize(static_cast<size_t>(T) * static_cast<size_t>(dModelKV));
		std::fill(dQfull.begin(), dQfull.end(), 0.0f);
		std::fill(dKfull.begin(), dKfull.end(), 0.0f);
		std::fill(dVfull.begin(), dVfull.end(), 0.0f);

		// Attention backward
		{
			const bool attnWorthParallel = (static_cast<unsigned long long>(T) * T * dHead >= 32768ULL);
			glades::ThreadPool& pool = glades::ThreadPool::instance();
			if (attnWorthParallel && nHeads > 1u && pool.numThreads() > 1u)
			{
				const unsigned int nThreads = pool.numThreads();
				unsigned int nChunksPerHead = 1u;
				if (nHeads < nThreads && T >= 512u)
				{
					nChunksPerHead = (nThreads + nHeads - 1u) / nHeads;
					if (nChunksPerHead > 4u) nChunksPerHead = 4u;
				}

				const unsigned int sinkCount_bw = trainingConfig.transformer.attnSinkCount > 0
				    ? static_cast<unsigned int>(trainingConfig.transformer.attnSinkCount) : 0u;
				const unsigned int windowSize_bw = trainingConfig.transformer.localAttnWindow > 0
				    ? static_cast<unsigned int>(trainingConfig.transformer.localAttnWindow) : 0u;
				if (nChunksPerHead <= 1u)
				{
					AttnBwdCtx actx;
					actx.Q = Qfull; actx.K = Kfull; actx.V = Vfull;
					actx.dO = dAttnConcat.data();
					actx.dQ = dQfull.data(); actx.dK = dKfull.data(); actx.dV = dVfull.data();
					actx.dModel = dModel; actx.dModelKV = dModelKV;
					actx.dHead = dHead; actx.nHeads = nHeads; actx.nKVHeads = nKVHeads;
					actx.T = T; actx.groupSize = groupSize;
					actx.causal = causal; actx.keyAllowed = NULL;
					actx.nChunksPerHead = 1u; actx.totalItems = nKVHeads;
					actx.dKVscratch = NULL;
					actx.sinkCount = sinkCount_bw; actx.windowSize = windowSize_bw;
					pool.parallel_for(nKVHeads, attn_bwd_body, &actx);
				}
				else
				{
					const unsigned int totalItems = nHeads * nChunksPerHead;
					const size_t scratchPerItem = static_cast<size_t>(T) * dHead * 2u;
					const size_t totalScratch = static_cast<size_t>(totalItems) * scratchPerItem;
					if (transformerScratch.dKVscratch.size() < totalScratch)
						transformerScratch.dKVscratch.resize(totalScratch);
					std::fill(transformerScratch.dKVscratch.begin(), transformerScratch.dKVscratch.begin() + totalScratch, 0.0f);

					AttnBwdCtx actx;
					actx.Q = Qfull; actx.K = Kfull; actx.V = Vfull;
					actx.dO = dAttnConcat.data();
					actx.dQ = dQfull.data(); actx.dK = dKfull.data(); actx.dV = dVfull.data();
					actx.dModel = dModel; actx.dModelKV = dModelKV;
					actx.dHead = dHead; actx.nHeads = nHeads; actx.nKVHeads = nKVHeads;
					actx.T = T; actx.groupSize = groupSize;
					actx.causal = causal; actx.keyAllowed = NULL;
					actx.nChunksPerHead = nChunksPerHead; actx.totalItems = totalItems;
					actx.dKVscratch = &transformerScratch.dKVscratch[0];
					actx.sinkCount = sinkCount_bw; actx.windowSize = windowSize_bw;
					pool.parallel_for(totalItems, attn_bwd_body, &actx);

					AttnBwdReduceCtx rctx;
					rctx.dKVscratch = &transformerScratch.dKVscratch[0];
					rctx.dK = dKfull.data(); rctx.dV = dVfull.data();
					rctx.dHead = dHead; rctx.dModelKV = dModelKV; rctx.T = T;
					rctx.nHeads = nHeads; rctx.nKVHeads = nKVHeads;
					rctx.groupSize = groupSize; rctx.nChunksPerHead = nChunksPerHead;
					pool.parallel_for(nKVHeads, attn_bwd_reduce_body, &rctx);
				}
			}
			else
			{
				const unsigned int sinkCount_fb = trainingConfig.transformer.attnSinkCount > 0
				    ? static_cast<unsigned int>(trainingConfig.transformer.attnSinkCount) : 0u;
				const unsigned int windowSize_fb = trainingConfig.transformer.localAttnWindow > 0
				    ? static_cast<unsigned int>(trainingConfig.transformer.localAttnWindow) : 0u;
				for (unsigned int h = 0; h < nHeads; ++h)
				{
					const unsigned int kvHead = (nKVHeads == nHeads) ? h : (groupSize > 0u ? (h / groupSize) : 0u);
					glades::transformer_ops::scaled_dot_product_attention_backward_recompute_flash_strided_sw(
					    Qfull + static_cast<size_t>(h) * static_cast<size_t>(dHead), dModel,
					    Kfull + static_cast<size_t>(kvHead) * static_cast<size_t>(dHead), dModelKV,
					    Vfull + static_cast<size_t>(kvHead) * static_cast<size_t>(dHead), dModelKV,
					    dAttnConcat.data() + static_cast<size_t>(h) * static_cast<size_t>(dHead), dModel,
					    T, dHead, dHead, causal,
					    sinkCount_fb, windowSize_fb,
					    dQfull.data() + static_cast<size_t>(h) * static_cast<size_t>(dHead), dModel,
					    dKfull.data() + static_cast<size_t>(kvHead) * static_cast<size_t>(dHead), dModelKV,
					    dVfull.data() + static_cast<size_t>(kvHead) * static_cast<size_t>(dHead), dModelKV,
					    NULL);
				}
			}
		}

		// RoPE backward
		if (useRope && ropeInvFreq)
		{
			const bool ropeWorthParallel = (static_cast<unsigned long long>(T) * ropeDim >= 4096ULL);
			glades::ThreadPool& pool = glades::ThreadPool::instance();
			if (ropeWorthParallel && pool.numThreads() > 1u)
			{
				if (nHeads > 1u)
				{
					RopeFwdCtx rctx;
					rctx.buf = dQfull.data(); rctx.T = T; rctx.rowStride = dModel; rctx.dHead = dHead;
					rctx.ropeDim = ropeDim; rctx.invFreq = make_double_buffer_view(*ropeInvFreq); rctx.inverse = true;
					pool.parallel_for(nHeads, rope_body, &rctx);
				}
				else
				{
					glades::transformer_kernels::rope_apply_inplace_strided(dQfull.data(), T, dModel, dHead, ropeDim, *ropeInvFreq, true);
				}
				if (nKVHeads > 1u)
				{
					RopeFwdCtx rctx;
					rctx.buf = dKfull.data(); rctx.T = T; rctx.rowStride = dModelKV; rctx.dHead = dHead;
					rctx.ropeDim = ropeDim; rctx.invFreq = make_double_buffer_view(*ropeInvFreq); rctx.inverse = true;
					pool.parallel_for(nKVHeads, rope_body, &rctx);
				}
				else
				{
					for (unsigned int hk = 0; hk < nKVHeads; ++hk)
						glades::transformer_kernels::rope_apply_inplace_strided(
						    dKfull.data() + static_cast<size_t>(hk) * static_cast<size_t>(dHead), T, dModelKV, dHead, ropeDim, *ropeInvFreq, true);
				}
			}
			else
			{
				for (unsigned int h = 0; h < nHeads; ++h)
					glades::transformer_kernels::rope_apply_inplace_strided(
					    dQfull.data() + static_cast<size_t>(h) * static_cast<size_t>(dHead), T, dModel, dHead, ropeDim, *ropeInvFreq, true);
				for (unsigned int hk = 0; hk < nKVHeads; ++hk)
					glades::transformer_kernels::rope_apply_inplace_strided(
					    dKfull.data() + static_cast<size_t>(hk) * static_cast<size_t>(dHead), T, dModelKV, dHead, ropeDim, *ropeInvFreq, true);
			}
		}

		// QKV projection backward
		std::vector<float, glades::AlignedAllocator<float, 64> >& dX1 = transformerScratch.dX1;
		std::vector<float, glades::AlignedAllocator<float, 64> >& dXtmp = transformerScratch.dXtmp;
		if (dX1.size() != (static_cast<size_t>(T) * static_cast<size_t>(dModel)))
			dX1.resize(static_cast<size_t>(T) * static_cast<size_t>(dModel));
		if (dXtmp.size() != dX1.size()) dXtmp.resize(dX1.size());
		std::fill(dX1.begin(), dX1.end(), 0.0f);
		{
			const LinearWeightView qProj = make_linear_weight_view(b.Wq, b.WqLowp, b.bq, useLowpWeights, lowpDType);
			linear_backward_accum_maybe_lowp(x1, dQfull.data(), T, dModel, dModel, b.gWq, b.gBq, qProj, dXtmp.data());
			GLADES_ECHO_CPU_OBSERVE(echo_scope_uses_decoder_matrix(ac, static_cast<unsigned int>(li), nLayers, dModel, dModel),
			                        b.echoWq, dQfull.data(), x1, T, dModel, dModel);
			for (size_t i = 0; i < dX1.size(); ++i) dX1[i] += dXtmp[i];
		}
		{
			const LinearWeightView kProj = make_linear_weight_view(b.Wk, b.WkLowp, b.bk, useLowpWeights, lowpDType);
			linear_backward_accum_maybe_lowp(x1, dKfull.data(), T, dModel, dModelKV, b.gWk, b.gBk, kProj, dXtmp.data());
			GLADES_ECHO_CPU_OBSERVE(echo_scope_uses_decoder_matrix(ac, static_cast<unsigned int>(li), nLayers, dModelKV, dModel),
			                        b.echoWk, dKfull.data(), x1, T, dModelKV, dModel);
			for (size_t i = 0; i < dX1.size(); ++i) dX1[i] += dXtmp[i];
		}
		{
			const LinearWeightView vProj = make_linear_weight_view(b.Wv, b.WvLowp, b.bv, useLowpWeights, lowpDType);
			linear_backward_accum_maybe_lowp(x1, dVfull.data(), T, dModel, dModelKV, b.gWv, b.gBv, vProj, dXtmp.data());
			GLADES_ECHO_CPU_OBSERVE(echo_scope_uses_decoder_matrix(ac, static_cast<unsigned int>(li), nLayers, dModelKV, dModel),
			                        b.echoWv, dVfull.data(), x1, T, dModelKV, dModel);
			for (size_t i = 0; i < dX1.size(); ++i) dX1[i] += dXtmp[i];
		}

		// LN1 backward
		std::vector<float, glades::AlignedAllocator<float, 64> >& dHInFromLN = transformerScratch.dHInFromLN;
		if (dHInFromLN.size() != (static_cast<size_t>(T) * static_cast<size_t>(dModel)))
			dHInFromLN.resize(static_cast<size_t>(T) * static_cast<size_t>(dModel));
		std::fill(dHInFromLN.begin(), dHInFromLN.end(), 0.0f);
		const float* ln1Mean = transformerScratch.ln1Mean.data() + (static_cast<size_t>(li) * static_cast<size_t>(T));
		const float* ln1InvStd = transformerScratch.ln1InvStd.data() + (static_cast<size_t>(li) * static_cast<size_t>(T));
		if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
		{
			glades::transformer_kernels::rmsnorm_backward_rows_accum(hIn, dX1.data(), T, dModel, b.ln1Gamma, ln1InvStd,
			                                                        dHInFromLN.data(), b.gLn1Gamma, b.gLn1Beta);
		}
		else
		{
			glades::transformer_kernels::layernorm_backward_rows_accum(hIn, dX1.data(), T, dModel, b.ln1Gamma, ln1Mean, ln1InvStd,
			                                                          dHInFromLN.data(), b.gLn1Gamma, b.gLn1Beta);
		}

		for (size_t i = 0; i < dH.size(); ++i)
			dH[i] += dHInFromLN[i];
	} // layers

	// Embedding dropout backward
	{
		const float embDropRate = trainingConfig.transformer.embeddingDropoutRate;
		if (embDropRate > 0.0f)
		{
			const size_t n = static_cast<size_t>(T) * static_cast<size_t>(dModel);
			const unsigned char* mask = transformerScratch.dropoutMaskEmb.empty() ? NULL : &transformerScratch.dropoutMaskEmb[0];
			if (mask)
			{
				const float scale = 1.0f / (1.0f - embDropRate);
				for (size_t i = 0; i < n; ++i)
					dH[i] *= mask[i] ? scale : 0.0f;
			}
		}
	}

	// Backprop input projection
	if (tokenLM)
	{
		for (unsigned int t = 0; t < T; ++t)
		{
			const int tid = tokenIds[t];
			const size_t eOff = static_cast<size_t>(tid) * static_cast<size_t>(dModel);
			const size_t hOff = static_cast<size_t>(t) * static_cast<size_t>(dModel);
			for (unsigned int i = 0; i < dModel; ++i)
				tt.gTokE[eOff + i] += dH[hOff + i];
		}
	}
	else
	{
		std::vector<float, glades::AlignedAllocator<float, 64> >& dX = transformerScratch.dInput;
		if (dX.size() != (static_cast<size_t>(T) * static_cast<size_t>(inputSize)))
			dX.resize(static_cast<size_t>(T) * static_cast<size_t>(inputSize));
		const LinearWeightView inputProj = make_linear_weight_view(tt.WIn, tt.WInLowp, tt.bIn, useLowpWeights, lowpDType);
		linear_backward_accum_maybe_lowp(transformerScratch.x.data(), dH.data(), T, inputSize, dModel,
		                                 tt.gWIn, tt.gBIn, inputProj, dX.data());
		GLADES_ECHO_CPU_OBSERVE(echo_scope_uses_input_matrix(ac, dModel, inputSize),
		                        tt.echoWIn, dH.data(), transformerScratch.x.data(), T, dModel, inputSize);
		(void)dX;
	}

#undef GLADES_ECHO_CPU_OBSERVE
}


// ---------------------------------------------------------------------------
// Extracted GPU training epoch (previously inlined in SGDHelper_TRANSFORMER).
// Runs the complete forward/backward/optimizer loop on GPU for all sequences
// in the epoch.
// ---------------------------------------------------------------------------
#ifdef GLADES_HAVE_CUDA
bool glades::NNetwork::tryRunTransformerGpuEpoch(const TransformerEpochCfg& cfg, unsigned int seqCount,
                                                 int epochIdx, int64_t epochStartMs,
                                                 unsigned long long& tokensProcessed,
                                                 unsigned long long& targetsProcessed,
                                                 double& tokenLmNllSum,
                                                 unsigned long long& tokenLmTokenCount,
                                                 unsigned long long& clsCorrect,
                                                 unsigned long long& clsTotal,
                                                 shmea::GLogger* logger)
{
	if (!trainingConfig.gpu.enable || !cfg.isTrain)
		return false;
	if (trainingConfig.optimizer.type == glades::OptimizerConfig::ATLAS
	    && (trainingConfig.atlas.helmEnabled
	        || trainingConfig.atlas.kronEnabled))
		return false;

	const bool gpuReady = ensureGpuState();
	if (!gpuReady || !gpuTransformerWeights || !gpuTransformerWeights->initialized)
		return false;

	transformerGpuTrainEpoch(cfg, seqCount, epochIdx, epochStartMs,
	                        tokensProcessed, targetsProcessed,
	                        tokenLmNllSum, tokenLmTokenCount,
	                        clsCorrect, clsTotal, logger);
	return true;
}

bool glades::NNetwork::ensureTransformerGpuTrainingScratch(const TransformerEpochCfg& cfg, unsigned int T)
{
	glades::gpu::TransformerGpuScratchConfig scratchCfg;
	scratchCfg.T = T;
	scratchCfg.inputSize = cfg.inputSize;
	scratchCfg.outSize = cfg.outSize;
	scratchCfg.dModel = cfg.dModel;
	scratchCfg.dFF = cfg.dFF;
	scratchCfg.dModelKV = cfg.dModelKV;
	scratchCfg.nHeads = cfg.nHeads;
	scratchCfg.nLayers = cfg.nLayers;
	scratchCfg.ff1Width = cfg.ff1Width;
	scratchCfg.activationCheckpoint = trainingConfig.mixedPrecision.activationCheckpoint;
	return glades::gpu::ensureTransformerScratch(gpuTransformerScratch, scratchCfg);
}

bool glades::NNetwork::syncTransformerGpuTrainingWeightsToCpu()
{
	if (!gpuTransformerWeights || !gpuTransformerWeights->initialized)
		return false;

	std::vector<glades::gpu::TransformerHostBlockWeightsView> blockViews(tensorTransformer.blocks.size());
	for (size_t l = 0; l < tensorTransformer.blocks.size(); ++l)
	{
		TensorTransformerState::Block& block = tensorTransformer.blocks[l];
		blockViews[l] = make_transformer_host_block_view(block.ln1Gamma, block.ln1Beta,
		                                                 block.Wq, block.Wk, block.Wv, block.Wo,
		                                                 block.bq, block.bk, block.bv, block.bo,
		                                                 block.ln2Gamma, block.ln2Beta,
		                                                 block.W1, block.W2, block.b1, block.b2);
	}

	glades::gpu::TransformerHostWeightsView hostView;
	hostView.tokE = make_host_float_buffer_view(tensorTransformer.tokE);
	hostView.WIn = make_host_float_buffer_view(tensorTransformer.WIn);
	hostView.bIn = make_host_float_buffer_view(tensorTransformer.bIn);
	hostView.WOut = make_host_float_buffer_view(tensorTransformer.WOut);
	hostView.bOut = make_host_float_buffer_view(tensorTransformer.bOut);
	hostView.lmBias = make_host_float_buffer_view(tensorTransformer.lmBias);
	hostView.lnFinalGamma = make_host_float_buffer_view(tensorTransformer.lnFinalGamma);
	hostView.lnFinalBeta = make_host_float_buffer_view(tensorTransformer.lnFinalBeta);
	hostView.blocks = blockViews.empty() ? NULL : &blockViews[0];
	hostView.blockCount = static_cast<unsigned int>(blockViews.size());

	return glades::gpu::downloadTransformerWeightsToHost(*gpuTransformerWeights, hostView);
}

namespace {

// BF16 mixed-precision GEMM dispatch helpers for the GPU training path.
//
// Each helper matches one of the three float variants in gpu_blas.h:
//   - gpu_gemm_mp     ~ sgemm_rowmajor       (no transpose; input-grad dX = dY * W)
//   - gpu_gemm_atb_mp ~ sgemm_rowmajor_atb   (A transposed; weight-grad gW = dY^T * X)
//   - gpu_gemm_abt_mp ~ sgemm_rowmajor_abt   (B transposed; forward O = X * W^T)
//
// When useBf16 is false, the helper falls through to the FP32 variant and
// the *_scratch / *_bf16 pointers are ignored. When useBf16 is true, the
// relevant FP32 inputs are cast into the provided BF16 scratch buffers on
// the fly (via glades::gpu::cast_f32_to_bf16) and cublasGemmEx runs with
// BF16 inputs and FP32 accumulate (CUBLAS_COMPUTE_32F).
//
// All pointers are device pointers. Scratch buffers must be at least
// size(M*K) or size(K*N) depending on the dispatched variant; the caller
// owns allocation (see GpuTransformerScratch::activationLowp,
// ::activationLowp2).

#ifdef GLADES_HAVE_CUDA

static inline bool gpu_gemm_abt_mp(bool useBf16,
                                   int M, int N, int K, float alpha,
                                   const float* A_f32, uint16_t* A_scratch_bf16, int lda,
                                   const float* B_f32, const uint16_t* B_bf16, int ldb,
                                   float beta, float* C, int ldc)
{
	if (useBf16)
	{
		const size_t aN = static_cast<size_t>(M) * static_cast<size_t>(K);
		if (!glades::gpu::cast_f32_to_bf16(A_f32, A_scratch_bf16, aN))
			return false;
		return glades::gpu::sgemm_rowmajor_abt_bf16(M, N, K, alpha,
		    A_scratch_bf16, lda, B_bf16, ldb, beta, C, ldc);
	}
	return glades::gpu::sgemm_rowmajor_abt(M, N, K, alpha,
	    A_f32, lda, B_f32, ldb, beta, C, ldc);
}

static inline bool gpu_gemm_mp(bool useBf16,
                               int M, int N, int K, float alpha,
                               const float* A_f32, uint16_t* A_scratch_bf16, int lda,
                               const float* B_f32, const uint16_t* B_bf16, int ldb,
                               float beta, float* C, int ldc)
{
	if (useBf16)
	{
		const size_t aN = static_cast<size_t>(M) * static_cast<size_t>(K);
		if (!glades::gpu::cast_f32_to_bf16(A_f32, A_scratch_bf16, aN))
			return false;
		return glades::gpu::sgemm_rowmajor_bf16(M, N, K, alpha,
		    A_scratch_bf16, lda, B_bf16, ldb, beta, C, ldc);
	}
	return glades::gpu::sgemm_rowmajor(M, N, K, alpha,
	    A_f32, lda, B_f32, ldb, beta, C, ldc);
}

// Weight-grad pattern: gW = alpha * dY^T * X + beta * gW. Both dY (A) and
// X (B) are FP32 activations; need two scratch buffers in BF16 mode.
static inline bool gpu_gemm_atb_mp(bool useBf16,
                                   int M, int N, int K, float alpha,
                                   const float* A_f32, uint16_t* A_scratch_bf16, int lda,
                                   const float* B_f32, uint16_t* B_scratch_bf16, int ldb,
                                   float beta, float* C, int ldc)
{
	if (useBf16)
	{
		// A has shape [K, M] row-major (so K*M entries).
		const size_t aN = static_cast<size_t>(K) * static_cast<size_t>(M);
		const size_t bN = static_cast<size_t>(K) * static_cast<size_t>(N);
		if (!glades::gpu::cast_f32_to_bf16(A_f32, A_scratch_bf16, aN))
			return false;
		if (!glades::gpu::cast_f32_to_bf16(B_f32, B_scratch_bf16, bN))
			return false;
		return glades::gpu::sgemm_rowmajor_atb_bf16(M, N, K, alpha,
		    A_scratch_bf16, lda, B_scratch_bf16, ldb, beta, C, ldc);
	}
	return glades::gpu::sgemm_rowmajor_atb(M, N, K, alpha,
	    A_f32, lda, B_f32, ldb, beta, C, ldc);
}

#endif // GLADES_HAVE_CUDA

} // anonymous namespace

#ifdef GLADES_HAVE_CUDA

// Extracted GPU forward pass for one sequence. Used by:
//   (1) transformerGpuTrainEpoch's per-sequence inner body (normal forward).
//   (2) HELIOS FD-HVP probe — called 2× with perturbed target weights to
//       compute kappa = (L_plus + L_minus - 2 L_0) / eps^2 (scalar-FD form).
// See research/HELIOS_framework.md §14a for the probe protocol.
//
// Preconditions:
//   - For tokenLM: token IDs already uploaded to gpuTransformerScratch->tokenIds.
//   - For dense: input features already uploaded to gpuTransformerScratch->x.
//   - RoPE invFreq uploaded to gpuTransformerScratch->gpuInvFreq (if useRope).
//   - BF16 mirrors refreshed (if useBf16) via ensureLowpMirrors().
//
// Writes to: gpuTransformerScratch activations (h, x1, Q, K, V, attnConcat,
// attnOut, hAfterAttn, x2, ff1, ff1Act, ffOut, hAfterFF, hPostFinalLN,
// logits, probs). Does not touch gradient buffers or loss accumulators.
bool glades::NNetwork::transformerGpuRunForwardOnly(
    const TransformerEpochCfg& cfg,
    unsigned int T,
    bool useBf16, bool useRope,
    bool bf16WIn, bool bf16Wq, bool bf16Wk,
    bool bf16Wv, bool bf16Wo, bool bf16W1,
    bool bf16W2, bool bf16Head,
    int ropeDimOverride,
    void* gpuPerfOpaque)
{
	if (!gpuTransformerWeights || !gpuTransformerScratch) return false;
	if (T == 0u) return false;

	const unsigned int dModel = cfg.dModel;
	const unsigned int dFF = cfg.dFF;
	const unsigned int nHeads = cfg.nHeads;
	const unsigned int nKVHeads = cfg.nKVHeads > 0u ? cfg.nKVHeads : cfg.nHeads;
	const unsigned int nLayers = cfg.nLayers;
	const unsigned int vocabSize = cfg.vocabSize;
	const unsigned int inputSize = cfg.inputSize;
	const unsigned int outSize = cfg.outSize;
	const unsigned int dHead = dModel / nHeads;
	const unsigned int dModelKV = nKVHeads * dHead;
	const unsigned int ffnKind = static_cast<unsigned int>(cfg.ffnKind);
	const unsigned int ff1Width = (ffnKind == 1u) ? (2u * dFF) : dFF;
	const bool tokenLM = cfg.tokenLM;
	const bool tieEmb = cfg.tieEmb;
	const bool causal = cfg.causal;
	const int normType = cfg.normType;
	const int ffnAct = cfg.ffnAct;
	const float lnEps = cfg.lnEps;

	TransformerGpuPerfBreakdown* gpuPerf =
	    reinterpret_cast<TransformerGpuPerfBreakdown*>(gpuPerfOpaque);
	(void)gpuPerf; // reserved for fine-grained timing; not used for probe calls

	// Embedding / input projection.
	if (tokenLM)
	{
		// bf16-weights master-retire path: route through bf16 mirror gather
		// when FP32 tokE is retired.
		if (gpuTransformerWeights->tokE.size() == 0
		    && gpuTransformerWeights->tokELowp.allocated())
		{
			gpu::embedding_gather_bf16(
			    gpuTransformerWeights->tokELowp.data(),
			    gpuTransformerScratch->tokenIds.data(),
			    static_cast<int>(T), static_cast<int>(vocabSize),
			    static_cast<int>(dModel),
			    gpuTransformerScratch->h.data());
		}
		else
		{
			gpu::embedding_gather(
			    gpuTransformerWeights->tokE.data(),
			    gpuTransformerScratch->tokenIds.data(),
			    static_cast<int>(T), static_cast<int>(vocabSize),
			    static_cast<int>(dModel),
			    gpuTransformerScratch->h.data());
		}
	}
	else
	{
		if (!gpu_gemm_abt_mp(bf16WIn,
		    static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(inputSize),
		    1.0f,
		    gpuTransformerScratch->x.data(),
		    gpuTransformerScratch->activationLowp.data(),
		    static_cast<int>(inputSize),
		    gpuTransformerWeights->WIn.data(),
		    gpuTransformerWeights->WInLowp.data(),
		    static_cast<int>(inputSize),
		    0.0f,
		    gpuTransformerScratch->h.data(), static_cast<int>(dModel)))
			return false;
		gpu::add_bias(gpuTransformerScratch->h.data(),
		              gpuTransformerWeights->bIn.data(),
		              static_cast<int>(T), static_cast<int>(dModel));
	}

	// Per-layer transformer blocks.
	const unsigned int slotsPerLayerFwdOnly = gpuTransformerScratch->slotsPerLayer
	    ? gpuTransformerScratch->slotsPerLayer : nLayers;
	for (unsigned int li = 0; li < nLayers; ++li)
	{
		gpu::GpuTransformerWeights::Block& gb = gpuTransformerWeights->blocks[li];
		// Activation-checkpoint: per-layer activations live in cyclic slots
		// (slot = li % K).  When checkpointing is off, slotsPerLayer == nLayers
		// and modulo collapses to identity.
		const size_t slot = static_cast<size_t>(li % slotsPerLayerFwdOnly);
		const size_t prevSlot = (li > 0u)
		    ? static_cast<size_t>((li - 1u) % slotsPerLayerFwdOnly)
		    : 0u;
		const size_t layerOff = slot * static_cast<size_t>(T);

		const float* layerIn = (li == 0)
		    ? gpuTransformerScratch->h.data()
		    : (gpuTransformerScratch->hAfterFF.data()
		       + prevSlot * T * dModel);

		float* x1_l = gpuTransformerScratch->x1.data()
		              + slot * T * dModel;
		float* ln1Mean_l = gpuTransformerScratch->ln1Mean.data() + layerOff;
		float* ln1InvStd_l = gpuTransformerScratch->ln1InvStd.data() + layerOff;

		if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
			gpu::rmsnorm_forward(layerIn, gb.ln1Gamma.data(), lnEps,
			                     static_cast<int>(T), static_cast<int>(dModel),
			                     x1_l, ln1InvStd_l);
		else
			gpu::layernorm_forward(layerIn, gb.ln1Gamma.data(), gb.ln1Beta.data(),
			                       lnEps, static_cast<int>(T), static_cast<int>(dModel),
			                       x1_l, ln1Mean_l, ln1InvStd_l);

		float* Q_l = gpuTransformerScratch->Q.data()
		             + slot * T * dModel;
		float* K_l = gpuTransformerScratch->K.data()
		             + slot * T * dModelKV;
		float* V_l = gpuTransformerScratch->V.data()
		             + slot * T * dModelKV;

		if (!gpu_gemm_abt_mp(bf16Wq,
		    static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(dModel), 1.0f,
		    x1_l, gpuTransformerScratch->activationLowp.data(), static_cast<int>(dModel),
		    gb.Wq.data(), gb.WqLowp.data(), static_cast<int>(dModel),
		    0.0f, Q_l, static_cast<int>(dModel)))
			return false;
		gpu::add_bias(Q_l, gb.bq.data(), static_cast<int>(T), static_cast<int>(dModel));

		// Paradigm shift #76 MLA dispatch: when mlaLatentDim > 0, replace the
		// standard W_K, W_V projections with the low-rank latent path:
		//   c = x1 @ Wdkv  ;  K = c @ Wuk  ;  V = c @ Wuv
		// Otherwise fall through to the standard MHA gemm.
		const int mlaDc = trainingConfig.transformer.mlaLatentDim;
		if (mlaDc > 0 && gb.Wdkv.allocated()) {
			// Allocate / size the per-step c scratch (size T * dC).
			const size_t cBytes = static_cast<size_t>(T) * static_cast<size_t>(mlaDc);
			if (gb.mlaC.size() < cBytes) gb.mlaC.allocate(cBytes);
			if (!gpu::mla_attention_forward_gpu(
			        x1_l, gb.Wdkv.data(), gb.Wuk.data(), gb.Wuv.data(),
			        static_cast<int>(T), static_cast<int>(dModel), mlaDc, static_cast<int>(dModelKV),
			        gb.mlaC.data(), K_l, V_l))
				return false;
			gpu::add_bias(K_l, gb.bk.data(), static_cast<int>(T), static_cast<int>(dModelKV));
			gpu::add_bias(V_l, gb.bv.data(), static_cast<int>(T), static_cast<int>(dModelKV));
		} else {
			if (!gpu_gemm_abt_mp(bf16Wk,
			    static_cast<int>(T), static_cast<int>(dModelKV), static_cast<int>(dModel), 1.0f,
			    x1_l, gpuTransformerScratch->activationLowp.data(), static_cast<int>(dModel),
			    gb.Wk.data(), gb.WkLowp.data(), static_cast<int>(dModel),
			    0.0f, K_l, static_cast<int>(dModelKV)))
				return false;
			gpu::add_bias(K_l, gb.bk.data(), static_cast<int>(T), static_cast<int>(dModelKV));

			if (!gpu_gemm_abt_mp(bf16Wv,
			    static_cast<int>(T), static_cast<int>(dModelKV), static_cast<int>(dModel), 1.0f,
			    x1_l, gpuTransformerScratch->activationLowp.data(), static_cast<int>(dModel),
			    gb.Wv.data(), gb.WvLowp.data(), static_cast<int>(dModel),
			    0.0f, V_l, static_cast<int>(dModelKV)))
				return false;
			gpu::add_bias(V_l, gb.bv.data(), static_cast<int>(T), static_cast<int>(dModelKV));
		}

		if (useRope && !transformerPosEncCache.ropeInvFreq.empty())
		{
			const unsigned int rd =
			    (ropeDimOverride > 0 && static_cast<unsigned int>(ropeDimOverride) < dHead)
			    ? static_cast<unsigned int>(ropeDimOverride) : dHead;
			gpu::rope_apply_qk(Q_l, K_l, gpuTransformerScratch->gpuInvFreq.data(),
			                   static_cast<int>(T), static_cast<int>(nHeads),
			                   static_cast<int>(nKVHeads), static_cast<int>(dHead),
			                   static_cast<int>(rd / 2u));
		}

		float* attnConcat_l = gpuTransformerScratch->attnConcat.data()
		                      + slot * T * dModel;
		if (useBf16)
		{
			glades::gpu::cast_f32_to_bf16(Q_l, gpuTransformerScratch->qLowp.data(),
			    static_cast<size_t>(T) * dModel);
			glades::gpu::cast_f32_to_bf16(K_l, gpuTransformerScratch->kLowp.data(),
			    static_cast<size_t>(T) * dModelKV);
			glades::gpu::cast_f32_to_bf16(V_l, gpuTransformerScratch->vLowp.data(),
			    static_cast<size_t>(T) * dModelKV);
			// Local-window attention (paradigm shift #6 port to main transformer).
			// When localAttnWindow > 0, use the O(T·W) variant.
			// Paradigm #78 ATTENTION-SINK: when attnSinkCount > 0, force the
			// local kernel even at full window so the first sinkCount keys are
			// always retained.
			const int localW = trainingConfig.transformer.localAttnWindow;
			const int sinkS = trainingConfig.transformer.attnSinkCount > 0
			    ? trainingConfig.transformer.attnSinkCount : 0;
			if ((localW > 0 && localW < static_cast<int>(T)) || sinkS > 0) {
				gpu::flash_attention_multihead_forward_bf16_local(
				    gpuTransformerScratch->qLowp.data(),
				    gpuTransformerScratch->kLowp.data(),
				    gpuTransformerScratch->vLowp.data(),
				    static_cast<int>(T), static_cast<int>(nHeads),
				    static_cast<int>(nKVHeads), static_cast<int>(dHead),
				    static_cast<int>(dModel), static_cast<int>(dModelKV),
				    causal, localW, attnConcat_l, sinkS);
			} else {
				gpu::flash_attention_multihead_forward_bf16(
				    gpuTransformerScratch->qLowp.data(),
				    gpuTransformerScratch->kLowp.data(),
				    gpuTransformerScratch->vLowp.data(),
				    static_cast<int>(T), static_cast<int>(nHeads),
				    static_cast<int>(nKVHeads), static_cast<int>(dHead),
				    static_cast<int>(dModel), static_cast<int>(dModelKV),
				    causal, attnConcat_l);
			}
		}
		else
		{
			// cuBLAS-tiled tensor-core attention (eval forward path).
			const bool chiron_fast_eval = (nHeads == nKVHeads);
			if (chiron_fast_eval)
			{
				const size_t scoresNeeded = static_cast<size_t>(nHeads) * T * T;
				if (gpuTransformerScratch->attnScoresScratch.size() < scoresNeeded)
					gpuTransformerScratch->attnScoresScratch.allocate(scoresNeeded);
				if (gpuTransformerScratch->attnScoresScratch.size() >= scoresNeeded)
				{
					gpu::flash_attention_cublas_tiled(Q_l, K_l, V_l,
					    static_cast<int>(T), static_cast<int>(nHeads),
					    static_cast<int>(dHead), static_cast<int>(dModel),
					    causal, attnConcat_l,
					    gpuTransformerScratch->attnScoresScratch.data());
				}
				else
				{
					gpu::flash_attention_multihead_forward(Q_l, K_l, V_l,
					    static_cast<int>(T), static_cast<int>(nHeads),
					    static_cast<int>(nKVHeads), static_cast<int>(dHead),
					    static_cast<int>(dModel), static_cast<int>(dModelKV),
					    causal, attnConcat_l);
				}
			}
			else
			{
			gpu::flash_attention_multihead_forward(Q_l, K_l, V_l,
			    static_cast<int>(T), static_cast<int>(nHeads),
			    static_cast<int>(nKVHeads), static_cast<int>(dHead),
			    static_cast<int>(dModel), static_cast<int>(dModelKV),
			    causal, attnConcat_l);
			}
		}

		float* attnOut_l = gpuTransformerScratch->attnOut.data()
		                   + slot * T * dModel;
		if (!gpu_gemm_abt_mp(bf16Wo,
		    static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(dModel), 1.0f,
		    attnConcat_l, gpuTransformerScratch->activationLowp.data(),
		    static_cast<int>(dModel),
		    gb.Wo.data(), gb.WoLowp.data(), static_cast<int>(dModel),
		    0.0f, attnOut_l, static_cast<int>(dModel)))
			return false;
		gpu::add_bias(attnOut_l, gb.bo.data(),
		              static_cast<int>(T), static_cast<int>(dModel));

		float* hAfterAttn_l = gpuTransformerScratch->hAfterAttn.data()
		                      + slot * T * dModel;
		gpu::add_two(hAfterAttn_l, layerIn, attnOut_l,
		             static_cast<int>(T * dModel));

		float* x2_l = gpuTransformerScratch->x2.data()
		              + slot * T * dModel;
		float* ln2Mean_l = gpuTransformerScratch->ln2Mean.data() + layerOff;
		float* ln2InvStd_l = gpuTransformerScratch->ln2InvStd.data() + layerOff;

		if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
			gpu::rmsnorm_forward(hAfterAttn_l, gb.ln2Gamma.data(), lnEps,
			                     static_cast<int>(T), static_cast<int>(dModel),
			                     x2_l, ln2InvStd_l);
		else
			gpu::layernorm_forward(hAfterAttn_l, gb.ln2Gamma.data(), gb.ln2Beta.data(),
			                       lnEps, static_cast<int>(T), static_cast<int>(dModel),
			                       x2_l, ln2Mean_l, ln2InvStd_l);

		float* ff1_l = gpuTransformerScratch->ff1.data()
		               + slot * T * ff1Width;
		float* ff1Act_l = gpuTransformerScratch->ff1Act.data()
		                  + slot * T * dFF;
		float* ffOut_l = gpuTransformerScratch->ffOut.data()
		                 + slot * T * dModel;

		// Paradigm #74 PHOENIX-1BIT W1 projection: Y = X @ sign(W1).T
		// (mirrors W2 path). When dModel%128==0, use full BitNet QAT via
		// WMMA B1 with per-row scales; otherwise tiled X(float)@sign(W).
		const bool useBinaryFFNW1 = trainingConfig.transformer.binaryFFN;
		if (useBinaryFFNW1)
		{
			const bool useBitNetW1 = (dModel % 128u) == 0u;
			if (useBitNetW1)
			{
				const size_t xBits = static_cast<size_t>(T) * (dModel / 32u);
				const size_t wBits = static_cast<size_t>(ff1Width) * (dModel / 32u);
				if (gb.bitnetXBitsW1.size() < xBits) gb.bitnetXBitsW1.allocate(xBits);
				if (gb.bitnetWBitsW1.size() < wBits) gb.bitnetWBitsW1.allocate(wBits);
				if (gb.bitnetAlphaXW1.size() < T) gb.bitnetAlphaXW1.allocate(T);
				if (gb.bitnetAlphaWW1.size() < ff1Width) gb.bitnetAlphaWW1.allocate(ff1Width);
				const size_t cPop = static_cast<size_t>(T) * ff1Width;
				if (gb.bitnetCPopW1.size() < cPop) gb.bitnetCPopW1.allocate(cPop);
				if (!gpu::bitnet_ffn_forward_gpu(
				        x2_l, gb.W1.data(),
				        static_cast<int>(T), static_cast<int>(ff1Width), static_cast<int>(dModel),
				        gb.bitnetXBitsW1.data(), gb.bitnetWBitsW1.data(),
				        gb.bitnetAlphaXW1.data(), gb.bitnetAlphaWW1.data(),
				        gb.bitnetCPopW1.data(), ff1_l))
					return false;
			}
			else
			{
				if (!gpu::binary_gemm_abt_from_float(
				        x2_l, gb.W1.data(),
				        static_cast<int>(T), static_cast<int>(ff1Width), static_cast<int>(dModel),
				        ff1_l))
					return false;
			}
		}
		else
		{
			if (!gpu_gemm_abt_mp(bf16W1,
			    static_cast<int>(T), static_cast<int>(ff1Width), static_cast<int>(dModel), 1.0f,
			    x2_l, gpuTransformerScratch->activationLowp.data(), static_cast<int>(dModel),
			    gb.W1.data(), gb.W1Lowp.data(), static_cast<int>(dModel),
			    0.0f, ff1_l, static_cast<int>(ff1Width)))
				return false;
		}
		gpu::add_bias(ff1_l, gb.b1.data(),
		              static_cast<int>(T), static_cast<int>(ff1Width));

		if (ffnKind == 1u)
			gpu::swiglu_forward(ff1_l, static_cast<int>(T), static_cast<int>(dFF),
			                    ff1Act_l);
		else if (ffnAct == static_cast<int>(glades::TransformerRunConfig::FFN_GELU))
			gpu::gelu_forward(ff1_l, static_cast<int>(T * dFF), ff1Act_l);
		else
			gpu::relu_forward(ff1_l, static_cast<int>(T * dFF), ff1Act_l);

		// Paradigm #74 PHOENIX-1BIT: binary W2 projection (Y = X @ sign(W2).T).
		// Float master weights still drive backward via STE.
		// When K=dFF is a multiple of 128, route through full BitNet QAT
		// path (quantize-X + WMMA-XOR + scale-recover); otherwise fall back
		// to the tiled X(float) @ sign(W) kernel.
		const bool useBinaryFFN = trainingConfig.transformer.binaryFFN;
		if (useBinaryFFN)
		{
			const bool useBitNet = (dFF % 128u) == 0u;
			if (useBitNet)
			{
				// Lazy-allocate scratch buffers
				const size_t xBits = static_cast<size_t>(T) * (dFF / 32u);
				const size_t wBits = static_cast<size_t>(dModel) * (dFF / 32u);
				if (gb.bitnetXBits.size() < xBits) gb.bitnetXBits.allocate(xBits);
				if (gb.bitnetWBits.size() < wBits) gb.bitnetWBits.allocate(wBits);
				if (gb.bitnetAlphaX.size() < T) gb.bitnetAlphaX.allocate(T);
				if (gb.bitnetAlphaW.size() < dModel) gb.bitnetAlphaW.allocate(dModel);
				const size_t cPop = static_cast<size_t>(T) * dModel;
				if (gb.bitnetCPop.size() < cPop) gb.bitnetCPop.allocate(cPop);
				if (!gpu::bitnet_ffn_forward_gpu(
				        ff1Act_l, gb.W2.data(),
				        static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(dFF),
				        gb.bitnetXBits.data(), gb.bitnetWBits.data(),
				        gb.bitnetAlphaX.data(), gb.bitnetAlphaW.data(),
				        gb.bitnetCPop.data(), ffOut_l))
					return false;
			}
			else
			{
				if (!gpu::binary_gemm_abt_from_float(
				        ff1Act_l, gb.W2.data(),
				        static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(dFF),
				        ffOut_l))
					return false;
			}
		}
		else
		{
			if (!gpu_gemm_abt_mp(bf16W2,
			    static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(dFF), 1.0f,
			    ff1Act_l, gpuTransformerScratch->activationLowp.data(), static_cast<int>(dFF),
			    gb.W2.data(), gb.W2Lowp.data(), static_cast<int>(dFF),
			    0.0f, ffOut_l, static_cast<int>(dModel)))
				return false;
		}
		gpu::add_bias(ffOut_l, gb.b2.data(),
		              static_cast<int>(T), static_cast<int>(dModel));

		float* hAfterFF_l = gpuTransformerScratch->hAfterFF.data()
		                    + slot * T * dModel;
		gpu::add_two(hAfterFF_l, hAfterAttn_l, ffOut_l,
		             static_cast<int>(T * dModel));
	}

	// Final LayerNorm.  In activation-checkpoint mode the last layer's hAfterFF
	// lives in slot ((nLayers - 1) % slotsPerLayer); otherwise modulo collapses
	// to (nLayers - 1).
	const float* finalH = gpuTransformerScratch->hAfterFF.data()
	                      + static_cast<size_t>((nLayers - 1) % slotsPerLayerFwdOnly) * T * dModel;
	float* hPostFinalLN = gpuTransformerScratch->hPostFinalLN.data();
	if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
		gpu::rmsnorm_forward(finalH, gpuTransformerWeights->lnFinalGamma.data(),
		                     lnEps, static_cast<int>(T), static_cast<int>(dModel),
		                     hPostFinalLN,
		                     gpuTransformerScratch->lnFinalInvStd.data());
	else
		gpu::layernorm_forward(finalH, gpuTransformerWeights->lnFinalGamma.data(),
		                       gpuTransformerWeights->lnFinalBeta.data(), lnEps,
		                       static_cast<int>(T), static_cast<int>(dModel),
		                       hPostFinalLN,
		                       gpuTransformerScratch->lnFinalMean.data(),
		                       gpuTransformerScratch->lnFinalInvStd.data());

	// Output head: tied (tokenLM + tieEmb) or WOut.
	if (tokenLM && tieEmb)
	{
		if (!gpu_gemm_abt_mp(bf16Head,
		    static_cast<int>(T), static_cast<int>(vocabSize), static_cast<int>(dModel),
		    1.0f,
		    hPostFinalLN, gpuTransformerScratch->activationLowp.data(),
		    static_cast<int>(dModel),
		    gpuTransformerWeights->tokE.data(),
		    gpuTransformerWeights->tokELowp.data(),
		    static_cast<int>(dModel),
		    0.0f, gpuTransformerScratch->logits.data(), static_cast<int>(vocabSize)))
			return false;
		gpu::add_bias(gpuTransformerScratch->logits.data(),
		              gpuTransformerWeights->lmBias.data(),
		              static_cast<int>(T), static_cast<int>(vocabSize));
	}
	else if (!tokenLM)
	{
		if (!gpu_gemm_abt_mp(bf16Head,
		    static_cast<int>(T), static_cast<int>(outSize), static_cast<int>(dModel),
		    1.0f,
		    hPostFinalLN, gpuTransformerScratch->activationLowp.data(),
		    static_cast<int>(dModel),
		    gpuTransformerWeights->WOut.data(),
		    gpuTransformerWeights->WOutLowp.data(),
		    static_cast<int>(dModel),
		    0.0f, gpuTransformerScratch->logits.data(), static_cast<int>(outSize)))
			return false;
		gpu::add_bias(gpuTransformerScratch->logits.data(),
		              gpuTransformerWeights->bOut.data(),
		              static_cast<int>(T), static_cast<int>(outSize));
	}

	gpu::softmax_forward(gpuTransformerScratch->logits.data(),
	                     static_cast<int>(T), static_cast<int>(outSize),
	                     gpuTransformerScratch->probs.data());
	return true;
}

// Activation-checkpoint helper.  Re-runs the per-layer forward body for layers
// in [segStart, segEnd), populating cyclic activation slots (slot = li % K)
// so that the backward path can read them.  Mirrors the kernel sequence in
// transformerGpuRunForwardOnly's layer loop, minus embedding/final-LN/output-
// head — those are handled once per step in the train epoch and don't need
// recomputing.  When segmentInputOverride is non-NULL it is used as the input
// to the FIRST recomputed layer (segStart); subsequent layers in the segment
// read from the just-written cyclic slots.
bool glades::NNetwork::transformerGpuLayerRangeForward(
    const TransformerEpochCfg& cfg,
    unsigned int T,
    unsigned int segStart, unsigned int segEnd,
    const float* segmentInputOverride,
    bool useBf16, bool useRope,
    bool bf16Wq, bool bf16Wk, bool bf16Wv,
    bool bf16Wo, bool bf16W1, bool bf16W2,
    int ropeDimOverride)
{
	if (!gpuTransformerWeights || !gpuTransformerScratch) return false;
	if (T == 0u) return false;
	if (segEnd <= segStart) return true;

	const unsigned int dModel = cfg.dModel;
	const unsigned int dFF = cfg.dFF;
	const unsigned int nHeads = cfg.nHeads;
	const unsigned int nKVHeads = cfg.nKVHeads > 0u ? cfg.nKVHeads : cfg.nHeads;
	const unsigned int nLayers = cfg.nLayers;
	const unsigned int dHead = dModel / nHeads;
	const unsigned int dModelKV = nKVHeads * dHead;
	const unsigned int ffnKind = static_cast<unsigned int>(cfg.ffnKind);
	const unsigned int ff1Width = (ffnKind == 1u) ? (2u * dFF) : dFF;
	const bool causal = cfg.causal;
	const int normType = cfg.normType;
	const int ffnAct = cfg.ffnAct;
	const float lnEps = cfg.lnEps;

	const unsigned int slotsPerLayer = gpuTransformerScratch->slotsPerLayer
	    ? gpuTransformerScratch->slotsPerLayer : nLayers;
	if (segEnd > nLayers) return false;

	for (unsigned int li = segStart; li < segEnd; ++li)
	{
		gpu::GpuTransformerWeights::Block& gb = gpuTransformerWeights->blocks[li];
		const size_t slot = static_cast<size_t>(li % slotsPerLayer);
		const size_t prevSlot = (li > 0u)
		    ? static_cast<size_t>((li - 1u) % slotsPerLayer)
		    : 0u;
		const size_t layerOff = slot * static_cast<size_t>(T);

		// First recomputed layer in the segment uses the override (a
		// hAfterFF checkpoint copy).  Subsequent layers read from the
		// just-populated cyclic slots.
		const float* layerIn;
		if (li == segStart && segmentInputOverride)
			layerIn = segmentInputOverride;
		else if (li == 0)
			layerIn = gpuTransformerScratch->h.data();
		else
			layerIn = gpuTransformerScratch->hAfterFF.data()
			          + prevSlot * T * dModel;

		float* x1_l = gpuTransformerScratch->x1.data() + slot * T * dModel;
		float* ln1Mean_l = gpuTransformerScratch->ln1Mean.data() + layerOff;
		float* ln1InvStd_l = gpuTransformerScratch->ln1InvStd.data() + layerOff;

		if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
			gpu::rmsnorm_forward(layerIn, gb.ln1Gamma.data(), lnEps,
			                     static_cast<int>(T), static_cast<int>(dModel),
			                     x1_l, ln1InvStd_l);
		else
			gpu::layernorm_forward(layerIn, gb.ln1Gamma.data(), gb.ln1Beta.data(),
			                       lnEps, static_cast<int>(T), static_cast<int>(dModel),
			                       x1_l, ln1Mean_l, ln1InvStd_l);

		float* Q_l = gpuTransformerScratch->Q.data() + slot * T * dModel;
		float* K_l = gpuTransformerScratch->K.data() + slot * T * dModelKV;
		float* V_l = gpuTransformerScratch->V.data() + slot * T * dModelKV;

		if (!gpu_gemm_abt_mp(bf16Wq,
		    static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(dModel), 1.0f,
		    x1_l, gpuTransformerScratch->activationLowp.data(), static_cast<int>(dModel),
		    gb.Wq.data(), gb.WqLowp.data(), static_cast<int>(dModel),
		    0.0f, Q_l, static_cast<int>(dModel)))
			return false;
		gpu::add_bias(Q_l, gb.bq.data(), static_cast<int>(T), static_cast<int>(dModel));

		const int mlaDc = trainingConfig.transformer.mlaLatentDim;
		if (mlaDc > 0 && gb.Wdkv.allocated()) {
			const size_t cBytes = static_cast<size_t>(T) * static_cast<size_t>(mlaDc);
			if (gb.mlaC.size() < cBytes) gb.mlaC.allocate(cBytes);
			if (!gpu::mla_attention_forward_gpu(
			        x1_l, gb.Wdkv.data(), gb.Wuk.data(), gb.Wuv.data(),
			        static_cast<int>(T), static_cast<int>(dModel), mlaDc, static_cast<int>(dModelKV),
			        gb.mlaC.data(), K_l, V_l))
				return false;
			gpu::add_bias(K_l, gb.bk.data(), static_cast<int>(T), static_cast<int>(dModelKV));
			gpu::add_bias(V_l, gb.bv.data(), static_cast<int>(T), static_cast<int>(dModelKV));
		} else {
			if (!gpu_gemm_abt_mp(bf16Wk,
			    static_cast<int>(T), static_cast<int>(dModelKV), static_cast<int>(dModel), 1.0f,
			    x1_l, gpuTransformerScratch->activationLowp.data(), static_cast<int>(dModel),
			    gb.Wk.data(), gb.WkLowp.data(), static_cast<int>(dModel),
			    0.0f, K_l, static_cast<int>(dModelKV)))
				return false;
			gpu::add_bias(K_l, gb.bk.data(), static_cast<int>(T), static_cast<int>(dModelKV));

			if (!gpu_gemm_abt_mp(bf16Wv,
			    static_cast<int>(T), static_cast<int>(dModelKV), static_cast<int>(dModel), 1.0f,
			    x1_l, gpuTransformerScratch->activationLowp.data(), static_cast<int>(dModel),
			    gb.Wv.data(), gb.WvLowp.data(), static_cast<int>(dModel),
			    0.0f, V_l, static_cast<int>(dModelKV)))
				return false;
			gpu::add_bias(V_l, gb.bv.data(), static_cast<int>(T), static_cast<int>(dModelKV));
		}

		if (useRope && !transformerPosEncCache.ropeInvFreq.empty())
		{
			const unsigned int rd =
			    (ropeDimOverride > 0 && static_cast<unsigned int>(ropeDimOverride) < dHead)
			    ? static_cast<unsigned int>(ropeDimOverride) : dHead;
			gpu::rope_apply_qk(Q_l, K_l, gpuTransformerScratch->gpuInvFreq.data(),
			                   static_cast<int>(T), static_cast<int>(nHeads),
			                   static_cast<int>(nKVHeads), static_cast<int>(dHead),
			                   static_cast<int>(rd / 2u));
		}

		float* attnConcat_l = gpuTransformerScratch->attnConcat.data() + slot * T * dModel;
		if (useBf16)
		{
			glades::gpu::cast_f32_to_bf16(Q_l, gpuTransformerScratch->qLowp.data(),
			    static_cast<size_t>(T) * dModel);
			glades::gpu::cast_f32_to_bf16(K_l, gpuTransformerScratch->kLowp.data(),
			    static_cast<size_t>(T) * dModelKV);
			glades::gpu::cast_f32_to_bf16(V_l, gpuTransformerScratch->vLowp.data(),
			    static_cast<size_t>(T) * dModelKV);
			const int localW = trainingConfig.transformer.localAttnWindow;
			const int sinkS = trainingConfig.transformer.attnSinkCount > 0
			    ? trainingConfig.transformer.attnSinkCount : 0;
			if ((localW > 0 && localW < static_cast<int>(T)) || sinkS > 0) {
				gpu::flash_attention_multihead_forward_bf16_local(
				    gpuTransformerScratch->qLowp.data(),
				    gpuTransformerScratch->kLowp.data(),
				    gpuTransformerScratch->vLowp.data(),
				    static_cast<int>(T), static_cast<int>(nHeads),
				    static_cast<int>(nKVHeads), static_cast<int>(dHead),
				    static_cast<int>(dModel), static_cast<int>(dModelKV),
				    causal, localW, attnConcat_l, sinkS);
			} else {
				gpu::flash_attention_multihead_forward_bf16(
				    gpuTransformerScratch->qLowp.data(),
				    gpuTransformerScratch->kLowp.data(),
				    gpuTransformerScratch->vLowp.data(),
				    static_cast<int>(T), static_cast<int>(nHeads),
				    static_cast<int>(nKVHeads), static_cast<int>(dHead),
				    static_cast<int>(dModel), static_cast<int>(dModelKV),
				    causal, attnConcat_l);
			}
		}
		else
		{
			const bool chiron_fast_eval = (nHeads == nKVHeads);
			if (chiron_fast_eval)
			{
				const size_t scoresNeeded = static_cast<size_t>(nHeads) * T * T;
				if (gpuTransformerScratch->attnScoresScratch.size() < scoresNeeded)
					gpuTransformerScratch->attnScoresScratch.allocate(scoresNeeded);
				if (gpuTransformerScratch->attnScoresScratch.size() >= scoresNeeded)
				{
					gpu::flash_attention_cublas_tiled(Q_l, K_l, V_l,
					    static_cast<int>(T), static_cast<int>(nHeads),
					    static_cast<int>(dHead), static_cast<int>(dModel),
					    causal, attnConcat_l,
					    gpuTransformerScratch->attnScoresScratch.data());
				}
				else
				{
					gpu::flash_attention_multihead_forward(Q_l, K_l, V_l,
					    static_cast<int>(T), static_cast<int>(nHeads),
					    static_cast<int>(nKVHeads), static_cast<int>(dHead),
					    static_cast<int>(dModel), static_cast<int>(dModelKV),
					    causal, attnConcat_l);
				}
			}
			else
			{
				gpu::flash_attention_multihead_forward(Q_l, K_l, V_l,
				    static_cast<int>(T), static_cast<int>(nHeads),
				    static_cast<int>(nKVHeads), static_cast<int>(dHead),
				    static_cast<int>(dModel), static_cast<int>(dModelKV),
				    causal, attnConcat_l);
			}
		}

		float* attnOut_l = gpuTransformerScratch->attnOut.data() + slot * T * dModel;
		if (!gpu_gemm_abt_mp(bf16Wo,
		    static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(dModel), 1.0f,
		    attnConcat_l, gpuTransformerScratch->activationLowp.data(),
		    static_cast<int>(dModel),
		    gb.Wo.data(), gb.WoLowp.data(), static_cast<int>(dModel),
		    0.0f, attnOut_l, static_cast<int>(dModel)))
			return false;
		gpu::add_bias(attnOut_l, gb.bo.data(),
		              static_cast<int>(T), static_cast<int>(dModel));

		float* hAfterAttn_l = gpuTransformerScratch->hAfterAttn.data() + slot * T * dModel;
		gpu::add_two(hAfterAttn_l, layerIn, attnOut_l,
		             static_cast<int>(T * dModel));

		float* x2_l = gpuTransformerScratch->x2.data() + slot * T * dModel;
		float* ln2Mean_l = gpuTransformerScratch->ln2Mean.data() + layerOff;
		float* ln2InvStd_l = gpuTransformerScratch->ln2InvStd.data() + layerOff;

		if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
			gpu::rmsnorm_forward(hAfterAttn_l, gb.ln2Gamma.data(), lnEps,
			                     static_cast<int>(T), static_cast<int>(dModel),
			                     x2_l, ln2InvStd_l);
		else
			gpu::layernorm_forward(hAfterAttn_l, gb.ln2Gamma.data(), gb.ln2Beta.data(),
			                       lnEps, static_cast<int>(T), static_cast<int>(dModel),
			                       x2_l, ln2Mean_l, ln2InvStd_l);

		float* ff1_l = gpuTransformerScratch->ff1.data() + slot * T * ff1Width;
		float* ff1Act_l = gpuTransformerScratch->ff1Act.data() + slot * T * dFF;
		float* ffOut_l = gpuTransformerScratch->ffOut.data() + slot * T * dModel;

		const bool useBinaryFFNW1 = trainingConfig.transformer.binaryFFN;
		if (useBinaryFFNW1)
		{
			const bool useBitNetW1 = (dModel % 128u) == 0u;
			if (useBitNetW1)
			{
				const size_t xBits = static_cast<size_t>(T) * (dModel / 32u);
				const size_t wBits = static_cast<size_t>(ff1Width) * (dModel / 32u);
				if (gb.bitnetXBitsW1.size() < xBits) gb.bitnetXBitsW1.allocate(xBits);
				if (gb.bitnetWBitsW1.size() < wBits) gb.bitnetWBitsW1.allocate(wBits);
				if (gb.bitnetAlphaXW1.size() < T) gb.bitnetAlphaXW1.allocate(T);
				if (gb.bitnetAlphaWW1.size() < ff1Width) gb.bitnetAlphaWW1.allocate(ff1Width);
				const size_t cPop = static_cast<size_t>(T) * ff1Width;
				if (gb.bitnetCPopW1.size() < cPop) gb.bitnetCPopW1.allocate(cPop);
				if (!gpu::bitnet_ffn_forward_gpu(
				        x2_l, gb.W1.data(),
				        static_cast<int>(T), static_cast<int>(ff1Width), static_cast<int>(dModel),
				        gb.bitnetXBitsW1.data(), gb.bitnetWBitsW1.data(),
				        gb.bitnetAlphaXW1.data(), gb.bitnetAlphaWW1.data(),
				        gb.bitnetCPopW1.data(), ff1_l))
					return false;
			}
			else
			{
				if (!gpu::binary_gemm_abt_from_float(
				        x2_l, gb.W1.data(),
				        static_cast<int>(T), static_cast<int>(ff1Width), static_cast<int>(dModel),
				        ff1_l))
					return false;
			}
		}
		else
		{
			if (!gpu_gemm_abt_mp(bf16W1,
			    static_cast<int>(T), static_cast<int>(ff1Width), static_cast<int>(dModel), 1.0f,
			    x2_l, gpuTransformerScratch->activationLowp.data(), static_cast<int>(dModel),
			    gb.W1.data(), gb.W1Lowp.data(), static_cast<int>(dModel),
			    0.0f, ff1_l, static_cast<int>(ff1Width)))
				return false;
		}
		gpu::add_bias(ff1_l, gb.b1.data(),
		              static_cast<int>(T), static_cast<int>(ff1Width));

		if (ffnKind == 1u)
			gpu::swiglu_forward(ff1_l, static_cast<int>(T), static_cast<int>(dFF), ff1Act_l);
		else if (ffnAct == static_cast<int>(glades::TransformerRunConfig::FFN_GELU))
			gpu::gelu_forward(ff1_l, static_cast<int>(T * dFF), ff1Act_l);
		else
			gpu::relu_forward(ff1_l, static_cast<int>(T * dFF), ff1Act_l);

		const bool useBinaryFFN = trainingConfig.transformer.binaryFFN;
		if (useBinaryFFN)
		{
			const bool useBitNet = (dFF % 128u) == 0u;
			if (useBitNet)
			{
				const size_t xBits = static_cast<size_t>(T) * (dFF / 32u);
				const size_t wBits = static_cast<size_t>(dModel) * (dFF / 32u);
				if (gb.bitnetXBits.size() < xBits) gb.bitnetXBits.allocate(xBits);
				if (gb.bitnetWBits.size() < wBits) gb.bitnetWBits.allocate(wBits);
				if (gb.bitnetAlphaX.size() < T) gb.bitnetAlphaX.allocate(T);
				if (gb.bitnetAlphaW.size() < dModel) gb.bitnetAlphaW.allocate(dModel);
				const size_t cPop = static_cast<size_t>(T) * dModel;
				if (gb.bitnetCPop.size() < cPop) gb.bitnetCPop.allocate(cPop);
				if (!gpu::bitnet_ffn_forward_gpu(
				        ff1Act_l, gb.W2.data(),
				        static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(dFF),
				        gb.bitnetXBits.data(), gb.bitnetWBits.data(),
				        gb.bitnetAlphaX.data(), gb.bitnetAlphaW.data(),
				        gb.bitnetCPop.data(), ffOut_l))
					return false;
			}
			else
			{
				if (!gpu::binary_gemm_abt_from_float(
				        ff1Act_l, gb.W2.data(),
				        static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(dFF),
				        ffOut_l))
					return false;
			}
		}
		else
		{
			if (!gpu_gemm_abt_mp(bf16W2,
			    static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(dFF), 1.0f,
			    ff1Act_l, gpuTransformerScratch->activationLowp.data(), static_cast<int>(dFF),
			    gb.W2.data(), gb.W2Lowp.data(), static_cast<int>(dFF),
			    0.0f, ffOut_l, static_cast<int>(dModel)))
				return false;
		}
		gpu::add_bias(ffOut_l, gb.b2.data(),
		              static_cast<int>(T), static_cast<int>(dModel));

		float* hAfterFF_l = gpuTransformerScratch->hAfterFF.data() + slot * T * dModel;
		gpu::add_two(hAfterFF_l, hAfterAttn_l, ffOut_l,
		             static_cast<int>(T * dModel));
	}
	return true;
}

#endif // GLADES_HAVE_CUDA

void glades::NNetwork::transformerGpuTrainEpoch(const TransformerEpochCfg& cfg, unsigned int seqCount,
                                                int epochIdx, int64_t epochStartMs,
                                                unsigned long long& tokensProcessed,
                                                unsigned long long& targetsProcessed,
                                                double& tokenLmNllSum,
                                                unsigned long long& tokenLmTokenCount,
                                                unsigned long long& clsCorrect,
                                                unsigned long long& clsTotal,
                                                shmea::GLogger* logger)
{
	using namespace glades::logfmt;

	TensorTransformerState& tt = tensorTransformer;
	const unsigned int dModel = cfg.dModel;
	const unsigned int dFF = cfg.dFF;
	const unsigned int nHeads = cfg.nHeads;
	const unsigned int nKVHeads = cfg.nKVHeads;
	const unsigned int nLayers = cfg.nLayers;
	const unsigned int vocabSize = cfg.vocabSize;
	const unsigned int inputSize = cfg.inputSize;
	const unsigned int outSize = cfg.outSize;
	const bool tokenLM = cfg.tokenLM;
	const bool tieEmb = cfg.tieEmb;
	const bool causal = cfg.causal;
	const int padTokenId = cfg.padTokenId;
	const int posEnc = cfg.posEnc;
	const int normType = cfg.normType;
	const int ffnKind = cfg.ffnKind;
	const int ffnAct = cfg.ffnAct;
	const float lnEps = cfg.lnEps;
	const int ropeDimOverride = cfg.ropeDimOverride;
	const unsigned int seqBatchMax = cfg.seqBatchMax;
	const unsigned int optimizerStepsPerEpoch = (seqCount + seqBatchMax - 1u) / seqBatchMax;

	unsigned int seqInBatch = 0u;
	unsigned int timeStepsInBatch = 0u;

	// Progress logging setup
	unsigned int progressEverySeq = 1u;
	if (seqCount > 20u)
		progressEverySeq = seqCount / 20u;
	if (progressEverySeq == 0u)
		progressEverySeq = 1u;
	int64_t lastProgressMs = epochStartMs;
	static const int64_t kProgressIntervalMs = 5000;

	if (nHeads == 0u || (dModel % nHeads) != 0u)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerGpuTrainEpoch: dModel is not divisible by nHeads");
		storeRunningFlag(false);
		return;
	}
	if ((nHeads % nKVHeads) != 0u)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "transformerGpuTrainEpoch: nHeads is not divisible by nKVHeads");
		storeRunningFlag(false);
		return;
	}
	const unsigned int dHead = dModel / nHeads;
	const unsigned int dModelKV = nKVHeads * dHead;
	const unsigned int ff1Width = (ffnKind == 1) ? (2u * dFF) : dFF;
	const bool useRope = (posEnc == static_cast<int>(glades::TransformerRunConfig::POSENC_ROPE));

	// BF16 mixed precision. Enabled when the outer config sets
	// mixedPrecision.enable and weightDType == BF16; per-site flags below
	// control which forward GEMMs actually take the BF16 path during
	// incremental rollout (initialized from env var GLADES_BF16_SITES —
	// see helper comment in gpu_gemm_*_mp for site numbers).
	const bool useBf16 = cfg.mpEnable && gpuTransformerWeights
	    && (trainingConfig.mixedPrecision.weightDType ==
	        glades::MixedPrecisionConfig::WEIGHT_BF16);
	// Per-site BF16 enable bitmask. Default when useBf16 is true is 0xFF
	// (all forward GEMMs take the BF16 path). GLADES_BF16_SITES env var
	// overrides for per-site rollout / parity debugging.
	// Bits: 0=WIn, 1=Wq, 2=Wk, 3=Wv, 4=Wo, 5=W1, 6=W2, 7=tied-head.
	const char* bf16SitesEnv = std::getenv("GLADES_BF16_SITES");
	unsigned int bf16SitesMask = useBf16 ? 0xFFu : 0u;
	if (bf16SitesEnv)
		bf16SitesMask = static_cast<unsigned int>(strtoul(bf16SitesEnv, NULL, 0));
	// Site bits: 0=WIn, 1=Wq, 2=Wk, 3=Wv, 4=Wo, 5=W1, 6=W2, 7=tied-head.
	const bool bf16WIn  = useBf16 && (bf16SitesMask & 0x01u);
	const bool bf16Wq   = useBf16 && (bf16SitesMask & 0x02u);
	const bool bf16Wk   = useBf16 && (bf16SitesMask & 0x04u);
	const bool bf16Wv   = useBf16 && (bf16SitesMask & 0x08u);
	const bool bf16Wo   = useBf16 && (bf16SitesMask & 0x10u);
	const bool bf16W1   = useBf16 && (bf16SitesMask & 0x20u);
	const bool bf16W2   = useBf16 && (bf16SitesMask & 0x40u);
	const bool bf16Head = useBf16 && (bf16SitesMask & 0x80u);
	if (cfg.mpEnable && gpuTransformerWeights
	    && (trainingConfig.mixedPrecision.weightDType ==
	        glades::MixedPrecisionConfig::WEIGHT_BF16))
	{
		gpuTransformerWeights->lowpDType = glades::transformer_kernels::LOWP_BF16;
		if (!gpuTransformerWeights->ensureLowpMirrors())
		{
			lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
			    "transformerGpuTrainEpoch: failed to build BF16 weight mirrors");
			storeRunningFlag(false);
			return;
		}
		// bf16-weights mode: now that the bf16 mirrors are populated, free the
		// FP32 weight masters to recover ~3.7 GB at 1.84B.  Guarded against
		// paradigm-#74 binary FFN (which reads gb.W1/W2 FP32 directly), atlas
		// optimizer (which reads gb.Wq/etc FP32 in atlas_gpu_update), and
		// MLA (Wdkv/Wuk/Wuv reads remain FP32 — those are NOT freed by
		// freeFp32Masters).  In Stage 7c default usage this means
		// `--bf16-weights` without `--binary-ffn` and with Adam (not atlas).
		if (trainingConfig.mixedPrecision.weightStorageBf16
		    && !trainingConfig.transformer.binaryFFN
		    && trainingConfig.optimizer.type != glades::OptimizerConfig::ATLAS)
		{
			gpuTransformerWeights->freeFp32Masters();
		}
	}

	TransformerGpuPerfBreakdown* gpuPerf =
	    ((transformerMetricsCfg.enable && transformerMetricsCfg.enableGpuPerf) ? &lastTransformerTrainGpuPerf : NULL);
	if (gpuPerf)
		gpuPerf->reset();
	const bool gpuUseAtlasConfig =
	    (trainingConfig.optimizer.type == glades::OptimizerConfig::ATLAS);
	const bool gpuFuseEchoConfig =
	    gpuUseAtlasConfig
	    && trainingConfig.atlas.echoEnabled
	    && !trainingConfig.atlas.geodeEnabled
	    && !trainingConfig.atlas.bimapEnabled
	    && !trainingConfig.atlas.pactEnabled
	    && !trainingConfig.atlas.racerEnabled
	    && !trainingConfig.atlas.matraEnabled
	    && !trainingConfig.atlas.argosEnabled
	    && !trainingConfig.atlas.muonEnabled;
	std::vector<glades::gpu::GpuEchoObserveEntry> echoObserveEntries;
	echoObserveEntries.reserve(static_cast<size_t>(2u + 6u * nLayers));
	glades::gpu::ScopedPerfTimerMs gpuTotal(gpuPerf ? &gpuPerf->msTotal : NULL);
	cudaEvent_t gpuTransferReadyEvent = gpu::createEvent(false);
	cudaEvent_t gpuComputeReadyEvent = gpu::createEvent(false);

	// HELIOS FD-HVP probe state: last-sequence metadata captured per-iteration
	// and used at the minibatch-boundary optimizer dispatch for the scalar-FD
	// loss evaluation (L_plus, L_minus) against L_0 = lastSeqLoss0. See the
	// gpuUseHelios branch below for the probe protocol. Only populated on
	// sequences that ran the tokenLM metrics path.
	float heliosProbeLastSeqLoss0 = 0.0f;
	unsigned int heliosProbeLastSeqT = 0u;
	bool heliosProbeLastSeqValid = false;

	for (unsigned int s = 0; s < seqCount; ++s)
	{
		if (!loadRunningFlag())
			break;

		const unsigned int T = di->getTrainSequenceLength(s);
		if (T == 0u)
			continue;
		echoObserveEntries.clear();
		int echoObserveTotalFeatures = 0;
		const bool gpuBatchEchoObserve =
		    gpuFuseEchoConfig
		    && trainingConfig.atlas.echoShouldRefresh(tensorTransformer.optimizerStep + 1ULL);
		const bool rebuildEchoObserveMeta =
		    gpuBatchEchoObserve
		    && (!gpuTransformerWeights->echoObserveMetaUploaded
		        || gpuTransformerWeights->echoObserveScope != trainingConfig.atlas.echoScope
		        || gpuTransformerWeights->echoObserveSeqLen != T
		        || gpuTransformerWeights->echoObserveTokenModel != tokenLM);
		tokensProcessed += static_cast<unsigned long long>(T);

			// Ensure GPU scratch ownership/capacity before any per-sequence uploads.
			if (!ensureTransformerGpuTrainingScratch(cfg, T))
			{
				// GPU scratch allocation failed, stop the GPU epoch cleanly.
				break;
			}

		// Upload token IDs for this sequence.
		if (tokenLM)
		{
			glades::gpu::ScopedPerfTimerMs gpuStage(gpuPerf ? &gpuPerf->msEmbed : NULL);
			std::vector<int> tokenIdsInt(T);
			for (unsigned int t = 0; t < T; ++t)
			{
				int tid = 0;
				di->getTrainSequenceTokenId(s, t, tid);
				tokenIdsInt[t] = tid;
			}
			if (gpuPerf)
			{
				gpu::perfRecordBytesH2D(&gpuPerf->counters, static_cast<size_t>(T) * sizeof(int));
				gpu::perfRecordSync(&gpuPerf->counters, 1u);
				gpu::perfRecordKernel(&gpuPerf->counters, 1u);
			}
			gpu::uploadTransformerTokenIds(*gpuTransformerScratch, &tokenIdsInt[0], T);
			gpu::recordEvent(gpuTransferReadyEvent, gpu::transferStream());
			gpu::streamWaitEvent(gpu::computeStream(), gpuTransferReadyEvent);

			// Forward: embedding gather.  Under bf16-weights mode the FP32
			// master may be retired (gb.tokE.size() == 0) and the canonical
			// store is the bf16 mirror tokELowp; route through the bf16
			// gather variant.  When FP32 master is alive (default), prefer
			// the FP32 path for bit-exact behavior with the prior code.
			if (gpuTransformerWeights->tokE.size() == 0
			    && gpuTransformerWeights->tokELowp.allocated())
			{
				gpu::embedding_gather_bf16(
				    gpuTransformerWeights->tokELowp.data(),
				    gpuTransformerScratch->tokenIds.data(),
				    static_cast<int>(T), static_cast<int>(vocabSize),
				    static_cast<int>(dModel),
				    gpuTransformerScratch->h.data());
			}
			else
			{
				gpu::embedding_gather(
				    gpuTransformerWeights->tokE.data(),
				    gpuTransformerScratch->tokenIds.data(),
				    static_cast<int>(T), static_cast<int>(vocabSize),
				    static_cast<int>(dModel),
				    gpuTransformerScratch->h.data());
			}
		}
		else
		{
			glades::gpu::ScopedPerfTimerMs gpuStage(gpuPerf ? &gpuPerf->msEmbed : NULL);
			// Upload input features and run linear projection.
			std::vector<float> xHost(static_cast<size_t>(T) * inputSize);
			for (unsigned int t = 0; t < T; ++t)
			{
				const float* row = NULL;
				unsigned int rowSize = 0u;
				di->getTrainSequenceRowView(s, t, row, rowSize);
				const size_t off = static_cast<size_t>(t) * inputSize;
				for (unsigned int f = 0; f < inputSize; ++f)
					xHost[off + f] = (row && f < rowSize) ? row[f] : 0.0f;
			}
			if (gpuPerf)
			{
				gpu::perfRecordBytesH2D(&gpuPerf->counters,
				                        static_cast<size_t>(T) * static_cast<size_t>(inputSize) * sizeof(float));
				gpu::perfRecordSync(&gpuPerf->counters, 1u);
				gpu::perfRecordKernel(&gpuPerf->counters, 2u);
			}
			gpu::uploadTransformerDenseInputs(*gpuTransformerScratch, &xHost[0], xHost.size());
			gpu::recordEvent(gpuTransferReadyEvent, gpu::transferStream());
			gpu::streamWaitEvent(gpu::computeStream(), gpuTransferReadyEvent);

			// Input projection: h = x * WIn^T + bIn
			gpu_gemm_abt_mp(bf16WIn,
			    static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(inputSize),
			    1.0f,
			    gpuTransformerScratch->x.data(),
			    gpuTransformerScratch->activationLowp.data(),
			    static_cast<int>(inputSize),
			    gpuTransformerWeights->WIn.data(),
			    gpuTransformerWeights->WInLowp.data(),
			    static_cast<int>(inputSize),
			    0.0f,
			    gpuTransformerScratch->h.data(), static_cast<int>(dModel));
			gpu::add_bias(gpuTransformerScratch->h.data(),
			              gpuTransformerWeights->bIn.data(),
			              static_cast<int>(T), static_cast<int>(dModel));
		}

		// Upload RoPE invFreq to scratch (shared across all layers).
		unsigned int fwdRopeHalfDim = 0u;
		if (useRope && !transformerPosEncCache.ropeInvFreq.empty())
		{
			glades::gpu::ScopedPerfTimerMs gpuStage(gpuPerf ? &gpuPerf->msPosEnc : NULL);
			const unsigned int rd = (ropeDimOverride > 0 && static_cast<unsigned int>(ropeDimOverride) < dHead)
			                        ? static_cast<unsigned int>(ropeDimOverride) : dHead;
			fwdRopeHalfDim = rd / 2u;
			std::vector<float> invFreqF(fwdRopeHalfDim);
			for (unsigned int i = 0; i < fwdRopeHalfDim && i < transformerPosEncCache.ropeInvFreq.size(); ++i)
				invFreqF[i] = static_cast<float>(transformerPosEncCache.ropeInvFreq[i]);
			if (gpuPerf)
			{
				gpu::perfRecordBytesH2D(&gpuPerf->counters, invFreqF.size() * sizeof(float));
				gpu::perfRecordSync(&gpuPerf->counters, 1u);
			}
			gpu::uploadTransformerRopeInvFreq(*gpuTransformerScratch, &invFreqF[0], invFreqF.size());
			gpu::recordEvent(gpuTransferReadyEvent, gpu::transferStream());
		}

		// Per-layer transformer blocks.
		const unsigned int slotsPerLayerFwd = gpuTransformerScratch->slotsPerLayer
		    ? gpuTransformerScratch->slotsPerLayer : nLayers;
		for (unsigned int li = 0; li < nLayers; ++li)
		{
			gpu::GpuTransformerWeights::Block& gb = gpuTransformerWeights->blocks[li];
			// Activation-checkpoint: per-layer activations live in cyclic slots
			// (slot = li % K).  When checkpointing is off, slotsPerLayer == nLayers
			// and modulo collapses to identity.
			const size_t slot = static_cast<size_t>(li % slotsPerLayerFwd);
			const size_t prevSlot = (li > 0u)
			    ? static_cast<size_t>((li - 1u) % slotsPerLayerFwd)
			    : 0u;
			const size_t layerOff = slot * static_cast<size_t>(T);

			// Input to this layer is h (or hAfterFF from previous layer).
			const float* layerIn = (li == 0) ? gpuTransformerScratch->h.data()
			                                 : (gpuTransformerScratch->hAfterFF.data() + prevSlot * T * dModel);

			float* x1_l = gpuTransformerScratch->x1.data() + slot * T * dModel;
			float* ln1Mean_l = gpuTransformerScratch->ln1Mean.data() + layerOff;
			float* ln1InvStd_l = gpuTransformerScratch->ln1InvStd.data() + layerOff;

			// Pre-LN 1
			if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
			{
				gpu::rmsnorm_forward(layerIn, gb.ln1Gamma.data(), lnEps,
				                      static_cast<int>(T), static_cast<int>(dModel),
				                      x1_l, ln1InvStd_l);
			}
			else
			{
				gpu::layernorm_forward(layerIn, gb.ln1Gamma.data(), gb.ln1Beta.data(),
				                        lnEps, static_cast<int>(T), static_cast<int>(dModel),
				                        x1_l, ln1Mean_l, ln1InvStd_l);
			}

			// QKV projections
			float* Q_l = gpuTransformerScratch->Q.data() + slot * T * dModel;
			float* K_l = gpuTransformerScratch->K.data() + slot * T * dModelKV;
			float* V_l = gpuTransformerScratch->V.data() + slot * T * dModelKV;

			gpu_gemm_abt_mp(bf16Wq,
			    static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(dModel), 1.0f,
			    x1_l, gpuTransformerScratch->activationLowp.data(), static_cast<int>(dModel),
			    gb.Wq.data(), gb.WqLowp.data(), static_cast<int>(dModel),
			    0.0f, Q_l, static_cast<int>(dModel));
			gpu::add_bias(Q_l, gb.bq.data(), static_cast<int>(T), static_cast<int>(dModel));

			// Paradigm shift #76 MLA dispatch (training path): when
			// mlaLatentDim > 0, replace the standard W_K, W_V projections
			// with the low-rank latent path:
			//   c = x1 @ Wdkv ; K = c @ Wuk ; V = c @ Wuv
			// Mirrors the dispatch in transformerGpuRunForwardOnly.
			{
			const int mlaDc = trainingConfig.transformer.mlaLatentDim;
			if (mlaDc > 0 && gb.Wdkv.allocated()) {
				const size_t cBytes = static_cast<size_t>(T) * static_cast<size_t>(mlaDc);
				if (gb.mlaC.size() < cBytes) gb.mlaC.allocate(cBytes);
				if (!gpu::mla_attention_forward_gpu(
				        x1_l, gb.Wdkv.data(), gb.Wuk.data(), gb.Wuv.data(),
				        static_cast<int>(T), static_cast<int>(dModel), mlaDc,
				        static_cast<int>(dModelKV),
				        gb.mlaC.data(), K_l, V_l))
					return;
				gpu::add_bias(K_l, gb.bk.data(), static_cast<int>(T), static_cast<int>(dModelKV));
				gpu::add_bias(V_l, gb.bv.data(), static_cast<int>(T), static_cast<int>(dModelKV));
			} else {
				gpu_gemm_abt_mp(bf16Wk,
				    static_cast<int>(T), static_cast<int>(dModelKV), static_cast<int>(dModel), 1.0f,
				    x1_l, gpuTransformerScratch->activationLowp.data(), static_cast<int>(dModel),
				    gb.Wk.data(), gb.WkLowp.data(), static_cast<int>(dModel),
				    0.0f, K_l, static_cast<int>(dModelKV));
				gpu::add_bias(K_l, gb.bk.data(), static_cast<int>(T), static_cast<int>(dModelKV));

				gpu_gemm_abt_mp(bf16Wv,
				    static_cast<int>(T), static_cast<int>(dModelKV), static_cast<int>(dModel), 1.0f,
				    x1_l, gpuTransformerScratch->activationLowp.data(), static_cast<int>(dModel),
				    gb.Wv.data(), gb.WvLowp.data(), static_cast<int>(dModel),
				    0.0f, V_l, static_cast<int>(dModelKV));
				gpu::add_bias(V_l, gb.bv.data(), static_cast<int>(T), static_cast<int>(dModelKV));
			}
			}

			// RoPE (if enabled) — fused Q+K in single kernel launch
			if (useRope && !transformerPosEncCache.ropeInvFreq.empty())
			{
				if (li == 0u)
					gpu::streamWaitEvent(gpu::computeStream(), gpuTransferReadyEvent);
				const unsigned int rd = (ropeDimOverride > 0 && static_cast<unsigned int>(ropeDimOverride) < dHead)
				                        ? static_cast<unsigned int>(ropeDimOverride) : dHead;
				// Use persistent gpuInvFreq from scratch (uploaded before layer loop).
				gpu::rope_apply_qk(Q_l, K_l, gpuTransformerScratch->gpuInvFreq.data(),
				                    static_cast<int>(T), static_cast<int>(nHeads),
				                    static_cast<int>(nKVHeads), static_cast<int>(dHead),
				                    static_cast<int>(rd / 2u));
			}

			// Flash-style packed multi-head attention without materializing T*T scores/probs.
			float* attnConcat_l = gpuTransformerScratch->attnConcat.data() + slot * T * dModel;
			if (useBf16)
			{
				// Cast Q/K/V to BF16 scratches and invoke the BF16 flash
				// kernel. Halves the attention memory bandwidth (dominant
				// at long seq) and leaves softmax/accumulation in FP32 for
				// numerical stability.
				glades::gpu::cast_f32_to_bf16(Q_l,
				    gpuTransformerScratch->qLowp.data(),
				    static_cast<size_t>(T) * dModel);
				glades::gpu::cast_f32_to_bf16(K_l,
				    gpuTransformerScratch->kLowp.data(),
				    static_cast<size_t>(T) * dModelKV);
				glades::gpu::cast_f32_to_bf16(V_l,
				    gpuTransformerScratch->vLowp.data(),
				    static_cast<size_t>(T) * dModelKV);
				// cuBLAS-tiled BF16 path when eligible (no GQA, scratch ok).
				// Q/K/V are already cast to BF16 above (qLowp/kLowp/vLowp)
				// so we pass them in via a dummy FP32 pointer that gets
				// ignored — use the BF16 variant which takes the cast pointers.
				// Falls back to the existing custom BF16 flash_attention
				// kernel when scratch alloc fails or GQA is configured.
				const bool chiron_fast_bf16 = (nHeads == nKVHeads);
				bool bf16_attn_done = false;
				if (chiron_fast_bf16)
				{
					const size_t scoresNeeded = static_cast<size_t>(nHeads) * T * T;
					if (gpuTransformerScratch->attnScoresScratch.size() < scoresNeeded)
						gpuTransformerScratch->attnScoresScratch.allocate(scoresNeeded);
					if (gpuTransformerScratch->attnPbf16.size() < scoresNeeded)
						gpuTransformerScratch->attnPbf16.allocate(scoresNeeded);
					if (gpuTransformerScratch->attnScoresScratch.size() >= scoresNeeded &&
					    gpuTransformerScratch->attnPbf16.size() >= scoresNeeded)
					{
						// Re-use existing qLowp/kLowp/vLowp BF16 scratch as inputs.
						// Call the BF16 variant but have it skip its internal cast
						// by passing the BF16 scratches as the scratch_*bf16 outputs
						// — we'll call a dedicated "from-bf16" inline implementation
						// to avoid an unneeded cast.  For now, just call the variant
						// that re-casts — cast overhead is small vs attention compute
						// at real production shapes.
						bf16_attn_done = gpu::flash_attention_cublas_tiled_bf16(
						    Q_l, K_l, V_l,
						    static_cast<int>(T), static_cast<int>(nHeads),
						    static_cast<int>(dHead), static_cast<int>(dModel),
						    causal, attnConcat_l,
						    gpuTransformerScratch->attnScoresScratch.data(),
						    gpuTransformerScratch->qLowp.data(),  // reuse
						    gpuTransformerScratch->kLowp.data(),  // reuse
						    gpuTransformerScratch->vLowp.data(),  // reuse
						    gpuTransformerScratch->attnPbf16.data());
					}
				}
				if (!bf16_attn_done)
				{
				const int localW_tr = trainingConfig.transformer.localAttnWindow;
				const int sinkS_tr = trainingConfig.transformer.attnSinkCount > 0
				    ? trainingConfig.transformer.attnSinkCount : 0;
				if ((localW_tr > 0 && localW_tr < static_cast<int>(T)) || sinkS_tr > 0) {
				gpu::flash_attention_multihead_forward_bf16_local(
				    gpuTransformerScratch->qLowp.data(),
				    gpuTransformerScratch->kLowp.data(),
				    gpuTransformerScratch->vLowp.data(),
				    static_cast<int>(T),
				    static_cast<int>(nHeads),
				    static_cast<int>(nKVHeads),
				    static_cast<int>(dHead),
				    static_cast<int>(dModel),
				    static_cast<int>(dModelKV),
				    causal,
				    localW_tr,
				    attnConcat_l,
				    sinkS_tr);
				} else {
				gpu::flash_attention_multihead_forward_bf16(
				    gpuTransformerScratch->qLowp.data(),
				    gpuTransformerScratch->kLowp.data(),
				    gpuTransformerScratch->vLowp.data(),
				    static_cast<int>(T),
				    static_cast<int>(nHeads),
				    static_cast<int>(nKVHeads),
				    static_cast<int>(dHead),
				    static_cast<int>(dModel),
				    static_cast<int>(dModelKV),
				    causal,
				    attnConcat_l);
				}
				}
			}
			else
			{
				// cuBLAS-tiled tensor-core attention path (40-46x faster
				// than the custom flash_attention kernel — see
				// research/WMMA_ATTENTION_PLAN.md).  Eligible only when
				// no GQA (nHeads == nKVHeads); falls back otherwise.
				const bool chiron_fast_attn = (nHeads == nKVHeads);
				if (chiron_fast_attn)
				{
					const size_t scoresNeeded = static_cast<size_t>(nHeads) * T * T;
					if (gpuTransformerScratch->attnScoresScratch.size() < scoresNeeded)
						gpuTransformerScratch->attnScoresScratch.allocate(scoresNeeded);
					// BF16 tensor-core path (opt-in): 2x GEMM throughput but
					// cast overhead dominates at small dHead.  Benefits larger
					// shapes (T>=2048, dHead>=128).  Gate via GLADES_CHIRON_ATTN=bf16.
					// Default "fp32" (or unset) keeps the proven 5.8x fp32 cuBLAS path.
					static const char* s_attnMode = std::getenv("GLADES_CHIRON_ATTN");
					const bool want_bf16 = s_attnMode && std::strcmp(s_attnMode, "bf16") == 0;
					bool bf16_ok = false;
					if (want_bf16)
					{
						const size_t qkvN = static_cast<size_t>(T) * dModel;
						if (gpuTransformerScratch->attnQbf16.size() < qkvN)
							gpuTransformerScratch->attnQbf16.allocate(qkvN);
						if (gpuTransformerScratch->attnKbf16.size() < qkvN)
							gpuTransformerScratch->attnKbf16.allocate(qkvN);
						if (gpuTransformerScratch->attnVbf16.size() < qkvN)
							gpuTransformerScratch->attnVbf16.allocate(qkvN);
						if (gpuTransformerScratch->attnPbf16.size() < scoresNeeded)
							gpuTransformerScratch->attnPbf16.allocate(scoresNeeded);
						bf16_ok =
						    gpuTransformerScratch->attnQbf16.size() >= qkvN &&
						    gpuTransformerScratch->attnKbf16.size() >= qkvN &&
						    gpuTransformerScratch->attnVbf16.size() >= qkvN &&
						    gpuTransformerScratch->attnPbf16.size() >= scoresNeeded &&
						    gpuTransformerScratch->attnScoresScratch.size() >= scoresNeeded;
					}
					if (bf16_ok)
					{
						gpu::flash_attention_cublas_tiled_bf16(
						    Q_l, K_l, V_l,
						    static_cast<int>(T), static_cast<int>(nHeads),
						    static_cast<int>(dHead), static_cast<int>(dModel),
						    causal, attnConcat_l,
						    gpuTransformerScratch->attnScoresScratch.data(),
						    gpuTransformerScratch->attnQbf16.data(),
						    gpuTransformerScratch->attnKbf16.data(),
						    gpuTransformerScratch->attnVbf16.data(),
						    gpuTransformerScratch->attnPbf16.data());
					}
					else if (gpuTransformerScratch->attnScoresScratch.size() >= scoresNeeded)
					{
						gpu::flash_attention_cublas_tiled(Q_l, K_l, V_l,
						    static_cast<int>(T), static_cast<int>(nHeads),
						    static_cast<int>(dHead), static_cast<int>(dModel),
						    causal, attnConcat_l,
						    gpuTransformerScratch->attnScoresScratch.data());
					}
					else
					{
						gpu::flash_attention_multihead_forward(
						    Q_l, K_l, V_l,
						    static_cast<int>(T),
						    static_cast<int>(nHeads),
						    static_cast<int>(nKVHeads),
						    static_cast<int>(dHead),
						    static_cast<int>(dModel),
						    static_cast<int>(dModelKV),
						    causal,
						    attnConcat_l);
					}
				}
				else
				{
				gpu::flash_attention_multihead_forward(
				    Q_l, K_l, V_l,
				    static_cast<int>(T),
				    static_cast<int>(nHeads),
				    static_cast<int>(nKVHeads),
				    static_cast<int>(dHead),
				    static_cast<int>(dModel),
				    static_cast<int>(dModelKV),
				    causal,
				    attnConcat_l);
				}
			}

			// Wo projection
			float* attnOut_l = gpuTransformerScratch->attnOut.data() + slot * T * dModel;
			gpu_gemm_abt_mp(bf16Wo,
			    static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(dModel), 1.0f,
			    attnConcat_l, gpuTransformerScratch->activationLowp.data(), static_cast<int>(dModel),
			    gb.Wo.data(), gb.WoLowp.data(), static_cast<int>(dModel),
			    0.0f, attnOut_l, static_cast<int>(dModel));
			gpu::add_bias(attnOut_l, gb.bo.data(), static_cast<int>(T), static_cast<int>(dModel));

			// Residual 1: hAfterAttn = layerIn + attnOut
			float* hAfterAttn_l = gpuTransformerScratch->hAfterAttn.data() + slot * T * dModel;
			gpu::add_two(hAfterAttn_l, layerIn, attnOut_l, static_cast<int>(T * dModel));

			// Pre-LN 2
			float* x2_l = gpuTransformerScratch->x2.data() + slot * T * dModel;
			float* ln2Mean_l = gpuTransformerScratch->ln2Mean.data() + layerOff;
			float* ln2InvStd_l = gpuTransformerScratch->ln2InvStd.data() + layerOff;

			if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
			{
				gpu::rmsnorm_forward(hAfterAttn_l, gb.ln2Gamma.data(), lnEps,
				                      static_cast<int>(T), static_cast<int>(dModel),
				                      x2_l, ln2InvStd_l);
			}
			else
			{
				gpu::layernorm_forward(hAfterAttn_l, gb.ln2Gamma.data(), gb.ln2Beta.data(),
				                        lnEps, static_cast<int>(T), static_cast<int>(dModel),
				                        x2_l, ln2Mean_l, ln2InvStd_l);
			}

			// FFN
			float* ff1_l = gpuTransformerScratch->ff1.data() + slot * T * ff1Width;
			float* ff1Act_l = gpuTransformerScratch->ff1Act.data() + slot * T * dFF;
			float* ffOut_l = gpuTransformerScratch->ffOut.data() + slot * T * dModel;

			// FF1: x2 * W1^T + b1
			gpu_gemm_abt_mp(bf16W1,
			    static_cast<int>(T), static_cast<int>(ff1Width), static_cast<int>(dModel), 1.0f,
			    x2_l, gpuTransformerScratch->activationLowp.data(), static_cast<int>(dModel),
			    gb.W1.data(), gb.W1Lowp.data(), static_cast<int>(dModel),
			    0.0f, ff1_l, static_cast<int>(ff1Width));
			gpu::add_bias(ff1_l, gb.b1.data(), static_cast<int>(T), static_cast<int>(ff1Width));

			// Activation
			if (ffnKind == 1) // SwiGLU
			{
				gpu::swiglu_forward(ff1_l, static_cast<int>(T), static_cast<int>(dFF), ff1Act_l);
			}
			else if (ffnAct == static_cast<int>(glades::TransformerRunConfig::FFN_GELU))
			{
				gpu::gelu_forward(ff1_l, static_cast<int>(T * dFF), ff1Act_l);
			}
			else
			{
				gpu::relu_forward(ff1_l, static_cast<int>(T * dFF), ff1Act_l);
			}

			// FF2: ffAct * W2^T + b2
			gpu_gemm_abt_mp(bf16W2,
			    static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(dFF), 1.0f,
			    ff1Act_l, gpuTransformerScratch->activationLowp.data(), static_cast<int>(dFF),
			    gb.W2.data(), gb.W2Lowp.data(), static_cast<int>(dFF),
			    0.0f, ffOut_l, static_cast<int>(dModel));
			gpu::add_bias(ffOut_l, gb.b2.data(), static_cast<int>(T), static_cast<int>(dModel));

			// Residual 2: hAfterFF = hAfterAttn + ffOut
			float* hAfterFF_l = gpuTransformerScratch->hAfterFF.data() + slot * T * dModel;
			gpu::add_two(hAfterFF_l, hAfterAttn_l, ffOut_l, static_cast<int>(T * dModel));

			// Activation-checkpoint: stash hAfterFF at every Kth layer boundary
			// (except the very last layer, whose output feeds final LN directly).
			// checkpoints[c] holds the input to segment c+1 = hAfterFF after layer
			// (c+1)*K - 1.  Skipped when checkpointing is off (nCheckpoints==0).
			if (gpuTransformerScratch->nCheckpoints > 0u
			    && ((li + 1u) % slotsPerLayerFwd) == 0u
			    && (li + 1u) < nLayers)
			{
				const unsigned int ckptIdx = (li + 1u) / slotsPerLayerFwd - 1u;
				if (ckptIdx < gpuTransformerScratch->nCheckpoints)
				{
					gpu::device_memcpy_d2d(
					    gpuTransformerScratch->checkpoints.data()
					        + static_cast<size_t>(ckptIdx) * T * dModel,
					    hAfterFF_l,
					    static_cast<size_t>(T) * dModel * sizeof(float));
				}
			}
		}

		// Final LayerNorm.  In activation-checkpoint mode the last layer's
		// hAfterFF lives in slot ((nLayers - 1) % slotsPerLayer); otherwise
		// modulo collapses to (nLayers - 1).
		const float* finalH = gpuTransformerScratch->hAfterFF.data() + static_cast<size_t>((nLayers - 1) % slotsPerLayerFwd) * T * dModel;
		float* hPostFinalLN = gpuTransformerScratch->hPostFinalLN.data();
		if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
		{
			gpu::rmsnorm_forward(finalH, gpuTransformerWeights->lnFinalGamma.data(), lnEps,
			                     static_cast<int>(T), static_cast<int>(dModel),
			                     hPostFinalLN, gpuTransformerScratch->lnFinalInvStd.data());
		}
		else
		{
			gpu::layernorm_forward(finalH, gpuTransformerWeights->lnFinalGamma.data(),
			                       gpuTransformerWeights->lnFinalBeta.data(), lnEps,
			                       static_cast<int>(T), static_cast<int>(dModel),
			                       hPostFinalLN, gpuTransformerScratch->lnFinalMean.data(),
			                       gpuTransformerScratch->lnFinalInvStd.data());
		}

		// Output logits
		if (tokenLM && tieEmb)
		{
			// logits = hPostFinalLN * E^T + lmBias
			gpu_gemm_abt_mp(bf16Head,
			    static_cast<int>(T), static_cast<int>(vocabSize), static_cast<int>(dModel), 1.0f,
			    hPostFinalLN, gpuTransformerScratch->activationLowp.data(), static_cast<int>(dModel),
			    gpuTransformerWeights->tokE.data(), gpuTransformerWeights->tokELowp.data(),
			    static_cast<int>(dModel),
			    0.0f, gpuTransformerScratch->logits.data(), static_cast<int>(vocabSize));
			gpu::add_bias(gpuTransformerScratch->logits.data(),
			              gpuTransformerWeights->lmBias.data(),
			              static_cast<int>(T), static_cast<int>(vocabSize));
		}

		// Softmax
		gpu::softmax_forward(gpuTransformerScratch->logits.data(),
		                      static_cast<int>(T), static_cast<int>(outSize),
		                      gpuTransformerScratch->probs.data());

		// === Loss / metrics ===
		std::vector<int> gpuTargetIds;
		unsigned int gpuValidTargets = 0u;
		if (tokenLM)
		{
			glades::gpu::ScopedPerfTimerMs gpuStage(gpuPerf ? &gpuPerf->msLoss : NULL);
			gpuTargetIds.resize(T);
			for (unsigned int t = 0; t < T; ++t)
			{
				int yid = padTokenId;
				di->getTrainSequenceExpectedTokenId(s, t, yid);
				gpuTargetIds[t] = yid;
			}
			// Upload targets to persistent scratch buffer.
			if (gpuPerf)
			{
				gpu::perfRecordBytesH2D(&gpuPerf->counters, static_cast<size_t>(T) * sizeof(int));
				gpu::perfRecordSync(&gpuPerf->counters, 2u);
				gpu::perfRecordKernel(&gpuPerf->counters, 1u);
			}
			gpuTransformerScratch->gpuTargetsT.uploadAsync(&gpuTargetIds[0], T);
			gpu::recordEvent(gpuTransferReadyEvent, gpu::transferStream());
			gpu::streamWaitEvent(gpu::computeStream(), gpuTransferReadyEvent);

			gpu::collect_token_lm_metrics(
			    gpuTransformerScratch->probs.data(),
			    gpuTransformerScratch->gpuTargetsT.data(),
			    static_cast<int>(T), static_cast<int>(vocabSize),
			    padTokenId,
			    gpuTransformerScratch->lossPack.data());
			int lossPacked[4];
			if (gpuPerf)
				gpu::perfRecordBytesD2H(&gpuPerf->counters, 4u * sizeof(int));
			gpu::recordEvent(gpuComputeReadyEvent, gpu::computeStream());
			gpu::streamWaitEvent(gpu::transferStream(), gpuComputeReadyEvent);
			gpuTransformerScratch->lossPack.downloadAsync(lossPacked, 4);
			gpu::synchronizeTransferStream();
			float lossVal;
			memcpy(&lossVal, &lossPacked[0], sizeof(float));
			int lossCountVal = lossPacked[1], correctVal = lossPacked[2], validVal = lossPacked[3];

			// Capture L_0 for the HELIOS FD-HVP scalar-FD probe. This value
			// is the per-sequence NLL sum at unperturbed weights; the probe
			// at minibatch boundary will compare it to L_plus and L_minus
			// obtained with perturbed weights via transformerGpuRunForwardOnly.
			heliosProbeLastSeqLoss0 = lossVal;
			heliosProbeLastSeqT = T;
			heliosProbeLastSeqValid = true;

			gpuValidTargets = static_cast<unsigned int>(lossCountVal);
			tokenLmNllSum += static_cast<double>(lossVal);
			tokenLmTokenCount += static_cast<unsigned long long>(lossCountVal);
			clsCorrect += static_cast<unsigned long long>(correctVal);
			clsTotal += static_cast<unsigned long long>(validVal);

			targetsProcessed += static_cast<unsigned long long>(gpuValidTargets);
		}
		else
		{
			targetsProcessed += static_cast<unsigned long long>(T);
		}

		// Periodic progress logging (mirrors CPU path).
		if (logger && (s + 1u) < seqCount)
		{
			const int64_t nowMs = getCurrentTimeMilliseconds();
			const bool dueBySeq = (((s + 1u) % progressEverySeq) == 0u);
			const bool dueByTime = ((nowMs - lastProgressMs) >= kProgressIntervalMs);
			if (dueBySeq || dueByTime)
			{
				lastProgressMs = nowMs;
				const double elapsedMs = static_cast<double>(nowMs - epochStartMs);
				const double tokPerSec = (elapsedMs > 0.0) ? (static_cast<double>(targetsProcessed) / (elapsedMs / 1000.0)) : 0.0;
				const double meanNll = (tokenLmTokenCount > 0ULL) ? (tokenLmNllSum / static_cast<double>(tokenLmTokenCount)) : 0.0;

				std::ostringstream oss;
				oss << "event=nn_epoch_progress";
				append_logfmt_kv(oss, "net_type", netType);
				append_logfmt_kv(oss, "run_type", std::string("train"));
				append_logfmt_kv(oss, "gpu", true);
				append_logfmt_kv(oss, "epoch", epochIdx);
				append_logfmt_kv(oss, "seq_done", s + 1u);
				append_logfmt_kv(oss, "seq_total", seqCount);
				append_logfmt_kv(oss, "tokens_seen", tokensProcessed);
				append_logfmt_kv(oss, "targets_seen", targetsProcessed);
				append_logfmt_kv(oss, "targets_per_sec", tokPerSec);
				if (tokenLM)
				{
					const bool tokenLmFullSoftmax =
					    (cfg.tokenLmLossKind == glades::TransformerRunConfig::TOKEN_LM_FULL_SOFTMAX);
					append_logfmt_kv(oss, "token_lm_loss_kind",
					                 std::string(tokenLmFullSoftmax ? "full_softmax" : "sampled_softmax"));
					append_logfmt_kv(oss, "nll", meanNll);
					if (tokenLmFullSoftmax)
					{
						double ppl = 0.0;
						if (tokenLmTokenCount > 0ULL)
						{
							double arg = meanNll;
							if (arg > 80.0) arg = 80.0;
							if (arg < -80.0) arg = -80.0;
							ppl = exp(arg);
						}
						append_logfmt_kv(oss, "perplexity", ppl);
						append_logfmt_kv(oss, "acc_top1", (clsTotal > 0ULL) ? (100.0 * static_cast<double>(clsCorrect) / static_cast<double>(clsTotal)) : 0.0);
					}
					else
					{
						append_logfmt_kv(oss, "perplexity", std::string("na"));
						append_logfmt_kv(oss, "acc_top1", std::string("na"));
					}
				}
				else
				{
					append_logfmt_kv(oss, "loss_so_far", overallTotalError);
				}
				append_logfmt_kv(oss, "lr_mult", lrScheduleMultiplier);
				if (trainingConfig.globalGradClipNorm > 0.0f)
				{
					append_logfmt_kv(oss, "grad_norm", lastGradNorm);
					append_logfmt_kv(oss, "grad_norm_scale", lastGradNormScale);
				}
				append_logfmt_kv(oss, "optimizer_step", static_cast<unsigned long long>(tensorTransformer.optimizerStep));
				logger->info("NNetwork", shmea::GString(oss.str().c_str()));
			}
		}

		// === GPU Backward pass ===
		if (seqInBatch == 0u)
		{
			gpu::zeroTransformerGradients(*gpuTransformerWeights);
			// Phase-2: bf16 mirrors are the cumulative grad store, so they
			// must start each Adam window at zero (bf16_accum_axpy with
			// beta=1 accumulates into them across micro-batches and layers).
			if (trainingConfig.mixedPrecision.gradStorageBf16Phase2)
				gpu::zeroTransformerGradientsBf16(*gpuTransformerWeights);
			timeStepsInBatch = 0u;
		}
		const bool useBf16GradsPh2_ = trainingConfig.mixedPrecision.gradStorageBf16Phase2;
		glades::gpu::ScopedPerfTimerMs gpuBackwardStage(gpuPerf ? &gpuPerf->msBackward : NULL);

		// Compute dLogits on GPU.  In activation-checkpoint mode the last layer's
		// hAfterFF lives in slot ((nLayers-1) % slotsPerLayer); else modulo
		// collapses to (nLayers-1).
		const unsigned int slotsPerLayerBwdFinal = gpuTransformerScratch->slotsPerLayer
		    ? gpuTransformerScratch->slotsPerLayer : nLayers;
		const float* bwdFinalH = gpuTransformerScratch->hAfterFF.data() +
		    static_cast<size_t>((nLayers - 1) % slotsPerLayerBwdFinal) * T * dModel;
		const float* bwdPostFinalLN = gpuTransformerScratch->hPostFinalLN.data();

#define GLADES_ECHO_GPU_OBSERVE(enabled_, state_, rowObs_, colObs_, samples_, rows_, cols_) do { \
	const bool gladesEchoEnabled_ = trainingConfig.atlas.echoEnabled && (enabled_); \
	if (gpuFuseEchoConfig) { \
		if (gladesEchoEnabled_ && gpuBatchEchoObserve) { \
			if (!gpu::echo_gpu_ensure_state((state_), (rows_), (cols_))) { \
				lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, \
				    "transformerGpuTrainEpoch: GPU ECHO state initialization failed"); \
				storeRunningFlag(false); \
				return; \
			} \
			if (rebuildEchoObserveMeta) { \
				glades::gpu::GpuEchoObserveEntry echoEntry_; \
				echoEntry_.rowObs = (rowObs_); \
				echoEntry_.colObs = (colObs_); \
				echoEntry_.rowSecond = (state_).rowSecond.data(); \
				echoEntry_.colSecond = (state_).colSecond.data(); \
				echoEntry_.samples = static_cast<int>(samples_); \
				echoEntry_.rows = static_cast<int>(rows_); \
				echoEntry_.cols = static_cast<int>(cols_); \
				echoEntry_.featureBase = echoObserveTotalFeatures; \
				echoObserveEntries.push_back(echoEntry_); \
				echoObserveTotalFeatures += echoEntry_.rows + echoEntry_.cols; \
			} \
		} \
	} else if (!observe_echo_gpu((state_), (rowObs_), (colObs_), (samples_), (rows_), (cols_), trainingConfig.atlas, gladesEchoEnabled_)) { \
		lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, \
		    "transformerGpuTrainEpoch: GPU ECHO operand observation failed"); \
		storeRunningFlag(false); \
		return; \
	} \
} while (0)

		if (tokenLM)
		{
			// dLogits = probs - one_hot(targets)
			gpu::softmax_cross_entropy_bwd(
			    gpuTransformerScratch->probs.data(),
			    gpuTransformerScratch->gpuTargetsT.data(),
			    static_cast<int>(T), static_cast<int>(vocabSize),
			    gpuTransformerScratch->dLogits.data());

			// Backprop tied LM head: logits = hPostFinalLN * E^T + lmBias
			// dH (w.r.t. hPostFinalLN) = dLogits * E
			gpu_gemm_mp(bf16Head,
			    static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(vocabSize), 1.0f,
			    gpuTransformerScratch->dLogits.data(),
			    gpuTransformerScratch->activationLowp.data(), static_cast<int>(vocabSize),
			    gpuTransformerWeights->tokE.data(), gpuTransformerWeights->tokELowp.data(),
			    static_cast<int>(dModel),
			    0.0f, gpuTransformerScratch->dH.data(), static_cast<int>(dModel));

			// gTokE += dLogits^T * hPostFinalLN  [vocabSize, dModel]
			// gTokE is a SHARED grad target between the head-tied GEMM here
			// and the embedding_scatter_add below.  Routing it through
			// scratch+commit-bf16 directly accumulates the head-tied
			// contribution into bf16 — but the per-element atomic-bf16-add
			// in embedding_scatter_add_bf16 introduced a 50+ mnat drift due
			// to bf16-precision loss across hundreds of per-token adds.  Until
			// a sparse-FP32-then-cast variant of scatter_add is in place,
			// gTokE stays on the Phase-1 cast path (FP32 grad -> bf16 mirror).
			gpu_gemm_atb_mp(bf16Head,
			    static_cast<int>(vocabSize), static_cast<int>(dModel), static_cast<int>(T), 1.0f,
			    gpuTransformerScratch->dLogits.data(),
			    gpuTransformerScratch->activationLowp.data(), static_cast<int>(vocabSize),
			    bwdPostFinalLN,
			    gpuTransformerScratch->activationLowp2.data(), static_cast<int>(dModel),
			    1.0f, gpuTransformerWeights->gTokE.data(), static_cast<int>(dModel));
			GLADES_ECHO_GPU_OBSERVE(echo_scope_uses_head_matrix(trainingConfig.atlas, vocabSize, dModel),
			                        gpuTransformerWeights->echoTokE,
			                        gpuTransformerScratch->dLogits.data(), bwdPostFinalLN,
			                        T, vocabSize, dModel);

			// gLmBias += sum_rows(dLogits)
			gpu::reduce_rows_sum(
			    gpuTransformerScratch->dLogits.data(),
			    static_cast<int>(T), static_cast<int>(vocabSize),
			    1.0f, gpuTransformerWeights->gLmBias.data());

			timeStepsInBatch += gpuValidTargets;
		}
		else
		{
			// Non-tokenLM: dLogits computed from probs - expected on CPU, upload.
			std::vector<float> probsHost(static_cast<size_t>(T) * outSize);
			gpuTransformerScratch->probs.download(&probsHost[0], probsHost.size());
			std::vector<float> dLogitsHost(static_cast<size_t>(T) * outSize, 0.0f);
			for (unsigned int t = 0; t < T; ++t)
			{
				const float* expRow = NULL;
				unsigned int expSize = 0u;
				di->getTrainSequenceExpectedRowView(s, t, expRow, expSize);
				const size_t off = static_cast<size_t>(t) * outSize;
				for (unsigned int k = 0; k < outSize; ++k)
				{
					const float expv = (expRow && k < expSize) ? expRow[k] : 0.0f;
					dLogitsHost[off + k] = probsHost[off + k] - expv;
				}
			}
			gpuTransformerScratch->dLogits.upload(&dLogitsHost[0], dLogitsHost.size());

			// dH (w.r.t. hPostFinalLN) = dLogits * WOut  [T, dModel]
			gpu_gemm_mp(bf16Head,
			    static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(outSize), 1.0f,
			    gpuTransformerScratch->dLogits.data(),
			    gpuTransformerScratch->activationLowp.data(), static_cast<int>(outSize),
			    gpuTransformerWeights->WOut.data(), gpuTransformerWeights->WOutLowp.data(),
			    static_cast<int>(dModel),
			    0.0f, gpuTransformerScratch->dH.data(), static_cast<int>(dModel));

			// gWOut += dLogits^T * hPostFinalLN  [outSize, dModel]
			{
				float* gradOut = useBf16GradsPh2_
				    ? gpuTransformerScratch->gradScratchFp32.data()
				    : gpuTransformerWeights->gWOut.data();
				const float gemmBeta = useBf16GradsPh2_ ? 0.0f : 1.0f;
				gpu_gemm_atb_mp(bf16Head,
				    static_cast<int>(outSize), static_cast<int>(dModel), static_cast<int>(T), 1.0f,
				    gpuTransformerScratch->dLogits.data(),
				    gpuTransformerScratch->activationLowp.data(), static_cast<int>(outSize),
				    bwdPostFinalLN,
				    gpuTransformerScratch->activationLowp2.data(), static_cast<int>(dModel),
				    gemmBeta, gradOut, static_cast<int>(dModel));
				if (useBf16GradsPh2_)
					gpu::bf16_accum_axpy(gpuTransformerWeights->gWOut_bf16.data(),
					    gpuTransformerScratch->gradScratchFp32.data(),
					    1.0f, 1.0f, gpuTransformerWeights->gWOut_bf16.size());
			}
			GLADES_ECHO_GPU_OBSERVE(echo_scope_uses_head_matrix(trainingConfig.atlas, outSize, dModel),
			                        gpuTransformerWeights->echoWOut,
			                        gpuTransformerScratch->dLogits.data(), bwdPostFinalLN,
			                        T, outSize, dModel);

			// gBOut += sum_rows(dLogits)
			gpu::reduce_rows_sum(
			    gpuTransformerScratch->dLogits.data(),
			    static_cast<int>(T), static_cast<int>(outSize),
			    1.0f, gpuTransformerWeights->gBOut.data());

			timeStepsInBatch += T;
		}

		// Backprop Final LayerNorm: dH (w.r.t. hPostFinalLN) -> dH (w.r.t. hFinal)
		if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
		{
			gpu::rmsnorm_backward(
			    gpuTransformerScratch->dH.data(), bwdFinalH,
			    gpuTransformerWeights->lnFinalGamma.data(),
			    gpuTransformerScratch->lnFinalInvStd.data(),
			    static_cast<int>(T), static_cast<int>(dModel),
			    gpuTransformerScratch->dH2.data(),
			    gpuTransformerWeights->gLnFinalGamma.data());
		}
		else
		{
			gpu::layernorm_backward(
			    gpuTransformerScratch->dH.data(), bwdFinalH,
			    gpuTransformerWeights->lnFinalGamma.data(),
			    gpuTransformerScratch->lnFinalMean.data(),
			    gpuTransformerScratch->lnFinalInvStd.data(),
			    static_cast<int>(T), static_cast<int>(dModel),
			    gpuTransformerScratch->dH2.data(),
			    gpuTransformerWeights->gLnFinalGamma.data(),
			    gpuTransformerWeights->gLnFinalBeta.data());
		}
		// dH2 now has gradient w.r.t. hFinal; swap into dH for block backprop.
		gpu::device_memcpy_d2d(gpuTransformerScratch->dH.data(),
		                       gpuTransformerScratch->dH2.data(),
		                       static_cast<size_t>(T) * dModel * sizeof(float));

		// RoPE invFreq already uploaded to scratch before forward layer loop.
		const unsigned int ropeHalfDim = fwdRopeHalfDim;

		// Backprop through blocks (reverse order).
		// Activation gradient checkpointing (sqrt-L scheme): walk segments
		// from the highest down to 0.  After the initial forward pass the
		// cyclic activation slots contain only the LAST segment's layers
		// (the rest were overwritten as forward marched through the layers).
		// For each non-last segment we re-run the per-layer forward body
		// starting from the saved checkpoint (or the embedding output `h`
		// for segment 0), repopulating the slots before the inner reverse
		// backward sweep reads them.  When activationCheckpoint is off,
		// slotsPerLayer == nLayers, nSegments == 1, and the recompute path
		// is skipped — bit-for-bit identical to the prior single-loop form.
		const unsigned int slotsPerLayerBwd = gpuTransformerScratch->slotsPerLayer
		    ? gpuTransformerScratch->slotsPerLayer : nLayers;
		const unsigned int nSegmentsBwd = (slotsPerLayerBwd >= nLayers)
		    ? 1u
		    : ((nLayers + slotsPerLayerBwd - 1u) / slotsPerLayerBwd);
		for (int seg = static_cast<int>(nSegmentsBwd) - 1; seg >= 0; --seg)
		{
			const unsigned int segStart = static_cast<unsigned int>(seg) * slotsPerLayerBwd;
			unsigned int segEndU = static_cast<unsigned int>(seg + 1) * slotsPerLayerBwd;
			if (segEndU > nLayers) segEndU = nLayers;
			const unsigned int segEnd = segEndU;

			// Recompute this segment's forward unless its slots are already
			// populated by the most recent forward pass (the LAST segment).
			if (seg < static_cast<int>(nSegmentsBwd) - 1)
			{
				const float* segIn;
				if (seg == 0)
				{
					segIn = gpuTransformerScratch->h.data();
				}
				else
				{
					segIn = gpuTransformerScratch->checkpoints.data()
					    + static_cast<size_t>(seg - 1) * T * dModel;
				}
				if (!transformerGpuLayerRangeForward(cfg, T, segStart, segEnd, segIn,
				                                     useBf16, useRope,
				                                     bf16Wq, bf16Wk, bf16Wv,
				                                     bf16Wo, bf16W1, bf16W2,
				                                     ropeDimOverride))
				{
					lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
					    "transformerGpuTrainEpoch: activation-checkpoint segment recompute failed");
					storeRunningFlag(false);
					return;
				}
			}

		for (int li = static_cast<int>(segEnd) - 1; li >= static_cast<int>(segStart); --li)
		{
			gpu::GpuTransformerWeights::Block& gb = gpuTransformerWeights->blocks[li];
			// Activation-checkpoint: per-layer activations live in cyclic slots.
			// When checkpointing is off, slotsPerLayer == nLayers and modulo
			// collapses to identity.  Otherwise the slots were just freshly
			// repopulated by the segment-recompute call above (or are the
			// initial forward's last-segment leftovers when seg == nSegments-1).
			const size_t slot = static_cast<size_t>(static_cast<unsigned int>(li) % slotsPerLayerBwd);
			const size_t prevSlot = (li > 0)
			    ? static_cast<size_t>(static_cast<unsigned int>(li - 1) % slotsPerLayerBwd)
			    : 0u;
			const size_t layerOff = slot * static_cast<size_t>(T);

			// layerIn = input to forward layer li.  Three cases:
			//   (1) li == 0: embedding output `h`.
			//   (2) li == segStart and seg > 0 (activation-checkpoint mode):
			//       the slot at prevSlot DOES NOT hold hAfterFF[li-1] — it
			//       holds either the recomputed segment's last layer (if
			//       this segment overlapped K-1 slot) or stale data.  Use
			//       the saved checkpoint[seg-1] instead.
			//   (3) Otherwise: cyclic-slot read.
			const float* layerIn;
			if (li == 0)
				layerIn = gpuTransformerScratch->h.data();
			else if (li == static_cast<int>(segStart) && seg > 0)
				layerIn = gpuTransformerScratch->checkpoints.data()
				    + static_cast<size_t>(seg - 1) * T * dModel;
			else
				layerIn = gpuTransformerScratch->hAfterFF.data() + prevSlot * T * dModel;
			const float* x1_l = gpuTransformerScratch->x1.data() + slot * T * dModel;
			const float* x2_l = gpuTransformerScratch->x2.data() + slot * T * dModel;
			const float* ff1_l = gpuTransformerScratch->ff1.data() + slot * T * ff1Width;
			const float* hAfterAttn_l = gpuTransformerScratch->hAfterAttn.data() + slot * T * dModel;
			const float* attnConcat_l = gpuTransformerScratch->attnConcat.data() + slot * T * dModel;
			const float* ff1Act_l = gpuTransformerScratch->ff1Act.data() + slot * T * dFF;

			// dH is gradient w.r.t. hAfterFF[li].
			// Residual: hAfterFF = hAfterAttn + ffOut => dFFOut = dH, dHAfterAttn (residual) = dH.
			// --- FFN backward ---
			// ffOut = W2 * ff1Act + b2  =>  dFF1Act = dH * W2^T, gW2 += dH^T * ff1Act
			gpu_gemm_mp(bf16W2,
			    static_cast<int>(T), static_cast<int>(dFF), static_cast<int>(dModel), 1.0f,
			    gpuTransformerScratch->dH.data(),
			    gpuTransformerScratch->activationLowp.data(), static_cast<int>(dModel),
			    gb.W2.data(), gb.W2Lowp.data(), static_cast<int>(dFF),
			    0.0f, gpuTransformerScratch->dFF1Act.data(), static_cast<int>(dFF));
			{
				float* gradOut = useBf16GradsPh2_
				    ? gpuTransformerScratch->gradScratchFp32.data()
				    : gb.gW2.data();
				const float gemmBeta = useBf16GradsPh2_ ? 0.0f : 1.0f;
				gpu_gemm_atb_mp(bf16W2,
				    static_cast<int>(dModel), static_cast<int>(dFF), static_cast<int>(T), 1.0f,
				    gpuTransformerScratch->dH.data(),
				    gpuTransformerScratch->activationLowp.data(), static_cast<int>(dModel),
				    ff1Act_l,
				    gpuTransformerScratch->activationLowp2.data(), static_cast<int>(dFF),
				    gemmBeta, gradOut, static_cast<int>(dFF));
				if (useBf16GradsPh2_)
					gpu::bf16_accum_axpy(gb.gW2_bf16.data(),
					    gpuTransformerScratch->gradScratchFp32.data(),
					    1.0f, 1.0f, gb.gW2_bf16.size());
			}
			GLADES_ECHO_GPU_OBSERVE(echo_scope_uses_decoder_matrix(trainingConfig.atlas, static_cast<unsigned int>(li), nLayers, dModel, dFF),
			                        gb.echoW2,
			                        gpuTransformerScratch->dH.data(), ff1Act_l,
			                        T, dModel, dFF);
			// gB2 += sum_rows(dH)
			gpu::reduce_rows_sum(
			    gpuTransformerScratch->dH.data(),
			    static_cast<int>(T), static_cast<int>(dModel),
			    1.0f, gb.gB2.data());

			// Activation backward.
			if (ffnKind == 1) // SwiGLU
			{
				gpu::swiglu_backward(
				    gpuTransformerScratch->dFF1Act.data(), ff1_l,
				    static_cast<int>(T), static_cast<int>(dFF),
				    gpuTransformerScratch->dFF1Cat.data());
				// ff1 = W1 * x2 + b1 => dX2 = dFF1Cat * W1^T, gW1 += dFF1Cat^T * x2
				gpu_gemm_mp(bf16W1,
				    static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(ff1Width), 1.0f,
				    gpuTransformerScratch->dFF1Cat.data(),
				    gpuTransformerScratch->activationLowp.data(), static_cast<int>(ff1Width),
				    gb.W1.data(), gb.W1Lowp.data(), static_cast<int>(dModel),
				    0.0f, gpuTransformerScratch->dX2.data(), static_cast<int>(dModel));
				{
					float* gradOut = useBf16GradsPh2_
					    ? gpuTransformerScratch->gradScratchFp32.data()
					    : gb.gW1.data();
					const float gemmBeta = useBf16GradsPh2_ ? 0.0f : 1.0f;
					gpu_gemm_atb_mp(bf16W1,
					    static_cast<int>(ff1Width), static_cast<int>(dModel), static_cast<int>(T), 1.0f,
					    gpuTransformerScratch->dFF1Cat.data(),
					    gpuTransformerScratch->activationLowp.data(), static_cast<int>(ff1Width),
					    x2_l,
					    gpuTransformerScratch->activationLowp2.data(), static_cast<int>(dModel),
					    gemmBeta, gradOut, static_cast<int>(dModel));
					if (useBf16GradsPh2_)
						gpu::bf16_accum_axpy(gb.gW1_bf16.data(),
						    gpuTransformerScratch->gradScratchFp32.data(),
						    1.0f, 1.0f, gb.gW1_bf16.size());
				}
				GLADES_ECHO_GPU_OBSERVE(echo_scope_uses_decoder_matrix(trainingConfig.atlas, static_cast<unsigned int>(li), nLayers, ff1Width, dModel),
				                        gb.echoW1,
				                        gpuTransformerScratch->dFF1Cat.data(), x2_l,
				                        T, ff1Width, dModel);
				gpu::reduce_rows_sum(
				    gpuTransformerScratch->dFF1Cat.data(),
				    static_cast<int>(T), static_cast<int>(ff1Width),
				    1.0f, gb.gB1.data());
			}
			else
			{
				// GELU/ReLU backward
				if (ffnAct == static_cast<int>(glades::TransformerRunConfig::FFN_GELU))
				{
					gpu::gelu_backward(
					    gpuTransformerScratch->dFF1Act.data(), ff1_l,
					    static_cast<int>(T * dFF),
					    gpuTransformerScratch->dFF1Act.data());
				}
				else
				{
					gpu::relu_backward(
					    gpuTransformerScratch->dFF1Act.data(), ff1_l,
					    static_cast<int>(T * dFF),
					    gpuTransformerScratch->dFF1Act.data());
				}
				// ff1 = W1 * x2 + b1 => dX2, gW1
				gpu_gemm_mp(bf16W1,
				    static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(dFF), 1.0f,
				    gpuTransformerScratch->dFF1Act.data(),
				    gpuTransformerScratch->activationLowp.data(), static_cast<int>(dFF),
				    gb.W1.data(), gb.W1Lowp.data(), static_cast<int>(dModel),
				    0.0f, gpuTransformerScratch->dX2.data(), static_cast<int>(dModel));
				{
					float* gradOut = useBf16GradsPh2_
					    ? gpuTransformerScratch->gradScratchFp32.data()
					    : gb.gW1.data();
					const float gemmBeta = useBf16GradsPh2_ ? 0.0f : 1.0f;
					gpu_gemm_atb_mp(bf16W1,
					    static_cast<int>(dFF), static_cast<int>(dModel), static_cast<int>(T), 1.0f,
					    gpuTransformerScratch->dFF1Act.data(),
					    gpuTransformerScratch->activationLowp.data(), static_cast<int>(dFF),
					    x2_l,
					    gpuTransformerScratch->activationLowp2.data(), static_cast<int>(dModel),
					    gemmBeta, gradOut, static_cast<int>(dModel));
					if (useBf16GradsPh2_)
						gpu::bf16_accum_axpy(gb.gW1_bf16.data(),
						    gpuTransformerScratch->gradScratchFp32.data(),
						    1.0f, 1.0f, gb.gW1_bf16.size());
				}
				GLADES_ECHO_GPU_OBSERVE(echo_scope_uses_decoder_matrix(trainingConfig.atlas, static_cast<unsigned int>(li), nLayers, dFF, dModel),
				                        gb.echoW1,
				                        gpuTransformerScratch->dFF1Act.data(), x2_l,
				                        T, dFF, dModel);
				gpu::reduce_rows_sum(
				    gpuTransformerScratch->dFF1Act.data(),
				    static_cast<int>(T), static_cast<int>(dFF),
				    1.0f, gb.gB1.data());
			}

			// --- LN2 backward ---
			const float* ln2Mean_l = gpuTransformerScratch->ln2Mean.data() + layerOff;
			const float* ln2InvStd_l = gpuTransformerScratch->ln2InvStd.data() + layerOff;
			if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
			{
				gpu::rmsnorm_backward(
				    gpuTransformerScratch->dX2.data(), hAfterAttn_l,
				    gb.ln2Gamma.data(), ln2InvStd_l,
				    static_cast<int>(T), static_cast<int>(dModel),
				    gpuTransformerScratch->dHAfterAttnFromLN.data(),
				    gb.gLn2Gamma.data());
			}
			else
			{
				gpu::layernorm_backward(
				    gpuTransformerScratch->dX2.data(), hAfterAttn_l,
				    gb.ln2Gamma.data(), ln2Mean_l, ln2InvStd_l,
				    static_cast<int>(T), static_cast<int>(dModel),
				    gpuTransformerScratch->dHAfterAttnFromLN.data(),
				    gb.gLn2Gamma.data(), gb.gLn2Beta.data());
			}

			// Combine: dHAfterAttn = dH (residual) + dHAfterAttnFromLN
			gpu::add_two(gpuTransformerScratch->dH2.data(),
			    gpuTransformerScratch->dH.data(),
			    gpuTransformerScratch->dHAfterAttnFromLN.data(),
			    static_cast<int>(T * dModel));

			// --- Wo backward ---
			// attnOut = Wo * attnConcat + bo  =>  dAttnConcat, gWo
			gpu_gemm_mp(bf16Wo,
			    static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(dModel), 1.0f,
			    gpuTransformerScratch->dH2.data(),
			    gpuTransformerScratch->activationLowp.data(), static_cast<int>(dModel),
			    gb.Wo.data(), gb.WoLowp.data(), static_cast<int>(dModel),
			    0.0f, gpuTransformerScratch->dAttnConcat.data(), static_cast<int>(dModel));
			{
				float* gradOut = useBf16GradsPh2_
				    ? gpuTransformerScratch->gradScratchFp32.data()
				    : gb.gWo.data();
				const float gemmBeta = useBf16GradsPh2_ ? 0.0f : 1.0f;
				gpu_gemm_atb_mp(bf16Wo,
				    static_cast<int>(dModel), static_cast<int>(dModel), static_cast<int>(T), 1.0f,
				    gpuTransformerScratch->dH2.data(),
				    gpuTransformerScratch->activationLowp.data(), static_cast<int>(dModel),
				    attnConcat_l,
				    gpuTransformerScratch->activationLowp2.data(), static_cast<int>(dModel),
				    gemmBeta, gradOut, static_cast<int>(dModel));
				if (useBf16GradsPh2_)
					gpu::bf16_accum_axpy(gb.gWo_bf16.data(),
					    gpuTransformerScratch->gradScratchFp32.data(),
					    1.0f, 1.0f, gb.gWo_bf16.size());
			}
			GLADES_ECHO_GPU_OBSERVE(echo_scope_uses_decoder_matrix(trainingConfig.atlas, static_cast<unsigned int>(li), nLayers, dModel, dModel),
			                        gb.echoWo,
			                        gpuTransformerScratch->dH2.data(), attnConcat_l,
			                        T, dModel, dModel);
			gpu::reduce_rows_sum(
			    gpuTransformerScratch->dH2.data(),
			    static_cast<int>(T), static_cast<int>(dModel),
			    1.0f, gb.gBo.data());

			// --- Attention backward (flash-style recompute) ---
			float* Q_l = gpuTransformerScratch->Q.data() + slot * T * dModel;
			float* K_l = gpuTransformerScratch->K.data() + slot * T * dModelKV;
			float* V_l = gpuTransformerScratch->V.data() + slot * T * dModelKV;

			// Zero dK/dV (dQ is overwritten per-head, but dK/dV accumulate for GQA).
			gpu::zero_buffers_batch(gpuTransformerScratch->d_dKdVZeroPtrs,
			    gpuTransformerScratch->d_dKdVZeroSizes, 2);
			if (gpuPerf)
				gpu::perfRecordKernel(&gpuPerf->counters, 2u);

			bool attnBwdDone = false;
			if (useBf16)
			{
				// The q/k/vLowp scratches were already populated by the
				// forward attention call, and Q/K/V haven't been modified
				// between forward and backward — so we can feed the same
				// BF16 views without re-casting. Fall back to FP32 on any
				// kernel launch failure (e.g., shape exceeds the
				// multi-query shmem budget on this device).
				// Local-window attention backward (paradigm shift #6 port).
				// Paradigm #78: also dispatched when attnSinkCount > 0.
				const int localW_bw = trainingConfig.transformer.localAttnWindow;
				const int sinkS_bw = trainingConfig.transformer.attnSinkCount > 0
				    ? trainingConfig.transformer.attnSinkCount : 0;
				if ((localW_bw > 0 && localW_bw < static_cast<int>(T)) || sinkS_bw > 0) {
				attnBwdDone = gpu::flash_attention_multihead_backward_bf16_local(
				    gpuTransformerScratch->qLowp.data(),
				    gpuTransformerScratch->kLowp.data(),
				    gpuTransformerScratch->vLowp.data(),
				    attnConcat_l,
				    gpuTransformerScratch->dAttnConcat.data(),
				    static_cast<int>(T),
				    static_cast<int>(nHeads),
				    static_cast<int>(nKVHeads),
				    static_cast<int>(dHead),
				    static_cast<int>(dModel),
				    static_cast<int>(dModelKV),
				    causal,
				    localW_bw,
				    gpuTransformerScratch->dQfull.data(),
				    gpuTransformerScratch->dKfull.data(),
				    gpuTransformerScratch->dVfull.data(),
				    sinkS_bw);
				} else {
				attnBwdDone = gpu::flash_attention_multihead_backward_bf16(
				    gpuTransformerScratch->qLowp.data(),
				    gpuTransformerScratch->kLowp.data(),
				    gpuTransformerScratch->vLowp.data(),
				    attnConcat_l,
				    gpuTransformerScratch->dAttnConcat.data(),
				    static_cast<int>(T),
				    static_cast<int>(nHeads),
				    static_cast<int>(nKVHeads),
				    static_cast<int>(dHead),
				    static_cast<int>(dModel),
				    static_cast<int>(dModelKV),
				    causal,
				    gpuTransformerScratch->dQfull.data(),
				    gpuTransformerScratch->dKfull.data(),
				    gpuTransformerScratch->dVfull.data());
				}
			}
			if (!attnBwdDone)
			{
				// cuBLAS-tiled tensor-core attention BACKWARD (companion to
				// the forward fast-attn path — research/WMMA_ATTENTION_PLAN.md).
				// Eligible only when no GQA.  Uses the same attnScoresScratch
				// buffer for P (recomputed) and allocates a second T*T*nH
				// buffer for dP.  Parity-verified against flash_attention_
				// multihead_backward (dQ/dK/dV max_err < 5e-3 in unit test).
				const bool chiron_fast_bwd = (nHeads == nKVHeads);
				if (chiron_fast_bwd)
				{
					const size_t scoresNeeded = static_cast<size_t>(nHeads) * T * T;
					if (gpuTransformerScratch->attnScoresScratch.size() < scoresNeeded)
						gpuTransformerScratch->attnScoresScratch.allocate(scoresNeeded);
					if (gpuTransformerScratch->attnDPScratch.size() < scoresNeeded)
						gpuTransformerScratch->attnDPScratch.allocate(scoresNeeded);
					if (gpuTransformerScratch->attnScoresScratch.size() >= scoresNeeded
					    && gpuTransformerScratch->attnDPScratch.size() >= scoresNeeded)
					{
						attnBwdDone = gpu::flash_attention_backward_cublas_tiled(
						    Q_l, K_l, V_l,
						    attnConcat_l,
						    gpuTransformerScratch->dAttnConcat.data(),
						    static_cast<int>(T), static_cast<int>(nHeads),
						    static_cast<int>(dHead), static_cast<int>(dModel),
						    causal,
						    gpuTransformerScratch->dQfull.data(),
						    gpuTransformerScratch->dKfull.data(),
						    gpuTransformerScratch->dVfull.data(),
						    gpuTransformerScratch->attnScoresScratch.data(),
						    gpuTransformerScratch->attnDPScratch.data());
					}
				}
				if (!attnBwdDone)
				{
				gpu::flash_attention_multihead_backward(
				    Q_l, K_l, V_l,
				    attnConcat_l,
				    gpuTransformerScratch->dAttnConcat.data(),
				    static_cast<int>(T),
				    static_cast<int>(nHeads),
				    static_cast<int>(nKVHeads),
				    static_cast<int>(dHead),
				    static_cast<int>(dModel),
				    static_cast<int>(dModelKV),
				    causal,
				    gpuTransformerScratch->dQfull.data(),
				    gpuTransformerScratch->dKfull.data(),
				    gpuTransformerScratch->dVfull.data());
				}
			}

			// --- RoPE backward (inverse rotation) — fused Q+K ---
			if (useRope && gpuTransformerScratch->gpuInvFreq.allocated())
			{
				gpu::rope_apply_qk(gpuTransformerScratch->dQfull.data(),
				    gpuTransformerScratch->dKfull.data(),
				    gpuTransformerScratch->gpuInvFreq.data(),
				    static_cast<int>(T), static_cast<int>(nHeads),
				    static_cast<int>(nKVHeads), static_cast<int>(dHead),
				    static_cast<int>(ropeHalfDim), true);
			}

			// --- Q/K/V projection backward ---
			// dX1 = dQ*Wq^T + dK*Wk^T + dV*Wv^T
			// Also accumulate gWq, gBq, gWk, gBk, gWv, gBv.

			// Q: dXtmp = dQ * Wq^T
			gpu_gemm_mp(bf16Wq,
			    static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(dModel), 1.0f,
			    gpuTransformerScratch->dQfull.data(),
			    gpuTransformerScratch->activationLowp.data(), static_cast<int>(dModel),
			    gb.Wq.data(), gb.WqLowp.data(), static_cast<int>(dModel),
			    0.0f, gpuTransformerScratch->dX1.data(), static_cast<int>(dModel));
			{
				float* gradOut = useBf16GradsPh2_
				    ? gpuTransformerScratch->gradScratchFp32.data()
				    : gb.gWq.data();
				const float gemmBeta = useBf16GradsPh2_ ? 0.0f : 1.0f;
				gpu_gemm_atb_mp(bf16Wq,
				    static_cast<int>(dModel), static_cast<int>(dModel), static_cast<int>(T), 1.0f,
				    gpuTransformerScratch->dQfull.data(),
				    gpuTransformerScratch->activationLowp.data(), static_cast<int>(dModel),
				    x1_l,
				    gpuTransformerScratch->activationLowp2.data(), static_cast<int>(dModel),
				    gemmBeta, gradOut, static_cast<int>(dModel));
				if (useBf16GradsPh2_)
					gpu::bf16_accum_axpy(gb.gWq_bf16.data(),
					    gpuTransformerScratch->gradScratchFp32.data(),
					    1.0f, 1.0f, gb.gWq_bf16.size());
			}
			GLADES_ECHO_GPU_OBSERVE(echo_scope_uses_decoder_matrix(trainingConfig.atlas, static_cast<unsigned int>(li), nLayers, dModel, dModel),
			                        gb.echoWq,
			                        gpuTransformerScratch->dQfull.data(), x1_l,
			                        T, dModel, dModel);
			gpu::reduce_rows_sum(
			    gpuTransformerScratch->dQfull.data(),
			    static_cast<int>(T), static_cast<int>(dModel),
			    1.0f, gb.gBq.data());

			// Paradigm shift #76 MLA backward: when active, replace W_K and
			// W_V backward gemms with the latent-path chain rule.
			const int mlaDcBwd = trainingConfig.transformer.mlaLatentDim;
			if (mlaDcBwd > 0 && gb.Wdkv.allocated())
			{
				// dX1 + dWdkv + dWuk + dWuv via single chain-rule call.
				// c was computed during forward and cached in gb.mlaC.
				// Allocate dc scratch + dedicated BF16 staging buffers.
				const size_t dcBytes = static_cast<size_t>(T) * static_cast<size_t>(mlaDcBwd);
				if (gb.mlaDc.size() < dcBytes) gb.mlaDc.allocate(dcBytes);
				// Sizing: max of all operand shapes used in the bf16 path —
				// T*dHidden, T*dKVtotal, T*dC, dHidden*dC, dC*dKVtotal.
				const size_t bf16Need =
				    std::max((size_t)T * dModel,
				    std::max((size_t)T * dModelKV,
				    std::max((size_t)T * mlaDcBwd,
				    std::max((size_t)dModel * mlaDcBwd,
				             (size_t)mlaDcBwd * dModelKV))));
				if (gb.mlaBf16ScratchA.size() < bf16Need) gb.mlaBf16ScratchA.allocate(bf16Need);
				if (gb.mlaBf16ScratchB.size() < bf16Need) gb.mlaBf16ScratchB.allocate(bf16Need);
				gpu::mla_attention_backward_gpu(
				    x1_l, gb.mlaC.data(),
				    gpuTransformerScratch->dKfull.data(),
				    gpuTransformerScratch->dVfull.data(),
				    gb.Wdkv.data(), gb.Wuk.data(), gb.Wuv.data(),
				    static_cast<int>(T), static_cast<int>(dModel), mlaDcBwd, static_cast<int>(dModelKV),
				    gpuTransformerScratch->dX1.data(),
				    gb.gWdkv.data(), gb.gWuk.data(), gb.gWuv.data(),
				    gb.mlaDc.data(),
				    gb.mlaBf16ScratchA.data(),
				    gb.mlaBf16ScratchB.data());
				// Bias gradients still applicable
				gpu::reduce_rows_sum(
				    gpuTransformerScratch->dKfull.data(),
				    static_cast<int>(T), static_cast<int>(dModelKV),
				    1.0f, gb.gBk.data());
				gpu::reduce_rows_sum(
				    gpuTransformerScratch->dVfull.data(),
				    static_cast<int>(T), static_cast<int>(dModelKV),
				    1.0f, gb.gBv.data());
			}
			else
			{
			// K: accumulate dK * Wk^T directly into dX1 (beta=1.0)
			gpu_gemm_mp(bf16Wk,
			    static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(dModelKV), 1.0f,
			    gpuTransformerScratch->dKfull.data(),
			    gpuTransformerScratch->activationLowp.data(), static_cast<int>(dModelKV),
			    gb.Wk.data(), gb.WkLowp.data(), static_cast<int>(dModel),
			    1.0f, gpuTransformerScratch->dX1.data(), static_cast<int>(dModel));
			{
				float* gradOut = useBf16GradsPh2_
				    ? gpuTransformerScratch->gradScratchFp32.data()
				    : gb.gWk.data();
				const float gemmBeta = useBf16GradsPh2_ ? 0.0f : 1.0f;
				gpu_gemm_atb_mp(bf16Wk,
				    static_cast<int>(dModelKV), static_cast<int>(dModel), static_cast<int>(T), 1.0f,
				    gpuTransformerScratch->dKfull.data(),
				    gpuTransformerScratch->activationLowp.data(), static_cast<int>(dModelKV),
				    x1_l,
				    gpuTransformerScratch->activationLowp2.data(), static_cast<int>(dModel),
				    gemmBeta, gradOut, static_cast<int>(dModel));
				if (useBf16GradsPh2_)
					gpu::bf16_accum_axpy(gb.gWk_bf16.data(),
					    gpuTransformerScratch->gradScratchFp32.data(),
					    1.0f, 1.0f, gb.gWk_bf16.size());
			}
			GLADES_ECHO_GPU_OBSERVE(echo_scope_uses_decoder_matrix(trainingConfig.atlas, static_cast<unsigned int>(li), nLayers, dModelKV, dModel),
			                        gb.echoWk,
			                        gpuTransformerScratch->dKfull.data(), x1_l,
			                        T, dModelKV, dModel);
			gpu::reduce_rows_sum(
			    gpuTransformerScratch->dKfull.data(),
			    static_cast<int>(T), static_cast<int>(dModelKV),
			    1.0f, gb.gBk.data());

			// V: accumulate dV * Wv^T directly into dX1 (beta=1.0)
			gpu_gemm_mp(bf16Wv,
			    static_cast<int>(T), static_cast<int>(dModel), static_cast<int>(dModelKV), 1.0f,
			    gpuTransformerScratch->dVfull.data(),
			    gpuTransformerScratch->activationLowp.data(), static_cast<int>(dModelKV),
			    gb.Wv.data(), gb.WvLowp.data(), static_cast<int>(dModel),
			    1.0f, gpuTransformerScratch->dX1.data(), static_cast<int>(dModel));
			{
				float* gradOut = useBf16GradsPh2_
				    ? gpuTransformerScratch->gradScratchFp32.data()
				    : gb.gWv.data();
				const float gemmBeta = useBf16GradsPh2_ ? 0.0f : 1.0f;
				gpu_gemm_atb_mp(bf16Wv,
				    static_cast<int>(dModelKV), static_cast<int>(dModel), static_cast<int>(T), 1.0f,
				    gpuTransformerScratch->dVfull.data(),
				    gpuTransformerScratch->activationLowp.data(), static_cast<int>(dModelKV),
				    x1_l,
				    gpuTransformerScratch->activationLowp2.data(), static_cast<int>(dModel),
				    gemmBeta, gradOut, static_cast<int>(dModel));
				if (useBf16GradsPh2_)
					gpu::bf16_accum_axpy(gb.gWv_bf16.data(),
					    gpuTransformerScratch->gradScratchFp32.data(),
					    1.0f, 1.0f, gb.gWv_bf16.size());
			}
			GLADES_ECHO_GPU_OBSERVE(echo_scope_uses_decoder_matrix(trainingConfig.atlas, static_cast<unsigned int>(li), nLayers, dModelKV, dModel),
			                        gb.echoWv,
			                        gpuTransformerScratch->dVfull.data(), x1_l,
			                        T, dModelKV, dModel);
			gpu::reduce_rows_sum(
			    gpuTransformerScratch->dVfull.data(),
			    static_cast<int>(T), static_cast<int>(dModelKV),
			    1.0f, gb.gBv.data());
			}

			// --- LN1 backward ---
			const float* ln1Mean_l = gpuTransformerScratch->ln1Mean.data() + layerOff;
			const float* ln1InvStd_l = gpuTransformerScratch->ln1InvStd.data() + layerOff;
			if (normType == static_cast<int>(glades::TransformerRunConfig::NORM_RMSNORM))
			{
				gpu::rmsnorm_backward(
				    gpuTransformerScratch->dX1.data(), layerIn,
				    gb.ln1Gamma.data(), ln1InvStd_l,
				    static_cast<int>(T), static_cast<int>(dModel),
				    gpuTransformerScratch->dHInFromLN.data(),
				    gb.gLn1Gamma.data());
			}
			else
			{
				gpu::layernorm_backward(
				    gpuTransformerScratch->dX1.data(), layerIn,
				    gb.ln1Gamma.data(), ln1Mean_l, ln1InvStd_l,
				    static_cast<int>(T), static_cast<int>(dModel),
				    gpuTransformerScratch->dHInFromLN.data(),
				    gb.gLn1Gamma.data(), gb.gLn1Beta.data());
			}

			// Combine into dH: dH = dH2 + dHInFromLN
			gpu::add_two(gpuTransformerScratch->dH.data(),
			    gpuTransformerScratch->dH2.data(),
			    gpuTransformerScratch->dHInFromLN.data(),
			    static_cast<int>(T * dModel));
		} // layers backward (within segment)
		} // activation-checkpoint segments backward

		// Backprop through input embedding/projection.
		if (tokenLM)
		{
			// gTokE[tokenId[t]] += dH[t] for each timestep.
			// gTokE stays on the FP32 grad path even under Phase-2 — the
			// embedding_scatter_add_bf16 kernel exists but per-element bf16
			// atomic-add accumulates round-off across hundreds of token
			// updates, producing 50+ mnat NLL drift.  See the head-tied GEMM
			// site (~line 10509) for the matching rationale.
			gpu::embedding_scatter_add(
			    gpuTransformerWeights->gTokE.data(),
			    gpuTransformerScratch->tokenIds.data(),
			    gpuTransformerScratch->dH.data(),
			    static_cast<int>(T), static_cast<int>(vocabSize), static_cast<int>(dModel));
		}
		else
		{
			// gWIn += dH^T * x, gBIn += sum_rows(dH)
			{
				float* gradOut = useBf16GradsPh2_
				    ? gpuTransformerScratch->gradScratchFp32.data()
				    : gpuTransformerWeights->gWIn.data();
				const float gemmBeta = useBf16GradsPh2_ ? 0.0f : 1.0f;
				gpu_gemm_atb_mp(bf16WIn,
				    static_cast<int>(dModel), static_cast<int>(inputSize), static_cast<int>(T), 1.0f,
				    gpuTransformerScratch->dH.data(),
				    gpuTransformerScratch->activationLowp.data(), static_cast<int>(dModel),
				    gpuTransformerScratch->x.data(),
				    gpuTransformerScratch->activationLowp2.data(), static_cast<int>(inputSize),
				    gemmBeta, gradOut, static_cast<int>(inputSize));
				if (useBf16GradsPh2_)
					gpu::bf16_accum_axpy(gpuTransformerWeights->gWIn_bf16.data(),
					    gpuTransformerScratch->gradScratchFp32.data(),
					    1.0f, 1.0f, gpuTransformerWeights->gWIn_bf16.size());
			}
			GLADES_ECHO_GPU_OBSERVE(echo_scope_uses_input_matrix(trainingConfig.atlas, dModel, inputSize),
			                        gpuTransformerWeights->echoWIn,
			                        gpuTransformerScratch->dH.data(), gpuTransformerScratch->x.data(),
			                        T, dModel, inputSize);
			gpu::reduce_rows_sum(
			    gpuTransformerScratch->dH.data(),
			    static_cast<int>(T), static_cast<int>(dModel),
			    1.0f, gpuTransformerWeights->gBIn.data());
		}

#undef GLADES_ECHO_GPU_OBSERVE

		if (gpuFuseEchoConfig && gpuBatchEchoObserve)
		{
			if (!gpuTransformerWeights->ensureEchoBuffers())
			{
				lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
				    "transformerGpuTrainEpoch: GPU ECHO observe buffer allocation failed");
				storeRunningFlag(false);
				return;
			}
			bool observeOk = true;
			if (rebuildEchoObserveMeta)
			{
				observeOk = gpu::echo_gpu_observe_batch(
				    gpuTransformerWeights->d_echoObserveEntries,
				    gpuTransformerWeights->echoObserveCapacity,
				    echoObserveEntries.empty() ? static_cast<const glades::gpu::GpuEchoObserveEntry*>(NULL)
				                              : echoObserveEntries.data(),
				    static_cast<int>(echoObserveEntries.size()),
				    echoObserveTotalFeatures,
				    tensorTransformer.optimizerStep + 1ULL,
				    trainingConfig.atlas);
				if (observeOk)
				{
					gpuTransformerWeights->echoObserveEntryCount =
					    static_cast<int>(echoObserveEntries.size());
					gpuTransformerWeights->echoObserveTotalFeatures = echoObserveTotalFeatures;
					gpuTransformerWeights->echoObserveSeqLen = T;
					gpuTransformerWeights->echoObserveScope = trainingConfig.atlas.echoScope;
					gpuTransformerWeights->echoObserveTokenModel = tokenLM;
					gpuTransformerWeights->echoObserveMetaUploaded = true;
				}
			}
			else if (gpuTransformerWeights->echoObserveMetaUploaded
			         && gpuTransformerWeights->echoObserveEntryCount > 0
			         && gpuTransformerWeights->echoObserveTotalFeatures > 0)
			{
				observeOk = gpu::echo_gpu_launch_observe_batch(
				    gpuTransformerWeights->d_echoObserveEntries,
				    gpuTransformerWeights->echoObserveEntryCount,
				    gpuTransformerWeights->echoObserveTotalFeatures,
				    tensorTransformer.optimizerStep + 1ULL,
				    trainingConfig.atlas);
			}
			if (!observeOk)
			{
				lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
				    "transformerGpuTrainEpoch: batched GPU ECHO operand observation failed");
				storeRunningFlag(false);
				return;
			}
		}

		++seqInBatch;

		// === Optimizer step (when batch is complete) ===
		if (seqInBatch >= seqBatchMax && timeStepsInBatch > 0u)
		{
			glades::gpu::ScopedPerfTimerMs gpuOptStage(gpuPerf ? &gpuPerf->msOptimizer : NULL);
			const float invBatch = 1.0f / static_cast<float>(timeStepsInBatch);
			tensorTransformer.optimizerStep += 1ULL;

			// Warmup + DDP LR scaling (matches CPU path).
			const float warmupMult = trainingConfig.warmup.multiplier(static_cast<int>(tensorTransformer.optimizerStep));
			const float ddpLRScale = (trainingConfig.ddp.enable && trainingConfig.ddp.linearLRScaling)
			                       ? static_cast<float>(glades::ddp::worldSize()) : 1.0f;
			const float gpuExtraLRMult = warmupMult * ddpLRScale;

			const unsigned int stepInEpoch = (s + 1u) / seqBatchMax;
			lrScheduleMultiplier = transformer_schedule_multiplier(
			    trainingConfig.lrSchedule,
			    epochIdx + lrScheduleEpochOffset,
			    stepInEpoch,
			    optimizerStepsPerEpoch);

			// Global gradient norm clipping on GPU.
			float gradScale = 1.0f;
			const float clipNorm = trainingConfig.globalGradClipNorm;
			if (clipNorm > 0.0f)
			{
				// Use lossSum buffer as temporary accumulator for gradient norm.
				gpu::device_memset_bytes(gpuTransformerScratch->lossSum.data(), 0, sizeof(float));

				// Accumulate sum(g^2) across all gradient buffers.
				if (tokenLM)
				{
					// gTokE stays on FP32 grad path even under Phase-2 (see
					// the embedding_scatter rationale).
					gpu::sum_squared_accumulate(gpuTransformerWeights->gTokE.data(),
					    static_cast<int>(gpuTransformerWeights->gTokE.size()),
					    gpuTransformerScratch->lossSum.data());
					gpu::sum_squared_accumulate(gpuTransformerWeights->gLmBias.data(),
					    static_cast<int>(gpuTransformerWeights->gLmBias.size()),
					    gpuTransformerScratch->lossSum.data());
				}
				else
				{
					if (useBf16GradsPh2_)
					{
						gpu::sum_squared_accumulate_bf16(gpuTransformerWeights->gWIn_bf16.data(),
						    static_cast<int>(gpuTransformerWeights->gWIn_bf16.size()),
						    gpuTransformerScratch->lossSum.data());
						gpu::sum_squared_accumulate_bf16(gpuTransformerWeights->gWOut_bf16.data(),
						    static_cast<int>(gpuTransformerWeights->gWOut_bf16.size()),
						    gpuTransformerScratch->lossSum.data());
					}
					else
					{
						gpu::sum_squared_accumulate(gpuTransformerWeights->gWIn.data(),
						    static_cast<int>(gpuTransformerWeights->gWIn.size()),
						    gpuTransformerScratch->lossSum.data());
						gpu::sum_squared_accumulate(gpuTransformerWeights->gWOut.data(),
						    static_cast<int>(gpuTransformerWeights->gWOut.size()),
						    gpuTransformerScratch->lossSum.data());
					}
					gpu::sum_squared_accumulate(gpuTransformerWeights->gBIn.data(),
					    static_cast<int>(gpuTransformerWeights->gBIn.size()),
					    gpuTransformerScratch->lossSum.data());
					gpu::sum_squared_accumulate(gpuTransformerWeights->gBOut.data(),
					    static_cast<int>(gpuTransformerWeights->gBOut.size()),
					    gpuTransformerScratch->lossSum.data());
				}
				for (unsigned int gli = 0; gli < nLayers; ++gli)
				{
					gpu::GpuTransformerWeights::Block& gb = gpuTransformerWeights->blocks[gli];
					if (useBf16GradsPh2_)
					{
						gpu::sum_squared_accumulate_bf16(gb.gWq_bf16.data(), static_cast<int>(gb.gWq_bf16.size()), gpuTransformerScratch->lossSum.data());
						gpu::sum_squared_accumulate_bf16(gb.gWk_bf16.data(), static_cast<int>(gb.gWk_bf16.size()), gpuTransformerScratch->lossSum.data());
						gpu::sum_squared_accumulate_bf16(gb.gWv_bf16.data(), static_cast<int>(gb.gWv_bf16.size()), gpuTransformerScratch->lossSum.data());
						gpu::sum_squared_accumulate_bf16(gb.gWo_bf16.data(), static_cast<int>(gb.gWo_bf16.size()), gpuTransformerScratch->lossSum.data());
						gpu::sum_squared_accumulate_bf16(gb.gW1_bf16.data(), static_cast<int>(gb.gW1_bf16.size()), gpuTransformerScratch->lossSum.data());
						gpu::sum_squared_accumulate_bf16(gb.gW2_bf16.data(), static_cast<int>(gb.gW2_bf16.size()), gpuTransformerScratch->lossSum.data());
					}
					else
					{
						gpu::sum_squared_accumulate(gb.gWq.data(), static_cast<int>(gb.gWq.size()), gpuTransformerScratch->lossSum.data());
						gpu::sum_squared_accumulate(gb.gWk.data(), static_cast<int>(gb.gWk.size()), gpuTransformerScratch->lossSum.data());
						gpu::sum_squared_accumulate(gb.gWv.data(), static_cast<int>(gb.gWv.size()), gpuTransformerScratch->lossSum.data());
						gpu::sum_squared_accumulate(gb.gWo.data(), static_cast<int>(gb.gWo.size()), gpuTransformerScratch->lossSum.data());
						gpu::sum_squared_accumulate(gb.gW1.data(), static_cast<int>(gb.gW1.size()), gpuTransformerScratch->lossSum.data());
						gpu::sum_squared_accumulate(gb.gW2.data(), static_cast<int>(gb.gW2.size()), gpuTransformerScratch->lossSum.data());
					}
					gpu::sum_squared_accumulate(gb.gBq.data(), static_cast<int>(gb.gBq.size()), gpuTransformerScratch->lossSum.data());
					gpu::sum_squared_accumulate(gb.gBk.data(), static_cast<int>(gb.gBk.size()), gpuTransformerScratch->lossSum.data());
					gpu::sum_squared_accumulate(gb.gBv.data(), static_cast<int>(gb.gBv.size()), gpuTransformerScratch->lossSum.data());
					gpu::sum_squared_accumulate(gb.gBo.data(), static_cast<int>(gb.gBo.size()), gpuTransformerScratch->lossSum.data());
					gpu::sum_squared_accumulate(gb.gB1.data(), static_cast<int>(gb.gB1.size()), gpuTransformerScratch->lossSum.data());
					gpu::sum_squared_accumulate(gb.gB2.data(), static_cast<int>(gb.gB2.size()), gpuTransformerScratch->lossSum.data());
					gpu::sum_squared_accumulate(gb.gLn1Gamma.data(), static_cast<int>(gb.gLn1Gamma.size()), gpuTransformerScratch->lossSum.data());
					gpu::sum_squared_accumulate(gb.gLn1Beta.data(), static_cast<int>(gb.gLn1Beta.size()), gpuTransformerScratch->lossSum.data());
					gpu::sum_squared_accumulate(gb.gLn2Gamma.data(), static_cast<int>(gb.gLn2Gamma.size()), gpuTransformerScratch->lossSum.data());
					gpu::sum_squared_accumulate(gb.gLn2Beta.data(), static_cast<int>(gb.gLn2Beta.size()), gpuTransformerScratch->lossSum.data());
				}
				// Final LayerNorm gradients
				gpu::sum_squared_accumulate(gpuTransformerWeights->gLnFinalGamma.data(),
				    static_cast<int>(gpuTransformerWeights->gLnFinalGamma.size()),
				    gpuTransformerScratch->lossSum.data());
				gpu::sum_squared_accumulate(gpuTransformerWeights->gLnFinalBeta.data(),
				    static_cast<int>(gpuTransformerWeights->gLnFinalBeta.size()),
				    gpuTransformerScratch->lossSum.data());

				float h_sumSq = 0.0f;
				gpu::recordEvent(gpuComputeReadyEvent, gpu::computeStream());
				gpu::streamWaitEvent(gpu::transferStream(), gpuComputeReadyEvent);
				gpuTransformerScratch->lossSum.downloadAsync(&h_sumSq, 1);
				gpu::synchronizeTransferStream();
				const float gradNorm = sqrtf(h_sumSq) * invBatch;
				if (gradNorm > clipNorm)
					gradScale = clipNorm / (gradNorm + 1e-12f);
				lastGradNorm = gradNorm;
				lastGradNormScale = gradScale;
			}

			const bool gpuUseAtlas = (trainingConfig.optimizer.type == glades::OptimizerConfig::ATLAS);
			const bool gpuUseVesta = (trainingConfig.optimizer.type == glades::OptimizerConfig::VESTA);
			const bool gpuUseHelios = (trainingConfig.optimizer.type == glades::OptimizerConfig::HELIOS);
			const bool gpuUseGeode = gpuUseAtlas && trainingConfig.atlas.geodeEnabled;
			const bool gpuUseEcho = gpuUseAtlas && trainingConfig.atlas.echoEnabled;
			const bool gpuUseBiMAP = gpuUseAtlas && trainingConfig.atlas.bimapEnabled;
			const bool gpuUsePact = gpuUseAtlas && trainingConfig.atlas.pactEnabled;
			const bool gpuUseRacer = gpuUseAtlas && trainingConfig.atlas.racerEnabled;
			const bool gpuUseMatra = gpuUseAtlas && trainingConfig.atlas.matraEnabled;
			const bool gpuUseArgos = gpuUseAtlas && trainingConfig.atlas.argosEnabled;
			const bool gpuUseMuon = gpuUseAtlas && trainingConfig.atlas.muonEnabled;
			const bool gpuFuseEcho =
			    gpuUseEcho && !gpuUseGeode && !gpuUseBiMAP && !gpuUsePact && !gpuUseRacer && !gpuUseMatra && !gpuUseArgos && !gpuUseMuon;
			const bool gpuUseGroupAdam =
			    (!gpuUseAtlas) && trainingConfig.optimizer.adamGroupwiseEnabled;

			if (gpuUseAtlas && !gpuUseGeode && !gpuUseEcho && !gpuUseBiMAP && !gpuUsePact && !gpuUseRacer && !gpuUseMatra && !gpuUseArgos && !gpuUseMuon)
			{
			// === GPU ATLAS optimizer ===
			// Weight matrices use atlas_gpu_step (BRSP subspace preconditioning).
			// Biases and LN params use vanilla SGD (matching CPU ATLAS path).
			const glades::ATLASConfig& ac = trainingConfig.atlas;

			bool gpuAtlasError = false;

			// Macro: vanilla SGD for 1D bias/LN param on GPU.
			// W -= lr * invBatch * gradScale * g;  then zero g.
#define GLADES_GPU_SGD_BIAS(param, grad, lr_) do { \
	const int sgd_sz_ = static_cast<int>((param).size()); \
	if (sgd_sz_ > 0) { \
gpu::atlas_gpu_baseline_update((param).data(), (grad).data(), sgd_sz_, (lr_) * invBatch * gradScale); \
gpu::atlas_gpu_guard((param).data(), sgd_sz_); \
(grad).zero(); \
	} \
} while(0)

			// Token embedding (layer index 0)
			if (tokenLM)
			{
				const float lr0 = skeleton->getLearningRate(0u) * lrScheduleMultiplier * gpuExtraLRMult;
				const float wd1_0 = skeleton->getWeightDecay1(0u);
				const float wd2_0 = skeleton->getWeightDecay2(0u);

				if (!gpuAtlasError && !gpu::atlas_gpu_update(gpuTransformerWeights->atlasTokE,
				    gpuTransformerWeights->tokE.data(), gpuTransformerWeights->gTokE.data(),
				    vocabSize, dModel, invBatch, lr0, wd1_0, wd2_0, gradScale,
				    ac, rngEngine, getLogger(), "tr.tokE"))
				    gpuAtlasError = true;

				GLADES_GPU_SGD_BIAS(gpuTransformerWeights->lmBias, gpuTransformerWeights->gLmBias, lr0);
			}

			// Input projection (layer index 0, not used for token-LM models)
			if (!tokenLM)
			{
				const float lr0 = skeleton->getLearningRate(0u) * lrScheduleMultiplier * gpuExtraLRMult;
				const float wd1_0 = skeleton->getWeightDecay1(0u);
				const float wd2_0 = skeleton->getWeightDecay2(0u);

				if (!gpuAtlasError && !gpu::atlas_gpu_update(gpuTransformerWeights->atlasWIn,
				    gpuTransformerWeights->WIn.data(), gpuTransformerWeights->gWIn.data(),
				    dModel, inputSize, invBatch, lr0, wd1_0, wd2_0, gradScale,
				    ac, rngEngine, getLogger(), "tr.WIn"))
				    gpuAtlasError = true;

				GLADES_GPU_SGD_BIAS(gpuTransformerWeights->bIn, gpuTransformerWeights->gBIn, lr0);
			}

			// Per-layer blocks (layer index 1..nLayers)
			for (unsigned int bli = 0; bli < nLayers; ++bli)
			{
				const float lr_l = skeleton->getLearningRate(bli + 1u) * lrScheduleMultiplier * gpuExtraLRMult;
				const float wd1_l = skeleton->getWeightDecay1(bli + 1u);
				const float wd2_l = skeleton->getWeightDecay2(bli + 1u);
				gpu::GpuTransformerWeights::Block& gb = gpuTransformerWeights->blocks[bli];

				if (!gpuAtlasError && !gpu::atlas_gpu_update(gb.atlasWq, gb.Wq.data(), gb.gWq.data(), dModel, dModel, invBatch, lr_l, wd1_l, wd2_l, gradScale, ac, rngEngine, getLogger(), "tr.Wq"))
				    gpuAtlasError = true;
				if (!gpuAtlasError && !gpu::atlas_gpu_update(gb.atlasWk, gb.Wk.data(), gb.gWk.data(), dModelKV, dModel, invBatch, lr_l, wd1_l, wd2_l, gradScale, ac, rngEngine, getLogger(), "tr.Wk"))
				    gpuAtlasError = true;
				if (!gpuAtlasError && !gpu::atlas_gpu_update(gb.atlasWv, gb.Wv.data(), gb.gWv.data(), dModelKV, dModel, invBatch, lr_l, wd1_l, wd2_l, gradScale, ac, rngEngine, getLogger(), "tr.Wv"))
				    gpuAtlasError = true;
				if (!gpuAtlasError && !gpu::atlas_gpu_update(gb.atlasWo, gb.Wo.data(), gb.gWo.data(), dModel, dModel, invBatch, lr_l, wd1_l, wd2_l, gradScale, ac, rngEngine, getLogger(), "tr.Wo"))
				    gpuAtlasError = true;
				if (!gpuAtlasError && !gpu::atlas_gpu_update(gb.atlasW1, gb.W1.data(), gb.gW1.data(), ff1Width, dModel, invBatch, lr_l, wd1_l, wd2_l, gradScale, ac, rngEngine, getLogger(), "tr.W1"))
				    gpuAtlasError = true;
				if (!gpuAtlasError && !gpu::atlas_gpu_update(gb.atlasW2, gb.W2.data(), gb.gW2.data(), dModel, dFF, invBatch, lr_l, wd1_l, wd2_l, gradScale, ac, rngEngine, getLogger(), "tr.W2"))
				    gpuAtlasError = true;

				// Biases and LN params: vanilla SGD (no subspace projection)
				GLADES_GPU_SGD_BIAS(gb.bq, gb.gBq, lr_l);
				GLADES_GPU_SGD_BIAS(gb.bk, gb.gBk, lr_l);
				GLADES_GPU_SGD_BIAS(gb.bv, gb.gBv, lr_l);
				GLADES_GPU_SGD_BIAS(gb.bo, gb.gBo, lr_l);
				GLADES_GPU_SGD_BIAS(gb.b1, gb.gB1, lr_l);
				GLADES_GPU_SGD_BIAS(gb.b2, gb.gB2, lr_l);
				GLADES_GPU_SGD_BIAS(gb.ln1Gamma, gb.gLn1Gamma, lr_l);
				GLADES_GPU_SGD_BIAS(gb.ln1Beta, gb.gLn1Beta, lr_l);
				GLADES_GPU_SGD_BIAS(gb.ln2Gamma, gb.gLn2Gamma, lr_l);
				GLADES_GPU_SGD_BIAS(gb.ln2Beta, gb.gLn2Beta, lr_l);
			}

			// Final LayerNorm (use block 0 LR; no weight decay)
			{
				const float lrLN = skeleton->getLearningRate(1u) * lrScheduleMultiplier * gpuExtraLRMult;
				GLADES_GPU_SGD_BIAS(gpuTransformerWeights->lnFinalGamma, gpuTransformerWeights->gLnFinalGamma, lrLN);
				GLADES_GPU_SGD_BIAS(gpuTransformerWeights->lnFinalBeta, gpuTransformerWeights->gLnFinalBeta, lrLN);
			}

			// Output projection (layer index nLayers, unused in tied-head mode)
			if (!tokenLM)
			{
				const float lrO = skeleton->getLearningRate(nLayers) * lrScheduleMultiplier * gpuExtraLRMult;
				const float wd1_o = skeleton->getWeightDecay1(nLayers);
				const float wd2_o = skeleton->getWeightDecay2(nLayers);

				if (!gpuAtlasError && !gpu::atlas_gpu_update(gpuTransformerWeights->atlasWOut,
				    gpuTransformerWeights->WOut.data(), gpuTransformerWeights->gWOut.data(),
				    outSize, dModel, invBatch, lrO, wd1_o, wd2_o, gradScale,
				    ac, rngEngine, getLogger(), "tr.WOut"))
				    gpuAtlasError = true;

				GLADES_GPU_SGD_BIAS(gpuTransformerWeights->bOut, gpuTransformerWeights->gBOut, lrO);
			}

			// --- GPU ATLAS periodic diagnostics ---
			if (!gpuAtlasError && logger && ac.tSub > 0u)
			{
				const unsigned long long tSubULL = static_cast<unsigned long long>(ac.tSub);

				// Helper lambda-like macro to log a single weight state
#define GLADES_GPU_ATLAS_DIAG(st, tag_str) do { \
	if ((st).initialized && ((st).step % tSubULL) == 0ULL) { \
gpu::AtlasGpuDiag ad_ = gpu::atlas_gpu_get_diag((st)); \
if (ad_.valid) { \
	std::ostringstream oss_; \
	oss_ << "event=gpu_atlas_step tag=" << (tag_str); \
	oss_ << " step=" << ad_.step; \
	oss_ << " m=" << (st).m << " n=" << (st).n << " rank=" << (st).r; \
	oss_ << " mu=" << ad_.mu; \
	oss_ << " sigma2=" << ad_.sigma2; \
	oss_ << " baseline_rate=" << ad_.baselineRate; \
	oss_ << " gz_norm=" << ad_.gzNorm; \
	oss_ << " update_norm=" << ad_.updateNorm; \
	oss_ << " fisher_min=" << ad_.fisherMin; \
	oss_ << " fisher_max=" << ad_.fisherMax; \
	oss_ << " fisher_mean=" << ad_.fisherMean; \
	logger->info("ATLAS", shmea::GString(oss_.str().c_str())); \
} \
	} \
} while(0)

				if (tokenLM)
					GLADES_GPU_ATLAS_DIAG(gpuTransformerWeights->atlasTokE, "tr.tokE");
				if (!tokenLM)
					GLADES_GPU_ATLAS_DIAG(gpuTransformerWeights->atlasWIn, "tr.WIn");

				for (unsigned int bli = 0; bli < nLayers; ++bli)
				{
					gpu::GpuTransformerWeights::Block& gb = gpuTransformerWeights->blocks[bli];
					GLADES_GPU_ATLAS_DIAG(gb.atlasWq, "tr.Wq");
					GLADES_GPU_ATLAS_DIAG(gb.atlasWk, "tr.Wk");
					GLADES_GPU_ATLAS_DIAG(gb.atlasWv, "tr.Wv");
					GLADES_GPU_ATLAS_DIAG(gb.atlasWo, "tr.Wo");
					GLADES_GPU_ATLAS_DIAG(gb.atlasW1, "tr.W1");
					GLADES_GPU_ATLAS_DIAG(gb.atlasW2, "tr.W2");
				}

				if (!tokenLM)
					GLADES_GPU_ATLAS_DIAG(gpuTransformerWeights->atlasWOut, "tr.WOut");

#undef GLADES_GPU_ATLAS_DIAG
			}

			if (gpuAtlasError)
			{
				lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
				    "SGDHelper_TRANSFORMER: GPU ATLAS update failed");
				storeRunningFlag(false);
			}
#undef GLADES_GPU_SGD_BIAS
			}
			else if (gpuUseVesta)
			{
			// === GPU VESTA optimizer ===
			// Weight matrices use vesta_gpu_step (Bregman-mirror spectral update).
			// Biases and LN params use vanilla SGD (matches CPU VESTA path).
			const glades::VestaConfig& vc = trainingConfig.vesta;

			bool gpuVestaError = false;

#define GLADES_GPU_VESTA_SGD_BIAS(param, grad, lr_) do { \
	const int sgd_sz_ = static_cast<int>((param).size()); \
	if (sgd_sz_ > 0) { \
		gpu::atlas_gpu_baseline_update((param).data(), (grad).data(), sgd_sz_, (lr_) * invBatch * gradScale); \
		gpu::atlas_gpu_guard((param).data(), sgd_sz_); \
		(grad).zero(); \
	} \
} while(0)

			// Token embedding (layer index 0).
			if (tokenLM)
			{
				const float lr0 = skeleton->getLearningRate(0u) * lrScheduleMultiplier * gpuExtraLRMult;
				const float wd1_0 = skeleton->getWeightDecay1(0u);
				const float wd2_0 = skeleton->getWeightDecay2(0u);
				if (!gpuTransformerWeights->vestaTokE.initialized)
				{
					if (!gpu::vesta_gpu_init(gpuTransformerWeights->vestaTokE,
					    gpuTransformerWeights->tokE.data(),
					    vocabSize, dModel, vc, rngEngine, getLogger()))
						gpuVestaError = true;
				}
				if (!gpuVestaError && !gpu::vesta_gpu_step(gpuTransformerWeights->vestaTokE,
				    gpuTransformerWeights->tokE.data(), gpuTransformerWeights->gTokE.data(),
				    vocabSize, dModel, invBatch, lr0, wd1_0, wd2_0, gradScale,
				    vc, rngEngine, getLogger(), "tr.tokE"))
					gpuVestaError = true;
				GLADES_GPU_VESTA_SGD_BIAS(gpuTransformerWeights->lmBias, gpuTransformerWeights->gLmBias, lr0);
			}

			// Input projection (for non-tokenLM models).
			if (!tokenLM)
			{
				const float lr0 = skeleton->getLearningRate(0u) * lrScheduleMultiplier * gpuExtraLRMult;
				const float wd1_0 = skeleton->getWeightDecay1(0u);
				const float wd2_0 = skeleton->getWeightDecay2(0u);
				if (!gpuTransformerWeights->vestaWIn.initialized)
				{
					if (!gpu::vesta_gpu_init(gpuTransformerWeights->vestaWIn,
					    gpuTransformerWeights->WIn.data(),
					    dModel, inputSize, vc, rngEngine, getLogger()))
						gpuVestaError = true;
				}
				if (!gpuVestaError && !gpu::vesta_gpu_step(gpuTransformerWeights->vestaWIn,
				    gpuTransformerWeights->WIn.data(), gpuTransformerWeights->gWIn.data(),
				    dModel, inputSize, invBatch, lr0, wd1_0, wd2_0, gradScale,
				    vc, rngEngine, getLogger(), "tr.WIn"))
					gpuVestaError = true;
				GLADES_GPU_VESTA_SGD_BIAS(gpuTransformerWeights->bIn, gpuTransformerWeights->gBIn, lr0);
			}

			// Per-layer blocks.
			for (unsigned int bli = 0; bli < nLayers; ++bli)
			{
				const float lr_l = skeleton->getLearningRate(bli + 1u) * lrScheduleMultiplier * gpuExtraLRMult;
				const float wd1_l = skeleton->getWeightDecay1(bli + 1u);
				const float wd2_l = skeleton->getWeightDecay2(bli + 1u);
				gpu::GpuTransformerWeights::Block& gb = gpuTransformerWeights->blocks[bli];

				#define GLADES_GPU_VESTA_INIT_STEP(st, W, gW, m_, n_, tag) do { \
					if (!gpuVestaError) { \
						if (!(st).initialized) { \
							if (!gpu::vesta_gpu_init((st), (W).data(), (m_), (n_), vc, rngEngine, getLogger())) \
								gpuVestaError = true; \
						} \
					} \
					if (!gpuVestaError && !gpu::vesta_gpu_step((st), (W).data(), (gW).data(), \
					    (m_), (n_), invBatch, lr_l, wd1_l, wd2_l, gradScale, \
					    vc, rngEngine, getLogger(), (tag))) \
						gpuVestaError = true; \
				} while(0)

				GLADES_GPU_VESTA_INIT_STEP(gb.vestaWq, gb.Wq, gb.gWq, dModel, dModel, "tr.Wq");
				GLADES_GPU_VESTA_INIT_STEP(gb.vestaWk, gb.Wk, gb.gWk, dModelKV, dModel, "tr.Wk");
				GLADES_GPU_VESTA_INIT_STEP(gb.vestaWv, gb.Wv, gb.gWv, dModelKV, dModel, "tr.Wv");
				GLADES_GPU_VESTA_INIT_STEP(gb.vestaWo, gb.Wo, gb.gWo, dModel, dModel, "tr.Wo");
				GLADES_GPU_VESTA_INIT_STEP(gb.vestaW1, gb.W1, gb.gW1, ff1Width, dModel, "tr.W1");
				GLADES_GPU_VESTA_INIT_STEP(gb.vestaW2, gb.W2, gb.gW2, dModel, dFF, "tr.W2");
				#undef GLADES_GPU_VESTA_INIT_STEP

				GLADES_GPU_VESTA_SGD_BIAS(gb.bq, gb.gBq, lr_l);
				GLADES_GPU_VESTA_SGD_BIAS(gb.bk, gb.gBk, lr_l);
				GLADES_GPU_VESTA_SGD_BIAS(gb.bv, gb.gBv, lr_l);
				GLADES_GPU_VESTA_SGD_BIAS(gb.bo, gb.gBo, lr_l);
				GLADES_GPU_VESTA_SGD_BIAS(gb.b1, gb.gB1, lr_l);
				GLADES_GPU_VESTA_SGD_BIAS(gb.b2, gb.gB2, lr_l);
				GLADES_GPU_VESTA_SGD_BIAS(gb.ln1Gamma, gb.gLn1Gamma, lr_l);
				GLADES_GPU_VESTA_SGD_BIAS(gb.ln1Beta, gb.gLn1Beta, lr_l);
				GLADES_GPU_VESTA_SGD_BIAS(gb.ln2Gamma, gb.gLn2Gamma, lr_l);
				GLADES_GPU_VESTA_SGD_BIAS(gb.ln2Beta, gb.gLn2Beta, lr_l);
			}

			// Final LayerNorm.
			{
				const float lrLN = skeleton->getLearningRate(1u) * lrScheduleMultiplier * gpuExtraLRMult;
				GLADES_GPU_VESTA_SGD_BIAS(gpuTransformerWeights->lnFinalGamma, gpuTransformerWeights->gLnFinalGamma, lrLN);
				GLADES_GPU_VESTA_SGD_BIAS(gpuTransformerWeights->lnFinalBeta, gpuTransformerWeights->gLnFinalBeta, lrLN);
			}

			// Output projection (skipped under tied heads in tokenLM mode).
			if (!tokenLM)
			{
				const float lrO = skeleton->getLearningRate(nLayers) * lrScheduleMultiplier * gpuExtraLRMult;
				const float wd1_o = skeleton->getWeightDecay1(nLayers);
				const float wd2_o = skeleton->getWeightDecay2(nLayers);
				if (!gpuTransformerWeights->vestaWOut.initialized)
				{
					if (!gpu::vesta_gpu_init(gpuTransformerWeights->vestaWOut,
					    gpuTransformerWeights->WOut.data(),
					    outSize, dModel, vc, rngEngine, getLogger()))
						gpuVestaError = true;
				}
				if (!gpuVestaError && !gpu::vesta_gpu_step(gpuTransformerWeights->vestaWOut,
				    gpuTransformerWeights->WOut.data(), gpuTransformerWeights->gWOut.data(),
				    outSize, dModel, invBatch, lrO, wd1_o, wd2_o, gradScale,
				    vc, rngEngine, getLogger(), "tr.WOut"))
					gpuVestaError = true;
				GLADES_GPU_VESTA_SGD_BIAS(gpuTransformerWeights->bOut, gpuTransformerWeights->gBOut, lrO);
			}

			if (gpuVestaError)
			{
				lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
				    "SGDHelper_TRANSFORMER: GPU VESTA update failed");
				storeRunningFlag(false);
			}
#undef GLADES_GPU_VESTA_SGD_BIAS
			}
			else if (gpuUseHelios)
			{
			// === GPU HELIOS optimizer ===
			// Weight matrices use helios_gpu_step (BAOAB Langevin integrator).
			// Biases and LN params use vanilla SGD (matching CPU HELIOS path).
			const glades::HeliosConfig& hc = trainingConfig.helios;

			bool gpuHeliosError = false;

			// === GPU FD-HVP probe (framework Section 7 step 10) ===
			// Runs before the optimizer step: perturbs one weight matrix
			// by +/- eps along v = p/||p||, replays the last sequence's
			// forward+backward twice, computes kappa = v.(gPlus-gMinus)/(2eps),
			// and EMA-updates state.kappa. The replay call site is currently
			// gated by hc.kHvp > 0 AND a not-yet-implemented helper method
			// `runGpuReplayForwardBackward` which requires extracting the
			// inline forward/backward blocks from transformerGpuTrainEpoch
			// into callable methods (see research/HELIOS_framework.md §14a).
			// The probe kernel infrastructure (helios_gpu_probe_*) is unit-
			// tested and ready; only the replay call-site plumbing remains.
			const bool gpuHvpProbeActive =
			    (hc.alpha > 0.0f && hc.kHvp > 0u);
			if (gpuHvpProbeActive)
			{
				tensorTransformer.heliosHvpStepCounter++;
				if ((tensorTransformer.heliosHvpStepCounter
				     % static_cast<unsigned long long>(hc.kHvp)) == 0ULL)
				{
					// Round-robin target matrix: 6 matrix types × nLayers.
					const unsigned int totalTargets = 6u * nLayers;
					const unsigned int cycleIdx = static_cast<unsigned int>(
					    tensorTransformer.heliosHvpCycleCounter
					    % static_cast<unsigned long long>(totalTargets));
					tensorTransformer.heliosHvpCycleCounter++;
					const unsigned int layerIdx = cycleIdx / 6u;
					const unsigned int mtypeIdx = cycleIdx % 6u;
					gpu::GpuTransformerWeights::Block& gbPrb =
					    gpuTransformerWeights->blocks[layerIdx];

					gpu::GpuBuffer<float>* W_b = 0;
					gpu::GpuBuffer<float>* gW_b = 0;
					gpu::GpuHeliosWeightState* hst = 0;
					unsigned int pM = 0u, pN = 0u;
					switch (mtypeIdx)
					{
						case 0: W_b=&gbPrb.Wq; gW_b=&gbPrb.gWq;
						        hst=&gbPrb.heliosWq; pM=dModel;    pN=dModel; break;
						case 1: W_b=&gbPrb.Wk; gW_b=&gbPrb.gWk;
						        hst=&gbPrb.heliosWk; pM=dModelKV;  pN=dModel; break;
						case 2: W_b=&gbPrb.Wv; gW_b=&gbPrb.gWv;
						        hst=&gbPrb.heliosWv; pM=dModelKV;  pN=dModel; break;
						case 3: W_b=&gbPrb.Wo; gW_b=&gbPrb.gWo;
						        hst=&gbPrb.heliosWo; pM=dModel;    pN=dModel; break;
						case 4: W_b=&gbPrb.W1; gW_b=&gbPrb.gW1;
						        hst=&gbPrb.heliosW1; pM=ff1Width;  pN=dModel; break;
						case 5: W_b=&gbPrb.W2; gW_b=&gbPrb.gW2;
						        hst=&gbPrb.heliosW2; pM=dModel;    pN=dFF;    break;
					}

					if (W_b && gW_b && hst && hst->initialized
					    && W_b->allocated() && W_b->size() == pM * pN
					    && hst->p.allocated() && hst->p.size() == pM * pN)
					{
						// Ensure probe scratch buffers are allocated.
						const size_t Np = static_cast<size_t>(pM) * pN;
						gpu::GpuBuffer<float> dWsave, dV, dGPlus, dGMinus;
						if (!dWsave.allocate(Np) || !dV.allocate(Np)
						    || !dGPlus.allocate(Np) || !dGMinus.allocate(Np))
						{
							// Allocation failure; skip this probe event.
						}
						else
						{
							float pNormOut = 0.0f;
							// Scalar-FD HVP (framework §7 step 10, lighter-
							// cost variant): kappa = (L_plus + L_minus - 2 L_0) / eps^2
							// where L_* is the per-sequence NLL sum at
							// +/- eps * v. Requires only 2 extra forwards
							// (no backward) per probe event on the last
							// sequence. Uses transformerGpuRunForwardOnly
							// which was extracted from this epoch's inline
							// forward block in 2026-04-21.
							if (heliosProbeLastSeqValid
							    && heliosProbeLastSeqT > 0u
							    && gpu::helios_gpu_probe_compute_v(
							           hst->p.data(), dV.data(),
							           pM, pN, pNormOut)
							    && pNormOut > 1e-6f
							    && gpu::helios_gpu_probe_snapshot_W(
							           dWsave.data(), W_b->data(), pM, pN))
							{
								const float epsFd = 1e-3f;
								const float L0 = heliosProbeLastSeqLoss0;
								const unsigned int Tprb = heliosProbeLastSeqT;

								// + perturbation.
								float Lplus = 0.0f, Lminus = 0.0f;
								bool probeOk =
								    gpu::helios_gpu_probe_perturb(
								        W_b->data(), dWsave.data(),
								        dV.data(), epsFd, +1.0f, pM, pN);
								if (probeOk)
								{
									// Refresh BF16 mirror of the perturbed
									// target if mixed precision is active.
									if (cfg.mpEnable)
										gpuTransformerWeights->ensureLowpMirrors();
									probeOk = transformerGpuRunForwardOnly(
									    cfg, Tprb,
									    useBf16, useRope,
									    bf16WIn, bf16Wq, bf16Wk, bf16Wv,
									    bf16Wo, bf16W1, bf16W2, bf16Head,
									    ropeDimOverride, gpuPerf);
								}
								if (probeOk)
								{
									// Compute per-seq NLL from probs+targets.
									// gpuTargetsT was populated for this seq.
									gpu::collect_token_lm_metrics(
									    gpuTransformerScratch->probs.data(),
									    gpuTransformerScratch->gpuTargetsT.data(),
									    static_cast<int>(Tprb),
									    static_cast<int>(vocabSize),
									    padTokenId,
									    gpuTransformerScratch->lossPack.data());
									int lp[4];
									gpuTransformerScratch->lossPack.downloadAsync(lp, 4);
									gpu::synchronizeTransferStream();
									memcpy(&Lplus, &lp[0], sizeof(float));
								}

								// - perturbation.
								if (probeOk)
								{
									probeOk = gpu::helios_gpu_probe_perturb(
									    W_b->data(), dWsave.data(),
									    dV.data(), epsFd, -1.0f, pM, pN);
								}
								if (probeOk)
								{
									if (cfg.mpEnable)
										gpuTransformerWeights->ensureLowpMirrors();
									probeOk = transformerGpuRunForwardOnly(
									    cfg, Tprb,
									    useBf16, useRope,
									    bf16WIn, bf16Wq, bf16Wk, bf16Wv,
									    bf16Wo, bf16W1, bf16W2, bf16Head,
									    ropeDimOverride, gpuPerf);
								}
								if (probeOk)
								{
									gpu::collect_token_lm_metrics(
									    gpuTransformerScratch->probs.data(),
									    gpuTransformerScratch->gpuTargetsT.data(),
									    static_cast<int>(Tprb),
									    static_cast<int>(vocabSize),
									    padTokenId,
									    gpuTransformerScratch->lossPack.data());
									int lm[4];
									gpuTransformerScratch->lossPack.downloadAsync(lm, 4);
									gpu::synchronizeTransferStream();
									memcpy(&Lminus, &lm[0], sizeof(float));
								}

								// Restore unperturbed weights unconditionally.
								gpu::helios_gpu_probe_restore_W(
								    W_b->data(), dWsave.data(), pM, pN);
								if (cfg.mpEnable)
									gpuTransformerWeights->ensureLowpMirrors();

								if (probeOk)
								{
									const float kappa =
									    (Lplus + Lminus - 2.0f * L0)
									    / (epsFd * epsFd);
									// GpuHeliosWeightState::kappa is a host
									// scalar; helios::updateSharpness operates
									// on CPU WeightState. Do the EMA inline
									// here to avoid creating a CPU stub state.
									if (kappa == kappa) // finite check
									{
										float clipped = kappa;
										if (clipped < 0.0f) clipped = 0.0f;
										if (clipped > hc.kappaMax)
											clipped = hc.kappaMax;
										const float betaK = 0.05f;
										hst->kappa =
										    (1.0f - betaK) * hst->kappa
										    + betaK * clipped;
									}
								}
							}
						}
					}
				}
			}

#define GLADES_GPU_HELIOS_SGD_BIAS(param, grad, lr_) do { \
	const int sgd_sz_ = static_cast<int>((param).size()); \
	if (sgd_sz_ > 0) { \
		gpu::atlas_gpu_baseline_update((param).data(), (grad).data(), sgd_sz_, (lr_) * invBatch * gradScale); \
		gpu::atlas_gpu_guard((param).data(), sgd_sz_); \
		(grad).zero(); \
	} \
} while(0)

			// Token embedding (tied-head LM).
			if (tokenLM)
			{
				const float lr0 = skeleton->getLearningRate(0u) * lrScheduleMultiplier * gpuExtraLRMult;
				const float wd1_0 = skeleton->getWeightDecay1(0u);
				const float wd2_0 = skeleton->getWeightDecay2(0u);
				if (!gpuTransformerWeights->heliosTokE.initialized)
				{
					if (!gpu::helios_gpu_init(gpuTransformerWeights->heliosTokE,
					    gpuTransformerWeights->tokE.data(),
					    vocabSize, dModel, hc, rngEngine, getLogger()))
						gpuHeliosError = true;
				}
				if (!gpuHeliosError && !gpu::helios_gpu_step(gpuTransformerWeights->heliosTokE,
				    gpuTransformerWeights->tokE.data(), gpuTransformerWeights->gTokE.data(),
				    vocabSize, dModel, invBatch, lr0, wd1_0, wd2_0, gradScale,
				    hc, rngEngine, getLogger(), "tr.tokE"))
					gpuHeliosError = true;
				GLADES_GPU_HELIOS_SGD_BIAS(gpuTransformerWeights->lmBias, gpuTransformerWeights->gLmBias, lr0);
			}

			// Input projection (non-tokenLM).
			if (!tokenLM)
			{
				const float lr0 = skeleton->getLearningRate(0u) * lrScheduleMultiplier * gpuExtraLRMult;
				const float wd1_0 = skeleton->getWeightDecay1(0u);
				const float wd2_0 = skeleton->getWeightDecay2(0u);
				if (!gpuTransformerWeights->heliosWIn.initialized)
				{
					if (!gpu::helios_gpu_init(gpuTransformerWeights->heliosWIn,
					    gpuTransformerWeights->WIn.data(),
					    dModel, inputSize, hc, rngEngine, getLogger()))
						gpuHeliosError = true;
				}
				if (!gpuHeliosError && !gpu::helios_gpu_step(gpuTransformerWeights->heliosWIn,
				    gpuTransformerWeights->WIn.data(), gpuTransformerWeights->gWIn.data(),
				    dModel, inputSize, invBatch, lr0, wd1_0, wd2_0, gradScale,
				    hc, rngEngine, getLogger(), "tr.WIn"))
					gpuHeliosError = true;
				GLADES_GPU_HELIOS_SGD_BIAS(gpuTransformerWeights->bIn, gpuTransformerWeights->gBIn, lr0);
			}

			// Per-layer blocks.
			for (unsigned int bli = 0; bli < nLayers; ++bli)
			{
				const float lr_l = skeleton->getLearningRate(bli + 1u) * lrScheduleMultiplier * gpuExtraLRMult;
				const float wd1_l = skeleton->getWeightDecay1(bli + 1u);
				const float wd2_l = skeleton->getWeightDecay2(bli + 1u);
				gpu::GpuTransformerWeights::Block& gb = gpuTransformerWeights->blocks[bli];

				#define GLADES_GPU_HELIOS_INIT_STEP(st, W, gW, m_, n_, tag) do { \
					if (!gpuHeliosError) { \
						if (!(st).initialized) { \
							if (!gpu::helios_gpu_init((st), (W).data(), (m_), (n_), hc, rngEngine, getLogger())) \
								gpuHeliosError = true; \
						} \
					} \
					if (!gpuHeliosError && !gpu::helios_gpu_step((st), (W).data(), (gW).data(), \
					    (m_), (n_), invBatch, lr_l, wd1_l, wd2_l, gradScale, \
					    hc, rngEngine, getLogger(), (tag))) \
						gpuHeliosError = true; \
				} while(0)

				GLADES_GPU_HELIOS_INIT_STEP(gb.heliosWq, gb.Wq, gb.gWq, dModel, dModel, "tr.Wq");
				GLADES_GPU_HELIOS_INIT_STEP(gb.heliosWk, gb.Wk, gb.gWk, dModelKV, dModel, "tr.Wk");
				GLADES_GPU_HELIOS_INIT_STEP(gb.heliosWv, gb.Wv, gb.gWv, dModelKV, dModel, "tr.Wv");
				GLADES_GPU_HELIOS_INIT_STEP(gb.heliosWo, gb.Wo, gb.gWo, dModel, dModel, "tr.Wo");
				GLADES_GPU_HELIOS_INIT_STEP(gb.heliosW1, gb.W1, gb.gW1, ff1Width, dModel, "tr.W1");
				GLADES_GPU_HELIOS_INIT_STEP(gb.heliosW2, gb.W2, gb.gW2, dModel, dFF, "tr.W2");
				#undef GLADES_GPU_HELIOS_INIT_STEP

				GLADES_GPU_HELIOS_SGD_BIAS(gb.bq, gb.gBq, lr_l);
				GLADES_GPU_HELIOS_SGD_BIAS(gb.bk, gb.gBk, lr_l);
				GLADES_GPU_HELIOS_SGD_BIAS(gb.bv, gb.gBv, lr_l);
				GLADES_GPU_HELIOS_SGD_BIAS(gb.bo, gb.gBo, lr_l);
				GLADES_GPU_HELIOS_SGD_BIAS(gb.b1, gb.gB1, lr_l);
				GLADES_GPU_HELIOS_SGD_BIAS(gb.b2, gb.gB2, lr_l);
				GLADES_GPU_HELIOS_SGD_BIAS(gb.ln1Gamma, gb.gLn1Gamma, lr_l);
				GLADES_GPU_HELIOS_SGD_BIAS(gb.ln1Beta, gb.gLn1Beta, lr_l);
				GLADES_GPU_HELIOS_SGD_BIAS(gb.ln2Gamma, gb.gLn2Gamma, lr_l);
				GLADES_GPU_HELIOS_SGD_BIAS(gb.ln2Beta, gb.gLn2Beta, lr_l);
			}

			// Final LayerNorm.
			{
				const float lrLN = skeleton->getLearningRate(1u) * lrScheduleMultiplier * gpuExtraLRMult;
				GLADES_GPU_HELIOS_SGD_BIAS(gpuTransformerWeights->lnFinalGamma, gpuTransformerWeights->gLnFinalGamma, lrLN);
				GLADES_GPU_HELIOS_SGD_BIAS(gpuTransformerWeights->lnFinalBeta, gpuTransformerWeights->gLnFinalBeta, lrLN);
			}

			// Output projection.
			if (!tokenLM)
			{
				const float lrO = skeleton->getLearningRate(nLayers) * lrScheduleMultiplier * gpuExtraLRMult;
				const float wd1_o = skeleton->getWeightDecay1(nLayers);
				const float wd2_o = skeleton->getWeightDecay2(nLayers);
				if (!gpuTransformerWeights->heliosWOut.initialized)
				{
					if (!gpu::helios_gpu_init(gpuTransformerWeights->heliosWOut,
					    gpuTransformerWeights->WOut.data(),
					    outSize, dModel, hc, rngEngine, getLogger()))
						gpuHeliosError = true;
				}
				if (!gpuHeliosError && !gpu::helios_gpu_step(gpuTransformerWeights->heliosWOut,
				    gpuTransformerWeights->WOut.data(), gpuTransformerWeights->gWOut.data(),
				    outSize, dModel, invBatch, lrO, wd1_o, wd2_o, gradScale,
				    hc, rngEngine, getLogger(), "tr.WOut"))
					gpuHeliosError = true;
				GLADES_GPU_HELIOS_SGD_BIAS(gpuTransformerWeights->bOut, gpuTransformerWeights->gBOut, lrO);
			}

			if (gpuHeliosError)
			{
				lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
				    "SGDHelper_TRANSFORMER: GPU HELIOS update failed");
				storeRunningFlag(false);
			}
#undef GLADES_GPU_HELIOS_SGD_BIAS
			}
			else
			{
			// === Batched Adam optimizer ===
			const glades::ATLASConfig& ac = trainingConfig.atlas;
			const float beta1 = trainingConfig.optimizer.adamBeta1;
			const float beta2 = trainingConfig.optimizer.adamBeta2;
			const float adamEps = trainingConfig.optimizer.adamEps;
			const int stepInt = static_cast<int>(tensorTransformer.optimizerStep);
			const double b1t = std::pow(static_cast<double>(beta1),
			                            static_cast<double>(tensorTransformer.optimizerStep));
			const double b2t = std::pow(static_cast<double>(beta2),
			                            static_cast<double>(tensorTransformer.optimizerStep));
			const float inv1mB1t = static_cast<float>(1.0 / (1.0 - b1t));
			const float inv1mB2t = static_cast<float>(1.0 / (1.0 - b2t));

			// BF16 optimizer state: when enabled, the 9 large weight matrices
			// (tokE, WIn, WOut, per-layer Wq/Wk/Wv/Wo/W1/W2) use BF16 m, v
			// buffers via adam_update_bf16_state (per-matrix, not batched),
			// halving their optimizer-state VRAM. Biases + LN params stay in
			// the batched FP32 path (their total size is < 1% of the model).
			const bool useInt8AdamState =
			    trainingConfig.mixedPrecision.adamStateInt8;
			// FACE Adafactor on the embedding (paradigm #28).  When set, the
			// tokE update path replaces dense Adam (m/v) with the FACE
			// preconditioner — but the OTHER 8 large weight tensors still
			// need a quantized per-matrix dispatch (see allocate logic in
			// gpu_transformer_state.cu, which only allocates bf16/int8
			// buffers for those when bf16/int8 flag is set).  Therefore
			// FACE alone is not a valid configuration for tokenLM mode;
			// the trainer should pass at least one of --adam-state-bf16
			// or --adam-state-int8 alongside --face-embedding.
			const bool useFaceEmbedding =
			    trainingConfig.transformer.faceEmbedding;
			// int8 path takes precedence over bf16 if both flags are set; the
			// per-matrix dispatch below picks one or the other.  When int8 is
			// active we still treat "useBf16AdamState" as true for the
			// "exclude these tensors from the batched FP32 path" decision —
			// they have no FP32 m/v buffers in either case.
			const bool useBf16AdamState =
			    trainingConfig.mixedPrecision.adamStateBf16
			    || useInt8AdamState
			    || useFaceEmbedding;

			// Build device pointer arrays on first step (pointers are fixed after GPU alloc).
			if (!gpuTransformerWeights->adamPtrsUploaded)
			{
				// 6 base + 16 per layer (standard) + 3 per layer (#76 MLA when active).
				const int mlaExtraPerLayer = (trainingConfig.transformer.mlaLatentDim > 0) ? 3 : 0;
				const int maxAdamGroups = 6 + (16 + mlaExtraPerLayer) * static_cast<int>(nLayers);
				std::vector<float*> hParams(static_cast<size_t>(maxAdamGroups), static_cast<float*>(NULL));
				std::vector<float*> hGrads(static_cast<size_t>(maxAdamGroups), static_cast<float*>(NULL));
				std::vector<float*> hMs(static_cast<size_t>(maxAdamGroups), static_cast<float*>(NULL));
				std::vector<float*> hVs(static_cast<size_t>(maxAdamGroups), static_cast<float*>(NULL));
				std::vector<float> hBaseLrs(static_cast<size_t>(maxAdamGroups), 0.0f);
				std::vector<float> hWds(static_cast<size_t>(maxAdamGroups), 0.0f);
				std::vector<int> hSizes(static_cast<size_t>(maxAdamGroups), 0);
				int gc = 0;
				int maxSz = 0;
				bool adamGroupOverflow = false;

#define GLADES_ADD_ADAM_GROUP_SAFE(p, g, m, v, sz, baseLr, wd) do { \
	if ((sz) > 0) { \
		if (gc >= maxAdamGroups) { \
			adamGroupOverflow = true; \
		} else { \
			hParams[static_cast<size_t>(gc)] = (p); \
			hGrads[static_cast<size_t>(gc)] = (g); \
			hMs[static_cast<size_t>(gc)] = (m); \
			hVs[static_cast<size_t>(gc)] = (v); \
			hBaseLrs[static_cast<size_t>(gc)] = (baseLr); \
			hWds[static_cast<size_t>(gc)] = (wd); \
			hSizes[static_cast<size_t>(gc)] = (sz); \
			if ((sz) > maxSz) maxSz = (sz); \
			++gc; \
		} \
	} \
} while (0)

				if (tokenLM)
				{
					const float lr0Base = skeleton->getLearningRate(0u);
					const float wd0 = skeleton->getWeightDecay2(0u);
					if (!useBf16AdamState)
						GLADES_ADD_ADAM_GROUP_SAFE(gpuTransformerWeights->tokE.data(),
						                           gpuTransformerWeights->gTokE.data(),
						                           gpuTransformerWeights->vTokE.data(),
						                           gpuTransformerWeights->v2TokE.data(),
						                           static_cast<int>(static_cast<size_t>(vocabSize) * dModel),
						                           lr0Base, wd0);
					GLADES_ADD_ADAM_GROUP_SAFE(gpuTransformerWeights->lmBias.data(),
					                           gpuTransformerWeights->gLmBias.data(),
					                           gpuTransformerWeights->mLmBias.data(),
					                           gpuTransformerWeights->v2LmBias.data(),
					                           static_cast<int>(vocabSize),
					                           lr0Base, 0.0f);
				}
				{
					const float lr0Base = skeleton->getLearningRate(0u);
					const float wd0 = skeleton->getWeightDecay2(0u);
					if (!useBf16AdamState)
						// Same Phase-2-retire null-pointer guard: gate on FP32 grad
						// size, not weight size.
						GLADES_ADD_ADAM_GROUP_SAFE(gpuTransformerWeights->WIn.data(),
						                           gpuTransformerWeights->gWIn.data(),
						                           gpuTransformerWeights->vWIn.data(),
						                           gpuTransformerWeights->v2WIn.data(),
						                           static_cast<int>(gpuTransformerWeights->gWIn.size()),
						                           lr0Base, wd0);
					GLADES_ADD_ADAM_GROUP_SAFE(gpuTransformerWeights->bIn.data(),
					                           gpuTransformerWeights->gBIn.data(),
					                           gpuTransformerWeights->mBIn.data(),
					                           gpuTransformerWeights->v2BIn.data(),
					                           static_cast<int>(gpuTransformerWeights->bIn.size()),
					                           lr0Base, 0.0f);
				}
				for (unsigned int bli = 0; bli < nLayers; ++bli)
				{
					gpu::GpuTransformerWeights::Block& gb = gpuTransformerWeights->blocks[bli];
					const float lrBase = skeleton->getLearningRate(bli + 1u);
					const float wdBase = skeleton->getWeightDecay2(bli + 1u);
					if (!useBf16AdamState)
					{
						// Gate group registration on the FP32 GRAD buffer's size,
						// not the weight's: under Phase-2 retire, gb.gW*.size() == 0
						// (FP32 grad buffer not allocated, gb.gW*.data() == NULL),
						// while gb.W*.size() stays positive.  Using Wq.size() here
						// would register a group with a NULL grad pointer, causing
						// the batched Adam kernel to dereference NULL.  When Phase-2
						// retire is on, the per-tensor bf16-grad Adam dispatch at
						// line ~13226 (GLADES_BF16_ADAM_BIG_BF16GRAD) handles these
						// tensors via gb.gW*_bf16 instead.
						GLADES_ADD_ADAM_GROUP_SAFE(gb.Wq.data(), gb.gWq.data(), gb.vWq.data(), gb.v2Wq.data(), static_cast<int>(gb.gWq.size()), lrBase, wdBase);
						GLADES_ADD_ADAM_GROUP_SAFE(gb.Wk.data(), gb.gWk.data(), gb.vWk.data(), gb.v2Wk.data(), static_cast<int>(gb.gWk.size()), lrBase, wdBase);
						GLADES_ADD_ADAM_GROUP_SAFE(gb.Wv.data(), gb.gWv.data(), gb.vWv.data(), gb.v2Wv.data(), static_cast<int>(gb.gWv.size()), lrBase, wdBase);
						GLADES_ADD_ADAM_GROUP_SAFE(gb.Wo.data(), gb.gWo.data(), gb.vWo.data(), gb.v2Wo.data(), static_cast<int>(gb.gWo.size()), lrBase, wdBase);
						GLADES_ADD_ADAM_GROUP_SAFE(gb.W1.data(), gb.gW1.data(), gb.vW1.data(), gb.v2W1.data(), static_cast<int>(gb.gW1.size()), lrBase, wdBase);
						GLADES_ADD_ADAM_GROUP_SAFE(gb.W2.data(), gb.gW2.data(), gb.vW2.data(), gb.v2W2.data(), static_cast<int>(gb.gW2.size()), lrBase, wdBase);
						// Paradigm shift #76 MLA Adam updates (when active).
						if (gb.Wdkv.allocated() && gb.vWdkv.allocated()) {
							GLADES_ADD_ADAM_GROUP_SAFE(gb.Wdkv.data(), gb.gWdkv.data(), gb.vWdkv.data(), gb.v2Wdkv.data(), static_cast<int>(gb.Wdkv.size()), lrBase, wdBase);
							GLADES_ADD_ADAM_GROUP_SAFE(gb.Wuk.data(),  gb.gWuk.data(),  gb.vWuk.data(),  gb.v2Wuk.data(),  static_cast<int>(gb.Wuk.size()),  lrBase, wdBase);
							GLADES_ADD_ADAM_GROUP_SAFE(gb.Wuv.data(),  gb.gWuv.data(),  gb.vWuv.data(),  gb.v2Wuv.data(),  static_cast<int>(gb.Wuv.size()),  lrBase, wdBase);
						}
					}
					GLADES_ADD_ADAM_GROUP_SAFE(gb.bq.data(), gb.gBq.data(), gb.mBq.data(), gb.v2Bq.data(), static_cast<int>(gb.bq.size()), lrBase, 0.0f);
					GLADES_ADD_ADAM_GROUP_SAFE(gb.bk.data(), gb.gBk.data(), gb.mBk.data(), gb.v2Bk.data(), static_cast<int>(gb.bk.size()), lrBase, 0.0f);
					GLADES_ADD_ADAM_GROUP_SAFE(gb.bv.data(), gb.gBv.data(), gb.mBv.data(), gb.v2Bv.data(), static_cast<int>(gb.bv.size()), lrBase, 0.0f);
					GLADES_ADD_ADAM_GROUP_SAFE(gb.bo.data(), gb.gBo.data(), gb.mBo.data(), gb.v2Bo.data(), static_cast<int>(gb.bo.size()), lrBase, 0.0f);
					GLADES_ADD_ADAM_GROUP_SAFE(gb.b1.data(), gb.gB1.data(), gb.mB1.data(), gb.v2B1.data(), static_cast<int>(gb.b1.size()), lrBase, 0.0f);
					GLADES_ADD_ADAM_GROUP_SAFE(gb.b2.data(), gb.gB2.data(), gb.mB2.data(), gb.v2B2.data(), static_cast<int>(gb.b2.size()), lrBase, 0.0f);
					GLADES_ADD_ADAM_GROUP_SAFE(gb.ln1Gamma.data(), gb.gLn1Gamma.data(), gb.mLn1Gamma.data(), gb.v2Ln1Gamma.data(), static_cast<int>(gb.ln1Gamma.size()), lrBase, 0.0f);
					GLADES_ADD_ADAM_GROUP_SAFE(gb.ln1Beta.data(), gb.gLn1Beta.data(), gb.mLn1Beta.data(), gb.v2Ln1Beta.data(), static_cast<int>(gb.ln1Beta.size()), lrBase, 0.0f);
					GLADES_ADD_ADAM_GROUP_SAFE(gb.ln2Gamma.data(), gb.gLn2Gamma.data(), gb.mLn2Gamma.data(), gb.v2Ln2Gamma.data(), static_cast<int>(gb.ln2Gamma.size()), lrBase, 0.0f);
					GLADES_ADD_ADAM_GROUP_SAFE(gb.ln2Beta.data(), gb.gLn2Beta.data(), gb.mLn2Beta.data(), gb.v2Ln2Beta.data(), static_cast<int>(gb.ln2Beta.size()), lrBase, 0.0f);
				}
				// Final LayerNorm
				const float lrLnBase = skeleton->getLearningRate(1u);
				GLADES_ADD_ADAM_GROUP_SAFE(gpuTransformerWeights->lnFinalGamma.data(),
				                           gpuTransformerWeights->gLnFinalGamma.data(),
				                           gpuTransformerWeights->mLnFinalGamma.data(),
				                           gpuTransformerWeights->v2LnFinalGamma.data(),
				                           static_cast<int>(gpuTransformerWeights->lnFinalGamma.size()),
				                           lrLnBase, 0.0f);
				GLADES_ADD_ADAM_GROUP_SAFE(gpuTransformerWeights->lnFinalBeta.data(),
				                           gpuTransformerWeights->gLnFinalBeta.data(),
				                           gpuTransformerWeights->mLnFinalBeta.data(),
				                           gpuTransformerWeights->v2LnFinalBeta.data(),
				                           static_cast<int>(gpuTransformerWeights->lnFinalBeta.size()),
				                           lrLnBase, 0.0f);
				if (!tokenLM)
				{
					const float lrOutBase = skeleton->getLearningRate(nLayers);
					const float wdOut = skeleton->getWeightDecay2(nLayers);
					if (!useBf16AdamState)
						// Phase-2-retire null-pointer guard: gate on FP32 grad size.
						GLADES_ADD_ADAM_GROUP_SAFE(gpuTransformerWeights->WOut.data(),
						                           gpuTransformerWeights->gWOut.data(),
						                           gpuTransformerWeights->vWOut.data(),
						                           gpuTransformerWeights->v2WOut.data(),
						                           static_cast<int>(gpuTransformerWeights->gWOut.size()),
						                           lrOutBase, wdOut);
					GLADES_ADD_ADAM_GROUP_SAFE(gpuTransformerWeights->bOut.data(),
					                           gpuTransformerWeights->gBOut.data(),
					                           gpuTransformerWeights->mBOut.data(),
					                           gpuTransformerWeights->v2BOut.data(),
					                           static_cast<int>(gpuTransformerWeights->bOut.size()),
					                           lrOutBase, 0.0f);
				}
				if (adamGroupOverflow)
				{
					lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
					    "transformerGpuTrainEpoch: Adam group packing overflow");
					storeRunningFlag(false);
					return;
				}

				gpu::device_memcpy_h2d(gpuTransformerWeights->d_adamParams, hParams.data(), gc * sizeof(float*));
				gpu::device_memcpy_h2d(gpuTransformerWeights->d_adamGrads, hGrads.data(), gc * sizeof(float*));
				gpu::device_memcpy_h2d(gpuTransformerWeights->d_adamM, hMs.data(), gc * sizeof(float*));
				gpu::device_memcpy_h2d(gpuTransformerWeights->d_adamV, hVs.data(), gc * sizeof(float*));
				gpu::device_memcpy_h2d(gpuTransformerWeights->d_adamBaseLr, hBaseLrs.data(), gc * sizeof(float));
				gpu::device_memcpy_h2d(gpuTransformerWeights->d_adamWd, hWds.data(), gc * sizeof(float));
				gpu::device_memcpy_h2d(gpuTransformerWeights->d_adamSizes, hSizes.data(), gc * sizeof(int));
				if (!gpu::synchronizeTransferStream())
				{
					lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
					    "transformerGpuTrainEpoch: failed to upload Adam group pointers");
					storeRunningFlag(false);
					return;
				}
				gpu::recordEvent(gpuTransferReadyEvent, gpu::transferStream());
				gpu::streamWaitEvent(gpu::computeStream(), gpuTransferReadyEvent);
				gpuTransformerWeights->adamGroupCount = gc;
				gpuTransformerWeights->adamMaxSize = maxSz;
				gpuTransformerWeights->adamPtrsUploaded = true;

#undef GLADES_ADD_ADAM_GROUP_SAFE
			}

			if (gpuFuseEcho
			    && (!gpuTransformerWeights->adamMetricMetaUploaded
			        || gpuTransformerWeights->adamMetricScope != ac.echoScope))
			{
				if (!gpuTransformerWeights->ensureEchoBuffers())
				{
					lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
					    "transformerGpuTrainEpoch: GPU ECHO metric buffer allocation failed");
					storeRunningFlag(false);
					return;
				}
				const int gc = gpuTransformerWeights->adamGroupCount;
				std::vector<float*> hRowSecond(static_cast<size_t>(gc), static_cast<float*>(NULL));
				std::vector<float*> hColSecond(static_cast<size_t>(gc), static_cast<float*>(NULL));
				std::vector<float*> hRowMetrics(static_cast<size_t>(gc), static_cast<float*>(NULL));
				std::vector<float*> hColMetrics(static_cast<size_t>(gc), static_cast<float*>(NULL));
				std::vector<float*> hRowStructMetrics(static_cast<size_t>(gc), static_cast<float*>(NULL));
				std::vector<float*> hColStructMetrics(static_cast<size_t>(gc), static_cast<float*>(NULL));
				std::vector<float*> hPrevMhat(static_cast<size_t>(gc), static_cast<float*>(NULL));
				std::vector<float*> hMetricScratch(static_cast<size_t>(gc), static_cast<float*>(NULL));
				std::vector<int> hMetricRows(static_cast<size_t>(gc), 0);
				std::vector<int> hMetricCols(static_cast<size_t>(gc), 0);
				int gi = 0;

#define GLADES_SET_STATIC_ECHO_META(enabled_, state_, rows_, cols_) do { \
	if ((enabled_)) { \
		hRowSecond[static_cast<size_t>(gi)] = (state_).rowSecond.data(); \
		hColSecond[static_cast<size_t>(gi)] = (state_).colSecond.data(); \
		hRowMetrics[static_cast<size_t>(gi)] = (state_).rowInvMetric.data(); \
		hColMetrics[static_cast<size_t>(gi)] = (state_).colInvMetric.data(); \
		hRowStructMetrics[static_cast<size_t>(gi)] = (state_).rowStructInvMetric.data(); \
		hColStructMetrics[static_cast<size_t>(gi)] = (state_).colStructInvMetric.data(); \
		hPrevMhat[static_cast<size_t>(gi)] = (state_).prevMhat.data(); \
		hMetricScratch[static_cast<size_t>(gi)] = (state_).scalarScratch.data(); \
		hMetricRows[static_cast<size_t>(gi)] = static_cast<int>(rows_); \
		hMetricCols[static_cast<size_t>(gi)] = static_cast<int>(cols_); \
	} \
	++gi; \
} while (0)

				if (tokenLM)
				{
					GLADES_SET_STATIC_ECHO_META(
					    echo_scope_uses_head_matrix(ac, vocabSize, dModel),
					    gpuTransformerWeights->echoTokE, vocabSize, dModel);
					GLADES_SET_STATIC_ECHO_META(false, gpuTransformerWeights->echoTokE, 0u, 0u);
				}
				{
					if (gpuTransformerWeights->WIn.size() > 0)
						GLADES_SET_STATIC_ECHO_META(
						    echo_scope_uses_input_matrix(ac, dModel, inputSize),
						    gpuTransformerWeights->echoWIn, dModel, inputSize);
					if (gpuTransformerWeights->bIn.size() > 0)
						GLADES_SET_STATIC_ECHO_META(false, gpuTransformerWeights->echoWIn, 0u, 0u);
				}
				for (unsigned int bli = 0; bli < nLayers; ++bli)
				{
					gpu::GpuTransformerWeights::Block& gb = gpuTransformerWeights->blocks[bli];
					GLADES_SET_STATIC_ECHO_META(echo_scope_uses_decoder_matrix(ac, bli, nLayers, dModel, dModel), gb.echoWq, dModel, dModel);
					GLADES_SET_STATIC_ECHO_META(echo_scope_uses_decoder_matrix(ac, bli, nLayers, dModelKV, dModel), gb.echoWk, dModelKV, dModel);
					GLADES_SET_STATIC_ECHO_META(echo_scope_uses_decoder_matrix(ac, bli, nLayers, dModelKV, dModel), gb.echoWv, dModelKV, dModel);
					GLADES_SET_STATIC_ECHO_META(echo_scope_uses_decoder_matrix(ac, bli, nLayers, dModel, dModel), gb.echoWo, dModel, dModel);
					GLADES_SET_STATIC_ECHO_META(echo_scope_uses_decoder_matrix(ac, bli, nLayers, ff1Width, dModel), gb.echoW1, ff1Width, dModel);
					GLADES_SET_STATIC_ECHO_META(echo_scope_uses_decoder_matrix(ac, bli, nLayers, dModel, dFF), gb.echoW2, dModel, dFF);
					for (int b = 0; b < 10; ++b)
						GLADES_SET_STATIC_ECHO_META(false, gb.echoWq, 0u, 0u);
				}
				GLADES_SET_STATIC_ECHO_META(false, gpuTransformerWeights->echoTokE, 0u, 0u);
				GLADES_SET_STATIC_ECHO_META(false, gpuTransformerWeights->echoTokE, 0u, 0u);
				if (!tokenLM)
				{
					if (gpuTransformerWeights->WOut.size() > 0)
						GLADES_SET_STATIC_ECHO_META(
						    echo_scope_uses_head_matrix(ac, outSize, dModel),
						    gpuTransformerWeights->echoWOut, outSize, dModel);
					if (gpuTransformerWeights->bOut.size() > 0)
						GLADES_SET_STATIC_ECHO_META(false, gpuTransformerWeights->echoWOut, 0u, 0u);
				}
				if (gi != gc)
				{
					lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
					    "transformerGpuTrainEpoch: Adam ECHO metadata group count mismatch");
					storeRunningFlag(false);
					return;
				}

				gpu::device_memcpy_h2d(gpuTransformerWeights->d_adamRowSecond, hRowSecond.data(), gc * sizeof(float*));
				gpu::device_memcpy_h2d(gpuTransformerWeights->d_adamColSecond, hColSecond.data(), gc * sizeof(float*));
				gpu::device_memcpy_h2d(gpuTransformerWeights->d_adamRowMetric, hRowMetrics.data(), gc * sizeof(float*));
				gpu::device_memcpy_h2d(gpuTransformerWeights->d_adamColMetric, hColMetrics.data(), gc * sizeof(float*));
				gpu::device_memcpy_h2d(gpuTransformerWeights->d_adamRowStructMetric, hRowStructMetrics.data(), gc * sizeof(float*));
				gpu::device_memcpy_h2d(gpuTransformerWeights->d_adamColStructMetric, hColStructMetrics.data(), gc * sizeof(float*));
				gpu::device_memcpy_h2d(gpuTransformerWeights->d_adamPrevMhat, hPrevMhat.data(), gc * sizeof(float*));
				gpu::device_memcpy_h2d(gpuTransformerWeights->d_adamMetricScratch, hMetricScratch.data(), gc * sizeof(float*));
				gpu::device_memcpy_h2d(gpuTransformerWeights->d_adamMetricRows, hMetricRows.data(), gc * sizeof(int));
				gpu::device_memcpy_h2d(gpuTransformerWeights->d_adamMetricCols, hMetricCols.data(), gc * sizeof(int));
				if (!gpu::synchronizeTransferStream())
				{
					lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
					    "transformerGpuTrainEpoch: failed to upload static ECHO metadata");
					storeRunningFlag(false);
					return;
				}
				gpu::recordEvent(gpuTransferReadyEvent, gpu::transferStream());
				gpu::streamWaitEvent(gpu::computeStream(), gpuTransferReadyEvent);
				gpuTransformerWeights->adamMetricMetaUploaded = true;
				gpuTransformerWeights->adamMetricScope = ac.echoScope;

#undef GLADES_SET_STATIC_ECHO_META
			}

			if (gpuFuseEcho)
			{
				if (!gpu::echo_gpu_prepare_metrics_batch(
				        gpuTransformerWeights->d_adamRowSecond,
				        gpuTransformerWeights->d_adamColSecond,
				        gpuTransformerWeights->d_adamRowMetric,
				        gpuTransformerWeights->d_adamColMetric,
				        gpuTransformerWeights->d_adamRowStructMetric,
				        gpuTransformerWeights->d_adamColStructMetric,
				        gpuTransformerWeights->d_adamMetricScratch,
				        gpuTransformerWeights->d_adamMetricRows,
				        gpuTransformerWeights->d_adamMetricCols,
				        gpuTransformerWeights->adamGroupCount,
				        tensorTransformer.optimizerStep,
				        adamEps,
				        ac))
				{
					lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
					    "SGDHelper_TRANSFORMER: GPU ECHO metric preparation failed");
					storeRunningFlag(false);
					return;
				}
				if (ac.echoShouldRefresh(tensorTransformer.optimizerStep))
				{
					const float echoGeomScale =
					    ac.echoEffectiveGeometryScale(tensorTransformer.optimizerStep);

#define GLADES_SET_ECHO_SCALE(enabled_, state_) do { \
	if ((enabled_)) \
		(state_).lastGeometryScale = echoGeomScale; \
} while (0)

					if (tokenLM)
						GLADES_SET_ECHO_SCALE(
						    echo_scope_uses_head_matrix(ac, vocabSize, dModel),
						    gpuTransformerWeights->echoTokE);
					else
						GLADES_SET_ECHO_SCALE(
						    echo_scope_uses_input_matrix(ac, dModel, inputSize),
						    gpuTransformerWeights->echoWIn);

					for (unsigned int bli = 0; bli < nLayers; ++bli)
					{
						gpu::GpuTransformerWeights::Block& gb = gpuTransformerWeights->blocks[bli];
						GLADES_SET_ECHO_SCALE(echo_scope_uses_decoder_matrix(ac, bli, nLayers, dModel, dModel), gb.echoWq);
						GLADES_SET_ECHO_SCALE(echo_scope_uses_decoder_matrix(ac, bli, nLayers, dModelKV, dModel), gb.echoWk);
						GLADES_SET_ECHO_SCALE(echo_scope_uses_decoder_matrix(ac, bli, nLayers, dModelKV, dModel), gb.echoWv);
						GLADES_SET_ECHO_SCALE(echo_scope_uses_decoder_matrix(ac, bli, nLayers, dModel, dModel), gb.echoWo);
						GLADES_SET_ECHO_SCALE(echo_scope_uses_decoder_matrix(ac, bli, nLayers, ff1Width, dModel), gb.echoW1);
						GLADES_SET_ECHO_SCALE(echo_scope_uses_decoder_matrix(ac, bli, nLayers, dModel, dFF), gb.echoW2);
					}

					if (!tokenLM)
						GLADES_SET_ECHO_SCALE(
						    echo_scope_uses_head_matrix(ac, outSize, dModel),
						    gpuTransformerWeights->echoWOut);

#undef GLADES_SET_ECHO_SCALE
				}
			}

			// Launch single batched kernel using static per-group lr/wd metadata
			// and the current scalar schedule multiplier.
			{
				const int gc = gpuTransformerWeights->adamGroupCount;
				const float adamLrScale = lrScheduleMultiplier * gpuExtraLRMult;
				if (gpuPerf)
					gpu::perfRecordKernel(&gpuPerf->counters, gpuUseGroupAdam ? 2u : 1u);

				const float* dAdamGroupScales = NULL;
				if (gpuUseGroupAdam)
				{
					gpu::adam_group_scale_batch(
					    gpuTransformerWeights->d_adamParams,
					    gpuTransformerWeights->d_adamGrads,
					    gpuTransformerWeights->d_adamM,
					    gpuTransformerWeights->d_adamV,
					    gpuTransformerWeights->d_adamGroupScales,
					    gpuTransformerWeights->d_adamGroupPrevStepRms,
					    gpuTransformerWeights->d_adamSizes,
					    beta1, beta2, adamEps,
					    invBatch * gradScale, stepInt, gc,
					    trainingConfig.optimizer.adamGroupMinSize,
					    trainingConfig.optimizer.adamGroupStabilityScale,
					    trainingConfig.optimizer.adamGroupSnrScale,
					    trainingConfig.optimizer.adamGroupRatioScale,
					    trainingConfig.optimizer.adamGroupMinScale,
					    trainingConfig.optimizer.adamGroupMaxScale);
					dAdamGroupScales = gpuTransformerWeights->d_adamGroupScales;
				}

				if (trainingConfig.optimizer.type == glades::OptimizerConfig::SOPHIA_G)
				{
					// Paradigm shift #55 SOPHIA-G: clipped second-order rule
					// using g² as Hessian proxy.  Reuses Adam's m/v slots
					// (h takes v's place).  Per Liu 2023 defaults.
					gpu::sophia_g_update_batch(
					    gpuTransformerWeights->d_adamParams,
					    gpuTransformerWeights->d_adamGrads,
					    gpuTransformerWeights->d_adamM,
					    gpuTransformerWeights->d_adamV,   // ← reused as h
					    gpuTransformerWeights->d_adamBaseLr,
					    gpuTransformerWeights->d_adamWd,
					    adamLrScale,
					    gpuTransformerWeights->d_adamSizes,
					    gpuTransformerWeights->adamMaxSize,
					    /*beta1=*/0.965f, /*beta2=*/0.99f,
					    trainingConfig.optimizer.sophiaGamma,
					    trainingConfig.optimizer.sophiaRho,
					    adamEps,
					    invBatch * gradScale, stepInt, gc);
				}
				else
				{
					gpu::adam_update_batch(
					    gpuTransformerWeights->d_adamParams,
					    gpuTransformerWeights->d_adamGrads,
					    gpuTransformerWeights->d_adamM,
					    gpuTransformerWeights->d_adamV,
					    gpuTransformerWeights->d_adamBaseLr,
					    gpuTransformerWeights->d_adamWd,
					    adamLrScale,
					    dAdamGroupScales,
					    gpuTransformerWeights->d_adamSizes,
					    gpuTransformerWeights->adamMaxSize,
					    gpuFuseEcho ? gpuTransformerWeights->d_adamRowMetric : NULL,
					    gpuFuseEcho ? gpuTransformerWeights->d_adamColMetric : NULL,
					    gpuFuseEcho ? gpuTransformerWeights->d_adamRowStructMetric : NULL,
					    gpuFuseEcho ? gpuTransformerWeights->d_adamColStructMetric : NULL,
					    gpuFuseEcho ? gpuTransformerWeights->d_adamPrevMhat : NULL,
					    gpuFuseEcho ? gpuTransformerWeights->d_adamMetricScratch : NULL,
					    gpuFuseEcho ? gpuTransformerWeights->d_adamMetricRows : NULL,
					    gpuFuseEcho ? gpuTransformerWeights->d_adamMetricCols : NULL,
					    beta1, beta2, adamEps,
					    invBatch * gradScale, stepInt, gc);
				}
			}

			// --- BF16 / int8 Adam state dispatch (large weight matrices) ---
			// When adamStateBf16 or adamStateInt8 is enabled, the 9 large
			// weight matrices were excluded from the batched Adam above. Run
			// per-matrix Adam for each.  Math is identical up to the
			// quantization noise on the m, v EMAs; weight + grad remain FP32
			// (or BF16 when MixedPrecisionConfig::gradStorageBf16 is set —
			// the cast pass below mirrors each FP32 grad to its BF16 buffer
			// just before dispatch, then the bf16grad Adam variants read
			// from the BF16 mirror).
			const bool useBf16Grads_ = trainingConfig.mixedPrecision.gradStorageBf16;
			const bool useBf16Weights_ = trainingConfig.mixedPrecision.weightStorageBf16
			    && useBf16AdamState && useBf16Grads_;
			// Stochastic-rounding seed for bf16-weights Adam.  Combines model
			// pointer + per-tensor step counter so per-(model,step,tensor)
			// noise is deterministic.
			const uint32_t bf16WeightSrSeed = 0xC0FFEE42u
			    ^ static_cast<uint32_t>(reinterpret_cast<uintptr_t>(gpuTransformerWeights) >> 4);
			if (useBf16AdamState)
			{
				const float bigLrScale = lrScheduleMultiplier * gpuExtraLRMult;
				const float gradScaleEff = invBatch * gradScale;

				// Cast every FP32 grad to its BF16 mirror once per Adam step
				// when bf16-grad mode is active.  Persistent FP32 grads still
				// exist (kept for grad-norm + clip + this cast); freeing them
				// is the next phase of memory work — see
				// research/PATH_TO_1.84B_FLAGSHIP.md §1.
				//
				// Phase-2 cast scope:
				//   - per-block weights (Wq/Wk/Wv/Wo/W1/W2): scratch+commit done
				//     in backward; skip cast here.
				//   - gWIn, gWOut (single GEMM writer each): scratch+commit done
				//     in backward; skip cast here.
				//   - gTokE: stays on FP32 path (per-element bf16 atomic-add
				//     for the embedding scatter accumulates too much round-off).
				//     Cast it here under Phase-2 just like Phase-1.
				const bool useBf16GradsPh2_castSite =
				    trainingConfig.mixedPrecision.gradStorageBf16Phase2;
				if (useBf16Grads_)
				{
					if (tokenLM)
					{
						gpu::cast_f32_to_bf16(gpuTransformerWeights->gTokE.data(),
						                     gpuTransformerWeights->gTokE_bf16.data(),
						                     gpuTransformerWeights->gTokE.size());
					}
					else if (!useBf16GradsPh2_castSite)
					{
						gpu::cast_f32_to_bf16(gpuTransformerWeights->gWIn.data(),
						                     gpuTransformerWeights->gWIn_bf16.data(),
						                     gpuTransformerWeights->gWIn.size());
						gpu::cast_f32_to_bf16(gpuTransformerWeights->gWOut.data(),
						                     gpuTransformerWeights->gWOut_bf16.data(),
						                     gpuTransformerWeights->gWOut.size());
					}
					if (!useBf16GradsPh2_castSite)
					{
						for (unsigned int bli = 0; bli < nLayers; ++bli)
						{
							gpu::GpuTransformerWeights::Block& gb = gpuTransformerWeights->blocks[bli];
							gpu::cast_f32_to_bf16(gb.gWq.data(), gb.gWq_bf16.data(), gb.gWq.size());
							gpu::cast_f32_to_bf16(gb.gWk.data(), gb.gWk_bf16.data(), gb.gWk.size());
							gpu::cast_f32_to_bf16(gb.gWv.data(), gb.gWv_bf16.data(), gb.gWv.size());
							gpu::cast_f32_to_bf16(gb.gWo.data(), gb.gWo_bf16.data(), gb.gWo.size());
							gpu::cast_f32_to_bf16(gb.gW1.data(), gb.gW1_bf16.data(), gb.gW1.size());
							gpu::cast_f32_to_bf16(gb.gW2.data(), gb.gW2_bf16.data(), gb.gW2.size());
						}
					}
				}
#define GLADES_BF16_ADAM_BIG(W, gW, vBf, v2Bf, baseLr_, wd_) do { \
	const int sz_ = static_cast<int>((W).size()); \
	if (sz_ > 0 && (vBf).allocated() && (v2Bf).allocated()) { \
		gpu::adam_update_bf16_state((W).data(), (gW).data(), \
		    (vBf).data(), (v2Bf).data(), \
		    (baseLr_) * bigLrScale, beta1, beta2, adamEps, \
		    (wd_), gradScaleEff, stepInt, sz_); \
	} \
} while (0)
#define GLADES_INT8_ADAM_BIG(W, gW, vI8, v2I8, vSc, v2Sc, baseLr_, wd_) do { \
	const int sz_ = static_cast<int>((W).size()); \
	if (sz_ > 0 && (vI8).allocated() && (v2I8).allocated()) { \
		gpu::adam_update_int8_state((W).data(), (gW).data(), \
		    (vI8).data(), (v2I8).data(), \
		    (vSc).data(), (v2Sc).data(), \
		    (baseLr_) * bigLrScale, beta1, beta2, adamEps, \
		    (wd_), gradScaleEff, stepInt, sz_); \
	} \
} while (0)
#define GLADES_BF16_ADAM_BIG_BF16GRAD(W, gWb16, vBf, v2Bf, baseLr_, wd_) do { \
	const int sz_ = static_cast<int>((W).size()); \
	if (sz_ > 0 && (vBf).allocated() && (v2Bf).allocated() && (gWb16).allocated()) { \
		gpu::adam_update_bf16_state_bf16grad((W).data(), (gWb16).data(), \
		    (vBf).data(), (v2Bf).data(), \
		    (baseLr_) * bigLrScale, beta1, beta2, adamEps, \
		    (wd_), gradScaleEff, stepInt, sz_); \
	} \
} while (0)
#define GLADES_INT8_ADAM_BIG_BF16GRAD(W, gWb16, vI8, v2I8, vSc, v2Sc, baseLr_, wd_) do { \
	const int sz_ = static_cast<int>((W).size()); \
	if (sz_ > 0 && (vI8).allocated() && (v2I8).allocated() && (gWb16).allocated()) { \
		gpu::adam_update_int8_state_bf16grad((W).data(), (gWb16).data(), \
		    (vI8).data(), (v2I8).data(), \
		    (vSc).data(), (v2Sc).data(), \
		    gpuTransformerScratch->gradScratchFp32.data(), \
		    (baseLr_) * bigLrScale, beta1, beta2, adamEps, \
		    (wd_), gradScaleEff, stepInt, sz_); \
	} \
} while (0)
/* BF16-WEIGHT variants: param IS the bf16 mirror; weight scratch holds
 * the per-step FP32 working copy.  Caller provides Wb16 (bf16 mirror,
 * canonical) instead of FP32 W.  Size taken from Wb16. */
#define GLADES_BF16_ADAM_BIG_BF16GRAD_BF16W(Wb16, gWb16, vBf, v2Bf, baseLr_, wd_) do { \
	const int sz_ = static_cast<int>((Wb16).size()); \
	if (sz_ > 0 && (vBf).allocated() && (v2Bf).allocated() && (gWb16).allocated() && (Wb16).allocated()) { \
		gpu::adam_update_bf16_state_bf16grad_bf16w((Wb16).data(), \
		    gpuTransformerScratch->weightScratchFp32.data(), \
		    (gWb16).data(), \
		    (vBf).data(), (v2Bf).data(), \
		    (baseLr_) * bigLrScale, beta1, beta2, adamEps, \
		    (wd_), gradScaleEff, stepInt, sz_, \
		    bf16WeightSrSeed, static_cast<uint32_t>(stepInt)); \
	} \
} while (0)
#define GLADES_INT8_ADAM_BIG_BF16GRAD_BF16W(Wb16, gWb16, vI8, v2I8, vSc, v2Sc, baseLr_, wd_) do { \
	const int sz_ = static_cast<int>((Wb16).size()); \
	if (sz_ > 0 && (vI8).allocated() && (v2I8).allocated() && (gWb16).allocated() && (Wb16).allocated()) { \
		gpu::adam_update_int8_state_bf16grad_bf16w((Wb16).data(), \
		    gpuTransformerScratch->weightScratchFp32.data(), \
		    (gWb16).data(), \
		    (vI8).data(), (v2I8).data(), \
		    (vSc).data(), (v2Sc).data(), \
		    gpuTransformerScratch->gradScratchFp32.data(), \
		    (baseLr_) * bigLrScale, beta1, beta2, adamEps, \
		    (wd_), gradScaleEff, stepInt, sz_, \
		    bf16WeightSrSeed, static_cast<uint32_t>(stepInt)); \
	} \
} while (0)

				if (tokenLM)
				{
					const float lr0Base = skeleton->getLearningRate(0u);
					const float wd0 = skeleton->getWeightDecay2(0u);
					// FACE Adafactor takes precedence over int8/bf16 dense Adam
					// on the embedding when faceEmbedding is set.  State is
					// orders-of-magnitude smaller; update is sparsity-invariant.
					if (trainingConfig.transformer.faceEmbedding
					    && gpuTransformerWeights->faceZnBar.allocated()) {
						const unsigned int V = (unsigned int)tensorTransformer.vocabSize;
						const unsigned int m = (unsigned int)tensorTransformer.dModel;
						const float bRow = trainingConfig.transformer.faceBetaRow;
						const float bCol = trainingConfig.transformer.faceBetaCol;
						const float fEps = trainingConfig.transformer.faceEps;
						const float lrEff = lr0Base * bigLrScale;
						// 1) compute fresh stats from gTokE
						gpu::face_compute_sparse_stats(
						    gpuTransformerWeights->gTokE.data(), V, m,
						    gpuTransformerWeights->faceZnNew.data(),
						    gpuTransformerWeights->faceDnRaw.data(),
						    gpuTransformerWeights->faceQStep.data(),
						    gpuTransformerWeights->faceGFStep.data());
						// 2) blend into EMAs
						gpu::face_update_emas(
						    gpuTransformerWeights->faceZnBar.data(),
						    gpuTransformerWeights->faceDnBar.data(),
						    gpuTransformerWeights->faceQHat.data(),
						    gpuTransformerWeights->faceGFHat.data(),
						    gpuTransformerWeights->faceZnNew.data(),
						    gpuTransformerWeights->faceDnRaw.data(),
						    gpuTransformerWeights->faceQStep.data(),
						    gpuTransformerWeights->faceGFStep.data(),
						    V, m, bRow, bCol);
						// 3) apply preconditioned update — skip on step 0
						//    where q̂, gF̄ are still zero from cudaMemset; the
						//    blend above seeds them so step 1 uses real values.
						if (stepInt > 0) {
							gpu::face_apply_preconditioned_update(
							    gpuTransformerWeights->tokE.data(),
							    gpuTransformerWeights->gTokE.data(),
							    gpuTransformerWeights->faceZnBar.data(),
							    gpuTransformerWeights->faceDnBar.data(),
							    gpuTransformerWeights->faceQHat.data(),
							    gpuTransformerWeights->faceGFHat.data(),
							    V, m, lrEff, fEps);
						}
					// Note: tokE EXCLUDED from bf16w dispatch.  embedding_gather
					// (forward) reads tokE FP32 master directly, no BF16 variant
					// exists; routing tokE through bf16w would leave the gather
					// reading stale FP32 weights.  Until embedding_gather_bf16
					// lands, tokE stays on the FP32-master + cast-pass path.
					} else if (useInt8AdamState && useBf16Grads_) {
						GLADES_INT8_ADAM_BIG_BF16GRAD(gpuTransformerWeights->tokE,
						                              gpuTransformerWeights->gTokE_bf16,
						                              gpuTransformerWeights->vTokE_int8,
						                              gpuTransformerWeights->v2TokE_int8,
						                              gpuTransformerWeights->vTokEScale,
						                              gpuTransformerWeights->v2TokEScale,
						                              lr0Base, wd0);
					} else if (useInt8AdamState) {
						GLADES_INT8_ADAM_BIG(gpuTransformerWeights->tokE,
						                     gpuTransformerWeights->gTokE,
						                     gpuTransformerWeights->vTokE_int8,
						                     gpuTransformerWeights->v2TokE_int8,
						                     gpuTransformerWeights->vTokEScale,
						                     gpuTransformerWeights->v2TokEScale,
						                     lr0Base, wd0);
					} else if (useBf16Grads_) {
						GLADES_BF16_ADAM_BIG_BF16GRAD(gpuTransformerWeights->tokE,
						                              gpuTransformerWeights->gTokE_bf16,
						                              gpuTransformerWeights->vTokE_bf16,
						                              gpuTransformerWeights->v2TokE_bf16,
						                              lr0Base, wd0);
					} else {
						GLADES_BF16_ADAM_BIG(gpuTransformerWeights->tokE,
						                     gpuTransformerWeights->gTokE,
						                     gpuTransformerWeights->vTokE_bf16,
						                     gpuTransformerWeights->v2TokE_bf16,
						                     lr0Base, wd0);
					}
				}
				else
				{
					const float lr0Base = skeleton->getLearningRate(0u);
					const float wd0 = skeleton->getWeightDecay2(0u);
					if (useBf16Weights_ && useInt8AdamState) {
						GLADES_INT8_ADAM_BIG_BF16GRAD_BF16W(gpuTransformerWeights->WInLowp,
						                                    gpuTransformerWeights->gWIn_bf16,
						                                    gpuTransformerWeights->vWIn_int8,
						                                    gpuTransformerWeights->v2WIn_int8,
						                                    gpuTransformerWeights->vWInScale,
						                                    gpuTransformerWeights->v2WInScale,
						                                    lr0Base, wd0);
					} else if (useBf16Weights_) {
						GLADES_BF16_ADAM_BIG_BF16GRAD_BF16W(gpuTransformerWeights->WInLowp,
						                                    gpuTransformerWeights->gWIn_bf16,
						                                    gpuTransformerWeights->vWIn_bf16,
						                                    gpuTransformerWeights->v2WIn_bf16,
						                                    lr0Base, wd0);
					} else if (useInt8AdamState && useBf16Grads_) {
						GLADES_INT8_ADAM_BIG_BF16GRAD(gpuTransformerWeights->WIn,
						                              gpuTransformerWeights->gWIn_bf16,
						                              gpuTransformerWeights->vWIn_int8,
						                              gpuTransformerWeights->v2WIn_int8,
						                              gpuTransformerWeights->vWInScale,
						                              gpuTransformerWeights->v2WInScale,
						                              lr0Base, wd0);
					} else if (useInt8AdamState) {
						GLADES_INT8_ADAM_BIG(gpuTransformerWeights->WIn,
						                     gpuTransformerWeights->gWIn,
						                     gpuTransformerWeights->vWIn_int8,
						                     gpuTransformerWeights->v2WIn_int8,
						                     gpuTransformerWeights->vWInScale,
						                     gpuTransformerWeights->v2WInScale,
						                     lr0Base, wd0);
					} else if (useBf16Grads_) {
						GLADES_BF16_ADAM_BIG_BF16GRAD(gpuTransformerWeights->WIn,
						                              gpuTransformerWeights->gWIn_bf16,
						                              gpuTransformerWeights->vWIn_bf16,
						                              gpuTransformerWeights->v2WIn_bf16,
						                              lr0Base, wd0);
					} else {
						GLADES_BF16_ADAM_BIG(gpuTransformerWeights->WIn,
						                     gpuTransformerWeights->gWIn,
						                     gpuTransformerWeights->vWIn_bf16,
						                     gpuTransformerWeights->v2WIn_bf16,
						                     lr0Base, wd0);
					}
					const float lrOutBase = skeleton->getLearningRate(nLayers);
					const float wdOut = skeleton->getWeightDecay2(nLayers);
					if (useBf16Weights_ && useInt8AdamState) {
						GLADES_INT8_ADAM_BIG_BF16GRAD_BF16W(gpuTransformerWeights->WOutLowp,
						                                    gpuTransformerWeights->gWOut_bf16,
						                                    gpuTransformerWeights->vWOut_int8,
						                                    gpuTransformerWeights->v2WOut_int8,
						                                    gpuTransformerWeights->vWOutScale,
						                                    gpuTransformerWeights->v2WOutScale,
						                                    lrOutBase, wdOut);
					} else if (useBf16Weights_) {
						GLADES_BF16_ADAM_BIG_BF16GRAD_BF16W(gpuTransformerWeights->WOutLowp,
						                                    gpuTransformerWeights->gWOut_bf16,
						                                    gpuTransformerWeights->vWOut_bf16,
						                                    gpuTransformerWeights->v2WOut_bf16,
						                                    lrOutBase, wdOut);
					} else if (useInt8AdamState && useBf16Grads_) {
						GLADES_INT8_ADAM_BIG_BF16GRAD(gpuTransformerWeights->WOut,
						                              gpuTransformerWeights->gWOut_bf16,
						                              gpuTransformerWeights->vWOut_int8,
						                              gpuTransformerWeights->v2WOut_int8,
						                              gpuTransformerWeights->vWOutScale,
						                              gpuTransformerWeights->v2WOutScale,
						                              lrOutBase, wdOut);
					} else if (useInt8AdamState) {
						GLADES_INT8_ADAM_BIG(gpuTransformerWeights->WOut,
						                     gpuTransformerWeights->gWOut,
						                     gpuTransformerWeights->vWOut_int8,
						                     gpuTransformerWeights->v2WOut_int8,
						                     gpuTransformerWeights->vWOutScale,
						                     gpuTransformerWeights->v2WOutScale,
						                     lrOutBase, wdOut);
					} else if (useBf16Grads_) {
						GLADES_BF16_ADAM_BIG_BF16GRAD(gpuTransformerWeights->WOut,
						                              gpuTransformerWeights->gWOut_bf16,
						                              gpuTransformerWeights->vWOut_bf16,
						                              gpuTransformerWeights->v2WOut_bf16,
						                              lrOutBase, wdOut);
					} else {
						GLADES_BF16_ADAM_BIG(gpuTransformerWeights->WOut,
						                     gpuTransformerWeights->gWOut,
						                     gpuTransformerWeights->vWOut_bf16,
						                     gpuTransformerWeights->v2WOut_bf16,
						                     lrOutBase, wdOut);
					}
				}
				for (unsigned int bli = 0; bli < nLayers; ++bli)
				{
					gpu::GpuTransformerWeights::Block& gb = gpuTransformerWeights->blocks[bli];
					const float lrBase = skeleton->getLearningRate(bli + 1u);
					const float wdBase = skeleton->getWeightDecay2(bli + 1u);
					if (useBf16Weights_ && useInt8AdamState) {
						GLADES_INT8_ADAM_BIG_BF16GRAD_BF16W(gb.WqLowp, gb.gWq_bf16, gb.vWq_int8, gb.v2Wq_int8, gb.vWqScale, gb.v2WqScale, lrBase, wdBase);
						GLADES_INT8_ADAM_BIG_BF16GRAD_BF16W(gb.WkLowp, gb.gWk_bf16, gb.vWk_int8, gb.v2Wk_int8, gb.vWkScale, gb.v2WkScale, lrBase, wdBase);
						GLADES_INT8_ADAM_BIG_BF16GRAD_BF16W(gb.WvLowp, gb.gWv_bf16, gb.vWv_int8, gb.v2Wv_int8, gb.vWvScale, gb.v2WvScale, lrBase, wdBase);
						GLADES_INT8_ADAM_BIG_BF16GRAD_BF16W(gb.WoLowp, gb.gWo_bf16, gb.vWo_int8, gb.v2Wo_int8, gb.vWoScale, gb.v2WoScale, lrBase, wdBase);
						// W1/W2 EXCLUDED from bf16w — paradigm #74 binary-FFN forward
						// (sgd_transformer.cpp lines 9534, 9544, 9591, 9601) reads
						// gb.W1.data()/gb.W2.data() (FP32 master) directly to compute
						// the sign() discretization.  Stays on Phase-2 (FP32 grads
						// retired but FP32 weights still alive + cast pass).
						GLADES_INT8_ADAM_BIG_BF16GRAD(gb.W1, gb.gW1_bf16, gb.vW1_int8, gb.v2W1_int8, gb.vW1Scale, gb.v2W1Scale, lrBase, wdBase);
						GLADES_INT8_ADAM_BIG_BF16GRAD(gb.W2, gb.gW2_bf16, gb.vW2_int8, gb.v2W2_int8, gb.vW2Scale, gb.v2W2Scale, lrBase, wdBase);
					} else if (useBf16Weights_) {
						GLADES_BF16_ADAM_BIG_BF16GRAD_BF16W(gb.WqLowp, gb.gWq_bf16, gb.vWq_bf16, gb.v2Wq_bf16, lrBase, wdBase);
						GLADES_BF16_ADAM_BIG_BF16GRAD_BF16W(gb.WkLowp, gb.gWk_bf16, gb.vWk_bf16, gb.v2Wk_bf16, lrBase, wdBase);
						GLADES_BF16_ADAM_BIG_BF16GRAD_BF16W(gb.WvLowp, gb.gWv_bf16, gb.vWv_bf16, gb.v2Wv_bf16, lrBase, wdBase);
						GLADES_BF16_ADAM_BIG_BF16GRAD_BF16W(gb.WoLowp, gb.gWo_bf16, gb.vWo_bf16, gb.v2Wo_bf16, lrBase, wdBase);
						GLADES_BF16_ADAM_BIG_BF16GRAD(gb.W1, gb.gW1_bf16, gb.vW1_bf16, gb.v2W1_bf16, lrBase, wdBase);
						GLADES_BF16_ADAM_BIG_BF16GRAD(gb.W2, gb.gW2_bf16, gb.vW2_bf16, gb.v2W2_bf16, lrBase, wdBase);
					} else if (useInt8AdamState && useBf16Grads_) {
						GLADES_INT8_ADAM_BIG_BF16GRAD(gb.Wq, gb.gWq_bf16, gb.vWq_int8, gb.v2Wq_int8, gb.vWqScale, gb.v2WqScale, lrBase, wdBase);
						GLADES_INT8_ADAM_BIG_BF16GRAD(gb.Wk, gb.gWk_bf16, gb.vWk_int8, gb.v2Wk_int8, gb.vWkScale, gb.v2WkScale, lrBase, wdBase);
						GLADES_INT8_ADAM_BIG_BF16GRAD(gb.Wv, gb.gWv_bf16, gb.vWv_int8, gb.v2Wv_int8, gb.vWvScale, gb.v2WvScale, lrBase, wdBase);
						GLADES_INT8_ADAM_BIG_BF16GRAD(gb.Wo, gb.gWo_bf16, gb.vWo_int8, gb.v2Wo_int8, gb.vWoScale, gb.v2WoScale, lrBase, wdBase);
						GLADES_INT8_ADAM_BIG_BF16GRAD(gb.W1, gb.gW1_bf16, gb.vW1_int8, gb.v2W1_int8, gb.vW1Scale, gb.v2W1Scale, lrBase, wdBase);
						GLADES_INT8_ADAM_BIG_BF16GRAD(gb.W2, gb.gW2_bf16, gb.vW2_int8, gb.v2W2_int8, gb.vW2Scale, gb.v2W2Scale, lrBase, wdBase);
					} else if (useInt8AdamState) {
						GLADES_INT8_ADAM_BIG(gb.Wq, gb.gWq, gb.vWq_int8, gb.v2Wq_int8, gb.vWqScale, gb.v2WqScale, lrBase, wdBase);
						GLADES_INT8_ADAM_BIG(gb.Wk, gb.gWk, gb.vWk_int8, gb.v2Wk_int8, gb.vWkScale, gb.v2WkScale, lrBase, wdBase);
						GLADES_INT8_ADAM_BIG(gb.Wv, gb.gWv, gb.vWv_int8, gb.v2Wv_int8, gb.vWvScale, gb.v2WvScale, lrBase, wdBase);
						GLADES_INT8_ADAM_BIG(gb.Wo, gb.gWo, gb.vWo_int8, gb.v2Wo_int8, gb.vWoScale, gb.v2WoScale, lrBase, wdBase);
						GLADES_INT8_ADAM_BIG(gb.W1, gb.gW1, gb.vW1_int8, gb.v2W1_int8, gb.vW1Scale, gb.v2W1Scale, lrBase, wdBase);
						GLADES_INT8_ADAM_BIG(gb.W2, gb.gW2, gb.vW2_int8, gb.v2W2_int8, gb.vW2Scale, gb.v2W2Scale, lrBase, wdBase);
					} else if (useBf16Grads_) {
						GLADES_BF16_ADAM_BIG_BF16GRAD(gb.Wq, gb.gWq_bf16, gb.vWq_bf16, gb.v2Wq_bf16, lrBase, wdBase);
						GLADES_BF16_ADAM_BIG_BF16GRAD(gb.Wk, gb.gWk_bf16, gb.vWk_bf16, gb.v2Wk_bf16, lrBase, wdBase);
						GLADES_BF16_ADAM_BIG_BF16GRAD(gb.Wv, gb.gWv_bf16, gb.vWv_bf16, gb.v2Wv_bf16, lrBase, wdBase);
						GLADES_BF16_ADAM_BIG_BF16GRAD(gb.Wo, gb.gWo_bf16, gb.vWo_bf16, gb.v2Wo_bf16, lrBase, wdBase);
						GLADES_BF16_ADAM_BIG_BF16GRAD(gb.W1, gb.gW1_bf16, gb.vW1_bf16, gb.v2W1_bf16, lrBase, wdBase);
						GLADES_BF16_ADAM_BIG_BF16GRAD(gb.W2, gb.gW2_bf16, gb.vW2_bf16, gb.v2W2_bf16, lrBase, wdBase);
					} else {
						GLADES_BF16_ADAM_BIG(gb.Wq, gb.gWq, gb.vWq_bf16, gb.v2Wq_bf16, lrBase, wdBase);
						GLADES_BF16_ADAM_BIG(gb.Wk, gb.gWk, gb.vWk_bf16, gb.v2Wk_bf16, lrBase, wdBase);
						GLADES_BF16_ADAM_BIG(gb.Wv, gb.gWv, gb.vWv_bf16, gb.v2Wv_bf16, lrBase, wdBase);
						GLADES_BF16_ADAM_BIG(gb.Wo, gb.gWo, gb.vWo_bf16, gb.v2Wo_bf16, lrBase, wdBase);
						GLADES_BF16_ADAM_BIG(gb.W1, gb.gW1, gb.vW1_bf16, gb.v2W1_bf16, lrBase, wdBase);
						GLADES_BF16_ADAM_BIG(gb.W2, gb.gW2, gb.vW2_bf16, gb.v2W2_bf16, lrBase, wdBase);
					}
				}
				// Gradients are zeroed at minibatch boundaries by the outer
				// training loop's clearGrads; no explicit zeroing needed here.
#undef GLADES_BF16_ADAM_BIG
#undef GLADES_INT8_ADAM_BIG
#undef GLADES_BF16_ADAM_BIG_BF16GRAD
#undef GLADES_INT8_ADAM_BIG_BF16GRAD
			}
				if (gpuUseGeode)
				{
					glades::ATLASConfig geodeAc = ac;
					geodeAc.complementRank = 0u;
					geodeAc.complementLrScale = 0.0f;
					const float predictiveScale =
					    std::max(0.0f, std::min(1.0f, ac.geodePredictiveScale));
					if (predictiveScale <= 0.0f)
					{
						geodeAc.muMin = 0.0f;
						geodeAc.muMax = 0.0f;
						geodeAc.muGrowthRate = 0.0f;
					}
					else
					{
						geodeAc.muMin *= predictiveScale;
						geodeAc.muMax *= predictiveScale;
						if (geodeAc.muMax < geodeAc.muMin)
							geodeAc.muMax = geodeAc.muMin;
						geodeAc.muGrowthRate *= predictiveScale;
					}
					const float geodeResidualScale = std::max(0.0f, ac.geodeGeometryScale);
					bool gpuGeodeError = false;

#define GLADES_GPU_GEODE_WEIGHT(state_, param_, grad_, rows_, cols_, lr_, tag_) do { \
	if (!gpuGeodeError && geodeResidualScale > 0.0f && (param_).size() > 0u) { \
		if (!gpu::atlas_gpu_residual_update((state_), (param_).data(), (grad_).data(), \
		                                    (rows_), (cols_), invBatch, \
		                                    (lr_) * geodeResidualScale, gradScale, \
		                                    geodeAc, rngEngine, getLogger(), (tag_))) \
			gpuGeodeError = true; \
	} \
} while (0)

					if (tokenLM)
					{
						const float lr0 = skeleton->getLearningRate(0u) * lrScheduleMultiplier * gpuExtraLRMult;
						GLADES_GPU_GEODE_WEIGHT(gpuTransformerWeights->atlasTokE,
						                        gpuTransformerWeights->tokE,
						                        gpuTransformerWeights->gTokE,
						                        vocabSize, dModel, lr0, "tr.tokE");
					}
					else
					{
						const float lr0 = skeleton->getLearningRate(0u) * lrScheduleMultiplier * gpuExtraLRMult;
						GLADES_GPU_GEODE_WEIGHT(gpuTransformerWeights->atlasWIn,
						                        gpuTransformerWeights->WIn,
						                        gpuTransformerWeights->gWIn,
						                        dModel, inputSize, lr0, "tr.WIn");
					}

					for (unsigned int bli = 0; bli < nLayers; ++bli)
					{
						const float lr_l =
						    skeleton->getLearningRate(bli + 1u) * lrScheduleMultiplier * gpuExtraLRMult;
						gpu::GpuTransformerWeights::Block& gb = gpuTransformerWeights->blocks[bli];
						GLADES_GPU_GEODE_WEIGHT(gb.atlasWq, gb.Wq, gb.gWq, dModel, dModel, lr_l, "tr.Wq");
						GLADES_GPU_GEODE_WEIGHT(gb.atlasWk, gb.Wk, gb.gWk, dModelKV, dModel, lr_l, "tr.Wk");
						GLADES_GPU_GEODE_WEIGHT(gb.atlasWv, gb.Wv, gb.gWv, dModelKV, dModel, lr_l, "tr.Wv");
						GLADES_GPU_GEODE_WEIGHT(gb.atlasWo, gb.Wo, gb.gWo, dModel, dModel, lr_l, "tr.Wo");
						GLADES_GPU_GEODE_WEIGHT(gb.atlasW1, gb.W1, gb.gW1, ff1Width, dModel, lr_l, "tr.W1");
						GLADES_GPU_GEODE_WEIGHT(gb.atlasW2, gb.W2, gb.gW2, dModel, dFF, lr_l, "tr.W2");
					}

					if (!tokenLM)
					{
						const float lrO = skeleton->getLearningRate(nLayers) * lrScheduleMultiplier * gpuExtraLRMult;
						GLADES_GPU_GEODE_WEIGHT(gpuTransformerWeights->atlasWOut,
						                        gpuTransformerWeights->WOut,
						                        gpuTransformerWeights->gWOut,
						                        outSize, dModel, lrO, "tr.WOut");
					}

#undef GLADES_GPU_GEODE_WEIGHT

					if (gpuGeodeError)
					{
						lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
						    "SGDHelper_TRANSFORMER: GPU GEODE residual update failed");
						storeRunningFlag(false);
					}
				}
				if (gpuFuseEcho)
				{
					bool gpuEchoError = false;

#define GLADES_GPU_ECHO_POST(enabled_, state_, rows_, cols_, tag_) do { \
	if (!gpuEchoError && (enabled_) && (rows_) > 0u && (cols_) > 0u) { \
		if (!gpu::echo_gpu_post_update((state_), ac, getLogger(), (tag_))) \
			gpuEchoError = true; \
	} \
} while (0)

					if (tokenLM)
						GLADES_GPU_ECHO_POST(echo_scope_uses_head_matrix(ac, vocabSize, dModel),
						                     gpuTransformerWeights->echoTokE, vocabSize, dModel, "tr.tokE");
					else
						GLADES_GPU_ECHO_POST(echo_scope_uses_input_matrix(ac, dModel, inputSize),
						                     gpuTransformerWeights->echoWIn, dModel, inputSize, "tr.WIn");

					for (unsigned int bli = 0; bli < nLayers; ++bli)
					{
						gpu::GpuTransformerWeights::Block& gb = gpuTransformerWeights->blocks[bli];
						GLADES_GPU_ECHO_POST(echo_scope_uses_decoder_matrix(ac, bli, nLayers, dModel, dModel), gb.echoWq, dModel, dModel, "tr.Wq");
						GLADES_GPU_ECHO_POST(echo_scope_uses_decoder_matrix(ac, bli, nLayers, dModelKV, dModel), gb.echoWk, dModelKV, dModel, "tr.Wk");
						GLADES_GPU_ECHO_POST(echo_scope_uses_decoder_matrix(ac, bli, nLayers, dModelKV, dModel), gb.echoWv, dModelKV, dModel, "tr.Wv");
						GLADES_GPU_ECHO_POST(echo_scope_uses_decoder_matrix(ac, bli, nLayers, dModel, dModel), gb.echoWo, dModel, dModel, "tr.Wo");
						GLADES_GPU_ECHO_POST(echo_scope_uses_decoder_matrix(ac, bli, nLayers, ff1Width, dModel), gb.echoW1, ff1Width, dModel, "tr.W1");
						GLADES_GPU_ECHO_POST(echo_scope_uses_decoder_matrix(ac, bli, nLayers, dModel, dFF), gb.echoW2, dModel, dFF, "tr.W2");
					}

					if (!tokenLM)
						GLADES_GPU_ECHO_POST(echo_scope_uses_head_matrix(ac, outSize, dModel),
						                     gpuTransformerWeights->echoWOut, outSize, dModel, "tr.WOut");

#undef GLADES_GPU_ECHO_POST

					if (gpuEchoError)
					{
						lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
						    "SGDHelper_TRANSFORMER: GPU ECHO post-update bookkeeping failed");
						storeRunningFlag(false);
					}
				}
				else if (gpuUseEcho)
				{
					bool gpuEchoError = false;

#define GLADES_GPU_ECHO_WEIGHT(enabled_, state_, param_, grad_, m1_, v2_, rows_, cols_, lr_, tag_) do { \
	if (!gpuEchoError && (enabled_) && (param_).size() > 0u) { \
		if (!gpu::echo_gpu_update((state_), (param_).data(), (grad_).data(), \
		                          (m1_).data(), (v2_).data(), \
		                          (rows_), (cols_), (lr_), invBatch * gradScale, \
		                          tensorTransformer.optimizerStep, \
		                          inv1mB1t, inv1mB2t, adamEps, \
		                          ac, getLogger(), (tag_))) \
			gpuEchoError = true; \
	} \
} while (0)

					if (tokenLM)
					{
						const float lr0 =
						    skeleton->getLearningRate(0u) * lrScheduleMultiplier * gpuExtraLRMult;
						GLADES_GPU_ECHO_WEIGHT(echo_scope_uses_head_matrix(ac, vocabSize, dModel),
						                       gpuTransformerWeights->echoTokE,
						                       gpuTransformerWeights->tokE,
						                       gpuTransformerWeights->gTokE,
						                       gpuTransformerWeights->vTokE,
						                       gpuTransformerWeights->v2TokE,
						                       vocabSize, dModel, lr0, "tr.tokE");
					}
					else
					{
						const float lr0 =
						    skeleton->getLearningRate(0u) * lrScheduleMultiplier * gpuExtraLRMult;
						GLADES_GPU_ECHO_WEIGHT(echo_scope_uses_input_matrix(ac, dModel, inputSize),
						                       gpuTransformerWeights->echoWIn,
						                       gpuTransformerWeights->WIn,
						                       gpuTransformerWeights->gWIn,
						                       gpuTransformerWeights->vWIn,
						                       gpuTransformerWeights->v2WIn,
						                       dModel, inputSize, lr0, "tr.WIn");
					}

					for (unsigned int bli = 0; bli < nLayers; ++bli)
					{
						const float lr_l =
						    skeleton->getLearningRate(bli + 1u) * lrScheduleMultiplier * gpuExtraLRMult;
						gpu::GpuTransformerWeights::Block& gb = gpuTransformerWeights->blocks[bli];
						GLADES_GPU_ECHO_WEIGHT(echo_scope_uses_decoder_matrix(ac, bli, nLayers, dModel, dModel), gb.echoWq, gb.Wq, gb.gWq, gb.vWq, gb.v2Wq, dModel, dModel, lr_l, "tr.Wq");
						GLADES_GPU_ECHO_WEIGHT(echo_scope_uses_decoder_matrix(ac, bli, nLayers, dModelKV, dModel), gb.echoWk, gb.Wk, gb.gWk, gb.vWk, gb.v2Wk, dModelKV, dModel, lr_l, "tr.Wk");
						GLADES_GPU_ECHO_WEIGHT(echo_scope_uses_decoder_matrix(ac, bli, nLayers, dModelKV, dModel), gb.echoWv, gb.Wv, gb.gWv, gb.vWv, gb.v2Wv, dModelKV, dModel, lr_l, "tr.Wv");
						GLADES_GPU_ECHO_WEIGHT(echo_scope_uses_decoder_matrix(ac, bli, nLayers, dModel, dModel), gb.echoWo, gb.Wo, gb.gWo, gb.vWo, gb.v2Wo, dModel, dModel, lr_l, "tr.Wo");
						GLADES_GPU_ECHO_WEIGHT(echo_scope_uses_decoder_matrix(ac, bli, nLayers, ff1Width, dModel), gb.echoW1, gb.W1, gb.gW1, gb.vW1, gb.v2W1, ff1Width, dModel, lr_l, "tr.W1");
						GLADES_GPU_ECHO_WEIGHT(echo_scope_uses_decoder_matrix(ac, bli, nLayers, dModel, dFF), gb.echoW2, gb.W2, gb.gW2, gb.vW2, gb.v2W2, dModel, dFF, lr_l, "tr.W2");
					}

					if (!tokenLM)
					{
						const float lrO =
						    skeleton->getLearningRate(nLayers) * lrScheduleMultiplier * gpuExtraLRMult;
						GLADES_GPU_ECHO_WEIGHT(echo_scope_uses_head_matrix(ac, outSize, dModel),
						                       gpuTransformerWeights->echoWOut,
						                       gpuTransformerWeights->WOut,
						                       gpuTransformerWeights->gWOut,
						                       gpuTransformerWeights->vWOut,
						                       gpuTransformerWeights->v2WOut,
						                       outSize, dModel, lrO, "tr.WOut");
					}

#undef GLADES_GPU_ECHO_WEIGHT

					if (gpuEchoError)
					{
						lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
						    "SGDHelper_TRANSFORMER: GPU ECHO residual update failed");
						storeRunningFlag(false);
					}
				}
				if (gpuUseBiMAP)
				{
					bool gpuBiMAPError = false;

#define GLADES_GPU_BIMAP_WEIGHT(state_, param_, grad_, m1_, v2_, rows_, cols_, lr_, tag_) do { \
	if (!gpuBiMAPError && (param_).size() > 0u) { \
		if (!gpu::bimap_gpu_update((state_), (param_).data(), (grad_).data(), \
		                           (m1_).data(), (v2_).data(), \
		                           (rows_), (cols_), (lr_), invBatch, gradScale, \
		                           inv1mB1t, inv1mB2t, adamEps, \
		                           ac, getLogger(), (tag_))) \
			gpuBiMAPError = true; \
	} \
} while (0)

					if (tokenLM)
					{
						const float lr0 =
						    skeleton->getLearningRate(0u) * lrScheduleMultiplier * gpuExtraLRMult;
						if (bimap_scope_uses_head(ac))
						{
							GLADES_GPU_BIMAP_WEIGHT(gpuTransformerWeights->bimapTokE,
							                        gpuTransformerWeights->tokE,
							                        gpuTransformerWeights->gTokE,
							                        gpuTransformerWeights->vTokE,
							                        gpuTransformerWeights->v2TokE,
							                        vocabSize, dModel, lr0, "tr.tokE");
						}
					}
					else
					{
						const float lr0 =
						    skeleton->getLearningRate(0u) * lrScheduleMultiplier * gpuExtraLRMult;
						if (bimap_scope_uses_input_block(ac))
						{
							GLADES_GPU_BIMAP_WEIGHT(gpuTransformerWeights->bimapWIn,
							                        gpuTransformerWeights->WIn,
							                        gpuTransformerWeights->gWIn,
							                        gpuTransformerWeights->vWIn,
							                        gpuTransformerWeights->v2WIn,
							                        dModel, inputSize, lr0, "tr.WIn");
						}
					}

					for (unsigned int bli = 0; bli < nLayers; ++bli)
					{
						if (!bimap_scope_uses_decoder_block(ac, bli, nLayers))
							continue;
						const float lr_l =
						    skeleton->getLearningRate(bli + 1u) * lrScheduleMultiplier * gpuExtraLRMult;
						gpu::GpuTransformerWeights::Block& gb = gpuTransformerWeights->blocks[bli];
						GLADES_GPU_BIMAP_WEIGHT(gb.bimapWq, gb.Wq, gb.gWq, gb.vWq, gb.v2Wq, dModel, dModel, lr_l, "tr.Wq");
						GLADES_GPU_BIMAP_WEIGHT(gb.bimapWk, gb.Wk, gb.gWk, gb.vWk, gb.v2Wk, dModelKV, dModel, lr_l, "tr.Wk");
						GLADES_GPU_BIMAP_WEIGHT(gb.bimapWv, gb.Wv, gb.gWv, gb.vWv, gb.v2Wv, dModelKV, dModel, lr_l, "tr.Wv");
						GLADES_GPU_BIMAP_WEIGHT(gb.bimapWo, gb.Wo, gb.gWo, gb.vWo, gb.v2Wo, dModel, dModel, lr_l, "tr.Wo");
						GLADES_GPU_BIMAP_WEIGHT(gb.bimapW1, gb.W1, gb.gW1, gb.vW1, gb.v2W1, ff1Width, dModel, lr_l, "tr.W1");
						GLADES_GPU_BIMAP_WEIGHT(gb.bimapW2, gb.W2, gb.gW2, gb.vW2, gb.v2W2, dModel, dFF, lr_l, "tr.W2");
					}

					if (!tokenLM)
					{
						const float lrO =
						    skeleton->getLearningRate(nLayers) * lrScheduleMultiplier * gpuExtraLRMult;
						if (bimap_scope_uses_head(ac))
						{
							GLADES_GPU_BIMAP_WEIGHT(gpuTransformerWeights->bimapWOut,
							                        gpuTransformerWeights->WOut,
							                        gpuTransformerWeights->gWOut,
							                        gpuTransformerWeights->vWOut,
							                        gpuTransformerWeights->v2WOut,
							                        outSize, dModel, lrO, "tr.WOut");
						}
					}

#undef GLADES_GPU_BIMAP_WEIGHT

					if (gpuBiMAPError)
					{
						lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
						    "SGDHelper_TRANSFORMER: GPU BiMAP-lite residual update failed");
						storeRunningFlag(false);
					}
				}
				if (gpuUsePact)
				{
					bool gpuPactError = false;

#define GLADES_GPU_PACT_WEIGHT(state_, param_, grad_, m1_, v2_, rows_, cols_, lr_, wd1_, tag_) do { \
	if (!gpuPactError && (param_).size() > 0u) { \
		if (!gpu::pact_gpu_update_lite((state_), (param_).data(), (grad_).data(), \
		                               (m1_).data(), (v2_).data(), \
		                               (rows_), (cols_), (lr_), invBatch, gradScale, \
		                               inv1mB1t, inv1mB2t, (wd1_), adamEps, \
		                               ac, getLogger(), (tag_))) \
			gpuPactError = true; \
	} \
} while (0)

					if (tokenLM)
					{
						const float lr0 =
						    skeleton->getLearningRate(0u) * lrScheduleMultiplier * gpuExtraLRMult;
						const float wd1_0 = skeleton->getWeightDecay1(0u);
						GLADES_GPU_PACT_WEIGHT(gpuTransformerWeights->pactTokE,
						                       gpuTransformerWeights->tokE,
						                       gpuTransformerWeights->gTokE,
						                       gpuTransformerWeights->vTokE,
						                       gpuTransformerWeights->v2TokE,
						                       vocabSize, dModel, lr0, wd1_0, "tr.tokE");
					}
					else
					{
						const float lr0 =
						    skeleton->getLearningRate(0u) * lrScheduleMultiplier * gpuExtraLRMult;
						const float wd1_0 = skeleton->getWeightDecay1(0u);
						GLADES_GPU_PACT_WEIGHT(gpuTransformerWeights->pactWIn,
						                       gpuTransformerWeights->WIn,
						                       gpuTransformerWeights->gWIn,
						                       gpuTransformerWeights->vWIn,
						                       gpuTransformerWeights->v2WIn,
						                       dModel, inputSize, lr0, wd1_0, "tr.WIn");
					}

					for (unsigned int bli = 0; bli < nLayers; ++bli)
					{
						const float lr_l =
						    skeleton->getLearningRate(bli + 1u) * lrScheduleMultiplier * gpuExtraLRMult;
						const float wd1_l = skeleton->getWeightDecay1(bli + 1u);
						gpu::GpuTransformerWeights::Block& gb = gpuTransformerWeights->blocks[bli];
						GLADES_GPU_PACT_WEIGHT(gb.pactWq, gb.Wq, gb.gWq, gb.vWq, gb.v2Wq, dModel, dModel, lr_l, wd1_l, "tr.Wq");
						GLADES_GPU_PACT_WEIGHT(gb.pactWk, gb.Wk, gb.gWk, gb.vWk, gb.v2Wk, dModelKV, dModel, lr_l, wd1_l, "tr.Wk");
						GLADES_GPU_PACT_WEIGHT(gb.pactWv, gb.Wv, gb.gWv, gb.vWv, gb.v2Wv, dModelKV, dModel, lr_l, wd1_l, "tr.Wv");
						GLADES_GPU_PACT_WEIGHT(gb.pactWo, gb.Wo, gb.gWo, gb.vWo, gb.v2Wo, dModel, dModel, lr_l, wd1_l, "tr.Wo");
						GLADES_GPU_PACT_WEIGHT(gb.pactW1, gb.W1, gb.gW1, gb.vW1, gb.v2W1, ff1Width, dModel, lr_l, wd1_l, "tr.W1");
						GLADES_GPU_PACT_WEIGHT(gb.pactW2, gb.W2, gb.gW2, gb.vW2, gb.v2W2, dModel, dFF, lr_l, wd1_l, "tr.W2");
					}

					if (!tokenLM)
					{
						const float lrO =
						    skeleton->getLearningRate(nLayers) * lrScheduleMultiplier * gpuExtraLRMult;
						const float wd1_o = skeleton->getWeightDecay1(nLayers);
						GLADES_GPU_PACT_WEIGHT(gpuTransformerWeights->pactWOut,
						                       gpuTransformerWeights->WOut,
						                       gpuTransformerWeights->gWOut,
						                       gpuTransformerWeights->vWOut,
						                       gpuTransformerWeights->v2WOut,
						                       outSize, dModel, lrO, wd1_o, "tr.WOut");
					}

#undef GLADES_GPU_PACT_WEIGHT

					if (gpuPactError)
					{
						lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
						    "SGDHelper_TRANSFORMER: GPU PACT-lite residual update failed");
						storeRunningFlag(false);
					}
				}
				if (gpuUseRacer)
				{
					bool gpuRacerError = false;

#define GLADES_GPU_RACER_WEIGHT(state_, param_, grad_, m1_, v2_, rows_, cols_, lr_, wd1_, tag_) do { \
	if (!gpuRacerError && (param_).size() > 0u) { \
		if (!gpu::racer_gpu_update_lite((state_), (param_).data(), (grad_).data(), \
		                                (m1_).data(), (v2_).data(), \
		                                (rows_), (cols_), (lr_), invBatch, gradScale, \
		                                inv1mB1t, inv1mB2t, (wd1_), adamEps, \
		                                ac, getLogger(), (tag_))) \
			gpuRacerError = true; \
	} \
} while (0)

					if (tokenLM)
					{
						const float lr0 =
						    skeleton->getLearningRate(0u) * lrScheduleMultiplier * gpuExtraLRMult;
						const float wd1_0 = skeleton->getWeightDecay1(0u);
						GLADES_GPU_RACER_WEIGHT(gpuTransformerWeights->racerTokE,
						                        gpuTransformerWeights->tokE,
						                        gpuTransformerWeights->gTokE,
						                        gpuTransformerWeights->vTokE,
						                        gpuTransformerWeights->v2TokE,
						                        vocabSize, dModel, lr0, wd1_0, "tr.tokE");
					}
					else
					{
						const float lr0 =
						    skeleton->getLearningRate(0u) * lrScheduleMultiplier * gpuExtraLRMult;
						const float wd1_0 = skeleton->getWeightDecay1(0u);
						GLADES_GPU_RACER_WEIGHT(gpuTransformerWeights->racerWIn,
						                        gpuTransformerWeights->WIn,
						                        gpuTransformerWeights->gWIn,
						                        gpuTransformerWeights->vWIn,
						                        gpuTransformerWeights->v2WIn,
						                        dModel, inputSize, lr0, wd1_0, "tr.WIn");
					}

					for (unsigned int bli = 0; bli < nLayers; ++bli)
					{
						const float lr_l =
						    skeleton->getLearningRate(bli + 1u) * lrScheduleMultiplier * gpuExtraLRMult;
						const float wd1_l = skeleton->getWeightDecay1(bli + 1u);
						gpu::GpuTransformerWeights::Block& gb = gpuTransformerWeights->blocks[bli];
						GLADES_GPU_RACER_WEIGHT(gb.racerWq, gb.Wq, gb.gWq, gb.vWq, gb.v2Wq, dModel, dModel, lr_l, wd1_l, "tr.Wq");
						GLADES_GPU_RACER_WEIGHT(gb.racerWk, gb.Wk, gb.gWk, gb.vWk, gb.v2Wk, dModelKV, dModel, lr_l, wd1_l, "tr.Wk");
						GLADES_GPU_RACER_WEIGHT(gb.racerWv, gb.Wv, gb.gWv, gb.vWv, gb.v2Wv, dModelKV, dModel, lr_l, wd1_l, "tr.Wv");
						GLADES_GPU_RACER_WEIGHT(gb.racerWo, gb.Wo, gb.gWo, gb.vWo, gb.v2Wo, dModel, dModel, lr_l, wd1_l, "tr.Wo");
						GLADES_GPU_RACER_WEIGHT(gb.racerW1, gb.W1, gb.gW1, gb.vW1, gb.v2W1, ff1Width, dModel, lr_l, wd1_l, "tr.W1");
						GLADES_GPU_RACER_WEIGHT(gb.racerW2, gb.W2, gb.gW2, gb.vW2, gb.v2W2, dModel, dFF, lr_l, wd1_l, "tr.W2");
					}

					if (!tokenLM)
					{
						const float lrO =
						    skeleton->getLearningRate(nLayers) * lrScheduleMultiplier * gpuExtraLRMult;
						const float wd1_o = skeleton->getWeightDecay1(nLayers);
						GLADES_GPU_RACER_WEIGHT(gpuTransformerWeights->racerWOut,
						                        gpuTransformerWeights->WOut,
						                        gpuTransformerWeights->gWOut,
						                        gpuTransformerWeights->vWOut,
						                        gpuTransformerWeights->v2WOut,
						                        outSize, dModel, lrO, wd1_o, "tr.WOut");
					}

#undef GLADES_GPU_RACER_WEIGHT

					if (gpuRacerError)
					{
						lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
						    "SGDHelper_TRANSFORMER: GPU RACER-lite residual update failed");
						storeRunningFlag(false);
					}
				}
				if (gpuUseMatra)
				{
					bool gpuMatraError = false;
					std::vector<MatraBatchGroup> matraSmallBatchGroups;
					const float matraLrScale = lrScheduleMultiplier * gpuExtraLRMult;

					if (tokenLM)
					{
						const float lr0Base = skeleton->getLearningRate(0u);
						if (!gpuMatraError
						    && !queue_or_run_matra_gpu(matraSmallBatchGroups,
						                               gpuTransformerWeights,
						                               gpuTransformerWeights->matraTokE,
						                               gpuTransformerWeights->tokE,
						                               gpuTransformerWeights->gTokE,
						                               gpuTransformerWeights->vTokE,
						                               gpuTransformerWeights->v2TokE,
						                               vocabSize, dModel, lr0Base, matraLrScale,
						                               invBatch, gradScale,
						                               inv1mB1t, inv1mB2t, adamEps,
						                               ac, getLogger(), "tr.tokE"))
							gpuMatraError = true;
					}
					else
					{
						const float lr0Base = skeleton->getLearningRate(0u);
						if (!gpuMatraError
						    && !queue_or_run_matra_gpu(matraSmallBatchGroups,
						                               gpuTransformerWeights,
						                               gpuTransformerWeights->matraWIn,
						                               gpuTransformerWeights->WIn,
						                               gpuTransformerWeights->gWIn,
						                               gpuTransformerWeights->vWIn,
						                               gpuTransformerWeights->v2WIn,
						                               dModel, inputSize, lr0Base, matraLrScale,
						                               invBatch, gradScale,
						                               inv1mB1t, inv1mB2t, adamEps,
						                               ac, getLogger(), "tr.WIn"))
							gpuMatraError = true;
					}

					for (unsigned int bli = 0; bli < nLayers; ++bli)
					{
						const float lrBase = skeleton->getLearningRate(bli + 1u);
						gpu::GpuTransformerWeights::Block& gb = gpuTransformerWeights->blocks[bli];
						if (!gpuMatraError
						    && !queue_or_run_matra_gpu(matraSmallBatchGroups, gpuTransformerWeights,
						                               gb.matraWq, gb.Wq, gb.gWq, gb.vWq, gb.v2Wq,
						                               dModel, dModel, lrBase, matraLrScale,
						                               invBatch, gradScale,
						                               inv1mB1t, inv1mB2t, adamEps,
						                               ac, getLogger(), "tr.Wq"))
							gpuMatraError = true;
						if (!gpuMatraError
						    && !queue_or_run_matra_gpu(matraSmallBatchGroups, gpuTransformerWeights,
						                               gb.matraWk, gb.Wk, gb.gWk, gb.vWk, gb.v2Wk,
						                               dModelKV, dModel, lrBase, matraLrScale,
						                               invBatch, gradScale,
						                               inv1mB1t, inv1mB2t, adamEps,
						                               ac, getLogger(), "tr.Wk"))
							gpuMatraError = true;
						if (!gpuMatraError
						    && !queue_or_run_matra_gpu(matraSmallBatchGroups, gpuTransformerWeights,
						                               gb.matraWv, gb.Wv, gb.gWv, gb.vWv, gb.v2Wv,
						                               dModelKV, dModel, lrBase, matraLrScale,
						                               invBatch, gradScale,
						                               inv1mB1t, inv1mB2t, adamEps,
						                               ac, getLogger(), "tr.Wv"))
							gpuMatraError = true;
						if (!gpuMatraError
						    && !queue_or_run_matra_gpu(matraSmallBatchGroups, gpuTransformerWeights,
						                               gb.matraWo, gb.Wo, gb.gWo, gb.vWo, gb.v2Wo,
						                               dModel, dModel, lrBase, matraLrScale,
						                               invBatch, gradScale,
						                               inv1mB1t, inv1mB2t, adamEps,
						                               ac, getLogger(), "tr.Wo"))
							gpuMatraError = true;
						if (!gpuMatraError
						    && !queue_or_run_matra_gpu(matraSmallBatchGroups, gpuTransformerWeights,
						                               gb.matraW1, gb.W1, gb.gW1, gb.vW1, gb.v2W1,
						                               ff1Width, dModel, lrBase, matraLrScale,
						                               invBatch, gradScale,
						                               inv1mB1t, inv1mB2t, adamEps,
						                               ac, getLogger(), "tr.W1"))
							gpuMatraError = true;
						if (!gpuMatraError
						    && !queue_or_run_matra_gpu(matraSmallBatchGroups, gpuTransformerWeights,
						                               gb.matraW2, gb.W2, gb.gW2, gb.vW2, gb.v2W2,
						                               dModel, dFF, lrBase, matraLrScale,
						                               invBatch, gradScale,
						                               inv1mB1t, inv1mB2t, adamEps,
						                               ac, getLogger(), "tr.W2"))
							gpuMatraError = true;
					}

					if (!tokenLM)
					{
						const float lrOBase = skeleton->getLearningRate(nLayers);
						if (!gpuMatraError
						    && !queue_or_run_matra_gpu(matraSmallBatchGroups,
						                               gpuTransformerWeights,
						                               gpuTransformerWeights->matraWOut,
						                               gpuTransformerWeights->WOut,
						                               gpuTransformerWeights->gWOut,
						                               gpuTransformerWeights->vWOut,
						                               gpuTransformerWeights->v2WOut,
						                               outSize, dModel, lrOBase, matraLrScale,
						                               invBatch, gradScale,
						                               inv1mB1t, inv1mB2t, adamEps,
						                               ac, getLogger(), "tr.WOut"))
							gpuMatraError = true;
					}

					if (!gpuMatraError && !matraSmallBatchGroups.empty())
					{
						size_t totalMatraBatchItems = 0u;
						for (size_t gi = 0; gi < matraSmallBatchGroups.size(); ++gi)
							totalMatraBatchItems += matraSmallBatchGroups[gi].items.size();

						std::vector<gpu::GpuMatraBatchItem> matraBatchItems;
						std::vector<int> matraBatchOffsets;
						std::vector<int> matraBatchCounts;
						matraBatchItems.reserve(totalMatraBatchItems);
						matraBatchOffsets.reserve(matraSmallBatchGroups.size());
						matraBatchCounts.reserve(matraSmallBatchGroups.size());

						for (size_t gi = 0; gi < matraSmallBatchGroups.size(); ++gi)
						{
							MatraBatchGroup& group = matraSmallBatchGroups[gi];
							if (group.items.empty())
								continue;
							matraBatchOffsets.push_back(static_cast<int>(matraBatchItems.size()));
							matraBatchCounts.push_back(static_cast<int>(group.items.size()));
							matraBatchItems.insert(matraBatchItems.end(),
							                       group.items.begin(),
							                       group.items.end());
						}

						if (!matraBatchItems.empty()
						    && !gpu::matra_gpu_update_small_batches(
						        matraBatchItems.data(),
						        static_cast<int>(matraBatchItems.size()),
						        matraBatchOffsets.data(),
						        matraBatchCounts.data(),
						        static_cast<int>(matraBatchOffsets.size()),
						        gpuTransformerWeights->d_matraBatchItems,
						        gpuTransformerWeights->d_matraStatsBatch,
						        gpuTransformerWeights->d_matraCoreBatchPtrs,
						        gpuTransformerWeights->d_matraStepBatchPtrs,
						        gpuTransformerWeights->d_matraInfoBatch,
						        gpuTransformerWeights->matraCoreBatchCapacity,
						        &gpuTransformerWeights->matraBatchDescriptorsUploaded,
						        &gpuTransformerWeights->matraBatchDescriptorCount,
						        &gpuTransformerWeights->matraBatchDescriptorHash,
						        matraLrScale,
						        invBatch, gradScale,
						        inv1mB1t, inv1mB2t,
						        adamEps,
						        ac,
						        getLogger()))
						{
							gpuMatraError = true;
						}
					}

					if (gpuMatraError)
					{
						lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
						                            "SGDHelper_TRANSFORMER: GPU MATRA residual update failed");
						storeRunningFlag(false);
					}
				}
				if (gpuUseArgos)
				{
					bool gpuArgosError = false;

					if (tokenLM)
					{
						const float lr0 =
						    skeleton->getLearningRate(0u) * lrScheduleMultiplier * gpuExtraLRMult;
						if (argos_scope_uses_head(ac)
						    && !gpuArgosError
						    && !run_argos_gpu(gpuTransformerWeights->argosTokE,
						                      gpuTransformerWeights->tokE,
						                      gpuTransformerWeights->gTokE,
						                      gpuTransformerWeights->vTokE,
						                      gpuTransformerWeights->v2TokE,
						                      vocabSize, dModel, lr0,
						                      invBatch, gradScale,
						                      inv1mB1t, inv1mB2t, adamEps,
						                      ac,
						                      argos_role_flags_for_head(),
						                      getLogger(), "tr.tokE"))
							gpuArgosError = true;
					}
					else
					{
						const float lr0 =
						    skeleton->getLearningRate(0u) * lrScheduleMultiplier * gpuExtraLRMult;
						if (argos_scope_uses_input_block(ac)
						    && !gpuArgosError
						    && !run_argos_gpu(gpuTransformerWeights->argosWIn,
						                      gpuTransformerWeights->WIn,
						                      gpuTransformerWeights->gWIn,
						                      gpuTransformerWeights->vWIn,
						                      gpuTransformerWeights->v2WIn,
						                      dModel, inputSize, lr0,
						                      invBatch, gradScale,
						                      inv1mB1t, inv1mB2t, adamEps,
						                      ac,
						                      glades::atlas::ARGOS_ROLE_NONE,
						                      getLogger(), "tr.WIn"))
							gpuArgosError = true;
					}

					for (unsigned int bli = 0; bli < nLayers; ++bli)
					{
						const float lr_l =
						    skeleton->getLearningRate(bli + 1u) * lrScheduleMultiplier * gpuExtraLRMult;
						const unsigned int argosRoleFlags = argos_role_flags_for_block(bli, nLayers);
						if (!argos_scope_uses_decoder_block(ac, bli, nLayers))
							continue;
						gpu::GpuTransformerWeights::Block& gb = gpuTransformerWeights->blocks[bli];
						if (!gpuArgosError
						    && !run_argos_gpu(gb.argosWq, gb.Wq, gb.gWq, gb.vWq, gb.v2Wq,
						                      dModel, dModel, lr_l,
						                      invBatch, gradScale,
						                      inv1mB1t, inv1mB2t, adamEps,
						                      ac, argosRoleFlags, getLogger(), "tr.Wq"))
							gpuArgosError = true;
						if (!gpuArgosError
						    && !run_argos_gpu(gb.argosWk, gb.Wk, gb.gWk, gb.vWk, gb.v2Wk,
						                      dModelKV, dModel, lr_l,
						                      invBatch, gradScale,
						                      inv1mB1t, inv1mB2t, adamEps,
						                      ac, argosRoleFlags, getLogger(), "tr.Wk"))
							gpuArgosError = true;
						if (!gpuArgosError
						    && !run_argos_gpu(gb.argosWv, gb.Wv, gb.gWv, gb.vWv, gb.v2Wv,
						                      dModelKV, dModel, lr_l,
						                      invBatch, gradScale,
						                      inv1mB1t, inv1mB2t, adamEps,
						                      ac, argosRoleFlags, getLogger(), "tr.Wv"))
							gpuArgosError = true;
						if (!gpuArgosError
						    && !run_argos_gpu(gb.argosWo, gb.Wo, gb.gWo, gb.vWo, gb.v2Wo,
						                      dModel, dModel, lr_l,
						                      invBatch, gradScale,
						                      inv1mB1t, inv1mB2t, adamEps,
						                      ac, argosRoleFlags, getLogger(), "tr.Wo"))
							gpuArgosError = true;
						if (!gpuArgosError
						    && !run_argos_gpu(gb.argosW1, gb.W1, gb.gW1, gb.vW1, gb.v2W1,
						                      ff1Width, dModel, lr_l,
						                      invBatch, gradScale,
						                      inv1mB1t, inv1mB2t, adamEps,
						                      ac, argosRoleFlags, getLogger(), "tr.W1"))
							gpuArgosError = true;
						if (!gpuArgosError
						    && !run_argos_gpu(gb.argosW2, gb.W2, gb.gW2, gb.vW2, gb.v2W2,
						                      dModel, dFF, lr_l,
						                      invBatch, gradScale,
						                      inv1mB1t, inv1mB2t, adamEps,
						                      ac, argosRoleFlags, getLogger(), "tr.W2"))
							gpuArgosError = true;
					}

					if (!tokenLM)
					{
						const float lrO =
						    skeleton->getLearningRate(nLayers) * lrScheduleMultiplier * gpuExtraLRMult;
						if (argos_scope_uses_head(ac)
						    && !gpuArgosError
						    && !run_argos_gpu(gpuTransformerWeights->argosWOut,
						                      gpuTransformerWeights->WOut,
						                      gpuTransformerWeights->gWOut,
						                      gpuTransformerWeights->vWOut,
						                      gpuTransformerWeights->v2WOut,
						                      outSize, dModel, lrO,
						                      invBatch, gradScale,
						                      inv1mB1t, inv1mB2t, adamEps,
						                      ac,
						                      argos_role_flags_for_head(),
						                      getLogger(), "tr.WOut"))
							gpuArgosError = true;
					}

					if (gpuArgosError)
					{
						lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
						                            "SGDHelper_TRANSFORMER: GPU ARGOS residual update failed");
						storeRunningFlag(false);
					}
				}
				if (gpuUseMuon)
				{
					bool gpuMuonError = false;
					bool gpuMuonZeroError = false;
					std::vector<MuonBatchGroup> muonSmallBatchGroups;

					if (tokenLM)
					{
						const float lr0 =
						    skeleton->getLearningRate(0u) * lrScheduleMultiplier * gpuExtraLRMult;
						if (!gpuMuonError
						    && !queue_or_run_muon_gpu(muonSmallBatchGroups,
						                              gpuTransformerWeights,
						                              gpuTransformerWeights->muonTokE,
						                              gpuTransformerWeights->tokE,
						                              gpuTransformerWeights->gTokE,
						                              gpuTransformerWeights->vTokE,
						                              gpuTransformerWeights->v2TokE,
						                              vocabSize, dModel, lr0,
						                              inv1mB1t, inv1mB2t, adamEps,
						                              ac, getLogger(), "tr.tokE"))
							gpuMuonError = true;
					}
					else
					{
						const float lr0 =
						    skeleton->getLearningRate(0u) * lrScheduleMultiplier * gpuExtraLRMult;
						if (!gpuMuonError
						    && !queue_or_run_muon_gpu(muonSmallBatchGroups,
						                              gpuTransformerWeights,
						                              gpuTransformerWeights->muonWIn,
						                              gpuTransformerWeights->WIn,
						                              gpuTransformerWeights->gWIn,
						                              gpuTransformerWeights->vWIn,
						                              gpuTransformerWeights->v2WIn,
						                              dModel, inputSize, lr0,
						                              inv1mB1t, inv1mB2t, adamEps,
						                              ac, getLogger(), "tr.WIn"))
							gpuMuonError = true;
					}

					for (unsigned int bli = 0; bli < nLayers; ++bli)
					{
						const float lr_l =
						    skeleton->getLearningRate(bli + 1u) * lrScheduleMultiplier * gpuExtraLRMult;
						gpu::GpuTransformerWeights::Block& gb = gpuTransformerWeights->blocks[bli];
						if (!gpuMuonError
						    && !queue_or_run_muon_gpu(muonSmallBatchGroups, gpuTransformerWeights,
						                              gb.muonWq, gb.Wq, gb.gWq, gb.vWq, gb.v2Wq,
						                              dModel, dModel, lr_l,
						                              inv1mB1t, inv1mB2t, adamEps,
						                              ac, getLogger(), "tr.Wq"))
							gpuMuonError = true;
						if (!gpuMuonError
						    && !queue_or_run_muon_gpu(muonSmallBatchGroups, gpuTransformerWeights,
						                              gb.muonWk, gb.Wk, gb.gWk, gb.vWk, gb.v2Wk,
						                              dModelKV, dModel, lr_l,
						                              inv1mB1t, inv1mB2t, adamEps,
						                              ac, getLogger(), "tr.Wk"))
							gpuMuonError = true;
						if (!gpuMuonError
						    && !queue_or_run_muon_gpu(muonSmallBatchGroups, gpuTransformerWeights,
						                              gb.muonWv, gb.Wv, gb.gWv, gb.vWv, gb.v2Wv,
						                              dModelKV, dModel, lr_l,
						                              inv1mB1t, inv1mB2t, adamEps,
						                              ac, getLogger(), "tr.Wv"))
							gpuMuonError = true;
						if (!gpuMuonError
						    && !queue_or_run_muon_gpu(muonSmallBatchGroups, gpuTransformerWeights,
						                              gb.muonWo, gb.Wo, gb.gWo, gb.vWo, gb.v2Wo,
						                              dModel, dModel, lr_l,
						                              inv1mB1t, inv1mB2t, adamEps,
						                              ac, getLogger(), "tr.Wo"))
							gpuMuonError = true;
						if (!gpuMuonError
						    && !queue_or_run_muon_gpu(muonSmallBatchGroups, gpuTransformerWeights,
						                              gb.muonW1, gb.W1, gb.gW1, gb.vW1, gb.v2W1,
						                              ff1Width, dModel, lr_l,
						                              inv1mB1t, inv1mB2t, adamEps,
						                              ac, getLogger(), "tr.W1"))
							gpuMuonError = true;
						if (!gpuMuonError
						    && !queue_or_run_muon_gpu(muonSmallBatchGroups, gpuTransformerWeights,
						                              gb.muonW2, gb.W2, gb.gW2, gb.vW2, gb.v2W2,
						                              dModel, dFF, lr_l,
						                              inv1mB1t, inv1mB2t, adamEps,
						                              ac, getLogger(), "tr.W2"))
							gpuMuonError = true;
					}

					if (!tokenLM)
					{
						const float lrO =
						    skeleton->getLearningRate(nLayers) * lrScheduleMultiplier * gpuExtraLRMult;
						if (!gpuMuonError
						    && !queue_or_run_muon_gpu(muonSmallBatchGroups,
						                              gpuTransformerWeights,
						                              gpuTransformerWeights->muonWOut,
						                              gpuTransformerWeights->WOut,
						                              gpuTransformerWeights->gWOut,
						                              gpuTransformerWeights->vWOut,
						                              gpuTransformerWeights->v2WOut,
						                              outSize, dModel, lrO,
						                              inv1mB1t, inv1mB2t, adamEps,
						                              ac, getLogger(), "tr.WOut"))
							gpuMuonError = true;
					}

					if (!gpuMuonError && !muonSmallBatchGroups.empty())
					{
						size_t totalMuonBatchItems = 0u;
						for (size_t gi = 0; gi < muonSmallBatchGroups.size(); ++gi)
							totalMuonBatchItems += muonSmallBatchGroups[gi].items.size();

						std::vector<gpu::GpuMuonBatchItem> muonBatchItems;
						std::vector<int> muonBatchOffsets;
						std::vector<int> muonBatchCounts;
						muonBatchItems.reserve(totalMuonBatchItems);
						muonBatchOffsets.reserve(muonSmallBatchGroups.size());
						muonBatchCounts.reserve(muonSmallBatchGroups.size());

						for (size_t gi = 0; gi < muonSmallBatchGroups.size(); ++gi)
						{
							MuonBatchGroup& group = muonSmallBatchGroups[gi];
							if (group.items.empty())
								continue;
							muonBatchOffsets.push_back(static_cast<int>(muonBatchItems.size()));
							muonBatchCounts.push_back(static_cast<int>(group.items.size()));
							muonBatchItems.insert(muonBatchItems.end(),
							                      group.items.begin(),
							                      group.items.end());
						}

						if (!muonBatchItems.empty()
						    && !gpu::muon_gpu_update_lite_small_batches(
						        muonBatchItems.data(),
						        static_cast<int>(muonBatchItems.size()),
						        muonBatchOffsets.data(),
						        muonBatchCounts.data(),
						        static_cast<int>(muonBatchOffsets.size()),
						        gpuTransformerWeights->d_muonBatchItems,
						        gpuTransformerWeights->d_muonCoreBatchPtrs,
						        gpuTransformerWeights->d_muonStepBatchPtrs,
						        gpuTransformerWeights->d_muonInfoBatch,
						        gpuTransformerWeights->muonCoreBatchCapacity,
						        inv1mB1t, inv1mB2t,
						        adamEps,
						        ac,
						        getLogger()))
						{
							gpuMuonError = true;
						}
					}

						// Clear all gradient buffers in one batch after the MUON
						// residual pass instead of routing permanently ineligible
						// matrices through MUON just to zero their gradients.
						if (!gpuMuonError
						    && !gpu::zero_buffers_batch(gpuTransformerWeights->d_adamGrads,
						                                gpuTransformerWeights->d_adamSizes,
						                                gpuTransformerWeights->adamGroupCount))
						{
							gpuMuonZeroError = true;
						}

						if (gpuMuonError)
						{
							lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
							    "SGDHelper_TRANSFORMER: GPU MUON-lite residual update failed");
							storeRunningFlag(false);
						}
						else if (gpuMuonZeroError)
						{
							lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
							    "SGDHelper_TRANSFORMER: GPU MUON gradient clear failed");
							storeRunningFlag(false);
						}
					}
				} // end Adam/GEODE branch

			if (gpuPerf)
				gpu::perfRecordSync(&gpuPerf->counters, 1u);
			gpu::synchronizeComputeStream();

			// Refresh the device-side BF16 weight mirrors from the just-updated
			// FP32 masters so the next forward pass reads weights consistent
			// with the optimizer's update. No-op when mixed precision is off.
			if (cfg.mpEnable && gpuTransformerWeights)
			{
				gpuTransformerWeights->lowpDType =
				    glades::transformer_kernels::LOWP_BF16;
				if (!gpuTransformerWeights->ensureLowpMirrors())
				{
					lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
					    "transformerGpuTrainEpoch: failed to refresh BF16 weight mirrors");
					storeRunningFlag(false);
				}
			}

			seqInBatch = 0u;
			timeStepsInBatch = 0u;
		}
	}

	gpu::destroyEvent(gpuTransferReadyEvent);
	gpu::destroyEvent(gpuComputeReadyEvent);

		// After GPU training loop: sync updated weights back to CPU-owned tensor state.
		if (!syncTransformerGpuTrainingWeightsToCpu())
		{
			lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
			                            "SGDHelper_TRANSFORMER: failed to download transformer GPU weights");
			storeRunningFlag(false);
			return;
		}
		if (gpuPerf)
			gpu::perfRecordSync(&gpuPerf->counters, 1u);

	// Finalize epoch-level loss before returning.
	// tokenLmNllSum / tokenLmTokenCount are locals; copy to the member
	// that the Trainer reads (overallTotalError).
	if (tokenLM)
	{
		if (tokenLmTokenCount > 0ULL)
			overallTotalError = static_cast<float>(tokenLmNllSum / static_cast<double>(tokenLmTokenCount));
		else
			overallTotalError = 0.0f;
	}
	if (gpuPerf && logger && transformerMetricsCfg.logGpuTrainSummary)
	{
		std::ostringstream oss;
		oss << "event=transformer_gpu_train_perf";
		append_logfmt_kv(oss, "epoch", epochIdx);
		append_logfmt_kv(oss, "kernel_launches", gpuPerf->counters.kernelLaunches);
		append_logfmt_kv(oss, "sync_points", gpuPerf->counters.syncPoints);
		append_logfmt_kv(oss, "bytes_h2d", gpuPerf->counters.bytesH2D);
		append_logfmt_kv(oss, "bytes_d2h", gpuPerf->counters.bytesD2H);
		append_logfmt_kv(oss, "bytes_d2d", gpuPerf->counters.bytesD2D);
		append_logfmt_kv(oss, "ms_total", gpuPerf->msTotal);
		append_logfmt_kv(oss, "ms_loss", gpuPerf->msLoss);
		append_logfmt_kv(oss, "ms_backward", gpuPerf->msBackward);
		append_logfmt_kv(oss, "ms_optimizer", gpuPerf->msOptimizer);
		logger->info("NNetwork", shmea::GString(oss.str().c_str()));
	}

}
#endif // GLADES_HAVE_CUDA
