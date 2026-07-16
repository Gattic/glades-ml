// CHIRON ORBIT optimizer CPU references and memory accounting.
#include "chiron_orbit.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace glades {
namespace chiron {

static uint64_t orbit_int8_blocks(uint64_t n)
{
	return (n + 255u) / 256u;
}

OrbitMemoryEstimate chiron_orbit_memory_estimate(int V, int m, int dModel,
                                                  int L, size_t smallParams)
{
	OrbitMemoryEstimate x;
	if (V <= 0 || m <= 0 || dModel <= 0 || L <= 0) return x;
	const uint64_t e = (uint64_t)V * (uint64_t)m;
	const uint64_t attnOne = (uint64_t)m * (uint64_t)dModel;
	const uint64_t matrix = e + (uint64_t)L * 4u * attnOne;
	x.parameterCount = matrix + (uint64_t)smallParams;
	// Baseline: signed-int8 m + uint8 v, each with one FP32 scale / block.
	x.baselineBytes = 2u * matrix + 8u * orbit_int8_blocks(matrix)
	                + 2u * (uint64_t)smallParams + 8u * orbit_int8_blocks((uint64_t)smallParams);
	// ORBIT matrix momentum: signed int8 + one FP32 scale / block.
	x.momentumBytes = matrix + 4u * orbit_int8_blocks(matrix);
	// Shared Wo readout rows, per-Wo activation cols, QKV row+col, E dual rows/cols,
	// one scale scalar per matrix, plus a small diagnostics/cap allowance.
	const uint64_t factorFloats = (uint64_t)m
	    + (uint64_t)L * (uint64_t)dModel
	    + (uint64_t)L * 3u * ((uint64_t)m + (uint64_t)dModel)
	    + (uint64_t)V + (uint64_t)m
	    + (uint64_t)(1 + 4 * L) + 64u;
	x.factorBytes = 4u * factorFloats;
	// Small tensors deliberately retain stock FP32 Adam for numerical simplicity.
	x.smallTensorBytes = 8u * (uint64_t)smallParams;
	x.orbitBytes = x.momentumBytes + x.factorBytes + x.smallTensorBytes;
	return x;
}

void chiron_orbit_row_col_sqsum_cpu(const float* g, int rows, int cols,
                                    float gradScale,
                                    float* rowMeanSq, float* colMeanSq,
                                    float* tensorMeanSq)
{
	if (!g || rows <= 0 || cols <= 0 || !rowMeanSq || !colMeanSq) return;
	std::fill(rowMeanSq, rowMeanSq + rows, 0.0f);
	std::fill(colMeanSq, colMeanSq + cols, 0.0f);
	double total = 0.0;
	for (int i = 0; i < rows; ++i)
	{
		double r = 0.0;
		for (int j = 0; j < cols; ++j)
		{
			const double v = (double)g[(size_t)i * cols + j] * gradScale;
			const double q = v * v;
			r += q; colMeanSq[j] += (float)q;
		}
		rowMeanSq[i] = (float)(r / (double)cols);
		total += r;
	}
	for (int j = 0; j < cols; ++j) colMeanSq[j] /= (float)rows;
	if (tensorMeanSq) *tensorMeanSq = (float)(total / ((double)rows * cols));
}

void chiron_orbit_adjoint_rows_cpu(const float* dp, int rows, int cols,
                                   float* channelMeanSq)
{
	if (!dp || rows <= 0 || cols <= 0 || !channelMeanSq) return;
	for (int j = 0; j < cols; ++j)
	{
		double s = 0.0;
		for (int i = 0; i < rows; ++i)
		{
			const double v = dp[(size_t)i * cols + j];
			s += v * v;
		}
		channelMeanSq[j] = (float)(s / rows);
	}
}

void chiron_orbit_h_colsq_strided_cpu(const float* h, int rows, int cols,
                                      int stride, float* colMeanSq)
{
	if (!h || rows <= 0 || cols <= 0 || !colMeanSq) return;
	if (stride < 1) stride = 1;
	const int count = (rows + stride - 1) / stride;
	for (int j = 0; j < cols; ++j)
	{
		double s = 0.0;
		for (int i = 0; i < rows; i += stride)
		{
			const double v = h[(size_t)i * cols + j];
			s += v * v;
		}
		colMeanSq[j] = (float)(s / std::max(1, count));
	}
}

void chiron_orbit_occupancy_cpu(const float* probs, int rows, int cols,
                                float* occupancy)
{
	if (!probs || rows <= 0 || cols <= 0 || !occupancy) return;
	for (int v = 0; v < cols; ++v)
	{
		double s = 0.0;
		for (int t = 0; t < rows; ++t)
		{
			const double p = probs[(size_t)t * cols + v];
			s += p * (1.0 - p);
		}
		occupancy[v] = (float)(s / rows);
	}
}

void chiron_orbit_embedding_input_cpu(const int* tokenIds, const float* dq,
                                      int rows, int vocab, int width,
                                      float* frequency, float* dqMeanSq)
{
	if (!tokenIds || !dq || rows <= 0 || vocab <= 0 || width <= 0 || !frequency) return;
	std::fill(frequency, frequency + vocab, 0.0f);
	double s = 0.0;
	for (int t = 0; t < rows; ++t)
	{
		const int id = tokenIds[t];
		if (id >= 0 && id < vocab) frequency[id] += 1.0f;
		for (int j = 0; j < width; ++j)
		{
			const double v = dq[(size_t)t * width + j];
			s += v * v;
		}
	}
	if (dqMeanSq) *dqMeanSq = (float)(s / ((double)rows * width));
}

void chiron_orbit_factored_step_cpu(float* param, const float* grad, float* momentum,
                                    int rows, int cols,
                                    const float* rowFactor, const float* colFactor,
                                    float scaleEma, float scaleBiasCorrection,
                                    float lr, float beta1, float eps,
                                    float weightDecay, float gradScale,
                                    int step, int freezeSteps, float globalScale,
                                    double* metricLengthSq)
{
	if (!param || !grad || !momentum || rows <= 0 || cols <= 0 || step <= 0) return;
	double rsum = 0.0, csum = 0.0;
	for (int i = 0; i < rows; ++i) rsum += rowFactor ? rowFactor[i] : 1.0;
	for (int j = 0; j < cols; ++j) csum += colFactor ? colFactor[j] : 1.0;
	const double rmean = rsum / rows;
	const double cmean = csum / cols;
	const double bc1 = 1.0 - std::pow((double)beta1, step);
	const double sbc = scaleBiasCorrection > 0.0f ? scaleBiasCorrection : 1.0;
	const double S = std::max(1e-30, (double)scaleEma / sbc);
	double length = 0.0;
	for (int i = 0; i < rows; ++i)
		for (int j = 0; j < cols; ++j)
		{
			const size_t k = (size_t)i * cols + j;
			const float g = grad[k] * gradScale;
			momentum[k] = beta1 * momentum[k] + (1.0f - beta1) * g;
			const double mh = momentum[k] / bc1;
			double vhat = S;
			if (step > freezeSteps)
			{
				const double rf = rowFactor ? std::max(1e-30, (double)rowFactor[i] / rmean) : 1.0;
				const double cf = colFactor ? std::max(1e-30, (double)colFactor[j] / cmean) : 1.0;
				vhat *= rf * cf;
			}
			const double u = mh / (std::sqrt(vhat) + eps);
			const double update = globalScale * lr * u;
			length += vhat * (lr * u) * (lr * u);
			param[k] -= (float)(lr * weightDecay * param[k] + update);
		}
	if (metricLengthSq) *metricLengthSq += length;
}

double chiron_orbit_correlation_cpu(const float* a, const float* b, size_t n)
{
	if (!a || !b || n < 2) return 0.0;
	double ma = 0.0, mb = 0.0;
	for (size_t i = 0; i < n; ++i) { ma += a[i]; mb += b[i]; }
	ma /= n; mb /= n;
	double aa = 0.0, bb = 0.0, ab = 0.0;
	for (size_t i = 0; i < n; ++i)
	{
		const double x = a[i] - ma, y = b[i] - mb;
		aa += x * x; bb += y * y; ab += x * y;
	}
	return (aa > 0.0 && bb > 0.0) ? ab / std::sqrt(aa * bb) : 0.0;
}

} // namespace chiron
} // namespace glades
