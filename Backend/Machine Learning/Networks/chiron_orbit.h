// CHIRON ORBIT optimizer CPU references and memory accounting.
// Design: docs/superpowers/specs/2026-07-09-chiron-native-optimizers-design.md
#ifndef _GLADES_CHIRON_ORBIT_H_
#define _GLADES_CHIRON_ORBIT_H_

#include <cstddef>
#include <stdint.h>
#include <vector>

namespace glades {
namespace chiron {

struct OrbitMemoryEstimate
{
	uint64_t parameterCount;
	uint64_t baselineBytes;
	uint64_t orbitBytes;
	uint64_t momentumBytes;
	uint64_t factorBytes;
	uint64_t smallTensorBytes;
	OrbitMemoryEstimate()
	    : parameterCount(0), baselineBytes(0), orbitBytes(0), momentumBytes(0),
	      factorBytes(0), smallTensorBytes(0) {}
};

// Flagship-oriented optimizer-state accounting. Matrix tensors are E plus the
// four attention matrices per layer; smallParams retain stock FP32 Adam.
OrbitMemoryEstimate chiron_orbit_memory_estimate(int V, int m, int dModel,
                                                  int L, size_t smallParams);

void chiron_orbit_row_col_sqsum_cpu(const float* g, int rows, int cols,
                                    float gradScale,
                                    float* rowMeanSq, float* colMeanSq,
                                    float* tensorMeanSq);
void chiron_orbit_adjoint_rows_cpu(const float* dp, int rows, int cols,
                                   float* channelMeanSq);
void chiron_orbit_h_colsq_strided_cpu(const float* h, int rows, int cols,
                                      int stride, float* colMeanSq);
void chiron_orbit_occupancy_cpu(const float* probs, int rows, int cols,
                                float* occupancy);
void chiron_orbit_embedding_input_cpu(const int* tokenIds, const float* dq,
                                      int rows, int vocab, int width,
                                      float* frequency, float* dqMeanSq);

// Dense reference for one ORBIT matrix step. Momentum is represented in FP32
// here so tests can isolate the factored-metric math from the GPU int8 codec.
// row/col factors are EMA values and are normalized to mean one internally.
void chiron_orbit_factored_step_cpu(float* param, const float* grad, float* momentum,
                                    int rows, int cols,
                                    const float* rowFactor, const float* colFactor,
                                    float scaleEma, float scaleBiasCorrection,
                                    float lr, float beta1, float eps,
                                    float weightDecay, float gradScale,
                                    int step, int freezeSteps, float globalScale,
                                    double* metricLengthSq);

// Pearson correlation used by E2 factorization-defect tests/telemetry.
double chiron_orbit_correlation_cpu(const float* a, const float* b, size_t n);

} // namespace chiron
} // namespace glades

#endif
