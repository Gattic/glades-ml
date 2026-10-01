// CUDA primitives for ORBIT, CHIRON's increment-space factored optimizer.
// Design: docs/superpowers/specs/2026-07-09-chiron-native-optimizers-design.md
#pragma once

#include <cstddef>
#include <stdint.h>

namespace glades {
namespace gpu {

#ifdef GLADES_HAVE_CUDA

// Scratch sizes for orbit_row_col_sqsum[_bf16]. Scratch is reusable across
// tensors because all launches use computeStream() in program order.
size_t orbit_row_partial_count(int rows, int cols);
size_t orbit_col_partial_count(int rows, int cols);

// Fused one-read tiled row+column squared-sum pass followed by EMA updates.
// rowEma/colEma hold mean-square factors; scaleEma holds tensor mean-square.
// rowSum/colSum are device scalars containing sums of the updated EMA vectors.
bool orbit_row_col_sqsum(const float* grad, int rows, int cols, float gradScale,
                         float betaF, float* rowEma, float* colEma, float* scaleEma,
                         float* rowSum, float* colSum,
                         float* rowPartial, float* colPartial, float* scalarScratch);
bool orbit_row_col_sqsum_bf16(const uint16_t* grad, int rows, int cols, float gradScale,
                              float betaF, float* rowEma, float* colEma, float* scaleEma,
                              float* rowSum, float* colSum,
                              float* rowPartial, float* colPartial, float* scalarScratch);

// Embedding-specialized col moments: coalesced 64-row tiles, no V*m scratch.
bool orbit_col_sqsum_ema(const float* grad, int rows, int cols, float gradScale,
                         float betaF, float* colEma, float* scaleEma,
                         float* colSum, float* colSample, float* scalarScratch);

// Tensor scale only (used by Wo, whose shape factors come from dp and h).
bool orbit_scale_ema(const float* grad, int n, float gradScale,
                     float betaF, float* scaleEma, float* scalarScratch);
bool orbit_scale_ema_bf16(const uint16_t* grad, int n, float gradScale,
                          float betaF, float* scaleEma, float* scalarScratch);

// Backward-time sample collectors. They only write sample buffers; persistent
// EMAs are updated later, after the finite-gradient guard accepts the step.
bool orbit_adjoint_rows(const float* dp, int rows, int cols,
                        float sampleWeight, bool reset, float* channelSample);
bool orbit_h_colsq_strided(const float* h, int rows, int cols, int stride,
                           float* colSample);
bool orbit_occupancy_bwd(const float* probs, int rows, int cols,
                         float* occupancySample);
bool orbit_occupancy_bwd_bf16(const uint16_t* probs, int rows, int cols,
                              float* occupancySample);
bool orbit_embedding_input_stats(const int* tokenIds, const float* dq,
                                 int rows, int vocab, int width,
                                 float* frequencySample, float* dqMeanSqSample);

// Accepted-step EMA and factor preparation helpers.
bool orbit_vector_ema(const float* sample, float* ema, int n, float betaF,
                      float* emaSum);
bool orbit_scalar_ema(const float* sample, float* ema, float betaF);
bool orbit_build_embedding_rows(const float* occupancy,
                                const float* frequency,
                                const float* dqMeanSq,
                                int vocab, float kappa, float factorBiasCorrection,
                                float* rowFactor, float* rowSum);
// Deterministic approximate quantile: at most 1024 evenly-spaced samples are
// bitonic-sorted on device. q must be in [0,1].
bool orbit_quantile(const float* values, int n, float q, float* quantileOut);
bool orbit_floor_and_sum(const float* values, int n, const float* floorValue,
                         float* prepared, float* preparedSum);

// Diagnostics slots emitted by the factored step on requested log steps.
enum OrbitDiagSlot
{
	ORBIT_DIAG_COUNT = 0,
	ORBIT_DIAG_UPDATE_SQ,
	ORBIT_DIAG_WEIGHT_SQ,
	ORBIT_DIAG_V_SUM,
	ORBIT_DIAG_G2_SUM,
	ORBIT_DIAG_V2_SUM,
	ORBIT_DIAG_G4_SUM,
	ORBIT_DIAG_VG2_SUM,
	ORBIT_DIAG_FACTOR_MIN,
	ORBIT_DIAG_FACTOR_MAX,
	ORBIT_DIAG_SIZE
};
bool orbit_diagnostics_reset(float* diagnostics);

// Int8-block momentum + factored second-moment ORBIT step. rowSum/colSum are
// sums (not means). scaleEma is bias-corrected in-kernel with betaF and step.
// When deltaF>0, delayedLengthSq supplies the previous accepted step length,
// currentLengthSq receives this step's metric length. deltaF=0 avoids length
// atomics and makes the multiplier exactly one.
bool orbit_factored_step(float* param, const float* grad,
                         int8_t* momentum, float* momentumScale,
                         const float* rowFactor, const float* rowSum,
                         const float* colFactor, const float* colSum,
                         const float* scaleEma,
                         int rows, int cols,
                         float lr, float beta1, float betaF, float eps,
                         float weightDecay, bool metricWeightDecay, float gradScale,
                         int step, int freezeSteps, float deltaF,
                         const float* delayedLengthSq, float* currentLengthSq,
                         float* diagnostics);
bool orbit_factored_step_bf16(uint16_t* param, const uint16_t* grad,
                              int8_t* momentum, float* momentumScale,
                              const float* rowFactor, const float* rowSum,
                              const float* colFactor, const float* colSum,
                              const float* scaleEma,
                              int rows, int cols,
                              float lr, float beta1, float betaF, float eps,
                              float weightDecay, bool metricWeightDecay, float gradScale,
                              int step, int freezeSteps, float deltaF,
                              const float* delayedLengthSq, float* currentLengthSq,
                              uint32_t srBaseSeed, uint32_t srStepIdx,
                              float* diagnostics);

#else

inline size_t orbit_row_partial_count(int rows, int cols) { return (size_t)rows * (size_t)((cols + 255) / 256); }
inline size_t orbit_col_partial_count(int rows, int cols) { return (size_t)cols * (size_t)((rows + 7) / 8); }
inline bool orbit_row_col_sqsum(const float*,int,int,float,float,float*,float*,float*,float*,float*,float*,float*,float*) { return false; }
inline bool orbit_row_col_sqsum_bf16(const uint16_t*,int,int,float,float,float*,float*,float*,float*,float*,float*,float*,float*) { return false; }
inline bool orbit_col_sqsum_ema(const float*,int,int,float,float,float*,float*,float*,float*,float*) { return false; }
inline bool orbit_scale_ema(const float*,int,float,float,float*,float*) { return false; }
inline bool orbit_scale_ema_bf16(const uint16_t*,int,float,float,float*,float*) { return false; }
inline bool orbit_adjoint_rows(const float*,int,int,float,bool,float*) { return false; }
inline bool orbit_h_colsq_strided(const float*,int,int,int,float*) { return false; }
inline bool orbit_occupancy_bwd(const float*,int,int,float*) { return false; }
inline bool orbit_occupancy_bwd_bf16(const uint16_t*,int,int,float*) { return false; }
inline bool orbit_embedding_input_stats(const int*,const float*,int,int,int,float*,float*) { return false; }
inline bool orbit_vector_ema(const float*,float*,int,float,float*) { return false; }
inline bool orbit_scalar_ema(const float*,float*,float) { return false; }
inline bool orbit_build_embedding_rows(const float*,const float*,const float*,int,float,float,float*,float*) { return false; }
inline bool orbit_quantile(const float*,int,float,float*) { return false; }
inline bool orbit_floor_and_sum(const float*,int,const float*,float*,float*) { return false; }
enum OrbitDiagSlot { ORBIT_DIAG_COUNT=0, ORBIT_DIAG_UPDATE_SQ, ORBIT_DIAG_WEIGHT_SQ, ORBIT_DIAG_V_SUM, ORBIT_DIAG_G2_SUM, ORBIT_DIAG_V2_SUM, ORBIT_DIAG_G4_SUM, ORBIT_DIAG_VG2_SUM, ORBIT_DIAG_FACTOR_MIN, ORBIT_DIAG_FACTOR_MAX, ORBIT_DIAG_SIZE };
inline bool orbit_diagnostics_reset(float*) { return false; }
inline bool orbit_factored_step(float*,const float*,int8_t*,float*,const float*,const float*,const float*,const float*,const float*,int,int,float,float,float,float,float,bool,float,int,int,float,const float*,float*,float*) { return false; }
inline bool orbit_factored_step_bf16(uint16_t*,const uint16_t*,int8_t*,float*,const float*,const float*,const float*,const float*,const float*,int,int,float,float,float,float,float,bool,float,int,int,float,const float*,float*,uint32_t,uint32_t,float*) { return false; }

#endif

} // namespace gpu
} // namespace glades
