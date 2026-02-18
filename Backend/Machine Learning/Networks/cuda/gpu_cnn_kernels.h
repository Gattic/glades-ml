// CNN CUDA kernel declarations + non-CUDA stubs.
#pragma once

#ifdef GLADES_HAVE_CUDA

namespace glades {
namespace gpu {

// im2col: extract patches from NCHW image into [outH*outW, inC*kH*kW] column matrix.
// input: [inC, inH, inW], output: [outH*outW, inC*kH*kW]
void im2col_gpu(const float* input,
                unsigned int inC, unsigned int inH, unsigned int inW,
                unsigned int kH, unsigned int kW,
                unsigned int strideH, unsigned int strideW,
                unsigned int padH, unsigned int padW,
                unsigned int outH, unsigned int outW,
                float* output);

// col2im: scatter column matrix back to image (accumulate, for gradient).
// cols: [outH*outW, inC*kH*kW], output: [inC, inH, inW] (accumulated with atomicAdd)
void col2im_gpu(const float* cols,
                unsigned int inC, unsigned int inH, unsigned int inW,
                unsigned int kH, unsigned int kW,
                unsigned int strideH, unsigned int strideW,
                unsigned int padH, unsigned int padW,
                unsigned int outH, unsigned int outW,
                float* output);

// Max pooling forward: input [C, H, W] -> output [C, poolOutH, poolOutW] + argmax indices.
void maxpool_forward_gpu(const float* input,
                         unsigned int C, unsigned int H, unsigned int W,
                         unsigned int poolH, unsigned int poolW,
                         unsigned int poolStrideH, unsigned int poolStrideW,
                         unsigned int outH, unsigned int outW,
                         float* output, int* argmax);

// Max pooling backward: scatter gradient via argmax indices.
void maxpool_backward_gpu(const float* dOutput,
                          const int* argmax,
                          unsigned int C, unsigned int inH, unsigned int inW,
                          unsigned int outH, unsigned int outW,
                          float* dInput);

// BatchNorm forward (training): per-channel over spatial dims.
// input: [C, N] where N = H*W spatial positions.
// Computes mean, invStd, normalized output, applies gamma/beta.
void batchnorm_forward_train_gpu(const float* input,
                                 unsigned int C, unsigned int N,
                                 const float* gamma, const float* beta,
                                 float eps,
                                 float* output,
                                 float* mean, float* invStd, float* normalized);

// BatchNorm forward (inference): uses running mean/var.
void batchnorm_forward_infer_gpu(const float* input,
                                 unsigned int C, unsigned int N,
                                 const float* gamma, const float* beta,
                                 const float* runMean, const float* runVar,
                                 float eps,
                                 float* output);

// BatchNorm backward: compute dx, dgamma, dbeta.
void batchnorm_backward_gpu(const float* dOutput,
                            const float* normalized,
                            const float* gamma,
                            const float* invStd,
                            unsigned int C, unsigned int N,
                            float* dInput,
                            float* dgamma, float* dbeta);

} // namespace gpu
} // namespace glades

#else // !GLADES_HAVE_CUDA

namespace glades {
namespace gpu {

inline void im2col_gpu(const float*, unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int, unsigned int, float*) {}
inline void col2im_gpu(const float*, unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int, unsigned int, float*) {}
inline void maxpool_forward_gpu(const float*, unsigned int, unsigned int, unsigned int,
                                unsigned int, unsigned int, unsigned int, unsigned int,
                                unsigned int, unsigned int, float*, int*) {}
inline void maxpool_backward_gpu(const float*, const int*, unsigned int, unsigned int,
                                 unsigned int, unsigned int, unsigned int, float*) {}
inline void batchnorm_forward_train_gpu(const float*, unsigned int, unsigned int,
                                        const float*, const float*, float,
                                        float*, float*, float*, float*) {}
inline void batchnorm_forward_infer_gpu(const float*, unsigned int, unsigned int,
                                        const float*, const float*,
                                        const float*, const float*, float, float*) {}
inline void batchnorm_backward_gpu(const float*, const float*, const float*,
                                   const float*, unsigned int, unsigned int,
                                   float*, float*, float*) {}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
