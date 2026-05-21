// Unit test for ORION INT8 V kernels.
//
// Validates:
//   1. Round-trip error of quantize → dequantize on random orthonormal-ish V.
//   2. Fused axpy matches naive (dequant + axpy).
//   3. V^T dot matches naive (dequant + dot).
//
// Build (from glades-ml repo root, after kernels are compiled into a shared lib):
//   nvcc -O2 -arch=sm_89 -c research/orion_int8_v_kernels.cu -o orion_kernels.o
//   nvcc -O2 -arch=sm_89 research/orion_int8_v_test.cpp orion_kernels.o -o research/orion_int8_v_test
//
// Usage:  research/orion_int8_v_test
//
// Expected output: all 3 tests PASS; quantization L2 error <= 0.005 (relative).

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

extern "C" {
    int  orion_v_scale_count(int n);
    bool orion_v_quantize_int8_column(const float* src, int8_t* dst_q, float* scales, int n, cudaStream_t);
    bool orion_v_dequantize_int8_column(const int8_t* src_q, const float* scales, float* dst, int n, cudaStream_t);
    bool orion_axpy_int8_v_column(float* theta, const int8_t* V_q, const float* scales, int n, int col_idx, float alpha, cudaStream_t);
    bool orion_vt_dot_int8_columns(const int8_t* V_q, const float* scales, const float* x, float* result, int n, int r, cudaStream_t);
}

static double l2(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = (double)a[i] - (double)b[i];
        s += d * d;
    }
    return std::sqrt(s);
}

static double l2_norm(const std::vector<float>& a) {
    double s = 0.0;
    for (float v : a) s += (double)v * (double)v;
    return std::sqrt(s);
}

int main(int argc, char** argv) {
    // ORION-relevant test shapes:
    //   E with r=4 at 1B: n = 32000*2048 = 65.5M, r = 4
    //   per-layer Wq at 1B: n = 2048*4096 = 8.4M, r = 4
    // We use a smaller test to fit the test budget; n is dialed up via argv[1].
    int n = 8388608;   // 8M elements (~ Wq scale)
    int r = 4;
    if (argc >= 2) n = std::atoi(argv[1]);
    if (argc >= 3) r = std::atoi(argv[2]);

    std::printf("== ORION INT8 V kernel test ==\n");
    std::printf("n = %d, r = %d, scale_count = %d\n", n, r, orion_v_scale_count(n));
    std::printf("VRAM:  BF16 V = %.2f MB,  INT8 V + scales = %.2f MB,  save = %.1f%%\n",
                (double)n * r * 2 / 1048576.0,
                ((double)n * r * 1 + (double)orion_v_scale_count(n) * r * 4) / 1048576.0,
                100.0 * (1.0 - ((double)n * r + (double)orion_v_scale_count(n) * r * 4) / ((double)n * r * 2)));

    // Generate a random V where each column is roughly orthonormal-ish
    // (unit-norm, gaussian).  Values in roughly [-3*sigma, 3*sigma].
    std::vector<float> h_V_fp32((size_t)n * r);
    {
        std::mt19937 rng(1337);
        std::normal_distribution<float> nd(0.0f, 1.0f / std::sqrt((float)n));
        for (size_t i = 0; i < h_V_fp32.size(); ++i) h_V_fp32[i] = nd(rng);
    }
    std::vector<float> h_x(n);
    {
        std::mt19937 rng(7777);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (int i = 0; i < n; ++i) h_x[i] = nd(rng);
    }

    // GPU allocations.
    float*  d_V_fp32   = nullptr;
    int8_t* d_V_int8   = nullptr;
    float*  d_scales   = nullptr;
    float*  d_x        = nullptr;
    float*  d_dst_fp32 = nullptr;
    float*  d_theta    = nullptr;
    float*  d_result   = nullptr;
    cudaMalloc(&d_V_fp32, sizeof(float) * (size_t)n * r);
    cudaMalloc(&d_V_int8, sizeof(int8_t) * (size_t)n * r);
    cudaMalloc(&d_scales, sizeof(float) * (size_t)orion_v_scale_count(n) * r);
    cudaMalloc(&d_x, sizeof(float) * n);
    cudaMalloc(&d_dst_fp32, sizeof(float) * n);
    cudaMalloc(&d_theta, sizeof(float) * n);
    cudaMalloc(&d_result, sizeof(float) * r);

    cudaMemcpy(d_V_fp32, h_V_fp32.data(), sizeof(float) * (size_t)n * r, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), sizeof(float) * n, cudaMemcpyHostToDevice);

    // 1. Quantize each column, dequantize, measure L2 error per column.
    std::printf("\n[Test 1] Quantize → Dequantize round-trip\n");
    double max_rel_err = 0.0;
    int failed = 0;
    for (int k = 0; k < r; ++k) {
        const int sc = orion_v_scale_count(n);
        orion_v_quantize_int8_column(
            d_V_fp32 + (size_t)k * n,
            d_V_int8 + (size_t)k * n,
            d_scales + (size_t)k * sc,
            n, 0);
        orion_v_dequantize_int8_column(
            d_V_int8 + (size_t)k * n,
            d_scales + (size_t)k * sc,
            d_dst_fp32, n, 0);
        cudaDeviceSynchronize();

        std::vector<float> h_rec(n);
        cudaMemcpy(h_rec.data(), d_dst_fp32, sizeof(float) * n, cudaMemcpyDeviceToHost);

        std::vector<float> h_orig(h_V_fp32.begin() + (size_t)k * n,
                                   h_V_fp32.begin() + (size_t)(k + 1) * n);
        const double err = l2(h_orig, h_rec);
        const double nrm = l2_norm(h_orig);
        const double rel = err / (nrm + 1e-30);
        if (rel > max_rel_err) max_rel_err = rel;
        const bool pass = rel <= 0.005;
        if (!pass) ++failed;
        std::printf("  col[%d]: ‖orig‖=%.4e  ‖err‖=%.4e  rel=%.5f  %s\n",
                    k, nrm, err, rel, pass ? "PASS" : "FAIL");
    }
    std::printf("[Test 1] max rel err = %.5f  ->  %s\n",
                max_rel_err, (failed == 0) ? "PASS" : "FAIL");

    // 2. Fused axpy: theta += alpha * V[:, k] vs (dequant + axpy).
    std::printf("\n[Test 2] Fused axpy_int8_v_column vs naive dequant+axpy\n");
    const float alpha = 0.3f;
    // theta starts at zeros.
    cudaMemset(d_theta, 0, sizeof(float) * n);
    for (int k = 0; k < r; ++k) {
        orion_axpy_int8_v_column(d_theta, d_V_int8, d_scales, n, k, alpha, 0);
    }
    cudaDeviceSynchronize();
    std::vector<float> h_fused(n);
    cudaMemcpy(h_fused.data(), d_theta, sizeof(float) * n, cudaMemcpyDeviceToHost);

    // Reference: dequant + axpy
    std::vector<float> h_naive(n, 0.0f);
    for (int k = 0; k < r; ++k) {
        const int sc = orion_v_scale_count(n);
        orion_v_dequantize_int8_column(
            d_V_int8 + (size_t)k * n,
            d_scales + (size_t)k * sc,
            d_dst_fp32, n, 0);
        cudaDeviceSynchronize();
        std::vector<float> h_dq(n);
        cudaMemcpy(h_dq.data(), d_dst_fp32, sizeof(float) * n, cudaMemcpyDeviceToHost);
        for (int i = 0; i < n; ++i) h_naive[i] += alpha * h_dq[i];
    }
    const double t2_err = l2(h_fused, h_naive);
    const double t2_nrm = l2_norm(h_naive);
    std::printf("  ‖fused − naive‖ = %.4e   relative to naive = %.5f   %s\n",
                t2_err, t2_err / (t2_nrm + 1e-30),
                (t2_err / (t2_nrm + 1e-30) <= 1e-5) ? "PASS" : "FAIL");

    // 3. V^T dot: result[k] = V[:, k] · x vs naive (dequant + dot).
    std::printf("\n[Test 3] orion_vt_dot_int8_columns vs naive dequant+dot\n");
    orion_vt_dot_int8_columns(d_V_int8, d_scales, d_x, d_result, n, r, 0);
    cudaDeviceSynchronize();
    std::vector<float> h_fused_dot(r);
    cudaMemcpy(h_fused_dot.data(), d_result, sizeof(float) * r, cudaMemcpyDeviceToHost);

    std::vector<float> h_naive_dot(r, 0.0f);
    for (int k = 0; k < r; ++k) {
        const int sc = orion_v_scale_count(n);
        orion_v_dequantize_int8_column(
            d_V_int8 + (size_t)k * n,
            d_scales + (size_t)k * sc,
            d_dst_fp32, n, 0);
        cudaDeviceSynchronize();
        std::vector<float> h_dq(n);
        cudaMemcpy(h_dq.data(), d_dst_fp32, sizeof(float) * n, cudaMemcpyDeviceToHost);
        double s = 0.0;
        for (int i = 0; i < n; ++i) s += (double)h_dq[i] * h_x[i];
        h_naive_dot[k] = (float)s;
    }
    int t3_failed = 0;
    for (int k = 0; k < r; ++k) {
        const float diff = std::fabs(h_fused_dot[k] - h_naive_dot[k]);
        const float rel  = diff / (std::fabs(h_naive_dot[k]) + 1e-30f);
        const bool pass = rel <= 1e-4f;
        if (!pass) ++t3_failed;
        std::printf("  result[%d]: fused=%.6e  naive=%.6e  rel=%.5e  %s\n",
                    k, h_fused_dot[k], h_naive_dot[k], rel, pass ? "PASS" : "FAIL");
    }

    cudaFree(d_V_fp32); cudaFree(d_V_int8); cudaFree(d_scales);
    cudaFree(d_x); cudaFree(d_dst_fp32); cudaFree(d_theta); cudaFree(d_result);

    const int total_failed = failed + (t2_err / (t2_nrm + 1e-30) > 1e-5 ? 1 : 0) + t3_failed;
    std::printf("\n== SUMMARY ==  %s\n", total_failed == 0 ? "ALL PASS" : "FAIL");
    return total_failed == 0 ? 0 : 1;
}
