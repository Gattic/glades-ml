// CSP (Compressed-Sensing Proxy FFN) GPU primitives.
// See gpu_csp.h and research/PARADIGM_SHIFT_27_DESIGN.md.

#include "gpu_csp.h"
#include "gpu_device.h"
#include "gpu_blas.h"
#include "gpu_kernels.h"

#ifdef GLADES_HAVE_CUDA

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

namespace glades {
namespace gpu {

// (k_csp_add was removed in the Phase 1b fusion — sgemm_rowmajor_abt with
// beta=1.0f now folds the residual add into the matmul epilogue.)

bool csp_forward(const float* h_in,
                 const float* W_up, const float* W_down,
                 const float* U_sigma, const float* V_sigma,
                 unsigned int T, unsigned int d_model,
                 unsigned int m, unsigned int r_sigma,
                 int sigma_kind,
                 float* z_sketch_scratch,
                 float* y_scratch,
                 float* y_res_scratch,
                 float* h_out)
{
	if (h_in == nullptr || W_up == nullptr || W_down == nullptr) return false;
	if (z_sketch_scratch == nullptr || y_scratch == nullptr || h_out == nullptr)
		return false;
	if (T == 0u || d_model == 0u || m == 0u) return false;
	if (r_sigma > 0u && (U_sigma == nullptr || V_sigma == nullptr || y_res_scratch == nullptr))
		return false;
	// ReLU (kind=2) not supported for σ̂ base since σ'' is distributional at
	// the kink — caller should use GELU (0), SiLU (1), or identity (3).
	if (sigma_kind == 2) return false;

	// Step 1: z_sketch [T × m] = h_in [T × d_model] · W_up [d_model × m]
	if (!sgemm_rowmajor((int)T, (int)m, (int)d_model, 1.0f,
	                    h_in, (int)d_model,
	                    W_up, (int)m,
	                    0.0f,
	                    z_sketch_scratch, (int)m)) return false;

	// Step 2a: y_base = σ_base(z_sketch), elementwise.
	const unsigned int Tm = T * m;
	if (sigma_kind == 0) {
		if (!gelu_forward(z_sketch_scratch, (int)Tm, y_scratch)) return false;
	} else if (sigma_kind == 1) {
		if (!silu_forward(z_sketch_scratch, (int)Tm, y_scratch)) return false;
	} else {
		// identity: y_base = z_sketch (D2D copy).
		cudaMemcpyAsync(y_scratch, z_sketch_scratch, (size_t)Tm * sizeof(float),
		                cudaMemcpyDeviceToDevice, computeStream());
	}

	// Step 2b: residual term, only if r_sigma > 0.
	//   tmp_r [T × r_sigma] = z_sketch [T × m] · V_sigma [m × r_sigma]
	//   σ_hidden(tmp_r) (GELU elementwise as a sensible default)
	//   y [T × m] += tmp_r · U_sigma^T   (via sgemm_abt with beta=1)
	//
	// OPTIMIZATION (fused accumulate): instead of a separate add kernel,
	// sgemm_rowmajor_abt with beta=1.0f fuses the residual accumulation
	// into the matmul epilogue.  Saves 1 kernel launch per forward.
	// This was surfaced by benchmark finding #8 (launch-bound regime at
	// L2-resident dims) — see STACK_VALIDATION_SUMMARY.md.
	if (r_sigma > 0u) {
		// Use y_res_scratch first T·r_σ entries as tmp_r.
		if (!sgemm_rowmajor((int)T, (int)r_sigma, (int)m, 1.0f,
		                    z_sketch_scratch, (int)m,
		                    V_sigma, (int)r_sigma,
		                    0.0f,
		                    y_res_scratch, (int)r_sigma)) return false;
		// Apply GELU to the hidden residual.
		const unsigned int Tr = T * r_sigma;
		if (!gelu_forward(y_res_scratch, (int)Tr, y_res_scratch)) return false;
		// y [T × m] += tmp_r [T × r_sigma] · U_sigma^T [r_sigma × m]
		// (beta=1 accumulates directly into y_scratch).
		if (!sgemm_rowmajor_abt((int)T, (int)m, (int)r_sigma, 1.0f,
		                        y_res_scratch, (int)r_sigma,
		                        U_sigma, (int)r_sigma,
		                        1.0f,        // beta=1: y += alpha · (tmp_r · U^T)
		                        y_scratch, (int)m)) return false;
	}

	// Step 3: h_out [T × d_model] = y [T × m] · W_down [m × d_model]
	if (!sgemm_rowmajor((int)T, (int)d_model, (int)m, 1.0f,
	                    y_scratch, (int)m,
	                    W_down, (int)d_model,
	                    0.0f,
	                    h_out, (int)d_model)) return false;

	return true;
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
