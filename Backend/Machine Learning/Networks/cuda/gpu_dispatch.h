// GPU dispatch helpers for Glades ML.
//
// Provides compile-time and runtime dispatch macros/functions so call sites
// can switch between CPU and GPU paths with minimal boilerplate.
#pragma once

#include "gpu_device.h"

namespace glades {
namespace gpu {

// Runtime check: returns true if GPU should be used for the current operation.
// Considers:
// 1. GLADES_HAVE_CUDA is defined (compile-time)
// 2. A GPU device was successfully initialized
// 3. The problem size exceeds a minimum threshold (optional)
inline bool shouldUseGpu(size_t problemSize = 0, size_t minProblemSize = 0)
{
#ifdef GLADES_HAVE_CUDA
	if (!isAvailable())
		return false;
	if (minProblemSize > 0 && problemSize < minProblemSize)
		return false;
	return true;
#else
	(void)problemSize;
	(void)minProblemSize;
	return false;
#endif
}

} // namespace gpu
} // namespace glades

// Compile-time + runtime dispatch macro.
//
// Usage:
//   GLADES_DISPATCH_KERNEL(cpuExpression, gpuExpression)
//
// If CUDA is compiled in and a GPU is available, evaluates gpuExpression;
// otherwise evaluates cpuExpression.
#ifdef GLADES_HAVE_CUDA
#define GLADES_DISPATCH_KERNEL(cpu_expr, gpu_expr) \
	do { \
		if (glades::gpu::shouldUseGpu()) { \
			gpu_expr; \
		} else { \
			cpu_expr; \
		} \
	} while (0)

// Dispatch with minimum problem size threshold.
#define GLADES_DISPATCH_KERNEL_SIZED(problemSize, minSize, cpu_expr, gpu_expr) \
	do { \
		if (glades::gpu::shouldUseGpu(problemSize, minSize)) { \
			gpu_expr; \
		} else { \
			cpu_expr; \
		} \
	} while (0)
#else
#define GLADES_DISPATCH_KERNEL(cpu_expr, gpu_expr) \
	do { cpu_expr; } while (0)

#define GLADES_DISPATCH_KERNEL_SIZED(problemSize, minSize, cpu_expr, gpu_expr) \
	do { (void)(problemSize); (void)(minSize); cpu_expr; } while (0)
#endif
