// GPU dispatch helpers for Glades ML.
//
// Provides compile-time and runtime dispatch macros/functions so call sites
// can switch between CPU and GPU paths with minimal boilerplate.
#pragma once

#include "gpu_device.h"

#include <cstddef>
#include <sys/time.h>

namespace glades {
namespace gpu {

struct DispatchPerfCounters
{
	unsigned long long kernelLaunches;
	unsigned long long syncPoints;
	unsigned long long bytesH2D;
	unsigned long long bytesD2H;
	unsigned long long bytesD2D;
	double msCompute;
	double msTransfer;
	double msSync;

	DispatchPerfCounters()
	    : kernelLaunches(0ULL),
	      syncPoints(0ULL),
	      bytesH2D(0ULL),
	      bytesD2H(0ULL),
	      bytesD2D(0ULL),
	      msCompute(0.0),
	      msTransfer(0.0),
	      msSync(0.0)
	{
	}

	void reset() { *this = DispatchPerfCounters(); }
};

inline void perfRecordKernel(DispatchPerfCounters* perf, unsigned long long count = 1ULL)
{
	if (perf)
		perf->kernelLaunches += count;
}

inline void perfRecordSync(DispatchPerfCounters* perf, unsigned long long count = 1ULL)
{
	if (perf)
		perf->syncPoints += count;
}

inline void perfRecordBytesH2D(DispatchPerfCounters* perf, size_t bytes)
{
	if (perf)
		perf->bytesH2D += static_cast<unsigned long long>(bytes);
}

inline void perfRecordBytesD2H(DispatchPerfCounters* perf, size_t bytes)
{
	if (perf)
		perf->bytesD2H += static_cast<unsigned long long>(bytes);
}

inline void perfRecordBytesD2D(DispatchPerfCounters* perf, size_t bytes)
{
	if (perf)
		perf->bytesD2D += static_cast<unsigned long long>(bytes);
}

class ScopedPerfTimerMs
{
public:
	explicit ScopedPerfTimerMs(double* acc)
	    : accumulator(acc), t0ms(0.0)
	{
		if (accumulator)
			t0ms = nowMs();
	}

	~ScopedPerfTimerMs()
	{
		if (!accumulator)
			return;
		*accumulator += (nowMs() - t0ms);
	}

private:
	static double nowMs()
	{
		struct timeval tv;
		gettimeofday(&tv, NULL);
		return static_cast<double>(tv.tv_sec) * 1000.0 + static_cast<double>(tv.tv_usec) / 1000.0;
	}

	double* accumulator;
	double t0ms;
};

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
