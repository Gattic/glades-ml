// GPU buffer implementation (CUDA).
#include "gpu_buffer.h"
#include "gpu_device.h"

#ifdef GLADES_HAVE_CUDA

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <stdint.h>

namespace glades {
namespace gpu {

namespace {

// Blocking buffer operations are expected to observe all prior queued work.
// After the stream split to non-blocking compute/transfer streams, plain
// cudaMemcpy/cudaMemset no longer implied that ordering, so synchronize here.
static bool syncBlockingBufferOp()
{
	return synchronizeComputeStream() && synchronizeTransferStream();
}

} // namespace

// ---- float specialization ----

template <>
GpuBuffer<float>::GpuBuffer() : d_ptr(0), n(0) {}

template <>
void GpuBuffer<float>::free()
{
	if (d_ptr)
	{
		cudaFree(d_ptr);
		d_ptr = 0;
	}
	n = 0;
}

template <>
GpuBuffer<float>::~GpuBuffer() { free(); }

template <>
bool GpuBuffer<float>::allocate(size_t count)
{
	free();
	if (count == 0)
		return true;

	// Diagnostic: when GLADES_LOG_GPU_ALLOC=1, print every >50M-float
	// (200 MB) cudaMalloc as it happens.  Useful for tracing which scratch
	// buffer is the binding constraint at the next param-ceiling lift.
	// GLADES_LOG_GPU_ALLOC=2 lowers the threshold to 1M floats (4 MB) to
	// catch the smaller per-block weight allocations.
	{
		static int s_log = -1;
		if (s_log < 0) { const char* e = getenv("GLADES_LOG_GPU_ALLOC");
		                 s_log = (e ? (*e == '2' ? 2 : (*e == '1' ? 1 : 0)) : 0); }
		const size_t threshFloats = (s_log == 2) ? (1ULL * 1024ULL * 1024ULL)
		                                          : (50ULL * 1024ULL * 1024ULL);
		if (s_log && count > threshFloats)
			fprintf(stderr, "[glades-cuda-alloc] %zu floats (%.2f MB)\n",
			        count, (double)(count * sizeof(float)) / (1024.0 * 1024.0));
	}

	cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_ptr), count * sizeof(float));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "[glades-cuda] cudaMalloc(%zu floats, %zu bytes) failed: %s\n",
		        count, count * sizeof(float), cudaGetErrorString(err));
		d_ptr = 0;
		n = 0;
		return false;
	}
	n = count;
	return true;
}

template <>
bool GpuBuffer<float>::upload(const float* src, size_t count)
{
	if (!d_ptr || !src)
		return false;
	if (count == 0)
		count = n;
	if (count > n)
		return false;
	if (!syncBlockingBufferOp())
		return false;
	cudaError_t err = cudaMemcpy(d_ptr, src, count * sizeof(float), cudaMemcpyHostToDevice);
	return err == cudaSuccess;
}

template <>
bool GpuBuffer<float>::uploadAsync(const float* src, size_t count)
{
	if (!d_ptr || !src)
		return false;
	if (count == 0)
		count = n;
	if (count > n)
		return false;
	cudaError_t err = cudaMemcpyAsync(d_ptr, src, count * sizeof(float), cudaMemcpyHostToDevice, transferStream());
	return err == cudaSuccess;
}

template <>
bool GpuBuffer<float>::download(float* dst, size_t count) const
{
	if (!d_ptr || !dst)
		return false;
	if (count == 0)
		count = n;
	if (count > n)
		return false;
	if (!syncBlockingBufferOp())
		return false;
	cudaError_t err = cudaMemcpy(dst, d_ptr, count * sizeof(float), cudaMemcpyDeviceToHost);
	return err == cudaSuccess;
}

template <>
bool GpuBuffer<float>::downloadAsync(float* dst, size_t count) const
{
	if (!d_ptr || !dst)
		return false;
	if (count == 0)
		count = n;
	if (count > n)
		return false;
	cudaError_t err = cudaMemcpyAsync(dst, d_ptr, count * sizeof(float), cudaMemcpyDeviceToHost, transferStream());
	return err == cudaSuccess;
}

template <>
bool GpuBuffer<float>::zero()
{
	if (!d_ptr)
		return false;
	if (!syncBlockingBufferOp())
		return false;
	cudaError_t err = cudaMemset(d_ptr, 0, n * sizeof(float));
	return err == cudaSuccess;
}

template <>
bool GpuBuffer<float>::zeroAsync()
{
	if (!d_ptr)
		return false;
	cudaError_t err = cudaMemsetAsync(d_ptr, 0, n * sizeof(float), computeStream());
	return err == cudaSuccess;
}

// ---- int specialization ----

template <>
GpuBuffer<int>::GpuBuffer() : d_ptr(0), n(0) {}

template <>
void GpuBuffer<int>::free()
{
	if (d_ptr)
	{
		cudaFree(d_ptr);
		d_ptr = 0;
	}
	n = 0;
}

template <>
GpuBuffer<int>::~GpuBuffer() { free(); }

template <>
bool GpuBuffer<int>::allocate(size_t count)
{
	free();
	if (count == 0)
		return true;
	cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_ptr), count * sizeof(int));
	if (err != cudaSuccess)
	{
		d_ptr = 0;
		n = 0;
		return false;
	}
	n = count;
	return true;
}

template <>
bool GpuBuffer<int>::upload(const int* src, size_t count)
{
	if (!d_ptr || !src)
		return false;
	if (count == 0)
		count = n;
	if (count > n)
		return false;
	if (!syncBlockingBufferOp())
		return false;
	cudaError_t err = cudaMemcpy(d_ptr, src, count * sizeof(int), cudaMemcpyHostToDevice);
	return err == cudaSuccess;
}

template <>
bool GpuBuffer<int>::uploadAsync(const int* src, size_t count)
{
	if (!d_ptr || !src)
		return false;
	if (count == 0)
		count = n;
	if (count > n)
		return false;
	cudaError_t err = cudaMemcpyAsync(d_ptr, src, count * sizeof(int), cudaMemcpyHostToDevice, transferStream());
	return err == cudaSuccess;
}

template <>
bool GpuBuffer<int>::download(int* dst, size_t count) const
{
	if (!d_ptr || !dst)
		return false;
	if (count == 0)
		count = n;
	if (count > n)
		return false;
	if (!syncBlockingBufferOp())
		return false;
	cudaError_t err = cudaMemcpy(dst, d_ptr, count * sizeof(int), cudaMemcpyDeviceToHost);
	return err == cudaSuccess;
}

template <>
bool GpuBuffer<int>::downloadAsync(int* dst, size_t count) const
{
	if (!d_ptr || !dst)
		return false;
	if (count == 0)
		count = n;
	if (count > n)
		return false;
	cudaError_t err = cudaMemcpyAsync(dst, d_ptr, count * sizeof(int), cudaMemcpyDeviceToHost, transferStream());
	return err == cudaSuccess;
}

template <>
bool GpuBuffer<int>::zero()
{
	if (!d_ptr)
		return false;
	if (!syncBlockingBufferOp())
		return false;
	cudaError_t err = cudaMemset(d_ptr, 0, n * sizeof(int));
	return err == cudaSuccess;
}

template <>
bool GpuBuffer<int>::zeroAsync()
{
	if (!d_ptr)
		return false;
	cudaError_t err = cudaMemsetAsync(d_ptr, 0, n * sizeof(int), computeStream());
	return err == cudaSuccess;
}

// ---- uint16_t specialization ----

template <>
GpuBuffer<uint16_t>::GpuBuffer() : d_ptr(0), n(0) {}

template <>
void GpuBuffer<uint16_t>::free()
{
	if (d_ptr)
	{
		cudaFree(d_ptr);
		d_ptr = 0;
	}
	n = 0;
}

template <>
GpuBuffer<uint16_t>::~GpuBuffer() { free(); }

template <>
bool GpuBuffer<uint16_t>::allocate(size_t count)
{
	free();
	if (count == 0)
		return true;
	cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_ptr), count * sizeof(uint16_t));
	if (err != cudaSuccess)
	{
		d_ptr = 0;
		n = 0;
		return false;
	}
	n = count;
	return true;
}

template <>
bool GpuBuffer<uint16_t>::upload(const uint16_t* src, size_t count)
{
	if (!d_ptr || !src)
		return false;
	if (count == 0)
		count = n;
	if (count > n)
		return false;
	if (!syncBlockingBufferOp())
		return false;
	cudaError_t err = cudaMemcpy(d_ptr, src, count * sizeof(uint16_t), cudaMemcpyHostToDevice);
	return err == cudaSuccess;
}

template <>
bool GpuBuffer<uint16_t>::uploadAsync(const uint16_t* src, size_t count)
{
	if (!d_ptr || !src)
		return false;
	if (count == 0)
		count = n;
	if (count > n)
		return false;
	cudaError_t err = cudaMemcpyAsync(d_ptr, src, count * sizeof(uint16_t), cudaMemcpyHostToDevice, transferStream());
	return err == cudaSuccess;
}

template <>
bool GpuBuffer<uint16_t>::download(uint16_t* dst, size_t count) const
{
	if (!d_ptr || !dst)
		return false;
	if (count == 0)
		count = n;
	if (count > n)
		return false;
	if (!syncBlockingBufferOp())
		return false;
	cudaError_t err = cudaMemcpy(dst, d_ptr, count * sizeof(uint16_t), cudaMemcpyDeviceToHost);
	return err == cudaSuccess;
}

template <>
bool GpuBuffer<uint16_t>::downloadAsync(uint16_t* dst, size_t count) const
{
	if (!d_ptr || !dst)
		return false;
	if (count == 0)
		count = n;
	if (count > n)
		return false;
	cudaError_t err = cudaMemcpyAsync(dst, d_ptr, count * sizeof(uint16_t), cudaMemcpyDeviceToHost, transferStream());
	return err == cudaSuccess;
}

template <>
bool GpuBuffer<uint16_t>::zero()
{
	if (!d_ptr)
		return false;
	if (!syncBlockingBufferOp())
		return false;
	cudaError_t err = cudaMemset(d_ptr, 0, n * sizeof(uint16_t));
	return err == cudaSuccess;
}

template <>
bool GpuBuffer<uint16_t>::zeroAsync()
{
	if (!d_ptr)
		return false;
	cudaError_t err = cudaMemsetAsync(d_ptr, 0, n * sizeof(uint16_t), computeStream());
	return err == cudaSuccess;
}

// ---- unsigned int specialization ----

template <>
GpuBuffer<unsigned int>::GpuBuffer() : d_ptr(0), n(0) {}

template <>
void GpuBuffer<unsigned int>::free()
{
	if (d_ptr)
	{
		cudaFree(d_ptr);
		d_ptr = 0;
	}
	n = 0;
}

template <>
GpuBuffer<unsigned int>::~GpuBuffer() { free(); }

template <>
bool GpuBuffer<unsigned int>::allocate(size_t count)
{
	free();
	if (count == 0)
		return true;
	cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_ptr), count * sizeof(unsigned int));
	if (err != cudaSuccess)
	{
		d_ptr = 0;
		n = 0;
		return false;
	}
	n = count;
	return true;
}

template <>
bool GpuBuffer<unsigned int>::upload(const unsigned int* src, size_t count)
{
	if (!d_ptr || !src)
		return false;
	if (count == 0)
		count = n;
	if (count > n)
		return false;
	if (!syncBlockingBufferOp())
		return false;
	cudaError_t err = cudaMemcpy(d_ptr, src, count * sizeof(unsigned int), cudaMemcpyHostToDevice);
	return err == cudaSuccess;
}

template <>
bool GpuBuffer<unsigned int>::uploadAsync(const unsigned int* src, size_t count)
{
	if (!d_ptr || !src)
		return false;
	if (count == 0)
		count = n;
	if (count > n)
		return false;
	cudaError_t err = cudaMemcpyAsync(d_ptr, src, count * sizeof(unsigned int), cudaMemcpyHostToDevice, transferStream());
	return err == cudaSuccess;
}

template <>
bool GpuBuffer<unsigned int>::download(unsigned int* dst, size_t count) const
{
	if (!d_ptr || !dst)
		return false;
	if (count == 0)
		count = n;
	if (count > n)
		return false;
	if (!syncBlockingBufferOp())
		return false;
	cudaError_t err = cudaMemcpy(dst, d_ptr, count * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	return err == cudaSuccess;
}

template <>
bool GpuBuffer<unsigned int>::downloadAsync(unsigned int* dst, size_t count) const
{
	if (!d_ptr || !dst)
		return false;
	if (count == 0)
		count = n;
	if (count > n)
		return false;
	cudaError_t err = cudaMemcpyAsync(dst, d_ptr, count * sizeof(unsigned int), cudaMemcpyDeviceToHost, transferStream());
	return err == cudaSuccess;
}

template <>
bool GpuBuffer<unsigned int>::zero()
{
	if (!d_ptr)
		return false;
	if (!syncBlockingBufferOp())
		return false;
	cudaError_t err = cudaMemset(d_ptr, 0, n * sizeof(unsigned int));
	return err == cudaSuccess;
}

template <>
bool GpuBuffer<unsigned int>::zeroAsync()
{
	if (!d_ptr)
		return false;
	cudaError_t err = cudaMemsetAsync(d_ptr, 0, n * sizeof(unsigned int), computeStream());
	return err == cudaSuccess;
}

// ---- unsigned char specialization ----

template <>
GpuBuffer<unsigned char>::GpuBuffer() : d_ptr(0), n(0) {}

template <>
void GpuBuffer<unsigned char>::free()
{
	if (d_ptr)
	{
		cudaFree(d_ptr);
		d_ptr = 0;
	}
	n = 0;
}

template <>
GpuBuffer<unsigned char>::~GpuBuffer() { free(); }

template <>
bool GpuBuffer<unsigned char>::allocate(size_t count)
{
	free();
	if (count == 0)
		return true;
	cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_ptr), count * sizeof(unsigned char));
	if (err != cudaSuccess)
	{
		d_ptr = 0;
		n = 0;
		return false;
	}
	n = count;
	return true;
}

template <>
bool GpuBuffer<unsigned char>::upload(const unsigned char* src, size_t count)
{
	if (!d_ptr || !src)
		return false;
	if (count == 0)
		count = n;
	if (count > n)
		return false;
	if (!syncBlockingBufferOp())
		return false;
	cudaError_t err = cudaMemcpy(d_ptr, src, count * sizeof(unsigned char), cudaMemcpyHostToDevice);
	return err == cudaSuccess;
}

template <>
bool GpuBuffer<unsigned char>::uploadAsync(const unsigned char* src, size_t count)
{
	if (!d_ptr || !src)
		return false;
	if (count == 0)
		count = n;
	if (count > n)
		return false;
	cudaError_t err = cudaMemcpyAsync(d_ptr, src, count * sizeof(unsigned char), cudaMemcpyHostToDevice, transferStream());
	return err == cudaSuccess;
}

template <>
bool GpuBuffer<unsigned char>::download(unsigned char* dst, size_t count) const
{
	if (!d_ptr || !dst)
		return false;
	if (count == 0)
		count = n;
	if (count > n)
		return false;
	if (!syncBlockingBufferOp())
		return false;
	cudaError_t err = cudaMemcpy(dst, d_ptr, count * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	return err == cudaSuccess;
}

template <>
bool GpuBuffer<unsigned char>::downloadAsync(unsigned char* dst, size_t count) const
{
	if (!d_ptr || !dst)
		return false;
	if (count == 0)
		count = n;
	if (count > n)
		return false;
	cudaError_t err = cudaMemcpyAsync(dst, d_ptr, count * sizeof(unsigned char), cudaMemcpyDeviceToHost, transferStream());
	return err == cudaSuccess;
}

template <>
bool GpuBuffer<unsigned char>::zero()
{
	if (!d_ptr)
		return false;
	if (!syncBlockingBufferOp())
		return false;
	cudaError_t err = cudaMemset(d_ptr, 0, n * sizeof(unsigned char));
	return err == cudaSuccess;
}

template <>
bool GpuBuffer<unsigned char>::zeroAsync()
{
	if (!d_ptr)
		return false;
	cudaError_t err = cudaMemsetAsync(d_ptr, 0, n * sizeof(unsigned char), computeStream());
	return err == cudaSuccess;
}

// ---- int8_t (signed char) specialization ----
// Used for int8-packed Adam optimizer state.

template <>
GpuBuffer<int8_t>::GpuBuffer() : d_ptr(0), n(0) {}

template <>
void GpuBuffer<int8_t>::free()
{
	if (d_ptr) { cudaFree(d_ptr); d_ptr = 0; }
	n = 0;
}

template <>
GpuBuffer<int8_t>::~GpuBuffer() { free(); }

template <>
bool GpuBuffer<int8_t>::allocate(size_t count)
{
	free();
	if (count == 0) return true;
	cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_ptr), count * sizeof(int8_t));
	if (err != cudaSuccess) { d_ptr = 0; n = 0; return false; }
	n = count;
	return true;
}

template <>
bool GpuBuffer<int8_t>::upload(const int8_t* src, size_t count)
{
	if (!d_ptr || !src) return false;
	if (count == 0) count = n;
	if (count > n) return false;
	if (!syncBlockingBufferOp()) return false;
	return cudaMemcpy(d_ptr, src, count * sizeof(int8_t), cudaMemcpyHostToDevice) == cudaSuccess;
}

template <>
bool GpuBuffer<int8_t>::uploadAsync(const int8_t* src, size_t count)
{
	if (!d_ptr || !src) return false;
	if (count == 0) count = n;
	if (count > n) return false;
	return cudaMemcpyAsync(d_ptr, src, count * sizeof(int8_t), cudaMemcpyHostToDevice, transferStream()) == cudaSuccess;
}

template <>
bool GpuBuffer<int8_t>::download(int8_t* dst, size_t count) const
{
	if (!d_ptr || !dst) return false;
	if (count == 0) count = n;
	if (count > n) return false;
	if (!syncBlockingBufferOp()) return false;
	return cudaMemcpy(dst, d_ptr, count * sizeof(int8_t), cudaMemcpyDeviceToHost) == cudaSuccess;
}

template <>
bool GpuBuffer<int8_t>::downloadAsync(int8_t* dst, size_t count) const
{
	if (!d_ptr || !dst) return false;
	if (count == 0) count = n;
	if (count > n) return false;
	return cudaMemcpyAsync(dst, d_ptr, count * sizeof(int8_t), cudaMemcpyDeviceToHost, transferStream()) == cudaSuccess;
}

template <>
bool GpuBuffer<int8_t>::zero()
{
	if (!d_ptr) return false;
	if (!syncBlockingBufferOp()) return false;
	return cudaMemset(d_ptr, 0, n * sizeof(int8_t)) == cudaSuccess;
}

template <>
bool GpuBuffer<int8_t>::zeroAsync()
{
	if (!d_ptr) return false;
	return cudaMemsetAsync(d_ptr, 0, n * sizeof(int8_t), computeStream()) == cudaSuccess;
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
