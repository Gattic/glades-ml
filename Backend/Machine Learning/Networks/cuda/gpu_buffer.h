// GPU device memory buffer for Glades ML.
//
// GpuBuffer manages a typed, contiguous device allocation with upload/download
// and zero-fill operations. It owns its memory and frees on destruction.
#pragma once

#include <cstddef>

#ifdef GLADES_HAVE_CUDA

namespace glades {
namespace gpu {

// RAII wrapper around a device memory allocation.
// T is typically float but could be int, uint16_t, etc.
template <typename T>
class GpuBuffer
{
public:
	GpuBuffer();
	~GpuBuffer();

	// Allocate `count` elements on the device. Frees any existing allocation first.
	// Returns true on success.
	bool allocate(size_t count);

	// Free device memory.
	void free();

	// Upload `count` elements from host `src` to device.
	// If count==0, uses the full allocation size.
	bool upload(const T* src, size_t count = 0);

	// Download `count` elements from device to host `dst`.
	// If count==0, uses the full allocation size.
	bool download(T* dst, size_t count = 0) const;

	// Zero-fill the device buffer.
	bool zero();

	// Accessors.
	T* data() { return d_ptr; }
	const T* data() const { return d_ptr; }
	size_t size() const { return n; }
	size_t bytes() const { return n * sizeof(T); }
	bool allocated() const { return d_ptr != 0; }

private:
	T* d_ptr;
	size_t n;

	// Non-copyable.
	GpuBuffer(const GpuBuffer&);
	GpuBuffer& operator=(const GpuBuffer&);
};

} // namespace gpu
} // namespace glades

#else // !GLADES_HAVE_CUDA

// Stub: compiles but all ops are no-ops / fail gracefully.
namespace glades {
namespace gpu {

template <typename T>
class GpuBuffer
{
public:
	GpuBuffer() : d_ptr(0), n(0) {}
	~GpuBuffer() {}
	bool allocate(size_t) { return false; }
	void free() {}
	bool upload(const T*, size_t = 0) { return false; }
	bool download(T*, size_t = 0) const { return false; }
	bool zero() { return false; }
	T* data() { return 0; }
	const T* data() const { return 0; }
	size_t size() const { return 0; }
	size_t bytes() const { return 0; }
	bool allocated() const { return false; }

private:
	T* d_ptr;
	size_t n;
	GpuBuffer(const GpuBuffer&);
	GpuBuffer& operator=(const GpuBuffer&);
};

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
