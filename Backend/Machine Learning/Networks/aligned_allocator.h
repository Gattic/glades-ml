// Aligned allocator for std::vector to improve SIMD performance.
//
// This is intentionally tiny and dependency-free (posix_memalign on POSIX).
// It is safe to use with trivial types (float, uint16_t, unsigned char, etc.).
//
// Notes:
// - Alignment must be a power of two and at least alignof(T).
// - On allocation failure, std::bad_alloc is thrown (as required by Allocator).
//
#ifndef GLADES_ALIGNED_ALLOCATOR_H
#define GLADES_ALIGNED_ALLOCATOR_H

#include <cstddef>
#include <cstdlib>
#include <stdlib.h>
#include <new>

namespace glades {

template <typename T, std::size_t Alignment>
class AlignedAllocator
{
public:
	// C++03-style allocator typedefs (still accepted in C++11+).
	typedef T value_type;
	typedef T* pointer;
	typedef const T* const_pointer;
	typedef T& reference;
	typedef const T& const_reference;
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;

	AlignedAllocator() {}

	template <class U>
	AlignedAllocator(const AlignedAllocator<U, Alignment>&)
	{
	}

	pointer allocate(size_type n, const void* /*hint*/ = NULL)
	{
		if (n == 0u)
			return static_cast<pointer>(NULL);
#if __cplusplus >= 201103L
		static_assert((Alignment & (Alignment - 1u)) == 0u, "Alignment must be power of two");
		static_assert(Alignment >= alignof(T), "Alignment must be >= alignof(T)");
#endif

		void* p = NULL;
		const size_type bytes = n * sizeof(T);
		// posix_memalign requires alignment to be a power of two and multiple of sizeof(void*).
		const size_type wantAlign = (Alignment < sizeof(void*)) ? sizeof(void*) : Alignment;
		const int rc = ::posix_memalign(&p, wantAlign, bytes);
		if (rc != 0 || !p)
			throw std::bad_alloc();
		return static_cast<pointer>(p);
	}

	void deallocate(pointer p, size_type /*n*/)
	{
		::free(static_cast<void*>(p));
	}

	// C++03 construct/destroy (no-op for trivials but required by some libstdc++ modes).
	void construct(pointer p, const_reference v)
	{
		::new (static_cast<void*>(p)) value_type(v);
	}

	void destroy(pointer p)
	{
		p->~value_type();
	}

	size_type max_size() const
	{
		return static_cast<size_type>(-1) / sizeof(value_type);
	}

	template <class U>
	struct rebind
	{
		typedef AlignedAllocator<U, Alignment> other;
	};
};

template <class T, std::size_t A, class U, std::size_t B>
inline bool operator==(const AlignedAllocator<T, A>&, const AlignedAllocator<U, B>&)
{
	return A == B;
}

template <class T, std::size_t A, class U, std::size_t B>
inline bool operator!=(const AlignedAllocator<T, A>& a, const AlignedAllocator<U, B>& b)
{
	return !(a == b);
}

} // namespace glades

#endif

