// Lightweight tensor view types (C++98-friendly).
//
// Motivation:
// - Make shape/stride/dtype explicit at API boundaries (especially kernels).
// - Reduce "naked pointer + implicit shape" bugs and simplify bounds checking.
//
// This is intentionally header-only and dependency-light.
//
#pragma once

#include <stddef.h> // size_t

namespace glades {

// Runtime dtype tag (useful for diagnostics/validation).
enum TensorDType
{
	TENSOR_DTYPE_UNKNOWN = 0,
	TENSOR_DTYPE_F32 = 1,
	TENSOR_DTYPE_U16 = 2,
	TENSOR_DTYPE_U8 = 3,
	TENSOR_DTYPE_I32 = 4,
	TENSOR_DTYPE_U32 = 5
};

template <typename T>
inline TensorDType tensor_dtype() { return TENSOR_DTYPE_UNKNOWN; }

template <>
inline TensorDType tensor_dtype<float>() { return TENSOR_DTYPE_F32; }

template <>
inline TensorDType tensor_dtype<const float>() { return TENSOR_DTYPE_F32; }

template <>
inline TensorDType tensor_dtype<unsigned short>() { return TENSOR_DTYPE_U16; }

template <>
inline TensorDType tensor_dtype<const unsigned short>() { return TENSOR_DTYPE_U16; }

template <>
inline TensorDType tensor_dtype<unsigned char>() { return TENSOR_DTYPE_U8; }

template <>
inline TensorDType tensor_dtype<const unsigned char>() { return TENSOR_DTYPE_U8; }

template <>
inline TensorDType tensor_dtype<int>() { return TENSOR_DTYPE_I32; }

template <>
inline TensorDType tensor_dtype<const int>() { return TENSOR_DTYPE_I32; }

template <>
inline TensorDType tensor_dtype<unsigned int>() { return TENSOR_DTYPE_U32; }

template <>
inline TensorDType tensor_dtype<const unsigned int>() { return TENSOR_DTYPE_U32; }

template <typename T>
struct Tensor1DView
{
	T* data;
	size_t size;
	TensorDType dtype;

	Tensor1DView() : data(NULL), size(0u), dtype(tensor_dtype<T>()) {}
	Tensor1DView(T* p, size_t n) : data(p), size(n), dtype(tensor_dtype<T>()) {}

	inline bool ok() const { return (data != NULL && size > 0u); }
	inline T& operator[](size_t i) const { return data[i]; }
};

template <typename T>
struct Tensor2DView
{
	// Row-major view: element(r,c) is at data[r*rowStride + c]
	T* data;
	size_t rows;
	size_t cols;
	size_t rowStride; // in elements
	TensorDType dtype;

	Tensor2DView() : data(NULL), rows(0u), cols(0u), rowStride(0u), dtype(tensor_dtype<T>()) {}
	Tensor2DView(T* p, size_t r, size_t c, size_t rs)
	    : data(p), rows(r), cols(c), rowStride(rs), dtype(tensor_dtype<T>())
	{
	}

	inline bool ok() const { return (data != NULL && rows > 0u && cols > 0u && rowStride >= cols); }

	inline T* row(size_t r) const { return data + r * rowStride; }
	inline T& at(size_t r, size_t c) const { return row(r)[c]; }
};

} // namespace glades

