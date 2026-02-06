#ifndef GLADES_MAPPED_MATRIX_H
#define GLADES_MAPPED_MATRIX_H
// Memory-mapped float32 matrix for large datasets.
//
// This is a small, self-contained utility to support "columnar / memory-mapped datasets"
// without forcing the rest of the engine to adopt a new container type.
//
// Design goals:
// - C++98 compatible (no <filesystem>, no unordered_map, no std::span).
// - Read-only mmap fast path (zero-copy row access).
// - Simple binary format with a fixed-size header (little-endian).
//
// File format (fixed header, 64 bytes):
//   magic[16]  = "GLADES_GCOL_V1\0"
//   version    = u32 (1)
//   dtype      = u32 (1 == float32)
//   rows       = u64
//   cols       = u64
//   dataOffset = u64 (bytes; must be >= 64 and aligned to 4)
//   reserved0  = u64
//   reserved1  = u64
// Followed by rows*cols float32 values in row-major order.
//
// NOTE: despite the name "columnar", this is a contiguous matrix container.
// It provides the key property we need for production-scale datasets: mmap'd, zero-copy access.

#include <string>

namespace shmea {
template<typename T> class GVector;
typedef GVector<GVector<float> > GMatrix;
}

namespace glades {

class MappedFloatMatrix
{
public:
	MappedFloatMatrix();
	~MappedFloatMatrix();

	// Non-copyable (owns an mmap region).
private:
	MappedFloatMatrix(const MappedFloatMatrix&);
	MappedFloatMatrix& operator=(const MappedFloatMatrix&);

public:
	void close();
	bool isOpen() const;

	// Open an existing file read-only and map it into memory.
	bool openReadOnly(const std::string& path, std::string* errMsg);

	// Create (overwrite) a file and write a dense matrix into it.
	// This does *not* memory-map the output.
	static bool writeFromDense(const std::string& path,
	                           unsigned long long rows,
	                           unsigned long long cols,
	                           const float* rowMajorData,
	                           std::string* errMsg);

	// Convenience: write from shmea::GMatrix.
	static bool writeFromGMatrix(const std::string& path, const shmea::GMatrix& m, std::string* errMsg);

	unsigned long long rows() const;
	unsigned long long cols() const;

	// Return a pointer to the start of row r (row-major). Returns NULL if out of range.
	const float* rowPtr(unsigned long long r) const;

	// Return pointer to full data region (row-major). NULL if not open.
	const float* data() const;

private:
	// Mapped memory region.
	void* mapBase;
	unsigned long long mapBytes;
	int fd;

	// Parsed header.
	unsigned long long nRows;
	unsigned long long nCols;
	unsigned long long dataOff;

	// Cached data pointer.
	const float* dataPtr;
};

} // namespace glades

#endif

