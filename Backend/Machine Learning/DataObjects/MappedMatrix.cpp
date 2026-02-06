// Memory-mapped float32 matrix implementation (C++98).
#include "MappedMatrix.h"

#include "Backend/Database/GVector.h" // shmea::GMatrix

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <limits>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

static void set_err(std::string* errMsg, const char* msg)
{
	if (errMsg)
		*errMsg = (msg ? std::string(msg) : std::string());
}

static void set_errno_err(std::string* errMsg, const char* prefix)
{
	if (!errMsg)
		return;
	char buf[512];
	const int e = errno;
	std::sprintf(buf, "%s: %s (errno=%d)", (prefix ? prefix : "error"), std::strerror(e), e);
	*errMsg = std::string(buf);
}

static void write_u32_le(unsigned char* out, unsigned int v)
{
	out[0] = static_cast<unsigned char>((v >> 0) & 0xFFu);
	out[1] = static_cast<unsigned char>((v >> 8) & 0xFFu);
	out[2] = static_cast<unsigned char>((v >> 16) & 0xFFu);
	out[3] = static_cast<unsigned char>((v >> 24) & 0xFFu);
}

static void write_u64_le(unsigned char* out, unsigned long long v)
{
	out[0] = static_cast<unsigned char>((v >> 0) & 0xFFull);
	out[1] = static_cast<unsigned char>((v >> 8) & 0xFFull);
	out[2] = static_cast<unsigned char>((v >> 16) & 0xFFull);
	out[3] = static_cast<unsigned char>((v >> 24) & 0xFFull);
	out[4] = static_cast<unsigned char>((v >> 32) & 0xFFull);
	out[5] = static_cast<unsigned char>((v >> 40) & 0xFFull);
	out[6] = static_cast<unsigned char>((v >> 48) & 0xFFull);
	out[7] = static_cast<unsigned char>((v >> 56) & 0xFFull);
}

static unsigned int read_u32_le(const unsigned char* p)
{
	return (static_cast<unsigned int>(p[0]) << 0) |
	       (static_cast<unsigned int>(p[1]) << 8) |
	       (static_cast<unsigned int>(p[2]) << 16) |
	       (static_cast<unsigned int>(p[3]) << 24);
}

static unsigned long long read_u64_le(const unsigned char* p)
{
	return (static_cast<unsigned long long>(p[0]) << 0) |
	       (static_cast<unsigned long long>(p[1]) << 8) |
	       (static_cast<unsigned long long>(p[2]) << 16) |
	       (static_cast<unsigned long long>(p[3]) << 24) |
	       (static_cast<unsigned long long>(p[4]) << 32) |
	       (static_cast<unsigned long long>(p[5]) << 40) |
	       (static_cast<unsigned long long>(p[6]) << 48) |
	       (static_cast<unsigned long long>(p[7]) << 56);
}

static const char* kMagic = "GLADES_GCOL_V1";
static const unsigned int kVersion = 1u;
static const unsigned int kDTypeF32 = 1u;
static const unsigned long long kHeaderBytes = 64ull;

} // namespace

// Link anchor:
// These symbols are intentionally referenced from core engine code so that when glades is built as a
// shared library that links static sub-libraries, the linker does not discard these translation units
// as "unused". Without this, consumers linking against libglades.so would get undefined symbols when
// using MappedFloatMatrix / MappedNumberInput.
extern "C" void glades_link_anchor_mappedmatrix()
{
	// no-op
}

glades::MappedFloatMatrix::MappedFloatMatrix()
    : mapBase(NULL),
      mapBytes(0ull),
      fd(-1),
      nRows(0ull),
      nCols(0ull),
      dataOff(0ull),
      dataPtr(NULL)
{
}

glades::MappedFloatMatrix::~MappedFloatMatrix()
{
	close();
}

void glades::MappedFloatMatrix::close()
{
	if (mapBase && mapBytes > 0ull)
	{
		::munmap(mapBase, static_cast<size_t>(mapBytes));
	}
	mapBase = NULL;
	mapBytes = 0ull;
	dataPtr = NULL;
	nRows = 0ull;
	nCols = 0ull;
	dataOff = 0ull;

	if (fd >= 0)
	{
		::close(fd);
	}
	fd = -1;
}

bool glades::MappedFloatMatrix::isOpen() const
{
	return (mapBase != NULL) && (dataPtr != NULL) && (nRows > 0ull) && (nCols > 0ull);
}

unsigned long long glades::MappedFloatMatrix::rows() const { return nRows; }
unsigned long long glades::MappedFloatMatrix::cols() const { return nCols; }

const float* glades::MappedFloatMatrix::data() const
{
	return dataPtr;
}

const float* glades::MappedFloatMatrix::rowPtr(unsigned long long r) const
{
	if (!isOpen())
		return NULL;
	if (r >= nRows)
		return NULL;
	// Defensive overflow check (should be redundant with open-time validation).
	if (nCols > 0ull && r > (std::numeric_limits<unsigned long long>::max() / nCols))
		return NULL;
	return dataPtr + static_cast<unsigned long long>(r) * static_cast<unsigned long long>(nCols);
}

bool glades::MappedFloatMatrix::openReadOnly(const std::string& path, std::string* errMsg)
{
	close();

	if (path.empty())
	{
		set_err(errMsg, "MappedFloatMatrix::openReadOnly: path is empty");
		return false;
	}

	fd = ::open(path.c_str(), O_RDONLY);
	if (fd < 0)
	{
		set_errno_err(errMsg, "MappedFloatMatrix::openReadOnly: open failed");
		return false;
	}

	struct stat st;
	if (::fstat(fd, &st) != 0)
	{
		set_errno_err(errMsg, "MappedFloatMatrix::openReadOnly: fstat failed");
		close();
		return false;
	}
	if (st.st_size < static_cast<off_t>(kHeaderBytes))
	{
		set_err(errMsg, "MappedFloatMatrix::openReadOnly: file too small");
		close();
		return false;
	}

	mapBytes = static_cast<unsigned long long>(st.st_size);
	mapBase = ::mmap(NULL, static_cast<size_t>(mapBytes), PROT_READ, MAP_PRIVATE, fd, 0);
	if (mapBase == MAP_FAILED)
	{
		mapBase = NULL;
		set_errno_err(errMsg, "MappedFloatMatrix::openReadOnly: mmap failed");
		close();
		return false;
	}

	const unsigned char* hdr = reinterpret_cast<const unsigned char*>(mapBase);

	// magic[16]
	{
		char magicBuf[17];
		std::memset(magicBuf, 0, sizeof(magicBuf));
		std::memcpy(magicBuf, hdr, 16);
		// Compare prefix (magic string is shorter than 16).
		const size_t wantLen = std::strlen(kMagic);
		if (wantLen > 16u || std::memcmp(magicBuf, kMagic, wantLen) != 0)
		{
			set_err(errMsg, "MappedFloatMatrix::openReadOnly: magic mismatch");
			close();
			return false;
		}
	}

	const unsigned int version = read_u32_le(hdr + 16);
	const unsigned int dtype = read_u32_le(hdr + 20);
	nRows = read_u64_le(hdr + 24);
	nCols = read_u64_le(hdr + 32);
	dataOff = read_u64_le(hdr + 40);

	if (version != kVersion)
	{
		set_err(errMsg, "MappedFloatMatrix::openReadOnly: unsupported version");
		close();
		return false;
	}
	if (dtype != kDTypeF32)
	{
		set_err(errMsg, "MappedFloatMatrix::openReadOnly: unsupported dtype (expected float32)");
		close();
		return false;
	}
	if (nRows == 0ull || nCols == 0ull)
	{
		set_err(errMsg, "MappedFloatMatrix::openReadOnly: rows/cols must be > 0");
		close();
		return false;
	}
	if (dataOff < kHeaderBytes || (dataOff % 4ull) != 0ull)
	{
		set_err(errMsg, "MappedFloatMatrix::openReadOnly: invalid dataOffset");
		close();
		return false;
	}

	// Size validation: data must fit in file.
	unsigned long long wantBytes = 0ull;
	// Compute bytes = dataOff + (rows * cols * 4) with overflow checks.
	if (nRows > 0ull && nCols > 0ull)
	{
		unsigned long long elemBytes = 0ull;
		unsigned long long elems = 0ull;
		if (nRows > (std::numeric_limits<unsigned long long>::max() / nCols))
		{
			set_err(errMsg, "MappedFloatMatrix::openReadOnly: element count overflow");
			close();
			return false;
		}
		elems = nRows * nCols;
		if (elems > (std::numeric_limits<unsigned long long>::max() / 4ull))
		{
			set_err(errMsg, "MappedFloatMatrix::openReadOnly: byte size overflow");
			close();
			return false;
		}
		elemBytes = elems * 4ull;
		if (dataOff > (std::numeric_limits<unsigned long long>::max() - elemBytes))
		{
			set_err(errMsg, "MappedFloatMatrix::openReadOnly: byte size overflow");
			close();
			return false;
		}
		wantBytes = dataOff + elemBytes;
	}

	if (wantBytes > mapBytes)
	{
		set_err(errMsg, "MappedFloatMatrix::openReadOnly: file truncated (data exceeds file size)");
		close();
		return false;
	}

	dataPtr = reinterpret_cast<const float*>(reinterpret_cast<const unsigned char*>(mapBase) + dataOff);
	return true;
}

bool glades::MappedFloatMatrix::writeFromDense(const std::string& path,
                                              unsigned long long rows,
                                              unsigned long long cols,
                                              const float* rowMajorData,
                                              std::string* errMsg)
{
	if (path.empty())
	{
		set_err(errMsg, "MappedFloatMatrix::writeFromDense: path is empty");
		return false;
	}
	if (rows == 0ull || cols == 0ull)
	{
		set_err(errMsg, "MappedFloatMatrix::writeFromDense: rows/cols must be > 0");
		return false;
	}
	if (!rowMajorData)
	{
		set_err(errMsg, "MappedFloatMatrix::writeFromDense: data is NULL");
		return false;
	}

	const unsigned long long dataOffset = kHeaderBytes;
	const unsigned long long dataBytes = rows * cols * 4ull;
	const unsigned long long fileBytes = dataOffset + dataBytes;

	const int outFd = ::open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
	if (outFd < 0)
	{
		set_errno_err(errMsg, "MappedFloatMatrix::writeFromDense: open failed");
		return false;
	}

	// Write header.
	unsigned char hdr[kHeaderBytes];
	std::memset(hdr, 0, sizeof(hdr));
	// magic[16]
	std::memcpy(hdr, kMagic, (std::strlen(kMagic) < 16u ? std::strlen(kMagic) : 16u));
	write_u32_le(hdr + 16, kVersion);
	write_u32_le(hdr + 20, kDTypeF32);
	write_u64_le(hdr + 24, rows);
	write_u64_le(hdr + 32, cols);
	write_u64_le(hdr + 40, dataOffset);
	write_u64_le(hdr + 48, 0ull);
	write_u64_le(hdr + 56, 0ull);

	ssize_t wr = ::write(outFd, hdr, static_cast<size_t>(kHeaderBytes));
	if (wr != static_cast<ssize_t>(kHeaderBytes))
	{
		set_errno_err(errMsg, "MappedFloatMatrix::writeFromDense: header write failed");
		::close(outFd);
		return false;
	}

	// Write body in chunks (avoid a single gigantic write on huge datasets).
	const unsigned char* bytes = reinterpret_cast<const unsigned char*>(rowMajorData);
	unsigned long long remaining = dataBytes;
	while (remaining > 0ull)
	{
		const unsigned long long chunk = (remaining > (16ull * 1024ull * 1024ull)) ? (16ull * 1024ull * 1024ull) : remaining;
		const ssize_t w = ::write(outFd, bytes, static_cast<size_t>(chunk));
		if (w <= 0)
		{
			set_errno_err(errMsg, "MappedFloatMatrix::writeFromDense: data write failed");
			::close(outFd);
			return false;
		}
		bytes += static_cast<unsigned long long>(w);
		remaining -= static_cast<unsigned long long>(w);
	}

	// Best-effort size sanity (optional).
	(void)fileBytes;
	::close(outFd);
	return true;
}

bool glades::MappedFloatMatrix::writeFromGMatrix(const std::string& path, const shmea::GMatrix& m, std::string* errMsg)
{
	const unsigned long long R = static_cast<unsigned long long>(m.size());
	if (R == 0ull)
	{
		set_err(errMsg, "MappedFloatMatrix::writeFromGMatrix: matrix has 0 rows");
		return false;
	}
	const unsigned long long C = static_cast<unsigned long long>(m[0].size());
	if (C == 0ull)
	{
		set_err(errMsg, "MappedFloatMatrix::writeFromGMatrix: matrix has 0 cols");
		return false;
	}
	for (unsigned long long r = 0ull; r < R; ++r)
	{
		if (static_cast<unsigned long long>(m[static_cast<unsigned int>(r)].size()) != C)
		{
			set_err(errMsg, "MappedFloatMatrix::writeFromGMatrix: ragged rows are not supported");
			return false;
		}
	}

	// Stream out row-by-row to avoid a second full copy.
	const unsigned long long dataOffset = kHeaderBytes;
	const unsigned long long dataBytes = R * C * 4ull;

	const int outFd = ::open(path.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
	if (outFd < 0)
	{
		set_errno_err(errMsg, "MappedFloatMatrix::writeFromGMatrix: open failed");
		return false;
	}

	unsigned char hdr[kHeaderBytes];
	std::memset(hdr, 0, sizeof(hdr));
	std::memcpy(hdr, kMagic, (std::strlen(kMagic) < 16u ? std::strlen(kMagic) : 16u));
	write_u32_le(hdr + 16, kVersion);
	write_u32_le(hdr + 20, kDTypeF32);
	write_u64_le(hdr + 24, R);
	write_u64_le(hdr + 32, C);
	write_u64_le(hdr + 40, dataOffset);
	write_u64_le(hdr + 48, 0ull);
	write_u64_le(hdr + 56, 0ull);

	ssize_t wr = ::write(outFd, hdr, static_cast<size_t>(kHeaderBytes));
	if (wr != static_cast<ssize_t>(kHeaderBytes))
	{
		set_errno_err(errMsg, "MappedFloatMatrix::writeFromGMatrix: header write failed");
		::close(outFd);
		return false;
	}

	// Write each row's contiguous floats.
	for (unsigned long long r = 0ull; r < R; ++r)
	{
		const shmea::GVector<float>& row = m[static_cast<unsigned int>(r)];
		const unsigned long long rowBytes = C * 4ull;
		const unsigned char* bytes = reinterpret_cast<const unsigned char*>(row.data());
		unsigned long long remaining = rowBytes;
		while (remaining > 0ull)
		{
			const ssize_t w = ::write(outFd, bytes, static_cast<size_t>(remaining));
			if (w <= 0)
			{
				set_errno_err(errMsg, "MappedFloatMatrix::writeFromGMatrix: data write failed");
				::close(outFd);
				return false;
			}
			bytes += static_cast<unsigned long long>(w);
			remaining -= static_cast<unsigned long long>(w);
		}
	}

	(void)dataBytes;
	::close(outFd);
	return true;
}

