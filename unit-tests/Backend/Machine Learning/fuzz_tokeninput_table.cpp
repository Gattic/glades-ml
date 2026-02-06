// libFuzzer harness (opt-in build) for TokenInput table import.
//
// This file is NOT compiled by default in the normal unit-tests build.
//
// Build example (clang):
//   clang++ -std=c++98 -O1 -g -fsanitize=fuzzer,address,undefined \
//     -I. -I./include -I./Backend -I./services \
//     unit-tests/Backend/Machine\ Learning/fuzz_tokeninput_table.cpp \
//     -lglades -lshmea -o fuzz_tokeninput_table
//
// Then run:
//   ./fuzz_tokeninput_table -runs=100000 corpus_dir/
//
// NOTE: This harness is intentionally pure in-memory (no filesystem writes).

#include "../../../Backend/Machine Learning/DataObjects/TokenInput.h"
#include "Backend/Database/GTable.h"
#include "Backend/Database/GList.h"

#include <sstream>
#include <string>
#include <vector>
#include <stdint.h>

namespace {
static inline uint64_t read_u64(const uint8_t* data, size_t& off, size_t n)
{
	uint64_t v = 0;
	for (unsigned int i = 0; i < 8; ++i)
	{
		if (off >= n) break;
		v |= (static_cast<uint64_t>(data[off++]) << (8u * i));
	}
	return v;
}
static inline unsigned int read_u32(const uint8_t* data, size_t& off, size_t n)
{
	return static_cast<unsigned int>(read_u64(data, off, n) & 0xFFFFFFFFu);
}
static inline bool read_bool(const uint8_t* data, size_t& off, size_t n)
{
	if (off >= n) return false;
	return (data[off++] & 1u) != 0u;
}
static std::string itos(unsigned int x)
{
	std::ostringstream oss;
	oss << x;
	return oss.str();
}
} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
	if (!data || size < 4u)
		return 0;

	size_t off = 0u;
	const unsigned int rows = (read_u32(data, off, size) % 32u);
	const unsigned int cols = 1u + (read_u32(data, off, size) % 16u);
	const bool usePad = read_bool(data, off, size);
	const int pad = usePad ? static_cast<int>(read_u32(data, off, size) % 128u) : -1;

	shmea::GVector<shmea::GString> headers;
	for (unsigned int c = 0u; c < cols; ++c)
		headers.push_back((std::string("c") + itos(c)).c_str());
	shmea::GTable tbl(',', headers);

	for (unsigned int r = 0u; r < rows; ++r)
	{
		shmea::GList row;
		for (unsigned int c = 0u; c < cols; ++c)
		{
			const unsigned int kind = (off < size) ? (data[off++] % 8u) : 0u;
			if (kind == 0u)
				row.addInt(static_cast<int>((off < size) ? data[off++] : 0u));
			else if (kind == 1u)
				row.addInt(-static_cast<int>((off < size) ? (data[off++] % 4u) : 1u));
			else if (kind == 2u)
				row.addFloat(static_cast<float>((off < size) ? data[off++] : 0u));
			else if (kind == 3u)
				row.addString("not_a_number");
			else if (kind == 4u)
				row.addString("2147483647");
			else if (kind == 5u)
				row.addString("9999999999999999999999999");
			else if (kind == 6u)
				row.addBoolean(read_bool(data, off, size));
			else
				row.addString("0");
		}
		tbl.addRow(row);
	}

	glades::TokenInput ti;
	ti.setPadTokenId(pad);
	ti.import(tbl);

	// Touch a few getters to exercise view/token-id paths.
	if (ti.loadedOk() && ti.getTrainSize() > 0u)
	{
		const unsigned int idx = (ti.getTrainSize() > 1u) ? (read_u32(data, off, size) % ti.getTrainSize()) : 0u;
		int tok = 0;
		(void)ti.getTrainTokenId(idx, tok);
		const float* p = 0;
		unsigned int psz = 0u;
		(void)ti.getTrainRowView(idx, p, psz);
	}

	return 0;
}

