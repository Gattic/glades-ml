// Shared internal transformer utilities for inference, generation, and training.
#ifndef _GLADES_TRANSFORMER_COMMON_UTILS_H
#define _GLADES_TRANSFORMER_COMMON_UTILS_H

#include "network.h"

#include <cerrno>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>

namespace glades {
namespace transformer_common {

inline bool checked_mul_size(size_t a, size_t b, size_t& out)
{
	if (a == 0u || b == 0u)
	{
		out = 0u;
		return true;
	}
	if (a > (std::numeric_limits<size_t>::max() / b))
		return false;
	out = a * b;
	return true;
}

inline bool checked_add_size(size_t a, size_t b, size_t& out)
{
	if (a > (std::numeric_limits<size_t>::max() - b))
		return false;
	out = a + b;
	return true;
}

inline bool parse_u64_env(const char* name, unsigned long long& out)
{
	out = 0ULL;
	if (!name || !name[0])
		return false;
	const char* v = ::getenv(name);
	if (!v || !v[0])
		return false;
	errno = 0;
	char* end = NULL;
	const unsigned long long x = ::strtoull(v, &end, 10);
	if (errno != 0 || end == v || (end && *end != '\0'))
		return false;
	out = x;
	return true;
}

inline std::string bytes_to_human(unsigned long long bytes)
{
	std::ostringstream oss;
	const double b = static_cast<double>(bytes);
	const double mib = b / (1024.0 * 1024.0);
	const double gib = mib / 1024.0;
	oss.setf(std::ios::fixed);
	oss.precision(2);
	if (gib >= 1.0)
		oss << gib << " GiB";
	else
		oss << mib << " MiB";
	return oss.str();
}

inline unsigned int resolve_rope_dim(unsigned int dHead, int ropeDimOverride)
{
	unsigned int ropeDim = dHead;
	if (ropeDimOverride > 0)
	{
		const unsigned int rd = static_cast<unsigned int>(ropeDimOverride);
		ropeDim = (rd < ropeDim) ? rd : ropeDim;
	}
	if ((ropeDim % 2u) != 0u)
		ropeDim -= 1u;
	return ropeDim;
}

inline unsigned int resolve_kv_head(unsigned int queryHead, unsigned int nHeads, unsigned int nKVHeads)
{
	if (nKVHeads == nHeads)
		return queryHead;
	const unsigned int groupSize = (nKVHeads > 0u ? (nHeads / nKVHeads) : 0u);
	return (groupSize > 0u ? (queryHead / groupSize) : 0u);
}

struct ScopedTimerMs
{
	const glades::NNetwork* net;
	double* acc;
	int64_t t0ms;

	explicit ScopedTimerMs(const glades::NNetwork* n, bool enabled, double* outAcc)
	    : net(n), acc((enabled && outAcc && n) ? outAcc : NULL), t0ms(0)
	{
		if (acc)
			t0ms = net->getCurrentTimeMilliseconds();
	}

	~ScopedTimerMs()
	{
		if (!acc)
			return;
		const int64_t t1ms = net->getCurrentTimeMilliseconds();
		*acc += static_cast<double>(t1ms - t0ms);
	}
};

} // namespace transformer_common
} // namespace glades

#endif
