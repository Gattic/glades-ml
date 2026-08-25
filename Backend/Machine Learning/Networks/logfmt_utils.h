// Shared logfmt key-value formatting utilities.
// Header-only; safe to include from any translation unit.
#pragma once

#include <sstream>
#include <string>

namespace glades {
namespace logfmt {

inline void append_logfmt_escaped(std::ostringstream& oss, const std::string& v)
{
	oss << '"';
	for (size_t i = 0; i < v.size(); ++i)
	{
		const char c = v[i];
		if (c == '\\' || c == '"')
			oss << '\\' << c;
		else if (c == '\n')
			oss << "\\n";
		else if (c == '\r')
			oss << "\\r";
		else if (c == '\t')
			oss << "\\t";
		else
			oss << c;
	}
	oss << '"';
}

inline void append_logfmt_kv(std::ostringstream& oss, const char* k, const std::string& v)
{
	oss << ' ' << k << '=';
	bool needQuote = false;
	for (size_t i = 0; i < v.size(); ++i)
	{
		const char c = v[i];
		if (c == ' ' || c == '=' || c == '"' || c == '\\' || c == '\n' || c == '\r' || c == '\t')
		{
			needQuote = true;
			break;
		}
	}
	if (!needQuote)
	{
		oss << v;
		return;
	}
	append_logfmt_escaped(oss, v);
}

inline void append_logfmt_kv(std::ostringstream& oss, const char* k, int v) { oss << ' ' << k << '=' << v; }
inline void append_logfmt_kv(std::ostringstream& oss, const char* k, unsigned int v) { oss << ' ' << k << '=' << v; }
inline void append_logfmt_kv(std::ostringstream& oss, const char* k, unsigned long v) { oss << ' ' << k << '=' << v; }
inline void append_logfmt_kv(std::ostringstream& oss, const char* k, unsigned long long v) { oss << ' ' << k << '=' << v; }
inline void append_logfmt_kv(std::ostringstream& oss, const char* k, float v) { oss << ' ' << k << '=' << v; }
inline void append_logfmt_kv(std::ostringstream& oss, const char* k, double v) { oss << ' ' << k << '=' << v; }
inline void append_logfmt_kv(std::ostringstream& oss, const char* k, bool v) { oss << ' ' << k << '=' << (v ? 1 : 0); }

} // namespace logfmt
} // namespace glades
