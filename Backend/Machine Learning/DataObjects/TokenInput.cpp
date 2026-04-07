// Copyright 2026
//
// Minimal token-id DataInput for language modeling.

#include "TokenInput.h"

#include "Backend/Database/GTable.h"

#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <limits>

using namespace glades;

namespace {
// Thread-local storage for row-view scratch scalars.
//
// This codebase may be compiled in a pre-C++11 mode, so we cannot rely on `thread_local`.
// Prefer standard TLS when available; otherwise fall back to compiler TLS.
#if defined(__cplusplus) && (__cplusplus >= 201103L)
#define GLADES_THREAD_LOCAL thread_local
#elif defined(_MSC_VER)
#define GLADES_THREAD_LOCAL __declspec(thread)
#elif defined(__GNUC__) || defined(__clang__)
#define GLADES_THREAD_LOCAL __thread
#else
#error "TokenInput row-view APIs require thread-local storage support"
#endif

static inline std::string to_std_string(const shmea::GString& s)
{
	return std::string(s.c_str());
}

static inline bool path_ends_with(const std::string& s, const std::string& suf)
{
	if (s.size() < suf.size())
		return false;
	return s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

static inline bool parse_strict_int_token(const std::string& s, int& outTok)
{
	if (s.empty())
		return false;

	char* end = NULL;
	errno = 0;
	const long long v = strtoll(s.c_str(), &end, 10);
	if (end == s.c_str() || errno != 0)
		return false;
	while (*end != '\0' && std::isspace(static_cast<unsigned char>(*end)))
		++end;
	if (*end != '\0')
		return false;
	if (v < static_cast<long long>(std::numeric_limits<int>::min()) ||
	    v > static_cast<long long>(std::numeric_limits<int>::max()))
		return false;

	outTok = static_cast<int>(v);
	return true;
}

static bool cell_to_int_token(const shmea::GType& cell, int& outTok)
{
	outTok = 0;
	switch (cell.getType())
	{
	case shmea::GType::CHAR_TYPE: outTok = static_cast<int>(cell.getChar()); return true;
	case shmea::GType::SHORT_TYPE: outTok = static_cast<int>(cell.getShort()); return true;
	case shmea::GType::INT_TYPE: outTok = cell.getInt(); return true;
	case shmea::GType::LONG_TYPE:
	{
		const long long v = cell.getLong();
		if (v < static_cast<long long>(std::numeric_limits<int>::min()) || v > static_cast<long long>(std::numeric_limits<int>::max()))
			return false;
		outTok = static_cast<int>(v);
		return true;
	}
	case shmea::GType::BOOLEAN_TYPE:
		return false;
	case shmea::GType::FLOAT_TYPE:
	{
		const double v = static_cast<double>(cell.getFloat());
		if (!std::isfinite(v))
			return false;
		if (std::floor(v) != v)
			return false;
		if (v < static_cast<double>(std::numeric_limits<int>::min()) || v > static_cast<double>(std::numeric_limits<int>::max()))
			return false;
		outTok = static_cast<int>(v);
		return true;
	}
	case shmea::GType::DOUBLE_TYPE:
	{
		const double v = cell.getDouble();
		if (!std::isfinite(v))
			return false;
		if (std::floor(v) != v)
			return false;
		if (v < static_cast<double>(std::numeric_limits<int>::min()) || v > static_cast<double>(std::numeric_limits<int>::max()))
			return false;
		outTok = static_cast<int>(v);
		return true;
	}
	case shmea::GType::STRING_TYPE:
	default:
	{
		return parse_strict_int_token(std::string(cell.c_str()), outTok);
	}
	}
}
} // namespace

glades::TokenInput::TokenInput()
    : loaded(false),
      padTokenId(-1),
      lastImportStatus(glades::NNetworkStatus::OK, std::string())
{
	trainTok.clear();
	trainNextTok.clear();
	testTok.clear();
	testNextTok.clear();
}

glades::TokenInput::~TokenInput()
{
	clearLoadedData();
}

void glades::TokenInput::clearLoadedData()
{
	loaded = false;
	trainTok.clear();
	trainNextTok.clear();
	testTok.clear();
	testNextTok.clear();
	clearTrainSequences();
	clearTestSequences();
}

glades::NNetworkStatus glades::TokenInput::loadTokenFile(const std::string& path,
                                                         std::vector<int>& outTok,
                                                         std::vector<int>& outNext,
                                                         std::vector<SequenceSpan>& outSeq,
                                                         unsigned int* outLineCount)
{
	outTok.clear();
	outNext.clear();
	outSeq.clear();
	if (outLineCount)
		*outLineCount = 0u;

	if (path.empty())
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_ARGUMENT, "TokenInput::loadTokenFile: path is empty");

	std::ifstream in(path.c_str());
	if (!in)
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_ARGUMENT, "TokenInput::loadTokenFile: unable to open file");

	std::string line;
	unsigned int cursor = 0u;
	unsigned int lineNo = 0u;
	while (std::getline(in, line))
	{
		++lineNo;
		if (outLineCount)
			*outLineCount = lineNo;

		// Allow empty lines (skip).
		std::istringstream iss(line);
		std::vector<int> toks;

		// Parse tokens as strings so invalid cells don't silently truncate the line.
		std::string tokStr;
		while (iss >> tokStr)
		{
			// Token IDs are modeled as signed ints throughout the transformer LM stack.
			// Reject negative values and values that cannot round-trip through int.
			char* end = NULL;
			errno = 0;
			const long long v = strtoll(tokStr.c_str(), &end, 10);
			if (end == tokStr.c_str() || *end != '\0' || errno != 0)
			{
				std::ostringstream oss;
				oss << "TokenInput::loadTokenFile: invalid token at line " << lineNo;
				return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_ARGUMENT, oss.str());
			}
			if (v < 0ll || v > static_cast<long long>(std::numeric_limits<int>::max()))
			{
				std::ostringstream oss;
				oss << "TokenInput::loadTokenFile: token out of range at line " << lineNo;
				return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_ARGUMENT, oss.str());
			}
			toks.push_back(static_cast<int>(v));
		}
		if (toks.empty())
			continue;

		// Emit language-model rows:
		// - For each token, the target is the next token in the same sequence.
		// - For the final token, there is no "next token". If padTokenId >= 0, we emit an
		//   explicit pad/ignore target; otherwise we DO NOT emit the final timestep.
		//
		// Rationale:
		// - Token IDs are first-class ints, but many downstream call sites historically cast
		//   expected rows to unsigned token IDs; emitting a negative pad/ignore id is not safe.
		const unsigned int len = static_cast<unsigned int>(toks.size());
		const bool usePad = (padTokenId >= 0);
		const unsigned int emitLen = (usePad ? len : (len > 0u ? (len - 1u) : 0u));
		if (emitLen == 0u)
			continue;

		const unsigned int start = cursor;
		outTok.reserve(outTok.size() + emitLen);
		outNext.reserve(outNext.size() + emitLen);
		for (unsigned int i = 0; i < emitLen; ++i)
		{
			outTok.push_back(toks[i]);
			if (i + 1u < len)
				outNext.push_back(toks[i + 1u]);
			else
				outNext.push_back(padTokenId); // only reachable when usePad==true
		}

		outSeq.push_back(SequenceSpan(start, emitLen));
		cursor += emitLen;
	}

	if (outTok.empty())
		return glades::NNetworkStatus(glades::NNetworkStatus::EMPTY_DATA, "TokenInput::loadTokenFile: no tokens found");
	if (outTok.size() != outNext.size())
		return glades::NNetworkStatus(glades::NNetworkStatus::INTERNAL_ERROR, "TokenInput::loadTokenFile: internal size mismatch (tok != next)");
	return glades::NNetworkStatus(glades::NNetworkStatus::OK, std::string());
}

void glades::TokenInput::import(shmea::GString fname, int /*standardizeFlag*/)
{
	const std::string p = to_std_string(fname);
	if (p.empty())
	{
		clearLoadedData();
		lastImportStatus = glades::NNetworkStatus(glades::NNetworkStatus::INVALID_ARGUMENT, "TokenInput::import: empty path");
		return;
	}

	std::vector<int> newTrainTok;
	std::vector<int> newTrainNextTok;
	std::vector<int> newTestTok;
	std::vector<int> newTestNextTok;
	// Directory semantics: if path ends with '/', accept "train.tok" + "test.tok".
	// File semantics: treat as train only (mirror to test).
	std::vector<SequenceSpan> trSeq;
	std::vector<SequenceSpan> teSeq;

	glades::NNetworkStatus stTrain(glades::NNetworkStatus::OK, std::string());
	if (path_ends_with(p, "/"))
	{
		stTrain = loadTokenFile(p + "train.tok", newTrainTok, newTrainNextTok, trSeq);
		// Test split is optional; ignore failures and mirror train->test below.
		(void)loadTokenFile(p + "test.tok", newTestTok, newTestNextTok, teSeq);
	}
	else
	{
		stTrain = loadTokenFile(p, newTrainTok, newTrainNextTok, trSeq);
	}

	if (!stTrain.ok())
	{
		clearLoadedData();
		lastImportStatus = stTrain;
		return;
	}

	// Default: if no explicit test split loaded, mirror train->test.
	if (newTestTok.empty() && !newTrainTok.empty())
	{
		newTestTok = newTrainTok;
		newTestNextTok = newTrainNextTok;
		teSeq = trSeq;
	}

	trainTok.swap(newTrainTok);
	trainNextTok.swap(newTrainNextTok);
	testTok.swap(newTestTok);
	testNextTok.swap(newTestNextTok);
	// Install sequence spans.
	if (!setTrainSequences(trSeq) || !setTestSequences(teSeq))
	{
		clearLoadedData();
		lastImportStatus = glades::NNetworkStatus(glades::NNetworkStatus::INTERNAL_ERROR,
		                                         "TokenInput::import: invalid sequence spans after load");
		return;
	}

	loaded = true;
	lastImportStatus = glades::NNetworkStatus(glades::NNetworkStatus::OK, std::string());
}

void glades::TokenInput::import(const shmea::GTable& t, int /*standardizeFlag*/)
{
	// Interpret a table as token IDs:
	// - If cols==1: each row is one timestep; the whole table is a single sequence.
	// - If cols>1: each table row is treated as one independent sequence spanning all columns.
	//
	// Cells are converted to int token IDs (string cells are parsed as integers).
	const unsigned int R = t.numberOfRows();
	const unsigned int C = t.numberOfCols();
	if (R == 0u || C == 0u)
	{
		clearLoadedData();
		lastImportStatus = glades::NNetworkStatus(glades::NNetworkStatus::EMPTY_DATA, "TokenInput::import(table): empty table");
		return;
	}

	std::vector<int> newTrainTok;
	std::vector<int> newTrainNextTok;
	std::vector<SequenceSpan> trSeq;

	if (C == 1u)
	{
		// One sequence spanning all rows.
		// When padTokenId < 0, we omit the final timestep to avoid negative expected token ids.
		const bool usePad = (padTokenId >= 0);
		const unsigned int emitLen = (usePad ? R : (R > 0u ? (R - 1u) : 0u));
		if (emitLen == 0u)
		{
			clearLoadedData();
			lastImportStatus = glades::NNetworkStatus(glades::NNetworkStatus::EMPTY_DATA, "TokenInput::import(table): empty after LM pair emission");
			return;
		}

		std::vector<int> toks;
		toks.reserve(R);
		for (unsigned int r = 0; r < R; ++r)
		{
			int tok = 0;
			if (!cell_to_int_token(t.getCell(r, 0u), tok))
			{
				clearLoadedData();
				lastImportStatus = glades::NNetworkStatus(glades::NNetworkStatus::INVALID_ARGUMENT, "TokenInput::import(table): invalid token cell");
				return;
			}
			if (tok < 0)
			{
				clearLoadedData();
				lastImportStatus = glades::NNetworkStatus(glades::NNetworkStatus::INVALID_ARGUMENT, "TokenInput::import(table): negative token id not allowed");
				return;
			}
			toks.push_back(tok);
		}

		newTrainTok.reserve(emitLen);
		newTrainNextTok.reserve(emitLen);
		for (unsigned int r = 0; r < emitLen; ++r)
		{
			newTrainTok.push_back(toks[r]);
			if (r + 1u < R)
				newTrainNextTok.push_back(toks[r + 1u]);
			else
				newTrainNextTok.push_back(padTokenId); // only when usePad==true
		}
		trSeq.push_back(SequenceSpan(0u, emitLen));
	}
	else
	{
		// Row-per-sequence.
		unsigned int cursor = 0u;
		for (unsigned int r = 0; r < R; ++r)
		{
			const unsigned int len = C;
			const bool usePad = (padTokenId >= 0);
			const unsigned int emitLen = (usePad ? len : (len > 0u ? (len - 1u) : 0u));
			if (emitLen == 0u)
				continue;

			std::vector<int> toks;
			toks.reserve(len);
			for (unsigned int c = 0; c < C; ++c)
			{
				int tok = 0;
				if (!cell_to_int_token(t.getCell(r, c), tok))
				{
					clearLoadedData();
					lastImportStatus = glades::NNetworkStatus(glades::NNetworkStatus::INVALID_ARGUMENT, "TokenInput::import(table): invalid token cell");
					return;
				}
				if (tok < 0)
				{
					clearLoadedData();
					lastImportStatus = glades::NNetworkStatus(glades::NNetworkStatus::INVALID_ARGUMENT, "TokenInput::import(table): negative token id not allowed");
					return;
				}
				toks.push_back(tok);
			}

			trSeq.push_back(SequenceSpan(cursor, emitLen));
			newTrainTok.reserve(newTrainTok.size() + emitLen);
			newTrainNextTok.reserve(newTrainNextTok.size() + emitLen);
			for (unsigned int c = 0; c < emitLen; ++c)
			{
				newTrainTok.push_back(toks[c]);
				if (c + 1u < len)
					newTrainNextTok.push_back(toks[c + 1u]);
				else
					newTrainNextTok.push_back(padTokenId); // only when usePad==true
			}
			cursor += emitLen;
		}
	}

	if (newTrainTok.empty() || newTrainNextTok.empty() || newTrainTok.size() != newTrainNextTok.size())
	{
		clearLoadedData();
		lastImportStatus = glades::NNetworkStatus(glades::NNetworkStatus::EMPTY_DATA, "TokenInput::import(table): no LM pairs produced");
		return;
	}

	// Mirror train->test by default (same semantics as file import when no explicit test split exists).
	trainTok.swap(newTrainTok);
	trainNextTok.swap(newTrainNextTok);
	testTok = trainTok;
	testNextTok = trainNextTok;
	if (!setTrainSequences(trSeq) || !setTestSequences(trSeq))
	{
		clearLoadedData();
		lastImportStatus = glades::NNetworkStatus(glades::NNetworkStatus::INTERNAL_ERROR,
		                                         "TokenInput::import(table): invalid sequence spans after load");
		return;
	}
	loaded = true;
	lastImportStatus = glades::NNetworkStatus(glades::NNetworkStatus::OK, std::string());
}

shmea::GVector<float> glades::TokenInput::getTrainRow(unsigned int index) const
{
	if (index >= trainTok.size())
		return shmea::GVector<float>();
	shmea::GVector<float> v(1, 0.0f);
	v[0] = static_cast<float>(trainTok[index]);
	return v;
}

shmea::GVector<float> glades::TokenInput::getTrainExpectedRow(unsigned int index) const
{
	if (index >= trainNextTok.size())
		return shmea::GVector<float>();
	shmea::GVector<float> v(1, 0.0f);
	v[0] = static_cast<float>(trainNextTok[index]);
	return v;
}

shmea::GVector<float> glades::TokenInput::getTestRow(unsigned int index) const
{
	if (index >= testTok.size())
		return shmea::GVector<float>();
	shmea::GVector<float> v(1, 0.0f);
	v[0] = static_cast<float>(testTok[index]);
	return v;
}

shmea::GVector<float> glades::TokenInput::getTestExpectedRow(unsigned int index) const
{
	if (index >= testNextTok.size())
		return shmea::GVector<float>();
	shmea::GVector<float> v(1, 0.0f);
	v[0] = static_cast<float>(testNextTok[index]);
	return v;
}

bool glades::TokenInput::getTrainRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
{
	outData = NULL;
	outSize = 0u;
	if (index >= trainTok.size())
		return false;
	static GLADES_THREAD_LOCAL float tlsTok;
	tlsTok = static_cast<float>(trainTok[index]);
	outData = &tlsTok;
	outSize = 1u;
	return true;
}

bool glades::TokenInput::getTrainExpectedRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
{
	outData = NULL;
	outSize = 0u;
	if (index >= trainNextTok.size())
		return false;
	static GLADES_THREAD_LOCAL float tlsNextTok;
	tlsNextTok = static_cast<float>(trainNextTok[index]);
	outData = &tlsNextTok;
	outSize = 1u;
	return true;
}

bool glades::TokenInput::getTestRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
{
	outData = NULL;
	outSize = 0u;
	if (index >= testTok.size())
		return false;
	static GLADES_THREAD_LOCAL float tlsTok;
	tlsTok = static_cast<float>(testTok[index]);
	outData = &tlsTok;
	outSize = 1u;
	return true;
}

bool glades::TokenInput::getTestExpectedRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
{
	outData = NULL;
	outSize = 0u;
	if (index >= testNextTok.size())
		return false;
	static GLADES_THREAD_LOCAL float tlsNextTok;
	tlsNextTok = static_cast<float>(testNextTok[index]);
	outData = &tlsNextTok;
	outSize = 1u;
	return true;
}

bool glades::TokenInput::getTrainTokenId(unsigned int index, int& outTokenId) const
{
	outTokenId = 0;
	if (index >= trainTok.size())
		return false;
	outTokenId = trainTok[index];
	return true;
}

bool glades::TokenInput::getTrainExpectedTokenId(unsigned int index, int& outTokenId) const
{
	outTokenId = 0;
	if (index >= trainNextTok.size())
		return false;
	outTokenId = trainNextTok[index];
	return true;
}

bool glades::TokenInput::getTestTokenId(unsigned int index, int& outTokenId) const
{
	outTokenId = 0;
	if (index >= testTok.size())
		return false;
	outTokenId = testTok[index];
	return true;
}

bool glades::TokenInput::getTestExpectedTokenId(unsigned int index, int& outTokenId) const
{
	outTokenId = 0;
	if (index >= testNextTok.size())
		return false;
	outTokenId = testNextTok[index];
	return true;
}

unsigned int glades::TokenInput::getTrainSize() const
{
	return static_cast<unsigned int>(trainTok.size());
}

unsigned int glades::TokenInput::getTestSize() const
{
	return static_cast<unsigned int>(testTok.size());
}

unsigned int glades::TokenInput::getFeatureCount() const
{
	return 1u;
}
