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
#include <sys/stat.h>

using namespace glades;

namespace {

static inline std::string to_std_string(const shmea::GString& s)
{
	return std::string(s.c_str());
}

static inline bool path_is_directory(const std::string& p)
{
	struct stat st;
	if (::stat(p.c_str(), &st) != 0)
		return false;
	return S_ISDIR(st.st_mode);
}

static inline std::string join_path(const std::string& base, const char* leaf)
{
	if (base.empty())
		return std::string(leaf);
	if (base[base.size() - 1] == '/')
		return base + leaf;
	return base + "/" + leaf;
}

static inline bool parse_non_negative_int_token(const std::string& s, int& outTok)
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
	if (v < 0ll)
		return false;

	outTok = static_cast<int>(v);
	return true;
}

static bool cell_to_int_token(const shmea::GType& cell, int& outTok)
{
	outTok = 0;
	switch (cell.getType())
	{
	case shmea::GType::CHAR_TYPE:
		outTok = static_cast<int>(cell.getChar());
		return outTok >= 0;
	case shmea::GType::SHORT_TYPE:
		outTok = static_cast<int>(cell.getShort());
		return outTok >= 0;
	case shmea::GType::INT_TYPE:
		outTok = cell.getInt();
		return outTok >= 0;
	case shmea::GType::LONG_TYPE:
	{
		const long long v = cell.getLong();
		if (v < static_cast<long long>(std::numeric_limits<int>::min()) || v > static_cast<long long>(std::numeric_limits<int>::max()))
			return false;
		if (v < 0ll)
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
		if (v < 0.0)
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
		if (v < 0.0)
			return false;
		outTok = static_cast<int>(v);
		return true;
	}
	case shmea::GType::STRING_TYPE:
	default:
	{
		return parse_non_negative_int_token(std::string(cell.c_str()), outTok);
	}
	}
}

static bool append_token_id_sequence(const std::vector<int>& toks,
                                     int padTokenId,
                                     std::vector<int>& outTok,
                                     std::vector<int>& outNext,
                                     std::vector<DataInput::SequenceSpan>& outSeq,
                                     unsigned int& cursor)
{
	if (toks.empty())
		return true;

	const unsigned int len = static_cast<unsigned int>(toks.size());
	const bool usePad = (padTokenId >= 0);
	const unsigned int emitLen = usePad ? len : (len > 0u ? (len - 1u) : 0u);
	if (emitLen == 0u)
		return true;

	const unsigned int start = cursor;
	outTok.reserve(outTok.size() + emitLen);
	outNext.reserve(outNext.size() + emitLen);
	for (unsigned int i = 0; i < emitLen; ++i)
	{
		outTok.push_back(toks[i]);
		if (i + 1u < len)
			outNext.push_back(toks[i + 1u]);
		else
			outNext.push_back(padTokenId);
	}

	outSeq.push_back(DataInput::SequenceSpan(start, emitLen));
	cursor += emitLen;
	return true;
}
} // namespace

glades::TokenInput::TokenInput()
    : loaded(false),
      padTokenId(-1),
      mirrorTrainToTestOnImplicitSplit(false),
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

		// Emit one language-model sequence span per non-empty line.
		append_token_id_sequence(toks, padTokenId, outTok, outNext, outSeq, cursor);
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
	const bool isDirectoryImport = path_is_directory(p);

	std::vector<int> newTrainTok;
	std::vector<int> newTrainNextTok;
	std::vector<int> newTestTok;
	std::vector<int> newTestNextTok;
	// Directory semantics: require both "train.tok" and "test.tok".
	// File semantics: treat the file as train-only unless mirroring was explicitly requested.
	std::vector<SequenceSpan> trSeq;
	std::vector<SequenceSpan> teSeq;

	glades::NNetworkStatus stTrain(glades::NNetworkStatus::OK, std::string());
	glades::NNetworkStatus stTest(glades::NNetworkStatus::OK, std::string());
	if (isDirectoryImport)
	{
		stTrain = loadTokenFile(join_path(p, "train.tok"), newTrainTok, newTrainNextTok, trSeq);
		stTest = loadTokenFile(join_path(p, "test.tok"), newTestTok, newTestNextTok, teSeq);
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
	if (isDirectoryImport && !stTest.ok())
	{
		clearLoadedData();
		lastImportStatus = stTest;
		return;
	}

	// Optional compatibility mode for single-input imports.
	if (!isDirectoryImport && mirrorTrainToTestOnImplicitSplit && newTestTok.empty() && !newTrainTok.empty())
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
	std::vector<SequenceSpan> teSeq;

	if (C == 1u)
	{
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
			toks.push_back(tok);
		}

		// One table column becomes one sequence whose timesteps are the rows.
		unsigned int cursor = 0u;
		append_token_id_sequence(toks, padTokenId, newTrainTok, newTrainNextTok, trSeq, cursor);
	}
	else
	{
		// Each table row becomes one independent sequence.
		unsigned int cursor = 0u;
		for (unsigned int r = 0; r < R; ++r)
		{
			std::vector<int> toks;
			toks.reserve(C);
			for (unsigned int c = 0; c < C; ++c)
			{
				int tok = 0;
				if (!cell_to_int_token(t.getCell(r, c), tok))
				{
					clearLoadedData();
					lastImportStatus = glades::NNetworkStatus(glades::NNetworkStatus::INVALID_ARGUMENT, "TokenInput::import(table): invalid token cell");
					return;
				}
				toks.push_back(tok);
			}

			append_token_id_sequence(toks, padTokenId, newTrainTok, newTrainNextTok, trSeq, cursor);
		}
	}

	if (newTrainTok.empty() || newTrainNextTok.empty() || newTrainTok.size() != newTrainNextTok.size())
	{
		clearLoadedData();
		lastImportStatus = glades::NNetworkStatus(glades::NNetworkStatus::EMPTY_DATA, "TokenInput::import(table): no LM pairs produced");
		return;
	}

	trainTok.swap(newTrainTok);
	trainNextTok.swap(newTrainNextTok);
	if (mirrorTrainToTestOnImplicitSplit)
	{
		testTok = trainTok;
		testNextTok = trainNextTok;
		teSeq = trSeq;
	}
	else
	{
		testTok.clear();
		testNextTok.clear();
		teSeq.clear();
	}
	if (!setTrainSequences(trSeq) || !setTestSequences(teSeq))
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
	(void)index;
	return shmea::GVector<float>();
}

shmea::GVector<float> glades::TokenInput::getTrainExpectedRow(unsigned int index) const
{
	(void)index;
	return shmea::GVector<float>();
}

shmea::GVector<float> glades::TokenInput::getTestRow(unsigned int index) const
{
	(void)index;
	return shmea::GVector<float>();
}

shmea::GVector<float> glades::TokenInput::getTestExpectedRow(unsigned int index) const
{
	(void)index;
	return shmea::GVector<float>();
}

bool glades::TokenInput::getTrainRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
{
	(void)index;
	outData = NULL;
	outSize = 0u;
	return false;
}

bool glades::TokenInput::getTrainExpectedRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
{
	(void)index;
	outData = NULL;
	outSize = 0u;
	return false;
}

bool glades::TokenInput::getTestRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
{
	(void)index;
	outData = NULL;
	outSize = 0u;
	return false;
}

bool glades::TokenInput::getTestExpectedRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
{
	(void)index;
	outData = NULL;
	outSize = 0u;
	return false;
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
