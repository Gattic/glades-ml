// Copyright 2026
//
// Unified, versioned model persistence for Glades ML.
//
// A "model package" is stored at:
//   database/models/<modelName>/{manifest.txt, nninfo.csv, weights.bin}
//
// This unifies architecture persistence (NNInfo) and parameter persistence (packed tensors)
// behind a single API. Legacy graph-based formats are not supported.

#include "network.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <time.h>
#ifndef _WIN32
#include <unistd.h>
#include <dirent.h>
#else
#include "Backend/Core/platform.h"
#include <io.h>
#include <process.h>
#define unlink _unlink
#define rmdir _rmdir
#define getpid _getpid
#ifndef S_ISDIR
#define S_ISDIR(m) (((m) & _S_IFMT) == _S_IFDIR)
#endif
#ifndef S_ISREG
#define S_ISREG(m) (((m) & _S_IFMT) == _S_IFREG)
#endif
#endif

namespace {

// v1: legacy graph weights (no longer supported).
// v2: packed tensor weights (authoritative parameters).
// v3: atomic package publish + file integrity metadata + stricter transformer config requirements.
static const int kModelFormatVersionMinSupported = 2;
static const int kModelFormatVersionLatest = 3;

// Tokenizer artifact format version (stored under modelDir/tokenizer/).
static const int kTokenizerFormatVersion = 1;

static inline bool is_path_separator(char c)
{
	return (c == '/') || (c == '\\');
}

// Reject traversal and unsafe names. This prevents `database/models/../../...` abuse.
static bool is_safe_path_component(const std::string& s)
{
	if (s.empty())
		return false;
	if (s == "." || s == "..")
		return false;
	if (s.size() > 128u)
		return false;
	if (s[0] == '.')
		return false;
	for (size_t i = 0; i < s.size(); ++i)
	{
		const char c = s[i];
		if (is_path_separator(c))
			return false;
		const bool ok =
		    (c >= 'a' && c <= 'z') ||
		    (c >= 'A' && c <= 'Z') ||
		    (c >= '0' && c <= '9') ||
		    (c == '_') || (c == '-') || (c == '.');
		if (!ok)
			return false;
	}
	if (s.find("..") != std::string::npos)
		return false;
	return true;
}

static bool stat_is_dir(const std::string& path)
{
	struct stat st;
#ifdef _WIN32
	if (::stat(path.c_str(), &st) != 0)
		return false;
#else
	if (::lstat(path.c_str(), &st) != 0)
		return false;
	// Do not follow symlinks for persistence roots.
	if (S_ISLNK(st.st_mode))
		return false;
#endif
	return S_ISDIR(st.st_mode) != 0;
}

static bool stat_is_file(const std::string& path)
{
	struct stat st;
#ifdef _WIN32
	if (::stat(path.c_str(), &st) != 0)
		return false;
#else
	if (::lstat(path.c_str(), &st) != 0)
		return false;
	// Do not follow symlinks for persistence files.
	if (S_ISLNK(st.st_mode))
		return false;
#endif
	return S_ISREG(st.st_mode) != 0;
}

static bool mkdir_if_missing(const std::string& path)
{
	if (path.empty())
		return false;

	// 0777 is filtered by umask; matches existing project style.
	if (::mkdir(path.c_str(), 0777) == 0)
		return true;

	// Already exists is fine.
	if (errno == EEXIST)
		return stat_is_dir(path);
	return false;
}

static bool ensure_models_root()
{
	// Keep behavior consistent with existing relative "database/..." paths.
	if (!mkdir_if_missing("database"))
		return false;
	if (!mkdir_if_missing("database/models"))
		return false;
	return true;
}

static bool ensure_models_dir(const std::string& modelName)
{
	if (!ensure_models_root())
		return false;

	if (modelName.empty())
		return false;
	if (!is_safe_path_component(modelName))
		return false;
	return mkdir_if_missing(std::string("database/models/") + modelName);
}

static std::string model_dir(const std::string& modelName)
{
	return std::string("database/models/") + modelName + "/";
}

static std::string model_dir_no_slash(const std::string& modelName)
{
	return std::string("database/models/") + modelName;
}

static bool rename_atomic(const std::string& from, const std::string& to)
{
	return ::rename(from.c_str(), to.c_str()) == 0;
}

static bool remove_tree_recursive(const std::string& path)
{
#ifdef _WIN32
	if (!stat_is_dir(path))
	{
		if (stat_is_file(path))
			return ::_unlink(path.c_str()) == 0;
		return true;
	}

	std::string pattern = path + "\\*";
	WIN32_FIND_DATAA fd;
	HANDLE hFind = FindFirstFileA(pattern.c_str(), &fd);
	if (hFind == INVALID_HANDLE_VALUE)
		return ::_rmdir(path.c_str()) == 0;

	do
	{
		const char* name = fd.cFileName;
		if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0)
			continue;
		const std::string child = path + "\\" + std::string(name);
		if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
		{
			(void)remove_tree_recursive(child);
			(void)::_rmdir(child.c_str());
		}
		else
		{
			(void)::_unlink(child.c_str());
		}
	} while (FindNextFileA(hFind, &fd));
	FindClose(hFind);
	return ::_rmdir(path.c_str()) == 0;
#else
	DIR* d = ::opendir(path.c_str());
	if (!d)
	{
		// If it's a file, try unlink; otherwise nothing to do.
		if (stat_is_file(path))
			return ::unlink(path.c_str()) == 0;
		return true;
	}

	struct dirent* ent = NULL;
	while ((ent = ::readdir(d)) != NULL)
	{
		const char* name = ent->d_name;
		if (!name)
			continue;
		if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0)
			continue;
		const std::string child = path + "/" + std::string(name);
		struct stat st;
		if (::lstat(child.c_str(), &st) != 0)
			continue;
		if (S_ISLNK(st.st_mode))
		{
			(void)::unlink(child.c_str());
			continue;
		}
		if (S_ISDIR(st.st_mode))
		{
			(void)remove_tree_recursive(child);
			(void)::rmdir(child.c_str());
		}
		else
		{
			(void)::unlink(child.c_str());
		}
	}
	::closedir(d);
	return ::rmdir(path.c_str()) == 0;
#endif
}

struct ModelManifest
{
	std::string magic;
	int version;
	int netType;
	std::string name;
	int epochs;
	bool hasEpochs;
	uint64_t rngSeed;
	bool hasSeed;
	glades::TrainingConfig trainingConfig;
	bool hasTrainingConfig;

	// Optional file integrity metadata (v3+; best-effort for older packages).
	uint64_t weightsBytes;
	uint64_t weightsFNV1a64;
	bool hasWeightsBytes;
	bool hasWeightsFNV;
	uint64_t nninfoBytes;
	bool hasNninfoBytes;

	// Tokenizer integrity metadata duplicated in the top-level manifest (v3+).
	bool tokenizerPresent;
	bool hasTokenizerPresent;
	uint64_t tokenizerVocabCount;
	bool hasTokenizerVocabCount;
	uint64_t tokenizerFNV1a64;
	bool hasTokenizerFNV;

	ModelManifest()
	    : magic(),
	      version(-1),
	      netType(-1),
	      name(),
	      epochs(0),
	      hasEpochs(false),
	      rngSeed(0u),
	      hasSeed(false),
	      trainingConfig(),
	      hasTrainingConfig(false),
	      weightsBytes(0ULL),
	      weightsFNV1a64(0ULL),
	      hasWeightsBytes(false),
	      hasWeightsFNV(false),
	      nninfoBytes(0ULL),
	      hasNninfoBytes(false),
	      tokenizerPresent(false),
	      hasTokenizerPresent(false),
	      tokenizerVocabCount(0ULL),
	      hasTokenizerVocabCount(false),
	      tokenizerFNV1a64(0ULL),
	      hasTokenizerFNV(false)
	{
	}
};

static bool parse_int_strict(const std::string& s, int& out)
{
	if (s.empty())
		return false;
	char* end = NULL;
	errno = 0;
	long v = ::strtol(s.c_str(), &end, 10);
	if (errno != 0 || end == s.c_str() || (end && *end != '\0'))
		return false;
	if (v < static_cast<long>(std::numeric_limits<int>::min()) || v > static_cast<long>(std::numeric_limits<int>::max()))
		return false;
	out = static_cast<int>(v);
	return true;
}

static bool parse_u64_strict(const std::string& s, uint64_t& out)
{
	if (s.empty())
		return false;
	char* end = NULL;
	errno = 0;
	unsigned long long v = ::strtoull(s.c_str(), &end, 10);
	if (errno != 0 || end == s.c_str() || (end && *end != '\0'))
		return false;
	out = static_cast<uint64_t>(v);
	return true;
}

static bool parse_float_strict(const std::string& s, float& out)
{
	if (s.empty())
		return false;
	char* end = NULL;
	errno = 0;
	float v = ::strtof(s.c_str(), &end);
	if (errno != 0 || end == s.c_str() || (end && *end != '\0'))
		return false;
	if (!std::isfinite(v))
		return false;
	out = v;
	return true;
}

static bool read_kv_manifest(const std::string& manifestPath, ModelManifest& m)
{
	m = ModelManifest();

	std::ifstream in(manifestPath.c_str());
	if (!in)
		return false;

	std::string line;
	if (!std::getline(in, line))
		return false;
	m.magic = line;

	while (std::getline(in, line))
	{
		if (line.empty())
			continue;

		const size_t eq = line.find('=');
		if (eq == std::string::npos)
			continue;
		const std::string key = line.substr(0, eq);
		const std::string val = line.substr(eq + 1);

		if (key == "version")
		{
			(void)parse_int_strict(val, m.version);
		}
		else if (key == "netType")
		{
			(void)parse_int_strict(val, m.netType);
		}
		else if (key == "name")
			m.name = val;
		else if (key == "epochs")
		{
			int e = 0;
			if (parse_int_strict(val, e))
			{
				m.epochs = e;
				m.hasEpochs = true;
			}
		}
		else if (key == "rngSeed")
		{
			// Stored as unsigned integer in text.
			uint64_t s = 0u;
			if (parse_u64_strict(val, s))
			{
				m.rngSeed = s;
				m.hasSeed = true;
			}
		}
		// File integrity metadata (optional; v3+).
		else if (key == "weights.bytes")
		{
			uint64_t v = 0ULL;
			if (parse_u64_strict(val, v)) { m.weightsBytes = v; m.hasWeightsBytes = true; }
		}
		else if (key == "weights.fnv1a64")
		{
			uint64_t v = 0ULL;
			if (parse_u64_strict(val, v)) { m.weightsFNV1a64 = v; m.hasWeightsFNV = true; }
		}
		else if (key == "nninfo.bytes")
		{
			uint64_t v = 0ULL;
			if (parse_u64_strict(val, v)) { m.nninfoBytes = v; m.hasNninfoBytes = true; }
		}
		// Tokenizer integrity duplicated in the top-level manifest (optional; v3+).
		else if (key == "tokenizer.present")
		{
			int v = 0;
			if (parse_int_strict(val, v)) { m.tokenizerPresent = (v != 0); m.hasTokenizerPresent = true; }
		}
		else if (key == "tokenizer.vocabCount")
		{
			uint64_t v = 0ULL;
			if (parse_u64_strict(val, v)) { m.tokenizerVocabCount = v; m.hasTokenizerVocabCount = true; }
		}
		else if (key == "tokenizer.fnv1a64")
		{
			uint64_t v = 0ULL;
			if (parse_u64_strict(val, v)) { m.tokenizerFNV1a64 = v; m.hasTokenizerFNV = true; }
		}
		// TrainingConfig (optional; missing keys keep defaults)
		else if (key == "training.minibatchSizeOverride")
		{
			int v = 0;
			if (parse_int_strict(val, v)) { m.trainingConfig.minibatchSizeOverride = v; m.hasTrainingConfig = true; }
		}
		else if (key == "training.tbpttWindowOverride")
		{
			int v = 0;
			if (parse_int_strict(val, v)) { m.trainingConfig.tbpttWindowOverride = v; m.hasTrainingConfig = true; }
		}
		else if (key == "training.globalGradClipNorm")
		{
			float v = 0.0f;
			if (parse_float_strict(val, v)) { m.trainingConfig.globalGradClipNorm = v; m.hasTrainingConfig = true; }
		}
		else if (key == "training.perElementGradClip")
		{
			float v = 0.0f;
			if (parse_float_strict(val, v)) { m.trainingConfig.perElementGradClip = v; m.hasTrainingConfig = true; }
		}
		else if (key == "training.optimizer.type")
		{
			int v = 0;
			if (parse_int_strict(val, v))
			{
				m.trainingConfig.optimizer.type = static_cast<glades::OptimizerConfig::Type>(v);
				m.hasTrainingConfig = true;
			}
		}
		else if (key == "training.optimizer.adamBeta1")
		{
			float v = 0.0f;
			if (parse_float_strict(val, v)) { m.trainingConfig.optimizer.adamBeta1 = v; m.hasTrainingConfig = true; }
		}
		else if (key == "training.optimizer.adamBeta2")
		{
			float v = 0.0f;
			if (parse_float_strict(val, v)) { m.trainingConfig.optimizer.adamBeta2 = v; m.hasTrainingConfig = true; }
		}
		else if (key == "training.optimizer.adamEps")
		{
			float v = 0.0f;
			if (parse_float_strict(val, v)) { m.trainingConfig.optimizer.adamEps = v; m.hasTrainingConfig = true; }
		}
		else if (key == "training.optimizer.adamBiasCorrection")
		{
			int v = 0;
			if (parse_int_strict(val, v)) { m.trainingConfig.optimizer.adamBiasCorrection = (v != 0); m.hasTrainingConfig = true; }
		}
		else if (key == "training.lrSchedule.type")
		{
			int v = 0;
			if (parse_int_strict(val, v))
			{
				m.trainingConfig.lrSchedule.type = static_cast<glades::LearningRateScheduleConfig::Type>(v);
				m.hasTrainingConfig = true;
			}
		}
		else if (key == "training.lrSchedule.stepSizeEpochs")
		{
			int v = 0;
			if (parse_int_strict(val, v)) { m.trainingConfig.lrSchedule.stepSizeEpochs = v; m.hasTrainingConfig = true; }
		}
		else if (key == "training.lrSchedule.gamma")
		{
			float v = 0.0f;
			if (parse_float_strict(val, v)) { m.trainingConfig.lrSchedule.gamma = v; m.hasTrainingConfig = true; }
		}
		else if (key == "training.lrSchedule.cosineTMaxEpochs")
		{
			int v = 0;
			if (parse_int_strict(val, v)) { m.trainingConfig.lrSchedule.cosineTMaxEpochs = v; m.hasTrainingConfig = true; }
		}
		else if (key == "training.lrSchedule.minMultiplier")
		{
			float v = 0.0f;
			if (parse_float_strict(val, v)) { m.trainingConfig.lrSchedule.minMultiplier = v; m.hasTrainingConfig = true; }
		}
		// Mixed precision (optional; missing keys keep defaults)
		else if (key == "training.mixedPrecision.enable")
		{
			int v = 0;
			if (parse_int_strict(val, v)) { m.trainingConfig.mixedPrecision.enable = (v != 0); m.hasTrainingConfig = true; }
		}
		else if (key == "training.mixedPrecision.weightDType")
		{
			int v = 0;
			if (parse_int_strict(val, v))
			{
				m.trainingConfig.mixedPrecision.weightDType = static_cast<glades::MixedPrecisionConfig::WeightDType>(v);
				m.hasTrainingConfig = true;
			}
		}
		else if (key == "training.mixedPrecision.useLossScaling")
		{
			int v = 0;
			if (parse_int_strict(val, v)) { m.trainingConfig.mixedPrecision.useLossScaling = (v != 0); m.hasTrainingConfig = true; }
		}
		else if (key == "training.mixedPrecision.dynamicLossScaling")
		{
			int v = 0;
			if (parse_int_strict(val, v)) { m.trainingConfig.mixedPrecision.dynamicLossScaling = (v != 0); m.hasTrainingConfig = true; }
		}
		else if (key == "training.mixedPrecision.lossScaleInit")
		{
			float v = 0.0f;
			if (parse_float_strict(val, v)) { m.trainingConfig.mixedPrecision.lossScaleInit = v; m.hasTrainingConfig = true; }
		}
		else if (key == "training.mixedPrecision.lossScaleMin")
		{
			float v = 0.0f;
			if (parse_float_strict(val, v)) { m.trainingConfig.mixedPrecision.lossScaleMin = v; m.hasTrainingConfig = true; }
		}
		else if (key == "training.mixedPrecision.lossScaleMax")
		{
			float v = 0.0f;
			if (parse_float_strict(val, v)) { m.trainingConfig.mixedPrecision.lossScaleMax = v; m.hasTrainingConfig = true; }
		}
		else if (key == "training.mixedPrecision.growthInterval")
		{
			int v = 0;
			if (parse_int_strict(val, v)) { m.trainingConfig.mixedPrecision.growthInterval = v; m.hasTrainingConfig = true; }
		}
		else if (key == "training.mixedPrecision.growthFactor")
		{
			float v = 0.0f;
			if (parse_float_strict(val, v)) { m.trainingConfig.mixedPrecision.growthFactor = v; m.hasTrainingConfig = true; }
		}
		else if (key == "training.mixedPrecision.backoffFactor")
		{
			float v = 0.0f;
			if (parse_float_strict(val, v)) { m.trainingConfig.mixedPrecision.backoffFactor = v; m.hasTrainingConfig = true; }
		}
		// TransformerRunConfig (optional; missing keys keep defaults)
		else if (key == "training.transformer.nHeadsOverride")
		{
			int v = 0;
			if (parse_int_strict(val, v)) { m.trainingConfig.transformer.nHeadsOverride = v; m.hasTrainingConfig = true; }
		}
		else if (key == "training.transformer.dFFOverride")
		{
			int v = 0;
			if (parse_int_strict(val, v)) { m.trainingConfig.transformer.dFFOverride = v; m.hasTrainingConfig = true; }
		}
		else if (key == "training.transformer.layerNormEps")
		{
			float v = 0.0f;
			if (parse_float_strict(val, v)) { m.trainingConfig.transformer.layerNormEps = v; m.hasTrainingConfig = true; }
		}
		else if (key == "training.transformer.normType")
		{
			int v = 0;
			if (parse_int_strict(val, v))
			{
				m.trainingConfig.transformer.normType = static_cast<glades::TransformerRunConfig::NormType>(v);
				m.hasTrainingConfig = true;
			}
		}
		else if (key == "training.transformer.positionalEncoding")
		{
			int v = 0;
			if (parse_int_strict(val, v))
			{
				m.trainingConfig.transformer.positionalEncoding = static_cast<glades::TransformerRunConfig::PositionalEncodingType>(v);
				m.hasTrainingConfig = true;
			}
		}
		else if (key == "training.transformer.kvCacheDType")
		{
			int v = 0;
			if (parse_int_strict(val, v))
			{
				m.trainingConfig.transformer.kvCacheDType = static_cast<glades::TransformerRunConfig::KVCacheDType>(v);
				m.hasTrainingConfig = true;
			}
		}
		else if (key == "training.transformer.ropeDimOverride")
		{
			int v = 0;
			if (parse_int_strict(val, v)) { m.trainingConfig.transformer.ropeDimOverride = v; m.hasTrainingConfig = true; }
		}
		else if (key == "training.transformer.ropeTheta")
		{
			float v = 0.0f;
			if (parse_float_strict(val, v)) { m.trainingConfig.transformer.ropeTheta = v; m.hasTrainingConfig = true; }
		}
		else if (key == "training.transformer.ffnKind")
		{
			int v = 0;
			if (parse_int_strict(val, v))
			{
				m.trainingConfig.transformer.ffnKind = static_cast<glades::TransformerRunConfig::FFNKind>(v);
				m.hasTrainingConfig = true;
			}
		}
		else if (key == "training.transformer.ffnActivation")
		{
			int v = 0;
			if (parse_int_strict(val, v))
			{
				m.trainingConfig.transformer.ffnActivation = static_cast<glades::TransformerRunConfig::FFNActivationType>(v);
				m.hasTrainingConfig = true;
			}
		}
		else if (key == "training.transformer.nKVHeadsOverride")
		{
			int v = 0;
			if (parse_int_strict(val, v)) { m.trainingConfig.transformer.nKVHeadsOverride = v; m.hasTrainingConfig = true; }
		}
		else if (key == "training.transformer.enableTokenEmbedding")
		{
			int v = 0;
			if (parse_int_strict(val, v)) { m.trainingConfig.transformer.enableTokenEmbedding = (v != 0); m.hasTrainingConfig = true; }
		}
		else if (key == "training.transformer.vocabSizeOverride")
		{
			int v = 0;
			if (parse_int_strict(val, v)) { m.trainingConfig.transformer.vocabSizeOverride = v; m.hasTrainingConfig = true; }
		}
		else if (key == "training.transformer.tieEmbeddings")
		{
			int v = 0;
			if (parse_int_strict(val, v)) { m.trainingConfig.transformer.tieEmbeddings = (v != 0); m.hasTrainingConfig = true; }
		}
		else if (key == "training.transformer.padTokenId")
		{
			int v = 0;
			if (parse_int_strict(val, v)) { m.trainingConfig.transformer.padTokenId = v; m.hasTrainingConfig = true; }
		}
		else if (key == "training.transformer.tokenLmLossKind")
		{
			int v = 0;
			if (parse_int_strict(val, v))
			{
				m.trainingConfig.transformer.tokenLmLossKind = static_cast<glades::TransformerRunConfig::TokenLMLossKind>(v);
				m.hasTrainingConfig = true;
			}
		}
		else if (key == "training.transformer.tokenLmSampledNegatives")
		{
			int v = 0;
			if (parse_int_strict(val, v)) { m.trainingConfig.transformer.tokenLmSampledNegatives = v; m.hasTrainingConfig = true; }
		}
		else if (key == "training.transformer.tokenLmAllowHugeFullSoftmax")
		{
			int v = 0;
			if (parse_int_strict(val, v)) { m.trainingConfig.transformer.tokenLmAllowHugeFullSoftmax = (v != 0); m.hasTrainingConfig = true; }
		}
	}

	return true;
}

static uint64_t fnv1a64_update(uint64_t h, const void* data, size_t n)
{
	// FNV-1a 64-bit
	static const uint64_t kOffset = 14695981039346656037ULL;
	static const uint64_t kPrime = 1099511628211ULL;
	if (h == 0ULL)
		h = kOffset;
	const unsigned char* p = static_cast<const unsigned char*>(data);
	for (size_t i = 0; i < n; ++i)
	{
		h ^= static_cast<uint64_t>(p[i]);
		h *= kPrime;
	}
	return h;
}

static bool fnv1a64_hash_file_prefix(const std::string& path, uint64_t bytesToHash, uint64_t& outHash)
{
	outHash = 0ULL;
	std::ifstream in(path.c_str(), std::ios::in | std::ios::binary);
	if (!in)
		return false;
	static const size_t kBuf = 1u << 16;
	char buf[kBuf];
	uint64_t remaining = bytesToHash;
	while (remaining > 0ULL)
	{
		const size_t want = (remaining > static_cast<uint64_t>(kBuf)) ? kBuf : static_cast<size_t>(remaining);
		in.read(buf, static_cast<std::streamsize>(want));
		const std::streamsize got = in.gcount();
		if (got <= 0)
			return false;
		outHash = fnv1a64_update(outHash, buf, static_cast<size_t>(got));
		remaining -= static_cast<uint64_t>(got);
	}
	return true;
}

static void write_u32_le(std::ostream& out, uint32_t v)
{
	unsigned char b[4];
	b[0] = static_cast<unsigned char>(v & 0xffu);
	b[1] = static_cast<unsigned char>((v >> 8) & 0xffu);
	b[2] = static_cast<unsigned char>((v >> 16) & 0xffu);
	b[3] = static_cast<unsigned char>((v >> 24) & 0xffu);
	out.write(reinterpret_cast<const char*>(b), 4);
}

static bool read_u32_le(std::istream& in, uint32_t& outV)
{
	unsigned char b[4];
	in.read(reinterpret_cast<char*>(b), 4);
	if (!in)
		return false;
	outV = (static_cast<uint32_t>(b[0])      ) |
	       (static_cast<uint32_t>(b[1]) <<  8) |
	       (static_cast<uint32_t>(b[2]) << 16) |
	       (static_cast<uint32_t>(b[3]) << 24);
	return true;
}

static bool is_safe_relative_file(const std::string& s)
{
	// Very strict: only allow a simple filename.
	// This prevents traversal like "../x" or "subdir/x".
	return is_safe_path_component(s) && (s.find('/') == std::string::npos) && (s.find('\\') == std::string::npos);
}

struct TokenizerManifest
{
	std::string magic;
	int version;
	std::string type;
	std::string vocabFile;
	uint64_t vocabCount;
	bool hasVocabCount;
	uint64_t fnv1a64;
	bool hasFNV;
	int padTokenId;
	int bosTokenId;
	int eosTokenId;
	int unkTokenId;

	TokenizerManifest()
	    : magic(),
	      version(-1),
	      type(),
	      vocabFile(),
	      vocabCount(0ULL),
	      hasVocabCount(false),
	      fnv1a64(0ULL),
	      hasFNV(false),
	      padTokenId(-1),
	      bosTokenId(-1),
	      eosTokenId(-1),
	      unkTokenId(-1)
	{
	}
};

static bool read_tokenizer_manifest(const std::string& manifestPath, TokenizerManifest& m)
{
	m = TokenizerManifest();
	std::ifstream in(manifestPath.c_str());
	if (!in)
		return false;

	std::string line;
	if (!std::getline(in, line))
		return false;
	m.magic = line;

	while (std::getline(in, line))
	{
		if (line.empty())
			continue;
		const size_t eq = line.find('=');
		if (eq == std::string::npos)
			continue;
		const std::string key = line.substr(0, eq);
		const std::string val = line.substr(eq + 1);

		if (key == "version")
		{
			(void)parse_int_strict(val, m.version);
		}
		else if (key == "type")
		{
			m.type = val;
		}
		else if (key == "vocabFile")
		{
			m.vocabFile = val;
		}
		else if (key == "vocabCount")
		{
			uint64_t v = 0ULL;
			if (parse_u64_strict(val, v))
			{
				m.vocabCount = v;
				m.hasVocabCount = true;
			}
		}
		else if (key == "fnv1a64")
		{
			uint64_t v = 0ULL;
			if (parse_u64_strict(val, v))
			{
				m.fnv1a64 = v;
				m.hasFNV = true;
			}
		}
		else if (key == "special.padTokenId")
		{
			(void)parse_int_strict(val, m.padTokenId);
		}
		else if (key == "special.bosTokenId")
		{
			(void)parse_int_strict(val, m.bosTokenId);
		}
		else if (key == "special.eosTokenId")
		{
			(void)parse_int_strict(val, m.eosTokenId);
		}
		else if (key == "special.unkTokenId")
		{
			(void)parse_int_strict(val, m.unkTokenId);
		}
	}

	return true;
}

static bool write_vocab_bin_atomic(const std::string& vocabPath, const std::vector<std::string>& vocab, uint64_t& outHash)
{
	const std::string tmp = vocabPath + ".tmp";
	std::ofstream out(tmp.c_str(), std::ios::out | std::ios::trunc | std::ios::binary);
	if (!out)
		return false;

	outHash = 0ULL;
	const uint32_t n = static_cast<uint32_t>(vocab.size());
	write_u32_le(out, n);
	outHash = fnv1a64_update(outHash, &n, sizeof(n));

	for (size_t i = 0; i < vocab.size(); ++i)
	{
		const std::string& s = vocab[i];
		const uint32_t len = static_cast<uint32_t>(s.size());
		write_u32_le(out, len);
		outHash = fnv1a64_update(outHash, &len, sizeof(len));
		if (len > 0u)
		{
			out.write(s.data(), static_cast<std::streamsize>(len));
			outHash = fnv1a64_update(outHash, s.data(), static_cast<size_t>(len));
		}
		if (!out)
		{
			out.close();
			(void)::remove(tmp.c_str());
			return false;
		}
	}
	out.flush();
	out.close();
	if (!out || ::rename(tmp.c_str(), vocabPath.c_str()) != 0)
	{
		(void)::remove(tmp.c_str());
		return false;
	}
	return true;
}

static bool read_vocab_bin(const std::string& vocabPath, std::vector<std::string>& outVocab, uint64_t& outHash)
{
	outVocab.clear();
	outHash = 0ULL;

	std::ifstream in(vocabPath.c_str(), std::ios::in | std::ios::binary);
	if (!in)
		return false;

	uint32_t n = 0u;
	if (!read_u32_le(in, n))
		return false;
	outHash = fnv1a64_update(outHash, &n, sizeof(n));

	// Conservative caps for hostile inputs.
	static const uint32_t kMaxVocab = 5u * 1000u * 1000u;
	static const uint32_t kMaxTokenBytes = 1024u * 1024u;
	static const uint64_t kMaxTotalBytes = 1024ull * 1024ull * 1024ull;
	if (n == 0u || n > kMaxVocab)
		return false;
	outVocab.reserve(static_cast<size_t>(n));

	uint64_t totalBytes = 0ULL;
	for (uint32_t i = 0u; i < n; ++i)
	{
		uint32_t len = 0u;
		if (!read_u32_le(in, len))
			return false;
		outHash = fnv1a64_update(outHash, &len, sizeof(len));
		if (len > kMaxTokenBytes)
			return false;
		totalBytes += static_cast<uint64_t>(len);
		if (totalBytes > kMaxTotalBytes)
			return false;

		std::string s;
		if (len > 0u)
		{
			s.resize(static_cast<size_t>(len));
			in.read(&s[0], static_cast<std::streamsize>(len));
			if (!in)
				return false;
			outHash = fnv1a64_update(outHash, s.data(), static_cast<size_t>(len));
		}
		outVocab.push_back(s);
	}
	return true;
}

} // namespace

namespace glades {

NNetworkStatus NNetwork::saveModel(const std::string& modelName, const DataInput* externalDI) const
{
	if (modelName.empty())
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "saveModel: modelName is empty");
	if (!is_safe_path_component(modelName))
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "saveModel: modelName contains unsafe characters");
	if (!skeleton)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "saveModel: skeleton is null");

	// Tensor-first: weights are persisted from packed tensors (the single source of truth).
	// If tensors are not initialized yet, try to initialize them from the attached DataInput.
	{
		const bool hasDff = (netType == TYPE_DFF) && tensorDff.initialized;
		const bool hasRnn = (netType == TYPE_RNN) && tensorRnn.initialized;
		const bool hasGru = (netType == TYPE_GRU) && tensorGru.initialized;
		const bool hasLstm = (netType == TYPE_LSTM) && tensorLstm.initialized;
		const bool hasTr = (netType == TYPE_TRANSFORMER_ENCODER || netType == TYPE_TRANSFORMER_DECODER) && tensorTransformer.initialized;
		const bool hasCnn = (netType == TYPE_CNN) && tensorCnn.initialized;
		if (!hasDff && !hasRnn && !hasGru && !hasLstm && !hasTr && !hasCnn)
		{
			// If caller provided an external DataInput (e.g. after test() detached the
			// internal pointer via RunDataAttachmentGuard), temporarily attach it so
			// ensureTensorParametersInitialized() can read the feature count.
			glades::NNetwork* mut = const_cast<glades::NNetwork*>(this);
			const DataInput* prevDI = mut->di;
			if (externalDI && !mut->di)
				mut->di = externalDI;
			const bool ok = mut->ensureTensorParametersInitialized();
			mut->di = prevDI;
			if (!ok)
				return lastStatus;
		}
	}

	if (!ensure_models_root())
		return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveModel: unable to create database/models directory");

	const std::string finalDirNoSlash = model_dir_no_slash(modelName);

	// Create a unique temp directory name under the same root so rename is atomic.
	const unsigned long long pid = static_cast<unsigned long long>(::getpid());
	const unsigned long long t = static_cast<unsigned long long>(::time(NULL));
	std::string tmpDirNoSlash;
	for (int attempt = 0; attempt < 16; ++attempt)
	{
		std::ostringstream tmpName;
		tmpName << modelName << ".tmp_" << pid << "_" << t << "_" << attempt;
		const std::string candidate = std::string("database/models/") + tmpName.str();
		errno = 0;
		if (::mkdir(candidate.c_str(), 0777) == 0)
		{
			tmpDirNoSlash = candidate;
			break;
		}
		if (errno == EEXIST)
			continue;
		// Some other error (permissions, etc).
		break;
	}
	if (tmpDirNoSlash.empty())
		return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveModel: unable to create temporary model directory");

	struct TmpDirGuard
	{
		std::string pathNoSlash;
		bool keep;
		explicit TmpDirGuard(const std::string& p) : pathNoSlash(p), keep(false) {}
		~TmpDirGuard()
		{
			if (!keep && !pathNoSlash.empty())
				(void)remove_tree_recursive(pathNoSlash);
		}
		void dismiss() { keep = true; }
	private:
		TmpDirGuard(const TmpDirGuard&);
		TmpDirGuard& operator=(const TmpDirGuard&);
	};
	TmpDirGuard tmpGuard(tmpDirNoSlash);

	const std::string dir = tmpDirNoSlash + "/";
	const std::string manifestPath = dir + "manifest.txt";
	const std::string nninfoPath = dir + "nninfo.csv";
	const std::string weightsPath = dir + "weights.bin";
	const std::string tokDir = dir + "tokenizer/";
	const std::string tokManifestPath = tokDir + "manifest.txt";
	const std::string tokVocabPath = tokDir + "vocab.bin";

	// 1) architecture
	{
		const shmea::GTable tinfo = skeleton->toGTable();
		const std::string tmp = nninfoPath + ".tmp";
		tinfo.save(shmea::GString(tmp.c_str()));
		if (!rename_atomic(tmp, nninfoPath))
		{
			(void)::remove(tmp.c_str());
			return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveModel: failed to publish nninfo.csv (rename failed)");
		}
	}

	// 2) weights
	{
		const NNetworkStatus stW = saveTensorWeightsToFile(weightsPath);
		if (!stW.ok())
			return stW;
	}

	// 3) tokenizer artifacts (optional)
	uint64_t vocabHash = 0ULL;
	if (tokenizerArtifactsPresent)
	{
		if (!mkdir_if_missing(tokDir.substr(0, tokDir.size() - 1)))
			return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveModel: unable to create tokenizer directory");

		if (!write_vocab_bin_atomic(tokVocabPath, tokenizerArtifacts.vocab, vocabHash))
			return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveModel: failed to write tokenizer vocab.bin");

		{
			const std::string tmp = tokManifestPath + ".tmp";
			std::ofstream out(tmp.c_str(), std::ios::out | std::ios::trunc);
			if (!out)
				return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveModel: unable to write tokenizer manifest.txt");

			out << "GLADES_TOKENIZER" << "\n";
			out << "version=" << kTokenizerFormatVersion << "\n";
			out << "type=" << tokenizerArtifacts.type << "\n";
			out << "vocabFile=vocab.bin" << "\n";
			out << "vocabCount=" << static_cast<unsigned long long>(tokenizerArtifacts.vocab.size()) << "\n";
			out << "fnv1a64=" << static_cast<unsigned long long>(vocabHash) << "\n";
			out << "special.padTokenId=" << tokenizerArtifacts.padTokenId << "\n";
			out << "special.bosTokenId=" << tokenizerArtifacts.bosTokenId << "\n";
			out << "special.eosTokenId=" << tokenizerArtifacts.eosTokenId << "\n";
			out << "special.unkTokenId=" << tokenizerArtifacts.unkTokenId << "\n";
			out.flush();
			out.close();
			if (!out || !rename_atomic(tmp, tokManifestPath))
			{
				(void)::remove(tmp.c_str());
				return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveModel: failed to publish tokenizer manifest.txt (rename failed)");
			}
		}
	}

	// Compute file integrity metadata (v3+).
	uint64_t nninfoBytes = 0ULL;
	uint64_t weightsBytes = 0ULL;
	uint64_t weightsHash = 0ULL;
	{
		struct stat st;
#ifdef _WIN32
		if (::stat(nninfoPath.c_str(), &st) == 0 && S_ISREG(st.st_mode))
			nninfoBytes = static_cast<uint64_t>(st.st_size);
		if (::stat(weightsPath.c_str(), &st) == 0 && S_ISREG(st.st_mode))
			weightsBytes = static_cast<uint64_t>(st.st_size);
#else
		if (::lstat(nninfoPath.c_str(), &st) == 0 && S_ISREG(st.st_mode))
			nninfoBytes = static_cast<uint64_t>(st.st_size);
		if (::lstat(weightsPath.c_str(), &st) == 0 && S_ISREG(st.st_mode))
			weightsBytes = static_cast<uint64_t>(st.st_size);
#endif
		if (weightsBytes > 0ULL)
		{
			if (!fnv1a64_hash_file_prefix(weightsPath, weightsBytes, weightsHash))
				return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveModel: failed to compute weights.bin checksum");
		}
	}

	// 4) manifest (written last)
	{
		const std::string tmp = manifestPath + ".tmp";
		std::ofstream out(tmp.c_str(), std::ios::out | std::ios::trunc);
		if (!out)
			return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveModel: unable to write manifest.txt");

		out << "GLADES_MODEL" << "\n";
		out << "version=" << kModelFormatVersionLatest << "\n";
		out << "name=" << modelName << "\n";
		out << "netType=" << netType << "\n";
		out << "epochs=" << epochs << "\n";
		out << "rngSeed=" << static_cast<unsigned long long>(rngSeed) << "\n";
		out << "nninfo.file=nninfo.csv\n";
		out << "nninfo.bytes=" << static_cast<unsigned long long>(nninfoBytes) << "\n";
		out << "weights.file=weights.bin\n";
		out << "weights.bytes=" << static_cast<unsigned long long>(weightsBytes) << "\n";
		out << "weights.fnv1a64=" << static_cast<unsigned long long>(weightsHash) << "\n";

		// Persist TrainingConfig so inference matches training-time conventions.
		out << "training.minibatchSizeOverride=" << trainingConfig.minibatchSizeOverride << "\n";
		out << "training.tbpttWindowOverride=" << trainingConfig.tbpttWindowOverride << "\n";
		out << "training.globalGradClipNorm=" << trainingConfig.globalGradClipNorm << "\n";
		out << "training.perElementGradClip=" << trainingConfig.perElementGradClip << "\n";
		out << "training.optimizer.type=" << static_cast<int>(trainingConfig.optimizer.type) << "\n";
		out << "training.optimizer.adamBeta1=" << trainingConfig.optimizer.adamBeta1 << "\n";
		out << "training.optimizer.adamBeta2=" << trainingConfig.optimizer.adamBeta2 << "\n";
		out << "training.optimizer.adamEps=" << trainingConfig.optimizer.adamEps << "\n";
		out << "training.optimizer.adamBiasCorrection=" << (trainingConfig.optimizer.adamBiasCorrection ? 1 : 0) << "\n";
		out << "training.lrSchedule.type=" << static_cast<int>(trainingConfig.lrSchedule.type) << "\n";
		out << "training.lrSchedule.stepSizeEpochs=" << trainingConfig.lrSchedule.stepSizeEpochs << "\n";
		out << "training.lrSchedule.gamma=" << trainingConfig.lrSchedule.gamma << "\n";
		out << "training.lrSchedule.cosineTMaxEpochs=" << trainingConfig.lrSchedule.cosineTMaxEpochs << "\n";
		out << "training.lrSchedule.minMultiplier=" << trainingConfig.lrSchedule.minMultiplier << "\n";
		// Mixed precision
		out << "training.mixedPrecision.enable=" << (trainingConfig.mixedPrecision.enable ? 1 : 0) << "\n";
		out << "training.mixedPrecision.weightDType=" << static_cast<int>(trainingConfig.mixedPrecision.weightDType) << "\n";
		out << "training.mixedPrecision.useLossScaling=" << (trainingConfig.mixedPrecision.useLossScaling ? 1 : 0) << "\n";
		out << "training.mixedPrecision.dynamicLossScaling=" << (trainingConfig.mixedPrecision.dynamicLossScaling ? 1 : 0) << "\n";
		out << "training.mixedPrecision.lossScaleInit=" << trainingConfig.mixedPrecision.lossScaleInit << "\n";
		out << "training.mixedPrecision.lossScaleMin=" << trainingConfig.mixedPrecision.lossScaleMin << "\n";
		out << "training.mixedPrecision.lossScaleMax=" << trainingConfig.mixedPrecision.lossScaleMax << "\n";
		out << "training.mixedPrecision.growthInterval=" << trainingConfig.mixedPrecision.growthInterval << "\n";
		out << "training.mixedPrecision.growthFactor=" << trainingConfig.mixedPrecision.growthFactor << "\n";
		out << "training.mixedPrecision.backoffFactor=" << trainingConfig.mixedPrecision.backoffFactor << "\n";
		// TransformerRunConfig (these affect inference too).
		out << "training.transformer.nHeadsOverride=" << trainingConfig.transformer.nHeadsOverride << "\n";
		out << "training.transformer.nKVHeadsOverride=" << trainingConfig.transformer.nKVHeadsOverride << "\n";
		out << "training.transformer.dFFOverride=" << trainingConfig.transformer.dFFOverride << "\n";
		out << "training.transformer.enableTokenEmbedding=" << (trainingConfig.transformer.enableTokenEmbedding ? 1 : 0) << "\n";
		out << "training.transformer.vocabSizeOverride=" << trainingConfig.transformer.vocabSizeOverride << "\n";
		out << "training.transformer.tieEmbeddings=" << (trainingConfig.transformer.tieEmbeddings ? 1 : 0) << "\n";
		out << "training.transformer.padTokenId=" << trainingConfig.transformer.padTokenId << "\n";
		out << "training.transformer.tokenLmLossKind=" << static_cast<int>(trainingConfig.transformer.tokenLmLossKind) << "\n";
		out << "training.transformer.tokenLmSampledNegatives=" << trainingConfig.transformer.tokenLmSampledNegatives << "\n";
		out << "training.transformer.tokenLmAllowHugeFullSoftmax=" << (trainingConfig.transformer.tokenLmAllowHugeFullSoftmax ? 1 : 0) << "\n";
		out << "training.transformer.layerNormEps=" << trainingConfig.transformer.layerNormEps << "\n";
		out << "training.transformer.normType=" << static_cast<int>(trainingConfig.transformer.normType) << "\n";
		out << "training.transformer.positionalEncoding=" << static_cast<int>(trainingConfig.transformer.positionalEncoding) << "\n";
		out << "training.transformer.kvCacheDType=" << static_cast<int>(trainingConfig.transformer.kvCacheDType) << "\n";
		out << "training.transformer.ropeDimOverride=" << trainingConfig.transformer.ropeDimOverride << "\n";
		out << "training.transformer.ropeTheta=" << trainingConfig.transformer.ropeTheta << "\n";
		out << "training.transformer.ffnKind=" << static_cast<int>(trainingConfig.transformer.ffnKind) << "\n";
		out << "training.transformer.ffnActivation=" << static_cast<int>(trainingConfig.transformer.ffnActivation) << "\n";

		// Tokenizer artifacts (optional)
		out << "tokenizer.present=" << (tokenizerArtifactsPresent ? 1 : 0) << "\n";
		if (tokenizerArtifactsPresent)
		{
			out << "tokenizer.formatVersion=" << kTokenizerFormatVersion << "\n";
			out << "tokenizer.type=" << tokenizerArtifacts.type << "\n";
			out << "tokenizer.vocabCount=" << static_cast<unsigned long long>(tokenizerArtifacts.vocab.size()) << "\n";
			out << "tokenizer.fnv1a64=" << static_cast<unsigned long long>(vocabHash) << "\n";
			out << "tokenizer.special.padTokenId=" << tokenizerArtifacts.padTokenId << "\n";
			out << "tokenizer.special.bosTokenId=" << tokenizerArtifacts.bosTokenId << "\n";
			out << "tokenizer.special.eosTokenId=" << tokenizerArtifacts.eosTokenId << "\n";
			out << "tokenizer.special.unkTokenId=" << tokenizerArtifacts.unkTokenId << "\n";
		}
		out.flush();
		out.close();
		if (!out || !rename_atomic(tmp, manifestPath))
		{
			(void)::remove(tmp.c_str());
			return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveModel: failed to publish manifest.txt (rename failed)");
		}
	}

	// 5) Atomically publish temp dir -> final dir (rotate existing).
	std::string backupDirNoSlash;
	if (stat_is_dir(finalDirNoSlash))
	{
		std::ostringstream bak;
		bak << modelName << ".bak_" << pid << "_" << t;
		backupDirNoSlash = std::string("database/models/") + bak.str();
		if (!rename_atomic(finalDirNoSlash, backupDirNoSlash))
			return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveModel: unable to rotate existing model directory");
	}
	if (!rename_atomic(tmpDirNoSlash, finalDirNoSlash))
	{
		if (!backupDirNoSlash.empty())
			(void)rename_atomic(backupDirNoSlash, finalDirNoSlash);
		return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveModel: unable to publish model directory (rename failed)");
	}
	tmpGuard.dismiss();
	if (!backupDirNoSlash.empty())
		(void)remove_tree_recursive(backupDirNoSlash);

	return NNetworkStatus(NNetworkStatus::OK, std::string());
}

NNetworkStatus NNetwork::loadModel(const std::string& modelName, const DataInput* forShape, int netTypeOverride)
{
	if (modelName.empty())
		return failStatus(NNetworkStatus::INVALID_ARGUMENT, "loadModel: modelName is empty");
	if (!is_safe_path_component(modelName))
		return failStatus(NNetworkStatus::INVALID_ARGUMENT, "loadModel: modelName contains unsafe characters");
	if (!forShape)
		return failStatus(NNetworkStatus::INVALID_ARGUMENT, "loadModel: forShape is null (required to rebuild input layer)");

	const std::string dir = model_dir(modelName);
	const std::string manifestPath = dir + "manifest.txt";
	const std::string nninfoPath = dir + "nninfo.csv";
	const std::string weightsPath = dir + "weights.bin";
	const std::string tokDir = dir + "tokenizer/";
	const std::string tokManifestPath = tokDir + "manifest.txt";
	const std::string tokVocabPath = tokDir + "vocab.bin";

	// Modern-only: unified model package must exist.
	{
		std::ifstream probe(manifestPath.c_str());
		if (!probe)
			return failStatus(NNetworkStatus::INVALID_STATE, "loadModel: manifest.txt not found (legacy formats are not supported)");
	}
	if (!stat_is_file(manifestPath) || !stat_is_file(nninfoPath) || !stat_is_file(weightsPath))
		return failStatus(NNetworkStatus::INVALID_STATE, "loadModel: model package files missing or not regular files");

	// Unified package load.
	ModelManifest mf;
	if (!read_kv_manifest(manifestPath, mf))
		return failStatus(NNetworkStatus::INTERNAL_ERROR, "loadModel: unable to read manifest.txt");

	if (mf.magic != "GLADES_MODEL")
		return failStatus(NNetworkStatus::INVALID_STATE, "loadModel: manifest magic mismatch");

	if (mf.version < kModelFormatVersionMinSupported || mf.version > kModelFormatVersionLatest)
	{
		std::ostringstream oss;
		oss << "loadModel: unsupported model format version " << mf.version << " (supported "
		    << kModelFormatVersionMinSupported << ".." << kModelFormatVersionLatest << ")";
		return failStatus(NNetworkStatus::INVALID_STATE, oss.str());
	}
	// Best-effort consistency check: if manifest name exists it must match the directory component.
	if (!mf.name.empty() && mf.name != modelName)
		return failStatus(NNetworkStatus::INVALID_STATE, "loadModel: manifest name mismatch vs requested modelName");

	int savedNetType = mf.netType;
	if (savedNetType < 0)
		savedNetType = TYPE_DFF;

	if (netTypeOverride >= 0)
		netType = netTypeOverride;
	else
		netType = savedNetType;

	// Load architecture from nninfo.csv (self-contained, file-based).
	{
		const shmea::GTable t(shmea::GString(nninfoPath.c_str()), ',', shmea::GTable::TYPE_FILE);
		ownedSkeleton = shmea::GPointer<NNInfo>(new NNInfo(shmea::GString(modelName.c_str()), t));
		skeleton = ownedSkeleton.get();
	}

	if (!skeleton)
		return failStatus(NNetworkStatus::INTERNAL_ERROR, "loadModel: failed to construct NNInfo from nninfo.csv");

	// Load packed tensor weights (format v2+).
	const NNetworkStatus stW = loadTensorWeightsFromFile(weightsPath);
	if (!stW.ok())
		return stW;

	// Restore metadata/config from manifest (best-effort; missing keys keep defaults).
	if (mf.hasEpochs)
		epochs = mf.epochs;
	if (mf.hasSeed)
		setSeed(mf.rngSeed);
	if (mf.hasTrainingConfig)
		trainingConfig = mf.trainingConfig;

	// For transformers, the TrainingConfig is part of the inference contract:
	// positional encoding, norm type, RoPE parameters, and token-LM settings must match training.
	if ((netType == TYPE_TRANSFORMER_ENCODER || netType == TYPE_TRANSFORMER_DECODER) && !mf.hasTrainingConfig)
		return failStatus(NNetworkStatus::INVALID_STATE, "loadModel: transformer model package missing TrainingConfig (cannot guarantee inference correctness)");

	// Load tokenizer artifacts if present (optional).
	// This is strict: if tokenizer/manifest.txt exists, it must be valid and consistent.
	clearTokenizerArtifacts();
	if (mf.hasTokenizerPresent)
	{
		const bool tokOnDisk = stat_is_file(tokManifestPath);
		if (mf.tokenizerPresent && !tokOnDisk)
			return failStatus(NNetworkStatus::INVALID_STATE, "loadModel: manifest claims tokenizer.present=1 but tokenizer/manifest.txt is missing");
		if (!mf.tokenizerPresent && tokOnDisk)
			return failStatus(NNetworkStatus::INVALID_STATE, "loadModel: manifest claims tokenizer.present=0 but tokenizer/manifest.txt exists");
	}
	if (stat_is_file(tokManifestPath))
	{
		if (!stat_is_file(tokVocabPath))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadModel: tokenizer vocab.bin missing or not a regular file");

		TokenizerManifest tm;
		if (!read_tokenizer_manifest(tokManifestPath, tm))
			return failStatus(NNetworkStatus::INTERNAL_ERROR, "loadModel: unable to read tokenizer manifest.txt");
		if (tm.magic != "GLADES_TOKENIZER")
			return failStatus(NNetworkStatus::INVALID_STATE, "loadModel: tokenizer manifest magic mismatch");
		if (tm.version != kTokenizerFormatVersion)
		{
			std::ostringstream oss;
			oss << "loadModel: unsupported tokenizer format version " << tm.version << " (expected " << kTokenizerFormatVersion << ")";
			return failStatus(NNetworkStatus::INVALID_STATE, oss.str());
		}
		if (!is_safe_relative_file(tm.vocabFile) || tm.vocabFile != "vocab.bin")
			return failStatus(NNetworkStatus::INVALID_STATE, "loadModel: tokenizer vocabFile must be 'vocab.bin'");

		std::vector<std::string> vocab;
		uint64_t vocabHash = 0ULL;
		if (!read_vocab_bin(tokVocabPath, vocab, vocabHash))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadModel: failed to read tokenizer vocab.bin");
		if (tm.hasVocabCount && tm.vocabCount != static_cast<uint64_t>(vocab.size()))
			return failStatus(NNetworkStatus::INVALID_STATE, "loadModel: tokenizer vocabCount mismatch vs vocab.bin");
		if (tm.hasFNV && tm.fnv1a64 != vocabHash)
			return failStatus(NNetworkStatus::INVALID_STATE, "loadModel: tokenizer vocab.bin checksum mismatch (fnv1a64)");

		TokenizerArtifacts a;
		a.type = tm.type;
		a.vocab = vocab;
		a.padTokenId = tm.padTokenId;
		a.bosTokenId = tm.bosTokenId;
		a.eosTokenId = tm.eosTokenId;
		a.unkTokenId = tm.unkTokenId;
		const NNetworkStatus stTok = setTokenizerArtifacts(a);
		if (!stTok.ok())
			return stTok;
	}

	// Strict compatibility checks for token LM packaging (end-to-end correctness).
	if (netType == TYPE_TRANSFORMER_ENCODER || netType == TYPE_TRANSFORMER_DECODER)
	{
		const bool tokenModel = tensorTransformer.initialized && tensorTransformer.tokenModel;
		if (tokenModel)
		{
			if (!trainingConfig.transformer.enableTokenEmbedding)
				return failStatus(NNetworkStatus::INVALID_STATE, "loadModel: weights indicate token LM mode but manifest TrainingConfig has enableTokenEmbedding=0");

			// Vocab + padding invariants.
			if (trainingConfig.transformer.vocabSizeOverride > 0 &&
			    static_cast<unsigned int>(trainingConfig.transformer.vocabSizeOverride) != tensorTransformer.vocabSize)
			{
				return failStatus(NNetworkStatus::INVALID_STATE, "loadModel: vocabSizeOverride mismatch vs loaded weights");
			}
			if (trainingConfig.transformer.padTokenId != tensorTransformer.padTokenId)
				return failStatus(NNetworkStatus::INVALID_STATE, "loadModel: padTokenId mismatch vs loaded weights");
			if (trainingConfig.transformer.tieEmbeddings != tensorTransformer.tieEmbeddings)
				return failStatus(NNetworkStatus::INVALID_STATE, "loadModel: tieEmbeddings mismatch vs loaded weights");

			// Tokenizer locking: if tokenizer artifacts exist, they must match vocab size.
			if (hasTokenizerArtifacts() && tokenizerArtifacts.vocab.size() != static_cast<size_t>(tensorTransformer.vocabSize))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadModel: tokenizer vocab size does not match model vocab size");

			// If both trainingConfig and tokenizer provide padTokenId, they must match.
			if (hasTokenizerArtifacts() && tokenizerArtifacts.padTokenId >= 0 && trainingConfig.transformer.padTokenId >= 0 &&
			    tokenizerArtifacts.padTokenId != trainingConfig.transformer.padTokenId)
			{
				return failStatus(NNetworkStatus::INVALID_STATE, "loadModel: tokenizer.padTokenId mismatch vs trainingConfig.padTokenId");
			}
		}
	}

	// Optional file-integrity verification when the manifest provides checksums/sizes.
	// This is intentionally opt-in so unit tests and power users can patch weights on disk
	// (for deterministic override scenarios) without having to rewrite the manifest.
	//
	// Enable by setting:
	//   GLADES_MODEL_VERIFY_FILES=1
	{
		const char* v = ::getenv("GLADES_MODEL_VERIFY_FILES");
		const bool verify = (v && std::strcmp(v, "1") == 0);

		if (verify && mf.hasWeightsBytes && mf.hasWeightsFNV)
		{
			struct stat st;
#ifdef _WIN32
			if (::stat(weightsPath.c_str(), &st) != 0 || !S_ISREG(st.st_mode))
#else
			if (::lstat(weightsPath.c_str(), &st) != 0 || !S_ISREG(st.st_mode))
#endif
				return failStatus(NNetworkStatus::INVALID_STATE, "loadModel: weights.bin missing or not a regular file");
			const uint64_t bytes = static_cast<uint64_t>(st.st_size);
			if (bytes != mf.weightsBytes)
				return failStatus(NNetworkStatus::INVALID_STATE, "loadModel: weights.bin size mismatch vs manifest");
			uint64_t h = 0ULL;
			if (!fnv1a64_hash_file_prefix(weightsPath, bytes, h) || h != mf.weightsFNV1a64)
				return failStatus(NNetworkStatus::INVALID_STATE, "loadModel: weights.bin checksum mismatch (fnv1a64)");
		}

		if (verify && mf.hasNninfoBytes)
		{
			struct stat st;
#ifdef _WIN32
			if (::stat(nninfoPath.c_str(), &st) != 0 || !S_ISREG(st.st_mode))
#else
			if (::lstat(nninfoPath.c_str(), &st) != 0 || !S_ISREG(st.st_mode))
#endif
				return failStatus(NNetworkStatus::INVALID_STATE, "loadModel: nninfo.csv missing or not a regular file");
			const uint64_t bytes = static_cast<uint64_t>(st.st_size);
			if (bytes != mf.nninfoBytes)
				return failStatus(NNetworkStatus::INVALID_STATE, "loadModel: nninfo.csv size mismatch vs manifest");
		}

		// Tokenizer integrity duplicated in top-level manifest: verify agreement if present.
		if (verify && mf.hasTokenizerPresent && mf.tokenizerPresent)
		{
			if (!hasTokenizerArtifacts())
				return failStatus(NNetworkStatus::INVALID_STATE, "loadModel: manifest claims tokenizer.present=1 but tokenizer artifacts were not loaded");
			if (mf.hasTokenizerVocabCount && mf.tokenizerVocabCount != static_cast<uint64_t>(tokenizerArtifacts.vocab.size()))
				return failStatus(NNetworkStatus::INVALID_STATE, "loadModel: tokenizer vocabCount mismatch vs loaded artifacts");
			if (mf.hasTokenizerFNV)
			{
				std::vector<std::string> tmpVocab;
				uint64_t h = 0ULL;
				if (!read_vocab_bin(tokVocabPath, tmpVocab, h))
					return failStatus(NNetworkStatus::INVALID_STATE, "loadModel: failed to re-read tokenizer vocab.bin for verification");
				if (h != mf.tokenizerFNV1a64)
					return failStatus(NNetworkStatus::INVALID_STATE, "loadModel: tokenizer vocab.bin checksum mismatch vs top-level manifest");
			}
		}
	}

	return NNetworkStatus(NNetworkStatus::OK, std::string());
}

} // namespace glades

