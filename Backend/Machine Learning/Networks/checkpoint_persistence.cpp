// Copyright 2026
//
// Scalable, resumable training checkpoint persistence for Glades ML.
//
// A "checkpoint package" is stored at:
//   database/checkpoints/<checkpointName>/{manifest.txt, nninfo.csv, shard_000.bin, shard_001.bin, ...}
//
// Goals:
// - Sharded tensor storage for very large models
// - Optional optimizer state persistence (SGD momentum, AdamW moments)
// - Strict validation by tensor name + element count
// - Corruption detection via streaming FNV-1a checksums per tensor and per shard
//
// NOTE: This is separate from saveModel/loadModel. Model packages are geared toward deployment
// and omit optimizer state; checkpoints are for resuming training.

#include "network.h"
#include "atlas_optimizer.h"
#include "logfmt_utils.h"
#include "Backend/Database/GLogger.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <limits>
#include <memory>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <vector>

namespace {

using namespace glades::logfmt;

// Checkpoint manifest format (current and only):
// - version=1
// - explicit file byte order + per-tensor dtype + per-tensor shape metadata
// - strict validation on load
static const int kCheckpointFormatVersion = 1;

static const char* status_code_name(glades::NNetworkStatus::Code code)
{
	switch (code)
	{
	case glades::NNetworkStatus::OK: return "OK";
	case glades::NNetworkStatus::INVALID_ARGUMENT: return "INVALID_ARGUMENT";
	case glades::NNetworkStatus::INVALID_STATE: return "INVALID_STATE";
	case glades::NNetworkStatus::INTERNAL_ERROR: return "INTERNAL_ERROR";
	default: return "UNKNOWN";
	}
}

static void emit_logger_line(shmea::GLogger* logger,
                             int level,
                             const char* component,
                             const std::string& line)
{
	if (!logger)
		return;
	switch (level)
	{
	case shmea::GLogger::LOG_DEBUG:
		logger->debug(component, shmea::GString(line.c_str()));
		break;
	case shmea::GLogger::LOG_WARNING:
		logger->warning(component, shmea::GString(line.c_str()));
		break;
	case shmea::GLogger::LOG_ERROR:
		logger->error(component, shmea::GString(line.c_str()));
		break;
	case shmea::GLogger::LOG_FATAL:
		logger->fatal(component, shmea::GString(line.c_str()));
		break;
	case shmea::GLogger::LOG_VERBOSE:
		logger->verbose(component, shmea::GString(line.c_str()));
		break;
	case shmea::GLogger::LOG_INFO:
	default:
		logger->info(component, shmea::GString(line.c_str()));
		break;
	}
}

static void log_checkpoint_publish_event(const glades::NNetwork* net,
                                         int level,
                                         const char* event,
                                         const char* stage,
                                         const std::string& checkpointName,
                                         int netType,
                                         bool includeOptimizerState,
                                         bool rotatedPrevious,
                                         uint64_t shardCount,
                                         uint64_t tensorCount,
                                         uint64_t maxShardBytes,
                                         const glades::NNetworkStatus* st)
{
	if (!net)
		return;
	shmea::GLogger* logger = net->getLogger();
	if (!logger)
		return;

	std::ostringstream oss;
	oss << "event=" << (event ? event : "checkpoint_publish_event");
	append_logfmt_kv(oss, "operation", std::string("save_checkpoint"));
	append_logfmt_kv(oss, "checkpoint_name", checkpointName);
	append_logfmt_kv(oss, "net_type", netType);
	append_logfmt_kv(oss, "stage", std::string(stage ? stage : "unknown"));
	append_logfmt_kv(oss, "include_optimizer_state", includeOptimizerState);
	append_logfmt_kv(oss, "rotated_previous", rotatedPrevious);
	append_logfmt_kv(oss, "max_shard_bytes", static_cast<unsigned long long>(maxShardBytes));
	append_logfmt_kv(oss, "shard_count", static_cast<unsigned long long>(shardCount));
	append_logfmt_kv(oss, "tensor_count", static_cast<unsigned long long>(tensorCount));
	if (st)
	{
		append_logfmt_kv(oss, "status_code", std::string(status_code_name(st->code)));
		append_logfmt_kv(oss, "status_ok", st->ok());
		if (!st->message.empty())
			append_logfmt_kv(oss, "error", st->message);
	}

	emit_logger_line(logger, level, "CheckpointPersist", oss.str());
}

static inline bool is_path_separator(char c)
{
	return (c == '/') || (c == '\\');
}

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
	if (::stat(path.c_str(), &st) != 0)
		return false;
	return S_ISDIR(st.st_mode) != 0;
}

static bool stat_is_file(const std::string& path)
{
	struct stat st;
	if (::stat(path.c_str(), &st) != 0)
		return false;
	return S_ISREG(st.st_mode) != 0;
}

static bool mkdir_if_missing_dir_strict(const std::string& path)
{
	if (path.empty())
		return false;
	if (::mkdir(path.c_str(), 0777) == 0)
		return true;
	if (errno == EEXIST)
		return stat_is_dir(path);
	return false;
}

// Minimal mkdir -p helper (POSIX paths only). Rejects ".." traversal.
static bool mkdirs_recursive(const std::string& path)
{
	if (path.empty())
		return false;
	if (stat_is_dir(path))
		return true;

	std::string cur;
	cur.reserve(path.size());
	size_t i = 0u;
	if (path[0] == '/')
	{
		cur.push_back('/');
		i = 1u;
	}
	while (i < path.size())
	{
		while (i < path.size() && path[i] == '/')
			++i;
		if (i >= path.size())
			break;
		size_t j = path.find('/', i);
		if (j == std::string::npos)
			j = path.size();
		const std::string part = path.substr(i, j - i);
		if (part.empty() || part == ".")
		{
			i = j;
			continue;
		}
		if (part == "..")
			return false;
		if (!cur.empty() && cur[cur.size() - 1u] != '/')
			cur.push_back('/');
		cur.append(part);
		if (!mkdir_if_missing_dir_strict(cur))
			return false;
		i = j;
	}
	return stat_is_dir(path);
}

static bool is_little_endian_host()
{
	const unsigned int x = 1u;
	return (*reinterpret_cast<const unsigned char*>(&x)) == 1u;
}

static uint64_t fnv1a64_init()
{
	return 1469598103934665603ULL;
}

static uint64_t fnv1a64_update(uint64_t h, const unsigned char* data, size_t n)
{
	const uint64_t prime = 1099511628211ULL;
	for (size_t i = 0; i < n; ++i)
	{
		h ^= static_cast<uint64_t>(data[i]);
		h *= prime;
	}
	return h;
}

static std::string u64_to_string(uint64_t v)
{
	std::ostringstream oss;
	oss << static_cast<unsigned long long>(v);
	return oss.str();
}

static std::string u64_list_to_string(const std::vector<uint64_t>& v)
{
	std::ostringstream oss;
	for (size_t i = 0; i < v.size(); ++i)
	{
		if (i)
			oss << ",";
		oss << static_cast<unsigned long long>(v[i]);
	}
	return oss.str();
}

static bool parse_u64_list_value(const std::string& s, std::vector<uint64_t>& out)
{
	out.clear();
	if (s.empty())
		return false;
	size_t pos = 0u;
	while (pos < s.size())
	{
		size_t comma = s.find(',', pos);
		if (comma == std::string::npos)
			comma = s.size();
		const std::string tok = s.substr(pos, comma - pos);
		if (tok.empty())
			return false;
		char* end = NULL;
		errno = 0;
		const unsigned long long v = ::strtoull(tok.c_str(), &end, 10);
		if (errno != 0 || end == tok.c_str() || (end && *end != '\0'))
			return false;
		out.push_back(static_cast<uint64_t>(v));
		pos = (comma < s.size()) ? (comma + 1u) : s.size();
	}
	return !out.empty();
}

static bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t& out)
{
	if (a == 0u || b == 0u)
	{
		out = 0u;
		return true;
	}
	if (a > (std::numeric_limits<uint64_t>::max() / b))
		return false;
	out = a * b;
	return true;
}

static bool checked_shape_elem_count(const std::vector<uint64_t>& shape, uint64_t& outCount)
{
	outCount = 0u;
	if (shape.empty())
		return false;
	uint64_t prod = 1u;
	for (size_t i = 0; i < shape.size(); ++i)
	{
		const uint64_t d = shape[i];
		// Allow zero-sized tensors: if any dimension is 0, the element count is 0.
		// (Validated by comparing to the manifest `count` field.)
		if (d == 0u)
		{
			outCount = 0u;
			return true;
		}
		uint64_t tmp = 0u;
		if (!checked_mul_u64(prod, d, tmp))
			return false;
		prod = tmp;
	}
	outCount = prod;
	return true;
}

// Root directory is configurable for production deployments.
// Default preserves legacy behavior.
static std::string checkpoint_root_dir()
{
	const char* env = ::getenv("GLADES_CHECKPOINT_ROOT");
	if (env && env[0] != '\0')
		return std::string(env);
	return std::string("database/checkpoints");
}

static std::string join_dir(const std::string& a, const std::string& b)
{
	if (a.empty())
		return b;
	if (b.empty())
		return a;
	if (a[a.size() - 1u] == '/')
		return a + b;
	return a + "/" + b;
}

static bool ensure_checkpoint_root()
{
	return mkdirs_recursive(checkpoint_root_dir());
}

static std::string checkpoint_dir(const std::string& checkpointName)
{
	// Always return a directory path with a trailing '/' because call sites append filenames by concatenation.
	const std::string base = join_dir(checkpoint_root_dir(), checkpointName);
	if (!base.empty() && base[base.size() - 1u] == '/')
		return base;
	return base + "/";
}

static bool is_safe_shard_filename(const std::string& file)
{
	// Must be a basename only.
	for (size_t i = 0; i < file.size(); ++i)
		if (is_path_separator(file[i]))
			return false;
	// shard_000.bin pattern
	if (file.size() < strlen("shard_000.bin"))
		return false;
	if (file.compare(0, 6, "shard_") != 0)
		return false;
	if (file.compare(file.size() - 4u, 4u, ".bin") != 0)
		return false;
	for (size_t i = 6; i + 4u < file.size(); ++i)
	{
		const char c = file[i];
		if (c < '0' || c > '9')
			return false;
	}
	return true;
}

static bool rename_atomic(const std::string& from, const std::string& to)
{
	return ::rename(from.c_str(), to.c_str()) == 0;
}

static bool remove_tree_recursive(const std::string& path)
{
	DIR* d = ::opendir(path.c_str());
	if (!d)
	{
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
		const std::string child = join_dir(path, std::string(name));
		struct stat st;
		if (::lstat(child.c_str(), &st) != 0)
			continue;
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
}

static bool write_kv(std::ostream& out, const std::string& k, const std::string& v)
{
	out << k << "=" << v << "\n";
	return static_cast<bool>(out);
}

static bool read_kv_file(const std::string& manifestPath, std::map<std::string, std::string>& kvOut)
{
	kvOut.clear();
	std::ifstream in(manifestPath.c_str());
	if (!in)
		return false;
	std::string line;
	while (std::getline(in, line))
	{
		if (line.empty())
			continue;
		const size_t eq = line.find('=');
		if (eq == std::string::npos)
		{
			// allow magic line without '='
			if (kvOut.find("__magic__") == kvOut.end())
				kvOut["__magic__"] = line;
			continue;
		}
		const std::string key = line.substr(0, eq);
		const std::string val = line.substr(eq + 1);
		kvOut[key] = val;
	}
	return true;
}

static bool parse_int(const std::map<std::string, std::string>& kv, const std::string& key, int& out)
{
	std::map<std::string, std::string>::const_iterator it = kv.find(key);
	if (it == kv.end())
		return false;
	const std::string& s = it->second;
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

static bool parse_u64(const std::map<std::string, std::string>& kv, const std::string& key, uint64_t& out)
{
	std::map<std::string, std::string>::const_iterator it = kv.find(key);
	if (it == kv.end())
		return false;
	const std::string& s = it->second;
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

static bool parse_size_t(const std::map<std::string, std::string>& kv, const std::string& key, size_t& out)
{
	uint64_t tmp = 0u;
	if (!parse_u64(kv, key, tmp))
		return false;
	if (tmp > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
		return false;
	out = static_cast<size_t>(tmp);
	return true;
}

static bool should_verify_shards()
{
	const char* v = ::getenv("GLADES_CHECKPOINT_VERIFY_SHARDS");
	if (!v)
		return false;
	if (std::strcmp(v, "1") == 0)
		return true;
	if (std::strcmp(v, "true") == 0)
		return true;
	if (std::strcmp(v, "TRUE") == 0)
		return true;
	return false;
}

static bool fnv1a64_hash_file_prefix(const std::string& path, uint64_t bytesToHash, uint64_t& out)
{
	out = 1469598103934665603ull;
	std::ifstream in(path.c_str(), std::ios::in | std::ios::binary);
	if (!in)
		return false;
	static const size_t kBuf = 1u << 16;
	char buf[kBuf];
	uint64_t remaining = bytesToHash;
	while (remaining > 0u)
	{
		const size_t want = (remaining > static_cast<uint64_t>(kBuf)) ? kBuf : static_cast<size_t>(remaining);
		in.read(buf, static_cast<std::streamsize>(want));
		const std::streamsize got = in.gcount();
		if (got <= 0)
			return false;
		out = fnv1a64_update(out, reinterpret_cast<const unsigned char*>(buf), static_cast<size_t>(got));
		remaining -= static_cast<uint64_t>(got);
	}
	return true;
}

static bool parse_bool01(const std::map<std::string, std::string>& kv, const std::string& key, bool& out)
{
	int x = 0;
	if (!parse_int(kv, key, x))
		return false;
	out = (x != 0);
	return true;
}

static bool parse_float(const std::map<std::string, std::string>& kv, const std::string& key, float& out)
{
	std::map<std::string, std::string>::const_iterator it = kv.find(key);
	if (it == kv.end())
		return false;
	const std::string& s = it->second;
	if (s.empty())
		return false;
	char* end = NULL;
	errno = 0;
	const float v = ::strtof(s.c_str(), &end);
	if (errno != 0 || end == s.c_str() || (end && *end != '\0'))
		return false;
	if (!std::isfinite(v))
		return false;
	out = v;
	return true;
}

static void apply_training_config_from_kv(const std::map<std::string, std::string>& kv, glades::TrainingConfig& cfg, bool& any)
{
	any = false;
	// Keep defaults for missing keys.
	int i = 0;
	float f = 0.0f;
	bool b = false;

	if (parse_int(kv, "training.minibatchSizeOverride", i)) { cfg.minibatchSizeOverride = i; any = true; }
	if (parse_int(kv, "training.tbpttWindowOverride", i)) { cfg.tbpttWindowOverride = i; any = true; }
	if (parse_float(kv, "training.globalGradClipNorm", f)) { cfg.globalGradClipNorm = f; any = true; }
	if (parse_float(kv, "training.perElementGradClip", f)) { cfg.perElementGradClip = f; any = true; }
	if (parse_int(kv, "training.optimizer.type", i)) { cfg.optimizer.type = static_cast<glades::OptimizerConfig::Type>(i); any = true; }
	if (parse_float(kv, "training.optimizer.adamBeta1", f)) { cfg.optimizer.adamBeta1 = f; any = true; }
	if (parse_float(kv, "training.optimizer.adamBeta2", f)) { cfg.optimizer.adamBeta2 = f; any = true; }
	if (parse_float(kv, "training.optimizer.adamEps", f)) { cfg.optimizer.adamEps = f; any = true; }
	if (parse_bool01(kv, "training.optimizer.adamBiasCorrection", b)) { cfg.optimizer.adamBiasCorrection = b; any = true; }
	if (parse_int(kv, "training.atlas.rank", i) && i >= 0) { cfg.atlas.rank = static_cast<unsigned int>(i); any = true; }
	if (parse_int(kv, "training.atlas.tSub", i) && i >= 0) { cfg.atlas.tSub = static_cast<unsigned int>(i); any = true; }

	if (parse_int(kv, "training.lrSchedule.type", i)) { cfg.lrSchedule.type = static_cast<glades::LearningRateScheduleConfig::Type>(i); any = true; }
	if (parse_int(kv, "training.lrSchedule.stepSizeEpochs", i)) { cfg.lrSchedule.stepSizeEpochs = i; any = true; }
	if (parse_float(kv, "training.lrSchedule.gamma", f)) { cfg.lrSchedule.gamma = f; any = true; }
	if (parse_int(kv, "training.lrSchedule.cosineTMaxEpochs", i)) { cfg.lrSchedule.cosineTMaxEpochs = i; any = true; }
	if (parse_float(kv, "training.lrSchedule.minMultiplier", f)) { cfg.lrSchedule.minMultiplier = f; any = true; }

	// Mixed precision (optional; missing keys keep defaults)
	if (parse_bool01(kv, "training.mixedPrecision.enable", b)) { cfg.mixedPrecision.enable = b; any = true; }
	if (parse_int(kv, "training.mixedPrecision.weightDType", i)) { cfg.mixedPrecision.weightDType = static_cast<glades::MixedPrecisionConfig::WeightDType>(i); any = true; }
	if (parse_bool01(kv, "training.mixedPrecision.useLossScaling", b)) { cfg.mixedPrecision.useLossScaling = b; any = true; }
	if (parse_bool01(kv, "training.mixedPrecision.dynamicLossScaling", b)) { cfg.mixedPrecision.dynamicLossScaling = b; any = true; }
	if (parse_float(kv, "training.mixedPrecision.lossScaleInit", f)) { cfg.mixedPrecision.lossScaleInit = f; any = true; }
	if (parse_float(kv, "training.mixedPrecision.lossScaleMin", f)) { cfg.mixedPrecision.lossScaleMin = f; any = true; }
	if (parse_float(kv, "training.mixedPrecision.lossScaleMax", f)) { cfg.mixedPrecision.lossScaleMax = f; any = true; }
	if (parse_int(kv, "training.mixedPrecision.growthInterval", i)) { cfg.mixedPrecision.growthInterval = i; any = true; }
	if (parse_float(kv, "training.mixedPrecision.growthFactor", f)) { cfg.mixedPrecision.growthFactor = f; any = true; }
	if (parse_float(kv, "training.mixedPrecision.backoffFactor", f)) { cfg.mixedPrecision.backoffFactor = f; any = true; }

	// TransformerRunConfig
	if (parse_int(kv, "training.transformer.nHeadsOverride", i)) { cfg.transformer.nHeadsOverride = i; any = true; }
	if (parse_int(kv, "training.transformer.nKVHeadsOverride", i)) { cfg.transformer.nKVHeadsOverride = i; any = true; }
	if (parse_int(kv, "training.transformer.dFFOverride", i)) { cfg.transformer.dFFOverride = i; any = true; }
	if (parse_bool01(kv, "training.transformer.enableTokenEmbedding", b)) { cfg.transformer.enableTokenEmbedding = b; any = true; }
	if (parse_int(kv, "training.transformer.vocabSizeOverride", i)) { cfg.transformer.vocabSizeOverride = i; any = true; }
	if (parse_bool01(kv, "training.transformer.tieEmbeddings", b)) { cfg.transformer.tieEmbeddings = b; any = true; }
	if (parse_int(kv, "training.transformer.padTokenId", i)) { cfg.transformer.padTokenId = i; any = true; }
	if (parse_int(kv, "training.transformer.tokenLmLossKind", i)) { cfg.transformer.tokenLmLossKind = static_cast<glades::TransformerRunConfig::TokenLMLossKind>(i); any = true; }
	if (parse_int(kv, "training.transformer.tokenLmSampledNegatives", i)) { cfg.transformer.tokenLmSampledNegatives = i; any = true; }
	if (parse_bool01(kv, "training.transformer.tokenLmAllowHugeFullSoftmax", b)) { cfg.transformer.tokenLmAllowHugeFullSoftmax = b; any = true; }
	if (parse_float(kv, "training.transformer.layerNormEps", f)) { cfg.transformer.layerNormEps = f; any = true; }
	if (parse_int(kv, "training.transformer.normType", i)) { cfg.transformer.normType = static_cast<glades::TransformerRunConfig::NormType>(i); any = true; }
	if (parse_int(kv, "training.transformer.positionalEncoding", i)) { cfg.transformer.positionalEncoding = static_cast<glades::TransformerRunConfig::PositionalEncodingType>(i); any = true; }
	if (parse_int(kv, "training.transformer.kvCacheDType", i)) { cfg.transformer.kvCacheDType = static_cast<glades::TransformerRunConfig::KVCacheDType>(i); any = true; }
	if (parse_int(kv, "training.transformer.ropeDimOverride", i)) { cfg.transformer.ropeDimOverride = i; any = true; }
	if (parse_float(kv, "training.transformer.ropeTheta", f)) { cfg.transformer.ropeTheta = f; any = true; }
	if (parse_int(kv, "training.transformer.ffnKind", i)) { cfg.transformer.ffnKind = static_cast<glades::TransformerRunConfig::FFNKind>(i); any = true; }
	if (parse_int(kv, "training.transformer.ffnActivation", i)) { cfg.transformer.ffnActivation = static_cast<glades::TransformerRunConfig::FFNActivationType>(i); any = true; }
}

struct TensorWriteRef
{
	std::string name;
	const std::vector<float>* vec;
	// v2 metadata:
	// - dtype is a stable string identifier ("f32" today)
	// - shape is the logical tensor shape (row-major semantics for matrices).
	std::string dtype;
	std::vector<uint64_t> shape;
	TensorWriteRef() : name(), vec(NULL), dtype(), shape() {}
	TensorWriteRef(const std::string& n, const std::vector<float>* v, const std::string& dt, const std::vector<uint64_t>& sh)
	    : name(n), vec(v), dtype(dt), shape(sh)
	{
	}
};

struct TensorReadRef
{
	std::string name;
	std::vector<float>* vec;
	std::string dtype;
	std::vector<uint64_t> shape;
	TensorReadRef() : name(), vec(NULL), dtype(), shape() {}
	TensorReadRef(const std::string& n, std::vector<float>* v, const std::string& dt, const std::vector<uint64_t>& sh)
	    : name(n), vec(v), dtype(dt), shape(sh)
	{
	}
};

// NOTE: Tensor enumeration for checkpoint save/load is implemented inside the NNetwork
// member functions below so it can access the private packed-tensor state.

struct TensorEntry
{
	std::string name;
	// v1: count is float32 count.
	// v2: count is element count (dtype is explicit).
	uint64_t count;
	unsigned int shardIdx; // which shard file
	uint64_t offsetBytes;  // byte offset within shard
	uint64_t bytes;        // byte size in shard (count * elemBytes)
	uint64_t fnv1a64;      // checksum over tensor bytes
	// v2 metadata:
	uint64_t elemBytes;
	std::string dtype;
	std::vector<uint64_t> shape;
	TensorEntry() : name(), count(0u), shardIdx(0u), offsetBytes(0u), bytes(0u), fnv1a64(0u), elemBytes(0u), dtype(), shape() {}
};

struct ShardEntry
{
	std::string file;
	uint64_t bytes;
	uint64_t fnv1a64;
	ShardEntry() : file(), bytes(0u), fnv1a64(0u) {}
};

class CheckpointShardWriter
{
public:
	explicit CheckpointShardWriter(const std::string& dir, size_t maxBytes)
	    : baseDir(dir),
	      maxShardBytes(maxBytes ? maxBytes : static_cast<size_t>(1024ull * 1024ull * 1024ull)),
	      curShardIdx(0u),
	      curShardBytes(0u),
	      curShardHash(fnv1a64_init()),
	      out(),
	      shards(),
	      tensors(),
	      hostLittle(is_little_endian_host())
	{
	}

	bool open_first()
	{
		curShardIdx = 0u;
		return open_new_shard();
	}

	bool add_tensor(const std::string& name, const std::vector<float>& v, const std::string& dtype, const std::vector<uint64_t>& shape)
	{
		// Current implementation persists FP32 tensors only (raw little-endian).
		// v2 makes this explicit via dtype/elemBytes keys in the manifest.
		if (dtype != "f32")
			return false;
		const uint64_t elemBytes = 4ull;
		const size_t countSz = v.size();
		const uint64_t count = static_cast<uint64_t>(countSz);
		uint64_t bytes = 0u;
		if (!checked_mul_u64(count, elemBytes, bytes))
			return false;
		// If tensor doesn't fit in remaining space and shard is non-empty, start a new shard.
		if (curShardBytes > 0u && (static_cast<uint64_t>(maxShardBytes) > 0ull) &&
		    (curShardBytes + bytes > static_cast<uint64_t>(maxShardBytes)))
		{
			if (!finalize_current_shard())
				return false;
			++curShardIdx;
			if (!open_new_shard())
				return false;
		}

		const uint64_t offset = curShardBytes;
		const uint64_t tensorHash = write_f32_blob(v);
		if (!out)
			return false;

		TensorEntry te;
		te.name = name;
		te.count = count;
		te.shardIdx = curShardIdx;
		te.offsetBytes = offset;
		te.bytes = bytes;
		te.fnv1a64 = tensorHash;
		te.elemBytes = elemBytes;
		te.dtype = dtype;
		te.shape = shape;
		tensors.push_back(te);
		return true;
	}

	bool finalize_all()
	{
		return finalize_current_shard();
	}

	const std::vector<ShardEntry>& get_shards() const { return shards; }
	const std::vector<TensorEntry>& get_tensors() const { return tensors; }

private:
	std::string baseDir;
	size_t maxShardBytes;
	unsigned int curShardIdx;
	uint64_t curShardBytes;
	uint64_t curShardHash;
	std::ofstream out;
	std::vector<ShardEntry> shards;
	std::vector<TensorEntry> tensors;
	bool hostLittle;

	static std::string shard_filename(unsigned int idx)
	{
		std::ostringstream oss;
		oss << "shard_";
		oss.width(3);
		oss.fill('0');
		oss << idx;
		oss << ".bin";
		return oss.str();
	}

	static std::string shard_tmp_filename(unsigned int idx)
	{
		return shard_filename(idx) + ".tmp";
	}

	bool open_new_shard()
	{
		curShardBytes = 0u;
		curShardHash = fnv1a64_init();
		const std::string file = shard_filename(curShardIdx);
		const std::string tmpFile = shard_tmp_filename(curShardIdx);
		out.close();
		out.clear();
		out.open((baseDir + tmpFile).c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
		if (!out)
			return false;

		ShardEntry se;
		se.file = file;
		se.bytes = 0u;
		se.fnv1a64 = 0u;
		if (shards.size() <= curShardIdx)
			shards.push_back(se);
		else
			shards[curShardIdx] = se;
		return true;
	}

	bool finalize_current_shard()
	{
		if (!out && shards.empty())
			return false;
		if (out)
		{
			out.flush();
			out.close();
		}
		// Atomically publish shard: rename tmp -> final.
		{
			const std::string tmpPath = baseDir + shard_tmp_filename(curShardIdx);
			const std::string finalPath = baseDir + shard_filename(curShardIdx);
			if (!rename_atomic(tmpPath, finalPath))
				return false;
		}
		if (curShardIdx < shards.size())
		{
			shards[curShardIdx].bytes = curShardBytes;
			shards[curShardIdx].fnv1a64 = curShardHash;
		}
		return true;
	}

	uint64_t write_f32_blob(const std::vector<float>& v)
	{
		uint64_t tensorHash = fnv1a64_init();
		if (v.empty())
			return tensorHash;

		const unsigned char* bytes = reinterpret_cast<const unsigned char*>(&v[0]);
		const size_t totalBytes = v.size() * sizeof(float);

		// Stream in chunks to keep memory use small.
		const size_t kChunk = 4u * 1024u * 1024u; // 4 MiB
		size_t off = 0u;
		while (off < totalBytes)
		{
			const size_t n = (totalBytes - off > kChunk) ? kChunk : (totalBytes - off);
			if (hostLittle)
			{
				out.write(reinterpret_cast<const char*>(bytes + off), static_cast<std::streamsize>(n));
				tensorHash = fnv1a64_update(tensorHash, bytes + off, n);
				curShardHash = fnv1a64_update(curShardHash, bytes + off, n);
			}
			else
			{
				// Rare: big-endian host. Convert each float to little-endian bytes.
				const size_t floatsN = n / sizeof(float);
				for (size_t i = 0; i < floatsN; ++i)
				{
					unsigned int bits = 0u;
					std::memcpy(&bits, &v[(off / sizeof(float)) + i], sizeof(float));
					unsigned char b4[4];
					b4[0] = static_cast<unsigned char>(bits & 0xffu);
					b4[1] = static_cast<unsigned char>((bits >> 8) & 0xffu);
					b4[2] = static_cast<unsigned char>((bits >> 16) & 0xffu);
					b4[3] = static_cast<unsigned char>((bits >> 24) & 0xffu);
					out.write(reinterpret_cast<const char*>(b4), 4);
					tensorHash = fnv1a64_update(tensorHash, b4, 4);
					curShardHash = fnv1a64_update(curShardHash, b4, 4);
				}
			}
			curShardBytes += static_cast<uint64_t>(n);
			off += n;
		}
		return tensorHash;
	}
};

static bool write_manifest(const std::string& manifestPath,
                           const std::string& checkpointName,
                           int netType,
                           int epochs,
                           uint64_t rngSeed,
                           const glades::TrainingConfig& trainingConfig,
                           bool includeOptimizerState,
                           unsigned long long transformerOptimizerStep,
                           float transformerLossScale,
                           unsigned long long transformerLossScaleGoodSteps,
                           size_t maxShardBytes,
                           const std::vector<ShardEntry>& shards,
                           const std::vector<TensorEntry>& tensors,
                           const std::map<std::string, std::string>& extraKV = std::map<std::string, std::string>())
{
	// Atomic write: write to temp file then rename into place.
	const std::string tmpPath = manifestPath + ".tmp";
	std::ofstream out(tmpPath.c_str(), std::ios::out | std::ios::trunc);
	if (!out)
		return false;

	out << "GLADES_CHECKPOINT" << "\n";
	write_kv(out, "version", u64_to_string(static_cast<uint64_t>(kCheckpointFormatVersion)));
	write_kv(out, "name", checkpointName);
	write_kv(out, "netType", u64_to_string(static_cast<uint64_t>(netType)));
	write_kv(out, "epochs", u64_to_string(static_cast<uint64_t>(epochs)));
	write_kv(out, "rngSeed", u64_to_string(rngSeed));
	write_kv(out, "includeOptimizerState", includeOptimizerState ? "1" : "0");
	write_kv(out, "maxShardBytes", u64_to_string(static_cast<uint64_t>(maxShardBytes)));
	write_kv(out, "shardCount", u64_to_string(static_cast<uint64_t>(shards.size())));
	write_kv(out, "tensorCount", u64_to_string(static_cast<uint64_t>(tensors.size())));

	// v2: file-level encoding metadata (explicit and validated on load).
	// All shard blobs are raw little-endian encodings of the tensor element type.
	write_kv(out, "file.endian", "little");
	write_kv(out, "file.scalar.f32", "ieee754");
	write_kv(out, "file.tensorEncoding", "raw_le");

	// TrainingConfig (copied from saveModel manifest for compatibility).
	write_kv(out, "training.minibatchSizeOverride", u64_to_string(static_cast<uint64_t>(trainingConfig.minibatchSizeOverride)));
	write_kv(out, "training.tbpttWindowOverride", u64_to_string(static_cast<uint64_t>(trainingConfig.tbpttWindowOverride)));
	{
		std::ostringstream oss; oss << trainingConfig.globalGradClipNorm; write_kv(out, "training.globalGradClipNorm", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.perElementGradClip; write_kv(out, "training.perElementGradClip", oss.str());
	}
	write_kv(out, "training.optimizer.type", u64_to_string(static_cast<uint64_t>(static_cast<int>(trainingConfig.optimizer.type))));
	{
		std::ostringstream oss; oss << trainingConfig.optimizer.adamBeta1; write_kv(out, "training.optimizer.adamBeta1", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.optimizer.adamBeta2; write_kv(out, "training.optimizer.adamBeta2", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.optimizer.adamEps; write_kv(out, "training.optimizer.adamEps", oss.str());
	}
	write_kv(out, "training.optimizer.adamBiasCorrection", trainingConfig.optimizer.adamBiasCorrection ? "1" : "0");
	write_kv(out, "training.atlas.rank", u64_to_string(static_cast<uint64_t>(trainingConfig.atlas.rank)));
	write_kv(out, "training.atlas.tSub", u64_to_string(static_cast<uint64_t>(trainingConfig.atlas.tSub)));
	write_kv(out, "training.lrSchedule.type", u64_to_string(static_cast<uint64_t>(static_cast<int>(trainingConfig.lrSchedule.type))));
	write_kv(out, "training.lrSchedule.stepSizeEpochs", u64_to_string(static_cast<uint64_t>(trainingConfig.lrSchedule.stepSizeEpochs)));
	{
		std::ostringstream oss; oss << trainingConfig.lrSchedule.gamma; write_kv(out, "training.lrSchedule.gamma", oss.str());
	}
	write_kv(out, "training.lrSchedule.cosineTMaxEpochs", u64_to_string(static_cast<uint64_t>(trainingConfig.lrSchedule.cosineTMaxEpochs)));
	{
		std::ostringstream oss; oss << trainingConfig.lrSchedule.minMultiplier; write_kv(out, "training.lrSchedule.minMultiplier", oss.str());
	}

	// Mixed precision
	write_kv(out, "training.mixedPrecision.enable", trainingConfig.mixedPrecision.enable ? "1" : "0");
	write_kv(out, "training.mixedPrecision.weightDType", u64_to_string(static_cast<uint64_t>(static_cast<int>(trainingConfig.mixedPrecision.weightDType))));
	write_kv(out, "training.mixedPrecision.useLossScaling", trainingConfig.mixedPrecision.useLossScaling ? "1" : "0");
	write_kv(out, "training.mixedPrecision.dynamicLossScaling", trainingConfig.mixedPrecision.dynamicLossScaling ? "1" : "0");
	{
		std::ostringstream oss; oss << trainingConfig.mixedPrecision.lossScaleInit; write_kv(out, "training.mixedPrecision.lossScaleInit", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.mixedPrecision.lossScaleMin; write_kv(out, "training.mixedPrecision.lossScaleMin", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.mixedPrecision.lossScaleMax; write_kv(out, "training.mixedPrecision.lossScaleMax", oss.str());
	}
	write_kv(out, "training.mixedPrecision.growthInterval", u64_to_string(static_cast<uint64_t>(trainingConfig.mixedPrecision.growthInterval)));
	{
		std::ostringstream oss; oss << trainingConfig.mixedPrecision.growthFactor; write_kv(out, "training.mixedPrecision.growthFactor", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.mixedPrecision.backoffFactor; write_kv(out, "training.mixedPrecision.backoffFactor", oss.str());
	}

	// TransformerRunConfig
	write_kv(out, "training.transformer.nHeadsOverride", u64_to_string(static_cast<uint64_t>(trainingConfig.transformer.nHeadsOverride)));
	write_kv(out, "training.transformer.nKVHeadsOverride", u64_to_string(static_cast<uint64_t>(trainingConfig.transformer.nKVHeadsOverride)));
	write_kv(out, "training.transformer.dFFOverride", u64_to_string(static_cast<uint64_t>(trainingConfig.transformer.dFFOverride)));
	write_kv(out, "training.transformer.enableTokenEmbedding", trainingConfig.transformer.enableTokenEmbedding ? "1" : "0");
	write_kv(out, "training.transformer.vocabSizeOverride", u64_to_string(static_cast<uint64_t>(trainingConfig.transformer.vocabSizeOverride)));
	write_kv(out, "training.transformer.tieEmbeddings", trainingConfig.transformer.tieEmbeddings ? "1" : "0");
	write_kv(out, "training.transformer.padTokenId", u64_to_string(static_cast<uint64_t>(trainingConfig.transformer.padTokenId)));
	write_kv(out, "training.transformer.tokenLmLossKind", u64_to_string(static_cast<uint64_t>(static_cast<int>(trainingConfig.transformer.tokenLmLossKind))));
	write_kv(out, "training.transformer.tokenLmSampledNegatives", u64_to_string(static_cast<uint64_t>(trainingConfig.transformer.tokenLmSampledNegatives)));
	write_kv(out, "training.transformer.tokenLmAllowHugeFullSoftmax", trainingConfig.transformer.tokenLmAllowHugeFullSoftmax ? "1" : "0");
	{
		std::ostringstream oss; oss << trainingConfig.transformer.layerNormEps; write_kv(out, "training.transformer.layerNormEps", oss.str());
	}
	write_kv(out, "training.transformer.normType", u64_to_string(static_cast<uint64_t>(static_cast<int>(trainingConfig.transformer.normType))));
	write_kv(out, "training.transformer.positionalEncoding", u64_to_string(static_cast<uint64_t>(static_cast<int>(trainingConfig.transformer.positionalEncoding))));
	write_kv(out, "training.transformer.kvCacheDType", u64_to_string(static_cast<uint64_t>(static_cast<int>(trainingConfig.transformer.kvCacheDType))));
	write_kv(out, "training.transformer.ropeDimOverride", u64_to_string(static_cast<uint64_t>(trainingConfig.transformer.ropeDimOverride)));
	{
		std::ostringstream oss; oss << trainingConfig.transformer.ropeTheta; write_kv(out, "training.transformer.ropeTheta", oss.str());
	}
	write_kv(out, "training.transformer.ffnKind", u64_to_string(static_cast<uint64_t>(static_cast<int>(trainingConfig.transformer.ffnKind))));
	write_kv(out, "training.transformer.ffnActivation", u64_to_string(static_cast<uint64_t>(static_cast<int>(trainingConfig.transformer.ffnActivation))));

	// Transformer optimizer step (for Adam bias correction on resume).
	write_kv(out, "transformer.optimizerStep", u64_to_string(static_cast<uint64_t>(transformerOptimizerStep)));
	// Mixed precision loss scaling state (best-effort; safe to ignore if absent).
	{
		std::ostringstream oss; oss << transformerLossScale; write_kv(out, "transformer.lossScale", oss.str());
	}
	write_kv(out, "transformer.lossScaleGoodSteps", u64_to_string(static_cast<uint64_t>(transformerLossScaleGoodSteps)));

	// Extra key-value pairs (e.g. ATLAS optimizer scalars).
	for (std::map<std::string, std::string>::const_iterator eit = extraKV.begin(); eit != extraKV.end(); ++eit)
		write_kv(out, eit->first, eit->second);

	// Shards
	for (size_t i = 0; i < shards.size(); ++i)
	{
		std::ostringstream kf; kf << "shard." << static_cast<unsigned long long>(i) << ".file";
		std::ostringstream kb; kb << "shard." << static_cast<unsigned long long>(i) << ".bytes";
		std::ostringstream kh; kh << "shard." << static_cast<unsigned long long>(i) << ".fnv1a64";
		write_kv(out, kf.str(), shards[i].file);
		write_kv(out, kb.str(), u64_to_string(shards[i].bytes));
		write_kv(out, kh.str(), u64_to_string(shards[i].fnv1a64));
	}

	// Tensors
	for (size_t i = 0; i < tensors.size(); ++i)
	{
		std::ostringstream kn; kn << "tensor." << static_cast<unsigned long long>(i) << ".name";
		std::ostringstream kc; kc << "tensor." << static_cast<unsigned long long>(i) << ".count";
		std::ostringstream ks; ks << "tensor." << static_cast<unsigned long long>(i) << ".shard";
		std::ostringstream ko; ko << "tensor." << static_cast<unsigned long long>(i) << ".offsetBytes";
		std::ostringstream kb; kb << "tensor." << static_cast<unsigned long long>(i) << ".bytes";
		std::ostringstream kh; kh << "tensor." << static_cast<unsigned long long>(i) << ".fnv1a64";
		// v2 tensor metadata keys:
		std::ostringstream kd; kd << "tensor." << static_cast<unsigned long long>(i) << ".dtype";
		std::ostringstream ke; ke << "tensor." << static_cast<unsigned long long>(i) << ".elemBytes";
		std::ostringstream kr; kr << "tensor." << static_cast<unsigned long long>(i) << ".rank";
		std::ostringstream ksh; ksh << "tensor." << static_cast<unsigned long long>(i) << ".shape";
		std::ostringstream kl; kl << "tensor." << static_cast<unsigned long long>(i) << ".layout";
		write_kv(out, kn.str(), tensors[i].name);
		write_kv(out, kc.str(), u64_to_string(static_cast<uint64_t>(tensors[i].count)));
		write_kv(out, ks.str(), u64_to_string(static_cast<uint64_t>(tensors[i].shardIdx)));
		write_kv(out, ko.str(), u64_to_string(tensors[i].offsetBytes));
		write_kv(out, kb.str(), u64_to_string(tensors[i].bytes));
		write_kv(out, kh.str(), u64_to_string(tensors[i].fnv1a64));

		// Only written for v2+ manifests. (Older readers ignore unknown keys.)
		// If writer metadata is missing (should not happen), fall back to explicit FP32 flat encoding.
		const std::string dt = (tensors[i].dtype.empty() ? std::string("f32") : tensors[i].dtype);
		const uint64_t eb = (tensors[i].elemBytes ? tensors[i].elemBytes : 4ull);
		write_kv(out, kd.str(), dt);
		write_kv(out, ke.str(), u64_to_string(eb));
		write_kv(out, kr.str(), u64_to_string(static_cast<uint64_t>(tensors[i].shape.size())));
		write_kv(out, ksh.str(), u64_list_to_string(tensors[i].shape));
		write_kv(out, kl.str(), "row-major");
	}

	if (!static_cast<bool>(out))
		return false;
	out.flush();
	out.close();
	return rename_atomic(tmpPath, manifestPath);
}

static bool read_f32_blob_from_shard(std::ifstream& in,
                                    uint64_t offsetBytes,
                                    std::vector<float>& outVec,
                                    size_t wantCount,
                                    uint64_t wantFNV,
                                    uint64_t& outComputedFNV)
{
	outComputedFNV = fnv1a64_init();
	outVec.assign(wantCount, 0.0f);
	if (wantCount == 0u)
		return (wantFNV == outComputedFNV);

	const uint64_t totalBytes = static_cast<uint64_t>(wantCount) * 4ull;
	in.clear();
	in.seekg(static_cast<std::streamoff>(offsetBytes), std::ios::beg);
	if (!in)
		return false;

	const bool hostLittle = is_little_endian_host();
	const size_t kChunk = 4u * 1024u * 1024u;
	uint64_t remaining = totalBytes;
	size_t outOffFloats = 0u;

	if (hostLittle)
	{
		unsigned char* dstBytes = reinterpret_cast<unsigned char*>(&outVec[0]);
		uint64_t readOff = 0u;
		while (remaining > 0u)
		{
			const size_t n = (remaining > static_cast<uint64_t>(kChunk)) ? kChunk : static_cast<size_t>(remaining);
			in.read(reinterpret_cast<char*>(dstBytes + readOff), static_cast<std::streamsize>(n));
			if (!in)
				return false;
			outComputedFNV = fnv1a64_update(outComputedFNV, dstBytes + readOff, n);
			readOff += n;
			remaining -= static_cast<uint64_t>(n);
		}
	}
	else
	{
		// big-endian host: read LE bytes and convert
		std::vector<unsigned char> buf;
		buf.resize(kChunk);
		while (remaining > 0u)
		{
			const size_t n = (remaining > static_cast<uint64_t>(kChunk)) ? kChunk : static_cast<size_t>(remaining);
			in.read(reinterpret_cast<char*>(&buf[0]), static_cast<std::streamsize>(n));
			if (!in)
				return false;
			outComputedFNV = fnv1a64_update(outComputedFNV, &buf[0], n);
			// Convert n bytes -> floats
			const size_t floatsN = n / 4u;
			for (size_t i = 0; i < floatsN; ++i)
			{
				const unsigned int bits =
				    static_cast<unsigned int>(buf[i * 4u + 0u]) |
				    (static_cast<unsigned int>(buf[i * 4u + 1u]) << 8) |
				    (static_cast<unsigned int>(buf[i * 4u + 2u]) << 16) |
				    (static_cast<unsigned int>(buf[i * 4u + 3u]) << 24);
				float f = 0.0f;
				std::memcpy(&f, &bits, sizeof(float));
				outVec[outOffFloats + i] = f;
			}
			outOffFloats += floatsN;
			remaining -= static_cast<uint64_t>(n);
		}
	}

	return (wantFNV == outComputedFNV);
}

// Explicit ownership: shard streams are managed by GPointer (RAII).
static void close_and_delete_shards(std::vector< shmea::GPointer<std::ifstream> >& shardStreams)
{
	for (size_t s = 0; s < shardStreams.size(); ++s)
	{
		if (shardStreams[s])
			shardStreams[s]->close();
		shardStreams[s].reset();
	}
}

struct ParsedCheckpointTensorEntry
{
	std::string name;
	uint64_t count;
	size_t shardIndex;
	uint64_t offsetBytes;
	uint64_t bytes;
	uint64_t fnv1a64;
	std::string dtype;
	uint64_t elemBytes;
	std::vector<uint64_t> shape;
	ParsedCheckpointTensorEntry()
	    : name(), count(0u), shardIndex(0u), offsetBytes(0u), bytes(0u),
	      fnv1a64(0u), dtype(), elemBytes(0u), shape()
	{
	}
};

static glades::NNetworkStatus parse_checkpoint_manifest_header(const std::string& manifestPath,
                                                               std::map<std::string, std::string>& kvOut,
                                                               int& savedNetTypeOut)
{
	kvOut.clear();
	savedNetTypeOut = glades::NNetwork::TYPE_DFF;
	if (!read_kv_file(manifestPath, kvOut))
	{
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE,
		                              "loadCheckpoint: manifest.txt not found or unreadable");
	}
	if (kvOut.find("__magic__") == kvOut.end() || kvOut["__magic__"] != "GLADES_CHECKPOINT")
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, "loadCheckpoint: manifest magic mismatch");

	int version = -1;
	if (!parse_int(kvOut, "version", version) || version != kCheckpointFormatVersion)
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, "loadCheckpoint: unsupported checkpoint format version");

	std::map<std::string, std::string>::const_iterator itEndian = kvOut.find("file.endian");
	if (itEndian == kvOut.end())
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, "loadCheckpoint: missing file.endian");
	if (itEndian->second != "little")
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, "loadCheckpoint: unsupported file.endian (expected little)");

	std::map<std::string, std::string>::const_iterator itEncoding = kvOut.find("file.tensorEncoding");
	if (itEncoding == kvOut.end())
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, "loadCheckpoint: missing file.tensorEncoding");
	if (itEncoding->second != "raw_le")
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, "loadCheckpoint: unsupported file.tensorEncoding");

	if (!parse_int(kvOut, "netType", savedNetTypeOut) || savedNetTypeOut < 0)
		savedNetTypeOut = glades::NNetwork::TYPE_DFF;
	return glades::NNetworkStatus(glades::NNetworkStatus::OK, std::string());
}

static void restore_checkpoint_runtime_metadata(const std::map<std::string, std::string>& kv,
                                                int& epochsOut,
                                                bool& includeOptimizerStateOut,
                                                uint64_t& seedOut,
                                                bool& hasSeedOut,
                                                glades::TrainingConfig& cfg)
{
	int savedEpochs = 0;
	if (parse_int(kv, "epochs", savedEpochs))
		epochsOut = savedEpochs;

	hasSeedOut = parse_u64(kv, "rngSeed", seedOut);
	includeOptimizerStateOut = true;
	(void)parse_bool01(kv, "includeOptimizerState", includeOptimizerStateOut);

	glades::TrainingConfig cfgTmp = cfg;
	bool any = false;
	apply_training_config_from_kv(kv, cfgTmp, any);
	if (any)
		cfg = cfgTmp;
}

static glades::NNetworkStatus validate_checkpoint_training_config_compatibility(
    const std::map<std::string, std::string>& kv,
    const glades::TrainingConfig& currentCfg)
{
	int savedOptimizerType = -1;
	if (!parse_int(kv, "training.optimizer.type", savedOptimizerType))
		return glades::NNetworkStatus(glades::NNetworkStatus::OK, std::string());

	if (currentCfg.optimizer.type != glades::OptimizerConfig::ATLAS ||
	    savedOptimizerType != static_cast<int>(glades::OptimizerConfig::ATLAS))
	{
		return glades::NNetworkStatus(glades::NNetworkStatus::OK, std::string());
	}

	int savedRank = -1;
	if (parse_int(kv, "training.atlas.rank", savedRank) &&
	    savedRank >= 0 &&
	    currentCfg.atlas.rank != static_cast<unsigned int>(savedRank))
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.rank mismatch vs requested resume config (checkpoint "
		    << savedRank << ", current " << currentCfg.atlas.rank << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	int savedTSub = -1;
	if (parse_int(kv, "training.atlas.tSub", savedTSub) &&
	    savedTSub >= 0 &&
	    currentCfg.atlas.tSub != static_cast<unsigned int>(savedTSub))
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.tSub mismatch vs requested resume config (checkpoint "
		    << savedTSub << ", current " << currentCfg.atlas.tSub << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	return glades::NNetworkStatus(glades::NNetworkStatus::OK, std::string());
}

static glades::NNetworkStatus open_checkpoint_shards(const std::string& dir,
                                                     const std::map<std::string, std::string>& kv,
                                                     size_t shardCount,
                                                     std::vector< shmea::GPointer<std::ifstream> >& shardStreams,
                                                     std::vector<uint64_t>& shardBytes,
                                                     std::vector<uint64_t>& shardHashExpected)
{
	shardStreams.clear();
	shardStreams.resize(shardCount);
	shardBytes.assign(shardCount, 0u);
	shardHashExpected.assign(shardCount, 0u);

	for (size_t s = 0; s < shardCount; ++s)
	{
		std::ostringstream kf; kf << "shard." << static_cast<unsigned long long>(s) << ".file";
		std::ostringstream kb; kb << "shard." << static_cast<unsigned long long>(s) << ".bytes";
		std::ostringstream kh; kh << "shard." << static_cast<unsigned long long>(s) << ".fnv1a64";
		if (kv.find(kf.str()) == kv.end())
			return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, "loadCheckpoint: missing shard file entry in manifest");

		uint64_t bytesU = 0u;
		if (!parse_u64(kv, kb.str(), bytesU))
			return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, "loadCheckpoint: missing shard bytes entry in manifest");

		uint64_t hashU = 0u;
		if (!parse_u64(kv, kh.str(), hashU))
			return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, "loadCheckpoint: missing shard checksum entry in manifest");

		const std::string shardFile = kv.find(kf.str())->second;
		if (!is_safe_shard_filename(shardFile))
			return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, "loadCheckpoint: unsafe shard filename in manifest");

		const std::string shardPath = dir + shardFile;
		struct stat st;
		if (::lstat(shardPath.c_str(), &st) != 0 || S_ISLNK(st.st_mode) || !S_ISREG(st.st_mode))
		{
			return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE,
			                              "loadCheckpoint: shard file missing or not a regular file");
		}
		if (static_cast<uint64_t>(st.st_size) < bytesU)
		{
			return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE,
			                              "loadCheckpoint: shard file is smaller than manifest bytes");
		}

		shmea::GPointer<std::ifstream> in(new std::ifstream(shardPath.c_str(), std::ios::in | std::ios::binary));
		if (!in || !(*in.get()))
			return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, "loadCheckpoint: unable to open shard file");

		shardStreams[s] = in;
		shardBytes[s] = bytesU;
		shardHashExpected[s] = hashU;

		if (should_verify_shards())
		{
			uint64_t computed = 0u;
			if (!fnv1a64_hash_file_prefix(shardPath, bytesU, computed) || computed != hashU)
				return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, "loadCheckpoint: shard checksum mismatch");
		}
	}

	return glades::NNetworkStatus(glades::NNetworkStatus::OK, std::string());
}

static glades::NNetworkStatus parse_checkpoint_tensor_manifest_entry(const std::map<std::string, std::string>& kv,
                                                                     size_t tensorIndex,
                                                                     size_t shardCount,
                                                                     const std::vector<uint64_t>& shardBytes,
                                                                     ParsedCheckpointTensorEntry& out)
{
	out = ParsedCheckpointTensorEntry();

	std::ostringstream kn; kn << "tensor." << static_cast<unsigned long long>(tensorIndex) << ".name";
	std::ostringstream kc; kc << "tensor." << static_cast<unsigned long long>(tensorIndex) << ".count";
	std::ostringstream ks; ks << "tensor." << static_cast<unsigned long long>(tensorIndex) << ".shard";
	std::ostringstream ko; ko << "tensor." << static_cast<unsigned long long>(tensorIndex) << ".offsetBytes";
	std::ostringstream kb; kb << "tensor." << static_cast<unsigned long long>(tensorIndex) << ".bytes";
	std::ostringstream kh; kh << "tensor." << static_cast<unsigned long long>(tensorIndex) << ".fnv1a64";
	std::ostringstream kd; kd << "tensor." << static_cast<unsigned long long>(tensorIndex) << ".dtype";
	std::ostringstream ke; ke << "tensor." << static_cast<unsigned long long>(tensorIndex) << ".elemBytes";
	std::ostringstream kr; kr << "tensor." << static_cast<unsigned long long>(tensorIndex) << ".rank";
	std::ostringstream ksh; ksh << "tensor." << static_cast<unsigned long long>(tensorIndex) << ".shape";

	std::map<std::string, std::string>::const_iterator itName = kv.find(kn.str());
	if (itName == kv.end())
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, "loadCheckpoint: missing tensor name in manifest");
	out.name = itName->second;

	uint64_t shardU = 0u;
	if (!parse_u64(kv, kc.str(), out.count) ||
	    !parse_u64(kv, ks.str(), shardU) ||
	    !parse_u64(kv, ko.str(), out.offsetBytes) ||
	    !parse_u64(kv, kb.str(), out.bytes) ||
	    !parse_u64(kv, kh.str(), out.fnv1a64))
	{
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, "loadCheckpoint: malformed tensor entry in manifest");
	}

	if (kv.find(kd.str()) == kv.end())
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, "loadCheckpoint: missing tensor dtype");
	out.dtype = kv.find(kd.str())->second;
	if (!parse_u64(kv, ke.str(), out.elemBytes))
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, "loadCheckpoint: missing tensor elemBytes");

	uint64_t rankU = 0u;
	if (!parse_u64(kv, kr.str(), rankU) || rankU > 8u)
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, "loadCheckpoint: invalid tensor rank");
	if (rankU == 0u && out.count != 0u)
	{
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE,
		                              "loadCheckpoint: invalid tensor rank (0 with non-zero count)");
	}
	if (rankU > 0u)
	{
		std::map<std::string, std::string>::const_iterator itShape = kv.find(ksh.str());
		if (itShape == kv.end() || !parse_u64_list_value(itShape->second, out.shape) || out.shape.size() != static_cast<size_t>(rankU))
			return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, "loadCheckpoint: invalid tensor shape");
	}
	else
	{
		out.shape.clear();
	}

	if (out.dtype != "f32" || out.elemBytes != 4ull)
	{
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE,
		                              "loadCheckpoint: unsupported tensor dtype/elemBytes (expected f32/4)");
	}

	uint64_t shapeCount = 0u;
	if (out.count == 0u)
		shapeCount = 0u;
	else if (rankU == 0u)
		shapeCount = 0u;
	else if (!checked_shape_elem_count(out.shape, shapeCount))
	{
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE,
		                              "loadCheckpoint: invalid tensor shape (overflow)");
	}
	if (shapeCount != out.count)
	{
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE,
		                              "loadCheckpoint: tensor shape product != count (manifest corrupted)");
	}
	if ((out.offsetBytes % 4ull) != 0ull)
	{
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE,
		                              "loadCheckpoint: tensor offsetBytes is not 4-byte aligned");
	}
	const uint64_t wantBytes = out.count * out.elemBytes;
	if (out.bytes != wantBytes)
	{
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE,
		                              "loadCheckpoint: tensor bytes != count*elemBytes (manifest corrupted)");
	}
	if (shardU >= static_cast<uint64_t>(shardCount))
	{
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE,
		                              "loadCheckpoint: tensor references out-of-range shard");
	}
	out.shardIndex = static_cast<size_t>(shardU);
	if (out.shardIndex < shardBytes.size())
	{
		const uint64_t shardByteCount = shardBytes[out.shardIndex];
		if (out.offsetBytes > shardByteCount || out.bytes > shardByteCount || out.offsetBytes + out.bytes > shardByteCount)
		{
			return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE,
			                              "loadCheckpoint: tensor range exceeds shard bytes (manifest corrupted)");
		}
	}

	return glades::NNetworkStatus(glades::NNetworkStatus::OK, std::string());
}

// --- ATLAS checkpoint helpers ---

static void enqueueAtlasWrite(std::vector<TensorWriteRef>& out,
                               const std::string& prefix,
                               const glades::atlas::WeightState& st,
                               const std::string& dt)
{
	if (!st.initialized)
		return;
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(st.m));
		sh.push_back(static_cast<uint64_t>(st.r));
		out.push_back(TensorWriteRef(prefix + ".atlas.U", &st.U, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(st.r));
		out.push_back(TensorWriteRef(prefix + ".atlas.fisher", &st.fisherDiag, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(st.r));
		sh.push_back(static_cast<uint64_t>(st.n));
		out.push_back(TensorWriteRef(prefix + ".atlas.prevGz", &st.prevGz, dt, sh));
	}
}

static void writeAtlasManifestKV(std::map<std::string, std::string>& kv,
                                  const std::string& prefix,
                                  const glades::atlas::WeightState& st)
{
	if (!st.initialized)
		return;
	{
		std::ostringstream oss; oss << st.mu;
		kv["atlas." + prefix + ".mu"] = oss.str();
	}
	{
		std::ostringstream oss; oss << st.totalTrace;
		kv["atlas." + prefix + ".totalTrace"] = oss.str();
	}
	{
		std::ostringstream oss; oss << st.sigma2;
		kv["atlas." + prefix + ".sigma2"] = oss.str();
	}
	{
		std::ostringstream oss; oss << static_cast<unsigned long long>(st.step);
		kv["atlas." + prefix + ".step"] = oss.str();
	}
	{
		std::ostringstream oss; oss << static_cast<unsigned long long>(st.activeRank);
		kv["atlas." + prefix + ".activeRank"] = oss.str();
	}
}

static void enqueueAtlasRead(std::vector<TensorReadRef>& out,
                              const std::string& prefix,
                              glades::atlas::WeightState& st,
                              unsigned int m, unsigned int n, unsigned int r,
                              const std::string& dt)
{
	st.m = m;
	st.n = n;
	st.r = r;
	st.activeRank = r;
	const size_t mr = static_cast<size_t>(m) * static_cast<size_t>(r);
	const size_t rn = static_cast<size_t>(r) * static_cast<size_t>(n);
	st.U.resize(mr);
	st.fisherDiag.resize(r);
	st.prevGz.resize(rn);

	// Allocate persistent scratch buffers (must match initWeightState).
	st.scratch_gz.resize(rn);
	st.scratch_corrected.resize(rn);
	st.scratch_U_old.resize(mr);
	st.scratch_f_old.resize(static_cast<size_t>(r));
	st.scratch_B.resize(rn);
	st.scratch_Z.resize(mr);
	st.scratch_overlap.resize(static_cast<size_t>(r) * static_cast<size_t>(r));
	st.scratch_prevGzOld.resize(rn);
	st.scratch_basisPacked.resize(mr);
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(m));
		sh.push_back(static_cast<uint64_t>(r));
		out.push_back(TensorReadRef(prefix + ".atlas.U", &st.U, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(r));
		out.push_back(TensorReadRef(prefix + ".atlas.fisher", &st.fisherDiag, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(r));
		sh.push_back(static_cast<uint64_t>(n));
		out.push_back(TensorReadRef(prefix + ".atlas.prevGz", &st.prevGz, dt, sh));
	}
}

static void readAtlasManifestKV(const std::map<std::string, std::string>& kv,
                                 const std::string& prefix,
                                 glades::atlas::WeightState& st)
{
	if (st.U.empty())
		return;
	{
		std::map<std::string, std::string>::const_iterator it = kv.find("atlas." + prefix + ".mu");
		if (it != kv.end())
		{
			std::istringstream iss(it->second);
			iss >> st.mu;
		}
	}
	{
		std::map<std::string, std::string>::const_iterator it = kv.find("atlas." + prefix + ".totalTrace");
		if (it != kv.end())
		{
			std::istringstream iss(it->second);
			iss >> st.totalTrace;
		}
	}
	{
		std::map<std::string, std::string>::const_iterator it = kv.find("atlas." + prefix + ".sigma2");
		if (it != kv.end())
		{
			std::istringstream iss(it->second);
			iss >> st.sigma2;
		}
	}
	{
		std::map<std::string, std::string>::const_iterator it = kv.find("atlas." + prefix + ".step");
		if (it != kv.end())
		{
			std::istringstream iss(it->second);
			unsigned long long s = 0;
			iss >> s;
			st.step = s;
		}
	}
	{
		std::map<std::string, std::string>::const_iterator it = kv.find("atlas." + prefix + ".activeRank");
		if (it != kv.end())
		{
			std::istringstream iss(it->second);
			unsigned long long s = 0;
			iss >> s;
			if (s > 0u && s <= static_cast<unsigned long long>(st.r))
				st.activeRank = static_cast<unsigned int>(s);
		}
	}
	if (st.totalTrace <= 0.0f)
	{
		double activeTrace = 0.0;
		for (unsigned int c = 0; c < st.activeRank; ++c)
			activeTrace += static_cast<double>(st.fisherDiag[c]);
		const unsigned int complementDim = (st.m > st.activeRank) ? (st.m - st.activeRank) : 0u;
		double closedTrace = activeTrace;
		if (complementDim > 0u)
			closedTrace += static_cast<double>(st.sigma2) * static_cast<double>(complementDim);
		st.totalTrace = static_cast<float>(closedTrace > 0.0 ? closedTrace : st.sigma2);
	}
	st.lastBaselineRate = 0.0f;
	st.initialized = true;
}

} // namespace

namespace glades {

NNetworkStatus NNetwork::saveCheckpoint(const std::string& checkpointName, const CheckpointConfig& cfg) const
{
	const uint64_t effectiveMaxShardBytes =
	    static_cast<uint64_t>(cfg.maxShardBytes ? cfg.maxShardBytes : static_cast<size_t>(1024ull * 1024ull * 1024ull));
	const NNetworkStatus okStatus(NNetworkStatus::OK, std::string());
	PersistenceDiagnostics& diag = persistenceDiagnostics;
	resetPersistenceDiagnosticsAttempt(diag, "save_checkpoint", checkpointName, netType, true, false,
	                                   cfg.includeOptimizerState, effectiveMaxShardBytes);

	if (checkpointName.empty())
	{
		const NNetworkStatus st(NNetworkStatus::INVALID_ARGUMENT, "saveCheckpoint: checkpointName is empty");
		notePersistenceDiagnosticsFailure(diag, "validate", true, false, st, 0ULL, 0ULL, 0ULL);
		log_checkpoint_publish_event(this, shmea::GLogger::LOG_WARNING, "checkpoint_publish_rejected",
		                             "validate", checkpointName, netType, cfg.includeOptimizerState,
		                             false, 0ULL, 0ULL, effectiveMaxShardBytes, &st);
		return st;
	}
	if (!is_safe_path_component(checkpointName))
	{
		const NNetworkStatus st(NNetworkStatus::INVALID_ARGUMENT, "saveCheckpoint: checkpointName contains unsafe characters");
		notePersistenceDiagnosticsFailure(diag, "validate", true, false, st, 0ULL, 0ULL, 0ULL);
		log_checkpoint_publish_event(this, shmea::GLogger::LOG_WARNING, "checkpoint_publish_rejected",
		                             "validate", checkpointName, netType, cfg.includeOptimizerState,
		                             false, 0ULL, 0ULL, effectiveMaxShardBytes, &st);
		return st;
	}
	if (!skeleton)
	{
		const NNetworkStatus st(NNetworkStatus::INVALID_STATE, "saveCheckpoint: skeleton is null");
		notePersistenceDiagnosticsFailure(diag, "validate", false, false, st, 0ULL, 0ULL, 0ULL);
		log_checkpoint_publish_event(this, shmea::GLogger::LOG_ERROR, "checkpoint_publish_fail",
		                             "validate", checkpointName, netType, cfg.includeOptimizerState,
		                             false, 0ULL, 0ULL, effectiveMaxShardBytes, &st);
		return st;
	}

	log_checkpoint_publish_event(this, shmea::GLogger::LOG_INFO, "checkpoint_publish_start",
	                             "begin", checkpointName, netType, cfg.includeOptimizerState,
	                             false, 0ULL, 0ULL, effectiveMaxShardBytes, &okStatus);

	// Require tensors already initialized (checkpointing is for resumable training).
	{
		const bool hasDff = (netType == TYPE_DFF) && tensorDff.initialized;
		const bool hasRnn = (netType == TYPE_RNN) && tensorRnn.initialized;
		const bool hasGru = (netType == TYPE_GRU) && tensorGru.initialized;
		const bool hasLstm = (netType == TYPE_LSTM) && tensorLstm.initialized;
		const bool hasTr = (netType == TYPE_TRANSFORMER_ENCODER || netType == TYPE_TRANSFORMER_DECODER) && tensorTransformer.initialized;
		if (!hasDff && !hasRnn && !hasGru && !hasLstm && !hasTr)
		{
			const NNetworkStatus st(NNetworkStatus::INVALID_STATE,
			                        "saveCheckpoint: tensors are not initialized (run train/test or loadModel first)");
			notePersistenceDiagnosticsFailure(diag, "validate_tensors", false, false, st, 0ULL, 0ULL, 0ULL);
			log_checkpoint_publish_event(this, shmea::GLogger::LOG_ERROR, "checkpoint_publish_fail",
			                             "validate_tensors", checkpointName, netType, cfg.includeOptimizerState,
			                             false, 0ULL, 0ULL, effectiveMaxShardBytes, &st);
			return st;
		}
	}

	// Optimizer-state completeness guard:
	// If the run is configured for AdamW, we must persist optimizer state to support true resume.
	// (AdamW is currently only implemented for transformer net types in this codebase.)
	{
		const bool wantAdamW = (trainingConfig.optimizer.type == glades::OptimizerConfig::ADAMW);
		const bool isTransformer = (netType == TYPE_TRANSFORMER_ENCODER || netType == TYPE_TRANSFORMER_DECODER);
		if (wantAdamW)
		{
			if (!isTransformer)
			{
				const NNetworkStatus st(NNetworkStatus::INVALID_ARGUMENT,
				                        "saveCheckpoint: ADAMW optimizer is only supported for transformer net types");
				notePersistenceDiagnosticsFailure(diag, "validate_optimizer", true, false, st, 0ULL, 0ULL, 0ULL);
				log_checkpoint_publish_event(this, shmea::GLogger::LOG_WARNING, "checkpoint_publish_rejected",
				                             "validate_optimizer", checkpointName, netType, cfg.includeOptimizerState,
				                             false, 0ULL, 0ULL, effectiveMaxShardBytes, &st);
				return st;
			}
			if (!cfg.includeOptimizerState)
			{
				const NNetworkStatus st(NNetworkStatus::INVALID_ARGUMENT,
				                        "saveCheckpoint: ADAMW requires includeOptimizerState=true for resumable checkpoints");
				notePersistenceDiagnosticsFailure(diag, "validate_optimizer", true, false, st, 0ULL, 0ULL, 0ULL);
				log_checkpoint_publish_event(this, shmea::GLogger::LOG_WARNING, "checkpoint_publish_rejected",
				                             "validate_optimizer", checkpointName, netType, cfg.includeOptimizerState,
				                             false, 0ULL, 0ULL, effectiveMaxShardBytes, &st);
				return st;
			}
		}
	}

	// Hardened write strategy:
	// - write into a temp directory under the checkpoint root
	// - atomically publish by renaming temp dir -> final dir
	// This avoids partially-written checkpoints if the process crashes mid-write.
	if (!ensure_checkpoint_root())
	{
		const NNetworkStatus st(NNetworkStatus::INTERNAL_ERROR, "saveCheckpoint: unable to create checkpoint root directory");
		notePersistenceDiagnosticsFailure(diag, "ensure_checkpoint_root", false, false, st, 0ULL, 0ULL, 0ULL);
		log_checkpoint_publish_event(this, shmea::GLogger::LOG_ERROR, "checkpoint_publish_fail",
		                             "ensure_checkpoint_root", checkpointName, netType, cfg.includeOptimizerState,
		                             false, 0ULL, 0ULL, effectiveMaxShardBytes, &st);
		return st;
	}

	const std::string root = checkpoint_root_dir();
	const std::string finalDirNoSlash = join_dir(root, checkpointName);

	// Create a unique temp directory name.
	const unsigned long long pid = static_cast<unsigned long long>(::getpid());
	const unsigned long long t = static_cast<unsigned long long>(time(NULL));
	std::ostringstream tmpName;
	tmpName << checkpointName << ".tmp_" << pid << "_" << t;
	const std::string tmpDirNoSlash = join_dir(root, tmpName.str());

	// Create tmp dir.
	if (!mkdir_if_missing_dir_strict(tmpDirNoSlash))
	{
		const NNetworkStatus st(NNetworkStatus::INTERNAL_ERROR, "saveCheckpoint: unable to create temporary checkpoint directory");
		notePersistenceDiagnosticsFailure(diag, "create_tmp_dir", false, false, st, 0ULL, 0ULL, 0ULL);
		log_checkpoint_publish_event(this, shmea::GLogger::LOG_ERROR, "checkpoint_publish_fail",
		                             "create_tmp_dir", checkpointName, netType, cfg.includeOptimizerState,
		                             false, 0ULL, 0ULL, effectiveMaxShardBytes, &st);
		return st;
	}

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

	// 1) Write architecture snapshot
	{
		const shmea::GTable t = skeleton->toGTable();
		// Best-effort atomic write inside the temp directory.
		const std::string nnTmp = nninfoPath + ".tmp";
		t.save(shmea::GString(nnTmp.c_str()));
		if (!rename_atomic(nnTmp, nninfoPath))
		{
			(void)remove_tree_recursive(tmpDirNoSlash);
			const NNetworkStatus st(NNetworkStatus::INTERNAL_ERROR, "saveCheckpoint: unable to write nninfo.csv");
			notePersistenceDiagnosticsFailure(diag, "write_nninfo", false, false, st, 0ULL, 0ULL, 0ULL);
			log_checkpoint_publish_event(this, shmea::GLogger::LOG_ERROR, "checkpoint_publish_fail",
			                             "write_nninfo", checkpointName, netType, cfg.includeOptimizerState,
			                             false, 0ULL, 0ULL, effectiveMaxShardBytes, &st);
			return st;
		}
	}

	// 2) Collect tensors
	std::vector<TensorWriteRef> tensorsToWrite;
	tensorsToWrite.clear();
	std::map<std::string, std::string> atlasKV;
	{
		const bool includeOpt = cfg.includeOptimizerState;
		const bool isAtlas = (trainingConfig.optimizer.type == glades::OptimizerConfig::ATLAS);
		const std::string dt = "f32";
		if (netType == TYPE_DFF)
		{
			for (size_t t = 0; t < tensorDff.T.size(); ++t)
			{
				const TensorDFFState::Transition& tr = tensorDff.T[t];
				{
					std::ostringstream oss; oss << "dff.t" << static_cast<unsigned long long>(t) << ".W";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(tr.out));
					sh.push_back(static_cast<uint64_t>(tr.in));
					tensorsToWrite.push_back(TensorWriteRef(oss.str(), &tr.W, dt, sh));
				}
				{
					std::ostringstream oss; oss << "dff.t" << static_cast<unsigned long long>(t) << ".bias";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(tr.out));
					tensorsToWrite.push_back(TensorWriteRef(oss.str(), &tr.bias, dt, sh));
				}
				if (includeOpt)
				{
					std::ostringstream oss; oss << "dff.t" << static_cast<unsigned long long>(t) << ".vW";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(tr.out));
					sh.push_back(static_cast<uint64_t>(tr.in));
					tensorsToWrite.push_back(TensorWriteRef(oss.str(), &tr.vW, dt, sh));
				}
				if (includeOpt && isAtlas && t < tensorDff.atlasState.size())
				{
					std::ostringstream oss; oss << "dff.t" << static_cast<unsigned long long>(t);
					enqueueAtlasWrite(tensorsToWrite, oss.str(), tensorDff.atlasState[t], dt);
					writeAtlasManifestKV(atlasKV, oss.str(), tensorDff.atlasState[t]);
				}
			}
		}
		else if (netType == TYPE_RNN)
		{
			for (size_t l = 0; l < tensorRnn.H.size(); ++l)
			{
				const TensorRNNState::Hidden& hl = tensorRnn.H[l];
				{
					std::ostringstream oss; oss << "rnn.h" << static_cast<unsigned long long>(l) << ".Wxh";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(hl.h));
					sh.push_back(static_cast<uint64_t>(hl.in));
					tensorsToWrite.push_back(TensorWriteRef(oss.str(), &hl.Wxh, dt, sh));
				}
				{
					std::ostringstream oss; oss << "rnn.h" << static_cast<unsigned long long>(l) << ".Whh";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(hl.h));
					sh.push_back(static_cast<uint64_t>(hl.h));
					tensorsToWrite.push_back(TensorWriteRef(oss.str(), &hl.Whh, dt, sh));
				}
				{
					std::ostringstream oss; oss << "rnn.h" << static_cast<unsigned long long>(l) << ".bias";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(hl.h));
					tensorsToWrite.push_back(TensorWriteRef(oss.str(), &hl.bias, dt, sh));
				}
				if (includeOpt)
				{
					{
						std::ostringstream oss; oss << "rnn.h" << static_cast<unsigned long long>(l) << ".vWxh";
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(hl.h));
						sh.push_back(static_cast<uint64_t>(hl.in));
						tensorsToWrite.push_back(TensorWriteRef(oss.str(), &hl.vWxh, dt, sh));
					}
					{
						std::ostringstream oss; oss << "rnn.h" << static_cast<unsigned long long>(l) << ".vWhh";
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(hl.h));
						sh.push_back(static_cast<uint64_t>(hl.h));
						tensorsToWrite.push_back(TensorWriteRef(oss.str(), &hl.vWhh, dt, sh));
					}
				}
				if (includeOpt && isAtlas)
				{
					{
						std::ostringstream oss; oss << "rnn.h" << static_cast<unsigned long long>(l) << ".Wxh";
						enqueueAtlasWrite(tensorsToWrite, oss.str(), hl.atlasWxh, dt);
						writeAtlasManifestKV(atlasKV, oss.str(), hl.atlasWxh);
					}
					{
						std::ostringstream oss; oss << "rnn.h" << static_cast<unsigned long long>(l) << ".Whh";
						enqueueAtlasWrite(tensorsToWrite, oss.str(), hl.atlasWhh, dt);
						writeAtlasManifestKV(atlasKV, oss.str(), hl.atlasWhh);
					}
				}
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(tensorRnn.O.out));
				sh.push_back(static_cast<uint64_t>(tensorRnn.O.in));
				tensorsToWrite.push_back(TensorWriteRef("rnn.o.Why", &tensorRnn.O.Why, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(tensorRnn.O.out));
				tensorsToWrite.push_back(TensorWriteRef("rnn.o.bias", &tensorRnn.O.bias, dt, sh));
			}
			if (includeOpt)
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(tensorRnn.O.out));
				sh.push_back(static_cast<uint64_t>(tensorRnn.O.in));
				tensorsToWrite.push_back(TensorWriteRef("rnn.o.vWhy", &tensorRnn.O.vWhy, dt, sh));
			}
			if (includeOpt && isAtlas)
			{
				enqueueAtlasWrite(tensorsToWrite, "rnn.o", tensorRnn.O.atlasWhy, dt);
				writeAtlasManifestKV(atlasKV, "rnn.o", tensorRnn.O.atlasWhy);
			}
		}
		else if (netType == TYPE_GRU || netType == TYPE_LSTM)
		{
			const TensorGatedState& tg = (netType == TYPE_GRU) ? tensorGru : tensorLstm;
			const char* prefix = (netType == TYPE_GRU) ? "gru" : "lstm";
			for (size_t l = 0; l < tg.H.size(); ++l)
			{
				const TensorGatedState::Hidden& hl = tg.H[l];
				{
					std::ostringstream oss; oss << prefix << ".h" << static_cast<unsigned long long>(l) << ".W";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(tg.gateCount));
					sh.push_back(static_cast<uint64_t>(hl.h));
					sh.push_back(static_cast<uint64_t>(hl.in));
					tensorsToWrite.push_back(TensorWriteRef(oss.str(), &hl.W, dt, sh));
				}
				{
					std::ostringstream oss; oss << prefix << ".h" << static_cast<unsigned long long>(l) << ".U";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(tg.gateCount));
					sh.push_back(static_cast<uint64_t>(hl.h));
					sh.push_back(static_cast<uint64_t>(hl.h));
					tensorsToWrite.push_back(TensorWriteRef(oss.str(), &hl.U, dt, sh));
				}
				{
					std::ostringstream oss; oss << prefix << ".h" << static_cast<unsigned long long>(l) << ".bias";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(tg.gateCount));
					sh.push_back(static_cast<uint64_t>(hl.h));
					tensorsToWrite.push_back(TensorWriteRef(oss.str(), &hl.bias, dt, sh));
				}
				if (includeOpt)
				{
					{
						std::ostringstream oss; oss << prefix << ".h" << static_cast<unsigned long long>(l) << ".vW";
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(tg.gateCount));
						sh.push_back(static_cast<uint64_t>(hl.h));
						sh.push_back(static_cast<uint64_t>(hl.in));
						tensorsToWrite.push_back(TensorWriteRef(oss.str(), &hl.vW, dt, sh));
					}
					{
						std::ostringstream oss; oss << prefix << ".h" << static_cast<unsigned long long>(l) << ".vU";
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(tg.gateCount));
						sh.push_back(static_cast<uint64_t>(hl.h));
						sh.push_back(static_cast<uint64_t>(hl.h));
						tensorsToWrite.push_back(TensorWriteRef(oss.str(), &hl.vU, dt, sh));
					}
				}
				if (includeOpt && isAtlas)
				{
					{
						std::ostringstream oss; oss << prefix << ".h" << static_cast<unsigned long long>(l) << ".W";
						enqueueAtlasWrite(tensorsToWrite, oss.str(), hl.atlasW, dt);
						writeAtlasManifestKV(atlasKV, oss.str(), hl.atlasW);
					}
					{
						std::ostringstream oss; oss << prefix << ".h" << static_cast<unsigned long long>(l) << ".U";
						enqueueAtlasWrite(tensorsToWrite, oss.str(), hl.atlasU, dt);
						writeAtlasManifestKV(atlasKV, oss.str(), hl.atlasU);
					}
				}
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(tg.O.out));
				sh.push_back(static_cast<uint64_t>(tg.O.in));
				tensorsToWrite.push_back(TensorWriteRef(std::string(prefix) + ".o.Why", &tg.O.Why, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(tg.O.out));
				tensorsToWrite.push_back(TensorWriteRef(std::string(prefix) + ".o.bias", &tg.O.bias, dt, sh));
			}
			if (includeOpt)
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(tg.O.out));
				sh.push_back(static_cast<uint64_t>(tg.O.in));
				tensorsToWrite.push_back(TensorWriteRef(std::string(prefix) + ".o.vWhy", &tg.O.vWhy, dt, sh));
			}
			if (includeOpt && isAtlas)
			{
				enqueueAtlasWrite(tensorsToWrite, std::string(prefix) + ".o", tg.O.atlasWhy, dt);
				writeAtlasManifestKV(atlasKV, std::string(prefix) + ".o", tg.O.atlasWhy);
			}
		}
		else if (netType == TYPE_TRANSFORMER_ENCODER || netType == TYPE_TRANSFORMER_DECODER)
		{
			const TensorTransformerState& tt = tensorTransformer;
			const unsigned int dModel = tt.dModel;
			const unsigned int inputSize = tt.inputSize;
			const unsigned int outSize = tt.outSize;
			const unsigned int nHeads = tt.nHeads;
			const unsigned int nKVHeads = (tt.nKVHeads > 0u ? tt.nKVHeads : nHeads);
			const unsigned int dHead = (nHeads > 0u ? (dModel / nHeads) : 0u);
			const unsigned int dModelKV = nKVHeads * dHead;
			const unsigned int ff1Width =
			    (tt.ffnKind == static_cast<unsigned int>(glades::TransformerRunConfig::FFN_SWIGLU)) ? (2u * tt.dFF) : tt.dFF;

			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(dModel));
				sh.push_back(static_cast<uint64_t>(inputSize));
				tensorsToWrite.push_back(TensorWriteRef("tr.WIn", &tt.WIn, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(dModel));
				tensorsToWrite.push_back(TensorWriteRef("tr.bIn", &tt.bIn, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(outSize));
				sh.push_back(static_cast<uint64_t>(dModel));
				tensorsToWrite.push_back(TensorWriteRef("tr.WOut", &tt.WOut, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(outSize));
				tensorsToWrite.push_back(TensorWriteRef("tr.bOut", &tt.bOut, dt, sh));
			}
			// Token-LM tensors are only present when tokenModel==true.
			if (tt.tokenModel)
			{
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(tt.vocabSize));
					sh.push_back(static_cast<uint64_t>(dModel));
					tensorsToWrite.push_back(TensorWriteRef("tr.tokE", &tt.tokE, dt, sh));
				}
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(tt.vocabSize));
					tensorsToWrite.push_back(TensorWriteRef("tr.lmBias", &tt.lmBias, dt, sh));
				}
			}

			// Final LayerNorm weights
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(dModel));
				tensorsToWrite.push_back(TensorWriteRef("tr.lnFinalGamma", &tt.lnFinalGamma, dt, sh));
				tensorsToWrite.push_back(TensorWriteRef("tr.lnFinalBeta", &tt.lnFinalBeta, dt, sh));
			}

			if (includeOpt)
			{
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModel));
					sh.push_back(static_cast<uint64_t>(inputSize));
					tensorsToWrite.push_back(TensorWriteRef("tr.vWIn", &tt.vWIn, dt, sh));
					tensorsToWrite.push_back(TensorWriteRef("tr.v2WIn", &tt.v2WIn, dt, sh));
				}
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModel));
					tensorsToWrite.push_back(TensorWriteRef("tr.mBIn", &tt.mBIn, dt, sh));
					tensorsToWrite.push_back(TensorWriteRef("tr.v2BIn", &tt.v2BIn, dt, sh));
				}
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(outSize));
					sh.push_back(static_cast<uint64_t>(dModel));
					tensorsToWrite.push_back(TensorWriteRef("tr.vWOut", &tt.vWOut, dt, sh));
					tensorsToWrite.push_back(TensorWriteRef("tr.v2WOut", &tt.v2WOut, dt, sh));
				}
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(outSize));
					tensorsToWrite.push_back(TensorWriteRef("tr.mBOut", &tt.mBOut, dt, sh));
					tensorsToWrite.push_back(TensorWriteRef("tr.v2BOut", &tt.v2BOut, dt, sh));
				}
				{
					// Token-LM optimizer tensors are only present when tokenModel==true.
					if (tt.tokenModel)
					{
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(tt.vocabSize));
						sh.push_back(static_cast<uint64_t>(dModel));
						tensorsToWrite.push_back(TensorWriteRef("tr.vTokE", &tt.vTokE, dt, sh));
						tensorsToWrite.push_back(TensorWriteRef("tr.v2TokE", &tt.v2TokE, dt, sh));
					}
				}
				{
					if (tt.tokenModel)
					{
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(tt.vocabSize));
						tensorsToWrite.push_back(TensorWriteRef("tr.mLmBias", &tt.mLmBias, dt, sh));
						tensorsToWrite.push_back(TensorWriteRef("tr.v2LmBias", &tt.v2LmBias, dt, sh));
					}
				}
				// Final LayerNorm optimizer state
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModel));
					tensorsToWrite.push_back(TensorWriteRef("tr.mLnFinalGamma", &tt.mLnFinalGamma, dt, sh));
					tensorsToWrite.push_back(TensorWriteRef("tr.v2LnFinalGamma", &tt.v2LnFinalGamma, dt, sh));
					tensorsToWrite.push_back(TensorWriteRef("tr.mLnFinalBeta", &tt.mLnFinalBeta, dt, sh));
					tensorsToWrite.push_back(TensorWriteRef("tr.v2LnFinalBeta", &tt.v2LnFinalBeta, dt, sh));
				}
			}

			for (size_t l = 0; l < tt.blocks.size(); ++l)
			{
				const TensorTransformerState::Block& b = tt.blocks[l];
				const unsigned long long li = static_cast<unsigned long long>(l);
				{
					std::ostringstream oss; oss << "tr.b" << li << ".ln1Gamma";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModel));
					tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.ln1Gamma, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".ln1Beta";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModel));
					tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.ln1Beta, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".Wq";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModel));
					sh.push_back(static_cast<uint64_t>(dModel));
					tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.Wq, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".Wk";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModelKV));
					sh.push_back(static_cast<uint64_t>(dModel));
					tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.Wk, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".Wv";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModelKV));
					sh.push_back(static_cast<uint64_t>(dModel));
					tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.Wv, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".Wo";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModel));
					sh.push_back(static_cast<uint64_t>(dModel));
					tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.Wo, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".bq";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModel));
					tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.bq, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".bk";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModelKV));
					tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.bk, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".bv";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModelKV));
					tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.bv, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".bo";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModel));
					tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.bo, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".ln2Gamma";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModel));
					tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.ln2Gamma, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".ln2Beta";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModel));
					tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.ln2Beta, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".W1";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(ff1Width));
					sh.push_back(static_cast<uint64_t>(dModel));
					tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.W1, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".b1";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(ff1Width));
					tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.b1, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".W2";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModel));
					sh.push_back(static_cast<uint64_t>(tt.dFF));
					tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.W2, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".b2";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModel));
					tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.b2, dt, sh));
				}

				if (includeOpt)
				{
					{
						std::ostringstream oss; oss << "tr.b" << li << ".mLn1Gamma";
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(dModel));
						tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.mLn1Gamma, dt, sh));
					}
					{
						std::ostringstream oss; oss << "tr.b" << li << ".v2Ln1Gamma";
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(dModel));
						tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.v2Ln1Gamma, dt, sh));
					}
					{
						std::ostringstream oss; oss << "tr.b" << li << ".mLn1Beta";
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(dModel));
						tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.mLn1Beta, dt, sh));
					}
					{
						std::ostringstream oss; oss << "tr.b" << li << ".v2Ln1Beta";
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(dModel));
						tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.v2Ln1Beta, dt, sh));
					}
					{
						std::ostringstream oss; oss << "tr.b" << li << ".vWq";
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(dModel));
						sh.push_back(static_cast<uint64_t>(dModel));
						tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.vWq, dt, sh));
					}
					{
						std::ostringstream oss; oss << "tr.b" << li << ".v2Wq";
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(dModel));
						sh.push_back(static_cast<uint64_t>(dModel));
						tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.v2Wq, dt, sh));
					}
					{
						std::ostringstream oss; oss << "tr.b" << li << ".vWk";
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(dModelKV));
						sh.push_back(static_cast<uint64_t>(dModel));
						tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.vWk, dt, sh));
					}
					{
						std::ostringstream oss; oss << "tr.b" << li << ".v2Wk";
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(dModelKV));
						sh.push_back(static_cast<uint64_t>(dModel));
						tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.v2Wk, dt, sh));
					}
					{
						std::ostringstream oss; oss << "tr.b" << li << ".vWv";
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(dModelKV));
						sh.push_back(static_cast<uint64_t>(dModel));
						tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.vWv, dt, sh));
					}
					{
						std::ostringstream oss; oss << "tr.b" << li << ".v2Wv";
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(dModelKV));
						sh.push_back(static_cast<uint64_t>(dModel));
						tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.v2Wv, dt, sh));
					}
					{
						std::ostringstream oss; oss << "tr.b" << li << ".vWo";
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(dModel));
						sh.push_back(static_cast<uint64_t>(dModel));
						tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.vWo, dt, sh));
					}
					{
						std::ostringstream oss; oss << "tr.b" << li << ".v2Wo";
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(dModel));
						sh.push_back(static_cast<uint64_t>(dModel));
						tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.v2Wo, dt, sh));
					}
					{
						std::ostringstream oss; oss << "tr.b" << li << ".mBq";
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(dModel));
						tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.mBq, dt, sh));
					}
					{
						std::ostringstream oss; oss << "tr.b" << li << ".v2Bq";
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(dModel));
						tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.v2Bq, dt, sh));
					}
					{
						std::ostringstream oss; oss << "tr.b" << li << ".mBk";
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(dModelKV));
						tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.mBk, dt, sh));
					}
					{
						std::ostringstream oss; oss << "tr.b" << li << ".v2Bk";
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(dModelKV));
						tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.v2Bk, dt, sh));
					}
					{
						std::ostringstream oss; oss << "tr.b" << li << ".mBv";
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(dModelKV));
						tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.mBv, dt, sh));
					}
					{
						std::ostringstream oss; oss << "tr.b" << li << ".v2Bv";
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(dModelKV));
						tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.v2Bv, dt, sh));
					}
					{
						std::ostringstream oss; oss << "tr.b" << li << ".mBo";
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(dModel));
						tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.mBo, dt, sh));
					}
					{
						std::ostringstream oss; oss << "tr.b" << li << ".v2Bo";
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(dModel));
						tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.v2Bo, dt, sh));
					}
					{
						std::ostringstream oss; oss << "tr.b" << li << ".mLn2Gamma";
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(dModel));
						tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.mLn2Gamma, dt, sh));
					}
					{
						std::ostringstream oss; oss << "tr.b" << li << ".v2Ln2Gamma";
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(dModel));
						tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.v2Ln2Gamma, dt, sh));
					}
					{
						std::ostringstream oss; oss << "tr.b" << li << ".mLn2Beta";
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(dModel));
						tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.mLn2Beta, dt, sh));
					}
					{
						std::ostringstream oss; oss << "tr.b" << li << ".v2Ln2Beta";
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(dModel));
						tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.v2Ln2Beta, dt, sh));
					}
					{
						std::ostringstream oss; oss << "tr.b" << li << ".vW1";
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(ff1Width));
						sh.push_back(static_cast<uint64_t>(dModel));
						tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.vW1, dt, sh));
					}
					{
						std::ostringstream oss; oss << "tr.b" << li << ".v2W1";
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(ff1Width));
						sh.push_back(static_cast<uint64_t>(dModel));
						tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.v2W1, dt, sh));
					}
					{
						std::ostringstream oss; oss << "tr.b" << li << ".mB1";
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(ff1Width));
						tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.mB1, dt, sh));
					}
					{
						std::ostringstream oss; oss << "tr.b" << li << ".v2B1";
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(ff1Width));
						tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.v2B1, dt, sh));
					}
					{
						std::ostringstream oss; oss << "tr.b" << li << ".vW2";
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(dModel));
						sh.push_back(static_cast<uint64_t>(tt.dFF));
						tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.vW2, dt, sh));
					}
					{
						std::ostringstream oss; oss << "tr.b" << li << ".v2W2";
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(dModel));
						sh.push_back(static_cast<uint64_t>(tt.dFF));
						tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.v2W2, dt, sh));
					}
					{
						std::ostringstream oss; oss << "tr.b" << li << ".mB2";
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(dModel));
						tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.mB2, dt, sh));
					}
					{
						std::ostringstream oss; oss << "tr.b" << li << ".v2B2";
						std::vector<uint64_t> sh;
						sh.push_back(static_cast<uint64_t>(dModel));
						tensorsToWrite.push_back(TensorWriteRef(oss.str(), &b.v2B2, dt, sh));
					}
				}
				if (includeOpt && isAtlas)
				{
					std::ostringstream base; base << "tr.b" << li;
					const std::string bp = base.str();
					enqueueAtlasWrite(tensorsToWrite, bp + ".Wq", b.atlasWq, dt);
					enqueueAtlasWrite(tensorsToWrite, bp + ".Wk", b.atlasWk, dt);
					enqueueAtlasWrite(tensorsToWrite, bp + ".Wv", b.atlasWv, dt);
					enqueueAtlasWrite(tensorsToWrite, bp + ".Wo", b.atlasWo, dt);
					enqueueAtlasWrite(tensorsToWrite, bp + ".W1", b.atlasW1, dt);
					enqueueAtlasWrite(tensorsToWrite, bp + ".W2", b.atlasW2, dt);
					writeAtlasManifestKV(atlasKV, bp + ".Wq", b.atlasWq);
					writeAtlasManifestKV(atlasKV, bp + ".Wk", b.atlasWk);
					writeAtlasManifestKV(atlasKV, bp + ".Wv", b.atlasWv);
					writeAtlasManifestKV(atlasKV, bp + ".Wo", b.atlasWo);
					writeAtlasManifestKV(atlasKV, bp + ".W1", b.atlasW1);
					writeAtlasManifestKV(atlasKV, bp + ".W2", b.atlasW2);
				}
			}
			if (includeOpt && isAtlas)
			{
				enqueueAtlasWrite(tensorsToWrite, "tr.WIn", tt.atlasWIn, dt);
				enqueueAtlasWrite(tensorsToWrite, "tr.WOut", tt.atlasWOut, dt);
				writeAtlasManifestKV(atlasKV, "tr.WIn", tt.atlasWIn);
				writeAtlasManifestKV(atlasKV, "tr.WOut", tt.atlasWOut);
				if (tt.tokenModel)
				{
					enqueueAtlasWrite(tensorsToWrite, "tr.tokE", tt.atlasTokE, dt);
					writeAtlasManifestKV(atlasKV, "tr.tokE", tt.atlasTokE);
				}
			}
		}
		else
		{
			const NNetworkStatus st(NNetworkStatus::INVALID_ARGUMENT, "saveCheckpoint: unknown netType");
			notePersistenceDiagnosticsFailure(diag, "collect_tensors", true, false, st, 0ULL, 0ULL, 0ULL);
			log_checkpoint_publish_event(this, shmea::GLogger::LOG_WARNING, "checkpoint_publish_rejected",
			                             "collect_tensors", checkpointName, netType, cfg.includeOptimizerState,
			                             false, 0ULL, 0ULL, effectiveMaxShardBytes, &st);
			return st;
		}
	}

	// 3) Write sharded blobs
	const size_t maxShardBytes = static_cast<size_t>(effectiveMaxShardBytes);
	CheckpointShardWriter writer(dir, maxShardBytes);
	if (!writer.open_first())
	{
		const NNetworkStatus st(NNetworkStatus::INTERNAL_ERROR, "saveCheckpoint: unable to open shard_000.bin for writing");
		notePersistenceDiagnosticsFailure(diag, "open_first_shard", false, false, st, 0ULL, 0ULL, 0ULL);
		log_checkpoint_publish_event(this, shmea::GLogger::LOG_ERROR, "checkpoint_publish_fail",
		                             "open_first_shard", checkpointName, netType, cfg.includeOptimizerState,
		                             false, 0ULL, static_cast<uint64_t>(tensorsToWrite.size()), effectiveMaxShardBytes, &st);
		return st;
	}

	for (size_t i = 0; i < tensorsToWrite.size(); ++i)
	{
		if (!tensorsToWrite[i].vec)
		{
			const NNetworkStatus st(NNetworkStatus::INTERNAL_ERROR, "saveCheckpoint: internal error (null tensor vec)");
			notePersistenceDiagnosticsFailure(diag, "write_shards", false, false, st, 0ULL,
			                                  static_cast<uint64_t>(tensorsToWrite.size()), 0ULL);
			log_checkpoint_publish_event(this, shmea::GLogger::LOG_ERROR, "checkpoint_publish_fail",
			                             "write_shards", checkpointName, netType, cfg.includeOptimizerState,
			                             false, 0ULL, static_cast<uint64_t>(tensorsToWrite.size()), effectiveMaxShardBytes, &st);
			return st;
		}
		if (!writer.add_tensor(tensorsToWrite[i].name, *tensorsToWrite[i].vec, tensorsToWrite[i].dtype, tensorsToWrite[i].shape))
		{
			const NNetworkStatus st(NNetworkStatus::INTERNAL_ERROR, "saveCheckpoint: failed while writing shard data");
			notePersistenceDiagnosticsFailure(diag, "write_shards", false, false, st,
			                                  static_cast<uint64_t>(writer.get_shards().size()),
			                                  static_cast<uint64_t>(tensorsToWrite.size()), 0ULL);
			log_checkpoint_publish_event(this, shmea::GLogger::LOG_ERROR, "checkpoint_publish_fail",
			                             "write_shards", checkpointName, netType, cfg.includeOptimizerState,
			                             false, static_cast<uint64_t>(writer.get_shards().size()),
			                             static_cast<uint64_t>(tensorsToWrite.size()), effectiveMaxShardBytes, &st);
			return st;
		}
	}
	if (!writer.finalize_all())
	{
		const NNetworkStatus st(NNetworkStatus::INTERNAL_ERROR, "saveCheckpoint: failed to finalize shard data");
		notePersistenceDiagnosticsFailure(diag, "finalize_shards", false, false, st,
		                                  static_cast<uint64_t>(writer.get_shards().size()),
		                                  static_cast<uint64_t>(tensorsToWrite.size()), 0ULL);
		log_checkpoint_publish_event(this, shmea::GLogger::LOG_ERROR, "checkpoint_publish_fail",
		                             "finalize_shards", checkpointName, netType, cfg.includeOptimizerState,
		                             false, static_cast<uint64_t>(writer.get_shards().size()),
		                             static_cast<uint64_t>(tensorsToWrite.size()), effectiveMaxShardBytes, &st);
		return st;
	}

	// 4) Manifest
	unsigned long long trStep = 0ULL;
	float trLossScale = 1.0f;
	unsigned long long trLossScaleGood = 0ULL;
	if (netType == TYPE_TRANSFORMER_ENCODER || netType == TYPE_TRANSFORMER_DECODER)
	{
		trStep = static_cast<unsigned long long>(tensorTransformer.optimizerStep);
		trLossScale = tensorTransformer.mpLossScale;
		trLossScaleGood = static_cast<unsigned long long>(tensorTransformer.mpLossScaleGoodSteps);
	}

	if (!write_manifest(manifestPath,
	                    checkpointName,
	                    netType,
	                    epochs,
	                    rngSeed,
	                    trainingConfig,
	                    cfg.includeOptimizerState,
	                    trStep,
	                    trLossScale,
	                    trLossScaleGood,
	                    maxShardBytes,
	                    writer.get_shards(),
	                    writer.get_tensors(),
	                    atlasKV))
	{
		const NNetworkStatus st(NNetworkStatus::INTERNAL_ERROR, "saveCheckpoint: unable to write manifest.txt");
		notePersistenceDiagnosticsFailure(diag, "write_manifest", false, false, st,
		                                  static_cast<uint64_t>(writer.get_shards().size()),
		                                  static_cast<uint64_t>(writer.get_tensors().size()), 0ULL);
		log_checkpoint_publish_event(this, shmea::GLogger::LOG_ERROR, "checkpoint_publish_fail",
		                             "write_manifest", checkpointName, netType, cfg.includeOptimizerState,
		                             false, static_cast<uint64_t>(writer.get_shards().size()),
		                             static_cast<uint64_t>(writer.get_tensors().size()), effectiveMaxShardBytes, &st);
		return st;
	}

	// 5) Atomically publish:
	// - if an existing checkpoint directory exists, rename it to a backup
	// - rename temp dir -> final dir
	// - best-effort delete the backup
	std::string backupDirNoSlash;
	bool rotatedPrevious = false;
	if (stat_is_dir(finalDirNoSlash))
	{
		std::ostringstream bak;
		bak << checkpointName << ".bak_" << static_cast<unsigned long long>(::getpid()) << "_" << static_cast<unsigned long long>(time(NULL));
		backupDirNoSlash = join_dir(root, bak.str());
		if (!rename_atomic(finalDirNoSlash, backupDirNoSlash))
		{
			const NNetworkStatus st(NNetworkStatus::INTERNAL_ERROR, "saveCheckpoint: unable to rotate existing checkpoint directory");
			notePersistenceDiagnosticsFailure(diag, "rotate_existing", false, false, st,
			                                  static_cast<uint64_t>(writer.get_shards().size()),
			                                  static_cast<uint64_t>(writer.get_tensors().size()), 0ULL);
			log_checkpoint_publish_event(this, shmea::GLogger::LOG_ERROR, "checkpoint_publish_fail",
			                             "rotate_existing", checkpointName, netType, cfg.includeOptimizerState,
			                             false, static_cast<uint64_t>(writer.get_shards().size()),
			                             static_cast<uint64_t>(writer.get_tensors().size()), effectiveMaxShardBytes, &st);
			return st;
		}
		rotatedPrevious = true;
	}
	if (!rename_atomic(tmpDirNoSlash, finalDirNoSlash))
	{
		// Best-effort rollback.
		if (!backupDirNoSlash.empty())
			(void)rename_atomic(backupDirNoSlash, finalDirNoSlash);
		const NNetworkStatus st(NNetworkStatus::INTERNAL_ERROR, "saveCheckpoint: unable to publish checkpoint directory (rename failed)");
		notePersistenceDiagnosticsFailure(diag, "publish", false, rotatedPrevious, st,
		                                  static_cast<uint64_t>(writer.get_shards().size()),
		                                  static_cast<uint64_t>(writer.get_tensors().size()), 0ULL);
		log_checkpoint_publish_event(this, shmea::GLogger::LOG_ERROR, "checkpoint_publish_fail",
		                             "publish", checkpointName, netType, cfg.includeOptimizerState,
		                             rotatedPrevious, static_cast<uint64_t>(writer.get_shards().size()),
		                             static_cast<uint64_t>(writer.get_tensors().size()), effectiveMaxShardBytes, &st);
		return st;
	}
	tmpGuard.dismiss();
	if (!backupDirNoSlash.empty())
		(void)remove_tree_recursive(backupDirNoSlash);

	notePersistenceDiagnosticsSuccess(diag, "publish_complete", rotatedPrevious, okStatus,
	                                  static_cast<uint64_t>(writer.get_shards().size()),
	                                  static_cast<uint64_t>(writer.get_tensors().size()), 0ULL);
	log_checkpoint_publish_event(this, shmea::GLogger::LOG_INFO, "checkpoint_publish_end",
	                             "publish_complete", checkpointName, netType, cfg.includeOptimizerState,
	                             rotatedPrevious, static_cast<uint64_t>(writer.get_shards().size()),
	                             static_cast<uint64_t>(writer.get_tensors().size()), effectiveMaxShardBytes, &okStatus);
	return okStatus;
}

NNetworkStatus NNetwork::loadCheckpoint(const std::string& checkpointName, const DataInput* forShape, int netTypeOverride)
{
	if (checkpointName.empty())
		return failStatus(NNetworkStatus::INVALID_ARGUMENT, "loadCheckpoint: checkpointName is empty");
	if (!is_safe_path_component(checkpointName))
		return failStatus(NNetworkStatus::INVALID_ARGUMENT, "loadCheckpoint: checkpointName contains unsafe characters");
	if (!forShape)
		return failStatus(NNetworkStatus::INVALID_ARGUMENT, "loadCheckpoint: forShape is null (required to allocate tensors)");

	// Do not create directories on load; but validate configured root exists.
	if (!stat_is_dir(checkpoint_root_dir()))
		return failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: checkpoint root directory does not exist");

	const std::string dir = checkpoint_dir(checkpointName);
	const std::string manifestPath = dir + "manifest.txt";
	const std::string nninfoPath = dir + "nninfo.csv";

	std::map<std::string, std::string> kv;
	int savedNetType = TYPE_DFF;
	{
		const NNetworkStatus stManifest = parse_checkpoint_manifest_header(manifestPath, kv, savedNetType);
		if (!stManifest.ok())
			return failStatus(stManifest.code, stManifest.message);
	}
	if (netTypeOverride >= 0)
		netType = netTypeOverride;
	else
		netType = savedNetType;

	{
		const NNetworkStatus stCfg = validate_checkpoint_training_config_compatibility(kv, trainingConfig);
		if (!stCfg.ok())
			return failStatus(stCfg.code, stCfg.message);
	}

	// Restore metadata/config before allocating tensors.
	uint64_t savedSeed = 0u;
	bool hasSavedSeed = false;
	bool includeOpt = true;
	restore_checkpoint_runtime_metadata(kv, epochs, includeOpt, savedSeed, hasSavedSeed, trainingConfig);
	if (hasSavedSeed)
		setSeed(savedSeed);

	// Optimizer-state completeness guard:
	// If the run is configured for AdamW, the checkpoint must include optimizer state.
	{
		const bool wantAdamW = (trainingConfig.optimizer.type == glades::OptimizerConfig::ADAMW);
		const bool isTransformer = (netType == TYPE_TRANSFORMER_ENCODER || netType == TYPE_TRANSFORMER_DECODER);
		if (wantAdamW)
		{
			if (!isTransformer)
				return failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: ADAMW optimizer checkpoints are only supported for transformer net types");
			if (!includeOpt)
				return failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: checkpoint does not include optimizer state required for ADAMW resume");
		}
	}

	// Load architecture snapshot
	{
		const shmea::GTable t(shmea::GString(nninfoPath.c_str()), ',', shmea::GTable::TYPE_FILE);
		ownedSkeleton = shmea::GPointer<NNInfo>(new NNInfo(shmea::GString(checkpointName.c_str()), t));
		skeleton = ownedSkeleton.get();
	}
	if (!skeleton)
		return failStatus(NNetworkStatus::INTERNAL_ERROR, "loadCheckpoint: failed to construct NNInfo from nninfo.csv");

	// Allocate tensors (requires attaching DataInput temporarily).
	const DataInput* prevDI = di;
	di = forShape;
	const bool okInit = ensureTensorParametersInitialized();
	di = prevDI;
	if (!okInit)
		return lastStatus;

	// Restore transformer optimizer step.
	// This is required for correct AdamW bias-correction behavior on resume.
	if (netType == TYPE_TRANSFORMER_ENCODER || netType == TYPE_TRANSFORMER_DECODER)
	{
		uint64_t step = 0u;
		const bool hasStep = parse_u64(kv, "transformer.optimizerStep", step);
		if (!hasStep && includeOpt && trainingConfig.optimizer.type == glades::OptimizerConfig::ADAMW)
			return failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: missing transformer.optimizerStep (required for ADAMW resume)");
		if (hasStep)
			tensorTransformer.optimizerStep = static_cast<unsigned long long>(step);

		// Restore mixed-precision loss scaling state (optional).
		float ls = 0.0f;
		if (parse_float(kv, "transformer.lossScale", ls))
			tensorTransformer.mpLossScale = ls;
		uint64_t good = 0u;
		if (parse_u64(kv, "transformer.lossScaleGoodSteps", good))
			tensorTransformer.mpLossScaleGoodSteps = static_cast<int>(good);
	}

	// Collect expected tensors for this net type.
	std::vector<TensorReadRef> expected;
	expected.clear();
	const std::string dt = "f32";
	const bool isAtlas = (trainingConfig.optimizer.type == glades::OptimizerConfig::ATLAS);
	const unsigned int atlasRank = trainingConfig.atlas.rank;
	if (netType == TYPE_DFF)
	{
		// Pre-allocate ATLAS state vector so checkpoint tensors can be read into it.
		if (includeOpt && isAtlas && tensorDff.atlasState.size() < tensorDff.T.size())
			tensorDff.atlasState.resize(tensorDff.T.size());

		for (size_t t = 0; t < tensorDff.T.size(); ++t)
		{
			TensorDFFState::Transition& tr = tensorDff.T[t];
			{
				std::ostringstream oss; oss << "dff.t" << static_cast<unsigned long long>(t) << ".W";
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(tr.out));
				sh.push_back(static_cast<uint64_t>(tr.in));
				expected.push_back(TensorReadRef(oss.str(), &tr.W, dt, sh));
			}
			{
				std::ostringstream oss; oss << "dff.t" << static_cast<unsigned long long>(t) << ".bias";
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(tr.out));
				expected.push_back(TensorReadRef(oss.str(), &tr.bias, dt, sh));
			}
			if (includeOpt)
			{
				std::ostringstream oss; oss << "dff.t" << static_cast<unsigned long long>(t) << ".vW";
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(tr.out));
				sh.push_back(static_cast<uint64_t>(tr.in));
				expected.push_back(TensorReadRef(oss.str(), &tr.vW, dt, sh));
			}
			if (includeOpt && isAtlas && t < tensorDff.atlasState.size())
			{
				const unsigned int m = tr.out, n = tr.in;
				const unsigned int r = std::min(atlasRank, std::min(m, n));
				std::ostringstream oss; oss << "dff.t" << static_cast<unsigned long long>(t);
				enqueueAtlasRead(expected, oss.str(), tensorDff.atlasState[t], m, n, r, dt);
			}
		}
	}
	else if (netType == TYPE_RNN)
	{
		for (size_t l = 0; l < tensorRnn.H.size(); ++l)
		{
			TensorRNNState::Hidden& hl = tensorRnn.H[l];
			{
				std::ostringstream oss; oss << "rnn.h" << static_cast<unsigned long long>(l) << ".Wxh";
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(hl.h));
				sh.push_back(static_cast<uint64_t>(hl.in));
				expected.push_back(TensorReadRef(oss.str(), &hl.Wxh, dt, sh));
			}
			{
				std::ostringstream oss; oss << "rnn.h" << static_cast<unsigned long long>(l) << ".Whh";
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(hl.h));
				sh.push_back(static_cast<uint64_t>(hl.h));
				expected.push_back(TensorReadRef(oss.str(), &hl.Whh, dt, sh));
			}
			{
				std::ostringstream oss; oss << "rnn.h" << static_cast<unsigned long long>(l) << ".bias";
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(hl.h));
				expected.push_back(TensorReadRef(oss.str(), &hl.bias, dt, sh));
			}
			if (includeOpt)
			{
				{
					std::ostringstream oss; oss << "rnn.h" << static_cast<unsigned long long>(l) << ".vWxh";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(hl.h));
					sh.push_back(static_cast<uint64_t>(hl.in));
					expected.push_back(TensorReadRef(oss.str(), &hl.vWxh, dt, sh));
				}
				{
					std::ostringstream oss; oss << "rnn.h" << static_cast<unsigned long long>(l) << ".vWhh";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(hl.h));
					sh.push_back(static_cast<uint64_t>(hl.h));
					expected.push_back(TensorReadRef(oss.str(), &hl.vWhh, dt, sh));
				}
			}
			if (includeOpt && isAtlas)
			{
				{
					const unsigned int m = hl.h, n = hl.in;
					const unsigned int r = std::min(atlasRank, std::min(m, n));
					std::ostringstream oss; oss << "rnn.h" << static_cast<unsigned long long>(l) << ".Wxh";
					enqueueAtlasRead(expected, oss.str(), hl.atlasWxh, m, n, r, dt);
				}
				{
					const unsigned int m = hl.h, n = hl.h;
					const unsigned int r = std::min(atlasRank, std::min(m, n));
					std::ostringstream oss; oss << "rnn.h" << static_cast<unsigned long long>(l) << ".Whh";
					enqueueAtlasRead(expected, oss.str(), hl.atlasWhh, m, n, r, dt);
				}
			}
		}
		{
			std::vector<uint64_t> sh;
			sh.push_back(static_cast<uint64_t>(tensorRnn.O.out));
			sh.push_back(static_cast<uint64_t>(tensorRnn.O.in));
			expected.push_back(TensorReadRef("rnn.o.Why", &tensorRnn.O.Why, dt, sh));
		}
		{
			std::vector<uint64_t> sh;
			sh.push_back(static_cast<uint64_t>(tensorRnn.O.out));
			expected.push_back(TensorReadRef("rnn.o.bias", &tensorRnn.O.bias, dt, sh));
		}
		if (includeOpt)
		{
			std::vector<uint64_t> sh;
			sh.push_back(static_cast<uint64_t>(tensorRnn.O.out));
			sh.push_back(static_cast<uint64_t>(tensorRnn.O.in));
			expected.push_back(TensorReadRef("rnn.o.vWhy", &tensorRnn.O.vWhy, dt, sh));
		}
		if (includeOpt && isAtlas)
		{
			const unsigned int m = tensorRnn.O.out, n = tensorRnn.O.in;
			const unsigned int r = std::min(atlasRank, std::min(m, n));
			enqueueAtlasRead(expected, "rnn.o", tensorRnn.O.atlasWhy, m, n, r, dt);
		}
	}
	else if (netType == TYPE_GRU || netType == TYPE_LSTM)
	{
		TensorGatedState& tg = (netType == TYPE_GRU) ? tensorGru : tensorLstm;
		const char* prefix = (netType == TYPE_GRU) ? "gru" : "lstm";
		for (size_t l = 0; l < tg.H.size(); ++l)
		{
			TensorGatedState::Hidden& hl = tg.H[l];
			{
				std::ostringstream oss; oss << prefix << ".h" << static_cast<unsigned long long>(l) << ".W";
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(tg.gateCount));
				sh.push_back(static_cast<uint64_t>(hl.h));
				sh.push_back(static_cast<uint64_t>(hl.in));
				expected.push_back(TensorReadRef(oss.str(), &hl.W, dt, sh));
			}
			{
				std::ostringstream oss; oss << prefix << ".h" << static_cast<unsigned long long>(l) << ".U";
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(tg.gateCount));
				sh.push_back(static_cast<uint64_t>(hl.h));
				sh.push_back(static_cast<uint64_t>(hl.h));
				expected.push_back(TensorReadRef(oss.str(), &hl.U, dt, sh));
			}
			{
				std::ostringstream oss; oss << prefix << ".h" << static_cast<unsigned long long>(l) << ".bias";
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(tg.gateCount));
				sh.push_back(static_cast<uint64_t>(hl.h));
				expected.push_back(TensorReadRef(oss.str(), &hl.bias, dt, sh));
			}
			if (includeOpt)
			{
				{
					std::ostringstream oss; oss << prefix << ".h" << static_cast<unsigned long long>(l) << ".vW";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(tg.gateCount));
					sh.push_back(static_cast<uint64_t>(hl.h));
					sh.push_back(static_cast<uint64_t>(hl.in));
					expected.push_back(TensorReadRef(oss.str(), &hl.vW, dt, sh));
				}
				{
					std::ostringstream oss; oss << prefix << ".h" << static_cast<unsigned long long>(l) << ".vU";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(tg.gateCount));
					sh.push_back(static_cast<uint64_t>(hl.h));
					sh.push_back(static_cast<uint64_t>(hl.h));
					expected.push_back(TensorReadRef(oss.str(), &hl.vU, dt, sh));
				}
			}
			if (includeOpt && isAtlas)
			{
				{
					const unsigned int m = tg.gateCount * hl.h, n = hl.in;
					const unsigned int r = std::min(atlasRank, std::min(m, n));
					std::ostringstream oss; oss << prefix << ".h" << static_cast<unsigned long long>(l) << ".W";
					enqueueAtlasRead(expected, oss.str(), hl.atlasW, m, n, r, dt);
				}
				{
					const unsigned int m = tg.gateCount * hl.h, n = hl.h;
					const unsigned int r = std::min(atlasRank, std::min(m, n));
					std::ostringstream oss; oss << prefix << ".h" << static_cast<unsigned long long>(l) << ".U";
					enqueueAtlasRead(expected, oss.str(), hl.atlasU, m, n, r, dt);
				}
			}
		}
		{
			std::vector<uint64_t> sh;
			sh.push_back(static_cast<uint64_t>(tg.O.out));
			sh.push_back(static_cast<uint64_t>(tg.O.in));
			expected.push_back(TensorReadRef(std::string(prefix) + ".o.Why", &tg.O.Why, dt, sh));
		}
		{
			std::vector<uint64_t> sh;
			sh.push_back(static_cast<uint64_t>(tg.O.out));
			expected.push_back(TensorReadRef(std::string(prefix) + ".o.bias", &tg.O.bias, dt, sh));
		}
		if (includeOpt)
		{
			std::vector<uint64_t> sh;
			sh.push_back(static_cast<uint64_t>(tg.O.out));
			sh.push_back(static_cast<uint64_t>(tg.O.in));
			expected.push_back(TensorReadRef(std::string(prefix) + ".o.vWhy", &tg.O.vWhy, dt, sh));
		}
		if (includeOpt && isAtlas)
		{
			const unsigned int m = tg.O.out, n = tg.O.in;
			const unsigned int r = std::min(atlasRank, std::min(m, n));
			enqueueAtlasRead(expected, std::string(prefix) + ".o", tg.O.atlasWhy, m, n, r, dt);
		}
	}
	else if (netType == TYPE_TRANSFORMER_ENCODER || netType == TYPE_TRANSFORMER_DECODER)
	{
		TensorTransformerState& tt = tensorTransformer;
		const unsigned int dModel = tt.dModel;
		const unsigned int inputSize = tt.inputSize;
		const unsigned int outSize = tt.outSize;
		const unsigned int nHeads = tt.nHeads;
		const unsigned int nKVHeads = (tt.nKVHeads > 0u ? tt.nKVHeads : nHeads);
		const unsigned int dHead = (nHeads > 0u ? (dModel / nHeads) : 0u);
		const unsigned int dModelKV = nKVHeads * dHead;
		const unsigned int ff1Width =
		    (tt.ffnKind == static_cast<unsigned int>(glades::TransformerRunConfig::FFN_SWIGLU)) ? (2u * tt.dFF) : tt.dFF;

		{
			std::vector<uint64_t> sh;
			sh.push_back(static_cast<uint64_t>(dModel));
			sh.push_back(static_cast<uint64_t>(inputSize));
			expected.push_back(TensorReadRef("tr.WIn", &tt.WIn, dt, sh));
		}
		{
			std::vector<uint64_t> sh;
			sh.push_back(static_cast<uint64_t>(dModel));
			expected.push_back(TensorReadRef("tr.bIn", &tt.bIn, dt, sh));
		}
		{
			std::vector<uint64_t> sh;
			sh.push_back(static_cast<uint64_t>(outSize));
			sh.push_back(static_cast<uint64_t>(dModel));
			expected.push_back(TensorReadRef("tr.WOut", &tt.WOut, dt, sh));
		}
		{
			std::vector<uint64_t> sh;
			sh.push_back(static_cast<uint64_t>(outSize));
			expected.push_back(TensorReadRef("tr.bOut", &tt.bOut, dt, sh));
		}
		// Token-LM tensors are only present when tokenModel==true.
		if (tt.tokenModel)
		{
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(tt.vocabSize));
				sh.push_back(static_cast<uint64_t>(dModel));
				expected.push_back(TensorReadRef("tr.tokE", &tt.tokE, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(tt.vocabSize));
				expected.push_back(TensorReadRef("tr.lmBias", &tt.lmBias, dt, sh));
			}
		}
		// Final LayerNorm (optional for backward compat with old checkpoints)
		{
			std::vector<uint64_t> sh;
			sh.push_back(static_cast<uint64_t>(dModel));
			expected.push_back(TensorReadRef("tr.lnFinalGamma", &tt.lnFinalGamma, dt, sh));
			expected.push_back(TensorReadRef("tr.lnFinalBeta", &tt.lnFinalBeta, dt, sh));
		}
		if (includeOpt)
		{
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(dModel));
				sh.push_back(static_cast<uint64_t>(inputSize));
				expected.push_back(TensorReadRef("tr.vWIn", &tt.vWIn, dt, sh));
				expected.push_back(TensorReadRef("tr.v2WIn", &tt.v2WIn, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(dModel));
				expected.push_back(TensorReadRef("tr.mBIn", &tt.mBIn, dt, sh));
				expected.push_back(TensorReadRef("tr.v2BIn", &tt.v2BIn, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(outSize));
				sh.push_back(static_cast<uint64_t>(dModel));
				expected.push_back(TensorReadRef("tr.vWOut", &tt.vWOut, dt, sh));
				expected.push_back(TensorReadRef("tr.v2WOut", &tt.v2WOut, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(outSize));
				expected.push_back(TensorReadRef("tr.mBOut", &tt.mBOut, dt, sh));
				expected.push_back(TensorReadRef("tr.v2BOut", &tt.v2BOut, dt, sh));
			}
			{
				// Token-LM optimizer tensors are only present when tokenModel==true.
				if (tt.tokenModel)
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(tt.vocabSize));
					sh.push_back(static_cast<uint64_t>(dModel));
					expected.push_back(TensorReadRef("tr.vTokE", &tt.vTokE, dt, sh));
					expected.push_back(TensorReadRef("tr.v2TokE", &tt.v2TokE, dt, sh));
				}
			}
			{
				if (tt.tokenModel)
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(tt.vocabSize));
					expected.push_back(TensorReadRef("tr.mLmBias", &tt.mLmBias, dt, sh));
					expected.push_back(TensorReadRef("tr.v2LmBias", &tt.v2LmBias, dt, sh));
				}
			}
			// Final LayerNorm optimizer state (optional for backward compat)
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(dModel));
				expected.push_back(TensorReadRef("tr.mLnFinalGamma", &tt.mLnFinalGamma, dt, sh));
				expected.push_back(TensorReadRef("tr.v2LnFinalGamma", &tt.v2LnFinalGamma, dt, sh));
				expected.push_back(TensorReadRef("tr.mLnFinalBeta", &tt.mLnFinalBeta, dt, sh));
				expected.push_back(TensorReadRef("tr.v2LnFinalBeta", &tt.v2LnFinalBeta, dt, sh));
			}
		}
		for (size_t l = 0; l < tt.blocks.size(); ++l)
		{
			TensorTransformerState::Block& b = tt.blocks[l];
			const unsigned long long li = static_cast<unsigned long long>(l);
			{
				std::ostringstream oss; oss << "tr.b" << li << ".ln1Gamma";
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(dModel));
				expected.push_back(TensorReadRef(oss.str(), &b.ln1Gamma, dt, sh));
			}
			{
				std::ostringstream oss; oss << "tr.b" << li << ".ln1Beta";
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(dModel));
				expected.push_back(TensorReadRef(oss.str(), &b.ln1Beta, dt, sh));
			}
			{
				std::ostringstream oss; oss << "tr.b" << li << ".Wq";
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(dModel));
				sh.push_back(static_cast<uint64_t>(dModel));
				expected.push_back(TensorReadRef(oss.str(), &b.Wq, dt, sh));
			}
			{
				std::ostringstream oss; oss << "tr.b" << li << ".Wk";
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(dModelKV));
				sh.push_back(static_cast<uint64_t>(dModel));
				expected.push_back(TensorReadRef(oss.str(), &b.Wk, dt, sh));
			}
			{
				std::ostringstream oss; oss << "tr.b" << li << ".Wv";
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(dModelKV));
				sh.push_back(static_cast<uint64_t>(dModel));
				expected.push_back(TensorReadRef(oss.str(), &b.Wv, dt, sh));
			}
			{
				std::ostringstream oss; oss << "tr.b" << li << ".Wo";
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(dModel));
				sh.push_back(static_cast<uint64_t>(dModel));
				expected.push_back(TensorReadRef(oss.str(), &b.Wo, dt, sh));
			}
			{
				std::ostringstream oss; oss << "tr.b" << li << ".bq";
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(dModel));
				expected.push_back(TensorReadRef(oss.str(), &b.bq, dt, sh));
			}
			{
				std::ostringstream oss; oss << "tr.b" << li << ".bk";
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(dModelKV));
				expected.push_back(TensorReadRef(oss.str(), &b.bk, dt, sh));
			}
			{
				std::ostringstream oss; oss << "tr.b" << li << ".bv";
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(dModelKV));
				expected.push_back(TensorReadRef(oss.str(), &b.bv, dt, sh));
			}
			{
				std::ostringstream oss; oss << "tr.b" << li << ".bo";
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(dModel));
				expected.push_back(TensorReadRef(oss.str(), &b.bo, dt, sh));
			}
			{
				std::ostringstream oss; oss << "tr.b" << li << ".ln2Gamma";
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(dModel));
				expected.push_back(TensorReadRef(oss.str(), &b.ln2Gamma, dt, sh));
			}
			{
				std::ostringstream oss; oss << "tr.b" << li << ".ln2Beta";
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(dModel));
				expected.push_back(TensorReadRef(oss.str(), &b.ln2Beta, dt, sh));
			}
			{
				std::ostringstream oss; oss << "tr.b" << li << ".W1";
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(ff1Width));
				sh.push_back(static_cast<uint64_t>(dModel));
				expected.push_back(TensorReadRef(oss.str(), &b.W1, dt, sh));
			}
			{
				std::ostringstream oss; oss << "tr.b" << li << ".b1";
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(ff1Width));
				expected.push_back(TensorReadRef(oss.str(), &b.b1, dt, sh));
			}
			{
				std::ostringstream oss; oss << "tr.b" << li << ".W2";
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(dModel));
				sh.push_back(static_cast<uint64_t>(tt.dFF));
				expected.push_back(TensorReadRef(oss.str(), &b.W2, dt, sh));
			}
			{
				std::ostringstream oss; oss << "tr.b" << li << ".b2";
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(dModel));
				expected.push_back(TensorReadRef(oss.str(), &b.b2, dt, sh));
			}
			if (includeOpt)
			{
				{
					std::ostringstream oss; oss << "tr.b" << li << ".mLn1Gamma";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModel));
					expected.push_back(TensorReadRef(oss.str(), &b.mLn1Gamma, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".v2Ln1Gamma";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModel));
					expected.push_back(TensorReadRef(oss.str(), &b.v2Ln1Gamma, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".mLn1Beta";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModel));
					expected.push_back(TensorReadRef(oss.str(), &b.mLn1Beta, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".v2Ln1Beta";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModel));
					expected.push_back(TensorReadRef(oss.str(), &b.v2Ln1Beta, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".vWq";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModel));
					sh.push_back(static_cast<uint64_t>(dModel));
					expected.push_back(TensorReadRef(oss.str(), &b.vWq, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".v2Wq";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModel));
					sh.push_back(static_cast<uint64_t>(dModel));
					expected.push_back(TensorReadRef(oss.str(), &b.v2Wq, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".vWk";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModelKV));
					sh.push_back(static_cast<uint64_t>(dModel));
					expected.push_back(TensorReadRef(oss.str(), &b.vWk, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".v2Wk";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModelKV));
					sh.push_back(static_cast<uint64_t>(dModel));
					expected.push_back(TensorReadRef(oss.str(), &b.v2Wk, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".vWv";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModelKV));
					sh.push_back(static_cast<uint64_t>(dModel));
					expected.push_back(TensorReadRef(oss.str(), &b.vWv, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".v2Wv";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModelKV));
					sh.push_back(static_cast<uint64_t>(dModel));
					expected.push_back(TensorReadRef(oss.str(), &b.v2Wv, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".vWo";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModel));
					sh.push_back(static_cast<uint64_t>(dModel));
					expected.push_back(TensorReadRef(oss.str(), &b.vWo, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".v2Wo";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModel));
					sh.push_back(static_cast<uint64_t>(dModel));
					expected.push_back(TensorReadRef(oss.str(), &b.v2Wo, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".mBq";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModel));
					expected.push_back(TensorReadRef(oss.str(), &b.mBq, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".v2Bq";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModel));
					expected.push_back(TensorReadRef(oss.str(), &b.v2Bq, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".mBk";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModelKV));
					expected.push_back(TensorReadRef(oss.str(), &b.mBk, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".v2Bk";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModelKV));
					expected.push_back(TensorReadRef(oss.str(), &b.v2Bk, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".mBv";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModelKV));
					expected.push_back(TensorReadRef(oss.str(), &b.mBv, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".v2Bv";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModelKV));
					expected.push_back(TensorReadRef(oss.str(), &b.v2Bv, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".mBo";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModel));
					expected.push_back(TensorReadRef(oss.str(), &b.mBo, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".v2Bo";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModel));
					expected.push_back(TensorReadRef(oss.str(), &b.v2Bo, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".mLn2Gamma";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModel));
					expected.push_back(TensorReadRef(oss.str(), &b.mLn2Gamma, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".v2Ln2Gamma";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModel));
					expected.push_back(TensorReadRef(oss.str(), &b.v2Ln2Gamma, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".mLn2Beta";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModel));
					expected.push_back(TensorReadRef(oss.str(), &b.mLn2Beta, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".v2Ln2Beta";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModel));
					expected.push_back(TensorReadRef(oss.str(), &b.v2Ln2Beta, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".vW1";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(ff1Width));
					sh.push_back(static_cast<uint64_t>(dModel));
					expected.push_back(TensorReadRef(oss.str(), &b.vW1, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".v2W1";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(ff1Width));
					sh.push_back(static_cast<uint64_t>(dModel));
					expected.push_back(TensorReadRef(oss.str(), &b.v2W1, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".mB1";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(ff1Width));
					expected.push_back(TensorReadRef(oss.str(), &b.mB1, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".v2B1";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(ff1Width));
					expected.push_back(TensorReadRef(oss.str(), &b.v2B1, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".vW2";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModel));
					sh.push_back(static_cast<uint64_t>(tt.dFF));
					expected.push_back(TensorReadRef(oss.str(), &b.vW2, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".v2W2";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModel));
					sh.push_back(static_cast<uint64_t>(tt.dFF));
					expected.push_back(TensorReadRef(oss.str(), &b.v2W2, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".mB2";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModel));
					expected.push_back(TensorReadRef(oss.str(), &b.mB2, dt, sh));
				}
				{
					std::ostringstream oss; oss << "tr.b" << li << ".v2B2";
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(dModel));
					expected.push_back(TensorReadRef(oss.str(), &b.v2B2, dt, sh));
				}
			}
			if (includeOpt && isAtlas)
			{
				{
					const unsigned int m = dModel, n = dModel;
					const unsigned int r = std::min(atlasRank, std::min(m, n));
					std::ostringstream oss; oss << "tr.b" << li << ".Wq";
					enqueueAtlasRead(expected, oss.str(), b.atlasWq, m, n, r, dt);
				}
				{
					const unsigned int m = dModelKV, n = dModel;
					const unsigned int r = std::min(atlasRank, std::min(m, n));
					std::ostringstream oss; oss << "tr.b" << li << ".Wk";
					enqueueAtlasRead(expected, oss.str(), b.atlasWk, m, n, r, dt);
				}
				{
					const unsigned int m = dModelKV, n = dModel;
					const unsigned int r = std::min(atlasRank, std::min(m, n));
					std::ostringstream oss; oss << "tr.b" << li << ".Wv";
					enqueueAtlasRead(expected, oss.str(), b.atlasWv, m, n, r, dt);
				}
				{
					const unsigned int m = dModel, n = dModel;
					const unsigned int r = std::min(atlasRank, std::min(m, n));
					std::ostringstream oss; oss << "tr.b" << li << ".Wo";
					enqueueAtlasRead(expected, oss.str(), b.atlasWo, m, n, r, dt);
				}
				{
					const unsigned int m = ff1Width, n = dModel;
					const unsigned int r = std::min(atlasRank, std::min(m, n));
					std::ostringstream oss; oss << "tr.b" << li << ".W1";
					enqueueAtlasRead(expected, oss.str(), b.atlasW1, m, n, r, dt);
				}
				{
					const unsigned int m = dModel, n = tt.dFF;
					const unsigned int r = std::min(atlasRank, std::min(m, n));
					std::ostringstream oss; oss << "tr.b" << li << ".W2";
					enqueueAtlasRead(expected, oss.str(), b.atlasW2, m, n, r, dt);
				}
			}
		}
		if (includeOpt && isAtlas)
		{
			{
				const unsigned int m = dModel, n = inputSize;
				const unsigned int r = std::min(atlasRank, std::min(m, n));
				enqueueAtlasRead(expected, "tr.WIn", tt.atlasWIn, m, n, r, dt);
			}
			{
				const unsigned int m = outSize, n = dModel;
				const unsigned int r = std::min(atlasRank, std::min(m, n));
				enqueueAtlasRead(expected, "tr.WOut", tt.atlasWOut, m, n, r, dt);
			}
			if (tt.tokenModel)
			{
				const unsigned int m = tt.vocabSize, n = dModel;
				const unsigned int r = std::min(atlasRank, std::min(m, n));
				enqueueAtlasRead(expected, "tr.tokE", tt.atlasTokE, m, n, r, dt);
			}
		}
	}
	else
	{
		return failStatus(NNetworkStatus::INVALID_ARGUMENT, "loadCheckpoint: unknown netType");
	}
	std::map<std::string, std::vector<float>*> nameToVec;
	std::map<std::string, std::string> nameToDType;
	std::map<std::string, std::vector<uint64_t> > nameToShape;
	for (size_t i = 0; i < expected.size(); ++i)
	{
		if (expected[i].vec)
		{
			nameToVec[expected[i].name] = expected[i].vec;
			nameToDType[expected[i].name] = expected[i].dtype;
			nameToShape[expected[i].name] = expected[i].shape;
		}
	}
	// Optional tensor names: these may be absent in old checkpoints (backward compat).
	// If not found in the checkpoint, they keep their initialized defaults.
	std::set<std::string> optionalNames;
	optionalNames.insert("tr.lnFinalGamma");
	optionalNames.insert("tr.lnFinalBeta");
	optionalNames.insert("tr.mLnFinalGamma");
	optionalNames.insert("tr.v2LnFinalGamma");
	optionalNames.insert("tr.mLnFinalBeta");
	optionalNames.insert("tr.v2LnFinalBeta");
	// ATLAS tensors are optional (old checkpoints won't have them).
	for (size_t i = 0; i < expected.size(); ++i)
	{
		if (expected[i].name.find(".atlas.") != std::string::npos)
			optionalNames.insert(expected[i].name);
	}

	// Parse shardCount/tensorCount
	size_t shardCount = 0u;
	size_t tensorCount = 0u;
	parse_size_t(kv, "shardCount", shardCount);
	parse_size_t(kv, "tensorCount", tensorCount);
	if (shardCount == 0u && tensorCount != 0u)
		return failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: shardCount is 0 but tensorCount is non-zero");

	std::vector< shmea::GPointer<std::ifstream> > shardStreams;
	std::vector<uint64_t> shardBytes;
	std::vector<uint64_t> shardHashExpected;
	{
		const NNetworkStatus stShards =
		    open_checkpoint_shards(dir, kv, shardCount, shardStreams, shardBytes, shardHashExpected);
		if (!stShards.ok())
		{
			close_and_delete_shards(shardStreams);
			return failStatus(stShards.code, stShards.message);
		}
	}

	// Load each tensor entry into its destination vector.
	for (size_t i = 0; i < tensorCount; ++i)
	{
		ParsedCheckpointTensorEntry tensorEntry;
		const NNetworkStatus stTensor =
		    parse_checkpoint_tensor_manifest_entry(kv, i, shardCount, shardBytes, tensorEntry);
		if (!stTensor.ok())
		{
			close_and_delete_shards(shardStreams);
			return failStatus(stTensor.code, stTensor.message);
		}

		std::map<std::string, std::vector<float>*>::iterator it = nameToVec.find(tensorEntry.name);
		if (it == nameToVec.end())
		{
			// ATLAS tensors in old checkpoints can be safely skipped if not expected.
			if (tensorEntry.name.find(".atlas.") != std::string::npos)
				continue;
			// Unknown tensor in checkpoint; reject (strict).
			const NNetworkStatus st =
			    failStatus(NNetworkStatus::INVALID_STATE, std::string("loadCheckpoint: unexpected tensor in checkpoint: ") + tensorEntry.name);
			close_and_delete_shards(shardStreams);
			return st;
		}
		std::vector<float>* dst = it->second;
		if (!dst)
		{
			const NNetworkStatus st = failStatus(NNetworkStatus::INTERNAL_ERROR, "loadCheckpoint: internal error (null dst)");
			close_and_delete_shards(shardStreams);
			return st;
		}

		const size_t wantCount = static_cast<size_t>(tensorEntry.count);
		// Strict shape compatibility: element count must match what we allocated.
		if (dst->size() != wantCount)
		{
			std::ostringstream oss;
			oss << "loadCheckpoint: tensor size mismatch for " << tensorEntry.name << " (checkpoint " << static_cast<unsigned long long>(wantCount)
			    << " vs model " << static_cast<unsigned long long>(dst->size()) << ")";
			const NNetworkStatus st = failStatus(NNetworkStatus::INVALID_STATE, oss.str());
			close_and_delete_shards(shardStreams);
			return st;
		}

		// Shape/dtype must match the expected tensor spec (stronger than count-only).
		{
			std::map<std::string, std::string>::const_iterator itd = nameToDType.find(tensorEntry.name);
			std::map<std::string, std::vector<uint64_t> >::const_iterator its = nameToShape.find(tensorEntry.name);
			if (itd == nameToDType.end() || its == nameToShape.end())
			{
				const NNetworkStatus st = failStatus(NNetworkStatus::INTERNAL_ERROR, "loadCheckpoint: internal error (missing expected tensor metadata)");
				close_and_delete_shards(shardStreams);
				return st;
			}
			if (tensorEntry.dtype != itd->second)
			{
				const NNetworkStatus st = failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: tensor dtype mismatch (checkpoint vs model)");
				close_and_delete_shards(shardStreams);
				return st;
			}
			if (tensorEntry.shape != its->second)
			{
				const NNetworkStatus st = failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: tensor shape mismatch (checkpoint vs model)");
				close_and_delete_shards(shardStreams);
				return st;
			}
		}

		uint64_t computed = 0u;
		if (!shardStreams[tensorEntry.shardIndex] ||
		    !read_f32_blob_from_shard(*shardStreams[tensorEntry.shardIndex], tensorEntry.offsetBytes,
		                              *dst, wantCount, tensorEntry.fnv1a64, computed))
		{
			const NNetworkStatus st =
			    failStatus(NNetworkStatus::INVALID_STATE, std::string("loadCheckpoint: checksum/read failed for tensor ") + tensorEntry.name);
			close_and_delete_shards(shardStreams);
			return st;
		}

		// Mark consumed to detect missing tensors later.
		nameToVec.erase(it);
	}

	// Remove optional tensors that were not in the checkpoint (backward compat).
	for (std::set<std::string>::const_iterator oit = optionalNames.begin(); oit != optionalNames.end(); ++oit)
		nameToVec.erase(*oit);

	// Ensure no expected tensors are missing (strict).
	if (!nameToVec.empty())
	{
		const NNetworkStatus st = failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: checkpoint missing required tensors for this model/config");
		close_and_delete_shards(shardStreams);
		return st;
	}

	// Cleanup streams
	close_and_delete_shards(shardStreams);

	// Restore ATLAS optimizer scalar state (mu, step) from manifest KV entries.
	if (includeOpt && isAtlas)
	{
		if (netType == TYPE_DFF)
		{
			for (size_t t = 0; t < tensorDff.atlasState.size(); ++t)
			{
				std::ostringstream oss; oss << "dff.t" << static_cast<unsigned long long>(t);
				readAtlasManifestKV(kv, oss.str(), tensorDff.atlasState[t]);
			}
		}
		else if (netType == TYPE_RNN)
		{
			for (size_t l = 0; l < tensorRnn.H.size(); ++l)
			{
				TensorRNNState::Hidden& hl = tensorRnn.H[l];
				{
					std::ostringstream oss; oss << "rnn.h" << static_cast<unsigned long long>(l) << ".Wxh";
					readAtlasManifestKV(kv, oss.str(), hl.atlasWxh);
				}
				{
					std::ostringstream oss; oss << "rnn.h" << static_cast<unsigned long long>(l) << ".Whh";
					readAtlasManifestKV(kv, oss.str(), hl.atlasWhh);
				}
			}
			readAtlasManifestKV(kv, "rnn.o", tensorRnn.O.atlasWhy);
		}
		else if (netType == TYPE_GRU || netType == TYPE_LSTM)
		{
			TensorGatedState& tg = (netType == TYPE_GRU) ? tensorGru : tensorLstm;
			const char* prefix = (netType == TYPE_GRU) ? "gru" : "lstm";
			for (size_t l = 0; l < tg.H.size(); ++l)
			{
				TensorGatedState::Hidden& hl = tg.H[l];
				{
					std::ostringstream oss; oss << prefix << ".h" << static_cast<unsigned long long>(l) << ".W";
					readAtlasManifestKV(kv, oss.str(), hl.atlasW);
				}
				{
					std::ostringstream oss; oss << prefix << ".h" << static_cast<unsigned long long>(l) << ".U";
					readAtlasManifestKV(kv, oss.str(), hl.atlasU);
				}
			}
			readAtlasManifestKV(kv, std::string(prefix) + ".o", tg.O.atlasWhy);
		}
		else if (netType == TYPE_TRANSFORMER_ENCODER || netType == TYPE_TRANSFORMER_DECODER)
		{
			TensorTransformerState& tt = tensorTransformer;
			readAtlasManifestKV(kv, "tr.WIn", tt.atlasWIn);
			readAtlasManifestKV(kv, "tr.WOut", tt.atlasWOut);
			if (tt.tokenModel)
				readAtlasManifestKV(kv, "tr.tokE", tt.atlasTokE);
			for (size_t l = 0; l < tt.blocks.size(); ++l)
			{
				TensorTransformerState::Block& b = tt.blocks[l];
				const unsigned long long li = static_cast<unsigned long long>(l);
				std::ostringstream base; base << "tr.b" << li;
				const std::string bp = base.str();
				readAtlasManifestKV(kv, bp + ".Wq", b.atlasWq);
				readAtlasManifestKV(kv, bp + ".Wk", b.atlasWk);
				readAtlasManifestKV(kv, bp + ".Wv", b.atlasWv);
				readAtlasManifestKV(kv, bp + ".Wo", b.atlasWo);
				readAtlasManifestKV(kv, bp + ".W1", b.atlasW1);
				readAtlasManifestKV(kv, bp + ".W2", b.atlasW2);
			}
		}
	}

	return NNetworkStatus(NNetworkStatus::OK, std::string());
}

} // namespace glades
