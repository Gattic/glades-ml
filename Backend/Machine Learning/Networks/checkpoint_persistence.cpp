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
	if (parse_bool01(kv, "training.optimizer.adamGroupwiseEnabled", b)) { cfg.optimizer.adamGroupwiseEnabled = b; any = true; }
	if (parse_float(kv, "training.optimizer.adamGroupStabilityScale", f)) { cfg.optimizer.adamGroupStabilityScale = f; any = true; }
	if (parse_float(kv, "training.optimizer.adamGroupSnrScale", f)) { cfg.optimizer.adamGroupSnrScale = f; any = true; }
	if (parse_float(kv, "training.optimizer.adamGroupRatioScale", f)) { cfg.optimizer.adamGroupRatioScale = f; any = true; }
	if (parse_float(kv, "training.optimizer.adamGroupMinScale", f)) { cfg.optimizer.adamGroupMinScale = f; any = true; }
	if (parse_float(kv, "training.optimizer.adamGroupMaxScale", f)) { cfg.optimizer.adamGroupMaxScale = f; any = true; }
	if (parse_int(kv, "training.optimizer.adamGroupMinSize", i) && i >= 0) { cfg.optimizer.adamGroupMinSize = static_cast<unsigned int>(i); any = true; }
	if (parse_int(kv, "training.atlas.rank", i) && i >= 0) { cfg.atlas.rank = static_cast<unsigned int>(i); any = true; }
	if (parse_int(kv, "training.atlas.complementRank", i) && i >= 0) { cfg.atlas.complementRank = static_cast<unsigned int>(i); any = true; }
	if (parse_float(kv, "training.atlas.kappaMax", f)) { cfg.atlas.kappaMax = f; any = true; }
	if (parse_float(kv, "training.atlas.complementLrScale", f)) { cfg.atlas.complementLrScale = f; any = true; }
	if (parse_float(kv, "training.atlas.complementKappaMax", f)) { cfg.atlas.complementKappaMax = f; any = true; }
	if (parse_bool01(kv, "training.atlas.prismEnabled", b)) { cfg.atlas.prismEnabled = b; any = true; }
	if (parse_int(kv, "training.atlas.prismLagHorizon", i) && i >= 0) { cfg.atlas.prismLagHorizon = static_cast<unsigned int>(i); any = true; }
	if (parse_float(kv, "training.atlas.prismMemoryScale", f)) { cfg.atlas.prismMemoryScale = f; any = true; }
	if (parse_float(kv, "training.atlas.prismPredictiveEdgeThreshold", f)) { cfg.atlas.prismPredictiveEdgeThreshold = f; any = true; }
	if (parse_bool01(kv, "training.atlas.resolveEnabled", b)) { cfg.atlas.resolveEnabled = b; any = true; }
	if (parse_int(kv, "training.atlas.resolveLagHorizon", i) && i >= 0) { cfg.atlas.resolveLagHorizon = static_cast<unsigned int>(i); any = true; }
	if (parse_float(kv, "training.atlas.resolveMemoryScale", f)) { cfg.atlas.resolveMemoryScale = f; any = true; }
	if (parse_float(kv, "training.atlas.resolvePredictiveEdgeThreshold", f)) { cfg.atlas.resolvePredictiveEdgeThreshold = f; any = true; }
	if (parse_bool01(kv, "training.atlas.heroEnabled", b)) { cfg.atlas.heroEnabled = b; any = true; }
	if (parse_int(kv, "training.atlas.heroLagHorizon", i) && i >= 0) { cfg.atlas.heroLagHorizon = static_cast<unsigned int>(i); any = true; }
	if (parse_float(kv, "training.atlas.heroMemoryScale", f)) { cfg.atlas.heroMemoryScale = f; any = true; }
	if (parse_float(kv, "training.atlas.heroEdgeThreshold", f)) { cfg.atlas.heroEdgeThreshold = f; any = true; }
	if (parse_bool01(kv, "training.atlas.cobaltEnabled", b)) { cfg.atlas.cobaltEnabled = b; any = true; }
	if (parse_int(kv, "training.atlas.cobaltLagHorizon", i) && i >= 0) { cfg.atlas.cobaltLagHorizon = static_cast<unsigned int>(i); any = true; }
	if (parse_float(kv, "training.atlas.cobaltMemoryScale", f)) { cfg.atlas.cobaltMemoryScale = f; any = true; }
	if (parse_float(kv, "training.atlas.cobaltEdgeThreshold", f)) { cfg.atlas.cobaltEdgeThreshold = f; any = true; }
	if (parse_bool01(kv, "training.atlas.birchEnabled", b)) { cfg.atlas.birchEnabled = b; any = true; }
	if (parse_int(kv, "training.atlas.birchPastHorizon", i) && i >= 0) { cfg.atlas.birchPastHorizon = static_cast<unsigned int>(i); any = true; }
	if (parse_int(kv, "training.atlas.birchFutureHorizon", i) && i >= 0) { cfg.atlas.birchFutureHorizon = static_cast<unsigned int>(i); any = true; }
	if (parse_float(kv, "training.atlas.birchMemoryScale", f)) { cfg.atlas.birchMemoryScale = f; any = true; }
	if (parse_float(kv, "training.atlas.birchEdgeThreshold", f)) { cfg.atlas.birchEdgeThreshold = f; any = true; }
	if (parse_bool01(kv, "training.atlas.ghostEnabled", b)) { cfg.atlas.ghostEnabled = b; any = true; }
	if (parse_int(kv, "training.atlas.ghostLagHorizon", i) && i >= 0) { cfg.atlas.ghostLagHorizon = static_cast<unsigned int>(i); any = true; }
	if (parse_float(kv, "training.atlas.ghostMemoryScale", f)) { cfg.atlas.ghostMemoryScale = f; any = true; }
	if (parse_float(kv, "training.atlas.ghostEdgeThreshold", f)) { cfg.atlas.ghostEdgeThreshold = f; any = true; }
	if (parse_bool01(kv, "training.atlas.sparrowEnabled", b)) { cfg.atlas.sparrowEnabled = b; any = true; }
	if (parse_int(kv, "training.atlas.sparrowModeRank", i) && i >= 0) { cfg.atlas.sparrowModeRank = static_cast<unsigned int>(i); any = true; }
	if (parse_bool01(kv, "training.atlas.sparrowAutoModeGate", b)) { cfg.atlas.sparrowAutoModeGate = b; any = true; }
	if (parse_float(kv, "training.atlas.sparrowMemoryScale", f)) { cfg.atlas.sparrowMemoryScale = f; any = true; }
	if (parse_float(kv, "training.atlas.sparrowEdgeThreshold", f)) { cfg.atlas.sparrowEdgeThreshold = f; any = true; }
	if (parse_float(kv, "training.atlas.sparrowSecondEdgeThreshold", f)) { cfg.atlas.sparrowSecondEdgeThreshold = f; any = true; }
	if (parse_float(kv, "training.atlas.sparrowSecondEdgeFraction", f)) { cfg.atlas.sparrowSecondEdgeFraction = f; any = true; }
	if (parse_float(kv, "training.atlas.sparrowPoleMax", f)) { cfg.atlas.sparrowPoleMax = f; any = true; }
	if (parse_bool01(kv, "training.atlas.qbrtEnabled", b)) { cfg.atlas.qbrtEnabled = b; any = true; }
	if (parse_int(kv, "training.atlas.qbrtLagHorizon", i) && i >= 0) { cfg.atlas.qbrtLagHorizon = static_cast<unsigned int>(i); any = true; }
	if (parse_float(kv, "training.atlas.qbrtMemoryScale", f)) { cfg.atlas.qbrtMemoryScale = f; any = true; }
	if (parse_float(kv, "training.atlas.qbrtEdgeThreshold", f)) { cfg.atlas.qbrtEdgeThreshold = f; any = true; }
	if (parse_float(kv, "training.atlas.qbrtPoleMax", f)) { cfg.atlas.qbrtPoleMax = f; any = true; }
	if (parse_bool01(kv, "training.atlas.qrcEnabled", b)) { cfg.atlas.qrcEnabled = b; any = true; }
	if (parse_int(kv, "training.atlas.qrcLagHorizon", i) && i >= 0) { cfg.atlas.qrcLagHorizon = static_cast<unsigned int>(i); any = true; }
	if (parse_float(kv, "training.atlas.qrcMemoryScale", f)) { cfg.atlas.qrcMemoryScale = f; any = true; }
	if (parse_float(kv, "training.atlas.qrcEdgeThreshold", f)) { cfg.atlas.qrcEdgeThreshold = f; any = true; }
	if (parse_float(kv, "training.atlas.qrcPoleMax", f)) { cfg.atlas.qrcPoleMax = f; any = true; }
	if (parse_bool01(kv, "training.atlas.riftEnabled", b)) { cfg.atlas.riftEnabled = b; any = true; }
	if (parse_int(kv, "training.atlas.riftLagHorizon", i) && i >= 0) { cfg.atlas.riftLagHorizon = static_cast<unsigned int>(i); any = true; }
	if (parse_float(kv, "training.atlas.riftMemoryScale", f)) { cfg.atlas.riftMemoryScale = f; any = true; }
	if (parse_float(kv, "training.atlas.riftEdgeThreshold", f)) { cfg.atlas.riftEdgeThreshold = f; any = true; }
	if (parse_float(kv, "training.atlas.riftPoleMax", f)) { cfg.atlas.riftPoleMax = f; any = true; }
	if (parse_bool01(kv, "training.atlas.orbitEnabled", b)) { cfg.atlas.orbitEnabled = b; any = true; }
	if (parse_float(kv, "training.atlas.orbitMemoryScale", f)) { cfg.atlas.orbitMemoryScale = f; any = true; }
	if (parse_float(kv, "training.atlas.orbitEdgeThreshold", f)) { cfg.atlas.orbitEdgeThreshold = f; any = true; }
	if (parse_float(kv, "training.atlas.orbitPoleMax", f)) { cfg.atlas.orbitPoleMax = f; any = true; }
	if (parse_bool01(kv, "training.atlas.helmEnabled", b)) { cfg.atlas.helmEnabled = b; any = true; }
	if (parse_float(kv, "training.atlas.helmMemoryScale", f)) { cfg.atlas.helmMemoryScale = f; any = true; }
	if (parse_float(kv, "training.atlas.helmEdgeThreshold", f)) { cfg.atlas.helmEdgeThreshold = f; any = true; }
	if (parse_int(kv, "training.atlas.helmModeRank", i) && i >= 0) { cfg.atlas.helmModeRank = static_cast<unsigned int>(i); any = true; }
	if (parse_int(kv, "training.atlas.helmHiddenStackDepth", i) && i >= 0) { cfg.atlas.helmHiddenStackDepth = static_cast<unsigned int>(i); any = true; }
	if (parse_float(kv, "training.atlas.helmPoleMax", f)) { cfg.atlas.helmPoleMax = f; any = true; }
	if (parse_bool01(kv, "training.atlas.asterEnabled", b)) { cfg.atlas.asterEnabled = b; any = true; }
	if (parse_bool01(kv, "training.atlas.aegisEnabled", b)) { cfg.atlas.aegisEnabled = b; any = true; }
	if (parse_bool01(kv, "training.atlas.citadelEnabled", b)) { cfg.atlas.citadelEnabled = b; any = true; }
	if (parse_bool01(kv, "training.atlas.rampartEnabled", b)) { cfg.atlas.rampartEnabled = b; any = true; }
	if (parse_bool01(kv, "training.atlas.meritEnabled", b)) { cfg.atlas.meritEnabled = b; any = true; }
	if (parse_bool01(kv, "training.atlas.strataEnabled", b)) { cfg.atlas.strataEnabled = b; any = true; }
	if (parse_float(kv, "training.atlas.asterMemoryScale", f)) { cfg.atlas.asterMemoryScale = f; any = true; }
	if (parse_float(kv, "training.atlas.asterEdgeThreshold", f)) { cfg.atlas.asterEdgeThreshold = f; any = true; }
	if (parse_int(kv, "training.atlas.asterStateRank", i) && i >= 0) { cfg.atlas.asterStateRank = static_cast<unsigned int>(i); any = true; }
	if (parse_int(kv, "training.atlas.asterHiddenStackDepth", i) && i >= 0) { cfg.atlas.asterHiddenStackDepth = static_cast<unsigned int>(i); any = true; }
	if (parse_float(kv, "training.atlas.asterPoleMax", f)) { cfg.atlas.asterPoleMax = f; any = true; }
	if (parse_bool01(kv, "training.atlas.kappaEnabled", b)) { cfg.atlas.kappaEnabled = b; any = true; }
	if (parse_int(kv, "training.atlas.kappaHeads", i) && i >= 0) { cfg.atlas.kappaHeads = static_cast<unsigned int>(i); any = true; }
	if (parse_int(kv, "training.atlas.kappaLagBuckets", i) && i >= 0) { cfg.atlas.kappaLagBuckets = static_cast<unsigned int>(i); any = true; }
	if (parse_int(kv, "training.atlas.kappaRank", i) && i >= 0) { cfg.atlas.kappaRank = static_cast<unsigned int>(i); any = true; }
	if (parse_float(kv, "training.atlas.auroraHorizonBlend", f)) { cfg.atlas.auroraHorizonBlend = f; any = true; }
	if (parse_float(kv, "training.atlas.auroraBudgetMax", f)) { cfg.atlas.auroraBudgetMax = f; any = true; }
	if (parse_bool01(kv, "training.atlas.auroraAdamwBackbone", b)) { cfg.atlas.auroraAdamwBackbone = b; any = true; }
	if (parse_float(kv, "training.atlas.auroraHeadGain", f)) { cfg.atlas.auroraHeadGain = f; any = true; }
	if (parse_float(kv, "training.atlas.auroraBodyTrustScale", f)) { cfg.atlas.auroraBodyTrustScale = f; any = true; }
	if (parse_bool01(kv, "training.atlas.geodeEnabled", b)) { cfg.atlas.geodeEnabled = b; any = true; }
	if (parse_float(kv, "training.atlas.geodeGeometryScale", f)) { cfg.atlas.geodeGeometryScale = f; any = true; }
	if (parse_float(kv, "training.atlas.geodePredictiveScale", f)) { cfg.atlas.geodePredictiveScale = f; any = true; }
	if (parse_bool01(kv, "training.atlas.bimapEnabled", b)) { cfg.atlas.bimapEnabled = b; any = true; }
	if (parse_bool01(kv, "training.atlas.bimapLowRankEnabled", b)) { cfg.atlas.bimapLowRankEnabled = b; any = true; }
	if (parse_float(kv, "training.atlas.bimapGeometryScale", f)) { cfg.atlas.bimapGeometryScale = f; any = true; }
	if (parse_float(kv, "training.atlas.bimapPredictiveScale", f)) { cfg.atlas.bimapPredictiveScale = f; any = true; }
	if (parse_int(kv, "training.atlas.bimapFactorCadence", i) && i >= 0) { cfg.atlas.bimapFactorCadence = static_cast<unsigned int>(i); any = true; }
	if (parse_bool01(kv, "training.atlas.pactEnabled", b)) { cfg.atlas.pactEnabled = b; any = true; }
	if (parse_bool01(kv, "training.atlas.pactLowRankEnabled", b)) { cfg.atlas.pactLowRankEnabled = b; any = true; }
	if (parse_float(kv, "training.atlas.pactGeometryScale", f)) { cfg.atlas.pactGeometryScale = f; any = true; }
	if (parse_float(kv, "training.atlas.pactPredictiveScale", f)) { cfg.atlas.pactPredictiveScale = f; any = true; }
	if (parse_int(kv, "training.atlas.pactFactorCadence", i) && i >= 0) { cfg.atlas.pactFactorCadence = static_cast<unsigned int>(i); any = true; }
	if (parse_float(kv, "training.atlas.pactCostScale", f)) { cfg.atlas.pactCostScale = f; any = true; }
	if (parse_float(kv, "training.atlas.pactPromoteThreshold", f)) { cfg.atlas.pactPromoteThreshold = f; any = true; }
	if (parse_float(kv, "training.atlas.pactDemoteThreshold", f)) { cfg.atlas.pactDemoteThreshold = f; any = true; }
	if (parse_bool01(kv, "training.atlas.racerEnabled", b)) { cfg.atlas.racerEnabled = b; any = true; }
	if (parse_float(kv, "training.atlas.racerGeometryScale", f)) { cfg.atlas.racerGeometryScale = f; any = true; }
	if (parse_float(kv, "training.atlas.racerPredictiveScale", f)) { cfg.atlas.racerPredictiveScale = f; any = true; }
	if (parse_int(kv, "training.atlas.racerFactorCadence", i) && i >= 0) { cfg.atlas.racerFactorCadence = static_cast<unsigned int>(i); any = true; }
	if (parse_float(kv, "training.atlas.racerRiskScale", f)) { cfg.atlas.racerRiskScale = f; any = true; }
	if (parse_float(kv, "training.atlas.racerCostScale", f)) { cfg.atlas.racerCostScale = f; any = true; }
	if (parse_float(kv, "training.atlas.racerPromoteThreshold", f)) { cfg.atlas.racerPromoteThreshold = f; any = true; }
	if (parse_float(kv, "training.atlas.racerDemoteThreshold", f)) { cfg.atlas.racerDemoteThreshold = f; any = true; }
	if (parse_bool01(kv, "training.atlas.kronEnabled", b)) { cfg.atlas.kronEnabled = b; any = true; }
	if (parse_float(kv, "training.atlas.kronGeometryScale", f)) { cfg.atlas.kronGeometryScale = f; any = true; }
	if (parse_float(kv, "training.atlas.kronPredictiveScale", f)) { cfg.atlas.kronPredictiveScale = f; any = true; }
	if (parse_int(kv, "training.atlas.kronFactorCadence", i) && i >= 0) { cfg.atlas.kronFactorCadence = static_cast<unsigned int>(i); any = true; }
	if (parse_float(kv, "training.atlas.kronDamping", f)) { cfg.atlas.kronDamping = f; any = true; }
	if (parse_bool01(kv, "training.atlas.muonEnabled", b)) { cfg.atlas.muonEnabled = b; any = true; }
	if (parse_float(kv, "training.atlas.muonGeometryScale", f)) { cfg.atlas.muonGeometryScale = f; any = true; }
	if (parse_float(kv, "training.atlas.muonPredictiveScale", f)) { cfg.atlas.muonPredictiveScale = f; any = true; }
	if (parse_float(kv, "training.atlas.muonMaxAspect", f)) { cfg.atlas.muonMaxAspect = f; any = true; }
	if (parse_int(kv, "training.atlas.muonMinDim", i) && i >= 0) { cfg.atlas.muonMinDim = static_cast<unsigned int>(i); any = true; }
	if (parse_float(kv, "training.atlas.muonDamping", f)) { cfg.atlas.muonDamping = f; any = true; }
	if (parse_float(kv, "training.atlas.seamMirrorStep", f)) { cfg.atlas.seamMirrorStep = f; any = true; }
	if (parse_float(kv, "training.atlas.seamBudgetMax", f)) { cfg.atlas.seamBudgetMax = f; any = true; }
	if (parse_float(kv, "training.atlas.quasarTemperature", f)) { cfg.atlas.quasarTemperature = f; any = true; }
	if (parse_float(kv, "training.atlas.quasarBudgetMax", f)) { cfg.atlas.quasarBudgetMax = f; any = true; }
	if (parse_float(kv, "training.atlas.aegisPredictiveScale", f)) { cfg.atlas.aegisPredictiveScale = f; any = true; }
	if (parse_float(kv, "training.atlas.aegisOutputScale", f)) { cfg.atlas.aegisOutputScale = f; any = true; }
	if (parse_float(kv, "training.atlas.citadelAnchorBase", f)) { cfg.atlas.citadelAnchorBase = f; any = true; }
	if (parse_float(kv, "training.atlas.citadelHardRegimeScale", f)) { cfg.atlas.citadelHardRegimeScale = f; any = true; }
	if (parse_float(kv, "training.atlas.citadelDisagreementScale", f)) { cfg.atlas.citadelDisagreementScale = f; any = true; }
	if (parse_float(kv, "training.atlas.citadelSpatialScale", f)) { cfg.atlas.citadelSpatialScale = f; any = true; }
	if (parse_float(kv, "training.atlas.rampartTauMin", f)) { cfg.atlas.rampartTauMin = f; any = true; }
	if (parse_float(kv, "training.atlas.rampartTauMax", f)) { cfg.atlas.rampartTauMax = f; any = true; }
	if (parse_float(kv, "training.atlas.rampartBudgetMax", f)) { cfg.atlas.rampartBudgetMax = f; any = true; }
	if (parse_float(kv, "training.atlas.rampartCovarianceMix", f)) { cfg.atlas.rampartCovarianceMix = f; any = true; }
	if (parse_float(kv, "training.atlas.meritGeometryScale", f)) { cfg.atlas.meritGeometryScale = f; any = true; }
	if (parse_float(kv, "training.atlas.meritTauMin", f)) { cfg.atlas.meritTauMin = f; any = true; }
	if (parse_float(kv, "training.atlas.meritTauMax", f)) { cfg.atlas.meritTauMax = f; any = true; }
	if (parse_float(kv, "training.atlas.meritBudgetMax", f)) { cfg.atlas.meritBudgetMax = f; any = true; }
	if (parse_float(kv, "training.atlas.meritCovarianceMix", f)) { cfg.atlas.meritCovarianceMix = f; any = true; }
	if (parse_float(kv, "training.atlas.strataNullBias", f)) { cfg.atlas.strataNullBias = f; any = true; }
	if (parse_float(kv, "training.atlas.strataDwellPenalty", f)) { cfg.atlas.strataDwellPenalty = f; any = true; }
	if (parse_float(kv, "training.atlas.strataBudgetMax", f)) { cfg.atlas.strataBudgetMax = f; any = true; }
	if (parse_float(kv, "training.atlas.strataPredictiveGeometryScale", f)) { cfg.atlas.strataPredictiveGeometryScale = f; any = true; }
	if (parse_float(kv, "training.atlas.strataCoupledGeometryScale", f)) { cfg.atlas.strataCoupledGeometryScale = f; any = true; }
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
	if (parse_bool01(kv, "training.transformer.captureOptimizerGapDiagnostics", b)) { cfg.transformer.captureOptimizerGapDiagnostics = b; any = true; }
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
	write_kv(out, "training.optimizer.adamGroupwiseEnabled", trainingConfig.optimizer.adamGroupwiseEnabled ? "1" : "0");
	{
		std::ostringstream oss; oss << trainingConfig.optimizer.adamGroupStabilityScale; write_kv(out, "training.optimizer.adamGroupStabilityScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.optimizer.adamGroupSnrScale; write_kv(out, "training.optimizer.adamGroupSnrScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.optimizer.adamGroupRatioScale; write_kv(out, "training.optimizer.adamGroupRatioScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.optimizer.adamGroupMinScale; write_kv(out, "training.optimizer.adamGroupMinScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.optimizer.adamGroupMaxScale; write_kv(out, "training.optimizer.adamGroupMaxScale", oss.str());
	}
	write_kv(out, "training.optimizer.adamGroupMinSize", u64_to_string(static_cast<uint64_t>(trainingConfig.optimizer.adamGroupMinSize)));
	write_kv(out, "training.atlas.rank", u64_to_string(static_cast<uint64_t>(trainingConfig.atlas.rank)));
	write_kv(out, "training.atlas.complementRank", u64_to_string(static_cast<uint64_t>(trainingConfig.atlas.complementRank)));
	{
		std::ostringstream oss; oss << trainingConfig.atlas.kappaMax; write_kv(out, "training.atlas.kappaMax", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.complementLrScale; write_kv(out, "training.atlas.complementLrScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.complementKappaMax; write_kv(out, "training.atlas.complementKappaMax", oss.str());
	}
	write_kv(out, "training.atlas.prismEnabled", trainingConfig.atlas.prismEnabled ? "1" : "0");
	write_kv(out, "training.atlas.prismLagHorizon", u64_to_string(static_cast<uint64_t>(trainingConfig.atlas.prismLagHorizon)));
	{
		std::ostringstream oss; oss << trainingConfig.atlas.prismMemoryScale; write_kv(out, "training.atlas.prismMemoryScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.prismPredictiveEdgeThreshold; write_kv(out, "training.atlas.prismPredictiveEdgeThreshold", oss.str());
	}
	write_kv(out, "training.atlas.resolveEnabled", trainingConfig.atlas.resolveEnabled ? "1" : "0");
	write_kv(out, "training.atlas.resolveLagHorizon", u64_to_string(static_cast<uint64_t>(trainingConfig.atlas.resolveLagHorizon)));
	{
		std::ostringstream oss; oss << trainingConfig.atlas.resolveMemoryScale; write_kv(out, "training.atlas.resolveMemoryScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.resolvePredictiveEdgeThreshold; write_kv(out, "training.atlas.resolvePredictiveEdgeThreshold", oss.str());
	}
	write_kv(out, "training.atlas.heroEnabled", trainingConfig.atlas.heroEnabled ? "1" : "0");
	write_kv(out, "training.atlas.heroLagHorizon", u64_to_string(static_cast<uint64_t>(trainingConfig.atlas.heroLagHorizon)));
	{
		std::ostringstream oss; oss << trainingConfig.atlas.heroMemoryScale; write_kv(out, "training.atlas.heroMemoryScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.heroEdgeThreshold; write_kv(out, "training.atlas.heroEdgeThreshold", oss.str());
	}
	write_kv(out, "training.atlas.cobaltEnabled", trainingConfig.atlas.cobaltEnabled ? "1" : "0");
	write_kv(out, "training.atlas.cobaltLagHorizon", u64_to_string(static_cast<uint64_t>(trainingConfig.atlas.cobaltLagHorizon)));
	{
		std::ostringstream oss; oss << trainingConfig.atlas.cobaltMemoryScale; write_kv(out, "training.atlas.cobaltMemoryScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.cobaltEdgeThreshold; write_kv(out, "training.atlas.cobaltEdgeThreshold", oss.str());
	}
	write_kv(out, "training.atlas.birchEnabled", trainingConfig.atlas.birchEnabled ? "1" : "0");
	write_kv(out, "training.atlas.birchPastHorizon", u64_to_string(static_cast<uint64_t>(trainingConfig.atlas.birchPastHorizon)));
	write_kv(out, "training.atlas.birchFutureHorizon", u64_to_string(static_cast<uint64_t>(trainingConfig.atlas.birchFutureHorizon)));
	{
		std::ostringstream oss; oss << trainingConfig.atlas.birchMemoryScale; write_kv(out, "training.atlas.birchMemoryScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.birchEdgeThreshold; write_kv(out, "training.atlas.birchEdgeThreshold", oss.str());
	}
	write_kv(out, "training.atlas.ghostEnabled", trainingConfig.atlas.ghostEnabled ? "1" : "0");
	write_kv(out, "training.atlas.ghostLagHorizon", u64_to_string(static_cast<uint64_t>(trainingConfig.atlas.ghostLagHorizon)));
	{
		std::ostringstream oss; oss << trainingConfig.atlas.ghostMemoryScale; write_kv(out, "training.atlas.ghostMemoryScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.ghostEdgeThreshold; write_kv(out, "training.atlas.ghostEdgeThreshold", oss.str());
	}
	write_kv(out, "training.atlas.sparrowEnabled", trainingConfig.atlas.sparrowEnabled ? "1" : "0");
	write_kv(out, "training.atlas.sparrowModeRank", u64_to_string(static_cast<uint64_t>(trainingConfig.atlas.sparrowModeRank)));
	write_kv(out, "training.atlas.sparrowAutoModeGate", trainingConfig.atlas.sparrowAutoModeGate ? "1" : "0");
	{
		std::ostringstream oss; oss << trainingConfig.atlas.sparrowMemoryScale; write_kv(out, "training.atlas.sparrowMemoryScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.sparrowEdgeThreshold; write_kv(out, "training.atlas.sparrowEdgeThreshold", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.sparrowSecondEdgeThreshold; write_kv(out, "training.atlas.sparrowSecondEdgeThreshold", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.sparrowSecondEdgeFraction; write_kv(out, "training.atlas.sparrowSecondEdgeFraction", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.sparrowPoleMax; write_kv(out, "training.atlas.sparrowPoleMax", oss.str());
	}
	write_kv(out, "training.atlas.qbrtEnabled", trainingConfig.atlas.qbrtEnabled ? "1" : "0");
	write_kv(out, "training.atlas.qbrtLagHorizon", u64_to_string(static_cast<uint64_t>(trainingConfig.atlas.qbrtLagHorizon)));
	{
		std::ostringstream oss; oss << trainingConfig.atlas.qbrtMemoryScale; write_kv(out, "training.atlas.qbrtMemoryScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.qbrtEdgeThreshold; write_kv(out, "training.atlas.qbrtEdgeThreshold", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.qbrtPoleMax; write_kv(out, "training.atlas.qbrtPoleMax", oss.str());
	}
	write_kv(out, "training.atlas.qrcEnabled", trainingConfig.atlas.qrcEnabled ? "1" : "0");
	write_kv(out, "training.atlas.qrcLagHorizon", u64_to_string(static_cast<uint64_t>(trainingConfig.atlas.qrcLagHorizon)));
	{
		std::ostringstream oss; oss << trainingConfig.atlas.qrcMemoryScale; write_kv(out, "training.atlas.qrcMemoryScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.qrcEdgeThreshold; write_kv(out, "training.atlas.qrcEdgeThreshold", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.qrcPoleMax; write_kv(out, "training.atlas.qrcPoleMax", oss.str());
	}
	write_kv(out, "training.atlas.riftEnabled", trainingConfig.atlas.riftEnabled ? "1" : "0");
	write_kv(out, "training.atlas.riftLagHorizon", u64_to_string(static_cast<uint64_t>(trainingConfig.atlas.riftLagHorizon)));
	{
		std::ostringstream oss; oss << trainingConfig.atlas.riftMemoryScale; write_kv(out, "training.atlas.riftMemoryScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.riftEdgeThreshold; write_kv(out, "training.atlas.riftEdgeThreshold", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.riftPoleMax; write_kv(out, "training.atlas.riftPoleMax", oss.str());
	}
	write_kv(out, "training.atlas.orbitEnabled", trainingConfig.atlas.orbitEnabled ? "1" : "0");
	{
		std::ostringstream oss; oss << trainingConfig.atlas.orbitMemoryScale; write_kv(out, "training.atlas.orbitMemoryScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.orbitEdgeThreshold; write_kv(out, "training.atlas.orbitEdgeThreshold", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.orbitPoleMax; write_kv(out, "training.atlas.orbitPoleMax", oss.str());
	}
	write_kv(out, "training.atlas.helmEnabled", trainingConfig.atlas.helmEnabled ? "1" : "0");
	{
		std::ostringstream oss; oss << trainingConfig.atlas.helmMemoryScale; write_kv(out, "training.atlas.helmMemoryScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.helmEdgeThreshold; write_kv(out, "training.atlas.helmEdgeThreshold", oss.str());
	}
	write_kv(out, "training.atlas.helmModeRank", u64_to_string(static_cast<uint64_t>(trainingConfig.atlas.helmModeRank)));
	write_kv(out, "training.atlas.helmHiddenStackDepth", u64_to_string(static_cast<uint64_t>(trainingConfig.atlas.helmHiddenStackDepth)));
	{
		std::ostringstream oss; oss << trainingConfig.atlas.helmPoleMax; write_kv(out, "training.atlas.helmPoleMax", oss.str());
	}
	write_kv(out, "training.atlas.asterEnabled", trainingConfig.atlas.asterEnabled ? "1" : "0");
	write_kv(out, "training.atlas.aegisEnabled", trainingConfig.atlas.aegisEnabled ? "1" : "0");
	write_kv(out, "training.atlas.citadelEnabled", trainingConfig.atlas.citadelEnabled ? "1" : "0");
	write_kv(out, "training.atlas.rampartEnabled", trainingConfig.atlas.rampartEnabled ? "1" : "0");
	write_kv(out, "training.atlas.meritEnabled", trainingConfig.atlas.meritEnabled ? "1" : "0");
	write_kv(out, "training.atlas.strataEnabled", trainingConfig.atlas.strataEnabled ? "1" : "0");
	{
		std::ostringstream oss; oss << trainingConfig.atlas.asterMemoryScale; write_kv(out, "training.atlas.asterMemoryScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.asterEdgeThreshold; write_kv(out, "training.atlas.asterEdgeThreshold", oss.str());
	}
	write_kv(out, "training.atlas.asterStateRank", u64_to_string(static_cast<uint64_t>(trainingConfig.atlas.asterStateRank)));
	write_kv(out, "training.atlas.asterHiddenStackDepth", u64_to_string(static_cast<uint64_t>(trainingConfig.atlas.asterHiddenStackDepth)));
	{
		std::ostringstream oss; oss << trainingConfig.atlas.asterPoleMax; write_kv(out, "training.atlas.asterPoleMax", oss.str());
	}
	write_kv(out, "training.atlas.kappaEnabled", trainingConfig.atlas.kappaEnabled ? "1" : "0");
	write_kv(out, "training.atlas.kappaHeads", u64_to_string(static_cast<uint64_t>(trainingConfig.atlas.kappaHeads)));
	write_kv(out, "training.atlas.kappaLagBuckets", u64_to_string(static_cast<uint64_t>(trainingConfig.atlas.kappaLagBuckets)));
	write_kv(out, "training.atlas.kappaRank", u64_to_string(static_cast<uint64_t>(trainingConfig.atlas.kappaRank)));
	{
		std::ostringstream oss; oss << trainingConfig.atlas.auroraHorizonBlend; write_kv(out, "training.atlas.auroraHorizonBlend", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.auroraBudgetMax; write_kv(out, "training.atlas.auroraBudgetMax", oss.str());
	}
	write_kv(out, "training.atlas.auroraAdamwBackbone", trainingConfig.atlas.auroraAdamwBackbone ? "1" : "0");
	{
		std::ostringstream oss; oss << trainingConfig.atlas.auroraHeadGain; write_kv(out, "training.atlas.auroraHeadGain", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.auroraBodyTrustScale; write_kv(out, "training.atlas.auroraBodyTrustScale", oss.str());
	}
	write_kv(out, "training.atlas.geodeEnabled", trainingConfig.atlas.geodeEnabled ? "1" : "0");
	{
		std::ostringstream oss; oss << trainingConfig.atlas.geodeGeometryScale; write_kv(out, "training.atlas.geodeGeometryScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.geodePredictiveScale; write_kv(out, "training.atlas.geodePredictiveScale", oss.str());
	}
	write_kv(out, "training.atlas.bimapEnabled", trainingConfig.atlas.bimapEnabled ? "1" : "0");
	write_kv(out, "training.atlas.bimapLowRankEnabled", trainingConfig.atlas.bimapLowRankEnabled ? "1" : "0");
	{
		std::ostringstream oss; oss << trainingConfig.atlas.bimapGeometryScale; write_kv(out, "training.atlas.bimapGeometryScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.bimapPredictiveScale; write_kv(out, "training.atlas.bimapPredictiveScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.bimapFactorCadence; write_kv(out, "training.atlas.bimapFactorCadence", oss.str());
	}
	write_kv(out, "training.atlas.pactEnabled", trainingConfig.atlas.pactEnabled ? "1" : "0");
	write_kv(out, "training.atlas.pactLowRankEnabled", trainingConfig.atlas.pactLowRankEnabled ? "1" : "0");
	{
		std::ostringstream oss; oss << trainingConfig.atlas.pactGeometryScale; write_kv(out, "training.atlas.pactGeometryScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.pactPredictiveScale; write_kv(out, "training.atlas.pactPredictiveScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.pactFactorCadence; write_kv(out, "training.atlas.pactFactorCadence", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.pactCostScale; write_kv(out, "training.atlas.pactCostScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.pactPromoteThreshold; write_kv(out, "training.atlas.pactPromoteThreshold", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.pactDemoteThreshold; write_kv(out, "training.atlas.pactDemoteThreshold", oss.str());
	}
	write_kv(out, "training.atlas.racerEnabled", trainingConfig.atlas.racerEnabled ? "1" : "0");
	{
		std::ostringstream oss; oss << trainingConfig.atlas.racerGeometryScale; write_kv(out, "training.atlas.racerGeometryScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.racerPredictiveScale; write_kv(out, "training.atlas.racerPredictiveScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.racerFactorCadence; write_kv(out, "training.atlas.racerFactorCadence", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.racerRiskScale; write_kv(out, "training.atlas.racerRiskScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.racerCostScale; write_kv(out, "training.atlas.racerCostScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.racerPromoteThreshold; write_kv(out, "training.atlas.racerPromoteThreshold", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.racerDemoteThreshold; write_kv(out, "training.atlas.racerDemoteThreshold", oss.str());
	}
	write_kv(out, "training.atlas.kronEnabled", trainingConfig.atlas.kronEnabled ? "1" : "0");
	{
		std::ostringstream oss; oss << trainingConfig.atlas.kronGeometryScale; write_kv(out, "training.atlas.kronGeometryScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.kronPredictiveScale; write_kv(out, "training.atlas.kronPredictiveScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.kronFactorCadence; write_kv(out, "training.atlas.kronFactorCadence", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.kronDamping; write_kv(out, "training.atlas.kronDamping", oss.str());
	}
	write_kv(out, "training.atlas.muonEnabled", trainingConfig.atlas.muonEnabled ? "1" : "0");
	{
		std::ostringstream oss; oss << trainingConfig.atlas.muonGeometryScale; write_kv(out, "training.atlas.muonGeometryScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.muonPredictiveScale; write_kv(out, "training.atlas.muonPredictiveScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.muonMaxAspect; write_kv(out, "training.atlas.muonMaxAspect", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.muonMinDim; write_kv(out, "training.atlas.muonMinDim", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.muonDamping; write_kv(out, "training.atlas.muonDamping", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.seamMirrorStep; write_kv(out, "training.atlas.seamMirrorStep", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.seamBudgetMax; write_kv(out, "training.atlas.seamBudgetMax", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.quasarTemperature; write_kv(out, "training.atlas.quasarTemperature", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.quasarBudgetMax; write_kv(out, "training.atlas.quasarBudgetMax", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.aegisPredictiveScale; write_kv(out, "training.atlas.aegisPredictiveScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.aegisOutputScale; write_kv(out, "training.atlas.aegisOutputScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.citadelAnchorBase; write_kv(out, "training.atlas.citadelAnchorBase", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.citadelHardRegimeScale; write_kv(out, "training.atlas.citadelHardRegimeScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.citadelDisagreementScale; write_kv(out, "training.atlas.citadelDisagreementScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.citadelSpatialScale; write_kv(out, "training.atlas.citadelSpatialScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.rampartTauMin; write_kv(out, "training.atlas.rampartTauMin", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.rampartTauMax; write_kv(out, "training.atlas.rampartTauMax", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.rampartBudgetMax; write_kv(out, "training.atlas.rampartBudgetMax", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.rampartCovarianceMix; write_kv(out, "training.atlas.rampartCovarianceMix", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.meritGeometryScale; write_kv(out, "training.atlas.meritGeometryScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.meritTauMin; write_kv(out, "training.atlas.meritTauMin", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.meritTauMax; write_kv(out, "training.atlas.meritTauMax", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.meritBudgetMax; write_kv(out, "training.atlas.meritBudgetMax", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.meritCovarianceMix; write_kv(out, "training.atlas.meritCovarianceMix", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.strataNullBias; write_kv(out, "training.atlas.strataNullBias", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.strataDwellPenalty; write_kv(out, "training.atlas.strataDwellPenalty", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.strataBudgetMax; write_kv(out, "training.atlas.strataBudgetMax", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.strataPredictiveGeometryScale; write_kv(out, "training.atlas.strataPredictiveGeometryScale", oss.str());
	}
	{
		std::ostringstream oss; oss << trainingConfig.atlas.strataCoupledGeometryScale; write_kv(out, "training.atlas.strataCoupledGeometryScale", oss.str());
	}
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
	write_kv(out, "training.transformer.captureOptimizerGapDiagnostics", trainingConfig.transformer.captureOptimizerGapDiagnostics ? "1" : "0");
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

	int savedComplementRank = -1;
	if (parse_int(kv, "training.atlas.complementRank", savedComplementRank) &&
	    savedComplementRank >= 0 &&
	    currentCfg.atlas.complementRank != static_cast<unsigned int>(savedComplementRank))
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.complementRank mismatch vs requested resume config (checkpoint "
		    << savedComplementRank << ", current " << currentCfg.atlas.complementRank << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedKappaMax = 0.0f;
	if (parse_float(kv, "training.atlas.kappaMax", savedKappaMax) &&
	    fabsf(currentCfg.atlas.kappaMax - savedKappaMax) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.kappaMax mismatch vs requested resume config (checkpoint "
		    << savedKappaMax << ", current " << currentCfg.atlas.kappaMax << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedComplementLrScale = 0.0f;
	if (parse_float(kv, "training.atlas.complementLrScale", savedComplementLrScale) &&
	    fabsf(currentCfg.atlas.complementLrScale - savedComplementLrScale) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.complementLrScale mismatch vs requested resume config (checkpoint "
		    << savedComplementLrScale << ", current " << currentCfg.atlas.complementLrScale << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedComplementKappaMax = 0.0f;
	if (parse_float(kv, "training.atlas.complementKappaMax", savedComplementKappaMax) &&
	    fabsf(currentCfg.atlas.complementKappaMax - savedComplementKappaMax) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.complementKappaMax mismatch vs requested resume config (checkpoint "
		    << savedComplementKappaMax << ", current " << currentCfg.atlas.complementKappaMax << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	bool savedPrismEnabled = false;
	if (parse_bool01(kv, "training.atlas.prismEnabled", savedPrismEnabled) &&
	    currentCfg.atlas.prismEnabled != savedPrismEnabled)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.prismEnabled mismatch vs requested resume config (checkpoint "
		    << (savedPrismEnabled ? 1 : 0) << ", current " << (currentCfg.atlas.prismEnabled ? 1 : 0) << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	int savedPrismLagHorizon = -1;
	if (parse_int(kv, "training.atlas.prismLagHorizon", savedPrismLagHorizon) &&
	    savedPrismLagHorizon >= 0 &&
	    currentCfg.atlas.prismLagHorizon != static_cast<unsigned int>(savedPrismLagHorizon))
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.prismLagHorizon mismatch vs requested resume config (checkpoint "
		    << savedPrismLagHorizon << ", current " << currentCfg.atlas.prismLagHorizon << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedPrismMemoryScale = 0.0f;
	if (parse_float(kv, "training.atlas.prismMemoryScale", savedPrismMemoryScale) &&
	    fabsf(currentCfg.atlas.prismMemoryScale - savedPrismMemoryScale) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.prismMemoryScale mismatch vs requested resume config (checkpoint "
		    << savedPrismMemoryScale << ", current " << currentCfg.atlas.prismMemoryScale << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedPrismEdgeThreshold = 0.0f;
	if (parse_float(kv, "training.atlas.prismPredictiveEdgeThreshold", savedPrismEdgeThreshold) &&
	    fabsf(currentCfg.atlas.prismPredictiveEdgeThreshold - savedPrismEdgeThreshold) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.prismPredictiveEdgeThreshold mismatch vs requested resume config (checkpoint "
		    << savedPrismEdgeThreshold << ", current " << currentCfg.atlas.prismPredictiveEdgeThreshold << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	bool savedResolveEnabled = false;
	if (parse_bool01(kv, "training.atlas.resolveEnabled", savedResolveEnabled) &&
	    currentCfg.atlas.resolveEnabled != savedResolveEnabled)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.resolveEnabled mismatch vs requested resume config (checkpoint "
		    << (savedResolveEnabled ? 1 : 0) << ", current " << (currentCfg.atlas.resolveEnabled ? 1 : 0) << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	int savedResolveLagHorizon = -1;
	if (parse_int(kv, "training.atlas.resolveLagHorizon", savedResolveLagHorizon) &&
	    savedResolveLagHorizon >= 0 &&
	    currentCfg.atlas.resolveLagHorizon != static_cast<unsigned int>(savedResolveLagHorizon))
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.resolveLagHorizon mismatch vs requested resume config (checkpoint "
		    << savedResolveLagHorizon << ", current " << currentCfg.atlas.resolveLagHorizon << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedResolveMemoryScale = 0.0f;
	if (parse_float(kv, "training.atlas.resolveMemoryScale", savedResolveMemoryScale) &&
	    fabsf(currentCfg.atlas.resolveMemoryScale - savedResolveMemoryScale) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.resolveMemoryScale mismatch vs requested resume config (checkpoint "
		    << savedResolveMemoryScale << ", current " << currentCfg.atlas.resolveMemoryScale << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedResolveEdgeThreshold = 0.0f;
	if (parse_float(kv, "training.atlas.resolvePredictiveEdgeThreshold", savedResolveEdgeThreshold) &&
	    fabsf(currentCfg.atlas.resolvePredictiveEdgeThreshold - savedResolveEdgeThreshold) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.resolvePredictiveEdgeThreshold mismatch vs requested resume config (checkpoint "
		    << savedResolveEdgeThreshold << ", current " << currentCfg.atlas.resolvePredictiveEdgeThreshold << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	bool savedHeroEnabled = false;
	if (parse_bool01(kv, "training.atlas.heroEnabled", savedHeroEnabled) &&
	    currentCfg.atlas.heroEnabled != savedHeroEnabled)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.heroEnabled mismatch vs requested resume config (checkpoint "
		    << (savedHeroEnabled ? 1 : 0) << ", current " << (currentCfg.atlas.heroEnabled ? 1 : 0) << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	int savedHeroLagHorizon = -1;
	if (parse_int(kv, "training.atlas.heroLagHorizon", savedHeroLagHorizon) &&
	    savedHeroLagHorizon >= 0 &&
	    currentCfg.atlas.heroLagHorizon != static_cast<unsigned int>(savedHeroLagHorizon))
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.heroLagHorizon mismatch vs requested resume config (checkpoint "
		    << savedHeroLagHorizon << ", current " << currentCfg.atlas.heroLagHorizon << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedHeroMemoryScale = 0.0f;
	if (parse_float(kv, "training.atlas.heroMemoryScale", savedHeroMemoryScale) &&
	    fabsf(currentCfg.atlas.heroMemoryScale - savedHeroMemoryScale) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.heroMemoryScale mismatch vs requested resume config (checkpoint "
		    << savedHeroMemoryScale << ", current " << currentCfg.atlas.heroMemoryScale << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedHeroEdgeThreshold = 0.0f;
	if (parse_float(kv, "training.atlas.heroEdgeThreshold", savedHeroEdgeThreshold) &&
	    fabsf(currentCfg.atlas.heroEdgeThreshold - savedHeroEdgeThreshold) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.heroEdgeThreshold mismatch vs requested resume config (checkpoint "
		    << savedHeroEdgeThreshold << ", current " << currentCfg.atlas.heroEdgeThreshold << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	bool savedCobaltEnabled = false;
	if (parse_bool01(kv, "training.atlas.cobaltEnabled", savedCobaltEnabled) &&
	    currentCfg.atlas.cobaltEnabled != savedCobaltEnabled)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.cobaltEnabled mismatch vs requested resume config (checkpoint "
		    << (savedCobaltEnabled ? 1 : 0) << ", current " << (currentCfg.atlas.cobaltEnabled ? 1 : 0) << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	int savedCobaltLagHorizon = -1;
	if (parse_int(kv, "training.atlas.cobaltLagHorizon", savedCobaltLagHorizon) &&
	    savedCobaltLagHorizon >= 0 &&
	    currentCfg.atlas.cobaltLagHorizon != static_cast<unsigned int>(savedCobaltLagHorizon))
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.cobaltLagHorizon mismatch vs requested resume config (checkpoint "
		    << savedCobaltLagHorizon << ", current " << currentCfg.atlas.cobaltLagHorizon << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedCobaltMemoryScale = 0.0f;
	if (parse_float(kv, "training.atlas.cobaltMemoryScale", savedCobaltMemoryScale) &&
	    fabsf(currentCfg.atlas.cobaltMemoryScale - savedCobaltMemoryScale) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.cobaltMemoryScale mismatch vs requested resume config (checkpoint "
		    << savedCobaltMemoryScale << ", current " << currentCfg.atlas.cobaltMemoryScale << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedCobaltEdgeThreshold = 0.0f;
	if (parse_float(kv, "training.atlas.cobaltEdgeThreshold", savedCobaltEdgeThreshold) &&
	    fabsf(currentCfg.atlas.cobaltEdgeThreshold - savedCobaltEdgeThreshold) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.cobaltEdgeThreshold mismatch vs requested resume config (checkpoint "
		    << savedCobaltEdgeThreshold << ", current " << currentCfg.atlas.cobaltEdgeThreshold << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	bool savedBirchEnabled = false;
	if (parse_bool01(kv, "training.atlas.birchEnabled", savedBirchEnabled) &&
	    currentCfg.atlas.birchEnabled != savedBirchEnabled)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.birchEnabled mismatch vs requested resume config (checkpoint "
		    << (savedBirchEnabled ? 1 : 0) << ", current " << (currentCfg.atlas.birchEnabled ? 1 : 0) << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	int savedBirchPastHorizon = -1;
	if (parse_int(kv, "training.atlas.birchPastHorizon", savedBirchPastHorizon) &&
	    savedBirchPastHorizon >= 0 &&
	    currentCfg.atlas.birchPastHorizon != static_cast<unsigned int>(savedBirchPastHorizon))
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.birchPastHorizon mismatch vs requested resume config (checkpoint "
		    << savedBirchPastHorizon << ", current " << currentCfg.atlas.birchPastHorizon << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	int savedBirchFutureHorizon = -1;
	if (parse_int(kv, "training.atlas.birchFutureHorizon", savedBirchFutureHorizon) &&
	    savedBirchFutureHorizon >= 0 &&
	    currentCfg.atlas.birchFutureHorizon != static_cast<unsigned int>(savedBirchFutureHorizon))
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.birchFutureHorizon mismatch vs requested resume config (checkpoint "
		    << savedBirchFutureHorizon << ", current " << currentCfg.atlas.birchFutureHorizon << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedBirchMemoryScale = 0.0f;
	if (parse_float(kv, "training.atlas.birchMemoryScale", savedBirchMemoryScale) &&
	    fabsf(currentCfg.atlas.birchMemoryScale - savedBirchMemoryScale) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.birchMemoryScale mismatch vs requested resume config (checkpoint "
		    << savedBirchMemoryScale << ", current " << currentCfg.atlas.birchMemoryScale << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedBirchEdgeThreshold = 0.0f;
	if (parse_float(kv, "training.atlas.birchEdgeThreshold", savedBirchEdgeThreshold) &&
	    fabsf(currentCfg.atlas.birchEdgeThreshold - savedBirchEdgeThreshold) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.birchEdgeThreshold mismatch vs requested resume config (checkpoint "
		    << savedBirchEdgeThreshold << ", current " << currentCfg.atlas.birchEdgeThreshold << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	bool savedGhostEnabled = false;
	if (parse_bool01(kv, "training.atlas.ghostEnabled", savedGhostEnabled) &&
	    currentCfg.atlas.ghostEnabled != savedGhostEnabled)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.ghostEnabled mismatch vs requested resume config (checkpoint "
		    << (savedGhostEnabled ? 1 : 0) << ", current " << (currentCfg.atlas.ghostEnabled ? 1 : 0) << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	int savedGhostLagHorizon = -1;
	if (parse_int(kv, "training.atlas.ghostLagHorizon", savedGhostLagHorizon) &&
	    savedGhostLagHorizon >= 0 &&
	    currentCfg.atlas.ghostLagHorizon != static_cast<unsigned int>(savedGhostLagHorizon))
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.ghostLagHorizon mismatch vs requested resume config (checkpoint "
		    << savedGhostLagHorizon << ", current " << currentCfg.atlas.ghostLagHorizon << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedGhostMemoryScale = 0.0f;
	if (parse_float(kv, "training.atlas.ghostMemoryScale", savedGhostMemoryScale) &&
	    fabsf(currentCfg.atlas.ghostMemoryScale - savedGhostMemoryScale) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.ghostMemoryScale mismatch vs requested resume config (checkpoint "
		    << savedGhostMemoryScale << ", current " << currentCfg.atlas.ghostMemoryScale << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedGhostEdgeThreshold = 0.0f;
	if (parse_float(kv, "training.atlas.ghostEdgeThreshold", savedGhostEdgeThreshold) &&
	    fabsf(currentCfg.atlas.ghostEdgeThreshold - savedGhostEdgeThreshold) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.ghostEdgeThreshold mismatch vs requested resume config (checkpoint "
		    << savedGhostEdgeThreshold << ", current " << currentCfg.atlas.ghostEdgeThreshold << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	bool savedSparrowEnabled = false;
	if (parse_bool01(kv, "training.atlas.sparrowEnabled", savedSparrowEnabled) &&
	    currentCfg.atlas.sparrowEnabled != savedSparrowEnabled)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.sparrowEnabled mismatch vs requested resume config (checkpoint "
		    << (savedSparrowEnabled ? 1 : 0) << ", current " << (currentCfg.atlas.sparrowEnabled ? 1 : 0) << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}
	int savedSparrowModeRank = 0;
	if (parse_int(kv, "training.atlas.sparrowModeRank", savedSparrowModeRank) &&
	    savedSparrowModeRank >= 0
	    && currentCfg.atlas.sparrowModeRank != static_cast<unsigned int>(savedSparrowModeRank))
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.sparrowModeRank mismatch vs requested resume config (checkpoint "
		    << savedSparrowModeRank << ", current " << currentCfg.atlas.sparrowModeRank << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}
	bool savedSparrowAutoModeGate = false;
	if (parse_bool01(kv, "training.atlas.sparrowAutoModeGate", savedSparrowAutoModeGate) &&
	    currentCfg.atlas.sparrowAutoModeGate != savedSparrowAutoModeGate)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.sparrowAutoModeGate mismatch vs requested resume config (checkpoint "
		    << (savedSparrowAutoModeGate ? 1 : 0) << ", current " << (currentCfg.atlas.sparrowAutoModeGate ? 1 : 0) << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedSparrowMemoryScale = 0.0f;
	if (parse_float(kv, "training.atlas.sparrowMemoryScale", savedSparrowMemoryScale) &&
	    fabsf(currentCfg.atlas.sparrowMemoryScale - savedSparrowMemoryScale) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.sparrowMemoryScale mismatch vs requested resume config (checkpoint "
		    << savedSparrowMemoryScale << ", current " << currentCfg.atlas.sparrowMemoryScale << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedSparrowEdgeThreshold = 0.0f;
	if (parse_float(kv, "training.atlas.sparrowEdgeThreshold", savedSparrowEdgeThreshold) &&
	    fabsf(currentCfg.atlas.sparrowEdgeThreshold - savedSparrowEdgeThreshold) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.sparrowEdgeThreshold mismatch vs requested resume config (checkpoint "
		    << savedSparrowEdgeThreshold << ", current " << currentCfg.atlas.sparrowEdgeThreshold << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}
	float savedSparrowSecondEdgeThreshold = 0.0f;
	if (parse_float(kv, "training.atlas.sparrowSecondEdgeThreshold", savedSparrowSecondEdgeThreshold) &&
	    fabsf(currentCfg.atlas.sparrowSecondEdgeThreshold - savedSparrowSecondEdgeThreshold) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.sparrowSecondEdgeThreshold mismatch vs requested resume config (checkpoint "
		    << savedSparrowSecondEdgeThreshold << ", current " << currentCfg.atlas.sparrowSecondEdgeThreshold << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}
	float savedSparrowSecondEdgeFraction = 0.0f;
	if (parse_float(kv, "training.atlas.sparrowSecondEdgeFraction", savedSparrowSecondEdgeFraction) &&
	    fabsf(currentCfg.atlas.sparrowSecondEdgeFraction - savedSparrowSecondEdgeFraction) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.sparrowSecondEdgeFraction mismatch vs requested resume config (checkpoint "
		    << savedSparrowSecondEdgeFraction << ", current " << currentCfg.atlas.sparrowSecondEdgeFraction << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedSparrowPoleMax = 0.0f;
	if (parse_float(kv, "training.atlas.sparrowPoleMax", savedSparrowPoleMax) &&
	    fabsf(currentCfg.atlas.sparrowPoleMax - savedSparrowPoleMax) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.sparrowPoleMax mismatch vs requested resume config (checkpoint "
		    << savedSparrowPoleMax << ", current " << currentCfg.atlas.sparrowPoleMax << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	bool savedQbrtEnabled = false;
	if (parse_bool01(kv, "training.atlas.qbrtEnabled", savedQbrtEnabled) &&
	    currentCfg.atlas.qbrtEnabled != savedQbrtEnabled)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.qbrtEnabled mismatch vs requested resume config (checkpoint "
		    << (savedQbrtEnabled ? 1 : 0) << ", current " << (currentCfg.atlas.qbrtEnabled ? 1 : 0) << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	int savedQbrtLagHorizon = 0;
	if (parse_int(kv, "training.atlas.qbrtLagHorizon", savedQbrtLagHorizon) &&
	    savedQbrtLagHorizon >= 0 &&
	    currentCfg.atlas.qbrtLagHorizon != static_cast<unsigned int>(savedQbrtLagHorizon))
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.qbrtLagHorizon mismatch vs requested resume config (checkpoint "
		    << savedQbrtLagHorizon << ", current " << currentCfg.atlas.qbrtLagHorizon << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedQbrtMemoryScale = 0.0f;
	if (parse_float(kv, "training.atlas.qbrtMemoryScale", savedQbrtMemoryScale) &&
	    fabsf(currentCfg.atlas.qbrtMemoryScale - savedQbrtMemoryScale) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.qbrtMemoryScale mismatch vs requested resume config (checkpoint "
		    << savedQbrtMemoryScale << ", current " << currentCfg.atlas.qbrtMemoryScale << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedQbrtEdgeThreshold = 0.0f;
	if (parse_float(kv, "training.atlas.qbrtEdgeThreshold", savedQbrtEdgeThreshold) &&
	    fabsf(currentCfg.atlas.qbrtEdgeThreshold - savedQbrtEdgeThreshold) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.qbrtEdgeThreshold mismatch vs requested resume config (checkpoint "
		    << savedQbrtEdgeThreshold << ", current " << currentCfg.atlas.qbrtEdgeThreshold << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedQbrtPoleMax = 0.0f;
	if (parse_float(kv, "training.atlas.qbrtPoleMax", savedQbrtPoleMax) &&
	    fabsf(currentCfg.atlas.qbrtPoleMax - savedQbrtPoleMax) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.qbrtPoleMax mismatch vs requested resume config (checkpoint "
		    << savedQbrtPoleMax << ", current " << currentCfg.atlas.qbrtPoleMax << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	bool savedQrcEnabled = false;
	if (parse_bool01(kv, "training.atlas.qrcEnabled", savedQrcEnabled) &&
	    currentCfg.atlas.qrcEnabled != savedQrcEnabled)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.qrcEnabled mismatch vs requested resume config (checkpoint "
		    << (savedQrcEnabled ? 1 : 0) << ", current " << (currentCfg.atlas.qrcEnabled ? 1 : 0) << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	int savedQrcLagHorizon = 0;
	if (parse_int(kv, "training.atlas.qrcLagHorizon", savedQrcLagHorizon) &&
	    savedQrcLagHorizon >= 0 &&
	    currentCfg.atlas.qrcLagHorizon != static_cast<unsigned int>(savedQrcLagHorizon))
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.qrcLagHorizon mismatch vs requested resume config (checkpoint "
		    << savedQrcLagHorizon << ", current " << currentCfg.atlas.qrcLagHorizon << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedQrcMemoryScale = 0.0f;
	if (parse_float(kv, "training.atlas.qrcMemoryScale", savedQrcMemoryScale) &&
	    fabsf(currentCfg.atlas.qrcMemoryScale - savedQrcMemoryScale) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.qrcMemoryScale mismatch vs requested resume config (checkpoint "
		    << savedQrcMemoryScale << ", current " << currentCfg.atlas.qrcMemoryScale << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedQrcEdgeThreshold = 0.0f;
	if (parse_float(kv, "training.atlas.qrcEdgeThreshold", savedQrcEdgeThreshold) &&
	    fabsf(currentCfg.atlas.qrcEdgeThreshold - savedQrcEdgeThreshold) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.qrcEdgeThreshold mismatch vs requested resume config (checkpoint "
		    << savedQrcEdgeThreshold << ", current " << currentCfg.atlas.qrcEdgeThreshold << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedQrcPoleMax = 0.0f;
	if (parse_float(kv, "training.atlas.qrcPoleMax", savedQrcPoleMax) &&
	    fabsf(currentCfg.atlas.qrcPoleMax - savedQrcPoleMax) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.qrcPoleMax mismatch vs requested resume config (checkpoint "
		    << savedQrcPoleMax << ", current " << currentCfg.atlas.qrcPoleMax << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	bool savedRiftEnabled = false;
	if (parse_bool01(kv, "training.atlas.riftEnabled", savedRiftEnabled) &&
	    currentCfg.atlas.riftEnabled != savedRiftEnabled)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.riftEnabled mismatch vs requested resume config (checkpoint "
		    << (savedRiftEnabled ? 1 : 0) << ", current " << (currentCfg.atlas.riftEnabled ? 1 : 0) << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	int savedRiftLagHorizon = 0;
	if (parse_int(kv, "training.atlas.riftLagHorizon", savedRiftLagHorizon) &&
	    savedRiftLagHorizon >= 0 &&
	    currentCfg.atlas.riftLagHorizon != static_cast<unsigned int>(savedRiftLagHorizon))
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.riftLagHorizon mismatch vs requested resume config (checkpoint "
		    << savedRiftLagHorizon << ", current " << currentCfg.atlas.riftLagHorizon << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedRiftMemoryScale = 0.0f;
	if (parse_float(kv, "training.atlas.riftMemoryScale", savedRiftMemoryScale) &&
	    fabsf(currentCfg.atlas.riftMemoryScale - savedRiftMemoryScale) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.riftMemoryScale mismatch vs requested resume config (checkpoint "
		    << savedRiftMemoryScale << ", current " << currentCfg.atlas.riftMemoryScale << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedRiftEdgeThreshold = 0.0f;
	if (parse_float(kv, "training.atlas.riftEdgeThreshold", savedRiftEdgeThreshold) &&
	    fabsf(currentCfg.atlas.riftEdgeThreshold - savedRiftEdgeThreshold) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.riftEdgeThreshold mismatch vs requested resume config (checkpoint "
		    << savedRiftEdgeThreshold << ", current " << currentCfg.atlas.riftEdgeThreshold << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedRiftPoleMax = 0.0f;
	if (parse_float(kv, "training.atlas.riftPoleMax", savedRiftPoleMax) &&
	    fabsf(currentCfg.atlas.riftPoleMax - savedRiftPoleMax) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.riftPoleMax mismatch vs requested resume config (checkpoint "
		    << savedRiftPoleMax << ", current " << currentCfg.atlas.riftPoleMax << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	bool savedOrbitEnabled = false;
	if (parse_bool01(kv, "training.atlas.orbitEnabled", savedOrbitEnabled) &&
	    currentCfg.atlas.orbitEnabled != savedOrbitEnabled)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.orbitEnabled mismatch vs requested resume config (checkpoint "
		    << (savedOrbitEnabled ? 1 : 0) << ", current " << (currentCfg.atlas.orbitEnabled ? 1 : 0) << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedOrbitMemoryScale = 0.0f;
	if (parse_float(kv, "training.atlas.orbitMemoryScale", savedOrbitMemoryScale) &&
	    fabsf(currentCfg.atlas.orbitMemoryScale - savedOrbitMemoryScale) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.orbitMemoryScale mismatch vs requested resume config (checkpoint "
		    << savedOrbitMemoryScale << ", current " << currentCfg.atlas.orbitMemoryScale << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedOrbitEdgeThreshold = 0.0f;
	if (parse_float(kv, "training.atlas.orbitEdgeThreshold", savedOrbitEdgeThreshold) &&
	    fabsf(currentCfg.atlas.orbitEdgeThreshold - savedOrbitEdgeThreshold) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.orbitEdgeThreshold mismatch vs requested resume config (checkpoint "
		    << savedOrbitEdgeThreshold << ", current " << currentCfg.atlas.orbitEdgeThreshold << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedOrbitPoleMax = 0.0f;
	if (parse_float(kv, "training.atlas.orbitPoleMax", savedOrbitPoleMax) &&
	    fabsf(currentCfg.atlas.orbitPoleMax - savedOrbitPoleMax) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.orbitPoleMax mismatch vs requested resume config (checkpoint "
		    << savedOrbitPoleMax << ", current " << currentCfg.atlas.orbitPoleMax << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	bool savedHelmEnabled = false;
	if (parse_bool01(kv, "training.atlas.helmEnabled", savedHelmEnabled) &&
	    currentCfg.atlas.helmEnabled != savedHelmEnabled)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.helmEnabled mismatch vs requested resume config (checkpoint "
		    << (savedHelmEnabled ? 1 : 0) << ", current " << (currentCfg.atlas.helmEnabled ? 1 : 0) << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedHelmMemoryScale = 0.0f;
	if (parse_float(kv, "training.atlas.helmMemoryScale", savedHelmMemoryScale) &&
	    fabsf(currentCfg.atlas.helmMemoryScale - savedHelmMemoryScale) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.helmMemoryScale mismatch vs requested resume config (checkpoint "
		    << savedHelmMemoryScale << ", current " << currentCfg.atlas.helmMemoryScale << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedHelmEdgeThreshold = 0.0f;
	if (parse_float(kv, "training.atlas.helmEdgeThreshold", savedHelmEdgeThreshold) &&
	    fabsf(currentCfg.atlas.helmEdgeThreshold - savedHelmEdgeThreshold) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.helmEdgeThreshold mismatch vs requested resume config (checkpoint "
		    << savedHelmEdgeThreshold << ", current " << currentCfg.atlas.helmEdgeThreshold << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	int savedHelmModeRank = -1;
	if (parse_int(kv, "training.atlas.helmModeRank", savedHelmModeRank) &&
	    savedHelmModeRank >= 0 &&
	    currentCfg.atlas.helmModeRank != static_cast<unsigned int>(savedHelmModeRank))
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.helmModeRank mismatch vs requested resume config (checkpoint "
		    << savedHelmModeRank << ", current " << currentCfg.atlas.helmModeRank << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	int savedHelmHiddenStackDepth = -1;
	if (parse_int(kv, "training.atlas.helmHiddenStackDepth", savedHelmHiddenStackDepth) &&
	    savedHelmHiddenStackDepth >= 0 &&
	    currentCfg.atlas.helmHiddenStackDepth != static_cast<unsigned int>(savedHelmHiddenStackDepth))
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.helmHiddenStackDepth mismatch vs requested resume config (checkpoint "
		    << savedHelmHiddenStackDepth << ", current " << currentCfg.atlas.helmHiddenStackDepth << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedHelmPoleMax = 0.0f;
	if (parse_float(kv, "training.atlas.helmPoleMax", savedHelmPoleMax) &&
	    fabsf(currentCfg.atlas.helmPoleMax - savedHelmPoleMax) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.helmPoleMax mismatch vs requested resume config (checkpoint "
		    << savedHelmPoleMax << ", current " << currentCfg.atlas.helmPoleMax << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	bool savedAsterEnabled = false;
	if (parse_bool01(kv, "training.atlas.asterEnabled", savedAsterEnabled) &&
	    currentCfg.atlas.asterEnabled != savedAsterEnabled)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.asterEnabled mismatch vs requested resume config (checkpoint "
		    << (savedAsterEnabled ? 1 : 0) << ", current " << (currentCfg.atlas.asterEnabled ? 1 : 0) << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	bool savedAegisEnabled = false;
	if (parse_bool01(kv, "training.atlas.aegisEnabled", savedAegisEnabled) &&
	    currentCfg.atlas.aegisEnabled != savedAegisEnabled)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.aegisEnabled mismatch vs requested resume config (checkpoint "
		    << (savedAegisEnabled ? 1 : 0) << ", current " << (currentCfg.atlas.aegisEnabled ? 1 : 0) << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	bool savedCitadelEnabled = false;
	if (parse_bool01(kv, "training.atlas.citadelEnabled", savedCitadelEnabled) &&
	    currentCfg.atlas.citadelEnabled != savedCitadelEnabled)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.citadelEnabled mismatch vs requested resume config (checkpoint "
		    << (savedCitadelEnabled ? 1 : 0) << ", current " << (currentCfg.atlas.citadelEnabled ? 1 : 0) << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	bool savedRampartEnabled = false;
	if (parse_bool01(kv, "training.atlas.rampartEnabled", savedRampartEnabled) &&
	    currentCfg.atlas.rampartEnabled != savedRampartEnabled)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.rampartEnabled mismatch vs requested resume config (checkpoint "
		    << (savedRampartEnabled ? 1 : 0) << ", current " << (currentCfg.atlas.rampartEnabled ? 1 : 0) << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	bool savedMeritEnabled = false;
	if (parse_bool01(kv, "training.atlas.meritEnabled", savedMeritEnabled) &&
	    currentCfg.atlas.meritEnabled != savedMeritEnabled)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.meritEnabled mismatch vs requested resume config (checkpoint "
		    << (savedMeritEnabled ? 1 : 0) << ", current " << (currentCfg.atlas.meritEnabled ? 1 : 0) << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	bool savedStrataEnabled = false;
	if (parse_bool01(kv, "training.atlas.strataEnabled", savedStrataEnabled) &&
	    currentCfg.atlas.strataEnabled != savedStrataEnabled)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.strataEnabled mismatch vs requested resume config (checkpoint "
		    << (savedStrataEnabled ? 1 : 0) << ", current " << (currentCfg.atlas.strataEnabled ? 1 : 0) << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedAsterMemoryScale = 0.0f;
	if (parse_float(kv, "training.atlas.asterMemoryScale", savedAsterMemoryScale) &&
	    fabsf(currentCfg.atlas.asterMemoryScale - savedAsterMemoryScale) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.asterMemoryScale mismatch vs requested resume config (checkpoint "
		    << savedAsterMemoryScale << ", current " << currentCfg.atlas.asterMemoryScale << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedAsterEdgeThreshold = 0.0f;
	if (parse_float(kv, "training.atlas.asterEdgeThreshold", savedAsterEdgeThreshold) &&
	    fabsf(currentCfg.atlas.asterEdgeThreshold - savedAsterEdgeThreshold) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.asterEdgeThreshold mismatch vs requested resume config (checkpoint "
		    << savedAsterEdgeThreshold << ", current " << currentCfg.atlas.asterEdgeThreshold << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedCitadelAnchorBase = 0.0f;
	if (parse_float(kv, "training.atlas.citadelAnchorBase", savedCitadelAnchorBase) &&
	    fabsf(currentCfg.atlas.citadelAnchorBase - savedCitadelAnchorBase) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.citadelAnchorBase mismatch vs requested resume config (checkpoint "
		    << savedCitadelAnchorBase << ", current " << currentCfg.atlas.citadelAnchorBase << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedCitadelHardRegimeScale = 0.0f;
	if (parse_float(kv, "training.atlas.citadelHardRegimeScale", savedCitadelHardRegimeScale) &&
	    fabsf(currentCfg.atlas.citadelHardRegimeScale - savedCitadelHardRegimeScale) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.citadelHardRegimeScale mismatch vs requested resume config (checkpoint "
		    << savedCitadelHardRegimeScale << ", current " << currentCfg.atlas.citadelHardRegimeScale << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedCitadelDisagreementScale = 0.0f;
	if (parse_float(kv, "training.atlas.citadelDisagreementScale", savedCitadelDisagreementScale) &&
	    fabsf(currentCfg.atlas.citadelDisagreementScale - savedCitadelDisagreementScale) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.citadelDisagreementScale mismatch vs requested resume config (checkpoint "
		    << savedCitadelDisagreementScale << ", current " << currentCfg.atlas.citadelDisagreementScale << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedCitadelSpatialScale = 0.0f;
	if (parse_float(kv, "training.atlas.citadelSpatialScale", savedCitadelSpatialScale) &&
	    fabsf(currentCfg.atlas.citadelSpatialScale - savedCitadelSpatialScale) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.citadelSpatialScale mismatch vs requested resume config (checkpoint "
		    << savedCitadelSpatialScale << ", current " << currentCfg.atlas.citadelSpatialScale << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	int savedAsterStateRank = -1;
	if (parse_int(kv, "training.atlas.asterStateRank", savedAsterStateRank) &&
	    savedAsterStateRank >= 0 &&
	    currentCfg.atlas.asterStateRank != static_cast<unsigned int>(savedAsterStateRank))
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.asterStateRank mismatch vs requested resume config (checkpoint "
		    << savedAsterStateRank << ", current " << currentCfg.atlas.asterStateRank << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	int savedAsterHiddenStackDepth = -1;
	if (parse_int(kv, "training.atlas.asterHiddenStackDepth", savedAsterHiddenStackDepth) &&
	    savedAsterHiddenStackDepth >= 0 &&
	    currentCfg.atlas.asterHiddenStackDepth != static_cast<unsigned int>(savedAsterHiddenStackDepth))
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.asterHiddenStackDepth mismatch vs requested resume config (checkpoint "
		    << savedAsterHiddenStackDepth << ", current " << currentCfg.atlas.asterHiddenStackDepth << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedAsterPoleMax = 0.0f;
	if (parse_float(kv, "training.atlas.asterPoleMax", savedAsterPoleMax) &&
	    fabsf(currentCfg.atlas.asterPoleMax - savedAsterPoleMax) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.asterPoleMax mismatch vs requested resume config (checkpoint "
		    << savedAsterPoleMax << ", current " << currentCfg.atlas.asterPoleMax << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	bool savedKappaEnabled = false;
	if (parse_bool01(kv, "training.atlas.kappaEnabled", savedKappaEnabled) &&
	    currentCfg.atlas.kappaEnabled != savedKappaEnabled)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.kappaEnabled mismatch vs requested resume config (checkpoint "
		    << (savedKappaEnabled ? 1 : 0) << ", current " << (currentCfg.atlas.kappaEnabled ? 1 : 0) << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	int savedKappaHeads = -1;
	if (parse_int(kv, "training.atlas.kappaHeads", savedKappaHeads) &&
	    savedKappaHeads >= 0 &&
	    currentCfg.atlas.kappaHeads != static_cast<unsigned int>(savedKappaHeads))
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.kappaHeads mismatch vs requested resume config (checkpoint "
		    << savedKappaHeads << ", current " << currentCfg.atlas.kappaHeads << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	int savedKappaLagBuckets = -1;
	if (parse_int(kv, "training.atlas.kappaLagBuckets", savedKappaLagBuckets) &&
	    savedKappaLagBuckets >= 0 &&
	    currentCfg.atlas.kappaLagBuckets != static_cast<unsigned int>(savedKappaLagBuckets))
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.kappaLagBuckets mismatch vs requested resume config (checkpoint "
		    << savedKappaLagBuckets << ", current " << currentCfg.atlas.kappaLagBuckets << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	int savedKappaRank = -1;
	if (parse_int(kv, "training.atlas.kappaRank", savedKappaRank) &&
	    savedKappaRank >= 0 &&
	    currentCfg.atlas.kappaRank != static_cast<unsigned int>(savedKappaRank))
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.kappaRank mismatch vs requested resume config (checkpoint "
		    << savedKappaRank << ", current " << currentCfg.atlas.kappaRank << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedAuroraHorizonBlend = 0.0f;
	if (parse_float(kv, "training.atlas.auroraHorizonBlend", savedAuroraHorizonBlend) &&
	    fabsf(currentCfg.atlas.auroraHorizonBlend - savedAuroraHorizonBlend) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.auroraHorizonBlend mismatch vs requested resume config (checkpoint "
		    << savedAuroraHorizonBlend << ", current " << currentCfg.atlas.auroraHorizonBlend << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedAuroraBudgetMax = 0.0f;
	if (parse_float(kv, "training.atlas.auroraBudgetMax", savedAuroraBudgetMax) &&
	    fabsf(currentCfg.atlas.auroraBudgetMax - savedAuroraBudgetMax) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.auroraBudgetMax mismatch vs requested resume config (checkpoint "
		    << savedAuroraBudgetMax << ", current " << currentCfg.atlas.auroraBudgetMax << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	bool savedAuroraAdamwBackbone = false;
	if (parse_bool01(kv, "training.atlas.auroraAdamwBackbone", savedAuroraAdamwBackbone) &&
	    currentCfg.atlas.auroraAdamwBackbone != savedAuroraAdamwBackbone)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.auroraAdamwBackbone mismatch vs requested resume config (checkpoint "
		    << (savedAuroraAdamwBackbone ? 1 : 0) << ", current " << (currentCfg.atlas.auroraAdamwBackbone ? 1 : 0) << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedAuroraHeadGain = 0.0f;
	if (parse_float(kv, "training.atlas.auroraHeadGain", savedAuroraHeadGain) &&
	    fabsf(currentCfg.atlas.auroraHeadGain - savedAuroraHeadGain) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.auroraHeadGain mismatch vs requested resume config (checkpoint "
		    << savedAuroraHeadGain << ", current " << currentCfg.atlas.auroraHeadGain << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedAuroraBodyTrustScale = 0.0f;
	if (parse_float(kv, "training.atlas.auroraBodyTrustScale", savedAuroraBodyTrustScale) &&
	    fabsf(currentCfg.atlas.auroraBodyTrustScale - savedAuroraBodyTrustScale) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.auroraBodyTrustScale mismatch vs requested resume config (checkpoint "
		    << savedAuroraBodyTrustScale << ", current " << currentCfg.atlas.auroraBodyTrustScale << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	bool savedGeodeEnabled = false;
	if (parse_bool01(kv, "training.atlas.geodeEnabled", savedGeodeEnabled) &&
	    currentCfg.atlas.geodeEnabled != savedGeodeEnabled)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.geodeEnabled mismatch vs requested resume config (checkpoint "
		    << (savedGeodeEnabled ? 1 : 0) << ", current " << (currentCfg.atlas.geodeEnabled ? 1 : 0) << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedGeodeGeometryScale = 0.0f;
	if (parse_float(kv, "training.atlas.geodeGeometryScale", savedGeodeGeometryScale) &&
	    fabsf(currentCfg.atlas.geodeGeometryScale - savedGeodeGeometryScale) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.geodeGeometryScale mismatch vs requested resume config (checkpoint "
		    << savedGeodeGeometryScale << ", current " << currentCfg.atlas.geodeGeometryScale << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedGeodePredictiveScale = 0.0f;
	if (parse_float(kv, "training.atlas.geodePredictiveScale", savedGeodePredictiveScale) &&
	    fabsf(currentCfg.atlas.geodePredictiveScale - savedGeodePredictiveScale) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.geodePredictiveScale mismatch vs requested resume config (checkpoint "
		    << savedGeodePredictiveScale << ", current " << currentCfg.atlas.geodePredictiveScale << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	bool savedBiMAPEnabled = false;
	if (parse_bool01(kv, "training.atlas.bimapEnabled", savedBiMAPEnabled) &&
	    currentCfg.atlas.bimapEnabled != savedBiMAPEnabled)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.bimapEnabled mismatch vs requested resume config (checkpoint "
		    << (savedBiMAPEnabled ? 1 : 0) << ", current " << (currentCfg.atlas.bimapEnabled ? 1 : 0) << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	bool savedBiMAPLowRankEnabled = false;
	if (parse_bool01(kv, "training.atlas.bimapLowRankEnabled", savedBiMAPLowRankEnabled) &&
	    currentCfg.atlas.bimapLowRankEnabled != savedBiMAPLowRankEnabled)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.bimapLowRankEnabled mismatch vs requested resume config (checkpoint "
		    << (savedBiMAPLowRankEnabled ? 1 : 0) << ", current " << (currentCfg.atlas.bimapLowRankEnabled ? 1 : 0) << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedBiMAPGeometryScale = 0.0f;
	if (parse_float(kv, "training.atlas.bimapGeometryScale", savedBiMAPGeometryScale) &&
	    fabsf(currentCfg.atlas.bimapGeometryScale - savedBiMAPGeometryScale) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.bimapGeometryScale mismatch vs requested resume config (checkpoint "
		    << savedBiMAPGeometryScale << ", current " << currentCfg.atlas.bimapGeometryScale << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedBiMAPPredictiveScale = 0.0f;
	if (parse_float(kv, "training.atlas.bimapPredictiveScale", savedBiMAPPredictiveScale) &&
	    fabsf(currentCfg.atlas.bimapPredictiveScale - savedBiMAPPredictiveScale) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.bimapPredictiveScale mismatch vs requested resume config (checkpoint "
		    << savedBiMAPPredictiveScale << ", current " << currentCfg.atlas.bimapPredictiveScale << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	int savedBiMAPFactorCadence = 0;
	if (parse_int(kv, "training.atlas.bimapFactorCadence", savedBiMAPFactorCadence) &&
	    currentCfg.atlas.bimapFactorCadence != static_cast<unsigned int>(savedBiMAPFactorCadence))
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.bimapFactorCadence mismatch vs requested resume config (checkpoint "
		    << savedBiMAPFactorCadence << ", current " << currentCfg.atlas.bimapFactorCadence << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	bool savedPACTEnabled = false;
	if (parse_bool01(kv, "training.atlas.pactEnabled", savedPACTEnabled) &&
	    currentCfg.atlas.pactEnabled != savedPACTEnabled)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.pactEnabled mismatch vs requested resume config (checkpoint "
		    << (savedPACTEnabled ? 1 : 0) << ", current " << (currentCfg.atlas.pactEnabled ? 1 : 0) << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	bool savedPACTLowRankEnabled = false;
	if (parse_bool01(kv, "training.atlas.pactLowRankEnabled", savedPACTLowRankEnabled) &&
	    currentCfg.atlas.pactLowRankEnabled != savedPACTLowRankEnabled)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.pactLowRankEnabled mismatch vs requested resume config (checkpoint "
		    << (savedPACTLowRankEnabled ? 1 : 0) << ", current " << (currentCfg.atlas.pactLowRankEnabled ? 1 : 0) << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedPACTGeometryScale = 0.0f;
	if (parse_float(kv, "training.atlas.pactGeometryScale", savedPACTGeometryScale) &&
	    fabsf(currentCfg.atlas.pactGeometryScale - savedPACTGeometryScale) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.pactGeometryScale mismatch vs requested resume config (checkpoint "
		    << savedPACTGeometryScale << ", current " << currentCfg.atlas.pactGeometryScale << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedPACTPredictiveScale = 0.0f;
	if (parse_float(kv, "training.atlas.pactPredictiveScale", savedPACTPredictiveScale) &&
	    fabsf(currentCfg.atlas.pactPredictiveScale - savedPACTPredictiveScale) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.pactPredictiveScale mismatch vs requested resume config (checkpoint "
		    << savedPACTPredictiveScale << ", current " << currentCfg.atlas.pactPredictiveScale << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	int savedPACTFactorCadence = 0;
	if (parse_int(kv, "training.atlas.pactFactorCadence", savedPACTFactorCadence) &&
	    currentCfg.atlas.pactFactorCadence != static_cast<unsigned int>(savedPACTFactorCadence))
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.pactFactorCadence mismatch vs requested resume config (checkpoint "
		    << savedPACTFactorCadence << ", current " << currentCfg.atlas.pactFactorCadence << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedPACTCostScale = 0.0f;
	if (parse_float(kv, "training.atlas.pactCostScale", savedPACTCostScale) &&
	    fabsf(currentCfg.atlas.pactCostScale - savedPACTCostScale) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.pactCostScale mismatch vs requested resume config (checkpoint "
		    << savedPACTCostScale << ", current " << currentCfg.atlas.pactCostScale << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedPACTPromoteThreshold = 0.0f;
	if (parse_float(kv, "training.atlas.pactPromoteThreshold", savedPACTPromoteThreshold) &&
	    fabsf(currentCfg.atlas.pactPromoteThreshold - savedPACTPromoteThreshold) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.pactPromoteThreshold mismatch vs requested resume config (checkpoint "
		    << savedPACTPromoteThreshold << ", current " << currentCfg.atlas.pactPromoteThreshold << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedPACTDemoteThreshold = 0.0f;
	if (parse_float(kv, "training.atlas.pactDemoteThreshold", savedPACTDemoteThreshold) &&
	    fabsf(currentCfg.atlas.pactDemoteThreshold - savedPACTDemoteThreshold) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.pactDemoteThreshold mismatch vs requested resume config (checkpoint "
		    << savedPACTDemoteThreshold << ", current " << currentCfg.atlas.pactDemoteThreshold << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	bool savedRACEREnabled = false;
	if (parse_bool01(kv, "training.atlas.racerEnabled", savedRACEREnabled) &&
	    currentCfg.atlas.racerEnabled != savedRACEREnabled)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.racerEnabled mismatch vs requested resume config (checkpoint "
		    << (savedRACEREnabled ? 1 : 0) << ", current " << (currentCfg.atlas.racerEnabled ? 1 : 0) << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedRACERGeometryScale = 0.0f;
	if (parse_float(kv, "training.atlas.racerGeometryScale", savedRACERGeometryScale) &&
	    fabsf(currentCfg.atlas.racerGeometryScale - savedRACERGeometryScale) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.racerGeometryScale mismatch vs requested resume config (checkpoint "
		    << savedRACERGeometryScale << ", current " << currentCfg.atlas.racerGeometryScale << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedRACERPredictiveScale = 0.0f;
	if (parse_float(kv, "training.atlas.racerPredictiveScale", savedRACERPredictiveScale) &&
	    fabsf(currentCfg.atlas.racerPredictiveScale - savedRACERPredictiveScale) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.racerPredictiveScale mismatch vs requested resume config (checkpoint "
		    << savedRACERPredictiveScale << ", current " << currentCfg.atlas.racerPredictiveScale << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	int savedRACERFactorCadence = 0;
	if (parse_int(kv, "training.atlas.racerFactorCadence", savedRACERFactorCadence) &&
	    currentCfg.atlas.racerFactorCadence != static_cast<unsigned int>(savedRACERFactorCadence))
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.racerFactorCadence mismatch vs requested resume config (checkpoint "
		    << savedRACERFactorCadence << ", current " << currentCfg.atlas.racerFactorCadence << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedRACERRiskScale = 0.0f;
	if (parse_float(kv, "training.atlas.racerRiskScale", savedRACERRiskScale) &&
	    fabsf(currentCfg.atlas.racerRiskScale - savedRACERRiskScale) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.racerRiskScale mismatch vs requested resume config (checkpoint "
		    << savedRACERRiskScale << ", current " << currentCfg.atlas.racerRiskScale << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedRACERCostScale = 0.0f;
	if (parse_float(kv, "training.atlas.racerCostScale", savedRACERCostScale) &&
	    fabsf(currentCfg.atlas.racerCostScale - savedRACERCostScale) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.racerCostScale mismatch vs requested resume config (checkpoint "
		    << savedRACERCostScale << ", current " << currentCfg.atlas.racerCostScale << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedRACERPromoteThreshold = 0.0f;
	if (parse_float(kv, "training.atlas.racerPromoteThreshold", savedRACERPromoteThreshold) &&
	    fabsf(currentCfg.atlas.racerPromoteThreshold - savedRACERPromoteThreshold) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.racerPromoteThreshold mismatch vs requested resume config (checkpoint "
		    << savedRACERPromoteThreshold << ", current " << currentCfg.atlas.racerPromoteThreshold << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedRACERDemoteThreshold = 0.0f;
	if (parse_float(kv, "training.atlas.racerDemoteThreshold", savedRACERDemoteThreshold) &&
	    fabsf(currentCfg.atlas.racerDemoteThreshold - savedRACERDemoteThreshold) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.racerDemoteThreshold mismatch vs requested resume config (checkpoint "
		    << savedRACERDemoteThreshold << ", current " << currentCfg.atlas.racerDemoteThreshold << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	bool savedKRONEnabled = false;
	if (parse_bool01(kv, "training.atlas.kronEnabled", savedKRONEnabled) &&
	    currentCfg.atlas.kronEnabled != savedKRONEnabled)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.kronEnabled mismatch vs requested resume config (checkpoint "
		    << (savedKRONEnabled ? 1 : 0) << ", current " << (currentCfg.atlas.kronEnabled ? 1 : 0) << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedKRONGeometryScale = 0.0f;
	if (parse_float(kv, "training.atlas.kronGeometryScale", savedKRONGeometryScale) &&
	    fabsf(currentCfg.atlas.kronGeometryScale - savedKRONGeometryScale) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.kronGeometryScale mismatch vs requested resume config (checkpoint "
		    << savedKRONGeometryScale << ", current " << currentCfg.atlas.kronGeometryScale << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedKRONPredictiveScale = 0.0f;
	if (parse_float(kv, "training.atlas.kronPredictiveScale", savedKRONPredictiveScale) &&
	    fabsf(currentCfg.atlas.kronPredictiveScale - savedKRONPredictiveScale) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.kronPredictiveScale mismatch vs requested resume config (checkpoint "
		    << savedKRONPredictiveScale << ", current " << currentCfg.atlas.kronPredictiveScale << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	int savedKRONFactorCadence = 0;
	if (parse_int(kv, "training.atlas.kronFactorCadence", savedKRONFactorCadence) &&
	    currentCfg.atlas.kronFactorCadence != static_cast<unsigned int>(savedKRONFactorCadence))
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.kronFactorCadence mismatch vs requested resume config (checkpoint "
		    << savedKRONFactorCadence << ", current " << currentCfg.atlas.kronFactorCadence << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedKRONDamping = 0.0f;
	if (parse_float(kv, "training.atlas.kronDamping", savedKRONDamping) &&
	    fabsf(currentCfg.atlas.kronDamping - savedKRONDamping) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.kronDamping mismatch vs requested resume config (checkpoint "
		    << savedKRONDamping << ", current " << currentCfg.atlas.kronDamping << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	bool savedMUONEnabled = false;
	if (parse_bool01(kv, "training.atlas.muonEnabled", savedMUONEnabled) &&
	    currentCfg.atlas.muonEnabled != savedMUONEnabled)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.muonEnabled mismatch vs requested resume config (checkpoint "
		    << (savedMUONEnabled ? 1 : 0) << ", current " << (currentCfg.atlas.muonEnabled ? 1 : 0) << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedMUONGeometryScale = 0.0f;
	if (parse_float(kv, "training.atlas.muonGeometryScale", savedMUONGeometryScale) &&
	    fabsf(currentCfg.atlas.muonGeometryScale - savedMUONGeometryScale) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.muonGeometryScale mismatch vs requested resume config (checkpoint "
		    << savedMUONGeometryScale << ", current " << currentCfg.atlas.muonGeometryScale << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedMUONPredictiveScale = 0.0f;
	if (parse_float(kv, "training.atlas.muonPredictiveScale", savedMUONPredictiveScale) &&
	    fabsf(currentCfg.atlas.muonPredictiveScale - savedMUONPredictiveScale) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.muonPredictiveScale mismatch vs requested resume config (checkpoint "
		    << savedMUONPredictiveScale << ", current " << currentCfg.atlas.muonPredictiveScale << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedMUONMaxAspect = 0.0f;
	if (parse_float(kv, "training.atlas.muonMaxAspect", savedMUONMaxAspect) &&
	    fabsf(currentCfg.atlas.muonMaxAspect - savedMUONMaxAspect) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.muonMaxAspect mismatch vs requested resume config (checkpoint "
		    << savedMUONMaxAspect << ", current " << currentCfg.atlas.muonMaxAspect << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	int savedMUONMinDim = 0;
	if (parse_int(kv, "training.atlas.muonMinDim", savedMUONMinDim) &&
	    currentCfg.atlas.muonMinDim != static_cast<unsigned int>(savedMUONMinDim))
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.muonMinDim mismatch vs requested resume config (checkpoint "
		    << savedMUONMinDim << ", current " << currentCfg.atlas.muonMinDim << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedMUONDamping = 0.0f;
	if (parse_float(kv, "training.atlas.muonDamping", savedMUONDamping) &&
	    fabsf(currentCfg.atlas.muonDamping - savedMUONDamping) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.muonDamping mismatch vs requested resume config (checkpoint "
		    << savedMUONDamping << ", current " << currentCfg.atlas.muonDamping << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedAegisPredictiveScale = 0.0f;
	if (parse_float(kv, "training.atlas.aegisPredictiveScale", savedAegisPredictiveScale) &&
	    fabsf(currentCfg.atlas.aegisPredictiveScale - savedAegisPredictiveScale) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.aegisPredictiveScale mismatch vs requested resume config (checkpoint "
		    << savedAegisPredictiveScale << ", current " << currentCfg.atlas.aegisPredictiveScale << ")";
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_STATE, oss.str());
	}

	float savedAegisOutputScale = 0.0f;
	if (parse_float(kv, "training.atlas.aegisOutputScale", savedAegisOutputScale) &&
	    fabsf(currentCfg.atlas.aegisOutputScale - savedAegisOutputScale) > 1e-6f)
	{
		std::ostringstream oss;
		oss << "loadCheckpoint: training.atlas.aegisOutputScale mismatch vs requested resume config (checkpoint "
		    << savedAegisOutputScale << ", current " << currentCfg.atlas.aegisOutputScale << ")";
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
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(st.r));
		sh.push_back(static_cast<uint64_t>(st.n));
		out.push_back(TensorWriteRef(prefix + ".atlas.prevPrevGz", &st.prevPrevGz, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(4ull);
		sh.push_back(static_cast<uint64_t>(st.r));
		sh.push_back(static_cast<uint64_t>(st.n));
		out.push_back(TensorWriteRef(prefix + ".atlas.resolveGzHistory", &st.resolveGzHistory, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(4ull);
		sh.push_back(static_cast<uint64_t>(st.complementRank ? st.complementRank : 1u));
		sh.push_back(static_cast<uint64_t>(st.n));
		out.push_back(TensorWriteRef(prefix + ".atlas.heroGwHistory", &st.heroGwHistory, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(st.m));
		sh.push_back(static_cast<uint64_t>(st.complementRank ? st.complementRank : 1u));
		out.push_back(TensorWriteRef(prefix + ".atlas.V", &st.V, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(st.m));
		sh.push_back(static_cast<uint64_t>(st.complementRank ? st.complementRank : 1u));
		out.push_back(TensorWriteRef(prefix + ".atlas.scoutBasis", &st.scoutBasis, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(st.complementRank ? st.complementRank : 1u));
		sh.push_back(static_cast<uint64_t>(st.n));
		out.push_back(TensorWriteRef(prefix + ".atlas.prevGv", &st.prevGv, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(st.complementRank ? st.complementRank : 1u));
		sh.push_back(static_cast<uint64_t>(st.complementRank ? st.complementRank : 1u));
		out.push_back(TensorWriteRef(prefix + ".atlas.scoutCov", &st.scoutCov, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(st.complementRank ? st.complementRank : 1u));
		sh.push_back(static_cast<uint64_t>(st.complementRank ? st.complementRank : 1u));
		out.push_back(TensorWriteRef(prefix + ".atlas.scoutNoise", &st.scoutNoise, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(st.r));
		sh.push_back(static_cast<uint64_t>(st.n));
		out.push_back(TensorWriteRef(prefix + ".atlas.sparrowPrevActive", &st.sparrowPrevActive, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(st.complementRank ? st.complementRank : 1u));
		sh.push_back(static_cast<uint64_t>(st.n));
		out.push_back(TensorWriteRef(prefix + ".atlas.sparrowPrevScout", &st.sparrowPrevScout, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(st.r));
		sh.push_back(static_cast<uint64_t>(st.r));
		out.push_back(TensorWriteRef(prefix + ".atlas.sparrowFutureCov", &st.sparrowFutureCov, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(st.r + (st.complementRank ? st.complementRank : 1u)));
		sh.push_back(static_cast<uint64_t>(st.r + (st.complementRank ? st.complementRank : 1u)));
		out.push_back(TensorWriteRef(prefix + ".atlas.sparrowPastCov", &st.sparrowPastCov, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(st.r));
		sh.push_back(static_cast<uint64_t>(st.r + (st.complementRank ? st.complementRank : 1u)));
		out.push_back(TensorWriteRef(prefix + ".atlas.sparrowCrossCov", &st.sparrowCrossCov, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(st.sparrowModeRank));
		sh.push_back(static_cast<uint64_t>(st.r));
		out.push_back(TensorWriteRef(prefix + ".atlas.sparrowLeftMode", &st.sparrowLeftMode, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(st.sparrowModeRank));
		sh.push_back(static_cast<uint64_t>(st.r + (st.complementRank ? st.complementRank : 1u)));
		out.push_back(TensorWriteRef(prefix + ".atlas.sparrowRightMode", &st.sparrowRightMode, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(st.sparrowModeRank));
		sh.push_back(static_cast<uint64_t>(st.n));
		out.push_back(TensorWriteRef(prefix + ".atlas.sparrowLatent", &st.sparrowLatent, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(st.r));
		out.push_back(TensorWriteRef(prefix + ".atlas.qbrtLeftMode", &st.qbrtLeftMode, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(st.n));
		out.push_back(TensorWriteRef(prefix + ".atlas.qbrtLatent", &st.qbrtLatent, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(st.r));
		out.push_back(TensorWriteRef(prefix + ".atlas.qrcLeftMode", &st.qrcLeftMode, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(st.n));
		out.push_back(TensorWriteRef(prefix + ".atlas.qrcLatent", &st.qrcLatent, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(st.r));
		out.push_back(TensorWriteRef(prefix + ".atlas.riftLeftMode", &st.riftLeftMode, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(st.n));
		out.push_back(TensorWriteRef(prefix + ".atlas.riftLatent", &st.riftLatent, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(st.n));
		out.push_back(TensorWriteRef(prefix + ".atlas.orbitPrevSignal", &st.orbitPrevSignal, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(st.r));
		out.push_back(TensorWriteRef(prefix + ".atlas.orbitLeftMode", &st.orbitLeftMode, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(st.n));
		out.push_back(TensorWriteRef(prefix + ".atlas.orbitLatent", &st.orbitLatent, dt, sh));
	}
}

static bool restoreAtlasComplementBasis(glades::atlas::WeightState& st)
{
	const unsigned int compRank = (st.complementRank > 0u) ? st.complementRank : 1u;
	if (st.V.size() != static_cast<size_t>(st.m) * compRank)
		st.V.assign(static_cast<size_t>(st.m) * compRank, 0.0f);

	const unsigned int activeRank = (st.activeRank <= st.r) ? st.activeRank : st.r;
	unsigned int informative = 0u;
	for (unsigned int c = 0; c < compRank; ++c)
	{
		std::vector<float> col(static_cast<size_t>(st.m), 0.0f);
		for (unsigned int i = 0; i < st.m; ++i)
			col[i] = st.V[static_cast<size_t>(i) * compRank + c];

		for (unsigned int a = 0; a < activeRank; ++a)
		{
			double dot = 0.0;
			for (unsigned int i = 0; i < st.m; ++i)
				dot += static_cast<double>(col[i])
				    * static_cast<double>(st.U[static_cast<size_t>(i) * st.r + a]);
			const float dotf = static_cast<float>(dot);
			for (unsigned int i = 0; i < st.m; ++i)
				col[i] -= dotf * st.U[static_cast<size_t>(i) * st.r + a];
		}
		for (unsigned int prev = 0; prev < c; ++prev)
		{
			double dot = 0.0;
			for (unsigned int i = 0; i < st.m; ++i)
				dot += static_cast<double>(col[i])
				    * static_cast<double>(st.V[static_cast<size_t>(i) * compRank + prev]);
			const float dotf = static_cast<float>(dot);
			for (unsigned int i = 0; i < st.m; ++i)
				col[i] -= dotf * st.V[static_cast<size_t>(i) * compRank + prev];
		}

		double normSq = 0.0;
		for (unsigned int i = 0; i < st.m; ++i)
		{
			const double v = static_cast<double>(col[i]);
			normSq += v * v;
		}
		if (normSq <= 1e-12)
		{
			bool seeded = false;
			for (unsigned int basisRow = 0; basisRow < st.m && !seeded; ++basisRow)
			{
				std::fill(col.begin(), col.end(), 0.0f);
				col[basisRow] = 1.0f;
				for (unsigned int a = 0; a < activeRank; ++a)
				{
					double dot = 0.0;
					for (unsigned int i = 0; i < st.m; ++i)
						dot += static_cast<double>(col[i])
						    * static_cast<double>(st.U[static_cast<size_t>(i) * st.r + a]);
					const float dotf = static_cast<float>(dot);
					for (unsigned int i = 0; i < st.m; ++i)
						col[i] -= dotf * st.U[static_cast<size_t>(i) * st.r + a];
				}
				for (unsigned int prev = 0; prev < c; ++prev)
				{
					double dot = 0.0;
					for (unsigned int i = 0; i < st.m; ++i)
						dot += static_cast<double>(col[i])
						    * static_cast<double>(st.V[static_cast<size_t>(i) * compRank + prev]);
					const float dotf = static_cast<float>(dot);
					for (unsigned int i = 0; i < st.m; ++i)
						col[i] -= dotf * st.V[static_cast<size_t>(i) * compRank + prev];
				}
				normSq = 0.0;
				for (unsigned int i = 0; i < st.m; ++i)
				{
					const double v = static_cast<double>(col[i]);
					normSq += v * v;
				}
				seeded = (normSq > 1e-12);
			}
			if (normSq <= 1e-12)
			{
				for (unsigned int i = 0; i < st.m; ++i)
					st.V[static_cast<size_t>(i) * compRank + c] = 0.0f;
				continue;
			}
		}

		const float invNorm = static_cast<float>(1.0 / std::sqrt(normSq));
		for (unsigned int i = 0; i < st.m; ++i)
			st.V[static_cast<size_t>(i) * compRank + c] = col[i] * invNorm;
		informative = c + 1u;
	}

	for (unsigned int c = informative; c < compRank; ++c)
		for (unsigned int i = 0; i < st.m; ++i)
			st.V[static_cast<size_t>(i) * compRank + c] = 0.0f;
	return informative > 0u || compRank == 0u;
}

static bool restoreAtlasScoutBasis(glades::atlas::WeightState& st)
{
	const unsigned int compRank = (st.complementRank > 0u) ? st.complementRank : 1u;
	if (st.scoutBasis.size() != static_cast<size_t>(st.m) * compRank)
		st.scoutBasis.assign(static_cast<size_t>(st.m) * compRank, 0.0f);

	const unsigned int activeRank = (st.activeRank <= st.r) ? st.activeRank : st.r;
	unsigned int informative = 0u;
	for (unsigned int c = 0; c < compRank; ++c)
	{
		std::vector<float> col(static_cast<size_t>(st.m), 0.0f);
		for (unsigned int i = 0; i < st.m; ++i)
			col[i] = st.scoutBasis[static_cast<size_t>(i) * compRank + c];

		for (unsigned int a = 0; a < activeRank; ++a)
		{
			double dot = 0.0;
			for (unsigned int i = 0; i < st.m; ++i)
				dot += static_cast<double>(col[i])
				    * static_cast<double>(st.U[static_cast<size_t>(i) * st.r + a]);
			const float dotf = static_cast<float>(dot);
			for (unsigned int i = 0; i < st.m; ++i)
				col[i] -= dotf * st.U[static_cast<size_t>(i) * st.r + a];
		}
		for (unsigned int prev = 0; prev < compRank; ++prev)
		{
			double dot = 0.0;
			for (unsigned int i = 0; i < st.m; ++i)
				dot += static_cast<double>(col[i])
				    * static_cast<double>(st.V[static_cast<size_t>(i) * compRank + prev]);
			const float dotf = static_cast<float>(dot);
			for (unsigned int i = 0; i < st.m; ++i)
				col[i] -= dotf * st.V[static_cast<size_t>(i) * compRank + prev];
		}
		for (unsigned int prev = 0; prev < c; ++prev)
		{
			double dot = 0.0;
			for (unsigned int i = 0; i < st.m; ++i)
				dot += static_cast<double>(col[i])
				    * static_cast<double>(st.scoutBasis[static_cast<size_t>(i) * compRank + prev]);
			const float dotf = static_cast<float>(dot);
			for (unsigned int i = 0; i < st.m; ++i)
				col[i] -= dotf * st.scoutBasis[static_cast<size_t>(i) * compRank + prev];
		}

		double normSq = 0.0;
		for (unsigned int i = 0; i < st.m; ++i)
		{
			const double v = static_cast<double>(col[i]);
			normSq += v * v;
		}
		if (normSq <= 1e-12)
		{
			bool seeded = false;
			for (unsigned int basisRow = 0; basisRow < st.m && !seeded; ++basisRow)
			{
				std::fill(col.begin(), col.end(), 0.0f);
				col[basisRow] = 1.0f;
				for (unsigned int a = 0; a < activeRank; ++a)
				{
					double dot = 0.0;
					for (unsigned int i = 0; i < st.m; ++i)
						dot += static_cast<double>(col[i])
						    * static_cast<double>(st.U[static_cast<size_t>(i) * st.r + a]);
					const float dotf = static_cast<float>(dot);
					for (unsigned int i = 0; i < st.m; ++i)
						col[i] -= dotf * st.U[static_cast<size_t>(i) * st.r + a];
				}
				for (unsigned int prev = 0; prev < compRank; ++prev)
				{
					double dot = 0.0;
					for (unsigned int i = 0; i < st.m; ++i)
						dot += static_cast<double>(col[i])
						    * static_cast<double>(st.V[static_cast<size_t>(i) * compRank + prev]);
					const float dotf = static_cast<float>(dot);
					for (unsigned int i = 0; i < st.m; ++i)
						col[i] -= dotf * st.V[static_cast<size_t>(i) * compRank + prev];
				}
				for (unsigned int prev = 0; prev < c; ++prev)
				{
					double dot = 0.0;
					for (unsigned int i = 0; i < st.m; ++i)
						dot += static_cast<double>(col[i])
						    * static_cast<double>(st.scoutBasis[static_cast<size_t>(i) * compRank + prev]);
					const float dotf = static_cast<float>(dot);
					for (unsigned int i = 0; i < st.m; ++i)
						col[i] -= dotf * st.scoutBasis[static_cast<size_t>(i) * compRank + prev];
				}
				normSq = 0.0;
				for (unsigned int i = 0; i < st.m; ++i)
				{
					const double v = static_cast<double>(col[i]);
					normSq += v * v;
				}
				seeded = (normSq > 1e-12);
			}
			if (normSq <= 1e-12)
			{
				for (unsigned int i = 0; i < st.m; ++i)
					st.scoutBasis[static_cast<size_t>(i) * compRank + c] = 0.0f;
				continue;
			}
		}

		const float invNorm = static_cast<float>(1.0 / std::sqrt(normSq));
		for (unsigned int i = 0; i < st.m; ++i)
			st.scoutBasis[static_cast<size_t>(i) * compRank + c] = col[i] * invNorm;
		informative = c + 1u;
	}

	for (unsigned int c = informative; c < compRank; ++c)
		for (unsigned int i = 0; i < st.m; ++i)
			st.scoutBasis[static_cast<size_t>(i) * compRank + c] = 0.0f;
	return informative > 0u || compRank == 0u;
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
		std::ostringstream oss; oss << st.complementFisher;
		kv["atlas." + prefix + ".complementFisher"] = oss.str();
	}
	{
		std::ostringstream oss;
		for (size_t i = 0; i < st.complementBlock.size(); ++i)
		{
			if (i > 0u) oss << ' ';
			oss << st.complementBlock[i];
		}
		kv["atlas." + prefix + ".complementBlock"] = oss.str();
	}
	{
		std::ostringstream oss; oss << static_cast<unsigned long long>(st.step);
		kv["atlas." + prefix + ".step"] = oss.str();
	}
	{
		std::ostringstream oss; oss << static_cast<unsigned long long>(st.activeRank);
		kv["atlas." + prefix + ".activeRank"] = oss.str();
	}
	{
		std::ostringstream oss; oss << static_cast<unsigned long long>(st.activeComplementRank);
		kv["atlas." + prefix + ".activeComplementRank"] = oss.str();
	}
	{
		std::ostringstream oss; oss << static_cast<unsigned long long>(st.trialComplementRank);
		kv["atlas." + prefix + ".trialComplementRank"] = oss.str();
	}
	{
		std::ostringstream oss; oss << static_cast<unsigned long long>(st.trialComplementWins);
		kv["atlas." + prefix + ".trialComplementWins"] = oss.str();
	}
	{
		std::ostringstream oss; oss << st.trialComplementMean;
		kv["atlas." + prefix + ".trialComplementMean"] = oss.str();
	}
	{
		std::ostringstream oss; oss << st.trialComplementVar;
		kv["atlas." + prefix + ".trialComplementVar"] = oss.str();
	}
	{
		std::ostringstream oss; oss << st.sparrowPoleNumer;
		kv["atlas." + prefix + ".sparrowPoleNumer"] = oss.str();
	}
	{
		std::ostringstream oss; oss << st.sparrowPoleDenom;
		kv["atlas." + prefix + ".sparrowPoleDenom"] = oss.str();
	}
	{
		std::ostringstream oss; oss << st.sparrowPole;
		kv["atlas." + prefix + ".sparrowPole"] = oss.str();
	}
	{
		std::ostringstream oss; oss << st.qbrtPoleNumer;
		kv["atlas." + prefix + ".qbrtPoleNumer"] = oss.str();
	}
	{
		std::ostringstream oss; oss << st.qbrtPoleDenom;
		kv["atlas." + prefix + ".qbrtPoleDenom"] = oss.str();
	}
	{
		std::ostringstream oss; oss << st.qbrtPole;
		kv["atlas." + prefix + ".qbrtPole"] = oss.str();
	}
	{
		std::ostringstream oss; oss << st.qrcPoleNumer;
		kv["atlas." + prefix + ".qrcPoleNumer"] = oss.str();
	}
	{
		std::ostringstream oss; oss << st.qrcPoleDenom;
		kv["atlas." + prefix + ".qrcPoleDenom"] = oss.str();
	}
	{
		std::ostringstream oss; oss << st.qrcPole;
		kv["atlas." + prefix + ".qrcPole"] = oss.str();
	}
	{
		std::ostringstream oss; oss << st.riftPoleNumer;
		kv["atlas." + prefix + ".riftPoleNumer"] = oss.str();
	}
	{
		std::ostringstream oss; oss << st.riftPoleDenom;
		kv["atlas." + prefix + ".riftPoleDenom"] = oss.str();
	}
	{
		std::ostringstream oss; oss << st.riftPole;
		kv["atlas." + prefix + ".riftPole"] = oss.str();
	}
	{
		std::ostringstream oss; oss << st.orbitPoleNumer;
		kv["atlas." + prefix + ".orbitPoleNumer"] = oss.str();
	}
	{
		std::ostringstream oss; oss << st.orbitPoleDenom;
		kv["atlas." + prefix + ".orbitPoleDenom"] = oss.str();
	}
	{
		std::ostringstream oss; oss << st.orbitPole;
		kv["atlas." + prefix + ".orbitPole"] = oss.str();
	}
}

static void enqueueAtlasRead(std::vector<TensorReadRef>& out,
                              const std::string& prefix,
                              glades::atlas::WeightState& st,
                              unsigned int m, unsigned int n, unsigned int r,
                              unsigned int complementRankCfg,
                              unsigned int sparrowModeRankCfg,
                              const std::string& dt)
{
	st.m = m;
	st.n = n;
	st.r = r;
	st.activeRank = r;
	st.complementRank = (complementRankCfg > 0u) ? complementRankCfg : 1u;
	st.sparrowModeRank = (sparrowModeRankCfg > 0u) ? sparrowModeRankCfg : 1u;
	st.activeComplementRank = 0u;
	st.trialComplementRank = 0u;
	st.trialComplementWins = 0u;
	st.trialComplementMean = 0.0f;
	st.trialComplementVar = 0.0f;
	st.complementFisher = 0.0f;
	const size_t mr = static_cast<size_t>(m) * static_cast<size_t>(r);
	const size_t rn = static_cast<size_t>(r) * static_cast<size_t>(n);
	const size_t cr = static_cast<size_t>(st.complementRank);
	const size_t cn = cr * static_cast<size_t>(n);
	const size_t mc = static_cast<size_t>(m) * cr;
	st.U.resize(mr);
	st.fisherDiag.resize(r);
	st.V.resize(mc);
	st.scoutBasis.resize(mc);
	st.complementBlock.assign(cr * cr, 0.0f);
	st.scoutCov.assign(cr * cr, 0.0f);
	st.scoutNoise.assign(cr * cr, 0.0f);
	st.prevGz.resize(rn);
	st.prevPrevGz.resize(rn);
	st.prevGv.resize(cn);
	st.resolveGzHistory.assign(static_cast<size_t>(4u) * rn, 0.0f);
	st.heroGwHistory.assign(static_cast<size_t>(4u) * cn, 0.0f);
	st.sparrowPrevActive.assign(rn, 0.0f);
	st.sparrowPrevScout.assign(cn, 0.0f);
	st.sparrowFutureCov.assign(static_cast<size_t>(r) * static_cast<size_t>(r), 0.0f);
	st.sparrowPastCov.assign(static_cast<size_t>(r + st.complementRank) * static_cast<size_t>(r + st.complementRank), 0.0f);
	st.sparrowCrossCov.assign(static_cast<size_t>(r) * static_cast<size_t>(r + st.complementRank), 0.0f);
	st.sparrowLeftMode.assign(static_cast<size_t>(st.sparrowModeRank) * static_cast<size_t>(r), 0.0f);
	st.sparrowRightMode.assign(static_cast<size_t>(st.sparrowModeRank) * static_cast<size_t>(r + st.complementRank), 0.0f);
	st.sparrowLatent.assign(static_cast<size_t>(st.sparrowModeRank) * static_cast<size_t>(n), 0.0f);
	st.qbrtLeftMode.assign(static_cast<size_t>(r), 0.0f);
	st.qbrtLatent.assign(static_cast<size_t>(n), 0.0f);
	st.qrcLeftMode.assign(static_cast<size_t>(r), 0.0f);
	st.qrcLatent.assign(static_cast<size_t>(n), 0.0f);
	st.riftLeftMode.assign(static_cast<size_t>(r), 0.0f);
	st.riftLatent.assign(static_cast<size_t>(n), 0.0f);
	st.orbitPrevSignal.assign(static_cast<size_t>(n), 0.0f);
	st.orbitLeftMode.assign(static_cast<size_t>(r), 0.0f);
	st.orbitLatent.assign(static_cast<size_t>(n), 0.0f);

	// Allocate persistent scratch buffers (must match initWeightState).
	st.scratch_gz.resize(rn);
	st.scratch_corrected.resize(rn);
	st.scratch_gv.resize(cn);
	st.scratch_correctedV.resize(cn);
	st.scratch_gwScout.resize(cn);
	st.scratch_U_old.resize(mr);
	st.scratch_f_old.resize(static_cast<size_t>(r));
	st.scratch_B.resize(rn);
	st.scratch_Z.resize(mr);
	st.scratch_overlap.resize(static_cast<size_t>(r) * static_cast<size_t>(r));
	st.scratch_prevGzOld.resize(rn);
	st.scratch_prevPrevGzOld.resize(rn);
	st.scratch_resolveGzHistoryOld.resize(static_cast<size_t>(4u) * rn);
	st.scratch_heroGwHistoryOld.resize(static_cast<size_t>(4u) * cn);
	st.scratch_basisPacked.resize(mr);
	st.scratch_V_old.resize(mc);
	st.scratch_Bv.resize(cn);
	st.scratch_Zv.resize(mc);
	st.scratch_W_old.resize(mc);
	st.scratch_Bw.resize(cn);
	st.scratch_Zw.resize(mc);
	st.scratch_complementMat.resize(cr * cr);
	st.scratch_complementEigVec.resize(cr * cr);
	st.scratch_complementEigVal.resize(cr);
	st.scratch_scoutMat.resize(cr * cr);
	st.scratch_scoutEigVec.resize(cr * cr);
	st.scratch_scoutEigVal.resize(cr);
	st.scratch_sparrowActive.resize(rn);
	st.scratch_sparrowScout.resize(cn);
	st.scratch_sparrowPastSignal.resize(static_cast<size_t>(n));
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
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(r));
		sh.push_back(static_cast<uint64_t>(n));
		out.push_back(TensorReadRef(prefix + ".atlas.prevPrevGz", &st.prevPrevGz, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(4ull);
		sh.push_back(static_cast<uint64_t>(r));
		sh.push_back(static_cast<uint64_t>(n));
		out.push_back(TensorReadRef(prefix + ".atlas.resolveGzHistory", &st.resolveGzHistory, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(4ull);
		sh.push_back(static_cast<uint64_t>(st.complementRank));
		sh.push_back(static_cast<uint64_t>(n));
		out.push_back(TensorReadRef(prefix + ".atlas.heroGwHistory", &st.heroGwHistory, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(m));
		sh.push_back(static_cast<uint64_t>(st.complementRank));
		out.push_back(TensorReadRef(prefix + ".atlas.V", &st.V, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(m));
		sh.push_back(static_cast<uint64_t>(st.complementRank));
		out.push_back(TensorReadRef(prefix + ".atlas.scoutBasis", &st.scoutBasis, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(st.complementRank));
		sh.push_back(static_cast<uint64_t>(n));
		out.push_back(TensorReadRef(prefix + ".atlas.prevGv", &st.prevGv, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(st.complementRank));
		sh.push_back(static_cast<uint64_t>(st.complementRank));
		out.push_back(TensorReadRef(prefix + ".atlas.scoutCov", &st.scoutCov, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(st.complementRank));
		sh.push_back(static_cast<uint64_t>(st.complementRank));
		out.push_back(TensorReadRef(prefix + ".atlas.scoutNoise", &st.scoutNoise, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(r));
		sh.push_back(static_cast<uint64_t>(n));
		out.push_back(TensorReadRef(prefix + ".atlas.sparrowPrevActive", &st.sparrowPrevActive, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(st.complementRank));
		sh.push_back(static_cast<uint64_t>(n));
		out.push_back(TensorReadRef(prefix + ".atlas.sparrowPrevScout", &st.sparrowPrevScout, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(r));
		sh.push_back(static_cast<uint64_t>(r));
		out.push_back(TensorReadRef(prefix + ".atlas.sparrowFutureCov", &st.sparrowFutureCov, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(r + st.complementRank));
		sh.push_back(static_cast<uint64_t>(r + st.complementRank));
		out.push_back(TensorReadRef(prefix + ".atlas.sparrowPastCov", &st.sparrowPastCov, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(r));
		sh.push_back(static_cast<uint64_t>(r + st.complementRank));
		out.push_back(TensorReadRef(prefix + ".atlas.sparrowCrossCov", &st.sparrowCrossCov, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(st.sparrowModeRank));
		sh.push_back(static_cast<uint64_t>(r));
		out.push_back(TensorReadRef(prefix + ".atlas.sparrowLeftMode", &st.sparrowLeftMode, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(st.sparrowModeRank));
		sh.push_back(static_cast<uint64_t>(r + st.complementRank));
		out.push_back(TensorReadRef(prefix + ".atlas.sparrowRightMode", &st.sparrowRightMode, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(st.sparrowModeRank));
		sh.push_back(static_cast<uint64_t>(n));
		out.push_back(TensorReadRef(prefix + ".atlas.sparrowLatent", &st.sparrowLatent, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(r));
		out.push_back(TensorReadRef(prefix + ".atlas.qbrtLeftMode", &st.qbrtLeftMode, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(n));
		out.push_back(TensorReadRef(prefix + ".atlas.qbrtLatent", &st.qbrtLatent, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(r));
		out.push_back(TensorReadRef(prefix + ".atlas.qrcLeftMode", &st.qrcLeftMode, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(n));
		out.push_back(TensorReadRef(prefix + ".atlas.qrcLatent", &st.qrcLatent, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(r));
		out.push_back(TensorReadRef(prefix + ".atlas.riftLeftMode", &st.riftLeftMode, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(n));
		out.push_back(TensorReadRef(prefix + ".atlas.riftLatent", &st.riftLatent, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(n));
		out.push_back(TensorReadRef(prefix + ".atlas.orbitPrevSignal", &st.orbitPrevSignal, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(r));
		out.push_back(TensorReadRef(prefix + ".atlas.orbitLeftMode", &st.orbitLeftMode, dt, sh));
	}
	{
		std::vector<uint64_t> sh;
		sh.push_back(static_cast<uint64_t>(n));
		out.push_back(TensorReadRef(prefix + ".atlas.orbitLatent", &st.orbitLatent, dt, sh));
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
		std::map<std::string, std::string>::const_iterator it = kv.find("atlas." + prefix + ".complementFisher");
		if (it != kv.end())
		{
			std::istringstream iss(it->second);
			iss >> st.complementFisher;
		}
	}
	{
		std::map<std::string, std::string>::const_iterator it = kv.find("atlas." + prefix + ".complementBlock");
		if (it != kv.end())
		{
			std::istringstream iss(it->second);
			for (size_t i = 0; i < st.complementBlock.size(); ++i)
			{
				if (!(iss >> st.complementBlock[i]))
					break;
			}
		}
		else if (!st.complementBlock.empty())
		{
			std::fill(st.complementBlock.begin(), st.complementBlock.end(), 0.0f);
			st.complementBlock[0] = st.complementFisher;
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
	{
		std::map<std::string, std::string>::const_iterator it = kv.find("atlas." + prefix + ".activeComplementRank");
		if (it != kv.end())
		{
			std::istringstream iss(it->second);
			unsigned long long s = 0;
			iss >> s;
			if (s <= static_cast<unsigned long long>(st.complementRank))
				st.activeComplementRank = static_cast<unsigned int>(s);
		}
	}
	{
		std::map<std::string, std::string>::const_iterator it = kv.find("atlas." + prefix + ".trialComplementRank");
		if (it != kv.end())
		{
			std::istringstream iss(it->second);
			unsigned long long s = 0;
			iss >> s;
			if (s <= static_cast<unsigned long long>(st.complementRank))
				st.trialComplementRank = static_cast<unsigned int>(s);
		}
	}
	{
		std::map<std::string, std::string>::const_iterator it = kv.find("atlas." + prefix + ".trialComplementWins");
		if (it != kv.end())
		{
			std::istringstream iss(it->second);
			unsigned long long s = 0;
			iss >> s;
			st.trialComplementWins = static_cast<unsigned int>(s);
		}
	}
	{
		std::map<std::string, std::string>::const_iterator it = kv.find("atlas." + prefix + ".trialComplementMean");
		if (it != kv.end())
		{
			std::istringstream iss(it->second);
			iss >> st.trialComplementMean;
		}
	}
	{
		std::map<std::string, std::string>::const_iterator it = kv.find("atlas." + prefix + ".trialComplementVar");
		if (it != kv.end())
		{
			std::istringstream iss(it->second);
			iss >> st.trialComplementVar;
		}
	}
	{
		std::map<std::string, std::string>::const_iterator it = kv.find("atlas." + prefix + ".sparrowPoleNumer");
		if (it != kv.end())
		{
			std::istringstream iss(it->second);
			iss >> st.sparrowPoleNumer;
		}
	}
	{
		std::map<std::string, std::string>::const_iterator it = kv.find("atlas." + prefix + ".sparrowPoleDenom");
		if (it != kv.end())
		{
			std::istringstream iss(it->second);
			iss >> st.sparrowPoleDenom;
		}
	}
	{
		std::map<std::string, std::string>::const_iterator it = kv.find("atlas." + prefix + ".sparrowPole");
		if (it != kv.end())
		{
			std::istringstream iss(it->second);
			iss >> st.sparrowPole;
		}
	}
	{
		std::map<std::string, std::string>::const_iterator it = kv.find("atlas." + prefix + ".qbrtPoleNumer");
		if (it != kv.end())
		{
			std::istringstream iss(it->second);
			iss >> st.qbrtPoleNumer;
		}
	}
	{
		std::map<std::string, std::string>::const_iterator it = kv.find("atlas." + prefix + ".qbrtPoleDenom");
		if (it != kv.end())
		{
			std::istringstream iss(it->second);
			iss >> st.qbrtPoleDenom;
		}
	}
	{
		std::map<std::string, std::string>::const_iterator it = kv.find("atlas." + prefix + ".qbrtPole");
		if (it != kv.end())
		{
			std::istringstream iss(it->second);
			iss >> st.qbrtPole;
		}
	}
	{
		std::map<std::string, std::string>::const_iterator it = kv.find("atlas." + prefix + ".qrcPoleNumer");
		if (it != kv.end())
		{
			std::istringstream iss(it->second);
			iss >> st.qrcPoleNumer;
		}
	}
	{
		std::map<std::string, std::string>::const_iterator it = kv.find("atlas." + prefix + ".qrcPoleDenom");
		if (it != kv.end())
		{
			std::istringstream iss(it->second);
			iss >> st.qrcPoleDenom;
		}
	}
	{
		std::map<std::string, std::string>::const_iterator it = kv.find("atlas." + prefix + ".qrcPole");
		if (it != kv.end())
		{
			std::istringstream iss(it->second);
			iss >> st.qrcPole;
		}
	}
	{
		std::map<std::string, std::string>::const_iterator it = kv.find("atlas." + prefix + ".riftPoleNumer");
		if (it != kv.end())
		{
			std::istringstream iss(it->second);
			iss >> st.riftPoleNumer;
		}
	}
	{
		std::map<std::string, std::string>::const_iterator it = kv.find("atlas." + prefix + ".riftPoleDenom");
		if (it != kv.end())
		{
			std::istringstream iss(it->second);
			iss >> st.riftPoleDenom;
		}
	}
	{
		std::map<std::string, std::string>::const_iterator it = kv.find("atlas." + prefix + ".riftPole");
		if (it != kv.end())
		{
			std::istringstream iss(it->second);
			iss >> st.riftPole;
		}
	}
	{
		std::map<std::string, std::string>::const_iterator it = kv.find("atlas." + prefix + ".orbitPoleNumer");
		if (it != kv.end())
		{
			std::istringstream iss(it->second);
			iss >> st.orbitPoleNumer;
		}
	}
	{
		std::map<std::string, std::string>::const_iterator it = kv.find("atlas." + prefix + ".orbitPoleDenom");
		if (it != kv.end())
		{
			std::istringstream iss(it->second);
			iss >> st.orbitPoleDenom;
		}
	}
	{
		std::map<std::string, std::string>::const_iterator it = kv.find("atlas." + prefix + ".orbitPole");
		if (it != kv.end())
		{
			std::istringstream iss(it->second);
			iss >> st.orbitPole;
		}
	}
	if (st.trialComplementRank <= st.activeComplementRank
	    || st.trialComplementRank > st.complementRank)
	{
		st.trialComplementRank = 0u;
		st.trialComplementWins = 0u;
		st.trialComplementMean = 0.0f;
		st.trialComplementVar = 0.0f;
	}
	if (st.prevGv.empty())
		st.prevGv.assign(static_cast<size_t>(st.complementRank ? st.complementRank : 1u) * static_cast<size_t>(st.n), 0.0f);
	if (st.prevPrevGz.empty())
		st.prevPrevGz.assign(static_cast<size_t>(st.r) * static_cast<size_t>(st.n), 0.0f);
	if (st.resolveGzHistory.empty())
		st.resolveGzHistory.assign(static_cast<size_t>(4u) * static_cast<size_t>(st.r) * static_cast<size_t>(st.n), 0.0f);
	if (st.scratch_resolveGzHistoryOld.empty())
		st.scratch_resolveGzHistoryOld.assign(static_cast<size_t>(4u) * static_cast<size_t>(st.r) * static_cast<size_t>(st.n), 0.0f);
	if (st.heroGwHistory.empty())
		st.heroGwHistory.assign(static_cast<size_t>(4u) * static_cast<size_t>(st.complementRank ? st.complementRank : 1u) * static_cast<size_t>(st.n), 0.0f);
	if (st.scratch_heroGwHistoryOld.empty())
		st.scratch_heroGwHistoryOld.assign(static_cast<size_t>(4u) * static_cast<size_t>(st.complementRank ? st.complementRank : 1u) * static_cast<size_t>(st.n), 0.0f);
	if (st.sparrowPrevActive.empty())
		st.sparrowPrevActive.assign(static_cast<size_t>(st.r) * static_cast<size_t>(st.n), 0.0f);
	if (st.sparrowPrevScout.empty())
		st.sparrowPrevScout.assign(static_cast<size_t>(st.complementRank ? st.complementRank : 1u) * static_cast<size_t>(st.n), 0.0f);
	if (st.sparrowFutureCov.empty())
		st.sparrowFutureCov.assign(static_cast<size_t>(st.r) * static_cast<size_t>(st.r), 0.0f);
	if (st.sparrowPastCov.empty())
		st.sparrowPastCov.assign(static_cast<size_t>(st.r + (st.complementRank ? st.complementRank : 1u))
		                         * static_cast<size_t>(st.r + (st.complementRank ? st.complementRank : 1u)),
		                         0.0f);
	if (st.sparrowCrossCov.empty())
		st.sparrowCrossCov.assign(static_cast<size_t>(st.r)
		                          * static_cast<size_t>(st.r + (st.complementRank ? st.complementRank : 1u)),
		                          0.0f);
	if (st.sparrowModeRank == 0u)
		st.sparrowModeRank = 1u;
	if (st.sparrowLeftMode.empty())
		st.sparrowLeftMode.assign(static_cast<size_t>(st.sparrowModeRank) * static_cast<size_t>(st.r), 0.0f);
	if (st.sparrowRightMode.empty())
		st.sparrowRightMode.assign(static_cast<size_t>(st.sparrowModeRank)
		                           * static_cast<size_t>(st.r + (st.complementRank ? st.complementRank : 1u)),
		                           0.0f);
	if (st.sparrowLatent.empty())
		st.sparrowLatent.assign(static_cast<size_t>(st.sparrowModeRank) * static_cast<size_t>(st.n), 0.0f);
	if (st.qbrtLeftMode.empty())
		st.qbrtLeftMode.assign(static_cast<size_t>(st.r), 0.0f);
	if (st.qbrtLatent.empty())
		st.qbrtLatent.assign(static_cast<size_t>(st.n), 0.0f);
	if (st.qrcLeftMode.empty())
		st.qrcLeftMode.assign(static_cast<size_t>(st.r), 0.0f);
	if (st.qrcLatent.empty())
		st.qrcLatent.assign(static_cast<size_t>(st.n), 0.0f);
	if (st.riftLeftMode.empty())
		st.riftLeftMode.assign(static_cast<size_t>(st.r), 0.0f);
	if (st.riftLatent.empty())
		st.riftLatent.assign(static_cast<size_t>(st.n), 0.0f);
	if (st.orbitPrevSignal.empty())
		st.orbitPrevSignal.assign(static_cast<size_t>(st.n), 0.0f);
	if (st.orbitLeftMode.empty())
		st.orbitLeftMode.assign(static_cast<size_t>(st.r), 0.0f);
	if (st.orbitLatent.empty())
		st.orbitLatent.assign(static_cast<size_t>(st.n), 0.0f);
	if (st.scratch_sparrowActive.empty())
		st.scratch_sparrowActive.assign(static_cast<size_t>(st.r) * static_cast<size_t>(st.n), 0.0f);
	if (st.scratch_sparrowScout.empty())
		st.scratch_sparrowScout.assign(static_cast<size_t>(st.complementRank ? st.complementRank : 1u) * static_cast<size_t>(st.n), 0.0f);
	if (st.scratch_sparrowPastSignal.empty())
		st.scratch_sparrowPastSignal.assign(static_cast<size_t>(st.n), 0.0f);
	restoreAtlasComplementBasis(st);
	restoreAtlasScoutBasis(st);
	{
		double compTrace = 0.0;
		for (unsigned int c = 0; c < st.complementRank; ++c)
			compTrace += static_cast<double>(st.complementBlock[static_cast<size_t>(c) * st.complementRank + c]);
		st.complementFisher = static_cast<float>(compTrace);
	}
	if (st.totalTrace <= 0.0f)
	{
		double activeTrace = 0.0;
		for (unsigned int c = 0; c < st.activeRank; ++c)
			activeTrace += static_cast<double>(st.fisherDiag[c]);
		activeTrace += static_cast<double>(st.complementFisher);
		unsigned int effectiveComplementRank = 0u;
		if (st.complementFisher > 1e-12f)
		{
			for (unsigned int c = 0; c < st.complementRank; ++c)
			{
				double normSq = 0.0;
				for (unsigned int i = 0; i < st.m; ++i)
				{
					const double v = static_cast<double>(st.V[static_cast<size_t>(i) * st.complementRank + c]);
					normSq += v * v;
				}
				if (normSq <= 1e-12)
					break;
				effectiveComplementRank = c + 1u;
			}
		}
		const unsigned int complementDim = (st.m > st.activeRank + effectiveComplementRank)
		    ? (st.m - st.activeRank - effectiveComplementRank)
		    : 0u;
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
			if (includeOpt && isAtlas && trainingConfig.atlas.helmEnabled && tensorDff.helm.initialized)
			{
				const TensorDFFState::HelmState& hs = tensorDff.helm;
				const unsigned int pastDim = hs.hiddenDim + hs.outputDim;
				const unsigned int modeRank = hs.modeRank;
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(hs.hiddenDim));
					tensorsToWrite.push_back(TensorWriteRef("dff.helm.prevHiddenMean", &hs.prevHiddenMean, dt, sh));
				}
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(hs.outputDim));
					tensorsToWrite.push_back(TensorWriteRef("dff.helm.prevResidualMean", &hs.prevResidualMean, dt, sh));
				}
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(hs.hiddenDim));
					tensorsToWrite.push_back(TensorWriteRef("dff.helm.hiddenVar", &hs.hiddenVar, dt, sh));
				}
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(hs.outputDim));
					tensorsToWrite.push_back(TensorWriteRef("dff.helm.residualVar", &hs.residualVar, dt, sh));
				}
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(hs.outputDim));
					sh.push_back(static_cast<uint64_t>(pastDim));
					tensorsToWrite.push_back(TensorWriteRef("dff.helm.crossCov", &hs.crossCov, dt, sh));
				}
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(modeRank));
					sh.push_back(static_cast<uint64_t>(hs.outputDim));
					tensorsToWrite.push_back(TensorWriteRef("dff.helm.leftMode", &hs.leftMode, dt, sh));
				}
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(modeRank));
					sh.push_back(static_cast<uint64_t>(pastDim));
					tensorsToWrite.push_back(TensorWriteRef("dff.helm.rightMode", &hs.rightMode, dt, sh));
				}
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(modeRank));
					tensorsToWrite.push_back(TensorWriteRef("dff.helm.sigma", &hs.sigma, dt, sh));
				}
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(modeRank));
					tensorsToWrite.push_back(TensorWriteRef("dff.helm.latent", &hs.latent, dt, sh));
				}
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(modeRank));
					tensorsToWrite.push_back(TensorWriteRef("dff.helm.poleNumer", &hs.poleNumer, dt, sh));
				}
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(modeRank));
					tensorsToWrite.push_back(TensorWriteRef("dff.helm.poleDenom", &hs.poleDenom, dt, sh));
				}
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(modeRank));
					tensorsToWrite.push_back(TensorWriteRef("dff.helm.pole", &hs.pole, dt, sh));
				}
			}
			if (includeOpt && isAtlas && trainingConfig.atlas.asterEnabled && tensorDff.aster.initialized)
			{
				const TensorDFFState::AsterState& as = tensorDff.aster;
				const unsigned int featureDim = as.outputDim + (2u * as.controlDim);
				const unsigned int stateRank = as.stateRank;
				const unsigned int stateFeatureDim = stateRank + (2u * as.controlDim);
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(as.controlDim));
					tensorsToWrite.push_back(TensorWriteRef("dff.aster.prevControlMean", &as.prevControlMean, dt, sh));
				}
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(as.outputDim));
					tensorsToWrite.push_back(TensorWriteRef("dff.aster.prevResidualMean", &as.prevResidualMean, dt, sh));
				}
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(as.controlDim));
					tensorsToWrite.push_back(TensorWriteRef("dff.aster.controlVar", &as.controlVar, dt, sh));
				}
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(as.outputDim));
					tensorsToWrite.push_back(TensorWriteRef("dff.aster.residualVar", &as.residualVar, dt, sh));
				}
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(featureDim));
					sh.push_back(static_cast<uint64_t>(featureDim));
					tensorsToWrite.push_back(TensorWriteRef("dff.aster.pastCov", &as.pastCov, dt, sh));
				}
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(as.outputDim));
					sh.push_back(static_cast<uint64_t>(featureDim));
					tensorsToWrite.push_back(TensorWriteRef("dff.aster.crossCov", &as.crossCov, dt, sh));
				}
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(as.outputDim));
					sh.push_back(static_cast<uint64_t>(featureDim));
					tensorsToWrite.push_back(TensorWriteRef("dff.aster.theta", &as.theta, dt, sh));
				}
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(stateFeatureDim));
					sh.push_back(static_cast<uint64_t>(stateFeatureDim));
					tensorsToWrite.push_back(TensorWriteRef("dff.aster.statePastCov", &as.statePastCov, dt, sh));
				}
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(stateRank));
					sh.push_back(static_cast<uint64_t>(stateFeatureDim));
					tensorsToWrite.push_back(TensorWriteRef("dff.aster.stateCrossCov", &as.stateCrossCov, dt, sh));
				}
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(as.outputDim));
					sh.push_back(static_cast<uint64_t>(as.outputDim));
					tensorsToWrite.push_back(TensorWriteRef("dff.aster.innovationCov", &as.innovationCov, dt, sh));
				}
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(stateRank));
					sh.push_back(static_cast<uint64_t>(as.outputDim));
					tensorsToWrite.push_back(TensorWriteRef("dff.aster.innovationCross", &as.innovationCross, dt, sh));
				}
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(stateRank));
					sh.push_back(static_cast<uint64_t>(as.outputDim));
					tensorsToWrite.push_back(TensorWriteRef("dff.aster.leftMode", &as.leftMode, dt, sh));
				}
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(stateRank));
					sh.push_back(static_cast<uint64_t>(featureDim));
					tensorsToWrite.push_back(TensorWriteRef("dff.aster.rightMode", &as.rightMode, dt, sh));
				}
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(stateRank));
					tensorsToWrite.push_back(TensorWriteRef("dff.aster.sigma", &as.sigma, dt, sh));
				}
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(stateRank));
					tensorsToWrite.push_back(TensorWriteRef("dff.aster.latent", &as.latent, dt, sh));
				}
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(stateRank));
					tensorsToWrite.push_back(TensorWriteRef("dff.aster.poleNumer", &as.poleNumer, dt, sh));
				}
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(stateRank));
					tensorsToWrite.push_back(TensorWriteRef("dff.aster.poleDenom", &as.poleDenom, dt, sh));
				}
				{
					std::vector<uint64_t> sh;
					sh.push_back(static_cast<uint64_t>(stateRank));
					tensorsToWrite.push_back(TensorWriteRef("dff.aster.pole", &as.pole, dt, sh));
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
	const unsigned int atlasComplementRank = trainingConfig.atlas.complementRank;
	const unsigned int atlasSparrowModeRank = trainingConfig.atlas.sparrowModeRank;
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
				enqueueAtlasRead(expected, oss.str(), tensorDff.atlasState[t], m, n, r, atlasComplementRank, atlasSparrowModeRank, dt);
			}
		}
		if (includeOpt && isAtlas && trainingConfig.atlas.helmEnabled && tensorDff.helm.initialized)
		{
			const TensorDFFState::HelmState& hs = tensorDff.helm;
			const unsigned int pastDim = hs.hiddenDim + hs.outputDim;
			const unsigned int modeRank = hs.modeRank;
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(hs.hiddenDim));
				expected.push_back(TensorReadRef("dff.helm.prevHiddenMean", &tensorDff.helm.prevHiddenMean, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(hs.outputDim));
				expected.push_back(TensorReadRef("dff.helm.prevResidualMean", &tensorDff.helm.prevResidualMean, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(hs.hiddenDim));
				expected.push_back(TensorReadRef("dff.helm.hiddenVar", &tensorDff.helm.hiddenVar, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(hs.outputDim));
				expected.push_back(TensorReadRef("dff.helm.residualVar", &tensorDff.helm.residualVar, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(hs.outputDim));
				sh.push_back(static_cast<uint64_t>(pastDim));
				expected.push_back(TensorReadRef("dff.helm.crossCov", &tensorDff.helm.crossCov, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(modeRank));
				sh.push_back(static_cast<uint64_t>(hs.outputDim));
				expected.push_back(TensorReadRef("dff.helm.leftMode", &tensorDff.helm.leftMode, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(modeRank));
				sh.push_back(static_cast<uint64_t>(pastDim));
				expected.push_back(TensorReadRef("dff.helm.rightMode", &tensorDff.helm.rightMode, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(modeRank));
				expected.push_back(TensorReadRef("dff.helm.sigma", &tensorDff.helm.sigma, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(modeRank));
				expected.push_back(TensorReadRef("dff.helm.latent", &tensorDff.helm.latent, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(modeRank));
				expected.push_back(TensorReadRef("dff.helm.poleNumer", &tensorDff.helm.poleNumer, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(modeRank));
				expected.push_back(TensorReadRef("dff.helm.poleDenom", &tensorDff.helm.poleDenom, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(modeRank));
				expected.push_back(TensorReadRef("dff.helm.pole", &tensorDff.helm.pole, dt, sh));
			}
		}
		if (includeOpt && isAtlas && trainingConfig.atlas.asterEnabled && tensorDff.aster.initialized)
		{
			const TensorDFFState::AsterState& as = tensorDff.aster;
			const unsigned int featureDim = as.outputDim + (2u * as.controlDim);
			const unsigned int stateRank = as.stateRank;
			const unsigned int stateFeatureDim = stateRank + (2u * as.controlDim);
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(as.controlDim));
				expected.push_back(TensorReadRef("dff.aster.prevControlMean", &tensorDff.aster.prevControlMean, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(as.outputDim));
				expected.push_back(TensorReadRef("dff.aster.prevResidualMean", &tensorDff.aster.prevResidualMean, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(as.controlDim));
				expected.push_back(TensorReadRef("dff.aster.controlVar", &tensorDff.aster.controlVar, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(as.outputDim));
				expected.push_back(TensorReadRef("dff.aster.residualVar", &tensorDff.aster.residualVar, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(featureDim));
				sh.push_back(static_cast<uint64_t>(featureDim));
				expected.push_back(TensorReadRef("dff.aster.pastCov", &tensorDff.aster.pastCov, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(as.outputDim));
				sh.push_back(static_cast<uint64_t>(featureDim));
				expected.push_back(TensorReadRef("dff.aster.crossCov", &tensorDff.aster.crossCov, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(as.outputDim));
				sh.push_back(static_cast<uint64_t>(featureDim));
				expected.push_back(TensorReadRef("dff.aster.theta", &tensorDff.aster.theta, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(stateFeatureDim));
				sh.push_back(static_cast<uint64_t>(stateFeatureDim));
				expected.push_back(TensorReadRef("dff.aster.statePastCov", &tensorDff.aster.statePastCov, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(stateRank));
				sh.push_back(static_cast<uint64_t>(stateFeatureDim));
				expected.push_back(TensorReadRef("dff.aster.stateCrossCov", &tensorDff.aster.stateCrossCov, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(as.outputDim));
				sh.push_back(static_cast<uint64_t>(as.outputDim));
				expected.push_back(TensorReadRef("dff.aster.innovationCov", &tensorDff.aster.innovationCov, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(stateRank));
				sh.push_back(static_cast<uint64_t>(as.outputDim));
				expected.push_back(TensorReadRef("dff.aster.innovationCross", &tensorDff.aster.innovationCross, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(stateRank));
				sh.push_back(static_cast<uint64_t>(as.outputDim));
				expected.push_back(TensorReadRef("dff.aster.leftMode", &tensorDff.aster.leftMode, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(stateRank));
				sh.push_back(static_cast<uint64_t>(featureDim));
				expected.push_back(TensorReadRef("dff.aster.rightMode", &tensorDff.aster.rightMode, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(stateRank));
				expected.push_back(TensorReadRef("dff.aster.sigma", &tensorDff.aster.sigma, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(stateRank));
				expected.push_back(TensorReadRef("dff.aster.latent", &tensorDff.aster.latent, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(stateRank));
				expected.push_back(TensorReadRef("dff.aster.poleNumer", &tensorDff.aster.poleNumer, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(stateRank));
				expected.push_back(TensorReadRef("dff.aster.poleDenom", &tensorDff.aster.poleDenom, dt, sh));
			}
			{
				std::vector<uint64_t> sh;
				sh.push_back(static_cast<uint64_t>(stateRank));
				expected.push_back(TensorReadRef("dff.aster.pole", &tensorDff.aster.pole, dt, sh));
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
					enqueueAtlasRead(expected, oss.str(), hl.atlasWxh, m, n, r, atlasComplementRank, atlasSparrowModeRank, dt);
				}
				{
					const unsigned int m = hl.h, n = hl.h;
					const unsigned int r = std::min(atlasRank, std::min(m, n));
					std::ostringstream oss; oss << "rnn.h" << static_cast<unsigned long long>(l) << ".Whh";
					enqueueAtlasRead(expected, oss.str(), hl.atlasWhh, m, n, r, atlasComplementRank, atlasSparrowModeRank, dt);
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
			enqueueAtlasRead(expected, "rnn.o", tensorRnn.O.atlasWhy, m, n, r, atlasComplementRank, atlasSparrowModeRank, dt);
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
					enqueueAtlasRead(expected, oss.str(), hl.atlasW, m, n, r, atlasComplementRank, atlasSparrowModeRank, dt);
				}
				{
					const unsigned int m = tg.gateCount * hl.h, n = hl.h;
					const unsigned int r = std::min(atlasRank, std::min(m, n));
					std::ostringstream oss; oss << prefix << ".h" << static_cast<unsigned long long>(l) << ".U";
					enqueueAtlasRead(expected, oss.str(), hl.atlasU, m, n, r, atlasComplementRank, atlasSparrowModeRank, dt);
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
			enqueueAtlasRead(expected, std::string(prefix) + ".o", tg.O.atlasWhy, m, n, r, atlasComplementRank, atlasSparrowModeRank, dt);
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
					enqueueAtlasRead(expected, oss.str(), b.atlasWq, m, n, r, atlasComplementRank, atlasSparrowModeRank, dt);
				}
				{
					const unsigned int m = dModelKV, n = dModel;
					const unsigned int r = std::min(atlasRank, std::min(m, n));
					std::ostringstream oss; oss << "tr.b" << li << ".Wk";
					enqueueAtlasRead(expected, oss.str(), b.atlasWk, m, n, r, atlasComplementRank, atlasSparrowModeRank, dt);
				}
				{
					const unsigned int m = dModelKV, n = dModel;
					const unsigned int r = std::min(atlasRank, std::min(m, n));
					std::ostringstream oss; oss << "tr.b" << li << ".Wv";
					enqueueAtlasRead(expected, oss.str(), b.atlasWv, m, n, r, atlasComplementRank, atlasSparrowModeRank, dt);
				}
				{
					const unsigned int m = dModel, n = dModel;
					const unsigned int r = std::min(atlasRank, std::min(m, n));
					std::ostringstream oss; oss << "tr.b" << li << ".Wo";
					enqueueAtlasRead(expected, oss.str(), b.atlasWo, m, n, r, atlasComplementRank, atlasSparrowModeRank, dt);
				}
				{
					const unsigned int m = ff1Width, n = dModel;
					const unsigned int r = std::min(atlasRank, std::min(m, n));
					std::ostringstream oss; oss << "tr.b" << li << ".W1";
					enqueueAtlasRead(expected, oss.str(), b.atlasW1, m, n, r, atlasComplementRank, atlasSparrowModeRank, dt);
				}
				{
					const unsigned int m = dModel, n = tt.dFF;
					const unsigned int r = std::min(atlasRank, std::min(m, n));
					std::ostringstream oss; oss << "tr.b" << li << ".W2";
					enqueueAtlasRead(expected, oss.str(), b.atlasW2, m, n, r, atlasComplementRank, atlasSparrowModeRank, dt);
				}
			}
		}
		if (includeOpt && isAtlas)
		{
			{
				const unsigned int m = dModel, n = inputSize;
				const unsigned int r = std::min(atlasRank, std::min(m, n));
				enqueueAtlasRead(expected, "tr.WIn", tt.atlasWIn, m, n, r, atlasComplementRank, atlasSparrowModeRank, dt);
			}
			{
				const unsigned int m = outSize, n = dModel;
				const unsigned int r = std::min(atlasRank, std::min(m, n));
				enqueueAtlasRead(expected, "tr.WOut", tt.atlasWOut, m, n, r, atlasComplementRank, atlasSparrowModeRank, dt);
			}
			if (tt.tokenModel)
			{
				const unsigned int m = tt.vocabSize, n = dModel;
				const unsigned int r = std::min(atlasRank, std::min(m, n));
				enqueueAtlasRead(expected, "tr.tokE", tt.atlasTokE, m, n, r, atlasComplementRank, atlasSparrowModeRank, dt);
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
			if (trainingConfig.atlas.helmEnabled && tensorDff.helm.initialized)
			{
				// Legacy scalar HELM checkpoints stored the latent/pole state in the
				// manifest rather than as tensors. Preserve that fallback for the
				// rank-1 case so older checkpoints still resume cleanly.
				if (tensorDff.helm.latent.size() == 1u)
				{
					{
						std::map<std::string, std::string>::const_iterator it = kv.find("dff.helm.latent");
						if (it != kv.end())
						{
							std::istringstream iss(it->second);
							iss >> tensorDff.helm.latent[0];
						}
					}
					{
						std::map<std::string, std::string>::const_iterator it = kv.find("dff.helm.poleNumer");
						if (it != kv.end())
						{
							std::istringstream iss(it->second);
							iss >> tensorDff.helm.poleNumer[0];
						}
					}
					{
						std::map<std::string, std::string>::const_iterator it = kv.find("dff.helm.poleDenom");
						if (it != kv.end())
						{
							std::istringstream iss(it->second);
							iss >> tensorDff.helm.poleDenom[0];
						}
					}
					{
						std::map<std::string, std::string>::const_iterator it = kv.find("dff.helm.pole");
						if (it != kv.end())
						{
							std::istringstream iss(it->second);
							iss >> tensorDff.helm.pole[0];
						}
					}
				}
				std::fill(tensorDff.helm.batchHiddenSum.begin(), tensorDff.helm.batchHiddenSum.end(), 0.0f);
				std::fill(tensorDff.helm.batchHiddenSqSum.begin(), tensorDff.helm.batchHiddenSqSum.end(), 0.0f);
				std::fill(tensorDff.helm.batchResidualSum.begin(), tensorDff.helm.batchResidualSum.end(), 0.0f);
				std::fill(tensorDff.helm.batchResidualSqSum.begin(), tensorDff.helm.batchResidualSqSum.end(), 0.0f);
			}
			if (trainingConfig.atlas.asterEnabled && tensorDff.aster.initialized)
			{
				std::fill(tensorDff.aster.batchHiddenSum.begin(), tensorDff.aster.batchHiddenSum.end(), 0.0f);
				std::fill(tensorDff.aster.batchHiddenSqSum.begin(), tensorDff.aster.batchHiddenSqSum.end(), 0.0f);
				std::fill(tensorDff.aster.batchResidualSum.begin(), tensorDff.aster.batchResidualSum.end(), 0.0f);
				std::fill(tensorDff.aster.batchResidualSqSum.begin(), tensorDff.aster.batchResidualSqSum.end(), 0.0f);
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
