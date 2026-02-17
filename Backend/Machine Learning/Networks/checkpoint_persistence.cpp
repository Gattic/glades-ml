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

// Checkpoint manifest format (current and only):
// - version=1
// - explicit file byte order + per-tensor dtype + per-tensor shape metadata
// - strict validation on load
static const int kCheckpointFormatVersion = 1;

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
                           const std::vector<TensorEntry>& tensors)
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

} // namespace

namespace glades {

NNetworkStatus NNetwork::saveCheckpoint(const std::string& checkpointName, const CheckpointConfig& cfg) const
{
	if (checkpointName.empty())
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "saveCheckpoint: checkpointName is empty");
	if (!is_safe_path_component(checkpointName))
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "saveCheckpoint: checkpointName contains unsafe characters");
	if (!skeleton)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "saveCheckpoint: skeleton is null");

	// Require tensors already initialized (checkpointing is for resumable training).
	{
		const bool hasDff = (netType == TYPE_DFF) && tensorDff.initialized;
		const bool hasRnn = (netType == TYPE_RNN) && tensorRnn.initialized;
		const bool hasGru = (netType == TYPE_GRU) && tensorGru.initialized;
		const bool hasLstm = (netType == TYPE_LSTM) && tensorLstm.initialized;
		const bool hasTr = (netType == TYPE_TRANSFORMER_ENCODER || netType == TYPE_TRANSFORMER_DECODER) && tensorTransformer.initialized;
		if (!hasDff && !hasRnn && !hasGru && !hasLstm && !hasTr)
			return NNetworkStatus(NNetworkStatus::INVALID_STATE, "saveCheckpoint: tensors are not initialized (run train/test or loadModel first)");
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
				return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "saveCheckpoint: ADAMW optimizer is only supported for transformer net types");
			if (!cfg.includeOptimizerState)
				return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "saveCheckpoint: ADAMW requires includeOptimizerState=true for resumable checkpoints");
		}
	}

	// Hardened write strategy:
	// - write into a temp directory under the checkpoint root
	// - atomically publish by renaming temp dir -> final dir
	// This avoids partially-written checkpoints if the process crashes mid-write.
	if (!ensure_checkpoint_root())
		return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveCheckpoint: unable to create checkpoint root directory");

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
		return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveCheckpoint: unable to create temporary checkpoint directory");

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
			return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveCheckpoint: unable to write nninfo.csv");
		}
	}

	// 2) Collect tensors
	std::vector<TensorWriteRef> tensorsToWrite;
	tensorsToWrite.clear();
	{
		const bool includeOpt = cfg.includeOptimizerState;
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
			}
		}
		else
		{
			return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "saveCheckpoint: unknown netType");
		}
	}

	// 3) Write sharded blobs
	const size_t maxShardBytes = (cfg.maxShardBytes ? cfg.maxShardBytes : static_cast<size_t>(1024ull * 1024ull * 1024ull));
	CheckpointShardWriter writer(dir, maxShardBytes);
	if (!writer.open_first())
		return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveCheckpoint: unable to open shard_000.bin for writing");

	for (size_t i = 0; i < tensorsToWrite.size(); ++i)
	{
		if (!tensorsToWrite[i].vec)
			return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveCheckpoint: internal error (null tensor vec)");
		if (!writer.add_tensor(tensorsToWrite[i].name, *tensorsToWrite[i].vec, tensorsToWrite[i].dtype, tensorsToWrite[i].shape))
			return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveCheckpoint: failed while writing shard data");
	}
	if (!writer.finalize_all())
		return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveCheckpoint: failed to finalize shard data");

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
	                    writer.get_tensors()))
	{
		return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveCheckpoint: unable to write manifest.txt");
	}

	// 5) Atomically publish:
	// - if an existing checkpoint directory exists, rename it to a backup
	// - rename temp dir -> final dir
	// - best-effort delete the backup
	std::string backupDirNoSlash;
	if (stat_is_dir(finalDirNoSlash))
	{
		std::ostringstream bak;
		bak << checkpointName << ".bak_" << static_cast<unsigned long long>(::getpid()) << "_" << static_cast<unsigned long long>(time(NULL));
		backupDirNoSlash = join_dir(root, bak.str());
		if (!rename_atomic(finalDirNoSlash, backupDirNoSlash))
			return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveCheckpoint: unable to rotate existing checkpoint directory");
	}
	if (!rename_atomic(tmpDirNoSlash, finalDirNoSlash))
	{
		// Best-effort rollback.
		if (!backupDirNoSlash.empty())
			(void)rename_atomic(backupDirNoSlash, finalDirNoSlash);
		return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "saveCheckpoint: unable to publish checkpoint directory (rename failed)");
	}
	tmpGuard.dismiss();
	if (!backupDirNoSlash.empty())
		(void)remove_tree_recursive(backupDirNoSlash);

	return NNetworkStatus(NNetworkStatus::OK, std::string());
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

	// Read manifest
	std::map<std::string, std::string> kv;
	if (!read_kv_file(manifestPath, kv))
		return failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: manifest.txt not found or unreadable");
	if (kv.find("__magic__") == kv.end() || kv["__magic__"] != "GLADES_CHECKPOINT")
		return failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: manifest magic mismatch");
	int version = -1;
	if (!parse_int(kv, "version", version) || version != kCheckpointFormatVersion)
		return failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: unsupported checkpoint format version");

	// Validate explicit file encoding metadata early.
	{
		std::map<std::string, std::string>::const_iterator itE = kv.find("file.endian");
		if (itE == kv.end())
			return failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: missing file.endian");
		if (itE->second != "little")
			return failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: unsupported file.endian (expected little)");

		std::map<std::string, std::string>::const_iterator itEnc = kv.find("file.tensorEncoding");
		if (itEnc == kv.end())
			return failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: missing file.tensorEncoding");
		if (itEnc->second != "raw_le")
			return failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: unsupported file.tensorEncoding");
	}

	int savedNetType = -1;
	parse_int(kv, "netType", savedNetType);
	if (savedNetType < 0)
		savedNetType = TYPE_DFF;
	if (netTypeOverride >= 0)
		netType = netTypeOverride;
	else
		netType = savedNetType;

	// Restore metadata/config before allocating tensors.
	int savedEpochs = 0;
	if (parse_int(kv, "epochs", savedEpochs))
		epochs = savedEpochs;
	uint64_t savedSeed = 0u;
	if (parse_u64(kv, "rngSeed", savedSeed))
		setSeed(savedSeed);
	bool includeOpt = true;
	parse_bool01(kv, "includeOptimizerState", includeOpt);
	{
		glades::TrainingConfig cfgTmp = trainingConfig;
		bool any = false;
		apply_training_config_from_kv(kv, cfgTmp, any);
		if (any)
			trainingConfig = cfgTmp;
	}

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
	if (netType == TYPE_DFF)
	{
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

	// Parse shardCount/tensorCount
	size_t shardCount = 0u;
	size_t tensorCount = 0u;
	parse_size_t(kv, "shardCount", shardCount);
	parse_size_t(kv, "tensorCount", tensorCount);
	if (shardCount == 0u && tensorCount != 0u)
		return failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: shardCount is 0 but tensorCount is non-zero");

	// Open shards
	std::vector< shmea::GPointer<std::ifstream> > shardStreams;
	shardStreams.resize(shardCount);
	std::vector<uint64_t> shardBytes;
	shardBytes.assign(shardCount, 0u);
	std::vector<uint64_t> shardHashExpected;
	shardHashExpected.assign(shardCount, 0u);
	for (size_t s = 0; s < shardCount; ++s)
	{
		std::ostringstream kf; kf << "shard." << static_cast<unsigned long long>(s) << ".file";
		std::ostringstream kb; kb << "shard." << static_cast<unsigned long long>(s) << ".bytes";
		std::ostringstream kh; kh << "shard." << static_cast<unsigned long long>(s) << ".fnv1a64";
		if (kv.find(kf.str()) == kv.end())
		{
			const NNetworkStatus st = failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: missing shard file entry in manifest");
			close_and_delete_shards(shardStreams);
			return st;
		}
		uint64_t bytesU = 0u;
		if (!parse_u64(kv, kb.str(), bytesU))
		{
			const NNetworkStatus st = failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: missing shard bytes entry in manifest");
			close_and_delete_shards(shardStreams);
			return st;
		}
		uint64_t hashU = 0u;
		if (!parse_u64(kv, kh.str(), hashU))
		{
			const NNetworkStatus st = failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: missing shard checksum entry in manifest");
			close_and_delete_shards(shardStreams);
			return st;
		}
		const std::string shardFile = kv[kf.str()];
		if (!is_safe_shard_filename(shardFile))
		{
			const NNetworkStatus st = failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: unsafe shard filename in manifest");
			close_and_delete_shards(shardStreams);
			return st;
		}
		// Validate file exists and is large enough.
		{
			struct stat st;
			const std::string shardPath = dir + shardFile;
			// Do not follow symlinks for checkpoint shards.
			if (::lstat(shardPath.c_str(), &st) != 0 || S_ISLNK(st.st_mode) || !S_ISREG(st.st_mode))
			{
				const NNetworkStatus stt = failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: shard file missing or not a regular file");
				close_and_delete_shards(shardStreams);
				return stt;
			}
			const uint64_t fileBytes = static_cast<uint64_t>(st.st_size);
			if (fileBytes < bytesU)
			{
				const NNetworkStatus stt = failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: shard file is smaller than manifest bytes");
				close_and_delete_shards(shardStreams);
				return stt;
			}
		}
		shmea::GPointer<std::ifstream> in(new std::ifstream((dir + shardFile).c_str(), std::ios::in | std::ios::binary));
		if (!in || !(*in.get()))
		{
			const NNetworkStatus st = failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: unable to open shard file");
			close_and_delete_shards(shardStreams);
			return st;
		}
		shardStreams[s] = in;
		shardBytes[s] = bytesU;
		shardHashExpected[s] = hashU;

		// Optional hostile-environment verification: validate whole-shard checksum.
		// This is stronger than per-tensor checks, but can be expensive for huge checkpoints.
		// Enable by setting `GLADES_CHECKPOINT_VERIFY_SHARDS=1`.
		if (should_verify_shards())
		{
			uint64_t computed = 0u;
			const std::string shardPath = dir + shardFile;
			if (!fnv1a64_hash_file_prefix(shardPath, bytesU, computed) || computed != hashU)
			{
				const NNetworkStatus st = failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: shard checksum mismatch");
				close_and_delete_shards(shardStreams);
				return st;
			}
		}
	}

	// Load each tensor entry into its destination vector.
	for (size_t i = 0; i < tensorCount; ++i)
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

		if (kv.find(kn.str()) == kv.end())
		{
			const NNetworkStatus st = failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: missing tensor name in manifest");
			close_and_delete_shards(shardStreams);
			return st;
		}
		const std::string tname = kv[kn.str()];

		uint64_t countU = 0u;
		uint64_t shardU = 0u;
		uint64_t offU = 0u;
		uint64_t bytesU = 0u;
		uint64_t hashU = 0u;
		if (!parse_u64(kv, kc.str(), countU) || !parse_u64(kv, ks.str(), shardU) || !parse_u64(kv, ko.str(), offU) ||
		    !parse_u64(kv, kb.str(), bytesU) || !parse_u64(kv, kh.str(), hashU))
		{
			const NNetworkStatus st = failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: malformed tensor entry in manifest");
			close_and_delete_shards(shardStreams);
			return st;
		}

		// v2: parse and validate per-tensor metadata.
		std::string dtypeU = "f32";
		uint64_t elemBytesU = 4ull;
		std::vector<uint64_t> shapeU64;
		{
			if (kv.find(kd.str()) == kv.end())
			{
				const NNetworkStatus st = failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: missing tensor dtype");
				close_and_delete_shards(shardStreams);
				return st;
			}
			dtypeU = kv[kd.str()];
			if (!parse_u64(kv, ke.str(), elemBytesU))
			{
				const NNetworkStatus st = failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: missing tensor elemBytes");
				close_and_delete_shards(shardStreams);
				return st;
			}
			uint64_t rankU = 0u;
			if (!parse_u64(kv, kr.str(), rankU) || rankU > 8u)
			{
				const NNetworkStatus st = failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: invalid tensor rank");
				close_and_delete_shards(shardStreams);
				return st;
			}
			// Rank==0 is allowed only if count==0 (scalar empty). Otherwise reject.
			if (rankU == 0u && countU != 0u)
			{
				const NNetworkStatus st = failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: invalid tensor rank (0 with non-zero count)");
				close_and_delete_shards(shardStreams);
				return st;
			}
			if (rankU > 0u)
			{
				if (kv.find(ksh.str()) == kv.end() || !parse_u64_list_value(kv[ksh.str()], shapeU64) || shapeU64.size() != static_cast<size_t>(rankU))
				{
					const NNetworkStatus st = failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: invalid tensor shape");
					close_and_delete_shards(shardStreams);
					return st;
				}
			}
			else
			{
				shapeU64.clear();
			}

			if (dtypeU != "f32" || elemBytesU != 4ull)
			{
				const NNetworkStatus st = failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: unsupported tensor dtype/elemBytes (expected f32/4)");
				close_and_delete_shards(shardStreams);
				return st;
			}
			uint64_t shapeCount = 0u;
			if (rankU == 0u)
			{
				shapeCount = 0u;
			}
			else if (!checked_shape_elem_count(shapeU64, shapeCount))
			{
				const NNetworkStatus st = failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: invalid tensor shape (overflow)");
				close_and_delete_shards(shardStreams);
				return st;
			}
			if (shapeCount != countU)
			{
				const NNetworkStatus st = failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: tensor shape product != count (manifest corrupted)");
				close_and_delete_shards(shardStreams);
				return st;
			}
		}
		// Manifest sanity checks.
		// Note: for now, only f32 is supported, so 4-byte alignment is required.
		if ((offU % 4ull) != 0ull)
		{
			const NNetworkStatus st = failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: tensor offsetBytes is not 4-byte aligned");
			close_and_delete_shards(shardStreams);
			return st;
		}
		const uint64_t wantBytes = (countU * elemBytesU);
		if (bytesU != wantBytes)
		{
			const NNetworkStatus st = failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: tensor bytes != count*elemBytes (manifest corrupted)");
			close_and_delete_shards(shardStreams);
			return st;
		}
		if (shardU >= static_cast<uint64_t>(shardCount))
		{
			const NNetworkStatus st = failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: tensor references out-of-range shard");
			close_and_delete_shards(shardStreams);
			return st;
		}
		// Ensure tensor window fits within shard bytes.
		if (!shardBytes.empty() && static_cast<size_t>(shardU) < shardBytes.size())
		{
			const uint64_t sb = shardBytes[static_cast<size_t>(shardU)];
			if (offU > sb || bytesU > sb || offU + bytesU > sb)
			{
				const NNetworkStatus st = failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: tensor range exceeds shard bytes (manifest corrupted)");
				close_and_delete_shards(shardStreams);
				return st;
			}
		}

		std::map<std::string, std::vector<float>*>::iterator it = nameToVec.find(tname);
		if (it == nameToVec.end())
		{
			// Unknown tensor in checkpoint; reject (strict).
			const NNetworkStatus st = failStatus(NNetworkStatus::INVALID_STATE, std::string("loadCheckpoint: unexpected tensor in checkpoint: ") + tname);
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

		const size_t wantCount = static_cast<size_t>(countU);
		// Strict shape compatibility: element count must match what we allocated.
		if (dst->size() != wantCount)
		{
			std::ostringstream oss;
			oss << "loadCheckpoint: tensor size mismatch for " << tname << " (checkpoint " << static_cast<unsigned long long>(wantCount)
			    << " vs model " << static_cast<unsigned long long>(dst->size()) << ")";
			const NNetworkStatus st = failStatus(NNetworkStatus::INVALID_STATE, oss.str());
			close_and_delete_shards(shardStreams);
			return st;
		}

		// Shape/dtype must match the expected tensor spec (stronger than count-only).
		{
			std::map<std::string, std::string>::const_iterator itd = nameToDType.find(tname);
			std::map<std::string, std::vector<uint64_t> >::const_iterator its = nameToShape.find(tname);
			if (itd == nameToDType.end() || its == nameToShape.end())
			{
				const NNetworkStatus st = failStatus(NNetworkStatus::INTERNAL_ERROR, "loadCheckpoint: internal error (missing expected tensor metadata)");
				close_and_delete_shards(shardStreams);
				return st;
			}
			if (dtypeU != itd->second)
			{
				const NNetworkStatus st = failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: tensor dtype mismatch (checkpoint vs model)");
				close_and_delete_shards(shardStreams);
				return st;
			}
			if (shapeU64 != its->second)
			{
				const NNetworkStatus st = failStatus(NNetworkStatus::INVALID_STATE, "loadCheckpoint: tensor shape mismatch (checkpoint vs model)");
				close_and_delete_shards(shardStreams);
				return st;
			}
		}

		uint64_t computed = 0u;
		if (!shardStreams[static_cast<size_t>(shardU)] ||
		    !read_f32_blob_from_shard(*shardStreams[static_cast<size_t>(shardU)], offU, *dst, wantCount, hashU, computed))
		{
			const NNetworkStatus st = failStatus(NNetworkStatus::INVALID_STATE, std::string("loadCheckpoint: checksum/read failed for tensor ") + tname);
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

	return NNetworkStatus(NNetworkStatus::OK, std::string());
}

} // namespace glades
