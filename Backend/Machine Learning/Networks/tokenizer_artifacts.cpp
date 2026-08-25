// Copyright 2026
//
// Tokenizer + vocabulary artifact management for model packages.
//
// This file intentionally does NOT implement tokenization algorithms. It only manages
// deployment metadata (vocab table + special token ids) so model packages are self-contained.

#include "network.h"

#include <sstream>

// Link anchor to force this translation unit into libglades.so builds
// when glades is linked from static sub-libraries.
extern "C" void glades_link_anchor_tokenizer_artifacts()
{
	// Intentionally empty.
}

namespace {

static glades::NNetworkStatus invalid_tokenizer_artifacts(const std::string& message)
{
	return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_ARGUMENT, message);
}

static bool contains_newline(const std::string& s)
{
	for (size_t i = 0; i < s.size(); ++i)
	{
		const char c = s[i];
		if (c == '\n' || c == '\r')
			return true;
	}
	return false;
}

static bool validate_special_id(const char* name, int id, size_t vocabSize, std::string& err)
{
	if (id < 0)
		return true; // unset is OK
	if (vocabSize == 0u)
	{
		std::ostringstream oss;
		oss << "setTokenizerArtifacts: special id '" << name << "' set but vocab is empty";
		err = oss.str();
		return false;
	}
	if (static_cast<size_t>(id) >= vocabSize)
	{
		std::ostringstream oss;
		oss << "setTokenizerArtifacts: special id '" << name << "' out of range: " << id
		    << " (vocab size " << vocabSize << ")";
		err = oss.str();
		return false;
	}
	return true;
}

} // namespace

namespace glades {

NNetworkStatus NNetwork::validateTokenizerArtifacts(const TokenizerArtifacts& a)
{
	// Basic validation (strict but dependency-free).
	// Type is optional but if set it must be safe for key/value manifests.
	if (a.type.size() > 64u)
		return invalid_tokenizer_artifacts("setTokenizerArtifacts: tokenizer type too long (max 64)");
	if (contains_newline(a.type))
		return invalid_tokenizer_artifacts("setTokenizerArtifacts: tokenizer type contains newline");

	// Vocab is required to make the artifact meaningful.
	if (!a.hasVocab())
		return invalid_tokenizer_artifacts("setTokenizerArtifacts: vocab is empty");

	// Hard caps for hostile environments (avoid pathological allocations).
	// These are generous for modern LLMs while still bounding memory use.
	static const size_t kMaxVocabSize = static_cast<size_t>(5u * 1000u * 1000u); // 5M tokens
	static const size_t kMaxTokenBytes = static_cast<size_t>(1024u * 1024u);     // 1 MiB per token
	static const size_t kMaxTotalBytes = static_cast<size_t>(1024ull * 1024ull * 1024ull); // 1 GiB total token bytes

	if (a.vocabSize() > kMaxVocabSize)
		return invalid_tokenizer_artifacts("setTokenizerArtifacts: vocab too large");

	// Duplicate detection + size accounting.
	// Avoid unordered_set to preserve compatibility with older toolchains.
	std::map<std::string, int> seen;
	size_t totalBytes = 0u;
	for (size_t i = 0; i < a.vocab.size(); ++i)
	{
		const std::string& tok = a.vocab[i];
		if (tok.size() > kMaxTokenBytes)
			return invalid_tokenizer_artifacts("setTokenizerArtifacts: vocab token too large");
		totalBytes += tok.size();
		if (totalBytes > kMaxTotalBytes)
			return invalid_tokenizer_artifacts("setTokenizerArtifacts: vocab total bytes too large");

		std::pair<std::map<std::string, int>::iterator, bool> ins = seen.insert(std::make_pair(tok, static_cast<int>(i)));
		if (!ins.second)
		{
			std::ostringstream oss;
			oss << "setTokenizerArtifacts: duplicate token detected at id " << i << " (also at id " << ins.first->second << ")";
			return invalid_tokenizer_artifacts(oss.str());
		}
	}

	std::string err;
	const size_t vocabSize = a.vocabSize();
	if (!validate_special_id("padTokenId", a.padTokenId, vocabSize, err))
		return invalid_tokenizer_artifacts(err);
	if (!validate_special_id("bosTokenId", a.bosTokenId, vocabSize, err))
		return invalid_tokenizer_artifacts(err);
	if (!validate_special_id("eosTokenId", a.eosTokenId, vocabSize, err))
		return invalid_tokenizer_artifacts(err);
	if (!validate_special_id("unkTokenId", a.unkTokenId, vocabSize, err))
		return invalid_tokenizer_artifacts(err);

	return NNetworkStatus(NNetworkStatus::OK, std::string());
}

NNetworkStatus NNetwork::setTokenizerArtifacts(const TokenizerArtifacts& a)
{
	// Enforce the same concurrency policy as other mutable config surfaces.
	RunLockGuard guard(*this);
	if (!guard.ok())
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "setTokenizerArtifacts: network is already running");

	const NNetworkStatus validation = validateTokenizerArtifacts(a);
	if (!validation.ok())
		return validation;

	tokenizerArtifacts = a;
	tokenizerArtifactsPresent = true;
	return NNetworkStatus(NNetworkStatus::OK, std::string());
}

void NNetwork::clearTokenizerArtifacts()
{
	RunLockGuard guard(*this);
	if (!guard.ok())
		return;
	tokenizerArtifactsPresent = false;
	tokenizerArtifacts.reset();
}

} // namespace glades
