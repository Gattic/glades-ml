// Lightweight status type used across Glades ML.
//
// Motivation:
// - Historically many code paths failed via silent early returns or boolean flags.
// - A unified status code + message makes failures explicit and debuggable.
//
// This header is intentionally dependency-light so it can be included from DataObjects,
// Networks, persistence, and utilities without circular includes.
//
// Copyright 2026 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan

#pragma once

#include <string>

namespace glades {

struct NNetworkStatus
{
	enum Code
	{
		OK = 0,
		INVALID_ARGUMENT,
		INVALID_STATE,
		EMPTY_DATA,
		BUILD_FAILED,
		INTERNAL_ERROR
	};

	Code code;
	std::string message;

	NNetworkStatus(Code c = OK, const std::string& msg = std::string()) : code(c), message(msg) {}

	bool ok() const { return code == OK; }
};

} // namespace glades

