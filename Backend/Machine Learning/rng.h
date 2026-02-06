#pragma once

// Deterministic RNG for the ML engine.
//
// Goals:
// - Remove all raw rand()/srand() usage from ML code.
// - Make training runs reproducible when a seed is provided.
// - Make RNG per-network (no cross-talk between networks), while keeping existing call sites:
//     glades::rng::uniform_int/uniform_double/...
//
// Notes:
// - This is intentionally header-only to avoid build-system changes.
// - Default seed is fixed (5489) until explicitly set by the caller.

#include <stdint.h> // C++98-friendly fixed-width ints

namespace glades {
namespace rng {

// ===== Thread-local default engine (fallback) =====
//
// Architectural policy:
// - Production/ML code should NOT rely on implicit global/TLS RNG state.
// - Callers should pass an explicit `Engine&` to RNG functions.
//
// We still provide a TLS-backed `default_engine()` for legacy call sites and simple
// utilities, but we intentionally do NOT expose a "current engine" override mechanism.
//
// Concurrency policy:
// - Each `Engine` instance is NOT internally synchronized. A single Engine must not be used
//   concurrently from multiple threads.
#if defined(__cpp_thread_local) || (__cplusplus >= 201103L)
#define GLADES_THREAD_LOCAL thread_local
#elif defined(__GNUC__)
#define GLADES_THREAD_LOCAL __thread
#else
#define GLADES_THREAD_LOCAL
#endif

// ===== Engine =====
struct Engine
{
	uint64_t s;    // internal state
	uint64_t seed; // original seed (for reporting)
	Engine() : s(5489ULL), seed(5489ULL) {}
};

inline void seed_engine(Engine& e, uint64_t newSeed)
{
	e.seed = newSeed;
	// Avoid the all-zero state (degenerate for xorshift).
	e.s = (newSeed == 0ULL) ? 0x9E3779B97F4A7C15ULL : newSeed;
}

inline Engine& default_engine()
{
	// Thread-safe default RNG engine.
	//
	// IMPORTANT:
	// - In C++98 builds, we use `__thread` (when available) for TLS. GCC's `__thread` does not
	//   support non-POD types with constructors/destructors, so we store a TLS *pointer* and
	//   lazily allocate the Engine.
	// - This intentionally leaks one Engine per thread for the lifetime of the process.
	//   (In the ML engine, thread count is expected to be small and long-lived.)
	//
	// If TLS is not available (GLADES_THREAD_LOCAL is empty), this falls back to a single
	// process-global default engine and is NOT thread-safe. Production builds should enable TLS.
	static GLADES_THREAD_LOCAL Engine* p = 0;
	if (!p)
		p = new Engine();
	return *p;
}

// ===== Explicit-engine API (preferred) =====
//
// Note: This RNG is deterministic and stable across platforms by construction
// (xorshift64* with explicit uint64_t state). It is NOT cryptographically secure.

// xorshift64* PRNG (fast, deterministic, good enough for dropout/init).
// Reference: Marsaglia xorshift family.
inline uint64_t next_u64(Engine& e)
{
	uint64_t x = e.s;
	x ^= x >> 12;
	x ^= x << 25;
	x ^= x >> 27;
	e.s = x;
	return x * 2685821657736338717ULL;
}

inline int uniform_int(Engine& e, int loInclusive, int hiInclusive)
{
	if (hiInclusive <= loInclusive)
		return loInclusive;
	const uint64_t span = static_cast<uint64_t>(hiInclusive - loInclusive + 1);
	return loInclusive + static_cast<int>(next_u64(e) % span);
}

inline unsigned int uniform_uint(Engine& e, unsigned int loInclusive, unsigned int hiInclusive)
{
	if (hiInclusive <= loInclusive)
		return loInclusive;
	const uint64_t span = static_cast<uint64_t>(hiInclusive - loInclusive + 1U);
	return loInclusive + static_cast<unsigned int>(next_u64(e) % span);
}

inline float unit_float01(Engine& e)
{
	// Use 24 high bits to populate float mantissa-ish range: [0,1).
	const uint32_t x = static_cast<uint32_t>(next_u64(e) >> 40); // 24 bits
	return static_cast<float>(x) / 16777216.0f; // 2^24
}

inline double unit_double01(Engine& e)
{
	// Use 53 bits for double: [0,1).
	const uint64_t x = next_u64(e) >> 11; // 53 bits
	return static_cast<double>(x) / 9007199254740992.0; // 2^53
}

inline float uniform_float(Engine& e, float loInclusive, float hiExclusive)
{
	if (hiExclusive <= loInclusive)
		return loInclusive;
	return loInclusive + (hiExclusive - loInclusive) * unit_float01(e);
}

inline double uniform_double(Engine& e, double loInclusive, double hiExclusive)
{
	if (hiExclusive <= loInclusive)
		return loInclusive;
	return loInclusive + (hiExclusive - loInclusive) * unit_double01(e);
}

// ===== Legacy implicit-engine API (discouraged) =====
//
// These functions use a per-thread default engine. Prefer passing an explicit Engine&.
inline void seed(uint64_t s) { seed_engine(default_engine(), s); }
inline uint64_t current_seed() { return default_engine().seed; }
inline uint64_t next_u64() { return next_u64(default_engine()); }
inline int uniform_int(int loInclusive, int hiInclusive) { return uniform_int(default_engine(), loInclusive, hiInclusive); }
inline unsigned int uniform_uint(unsigned int loInclusive, unsigned int hiInclusive)
{
	return uniform_uint(default_engine(), loInclusive, hiInclusive);
}
inline float unit_float01() { return unit_float01(default_engine()); }
inline double unit_double01() { return unit_double01(default_engine()); }
inline float uniform_float(float loInclusive, float hiExclusive) { return uniform_float(default_engine(), loInclusive, hiExclusive); }
inline double uniform_double(double loInclusive, double hiExclusive) { return uniform_double(default_engine(), loInclusive, hiExclusive); }

} // namespace rng
} // namespace glades

