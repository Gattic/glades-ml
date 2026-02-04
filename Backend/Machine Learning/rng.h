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

// ===== Thread-local "current engine" plumbing =====
//
// We use a thread-local pointer so concurrent training in different threads doesn't interfere.
// If thread-local isn't available, this still works for single-threaded code, but won't be
// thread-safe.
//
// Concurrency policy note:
// - Each `Engine` instance is NOT internally synchronized. A single Engine must not be used
//   concurrently from multiple threads.
// - The training driver installs a per-network Engine into thread-local storage for the
//   duration of a run. This makes RNG "per-network" as long as each NNetwork instance is
//   run by at most one thread at a time.
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

inline Engine*& current_engine_ptr()
{
	static GLADES_THREAD_LOCAL Engine* p = 0;
	return p;
}

class ScopedEngine
{
public:
	explicit ScopedEngine(Engine* e) : prev_(current_engine_ptr())
	{
		current_engine_ptr() = e;
	}
	~ScopedEngine()
	{
		current_engine_ptr() = prev_;
	}

private:
	Engine* prev_;
};

inline Engine& current_engine()
{
	Engine* p = current_engine_ptr();
	return (p ? *p : default_engine());
}

// ===== Legacy API (now routed through current_engine()) =====
inline void seed(uint64_t s) { seed_engine(default_engine(), s); }
inline uint64_t current_seed() { return current_engine().seed; }

// xorshift64* PRNG (fast, deterministic, good enough for dropout/init).
// Reference: Marsaglia xorshift family.
inline uint64_t next_u64()
{
	Engine& e = current_engine();
	uint64_t x = e.s;
	x ^= x >> 12;
	x ^= x << 25;
	x ^= x >> 27;
	e.s = x;
	return x * 2685821657736338717ULL;
}

inline int uniform_int(int loInclusive, int hiInclusive)
{
	if (hiInclusive <= loInclusive)
		return loInclusive;
	const uint64_t span = static_cast<uint64_t>(hiInclusive - loInclusive + 1);
	return loInclusive + static_cast<int>(next_u64() % span);
}

inline unsigned int uniform_uint(unsigned int loInclusive, unsigned int hiInclusive)
{
	if (hiInclusive <= loInclusive)
		return loInclusive;
	const uint64_t span = static_cast<uint64_t>(hiInclusive - loInclusive + 1U);
	return loInclusive + static_cast<unsigned int>(next_u64() % span);
}

inline float unit_float01()
{
	// Use 24 high bits to populate float mantissa-ish range: [0,1).
	const uint32_t x = static_cast<uint32_t>(next_u64() >> 40); // 24 bits
	return static_cast<float>(x) / 16777216.0f; // 2^24
}

inline double unit_double01()
{
	// Use 53 bits for double: [0,1).
	const uint64_t x = next_u64() >> 11; // 53 bits
	return static_cast<double>(x) / 9007199254740992.0; // 2^53
}

inline float uniform_float(float loInclusive, float hiExclusive)
{
	if (hiExclusive <= loInclusive)
		return loInclusive;
	return loInclusive + (hiExclusive - loInclusive) * unit_float01();
}

inline double uniform_double(double loInclusive, double hiExclusive)
{
	if (hiExclusive <= loInclusive)
		return loInclusive;
	return loInclusive + (hiExclusive - loInclusive) * unit_double01();
}

} // namespace rng
} // namespace glades

