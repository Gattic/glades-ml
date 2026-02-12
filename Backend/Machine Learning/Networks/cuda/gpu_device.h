// GPU device detection and management for Glades ML.
//
// Provides a thin abstraction over CUDA device queries so the rest of the
// codebase can check availability and select devices without touching CUDA
// headers directly.
#pragma once

#ifdef GLADES_HAVE_CUDA

namespace glades {
namespace gpu {

// Initialize the CUDA runtime and select a device.
// Returns true if a usable GPU was found and selected.
// Safe to call multiple times (idempotent after first success).
bool initDevice(int deviceId = 0);

// Returns true if initDevice() succeeded and a GPU is ready.
bool isAvailable();

// Returns the device ID currently selected (-1 if none).
int currentDevice();

// Query device properties.
const char* deviceName();
int computeCapabilityMajor();
int computeCapabilityMinor();
size_t totalGlobalMemBytes();
int multiprocessorCount();
int maxThreadsPerBlock();

// Synchronize the current device (blocks until all kernels complete).
void synchronize();

// Reset/release the current device (called at shutdown).
void resetDevice();

} // namespace gpu
} // namespace glades

#else // !GLADES_HAVE_CUDA

namespace glades {
namespace gpu {

inline bool initDevice(int /*deviceId*/ = 0) { return false; }
inline bool isAvailable() { return false; }
inline int currentDevice() { return -1; }
inline const char* deviceName() { return "none"; }
inline int computeCapabilityMajor() { return 0; }
inline int computeCapabilityMinor() { return 0; }
inline size_t totalGlobalMemBytes() { return 0; }
inline int multiprocessorCount() { return 0; }
inline int maxThreadsPerBlock() { return 0; }
inline void synchronize() {}
inline void resetDevice() {}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
