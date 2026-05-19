// GPU device detection and management for Glades ML.
//
// Provides a thin abstraction over CUDA device queries so the rest of the
// codebase can check availability and select devices without touching CUDA
// headers directly.
#pragma once

#ifdef GLADES_HAVE_CUDA

struct CUstream_st;
struct CUevent_st;
typedef CUstream_st* cudaStream_t;
typedef CUevent_st* cudaEvent_t;

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

// Persistent non-blocking streams used by the training path.
cudaStream_t computeStream();
cudaStream_t transferStream();

// Synchronization helpers for specific streams.
bool synchronizeStream(cudaStream_t stream);
bool synchronizeComputeStream();
bool synchronizeTransferStream();

// Event helpers for explicit cross-stream dependencies.
cudaEvent_t createEvent(bool enableTiming = false);
void destroyEvent(cudaEvent_t eventHandle);
bool recordEvent(cudaEvent_t eventHandle, cudaStream_t stream);
bool streamWaitEvent(cudaStream_t stream, cudaEvent_t eventHandle);
bool synchronizeEvent(cudaEvent_t eventHandle);

// Synchronize the current device (blocks until all kernels complete).
void synchronize();

// Synchronize and check for errors.  Returns true on success.
// On failure, prints the phase label and CUDA error string to stderr.
bool synchronizeCheck(const char* phase);

// Reset/release the current device (called at shutdown).
void resetDevice();

} // namespace gpu
} // namespace glades

#else // !GLADES_HAVE_CUDA

typedef void* cudaStream_t;
typedef void* cudaEvent_t;

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
inline cudaStream_t computeStream() { return 0; }
inline cudaStream_t transferStream() { return 0; }
inline bool synchronizeStream(cudaStream_t) { return true; }
inline bool synchronizeComputeStream() { return true; }
inline bool synchronizeTransferStream() { return true; }
inline cudaEvent_t createEvent(bool = false) { return 0; }
inline void destroyEvent(cudaEvent_t) {}
inline bool recordEvent(cudaEvent_t, cudaStream_t) { return true; }
inline bool streamWaitEvent(cudaStream_t, cudaEvent_t) { return true; }
inline bool synchronizeEvent(cudaEvent_t) { return true; }
inline void synchronize() {}
inline bool synchronizeCheck(const char*) { return true; }
inline void resetDevice() {}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
