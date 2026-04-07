// GPU device detection and management (CUDA implementation).
#include "gpu_device.h"

#ifdef GLADES_HAVE_CUDA

#include <cuda_runtime.h>
#include <cstdio>

namespace glades {
namespace gpu {

namespace {

// Module-level state (singleton; not thread-safe for init, but init is
// expected to happen once at startup before any concurrent usage).
struct DeviceState
{
	bool initialized;
	bool available;
	int deviceId;
	cudaDeviceProp props;
	cudaStream_t computeStream;
	cudaStream_t transferStream;

	DeviceState()
	    : initialized(false),
	      available(false),
	      deviceId(-1),
	      props(),
	      computeStream(0),
	      transferStream(0)
	{
	}
};

static DeviceState g_state;

} // namespace

bool initDevice(int deviceId)
{
	if (g_state.initialized)
		return g_state.available;

	g_state.initialized = true;

	int deviceCount = 0;
	cudaError_t err = cudaGetDeviceCount(&deviceCount);
	if (err != cudaSuccess || deviceCount == 0)
	{
		fprintf(stderr, "[glades-cuda] No CUDA devices found (err=%d, count=%d)\n",
		        static_cast<int>(err), deviceCount);
		g_state.available = false;
		return false;
	}

	if (deviceId < 0 || deviceId >= deviceCount)
	{
		fprintf(stderr, "[glades-cuda] Requested device %d but only %d device(s) available\n",
		        deviceId, deviceCount);
		g_state.available = false;
		return false;
	}

	err = cudaSetDevice(deviceId);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "[glades-cuda] cudaSetDevice(%d) failed: %s\n",
		        deviceId, cudaGetErrorString(err));
		g_state.available = false;
		return false;
	}

	err = cudaGetDeviceProperties(&g_state.props, deviceId);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "[glades-cuda] cudaGetDeviceProperties failed: %s\n",
		        cudaGetErrorString(err));
		g_state.available = false;
		return false;
	}

	g_state.deviceId = deviceId;
	g_state.available = true;

	err = cudaStreamCreateWithFlags(&g_state.computeStream, cudaStreamNonBlocking);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "[glades-cuda] cudaStreamCreateWithFlags(compute) failed: %s\n",
		        cudaGetErrorString(err));
		g_state.available = false;
		return false;
	}
	err = cudaStreamCreateWithFlags(&g_state.transferStream, cudaStreamNonBlocking);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "[glades-cuda] cudaStreamCreateWithFlags(transfer) failed: %s\n",
		        cudaGetErrorString(err));
		cudaStreamDestroy(g_state.computeStream);
		g_state.computeStream = 0;
		g_state.available = false;
		return false;
	}

	fprintf(stderr, "[glades-cuda] Using device %d: %s (SM %d.%d, %zu MB, %d SMs)\n",
	        deviceId,
	        g_state.props.name,
	        g_state.props.major,
	        g_state.props.minor,
	        g_state.props.totalGlobalMem / (1024ULL * 1024ULL),
	        g_state.props.multiProcessorCount);

	return true;
}

bool isAvailable()
{
	return g_state.available;
}

int currentDevice()
{
	return g_state.deviceId;
}

const char* deviceName()
{
	return g_state.available ? g_state.props.name : "none";
}

int computeCapabilityMajor()
{
	return g_state.available ? g_state.props.major : 0;
}

int computeCapabilityMinor()
{
	return g_state.available ? g_state.props.minor : 0;
}

size_t totalGlobalMemBytes()
{
	return g_state.available ? g_state.props.totalGlobalMem : 0;
}

int multiprocessorCount()
{
	return g_state.available ? g_state.props.multiProcessorCount : 0;
}

int maxThreadsPerBlock()
{
	return g_state.available ? g_state.props.maxThreadsPerBlock : 0;
}

cudaStream_t computeStream()
{
	return g_state.available ? g_state.computeStream : 0;
}

cudaStream_t transferStream()
{
	return g_state.available ? g_state.transferStream : 0;
}

bool synchronizeStream(cudaStream_t stream)
{
	if (!g_state.available || !stream)
		return true;
	return cudaStreamSynchronize(stream) == cudaSuccess;
}

bool synchronizeComputeStream()
{
	return synchronizeStream(g_state.computeStream);
}

bool synchronizeTransferStream()
{
	return synchronizeStream(g_state.transferStream);
}

cudaEvent_t createEvent(bool enableTiming)
{
	if (!g_state.available)
		return 0;
	unsigned int flags = enableTiming ? cudaEventDefault : cudaEventDisableTiming;
	cudaEvent_t eventHandle = 0;
	if (cudaEventCreateWithFlags(&eventHandle, flags) != cudaSuccess)
		return 0;
	return eventHandle;
}

void destroyEvent(cudaEvent_t eventHandle)
{
	if (eventHandle)
		cudaEventDestroy(eventHandle);
}

bool recordEvent(cudaEvent_t eventHandle, cudaStream_t stream)
{
	if (!g_state.available || !eventHandle)
		return false;
	return cudaEventRecord(eventHandle, stream) == cudaSuccess;
}

bool streamWaitEvent(cudaStream_t stream, cudaEvent_t eventHandle)
{
	if (!g_state.available || !eventHandle)
		return false;
	return cudaStreamWaitEvent(stream, eventHandle, 0) == cudaSuccess;
}

bool synchronizeEvent(cudaEvent_t eventHandle)
{
	if (!g_state.available || !eventHandle)
		return false;
	return cudaEventSynchronize(eventHandle) == cudaSuccess;
}

void synchronize()
{
	if (g_state.available)
		cudaDeviceSynchronize();
}

bool synchronizeCheck(const char* phase)
{
	if (!g_state.available)
		return true;
	cudaError_t err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "[glades-cuda] ERROR at '%s': %s\n",
		        phase ? phase : "?", cudaGetErrorString(err));
		return false;
	}
	return true;
}

void resetDevice()
{
	if (g_state.available)
	{
		if (g_state.computeStream)
		{
			cudaStreamDestroy(g_state.computeStream);
			g_state.computeStream = 0;
		}
		if (g_state.transferStream)
		{
			cudaStreamDestroy(g_state.transferStream);
			g_state.transferStream = 0;
		}
		cudaDeviceReset();
		g_state.available = false;
		g_state.deviceId = -1;
	}
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
