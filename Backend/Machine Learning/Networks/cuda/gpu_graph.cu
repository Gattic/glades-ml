// CUDA Graph capture/launch wrapper.  See gpu_graph.h.
#ifdef GLADES_HAVE_CUDA

#include "gpu_graph.h"
#include <cuda_runtime.h>
#include <cstdio>

namespace glades {
namespace gpu {

GraphExec::GraphExec()
    : m_graph(0), m_exec(0), m_capturing(false)
{
}

GraphExec::~GraphExec()
{
	destroy();
}

bool GraphExec::beginCapture()
{
	if (m_capturing) {
		std::fprintf(stderr, "[gpu-graph] beginCapture: already capturing\n");
		return false;
	}
	destroy();

	cudaStream_t stream = computeStream();
	// Relaxed mode allows more ops (e.g. some sync points) than ThreadLocal,
	// which is necessary because some forward paths in CHIRON call
	// cudaStreamSynchronize for diagnostics or rare optimizer fallbacks.
	// The captured graph still respects all stream-ordered dependencies.
	cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
	if (err != cudaSuccess) {
		std::fprintf(stderr, "[gpu-graph] cudaStreamBeginCapture failed: %s\n",
		             cudaGetErrorString(err));
		return false;
	}
	m_capturing = true;
	return true;
}

bool GraphExec::endCaptureAndInstantiate()
{
	if (!m_capturing) {
		std::fprintf(stderr, "[gpu-graph] endCaptureAndInstantiate: not capturing\n");
		return false;
	}

	cudaStream_t stream = computeStream();
	cudaError_t err = cudaStreamEndCapture(stream, &m_graph);
	m_capturing = false;
	if (err != cudaSuccess) {
		std::fprintf(stderr, "[gpu-graph] cudaStreamEndCapture failed: %s\n",
		             cudaGetErrorString(err));
		if (m_graph) { cudaGraphDestroy(m_graph); m_graph = 0; }
		return false;
	}

	err = cudaGraphInstantiate(&m_exec, m_graph, 0, 0, 0);
	if (err != cudaSuccess) {
		std::fprintf(stderr, "[gpu-graph] cudaGraphInstantiate failed: %s\n",
		             cudaGetErrorString(err));
		cudaGraphDestroy(m_graph);
		m_graph = 0;
		m_exec = 0;
		return false;
	}
	return true;
}

bool GraphExec::launch()
{
	if (!m_exec) return false;
	cudaStream_t stream = computeStream();
	cudaError_t err = cudaGraphLaunch(m_exec, stream);
	if (err != cudaSuccess) {
		std::fprintf(stderr, "[gpu-graph] cudaGraphLaunch failed: %s\n",
		             cudaGetErrorString(err));
		return false;
	}
	return true;
}

void GraphExec::destroy()
{
	if (m_exec) {
		cudaGraphExecDestroy(m_exec);
		m_exec = 0;
	}
	if (m_graph) {
		cudaGraphDestroy(m_graph);
		m_graph = 0;
	}
	m_capturing = false;
	// Drain any sticky error left by a failed capture so subsequent CUDA
	// calls don't see "operation failed due to a previous error during
	// capture".  Safe regardless of current error state.
	(void)cudaGetLastError();
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
