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

	// Upgrade to cudaGraphInstantiateWithFlags (CUDA 11.4+).
	// The legacy cudaGraphInstantiate(graph, 0, 0, 0) uses conservative
	// default scheduling. cudaGraphInstantiateWithFlags with USE_NODE_PRIORITY
	// allows the runtime to schedule based on per-node priority attributes
	// rather than enforcing strict stream-order serialization.
	//
	// CUDA 13.2 does NOT expose CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_PARALLELISM
	// (unavailable in this version). USE_NODE_PRIORITY is a weaker but
	// related feature that may help with scheduling.  If memset serialization
	// remains a bottleneck, the next step is to either:
	// - Split the graph to isolate memsets into separate smaller graphs, or
	// - Capture memsets on a separate stream and skip graph capture for them.
	err = cudaGraphInstantiateWithFlags(&m_exec, m_graph, cudaGraphInstantiateFlagUseNodePriority);
	if (err != cudaSuccess) {
		std::fprintf(stderr, "[gpu-graph] cudaGraphInstantiateWithFlags failed: %s\n",
		             cudaGetErrorString(err));
		// Fall back to legacy API (shouldn't happen on CUDA 11.4+)
		err = cudaGraphInstantiate(&m_exec, m_graph, 0, 0, 0);
		if (err != cudaSuccess) {
			std::fprintf(stderr, "[gpu-graph] cudaGraphInstantiate fallback also failed: %s\n",
			             cudaGetErrorString(err));
			cudaGraphDestroy(m_graph);
			m_graph = 0;
			m_exec = 0;
			return false;
		}
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
