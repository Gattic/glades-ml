// CUDA Graph capture/launch wrapper (paradigm #51 ATLAS-COMPILE).
//
// Captures a sequence of CUDA ops on `computeStream()` into a persistent
// `cudaGraphExec_t` that can be re-launched with a single API call instead
// of re-issuing each kernel/cuBLAS launch.  At small T launch overhead is
// 5–10× of useful work; at large T (1B class, T=512+) it's <1%.  ATLAS-COMPILE
// shows the largest wins in the small-T regime, but stays neutral or slightly
// positive at scale, so it's worth wiring for both.
//
// Constraints:
//   - The captured ops must touch only persistent device pointers (i.e. the
//     same allocations each step).  ANY cudaMalloc / cudaFree / cudaMallocAsync
//     in the capture window aborts the capture.
//   - Branches that change which kernels are launched (e.g. SAS-active mask
//     toggling layers on/off, ORION non-anchor skips) cannot be captured in
//     a single graph — capture one graph per distinct kernel-sequence.
//   - cuBLAS / cuBLASLt handles must have their stream set to the captured
//     stream BEFORE capture begins.  Our `gpu_blas.cu` already binds the
//     global cublas handle to `computeStream()` at init.
//
// Usage:
//   glades::gpu::GraphExec g;
//   if (!g.captured()) {
//       g.beginCapture();
//       // ... emit forward + backward kernels on computeStream() ...
//       g.endCaptureAndInstantiate();
//   }
//   g.launch();
//
// On any failure (e.g. an op inside the capture window does an unsafe alloc),
// `endCaptureAndInstantiate()` returns false and `g.captured()` stays false —
// the caller should fall back to direct kernel launches for that step.
#pragma once

#ifdef GLADES_HAVE_CUDA

#include "gpu_device.h"

struct CUgraph_st;
struct CUgraphExec_st;
typedef CUgraph_st* cudaGraph_t;
typedef CUgraphExec_st* cudaGraphExec_t;

namespace glades {
namespace gpu {

class GraphExec {
public:
	GraphExec();
	~GraphExec();

	// Begin capture on `computeStream()`.  Uses cudaStreamCaptureModeThreadLocal
	// so capture state doesn't leak into other threads.  Returns false if a
	// previous capture is still in flight or CUDA reports an error.
	bool beginCapture();

	// End capture and instantiate the executable graph.  On success,
	// `captured()` becomes true.  On failure (any unsafe op during the window),
	// any partial graph is destroyed and `captured()` stays false.
	bool endCaptureAndInstantiate();

	// Launch the previously captured graph on `computeStream()`.  Returns
	// false if `captured()` is false or CUDA reports an error.  Does NOT
	// implicitly synchronize.
	bool launch();

	// Tear down the instantiated graph.  Safe to call multiple times.  After
	// destroy(), `captured()` returns false again.
	void destroy();

	bool captured() const { return m_exec != 0; }
	bool capturing() const { return m_capturing; }

private:
	GraphExec(const GraphExec&);
	GraphExec& operator=(const GraphExec&);

	cudaGraph_t     m_graph;
	cudaGraphExec_t m_exec;
	bool            m_capturing;
};

} // namespace gpu
} // namespace glades

#else // !GLADES_HAVE_CUDA

namespace glades {
namespace gpu {

class GraphExec {
public:
	GraphExec() {}
	bool beginCapture() { return false; }
	bool endCaptureAndInstantiate() { return false; }
	bool launch() { return false; }
	void destroy() {}
	bool captured() const { return false; }
	bool capturing() const { return false; }
};

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
