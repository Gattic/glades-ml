// DDP (Distributed Data Parallel) communication abstraction.
//
// Uses ShmeaDB's GServer networking for gradient exchange between workers.
// All functions are no-ops when DDP is not initialized or worldSize==1.
#ifndef _GLADES_DDP_COMM_H
#define _GLADES_DDP_COMM_H

#include <stddef.h>
#include <string>
#include <vector>

namespace glades {
namespace ddp {

struct WorkerAddress
{
	std::string host;
	int port;
	WorkerAddress() : host(), port(0) {}
	WorkerAddress(const std::string& h, int p) : host(h), port(p) {}
};

// Initialize DDP. Must be called before any other ddp:: function.
// rootAddr: hostname/IP of root worker (e.g. "192.168.1.100")
// rootPort: port root listens on for DDP traffic
// rank: this worker's rank (0 = root)
// worldSize: total number of workers
void init(const char* rootAddr, int rootPort, int rank, int worldSize);

// Explicit address list: addresses[i] is listen address for rank i.
// addresses[0] is root. worldSize = addresses.size().
void init(const std::vector<WorkerAddress>& addresses, int rank);
void finalize();

int worldSize();        // 1 if not initialized
int rank();             // 0 if not initialized
bool isRoot();          // rank() == 0

// Blocking AllReduce SUM in-place. All workers must call simultaneously.
void allReduceSumInPlace(float* data, size_t count);
void allReduceSumInPlace(unsigned int* data, size_t count);
void allReduceSumInPlace(double* data, size_t count);
void allReduceSumInPlace(unsigned long long* data, size_t count);

// Blocking barrier.
void barrier();

// Root broadcasts data to all workers. Workers receive into same buffer.
void broadcastFromRoot(float* data, size_t count);

// Compression configuration (call after init, before training).
// mode: 0=none (raw FP32), 1=FP16, 2=FP16+TopK
void setCompression(int mode);
void setTopKRatio(float ratio);
void setTopKWarmupSteps(int steps);

// Bucketed AllReduce: concatenates numBuffers gradient vectors into one
// flat buffer, performs a single compressed allReduce, then scatters
// results back into the original buffers.
// scalarBuf/scalarCount: small exact-precision values (e.g. timeStepsInBatch)
// that bypass compression and are always sent as raw float32.
void allReduceSumInPlaceBucketed(float** buffers, size_t* counts,
                                 int numBuffers,
                                 unsigned int* scalarBuf,
                                 size_t scalarCount);

} // namespace ddp
} // namespace glades

#endif
