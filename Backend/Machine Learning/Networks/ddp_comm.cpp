// DDP (Distributed Data Parallel) communication implementation.
//
// Centralized reduce through root (rank 0):
// - Root runs a GServer; non-root workers connect to it.
// - AllReduce: workers send gradients to root, root sums, root sends back.
// - Uses pthread_mutex/cond for blocking synchronization.
#include "ddp_comm.h"

#include "Backend/Networking/main.h"
#include "Backend/Networking/service.h"
#include "Backend/Networking/connection.h"
#include "Backend/Database/ServiceData.h"
#include "Backend/Database/GString.h"
#include "Backend/Database/GList.h"
#include "Backend/Database/GPointer.h"

#include <pthread.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <algorithm>
#include <vector>

// ============================================================
// File-static DDP state
// ============================================================
namespace {

static bool ddpInitialized = false;
static int ddpRank = 0;
static int ddpWorldSize = 1;
static GNet::GServer* ddpServer = NULL;

// Synchronization for blocking allreduce/barrier/broadcast.
static pthread_mutex_t ddpMutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t ddpCond = PTHREAD_COND_INITIALIZER;

// --- AllReduce state ---
// Root accumulation buffer (float). Root copies own data here first,
// then each arriving worker's data is summed in.
static std::vector<float> reduceAccumF;
// Result buffer for non-root workers (filled by DDPResultService).
static std::vector<float> resultBufF;
static unsigned int reduceWorkersArrived = 0;
static bool reduceResultReady = false; // for non-root: result delivered

// --- Barrier state ---
static unsigned int barrierWorkersArrived = 0;
static bool barrierReleased = false;

// --- Broadcast state ---
static std::vector<float> broadcastBuf;
static bool broadcastReady = false;

// --- Login tracking ---
// Root waits for all N-1 workers to connect before init() returns.
static unsigned int loginCount = 0;

// Connection list for root to send back results.
// Stored by DDPReduceService as workers arrive.
static std::vector<GNet::Connection*> workerConnections;

// --- Compression state ---
static int ddpCompressionMode = 0;   // 0=none, 1=FP16, 2=FP16+TopK
static float ddpTopKRatio = 0.01f;
static int ddpTopKWarmupSteps = 0;
static int ddpBucketedStepCount = 0;
static std::vector<float> ddpResidual;

static inline uint16_t float_to_fp16(float f)
{
	unsigned int x;
	memcpy(&x, &f, 4);
	unsigned int sign = (x >> 16) & 0x8000u;
	int exp = ((x >> 23) & 0xFF) - 127;
	unsigned int mant = x & 0x007FFFFFu;

	if (exp > 15)
		return static_cast<uint16_t>(sign | 0x7C00u);
	if (exp < -14)
	{
		if (exp < -24) return static_cast<uint16_t>(sign);
		mant |= 0x00800000u;
		int shift = -1 - exp;
		unsigned int half = 1u << (shift - 1 + 13);
		unsigned int rounded = (mant + half - 1 + ((mant >> (shift + 13)) & 1u)) >> (shift + 13);
		return static_cast<uint16_t>(sign | rounded);
	}
	unsigned int hexp = static_cast<unsigned int>(exp + 15) << 10;
	unsigned int rounded = mant + 0x00000FFFu + ((mant >> 13) & 1u);
	if (rounded & 0x00800000u)
	{
		rounded = 0;
		hexp += 0x0400u;
		if (hexp >= 0x7C00u) hexp = 0x7C00u;
	}
	return static_cast<uint16_t>(sign | hexp | (rounded >> 13));
}

static inline float fp16_to_float(uint16_t h)
{
	unsigned int sign = (static_cast<unsigned int>(h) & 0x8000u) << 16;
	unsigned int hexp = (h >> 10) & 0x1Fu;
	unsigned int mant = h & 0x03FFu;
	unsigned int result;
	if (hexp == 0)
	{
		if (mant == 0) { result = sign; }
		else
		{
			hexp = 1;
			while (!(mant & 0x0400u)) { mant <<= 1; hexp--; }
			mant &= 0x03FFu;
			result = sign | (static_cast<unsigned int>(127 - 15 + hexp) << 23) | (mant << 13);
		}
	}
	else if (hexp == 31) { result = sign | 0x7F800000u | (mant << 13); }
	else { result = sign | (static_cast<unsigned int>(hexp + 127 - 15) << 23) | (mant << 13); }
	float f;
	memcpy(&f, &result, 4);
	return f;
}

// ============================================================
// DDP Services (run inside GServer's service pool)
// ============================================================

// --- DDPReduceService (runs on root) ---
// Receives gradient buffer from a worker, adds to accumulation buffer.
class DDPReduceService : public GNet::Service
{
	GNet::GServer* srv;
public:
	DDPReduceService(GNet::GServer* s) : srv(s) {}

	shmea::ServiceData* execute(const shmea::ServiceData* cData)
	{
		const shmea::GString& payload = cData->getBinaryPayload();
		const unsigned int payloadSize = cData->getBinaryPayloadSize();

		pthread_mutex_lock(&ddpMutex);

		if (payloadSize >= 4)
		{
			const unsigned char* raw = reinterpret_cast<const unsigned char*>(payload.c_str());
			const unsigned char mode = raw[0];

			if (mode == 1)
			{
				const unsigned int fp16Count = (payloadSize - 4) / sizeof(uint16_t);
				const uint16_t* src = reinterpret_cast<const uint16_t*>(raw + 4);
				if (reduceAccumF.size() == static_cast<size_t>(fp16Count))
				{
					for (size_t i = 0; i < fp16Count; ++i)
						reduceAccumF[i] += fp16_to_float(src[i]);
				}
			}
			else if (mode == 2)
			{
				if (payloadSize >= 8)
				{
					unsigned int nnz;
					memcpy(&nnz, raw + 4, 4);
					const unsigned int expectedSize = 8 + nnz * 4 + nnz * 2;
					if (payloadSize >= expectedSize)
					{
						const uint32_t* indices = reinterpret_cast<const uint32_t*>(raw + 8);
						const uint16_t* values = reinterpret_cast<const uint16_t*>(raw + 8 + nnz * 4);
						for (unsigned int j = 0; j < nnz; ++j)
						{
							if (indices[j] < reduceAccumF.size())
								reduceAccumF[indices[j]] += fp16_to_float(values[j]);
						}
					}
				}
			}
			else
			{
				// Mode 0 or unknown: header(4) + float[count]
				const unsigned int floatCount = (payloadSize - 4) / sizeof(float);
				const float* src = reinterpret_cast<const float*>(raw + 4);
				if (reduceAccumF.size() == static_cast<size_t>(floatCount))
				{
					for (size_t i = 0; i < floatCount; ++i)
						reduceAccumF[i] += src[i];
				}
			}
		}
		else
		{
			// Legacy: no header, raw floats
			const unsigned int floatCount = payloadSize / sizeof(float);
			const float* src = reinterpret_cast<const float*>(payload.c_str());
			if (reduceAccumF.size() == static_cast<size_t>(floatCount))
			{
				for (size_t i = 0; i < floatCount; ++i)
					reduceAccumF[i] += src[i];
			}
		}

		workerConnections.push_back(cData->getConnection());
		++reduceWorkersArrived;
		pthread_cond_broadcast(&ddpCond);
		pthread_mutex_unlock(&ddpMutex);
		return NULL;
	}

	shmea::GString getName() const { return "DDPReduce"; }
	GNet::Service* MakeService(GNet::GServer* s) const { return new DDPReduceService(s); }
};

// --- DDPResultService (runs on non-root) ---
// Receives the reduced gradient buffer from root.
class DDPResultService : public GNet::Service
{
	GNet::GServer* srv;
public:
	DDPResultService(GNet::GServer* s) : srv(s) {}

	shmea::ServiceData* execute(const shmea::ServiceData* cData)
	{
		const shmea::GString& payload = cData->getBinaryPayload();
		const unsigned int payloadSize = cData->getBinaryPayloadSize();

		pthread_mutex_lock(&ddpMutex);

		if (payloadSize >= 4)
		{
			const unsigned char* raw = reinterpret_cast<const unsigned char*>(payload.c_str());
			const unsigned char mode = raw[0];

			if (mode == 1 || mode == 2)
			{
				// FP16 dense result (result broadcast is always dense FP16 for modes 1 and 2)
				const unsigned int fp16Count = (payloadSize - 4) / sizeof(uint16_t);
				resultBufF.resize(fp16Count);
				const uint16_t* src = reinterpret_cast<const uint16_t*>(raw + 4);
				for (size_t i = 0; i < fp16Count; ++i)
					resultBufF[i] = fp16_to_float(src[i]);
			}
			else
			{
				// Mode 0: raw FP32 with header
				const unsigned int floatCount = (payloadSize - 4) / sizeof(float);
				resultBufF.resize(floatCount);
				if (floatCount > 0)
					memcpy(&resultBufF[0], raw + 4, floatCount * sizeof(float));
			}
		}
		else
		{
			// Legacy: no header
			const unsigned int floatCount = payloadSize / sizeof(float);
			resultBufF.resize(floatCount);
			if (floatCount > 0)
				memcpy(&resultBufF[0], payload.c_str(), payloadSize);
		}

		reduceResultReady = true;
		pthread_cond_broadcast(&ddpCond);
		pthread_mutex_unlock(&ddpMutex);
		return NULL;
	}

	shmea::GString getName() const { return "DDPResult"; }
	GNet::Service* MakeService(GNet::GServer* s) const { return new DDPResultService(s); }
};

// --- DDPBarrierService (runs on root) ---
// A worker reports arrival at a barrier.
class DDPBarrierService : public GNet::Service
{
	GNet::GServer* srv;
public:
	DDPBarrierService(GNet::GServer* s) : srv(s) {}

	shmea::ServiceData* execute(const shmea::ServiceData* cData)
	{
		pthread_mutex_lock(&ddpMutex);
		workerConnections.push_back(cData->getConnection());
		++barrierWorkersArrived;
		pthread_cond_broadcast(&ddpCond);
		pthread_mutex_unlock(&ddpMutex);
		return NULL;
	}

	shmea::GString getName() const { return "DDPBarrier"; }
	GNet::Service* MakeService(GNet::GServer* s) const { return new DDPBarrierService(s); }
};

// --- DDPBarrierReleaseService (runs on non-root) ---
// Root sends this to release the worker from the barrier.
class DDPBarrierReleaseService : public GNet::Service
{
	GNet::GServer* srv;
public:
	DDPBarrierReleaseService(GNet::GServer* s) : srv(s) {}

	shmea::ServiceData* execute(const shmea::ServiceData* cData)
	{
		(void)cData;
		pthread_mutex_lock(&ddpMutex);
		barrierReleased = true;
		pthread_cond_broadcast(&ddpCond);
		pthread_mutex_unlock(&ddpMutex);
		return NULL;
	}

	shmea::GString getName() const { return "DDPBarrierRelease"; }
	GNet::Service* MakeService(GNet::GServer* s) const { return new DDPBarrierReleaseService(s); }
};

// --- DDPBroadcastService (runs on non-root) ---
// Receives broadcast data from root.
class DDPBroadcastService : public GNet::Service
{
	GNet::GServer* srv;
public:
	DDPBroadcastService(GNet::GServer* s) : srv(s) {}

	shmea::ServiceData* execute(const shmea::ServiceData* cData)
	{
		const shmea::GString& payload = cData->getBinaryPayload();
		const unsigned int payloadSize = cData->getBinaryPayloadSize();
		const unsigned int floatCount = payloadSize / sizeof(float);

		pthread_mutex_lock(&ddpMutex);
		broadcastBuf.resize(floatCount);
		if (floatCount > 0)
			memcpy(&broadcastBuf[0], payload.c_str(), payloadSize);
		broadcastReady = true;
		pthread_cond_broadcast(&ddpCond);
		pthread_mutex_unlock(&ddpMutex);
		return NULL;
	}

	shmea::GString getName() const { return "DDPBroadcast"; }
	GNet::Service* MakeService(GNet::GServer* s) const { return new DDPBroadcastService(s); }
};

// --- DDPBroadcastReadyService (runs on root) ---
// Non-root workers signal they are ready to receive a broadcast.
class DDPBroadcastReadyService : public GNet::Service
{
	GNet::GServer* srv;
public:
	DDPBroadcastReadyService(GNet::GServer* s) : srv(s) {}

	shmea::ServiceData* execute(const shmea::ServiceData* cData)
	{
		pthread_mutex_lock(&ddpMutex);
		workerConnections.push_back(cData->getConnection());
		++barrierWorkersArrived; // reuse barrier counter for broadcast sync
		pthread_cond_broadcast(&ddpCond);
		pthread_mutex_unlock(&ddpMutex);
		return NULL;
	}

	shmea::GString getName() const { return "DDPBroadcastReady"; }
	GNet::Service* MakeService(GNet::GServer* s) const { return new DDPBroadcastReadyService(s); }
};

// --- Login listener for root to track connecting workers ---
class DDPLoginListener : public GNet::LoginListener
{
public:
	void onClientLogin(GNet::Connection* c)
	{
		(void)c;
		pthread_mutex_lock(&ddpMutex);
		++loginCount;
		pthread_cond_broadcast(&ddpCond);
		pthread_mutex_unlock(&ddpMutex);
	}
	void onServerLogin(GNet::Connection* c) { (void)c; }
};

} // anonymous namespace

// ============================================================
// Internal init helper
// ============================================================

static void initInternal(const char* rootHost, int rootPort,
                         int myListenPort, int rank, int worldSize)
{
	if (ddpInitialized)
		return;
	if (worldSize <= 1)
	{
		ddpRank = 0;
		ddpWorldSize = 1;
		ddpInitialized = true;
		return;
	}

	ddpRank = rank;
	ddpWorldSize = worldSize;

	ddpServer = new GNet::GServer();

	// Register services on all workers.
	ddpServer->addService(new DDPReduceService(ddpServer));
	ddpServer->addService(new DDPResultService(ddpServer));
	ddpServer->addService(new DDPBarrierService(ddpServer));
	ddpServer->addService(new DDPBarrierReleaseService(ddpServer));
	ddpServer->addService(new DDPBroadcastService(ddpServer));
	ddpServer->addService(new DDPBroadcastReadyService(ddpServer));

	char rootPortBuf[32];
	sprintf(rootPortBuf, "%d", rootPort);

	if (rank == 0)
	{
		// Root: set up login listener and start listening.
		ddpServer->setLoginListener(shmea::GPointer<GNet::LoginListener>(new DDPLoginListener()));
		char listenBuf[32];
		sprintf(listenBuf, "%d", myListenPort);
		ddpServer->run(shmea::GString(listenBuf), false);

		// Wait for all N-1 workers to connect.
		pthread_mutex_lock(&ddpMutex);
		while (loginCount < static_cast<unsigned int>(worldSize - 1))
			pthread_cond_wait(&ddpCond, &ddpMutex);
		pthread_mutex_unlock(&ddpMutex);
	}
	else
	{
		// Non-root: start local server (for receiving results) then connect to root.
		char workerPort[32];
		sprintf(workerPort, "%d", myListenPort);
		ddpServer->run(shmea::GString(workerPort), false);

		char workerName[64];
		sprintf(workerName, "ddp_worker_%d", rank);
		ddpServer->LaunchInstance(shmea::GString(rootHost),
		                          shmea::GString(rootPortBuf),
		                          shmea::GString(workerName));

		// Small delay to allow connection to establish.
		// (LaunchInstance is async; the handshake completes in the background.)
		struct timespec ts;
		ts.tv_sec = 0;
		ts.tv_nsec = 500 * 1000 * 1000; // 500ms
		nanosleep(&ts, NULL);
	}

	ddpInitialized = true;
}

// ============================================================
// Public API
// ============================================================

void glades::ddp::init(const char* rootAddr, int rootPort, int rank, int worldSize)
{
	int myPort = (rank == 0) ? rootPort : (rootPort + rank);
	initInternal(rootAddr, rootPort, myPort, rank, worldSize);
}

void glades::ddp::init(const std::vector<WorkerAddress>& addresses, int rank)
{
	int worldSize = (int)addresses.size();
	if (worldSize <= 0 || rank < 0 || rank >= worldSize)
	{
		// Degenerate: treat as no-op single-worker init.
		ddpRank = 0;
		ddpWorldSize = 1;
		ddpInitialized = true;
		return;
	}
	initInternal(addresses[0].host.c_str(), addresses[0].port,
	             addresses[rank].port, rank, worldSize);
}

void glades::ddp::finalize()
{
	if (!ddpInitialized)
		return;
	if (ddpServer)
	{
		ddpServer->stop();
		delete ddpServer;
		ddpServer = NULL;
	}
	ddpRank = 0;
	ddpWorldSize = 1;
	loginCount = 0;
	ddpCompressionMode = 0;
	ddpTopKRatio = 0.01f;
	ddpTopKWarmupSteps = 0;
	ddpBucketedStepCount = 0;
	ddpResidual.clear();
	ddpInitialized = false;
}

int glades::ddp::worldSize()
{
	return ddpWorldSize;
}

int glades::ddp::rank()
{
	return ddpRank;
}

bool glades::ddp::isRoot()
{
	return ddpRank == 0;
}

// ============================================================
// AllReduce SUM in-place (float)
// ============================================================
void glades::ddp::allReduceSumInPlace(float* data, size_t count)
{
	if (!ddpInitialized || ddpWorldSize <= 1 || count == 0)
		return;

	const unsigned int byteSize = static_cast<unsigned int>(count * sizeof(float));

	if (ddpRank == 0)
	{
		// Root: copy own data into accumulation buffer, wait for all workers.
		pthread_mutex_lock(&ddpMutex);
		reduceAccumF.resize(count);
		memcpy(&reduceAccumF[0], data, byteSize);
		reduceWorkersArrived = 0;
		workerConnections.clear();
		pthread_mutex_unlock(&ddpMutex);

		// Wait for all N-1 workers to send their gradients.
		pthread_mutex_lock(&ddpMutex);
		while (reduceWorkersArrived < static_cast<unsigned int>(ddpWorldSize - 1))
			pthread_cond_wait(&ddpCond, &ddpMutex);

		// Copy summed result back to caller.
		memcpy(data, &reduceAccumF[0], byteSize);

		// Send result to each non-root worker.
		for (size_t i = 0; i < workerConnections.size(); ++i)
		{
			shmea::ServiceData* sd = new shmea::ServiceData(workerConnections[i], "DDPResult");
			unsigned int resultSize = 4 + byteSize;
			std::vector<unsigned char> resultBuf(resultSize);
			resultBuf[0] = 0; resultBuf[1] = 0; resultBuf[2] = 0; resultBuf[3] = 0;
			memcpy(&resultBuf[4], &reduceAccumF[0], byteSize);
			sd->setBinaryPayload(reinterpret_cast<const char*>(&resultBuf[0]), resultSize);
			ddpServer->send(shmea::GPointer<shmea::ServiceData>(sd));
		}

		workerConnections.clear();
		reduceWorkersArrived = 0;
		pthread_mutex_unlock(&ddpMutex);
	}
	else
	{
		// Non-root: send our data to root, then wait for result.
		pthread_mutex_lock(&ddpMutex);
		reduceResultReady = false;
		pthread_mutex_unlock(&ddpMutex);

		// Find root connection (server connection established during init).
		GNet::Connection* rootConn = ddpServer->getConnectionFromName(shmea::GString(""));
		// Fallback: try to find any server connection.
		if (!rootConn)
		{
			char workerName[64];
			sprintf(workerName, "ddp_worker_%d", ddpRank);
			rootConn = ddpServer->getConnectionFromName(shmea::GString(workerName));
		}

		if (rootConn)
		{
			shmea::ServiceData* sd = new shmea::ServiceData(rootConn, "DDPReduce");
			unsigned int sendSize = 4 + byteSize;
			std::vector<unsigned char> sendBuf(sendSize);
			sendBuf[0] = 0; sendBuf[1] = 0; sendBuf[2] = 0; sendBuf[3] = 0;
			memcpy(&sendBuf[4], data, byteSize);
			sd->setBinaryPayload(reinterpret_cast<const char*>(&sendBuf[0]), sendSize);
			ddpServer->send(shmea::GPointer<shmea::ServiceData>(sd));
		}

		// Wait for result from root.
		pthread_mutex_lock(&ddpMutex);
		while (!reduceResultReady)
			pthread_cond_wait(&ddpCond, &ddpMutex);

		if (resultBufF.size() == count)
			memcpy(data, &resultBufF[0], byteSize);

		reduceResultReady = false;
		pthread_mutex_unlock(&ddpMutex);
	}
}

// ============================================================
// AllReduce SUM in-place (unsigned int)
// ============================================================
void glades::ddp::allReduceSumInPlace(unsigned int* data, size_t count)
{
	if (!ddpInitialized || ddpWorldSize <= 1 || count == 0)
		return;

	// Reuse float path by casting. unsigned int and float are both 4 bytes.
	// We treat the buffer as raw bytes for network transport, then do the
	// element-wise sum ourselves.
	//
	// Simple approach: convert to float, allreduce, convert back.
	// For small counts (typically 1 for timeStepsInBatch), this is fine.
	std::vector<float> fbuf(count);
	for (size_t i = 0; i < count; ++i)
		fbuf[i] = static_cast<float>(data[i]);
	allReduceSumInPlace(&fbuf[0], count);
	for (size_t i = 0; i < count; ++i)
		data[i] = static_cast<unsigned int>(fbuf[i] + 0.5f);
}

// ============================================================
// AllReduce SUM in-place (double)
// ============================================================
void glades::ddp::allReduceSumInPlace(double* data, size_t count)
{
	if (!ddpInitialized || ddpWorldSize <= 1 || count == 0)
		return;

	// Convert to float for transport. Precision loss is acceptable for
	// metrics aggregation (not used for gradient math).
	std::vector<float> fbuf(count);
	for (size_t i = 0; i < count; ++i)
		fbuf[i] = static_cast<float>(data[i]);
	allReduceSumInPlace(&fbuf[0], count);
	for (size_t i = 0; i < count; ++i)
		data[i] = static_cast<double>(fbuf[i]);
}

// ============================================================
// AllReduce SUM in-place (unsigned long long)
// ============================================================
void glades::ddp::allReduceSumInPlace(unsigned long long* data, size_t count)
{
	if (!ddpInitialized || ddpWorldSize <= 1 || count == 0)
		return;

	std::vector<float> fbuf(count);
	for (size_t i = 0; i < count; ++i)
		fbuf[i] = static_cast<float>(static_cast<double>(data[i]));
	allReduceSumInPlace(&fbuf[0], count);
	for (size_t i = 0; i < count; ++i)
		data[i] = static_cast<unsigned long long>(static_cast<double>(fbuf[i]) + 0.5);
}

// ============================================================
// Barrier
// ============================================================
void glades::ddp::barrier()
{
	if (!ddpInitialized || ddpWorldSize <= 1)
		return;

	if (ddpRank == 0)
	{
		// Root: wait for all N-1 workers to arrive.
		pthread_mutex_lock(&ddpMutex);
		barrierWorkersArrived = 0;
		workerConnections.clear();
		pthread_mutex_unlock(&ddpMutex);

		pthread_mutex_lock(&ddpMutex);
		while (barrierWorkersArrived < static_cast<unsigned int>(ddpWorldSize - 1))
			pthread_cond_wait(&ddpCond, &ddpMutex);

		// Release all workers.
		for (size_t i = 0; i < workerConnections.size(); ++i)
		{
			shmea::ServiceData* sd = new shmea::ServiceData(workerConnections[i], "DDPBarrierRelease");
			sd->set(shmea::GString("DDPBarrierRelease"));
			ddpServer->send(shmea::GPointer<shmea::ServiceData>(sd));
		}
		workerConnections.clear();
		barrierWorkersArrived = 0;
		pthread_mutex_unlock(&ddpMutex);
	}
	else
	{
		pthread_mutex_lock(&ddpMutex);
		barrierReleased = false;
		pthread_mutex_unlock(&ddpMutex);

		// Send barrier arrival to root.
		GNet::Connection* rootConn = ddpServer->getConnectionFromName(shmea::GString(""));
		if (!rootConn)
		{
			char workerName[64];
			sprintf(workerName, "ddp_worker_%d", ddpRank);
			rootConn = ddpServer->getConnectionFromName(shmea::GString(workerName));
		}

		if (rootConn)
		{
			shmea::ServiceData* sd = new shmea::ServiceData(rootConn, "DDPBarrier");
			sd->set(shmea::GString("DDPBarrier"));
			ddpServer->send(shmea::GPointer<shmea::ServiceData>(sd));
		}

		// Wait for release.
		pthread_mutex_lock(&ddpMutex);
		while (!barrierReleased)
			pthread_cond_wait(&ddpCond, &ddpMutex);
		barrierReleased = false;
		pthread_mutex_unlock(&ddpMutex);
	}
}

// ============================================================
// Broadcast from root
// ============================================================
void glades::ddp::broadcastFromRoot(float* data, size_t count)
{
	if (!ddpInitialized || ddpWorldSize <= 1 || count == 0)
		return;

	const unsigned int byteSize = static_cast<unsigned int>(count * sizeof(float));

	if (ddpRank == 0)
	{
		// Root: wait for all workers to signal ready, then send data.
		pthread_mutex_lock(&ddpMutex);
		barrierWorkersArrived = 0;
		workerConnections.clear();
		pthread_mutex_unlock(&ddpMutex);

		pthread_mutex_lock(&ddpMutex);
		while (barrierWorkersArrived < static_cast<unsigned int>(ddpWorldSize - 1))
			pthread_cond_wait(&ddpCond, &ddpMutex);

		for (size_t i = 0; i < workerConnections.size(); ++i)
		{
			shmea::ServiceData* sd = new shmea::ServiceData(workerConnections[i], "DDPBroadcast");
			sd->setBinaryPayload(reinterpret_cast<const char*>(data), byteSize);
			ddpServer->send(shmea::GPointer<shmea::ServiceData>(sd));
		}
		workerConnections.clear();
		barrierWorkersArrived = 0;
		pthread_mutex_unlock(&ddpMutex);
	}
	else
	{
		pthread_mutex_lock(&ddpMutex);
		broadcastReady = false;
		pthread_mutex_unlock(&ddpMutex);

		// Signal root we are ready.
		GNet::Connection* rootConn = ddpServer->getConnectionFromName(shmea::GString(""));
		if (!rootConn)
		{
			char workerName[64];
			sprintf(workerName, "ddp_worker_%d", ddpRank);
			rootConn = ddpServer->getConnectionFromName(shmea::GString(workerName));
		}

		if (rootConn)
		{
			shmea::ServiceData* sd = new shmea::ServiceData(rootConn, "DDPBroadcastReady");
			sd->set(shmea::GString("DDPBroadcastReady"));
			ddpServer->send(shmea::GPointer<shmea::ServiceData>(sd));
		}

		// Wait for broadcast data.
		pthread_mutex_lock(&ddpMutex);
		while (!broadcastReady)
			pthread_cond_wait(&ddpCond, &ddpMutex);

		if (broadcastBuf.size() == count)
			memcpy(data, &broadcastBuf[0], byteSize);

		broadcastReady = false;
		pthread_mutex_unlock(&ddpMutex);
	}
}

// ============================================================
// Compressed AllReduce (modes 0, 1, 2)
// ============================================================
static void allReduceCompressed(float* data, size_t count, int effectiveMode)
{
	if (!ddpInitialized || ddpWorldSize <= 1 || count == 0)
		return;

	if (ddpRank == 0)
	{
		// === ROOT ===
		pthread_mutex_lock(&ddpMutex);
		reduceAccumF.resize(count);

		if (effectiveMode == 2)
		{
			// Top-K: root sparsifies its own contribution.
			if (ddpResidual.size() != count)
				ddpResidual.assign(count, 0.0f);
			for (size_t i = 0; i < count; ++i)
				data[i] += ddpResidual[i];

			size_t k = static_cast<size_t>(static_cast<float>(count) * ddpTopKRatio);
			if (k == 0) k = 1;
			if (k > count) k = count;

			std::vector<float> absVals(count);
			for (size_t i = 0; i < count; ++i)
				absVals[i] = (data[i] >= 0.0f) ? data[i] : -data[i];

			std::vector<float> absSort(absVals);
			std::nth_element(absSort.begin(), absSort.begin() + static_cast<long>(count - k), absSort.end());
			float threshold = absSort[count - k];

			memset(&reduceAccumF[0], 0, count * sizeof(float));
			for (size_t i = 0; i < count; ++i)
			{
				if (absVals[i] >= threshold)
				{
					reduceAccumF[i] = data[i];
					ddpResidual[i] = 0.0f;
				}
				else
				{
					ddpResidual[i] = data[i];
				}
			}
		}
		else
		{
			memcpy(&reduceAccumF[0], data, count * sizeof(float));
		}

		reduceWorkersArrived = 0;
		workerConnections.clear();
		pthread_mutex_unlock(&ddpMutex);

		// Wait for all workers.
		pthread_mutex_lock(&ddpMutex);
		while (reduceWorkersArrived < static_cast<unsigned int>(ddpWorldSize - 1))
			pthread_cond_wait(&ddpCond, &ddpMutex);

		// Copy result locally.
		memcpy(data, &reduceAccumF[0], count * sizeof(float));

		// Send result to workers.
		if (effectiveMode >= 1)
		{
			// Dense FP16 result.
			const unsigned int resultSize = 4 + static_cast<unsigned int>(count) * 2;
			std::vector<unsigned char> resultBuf(resultSize);
			resultBuf[0] = static_cast<unsigned char>(effectiveMode);
			resultBuf[1] = 0; resultBuf[2] = 0; resultBuf[3] = 0;
			uint16_t* dst = reinterpret_cast<uint16_t*>(&resultBuf[4]);
			for (size_t i = 0; i < count; ++i)
				dst[i] = float_to_fp16(reduceAccumF[i]);

			for (size_t i = 0; i < workerConnections.size(); ++i)
			{
				shmea::ServiceData* sd = new shmea::ServiceData(workerConnections[i], "DDPResult");
				sd->setBinaryPayload(reinterpret_cast<const char*>(&resultBuf[0]), resultSize);
				ddpServer->send(shmea::GPointer<shmea::ServiceData>(sd));
			}
		}
		else
		{
			// Mode 0: raw FP32 with header.
			const unsigned int resultSize = 4 + static_cast<unsigned int>(count) * 4;
			std::vector<unsigned char> resultBuf(resultSize);
			resultBuf[0] = 0; resultBuf[1] = 0; resultBuf[2] = 0; resultBuf[3] = 0;
			memcpy(&resultBuf[4], &reduceAccumF[0], count * sizeof(float));

			for (size_t i = 0; i < workerConnections.size(); ++i)
			{
				shmea::ServiceData* sd = new shmea::ServiceData(workerConnections[i], "DDPResult");
				sd->setBinaryPayload(reinterpret_cast<const char*>(&resultBuf[0]), resultSize);
				ddpServer->send(shmea::GPointer<shmea::ServiceData>(sd));
			}
		}

		workerConnections.clear();
		reduceWorkersArrived = 0;
		pthread_mutex_unlock(&ddpMutex);
	}
	else
	{
		// === NON-ROOT WORKER ===
		pthread_mutex_lock(&ddpMutex);
		reduceResultReady = false;
		pthread_mutex_unlock(&ddpMutex);

		GNet::Connection* rootConn = ddpServer->getConnectionFromName(shmea::GString(""));
		if (!rootConn)
		{
			char workerName[64];
			sprintf(workerName, "ddp_worker_%d", ddpRank);
			rootConn = ddpServer->getConnectionFromName(shmea::GString(workerName));
		}

		if (rootConn)
		{
			if (effectiveMode == 2)
			{
				// Top-K sparse send.
				if (ddpResidual.size() != count)
					ddpResidual.assign(count, 0.0f);
				for (size_t i = 0; i < count; ++i)
					data[i] += ddpResidual[i];

				size_t k = static_cast<size_t>(static_cast<float>(count) * ddpTopKRatio);
				if (k == 0) k = 1;
				if (k > count) k = count;

				std::vector<float> absVals(count);
				for (size_t i = 0; i < count; ++i)
					absVals[i] = (data[i] >= 0.0f) ? data[i] : -data[i];

				std::vector<float> absSort(absVals);
				std::nth_element(absSort.begin(), absSort.begin() + static_cast<long>(count - k), absSort.end());
				float threshold = absSort[count - k];

				std::vector<uint32_t> indices;
				std::vector<uint16_t> values;
				indices.reserve(k);
				values.reserve(k);

				for (size_t i = 0; i < count; ++i)
				{
					if (absVals[i] >= threshold)
					{
						indices.push_back(static_cast<uint32_t>(i));
						values.push_back(float_to_fp16(data[i]));
						ddpResidual[i] = 0.0f;
					}
					else
					{
						ddpResidual[i] = data[i];
					}
				}

				uint32_t nnz = static_cast<uint32_t>(indices.size());
				unsigned int sendSize = 4 + 4 + nnz * 4 + nnz * 2;
				std::vector<unsigned char> sendBuf(sendSize);
				sendBuf[0] = 2; sendBuf[1] = 0; sendBuf[2] = 0; sendBuf[3] = 0;
				memcpy(&sendBuf[4], &nnz, 4);
				if (nnz > 0)
				{
					memcpy(&sendBuf[8], &indices[0], nnz * 4);
					memcpy(&sendBuf[8 + nnz * 4], &values[0], nnz * 2);
				}

				shmea::ServiceData* sd = new shmea::ServiceData(rootConn, "DDPReduce");
				sd->setBinaryPayload(reinterpret_cast<const char*>(&sendBuf[0]), sendSize);
				ddpServer->send(shmea::GPointer<shmea::ServiceData>(sd));
			}
			else if (effectiveMode == 1)
			{
				// FP16 dense send.
				unsigned int sendSize = 4 + static_cast<unsigned int>(count) * 2;
				std::vector<unsigned char> sendBuf(sendSize);
				sendBuf[0] = 1; sendBuf[1] = 0; sendBuf[2] = 0; sendBuf[3] = 0;
				uint16_t* dst = reinterpret_cast<uint16_t*>(&sendBuf[4]);
				for (size_t i = 0; i < count; ++i)
					dst[i] = float_to_fp16(data[i]);

				shmea::ServiceData* sd = new shmea::ServiceData(rootConn, "DDPReduce");
				sd->setBinaryPayload(reinterpret_cast<const char*>(&sendBuf[0]), sendSize);
				ddpServer->send(shmea::GPointer<shmea::ServiceData>(sd));
			}
			else
			{
				// Mode 0: raw FP32 with header.
				unsigned int sendSize = 4 + static_cast<unsigned int>(count) * 4;
				std::vector<unsigned char> sendBuf(sendSize);
				sendBuf[0] = 0; sendBuf[1] = 0; sendBuf[2] = 0; sendBuf[3] = 0;
				memcpy(&sendBuf[4], data, count * sizeof(float));

				shmea::ServiceData* sd = new shmea::ServiceData(rootConn, "DDPReduce");
				sd->setBinaryPayload(reinterpret_cast<const char*>(&sendBuf[0]), sendSize);
				ddpServer->send(shmea::GPointer<shmea::ServiceData>(sd));
			}
		}

		// Wait for result.
		pthread_mutex_lock(&ddpMutex);
		while (!reduceResultReady)
			pthread_cond_wait(&ddpCond, &ddpMutex);

		if (resultBufF.size() == count)
			memcpy(data, &resultBufF[0], count * sizeof(float));

		reduceResultReady = false;
		pthread_mutex_unlock(&ddpMutex);
	}
}

// ============================================================
// Bucketed AllReduce (compressed)
// ============================================================
void glades::ddp::allReduceSumInPlaceBucketed(float** buffers, size_t* counts,
                                              int numBuffers,
                                              unsigned int* scalarBuf,
                                              size_t scalarCount)
{
	if (!ddpInitialized || ddpWorldSize <= 1)
		return;
	if (numBuffers <= 0 && scalarCount == 0)
		return;

	size_t totalCount = 0;
	for (int i = 0; i < numBuffers; ++i)
		totalCount += counts[i];

	if (totalCount == 0 && scalarCount == 0)
		return;

	int effectiveMode = ddpCompressionMode;
	if (effectiveMode == 2 && ddpBucketedStepCount < ddpTopKWarmupSteps)
		effectiveMode = 1;

	// Flatten.
	std::vector<float> flat(totalCount);
	{
		size_t offset = 0;
		for (int i = 0; i < numBuffers; ++i)
		{
			if (counts[i] > 0)
				memcpy(&flat[offset], buffers[i], counts[i] * sizeof(float));
			offset += counts[i];
		}
	}

	// Dispatch.
	if (effectiveMode == 0)
		allReduceSumInPlace(&flat[0], totalCount);
	else
		allReduceCompressed(&flat[0], totalCount, effectiveMode);

	// Scatter back.
	{
		size_t offset = 0;
		for (int i = 0; i < numBuffers; ++i)
		{
			if (counts[i] > 0)
				memcpy(buffers[i], &flat[offset], counts[i] * sizeof(float));
			offset += counts[i];
		}
	}

	// Scalars always raw FP32.
	if (scalarCount > 0 && scalarBuf)
		allReduceSumInPlace(scalarBuf, scalarCount);

	++ddpBucketedStepCount;
}

// ============================================================
// Compression configuration
// ============================================================
void glades::ddp::setCompression(int mode)
{
	ddpCompressionMode = mode;
}

void glades::ddp::setTopKRatio(float ratio)
{
	if (ratio > 0.0f && ratio <= 1.0f)
		ddpTopKRatio = ratio;
}

void glades::ddp::setTopKWarmupSteps(int steps)
{
	ddpTopKWarmupSteps = steps;
}
