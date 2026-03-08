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

#ifdef _WIN32
#include "Backend/Core/platform.h"
#else
#include <pthread.h>
#endif
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
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
#ifdef _WIN32
static CRITICAL_SECTION ddpMutex;
static CONDITION_VARIABLE ddpCond;
static bool ddpSyncInit = false;
static void ensureDdpSync()
{
	if (!ddpSyncInit) { InitializeCriticalSection(&ddpMutex); InitializeConditionVariable(&ddpCond); ddpSyncInit = true; }
}
#define DDP_LOCK()   do { ensureDdpSync(); EnterCriticalSection(&ddpMutex); } while(0)
#define DDP_UNLOCK() LeaveCriticalSection(&ddpMutex)
#define DDP_WAIT()   SleepConditionVariableCS(&ddpCond, &ddpMutex, INFINITE)
#define DDP_BROADCAST() WakeAllConditionVariable(&ddpCond)
#else
static pthread_mutex_t ddpMutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t ddpCond = PTHREAD_COND_INITIALIZER;
#define DDP_LOCK()   pthread_mutex_lock(&ddpMutex)
#define DDP_UNLOCK() pthread_mutex_unlock(&ddpMutex)
#define DDP_WAIT()   pthread_cond_wait(&ddpCond, &ddpMutex)
#define DDP_BROADCAST() pthread_cond_broadcast(&ddpCond)
#endif

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
		const unsigned int floatCount = payloadSize / sizeof(float);

		DDP_LOCK();

		// Sum into accumulation buffer.
		if (reduceAccumF.size() == static_cast<size_t>(floatCount))
		{
			const float* src = reinterpret_cast<const float*>(payload.c_str());
			for (size_t i = 0; i < floatCount; ++i)
				reduceAccumF[i] += src[i];
		}

		// Track which worker sent this so we can send back results.
		workerConnections.push_back(cData->getConnection());

		++reduceWorkersArrived;
		DDP_BROADCAST();
		DDP_UNLOCK();

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
		const unsigned int floatCount = payloadSize / sizeof(float);

		DDP_LOCK();
		resultBufF.resize(floatCount);
		if (floatCount > 0)
			memcpy(&resultBufF[0], payload.c_str(), payloadSize);
		reduceResultReady = true;
		DDP_BROADCAST();
		DDP_UNLOCK();

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
		DDP_LOCK();
		workerConnections.push_back(cData->getConnection());
		++barrierWorkersArrived;
		DDP_BROADCAST();
		DDP_UNLOCK();
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
		DDP_LOCK();
		barrierReleased = true;
		DDP_BROADCAST();
		DDP_UNLOCK();
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

		DDP_LOCK();
		broadcastBuf.resize(floatCount);
		if (floatCount > 0)
			memcpy(&broadcastBuf[0], payload.c_str(), payloadSize);
		broadcastReady = true;
		DDP_BROADCAST();
		DDP_UNLOCK();
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
		DDP_LOCK();
		workerConnections.push_back(cData->getConnection());
		++barrierWorkersArrived; // reuse barrier counter for broadcast sync
		DDP_BROADCAST();
		DDP_UNLOCK();
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
		DDP_LOCK();
		++loginCount;
		DDP_BROADCAST();
		DDP_UNLOCK();
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
		DDP_LOCK();
		while (loginCount < static_cast<unsigned int>(worldSize - 1))
			DDP_WAIT();
		DDP_UNLOCK();
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
#ifdef _WIN32
		Sleep(500);
#else
		struct timespec ts;
		ts.tv_sec = 0;
		ts.tv_nsec = 500 * 1000 * 1000; // 500ms
		nanosleep(&ts, NULL);
#endif
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
		DDP_LOCK();
		reduceAccumF.resize(count);
		memcpy(&reduceAccumF[0], data, byteSize);
		reduceWorkersArrived = 0;
		workerConnections.clear();
		DDP_UNLOCK();

		// Wait for all N-1 workers to send their gradients.
		DDP_LOCK();
		while (reduceWorkersArrived < static_cast<unsigned int>(ddpWorldSize - 1))
			DDP_WAIT();

		// Copy summed result back to caller.
		memcpy(data, &reduceAccumF[0], byteSize);

		// Send result to each non-root worker.
		for (size_t i = 0; i < workerConnections.size(); ++i)
		{
			shmea::ServiceData* sd = new shmea::ServiceData(workerConnections[i], "DDPResult");
			sd->setBinaryPayload(reinterpret_cast<const char*>(&reduceAccumF[0]), byteSize);
			ddpServer->send(shmea::GPointer<shmea::ServiceData>(sd));
		}

		workerConnections.clear();
		reduceWorkersArrived = 0;
		DDP_UNLOCK();
	}
	else
	{
		// Non-root: send our data to root, then wait for result.
		DDP_LOCK();
		reduceResultReady = false;
		DDP_UNLOCK();

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
			sd->setBinaryPayload(reinterpret_cast<const char*>(data), byteSize);
			ddpServer->send(shmea::GPointer<shmea::ServiceData>(sd));
		}

		// Wait for result from root.
		DDP_LOCK();
		while (!reduceResultReady)
			DDP_WAIT();

		if (resultBufF.size() == count)
			memcpy(data, &resultBufF[0], byteSize);

		reduceResultReady = false;
		DDP_UNLOCK();
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
		DDP_LOCK();
		barrierWorkersArrived = 0;
		workerConnections.clear();
		DDP_UNLOCK();

		DDP_LOCK();
		while (barrierWorkersArrived < static_cast<unsigned int>(ddpWorldSize - 1))
			DDP_WAIT();

		// Release all workers.
		for (size_t i = 0; i < workerConnections.size(); ++i)
		{
			shmea::ServiceData* sd = new shmea::ServiceData(workerConnections[i], "DDPBarrierRelease");
			sd->set(shmea::GString("DDPBarrierRelease"));
			ddpServer->send(shmea::GPointer<shmea::ServiceData>(sd));
		}
		workerConnections.clear();
		barrierWorkersArrived = 0;
		DDP_UNLOCK();
	}
	else
	{
		DDP_LOCK();
		barrierReleased = false;
		DDP_UNLOCK();

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
		DDP_LOCK();
		while (!barrierReleased)
			DDP_WAIT();
		barrierReleased = false;
		DDP_UNLOCK();
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
		DDP_LOCK();
		barrierWorkersArrived = 0;
		workerConnections.clear();
		DDP_UNLOCK();

		DDP_LOCK();
		while (barrierWorkersArrived < static_cast<unsigned int>(ddpWorldSize - 1))
			DDP_WAIT();

		for (size_t i = 0; i < workerConnections.size(); ++i)
		{
			shmea::ServiceData* sd = new shmea::ServiceData(workerConnections[i], "DDPBroadcast");
			sd->setBinaryPayload(reinterpret_cast<const char*>(data), byteSize);
			ddpServer->send(shmea::GPointer<shmea::ServiceData>(sd));
		}
		workerConnections.clear();
		barrierWorkersArrived = 0;
		DDP_UNLOCK();
	}
	else
	{
		DDP_LOCK();
		broadcastReady = false;
		DDP_UNLOCK();

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
		DDP_LOCK();
		while (!broadcastReady)
			DDP_WAIT();

		if (broadcastBuf.size() == count)
			memcpy(data, &broadcastBuf[0], byteSize);

		broadcastReady = false;
		DDP_UNLOCK();
	}
}
