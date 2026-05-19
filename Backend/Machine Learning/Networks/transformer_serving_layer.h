// Serving layer for Transformer token-LM generation.
//
// This is a thin "production-ish" scheduler built on top of:
//   NNetwork::TransformerServeBatcher
//
// Goals:
// - A stable "serving-grade" API on top of TransformerServeBatcher
// - Continuous micro-batching into a fixed-capacity batcher
// - Streaming via polling (popNewTokens) and/or callbacks (onToken)
// - Designed to be driven by an external server/event loop (pre-C++11 compatible)
//
// Non-goals:
// - Distributed serving, GPU scheduling, network RPC, tokenization
//
// Thread-safety:
// - This layer is internally synchronized: all public APIs are safe to call concurrently from
//   multiple threads.
// - `step()` drives the batcher forward and invokes user callbacks on the caller's thread.
// - Internal state mutation is serialized under the layer mutex, but user callbacks execute
//   after the serving layer releases that mutex. Callbacks may call other public APIs, but
//   must not call `step()` re-entrantly.
//
// Copyright 2026
//
#pragma once

#include "network.h"

#include <deque>
#include <map>
#include <stdint.h>
#include <vector>

namespace glades {

// Callbacks for the serving layer, keyed by requestId (not slot index).
class ITransformerServingCallbacks
{
public:
	virtual ~ITransformerServingCallbacks() {}

	// Called after a token is emitted for a request (on the serving worker thread).
	// Return true to stop that request early.
	virtual bool onToken(uint64_t /*requestId*/,
	                     const NNetwork& /*net*/,
	                     unsigned int /*tokenId*/,
	                     unsigned int /*generatedIndex*/)
	{
		return false;
	}

	// Polled before sampling for a request each step (on the serving worker thread).
	// Return true to cancel the request.
	virtual bool shouldCancel(uint64_t /*requestId*/, const NNetwork& /*net*/) { return false; }
};

class TransformerServingLayer
{
public:
	typedef shmea::GPointer<ITransformerServingCallbacks> CallbackHandle;

	struct Config
	{
		// Batcher capacity.
		unsigned int maxBatchSize;
		// Max KV cache length per request.
		unsigned int maxSeqLen;

		// Queue/backpressure:
		// - submit() fails if pending queue would exceed this.
		// - 0 => unlimited (not recommended for production).
		unsigned int maxPendingRequests;

		// Security/hygiene: wipe KV prefix when removing a slot.
		bool wipeKvOnRemove;

		// RNG seed for the batcher stream (0 => derive from network seed).
		uint64_t rngSeed;

		// If true, finished requests are removed from the batcher immediately.
		// Their final results remain available via snapshots until explicitly cleared.
		bool autoRemoveFinished;

		// Structured logs (best-effort) using net.getLogger().
		bool enableLogs;

		Config()
		    : maxBatchSize(0u),
		      maxSeqLen(0u),
		      maxPendingRequests(0u),
		      wipeKvOnRemove(false),
		      rngSeed(0ULL),
		      autoRemoveFinished(true),
		      enableLogs(true)
		{
		}
	};

	struct RequestSnapshot
	{
		uint64_t requestId;
		bool done;
		NNetworkStatus status; // OK if successful or cancelled-by-callback; INTERNAL/INVALID_* on failure.
		NNetwork::TransformerGenerateResult result; // includes stop flags + tokens (as accumulated by this layer)
		uint64_t submittedAtUs;
		uint64_t admittedAtUs;
		uint64_t completedAtUs;
		uint64_t queueWaitUs;
		uint64_t serviceTimeUs;
		uint64_t endToEndTimeUs;
		unsigned int promptTokenCount;
		unsigned int generatedTokenCount;

		// Number of tokens already delivered through popNewTokens().
		unsigned int streamedTokenCount;

		RequestSnapshot()
		    : requestId(0ULL),
		      done(false),
		      status(NNetworkStatus::OK, std::string()),
		      result(),
		      submittedAtUs(0ULL),
		      admittedAtUs(0ULL),
		      completedAtUs(0ULL),
		      queueWaitUs(0ULL),
		      serviceTimeUs(0ULL),
		      endToEndTimeUs(0ULL),
		      promptTokenCount(0u),
		      generatedTokenCount(0u),
		      streamedTokenCount(0u)
		{
		}
	};

	struct Diagnostics
	{
		bool running;
		bool stopRequested;
		bool inStep;
		unsigned int maxBatchSize;
		unsigned int maxSeqLen;
		unsigned int maxPendingRequests;
		unsigned int pendingRequests;
		unsigned int activeRequests;
		unsigned int doneSnapshots;
		unsigned int snapshotCount;
		unsigned int completedCallbackCount;
		unsigned int peakPendingRequests;
		unsigned int peakActiveRequests;
		unsigned int peakDoneSnapshots;
		uint64_t nextRequestId;
		uint64_t startTimeUs;
		uint64_t uptimeUs;
		uint64_t totalSubmitted;
		uint64_t totalSubmitRejected;
		uint64_t totalBackpressureRejected;
		uint64_t totalAdmitted;
		uint64_t totalCompleted;
		uint64_t totalCompletedSuccess;
		uint64_t totalCompletedCancelled;
		uint64_t totalCompletedFailed;
		uint64_t totalPendingCancels;
		uint64_t totalLiveCancelRequests;
		uint64_t totalCallbackStops;
		uint64_t totalLimitStops;
		uint64_t totalStopTokenStops;
		uint64_t totalEosStops;
		uint64_t totalPromptTokensSubmitted;
		uint64_t totalPromptTokensAdmitted;
		uint64_t totalGeneratedTokens;
		uint64_t totalStepCalls;
		uint64_t totalIdleSteps;
		uint64_t totalAdmitFailures;
		uint64_t totalStepFailures;
		uint64_t totalCallbackExceptions;
		uint64_t totalReentrantStepRejected;
		uint64_t totalSnapshotClears;
		uint64_t totalQueueWaitUs;
		uint64_t totalServiceTimeUs;
		uint64_t totalEndToEndTimeUs;
		uint64_t totalStepDurationUs;
		uint64_t lastQueueWaitUs;
		uint64_t lastServiceTimeUs;
		uint64_t lastEndToEndTimeUs;
		uint64_t lastStepDurationUs;
		uint64_t maxQueueWaitUs;
		uint64_t maxServiceTimeUs;
		uint64_t maxEndToEndTimeUs;
		uint64_t maxStepDurationUs;
		uint64_t recentRequestLatencyP50Us;
		uint64_t recentRequestLatencyP99Us;
		uint64_t recentRequestLatencyP999Us;
		uint64_t recentStepDurationP50Us;
		uint64_t recentStepDurationP99Us;
		uint64_t recentStepDurationP999Us;
		float submittedPerSec;
		float admittedPerSec;
		float completedPerSec;
		float generatedTokensPerSec;
		uint64_t lastFailureRequestId;
		unsigned int lastFailureSlot;
		NNetworkStatus lastFailureStatus;
		NNetworkStatus lastStepStatus;

		Diagnostics()
		    : running(false),
		      stopRequested(false),
		      inStep(false),
		      maxBatchSize(0u),
		      maxSeqLen(0u),
		      maxPendingRequests(0u),
		      pendingRequests(0u),
		      activeRequests(0u),
		      doneSnapshots(0u),
		      snapshotCount(0u),
		      completedCallbackCount(0u),
		      peakPendingRequests(0u),
		      peakActiveRequests(0u),
		      peakDoneSnapshots(0u),
		      nextRequestId(0ULL),
		      startTimeUs(0ULL),
		      uptimeUs(0ULL),
		      totalSubmitted(0ULL),
		      totalSubmitRejected(0ULL),
		      totalBackpressureRejected(0ULL),
		      totalAdmitted(0ULL),
		      totalCompleted(0ULL),
		      totalCompletedSuccess(0ULL),
		      totalCompletedCancelled(0ULL),
		      totalCompletedFailed(0ULL),
		      totalPendingCancels(0ULL),
		      totalLiveCancelRequests(0ULL),
		      totalCallbackStops(0ULL),
		      totalLimitStops(0ULL),
		      totalStopTokenStops(0ULL),
		      totalEosStops(0ULL),
		      totalPromptTokensSubmitted(0ULL),
		      totalPromptTokensAdmitted(0ULL),
		      totalGeneratedTokens(0ULL),
		      totalStepCalls(0ULL),
		      totalIdleSteps(0ULL),
		      totalAdmitFailures(0ULL),
		      totalStepFailures(0ULL),
		      totalCallbackExceptions(0ULL),
		      totalReentrantStepRejected(0ULL),
		      totalSnapshotClears(0ULL),
		      totalQueueWaitUs(0ULL),
		      totalServiceTimeUs(0ULL),
		      totalEndToEndTimeUs(0ULL),
		      totalStepDurationUs(0ULL),
		      lastQueueWaitUs(0ULL),
		      lastServiceTimeUs(0ULL),
		      lastEndToEndTimeUs(0ULL),
		      lastStepDurationUs(0ULL),
		      maxQueueWaitUs(0ULL),
		      maxServiceTimeUs(0ULL),
		      maxEndToEndTimeUs(0ULL),
		      maxStepDurationUs(0ULL),
		      recentRequestLatencyP50Us(0ULL),
		      recentRequestLatencyP99Us(0ULL),
		      recentRequestLatencyP999Us(0ULL),
		      recentStepDurationP50Us(0ULL),
		      recentStepDurationP99Us(0ULL),
		      recentStepDurationP999Us(0ULL),
		      submittedPerSec(0.0f),
		      admittedPerSec(0.0f),
		      completedPerSec(0.0f),
		      generatedTokensPerSec(0.0f),
		      lastFailureRequestId(0ULL),
		      lastFailureSlot(static_cast<unsigned int>(-1)),
		      lastFailureStatus(NNetworkStatus::OK, std::string()),
		      lastStepStatus(NNetworkStatus::OK, std::string())
		{
		}
	};

	TransformerServingLayer();
	~TransformerServingLayer();

	// Initialize/reset the serving layer.
	// The referenced `net` must outlive this serving layer and any in-flight callbacks.
	NNetworkStatus start(const NNetwork& net, const Config& cfg);

	// Stop serving (clears pending/live state; keeps snapshots for inspection unless cleared explicitly).
	void stop();

	bool isRunning() const;

	// Drive serving forward by one "global append step" across active requests.
	// Call this in a loop from your server/event loop.
	//
	// Returns:
	// - OK: step completed (some progress may or may not have happened)
	// - error: fatal internal error from the underlying batcher implementation
	NNetworkStatus step();

	// Submit a request. Returns a requestId that can be used for polling/streaming/cancel.
	// If callbacks is non-null, the serving layer takes shared ownership of a heap-allocated
	// callback object and retains it until the request snapshot is cleared or the layer stops.
	// Callbacks may be invoked from step() on the caller's thread.
	NNetworkStatus submit(const NNetwork::TransformerServeRequest& req,
	                      uint64_t& outRequestId,
	                      CallbackHandle callbacks = CallbackHandle());

	// Request cancellation. Best-effort: takes effect on the next decode step.
	// Returns false if requestId not found (already done/removed or never existed).
	bool cancel(uint64_t requestId);

	// Get a snapshot of a request's current state. Returns false if not found.
	bool getSnapshot(uint64_t requestId, RequestSnapshot& out) const;

	// Pop tokens generated since the last pop for this request.
	// Returns false if requestId not found.
	bool popNewTokens(uint64_t requestId, std::vector<unsigned int>& outNewTokens, bool& outDone, NNetworkStatus& outStatus);

	// Forget a completed request snapshot (does not affect the model/batcher).
	// Returns false if requestId not found or the request is still pending/live.
	bool clearSnapshot(uint64_t requestId);

	// Query current serving state and cumulative counters for the current layer run.
	bool getDiagnostics(Diagnostics& out) const;

private:
	// Non-copyable (C++98 style).
	TransformerServingLayer(const TransformerServingLayer&);
	TransformerServingLayer& operator=(const TransformerServingLayer&);

	// Internal mutex used to synchronize all public APIs.
	// Implemented in the .cpp to keep this header pre-C++11 compatible.
	struct MutexImpl;
	class Mutex
	{
	public:
		Mutex();
		~Mutex();
		void lock() const;
		void unlock() const;
		bool ok() const;

	private:
		// PIMPL so we don't expose pthread headers here.
		mutable MutexImpl* impl_;
		Mutex(const Mutex&);
		Mutex& operator=(const Mutex&);
	};

	class LockGuard
	{
	public:
		explicit LockGuard(const Mutex& m) : m_(m), locked_(true) { m_.lock(); }
		~LockGuard()
		{
			if (locked_)
				m_.unlock();
		}
		void unlock()
		{
			if (!locked_)
				return;
			m_.unlock();
			locked_ = false;
		}
		void lock()
		{
			if (locked_)
				return;
			m_.lock();
			locked_ = true;
		}

	private:
		const Mutex& m_;
		bool locked_;
		LockGuard(const LockGuard&);
		LockGuard& operator=(const LockGuard&);
	};

	struct Pending
	{
		uint64_t id;
		NNetwork::TransformerServeRequest req;
		CallbackHandle cb;
		uint64_t submittedAtUs;
		Pending() : id(0ULL), req(), cb(), submittedAtUs(0ULL) {}
		Pending(uint64_t newId, const NNetwork::TransformerServeRequest& newReq, const CallbackHandle& newCb, uint64_t newSubmittedAtUs)
		    : id(newId), req(newReq), cb(newCb), submittedAtUs(newSubmittedAtUs)
		{
		}
	};

	struct LiveSlot
	{
		uint64_t id;
		CallbackHandle cb;
		uint64_t admittedAtUs;
		LiveSlot() : id(0ULL), cb(), admittedAtUs(0ULL) {}
	};

	struct DurationWindow
	{
		std::vector<uint64_t> samples;
		unsigned int nextIndex;
		bool filled;

		DurationWindow()
		    : samples(128u, 0ULL),
		      nextIndex(0u),
		      filled(false)
		{
		}

		void clear()
		{
			std::fill(samples.begin(), samples.end(), 0ULL);
			nextIndex = 0u;
			filled = false;
		}

		void add(uint64_t sample)
		{
			if (samples.empty())
				return;
			samples[nextIndex] = sample;
			nextIndex += 1u;
			if (nextIndex >= samples.size())
			{
				nextIndex = 0u;
				filled = true;
			}
		}

		unsigned int size() const
		{
			return filled ? static_cast<unsigned int>(samples.size()) : nextIndex;
		}
	};

	struct DeferredTokenCallback
	{
		uint64_t requestId;
		CallbackHandle cb;
		unsigned int tokenId;
		unsigned int generatedIndex;
		DeferredTokenCallback()
		    : requestId(0ULL), cb(), tokenId(0u), generatedIndex(0u)
		{
		}
		DeferredTokenCallback(uint64_t newId,
		                      const CallbackHandle& newCb,
		                      unsigned int newTokenId,
		                      unsigned int newGeneratedIndex)
		    : requestId(newId), cb(newCb), tokenId(newTokenId), generatedIndex(newGeneratedIndex)
		{
		}
	};

	struct DeferredCancelCheck
	{
		unsigned int slot;
		uint64_t requestId;
		CallbackHandle cb;
		DeferredCancelCheck() : slot(0u), requestId(0ULL), cb() {}
		DeferredCancelCheck(unsigned int newSlot, uint64_t newRequestId, const CallbackHandle& newCb)
		    : slot(newSlot), requestId(newRequestId), cb(newCb)
		{
		}
	};

	// Adapter used by NNetwork::transformerLmServeBatcherStep.
	class BatcherCallbacks : public ITransformerServeCallbacks
	{
	public:
		BatcherCallbacks(const TransformerServingLayer& layer) : layer_(layer) {}
		virtual bool onToken(const NNetwork& net, unsigned int requestIndex, unsigned int tokenId, unsigned int generatedIndex);
		virtual bool shouldStopAll(const NNetwork& net);
		virtual bool shouldStopRequest(const NNetwork& net, unsigned int requestIndex);

	private:
		const TransformerServingLayer& layer_;
	};

	void logEvent(const char* event,
	             int level,
	             uint64_t requestId,
	             const char* msg,
	             const NNetworkStatus* st = NULL,
	             unsigned int slot = static_cast<unsigned int>(-1)) const;
	void logRequestDone_(uint64_t requestId,
	                    const RequestSnapshot& snap,
	                    const NNetworkStatus& finalStatus,
	                    unsigned int slot) const;
	unsigned int countDoneSnapshots_() const;
	void noteFailure_(uint64_t requestId, unsigned int slot, const NNetworkStatus& st);
	void noteSubmitRejected_(bool backpressure);
	void updatePeakDepths_();
	void noteStepDuration_(uint64_t stepStartUs);
	void finalizeRequestMetrics_(uint64_t requestId, RequestSnapshot& snap);
	void fillDurationPercentiles_(const DurationWindow& window,
	                             uint64_t& outP50,
	                             uint64_t& outP99,
	                             uint64_t& outP999) const;

private:
	// Helpers: step() only (single-threaded).
	unsigned int countActiveSlots_() const;
	bool findFreeSlot_(unsigned int& outSlot) const;
	void admitPending_();
	void applyBatcherResult_(RequestSnapshot& snap, const NNetwork::TransformerGenerateResult& rr) const;
	void markSnapshotStoppedByCallback_(RequestSnapshot& snap, const NNetworkStatus* terminalStatus) const;
	void clearLiveSlot_(unsigned int slot);
	void collectDeferredCancelChecks_(std::vector<DeferredCancelCheck>& out) const;
	void applyDeferredCancelDecisions_(const std::vector<unsigned int>& cancelSlots,
	                                  const std::vector<uint64_t>& exceptionIds);
	void markLiveRequestsFailed_(const NNetworkStatus& st);
	void applyTokenCallbackStops_(const std::vector<uint64_t>& stopIds,
	                             const std::vector<uint64_t>& exceptionIds);
	void updateSnapshotsFromBatcher_();
	void finalizeDoneSlots_();
	void finalizeSnapshotForShutdown_(uint64_t requestId, const NNetworkStatus* terminalStatus);
	void shutdownLocked_(bool clearSnapshots, const NNetworkStatus* terminalStatus, const char* logMsg);
	bool mutexOk_() const;

private:
	// Owned by user; must outlive this layer.
	const NNetwork* net_;

	Config cfg_;

	// Worker owns this batcher (thread confinement).
	NNetwork::TransformerServeBatcher batcher_;
	BatcherCallbacks batcherCallbacks_;

	// Control flags.
	bool running_;
	bool stopRequested_;

	// Re-entrancy guard: true while step() is executing batcher callbacks.
	// Prevents user callbacks from re-entering step() (which would corrupt batcher state).
	// Also used to detect blocking callbacks (via diagnostic logging).
	bool inStep_;

	// Synchronizes all public APIs and internal state.
	mutable Mutex mu_;

	// Cancellation flags per slot (read by callbacks).
	// Size == cfg_.maxBatchSize after start().
	std::vector<unsigned char> slotCancel_;

	// Live slot metadata (id + cb) owned by worker thread, but read by callbacks.
	// Size == cfg_.maxBatchSize; updated only by worker thread.
	std::vector<LiveSlot> live_;

	// Pending queue and snapshots (single-threaded; external server should synchronize if needed).
	std::deque<Pending> pending_;
	std::map<uint64_t, RequestSnapshot> snapshots_;
	mutable std::vector<DeferredTokenCallback> deferredTokenCallbacks_;
	std::map<uint64_t, CallbackHandle> completedCallbacks_;
	DurationWindow requestLatencySamples_;
	DurationWindow stepDurationSamples_;

	Diagnostics diagnostics_;

	// Request id generator.
	uint64_t nextId_;
	};

} // namespace glades
