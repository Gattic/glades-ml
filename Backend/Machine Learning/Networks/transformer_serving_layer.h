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
// - For simplicity and determinism, callbacks may execute while the serving layer holds its
//   internal lock. Callbacks should be fast and must not call `step()` re-entrantly.
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

		// Number of tokens already delivered through popNewTokens().
		unsigned int streamedTokenCount;

		RequestSnapshot()
		    : requestId(0ULL),
		      done(false),
		      status(NNetworkStatus::OK, std::string()),
		      result(),
		      streamedTokenCount(0u)
		{
		}
	};

	TransformerServingLayer();
	~TransformerServingLayer();

	// Initialize/reset the serving layer.
	// The referenced `net` must outlive this serving layer.
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
	// If callbacks is non-null, they may be invoked from step() on the caller's thread.
	NNetworkStatus submit(const NNetwork::TransformerServeRequest& req, uint64_t& outRequestId, ITransformerServingCallbacks* callbacks = NULL);

	// Request cancellation. Best-effort: takes effect on the next decode step.
	// Returns false if requestId not found (already done/removed or never existed).
	bool cancel(uint64_t requestId);

	// Get a snapshot of a request's current state. Returns false if not found.
	bool getSnapshot(uint64_t requestId, RequestSnapshot& out) const;

	// Pop tokens generated since the last pop for this request.
	// Returns false if requestId not found.
	bool popNewTokens(uint64_t requestId, std::vector<unsigned int>& outNewTokens, bool& outDone, NNetworkStatus& outStatus);

	// Forget a completed request snapshot (does not affect the model/batcher).
	// Returns false if requestId not found.
	bool clearSnapshot(uint64_t requestId);

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
		ITransformerServingCallbacks* cb;
		Pending() : id(0ULL), req(), cb(NULL) {}
	};

	struct LiveSlot
	{
		uint64_t id;
		ITransformerServingCallbacks* cb;
		LiveSlot() : id(0ULL), cb(NULL) {}
	};

	// Adapter used by NNetwork::transformerLmServeBatcherStep.
	class BatcherCallbacks : public NNetwork::ITransformerServeCallbacks
	{
	public:
		BatcherCallbacks(const TransformerServingLayer& layer) : layer_(layer) {}
		virtual bool onToken(const NNetwork& net, unsigned int requestIndex, unsigned int tokenId, unsigned int generatedIndex);
		virtual bool shouldStopAll(const NNetwork& net);
		virtual bool shouldStopRequest(const NNetwork& net, unsigned int requestIndex);

	private:
		const TransformerServingLayer& layer_;
	};

	void logEvent(const char* event, uint64_t requestId, const char* msg) const;

private:
	// Helpers: step() only (single-threaded).
	unsigned int countActiveSlots_() const;
	bool findFreeSlot_(unsigned int& outSlot) const;
	void admitPending_();
	void updateSnapshotsFromBatcher_();
	void finalizeDoneSlots_();

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

	// Request id generator.
	uint64_t nextId_;
};

} // namespace glades

