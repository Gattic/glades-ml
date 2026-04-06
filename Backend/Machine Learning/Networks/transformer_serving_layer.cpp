// Serving layer implementation for Transformer token-LM generation.

#include "transformer_serving_layer.h"

#include "Backend/Database/GLogger.h"

#include <algorithm>
#include <pthread.h>
#include <sstream>

#include "logfmt_utils.h"

namespace glades {

// ===== Internal synchronization (recursive mutex) =====
struct TransformerServingLayer::MutexImpl
{
	pthread_mutex_t m;
	bool ok;
};

TransformerServingLayer::Mutex::Mutex() : impl_(NULL)
{
	impl_ = new MutexImpl();
	impl_->ok = false;
	pthread_mutexattr_t attr;
	if (pthread_mutexattr_init(&attr) == 0)
	{
		// True recursive mutex (POSIX).
		(void)pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
		impl_->ok = (pthread_mutex_init(&impl_->m, &attr) == 0);
		(void)pthread_mutexattr_destroy(&attr);
	}
	else
	{
		// Fallback: plain mutex (best-effort).
		impl_->ok = (pthread_mutex_init(&impl_->m, NULL) == 0);
	}
}

TransformerServingLayer::Mutex::~Mutex()
{
	if (!impl_)
		return;
	if (impl_->ok)
		(void)pthread_mutex_destroy(&impl_->m);
	delete impl_;
	impl_ = NULL;
}

void TransformerServingLayer::Mutex::lock() const
{
	if (!impl_)
		return;
	if (!impl_->ok)
		return;
	(void)pthread_mutex_lock(&impl_->m);
}

void TransformerServingLayer::Mutex::unlock() const
{
	if (!impl_)
		return;
	if (!impl_->ok)
		return;
	(void)pthread_mutex_unlock(&impl_->m);
}

using namespace glades::logfmt;

TransformerServingLayer::TransformerServingLayer()
    : net_(NULL),
      cfg_(),
      batcher_(),
      batcherCallbacks_(*this),
      running_(false),
      stopRequested_(false),
      inStep_(false),
      mu_(),
      slotCancel_(),
      live_(),
      pending_(),
      snapshots_(),
      nextId_(1ULL)
{
}

TransformerServingLayer::~TransformerServingLayer()
{
	stop();
}

bool TransformerServingLayer::isRunning() const
{
	LockGuard lock(mu_);
	return running_;
}

NNetworkStatus TransformerServingLayer::start(const NNetwork& net, const Config& cfg)
{
	LockGuard lock(mu_);
	stop(); // idempotent reset

	if (cfg.maxBatchSize == 0u)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "TransformerServingLayer::start: maxBatchSize is 0");
	if (cfg.maxSeqLen == 0u)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "TransformerServingLayer::start: maxSeqLen is 0");

	net_ = &net;
	cfg_ = cfg;

	// Initialize cancellation flags and live slot metadata.
	slotCancel_.assign(cfg_.maxBatchSize, 0u);
	live_.assign(cfg_.maxBatchSize, LiveSlot());

	// Reset batcher.
	NNetwork::TransformerServeBatcherConfig bcfg;
	bcfg.maxBatchSize = cfg_.maxBatchSize;
	bcfg.maxSeqLen = cfg_.maxSeqLen;
	bcfg.wipeKvOnRemove = cfg_.wipeKvOnRemove;
	bcfg.rngSeed = cfg_.rngSeed;

	const NNetworkStatus st = net_->transformerLmServeBatcherReset(batcher_, bcfg);
	if (!st.ok())
	{
		net_ = NULL;
		return st;
	}

	stopRequested_ = false;
	running_ = true;

	logEvent("transformer_serving_start", 0ULL, "ok");
	return NNetworkStatus(NNetworkStatus::OK, std::string());
}

void TransformerServingLayer::stop()
{
	LockGuard lock(mu_);
	if (!running_)
		return;

	running_ = false;
	stopRequested_ = true;

	// Clear state.
	pending_.clear();
	// Keep snapshots for post-mortem inspection? In production you'd probably clear; here we keep by default.
	net_ = NULL;
	logEvent("transformer_serving_stop", 0ULL, "ok");
}

NNetworkStatus TransformerServingLayer::submit(const NNetwork::TransformerServeRequest& req,
                                              uint64_t& outRequestId,
                                              ITransformerServingCallbacks* callbacks)
{
	LockGuard lock(mu_);
	outRequestId = 0ULL;
	if (!running_ || !net_)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "TransformerServingLayer::submit: serving layer not running");

	// Lightweight validation here; deeper validation happens inside the model submit/reset paths.
	if (req.promptTokens.empty())
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "TransformerServingLayer::submit: promptTokens is empty");
	if (req.cfg.maxSeqLen > 0u && req.cfg.maxSeqLen > cfg_.maxSeqLen)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "TransformerServingLayer::submit: request maxSeqLen exceeds serving maxSeqLen");

	const uint64_t id = nextId_++;
	if (cfg_.maxPendingRequests > 0u && pending_.size() >= static_cast<size_t>(cfg_.maxPendingRequests))
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "TransformerServingLayer::submit: pending queue full (backpressure)");

	Pending p;
	p.id = id;
	p.req = req;
	p.cb = callbacks;
	pending_.push_back(p);

	RequestSnapshot snap;
	snap.requestId = id;
	snap.done = false;
	snap.status = NNetworkStatus(NNetworkStatus::OK, std::string());
	snap.result = NNetwork::TransformerGenerateResult(); // empty
	snap.streamedTokenCount = 0u;
	snapshots_[id] = snap;

	outRequestId = id;
	logEvent("transformer_serving_submit", id, "queued");
	return NNetworkStatus(NNetworkStatus::OK, std::string());
}

bool TransformerServingLayer::cancel(uint64_t requestId)
{
	LockGuard lock(mu_);
	// Mark as cancelled in snapshot and (if live) set slot cancel flag.
	std::map<uint64_t, RequestSnapshot>::iterator it = snapshots_.find(requestId);
	if (it == snapshots_.end())
		return false;

	// If already done, nothing to do (idempotent).
	if (it->second.done)
		return true;

	// If still pending in queue, mark snapshot as done-by-callback and remove from queue.
	for (std::deque<Pending>::iterator q = pending_.begin(); q != pending_.end(); ++q)
	{
		if (q->id == requestId)
		{
			it->second.done = true;
			it->second.result.stoppedByCallback = true;
			it->second.status = NNetworkStatus(NNetworkStatus::OK, std::string());
			pending_.erase(q);
			logEvent("transformer_serving_cancel", requestId, "cancelled_pending");
			return true;
		}
	}

	// Live slot cancellation is handled by shouldStopRequest() via slotCancel_.
	for (unsigned int s = 0u; s < live_.size(); ++s)
	{
		if (live_[s].id == requestId)
		{
			slotCancel_[s] = 1u;
			logEvent("transformer_serving_cancel", requestId, "cancel_requested");
			return true;
		}
	}

	// If we couldn't find it pending or live, it may have just finished; treat as not found.
	return false;
}

bool TransformerServingLayer::getSnapshot(uint64_t requestId, RequestSnapshot& out) const
{
	LockGuard lock(mu_);
	std::map<uint64_t, RequestSnapshot>::const_iterator it = snapshots_.find(requestId);
	if (it == snapshots_.end())
		return false;
	out = it->second;
	return true;
}

bool TransformerServingLayer::popNewTokens(uint64_t requestId,
                                          std::vector<unsigned int>& outNewTokens,
                                          bool& outDone,
                                          NNetworkStatus& outStatus)
{
	LockGuard lock(mu_);
	outNewTokens.clear();
	outDone = false;
	outStatus = NNetworkStatus(NNetworkStatus::OK, std::string());

	std::map<uint64_t, RequestSnapshot>::iterator it = snapshots_.find(requestId);
	if (it == snapshots_.end())
		return false;

	RequestSnapshot& s = it->second;
	outDone = s.done;
	outStatus = s.status;

	const unsigned int have = static_cast<unsigned int>(s.result.tokens.size());
	const unsigned int from = s.streamedTokenCount;
	if (from < have)
	{
		outNewTokens.insert(outNewTokens.end(), s.result.tokens.begin() + from, s.result.tokens.end());
		s.streamedTokenCount = have;
	}
	return true;
}

bool TransformerServingLayer::clearSnapshot(uint64_t requestId)
{
	LockGuard lock(mu_);
	std::map<uint64_t, RequestSnapshot>::iterator it = snapshots_.find(requestId);
	if (it == snapshots_.end())
		return false;
	snapshots_.erase(it);
	return true;
}

bool TransformerServingLayer::BatcherCallbacks::onToken(const NNetwork& net, unsigned int requestIndex, unsigned int tokenId, unsigned int generatedIndex)
{
	// requestIndex is the batcher slot index.
	if (requestIndex >= layer_.live_.size())
		return false;
	const uint64_t id = layer_.live_[requestIndex].id;
	ITransformerServingCallbacks* cb = layer_.live_[requestIndex].cb;
	if (!cb || id == 0ULL)
		return false;
	try
	{
		return cb->onToken(id, net, tokenId, generatedIndex);
	}
	catch (...)
	{
		// User callback threw — treat as cancellation to prevent state corruption.
		layer_.logEvent("transformer_serving_callback_exception", id, "onToken threw; treating as cancel");
		return true;
	}
}

bool TransformerServingLayer::BatcherCallbacks::shouldStopAll(const NNetwork& /*net*/)
{
	// This serving layer does not implement global cancellation; the owner can call stop().
	return layer_.stopRequested_;
}

bool TransformerServingLayer::BatcherCallbacks::shouldStopRequest(const NNetwork& net, unsigned int requestIndex)
{
	if (requestIndex >= layer_.slotCancel_.size())
		return false;
	if (layer_.slotCancel_[requestIndex] != 0u)
		return true;
	const uint64_t id = (requestIndex < layer_.live_.size()) ? layer_.live_[requestIndex].id : 0ULL;
	ITransformerServingCallbacks* cb = (requestIndex < layer_.live_.size()) ? layer_.live_[requestIndex].cb : NULL;
	if (cb && id != 0ULL)
	{
		try
		{
			return cb->shouldCancel(id, net);
		}
		catch (...)
		{
			// User callback threw — treat as cancellation to prevent state corruption.
			layer_.logEvent("transformer_serving_callback_exception", id, "shouldCancel threw; treating as cancel");
			return true;
		}
	}
	return false;
}

void TransformerServingLayer::logEvent(const char* event, uint64_t requestId, const char* msg) const
{
	if (!cfg_.enableLogs || !net_)
		return;
	shmea::GLogger* logger = net_->getLogger();
	if (!logger)
		return;

	std::ostringstream oss;
	oss << "event=" << (event ? event : "transformer_serving_event");
	append_logfmt_kv(oss, "request_id", requestId);
	append_logfmt_kv(oss, "max_batch", cfg_.maxBatchSize);
	append_logfmt_kv(oss, "max_seq_len", cfg_.maxSeqLen);
	append_logfmt_kv(oss, "wipe_kv", cfg_.wipeKvOnRemove);
	append_logfmt_kv(oss, "auto_remove", cfg_.autoRemoveFinished);
	if (msg)
		append_logfmt_kv(oss, "msg", std::string(msg));

	logger->info("TransformerServe", shmea::GString(oss.str().c_str()));
}

unsigned int TransformerServingLayer::countActiveSlots_() const
{
	unsigned int n = 0u;
	if (batcher_.inUse.empty())
		return 0u;
	for (unsigned int s = 0u; s < batcher_.maxBatchSize; ++s)
		if (batcher_.inUse[s] != 0u)
			++n;
	return n;
}

bool TransformerServingLayer::findFreeSlot_(unsigned int& outSlot) const
{
	outSlot = batcher_.maxBatchSize;
	for (unsigned int s = 0u; s < batcher_.maxBatchSize; ++s)
	{
		if (s < batcher_.inUse.size() && batcher_.inUse[s] == 0u)
		{
			outSlot = s;
			return true;
		}
	}
	return false;
}

void TransformerServingLayer::admitPending_()
{
	// Admit as many as possible into free slots.
	while (!pending_.empty())
	{
		unsigned int slot = 0u;
		if (!findFreeSlot_(slot))
			return;

		Pending p = pending_.front();

		unsigned int outSlot = 0u;
		const NNetworkStatus st = net_->transformerLmServeBatcherSubmit(batcher_, p.req, outSlot);
		if (!st.ok())
		{
			// Mark request failed and drop it.
			std::map<uint64_t, RequestSnapshot>::iterator sit = snapshots_.find(p.id);
			if (sit != snapshots_.end())
			{
				sit->second.done = true;
				sit->second.status = st;
			}
			logEvent("transformer_serving_submit_fail", p.id, st.message.c_str());
			pending_.pop_front();
			continue;
		}

		// Successful submit.
		pending_.pop_front();
		if (outSlot < live_.size())
		{
			live_[outSlot].id = p.id;
			live_[outSlot].cb = p.cb;
		}
		if (outSlot < slotCancel_.size())
			slotCancel_[outSlot] = 0u;

		logEvent("transformer_serving_admit", p.id, "admitted");
	}
}

void TransformerServingLayer::updateSnapshotsFromBatcher_()
{
	for (unsigned int s = 0u; s < live_.size(); ++s)
	{
		const uint64_t id = live_[s].id;
		if (id == 0ULL)
			continue;
		std::map<uint64_t, RequestSnapshot>::iterator it = snapshots_.find(id);
		if (it == snapshots_.end())
			continue;
		RequestSnapshot& snap = it->second;
		const NNetwork::TransformerGenerateResult& rr = (s < batcher_.results.size()) ? batcher_.results[s] : snap.result;
		// Append new tokens since last snapshot.
		if (rr.tokens.size() > snap.result.tokens.size())
		{
			snap.result.tokens.insert(snap.result.tokens.end(),
			                         rr.tokens.begin() + snap.result.tokens.size(),
			                         rr.tokens.end());
		}
		snap.result.lastToken = rr.lastToken;
		snap.result.stoppedByCallback = rr.stoppedByCallback;
		snap.result.stoppedOnEos = rr.stoppedOnEos;
		snap.result.stoppedByStopToken = rr.stoppedByStopToken;
		snap.result.stoppedByLimit = rr.stoppedByLimit;
	}
}

void TransformerServingLayer::finalizeDoneSlots_()
{
	// Called after a successful Step().
	if (batcher_.inUse.empty() || batcher_.done.empty())
		return;

	for (unsigned int s = 0u; s < batcher_.maxBatchSize; ++s)
	{
		if (s >= batcher_.inUse.size() || s >= batcher_.done.size())
			continue;
		if (batcher_.inUse[s] == 0u || batcher_.done[s] == 0u)
			continue;

		const uint64_t id = (s < live_.size()) ? live_[s].id : 0ULL;
		if (id == 0ULL)
		{
			// Defensive: unknown slot state; remove it.
			(void)net_->transformerLmServeBatcherRemove(batcher_, s);
			continue;
		}

		// Update final snapshot.
		std::map<uint64_t, RequestSnapshot>::iterator it = snapshots_.find(id);
		if (it != snapshots_.end())
		{
			RequestSnapshot& snap = it->second;
			// Merge tokens incrementally (avoid full copies each tick).
			const NNetwork::TransformerGenerateResult& rr = batcher_.results[s];
			if (rr.tokens.size() > snap.result.tokens.size())
			{
				snap.result.tokens.insert(snap.result.tokens.end(),
				                         rr.tokens.begin() + snap.result.tokens.size(),
				                         rr.tokens.end());
			}
			snap.result.lastToken = rr.lastToken;
			snap.result.stoppedByCallback = rr.stoppedByCallback;
			snap.result.stoppedOnEos = rr.stoppedOnEos;
			snap.result.stoppedByStopToken = rr.stoppedByStopToken;
			snap.result.stoppedByLimit = rr.stoppedByLimit;
			snap.done = true;
			snap.status = NNetworkStatus(NNetworkStatus::OK, std::string());
		}

		logEvent("transformer_serving_request_done", id, "done");

		// Clear live slot metadata.
		if (s < live_.size())
		{
			live_[s].id = 0ULL;
			live_[s].cb = NULL;
		}
		if (s < slotCancel_.size())
			slotCancel_[s] = 0u;

		// Remove from batcher (optional; but default enabled for serving).
		if (cfg_.autoRemoveFinished)
			(void)net_->transformerLmServeBatcherRemove(batcher_, s);
	}
}

NNetworkStatus TransformerServingLayer::step()
{
	LockGuard lock(mu_);
	if (!running_ || !net_)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "TransformerServingLayer::step: layer not running");
	if (stopRequested_)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "TransformerServingLayer::step: stop requested");

	// Re-entrancy guard: prevent step() from being called from within a batcher callback.
	// The recursive mutex would allow the lock, but batcher state is not re-entrant-safe.
	if (inStep_)
	{
		logEvent("transformer_serving_reentrant_step", 0ULL, "step() called from callback; rejected");
		return NNetworkStatus(NNetworkStatus::INVALID_STATE,
		    "TransformerServingLayer::step: re-entrant call detected (likely from a callback). "
		    "Callbacks must not call step().");
	}

	// 1) Admit as many pending requests as possible.
	admitPending_();

	// 2) If nothing active, nothing to do.
	if (countActiveSlots_() == 0u)
		return NNetworkStatus(NNetworkStatus::OK, std::string());

	// 3) Advance one global step.
	// Set inStep_ around the batcher call so callbacks can detect re-entrancy.
	inStep_ = true;
	const NNetworkStatus stStep = net_->transformerLmServeBatcherStep(batcher_, &batcherCallbacks_);
	inStep_ = false;

	if (!stStep.ok())
	{
		// Mark all live requests failed.
		for (unsigned int s = 0u; s < live_.size(); ++s)
		{
			const uint64_t id = live_[s].id;
			if (id == 0ULL)
				continue;
			std::map<uint64_t, RequestSnapshot>::iterator it = snapshots_.find(id);
			if (it != snapshots_.end() && !it->second.done)
			{
				it->second.done = true;
				it->second.status = stStep;
			}
		}
		logEvent("transformer_serving_step_fail", 0ULL, stStep.message.c_str());
		stopRequested_ = true;
		return stStep;
	}

	// 4) Update snapshots and finalize done slots.
	updateSnapshotsFromBatcher_();
	finalizeDoneSlots_();

	return NNetworkStatus(NNetworkStatus::OK, std::string());
}

} // namespace glades

