// Serving layer implementation for Transformer token-LM generation.

#include "transformer_serving_layer.h"

#include "Backend/Database/GLogger.h"

#include <algorithm>
#include <pthread.h>
#include <sstream>

#include "logfmt_utils.h"

namespace glades {

namespace {

static void append_token_delta(std::vector<unsigned int>& dst, const std::vector<unsigned int>& src)
{
	if (src.size() <= dst.size())
		return;
	const size_t from = dst.size();
	dst.reserve(src.size());
	dst.insert(dst.end(), src.begin() + static_cast<std::ptrdiff_t>(from), src.end());
}

struct DeferredCancelCheck
{
	unsigned int slot;
	uint64_t requestId;
	glades::ITransformerServingCallbacks* cb;
	DeferredCancelCheck() : slot(0u), requestId(0ULL), cb(NULL) {}
	DeferredCancelCheck(unsigned int newSlot, uint64_t newRequestId, glades::ITransformerServingCallbacks* newCb)
	    : slot(newSlot), requestId(newRequestId), cb(newCb)
	{
	}
};

} // namespace

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
	if (pthread_mutexattr_init(&attr) != 0)
		return;
	if (pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE) != 0)
	{
		(void)pthread_mutexattr_destroy(&attr);
		return;
	}
	impl_->ok = (pthread_mutex_init(&impl_->m, &attr) == 0);
	(void)pthread_mutexattr_destroy(&attr);
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

bool TransformerServingLayer::Mutex::ok() const
{
	return impl_ && impl_->ok;
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
	shutdownLocked_(false, NULL);

	if (!mutexOk_())
		return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "TransformerServingLayer::start: recursive mutex initialization failed");

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
	if (inStep_)
	{
		// Callbacks are allowed to request shutdown, but destructive teardown must wait
		// until step() has finished using the current batcher/net state.
		running_ = false;
		stopRequested_ = true;
		logEvent("transformer_serving_stop_deferred", 0ULL, "deferred_until_step_exit");
		return;
	}
	shutdownLocked_(false, "ok");
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

	pending_.push_back(Pending(id, req, callbacks));

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
	if (!it->second.done)
		return false;
	snapshots_.erase(it);
	return true;
}

bool TransformerServingLayer::BatcherCallbacks::onToken(const NNetwork& net, unsigned int requestIndex, unsigned int tokenId, unsigned int generatedIndex)
{
	(void)net;
	// requestIndex is the batcher slot index.
	if (requestIndex >= layer_.live_.size())
		return false;
	const uint64_t id = layer_.live_[requestIndex].id;
	ITransformerServingCallbacks* cb = layer_.live_[requestIndex].cb;
	if (!cb || id == 0ULL)
		return false;
	layer_.deferredTokenCallbacks_.push_back(DeferredTokenCallback(id, cb, tokenId, generatedIndex));
	return false;
}

bool TransformerServingLayer::BatcherCallbacks::shouldStopAll(const NNetwork& /*net*/)
{
	// This serving layer does not implement global cancellation; the owner can call stop().
	LockGuard lock(layer_.mu_);
	return layer_.stopRequested_;
}

bool TransformerServingLayer::BatcherCallbacks::shouldStopRequest(const NNetwork& net, unsigned int requestIndex)
{
	(void)net;
	if (requestIndex >= layer_.slotCancel_.size())
		return false;
	return (layer_.slotCancel_[requestIndex] != 0u);
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

		Pending& p = pending_.front();
		const uint64_t requestId = p.id;
		ITransformerServingCallbacks* cb = p.cb;

		unsigned int outSlot = 0u;
		const NNetworkStatus st = net_->transformerLmServeBatcherSubmit(batcher_, p.req, outSlot);
		if (!st.ok())
		{
			// Mark request failed and drop it.
			std::map<uint64_t, RequestSnapshot>::iterator sit = snapshots_.find(requestId);
			if (sit != snapshots_.end())
			{
				sit->second.done = true;
				sit->second.status = st;
			}
			logEvent("transformer_serving_submit_fail", requestId, st.message.c_str());
			pending_.pop_front();
			continue;
		}

		// Successful submit.
		pending_.pop_front();
		if (outSlot < live_.size())
		{
			live_[outSlot].id = requestId;
			live_[outSlot].cb = cb;
		}
		if (outSlot < slotCancel_.size())
			slotCancel_[outSlot] = 0u;

		logEvent("transformer_serving_admit", requestId, "admitted");
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
		append_token_delta(snap.result.tokens, rr.tokens);
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
	if (!net_ || batcher_.inUse.empty() || batcher_.done.empty())
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
			const NNetwork::TransformerGenerateResult& rr = batcher_.results[s];
			append_token_delta(snap.result.tokens, rr.tokens);
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
	if (!mutexOk_())
		return NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "TransformerServingLayer::step: recursive mutex initialization failed");
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
	// Keep inStep_ true for the entire operation so callbacks can reject step() re-entry,
	// even though user callbacks themselves execute without the layer mutex held.
	inStep_ = true;
	const NNetwork* stepNet = net_;
	std::vector<DeferredCancelCheck> cancelChecks;
	cancelChecks.reserve(live_.size());
	for (unsigned int s = 0u; s < live_.size(); ++s)
	{
		if (s >= batcher_.inUse.size() || s >= batcher_.done.size() ||
		    s >= batcher_.promptPos.size() || s >= batcher_.promptLen.size() ||
		    s >= batcher_.generated.size() || s >= batcher_.reqMaxNew.size())
			continue;
		if (batcher_.inUse[s] == 0u || batcher_.done[s] != 0u)
			continue;
		if (batcher_.promptPos[s] < batcher_.promptLen[s])
			continue;
		if (batcher_.generated[s] >= batcher_.reqMaxNew[s])
			continue;
		if (live_[s].id == 0ULL || live_[s].cb == NULL)
			continue;
		cancelChecks.push_back(DeferredCancelCheck(s, live_[s].id, live_[s].cb));
	}

	lock.unlock();
	std::vector<unsigned int> cancelSlots;
	std::vector<uint64_t> cancelLogIds;
	for (size_t i = 0u; i < cancelChecks.size(); ++i)
	{
		bool stopReq = false;
		try
		{
			stopReq = cancelChecks[i].cb->shouldCancel(cancelChecks[i].requestId, *stepNet);
		}
		catch (...)
		{
			stopReq = true;
			cancelLogIds.push_back(cancelChecks[i].requestId);
		}
		if (stopReq)
			cancelSlots.push_back(cancelChecks[i].slot);
	}
	lock.lock();

	if (stopRequested_)
	{
		inStep_ = false;
		shutdownLocked_(false, "ok");
		return NNetworkStatus(NNetworkStatus::OK, std::string());
	}
	for (size_t i = 0u; i < cancelLogIds.size(); ++i)
		logEvent("transformer_serving_callback_exception", cancelLogIds[i], "shouldCancel threw; treating as cancel");
	for (size_t i = 0u; i < cancelSlots.size(); ++i)
	{
		const unsigned int slot = cancelSlots[i];
		if (slot < live_.size() && slot < slotCancel_.size() && live_[slot].id != 0ULL)
			slotCancel_[slot] = 1u;
	}

	deferredTokenCallbacks_.clear();
	const NNetworkStatus stStep = stepNet->transformerLmServeBatcherStep(batcher_, &batcherCallbacks_);

	if (!stStep.ok())
	{
		inStep_ = false;
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
		shutdownLocked_(false, "step_failed");
		return stStep;
	}

	// 4) Update snapshots and finalize done slots.
	updateSnapshotsFromBatcher_();
	finalizeDoneSlots_();
	std::vector<DeferredTokenCallback> deferred = deferredTokenCallbacks_;
	deferredTokenCallbacks_.clear();

	lock.unlock();
	std::vector<uint64_t> stopIds;
	std::vector<uint64_t> tokenExceptionIds;
	for (size_t i = 0u; i < deferred.size(); ++i)
	{
		ITransformerServingCallbacks* cb = deferred[i].cb;
		if (!cb || deferred[i].requestId == 0ULL)
			continue;
		bool stopReq = false;
		try
		{
			stopReq = cb->onToken(deferred[i].requestId, *stepNet, deferred[i].tokenId, deferred[i].generatedIndex);
		}
		catch (...)
		{
			stopReq = true;
			tokenExceptionIds.push_back(deferred[i].requestId);
		}
		if (stopReq)
			stopIds.push_back(deferred[i].requestId);
	}
	lock.lock();

	for (size_t i = 0u; i < tokenExceptionIds.size(); ++i)
		logEvent("transformer_serving_callback_exception", tokenExceptionIds[i], "onToken threw; treating as cancel");
	for (size_t i = 0u; i < stopIds.size(); ++i)
	{
		const uint64_t id = stopIds[i];
		std::map<uint64_t, RequestSnapshot>::iterator it = snapshots_.find(id);
		if (it != snapshots_.end())
		{
			it->second.done = true;
			it->second.status = NNetworkStatus(NNetworkStatus::OK, std::string());
			it->second.result.stoppedByCallback = true;
			it->second.result.stoppedOnEos = false;
			it->second.result.stoppedByStopToken = false;
			it->second.result.stoppedByLimit = false;
		}
		for (unsigned int s = 0u; s < live_.size(); ++s)
		{
			if (live_[s].id != id)
				continue;
			if (s < batcher_.results.size())
			{
				batcher_.results[s].stoppedByCallback = true;
				batcher_.results[s].stoppedOnEos = false;
				batcher_.results[s].stoppedByStopToken = false;
				batcher_.results[s].stoppedByLimit = false;
			}
			if (s < batcher_.done.size())
				batcher_.done[s] = 1u;
			if (s < live_.size())
			{
				live_[s].id = 0ULL;
				live_[s].cb = NULL;
			}
			if (s < slotCancel_.size())
				slotCancel_[s] = 0u;
			if (cfg_.autoRemoveFinished)
				(void)net_->transformerLmServeBatcherRemove(batcher_, s);
			logEvent("transformer_serving_request_done", id, "done");
			break;
		}
	}
	inStep_ = false;

	if (stopRequested_)
	{
		shutdownLocked_(false, "ok");
		return NNetworkStatus(NNetworkStatus::OK, std::string());
	}

	return NNetworkStatus(NNetworkStatus::OK, std::string());
}

void TransformerServingLayer::shutdownLocked_(bool clearSnapshots, const char* logMsg)
{
	const bool hadNet = (net_ != NULL);
	if (hadNet && logMsg)
		logEvent("transformer_serving_stop", 0ULL, logMsg);
	running_ = false;
	stopRequested_ = false;
	inStep_ = false;
	pending_.clear();
	batcher_.reset();
	std::fill(slotCancel_.begin(), slotCancel_.end(), static_cast<unsigned char>(0u));
	for (size_t i = 0u; i < live_.size(); ++i)
	{
		live_[i].id = 0ULL;
		live_[i].cb = NULL;
	}
	if (clearSnapshots)
		snapshots_.clear();
	net_ = NULL;
}

bool TransformerServingLayer::mutexOk_() const
{
	return mu_.ok();
}

} // namespace glades
