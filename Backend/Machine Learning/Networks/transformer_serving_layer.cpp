// Serving layer implementation for Transformer token-LM generation.

#include "transformer_serving_layer.h"
#include "transformer_public_api.h"

#include "Backend/Database/GLogger.h"

#include <algorithm>
#include <pthread.h>
#include <sstream>

#include "logfmt_utils.h"

namespace glades {

namespace {

static const char* status_code_name(glades::NNetworkStatus::Code code)
{
	switch (code)
	{
	case glades::NNetworkStatus::OK: return "ok";
	case glades::NNetworkStatus::INVALID_ARGUMENT: return "invalid_argument";
	case glades::NNetworkStatus::INVALID_STATE: return "invalid_state";
	case glades::NNetworkStatus::EMPTY_DATA: return "empty_data";
	case glades::NNetworkStatus::BUILD_FAILED: return "build_failed";
	case glades::NNetworkStatus::INTERNAL_ERROR: return "internal_error";
	default: return "unknown";
	}
}

static void emit_logger_line(shmea::GLogger* logger,
                             int level,
                             const char* component,
                             const std::string& line)
{
	if (!logger)
		return;
	switch (level)
	{
	case shmea::GLogger::LOG_DEBUG:
		logger->debug(component, shmea::GString(line.c_str()));
		break;
	case shmea::GLogger::LOG_WARNING:
		logger->warning(component, shmea::GString(line.c_str()));
		break;
	case shmea::GLogger::LOG_ERROR:
		logger->error(component, shmea::GString(line.c_str()));
		break;
	case shmea::GLogger::LOG_FATAL:
		logger->fatal(component, shmea::GString(line.c_str()));
		break;
	case shmea::GLogger::LOG_VERBOSE:
		logger->verbose(component, shmea::GString(line.c_str()));
		break;
	case shmea::GLogger::LOG_INFO:
	default:
		logger->info(component, shmea::GString(line.c_str()));
		break;
	}
}

static void append_token_delta(std::vector<unsigned int>& dst, const std::vector<unsigned int>& src)
{
	if (src.size() <= dst.size())
		return;
	const size_t from = dst.size();
	dst.reserve(src.size());
	dst.insert(dst.end(), src.begin() + static_cast<std::ptrdiff_t>(from), src.end());
}

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
      diagnostics_(),
      nextId_(1ULL)
{
}

TransformerServingLayer::~TransformerServingLayer()
{
	LockGuard lock(mu_);
	shutdownLocked_(true, NULL, running_ ? "destroy" : NULL);
}

bool TransformerServingLayer::isRunning() const
{
	LockGuard lock(mu_);
	return running_;
}

NNetworkStatus TransformerServingLayer::start(const NNetwork& net, const Config& cfg)
{
	LockGuard lock(mu_);
	shutdownLocked_(false, NULL, NULL);

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

	const NNetworkStatus st = TransformerPublicAPI::serving(*net_).resetBatcher(batcher_, bcfg);
	if (!st.ok())
	{
		noteFailure_(0ULL, static_cast<unsigned int>(-1), st);
		logEvent("transformer_serving_start_fail", shmea::GLogger::LOG_ERROR, 0ULL, "reset_batcher_failed", &st);
		net_ = NULL;
		return st;
	}

	diagnostics_ = Diagnostics();
	diagnostics_.running = true;
	diagnostics_.maxBatchSize = cfg_.maxBatchSize;
	diagnostics_.maxSeqLen = cfg_.maxSeqLen;
	diagnostics_.maxPendingRequests = cfg_.maxPendingRequests;
	diagnostics_.nextRequestId = nextId_;
	stopRequested_ = false;
	running_ = true;

	logEvent("transformer_serving_start", shmea::GLogger::LOG_INFO, 0ULL, "ok");
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
		diagnostics_.running = false;
		diagnostics_.stopRequested = true;
		logEvent("transformer_serving_stop_deferred", shmea::GLogger::LOG_INFO, 0ULL, "deferred_until_step_exit");
		return;
	}
	shutdownLocked_(false, NULL, "ok");
}

NNetworkStatus TransformerServingLayer::submit(const NNetwork::TransformerServeRequest& req,
                                              uint64_t& outRequestId,
                                              CallbackHandle callbacks)
{
	LockGuard lock(mu_);
	outRequestId = 0ULL;
	if (!running_ || !net_)
	{
		const NNetworkStatus st(NNetworkStatus::INVALID_STATE, "TransformerServingLayer::submit: serving layer not running");
		noteFailure_(0ULL, static_cast<unsigned int>(-1), st);
		return st;
	}

	// Lightweight validation here; deeper validation happens inside the model submit/reset paths.
	if (req.promptTokens.empty())
	{
		const NNetworkStatus st(NNetworkStatus::INVALID_ARGUMENT, "TransformerServingLayer::submit: promptTokens is empty");
		noteFailure_(0ULL, static_cast<unsigned int>(-1), st);
		logEvent("transformer_serving_submit_rejected", shmea::GLogger::LOG_WARNING, 0ULL, "prompt_empty", &st);
		return st;
	}
	if (req.cfg.maxSeqLen > 0u && req.cfg.maxSeqLen > cfg_.maxSeqLen)
	{
		const NNetworkStatus st(NNetworkStatus::INVALID_ARGUMENT, "TransformerServingLayer::submit: request maxSeqLen exceeds serving maxSeqLen");
		noteFailure_(0ULL, static_cast<unsigned int>(-1), st);
		logEvent("transformer_serving_submit_rejected", shmea::GLogger::LOG_WARNING, 0ULL, "max_seq_len_exceeded", &st);
		return st;
	}

	const uint64_t id = nextId_++;
	if (cfg_.maxPendingRequests > 0u && pending_.size() >= static_cast<size_t>(cfg_.maxPendingRequests))
	{
		const NNetworkStatus st(NNetworkStatus::INVALID_STATE, "TransformerServingLayer::submit: pending queue full (backpressure)");
		noteFailure_(id, static_cast<unsigned int>(-1), st);
		logEvent("transformer_serving_submit_rejected", shmea::GLogger::LOG_WARNING, id, "pending_queue_full", &st);
		return st;
	}

	pending_.push_back(Pending(id, req, callbacks));

	RequestSnapshot snap;
	snap.requestId = id;
	snap.done = false;
	snap.status = NNetworkStatus(NNetworkStatus::OK, std::string());
	snap.result = NNetwork::TransformerGenerateResult(); // empty
	snap.streamedTokenCount = 0u;
	snapshots_[id] = snap;

	outRequestId = id;
	diagnostics_.totalSubmitted += 1ULL;
	diagnostics_.nextRequestId = nextId_;
	logEvent("transformer_serving_submit", shmea::GLogger::LOG_DEBUG, id, "queued");
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
			markSnapshotStoppedByCallback_(it->second, NULL);
			completedCallbacks_[requestId] = q->cb;
			pending_.erase(q);
			diagnostics_.totalPendingCancels += 1ULL;
			logEvent("transformer_serving_cancel", shmea::GLogger::LOG_DEBUG, requestId, "cancelled_pending");
			return true;
		}
	}

	// Live slot cancellation is handled by shouldStopRequest() via slotCancel_.
	for (unsigned int s = 0u; s < live_.size(); ++s)
	{
		if (live_[s].id == requestId)
		{
			slotCancel_[s] = 1u;
			diagnostics_.totalLiveCancelRequests += 1ULL;
			logEvent("transformer_serving_cancel", shmea::GLogger::LOG_DEBUG, requestId, "cancel_requested");
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
	completedCallbacks_.erase(requestId);
	diagnostics_.totalSnapshotClears += 1ULL;
	return true;
}

bool TransformerServingLayer::getDiagnostics(Diagnostics& out) const
{
	LockGuard lock(mu_);
	out = diagnostics_;
	out.running = running_;
	out.stopRequested = stopRequested_;
	out.inStep = inStep_;
	out.maxBatchSize = cfg_.maxBatchSize;
	out.maxSeqLen = cfg_.maxSeqLen;
	out.maxPendingRequests = cfg_.maxPendingRequests;
	out.pendingRequests = static_cast<unsigned int>(pending_.size());
	out.activeRequests = countActiveSlots_();
	out.doneSnapshots = countDoneSnapshots_();
	out.snapshotCount = static_cast<unsigned int>(snapshots_.size());
	out.completedCallbackCount = static_cast<unsigned int>(completedCallbacks_.size());
	out.nextRequestId = nextId_;
	return true;
}

bool TransformerServingLayer::BatcherCallbacks::onToken(const NNetwork& net, unsigned int requestIndex, unsigned int tokenId, unsigned int generatedIndex)
{
	(void)net;
	// requestIndex is the batcher slot index.
	if (requestIndex >= layer_.live_.size())
		return false;
	const uint64_t id = layer_.live_[requestIndex].id;
	TransformerServingLayer::CallbackHandle cb = layer_.live_[requestIndex].cb;
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

void TransformerServingLayer::logEvent(const char* event,
                                      int level,
                                      uint64_t requestId,
                                      const char* msg,
                                      const NNetworkStatus* st,
                                      unsigned int slot) const
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
	append_logfmt_kv(oss, "pending_requests", static_cast<unsigned int>(pending_.size()));
	append_logfmt_kv(oss, "active_requests", countActiveSlots_());
	append_logfmt_kv(oss, "done_snapshots", countDoneSnapshots_());
	append_logfmt_kv(oss, "snapshot_count", static_cast<unsigned int>(snapshots_.size()));
	append_logfmt_kv(oss, "step_calls", static_cast<unsigned long long>(diagnostics_.totalStepCalls));
	append_logfmt_kv(oss, "step_failures", static_cast<unsigned long long>(diagnostics_.totalStepFailures));
	append_logfmt_kv(oss, "callback_exceptions", static_cast<unsigned long long>(diagnostics_.totalCallbackExceptions));
	if (slot != static_cast<unsigned int>(-1))
		append_logfmt_kv(oss, "slot", slot);
	if (st)
	{
		append_logfmt_kv(oss, "status_code", std::string(status_code_name(st->code)));
		append_logfmt_kv(oss, "status_ok", st->ok());
		if (!st->message.empty())
			append_logfmt_kv(oss, "error", st->message);
	}
	if (msg)
		append_logfmt_kv(oss, "msg", std::string(msg));

	emit_logger_line(logger, level, "TransformerServe", oss.str());
}

void TransformerServingLayer::finalizeSnapshotForShutdown_(uint64_t requestId, const NNetworkStatus* terminalStatus)
{
	std::map<uint64_t, RequestSnapshot>::iterator it = snapshots_.find(requestId);
	if (it == snapshots_.end() || it->second.done)
		return;
	markSnapshotStoppedByCallback_(it->second, terminalStatus);
}

unsigned int TransformerServingLayer::countActiveSlots_() const
{
	unsigned int n = 0u;
	if (!batcher_.isInitialized())
		return 0u;
	for (unsigned int s = 0u; s < batcher_.capacity(); ++s)
		if (batcher_.slotInUse(s))
			++n;
	return n;
}

unsigned int TransformerServingLayer::countDoneSnapshots_() const
{
	unsigned int n = 0u;
	for (std::map<uint64_t, RequestSnapshot>::const_iterator it = snapshots_.begin(); it != snapshots_.end(); ++it)
	{
		if (it->second.done)
			++n;
	}
	return n;
}

void TransformerServingLayer::noteFailure_(uint64_t requestId, unsigned int slot, const NNetworkStatus& st)
{
	diagnostics_.lastFailureRequestId = requestId;
	diagnostics_.lastFailureSlot = slot;
	diagnostics_.lastFailureStatus = st;
}

bool TransformerServingLayer::findFreeSlot_(unsigned int& outSlot) const
{
	outSlot = batcher_.capacity();
	for (unsigned int s = 0u; s < batcher_.capacity(); ++s)
	{
		if (!batcher_.slotInUse(s))
		{
			outSlot = s;
			return true;
		}
	}
	return false;
}

void TransformerServingLayer::applyBatcherResult_(RequestSnapshot& snap,
                                                  const NNetwork::TransformerGenerateResult& rr) const
{
	append_token_delta(snap.result.tokens, rr.tokens);
	snap.result.lastToken = rr.lastToken;
	snap.result.stoppedByCallback = rr.stoppedByCallback;
	snap.result.stoppedOnEos = rr.stoppedOnEos;
	snap.result.stoppedByStopToken = rr.stoppedByStopToken;
	snap.result.stoppedByLimit = rr.stoppedByLimit;
}

void TransformerServingLayer::markSnapshotStoppedByCallback_(RequestSnapshot& snap,
                                                             const NNetworkStatus* terminalStatus) const
{
	snap.done = true;
	snap.result.stoppedByCallback = true;
	snap.result.stoppedOnEos = false;
	snap.result.stoppedByStopToken = false;
	snap.result.stoppedByLimit = false;
	if (terminalStatus)
		snap.status = *terminalStatus;
	else
		snap.status = NNetworkStatus(NNetworkStatus::OK, std::string());
}

void TransformerServingLayer::clearLiveSlot_(unsigned int slot)
{
	if (slot < live_.size())
	{
		live_[slot].id = 0ULL;
		live_[slot].cb = CallbackHandle();
	}
	if (slot < slotCancel_.size())
		slotCancel_[slot] = 0u;
}

void TransformerServingLayer::collectDeferredCancelChecks_(std::vector<DeferredCancelCheck>& out) const
{
	out.clear();
	out.reserve(live_.size());
	for (unsigned int s = 0u; s < live_.size(); ++s)
	{
		if (!batcher_.slotCanDecode(s))
			continue;
		if (live_[s].id == 0ULL || !live_[s].cb)
			continue;
		out.push_back(DeferredCancelCheck(s, live_[s].id, live_[s].cb));
	}
}

void TransformerServingLayer::applyDeferredCancelDecisions_(const std::vector<unsigned int>& cancelSlots,
                                                            const std::vector<uint64_t>& exceptionIds)
{
	for (size_t i = 0u; i < exceptionIds.size(); ++i)
	{
		diagnostics_.totalCallbackExceptions += 1ULL;
		logEvent("transformer_serving_callback_exception", shmea::GLogger::LOG_WARNING, exceptionIds[i], "shouldCancel threw; treating as cancel");
	}
	for (size_t i = 0u; i < cancelSlots.size(); ++i)
	{
		const unsigned int slot = cancelSlots[i];
		if (slot < live_.size() && slot < slotCancel_.size() && live_[slot].id != 0ULL)
			slotCancel_[slot] = 1u;
	}
}

void TransformerServingLayer::markLiveRequestsFailed_(const NNetworkStatus& st)
{
	for (unsigned int s = 0u; s < live_.size(); ++s)
	{
		const uint64_t id = live_[s].id;
		if (id == 0ULL)
			continue;
		std::map<uint64_t, RequestSnapshot>::iterator it = snapshots_.find(id);
		if (it != snapshots_.end() && !it->second.done)
		{
			it->second.done = true;
			it->second.status = st;
		}
	}
}

void TransformerServingLayer::applyTokenCallbackStops_(const std::vector<uint64_t>& stopIds,
                                                       const std::vector<uint64_t>& exceptionIds)
{
	for (size_t i = 0u; i < exceptionIds.size(); ++i)
	{
		diagnostics_.totalCallbackExceptions += 1ULL;
		logEvent("transformer_serving_callback_exception", shmea::GLogger::LOG_WARNING, exceptionIds[i], "onToken threw; treating as cancel");
	}
	for (size_t i = 0u; i < stopIds.size(); ++i)
	{
		const uint64_t id = stopIds[i];
		std::map<uint64_t, RequestSnapshot>::iterator it = snapshots_.find(id);
		NNetworkStatus finalStatus(NNetworkStatus::OK, std::string());
		if (it != snapshots_.end())
		{
			markSnapshotStoppedByCallback_(it->second, NULL);
			finalStatus = it->second.status;
		}
		for (unsigned int s = 0u; s < live_.size(); ++s)
		{
			if (live_[s].id != id)
				continue;
			(void)TransformerPublicAPI::serving(*net_).cancelSlot(batcher_, s);
			completedCallbacks_[id] = live_[s].cb;
			clearLiveSlot_(s);
			if (cfg_.autoRemoveFinished)
				(void)TransformerPublicAPI::serving(*net_).remove(batcher_, s);
			logEvent("transformer_serving_request_done", shmea::GLogger::LOG_DEBUG, id, "done", &finalStatus, s);
			break;
		}
	}
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
		CallbackHandle cb = p.cb;

		unsigned int outSlot = 0u;
		const NNetworkStatus st = TransformerPublicAPI::serving(*net_).submit(batcher_, p.req, outSlot);
		if (!st.ok())
		{
			// Mark request failed and drop it.
			std::map<uint64_t, RequestSnapshot>::iterator sit = snapshots_.find(requestId);
			if (sit != snapshots_.end())
			{
				sit->second.done = true;
				sit->second.status = st;
			}
			diagnostics_.totalAdmitFailures += 1ULL;
			noteFailure_(requestId, outSlot, st);
			logEvent("transformer_serving_submit_fail", shmea::GLogger::LOG_WARNING, requestId, st.message.c_str(), &st, outSlot);
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

		diagnostics_.totalAdmitted += 1ULL;
		logEvent("transformer_serving_admit", shmea::GLogger::LOG_DEBUG, requestId, "admitted", NULL, outSlot);
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
		const NNetwork::TransformerGenerateResult* rr = batcher_.slotResult(s);
		if (!rr)
			rr = &snap.result;
		applyBatcherResult_(snap, *rr);
	}
}

void TransformerServingLayer::finalizeDoneSlots_()
{
	// Called after a successful Step().
	if (!net_ || !batcher_.isInitialized())
		return;

	for (unsigned int s = 0u; s < batcher_.capacity(); ++s)
	{
		if (!batcher_.slotInUse(s) || !batcher_.slotDone(s))
			continue;

		const uint64_t id = (s < live_.size()) ? live_[s].id : 0ULL;
		if (id == 0ULL)
		{
			// Defensive: unknown slot state; remove it.
			(void)TransformerPublicAPI::serving(*net_).remove(batcher_, s);
			clearLiveSlot_(s);
			continue;
		}

		// Update final snapshot.
		std::map<uint64_t, RequestSnapshot>::iterator it = snapshots_.find(id);
		NNetworkStatus finalStatus(NNetworkStatus::OK, std::string());
		if (it != snapshots_.end())
		{
			RequestSnapshot& snap = it->second;
			const NNetwork::TransformerGenerateResult* rr = batcher_.slotResult(s);
			if (!rr)
				continue;
			applyBatcherResult_(snap, *rr);
			snap.done = true;
			snap.status = NNetworkStatus(NNetworkStatus::OK, std::string());
			finalStatus = snap.status;
		}

		logEvent("transformer_serving_request_done", shmea::GLogger::LOG_DEBUG, id, "done", &finalStatus, s);
		completedCallbacks_[id] = live_[s].cb;

		clearLiveSlot_(s);

		// Remove from batcher (optional; but default enabled for serving).
		if (cfg_.autoRemoveFinished)
			(void)TransformerPublicAPI::serving(*net_).remove(batcher_, s);
	}
}

NNetworkStatus TransformerServingLayer::step()
{
	LockGuard lock(mu_);
	if (!mutexOk_())
	{
		diagnostics_.lastStepStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "TransformerServingLayer::step: recursive mutex initialization failed");
		return diagnostics_.lastStepStatus;
	}
	if (!running_ || !net_)
	{
		diagnostics_.lastStepStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "TransformerServingLayer::step: layer not running");
		return diagnostics_.lastStepStatus;
	}
	if (stopRequested_)
	{
		diagnostics_.lastStepStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "TransformerServingLayer::step: stop requested");
		return diagnostics_.lastStepStatus;
	}
	diagnostics_.totalStepCalls += 1ULL;

	// Re-entrancy guard: prevent step() from being called from within a batcher callback.
	// The recursive mutex would allow the lock, but batcher state is not re-entrant-safe.
	if (inStep_)
	{
		const NNetworkStatus st(NNetworkStatus::INVALID_STATE,
		    "TransformerServingLayer::step: re-entrant call detected (likely from a callback). "
		    "Callbacks must not call step().");
		diagnostics_.totalReentrantStepRejected += 1ULL;
		diagnostics_.lastStepStatus = st;
		noteFailure_(0ULL, static_cast<unsigned int>(-1), st);
		logEvent("transformer_serving_reentrant_step", shmea::GLogger::LOG_WARNING, 0ULL, "step() called from callback; rejected", &st);
		return st;
	}

	// 1) Admit as many pending requests as possible.
	admitPending_();

	// 2) If nothing active, nothing to do.
	if (countActiveSlots_() == 0u)
	{
		diagnostics_.totalIdleSteps += 1ULL;
		diagnostics_.lastStepStatus = NNetworkStatus(NNetworkStatus::OK, std::string());
		return diagnostics_.lastStepStatus;
	}

	// 3) Advance one global step.
	// Keep inStep_ true for the entire operation so callbacks can reject step() re-entry,
	// even though user callbacks themselves execute without the layer mutex held.
	inStep_ = true;
	const NNetwork* stepNet = net_;
	std::vector<DeferredCancelCheck> cancelChecks;
	collectDeferredCancelChecks_(cancelChecks);

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
		diagnostics_.lastStepStatus = NNetworkStatus(NNetworkStatus::OK, std::string());
		shutdownLocked_(false, NULL, "ok");
		return diagnostics_.lastStepStatus;
	}
	applyDeferredCancelDecisions_(cancelSlots, cancelLogIds);

	deferredTokenCallbacks_.clear();
	const NNetworkStatus stStep = TransformerPublicAPI::serving(*stepNet).step(batcher_, &batcherCallbacks_);

	if (!stStep.ok())
	{
		inStep_ = false;
		diagnostics_.totalStepFailures += 1ULL;
		diagnostics_.lastStepStatus = stStep;
		noteFailure_(0ULL, static_cast<unsigned int>(-1), stStep);
		markLiveRequestsFailed_(stStep);
		logEvent("transformer_serving_step_fail", shmea::GLogger::LOG_ERROR, 0ULL, stStep.message.c_str(), &stStep);
		stopRequested_ = true;
		shutdownLocked_(false, &stStep, "step_failed");
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
		CallbackHandle cb = deferred[i].cb;
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

	applyTokenCallbackStops_(stopIds, tokenExceptionIds);
	inStep_ = false;

	if (stopRequested_)
	{
		diagnostics_.lastStepStatus = NNetworkStatus(NNetworkStatus::OK, std::string());
		shutdownLocked_(false, NULL, "ok");
		return diagnostics_.lastStepStatus;
	}

	diagnostics_.lastStepStatus = NNetworkStatus(NNetworkStatus::OK, std::string());
	return diagnostics_.lastStepStatus;
}

void TransformerServingLayer::shutdownLocked_(bool clearSnapshots, const NNetworkStatus* terminalStatus, const char* logMsg)
{
	const bool hadNet = (net_ != NULL);
	if (hadNet && logMsg)
		logEvent("transformer_serving_stop", shmea::GLogger::LOG_INFO, 0ULL, logMsg, terminalStatus);
	for (std::deque<Pending>::iterator it = pending_.begin(); it != pending_.end(); ++it)
	{
		finalizeSnapshotForShutdown_(it->id, terminalStatus);
		completedCallbacks_[it->id] = it->cb;
	}
	for (size_t i = 0u; i < live_.size(); ++i)
	{
		if (live_[i].id == 0ULL)
			continue;
		finalizeSnapshotForShutdown_(live_[i].id, terminalStatus);
		completedCallbacks_[live_[i].id] = live_[i].cb;
	}
	running_ = false;
	stopRequested_ = false;
	inStep_ = false;
	pending_.clear();
	batcher_.reset();
	std::fill(slotCancel_.begin(), slotCancel_.end(), static_cast<unsigned char>(0u));
	for (size_t i = 0u; i < live_.size(); ++i)
	{
		live_[i].id = 0ULL;
		live_[i].cb = CallbackHandle();
	}
	if (clearSnapshots)
	{
		completedCallbacks_.clear();
	}
	if (clearSnapshots)
		snapshots_.clear();
	net_ = NULL;
	diagnostics_.running = false;
	diagnostics_.stopRequested = false;
	diagnostics_.inStep = false;
}

bool TransformerServingLayer::mutexOk_() const
{
	return mu_.ok();
}

} // namespace glades
