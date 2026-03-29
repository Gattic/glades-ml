#include "glades_thread_pool.h"

#include <pthread.h>
#include <unistd.h>

#include <cstdlib>
#include <xmmintrin.h>
#include <pmmintrin.h>

namespace glades {

struct ThreadPool::Impl
{
	// Work descriptor (read by workers after being set by parallel_for).
	ParallelForBody fn_;
	void* userData_;
	unsigned int count_;

	// Synchronization state.
	pthread_mutex_t mutex_;
	pthread_cond_t workCond_;  // signaled when new work is available
	pthread_cond_t doneCond_;  // signaled when all workers finish

	unsigned int generation_; // incremented per parallel_for call
	unsigned int doneCount_;  // decremented by workers; 0 means all done
	bool shutdown_;

	// Workers.
	unsigned int nThreads_;
	pthread_t* threads_;

	static void* worker_main(void* arg);
};

static unsigned int detect_num_threads()
{
	const char* env = getenv("GLADES_NUM_THREADS");
	if (env)
	{
		int n = atoi(env);
		if (n >= 1)
			return static_cast<unsigned int>(n);
	}

#ifdef _SC_NPROCESSORS_ONLN
	long n = sysconf(_SC_NPROCESSORS_ONLN);
	if (n >= 1)
		return static_cast<unsigned int>(n);
#endif
	return 1u;
}

void* ThreadPool::Impl::worker_main(void* arg)
{
	struct WorkerArg
	{
		Impl* impl;
		unsigned int id;
	};
	WorkerArg wa = *static_cast<WorkerArg*>(arg);
	delete static_cast<WorkerArg*>(arg);

	Impl& self = *wa.impl;
	const unsigned int myId = wa.id;
	unsigned int lastGen = 0u;

	// Flush subnormal floats to zero — prevents 10-100x FP slowdown on x86.
	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
	_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

	for (;;)
	{
		ParallelForBody fn;
		void* userData;
		unsigned int count;
		unsigned int nThreads;

		// Wait for work.
		pthread_mutex_lock(&self.mutex_);
		while (self.generation_ == lastGen && !self.shutdown_)
			pthread_cond_wait(&self.workCond_, &self.mutex_);

		if (self.shutdown_)
		{
			pthread_mutex_unlock(&self.mutex_);
			break;
		}

		lastGen = self.generation_;
		fn = self.fn_;
		userData = self.userData_;
		count = self.count_;
		nThreads = self.nThreads_;
		pthread_mutex_unlock(&self.mutex_);

		// Compute my chunk.
		const unsigned int begin = myId * count / nThreads;
		const unsigned int end = (myId + 1u) * count / nThreads;
		if (begin < end)
			fn(userData, begin, end);

		// Signal done.
		pthread_mutex_lock(&self.mutex_);
		self.doneCount_--;
		if (self.doneCount_ == 0u)
			pthread_cond_signal(&self.doneCond_);
		pthread_mutex_unlock(&self.mutex_);
	}

	return NULL;
}

// Pointer to the singleton Impl, used by the atexit handler.
// Stored as void* to avoid naming the private Impl type at file scope.
static void* g_poolImpl = NULL;

void ThreadPool::atexit_shutdown()
{
	Impl* impl = static_cast<Impl*>(g_poolImpl);
	if (!impl)
		return;
	g_poolImpl = NULL;

	const unsigned int nWorkers = impl->nThreads_ - 1u;

	pthread_mutex_lock(&impl->mutex_);
	impl->shutdown_ = true;
	pthread_cond_broadcast(&impl->workCond_);
	pthread_mutex_unlock(&impl->mutex_);

	for (unsigned int i = 0; i < nWorkers; ++i)
		pthread_join(impl->threads_[i], NULL);
}

ThreadPool::ThreadPool(unsigned int nThreads)
{
	impl_ = new Impl();
	impl_->fn_ = NULL;
	impl_->userData_ = NULL;
	impl_->count_ = 0u;
	impl_->generation_ = 0u;
	impl_->doneCount_ = 0u;
	impl_->shutdown_ = false;
	impl_->nThreads_ = (nThreads > 0u) ? nThreads : 1u;

	pthread_mutex_init(&impl_->mutex_, NULL);
	pthread_cond_init(&impl_->workCond_, NULL);
	pthread_cond_init(&impl_->doneCond_, NULL);

	// Spawn N-1 worker threads (thread 0 is the calling thread).
	const unsigned int nWorkers = impl_->nThreads_ - 1u;
	impl_->threads_ = NULL;
	if (nWorkers > 0u)
	{
		impl_->threads_ = new pthread_t[nWorkers];
		for (unsigned int i = 0; i < nWorkers; ++i)
		{
			struct WorkerArg
			{
				Impl* impl;
				unsigned int id;
			};
			WorkerArg* wa = new WorkerArg();
			wa->impl = impl_;
			wa->id = i + 1u; // worker ids are 1..N-1
			pthread_create(&impl_->threads_[i], NULL, Impl::worker_main, wa);
		}
	}
}

ThreadPool::~ThreadPool()
{
	const unsigned int nWorkers = impl_->nThreads_ - 1u;

	pthread_mutex_lock(&impl_->mutex_);
	impl_->shutdown_ = true;
	pthread_cond_broadcast(&impl_->workCond_);
	pthread_mutex_unlock(&impl_->mutex_);

	for (unsigned int i = 0; i < nWorkers; ++i)
		pthread_join(impl_->threads_[i], NULL);

	delete[] impl_->threads_;
	pthread_mutex_destroy(&impl_->mutex_);
	pthread_cond_destroy(&impl_->workCond_);
	pthread_cond_destroy(&impl_->doneCond_);
	delete impl_;
}

ThreadPool& ThreadPool::instance(unsigned int nThreads)
{
	// Intentionally leaked (allocated with new, never deleted).
	// Static destruction of a thread pool in a shared library is unreliable —
	// the destructor order relative to other statics is undefined.
	// Instead, the atexit handler shuts down workers cleanly.
	static ThreadPool* pool = NULL;
	if (!pool)
	{
		pool = new ThreadPool(nThreads > 0u ? nThreads : detect_num_threads());
		g_poolImpl = pool->impl_;
		atexit(atexit_shutdown);
	}
	return *pool;
}

unsigned int ThreadPool::numThreads() const
{
	return impl_->nThreads_;
}

void ThreadPool::parallel_for(unsigned int count, ParallelForBody fn, void* userData)
{
	// Fast path: no work or single-threaded.
	if (count == 0u)
		return;
	if (count == 1u || impl_->nThreads_ <= 1u)
	{
		fn(userData, 0u, count);
		return;
	}

	const unsigned int nWorkers = impl_->nThreads_ - 1u;

	// Set work descriptor and wake workers.
	pthread_mutex_lock(&impl_->mutex_);
	impl_->fn_ = fn;
	impl_->userData_ = userData;
	impl_->count_ = count;
	impl_->doneCount_ = nWorkers; // workers 1..N-1; thread 0 doesn't count
	impl_->generation_++;
	pthread_cond_broadcast(&impl_->workCond_);
	pthread_mutex_unlock(&impl_->mutex_);

	// Thread 0 (calling thread) executes chunk 0.
	{
		const unsigned int begin = 0u;
		const unsigned int end = count / impl_->nThreads_;
		if (begin < end)
			fn(userData, begin, end);
	}

	// Wait for all workers to finish.
	pthread_mutex_lock(&impl_->mutex_);
	while (impl_->doneCount_ != 0u)
		pthread_cond_wait(&impl_->doneCond_, &impl_->mutex_);
	pthread_mutex_unlock(&impl_->mutex_);
}

} // namespace glades
