#ifndef GLADES_THREAD_POOL_H
#define GLADES_THREAD_POOL_H

namespace glades {

typedef void (*ParallelForBody)(void* userData, unsigned int begin, unsigned int end);

class ThreadPool
{
public:
	// Get the singleton instance. nThreads=0 means auto-detect (number of online CPUs).
	static ThreadPool& instance(unsigned int nThreads = 0);

	// Divide [0, count) into chunks across threads and execute fn(userData, begin, end) per chunk.
	// Calling thread participates as chunk 0 to avoid one context switch.
	// Fast path: if count <= 1 or nThreads == 1, executes synchronously.
	void parallel_for(unsigned int count, ParallelForBody fn, void* userData);

	unsigned int numThreads() const;

	~ThreadPool();

private:
	ThreadPool(unsigned int nThreads);
	ThreadPool(const ThreadPool&);
	ThreadPool& operator=(const ThreadPool&);

	static void atexit_shutdown();

	struct Impl;
	Impl* impl_;
};

} // namespace glades

#endif // GLADES_THREAD_POOL_H
