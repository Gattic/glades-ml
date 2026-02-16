// Copyright 2026 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
// associated documentation files (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge, publish, distribute,
// sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef _GMUTEX
#define _GMUTEX

#include <pthread.h>

namespace shmea {

class GMutex
{
private:
	pthread_mutex_t mutex;

	GMutex(const GMutex&);
	GMutex& operator=(const GMutex&);

public:
	GMutex()
	{
		pthread_mutex_init(&mutex, NULL);
	}

	~GMutex()
	{
		pthread_mutex_destroy(&mutex);
	}

	void lock()
	{
		pthread_mutex_lock(&mutex);
	}

	void unlock()
	{
		pthread_mutex_unlock(&mutex);
	}
};

class GMutexLock
{
private:
	GMutex* mutex;

	GMutexLock(const GMutexLock&);
	GMutexLock& operator=(const GMutexLock&);

public:
	GMutexLock(GMutex* m) : mutex(m)
	{
		if (mutex)
			mutex->lock();
	}

	~GMutexLock()
	{
		if (mutex)
			mutex->unlock();
	}
};
};
#endif
