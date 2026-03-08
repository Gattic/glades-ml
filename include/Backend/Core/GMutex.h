#ifndef _GMUTEX
#define _GMUTEX

#ifdef _WIN32
    #include "platform.h"
#else
    #include <pthread.h>
#endif

namespace shmea {

class GMutex
{
private:
#ifdef _WIN32
    CRITICAL_SECTION mutex;
#else
    pthread_mutex_t mutex;
#endif

    GMutex(const GMutex&);
    GMutex& operator=(const GMutex&);

public:
    GMutex();
    ~GMutex();
    void lock();
    void unlock();

#ifdef _WIN32
    CRITICAL_SECTION* nativeHandle() { return &mutex; }
#else
    pthread_mutex_t* nativeHandle() { return &mutex; }
#endif
};

class GMutexLock
{
private:
    GMutex* mutex;
    GMutexLock(const GMutexLock&);
    GMutexLock& operator=(const GMutexLock&);

public:
    GMutexLock(GMutex* m) : mutex(m) { if (mutex) mutex->lock(); }
    ~GMutexLock()                    { if (mutex) mutex->unlock(); }
};

} // namespace shmea

#endif // _GMUTEX
