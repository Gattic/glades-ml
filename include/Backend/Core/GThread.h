#ifndef _GTHREAD
#define _GTHREAD

#ifdef _WIN32
    #include "platform.h"
#else
    #include <pthread.h>
#endif

namespace shmea {

class GThread
{
public:
    typedef void* (*ThreadFunc)(void*);

private:
#ifdef _WIN32
    HANDLE handle;
#else
    pthread_t handle;
#endif
    bool started;

    GThread(const GThread&);
    GThread& operator=(const GThread&);

public:
    GThread();
    ~GThread();

    bool start(ThreadFunc func, void* arg);
    bool join();
    bool detach();
    bool isStarted() const;
};

} // namespace shmea

#endif // _GTHREAD
