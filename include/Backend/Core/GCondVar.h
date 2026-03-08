#ifndef _GCONDVAR
#define _GCONDVAR

#include "GMutex.h"

#ifdef _WIN32
    #include "platform.h"
#else
    #include <pthread.h>
    #include <time.h>
#endif

namespace shmea {

class GCondVar
{
private:
#ifdef _WIN32
    CONDITION_VARIABLE cond;
#else
    pthread_cond_t cond;
#endif

    GCondVar(const GCondVar&);
    GCondVar& operator=(const GCondVar&);

public:
    GCondVar();
    ~GCondVar();

    void wait(GMutex& m);
    bool timedWait(GMutex& m, unsigned long ms); // returns false on timeout
    void signal();
    void broadcast();
};

} // namespace shmea

#endif // _GCONDVAR
