#ifndef GLADES_UNIT_TEST_COMPAT_PTHREAD_H
#define GLADES_UNIT_TEST_COMPAT_PTHREAD_H

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <process.h>
#include <windows.h>

typedef HANDLE pthread_t;

typedef struct pthread_mutex_t
{
	CRITICAL_SECTION cs;
} pthread_mutex_t;

typedef struct pthread_cond_t
{
	CONDITION_VARIABLE cv;
} pthread_cond_t;

struct glades_pthread_start_data
{
	void* (*start)(void*);
	void* arg;
};

static unsigned __stdcall glades_pthread_start(void* arg)
{
	glades_pthread_start_data* data = static_cast<glades_pthread_start_data*>(arg);
	void* (*start)(void*) = data->start;
	void* startArg = data->arg;
	delete data;
	(void)start(startArg);
	return 0u;
}

static inline int pthread_mutex_init(pthread_mutex_t* mutex, const void*)
{
	InitializeCriticalSection(&mutex->cs);
	return 0;
}

static inline int pthread_mutex_destroy(pthread_mutex_t* mutex)
{
	DeleteCriticalSection(&mutex->cs);
	return 0;
}

static inline int pthread_mutex_lock(pthread_mutex_t* mutex)
{
	EnterCriticalSection(&mutex->cs);
	return 0;
}

static inline int pthread_mutex_unlock(pthread_mutex_t* mutex)
{
	LeaveCriticalSection(&mutex->cs);
	return 0;
}

static inline int pthread_cond_init(pthread_cond_t* cond, const void*)
{
	InitializeConditionVariable(&cond->cv);
	return 0;
}

static inline int pthread_cond_destroy(pthread_cond_t*)
{
	return 0;
}

static inline int pthread_cond_broadcast(pthread_cond_t* cond)
{
	WakeAllConditionVariable(&cond->cv);
	return 0;
}

static inline int pthread_cond_wait(pthread_cond_t* cond, pthread_mutex_t* mutex)
{
	return SleepConditionVariableCS(&cond->cv, &mutex->cs, INFINITE) ? 0 : -1;
}

static inline int pthread_create(pthread_t* thread, const void*, void* (*start)(void*), void* arg)
{
	glades_pthread_start_data* data = new glades_pthread_start_data;
	data->start = start;
	data->arg = arg;
	uintptr_t handle = _beginthreadex(NULL, 0, glades_pthread_start, data, 0, NULL);
	if (handle == 0)
	{
		delete data;
		return -1;
	}
	*thread = reinterpret_cast<HANDLE>(handle);
	return 0;
}

static inline int pthread_join(pthread_t thread, void**)
{
	WaitForSingleObject(thread, INFINITE);
	CloseHandle(thread);
	return 0;
}
#else
#include_next <pthread.h>
#endif

#endif
