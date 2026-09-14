#ifndef GLADES_UNIT_TEST_COMPAT_UNISTD_H
#define GLADES_UNIT_TEST_COMPAT_UNISTD_H

#ifdef _WIN32
#include <process.h>
#include <sys/stat.h>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <time.h>
#include <windows.h>
#define getpid _getpid
#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif
static inline int clock_gettime(int, struct timespec* ts)
{
	if (!ts)
		return -1;
	LARGE_INTEGER freq;
	LARGE_INTEGER counter;
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&counter);
	ts->tv_sec = static_cast<time_t>(counter.QuadPart / freq.QuadPart);
	ts->tv_nsec = static_cast<long>((counter.QuadPart % freq.QuadPart) * 1000000000LL / freq.QuadPart);
	return 0;
}
#ifndef S_ISDIR
#define S_ISDIR(m) (((m) & _S_IFMT) == _S_IFDIR)
#endif
#ifndef S_ISREG
#define S_ISREG(m) (((m) & _S_IFMT) == _S_IFREG)
#endif
#else
#include_next <unistd.h>
#endif

#endif
