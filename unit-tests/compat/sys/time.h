#ifndef GLADES_UNIT_TEST_COMPAT_SYS_TIME_H
#define GLADES_UNIT_TEST_COMPAT_SYS_TIME_H

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <windows.h>

static inline int gettimeofday(struct timeval* tv, void*)
{
	if (!tv)
		return -1;
	FILETIME ft;
	ULARGE_INTEGER uli;
	GetSystemTimeAsFileTime(&ft);
	uli.LowPart = ft.dwLowDateTime;
	uli.HighPart = ft.dwHighDateTime;
	const unsigned long long usec = (uli.QuadPart - 116444736000000000ULL) / 10ULL;
	tv->tv_sec = static_cast<long>(usec / 1000000ULL);
	tv->tv_usec = static_cast<long>(usec % 1000000ULL);
	return 0;
}
#else
#include_next <sys/time.h>
#endif

#endif
