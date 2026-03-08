#ifndef _GPLATFORM
#define _GPLATFORM

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #include <windows.h>

    typedef SOCKET socket_t;
    /* ssize_t: MinGW provides it via corecrt.h; MSVC does not have it at all */
    #ifdef _MSC_VER
        #include <BaseTsd.h>
        typedef SSIZE_T ssize_t;
    #endif
    static const socket_t INVALID_SOCKET_VALUE = INVALID_SOCKET;

    #define G_CLOSE_SOCKET(fd)  closesocket(fd)
    #define G_EWOULDBLOCK       WSAEWOULDBLOCK
    #define G_EINTR             WSAEINTR
    #define G_EINPROGRESS       WSAEINPROGRESS
    #define G_LAST_SOCK_ERROR() WSAGetLastError()
    #define G_MSG_NOSIGNAL      0
    #define G_MSG_DONTWAIT      0
    #define G_SETSOCKOPT_VAL(v) ((const char*)&(v))
    #define G_GETSOCKOPT_VAL(v) ((char*)&(v))

    static inline int g_set_nonblocking(socket_t fd, int nonblocking)
    {
        u_long mode = (u_long)nonblocking;
        return (ioctlsocket(fd, FIONBIO, &mode) == 0) ? 0 : -1;
    }

    /* string.h needed for memset used in g_socketpair */
    #include <string.h>

    static inline int g_socketpair(socket_t fds[2])
    {
        /* Windows has no socketpair(); emulate with two connected TCP loopback sockets */
        socket_t listener;
        struct sockaddr_in addr;
        int addrlen;
        listener = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (listener == INVALID_SOCKET) return -1;
        memset(&addr, 0, sizeof(addr));
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = 0;
        if (bind(listener, (struct sockaddr*)&addr, sizeof(addr)) != 0)
            { closesocket(listener); return -1; }
        addrlen = (int)sizeof(addr);
        if (getsockname(listener, (struct sockaddr*)&addr, &addrlen) != 0)
            { closesocket(listener); return -1; }
        if (listen(listener, 1) != 0)
            { closesocket(listener); return -1; }
        fds[0] = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (fds[0] == INVALID_SOCKET) { closesocket(listener); return -1; }
        if (connect(fds[0], (struct sockaddr*)&addr, sizeof(addr)) != 0)
            { closesocket(fds[0]); closesocket(listener); return -1; }
        fds[1] = accept(listener, NULL, NULL);
        closesocket(listener);
        if (fds[1] == INVALID_SOCKET) { closesocket(fds[0]); return -1; }
        return 0;
    }

    static inline void g_sleep_ms(int ms) { if (ms > 0) Sleep((DWORD)ms); }

    /* mkdir on Windows takes only one argument (no mode bits) */
    #include <direct.h>
    #define mkdir(path, mode) _mkdir(path)

    /* bzero is not available on Windows; emulate with memset */
    static inline void bzero(void* s, size_t n) { memset(s, 0, n); }

    /* M_PI may not be defined unless _USE_MATH_DEFINES is set */
    #ifndef M_PI
        #define M_PI 3.14159265358979323846
    #endif

    /* gettimeofday: MinGW already provides struct timeval and struct timezone via time.h */
    /* No additional struct definitions needed here */

#else
    #include <arpa/inet.h>
    #include <netinet/tcp.h>
    #include <netdb.h>
    #include <sys/select.h>
    #include <sys/socket.h>
    #include <sys/types.h>
    #include <sys/signal.h>
    #include <fcntl.h>
    #include <unistd.h>
    #include <errno.h>

    typedef int socket_t;
    static const socket_t INVALID_SOCKET_VALUE = -1;

    #define G_CLOSE_SOCKET(fd)  close(fd)
    #define G_EWOULDBLOCK       EWOULDBLOCK
    #define G_EINTR             EINTR
    #define G_EINPROGRESS       EINPROGRESS
    #define G_LAST_SOCK_ERROR() errno
    #define G_MSG_NOSIGNAL      MSG_NOSIGNAL
    #define G_MSG_DONTWAIT      MSG_DONTWAIT
    #define G_SETSOCKOPT_VAL(v) (&(v))
    #define G_GETSOCKOPT_VAL(v) (&(v))

    static inline int g_set_nonblocking(socket_t fd, int nonblocking)
    {
        int flags = fcntl(fd, F_GETFL, 0);
        if (flags < 0) return -1;
        if (nonblocking) flags |= O_NONBLOCK;
        else             flags &= ~O_NONBLOCK;
        return (fcntl(fd, F_SETFL, flags) == 0) ? 0 : -1;
    }

    static inline int g_socketpair(socket_t fds[2])
    {
        return socketpair(AF_UNIX, SOCK_STREAM, 0, fds);
    }

    static inline void g_sleep_ms(int ms)
    {
        if (ms > 0) usleep((useconds_t)ms * 1000);
    }
#endif

#endif // _GPLATFORM
