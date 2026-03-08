#ifndef _GDIR
#define _GDIR

#ifdef _WIN32
    #include "platform.h"
#else
    #include <dirent.h>
    #include <sys/stat.h>
    #include <sys/types.h>
#endif

#include <string>

namespace shmea {

struct GDirEntry
{
    std::string name;  // filename only, no path
    bool isDirectory;

    GDirEntry() : isDirectory(false) {}
};

class GDir
{
private:
#ifdef _WIN32
    HANDLE           hFind;
    WIN32_FIND_DATAA findData;
    bool             firstDone;
#else
    DIR*             dir;
    struct dirent*   entry;
#endif
    bool valid;

    GDir(const GDir&);
    GDir& operator=(const GDir&);

public:
    GDir();
    ~GDir();

    bool open(const std::string& path);
    bool next(GDirEntry& out);   // returns false when exhausted; skips . and ..
    void close();
    bool isOpen() const;

    static bool exists(const std::string& path);
    static bool makeDir(const std::string& path);
};

} // namespace shmea

#endif // _GDIR
