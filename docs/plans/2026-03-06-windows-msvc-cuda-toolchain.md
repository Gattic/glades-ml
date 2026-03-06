# Windows MSVC + CUDA Toolchain Migration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate ShmeaDB and glades-ml from Strawberry Perl (MinGW) to MSVC + vcpkg + Ninja on Windows, enabling CUDA support in glades-ml.

**Architecture:** Phase 1 updates ShmeaDB (the dependency) to build with MSVC on Windows via CMake presets and vcpkg for FreeType. Phase 2 ports glades-ml to Windows with MSVC, adding CUDA support via the NVIDIA CUDA Toolkit. Both phases replace GCC-specific flags with platform-appropriate alternatives and add CMakePresets.json for cross-platform configuration.

**Tech Stack:** CMake 3.18+, MSVC (Visual Studio Build Tools), Ninja, vcpkg, CUDA Toolkit (optional)

---

## Phase 1: ShmeaDB Toolchain Migration

### Task 1: Update ShmeaDB Root CMakeLists.txt for MSVC

**Files:**
- Modify: `C:\Users\Matt\dev\ShmeaDB\CMakeLists.txt:22-25` (compiler flags)
- Modify: `C:\Users\Matt\dev\ShmeaDB\CMakeLists.txt:85-105` (debug/mem/profile targets)

**Step 1: Fix compiler flags for MSVC vs GCC**

Currently line 22-25 appends GCC flags even on Windows:
```cmake
if(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter")
else()
    set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-variable -Wno-unused-parameter")
endif()
```

Replace with:
```cmake
if(MSVC)
    add_compile_options(/W3 /wd4101 /wd4100)
elseif(WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter")
else()
    set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-variable -Wno-unused-parameter")
endif()
```

This keeps MinGW as a fallback on Windows if someone still uses it, but adds MSVC support. MSVC equivalents:
- `/W3` = moderate warnings (like `-Wall -Wextra`)
- `/wd4101` = suppress unused variable
- `/wd4100` = suppress unused parameter

**Step 2: Fix release flags for MSVC**

Line 29: `set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g")` — these are GCC flags.

Replace with:
```cmake
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g")
endif()
```

MSVC uses `/O2` for full optimization (equivalent to `-O3`). Debug info in MSVC is controlled separately via `/Zi` and is on by default in Debug config.

**Step 3: Guard Linux-only make targets**

Lines 85-105 have `debug`, `mem`, `profile` targets wrapped in `if(NOT WIN32)` — this is already correct. But the `uninstall` target (line 108) uses `xargs rm` which doesn't work on Windows. Wrap it:

```cmake
if(NOT WIN32)
    add_custom_target(uninstall
        COMMAND xargs rm < ./build/install_manifest.txt
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
endif()
```

**Step 4: Build and verify on Windows**

Run:
```bash
cd C:\Users\Matt\dev\ShmeaDB
rm -rf build && mkdir build && cd build
cmake .. -G Ninja -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
ninja
```

Expected: Compiles without errors using MSVC `cl.exe`.

**Step 5: Commit**

```bash
git add CMakeLists.txt
git commit -m "Add MSVC compiler flag support for Windows builds"
```

---

### Task 2: Update ShmeaDB Backend CMakeLists Files for MSVC

**Files:**
- Modify: `C:\Users\Matt\dev\ShmeaDB\Backend\Networking\CMakeLists.txt:3` (FindPkgConfig)
- Review: `C:\Users\Matt\dev\ShmeaDB\Backend\Core\CMakeLists.txt` (already has WIN32 guard)
- Review: `C:\Users\Matt\dev\ShmeaDB\Backend\Database\CMakeLists.txt:8` (FindPkgConfig)
- Review: `C:\Users\Matt\dev\ShmeaDB\Backend\Plotter\CMakeLists.txt:8` (FindPkgConfig)

**Step 1: Fix FindPkgConfig usage on Windows**

`FindPkgConfig` is not available on Windows with MSVC (pkg-config is a Linux tool). It's included in:
- `Backend/Networking/CMakeLists.txt:3` — `include(FindPkgConfig)`
- `Backend/Database/CMakeLists.txt:8` — `include(FindPkgConfig)`
- `Backend/Plotter/CMakeLists.txt:8` — `include(FindPkgConfig)`

In each file, wrap with a platform guard:
```cmake
if(NOT WIN32)
    include(FindPkgConfig)
endif()
```

**Step 2: Verify Freetype detection via vcpkg**

`find_package(Freetype REQUIRED)` in Database and Plotter CMakeLists should work as-is when vcpkg toolchain is provided. Verify by building.

**Step 3: Verify Core's ws2_32 linking**

`Backend/Core/CMakeLists.txt` already has `if(WIN32) target_link_libraries(Core ws2_32) endif()`. This is correct for MSVC.

**Step 4: Build and verify**

Run:
```bash
cd C:\Users\Matt\dev\ShmeaDB\build
ninja
```

Expected: Compiles cleanly with vcpkg-provided FreeType.

**Step 5: Commit**

```bash
git add Backend/Networking/CMakeLists.txt Backend/Database/CMakeLists.txt Backend/Plotter/CMakeLists.txt
git commit -m "Guard FindPkgConfig behind platform check for MSVC compatibility"
```

---

### Task 3: Update ShmeaDB Unit Tests CMakeLists.txt for Windows

**Files:**
- Modify: `C:\Users\Matt\dev\ShmeaDB\unit-tests\CMakeLists.txt:5,14,15,83-116`

**Step 1: Fix install prefix for Windows**

Line 5: `set (CMAKE_INSTALL_PREFIX "$ENV{HOME}/.local" ...)` — `$HOME` is not set on Windows with MSVC.

Replace with:
```cmake
if(WIN32)
    set(CMAKE_INSTALL_PREFIX "$ENV{USERPROFILE}/shmea" CACHE PATH "default install path" FORCE)
else()
    set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.local" CACHE PATH "default install path" FORCE)
endif()
```

**Step 2: Fix compiler flags for MSVC**

Line 14-15:
```cmake
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-variable -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
```

Replace with:
```cmake
if(MSVC)
    add_compile_options(/W3 /wd4101 /wd4100)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2")
else()
    set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-variable -Wno-unused-parameter")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2")
endif()
```

**Step 3: Guard debug/mem/profile targets**

Lines 83-116 define `run`, `debug`, `mem`, `profile` targets and the GNUmakefile wrapper. The GNUmakefile wrapper only makes sense for Make (not Ninja). The `debug`/`mem`/`profile` targets use Linux-only tools (gdb, valgrind).

Already wrapped in `if(NOT CMAKE_GENERATOR MATCHES "Ninja")` — but gdb/valgrind targets also need `if(NOT WIN32)`. Leave `run` target available everywhere. Wrap debug/mem/profile:

```cmake
if(NOT CMAKE_GENERATOR MATCHES "Ninja")
    add_custom_target(run
        COMMAND ${PROJECT_NAME} $(ARGS)
        DEPENDS ${PROJECT_NAME}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )

    if(NOT WIN32)
        add_custom_target(debug ...)
        add_custom_target(mem ...)
        add_custom_target(profile ...)
    endif()

    # GNUmakefile wrapper ...
endif()
```

**Step 4: Build and run unit tests on Windows**

```bash
cd C:\Users\Matt\dev\ShmeaDB\unit-tests
rm -rf build && mkdir build && cd build
cmake .. -G Ninja -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" -DCMAKE_PREFIX_PATH="$USERPROFILE/shmea"
ninja
cd ..
./build/shmea-unit-tests.exe
```

Expected: All tests pass.

**Step 5: Commit**

```bash
git add unit-tests/CMakeLists.txt
git commit -m "Add MSVC and Windows support to unit test CMakeLists"
```

---

### Task 4: Add CMakePresets.json to ShmeaDB

**Files:**
- Create: `C:\Users\Matt\dev\ShmeaDB\CMakePresets.json`

**Step 1: Create CMakePresets.json**

```json
{
    "version": 6,
    "configurePresets": [
        {
            "name": "linux-release",
            "displayName": "Linux Release",
            "generator": "Unix Makefiles",
            "binaryDir": "${sourceDir}/build",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Release",
                "CMAKE_INSTALL_PREFIX": "$env{HOME}/.local"
            },
            "condition": {
                "type": "notEquals",
                "lhs": "${hostSystemName}",
                "rhs": "Windows"
            }
        },
        {
            "name": "windows-release",
            "displayName": "Windows Release (MSVC)",
            "generator": "Ninja",
            "binaryDir": "${sourceDir}/build",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Release",
                "CMAKE_INSTALL_PREFIX": "$env{USERPROFILE}/shmea",
                "CMAKE_TOOLCHAIN_FILE": "$env{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
            },
            "condition": {
                "type": "equals",
                "lhs": "${hostSystemName}",
                "rhs": "Windows"
            }
        }
    ],
    "buildPresets": [
        {
            "name": "linux-release",
            "configurePreset": "linux-release"
        },
        {
            "name": "windows-release",
            "configurePreset": "windows-release"
        }
    ]
}
```

**Step 2: Verify preset works**

```bash
cd C:\Users\Matt\dev\ShmeaDB
rm -rf build
cmake --preset windows-release
cmake --build --preset windows-release
```

Expected: Builds successfully using MSVC + Ninja + vcpkg.

**Step 3: Install and verify**

```bash
cd C:\Users\Matt\dev\ShmeaDB\build
ninja install
```

Expected: Installs to `%USERPROFILE%\shmea`.

**Step 4: Commit**

```bash
git add CMakePresets.json
git commit -m "Add CMake presets for cross-platform build configuration"
```

---

### Task 5: Update ShmeaDB INSTALL.md

**Files:**
- Modify: `C:\Users\Matt\dev\ShmeaDB\INSTALL.md`

**Step 1: Rewrite the Windows section**

Replace the existing Windows section (everything under `## Windows`) with:

```markdown
## Windows

### Prerequisites

Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) with the "Desktop development with C++" workload. This provides `cl.exe` (MSVC compiler), `cmake`, and `ninja`.

Install [vcpkg](https://github.com/microsoft/vcpkg):
```bash
git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
cd C:\vcpkg
.\bootstrap-vcpkg.bat
```

Set the `VCPKG_ROOT` environment variable to `C:\vcpkg` (or wherever you cloned it).

Install FreeType:
```bash
vcpkg install freetype
```

### Compilation

Open a **Developer Command Prompt for VS** (or run `vcvarsall.bat x64`), then:

```bash
cmake --preset windows-release
cmake --build --preset windows-release
```

### Installation

```bash
cd build
ninja install
```

Installs to `%USERPROFILE%\shmea` (e.g. `C:\Users\YourName\shmea`).

### Unit Tests

```bash
cd unit-tests
mkdir build
cd build
cmake .. -G Ninja -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake" -DCMAKE_PREFIX_PATH="%USERPROFILE%/shmea"
ninja
cd ..
.\build\shmea-unit-tests.exe
```
```

**Step 2: Verify accuracy by re-reading the full file**

Read the modified file end-to-end and confirm Linux section is unchanged.

**Step 3: Commit**

```bash
git add INSTALL.md
git commit -m "Update Windows install docs: MSVC + vcpkg replaces Strawberry Perl"
```

---

### Task 6: Verify ShmeaDB End-to-End on Windows

**Step 1: Clean build from scratch**

```bash
cd C:\Users\Matt\dev\ShmeaDB
rm -rf build
cmake --preset windows-release
cmake --build --preset windows-release
```

**Step 2: Install**

```bash
cd build && ninja install
```

**Step 3: Build and run unit tests**

```bash
cd C:\Users\Matt\dev\ShmeaDB\unit-tests
rm -rf build && mkdir build && cd build
cmake .. -G Ninja -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" -DCMAKE_PREFIX_PATH="$USERPROFILE/shmea"
ninja
cd ..
./build/shmea-unit-tests.exe
```

Expected: All tests pass.

**Step 4: Verify on Linux still works (if available)**

```bash
cmake --preset linux-release
cmake --build --preset linux-release
```

Expected: No regressions.

---

## Phase 2: glades-ml Windows Port + CUDA

### Task 7: Update glades-ml Root CMakeLists.txt for MSVC

**Files:**
- Modify: `C:\Users\Matt\dev\glades-ml\CMakeLists.txt:5,15-77,80-89,138-139,197-230`

**Step 1: Fix install prefix for Windows**

Line 5: `set (CMAKE_INSTALL_PREFIX "$ENV{HOME}/.local" ...)` — needs Windows path.

Replace with:
```cmake
if(WIN32)
    set(CMAKE_INSTALL_PREFIX "$ENV{USERPROFILE}/glades" CACHE PATH "default install path" FORCE)
else()
    set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.local" CACHE PATH "default install path" FORCE)
endif()
```

**Step 2: Update CUDA detection for Windows**

Lines 15-77 handle CUDA. Several changes needed:

a) The nvcc search paths (line 21-24) are Linux-only. Add Windows paths:
```cmake
if(NOT CMAKE_CUDA_COMPILER)
    find_program(_NVCC_BIN nvcc
        PATHS /usr/local/cuda/bin /usr/local/cuda-12/bin
              /opt/cuda/bin
              "$ENV{CUDA_PATH}/bin"
              "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin"
              "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6/bin"
        ENV CUDA_PATH
        NO_DEFAULT_PATH)
    if(_NVCC_BIN)
        set(CMAKE_CUDA_COMPILER "${_NVCC_BIN}" CACHE FILEPATH "CUDA compiler")
    endif()
endif()
```

b) The glibc compat (lines 36-37) should be skipped on Windows:
```cmake
if(NOT WIN32)
    include(${CMAKE_SOURCE_DIR}/cmake/CudaGlibcCompat.cmake)
    cuda_glibc_compat_apply()
endif()
```

c) The g++ host compiler search (lines 46-72) is Linux-only. On Windows, MSVC is the host compiler. Wrap in `if(NOT WIN32)`:
```cmake
if(NOT WIN32)
    # When the system default g++ is too new for nvcc...
    execute_process(COMMAND g++ -dumpversion ...)
    # ... existing g++ search logic ...
endif()
```

**Step 3: Fix compiler flags for MSVC**

Lines 80-89:
```cmake
if(MSVC)
    add_compile_options(/W3 /wd4101 /wd4100)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2")
elseif(WIN32)
    # MinGW fallback
else()
    set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -march=native")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g")
endif()
```

**Step 4: Fix pthread linking**

Line 138-139: `target_link_libraries(${PROJECT_NAME} ML GNet DB pthread)` — MSVC doesn't use pthread.

Replace with:
```cmake
if(WIN32)
    target_link_libraries(${PROJECT_NAME} ML GNet DB)
else()
    target_link_libraries(${PROJECT_NAME} ML GNet DB pthread)
endif()
```

**Step 5: Guard Linux-only make targets**

Lines 197-230 define `debug`, `ddd`, `mem`, `profile`, `uninstall` — all Linux-only. Wrap in `if(NOT WIN32)`:

```cmake
if(NOT WIN32)
    add_custom_target(debug ...)
    add_custom_target(ddd ...)
    add_custom_target(mem ...)
    add_custom_target(profile ...)
    add_custom_target(uninstall ...)
endif()
```

**Step 6: Build and verify**

```bash
cd C:\Users\Matt\dev\glades-ml
rm -rf build && mkdir build && cd build
cmake .. -G Ninja -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
ninja
```

Expected: Compiles with MSVC (no CUDA yet).

**Step 7: Commit**

```bash
git add CMakeLists.txt
git commit -m "Add MSVC compiler and Windows support to root CMakeLists"
```

---

### Task 8: Update glades-ml Networks CMakeLists.txt for Windows

**Files:**
- Modify: `C:\Users\Matt\dev\glades-ml\Backend\Machine Learning\Networks\CMakeLists.txt:35`

**Step 1: Fix pthread linking**

Line 35: `target_link_libraries(Networks pthread)` — MSVC doesn't have pthread.

Replace with:
```cmake
if(WIN32)
    # Windows threads are linked automatically by MSVC
else()
    target_link_libraries(Networks pthread)
endif()
```

**Step 2: Build and verify**

```bash
cd C:\Users\Matt\dev\glades-ml\build
ninja
```

Expected: Networks library compiles.

**Step 3: Commit**

```bash
git add "Backend/Machine Learning/Networks/CMakeLists.txt"
git commit -m "Guard pthread linking behind platform check in Networks"
```

---

### Task 9: Update glades-ml CUDA CMakeLists.txt for Windows

**Files:**
- Modify: `C:\Users\Matt\dev\glades-ml\Backend\Machine Learning\Networks\cuda\CMakeLists.txt:35-48`

**Step 1: Fix cuBLAS and CUDA runtime discovery for Windows**

Lines 35-48 use `find_library` with `CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES`. On Windows with MSVC, CUDA libraries are in `%CUDA_PATH%\lib\x64`. The existing code should work if CUDA Toolkit is installed (CMake sets up implicit dirs), but add a fallback:

```cmake
find_library(CUBLAS_LIB cublas
    HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
          "$ENV{CUDA_PATH}/lib/x64")
if(CUBLAS_LIB)
    target_link_libraries(GladesCUDA ${CUBLAS_LIB})
else()
    target_link_libraries(GladesCUDA cublas)
endif()

find_library(CUDART_LIB cudart
    HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
          "$ENV{CUDA_PATH}/lib/x64")
if(CUDART_LIB)
    target_link_libraries(GladesCUDA ${CUDART_LIB})
else()
    target_link_libraries(GladesCUDA cudart)
endif()
```

**Step 2: Build with CUDA on Windows**

```bash
cd C:\Users\Matt\dev\glades-ml
rm -rf build && mkdir build && cd build
cmake .. -G Ninja -DGLADES_ENABLE_CUDA=ON -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
ninja
```

Expected: Compiles with CUDA using MSVC as host compiler.

**Step 3: Commit**

```bash
git add "Backend/Machine Learning/Networks/cuda/CMakeLists.txt"
git commit -m "Add Windows CUDA library search paths for MSVC builds"
```

---

### Task 10: Update glades-ml Unit Tests CMakeLists.txt for Windows

**Files:**
- Modify: `C:\Users\Matt\dev\glades-ml\unit-tests\CMakeLists.txt:5,13-15,25,83-138`

**Step 1: Fix install prefix for Windows**

Line 5: Replace with:
```cmake
if(WIN32)
    set(CMAKE_INSTALL_PREFIX "$ENV{USERPROFILE}/glades" CACHE PATH "default install path" FORCE)
else()
    set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.local" CACHE PATH "default install path" FORCE)
endif()
```

**Step 2: Fix CUDA glibc compat for Windows**

Lines 13-19: The CudaGlibcCompat include should be skipped on Windows:
```cmake
if(GLADES_ENABLE_CUDA)
    cmake_minimum_required(VERSION 3.18)
    if(NOT WIN32)
        include(${CMAKE_SOURCE_DIR}/../cmake/CudaGlibcCompat.cmake)
        cuda_glibc_compat_apply()
    endif()
    enable_language(CUDA)
    add_definitions(-DGLADES_HAVE_CUDA=1)
endif()
```

**Step 3: Fix compiler flags for MSVC**

Line 25: `set(CMAKE_CXX_FLAGS "-Wall ... -march=native")` — GCC-only.

Replace with:
```cmake
if(MSVC)
    add_compile_options(/W3 /wd4101 /wd4100)
else()
    set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -march=native")
endif()
```

**Step 4: Guard Linux-only targets**

Wrap `debug`, `ddd`, `mem`, `profile` targets (lines 90-116) in `if(NOT WIN32)`. Keep `run` target available. Wrap the GNUmakefile wrapper generation in `if(NOT CMAKE_GENERATOR MATCHES "Ninja")` (already is).

**Step 5: Build and run unit tests on Windows**

```bash
cd C:\Users\Matt\dev\glades-ml\unit-tests
rm -rf build && mkdir build && cd build
cmake .. -G Ninja -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" -DCMAKE_PREFIX_PATH="$USERPROFILE/shmea;$USERPROFILE/glades"
ninja
cd ..
./build/glades-unit-tests.exe
```

Expected: Tests compile and pass.

**Step 6: Commit**

```bash
git add unit-tests/CMakeLists.txt
git commit -m "Add MSVC and Windows support to unit test CMakeLists"
```

---

### Task 11: Add CMakePresets.json to glades-ml

**Files:**
- Create: `C:\Users\Matt\dev\glades-ml\CMakePresets.json`

**Step 1: Create CMakePresets.json**

```json
{
    "version": 6,
    "configurePresets": [
        {
            "name": "linux-release",
            "displayName": "Linux Release",
            "generator": "Unix Makefiles",
            "binaryDir": "${sourceDir}/build",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Release",
                "CMAKE_INSTALL_PREFIX": "$env{HOME}/.local"
            },
            "condition": {
                "type": "notEquals",
                "lhs": "${hostSystemName}",
                "rhs": "Windows"
            }
        },
        {
            "name": "linux-cuda",
            "displayName": "Linux Release + CUDA",
            "inherits": "linux-release",
            "cacheVariables": {
                "GLADES_ENABLE_CUDA": "ON"
            }
        },
        {
            "name": "windows-release",
            "displayName": "Windows Release (MSVC)",
            "generator": "Ninja",
            "binaryDir": "${sourceDir}/build",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Release",
                "CMAKE_INSTALL_PREFIX": "$env{USERPROFILE}/glades",
                "CMAKE_TOOLCHAIN_FILE": "$env{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
            },
            "condition": {
                "type": "equals",
                "lhs": "${hostSystemName}",
                "rhs": "Windows"
            }
        },
        {
            "name": "windows-cuda",
            "displayName": "Windows Release + CUDA (MSVC)",
            "inherits": "windows-release",
            "cacheVariables": {
                "GLADES_ENABLE_CUDA": "ON"
            }
        }
    ],
    "buildPresets": [
        {
            "name": "linux-release",
            "configurePreset": "linux-release"
        },
        {
            "name": "linux-cuda",
            "configurePreset": "linux-cuda"
        },
        {
            "name": "windows-release",
            "configurePreset": "windows-release"
        },
        {
            "name": "windows-cuda",
            "configurePreset": "windows-cuda"
        }
    ]
}
```

**Step 2: Verify preset works**

```bash
cd C:\Users\Matt\dev\glades-ml
rm -rf build
cmake --preset windows-release
cmake --build --preset windows-release
```

Expected: Builds with MSVC.

**Step 3: Verify CUDA preset**

```bash
rm -rf build
cmake --preset windows-cuda
cmake --build --preset windows-cuda
```

Expected: Builds with MSVC + CUDA.

**Step 4: Commit**

```bash
git add CMakePresets.json
git commit -m "Add CMake presets for cross-platform and CUDA build configuration"
```

---

### Task 12: Update glades-ml CLAUDE.md with Windows Install Section

**Files:**
- Modify: `C:\Users\Matt\dev\glades-ml\CLAUDE.md`

**Step 1: Add Windows installation section**

After the existing `## Build Commands` section, add a new `## Windows Installation` section:

```markdown
## Windows Installation

### Prerequisites

Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022) with the "Desktop development with C++" workload. This provides `cl.exe` (MSVC compiler), `cmake`, and `ninja`.

Install [vcpkg](https://github.com/microsoft/vcpkg):
```bash
git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
cd C:\vcpkg
.\bootstrap-vcpkg.bat
```

Set the `VCPKG_ROOT` environment variable to `C:\vcpkg` (or wherever you cloned it).

Install FreeType (needed by shmea):
```bash
vcpkg install freetype
```

**Optional:** For CUDA GPU support, install the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads). Set `CUDA_PATH` environment variable if not auto-detected.

### Build shmea first

shmea must be built and installed before glades-ml. From a Developer Command Prompt:

```bash
cd path\to\ShmeaDB
cmake --preset windows-release
cmake --build --preset windows-release
cd build && ninja install
```

### Build glades-ml

```bash
cmake --preset windows-release
cmake --build --preset windows-release
```

Or with CUDA:
```bash
cmake --preset windows-cuda
cmake --build --preset windows-cuda
```

### Install

```bash
cd build && ninja install
```

### Unit Tests

```bash
cd unit-tests
mkdir build && cd build
cmake .. -G Ninja -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake" -DCMAKE_PREFIX_PATH="%USERPROFILE%/shmea;%USERPROFILE%/glades"
ninja
cd ..
.\build\glades-unit-tests.exe
```
```

**Step 2: Update existing Build Commands to clarify they're for Linux**

Change `## Build Commands` to `## Build Commands (Linux)`.

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "Add Windows installation section to CLAUDE.md"
```

---

### Task 13: Fix Any MSVC Compilation Errors in glades-ml Source

**Files:**
- Various source files under `Backend/Machine Learning/` (identified during build)

This task is iterative. Common GCC-isms that MSVC will reject:

1. **Variable-length arrays (VLAs)** — `int arr[n]` where n is not const. Replace with `std::vector<int> arr(n)` or stack allocation via `_alloca` on Windows.
2. **`__attribute__((unused))`** — Replace with `(void)var;` or remove.
3. **`__builtin_expect`** — Wrap in a macro that's a no-op on MSVC.
4. **`typeof` / `__typeof__`** — Not available in MSVC C++98. Use explicit types.
5. **Named struct initializers** — `.field = value` syntax in C++ (C99 but not C++98). May work in MSVC depending on version.

**Step 1: Attempt build, collect errors**

```bash
cd C:\Users\Matt\dev\glades-ml\build
ninja 2>&1 | head -100
```

**Step 2: Fix each error**

Address errors one file at a time. Keep changes minimal — only fix what MSVC rejects.

**Step 3: Rebuild and iterate**

Repeat until `ninja` completes successfully.

**Step 4: Commit**

```bash
git add -A
git commit -m "Fix GCC-specific code for MSVC compatibility"
```

---

### Task 14: End-to-End Verification

**Step 1: Clean build ShmeaDB**

```bash
cd C:\Users\Matt\dev\ShmeaDB
rm -rf build
cmake --preset windows-release
cmake --build --preset windows-release
cd build && ninja install
```

**Step 2: Clean build glades-ml (without CUDA)**

```bash
cd C:\Users\Matt\dev\glades-ml
rm -rf build
cmake --preset windows-release
cmake --build --preset windows-release
cd build && ninja install
```

**Step 3: Clean build glades-ml (with CUDA)**

```bash
cd C:\Users\Matt\dev\glades-ml
rm -rf build
cmake --preset windows-cuda
cmake --build --preset windows-cuda
cd build && ninja install
```

**Step 4: Run unit tests**

```bash
cd C:\Users\Matt\dev\glades-ml\unit-tests
rm -rf build && mkdir build && cd build
cmake .. -G Ninja -DCMAKE_TOOLCHAIN_FILE="$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" -DCMAKE_PREFIX_PATH="$USERPROFILE/shmea;$USERPROFILE/glades"
ninja
cd ..
./build/glades-unit-tests.exe
```

Expected: All tests pass.

**Step 5: Commit any remaining fixes**

```bash
git add -A
git commit -m "Final Windows build fixes after end-to-end verification"
```
